import torch
from torch import nn
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context

import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r'''
#include <cuda_bf16.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

__global__ void store_kvcache_kernel_float4(
    const float* __restrict__ key,
    int key_stride0,
    const float* __restrict__ value,
    int value_stride0,
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const int* __restrict__ slot_mapping,
    int D
) {
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int slot = slot_mapping[idx];
    if (slot < 0) return;
    int D4 = D / 4;
    const float* key_ptr   = key + idx * key_stride0;
    const float* value_ptr = value + idx * value_stride0;
    float* k_cache_ptr = k_cache + slot * D;
    float* v_cache_ptr = v_cache + slot * D;
    const float4* key4 = reinterpret_cast<const float4*>(key_ptr);
    const float4* value4 = reinterpret_cast<const float4*>(value_ptr);
    float4* k_cache4 = reinterpret_cast<float4*>(k_cache_ptr);
    float4* v_cache4 = reinterpret_cast<float4*>(v_cache_ptr);
    for (int t = tid; t < D4; t += blockDim.x) {
        float4 k = key4[t];
        float4 v = value4[t];
        k_cache4[t] = k;
        v_cache4[t] = v;
    }
}

struct __align__(8) Bf16Vec4 {
    __nv_bfloat16 x, y, z, w;
};

__global__ void store_kvcache_kernel_bf16(
    const __nv_bfloat16* __restrict__ key,
    int key_stride0,
    const __nv_bfloat16* __restrict__ value,
    int value_stride0,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ slot_mapping,
    int D
) {
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int slot = slot_mapping[idx];
    if (slot < 0) return;

    int D4 = D / 4;
    const __nv_bfloat16* key_ptr   = key + idx * key_stride0;
    const __nv_bfloat16* value_ptr = value + idx * value_stride0;
    __nv_bfloat16* k_cache_ptr = k_cache + slot * D;
    __nv_bfloat16* v_cache_ptr = v_cache + slot * D;

    const Bf16Vec4* key4 = reinterpret_cast<const Bf16Vec4*>(key_ptr);
    const Bf16Vec4* value4 = reinterpret_cast<const Bf16Vec4*>(value_ptr);
    Bf16Vec4* k_cache4 = reinterpret_cast<Bf16Vec4*>(k_cache_ptr);
    Bf16Vec4* v_cache4 = reinterpret_cast<Bf16Vec4*>(v_cache_ptr);

    for (int t = tid; t < D4; t += blockDim.x) {
        Bf16Vec4 k = key4[t];
        Bf16Vec4 v = value4[t];
        k_cache4[t] = k;
        v_cache4[t] = v;
    }
}

torch::Tensor store_kvcache_cuda(
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor slot_mapping
) {
    int N = slot_mapping.numel();
    int D = key.size(1) * key.size(2);
    const c10::cuda::CUDAGuard device_guard(key.device());
    auto stream = c10::cuda::getCurrentCUDAStream();
    TORCH_CHECK(D % 4 == 0, "KV vector copy requires D divisible by 4");
    TORCH_CHECK(key.scalar_type() == value.scalar_type() &&
                key.scalar_type() == k_cache.scalar_type() &&
                key.scalar_type() == v_cache.scalar_type(), "KV dtypes must match");
    if (N == 0) return k_cache;
    int threads;
    int D4 = D / 4;
    if (D4 <= 32) threads = 32;
    else if (D4 <= 64) threads = 64;
    else if (D4 <= 128) threads = 128;
    else threads = 256;

    if (key.scalar_type() == torch::kFloat32) {
        store_kvcache_kernel_float4<<<N, threads, 0, stream>>>(
            key.data_ptr<float>(),
            key.stride(0),
            value.data_ptr<float>(),
            value.stride(0),
            k_cache.data_ptr<float>(),
            v_cache.data_ptr<float>(),
            slot_mapping.data_ptr<int>(),
            D
        );
    } else if (key.scalar_type() == torch::kBFloat16) {
        store_kvcache_kernel_bf16<<<N, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(key.data_ptr()),
            key.stride(0),
            reinterpret_cast<const __nv_bfloat16*>(value.data_ptr()),
            value.stride(0),
            reinterpret_cast<__nv_bfloat16*>(k_cache.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(v_cache.data_ptr()),
            slot_mapping.data_ptr<int>(),
            D
        );
    } else {
        TORCH_CHECK(false, "Only support float32 and bfloat16");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return k_cache;
}
'''
cpp_source = r'''
#include <torch/torch.h>
torch::Tensor store_kvcache_cuda(
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor slot_mapping
);
'''
_module = load_inline(
    name="store_kvcache_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["store_kvcache_cuda"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)

def store_kvcache_cuda(key: torch.Tensor, value: torch.Tensor,
                       k_cache: torch.Tensor, v_cache: torch.Tensor,
                       slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert slot_mapping.numel() == N
    _module.store_kvcache_cuda(key, value, k_cache, v_cache, slot_mapping)


class Attention(nn.Module):
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache_cuda(k, v, k_cache, v_cache, context.slot_mapping)
        if context.num_decode:
            n = context.num_decode
            decode = flash_attn_with_kvcache(
                q[:n].unsqueeze(1), k_cache, v_cache,
                cache_seqlens=context.context_lens, block_table=context.block_tables[:n],
                softmax_scale=self.scale, causal=True).squeeze(1)
            prefill = flash_attn_varlen_func(
                q[n:], k_cache, v_cache,
                max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.prefill_cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.prefill_cu_seqlens_k,
                softmax_scale=self.scale, causal=True, block_table=context.block_tables[n:])
            o = torch.cat((decode, prefill), dim=0)
        elif context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                      max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                      max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                      softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                       cache_seqlens=context.context_lens, block_table=context.block_tables,
                                       softmax_scale=self.scale, causal=True)
        return o
