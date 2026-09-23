# Nano-vLLM 二次开发：推理优化

基于 [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 的轻量级推理引擎扩展，面向多请求生成场景加入混合连续批处理、长序列分块预填充和 CUDA KV cache 路径，在有限显存下平衡吞吐、解码间隔与首 token 延迟。

## 功能概览

- **Mixed continuous batching**：在同一调度周期中组合 decode 和 prefill 请求。
- **Chunked prefill**：按 token budget 分段处理长 prompt，避免单个请求占满一步计算。
- **Prefix KV cache**：复用相同前缀的完整 KV block，减少重复计算。
- **Selective LM head / sampling**：未完成的 prefill chunk 跳过词表投影和采样。
- **CUDA KV cache write**：提供按 `slot_mapping` 写入 paged KV cache 的 CUDA 路径。

请求经过 waiting/running 队列后，由 scheduler 生成 decode、prefill 或 mixed batch；模型执行完成后更新 KV cache、采样结果和请求状态。

## 关键改动

| 方向 | 作用 |
|---|---|
| Decode-first mixed 调度 | 优先推进已有请求，降低多请求场景的 TPOT 和 E2E 延迟 |
| Chunked prefill 与 token budget | 限制长 prompt 的单步计算量，保留后续请求的调度机会 |
| Prefix cache | 命中共享前缀时直接复用已计算的 KV block |
| 中间 chunk 跳过采样 | 避免对未完成 prompt 反复执行大词表 LM head |
| CUDA KV 写入 | 减少 paged KV cache 写入路径的额外开销 |

## 性能概览

测试配置：NVIDIA GeForce RTX 4060 Ti 16GB，单 GPU，Qwen3-0.6B，PyTorch 2.13.0+cu130。Baseline 为 prefill-first 的 Nano-vLLM 源版本；数值为相同输入与生成配置下的代表性平均结果，负延迟变化和正吞吐变化表示改善。

| Workload | Runtime | Throughput | TTFT mean | TPOT mean | E2E mean |
|---|---:|---:|---:|---:|---:|
| single-batch | 71.734 → 70.756 s (-1.36%) | 1867.55 → 1893.43 (+1.39%) | -1.01% | -1.37% | -1.51% |
| single-long | 1.133 → 1.160 s (+2.33%) | 113.19 → 110.58 (-2.30%) | +9.93% | +0.60% | +2.32% |
| multi-constant | 3.201 → 3.112 s (-2.79%) | 964.13 → 991.80 (+2.87%) | +50.77% | -7.95% | -5.46% |
| multi-poisson | 3.029 → 3.034 s (+0.17%) | 1018.89 → 1017.34 (-0.15%) | +57.64% | -3.11% | +0.54% |
| multi-wave | 7.874 → 7.587 s (-3.65%) | 391.51 → 406.24 (+3.76%) | +40.95% | -23.83% | -9.97% |

Mixed 调度更适合并发请求：`multi-constant` 和 `multi-wave` 的解码间隔、端到端延迟与吞吐得到改善。由于 decode 优先，新请求的 TTFT 可能增加；单条长请求没有 mixed batch，结果主要反映调度和执行路径本身的开销。

## 快速开始

环境需要 Linux、NVIDIA GPU、Python 3.10–3.12，以及兼容的 PyTorch、CUDA Toolkit 和 FlashAttention。自定义 CUDA 扩展会在首次导入时编译。

```bash
python -m pip install -e .
python example.py --model /path/to/Qwen3-0.6B --mode multi --no-enforce-eager
```

运行批处理测试：

```bash
python bench.py --model /path/to/Qwen3-0.6B \
  --workload-mode single-batch --num-seqs 256 \
  --min-input-len 100 --max-input-len 1024 \
  --min-output-len 100 --max-output-len 1024 \
  --output single-batch.json
```

## 主要文件

| 文件 | 内容 |
|---|---|
| [scheduler.py](nanovllm/engine/scheduler.py) | mixed 调度、chunked prefill 与 token budget |
| [model_runner.py](nanovllm/engine/model_runner.py) | mixed batch 的输入和 attention 元数据 |
| [attention.py](nanovllm/layers/attention.py) | prefill/decode attention 与 KV cache 写入 |
| [embed_head.py](nanovllm/layers/embed_head.py) | 选择性 LM head 与采样入口 |
| [bench.py](bench.py) | 可重复的 workload 和延迟统计 |

## 验证范围

调度器提供 CPU 测试，GPU 路径覆盖 mixed forward、chunked prefill、prefix cache、纯 decode CUDA Graph 和 KV cache 写入。性能结果与硬件、模型、PyTorch/CUDA 版本及请求分布有关，不能直接外推到其他平台。

## 致谢与许可证

本项目基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)，调度、执行路径、CUDA KV 写入和测试工具为此仓库的扩展。项目参考 vLLM 的调度思路，不代表 vLLM 官方实现。

遵循 [MIT License](LICENSE)，保留上游作者 Xingkai Yu 的版权与许可声明。
