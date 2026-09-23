"""CPU scheduler tests; add --gpu --model PATH for real mixed-forward checks."""
import argparse
import atexit
import importlib
import pickle
from pathlib import Path
import random
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def cpu_tests():
    package = ModuleType("nanovllm")
    package.__path__ = [str(ROOT / "nanovllm")]
    config = ModuleType("nanovllm.config")
    config.Config = SimpleNamespace
    with patch.dict(sys.modules, {"nanovllm": package, "nanovllm.config": config}):
        Scheduler = importlib.import_module("nanovllm.engine.scheduler").Scheduler
        Sequence = importlib.import_module("nanovllm.engine.sequence").Sequence
        SamplingParams = importlib.import_module("nanovllm.sampling_params").SamplingParams

        class Tests(unittest.TestCase):
            def scheduler(self, blocks=64, budget=512, slots=16):
                Sequence.block_size = 256
                return Scheduler(SimpleNamespace(
                    max_num_seqs=slots, max_num_batched_tokens=budget, eos=-1,
                    kvcache_block_size=256, num_kvcache_blocks=blocks))

            def request(self, length, output=4, offset=0):
                return Sequence(list(range(offset, offset + length)),
                                SamplingParams(max_tokens=output, ignore_eos=True))

            def step(self, scheduler):
                seqs, prefill = scheduler.schedule()
                self.assertEqual(len(seqs), len(set(seqs)))
                self.assertLessEqual(len(seqs), scheduler.max_num_seqs)
                self.assertLessEqual(sum(s.num_scheduled_tokens for s in seqs),
                                     scheduler.max_num_batched_tokens)
                for s in seqs:
                    self.assertGreater(s.num_scheduled_tokens, 0)
                    self.assertLessEqual(s.num_cached_tokens + s.num_scheduled_tokens, len(s))
                token_ids = [
                    None if prefill and s.num_cached_tokens + s.num_scheduled_tokens < len(s)
                    else 50000 + s.seq_id
                    for s in seqs
                ]
                scheduler.postprocess(seqs, token_ids, prefill)
                return seqs, prefill

            def test_mixed_budget_serialization_and_completion(self):
                scheduler = self.scheduler(budget=256)
                active = self.request(100)
                scheduler.add(active)
                self.step(scheduler)
                incoming = self.request(600, offset=1000)
                scheduler.add(incoming)
                seqs, prefill = scheduler.schedule()
                self.assertTrue(prefill)
                self.assertEqual(seqs, [active, incoming])
                self.assertEqual([s.num_scheduled_tokens for s in seqs], [1, 255])
                copied = pickle.loads(pickle.dumps(seqs))
                self.assertFalse(copied[0].is_prefill)
                self.assertEqual(copied[0].token_ids, [])
                self.assertEqual(copied[0].last_token, active.last_token)
                self.assertTrue(copied[1].is_prefill)
                self.assertEqual(copied[1].token_ids, incoming.token_ids)
                scheduler.postprocess(seqs, [9, None], prefill)
                self.assertEqual(active.num_completion_tokens, 2)
                self.assertEqual(incoming.num_completion_tokens, 0)
                for _ in range(20):
                    if scheduler.is_finished(): break
                    self.step(scheduler)
                self.assertTrue(scheduler.is_finished())

            def test_single_long_four_chunks(self):
                scheduler = self.scheduler(budget=1024)
                seq = self.request(3968)
                scheduler.add(seq)
                for step in range(4):
                    self.assertTrue(self.step(scheduler)[1])
                    self.assertEqual(seq.num_completion_tokens, int(step == 3))
                self.assertFalse(self.step(scheduler)[1])

            def test_full_decode_budget_leaves_prompt_waiting(self):
                scheduler = self.scheduler(budget=1)
                active = self.request(1)
                scheduler.add(active)
                self.step(scheduler)
                waiting = self.request(10)
                scheduler.add(waiting)
                self.assertEqual(self.step(scheduler), ([active], False))
                self.assertEqual(waiting.num_cached_tokens, 0)

            def test_prefill_uses_batch_tail(self):
                scheduler = self.scheduler(budget=512)
                a, b = self.request(400), self.request(400, offset=1000)
                scheduler.add(a); scheduler.add(b)
                seqs, mode = scheduler.schedule()
                self.assertTrue(mode)
                self.assertEqual([s.num_scheduled_tokens for s in seqs], [400, 112])
                scheduler.postprocess(seqs, [9, None], mode)
                self.assertEqual(b.num_completion_tokens, 0)

            def test_resident_chunk_behind_blocked_head(self):
                scheduler = self.scheduler(blocks=2, budget=256)
                partial = self.request(512)
                scheduler.add(partial)
                self.step(scheduler)
                scheduler.waiting.appendleft(self.request(512, offset=1000))
                self.assertEqual(self.step(scheduler)[0], [partial])

            def test_prefix_reuse(self):
                scheduler = self.scheduler(budget=1024)
                scheduler.add(self.request(768, output=1))
                self.step(scheduler)
                duplicate = self.request(768)
                scheduler.add(duplicate)
                scheduler.schedule()
                self.assertEqual(duplicate.num_cached_tokens, 512)
                self.assertEqual(duplicate.num_scheduled_tokens, 256)

            def test_impossible_request_fails_clearly(self):
                scheduler = self.scheduler(blocks=1)
                scheduler.add(self.request(512))
                with self.assertRaisesRegex(RuntimeError, "KV cache"):
                    scheduler.schedule()

            def test_kv_pressure(self):
                rng = random.Random(0)
                prompts = [[rng.randint(0, 10000) for _ in range(rng.randint(100, 1024))]
                           for _ in range(256)]
                limits = [rng.randint(100, 1024) for _ in prompts]
                scheduler = self.scheduler(blocks=128, budget=1024, slots=512)
                seqs = [Sequence(p, SamplingParams(max_tokens=n, ignore_eos=True))
                        for p, n in zip(prompts, limits)]
                for seq in seqs: scheduler.add(seq)
                for _ in range(20000):
                    if scheduler.is_finished(): break
                    self.step(scheduler)
                self.assertTrue(scheduler.is_finished())
                self.assertEqual([s.num_completion_tokens for s in seqs], limits)
                self.assertEqual(len(scheduler.block_manager.free_block_ids), 128)

        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
        if not result.wasSuccessful(): raise SystemExit(1)


def gpu_tests(model):
    import torch
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.sequence import Sequence
    from nanovllm.layers.attention import store_kvcache_cuda
    from nanovllm.utils.context import get_context, reset_context

    # Noncontiguous QKV rows, a skipped padding slot, current stream and replay.
    with torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream()):
        for dtype in (torch.float32, torch.bfloat16):
            source = torch.randn(3, 3, 2, 128, dtype=dtype, device="cuda")
            key, value = source[:, 1], source[:, 2]
            kcache = torch.zeros(1, 256, 2, 128, dtype=dtype, device="cuda")
            vcache = torch.zeros_like(kcache)
            slots = torch.tensor([3, -1, 9], dtype=torch.int32, device="cuda")
            store_kvcache_cuda(key, value, kcache, vcache, slots)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                store_kvcache_cuda(key, value, kcache, vcache, slots)
            source.normal_()
            kcache.zero_(); vcache.zero_()
            graph.replay()
            expected_k = torch.zeros_like(kcache)
            expected_v = torch.zeros_like(vcache)
            expected_k[0, [3, 9]] = key[[0, 2]]
            expected_v[0, [3, 9]] = value[[0, 2]]
            torch.testing.assert_close(kcache, expected_k, rtol=0, atol=0)
            torch.testing.assert_close(vcache, expected_v, rtol=0, atol=0)
    torch.cuda.synchronize()
    print("KV copy PASS: FP32/BF16, strided inputs, padding, nondefault stream, CUDA Graph replay")

    with torch.inference_mode():
        llm = LLM(model, max_num_batched_tokens=512, max_num_seqs=8,
                  max_model_len=2048, gpu_memory_utilization=0.7, enforce_eager=False)
        runner = llm.model_runner
        params = SamplingParams(max_tokens=6, ignore_eos=True)
        rng = random.Random(0)
        short = [rng.randint(100, 5000) for _ in range(128)]
        long = [rng.randint(100, 5000) for _ in range(700)]

        # A non-final chunk must stop before the vocabulary projection and sampler.
        llm.add_request([rng.randint(100, 5000) for _ in range(700)], params)
        seqs, prefill = llm.scheduler.schedule()
        assert prefill and all(s.num_cached_tokens + s.num_scheduled_tokens < len(s) for s in seqs)
        head_calls = []
        sampler_calls = []
        head_hook = runner.model.lm_head.register_forward_hook(lambda *args: head_calls.append(1))
        sampler_hook = runner.sampler.register_forward_hook(lambda *args: sampler_calls.append(1))
        token_ids = runner.run(seqs, prefill)
        head_hook.remove(); sampler_hook.remove()
        assert token_ids == [None] * len(seqs)
        assert not head_calls and not sampler_calls
        llm.scheduler.postprocess(seqs, token_ids, prefill)
        while not llm.is_finished():
            llm.step()
        print("Chunk sampling PASS: intermediate chunk skipped LM head and sampler")

        llm.add_request(short, params)
        seen = set()
        checked = 0
        max_error = 0.0
        mismatches = 0
        forward_calls = []
        hook = runner.model.register_forward_hook(lambda *args: forward_calls.append(1))
        try:
            for step in range(40):
                if step == 1: llm.add_request(long, params)
                if step == 3:
                    duplicate = Sequence(long, params)
                    llm.scheduler.add(duplicate)
                if llm.is_finished(): break
                seqs, prefill = llm.scheduler.schedule()
                if not prefill: seen.add("decode_graph")
                for seq in seqs:
                    if seq.num_cached_tokens + seq.num_scheduled_tokens < len(seq): seen.add("chunk")
                    if step == 3 and seq is duplicate and seq.num_cached_tokens == 512: seen.add("prefix_hit")
                prepare = runner.prepare_prefill if prefill else runner.prepare_decode
                inputs, positions = prepare(seqs)
                sample_seq_indices = [
                    i for i, seq in enumerate(seqs)
                    if not prefill or seq.num_cached_tokens + seq.num_scheduled_tokens == len(seq)
                ]
                n_decode = get_context().num_decode
                if n_decode:
                    seen.add("mixed")
                    # Check the compact TP representation produces identical inputs.
                    copied_inputs, copied_positions = prepare(pickle.loads(pickle.dumps(seqs)))
                    torch.testing.assert_close(inputs, copied_inputs, rtol=0, atol=0)
                    torch.testing.assert_close(positions, copied_positions, rtol=0, atol=0)
                before = len(forward_calls)
                actual = runner.run_model(inputs, positions, prefill).float().clone()
                if n_decode: assert len(forward_calls) - before == 1
                assert actual.shape[0] == len(sample_seq_indices) and torch.isfinite(actual).all()
                reset_context()
                caches = [(m, m.k_cache, m.v_cache) for m in runner.model.modules() if hasattr(m, "k_cache")]
                try:
                    for m, _, _ in caches:
                        m.k_cache = m.v_cache = torch.empty(0, device="cuda")
                    for output_index, seq_index in enumerate(sample_seq_indices):
                        seq = seqs[seq_index]
                        end = seq.num_cached_tokens + seq.num_scheduled_tokens
                        ref = Sequence(seq.token_ids[:end], params)
                        ref.num_scheduled_tokens = end
                        ref_inputs, ref_positions = runner.prepare_prefill([ref])
                        expected = runner.run_model(ref_inputs, ref_positions, True)[0].float()
                        max_error = max(max_error, (actual[output_index] - expected).abs().max().item())
                        mismatches += int(actual[output_index].argmax().item() != expected.argmax().item())
                        torch.testing.assert_close(actual[output_index], expected, atol=0.25, rtol=0.02)
                        checked += 1
                        reset_context()
                finally:
                    for m, k, v in caches: m.k_cache, m.v_cache = k, v
                token_ids = [None] * len(seqs)
                for seq_index, token_id in zip(sample_seq_indices, actual.argmax(-1).tolist()):
                    token_ids[seq_index] = token_id
                llm.scheduler.postprocess(seqs, token_ids, prefill)
            assert llm.is_finished()
            assert seen == {"mixed", "chunk", "prefix_hit", "decode_graph"}, seen
            assert mismatches == 0, mismatches
            print(f"GPU logits PASS: rows={checked}, max_abs={max_error:.6f}, argmax_mismatches={mismatches}, paths={sorted(seen)}")
        finally:
            hook.remove()
            atexit.unregister(llm.exit)
            llm.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--model", default=str(Path.home() / "huggingface/Qwen3-0.6B"))
    args = parser.parse_args()
    if args.gpu: gpu_tests(args.model)
    else: cpu_tests()
