"""Online benchmark for Nano-vLLM.

Run this file once on the prefill-first implementation and once on the mixed
implementation with exactly the same workload arguments/seed, then compare the
two JSON files. Model loading and warmup are excluded from all reported times.

Examples:
    python bench.py --workload-mode multi --users 32 --requests-per-user 4
    python bench.py --workload-mode single-batch --num-seqs 256 \
        --max-input-len 1024 --max-output-len 1024
    python bench.py --workload-mode single-long --long-prompt-tokens 32000 \
        --long-output-tokens 128 --max-model-len 32768
    python bench.py --compare prefill.json mixed.json

Custom trace format (CSV, no header):
    arrival_ms,prompt_tokens,output_tokens[,user_id]
"""

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RequestSpec:
    request_id: int
    user_id: str
    arrival_ms: float
    prompt_tokens: int
    output_tokens: int


@dataclass
class RequestState:
    spec: RequestSpec
    sequence: Any
    first_token_s: float | None = None
    finish_s: float | None = None
    generated_tokens: int = 0


def percentile(values: list[float], percent: float) -> float:
    """Nearest-rank-like percentile that has no NumPy dependency."""
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(percent / 100.0 * (len(ordered) - 1))))
    return ordered[index]


def describe_ms(values_s: list[float]) -> dict[str, float]:
    values_ms = [value * 1000.0 for value in values_s]
    if not values_ms:
        return {name: math.nan for name in
                ("mean_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "max_ms")}
    return {
        "mean_ms": statistics.fmean(values_ms),
        "p50_ms": percentile(values_ms, 50),
        "p90_ms": percentile(values_ms, 90),
        "p95_ms": percentile(values_ms, 95),
        "p99_ms": percentile(values_ms, 99),
        "max_ms": max(values_ms),
    }


def parse_trace(path: str) -> list[RequestSpec]:
    requests = []
    for line_no, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        values = [value.strip() for value in line.split(",")]
        if len(values) not in (3, 4):
            raise ValueError(f"{path}:{line_no}: expected 3 or 4 CSV fields")
        try:
            arrival_ms = float(values[0])
            prompt_tokens = int(values[1])
            output_tokens = int(values[2])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: invalid numeric value") from exc
        user_id = values[3] if len(values) == 4 and values[3] else f"user-{len(requests)}"
        if arrival_ms < 0 or prompt_tokens <= 0 or output_tokens <= 0:
            raise ValueError(f"{path}:{line_no}: arrival must be >= 0 and token counts > 0")
        requests.append(RequestSpec(len(requests), user_id, arrival_ms,
                                    prompt_tokens, output_tokens))
    if not requests:
        raise ValueError(f"{path}: trace contains no requests")
    return sorted(requests, key=lambda request: (request.arrival_ms, request.request_id))


def make_workload(args: argparse.Namespace) -> list[RequestSpec]:
    """Create an identical deterministic workload for both engine versions."""
    if args.trace:
        return parse_trace(args.trace)

    if args.workload_mode == "single-long":
        return [RequestSpec(
            request_id=0,
            user_id="user-0",
            arrival_ms=0.0,
            prompt_tokens=args.long_prompt_tokens,
            output_tokens=args.long_output_tokens,
        )]

    if args.workload_mode == "single-batch":
        # Match the original one-shot benchmark exactly: prompt lengths and
        # prompt IDs are generated first, then all sampling lengths.  The
        # token-ID draws must still be consumed here even though this helper
        # returns only request metadata, otherwise the output-length stream
        # differs from the original script.
        length_rng = random.Random(args.seed)
        prompt_lengths = []
        for _ in range(args.num_seqs):
            prompt_len = length_rng.randint(args.min_input_len, args.max_input_len)
            prompt_lengths.append(prompt_len)
            for _ in range(prompt_len):
                length_rng.randint(0, args.vocab_size)
        output_lengths = [
            length_rng.randint(args.min_output_len, args.max_output_len)
            for _ in range(args.num_seqs)
        ]
        return [RequestSpec(
            request_id=request_id,
            user_id="user-0",
            arrival_ms=0.0,
            prompt_tokens=prompt_lengths[request_id],
            output_tokens=output_lengths[request_id],
        ) for request_id in range(args.num_seqs)]

    length_rng = random.Random(args.seed)
    arrival_rng = random.Random(args.seed + 1)
    requests = []
    request_id = 0

    if args.arrival_mode == "wave":
        # Every user submits one request per wave. This deliberately produces
        # repeated bursts and is useful for observing decode interruption.
        for request_index in range(args.requests_per_user):
            arrival_ms = request_index * args.wave_interval_ms
            for user_index in range(args.users):
                requests.append(RequestSpec(
                    request_id,
                    f"user-{user_index}",
                    float(arrival_ms),
                    length_rng.randint(args.min_prompt, args.max_prompt),
                    length_rng.randint(args.min_output, args.max_output),
                ))
                request_id += 1
    else:
        total_requests = args.users * args.requests_per_user
        arrival_s = 0.0
        for index in range(total_requests):
            if index:
                if args.arrival_mode == "poisson":
                    arrival_s += arrival_rng.expovariate(args.request_rate)
                else:  # constant
                    arrival_s += 1.0 / args.request_rate
            user_index = index % args.users
            requests.append(RequestSpec(
                request_id,
                f"user-{user_index}",
                arrival_s * 1000.0,
                length_rng.randint(args.min_prompt, args.max_prompt),
                length_rng.randint(args.min_output, args.max_output),
            ))
            request_id += 1

    return sorted(requests, key=lambda request: (request.arrival_ms, request.request_id))


def validate_args(args: argparse.Namespace, workload: list[RequestSpec]):
    if not args.trace and args.workload_mode == "multi":
        if args.users <= 0 or args.requests_per_user <= 0:
            raise SystemExit("--users and --requests-per-user must be positive")
        if args.request_rate <= 0 or args.wave_interval_ms < 0:
            raise SystemExit("--request-rate must be positive and --wave-interval-ms must be >= 0")
        if args.min_prompt <= 0 or args.min_output <= 0:
            raise SystemExit("minimum token lengths must be positive")
        if args.max_prompt < args.min_prompt or args.max_output < args.min_output:
            raise SystemExit("maximum token lengths must be >= minimum token lengths")
    if (not args.trace and args.workload_mode == "single-long"
            and (args.long_prompt_tokens <= 0 or args.long_output_tokens <= 0)):
        raise SystemExit("--long-prompt-tokens and --long-output-tokens must be positive")
    if not args.trace and args.workload_mode == "single-batch":
        if args.num_seqs <= 0:
            raise SystemExit("--num-seqs must be positive")
        if args.min_input_len <= 0 or args.min_output_len <= 0:
            raise SystemExit("single-batch minimum token lengths must be positive")
        if (args.max_input_len < args.min_input_len
                or args.max_output_len < args.min_output_len):
            raise SystemExit("single-batch maximum token lengths must be >= minimum lengths")
    if not 0 < args.vocab_size <= 0xFFFFFFFF or args.temperature <= 0:
        raise SystemExit("--vocab-size must fit uint32 and --temperature must be positive")
    if args.warmup_tokens < 0 or args.progress_every < 0:
        raise SystemExit("--warmup-tokens and --progress-every must be >= 0")
    longest = max(request.prompt_tokens + request.output_tokens for request in workload)
    if longest > args.max_model_len:
        raise SystemExit(
            f"longest request needs {longest} tokens, exceeding --max-model-len={args.max_model_len}"
        )


def synchronize_cuda():
    # step() currently synchronizes while copying sampled token IDs to the CPU,
    # but an explicit sync keeps timing correct if that implementation changes.
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def find_new_sequence(llm, known_ids: set[int]):
    """Find the Sequence created by add_request without changing engine code."""
    candidates = list(llm.scheduler.waiting) + list(llm.scheduler.running)
    new_sequences = [seq for seq in candidates if seq.seq_id not in known_ids]
    if len(new_sequences) != 1:
        raise RuntimeError(
            "could not identify the newly added Sequence; add_request/scheduler API has changed"
        )
    return new_sequences[0]


def run_original_single_batch(args):
    """Run the historical single-batch benchmark with passive latency metrics."""
    from random import randint, seed
    from nanovllm import LLM, SamplingParams

    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))]
                        for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True,
                                      max_tokens=randint(100, max_ouput_len))
                       for _ in range(num_seqs)]
    llm.generate(["Benchmark: "], SamplingParams())

    request_timings = {}
    metric_start = 0.0
    original_postprocess = llm.scheduler.postprocess

    def measured_postprocess(seqs, token_ids, is_prefill):
        original_postprocess(seqs, token_ids, is_prefill)
        now = time.perf_counter() - metric_start
        for seq in seqs:
            timing = request_timings.setdefault(seq.seq_id, {
                "first_token_s": None,
                "finish_s": None,
                "generated_tokens": 0,
            })
            generated_tokens = seq.num_completion_tokens
            if generated_tokens > 0 and timing["first_token_s"] is None:
                timing["first_token_s"] = now
            if seq.is_finished:
                timing["finish_s"] = now
                timing["generated_tokens"] = generated_tokens

    llm.scheduler.postprocess = measured_postprocess
    metric_start = time.perf_counter()
    start = time.time()
    try:
        llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    finally:
        llm.scheduler.postprocess = original_postprocess
    elapsed = time.time() - start
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / elapsed

    timings = list(request_timings.values())
    if len(timings) != num_seqs or any(timing["finish_s"] is None for timing in timings):
        raise RuntimeError("single-batch latency instrumentation missed requests")
    ttft_s = [timing["first_token_s"] for timing in timings]
    e2e_s = [timing["finish_s"] for timing in timings]
    tpot_s = [
        (timing["finish_s"] - timing["first_token_s"])
        / (timing["generated_tokens"] - 1)
        for timing in timings
        if timing["generated_tokens"] > 1
    ]
    ttft = describe_ms(ttft_s)
    tpot = describe_ms(tpot_s)
    e2e = describe_ms(e2e_s)

    print(f"Total: {total_tokens}tok, Time: {elapsed:.2f}s, Throughput: {throughput:.2f}tok/s")
    print(f"TTFT: mean={ttft['mean_ms']:.2f} ms  p50={ttft['p50_ms']:.2f}  "
          f"p95={ttft['p95_ms']:.2f}  p99={ttft['p99_ms']:.2f}")
    print(f"TPOT: mean={tpot['mean_ms']:.2f} ms/token  p50={tpot['p50_ms']:.2f}  "
          f"p95={tpot['p95_ms']:.2f}  p99={tpot['p99_ms']:.2f}")
    print(f"E2E:  mean={e2e['mean_ms']:.2f} ms  p50={e2e['p50_ms']:.2f}  "
          f"p95={e2e['p95_ms']:.2f}  p99={e2e['p99_ms']:.2f}")
    return {
        "label": args.label,
        "workload": {
            "mode": "single-batch",
            "seed": 0,
            "request_count": num_seqs,
            "requested_output_tokens": total_tokens,
        },
        "metrics": {
            "total_runtime_s": elapsed,
            "output_tokens": total_tokens,
            "output_throughput_tok_s": throughput,
            "ttft": ttft,
            "tpot": tpot,
            "e2e": e2e,
        },
    }


def run_online_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if args.engine_dir:
        engine_dir = Path(args.engine_dir).expanduser().resolve()
        if not (engine_dir / "nanovllm" / "engine" / "scheduler.py").is_file():
            raise SystemExit(f"not a Nano-vLLM checkout: {engine_dir}")
        sys.path.insert(0, str(engine_dir))
    from nanovllm import LLM, SamplingParams
    import nanovllm
    import torch

    workload = make_workload(args)
    validate_args(args, workload)
    package_dir = Path(nanovllm.__file__).resolve().parent
    if args.engine_dir and package_dir.parent != engine_dir:
        raise RuntimeError(f"imported unexpected engine: {package_dir}")
    source_hashes = {
        str(path.relative_to(package_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package_dir.rglob("*.py"))
    }
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"Loading model: {os.path.expanduser(args.model)}")
    llm = LLM(
        os.path.expanduser(args.model),
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    if args.warmup_tokens:
        print(f"Warmup: {args.warmup_tokens} output tokens (excluded from metrics)")
        llm.generate(
            ["Benchmark warmup"],
            SamplingParams(temperature=args.temperature, max_tokens=args.warmup_tokens,
                           ignore_eos=True),
            use_tqdm=False,
        )
    synchronize_cuda()

    diagnostics = None
    if args.diagnostics:
        diagnostics = {name: 0 for name in (
            "prefill_steps", "decode_steps", "mixed_steps", "chunked_requests",
            "scheduled_tokens", "preemptions", "discarded_cached_tokens",
        )}
        original_schedule = llm.scheduler.schedule
        original_preempt = llm.scheduler.preempt

        def measured_schedule():
            running_ids = {seq.seq_id for seq in llm.scheduler.running}
            seqs, is_prefill = original_schedule()
            decodes = sum(seq.seq_id in running_ids and
                          seq.num_cached_tokens == len(seq) - 1 for seq in seqs)
            mode = "mixed" if is_prefill and decodes else "prefill" if is_prefill else "decode"
            diagnostics[f"{mode}_steps"] += 1
            diagnostics["scheduled_tokens"] += sum(seq.num_scheduled_tokens for seq in seqs)
            diagnostics["chunked_requests"] += sum(
                seq.num_cached_tokens + seq.num_scheduled_tokens < len(seq) for seq in seqs
            )
            return seqs, is_prefill

        def measured_preempt(seq):
            diagnostics["preemptions"] += 1
            diagnostics["discarded_cached_tokens"] += seq.num_cached_tokens
            return original_preempt(seq)

        llm.scheduler.schedule = measured_schedule
        llm.scheduler.preempt = measured_preempt

    if not args.trace and args.workload_mode == "single-batch":
        # Recreate the same random stream as the original benchmark.  randint
        # is inclusive at both ends, matching the original randint(0, 10000).
        prompt_rng = random.Random(args.seed)
        prompts = {}
        for request in workload:
            prompt_len = prompt_rng.randint(args.min_input_len, args.max_input_len)
            prompt = [prompt_rng.randint(0, args.vocab_size) for _ in range(prompt_len)]
            if prompt_len != request.prompt_tokens:
                raise RuntimeError("single-batch workload RNG reconstruction diverged")
            prompts[request.request_id] = prompt
        for request in workload:
            output_len = prompt_rng.randint(args.min_output_len, args.max_output_len)
            if output_len != request.output_tokens:
                raise RuntimeError("single-batch output RNG reconstruction diverged")
    else:
        prompt_rng = random.Random(args.seed + 2)
        prompts = {
            request.request_id: [prompt_rng.randrange(args.vocab_size)
                                 for _ in range(request.prompt_tokens)]
            for request in workload
        }
    workload_hash = hashlib.sha256()
    for request in workload:
        workload_hash.update(
            f"{request.request_id},{request.user_id},{request.arrival_ms:.9f},"
            f"{request.prompt_tokens},{request.output_tokens}:".encode()
        )
        for token_id in prompts[request.request_id]:
            workload_hash.update(token_id.to_bytes(4, "little", signed=False))
    states: dict[int, RequestState] = {}
    known_sequence_ids: set[int] = set()
    active_sequence_ids: set[int] = set()
    next_request = 0
    completed = 0
    next_progress = args.progress_every
    step_count = 0
    busy_s = 0.0

    workload_mode = "trace" if args.trace else args.workload_mode
    num_users = len({request.user_id for request in workload})
    if workload_mode == "multi":
        requests_per_user = args.requests_per_user
    elif workload_mode == "single-batch":
        requests_per_user = args.num_seqs
    else:
        requests_per_user = 1
    print(
        f"Run [{args.label}]: mode={workload_mode}, {len(workload)} requests, {num_users} users, "
        f"arrival={workload[0].arrival_ms:.1f}..{workload[-1].arrival_ms:.1f} ms"
    )
    benchmark_start = time.perf_counter()

    while completed < len(workload):
        now_s = time.perf_counter() - benchmark_start

        # If the engine is empty, wait for the next request instead of spinning.
        if not active_sequence_ids and next_request < len(workload):
            next_arrival_s = workload[next_request].arrival_ms / 1000.0
            if now_s < next_arrival_s:
                time.sleep(next_arrival_s - now_s)
                now_s = time.perf_counter() - benchmark_start

        # Requests that arrived during the previous GPU step are admitted here.
        while (next_request < len(workload)
               and workload[next_request].arrival_ms / 1000.0 <= now_s):
            spec = workload[next_request]
            params = SamplingParams(
                temperature=args.temperature,
                max_tokens=spec.output_tokens,
                ignore_eos=True,
            )
            llm.add_request(prompts[spec.request_id], params)
            sequence = find_new_sequence(llm, known_sequence_ids)
            known_sequence_ids.add(sequence.seq_id)
            active_sequence_ids.add(sequence.seq_id)
            states[sequence.seq_id] = RequestState(spec, sequence)
            next_request += 1

        if not active_sequence_ids:
            continue

        step_start = time.perf_counter()
        llm.step()
        step_end = time.perf_counter()
        busy_s += step_end - step_start
        step_count += 1

        finished_ids = []
        for seq_id in active_sequence_ids:
            state = states[seq_id]
            generated_tokens = state.sequence.num_completion_tokens
            if generated_tokens > 0 and state.first_token_s is None:
                state.first_token_s = step_end - benchmark_start
            if state.sequence.is_finished:
                state.finish_s = step_end - benchmark_start
                state.generated_tokens = generated_tokens
                if generated_tokens != state.spec.output_tokens:
                    raise RuntimeError(f"request {state.spec.request_id}: unexpected output length")
                finished_ids.append(seq_id)
        for seq_id in finished_ids:
            active_sequence_ids.remove(seq_id)
            completed += 1

        if args.progress_every and finished_ids and completed >= next_progress:
            print(f"  completed {completed}/{len(workload)}, steps={step_count}", end="\r")
            next_progress = (completed // args.progress_every + 1) * args.progress_every

    synchronize_cuda()
    total_runtime_s = time.perf_counter() - benchmark_start
    if args.progress_every:
        print()

    ordered_states = sorted(states.values(), key=lambda state: state.spec.request_id)
    request_results = []
    ttft_s = []
    tpot_s = []
    e2e_s = []
    for state in ordered_states:
        arrival_s = state.spec.arrival_ms / 1000.0
        first_token_s = state.first_token_s - arrival_s
        finish_s = state.finish_s - arrival_s
        per_request_tpot_s = (
            (state.finish_s - state.first_token_s) / (state.generated_tokens - 1)
            if state.generated_tokens > 1 else None
        )
        ttft_s.append(first_token_s)
        e2e_s.append(finish_s)
        if per_request_tpot_s is not None:
            tpot_s.append(per_request_tpot_s)
        request_results.append({
            **asdict(state.spec),
            "generated_tokens": state.generated_tokens,
            "ttft_ms": first_token_s * 1000.0,
            "tpot_ms": None if per_request_tpot_s is None else per_request_tpot_s * 1000.0,
            "e2e_ms": finish_s * 1000.0,
        })

    output_tokens = sum(state.generated_tokens for state in ordered_states)
    result = {
        "label": args.label,
        "provenance": {
            "engine_dir": str(package_dir.parent),
            "sources_sha256": source_hashes,
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "num_kvcache_blocks": len(llm.scheduler.block_manager.blocks),
        },
        "diagnostics": diagnostics,
        "workload": {
            "mode": workload_mode,
            "seed": args.seed,
            "users": num_users,
            "requests_per_user": requests_per_user,
            "arrival_mode": args.arrival_mode,
            "request_rate": args.request_rate,
            "wave_interval_ms": args.wave_interval_ms,
            "trace": args.trace,
            "temperature": args.temperature,
            "vocab_size": args.vocab_size,
            "request_count": len(workload),
            "prompt_tokens": sum(request.prompt_tokens for request in workload),
            "requested_output_tokens": sum(request.output_tokens for request in workload),
            "first_arrival_ms": workload[0].arrival_ms,
            "last_arrival_ms": workload[-1].arrival_ms,
            "sha256": workload_hash.hexdigest(),
        },
        "engine": {
            "model": os.path.expanduser(args.model),
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
            "tensor_parallel_size": args.tensor_parallel_size,
            "diagnostics": args.diagnostics,
            "warmup_tokens": args.warmup_tokens,
        },
        "metrics": {
            "total_runtime_s": total_runtime_s,
            "gpu_busy_s": busy_s,
            "idle_s": max(0.0, total_runtime_s - busy_s),
            "step_count": step_count,
            "output_tokens": output_tokens,
            "output_throughput_tok_s": output_tokens / total_runtime_s,
            "ttft": describe_ms(ttft_s),
            "tpot": describe_ms(tpot_s),
            "e2e": describe_ms(e2e_s),
        },
        "requests": request_results,
    }
    return result


def print_result(result: dict[str, Any]):
    metrics = result["metrics"]
    ttft = metrics["ttft"]
    tpot = metrics["tpot"]
    e2e = metrics["e2e"]
    print(f"\n[{result['label']}] {result['workload']['request_count']} requests")
    print(f"Total runtime: {metrics['total_runtime_s']:.3f} s  "
          f"GPU-step busy: {metrics['gpu_busy_s']:.3f} s  steps: {metrics['step_count']}")
    print(f"Output throughput: {metrics['output_throughput_tok_s']:.2f} tok/s")
    print(f"TTFT: mean={ttft['mean_ms']:.2f} ms  p50={ttft['p50_ms']:.2f}  "
          f"p95={ttft['p95_ms']:.2f}  p99={ttft['p99_ms']:.2f}")
    print(f"TPOT: mean={tpot['mean_ms']:.2f} ms/token  p50={tpot['p50_ms']:.2f}  "
          f"p95={tpot['p95_ms']:.2f}  p99={tpot['p99_ms']:.2f}")
    print(f"E2E:  mean={e2e['mean_ms']:.2f} ms  p50={e2e['p50_ms']:.2f}  "
          f"p95={e2e['p95_ms']:.2f}  p99={e2e['p99_ms']:.2f}")


def ensure_comparable(base: dict[str, Any], candidate: dict[str, Any]):
    if base["workload"].get("sha256") != candidate["workload"].get("sha256"):
        raise SystemExit("result files used different prompt contents or request traces")
    base_requests = [
        (request["request_id"], request["user_id"], request["arrival_ms"],
         request["prompt_tokens"], request["output_tokens"])
        for request in base["requests"]
    ]
    candidate_requests = [
        (request["request_id"], request["user_id"], request["arrival_ms"],
         request["prompt_tokens"], request["output_tokens"])
        for request in candidate["requests"]
    ]
    if base_requests != candidate_requests:
        raise SystemExit("result files used different request traces; comparison would be invalid")
    if base["engine"] != candidate["engine"]:
        raise SystemExit("result files used different engine capacity/model settings")


def compare_results(base_path: str, candidate_path: str):
    base = json.loads(Path(base_path).read_text())
    candidate = json.loads(Path(candidate_path).read_text())
    ensure_comparable(base, candidate)
    print_result(base)
    print_result(candidate)
    print(f"\n[{candidate['label']} relative to {base['label']}]")

    comparisons = [
        ("TTFT mean", base["metrics"]["ttft"]["mean_ms"],
         candidate["metrics"]["ttft"]["mean_ms"], "ms", False),
        ("TTFT p95", base["metrics"]["ttft"]["p95_ms"],
         candidate["metrics"]["ttft"]["p95_ms"], "ms", False),
        ("TPOT mean", base["metrics"]["tpot"]["mean_ms"],
         candidate["metrics"]["tpot"]["mean_ms"], "ms/token", False),
        ("TPOT p95", base["metrics"]["tpot"]["p95_ms"],
         candidate["metrics"]["tpot"]["p95_ms"], "ms/token", False),
        ("Total runtime", base["metrics"]["total_runtime_s"],
         candidate["metrics"]["total_runtime_s"], "s", False),
        ("Output throughput", base["metrics"]["output_throughput_tok_s"],
         candidate["metrics"]["output_throughput_tok_s"], "tok/s", True),
    ]
    for name, old, new, unit, higher_is_better in comparisons:
        change = (new / old - 1.0) * 100.0
        improved = change > 0 if higher_is_better else change < 0
        verdict = "better" if improved else ("same" if abs(change) < 1e-9 else "worse")
        print(f"{name:18s} {old:10.3f} -> {new:10.3f} {unit:8s}  "
              f"{change:+7.2f}%  {verdict}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Nano-vLLM multi-user, single-batch, or single-long benchmark"
    )
    parser.add_argument("--compare", nargs=2, metavar=("BASE.json", "CANDIDATE.json"),
                        help="compare two previously saved runs without loading a model")
    parser.add_argument("--label", default="current", help="name stored in the result file")
    parser.add_argument("--engine-dir", help="import nanovllm from this checkout using this same harness")
    parser.add_argument("--diagnostics", action="store_true",
                        help="count scheduler modes and preemptions; adds CPU overhead")
    parser.add_argument("--legacy-single-batch", action="store_true",
                        help="reproduce the old fixed 256-request benchmark; ignores engine/workload options")
    parser.add_argument("--output", help="write complete measurements to this JSON file")
    parser.add_argument("--workload-mode", choices=("multi", "single-batch", "single-long"),
                        default="multi")
    parser.add_argument("--trace", help="CSV: arrival_ms,prompt_tokens,output_tokens[,user_id]; overrides workload mode")
    parser.add_argument("--arrival-mode", choices=("constant", "poisson", "wave"),
                        default="constant")
    parser.add_argument("--users", type=int, default=32)
    parser.add_argument("--requests-per-user", type=int, default=4)
    parser.add_argument("--num-seqs", type=int, default=256,
                        help="number of simultaneous requests in single-batch mode")
    parser.add_argument("--min-input-len", type=int, default=100)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--min-output-len", type=int, default=100)
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument("--long-prompt-tokens", type=int, default=3968,
                        help="prompt length in single-long mode")
    parser.add_argument("--long-output-tokens", type=int, default=128,
                        help="generated token count in single-long mode")
    parser.add_argument("--request-rate", type=float, default=20.0,
                        help="global arrivals/second for constant or poisson mode")
    parser.add_argument("--wave-interval-ms", type=float, default=1000.0,
                        help="time between all-user request waves")
    parser.add_argument("--min-prompt", type=int, default=128)
    parser.add_argument("--max-prompt", type=int, default=1024)
    parser.add_argument("--min-output", type=int, default=32)
    parser.add_argument("--max-output", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab-size", type=int, default=10000,
                        help="random prompt token IDs are sampled from [0, vocab-size)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=16,
                        help="print progress every N completions; 0 disables it")
    return parser


def main():
    args = build_parser().parse_args()
    if args.compare:
        compare_results(*args.compare)
        return
    if args.legacy_single_batch:
        if args.engine_dir or args.diagnostics or args.workload_mode != "single-batch" or args.trace:
            raise SystemExit("legacy mode requires --workload-mode single-batch without engine-dir/diagnostics/trace")
        result = run_original_single_batch(args)
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
            print(f"Saved: {output_path}")
        return
    result = run_online_benchmark(args)
    print_result(result)
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
