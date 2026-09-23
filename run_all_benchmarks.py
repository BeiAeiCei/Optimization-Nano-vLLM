#!/usr/bin/env python3
"""Run identical Nano-vLLM workloads in the baseline and changed checkouts."""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from bench import ensure_comparable


ROOT = Path(__file__).resolve().parent
DEFAULT_BASELINE = ROOT.parent / "nano-vllm"

WORKLOADS = {
    "single-batch": [
        "--workload-mode", "single-batch",
    ],
    "single-long": [
        "--workload-mode", "single-long",
        "--long-prompt-tokens", "3968",
        "--long-output-tokens", "128",
        "--max-model-len", "4096",
        "--max-num-batched-tokens", "1024",
    ],
    "multi-constant": [
        "--workload-mode", "multi",
        "--arrival-mode", "constant",
        "--users", "8",
        "--requests-per-user", "4",
        "--request-rate", "20",
        "--min-prompt", "512",
        "--max-prompt", "1024",
        "--min-output", "64",
        "--max-output", "128",
        "--max-num-batched-tokens", "2048",
        "--max-model-len", "4096",
        "--seed", "0",
    ],
    "multi-poisson": [
        "--workload-mode", "multi",
        "--arrival-mode", "poisson",
        "--users", "8",
        "--requests-per-user", "4",
        "--request-rate", "20",
        "--min-prompt", "512",
        "--max-prompt", "1024",
        "--min-output", "64",
        "--max-output", "128",
        "--max-num-batched-tokens", "2048",
        "--max-model-len", "4096",
        "--seed", "0",
    ],
    "multi-wave": [
        "--workload-mode", "multi",
        "--arrival-mode", "wave",
        "--users", "8",
        "--requests-per-user", "4",
        "--wave-interval-ms", "250",
        "--min-prompt", "2048",
        "--max-prompt", "3000",
        "--min-output", "64",
        "--max-output", "128",
        "--max-num-batched-tokens", "1024",
        "--max-model-len", "4096",
        "--seed", "0",
    ],
}

METRICS = [
    ("Runtime", ("total_runtime_s",), "s", False),
    ("Throughput", ("output_throughput_tok_s",), "tok/s", True),
    ("TTFT mean", ("ttft", "mean_ms"), "ms", False),
    ("TTFT p95", ("ttft", "p95_ms"), "ms", False),
    ("TPOT mean", ("tpot", "mean_ms"), "ms/token", False),
    ("TPOT p95", ("tpot", "p95_ms"), "ms/token", False),
    ("E2E mean", ("e2e", "mean_ms"), "ms", False),
    ("E2E p95", ("e2e", "p95_ms"), "ms", False),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all benchmark modes on baseline and changed Nano-vLLM"
    )
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--changed-dir", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--only", nargs="+", choices=tuple(WORKLOADS),
                        help="run only selected workloads")
    parser.add_argument("--dry-run", action="store_true",
                        help="print commands without loading the model")
    parser.add_argument("--diagnostics", action="store_true",
                        help="enable the same scheduler counters in both engines")
    return parser.parse_args()


def validate_repo(path: Path):
    if not (path / "bench.py").is_file() or not (path / "nanovllm").is_dir():
        raise SystemExit(f"not a Nano-vLLM checkout: {path}")


def run_one(repo: Path, label: str, workload: str, output: Path, dry_run: bool,
            diagnostics: bool = False):
    if output.exists():
        raise SystemExit(f"refusing to overwrite benchmark result: {output}")
    command = [
        sys.executable,
        str(ROOT / "bench.py"),
        "--engine-dir", str(repo),
        "--label", label,
        "--output", str(output),
        *WORKLOADS[workload],
    ]
    if diagnostics:
        command.append("--diagnostics")
    print(f"\n[{workload}] {label}", flush=True)
    print("  " + " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=repo, check=True)


def nested_metric(metrics: dict, path: tuple[str, ...]):
    value = metrics
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return float(value)


def validate_pair(workload: str, baseline: dict, changed: dict):
    ensure_comparable(baseline, changed)
    if baseline["provenance"]["harness_sha256"] != changed["provenance"]["harness_sha256"]:
        raise RuntimeError(f"{workload}: benchmark harness changed between runs")
    base_workload = baseline["workload"]
    changed_workload = changed["workload"]
    if base_workload.get("request_count") != changed_workload.get("request_count"):
        raise RuntimeError(f"{workload}: request counts differ")
    base_hash = base_workload.get("sha256")
    changed_hash = changed_workload.get("sha256")
    if base_hash is not None and base_hash != changed_hash:
        raise RuntimeError(f"{workload}: workload hashes differ")
    if base_workload.get("requested_output_tokens") != changed_workload.get("requested_output_tokens"):
        raise RuntimeError(f"{workload}: requested output token counts differ")


def format_value(value: float, unit: str):
    precision = 3 if unit == "s" else 2
    return f"{value:.{precision}f}"


def make_report(workloads: list[str], output_dir: Path):
    lines = [
        "# Nano-vLLM benchmark comparison",
        "",
        "Negative change is better for latency/runtime; positive change is better for throughput.",
        "",
        "| Workload | Metric | Baseline | Changed | Change | Result |",
        "|---|---|---:|---:|---:|:---:|",
    ]
    summary = {}
    for workload in workloads:
        baseline = json.loads((output_dir / f"{workload}-baseline.json").read_text())
        changed = json.loads((output_dir / f"{workload}-changed.json").read_text())
        validate_pair(workload, baseline, changed)
        summary[workload] = {"baseline": baseline, "changed": changed}
        for name, path, unit, higher_is_better in METRICS:
            base_value = nested_metric(baseline["metrics"], path)
            changed_value = nested_metric(changed["metrics"], path)
            if base_value is None or changed_value is None:
                continue
            change = (changed_value / base_value - 1.0) * 100.0 if base_value else 0.0
            improved = change > 0 if higher_is_better else change < 0
            result = "better" if improved else ("same" if abs(change) < 0.01 else "worse")
            lines.append(
                f"| {workload} | {name} ({unit}) | {format_value(base_value, unit)} | "
                f"{format_value(changed_value, unit)} | {change:+.2f}% | {result} |"
            )
    report = "\n".join(lines) + "\n"
    (output_dir / "comparison.md").write_text(report)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    return report


def main():
    args = parse_args()
    baseline_dir = args.baseline_dir.expanduser().resolve()
    changed_dir = args.changed_dir.expanduser().resolve()
    validate_repo(baseline_dir)
    validate_repo(changed_dir)

    workloads = args.only or list(WORKLOADS)
    output_dir = args.output_dir or (
        changed_dir / "benchmark-results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for workload in workloads:
        run_one(
            baseline_dir,
            "baseline",
            workload,
            output_dir / f"{workload}-baseline.json",
            args.dry_run,
            args.diagnostics,
        )
        run_one(
            changed_dir,
            "changed",
            workload,
            output_dir / f"{workload}-changed.json",
            args.dry_run,
            args.diagnostics,
        )

    if args.dry_run:
        print(f"\nDry run only. Results would be written to: {output_dir}")
        return

    report = make_report(workloads, output_dir)
    print("\n" + report)
    print(f"Saved report: {output_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
