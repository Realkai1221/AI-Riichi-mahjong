"""Local research commands with machine-readable reports."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path


def _write_report(report: dict, output: str | None) -> None:
    content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    print(content, end="")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Riichi AI research foundation checks")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor", help="Check Python, simulator and tensor backend")
    doctor.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    doctor.add_argument("--output")
    smoke = subparsers.add_parser("smoke", help="Run bounded random legal full matches")
    smoke.add_argument("--games", type=int, default=10)
    smoke.add_argument("--seed", type=int, default=20261016)
    smoke.add_argument("--max-events", type=int, default=100_000,
                       help="Maximum simulator step calls per match")
    smoke.add_argument("--output")
    benchmark = subparsers.add_parser("benchmark", help="Measure a synthetic residual CNN")
    benchmark.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    benchmark.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32, 128])
    benchmark.add_argument("--iterations", type=int, default=10)
    benchmark.add_argument("--warmup", type=int, default=3)
    benchmark.add_argument("--width", type=int, default=256)
    benchmark.add_argument("--blocks", type=int, default=12)
    benchmark.add_argument("--output")
    evaluate = subparsers.add_parser("evaluate", help="Paired full-match evaluation with all four focal seats")
    evaluate.add_argument("--candidate", default="shanten", help="Heuristic name or checkpoint:/absolute/path.pt")
    evaluate.add_argument("--baseline", default="random")
    evaluate.add_argument("--opponents", nargs=3, default=["random"] * 3)
    evaluate.add_argument("--blocks", type=int, default=16, help="Independent seed blocks; eight full matches per block")
    evaluate.add_argument("--seed", type=int, default=20261016)
    evaluate.add_argument("--max-steps", type=int, default=100_000)
    evaluate.add_argument("--failure-dir", default="reports/failures")
    evaluate.add_argument("--workers", type=int, default=1)
    evaluate.add_argument("--output")
    generate = subparsers.add_parser("generate", help="Collect audited synthetic complete matches")
    generate.add_argument("--directory", required=True, help="Empty local directory outside iCloud for bulk data")
    generate.add_argument("--matches", type=int, default=1000)
    generate.add_argument("--seed", type=int, default=2026101701)
    generate.add_argument("--workers", type=int, default=4)
    generate.add_argument("--max-steps", type=int, default=100_000)
    generate.add_argument("--teacher", choices=["shanten_v2", "defense_v2"], default="shanten_v2")
    generate.add_argument("--output")
    args = parser.parse_args(argv)
    if args.command == "generate":
        from .dataset import generate_dataset
        report = generate_dataset(args.directory, args.matches, args.seed, args.workers,
                                  args.max_steps, teacher=args.teacher)
        _write_report(report, args.output)
        return 0 if report["valid"] else 1
    if args.command == "evaluate":
        if args.blocks < 1 or args.max_steps < 1 or not 0 <= args.seed < 2**64:
            parser.error("blocks/max-steps must be positive and seed must be in [0, 2**64)")
        from .evaluation import evaluate_paired
        report = evaluate_paired(args.candidate, args.baseline, opponents=args.opponents,
                                 blocks=args.blocks, seed=args.seed, max_steps=args.max_steps,
                                 failure_dir=args.failure_dir, workers=args.workers)
        _write_report(report, args.output)
        return 0 if report["valid"] else 1
    if args.command == "doctor":
        from .diagnostics import check_backend, runtime_report
        report = runtime_report()
        report["backend_check"] = check_backend(args.device)
        _write_report(report, args.output)
        return 0 if report["backend_check"]["passed"] else 1
    if args.command == "smoke":
        if args.games < 1 or args.seed < 0 or args.max_events < 1:
            parser.error("games/max-events must be positive and seed must be nonnegative")
        if args.seed + args.games - 1 >= 2**64:
            parser.error("every match seed must be less than 2**64")
        from .simulator import run_random_hanchan
        start = time.perf_counter()
        summaries = [asdict(run_random_hanchan(args.seed + i, args.max_events))
                     for i in range(args.games)]
        elapsed = time.perf_counter() - start
        completed = sum(s["completed"] for s in summaries)
        decisions = sum(s["decisions"] for s in summaries)
        report = {
            "games": args.games, "first_seed": args.seed,
            "completed_games": completed,
            "truncated_games": args.games - completed,
            "decisions": decisions,
            "elapsed_seconds": elapsed,
            "hanchans_per_hour": completed * 3600 / elapsed,
            "decisions_per_second": decisions / elapsed,
            "scope": "Random legal policy with audit overhead; not neural self-play throughput or rule certification.",
            "matches": summaries,
        }
        _write_report(report, args.output)
        return 0 if all(s["completed"] and s["conservation_ok"] for s in summaries) else 1
    from .benchmark import benchmark_compute
    report = benchmark_compute(args.device, args.batch_sizes, args.iterations,
                               args.warmup, args.width, args.blocks)
    _write_report(report, args.output)
    return 0

