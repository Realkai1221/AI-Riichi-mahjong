"""Leased, read-only Stage 3B action-support and gradient diagnostic.

Numerical imports, preparation, tests and failures all consume the same durable
300-second ledger. No match collector, optimizer or checkpoint writer is called.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

from package_a_stability import Ledger, atomic_json, digest, disk_guard, require_worker_lease

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = [
    dict(label="initial", path="/Users/example/Library/Application Support/RiichiAI/stage2b/models/selected.pt",
         sha256="5130185ba8a397b84446ba8dc2d02f068a9448db5c0373f2aeb357e4af525a6f"),
    dict(label="final", path="/Users/example/Library/Application Support/RiichiAI/stage3a/boundaries/batch-3.pt",
         sha256="dff490dc2033a057c5d869e4cf4e215187b885edfc944d415643c296a600f05a"),
]
CONSTRAINTS = dict(new_matches=0, optimizer_updates=0, checkpoint_writes=0,
    book_training=False, test_split_loading=False, external_acquisition=False,
    paid_compute=False, strength_evaluation=False, sampling_policy_promotion=False)


def settings(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    exact = dict(schema="stage3b-diagnostic-config-v1", limit_seconds=300,
        reserve_seconds=20, prepare_allowance_seconds=30, diagnostic_allowance_seconds=180,
        test_allowance_seconds=60, memory_bytes=32 * 1024**3, artifact_bytes=500_000_000,
        maximum_rows=512, probe_rows=32, torch_threads=1, temperatures=[1., 10., 100.],
        mixture_epsilon=.05, include_mps=True, constraints=CONSTRAINTS,
        checkpoints=CHECKPOINTS)
    if any(config.get(key) != value for key, value in exact.items()):
        raise ValueError("Authorized Stage 3B diagnostic scope changed")
    for key in ("data_root", "reports_root", "source_root"):
        config[key] = str((ROOT / config[key]).resolve())
    if Path(config["data_root"]) == Path(config["source_root"]):
        raise ValueError("Diagnostic output cannot overwrite the source")
    return path, config, digest(path)


def implementation():
    # The book converter is independent and may be authored during preparation.
    files = [p for p in (ROOT / "src/riichi_ai").glob("*.py")
             if not p.name.startswith("book_scenarios")]
    files += [Path(__file__).resolve(), ROOT / "scripts/package_a_stability.py",
              ROOT / "pyproject.toml", ROOT / "requirements-macos-arm64.lock"]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}


def verify_inputs(config):
    for item in config["checkpoints"]:
        if digest(item["path"]) != item["sha256"]:
            raise ValueError("Frozen checkpoint changed")
    previous = json.loads((ROOT / "reports/stage3b/previous-stage3a-snapshot.json").read_text())
    if previous.get("valid") is not True:
        raise ValueError("Stage 3A historical snapshot must verify before diagnosis")


def execute(command, allowance, config, evidence, source):
    import psutil  # Deliberately after durable reservation.
    started = time.perf_counter()
    process = None
    evidence.update(sampled_peak_tree_rss_bytes=0, artifact_bytes_peak=0,
                    sample_interval_seconds=.2, exit_code=None)
    try:
        process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
        while True:
            parent = psutil.Process(os.getpid())
            rss = 0
            for child in [parent, *parent.children(recursive=True)]:
                try:
                    rss += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            evidence["sampled_peak_tree_rss_bytes"] = max(evidence["sampled_peak_tree_rss_bytes"], rss)
            evidence["artifact_bytes_peak"] = max(evidence["artifact_bytes_peak"], disk_guard(config))
            if rss >= config["memory_bytes"]:
                raise MemoryError("Stage 3B process-tree RSS ceiling reached")
            if source != implementation():
                raise RuntimeError("Numerical implementation changed during operation")
            if process.poll() is not None:
                if process.returncode:
                    raise RuntimeError(f"Worker returned {process.returncode}")
                return
            if time.perf_counter() - started >= max(.1, allowance - 10):
                raise TimeoutError("Stage 3B wall-time ceiling reached")
            time.sleep(.2)
    finally:
        if process is not None:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            evidence["exit_code"] = process.returncode
        evidence["elapsed_seconds"] = time.perf_counter() - started


def worker(config, identity, allowance, phase):
    started = time.perf_counter()
    source = implementation()
    verify_inputs(config)
    import torch
    from riichi_ai.stage3b import select_rows, run_diagnostic
    torch.set_num_threads(config["torch_threads"])
    reports = Path(config["reports_root"])
    plan_path = reports / "plan.json"
    if phase == "prepare":
        if plan_path.exists():
            raise FileExistsError("Frozen selection already exists")
        selection = select_rows(config["source_root"], max_rows=config["maximum_rows"],
                                probe_rows=config["probe_rows"])
        result = dict(schema="stage3b-plan-v1", config_sha256=identity,
            implementation=source, frozen_before_diagnostic_outputs=True,
            selection=selection, checkpoints=config["checkpoints"],
            temperatures=config["temperatures"], mixture_epsilon=config["mixture_epsilon"],
            constraints=config["constraints"], python=platform.python_version(),
            versions={name: importlib.metadata.version(name)
                      for name in ("torch", "numpy", "riichienv", "mahjong", "psutil")},
            implementation_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
        atomic_json(plan_path, result)
        print(json.dumps({"phase": phase, "plan_sha256": digest(plan_path)}), flush=True)
    else:
        plan = json.loads(plan_path.read_text())
        if plan["implementation"] != source or plan["config_sha256"] != identity:
            raise ValueError("Frozen implementation or protocol changed")
        output = reports / "diagnostic.json"
        if output.exists():
            raise FileExistsError("Do not replace diagnostic evidence")
        result = run_diagnostic(plan["selection"], config["checkpoints"],
            deadline=time.monotonic() + max(.1, allowance - (time.perf_counter() - started) - 20),
            include_mps=config["include_mps"])
        result.update(config_sha256=identity, plan_sha256=digest(plan_path),
                      implementation=source, constraints=config["constraints"])
        atomic_json(output, result)
        print(json.dumps({"phase": phase, "report_sha256": digest(output)}), flush=True)
    verify_inputs(config)
    if implementation() != source:
        raise RuntimeError("Numerical source changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/stage3b.json"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("budget")
    commands.add_parser("prepare")
    commands.add_parser("diagnose")
    check = commands.add_parser("budgeted")
    check.add_argument("--name", required=True)
    check.add_argument("--max-seconds", type=float, default=60)
    check.add_argument("arguments", nargs=argparse.REMAINDER)
    hidden = commands.add_parser("_worker")
    hidden.add_argument("--phase", choices=("prepare", "diagnose"), required=True)
    hidden.add_argument("--allowance", type=float, required=True)
    hidden.add_argument("--lease-token", required=True)
    args = parser.parse_args()
    path, config, identity = settings(args.config)
    ledger = Ledger(Path(config["data_root"]) / "budget-ledger.json", identity, 300)
    if args.command == "budget":
        print(json.dumps(ledger.snapshot(), indent=2))
        return
    if args.command == "_worker":
        require_worker_lease(ledger, args.lease_token, args.allowance, args.phase)
        worker(config, identity, args.allowance, args.phase)
        return
    reports = Path(config["reports_root"])
    reports.mkdir(parents=True, exist_ok=True)
    source = implementation()
    if args.command == "budgeted":
        if not args.name.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Unsafe check name")
        name, maximum = f"check:{args.name}", min(args.max_seconds, config["test_allowance_seconds"])
        command = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
        if not command:
            raise ValueError("An explicit check command is required")
        marker = reports / f"checks-{args.name}.json"
    else:
        name = args.command
        maximum = config["prepare_allowance_seconds" if name == "prepare" else "diagnostic_allowance_seconds"]
        command = None
        marker = reports / f"{name}-attempt.json"
        verify_inputs(config)
        if name == "diagnose":
            prepare = json.loads((reports / "prepare-attempt.json").read_text())
            if prepare["status"] != "completed":
                raise ValueError("Successful frozen preparation required")
    if marker.exists():
        raise FileExistsError("Operation already attempted; preserve evidence")
    record = dict(name=name, config_sha256=identity, implementation=source,
                  status="reserved", started_unix=time.time(), watchdog={})
    atomic_json(marker, record)
    failure = None
    try:
        with ledger.operation(name, maximum, config["reserve_seconds"]) as allowance:
            if command is None:
                lease = next(x for x in ledger.snapshot()["operations"] if x["status"] == "active")
                command = [sys.executable, str(Path(__file__).resolve()), "--config", str(path),
                    "_worker", "--phase", name, "--allowance", str(allowance),
                    "--lease-token", lease["token"]]
            record["command"] = command
            execute(command, allowance, config, record["watchdog"], source)
            if digest(path) != identity or implementation() != source:
                raise RuntimeError("Source or configuration changed")
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        record.update(status="failed" if failure else "completed", failure=failure,
                      finished_unix=time.time(), budget=ledger.snapshot())
        atomic_json(marker, record)
        atomic_json(reports / "budget.json", ledger.snapshot())
    print(json.dumps({"phase": name, "charged_seconds": ledger.snapshot()["charged_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
