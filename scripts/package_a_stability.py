"""Bounded Package A2 launcher. Numerical work is always inside a durable lease.

The controller imports only stdlib before acquiring its time lease, then launches
an isolated worker. Import, data loading, numerical tests, failures and shutdown
are charged. This script never dispatches on import.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import secrets
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
DEFAULTS = {"limit_seconds": 2700, "maximum_attempts": 4, "memory_bytes": 32 * 1024**3,
    "artifact_bytes": 2_000_000_000, "steps": 18500, "seeds": [202610171, 202610172],
    "epsilon": .01, "batch_size": 256, "learning_rate": .0003, "validation_seed": 2026101720,
    "validation_per_stratum": 5000, "validate_every": 500, "diagnostic_every": 100,
    "diagnostic_per_stratum": 32, "probe_steps": 256, "reserve_seconds": 120,
    "safety": {"warmup_steps": 1000, "scale_factor": 10., "scale_floor": 1.,
        "raw_logit_absolute_floor": 100., "persistence": 3, "gradient_threshold": 50.,
        "gradient_persistence": 3, "policy_ce_threshold": 10., "policy_ce_persistence": 50}}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def config_at(path):
    path = Path(path).resolve()
    config = {**DEFAULTS, **json.loads(path.read_text())["stability"]}
    config["safety"] = {**DEFAULTS["safety"], **config["safety"]}
    for key in ("data_root", "reports_root", "stage2b_config"):
        if key not in config:
            raise ValueError(f"Missing stability.{key}")
        config[key] = str((PROJECT / config[key]).resolve())
    return path, config, digest(path)


class Ledger:
    """File-locked reservations survive kills; a lost lease is never refunded."""
    def __init__(self, path, identity, limit=2700):
        self.path, self.identity, self.limit = Path(path), identity, limit

    @contextmanager
    def locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            data = json.loads(self.path.read_text()) if self.path.exists() else {
                "schema": "package-a2-budget-v1", "identity": self.identity,
                "limit_seconds": self.limit, "operations": []}
            if data["identity"] != self.identity or data["limit_seconds"] != self.limit:
                raise ValueError("Ledger identity or ceiling changed")
            yield data
            atomic_json(self.path, data)

    @staticmethod
    def charged(data):
        return sum(x["reserved_seconds"] if x["status"] == "active" else x["elapsed_seconds"]
                   for x in data["operations"])

    def snapshot(self):
        with self.locked() as data:
            return {**data, "charged_seconds": self.charged(data),
                    "remaining_seconds": max(0., self.limit - self.charged(data))}

    @contextmanager
    def operation(self, name, allowance, reserve=0):
        if not math.isfinite(allowance) or allowance <= 0 or reserve < 0:
            raise ValueError("Invalid budget reservation")
        with self.locked() as data:
            if any(x["status"] == "active" for x in data["operations"]):
                raise RuntimeError("An active or unreconciled lease already exists")
            available = self.limit - self.charged(data) - reserve
            allowed = min(allowance, available)
            if allowed <= 0:
                raise RuntimeError("Cumulative A2 budget exhausted")
            identifier = len(data["operations"])
            data["operations"].append({"id": identifier, "name": name, "status": "active",
                "reserved_seconds": allowed, "elapsed_seconds": None, "started_unix": time.time(),
                "token": secrets.token_hex(24)})
        started = time.perf_counter()
        failure = None
        try:
            yield allowed
        except BaseException as error:
            failure = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            elapsed = time.perf_counter() - started
            with self.locked() as data:
                data["operations"][identifier].update(status="failed" if failure else "completed",
                    elapsed_seconds=elapsed, failure=failure, exceeded_reservation=elapsed > allowed)


def implementation_identity(config):
    paths = sorted((PROJECT / "src/riichi_ai").glob("*.py")) + [Path(__file__).resolve(), Path(config["stage2b_config"])]
    return {str(path.relative_to(PROJECT)): digest(path) for path in paths}


def require_worker_lease(ledger, token, allowance, expected_name):
    active = [x for x in ledger.snapshot()["operations"] if x["status"] == "active"]
    if (len(active) != 1 or active[0].get("token") != token or active[0]["name"] != expected_name
            or allowance > active[0]["reserved_seconds"] or allowance <= 0
            or time.time() - active[0]["started_unix"] >= active[0]["reserved_seconds"]):
        raise RuntimeError("Worker requires its matching active, unexpired controller lease")
    return active[0]


def disk_bytes(config):
    total = 0
    for root in (config["data_root"], config["reports_root"]):
        for path in Path(root).rglob("*"):
            try:
                if path.is_file(): total += path.stat().st_size
            except FileNotFoundError:
                pass  # Atomic checkpoint rename may remove a temporary file.
    return total


def disk_guard(config):
    total = disk_bytes(config)
    if total >= config["artifact_bytes"]:
        raise RuntimeError("A2 artifact ceiling reached; preserve evidence and stop")
    return total


def execute(command, allowance, config):
    # Parent reserves shutdown time; the cooperative worker must stop earlier.
    import psutil
    controller_hash = digest(__file__)
    process = subprocess.Popen(command, cwd=PROJECT, start_new_session=True)
    started = time.perf_counter()
    peak_tree_rss = 0
    try:
        while process.poll() is None:
            if time.perf_counter() - started >= max(.1, allowance - 10):
                raise TimeoutError("A2 subprocess wall-time ceiling reached")
            processes = []
            try:
                parent = psutil.Process(process.pid)
                processes = [parent, *parent.children(recursive=True)]
            except psutil.NoSuchProcess:
                pass
            rss = 0
            for child in processes:
                try:
                    rss += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            peak_tree_rss = max(peak_tree_rss, rss)
            if rss >= config["memory_bytes"]:
                raise MemoryError("A2 sampled process-tree RSS ceiling reached")
            disk_guard(config)
            if digest(__file__) != controller_hash:
                raise RuntimeError("A2 controller changed during leased work")
            time.sleep(.2)
        if process.returncode:
            raise RuntimeError(f"Worker returned {process.returncode}")
        return {"sampled_peak_tree_rss_bytes": peak_tree_rss, "elapsed_seconds": time.perf_counter() - started}
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise


def worker(config, identity, recipe, seed, device, allowance, probe=False, baseline=False):
    # The controller's lease began before this interpreter and these imports.
    started = time.perf_counter()
    implementation = implementation_identity(config)
    import torch
    from riichi_ai.package_a_stability import train_stability
    from riichi_ai.stage2b_data import load_efficiency_split
    torch.set_num_threads(1)
    stage = json.loads(Path(config["stage2b_config"]).read_text())
    if digest(Path(stage["stage2a_dataset"]) / "manifest.json") != stage["stage2a_manifest_sha256"]:
        raise RuntimeError("Historical training source identity differs")
    if json.loads((Path(stage["data_root"]) / "dataset/manifest.json").read_text())["config"] != stage:
        raise RuntimeError("Stage 2B validation provenance differs")
    data = None if baseline else load_efficiency_split(stage["stage2a_dataset"], Path(stage["data_root"]) / "derived-stage2a", "train",
        assisted=True, max_matches=stage["training"]["training_matches"], memory_limit_bytes=config["memory_bytes"] // 4)
    validation = load_efficiency_split(Path(stage["data_root"]) / "dataset", Path(stage["data_root"]) / "derived-stage2b",
        "validation", assisted=True, memory_limit_bytes=config["memory_bytes"] // 4)
    load_seconds = time.perf_counter() - started
    if baseline:
        from riichi_ai.model import NeuralPolicy
        from riichi_ai.package_a_stability import evaluate_probability
        selection_path = PROJECT / "reports/stage2b/selection.json"
        selection = json.loads(selection_path.read_text())
        if digest(selection["checkpoint"]) != selection["checkpoint_sha256"]:
            raise RuntimeError("Frozen primary checkpoint hash differs")
        primary = NeuralPolicy.from_checkpoint(selection["checkpoint"], device=device)
        probabilities = evaluate_probability(primary.model, validation, device=device,
            batch_size=config["batch_size"], max_seconds=max(.1, allowance - (time.perf_counter() - started) - 15))
        output = {"schema": "package-a2-primary-probability-v1", "config_sha256": identity,
            "implementation": implementation, "sources_unchanged": implementation == implementation_identity(config),
            "checkpoint_sha256": selection["checkpoint_sha256"], "selection_sha256": digest(selection_path),
            "probability_metrics": probabilities, "elapsed_seconds": time.perf_counter() - started}
        atomic_json(Path(config["reports_root"]) / "baseline-probability.json", output)
        if not probabilities["complete"] or not output["sources_unchanged"]:
            raise RuntimeError("Primary probability measurement incomplete or sources changed")
        return
    name = "probe" if probe else f"{recipe}-{seed}"
    report = train_stability(data, Path(config["data_root"]) / "models" / name, validation=validation,
        seed=seed, epsilon=0 if recipe == "control" else config["epsilon"],
        steps=config["probe_steps"] if probe else config["steps"], batch_size=config["batch_size"],
        lr=config["learning_rate"], device=device, max_seconds=max(.1, allowance - load_seconds - 15),
        memory_limit_bytes=config["memory_bytes"], validate_every=config["validate_every"],
        validation_seed=config["validation_seed"], validation_per_stratum=config["validation_per_stratum"],
        diagnostic_every=config["diagnostic_every"], diagnostic_per_stratum=config["diagnostic_per_stratum"],
        safety=config["safety"], protocol_manifest={"config_sha256": identity,
            "stage2b_config_sha256": digest(config["stage2b_config"]),
            "controller_sha256": digest(__file__), "recipe": recipe, "probe": probe},
        progress_callback=lambda event: print(json.dumps({"recipe": name, **event}), flush=True),
        artifact_guard=lambda: disk_guard(config))
    report.update(config_sha256=identity, controller_sha256=digest(__file__), implementation=implementation,
                  recipe=recipe, seed=seed, import_and_loading_seconds=load_seconds)
    if implementation != implementation_identity(config):
        report.update(complete=False, durability_passed=False, failure={"type": "IntegrityError", "message": "Implementation changed during worker"})
    if probe:
        fixed = sum(v["elapsed_seconds"] for v in report["validations"])
        full = (report["terminal_full_validation"] or {}).get("elapsed_seconds", 0.)
        prob = (report["terminal_probability_metrics"] or {}).get("elapsed_seconds", 0.)
        selected = (report["selected_full_validation"] or {}).get("elapsed_seconds", 0.) if report["best_step"] != report["completed_steps"] else 0.
        residual = max(0., report["elapsed_total_seconds"] - report["elapsed_training_seconds"] - fixed - full - prob - selected)
        per_run = (report["elapsed_training_seconds"] / max(1, report["completed_steps"]) * config["steps"]
                   + fixed * math.ceil(config["steps"] / config["validate_every"]) + full * 2 + prob + residual + load_seconds)
        report["planning"] = {"single_full_run_seconds": per_run,
            "four_runs_with_25_percent_allowance_seconds": per_run * 4 * 1.25,
            "reserve_seconds": config["reserve_seconds"],
            "requires_complete_probe": report["complete"],
            "scope": "Measured short instrumented path extrapolation; full-horizon duration is not guaranteed."}
    else:
        historical_path = PROJECT / "reports/stage2b" / f"run-features_rank0-{seed}.json"
        old = json.loads(historical_path.read_text())
        historical = old["training"]
        report["historical_comparison"] = {
            "report_sha256": digest(historical_path),
            "initial_state_matches": report["initial_state_sha256"] == historical["initialization"]["state_sha256"],
            "common_initialization_matches": report["initialization"]["common_weights_sha256"] == historical["initialization"]["manifest"]["common_weights_sha256"],
            "data_manifest_matches": report["data_manifest"] == historical["data_manifest"],
            "full_stream_matches_if_full_horizon": (report["sampled_index_digest"] == historical["sampled_index_digest"]
                if report["completed_steps"] == historical["completed_steps"] else None),
            "historical_first_policy_ce_over_10": 9380 if seed == 202610171 else 10888,
            "alarm_before_historical_ce_collapse": report["stop_reason"] == "safety_alarm" and report["completed_steps"] < (9380 if seed == 202610171 else 10888)}
        if (not all(report["historical_comparison"][k] for k in ("initial_state_matches", "common_initialization_matches", "data_manifest_matches"))
                or report["historical_comparison"]["full_stream_matches_if_full_horizon"] is False):
            report.update(complete=False, durability_passed=False,
                failure={"type": "IntegrityError", "message": "Historical initialization/data/full-stream comparison failed"})
    atomic_json(Path(config["reports_root"]) / ("probe.json" if probe else f"run-{name}.json"), report)
    print(json.dumps({"phase": "a2-probe" if probe else "a2-run", "recipe": recipe, "seed": seed,
        "completed_steps": report["completed_steps"], "stop_reason": report["stop_reason"],
        "durability_passed": report["durability_passed"], "planning": report.get("planning")}), flush=True)
    if report["failure"] or report["stop_reason"] in ("time_budget", "memory_budget"):
        raise RuntimeError("A2 attempt did not finish within operational constraints; evidence retained")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT / "configs/package_a.json"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("budget")
    budgeted = commands.add_parser("budgeted")
    budgeted.add_argument("--name", required=True)
    budgeted.add_argument("--max-seconds", type=float, default=180)
    budgeted.add_argument("--reserve-seconds", type=float, default=120)
    budgeted.add_argument("subprocess_command", nargs=argparse.REMAINDER)
    probe = commands.add_parser("probe")
    probe.add_argument("--device", default="mps", choices=("cpu", "mps", "cuda"))
    probe.add_argument("--max-seconds", type=float, default=180)
    baseline = commands.add_parser("baseline-probability")
    baseline.add_argument("--device", default="mps", choices=("cpu", "mps", "cuda"))
    baseline.add_argument("--max-seconds", type=float, default=120)
    run = commands.add_parser("run")
    run.add_argument("--recipe", choices=("control", "smooth"), required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--device", default="mps", choices=("cpu", "mps", "cuda"))
    run.add_argument("--max-seconds", type=float)
    hidden = commands.add_parser("_worker")
    hidden.add_argument("--recipe", choices=("control", "smooth"), required=True)
    hidden.add_argument("--seed", type=int, required=True)
    hidden.add_argument("--device", required=True)
    hidden.add_argument("--allowance", type=float, required=True)
    hidden.add_argument("--probe", action="store_true")
    hidden.add_argument("--baseline", action="store_true")
    hidden.add_argument("--lease-token", required=True)
    args = parser.parse_args()
    config_path, config, identity = config_at(args.config)
    ledger = Ledger(Path(config["data_root"]) / "budget-ledger.json", identity, config["limit_seconds"])
    if args.command == "_worker":
        expected = "baseline-probability" if args.baseline else "probe" if args.probe else f"formal:{args.recipe}:{args.seed}"
        require_worker_lease(ledger, args.lease_token, args.allowance, expected)
        worker(config, identity, args.recipe, args.seed, args.device, args.allowance, args.probe, args.baseline)
        return
    if args.command == "budget":
        print(json.dumps(ledger.snapshot()), flush=True)
        return
    if args.command == "budgeted":
        command = args.subprocess_command
        if command and command[0] == "--": command = command[1:]
        if not command: raise ValueError("A subprocess command is required")
        with ledger.operation(args.name, args.max_seconds, args.reserve_seconds) as allowance:
            disk_guard(config)
            watchdog = execute(command, allowance, config)
            if digest(config_path) != identity:
                raise RuntimeError("Package A configuration changed during leased work")
            disk_guard(config)
        print(json.dumps(ledger.snapshot()), flush=True)
        return
    reports = Path(config["reports_root"])
    reports.mkdir(parents=True, exist_ok=True)
    is_probe = args.command == "probe"
    is_baseline = args.command == "baseline-probability"
    exploratory = is_probe or is_baseline
    recipe = "control" if exploratory else args.recipe
    seed = config["seeds"][0] if exploratory else args.seed
    attempts = reports / "attempts"
    attempts.mkdir(exist_ok=True)
    marker = reports / ("baseline-probability-attempt.json" if is_baseline else "probe-attempt.json") if exploratory else attempts / f"{recipe}-{seed}.json"
    if marker.exists(): raise FileExistsError("This attempt already exists; no automatic retries")
    count = len(list(attempts.glob("*.json")))
    if not exploratory:
        if seed not in config["seeds"] or count >= config["maximum_attempts"]:
            raise ValueError("Undeclared seed or maximum formal attempts reached")
        planned = [(r, s) for r in ("control", "smooth") for s in config["seeds"]]
        if (recipe, seed) != planned[count]:
            raise ValueError("Run order is frozen: both controls, then both smoothing seeds")
        probe_report = json.loads((reports / "probe.json").read_text())
        if (not probe_report["complete"] or probe_report["config_sha256"] != identity
                or probe_report["implementation"] != implementation_identity(config)):
            raise RuntimeError("A complete unchanged instrumented probe is required")
        if count == 0 and (probe_report["planning"]["four_runs_with_25_percent_allowance_seconds"]
                + config["reserve_seconds"] > ledger.snapshot()["remaining_seconds"]):
            raise RuntimeError("Four full-horizon comparisons do not fit the measured budget estimate")
    available = ledger.snapshot()["remaining_seconds"] - config["reserve_seconds"]
    allowance_request = args.max_seconds if args.max_seconds is not None else available / (config["maximum_attempts"] - count)
    marker_record = {"recipe": recipe, "seed": seed, "config_sha256": identity, "status": "reserved",
                     "requested_seconds": allowance_request, "started_unix": time.time(), "failure": None}
    # Exclusive marker counts crashes, stopped attempts and failed starts.
    with marker.open("x") as stream: json.dump(marker_record, stream, indent=2)
    try:
        with ledger.operation("baseline-probability" if is_baseline else "probe" if is_probe else f"formal:{recipe}:{seed}", allowance_request,
                              config["reserve_seconds"]) as allowance:
            disk_guard(config)
            command = [sys.executable, str(Path(__file__).resolve()), "--config", str(config_path), "_worker",
                       "--recipe", recipe, "--seed", str(seed), "--device", args.device,
                       "--allowance", str(allowance - 10),
                       "--lease-token", next(x["token"] for x in ledger.snapshot()["operations"] if x["status"] == "active")]
            if is_probe: command.append("--probe")
            if is_baseline: command.append("--baseline")
            marker_record["watchdog"] = execute(command, allowance, config)
            if digest(config_path) != identity:
                raise RuntimeError("Package A configuration changed during leased work")
            disk_guard(config)
        marker_record["status"] = "completed"
    except BaseException as error:
        marker_record.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        marker_record["budget"] = ledger.snapshot()
        marker_record["finished_unix"] = time.time()
        atomic_json(marker, marker_record)
    print(json.dumps({"phase": "budget", **ledger.snapshot()}), flush=True)


if __name__ == "__main__":
    main()
