"""Reproduce the bounded Stage 2A campaign one reviewable step at a time.

Run from the project with .venv/bin/python scripts/stage2.py COMMAND.
Bulk artifacts use configs/stage2_pilot.json's local, non-iCloud data root.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time


PROJECT = Path(__file__).resolve().parents[1]
PLAN = json.loads((PROJECT / "configs/stage2_pilot.json").read_text())
DATA = Path(PLAN["data_root"])
REPORTS = PROJECT / "reports/stage2"


def save(name: str, report: dict) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / name).write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def disk_usage() -> int:
    return sum(path.stat().st_size for root in (DATA, REPORTS)
               for path in root.rglob("*") if path.is_file())


def check_disk() -> None:
    if disk_usage() > PLAN["limits"]["new_artifact_bytes"]:
        raise RuntimeError("Stage 2 artifact budget exceeded; retain evidence and stop")


def collect(*, smoke: bool = False) -> None:
    from riichi_ai.dataset import generate_dataset
    from riichi_ai.dataset_validation import verify_dataset

    target = DATA / ("checks-data" if smoke else "dataset")
    report = generate_dataset(
        target, matches=10 if smoke else PLAN["dataset"]["matches"],
        root_seed=PLAN["seed_namespaces"]["learning_checks" if smoke else "dataset"],
        workers=1 if smoke else PLAN["dataset"]["workers"],
        disk_limit_bytes=PLAN["limits"]["new_artifact_bytes"] - disk_usage() - 20_000_000,
        memory_limit_bytes=PLAN["limits"]["memory_gib"] * 1024**3,
    )
    label = "checks-data" if smoke else "dataset"
    save(label + "-manifest.json", report)
    if not report["valid"]:
        raise RuntimeError("Collection failed; inspect the preserved manifest")
    audit = verify_dataset(target, replay_matches=3)
    save(label + "-audit.json", audit)
    if not audit["valid"]:
        raise RuntimeError("Dataset verification failed; inspect the preserved audit report")
    check_disk()
    print(json.dumps({"stage": label, "valid": report["valid"], "matches": report["completed_matches"],
                      "decisions": report["decisions"], "bytes": disk_usage(), "audit": audit}, default=str), flush=True)


def learning_checks() -> None:
    import torch
    from riichi_ai.dataset import load_split
    from riichi_ai.training import check_cpu_mps_parity, check_resume, tiny_overfit

    torch.set_num_threads(1)
    dataset = load_split(DATA / "checks-data", "train")
    optional = [dataset[index] for index in range(min(len(dataset), 1000)) if dataset[index]["optional"]][:32]
    started = time.perf_counter()
    parity = check_cpu_mps_parity(optional)
    save("cpu-mps-parity.json", parity)
    if not parity["passed"]:
        raise RuntimeError("Model CPU/MPS parity failed; do not scale training")
    resume = {device: check_resume(optional, DATA / ("resume-check-" + device), device=device)
              for device in ("cpu", "mps")}
    save("resume-checks.json", resume)
    if not all(result["passed"] for result in resume.values()):
        raise RuntimeError("Resume check failed; do not scale training")
    tiny = tiny_overfit(dataset, DATA / "tiny-fit", device="mps", sample_count=32,
                        max_seconds=180, max_steps=1500)
    save("tiny-fit.json", tiny)
    summary = {"elapsed_seconds": time.perf_counter() - started,
               "tiny_fit_passed": tiny["tiny_fit_passed"],
               "parity_passed": parity["passed"], "resume_passed": True}
    save("learning-checks.json", summary)
    check_disk()
    print(json.dumps(summary), flush=True)
    if not tiny["tiny_fit_passed"]:
        raise RuntimeError("Tiny distribution-learning check failed; do not scale training")


def train_curves() -> None:
    import torch
    from riichi_ai.dataset import load_split
    from riichi_ai.evaluation import source_manifest
    from riichi_ai.model import NeuralPolicy
    from riichi_ai.training import evaluate_model, train

    campaign_started = time.perf_counter()
    checks = json.loads((REPORTS / "learning-checks.json").read_text())
    if not checks["tiny_fit_passed"] or not checks["resume_passed"] or not checks["parity_passed"]:
        raise RuntimeError("Learning checks must pass first")
    torch.set_num_threads(1)
    validation = load_split(DATA / "dataset", "validation")
    sources = source_manifest()
    jobs = [(matches, "soft") for matches in PLAN["learning_curve_training_matches"]]
    jobs.append((PLAN["learning_curve_training_matches"][-1], "one_hot"))
    consumed = checks["elapsed_seconds"]
    runs = []
    for index, (matches, mode) in enumerate(jobs):
        consumed = checks["elapsed_seconds"] + time.perf_counter() - campaign_started
        remaining = PLAN["limits"]["total_training_seconds"] - consumed - 60
        if remaining < 30:
            break
        dataset = load_split(DATA / "dataset", "train", max_matches=matches)
        consumed = checks["elapsed_seconds"] + time.perf_counter() - campaign_started
        remaining = PLAN["limits"]["total_training_seconds"] - consumed - 60
        if remaining < 30:
            break
        allocation = remaining / (len(jobs) - index)
        name = f"{mode}-{matches}"
        print(json.dumps({"stage": "train", "run": name, "rows": len(dataset), "seconds_cap": allocation}), flush=True)
        result = train(
            dataset, DATA / "models" / name, validation=validation,
            device="mps", steps=PLAN["training"]["initial_steps_per_curve"],
            batch_size=PLAN["training"]["batch_size"], seed=PLAN["training"]["seed"],
            max_seconds=allocation, target_mode=mode,
            rank_weight=PLAN["training"]["rank_loss_weight"], source_manifest=sources,
            memory_limit_bytes=PLAN["limits"]["memory_gib"] * 1024**3,
        )
        save(name + "-training.json", result)
        if (result["failure"] or result["stop_reason"] == "memory_budget"
                or not result["sources_unchanged"] or result["best_validation"] is None):
            raise RuntimeError(f"Training run {name} failed its engineering checks")
        # Use the same soft-target metric definition for both target experiments.
        selected = NeuralPolicy.from_checkpoint(result["best_checkpoint"], device="mps")
        remaining = PLAN["limits"]["total_training_seconds"] - checks["elapsed_seconds"] - (time.perf_counter() - campaign_started) - 30
        metrics = evaluate_model(selected.model, validation, device="mps", max_samples=10000,
                                 seed=PLAN["training"]["seed"] ^ 0x5A17, target_mode="soft", max_seconds=max(0, remaining))
        if not metrics["complete"]:
            raise RuntimeError("Validation exceeded the cumulative budget; no checkpoint selected")
        consumed = checks["elapsed_seconds"] + time.perf_counter() - campaign_started
        runs.append({"run": name, "training_matches": matches, "target_mode": mode,
                     "checkpoint": result["best_checkpoint"], "checkpoint_sha256": digest(Path(result["best_checkpoint"])),
                     "completed_steps": result["completed_steps"], "best_step": result["best_step"],
                     "elapsed_total_seconds": result["elapsed_total_seconds"], "validation": metrics})
        save("learning-curves.json", {"runs": runs, "cumulative_training_budget_seconds": consumed})
        del dataset, selected
        gc.collect()
        torch.mps.empty_cache()
        check_disk()
        print(json.dumps(runs[-1]), flush=True)
    if not runs:
        raise RuntimeError("No training run completed")
    selected = max(runs, key=lambda run: (run["validation"]["tie_agreement"], -run["validation"]["policy_kl"]))
    frozen = DATA / "models" / "selected.pt"
    if frozen.exists():
        raise FileExistsError("A frozen selected checkpoint already exists")
    shutil.copyfile(selected["checkpoint"], frozen)
    selection = {"selection_rule": PLAN["training"]["selection"], "selected": selected,
                 "checkpoint": str(frozen), "checkpoint_sha256": digest(frozen),
                 "cumulative_training_budget_seconds": consumed,
                 "test_read_before_freeze": False, "source_manifest": sources}
    save("selection.json", selection)  # Freeze before opening the test split.
    policy = NeuralPolicy.from_checkpoint(frozen, device="mps")
    heldout = {}
    for split in ("validation", "test"):
        dataset = validation if split == "validation" else load_split(DATA / "dataset", split)
        remaining = PLAN["limits"]["total_training_seconds"] - checks["elapsed_seconds"] - (time.perf_counter() - campaign_started) - 5
        heldout[split] = evaluate_model(policy.model, dataset, device="mps", target_mode="soft", max_seconds=max(0, remaining))
    save("heldout-metrics.json", {"checkpoint_sha256": digest(frozen), "metrics": heldout})
    consumed = checks["elapsed_seconds"] + time.perf_counter() - campaign_started
    save("training-budget.json", {"elapsed_seconds_including_learning_checks_loading_and_validation": consumed,
                                  "limit_seconds": PLAN["limits"]["total_training_seconds"],
                                  "within_budget": consumed <= PLAN["limits"]["total_training_seconds"]})
    check_disk()
    print(json.dumps({"stage": "frozen", "selection": selection, "heldout": heldout}), flush=True)


def evaluate() -> None:
    from riichi_ai.evaluation import evaluate_paired
    from riichi_ai.statistics import noninferiority_test, plan_noninferiority

    selected = json.loads((REPORTS / "selection.json").read_text())
    checkpoint = Path(selected["checkpoint"])
    if digest(checkpoint) != selected["checkpoint_sha256"]:
        raise RuntimeError("Frozen checkpoint has changed")
    options = PLAN["evaluation"]
    common = {"candidate": "checkpoint:" + str(checkpoint), "baseline": options["baseline"],
              "opponents": options["opponents"], "workers": 4, "failure_dir": REPORTS / "failures"}
    pilot = evaluate_paired(**common, blocks=options["pilot_blocks"], seed=PLAN["seed_namespaces"]["student_pilot"])
    save("student-pilot.json", pilot)
    if not pilot["valid"]:
        raise RuntimeError("Pilot evaluation invalid; inspect failure evidence")
    planning = plan_noninferiority([block["mean_utility_difference"] for block in pilot["blocks"]])
    save("confirmation-plan.json", planning)  # Persist count before fresh outcomes.
    print(json.dumps({"stage": "evaluation-pilot", "summary": pilot["arm_summaries"], "confirmation": planning}), flush=True)
    if planning["run_confirmation"]:
        confirmation = evaluate_paired(**common, blocks=planning["confirmation_blocks"],
                                       seed=PLAN["seed_namespaces"]["student_confirmation"])
        confirmation["purpose"] = "prespecified_fixed_budget_noninferiority_confirmation"
        save("student-confirmation.json", confirmation)
        result = (noninferiority_test([block["mean_utility_difference"] for block in confirmation["blocks"]],
                                      planned_blocks=planning["confirmation_blocks"])
                  if confirmation["valid"] else {"valid": False, "passed": False, "status": "invalid_campaign"})
        save("noninferiority-result.json", result)
        print(json.dumps(result), flush=True)


def benchmarks() -> None:
    from riichi_ai.evaluation import evaluate_paired

    panel = PLAN["evaluation"]["opponents"]
    comparison = evaluate_paired("shanten_v2", "shanten", opponents=panel, blocks=8,
                                 seed=PLAN["seed_namespaces"]["teacher_comparison"], workers=4)
    save("teacher-v2-v1.json", comparison)
    results = []
    for workers in (1, 4):
        report = evaluate_paired("defense_v2", "shanten_v2", opponents=panel, blocks=8,
                                 seed=PLAN["seed_namespaces"]["throughput"], workers=workers)
        save(f"throughput-{workers}-workers.json", report)
        results.append(report)
    serial, parallel = results
    same = ([match["digest"] for match in serial["matches"]]
            == [match["digest"] for match in parallel["matches"]]
            and serial["statistics"] == parallel["statistics"])
    summary = {"same_deterministic_results": same, "serial_seconds": serial["elapsed_seconds"],
               "parallel_seconds": parallel["elapsed_seconds"],
               "speed_ratio": serial["elapsed_seconds"] / parallel["elapsed_seconds"],
               "all_valid": all(report["valid"] for report in (*results, comparison)),
               "scope": "Short audited CPU heuristic campaigns, ordered serial then parallel; not sustained thermal or neural throughput."}
    save("throughput-comparison.json", summary)
    print(json.dumps(summary), flush=True)
    if not same or not summary["all_valid"]:
        raise RuntimeError("Evaluation reproducibility benchmark failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare-checks", "learning-checks", "collect", "train", "evaluate", "benchmarks"))
    args = parser.parse_args()
    {"prepare-checks": lambda: collect(smoke=True), "learning-checks": learning_checks,
     "collect": collect, "train": train_curves, "evaluate": evaluate, "benchmarks": benchmarks}[args.command]()


if __name__ == "__main__":
    main()
