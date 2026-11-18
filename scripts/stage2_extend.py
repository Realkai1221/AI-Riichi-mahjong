"""One declared continuation of optimization; preserve the initial pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import torch

from stage2 import DATA, PLAN, REPORTS, check_disk, digest, save


PROTOCOL = json.loads((Path(__file__).resolve().parents[1] / "configs/stage2_extension.json").read_text())


def optimize() -> None:
    from riichi_ai.dataset import load_split
    from riichi_ai.model import NeuralPolicy
    from riichi_ai.training import evaluate_model, train

    began = time.perf_counter()
    prior = json.loads((REPORTS / "training-budget.json").read_text())["elapsed_seconds_including_learning_checks_loading_and_validation"]
    torch.set_num_threads(1)
    training = load_split(DATA / "dataset", "train", max_matches=PROTOCOL["training_matches"])
    validation = load_split(DATA / "dataset", "validation")
    previous = DATA / "models" / PROTOCOL["resume_run"] / "checkpoint.pt"
    saved = torch.load(previous, map_location="cpu", weights_only=True)
    remaining = PLAN["limits"]["total_training_seconds"] - prior - (time.perf_counter() - began) - 30
    cap = min(PROTOCOL["additional_training_seconds_cap"], remaining)
    if cap < 30:
        raise RuntimeError("Insufficient remaining approved training budget")
    result = train(
        training, DATA / "models" / "soft-800-extended", validation=validation,
        device="mps", steps=PROTOCOL["total_target_updates"], batch_size=PLAN["training"]["batch_size"],
        seed=PLAN["training"]["seed"], max_seconds=cap,
        memory_limit_bytes=PLAN["limits"]["memory_gib"] * 1024**3,
        source_manifest=saved["source_manifest"], resume_from=previous,
    )
    save("extended-training.json", result)
    if result["failure"] or result["stop_reason"] == "memory_budget" or not result["sources_unchanged"]:
        raise RuntimeError("Extended run failed engineering checks")
    frozen = DATA / "models" / "selected-extended.pt"
    if frozen.exists():
        raise FileExistsError("An extended frozen checkpoint already exists")
    shutil.copyfile(result["best_checkpoint"], frozen)
    selection = {"protocol": PROTOCOL, "checkpoint": str(frozen), "checkpoint_sha256": digest(frozen),
                 "best_step": result["best_step"], "validation_sample": result["best_validation"],
                 "test_split_reused": False, "initial_selection_preserved": "selection.json"}
    save("extended-selection.json", selection)
    model = NeuralPolicy.from_checkpoint(frozen, device="mps")
    remaining = PLAN["limits"]["total_training_seconds"] - prior - (time.perf_counter() - began) - 5
    metrics = evaluate_model(model.model, validation, device="mps", max_seconds=max(0, remaining))
    save("extended-validation.json", {"checkpoint_sha256": digest(frozen), "metrics": metrics})
    used = prior + time.perf_counter() - began
    save("extended-training-budget.json", {"initial_seconds": prior, "cumulative_seconds": used,
                                            "limit_seconds": PLAN["limits"]["total_training_seconds"],
                                            "within_budget": used <= PLAN["limits"]["total_training_seconds"]})
    check_disk()
    print(json.dumps({"stage": "extended-frozen", "selection": selection,
                      "validation": metrics, "cumulative_seconds": used}), flush=True)


def evaluate() -> None:
    from riichi_ai.evaluation import evaluate_paired
    from riichi_ai.statistics import noninferiority_test, plan_noninferiority

    selected = json.loads((REPORTS / "extended-selection.json").read_text())
    checkpoint = Path(selected["checkpoint"])
    if digest(checkpoint) != selected["checkpoint_sha256"]:
        raise RuntimeError("Frozen extended checkpoint changed")
    protocol = PROTOCOL["evaluation"]
    common = {"candidate": "checkpoint:" + str(checkpoint), "baseline": protocol["baseline"],
              "opponents": protocol["opponents"], "workers": 4, "failure_dir": REPORTS / "failures"}
    pilot = evaluate_paired(**common, blocks=protocol["pilot_blocks"], seed=protocol["pilot_seed"])
    save("extended-student-pilot.json", pilot)
    if not pilot["valid"]:
        raise RuntimeError("Extended student pilot is invalid")
    plan = plan_noninferiority([block["mean_utility_difference"] for block in pilot["blocks"]])
    save("extended-confirmation-plan.json", plan)
    print(json.dumps({"stage": "extended-evaluation", "summary": pilot["arm_summaries"], "confirmation": plan}), flush=True)
    if plan["run_confirmation"]:
        confirmation = evaluate_paired(**common, blocks=plan["confirmation_blocks"], seed=protocol["confirmation_seed"])
        confirmation["purpose"] = "prespecified_fixed_budget_noninferiority_confirmation"
        save("extended-student-confirmation.json", confirmation)
        result = (noninferiority_test([block["mean_utility_difference"] for block in confirmation["blocks"]],
                                      planned_blocks=plan["confirmation_blocks"])
                  if confirmation["valid"] else {"valid": False, "passed": False, "status": "invalid_campaign"})
        save("extended-noninferiority-result.json", result)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate"))
    args = parser.parse_args()
    {"train": optimize, "evaluate": evaluate}[args.command]()
