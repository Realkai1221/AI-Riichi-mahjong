#!/usr/bin/env python3
"""Fixed six-fit placement experiment; run only inside the Stage 3G lease."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from riichi_ai.stage3g_grp import (BATCH_ROWS, CALIBRATION, EPOCHS, FEATURE_NAMES, FOLD_SEED,
    INITIALIZATION_SEED, LEARNING_RATE, assign_folds, bootstrap_report, calibration_report,
    canonical, digest, fit_predictor, hand_targets, load_public_dataset, predict)


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def lease_context(args) -> dict:
    """Authenticate a live root-owned lease and the frozen fit configuration."""
    operation = os.environ.get("RIICHI_STAGE3G_OPERATION")
    token = os.environ.get("RIICHI_STAGE3G_TOKEN")
    config_path = ROOT / "configs/stage3g.json"
    raw = config_path.read_bytes()
    config = json.loads(raw)
    data = Path(config["data_root"]).resolve()
    ledger = json.loads((data / "budget-ledger.json").read_text())
    active = [item for item in ledger["operations"] if item["status"] == "active"]
    if (not token or not operation or len(active) != 1
            or active[0]["name"] != operation or active[0]["token"] != token
            or operation != "grp-fit"):
        raise ValueError("Placement fitting requires the active grp-fit numerical lease")
    expected = {"hidden_width": 32, "epochs": EPOCHS, "batch_rows": BATCH_ROWS,
        "learning_rate": LEARNING_RATE, "folds": 5, "bootstrap_replicates": 1000,
        "min_stage_games": CALIBRATION["minimum_unique_games"],
        "min_bin_games": CALIBRATION["minimum_bin_unique_games"],
        "supported_mass_min": CALIBRATION["minimum_supported_mass"],
        "weighted_absolute_bin_bias_max": CALIBRATION["maximum_weighted_absolute_bias"],
        "maximum_supported_bin_bias_max": CALIBRATION["maximum_supported_bin_bias"],
        "utility_bin_edges": CALIBRATION["utility_bins"]}
    if (config["grp"] != expected or config["grp_seed"] != FOLD_SEED
            or config["training_matches"] != 3200 or config["development_matches"] != 800
            or Path(args.corpus).resolve() != data / "corpus"
            or Path(args.output_dir).resolve() != data / "grp"):
        raise ValueError("Fixed placement fit recipe or canonical corpus/output path changed")
    return {"config_sha256": hashlib.sha256(raw).hexdigest(), "lease_operation": operation}


def run(args) -> dict:
    torch.set_num_threads(1)
    started = time.monotonic()
    deadline = started + args.max_seconds
    context = lease_context(args)
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Placement output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_public_dataset(args.corpus)
    folds = assign_folds(dataset.splits["training"])
    assignment = {"schema": "riichi-ai-stage3g-crossfit-v1", "seed": FOLD_SEED,
        "corpus_manifest_sha256": dataset.manifest_sha256, "folds": folds,
        "assignment_sha256": digest(folds), "fold_unit": "complete match"}
    write_new(output / "fold-assignment.json", canonical(assignment))
    probabilities = np.full((len(dataset.y), 4), np.nan, dtype=np.float64)
    fitted, identities = [], {}
    for fold in list(range(5)) + [None]:
        name = f"fold-{fold}" if fold is not None else "final"
        fit_ids = [match for match in dataset.splits["training"] if fold is None or folds[match] != fold]
        predict_ids = (dataset.splits["development"] if fold is None else
                       [match for match in dataset.splits["training"] if folds[match] == fold])
        if set(fit_ids) & set(predict_ids):
            raise ValueError("Cross-fitting leaks held-out games")
        model, result = fit_predictor(dataset, fit_ids, deadline)
        rows = dataset.rows_for(predict_ids)
        probabilities[rows] = predict(model, dataset.x[rows])
        saved = {"schema": "riichi-ai-stage3g-placement-predictor-v1",
            "model_state_dict": model.state_dict(), "architecture": [len(FEATURE_NAMES), 32, 32, 4],
            "activation": "tanh", "feature_names": list(FEATURE_NAMES),
            "corpus_manifest_sha256": dataset.manifest_sha256, "fold": fold,
            "initialization_seed": INITIALIZATION_SEED, "epochs": EPOCHS,
            "training_match_ids": sorted(fit_ids), "prediction_match_ids": sorted(predict_ids),
            "crossfit_assignment_sha256": digest(folds)}
        payload = io.BytesIO()
        torch.save(saved, payload)
        raw = payload.getvalue()
        identities[name] = hashlib.sha256(raw).hexdigest()
        write_new(output / f"predictor-{name}.pt", raw)
        result.update(name=name, predictor_sha256=identities[name], held_out_matches=len(predict_ids),
                      held_out_match_ids_sha256=digest(sorted(predict_ids)))
        fitted.append(result)
        progress = {"schema": "riichi-ai-stage3g-placement-progress-v1", "complete": False,
            **context, "corpus_manifest_sha256": dataset.manifest_sha256,
            "completed_fits": fitted, "elapsed_seconds": time.monotonic() - started}
        # Each completion has its own immutable receipt; failures retain the prefix.
        write_new(output / f"completed-{name}.json", canonical(progress))
        print(json.dumps({"event": "placement_fit_completed", "name": name,
                          "elapsed_seconds": result["elapsed_seconds"], "training_matches": len(fit_ids)}), flush=True)
    dev = dataset.rows_for(dataset.splits["development"])
    gate = calibration_report(probabilities[dev], dataset.baseline[dev], dataset.y[dev],
                              dataset.weights[dev], dataset.groups[dev], dataset.stages[dev])
    bootstrap = bootstrap_report(probabilities[dev], dataset.baseline[dev], dataset.y[dev],
        dataset.weights[dev], dataset.groups[dev], dataset.stages[dev], gate, deadline)
    targets = hand_targets(dataset, probabilities, folds, identities, gate)
    targets.update(context)
    target_raw = canonical(targets)
    write_new(output / "hand-targets.json", target_raw)
    prediction_buffer = io.BytesIO()
    np.savez_compressed(prediction_buffer, probabilities=probabilities, ranks=dataset.y,
        weights=dataset.weights, groups=dataset.groups, stages=dataset.stages, baseline=dataset.baseline,
        match_ids=np.asarray(dataset.match_ids), public_features=dataset.x)
    write_new(output / "public-predictions.npz", prediction_buffer.getvalue())
    if time.monotonic() >= deadline:
        raise TimeoutError("Placement report deadline exceeded")
    report = {"schema": "riichi-ai-stage3g-placement-report-v1", "complete": True,
        **context,
        "corpus_manifest_sha256": dataset.manifest_sha256, "training_matches": 3200,
        "development_matches": 800, "hand_states": len(dataset.y),
        "public_feature_names": list(FEATURE_NAMES), "source_target_hashes": dataset.source_hashes,
        "fit_recipe": {"folds": 5, "fits": 6, "epochs": EPOCHS, "batch_rows": BATCH_ROWS,
            "optimizer": "Adam", "learning_rate": LEARNING_RATE, "device": "cpu", "threads": 1,
            "initialization_seed": INITIALIZATION_SEED, "fold_seed": FOLD_SEED,
            "architecture": [len(FEATURE_NAMES), 32, 32, 4], "activation": "tanh"},
        "fits": fitted, "gate": gate, "calibration_gate_passed": gate["passed"],
        "crossfit_assignment_sha256": digest(folds), "bootstrap": bootstrap,
        "target_path": str(output / "hand-targets.json"), "target_sha256": hashlib.sha256(target_raw).hexdigest(),
        "elapsed_seconds": time.monotonic() - started,
        "versions": {"python": platform.python_version(), "torch": version("torch"), "numpy": version("numpy")},
        "limitations": ["Single initialization mechanism experiment", "Public hand-start predictor for a fixed opponent panel",
            "Calibration gate is development evidence, not causal or competitive strength evidence",
            "One-hand placement differences are a surrogate objective, not a full decision return"]}
    raw = canonical(report)
    write_new(output / "report.json", raw)
    if args.report:
        write_new(Path(args.report), raw)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report")
    parser.add_argument("--max-seconds", type=float, default=300.)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 600:
        parser.error("--max-seconds must fit the declared <=600-second GRP reservation")
    # Controller owns the process-tree deadline and durable numerical ledger.
    if not os.environ.get("RIICHI_STAGE3G_TOKEN"):
        parser.error("Run through the Stage 3G numerical lease")
    started = time.monotonic()
    try:
        result = run(args)
    except Exception as error:
        output = Path(args.output_dir).resolve()
        # Do not mask the original exception if the controller kills this process.
        if output.exists() and not (output / "failure.json").exists():
            write_new(output / "failure.json", canonical({
                "schema": "riichi-ai-stage3g-placement-failure-v1", "complete": False,
                "failure": {"type": type(error).__name__, "message": str(error)},
                "elapsed_seconds": time.monotonic() - started,
                "completed_fit_receipts": [path.name for path in sorted(output.glob("completed-*.json"))]}))
        raise
    print(json.dumps({"complete": True, "calibration_gate_passed": result["calibration_gate_passed"],
                      "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
