"""Diagnose the INITIAL frozen Stage 2 rank head on the already-opened test set.

This fits only a Laplace-smoothed 4x4 frequency table from the same first 300
training matches. It never updates a neural weight or opens auditor files.
Run with .venv/bin/python scripts/stage2_rank_diagnostic.py.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from riichi_ai.dataset import load_split
from riichi_ai.features import FEATURE_SCHEMA
from riichi_ai.model import NeuralPolicy, collate_encoded


PROJECT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT / "reports/stage2"
SCHEMA = "riichi-ai-initial-rank-diagnostic-v1"
INITIAL_CHECKPOINT_SHA256 = "a80bd9ae91dd72451279b73ebfd8efefbe26660f984ee4719149c6b170f201a8"
BIN_EDGES = np.linspace(0.0, 1.0, 11)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_rank(globals_: np.ndarray) -> int:
    """Zero-based standing from relative visible scores and initial-seat ties.

    globals[0:4] are own/next/opposite/previous scores divided by 50,000;
    globals[24:28] identify the absolute starting seat. Integer score recovery
    avoids inventing distinctions between floating-point encodings of ties.
    """
    if FEATURE_SCHEMA != "actor-features-v1" or globals_.shape != (45,):
        raise ValueError("This diagnostic requires actor-features-v1")
    if not np.isfinite(globals_).all():
        raise ValueError("Nonfinite actor features")
    identity = globals_[24:28]
    if not np.array_equal(identity, np.eye(4, dtype=identity.dtype)[identity.argmax()]):
        raise ValueError("Invalid absolute initial-seat encoding")
    seat = int(identity.argmax())
    scores = np.rint(globals_[:4].astype(np.float64) * 50_000).astype(np.int64)
    order = sorted(range(4), key=lambda relative: (-int(scores[relative]), (seat + relative) % 4))
    return order.index(0)


def metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if (probabilities.shape != (len(labels), 4) or not len(labels)
            or np.any((labels < 0) | (labels > 3))
            or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6, rtol=0)):
        raise ValueError("Invalid rank probabilities or labels")
    predicted = probabilities.argmax(axis=1)  # Fixed first-index tie rule.
    confidence = probabilities.max(axis=1)
    correct = predicted == labels
    bin_ids = np.minimum((confidence * 10).astype(np.int64), 9)
    bins = []
    ece = 0.0
    for bin_index in range(10):
        selected = bin_ids == bin_index
        count = int(selected.sum())
        mean_confidence = float(confidence[selected].mean()) if count else None
        accuracy = float(correct[selected].mean()) if count else None
        gap = abs(accuracy - mean_confidence) if count else None
        ece += 0 if count == 0 else count / len(labels) * gap
        bins.append({"lower": float(BIN_EDGES[bin_index]),
                     "upper": float(BIN_EDGES[bin_index + 1]),
                     "upper_inclusive": bin_index == 9, "count": count,
                     "mean_top_confidence": mean_confidence, "top_label_accuracy": accuracy,
                     "absolute_calibration_gap": gap})
    nll = -np.log(np.maximum(probabilities[np.arange(len(labels)), labels], np.finfo(np.float64).tiny)).mean()
    brier = ((probabilities - np.eye(4)[labels]) ** 2).sum(axis=1).mean()
    return {"rows": len(labels), "rank_nll": float(nll), "rank_brier": float(brier),
            "rank_accuracy": float(correct.mean()), "mean_top_confidence": float(confidence.mean()),
            "top_label_ece_10_fixed_bins": float(ece), "reliability": bins,
            "terminal_rank_counts": np.bincount(labels, minlength=4).tolist()}


def self_check() -> None:
    # Known score/rank/tie cases and known uniform-prediction metrics.
    for seat in range(4):
        global_features = np.zeros(45, np.float32)
        global_features[:4] = .5
        global_features[24 + seat] = 1
        assert current_rank(global_features) == seat
        absolute = np.asarray([10_000, 40_000, 20_000, 30_000])
        global_features[:4] = np.roll(absolute, -seat) / 50_000
        assert current_rank(global_features) == [3, 0, 2, 1][seat]
    uniform = metrics(np.full((4, 4), .25), np.arange(4))
    assert abs(uniform["rank_nll"] - np.log(4)) < 1e-12
    assert uniform["rank_brier"] == .75 and uniform["rank_accuracy"] == .25
    assert uniform["top_label_ece_10_fixed_bins"] == 0
    certain = metrics(np.eye(4), np.arange(4))
    assert certain["reliability"][9]["count"] == 4
    assert certain["rank_nll"] == certain["rank_brier"] == 0


def main() -> None:
    started = time.perf_counter()
    self_check()
    torch.set_num_threads(1)
    selection_path = REPORTS / "selection.json"
    selection_bytes = selection_path.read_bytes()
    selection = json.loads(selection_bytes)
    if (selection["selected"]["run"] != "soft-300"
            or selection["selected"]["training_matches"] != 300
            or selection["checkpoint_sha256"] != INITIAL_CHECKPOINT_SHA256):
        raise ValueError("This report is restricted to the INITIAL selected soft-300 checkpoint")
    checkpoint = Path(selection["checkpoint"])
    policy = NeuralPolicy.from_checkpoint(checkpoint, device="cpu")
    if policy.checkpoint_hash != INITIAL_CHECKPOINT_SHA256:
        raise ValueError("Initial frozen checkpoint bytes changed")
    plan_path = PROJECT / "configs/stage2_pilot.json"
    plan = json.loads(plan_path.read_text())
    directory = Path(plan["data_root"]) / "dataset"
    manifest_path = directory / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["rule_profile"]["match"]["tie_priority"] != "initial_seat_ascending":
        raise ValueError("Unsupported current-standing tie priority")
    training = load_split(directory, "train", max_matches=300)
    # Supervision is terminal rank, but every conditioning input is public.
    counts = np.zeros((4, 4), dtype=np.int64)
    for record in training:
        standing = current_rank(record["encoded"]["globals"])
        assert int(record["encoded"]["globals"][24:28].argmax()) == record["seat"]
        counts[standing, record["rank_target"]] += 1
    baseline = (counts + 1) / (counts.sum(axis=1, keepdims=True) + 4)
    train_ids = list(training.match_ids)
    train_rows = len(training)
    del training
    test = load_split(directory, "test")
    if set(train_ids) & set(test.match_ids):
        raise ValueError("Training and test match identities overlap")
    labels = np.empty(len(test), dtype=np.int64)
    standings = np.empty(len(test), dtype=np.int64)
    model_probabilities = np.empty((len(test), 4), dtype=np.float64)
    first: dict[tuple[str, int], tuple[int, int]] = {}
    for start in range(0, len(test), 256):
        records = [test[index] for index in range(start, min(start + 256, len(test)))]
        batch = collate_encoded([record["encoded"] for record in records], device="cpu")
        with torch.inference_mode():
            _, rank_logits = policy.model(batch)
            probabilities = rank_logits.to(torch.float64).softmax(dim=-1).numpy()
        model_probabilities[start:start + len(records)] = probabilities
        for offset, record in enumerate(records):
            index = start + offset
            labels[index] = record["rank_target"]
            standings[index] = current_rank(record["encoded"]["globals"])
            assert int(record["encoded"]["globals"][24:28].argmax()) == record["seat"]
            key = (record["match_id"], record["seat"])
            candidate = (record["step"], index)
            if key not in first or candidate < first[key]:
                first[key] = candidate
    if len(first) != 4 * len(test.match_ids):
        raise ValueError("First-decision subset does not cover every match and seat")
    first_indices = np.asarray(sorted(index for _, index in first.values()), dtype=np.int64)
    baseline_probabilities = baseline[standings]
    comparisons = {}
    for name, indices in (("all_decisions", np.arange(len(test))),
                          ("first_decision_per_match_seat", first_indices)):
        model_metrics = metrics(model_probabilities[indices], labels[indices])
        baseline_metrics = metrics(baseline_probabilities[indices], labels[indices])
        uniform_metrics = metrics(np.full((len(indices), 4), .25), labels[indices])
        comparisons[name] = {
            "rows": len(indices), "independent_matches": len(test.match_ids),
            "current_rank_counts": np.bincount(standings[indices], minlength=4).tolist(),
            "model": model_metrics, "current_rank_frequency_baseline": baseline_metrics,
            "uniform_baseline": uniform_metrics,
            "model_improvement_over_current_rank_baseline": {
                "nll_reduction": baseline_metrics["rank_nll"] - model_metrics["rank_nll"],
                "brier_reduction": baseline_metrics["rank_brier"] - model_metrics["rank_brier"],
                "accuracy_difference": model_metrics["rank_accuracy"] - baseline_metrics["rank_accuracy"],
            },
        }
    source_paths = [Path(__file__).resolve()] + [PROJECT / "src/riichi_ai" / (name + ".py")
                                             for name in ("features", "model", "dataset", "rules")]
    if (digest(checkpoint) != INITIAL_CHECKPOINT_SHA256
            or manifest_path.read_bytes() != manifest_bytes or selection_path.read_bytes() != selection_bytes):
        raise ValueError("Frozen checkpoint or input manifests changed during diagnosis")
    report = {
        "schema": SCHEMA, "scope": "Post hoc descriptive diagnosis of the initial frozen soft-300 rank head only.",
        "checkpoint": str(checkpoint), "checkpoint_sha256": policy.checkpoint_hash,
        "model_weights_sha256": policy.weights_hash, "device": "cpu", "cpu_threads": 1,
        "neural_training_performed": False, "auditor_files_opened": False,
        "baseline": {"conditioning": "Current visible score standing, with absolute initial-seat ascending tie priority.",
                     "rank_index_order": [1, 2, 3, 4], "laplace_pseudocount_per_cell": 1,
                     "training_matches": 300, "training_rows": train_rows,
                     "counts_current_rank_by_terminal_rank": counts.tolist(),
                     "probabilities_terminal_rank_given_current_rank": baseline.tolist(),
                     "fitting_weight": "Each recorded decision has equal weight, including forced decisions.",
                     "first_decision_use": "The same all-decision-fitted table is used without refitting on this early-game subset."},
        "test": {"matches": len(test.match_ids), "rows": len(test),
                 "match_ids": list(test.match_ids), "first_decision_rows": len(first_indices),
                 "first_decision_selection": "Minimum recorded (step, row index) for each (match, absolute seat).",
                 "first_decision_indices_sha256": hashlib.sha256(first_indices.astype("<i8").tobytes()).hexdigest()},
        "metrics": comparisons,
        "calibration_definition": "Top-label ECE: sum over ten fixed confidence bins of (bin_count/N)*abs(bin_accuracy-bin_mean_confidence). Bins are [0,.1),...,[.9,1]. Argmax ties select the first rank index. Brier is the sum over four classes.",
        "limitations": [
            "Test outcomes were already opened for the frozen initial model; this is post hoc diagnosis, not a new untouched confirmation set.",
            "All-decision metrics overweight longer matches and repeatedly count the same terminal outcomes; rows are not independent.",
            "The first-decision subset has four correlated predictions per match and only 100 independent matches; no row-independent confidence interval or significance claim is made.",
            "The four-by-four baseline discards score gaps, game progress, tiles and history. Beating it would not by itself prove foresight beyond richer public-score baselines.",
            "First-decision baseline probabilities are fitted to all game stages and can be poorly calibrated early; their score is not a calibrated early-game reference.",
            "Top-label ECE is a finite-sample, bin-dependent summary and can hide class-conditional miscalibration; lower ECE is not proof of full calibration.",
            "Rank labels describe the synthetic teacher policy and its opponents, not outcomes after deploying this neural policy or changing strategy.",
        ],
        "provenance": {"feature_schema": FEATURE_SCHEMA,
                       "selection_manifest_sha256": hashlib.sha256(selection_bytes).hexdigest(),
                       "dataset_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                       "dataset_plan_sha256": manifest["plan_sha256"],
                       "training_match_ids": train_ids,
                       "loader_verified_actor_and_target_shard_hashes": True,
                       "plan_sha256": digest(plan_path),
                       "sources": {str(path.relative_to(PROJECT)): digest(path) for path in source_paths},
                       "python": platform.python_version(), "torch": str(torch.__version__),
                       "numpy": str(np.__version__)},
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = REPORTS / "rank-diagnostic.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"report": str(output), "elapsed_seconds": report["elapsed_seconds"],
                      "comparisons": {name: {"model_nll": result["model"]["rank_nll"],
                                             "baseline_nll": result["current_rank_frequency_baseline"]["rank_nll"],
                                             "model_ece": result["model"]["top_label_ece_10_fixed_bins"],
                                             "baseline_ece": result["current_rank_frequency_baseline"]["top_label_ece_10_fixed_bins"]}
                                      for name, result in comparisons.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
