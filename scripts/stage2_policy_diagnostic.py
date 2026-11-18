"""CPU-only descriptive error audit of the frozen extended student's validation.

Never opens test or auditor data, changes a neural weight, or fits a model.
Teacher preferences are consulted only after encoded-only model inference.
"""

from __future__ import annotations

from collections import Counter
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
from riichi_ai.semantics import ACTION_KINDS


PROJECT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT / "reports/stage2"
CHECKPOINT_SHA256 = "09d42bf7c2b881f57d76f4ed38b33fa04070a90c7294d183338e5d0a30d1537d"
COMPONENTS = ("branch", "danger", "shanten", "negative_ukeire", "red")
DISCARD_ID = ACTION_KINDS.index("DISCARD") + 1


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_difference(best: np.ndarray, selected: np.ndarray) -> str:
    different = np.flatnonzero(best != selected)
    if not len(different):
        raise ValueError("An allegedly incorrect action has the best preference key")
    index = int(different[0])
    if selected[index] <= best[index]:
        raise ValueError("Selected action unexpectedly improves the minimum teacher key")
    return COMPONENTS[index]


def severity(values: list[int]) -> dict:
    array = np.asarray(values, dtype=np.int64)
    return {"count": len(values), "total": int(array.sum()),
            "mean": float(array.mean()) if len(values) else None,
            "median": float(np.median(array)) if len(values) else None,
            "maximum": int(array.max()) if len(values) else None,
            "histogram": {str(value): count for value, count in sorted(Counter(values).items())}}


def tile_label(kind: int) -> str:
    return f"{kind % 9 + 1}{'mps'[kind // 9]}" if kind < 27 else f"{kind - 26}z"


def self_check() -> None:
    reference = np.asarray([2, 0, 1, -12, 0])
    for index, component in enumerate(COMPONENTS):
        changed = reference.copy()
        changed[index] += 1
        assert first_difference(reference, changed) == component
    assert severity([1, 1, 2])["histogram"] == {"1": 2, "2": 1}
    assert [tile_label(index) for index in (0, 8, 9, 17, 18, 26, 27, 33)] == [
        "1m", "9m", "1p", "9p", "1s", "9s", "1z", "7z"]


def main() -> None:
    started = time.perf_counter()
    self_check()
    torch.set_num_threads(1)
    selection_path = REPORTS / "extended-selection.json"
    selection_bytes = selection_path.read_bytes()
    selection = json.loads(selection_bytes)
    if selection["checkpoint_sha256"] != CHECKPOINT_SHA256 or selection["best_step"] != 20_000:
        raise ValueError("This diagnostic requires the frozen extended 20,000-update checkpoint")
    checkpoint = Path(selection["checkpoint"])
    policy = NeuralPolicy.from_checkpoint(checkpoint, device="cpu")
    if policy.checkpoint_hash != CHECKPOINT_SHA256:
        raise ValueError("Extended frozen checkpoint bytes changed")
    plan_path = PROJECT / "configs/stage2_pilot.json"
    plan = json.loads(plan_path.read_text())
    directory = Path(plan["data_root"]) / "dataset"
    manifest_path = directory / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    validation = load_split(directory, "validation")
    if len(validation.match_ids) != 100 or validation.split != "validation":
        raise ValueError("Expected the complete 100-match validation split")
    source_paths = [Path(__file__).resolve()] + [PROJECT / "src/riichi_ai" / (name + ".py")
                                              for name in ("features", "model", "dataset", "semantics")]
    initial_sources = {str(path.relative_to(PROJECT)): digest(path) for path in source_paths}

    # Complete all predictions using only encoder fields before inspecting
    # candidate preference scores. Forced choices need no network prediction.
    predictions = np.empty(len(validation), dtype=np.int64)
    prediction_digest = hashlib.sha256()
    network_rows = forced_rows = 0
    inference_started = time.perf_counter()
    for start in range(0, len(validation), 256):
        records = [validation[index] for index in range(start, min(start + 256, len(validation)))]
        optional = [offset for offset, record in enumerate(records) if record["optional"]]
        for offset, record in enumerate(records):
            if not record["optional"]:
                predictions[start + offset] = 0
                forced_rows += 1
        if optional:
            batch = collate_encoded([records[offset]["encoded"] for offset in optional], device="cpu")
            with torch.inference_mode():
                logits, _ = policy.model(batch)
                if not bool(torch.isfinite(logits[batch["candidate_mask"]]).all()):
                    raise ValueError("Nonfinite policy logits")
                selected = logits.argmax(dim=-1).numpy()
            for offset, chosen in zip(optional, selected):
                predictions[start + offset] = int(chosen)
            network_rows += len(optional)
    inference_seconds = time.perf_counter() - inference_started
    prediction_digest.update(predictions.astype("<i8").tobytes())

    first_counts = Counter({name: 0 for name in COMPONENTS})
    error_kinds: Counter[str] = Counter()
    shanten_increments: list[int] = []
    ukeire_losses: list[int] = []
    per_student_tile = {kind: {"tile_type": kind, "tile": tile_label(kind), "selected_rows": 0,
                              "incorrect_rows": 0, "first_difference_counts": Counter(),
                              "worse_shanten_increments": [], "same_shanten_ukeire_losses": []}
                        for kind in range(34)}
    per_teacher_tile: Counter[int] = Counter()
    per_match: dict[str, dict] = {}
    examples = []
    optional_correct = discard_rows = discard_correct = discard_exact = 0
    both_discard_rows = same_shanten_rows = 0
    for index, sample in enumerate(validation):
        if not sample["optional"]:
            continue
        chosen = int(predictions[index])
        actions = sample["encoded"]["action_tokens"]
        if not 0 <= chosen < len(actions):
            raise ValueError("Model selected a padded/nonlegal candidate")
        teacher_chosen = sample["chosen_index"]
        correct = bool(sample["best_mask"][chosen])
        optional_correct += int(correct)
        if int(actions[teacher_chosen, 0]) != DISCARD_ID:
            continue
        discard_rows += 1
        discard_correct += int(correct)
        discard_exact += int(chosen == teacher_chosen)
        # The actor loader already verified this target shard's SHA-256. Its
        # preference rows are diagnostic labels and never enter model(batch).
        match_index, row = map(int, validation.index[index])
        match_id = validation.match_ids[match_index]
        actor, targets = validation._load(match_id)
        start, end = map(int, actor["candidate_offsets"][row:row + 2])
        preferences = targets["teacher_preference_keys"][start:end]
        best_key, chosen_key = preferences[teacher_chosen], preferences[chosen]
        match = per_match.setdefault(match_id, {"match_id": match_id, "optional_teacher_discard_rows": 0,
                                               "incorrect_rows": 0, "worse_shanten_rows": 0,
                                               "same_shanten_lost_ukeire_rows": 0})
        match["optional_teacher_discard_rows"] += 1
        student_kind = ACTION_KINDS[int(actions[chosen, 0]) - 1]
        tile = None
        if student_kind == "DISCARD":
            kind = int(actions[chosen, 1]) - 1
            tile = per_student_tile[kind]
            tile["selected_rows"] += 1
            both_discard_rows += 1
            same_shanten_rows += int(chosen_key[2] == best_key[2])
        if correct:
            if not np.array_equal(best_key, chosen_key):
                raise ValueError("Best-mask membership differs from the recorded preference key")
            continue
        component = first_difference(best_key, chosen_key)
        first_counts[component] += 1
        error_kinds[student_kind] += 1
        per_teacher_tile[int(actions[teacher_chosen, 1]) - 1] += 1
        match["incorrect_rows"] += 1
        shanten_delta = lost_ukeire = None
        if tile is not None:
            tile["incorrect_rows"] += 1
            tile["first_difference_counts"][component] += 1
            shanten_delta = int(chosen_key[2]) - int(best_key[2])
            if shanten_delta > 0:
                shanten_increments.append(shanten_delta)
                tile["worse_shanten_increments"].append(shanten_delta)
                match["worse_shanten_rows"] += 1
            if shanten_delta == 0:
                lost_ukeire = int(chosen_key[3]) - int(best_key[3])
                if lost_ukeire > 0:
                    ukeire_losses.append(lost_ukeire)
                    tile["same_shanten_ukeire_losses"].append(lost_ukeire)
                    match["same_shanten_lost_ukeire_rows"] += 1
        examples.append({"validation_row": index, "match_id": match_id, "seat": sample["seat"],
                         "step": sample["step"], "hand_id": sample["hand_id"], "first_difference": component,
                         "teacher_preference": best_key.tolist(), "selected_preference": chosen_key.tolist(),
                         "teacher_canonical_candidate": actions[teacher_chosen].tolist(),
                         "selected_candidate": actions[chosen].tolist(),
                         "shanten_increment": shanten_delta, "same_shanten_ukeire_loss": lost_ukeire})

    incorrect = discard_rows - discard_correct
    if sum(first_counts.values()) != incorrect or incorrect != len(examples):
        raise ValueError("Every incorrect optional teacher-discard choice must have exactly one diagnosis")
    examples.sort(key=lambda item: (-(item["shanten_increment"] or 0),
                                    -(item["same_shanten_ukeire_loss"] or 0), item["validation_row"]))
    tiles = []
    for kind in range(34):
        item = per_student_tile[kind]
        item["incorrect_fraction"] = item["incorrect_rows"] / item["selected_rows"] if item["selected_rows"] else None
        item["first_difference_counts"] = {name: item["first_difference_counts"][name] for name in COMPONENTS}
        item["worse_shanten"] = severity(item.pop("worse_shanten_increments"))
        item["same_shanten_lost_ukeire"] = severity(item.pop("same_shanten_ukeire_losses"))
        tiles.append(item)

    training_path = REPORTS / "extended-training.json"
    full_validation_path = REPORTS / "extended-validation.json"
    training_bytes, metrics_bytes = training_path.read_bytes(), full_validation_path.read_bytes()
    training, full_validation = json.loads(training_bytes), json.loads(metrics_bytes)
    if full_validation["checkpoint_sha256"] != CHECKPOINT_SHA256 or training["best_step"] != selection["best_step"]:
        raise ValueError("Training/validation reports describe another checkpoint")
    last = training["history"][-250:]
    means = {name: float(np.mean([row[name] for row in last])) for name in ("policy_kl", "policy_ce", "rank_ce")}
    actual_tie = discard_correct / discard_rows
    recorded_tie = full_validation["metrics"]["per_action_kind"]["DISCARD"]["tie"]
    if abs(actual_tie - recorded_tie) > 1e-4:
        raise ValueError("CPU diagnostic choices materially differ from reported full-validation discard agreement")
    if (digest(checkpoint) != CHECKPOINT_SHA256 or selection_path.read_bytes() != selection_bytes
            or manifest_path.read_bytes() != manifest_bytes
            or initial_sources != {str(path.relative_to(PROJECT)): digest(path) for path in source_paths}):
        raise ValueError("Diagnostic inputs or implementation changed")
    result = {
        "schema": "riichi-ai-extended-policy-diagnostic-v1",
        "scope": "Post hoc descriptive error severity on full validation only; no new optimization or test read.",
        "checkpoint": str(checkpoint), "checkpoint_sha256": CHECKPOINT_SHA256,
        "model_weights_sha256": policy.weights_hash, "device": "cpu", "cpu_threads": 1,
        "neural_training_performed": False, "test_split_opened": False, "auditor_files_opened": False,
        "validation": {"matches": len(validation.match_ids), "rows": len(validation), "match_ids": list(validation.match_ids),
                       "optional_rows": network_rows, "forced_rows": forced_rows,
                       "optional_tie_agreement": optional_correct / network_rows},
        "teacher_discard_choices": {"optional_rows": discard_rows, "tie_correct_rows": discard_correct,
                                    "exact_canonical_choice_rows": discard_exact,
                                    "tie_agreement": actual_tie, "incorrect_rows": incorrect,
                                    "incorrect_fraction": incorrect / discard_rows,
                                    "cpu_minus_recorded_mps_tie_agreement": actual_tie - recorded_tie,
                                    "first_difference_counts": dict(first_counts),
                                    "first_difference_fraction_of_errors": {name: count / incorrect for name, count in first_counts.items()},
                                    "incorrect_selected_action_kinds": dict(error_kinds)},
        "severity": {"both_teacher_and_student_discard_rows": both_discard_rows,
                     "worse_shanten": {**severity(shanten_increments),
                                        "fraction_of_teacher_discard_rows": len(shanten_increments) / discard_rows,
                                        "fraction_of_discard_errors": len(shanten_increments) / incorrect},
                     "same_shanten_rows_including_correct": same_shanten_rows,
                     "same_shanten_lost_ukeire": {**severity(ukeire_losses),
                                                 "fraction_of_teacher_discard_rows": len(ukeire_losses) / discard_rows,
                                                 "fraction_of_discard_errors": len(ukeire_losses) / incorrect}},
        "by_student_discard_tile_type": tiles,
        "incorrect_by_canonical_teacher_discard_tile_type": [
            {"tile_type": kind, "tile": tile_label(kind), "incorrect_rows": per_teacher_tile[kind]} for kind in range(34)],
        "by_validation_match": list(per_match.values()),
        "largest_shanten_then_ukeire_error_examples": examples[:20],
        "preference_definition": {"order": list(COMPONENTS), "direction": "Lexicographically smaller is better.",
                                  "reference": "The deterministic canonical teacher action; all best semantic actions share its preference key.",
                                  "classification": "First differing component for every optional teacher-DISCARD argmax outside the complete best-mask set.",
                                  "shanten": "Selected discard's post-discard shanten minus teacher-best post-discard shanten.",
                                  "lost_ukeire": "For equal-shanten discards, teacher-best visible ukeire minus selected visible ukeire; not actual hidden-wall availability."},
        "recent_training_vs_validation": {"steps": [last[0]["step"], last[-1]["step"]], "minibatches": len(last),
                                          "unweighted_minibatch_means": means,
                                          "full_validation_policy_kl": full_validation["metrics"]["policy_kl"],
                                          "validation_minus_recent_training_kl": full_validation["metrics"]["policy_kl"] - means["policy_kl"],
                                          "limitation": "Recent minibatch means average changing weights and different random samples; this is not a frozen-model train/validation generalization estimate."},
        "limitations": [
            "Validation was used for checkpoint selection; this diagnosis is exploratory and supplies no untouched-test inference.",
            "Rows are correlated and decision-weighted; there are 100 independent matches, not tens of thousands of independent trials.",
            "Teacher-distribution mistakes can compound after deployment changes the visited states; this audit does not measure those off-distribution states.",
            "Shanten and visible ukeire are the heuristic teacher's shape criteria, not calibrated point/rank regret or actual deal-in risk.",
            "No counterfactual simulation was run, so this does not causally assign the observed match-rank deficit to a specific error category.",
            "Teacher tile counts use its deterministic representative among tied best types; another equally best type may differ.",
        ],
        "provenance": {"feature_schema": FEATURE_SCHEMA, "source_sha256": initial_sources,
                       "selection_manifest_sha256": hashlib.sha256(selection_bytes).hexdigest(),
                       "dataset_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                       "dataset_plan_sha256": manifest["plan_sha256"], "campaign_plan_sha256": digest(plan_path),
                       "extended_training_report_sha256": hashlib.sha256(training_bytes).hexdigest(),
                       "extended_validation_report_sha256": hashlib.sha256(metrics_bytes).hexdigest(),
                       "loader_verified_actor_and_target_shard_hashes": True,
                       "ordered_argmax_indices_sha256": prediction_digest.hexdigest(),
                       "python": platform.python_version(), "torch": str(torch.__version__), "numpy": str(np.__version__)},
        "inference_seconds": inference_seconds, "elapsed_seconds": time.perf_counter() - started,
    }
    output = REPORTS / "policy-diagnostic.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"report": str(output), "teacher_discard_choices": result["teacher_discard_choices"],
                      "severity": result["severity"], "recent_training_vs_validation": result["recent_training_vs_validation"],
                      "elapsed_seconds": result["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
