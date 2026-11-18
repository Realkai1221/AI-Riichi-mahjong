"""Descriptive summaries of saved Stage 3C data; no inference or new games.

Run inside the root Stage 3C numerical lease after all fixed games finish.
This script imports only the standard library and never loads a checkpoint.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CALL_KINDS = ("CHI", "PON", "DAIMINKAN", "ANKAN", "KAKAN")
WIN_KINDS = ("RON", "TSUMO")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_numbers(values) -> dict[str, Any]:
    values = list(values)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("A descriptive population contains non-finite values")
    return {"count": len(values), "mean": math.fsum(values) / len(values) if values else None,
            "minimum": min(values) if values else None, "maximum": max(values) if values else None}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if any(not record["completed"] or not record["terminal_result_verified"] for record in records):
        raise ValueError("Only complete audited match outcomes may enter the final summary")
    rows = [row for record in records for row in record["learner_rows"]]
    calls = {}
    for kind in CALL_KINDS:
        opportunities = [row for row in rows if kind in row["action_kinds"]]
        calls[kind] = {
            "opportunity_rows": len(opportunities),
            "candidate_count": sum(row["action_kinds"].count(kind) for row in opportunities),
            "requested": sum(row["selected_kind"] == kind for row in rows),
            "declared": sum(record["declared_calls"][kind] for record in records),
            "resolved": sum(record["resolved_calls"][kind] for record in records),
            "matches_with_request": sum(record["requested_calls"][kind] > 0 for record in records),
            "matches_with_resolved_call": sum(record["resolved_calls"][kind] > 0 for record in records),
            "gate_mass_on_opportunity_rows": summarize_numbers(row["gate_mass"] for row in opportunities),
            "total_call_mass_on_opportunity_rows": summarize_numbers(row["call_mass"] for row in opportunities),
        }
        if not 0 <= calls[kind]["resolved"] <= calls[kind]["declared"] <= calls[kind]["requested"] <= calls[kind]["opportunity_rows"]:
            raise ValueError(f"Inconsistent opportunity/request/declaration/resolution counts for {kind}")
        if calls[kind]["requested"] != sum(record["requested_calls"][kind] for record in records):
            raise ValueError("Saved match and row request summaries disagree")
    call_rows = [row for row in rows if row["call_candidates"] > 0]
    multiple_calls = [row for row in call_rows if row["call_candidates"] > 1]
    mixed_rows = [row for row in call_rows if row["call_candidates"] < len(row["action_kinds"])]
    only_call_rows = [row for row in call_rows if row["call_candidates"] == len(row["action_kinds"])]
    if any(row["call_candidates"] != sum(kind in CALL_KINDS for kind in row["action_kinds"]) for row in rows):
        raise ValueError("Saved call-group membership disagrees with candidate kinds")
    win_rows = [row for row in rows if any(kind in WIN_KINDS for kind in row["action_kinds"])]
    competition = [row for row in win_rows if row["call_candidates"] > 0]
    post_call = [row for row in rows if row["post_call_state"]]
    open_hand = [row for row in rows if row["open_hand_state"]]
    if len(post_call) != sum(record["post_call_rows"] for record in records) or len(open_hand) != sum(record["open_hand_rows"] for record in records):
        raise ValueError("Saved match and row post-call summaries disagree")
    outcomes = [record["ranks"][record["learner_seat"]] for record in records]
    categories = {"any_call": call_rows, "single_call": [r for r in call_rows if r["call_candidates"] == 1],
                  "multiple_calls": multiple_calls, "mixed_call_and_noncall": mixed_rows,
                  "only_calls": only_call_rows, "win_competition": competition,
                  "open_hand_with_call": [r for r in call_rows if r["open_hand_state"]]}
    return {
        "matches": len(records), "hands": sum(record["hands"] for record in records),
        "engine_steps": sum(record["steps"] for record in records), "learner_rows": len(rows),
        "optional_rows": sum(not row["forced"] for row in rows), "forced_rows": sum(row["forced"] for row in rows),
        "selected_action_kinds": dict(sorted(Counter(row["selected_kind"] for row in rows).items())),
        "call_kinds": calls, "call_opportunity_rows": len(call_rows),
        "multiple_call_opportunity_rows": len(multiple_calls),
        "multiple_call_selections": sum(row["selected_kind"] in CALL_KINDS for row in multiple_calls),
        "gate_mass_categories": {name: summarize_numbers(row["gate_mass"] for row in population)
                                 for name, population in categories.items()},
        "post_call_rows": len(post_call), "post_call_matches": sum(record["post_call_rows"] > 0 for record in records),
        "post_call_optional_rows": sum(not row["forced"] for row in post_call),
        "open_hand_rows": len(open_hand), "open_hand_matches": sum(record["open_hand_rows"] > 0 for record in records),
        "post_call_hands": sum(len({row["hand_id"] for row in record["learner_rows"] if row["post_call_state"]}) for record in records),
        "legal_win_rows": len(win_rows),
        "legal_win_nonwin_selections": sum(row["selected_kind"] not in WIN_KINDS for row in win_rows),
        "win_call_competition": {
            "rows": len(competition), "available_ron_rows": sum("RON" in row["action_kinds"] for row in competition),
            "available_tsumo_rows": sum("TSUMO" in row["action_kinds"] for row in competition),
            "selected_win": sum(row["selected_kind"] in WIN_KINDS for row in competition),
            "selected_nonwin": sum(row["selected_kind"] not in WIN_KINDS for row in competition),
            "selected_call": sum(row["selected_kind"] in CALL_KINDS for row in competition),
            "selected_call_resolved": sum(row["selected_kind"] in CALL_KINDS and row["requested_call_resolved"] is True for row in competition),
            "win_mass_without_gate": summarize_numbers(row["win_mass_without_gate"] for row in competition),
            "actual_win_mass": summarize_numbers(row["win_mass_actual"] for row in competition),
            "win_mass_lost": summarize_numbers(row["win_mass_lost"] for row in competition),
            "sum_win_mass_lost": math.fsum(row["win_mass_lost"] for row in competition),
        },
        "sum_win_probability_mass_lost_all_rows": math.fsum(row["win_mass_lost"] for row in rows),
        "placement_counts": {str(rank): outcomes.count(rank) for rank in range(1, 5)},
        "mean_rank": math.fsum(outcomes) / len(outcomes) if outcomes else None,
        "mean_utility": math.fsum(record["learner_utility"] for record in records) / len(records) if records else None,
        "note": "Descriptive sampled populations; opportunity categories overlap. No causal or strength inference."
    }


def action_sequence(record):
    return [{seat: choice["physical_token"] for seat, choice in sorted(request["choices"].items())}
            for request in record["requests"]]


def compare_pair(candidate, baseline, block, learner_seat):
    if (candidate["seed"] != baseline["seed"] or candidate["sampling_seed"] != baseline["sampling_seed"]
            or candidate["learner_seat"] != learner_seat or baseline["learner_seat"] != learner_seat):
        raise ValueError("Pair does not share the prescribed wall/sampling seeds and learner seat")
    first, second = action_sequence(candidate), action_sequence(baseline)
    shared = 0
    for a, b in zip(first, second):
        if a != b:
            break
        shared += 1
    identical = first == second
    same_scores = candidate["scores"] == baseline["scores"]
    same_ranks = candidate["ranks"] == baseline["ranks"]
    if identical and not (same_scores and same_ranks):
        raise ValueError("Identical paired action histories yielded different terminal results")
    return {"block": block, "learner_seat": learner_seat,
        "candidate_match_index": candidate["match_index"], "baseline_match_index": baseline["match_index"],
        "identical_action_sequence": identical, "identical_all_seat_scores": same_scores,
        "identical_all_seat_placements": same_ranks,
        "same_learner_placement": candidate["ranks"][learner_seat] == baseline["ranks"][learner_seat],
        "common_action_prefix_steps": shared,
        "first_divergent_step": None if identical else shared,
        "candidate_actions_at_first_difference": None if identical or shared >= len(first) else first[shared],
        "baseline_actions_at_first_difference": None if identical or shared >= len(second) else second[shared],
        "learner_utility_difference": candidate["learner_utility"] - baseline["learner_utility"]}


def analyze(config_path: Path, output_path: Path, *, allow_incomplete: bool = False) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    config_hash = digest(config_path)
    reports = (ROOT / config["reports_root"]).resolve()
    data_root = (ROOT / config["data_root"]).resolve()
    inputs, seen = [], set()

    def read_report(path, require_complete=True):
        payload = json.loads(path.read_text())
        inputs.append({"path": str(path), "sha256": digest(path)})
        if payload.get("config_sha256") != config_hash or (require_complete and payload.get("complete") is not True):
            raise ValueError("Final summary needs complete reports from the fixed configuration")
        return payload

    def read_match(evidence):
        path = Path(evidence["json_path"]).resolve()
        if not path.is_relative_to(data_root) or path in seen or digest(path) != evidence["json_sha256"]:
            raise ValueError("Match path/hash/uniqueness does not match saved evidence")
        seen.add(path)
        record = json.loads(path.read_text())
        if record["protocol_sha256"] != config_hash:
            raise ValueError("Match configuration identity changed")
        inputs.append({"path": str(path), "sha256": evidence["json_sha256"]})
        return record

    training, batches, updates = [], {}, []
    for index in range(4):
        batch = read_report(reports / f"batch-{index}.json")
        records = [read_match(evidence) for evidence in batch["matches"]]
        if len(records) != 16 or any(record["collection_role"] != "training" or not record["training_eligible"] for record in records):
            raise ValueError("Expected the four complete, eligible 16-match training batches")
        training.extend(records)
        batches[str(index)] = summarize(records)
        update = batch["update"]
        updates.append({key: update[key] for key in ("batch_index", "gradient_norm_before_clip", "gradient_components",
            "parameter_component_l2_change", "parameter_max_absolute_change", "parameter_l2_change",
            "gradient_eligible_rows", "base_unchanged", "rank_head_unchanged", "distribution_mean_l1_change",
            "distribution_max_absolute_change", "probability_replay_max_absolute_error",
            "selected_log_probability_replay_max_error", "loss_numerical_difference")})
    arms = {"candidate": [], "baseline": []}
    paired, partial_blocks, complete_blocks = [], [], []
    for block in range(32):
        report_path = reports / f"eval-{block}.json"
        attempt_path = reports / f"eval-attempt-{block}.json"
        if not report_path.exists():
            if not allow_incomplete:
                raise ValueError("Missing evaluation block; fixed endpoint is incomplete")
            if attempt_path.exists():
                attempt = json.loads(attempt_path.read_text())
                inputs.append({"path": str(attempt_path), "sha256": digest(attempt_path)})
                partial_blocks.append({"block": block, "attempt_status": attempt.get("status"),
                                       "failure": attempt.get("failure"), "persisted_matches": 0,
                                       "excluded_from_paired_comparison": True})
            break
        report = read_report(report_path, require_complete=False)
        attempt = json.loads(attempt_path.read_text()) if attempt_path.exists() else {}
        if attempt_path.exists():
            inputs.append({"path": str(attempt_path), "sha256": digest(attempt_path)})
        valid_block = report.get("complete") is True and attempt.get("status") == "completed"
        if not valid_block and not allow_incomplete:
            raise ValueError("Incomplete or failed evaluation block; fixed endpoint unavailable")
        lookup, partial_records = {}, {"candidate": [], "baseline": []}
        incomplete_match_count = 0
        for evidence in report["matches"]:
            record = read_match(evidence)
            if record["collection_role"] != "evaluation" or record["training_eligible"]:
                raise ValueError("Evaluation match is eligible for training")
            key = evidence["arm"], evidence["learner_seat"]
            if key in lookup:
                raise ValueError("Duplicate evaluation arm/seat")
            lookup[key] = record
            if valid_block:
                arms[evidence["arm"]].append(record)
            elif record["completed"] and record["terminal_result_verified"]:
                partial_records[evidence["arm"]].append(record)
            else:
                incomplete_match_count += 1
        if not valid_block:
            partial_blocks.append({"block": block, "attempt_status": attempt.get("status"),
                "failure": attempt.get("failure") or report.get("failure"),
                "persisted_matches": len(lookup), "incomplete_matches": incomplete_match_count,
                "completed_matches_in_incomplete_block": {arm: summarize(records) for arm, records in partial_records.items()},
                "excluded_from_paired_comparison": True})
            break
        if set(lookup) != {(arm, seat) for arm in arms for seat in range(4)}:
            raise ValueError("Each complete block requires both arms at all four learner seats")
        complete_blocks.append(block)
        for seat in range(4):
            paired.append(compare_pair(lookup["candidate", seat], lookup["baseline", seat], block, seat))
    complete_experiment = len(complete_blocks) == 32 and not partial_blocks
    result = {"schema": "stage3c-saved-data-descriptive-behavior-v1", "valid": True,
        "config_path": str(config_path.resolve()), "config_sha256": config_hash,
        "analysis_source_sha256": digest(Path(__file__).resolve()),
        "models_loaded": 0, "new_games": 0, "new_optimizer_updates": 0,
        "input_match_count": len(seen), "input_files": inputs,
        "complete_experiment": complete_experiment, "fixed_endpoint_available": complete_experiment,
        "complete_evaluation_blocks": complete_blocks, "partial_evaluation_blocks": partial_blocks,
        "planned_evaluation_blocks": 32, "planned_total_matches": 320,
        "inferential_comparison_computed": False,
        "evaluation_summary_scope": "Only complete blocks whose operation marker completed; partial blocks kept separate.",
        "training": summarize(training), "training_by_batch": batches, "training_updates": updates,
        "evaluation_by_arm": {arm: summarize(records) for arm, records in arms.items()},
        "paired_behavior": {"pairs": len(paired),
            "identical_action_sequences": sum(pair["identical_action_sequence"] for pair in paired),
            "divergent_action_sequences": sum(not pair["identical_action_sequence"] for pair in paired),
            "identical_all_seat_scores": sum(pair["identical_all_seat_scores"] for pair in paired),
            "identical_all_seat_placements": sum(pair["identical_all_seat_placements"] for pair in paired),
            "same_learner_placement": sum(pair["same_learner_placement"] for pair in paired), "details": paired},
        "interpretation": ("Post-run descriptive summary only; use only the separate preregistered fixed-sample result for outcome uncertainty."
                           if complete_experiment else
                           "INCOMPLETE experiment: only completed blocks summarized descriptively. No fixed endpoint, confidence interval, strength conclusion, or inferential comparison is available. No failed block is replaced.")}
    if complete_experiment and result["input_match_count"] != 320:
        raise ValueError("Fixed experiment must contain exactly 320 distinct matches")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage3c.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/stage3c/behavior.json")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Preserve incomplete evaluation separately; never compute a fixed endpoint or interval")
    arguments = parser.parse_args()
    result = analyze(arguments.config, arguments.output, allow_incomplete=arguments.allow_incomplete)
    print(json.dumps({"valid": result["valid"], "input_matches": result["input_match_count"],
                      "output": str(arguments.output.resolve()),
                      "paired_identical_action_sequences": result["paired_behavior"]["identical_action_sequences"]}))


if __name__ == "__main__":
    main()
