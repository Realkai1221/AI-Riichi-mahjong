"""Descriptive saved-data analysis of the complete Stage 3D recovery only.

No models, forward passes, new games, training, tuning or additional inference.
The separate independent audit remains responsible for full trace validation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from analyze_stage3c import compare_pair, digest, summarize, summarize_numbers

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("candidate", "baseline")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_closed_evaluation_budget(receipt, campaign):
    require(receipt.get("name") == "eval:31" and receipt.get("config_sha256") == campaign
            and receipt.get("status") == "completed" and receipt.get("failure") is None,
            "Final evaluation receipt is not closed successfully")
    ledger = receipt["budget"]
    operations = [op for op in ledger["operations"] if op["name"].startswith("eval:")]
    require(ledger["identity"] == campaign
            and [op["name"] for op in operations] == [f"eval:{i}" for i in range(32)]
            and all(op["status"] == "completed" and op["failure"] is None for op in operations),
            "Fixed evaluation operations are incomplete, failed or duplicated")


def validate_closed_block(block, receipt, *, index, campaign, plan_sha, planned, states, implementation):
    require(block.get("complete") is True and receipt.get("status") == "completed"
            and receipt.get("failure") is None, "All 32 complete blocks need closed successful receipts")
    require(block.get("schema") == "stage3d-evaluation-block-v1"
            and block.get("block") == index and block.get("config_sha256") == campaign
            and block.get("plan_sha256") == plan_sha and block.get("planned") == planned
            and block.get("states_before") == states and block.get("states_after") == states
            and block.get("implementation") == implementation
            and block.get("termination_requested") is False,
            "Block identity or fixed model states differ")
    require(receipt.get("name") == f"eval:{index}" and receipt.get("config_sha256") == campaign
            and receipt.get("implementation") == implementation
            and receipt.get("finished_unix", -1) >= receipt.get("started_unix", 0),
            "Closed receipt provenance differs")
    require(len(block.get("matches", [])) == 8, "Each block requires exactly eight matches")


def validate_record(record, *, campaign, behavior, plan_sha, block, arm, planned, state):
    require(record.get("completed") is True and record.get("terminated") is True
            and record.get("terminal_result_verified") is True and record.get("failure") is None
            and record.get("truncated") is False and record.get("interrupted") is False,
            "Only complete uninterrupted audited outcomes may enter this summary")
    require(record.get("collection_role") == "evaluation" and record.get("training_eligible") is False
            and all(row.get("training_eligible") is False for row in record["learner_rows"]),
            "Training-eligible data cannot enter recovery analysis")
    require(record.get("campaign_sha256") == campaign and record.get("protocol_sha256") == behavior
            and record.get("campaign_plan_sha256") == plan_sha
            and record.get("campaign_block") == block and record.get("campaign_arm") == arm
            and record.get("batch_index") == 1000 + block
            and record.get("policy_id") == state["policy_id"], "Match campaign/policy provenance differs")
    require(all(record.get(key) == value for key, value in planned.items()), "Prescribed match identity differs")


def common_state_probability_rows(candidate, baseline, comparison):
    """Compare probabilities only for identical views before action divergence.

    Include the first divergent response: its actions differ, but its pre-response
    state still shares the complete preceding action history. Exclude all later
    rows, even if a later board happens to look similar.
    """
    stop = comparison["first_divergent_step"]
    rows = []
    baseline_rows = {row["step"]: row for row in baseline["learner_rows"]}
    for first in candidate["learner_rows"]:
        if stop is not None and first["step"] > stop:
            continue
        second = baseline_rows.get(first["step"])
        require(second is not None, "Shared response prefix has unmatched learner rows")
        require(all(first[key] == second[key] for key in
                    ("player_view_sha256", "encoded_sha256", "semantic_keys", "uniform_draw")),
                "Probability comparison requires identical player-visible inputs and sampler draws")
        a, b = first["probabilities"], second["probabilities"]
        require(len(a) == len(b) and bool(a) and all(math.isfinite(p) and 0 <= p <= 1 for p in a + b),
                "Invalid saved probability distribution")
        differences = [abs(x - y) for x, y in zip(a, b)]
        rows.append({"step": first["step"], "call_opportunity": first["call_candidates"] > 0,
            "maximum_absolute_probability_change": max(differences),
            "probability_l1_change": math.fsum(differences),
            "call_mass_change": first["call_mass"] - second["call_mass"],
            "gate_mass_change": first["gate_mass"] - second["gate_mass"],
            "selected_action_changed": first["selected_semantic_key"] != second["selected_semantic_key"]})
    return rows


def summarize_probability_rows(rows):
    result = {}
    for label, population in (("all_common_state_rows", rows),
            ("call_opportunity_rows", [r for r in rows if r["call_opportunity"]]),
            ("no_call_rows", [r for r in rows if not r["call_opportunity"]])):
        result[label] = {"rows": len(population),
            "exactly_unchanged_probability_rows": sum(r["maximum_absolute_probability_change"] == 0 for r in population),
            "changed_probability_rows": sum(r["maximum_absolute_probability_change"] > 0 for r in population),
            "changed_selected_actions": sum(r["selected_action_changed"] for r in population),
            **{name: summarize_numbers(r[name] for r in population) for name in
               ("maximum_absolute_probability_change", "probability_l1_change", "call_mass_change", "gate_mass_change")}}
    result["scope"] = "Identical visible inputs in each paired action prefix, including its first divergent response; no later states are compared."
    return result


def analyze(config_path: Path, output_path: Path, *, expected_config_sha256: str) -> dict[str, Any]:
    config_path = config_path.resolve()
    require(digest(config_path) == expected_config_sha256, "Configuration hash differs from requested campaign")
    config = json.loads(config_path.read_text())
    campaign, behavior = expected_config_sha256, config["behavior_protocol_sha256"]
    reports, data = (ROOT / config["reports_root"]).resolve(), (ROOT / config["data_root"]).resolve()
    inputs = [{"path": str(config_path), "sha256": campaign}]
    seen = set()
    def read(path, expected_hash=None, under=None, unique=False):
        path = Path(path).resolve()
        require(under is None or path.is_relative_to(under), "Input escaped its campaign artifact root")
        require(not unique or path not in seen, "Duplicate match/lifecycle artifact")
        actual = digest(path)
        require(expected_hash is None or actual == expected_hash, "Saved JSON artifact hash differs")
        if unique:
            seen.add(path)
        inputs.append({"path": str(path), "sha256": actual})
        return json.loads(path.read_text())
    require(config["evaluation_blocks"] == 32 and config["maximum_matches"] == 256
            and len(config["evaluation"]) == 32, "Expected the fixed 32-block recovery")
    plan_path = reports / "plan.json"
    plan, plan_sha = read(plan_path), digest(plan_path)
    require(plan["config_sha256"] == campaign and plan["behavior_protocol_sha256"] == behavior
            and plan["evaluation"] == config["evaluation"] and plan["boundaries"] == config["boundaries"],
            "Frozen plan and current configuration differ")
    states, implementation = plan["states"], plan["implementation"]
    require(states["candidate"]["updates"] == 4 and states["baseline"]["updates"] == 0
            and states["candidate"]["base_weights_sha256"] == states["baseline"]["base_weights_sha256"],
            "Expected fixed fourth-update candidate and identical initialized baseline base")
    # The live ledger necessarily changes when this analysis lease closes.
    # Use the immutable final evaluation receipt's closed prefix as evidence;
    # the separate final audit checks the current full ledger independently.
    final_receipt = read(reports / "eval-attempt-31.json")
    validate_closed_evaluation_budget(final_receipt, campaign)
    arms, pairs, probability_rows = {arm: [] for arm in ARMS}, [], []
    for index, planned_block in enumerate(config["evaluation"]):
        block, receipt = read(reports / f"eval-{index}.json"), read(reports / f"eval-attempt-{index}.json")
        validate_closed_block(block, receipt, index=index, campaign=campaign, plan_sha=plan_sha,
            planned=planned_block, states=states, implementation=implementation)
        require(block["behavior_protocol_sha256"] == behavior and block["boundaries"] == config["boundaries"],
                "Block policy/protocol identity differs")
        lookup = {}
        for position, evidence in enumerate(block["matches"]):
            planned_pair, arm = planned_block["pairs"][position // 2], ARMS[position % 2]
            seat = planned_pair["learner_seat"]
            require(evidence["arm"] == arm and evidence["learner_seat"] == seat, "Arm/seat order differs")
            planned = {"match_index": planned_pair[f"{arm}_match_index"], "seed": planned_block["wall_seed"],
                       "sampling_seed": planned_pair["sampling_seed"], "learner_seat": seat}
            record = read(evidence["json_path"], evidence["json_sha256"], under=data, unique=True)
            validate_record(record, campaign=campaign, behavior=behavior, plan_sha=plan_sha,
                            block=index, arm=arm, planned=planned, state=states[arm])
            start = read(evidence["lifecycle_start_path"], evidence["lifecycle_start_sha256"], under=data, unique=True)
            finish = read(evidence["lifecycle_finish_path"], evidence["lifecycle_finish_sha256"], under=data, unique=True)
            require(start["campaign_sha256"] == campaign and start["plan_sha256"] == plan_sha
                    and start["planned_match"] == planned and start["state"] == states[arm]
                    and start["arm"] == finish["arm"] == arm and start["block"] == finish["block"] == index
                    and finish["campaign_sha256"] == campaign and finish["completed"] is True
                    and finish["interrupted"] is False and finish["failure"] is None
                    and finish["start_sha256"] == evidence["lifecycle_start_sha256"]
                    and finish["evidence"]["json_sha256"] == evidence["json_sha256"]
                    and finish["evidence"]["json_path"] == evidence["json_path"]
                    and receipt["started_unix"] <= start["started_unix"] <= finish["finished_unix"] <= receipt["finished_unix"],
                    "Durable match lifecycle receipt differs")
            require((arm, seat) not in lookup, "Duplicate arm/seat pair")
            lookup[arm, seat] = record
            arms[arm].append(record)
        require(set(lookup) == {(arm, seat) for arm in ARMS for seat in range(4)}, "Missing fixed arm/seat pair")
        for seat in range(4):
            first, second = lookup["candidate", seat], lookup["baseline", seat]
            comparison = compare_pair(first, second, index, seat)
            pairs.append(comparison)
            probability_rows.extend(common_state_probability_rows(first, second, comparison))
    require(all(len(arm) == 128 for arm in arms.values()) and len(pairs) == 128, "Fixed endpoint sample size differs")
    endpoint_path = reports / "evaluation.json"
    endpoint = read(endpoint_path)
    require(endpoint["blocks"] == 32 and endpoint["matches"] == 256
            and endpoint["old_evaluation_pooled"] is False and endpoint["optimizer_updates"] == 0,
            "Fixed endpoint missing or pooled historical evaluation")
    result = {"schema": "stage3d-saved-data-descriptive-behavior-v1", "valid": True,
        "config_sha256": campaign, "behavior_protocol_sha256": behavior, "plan_sha256": plan_sha,
        "analysis_source_sha256": digest(Path(__file__)), "analysis_helper_sha256": digest(ROOT / "scripts/analyze_stage3c.py"),
        "complete_experiment": True, "complete_evaluation_blocks": list(range(32)), "input_match_count": 256,
        "input_files": inputs, "budget_evidence_scope": "Immutable eval-attempt-31.json closed evaluation prefix; current ledger checked by independent final audit.",
        "models_loaded": 0, "new_games": 0, "new_optimizer_updates": 0,
        "old_evaluation_pooled": False, "inferential_comparison_computed": False,
        "fixed_endpoint_reference": {"path": str(endpoint_path), "sha256": digest(endpoint_path)},
        "fixed_endpoints": {arm: {"checkpoint": config["boundaries"][arm], "state": states[arm]} for arm in ARMS},
        "evaluation_by_arm": {arm: summarize(records) for arm, records in arms.items()},
        "paired_behavior": {"pairs": len(pairs),
            "identical_action_sequences": sum(p["identical_action_sequence"] for p in pairs),
            "divergent_action_sequences": sum(not p["identical_action_sequence"] for p in pairs),
            "identical_all_seat_scores": sum(p["identical_all_seat_scores"] for p in pairs),
            "identical_all_seat_placements": sum(p["identical_all_seat_placements"] for p in pairs),
            "same_learner_placement": sum(p["same_learner_placement"] for p in pairs), "details": pairs},
        "common_state_probability_changes": summarize_probability_rows(probability_rows),
        "interpretation": "Descriptive recovery-only populations. Call opportunity categories overlap. Shared-state probability changes describe mechanisms; strength uncertainty belongs solely to the separate preregistered fixed endpoint. No models were evaluated anew or promoted."}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage3d.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/stage3d/behavior.json")
    parser.add_argument("--expected-config-sha256", required=True)
    args = parser.parse_args()
    result = analyze(args.config, args.output, expected_config_sha256=args.expected_config_sha256)
    print(json.dumps({"valid": True, "input_matches": result["input_match_count"],
        "identical_pairs": result["paired_behavior"]["identical_action_sequences"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
