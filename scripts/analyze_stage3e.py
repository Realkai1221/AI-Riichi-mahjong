"""Descriptive analysis of the complete fixed Stage 3E continuation campaign.

Only saved data are read. No models, new games, optimization, checkpoint selection,
new confidence intervals or additional inferential tests are performed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from analyze_stage3c import compare_pair, digest, summarize, summarize_numbers
from analyze_stage3d import common_state_probability_rows, summarize_probability_rows

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("candidate", "baseline")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_closed_campaign_budget(receipt, campaign):
    require(receipt.get("name") == "eval:31" and receipt.get("config_sha256") == campaign
            and receipt.get("status") == "completed" and receipt.get("failure") is None,
            "Final evaluation receipt is not closed successfully")
    ledger = receipt["budget"]
    operations = ledger["operations"]
    formal = [op for op in operations if op["name"].startswith(("batch:", "eval:"))]
    expected = [f"batch:{i}" for i in range(4, 32)] + [f"eval:{i}" for i in range(32)]
    require(ledger["identity"] == campaign and ledger["limit_seconds"] == 1200
            and [op["name"] for op in formal] == expected
            and all(op["status"] == "completed" and op["failure"] is None for op in formal)
            and all(op["status"] in ("completed", "failed") for op in operations),
            "Fixed training/evaluation operations are incomplete, failed, unclosed or duplicated")


def validate_receipt(receipt, *, name, campaign, implementation):
    require(receipt.get("name") == name and receipt.get("config_sha256") == campaign
            and receipt.get("implementation") == implementation
            and receipt.get("status") == "completed" and receipt.get("failure") is None
            and receipt.get("finished_unix", -1) >= receipt.get("started_unix", 0),
            "Closed successful receipt provenance differs")


def validate_record(record, *, campaign, behavior, plan_sha, role, batch_index, block, arm, planned, state):
    require(record.get("completed") is True and record.get("terminated") is True
            and record.get("terminal_result_verified") is True and record.get("failure") is None
            and record.get("truncated") is False and record.get("interrupted") is False,
            "Only complete uninterrupted audited outcomes may enter this summary")
    eligible = role == "training"
    require(role in ("training", "evaluation") and record.get("collection_role") == role
            and record.get("training_eligible") is eligible
            and all(row.get("training_eligible") is eligible for row in record["learner_rows"]),
            "Match or learner-row training eligibility differs from its fixed role")
    require(record.get("schema") == "stage3c-frozen-base-call-heads-v1"
            and record.get("collector_schema") == "stage3e-continuation-lifecycle-v1"
            and record.get("campaign_sha256") == campaign and record.get("protocol_sha256") == behavior
            and record.get("campaign_plan_sha256") == plan_sha and record.get("campaign_phase") == role
            and record.get("campaign_block") == block and record.get("campaign_arm") == arm
            and record.get("campaign_batch_index") == record.get("batch_index") == batch_index
            and record.get("policy_id") == state["policy_id"], "Match campaign/policy provenance differs")
    require(all(record.get(key) == value for key, value in planned.items()), "Prescribed match identity differs")
    require(all(row.get("eligible") is True and row.get("policy_id") == state["policy_id"]
                and row.get("batch_index") == batch_index and row.get("match_index") == record["match_index"]
                and row.get("seat") == record["learner_seat"]
                and row.get("fallback") is False and row.get("overridden") is False
                for row in record["learner_rows"]), "Learner row ownership or behavior provenance differs")


def validate_training_batch(batch, receipt, *, index, campaign, behavior, plan_sha, planned,
                            state, boundary, implementation):
    validate_receipt(receipt, name=f"batch:{index}", campaign=campaign, implementation=implementation)
    require(batch.get("schema") == "stage3e-batch-v1" and batch.get("complete") is True
            and batch.get("batch_index") == index and batch.get("config_sha256") == campaign
            and batch.get("behavior_protocol_sha256") == behavior and batch.get("plan_sha256") == plan_sha
            and batch.get("planned_matches") == planned and batch.get("input_state") == state
            and batch.get("input_boundary") == boundary and batch.get("implementation") == implementation
            and batch.get("termination_requested") is False and len(batch.get("matches", [])) == 16,
            "Training batch does not match the fixed complete continuation chain")
    after, update = batch["output_state"], batch["update"]
    require(state["updates"] == state["optimizer_steps_executed"] == index
            and state["retired_batches"] == list(range(index))
            and after["updates"] == after["optimizer_steps_executed"] == index + 1
            and after["retired_batches"] == list(range(index + 1))
            and update["before"] == state and update["after"] == after
            and batch["boundary_roundtrip_verified"] is True
            and all(after[key] == state[key] for key in
                    ("base_weights_sha256", "immutable_base_sha256", "rank_head_sha256",
                     "optimizer_config", "model_config", "head_config", "reference_sha256", "protocol_sha256")),
            "Continuation reset optimizer/settings, changed frozen weights or skipped an update")
    require(update["batch_index"] == index and update["planned_matches"] == planned
            and update["completed_matches"] == update["loss_denominator_matches"] == 16
            and update["optimizer_steps"] == 1 and update["opponent_rows_in_loss"] == 0
            and update["probability_replay_max_absolute_error"] == 0
            and update["selected_log_probability_replay_max_error"] == 0
            and update["distribution_diagnostic_complete"] is True
            and update["distribution_diagnostic_rows"] == update["optional_learner_rows"]
            and all(update[key] is True for key in ("gradients_finite", "parameters_finite", "base_unchanged", "rank_head_unchanged")),
            "Update diagnostics are incomplete or violate the fixed learner protocol")
    require(batch["output_boundary"]["updates"] == index + 1
            and all(batch["output_boundary"][key] == value for key, value in after.items()),
            "Saved boundary differs from the final batch state")


def validate_closed_block(block, receipt, *, index, campaign, behavior, plan_sha, planned,
                          states, boundaries, implementation):
    validate_receipt(receipt, name=f"eval:{index}", campaign=campaign, implementation=implementation)
    require(block.get("schema") == "stage3e-evaluation-block-v1" and block.get("complete") is True
            and block.get("block") == index and block.get("config_sha256") == campaign
            and block.get("behavior_protocol_sha256") == behavior and block.get("plan_sha256") == plan_sha
            and block.get("planned") == planned and block.get("states_before") == states
            and block.get("states_after") == states and block.get("boundaries") == boundaries
            and block.get("implementation") == implementation and block.get("termination_requested") is False
            and len(block.get("matches", [])) == 8,
            "Fixed evaluation block or endpoint state differs")


def update_diagnostics(batch):
    update = batch["update"]
    keys = ("completed_matches", "optional_learner_rows", "forced_learner_rows", "gradient_eligible_rows",
            "gradient_norm_before_clip", "gradient_components", "parameter_max_absolute_change", "parameter_l2_change",
            "parameter_component_l2_change", "distribution_max_absolute_change", "distribution_mean_l1_change",
            "mean_optional_entropy_before", "mean_optional_entropy_after", "mean_call_mass_after_on_call_rows",
            "distribution_diagnostic_complete", "distribution_diagnostic_rows", "numerical_no_op",
            "probability_replay_max_absolute_error", "selected_log_probability_replay_max_error",
            "opponent_rows_in_loss", "base_unchanged", "rank_head_unchanged")
    return {"batch_index": batch["batch_index"], "updates_before": batch["input_state"]["updates"],
            "updates_after": batch["output_state"]["updates"], **{key: update[key] for key in keys}}


def analyze(config_path: Path, output_path: Path, *, expected_config_sha256: str) -> dict[str, Any]:
    config_path = config_path.resolve()
    require(digest(config_path) == expected_config_sha256, "Configuration hash differs from requested campaign")
    config = json.loads(config_path.read_text())
    campaign, behavior = expected_config_sha256, config["behavior_protocol_sha256"]
    reports, data = (ROOT / config["reports_root"]).resolve(), (ROOT / config["data_root"]).resolve()
    inputs, seen = [{"path": str(config_path), "sha256": campaign}], set()
    def bind(path, expected_hash=None, under=None, unique=False):
        path = Path(path).resolve()
        require(under is None or path.is_relative_to(under), "Input escaped its campaign artifact root")
        require(not unique or path not in seen, "Duplicate match/lifecycle artifact")
        actual = digest(path)
        require(expected_hash is None or actual == expected_hash, "Saved artifact hash differs")
        if unique:
            seen.add(path)
        inputs.append({"path": str(path), "sha256": actual})
        return path
    def read(path, expected_hash=None, under=None, unique=False):
        return json.loads(bind(path, expected_hash, under, unique).read_text())
    require(config["starting_updates"] == 4 and config["maximum_updates"] == 32
            and config["maximum_batches"] == 28 and config["matches_per_batch"] == 16
            and config["training_matches"] == 448 and config["evaluation_blocks"] == 32
            and config["evaluation_matches"] == 256 and config["maximum_matches"] == 704
            and [b["batch_index"] for b in config["batches"]] == list(range(4, 32))
            and len(config["evaluation"]) == 32, "Expected the complete fixed Stage 3E horizon")
    require([m["match_index"] for b in config["batches"] for m in b["matches"]] == list(range(576, 1024))
            and [p[f"{arm}_match_index"] for b in config["evaluation"] for p in b["pairs"] for arm in ARMS]
                == list(range(1024, 1280)), "Fixed match identities differ")
    plan_path = reports / "plan.json"
    plan, plan_sha = read(plan_path), digest(plan_path)
    require(plan["config_sha256"] == campaign and plan["behavior_protocol_sha256"] == behavior
            and plan["batches"] == config["batches"] and plan["evaluation"] == config["evaluation"]
            and plan["initial_boundary"] == config["initial_boundary"], "Frozen plan and configuration differ")
    implementation = plan["implementation"]
    for relative, sha in implementation.items():
        bind(ROOT / relative, sha, under=ROOT)
    # The active analysis lease necessarily changes the live ledger. Bind the
    # immutable final evaluation receipt; the independent audit checks the full ledger.
    final_receipt = read(reports / "eval-attempt-31.json")
    validate_closed_campaign_budget(final_receipt, campaign)

    def match_record(evidence, *, receipt, role, batch_index, block, arm, planned, state):
        require(evidence["arm"] == arm and evidence["learner_seat"] == planned["learner_seat"], "Arm/seat order differs")
        record = read(evidence["json_path"], evidence["json_sha256"], under=data, unique=True)
        bind(evidence["npz_path"], evidence["npz_sha256"], under=data, unique=True)
        require(record["encoded_artifact"] == {"path": evidence["npz_path"], "sha256": evidence["npz_sha256"],
                "rows": len(record["learner_rows"])}, "Saved row-array identity differs")
        validate_record(record, campaign=campaign, behavior=behavior, plan_sha=plan_sha,
            role=role, batch_index=batch_index, block=block, arm=arm, planned=planned, state=state)
        start = read(evidence["lifecycle_start_path"], evidence["lifecycle_start_sha256"], under=data, unique=True)
        finish = read(evidence["lifecycle_finish_path"], evidence["lifecycle_finish_sha256"], under=data, unique=True)
        require(start["schema"] == "stage3e-match-start-v1" and finish["schema"] == "stage3e-match-finish-v1"
                and start["campaign_sha256"] == finish["campaign_sha256"] == campaign
                and start["behavior_protocol_sha256"] == finish["behavior_protocol_sha256"] == behavior
                and start["plan_sha256"] == plan_sha and start["planned_match"] == planned and start["state"] == state
                and start["arm"] == finish["arm"] == arm and start["block"] == finish["block"] == block
                and start["phase"] == finish["phase"] == role and start["collection_batch_index"] == batch_index
                and finish["match_index"] == planned["match_index"] and finish["completed"] is True
                and finish["interrupted"] is False and finish["failure"] is None
                and finish["start_path"] == evidence["lifecycle_start_path"]
                and finish["start_sha256"] == evidence["lifecycle_start_sha256"]
                and finish["evidence"] == {key: evidence[key] for key in ("json_path", "json_sha256", "npz_path", "npz_sha256")}
                and receipt["started_unix"] <= start["started_unix"] <= finish["finished_unix"] <= receipt["finished_unix"],
                "Durable match lifecycle receipt differs")
        return record

    training, training_batches, diagnostics = [], [], []
    initial_state, initial_boundary = plan["initial_state"], plan["initial_boundary"]
    state, boundary = initial_state, initial_boundary
    bind(boundary["path"], boundary["sha256"])
    for index, planned_batch in zip(range(4, 32), config["batches"]):
        batch, receipt = read(reports / f"batch-{index}.json"), read(reports / f"attempt-{index}.json")
        validate_training_batch(batch, receipt, index=index, campaign=campaign, behavior=behavior, plan_sha=plan_sha,
            planned=planned_batch["matches"], state=state, boundary=boundary, implementation=implementation)
        records = [match_record(evidence, receipt=receipt, role="training", batch_index=index, block=None,
            arm="learner", planned=planned, state=state)
            for evidence, planned in zip(batch["matches"], planned_batch["matches"])]
        summary = summarize(records)
        require(summary["optional_rows"] == batch["update"]["optional_learner_rows"]
                and summary["forced_rows"] == batch["update"]["forced_learner_rows"],
                "Update row counts differ from saved behavior")
        training_batches.append({"batch_index": index, "behavior_update": index, "summary": summary})
        diagnostics.append(update_diagnostics(batch))
        training.extend(records)
        state, boundary = batch["output_state"], batch["output_boundary"]
        bind(boundary["path"], boundary["sha256"], under=data)
    states = {"candidate": state, "baseline": initial_state}
    boundaries = {"candidate": boundary, "baseline": initial_boundary}
    require(states["candidate"]["updates"] == 32 and states["baseline"]["updates"] == 4,
            "Expected fixed update-32 candidate versus update-4 baseline")
    arms, pairs, probability_rows = {arm: [] for arm in ARMS}, [], []
    block_differences = []
    for index, planned_block in enumerate(config["evaluation"]):
        block, receipt = read(reports / f"eval-{index}.json"), read(reports / f"eval-attempt-{index}.json")
        validate_closed_block(block, receipt, index=index, campaign=campaign, behavior=behavior, plan_sha=plan_sha,
            planned=planned_block, states=states, boundaries=boundaries, implementation=implementation)
        lookup = {}
        for position, evidence in enumerate(block["matches"]):
            planned_pair, arm = planned_block["pairs"][position // 2], ARMS[position % 2]
            seat = planned_pair["learner_seat"]
            planned = {"match_index": planned_pair[f"{arm}_match_index"], "seed": planned_block["wall_seed"],
                       "sampling_seed": planned_pair["sampling_seed"], "learner_seat": seat}
            record = match_record(evidence, receipt=receipt, role="evaluation", batch_index=2000 + index,
                                  block=index, arm=arm, planned=planned, state=states[arm])
            require((arm, seat) not in lookup, "Duplicate arm/seat pair")
            lookup[arm, seat] = record
            arms[arm].append(record)
        require(set(lookup) == {(arm, seat) for arm in ARMS for seat in range(4)}, "Missing fixed arm/seat pair")
        block_pairs = []
        for seat in range(4):
            first, second = lookup["candidate", seat], lookup["baseline", seat]
            comparison = compare_pair(first, second, index, seat)
            pairs.append(comparison)
            block_pairs.append(comparison)
            probability_rows.extend(common_state_probability_rows(first, second, comparison))
        difference = math.fsum(p["learner_utility_difference"] for p in block_pairs) / 4
        require(block["mean_utility_difference"] == difference, "Recorded block endpoint differs from saved outcomes")
        block_differences.append(difference)
    require(len(training) == 448 and all(len(arm) == 128 for arm in arms.values()) and len(pairs) == 128,
            "Fixed horizon sample size differs")
    endpoint_path = reports / "evaluation.json"
    endpoint = read(endpoint_path)
    require(endpoint["schema"] == "stage3e-evaluation-v1" and endpoint["blocks"] == 32 and endpoint["matches"] == 256
            and endpoint["old_evaluation_pooled"] is False and endpoint["training_updates"] == 28
            and endpoint["evaluation_optimizer_updates"] == 0 and endpoint["candidate_promoted"] is False
            and endpoint["block_differences"] == block_differences
            and endpoint["mean_utility_difference"] == math.fsum(block_differences) / 32,
            "Fixed endpoint missing, altered or pooled historical evaluation")
    evaluation_summaries = {arm: summarize(records) for arm, records in arms.items()}
    require(all(endpoint[f"{arm}_mean_rank"] == evaluation_summaries[arm]["mean_rank"] for arm in ARMS),
            "Fixed endpoint placements differ from saved outcomes")
    result = {"schema": "stage3e-saved-data-fixed-horizon-descriptive-v1", "valid": True,
        "config_sha256": campaign, "behavior_protocol_sha256": behavior, "plan_sha256": plan_sha,
        "analysis_source_sha256": digest(Path(__file__)),
        "analysis_helpers": {name: digest(ROOT / "scripts" / name) for name in ("analyze_stage3c.py", "analyze_stage3d.py")},
        "complete_experiment": True, "complete_training_batches": list(range(4, 32)),
        "complete_evaluation_blocks": list(range(32)), "input_match_count": 704,
        "input_files": inputs, "budget_evidence_scope": "Immutable eval-attempt-31.json closed training/evaluation prefix; current ledger checked by independent final audit.",
        "models_loaded": 0, "new_games": 0, "new_optimizer_updates": 0,
        "old_evaluation_pooled": False, "inferential_comparison_computed": False,
        "fixed_endpoint_reference": {"path": str(endpoint_path), "sha256": digest(endpoint_path)},
        "fixed_endpoints": {arm: {"checkpoint": boundaries[arm], "state": states[arm]} for arm in ARMS},
        "training": {"additional_matches": 448, "additional_updates": 28, "updates_before": 4, "updates_after": 32,
            "summary": summarize(training), "by_batch": training_batches, "update_diagnostics": diagnostics,
            "numerical_no_op_updates": sum(d["numerical_no_op"] for d in diagnostics),
            "maximum_saved_replay_error": max(d["probability_replay_max_absolute_error"] for d in diagnostics),
            "maximum_saved_log_replay_error": max(d["selected_log_probability_replay_max_error"] for d in diagnostics),
            "diagnostic_populations": {name: summarize_numbers(d[name] for d in diagnostics) for name in
                ("parameter_max_absolute_change", "parameter_l2_change", "distribution_max_absolute_change",
                 "distribution_mean_l1_change", "gradient_norm_before_clip")},
            "interpretation": "Descriptive training populations change with each behavior policy. Per-update distribution diagnostics reuse that batch's pre-update states and are not an independent strength endpoint."},
        "evaluation_by_arm": evaluation_summaries,
        "paired_behavior": {"pairs": len(pairs),
            "identical_action_sequences": sum(p["identical_action_sequence"] for p in pairs),
            "divergent_action_sequences": sum(not p["identical_action_sequence"] for p in pairs),
            "identical_all_seat_scores": sum(p["identical_all_seat_scores"] for p in pairs),
            "identical_all_seat_placements": sum(p["identical_all_seat_placements"] for p in pairs),
            "same_learner_placement": sum(p["same_learner_placement"] for p in pairs), "details": pairs},
        "common_state_probability_changes": summarize_probability_rows(probability_rows),
        "interpretation": "Fixed-horizon descriptive populations only. Opportunity categories overlap. Probability changes compare identical visible inputs only through the first divergent response. Strength uncertainty belongs to the separately preregistered fixed endpoint; no new inference, checkpoint selection or promotion."}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage3e.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/stage3e/behavior.json")
    parser.add_argument("--expected-config-sha256", required=True)
    args = parser.parse_args()
    result = analyze(args.config, args.output, expected_config_sha256=args.expected_config_sha256)
    print(json.dumps({"valid": True, "input_matches": result["input_match_count"],
        "training_updates": result["training"]["additional_updates"],
        "identical_pairs": result["paired_behavior"]["identical_action_sequences"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
