"""Audit saved Package A evidence using only the Python standard library.

Run after the four formal A2 reports and all numerical checks are finished.
Reads existing evidence; never imports project/model code or runs experiments.
The only output is reports/package_a/audit.json. Integrity success is distinct
from the scientific non-inferiority and durability decisions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist, mean, stdev


ROOT = Path(__file__).resolve().parents[1]
INVENTORY_SHA256 = "7743a8582b39343ceef16ac3b3ff3373bbf8d5e437b6e46718517307b11fc87b"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def tree_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-12)


class Audit:
    def __init__(self):
        self.checks = []
        self.inputs = {}

    def check(self, name, value):
        self.checks.append({"name": name, "passed": bool(value)})

    def load(self, path):
        path = Path(path)
        self.inputs[str(path)] = digest(path)
        return read(path)


def probability_summary(record):
    groups = record["groups"]
    return {
        "rows": record["rows"],
        "optional_rows": groups["optional"]["rows"],
        "optional_entropy_nats": groups["optional"]["entropy_nats"]["mean"],
        "optional_normalized_entropy": groups["optional"]["normalized_entropy"]["mean"],
        "optional_teacher_best_mass": groups["optional"]["teacher_best_mass"]["mean"],
        "call_available_rows": groups["call_available:True"]["rows"],
        "call_mass_when_available": groups["call_available:True"]["call_mass"]["mean"],
        "kan_available_rows": groups["kan_available:True"]["rows"],
        "kan_mass_when_available": groups["kan_available:True"]["kan_mass"]["mean"],
    }


def audit(config_path):
    a = Audit()
    config_path = Path(config_path).resolve()
    config = a.load(config_path)
    identity = digest(config_path)
    cfg = config["stability"]
    a1_cfg = config["confirmation"]
    reports = ROOT / cfg["reports_root"]
    data = ROOT / cfg["data_root"]
    a1_root = ROOT / a1_cfg["reports_root"]
    planned = [(r, s) for r in ("control", "smooth") for s in cfg["seeds"]]
    expected_files = {f"run-{r}-{s}.json" for r, s in planned}
    actual_files = {p.name for p in reports.glob("run-*.json")}
    if actual_files != expected_files:
        raise ValueError("All four and only four final run reports are required before auditing")
    ledger = a.load(data / "budget-ledger.json")
    if any(op["status"] == "active" for op in ledger["operations"]):
        raise ValueError("A2 numerical work is active; run the final audit after all leases close")
    plan = a.load(reports / "formal-plan.json")
    probe = a.load(reports / "probe.json")
    baseline = a.load(reports / "baseline-probability.json")
    runs = {(r, s): a.load(reports / f"run-{r}-{s}.json") for r, s in planned}
    markers = {(r, s): a.load(reports / "attempts" / f"{r}-{s}.json") for r, s in planned}
    extra_markers = [a.load(reports / f"{name}-attempt.json") for name in ("probe", "baseline-probability")]
    a.check("four_attempt_markers_only", {p.name for p in (reports / "attempts").glob("*.json")}
            == {f"{r}-{s}.json" for r, s in planned} and cfg["maximum_attempts"] == 4)
    a.check("declared_formal_order", plan["attempt_order"] == [list(x) for x in planned])
    a.check("plan_frozen_before_formal_attempts", plan["frozen_before_formal_attempts"])
    a.check("formal_settings_equal_config", plan["settings"] == cfg and plan["planned_steps_each"] == cfg["steps"] == 18500)
    records = [plan, probe, baseline, *runs.values(), *markers.values(), *extra_markers]
    a.check("config_sha256_all_records", all(r["config_sha256"] == identity for r in records))
    a.check("ledger_identity_and_ceiling", ledger["identity"] == identity and ledger["limit_seconds"] == cfg["limit_seconds"] == 2700)
    a.check("probe_identity_and_completion", digest(reports / "probe.json") == plan["probe_sha256"] and probe["complete"])
    a.check("baseline_probability_identity", digest(reports / "baseline-probability.json") == plan["baseline_probability_sha256"]
            and baseline["probability_metrics"]["complete"] and baseline["sources_unchanged"])
    source_paths = sorted((ROOT / "src/riichi_ai").glob("*.py")) + [ROOT / "scripts/package_a_stability.py", ROOT / cfg["stage2b_config"]]
    implementation = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    a.check("all_source_and_controller_hashes", all(r["implementation"] == implementation for r in [plan, probe, baseline, *runs.values()]))
    controller_hash = implementation["scripts/package_a_stability.py"]
    stage_hash = implementation[cfg["stage2b_config"]]
    for label, r in [("probe", probe), *[(f"{recipe}-{seed}", run) for (recipe, seed), run in runs.items()]]:
        settings = r["settings"]
        protocol = settings["protocol_manifest"]
        a.check(f"{label}:source_identity", r["controller_sha256"] == controller_hash and r["sources_unchanged"]
                and all(digest(ROOT / "src/riichi_ai" / p) == h for p, h in r["source_manifest"]["files"].items()))
        a.check(f"{label}:protocol_identity", protocol["config_sha256"] == identity
                and protocol["controller_sha256"] == controller_hash and protocol["stage2b_config_sha256"] == stage_hash)
        a.check(f"{label}:frozen_rank_head", r["frozen_rank_head_unchanged"]
                and all(d["parameters"]["rank_head"]["update_l2"] == 0 for d in r["diagnostics"]))
        a.check(f"{label}:fixed_indices", settings["validation_indices_sha256"] == probe["settings"]["validation_indices_sha256"]
                and settings["diagnostic_indices_sha256"] == probe["settings"]["diagnostic_indices_sha256"]
                and settings["validation_seed"] == cfg["validation_seed"])
        a.check(f"{label}:data_matches_probe", r["data_manifest"] == probe["data_manifest"]
                and r["validation_manifest"] == probe["validation_manifest"])
        a.check(f"{label}:conservative_memory_limit", 0 < r["sampled_peak_conservative_memory_bytes"] < cfg["memory_bytes"])
    stage = a.load(ROOT / cfg["stage2b_config"])
    for key, derived in (("data_manifest", "derived-stage2a"), ("validation_manifest", "derived-stage2b")):
        manifest = probe[key]
        a.check(f"{key}:current_source_manifest", digest(Path(manifest["path"]) / "manifest.json") == manifest["source_manifest_sha256"])
        a.check(f"{key}:current_derived_manifest", digest(Path(stage["data_root"]) / derived / "manifest.json") == manifest["derived_manifest_sha256"])
    a.check("historical_training_source", probe["data_manifest"]["source_manifest_sha256"] == stage["stage2a_manifest_sha256"])
    inventory_path = ROOT / "reports/stage2b/manifest.json"
    inventory = a.load(inventory_path)
    a.check("stage2b_inventory_preserved", digest(inventory_path) == INVENTORY_SHA256)
    historical_hashes = {e["path"]: e["sha256"] for e in inventory["files"] if e["scope"] == "project"}
    rows = []
    for recipe, seed in planned:
        name = f"{recipe}-{seed}"
        r = runs[recipe, seed]
        settings = r["settings"]
        marker = markers[recipe, seed]
        a.check(f"{name}:attempt_identity", marker["recipe"] == r["recipe"] == recipe
                and marker["seed"] == r["seed"] == settings["seed"] == seed
                and marker["status"] == "completed" and marker["failure"] is None)
        a.check(f"{name}:settings", settings["epsilon"] == (0 if recipe == "control" else cfg["epsilon"])
                and settings["steps"] == r["requested_steps"] == cfg["steps"]
                and settings["batch_size"] == cfg["batch_size"] and settings["lr"] == cfg["learning_rate"]
                and settings["rank_weight"] == 0 and settings["weight_decay"] == 1e-4
                and settings["gradient_clip"] == 1 and settings["safety"] == cfg["safety"]
                and settings["validate_every"] == cfg["validate_every"] and settings["diagnostic_every"] == cfg["diagnostic_every"]
                and settings["protocol_manifest"]["recipe"] == recipe and settings["protocol_manifest"]["probe"] is False)
        old_path = ROOT / "reports/stage2b" / f"run-features_rank0-{seed}.json"
        old = a.load(old_path)["training"]
        comparison = r["historical_comparison"]
        a.check(f"{name}:historical_report_identity", digest(old_path) == comparison["report_sha256"]
                == historical_hashes[str(old_path.relative_to(ROOT))])
        a.check(f"{name}:historical_initialization", r["initial_state_sha256"] == old["initialization"]["state_sha256"]
                and r["initialization"]["common_weights_sha256"] == old["initialization"]["manifest"]["common_weights_sha256"]
                and comparison["initial_state_matches"] and comparison["common_initialization_matches"])
        a.check(f"{name}:historical_data", r["data_manifest"] == old["data_manifest"] and comparison["data_manifest_matches"])
        full_stream = r["sampled_index_digest"] == old["sampled_index_digest"] if r["completed_steps"] == old["completed_steps"] else None
        a.check(f"{name}:historical_full_stream", comparison["full_stream_matches_if_full_horizon"] == full_stream
                and (full_stream is True if r["completed_steps"] == cfg["steps"] else full_stream is None))
        a.check(f"{name}:history_complete", len(r["history"]) == r["completed_steps"]
                and [h["step"] for h in r["history"]] == list(range(1, r["completed_steps"] + 1)))
        fixed = r["terminal_fixed_validation"]
        full = r["terminal_full_validation"]
        probability = r["terminal_probability_metrics"]
        agreement = fixed["equal_stratum_means"]["discard_tie_agreement"]
        complete = bool(r["completed_steps"] == r["requested_steps"] and r["stop_reason"] == "steps_completed"
            and not r["failure"] and fixed["complete"] and full["complete"] and probability["complete"]
            and r["frozen_rank_head_unchanged"] and r["sources_unchanged"])
        durable = bool(complete and not r["safety"]["scale_alarm"] and not r["safety"]["tripped"] and agreement >= .999)
        a.check(f"{name}:terminal_validation_identity", fixed["indices_sha256"] == settings["validation_indices_sha256"]
                and agreement == r["terminal_fixed_discard_agreement"]
                and full["rows"] == full["requested_rows"] == r["validation_manifest"]["rows"]
                and probability["rows"] == probability["requested_rows"] == full["rows"])
        a.check(f"{name}:recomputed_completion_and_durability", r["complete"] == complete and r["durability_passed"] == durable)
        a.check(f"{name}:valid_stop_reason", (r["stop_reason"] == "steps_completed" and complete)
                or (r["stop_reason"] == "safety_alarm" and r["safety"]["tripped"] and not r["failure"]))
        last = r["diagnostics"][-1]
        terminal_path = Path(r["terminal_checkpoint"])
        a.inputs[str(terminal_path)] = digest(terminal_path)
        if r["selected_checkpoint"]:
            selected_path = Path(r["selected_checkpoint"])
            a.inputs[str(selected_path)] = digest(selected_path)
        rows.append({"recipe": recipe, "seed": seed, "completed_steps": r["completed_steps"],
            "stop_reason": r["stop_reason"], "complete": complete, "durability_passed": durable,
            "terminal_fixed_discard_agreement": agreement,
            "terminal_full_discard_agreement": full["equal_stratum_means"]["discard_tie_agreement"],
            "terminal_full_policy_kl": full["equal_stratum_means"]["policy_kl"],
            "selected_step": r["best_step"],
            "selected_full_discard_agreement": (r["selected_full_validation"] or {}).get("equal_stratum_means", {}).get("discard_tie_agreement"),
            "scale_alarm": r["safety"]["scale_alarm"], "safety_tripped": r["safety"]["tripped"],
            "terminal_fixed_context_rms_mean": last["fixed"]["context_rms"]["mean"],
            "terminal_fixed_legal_logit_abs_mean": last["fixed"]["legal_logit_abs"]["mean"],
            "terminal_fixed_centered_legal_logit_abs_mean": last["fixed"]["centered_legal_logit_abs"]["mean"],
            "maximum_fixed_legal_logit_abs": max(d["fixed"]["legal_logit_abs"]["maximum"] for d in r["diagnostics"]),
            "maximum_gradient_norm": max(h["gradient_norm"] for h in r["history"]),
            "probability": probability_summary(probability),
            "sampled_peak_conservative_memory_bytes": r["sampled_peak_conservative_memory_bytes"],
            "sampled_peak_tree_rss_bytes": marker["watchdog"]["sampled_peak_tree_rss_bytes"],
            "historical_full_stream_matches": full_stream})
    pairing = []
    for seed in cfg["seeds"]:
        control, smooth = runs["control", seed], runs["smooth", seed]
        c = {p["step"]: p["sha256"] for p in control["sampled_prefixes"]}
        s = {p["step"]: p["sha256"] for p in smooth["sampled_prefixes"]}
        through = min(control["completed_steps"], smooth["completed_steps"])
        common = sorted(set(c) & set(s))
        expected = list(range(cfg["diagnostic_every"], through + 1, cfg["diagnostic_every"]))
        matches = all(c[step] == s[step] for step in common)
        a.check(f"seed-{seed}:paired_sampler_prefixes", common == expected and bool(common) and matches
                and len(c) == len(control["sampled_prefixes"]) and len(s) == len(smooth["sampled_prefixes"]))
        a.check(f"seed-{seed}:paired_initial_state", control["initial_state_sha256"] == smooth["initial_state_sha256"])
        pairing.append({"seed": seed, "matched_prefix_count": len(common), "through_step": common[-1] if common else None,
                        "all_recorded_shared_prefixes_match": matches})
    operations = ledger["operations"]
    charged = sum(op["elapsed_seconds"] for op in operations)
    formal = [op for op in operations if op["name"].startswith("formal:")]
    a.check("formal_ledger_order_and_count", [op["name"] for op in formal] == [f"formal:{r}:{s}" for r, s in planned])
    a.check("all_ledger_leases_closed_and_charged", all(op["status"] in ("completed", "failed")
            and math.isfinite(op["elapsed_seconds"]) and op["elapsed_seconds"] >= 0 for op in operations))
    a.check("no_lease_reservation_exceeded", all(not op["exceeded_reservation"] and op["elapsed_seconds"] <= op["reserved_seconds"] for op in operations))
    a.check("cumulative_a2_budget", charged <= cfg["limit_seconds"])
    ordered_markers = [markers[item] for item in planned]
    a.check("attempt_timestamp_order", all(x["finished_unix"] <= y["started_unix"] for x, y in zip(ordered_markers, ordered_markers[1:])))
    for i, marker in enumerate([*extra_markers, *ordered_markers]):
        saved = marker["budget"]
        n = len(saved["operations"])
        a.check(f"marker-{i}:ledger_prefix", saved["identity"] == identity and saved["operations"] == operations[:n]
                and close(saved["charged_seconds"], sum(op["elapsed_seconds"] for op in saved["operations"])))
        a.check(f"marker-{i}:watchdog_memory", marker["status"] == "completed" and marker["failure"] is None
                and 0 < marker["watchdog"]["sampled_peak_tree_rss_bytes"] < cfg["memory_bytes"])
    summary = a.load(a1_root / "summary.json")
    confirmation = a.load(a1_root / "confirmation.json")
    ni = a.load(a1_root / "noninferiority-result.json")
    a1_plan = a.load(a1_root / "plan.json")
    a.check("a1_completed_valid_fixed_count", summary["completed"] and summary["within_limits"] and confirmation["valid"]
            and confirmation["completed_blocks"] == confirmation["requested_blocks"] == len(confirmation["blocks"]) == 319
            and confirmation["completed_games"] == len(confirmation["matches"]) == 2552
            and confirmation["failed_games"] == confirmation["truncated_games"] == 0)
    a.check("a1_frozen_plan_identity", digest(a1_root / "plan.json") == summary["plan_sha256"]
            and a1_plan["config_sha256"] == identity and a1_plan["frozen_before_outcomes"]
            and a1_plan["no_pilot_pooling"] and a1_plan["no_candidate_switching"]
            and a1_plan["seed_overlap_count"] == 0 and len(set(a1_plan["wall_seeds"])) == 319)
    a.check("a1_campaign_recorded_unchanged", all(summary[k] for k in
            ("checkpoint_unchanged", "source_unchanged", "config_unchanged", "runner_unchanged"))
            and confirmation["sources_unchanged"] and confirmation["policies_unchanged"]
            and confirmation["match_policies_match_frozen_panel"])
    a.check("a1_original_source_bytes_preserved", all(digest(ROOT / "src/riichi_ai" / name) == h
            for name, h in a1_plan["source_manifest"]["sources"].items()))
    a.check("original_selected_checkpoint_unchanged", digest(a1_plan["checkpoint"]) == a1_plan["checkpoint_sha256"]
            == baseline["checkpoint_sha256"])
    differences = [b["mean_utility_difference"] for b in confirmation["blocks"]]
    a.check("a1_blocks_valid_and_means", all(b["valid"] and b["policies_match_frozen_panel"]
            and len(b["seat_utility_differences"]) == 4
            and close(b["mean_utility_difference"], mean(b["seat_utility_differences"])) for b in confirmation["blocks"]))
    a.check("a1_block_seeds_match_plan", [b["block"] for b in confirmation["blocks"]] == list(range(319))
            and [b["wall_seed"] for b in confirmation["blocks"]] == a1_plan["wall_seeds"])
    average, sd = mean(differences), stdev(differences)
    se = sd / math.sqrt(len(differences))
    lower = average - NormalDist().inv_cdf(1 - a1_cfg["alpha"]) * se
    pvalue = 1 - NormalDist().cdf((average + a1_cfg["margin"]) / se)
    computed = {"mean_utility_difference": average, "sample_sd": sd, "standard_error": se,
                "one_sided_lower_bound": lower, "one_sided_p_value": pvalue}
    a.check("a1_independent_noninferiority_arithmetic", all(close(ni[k], v) for k, v in computed.items())
            and ni["passed"] == (lower > -a1_cfg["margin"]) and ni["valid"] and ni["independent_blocks"] == 319
            and ni["margin"] == a1_cfg["margin"] and ni["alpha"] == a1_cfg["alpha"]
            and summary["noninferiority"] == ni)
    a1_bytes = tree_bytes(a1_root)
    a2_bytes = tree_bytes(data) + tree_bytes(reports)
    a.check("a1_resource_limits", summary["elapsed_seconds"] <= a1_cfg["limit_seconds"]
            and summary["sampled_peak_rss_bytes"] < a1_cfg["memory_bytes"] and a1_bytes <= a1_cfg["artifact_bytes"])
    a.check("a2_artifact_limit", a2_bytes <= cfg["artifact_bytes"])
    return {"schema": "package-a-final-audit-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "valid": all(c["passed"] for c in a.checks), "checks": a.checks, "checked_input_sha256": a.inputs,
        "config_sha256": identity, "a1": {**computed, "status": ni["status"], "noninferiority_passed": ni["passed"],
            "completed_blocks": 319, "completed_games": 2552, "candidate_mean_rank": confirmation["arm_summaries"]["candidate"]["mean_focal_rank"],
            "teacher_mean_rank": confirmation["arm_summaries"]["baseline"]["mean_focal_rank"]},
        "a2_runs": rows, "a2_paired_sampler_checks": pairing,
        "a2_both_smoothing_seeds_durable": all(r["durability_passed"] for r in rows if r["recipe"] == "smooth"),
        "original_selected_probability": probability_summary(baseline["probability_metrics"]),
        "budget": {"a1_charged_seconds": summary["elapsed_seconds"], "a1_limit_seconds": a1_cfg["limit_seconds"],
            "a1_sampled_peak_rss_bytes": summary["sampled_peak_rss_bytes"], "a1_artifact_bytes": a1_bytes,
            "a2_charged_seconds": charged, "a2_remaining_seconds": cfg["limit_seconds"] - charged,
            "a2_limit_seconds": cfg["limit_seconds"], "a2_artifact_bytes": a2_bytes,
            "a2_maximum_conservative_memory_bytes": max(r["sampled_peak_conservative_memory_bytes"] for r in [probe, *runs.values()]),
            "a2_maximum_marker_tree_rss_bytes": max(m["watchdog"]["sampled_peak_tree_rss_bytes"] for m in [*extra_markers, *markers.values()]),
            "a2_operations": [{k: op[k] for k in ("id", "name", "status", "elapsed_seconds", "reserved_seconds", "exceeded_reservation")} for op in operations]},
        "limitations": ["Integrity/protocol validity is separate from non-inferiority or durability success.",
            "Model tensors are not deserialized; frozen-rank and initial-state checks cross-check recorded runtime assertions, diagnostic updates and historical hashes.",
            "Sampler prefixes compare recorded hash chains; this audit does not regenerate training draws or inference.",
            "Memory peaks are sampled, and artifact totals describe the files present at audit time.",
            "The A1 normal approximation is not an exact, distribution-free or sequential guarantee; its non-pass is not proof of inferiority."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/package_a.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/package_a/audit.json")
    args = parser.parse_args()
    try:
        result = audit(args.config)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = {"schema": "package-a-final-audit-v1", "valid": False,
                  "failure": {"type": type(error).__name__, "message": str(error)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
