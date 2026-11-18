"""Independent stdlib audit of saved Stage 3B diagnostic evidence.

Reconstructs selection, probability arithmetic and score-function gradients from
saved records. It does not reload models, run matches or take optimizer steps.
Parameter gradients and backend forwards remain measured worker evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from verify_stage3a_snapshot import verify as verify_previous


ROOT = Path(__file__).resolve().parents[1]
GRID = (
    {"name": "unchanged", "temperature": 1.0, "epsilon": 0.0},
    {"name": "temperature10", "temperature": 10.0, "epsilon": 0.0},
    {"name": "temperature100", "temperature": 100.0, "epsilon": 0.0},
    {"name": "uniform05", "temperature": 1.0, "epsilon": 0.05},
)
CALLS = ("CHI", "PON", "DAIMINKAN", "ANKAN", "KAKAN")
STRATA = ("ANKAN", "DAIMINKAN", "KAKAN", "CHI", "PON", "WIN", "DISCARD", "PASS", "OTHER")
CONSTRAINTS = dict(new_matches=0, optimizer_updates=0, checkpoint_writes=0,
    book_training=False, test_split_loading=False, external_acquisition=False,
    paid_compute=False, strength_evaluation=False, sampling_policy_promotion=False)
# Arithmetic tolerances are independent of CPU/MPS forward differences. No
# cross-backend acceptance threshold is inferred from diagnostic outcomes.
PROBABILITY_ATOL = 2e-12
LOG_ATOL = 2e-9
RAW_GRADIENT_ATOL = 2e-6  # Backcast to float32 network logits is recorded.


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


class Checks:
    def __init__(self):
        self.counts = Counter()
        self.maximum_errors = defaultdict(float)

    def require(self, condition, message, category="integrity"):
        if not condition:
            raise ValueError(message)
        self.counts[category] += 1

    def near(self, actual, expected, message, *, atol=PROBABILITY_ATOL, rtol=2e-12, category="arithmetic"):
        self.require(type(actual) in (float, int) and math.isfinite(actual)
                     and math.isfinite(expected), message + ": nonfinite", category)
        error = abs(actual - expected)
        self.maximum_errors[category] = max(self.maximum_errors[category], error)
        self.require(error <= atol + rtol * abs(expected), message, category)


def distribution(logits, temperature=1.0, epsilon=0.0):
    if not logits or not all(type(z) in (int, float) and math.isfinite(z) for z in logits):
        raise ValueError("Expected finite legal logits")
    scaled = [z / temperature for z in logits]
    maximum = max(scaled)
    shifted = [z - maximum for z in scaled]
    denominator = math.fsum(math.exp(z) for z in shifted)
    log_denominator = math.log(denominator)
    log_base = [z - log_denominator for z in shifted]
    base = [math.exp(z) / denominator for z in shifted]
    actual = [(1 - epsilon) * p + epsilon / len(base) for p in base]
    if epsilon:
        log_actual = [math.log(p) for p in actual]
        attenuation = [(1 - epsilon) * p / q for p, q in zip(base, actual)]
    else:
        log_actual, attenuation = log_base[:], [1.0] * len(base)
    return dict(probabilities=actual, log_probabilities=log_actual,
                base_probabilities=base, base_log_probabilities=log_base,
                mixture_gradient_attenuation=attenuation)


def stats(values):
    if not values:
        return dict(count=0, min=None, median=None, mean=None, max=None)
    values = sorted(values)
    middle = len(values) // 2
    return dict(count=len(values), min=values[0], max=values[-1],
                median=values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2,
                mean=math.fsum(values) / len(values))


def check_stats(actual, values, checked, label):
    expected = stats(values)
    checked.require(set(actual) == set(expected), f"{label}: statistic keys")
    for key, value in expected.items():
        if value is None or key == "count":
            checked.require(actual[key] == value, f"{label}: {key}")
        else:
            checked.near(actual[key], value, f"{label}: {key}", atol=LOG_ATOL, category="summary")


def round_robin(rows, maximum):
    groups = {kind: sorted((row for row in rows if row["stratum"] == kind),
                           key=lambda row: (row["selection_key"], row["id"])) for kind in STRATA}
    selected = []
    while len(selected) < maximum and any(groups.values()):
        for kind in STRATA:
            if groups[kind] and len(selected) < maximum:
                selected.append(groups[kind].pop(0))
    return selected


def audit_selection(selection, config, checked):
    root = Path(config["source_root"]).resolve()
    paths = sorted((root / "trajectories").rglob("match-*.json"))
    checked.require(len(paths) == 64, "Expected exactly the 64 preserved development matches")
    rows, sources = [], []
    for path in paths:
        record = read(path)
        checked.require(record["completed"] is True and record["failure"] is None
                        and record["schema"] == "stage3a-complete-match-reinforce-v1",
                        "Unexpected source match")
        arrays = Path(record["encoded_artifact"]["path"]).resolve()
        checked.require(arrays.is_relative_to(root / "trajectories")
                        and digest(arrays) == record["encoded_artifact"]["sha256"],
                        "Saved actor source identity differs")
        sources.append(dict(json_path=str(path.resolve()), json_sha256=digest(path),
                            npz_path=str(arrays), npz_sha256=digest(arrays)))
        for index, row in enumerate(record["learner_rows"]):
            checked.require(row["row_index"] == index, "Source row indexing differs")
            if row["forced"]:
                continue
            checked.require(row["eligible"] and not row["fallback"] and not row["overridden"],
                            "Ineligible source actor row")
            kinds = [json.loads(key)[1] for key in row["semantic_keys"]]
            stratum = next((kind for kind in STRATA[:5] if kind in kinds), None)
            if stratum is None:
                stratum = ("WIN" if "RON" in kinds or "TSUMO" in kinds else
                           "DISCARD" if "DISCARD" in kinds else "PASS" if "PASS" in kinds else "OTHER")
            target_kind = (stratum if stratum in CALLS else "RON" if "RON" in kinds else
                           "TSUMO" if "TSUMO" in kinds else "PASS" if stratum == "PASS" else None)
            target = kinds.index(target_kind) if target_kind else row["selected_index"]
            identity = f"batch-{record['batch_index']}/match-{record['match_index']:03d}/row-{index:05d}"
            rows.append(dict(id=identity, stratum=stratum,
                selection_key=object_hash(["stage3b-row-selection-v1", identity]),
                json_path=str(path.resolve()), npz_path=str(arrays), row_index=index,
                match_index=record["match_index"], batch_index=record["batch_index"], seat=row["seat"],
                policy_id=row["policy_id"], player_view_sha256=row["player_view_sha256"],
                encoded_sha256=row["encoded_sha256"], semantic_keys=row["semantic_keys"], action_kinds=kinds,
                selected_index=row["selected_index"], target_index=target, target_kind=kinds[target]))
    expected = round_robin(rows, config["maximum_rows"])
    probes = round_robin(expected, config["probe_rows"])
    checked.require(selection["sources"] == sources and selection["rows"] == expected,
                    "Frozen source/row selection does not reproduce independently", "selection")
    checked.require(len({row["id"] for row in expected}) == len(expected) <= 512,
                    "Duplicate or excessive row selection", "selection")
    checked.require(selection["probe_ids"] == [row["id"] for row in probes]
                    and len(probes) <= 32, "Frozen probes do not reproduce", "selection")
    expected_counts = dict(
        optional_rows_available=len(rows), rows_selected=len(expected),
        available_by_stratum=dict(Counter(row["stratum"] for row in rows)),
        selected_by_stratum=dict(Counter(row["stratum"] for row in expected)),
        selected_offered_by_kind=dict(Counter(kind for row in expected for kind in set(row["action_kinds"]))),
        probe_targets_by_kind=dict(Counter(row["target_kind"] for row in probes)))
    checked.require(all(selection.get(key) == value for key, value in expected_counts.items())
                    and selection["selection_uses_rewards_or_model_outputs"] is False
                    and selection["stratum_priority"] == list(STRATA)
                    and selection["grid"] == list(GRID), "Selection metadata differs", "selection")
    return {row["id"]: row for row in expected}


def audit_row(row, selected, checked):
    for key in ("id", "stratum", "action_kinds", "semantic_keys", "target_index", "target_kind"):
        checked.require(row[key] == selected[key], f"Measured row differs: {key}", "rows")
    checked.require(row["recorded_policy_id"] == selected["policy_id"]
                    and row["recorded_selected_index"] == selected["selected_index"], "Behavior provenance differs", "rows")
    logits, kinds = row["logits"], row["action_kinds"]
    checked.require(len(logits) == len(kinds) > 1 and row["candidate_mask"] == [True] * len(logits)
                    and row["mask_all_saved_candidates_legal"] is True
                    and len(row["gaps_to_best_logit"]) == len(logits), "Legal mask or candidate shape differs", "rows")
    checked.require(set(row["variants"]) == {option["name"] for option in GRID}, "Unexpected diagnostic alternatives")
    base = distribution(logits)["probabilities"]
    maximum = max(logits)
    for actual, value in zip(row["gaps_to_best_logit"], logits):
        checked.near(actual, maximum - value, "Legal logit gap differs", atol=LOG_ATOL)
    checked.near(row["logit_range"], maximum - min(logits), "Logit range differs", atol=LOG_ATOL)
    for kind in set(kinds):
        values = [z for z, k in zip(logits, kinds) if k == kind]
        check_stats(row["by_kind"][kind]["logits"], values, checked, "Kind logits")
        check_stats(row["by_kind"][kind]["gap_to_best"], [maximum - z for z in values], checked, "Kind gaps")
    for option in GRID:
        actual = row["variants"][option["name"]]
        expected = distribution(logits, option["temperature"], option["epsilon"])
        for key in ("probabilities", "log_probabilities", "base_log_probabilities", "mixture_gradient_attenuation"):
            checked.require(len(actual[key]) == len(logits), "Probability array length differs")
            for left, right in zip(actual[key], expected[key]):
                if key == "probabilities":
                    checked.require((left == 0) == (right == 0),
                                    "Exact float64 probability support differs from saved logits",
                                    "probability_support")
                checked.near(left, right, f"{key} differs from saved legal logits",
                             atol=LOG_ATOL if "log_probabilities" in key else PROBABILITY_ATOL,
                             category="probability")
        p = actual["probabilities"]
        checked.near(math.fsum(p), 1.0, "Probability mass does not normalize")
        checked.require(all(value >= 0 for value in p), "Negative probability")
        zero = sum(value == 0 for value in p)
        checked.require(actual["legal_candidates"] == len(p)
                        and actual["zero_probability_candidates"] == zero
                        and actual["underflow_with_finite_log_probability"] == zero,
                        "Support counters differ")
        checked.near(actual["entropy_nats"], -math.fsum(value * math.log(value) for value in p if value),
                     "Entropy differs")
        checked.near(actual["maximum_probability"], max(p), "Maximum probability differs")
        checked.near(actual["minimum_positive_probability"], min(value for value in p if value),
                     "Minimum positive probability differs")
        checked.near(actual["l1_from_same_checkpoint_unchanged"], math.fsum(abs(a - b) for a, b in zip(p, base)),
                     "Distribution displacement differs")
        checked.near(actual["total_variation_from_same_checkpoint_unchanged"],
                     actual["l1_from_same_checkpoint_unchanged"] / 2, "Total variation differs")
        winner = max(range(len(base)), key=lambda i: base[i])
        checked.require(actual["unchanged_argmax_index"] == winner
                        and actual["argmax_index_preserved"] == (max(range(len(p)), key=lambda i: p[i]) == winner),
                        "Original winner identity differs")
        checked.near(actual["unchanged_argmax_probability"], p[winner], "Original winner probability differs")
        cumulative, zero_width = 0.0, 0
        for value in p:
            following = cumulative + value
            zero_width += value > 0 and following == cumulative
            cumulative = following
        checked.require(actual["positive_probability_zero_width_float64_cdf_increments"] == zero_width
                        and actual["positive_probabilities_below_2_minus53"] == sum(0 < value < 2**-53 for value in p),
                        "Small-probability/CDF width counts differ")
        checked.near(actual["call_kan_mass"], math.fsum(value for value, kind in zip(p, kinds) if kind in CALLS),
                     "Call mass differs")
        for kind in set(kinds):
            values = [value for value, other in zip(p, kinds) if other == kind]
            group = actual["by_kind"][kind]
            checked.require(group["candidates"] == len(values)
                            and group["zero_probability_candidates"] == sum(value == 0 for value in values),
                            "Per-kind support differs")
            checked.near(group["mass"], math.fsum(values), "Per-kind mass differs")


def audit_gradient(probe, row, checked):
    target = row["target_index"]
    checked.require(probe["id"] == row["id"] and probe["target_index"] == target
                    and probe["target_kind"] == row["target_kind"], "Gradient target differs", "gradients")
    checked.require(set(probe["variants"]) == {option["name"] for option in GRID}, "Gradient grid differs")
    for option in GRID:
        item = probe["variants"][option["name"]]
        dist = distribution(row["logits"], option["temperature"], option["epsilon"])
        for key, field in (("target_probability", "probabilities"), ("target_log_probability", "log_probabilities"),
                           ("target_base_log_probability", "base_log_probabilities"),
                           ("target_mixture_gradient_attenuation", "mixture_gradient_attenuation")):
            checked.near(item[key], dist[field][target], "Gradient target distribution differs",
                         atol=LOG_ATOL if "log_probability" in key else PROBABILITY_ATOL, category="gradients")
        analytic = [dist["mixture_gradient_attenuation"][target] * ((i == target) - p) / option["temperature"]
                    for i, p in enumerate(dist["base_probabilities"])]
        checked.require(len(item["logit_gradients"]) == len(analytic)
                        and item["gradients_finite"] is True, "Invalid gradient shape/finite status", "gradients")
        for actual, expected in zip(item["logit_gradients"], analytic):
            checked.near(actual, expected, "Score-function gradient differs", atol=RAW_GRADIENT_ATOL,
                         rtol=2e-6, category="gradients")
        normalization = item["normalization_float64_logit_gradients"]
        checked.require(len(normalization) == len(analytic), "Normalization gradient shape differs")
        for actual, expected in zip(normalization, analytic):
            checked.near(actual, expected, "Float64 normalization score-function gradient differs",
                         atol=PROBABILITY_ATOL, category="normalization_gradients")
        checked.near(item["normalization_float64_logit_gradient_l2"], math.hypot(*normalization),
                     "Float64 normalization gradient norm differs", category="normalization_gradients")
        raw = item["logit_gradients"]
        checked.near(item["logit_gradient_l2"], math.sqrt(math.fsum(g * g for g in raw)),
                     "Raw gradient norm differs", atol=LOG_ATOL, category="gradients")
        checked.near(item["logit_gradient_max_abs"], max(abs(g) for g in raw),
                     "Raw gradient max differs", category="gradients")
        parameters = item["parameters"]
        checked.require(bool(parameters) and not any(name.startswith("rank_head.") for name in parameters),
                        "Frozen rank head unexpectedly participates in actor gradients", "gradients")
        groups = defaultdict(list)
        for name, entry in parameters.items():
            checked.require(type(entry["elements"]) is int and entry["elements"] > 0
                            and type(entry["nonzero_elements"]) is int
                            and 0 <= entry["nonzero_elements"] <= entry["elements"]
                            and all(type(entry[k]) in (int, float) and math.isfinite(entry[k]) and entry[k] >= 0
                                    for k in ("l2_norm", "max_abs")), "Invalid parameter gradient statistics", "gradients")
            checked.require((entry["l2_norm"] == 0) == (entry["nonzero_elements"] == 0)
                            and entry["max_abs"] <= entry["l2_norm"] + LOG_ATOL,
                            "Inconsistent parameter gradient support", "gradients")
            if entry["connected"]:
                groups[name.split(".")[0]].append(entry)
            else:
                checked.require(entry["l2_norm"] == entry["max_abs"] == entry["nonzero_elements"] == 0,
                                "Disconnected parameter has nonzero gradient", "gradients")
        norm = math.sqrt(math.fsum(value["l2_norm"] ** 2 for value in parameters.values()))
        checked.near(item["parameter_gradient_l2"], norm, "Total parameter norm differs", atol=LOG_ATOL, category="gradients")
        checked.require(item["parameter_gradient_exactly_zero"] == (norm == 0)
                        and item["parameter_gradient_below_1e_minus12"] == (norm < 1e-12),
                        "Parameter-gradient support classification differs", "gradients")
        checked.require(set(item["parameter_groups"]) == set(groups), "Parameter group membership differs")
        for name, entries in groups.items():
            group = item["parameter_groups"][name]
            checked.near(group["l2_norm"], math.sqrt(math.fsum(e["l2_norm"] ** 2 for e in entries)),
                         "Parameter group norm differs", atol=LOG_ATOL, category="gradients")
            checked.near(group["max_abs"], max(e["max_abs"] for e in entries), "Parameter group maximum differs")
            checked.require(all(group[k] == sum(e[k] for e in entries) for k in ("elements", "nonzero_elements")),
                            "Parameter group totals differ")


def audit_summary(summary, rows, gradients, checked):
    checked.require(set(summary) == {option["name"] for option in GRID}, "Summary grid differs")
    for option in GRID:
        name = option["name"]
        item = summary[name]
        values = [row["variants"][name] for row in rows]
        probes = [row["variants"][name] for row in gradients]
        checked.require(item["rows"] == len(rows) and item["gradient_probes"] == len(probes),
                        "Summary counts differ")
        for field in ("entropy_nats", "l1_from_same_checkpoint_unchanged"):
            check_stats(item[field], [value[field] for value in values], checked, field)
        for field, source in (("ordinary_discard_l1", "l1_from_same_checkpoint_unchanged"),
                              ("ordinary_discard_total_variation", "total_variation_from_same_checkpoint_unchanged"),
                              ("ordinary_discard_unchanged_argmax_probability", "unchanged_argmax_probability")):
            check_stats(item[field], [row["variants"][name][source] for row in rows if row["stratum"] == "DISCARD"],
                        checked, field)
        for field in ("zero_probability_candidates", "underflow_with_finite_log_probability",
                      "positive_probabilities_below_2_minus53", "positive_probability_zero_width_float64_cdf_increments"):
            checked.require(item[field] == sum(value[field] for value in values), f"Summary {field} differs")
        checked.require(item["rows_with_maximum_probability_at_least_0999"]
                        == sum(value["maximum_probability"] >= .999 for value in values),
                        "Concentrated row count differs")
        checked.require(item["gradient_probes_exact_zero_parameters"]
                        == sum(value["parameter_gradient_exactly_zero"] is True for value in probes)
                        and item["gradient_probes_parameters_below_1e_minus12"]
                        == sum(value["parameter_gradient_below_1e_minus12"] is True for value in probes)
                        and item["all_gradient_probes_finite"] == all(value["gradients_finite"] for value in probes),
                        "Gradient summary status differs")
        check_stats(item["parameter_gradient_l2"], [value["parameter_gradient_l2"] for value in probes
                                                   if value["parameter_gradient_l2"] is not None],
                    checked, "Parameter gradient norm summary")
        all_kinds = {kind for row in rows for kind in row["action_kinds"]}
        checked.require(set(item["by_kind"]) == all_kinds, "Summary action kinds differ")
        for kind in all_kinds:
            group = item["by_kind"][kind]
            groups = [value["by_kind"][kind] for value in values if kind in value["by_kind"]]
            checked.require(group["offered_rows"] == len(groups)
                            and group["candidates"] == sum(value["candidates"] for value in groups)
                            and group["zero_probability_candidates"] == sum(value["zero_probability_candidates"] for value in groups),
                            "Summary kind counts differ")
            check_stats(group["probability_mass_when_offered"], [value["mass"] for value in groups],
                        checked, "Offered probability mass")
            check_stats(group["gap_to_best_logit"], [gap for row in rows for gap, action_kind
                        in zip(row["gaps_to_best_logit"], row["action_kinds"]) if action_kind == kind],
                        checked, "Offered logit gap")


def audit_comparison(report, left, right, checked):
    lookup = {row["id"]: row for row in left}
    checked.require([row["id"] for row in report["rows"]] == [row["id"] for row in right],
                    "Backend/checkpoint comparison row identity differs")
    deltas = defaultdict(list)
    raw_deltas = []
    for item, row in zip(report["rows"], right):
        reference = lookup[row["id"]]
        maximum = max(abs(a - b) for a, b in zip(reference["logits"], row["logits"]))
        raw_deltas.append(maximum)
        checked.near(item["max_logit_difference"], maximum, "Compared logit difference differs", atol=LOG_ATOL)
        for option in GRID:
            name = option["name"]
            differences = [abs(a - b) for a, b in zip(reference["variants"][name]["probabilities"],
                                                      row["variants"][name]["probabilities"])]
            checked.near(item["variants"][name]["max_probability_difference"], max(differences),
                         "Compared probability maximum differs")
            checked.near(item["variants"][name]["l1_probability_difference"], math.fsum(differences),
                         "Compared probability L1 differs")
            deltas[name].append(differences)
    checked.require(report["max_logit_difference"] == max(raw_deltas, default=None), "Comparison maximum differs")
    for option in GRID:
        name = option["name"]
        values = deltas[name]
        checked.require(report["variants"][name]["max_probability_difference"]
                        == max((max(d) for d in values), default=None), "Compared maximum differs")
        check_stats(report["variants"][name]["l1_probability_difference"],
                    [sum(d) for d in values], checked, "Comparison L1 summary")


def audit_repair(reports, plan, diagnostic, identity, checked):
    repair_path = reports / "repair-plan.json"
    repair = read(repair_path)
    checked.require(repair["original_plan_sha256"] == digest(reports / "plan.json")
                    and repair["config_sha256"] == identity
                    and repair["implementation_before"] == plan["implementation"]
                    and repair["implementation_after"] == diagnostic["implementation"]
                    and diagnostic["repair_plan_sha256"] == digest(repair_path),
                    "Transfer repair identities differ")
    before, after = repair["implementation_before"], repair["implementation_after"]
    changed = [name for name in before if before[name] != after.get(name)]
    checked.require(set(before) == set(after) and changed == ["src/riichi_ai/stage3b.py"],
                    "Numerical repair exceeds the single declared diagnostic source")
    checked.require(repair["wrapper_sha256"] == digest(ROOT / "scripts/stage3b_repaired_run.py"),
                    "Repair wrapper changed")
    for evidence in repair["evidence"]:
        checked.require(digest(ROOT / evidence["path"]) == evidence["sha256"], "Preserved repair evidence changed")
    recovered = read(reports / "diagnostic-recovery.json")
    checked.require(recovered["plan_sha256"] == digest(reports / "plan.json")
                    and recovered["implementation"] == before and recovered["config_sha256"] == identity
                    and recovered["selection_sha256"] == diagnostic["selection_sha256"]
                    and recovered["grid"] == diagnostic["grid"], "Recovery selection/grid differs")
    recovery = recovered["recovery"]
    checked.require(recovery["wrapper_sha256"] == digest(ROOT / "scripts/stage3b_recover.py")
                    and recovery["preserved_failed_attempt_sha256"] == digest(reports / "diagnose-attempt.json")
                    and recovery["kind"] == "serialization-only-wrapper"
                    and recovery["selection_and_grid_unchanged"] is True
                    and recovery["numerical_measurements_repeated"] is True, "Recovery provenance differs")
    null_paths = set()
    def visit(value, path="$"):
        if value is None:
            null_paths.add(path)
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
    visit(recovered)
    anomalies = recovery["nonfinite_metrics"]
    checked.require(bool(anomalies) and len({entry["path"] for entry in anomalies}) == len(anomalies),
                    "Nonfinite recovery evidence is absent or duplicated")
    for entry in anomalies:
        checked.require(entry["path"] in null_paths and entry["value"] in ("inf", "-inf", "nan")
                        and (".mps_gradients[" in entry["path"] or ".mps_summary." in entry["path"]),
                        "Nonfinite finding is not preserved at its MPS metric path")
    checked.require(diagnostic["cpu_results_exactly_match_recovery"] is True
                    and all(left[key] == right[key] for left, right in zip(
                        diagnostic["checkpoints"], recovered["checkpoints"])
                        for key in ("cpu_rows", "cpu_gradients", "cpu_summary")),
                    "Transfer repair changed the CPU experiment")
    transfer_path = reports / "mps-transfer-probe.json"
    transfer = read(transfer_path)
    checked.require(transfer["schema"] == "stage3b-mps-transfer-probe-v1"
                    and transfer["config_sha256"] == identity
                    and transfer["plan_sha256"] == digest(reports / "plan.json")
                    and transfer["script_sha256"] == digest(ROOT / "scripts/stage3b_mps_transfer_probe.py")
                    and transfer["row_id"] == plan["selection"]["probe_ids"][0]
                    and transfer["checkpoint"] == plan["checkpoints"][0],
                    "Transfer fixture provenance differs")
    for name in ("known_tensor", "known_noncontiguous_tensor", "logits"):
        item = transfer[name]
        original, staged, direct = (item[key] for key in
            ("cpu_original_dtype", "staged_cpu_float64", "direct_cpu_float64"))
        checked.require(original["first_values"] == staged["first_values"]
                        and original["nonfinite_elements"] == staged["nonfinite_elements"] == 0
                        and original["first_values"] != direct["first_values"]
                        and item["direct_equals_staged"] is False,
                        "Focused transfer fixture does not substantiate the repair")
    checked.require(transfer["known_tensor"]["cpu_original_dtype"]["first_values"][:4] == [0., 1., -1., .125]
                    and transfer["known_tensor"]["cpu_original_dtype"]["first_values"][-1] == 10000.,
                    "Known transfer-control values changed")
    return dict(repair_plan_sha256=digest(repair_path), original_formal_attempt_status="failed",
                recovery_sha256=digest(reports / "diagnostic-recovery.json"),
                recovery_nonfinite_metric_paths=len(anomalies),
                recovered_cpu_values_exactly_preserved=True,
                recovered_mps_inference_and_gradients_valid=False,
                recovered_mps_limitation="Combined MPS-to-CPU float64 transfer corrupted values in the pinned environment.",
                transfer_fixture_sha256=digest(transfer_path))


def audit_budget(config, reports, identity, checked):
    ledger = read(Path(config["data_root"]) / "budget-ledger.json")
    saved = read(reports / "budget.json")
    checked.require(ledger["schema"] == "package-a2-budget-v1" and ledger["identity"] == identity
                    and ledger["limit_seconds"] == 300
                    and all(saved.get(key) == value for key, value in ledger.items()), "Budget identity differs")
    operations = ledger["operations"]
    checked.require([op["id"] for op in operations] == list(range(len(operations)))
                    and len({op["name"] for op in operations}) == len(operations), "Duplicate numerical operation")
    peak_memory, peak_artifacts = 0, 0
    for operation in operations:
        elapsed, reserved = operation["elapsed_seconds"], operation["reserved_seconds"]
        checked.require(operation["status"] in ("completed", "failed") and math.isfinite(elapsed)
                        and math.isfinite(reserved) and 0 <= elapsed <= reserved
                        and operation["exceeded_reservation"] is False, "Unclosed or excessive operation")
        name = operation["name"]
        filename = f"checks-{name[6:]}.json" if name.startswith("check:") else f"{name}-attempt.json"
        marker = read(reports / filename)
        checked.require(marker["name"] == name and marker["status"] == operation["status"]
                        and marker["config_sha256"] == identity, "Operation marker differs")
        watchdog = marker["watchdog"]
        checked.require(watchdog["sampled_peak_tree_rss_bytes"] < config["memory_bytes"]
                        and watchdog["artifact_bytes_peak"] < config["artifact_bytes"], "Resource cap exceeded")
        peak_memory = max(peak_memory, watchdog["sampled_peak_tree_rss_bytes"])
        peak_artifacts = max(peak_artifacts, watchdog["artifact_bytes_peak"])
    formal = [op for op in operations if not op["name"].startswith("check:")]
    checked.require([op["name"] for op in formal] == ["prepare", "diagnose"]
                    and [op["status"] for op in formal] == ["completed", "failed"],
                    "Original preparation/failure sequence differs")
    by_name = {op["name"]: op for op in operations}
    checked.require(all(by_name[name]["status"] == "completed" for name in (
        "check:serialization-recovery", "check:mps-transfer-probe", "check:repaired-diagnostic")),
        "Recovery, transfer fixture or repaired diagnostic did not complete")
    charged = sum(op["elapsed_seconds"] for op in operations)
    checked.require(charged <= 300 and saved["charged_seconds"] == charged
                    and saved["remaining_seconds"] == 300 - charged, "Budget totals differ")
    return dict(limit_seconds=300, charged_seconds=charged, remaining_seconds=300 - charged,
                operations=len(operations), all_closed=True,
                completed_operations=sum(op["status"] == "completed" for op in operations),
                failed_operations=sum(op["status"] == "failed" for op in operations),
                sampled_peak_tree_rss_bytes=peak_memory, sampled_peak_artifact_bytes=peak_artifacts)


def audit(config_path, expected_config_sha256):
    checked = Checks()
    identity = digest(config_path)
    checked.require(identity == expected_config_sha256, "Expected configuration identity differs")
    config = read(config_path)
    reports = ROOT / config["reports_root"]
    plan, diagnostic = read(reports / "plan.json"), read(reports / "diagnostic.json")
    checked.require(config["limit_seconds"] == 300 and config["maximum_rows"] == 512
                    and config["probe_rows"] == 32 and config["temperatures"] == [1., 10., 100.]
                    and config["mixture_epsilon"] == .05 and config["torch_threads"] == 1,
                    "Authorized diagnostic limits changed")
    checked.require(config["constraints"] == plan["constraints"] == diagnostic["constraints"] == CONSTRAINTS,
                    "Authorized zero-update scope differs")
    checked.require(plan["config_sha256"] == diagnostic["config_sha256"] == identity
                    and diagnostic["plan_sha256"] == digest(reports / "plan.json")
                    and plan["frozen_before_diagnostic_outputs"] is True, "Frozen plan identity differs")
    repair = audit_repair(reports, plan, diagnostic, identity, checked)
    for path, expected in diagnostic["implementation"].items():
        checked.require(digest(ROOT / path) == expected, f"Numerical source changed: {path}", "identity")
    previous = verify_previous()
    checked.require(read(reports / "previous-stage3a-snapshot.json") == previous, "Historical snapshot evidence differs")
    checked.require(diagnostic["schema"] == "stage3b-action-support-diagnostic-v1"
                    and diagnostic["grid"] == list(GRID) and diagnostic["cpu_complete"] is True
                    and diagnostic["new_matches"] == diagnostic["optimizer_updates"] == diagnostic["checkpoint_writes"] == 0,
                    "Diagnostic scope/outcome differs")
    selection = plan["selection"]
    checked.require(diagnostic["selection_sha256"] == object_hash(selection), "Selection hash differs")
    selected = audit_selection(selection, config, checked)
    checked.require(diagnostic["rows"] == len(selected) and diagnostic["gradient_rows_per_checkpoint_and_device"]
                    == len(selection["probe_ids"]), "Measured cohort size differs")
    checked.require(diagnostic["cpu_rng_unchanged"] is True
                    and diagnostic["cpu_rng_before_sha256"] == diagnostic["cpu_rng_after_sha256"], "CPU RNG changed")
    checkpoints = diagnostic["checkpoints"]
    checked.require(len(checkpoints) == len(config["checkpoints"]) == 2
                    and plan["checkpoints"] == config["checkpoints"], "Checkpoint grid differs")
    mps_complete = diagnostic["mps"]["status"] == "completed"
    for checkpoint, configured in zip(checkpoints, config["checkpoints"]):
        checked.require(all(checkpoint[key] == value for key, value in configured.items())
                        and digest(configured["path"]) == configured["sha256"], "Checkpoint bytes/identity changed", "identity")
        checked.require(checkpoint["parameters_unchanged"] is True and checkpoint["parameter_grad_fields_unchanged"] is True
                        and checkpoint["weights_before_sha256"] == checkpoint["weights_after_sha256"],
                        "CPU parameters changed")
        checked.require([row["id"] for row in checkpoint["cpu_rows"]] == list(selected)
                        and [row["id"] for row in checkpoint["cpu_gradients"]] == selection["probe_ids"],
                        "CPU measurement identities differ")
        for backend in ("cpu", "mps"):
            rows = checkpoint.get(f"{backend}_rows", [])
            gradients = checkpoint.get(f"{backend}_gradients", [])
            if backend == "mps":
                checked.require([row["id"] for row in rows] == selection["probe_ids"][:len(rows)]
                                and [row["id"] for row in gradients] == selection["probe_ids"][:len(gradients)],
                                "MPS probe sequence differs")
                if mps_complete:
                    checked.require(len(rows) == len(gradients) == len(selection["probe_ids"])
                                    and checkpoint["mps_parameters_unchanged"] is True
                                    and checkpoint["mps_parameter_grad_fields_unchanged"] is True
                                    and checkpoint["mps_parameters_before_sha256"] == checkpoint["mps_parameters_after_sha256"]
                                    == checkpoint["weights_before_sha256"], "MPS state/coverage differs")
            for row in rows:
                audit_row(row, selected[row["id"]], checked)
            lookup = {row["id"]: row for row in rows}
            for gradient in gradients:
                audit_gradient(gradient, lookup[gradient["id"]], checked)
            if f"{backend}_summary" in checkpoint:
                audit_summary(checkpoint[f"{backend}_summary"], rows, gradients, checked)
            if backend == "mps" and rows:
                audit_comparison(checkpoint["cpu_vs_mps"], checkpoint["cpu_rows"], rows, checked)
    audit_comparison(diagnostic["initial_vs_final_cpu"], checkpoints[0]["cpu_rows"], checkpoints[1]["cpu_rows"], checked)
    if mps_complete:
        checked.require(diagnostic["mps"]["available"] is True and diagnostic["mps"]["rng_unchanged"] is True
                        and diagnostic["mps"]["rng_before_sha256"] == diagnostic["mps"]["rng_after_sha256"],
                        "MPS RNG or availability evidence differs")
    budget = audit_budget(config, reports, identity, checked)
    return dict(schema="stage3b-independent-audit-v1", created_utc=datetime.now(timezone.utc).isoformat(),
        valid=True, config_sha256=identity, plan_sha256=digest(reports / "plan.json"),
        diagnostic_sha256=digest(reports / "diagnostic.json"), implementation=diagnostic["implementation"],
        repair_plan_sha256=repair["repair_plan_sha256"], repair=repair,
        previous_stage3a_verified_files=previous["verified_files"], rows=len(selected),
        gradient_rows_per_checkpoint_and_device=len(selection["probe_ids"]), checkpoints=len(checkpoints),
        mps_status=diagnostic["mps"]["status"], complete=diagnostic["complete"], budget=budget,
        checks=dict(sorted(checked.counts.items())), check_count=sum(checked.counts.values()),
        maximum_arithmetic_errors=dict(checked.maximum_errors),
        tolerances=dict(probability_absolute=PROBABILITY_ATOL, log_absolute=LOG_ATOL,
                        raw_float32_gradient_absolute=RAW_GRADIENT_ATOL),
        limitations=["No playing-strength inference or policy promotion.",
            "Saved-logit normalization, selection, score-function gradients and norm aggregation are independently reconstructed; model forwards and parameter gradients remain measured worker evidence.",
            "CPU/MPS differences are descriptive; there is no post-hoc backend parity threshold.",
            "Checks are data-level assertions, not independent statistical trials."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/stage3b.json"))
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = audit(args.config, args.expected_config_sha256)
    if args.output:
        with Path(args.output).open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
