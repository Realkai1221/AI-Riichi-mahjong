"""Audit completed development shards and optionally replay prespecified matches.

Bulk checks cover every provenance row and RNG draw. Complete-match replays
verify the selected matches' actor arrays and actual engine responses. Reusing
the same engine is reproducibility evidence, not independent rule certification.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np

from riichi_ai.stage3f_collect import BEHAVIORS, EXPLORATION, SCHEMA, SAMPLER, run_one, validate_plan
from riichi_ai.stage3f_stream import StreamDataset
from riichi_ai.stage3c import CALL_KINDS, call_resolutions, terminal_result
from riichi_ai.semantics import ACTION_KINDS

FAMILIES = {"DISCARD": "discard", "CHI": "call", "PON": "call", "DAIMINKAN": "call",
    "ANKAN": "call", "KAKAN": "call", "RIICHI": "riichi", "RON": "win", "TSUMO": "win",
    "PASS": "pass", "KYUSHU_KYUHAI": "abort"}
PROBABILITY_BOUNDS = (0., .001, .01, .05, .1, .25, .5, .75, .95, 1.)


def empty_stats() -> dict:
    return {"count": 0, "sum": 0., "minimum": None, "maximum": None,
            "histogram": [0] * (len(PROBABILITY_BOUNDS) - 1)}


def add_stat(stats: dict, value: float) -> None:
    require(math.isfinite(value) and -1e-12 <= value <= 1 + 1e-12, "Probability summary out of range")
    stats["count"] += 1
    stats["sum"] += value
    stats["minimum"] = value if stats["minimum"] is None else min(stats["minimum"], value)
    stats["maximum"] = value if stats["maximum"] is None else max(stats["maximum"], value)
    index = len(PROBABILITY_BOUNDS) - 2
    for candidate, bound in enumerate(PROBABILITY_BOUNDS[1:]):
        if value < bound:
            index = candidate
            break
    stats["histogram"][index] += 1


def empty_coverage() -> dict:
    counts = {key: 0 for key in ("rows", "semantic_forced_rows", "semantic_optional_rows",
        "legal_candidates", "positive_probability_candidates", "optional_rows_with_full_positive_support",
        "call_opportunity_rows", "optional_call_opportunity_rows", "multiple_call_candidate_rows",
        "call_and_noncall_rows", "selected_call_rows", "resolved_selected_call_rows",
        "selected_calls_among_multiple_call_candidates", "resolved_calls_among_multiple_call_candidates",
        "post_call_rows", "optional_post_call_rows", "open_hand_rows", "history_overflow_rows",
        "maximum_history_prefix_length", "win_opportunity_rows", "win_call_competition_rows")}
    counts.update({key: {} for key in ("chosen_action_counts", "candidate_action_opportunities",
        "chosen_action_families", "candidate_family_opportunities", "requested_calls", "resolved_calls",
        "legal_candidate_count_histogram", "call_candidate_count_histogram")})
    counts.update({key: empty_stats() for key in ("selected_behavior_probability", "actual_call_mass_on_call_opportunities",
                                                "actual_win_mass_on_win_opportunities")})
    return counts


def merge_coverage(destination: dict, source: dict) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and "histogram" in value:
            stats = destination[key]
            stats["count"] += value["count"]
            stats["sum"] += value["sum"]
            if value["minimum"] is not None:
                stats["minimum"] = value["minimum"] if stats["minimum"] is None else min(stats["minimum"], value["minimum"])
                stats["maximum"] = value["maximum"] if stats["maximum"] is None else max(stats["maximum"], value["maximum"])
            stats["histogram"] = [a + b for a, b in zip(stats["histogram"], value["histogram"])]
        elif isinstance(value, dict):
            counts = Counter(destination[key])
            counts.update(value)
            destination[key] = dict(sorted(counts.items()))
        elif key == "maximum_history_prefix_length":
            destination[key] = max(destination[key], value)
        else:
            destination[key] += value


def finalize_coverage(coverage: dict) -> dict:
    for value in coverage.values():
        if isinstance(value, dict) and "histogram" in value:
            value["mean"] = value["sum"] / value["count"] if value["count"] else None
            value["bin_edges"] = list(PROBABILITY_BOUNDS)
            value["bin_convention"] = "left-inclusive/right-exclusive, last bin includes 1"
    return coverage


def describe_coverage(rows: list[dict], actor: dict) -> dict:
    """Describe support and selected experience separately, with no outcome filter."""
    result = {name: empty_coverage() for name in BEHAVIORS}
    for index, row in enumerate(rows):
        coverage = result[BEHAVIORS[row["behavior_id"]]]
        kinds, chosen = row["action_kinds"], row["selected_kind"]
        n = len(kinds)
        p = np.asarray(row["probabilities"], dtype=np.float64)
        calls = np.asarray([kind in CALL_KINDS for kind in kinds])
        wins = np.asarray([kind in ("RON", "TSUMO") for kind in kinds])
        count_calls = int(calls.sum())
        chosen_call = chosen in CALL_KINDS
        resolved = row["requested_call_resolved"] is True
        prefix = int(actor["history_prefix_length"][index])
        coverage["rows"] += 1
        coverage["semantic_forced_rows"] += int(n == 1)
        coverage["semantic_optional_rows"] += int(n > 1)
        coverage["legal_candidates"] += n
        coverage["positive_probability_candidates"] += int((p > 0).sum())
        coverage["optional_rows_with_full_positive_support"] += int(n > 1 and (p > 0).all())
        coverage["call_opportunity_rows"] += int(count_calls > 0)
        coverage["optional_call_opportunity_rows"] += int(count_calls > 0 and n > 1)
        coverage["multiple_call_candidate_rows"] += int(count_calls > 1)
        coverage["call_and_noncall_rows"] += int(0 < count_calls < n)
        coverage["selected_call_rows"] += int(chosen_call)
        coverage["resolved_selected_call_rows"] += int(chosen_call and resolved)
        coverage["selected_calls_among_multiple_call_candidates"] += int(chosen_call and count_calls > 1)
        coverage["resolved_calls_among_multiple_call_candidates"] += int(chosen_call and resolved and count_calls > 1)
        coverage["post_call_rows"] += int(row["post_call_state"])
        coverage["optional_post_call_rows"] += int(row["post_call_state"] and n > 1)
        coverage["open_hand_rows"] += int(row["open_hand_state"])
        coverage["history_overflow_rows"] += int(prefix > 128)
        coverage["maximum_history_prefix_length"] = max(coverage["maximum_history_prefix_length"], prefix)
        coverage["win_opportunity_rows"] += int(wins.any())
        coverage["win_call_competition_rows"] += int(wins.any() and count_calls > 0)
        def increment(key, label):
            coverage[key][label] = coverage[key].get(label, 0) + 1
        increment("chosen_action_counts", chosen)
        increment("chosen_action_families", FAMILIES[chosen])
        increment("legal_candidate_count_histogram", str(n))
        increment("call_candidate_count_histogram", str(count_calls))
        for kind in set(kinds):
            increment("candidate_action_opportunities", kind)
        for family in {FAMILIES[kind] for kind in kinds}:
            increment("candidate_family_opportunities", family)
        if chosen_call:
            increment("requested_calls", chosen)
            if resolved:
                increment("resolved_calls", chosen)
        add_stat(coverage["selected_behavior_probability"], row["probability"])
        if count_calls:
            add_stat(coverage["actual_call_mass_on_call_opportunities"], float(p[calls].sum()))
        if wins.any():
            add_stat(coverage["actual_win_mass_on_win_opportunities"], float(p[wins].sum()))
    return result


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def cdf_choice(values: np.ndarray, draw: float | None) -> int:
    if len(values) == 1:
        require(draw is None, "Forced candidate consumed an action draw")
        return 0
    require(type(draw) is float and 0 <= draw < 1, "Invalid action draw")
    total = 0.
    for index, value in enumerate(values):
        total += float(value)
        if draw < total:
            return index
    return int(np.flatnonzero(values > 0)[-1])


def audit_behavior(provenance: dict[str, Any]) -> dict[str, Any]:
    """Independently reconstruct latent sampling from per-seat seed streams."""
    plan, rows, policies = provenance["plan"], provenance["rows"], provenance["policies"]
    validate_plan(plan)
    require(provenance["schema"] == SCHEMA and provenance["completed"] is True
            and provenance["failure"] is None, "Not a complete collector provenance record")
    require(provenance["training_eligible"] is False and provenance["ppo_eligible"] is False, "Development eligibility changed")
    require(len(policies) == 4, "Missing policy provenance")
    rngs = [random.Random(seed) for seed in plan["sampling_seeds"]]
    require(canonical([rng.getstate() for rng in rngs]) == canonical(provenance["sampling_rng_start"]), "Initial RNG states differ from declared seeds")
    for seat, policy in enumerate(policies):
        name = plan["seat_behaviors"][seat]
        require(policy["seat"] == seat and policy["behavior"] == name
                and policy["behavior_id"] == BEHAVIORS.index(name)
                and policy["sampling_seed"] == plan["sampling_seeds"][seat], "Seat behavior identity changed")
        require(policy["policy_id"] == name + ":" + digest(policy["details"]), "Policy identity omits or misstates details")
        if name == "explorer":
            details = policy["details"]
            require(details["collector_schema"] == SCHEMA and details["sampler"] == SAMPLER
                    and details["epsilon"] == EXPLORATION
                    and details["win_rule"] == "first legal semantic RON/TSUMO, probability 1"
                    and details["component_scope"] == "sampled independently at each non-win explorer decision"
                    and details["uniform_scope"] == "legal semantic non-win candidates", "Explorer policy details changed")
    counts = Counter()
    previous_step = -1
    for index, row in enumerate(rows):
        require(row["row_index"] == index and row["step"] >= previous_step, "Rows reordered")
        previous_step = row["step"]
        seat = row["seat"]
        require(type(seat) is int and seat in range(4), "Invalid row seat")
        name, policy = plan["seat_behaviors"][seat], policies[seat]
        require(row["policy_id"] == policy["policy_id"] and row["behavior_id"] == BEHAVIORS.index(name), "Row behavior attribution differs")
        require(row["training_eligible"] is False and row["ppo_eligible"] is False, "Row eligible for on-policy learning")
        kinds, keys = row["action_kinds"], row["semantic_keys"]
        n, selected = len(keys), row["selected_index"]
        require(n > 0 and len(kinds) == n and len(set(keys)) == n and 0 <= selected < n
                and all(kind in ACTION_KINDS for kind in kinds), "Invalid semantic candidates")
        require(len(row["representative_tokens"]) == n and row["selected_kind"] == kinds[selected]
                and row["selected_semantic_key"] == keys[selected]
                and row["selected_physical_token"] == row["representative_tokens"][selected]
                and row["selected_physical_token"] in row["physical_candidates"], "Selected action identity differs")
        probabilities = np.asarray(row["probabilities"], dtype=np.float64)
        require(probabilities.shape == (n,) and bool(np.isfinite(probabilities).all()) and bool((probabilities >= 0).all())
                and math.isclose(float(probabilities.sum()), 1., rel_tol=0, abs_tol=1e-12)
                and row["probability"] == float(probabilities[selected]) and row["probability"] > 0, "Invalid marginal propensity")
        expected = np.zeros(n, dtype=np.float64)
        wins = [i for i, kind in enumerate(kinds) if kind in ("RON", "TSUMO")]
        if name == "explorer":
            base = np.asarray(row["base_probabilities"], dtype=np.float64)
            require(base.shape == (n,) and bool(np.isfinite(base).all()) and bool((base >= 0).all())
                    and math.isclose(float(base.sum()), 1., rel_tol=0, abs_tol=1e-12), "Invalid base distribution")
        if name != "explorer" or wins:
            expected[selected] = 1.
            component = name if name != "explorer" else "win_heuristic"
            require(not wins or name != "explorer" or selected == wins[0], "Explorer declined declared deterministic win heuristic")
            require(row["selected_component"] == component and row["component_probability"] == 1.
                    and row["conditional_probability"] == 1. and row["joint_probability"] == 1.
                    and row["component_draw"] is None and row["uniform_draw"] is None, "Deterministic branch provenance changed")
        else:
            base = np.asarray(row["base_probabilities"], dtype=np.float64)
            uniform = np.full(n, 1 / n, dtype=np.float64)
            require(base.shape == (n,) and bool(np.isfinite(base).all()) and bool((base >= 0).all())
                    and math.isclose(float(base.sum()), 1., rel_tol=0, abs_tol=1e-12)
                    and np.array_equal(uniform, row["uniform_probabilities"]), "Invalid component distributions")
            expected = (1 - EXPLORATION) * base + EXPLORATION * uniform
            expected /= expected.sum()
            component_draw = rngs[seat].random()
            use_uniform = component_draw < EXPLORATION
            conditional = uniform if use_uniform else base
            component = "semantic_uniform" if use_uniform else "gated_base"
            weight = EXPLORATION if use_uniform else 1 - EXPLORATION
            action_draw = rngs[seat].random() if n > 1 else None
            require(row["component_draw"] == component_draw and row["uniform_draw"] == action_draw,
                    "Recorded draw does not replay from per-seat RNG")
            require(row["selected_component"] == component and row["component_probability"] == weight
                    and row["conditional_probability"] == float(conditional[selected])
                    and row["joint_probability"] == weight * float(conditional[selected])
                    and selected == cdf_choice(conditional, action_draw), "Latent component selection does not replay")
        require(np.array_equal(expected, probabilities), "Stored marginal does not equal declared sampling mixture")
        counts[component] += 1
        counts["rows"] += 1
        counts["win_heuristic_rows"] += int(component == "win_heuristic")
    require(canonical([rng.getstate() for rng in rngs]) == canonical(provenance["sampling_rng_end"]), "Final per-seat RNG state differs")
    requests = provenance["requests"]
    require(all(item["step"] == index and item["submitted"] is True for index, item in enumerate(requests)), "Missing or unsubmitted response")
    row_order = []
    for request in requests:
        require(request["pre_response_id"] == digest({"step": request["step"], "views": request["player_views"]}), "Response-state commitment differs")
        seats = []
        for index in request["rows"]:
            row = rows[index]
            require(row["step"] == request["step"] and row["pre_response_id"] == request["pre_response_id"]
                    and row["player_view_sha256"] == request["player_views"][str(row["seat"])], "Simultaneous response state differs")
            seats.append(row["seat"])
            row_order.append(index)
        require(len(seats) == len(set(seats)) and seats == sorted(seats), "Duplicate or reordered seat response")
    require(row_order == list(range(len(rows))), "Response provenance drops or duplicates a row")
    return dict(sorted(counts.items()))


def _read_artifact(directory: Path, report: dict, relative: str) -> bytes:
    metadata = report["artifacts"][relative]
    path = directory / relative
    require(path.resolve().is_relative_to(directory) and not path.is_symlink(), "Artifact escapes corpus")
    data = path.read_bytes()
    require(len(data) == metadata["bytes"] and hashlib.sha256(data).hexdigest() == metadata["sha256"], "Artifact hash differs: " + relative)
    return data


def reconcile_calls(rows: list[dict], requests: list[dict], resolutions: list[dict], log_length: int) -> None:
    """Attribute each accepted declaration once; rejected requests may have none."""
    require(not any(item["resolved"] is None for item in resolutions), "Complete match has unresolved call declaration")
    attributed = []
    for row in rows:
        require(row["requested_call"] is (row["selected_kind"] in CALL_KINDS), "Requested-call flag differs from selected action")
        if not row["requested_call"]:
            require(row["requested_call_resolved"] is None, "Non-call has a claimed call resolution")
            continue
        request = requests[row["step"]]
        lower, upper = request["auditor_event_start"], request["auditor_event_end"]
        require(type(lower) is int and type(upper) is int and 0 <= lower <= upper <= log_length,
                "Call event interval is outside auditor log")
        matching = [item for item in resolutions if item["actor"] == row["seat"] and item["kind"] == row["selected_kind"]
                    and lower <= item["event_index"] < upper]
        resolved = any(item["resolved"] for item in matching)
        require(row["call_resolution_evidence"] == matching and row["requested_call_resolved"] is resolved,
                "Requested call resolution differs")
        attributed.extend(item["event_index"] for item in matching)
    require(sorted(attributed) == [item["event_index"] for item in resolutions],
            "Call declaration lacks one unique requested-action attribution")


def audit_match(directory: Path, report: dict, actor: dict, targets: dict) -> dict:
    name = report["match_id"]
    require(json.loads((directory / "reports" / f"{name}.json").read_bytes()) == report,
            "Corpus manifest differs from the durable match report")
    for relative in report["artifacts"]:
        _read_artifact(directory, report, relative)
    provenance = json.loads(gzip.decompress(_read_artifact(directory, report, f"provenance/{name}.json.gz")))
    auditor = json.loads(gzip.decompress(_read_artifact(directory, report, f"auditor/{name}.json.gz")))
    labels = json.loads(_read_artifact(directory, report, f"targets/{name}.json"))
    counts = audit_behavior(provenance)
    rows, log = provenance["rows"], auditor["canonical_events"]
    require(len(rows) == report["rows"] and auditor["completed"] is True and auditor["failure"] is None, "Auditor match completion differs")
    require(report["steps"] == len(provenance["requests"]) and report["truncated"] is False
            and report["terminal_result_verified"] is True and report["physical_and_score_audits_passed"] is True,
            "Report completion or conservation assertions differ")
    require(provenance["plan"] == auditor["plan"] and all(report[key] == value for key, value in provenance["plan"].items()), "Saved plans differ")
    start = json.loads((directory / "lifecycle" / f"{name}.started.json").read_bytes())
    finish = json.loads((directory / "lifecycle" / f"{name}.finished.json").read_bytes())
    require(start["schema"] == SCHEMA and start["plan"] == provenance["plan"]
            and start["reference_sha256"] == report["reference_sha256"]
            and start["call_checkpoint_sha256"] == report["call_checkpoint_sha256"], "Start lifecycle identity differs")
    require(finish["schema"] == SCHEMA and finish["match_id"] == name and finish["completed"] is True
            and finish["payload_sha256"] == report["payload_sha256"] and finish["report_sha256"] == digest(report),
            "Finish lifecycle commitment differs")
    for policy in provenance["policies"]:
        if policy["behavior"] == "reference":
            require(policy["details"]["checkpoint_sha256"] == report["reference_sha256"], "Reference checkpoint attribution differs")
        if policy["behavior"] == "explorer":
            require(policy["details"]["checkpoint_sha256"] == report["call_checkpoint_sha256"]
                    and policy["details"]["reference_sha256"] == report["reference_sha256"], "Explorer checkpoint attribution differs")
    result = terminal_result(report["scores"], report["ranks"], log, completed=True)
    require(all(labels[key] == value for key, value in result.items()), "Terminal target score reconstruction differs")
    require(labels["training_eligible"] is False and labels["source_role"] == "development-only", "Hand targets eligibility changed")
    resolutions = call_resolutions(log)
    require(resolutions == auditor["call_resolution_evidence"], "Saved call resolution evidence differs")
    reconcile_calls(rows, provenance["requests"], resolutions, len(log))
    starts = [event for event in log if event.get("type") == "start_kyoku"]
    require(len(starts) == report["hands"] == len(labels["hand_starts"]) == len(auditor["ordinal_wall_fingerprints"]), "Ordinal hand or wall commitment omitted")
    for ordinal, (start, saved) in enumerate(zip(starts, labels["hand_starts"]), 1):
        require(saved == {"hand_id": ordinal, **{key: start[key] for key in ("bakaze", "kyoku", "honba", "kyotaku", "oya", "scores")}}, "Public hand-start target differs")
    for index, row in enumerate(rows):
        for target, key in (("chosen_index", "selected_index"), ("seat", "seat"), ("step", "step"),
                            ("hand_id", "hand_id"), ("behavior_id", "behavior_id"), ("behavior_probability", "probability")):
            require(targets[target][index] == row[key], "Numeric target differs from recorded behavior")
        lo, hi = actor["candidate_offsets"][index:index + 2]
        require(np.array_equal(targets["behavior_probabilities"][lo:hi], row["probabilities"]), "Candidate targets differ from recorded probabilities")
        require(row["action_kinds"] == [ACTION_KINDS[int(token) - 1] for token in actor["action_tokens"][lo:hi, 0]],
                "Provenance action kinds differ from actual actor candidates")
        require(targets["final_rank"][index] == result["ranks"][row["seat"]]
                and targets["final_utility"][index] == result["utilities"][row["seat"]], "Numeric outcome target differs")
    for kind in CALL_KINDS:
        require(report["requested_calls"][kind] == sum(row["selected_kind"] == kind for row in rows)
                and report["resolved_calls"][kind] == sum(row["selected_kind"] == kind and row["requested_call_resolved"] is True for row in rows),
                "Saved call aggregate differs from audited rows")
    require(report["post_call_rows"] == sum(row["post_call_state"] for row in rows)
            and report["open_hand_rows"] == sum(row["open_hand_state"] for row in rows), "Saved hand-state aggregate differs")
    require(report["chosen_action_counts"] == dict(sorted(Counter(row["selected_kind"] for row in rows).items()))
            and report["seat_rows"] == {str(seat): sum(row["seat"] == seat for row in rows) for seat in range(4)},
            "Saved chosen-action or per-seat counts differ")
    require(report["game_sha256"] == digest(log), "Full game commitment differs")
    deterministic = {key: value for key, value in report.items()
                     if key not in ("payload_sha256", "elapsed_seconds", "peak_worker_rss_bytes", "timing_seconds")}
    require(digest(deterministic) == report["payload_sha256"], "Deterministic payload commitment differs")
    kan = {kind: {"requested": sum(row["selected_kind"] == kind for row in rows),
                  "declared": sum(item["kind"] == kind for item in resolutions),
                  "resolved": sum(item["kind"] == kind and item["resolved"] is True for item in resolutions),
                  "not_completed_before_hand_end_or_win": sum(item["kind"] == kind and item["resolved"] is False for item in resolutions),
                  "unresolved": sum(item["kind"] == kind and item["resolved"] is None for item in resolutions),
                  "declarations": [item for item in resolutions if item["kind"] == kind]}
           for kind in ("DAIMINKAN", "ANKAN", "KAKAN")}
    continuations = sum(all(a[key] == b[key] for key in ("bakaze", "kyoku", "oya")) for a, b in zip(starts, starts[1:]))
    return {"match_id": name, "valid": True, "rows": len(rows), "behavior_counts": counts, "kan_paths": kan,
            "hands": len(starts), "dealer_continuations": continuations,
            "coverage_by_behavior": describe_coverage(rows, actor), "payload_sha256": report["payload_sha256"]}


def audit_corpus(directory: str | Path) -> dict:
    directory = Path(directory).resolve()
    started = time.monotonic()
    dataset = StreamDataset(directory)
    require(len({item["seed"] for item in dataset.reports.values()}) == len(dataset.reports)
            and len({item["independent_group"] for item in dataset.reports.values()}) == len(dataset.reports),
            "Development corpus repeats a seed or independent match group")
    outcomes = []
    coverage = {name: empty_coverage() for name in BEHAVIORS}
    behavior_counts = Counter()
    kan_totals = {kind: Counter() for kind in ("DAIMINKAN", "ANKAN", "KAKAN")}
    for name in dataset.match_ids:
        actor, targets, _ = dataset._load(name)
        match = audit_match(directory, dataset.reports[name], actor, targets)
        for behavior, values in match.pop("coverage_by_behavior").items():
            merge_coverage(coverage[behavior], values)
        behavior_counts.update(match["behavior_counts"])
        for kind, values in match["kan_paths"].items():
            kan_totals[kind].update({key: value for key, value in values.items() if key != "declarations"})
        outcomes.append(match)
    total_coverage = empty_coverage()
    for values in coverage.values():
        merge_coverage(total_coverage, values)
    return {"schema": "riichi-ai-stage3f-corpus-audit-v1", "valid": True,
            "manifest_sha256": dataset.manifest_sha256, "matches": outcomes,
            "completed_matches": len(outcomes), "audited_rows": sum(item["rows"] for item in outcomes),
            "behavior_counts": dict(sorted(behavior_counts.items())),
            "coverage_by_behavior": {key: finalize_coverage(value) for key, value in coverage.items()},
            "coverage_all_seats": finalize_coverage(total_coverage),
            "kan_paths": {key: dict(value) for key, value in kan_totals.items()},
            "hand_coverage": {"explicit_hand_starts": sum(item["hands"] for item in outcomes),
                "dealer_continuations": sum(item["dealer_continuations"] for item in outcomes),
                "matches_with_dealer_continuations": sum(item["dealer_continuations"] > 0 for item in outcomes),
                "maximum_hands_in_match": max(item["hands"] for item in outcomes),
                "method": "Count every explicit start_kyoku; continuations are adjacent starts sharing bakaze, kyoku and dealer"},
            "coverage_interpretation": "Legal/support opportunities are not chosen experience. Marginal propensities describe fixed collector behaviors, not action values or playing strength.",
            "audit_limitations": ["Stored distributions and RNG are audited on every row; frozen neural logits are not recomputed by this bulk audit.",
                "All recorded kan declarations are reconciled with requested actions and replacement draws; absent rare paths are not certified.",
                "Independent rule-engine certification is not performed; explicit selected-match replay is a separate operation."],
            "same_engine_replay_performed": False, "independent_rule_certification": False,
            "elapsed_seconds": time.monotonic() - started}


def replay_matches(directory: str | Path, replay_dir: str | Path, match_ids: list[str], *,
                   reference_path: str | Path, call_path: str | Path, deadline: float | None = None) -> dict:
    """Replay explicitly preselected complete matches into a fresh destination."""
    source = Path(directory).resolve()
    dataset = StreamDataset(source)
    require(bool(match_ids) and len(set(match_ids)) == len(match_ids) and set(match_ids) <= set(dataset.match_ids), "Invalid prespecified replay match IDs")
    destination = Path(replay_dir).resolve()
    require(not destination.exists(), "Replay destination must be new")
    destination.mkdir(parents=True)
    results = []
    for name in match_ids:
        original = dataset.reports[name]
        provenance = json.loads(gzip.decompress(_read_artifact(source, original, f"provenance/{name}.json.gz")))
        replay = run_one(provenance["plan"], reference_path, original["reference_sha256"], call_path,
                         destination, expected_call_sha256=original["call_checkpoint_sha256"], deadline=deadline)
        require(replay["completed"] and replay["payload_sha256"] == original["payload_sha256"], "Complete-match replay differs: " + name)
        results.append({"match_id": name, "rows": replay["rows"], "payload_sha256": replay["payload_sha256"], "matched": True})
    return {"schema": "riichi-ai-stage3f-replay-audit-v1", "valid": True, "matches": results,
            "replayed_rows": sum(item["rows"] for item in results),
            "total_corpus_rows": sum(item["rows"] for item in dataset.reports.values()),
            "replayed_row_fraction": sum(item["rows"] for item in results) / sum(item["rows"] for item in dataset.reports.values()),
            "replayed_match_fraction": len(results) / len(dataset.reports),
            "same_engine_replay_performed": True, "independent_rule_certification": False,
            "selection": "Explicit match IDs fixed by caller before replay results"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--replay-dir")
    parser.add_argument("--replay-match", action="append", default=[])
    parser.add_argument("--reference")
    parser.add_argument("--call-checkpoint")
    parser.add_argument("--deadline-seconds", type=float, default=120.)
    args = parser.parse_args()
    output = Path(args.output)
    require(not output.exists(), "Audit output already exists")
    try:
        if args.replay_match:
            require(bool(args.replay_dir and args.reference and args.call_checkpoint), "Replay requires destination and frozen checkpoints")
            result = replay_matches(args.directory, args.replay_dir, args.replay_match,
                reference_path=args.reference, call_path=args.call_checkpoint,
                deadline=time.monotonic() + args.deadline_seconds)
        else:
            result = audit_corpus(args.directory)
    except Exception as error:
        result = {"schema": "riichi-ai-stage3f-corpus-audit-v1", "valid": False,
                  "failure": {"type": type(error).__name__, "message": str(error)}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"valid": result["valid"], "output": str(output)}, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
