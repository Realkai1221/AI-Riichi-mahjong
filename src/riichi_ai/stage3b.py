"""Read-only action-support diagnostics on already exposed Stage 3A actor rows.

Selection uses only saved learner metadata. Numerical imports are lazy so the
controller can keep all model work inside its durable numerical lease.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import time
from typing import Any


SCHEMA = "stage3b-action-support-diagnostic-v1"
GRID = (
    {"name": "unchanged", "temperature": 1.0, "epsilon": 0.0},
    {"name": "temperature10", "temperature": 10.0, "epsilon": 0.0},
    {"name": "temperature100", "temperature": 100.0, "epsilon": 0.0},
    {"name": "uniform05", "temperature": 1.0, "epsilon": 0.05},
)
CALLS = ("CHI", "PON", "DAIMINKAN", "ANKAN", "KAKAN")
STRATA = ("ANKAN", "DAIMINKAN", "KAKAN", "CHI", "PON", "WIN", "DISCARD", "PASS", "OTHER")
ARRAY_FIELDS = ("tiles", "globals", "history", "history_mask", "action_tokens", "candidate_efficiency")


def _digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _object_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Stage 3B cooperative deadline reached")


def _stratum(kinds: list[str]) -> str:
    for kind in STRATA[:5]:
        if kind in kinds:
            return kind
    if "RON" in kinds or "TSUMO" in kinds:
        return "WIN"
    if "DISCARD" in kinds:
        return "DISCARD"
    if "PASS" in kinds:
        return "PASS"
    return "OTHER"


def _round_robin(rows: list[dict], count: int) -> list[dict]:
    groups = {kind: [] for kind in STRATA}
    for row in rows:
        groups[row["stratum"]].append(row)
    for values in groups.values():
        values.sort(key=lambda row: (row["selection_key"], row["id"]))
    selected = []
    while len(selected) < count and any(groups.values()):
        for kind in STRATA:
            if groups[kind] and len(selected) < count:
                selected.append(groups[kind].pop(0))
    return selected


def select_rows(source_root: str | Path, max_rows: int = 512, probe_rows: int = 32) -> dict:
    """Freeze outcome-independent SHA ordering across exclusive action strata.

    No logits, ranks, scores, utilities or raw engine state enter selection.
    Hashes bind the complete source files, while only actor metadata is retained.
    """
    if type(max_rows) is not int or not 1 <= max_rows <= 512:
        raise ValueError("At most 512 diagnostic rows are authorized")
    if type(probe_rows) is not int or not 0 <= probe_rows <= min(32, max_rows):
        raise ValueError("At most 32 paired gradient rows are authorized")
    root = Path(source_root).resolve()
    trajectory_root = root / "trajectories"
    records = sorted(trajectory_root.rglob("match-*.json"))
    if not records:
        raise ValueError("No saved Stage 3A trajectories found")
    sources, rows, seen = [], [], set()
    for path in records:
        record = json.loads(path.read_text())
        if not record.get("completed") or record.get("failure") is not None:
            raise ValueError("Only completed Stage 3A development matches are eligible")
        if record.get("schema") != "stage3a-complete-match-reinforce-v1":
            raise ValueError("Unexpected trajectory schema")
        arrays = Path(record["encoded_artifact"]["path"]).resolve()
        if not arrays.is_relative_to(trajectory_root) or arrays.suffix != ".npz":
            raise ValueError("Encoded artifact escapes the Stage 3A trajectory root")
        array_sha = _digest(arrays)
        if array_sha != record["encoded_artifact"]["sha256"]:
            raise ValueError("Encoded source hash mismatch")
        sources.append({"json_path": str(path.resolve()), "json_sha256": _digest(path),
                        "npz_path": str(arrays), "npz_sha256": array_sha})
        for index, row in enumerate(record["learner_rows"]):
            if row["row_index"] != index:
                raise ValueError("Saved row indices are not contiguous")
            if row["forced"]:
                continue
            if not row["eligible"] or row["fallback"] or row["overridden"]:
                raise ValueError("Ineligible actor row")
            kinds = [json.loads(key)[1] for key in row["semantic_keys"]]
            row_id = f"batch-{record['batch_index']}/match-{record['match_index']:03d}/row-{index:05d}"
            if row_id in seen:
                raise ValueError("Duplicate actor identity")
            seen.add(row_id)
            stratum = _stratum(kinds)
            target_kind = stratum if stratum in CALLS else (
                "RON" if "RON" in kinds else "TSUMO" if "TSUMO" in kinds else
                "PASS" if stratum == "PASS" else None)
            target = kinds.index(target_kind) if target_kind else row["selected_index"]
            rows.append({
                "id": row_id, "stratum": stratum,
                "selection_key": _object_hash(["stage3b-row-selection-v1", row_id]),
                "json_path": str(path.resolve()), "npz_path": str(arrays),
                "row_index": index, "match_index": record["match_index"],
                "batch_index": record["batch_index"], "seat": row["seat"],
                "policy_id": row["policy_id"],
                "player_view_sha256": row["player_view_sha256"],
                "encoded_sha256": row["encoded_sha256"],
                "semantic_keys": row["semantic_keys"], "action_kinds": kinds,
                "selected_index": row["selected_index"], "target_index": target,
                "target_kind": kinds[target],
            })
    selected = _round_robin(rows, max_rows)
    probes = _round_robin(selected, probe_rows)
    return {
        "schema": SCHEMA, "source_root": str(root), "sources": sources,
        "selection_rule": "Exclusive action-type priority; SHA256 identity order within strata; round-robin strata.",
        "stratum_priority": list(STRATA),
        "target_rule": "First candidate of the stratum call type; RON then TSUMO for WIN; PASS for PASS; recorded chosen action otherwise.",
        "selection_uses_rewards_or_model_outputs": False,
        "split": "previously exposed Stage 3A development states",
        "optional_rows_available": len(rows), "rows_selected": len(selected),
        "available_by_stratum": dict(Counter(row["stratum"] for row in rows)),
        "selected_by_stratum": dict(Counter(row["stratum"] for row in selected)),
        "selected_offered_by_kind": dict(Counter(kind for row in selected for kind in set(row["action_kinds"]))),
        "probe_ids": [row["id"] for row in probes],
        "probe_targets_by_kind": dict(Counter(row["target_kind"] for row in probes)),
        "rows": selected, "grid": [dict(item) for item in GRID],
    }


def distribution(logits: Any, legal_mask: Any = None, *,
                 temperature: float = 1.0, epsilon: float = 0.0) -> dict:
    """Differentiable CPU-float64 pi/q and stable log probabilities.

    Only legal logits normalize. A CPU transfer remains in the autograd graph.
    Stable log(pi) can be finite even where exp(log(pi)) underflows to zero.
    The mixture differentiates log(q), including its attenuated base-policy
    derivative, rather than substituting log(pi).
    """
    import torch

    if logits.ndim != 1 or not logits.numel():
        raise ValueError("Expected a nonempty vector of candidate logits")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    if not math.isfinite(epsilon) or not 0 <= epsilon < 1:
        raise ValueError("Epsilon must be finite in [0,1)")
    # On the pinned MPS runtime, a combined device+float64 conversion corrupts
    # values. Move in the original dtype first, then cast on CPU. Both copies
    # stay differentiable; the transfer fixture records the failing route.
    values = logits.to(device="cpu").to(dtype=torch.float64)
    mask = (torch.ones_like(values, dtype=torch.bool) if legal_mask is None else
            legal_mask.to(device="cpu", dtype=torch.bool))
    if mask.shape != values.shape or not bool(mask.any()):
        raise ValueError("Expected at least one legal candidate")
    legal = values[mask] / temperature
    if not bool(torch.isfinite(legal).all()):
        raise ValueError("Legal logits must be finite")
    base = torch.softmax(legal, dim=0)
    total = base.sum()
    base = base / total
    log_base = torch.log_softmax(legal, dim=0) - torch.log(total)
    if epsilon:
        actual = (1.0 - epsilon) * base + epsilon / legal.numel()
        log_actual = torch.logaddexp(log_base + math.log1p(-epsilon),
                                    torch.full_like(log_base, math.log(epsilon / legal.numel())))
    else:
        actual, log_actual = base, log_base
    def expand(tensor: Any, fill: float) -> Any:
        return torch.full_like(values, fill).masked_scatter(mask, tensor)
    return {
        "probabilities": expand(actual, 0.0),
        "log_probabilities": expand(log_actual, -math.inf),
        "base_probabilities": expand(base, 0.0),
        "base_log_probabilities": expand(log_base, -math.inf),
        "mixture_gradient_attenuation": expand((1.0 - epsilon) * base / actual
                                               if epsilon else torch.ones_like(base), 0.0),
        "legal_mask": mask,
        "normalization_logits": values,
    }


def _tensor_values(value: Any) -> list:
    return [float(item) if math.isfinite(float(item)) else None
            for item in value.detach().to("cpu").reshape(-1).tolist()]


def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return {"count": len(values), "min": ordered[0], "median": median,
            "mean": sum(values) / len(values), "max": ordered[-1]}


def _distribution_metrics(result: dict, kinds: list[str], baseline: Any) -> dict:
    import torch

    probabilities = result["probabilities"].detach()
    legal = result["legal_mask"]
    positive = probabilities > 0
    entropy = -float((probabilities[positive] * probabilities[positive].log()).sum())
    probability_values = probabilities.tolist()
    cumulative, zero_width = 0.0, 0
    for probability in probability_values:
        following = cumulative + probability
        zero_width += probability > 0 and following == cumulative
        cumulative = following
    l1 = float((probabilities - baseline).abs().sum())
    original_winner = int(baseline.argmax())
    by_kind = {}
    for kind in sorted(set(kinds)):
        mask = torch.tensor([candidate == kind for candidate in kinds], dtype=torch.bool)
        subset = probabilities[mask]
        by_kind[kind] = {"candidates": int(mask.sum()), "mass": float(subset.sum()),
                         "zero_probability_candidates": int((subset == 0).sum())}
    return {
        "probabilities": _tensor_values(probabilities),
        "log_probabilities": _tensor_values(result["log_probabilities"]),
        "base_log_probabilities": _tensor_values(result["base_log_probabilities"]),
        "mixture_gradient_attenuation": _tensor_values(result["mixture_gradient_attenuation"]),
        "entropy_nats": entropy, "maximum_probability": float(probabilities.max()),
        "legal_candidates": int(legal.sum()),
        "zero_probability_candidates": int((legal & ~positive).sum()),
        "underflow_with_finite_log_probability": int((legal & ~positive & torch.isfinite(result["log_probabilities"])).sum()),
        "minimum_positive_probability": float(probabilities[positive].min()),
        "l1_from_same_checkpoint_unchanged": l1,
        "total_variation_from_same_checkpoint_unchanged": l1 / 2,
        "unchanged_argmax_index": original_winner,
        "unchanged_argmax_probability": float(probabilities[original_winner]),
        "argmax_index_preserved": int(probabilities.argmax()) == original_winner,
        "positive_probabilities_below_2_minus53": sum(0 < value < 2**-53 for value in probability_values),
        "positive_probability_zero_width_float64_cdf_increments": zero_width,
        "cdf_note": "Zero-width increments are an order-sensitive float64 diagnostic; positive probability alone does not guarantee useful sampling. No random samples were drawn.",
        "call_kan_mass": sum(by_kind.get(kind, {}).get("mass", 0.0) for kind in CALLS),
        "by_kind": by_kind,
    }


def gradient_probe(model: Any, encoded: dict, *, target_index: int,
                   device: str, grid: Any = GRID) -> dict:
    """Backward-only responses; autograd.grad leaves model .grad fields alone."""
    import torch
    from .model import collate_encoded

    batch = collate_encoded([encoded], device)
    logits = model(batch)[0][0]
    parameters = [(name, parameter) for name, parameter in model.named_parameters()
                  if parameter.requires_grad]
    results = {}
    for option in grid:
        result = distribution(logits, batch["candidate_mask"][0],
                              temperature=option["temperature"], epsilon=option["epsilon"])
        target = result["log_probabilities"][target_index]
        gradients = torch.autograd.grad(target, [result["normalization_logits"], logits, *[p for _, p in parameters]],
                                        retain_graph=True, allow_unused=True)
        by_parameter, groups = {}, defaultdict(lambda: {"squared_norm": 0.0, "max_abs": 0.0,
                                                        "nonzero_elements": 0, "elements": 0})
        finite = True
        for (name, parameter), gradient in zip(parameters, gradients[2:]):
            if gradient is None:
                by_parameter[name] = {"connected": False, "l2_norm": 0.0, "max_abs": 0.0,
                                      "nonzero_elements": 0, "elements": parameter.numel()}
                continue
            array = gradient.detach().to("cpu").to(dtype=torch.float64)
            valid = bool(torch.isfinite(array).all())
            norm2 = float(array.square().sum()) if valid else None
            max_abs = float(array.abs().max()) if valid else None
            valid = valid and math.isfinite(norm2) and math.isfinite(max_abs)
            finite = finite and valid
            nonzero = int(torch.count_nonzero(array))
            by_parameter[name] = {"connected": True, "l2_norm": math.sqrt(norm2) if valid else None,
                                  "max_abs": max_abs, "nonzero_elements": nonzero, "elements": array.numel()}
            group = groups[name.split(".")[0]]
            if valid:
                group["squared_norm"] += norm2
                group["max_abs"] = max(group["max_abs"], max_abs)
            group["nonzero_elements"] += nonzero
            group["elements"] += array.numel()
        finite = finite and all(math.isfinite(group["squared_norm"]) for group in groups.values())
        for group in groups.values():
            squared_norm = group.pop("squared_norm")
            group["l2_norm"] = math.sqrt(squared_norm) if finite else None
        normalization_raw = gradients[0].detach().to("cpu").to(dtype=torch.float64)
        raw = gradients[1].detach().to("cpu").to(dtype=torch.float64)
        raw_finite = bool(torch.isfinite(raw).all())
        raw_norm = float(raw.square().sum().sqrt()) if raw_finite else None
        raw_max = float(raw.abs().max()) if raw_finite else None
        raw_finite = raw_finite and math.isfinite(raw_norm) and math.isfinite(raw_max)
        normalization_finite = bool(torch.isfinite(normalization_raw).all())
        normalization_norm = math.hypot(*normalization_raw.tolist()) if normalization_finite else None
        normalization_finite = normalization_finite and math.isfinite(normalization_norm)
        finite = finite and raw_finite and normalization_finite
        try:
            aggregate_norm = math.sqrt(sum(item["l2_norm"] ** 2 for item in by_parameter.values())) if finite else None
        except OverflowError:
            aggregate_norm = None
        finite = finite and aggregate_norm is not None and math.isfinite(aggregate_norm)
        if not finite:
            aggregate_norm = None
        results[option["name"]] = {
            "target_probability": float(result["probabilities"][target_index].detach()),
            "target_log_probability": float(target.detach()),
            "target_base_log_probability": float(result["base_log_probabilities"][target_index].detach()),
            "target_mixture_gradient_attenuation": float(result["mixture_gradient_attenuation"][target_index].detach()),
            "logit_gradients": _tensor_values(raw),
            "logit_gradient_l2": raw_norm if raw_finite else None,
            "logit_gradient_max_abs": raw_max if raw_finite else None,
            "normalization_float64_logit_gradients": _tensor_values(normalization_raw),
            "normalization_float64_logit_gradient_l2": normalization_norm if normalization_finite else None,
            "logit_gradient_note": "logit_gradients are after the cast back into float32 network logits; normalization_float64_logit_gradients are before that cast.",
            "parameter_gradient_l2": aggregate_norm, "gradients_finite": finite,
            "parameter_gradient_exactly_zero": aggregate_norm == 0.0 if finite else None,
            "parameter_gradient_below_1e_minus12": aggregate_norm < 1e-12 if finite else None,
            "parameter_groups": dict(groups), "parameters": by_parameter,
        }
    return results


def _load_actor_rows(selection: dict, deadline: float | None) -> dict:
    import numpy as np
    from .stage3a import encoded_hash
    from .semantics import ACTION_KINDS

    for source in selection["sources"]:
        _check_deadline(deadline)
        for prefix in ("json", "npz"):
            if _digest(source[f"{prefix}_path"]) != source[f"{prefix}_sha256"]:
                raise ValueError("Frozen trajectory source changed")
    result, grouped = {}, defaultdict(list)
    for row in selection["rows"]:
        grouped[row["npz_path"]].append(row)
    for path, rows in grouped.items():
        _check_deadline(deadline)
        with np.load(path, allow_pickle=False) as archive:
            for row in rows:
                prefix = f"row_{row['row_index']:05d}__"
                keys = [key for key in archive.files if key.startswith(prefix)]
                if set(key[len(prefix):] for key in keys) != set(ARRAY_FIELDS):
                    raise ValueError("Unexpected saved actor feature keys")
                encoded = {key[len(prefix):]: archive[key].copy() for key in keys}
                if encoded_hash(encoded) != row["encoded_sha256"]:
                    raise ValueError("Saved actor array identity mismatch")
                kinds = [ACTION_KINDS[int(value) - 1] for value in encoded["action_tokens"][:, 0]]
                if kinds != row["action_kinds"]:
                    raise ValueError("Actor candidate types differ from semantic metadata")
                if len(kinds) <= 1:
                    raise ValueError("Forced rows are outside this diagnostic")
                result[row["id"]] = encoded
    return result


def _load_model(checkpoint: dict, device: str) -> tuple[Any, str]:
    import torch
    from .model import PolicyNetwork, _weights_hash

    payload = Path(checkpoint["path"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != checkpoint["sha256"]:
        raise ValueError("Frozen checkpoint identity mismatch")
    saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    with torch.random.fork_rng(devices=[]):
        model = PolicyNetwork(saved["model_config"])
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.to(device).eval()
    for parameter in model.rank_head.parameters():
        parameter.requires_grad_(False)
    return model, _weights_hash(model)


def _row_forward(model: Any, encoded: dict, row: dict, device: str) -> dict:
    import torch
    from .model import collate_encoded

    with torch.inference_mode():
        batch = collate_encoded([encoded], device)
        logits = model(batch)[0][0]
        if not bool(batch["candidate_mask"].all()):
            raise ValueError("A single unpadded saved actor row unexpectedly contains masked candidates")
        baseline = distribution(logits)["probabilities"]
        outputs = {option["name"]: _distribution_metrics(
            distribution(logits, batch["candidate_mask"][0], temperature=option["temperature"],
                         epsilon=option["epsilon"]), row["action_kinds"], baseline) for option in GRID}
        raw = logits.detach().to("cpu").to(dtype=torch.float64)
        maximum = float(raw.max())
        by_kind = {}
        for kind in sorted(set(row["action_kinds"])):
            values = [float(raw[index]) for index, other in enumerate(row["action_kinds"]) if other == kind]
            by_kind[kind] = {"logits": _stats(values), "gap_to_best": _stats([maximum - value for value in values])}
        return {
            "id": row["id"], "stratum": row["stratum"], "action_kinds": row["action_kinds"],
            "semantic_keys": row["semantic_keys"], "recorded_policy_id": row["policy_id"],
            "recorded_selected_index": row["selected_index"], "target_index": row["target_index"],
            "target_kind": row["target_kind"], "logits": _tensor_values(raw),
            "gaps_to_best_logit": _tensor_values(raw.max() - raw),
            "logit_range": float(raw.max() - raw.min()),
            "candidate_mask": batch["candidate_mask"][0].to("cpu").tolist(),
            "mask_all_saved_candidates_legal": True, "by_kind": by_kind, "variants": outputs,
        }


def _summarize(rows: list[dict], gradients: list[dict]) -> dict:
    summary = {}
    for option in GRID:
        name = option["name"]
        values = [row["variants"][name] for row in rows]
        kinds = {}
        for kind in sorted({kind for row in rows for kind in row["action_kinds"]}):
            subsets = [value["by_kind"][kind] for value in values if kind in value["by_kind"]]
            gaps = [gap for row in rows for index, gap in enumerate(row["gaps_to_best_logit"])
                    if row["action_kinds"][index] == kind]
            kinds[kind] = {"offered_rows": len(subsets),
                           "candidates": sum(value["candidates"] for value in subsets),
                           "zero_probability_candidates": sum(value["zero_probability_candidates"] for value in subsets),
                           "probability_mass_when_offered": _stats([value["mass"] for value in subsets]),
                           "gap_to_best_logit": _stats(gaps)}
        probe_values = [row["variants"][name] for row in gradients]
        summary[name] = {
            "rows": len(rows),
            "entropy_nats": _stats([value["entropy_nats"] for value in values]),
            "l1_from_same_checkpoint_unchanged": _stats([value["l1_from_same_checkpoint_unchanged"] for value in values]),
            "ordinary_discard_l1": _stats([row["variants"][name]["l1_from_same_checkpoint_unchanged"]
                                           for row in rows if row["stratum"] == "DISCARD"]),
            "ordinary_discard_total_variation": _stats([row["variants"][name]["total_variation_from_same_checkpoint_unchanged"]
                                                         for row in rows if row["stratum"] == "DISCARD"]),
            "ordinary_discard_unchanged_argmax_probability": _stats([row["variants"][name]["unchanged_argmax_probability"]
                                                                     for row in rows if row["stratum"] == "DISCARD"]),
            "positive_probability_zero_width_float64_cdf_increments": sum(value["positive_probability_zero_width_float64_cdf_increments"] for value in values),
            "positive_probabilities_below_2_minus53": sum(value["positive_probabilities_below_2_minus53"] for value in values),
            "zero_probability_candidates": sum(value["zero_probability_candidates"] for value in values),
            "underflow_with_finite_log_probability": sum(value["underflow_with_finite_log_probability"] for value in values),
            "rows_with_maximum_probability_at_least_0999": sum(value["maximum_probability"] >= .999 for value in values),
            "by_kind": kinds,
            "gradient_probes": len(probe_values),
            "gradient_probes_exact_zero_parameters": sum(value["parameter_gradient_exactly_zero"] is True for value in probe_values),
            "gradient_probes_parameters_below_1e_minus12": sum(value["parameter_gradient_below_1e_minus12"] is True for value in probe_values),
            "parameter_gradient_l2": _stats([value["parameter_gradient_l2"] for value in probe_values
                                            if value["parameter_gradient_l2"] is not None]),
            "all_gradient_probes_finite": all(value["gradients_finite"] for value in probe_values),
        }
    return summary


def _compare_rows(cpu: list[dict], other: list[dict]) -> dict:
    lookup = {row["id"]: row for row in cpu}
    results = []
    for row in other:
        reference = lookup[row["id"]]
        variants = {}
        for option in GRID:
            name = option["name"]
            left, right = reference["variants"][name]["probabilities"], row["variants"][name]["probabilities"]
            variants[name] = {"max_probability_difference": max(abs(a - b) for a, b in zip(left, right)),
                              "l1_probability_difference": sum(abs(a - b) for a, b in zip(left, right))}
        results.append({"id": row["id"],
                        "max_logit_difference": max(abs(a - b) for a, b in zip(reference["logits"], row["logits"])),
                        "variants": variants})
    return {"rows": results,
            "max_logit_difference": max((row["max_logit_difference"] for row in results), default=None),
            "variants": {option["name"]: {
                "max_probability_difference": max((row["variants"][option["name"]]["max_probability_difference"]
                                                    for row in results), default=None),
                "l1_probability_difference": _stats([row["variants"][option["name"]]["l1_probability_difference"]
                                                       for row in results])} for option in GRID}}


def run_diagnostic(selection: dict, checkpoints: list[dict], deadline: float | None = None,
                   include_mps: bool = True) -> dict:
    """Run the predeclared finite grid. No environment, optimizer or save API."""
    import torch
    from .model import _weights_hash

    if (selection["schema"] != SCHEMA or selection["grid"] != list(GRID)
            or not 1 <= len(selection["rows"]) <= 512 or len(selection["probe_ids"]) > 32
            or len({row["id"] for row in selection["rows"]}) != len(selection["rows"])):
        raise ValueError("Frozen Stage 3B selection or grid is invalid")
    if len(checkpoints) != 2 or [item["label"] for item in checkpoints] != ["initial", "final"]:
        raise ValueError("Exactly the initial and final checkpoints are required")
    torch.set_num_threads(1)
    before_rng = _object_hash(torch.get_rng_state().tolist())
    arrays = _load_actor_rows(selection, deadline)
    by_id = {row["id"]: row for row in selection["rows"]}
    if len(set(selection["probe_ids"])) != len(selection["probe_ids"]) or not set(selection["probe_ids"]) <= set(by_id):
        raise ValueError("Invalid frozen probe identities")
    result = {
        "schema": SCHEMA, "selection_sha256": _object_hash(selection),
        "grid": [dict(item) for item in GRID], "rows": len(by_id),
        "gradient_rows_per_checkpoint_and_device": len(selection["probe_ids"]),
        "precision": "CPU/MPS float32 single-row forward; differentiable CPU float64 normalization",
        "new_matches": 0, "optimizer_updates": 0, "checkpoint_writes": 0,
        "gradient_interpretation": "Counterfactual d log(q_target) probes, not samples or learning updates. Near-zero threshold 1e-12 is descriptive.",
        "checkpoints": [], "cpu_complete": False, "mps": {"requested": include_mps},
    }
    for checkpoint in checkpoints:
        _check_deadline(deadline)
        model, before_hash = _load_model(checkpoint, "cpu")
        rows, gradients = [], []
        for row in selection["rows"]:
            _check_deadline(deadline)
            rows.append(_row_forward(model, arrays[row["id"]], row, "cpu"))
        for identity in selection["probe_ids"]:
            _check_deadline(deadline)
            row = by_id[identity]
            gradients.append({"id": identity, "target_index": row["target_index"], "target_kind": row["target_kind"],
                              "variants": gradient_probe(model, arrays[identity], target_index=row["target_index"], device="cpu")})
        after_hash = _weights_hash(model)
        if before_hash != after_hash or any(parameter.grad is not None for parameter in model.parameters()):
            raise ValueError("Read-only gradient diagnostic mutated the CPU model")
        result["checkpoints"].append({
            **checkpoint, "weights_before_sha256": before_hash, "weights_after_sha256": after_hash,
            "parameters_unchanged": True, "parameter_grad_fields_unchanged": True,
            "cpu_rows": rows, "cpu_gradients": gradients, "cpu_summary": _summarize(rows, gradients),
        })
        del model
    result["cpu_complete"] = True
    result["initial_vs_final_cpu"] = _compare_rows(result["checkpoints"][0]["cpu_rows"],
                                                   result["checkpoints"][1]["cpu_rows"])
    mps_available = torch.backends.mps.is_available()
    result["mps"]["available"] = mps_available
    if not include_mps or not mps_available:
        result["mps"]["status"] = "not_requested" if not include_mps else "unavailable"
    else:
        result["mps"]["status"] = "running"
        mps_rng_before = _object_hash(torch.mps.get_rng_state().tolist())
        try:
            for checkpoint, output in zip(checkpoints, result["checkpoints"]):
                _check_deadline(deadline)
                model, before_hash = _load_model(checkpoint, "mps")
                rows, gradients = [], []
                output["mps_rows"], output["mps_gradients"] = rows, gradients
                output["mps_parameters_before_sha256"] = before_hash
                try:
                    for identity in selection["probe_ids"]:
                        _check_deadline(deadline)
                        row = by_id[identity]
                        rows.append(_row_forward(model, arrays[identity], row, "mps"))
                        gradients.append({"id": identity, "target_index": row["target_index"], "target_kind": row["target_kind"],
                                          "variants": gradient_probe(model, arrays[identity], target_index=row["target_index"], device="mps")})
                finally:
                    torch.mps.synchronize()
                    after_hash = _weights_hash(model)
                    output["mps_parameters_after_sha256"] = after_hash
                    output["mps_parameters_unchanged"] = before_hash == after_hash
                    output["mps_parameter_grad_fields_unchanged"] = all(parameter.grad is None for parameter in model.parameters())
                    output["mps_summary"] = _summarize(rows, gradients)
                    output["cpu_vs_mps"] = _compare_rows(output["cpu_rows"], rows)
                if not output["mps_parameters_unchanged"] or not output["mps_parameter_grad_fields_unchanged"]:
                    raise ValueError("Read-only gradient diagnostic mutated the MPS model")
                del model
            result["mps"]["status"] = "completed"
        except (TimeoutError, RuntimeError) as error:
            result["mps"]["status"] = "partial"
            result["mps"]["error"] = {"type": type(error).__name__, "message": str(error)}
        finally:
            mps_rng_after = _object_hash(torch.mps.get_rng_state().tolist())
            result["mps"].update(rng_before_sha256=mps_rng_before, rng_after_sha256=mps_rng_after,
                                 rng_unchanged=mps_rng_before == mps_rng_after)
    after_rng = _object_hash(torch.get_rng_state().tolist())
    result.update(cpu_rng_before_sha256=before_rng, cpu_rng_after_sha256=after_rng,
                  cpu_rng_unchanged=before_rng == after_rng,
                  complete=result["cpu_complete"] and result["mps"]["status"] in ("completed", "unavailable", "not_requested"))
    return result
