"""Inference-only precision comparisons on frozen, player-visible states.

The canonical PolicyNetwork already preserves floating input/parameter dtypes.
Consequently CPU float64 uses its unchanged forward on an independent model
copy: no alternative actor implementation, optimizer, or training trajectory is
introduced. Float64 normalization of *saved float32 logits* is a separate check
and cannot restore distinctions lost upstream.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from typing import Any

import numpy as np
import torch

from .model import _state_hash
from .semantics import ACTION_KINDS

SCHEMA = "package-a-diagnosis-precision-v1"
DISCARD = ACTION_KINDS.index("DISCARD") + 1


def _stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    return {"count": int(values.size), "nonfinite_count": int(values.size - finite.size),
            "mean": float(finite.mean()) if finite.size else None,
            "median": float(np.median(finite)) if finite.size else None,
            "maximum": float(finite.max()) if finite.size else None}


def _array(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


def _grad_hash(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        if parameter.grad is None:
            digest.update(b"None")
        else:
            value = parameter.grad.detach().cpu().contiguous()
            digest.update(json.dumps([str(value.dtype), list(value.shape)]).encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def prediction_error(record: Mapping, choice: int) -> dict[str, Any]:
    """Target-only error annotation, never an input to the actor."""
    teacher = int(record["chosen_index"])
    tokens = np.asarray(record["encoded"]["action_tokens"])
    result = {"teacher_best": bool(record["best_mask"][choice]),
              "chosen_index": int(choice), "teacher_chosen_index": teacher,
              "teacher_discard": bool(tokens[teacher, 0] == DISCARD),
              "chosen_discard": bool(tokens[choice, 0] == DISCARD),
              "shanten_increment": None, "same_shanten_lost_ukeire": None}
    if (result["teacher_discard"] and result["chosen_discard"]
            and "teacher_preference_keys" in record):
        keys = np.asarray(record["teacher_preference_keys"])
        if keys.shape != (len(tokens), 5):
            raise ValueError("Diagnostic preferences must have shape [candidates,5]")
        delta = int(keys[choice, 2]) - int(keys[teacher, 2])
        result["shanten_increment"] = max(0, delta)
        result["same_shanten_lost_ukeire"] = (
            max(0, int(keys[choice, 3]) - int(keys[teacher, 3])) if delta == 0 else None)
    return result


def summarize_outputs(logits: np.ndarray, probabilities: np.ndarray, context: np.ndarray,
                      mask: np.ndarray, best: np.ndarray) -> tuple[dict, dict]:
    """Exclude padding from every logit statistic and preserve row-level arrays.

    Centering and descriptive statistics use CPU float64 *after* the native
    forward. This is analysis, not a changed network computation. Nonfinite
    outputs are counted explicitly; no invalid prediction is interpreted.
    """
    z = np.asarray(logits)
    p = np.asarray(probabilities)
    c = np.asarray(context, dtype=np.float64)
    mask, best = np.asarray(mask, bool), np.asarray(best, bool)
    if z.shape != p.shape or z.shape != mask.shape or best.shape != mask.shape:
        raise ValueError("Logits, probabilities, legal mask and best mask must have equal shapes")
    if z.ndim != 2 or c.ndim != 2 or len(c) != len(z):
        raise ValueError("Expected row-by-candidate logits and row-by-feature context")
    if not mask.any(-1).all() or not best.any(-1).all() or (best & ~mask).any():
        raise ValueError("Every row needs legal candidates and a nonempty legal best set")
    legal = z[mask].astype(np.float64)
    finite = bool(np.isfinite(legal).all() and np.isfinite(p).all() and np.isfinite(c).all())
    report = {"rows": len(z), "optional_rows": int((mask.sum(-1) > 1).sum()),
              "legal_candidates": int(mask.sum()), "padding_excluded": int((~mask).sum()),
              "finite": finite, "nonfinite_legal_logits": int((~np.isfinite(legal)).sum()),
              "nonfinite_probabilities": int((~np.isfinite(p)).sum()),
              "nonfinite_context": int((~np.isfinite(c)).sum()),
              "native_logit_dtype": str(z.dtype), "probability_dtype": str(p.dtype)}
    arrays = {"logits": np.where(mask, z, 0), "probabilities": p.copy()}
    if not finite:
        report["interpretation"] = "Nonfinite output; prediction comparisons invalid."
        return report, arrays
    z64, p64 = z.astype(np.float64), p.astype(np.float64)
    offsets = np.where(mask, z64, 0).sum(-1) / mask.sum(-1)
    centered = np.where(mask, z64 - offsets[:, None], 0)
    choice = np.where(mask, z64, -np.inf).argmax(-1)
    optional = mask.sum(-1) > 1
    top_margin = np.zeros(len(z), dtype=np.float64)
    ties = np.zeros(len(z), dtype=np.int64)
    best_gap = np.zeros(len(z), dtype=np.float64)
    has_nonbest = (mask & ~best).any(-1)
    for index in range(len(z)):
        values = z64[index, mask[index]]
        ties[index] = int((values == values.max()).sum())
        if optional[index]:
            top = np.sort(values)[-2:]
            top_margin[index] = top[1] - top[0]
        if has_nonbest[index]:
            best_gap[index] = z64[index, best[index]].max() - z64[index, mask[index] & ~best[index]].max()
    context_rms = np.sqrt((c * c).mean(-1))
    membership = best[np.arange(len(z)), choice]
    best_mass = np.where(best, p64, 0).sum(-1)
    report.update({"legal_logit_abs": _stats(np.abs(legal)),
                   "legal_logit_row_offset_abs": _stats(np.abs(offsets)),
                   "centered_legal_logit_abs": _stats(np.abs(centered[mask])),
                   "context_rms": _stats(context_rms),
                   "top_two_margin_optional": _stats(top_margin[optional]),
                   "exact_top_tie_rows_optional": int((optional & (ties > 1)).sum()),
                   "best_minus_nonbest_gap": _stats(best_gap[has_nonbest]),
                   "teacher_best_agreement_optional": float(membership[optional].mean()) if optional.any() else None,
                   "teacher_best_probability_optional": _stats(best_mass[optional]),
                   "zero_probability_legal_candidates": int((p[mask] == 0).sum()),
                   "probability_row_sum_absolute_error": _stats(np.abs(p64.sum(-1) - 1)),
                   "padding_probability_maximum": float(p64[~mask].max()) if (~mask).any() else 0.,
                   "argmax_illegal_count": int((~mask[np.arange(len(z)), choice]).sum())})
    arrays.update({"centered_logits": centered, "row_offsets": offsets,
                   "context_rms": context_rms, "argmax": choice, "top_two_margin": top_margin,
                   "top_tie_count": ties, "best_minus_nonbest_gap": best_gap,
                   "has_nonbest": has_nonbest, "teacher_best_probability": best_mass})
    return report, arrays


def _compare(left: dict, right: dict, mask: np.ndarray, best: np.ndarray,
             row_ids: list[int], records: Sequence[Mapping] | None) -> dict:
    optional = mask.sum(-1) > 1
    changed = left["argmax"] != right["argmax"]
    centered_delta = np.abs(left["centered_logits"] - right["centered_logits"])
    row_uncertainty = np.where(mask, centered_delta, 0).max(-1)
    # If each centered score changes by at most d, a strict margin > 2d
    # certifies that this particular comparison cannot change the winner.
    near = optional & (left["top_two_margin"] <= 2 * row_uncertainty)
    changed_rows = []
    for index in np.flatnonzero(changed):
        item = {"row_id": row_ids[index], "left_choice": int(left["argmax"][index]),
                "right_choice": int(right["argmax"][index]),
                "left_top_two_margin": float(left["top_two_margin"][index]),
                "right_top_two_margin": float(right["top_two_margin"][index]),
                "maximum_centered_logit_difference": float(row_uncertainty[index]),
                "left_teacher_best": bool(best[index, left["argmax"][index]]),
                "right_teacher_best": bool(best[index, right["argmax"][index]])}
        if records is not None:
            item["left_error"] = prediction_error(records[index], int(left["argmax"][index]))
            item["right_error"] = prediction_error(records[index], int(right["argmax"][index]))
        changed_rows.append(item)
    left_membership = best[np.arange(len(mask)), left["argmax"]]
    right_membership = best[np.arange(len(mask)), right["argmax"]]
    return {"rows": len(mask), "optional_rows": int(optional.sum()),
            "argmax_disagreement_rows": int(changed.sum()),
            "argmax_disagreement_optional_rate": float(changed[optional].mean()) if optional.any() else None,
            "teacher_best_membership_changed_rows": int((left_membership != right_membership).sum()),
            "legal_logit_absolute_difference": _stats(np.abs(left["logits"].astype(np.float64) - right["logits"].astype(np.float64))[mask]),
            "centered_legal_logit_absolute_difference": _stats(centered_delta[mask]),
            "probability_absolute_difference": _stats(np.abs(left["probabilities"].astype(np.float64) - right["probabilities"].astype(np.float64))[mask]),
            "probability_row_l1_difference": _stats(np.abs(left["probabilities"].astype(np.float64) - right["probabilities"].astype(np.float64)).sum(-1)),
            "teacher_best_probability_absolute_difference": _stats(np.abs(left["teacher_best_probability"] - right["teacher_best_probability"])),
            "margin_within_twice_observed_centered_difference_rows": int(near.sum()),
            "margin_diagnostic_note": "Descriptive perturbation bound for these two forwards; not an error bar or a training-instability predictor.",
            "changed_rows": changed_rows}


def compare_precision(model, batch: Mapping[str, torch.Tensor], labels: Mapping[str, torch.Tensor],
                      *, row_ids: Sequence[int] | None = None, records: Sequence[Mapping] | None = None,
                      include_mps: bool = True, check=None) -> tuple[dict, dict[str, np.ndarray]]:
    """Compare canonical CPU32/MPS32/CPU64 forwards without changing the caller.

    Returns JSON-safe summaries and compact arrays for parent-controlled NPZ
    persistence. The caller owns budget enforcement and selection of fixed rows.
    This helper performs no loading, filesystem writes, or optimizer updates.
    """
    mask = _array(batch["candidate_mask"]).astype(bool)
    best = _array(labels["best"]).astype(bool)
    count = len(mask)
    ids = list(range(count)) if row_ids is None else [int(value) for value in row_ids]
    if len(ids) != count or len(set(ids)) != count or (records is not None and len(records) != count):
        raise ValueError("Row IDs and optional records must match the unique fixed rows")
    if mask.ndim != 2 or best.shape != mask.shape or not mask.any(-1).all() or not best.any(-1).all() or (best & ~mask).any():
        raise ValueError("Every row needs legal candidates and a nonempty legal best set")
    mps_available = include_mps and torch.backends.mps.is_available()
    cpu_rng = torch.get_rng_state().clone()
    mps_rng = torch.mps.get_rng_state().clone() if mps_available else None
    state_before, grads_before = _state_hash(model.state_dict()), _grad_hash(model)
    mode_before = {name: module.training for name, module in model.named_modules()}
    source_tensors = {name: (str(value.device), str(value.dtype)) for name, value in model.state_dict().items()}
    inputs_before = _state_hash(batch)
    summaries, outputs = {}, {}
    modes = [("cpu_float32", "cpu", torch.float32), ("cpu_float64", "cpu", torch.float64)]
    if mps_available:
        modes.insert(1, ("mps_float32", "mps", torch.float32))
    for name, device, dtype in modes:
        if check is not None:
            check()
        current = deepcopy(model).to(device=device, dtype=dtype).eval()
        current_batch = {key: value.detach().to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
                         for key, value in batch.items()}
        captured = []
        handle = current.context.register_forward_hook(lambda _module, _inputs, output: captured.append(output.detach()))
        try:
            with torch.inference_mode():
                logits, _ = current(current_batch)
                probabilities = logits.softmax(-1)
                raw, probs, context = _array(logits), _array(probabilities), _array(captured[-1])
        finally:
            handle.remove()
        summaries[name], outputs[name] = summarize_outputs(raw, probs, context, mask, best)
        if dtype == torch.float32:
            saved = torch.from_numpy(raw).to(torch.float64)
            # Use the same legal candidates even if a diagnostic model emitted
            # an unusual padded value. Actual canonical padding remains finite.
            saved = saved.masked_fill(~torch.from_numpy(mask), -torch.inf)
            with torch.inference_mode():
                normalized = saved.softmax(-1).numpy().copy()
            key = name + "_softmax_float64"
            summaries[key], outputs[key] = summarize_outputs(raw, normalized, context, mask, best)
        del current, captured, current_batch, logits, probabilities
    checks = {"caller_model_state_unchanged": state_before == _state_hash(model.state_dict()),
              "caller_gradients_unchanged": grads_before == _grad_hash(model),
              "caller_module_modes_unchanged": mode_before == {name: module.training for name, module in model.named_modules()},
              "caller_model_dtypes_devices_unchanged": source_tensors == {name: (str(value.device), str(value.dtype)) for name, value in model.state_dict().items()},
              "caller_batch_unchanged": inputs_before == _state_hash(batch),
              "cpu_rng_unchanged": torch.equal(cpu_rng, torch.get_rng_state()),
              "mps_rng_unchanged": torch.equal(mps_rng, torch.mps.get_rng_state()) if mps_rng is not None else None,
              "canonical_forward_used_without_adapter": True,
              "legal_mask_validated": True}
    if any(value is False for value in checks.values()):
        raise RuntimeError(f"Inference-only precision check changed caller state: {checks}")
    comparisons = {}
    pairs = [("cpu_float32", "cpu_float32_softmax_float64"), ("cpu_float32", "cpu_float64")]
    if mps_available:
        pairs.extend([("cpu_float32", "mps_float32"), ("mps_float32", "cpu_float64"),
                      ("mps_float32", "mps_float32_softmax_float64")])
    for left, right in pairs:
        key = left + "__vs__" + right
        comparisons[key] = (_compare(outputs[left], outputs[right], mask, best, ids, records)
                            if summaries[left]["finite"] and summaries[right]["finite"]
                            else {"valid": False, "reason": "Nonfinite branch output"})
    summary = {"schema": SCHEMA, "rows": count, "row_ids": ids,
               "input_model_weights_sha256": state_before, "input_batch_sha256": inputs_before,
               "mps_requested": include_mps, "mps_available": bool(mps_available),
               "checks": checks, "branches": summaries, "comparisons": comparisons,
               "interpretation_limits": [
                   "All weights are frozen. CPU float64 widens the saved weights and inputs; it is not a high-precision training replay.",
                   "Float64 normalization of saved CPU float32 logits cannot recover already-rounded score differences or change the raw-logit argmax.",
                   "Top-tie first-index resolution can change across precision when upstream score differences are very small.",
                   "These endpoint states do not establish guard predictive value, false-positive rate, or future training stability."],
               "array_convention": "Padding logits and centered logits are stored as zero; use candidate_mask. Forced-row top-two margins and absent-nonbest gaps are zero sentinels; use mask counts and has_nonbest."}
    arrays = {"candidate_mask": mask, "best_mask": best, "row_ids": np.asarray(ids, dtype=np.int64)}
    for name, branch in outputs.items():
        arrays.update({name + "__" + key: value for key, value in branch.items()})
    if check is not None:
        check()
    return summary, arrays
