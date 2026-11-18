"""Read-only endpoint fit diagnostics on explicitly frozen row indices.

No data selection, checkpoint loading, training, or persistence occurs here.
The caller owns the numerical lease and supplies its deadline/resource check.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .package_a_stability import CALL_KINDS, KAN_KINDS, ContextCapture, smoothed_targets
from .semantics import ACTION_KINDS
from .stage2b_training import COMPONENTS, DISCARD, STRATA, _index_hash
from .training import _sync, collate_training

SCHEMA = "package-a-diagnosis-fit-v1"
PROBABILITY_NAMES = (
    "original_ce", "original_kl", "smooth_ce", "smooth_kl", "entropy",
    "normalized_entropy", "best_mass", "nonbest_mass", "call_mass", "kan_mass",
)
METRIC_NAMES = tuple(f"{name}_f32" for name in PROBABILITY_NAMES) + tuple(
    f"{name}_f64" for name in PROBABILITY_NAMES) + (
    "tie_agreement", "exact_agreement", "target_nonbest_mass", "best_minus_nonbest_max_gap",
    "row_offset", "row_offset_abs", "raw_logit_abs_mean", "raw_logit_abs_max", "centered_logit_abs_mean",
    "centered_logit_abs_max", "context_l2", "context_rms", "frozen_rank_ce_f32",
)


def discard_error(record: Mapping, chosen: int) -> tuple[int, int, int, int, bool]:
    """Match Stage 2B severity semantics; first four values are zero when correct.

    A different action kind is counted separately, without pretending its
    diagnostic shanten/ukeire sentinel is an actual discard consequence.
    """
    tokens = record["encoded"]["action_tokens"]
    teacher = int(record["chosen_index"])
    applicable = len(tokens) > 1 and int(tokens[teacher, 0]) == DISCARD
    if not applicable:
        return 0, 0, -1, 0, False
    preferences = np.asarray(record["teacher_preference_keys"])
    if preferences.shape != (len(tokens), len(COMPONENTS)):
        raise ValueError("Diagnostic preference keys must have shape [candidates,5]")
    best, predicted = preferences[teacher], preferences[chosen]
    correct = bool(record["best_mask"][chosen])
    if correct != bool(np.array_equal(best, predicted)):
        raise ValueError("Preference keys disagree with teacher best mask")
    if correct:
        return 0, 0, -1, 0, True
    first = int(np.flatnonzero(best != predicted)[0])
    if predicted[first] <= best[first]:
        raise ValueError("Chosen preference unexpectedly improves the teacher minimum")
    wrong_kind = int(int(tokens[chosen, 0]) != DISCARD)
    if wrong_kind:
        return 0, 0, first, 1, True
    delta = int(predicted[2]) - int(best[2])
    lost = max(0, int(predicted[3]) - int(best[3])) if delta == 0 else 0
    return max(0, delta), lost, first, 0, True


def _float64_probabilities(z, mask, q, smooth, best, calls, kans):
    """Normalize the identical stored forward logits, not a float64 forward."""
    shifted = np.where(mask, z - np.max(np.where(mask, z, -np.inf), axis=1)[:, None], -np.inf)
    denominator = np.exp(shifted).sum(axis=1, keepdims=True)
    logp = shifted - np.log(denominator)
    p = np.exp(logp)
    safe_logp = np.where(mask, logp, 0.)
    # This corrects only float32 target-sum rounding for normalized-distribution
    # KL; the f32 columns above retain the exact training target arithmetic.
    q = q / q.sum(axis=1, keepdims=True)
    smooth = smooth / smooth.sum(axis=1, keepdims=True)
    def ce_kl(target):
        ce = -(target * safe_logp).sum(axis=1)
        entropy = -(target * np.log(np.maximum(target, np.finfo(np.float64).tiny))).sum(axis=1)
        return ce, ce - entropy
    ce, kl = ce_kl(q)
    smooth_ce, smooth_kl = ce_kl(smooth)
    entropy = -(p * safe_logp).sum(axis=1)
    counts = mask.sum(axis=1)
    normalized = np.divide(entropy, np.log(np.maximum(counts, 2)), out=np.zeros_like(entropy), where=counts > 1)
    return np.column_stack((ce, kl, smooth_ce, smooth_kl, entropy, normalized,
        (p * best).sum(axis=1), (p * (mask & ~best)).sum(axis=1),
        (p * calls).sum(axis=1), (p * kans).sum(axis=1)))


def _group_summary(arrays: Mapping[str, np.ndarray], selected: np.ndarray) -> dict[str, Any]:
    values = arrays["row_metrics"][selected]
    metrics = {}
    for column, name in enumerate(METRIC_NAMES):
        finite = values[:, column][np.isfinite(values[:, column])]
        n = int(finite.size)
        total = float(finite.sum(dtype=np.float64))
        metrics[name] = {"sum": total, "denominator": n, "mean": total / n if n else None,
                         "minimum": float(finite.min()) if n else None,
                         "maximum": float(finite.max()) if n else None}
    discard = selected & arrays["optional_teacher_discard"]
    denominator = int(discard.sum())
    def severity(key):
        positive = arrays[key][discard]
        positive = positive[positive > 0]
        histogram = Counter(map(int, positive))
        count, total = int(positive.size), int(positive.sum())
        return {"count": count, "total": total, "denominator": denominator,
                "rate": count / denominator if denominator else None,
                "mean_when_positive": total / count if count else None,
                "maximum": max(histogram) if histogram else None,
                "histogram": {str(k): v for k, v in sorted(histogram.items())}}
    first = arrays["first_preference_difference"][discard]
    return {"rows": int(selected.sum()), "metrics": metrics,
            "severity": {"optional_teacher_discard_rows": denominator,
                "shanten_increments": severity("shanten_increment"),
                "same_shanten_lost_ukeire": severity("lost_ukeire"),
                "wrong_action_kind_errors": int(arrays["wrong_action_kind"][discard].sum()),
                "first_difference_counts": {key: int((first == i).sum()) for i, key in enumerate(COMPONENTS)}}}


def summarize_rows(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Aggregate row-weighted means with their exact numerators/denominators."""
    n = len(arrays["indices"])
    optional = arrays["legal_count"] > 1
    discard = arrays["optional_teacher_discard"]
    masks = {"all": np.ones(n, dtype=bool), "optional": optional, "forced": ~optional,
             "optional_teacher_discard": discard}
    for code, name in enumerate(STRATA):
        stratum = arrays["stratum"] == code
        masks[f"stratum:{name}"] = stratum
        masks[f"stratum:{name}:optional"] = stratum & optional
        masks[f"stratum:{name}:forced"] = stratum & ~optional
        masks[f"stratum:{name}:optional_teacher_discard"] = stratum & discard
    for count in sorted(set(map(int, arrays["legal_count"]))):
        masks[f"legal_count:{count}"] = arrays["legal_count"] == count
    for feature in ("call_available", "kan_available"):
        for available in (False, True):
            mask = arrays[feature] == available
            masks[f"{feature}:{available}"] = mask
            masks[f"optional:{feature}:{available}"] = optional & mask
            masks[f"optional_teacher_discard:{feature}:{available}"] = discard & mask
            for code, name in enumerate(STRATA):
                masks[f"stratum:{name}:optional:{feature}:{available}"] = optional & mask & (arrays["stratum"] == code)
    return {name: _group_summary(arrays, mask) for name, mask in masks.items()}


def evaluate_fit(model: Any, dataset: Sequence, *, indices: Sequence[int], device: Any = "cpu",
                 batch_size: int = 256, epsilon: float = .01,
                 check: Callable[[], None] | None = None,
                 progress: Callable[[int, int], None] | None = None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """One inference pass returning JSON summaries and pickle-free flat arrays.

    All candidate arrays share candidate_offsets. Only legal candidates are
    serialized; original labels are copied without mutation. NaN in the gap
    column denotes no non-best candidate and receives a separate denominator.
    """
    started = time.perf_counter()
    indices = list(indices)
    if (not indices or batch_size < 1 or len(indices) != len(set(indices))
            or any(type(i) is not int or not 0 <= i < len(dataset) for i in indices)):
        raise ValueError("Fit diagnosis requires unique valid explicit indices")
    if next(model.parameters()).dtype != torch.float32:
        raise ValueError("Fit diagnosis requires the deployed float32 forward; precision forwards are separate")
    device = torch.device(device)
    saved_modes = [(module, module.training) for module in model.modules()]
    model.eval()
    capture = ContextCapture(model)
    chunks: dict[str, list[np.ndarray]] = {key: [] for key in (
        "logits", "action_kinds", "best", "q", "q_smooth", "row_metrics", "chosen_index",
        "teacher_index", "stratum", "legal_count", "call_available", "kan_available",
        "optional_teacher_discard", "shanten_increment", "lost_ukeire", "first_preference_difference",
        "wrong_action_kind")}
    try:
        with torch.inference_mode():
            for offset in range(0, len(indices), batch_size):
                if check is not None:
                    check()
                records = [dataset[i] for i in indices[offset:offset + batch_size]]
                if any(record.get("stratum") not in STRATA for record in records):
                    raise ValueError("Fit rows require a teacher/student stratum")
                batch, labels = collate_training(records, device, "soft")
                logits, ranks = model(batch)
                mask = batch["candidate_mask"]
                if capture.value is None or not bool(torch.isfinite(logits[mask]).all()
                    and torch.isfinite(ranks).all() and torch.isfinite(capture.value).all()):
                    raise FloatingPointError("Nonfinite or missing forward diagnostics")
                smooth = smoothed_targets(labels, mask, epsilon)
                logp = logits.log_softmax(-1)
                p = logits.softmax(-1)
                q = labels["q"]
                ce = -(q * logp).sum(-1)
                kl = ce + (q * q.clamp_min(1e-30).log()).sum(-1)
                sce = -(smooth * logp).sum(-1)
                skl = sce + (smooth * smooth.clamp_min(1e-30).log()).sum(-1)
                entropy = -(p * logp).sum(-1)
                count = mask.sum(-1)
                normalized = torch.where(count > 1, entropy / count.clamp_min(2).log(), torch.zeros_like(entropy))
                kind = batch["action_tokens"][..., 0]
                calls, kans = torch.zeros_like(mask), torch.zeros_like(mask)
                for value in CALL_KINDS:
                    calls |= (kind == value) & mask
                for value in KAN_KINDS:
                    kans |= (kind == value) & mask
                best = labels["best"]
                chosen = logits.argmax(-1)
                if not bool(mask.gather(1, chosen[:, None]).all()):
                    raise ValueError("Policy chose a padded candidate")
                f32 = torch.stack((ce, kl, sce, skl, entropy, normalized,
                    (p * best).sum(-1), (p * (mask & ~best)).sum(-1),
                    (p * calls).sum(-1), (p * kans).sum(-1)), 1).cpu().numpy().astype(np.float64)
                z = logits.cpu().numpy().astype(np.float64)
                legal = mask.cpu().numpy()
                best_np = best.cpu().numpy()
                q_np, smooth_np = q.cpu().numpy(), smooth.cpu().numpy()
                call_np, kan_np = calls.cpu().numpy(), kans.cpu().numpy()
                count_np = count.cpu().numpy()
                context = capture.value.cpu().numpy().astype(np.float64)
                f64 = _float64_probabilities(z, legal, q_np.astype(np.float64), smooth_np.astype(np.float64),
                                             best_np, call_np, kan_np)
                row_offset = np.where(legal, z, 0.).sum(1) / count_np
                centered = z - row_offset[:, None]
                raw_abs, centered_abs = np.where(legal, np.abs(z), 0.), np.where(legal, np.abs(centered), 0.)
                nonbest = legal & ~best_np
                gaps = np.max(np.where(best_np, z, -np.inf), axis=1) - np.max(np.where(nonbest, z, -np.inf), axis=1)
                gaps[~nonbest.any(1)] = np.nan
                choice_np = chosen.cpu().numpy()
                teacher_np = labels["chosen"].cpu().numpy()
                auxiliary = np.column_stack((best_np[np.arange(len(records)), choice_np], choice_np == teacher_np,
                    (smooth_np * nonbest).sum(1), gaps, row_offset, np.abs(row_offset), raw_abs.sum(1) / count_np, raw_abs.max(1),
                    centered_abs.sum(1) / count_np, centered_abs.max(1), np.sqrt((context * context).sum(1)),
                    np.sqrt((context * context).mean(1)), F.cross_entropy(ranks, labels["rank"], reduction="none").cpu().numpy()))
                metrics = np.column_stack((f32, f64, auxiliary))
                allowed_nan = METRIC_NAMES.index("best_minus_nonbest_max_gap")
                if not np.isfinite(np.delete(metrics, allowed_nan, axis=1)).all():
                    raise FloatingPointError("Nonfinite diagnostic metric")
                severity = [discard_error(record, int(choice)) for record, choice in zip(records, choice_np)]
                values = {"logits": z[legal].astype(np.float32), "action_kinds": kind.cpu().numpy()[legal].astype(np.int8),
                    "best": best_np[legal], "q": q_np[legal], "q_smooth": smooth_np[legal], "row_metrics": metrics,
                    "chosen_index": choice_np.astype(np.int16), "teacher_index": teacher_np.astype(np.int16),
                    "stratum": np.asarray([STRATA.index(record["stratum"]) for record in records], dtype=np.int8),
                    "legal_count": count_np.astype(np.int16), "call_available": call_np.any(1), "kan_available": kan_np.any(1),
                    "optional_teacher_discard": np.asarray([v[4] for v in severity], dtype=bool),
                    "shanten_increment": np.asarray([v[0] for v in severity], dtype=np.int16),
                    "lost_ukeire": np.asarray([v[1] for v in severity], dtype=np.int16),
                    "first_preference_difference": np.asarray([v[2] for v in severity], dtype=np.int8),
                    "wrong_action_kind": np.asarray([v[3] for v in severity], dtype=bool)}
                for key, value in values.items():
                    chunks[key].append(value)
                if progress is not None:
                    progress(offset + len(records), len(indices))
        _sync(device)
        if check is not None:
            check()
        arrays = {key: np.concatenate(value, axis=0) for key, value in chunks.items()}
        arrays["indices"] = np.asarray(indices, dtype=np.int64)
        arrays["candidate_offsets"] = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(arrays["legal_count"], dtype=np.int64)))
        arrays["metric_names"] = np.asarray(METRIC_NAMES)
        groups = summarize_rows(arrays)
        return {"schema": SCHEMA, "complete": True, "rows": len(indices), "requested_rows": len(indices),
                "indices_sha256": _index_hash(indices), "epsilon": epsilon, "device": str(device),
                "forward_dtype": "float32", "elapsed_seconds": time.perf_counter() - started,
                "groups": groups, "array_bytes": sum(value.nbytes for value in arrays.values()),
                "scope": "Descriptive decision-weighted endpoint fit. Optional and forced rows remain separate. Training is teacher-generated; compare its teacher rows with validation teacher rows before interpreting a generalization gap.",
                "precision": "f32 is actual backend float32 softmax/log_softmax and training-target arithmetic. f64 normalizes the identical forward logits on CPU and renormalizes copied float32 targets to unit mass; it is not a float64 model forward. NaN gap means no legal non-best candidate and is excluded using a separately reported denominator.",
                "candidate_storage": "Legal candidates only, indexed by candidate_offsets. Centered logits are logits minus each row_offset. Stored q and q_smooth are unmodified float32 target values."}, arrays
    finally:
        capture.close()
        for module, training in saved_modes:
            module.training = training
