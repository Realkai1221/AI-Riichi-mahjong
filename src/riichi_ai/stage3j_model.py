"""Fixed Stage 3J choices using immutable Stage 3I actor/head outputs.

An exact tie means equality of the original finite float32 actor logits. It
does not mean equal Mahjong value. Fractional scores reuse Stage 3I's float64
arithmetic; neither choices nor fingerprints accept complete simulator state.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from .features import ACTION_FIELDS, GLOBAL_FEATURES, HISTORY_FIELDS, MAX_HISTORY, TILE_CHANNELS
from .semantics import ACTION_KINDS, SEMANTIC_ACTION_VERSION
from .stage3i_model import (
    ELIGIBILITY_THRESHOLD, PRIOR_FLOOR, WIN_IDS, correction_choices, first_win_index,
)


SCHEMA = "riichi-ai-stage3j-choices-v1"
ACTOR_FINGERPRINT_SCHEMA = "riichi-ai-stage3j-actor-feature-fingerprint-v1"
HASH_TIE_SALT = "riichi-ai-stage3j-hash-tie-v1"
FRACTIONAL_STRENGTH = .25
MODES = ("reference_wrapped", "learned_tie", "fractional", "hash_tie")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def actor_feature_fingerprint(encoded: Mapping[str, np.ndarray]) -> str:
    """Hash exactly one audited v2 ``encode_view`` result, without padding.

    This deliberately does not accept ``PlayerViewHasher`` output, physical
    tokens, seed, wall, labels or provenance. The caller must obtain the arrays
    from ``encode_view`` rather than manufacture an untrusted dictionary.
    Explicit little-endian dtypes and C-order bytes make the commitment
    reproducible regardless of array memory layout. Original dtype is checked
    before normalization: a silent float64-to-float32 rounding is forbidden.
    """
    fields = {
        "tiles": (np.dtype("float32"), (34, TILE_CHANNELS)),
        "globals": (np.dtype("float32"), (GLOBAL_FEATURES,)),
        "history": (np.dtype("int16"), (MAX_HISTORY, HISTORY_FIELDS)),
        "history_mask": (np.dtype("bool"), (MAX_HISTORY,)),
        "action_tokens": (np.dtype("int16"), None),
        "candidate_efficiency": (np.dtype("float32"), None),
    }
    if not isinstance(encoded, Mapping) or set(encoded) != set(fields):
        raise ValueError("Fingerprint accepts audited v2 actor feature fields only")
    actions = encoded["action_tokens"]
    if (not isinstance(actions, np.ndarray) or actions.ndim != 2
            or actions.shape[0] < 1 or actions.shape[1] != ACTION_FIELDS):
        raise ValueError("Fingerprint requires a nonempty semantic action matrix")
    fields["action_tokens"] = (np.dtype("int16"), actions.shape)
    fields["candidate_efficiency"] = (np.dtype("float32"), (actions.shape[0], 4))
    digest = hashlib.sha256(_json_bytes(ACTOR_FINGERPRINT_SCHEMA))
    for name in sorted(fields):
        dtype, shape = fields[name]
        value = encoded[name]
        if (not isinstance(value, np.ndarray) or value.dtype != dtype or value.shape != shape
                or not np.isfinite(value).all()):
            raise ValueError(f"Invalid audited actor feature dtype, shape or finiteness: {name}")
        canonical_dtype = dtype.newbyteorder("<")
        digest.update(_json_bytes([name, canonical_dtype.str, list(shape)]))
        digest.update(np.asarray(value, dtype=canonical_dtype, order="C").tobytes(order="C"))
    return digest.hexdigest()


def _semantic_keys(keys: Sequence[str] | None, kinds: np.ndarray) -> tuple[str, ...]:
    """Validate semantic keys and the canonical first-index tie convention."""
    if (not isinstance(keys, (tuple, list)) or len(keys) != len(kinds)
            or any(type(key) is not str for key in keys) or len(set(keys)) != len(keys)):
        raise ValueError("Hash tie control requires unique semantic keys for every candidate")
    ordering = []
    for key, kind in zip(keys, kinds):
        try:
            item = json.loads(key)
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid semantic action key") from exc
        if (not isinstance(item, list) or len(item) != 6 or item[0] != SEMANTIC_ACTION_VERSION
                or item[1] != ACTION_KINDS[int(kind) - 1]
                or (item[2] is not None and (type(item[2]) is not int or not 0 <= item[2] < 34))
                or type(item[3]) is not bool
                or (item[4] is not None and type(item[4]) is not bool)
                or not isinstance(item[5], list) or len(item[5]) > 4
                or any(not isinstance(tile, list) or len(tile) != 2 or type(tile[0]) is not int
                       or not 0 <= tile[0] < 34 or type(tile[1]) is not bool for tile in item[5])
                or key.encode("utf-8") != _json_bytes(item)):
            raise ValueError("Invalid or noncanonical semantic action key")
        consumed = tuple((tile[0], tile[1]) for tile in item[5])
        if consumed != tuple(sorted(consumed)):
            raise ValueError("Noncanonical consumed semantic tiles")
        ordering.append((int(kind) - 1, -1 if item[2] is None else item[2], int(item[3]),
                         0 if item[4] is True else 1 if item[4] is False else 2, consumed))
    if ordering != sorted(ordering):
        raise ValueError("Candidates must follow canonical semantic ordering")
    return tuple(keys)


def stage3j_choices(logits: np.ndarray, residuals: np.ndarray | None,
                    probabilities: np.ndarray, kinds: np.ndarray, *, mode: str,
                    actor_fingerprint: str | None = None,
                    semantic_keys: Sequence[str] | None = None) -> dict[str, Any]:
    """Return a fixed choice and sufficient evidence for independent arithmetic.

    The candidate order must be ``semantic_actions(view)`` order. All modes
    preserve the first legal win wrapper. Learned ties maximize A among
    eligible exact raw-top-logit candidates; hash ties minimize a fixed SHA256
    ranking of audited actor features plus semantic key. Neither tie policy
    may select a lower-logit action. The fractional mode is exactly the Stage
    3I correction at lambda .25, including its clipped prior and q mask.

    JSON evidence should store ``logits.tolist()``. An auditor reconstructing
    it must use float32 and verify equality after float64 round-trip, rather
    than silently round arbitrary recorded numbers. ``scores`` describes the
    relevant ranking (negative integer rank for hash ties); only ``prior`` is
    a reference policy score. None is a utility estimate.
    """
    if mode not in MODES:
        raise ValueError("Undeclared Stage 3J choice mode")
    if mode != "hash_tie" and actor_fingerprint is not None:
        raise ValueError("Actor fingerprint is only an input to the fixed hash tie control")
    if (not isinstance(logits, np.ndarray) or logits.dtype != np.dtype("float32")
            or logits.ndim != 1 or not len(logits) or not np.isfinite(logits).all()):
        raise ValueError("Exact ties require original finite one-dimensional float32 logits")
    kinds = np.asarray(kinds)
    if (kinds.shape != logits.shape or kinds.dtype.kind not in "iu"
            or np.any(kinds < 1) or np.any(kinds > len(ACTION_KINDS))):
        raise ValueError("Invalid semantic action kind IDs")
    if mode in ("learned_tie", "fractional") and residuals is None:
        raise ValueError("Learned modes require action residuals")
    residuals = np.zeros(len(logits), dtype=np.float64) if residuals is None else np.asarray(residuals, dtype=np.float64)
    # Shared primitive validates every q/residual entry and computes the exact
    # original float64 correction. Its strength-zero branch protects identity.
    strength = FRACTIONAL_STRENGTH if mode == "fractional" else 0.
    result = correction_choices(logits, residuals, probabilities, kinds, strength, wrapped=True)
    raw = result["raw_reference_index"]
    exact_ties = logits == logits[raw]
    eligible_ties = exact_ties & result["eligible"]
    indices = np.flatnonzero(eligible_ties)
    tie_hashes: list[str | None] = [None] * len(logits)
    if semantic_keys is not None:
        keys = _semantic_keys(semantic_keys, kinds)
    else:
        keys = None
    if mode == "learned_tie":
        scores = np.full(len(logits), -np.inf, dtype=np.float64)
        scores[eligible_ties] = residuals[eligible_ties]
        result["raw_corrected_index"] = int(scores.argmax())
        result["scores"] = scores
    elif mode == "hash_tie":
        if (type(actor_fingerprint) is not str or len(actor_fingerprint) != 64
                or any(ch not in "0123456789abcdef" for ch in actor_fingerprint) or keys is None):
            raise ValueError("Hash tie control requires an audited actor fingerprint and semantic keys")
        for index in indices:
            tie_hashes[index] = hashlib.sha256(_json_bytes([HASH_TIE_SALT, actor_fingerprint, keys[index]])).hexdigest()
        ordered = sorted(indices.tolist(), key=lambda index: (tie_hashes[index], index))
        scores = np.full(len(logits), -np.inf, dtype=np.float64)
        for rank, index in enumerate(ordered):
            scores[index] = -float(rank)
        result["raw_corrected_index"] = ordered[0]
        result["scores"] = scores
    result["selected_index"] = first_win_index(kinds, result["raw_corrected_index"])
    selected = result["selected_index"]
    gap = float(np.float64(logits[raw]) - np.float64(logits[selected]))
    unchanged = selected == result["wrapped_reference_index"]
    result.update(schema=SCHEMA, mode=mode, strength=strength,
                  exact_tie_mask=exact_ties, eligible_tie_mask=eligible_ties,
                  selected_reference_gap=gap,
                  change_classification="unchanged" if unchanged else "exact_tie" if gap == 0 else "positive_margin",
                  wrapper_active=bool(np.isin(kinds, WIN_IDS).any()),
                  wrapper_applied=selected != result["raw_corrected_index"],
                  actor_fingerprint=actor_fingerprint if mode == "hash_tie" else None,
                  tie_hashes=tie_hashes)
    return result
