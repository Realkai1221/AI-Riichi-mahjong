"""Exact discard-shape features from actor count channels, never labels.

This is the pinned teacher's public-information calculation, not an independent
Mahjong oracle. Improving types exclude types with no unseen physical copies;
they are not yaku/furiten-aware legal winning waits.
"""

from __future__ import annotations

import numpy as np

from .policies import _discard_score
from .semantics import ACTION_KINDS


EFFICIENCY_SCHEMA = "discard-efficiency-v1"
EFFICIENCY_FIELDS = 4
EFFICIENCY_SCALES = (8.0, 136.0, 34.0, 1.0)
_DISCARD_ID = ACTION_KINDS.index("DISCARD") + 1


def efficiency_manifest() -> dict:
    return {"schema": EFFICIENCY_SCHEMA,
            "columns": ["post_discard_shanten", "visible_ukeire", "nonexhausted_improving_type_count", "applicable"],
            "divisors": list(EFFICIENCY_SCALES), "dtype": "float32", "not_applicable": [0, 0, 0, 0],
            "visibility": "Own hand plus deduplicated public tiles; unchanged when discarding.",
            "source": "Pinned riichienv-0.4.10 shanten via policies._discard_score; no teacher labels as inputs."}


def exact_efficiency_from_encoded(tiles: np.ndarray, action_tokens: np.ndarray) -> np.ndarray:
    """Return unscaled int16 [N,4] from actor-only arrays for exact audits.

    Own and visible counts are exact quarter-count float channels. Red/drawn
    semantic variants retain separate rows and share the same shape score.
    No caller-provided teacher preference or target mask is accepted.
    """
    tiles, action_tokens = np.asarray(tiles), np.asarray(action_tokens)
    if tiles.shape != (34, 23) or tiles.dtype.kind not in "fi":
        raise ValueError("Expected actor tile channels with shape (34, 23)")
    if action_tokens.ndim != 2 or action_tokens.shape[1] != 8 or action_tokens.dtype.kind not in "iu":
        raise ValueError("Expected integer semantic action tokens with shape (N, 8)")
    own, seen = tiles[:, 0] * 4, tiles[:, 4] * 4
    if (not np.isfinite(own).all() or not np.isfinite(seen).all()
            or np.any((own < 0) | (own > 4) | (seen < own) | (seen > 4))
            or not np.array_equal(own, own.astype(np.int64))
            or not np.array_equal(seen, seen.astype(np.int64))):
        raise ValueError("Actor counts must decode exactly to integral physical-copy counts in [0,4]")
    if np.any((action_tokens[:, 0] < 1) | (action_tokens[:, 0] > len(ACTION_KINDS))):
        raise ValueError("Unknown semantic action kind")
    counts, visible = tuple(map(int, own)), tuple(map(int, seen))
    result = np.zeros((len(action_tokens), EFFICIENCY_FIELDS), dtype=np.int16)
    scored = {}
    for index, token in enumerate(action_tokens):
        if int(token[0]) != _DISCARD_ID:
            continue
        kind = int(token[1]) - 1
        if not 0 <= kind < 34 or counts[kind] == 0:
            raise ValueError("Legal discard type is absent from the actor hand")
        if sum(counts) not in (2, 5, 8, 11, 14):
            raise ValueError("A discard requires a valid concealed hand length before discard")
        if kind not in scored:
            score = _discard_score(counts, visible, kind)
            scored[kind] = (score.shanten, score.ukeire, len(score.improving_types), 1)
        result[index] = scored[kind]
    return result


def efficiency_from_encoded(tiles: np.ndarray, action_tokens: np.ndarray) -> np.ndarray:
    """Return float32 candidate inputs; exact unscaled values remain auditable."""
    exact = exact_efficiency_from_encoded(tiles, action_tokens)
    return exact.astype(np.float32) / np.asarray(EFFICIENCY_SCALES, dtype=np.float32)
