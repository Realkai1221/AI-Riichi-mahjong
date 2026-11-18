"""Auditor-only commitments to the exact immutable actor view.

This is distinct from PlayerAdapter's input/idempotence fingerprint. It binds
the accumulated safe history and hand ID, as well as every current view field.
"""

from __future__ import annotations

from dataclasses import fields
import hashlib
from typing import Any

from .player import PlayerView


VIEW_HASH_SCHEMA = "riichi-ai-player-view-v2"
_STATE_FIELDS = tuple(field.name for field in fields(PlayerView) if field.name != "history")


class PlayerViewHasher:
    """Cache an append-only history prefix; reset safely if the prefix changes.

    Views/events are deeply immutable. Comparing the old prefix verifies that
    the bytes committed are those of the actual supplied view, not an assumed
    event stream. A corrupted/reset/truncated prefix is rehashed, not ignored.
    """

    def __init__(self) -> None:
        self._history: tuple[Any, ...] = ()
        self._history_hash = hashlib.sha256()

    def digest(self, view: PlayerView) -> str:
        history = view.history
        if history[:len(self._history)] != self._history:
            self._history = ()
            self._history_hash = hashlib.sha256()
        for event in history[len(self._history):]:
            self._history_hash.update((repr(event) + "\n").encode())
        self._history = history
        state = tuple((name, getattr(view, name)) for name in _STATE_FIELDS)
        payload = (VIEW_HASH_SCHEMA, state, self._history_hash.hexdigest())
        return hashlib.sha256(repr(payload).encode()).hexdigest()


def reference_view_digest(view: PlayerView) -> str:
    """Recompute all history for reference checks; no cache is reused."""
    return PlayerViewHasher().digest(view)
