"""Versioned actor-only encodings for semantic legal actions.

Physical IDs are used only to count/deduplicate known tiles, then discarded.
No function accepts an auditor state, teacher scores, or outcome labels.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .player import EventView, PlayerView, is_closed_hand, mjai_tile_type, tile_type, visible_tile_counts
from .semantics import ACTION_KINDS, SemanticAction, semantic_actions
from .efficiency import efficiency_from_encoded, efficiency_manifest


FEATURE_SCHEMA = "actor-features-v1"
FEATURE_SCHEMA_V2 = "actor-features-v2"
MAX_HISTORY = 128
TILE_CHANNELS = 23
GLOBAL_FEATURES = 45
HISTORY_FIELDS = 10
ACTION_FIELDS = 8
RED_FIVES = frozenset((16, 52, 88))
EVENT_KINDS = (
    "start_game", "start_kyoku", "tsumo", "dahai", "chi", "pon", "daiminkan",
    "ankan", "kakan", "reach", "reach_accepted", "dora", "hora", "ryukyoku",
    "end_kyoku", "end_game",
)
_EVENT_IDS = {kind: index + 1 for index, kind in enumerate(EVENT_KINDS)}
_ACTION_IDS = {kind: index + 1 for index, kind in enumerate(ACTION_KINDS)}


def feature_manifest(feature_schema: str = FEATURE_SCHEMA) -> dict:
    if feature_schema not in (FEATURE_SCHEMA, FEATURE_SCHEMA_V2):
        raise ValueError(f"Unsupported feature schema: {feature_schema}")
    manifest = {
        "schema": feature_schema,
        "tiles": [34, TILE_CHANNELS], "globals": [GLOBAL_FEATURES],
        "history": [MAX_HISTORY, HISTORY_FIELDS], "history_mask": [MAX_HISTORY],
        "action_tokens": ["legal_semantic_candidates", ACTION_FIELDS],
        "history_policy": "latest_128_events; explicit length/overflow globals; full current public tile summaries",
        "event_kinds": list(EVENT_KINDS), "action_kinds": list(ACTION_KINDS),
        "physical_copy_ids": "not encoded",
    }
    if feature_schema == FEATURE_SCHEMA_V2:
        manifest["candidate_efficiency"] = {"shape": ["legal_semantic_candidates", 4], **efficiency_manifest()}
    return manifest


def _relative(seat: int | None, actor: int) -> int:
    return 0 if seat is None else 1 + (seat - actor) % 4


def _ternary(value: bool | None) -> int:
    return 0 if value is None else (2 if value else 1)


def _semantic_tile(kind: int, red: bool) -> int:
    if not 0 <= kind < 34:
        raise ValueError("Invalid semantic tile type")
    return 1 + kind + 34 * bool(red)


def encode_history_events(events: Sequence[EventView], seat: int) -> np.ndarray:
    """Encode stable event rows, independent of the current prefix length.

    Columns: kind, relative actor, relative target, tile type+1, red flag,
    ternary tsumogiri, four consumed semantic tiles. Zero is padding/unknown.
    Event.data is deliberately not traversed: scores/state have explicit fields.
    """
    rows = np.zeros((len(events), HISTORY_FIELDS), dtype=np.int16)
    for index, event in enumerate(events):
        if event.kind not in _EVENT_IDS:
            raise ValueError(f"Unsupported actor event kind {event.kind!r}")
        consumed = []
        for spelling in event.consumed:
            kind = mjai_tile_type(spelling)
            if kind is None:
                raise ValueError("A public consumed tile cannot be unknown")
            consumed.append(_semantic_tile(kind, spelling.endswith("r")))
        if len(consumed) > 4:
            raise ValueError("More than four consumed tiles in one event")
        rows[index, :6] = (
            _EVENT_IDS[event.kind], _relative(event.actor, seat),
            _relative(event.target, seat),
            0 if event.tile_type is None else 1 + event.tile_type,
            int(bool(event.tile and event.tile.endswith("r"))),
            _ternary(event.tsumogiri),
        )
        rows[index, 6:6 + len(consumed)] = sorted(consumed)
    return rows


def _dora_type(indicator: int) -> int:
    if indicator < 27:
        return (indicator // 9) * 9 + (indicator % 9 + 1) % 9
    if indicator < 31:
        return 27 + (indicator - 27 + 1) % 4
    return 31 + (indicator - 31 + 1) % 3


def _tile_features(view: PlayerView) -> np.ndarray:
    tiles = np.zeros((34, TILE_CHANNELS), dtype=np.float32)
    for tile in view.hand:
        kind = tile_type(tile)
        tiles[kind, 0] += 0.25
        tiles[kind, 1] += int(tile in RED_FIVES)
    if view.drawn_tile is not None:
        if view.drawn_tile not in view.hand:
            raise ValueError("Actor draw is not in the actor's own hand")
        kind = tile_type(view.drawn_tile)
        tiles[kind, 2] = 1
        tiles[kind, 3] = int(view.drawn_tile in RED_FIVES)
    tiles[:, 4] = np.asarray(visible_tile_counts(view), dtype=np.float32) / 4
    for tile in view.dora_indicators:
        kind = tile_type(tile)
        tiles[kind, 5] += 0.25
        tiles[_dora_type(kind), 6] += 0.2
    for relative in range(4):
        seat = (view.seat + relative) % 4
        for tile in view.discards[seat]:
            kind = tile_type(tile)
            tiles[kind, 7 + relative] += 0.25
            tiles[kind, 19 + relative] += int(tile in RED_FIVES)
        for meld in view.melds[seat]:
            for tile in meld.tiles:
                kind = tile_type(tile)
                tiles[kind, 11 + relative] += 0.25
                tiles[kind, 15 + relative] += int(tile in RED_FIVES)
    return tiles


def _global_features(view: PlayerView) -> np.ndarray:
    values = np.zeros(GLOBAL_FEATURES, dtype=np.float32)
    for relative in range(4):
        seat = (view.seat + relative) % 4
        melds = view.melds[seat]
        values[relative] = view.scores[seat] / 50_000
        values[4 + relative] = view.riichi_declared[seat]
        values[8 + relative] = len(view.discards[seat]) / 30
        values[12 + relative] = len(melds) / 4
        values[16 + relative] = sum(meld.opened for meld in melds) / 4
        values[20 + relative] = sum(not meld.opened and len(meld.tiles) == 4 for meld in melds) / 4
    values[24 + view.seat] = 1  # Absolute starting-seat identity matters for ties.
    values[28 + (view.oya - view.seat) % 4] = 1
    if not 0 <= view.round_wind < 4:
        raise ValueError("Unsupported round wind")
    values[32 + view.round_wind] = 1
    overflow = max(0, len(view.history) - MAX_HISTORY)
    values[36:] = (
        view.honba / 10, view.riichi_sticks / 10, view.hand_id / 20,
        len(view.hand) / 14, is_closed_hand(view), len(view.history) / MAX_HISTORY,
        bool(overflow), overflow / MAX_HISTORY,
        sum(event.kind == "tsumo" for event in view.history) / 70,
    )
    return values


def encode_view(view: PlayerView, actions: Sequence[SemanticAction] | None = None, *,
                feature_schema: str = FEATURE_SCHEMA) -> dict[str, np.ndarray]:
    """Encode one legal player view; action row order follows ``actions``.

    Supplied candidates must cover the full semantic legal set exactly once.
    Representatives and arbitrary copy IDs never enter any feature array.
    """
    if feature_schema not in (FEATURE_SCHEMA, FEATURE_SCHEMA_V2):
        raise ValueError(f"Unsupported feature schema: {feature_schema}")
    legal = semantic_actions(view)
    selected = tuple(legal if actions is None else actions)
    legal_by_key = {action.key: action for action in legal}
    if (len(selected) != len(legal) or len({action.key for action in selected}) != len(selected)
            or any(action != legal_by_key.get(action.key) for action in selected)):
        raise ValueError("Candidates must contain each current semantic legal action exactly once")
    history = np.zeros((MAX_HISTORY, HISTORY_FIELDS), dtype=np.int16)
    recent = encode_history_events(view.history[-MAX_HISTORY:], view.seat)
    history[:len(recent)] = recent
    history_mask = np.zeros(MAX_HISTORY, dtype=np.bool_)
    history_mask[:len(recent)] = True
    candidates = np.zeros((len(selected), ACTION_FIELDS), dtype=np.int16)
    for index, action in enumerate(selected):
        if action.kind not in _ACTION_IDS:
            raise ValueError(f"Unsupported action kind {action.kind!r}")
        if len(action.consumed) > 4:
            raise ValueError("More than four consumed tiles in one action")
        candidates[index, :4] = (
            _ACTION_IDS[action.kind], 0 if action.tile_type is None else 1 + action.tile_type,
            action.red, _ternary(action.tsumogiri),
        )
        candidates[index, 4:4 + len(action.consumed)] = [
            _semantic_tile(kind, red) for kind, red in action.consumed
        ]
    result = {"tiles": _tile_features(view), "globals": _global_features(view),
              "history": history, "history_mask": history_mask, "action_tokens": candidates}
    if feature_schema == FEATURE_SCHEMA_V2:
        result["candidate_efficiency"] = efficiency_from_encoded(result["tiles"], candidates)
    return result
