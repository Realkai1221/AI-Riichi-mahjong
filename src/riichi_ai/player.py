"""Immutable, information-limited player boundary for RiichiEnv 0.4.10.

Only the runner owns native observations and resolves actions. Policies receive
PlayerView values and return one of their ActionSpec candidates.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
from typing import Any


@dataclass(frozen=True, slots=True)
class ActionSpec:
    kind: str
    tile: int | None
    consume_tiles: tuple[int, ...]
    token: str


@dataclass(frozen=True, slots=True)
class MeldView:
    kind: str
    tiles: tuple[int, ...]
    opened: bool
    from_player: int | None
    called_tile: int | None


@dataclass(frozen=True, slots=True)
class EventView:
    kind: str
    actor: int | None = None
    target: int | None = None
    tile: str | None = None
    tile_type: int | None = None
    consumed: tuple[str, ...] = ()
    tsumogiri: bool | None = None
    data: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class PlayerView:
    seat: int
    hand: tuple[int, ...]
    legal_actions: tuple[ActionSpec, ...]
    discards: tuple[tuple[int, ...], ...]
    melds: tuple[tuple[MeldView, ...], ...]
    scores: tuple[int, ...]
    riichi_declared: tuple[bool, ...]
    drawn_tile: int | None
    dora_indicators: tuple[int, ...]
    round_wind: int
    oya: int
    honba: int
    riichi_sticks: int
    hand_id: int
    history: tuple[EventView, ...]


def tile_type(physical_tile: int) -> int:
    if type(physical_tile) is not int or not 0 <= physical_tile < 136:
        raise ValueError("Expected a physical tile ID from 0 through 135")
    return physical_tile // 4


def mjai_tile_type(tile: str | None) -> int | None:
    """Decode a visible MJAI tile; unknown/private draws have no tile type."""
    if tile is None or tile == "?":
        return None
    honors = {"E": 27, "S": 28, "W": 29, "N": 30, "P": 31, "F": 32, "C": 33}
    if tile in honors:
        return honors[tile]
    plain = tile.removesuffix("r")
    if len(plain) == 2 and plain[0] in "123456789" and plain[1] in "mps":
        return "mps".index(plain[1]) * 9 + int(plain[0]) - 1
    raise ValueError(f"Unsupported MJAI tile: {tile!r}")


def is_closed_hand(view: PlayerView) -> bool:
    """Concealed kans preserve a closed hand; open calls do not."""
    return not any(meld.opened for meld in view.melds[view.seat])


def visible_tile_counts(view: PlayerView) -> tuple[int, ...]:
    """Count own concealed and publicly visible physical tiles exactly once.

    A called discard occurs in both a river and a meld in this engine. Set
    union removes that alias, and also prevents a dora/meld overlap from being
    counted twice. Opponents' concealed hands and unseen wall are absent.
    """
    visible = set(view.hand) | set(view.dora_indicators)
    for river in view.discards:
        visible.update(river)
    for melds in view.melds:
        for meld in melds:
            visible.update(meld.tiles)
    counts = [0] * 34
    for tile in visible:
        counts[tile_type(tile)] += 1
    return tuple(counts)


def action_spec(action: Any) -> ActionSpec:
    """Preserve every physical candidate, independent of the engine's ID mask."""
    kind = str(action.action_type).rsplit(".", 1)[-1]
    tile = action.tile
    consumed = tuple(sorted(action.consume_tiles))
    if tile is not None:
        tile_type(tile)
    for physical in consumed:
        tile_type(physical)
    token = json.dumps([kind, tile, consumed], separators=(",", ":"))
    return ActionSpec(kind, tile, consumed, token)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple((key, _freeze(item)) for key, item in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise ValueError("Unexpected event value")


# Per-event allowlists deliberately omit wall/seed/salt, opponent starting
# hands, hidden draws, and unreconciled hand-reveal/scoring-detail payloads.
_EVENT_FIELDS = {
    "start_game": (),
    "start_kyoku": ("bakaze", "kyoku", "honba", "kyotaku", "oya", "scores", "dora_marker"),
    "tsumo": (), "dahai": (),
    "chi": (), "pon": (), "daiminkan": (), "ankan": (), "kakan": (),
    "reach": (), "reach_accepted": ("scores",),
    "dora": (), "hora": ("deltas", "scores"),
    "ryukyoku": ("deltas", "scores", "reason"),
    "end_kyoku": (), "end_game": ("scores",),
}
_TILE_EVENTS = {"tsumo", "dahai", "chi", "pon", "daiminkan", "kakan"}
_CALL_EVENTS = {"chi", "pon", "daiminkan", "ankan", "kakan"}


def _safe_event(raw: dict[str, Any], seat: int) -> EventView:
    kind = raw.get("type")
    if kind not in _EVENT_FIELDS:
        raise ValueError(f"Unsupported player event type: {kind!r}")
    actor = raw.get("actor") if kind not in {"start_game", "start_kyoku", "dora", "end_kyoku", "end_game"} else None
    target = raw.get("target") if kind in _CALL_EVENTS | {"hora"} else None
    for participant in (actor, target):
        if participant is not None and (type(participant) is not int or participant not in range(4)):
            raise ValueError("Invalid player event participant")
    tile = raw.get("pai") if kind in _TILE_EVENTS else None
    if kind == "tsumo" and actor != seat:
        tile = None
    if tile == "?":
        tile = None
    if kind == "dora":
        tile = raw.get("dora_marker")
    consumed = tuple(raw.get("consumed", ())) if kind in _CALL_EVENTS else ()
    tsumogiri = raw.get("tsumogiri") if kind == "dahai" else None
    if tsumogiri is not None and type(tsumogiri) is not bool:
        raise ValueError("Invalid tsumogiri flag")
    # Validate public tile spellings; do not retain arbitrary nested payloads.
    for item in consumed:
        mjai_tile_type(item)
    data = {key: _freeze(raw[key]) for key in _EVENT_FIELDS[kind] if key in raw}
    if kind == "start_kyoku" and "tehais" in raw:
        own_hand = tuple(raw["tehais"][seat])
        for item in own_hand:
            mjai_tile_type(item)
        data["hand"] = own_hand
    return EventView(
        kind=kind, actor=actor, target=target, tile=tile,
        tile_type=mjai_tile_type(tile), consumed=consumed,
        tsumogiri=tsumogiri,
        data=tuple(sorted(data.items())),
    )


class PlayerAdapter:
    """Consume one seat's ordered observation stream and return immutable views.

    The runner must route every observation returned for this seat through this
    adapter: obtaining a native observation advances the engine's event cursor.
    An immediate repeat/copy of the same snapshot is idempotent. Reusing older
    snapshots is rejected. Call reset() explicitly at each new match.
    """

    def __init__(self, seat: int):
        if type(seat) is not int or seat not in range(4):
            raise ValueError("seat must be 0, 1, 2 or 3")
        self.seat = seat
        self.reset()

    def reset(self) -> None:
        self._hand_id = 0
        self._history: tuple[EventView, ...] = ()
        self._last_fingerprint: str | None = None
        self._last_view: PlayerView | None = None
        self._seen: set[str] = set()

    def _candidates(self, observation: Any) -> dict[str, tuple[ActionSpec, Any]]:
        if observation.player_id != self.seat:
            raise ValueError("Observation belongs to a different seat")
        candidates = {}
        for native in observation.legal_actions():
            if native.actor is not None and native.actor != self.seat:
                raise ValueError("Legal action belongs to a different seat")
            spec = action_spec(native)
            candidates.setdefault(spec.token, (spec, native))
        return candidates

    def resolve(self, spec: ActionSpec, observation: Any) -> Any:
        """Resolve against the current legal set; never synthesize a native move."""
        candidate = self._candidates(observation).get(spec.token)
        if candidate is None or candidate[0] != spec:
            raise ValueError("ActionSpec is not a current legal candidate")
        return candidate[1]

    def build(self, observation: Any) -> PlayerView:
        candidates = self._candidates(observation)
        raw = json.loads(base64.b64decode(observation.serialize_to_base64()))
        hand = tuple(sorted(observation.hand))
        drawn = raw.get("drawn_tile")
        melds = tuple(tuple(MeldView(
            kind=meld["meld_type"].upper(), tiles=tuple(sorted(meld["tiles"])),
            opened=meld["opened"],
            from_player=meld["from_who"] if meld["from_who"] >= 0 else None,
            called_tile=meld.get("called_tile"),
        ) for meld in group) for group in raw["melds"])
        state = dict(
            seat=self.seat, hand=hand,
            legal_actions=tuple(candidates[token][0] for token in sorted(candidates)),
            discards=tuple(tuple(river) for river in raw["discards"]), melds=melds,
            scores=tuple(raw["scores"]), riichi_declared=tuple(raw["riichi_declared"]),
            drawn_tile=drawn if drawn in hand else None,
            dora_indicators=tuple(raw["dora_indicators"]),
            round_wind=raw["round_wind"], oya=raw["oya"], honba=raw["honba"],
            riichi_sticks=raw["riichi_sticks"],
        )
        events = tuple(_safe_event(json.loads(event), self.seat) for event in raw["events"])
        # Hash only safe fields. Hidden changes must not alter player behavior,
        # including the adapter's idempotence/out-of-order checks.
        fingerprint = hashlib.sha256(repr((state, events)).encode()).hexdigest()
        if fingerprint == self._last_fingerprint:
            assert self._last_view is not None
            return self._last_view
        if fingerprint in self._seen:
            raise ValueError("Previously consumed observation supplied out of order")
        hand_id, history = self._hand_id, list(self._history)
        for event in events:
            if event.kind == "start_game":
                if hand_id:
                    raise ValueError("Call adapter.reset() before a new match")
                history = []
            elif event.kind == "start_kyoku":
                hand_id += 1
                history = []
            history.append(event)
        if hand_id == 0:
            raise ValueError("Missing start_kyoku: observation history was consumed or omitted")
        view = PlayerView(**state, hand_id=hand_id, history=tuple(history))
        self._hand_id, self._history = hand_id, view.history
        self._last_fingerprint, self._last_view = fingerprint, view
        self._seen.add(fingerprint)
        return view
