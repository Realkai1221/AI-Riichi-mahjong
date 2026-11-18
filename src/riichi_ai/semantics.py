"""Versioned semantic legal candidates and transparent imitation targets.

Physical copies remain at the engine boundary. Policy/model features must use
the semantic fields, never representative IDs or physical token strings.
"""

from __future__ import annotations

from dataclasses import dataclass
import json

from .player import ActionSpec, PlayerView, tile_type, visible_tile_counts


SEMANTIC_ACTION_VERSION = "semantic-actions-v1"
TEACHER_VERSION = "heuristics-v2"
ACTION_KINDS = (
    "DISCARD", "CHI", "PON", "DAIMINKAN", "ANKAN", "KAKAN", "RIICHI",
    "RON", "TSUMO", "KYUSHU_KYUHAI", "PASS",
)
_KIND_ORDER = {kind: index for index, kind in enumerate(ACTION_KINDS)}
_RED_FIVES = frozenset((16, 52, 88))


@dataclass(frozen=True, slots=True)
class SemanticAction:
    kind: str
    tile_type: int | None
    red: bool
    tsumogiri: bool | None
    consumed: tuple[tuple[int, bool], ...]
    key: str
    representative: ActionSpec
    physical_tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TeacherDecision:
    actions: tuple[SemanticAction, ...]
    preference_keys: tuple[tuple[int, ...], ...]
    best_mask: tuple[bool, ...]
    chosen_index: int
    soft_targets: tuple[float, ...]


def _semantic_order(action: SemanticAction) -> tuple:
    """Numeric type order precedes the within-type drawn-tile preference."""
    return (
        _KIND_ORDER[action.kind],
        -1 if action.tile_type is None else action.tile_type,
        int(action.red),
        0 if action.tsumogiri is True else 1 if action.tsumogiri is False else 2,
        action.consumed,
    )


def _physical_order(action: ActionSpec) -> tuple:
    return (-1 if action.tile is None else action.tile,
            tuple(sorted(action.consume_tiles)), action.token)


def semantic_actions(view: PlayerView) -> tuple[SemanticAction, ...]:
    """Group only same-view candidates with identical observable semantics.

    Red identity, call consumption, and drawn versus held discards remain
    distinct. Source/target are fixed by this player's current decision state;
    this function is not a vocabulary for arbitrary cross-state action reuse.
    Empty legal sets return an empty tuple. Unknown kinds fail explicitly.
    """
    groups: dict[tuple, dict[str, ActionSpec]] = {}
    token_specs: dict[str, ActionSpec] = {}
    for action in view.legal_actions:
        if action.kind not in _KIND_ORDER:
            raise ValueError(f"Unsupported semantic action kind: {action.kind!r}")
        previous = token_specs.setdefault(action.token, action)
        if previous != action:
            raise ValueError("One physical action token describes different actions")
        kind = None if action.tile is None else tile_type(action.tile)
        red = action.tile in _RED_FIVES
        consumed = tuple(sorted((tile_type(tile), tile in _RED_FIVES)
                                for tile in action.consume_tiles))
        if len(set(action.consume_tiles)) != len(action.consume_tiles):
            raise ValueError("An action cannot consume a physical tile twice")
        tsumogiri = None
        if action.kind == "DISCARD":
            if action.tile is None or action.tile not in view.hand:
                raise ValueError("Legal discard is absent from player hand")
            tsumogiri = action.tile == view.drawn_tile
        signature = (action.kind, kind, red, tsumogiri, consumed)
        groups.setdefault(signature, {})[action.token] = action
    result = []
    for signature, physical in groups.items():
        ordered = sorted(physical.values(), key=_physical_order)
        key = json.dumps([SEMANTIC_ACTION_VERSION, *signature], separators=(",", ":"))
        result.append(SemanticAction(
            kind=signature[0], tile_type=signature[1], red=signature[2],
            tsumogiri=signature[3], consumed=signature[4], key=key,
            representative=ordered[0], physical_tokens=tuple(a.token for a in ordered),
        ))
    return tuple(sorted(result, key=_semantic_order))


def teacher_decision(
    view: PlayerView, defense: bool = False, *, allow_riichi: bool = True,
) -> TeacherDecision:
    """Score all semantic candidates using the v1 tactical priorities.

    Lower lexicographic keys win: (branch, unsafe riichi opponents, shanten,
    negative visible ukeire, red discard). Branches are win=0, permitted
    riichi=1, discard=2, pass=3, unsupported fallback=4. Inapplicable numeric
    fields are zero. These are heuristic preferences, not estimates of value.

    Soft targets are uniform over the best semantic classes. Actual behavior
    is deterministic: semantic order resolves ties, then the representative
    uses numeric physical IDs solely to resolve interchangeable copies.
    """
    # Lazy import keeps additive v2 policy factories independent of import order.
    from .policies import _counts, _discard_score, genbutsu_types

    actions = semantic_actions(view)
    if not actions:
        raise ValueError("The player has no legal action to choose")
    threats = tuple(genbutsu_types(view, seat) for seat in range(4)
                    if defense and seat != view.seat and view.riichi_declared[seat])
    discard_scores = {}
    if any(action.kind == "DISCARD" for action in actions):
        counts, visible = _counts(view.hand), visible_tile_counts(view)
        for action in actions:
            if action.kind == "DISCARD" and action.tile_type not in discard_scores:
                discard_scores[action.tile_type] = _discard_score(counts, visible, action.tile_type)
    keys = []
    for action in actions:
        if action.kind in ("RON", "TSUMO"):
            key = (0, 0, 0, 0, 0)
        elif action.kind == "RIICHI" and allow_riichi and not threats:
            key = (1, 0, 0, 0, 0)
        elif action.kind == "DISCARD":
            score = discard_scores[action.tile_type]
            key = (2, sum(action.tile_type not in safe for safe in threats),
                   score.shanten, -score.ukeire, int(action.red))
        elif action.kind == "PASS":
            key = (3, 0, 0, 0, 0)
        else:
            key = (4, 0, 0, 0, 0)
        keys.append(key)
    best = min(keys)
    best_mask = tuple(key == best for key in keys)
    count = sum(best_mask)
    return TeacherDecision(
        actions=actions, preference_keys=tuple(keys), best_mask=best_mask,
        chosen_index=best_mask.index(True),
        soft_targets=tuple(1.0 / count if chosen else 0.0 for chosen in best_mask),
    )
