"""Transparent player-view-only baselines; these are not strength claims.

The two heuristics keep a closed hand, take offered wins, optionally declare
riichi, and otherwise minimize shanten then maximize visible-tile ukeire.
They deliberately pass optional calls and do not model kan or hand value.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import random
from typing import Any, Protocol

from riichienv import calculate_shanten

from .player import ActionSpec, PlayerView, tile_type, visible_tile_counts


POLICY_VERSION = "heuristics-v1"
RED_FIVES = frozenset((16, 52, 88))
_WIN_KINDS = frozenset(("RON", "TSUMO"))
_ALIASES = {
    "random": "random",
    "shanten": "shanten",
    "shanten_ukeire": "shanten",
    "defense": "defense",
    "conservative": "defense",
    "shanten_v2": "shanten_v2",
    "defense_v2": "defense_v2",
}


class Policy(Protocol):
    def choose(self, view: PlayerView) -> ActionSpec: ...

    def manifest(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class DiscardScore:
    """Shanten after discard and copies of improving tiles not yet visible.

    Ukeire counts possible unseen copies, not actual remaining wall tiles.
    The discarded tile stays visible and cannot become available again.
    """

    shanten: int
    ukeire: int
    improving_types: tuple[int, ...]


def _counts(hand: tuple[int, ...]) -> tuple[int, ...]:
    counts = [0] * 34
    for tile in hand:
        counts[tile_type(tile)] += 1
    return tuple(counts)


@lru_cache(maxsize=131_072)
def _cached_shanten(counts: tuple[int, ...]) -> int:
    # RiichiEnv 0.4.10 takes physical IDs and infers meld count from concealed
    # hand length. Red identity is irrelevant to shanten, so canonicalize it.
    physical = [kind * 4 + copy for kind, count in enumerate(counts)
                for copy in range(count)]
    return calculate_shanten(physical)


def hand_shanten(hand: tuple[int, ...]) -> int:
    """Exact pinned-engine shanten for a legal concealed physical-tile hand."""
    if len(set(hand)) != len(hand):
        raise ValueError("Physical tiles in the hand must be unique")
    if len(hand) not in (1, 2, 4, 5, 7, 8, 10, 11, 13, 14):
        raise ValueError("Expected a legal concealed hand length")
    return _cached_shanten(_counts(hand))


def shanten_cache_info() -> Any:
    """Expose bounded cache accounting for profiling; no game state is held."""
    return _cached_shanten.cache_info()


def _discard_score(
    counts: tuple[int, ...], visible: tuple[int, ...], discard_type: int,
) -> DiscardScore:
    remaining = list(counts)
    remaining[discard_type] -= 1
    if remaining[discard_type] < 0:
        raise ValueError("Discarded tile is absent from the hand")
    baseline = _cached_shanten(tuple(remaining))
    improving = []
    ukeire = 0
    for kind, seen in enumerate(visible):
        available = 4 - seen
        if available <= 0:
            continue
        remaining[kind] += 1
        improved = _cached_shanten(tuple(remaining)) < baseline
        remaining[kind] -= 1
        if improved:
            improving.append(kind)
            ukeire += available
    return DiscardScore(baseline, ukeire, tuple(improving))


def evaluate_discard(view: PlayerView, action: ActionSpec) -> DiscardScore:
    """Explain a legal discard using only the immutable public player view."""
    if action not in view.legal_actions or action.kind != "DISCARD":
        raise ValueError("Expected a legal DISCARD action")
    if action.tile not in view.hand:
        raise ValueError("Discarded physical tile is absent from the hand")
    return _discard_score(
        _counts(view.hand), visible_tile_counts(view), tile_type(action.tile),
    )


def _legal(view: PlayerView) -> tuple[ActionSpec, ...]:
    if not view.legal_actions:
        raise ValueError("The player has no legal action to choose")
    return tuple(sorted(view.legal_actions, key=lambda action: action.token))


def _first_kind(actions: tuple[ActionSpec, ...], kinds: set[str] | frozenset[str]) -> ActionSpec | None:
    return next((action for action in actions if action.kind in kinds), None)


def genbutsu_types(view: PlayerView, opponent: int) -> frozenset[int]:
    """Tiles known unable to ron for this riichi opponent in the current hand.

    Use their own river plus all public discards after their riichi acceptance.
    Riichi furiten makes passed winning discards safe thereafter. This only
    addresses that opponent's ron, not other opponents or losing future value.
    """
    if opponent == view.seat or not view.riichi_declared[opponent]:
        return frozenset()
    safe = {tile_type(tile) for tile in view.discards[opponent]}
    accepted = False
    for event in view.history:
        if event.kind == "reach_accepted" and event.actor == opponent:
            accepted = True
        elif accepted and event.kind == "dahai" and event.tile_type is not None:
            safe.add(event.tile_type)
    return frozenset(safe)


def _manifest(name: str, seed: int, allow_riichi: bool = True) -> dict[str, Any]:
    parameters: dict[str, Any]
    if name == "random":
        parameters = {
            "sampling": "uniform_over_canonical_legal_actions",
            "takes_wins_automatically": False,
            "rng": "python.random.Random",
        }
    else:
        parameters = {
            "allow_riichi": allow_riichi,
            "take_legal_wins": True,
            "optional_calls": "pass",
            "kan_model": "none",
            "offense": "min_shanten_then_max_visible_ukeire",
            "tie_break": "retain_red_five_then_stable_action_token",
            "shanten_engine": "riichienv-0.4.10.calculate_shanten",
            "shanten_cache_entries": 131_072,
            "defense": (
                "min_non_genbutsu_riichi_opponents_before_offense; no_new_riichi_under_threat"
                if name == "defense" else "none"
            ),
        }
    return {"policy": name, "version": POLICY_VERSION, "seed": seed, "parameters": parameters}


class RandomPolicy:
    """Uniform legal-action baseline, intentionally allowed to pass a win."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self._rng = random.Random(seed)

    def choose(self, view: PlayerView) -> ActionSpec:
        actions = _legal(view)
        return actions[self._rng.randrange(len(actions))]

    def manifest(self) -> dict[str, Any]:
        return _manifest("random", self.seed)


class ShantenUkeirePolicy:
    """Closed-hand baseline with exact shanten and visible-availability ukeire."""

    def __init__(self, seed: int = 0, *, allow_riichi: bool = True):
        self.seed = seed
        self.allow_riichi = allow_riichi

    def _threats(self, view: PlayerView) -> tuple[frozenset[int], ...]:
        return ()

    def choose(self, view: PlayerView) -> ActionSpec:
        actions = _legal(view)
        if len(actions) == 1:
            return actions[0]
        win = _first_kind(actions, _WIN_KINDS)
        if win is not None:
            return win
        threats = self._threats(view)
        # RiichiEnv declares reach first; its following discard is a separate
        # decision with the appropriate legal candidates supplied by the engine.
        if self.allow_riichi and not threats:
            reach = _first_kind(actions, {"RIICHI"})
            if reach is not None:
                return reach
        discards = tuple(action for action in actions if action.kind == "DISCARD")
        if discards:
            counts = _counts(view.hand)
            visible = visible_tile_counts(view)
            # Physical copies/red fives have identical shape/ukeire scores.
            scores: dict[int, DiscardScore] = {}

            def key(action: ActionSpec) -> tuple[int, int, int, int, str]:
                if action.tile not in view.hand:
                    raise ValueError("Legal discard is absent from player hand")
                kind = tile_type(action.tile)
                if kind not in scores:
                    scores[kind] = _discard_score(counts, visible, kind)
                score = scores[kind]
                unsafe_count = sum(kind not in safe for safe in threats)
                return (unsafe_count, score.shanten, -score.ukeire,
                        int(action.tile in RED_FIVES), action.token)

            return min(discards, key=key)
        passed = _first_kind(actions, {"PASS"})
        if passed is not None:
            return passed
        # A forced/unsupported legal action must still round-trip. This fallback
        # is deliberately identifiable; no strategic call evaluation is claimed.
        return actions[0]

    def manifest(self) -> dict[str, Any]:
        return _manifest("shanten", self.seed, self.allow_riichi)


class ConservativePolicy(ShantenUkeirePolicy):
    """Fold-first reference against riichi, with no danger-model inference."""

    def _threats(self, view: PlayerView) -> tuple[frozenset[int], ...]:
        return tuple(
            genbutsu_types(view, opponent)
            for opponent in range(4)
            if opponent != view.seat and view.riichi_declared[opponent]
        )

    def manifest(self) -> dict[str, Any]:
        return _manifest("defense", self.seed, self.allow_riichi)


class SemanticHeuristicPolicy:
    """Additive v2 teacher: same tactical scores, semantic tie resolution."""

    def __init__(self, seed: int = 0, *, defense: bool = False, allow_riichi: bool = True):
        self.seed = seed
        self.defense = defense
        self.allow_riichi = allow_riichi

    def choose(self, view: PlayerView) -> ActionSpec:
        from .semantics import teacher_decision
        decision = teacher_decision(view, defense=self.defense, allow_riichi=self.allow_riichi)
        return decision.actions[decision.chosen_index].representative

    def manifest(self) -> dict[str, Any]:
        from .semantics import SEMANTIC_ACTION_VERSION, TEACHER_VERSION
        manifest = _manifest("defense" if self.defense else "shanten", self.seed, self.allow_riichi)
        manifest["policy"] += "_v2"
        manifest["version"] = TEACHER_VERSION
        manifest["parameters"].update({
            "semantic_action_version": SEMANTIC_ACTION_VERSION,
            "tie_break": "numeric_tile_type_then_within_type_drawn_then_numeric_physical_representative",
            "soft_targets": "uniform_over_best_semantic_classes; distinct_from_deterministic_behavior",
        })
        return manifest


def make_policy(name: str, seed: int = 0) -> Policy:
    try:
        canonical = _ALIASES[name]
    except KeyError as error:
        raise ValueError(f"Unknown policy {name!r}; choose random, shanten, defense, shanten_v2, or defense_v2") from error
    if canonical in ("shanten_v2", "defense_v2"):
        return SemanticHeuristicPolicy(seed, defense=canonical == "defense_v2")
    return {"random": RandomPolicy, "shanten": ShantenUkeirePolicy,
            "defense": ConservativePolicy}[canonical](seed)


def policy_manifest(name: str, seed: int = 0) -> dict[str, Any]:
    """Return fresh JSON-compatible metadata for the default named policy."""
    return make_policy(name, seed).manifest()
