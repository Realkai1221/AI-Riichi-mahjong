"""Small deterministic smoke-test adapter for RiichiEnv 0.4.10.

This is an audited integration surface, not an independent rules implementation
or evidence of full rules conformance. Engine state and full MJAI logs are for
the auditor only. The smoke policy reads legal actions; learned policies will
receive audited features rather than arbitrary raw Observation fields.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
import random
from typing import Any, Mapping

from riichienv import GameRule, RiichiEnv


ENGINE_VERSION = "0.4.10"
GAME_MODE = "4p-red-half"
RULE_PROFILE = "riichienv-0.4.10/default_tenhou/4p-red-half"
INITIAL_SCORES = (25_000, 25_000, 25_000, 25_000)
DIGEST_SCHEMA = "riichi-ai-smoke-v1"


@dataclass(frozen=True)
class SimulationSummary:
    seed: int
    policy_seed: int
    engine_version: str
    rule_profile: str
    completed: bool
    stop_reason: str
    steps: int
    decisions: int
    events: int
    rounds: int
    scores: tuple[int, ...]
    ranks: tuple[int, ...]
    riichi_sticks: int
    conservation_ok: bool
    digest: str
    digest_schema: str = DIGEST_SCHEMA


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _validate_seed(seed: int) -> None:
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an integer in [0, 2**64)")


def create_env(seed: int) -> tuple[RiichiEnv, dict[int, Any]]:
    """Start a seeded four-player red-five hanchan with explicit settings.

    Recreate the instance to restart a seed: this engine's ``reset(seed=...)``
    does not rewind its per-hand shuffle counter. The constructor and one reset
    below deliberately define a consistent initialization sequence.
    """
    _validate_seed(seed)
    installed = version("riichienv")
    if installed != ENGINE_VERSION:
        raise RuntimeError(f"Expected riichienv=={ENGINE_VERSION}, found {installed}")
    env = RiichiEnv(game_mode=GAME_MODE, rule=GameRule.default_tenhou(), seed=seed)
    observations = env.reset(
        seed=seed, oya=0, round_wind=0, honba=0, kyotaku=0,
        scores=list(INITIAL_SCORES),
    )
    return env, observations


def canonical_action(action: Any) -> dict[str, Any]:
    """Return a stable physical-tile action representation for this engine.

    Physical tile IDs retain red-five and consumed-tile distinctions; this is
    not a portable action vocabulary for cross-engine comparisons.
    """
    return {
        "type": str(action.action_type),
        "tile": action.tile,
        "consume_tiles": sorted(action.consume_tiles),
    }


def canonical_observation(observation: Any) -> dict[str, Any]:
    """Return an actor-visible current-state snapshot, including its own hand.

    This intentionally excludes event history and every other player's hand.
    It is a smoke-audit snapshot, not a complete training representation.
    """
    raw = json.loads(base64.b64decode(observation.serialize_to_base64()))
    fields = (
        "player_id", "melds", "discards", "dora_indicators", "scores",
        "riichi_declared", "honba", "riichi_sticks", "round_wind", "oya",
        "kyoku_index", "waits", "is_tenpai", "riichi_sutehais",
        "last_tedashis", "last_discard", "forced_tedashi",
        "tsumogiri_flags",
    )
    result = {field: raw[field] for field in fields if field in raw}
    result["hand"] = sorted(observation.hand)
    # In 0.4.10, inactive players' raw observations also carry the current
    # player's drawn tile. Physical tile IDs are unique, so membership in the
    # observer's own hand safely gates this private feature.
    drawn_tile = raw.get("drawn_tile")
    result["drawn_tile"] = drawn_tile if drawn_tile in result["hand"] else None
    result["legal_actions"] = sorted(
        (canonical_action(action) for action in observation.legal_actions()), key=_json
    )
    return result


def choose_legal_actions(
    observations: Mapping[int, Any], rng: random.Random
) -> dict[int, Any]:
    """Sample uniformly over each active seat's engine-provided legal actions.

    Sorting removes dependence on Rust HashMap and input dictionary iteration.
    This is a reproducible random smoke policy, not a competitive baseline.
    """
    chosen = {}
    for player in sorted(observations):
        legal = sorted(
            observations[player].legal_actions(),
            key=lambda action: _json(canonical_action(action)),
        )
        if legal:
            chosen[player] = legal[rng.randrange(len(legal))]
    return chosen


def score_total(env: RiichiEnv) -> int:
    """Scores plus the table's outstanding 1,000-point riichi deposits."""
    return sum(env.scores()) + 1_000 * env.riichi_sticks


def run_random_hanchan(seed: int, max_events: int = 100_000) -> SimulationSummary:
    """Run one deterministic smoke match and return a reproducibility record.

    ``max_events`` limits atomic ``env.step`` calls (decision batches), not MJAI
    messages: one engine step can emit several protocol events. ``events`` in
    the summary counts MJAI messages. Reproducibility is scoped to this adapter,
    the pinned engine/dependencies, and the Python RNG implementation; a digest
    mismatch across different stacks is an audit signal, not necessarily a bug.
    """
    _validate_seed(seed)
    if type(max_events) is not int or max_events <= 0:
        raise ValueError("max_events must be a positive integer")
    env, observations = create_env(seed)
    policy_seed = seed ^ 0x9E3779B97F4A7C15
    rng = random.Random(policy_seed)
    audit = hashlib.sha256()
    audit.update((_json({
        "schema": DIGEST_SCHEMA, "engine": ENGINE_VERSION, "rule": RULE_PROFILE,
        "seed": seed, "policy_seed": policy_seed,
    }) + "\n").encode())
    initial_total = sum(INITIAL_SCORES)
    steps = decisions = 0
    conservation_ok = score_total(env) == initial_total
    while not env.done() and steps < max_events:
        chosen = choose_legal_actions(observations, rng)
        if not chosen:
            raise RuntimeError("Unfinished environment supplied no legal actions")
        audit.update((_json({
            str(player): canonical_action(action) for player, action in chosen.items()
        }) + "\n").encode())
        observations = env.step(chosen)
        steps += 1
        decisions += len(chosen)
        conservation_ok = conservation_ok and score_total(env) == initial_total

    # The complete engine log is auditor-only. Serialize it once, not on every
    # step; repeatedly exporting the growing history would be quadratic work.
    log = env.mjai_log
    scores = tuple(env.scores())
    completed = env.done()
    audit.update((_json({
        "log": log, "scores": scores, "riichi_sticks": env.riichi_sticks,
        "completed": completed,
    }) + "\n").encode())
    return SimulationSummary(
        seed=seed, policy_seed=policy_seed, engine_version=ENGINE_VERSION,
        rule_profile=RULE_PROFILE, completed=completed,
        stop_reason="completed" if completed else "max_events",
        steps=steps, decisions=decisions, events=len(log),
        rounds=sum(event.get("type") == "start_kyoku" for event in log),
        scores=scores, ranks=tuple(env.ranks()), riichi_sticks=env.riichi_sticks,
        conservation_ok=conservation_ok, digest=audit.hexdigest(),
    )
