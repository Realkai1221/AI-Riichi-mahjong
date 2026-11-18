"""Semantic-boundary fixtures; not a claim of complete rules conformance."""

from dataclasses import FrozenInstanceError, replace
import json

import pytest

from riichi_ai.player import PlayerAdapter, action_spec
from riichi_ai.policies import ConservativePolicy, ShantenUkeirePolicy, make_policy
from riichi_ai.rules import audit_physical_tiles, wall_for_hands
from riichi_ai.semantics import ACTION_KINDS, semantic_actions, teacher_decision
from riichi_ai.simulator import create_env


def fixture(hands, draws, *, wall=None, dead_wall=None):
    env, _ = create_env(0)
    obs = env.reset(wall=wall if wall is not None else wall_for_hands(hands, draws, dead_wall=dead_wall),
                    seed=0, oya=0, round_wind=0, honba=0, kyotaku=0, scores=[25_000] * 4)
    audit_physical_tiles(env)
    return env, obs


def pick(observation, kind, tile=None):
    return next(action for action in observation.legal_actions()
                if action_spec(action).kind == kind and (tile is None or action.tile == tile))


def semantic_fields(actions):
    return [(a.kind, a.tile_type, a.red, a.tsumogiri, a.consumed, a.key) for a in actions]


def test_duplicate_physical_copies_do_not_receive_duplicate_target_mass():
    env, obs = fixture({0: "111m234678p234s5z"}, ["1m"])
    adapter = PlayerAdapter(0)
    view = adapter.build(obs[0])
    decision = teacher_decision(view, allow_riichi=False)
    best = [a for a, selected in zip(decision.actions, decision.best_mask) if selected]
    assert [(a.tile_type, a.tsumogiri, len(a.physical_tokens)) for a in best] == [(0, True, 1), (0, False, 3)]
    assert [p for p in decision.soft_targets if p] == [.5, .5]
    assert sum(decision.soft_targets) == pytest.approx(1)
    selected = decision.actions[decision.chosen_index]
    assert selected.representative.tile == env.drawn_tile
    v1 = ShantenUkeirePolicy(allow_riichi=False).choose(view)
    assert v1 != selected.representative
    assert any(v1.token in action.physical_tokens for action in best)
    for semantic in decision.actions:
        assert action_spec(adapter.resolve(semantic.representative, obs[0])) == semantic.representative
    with pytest.raises(FrozenInstanceError):
        selected.red = True


def test_red_tiles_and_drawn_versus_held_copies_remain_distinct():
    _, obs = fixture({0: "055m123p123s789s1z"}, ["5m"])
    view = PlayerAdapter(0).build(obs[0])
    fives = [a for a in semantic_actions(view) if a.kind == "DISCARD" and a.tile_type == 4]
    assert {(a.red, a.tsumogiri, len(a.physical_tokens)) for a in fives} == {
        (False, True, 1), (False, False, 2), (True, False, 1),
    }
    decision = teacher_decision(view, allow_riichi=False)
    best = [a for a, flag in zip(decision.actions, decision.best_mask) if flag]
    assert all(not a.red for a in best)
    assert decision.actions[decision.chosen_index].tsumogiri is True


def test_full_wall_copy_permutations_preserve_semantic_teacher_outputs():
    wall = wall_for_hands({0: "055m123p123s789s1z"}, ["5m"])
    permutation = {}
    for kind in range(34):
        copies = list(range(1, 4)) if kind in (4, 13, 22) else list(range(4))
        permutation[kind * 4] = kind * 4
        for current, following in zip(copies, copies[1:] + copies[:1]):
            permutation[kind * 4 + current] = kind * 4 + following
    decisions = []
    for complete in (wall, [permutation[t] for t in wall]):
        _, obs = fixture({}, [], wall=complete)
        view = PlayerAdapter(0).build(obs[0])
        decisions.append(teacher_decision(view, allow_riichi=False))
        reverse = teacher_decision(replace(view, legal_actions=tuple(reversed(view.legal_actions))), allow_riichi=False)
        assert reverse == decisions[-1]
    a, b = decisions
    assert semantic_fields(a.actions) == semantic_fields(b.actions)
    assert (a.preference_keys, a.best_mask, a.chosen_index, a.soft_targets) == (
        b.preference_keys, b.best_mask, b.chosen_index, b.soft_targets)
    assert a.actions[a.chosen_index].representative.tile != b.actions[b.chosen_index].representative.tile


def test_consumed_red_identity_and_chi_sequence_are_preserved():
    env, obs = fixture({1: "055m123p456s789s1z"}, ["5m"])
    obs = env.step({0: pick(obs[0], "DISCARD", env.drawn_tile)})
    view = PlayerAdapter(1).build(obs[1])
    pon = [a for a in semantic_actions(view) if a.kind == "PON"]
    assert {a.consumed for a in pon} == {((4, False), (4, False)), ((4, False), (4, True))}
    assert sorted(len(a.physical_tokens) for a in pon) == [1, 2]
    # Distinct chi sequences must never collapse into a generic 'call 3m'.
    env, obs = fixture({1: "1240m123p456s123z"}, ["3m"])
    obs = env.step({0: pick(obs[0], "DISCARD", env.drawn_tile)})
    choices = [a for a in semantic_actions(PlayerAdapter(1).build(obs[1])) if a.kind == "CHI"]
    assert len(choices) == 3
    assert len({a.consumed for a in choices}) == 3


def test_every_supported_action_family_groups_and_resolves_real_candidates():
    found = set()

    def check(adapter, observation):
        view = adapter.build(observation)
        actions = semantic_actions(view)
        assert {token for a in actions for token in a.physical_tokens} == {a.token for a in view.legal_actions}
        assert len(actions) == len({a.key for a in actions})
        for action in actions:
            assert action_spec(adapter.resolve(action.representative, observation)) == action.representative
            if action.kind != "DISCARD":
                assert action.tsumogiri is None
            found.add(action.kind)
        return view

    examples = [
        ({1: "12m456m789p456s23z", 2: "33m234p678234s23z", 3: "12m123p123789s11z"}, ["3m"], True),
        ({1: "555m123p456s789s1z"}, ["0m"], True),
        ({0: "1111m234678p234s"}, ["5z"], False),
        ({0: "123456m123p123s1z"}, ["1z"], False),
        ({0: "123456m123p123s1z"}, ["2z"], False),
        ({0: "19m19p19s1234567z"}, ["2m"], False),
    ]
    for hands, draws, discard_first in examples:
        env, obs = fixture(hands, draws)
        if discard_first:
            obs = env.step({0: pick(obs[0], "DISCARD", env.drawn_tile)})
        for seat, observation in obs.items():
            check(PlayerAdapter(seat), observation)
        audit_physical_tiles(env)

    # A real pon -> discard -> three turns -> added-kan opportunity.
    env, obs = fixture({1: "11m234678p234s56z"}, ["1m", "2z", "3z", "4z", "1m"], dead_wall={131: "7z"})
    adapters = [PlayerAdapter(seat) for seat in range(4)]
    for turn in range(20):
        views = {seat: check(adapters[seat], observation) for seat, observation in obs.items()}
        if "KAKAN" in found:
            break
        choices = {}
        for seat, view in views.items():
            kinds = {a.kind for a in view.legal_actions}
            if "PON" in kinds and seat == 1:
                choices[seat] = pick(obs[seat], "PON")
            elif "PASS" in kinds:
                choices[seat] = pick(obs[seat], "PASS")
            elif seat == 1 and not any(event.kind == "dahai" and event.actor == 1 for event in view.history):
                green = next(t for t in view.hand if t // 4 == 32)
                choices[seat] = pick(obs[seat], "DISCARD", green)
            else:
                choices[seat] = pick(obs[seat], "DISCARD", env.drawn_tile)
        obs = env.step(choices)
        audit_physical_tiles(env)
    assert found == set(ACTION_KINDS)


def test_branch_preferences_and_factory_provenance():
    env, obs = fixture({0: "123456m123p123s1z"}, ["1z"])
    view = PlayerAdapter(0).build(obs[0])
    decision = teacher_decision(view)
    assert {a.kind for a, flag in zip(decision.actions, decision.best_mask) if flag} == {"TSUMO"}
    env, obs = fixture({0: "123456m123p123s1z"}, ["2z"])
    view = PlayerAdapter(0).build(obs[0])
    assert teacher_decision(view).actions[teacher_decision(view).chosen_index].kind == "RIICHI"
    threatened = replace(view, riichi_declared=(False, True, False, False))
    defense = teacher_decision(threatened, defense=True)
    assert defense.actions[defense.chosen_index].kind == "DISCARD"
    for name in ("shanten_v2", "defense_v2"):
        policy = make_policy(name, 77)
        expected = teacher_decision(view, defense=name == "defense_v2")
        assert policy.choose(view) == expected.actions[expected.chosen_index].representative
        assert policy.manifest()["version"] == "heuristics-v2"
        assert policy.manifest()["policy"] == name
        json.dumps(policy.manifest(), allow_nan=False)
    for name in ("random", "shanten", "shanten_ukeire", "defense", "conservative"):
        assert make_policy(name).manifest()["version"] == "heuristics-v1"


def test_v1_v2_changes_are_only_pre_tiebreak_ties_on_shared_real_views():
    env, obs = create_env(319)
    adapters = [PlayerAdapter(seat) for seat in range(4)]
    comparisons = 0
    for _ in range(120):
        if env.done():
            break
        views = {seat: adapters[seat].build(observation) for seat, observation in obs.items()}
        moves = {}
        for seat, view in views.items():
            for defense, v1 in ((False, ShantenUkeirePolicy()), (True, ConservativePolicy())):
                result = teacher_decision(view, defense=defense)
                old = v1.choose(view)
                index = next(i for i, a in enumerate(result.actions) if old.token in a.physical_tokens)
                assert result.preference_keys[index] == min(result.preference_keys)
                assert result.best_mask[index]
                comparisons += 1
            chosen = make_policy("shanten_v2").choose(view)
            moves[seat] = adapters[seat].resolve(chosen, obs[seat])
        obs = env.step(moves)
    assert comparisons >= 200


def test_empty_unknown_and_colliding_contracts_fail_explicitly():
    _, obs = create_env(0)
    view = PlayerAdapter(0).build(obs[0])
    assert semantic_actions(replace(view, legal_actions=())) == ()
    with pytest.raises(ValueError, match="no legal action"):
        teacher_decision(replace(view, legal_actions=()))
    action = view.legal_actions[0]
    with pytest.raises(ValueError, match="Unsupported"):
        semantic_actions(replace(view, legal_actions=(replace(action, kind="FUTURE_ACTION"),)))
    with pytest.raises(ValueError, match="different actions"):
        semantic_actions(replace(view, legal_actions=(action, replace(action, kind="PASS"))))
