"""Actor-only exact features: a limited fixture family, not rules certification."""

from dataclasses import replace

import numpy as np
import pytest

from riichi_ai.efficiency import EFFICIENCY_SCALES, efficiency_from_encoded, exact_efficiency_from_encoded
from riichi_ai.features import FEATURE_SCHEMA_V2, encode_view
from riichi_ai.player import PlayerAdapter, action_spec
from riichi_ai.policies import evaluate_discard
from riichi_ai.rules import audit_physical_tiles, wall_for_hands
from riichi_ai.semantics import semantic_actions
from riichi_ai.simulator import create_env


def fixture(wall):
    env, _ = create_env(0)
    observations = env.reset(wall=wall, seed=0, oya=0)
    return env, observations


def arrays_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        np.testing.assert_array_equal(left[key], right[key], err_msg=key)


def test_known_wait_ukeire_and_exhausted_wait_definition():
    wall = wall_for_hands({0: "123m123p123s1112z"}, draws=["7z"])
    _, obs = fixture(wall)
    view = PlayerAdapter(0).build(obs[0])
    # Remove the random indicator for this arithmetic fixture; it changes only
    # public visibility, not the hand. All remaining tile IDs stay unique.
    view = replace(view, dora_indicators=())
    actions = semantic_actions(view)
    encoded = encode_view(view, actions, feature_schema=FEATURE_SCHEMA_V2)
    exact = exact_efficiency_from_encoded(encoded["tiles"], encoded["action_tokens"])
    index = next(i for i, action in enumerate(actions) if action.kind == "DISCARD" and action.tile_type == 33)
    assert tuple(exact[index]) == (0, 3, 1, 1)
    waiting = next(tile for tile in view.hand if tile // 4 == 28)
    all_waiting_copies = tuple(tile for tile in range(112, 116) if tile != waiting)
    exhausted = replace(view, discards=((), all_waiting_copies, (), ()))
    changed = encode_view(exhausted, actions, feature_schema=FEATURE_SCHEMA_V2)
    assert tuple(exact_efficiency_from_encoded(changed["tiles"], changed["action_tokens"])[index]) == (0, 0, 0, 1)
    np.testing.assert_array_equal(encoded["candidate_efficiency"],
                                  exact.astype(np.float32) / np.asarray(EFFICIENCY_SCALES, np.float32))


def test_red_drawn_and_normal_copies_keep_distinct_candidates_equal_shape_scores():
    _, obs = fixture(wall_for_hands({0: "055m123p123s789s1z"}, draws=["5m"]))
    view = PlayerAdapter(0).build(obs[0])
    actions = semantic_actions(view)
    encoded = encode_view(view, actions, feature_schema=FEATURE_SCHEMA_V2)
    fives = [i for i, action in enumerate(actions) if action.kind == "DISCARD" and action.tile_type == 4]
    assert len(fives) == 3
    assert len({tuple(encoded["action_tokens"][i]) for i in fives}) == 3
    for index in fives[1:]:
        np.testing.assert_array_equal(encoded["candidate_efficiency"][fives[0]], encoded["candidate_efficiency"][index])
    for index, action in enumerate(actions):
        if action.kind == "DISCARD":
            score = evaluate_discard(view, action.representative)
            raw = exact_efficiency_from_encoded(encoded["tiles"], encoded["action_tokens"])[index]
            assert tuple(raw) == (score.shanten, score.ukeire, len(score.improving_types), 1)


def test_hidden_worlds_and_copy_relabeling_after_common_public_prefix():
    wall = wall_for_hands({0: "234m567p678s1122z"}, draws=["8m", "3p", "4s", "7m", "6z"])
    initial, _ = fixture(wall)
    locations = {tile: index for index, tile in enumerate(wall)}
    changed = list(wall)
    a, b = locations[initial.hands[1][0]], locations[initial.hands[2][0]]
    changed[a], changed[b] = changed[b], changed[a]
    # Also permute two still-unseen wall positions beyond this short prefix.
    changed[100], changed[101] = changed[101], changed[100]
    encodings = []
    for complete in (wall, changed):
        env, obs = fixture(complete)
        adapters = [PlayerAdapter(seat) for seat in range(4)]
        for step in range(30):
            views = {seat: adapters[seat].build(obs[seat]) for seat in sorted(obs)}
            own = views.get(0)
            if step > 0 and own is not None and any(action.kind == "DISCARD" for action in own.legal_actions):
                audit_physical_tiles(env)
                encodings.append(encode_view(own, feature_schema=FEATURE_SCHEMA_V2))
                break
            selected = {}
            for seat, view in views.items():
                discards = [action for action in view.legal_actions if action.kind == "DISCARD"]
                if discards:
                    action = next((action for action in discards if action.tile == view.drawn_tile), discards[0])
                elif view.legal_actions:
                    action = next(action for action in view.legal_actions if action.kind == "PASS")
                else:
                    continue
                selected[seat] = adapters[seat].resolve(action, obs[seat])
            obs = env.step(selected)
        else:
            pytest.fail("Common prefix did not return to actor 0")
    arrays_equal(*encodings)
    # A complete physical-copy permutation preserves all actor semantic inputs.
    permutation = {}
    for kind in range(34):
        ids = list(range(kind * 4, kind * 4 + 4))
        movable = ids[1:] if kind in (4, 13, 22) else ids
        permutation.update(zip(movable, movable[1:] + movable[:1]))
    first_views = []
    for complete in (wall, [permutation.get(tile, tile) for tile in wall]):
        env, obs = fixture(complete)
        audit_physical_tiles(env)
        first_views.append(encode_view(PlayerAdapter(0).build(obs[0]), feature_schema=FEATURE_SCHEMA_V2))
    arrays_equal(*first_views)


def test_riichi_is_not_a_discard_and_following_discard_is_scored():
    env, obs = fixture(wall_for_hands({0: "123456m123p123s1z"}, draws=["6z"]))
    adapter = PlayerAdapter(0)
    view = adapter.build(obs[0])
    actions = semantic_actions(view)
    index = next(i for i, action in enumerate(actions) if action.kind == "RIICHI")
    assert actions[index].tile_type is None
    encoded = encode_view(view, actions, feature_schema=FEATURE_SCHEMA_V2)
    assert not encoded["candidate_efficiency"][index].any()
    obs = env.step({0: adapter.resolve(actions[index].representative, obs[0])})
    after = adapter.build(obs[0])
    next_actions = semantic_actions(after)
    assert next_actions and all(action.kind == "DISCARD" for action in next_actions)
    assert np.all(encode_view(after, feature_schema=FEATURE_SCHEMA_V2)["candidate_efficiency"][:, 3] == 1)


def test_post_call_short_concealed_hand_scores_only_actual_legal_discards():
    env, obs = fixture(wall_for_hands({1: "123456m789p456s2z"}, draws=["3m"]))
    adapters = [PlayerAdapter(seat) for seat in range(4)]
    for seat in obs:
        adapters[seat].build(obs[seat])
    discard = next(action for action in obs[0].legal_actions() if action_spec(action).kind == "DISCARD" and action.tile == env.drawn_tile)
    obs = env.step({0: discard})
    views = {seat: adapters[seat].build(obs[seat]) for seat in sorted(obs)}
    chosen = {}
    for seat, view in views.items():
        if not view.legal_actions:
            continue
        action = (next(action for action in view.legal_actions if action.kind == "CHI"
                       and {tile // 4 for tile in action.consume_tiles} == {3, 4}) if seat == 1
                  else next(action for action in view.legal_actions if action.kind == "PASS"))
        chosen[seat] = adapters[seat].resolve(action, obs[seat])
    obs = env.step(chosen)
    view = adapters[1].build(obs[1])
    assert len(view.hand) == 11
    actions = semantic_actions(view)
    assert not {2, 5}.intersection(action.tile_type for action in actions)
    encoded = encode_view(view, actions, feature_schema=FEATURE_SCHEMA_V2)
    exact = exact_efficiency_from_encoded(encoded["tiles"], encoded["action_tokens"])
    for index, action in enumerate(actions):
        score = evaluate_discard(view, action.representative)
        assert tuple(exact[index]) == (score.shanten, score.ukeire, len(score.improving_types), 1)


def test_suit_permutation_preserves_exact_efficiency_and_candidate_order():
    _, obs = fixture(wall_for_hands({0: "234m567p678s1122z"}, draws=["8m"]))
    encoded = encode_view(PlayerAdapter(0).build(obs[0]))
    mapping = np.asarray([(kind + 9) % 27 if kind < 27 else kind for kind in range(34)])
    moved_tiles = np.empty_like(encoded["tiles"])
    moved_tiles[mapping] = encoded["tiles"]
    moved_actions = encoded["action_tokens"].copy()
    present = moved_actions[:, 1] != 0
    moved_actions[present, 1] = mapping[moved_actions[present, 1] - 1] + 1
    expected = exact_efficiency_from_encoded(encoded["tiles"], encoded["action_tokens"])
    np.testing.assert_array_equal(expected, exact_efficiency_from_encoded(moved_tiles, moved_actions))
    np.testing.assert_array_equal(expected[::-1], exact_efficiency_from_encoded(encoded["tiles"], encoded["action_tokens"][::-1]))


def test_not_applicable_all_other_kinds_and_invalid_count_decoding():
    _, obs = fixture(wall_for_hands({0: "123456m123p123s1z"}, draws=["6z"]))
    encoded = encode_view(PlayerAdapter(0).build(obs[0]))
    other = np.zeros((10, 8), dtype=np.int16)
    other[:, 0] = np.arange(2, 12)
    assert not exact_efficiency_from_encoded(encoded["tiles"], other).any()
    invalid = encoded["tiles"].copy()
    invalid[0, 0] = .125
    with pytest.raises(ValueError, match="decode exactly"):
        efficiency_from_encoded(invalid, encoded["action_tokens"])
    invalid = encoded["tiles"].copy()
    invalid[0, 4] = -1
    with pytest.raises(ValueError, match="decode exactly"):
        efficiency_from_encoded(invalid, encoded["action_tokens"])
