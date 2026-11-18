"""Semantic identity, visible information, and explicit history truncation."""

from dataclasses import replace

import numpy as np
import pytest

from riichi_ai.features import (
    ACTION_FIELDS, FEATURE_SCHEMA, GLOBAL_FEATURES, HISTORY_FIELDS, MAX_HISTORY,
    TILE_CHANNELS, encode_history_events, encode_view, feature_manifest,
)
from riichi_ai.player import EventView, PlayerAdapter
from riichi_ai.rules import audit_physical_tiles, wall_for_hands
from riichi_ai.semantics import semantic_actions
from riichi_ai.simulator import create_env


def fixture_view(wall=None):
    env, observations = create_env(67)
    if wall is not None:
        observations = env.reset(wall=wall, seed=0, oya=0, round_wind=0,
                                 honba=0, kyotaku=0, scores=[25_000] * 4)
    return env, PlayerAdapter(0).build(observations[0])


def assert_arrays_equal(first, second):
    assert first.keys() == second.keys()
    for key in first:
        np.testing.assert_array_equal(first[key], second[key], err_msg=key)


def test_native_view_schema_and_only_actor_arrays():
    _, view = fixture_view()
    encoded = encode_view(view)
    assert set(encoded) == {"tiles", "globals", "history", "history_mask", "action_tokens"}
    assert encoded["tiles"].shape == (34, TILE_CHANNELS)
    assert encoded["globals"].shape == (GLOBAL_FEATURES,)
    assert encoded["history"].shape == (MAX_HISTORY, HISTORY_FIELDS)
    assert encoded["action_tokens"].shape == (len(semantic_actions(view)), ACTION_FIELDS)
    assert encoded["tiles"].dtype == encoded["globals"].dtype == np.float32
    assert encoded["history"].dtype == encoded["action_tokens"].dtype == np.int16
    assert encoded["history_mask"].dtype == np.bool_
    assert np.isfinite(encoded["tiles"]).all()
    assert encoded["tiles"][:, 0].sum() == len(view.hand) / 4
    assert feature_manifest()["schema"] == FEATURE_SCHEMA


def test_consistent_hidden_world_changes_leave_encoder_identical():
    wall = wall_for_hands({0: "234m567p678s1122z"}, draws=["8m"])
    env, first = fixture_view(wall)
    # Swap only two concealed opponents' physical tiles, preserving the full
    # legal wall, actor hand/draw, public marker, and initial public history.
    changed = list(wall)
    positions = {tile: index for index, tile in enumerate(wall)}
    left, right = positions[env.hands[1][0]], positions[env.hands[2][0]]
    changed[left], changed[right] = changed[right], changed[left]
    other_env, second = fixture_view(changed)
    audit_physical_tiles(env)
    audit_physical_tiles(other_env)
    assert env.hands[1] != other_env.hands[1]
    assert first == second
    assert_arrays_equal(encode_view(first), encode_view(second))


def test_arbitrary_normal_copy_relabeling_does_not_enter_inputs():
    _, view = fixture_view(wall_for_hands({0: "555m123p123s789s1z"}, draws=["0m"]))
    mapping = {17: 19, 19: 17}
    convert = lambda tile: mapping.get(tile, tile)
    actions = tuple(replace(action, tile=convert(action.tile),
                            consume_tiles=tuple(convert(tile) for tile in action.consume_tiles),
                            token=f"relabel-{index}")
                    for index, action in enumerate(view.legal_actions))
    changed = replace(view, hand=tuple(convert(tile) for tile in view.hand), legal_actions=actions,
                      drawn_tile=convert(view.drawn_tile),
                      discards=tuple(tuple(convert(tile) for tile in river) for river in view.discards),
                      dora_indicators=tuple(convert(tile) for tile in view.dora_indicators))
    assert_arrays_equal(encode_view(view), encode_view(changed))
    assert len(semantic_actions(view)) < len(view.legal_actions)


def test_red_and_tsumogiri_are_distinct_candidate_features():
    _, view = fixture_view(wall_for_hands({0: "055m123p123s789s1z"}, draws=["5m"]))
    actions = semantic_actions(view)
    indices = [i for i, action in enumerate(actions) if action.kind == "DISCARD" and action.tile_type == 4]
    features = encode_view(view, actions)["action_tokens"][indices]
    assert len(features) == 3  # Red-held, normal-held, normal-drawn.
    assert len({tuple(row) for row in features}) == 3


def test_event_prefix_rows_stable_and_history_overflow_explicit():
    _, view = fixture_view()
    history = tuple(EventView("dahai", actor=i % 4, tile="1m", tile_type=0,
                              tsumogiri=bool(i % 2)) for i in range(MAX_HISTORY + 9))
    changed = replace(view, history=history)
    rows = encode_history_events(history, view.seat)
    np.testing.assert_array_equal(rows[:20], encode_history_events(history[:20], view.seat))
    encoded = encode_view(changed)
    np.testing.assert_array_equal(encoded["history"], rows[-MAX_HISTORY:])
    assert encoded["history_mask"].all()
    assert encoded["globals"][41] == pytest.approx(len(history) / MAX_HISTORY)
    assert encoded["globals"][42] == 1
    assert encoded["globals"][43] == pytest.approx(9 / MAX_HISTORY)


def test_event_data_and_external_labels_are_not_model_inputs():
    _, view = fixture_view()
    changed = replace(view, history=tuple(replace(event, data=event.data + (
        ("teacher_scores", ((-999,),)), ("final_rank", 1), ("hidden_wall", (135,)),
    )) for event in view.history))
    assert_arrays_equal(encode_view(view), encode_view(changed))
    altered_scores = replace(view, scores=(view.scores[0] + 1000, *view.scores[1:]))
    assert not np.array_equal(encode_view(view)["globals"], encode_view(altered_scores)["globals"])


def test_candidate_order_is_explicit_and_illegal_subsets_fail():
    _, view = fixture_view()
    actions = semantic_actions(view)
    first, reversed_encoding = encode_view(view, actions), encode_view(view, actions[::-1])
    np.testing.assert_array_equal(first["action_tokens"][::-1], reversed_encoding["action_tokens"])
    for name in ("tiles", "globals", "history", "history_mask"):
        np.testing.assert_array_equal(first[name], reversed_encoding[name])
    with pytest.raises(ValueError, match="each current semantic legal action"):
        encode_view(view, actions[:-1])
    with pytest.raises(ValueError, match="Unsupported actor event"):
        encode_view(replace(view, history=(EventView("unreviewed_new_event"),)))


def test_empty_history_and_zero_legal_view_are_explicit():
    _, view = fixture_view()
    empty = encode_view(replace(view, history=(), legal_actions=()))
    assert empty["action_tokens"].shape == (0, ACTION_FIELDS)
    assert not empty["history_mask"].any()
    assert not empty["history"].any()
