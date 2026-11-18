"""Known hand/safety fixtures and legal-view-only policy regressions."""

from dataclasses import replace
import json

import pytest

from riichi_ai.player import ActionSpec, EventView, MeldView, PlayerView
from riichi_ai.policies import (
    ConservativePolicy,
    RandomPolicy,
    ShantenUkeirePolicy,
    evaluate_discard,
    genbutsu_types,
    hand_shanten,
    make_policy,
    policy_manifest,
)


def action(kind: str, tile: int | None = None) -> ActionSpec:
    return ActionSpec(kind=kind, tile=tile, consume_tiles=(),
                      token=json.dumps([kind, tile], separators=(",", ":")))


def physical(types: tuple[int, ...]) -> tuple[int, ...]:
    used: dict[int, int] = {}
    hand = []
    for kind in types:
        hand.append(4 * kind + used.get(kind, 0))
        used[kind] = used.get(kind, 0) + 1
    return tuple(hand)


# 123m 123p 123s EEE S plus a spare red dragon: discarding C waits on S.
WAITING_HAND = (0, 4, 8, 36, 40, 44, 72, 76, 80, 108, 109, 110, 112, 132)


def view(hand=WAITING_HAND, actions=None, **changes) -> PlayerView:
    result = PlayerView(
        seat=0, hand=tuple(hand),
        legal_actions=tuple(actions if actions is not None else
                            (action("DISCARD", tile) for tile in hand)),
        discards=((), (), (), ()), melds=((), (), (), ()),
        scores=(25_000,) * 4, riichi_declared=(False,) * 4,
        drawn_tile=hand[-1] if hand else None, dora_indicators=(),
        round_wind=0, oya=0, honba=0, riichi_sticks=0,
        hand_id=1, history=(),
    )
    return replace(result, **changes)


def event(kind: str, actor: int, tile_kind: int | None = None) -> EventView:
    return EventView(kind=kind, actor=actor, tile=None, tile_type=tile_kind,
                     consumed=(), target=None, tsumogiri=None, data=())


@pytest.mark.parametrize("types,expected", [
    ((0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 27, 28), 0),
    ((0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 27, 28, 28), -1),
    ((0, 0, 8, 8, 9, 9, 17, 17, 18, 18, 26, 26, 27), 0),
    ((0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33), 0),
    ((0, 0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33), -1),
    ((0, 1, 2, 9, 10, 11, 18, 19, 20, 27), 0),
])
def test_known_standard_special_and_short_concealed_shanten(types, expected):
    assert hand_shanten(physical(types)) == expected


def test_visible_ukeire_counts_unseen_copies_and_keeps_discard_visible():
    discard = action("DISCARD", 132)
    state = view(actions=(discard,))
    score = evaluate_discard(state, discard)
    assert (score.shanten, score.ukeire, score.improving_types) == (0, 3, (28,))
    # Own S plus indicator S plus one S in an opponent's river leaves one copy.
    state = replace(state, dora_indicators=(113,), discards=((), (114,), (), ()))
    assert evaluate_discard(state, discard).ukeire == 1


def test_called_discard_is_not_counted_twice_in_ukeire():
    discard = action("DISCARD", 132)
    meld = MeldView(kind="CHI", tiles=(0, 5, 9), opened=True,
                    from_player=0, called_tile=0)
    # 23m plus three complete sets and a pair waits on 1m/4m. One 1m is
    # called into the public chi but remains in the source river: 3 + 4 copies.
    hand = (4, 8, 36, 40, 44, 72, 76, 80, 108, 109, 110, 112, 113, 132)
    state = view(hand=hand, actions=(discard,), melds=((), (meld,), (), ()),
                 discards=((0,), (), (), ()))
    score = evaluate_discard(state, discard)
    assert (score.shanten, score.ukeire, score.improving_types) == (0, 7, (0, 3))


def test_shanten_precedes_ukeire_and_policy_is_deterministic():
    discard_good, discard_bad = action("DISCARD", 132), action("DISCARD", 0)
    state = view(actions=(discard_bad, discard_good))
    policy = ShantenUkeirePolicy(seed=10)
    assert evaluate_discard(state, discard_good).shanten == 0
    assert evaluate_discard(state, discard_bad).shanten == 1
    assert policy.choose(state) == discard_good
    assert ShantenUkeirePolicy(seed=999).choose(state) == discard_good
    assert policy.choose(replace(state, legal_actions=(discard_good, discard_bad))) == discard_good


def test_red_five_retained_only_after_shape_and_ukeire_tie():
    hand = (16, 17, 0, 4, 8, 36, 40, 44, 72, 76, 80, 108, 112, 116)
    red, normal = action("DISCARD", 16), action("DISCARD", 17)
    state = view(hand=hand, actions=(red, normal))
    assert evaluate_discard(state, red) == evaluate_discard(state, normal)
    assert ShantenUkeirePolicy().choose(state) == normal


@pytest.mark.parametrize("policy_type", [ShantenUkeirePolicy, ConservativePolicy])
def test_wins_precede_riichi_defense_and_calls(policy_type):
    ron, passed, reach = action("RON"), action("PASS"), action("RIICHI")
    state = view(actions=(passed, reach, ron), riichi_declared=(False, True, False, False))
    assert policy_type().choose(state) == ron
    assert policy_type().choose(view(actions=(action("CHI", 0), passed))) == passed


def test_riichi_is_a_separate_optional_action():
    reach, discarded = action("RIICHI"), action("DISCARD", 132)
    state = view(actions=(discarded, reach))
    assert ShantenUkeirePolicy().choose(state) == reach
    assert ShantenUkeirePolicy(allow_riichi=False).choose(state) == discarded
    threatened = replace(state, riichi_declared=(False, True, False, False))
    assert ConservativePolicy().choose(threatened) == discarded


def test_genbutsu_uses_own_river_and_only_post_acceptance_public_discards():
    state = view(
        riichi_declared=(False, True, False, False),
        discards=((), (3,), (), ()),
        history=(event("dahai", 2, 9), event("reach", 1),
                 event("dahai", 1, 0), event("reach_accepted", 1),
                 event("dahai", 3, 18)),
    )
    assert genbutsu_types(state, 1) == frozenset((0, 18))
    assert genbutsu_types(state, 2) == frozenset()
    assert genbutsu_types(state, 0) == frozenset()


def test_defense_prefers_riichi_genbutsu_even_when_shape_worsens():
    offensive, safe = action("DISCARD", 132), action("DISCARD", 0)
    state = view(actions=(offensive, safe), riichi_declared=(False, True, False, False),
                 discards=((), (3,), (), ()))
    assert ShantenUkeirePolicy().choose(state) == offensive
    assert ConservativePolicy().choose(state) == safe


def test_defense_counts_each_riichi_opponent_without_claiming_global_safety():
    first, both = action("DISCARD", 0), action("DISCARD", 132)
    state = view(actions=(first, both), riichi_declared=(False, True, True, False),
                 discards=((), (3, 133), (134,), ()))
    assert genbutsu_types(state, 1) == frozenset((0, 33))
    assert genbutsu_types(state, 2) == frozenset((33,))
    assert ConservativePolicy().choose(state) == both


def test_seeded_random_is_order_invariant_reproducible_and_can_pass_win():
    ron, passed = action("RON"), action("PASS")
    state = view(actions=(ron, passed))
    first, second = RandomPolicy(7), RandomPolicy(7)
    sequence = [first.choose(state) for _ in range(30)]
    assert sequence == [second.choose(replace(state, legal_actions=(passed, ron))) for _ in range(30)]
    assert set(sequence) == {ron, passed}


@pytest.mark.parametrize("name", ["random", "shanten", "shanten_ukeire", "defense", "conservative"])
def test_factory_legal_choice_and_manifest(name):
    state = view()
    policy = make_policy(name, 123)
    assert policy.choose(state) in state.legal_actions
    manifest = policy.manifest()
    assert manifest == policy_manifest(name, 123)
    assert manifest["version"] == "heuristics-v1"
    json.dumps(manifest, allow_nan=False)


def test_invalid_requests_fail_instead_of_inventing_actions():
    with pytest.raises(ValueError, match="Unknown policy"):
        make_policy("unregistered")
    with pytest.raises(ValueError, match="no legal action"):
        ShantenUkeirePolicy().choose(view(actions=()))
    with pytest.raises(ValueError, match="unique"):
        hand_shanten((0, 0))
    with pytest.raises(ValueError, match="legal DISCARD"):
        evaluate_discard(view(), action("PASS"))
