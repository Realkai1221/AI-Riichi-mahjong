"""Real post-call actor coverage for the bounded Stage 3C engineering gate.

These locally authored explicit-wall fixtures extend existing Stage 1 cases.
They are neither complete rules certification nor examples for model training.
Full engine state is inspected only by the test auditor; feature/efficiency
calculations below receive immutable PlayerView values and encoded arrays.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path

from mahjong.shanten import Shanten
import numpy as np
import pytest

from riichi_ai.efficiency import EFFICIENCY_SCALES, exact_efficiency_from_encoded
from riichi_ai.features import FEATURE_SCHEMA_V2, encode_view
from riichi_ai.player import PlayerAdapter, action_spec, is_closed_hand, visible_tile_counts
from riichi_ai.policies import evaluate_discard
from riichi_ai.rules import audit_physical_tiles, wall_for_hands
from riichi_ai.semantics import semantic_actions
from riichi_ai.simulator import create_env, score_total


_CASES = json.loads((Path(__file__).parent / "fixtures/stage1/cases.json").read_text())["cases"]
CALL_FAMILIES = ("CHI", "PON", "DAIMINKAN", "ANKAN", "KAKAN")


class TransitionFixture:
    """Route every returned observation before resolving a response batch."""

    def __init__(self, hands, draws, dead_wall=None):
        self.wall = wall_for_hands(hands, draws, dead_wall=dead_wall or {})
        assert sorted(self.wall) == list(range(136))
        self.env, _ = create_env(0)
        self.adapters = [PlayerAdapter(seat) for seat in range(4)]
        observations = self.env.reset(wall=self.wall, seed=0, oya=0, round_wind=0,
                                      honba=0, kyotaku=0, scores=[25000] * 4)
        self.initial_total = score_total(self.env)
        self.steps = 0
        self.capture(observations)

    def capture(self, observations):
        self.observations = observations
        missing = [seat for seat in range(4) if seat not in observations]
        # Extra observer snapshots are audit-only. They preserve a continuous
        # event stream for each adapter and do not enter another seat's policy.
        every = observations | self.env.get_observations(players=missing)
        self.views = {seat: self.adapters[seat].build(every[seat]) for seat in range(4)}
        self.physical = audit_physical_tiles(self.env)
        assert self.physical["physical_tiles"] == 136
        assert score_total(self.env) == self.initial_total
        for seat, view in self.views.items():
            assert tuple(sorted(self.env.hands[seat])) == view.hand
            for event in view.history:
                if event.kind == "tsumo" and event.actor != seat:
                    assert event.tile is None and event.tile_type is None
            assert all(key not in {"wall", "seed", "tehais"}
                       for event in view.history for key, _ in event.data)
            if seat in observations:
                assert set(view.legal_actions) == {
                    action_spec(native) for native in observations[seat].legal_actions()
                }
                for action in view.legal_actions:
                    assert action_spec(self.adapters[seat].resolve(action, observations[seat])) == action

    def pick(self, seat, kind, predicate=lambda action: True):
        return next(action for action in self.views[seat].legal_actions
                    if action.kind == kind and predicate(action))

    def step(self, choices):
        # Materialize all responses from the same immutable snapshot before
        # asking the simulator to resolve priority or simultaneous wins.
        views_before = dict(self.views)
        moves = {}
        for seat in sorted(self.observations):
            if not views_before[seat].legal_actions:
                continue
            action = choices.get(seat)
            if action is None:
                action = next(action for action in views_before[seat].legal_actions if action.kind == "PASS")
            moves[seat] = self.adapters[seat].resolve(action, self.observations[seat])
        self.capture(self.env.step(moves))
        self.steps += 1

    def discard_draw(self, seat):
        own = self.views[seat]
        assert own.drawn_tile in own.hand
        self.step({seat: self.pick(seat, "DISCARD", lambda action: action.tile == own.drawn_tile)})

    def quiet_step(self):
        choices = {}
        for seat in sorted(self.observations):
            view = self.views[seat]
            if not view.legal_actions:
                continue
            passes = [action for action in view.legal_actions if action.kind == "PASS"]
            choices[seat] = passes[0] if passes else self.pick(
                seat, "DISCARD", lambda action: action.tile == view.drawn_tile)
        self.step(choices)

    def wait_for(self, seat, kind):
        for _ in range(24):
            if any(action.kind == kind for action in self.views[seat].legal_actions):
                return self.views[seat]
            self.quiet_step()
        pytest.fail(f"Fixed transition did not reach {kind} for seat {seat}")


def _existing(name):
    case = _CASES[name]
    return TransitionFixture({int(seat): hand for seat, hand in case["hands"].items()},
                             case["draws"], {int(index): tile for index, tile in case.get("dead_wall", {}).items()})


def _call_transition(family):
    if family == "CHI":
        fixture = TransitionFixture({1: "123456m789p456s2z"}, ["3m"])
        fixture.discard_draw(0)
        actor = 1
        call = fixture.pick(actor, "CHI", lambda action: {tile // 4 for tile in action.consume_tiles} == {3, 4})
    elif family == "PON":
        fixture = _existing("call_priority")
        fixture.discard_draw(0)
        actor = 2
        call = fixture.pick(actor, "PON")
    elif family == "DAIMINKAN":
        # Red called tile and three normal fives exercise physical identity.
        fixture = TransitionFixture({1: "555m123p456s789s1z"}, ["0m"])
        fixture.discard_draw(0)
        actor = 1
        call = fixture.pick(actor, "DAIMINKAN")
    elif family == "ANKAN":
        fixture = _existing("ankan")
        actor = 0
        call = fixture.pick(actor, "ANKAN")
    elif family == "KAKAN":
        fixture = _existing("kakan")
        actor = 1
        fixture.discard_draw(0)
        fixture.step({actor: fixture.pick(actor, "PON")})
        fixture.step({actor: fixture.pick(actor, "DISCARD", lambda action: action.tile // 4 == 32)})
        fixture.wait_for(actor, "KAKAN")
        call = fixture.pick(actor, "KAKAN")
    else:
        raise AssertionError(family)
    before = fixture.views[actor]
    fixture.step({actor: call})
    # Added-kan robbing responses are declined in one common response batch.
    after = fixture.wait_for(actor, "DISCARD")
    return fixture, actor, call, before, after


def _independent_discard_score(view, action):
    """Independent shanten engine, with the same explicit visibility contract."""
    counts = [0] * 34
    for tile in view.hand:
        counts[tile // 4] += 1
    counts[action.tile_type] -= 1
    # A declared kan still occupies a fixed meld. Neither special closed-hand
    # pattern is possible with any fixed meld, including a concealed kan.
    assert len(view.melds[view.seat]) >= 1
    baseline = Shanten.calculate_shanten(counts, use_chiitoitsu=False, use_kokushi=False)
    public_ids = set(view.hand) | set(view.dora_indicators)
    public_ids.update(tile for river in view.discards for tile in river)
    public_ids.update(tile for group in view.melds for meld in group for tile in meld.tiles)
    visible = Counter(tile // 4 for tile in public_ids)
    improving = []
    for kind in range(34):
        if visible[kind] == 4:
            continue
        counts[kind] += 1
        if Shanten.calculate_shanten(counts, use_chiitoitsu=False, use_kokushi=False) < baseline:
            improving.append(kind)
        counts[kind] -= 1
    return baseline, sum(4 - visible[kind] for kind in improving), len(improving), 1


def _audit_actor_encoding(view):
    actions = semantic_actions(view)
    encoded = encode_view(view, actions, feature_schema=FEATURE_SCHEMA_V2)
    exact = exact_efficiency_from_encoded(encoded["tiles"], encoded["action_tokens"])
    np.testing.assert_array_equal(encoded["tiles"][:, 0] * 4,
                                  [sum(tile // 4 == kind for tile in view.hand) for kind in range(34)])
    np.testing.assert_array_equal(encoded["tiles"][:, 4] * 4, visible_tile_counts(view))
    np.testing.assert_array_equal(encoded["candidate_efficiency"],
                                  exact.astype(np.float32) / np.asarray(EFFICIENCY_SCALES, dtype=np.float32))
    seen = set(view.hand) | set(view.dora_indicators)
    seen.update(tile for river in view.discards for tile in river)
    seen.update(tile for group in view.melds for meld in group for tile in meld.tiles)
    assert sum(visible_tile_counts(view)) == len(seen)
    assert all(0 <= count <= 4 for count in visible_tile_counts(view))
    discards = 0
    for index, action in enumerate(actions):
        if action.kind == "DISCARD":
            discards += 1
            score = evaluate_discard(view, action.representative)
            assert tuple(exact[index]) == (score.shanten, score.ukeire, len(score.improving_types), 1)
            assert tuple(exact[index]) == _independent_discard_score(view, action)
        else:
            assert tuple(exact[index]) == (0, 0, 0, 0)
    assert discards > 0
    return discards


@pytest.mark.parametrize("family", CALL_FAMILIES)
def test_resolved_call_has_audited_post_call_and_next_draw_features(family, record_property):
    fixture, actor, call, before, after = _call_transition(family)
    own_melds = after.melds[actor]
    assert len(own_melds) == 1
    meld = own_melds[0]
    assert len(after.hand) == 11
    assert len(set(after.hand)) == 11
    assert not set(after.hand).intersection(meld.tiles)
    assert len(meld.tiles) == (3 if family in {"CHI", "PON"} else 4)
    assert len(after.hand) + sum(3 for _ in own_melds) == 14
    assert meld.opened == (family != "ANKAN")
    assert is_closed_hand(after) == (family == "ANKAN")
    assert any(event.kind == family.lower() and event.actor == actor for event in after.history)
    assert before.hand_id == after.hand_id == 1
    assert fixture.physical["called_discard_aliases"] == int(family != "ANKAN")
    # Retained caller hand plus newly exposed meld must cover every consumed
    # physical tile. Kakan upgrades its previous pon, rather than adding a meld.
    assert set(call.consume_tiles).issubset(set(before.hand) | set(meld.tiles))
    assert set(call.consume_tiles).issubset(meld.tiles)
    if family in {"CHI", "PON", "DAIMINKAN"}:
        assert call.tile == meld.called_tile
        assert meld.called_tile in after.discards[meld.from_player]
    if family == "DAIMINKAN":
        assert 16 in meld.tiles
        assert visible_tile_counts(after)[4] == 4
    if family == "CHI":
        # Same-type and sequence-swap kuikae remain prohibited after the call.
        assert {2, 5}.issubset({tile // 4 for tile in after.hand})
        assert not {2, 5}.intersection(action.tile // 4 for action in after.legal_actions if action.kind == "DISCARD")
        assert after.drawn_tile is None
    if family == "PON":
        assert after.drawn_tile is None
    if family in {"DAIMINKAN", "ANKAN", "KAKAN"}:
        assert after.drawn_tile in after.hand
        assert len(after.dora_indicators) == (2 if family == "ANKAN" else 1)

    audited_candidates = _audit_actor_encoding(after)
    discard = next(action for action in after.legal_actions if action.kind == "DISCARD")
    fixture.step({actor: discard})
    post_discard = fixture.views[actor]
    assert len(post_discard.hand) == 10
    assert post_discard.melds[actor] == own_melds
    assert discard.tile in post_discard.discards[actor]
    assert post_discard.drawn_tile is None
    if family in {"DAIMINKAN", "KAKAN"}:
        assert len(post_discard.dora_indicators) == 2
    # Reach an actual following actor turn, exercising abbreviated hand sizes
    # after other seats' hidden draws and observable discards have accumulated.
    next_turn = fixture.wait_for(actor, "DISCARD")
    assert len(next_turn.hand) == 11 and next_turn.drawn_tile in next_turn.hand
    assert next_turn.melds[actor] == own_melds
    audited_candidates += _audit_actor_encoding(next_turn)
    record_property("call_family", family)
    record_property("resolved_calls", 1)
    record_property("actor_efficiency_snapshots", 2)
    record_property("independent_discard_candidate_checks", audited_candidates)
    record_property("explicit_wall_sha256", hashlib.sha256(json.dumps(fixture.wall).encode()).hexdigest())
    record_property("engine_steps", fixture.steps)


def test_requested_chi_preempted_by_pon_is_not_resolved_call_coverage(record_property):
    fixture = _existing("call_priority")
    fixture.discard_draw(0)
    chi = fixture.pick(1, "CHI")
    pon = fixture.pick(2, "PON")
    before = dict(fixture.views)
    fixture.step({1: chi, 2: pon})
    assert not fixture.views[1].melds[1]
    assert fixture.views[1].hand == before[1].hand
    assert len(fixture.views[2].melds[2]) == 1
    assert [event.kind for event in fixture.views[2].history if event.kind in {"chi", "pon"}] == ["pon"]
    assert fixture.physical["called_discard_aliases"] == 1
    _audit_actor_encoding(fixture.views[2])
    record_property("requests", "CHI:1,PON:1")
    record_property("resolved_calls", "CHI:0,PON:1")


def test_two_concealed_kans_preserve_closed_status_and_eight_tile_efficiency(record_property):
    fixture = TransitionFixture({0: "11112222m345p67s"}, ["8s"])
    fixture.step({0: fixture.pick(0, "ANKAN", lambda action: {tile // 4 for tile in action.consume_tiles} == {0})})
    fixture.wait_for(0, "ANKAN")
    fixture.step({0: fixture.pick(0, "ANKAN", lambda action: {tile // 4 for tile in action.consume_tiles} == {1})})
    after = fixture.wait_for(0, "DISCARD")
    assert len(after.hand) == 8
    assert len(after.melds[0]) == 2
    assert all(not meld.opened and len(meld.tiles) == 4 for meld in after.melds[0])
    assert is_closed_hand(after)
    assert len(after.dora_indicators) == 3
    assert len(after.hand) + 3 * len(after.melds[0]) == 14
    assert sum(event.kind == "ankan" and event.actor == 0 for event in after.history) == 2
    assert fixture.physical["called_discard_aliases"] == 0
    checked = _audit_actor_encoding(after)
    record_property("call_family", "ANKAN,ANKAN")
    record_property("resolved_calls", 2)
    record_property("independent_discard_candidate_checks", checked)


def test_robbed_added_kan_is_a_declaration_without_a_replacement_draw(record_property):
    # Extend the historical added-kan recipe with a genuine 1m/4m waiter.
    # Seat 2 takes a draw before the added kan, clearing any temporary furiten
    # from the first 1m discard. Chankan supplies a yaku on the later 1m.
    fixture = TransitionFixture(
        {1: "11m234678p234s56z", 2: "23m123456p789s55z"},
        ["1m", "2z", "3z", "4z", "1m"], {131: "7z"})
    fixture.discard_draw(0)
    fixture.step({1: fixture.pick(1, "PON")})
    fixture.step({1: fixture.pick(1, "DISCARD", lambda action: action.tile // 4 == 32)})
    fixture.wait_for(1, "KAKAN")
    assert len(fixture.views[1].melds[1][0].tiles) == 3
    log_start = len(fixture.env.mjai_log)
    fixture.step({1: fixture.pick(1, "KAKAN")})
    declared = list(fixture.env.mjai_log[log_start:])
    assert any(event.get("type") == "kakan" and event.get("actor") == 1 for event in declared)
    assert not any(event.get("type") == "tsumo" and event.get("actor") == 1 for event in declared)
    declared_meld_size = len(fixture.views[1].melds[1][0].tiles)
    ron = fixture.pick(2, "RON")
    fixture.step({2: ron})
    events = fixture.env.mjai_log[log_start:]
    win = next(index for index, event in enumerate(events)
               if event.get("type") == "hora" and event.get("actor") == 2 and event.get("target") == 1)
    assert not any(event.get("type") == "tsumo" and event.get("actor") == 1 for event in events[:win])
    assert sum(event.get("type") == "kakan" and event.get("actor") == 1 for event in events[:win]) == 1
    from riichi_ai.stage3c import call_resolutions
    outcomes = [outcome for outcome in call_resolutions(fixture.env.mjai_log)
                if outcome["kind"] == "KAKAN" and outcome["actor"] == 1]
    assert len(outcomes) == 1
    assert outcomes[0]["resolved"] is False and outcomes[0]["completion_event_index"] is None
    record_property("call_family", "KAKAN")
    record_property("declared_kans", 1)
    record_property("completed_kans_with_replacement_draw", 0)
    record_property("meld_size_at_pending_chankan", declared_meld_size)
