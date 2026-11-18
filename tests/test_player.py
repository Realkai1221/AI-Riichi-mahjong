"""Player-boundary tests; these fixtures do not certify the full game rules."""

import base64
from dataclasses import FrozenInstanceError, replace
import json
import random
import unittest

from riichienv import Observation

from riichi_ai.player import (
    PlayerAdapter, action_spec, is_closed_hand, mjai_tile_type,
    tile_type, visible_tile_counts,
)
from riichi_ai.rules import audit_physical_tiles, wall_for_hands
from riichi_ai.simulator import create_env


def fixture(wall):
    env, _ = create_env(0)
    observations = env.reset(wall=wall, seed=0, oya=0, round_wind=0, honba=0,
                             kyotaku=0, scores=[25_000] * 4)
    return env, observations


def kind(action):
    return str(action.action_type).rsplit(".", 1)[-1]


def pick(obs, wanted, tile=None):
    return next(action for action in obs.legal_actions()
                if kind(action) == wanted and (tile is None or action.tile == tile))


def adapters_with_initial_views(env, observations):
    adapters = [PlayerAdapter(seat) for seat in range(4)]
    missing = [seat for seat in range(4) if seat not in observations]
    observations = observations | env.get_observations(players=missing)
    return adapters, {seat: adapters[seat].build(obs) for seat, obs in observations.items()}


class PlayerTests(unittest.TestCase):
    def assert_roundtrip(self, adapter, observation):
        view = adapter.build(observation)
        native = {action_spec(action) for action in observation.legal_actions()}
        self.assertEqual(set(view.legal_actions), native)
        self.assertEqual(len(view.legal_actions), len({a.token for a in native}))
        for spec in view.legal_actions:
            self.assertEqual(action_spec(adapter.resolve(spec, observation)), spec)
        return view

    def test_frozen_views_and_exact_physical_discard_candidates(self):
        wall = wall_for_hands({0: "555m123p123s789s1z"}, draws=["0m"])
        env, obs = fixture(wall)
        adapter = PlayerAdapter(0)
        view = self.assert_roundtrip(adapter, obs[0])
        fives = [a for a in view.legal_actions if a.kind == "DISCARD" and tile_type(a.tile) == 4]
        self.assertEqual({a.tile for a in fives}, {16, 17, 18, 19})
        self.assertEqual(len({a.token for a in fives}), 4)
        # The engine's integer encoding aliases physical normal-five copies.
        encoded = {adapter.resolve(a, obs[0]).encode() for a in fives}
        self.assertLess(len(encoded), len(fives))
        with self.assertRaises(FrozenInstanceError):
            view.hand_id = 9
        with self.assertRaises(FrozenInstanceError):
            view.legal_actions[0].tile = 2
        with self.assertRaises(TypeError):
            view.hand[0] = 2
        forged = replace(fives[0], tile=135)
        with self.assertRaises(ValueError):
            adapter.resolve(forged, obs[0])
        self.assertEqual(audit_physical_tiles(env)["physical_tiles"], 136)

    def test_chi_alternatives_response_snapshot_and_visible_count(self):
        wall = wall_for_hands({1: "1240m123p456s123z"}, draws=["3m"])
        env, obs = fixture(wall)
        adapters, _ = adapters_with_initial_views(env, obs)
        obs = env.step({0: pick(obs[0], "DISCARD", env.drawn_tile)})
        response = self.assert_roundtrip(adapters[1], obs[1])
        choices = [a for a in response.legal_actions if a.kind == "CHI"]
        self.assertEqual(len({tuple(tile_type(t) for t in a.consume_tiles) for a in choices}), 3)
        self.assertTrue(any(a.kind == "PASS" for a in response.legal_actions))
        chosen = choices[0]
        saved = repr(response)
        actions = {seat: pick(o, "PASS") for seat, o in obs.items() if seat != 1}
        actions[1] = adapters[1].resolve(chosen, obs[1])
        obs = env.step(actions)
        after = self.assert_roundtrip(adapters[1], obs[1])
        self.assertEqual(repr(response), saved)
        self.assertFalse(is_closed_hand(after))
        counts = visible_tile_counts(after)
        physical = set(after.hand) | set(after.dora_indicators)
        physical.update(tile for river in after.discards for tile in river)
        physical.update(tile for melds in after.melds for meld in melds for tile in meld.tiles)
        self.assertEqual(sum(counts), len(physical))
        called = after.melds[1][0].called_tile
        self.assertIn(called, after.discards[0])
        self.assertEqual(counts[tile_type(called)], sum(tile_type(t) == tile_type(called) for t in physical))
        with self.assertRaises(ValueError):
            adapters[1].resolve(chosen, obs[1])
        audit_physical_tiles(env)

    def test_real_call_and_win_candidate_roundtrips(self):
        wall = wall_for_hands({
            1: "12m456m789p456s23z", 2: "33m234p678234s23z",
            3: "12m123p123789s11z",
        }, draws=["3m"])
        env, obs = fixture(wall)
        adapters, _ = adapters_with_initial_views(env, obs)
        obs = env.step({0: pick(obs[0], "DISCARD", env.drawn_tile)})
        found = set()
        for seat, o in obs.items():
            found.update(a.kind for a in self.assert_roundtrip(adapters[seat], o).legal_actions)
        self.assertTrue({"CHI", "PON", "RON", "PASS"}.issubset(found))

        examples = (
            ("DAIMINKAN", {1: "555m123p456s789s1z"}, ["0m"], 1),
            ("ANKAN", {0: "1111m234678p234s"}, ["5z"], 0),
            ("TSUMO", {0: "123456m123p123s1z"}, ["1z"], 0),
            ("RIICHI", {0: "123456m123p123s1z"}, ["2z"], 0),
            ("KYUSHU_KYUHAI", {0: "19m19p19s1234567z"}, ["2m"], 0),
        )
        for wanted, hands, draws, seat in examples:
            with self.subTest(kind=wanted):
                env, obs = fixture(wall_for_hands(hands, draws=draws))
                adapters, _ = adapters_with_initial_views(env, obs)
                if seat:
                    obs = env.step({0: pick(obs[0], "DISCARD", env.drawn_tile)})
                view = self.assert_roundtrip(adapters[seat], obs[seat])
                self.assertIn(wanted, {action.kind for action in view.legal_actions})
                if wanted == "RIICHI":
                    declaration = next(a for a in view.legal_actions if a.kind == wanted)
                    self.assertIsNone(declaration.tile)
                    following = env.step({seat: adapters[seat].resolve(declaration, obs[seat])})
                    after = self.assert_roundtrip(adapters[seat], following[seat])
                    self.assertTrue(all(a.kind == "DISCARD" for a in after.legal_actions))

    def test_history_idempotence_renchan_and_match_reset(self):
        wall = wall_for_hands({0: "19m19p19s1234567z"}, draws=["2m"])
        env, obs = fixture(wall)
        adapter = PlayerAdapter(0)
        first = adapter.build(obs[0])
        copied = Observation.deserialize_from_base64(obs[0].serialize_to_base64())
        self.assertIs(adapter.build(copied), first)
        self.assertEqual(first.hand_id, 1)
        self.assertEqual(sum(event.kind == "start_kyoku" for event in first.history), 1)
        following = env.step({0: pick(obs[0], "KYUSHU_KYUHAI")})
        second = adapter.build(following[0])
        self.assertEqual(second.oya, first.oya)
        self.assertEqual(second.hand_id, 2)
        self.assertEqual(second.honba, 1)
        self.assertEqual(sum(event.kind == "start_kyoku" for event in second.history), 1)
        with self.assertRaises(ValueError):
            adapter.build(obs[0])
        adapter.reset()
        fresh, fresh_obs = fixture(wall)
        self.assertEqual(adapter.build(fresh_obs[0]), first)
        audit_physical_tiles(fresh)

    def test_kakan_candidate_roundtrip_from_real_pon_continuation(self):
        wall = wall_for_hands(
            {1: "11m234678p234s56z"},
            draws=["1m", "2z", "3z", "4z", "1m"], dead_wall={131: "7z"},
        )
        env, obs = fixture(wall)
        adapters, _ = adapters_with_initial_views(env, obs)
        obs = env.step({0: pick(obs[0], "DISCARD", env.drawn_tile)})
        response = self.assert_roundtrip(adapters[1], obs[1])
        pon = next(a for a in response.legal_actions if a.kind == "PON")
        moves = {seat: pick(o, "PASS") for seat, o in obs.items() if seat != 1}
        moves[1] = adapters[1].resolve(pon, obs[1])
        obs = env.step(moves)
        self.assert_roundtrip(adapters[1], obs[1])
        green = next(tile for tile in obs[1].hand if tile_type(tile) == 32)
        obs = env.step({1: pick(obs[1], "DISCARD", green)})
        kakan = None
        for _ in range(20):
            if 1 in obs:
                view = self.assert_roundtrip(adapters[1], obs[1])
                kakan = next((a for a in view.legal_actions if a.kind == "KAKAN"), None)
                if kakan is not None:
                    break
            moves = {}
            for seat, o in obs.items():
                passes = [a for a in o.legal_actions() if kind(a) == "PASS"]
                moves[seat] = passes[0] if passes else pick(o, "DISCARD", env.drawn_tile)
            obs = env.step(moves)
        self.assertIsNotNone(kakan)
        self.assertEqual(action_spec(adapters[1].resolve(kakan, obs[1])), kakan)
        audit_physical_tiles(env)

    def test_missing_or_wrong_seat_history_fails_closed(self):
        env, obs = create_env(12)
        with self.assertRaises(ValueError):
            PlayerAdapter(1).build(obs[0])
        consumed = env.get_observation(0)
        self.assertEqual(consumed.new_events(), [])
        with self.assertRaisesRegex(ValueError, "Missing start_kyoku"):
            PlayerAdapter(0).build(consumed)

    def test_event_whitelist_hides_extra_secrets(self):
        _, obs = create_env(99)
        raw = json.loads(base64.b64decode(obs[0].serialize_to_base64()))
        for index, event_text in enumerate(raw["events"]):
            event = json.loads(event_text)
            event["wall"] = ["SECRET_WALL"]
            event["seed"] = "SECRET_SEED"
            if event["type"] == "start_kyoku":
                event["tehais"][1] = ["SECRET_OPPONENT_HAND"] * 13
            raw["events"][index] = json.dumps(event)
        raw["events"].append(json.dumps({"type": "tsumo", "actor": 1, "pai": "9s", "wall": "SECRET"}))
        modified = Observation.deserialize_from_base64(base64.b64encode(json.dumps(raw).encode()).decode())
        view = PlayerAdapter(0).build(modified)
        self.assertNotIn("SECRET", repr(view))
        self.assertIsNone(view.history[-1].tile)
        self.assertIsNone(view.history[-1].tile_type)
        self.assertIsInstance(dict(view.history[0].data)["hand"], tuple)

    def test_hidden_world_invariance_after_public_action_prefix(self):
        # Limited two-world fixtures: initial hands may differ, but all four
        # players make the same first public discards. Future unseen draws
        # differ only after the observer's second draw. Full walls keep each
        # world's private history and 136-tile allocation coherent.
        slots = lambda seat: [block * 16 + seat * 4 + offset
                              for block in range(3) for offset in range(4)] + [48 + seat]
        for seed in (1, 17, 42):
            with self.subTest(seed=seed):
                wall = list(range(136))
                random.Random(seed).shuffle(wall)
                changed = wall.copy()
                for first, second in zip(slots(1), slots(2)):
                    changed[first], changed[second] = changed[second], changed[first]
                changed[57:122] = reversed(changed[57:122])
                traces = []
                for complete_wall in (wall, changed):
                    env, obs = fixture(complete_wall)
                    adapter = PlayerAdapter(0)
                    trace = [adapter.build(obs[0])]
                    for _ in range(30):
                        moves = {}
                        for seat, o in obs.items():
                            legal = o.legal_actions()
                            passes = [a for a in legal if kind(a) == "PASS"]
                            moves[seat] = passes[0] if passes else pick(o, "DISCARD", env.drawn_tile)
                        obs = env.step(moves)
                        audit_physical_tiles(env)
                        if 0 in obs:
                            trace.append(adapter.build(obs[0]))
                            if any(kind(a) == "DISCARD" for a in obs[0].legal_actions()):
                                break
                    self.assertTrue(any(event.kind == "dahai" for event in trace[-1].history))
                    traces.append(trace)
                self.assertEqual(traces[0], traces[1])

    def test_inactive_history_hides_changed_private_draw_after_discard(self):
        wall = list(range(136))
        random.Random(17).shuffle(wall)
        changed = wall.copy()
        changed[0], changed[52] = changed[52], changed[0]
        traces = []
        for complete_wall in (wall, changed):
            env, obs = fixture(complete_wall)
            adapter = PlayerAdapter(1)
            initial = adapter.build(env.get_observation(1))
            # Same dealer hand-discard in both worlds, distinct from either
            # private draw, so public tsumogiri status also stays identical.
            common_discard = wall[1]
            obs = env.step({0: pick(obs[0], "DISCARD", common_discard)})
            trace = [initial]
            for _ in range(10):
                if 1 in obs:
                    trace.append(adapter.build(obs[1]))
                    if any(kind(a) == "DISCARD" for a in obs[1].legal_actions()):
                        break
                obs = env.step({seat: pick(o, "PASS") for seat, o in obs.items()})
            audit_physical_tiles(env)
            self.assertTrue(any(event.kind == "dahai" for event in trace[-1].history))
            traces.append(trace)
        self.assertEqual(traces[0], traces[1])

    def test_tile_helpers(self):
        self.assertEqual([mjai_tile_type(t) for t in ("5mr", "5pr", "5sr", "C", "?")],
                         [4, 13, 22, 33, None])
        for bad in (-1, 136, True):
            with self.subTest(tile=bad), self.assertRaises(ValueError):
                tile_type(bad)


if __name__ == "__main__":
    unittest.main()
