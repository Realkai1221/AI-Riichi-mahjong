"""Targeted synthetic conformance fixtures; no claim of complete certification."""

import json
from pathlib import Path
import random
import unittest

from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.hand_calculating.scores import ScoresCalculator
from mahjong.tile import TilesConverter
from mahjong.shanten import Shanten
from mahjong.meld import Meld as OracleMeld
from riichienv import ActionType as AT, Conditions, GameRule, HandEvaluator, Meld, MeldType, calculate_score

from riichi_ai.rules import (
    RULE_FIELDS, audit_physical_tiles, resolved_rule_profile, rule_profile_hash, wall_for_hands,
)
from riichi_ai.simulator import choose_legal_actions, create_env, score_total


CASES = json.loads((Path(__file__).parent / "fixtures/stage1/cases.json").read_text())["cases"]


def legal(obs, seat, kind):
    return [a for a in obs.get(seat, ()).legal_actions() if a.action_type == kind] if seat in obs else []


class RuleFixtures(unittest.TestCase):
    def start(self, name, *, dealer=0, **metadata):
        case = CASES[name]
        hands = {(int(s) + dealer) % 4: h for s, h in case["hands"].items()}
        wall = wall_for_hands(hands, case["draws"], dealer=dealer,
                              dead_wall={int(k): v for k, v in case.get("dead_wall", {}).items()})
        self.env, _ = create_env(0)
        args = dict(seed=0, oya=dealer, honba=0, kyotaku=0, round_wind=0,
                    scores=[25000] * 4)
        args.update(metadata)
        self.obs = self.env.reset(wall=wall, **args)
        self.initial_total = score_total(self.env)
        audit_physical_tiles(self.env)

    def step(self, choices):
        audit_physical_tiles(self.env)
        self.obs = self.env.step(choices)
        audit_physical_tiles(self.env)
        self.assertEqual(score_total(self.env), self.initial_total)

    def act(self, seat, kind, tile=None):
        actions = legal(self.obs, seat, kind)
        if tile is not None:
            actions = [a for a in actions if a.tile == tile]
        self.assertTrue(actions, (seat, str(kind), tile, self.env.mjai_log[-5:]))
        self.step({seat: actions[0]})

    def pass_responses(self):
        if self.obs and all(legal(self.obs, s, AT.PASS) for s in self.obs):
            self.step({s: legal(self.obs, s, AT.PASS)[0] for s in self.obs})

    def discard_draw(self, seat):
        self.assertEqual(self.env.current_player, seat)
        self.act(seat, AT.DISCARD, self.env.drawn_tile)

    def exhaust_first_hand(self):
        for _ in range(300):
            if self.env.done() or any(e["type"] == "end_kyoku" for e in self.env.mjai_log):
                return
            if all(legal(self.obs, s, AT.PASS) for s in self.obs):
                self.pass_responses()
            else:
                self.discard_draw(self.env.current_player)
        self.fail("fixture did not exhaust its first hand")

    def test_rule_profile_resolves_every_exposed_flag(self):
        profile = resolved_rule_profile()
        actual = GameRule.default_tenhou()
        exposed = {n for n in dir(actual) if not n.startswith("_") and not callable(getattr(actual, n))}
        self.assertEqual(exposed, set(RULE_FIELDS))
        self.assertEqual(profile["game_rule"], {
            "allows_ron_on_ankan_for_kokushi_musou": False,
            "is_kokushi_musou_13machi_double": False, "is_suuankou_tanki_double": False,
            "is_junsei_chuurenpoutou_double": False, "is_daisuushii_double": False,
            "yakuman_pao_is_liability_only": False, "sanchaho_is_draw": True,
            "kuikae_forbidden": True, "dealer_first_discard_is_tedashi": False,
        })
        self.assertEqual(profile["engine"]["version"], "0.4.10")
        self.assertEqual(len(rule_profile_hash()), 64)
        env, _ = create_env(1)
        self.assertEqual(env.game_mode, profile["resolved_game_mode_id"])
        fixture = json.loads((Path(__file__).parent / "fixtures/stage1/profile.json").read_text())
        self.assertEqual(profile, fixture["profile"])
        self.assertEqual(rule_profile_hash(), fixture["sha256"])

    def test_explicit_wall_fixture_recipes_reproduce_saved_allocations(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/stage1/walls.json").read_text())
        self.assertEqual(set(fixture["walls"]), set(CASES))
        for name, case in CASES.items():
            with self.subTest(name=name):
                wall = wall_for_hands({int(s): h for s, h in case["hands"].items()}, case["draws"],
                                     dead_wall={int(i): t for i, t in case.get("dead_wall", {}).items()})
                self.assertEqual(sorted(wall), list(range(136)))
                self.assertEqual(wall, fixture["walls"][name])

    def test_independent_scoring_payment_grid(self):
        # Independent calculator, all dealer/ron/tsumo/honba combinations;
        # include limit transitions and rounding, no impossible 1-han 25 fu.
        pairs = [(1, 30), (1, 40), (2, 25), (2, 30), (3, 20), (3, 40),
                 (4, 30), (4, 40), (5, 30), (6, 30), (8, 30), (11, 30), (13, 30)]
        for han, fu in pairs:
            for dealer in (False, True):
                for tsumo in (False, True):
                    for honba in (0, 2):
                        with self.subTest(han=han, fu=fu, dealer=dealer, tsumo=tsumo, honba=honba):
                            config = HandConfig(player_wind=27 if dealer else 28,
                                                is_tsumo=tsumo, tsumi_number=honba,
                                                options=OptionalRules(has_double_yakuman=False))
                            expected = ScoresCalculator.calculate_scores(han, fu, config)
                            actual = calculate_score(han, fu, dealer, tsumo, honba)
                            self.assertEqual(actual.total, expected["total"])
                            if not tsumo:
                                self.assertEqual(actual.pay_ron, expected["main"] + expected["main_bonus"])
                            else:
                                self.assertEqual(actual.pay_tsumo_ko, expected["additional"] + expected["additional_bonus"])
                                if not dealer:
                                    self.assertEqual(actual.pay_tsumo_oya, expected["main"] + expected["main_bonus"])

    def test_independent_closed_hand_scoring_and_red_five(self):
        examples = [
            (dict(man="123456", pin="123", sou="123", honors="11"), "honors", "1", False, False),
            (dict(man="1122", pin="3344", sou="5566", honors="77"), "honors", "7", False, False),
            (dict(man="123456", pin="456", sou="67822"), "man", "6", True, False),
            (dict(man="123406", pin="456", sou="67822"), "man", "6", True, False),
            (dict(man="123456", pin="789", sou="123", honors="11"), "honors", "1", False, True),
        ]
        for spec, suit, digit, tsumo, riichi in examples:
            with self.subTest(spec=spec, tsumo=tsumo, riichi=riichi):
                tiles = TilesConverter.string_to_136_array(**spec, has_aka_dora=True)
                offset = {"man": 0, "pin": 9, "sou": 18, "honors": 27}[suit]
                win = next(t for t in reversed(tiles) if t // 4 == offset + int(digit) - 1)
                config = HandConfig(is_tsumo=tsumo, is_riichi=riichi, player_wind=28, round_wind=27,
                                    options=OptionalRules(has_aka_dora=True, has_open_tanyao=True,
                                                          has_double_yakuman=False))
                expected = HandCalculator.estimate_hand_value(tiles, win, config=config)
                actual = HandEvaluator(tiles).calc(win, conditions=Conditions(tsumo=tsumo, riichi=riichi,
                                                                                 player_wind=1, round_wind=0))
                self.assertIsNone(expected.error)
                self.assertTrue(actual.is_win)
                self.assertEqual((actual.han, actual.fu), (expected.han, expected.fu))
                if tsumo:
                    self.assertEqual((actual.tsumo_agari_oya, actual.tsumo_agari_ko),
                                     (expected.cost["main"], expected.cost["additional"]))
                else:
                    self.assertEqual(actual.ron_agari, expected.cost["main"])

    def test_independent_open_tanyao_and_yakuhai_scoring(self):
        examples = [
            (dict(man="234", pin="345678", sou="23455"), range(0, 9), MeldType.Chi, OracleMeld.CHI, 22),
            (dict(man="234", pin="678", sou="789", honors="22555"), [31], MeldType.Pon, OracleMeld.PON, 28),
        ]
        for spec, kinds, native_kind, oracle_kind, winning_kind in examples:
            with self.subTest(spec=spec):
                all_tiles = TilesConverter.string_to_136_array(**spec, has_aka_dora=True)
                meld_tiles = [t for t in all_tiles if t // 4 in kinds]
                standing = [t for t in all_tiles if t not in meld_tiles]
                win = next(t for t in reversed(standing) if t // 4 == winning_kind)
                expected = HandCalculator.estimate_hand_value(
                    all_tiles, win, melds=[OracleMeld(oracle_kind, tiles=meld_tiles, opened=True)],
                    config=HandConfig(player_wind=28, round_wind=27,
                                      options=OptionalRules(has_open_tanyao=True, has_aka_dora=True,
                                                            has_double_yakuman=False)))
                actual = HandEvaluator(standing, [Meld(native_kind, meld_tiles, True, 0)]).calc(
                    win, conditions=Conditions(player_wind=1, round_wind=0))
                self.assertIsNone(expected.error)
                self.assertEqual((actual.han, actual.fu, actual.ron_agari),
                                 (expected.han, expected.fu, expected.cost["main"]))
                self.assertEqual((actual.han, actual.fu, actual.ron_agari), (1, 30, 1000))

    def test_dora_does_not_supply_a_yaku(self):
        tiles = TilesConverter.string_to_136_array(man="123456", pin="789", sou="123",
                                                   honors="11", has_aka_dora=True)
        win = tiles[-1]
        indicator = next(t for t in range(4) if t not in tiles)
        expected = HandCalculator.estimate_hand_value(
            tiles, win, dora_indicators=[indicator],
            config=HandConfig(player_wind=28, round_wind=27, options=OptionalRules(has_aka_dora=True)))
        actual = HandEvaluator(tiles).calc(win, [indicator], Conditions(player_wind=1, round_wind=0))
        self.assertIsNotNone(expected.error)
        self.assertFalse(actual.is_win)
        self.assertTrue(actual.has_win_shape)

    def test_double_ron_nearest_winner_gets_honba_and_deposits(self):
        self.start("double_ron", honba=2, kyotaku=2, scores=[23000, 25000, 25000, 25000])
        self.discard_draw(0)
        self.assertEqual({s for s in self.obs if legal(self.obs, s, AT.RON)}, {1, 2})
        self.step({s: legal(self.obs, s, AT.RON)[0] for s in self.obs})
        self.assertEqual(self.env.scores(), [11800, 35600, 27600, 25000])
        self.assertEqual(self.env.riichi_sticks, 0)
        self.assertEqual((self.env.oya, self.env.honba), (1, 0))

    def test_triple_ron_is_abortive_and_keeps_deposits(self):
        self.start("triple_ron", honba=2, kyotaku=2, scores=[23000, 25000, 25000, 25000])
        self.discard_draw(0)
        self.step({s: legal(self.obs, s, AT.RON)[0] for s in self.obs})
        self.assertEqual(self.env.scores(), [23000, 25000, 25000, 25000])
        self.assertEqual((self.env.oya, self.env.honba, self.env.riichi_sticks), (0, 3, 2))
        self.assertTrue(any(e.get("reason") == "sanchaho" for e in self.env.mjai_log))

    def test_ron_preempts_pon_and_chi(self):
        self.start("call_priority")
        self.discard_draw(0)
        self.step({1: legal(self.obs, 1, AT.CHI)[0], 2: legal(self.obs, 2, AT.PON)[0],
                   3: legal(self.obs, 3, AT.RON)[0]})
        events = self.env.mjai_log
        self.assertTrue(any(e.get("type") == "hora" and e["actor"] == 3 for e in events))
        self.assertFalse(any(e.get("type") in ("pon", "chi") for e in events))

    def test_pon_preempts_chi_and_called_tile_is_counted_once(self):
        self.start("call_priority")
        self.discard_draw(0)
        self.step({1: legal(self.obs, 1, AT.CHI)[0], 2: legal(self.obs, 2, AT.PON)[0],
                   3: legal(self.obs, 3, AT.PASS)[0]})
        self.assertEqual(self.env.current_player, 2)
        self.assertEqual(audit_physical_tiles(self.env)["called_discard_aliases"], 1)
        self.assertEqual([e["type"] for e in self.env.mjai_log if e["type"] in ("chi", "pon")], ["pon"])

    def test_temporary_furiten_clears_after_own_turn(self):
        self.start("temporary_furiten")
        self.discard_draw(0)
        self.assertTrue(legal(self.obs, 3, AT.RON))
        self.pass_responses()
        self.discard_draw(1)
        self.assertFalse(legal(self.obs, 3, AT.RON))
        self.pass_responses()
        for seat in (2, 3):
            self.discard_draw(seat)
            self.pass_responses()
        self.discard_draw(0)
        self.assertTrue(legal(self.obs, 3, AT.RON))

    def test_own_discard_furiten_blocks_ron_but_not_tsumo(self):
        self.start("discard_furiten")
        self.discard_draw(0)
        self.pass_responses()
        self.discard_draw(1)
        self.assertFalse(legal(self.obs, 0, AT.RON))
        self.pass_responses()
        for seat in (2, 3):
            self.discard_draw(seat)
            self.pass_responses()
        self.assertTrue(legal(self.obs, 0, AT.TSUMO))

    def test_closed_kan_reveals_dora_before_replacement_draw(self):
        self.start("ankan")
        n = len(self.env.mjai_log)
        self.act(0, AT.ANKAN)
        events = [e["type"] for e in self.env.mjai_log[n:]]
        self.assertEqual(events[:3], ["ankan", "dora", "tsumo"])
        self.assertEqual(len(self.env.dora_indicators), 2)
        self.assertEqual(len(self.env.melds[0][0].tiles), 4)

    def test_open_kan_delays_dora_until_discard(self):
        self.env, _ = create_env(0)
        wall = wall_for_hands({1: "111m234678p234s5z"}, ["1m"])
        self.obs = self.env.reset(wall=wall, seed=0, oya=0)
        self.initial_total = score_total(self.env)
        self.discard_draw(0)
        self.step({s: (legal(self.obs, s, AT.DAIMINKAN)[0] if s == 1 else legal(self.obs, s, AT.PASS)[0]) for s in self.obs})
        self.assertEqual(len(self.env.dora_indicators), 1)
        self.assertEqual(len(self.env.melds[1][0].tiles), 4)
        self.discard_draw(1)
        self.assertEqual(len(self.env.dora_indicators), 2)

    def test_chi_kuikae_forbids_same_tile_and_sequence_swap(self):
        self.env, _ = create_env(0)
        wall = wall_for_hands({1: "123456m789p456s2z"}, ["3m"])
        self.obs = self.env.reset(wall=wall, seed=0, oya=0)
        self.initial_total = score_total(self.env)
        self.discard_draw(0)
        chi = next(a for a in legal(self.obs, 1, AT.CHI) if {t // 4 for t in a.consume_tiles} == {3, 4})
        self.step({s: (chi if s == 1 else legal(self.obs, s, AT.PASS)[0]) for s in self.obs})
        self.assertTrue({2, 5}.issubset({t // 4 for t in self.env.hands[1]}))
        self.assertFalse({2, 5}.intersection({a.tile // 4 for a in legal(self.obs, 1, AT.DISCARD)}))

    def test_riichi_ankan_requires_unchanged_wait(self):
        for name, allowed in (("riichi_ankan_allowed", True), ("riichi_ankan_forbidden", False)):
            with self.subTest(name=name):
                self.start(name)
                self.act(0, AT.RIICHI)
                self.discard_draw(0)
                self.pass_responses()
                for seat in (1, 2, 3):
                    self.discard_draw(seat)
                    self.pass_responses()
                self.assertEqual(bool(legal(self.obs, 0, AT.ANKAN)), allowed)
                self.assertEqual(self.env.riichi_sticks, 1)
                self.assertEqual(self.env.scores()[0], 24000)

    def test_riichi_requires_at_least_one_thousand_points(self):
        for score, allowed in ((900, False), (1000, True)):
            with self.subTest(score=score):
                self.start("riichi_ankan_allowed", scores=[score, 33000, 33000, 34000 - score])
                self.assertEqual(bool(legal(self.obs, 0, AT.RIICHI)), allowed)

    def test_ron_on_riichi_discard_prevents_deposit_acceptance(self):
        self.start("riichi_discard_ron")
        self.act(0, AT.RIICHI)
        self.assertEqual(self.env.scores(), [25000] * 4)
        self.assertEqual(self.env.riichi_sticks, 0)
        self.discard_draw(0)
        self.step({s: (legal(self.obs, s, AT.RON)[0] if s == 1 else legal(self.obs, s, AT.PASS)[0]) for s in self.obs})
        self.assertEqual(self.env.scores(), [17000, 33000, 25000, 25000])
        self.assertEqual(self.env.riichi_sticks, 0)
        self.assertFalse(any(e["type"] == "reach_accepted" for e in self.env.mjai_log))

    def test_declining_ron_after_riichi_remains_furiten(self):
        self.start("riichi_furiten")
        for seat in (0, 1, 2):
            self.discard_draw(seat)
            self.pass_responses()
        self.act(3, AT.RIICHI)
        self.discard_draw(3)
        self.pass_responses()
        self.discard_draw(0)
        self.assertTrue(legal(self.obs, 3, AT.RON))
        self.pass_responses()
        for seat in (1, 2, 3):
            self.discard_draw(seat)
            self.pass_responses()
        self.discard_draw(0)
        self.assertFalse(legal(self.obs, 3, AT.RON))

    def test_added_kan_delays_dora_until_discard(self):
        self.start("kakan")
        self.discard_draw(0)
        self.step({s: (legal(self.obs, s, AT.PON)[0] if s == 1 else legal(self.obs, s, AT.PASS)[0]) for s in self.obs})
        green = next(t for t in self.env.hands[1] if t // 4 == 32)
        self.act(1, AT.DISCARD, green)
        self.pass_responses()
        for seat in (2, 3, 0):
            self.discard_draw(seat)
            self.pass_responses()
        self.act(1, AT.KAKAN)
        self.pass_responses()
        self.assertEqual(len(self.env.dora_indicators), 1)
        self.discard_draw(1)
        self.assertEqual(len(self.env.dora_indicators), 2)
        self.assertEqual(len(self.env.melds[1][0].tiles), 4)

    def test_nine_terminals_abort_repeats_dealer(self):
        self.start("kyuushu")
        self.act(0, AT.KYUSHU_KYUHAI)
        self.assertEqual((self.env.oya, self.env.honba), (0, 1))
        self.assertEqual(self.env.scores(), [25000] * 4)

    def test_dealer_win_continues_and_collects_deposits(self):
        self.start("dealer_tenhou", honba=2, kyotaku=1, scores=[24000, 25000, 25000, 25000])
        self.act(0, AT.TSUMO)
        self.assertEqual(self.env.scores(), [73600, 8800, 8800, 8800])
        self.assertEqual((self.env.oya, self.env.honba, self.env.riichi_sticks), (0, 3, 0))

    def test_last_dealer_top_win_ends_hanchan(self):
        self.start("dealer_tenhou", dealer=3, round_wind=1)
        self.act(3, AT.TSUMO)
        self.assertTrue(self.env.done())
        self.assertEqual(self.env.ranks()[3], 1)
        self.assertEqual(self.env.scores(), [9000, 9000, 9000, 73000])

    def test_tied_ranks_use_initial_seat_order(self):
        self.start("dealer_tenhou", dealer=3, round_wind=1, scores=[25000] * 4)
        self.assertEqual(self.env.ranks(), [1, 2, 3, 4])

    def test_exhaustive_draw_settlement_and_dealer_continuation(self):
        for name, ready, next_dealer in (("exhaustive_dealer_tenpai", 0, 0),
                                         ("exhaustive_dealer_noten", 1, 1)):
            with self.subTest(name=name):
                self.start(name)
                # Independently establish readiness of every starting hand.
                for seat, hand in enumerate(self.env.hands):
                    tiles = hand if seat else [t for t in hand if t != self.env.drawn_tile]
                    shanten = Shanten.calculate_shanten(TilesConverter.to_34_array(tiles))
                    self.assertEqual(shanten == 0, seat == ready)
                self.exhaust_first_hand()
                expected = [24000] * 4
                expected[ready] = 28000
                self.assertEqual(self.env.scores(), expected)
                self.assertEqual((self.env.oya, self.env.honba), (next_dealer, 1))

    def test_south_four_extends_when_everyone_below_target(self):
        self.start("exhaustive_dealer_noten", dealer=3, round_wind=1)
        self.exhaust_first_hand()
        self.assertFalse(self.env.done())
        self.assertEqual((self.env.round_wind, self.env.oya, self.env.honba), (2, 0, 1))

    def test_last_dealer_tenpai_below_target_continues(self):
        self.start("exhaustive_dealer_tenpai", dealer=3, round_wind=1)
        self.exhaust_first_hand()
        self.assertFalse(self.env.done())
        self.assertEqual((self.env.round_wind, self.env.oya, self.env.honba), (1, 3, 1))
        self.assertEqual(self.env.scores()[3], 28000)

    def test_last_dealer_tenpai_top_at_target_stops(self):
        self.start("exhaustive_dealer_tenpai", dealer=3, round_wind=1,
                   scores=[23700, 23700, 23600, 29000])
        self.exhaust_first_hand()
        self.assertTrue(self.env.done())
        self.assertEqual(self.env.scores()[3], 32000)

    def test_final_draw_outstanding_deposit_goes_to_top(self):
        self.start("exhaustive_dealer_noten", dealer=3, round_wind=1, kyotaku=1,
                   scores=[29000, 23000, 23000, 24000])
        self.exhaust_first_hand()
        self.assertTrue(self.env.done())
        self.assertEqual(self.env.scores(), [33000, 22000, 22000, 23000])
        self.assertEqual(self.env.riichi_sticks, 0)

    def test_west_four_stops_at_extension_cap(self):
        self.start("exhaustive_dealer_noten", dealer=3, round_wind=2)
        self.exhaust_first_hand()
        self.assertTrue(self.env.done())
        self.assertEqual(max(self.env.scores()), 28000)

    def test_bankruptcy_is_negative_not_zero(self):
        for score, ends in ((8000, False), (7900, True)):
            with self.subTest(starting_score=score):
                self.start("double_ron", scores=[score, 30000, 30000, 40000 - score])
                self.discard_draw(0)
                self.step({s: (legal(self.obs, s, AT.RON)[0] if s == 1 else legal(self.obs, s, AT.PASS)[0]) for s in self.obs})
                self.assertEqual(self.env.scores()[0], score - 8000)
                self.assertEqual(self.env.done(), ends)

    def test_west_round_non_dealer_crossing_target_ends_match(self):
        self.start("double_ron", round_wind=2)
        self.discard_draw(0)
        self.step({s: (legal(self.obs, s, AT.RON)[0] if s == 1 else legal(self.obs, s, AT.PASS)[0]) for s in self.obs})
        self.assertTrue(self.env.done())
        self.assertEqual(self.env.scores()[1], 33000)

    def test_physical_conservation_during_genuine_full_matches(self):
        total_melds = 0
        for seed in (10, 17, 42):
            with self.subTest(seed=seed):
                env, observations = create_env(seed)
                rng = random.Random(seed ^ 1234)
                for _ in range(10000):
                    report = audit_physical_tiles(env)
                    total_melds += report["melds"]
                    if env.done():
                        break
                    observations = env.step(choose_legal_actions(observations, rng))
                self.assertTrue(env.done())
        self.assertGreater(total_melds, 0)


if __name__ == "__main__":
    unittest.main()
