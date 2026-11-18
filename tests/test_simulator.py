"""Integration checks for the pinned engine, not a full conformance suite."""

import random
import unittest

from riichi_ai.simulator import (
    INITIAL_SCORES,
    canonical_action,
    canonical_observation,
    choose_legal_actions,
    create_env,
    run_random_hanchan,
    score_total,
)


def views(observations):
    return {seat: canonical_observation(obs) for seat, obs in observations.items()}


def initial_deal(wall):
    """Build a coherent initial-deal fixture from all 136 physical tile IDs."""
    env, _ = create_env(0)
    env.reset(wall=wall, seed=0, oya=0, round_wind=0, honba=0, kyotaku=0,
              scores=list(INITIAL_SCORES))
    return env


class SimulatorTests(unittest.TestCase):
    def assert_complete_initial_allocation(self, env):
        self.assertEqual([len(hand) for hand in env.hands], [14, 13, 13, 13])
        self.assertEqual(
            sorted(env.wall + [tile for hand in env.hands for tile in hand]),
            list(range(136)),
        )
        self.assertEqual(env.discards, [[], [], [], []])
        self.assertEqual(env.melds, [[], [], [], []])

    def test_same_seed_reset_and_steps_match(self):
        first, obs1 = create_env(426)
        second, obs2 = create_env(426)
        rng1, rng2 = random.Random(19), random.Random(19)
        for _ in range(80):
            self.assertEqual(views(obs1), views(obs2))
            actions1 = choose_legal_actions(obs1, rng1)
            # Deliberately reverse the dictionary to exercise seat ordering.
            actions2 = choose_legal_actions(dict(reversed(list(obs2.items()))), rng2)
            self.assertEqual(
                {seat: canonical_action(action) for seat, action in actions1.items()},
                {seat: canonical_action(action) for seat, action in actions2.items()},
            )
            for seat, action in actions1.items():
                self.assertIn(canonical_action(action), views(obs1)[seat]["legal_actions"])
            obs1, obs2 = first.step(actions1), second.step(actions2)
        self.assertEqual(first.mjai_log, second.mjai_log)
        self.assertEqual(score_total(first), sum(INITIAL_SCORES))

    def test_fresh_instance_restarts_seeded_match(self):
        env, initial = create_env(73)
        expected = views(initial)
        obs = initial
        rng = random.Random(11)
        for _ in range(30):
            obs = env.step(choose_legal_actions(obs, rng))
        # Upstream reset(seed=...) retains its hand counter. The adapter's
        # supported reset path is a fresh instance with the same seed.
        env, reset = create_env(73)
        self.assertEqual(expected, views(reset))

    def test_clone_continuation_and_independence(self):
        env, obs = create_env(891)
        rng = random.Random(12)
        for _ in range(25):
            obs = env.step(choose_legal_actions(obs, rng))
        clone = env.clone()
        clone_obs = clone.get_observations(players=clone.active_players)
        self.assertEqual(views(obs), views(clone_obs))
        clone_rng = random.Random()
        clone_rng.setstate(rng.getstate())
        for _ in range(100):
            actions = choose_legal_actions(obs, rng)
            clone_actions = choose_legal_actions(clone_obs, clone_rng)
            original_log = env.mjai_log
            clone_obs = clone.step(clone_actions)
            self.assertEqual(env.mjai_log, original_log)
            obs = env.step(actions)
            self.assertEqual(views(obs), views(clone_obs))
            self.assertEqual(env.mjai_log, clone.mjai_log)
        clone.set_scores([24_000, 26_000, 25_000, 25_000])
        self.assertNotEqual(env.scores(), clone.scores())

    def test_observation_contains_own_hand_only(self):
        env, _ = create_env(99)
        for seat, obs in env.get_observations(players=[0, 1, 2, 3]).items():
            snapshot = canonical_observation(obs)
            self.assertEqual(snapshot["hand"], sorted(obs.hand))
            self.assertEqual(snapshot["player_id"], seat)
            self.assertNotIn("hands", snapshot)
            self.assertNotIn("wall", snapshot)
            self.assertNotIn("events", snapshot)
            if seat == env.current_player:
                self.assertEqual(snapshot["drawn_tile"], env.drawn_tile)
            else:
                self.assertIsNone(snapshot["drawn_tile"])

    def test_own_view_invariant_to_hidden_initial_deal_changes(self):
        # Limited fixture family: the dealer's first decision, without calls,
        # discards or riichi. Alternate full walls keep the logs and state
        # coherent; this does not certify information safety in later phases.
        # RiichiEnv deals three groups of four, then one tile to each seat.
        hand_slots = {
            seat: [block * 16 + seat * 4 + offset
                   for block in range(3) for offset in range(4)] + [48 + seat]
            for seat in range(4)
        }
        for seed in (1, 17, 42):
            with self.subTest(seed=seed):
                wall = list(range(136))
                random.Random(seed).shuffle(wall)
                alternate_wall = wall.copy()
                for first, second in zip(hand_slots[1], hand_slots[2]):
                    alternate_wall[first], alternate_wall[second] = (
                        alternate_wall[second], alternate_wall[first]
                    )
                # The first 53 tiles are dealt/drawn; the last 14 form the
                # dead wall. Preserve every dead-wall slot, including dora.
                alternate_wall[53:122] = reversed(alternate_wall[53:122])
                original, alternate = initial_deal(wall), initial_deal(alternate_wall)
                for env in (original, alternate):
                    self.assert_complete_initial_allocation(env)
                self.assertEqual(original.hands[0], alternate.hands[0])
                self.assertEqual(original.drawn_tile, alternate.drawn_tile)
                self.assertEqual(original.hands[1], alternate.hands[2])
                self.assertEqual(original.hands[2], alternate.hands[1])
                self.assertNotEqual(original.wall, alternate.wall)
                self.assertEqual(original.wall[:14], alternate.wall[:14])
                self.assertEqual(original.dora_indicators, alternate.dora_indicators)
                self.assertEqual(
                    canonical_observation(original.get_observation(0)),
                    canonical_observation(alternate.get_observation(0)),
                )

    def test_inactive_view_invariant_to_opponents_private_draw(self):
        # Another limited initial-deal family: exchange one of the dealer's
        # initial 13 tiles with its first draw. All current hands and public
        # information stay identical; only the dealer knows which tile arrived.
        for seed in (1, 17, 42):
            with self.subTest(seed=seed):
                wall = list(range(136))
                random.Random(seed).shuffle(wall)
                alternate_wall = wall.copy()
                alternate_wall[0], alternate_wall[52] = wall[52], wall[0]
                original, alternate = initial_deal(wall), initial_deal(alternate_wall)
                for env in (original, alternate):
                    self.assert_complete_initial_allocation(env)
                # The engine appends the drawn tile after its sorted 13-tile
                # hand, so compare physical allocations rather than list order.
                self.assertEqual([sorted(hand) for hand in original.hands],
                                 [sorted(hand) for hand in alternate.hands])
                self.assertEqual(original.wall, alternate.wall)
                self.assertEqual(original.dora_indicators, alternate.dora_indicators)
                self.assertNotEqual(original.drawn_tile, alternate.drawn_tile)
                for observer in (1, 2, 3):
                    before = canonical_observation(original.get_observation(observer))
                    after = canonical_observation(alternate.get_observation(observer))
                    self.assertEqual(before, after)
                    self.assertIsNone(before["drawn_tile"])
                # Preserve the private draw for the player entitled to see it.
                self.assertEqual(
                    canonical_observation(original.get_observation(0))["drawn_tile"],
                    original.drawn_tile,
                )
                self.assertEqual(
                    canonical_observation(alternate.get_observation(0))["drawn_tile"],
                    alternate.drawn_tile,
                )

    def test_finished_match_conserves_scores_and_deposits(self):
        for seed in (1, 17, 42):
            with self.subTest(seed=seed):
                result = run_random_hanchan(seed)
                self.assertTrue(result.completed)
                self.assertTrue(result.conservation_ok)
                self.assertEqual(sum(result.scores) + 1_000 * result.riichi_sticks,
                                 sum(INITIAL_SCORES))
                self.assertEqual(sorted(result.ranks), [1, 2, 3, 4])
                self.assertGreater(result.steps, 0)
                self.assertGreater(result.rounds, 0)

    def test_outstanding_deposit_conservation_through_match_end(self):
        env, _ = create_env(4)
        # A valid match-start metadata fixture with one outstanding deposit.
        obs = env.reset(seed=4, scores=[24_000, 25_000, 25_000, 25_000], kyotaku=1)
        self.assertEqual(env.riichi_sticks, 1)
        rng = random.Random(29)
        for _ in range(10_000):
            self.assertEqual(score_total(env), sum(INITIAL_SCORES))
            if env.done():
                break
            obs = env.step(choose_legal_actions(obs, rng))
        self.assertTrue(env.done())
        self.assertEqual(score_total(env), sum(INITIAL_SCORES))

    def test_digest_and_limit_are_reproducible(self):
        first = run_random_hanchan(245, max_events=40)
        second = run_random_hanchan(245, max_events=40)
        self.assertEqual(first, second)
        self.assertFalse(first.completed)
        self.assertEqual(first.stop_reason, "max_events")
        self.assertEqual(first.steps, 40)
        self.assertEqual(len(first.digest), 64)
        self.assertNotEqual(first.digest, run_random_hanchan(246, 40).digest)

    def test_input_validation(self):
        for seed in (-1, 2**64, True, 1.5):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                create_env(seed)
        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                run_random_hanchan(1, limit)


if __name__ == "__main__":
    unittest.main()
