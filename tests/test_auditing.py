from dataclasses import replace

from riichi_ai.auditing import PlayerViewHasher, reference_view_digest
from riichi_ai.player import EventView, PlayerAdapter
from riichi_ai.policies import make_policy
from riichi_ai.simulator import create_env


def test_cached_commitment_matches_actual_full_history_and_resets():
    env, observations = create_env(17)
    adapters = [PlayerAdapter(seat) for seat in range(4)]
    hashes = [PlayerViewHasher() for _ in range(4)]
    policies = [make_policy("shanten", seat) for seat in range(4)]
    count = 0
    while not env.done():
        views = {seat: adapters[seat].build(obs) for seat, obs in observations.items()}
        for seat, view in views.items():
            assert hashes[seat].digest(view) == reference_view_digest(view)
            count += 1
        observations = env.step({seat: adapters[seat].resolve(policies[seat].choose(view), observations[seat])
                                 for seat, view in views.items()})
    assert count > 100


def test_history_hand_id_and_visible_state_are_committed():
    _, observations = create_env(7)
    view = PlayerAdapter(0).build(observations[0])
    hasher = PlayerViewHasher()
    original = hasher.digest(view)
    changed_history = replace(view, history=(*view.history, EventView("reach", actor=1)))
    assert hasher.digest(changed_history) != original
    assert hasher.digest(changed_history) == reference_view_digest(changed_history)
    assert hasher.digest(view) == original
    assert hasher.digest(replace(view, hand_id=view.hand_id + 1)) != original
    assert hasher.digest(replace(view, scores=(24000, 26000, 25000, 25000))) != original
