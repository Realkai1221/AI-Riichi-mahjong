"""Development safety, fixed panel membership, physical pairing and selection."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from riichi_ai import stage3g_eval as m


def fixture(tmp_path, *, fallback=False):
    names = list(m.ARM_ORDER[:2] if fallback else m.ARM_ORDER)
    arms = []
    for name in names:
        path = tmp_path / (name+'.pt')
        path.write_bytes(name.encode())
        arms.append(dict(name=name, checkpoint=str(path), sha256=hashlib.sha256(name.encode()).hexdigest()))
    config = dict(development_blocks=64, workers=14, reference_checkpoint='/reference.pt', reference_sha256='a'*64)
    manifest = dict(schema='stage3g-arms-v1', arms=arms, skipped_arms=list(m.ARM_ORDER[2:]) if fallback else [],
                    delta_gate_passed=not fallback)
    plan = dict(evaluation_seeds=list(range(64)))
    return plan, manifest, config


def prepare(tmp_path, *, fallback=False):
    plan, arms, config = fixture(tmp_path, fallback=fallback)
    return m.prepare_execution(plan, arms, config, plan_sha256='b'*64, arms_sha256='c'*64)


def test_fixed_panel_and_shared_baseline_count(tmp_path):
    execution = prepare(tmp_path)
    assert execution['planned_matches'] == 1280
    assert sum(j['arm'] == 'baseline' for j in execution['matches']) == 256
    for block in range(64):
        jobs = [j for j in execution['matches'] if j['block'] == block]
        assert {j['seed'] for j in jobs} == {block}
        for focal in range(4):
            assert len([j for j in jobs if j['focal_seat'] == focal]) == 5
            for job in jobs:
                if job['focal_seat'] == focal:
                    assert [job['lineup'][(focal+k)%4] for k in (1,2,3)] == list(m.OPPONENTS)


def test_predeclared_calibration_fallback_has_768_games(tmp_path):
    execution = prepare(tmp_path, fallback=True)
    assert execution['planned_matches'] == 768
    assert execution['skipped_arms'] == ['delta_a0', 'delta_a1']


@pytest.mark.parametrize('change', ['one_control', 'one_delta', 'unaccounted', 'missing_gate', 'unpassed_gate', 'order', 'hash', 'duplicate_seed'])
def test_plan_rejects_adaptive_or_invalid_membership(tmp_path, change):
    plan, manifest, config = fixture(tmp_path)
    if change == 'one_control':
        manifest['arms'] = manifest['arms'][1:]
    elif change == 'one_delta':
        manifest['arms'] = manifest['arms'][:3]
    elif change == 'unaccounted':
        manifest['arms'] = manifest['arms'][:2]
        manifest['delta_gate_passed'] = False
    elif change == 'missing_gate':
        manifest.pop('delta_gate_passed')
    elif change == 'unpassed_gate':
        manifest['delta_gate_passed'] = False
    elif change == 'order':
        manifest['arms'].reverse()
    elif change == 'hash':
        manifest['arms'][0]['sha256'] = '0'*64
    else:
        plan['evaluation_seeds'][-1] = 0
    with pytest.raises(ValueError):
        m.prepare_execution(plan, manifest, config, plan_sha256='b'*64, arms_sha256='c'*64)


def test_value_choice_ties_wins_and_safety():
    action = lambda kind: SimpleNamespace(kind=kind)
    assert m.value_choice([action('PON'), action('PASS')], [1,1]) == 0
    assert m.value_choice([action('PON'), action('RON')], [99,-99]) == 1
    assert m.value_choice([action('TSUMO'), action('RON')], [-1,1]) == 0
    for scores in ([float('nan'), 0], [101, 0], [0], [float('inf'), 0]):
        with pytest.raises(ValueError):
            m.value_choice([action('PON'), action('RON')], scores)


def records(execution):
    result = []
    for job in execution['matches']:
        result.append(dict(**job, completed=True, failure=None, training_eligible=False,
                           ppo_eligible=False, terminal_result_verified=True,
                           utilities=[1.5,.5,-.5,-1.5], ordinal_wall_fingerprints=['same-first', 'same-second']))
    return result


def test_zero_difference_selects_no_candidate(tmp_path):
    execution = prepare(tmp_path, fallback=True)
    result = m.summarize(execution, records(execution))
    assert result['selected_candidate'] is None
    assert result['best_observed_arm'] == 'mc_a0'
    assert result['factorial_complete'] is False
    assert all(arm['statistics']['mean_utility_difference'] == 0 for arm in result['arms'].values())


def test_positive_tie_uses_frozen_arm_order(tmp_path):
    execution = prepare(tmp_path, fallback=True)
    saved = records(execution)
    # Both candidates improve the focal seat from fourth to first in every game.
    for record in saved:
        focal = record['focal_seat']
        score = [-.5,.5,1.5]
        record['utilities'] = [None]*4
        record['utilities'][focal] = -1.5 if record['arm'] == 'baseline' else 1.5
        other = [-1.5,-.5,.5] if record['arm'] != 'baseline' else score
        for seat, value in zip([s for s in range(4) if s != focal], other):
            record['utilities'][seat] = value
    result = m.summarize(execution, saved)
    assert result['selected_candidate'] == 'mc_a0'
    assert result['arms']['mc_a1']['statistics']['mean_utility_difference'] == 3


def test_equal_seed_without_matching_physical_wall_is_rejected(tmp_path):
    execution = prepare(tmp_path, fallback=True)
    saved = records(execution)
    saved[1]['ordinal_wall_fingerprints'][0] = 'different-physical-wall'
    with pytest.raises(ValueError, match='physical walls differ'):
        m.summarize(execution, saved)


def test_different_hand_counts_compare_all_common_ordinals(tmp_path):
    execution = prepare(tmp_path, fallback=True)
    saved = records(execution)
    saved[1]['ordinal_wall_fingerprints'].append('extra-continuation')
    assert m.summarize(execution, saved)['complete']


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'failed', 'wrong_seat', 'eligible', 'missing_wall'])
def test_incomplete_or_mislabelled_panels_never_select(tmp_path, change):
    execution = prepare(tmp_path, fallback=True)
    saved = records(execution)
    if change == 'missing': saved.pop()
    elif change == 'duplicate': saved.append(deepcopy(saved[-1]))
    elif change == 'failed': saved[-1]['completed'] = False
    elif change == 'wrong_seat': saved[-1]['focal_seat'] = 9
    elif change == 'eligible': saved[-1]['training_eligible'] = True
    else: saved[-1]['ordinal_wall_fingerprints'] = []
    with pytest.raises(ValueError):
        m.summarize(execution, saved)
