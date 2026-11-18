"""Independent evidence arithmetic guards; no formal training or games."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location('audit_stage3g_learning_tests',
    Path(__file__).resolve().parents[1] / 'scripts/audit_stage3g_learning.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def gate_fixture():
    p = np.full((360, 4), .25)
    ranks = np.repeat(np.arange(120) % 4, 3)
    weights = np.full(360, 1 / 3)
    groups = np.repeat(np.arange(120), 3)
    stages = np.tile(np.arange(3), 120)
    return p, p.copy(), ranks, weights, groups, stages


def update_fixture():
    schedule, records = [], []
    for index in range(400):
        expected = {'epoch': index // 200, 'match_ids': [f'match-{(index % 200) * 16 + i:06d}' for i in range(16)],
            'row_order_sha256': f'order-{index}', 'optional_rows': 160, 'explorer_rows': 200,
            'all_seat_rows': 800, 'empty_matches': 0}
        schedule.append(expected)
        records.append({**expected, 'scheduled_update': index + 1, 'optimizer_updates': index + 1,
            'optimizer_step': True, 'matches': 16, 'clipped': False, 'gradient_l2_before_clip': .5,
            'objective': 1., 'mse': .4, 'conservative': .6,
            'cursor': {'epoch': (index + 1) // 200, 'next_match_position': (index + 1) % 200 * 16}})
    return records, schedule


def development_fixture():
    jobs, records = [], []
    for block in range(64):
        for seat in range(4):
            for arm in ('baseline', 'mc_a0', 'mc_a1'):
                job = {'match_id': f'block-{block:03d}-seat-{seat}-{arm}', 'block': block,
                       'focal_seat': seat, 'arm': arm, 'seed': block}
                jobs.append(job)
                scores = [40000, 30000, 20000, 10000]
                if arm != 'baseline':
                    scores = [10000] * 4
                    scores[seat] = 70000
                ranks = [0] * 4
                for rank, winner in enumerate(sorted(range(4), key=lambda i: (-scores[i], i)), 1):
                    ranks[winner] = rank
                records.append({**job, 'completed': True, 'failure': None, 'training_eligible': False,
                    'ppo_eligible': False, 'source_role': 'development-only', 'terminal_result_verified': True,
                    'scores': scores, 'ranks': ranks, 'utilities': [float(audit.UTILITY[r - 1]) for r in ranks],
                    'ordinal_wall_fingerprints': ['physical-wall-A']})
    return {'matches': jobs, 'blocks': 64, 'rotations': 4, 'tie_order': ['mc_a0', 'mc_a1']}, records


def test_public_reconstruction_ignores_answer_and_hidden_fields():
    hand = {'hand_id': 2, 'bakaze': 'E', 'kyoku': 1, 'honba': 1, 'kyotaku': 0,
            'oya': 0, 'scores': [25000] * 4}
    expected = audit.public_features(hand, 2)
    changed = dict(hand, terminal_ranks=[4, 3, 2, 1], hidden_hands='secret', policy_id='new')
    np.testing.assert_array_equal(expected, audit.public_features(changed, 2))
    assert expected.shape == (31,)
    assert expected[-1] == pytest.approx(.1)


def test_independent_baseline_is_uniform_and_normalized():
    p = audit.baseline_probabilities(np.asarray([[2.5] * 4, [5.5, 1.5, 1.5, 1.5]]))
    np.testing.assert_allclose(p[0], .25, atol=1e-12)
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-12)
    assert p[1, 0] > .8


def test_independent_gate_accepts_calibrated_supported_equal_baseline():
    result = audit.gate_arithmetic(*gate_fixture())
    assert result['passed']
    assert result['proper_score_gate_passed']
    assert result['strata']['late']['unique_games'] == 120


def test_independent_gate_rejects_unsupported_stage():
    p, b, ranks, weights, groups, stages = gate_fixture()
    stages[:] = 0
    result = audit.gate_arithmetic(p, b, ranks, weights, groups, stages)
    assert not result['passed']
    assert result['strata']['late']['supported_mass'] == 0


def test_independent_gate_rejects_inaccurate_utility_even_if_proper_scores_tie():
    p, b, ranks, weights, groups, stages = gate_fixture()
    ranks[:] = 0
    result = audit.gate_arithmetic(p, b, ranks, weights, groups, stages)
    assert not result['passed']
    assert result['proper_score_gate_passed']
    assert result['strata']['overall']['weighted_absolute_bias'] == pytest.approx(1.5)


def test_independent_gate_detects_better_baseline():
    p, _, ranks, weights, groups, stages = gate_fixture()
    result = audit.gate_arithmetic(p, np.eye(4)[ranks], ranks, weights, groups, stages)
    assert not result['proper_score_gate_passed']


def test_exact_update_schedule_and_prefix_counts():
    records, schedule = update_fixture()
    total, prefixes, steps = audit.check_update_records(records, schedule)
    assert steps == 400
    assert total['match_presentations'] == 6400
    assert total['optional_row_presentations'] == 64000
    assert prefixes[200]['match_presentations'] == 3200


@pytest.mark.parametrize('key,value', [('match_ids', ['wrong'] * 16), ('row_order_sha256', 'wrong'),
    ('optional_rows', 159), ('epoch', 1), ('scheduled_update', 2), ('optimizer_updates', 2),
    ('optimizer_step', False), ('matches', 15), ('clipped', True), ('objective', float('nan'))])
def test_wrong_update_evidence_rejected(key, value):
    records, schedule = update_fixture()
    records[0][key] = value
    with pytest.raises(ValueError):
        audit.check_update_records(records, schedule)


def test_missing_update_cannot_pass_completion():
    records, schedule = update_fixture()
    with pytest.raises(ValueError, match='400'):
        audit.check_update_records(records[:-1], schedule)


def test_empty_group_has_no_adam_step_but_keeps_fixed_match_denominator():
    records, schedule = update_fixture()
    schedule[0]['optional_rows'] = records[0]['optional_rows'] = 0
    schedule[0]['empty_matches'] = records[0]['empty_matches'] = 16
    records[0]['optimizer_step'] = False
    for record in records:
        record['optimizer_updates'] -= 1
    total, _, steps = audit.check_update_records(records, schedule)
    assert steps == 399
    assert total['match_presentations'] == 6400
    assert total['empty_match_presentations'] == 16


def test_development_selection_recomputed_and_tie_order_preserved():
    execution, records = development_fixture()
    result = audit.development_arithmetic(execution, records)
    assert result['selected_candidate'] == result['best_observed_arm'] == 'mc_a0'
    assert result['arms']['mc_a0']['mean_utility'] == 1.5
    assert result['arms']['mc_a0']['baseline_mean_utility'] == 0
    assert result['shared_physical_wall_pairs_checked'] == 512


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'walls', 'ranks', 'roles', 'seed'])
def test_development_bad_evidence_rejected(mutation):
    execution, records = development_fixture()
    if mutation == 'missing':
        records.pop()
    elif mutation == 'duplicate':
        records[-1] = deepcopy(records[0])
    elif mutation == 'walls':
        records[1]['ordinal_wall_fingerprints'] = ['different']
    elif mutation == 'ranks':
        records[0]['ranks'] = [4, 3, 2, 1]
    elif mutation == 'roles':
        records[0]['training_eligible'] = True
    else:
        records[0]['seed'] = 999
    with pytest.raises(ValueError):
        audit.development_arithmetic(execution, records)
