"""Independent audit arithmetic and corruption checks; no actual game starts."""
import importlib.util
from copy import deepcopy
from pathlib import Path
import hashlib

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('stage3j_audit', ROOT / 'scripts/stage3j_audit.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def decision(mode='learned_tie'):
    keys = ['discard-a', 'discard-b', 'discard-c']
    logits, q = [2., 2., -20.], [.49, .49, .02]
    residuals = [0., 1., 100.] if mode in ('learned_tie', 'fractional') else [0., 0., 0.]
    chosen = {'reference_wrapped': 0, 'learned_tie': 1, 'fractional': 2}.get(mode, 0)
    hashes = [None] * 3
    scores = {'reference_wrapped': [0., 0., -8.], 'learned_tie': [0., 1., None],
              'fractional': [0., .25, 17.]}.get(mode)
    if mode == 'hash_tie':
        for i in (0, 1):
            hashes[i] = hashlib.sha256(m.encoded([m.HASH_SALT, 'a' * 64, keys[i]])).hexdigest()
        order = sorted((0, 1), key=lambda i: (hashes[i], i))
        chosen = order[0]
        scores = [-float(order.index(i)) if i in order else None for i in range(3)]
    detail = dict(mode=mode, selected_index=chosen, raw_reference_index=0, wrapped_reference_index=0,
                  raw_corrected_index=chosen, explorer_probabilities=q, action_residuals=residuals,
                  eligible=[True] * 3, prior=[0., 0., -8.], strength=.25 if mode == 'fractional' else 0.,
                  exact_tie_mask=[True, True, False], eligible_tie_mask=[True, True, False],
                  scores=scores, tie_hashes=hashes, actor_fingerprint='a' * 64 if mode == 'hash_tie' else None,
                  selected_reference_gap=logits[0] - logits[chosen],
                  change_classification='unchanged' if chosen == 0 else 'exact_tie' if chosen == 1 else 'positive_margin',
                  wrapper_active=False, wrapper_applied=False)
    if mode == 'fractional':
        detail['companion_tie_selected_index'] = 1
    return dict(row_index=0, seat=0, semantic_keys=keys, action_kinds=['DISCARD'] * 3,
                representative_tokens=['a', 'b', 'c'], selected_index=chosen, selected_semantic_key=keys[chosen],
                selected_kind='DISCARD', selected_physical_token=['a', 'b', 'c'][chosen], probability=1.,
                training_eligible=False, ppo_eligible=False, legal_scores=logits, correction=detail)


@pytest.mark.parametrize('mode', m.MODES)
def test_independent_choice_formula_all_modes(mode):
    checked = m.audit_decision(decision(mode), mode, mode)
    assert checked['optional'] is True
    assert checked['exact_top_tie'] is True
    assert checked['eligible_top_tie'] is True
    assert checked['classification'] == decision(mode)['correction']['change_classification']


@pytest.mark.parametrize('field', ['token', 'selected', 'eligibility', 'prior', 'tie_mask', 'float32',
                                   'nonfinite', 'strength', 'mode', 'probability', 'head', 'score',
                                   'classification', 'margin', 'wrapper', 'hidden_hash', 'row_role'])
def test_independent_choice_rejects_corrupted_arithmetic(field):
    row = decision()
    detail = row['correction']
    if field == 'token': row['selected_physical_token'] = 'd'
    elif field == 'selected': row['selected_index'] = 2
    elif field == 'eligibility': detail['eligible'][2] = False
    elif field == 'prior': detail['prior'][2] = -22.
    elif field == 'tie_mask': detail['exact_tie_mask'][2] = True
    elif field == 'float32': row['legal_scores'][1] = 2. + 1e-12
    elif field == 'nonfinite': row['legal_scores'][1] = float('nan')
    elif field == 'strength': detail['strength'] = .25
    elif field == 'mode': detail['mode'] = 'fractional'
    elif field == 'probability': detail['explorer_probabilities'][0] = .1
    elif field == 'head': detail['action_residuals'][1] = -1.
    elif field == 'score': detail['scores'][1] = 100.
    elif field == 'classification': detail['change_classification'] = 'positive_margin'
    elif field == 'margin': detail['selected_reference_gap'] = 1.
    elif field == 'wrapper': detail['wrapper_active'] = True
    elif field == 'hidden_hash': detail['actor_fingerprint'] = 'b' * 64
    else: row['ppo_eligible'] = True
    with pytest.raises(ValueError):
        m.audit_decision(row, 'tie_seed_20261024011', 'learned_tie')


def test_fractional_companion_arithmetic_is_independent():
    row = decision('fractional')
    assert m.audit_decision(row, 'small_seed_20261024011', 'fractional')['fractional_companion_diff']
    row['correction']['companion_tie_selected_index'] = 0
    with pytest.raises(ValueError, match='companion'):
        m.audit_decision(row, 'small_seed_20261024011', 'fractional')


def test_hash_control_rejects_modified_rank_and_learned_inputs():
    row = decision('hash_tie')
    row['correction']['tie_hashes'][0] = '0' * 64
    with pytest.raises(ValueError, match='hash'):
        m.audit_decision(row, 'hash_tie')
    row = decision('hash_tie')
    row['correction']['action_residuals'][2] = 1.
    with pytest.raises(ValueError, match='residual'):
        m.audit_decision(row, 'hash_tie')


def test_win_wrapper_positive_raw_gap_is_unchanged_not_override():
    row = decision('learned_tie')
    row['action_kinds'][2] = 'RON'
    row['selected_index'] = 2
    row['selected_semantic_key'] = row['semantic_keys'][2]
    row['selected_kind'] = 'RON'
    row['selected_physical_token'] = 'c'
    row['correction'].update(selected_index=2, wrapped_reference_index=2, wrapper_active=True,
                             wrapper_applied=True, selected_reference_gap=22., change_classification='unchanged',
                             explorer_probabilities=[0., 0., 1.], eligible=[True, False, True],
                             eligible_tie_mask=[True, False, False], raw_corrected_index=0,
                             scores=[0., None, None])
    checked = m.audit_decision(row, 'tie_seed_20261024011')
    assert checked['changed'] is False
    assert checked['classification'] == 'unchanged'
    row['correction']['explorer_probabilities'] = [.49, .49, .02]
    with pytest.raises(ValueError, match='one-hot'):
        m.audit_decision(row, 'tie_seed_20261024011')


def response_fixture():
    policies = [dict(policy_id=f'policy-{seat}') for seat in range(4)]
    request = dict(step=0, player_views={'0': 'view-zero', '1': 'view-one'}, rows=[0, 1], submitted=True)
    identity = hashlib.sha256(m.encoded(dict(step=0, player_views=request['player_views']))).hexdigest()
    request['pre_response_id'] = identity
    rows = [dict(row_index=seat, step=0, seat=seat, pre_response_id=identity,
                 player_view_sha256=request['player_views'][str(seat)], policy_id=policies[seat]['policy_id'])
            for seat in (0, 1)]
    return rows, [request], policies


def test_simultaneous_responses_bind_same_pre_response_state():
    m.audit_response_links(*response_fixture())


@pytest.mark.parametrize('field', ['state', 'membership', 'seat', 'policy', 'step'])
def test_simultaneous_response_corruption_rejected(field):
    rows, requests, policies = response_fixture()
    if field == 'state': rows[1]['pre_response_id'] = 'after-first-response'
    elif field == 'membership': requests[0]['rows'] = [0]
    elif field == 'seat': rows[1]['seat'] = 0
    elif field == 'policy': rows[1]['policy_id'] = 'different-policy'
    else: rows[1]['step'] = 1
    with pytest.raises(ValueError):
        m.audit_response_links(rows, requests, policies)


def replay_fixture():
    arms = [dict(name=f'arm-{i}') for i in range(8)]
    jobs = [dict(match_id=f'block-{block}-{arm["name"]}', arm=arm['name'], scores=[0, 1, 2, 3])
            for arm in arms for block in range(16)]
    return dict(arms=arms, matches=jobs)


def test_replays_stratified_and_independent_of_outcome_or_record_order():
    execution = replay_fixture()
    left = m.replay_selection(execution)
    altered = deepcopy(execution)
    altered['matches'].reverse()
    for job in altered['matches']: job['scores'] = [100, -100, 0, 0]
    right = m.replay_selection(altered)
    assert [x['match_id'] for x in left] == [x['match_id'] for x in right]
    assert len(left) == 32
    assert all(sum(x['arm'] == arm['name'] for x in left) == 4 for arm in execution['arms'])


def test_replay_rejects_missing_arm():
    execution = replay_fixture()
    execution['matches'] = [x for x in execution['matches'] if x['arm'] != 'arm-3']
    with pytest.raises(ValueError):
        m.replay_selection(execution)


def test_history_extracts_declared_seeds_and_first_wall_only():
    data = dict(seed=1, seeds=[2, '3'], evaluation_seeds=[4, 5], not_a_seed=6,
                rows=[dict(seed=7, ordinal_wall_fingerprints=['first', 'second'])])
    seeds, walls = m.historical_values(data)
    assert seeds == {1, 2, 4, 5, 7}
    assert walls == {'first'}


def arrays():
    return dict(tiles=np.zeros((34, 23), np.float32), globals=np.zeros(45, np.float32),
                history=np.zeros((128, 10), np.int16), history_mask=np.zeros(128, bool),
                action_tokens=np.zeros((3, 8), np.int16), candidate_efficiency=np.zeros((3, 4), np.float32))


def test_independent_fingerprint_matches_declared_bytes_and_rejects_hidden_fields():
    from riichi_ai.stage3j_model import actor_feature_fingerprint
    value = arrays()
    assert m.independent_actor_fingerprint(value) == actor_feature_fingerprint(value)
    value['wall'] = np.zeros(136, np.int16)
    with pytest.raises(ValueError, match='allowlist'):
        m.independent_actor_fingerprint(value)


def test_statistics_keeps_independent_blocks_and_honest_zero_variance():
    result = m.statistics([0.] * 4)
    assert result['independent_blocks'] == 4
    assert result['standard_error'] == 0
    assert result['bootstrap_percentile_95_exploratory'] == [0., 0.]
    assert result['fixed_budget_hoeffding_95'][0] < 0 < result['fixed_budget_hoeffding_95'][1]


def test_independent_descriptive_statistics_matches_fixed_stream():
    from riichi_ai.evaluation import summarize_blocks
    values = [-1., .5, .25, -.75, 1.5, 0]
    expected = summarize_blocks(values, bootstrap_seed=20261029031)
    actual = m.statistics(values)
    for key in actual:
        assert np.allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12)
