"""Fixed membership, historical identity, fresh walls and match accounting."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from riichi_ai import stage3j_evaluation as m


def fixture(tmp_path, blocks=2):
    bundles = []
    for seed in m.TRAINING_SEEDS:
        path = tmp_path / f'{seed}.pt'
        path.write_bytes(str(seed).encode())
        bundles.append(dict(seed=seed, checkpoint=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    source = dict(reference_sha256='b'*64, call_sha256='c'*64,
                  protocol_sha256='d'*64, cache_manifest_sha256='e'*64)
    plan = dict(training_seeds=list(m.TRAINING_SEEDS), selected_bundles=bundles,
                source_identity=source, protocol_sha256='f'*64,
                evaluation_seeds=list(range(202610290000, 202610290000+blocks)),
                preflight_evaluation_seeds=[202610289000,202610289001],
                forward_sample_blocks=list(m.FORWARD_SAMPLE_BLOCKS))
    config = dict(reference_checkpoint='/reference.pt', reference_sha256='b'*64,
                  call_checkpoint='/call.pt', call_sha256='c'*64, workers=14,
                  evaluation_blocks=blocks, preflight_evaluation_blocks=2)
    return plan, config


def prepare(tmp_path, blocks=2, preflight=False):
    plan, config = fixture(tmp_path, blocks)
    return m.prepare_execution(plan, config, plan_sha256='a'*64, preflight=preflight)


def test_fixed_panel_all_seeds_modes_rotations_and_fresh_sample(tmp_path):
    execution = prepare(tmp_path, blocks=256)
    assert execution['planned_matches'] == 8192
    assert [entry['name'] for entry in execution['arms']] == list(m.ARM_NAMES)
    assert execution['source_identity']['protocol_sha256'] == 'd'*64
    assert execution['protocol_sha256'] == 'f'*64
    assert execution['forward_sample_blocks'] == list(m.FORWARD_SAMPLE_BLOCKS)
    assert sum(job['block'] in execution['forward_sample_blocks'] for job in execution['matches']) == 256
    assert execution['seed_selection'] is False
    assert execution['policy_promotion'] is False
    for job in execution['matches']:
        focal = job['focal_seat']
        assert [job['lineup'][(focal+offset) % 4] for offset in (1,2,3)] == list(m.OPPONENTS)
    assert {entry['strength'] for entry in execution['arms'] if entry['mode'] == 'fractional'} == {.25}


def test_preflight_same_arms_disjoint_walls_and_fixed_sample(tmp_path):
    execution = prepare(tmp_path, preflight=True)
    assert execution['planned_matches'] == 64
    assert execution['forward_sample_blocks'] == [0]
    assert {job['seed'] for job in execution['matches']} == {202610289000,202610289001}
    assert execution['source_role'] == 'preflight-only'


@pytest.mark.parametrize('change', ['missing_seed','seed_order','bundle_order','bundle_hash',
                                   'source_reference','source_missing','overlap','duplicate','bad_seed','samples'])
def test_prepare_rejects_selection_identity_or_freshness_changes(tmp_path, change):
    plan, config = fixture(tmp_path)
    if change == 'missing_seed': plan['training_seeds'].pop()
    elif change == 'seed_order': plan['training_seeds'].reverse()
    elif change == 'bundle_order': plan['selected_bundles'].reverse()
    elif change == 'bundle_hash': plan['selected_bundles'][0]['sha256'] = '0'*64
    elif change == 'source_reference': plan['source_identity']['reference_sha256'] = '0'*64
    elif change == 'source_missing': del plan['source_identity']['protocol_sha256']
    elif change == 'overlap': plan['preflight_evaluation_seeds'][0] = plan['evaluation_seeds'][0]
    elif change == 'duplicate': plan['evaluation_seeds'][1] = plan['evaluation_seeds'][0]
    elif change == 'bad_seed': plan['evaluation_seeds'][0] = True
    else: plan['forward_sample_blocks'] = [0]
    with pytest.raises(ValueError):
        m.prepare_execution(plan, config, plan_sha256='a'*64)


def test_policy_validation_uses_historical_source_not_new_protocol(tmp_path, monkeypatch):
    execution = prepare(tmp_path)
    monkeypatch.setattr(m, 'frozen_policy', lambda *args: SimpleNamespace(metadata={}))
    by_path = {entry['checkpoint']: entry['seed'] for entry in execution['arms'] if 'checkpoint' in entry}
    def load(path, sha):
        return None, dict(seed=by_path[path], identity=dict(execution['source_identity']),
                          complete=True, benchmark_only=False)
    monkeypatch.setattr(m, 'candidate_policy', load)
    assert len(m.validate_policies(execution)['candidates']) == 6
    execution['source_identity']['protocol_sha256'] = execution['protocol_sha256']
    def mismatched(path, sha):
        _, metadata = load(path, sha)
        metadata['identity']['protocol_sha256'] = 'd'*64
        return None, metadata
    monkeypatch.setattr(m, 'candidate_policy', mismatched)
    with pytest.raises(ValueError):
        m.validate_policies(execution)


def test_dependencies_authenticate_and_reconstruct_historical_seed_wall_union(tmp_path):
    plan, config = fixture(tmp_path)
    dependencies = []
    for stem in ('reference', 'call'):
        path = tmp_path / (stem+'.pt')
        path.write_bytes(stem.encode())
        config[stem+'_checkpoint'] = str(path)
        config[stem+'_sha256'] = m.digest(path)
        dependencies.append(dict(path=str(path), sha256=m.digest(path), role=stem))
    dependencies.extend(dict(path=bundle['checkpoint'], sha256=bundle['sha256'], role='saved_residual_head')
                        for bundle in plan['selected_bundles'])
    old = tmp_path / 'historical.json'
    old.write_text(json.dumps(dict(seed=42, nested=dict(evaluation_seeds=[43,44]),
                                  ordinal_wall_fingerprints=['wall-first','wall-second'])))
    snapshot = tmp_path / 'history-snapshot.json'
    snapshot.write_text(json.dumps(dict(schema='stage3j-historical-inputs-v1',
        inputs=[dict(path=str(old), sha256=m.digest(old))], historical_seeds=[42,43,44],
        historical_first_wall_fingerprints=['wall-first'])))
    reference = dict(path=str(snapshot), sha256=m.digest(snapshot))
    dependencies.append(dict(**reference, role='historical_seed_wall_inventory'))
    plan['dependencies'] = dependencies
    plan['freshness'] = dict(historical_seed_sources=[reference], historical_seeds=[42,43,44],
                             historical_first_wall_fingerprints=['wall-first'])
    m.validate_dependencies(plan, config)
    plan['evaluation_seeds'][0] = 42
    with pytest.raises(ValueError, match='historical seed'):
        m.validate_dependencies(plan, config)
    plan['evaluation_seeds'][0] = 202610290000
    old.write_text('{}')
    with pytest.raises(ValueError, match='identity failed'):
        m.validate_dependencies(plan, config)


def records(execution):
    return [dict(**job, completed=True, failure=None, training_eligible=False, ppo_eligible=False,
                 terminal_result_verified=True, utilities=[1.5,.5,-.5,-1.5],
                 ordinal_wall_fingerprints=['first','second'], game_sha256='a'*64,
                 focal_diagnostics=dict(optional_decisions=10, changed=0)) for job in execution['matches']]


def cheap_statistics(monkeypatch):
    monkeypatch.setattr(m, 'summarize_blocks', lambda values, **kwargs: dict(
        mean_utility_difference=float(np.mean(values)), independent_blocks=len(values)))


def test_summary_compares_fixed_membership_with_no_seed_selection(tmp_path, monkeypatch):
    cheap_statistics(monkeypatch)
    execution = prepare(tmp_path)
    result = m.summarize(execution, records(execution))
    assert result['selected_candidate'] is None and result['seed_selection'] is False
    assert len(result['contrasts']) == 13
    assert len(result['pairing']) == 13*2*4
    assert result['contrasts']['small_seed_20261024011_vs_tie_seed_20261024011']['block_utility_differences'] == [0,0]
    assert result['matched_tie_small']['20261024011']['identical_canonical_games'] == 8
    assert result['by_arm']['hash_tie']['games'] == 8
    assert result['by_arm']['hash_tie']['focal_diagnostics']['optional_decisions'] == 80


@pytest.mark.parametrize('change', ['wall','missing','duplicate','failed','eligibility','utility'])
def test_summary_rejects_unpaired_incomplete_or_mislabelled_games(tmp_path, monkeypatch, change):
    cheap_statistics(monkeypatch)
    execution = prepare(tmp_path)
    saved = records(execution)
    if change == 'wall': saved[1]['ordinal_wall_fingerprints'][0] = 'different'
    elif change == 'missing': saved.pop()
    elif change == 'duplicate': saved.append(deepcopy(saved[0]))
    elif change == 'failed': saved[0]['completed'] = False
    elif change == 'eligibility': saved[0]['training_eligible'] = True
    else: saved[0]['utilities'] = [0,0,0,0]
    with pytest.raises(ValueError):
        m.summarize(execution, saved)


def test_diagnostics_separate_exact_ties_and_positive_margin_overrides():
    def row(selected, reference, category, kinds=('DISCARD','DISCARD')):
        return dict(seat=0, selected_index=selected, legal_scores=[1,1], action_kinds=kinds,
                    correction=dict(change_classification=category, wrapped_reference_index=reference,
                                    exact_tie_mask=[True,True], eligible_tie_mask=[True,True],
                                    wrapper_active=False, wrapper_applied=False))
    rows = [row(1,0,'exact_tie'), row(1,0,'positive_margin'), row(0,0,'unchanged')]
    rows[1]['correction']['companion_tie_selected_index'] = 0
    rows.append(dict(seat=0, legal_scores=None, correction=dict(forced=True)))
    result = m.decision_diagnostics(rows, 0)
    assert result['optional_decisions'] == 3 and result['forced_decisions'] == 1
    assert result['changed'] == 2 and result['exact_tie_changes'] == result['positive_margin_changes'] == 1
    assert result['reference_discard_changes'] == 2
    assert result['small_vs_tie_decision_disagreements'] == 1
