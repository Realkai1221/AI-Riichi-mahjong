"""Small genuine matches; no formal corpus or training is created by tests."""

import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from riichi_ai.dataset import generate_dataset
from riichi_ai.model import PolicyNetwork
from riichi_ai.stage2b_data import (
    derive_efficiency, generate_stage2b_dataset, load_efficiency_split,
    plan_stage2b_matches, validate_stage2b_manifest, verify_stage2b_dataset,
    make_aggregation_config,
)


def small_config(checkpoint):
    path = Path(__file__).resolve().parents[1] / 'configs/stage2b.json'
    config = json.loads(path.read_text())
    for index, group in enumerate(config['data_groups']):
        group['matches'] = 1
        group['root_seed'] = 2026101781 + index
    config['historical_checkpoint'] = str(checkpoint)
    config['historical_checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return config


@pytest.fixture(scope='module')
def source_checkpoint(tmp_path_factory):
    # Exercise real checkpoint loading with a small deterministic test model.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(992)
        model = PolicyNetwork({'suit_width': 8, 'history_width': 4, 'hidden_width': 16,
                               'context_width': 8, 'candidate_width': 8})
    path = tmp_path_factory.mktemp('student-checkpoint') / 'student.pt'
    torch.save({'model_config': model.config, 'model_state_dict': model.state_dict()}, path)
    return path


@pytest.fixture(scope='module')
def focal_pilot(tmp_path_factory, source_checkpoint):
    path = tmp_path_factory.mktemp('focal-pilot')
    config = small_config(source_checkpoint)
    report = generate_stage2b_dataset(path / 'data', config, workers=1)
    assert report['valid'], report['failure']
    derived = derive_efficiency(path / 'data', path / 'derived', splits=('diagnostic', 'validation', 'test'))
    assert derived['valid'], derived['failure']
    return path, config, report, derived


def test_full_plan_has_frozen_groups_seat_balance_and_no_seed_reuse():
    config = json.loads((Path(__file__).resolve().parents[1] / 'configs/stage2b.json').read_text())
    plan = plan_stage2b_matches(config)
    assert len(plan) == 300
    assert len({r['seed'] for r in plan}) == 300
    assert len({r['independent_group'] for r in plan}) == 300
    for group in config['data_groups']:
        records = [r for r in plan if r['group'] == group['name']]
        counts = [sum(r['focal_seat'] == seat for r in records) for seat in range(4)]
        assert max(counts) - min(counts) <= 1
    assert [sum(r['split'] == split for r in plan) for split in ('diagnostic', 'validation', 'test')] == [100, 100, 100]
    bad = copy.deepcopy(config)
    bad['data_groups'][0]['root_seed'] = 2026101701
    with pytest.raises(ValueError, match='earlier'):
        plan_stage2b_matches(bad)


def test_only_focal_rows_teacher_labels_separate_from_behavior(focal_pilot):
    path, config, report, _ = focal_pilot
    disagreements = 0
    for record in report['matches']:
        auditor = json.loads(gzip.decompress((path / 'data/auditor' / (record['match_id'] + '.json.gz')).read_bytes()))
        focal = record['focal_seat']
        for relative, opponent in enumerate(config['opponents'], 1):
            assert auditor['policy_provenance'][(focal + relative) % 4]['policy'] == opponent
        rows = []
        for batch in auditor['policy_requests']:
            assert 0 <= batch['event_start'] <= batch['event_end'] <= len(auditor['canonical_events'])
            for request in batch['requests']:
                if request['seat'] != focal:
                    assert request['row'] is None and 'teacher' not in request
                    continue
                rows.append(request['row'])
                assert request['behavior_probability'] == 1
                disagreements += request['behavior_chosen_index'] != request['teacher']['chosen_index']
                for index in request['resolved_event_indices']:
                    assert batch['event_start'] <= index < batch['event_end']
                    assert auditor['canonical_events'][index]['actor'] == focal
        assert rows == list(range(record['decisions']))
        with np.load(path / 'data/targets' / (record['match_id'] + '.npz')) as targets:
            assert np.all(targets['seat'] == focal)
            assert np.all(targets['final_rank'] == record['final_ranks'][focal])
    assert disagreements > 0  # Actual student behavior is not replaced by teacher labels.


def test_all_groups_full_replay_reproduces_behavior_targets_and_features(focal_pilot):
    path, _, manifest, _ = focal_pilot
    report = verify_stage2b_dataset(path / 'data', replay_matches_per_group=1)
    assert report['valid'], report['errors']
    assert report['replayed_matches'] == 5
    assert report['replayed_rows'] == manifest['decisions']


def test_raw_and_assisted_samples_share_exact_labels_indices_and_strata(focal_pilot):
    path, _, _, _ = focal_pilot
    raw = load_efficiency_split(path / 'data', None, 'validation', assisted=False)
    assisted = load_efficiency_split(path / 'data', path / 'derived', 'validation')
    assert np.array_equal(raw.index, assisted.index)
    assert set(raw.stratum_indices) == {'teacher', 'student'}
    assert all(len(values) for values in raw.stratum_indices.values())
    assert sum(map(len, raw.stratum_indices.values())) == len(raw)
    for index in range(len(raw)):
        one, two = raw[index], assisted[index]
        for key in one['encoded']:
            assert np.array_equal(one['encoded'][key], two['encoded'][key])
        for key in ('policy_target', 'best_mask', 'teacher_preference_keys'):
            assert np.array_equal(one[key], two[key])
        assert one['stratum'] == two['stratum']
        assert 'teacher_preference_keys' not in two['encoded']
        assert two['encoded']['candidate_efficiency'].shape == (len(two['policy_target']), 4)
    assert raw.dataset_identity()['index_sha256'] == assisted.dataset_identity()['index_sha256']
    assert raw.dataset_identity()['feature_schema'] != assisted.dataset_identity()['feature_schema']


def test_auditor_files_cannot_affect_actor_loader(focal_pilot, tmp_path):
    path, _, _, _ = focal_pilot
    shutil.copytree(path / 'data', tmp_path / 'data')
    expected = load_efficiency_split(tmp_path / 'data', path / 'derived', 'validation')[0]
    shutil.rmtree(tmp_path / 'data/auditor')
    actual = load_efficiency_split(tmp_path / 'data', path / 'derived', 'validation')[0]
    for key in expected['encoded']:
        assert np.array_equal(expected['encoded'][key], actual['encoded'][key])


def test_new_test_is_sealed_until_explicit_final_access(focal_pilot):
    path, _, _, _ = focal_pilot
    with pytest.raises(ValueError, match='sealed'):
        load_efficiency_split(path / 'data', path / 'derived', 'test')
    data = load_efficiency_split(path / 'data', path / 'derived', 'test', allow_test=True)
    assert len(data) > 0


def test_derive_does_not_reuse_old_test_and_checks_exact_old_teacher_keys(tmp_path):
    old = generate_dataset(tmp_path / 'old', matches=3, root_seed=2026101789)
    assert old['valid'], old['failure']
    report = derive_efficiency(tmp_path / 'old', tmp_path / 'old-derived')
    assert report['valid'], report['failure']
    assert report['completed_matches'] == 2
    assert report['checked_rows'] == sum(r['decisions'] for r in old['matches'] if r['split'] == 'train')
    with pytest.raises(ValueError, match='test'):
        derive_efficiency(tmp_path / 'old', tmp_path / 'forbidden', splits=('test',))
    with pytest.raises(ValueError, match='Stage 2A'):
        load_efficiency_split(tmp_path / 'old', None, 'test', assisted=False, allow_test=True)


def test_truncation_preserves_failure_audit_without_rank_training_shards(tmp_path, source_checkpoint):
    report = generate_stage2b_dataset(tmp_path, small_config(source_checkpoint), workers=1, max_steps=1)
    assert not report['valid'] and report['failure']['type'] == 'TruncatedMatch'
    assert not (tmp_path / 'actor').exists()
    assert report['completed_matches'] == 0 and len(report['not_written_matches']) == 4
    with pytest.raises(ValueError, match='valid'):
        validate_stage2b_manifest(tmp_path)


def test_split_metadata_corruption_is_rejected(focal_pilot, tmp_path):
    path, _, _, _ = focal_pilot
    shutil.copytree(path / 'data', tmp_path / 'data')
    file = tmp_path / 'data/manifest.json'
    manifest = json.loads(file.read_text())
    manifest['splits']['validation'].append(manifest['splits']['test'][0])
    file.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='splits'):
        validate_stage2b_manifest(tmp_path / 'data')


def test_derived_hash_corruption_and_memory_limit_fail(focal_pilot, tmp_path):
    path, _, _, _ = focal_pilot
    shutil.copytree(path / 'derived', tmp_path / 'derived')
    name = json.loads((path / 'data/manifest.json').read_text())['splits']['validation'][0]
    file = tmp_path / 'derived/efficiency' / (name + '.npz')
    file.write_bytes(file.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='Corrupted'):
        load_efficiency_split(path / 'data', tmp_path / 'derived', 'validation')
    with pytest.raises(MemoryError):
        load_efficiency_split(path / 'data', path / 'derived', 'validation', memory_limit_bytes=1)


def test_bounded_parallel_collection_matches_serial_artifact_hashes(focal_pilot, tmp_path):
    _, config, serial, _ = focal_pilot
    parallel = generate_stage2b_dataset(tmp_path, config, workers=2)
    assert parallel['valid'], parallel['failure']
    assert [r['artifacts'] for r in serial['matches']] == [r['artifacts'] for r in parallel['matches']]


def test_conditional_training_is_separate_bound_to_frozen_base_and_checkpoint(focal_pilot, tmp_path):
    path, config, base, _ = focal_pilot
    conditional = make_aggregation_config(config, path / 'data', config['historical_checkpoint'],
                                         config['historical_checkpoint_sha256'], matches=1)
    plan = plan_stage2b_matches(conditional)
    assert len(plan) == 1 and plan[0]['split'] == 'train'
    assert plan[0]['behavior'] == 'frozen_candidate'
    assert plan[0]['seed'] not in {r['seed'] for r in base['matches']}
    generated = generate_stage2b_dataset(tmp_path / 'aggregation', conditional, workers=1)
    assert generated['valid'], generated['failure']
    audit = verify_stage2b_dataset(tmp_path / 'aggregation')
    assert audit['valid'], audit['errors']
    derived = derive_efficiency(tmp_path / 'aggregation', tmp_path / 'derived', splits=('train',))
    assert derived['valid'], derived['failure']
    data = load_efficiency_split(tmp_path / 'aggregation', tmp_path / 'derived', 'train')
    assert len(data.stratum_indices['student']) == len(data) > 0
    assert len(data.stratum_indices['teacher']) == 0
    with pytest.raises(ValueError, match='at most 300'):
        make_aggregation_config(config, path / 'data', config['historical_checkpoint'],
                                config['historical_checkpoint_sha256'], matches=301)
    wrong = copy.deepcopy(conditional)
    wrong['opponents'][0] = 'defense_v2'
    with pytest.raises(ValueError, match='main frozen protocol'):
        plan_stage2b_matches(wrong)
    wrong = copy.deepcopy(conditional)
    wrong['conditional_collection']['base_manifest_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='base manifest changed'):
        plan_stage2b_matches(wrong)


def test_loaded_checkpoint_hash_is_enforced_before_first_request(focal_pilot, monkeypatch):
    from riichi_ai import stage2b_data
    _, config, _, _ = focal_pilot
    plan = plan_stage2b_matches(config)[0]
    wrong = copy.deepcopy(config)
    wrong['historical_checkpoint_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='Loaded focal checkpoint'):
        stage2b_data._policies(wrong, plan)
