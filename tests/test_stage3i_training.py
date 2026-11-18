"""Meaningful objective, partition, frozen-target and checkpoint invariants."""
import copy
import hashlib
import json
import math

import numpy as np
import pytest
import torch

from riichi_ai.model import _state_hash
from riichi_ai.stage3i_model import ResidualHeads, load_head_bundle
from riichi_ai.stage3i_training import (CACHE_SCHEMA, FeatureCache, RECIPE, epoch_order,
    evaluate, head_losses, optimizer_for, resume_checkpoint, save_checkpoint,
    train_head, train_seed, update)


def cache_files(tmp_path, *, corrupt=None):
    shards, partitions = [], {'training': [], 'selection': [], 'calibration': [], 'gate': []}
    for partition, n, number in [('training', 5, 0), ('training', 3, 1), ('selection', 4, 2), ('gate', 2, 3)]:
        name = f'match-{number}'
        counts = np.asarray([2 + index % 2 for index in range(n)], dtype=np.int64)
        offsets = np.r_[0, np.cumsum(counts)]
        state = np.zeros((n, 256), dtype=np.float32)
        state[:, 0] = np.arange(n) / 4
        action = np.zeros((offsets[-1], 128), dtype=np.float32)
        action[:, 0] = np.arange(offsets[-1]) / 10
        values = {'state_features': state, 'action_features': action,
            'candidate_offsets': offsets.astype(np.int64), 'chosen_index': np.arange(n, dtype=np.int64) % counts,
            'target': (np.arange(n) / 8 - .2).astype(np.float32), 'row_indices': np.arange(n, dtype=np.int64) * 2}
        if corrupt and number == 0:
            corrupt(values)
        path = tmp_path / (name + '.npz')
        np.savez(path, **values)
        partitions[partition].append(name)
        shards.append({'match_id': name, 'partition': partition, 'path': str(path),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'rows': n, 'candidates': int(offsets[-1])})
    payload = {'schema': CACHE_SCHEMA, 'purpose': 'unit-test', 'target': 'one_hand_delta',
        'behavior_id': 3, 'reference_sha256': 'a' * 64, 'partitions': partitions, 'shards': shards}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(payload))
    return path


@pytest.fixture
def caches(tmp_path):
    manifest = cache_files(tmp_path)
    return FeatureCache(manifest, 'training'), FeatureCache(manifest, 'selection')


def identity():
    return {'reference_sha256': 'a' * 64, 'call_sha256': 'b' * 64,
        'protocol_sha256': 'c' * 64, 'cache_manifest_sha256': 'd' * 64,
        'recipe': RECIPE, 'source_manifest_sha256': 'e' * 64}


def test_cache_packed_gather_does_not_confuse_candidate_or_match_indices(caches):
    training, selection = caches
    assert training.rows == 8 and len(training.match_ids) == 2
    assert set(training.match_ids).isdisjoint(selection.match_ids)
    indices = np.asarray([6, 0, 4])
    batch = training.batch(indices, 'cpu')
    for local, original in enumerate(indices):
        chosen = int(batch['chosen'][local])
        np.testing.assert_array_equal(batch['action'][chosen, :256], training.state[original])
        np.testing.assert_array_equal(batch['action'][chosen, 256:],
            training.action[training.offsets[original] + training.chosen[original]])
    assert set(batch) == {'state', 'target', 'action', 'owner', 'counts', 'chosen'}
    # Target/provenance do not become columns in either head input.
    assert batch['state'].shape[1] == 256 and batch['action'].shape[1] == 384


@pytest.mark.parametrize('corrupt', [
    lambda a: a['chosen_index'].__setitem__(0, 999),
    lambda a: a['state_features'].__setitem__((0, 0), np.nan),
    lambda a: a['candidate_offsets'].__setitem__(1, 0),
    lambda a: a['target'].__setitem__(0, 100.),
    lambda a: a['row_indices'].__setitem__(1, 0),
])
def test_cache_rejects_invalid_training_evidence(tmp_path, corrupt):
    with pytest.raises(ValueError):
        FeatureCache(cache_files(tmp_path, corrupt=corrupt), 'training')


def test_cache_rejects_hash_mutation_and_partition_overlap(tmp_path):
    path = cache_files(tmp_path)
    manifest = json.loads(path.read_text())
    shard = tmp_path / 'match-0.npz'
    shard.write_bytes(shard.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        FeatureCache(path, 'training')
    manifest['partitions']['gate'].append('match-0')
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='overlap'):
        FeatureCache(path, 'training')


def test_action_ridge_is_equal_decision_weight_and_uses_all_legal_candidates():
    class Column(torch.nn.Module):
        def forward(self, x):
            return x[:, 0]
    batch = {'state': torch.zeros(2, 256), 'action': torch.zeros(5, 384),
        'owner': torch.tensor([0, 0, 1, 1, 1]), 'counts': torch.tensor([2., 3.]),
        'chosen': torch.tensor([0, 3]), 'target': torch.tensor([2., 3.])}
    batch['action'][:, 0] = torch.tensor([1., 3., 2., 4., 6.])
    fixed = torch.nn.Sequential(torch.nn.Linear(256, 1), torch.nn.Flatten(0))
    with torch.no_grad():
        fixed[0].weight.zero_(); fixed[0].bias.fill_(1)
    for p in fixed.parameters():
        p.requires_grad_(False)
    losses, squared, penalty, predicted, target = head_losses(Column(), 'action', batch, fixed)
    torch.testing.assert_close(target, torch.tensor([1., 2.]))
    torch.testing.assert_close(predicted, torch.tensor([1., 4.]))
    torch.testing.assert_close(squared, torch.tensor([0., 4.]))
    torch.testing.assert_close(penalty, torch.tensor([5., 56. / 3.]))
    torch.testing.assert_close(losses, squared + .01 * penalty)


def test_residual_training_rejects_a_moving_value_target(caches):
    training, _ = caches
    heads = ResidualHeads(19)
    with pytest.raises(ValueError, match='trainable'):
        head_losses(heads.action, 'action', training.batch([0, 1], 'cpu'), heads.value)


def test_epoch_orders_cover_every_row_once_and_are_seed_head_specific():
    first = epoch_order(100, 42, 'value', 1)
    np.testing.assert_array_equal(np.sort(first), np.arange(100))
    np.testing.assert_array_equal(first, epoch_order(100, 42, 'value', 1))
    assert not np.array_equal(first, epoch_order(100, 43, 'value', 1))
    assert not np.array_equal(first, epoch_order(100, 42, 'action', 1))
    assert not np.array_equal(first, epoch_order(100, 42, 'value', 2))


def test_validation_separates_row_and_equal_match_weights(caches):
    training, _ = caches
    head = ResidualHeads(31).value
    result = evaluate(head, 'value', training, 'cpu', math.inf)
    expected_match = np.mean([np.mean(training.target[training.match_index == index] ** 2)
                              for index in range(len(training.match_ids))])
    assert result['row_mse'] == pytest.approx(np.mean(training.target ** 2))
    assert result['match_mse'] == pytest.approx(expected_match)
    assert result['row_mse'] == pytest.approx(result['zero_prediction_row_mse'])


def test_checkpoint_resume_preserves_optimizer_moments_and_rejects_new_target(caches, tmp_path):
    training, _ = caches
    head = ResidualHeads(3).value
    optimizer = optimizer_for(head)
    batch = training.batch([0, 1, 2], 'cpu', all_actions=False)
    update(head, optimizer, 'value', batch)
    path = tmp_path / 'checkpoint.pt'
    save_checkpoint(path, head, optimizer, name='value', seed=3, epoch=1, updates=1,
                    identity=identity(), device='cpu')
    restored, restored_optimizer, metadata = resume_checkpoint(path, copy.deepcopy(head),
        expected_identity=identity(), expected_head='value', expected_seed=3, device='cpu')
    assert metadata['optimizer_updates'] == 1
    update(head, optimizer, 'value', batch)
    update(restored, restored_optimizer, 'value', batch)
    assert _state_hash(head.state_dict()) == _state_hash(restored.state_dict())
    with pytest.raises(ValueError, match='identity'):
        resume_checkpoint(path, copy.deepcopy(head), expected_identity=identity(),
            expected_head='value', expected_seed=3, device='cpu', fixed_value_sha256='f' * 64)
    with pytest.raises(FileExistsError):
        save_checkpoint(path, head, optimizer, name='value', seed=3, epoch=2, updates=2,
                        identity=identity(), device='cpu')


def test_selected_value_is_frozen_and_residual_fit_preserves_it(caches, tmp_path):
    training, selection = caches
    heads = ResidualHeads(4)
    value = train_head(heads.value, 'value', training, selection, seed=4, epochs=2,
        device='cpu', deadline=math.inf, output_dir=tmp_path / 'v', identity=identity())
    before = _state_hash(heads.value.state_dict())
    assert not any(p.requires_grad for p in heads.value.parameters())
    action = train_head(heads.action, 'action', training, selection, seed=4, epochs=2,
        device='cpu', deadline=math.inf, output_dir=tmp_path / 'a', identity=identity(), fixed_value=heads.value)
    assert before == _state_hash(heads.value.state_dict())
    assert value['row_presentations'] == action['row_presentations'] == 2 * training.rows
    assert value['selected_epoch'] == min(value['epochs'], key=lambda r: (r['selection']['row_mse'], r['epoch']))['epoch']


def test_preflight_update_boundary_cannot_be_mistaken_for_a_full_epoch(tmp_path):
    head = ResidualHeads(8).value
    optimizer = optimizer_for(head)
    path = tmp_path / 'preflight.pt'
    save_checkpoint(path, head, optimizer, name='value', seed=8, epoch=0, updates=0,
        identity=identity(), device='cpu', boundary_kind='preflight-update')
    with pytest.raises(ValueError, match='identity'):
        resume_checkpoint(path, copy.deepcopy(head), expected_identity=identity(),
            expected_head='value', expected_seed=8, device='cpu')
    _, _, metadata = resume_checkpoint(path, copy.deepcopy(head), expected_identity=identity(),
        expected_head='value', expected_seed=8, device='cpu', expected_boundary_kind='preflight-update')
    assert metadata['complete_epoch'] is False and metadata['epoch'] == 0


def test_complete_seed_bundle_matches_selected_weights_and_is_loadable(caches, tmp_path):
    training, selection = caches
    result = train_seed(training, selection, seed=41, epochs=4, device='cpu', deadline=math.inf,
        output_dir=tmp_path / 'seed', identity=identity())
    assert result['complete'] and result['optimizer_updates'] == 12
    assert result['row_presentations'] == training.rows * 12
    heads, metadata = load_head_bundle(result['checkpoint'], result['sha256'], identity())
    assert metadata['weights_sha256'] == result['weights_sha256'] == _state_hash(heads.state_dict())
    assert not any(p.requires_grad for p in heads.parameters())
