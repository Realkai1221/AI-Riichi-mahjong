"""Stage 3G role isolation, match weighting, safe values and resume identities."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest
import torch

from riichi_ai.dataset import _pack_arrays
from riichi_ai.model import PolicyNetwork, efficiency_model_config, _state_hash
from riichi_ai.stage3g_data import LearningDataset, HandTargets, CORPUS_SCHEMA, COLLECTION_SCHEMA, BEHAVIORS
from riichi_ai.stage3g_learning import (ARMS, RECIPE, build_value_model, value_losses,
    update_group, _optimizer, _rank_hash, _identity, save_checkpoint, resume_checkpoint,
    load_value_checkpoint, _optimizer_state_equal)
from test_stage3f_stream import write_corpus


def corpus(directory: Path, *, split='training', matches=3):
    manifest = write_corpus(directory, matches=matches)
    manifest['schema'] = CORPUS_SCHEMA
    manifest['source_role'] = 'offline-learning-corpus'
    manifest['splits'] = ({'preflight': []} if split == 'preflight' else {'training': [], 'development': []})
    for report in manifest['matches']:
        report['schema'] = COLLECTION_SCHEMA
        report['seat_behaviors'] = list(BEHAVIORS)
        report['role'] = split
        report['source_role'] = {'training': 'offline-training', 'development': 'development-only', 'preflight': 'preflight-only'}[split]
        report['training_eligible'] = split == 'training'
        manifest['splits'][split].append(report['match_id'])
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    return manifest


def mapping(directory: Path, dataset):
    p = {'schema': 'riichi-ai-stage3g-hand-targets-v1', 'corpus_manifest_sha256': dataset.manifest_sha256,
        'calibration_gate_passed': True, 'gate': {'passed': True},
        'target': 'one_hand_placement_potential_difference',
        'matches': {name: {'source_role': report['source_role'], 'explorer_seat': 3,
            'predictor_sha256': 'a' * 64, 'hand_deltas': {'0': .125}} for name, report in dataset.reports.items()}}
    path = directory / 'hand-targets.json'
    path.write_text(json.dumps(p))
    return path, p


def test_historical_pilot_cannot_become_training_data(tmp_path):
    write_corpus(tmp_path)
    with pytest.raises(ValueError, match='Stage 3G'):
        LearningDataset(tmp_path)


@pytest.mark.parametrize('split', ['training', 'development', 'preflight'])
def test_explicit_roles_filter_explorer_optional_rows_and_keep_targets_separate(tmp_path, split):
    corpus(tmp_path, split=split)
    ds = LearningDataset(tmp_path, split)
    assert ds.shards_opened == 0
    match = ds.load_explorer(ds.match_ids[0], epoch=0, seed=1)
    assert match.total_rows == 8 and match.explorer_rows == match.optional_rows == 2
    for batch in ds.batches(match, 1):
        assert set(batch.targets['behavior_id']) == {3}
        assert set(batch.targets['seat']) == {3}
        assert not set(batch.actor) & {'final_utility', 'behavior_id', 'seat', 'hand_id'}
    assert not (tmp_path / 'auditor').exists()
    assert ds.peak_working_array_bytes < ds.memory_limit_bytes


@pytest.mark.parametrize('mutation', ['role', 'training_eligible', 'ppo_eligible', 'source_role', 'split_overlap', 'missing_member'])
def test_roles_and_eligibility_fail_closed(tmp_path, mutation):
    p = corpus(tmp_path)
    if mutation == 'role': p['matches'][0]['role'] = 'development'
    elif mutation == 'training_eligible': p['matches'][0]['training_eligible'] = False
    elif mutation == 'ppo_eligible': p['matches'][0]['ppo_eligible'] = True
    elif mutation == 'source_role': p['matches'][0]['source_role'] = 'development-only'
    elif mutation == 'split_overlap': p['splits']['development'] = [p['splits']['training'][0]]
    else: p['splits']['training'].pop()
    (tmp_path / 'manifest.json').write_text(json.dumps(p))
    with pytest.raises(ValueError): LearningDataset(tmp_path)


def test_preflight_is_not_formal_training(tmp_path):
    corpus(tmp_path, split='preflight')
    with pytest.raises(ValueError, match='split'):
        LearningDataset(tmp_path, 'training')


def test_whole_match_order_and_filtered_row_order_reproduce(tmp_path):
    corpus(tmp_path, matches=16)
    ds = LearningDataset(tmp_path)
    assert ds.match_order(0, 22) == ds.match_order(0, 22)
    assert ds.match_order(0, 22) != ds.match_order(1, 22)
    one = ds.load_explorer(ds.match_ids[0], epoch=1, seed=22)
    two = ds.load_explorer(ds.match_ids[0], epoch=1, seed=22)
    np.testing.assert_array_equal(one.indices, two.indices)
    with pytest.raises(ValueError, match='outside'):
        ds.load_explorer('unassigned', epoch=0, seed=1)


def test_hand_deltas_bound_to_hash_gate_and_explorer_only(tmp_path):
    corpus(tmp_path)
    ds = LearningDataset(tmp_path)
    path, _ = mapping(tmp_path, ds)
    labels = HandTargets(path, ds)
    match = ds.load_explorer(ds.match_ids[0], epoch=0, seed=1)
    batch = next(ds.batches(match))
    np.testing.assert_array_equal(labels.for_batch(batch), [.125, .125])
    batch.targets['seat'][:] = 0
    with pytest.raises(ValueError, match='non-explorer'):
        labels.for_batch(batch)


@pytest.mark.parametrize('mutation', ['hash', 'gate', 'top_gate', 'missing_match', 'wrong_role', 'nonfinite', 'range', 'hand', 'seat'])
def test_invalid_hand_target_mapping_rejected(tmp_path, mutation):
    corpus(tmp_path)
    ds = LearningDataset(tmp_path)
    path, p = mapping(tmp_path, ds)
    name = ds.match_ids[0]
    if mutation == 'hash': p['corpus_manifest_sha256'] = '0' * 64
    elif mutation == 'gate': p['gate']['passed'] = False
    elif mutation == 'top_gate': p['calibration_gate_passed'] = False
    elif mutation == 'missing_match': del p['matches'][name]
    elif mutation == 'wrong_role': p['matches'][name]['source_role'] = 'development-only'
    elif mutation == 'nonfinite': p['matches'][name]['hand_deltas']['0'] = float('nan')
    elif mutation == 'range': p['matches'][name]['hand_deltas']['0'] = 3.1
    elif mutation == 'hand': p['matches'][name]['hand_deltas'] = {'h1': 0.}
    else: p['matches'][name]['explorer_seat'] = 4
    path.write_text(json.dumps(p))
    with pytest.raises(ValueError): HandTargets(path, ds)


def test_cql_shaped_term_is_legal_chosen_cross_entropy():
    values = torch.tensor([[.2, -.4, torch.finfo(torch.float32).min]], requires_grad=True)
    chosen, targets = torch.tensor([1]), torch.tensor([.5])
    losses, mse, penalty = value_losses(values, chosen, targets, 1.)
    assert torch.allclose(penalty, torch.nn.functional.cross_entropy(values, chosen, reduction='none'))
    assert torch.allclose(losses, mse + penalty)
    losses.sum().backward()
    assert values.grad[0, 2] == 0


class ToyModel(torch.nn.Module):
    def __init__(self, value=0.):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value))
    def forward(self, actor):
        rows = len(actor['candidate_mask'])
        return torch.stack((self.weight, -self.weight)).expand(rows, -1), None


class ToyDataset:
    peak_working_array_bytes = 0
    loaded_match = None
    def __init__(self, counts): self.counts = counts
    def load_explorer(self, name, **kwargs):
        count = self.counts[int(name)]
        return SimpleNamespace(optional_rows=count, explorer_rows=count + 1,
            total_rows=count + 4, indices=np.arange(count), name=name)
    def batches(self, match, batch_rows):
        for start in range(0, match.optional_rows, 2):
            count = min(2, match.optional_rows - start)
            yield SimpleNamespace(targets={'chosen_index': np.zeros(count, dtype=np.int64),
                'final_utility': np.ones(count, dtype=np.float32)},
                torch_actor=lambda device, count=count: {'candidate_mask': torch.ones(count, 2, dtype=torch.bool)})


def test_match_means_include_empty_matches_with_fixed_sixteen_denominator():
    model = ToyModel()
    ds = ToyDataset([1, 3] * 4 + [0] * 8)
    stats, snap = update_group(model, _optimizer(model), ds, tuple(map(str, range(16))),
        epoch=0, arm='mc_a0', device='cpu', deadline=time.perf_counter() + 30, capture=True)
    assert stats['empty_matches'] == 8
    assert stats['optional_rows'] == 16
    assert stats['objective'] == pytest.approx(.5)
    assert snap['gradients'][0] == pytest.approx(-1.)
    assert float(model.weight.detach()) == pytest.approx(1e-4)


def test_entirely_empty_group_does_not_advance_adam_or_change_parameters():
    model = ToyModel()
    optimizer = _optimizer(model)
    stats, snap = update_group(model, optimizer, ToyDataset([0] * 16), tuple(map(str, range(16))),
        epoch=0, arm='mc_a0', device='cpu', deadline=time.perf_counter() + 30, capture=True)
    assert not stats['optimizer_step'] and not optimizer.state
    assert stats['objective'] == 0 and float(model.weight.detach()) == 0
    np.testing.assert_array_equal(snap['gradients'], [0])


@pytest.mark.parametrize('initial', [101., float('nan'), float('inf')])
def test_invalid_legal_values_stop_before_optimizer(initial):
    model = ToyModel(initial)
    optimizer = _optimizer(model)
    with pytest.raises(FloatingPointError):
        update_group(model, optimizer, ToyDataset([1] * 16), tuple(map(str, range(16))),
            epoch=0, arm='mc_a0', device='cpu', deadline=time.perf_counter() + 30)
    assert not optimizer.state


def test_deadline_prevents_next_match_read():
    model = ToyModel()
    with pytest.raises(TimeoutError):
        update_group(model, _optimizer(model), ToyDataset([1] * 16), tuple(map(str, range(16))),
            epoch=0, arm='mc_a0', device='cpu', deadline=0.)


def reference(path):
    config = efficiency_model_config({'suit_width': 4, 'history_width': 4,
        'hidden_width': 8, 'context_width': 8, 'candidate_width': 4})
    torch.manual_seed(17)
    model = PolicyNetwork(config)
    torch.save({'model_config': model.config, 'model_state_dict': model.state_dict()}, path)
    return model, hashlib.sha256(path.read_bytes()).hexdigest()


def test_common_initialization_only_resets_action_head_and_freezes_unused_rank(tmp_path):
    path = tmp_path / 'reference.pt'
    original, sha = reference(path)
    first = build_value_model(path, sha)
    second = build_value_model(path, sha)
    assert _state_hash(first.state_dict()) == _state_hash(second.state_dict())
    for name, tensor in first.state_dict().items():
        if not name.startswith('action_score.'):
            assert torch.equal(tensor, original.state_dict()[name])
    assert _state_hash(first.action_score.state_dict()) != _state_hash(original.action_score.state_dict())
    assert all(not p.requires_grad for p in first.rank_head.parameters())
    assert all(p.requires_grad for n, p in first.named_parameters() if not n.startswith('rank_head.'))
    with pytest.raises(ValueError, match='SHA'):
        build_value_model(path, '0' * 64)


def test_checkpoint_resume_identity_optimizer_and_transient_evaluation_rejection(tmp_path):
    path = tmp_path / 'reference.pt'
    _, sha = reference(path)
    model = build_value_model(path, sha)
    optimizer = _optimizer(model)
    identity = {'purpose': 'transient-preflight', 'recipe': RECIPE, 'manifest_sha256': 'a' * 64}
    initial_hash = _state_hash(model.state_dict())
    # A genuine optimizer state, with untouched frozen rank head.
    sum(p.square().sum() for p in model.parameters() if p.requires_grad).backward()
    optimizer.step()
    checkpoint = tmp_path / 'boundary.pt'
    saved = save_checkpoint(checkpoint, model, optimizer, identity=identity,
        cursor={'epoch': 1, 'next_match_position': 0}, device='cpu', initial_hash=initial_hash,
        initial_rank_hash=_rank_hash(model), scheduled_updates=1, optimizer_updates=1, complete=False)
    loaded, restored, metadata = resume_checkpoint(checkpoint, expected_identity=identity, device='cpu')
    assert _state_hash(loaded.state_dict()) == _state_hash(model.state_dict())
    assert _optimizer_state_equal(optimizer.state_dict(), restored.state_dict())
    assert metadata['cursor'] == {'epoch': 1, 'next_match_position': 0}
    with pytest.raises(ValueError, match='identity'):
        resume_checkpoint(checkpoint, expected_identity={**identity, 'recipe': {}}, device='cpu')
    with pytest.raises(ValueError, match='completed'):
        load_value_checkpoint(checkpoint)
    with pytest.raises(FileExistsError):
        save_checkpoint(checkpoint, model, optimizer, identity=identity, cursor={}, device='cpu',
            initial_hash=initial_hash, initial_rank_hash=_rank_hash(model), scheduled_updates=1,
            optimizer_updates=1, complete=False)


@pytest.mark.parametrize('schema', [None, 'riichi-ai-stage3f-collection-v1'])
def test_collection_report_requires_stage3g_schema(tmp_path, schema):
    manifest = corpus(tmp_path)
    manifest['matches'][0]['schema'] = schema
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='schema'):
        LearningDataset(tmp_path)


@pytest.mark.parametrize('behaviors', [None, [], ['explorer'] * 4,
    ['shanten_v2', 'defense_v2', 'reference', 3]])
def test_declared_behaviors_must_be_exact_four_behavior_permutation(tmp_path, behaviors):
    manifest = corpus(tmp_path)
    manifest['matches'][0]['seat_behaviors'] = behaviors
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='seat behavior'):
        LearningDataset(tmp_path)


def _swap_explorer_numeric_ids(directory, manifest):
    report = manifest['matches'][0]
    relative = f"targets/{report['match_id']}.npz"
    path = directory / relative
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    # Deliberately preserve per-seat constancy, all-seat final rewards, array
    # structure and valid behavior ranges while changing the training mask.
    arrays['behavior_id'] = np.asarray([3, 1, 2, 0], dtype=np.int8)[arrays['seat']]
    payload = _pack_arrays(arrays)
    path.write_bytes(payload)
    report['artifacts'][relative] = {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}
    report['uncompressed_target_bytes'] = sum(array.nbytes for array in arrays.values())


def test_rehashed_numeric_shard_cannot_relabel_frozen_opponent_as_explorer(tmp_path):
    manifest = corpus(tmp_path)
    _swap_explorer_numeric_ids(tmp_path, manifest)
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    dataset = LearningDataset(tmp_path)
    with pytest.raises(ValueError, match='Numeric behavior IDs'):
        dataset.load_explorer(dataset.match_ids[0], epoch=0, seed=1)


def test_matching_declared_permutation_selects_correct_explorer_seat(tmp_path):
    manifest = corpus(tmp_path)
    _swap_explorer_numeric_ids(tmp_path, manifest)
    manifest['matches'][0]['seat_behaviors'] = ['explorer', 'defense_v2', 'reference', 'shanten_v2']
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    dataset = LearningDataset(tmp_path)
    match = dataset.load_explorer(dataset.match_ids[0], epoch=0, seed=1)
    assert match.optional_rows == 2
    for batch in dataset.batches(match):
        assert set(batch.targets['seat']) == {0}
        assert set(batch.targets['behavior_id']) == {3}
