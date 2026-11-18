"""Saved-gradient diagnostics: exact weighting, separation and zero updates."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from riichi_ai.model import _state_hash
from riichi_ai.stage3h_gradients import (BLOCK_NAMES, cancellation, compare_backends,
    describe_groups, gradient_group, pair_statistics, parameter_layout)


class ToyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(1, 1, bias=False)
        self.action_score = torch.nn.Linear(1, 1, bias=False)
        self.efficiency_encoder = torch.nn.Linear(1, 1, bias=False)
        self.rank_head = torch.nn.Linear(1, 4)
        with torch.no_grad():
            self.backbone.weight.fill_(1.)
            self.action_score.weight.fill_(.2)
            self.efficiency_encoder.weight.fill_(.1)
        for parameter in self.rank_head.parameters():
            parameter.requires_grad_(False)

    def forward(self, actor):
        context = self.backbone(actor['globals'])
        score = self.action_score(context)
        values = torch.cat((score, -score), dim=1)
        values = values + self.efficiency_encoder(actor['candidate_efficiency']).squeeze(-1)
        return values, self.rank_head(context)


class ToyBatch:
    def __init__(self, match, begin, end):
        self.targets = {'chosen_index': np.arange(begin, end, dtype=np.int64) % 2,
            'final_utility': np.full(end - begin, .5 if match % 2 else -1.5, dtype=np.float32)}
        self.rows = end - begin
        self.match = match

    def torch_actor(self, device):
        return {'globals': torch.full((self.rows, 1), .5 + self.match / 16, device=device),
            'candidate_mask': torch.ones((self.rows, 2), dtype=torch.bool, device=device),
            'candidate_efficiency': torch.tensor([[[.1], [.2]]], device=device).repeat(self.rows, 1, 1)}


class ToyDataset:
    def __init__(self, empty=False):
        self.split = 'training'
        self.match_ids = [f'match-{i}' for i in range(16)]
        self.peak_working_array_bytes = 128
        self.loaded_match = None
        self.empty = empty

    def load_explorer(self, name, epoch, seed):
        index = self.match_ids.index(name)
        count = 0 if self.empty and index == 0 else 2 + index % 3
        return SimpleNamespace(optional_rows=count, explorer_rows=count + 1,
            indices=np.arange(count), index=index,
            targets={'behavior_id': np.full(count, 3), 'seat': np.zeros(count), 'hand_id': np.ones(count)})

    def batches(self, match, batch_rows):
        for begin in range(0, match.optional_rows, 2):
            yield ToyBatch(match.index, begin, min(begin + 2, match.optional_rows))


def test_parameter_partition_preserves_names_offsets_and_excludes_rank():
    named, ranges, metadata = parameter_layout(ToyActor())
    assert len(named) == 3
    assert set(ranges) == set(BLOCK_NAMES)
    assert metadata['trainable_parameters'] == 3
    assert [(v['start'], v['stop']) for v in metadata['ordered_parameters']] == [(0, 1), (1, 2), (2, 3)]
    assert metadata['frozen_rank_parameters'] == 8


def test_unfrozen_rank_is_rejected():
    model = ToyActor()
    model.rank_head.weight.requires_grad_(True)
    with pytest.raises(ValueError, match='frozen'):
        parameter_layout(model)


def test_gradient_decomposition_matches_independent_whole_match_objective(monkeypatch):
    monkeypatch.setattr(torch.optim, 'Adam', lambda *args, **kwargs: pytest.fail('No optimizer may be constructed'))
    dataset = ToyDataset(empty=True)
    model, independent = ToyActor(), ToyActor()
    independent.load_state_dict(model.state_dict())
    before = _state_hash(model.state_dict())
    result = gradient_group(model, dataset, dataset.match_ids, target='terminal_utility', device='cpu', deadline=time.monotonic() + 30)
    mse = ce = 0.
    for index, name in enumerate(dataset.match_ids):
        match = dataset.load_explorer(name, epoch=0, seed=0)
        if not match.optional_rows:
            continue
        batch = ToyBatch(index, 0, match.optional_rows)
        values, _ = independent(batch.torch_actor('cpu'))
        chosen = values.gather(1, torch.from_numpy(batch.targets['chosen_index'])[:, None]).squeeze(1)
        mse = mse + ((chosen - torch.from_numpy(batch.targets['final_utility'])).square().mean() / 16)
        ce = ce + ((torch.logsumexp(values, dim=1) - chosen).mean() / 16)
    parameters = [p for p in independent.parameters() if p.requires_grad]
    expected_mse = torch.autograd.grad(mse, parameters, retain_graph=True)
    expected_ce = torch.autograd.grad(ce, parameters)
    np.testing.assert_allclose(result['mse_gradient'], np.concatenate([g.numpy().ravel() for g in expected_mse]), atol=1e-6)
    np.testing.assert_allclose(result['ce_gradient'], np.concatenate([g.numpy().ravel() for g in expected_ce]), atol=1e-6)
    assert result['mse'] == pytest.approx(float(mse.detach()), abs=1e-6)
    assert result['ce'] == pytest.approx(float(ce.detach()), abs=1e-6)
    assert result['empty_matches'] == 1
    assert result['weights_unchanged'] and _state_hash(model.state_dict()) == before
    assert result['optimizer_steps'] == 0
    assert all(p.grad is None for p in model.parameters())


def test_wrong_split_and_duplicate_group_rejected():
    dataset = ToyDataset()
    dataset.split = 'development'
    with pytest.raises(ValueError, match='training'):
        gradient_group(ToyActor(), dataset, dataset.match_ids, target='terminal_utility', device='cpu', deadline=time.monotonic() + 30)
    dataset.split = 'training'
    with pytest.raises(ValueError, match='distinct'):
        gradient_group(ToyActor(), dataset, [dataset.match_ids[0]] * 16, target='terminal_utility', device='cpu', deadline=time.monotonic() + 30)


def test_delta_requires_separate_immutable_target_mapping():
    data = ToyDataset()
    with pytest.raises(ValueError, match='mapping'):
        gradient_group(ToyActor(), data, data.match_ids, target='one_hand_delta', device='cpu', deadline=time.monotonic() + 30)


def test_outcome_cannot_be_added_as_actor_tensor(monkeypatch):
    original = ToyBatch.torch_actor
    def leaked(batch, device):
        return {**original(batch, device), 'final_utility': torch.zeros(batch.rows, device=device)}
    monkeypatch.setattr(ToyBatch, 'torch_actor', leaked)
    data = ToyDataset()
    with pytest.raises(ValueError, match='actor tensor'):
        gradient_group(ToyActor(), data, data.match_ids, target='terminal_utility', device='cpu', deadline=time.monotonic() + 30)


def test_expired_deadline_cannot_modify_weights():
    model, dataset = ToyActor(), ToyDataset()
    before = _state_hash(model.state_dict())
    with pytest.raises(TimeoutError):
        gradient_group(model, dataset, dataset.match_ids, target='terminal_utility', device='cpu', deadline=time.monotonic() - 1)
    assert _state_hash(model.state_dict()) == before


def test_pair_cosine_and_combination_clip_arithmetic():
    result = pair_statistics(np.asarray([3., 0.]), np.asarray([-1., 0.]))
    assert result['mse_ce_cosine'] == -1
    assert result['combined']['0']['gradient_l2'] == 3
    assert result['combined']['1']['gradient_l2'] == 2
    assert result['combined']['1']['clip_factor'] == pytest.approx(1 / (2 + 1e-6))


def test_zero_gradient_cosine_is_not_presented_as_alignment():
    result = pair_statistics(np.zeros(3), np.ones(3))
    assert result['mse_ce_cosine'] is None
    assert result['norm_below_interpretation_threshold']['mse']


def test_cancellation_catches_opposing_groups():
    result = cancellation([np.asarray([1., 0.]), np.asarray([-1., 0.]), np.asarray([2., 0.]), np.asarray([-2., 0.])])
    assert result['mean_gradient_l2'] == 0
    assert result['mean_norm_over_mean_group_norm'] == 0
    assert result['group_to_mean_cosines'] == [None] * 4


def test_group_mean_is_equal_weighted_and_module_decomposition_consistent():
    ranges = {'backbone': [(0, 1)], 'scorer': [(1, 2)], 'efficiency': [(2, 3)]}
    groups = [{'mse_gradient': np.asarray([i, 2 * i, 3 * i], dtype=float),
        'ce_gradient': np.asarray([-i, 0., i]), 'mse': float(i), 'ce': float(i * 2)} for i in range(1, 5)]
    result = describe_groups(groups, ranges)
    assert result['mean_mse'] == 2.5
    assert result['mean_ce'] == 5
    assert result['mean_of_four_group_gradients']['parameter_blocks']['backbone']['combined']['1']['gradient_l2'] == 0
    assert result['cancellation']['mse']['mean_norm_over_mean_group_norm'] == pytest.approx(1)


def snapshot(scale=1):
    return {'row_order_sha256': 'same', 'optional_rows': 10, 'weights_sha256_before': 'weights',
        'weights_unchanged': True, 'mse_gradient': np.asarray([1., 2.]) * scale,
        'ce_gradient': np.asarray([.5, 1.]) * scale, 'mse': 2. * scale, 'ce': 1. * scale}


def test_backend_comparison_accepts_small_error_but_rejects_material_gradient_change():
    assert compare_backends(snapshot(), snapshot(1.0001))['passed']
    assert not compare_backends(snapshot(), snapshot(1.02))['passed']


def test_backend_comparison_refuses_changed_inputs():
    other = snapshot()
    other['row_order_sha256'] = 'different'
    with pytest.raises(ValueError, match='rows'):
        compare_backends(snapshot(), other)
