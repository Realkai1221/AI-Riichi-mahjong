"""Saved-data objective/gradient decomposition without optimizer construction.

The MSE and chosen-action cross-entropy gradients are separately accumulated as
mean optional explorer row within match, then a fixed mean of sixteen matches.
Empty matches retain their denominator. Actor inputs are the frozen audited
Stage 3G tensors; outcomes and hand potentials enter only the scalar losses.

Four disjoint groups describe this fixed 64-match sample. Cancellation and
alignment are descriptive mechanism measurements, not population SNR estimates,
causal explanations, training proposals selected by significance, or strength.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .model import _state_hash
from .stage3g_data import LearningDataset, HandTargets

ORDER_SEED = 20261023012
BATCH_ROWS = 256
MATCHES_PER_GROUP = 16
CLIP_NORM = 1.
EPSILON = 1e-12
BLOCK_NAMES = ('backbone', 'scorer', 'efficiency')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def synchronize(device: str) -> None:
    if device == 'mps':
        torch.mps.synchronize()
    elif device.startswith('cuda'):
        torch.cuda.synchronize(device)


def check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError('Stage 3H gradient diagnostic deadline; no optimization performed')


def parameter_layout(model) -> tuple[list[tuple[str, torch.nn.Parameter]], dict[str, list[tuple[int, int]]], dict]:
    require(all(not p.requires_grad for p in model.rank_head.parameters()), 'Archived rank head must remain frozen')
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    require(bool(named) and all(not name.startswith('rank_head.') for name, _ in named), 'No trainable actor parameters')
    ranges = {name: [] for name in BLOCK_NAMES}
    members = {name: [] for name in BLOCK_NAMES}
    ordered_parameters = []
    start = 0
    for name, parameter in named:
        block = 'scorer' if name.startswith('action_score.') else 'efficiency' if name.startswith('efficiency_encoder.') else 'backbone'
        ranges[block].append((start, start + parameter.numel()))
        members[block].append({'name': name, 'shape': list(parameter.shape), 'parameters': parameter.numel()})
        ordered_parameters.append({'name': name, 'shape': list(parameter.shape), 'start': start,
                                   'stop': start + parameter.numel(), 'block': block})
        start += parameter.numel()
    metadata = {'trainable_parameters': start, 'frozen_rank_parameters': sum(p.numel() for p in model.rank_head.parameters()),
        'ordered_parameters': ordered_parameters,
        'blocks': members, 'definitions': {'scorer': 'Only action_score, the scalar candidate scorer reset at Stage 3G initialization',
            'efficiency': 'Only efficiency_encoder, the additive public efficiency feature branch',
            'backbone': 'All other trainable actor modules, including action encoders and shared embeddings'}}
    return named, ranges, metadata


def _norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector.astype(np.float64, copy=False)))


def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    a, b = _norm(left), _norm(right)
    if a <= EPSILON or b <= EPSILON:
        return None
    return float(np.clip(np.dot(left.astype(np.float64), right.astype(np.float64)) / (a * b), -1., 1.))


def pair_statistics(mse: np.ndarray, ce: np.ndarray) -> dict[str, Any]:
    require(mse.shape == ce.shape and np.isfinite(mse).all() and np.isfinite(ce).all(), 'Invalid gradient vectors')
    combined = {}
    for alpha in (0, 1):
        length = _norm(mse + alpha * ce)
        combined[str(alpha)] = {'alpha': alpha, 'gradient_l2': length,
            'clip_factor': min(1., CLIP_NORM / (length + 1e-6)),
            'clipped': length > CLIP_NORM}
    return {'mse_gradient_l2': _norm(mse), 'ce_gradient_l2': _norm(ce),
        'mse_ce_cosine': cosine(mse, ce), 'mse_ce_dot': float(np.dot(mse.astype(np.float64), ce.astype(np.float64))),
        'combined': combined, 'norm_below_interpretation_threshold': {'mse': _norm(mse) <= EPSILON, 'ce': _norm(ce) <= EPSILON}}


def slice_block(vector: np.ndarray, ranges: list[tuple[int, int]]) -> np.ndarray:
    return np.concatenate([vector[a:b] for a, b in ranges]) if ranges else np.empty(0, dtype=np.float64)


def describe_pair(mse: np.ndarray, ce: np.ndarray, ranges: dict[str, list[tuple[int, int]]]) -> dict:
    return {'all_actor_parameters': pair_statistics(mse, ce), 'parameter_blocks': {
        name: pair_statistics(slice_block(mse, ranges[name]), slice_block(ce, ranges[name])) for name in BLOCK_NAMES}}


def cancellation(vectors: list[np.ndarray]) -> dict:
    require(len(vectors) == 4 and len({vector.shape for vector in vectors}) == 1, 'Cancellation requires four comparable fixed groups')
    mean = np.mean(np.stack(vectors).astype(np.float64), axis=0)
    lengths = [_norm(vector) for vector in vectors]
    average = float(np.mean(lengths))
    return {'mean_gradient_l2': _norm(mean), 'mean_group_gradient_l2': average,
        'mean_norm_over_mean_group_norm': _norm(mean) / average if average > EPSILON else None,
        'group_to_mean_cosines': [cosine(vector, mean) for vector in vectors],
        'opposing_groups': sum(float(np.dot(vector, mean)) < 0 for vector in vectors),
        'pairwise_group_cosines': [{'left_group': a, 'right_group': b, 'cosine': cosine(vectors[a], vectors[b])}
                                  for a in range(4) for b in range(a + 1, 4)]}


def describe_groups(groups: list[dict], ranges: dict[str, list[tuple[int, int]]]) -> dict:
    require(len(groups) == 4, 'Exactly four fixed gradient groups required')
    mse = [group['mse_gradient'] for group in groups]
    ce = [group['ce_gradient'] for group in groups]
    result = {'mean_of_four_group_gradients': describe_pair(np.mean(mse, axis=0), np.mean(ce, axis=0), ranges),
        'cancellation': {'mse': cancellation(mse), 'ce': cancellation(ce),
            'mse_plus_ce': cancellation([a + b for a, b in zip(mse, ce)])},
        'parameter_block_cancellation': {},
        'mean_mse': float(np.mean([group['mse'] for group in groups])),
        'mean_ce': float(np.mean([group['ce'] for group in groups])),
        'weighting': 'Equal mean of four disjoint sixteen-match means; equivalent to the fixed64-match objective'}
    for name in BLOCK_NAMES:
        m = [slice_block(vector, ranges[name]) for vector in mse]
        c = [slice_block(vector, ranges[name]) for vector in ce]
        result['parameter_block_cancellation'][name] = {'mse': cancellation(m), 'ce': cancellation(c),
            'mse_plus_ce': cancellation([a + b for a, b in zip(m, c)])}
    return result


def gradient_group(model, dataset, match_ids: list[str], *, target: str, device: str,
                   deadline: float, hand_targets=None) -> dict:
    require(target in ('terminal_utility', 'one_hand_delta'), 'Unknown diagnostic objective target')
    require((target == 'one_hand_delta') == (hand_targets is not None), 'Diagnostic target and hand mapping disagree')
    require(len(match_ids) == 16 and len(set(match_ids)) == 16, 'Sixteen distinct complete matches required')
    require(dataset.split == 'training' and set(match_ids) <= set(dataset.match_ids), 'Gradient groups must use selected training matches')
    named, ranges, layout = parameter_layout(model)
    parameters = [parameter for _, parameter in named]
    require(all(parameter.grad is None for parameter in model.parameters()), 'Diagnostic model already has accumulated gradients')
    model.eval()
    before = _state_hash(model.state_dict())
    rank_before = _state_hash(model.rank_head.state_dict())
    mse_accumulator = [torch.zeros_like(parameter, memory_format=torch.preserve_format) for parameter in parameters]
    ce_accumulator = [torch.zeros_like(parameter, memory_format=torch.preserve_format) for parameter in parameters]
    rows = explorer_rows = empty = batches = 0
    mse_value = ce_value = 0.
    row_hash = hashlib.sha256()
    started = time.monotonic()
    maximum = 0.
    for name in match_ids:
        check_deadline(deadline)
        match = dataset.load_explorer(name, epoch=0, seed=ORDER_SEED)
        rows += match.optional_rows
        explorer_rows += match.explorer_rows
        row_hash.update(name.encode())
        row_hash.update(match.indices.tobytes())
        if not match.optional_rows:
            empty += 1
        if hand_targets is not None:
            entry = hand_targets.payload['matches'][name]
            explorer = match.targets['behavior_id'] == 3
            require(np.all(match.targets['seat'][explorer] == entry['explorer_seat'])
                and {str(int(h)) for h in match.targets['hand_id'][explorer]} <= set(entry['hand_deltas']),
                'Wrong-seat or missing explicit hand target')
        for batch in dataset.batches(match, BATCH_ROWS):
            check_deadline(deadline)
            actor = batch.torch_actor(device)
            require(set(actor) <= {'tiles', 'globals', 'history', 'history_mask', 'action_tokens', 'candidate_mask', 'candidate_efficiency'},
                'Unexpected actor tensor channel')
            selected = torch.from_numpy(batch.targets['chosen_index'].astype(np.int64)).clone().to(device)
            labels = batch.targets['final_utility'].astype(np.float32) if hand_targets is None else hand_targets.for_batch(batch)
            labels = torch.from_numpy(labels.copy()).to(device)
            values, rank_output = model(actor)
            legal = values[actor['candidate_mask']]
            require(bool(torch.isfinite(legal).all().detach().cpu()), 'Nonfinite legal diagnostic values')
            maximum = max(maximum, float(legal.detach().abs().max().cpu()))
            require(maximum <= 100, 'Diagnostic values exceed the archived safety bound')
            chosen = values.gather(1, selected[:, None]).squeeze(1)
            denominator = match.optional_rows * 16
            mse = (chosen - labels).square().sum() / denominator
            ce = (torch.logsumexp(values, dim=-1) - chosen).sum() / denominator
            require(bool(torch.isfinite(mse).detach().cpu()) and bool(torch.isfinite(ce).detach().cpu()), 'Nonfinite diagnostic objective')
            mse_gradients = torch.autograd.grad(mse, parameters, retain_graph=True, allow_unused=True)
            ce_gradients = torch.autograd.grad(ce, parameters, allow_unused=True)
            with torch.no_grad():
                for accumulator, gradient in zip(mse_accumulator, mse_gradients):
                    if gradient is not None:
                        accumulator.add_(gradient.detach())
                for accumulator, gradient in zip(ce_accumulator, ce_gradients):
                    if gradient is not None:
                        accumulator.add_(gradient.detach())
            mse_value += float(mse.detach().cpu())
            ce_value += float(ce.detach().cpu())
            batches += 1
            del actor, selected, labels, values, rank_output, legal, chosen, mse, ce, mse_gradients, ce_gradients, batch
        del match
        dataset.loaded_match = None
    synchronize(device)
    flat_mse = np.concatenate([value.detach().cpu().reshape(-1).numpy().astype(np.float64) for value in mse_accumulator])
    flat_ce = np.concatenate([value.detach().cpu().reshape(-1).numpy().astype(np.float64) for value in ce_accumulator])
    require(np.isfinite(flat_mse).all() and np.isfinite(flat_ce).all(), 'Nonfinite diagnostic gradient')
    after = _state_hash(model.state_dict())
    require(before == after and rank_before == _state_hash(model.rank_head.state_dict()), 'Read-only diagnostic mutated parameters')
    require(all(parameter.grad is None for parameter in model.parameters()), 'autograd.grad unexpectedly accumulated parameter gradients')
    return {'matches': match_ids, 'optional_rows': rows, 'explorer_rows': explorer_rows, 'empty_matches': empty,
        'batches': batches, 'mse': mse_value, 'ce': ce_value, 'maximum_abs_legal_value': maximum,
        'row_order_sha256': row_hash.hexdigest(), 'weights_sha256_before': before, 'weights_sha256_after': after,
        'rank_weights_sha256': rank_before, 'layout': layout, 'ranges': ranges,
        'gradient_summary': describe_pair(flat_mse, flat_ce, ranges),
        'mse_gradient': flat_mse, 'ce_gradient': flat_ce, 'elapsed_seconds': time.monotonic() - started,
        'working_array_bytes': dataset.peak_working_array_bytes, 'device': device,
        'optimizer_steps': 0, 'weights_unchanged': True}


def compare_backends(cpu: dict, device: dict) -> dict:
    require(cpu['row_order_sha256'] == device['row_order_sha256'] and cpu['optional_rows'] == device['optional_rows'], 'Backend parity rows differ')
    require(cpu['weights_sha256_before'] == device['weights_sha256_before'], 'Backend parity starting weights differ')
    entries = {}
    for label in ('mse', 'ce'):
        left, right = cpu[f'{label}_gradient'], device[f'{label}_gradient']
        require(left.shape == right.shape, 'Backend gradient layout differs')
        relative = _norm(right - left) / max(_norm(left), EPSILON)
        difference = abs(device[label] - cpu[label])
        tolerance = .001 * max(abs(cpu[label]), 1.)
        entries[label] = {'gradient_relative_l2': relative, 'gradient_max_abs_difference': float(np.max(np.abs(right - left))),
            'objective_absolute_difference': difference, 'objective_tolerance': tolerance,
            'passed': relative <= .01 and difference <= tolerance}
    return {'passed': all(entry['passed'] for entry in entries.values()), 'objectives': entries,
        'tolerances': {'gradient_relative_l2': .01, 'objective_scaled_absolute': .001},
        'row_order_exact': True, 'starting_weights_exact': True,
        'cpu_weights_unchanged': cpu['weights_unchanged'], 'device_weights_unchanged': device['weights_unchanged']}


def public_group(result: dict) -> dict:
    return {key: value for key, value in result.items() if key not in ('mse_gradient', 'ce_gradient', 'ranges', 'layout')}
