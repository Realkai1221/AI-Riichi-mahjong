"""Controlled offline action-value learning with exact complete-match weighting.

No PPO update is implemented. Labels are either recorded terminal rank utility
or a separately calibrated one-hand placement-potential difference. Actor input
is restricted to the audited visible-state tensors exposed by LearningDataset.
"""
from __future__ import annotations
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from .model import PolicyNetwork, _state_hash
from .stage3g_data import LearningDataset, HandTargets

CHECKPOINT_SCHEMA = 'riichi-ai-stage3g-value-checkpoint-v1'
INIT_SEED = 20261023011
ORDER_SEED = 20261023012
ARMS = {'mc_a0': ('terminal_utility', 0.), 'mc_a1': ('terminal_utility', 1.),
        'delta_a0': ('one_hand_delta', 0.), 'delta_a1': ('one_hand_delta', 1.)}
RECIPE = {'epochs': 2, 'matches_per_update': 16, 'batch_rows': 256,
          'learning_rate': 1e-4, 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 0.,
          'gradient_clip_l2': 1., 'initialization_seed': INIT_SEED, 'order_seed': ORDER_SEED,
          'max_abs_legal_value': 100., 'max_match_loss': 10000.,
          'weighting': 'mean eligible row within match, then fixed denominator 16 matches'}


def synchronize(device: str) -> None:
    if device == 'mps':
        torch.mps.synchronize()
    elif device.startswith('cuda'):
        torch.cuda.synchronize(device)


def build_value_model(reference: str | Path, sha: str, seed: int = INIT_SEED) -> PolicyNetwork:
    payload = Path(reference).read_bytes()
    if hashlib.sha256(payload).hexdigest() != sha:
        raise ValueError('Reference checkpoint SHA-256 mismatch')
    saved = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = PolicyNetwork(saved['model_config'])
        model.load_state_dict(saved['model_state_dict'], strict=True)
        for module in model.action_score.modules():
            if isinstance(module, torch.nn.Linear):
                module.reset_parameters()
    for parameter in model.rank_head.parameters():
        parameter.requires_grad_(False)
    return model


def _rank_hash(model: PolicyNetwork) -> str:
    return _state_hash(model.rank_head.state_dict())


def _finite_state(model: PolicyNetwork) -> bool:
    return all(bool(torch.isfinite(t).all().cpu()) for t in model.state_dict().values())


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu(v) for v in value)
    return value


def _optimizer(model):
    return torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
        lr=RECIPE['learning_rate'], betas=tuple(RECIPE['betas']), eps=RECIPE['eps'],
        weight_decay=RECIPE['weight_decay'])


def _identity(dataset: LearningDataset, reference_sha: str, arm: str, hand_targets: HandTargets | None,
              purpose: str) -> dict:
    return {'manifest_sha256': dataset.manifest_sha256, 'split': dataset.split,
        'reference_sha256': reference_sha, 'arm': arm, 'recipe': RECIPE,
        'hand_targets_sha256': hand_targets.sha256 if hand_targets is not None else None,
        'purpose': purpose}


def _rng_state(device: str) -> dict:
    result = {'torch_cpu': torch.get_rng_state().cpu(), 'python': random.getstate(),
              'orders': 'Stateless per-epoch/per-match NumPy PCG64 generators keyed by recipe and cursor'}
    if device == 'mps':
        result['torch_device'] = torch.mps.get_rng_state().cpu()
    elif device.startswith('cuda'):
        result['torch_device'] = torch.cuda.get_rng_state(device).cpu()
    return result


def _restore_rng(state: dict, device: str) -> None:
    torch.set_rng_state(state['torch_cpu'])
    random.setstate(state['python'])
    if 'torch_device' in state:
        if device == 'mps':
            torch.mps.set_rng_state(state['torch_device'])
        elif device.startswith('cuda'):
            torch.cuda.set_rng_state(state['torch_device'], device)


def save_checkpoint(path: str | Path, model: PolicyNetwork, optimizer, *, identity: dict,
                    cursor: dict, device: str, initial_hash: str, initial_rank_hash: str,
                    scheduled_updates: int, optimizer_updates: int, complete: bool,
                    statistics: dict | None = None) -> dict:
    path = Path(path)
    if path.exists():
        raise FileExistsError('Checkpoints are immutable')
    if _rank_hash(model) != initial_rank_hash or not _finite_state(model):
        raise ValueError('Rank head changed or checkpoint has nonfinite weights')
    model_state = _cpu(model.state_dict())
    metadata = {'schema': CHECKPOINT_SCHEMA, 'identity': identity, 'cursor': cursor,
        'device': device, 'initial_weights_sha256': initial_hash,
        'weights_sha256': _state_hash(model_state), 'frozen_rank_sha256': initial_rank_hash,
        'scheduled_updates': scheduled_updates, 'optimizer_updates': optimizer_updates,
        'complete': complete, 'policy_promotion': False,
        'statistics': statistics or {}}
    payload = {'metadata': metadata, 'model_config': model.config, 'model_state_dict': model_state,
        'optimizer_state_dict': _cpu(optimizer.state_dict()), 'rng_state': _rng_state(device)}
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    raw = buffer.getvalue()
    with path.open('xb') as handle:
        handle.write(raw)
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest(),
            'bytes': len(raw), 'weights_sha256': metadata['weights_sha256']}


def _read_checkpoint(path: str | Path):
    raw = Path(path).read_bytes()
    saved = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    metadata = saved.get('metadata', {})
    if metadata.get('schema') != CHECKPOINT_SCHEMA or metadata.get('weights_sha256') != _state_hash(saved['model_state_dict']):
        raise ValueError('Invalid Stage 3G checkpoint or weight hash')
    model = PolicyNetwork(saved['model_config'])
    model.load_state_dict(saved['model_state_dict'], strict=True)
    for parameter in model.rank_head.parameters():
        parameter.requires_grad_(False)
    if _rank_hash(model) != metadata['frozen_rank_sha256'] or not _finite_state(model):
        raise ValueError('Checkpoint rank identity or finiteness failed')
    return model, saved, hashlib.sha256(raw).hexdigest()


def load_value_checkpoint(path: str | Path, device: str = 'cpu') -> tuple[PolicyNetwork, dict]:
    """Load a fully completed formal arm for value-greedy development evaluation."""
    model, saved, digest = _read_checkpoint(path)
    metadata = saved['metadata']
    if metadata.get('complete') is not True or metadata['identity'].get('purpose') != 'formal-training':
        raise ValueError('Evaluation requires a completed formal-training checkpoint')
    return model.to(device).eval(), {**metadata, 'checkpoint_sha256': digest}


def resume_checkpoint(path: str | Path, *, expected_identity: dict, device: str):
    model, saved, _ = _read_checkpoint(path)
    metadata = saved['metadata']
    if metadata['identity'] != expected_identity or metadata['device'] != device:
        raise ValueError('Resume identity, recipe, split or backend changed')
    model = model.to(device).train()
    optimizer = _optimizer(model)
    optimizer.load_state_dict(saved['optimizer_state_dict'])
    _restore_rng(saved['rng_state'], device)
    return model, optimizer, metadata


def value_losses(values: torch.Tensor, chosen: torch.Tensor, target: torch.Tensor,
                 alpha: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = values.gather(1, chosen[:, None]).squeeze(1)
    squared = (selected - target).square()
    conservative = torch.logsumexp(values, dim=-1) - selected
    return squared + alpha * conservative, squared, conservative


def update_group(model: PolicyNetwork, optimizer, dataset: LearningDataset, match_ids: tuple[str, ...],
                 *, epoch: int, arm: str, device: str, deadline: float,
                 hand_targets: HandTargets | None = None, capture: bool = False) -> tuple[dict, dict | None]:
    """Accumulate sixteen whole-match means, including zero-loss empty matches."""
    if len(match_ids) != RECIPE['matches_per_update']:
        raise ValueError('An update must contain exactly sixteen complete matches')
    if len(set(match_ids)) != len(match_ids):
        raise ValueError('An update group contains duplicate matches')
    target_kind, alpha = ARMS[arm]
    if (target_kind == 'one_hand_delta') != (hand_targets is not None):
        raise ValueError('Arm and immutable hand-target mapping disagree')
    optimizer.zero_grad(set_to_none=True)
    stats = {'matches': 16, 'optional_rows': 0, 'explorer_rows': 0, 'all_seat_rows': 0,
             'empty_matches': 0, 'objective': 0., 'mse': 0., 'conservative': 0.,
             'maximum_abs_legal_value': 0., 'maximum_row_loss': 0., 'match_ids': list(match_ids)}
    values_capture = []
    order_hash = hashlib.sha256()
    tick = time.perf_counter()
    for name in match_ids:
        if time.perf_counter() >= deadline:
            raise TimeoutError('Learning group deadline before loading next match')
        match = dataset.load_explorer(name, epoch=epoch, seed=ORDER_SEED)
        order_hash.update(name.encode())
        order_hash.update(match.indices.tobytes())
        stats['optional_rows'] += match.optional_rows
        stats['explorer_rows'] += match.explorer_rows
        stats['all_seat_rows'] += match.total_rows
        if hand_targets is not None:
            entry = hand_targets.payload['matches'][name]
            explorer = match.targets['behavior_id'] == 3
            if (np.any(match.targets['seat'][explorer] != entry['explorer_seat'])
                    or not {str(int(h)) for h in match.targets['hand_id'][explorer]} <= set(entry['hand_deltas'])):
                raise ValueError('Incomplete or wrong-seat hand target coverage')
        if not match.optional_rows:
            stats['empty_matches'] += 1
        for batch in dataset.batches(match, RECIPE['batch_rows']):
            if time.perf_counter() >= deadline:
                raise TimeoutError('Learning deadline inside complete match')
            actor = batch.torch_actor(device)
            selected = torch.from_numpy(batch.targets['chosen_index'].astype(np.int64)).clone().to(device)
            labels = (batch.targets['final_utility'].astype(np.float32) if hand_targets is None
                      else hand_targets.for_batch(batch))
            target = torch.from_numpy(labels.copy()).to(device)
            values, _ = model(actor)
            legal = values[actor['candidate_mask']]
            if not bool(torch.isfinite(legal).all().detach().cpu()):
                raise FloatingPointError('Nonfinite legal action value')
            maximum = float(legal.detach().abs().max().cpu())
            if maximum > RECIPE['max_abs_legal_value']:
                raise FloatingPointError('Legal action value exceeded fixed safety bound')
            losses, mse, conservative = value_losses(values, selected, target, alpha)
            maximum_loss = float(losses.detach().max().cpu())
            if not bool(torch.isfinite(losses).all().detach().cpu()) or maximum_loss > RECIPE['max_match_loss']:
                raise FloatingPointError('Objective exceeded fixed finite safety bound')
            denominator = match.optional_rows * RECIPE['matches_per_update']
            (losses.sum() / denominator).backward()
            stats['objective'] += float(losses.detach().sum().cpu()) / denominator
            stats['mse'] += float(mse.detach().sum().cpu()) / denominator
            stats['conservative'] += float(conservative.detach().sum().cpu()) / denominator
            stats['maximum_abs_legal_value'] = max(stats['maximum_abs_legal_value'], maximum)
            stats['maximum_row_loss'] = max(stats['maximum_row_loss'], maximum_loss)
            if capture:
                values_capture.append(legal.detach().cpu().numpy().copy())
            del actor, values, legal, losses, mse, conservative, batch
        del match
        dataset.loaded_match = None
    parameters = [p for p in model.parameters() if p.requires_grad]
    gradient = torch.cat([p.grad.detach().reshape(-1).cpu() if p.grad is not None
                          else torch.zeros(p.numel()) for p in parameters])
    if not bool(torch.isfinite(gradient).all()):
        raise FloatingPointError('Nonfinite accumulated gradient')
    gradient_l2 = float(torch.linalg.vector_norm(gradient))
    stats['gradient_l2_before_clip'] = gradient_l2
    stats['clipped'] = gradient_l2 > RECIPE['gradient_clip_l2']
    stats['optimizer_step'] = stats['optional_rows'] > 0
    if stats['optimizer_step']:
        torch.nn.utils.clip_grad_norm_(parameters, RECIPE['gradient_clip_l2'], error_if_nonfinite=True)
        optimizer.step()
    synchronize(device)
    if not _finite_state(model):
        raise FloatingPointError('Optimizer produced nonfinite parameters')
    stats['row_order_sha256'] = order_hash.hexdigest()
    stats['seconds'] = time.perf_counter() - tick
    stats['working_array_bytes'] = dataset.peak_working_array_bytes
    snapshot = None if not capture else {'gradients': gradient.numpy().copy(),
        'values': np.concatenate(values_capture) if values_capture else np.empty(0, dtype=np.float32),
        'objective': stats['objective']}
    return stats, snapshot


def _movement(initial: dict, model: PolicyNetwork) -> dict:
    squared = maximum = 0.
    changed = 0
    for name, value in model.state_dict().items():
        difference = value.detach().cpu().double() - initial[name].double()
        squared += float(difference.square().sum())
        maximum = max(maximum, float(difference.abs().max()))
        changed += int(bool(torch.any(difference != 0)))
    return {'parameter_l2': math.sqrt(squared), 'parameter_max_abs': maximum, 'changed_tensors': changed}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def train(corpus: str | Path, reference: str | Path, reference_sha: str, output: str | Path,
          *, arm: str, device: str = 'mps', max_seconds: float = 1200,
          hand_targets_path: str | Path | None = None, resume: str | Path | None = None) -> dict:
    if arm not in ARMS or max_seconds <= 0:
        raise ValueError('Unknown arm or nonpositive time allowance')
    started = time.perf_counter()
    deadline = started + max_seconds
    dataset = LearningDataset(corpus, 'training')
    if len(dataset.match_ids) != 3200 or len(dataset.manifest['splits']['development']) != 800:
        raise ValueError('Formal Stage 3G requires the frozen 3200/800 complete-match split')
    hand_targets = HandTargets(hand_targets_path, dataset) if hand_targets_path else None
    if (ARMS[arm][0] == 'one_hand_delta') != (hand_targets is not None):
        raise ValueError('Training arm and immutable target mapping disagree')
    identity = _identity(dataset, reference_sha, arm, hand_targets, 'formal-training')
    output = Path(output)
    if output.exists():
        raise FileExistsError('A learning arm output directory is immutable')
    output.mkdir(parents=True)
    initial_model = build_value_model(reference, reference_sha)
    initial = _cpu(initial_model.state_dict())
    initial_hash, rank_hash = _state_hash(initial), _rank_hash(initial_model)
    model, optimizer = initial_model.to(device).train(), None
    optimizer = _optimizer(model)
    scheduled = updates = 0
    cursor = {'epoch': 0, 'next_match_position': 0}
    aggregate = {'optional_row_presentations': 0, 'all_seat_rows_loaded': 0,
        'explorer_rows_loaded': 0, 'empty_match_presentations': 0, 'clipped_updates': 0,
        'match_presentations': 0}
    if resume:
        model, optimizer, metadata = resume_checkpoint(resume, expected_identity=identity, device=device)
        if metadata['initial_weights_sha256'] != initial_hash or metadata['frozen_rank_sha256'] != rank_hash:
            raise ValueError('Resume initialization identity changed')
        cursor = metadata['cursor']
        scheduled, updates = metadata['scheduled_updates'], metadata['optimizer_updates']
        aggregate = metadata['statistics']
    if (set(cursor) != {'epoch', 'next_match_position'} or type(cursor['epoch']) is not int
            or type(cursor['next_match_position']) is not int or not 0 <= cursor['epoch'] <= 2
            or not 0 <= cursor['next_match_position'] < 3200 or cursor['next_match_position'] % 16
            or (cursor['epoch'] == 2 and cursor['next_match_position'] != 0)
            or scheduled != cursor['epoch'] * 200 + cursor['next_match_position'] // 16):
        raise ValueError('Resume cursor is not an exact saved complete-group boundary')
    checkpoint_list = [save_checkpoint(output / 'initial.pt', model, optimizer,
        identity=identity, cursor=cursor, device=device, initial_hash=initial_hash,
        initial_rank_hash=rank_hash, scheduled_updates=scheduled, optimizer_updates=updates,
        complete=False, statistics=aggregate)]
    losses = []
    progress = output / 'updates.jsonl'
    with progress.open('x') as log:
        for epoch in range(cursor['epoch'], 2):
            order = dataset.match_order(epoch, ORDER_SEED)
            start = cursor['next_match_position'] if epoch == cursor['epoch'] else 0
            for position in range(start, len(order), 16):
                stats, _ = update_group(model, optimizer, dataset, order[position:position + 16],
                    epoch=epoch, arm=arm, device=device, deadline=deadline, hand_targets=hand_targets)
                scheduled += 1
                updates += int(stats['optimizer_step'])
                aggregate['optional_row_presentations'] += stats['optional_rows']
                aggregate['all_seat_rows_loaded'] += stats['all_seat_rows']
                aggregate['explorer_rows_loaded'] += stats['explorer_rows']
                aggregate['empty_match_presentations'] += stats['empty_matches']
                aggregate['clipped_updates'] += int(stats['clipped'])
                aggregate['match_presentations'] += 16
                cursor = ({'epoch': epoch + 1, 'next_match_position': 0} if position + 16 == len(order)
                          else {'epoch': epoch, 'next_match_position': position + 16})
                record = {**stats, 'epoch': epoch, 'scheduled_update': scheduled,
                          'optimizer_updates': updates, 'cursor': cursor}
                log.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
                log.flush()
                losses.append(stats['objective'])
                if _rank_hash(model) != rank_hash:
                    raise AssertionError('Frozen rank head changed during optimization')
                if scheduled % 100 == 0 and scheduled < 400:
                    checkpoint_list.append(save_checkpoint(output / f'update-{scheduled:04d}.pt', model, optimizer,
                        identity=identity, cursor=cursor, device=device, initial_hash=initial_hash,
                        initial_rank_hash=rank_hash, scheduled_updates=scheduled, optimizer_updates=updates,
                        complete=False, statistics=aggregate))
    checkpoint_list.append(save_checkpoint(output / 'final.pt', model, optimizer,
        identity=identity, cursor=cursor, device=device, initial_hash=initial_hash,
        initial_rank_hash=rank_hash, scheduled_updates=scheduled, optimizer_updates=updates,
        complete=True, statistics=aggregate))
    result = {'schema': 'riichi-ai-stage3g-learning-v1', 'complete': scheduled == 400,
        'identity': identity, 'device': device, 'scheduled_updates': scheduled,
        'optimizer_updates': updates, 'statistics': aggregate, 'checkpoints': checkpoint_list,
        'initial_weights_sha256': initial_hash, 'final_weights_sha256': _state_hash(model.state_dict()),
        'rank_head_unchanged': _rank_hash(model) == rank_hash, 'parameter_movement': _movement(initial, model),
        'first_objective': losses[0] if losses else None, 'last_objective': losses[-1] if losses else None,
        'seconds': time.perf_counter() - started, 'working_array_bytes': dataset.peak_working_array_bytes,
        'numeric_bytes_read': dataset.bytes_read, 'policy_promotion': False,
        'limitations': 'Single initialization; offline explorer continuation and fixed opponent population; training loss is not playing strength.'}
    _write_json(output / 'result.json', result)
    return result


def _array_comparison(actual: np.ndarray, expected: np.ndarray, rtol: float, atol: float) -> dict:
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    return {'passed': bool(np.isfinite(actual).all() and np.allclose(actual, expected, rtol=rtol, atol=atol)),
        'max_abs': float(np.max(np.abs(difference))) if difference.size else 0.,
        'relative_l2': float(np.linalg.norm(difference) / max(np.linalg.norm(expected.astype(np.float64)), 1e-12)),
        'rtol': rtol, 'atol': atol}


def preflight(corpus: str | Path, reference: str | Path, reference_sha: str, output: str | Path,
              *, max_seconds: float = 180) -> dict:
    started = time.perf_counter()
    deadline = started + max_seconds
    dataset = LearningDataset(corpus, 'preflight')
    if len(dataset.match_ids) != 16:
        raise ValueError('Optimizer preflight requires exactly sixteen fresh matches')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preflight output directory already exists')
    output.mkdir(parents=True)
    initial_model = build_value_model(reference, reference_sha)
    initial = _cpu(initial_model.state_dict())
    initial_hash, rank_hash = _state_hash(initial), _rank_hash(initial_model)
    cpu_snapshot = cpu_final = None
    device_results = []
    for device in ('cpu', 'mps'):
        if device == 'mps' and not torch.backends.mps.is_available():
            device_results.append({'device': device, 'available': False, 'passed': False})
            continue
        measured = LearningDataset(corpus, 'preflight')
        model = copy.deepcopy(initial_model).to(device).train()
        optimizer = _optimizer(model)
        identity = _identity(measured, reference_sha, 'mc_a1', None, 'transient-preflight')
        tick = time.perf_counter()
        first, snapshot = update_group(model, optimizer, measured, measured.match_order(0, ORDER_SEED),
            epoch=0, arm='mc_a1', device=device, deadline=deadline, capture=True)
        boundary = save_checkpoint(output / f'{device}-after-first.pt', model, optimizer,
            identity=identity, cursor={'epoch': 1, 'next_match_position': 0}, device=device,
            initial_hash=initial_hash, initial_rank_hash=rank_hash,
            scheduled_updates=1, optimizer_updates=int(first['optimizer_step']), complete=False)
        second, _ = update_group(model, optimizer, measured, measured.match_order(1, ORDER_SEED),
            epoch=1, arm='mc_a1', device=device, deadline=deadline)
        elapsed = time.perf_counter() - tick
        final_state = _cpu(model.state_dict())
        if device == 'cpu':
            cpu_snapshot, cpu_final = snapshot, final_state
        value_comparison = _array_comparison(snapshot['values'], cpu_snapshot['values'], .001, .001)
        gradient_comparison = _array_comparison(snapshot['gradients'], cpu_snapshot['gradients'], .01, .0001)
        loss_difference = abs(snapshot['objective'] - cpu_snapshot['objective'])
        gradient_ok = gradient_comparison['relative_l2'] <= .01
        loss_ok = loss_difference <= .001 * max(abs(cpu_snapshot['objective']), 1.)
        state_comparison = {name: _array_comparison(t.numpy(), cpu_final[name].numpy(), .0001, .00001)
                            for name, t in final_state.items()}
        # MPS versus CPU Adam weights are reported, while acceptance is defined
        # on forward/gradient/loss. Resume comparison below is the strict weight gate.
        resume_check = None
        if device == 'mps':
            resumed_model, resumed_optimizer, metadata = resume_checkpoint(boundary['path'],
                expected_identity=identity, device=device)
            resumed_dataset = LearningDataset(corpus, 'preflight')
            resumed_second, _ = update_group(resumed_model, resumed_optimizer, resumed_dataset,
                resumed_dataset.match_order(1, ORDER_SEED), epoch=1, arm='mc_a1', device=device,
                deadline=deadline)
            comparisons = {name: _array_comparison(t.detach().cpu().numpy(), final_state[name].numpy(),
                                                   .0001, .00001)
                           for name, t in resumed_model.state_dict().items()}
            optimizer_identical = _optimizer_state_equal(optimizer.state_dict(), resumed_optimizer.state_dict())
            optimizer_close = _optimizer_state_close(optimizer.state_dict(), resumed_optimizer.state_dict())
            resume_check = {'passed': all(c['passed'] for c in comparisons.values()) and optimizer_close,
                'weights_exact': all(c['max_abs'] == 0 for c in comparisons.values()),
                'maximum_weight_abs_difference': max(c['max_abs'] for c in comparisons.values()),
                'optimizer_exact': optimizer_identical, 'optimizer_allclose': optimizer_close,
                'row_order_exact': resumed_second['row_order_sha256'] == second['row_order_sha256'],
                'cursor': metadata['cursor'], 'weight_rtol': .0001, 'weight_atol': .00001}
            resume_check['passed'] = resume_check['passed'] and resume_check['row_order_exact']
        device_results.append({'device': device, 'available': True,
            'passed': value_comparison['passed'] and gradient_ok and loss_ok
                      and _rank_hash(model) == rank_hash and (resume_check is None or resume_check['passed']),
            'first_update': first, 'second_update': second, 'checkpoint': boundary,
            'values_vs_cpu': value_comparison, 'gradients_vs_cpu': gradient_comparison,
            'objective_abs_difference_vs_cpu': loss_difference, 'objective_check_passed': loss_ok,
            'two_update_weights_vs_cpu': {'allclose': all(c['passed'] for c in state_comparison.values()),
                'maximum_abs_difference': max(c['max_abs'] for c in state_comparison.values()),
                'note': 'Diagnostic; resume weight agreement is the declared strict optimizer-state gate.'},
            'resume': resume_check, 'seconds_including_loading_optimizer_and_boundary_save': elapsed,
            'optional_rows_per_second': (first['optional_rows'] + second['optional_rows']) / elapsed,
            'matches_per_second': 32 / elapsed, 'rank_head_unchanged': _rank_hash(model) == rank_hash,
            'parameter_movement': _movement(initial, model), 'working_array_bytes': measured.peak_working_array_bytes})
        del model, optimizer
        if device == 'mps':
            torch.mps.empty_cache()
    passed = all(item['passed'] for item in device_results)
    report = {'schema': 'riichi-ai-stage3g-optimizer-preflight-v1', 'complete': True, 'passed': passed,
        'manifest_sha256': dataset.manifest_sha256, 'initial_weights_sha256': initial_hash,
        'reference_sha256': reference_sha, 'recipe': RECIPE, 'devices': device_results,
        'seconds': time.perf_counter() - started, 'formal_training_updates': 0,
        'transient_updates': sum(2 + (1 if item.get('resume') else 0) for item in device_results if item['available']),
        'checkpoint_role': 'Transient preflight only; never eligible for candidate evaluation or promotion',
        'cuda': 'Portable code path; unavailable in this Apple hardware preflight'}
    _write_json(output / 'result.json', report)
    return report


def _optimizer_state_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_optimizer_state_equal(v, right[k]) for k, v in left.items())
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(_optimizer_state_equal(a, b) for a, b in zip(left, right))
    return left == right


def validate(corpus: str | Path, checkpoint: str | Path, output: str | Path, *,
             device: str = 'cpu', hand_targets_path: str | Path | None = None,
             max_seconds: float = 300) -> dict:
    """Descriptive matched-objective loss on the fixed development split."""
    started = time.perf_counter()
    deadline = started + max_seconds
    dataset = LearningDataset(corpus, 'development')
    model, metadata = load_value_checkpoint(checkpoint, device)
    identity = metadata['identity']
    if identity['manifest_sha256'] != dataset.manifest_sha256:
        raise ValueError('Checkpoint and validation corpus differ')
    arm = identity['arm']
    kind, alpha = ARMS[arm]
    mapping = HandTargets(hand_targets_path, dataset) if hand_targets_path else None
    if (kind == 'one_hand_delta') != (mapping is not None):
        raise ValueError('Validation target differs from arm')
    if identity['hand_targets_sha256'] != (mapping.sha256 if mapping else None):
        raise ValueError('Validation hand-target mapping differs from training')
    count = rows = empty = 0
    loss_sum = mse_sum = penalty_sum = 0.
    with torch.inference_mode():
        for name in dataset.match_order(0, ORDER_SEED):
            match = dataset.load_explorer(name, epoch=0, seed=ORDER_SEED)
            count += 1
            rows += match.optional_rows
            empty += int(match.optional_rows == 0)
            for batch in dataset.batches(match):
                if time.perf_counter() >= deadline:
                    raise TimeoutError('Development-loss validation deadline')
                actor = batch.torch_actor(device)
                selected = torch.from_numpy(batch.targets['chosen_index'].astype(np.int64)).clone().to(device)
                label = batch.targets['final_utility'].astype(np.float32) if mapping is None else mapping.for_batch(batch)
                target = torch.from_numpy(label.copy()).to(device)
                values, _ = model(actor)
                legal = values[actor['candidate_mask']]
                if not bool(torch.isfinite(legal).all().cpu()) or float(legal.abs().max().cpu()) > 100:
                    raise FloatingPointError('Invalid legal action value in validation')
                losses, mse, penalty = value_losses(values, selected, target, alpha)
                loss_sum += float(losses.sum().cpu()) / match.optional_rows
                mse_sum += float(mse.sum().cpu()) / match.optional_rows
                penalty_sum += float(penalty.sum().cpu()) / match.optional_rows
    unchanged = _state_hash(model.state_dict()) == metadata['weights_sha256']
    report = {'schema': 'riichi-ai-stage3g-validation-v1', 'complete': count == len(dataset.match_ids),
        'checkpoint_sha256': metadata['checkpoint_sha256'], 'manifest_sha256': dataset.manifest_sha256,
        'arm': arm, 'matches': count, 'optional_rows': rows, 'empty_matches': empty,
        'objective': loss_sum / count, 'mse': mse_sum / count, 'conservative': penalty_sum / count,
        'weights_unchanged': unchanged, 'seconds': time.perf_counter() - started,
        'limitations': 'Descriptive same-target loss only; losses across different target definitions cannot select playing policies.'}
    if not unchanged:
        raise AssertionError('Validation mutated model')
    _write_json(Path(output) / 'result.json', report)
    return report


def _optimizer_state_close(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.shape == right.shape
                and torch.allclose(left.detach().cpu(), right.detach().cpu(), rtol=.0001, atol=.00001))
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_optimizer_state_close(v, right[k]) for k, v in left.items())
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(_optimizer_state_close(a, b) for a, b in zip(left, right))
    return left == right
