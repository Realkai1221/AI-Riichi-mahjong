"""Frozen-feature supervised value and residual regression for Stage 3I.

The actor is never constructed by this module. Targets are auditor-side inputs
to losses only. Decision-weighted regression is deliberately different from the
previous complete-match-weighted objective; validation reports both weights.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .model import _state_hash
from .stage3i_model import ResidualHeads

CHECKPOINT_SCHEMA = 'riichi-ai-stage3i-training-checkpoint-v1'
BUNDLE_SCHEMA = 'riichi-ai-stage3i-head-bundle-v1'
CACHE_SCHEMA = 'riichi-ai-stage3i-feature-cache-v1'
RECIPE = {'learning_rate': 3e-4, 'betas': [.9, .999], 'eps': 1e-8,
          'weight_decay': 0., 'gradient_clip_l2': 1., 'batch_rows': 1024,
          'ridge': .01, 'target': 'one_hand_delta', 'behavior_id': 3,
          'training_weighting': 'mean eligible explorer decision per optimizer batch',
          'selection': 'minimum selection-partition row MSE; earliest epoch on exact tie',
          'residual_target': 'target minus selected frozen value prediction',
          'action_ridge': 'mean squared residual over every legal candidate, then mean decision',
          'state_ridge': 'mean squared state residual per decision',
          'order': 'independent PCG64 permutation by seed, head and epoch'}
HEAD_NAMES = ('value', 'action', 'state')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def write_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(payload)
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Stage 3I training deadline reached; no partial model is promoted')


def synchronize(device):
    if device == 'mps':
        torch.mps.synchronize()
    elif device.startswith('cuda'):
        torch.cuda.synchronize(device)


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return value


class FeatureCache:
    """Hash-checked packed visible features; outcomes cannot enter head inputs."""

    def __init__(self, manifest_path, partition, *, expected_sha256=None, deadline=math.inf):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest_sha256 = digest(self.manifest_path)
        require(expected_sha256 is None or expected_sha256 == self.manifest_sha256, 'Feature-cache manifest hash mismatch')
        manifest = json.loads(self.manifest_path.read_text())
        require(manifest.get('schema') == CACHE_SCHEMA, 'Unsupported feature-cache schema')
        require(partition in ('training', 'selection', 'calibration', 'gate', 'preflight'), 'Unknown cache partition')
        require(manifest.get('target') == 'one_hand_delta' and manifest.get('behavior_id') == 3,
                'Cache must contain explorer-only one-hand targets')
        partitions = manifest['partitions']
        require(partition in partitions and len(set(partitions[partition])) == len(partitions[partition]), 'Invalid partition membership')
        memberships = [name for names in partitions.values() for name in names]
        require(len(memberships) == len(set(memberships)), 'Cache match partitions overlap')
        by_name = {record['match_id']: record for record in manifest['shards']}
        require(len(by_name) == len(manifest['shards']) and set(memberships) == set(by_name), 'Cache membership and shards differ')
        self.partition, self.manifest = partition, manifest
        self.match_ids = tuple(partitions[partition])
        require(bool(self.match_ids), 'Cannot train or assess an empty partition')
        states, actions, offsets, choices, targets, owners, source_rows = [], [], [0], [], [], [], []
        row_count = candidate_count = 0
        self.shards = []
        for match_index, match_id in enumerate(self.match_ids):
            check_deadline(deadline)
            record = by_name[match_id]
            require(record['partition'] == partition, 'Shard partition mismatch')
            path = Path(record['path'])
            if not path.is_absolute():
                path = self.manifest_path.parent / path
            require(digest(path) == record['sha256'], 'Feature shard hash mismatch')
            with np.load(path, allow_pickle=False) as saved:
                a = {key: saved[key].copy() for key in ('state_features', 'action_features', 'candidate_offsets', 'chosen_index', 'target')}
                rows = saved['row_indices'].copy() if 'row_indices' in saved else np.arange(len(a['target']), dtype=np.int64)
            n, c = len(a['target']), len(a['action_features'])
            require(n == record['rows'] and c == record['candidates'], 'Shard declared counts mismatch')
            require(a['state_features'].shape == (n, 256) and a['action_features'].shape == (c, 128), 'Frozen feature dimensions changed')
            require(a['candidate_offsets'].shape == (n + 1,) and a['candidate_offsets'].dtype == np.int64
                    and a['candidate_offsets'][0] == 0 and a['candidate_offsets'][-1] == c, 'Invalid packed candidate offsets')
            counts = np.diff(a['candidate_offsets'])
            require(np.all(counts > 1) and np.all(counts <= 256), 'Only optional decisions with bounded legal candidates are allowed')
            require(a['chosen_index'].shape == (n,) and a['chosen_index'].dtype == np.int64
                    and np.all(a['chosen_index'] >= 0) and np.all(a['chosen_index'] < counts), 'Chosen candidate outside legal set')
            require(rows.shape == (n,) and np.issubdtype(rows.dtype, np.integer)
                    and np.all(rows >= 0) and (n < 2 or np.all(np.diff(rows) > 0)), 'Source rows must be unique and ascending')
            for name in ('state_features', 'action_features', 'target'):
                require(a[name].dtype == np.float32 and np.isfinite(a[name]).all(), 'Features/targets must be finite float32')
            require(a['target'].shape == (n,) and np.all(np.abs(a['target']) <= 3.00001), 'Invalid one-hand rank-utility difference')
            states.append(a['state_features']); actions.append(a['action_features'])
            offsets.extend((a['candidate_offsets'][1:] + candidate_count).tolist())
            choices.append(a['chosen_index']); targets.append(a['target'])
            owners.append(np.full(n, match_index, dtype=np.int32)); source_rows.append(rows)
            row_count += n; candidate_count += c
            self.shards.append({**record, 'path': str(path.resolve())})
        require(row_count > 0, 'Partition contains no eligible rows')
        self.state = np.concatenate(states)
        self.action = np.concatenate(actions)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.chosen = np.concatenate(choices)
        self.target = np.concatenate(targets)
        self.match_index = np.concatenate(owners)
        self.source_rows = np.concatenate(source_rows)
        self.rows, self.candidates = row_count, candidate_count
        self.array_bytes = sum(value.nbytes for value in (self.state, self.action, self.offsets,
            self.chosen, self.target, self.match_index, self.source_rows))

    def identity(self):
        return {'manifest_sha256': self.manifest_sha256, 'partition': self.partition,
                'matches': len(self.match_ids), 'rows': self.rows, 'candidates': self.candidates,
                'array_bytes': self.array_bytes}

    def batch(self, indices, device, *, all_actions=True):
        indices = np.asarray(indices, dtype=np.int64)
        require(indices.ndim == 1 and len(indices) > 0 and np.all(indices >= 0)
                and np.all(indices < self.rows), 'Invalid cache row indices')
        state = torch.from_numpy(self.state[indices].copy()).to(device)
        result = {'state': state, 'target': torch.from_numpy(self.target[indices].copy()).to(device)}
        if all_actions:
            counts = np.diff(self.offsets)[indices]
            owner = np.repeat(np.arange(len(indices), dtype=np.int64), counts)
            starts = np.repeat(self.offsets[indices], counts)
            local = np.arange(len(owner)) - np.repeat(np.r_[0, np.cumsum(counts)[:-1]], counts)
            candidate_indices = starts + local
            packed = torch.from_numpy(self.action[candidate_indices].copy()).to(device)
            owner_tensor = torch.from_numpy(owner).to(device)
            result.update(action=torch.cat((state[owner_tensor], packed), dim=-1), owner=owner_tensor,
                counts=torch.from_numpy(counts.astype(np.float32)).to(device),
                chosen=torch.from_numpy((np.r_[0, np.cumsum(counts)[:-1]] + self.chosen[indices]).astype(np.int64)).to(device))
        return result


def optimizer_for(head):
    return torch.optim.Adam(head.parameters(), lr=RECIPE['learning_rate'],
        betas=tuple(RECIPE['betas']), eps=RECIPE['eps'], weight_decay=0.)


def epoch_order(rows, seed, head, epoch):
    require(head in HEAD_NAMES and epoch >= 0, 'Invalid deterministic epoch identity')
    sequence = np.random.SeedSequence([int(seed), HEAD_NAMES.index(head), int(epoch), 20261022])
    return np.random.default_rng(sequence).permutation(rows)


def head_losses(head, name, batch, fixed_value=None):
    require(name in HEAD_NAMES, 'Unknown fitted head')
    require((name == 'value') == (fixed_value is None), 'Residual losses require the selected frozen value head')
    target = batch['target']
    if fixed_value is not None:
        require(not any(p.requires_grad for p in fixed_value.parameters()), 'Residual target value head is trainable')
        with torch.no_grad():
            target = target - fixed_value(batch['state'])
    if name == 'action':
        values = head(batch['action'])
        predictions = values[batch['chosen']]
        penalty = torch.zeros_like(target).scatter_add(0, batch['owner'], values.square()) / batch['counts']
    else:
        predictions = head(batch['state'])
        penalty = predictions.square() if name == 'state' else torch.zeros_like(target)
    squared = (predictions - target).square()
    return squared + RECIPE['ridge'] * penalty, squared, penalty, predictions, target


def update(head, optimizer, name, batch, fixed_value=None):
    optimizer.zero_grad(set_to_none=True)
    losses, squared, penalty, _, _ = head_losses(head, name, batch, fixed_value)
    require(bool(torch.isfinite(losses).all().detach().cpu()), 'Nonfinite training loss')
    loss = losses.mean()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(head.parameters(), RECIPE['gradient_clip_l2'], error_if_nonfinite=True)
    optimizer.step()
    require(all(bool(torch.isfinite(p).all().detach().cpu()) for p in head.parameters()), 'Nonfinite head weights')
    return {'rows': len(losses), 'loss_sum': float(losses.detach().sum().cpu()),
            'squared_sum': float(squared.detach().sum().cpu()), 'ridge_sum': float(penalty.detach().sum().cpu()),
            'gradient_l2': float(norm.detach().cpu()), 'clipped': bool(norm.detach().cpu() > 1)}


def evaluate(head, name, cache, device, deadline, fixed_value=None):
    head.eval()
    sums = np.zeros(len(cache.match_ids), dtype=np.float64)
    counts = np.zeros(len(cache.match_ids), dtype=np.int64)
    squared_sum = penalty_sum = target_sum = target_squared_sum = prediction_sum = 0.
    with torch.inference_mode():
        for start in range(0, cache.rows, RECIPE['batch_rows']):
            check_deadline(deadline)
            indices = np.arange(start, min(start + RECIPE['batch_rows'], cache.rows))
            batch = cache.batch(indices, device, all_actions=name == 'action')
            _, squared, penalty, predicted, targets = head_losses(head, name, batch, fixed_value)
            error = squared.cpu().numpy().astype(np.float64)
            require(np.isfinite(error).all(), 'Nonfinite selection error')
            np.add.at(sums, cache.match_index[indices], error)
            np.add.at(counts, cache.match_index[indices], 1)
            squared_sum += float(error.sum()); penalty_sum += float(penalty.sum().cpu())
            target_sum += float(targets.double().sum().cpu()) if device != 'mps' else float(targets.cpu().double().sum())
            target_squared_sum += float(targets.cpu().double().square().sum())
            prediction_sum += float(predicted.cpu().double().sum())
    per_match = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return {'rows': cache.rows, 'matches': len(cache.match_ids), 'empty_matches': int(np.sum(counts == 0)),
        'row_mse': squared_sum / cache.rows, 'match_mse': float(per_match.mean()),
        'ridge_mean': penalty_sum / cache.rows, 'target_mean': target_sum / cache.rows,
        'zero_prediction_row_mse': target_squared_sum / cache.rows,
        'prediction_mean': prediction_sum / cache.rows,
        'match_weighting': 'mean per-match row MSE; empty matches contribute zero and retain denominator'}


def save_checkpoint(path, head, optimizer, *, name, seed, epoch, updates, identity, device,
                    fixed_value_sha256=None, selection=None, boundary_kind='full-epoch'):
    require(boundary_kind in ('full-epoch', 'preflight-update'), 'Unknown resumable checkpoint boundary')
    require((boundary_kind == 'full-epoch' and epoch >= 1) or (boundary_kind == 'preflight-update' and epoch == 0),
            'Preflight update boundaries must not claim a completed full data pass')
    state = cpu_copy(head.state_dict())
    metadata = {'schema': CHECKPOINT_SCHEMA, 'identity': identity, 'head': name, 'seed': seed,
        'epoch': epoch, 'optimizer_updates': updates, 'device': device,
        'weights_sha256': _state_hash(state), 'fixed_value_sha256': fixed_value_sha256,
        'complete_epoch': boundary_kind == 'full-epoch', 'boundary_kind': boundary_kind,
        'selection': selection, 'policy_promotion': False}
    payload = {'metadata': metadata, 'head_state_dict': state,
        'optimizer_state_dict': cpu_copy(optimizer.state_dict())}
    buffer = io.BytesIO(); torch.save(payload, buffer)
    return {**write_new(path, buffer.getvalue()), 'weights_sha256': metadata['weights_sha256']}


def resume_checkpoint(path, head, *, expected_identity, expected_head, expected_seed, device,
                      fixed_value_sha256=None, expected_boundary_kind='full-epoch'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    metadata = saved['metadata']
    require(metadata.get('schema') == CHECKPOINT_SCHEMA and metadata['identity'] == expected_identity
        and metadata['head'] == expected_head and metadata['seed'] == expected_seed
        and metadata['device'] == device and metadata['fixed_value_sha256'] == fixed_value_sha256
        and metadata.get('boundary_kind') == expected_boundary_kind
        and metadata.get('complete_epoch') is (expected_boundary_kind == 'full-epoch'),
        'Resume identity/backend/frozen target mismatch')
    require(_state_hash(saved['head_state_dict']) == metadata['weights_sha256'], 'Resume weight hash mismatch')
    head.load_state_dict(saved['head_state_dict'], strict=True)
    head.to(device)
    optimizer = optimizer_for(head)
    optimizer.load_state_dict(saved['optimizer_state_dict'])
    return head, optimizer, metadata


def train_head(head, name, training, selection, *, seed, epochs, device, deadline,
               output_dir, identity, fixed_value=None, emit=None):
    optimizer = optimizer_for(head)
    best, best_state, records, updates = None, None, [], 0
    fixed_sha = _state_hash(fixed_value.state_dict()) if fixed_value is not None else None
    for epoch in range(1, epochs + 1):
        check_deadline(deadline)
        tick = time.monotonic()
        head.train()
        order = epoch_order(training.rows, seed, name, epoch)
        statistics = {'rows': 0, 'loss_sum': 0., 'squared_sum': 0., 'ridge_sum': 0.,
                      'clipped_updates': 0, 'maximum_gradient_l2': 0.}
        for start in range(0, len(order), RECIPE['batch_rows']):
            check_deadline(deadline)
            indices = order[start:start + RECIPE['batch_rows']]
            batch = training.batch(indices, device, all_actions=name == 'action')
            item = update(head, optimizer, name, batch, fixed_value)
            for key in ('rows', 'loss_sum', 'squared_sum', 'ridge_sum'):
                statistics[key] += item[key]
            statistics['clipped_updates'] += item['clipped']
            statistics['maximum_gradient_l2'] = max(statistics['maximum_gradient_l2'], item['gradient_l2'])
            updates += 1
        synchronize(device)
        train_seconds = time.monotonic() - tick
        validation = evaluate(head, name, selection, device, deadline, fixed_value)
        if best is None or validation['row_mse'] < best['selection']['row_mse']:
            best = {'epoch': epoch, 'selection': validation}
            best_state = cpu_copy(head.state_dict())
        checkpoint = save_checkpoint(Path(output_dir) / f'{name}-epoch-{epoch:03d}.pt', head, optimizer,
            name=name, seed=seed, epoch=epoch, updates=updates, identity=identity, device=device,
            fixed_value_sha256=fixed_sha, selection=validation)
        record = {'epoch': epoch, 'head': name, 'seed': seed,
            'order_sha256': hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),
            'training': {**statistics, 'row_mse': statistics['squared_sum'] / training.rows,
                'objective_mean': statistics['loss_sum'] / training.rows},
            'selection': validation, 'checkpoint': checkpoint,
            'training_seconds': train_seconds, 'epoch_seconds': time.monotonic() - tick,
            'optimizer_updates': updates}
        records.append(record)
        write_new(Path(output_dir) / f'{name}-epoch-{epoch:03d}.json', json_bytes(record))
        if emit is not None:
            emit({'phase': 'training', 'seed': seed, 'head': name, 'epoch': epoch,
                'selection_row_mse': validation['row_mse'], 'epoch_seconds': record['epoch_seconds']})
    require(best_state is not None and updates == epochs * math.ceil(training.rows / RECIPE['batch_rows']), 'Incomplete fixed full passes')
    head.load_state_dict(best_state, strict=True)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    if fixed_value is not None:
        require(_state_hash(fixed_value.state_dict()) == fixed_sha, 'Residual fitting changed its fixed value target')
    return {'head': name, 'selected_epoch': best['epoch'], 'selection': best['selection'],
        'optimizer_updates': updates, 'row_presentations': epochs * training.rows,
        'weights_sha256': _state_hash(head.state_dict()), 'epochs': records}


def train_seed(training, selection, *, seed, epochs, device, deadline, output_dir, identity, emit=None):
    require(training.partition == 'training' and selection.partition == 'selection', 'Training/selection roles changed')
    require(epochs in (4, 8, 12), 'Formal passes must be one of the predeclared admitted levels')
    heads = ResidualHeads(seed=seed).to(device)
    results = {}
    for name in HEAD_NAMES:
        results[name] = train_head(getattr(heads, name), name, training, selection, seed=seed,
            epochs=epochs, device=device, deadline=deadline, output_dir=output_dir, identity=identity,
            fixed_value=None if name == 'value' else heads.value, emit=emit)
    state_dicts = {name: cpu_copy(getattr(heads, name).state_dict()) for name in HEAD_NAMES}
    selected_epochs = {name: results[name]['selected_epoch'] for name in HEAD_NAMES}
    payload = {'schema': BUNDLE_SCHEMA, 'seed': seed, 'identity': identity,
        'head_config': heads.config, 'head_state_dicts': state_dicts,
        'selected_epochs': selected_epochs, 'weights_sha256': _state_hash(heads.state_dict()),
        'complete': True, 'policy_promotion': False, 'optimizer_updates': sum(v['optimizer_updates'] for v in results.values())}
    buffer = io.BytesIO(); torch.save(payload, buffer)
    saved = write_new(Path(output_dir) / 'selected.pt', buffer.getvalue())
    result = {'seed': seed, 'checkpoint': saved['path'], 'sha256': saved['sha256'],
        'bytes': saved['bytes'], 'weights_sha256': payload['weights_sha256'], 'selected_epochs': selected_epochs,
        'complete': True, 'heads': results, 'optimizer_updates': payload['optimizer_updates'],
        'row_presentations': sum(v['row_presentations'] for v in results.values()),
        'validation': {name: results[name]['selection'] for name in HEAD_NAMES}}
    write_new(Path(output_dir) / 'seed-report.json', json_bytes(result))
    return result


def preflight(training, selection, *, seed, device, deadline, output_dir, identity,
              benchmark_seeds=None):
    """Real Adam timing, CPU/accelerator parity and exact same-backend resume."""
    require(device == 'mps' or device.startswith('cuda'), 'Preflight requires CPU and accelerator comparison')
    count = min(RECIPE['batch_rows'], training.rows)
    indices = np.arange(count)
    snapshots, timings = {}, {}
    for backend in ('cpu', device):
        heads = ResidualHeads(seed=seed).to(backend)
        observations = {}
        for name in HEAD_NAMES:
            check_deadline(deadline)
            head = getattr(heads, name)
            fixed = None if name == 'value' else heads.value
            if fixed is not None:
                for parameter in fixed.parameters():
                    parameter.requires_grad_(False)
            opt = optimizer_for(head)
            # Include feature gather, actual gradient clipping and Adam, and
            # synchronize before timing. Three updates permit optimizer moments.
            tick = time.monotonic()
            for _ in range(3):
                item = update(head, opt, name, training.batch(indices, backend, all_actions=name == 'action'), fixed)
            synchronize(backend)
            step_seconds = (time.monotonic() - tick) / 3
            checkpoint = save_checkpoint(Path(output_dir) / f'{backend.replace(":", "-")}-{name}.pt', head, opt,
                name=name, seed=seed, epoch=0, updates=3, identity=identity, device=backend,
                fixed_value_sha256=_state_hash(fixed.state_dict()) if fixed is not None else None,
                boundary_kind='preflight-update')
            resumed = copy.deepcopy(head)
            resumed, resumed_opt, _ = resume_checkpoint(checkpoint['path'], resumed,
                expected_identity=identity, expected_head=name, expected_seed=seed, device=backend,
                fixed_value_sha256=_state_hash(fixed.state_dict()) if fixed is not None else None,
                expected_boundary_kind='preflight-update')
            batch = training.batch(indices, backend, all_actions=name == 'action')
            update(head, opt, name, batch, fixed)
            update(resumed, resumed_opt, name, batch, fixed)
            resume_max = max(float((a.detach().cpu() - b.detach().cpu()).abs().max())
                for a, b in zip(head.parameters(), resumed.parameters()))
            require(resume_max == 0., 'Same-backend optimizer resume must be bitwise identical')
            tick = time.monotonic()
            assessment = evaluate(head, name, selection, backend, deadline, fixed)
            synchronize(backend)
            eval_seconds = time.monotonic() - tick
            observations[name] = {'step_seconds': step_seconds, 'selection_seconds': eval_seconds,
                'resume_max_abs_difference': resume_max, 'selection': assessment,
                'last_step': item, 'checkpoint': checkpoint}
        snapshots[backend] = cpu_copy(heads.state_dict())
        timings[backend] = observations
    # Adam can amplify near-zero floating-point gradient differences; compare
    # observable predictions as well as all parameter differences, reporting
    # both rather than pretending cross-backend bitwise identity is expected.
    cpu_heads, accelerator_heads = ResidualHeads(seed=seed), ResidualHeads(seed=seed)
    cpu_heads.load_state_dict(snapshots['cpu']); accelerator_heads.load_state_dict(snapshots[device])
    comparisons = {}
    with torch.inference_mode():
        batch = training.batch(indices, 'cpu', all_actions=True)
        for name in HEAD_NAMES:
            features = batch['action'] if name == 'action' else batch['state']
            left, right = getattr(cpu_heads, name)(features), getattr(accelerator_heads, name)(features)
            absolute = float((left - right).abs().max())
            scale = float(left.abs().max())
            passed = bool(torch.allclose(left, right, atol=2e-3, rtol=2e-3))
            comparisons[name] = {'prediction_max_abs_difference': absolute, 'cpu_max_abs_prediction': scale,
                'parameter_max_abs_difference': max(float((snapshots['cpu'][key] - snapshots[device][key]).abs().max())
                    for key in snapshots['cpu'] if key.startswith(name + '.')),
                'passed': passed, 'atol': 2e-3, 'rtol': 2e-3}
    benchmark_records = []
    for benchmark_seed in (benchmark_seeds or [seed]):
        heads = ResidualHeads(seed=benchmark_seed)
        if benchmark_seed == seed:
            heads.load_state_dict(snapshots[device], strict=True)
        payload = {'schema': BUNDLE_SCHEMA, 'seed': benchmark_seed, 'identity': identity,
            'head_config': heads.config,
            'head_state_dicts': {name: cpu_copy(getattr(heads, name).state_dict()) for name in HEAD_NAMES},
            'selected_epochs': {name: 0 for name in HEAD_NAMES},
            'weights_sha256': _state_hash(heads.state_dict()), 'complete': False,
            'benchmark_only': True, 'policy_promotion': False,
            'initialization_note': 'First seed contains the measured preflight Adam updates; others retain zero output initialization.'}
        buffer = io.BytesIO(); torch.save(payload, buffer)
        saved = write_new(Path(output_dir) / f'benchmark-seed-{benchmark_seed}' / 'selected.pt', buffer.getvalue())
        benchmark_records.append({'seed': benchmark_seed, 'checkpoint': saved['path'], 'sha256': saved['sha256'],
            'weights_sha256': payload['weights_sha256'], 'selected_epochs': payload['selected_epochs'],
            'complete': False, 'benchmark_only': True})
    return {'passed': all(record['passed'] for record in comparisons.values()), 'complete': True,
        'rows_per_step': count, 'actual_optimizer_updates': 30,
        'backend_comparison': comparisons, 'timings': timings, 'seeds': benchmark_records,
        'timing_note': 'Three same fixed-row updates per head/backend; formal cache loading, checkpoint persistence and thermal variation require admission allowance.'}
