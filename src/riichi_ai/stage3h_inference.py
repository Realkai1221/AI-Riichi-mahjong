"""Fixed saved-state diagnosis with deployed CPU single-row inference.

Every loaded policy is read-only. Full legal score vectors are retained for an
independent arithmetic audit; terminal/delta labels never enter actor tensors.
"""
from __future__ import annotations
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .dataset import _pack_arrays
from .model import PolicyNetwork, _state_hash, collate_encoded
from .semantics import ACTION_KINDS
from .stage3f_stream import StreamBatch
from .stage3g_data import LearningDataset, HandTargets
from .stage3g_learning import CHECKPOINT_SCHEMA, RECIPE, ARMS

PLAN_SCHEMA = 'stage3h-diagnostic-plan-v1'
MATCH_SCHEMA = 'riichi-ai-stage3h-inference-match-v1'
RESULT_SCHEMA = 'riichi-ai-stage3h-inference-v1'
SLICE_NAMES = ('all', 'discard', 'call', 'riichi', 'kan', 'win', 'call_and_noncall')
PROBABILITY_BUCKETS = ('zero', '(0,.01]', '(.01,.05]', '(.05,.2]', '(.2,1]')
KINDS = {name: index + 1 for index, name in enumerate(ACTION_KINDS)}
CALL_KINDS = frozenset(KINDS[name] for name in ('CHI', 'PON', 'DAIMINKAN', 'ANKAN', 'KAKAN'))
KAN_KINDS = frozenset(KINDS[name] for name in ('DAIMINKAN', 'ANKAN', 'KAKAN'))
WIN_KINDS = frozenset(KINDS[name] for name in ('RON', 'TSUMO'))


def sha(path: str | Path) -> str:
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def write_new(path: Path, data: bytes) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(data)
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def load_checkpoint(record: dict, device: str = 'cpu') -> tuple[PolicyNetwork, dict]:
    """Strict intermediate/final loader; never constructs or restores Adam."""
    if (record.get('kind') not in ('reference', 'value') or not isinstance(record.get('id'), str)
            or not record['id'] or type(record.get('update')) is not int):
        raise ValueError('Malformed diagnostic model record')
    raw = Path(record['path']).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != record.get('sha256'):
        raise ValueError('Diagnostic checkpoint file hash mismatch')
    saved = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    model = PolicyNetwork(saved['model_config'])
    model.load_state_dict(saved['model_state_dict'], strict=True)
    tensor_hash = _state_hash(model.state_dict())
    rank_hash = _state_hash(model.rank_head.state_dict())
    if record['kind'] == 'value':
        metadata = saved.get('metadata', {})
        identity = metadata.get('identity', {})
        if (metadata.get('schema') != CHECKPOINT_SCHEMA or metadata.get('weights_sha256') != tensor_hash
                or metadata.get('frozen_rank_sha256') != rank_hash or identity.get('purpose') != 'formal-training'
                or identity.get('split') != 'training' or identity.get('arm') != record.get('arm')
                or identity.get('recipe') != RECIPE or metadata.get('scheduled_updates') != record['update']
                or metadata.get('optimizer_updates') != record['update']
                or record['update'] not in (0, 100, 200, 300, 400)
                or metadata.get('complete') is not (record['update'] == 400)
                or identity.get('reference_sha256') != record.get('reference_sha256')):
            raise ValueError('Diagnostic value checkpoint identity or boundary mismatch')
        if record.get('corpus_manifest_sha256') is not None and identity.get('manifest_sha256') != record['corpus_manifest_sha256']:
            raise ValueError('Value checkpoint has a different corpus identity')
        if record.get('weights_sha256') is not None and record['weights_sha256'] != tensor_hash:
            raise ValueError('Value tensor identity differs from fixed plan')
    else:
        if record.get('arm') is not None or record['update'] != 0:
            raise ValueError('Reference record cannot name a training arm or update')
        metadata = {'schema': 'stage3h-reference-policy-metadata-v1',
                    'outputs': 'Policy logits, not utility predictions'}
    for name, parameter in model.named_parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise ValueError('Diagnostic checkpoint has nonfinite parameters')
        parameter.requires_grad_(not name.startswith('rank_head.'))
    metadata = {**metadata, 'checkpoint_sha256': digest, 'weights_sha256': tensor_hash,
        'frozen_rank_sha256': rank_hash, 'diagnostic_model_id': record['id'], 'kind': record['kind']}
    return model.to(device).eval(), metadata


def validate_plan(plan: dict, phase: str) -> None:
    if plan.get('schema') != PLAN_SCHEMA or phase not in ('preflight', 'run'):
        raise ValueError('Unknown diagnostic plan or phase')
    if phase == 'run' and plan.get('execution_admitted') is not True:
        raise ValueError('Formal saved-state inference requires measured admission')
    if plan.get('workers') != 14:
        raise ValueError('Fixed fourteen CPU workers required')
    models = plan.get('models')
    if not isinstance(models, list) or len(models) != 18:
        raise ValueError('Exactly eighteen fixed weight sets required')
    ids = [entry.get('id') for entry in models]
    if len(set(ids)) != 18 or any(not isinstance(name, str) or not name for name in ids):
        raise ValueError('Model IDs must be unique strings')
    refs = [r for r in models if r.get('kind') == 'reference']
    initials = [r for r in models if r.get('kind') == 'value' and r.get('update') == 0]
    if (len(refs) != 1 or len(initials) != 1 or refs[0]['id'] != plan.get('reference_model_id')
            or initials[0]['id'] != plan.get('initial_model_id')):
        raise ValueError('Exactly one declared reference and common initialization required')
    boundaries = [(r.get('arm'), r.get('update')) for r in models if r.get('kind') == 'value' and r.get('update') != 0]
    if sorted(boundaries) != sorted((arm, update) for arm in ARMS for update in (100, 200, 300, 400)):
        raise ValueError('All fixed arm boundaries must be represented once')
    for entry in models:
        if (entry.get('kind') == 'value' and entry.get('reference_sha256') != refs[0].get('sha256')):
            raise ValueError('Reference identity differs across planned models')
    for key, hash_key in (('corpus_path', 'corpus_manifest_sha256'), ('preflight_corpus_path', 'preflight_manifest_sha256')):
        if sha(Path(plan[key]) / 'manifest.json') != plan[hash_key]:
            raise ValueError('Input corpus manifest hash mismatch')
    if sha(plan['hand_targets_path']) != plan['hand_targets_sha256']:
        raise ValueError('Immutable hand target hash mismatch')
    selection = plan.get('selected_matches')
    if not isinstance(selection, dict) or set(selection) != {'training', 'development'}:
        raise ValueError('Both split selections must be explicit')
    for split, names in selection.items():
        if (not isinstance(names, list) or not names or len(names) != len(set(names))
                or any(not isinstance(name, str) for name in names)):
            raise ValueError('Invalid selected match IDs')
    if set(selection['training']) & set(selection['development']):
        raise ValueError('Selected split overlap')
    rows = plan.get('preflight_rows')
    if (not isinstance(rows, list) or not 16 <= len(rows) <= 1024
            or any(not isinstance(r, dict) or not isinstance(r.get('match_id'), str)
                   or type(r.get('row_index')) is not int or r['row_index'] < 0 for r in rows)
            or len({(r['match_id'], r['row_index']) for r in rows}) != len(rows)
            or len({r['match_id'] for r in rows}) != 16):
        raise ValueError('Preflight must name distinct fixed rows in sixteen matches')


def one_row_actor(actor: dict, targets: dict, row: int):
    arrays, labels = LearningDataset._batch(actor, targets, np.asarray([row], dtype=np.int64))
    batch = StreamBatch(arrays, labels, '', np.asarray([row], dtype=np.int64), {}, 0)
    return batch.torch_actor('cpu'), arrays


def evaluate_one(model: PolicyNetwork, actor: dict[str, torch.Tensor]) -> np.ndarray:
    if actor['tiles'].shape[0] != 1 or any(value.device.type != 'cpu' for value in actor.values()):
        raise ValueError('Deployed numerical comparison requires one CPU row')
    with torch.inference_mode():
        values, _ = model(actor)
    result = values[0].detach().cpu().numpy().copy()
    if result.shape != (int(actor['candidate_mask'][0].sum()),) or not np.isfinite(result).all():
        raise ValueError('Invalid legal score vector')
    return result


def probability_bucket(probability: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability)
    if not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability > 1):
        raise ValueError('Invalid recorded behavior probability')
    return np.where(probability == 0, 0, np.searchsorted(np.asarray([.01, .05, .2, 1.]), probability, side='left') + 1).astype(np.int8)


def opportunity_mask(kinds: np.ndarray) -> int:
    present = set(map(int, kinds))
    call = bool(present & CALL_KINDS)
    flags = (True, KINDS['DISCARD'] in present, call, KINDS['RIICHI'] in present,
             bool(present & KAN_KINDS), bool(present & WIN_KINDS), call and bool(present - CALL_KINDS))
    return sum(int(value) << index for index, value in enumerate(flags))


def choices(scores: np.ndarray, kinds: np.ndarray) -> tuple[int, int]:
    raw = int(scores.argmax())
    wins = np.flatnonzero(np.isin(kinds, tuple(WIN_KINDS)))
    return raw, int(wins[0]) if len(wins) else raw


_STATE = {}


def _initialize_worker(plan: dict, phase: str, deadline: float):
    global _STATE
    tick = time.perf_counter()
    torch.set_num_threads(1)
    models = []
    for record in plan['models']:
        if time.monotonic() >= deadline:
            raise TimeoutError('Diagnostic deadline during model loading')
        model, metadata = load_checkpoint(record, 'cpu')
        models.append((model, metadata))
    common_hash = next(metadata['weights_sha256'] for _, metadata in models
                       if metadata['diagnostic_model_id'] == plan['initial_model_id'])
    for (_, metadata), record in zip(models, plan['models']):
        if record['kind'] == 'value':
            identity = metadata['identity']
            if (identity['manifest_sha256'] != plan['corpus_manifest_sha256']
                    or metadata['initial_weights_sha256'] != common_hash
                    or identity['hand_targets_sha256'] != (plan['hand_targets_sha256'] if record['arm'].startswith('delta_') else None)):
                raise ValueError('Diagnostic checkpoint corpus, target or common initialization differs')
    model_load = time.perf_counter() - tick
    if phase == 'preflight':
        primary = LearningDataset(plan['preflight_corpus_path'], 'preflight')
        datasets = {'preflight': primary}
        targets = None
    else:
        primary = LearningDataset(plan['corpus_path'], 'training')
        other = copy.copy(primary)
        other.split = 'development'
        other.match_ids = tuple(primary.manifest['splits']['development'])
        datasets = {'training': primary, 'development': other}
        mapping = HandTargets(plan['hand_targets_path'], primary)
        if mapping.sha256 != plan['hand_targets_sha256']:
            raise ValueError('Hand-target mapping changed at worker load')
        selected = set(plan['selected_matches']['training'] + plan['selected_matches']['development'])
        targets = {name: mapping.payload['matches'][name] for name in selected}
    parity = {(r['match_id'], r['row_index']) for r in plan['preflight_rows'][:16]}
    _STATE = {'plan': plan, 'phase': phase, 'models': models, 'datasets': datasets,
              'targets': targets, 'deadline': deadline, 'parity_rows': parity,
              'worker_model_load_seconds': model_load,
              'worker_initialization_seconds': time.perf_counter() - tick, 'jobs': 0}


def _worker_match(job: dict) -> dict:
    state = _STATE
    if time.monotonic() >= state['deadline']:
        raise TimeoutError('No diagnostic match may start after deadline')
    started = time.perf_counter()
    plan, phase, name, split = state['plan'], state['phase'], job['match_id'], job['split']
    dataset = state['datasets'][split]
    if name not in dataset.match_ids:
        raise ValueError('Selected match belongs to another split')
    actor, targets, _ = dataset._load(name)
    eligible = np.flatnonzero((targets['behavior_id'] == 3) & (np.diff(actor['candidate_offsets']) > 1))
    rows = np.asarray(job.get('row_indices', eligible), dtype=np.int64)
    if len(set(map(int, rows))) != len(rows) or not set(map(int, rows)) <= set(map(int, eligible)):
        raise ValueError('Selected rows are not distinct optional explorer rows')
    if phase == 'run' and not np.array_equal(rows, eligible):
        raise ValueError('Formal diagnosis must include every eligible row')
    counts = np.diff(actor['candidate_offsets'])[rows]
    offsets = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(counts, dtype=np.int64)))
    total, n_models, n_rows = int(offsets[-1]), len(state['models']), len(rows)
    kinds = np.empty(total, dtype=np.int8)
    probabilities = np.empty(total, dtype=np.float64)
    scores = np.empty((n_models, total), dtype=np.float32)
    selection = {key: np.empty((n_models, n_rows), dtype=np.int16 if key.endswith('index') else np.int8)
                 for key in ('raw_index', 'wrapped_index', 'deployed_index', 'raw_kind', 'wrapped_kind', 'deployed_kind')}
    propensities = {key: np.empty((n_models, n_rows), dtype=np.float64)
                   for key in ('raw_probability', 'wrapped_probability', 'deployed_probability')}
    logged = targets['chosen_index'][rows].astype(np.int16)
    logged_kind = np.empty(n_rows, dtype=np.int8)
    mask = np.empty(n_rows, dtype=np.uint8)
    delta = np.zeros(n_rows, dtype=np.float32)
    if state['targets'] is not None:
        entry = state['targets'][name]
        if np.any(targets['seat'][rows] != entry['explorer_seat']):
            raise ValueError('Hand-target explorer seat differs')
        delta = np.asarray([entry['hand_deltas'][str(int(hand))] for hand in targets['hand_id'][rows]], dtype=np.float32)
    forward_seconds = 0.
    path_checks = []
    reference_id, initial_id = plan['reference_model_id'], plan['initial_model_id']
    for position, row in enumerate(rows):
        if time.monotonic() >= state['deadline']:
            raise TimeoutError('Diagnostic deadline inside selected match')
        start, end = map(int, actor['candidate_offsets'][int(row):int(row) + 2])
        output_start, output_end = map(int, offsets[position:position + 2])
        row_kinds = actor['action_tokens'][start:end, 0].astype(np.int8)
        row_probabilities = targets['behavior_probabilities'][start:end]
        kinds[output_start:output_end] = row_kinds
        probabilities[output_start:output_end] = row_probabilities
        logged_kind[position] = row_kinds[logged[position]]
        mask[position] = opportunity_mask(row_kinds)
        batch, arrays = one_row_actor(actor, targets, int(row))
        for model_index, (model, metadata) in enumerate(state['models']):
            tick = time.perf_counter()
            prediction = evaluate_one(model, batch)
            forward_seconds += time.perf_counter() - tick
            if metadata['kind'] == 'value' and np.max(np.abs(prediction)) > 100:
                raise ValueError('Value checkpoint exceeded the frozen legal-value safety bound')
            scores[model_index, output_start:output_end] = prediction
            raw, wrapped = choices(prediction, row_kinds)
            deployed = raw if metadata['kind'] == 'reference' else wrapped
            for label, index in (('raw', raw), ('wrapped', wrapped), ('deployed', deployed)):
                selection[f'{label}_index'][model_index, position] = index
                selection[f'{label}_kind'][model_index, position] = row_kinds[index]
                propensities[f'{label}_probability'][model_index, position] = row_probabilities[index]
            if (phase == 'preflight' and (name, int(row)) in state['parity_rows']
                    and metadata['diagnostic_model_id'] in (reference_id, initial_id)):
                sample = {key: value[0] for key, value in arrays.items() if key != 'candidate_mask'}
                deployed_batch = collate_encoded([sample], 'cpu')
                tensors_exact = (set(deployed_batch) == set(batch)
                    and all(torch.equal(batch[key], deployed_batch[key]) and batch[key].dtype == deployed_batch[key].dtype for key in batch))
                deployed_prediction = evaluate_one(model, deployed_batch)
                exact = np.array_equal(prediction, deployed_prediction)
                path_checks.append({'match_id': name, 'row_index': int(row), 'model_id': metadata['diagnostic_model_id'],
                    'tensors_exact': tensors_exact, 'scores_exact': exact,
                    'maximum_score_abs_difference': float(np.max(np.abs(prediction - deployed_prediction)))})
                if not tensors_exact or not exact:
                    raise ValueError('Diagnostic path differs from production single-row collation')
    hashes = []
    for model, metadata in state['models']:
        current = _state_hash(model.state_dict())
        if current != metadata['weights_sha256'] or _state_hash(model.rank_head.state_dict()) != metadata['frozen_rank_sha256']:
            raise ValueError('Read-only diagnostic mutated model weights')
        hashes.append({'model_id': metadata['diagnostic_model_id'], 'before': metadata['weights_sha256'], 'after': current})
    arrays = {'row_indices': rows.astype(np.int32), 'seat': targets['seat'][rows], 'step': targets['step'][rows],
        'hand_id': targets['hand_id'][rows], 'logged_index': logged, 'logged_kind': logged_kind,
        'candidate_offsets': offsets, 'candidate_kinds': kinds, 'behavior_probabilities': probabilities,
        'terminal_targets': targets['final_utility'][rows].astype(np.float32), 'delta_targets': delta,
        'opportunity_mask': mask, 'legal_scores': scores, **selection, **propensities}
    output = Path(job['output_dir'])
    numeric = write_new(output / 'arrays' / f'{split}-{name}.npz', _pack_arrays(arrays))
    first_job = state['jobs'] == 0
    state['jobs'] += 1
    report = {'schema': MATCH_SCHEMA, 'phase': phase, 'split': split, 'match_id': name,
        'plan_sha256': job['plan_sha256'], 'rows': n_rows, 'model_rows': n_models * n_rows,
        'candidate_values_per_model': total, 'model_ids': [r['id'] for r in plan['models']],
        'numeric': numeric, 'delta_target_available': state['targets'] is not None,
        'source_actor': dataset.reports[name]['artifacts'][f'actor/{name}.npz'],
        'source_targets': dataset.reports[name]['artifacts'][f'targets/{name}.npz'],
        'models_unchanged': hashes, 'production_path_checks': path_checks, 'worker_pid': os.getpid(),
        'worker_model_load_seconds': state['worker_model_load_seconds'] if first_job else None,
        'worker_initialization_seconds': state['worker_initialization_seconds'] if first_job else None,
        'forward_seconds': forward_seconds, 'match_seconds': time.perf_counter() - started,
        'complete': True, 'optimizer_steps': 0, 'new_matches': 0}
    write_new(output / 'matches' / f'{split}-{name}.json', json_bytes(report))
    return report


def row_metrics(arrays: dict, model_index: int, record: dict, *, reference_index: int,
                initial_index: int, delta_available: bool) -> dict[str, np.ndarray]:
    """Descriptive same-state metrics; no unobserved-action outcome labels."""
    raw = arrays['raw_index'][model_index]
    wrapped = arrays['wrapped_index'][model_index]
    deployed = arrays['deployed_index'][model_index]
    logged = arrays['logged_index']
    reference = arrays['deployed_index'][reference_index]
    initial = arrays['deployed_index'][initial_index]
    metrics = {
        'raw_matches_reference': (raw == reference).astype(np.float64),
        'wrapped_matches_reference': (wrapped == reference).astype(np.float64),
        'deployed_matches_reference': (deployed == reference).astype(np.float64),
        'deployed_matches_initial': (deployed == initial).astype(np.float64),
        'raw_matches_logged': (raw == logged).astype(np.float64),
        'wrapped_matches_logged': (wrapped == logged).astype(np.float64),
        'deployed_matches_logged': (deployed == logged).astype(np.float64),
        'win_wrapper_changes_raw': (raw != wrapped).astype(np.float64),
        **{name: arrays[name][model_index] for name in ('raw_probability', 'wrapped_probability', 'deployed_probability')},
    }
    for policy in ('raw', 'wrapped', 'deployed'):
        for name, token in KINDS.items():
            metrics[f'{policy}_kind_{name}'] = (arrays[f'{policy}_kind'][model_index] == token).astype(np.float64)
        buckets = probability_bucket(arrays[f'{policy}_probability'][model_index])
        for bucket, label in enumerate(PROBABILITY_BUCKETS):
            metrics[f'{policy}_probability_bucket_{label}'] = (buckets == bucket).astype(np.float64)
    rows = len(logged)
    chosen_scores = np.empty(rows, dtype=np.float64)
    maximum, minimum, margin, ce = (np.empty(rows, dtype=np.float64) for _ in range(4))
    for row, (start, end) in enumerate(zip(arrays['candidate_offsets'][:-1], arrays['candidate_offsets'][1:])):
        values = arrays['legal_scores'][model_index, int(start):int(end)].astype(np.float64)
        if len(values) <= 1 or not np.isfinite(values).all():
            raise ValueError('Diagnostic metrics require finite optional legal scores')
        chosen_scores[row] = values[int(logged[row])]
        maximum[row], minimum[row] = float(values.max()), float(values.min())
        margin[row] = maximum[row] - np.partition(values, -2)[-2]
        ce[row] = maximum[row] + np.log(np.exp(values - maximum[row]).sum()) - chosen_scores[row]
    prefix = 'logit' if record['kind'] == 'reference' else 'value'
    metrics.update({f'{prefix}_maximum': maximum, f'{prefix}_minimum': minimum,
        f'{prefix}_spread': maximum - minimum, f'{prefix}_top_two_margin': margin,
        f'{prefix}_greedy_minus_logged': maximum - chosen_scores})
    if record['kind'] == 'value':
        metrics['chosen_ce'] = ce
        metrics['mc_chosen_mse'] = (chosen_scores - arrays['terminal_targets']) ** 2
        metrics['mc_zero_prediction_mse'] = arrays['terminal_targets'].astype(np.float64) ** 2
        if delta_available:
            metrics['delta_chosen_mse'] = (chosen_scores - arrays['delta_targets']) ** 2
            metrics['delta_zero_prediction_mse'] = arrays['delta_targets'].astype(np.float64) ** 2
        if record['update'] != 0:
            target, alpha = ARMS[record['arm']]
            key = 'mc_chosen_mse' if target == 'terminal_utility' else 'delta_chosen_mse'
            if key in metrics:
                metrics['declared_objective'] = metrics[key] + alpha * ce
    return metrics


def aggregate_matches(plan: dict, records: list[dict]) -> dict:
    """Whole-match and row-weighted descriptions with explicit slice support."""
    ids = [record['id'] for record in plan['models']]
    reference_index = ids.index(plan['reference_model_id'])
    initial_index = ids.index(plan['initial_model_id'])
    totals: dict = {}
    split_counts: dict = {}
    for report in records:
        split = report['split']
        split_counts.setdefault(split, {'matches': 0, 'optional_rows': 0, 'empty_matches': 0})
        split_counts[split]['matches'] += 1
        split_counts[split]['optional_rows'] += report['rows']
        split_counts[split]['empty_matches'] += int(report['rows'] == 0)
        path = Path(report['numeric']['path'])
        if path.stat().st_size != report['numeric']['bytes'] or sha(path) != report['numeric']['sha256']:
            raise ValueError('Stored inference array hash mismatch before aggregation')
        with np.load(path, allow_pickle=False) as stored:
            arrays = {name: stored[name] for name in stored.files}
        for model_index, model_record in enumerate(plan['models']):
            metrics = row_metrics(arrays, model_index, model_record, reference_index=reference_index,
                initial_index=initial_index, delta_available=report['delta_target_available'])
            for slice_index, slice_name in enumerate(SLICE_NAMES):
                mask = (arrays['opportunity_mask'] & (1 << slice_index)) != 0
                count = int(mask.sum())
                key = (split, model_record['id'], slice_name)
                total = totals.setdefault(key, {'rows': 0, 'supporting_matches': 0, 'row_sums': {}, 'match_sums': {},
                    'transition_row_sum': np.zeros((len(ACTION_KINDS), len(ACTION_KINDS)), dtype=np.float64),
                    'transition_match_sum': np.zeros((len(ACTION_KINDS), len(ACTION_KINDS)), dtype=np.float64)})
                if not count:
                    continue
                total['rows'] += count
                total['supporting_matches'] += 1
                for metric, values in metrics.items():
                    value = float(values[mask].sum())
                    total['row_sums'][metric] = total['row_sums'].get(metric, 0.) + value
                    total['match_sums'][metric] = total['match_sums'].get(metric, 0.) + value / count
                transition = np.zeros((len(ACTION_KINDS), len(ACTION_KINDS)), dtype=np.float64)
                np.add.at(transition, (arrays['deployed_kind'][reference_index, mask] - 1,
                                      arrays['deployed_kind'][model_index, mask] - 1), 1)
                total['transition_row_sum'] += transition
                total['transition_match_sum'] += transition / count
    summary = {}
    for (split, model_id, slice_name), total in totals.items():
        rows, matches = total['rows'], total['supporting_matches']
        summary.setdefault(split, {}).setdefault(model_id, {})[slice_name] = {
            'rows': rows, 'supporting_matches': matches,
            'row_weighted': {k: v / rows for k, v in total['row_sums'].items()} if rows else {},
            'match_weighted': {k: v / matches for k, v in total['match_sums'].items()} if matches else {},
            'reference_to_deployed_kind_transition_row_weighted': (total['transition_row_sum'] / rows).tolist() if rows else None,
            'reference_to_deployed_kind_transition_match_weighted': (total['transition_match_sum'] / matches).tolist() if matches else None,
        }
    return {'splits': split_counts, 'by_split_model_slice': summary,
        'action_kind_order': list(ACTION_KINDS), 'opportunity_slices': list(SLICE_NAMES),
        'probability_buckets': list(PROBABILITY_BUCKETS),
        'normalization': 'Row means over the slice; match means average within-match slice means over matches with at least one slice row. Empty support is reported, never silently assigned an agreement rate.',
        'counterfactual_limitation': 'Behavior propensities measure collection support. Scores and score gaps do not supply outcomes for unchosen actions. Reference logits are never scored as utility predictions.'}


def run_plan(plan: dict, output_dir: str | Path, *, phase: str, max_seconds: float,
             plan_sha256: str | None = None) -> dict:
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError('A positive finite cooperative deadline is required')
    started = time.monotonic()
    deadline = started + max_seconds
    validate_plan(plan, phase)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    plan_hash = plan_sha256 or hashlib.sha256(json_bytes(plan)).hexdigest()
    write_new(output / 'execution-plan.json', json_bytes(plan))
    jobs = []
    if phase == 'preflight':
        grouped = {}
        for row in plan['preflight_rows']:
            grouped.setdefault(row['match_id'], []).append(row['row_index'])
        jobs = [{'match_id': name, 'split': 'preflight', 'row_indices': rows} for name, rows in grouped.items()]
    else:
        for split in ('training', 'development'):
            jobs.extend({'match_id': name, 'split': split} for name in plan['selected_matches'][split])
    for job in jobs:
        job.update(output_dir=str(output), plan_sha256=plan_hash)
    failure = None
    parent_compute_started = time.monotonic()
    first_result_at = None
    finished = 0
    try:
        with ProcessPoolExecutor(max_workers=plan['workers'], mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_initialize_worker, initargs=(plan, phase, deadline)) as pool:
            pending = deque()
            remaining = iter(jobs)
            for _ in range(plan['workers']):
                job = next(remaining, None)
                if job is not None:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('No initial submission after diagnostic deadline')
                    pending.append(pool.submit(_worker_match, job))
            while pending:
                result = pending.popleft().result()
                if first_result_at is None:
                    first_result_at = time.monotonic()
                finished += 1
                if finished % 16 == 0 or finished == len(jobs):
                    print(json.dumps({'phase': phase, 'completed_saved_matches': finished,
                                      'planned_saved_matches': len(jobs)}), flush=True)
                job = next(remaining, None)
                if job is not None:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('No replacement submission after diagnostic deadline')
                    pending.append(pool.submit(_worker_match, job))
    except BaseException as exc:
        failure = {'type': type(exc).__name__, 'message': str(exc)}
    pool_seconds = time.monotonic() - parent_compute_started
    records = [json.loads(path.read_text()) for path in sorted((output / 'matches').glob('*.json'))]
    expected = {(job['split'], job['match_id']) for job in jobs}
    actual = {(report['split'], report['match_id']) for report in records}
    if failure is None and (actual != expected or len(records) != len(expected)):
        failure = {'type': 'CoverageError', 'message': 'Missing or duplicate selected match outputs'}
    summary = None
    if failure is None:
        try:
            if time.monotonic() >= deadline:
                raise TimeoutError('Diagnostic deadline before fixed aggregation')
            summary = aggregate_matches(plan, records)
            if time.monotonic() >= deadline:
                raise TimeoutError('Diagnostic deadline after fixed aggregation')
        except BaseException as exc:
            failure = {'type': type(exc).__name__, 'message': str(exc)}
    model_rows = sum(report['model_rows'] for report in records)
    rows = sum(report['rows'] for report in records)
    worker_load = [report['worker_initialization_seconds'] for report in records if report['worker_initialization_seconds'] is not None]
    # A conservative measured cold-start allowance: parent-to-first-completed
    # match includes process startup, model loading and one match's real work.
    cold_to_first = (first_result_at - parent_compute_started) if first_result_at is not None else pool_seconds
    steady_seconds = max(pool_seconds - max(worker_load, default=0.), 1e-9)
    result = {'schema': RESULT_SCHEMA, 'phase': phase, 'complete': failure is None, 'failure': failure,
        'plan_sha256': plan_hash, 'planned_saved_matches': len(jobs), 'completed_saved_matches': len(records),
        'optional_rows': rows, 'model_rows': model_rows, 'models': plan['models'], 'matches': records,
        'summary': summary, 'optimizer_steps': 0, 'new_matches': 0, 'policy_promotion': False,
        'inference_path': 'CPU float32; one actor row per forward; one Torch thread per worker; fourteen independent worker processes',
        'timing': {'pool_seconds_including_spawn_load_and_shutdown': pool_seconds,
            'cold_start_to_first_completed_match_seconds': cold_to_first,
            'worker_initialization_seconds': worker_load,
            'model_rows_per_second_including_pool_startup': model_rows / pool_seconds if pool_seconds else 0.,
            'estimated_pool_seconds_after_max_worker_initialization': steady_seconds,
            'estimated_steady_model_rows_per_second': model_rows / steady_seconds,
            'steady_estimate_limitation': 'Subtracts maximum measured worker initialization, not a separately long sustained benchmark; admission should preserve a twofold allowance and cold-start reserve.',
            'summed_worker_forward_seconds': sum(r['forward_seconds'] for r in records)},
        'elapsed_seconds': time.monotonic() - started,
        'production_path_checks': [check for r in records for check in r['production_path_checks']],
        'limitations': 'Fixed saved-state diagnosis only; no new action outcomes, training, policy-strength estimate or policy promotion.'}
    if phase == 'preflight' and failure is None:
        checks = result['production_path_checks']
        if len(checks) != 32 or not all(c['tensors_exact'] and c['scores_exact'] for c in checks):
            result['complete'] = False
            result['failure'] = {'type': 'ParityError', 'message': 'All sixteen rows × two models require exact production-path checks'}
    write_new(output / 'result.json', json_bytes(result))
    return result


def preflight_plan(plan: dict, output_dir: str | Path, *, max_seconds: float,
                   plan_sha256: str | None = None) -> dict:
    return run_plan(plan, output_dir, phase='preflight', max_seconds=max_seconds, plan_sha256=plan_sha256)
