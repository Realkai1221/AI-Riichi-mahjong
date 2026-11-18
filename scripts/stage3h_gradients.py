#!/usr/bin/env python3
"""Leased saved-data gradient mechanism diagnostic with zero optimizer steps."""
from __future__ import annotations
import argparse
import hashlib
from importlib.metadata import version
import io
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from riichi_ai.stage3g_data import LearningDataset, HandTargets
from riichi_ai.stage3h_gradients import (EPSILON, check_deadline, compare_backends, describe_groups,
    gradient_group, public_group, require)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def write_new(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def checkpoint_records(plan):
    models = {record['id']: record for record in plan['models']}
    require(len(models) == len(plan['models']), 'Duplicate frozen checkpoint IDs')
    conditions = plan['gradient_conditions']
    require(len(conditions) == 6 and len({c['id'] for c in conditions}) == 6, 'Six distinct gradient conditions required')
    initial = conditions[:2]
    require([c['target'] for c in initial] == ['terminal_utility', 'one_hand_delta']
            and initial[0]['checkpoint_id'] == initial[1]['checkpoint_id'], 'Common initial terminal/delta conditions required')
    common = models[initial[0]['checkpoint_id']]
    require(common['kind'] == 'value' and common['arm'] == 'mc_a0' and common['update'] == 0, 'Wrong common initial checkpoint')
    for condition, arm in zip(conditions[2:], ('mc_a0', 'mc_a1', 'delta_a0', 'delta_a1')):
        record = models[condition['checkpoint_id']]
        require(record['kind'] == 'value' and record['arm'] == arm and record['update'] == 400
                and condition['target'] == ('one_hand_delta' if arm.startswith('delta_') else 'terminal_utility')
                and condition['trained_alpha'] == int(arm[-1]), 'Wrong final model/own-target gradient condition')
    return models, conditions


def validate_groups(plan, dataset):
    groups = plan['gradient_groups']
    require(len(groups) == 4 and all(len(group) == 16 and len(set(group)) == 16 for group in groups),
            'Four disjoint complete sixteen-match groups required')
    names = [name for group in groups for name in group]
    require(len(set(names)) == 64 and names == plan['selected_matches']['training'][:64]
            and set(names) <= set(dataset.match_ids), 'Gradient sample differs from the fixed first64 training matches')
    return groups


def identity(plan, models, conditions, groups):
    selected_models = {c['checkpoint_id']: models[c['checkpoint_id']]['sha256'] for c in conditions}
    return {'corpus_manifest_sha256': plan['corpus_manifest_sha256'], 'hand_targets_sha256': plan['hand_targets_sha256'],
        'selected_models': selected_models, 'gradient_groups': groups, 'gradient_conditions': conditions}


def run(args):
    from stage3h import read_config, verify_sources
    from riichi_ai.stage3h_inference import load_checkpoint, validate_plan
    config, config_sha = read_config()
    active = [item for item in read(Path(config['data_root']) / 'budget-ledger.json')['operations'] if item['status'] == 'active']
    require(len(active) == 1 and active[0]['name'] == os.environ.get('RIICHI_STAGE3H_OPERATION')
            and active[0]['token'] == os.environ.get('RIICHI_STAGE3H_TOKEN'), 'Matching active Stage3H numerical lease required')
    require(active[0]['name'].startswith('preflight') if args.phase == 'preflight' else active[0]['name'] == 'gradients',
            'Wrong numerical lease for gradient phase')
    started = time.monotonic()
    remaining = active[0]['reserved_seconds'] - (time.time() - active[0]['started_unix'])
    require(remaining > 12 and math.isfinite(args.max_seconds) and 0 < args.max_seconds <= 280, 'Invalid gradient deadline')
    deadline = started + min(args.max_seconds, remaining - 10)
    plan = read(args.plan)
    validate_plan(plan, 'preflight' if args.phase == 'preflight' else 'run')
    source_path = args.source_manifest or args.plan
    sources = read(source_path)['sources']
    require('src/riichi_ai/stage3h_gradients.py' in sources and 'scripts/stage3h_gradients.py' in sources,
            'Gradient source manifest membership missing')
    verify_sources(sources)
    require(digest(Path(plan['corpus_path']) / 'manifest.json') == plan['corpus_manifest_sha256']
            and digest(plan['hand_targets_path']) == plan['hand_targets_sha256'], 'Archived diagnostic input hash changed')
    dataset = LearningDataset(plan['corpus_path'], 'training')
    targets = HandTargets(plan['hand_targets_path'], dataset)
    groups = validate_groups(plan, dataset)
    models, conditions = checkpoint_records(plan)
    binding = identity(plan, models, conditions, groups)
    torch.set_num_threads(1)
    result = {'schema': 'riichi-ai-stage3h-gradient-' + ('preflight' if args.phase == 'preflight' else 'diagnostic') + '-v1',
        'complete': False, 'passed': False, 'phase': args.phase, 'device': args.device,
        'config_sha256': config_sha, 'plan_sha256': digest(args.plan), 'source_manifest_sha256': digest(source_path),
        'input_identity': binding, 'optimizer_steps': 0, 'new_matches': 0, 'checkpoint_writes': 0,
        'policy_promotion': False, 'interpretation_threshold': EPSILON,
        'clip_factor_definition': 'min(1, 1/(combined_gradient_l2+1e-6)), matching archived clip_grad_norm_ arithmetic',
        'versions': {'torch': version('torch'), 'numpy': version('numpy')}}
    require(args.device in ('cpu', 'mps') or args.device.startswith('cuda'), 'Unsupported gradient backend')
    if args.device == 'mps':
        require(torch.backends.mps.is_available(), 'MPS unavailable: diagnostic admission must be reconsidered')
    elif args.device.startswith('cuda'):
        require(torch.cuda.is_available(), 'CUDA unavailable')
    if args.phase == 'preflight':
        require(args.device != 'cpu', 'Preflight must compare CPU against a declared accelerator')
        snapshots = []
        for device in ('cpu', args.device):
            model, metadata = load_checkpoint(models[conditions[0]['checkpoint_id']], device=device)
            snapshot = gradient_group(model, dataset, groups[0], target='terminal_utility', device=device, deadline=deadline)
            snapshots.append(snapshot)
            del model
            if device == 'mps':
                torch.mps.empty_cache()
        comparison = compare_backends(snapshots[0], snapshots[1])
        result.update(complete=True, passed=comparison['passed'], backend_comparison=comparison,
            groups=[public_group(value) for value in snapshots], parameter_layout=snapshots[0]['layout'],
            projected_formal_compute_seconds_24_groups=snapshots[1]['elapsed_seconds'] * 24,
            projection_note='One fixed group is a feasibility measurement, not guaranteed later throughput; model loads and vector persistence are additional.')
    else:
        previous_path = Path(config['reports_root']) / 'gradient-preflight.json'
        previous = read(previous_path)
        require(previous['passed'] is True and previous['complete'] is True and previous['input_identity'] == binding
                and previous['device'] == args.device, 'Matching successful frozen gradient preflight required')
        vectors = Path(args.vectors_dir).resolve()
        require(vectors.is_relative_to(Path(config['data_root'])) and vectors != Path(config['data_root'])
                and not vectors.exists(), 'Raw vectors require a fresh directory in Stage3H external evidence')
        vectors.mkdir(parents=True)
        completed, layout = [], None
        for condition in conditions:
            check_deadline(deadline)
            model, metadata = load_checkpoint(models[condition['checkpoint_id']], device=args.device)
            observations = []
            for group_index, group in enumerate(groups):
                observation = gradient_group(model, dataset, group, target=condition['target'], device=args.device,
                    deadline=deadline, hand_targets=targets if condition['target'] == 'one_hand_delta' else None)
                layout = observation['layout'] if layout is None else layout
                require(layout == observation['layout'], 'Parameter layout changed across checkpoints')
                observations.append(observation)
                print(json.dumps({'phase': 'gradients', 'condition': condition['id'], 'group': group_index,
                    'optional_rows': observation['optional_rows'], 'elapsed_seconds': observation['elapsed_seconds']}), flush=True)
            payload = io.BytesIO()
            np.savez(payload, mse=np.stack([item['mse_gradient'] for item in observations]).astype(np.float32),
                     ce=np.stack([item['ce_gradient'] for item in observations]).astype(np.float32))
            raw = payload.getvalue()
            require(condition['id'] and Path(condition['id']).name == condition['id'], 'Unsafe condition ID')
            path = vectors / (condition['id'] + '.npz')
            write_new(path, raw)
            item = {**condition, 'checkpoint_sha256': models[condition['checkpoint_id']]['sha256'],
                'groups': [public_group(value) for value in observations],
                'summary': describe_groups(observations, observations[0]['ranges']),
                'vectors': {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
                    'dtype': 'float32', 'shape_per_component': [4, layout['trainable_parameters']],
                    'array_keys': ['mse', 'ce'], 'parameter_layout_sha256': hashlib.sha256(json_bytes(layout)).hexdigest()}}
            completed.append(item)
            # Preserve an immutable completed-condition prefix if a later deadline fails.
            write_new(vectors / (condition['id'] + '.json'), json_bytes(item))
            del model, observations
            if args.device == 'mps':
                torch.mps.empty_cache()
        result.update(complete=True, passed=True, conditions=completed, parameter_layout=layout,
            preflight_report_sha256=digest(previous_path),
            limitations=['Fixed64-match descriptive gradient sample, not population signal-to-noise or causal evidence',
                'MSE and chosen-action CE decomposed at archived parameters; no optimizer update, new policy or competitive evaluation',
                'Only targets vary between the two identical-initialization conditions; final-model comparisons combine differing learned weights and targets'])
    verify_sources(sources)
    result['elapsed_seconds'] = time.monotonic() - started
    check_deadline(deadline)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['preflight', 'formal'])
    parser.add_argument('--plan', required=True)
    parser.add_argument('--source-manifest')
    parser.add_argument('--output', required=True)
    parser.add_argument('--vectors-dir')
    parser.add_argument('--device', default='mps')
    parser.add_argument('--max-seconds', type=float, default=280.)
    args = parser.parse_args()
    require(not Path(args.output).exists(), 'Gradient report is immutable')
    require(args.phase != 'formal' or args.vectors_dir is not None, 'Formal diagnostics require raw-vector destination')
    started = time.monotonic()
    try:
        result = run(args)
    except Exception as error:
        result = {'schema': 'riichi-ai-stage3h-gradient-failure-v1', 'complete': False, 'passed': False,
            'phase': args.phase, 'failure': {'type': type(error).__name__, 'message': str(error)},
            'elapsed_seconds': time.monotonic() - started, 'optimizer_steps': 0, 'new_matches': 0, 'checkpoint_writes': 0}
    write_new(args.output, json_bytes(result))
    print(json.dumps({'complete': result['complete'], 'passed': result['passed'], 'elapsed_seconds': result['elapsed_seconds']}), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
