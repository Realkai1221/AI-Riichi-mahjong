#!/usr/bin/env python3
"""Leased Stage 3I frozen-feature value/residual training and preflight."""
from __future__ import annotations
import argparse
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch

from riichi_ai.stage3i_training import (FeatureCache, RECIPE, check_deadline, digest,
    json_bytes, preflight, require, train_seed, write_new)


def read(path):
    return json.loads(Path(path).read_text())


def identity_for(plan, source_sha):
    return {'reference_sha256': plan['reference_sha256'], 'call_sha256': plan['call_sha256'],
        'protocol_sha256': plan['protocol_sha256'], 'cache_manifest_sha256': plan['cache_manifest_sha256'],
        'recipe': RECIPE, 'source_manifest_sha256': source_sha}


def run(args):
    from stage3i import read_config, verify_sources
    config, config_sha = read_config()
    active = [item for item in read(Path(config['data_root']) / 'budget-ledger.json')['operations'] if item['status'] == 'active']
    require(len(active) == 1 and active[0]['name'] == os.environ.get('RIICHI_STAGE3I_OPERATION')
        and active[0]['token'] == os.environ.get('RIICHI_STAGE3I_TOKEN'), 'Matching active Stage 3I numerical lease required')
    require(active[0]['name'].startswith('preflight') if args.phase == 'preflight' else active[0]['name'] == 'training-formal',
            'Wrong numerical lease for training phase')
    started = time.monotonic()
    remaining = active[0]['reserved_seconds'] - (time.time() - active[0]['started_unix'])
    require(math.isfinite(args.max_seconds) and args.max_seconds > 0 and remaining > 12, 'Invalid training deadline')
    deadline = started + min(args.max_seconds, remaining - 10)
    plan = read(args.plan)
    require(plan.get('schema') == 'stage3i-plan-v1', 'Unsupported Stage 3I plan')
    require(plan['config_sha256'] == config_sha and digest(plan['protocol_path']) == plan['protocol_sha256'],
            'Frozen configuration or written protocol identity changed')
    require(args.phase == 'preflight' or plan.get('execution_admitted') is True, 'Formal training has not been measurement-admitted')
    require(plan['seeds'] == [20261024011, 20261024012, 20261024013], 'Frozen three-seed programme changed')
    source_path = args.source_manifest or args.plan
    sources = read(source_path)['sources']
    require({'src/riichi_ai/stage3i_training.py', 'scripts/stage3i_training.py', 'src/riichi_ai/stage3i_model.py'} <= set(sources),
            'Training source membership missing')
    verify_sources(sources)
    require(digest(Path(plan['corpus_path']) / 'manifest.json') == plan['corpus_manifest_sha256']
        and digest(plan['hand_targets_path']) == plan['hand_targets_sha256'], 'Historical corpus or target identity changed')
    for stem in ('reference', 'call'):
        require(digest(plan[stem + '_checkpoint']) == plan[stem + '_sha256'] == config[stem + '_sha256'],
                'Frozen historical actor checkpoint changed')
    identity = identity_for(plan, digest(source_path))
    output_dir = Path(args.output_dir).resolve()
    require(output_dir.is_relative_to(Path(config['data_root'])) and output_dir != Path(config['data_root'])
            and not output_dir.exists(), 'Training outputs require a fresh external evidence directory')
    output_dir.mkdir(parents=True)
    require(args.device in ('cpu', 'mps') or args.device.startswith('cuda'), 'Unsupported training backend')
    if args.device == 'mps':
        require(torch.backends.mps.is_available(), 'MPS unavailable')
    elif args.device.startswith('cuda'):
        require(torch.cuda.is_available(), 'CUDA unavailable')
    torch.set_num_threads(1)
    cache_started = time.monotonic()
    training = FeatureCache(plan['cache_manifest_path'], 'training', expected_sha256=plan['cache_manifest_sha256'], deadline=deadline)
    selection = FeatureCache(plan['cache_manifest_path'], 'selection', expected_sha256=plan['cache_manifest_sha256'], deadline=deadline)
    for cache in (training, selection):
        require(all(cache.manifest[key] == plan[key] for key in ('reference_sha256', 'call_sha256',
            'corpus_manifest_sha256', 'hand_targets_sha256')), 'Cached feature/target historical identities changed')
        require(cache.manifest.get('complete') is True and cache.manifest.get('labels_are_actor_inputs') is False,
                'Incomplete or role-unsafe cache manifest')
        feature_sources = cache.manifest['feature_sources']
        require(isinstance(feature_sources, dict) and bool(feature_sources)
            and all(sources.get(name) == sha for name, sha in feature_sources.items()),
            'Feature preparation sources differ from the frozen training programme')
        verify_sources(feature_sources)
    expected_parts = plan['partitions']
    if args.phase == 'preflight':
        require(training.manifest.get('purpose') == 'preflight-only'
            and len(training.match_ids) == 16 and len(selection.match_ids) == 16,
            'Preflight requires the explicitly isolated sixteen-match training/selection cache')
        expected_parts = {key: (value[:16] if key in ('training', 'selection') else [])
                          for key, value in expected_parts.items()}
    else:
        require(training.manifest.get('purpose') == 'formal'
            and {key: len(value) for key, value in training.manifest['partitions'].items()}
                == {'training': 3200, 'selection': 200, 'calibration': 200, 'gate': 400},
            'Fixed whole-match cache partition membership changed')
        require(training.rows == 471627, 'Explorer training population changed')
    require(training.manifest['partitions'] == expected_parts, 'Cache matches differ from the frozen whole-match partition plan')
    cache_seconds = time.monotonic() - cache_started
    result = {'schema': 'riichi-ai-stage3i-training-' + ('preflight' if args.phase == 'preflight' else 'report') + '-v1',
        'phase': args.phase, 'complete': False, 'passed': False, 'identity': identity,
        'config_sha256': config_sha, 'plan_sha256': digest(args.plan), 'source_manifest_sha256': digest(source_path),
        'device': args.device, 'training_cache': training.identity(), 'selection_cache': selection.identity(),
        'cache_load_seconds': cache_seconds, 'new_training_matches': 0, 'policy_promotion': False,
        'versions': {'torch': version('torch'), 'numpy': version('numpy')},
        'frozen_actor_checkpoint_sha256_before': plan['reference_sha256'],
        'semantics': 'Supervised decision-weighted regression of recorded explorer outcomes; no PPO and no causal action-advantage claim'}
    if args.phase == 'preflight':
        result.update(preflight(training, selection, seed=plan['seeds'][0], device=args.device,
            deadline=deadline, output_dir=output_dir, identity=identity, benchmark_seeds=plan['seeds']))
    else:
        previous = read(plan['training_preflight_path'])
        require(previous.get('passed') is True and previous.get('complete') is True
            and previous['identity']['reference_sha256'] == identity['reference_sha256']
            and previous['identity']['call_sha256'] == identity['call_sha256']
            and previous['identity']['cache_manifest_sha256'] == plan['preflight_cache_manifest_sha256']
            and previous['identity']['recipe'] == RECIPE
            and args.device in previous['timings'], 'Matching successful real-optimizer preflight required')
        require(plan['epochs'] in (4, 8, 12), 'Unadmitted training epoch count')
        seeds = []
        for seed in plan['seeds']:
            check_deadline(deadline)
            seeds.append(train_seed(training, selection, seed=seed, epochs=plan['epochs'],
                device=args.device, deadline=deadline, output_dir=output_dir / f'seed-{seed}', identity=identity,
                emit=lambda value: print(json.dumps(value), flush=True)))
        result.update(complete=True, passed=True, seeds=seeds, epochs=plan['epochs'],
            optimizer_updates=sum(item['optimizer_updates'] for item in seeds),
            row_presentations=sum(item['row_presentations'] for item in seeds),
            training_preflight_sha256=digest(plan['training_preflight_path']))
    verify_sources(sources)
    require(digest(plan['protocol_path']) == plan['protocol_sha256'], 'Written protocol changed during training')
    for stem in ('reference', 'call'):
        require(digest(plan[stem + '_checkpoint']) == plan[stem + '_sha256'], 'Training changed an archived actor checkpoint')
    require(digest(plan['cache_manifest_path']) == plan['cache_manifest_sha256'], 'Feature manifest changed during training')
    result['frozen_actor_checkpoint_sha256_after'] = digest(plan['reference_checkpoint'])
    result['elapsed_seconds'] = time.monotonic() - started
    check_deadline(deadline)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['preflight', 'run'])
    parser.add_argument('--plan', required=True)
    parser.add_argument('--source-manifest')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--max-seconds', required=True, type=float)
    args = parser.parse_args()
    output = Path(args.output_dir) / ('preflight.json' if args.phase == 'preflight' else 'training-report.json')
    try:
        result = run(args)
    except Exception as error:
        # The controller records failure and stops later formal phases. Preserve
        # completed immutable epoch checkpoints for diagnosis, never evaluation.
        if Path(args.output_dir).is_dir() and not output.exists():
            write_new(output, json_bytes({'schema': 'riichi-ai-stage3i-training-failure-v1',
                'complete': False, 'passed': False, 'phase': args.phase,
                'error_type': type(error).__name__, 'error': str(error), 'policy_promotion': False}))
        raise
    write_new(output, json_bytes(result))
    print(json.dumps({'output': str(output.resolve()), 'complete': result['complete'],
                      'passed': result['passed'], 'elapsed_seconds': result['elapsed_seconds']}), flush=True)
    if not result['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
