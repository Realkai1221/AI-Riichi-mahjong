"""Fixed fresh corpus and resource admission for Stage 3G."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from stage3g import read_config, digest, verify_sources


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def lease(config):
    operations = read(Path(config['data_root']) / 'budget-ledger.json')['operations']
    active = [op for op in operations if op['status'] == 'active']
    if len(active) != 1 or active[0]['token'] != os.environ.get('RIICHI_STAGE3G_TOKEN') or active[0]['name'] != os.environ.get('RIICHI_STAGE3G_OPERATION'):
        raise ValueError('Exclusive Stage 3G numerical lease required')


def seed_plan(config):
    from riichi_ai.stage3g_collect import derive_plan
    if (config['corpus_matches'], config['training_matches'], config['development_matches'], config['preflight_matches']) != (4000, 3200, 800, 16):
        raise ValueError('Fixed corpus counts changed')
    corpus = [derive_plan(config['root_seed'], i, 'training' if i < 3200 else 'development') for i in range(4000)]
    preflight = [derive_plan(config['preflight_seed'], i, 'preflight') for i in range(16)]
    evaluation = [int(hashlib.sha256(json.dumps(['stage3g-evaluation-v1', config['evaluation_seed'], i]).encode()).hexdigest()[:16], 16) for i in range(64)]
    fresh = [p['seed'] for p in corpus + preflight] + [s for p in corpus + preflight for s in p['sampling_seeds']] + evaluation
    if len(set(fresh)) != len(fresh):
        raise ValueError('New seed collision')
    old = set()
    def walk(value, key=''):
        if isinstance(value, dict):
            for k, v in value.items():
                walk(v, k)
        elif isinstance(value, list):
            for v in value:
                walk(v, key)
        elif type(value) is int and 'seed' in key:
            old.add(value)
    for path in list((ROOT / 'configs').glob('*.json')) + list((ROOT / 'reports').rglob('plan.json')):
        if 'stage3g' not in path.parts and path.name != 'stage3g.json':
            walk(read(path))
    if old.intersection(fresh):
        raise ValueError('Fresh seeds overlap screened historical seed fields')
    return dict(schema='stage3g-seed-plan-v1', config_sha256=digest(ROOT / 'configs/stage3g.json'),
                corpus=corpus, preflight=preflight, evaluation_seeds=evaluation,
                corpus_replay_indices=list(range(0, 4000, 100)),
                historical_seed_disjoint=True,
                historical_seed_check_scope='Integer seed fields in project configurations and historical reports/**/plan.json, not every historical trajectory')


def source_spec():
    paths = set(read(ROOT / 'reports/stage3f/plan.json')['sources'])
    paths.update(str(p.relative_to(ROOT)) for pattern in ('scripts/stage3g*.py', 'src/riichi_ai/stage3g*.py', 'tests/test_stage3g*.py') for p in ROOT.glob(pattern))
    paths.update(('configs/stage3g.json', 'reports/stage3g/seed-plan.json', 'tests/test_stage3c_postcall.py', 'scripts/audit_stage3g_corpus.py'))
    return {p: digest(ROOT / p) for p in sorted(paths)}


def worker(job):
    import torch
    from riichi_ai.stage3g_collect import run_one
    torch.set_num_threads(1)
    if time.monotonic() >= job['deadline']:
        raise TimeoutError('No new match after deadline')
    c = job['config']
    return run_one(job['plan'], c['reference_checkpoint'], c['reference_sha256'], c['call_checkpoint'],
                   job['output'], expected_call_sha256=c['call_sha256'], deadline=job['deadline'])


def collect(config, plans, phase, deadline, plan_hash):
    root = Path(config['data_root'])
    if len(list(root.glob('**/lifecycle/*.started.json'))) + len(plans) > config['max_match_executions']:
        raise ValueError('Execution count exceeds cap')
    output = root / phase
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    failure = None
    finished = 0
    jobs = iter(dict(config=config, plan=p, deadline=deadline, output=str(output)) for p in plans)
    try:
        with ProcessPoolExecutor(max_workers=config['workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
            pending = deque()
            for _ in range(config['workers']):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Collection deadline')
                job = next(jobs, None)
                if job:
                    pending.append(pool.submit(worker, job))
            while pending:
                result = pending.popleft().result()
                finished += 1
                if not result['completed']:
                    raise RuntimeError('Incomplete match: stop, preserve, never replace')
                if finished % 128 == 0 or finished == len(plans):
                    print(json.dumps(dict(phase=phase, completed=finished, planned=len(plans))), flush=True)
                if time.monotonic() >= deadline:
                    raise TimeoutError('Collection deadline')
                job = next(jobs, None)
                if job:
                    pending.append(pool.submit(worker, job))
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error))
    records = [read(path) for path in sorted((output / 'reports').glob('match-*.json'))]
    complete = failure is None and len(records) == len(plans) and all(r['completed'] for r in records)
    splits = {role: [r['match_id'] for r in records if r['role'] == role] for role in sorted({p['role'] for p in plans})}
    manifest = dict(schema='riichi-ai-stage3g-corpus-v1', source_role='offline-learning-corpus', complete=complete,
                    config_sha256=digest(ROOT / 'configs/stage3g.json'), plan_sha256=plan_hash,
                    phase=phase, planned=plans, matches=records, splits=splits, failure=failure,
                    workers=config['workers'], ppo_eligible=False, elapsed_seconds=time.perf_counter() - start)
    write(output / 'manifest.json', manifest)
    summary = dict(schema='stage3g-collection-summary-v1', complete=complete, failure=failure,
                   phase=phase, planned_matches=len(plans), completed_matches=sum(r['completed'] for r in records),
                   rows=sum(r['rows'] for r in records), elapsed_seconds=manifest['elapsed_seconds'],
                   matches_per_second=len(records) / manifest['elapsed_seconds'],
                   compressed_artifact_bytes=sum(a['bytes'] for r in records for a in r['artifacts'].values()),
                   uncompressed_array_bytes=sum(r['uncompressed_actor_bytes'] + r['uncompressed_target_bytes'] for r in records),
                   manifest_path=str(output / 'manifest.json'), manifest_sha256=digest(output / 'manifest.json'))
    write(Path(config['reports_root']) / (phase + '-collection.json'), summary)
    if not complete:
        raise RuntimeError('Formal collection incomplete: ' + str(failure))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'preflight', 'freeze', 'collect'])
    parser.add_argument('--max-seconds', type=float, default=850)
    args = parser.parse_args()
    config, _ = read_config()
    lease(config)
    reports = Path(config['reports_root'])
    if args.phase == 'prepare':
        write(reports / 'seed-plan.json', seed_plan(config))
        result = dict(prepared=True, corpus=4000, preflight=16, development_blocks=64)
    elif args.phase == 'freeze':
        plan = read(reports / 'seed-plan.json')
        plan.update(schema='stage3g-fixed-learning-plan-v1', sources=source_spec(),
                    optimizer_updates_per_arm=400, epochs=2, arms=config['arms'],
                    max_match_executions=config['max_match_executions'], policy_promotion=False)
        write(reports / 'plan.json', plan)
        result = dict(frozen=True, sources=len(plan['sources']))
    else:
        formal = args.phase == 'collect'
        path = reports / ('plan.json' if formal else 'seed-plan.json')
        plan = read(path)
        if formal:
            verify_sources(plan['sources'])
            receipt = read(reports / 'operation-preflight-release.json')
            if receipt['status'] != 'completed' or any(receipt['sources'].get(k) != v for k, v in plan['sources'].items()):
                raise ValueError('Source-bound release preflight required')
            admission = read(reports / 'admission.json')
            if not admission['valid']:
                raise ValueError('Fixed experiment did not pass resource admission')
        result = collect(config, plan['corpus' if formal else 'preflight'], 'corpus' if formal else 'preflight',
                         time.monotonic() + args.max_seconds, digest(path))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
