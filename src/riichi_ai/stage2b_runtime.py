"""Bounded, audited inference-only runtime measurement for Stage 2B.

Every submitted match has a distinct precommitted deterministic seed. Completed
matches retain compact identities, not a training corpus. Setup and drain count
against the shared wall-time ceiling; failed/truncated work is never promoted.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time
from typing import Any

import psutil

from .evaluation import _evaluation_manifest, _record_matches_frozen_panel, _seed, run_match, source_manifest
from .policies import shanten_cache_info


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def tree_rss() -> int:
    process = psutil.Process()
    total = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total


def corpus_and_historical_wall_seeds(plan: dict) -> set[int]:
    """Check actual recorded seed values as well as distinct namespace roots."""
    project = Path(__file__).resolve().parents[2]
    paths = [Path(plan['stage2a_dataset']) / 'manifest.json', Path(plan['data_root']) / 'dataset/manifest.json',
             Path(plan['data_root']) / 'aggregation/manifest.json']
    paths += [path for name in ('stage1', 'stage2') for path in (project / 'reports' / name).glob('*.json')]
    result: set[int] = set()
    for path in paths:
        if not path.exists():
            continue
        value = json.loads(path.read_text())
        for match in value.get('matches', []) if isinstance(value.get('matches', []), list) else []:
            if isinstance(match, dict) and isinstance(match.get('seed'), int):
                result.add(match['seed'])
    return result


def _init_worker() -> None:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _benchmark_match(seed: int, checkpoint: str, expected_hash: str, failure_dir: str) -> dict:
    before = shanten_cache_info()._asdict()
    result = run_match(seed, ['checkpoint:' + checkpoint] * 4, failure_dir=failure_dir)
    keep = ('seed', 'completed', 'stop_reason', 'failure', 'failure_record', 'digest',
            'steps', 'decisions', 'hands', 'scores', 'ranks', 'trajectory_hash',
            'player_views_hash', 'auditor_log_hash', 'conservation_ok', 'timing_seconds',
            'policy_inference_by_seat', 'policy_inference_timing_by_seat', 'elapsed_seconds',
            'setup_seconds', 'finalization_seconds', 'unallocated_loop_seconds')
    compact = {key: result[key] for key in keep if key in result}
    compact['loaded_policies'] = [{key: policy.get(key) for key in
                                  ('policy', 'version', 'feature_schema', 'checkpoint_sha256', 'weights_sha256')}
                                 for policy in result['policies']]
    if any(policy['checkpoint_sha256'] != expected_hash for policy in compact['loaded_policies']):
        compact.update(completed=False, stop_reason='policy_identity_mismatch',
                       failure={'type': 'PolicyIdentityMismatch', 'message': 'Loaded model differs from frozen checkpoint'})
    compact.update(worker_pid=os.getpid(), cache_before=before,
                   cache_after=shanten_cache_info()._asdict(), worker_rss=psutil.Process().memory_info().rss)
    return compact


def verify_paired_identity(result: dict, *, checkpoint: str, expected_hash: str, plan: dict,
                           root_seed: int, blocks: int) -> None:
    """Bind resumed or newly produced evidence to this exact frozen protocol."""
    configuration = result.get('configuration', {})
    expected = {'candidate': _evaluation_manifest('checkpoint:' + checkpoint),
                'baseline': _evaluation_manifest(plan['evaluation']['baseline']),
                'opponents_clockwise': [_evaluation_manifest(name) for name in plan['opponents']],
                'blocks': blocks, 'root_seed': root_seed, 'max_steps': 100000,
                'workers': plan['evaluation']['workers'], 'utility_by_rank': plan['utility'], 'matches_per_block': 8}
    if (configuration != expected or configuration['candidate'].get('checkpoint_sha256') != expected_hash
            or result.get('rule_profile_hash') != plan['rule_profile_hash']
            or result.get('source_manifest') != source_manifest()
            or not result.get('valid') or result.get('completed_games') != blocks * 8
            or len(result.get('matches', [])) != blocks * 8 or len(result.get('blocks', [])) != blocks):
        raise ValueError('Paired evaluation identity/count differs from the frozen protocol')
    for match in result['matches']:
        if (match['seed'] != _seed(root_seed, 'block', match['block']) or not match['completed']
                or not _record_matches_frozen_panel(match, [expected['candidate'], expected['baseline'],
                                                            *expected['opponents_clockwise']], root_seed)):
            raise ValueError('Paired evaluation wall seed differs from plan')
    grid = {(match['block'], match['focal_seat'], match['arm']) for match in result['matches']}
    if grid != {(block, seat, arm) for block in range(blocks) for seat in range(4) for arm in ('candidate', 'baseline')}:
        raise ValueError('Paired evaluation seat/arm/block grid is incomplete or duplicated')
    if {block['block'] for block in result['blocks']} != set(range(blocks)):
        raise ValueError('Paired evaluation block IDs differ from plan')
    if any(not block['valid'] or block['wall_seed'] != _seed(root_seed, 'block', block['block'])
           for block in result['blocks']):
        raise ValueError('Paired evaluation block integrity failed')


def benchmark_trial(*, checkpoint: str, workers: int, duration: float, root_seed: int,
                    namespace: str, output: Path, deadline: float, memory_limit: int,
                    artifact_roots: list[Path], disk_limit: int, expected_hash: str = '',
                    forbidden_seeds: set[int] | None = None) -> dict:
    start = time.monotonic()
    stop_submitting = start + duration
    if stop_submitting + 30 > deadline:
        raise RuntimeError('Insufficient benchmark budget for bounded drain')
    records_path = output.with_suffix('.jsonl')
    if records_path.exists() or output.exists():
        raise FileExistsError('Benchmark evidence exists; use the recorded result, do not overwrite')
    output.parent.mkdir(parents=True, exist_ok=True)
    peak = tree_rss()
    windows, matches, decisions, hands, failures = [], 0, 0, 0, 0
    last_window = start
    window_counts = [0, 0, 0]
    total_timings: dict[str, float] = {}
    total_inference: dict[str, int] = {}
    feature_timings: dict[str, float] = {}
    stopping_reason = 'duration'
    submitted = 0
    max_rss_worker = 0
    cache_last: dict[str, dict] = {}
    sources = source_manifest()
    model_hash = file_hash(checkpoint)
    if model_hash != expected_hash:
        raise ValueError('Trial checkpoint differs from the campaign frozen checkpoint')
    # Persist all prerequisites before outcomes exist.
    save_json(output.with_suffix('.plan.json'), {
        'checkpoint': checkpoint, 'checkpoint_sha256': model_hash, 'workers': workers,
        'duration_seconds': duration, 'root_seed': root_seed, 'namespace': namespace,
        'seed_formula': 'evaluation._seed(root_seed, namespace, zero_based_submission)',
        'source_manifest': sources, 'cpu_threads_per_worker': 1,
        'all_seats': 'frozen candidate', 'inference_only': True,
    })
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                               initializer=_init_worker)
    pending: dict[Any, tuple[int, int]] = {}
    terminated = False
    runner_failure = None
    last_resource = start - 10
    try:
        with records_path.open('x') as stream:
            while True:
                now = time.monotonic()
                if now >= deadline - 3:
                    stopping_reason = 'aggregate_deadline'
                    terminated = True
                    break
                if now - last_resource >= 5:
                    rss = tree_rss()
                    peak = max(peak, rss)
                    size = sum(p.stat().st_size for root in artifact_roots for p in root.rglob('*') if p.is_file())
                    if rss > memory_limit or size > disk_limit:
                        stopping_reason = 'memory_budget' if rss > memory_limit else 'artifact_budget'
                        terminated = True
                        break
                    last_resource = now
                while len(pending) < workers and now < stop_submitting:
                    seed = _seed(root_seed, namespace, submitted)
                    if seed in (forbidden_seeds or set()):
                        raise ValueError('Benchmark wall seed overlaps recorded or planned non-benchmark work')
                    if forbidden_seeds is not None:
                        forbidden_seeds.add(seed)
                    future = pool.submit(_benchmark_match, seed, checkpoint, model_hash, str(output.parent / 'failures'))
                    pending[future] = (submitted, seed)
                    submitted += 1
                if not pending:
                    break
                done, _ = wait(pending, timeout=min(1.0, max(.01, deadline - now - 3)), return_when=FIRST_COMPLETED)
                for future in done:
                    index, seed = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception as error:
                        result = {'seed': seed, 'completed': False, 'failure': {'type': type(error).__name__, 'message': str(error)}}
                    result['submission_index'] = index
                    result['completion_elapsed_seconds'] = time.monotonic() - start
                    stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + '\n')
                    stream.flush()
                    if not result['completed']:
                        failures += 1
                        stop_submitting = 0
                        stopping_reason = 'match_failure'
                    else:
                        matches += 1
                        decisions += result['decisions']
                        hands += result['hands']
                        window_counts[0] += 1
                        window_counts[1] += result['decisions']
                        window_counts[2] += result['hands']
                    max_rss_worker = max(max_rss_worker, result.get('worker_rss', 0))
                    if 'cache_after' in result:
                        cache_last[str(result['worker_pid'])] = result['cache_after']
                    for key, value in result.get('timing_seconds', {}).items():
                        total_timings[key] = total_timings.get(key, 0.0) + value
                    for values in result.get('policy_inference_by_seat', []):
                        for key, value in (values or {}).items():
                            total_inference[key] = total_inference.get(key, 0) + value
                    for values in result.get('policy_inference_timing_by_seat', []):
                        for key, value in (values or {}).items():
                            feature_timings[key] = feature_timings.get(key, 0.0) + value
                now = time.monotonic()
                if now - last_window >= 60:
                    span = now - last_window
                    window = {'elapsed_seconds': now - start, 'window_seconds': span,
                              'completed_matches': window_counts[0], 'decisions': window_counts[1],
                              'hands': window_counts[2], 'decisions_per_second': window_counts[1] / span,
                              'process_tree_rss': tree_rss(), 'pending_matches': len(pending)}
                    windows.append(window)
                    print(json.dumps({'stage': namespace, **window}), flush=True)
                    save_json(output.with_suffix('.progress.json'), {'windows': windows, 'completed_matches': matches,
                                                                     'peak_process_tree_rss': peak})
                    last_window = now
                    window_counts = [0, 0, 0]
    except Exception as error:
        terminated = True
        stopping_reason = 'runner_failure'
        runner_failure = {'type': type(error).__name__, 'message': str(error)}
    finally:
        if terminated:
            pool.terminate_workers()
        else:
            pool.shutdown(wait=True)
    elapsed = time.monotonic() - start
    span = time.monotonic() - last_window
    if span > 0:
        windows.append({'elapsed_seconds': elapsed, 'window_seconds': span, 'completed_matches': window_counts[0],
                        'decisions': window_counts[1], 'hands': window_counts[2],
                        'decisions_per_second': window_counts[1] / span, 'partial_window': True})
    report = {'schema': 'riichi-ai-stage2b-runtime-v1', 'checkpoint_sha256': model_hash,
              'workers': workers, 'requested_duration_seconds': duration, 'elapsed_seconds': elapsed,
              'source_manifest': sources, 'sources_unchanged': sources == source_manifest(),
              'checkpoint_unchanged': file_hash(checkpoint) == model_hash, 'stop_reason': stopping_reason,
              'runner_failure': runner_failure,
              'submitted_matches': submitted, 'completed_matches': matches, 'failed_matches': failures,
              'unresolved_matches': [dict(submission_index=i, seed=s) for i, s in pending.values()],
              'decisions': decisions, 'hands': hands, 'decisions_per_second': decisions / elapsed,
              'matches_per_second': matches / elapsed, 'peak_process_tree_rss': peak,
              'maximum_worker_rss': max_rss_worker, 'cache_final_by_worker': cache_last,
              'summed_worker_timing_seconds': total_timings, 'inference_counts': total_inference,
              'summed_worker_inference_timing_seconds': feature_timings, 'windows': windows,
              'records': str(records_path), 'records_sha256': file_hash(records_path),
              'inference_only': True,
              'limitations': 'Audited CPU runtime, including hashing and simulator work; not RL throughput. Windows credit whole completed games at completion time, not work executed precisely within each minute. Process-tree RSS sums shared pages conservatively. Cache state and thermal conditions affect throughput. Timing phases may include validation and transfer overhead; cumulative worker seconds differ from wall seconds.'}
    report['valid'] = (not failures and not pending and stopping_reason == 'duration' and
                       matches > 0 and report['sources_unchanged'] and report['checkpoint_unchanged'])
    save_json(output, report)
    print(json.dumps({'stage': namespace, 'valid': report['valid'], 'completed_matches': matches,
                      'workers': workers, 'decisions_per_second': decisions / elapsed,
                      'elapsed_seconds': elapsed}), flush=True)
    return report


def benchmark_campaign(plan: dict, selection: dict, reports: Path) -> dict:
    started = time.monotonic()
    deadline = started + plan['limits']['benchmark_total_seconds']
    checkpoint = selection['checkpoint']
    if file_hash(checkpoint) != selection['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint hash mismatch')
    trials = []
    options = plan['benchmark']
    forbidden_seeds = corpus_and_historical_wall_seeds(plan)
    for label, count in (('development', plan['evaluation']['pilot_blocks']),
                         ('confirmation', plan['limits']['confirmation_max_blocks'])):
        forbidden_seeds.update(_seed(plan['seed_namespaces'][label], 'block', block) for block in range(count))
    common = dict(checkpoint=checkpoint, expected_hash=selection['checkpoint_sha256'],
                  root_seed=plan['seed_namespaces']['benchmark'], deadline=deadline,
                  forbidden_seeds=forbidden_seeds,
                  memory_limit=plan['limits']['memory_gib'] * 1024**3,
                  artifact_roots=[Path(plan['data_root']), reports], disk_limit=plan['limits']['new_artifact_bytes'])
    for workers in options['worker_trials']:
        result = benchmark_trial(**common, workers=workers, duration=options['short_trial_seconds'],
                                 namespace=f'worker-trial-{workers}', output=reports / f'benchmark-workers-{workers}.json')
        trials.append(result)
        if not result['valid']:
            raise RuntimeError('Benchmark resource/integrity check failed; evidence retained')
        if len(trials) > 1 and result['decisions_per_second'] <= trials[-2]['decisions_per_second']:
            break
    best = max(trials, key=lambda r: r['decisions_per_second'])
    # This choice uses runtime only, never game strength or training/test rows.
    save_json(reports / 'benchmark-worker-selection.json', {
        'workers': best['workers'], 'criterion': 'maximum completed audited decisions per total wall second',
        'trial_workers': [r['workers'] for r in trials], 'checkpoint_sha256': selection['checkpoint_sha256']})
    sustained = benchmark_trial(**common, workers=best['workers'], duration=options['sustained_seconds'],
                                namespace='sustained', output=reports / 'benchmark-sustained.json')
    report = {'valid': sustained['valid'], 'elapsed_seconds': time.monotonic() - started,
              'limit_seconds': plan['limits']['benchmark_total_seconds'], 'selected_workers': best['workers'],
              'sustained_decisions_per_second': sustained['decisions_per_second'],
              'sustained_completed_matches': sustained['completed_matches'],
              'peak_process_tree_rss': max(r['peak_process_tree_rss'] for r in trials + [sustained]),
              'trials': [str(reports / f"benchmark-workers-{r['workers']}.json") for r in trials],
              'sustained': str(reports / 'benchmark-sustained.json')}
    report['within_budget'] = report['elapsed_seconds'] <= report['limit_seconds']
    report['valid'] &= report['within_budget']
    save_json(reports / 'benchmark-summary.json', report)
    return report
