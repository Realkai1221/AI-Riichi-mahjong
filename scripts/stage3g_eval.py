"""Execute the complete frozen Stage 3G paired development panel once."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
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


def write_new(path, value):
    from riichi_ai.stage3g_collect import _write
    from riichi_ai.stage3g_eval import json_bytes
    _write(Path(path), json_bytes(value))


def require_lease(config, max_seconds):
    active = [op for op in read(Path(config['data_root']) / 'budget-ledger.json')['operations'] if op['status'] == 'active']
    if (len(active) != 1 or active[0]['name'] != 'development'
            or os.environ.get('RIICHI_STAGE3G_OPERATION') != active[0]['name']
            or os.environ.get('RIICHI_STAGE3G_TOKEN') != active[0]['token']):
        raise ValueError('Development requires its matching active numerical lease')
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError('Invalid cooperative deadline')
    remaining = active[0]['reserved_seconds'] - (time.time() - active[0]['started_unix'])
    if remaining <= 15:
        raise TimeoutError('Insufficient development lease remains')
    return time.monotonic() + min(max_seconds, remaining - 12)


def worker(item):
    from riichi_ai.stage3g_eval import run_one
    if time.monotonic() >= item['deadline']:
        raise TimeoutError('No new development match may start after deadline')
    return run_one(item['job'], item['execution'], item['output'], deadline=item['deadline'])


def run(execution, output, config, deadline):
    from riichi_ai.stage3g_eval import summarize
    root = Path(config['data_root'])
    starts = sum(1 for _ in root.glob('**/lifecycle/*.started.json'))
    if starts + execution['planned_matches'] > config['max_match_executions']:
        raise ValueError('Development would exceed the global match-execution ceiling')
    if time.monotonic() >= deadline:
        raise TimeoutError('Expired development admission')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / 'execution-plan.json', execution)
    jobs = iter(dict(job=job, execution=execution, output=str(output), deadline=deadline)
                for job in execution['matches'])
    started, failure, finished = time.perf_counter(), None, 0
    try:
        with ProcessPoolExecutor(max_workers=14, mp_context=multiprocessing.get_context('spawn')) as pool:
            pending = deque()
            for _ in range(14):
                if time.monotonic() >= deadline:
                    raise TimeoutError('No new development submission after deadline')
                job = next(jobs, None)
                if job:
                    pending.append(pool.submit(worker, job))
            while pending:
                result = pending.popleft().result()
                finished += 1
                if not result['completed']:
                    raise RuntimeError('A development match failed: preserve, stop, never replace')
                if finished % 64 == 0 or finished == execution['planned_matches']:
                    print(json.dumps(dict(phase='development', completed=finished, planned=execution['planned_matches'])), flush=True)
                if time.monotonic() >= deadline:
                    raise TimeoutError('No new development submission after deadline')
                job = next(jobs, None)
                if job:
                    pending.append(pool.submit(worker, job))
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error))
    # Include every durably finished in-flight job, even if an earlier job failed.
    records = [read(path) for path in sorted((output / 'reports').glob('block-*.json'))]
    summary = None
    if failure is None:
        try:
            summary = summarize(execution, records)
        except Exception as error:
            failure = dict(type=type(error).__name__, message=str(error))
    complete = failure is None and summary is not None
    manifest = dict(schema='riichi-ai-stage3g-development-v1', complete=complete,
                    execution_plan_sha256=digest(output / 'execution-plan.json'),
                    execution_plan=execution, matches=records, failure=failure,
                    statistics=summary, elapsed_seconds=time.perf_counter()-started,
                    training_eligible=False, ppo_eligible=False, policy_promotion=False)
    write_new(output / 'manifest.json', manifest)
    return dict(schema='stage3g-development-summary-v1', complete=complete, failure=failure,
                planned_games=execution['planned_matches'], completed_games=sum(r['completed'] for r in records),
                attempted_games=sum(1 for _ in (output / 'lifecycle').glob('*.started.json')),
                rows=sum(r['rows'] for r in records), elapsed_seconds=manifest['elapsed_seconds'],
                matches_per_second=len(records)/manifest['elapsed_seconds'],
                statistics=summary, manifest_path=str(output / 'manifest.json'),
                manifest_sha256=digest(output / 'manifest.json'),
                training_eligible=False, policy_promotion=False, confirmation_started=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--arms-manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--max-seconds', type=float, default=660)
    args = parser.parse_args()
    config, identity = read_config()
    deadline = require_lease(config, args.max_seconds)
    plan, arms = read(args.plan), read(args.arms_manifest)
    if plan.get('schema') != 'stage3g-fixed-learning-plan-v1' or plan.get('config_sha256') != identity:
        raise ValueError('Frozen development plan or configuration changed')
    verify_sources(plan['sources'])
    output, report = Path(args.output_dir).resolve(), Path(args.report).resolve()
    if (output == Path(config['data_root']) or not output.is_relative_to(Path(config['data_root']))
            or not report.is_relative_to(Path(config['reports_root'])) or report.exists()):
        raise ValueError('Output must be fresh and remain in the declared evidence roots')
    from riichi_ai.stage3g_eval import prepare_execution, validate_policies
    execution = prepare_execution(plan, arms, config, plan_sha256=digest(args.plan), arms_sha256=digest(args.arms_manifest))
    reports = Path(config['reports_root'])
    for phase in ['collect', 'grp-fit', *['train-'+arm['name'] for arm in execution['arms']]]:
        if read(reports / f'operation-{phase}.json')['status'] != 'completed':
            raise ValueError('Every preceding formal operation must have completed')
    execution['authenticated_policies'] = validate_policies(execution)
    result = run(execution, output, config, deadline)
    write_new(report, result)
    print(json.dumps(dict(complete=result['complete'], games=result['completed_games'],
                         selected=result['statistics']['selected_candidate'] if result['statistics'] else None)), flush=True)
    if not result['complete']:
        raise RuntimeError('Development panel incomplete: '+str(result['failure']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
