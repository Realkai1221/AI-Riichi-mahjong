"""One fixed confirmation using unchanged Stage 2B gameplay/evaluation code.

The parent imposes a wall/resource deadline without changing the evaluator.
No model outcome is inspected until the fixed campaign returns. A killed or
incomplete campaign cannot pass and cannot be silently restarted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def prepare(config_path):
    from riichi_ai.evaluation import _seed, source_manifest
    from riichi_ai.stage2b_runtime import corpus_and_historical_wall_seeds
    config = json.loads(config_path.read_text())
    settings = config['confirmation']
    reports = PROJECT / settings['reports_root']
    if (reports / 'plan.json').exists():
        raise FileExistsError('A1 is already frozen/attempted; inspect its evidence instead of rerunning')
    old = json.loads((PROJECT / settings['stage2b_config']).read_text())
    selection_path = PROJECT / settings['selection']
    selection = json.loads(selection_path.read_text())
    if digest(selection['checkpoint']) != selection['checkpoint_sha256']:
        raise ValueError('Frozen primary checkpoint changed')
    # Compare executable package bytes against the actual completed inventory.
    inventory_path = PROJECT / 'reports/stage2b/manifest.json'
    if digest(inventory_path) != '7743a8582b39343ceef16ac3b3ff3373bbf8d5e437b6e46718517307b11fc87b':
        raise ValueError('Historical inventory changed')
    entries = json.loads(inventory_path.read_text())['files']
    historical = {e['path']: e['sha256'] for e in entries if e['scope'] == 'project'}
    for path in (settings['stage2b_config'], settings['selection']):
        if digest(PROJECT / path) != historical[path]:
            raise ValueError('Historical configuration or selection changed')
    expected = {Path(e['path']).name: e['sha256'] for e in entries
                if e['scope'] == 'project' and e['path'].startswith('src/riichi_ai/')}
    sources = source_manifest()
    if sources['sources'] != expected:
        raise ValueError('Run A1 before adding/changing package modules')
    earlier = corpus_and_historical_wall_seeds(old)
    for path in (PROJECT / 'reports/stage2b').glob('development-*.json'):
        value = json.loads(path.read_text())
        earlier.update(m['seed'] for m in value.get('matches', []))
    for path in (PROJECT / 'reports/stage2b').glob('benchmark*.jsonl'):
        with path.open() as stream:
            earlier.update(json.loads(line)['seed'] for line in stream if line.strip())
    wall_seeds = [_seed(settings['root_seed'], 'block', i) for i in range(settings['blocks'])]
    if len(set(wall_seeds)) != 319 or set(wall_seeds) & earlier:
        raise ValueError('Confirmation seeds overlap or are duplicated')
    if (settings['blocks'], settings['margin'], settings['alpha'], settings['workers']) != (319, .1, .05, 4):
        raise ValueError('Unexpected confirmation protocol')
    # Creating the plan directory may change the dirty flag, never source bytes.
    reports.mkdir(parents=True, exist_ok=True)
    plan = {'schema': 'riichi-ai-package-a1-frozen-v1', 'settings': settings,
            'checkpoint': selection['checkpoint'], 'checkpoint_sha256': selection['checkpoint_sha256'],
            'selection_sha256': digest(selection_path), 'source_manifest': sources,
            'stage2b_plan': old, 'config_sha256': digest(config_path),
            'runner_sha256': digest(__file__), 'wall_seeds': wall_seeds,
            'earlier_seeds_checked': len(earlier), 'seed_overlap_count': 0,
            'frozen_before_outcomes': True, 'planned_executions': 2552,
            'hypotheses': 'H0: mean(student utility - teacher utility) <= -0.10; H1: mean > -0.10',
            'no_pilot_pooling': True, 'no_candidate_switching': True,
            'started_unix': time.time()}
    save(reports / 'plan.json', plan)
    # The persisted plan itself ensures a stable dirty flag through the campaign.
    plan['source_manifest'] = source_manifest()
    save(reports / 'plan.json', plan)
    return plan, reports


def child(config_path):
    from riichi_ai.evaluation import evaluate_paired, source_manifest
    from riichi_ai.stage2b_runtime import verify_paired_identity
    from riichi_ai.statistics import noninferiority_test
    config = json.loads(config_path.read_text())
    settings = config['confirmation']
    reports = PROJECT / settings['reports_root']
    plan = json.loads((reports / 'plan.json').read_text())
    if digest(config_path) != plan['config_sha256'] or digest(__file__) != plan['runner_sha256']:
        raise ValueError('Frozen orchestration changed')
    if source_manifest() != plan['source_manifest']:
        raise ValueError('Frozen package source identity changed')
    old = plan['stage2b_plan']
    result = evaluate_paired(
        candidate='checkpoint:' + plan['checkpoint'], baseline=old['evaluation']['baseline'],
        opponents=old['opponents'], blocks=settings['blocks'], seed=settings['root_seed'],
        workers=settings['workers'], max_steps=100000, failure_dir=reports / 'failures',
        memory_limit_bytes=settings['memory_bytes'], artifact_roots=[reports],
        disk_limit_bytes=settings['artifact_bytes'])
    save(reports / 'confirmation.json', result)
    if result['valid']:
        verify_paired_identity(result, checkpoint=plan['checkpoint'], expected_hash=plan['checkpoint_sha256'],
                               plan=old, root_seed=settings['root_seed'], blocks=settings['blocks'])
        test = noninferiority_test([b['mean_utility_difference'] for b in result['blocks']],
                                   planned_blocks=settings['blocks'], margin=settings['margin'],
                                   alpha=settings['alpha'], min_blocks=settings['minimum_blocks'],
                                   max_blocks=settings['maximum_blocks'])
    else:
        test = {'valid': False, 'passed': False, 'status': 'invalid_campaign'}
    save(reports / 'noninferiority-result.json', test)
    print(json.dumps({'completed_blocks': result['completed_blocks'], 'completed_games': result['completed_games'],
                      'noninferiority': test}), flush=True)


def stop_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=3)


def run(config_path):
    started = time.monotonic()
    import psutil
    config = json.loads(config_path.read_text())
    previous = PROJECT / config['confirmation']['reports_root'] / 'setup-attempts.json'
    previous_attempts = json.loads(previous.read_text()) if previous.exists() else []
    setup_charge = sum(item['charged_seconds'] for item in previous_attempts)
    started -= setup_charge
    plan, reports = prepare(config_path)
    settings = plan['settings']
    deadline = started + settings['limit_seconds'] - settings['finalization_reserve_seconds']
    peak, samples, reason = 0, 0, None
    with (reports / 'console.log').open('x') as stream:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--config', str(config_path), '--child'],
                                   cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        root = psutil.Process(process.pid)
        last_progress = 0
        try:
            while process.poll() is None:
                now = time.monotonic()
                try:
                    rss = psutil.Process().memory_info().rss + sum(
                        p.memory_info().rss for p in [root, *root.children(recursive=True)] if p.is_running())
                except psutil.NoSuchProcess:
                    rss = 0
                peak = max(peak, rss)
                size = sum(p.stat().st_size for p in reports.rglob('*') if p.is_file())
                samples += 1
                reason = ('wall_time_limit' if now >= deadline else 'memory_limit' if rss > settings['memory_bytes']
                          else 'artifact_limit' if size > settings['artifact_bytes'] else None)
                if reason:
                    stop_group(process)
                    break
                if now - last_progress >= 30:
                    progress = {'stage': 'A1 confirmation', 'elapsed_seconds': now - started,
                                'sampled_peak_rss_bytes': peak, 'status': 'running; outcomes not inspected'}
                    save(reports / 'progress.json', progress)
                    print(json.dumps(progress), flush=True)
                    last_progress = now
                time.sleep(.5)
        except BaseException:
            stop_group(process)
            raise
    from riichi_ai.evaluation import source_manifest
    test_path = reports / 'noninferiority-result.json'
    test = json.loads(test_path.read_text()) if test_path.exists() else None
    size = sum(p.stat().st_size for p in reports.rglob('*') if p.is_file())
    identity = {'checkpoint_unchanged': digest(plan['checkpoint']) == plan['checkpoint_sha256'],
                'source_unchanged': source_manifest() == plan['source_manifest'],
                'config_unchanged': digest(config_path) == plan['config_sha256'],
                'runner_unchanged': digest(__file__) == plan['runner_sha256']}
    elapsed = time.monotonic() - started
    within = elapsed < settings['limit_seconds'] and peak <= settings['memory_bytes'] and size < settings['artifact_bytes']
    complete = process.returncode == 0 and reason is None and within and test is not None and all(identity.values())
    summary = {'schema': 'riichi-ai-package-a1-completion-v1', 'completed': complete,
               'stop_reason': reason or ('completed' if complete else 'runner_failed'),
               'exit_code': process.returncode, 'elapsed_seconds': elapsed,
               'prior_failed_setup_seconds_charged': setup_charge,
               'sampled_peak_rss_bytes': peak, 'resource_samples': samples, 'artifact_bytes_before_summary': size,
               'within_limits': within, 'noninferiority': test if complete else {'valid': False, 'passed': False, 'status': 'incomplete'},
               **identity,
               'plan_sha256': digest(reports / 'plan.json'),
               'scope': 'One unchanged fixed candidate/panel, fresh fixed 319-block normal-approximation test; local heuristic retention only.'}
    save(reports / 'summary.json', summary)
    print(json.dumps(summary), flush=True)
    return 0 if complete else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT / 'configs/package_a.json')
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        child(args.config.resolve())
    else:
        raise SystemExit(run(args.config.resolve()))
