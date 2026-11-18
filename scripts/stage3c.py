"""Leased Stage 3C: four fixed full-match policy-gradient batches.

Only standard-library code runs before a durable numerical lease. Failed formal
batches are preserved and stop the chain; this launcher never substitutes games.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import statistics
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from package_a_stability import Ledger, atomic_json, digest, disk_guard as base_disk_guard, require_worker_lease

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SHA = '5130185ba8a397b84446ba8dc2d02f068a9448db5c0373f2aeb357e4af525a6f'


def seed_for(kind, index):
    return int.from_bytes(hashlib.sha256(f'riichi-ai-stage3c-20261021:{kind}:{index}'.encode()).digest()[:8], 'big')


def planned_batches():
    return [{'batch_index': batch, 'matches': [
        {'match_index': batch * 16 + offset, 'seed': seed_for('wall', batch * 16 + offset),
         'sampling_seed': seed_for('sampling', batch * 16 + offset), 'learner_seat': offset % 4}
        for offset in range(16)]} for batch in range(4)]


def planned_evaluation():
    return [{'block': block, 'wall_seed': seed_for('evaluation-wall', block), 'pairs': [
        {'learner_seat': seat, 'sampling_seed': seed_for('evaluation-sampling', block * 4 + seat),
         'candidate_match_index': 64 + block * 8 + seat * 2,
         'baseline_match_index': 65 + block * 8 + seat * 2}
        for seat in range(4)]} for block in range(32)]


def settings(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    exact = dict(schema="stage3c-config-v1", limit_seconds=600, memory_bytes=32 * 1024**3, artifact_bytes=1_000_000_000,
        reserve_seconds=45, per_batch_allowance_seconds=100, prepare_allowance_seconds=30,
        test_allowance_seconds=90, device='cpu', torch_threads=1, maximum_batches=4,
        matches_per_batch=16, maximum_matches=320, training_matches=64, evaluation_matches=256, evaluation_blocks=32,
        per_block_allowance_seconds=40, preparation_limit_seconds=90, initialization_seed=310211, maximum_updates=4, learning_rate=1e-4,
        weight_decay=0.0, gradient_clip=1.0, replay_atol=0., replay_rtol=0., replay_log_atol=0.,
        microbatch_size=1, max_steps=100000, reference_sha256=REFERENCE_SHA)
    if any(config.get(key) != value for key, value in exact.items()):
        raise ValueError('Authorized Stage 3C limits or fixed recipe changed')
    if config['batches'] != planned_batches() or config['evaluation'] != planned_evaluation():
        raise ValueError('Fixed seed, count, seat or batch order changed')
    if config['constraints'] != dict(book_training=False, test_split_loading=False,
            external_acquisition=False, paid_compute=False, fresh_strength_evaluation=True,
            opponent_actions_eligible=False, baseline='zero', entropy_bonus=0.0,
            reward_shaping=False, replay_reuse=False, rank_head_frozen=True, base_frozen=True):
        raise ValueError('Stage 3C scope changed')
    for key in ('data_root', 'reports_root', 'reference_checkpoint'):
        config[key] = str((ROOT / config[key]).resolve())
    return path, config, digest(path)


def implementation(scope='full'):
    paths = sorted(p for p in (ROOT / 'src/riichi_ai').glob('*.py') if not p.name.startswith('book_scenarios')) if scope == 'full' else []
    if scope == 'full':
        paths += sorted((ROOT / 'tests').glob('test_stage3c*.py'))
    paths += [Path(__file__), ROOT / 'scripts/package_a_stability.py', ROOT / 'pyproject.toml', ROOT / 'requirements-macos-arm64.lock']
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}



def disk_guard(config):
    total = base_disk_guard(config)
    total += sum(p.stat().st_size for p in (ROOT / 'Books/stage3c').rglob('*') if p.is_file())
    if total >= config['artifact_bytes']:
        raise RuntimeError('Stage 3C aggregate artifact ceiling reached')
    return total


def verify_preflight(reports):
    marker = json.loads((reports / 'checks-final-preflight.json').read_text())
    if marker['status'] != 'completed' or marker['implementation'] != implementation():
        raise ValueError('Successful current-implementation preflight required')
    document = ET.parse(reports / 'final-preflight.xml')
    if document.findall('.//failure') or document.findall('.//error') or document.findall('.//skipped'):
        raise ValueError('Preflight contains failed/error/skipped cases')
    cases = document.findall('.//testcase')
    postcall = [case for case in cases if case.get('classname', '').endswith('test_stage3c_postcall')]
    if len(postcall) < 8 or len(cases) < 43:
        raise ValueError('Missing required post-call or learner/controller coverage')
    return {'receipt_sha256': digest(reports / 'checks-final-preflight.json'),
            'junit_sha256': digest(reports / 'final-preflight.xml'),
            'passing_cases': len(cases), 'postcall_cases': len(postcall),
            'tests': {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / 'tests').glob('test_stage3c*.py'))}}

def verify_inputs(config):
    if digest(config['reference_checkpoint']) != config['reference_sha256']:
        raise ValueError('Frozen reference checkpoint changed')
    previous = json.loads((ROOT / 'reports/stage3c/previous-stage3b-snapshot.json').read_text())
    if previous.get('valid') is not True:
        raise ValueError('Historical Stage 3B must verify')


def check_chain(reports, index):
    if not 0 <= index < 4:
        raise ValueError('At most four fixed batches are authorized')
    preparation = reports / 'prepare-attempt.json'
    if (not preparation.exists() or json.loads(preparation.read_text())['status'] != 'completed'
            or not (reports / 'plan.json').is_file()):
        raise ValueError('Successful preparation and a frozen plan are required before collection')
    attempted = sorted(reports.glob('attempt-*.json'))
    if len(attempted) != index:
        raise ValueError('Fixed batch order; no retries or replacements')
    for prior in range(index):
        marker = json.loads((reports / f'attempt-{prior}.json').read_text())
        result = json.loads((reports / f'batch-{prior}.json').read_text())
        if marker['status'] != 'completed' or not result['complete']:
            raise ValueError('A failed or incomplete formal batch stops this experiment')


def execute(command, allowance, config, evidence, scope='full'):
    """Record watchdog evidence on success, worker error and resource timeout."""
    import psutil
    started = time.perf_counter()
    source = implementation(scope)
    process = None
    evidence.update(sampled_peak_tree_rss_bytes=0, elapsed_seconds=0.0,
                    artifact_bytes_peak=0, exit_code=None, sample_interval_seconds=0.2)
    try:
        process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
        while True:
            rss = 0
            try:
                parent = psutil.Process(os.getpid())
                processes = [parent, *parent.children(recursive=True)]
            except psutil.NoSuchProcess:
                processes = []
            for child in processes:
                try:
                    rss += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            evidence['sampled_peak_tree_rss_bytes'] = max(evidence['sampled_peak_tree_rss_bytes'], rss)
            if rss >= config['memory_bytes']:
                raise MemoryError('Stage 3C sampled process-tree memory ceiling reached')
            evidence['artifact_bytes_peak'] = max(evidence['artifact_bytes_peak'], disk_guard(config))
            if source != implementation(scope):
                raise RuntimeError('Stage 3C implementation changed during leased work')
            if process.poll() is not None:
                if process.returncode:
                    raise RuntimeError(f'Worker returned {process.returncode}')
                return
            if time.perf_counter() - started >= max(0.1, allowance - 10):
                raise TimeoutError('Stage 3C subprocess wall-time ceiling reached')
            time.sleep(0.2)
    finally:
        if process is not None:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            evidence['exit_code'] = process.returncode
        evidence['elapsed_seconds'] = time.perf_counter() - started


def worker(config, identity, allowance, index, block=None):
    started = time.perf_counter()
    deadline = time.monotonic() + allowance - 20
    source = implementation()
    verify_inputs(config)
    preflight_receipt = verify_preflight(Path(config['reports_root']))
    import torch
    from riichi_ai.model import NeuralPolicy
    from riichi_ai.stage3c import (initialize, load_boundary, save_boundary, state_manifest,
                                 collect_match, persist_match, update_batch)
    torch.set_num_threads(config['torch_threads'])
    reports, data = Path(config['reports_root']), Path(config['data_root'])
    plan_path = reports / 'plan.json'
    if block is not None:
        return evaluate_block(config, identity, deadline, source, block)
    if index is None:
        if plan_path.exists():
            raise FileExistsError('Do not replace frozen Stage 3C plan')
        state = initialize(config['reference_checkpoint'], config['reference_sha256'],
            protocol_sha256=identity, device=config['device'],
            learning_rate=config['learning_rate'], weight_decay=config['weight_decay'], initialization_seed=config['initialization_seed'])
        from riichi_ai.stage3c import prepare
        preflight = prepare(state)
        boundary = save_boundary(state, data / 'boundaries' / 'initial.pt')
        restored = load_boundary(data / 'boundaries' / 'initial.pt',
            expected_protocol_sha256=identity, device=config['device'])
        if state_manifest(state) != state_manifest(restored):
            raise RuntimeError('Initial boundary round trip differs')
        opponent = NeuralPolicy.from_checkpoint(config['reference_checkpoint'], device=config['device'])
        plan = dict(schema='stage3c-plan-v1', config_sha256=identity, implementation=source,
            frozen_before_formal_collection=True, batches=config['batches'], evaluation=config['evaluation'],
            preflight=preflight, preflight_receipt=preflight_receipt, constraints=config['constraints'],
            reference_checkpoint=config['reference_checkpoint'], reference_sha256=config['reference_sha256'],
            initial_boundary=boundary, initial_state=state_manifest(state), opponent=opponent.manifest(),
            versions={name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'riichienv', 'mahjong', 'psutil')},
            python=platform.python_version(), platform=platform.platform(),
            implementation_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                check=True, capture_output=True, text=True).stdout.strip())
        atomic_json(plan_path, plan)
        return
    plan = json.loads(plan_path.read_text())
    if (plan['config_sha256'] != identity or plan['implementation'] != source
            or plan['batches'] != config['batches']):
        raise ValueError('Frozen Stage 3C protocol or implementation changed')
    previous = plan['initial_boundary'] if index == 0 else json.loads(
        (reports / f'batch-{index - 1}.json').read_text())['output_boundary']
    boundary_path = data / 'boundaries' / ('initial.pt' if index == 0 else f'batch-{index - 1}.pt')
    if digest(boundary_path) != previous['sha256']:
        raise ValueError('Incoming boundary identity changed')
    state = load_boundary(boundary_path, expected_protocol_sha256=identity, device=config['device'])
    if state.updates != index:
        raise ValueError('Incoming update count differs from fixed chain')
    opponent = NeuralPolicy.from_checkpoint(config['reference_checkpoint'], device=config['device'])
    output = dict(schema='stage3c-batch-v1', batch_index=index, config_sha256=identity,
        plan_sha256=digest(plan_path), implementation=source, complete=False,
        input_boundary=previous, input_state=state_manifest(state), planned_matches=config['batches'][index]['matches'],
        opponent=opponent.manifest(), matches=[], update=None, output_boundary=None)
    trajectories = []
    try:
        for match in config['batches'][index]['matches']:
            trajectory = collect_match(state, opponent, batch_index=index, **match,
                max_steps=config['max_steps'], deadline=deadline)
            evidence = persist_match(trajectory, data / 'trajectories' / f'batch-{index}' / f"match-{match['match_index']}")
            output['matches'].append(evidence)
            atomic_json(reports / f'batch-{index}.json', output)
            trajectories.append(trajectory)
            if not trajectory.record['completed']:
                raise RuntimeError('Incomplete or failed planned match; batch preserved without an update')
            disk_guard(config)
            print(json.dumps({'phase': 'collection', 'batch_index': index,
                'persisted_matches': len(trajectories)}), flush=True)
        output['update'] = update_batch(state, trajectories, batch_index=index,
            planned_matches=config['batches'][index]['matches'], gradient_clip=config['gradient_clip'],
            replay_atol=config['replay_atol'], replay_rtol=config['replay_rtol'],
            replay_log_atol=config['replay_log_atol'],
            microbatch_size=config['microbatch_size'], deadline=deadline)
        output['output_boundary'] = save_boundary(state, data / 'boundaries' / f'batch-{index}.pt')
        restored = load_boundary(data / 'boundaries' / f'batch-{index}.pt',
            expected_protocol_sha256=identity, device=config['device'])
        output['output_state'] = state_manifest(state)
        output['boundary_roundtrip_verified'] = state_manifest(restored) == output['output_state']
        if not output['boundary_roundtrip_verified']:
            raise RuntimeError('Updated boundary round trip differs')
        verify_inputs(config)
        output['sources_unchanged'] = source == implementation()
        if not output['sources_unchanged']:
            raise RuntimeError('Implementation changed during formal batch')
        output['complete'] = True
    except BaseException as error:
        output['failure'] = dict(type=type(error).__name__, message=str(error))
        # The collector attaches its partial evidence on exceptions when available.
        partial = getattr(error, 'trajectory', None)
        if partial is not None:
            output['partial_match'] = persist_match(partial, data / 'trajectories' / f'batch-{index}' / 'partial')
        raise
    finally:
        output['final_in_memory_state'] = state_manifest(state)
        output['completed_update_count_delta'] = state.updates - index
        output['optimizer_steps_executed_delta'] = state.optimizer_steps_executed - index
        output['elapsed_seconds'] = time.perf_counter() - started
        atomic_json(reports / f'batch-{index}.json', output)
    print(json.dumps({'phase': 'update', 'batch_index': index, 'complete': output['complete'],
        'elapsed_seconds': output['elapsed_seconds'], 'update': output['update']}), flush=True)


def check_evaluation_chain(reports, block):
    if not 0 <= block < 32:
        raise ValueError('Exactly 32 fixed blocks; no extension')
    if len(list(reports.glob('attempt-*.json'))) != 4:
        raise ValueError('All four training batches required before evaluation')
    for index in range(4):
        if (json.loads((reports / f'attempt-{index}.json').read_text())['status'] != 'completed'
                or not json.loads((reports / f'batch-{index}.json').read_text())['complete']):
            raise ValueError('Training chain must finish without failure')
    if len(list(reports.glob('eval-attempt-*.json'))) != block:
        raise ValueError('Evaluation blocks are sequential without retries')
    for index in range(block):
        if (json.loads((reports / f'eval-attempt-{index}.json').read_text())['status'] != 'completed'
                or not json.loads((reports / f'eval-{index}.json').read_text())['complete']):
            raise ValueError('Failed evaluation block stops the experiment')


def evaluation_summary(reports):
    blocks = [json.loads((reports / f'eval-{i}.json').read_text()) for i in range(32)]
    if not all(b['complete'] for b in blocks):
        raise ValueError('Fixed endpoint requires all 32 valid blocks')
    differences = [b['mean_utility_difference'] for b in blocks]
    mean = statistics.mean(differences)
    halfwidth = 6 * math.sqrt(math.log(40) / (2 * 32))
    result = dict(schema='stage3c-evaluation-v1', blocks=32, matches=256,
        block_differences=differences, mean_utility_difference=mean,
        interval_method='fixed two-sided 95% Hoeffding; independent seeded blocks; D in [-3,3]',
        halfwidth=halfwidth, confidence_interval=[max(-3., mean-halfwidth), min(3., mean+halfwidth)],
        candidate_mean_rank=statistics.mean(x['rank'] for b in blocks for x in b['outcomes'] if x['arm']=='candidate'),
        baseline_mean_rank=statistics.mean(x['rank'] for b in blocks for x in b['outcomes'] if x['arm']=='baseline'),
        candidate_promoted=False, interpretation='Small engineering pilot; no strength or retention certification')
    atomic_json(reports / 'evaluation.json', result)
    return result


def evaluate_block(config, identity, deadline, source, block):
    from riichi_ai.model import NeuralPolicy
    from riichi_ai.stage3c import load_boundary, state_manifest, collect_match, persist_match
    reports, data = Path(config['reports_root']), Path(config['data_root'])
    plan_path = reports / 'plan.json'
    plan = json.loads(plan_path.read_text())
    if plan['config_sha256'] != identity or plan['implementation'] != source:
        raise ValueError('Frozen evaluation implementation/protocol changed')
    boundaries = {'candidate': json.loads((reports / 'batch-3.json').read_text())['output_boundary'],
                  'baseline': plan['initial_boundary']}
    states = {}
    for arm, boundary in boundaries.items():
        if digest(boundary['path']) != boundary['sha256']:
            raise ValueError('Fixed evaluation checkpoint changed')
        states[arm] = load_boundary(boundary['path'], expected_protocol_sha256=identity, device=config['device'])
    before = {arm: state_manifest(state) for arm, state in states.items()}
    opponent = NeuralPolicy.from_checkpoint(config['reference_checkpoint'], device=config['device'])
    planned = config['evaluation'][block]
    output = dict(schema='stage3c-evaluation-block-v1', block=block, config_sha256=identity,
        plan_sha256=digest(plan_path), implementation=source, complete=False, planned=planned,
        boundaries=boundaries, states_before=before, matches=[], outcomes=[], pairs=[], mean_utility_difference=None)
    try:
        for pair in planned['pairs']:
            records = {}
            for arm in ('candidate', 'baseline'):
                match_index = pair[f'{arm}_match_index']
                trajectory = collect_match(states[arm], opponent, batch_index=4+block, match_index=match_index, collection_role='evaluation',
                    seed=planned['wall_seed'], sampling_seed=pair['sampling_seed'], learner_seat=pair['learner_seat'],
                    max_steps=config['max_steps'], deadline=deadline)
                evidence = persist_match(trajectory, data / 'evaluation' / f'block-{block}' / f'match-{match_index}')
                output['matches'].append({'arm': arm, 'learner_seat': pair['learner_seat'], **evidence})
                records[arm] = trajectory.record
                atomic_json(reports / f'eval-{block}.json', output)
                if not trajectory.record['completed']:
                    raise RuntimeError('Incomplete fixed evaluation match; no replacement')
                output['outcomes'].append(dict(arm=arm, learner_seat=pair['learner_seat'],
                    utility=trajectory.record['learner_utility'], rank=trajectory.record['ranks'][pair['learner_seat']]))
                disk_guard(config)
            a, b = records['candidate'], records['baseline']
            common = min(len(a['ordinal_wall_fingerprints']), len(b['ordinal_wall_fingerprints']))
            if common == 0 or a['ordinal_wall_fingerprints'][:common] != b['ordinal_wall_fingerprints'][:common]:
                raise ValueError('Paired ordinal physical wall commitments differ')
            output['pairs'].append(dict(learner_seat=pair['learner_seat'], common_hands=common,
                common_walls_verified=True, utility_difference=a['learner_utility']-b['learner_utility']))
        output['states_after'] = {arm: state_manifest(state) for arm, state in states.items()}
        if output['states_after'] != before:
            raise ValueError('Evaluation altered parameters or optimizer')
        output['mean_utility_difference'] = statistics.mean(p['utility_difference'] for p in output['pairs'])
        output['complete'] = True
    except BaseException as error:
        output['failure'] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        atomic_json(reports / f'eval-{block}.json', output)
    print(json.dumps({'phase': 'evaluation', 'block': block, 'difference': output['mean_utility_difference']}), flush=True)
    if block == 31:
        print(json.dumps(evaluation_summary(reports)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/stage3c.json'))
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('budget')
    commands.add_parser('prepare')
    leased = commands.add_parser('budgeted')
    leased.add_argument('--name', required=True)
    leased.add_argument('--max-seconds', type=float, default=60)
    leased.add_argument('arguments', nargs=argparse.REMAINDER)
    run = commands.add_parser('run')
    run.add_argument('--index', type=int, required=True)
    evaluate = commands.add_parser('evaluate')
    evaluate.add_argument('--block', type=int, required=True)
    hidden = commands.add_parser('_worker')
    hidden.add_argument('--index', type=int)
    hidden.add_argument('--block', type=int)
    hidden.add_argument('--allowance', type=float, required=True)
    hidden.add_argument('--lease-token', required=True)
    args = parser.parse_args()
    path, config, identity = settings(args.config)
    ledger = Ledger(Path(config['data_root']) / 'budget-ledger.json', identity, config['limit_seconds'])
    if args.command == 'budget':
        print(json.dumps(ledger.snapshot(), indent=2))
        return
    if args.command == '_worker':
        name = f'eval:{args.block}' if args.block is not None else 'prepare' if args.index is None else f'batch:{args.index}'
        require_worker_lease(ledger, args.lease_token, args.allowance, name)
        worker(config, identity, args.allowance, args.index, args.block)
        return
    reports = Path(config['reports_root'])
    reports.mkdir(parents=True, exist_ok=True)
    index = block = None
    scope = 'full'
    if args.command == 'budgeted':
        if not args.name.replace('-', '').replace('_', '').isalnum():
            raise ValueError('Unsafe check name')
        command = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
        if not command:
            raise ValueError('Need an explicit check command')
        name, maximum = f'check:{args.name}', min(args.max_seconds, config['test_allowance_seconds'])
        marker = reports / f'checks-{args.name}.json'
        if args.name.startswith('book-'):
            scope = 'controller-only'  # Independent source-page rendering may run while learner is authored.
    elif args.command == 'prepare':
        name, maximum = 'prepare', config['prepare_allowance_seconds']
        marker, command = reports / 'prepare-attempt.json', None
    elif args.command == 'run':
        index = args.index
        verify_preflight(reports)
        check_chain(reports, index)
        name, maximum = f'batch:{index}', config['per_batch_allowance_seconds']
        marker, command = reports / f'attempt-{index}.json', None
    else:
        block = args.block
        check_evaluation_chain(reports, block)
        name, maximum = f'eval:{block}', config['per_block_allowance_seconds']
        marker, command = reports / f'eval-attempt-{block}.json', None
    if marker.exists():
        raise FileExistsError('Operation already attempted; preserve evidence')
    source = implementation(scope)
    snapshot = ledger.snapshot()
    if not list(reports.glob('attempt-*.json')):
        prep_remaining = config['preparation_limit_seconds'] - snapshot['charged_seconds']
        if name.startswith('batch:'):
            if prep_remaining < 0:
                raise RuntimeError('Preparation exceeded 90 seconds; no formal games')
        else:
            maximum = min(maximum, prep_remaining)
            if maximum <= 10:
                raise RuntimeError('Insufficient preparation budget')
    record = dict(name=name, config_sha256=identity, implementation=source, source_scope=scope,
        status='reserved', started_unix=time.time(), watchdog={})
    atomic_json(marker, record)
    failure = None
    try:
        with ledger.operation(name, maximum, config['reserve_seconds']) as allowance:
            if command is None:
                lease = next(x for x in ledger.snapshot()['operations'] if x['status'] == 'active')
                command = [sys.executable, str(Path(__file__).resolve()), '--config', str(path),
                    '_worker', '--allowance', str(allowance), '--lease-token', lease['token']]
                if index is not None:
                    command += ['--index', str(index)]
                if block is not None:
                    command += ['--block', str(block)]
            record['command'] = command
            execute(command, allowance, config, record['watchdog'], scope)
            if digest(path) != identity or source != implementation(scope):
                raise RuntimeError('Source/config changed during operation')
            disk_guard(config)
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        record.update(status='failed' if failure else 'completed', failure=failure,
            finished_unix=time.time(), budget=ledger.snapshot())
        atomic_json(marker, record)
        atomic_json(reports / 'budget.json', ledger.snapshot())
    print(json.dumps({'phase': name, 'charged_seconds': ledger.snapshot()['charged_seconds']}), flush=True)


if __name__ == '__main__':
    main()
