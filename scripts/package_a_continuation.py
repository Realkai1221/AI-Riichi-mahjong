"""Leased four-attempt learning-rate continuation; no test split or new matches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from package_a_stability import Ledger, atomic_json, digest, disk_guard, execute, require_worker_lease

ROOT = Path(__file__).resolve().parents[1]


def settings(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    if (config['limit_seconds'], config['maximum_attempts'], config['starting_step'],
            config['additional_steps'], config['terminal_step']) != (900, 4, 18500, 5000, 23500):
        raise ValueError('Authorized experiment limits changed')
    if (config['memory_bytes'], config['artifact_bytes'], config['epsilon'], config['batch_size'],
            config['device']) != (32 * 1024**3, 500000000, .01, 256, 'mps'):
        raise ValueError('Authorized resources or fixed recipe changed')
    if [(a['seed'], a['lr']) for a in config['attempts']] != [
            (202610171, .0003), (202610171, .0001), (202610172, .0001), (202610172, .0003)]:
        raise ValueError('Declared paired order changed')
    if config['constraints'] != dict(new_matches=0, action_test_predictions=0,
            test_split_loading=False, external_acquisition=False, reinforcement_learning=False):
        raise ValueError('Scope changed')
    for key in ('data_root', 'reports_root', 'stage2b_config', 'prior_config', 'diagnosis_plan'):
        config[key] = str((ROOT / config[key]).resolve())
    return path, config, digest(path)


def implementation(config):
    # Book intake is an independent stdlib-only workstream, not a model dependency.
    files = [p for p in sorted((ROOT / 'src/riichi_ai').glob('*.py')) if p.name != 'book_scenarios.py']
    files += [Path(__file__), ROOT / 'scripts/package_a_stability.py',
              ROOT / 'scripts/package_a_diagnosis.py', Path(config['stage2b_config']), Path(config['prior_config'])]
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def input_identity(config):
    if digest(config['diagnosis_plan']) != config['diagnosis_plan_sha256']:
        raise ValueError('Frozen diagnosis rows changed')
    inputs = {}
    for a in config['attempts']:
        if digest(a['checkpoint']) != a['checkpoint_sha256']:
            raise ValueError('Starting checkpoint changed')
        report_path = ROOT / a['source_report']
        report = json.loads(report_path.read_text())
        if not report['complete'] or report['completed_steps'] != config['starting_step']:
            raise ValueError('Starting source run not complete at the expected step')
        s = report['settings']
        if (s['seed'], s['lr'], s['epsilon'], s['batch_size'], s['rank_weight']) != (
                a['seed'], .0003, config['epsilon'], config['batch_size'], 0.):
            raise ValueError('Starting recipe mismatch')
        inputs[a['name']] = {'checkpoint_sha256': a['checkpoint_sha256'],
                             'source_report_sha256': digest(report_path)}
    return inputs


def worker(config, identity, allowance, index):
    started = time.perf_counter()
    source, inputs = implementation(config), input_identity(config)
    import numpy as np
    import torch
    from package_a_diagnosis import load_data
    from riichi_ai.training import _data_manifest, _rss
    torch.set_num_threads(1)
    if not torch.backends.mps.is_available():
        raise RuntimeError('Declared MPS device unavailable; no silent CPU fallback')
    training, validation = load_data(config)
    diagnosis = json.loads(Path(config['diagnosis_plan']).read_text())
    if (_data_manifest(training) != diagnosis['training_manifest']
            or _data_manifest(validation) != diagnosis['validation_manifest']):
        raise ValueError('Training or validation identity changed')
    reports = Path(config['reports_root'])
    plan_path = reports / 'plan.json'
    if index is None:
        if plan_path.exists():
            raise FileExistsError('Do not replace frozen plan')
        plan = {'schema': 'package-a-continuation-frozen-plan-v1', 'config_sha256': identity,
            'implementation': source, 'inputs': inputs, 'attempts': config['attempts'],
            'diagnosis_plan_sha256': digest(config['diagnosis_plan']),
            'train_indices': diagnosis['train_indices'], 'validation_indices': diagnosis['validation_indices'],
            'training_manifest': _data_manifest(training), 'validation_manifest': _data_manifest(validation),
            'frozen_before_formal_updates': True, 'constraints': config['constraints']}
        atomic_json(plan_path, plan)
        return
    plan = json.loads(plan_path.read_text())
    if (plan['config_sha256'] != identity or plan['implementation'] != source
            or plan['inputs'] != inputs or plan['attempts'] != config['attempts']):
        raise ValueError('Frozen protocol/source/input identity changed')
    from riichi_ai.package_a_continuation import continue_stability
    from riichi_ai.package_a_diagnosis_fit import evaluate_fit
    from riichi_ai.model import NeuralPolicy
    a = config['attempts'][index]
    output_path = reports / f"run-{a['name']}.json"
    output = {'schema': 'package-a-continuation-evidence-v1', 'attempt': a,
        'config_sha256': identity, 'plan_sha256': digest(plan_path), 'implementation': source,
        'constraints': config['constraints'], 'complete': False, 'fit': {}, 'arrays': {}}
    peak_memory = 0
    def check():
        nonlocal peak_memory
        if time.perf_counter() - started >= allowance - 12:
            raise TimeoutError('Endpoint finalization deadline reached')
        peak_memory = max(peak_memory, _rss() + torch.mps.driver_allocated_memory())
        if peak_memory >= config['memory_bytes']:
            raise MemoryError('Endpoint conservative memory ceiling')
        disk_guard(config)
    try:
        run = continue_stability(training, Path(config['data_root']) / 'models' / a['name'],
            validation=validation, checkpoint_path=a['checkpoint'], source_report=ROOT / a['source_report'],
            lr=a['lr'], additional_steps=config['additional_steps'], device=config['device'],
            expected_start_step=config['starting_step'], expected_checkpoint_sha256=a['checkpoint_sha256'],
            max_seconds=max(.1, allowance - (time.perf_counter() - started) - 30),
            memory_limit_bytes=config['memory_bytes'], protocol_manifest={'config_sha256': identity,
                'plan_sha256': digest(plan_path), 'attempt': a},
            progress_callback=lambda message: print(json.dumps({'attempt': a['name'], **message}), flush=True),
            artifact_guard=lambda: disk_guard(config))
        output['training'] = run
        check()
        model = NeuralPolicy.from_checkpoint(run['terminal_checkpoint'], device=config['device']).model.eval()
        arrays_root = Path(config['data_root']) / 'arrays'
        arrays_root.mkdir(parents=True, exist_ok=True)
        for split, dataset, indices in [('training', training, plan['train_indices']),
                                       ('validation', validation, plan['validation_indices'])]:
            summary, arrays = evaluate_fit(model, dataset, indices=indices, device=config['device'],
                batch_size=config['batch_size'], epsilon=config['epsilon'], check=check)
            output['fit'][split] = summary
            path = arrays_root / f"{a['name']}-{split}.npz"
            if path.exists():
                raise FileExistsError('Do not overwrite endpoint arrays')
            np.savez_compressed(path, **arrays)
            output['arrays'][split] = {'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size}
        output['checkpoints'] = {kind: {'path': run[f'{kind}_checkpoint'],
            'sha256': digest(run[f'{kind}_checkpoint'])} for kind in ('terminal', 'selected')
            if run.get(f'{kind}_checkpoint')}
        output['complete'] = bool(run['complete'])
        output['sources_unchanged'] = source == implementation(config)
        output['inputs_unchanged'] = inputs == input_identity(config)
        if not output['sources_unchanged'] or not output['inputs_unchanged']:
            raise RuntimeError('Source or input changed during attempt')
    except BaseException as error:
        output['complete'] = False
        output['failure'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        output['elapsed_seconds'] = time.perf_counter() - started
        output['sampled_peak_endpoint_conservative_memory_bytes'] = peak_memory
        atomic_json(output_path, output)
    print(json.dumps({'attempt': a['name'], 'complete': output['complete'],
                     'elapsed_seconds': output['elapsed_seconds']}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(ROOT / 'configs/package_a_continuation.json'))
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('budget'); sub.add_parser('prepare')
    test = sub.add_parser('budgeted')
    test.add_argument('--name', required=True)
    test.add_argument('--max-seconds', type=float, default=120)
    test.add_argument('arguments', nargs=argparse.REMAINDER)
    run = sub.add_parser('run'); run.add_argument('--index', type=int, required=True)
    hidden = sub.add_parser('_worker')
    hidden.add_argument('--index', type=int)
    hidden.add_argument('--allowance', type=float, required=True)
    hidden.add_argument('--lease-token', required=True)
    args = p.parse_args()
    config_path, config, identity = settings(args.config)
    ledger = Ledger(Path(config['data_root']) / 'budget-ledger.json', identity, config['limit_seconds'])
    if args.command == 'budget':
        print(json.dumps(ledger.snapshot()), flush=True); return
    if args.command == '_worker':
        name = 'prepare' if args.index is None else f"continuation:{config['attempts'][args.index]['name']}"
        require_worker_lease(ledger, args.lease_token, args.allowance, name)
        worker(config, identity, args.allowance, args.index); return
    reports = Path(config['reports_root']); reports.mkdir(parents=True, exist_ok=True)
    source = implementation(config)
    if args.command == 'budgeted':
        command = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
        if not command: raise ValueError('Need an explicit subprocess command')
        name, maximum = args.name, min(args.max_seconds, config['test_allowance_seconds'])
        marker = reports / f'checks-{name}.json'
    else:
        index = None if args.command == 'prepare' else args.index
        if index is not None:
            done = sorted(reports.glob('attempt-*.json'))
            if not 0 <= index < config['maximum_attempts'] or index != len(done):
                raise ValueError('Fixed four-attempt order; no replacements')
            name = f"continuation:{config['attempts'][index]['name']}"
            maximum = config['per_attempt_allowance_seconds']
            marker = reports / f'attempt-{index}.json'
        else:
            name, maximum = 'prepare', config['prepare_allowance_seconds']
            marker = reports / 'prepare-attempt.json'
        command = None
    if marker.exists(): raise FileExistsError('Operation already attempted; preserve it')
    record = {'name': name, 'config_sha256': identity, 'implementation': source,
              'status': 'reserved', 'started_unix': time.time()}
    atomic_json(marker, record)
    failure = None
    try:
        with ledger.operation(name, maximum, config['reserve_seconds']) as allowed:
            if command is None:
                active = [x for x in ledger.snapshot()['operations'] if x['status'] == 'active'][0]
                command = [sys.executable, str(Path(__file__).resolve()), '--config', str(config_path),
                    '_worker', '--allowance', str(allowed), '--lease-token', active['token']]
                if index is not None: command += ['--index', str(index)]
            record['watchdog'] = execute(command, allowed, config)
            if identity != digest(config_path) or source != implementation(config):
                raise RuntimeError('Source/config changed during operation')
            disk_guard(config)
    except BaseException as error:
        failure = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        record.update(status='failed' if failure else 'completed', failure=failure,
                      finished_unix=time.time(), budget=ledger.snapshot())
        atomic_json(marker, record)
    print(json.dumps({'phase': 'budget', **ledger.snapshot()}), flush=True)


if __name__ == '__main__':
    main()
