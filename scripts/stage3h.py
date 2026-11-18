"""Bounded Stage 3H saved-state diagnostic launcher; charge every numerical subprocess."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
from package_a_stability import Ledger, atomic_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/stage3h.json'
FORMAL_NAMES = {'inference', 'gradients'}
PHASE_LIMITS = {'preflight': 120, 'inference': 600, 'gradients': 300, 'audit': 120}


def phase_for(name):
    """Phase membership is fixed by name, so operations cannot evade subcaps."""
    for prefixes, phase in (
        (('preflight', 'prepare', 'controller', 'smoke', 'unit', 'tests'), 'preflight'),
        (('inference',), 'inference'),
        (('gradients',), 'gradients'),
        (('audit', 'verify'), 'audit'),
    ):
        if name.startswith(prefixes):
            return phase
    raise ValueError('Unrecognized numerical phase: '+name)


@contextmanager
def controller_lock(data):
    """Serialize admission and execution, including gaps between ledger locks."""
    with (Path(data) / 'controller.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another Stage 3H controller is active') from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)



def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_config():
    config = json.loads(CONFIG.read_text())
    if (config['schema'] != 'stage3h-diagnostic-config-v1' or config['limit_seconds'] != 1200
            or config['reserve_seconds'] != 60 or config['memory_bytes'] != 32 * 1024**3
            or config['artifact_bytes'] != 1_000_000_000
            or config['free_disk_reserve_bytes'] != 10_000_000_000
            or config['optimizer_steps'] != 0 or config['new_matches'] != 0
            or config['policy_promotion'] is not False
            or config['phase_limits_seconds'] != PHASE_LIMITS):
        raise ValueError('Stage 3H bounded diagnostic scope changed')
    config['reports_root'] = str(ROOT / config['reports_root'])
    data = Path(config['data_root'])
    if not data.is_absolute() or data.resolve() != data or data.is_relative_to(ROOT):
        raise ValueError('Bulk data must remain canonical and outside iCloud project')
    for stem in ('reference', 'call'):
        if digest(config[stem+'_checkpoint']) != config[stem+'_sha256']:
            raise ValueError('Historical checkpoint changed')
    return config, digest(CONFIG)


def admission(snapshot, name, allowance, config, future_reserve_seconds=0):
    if (not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in name)
            or not math.isfinite(allowance) or allowance < 15
            or not math.isfinite(future_reserve_seconds) or future_reserve_seconds < 0):
        raise ValueError('Invalid operation name, allowance, or future reserve')
    phase = phase_for(name)
    if any(op['name'] == name or op['status'] == 'active' for op in snapshot['operations']):
        raise ValueError('Repeated operation or unresolved lease')
    if name in FORMAL_NAMES and any(op['name'] in FORMAL_NAMES and op['status'] == 'failed'
                                    for op in snapshot['operations']):
        raise ValueError('A failed formal phase stops later formal work')
    if allowance + config['reserve_seconds'] + future_reserve_seconds > snapshot['remaining_seconds']:
        raise ValueError('Whole reservation does not fit beyond protected reserves')
    prior = sum(op['elapsed_seconds'] for op in snapshot['operations']
                if phase_for(op['name']) == phase)
    if prior + allowance > config['phase_limits_seconds'][phase]:
        raise ValueError('Reservation exceeds the cumulative '+phase+' cap')
    return phase


def artifacts_bytes(config):
    total = 0
    for root in (Path(config['data_root']), Path(config['reports_root'])):
        for path in root.rglob('*'):
            try:
                if path.is_file(): total += path.stat().st_size
            except FileNotFoundError:
                pass
    return total


def verify_sources(source):
    for name, expected in source.items():
        path = ROOT / name
        if not path.resolve().is_relative_to(ROOT.resolve()) or path.is_symlink():
            raise ValueError('Frozen source escapes project or is linked: '+name)
        if digest(path) != expected:
            raise RuntimeError('Frozen source changed: '+name)


def sources(manifest):
    fixed = ['scripts/stage3h.py', 'scripts/package_a_stability.py', 'configs/stage3h.json']
    result = {p: digest(ROOT / p) for p in fixed}
    if manifest:
        frozen = json.loads(Path(manifest).read_text())
        membership = frozen['sources']
        if not isinstance(membership, dict) or not membership:
            raise ValueError('Explicit nonempty source membership required')
        for key, value in membership.items():
            if (not isinstance(key, str) or not key or key.startswith('/')
                    or '\\' in key or '..' in Path(key).parts or Path(key).as_posix() != key
                    or not isinstance(value, str) or len(value) != 64
                    or any(c not in '0123456789abcdef' for c in value)):
                raise ValueError('Invalid source membership or digest')
            result[key] = value
    verify_sources(result)
    return result


def execute(command, allowance, config, source, receipt):
    import psutil
    started = time.perf_counter()
    process = None
    receipt.update(sampled_peak_tree_rss_bytes=0, artifact_bytes_peak=0,
                   sample_interval_seconds=.25, termination_requested=False)
    next_check = 0.
    try:
        active = [op for op in json.loads((Path(config['data_root'])/'budget-ledger.json').read_text())['operations'] if op['status']=='active']
        if len(active)!=1 or active[0]['name']!=receipt['name']:
            raise RuntimeError('Missing exclusive numerical lease')
        environment = dict(os.environ, RIICHI_STAGE3H_OPERATION=active[0]['name'], RIICHI_STAGE3H_TOKEN=active[0]['token'])
        process = subprocess.Popen(command, cwd=ROOT, start_new_session=True, env=environment)
        while True:
            rss = 0
            parent = psutil.Process(os.getpid())
            for item in (parent, *parent.children(recursive=True)):
                try: rss += item.memory_info().rss
                except psutil.NoSuchProcess: pass
            receipt['sampled_peak_tree_rss_bytes'] = max(receipt['sampled_peak_tree_rss_bytes'], rss)
            if rss >= config['memory_bytes']:
                raise MemoryError('Sampled process-tree RSS ceiling reached')
            now = time.perf_counter()
            if now >= next_check:
                size = artifacts_bytes(config)
                receipt['artifact_bytes_peak'] = max(receipt['artifact_bytes_peak'], size)
                if size >= config['artifact_bytes']:
                    raise RuntimeError('Artifact ceiling reached')
                if shutil.disk_usage(config['data_root']).free < config['free_disk_reserve_bytes']:
                    raise RuntimeError('Free disk reserve reached')
                verify_sources(source)
                next_check = now + 2
            if process.poll() is not None:
                if process.returncode:
                    raise RuntimeError(f'Numerical subprocess returned {process.returncode}')
                verify_sources(source)
                return
            if now - started >= allowance - 7:
                raise TimeoutError('Numerical subprocess watchdog deadline reached')
            time.sleep(.25)
    finally:
        if process is not None:
            # The main worker can exit while spawned descendants are still alive.
            # Terminate its entire private process group on every exit path.
            try:
                os.killpg(process.pid, 0)
                receipt['termination_requested'] = True
                os.killpg(process.pid, signal.SIGTERM)
                stop_at = time.monotonic()+3
                while time.monotonic()<stop_at:
                    process.poll()
                    try: os.killpg(process.pid, 0)
                    except ProcessLookupError: break
                    time.sleep(.02)
                else:
                    receipt['forced_kill'] = True
                    os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            receipt['exit_code'] = process.returncode
        receipt['monitored_elapsed_seconds'] = time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--seconds', required=True, type=float)
    parser.add_argument('--source-manifest')
    parser.add_argument('--future-reserve-seconds', type=float, default=0)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command: parser.error('A numerical command is required')
    config, identity = read_config()
    reports = Path(config['reports_root'])
    reports.mkdir(parents=True, exist_ok=True)
    data = Path(config['data_root'])
    data.mkdir(parents=True, exist_ok=True)
    with controller_lock(data):
        return run_leased(args, command, config, identity, reports, data)


def run_leased(args, command, config, identity, reports, data):
    ledger = Ledger(data / 'budget-ledger.json', identity, config['limit_seconds'])
    phase = admission(ledger.snapshot(), args.name, args.seconds, config, args.future_reserve_seconds)
    if args.name in FORMAL_NAMES and not args.source_manifest:
        raise ValueError('Formal work requires an explicit frozen source manifest')
    report = reports / ('operation-'+args.name+'.json')
    if report.exists(): raise FileExistsError('Preserve prior operation receipt')
    source = sources(args.source_manifest)
    receipt = dict(schema='stage3h-operation-v1', name=args.name, config_sha256=identity,
                   command=command, sources=source, reserved_seconds=args.seconds, phase=phase,
                   future_reserve_seconds=args.future_reserve_seconds,
                   status='prepared', versions={name: importlib.metadata.version(name)
                       for name in ('torch','numpy','riichienv','mahjong','pytest','psutil')},
                   python=platform.python_version(), platform=platform.platform())
    atomic_json(report, receipt)
    try:
        with ledger.operation(args.name, args.seconds,
                              reserve=config['reserve_seconds']+args.future_reserve_seconds) as allowed:
            if allowed != args.seconds:
                raise RuntimeError('Whole reservation changed after admission')
            receipt['status'] = 'running'
            atomic_json(report, receipt)
            execute(command, allowed, config, source, receipt)
        receipt['status'] = 'completed'
    except BaseException as error:
        receipt['status'] = 'failed'
        receipt['failure'] = {'type':type(error).__name__, 'message':str(error)}
        raise
    finally:
        snapshot = ledger.snapshot()
        receipt['ledger_operation'] = next((op for op in snapshot['operations']
                                          if op['name']==args.name), None)
        atomic_json(report, receipt)
        atomic_json(reports/'budget.json', snapshot)
        print(json.dumps({'name':args.name,'status':receipt['status'],
                          'charged_seconds':snapshot['charged_seconds'],
                          'remaining_seconds':snapshot['remaining_seconds']}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
