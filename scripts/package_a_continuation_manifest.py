"""Write-once inventory for the completed continuation and review artifacts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess

from package_a_manifest import digest, excluded_report, files_under, project_files, record, safe_recorded_path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/package_a_continuation'
MANIFEST = REPORTS / 'manifest.json'


def packaging_state(config_path, config, external):
    """Bind the final saved audit to this config, plan, roots and closed ledger.

    This is a state boundary check, not a rerun of the scientific audit. Run the
    final auditor immediately before packaging and stop all budgeted workers.
    """
    declared_external = Path(config['data_root'])
    declared_reports = ROOT / config['reports_root']
    if (not declared_external.is_absolute() or external != declared_external
            or declared_reports.resolve() != REPORTS.resolve()):
        raise ValueError('Configured artifact root identity mismatch')
    if not external.is_dir() or not REPORTS.is_dir():
        raise ValueError('Both configured report and external artifact roots must exist')
    identity = digest(config_path)
    audit_path, plan_path = REPORTS / 'audit.json', REPORTS / 'plan.json'
    audit = json.loads(audit_path.read_text())
    if (audit.get('schema') != 'package-a-continuation-audit-v1'
            or audit.get('valid') is not True or audit.get('config_sha256') != identity):
        raise ValueError('Successful audit must match the live config identity')
    plan = json.loads(plan_path.read_text())
    if audit.get('plan_sha256') != digest(plan_path) or plan.get('config_sha256') != identity:
        raise ValueError('Audit must match the live frozen plan identity')
    live_path, copy_path = external / 'budget-ledger.json', REPORTS / 'budget-ledger.json'
    live_bytes, copy_bytes = live_path.read_bytes(), copy_path.read_bytes()
    if audit.get('ledger_report_copy_verified') is not True or live_bytes != copy_bytes:
        raise ValueError('Verified report ledger copy must match the live ledger bytes')
    ledger = json.loads(live_bytes)
    limit = config['limit_seconds']
    if (ledger.get('schema') != 'package-a2-budget-v1' or ledger.get('identity') != identity
            or ledger.get('limit_seconds') != limit):
        raise ValueError('Live ledger config identity or limit mismatch')
    operations = ledger['operations']
    if ([op['id'] for op in operations] != list(range(len(operations)))
            or len({op['name'] for op in operations}) != len(operations)):
        raise ValueError('Duplicate or unordered ledger operations')
    for op in operations:
        elapsed, reserved = op.get('elapsed_seconds'), op.get('reserved_seconds')
        if (op.get('status') not in ('completed', 'failed')
                or type(elapsed) not in (int, float) or not math.isfinite(elapsed)
                or type(reserved) not in (int, float) or not math.isfinite(reserved)
                or not 0 <= elapsed <= reserved or reserved <= 0
                or op.get('exceeded_reservation') is not False):
            raise ValueError('Ledger has an active, invalid or excessive lease')
    formal_names = [op['name'] for op in operations if op['name'].startswith('continuation:')]
    maximum = config['maximum_attempts']
    if (type(maximum) is not int or not 0 < maximum <= 4 or len(formal_names) > maximum
            or formal_names != [f"continuation:{a['name']}" for a in config['attempts']]):
        raise ValueError('Formal attempts differ from the configured bounded sequence')
    charged = sum(op['elapsed_seconds'] for op in operations)
    expected_budget = {'charged_seconds': charged, 'limit_seconds': limit,
        'remaining_seconds': limit - charged, 'operation_count': len(operations),
        'active_leases': 0, 'formal_attempts': len(formal_names)}
    if (not 0 <= charged <= limit
            or any(audit['budget'].get(key) != value for key, value in expected_budget.items())):
        raise ValueError('Saved audit budget is stale or differs from the final closed ledger')
    return {'config_sha256': identity, 'audit_sha256': digest(audit_path),
            'plan_sha256': digest(plan_path), 'ledger_sha256': digest(live_path),
            'external_root': str(external), 'budget': expected_budget}


def create():
    if MANIFEST.exists():
        raise FileExistsError('Completed inventory already exists; verify instead')
    config_path = ROOT / 'configs/package_a_continuation.json'
    config = json.loads(config_path.read_text())
    external = Path(config['data_root'])
    gate = packaging_state(config_path, config, external)
    historical_path = REPORTS / 'prior-diagnosis-integrity.json'
    historical = json.loads(historical_path.read_text())
    if (not historical['valid'] or historical['snapshot_commit'] != config['diagnosis_snapshot_commit']
            or historical['manifest_sha256'] != config['diagnosis_manifest_sha256']):
        raise ValueError('Prior diagnosis snapshot identity mismatch')
    paths = project_files(ROOT, REPORTS)
    # Authored schema example is reproducible project data; supplied books stay local.
    paths.update(files_under(ROOT / 'examples', exclude_runtime_caches=True))
    paths.add(ROOT / '.gitignore')
    entries = [record('project', p, ROOT) for p in sorted(paths)]
    entries += [record('external', p, external) for p in sorted(files_under(external))]
    result = {'schema': 'package-a-continuation-inventory-v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'working_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'roots': {'project': str(ROOT), 'reports': str(REPORTS), 'external': str(external)},
        'config_sha256': digest(config_path), 'prior_snapshot_verification': historical,
        'prior_snapshot_verification_sha256': digest(historical_path),
        'packaging_state': gate,
        'file_count': len(entries), 'total_bytes': sum(e['bytes'] for e in entries), 'files': entries,
        'exclusions': ['This inventory and inventory/manifest-verification output files.',
            'Generated Python/runtime caches under source directories.',
            'Historical datasets/reports verified through their original snapshots.',
            'User-supplied Books directory: sources and quarantined transcriptions remain local, with identities recorded separately.'],
        'limitations': ['Integrity is not evidence of scientific acceptance or playing strength.',
            'Working HEAD precedes the results commit; explicit file hashes identify final bytes.']}
    with MANIFEST.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return {'created': True, 'file_count': len(entries), 'total_bytes': result['total_bytes'],
            'manifest_sha256': digest(MANIFEST)}


def verify():
    inventory = json.loads(MANIFEST.read_text())
    if inventory['schema'] != 'package-a-continuation-inventory-v1':
        raise ValueError('Unknown inventory schema')
    roots = {k: Path(v) for k,v in inventory['roots'].items()}
    known = {'project': set(), 'external': set()}
    errors = []
    for e in inventory['files']:
        if e['scope'] not in known:
            raise ValueError('Unknown inventory scope')
        p = safe_recorded_path(roots[e['scope']], e['path'])
        if p in known[e['scope']]: errors.append({'path': str(p), 'reason': 'duplicate'})
        known[e['scope']].add(p)
        if not p.is_file() or p.stat().st_size != e['bytes'] or digest(p) != e['sha256']:
            errors.append({'path': str(p), 'reason': 'missing or changed'})
    extra = {p for p in files_under(roots['reports']) if not excluded_report(p, roots['reports'])} - known['project']
    extra |= files_under(roots['external']) - known['external']
    counts_ok = (inventory['file_count'] == len(inventory['files'])
        and inventory['total_bytes'] == sum(e['bytes'] for e in inventory['files']))
    return {'valid': not errors and not extra and counts_ok, 'files_checked': len(inventory['files']),
        'errors': errors, 'extra_phase_files': sorted(map(str, extra)), 'counts_match': counts_ok,
        'manifest_sha256': digest(MANIFEST)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('create','verify'))
    args = p.parse_args()
    result = create() if args.command == 'create' else verify()
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result.get('created') or result.get('valid') else 1


if __name__ == '__main__':
    raise SystemExit(main())
