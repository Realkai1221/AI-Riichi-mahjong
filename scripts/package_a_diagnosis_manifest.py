"""Inventory the completed diagnosis without loading numerical/model libraries."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from package_a_manifest import digest, excluded_report, files_under, project_files, record, safe_recorded_path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/package_a_diagnosis'
MANIFEST = REPORTS / 'manifest.json'


def create():
    if MANIFEST.exists():
        raise FileExistsError('Completed inventory exists; do not rewrite it')
    config_path = ROOT / 'configs/package_a_diagnosis.json'
    config = json.loads(config_path.read_text())
    external = Path(config['data_root'])
    historical_path = REPORTS / 'package-a-integrity.json'
    historical = json.loads(historical_path.read_text())
    if (not historical['valid'] or historical['snapshot_commit'] != config['package_a_snapshot_commit']
            or historical['manifest_sha256'] != config['package_a_manifest_sha256']):
        raise ValueError('Historical snapshot evidence invalid')
    audit = json.loads((REPORTS / 'audit.json').read_text())
    if not audit['valid']:
        raise ValueError('Diagnosis audit must pass before inventory creation')
    entries = [record('project', p, ROOT) for p in sorted(project_files(ROOT, REPORTS))]
    entries += [record('external', p, external) for p in sorted(files_under(external))]
    inventory = {'schema': 'package-a-diagnosis-inventory-v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'working_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'roots': {'project': str(ROOT), 'reports': str(REPORTS), 'external': str(external)},
        'config_sha256': digest(config_path), 'prior_snapshot_verification': historical,
        'prior_snapshot_verification_sha256': digest(historical_path),
        'file_count': len(entries), 'total_bytes': sum(e['bytes'] for e in entries), 'files': entries,
        'exclusions': ['This inventory itself and inventory/manifest-verification outputs.',
            'Generated __pycache__, .pytest_cache, .pyc/.pyo and .DS_Store under project source directories.',
            'Historical bulk corpora and old report inventories are preserved, referenced through snapshots, not rewritten.'],
        'limitations': ['Recorded-byte integrity does not establish scientific success.',
            'Working HEAD precedes this report commit; explicit file hashes identify the final bytes.']}
    with MANIFEST.open('x') as stream:
        json.dump(inventory, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return {'created': True, 'file_count': len(entries), 'manifest_sha256': digest(MANIFEST),
            'total_bytes': inventory['total_bytes']}


def verify():
    inventory = json.loads(MANIFEST.read_text())
    if inventory['schema'] != 'package-a-diagnosis-inventory-v1':
        raise ValueError('Unrecognized inventory schema')
    roots = {k: Path(v) for k, v in inventory['roots'].items()}
    known = {'project': set(), 'external': set()}
    errors = []
    for e in inventory['files']:
        p = safe_recorded_path(roots[e['scope']], e['path'])
        if p in known[e['scope']]:
            errors.append({'path': str(p), 'reason': 'duplicate'})
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('create', 'verify'))
    args = parser.parse_args()
    result = create() if args.command == 'create' else verify()
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result.get('created') or result.get('valid') else 1


if __name__ == '__main__':
    raise SystemExit(main())
