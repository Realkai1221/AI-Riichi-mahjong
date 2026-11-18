"""Inventory or verify the complete local Stage 2B evidence without rewriting Stage 2A."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

PROJECT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT / 'reports/stage2b'
MANIFEST = REPORTS / 'manifest.json'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def inventory():
    plan = json.loads((PROJECT / 'configs/stage2b.json').read_text())
    bulk = Path(plan['data_root'])
    project_files = set(PROJECT.glob('src/riichi_ai/*.py')) | set(PROJECT.glob('tests/test*.py'))
    project_files |= set(PROJECT.glob('scripts/stage2b*.py'))
    project_files |= {PROJECT / name for name in ('AGENTS.md', 'README.md', 'pyproject.toml', 'configs/stage2b.json',
        'docs/STATUS.md', 'docs/IMPLEMENTATION_PLAN.md', 'docs/DECISIONS.md', 'scripts/verify_stage2a_snapshot.py')}
    project_files |= {path for folder in (REPORTS, PROJECT / 'docs/stage2b') for path in folder.rglob('*') if path.is_file()}
    project_files.discard(MANIFEST)
    files = []
    for scope, paths in (('project', sorted(project_files)), ('bulk', sorted(path for path in bulk.rglob('*') if path.is_file()))):
        for path in paths:
            files.append({'scope': scope, 'path': str(path.relative_to(PROJECT)) if scope == 'project' else str(path),
                          'bytes': path.stat().st_size, 'sha256': digest(path)})
    artifact_bytes = sum(path.stat().st_size for root in (REPORTS, bulk) for path in root.rglob('*')
                         if path.is_file() and path != MANIFEST)
    report = {'schema': 'riichi-ai-stage2b-final-inventory-v1', 'files': files,
              'project_root': str(PROJECT), 'bulk_root': str(bulk), 'file_count': len(files),
              'git_commit_at_inventory': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip(),
              'artifact_bytes_excluding_inventory': artifact_bytes, 'artifact_limit_bytes': plan['limits']['new_artifact_bytes'],
              'stage2a_snapshot': plan['stage2a_snapshot_commit'],
              'scope': 'Actual current project sources/docs and all new Stage 2B artifacts; historical Stage 2A verified separately against its original Git snapshot.'}
    MANIFEST.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    total = artifact_bytes + MANIFEST.stat().st_size
    if total > plan['limits']['new_artifact_bytes']:
        raise RuntimeError('Actual final artifacts exceed approved limit')
    return {'file_count': len(files), 'new_artifact_bytes_including_inventory': total, 'within_budget': True,
            'manifest_sha256': digest(MANIFEST)}


def verify():
    record = json.loads(MANIFEST.read_text())
    for item in record['files']:
        path = PROJECT / item['path'] if item['scope'] == 'project' else Path(item['path'])
        if not path.is_file() or path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            raise ValueError(f"Evidence missing or changed: {path}")
    return {'valid': True, 'verified_files': len(record['files']), 'manifest_sha256': digest(MANIFEST)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['generate', 'verify'])
    args = parser.parse_args()
    print(json.dumps(inventory() if args.command == 'generate' else verify(), indent=2))
