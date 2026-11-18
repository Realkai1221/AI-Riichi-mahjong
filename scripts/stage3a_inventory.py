"""Write-once, stdlib-only inventory of the completed Stage 3A review bundle.

Create only after the final audit and all numerical leases are closed. Verify
checks both recorded bytes and exact current membership in each selected scope.
Original supplied PDFs remain separate pinned inputs; they are not duplicated.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/stage3a'
MANIFEST = REPORTS / 'manifest.json'
VERIFICATION = REPORTS / 'inventory-verification.json'
CONFIG_SHA = 'e18412e95adffbf1fb9aa7ceda6caac5f07867eefe04e9e136e772ec4710a52e'
PRIOR_COMMIT = 'd228df07128d50cfdccdceb29f81d37119ee843b'
PRIOR_MANIFEST_SHA = '63d1017d9b5bae2129b31ac80d66ab2c675a0161d2d62948aa8bce05f548e801'
SCHEMA = 'stage3a-inventory-v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def safe_path(root, relative):
    require(isinstance(relative, str) and relative != '' and '\\' not in relative,
            'Inventory paths must be nonempty portable relative paths')
    parts = PurePosixPath(relative)
    require(bool(parts.parts) and not parts.is_absolute() and all(p not in ('.', '..') for p in parts.parts)
            and parts.as_posix() == relative, 'Unsafe or noncanonical inventory path')
    path = root / relative
    require(path.resolve().is_relative_to(root.resolve()), 'Inventory path escapes its root')
    current = root
    for part in parts.parts:
        current /= part
        require(not current.is_symlink(), 'Symlinks are not accepted inventory artifacts')
    return path


def files_under(root, *, caches=False, locks=False):
    require(root.is_dir() and not root.is_symlink(), f'Missing or linked scope root: {root}')
    result = set()
    for path in root.rglob('*'):
        relative = path.relative_to(root)
        if caches and ('__pycache__' in relative.parts or '.pytest_cache' in relative.parts
                       or path.name == '.DS_Store'):
            continue
        if locks and path.name.endswith('.lock'):
            continue
        require(not path.is_symlink(), f'Unexpected linked file/directory: {path}')
        if path.is_file():
            result.add(path)
    return result


def roots_and_files():
    config = read(ROOT / 'configs/stage3a.json')
    external = Path(config['data_root'])
    books = ROOT / 'Books/stage3a'
    require(external.is_absolute() and external == external.resolve(), 'External root must be canonical and absolute')
    require((ROOT / config['reports_root']).resolve() == REPORTS.resolve(), 'Configured report root changed')
    require(not external.is_relative_to(ROOT), 'Training artifacts must remain outside project/iCloud')
    project = set((ROOT / 'src/riichi_ai').glob('*.py'))
    project.update(files_under(ROOT / 'tests', caches=True))
    project.update((ROOT / 'scripts').glob('stage3a*.py'))
    project.update(ROOT / p for p in (
        'scripts/verify_package_a_continuation_snapshot.py', 'scripts/package_a_stability.py',
        'scripts/book_count_conventions.py', 'scripts/book_efficiency_probe.py',
        'configs/stage3a.json', 'pyproject.toml', 'requirements-macos-arm64.lock',
        '.gitignore', 'AGENTS.md', 'README.md', 'docs/STATUS.md', 'docs/DECISIONS.md',
        'docs/IMPLEMENTATION_PLAN.md'))
    project.update((ROOT / 'docs').glob('STAGE3A*.md'))
    project.update(files_under(REPORTS))
    project.difference_update((MANIFEST, VERIFICATION))
    roots = {'project': ROOT, 'external': external, 'books': books}
    selected = {'project': project, 'external': files_under(external, locks=True),
                'books': files_under(books)}
    for scope, paths in selected.items():
        require(bool(paths), f'Empty inventory scope: {scope}')
        for path in paths:
            require(path.is_file(), f'Missing selected artifact: {path}')
            safe_path(roots[scope], path.relative_to(roots[scope]).as_posix())
    all_paths = [p.resolve() for paths in selected.values() for p in paths]
    require(len(set(all_paths)) == len(all_paths), 'One artifact appears in multiple scopes')
    return config, roots, selected


def packaging_state(config, roots):
    identity = digest(ROOT / 'configs/stage3a.json')
    require(identity == CONFIG_SHA, 'Authorized Stage 3A configuration changed')
    audit_path, plan_path = REPORTS / 'audit.json', REPORTS / 'plan.json'
    audit, plan = read(audit_path), read(plan_path)
    require(audit.get('schema') == 'stage3a-independent-audit-v1' and audit.get('valid') is True
            and audit.get('config_sha256') == identity and plan.get('config_sha256') == identity,
            'Final successful audit must match the live configuration')
    require(audit.get('plan_sha256') == digest(plan_path), 'Final audit does not bind the current frozen plan')
    require(audit.get('implementation') == plan['implementation'], 'Audit/plan implementation identity differs')
    for relative, expected in plan['implementation'].items():
        require(digest(safe_path(ROOT, relative)) == expected, 'Frozen numerical implementation changed')
    require(digest(config['reference_checkpoint']) == config['reference_sha256'], 'Original reference checkpoint changed')
    live_path = roots['external'] / 'budget-ledger.json'
    ledger, saved = read(live_path), read(REPORTS / 'budget.json')
    require(ledger.get('schema') == 'package-a2-budget-v1' and ledger.get('identity') == identity
            and ledger.get('limit_seconds') == config['limit_seconds'] == 600
            and all(saved.get(k) == value for k, value in ledger.items()),
            'Live ledger and saved budget identity/operations differ')
    operations = ledger['operations']
    require([op['id'] for op in operations] == list(range(len(operations)))
            and len({op['name'] for op in operations}) == len(operations), 'Duplicate/nonsequential operation')
    for operation in operations:
        elapsed, reserved = operation.get('elapsed_seconds'), operation.get('reserved_seconds')
        require(operation['status'] in ('completed', 'failed') and finite(elapsed) and finite(reserved)
                and 0 <= elapsed <= reserved and reserved > 0 and operation['exceeded_reservation'] is False,
                'Active, invalid or excessive numerical lease prevents final packaging')
    formal = [op['name'] for op in operations if op['name'].startswith('batch:')]
    require(formal == [f'batch:{i}' for i in range(len(formal))] and len(formal) <= config['maximum_batches'] == 4,
            'Formal operation order/count exceeds protocol')
    charged = sum(op['elapsed_seconds'] for op in operations)
    budget = dict(charged_seconds=charged, remaining_seconds=600 - charged, limit_seconds=600,
        operations=len(operations), completed_operations=sum(op['status'] == 'completed' for op in operations),
        failed_operations=sum(op['status'] == 'failed' for op in operations), all_closed=True)
    require(0 <= charged <= 600 and saved['charged_seconds'] == charged
            and saved['remaining_seconds'] == 600 - charged
            and all(audit['budget'].get(key) == value for key, value in budget.items())
            and audit['formal_attempts'] == len(formal), 'Final audit is stale relative to the closed ledger')
    prior_path = REPORTS / 'previous-continuation-snapshot.json'
    prior = read(prior_path)
    require(prior.get('valid') is True and prior.get('snapshot_commit') == PRIOR_COMMIT
            and prior.get('manifest_sha256') == PRIOR_MANIFEST_SHA and prior.get('verified_files') == 212,
            'Previous continuation snapshot verification differs')
    return dict(config_sha256=identity, plan_sha256=digest(plan_path), audit_sha256=digest(audit_path),
        ledger_sha256=digest(live_path), saved_budget_sha256=digest(REPORTS / 'budget.json'),
        prior_snapshot_verification_sha256=digest(prior_path), budget=budget)


def entries_for(roots, selected):
    return [dict(scope=scope, path=path.relative_to(roots[scope]).as_posix(),
                 bytes=path.stat().st_size, sha256=digest(path))
            for scope in sorted(selected) for path in sorted(selected[scope])]


def create():
    require(not MANIFEST.exists(), 'Final inventory already exists; verify it instead')
    config, roots, selected = roots_and_files()
    gate = packaging_state(config, roots)
    entries = entries_for(roots, selected)
    result = dict(schema=SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
        working_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        roots={scope: str(path) for scope, path in roots.items()}, packaging_state=gate,
        file_count=len(entries), total_bytes=sum(e['bytes'] for e in entries),
        scope_counts=dict(Counter(e['scope'] for e in entries)), files=entries,
        exclusions=['This manifest and reports/stage3a/inventory-verification.json.',
            'External runtime .lock files; source/test runtime caches.',
            'Original supplied book PDFs: existing input identities remain in the source catalog; no copies added.',
            'Previous phase artifacts verified through their pinned immutable snapshot.'],
        limitations=['Integrity does not establish playing strength or strategic label correctness.',
            'Books scope includes both original and corrected local packets and their visual evidence.',
            'Working HEAD may precede the result commit; explicit file hashes bind final bytes.'])
    with MANIFEST.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return dict(created=True, file_count=len(entries), total_bytes=result['total_bytes'],
                scope_counts=result['scope_counts'], manifest_sha256=digest(MANIFEST))


def verify():
    inventory = read(MANIFEST)
    require(inventory.get('schema') == SCHEMA, 'Unknown inventory schema')
    config, roots, selected = roots_and_files()
    require(inventory['roots'] == {scope: str(path) for scope, path in roots.items()}, 'Inventory root identity changed')
    require(inventory['packaging_state'] == packaging_state(config, roots), 'Final packaging state changed')
    known = {scope: set() for scope in roots}
    global_paths, errors = set(), []
    for entry in inventory['files']:
        scope = entry['scope']
        require(scope in roots, 'Unknown inventory scope')
        path = safe_path(roots[scope], entry['path'])
        require(path not in known[scope] and path.resolve() not in global_paths, 'Duplicate inventory artifact')
        known[scope].add(path)
        global_paths.add(path.resolve())
        require(type(entry['bytes']) is int and entry['bytes'] >= 0, 'Invalid recorded byte count')
        if not path.is_file() or path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            errors.append({'scope': scope, 'path': entry['path'], 'reason': 'missing or changed'})
    extras = {scope: sorted(str(p) for p in selected[scope] - known[scope]) for scope in roots}
    missing = {scope: sorted(str(p) for p in known[scope] - selected[scope]) for scope in roots}
    counts = dict(Counter(e['scope'] for e in inventory['files']))
    counts_match = (inventory['file_count'] == len(inventory['files'])
        and inventory['total_bytes'] == sum(e['bytes'] for e in inventory['files'])
        and inventory['scope_counts'] == counts)
    return dict(valid=not errors and not any(extras.values()) and not any(missing.values()) and counts_match,
        files_checked=len(inventory['files']), total_bytes=inventory['total_bytes'], scope_counts=counts,
        counts_match=counts_match, errors=errors, extra_scope_files=extras, missing_scope_files=missing,
        manifest_sha256=digest(MANIFEST), scope='Live final bundle; historical continuation verified separately.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_mutually_exclusive_group(required=True)
    command.add_argument('--create', action='store_true')
    command.add_argument('--verify', action='store_true')
    parser.add_argument('--output', help='Verification output must be the excluded standard report path.')
    args = parser.parse_args()
    require(not args.output or (args.verify and Path(args.output).resolve() == VERIFICATION),
            'Only the standard verification-report output is permitted')
    result = create() if args.create else verify()
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result.get('created') or result.get('valid') else 1


if __name__ == '__main__':
    raise SystemExit(main())
