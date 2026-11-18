#!/usr/bin/env python3
"""Write-once Stage 3H diagnostic inventory; standard-library accounting only.

This command hashes files and inspects saved JSON/JUnit evidence. It does not
load tensors, perform numerical inference, construct optimizers or run games.
The separately charged audit owns all numerical verification.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from stage3d_inventory import require, read, digest, safe_path, files_under, entries_for
from stage3h import PHASE_LIMITS, FORMAL_NAMES, phase_for

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/stage3h'
MANIFEST = REPORTS / 'manifest.json'
VERIFICATION = REPORTS / 'inventory-verification.json'
SCHEMA = 'stage3h-inventory-v1'
HISTORICAL_COMMIT = '420a4163f7b8ef210b800115133aea09ee1df29e'
HISTORICAL_MANIFEST = '519d5c0bbe3833e476d546cda049960af0c0df7e41752e49bd039d92920ece0a'
FORMAL_ORDER = ('inference', 'gradients')
AUXILIARY = ('scripts/stage3h_inventory.py', 'scripts/stage3d_inventory.py', 'scripts/package_a_stability.py',
    'scripts/verify_stage3g_snapshot.py', 'configs/stage3h.json', 'docs/STAGE3H_REVIEW.md',
    'reports/stage3h/REPORT.md', 'AGENTS.md', '.gitignore', 'README.md', 'docs/STATUS.md',
    'docs/DECISIONS.md', 'docs/IMPLEMENTATION_PLAN.md', 'pyproject.toml', 'requirements-macos-arm64.lock')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def selected_files(final=True):
    config, plan = read(ROOT / 'configs/stage3h.json'), read(REPORTS / 'plan.json')
    external = Path(config['data_root'])
    require(external.is_absolute() and external == external.resolve() and not external.is_relative_to(ROOT),
            'External root must remain canonical and outside project')
    require((ROOT / config['reports_root']).resolve() == REPORTS.resolve(), 'Report root differs')
    require(isinstance(plan.get('sources'), dict) and plan['sources'], 'Frozen runtime sources missing')
    roots = {'project': ROOT, 'external': external}
    auxiliary = AUXILIARY if final else tuple(name for name in AUXILIARY if (ROOT / name).is_file())
    project = {safe_path(ROOT, name) for name in (*plan['sources'], *auxiliary)}
    for directory, pattern in (('scripts', '*stage3h*.py'), ('src/riichi_ai', 'stage3h*.py'),
                               ('tests', '*stage3h*.py'), ('docs', 'STAGE3H*.md')):
        project.update((ROOT / directory).glob(pattern))
    project.update(files_under(REPORTS))
    project.difference_update((MANIFEST, VERIFICATION))
    selected = {'project': project, 'external': files_under(external, ignore_locks=True)}
    seen = set()
    for scope, paths in selected.items():
        require(paths, 'Empty inventory scope')
        for path in paths:
            safe_path(roots[scope], path.relative_to(roots[scope]).as_posix())
            require(path.is_file() and path.resolve() not in seen, 'Missing, linked or duplicated inventory member: ' + str(path))
            seen.add(path.resolve())
    return config, plan, roots, selected


def junit(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == 'testsuite' else list(root.findall('testsuite'))
    require(suites, 'JUnit output contains no test suite')
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path),
        **{key: sum(int(suite.get(key, '0')) for suite in suites) for key in ('tests', 'failures', 'errors', 'skipped')}}


def budget_state(config, plan, roots):
    identity = digest(ROOT / 'configs/stage3h.json')
    ledger, saved = read(roots['external'] / 'budget-ledger.json'), read(REPORTS / 'budget.json')
    require(ledger['identity'] == identity and ledger['limit_seconds'] == config['limit_seconds'] == 1200
            and all(saved.get(key) == value for key, value in ledger.items()), 'Live/saved budget identity differs')
    require(config['phase_limits_seconds'] == PHASE_LIMITS and config['reserve_seconds'] == 60, 'Phase caps or shutdown reserve changed')
    operations = ledger['operations']
    require([op['id'] for op in operations] == list(range(len(operations)))
            and len({op['name'] for op in operations}) == len(operations), 'Operation IDs/names are not exclusive')
    archives = {digest(path) for path in (REPORTS / 'source-archive').rglob('*') if path.is_file()}
    charges = dict.fromkeys(PHASE_LIMITS, 0.)
    total, previous_end, previous_formal = 0., None, -1
    failures, formal, receipts, tests = [], [], [], []
    failed_formal = False
    for op in operations:
        elapsed, reserved = op['elapsed_seconds'], op['reserved_seconds']
        require(op['status'] in ('completed', 'failed') and type(elapsed) in (int, float)
                and type(reserved) in (int, float) and math.isfinite(elapsed) and math.isfinite(reserved)
                and 0 <= elapsed <= reserved and reserved >= 15 and op['exceeded_reservation'] is False,
                'Unclosed or exceeded numerical lease')
        require(previous_end is None or op['started_unix'] + .05 >= previous_end, 'Numerical operations overlap')
        previous_end = op['started_unix'] + elapsed
        phase = phase_for(op['name'])
        path = REPORTS / f"operation-{op['name']}.json"
        receipt = read(path)
        future = receipt.get('future_reserve_seconds', 0)
        require(type(future) in (int, float) and math.isfinite(future) and future >= 0, 'Invalid future reservation')
        require(reserved + 60 + future <= 1200 - total + 1e-8, 'Protected numerical reserve consumed')
        require(charges[phase] + reserved <= PHASE_LIMITS[phase] + 1e-8, 'Phase reservation cap exceeded')
        require(receipt['schema'] == 'stage3h-operation-v1' and receipt['name'] == op['name']
                and receipt['status'] == op['status'] and receipt['ledger_operation'] == op
                and receipt['config_sha256'] == identity and receipt['reserved_seconds'] == reserved
                and receipt['phase'] == phase, 'Operation receipt differs from ledger')
        require(receipt.get('sampled_peak_tree_rss_bytes', 0) < config['memory_bytes']
                and receipt.get('artifact_bytes_peak', 0) < config['artifact_bytes'], 'Resource ceiling reached')
        for name, expected in receipt['sources'].items():
            require(digest(safe_path(ROOT, name)) == expected or expected in archives,
                    'Prior charged source is neither current nor archived: ' + name)
        command = receipt.get('command', [])
        for index, argument in enumerate(command):
            value = argument.split('=', 1)[1] if argument.startswith('--junitxml=') else command[index + 1] if argument == '--junitxml' and index + 1 < len(command) else None
            if value is not None:
                test_path = Path(value) if Path(value).is_absolute() else ROOT / value
                require(test_path.is_relative_to(ROOT), 'Test evidence outside project')
                test = junit(test_path)
                require(op['status'] != 'completed' or test['failures'] + test['errors'] == 0, 'Successful operation contains failed tests')
                tests.append({'operation': op['name'], **test})
        if op['status'] == 'failed':
            require(receipt.get('failure') and op.get('failure'), 'Failed operation lost its evidence')
            failures.append(op['name'])
        if op['name'] in FORMAL_NAMES:
            position = FORMAL_ORDER.index(op['name'])
            require(not failed_formal and position > previous_formal, 'Formal phases reordered or continued after failure')
            previous_formal = position
            failed_formal = op['status'] == 'failed'
            require(all(receipt['sources'].get(name) == value for name, value in plan['sources'].items()), 'Formal source binding differs')
            if not failed_formal:
                formal.append(op['name'])
        total += elapsed
        charges[phase] += elapsed
        receipts.append({'name': op['name'], 'sha256': digest(path)})
    require(total <= 1200 and saved['charged_seconds'] == total and saved['remaining_seconds'] == 1200 - total, 'Budget arithmetic differs')
    require(set(REPORTS.glob('operation-*.json')) == {REPORTS / f"operation-{op['name']}.json" for op in operations}, 'Receipt inventory differs from ledger')
    require(tests and any(item['tests'] > 0 and item['failures'] + item['errors'] == 0 for item in tests), 'Passing leased test evidence missing')
    return {'limit_seconds': 1200, 'charged_seconds': total, 'remaining_seconds': 1200 - total,
        'phase_charged_seconds': charges, 'phase_caps_seconds': PHASE_LIMITS, 'all_closed': True,
        'operations': len(operations), 'failed_operations': failures, 'completed_formal_phases': formal,
        'formal_phase_failed': failed_formal, 'receipts': receipts, 'test_reports': tests,
        'ledger_sha256': digest(roots['external'] / 'budget-ledger.json')}


def producing_operation(path, operations):
    aliases = {str(path), path.relative_to(ROOT).as_posix()} if path.is_relative_to(ROOT) else {str(path)}
    found = []
    for op in operations:
        receipt = read(REPORTS / f"operation-{op['name']}.json")
        if op['status'] == 'completed' and aliases.intersection(receipt.get('command', [])):
            found.append(op['name'])
    require(found, 'No completed charged producer for report: ' + str(path))
    return found


def historical_state(config):
    require(config['historical_stage3g_commit'] == HISTORICAL_COMMIT
            and config['historical_stage3g_manifest_sha256'] == HISTORICAL_MANIFEST, 'Historical Stage3G identity changed')
    previous = read(REPORTS / 'previous-stage3g-snapshot.json')
    require(previous['valid'] is True and previous['snapshot_commit'] == HISTORICAL_COMMIT
            and previous['manifest_sha256'] == HISTORICAL_MANIFEST and previous['verified_files'] == 37758
            and previous['verified_project_files'] == 138 and previous['verified_external_files'] == 37620
            and previous['verified_bytes_excluding_inventory'] == 1647215930, 'Historical full-verification receipt differs')
    committed = subprocess.check_output(['git', 'show', HISTORICAL_COMMIT + ':reports/stage3g/manifest.json'], cwd=ROOT)
    require(hashlib.sha256(committed).hexdigest() == HISTORICAL_MANIFEST
            and digest(ROOT / 'reports/stage3g/manifest.json') == HISTORICAL_MANIFEST, 'Historical committed or live Stage3G manifest changed')
    return {'commit': HISTORICAL_COMMIT, 'manifest_sha256': HISTORICAL_MANIFEST,
        'previous_verification_receipt_sha256': digest(REPORTS / 'previous-stage3g-snapshot.json'),
        'previously_verified_files': 37758,
        'scope': 'Pinned committed/live historical manifest and prior full37758-file verification receipt; currently selected immutable inputs rehashed below.'}


def checked_artifact(record, root):
    path = Path(record['path'])
    require(path.is_absolute() and path.resolve() == path and path.is_relative_to(root)
            and path.is_file() and not path.is_symlink(), 'Artifact path escapes declared output root')
    require(path.stat().st_size == record['bytes'] and digest(path) == record['sha256'], 'Artifact size/hash changed: ' + str(path))
    return path


def inference_state(plan, roots):
    root = roots['external']
    path = root / 'inference/result.json'
    result = read(path)
    require(result['schema'] == 'riichi-ai-stage3h-inference-v1' and result['phase'] == 'run'
            and result['complete'] is True and result['failure'] is None
            and result['plan_sha256'] == digest(REPORTS / 'plan.json') and result['models'] == plan['models']
            and result['optimizer_steps'] == result['new_matches'] == 0 and result['policy_promotion'] is False,
            'Inference did not complete the fixed read-only plan')
    require(digest(REPORTS / 'inference.json') == digest(path), 'Project/external inference reports differ')
    expected = {(split, name) for split, names in plan['selected_matches'].items() for name in names}
    records = result['matches']
    require(len(records) == len(expected) == result['planned_saved_matches'] == result['completed_saved_matches']
            and {(record['split'], record['match_id']) for record in records} == expected, 'Saved-state match coverage differs')
    rows = 0
    for record in records:
        require(record['complete'] is True and record['phase'] == 'run' and record['plan_sha256'] == result['plan_sha256']
                and record['optimizer_steps'] == record['new_matches'] == 0
                and record['model_ids'] == [item['id'] for item in plan['models']], 'Inference match provenance differs')
        expected_rows = plan['optional_rows_by_match'][record['split']][record['match_id']]
        require(record['rows'] == expected_rows and record['model_rows'] == expected_rows * 18, 'Optional row/model exposure differs')
        require(all(item['before'] == item['after'] for item in record['models_unchanged'])
                and len(record['models_unchanged']) == 18, 'Read-only inference weight checks incomplete')
        checked_artifact(record['numeric'], root)
        stored = root / 'inference/matches' / f"{record['split']}-{record['match_id']}.json"
        require(read(stored) == record, 'Per-match saved inference report differs')
        rows += record['rows']
    require(result['optional_rows'] == rows and result['model_rows'] == 18 * rows, 'Total inference exposure differs')
    preflight = read(REPORTS / 'inference-preflight.json')
    require(preflight['complete'] is True and preflight['failure'] is None and preflight['phase'] == 'preflight'
            and preflight['optimizer_steps'] == preflight['new_matches'] == 0
            and len(preflight['production_path_checks']) == 32
            and all(item['tensors_exact'] and item['scores_exact'] for item in preflight['production_path_checks']),
            'Exact deployed-path preflight incomplete')
    return {'complete': True, 'selected_matches': len(expected), 'optional_rows': rows, 'model_rows': 18 * rows,
        'models': 18, 'report_sha256': digest(path), 'project_report_sha256': digest(REPORTS / 'inference.json'),
        'preflight_report_sha256': digest(REPORTS / 'inference-preflight.json'), 'production_path_checks': 32}


def gradient_state(plan, roots):
    path = REPORTS / 'gradients.json'
    result, preflight = read(path), read(REPORTS / 'gradient-preflight.json')
    require(result['schema'] == 'riichi-ai-stage3h-gradient-diagnostic-v1' and result['phase'] == 'formal'
            and result['complete'] is True and result['passed'] is True
            and result['optimizer_steps'] == result['new_matches'] == result['checkpoint_writes'] == 0
            and result['policy_promotion'] is False and result['plan_sha256'] == digest(REPORTS / 'plan.json'),
            'Gradient diagnostic scope or completion differs')
    require(preflight['schema'] == 'riichi-ai-stage3h-gradient-preflight-v1'
            and preflight['complete'] is True and preflight['passed'] is True
            and preflight['backend_comparison']['passed'] is True and preflight['device'] == result['device']
            and result['input_identity'] == preflight['input_identity']
            and result['preflight_report_sha256'] == digest(REPORTS / 'gradient-preflight.json'), 'Gradient backend preflight changed')
    conditions = result['conditions']
    require(len(conditions) == 6 and [{key: item[key] for key in ('id', 'checkpoint_id', 'target', 'trained_alpha')} for item in conditions] == plan['gradient_conditions'], 'Gradient condition membership differs')
    layout = result['parameter_layout']
    cursor = 0
    for parameter in layout['ordered_parameters']:
        require(parameter['start'] == cursor and parameter['stop'] > cursor
                and math.prod(parameter['shape']) == parameter['stop'] - cursor, 'Gradient parameter offsets differ')
        cursor = parameter['stop']
    require(cursor == layout['trainable_parameters'], 'Gradient flattened size differs')
    groups = plan['gradient_groups']
    model_hashes = {model['id']: model['sha256'] for model in plan['models']}
    total_rows = 0
    for item in conditions:
        require(item['checkpoint_sha256'] == model_hashes[item['checkpoint_id']] and len(item['groups']) == 4, 'Gradient checkpoint/group coverage differs')
        for group, report in zip(groups, item['groups']):
            require(report['matches'] == group and report['weights_unchanged'] is True
                    and report['weights_sha256_before'] == report['weights_sha256_after'] and report['optimizer_steps'] == 0
                    and report['optional_rows'] == sum(plan['optional_rows_by_match']['training'][name] for name in group),
                    'Gradient sample exposure or immutability differs')
            total_rows += report['optional_rows']
        vectors = item['vectors']
        checked_artifact(vectors, roots['external'])
        require(vectors['dtype'] == 'float32' and vectors['shape_per_component'] == [4, cursor]
                and vectors['array_keys'] == ['mse', 'ce']
                and vectors['parameter_layout_sha256'] == canonical_hash(layout), 'Raw-gradient schema/layout commitment differs')
        require(read(Path(vectors['path']).with_suffix('.json')) == item, 'Completed gradient-condition receipt differs')
    return {'complete': True, 'conditions': 6, 'groups_per_condition': 4, 'distinct_training_matches': 64,
        'optional_row_presentations': total_rows, 'parameters_per_vector': cursor,
        'report_sha256': digest(path), 'preflight_report_sha256': digest(REPORTS / 'gradient-preflight.json'),
        'all_vectors_preserved': True, 'optimizer_steps': 0}


def audit_state(plan, roots, inference, gradients, operations):
    path = REPORTS / 'audit.json'
    audit = read(path)
    require(audit.get('schema') == 'riichi-ai-stage3h-evidence-audit-v1' and audit.get('valid') is True,
            'Independent numerical audit did not pass')
    require(audit['optimizer_steps'] == audit['new_matches'] == audit['checkpoint_writes'] == 0
            and audit['policy_promotion'] is False, 'Independent audit performed unplanned work')
    require(audit['plan_sha256'] == digest(REPORTS / 'plan.json')
            and audit['config_sha256'] == digest(ROOT / 'configs/stage3h.json')
            and audit['inference_report_sha256'] == inference['report_sha256']
            and audit['gradients_report_sha256'] == gradients['report_sha256']
            and audit['input_files_verified'] == len(plan['inputs']), 'Independent audit input/report identity differs')
    source_manifests = [candidate for candidate in REPORTS.glob('*.json')
        if candidate not in (MANIFEST, VERIFICATION) and digest(candidate) == audit['source_manifest_sha256']]
    require(len(source_manifests) == 1 and isinstance(read(source_manifests[0]).get('sources'), dict),
            'Independent audit source manifest missing')
    ia, ga = audit['inference'], audit['gradients']
    require(ia['valid'] is True and ia['saved_matches'] == inference['selected_matches']
            and ia['optional_rows'] == inference['optional_rows'] and ia['model_rows'] == inference['model_rows']
            and ia['cpu_reforward_rows'] == 32 and ia['cpu_reforward_model_rows'] == 576,
            'Independent inference coverage differs')
    require(ga['valid'] is True and ga['groups'] == 24 and ga['parameter_layout_verified'] is True
            and [item['id'] for item in ga['conditions']] == [item['id'] for item in plan['gradient_conditions']],
            'Independent raw-gradient audit coverage differs')
    result_conditions = {item['id']: item for item in read(REPORTS / 'gradients.json')['conditions']}
    for item in ga['conditions']:
        expected = result_conditions[item['id']]
        require(item['vectors_sha256'] == expected['vectors']['sha256'] and item['groups'] == 4
                and item['parameters'] == gradients['parameters_per_vector']
                and item['checkpoint_sha256'] == expected['checkpoint_sha256']
                and item['raw_vector_arithmetic_valid'] is True, 'Independent gradient arithmetic commitment differs')
    producing_operation(path, operations)
    return {'valid': True, 'schema': audit['schema'], 'report_sha256': digest(path),
        'plan_sha256': audit['plan_sha256'], 'source_manifest_sha256': audit['source_manifest_sha256'],
        'cpu_reforward_rows': 32, 'cpu_reforward_model_rows': 576, 'raw_gradient_groups_verified': 24,
        'optimizer_steps': 0, 'new_matches': 0}


def packaging_state(config, plan, roots, final=True):
    require(config['schema'] == 'stage3h-diagnostic-config-v1' and plan['schema'] == 'stage3h-diagnostic-plan-v1'
            and plan['execution_admitted'] is True and config['optimizer_steps'] == config['new_matches'] == 0
            and config['policy_promotion'] is plan['policy_promotion'] is False
            and plan['config_sha256'] == digest(ROOT / 'configs/stage3h.json'), 'Read-only fixed diagnostic scope changed')
    for name, expected in plan['sources'].items():
        require(digest(safe_path(ROOT, name)) == expected, 'Frozen runtime source changed: ' + name)
    require(isinstance(plan.get('inputs'), dict) and plan['inputs'], 'Immutable diagnostic inputs missing')
    for name, expected in plan['inputs'].items():
        path = Path(name)
        require(path.is_absolute() and path.resolve() == path and path.is_file() and not path.is_symlink()
                and digest(path) == expected, 'Historical diagnostic input changed: ' + name)
    require(not list(roots['external'].glob('**/lifecycle/*.started.json'))
            and not list(roots['external'].rglob('*.pt')), 'Stage3H unexpectedly contains game starts or model checkpoints')
    budget = budget_state(config, plan, roots)
    state = {'config_sha256': digest(ROOT / 'configs/stage3h.json'), 'plan_sha256': digest(REPORTS / 'plan.json'),
        'budget': budget, 'historical_stage3g': historical_state(config), 'immutable_inputs_verified': len(plan['inputs']),
        'optimizer_steps': 0, 'new_matches': 0, 'policy_promotion': False}
    if final:
        require(not budget['formal_phase_failed'] and budget['completed_formal_phases'] == list(FORMAL_ORDER), 'Ordered inference/gradient phases did not both finish')
        inference = inference_state(plan, roots)
        gradients = gradient_state(plan, roots)
        state.update(inference=inference, gradients=gradients,
            independent_audit=audit_state(plan, roots, inference, gradients, read(roots['external'] / 'budget-ledger.json')['operations']),
            completed_predeclared_protocol=True)
        require(any(Path(test['path']).name in ('release.xml', 'release-tests.xml')
                    and test['failures'] + test['errors'] == 0 for test in budget['test_reports']), 'Passed release tests missing')
        require(any(test['path'].endswith('/audit-tests.xml') and test['failures'] + test['errors'] == 0 for test in budget['test_reports']), 'Passed audit tests missing')
    size = sum(path.stat().st_size for base in (roots['external'], REPORTS) for path in base.rglob('*') if path.is_file() and path not in (MANIFEST, VERIFICATION))
    require(size < config['artifact_bytes'] == 1_000_000_000, 'Diagnostic artifact ceiling exceeded')
    state['artifact_bytes_excluding_inventory'] = size
    return state


def create():
    require(not MANIFEST.exists(), 'Inventory already exists; verify instead')
    config, plan, roots, selected = selected_files()
    state = packaging_state(config, plan, roots)
    entries = entries_for(roots, selected)
    result = {'schema': SCHEMA, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'working_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'roots': {key: str(value) for key, value in roots.items()}, 'packaging_state': state,
        'explicit_auxiliary_project_members': list(AUXILIARY), 'file_count': len(entries),
        'total_bytes': sum(entry['bytes'] for entry in entries), 'scope_counts': dict(Counter(entry['scope'] for entry in entries)),
        'files': entries, 'exclusions': ['This manifest and inventory-verification.json', 'External runtime lock files',
            'Historical model/data inputs remain separately pinned and verified'],
        'limitations': ['File integrity and bookkeeping do not independently recompute numerical results or certify Mahjong rules',
            'Stage3H changes no policy and estimates no playing strength', 'Current HEAD may precede the result commit; hashes bind final bytes']}
    raw = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    require(state['artifact_bytes_excluding_inventory'] + len(raw) + 1_000_000 < config['artifact_bytes'], 'Final inventory reserve exceeds cap')
    with MANIFEST.open('xb') as handle:
        handle.write(raw)
    return {'created': True, 'file_count': len(entries), 'total_bytes': result['total_bytes'],
        'scope_counts': result['scope_counts'], 'manifest_sha256': digest(MANIFEST), 'completed_predeclared_protocol': True}


def verify():
    inventory = read(MANIFEST)
    config, plan, roots, selected = selected_files()
    require(inventory['schema'] == SCHEMA and inventory['roots'] == {key: str(value) for key, value in roots.items()}
            and inventory['explicit_auxiliary_project_members'] == list(AUXILIARY), 'Inventory schema/roots differ')
    require(inventory['packaging_state'] == packaging_state(config, plan, roots), 'Closed protocol accounting changed')
    known, seen, errors = {scope: set() for scope in roots}, set(), []
    for entry in inventory['files']:
        scope = entry['scope']
        require(scope in roots, 'Unknown artifact scope')
        path = safe_path(roots[scope], entry['path'])
        require(path not in known[scope] and path.resolve() not in seen and type(entry['bytes']) is int and entry['bytes'] >= 0, 'Duplicate inventory artifact')
        known[scope].add(path)
        seen.add(path.resolve())
        if not path.is_file() or path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            errors.append({'scope': scope, 'path': entry['path'], 'reason': 'missing or changed'})
    extra = {scope: sorted(str(path) for path in selected[scope] - known[scope]) for scope in roots}
    missing = {scope: sorted(str(path) for path in known[scope] - selected[scope]) for scope in roots}
    counts = dict(Counter(entry['scope'] for entry in inventory['files']))
    totals = inventory['file_count'] == len(inventory['files']) and inventory['scope_counts'] == counts and inventory['total_bytes'] == sum(entry['bytes'] for entry in inventory['files'])
    actual_size = sum(path.stat().st_size for base in (roots['external'], REPORTS) for path in base.rglob('*') if path.is_file())
    require(actual_size < config['artifact_bytes'], 'Final bundle including inventory exceeds cap')
    return {'valid': not errors and not any(extra.values()) and not any(missing.values()) and totals,
        'files_checked': len(inventory['files']), 'total_bytes': inventory['total_bytes'], 'scope_counts': counts,
        'counts_match': totals, 'errors': errors, 'extra_scope_files': extra, 'missing_scope_files': missing,
        'manifest_sha256': digest(MANIFEST)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--write', '--create', dest='write', action='store_true')
    group.add_argument('--verify', action='store_true')
    group.add_argument('--precheck', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    require(not args.output or (args.verify and Path(args.output).resolve() == VERIFICATION), 'Only excluded verification output is permitted')
    if args.precheck:
        config, plan, roots, _ = selected_files(final=False)
        result = {'valid': True, 'packaging_state': packaging_state(config, plan, roots, final=False)}
    else:
        result = create() if args.write else verify()
    if args.output:
        with Path(args.output).open('x') as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.get('valid', True) else 1


if __name__ == '__main__':
    raise SystemExit(main())
