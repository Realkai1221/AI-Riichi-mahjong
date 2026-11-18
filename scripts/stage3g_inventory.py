"""Write-once Stage 3G evidence inventory and protocol-accounting checks.

Standard-library file/hash/JSON accounting only: no engines, checkpoint loading,
array calculations, model inference or repeated statistical analysis. Completed
fallback and incomplete factorial remain distinct facts.
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
from stage3g import PHASE_LIMITS, FORMAL_NAMES, phase_for

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/stage3g'
MANIFEST = REPORTS / 'manifest.json'
VERIFICATION = REPORTS / 'inventory-verification.json'
SCHEMA = 'stage3g-inventory-v1'
ARM_ORDER = ('mc_a0', 'mc_a1', 'delta_a0', 'delta_a1')
HISTORICAL_COMMIT = '094f18c9f8561e0b1ae8954f2692396ca5528b59'
HISTORICAL_MANIFEST = '1615ee08038c386e5884a6d607bac6ff2c62529e49686ebeb89dc353ba6a1812'
FORMAL_ORDER = ('collect', 'grp-fit', *(f'train-{arm}' for arm in ARM_ORDER), 'development')
AUXILIARY_PROJECT_PATHS = (
    'scripts/stage3g_inventory.py', 'scripts/stage3d_inventory.py',
    'scripts/verify_stage3f_snapshot.py', 'configs/stage3g.json',
    'docs/STAGE3G_REVIEW.md', 'reports/stage3g/REPORT.md',
    'AGENTS.md', '.gitignore', 'README.md', 'docs/STATUS.md',
    'docs/IMPLEMENTATION_PLAN.md', 'docs/DECISIONS.md', 'pyproject.toml',
    'requirements-macos-arm64.lock', 'tests/fixtures/stage1/cases.json',
)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def selected_files(*, final=True):
    config, plan = read(ROOT / 'configs/stage3g.json'), read(REPORTS / 'plan.json')
    external = Path(config['data_root'])
    require(external.is_absolute() and external == external.resolve() and not external.is_relative_to(ROOT),
            'External output root must be canonical and outside project')
    require((ROOT / config['reports_root']).resolve() == REPORTS.resolve(), 'Report root differs')
    require(isinstance(plan.get('sources'), dict) and plan['sources'], 'Frozen plan sources missing')
    roots = {'project': ROOT, 'external': external}
    auxiliary = AUXILIARY_PROJECT_PATHS if final else tuple(p for p in AUXILIARY_PROJECT_PATHS if (ROOT / p).is_file())
    project = {safe_path(ROOT, name) for name in (*plan['sources'], *auxiliary)}
    for directory, pattern in (('scripts', '*stage3g*.py'), ('src/riichi_ai', 'stage3g*.py'),
                               ('tests', '*stage3g*.py'), ('docs', 'STAGE3G*.md')):
        project.update((ROOT / directory).glob(pattern))
    project.update(files_under(REPORTS))
    project.difference_update((MANIFEST, VERIFICATION))
    selected = {'project': project, 'external': files_under(external, ignore_locks=True)}
    unique = set()
    for scope, paths in selected.items():
        require(paths, 'Empty scope: '+scope)
        for path in paths:
            safe_path(roots[scope], path.relative_to(roots[scope]).as_posix())
            require(path.is_file() and path.resolve() not in unique, 'Missing or duplicate artifact: '+str(path))
            unique.add(path.resolve())
    return config, plan, roots, selected


def junit(path):
    document = ET.parse(path).getroot()
    suites = [document] if document.tag == 'testsuite' else list(document.findall('testsuite'))
    require(suites, 'JUnit report contains no suites')
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=digest(path),
                **{key: sum(int(s.get(key, '0')) for s in suites) for key in ('tests','failures','errors','skipped')})


def budget_state(config, roots, plan):
    config_sha = digest(ROOT / 'configs/stage3g.json')
    ledger, saved = read(roots['external'] / 'budget-ledger.json'), read(REPORTS / 'budget.json')
    require(ledger['identity'] == config_sha and ledger['limit_seconds'] == config['limit_seconds'] == 7200
            and all(saved.get(key) == value for key, value in ledger.items()), 'Live and saved ledger differ')
    require(config['phase_limits_seconds'] == PHASE_LIMITS and config['reserve_seconds'] == 120,
            'Fixed phase caps or shutdown reserve changed')
    operations = ledger['operations']
    require([op['id'] for op in operations] == list(range(len(operations)))
            and len({op['name'] for op in operations}) == len(operations), 'Duplicate or nonsequential operation')
    archives = {digest(p) for p in (REPORTS / 'source-archive').rglob('*') if p.is_file()}
    charged, phase_charges, failed, tests, receipts = 0., dict.fromkeys(PHASE_LIMITS, 0.), [], [], []
    formal_failed, previous_formal, completed_formal = False, -1, []
    for op in operations:
        elapsed, reserved = op.get('elapsed_seconds'), op.get('reserved_seconds')
        require(op['status'] in ('completed','failed') and type(elapsed) in (int,float)
                and type(reserved) in (int,float) and math.isfinite(elapsed) and math.isfinite(reserved)
                and 0 <= elapsed <= reserved and reserved >= 15 and op['exceeded_reservation'] is False,
                'Unclosed, invalid or exceeded numerical operation')
        phase = phase_for(op['name'])
        path = REPORTS / f"operation-{op['name']}.json"
        receipt = read(path)
        future = receipt.get('future_reserve_seconds', 0)
        require(type(future) in (int,float) and math.isfinite(future) and future >= 0,
                'Invalid protected future reserve')
        require(reserved + 120 + future <= 7200 - charged + 1e-8, 'Lease consumed protected reserves')
        require(phase_charges[phase] + reserved <= PHASE_LIMITS[phase] + 1e-8, 'Cumulative phase reservation exceeded')
        require(receipt['schema'] == 'stage3g-operation-v1' and receipt['name'] == op['name']
                and receipt['status'] == op['status'] and receipt['ledger_operation'] == op
                and receipt['config_sha256'] == config_sha and receipt['reserved_seconds'] == reserved
                and receipt['phase'] == phase, 'Operation receipt differs from ledger')
        require(receipt.get('sampled_peak_tree_rss_bytes',0) < config['memory_bytes']
                and receipt.get('artifact_bytes_peak',0) < config['artifact_bytes'], 'Recorded resource ceiling reached')
        for name, expected in receipt['sources'].items():
            current = safe_path(ROOT, name)
            require(digest(current) == expected or expected in archives,
                    'Old operation source is neither current nor archived: '+name)
        command = receipt.get('command', [])
        test_paths = []
        for i, argument in enumerate(command):
            if argument.startswith('--junitxml='):
                test_paths.append(argument.split('=',1)[1])
            elif argument == '--junitxml' and i+1 < len(command):
                test_paths.append(command[i+1])
        for argument in test_paths:
            path_arg = Path(argument)
            path_arg = path_arg if path_arg.is_absolute() else ROOT / path_arg
            require(path_arg.is_relative_to(ROOT), 'JUnit output is outside project evidence')
            result = junit(path_arg)
            require(op['status'] != 'completed' or result['failures']+result['errors'] == 0,
                    'Successful test operation contains failures')
            tests.append(dict(operation=op['name'], **result))
        if op['status'] == 'failed':
            require(receipt.get('failure') and op.get('failure'), 'Failed operation lost its evidence')
            failed.append(op['name'])
        if op['name'] in FORMAL_NAMES:
            position = FORMAL_ORDER.index(op['name'])
            require(not formal_failed and position > previous_formal, 'Formal work continued after failure or reordered')
            previous_formal = position
            formal_failed = op['status'] == 'failed'
            if not formal_failed:
                completed_formal.append(op['name'])
        charged += elapsed
        phase_charges[phase] += elapsed
        receipts.append(dict(name=op['name'], sha256=digest(path)))
    require(charged <= 7200 and saved['charged_seconds'] == charged and saved['remaining_seconds'] == 7200-charged,
            'Cumulative numerical totals differ')
    require(set(REPORTS.glob('operation-*.json')) == {REPORTS/f"operation-{op['name']}.json" for op in operations},
            'Receipt membership differs from ledger')
    release = read(REPORTS / 'operation-preflight-release.json')
    require(release['status'] == 'completed' and all(release['sources'].get(k) == v for k,v in plan['sources'].items()),
            'Successful source-bound release preflight missing')
    return dict(limit_seconds=7200, charged_seconds=charged, remaining_seconds=7200-charged,
                phase_charged_seconds=phase_charges, phase_caps_seconds=PHASE_LIMITS,
                all_closed=True, operations=len(operations), failed_operations=failed,
                completed_formal_phases=completed_formal, formal_phase_failed=formal_failed,
                receipts=receipts, test_reports=tests, ledger_sha256=digest(roots['external']/'budget-ledger.json'))


def fixture_counts(budget):
    known = {
        'test_truncated_attempt_is_reproducible_durable_and_never_eligible': (2,2),
        'test_all_views_built_before_any_policy_and_actor_input_is_immutable': (1,1),
        'test_expired_deadline_saves_failure_without_fake_terminal_return': (1,1),
        'test_resolve_failure_retains_attempted_choices_without_claiming_submission': (1,1),
        'test_checkpoint_mismatch_is_persisted_before_any_engine_creation': (1,0),
    }
    evidence = []
    scenarios = {
        'test_resolved_call_has_audited_post_call_and_next_draw_features',
        'test_requested_chi_preempted_by_pon_is_not_resolved_call_coverage',
        'test_two_concealed_kans_preserve_closed_status_and_eight_tile_efficiency',
        'test_robbed_added_kan_is_a_declaration_without_a_replacement_draw',
    }
    for report in budget['test_reports']:
        for case in ET.parse(ROOT / report['path']).getroot().iter('testcase'):
            name = case.get('name','').split('[',1)[0]
            if name in known and 'stage3f_collect' in case.get('classname',''):
                require(not any(case.find(key) is not None for key in ('failure','error','skipped')),
                        'Fixture accounting requires completed fixture evidence')
                starts, engines = known[name]
                evidence.append(dict(operation=report['operation'], test=name, starts=starts, engine_initializations=engines, admissions=starts))
            elif name in scenarios and 'stage3c_postcall' in case.get('classname',''):
                require(not any(case.find(key) is not None for key in ('failure','error','skipped')),
                        'Short explicit-wall scenario accounting requires passed fixture evidence')
                evidence.append(dict(operation=report['operation'], test=case.get('name'), starts=0, engine_initializations=1, admissions=1))
    return dict(known_partial_execution_admissions=sum(x['admissions'] for x in evidence),
                known_partial_lifecycle_starts=sum(x['starts'] for x in evidence),
                known_partial_engine_initializations=sum(x['engine_initializations'] for x in evidence),
                cases=evidence, limitation='Temporary pytest lifecycle artifacts are not inventoried; counts follow preserved successful fixture definitions. This does not exclude unregistered executions outside the campaign.')


def lifecycle_state(config, plan, roots, budget):
    external = roots['external']
    starts = sorted(external.glob('**/lifecycle/*.started.json'))
    finishes = set(external.glob('**/lifecycle/*.finished.json'))
    matched, complete, incomplete, unclosed, all_reports = set(), [], [], [], {}
    for path in starts:
        directory = path.parent.parent
        start = read(path)
        job = start['plan']
        name = job['match_id']
        require(path.name == name+'.started.json', 'Start identity differs')
        finish = path.with_name(name+'.finished.json')
        if not finish.exists():
            unclosed.append(path.relative_to(external).as_posix())
            continue
        matched.add(finish)
        report_path = directory / 'reports' / (name+'.json')
        report, marker = read(report_path), read(finish)
        require(marker['report_sha256'] == canonical_hash(report) and marker['payload_sha256'] == report['payload_sha256']
                and marker['completed'] == report['completed'] and all(report.get(k) == v for k,v in job.items()),
                'Durable report commitment differs')
        deterministic = {k:v for k,v in report.items() if k not in ('payload_sha256','elapsed_seconds','peak_worker_rss_bytes','timing_seconds')}
        require(canonical_hash(deterministic) == report['payload_sha256'] and report['ppo_eligible'] is False,
                'Match payload or PPO eligibility changed')
        if report['schema'] == 'riichi-ai-stage3g-collection-v1':
            require(start['reference_sha256'] == report['reference_sha256'] == config['reference_sha256']
                    and start['call_checkpoint_sha256'] == report['call_checkpoint_sha256'] == config['call_sha256'],
                    'Collector checkpoint identity changed')
            source = {'training':'offline-training','development':'development-only','preflight':'preflight-only'}[report['role']]
            require(report['source_role'] == source and report['training_eligible'] is (report['role']=='training' and report['completed']),
                    'Collector role or offline eligibility changed')
            artifacts = report['artifacts']
        elif report['schema'] == 'riichi-ai-stage3g-development-match-v1':
            require(report['source_role'] == 'development-only' and report['training_eligible'] is False,
                    'Development records became training data')
            item = report['artifact']
            artifacts = {item['path']: item}
        else:
            raise ValueError('Unknown lifecycle schema')
        for relative, artifact in artifacts.items():
            target = safe_path(directory, relative)
            require(target.stat().st_size == artifact['bytes'] and digest(target) == artifact['sha256'],
                    'Lifecycle artifact differs: '+str(target))
        key = report_path.relative_to(external).as_posix()
        all_reports[key] = report
        item = dict(directory=directory.relative_to(external).as_posix(), match_id=name, rows=report['rows'])
        if report['completed']:
            require(report['failure'] is None and report['terminal_result_verified'] is True, 'Completed match lacks terminal proof')
            complete.append(item)
        else:
            require(report['failure'] is not None, 'Incomplete match hides failure')
            incomplete.append(item)
    require(matched == finishes, 'Finish marker lacks a start')
    persisted_reports = {p.relative_to(external).as_posix() for p in external.glob('**/reports/*.json')}
    require(persisted_reports == set(all_reports), 'Persisted match report lacks matching lifecycle')
    fixtures = fixture_counts(budget)
    require(len(starts)+fixtures['known_partial_execution_admissions'] <= config['max_match_executions'] == 5500,
            'Recorded and known partial-fixture match attempts exceed ceiling')
    phases = {}
    for phase, assignments, commitment in (
        ('preflight',plan['preflight'],digest(REPORTS/'seed-plan.json')),
        ('corpus',plan['corpus'],digest(REPORTS/'plan.json')),
    ):
        directory = external / phase
        manifest = read(directory/'manifest.json')
        records = [all_reports[key] for key in sorted(all_reports) if key.startswith(phase+'/reports/')]
        require(manifest['schema'] == 'riichi-ai-stage3g-corpus-v1' and manifest['planned'] == assignments
                and manifest['matches'] == records and manifest['config_sha256'] == digest(ROOT/'configs/stage3g.json')
                and manifest['plan_sha256'] == commitment and manifest['ppo_eligible'] is False,
                'Corpus manifest identities or membership differ')
        expected = {job['match_id']:job for job in assignments}
        require(all(all(record.get(k)==v for k,v in expected[record['match_id']].items()) for record in records),
                'Collected match differs from fixed assignment')
        splits = {role:[r['match_id'] for r in records if r['role']==role] for role in sorted({p['role'] for p in assignments})}
        require(manifest['splits'] == splits, 'Corpus split membership differs')
        if manifest['complete']:
            require(manifest['failure'] is None and len(records)==len(assignments) and all(r['completed'] for r in records),
                    'Complete corpus lacks complete fixed games')
        phases[phase] = dict(complete=manifest['complete'], planned=len(assignments), completed=sum(r['completed'] for r in records),
                             rows=sum(r['rows'] for r in records), manifest_sha256=digest(directory/'manifest.json'))
    require(len(plan['preflight']) == 16 and len(plan['corpus']) == 4000
            and Counter(job['role'] for job in plan['corpus']) == {'training':3200,'development':800},
            'Fixed preflight/corpus/split counts changed')
    replay_ids = [plan['corpus'][i]['match_id'] for i in plan['corpus_replay_indices']]
    require(plan['corpus_replay_indices'] == list(range(0,4000,100)), 'Fixed one-percent replay selection changed')
    replay_reports = [record for key,record in all_reports.items() if not key.startswith(('preflight/','corpus/','development/'))]
    require(len(replay_reports) == len({r['match_id'] for r in replay_reports}), 'Duplicate extra replay game')
    require({r['match_id'] for r in replay_reports} <= set(replay_ids), 'Unplanned extra game outside corpus/development')
    for replay in replay_reports:
        original = all_reports['corpus/reports/'+replay['match_id']+'.json']
        require(replay['payload_sha256'] == original['payload_sha256'] and replay['artifacts'] == original['artifacts'],
                'Complete replay differs from original deterministic payload')
    return dict(phases=phases, durable_match_starts=len(starts), completed_match_executions=len(complete),
                incomplete_matches=incomplete, unclosed_starts=unclosed, known_partial_fixtures=fixtures,
                counted_starts_including_partial_fixtures=len(starts)+fixtures['known_partial_execution_admissions'],
                replay_count=len(replay_reports), planned_replays=40,
                all_replays_complete=len(replay_reports)==40 and all(r['completed'] for r in replay_reports))


def checkpoint_identity(path, expected, external):
    path = Path(path)
    require(path.is_absolute() and path.resolve()==path and path.is_relative_to(external)
            and path.is_file() and digest(path)==expected, 'Saved checkpoint identity differs')


def learning_state(config, plan, roots):
    external = roots['external']
    grp_path = external/'grp/report.json'
    grp = read(grp_path)
    corpus_sha = digest(external/'corpus/manifest.json')
    require(grp['schema']=='riichi-ai-stage3g-placement-report-v1' and grp['complete'] is True
            and grp['training_matches']==3200 and grp['development_matches']==800
            and grp['corpus_manifest_sha256']==corpus_sha and grp['fit_recipe']['fits']==6
            and len(grp['fits'])==6 and grp['calibration_gate_passed'] is grp['gate']['passed'],
            'Placement fitting is incomplete or provenance differs')
    expected_fits = [f'fold-{i}' for i in range(5)]+['final']
    require([item['name'] for item in grp['fits']]==expected_fits, 'Placement fit order differs')
    for item in grp['fits']:
        checkpoint_identity(external/'grp'/f"predictor-{item['name']}.pt", item['predictor_sha256'], external)
    targets = Path(grp['target_path'])
    checkpoint_identity(targets,grp['target_sha256'],external)
    mapping = read(targets)
    require(mapping['corpus_manifest_sha256']==corpus_sha and mapping['calibration_gate_passed'] is grp['calibration_gate_passed']
            and mapping['gate']==grp['gate'], 'Hand-target gate or provenance differs')
    expected_arms = list(ARM_ORDER if grp['calibration_gate_passed'] else ARM_ORDER[:2])
    results = {}
    for path in external.rglob('result.json'):
        result = read(path)
        if result.get('schema')!='riichi-ai-stage3g-learning-v1':
            continue
        arm = result['identity']['arm']
        require(arm in expected_arms and arm not in results, 'Unexpected or duplicate learning arm')
        require(result['complete'] is True and result['scheduled_updates']==400 and result['rank_head_unchanged'] is True
                and result['policy_promotion'] is False and result['identity']['manifest_sha256']==corpus_sha
                and result['identity']['reference_sha256']==config['reference_sha256']
                and result['identity']['split']=='training' and result['identity']['purpose']=='formal-training',
                'Learning arm is incomplete or has invalid provenance')
        recipe = result['identity']['recipe']
        require(recipe['epochs']==2 and recipe['matches_per_update']==16 and recipe['initialization_seed']==config['model_seed']
                and recipe['order_seed']==config['order_seed'], 'Learning arm exposure or seed recipe differs')
        require(result['identity']['hand_targets_sha256']==(grp['target_sha256'] if arm.startswith('delta_') else None),
                'Learning target identity differs')
        updates = [json.loads(line) for line in (path.parent/'updates.jsonl').read_text().splitlines()]
        require(len(updates)==400 and [u['scheduled_update'] for u in updates]==list(range(1,401)), 'Update sequence differs')
        trained = [job['match_id'] for job in plan['corpus'] if job['role']=='training']
        for epoch in range(2):
            epoch_rows = [u for u in updates if u['epoch']==epoch]
            require(len(epoch_rows)==200 and all(u['matches']==16 and len(u['match_ids'])==16 and len(set(u['match_ids']))==16 for u in epoch_rows)
                    and Counter(name for u in epoch_rows for name in u['match_ids'])==Counter(trained),
                    'Each full training match must appear exactly once per epoch')
        steps = 0
        for i, update in enumerate(updates):
            require(update['optimizer_step'] is (update['optional_rows']>0), 'Empty group caused optimizer movement')
            steps += int(update['optimizer_step'])
            require(update['optimizer_updates']==steps, 'Optimizer counter differs from executed updates')
            expected_cursor = {'epoch':(i+1)//200,'next_match_position':((i+1)%200)*16}
            require(update['cursor']==expected_cursor, 'Update cursor differs from full-match group boundary')
        require(result['optimizer_updates']==steps, 'Final optimizer count differs')
        counters = {'optional_row_presentations':'optional_rows','all_seat_rows_loaded':'all_seat_rows',
                    'explorer_rows_loaded':'explorer_rows','empty_match_presentations':'empty_matches',
                    'clipped_updates':'clipped','match_presentations':'matches'}
        require(all(result['statistics'][key]==sum(u[field] for u in updates) for key,field in counters.items()),
                'Aggregate training counters differ from update records')
        checkpoints = result['checkpoints']
        require([Path(c['path']).name for c in checkpoints]==['initial.pt','update-0100.pt','update-0200.pt','update-0300.pt','final.pt'],
                'Expected fixed checkpoint boundaries missing')
        for checkpoint in checkpoints:
            checkpoint_identity(checkpoint['path'],checkpoint['sha256'],external)
            require(Path(checkpoint['path']).parent==path.parent and Path(checkpoint['path']).stat().st_size==checkpoint['bytes'],
                    'Checkpoint path or byte count differs')
        require(checkpoints[0]['weights_sha256']==result['initial_weights_sha256']
                and checkpoints[-1]['weights_sha256']==result['final_weights_sha256'], 'Initial/final weight commitments differ')
        results[arm] = dict(path=path.relative_to(external).as_posix(), sha256=digest(path),
                           initial_weights_sha256=result['initial_weights_sha256'], final_checkpoint=checkpoints[-1],
                           scheduled_updates=400, optimizer_updates=steps, statistics=result['statistics'],
                           update_order=[dict(epoch=u['epoch'],match_ids=u['match_ids'],row_order_sha256=u['row_order_sha256']) for u in updates])
    require(set(results)==set(expected_arms), 'Required learning arms did not all finish')
    first = results[expected_arms[0]]
    require(all(item['initial_weights_sha256']==first['initial_weights_sha256'] and item['update_order']==first['update_order'] for item in results.values()),
            'Learning arms differ in initialization or complete-match/row order')
    for item in results.values():
        item.pop('update_order')
    arms = read(REPORTS/'arms.json')
    require(arms['schema']=='stage3g-arms-v1' and [a['name'] for a in arms['arms']]==expected_arms
            and arms['skipped_arms']==[arm for arm in ARM_ORDER if arm not in expected_arms]
            and arms['delta_gate_passed'] is grp['calibration_gate_passed'], 'Evaluation arm eligibility differs from calibration fallback')
    for arm in arms['arms']:
        require(arm['checkpoint']==results[arm['name']]['final_checkpoint']['path']
                and arm['sha256']==results[arm['name']]['final_checkpoint']['sha256'], 'Evaluation did not use final completed arm')
    return dict(grp_complete=True, grp_report_sha256=digest(grp_path), calibration_gate_passed=grp['calibration_gate_passed'],
                factorial_complete=grp['calibration_gate_passed'], required_arms=expected_arms,
                skipped_arms=arms['skipped_arms'], arms=results, all_required_arms_complete=True,
                total_formal_scheduled_updates=sum(item['scheduled_updates'] for item in results.values()),
                total_formal_optimizer_updates=sum(item['optimizer_updates'] for item in results.values()),
                common_initial_weights_sha256=first['initial_weights_sha256'])


def development_state(config, roots, learning):
    external = roots['external']
    manifest_path = external/'development/manifest.json'
    manifest = read(manifest_path)
    execution = read(external/'development/execution-plan.json')
    expected_games = 64*4*(1+len(learning['required_arms']))
    require(manifest['schema']=='riichi-ai-stage3g-development-v1' and manifest['complete'] is True
            and manifest['failure'] is None and manifest['execution_plan']==execution
            and manifest['execution_plan_sha256']==digest(external/'development/execution-plan.json')
            and manifest['training_eligible'] is False and manifest['ppo_eligible'] is False
            and manifest['policy_promotion'] is False, 'Development manifest incomplete or identity differs')
    require(execution['blocks']==64 and execution['rotations']==4 and execution['planned_matches']==expected_games
            and execution['tie_order']==learning['required_arms'] and execution['skipped_arms']==learning['skipped_arms']
            and execution['arms_manifest_sha256']==digest(REPORTS/'arms.json')
            and execution['plan_sha256']==digest(REPORTS/'plan.json'), 'Fixed development panel differs')
    records = [read(p) for p in sorted((external/'development/reports').glob('*.json'))]
    require(manifest['matches']==records and len(records)==expected_games and all(r['completed'] for r in records),
            'Development manifest game membership differs')
    expected = {j['match_id']:j for j in execution['matches']}
    require(len(expected)==expected_games and {r['match_id'] for r in records}==set(expected)
            and all(all(r.get(k)==v for k,v in expected[r['match_id']].items()) for r in records), 'Development game assignments differ')
    statistics = manifest['statistics']
    require(statistics['complete'] is True and statistics['games']==expected_games and statistics['blocks']==64
            and statistics['confirmation_started'] is False and statistics['policy_promotion'] is False
            and statistics['factorial_complete'] is learning['factorial_complete'], 'Development summary contradicts scope')
    require(len(statistics['pairing'])==64*4*len(learning['required_arms'])
            and all(p['paired_common_walls_match'] is True and p['common_hands']>0 for p in statistics['pairing']),
            'Missing same-engine physical-wall pairing evidence')
    by_id = {r['match_id']:r for r in records}
    for entry in statistics['pairing']:
        prefix = f"block-{entry['block']:03d}-seat-{entry['focal_seat']}-"
        baseline = by_id[prefix+'baseline']['ordinal_wall_fingerprints']
        candidate = by_id[prefix+entry['arm']]['ordinal_wall_fingerprints']
        common = min(len(baseline),len(candidate))
        require(entry['common_hands']==common and baseline[:common]==candidate[:common], 'Physical-wall pairing report differs')
    require(statistics['selected_candidate'] is None or statistics['selected_candidate'] in learning['required_arms'], 'Invalid selected candidate')
    summary = read(REPORTS/'development.json')
    require(summary['complete'] is True and summary['completed_games']==expected_games and summary['statistics']==statistics
            and summary['manifest_sha256']==digest(manifest_path), 'Project development report differs')
    return dict(complete=True, games=expected_games, blocks=64,
                common_ordinal_wall_pairs=sum(item['common_hands'] for item in statistics['pairing']),
                selected_candidate=statistics['selected_candidate'],
                best_observed_arm=statistics['best_observed_arm'], manifest_sha256=digest(manifest_path),
                policy_promotion=False, confirmation_started=False)



def numerical_audit_state(config, plan, roots, lifecycle, learning, development, budget):
    corpus_path, replay_path, learning_path = [REPORTS / name for name in
        ('corpus-audit.json','replay-audit.json','learning-audit.json')]
    corpus, replay, audit = read(corpus_path), read(replay_path), read(learning_path)
    require(corpus['schema']=='riichi-ai-stage3g-corpus-audit-v1' and corpus['valid'] is True
            and corpus['completed_matches']==4000 and len(corpus['matches'])==4000
            and corpus['manifest_sha256']==lifecycle['phases']['corpus']['manifest_sha256']
            and corpus['audited_rows']==lifecycle['phases']['corpus']['rows']
            and all(item['valid'] is True for item in corpus['matches']), 'Full-corpus numerical audit is incomplete or unbound')
    require({item['match_id'] for item in corpus['matches']}=={item['match_id'] for item in plan['corpus']},
            'Corpus audit match coverage differs')
    selected = [plan['corpus'][index]['match_id'] for index in plan['corpus_replay_indices']]
    require(replay['schema']=='riichi-ai-stage3g-replay-audit-v1' and replay['valid'] is True
            and [item['match_id'] for item in replay['matches']]==selected
            and all(item['matched'] is True for item in replay['matches'])
            and replay['total_corpus_rows']==corpus['audited_rows']
            and replay['same_engine_replay_performed'] is True
            and replay['independent_rule_certification'] is False, 'Fixed complete replay audit is incomplete or reselected')
    require(audit['schema']=='riichi-ai-stage3g-learning-audit-v1' and audit['valid'] is True
            and audit['optimizer_steps_performed']==0 and audit['new_matches']==0
            and audit['config_sha256']==digest(ROOT/'configs/stage3g.json')
            and audit['corpus_manifest_sha256']==lifecycle['phases']['corpus']['manifest_sha256']
            and audit['development_manifest_sha256']==development['manifest_sha256']
            and audit['arms_manifest_sha256']==digest(REPORTS/'arms.json')
            and audit['evaluation_report_sha256']==digest(REPORTS/'development.json'),
            'Independent numerical audit identity or scope differs')
    manifests = [path for path in REPORTS.glob('*.json') if path not in (MANIFEST,VERIFICATION)
                 and digest(path)==audit['source_manifest_sha256']]
    require(len(manifests)==1 and isinstance(read(manifests[0]).get('sources'),dict), 'Independent audit source manifest missing')
    la, ga, da = audit['learning'],audit['grp'],audit['development']
    require(la['valid'] is True and ga['valid'] is True and da['valid'] is True
            and la['common_initial_weights_sha256']==learning['common_initial_weights_sha256']
            and la['training_matches']==3200 and la['common_group_and_row_order'] is True
            and [a['arm'] for a in la['arms']]==learning['required_arms'], 'Independent learner checks incomplete')
    for arm in la['arms']:
        expected = learning['arms'][arm['arm']]
        require(arm['scheduled_updates']==400 and arm['optimizer_updates']==expected['optimizer_updates']
                and arm['epochs']==2 and arm['training_matches']==3200
                and arm['statistics']==expected['statistics']
                and len(arm['checkpoints'])==5 and all(c['tensor_hash_verified'] is True for c in arm['checkpoints']),
                'Independent checkpoint/update audit differs')
    require(ga['calibration_gate_passed'] is learning['calibration_gate_passed'] and ga['matches']==4000
            and ga['models_verified']==6 and ga['inference_sample_games']==96,
            'Independent placement audit has wrong model/sample/gate coverage')
    require(da['games']==development['games'] and da['blocks']==64 and da['rotations']==4
            and da['selected_candidate']==development['selected_candidate']
            and da['best_observed_arm']==development['best_observed_arm']
            and da['manifest_sha256']==development['manifest_sha256']
            and da['shared_physical_wall_pairs_checked']==development['common_ordinal_wall_pairs']
            and da['policy_promotion'] is False and da['confirmation_started'] is False,
            'Independent development arithmetic or pairing audit differs')
    # A saved valid report must be associated with completed charged execution.
    operations = read(roots['external']/'budget-ledger.json')['operations']
    for path in (corpus_path,replay_path,learning_path):
        aliases = {str(path),path.relative_to(ROOT).as_posix()}
        matched = [op for op in operations if op['status']=='completed'
                   and aliases.intersection(read(REPORTS/f"operation-{op['name']}.json").get('command',[]))]
        require(matched, 'Numerical audit report has no completed leased producing command: '+path.name)
    return dict(valid=True,corpus_audit_sha256=digest(corpus_path),replay_audit_sha256=digest(replay_path),
                learning_audit_sha256=digest(learning_path),audit_source_manifest_sha256=audit['source_manifest_sha256'],
                corpus_rows_audited=corpus['audited_rows'],complete_match_replays=40,
                checkpoint_tensor_hashes_verified=5*len(learning['required_arms']),grp_models_verified=6,
                grp_inference_sample_games=96,independent_rule_certification=False,
                optimizer_steps_performed=0,new_matches_in_learning_audit=0)


def packaging_state(config, plan, roots, *, final=True):
    require(config['schema']=='stage3g-learning-config-v1' and plan['schema']=='stage3g-fixed-learning-plan-v1'
            and config['policy_promotion'] is False and config['confirmation_started'] is False
            and plan['policy_promotion'] is False, 'Fixed Stage 3G scope differs')
    for name, expected in plan['sources'].items():
        require(digest(safe_path(ROOT,name))==expected, 'Frozen formal source changed: '+name)
    for stem in ('reference','call'):
        require(digest(Path(config[stem+'_checkpoint']))==config[stem+'_sha256'], 'Historical input checkpoint changed: '+stem)
    fixture = 'tests/fixtures/stage1/cases.json'
    prior_fixture = subprocess.check_output(['git','show',HISTORICAL_COMMIT+':'+fixture],cwd=ROOT)
    require(hashlib.sha256(prior_fixture).hexdigest()==digest(ROOT/fixture), 'Inherited explicit-wall fixtures changed since Stage 3F')
    previous = read(REPORTS/'previous-stage3f-snapshot.json')
    require(previous['valid'] is True and previous['verified_files']==9493
            and previous['verified_project_files']==125 and previous['verified_external_files']==9368
            and previous['verified_bytes_excluding_inventory']==337077890
            and previous['snapshot_commit']==config['historical_stage3f_commit']==HISTORICAL_COMMIT
            and previous['manifest_sha256']==HISTORICAL_MANIFEST, 'Historical Stage 3F verification receipt differs')
    budget = budget_state(config, roots, plan)
    lifecycle = lifecycle_state(config,plan,roots,budget)
    state = dict(budget=budget, lifecycle=lifecycle, historical_snapshot_files_previously_verified=9493,
                 config_sha256=digest(ROOT/'configs/stage3g.json'), plan_sha256=digest(REPORTS/'plan.json'),
                 previous_stage3f_receipt_sha256=digest(REPORTS/'previous-stage3f-snapshot.json'),
                 policy_promotion=False, confirmation_started=False)
    if final:
        require(not budget['formal_phase_failed'] and not lifecycle['unclosed_starts'] and not lifecycle['incomplete_matches']
                and all(p['complete'] for p in lifecycle['phases'].values()) and lifecycle['all_replays_complete'],
                'Final learning package requires closed complete corpus, preflight and all fixed replays')
        state['learning'] = learning_state(config,plan,roots)
        state['development'] = development_state(config,roots,state['learning'])
        state['numerical_audits'] = numerical_audit_state(config,plan,roots,lifecycle,state['learning'],state['development'],budget)
        expected_formal = ['collect','grp-fit',*[f'train-{name}' for name in state['learning']['required_arms']],'development']
        require(budget['completed_formal_phases']==expected_formal, 'Required formal phase completion differs')
        state['completed_predeclared_protocol'] = True
        state['factorial_complete'] = state['learning']['factorial_complete']
    size = sum(p.stat().st_size for base in (roots['external'],REPORTS) for p in base.rglob('*') if p.is_file() and p not in (MANIFEST,VERIFICATION))
    require(size < config['artifact_bytes']==8_000_000_000, 'Artifact ceiling exceeded')
    state['artifact_bytes_excluding_inventory'] = size
    return state


def create():
    require(not MANIFEST.exists(), 'Inventory already exists; verify instead')
    config,plan,roots,selected = selected_files()
    state = packaging_state(config,plan,roots)
    entries = entries_for(roots,selected)
    result = dict(schema=SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
                  working_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  roots={k:str(v) for k,v in roots.items()}, packaging_state=state,
                  explicit_auxiliary_project_members=list(AUXILIARY_PROJECT_PATHS),
                  file_count=len(entries), total_bytes=sum(e['bytes'] for e in entries),
                  scope_counts=dict(Counter(e['scope'] for e in entries)), files=entries,
                  exclusions=['This manifest and inventory-verification.json.','External runtime lock files.','Historical inputs remain pinned to the separately verified Stage 3F snapshot.'],
                  limitations=['File integrity and protocol accounting do not establish playing strength or independently certify rules.','Checkpoint numerical correctness belongs to separately recorded numerical audits; this command does not deserialize model state.','Calibration-gated fallback completion does not mean all four factorial arms ran.','Current HEAD may precede the result commit; hashes bind the final bytes.'])
    payload = json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n'
    require(state['artifact_bytes_excluding_inventory']+len(payload.encode())+1_000_000 < config['artifact_bytes'], 'Final inventory reserve exceeds artifact cap')
    with MANIFEST.open('x') as handle:
        handle.write(payload)
    return dict(created=True,file_count=len(entries),total_bytes=result['total_bytes'],scope_counts=result['scope_counts'],
                manifest_sha256=digest(MANIFEST),completed_predeclared_protocol=True,factorial_complete=state['factorial_complete'])


def verify():
    inventory = read(MANIFEST)
    config,plan,roots,selected = selected_files()
    require(inventory['schema']==SCHEMA and inventory['roots']=={k:str(v) for k,v in roots.items()}
            and inventory['explicit_auxiliary_project_members']==list(AUXILIARY_PROJECT_PATHS), 'Inventory roots or schema changed')
    require(inventory['packaging_state']==packaging_state(config,plan,roots), 'Final packaging state changed')
    known,unique,errors = {scope:set() for scope in roots},set(),[]
    for entry in inventory['files']:
        scope = entry['scope']
        require(scope in roots, 'Unknown inventory scope')
        path = safe_path(roots[scope],entry['path'])
        require(path not in known[scope] and path.resolve() not in unique and type(entry['bytes']) is int and entry['bytes']>=0,
                'Duplicate artifact or invalid size')
        known[scope].add(path);unique.add(path.resolve())
        if not path.is_file() or path.stat().st_size!=entry['bytes'] or digest(path)!=entry['sha256']:
            errors.append(dict(scope=scope,path=entry['path'],reason='missing or changed'))
    extra = {scope:sorted(str(p) for p in selected[scope]-known[scope]) for scope in roots}
    missing = {scope:sorted(str(p) for p in known[scope]-selected[scope]) for scope in roots}
    counts = dict(Counter(e['scope'] for e in inventory['files']))
    totals = inventory['file_count']==len(inventory['files']) and inventory['scope_counts']==counts and inventory['total_bytes']==sum(e['bytes'] for e in inventory['files'])
    actual = sum(p.stat().st_size for base in (roots['external'],REPORTS) for p in base.rglob('*') if p.is_file())
    require(actual < config['artifact_bytes'], 'Bundle including inventory exceeds artifact cap')
    return dict(valid=not errors and not any(extra.values()) and not any(missing.values()) and totals,
                files_checked=len(inventory['files']),total_bytes=inventory['total_bytes'],scope_counts=counts,counts_match=totals,
                errors=errors,extra_scope_files=extra,missing_scope_files=missing,manifest_sha256=digest(MANIFEST),
                scope='Live Stage 3G bundle; earlier Stage 3F commit/external verification receipt retained.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--create',action='store_true')
    group.add_argument('--verify',action='store_true')
    group.add_argument('--precheck',action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    require(not args.output or (args.verify and Path(args.output).resolve()==VERIFICATION), 'Only excluded verification output is permitted')
    if args.precheck:
        config,plan,roots,_ = selected_files(final=False)
        result = dict(precheck_valid=True, **packaging_state(config,plan,roots,final=False))
    else:
        result = create() if args.create else verify()
    if args.output:
        Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))
    return 0 if result.get('created') or result.get('valid') or result.get('precheck_valid') else 1


if __name__=='__main__':
    raise SystemExit(main())
