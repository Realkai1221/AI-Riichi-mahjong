"""Run frozen Stage 2B development/conditional confirmation or CPU benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from riichi_ai.evaluation import _seed, evaluate_paired, source_manifest
from riichi_ai.stage2b_runtime import (benchmark_campaign, corpus_and_historical_wall_seeds,
                                     file_hash, save_json, verify_paired_identity)
from riichi_ai.statistics import noninferiority_test, plan_noninferiority

PROJECT = Path(__file__).resolve().parents[1]


def evaluate(plan: dict, selection: dict, reports: Path) -> dict:
    start = time.perf_counter()
    frozen = Path(selection['checkpoint'])
    if file_hash(frozen) != selection['checkpoint_sha256']:
        raise ValueError('Frozen final checkpoint hash mismatch')
    if (reports / 'evaluation-summary.json').exists():
        raise FileExistsError('Evaluation is already complete; review the saved summary instead of rerunning')
    eligible = selection['eligible_recipes'][:2]
    if len(eligible) != 2 or any(len(recipe['runs']) != 2 for recipe in eligible):
        raise ValueError('Four complete development candidates required')
    earlier = corpus_and_historical_wall_seeds(plan)
    development = {_seed(plan['seed_namespaces']['development'], 'block', b)
                   for b in range(plan['evaluation']['pilot_blocks'])}
    confirmation = {_seed(plan['seed_namespaces']['confirmation'], 'block', b)
                    for b in range(plan['limits']['confirmation_max_blocks'])}
    if development & confirmation or (development | confirmation) & earlier:
        raise ValueError('Evaluation seed namespaces overlap recorded corpus/historical matches')
    campaign_plan = {
        'protocol_sha256': hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
        'protocol': plan, 'source_manifest': source_manifest(),
        'final_checkpoint_sha256': selection['checkpoint_sha256'],
        'selection_sha256': file_hash(reports / 'selection.json'),
        'eligible_recipes': eligible, 'seed': plan['seed_namespaces']['development'],
        'blocks_each': plan['evaluation']['pilot_blocks'], 'frozen_before_outcomes': True,
        'common_blocks_across_candidates': True,
        'historical_and_corpus_seeds_checked': len(earlier), 'seed_overlap_count': 0,
        'unique_wall_seeds': [_seed(plan['seed_namespaces']['development'], 'block', b)
                              for b in range(plan['evaluation']['pilot_blocks'])],
    }
    target = reports / 'development-plan.json'
    if target.exists():
        if json.loads(target.read_text()) != campaign_plan:
            raise ValueError('Previously frozen development plan differs')
    else:
        save_json(target, campaign_plan)
    summaries = []
    final_pilot = None
    common = dict(baseline=plan['evaluation']['baseline'], opponents=plan['opponents'],
                  workers=plan['evaluation']['workers'], failure_dir=reports / 'failures',
                  memory_limit_bytes=plan['limits']['memory_gib'] * 1024**3,
                  artifact_roots=[Path(plan['data_root']), reports],
                  disk_limit_bytes=plan['limits']['new_artifact_bytes'])
    for recipe in eligible:
        for run in sorted(recipe['runs'], key=lambda row: row['seed']):
            if file_hash(run['checkpoint']) != run['checkpoint_sha256']:
                raise ValueError('Development checkpoint hash mismatch')
            path = reports / f"development-{recipe['name']}-{run['seed']}.json"
            if path.exists():
                result = json.loads(path.read_text())
            else:
                print(json.dumps({'stage': 'development', 'recipe': recipe['name'], 'seed': run['seed'],
                                  'blocks': plan['evaluation']['pilot_blocks']}), flush=True)
                result = evaluate_paired(candidate='checkpoint:' + run['checkpoint'], **common,
                                         blocks=plan['evaluation']['pilot_blocks'],
                                         seed=plan['seed_namespaces']['development'])
                save_json(path, result)
            verify_paired_identity(result, checkpoint=run['checkpoint'], expected_hash=run['checkpoint_sha256'],
                                   plan=plan, root_seed=plan['seed_namespaces']['development'],
                                   blocks=plan['evaluation']['pilot_blocks'])
            summaries.append({'recipe': recipe['name'], 'seed': run['seed'],
                              'checkpoint_sha256': run['checkpoint_sha256'],
                              'statistics': result['statistics'], 'arm_summaries': result['arm_summaries'],
                              'elapsed_seconds': result['elapsed_seconds'], 'path': str(path)})
            if run['checkpoint_sha256'] == selection['checkpoint_sha256']:
                final_pilot = result
            print(json.dumps({'stage': 'development-complete', **summaries[-1]}), flush=True)
    if final_pilot is None:
        raise RuntimeError('Final primary checkpoint is missing from the development panel')
    options = plan['evaluation']['confirmation']
    planning = plan_noninferiority([b['mean_utility_difference'] for b in final_pilot['blocks']],
                                  margin=options['margin'], alpha=options['alpha'],
                                  power=options['planning_power'], variance_floor_sd=options['variance_floor_sd'],
                                  pilot_sd_multiplier=options['pilot_sd_multiplier'],
                                  min_blocks=options['minimum_blocks'], max_blocks=options['maximum_blocks'])
    planning.update(checkpoint_sha256=selection['checkpoint_sha256'],
                    confirmation_root_seed=plan['seed_namespaces']['confirmation'])
    confirmation_plan_path = reports / 'confirmation-plan.json'
    if confirmation_plan_path.exists():
        if json.loads(confirmation_plan_path.read_text()) != planning:
            raise ValueError('Previously frozen confirmation plan differs')
    else:
        save_json(confirmation_plan_path, planning)
    confirmation_result = {'valid': False, 'passed': False, 'status': planning['status'], 'confirmation_run': False}
    if planning['run_confirmation']:
        confirmation_path = reports / 'confirmation.json'
        if confirmation_path.exists():
            result = json.loads(confirmation_path.read_text())
        else:
            result = evaluate_paired(candidate='checkpoint:' + str(frozen), **common,
                                     blocks=planning['confirmation_blocks'], seed=plan['seed_namespaces']['confirmation'])
            save_json(confirmation_path, result)
        verify_paired_identity(result, checkpoint=str(frozen), expected_hash=selection['checkpoint_sha256'],
                               plan=plan, root_seed=plan['seed_namespaces']['confirmation'],
                               blocks=planning['confirmation_blocks'])
        confirmation_result = (noninferiority_test([b['mean_utility_difference'] for b in result['blocks']],
                                                   planned_blocks=planning['confirmation_blocks'],
                                                   margin=options['margin'], alpha=options['alpha'],
                                                   min_blocks=options['minimum_blocks'], max_blocks=options['maximum_blocks'])
                               if result['valid'] else {'valid': False, 'passed': False, 'status': 'invalid_campaign'})
        confirmation_result['confirmation_run'] = True
    save_json(reports / 'noninferiority-result.json', confirmation_result)
    summary = {'development': summaries, 'confirmation_plan': planning,
               'noninferiority': confirmation_result, 'elapsed_seconds': time.perf_counter() - start,
               'final_checkpoint_unchanged': file_hash(frozen) == selection['checkpoint_sha256']}
    save_json(reports / 'evaluation-summary.json', summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['evaluate', 'benchmark'])
    args = parser.parse_args()
    plan = json.loads((PROJECT / 'configs/stage2b.json').read_text())
    reports = PROJECT / plan['reports_root']
    selection = json.loads((reports / 'selection.json').read_text())
    result = (evaluate(plan, selection, reports) if args.command == 'evaluate'
              else benchmark_campaign(plan, selection, reports))
    print(json.dumps({'stage': args.command, 'result': result}), flush=True)


if __name__ == '__main__':
    main()
