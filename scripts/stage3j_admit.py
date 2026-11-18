"""Resource-only admission plus immutable historical eligibility checks."""
import argparse
import json
import os
from pathlib import Path
import time
from stage3j import read_config, digest, verify_sources

def read(path): return json.loads(Path(path).read_text())
def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False); handle.write('\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--output-plan',required=True);ap.add_argument('--report',required=True);args=ap.parse_args()
    config,identity=read_config();root=Path(config['data_root']);reports=Path(config['reports_root'])
    active=[o for o in read(root/'budget-ledger.json')['operations'] if o['status']=='active']
    if len(active)!=1 or active[0]['token']!=os.environ.get('RIICHI_STAGE3J_TOKEN') or not active[0]['name'].startswith('preflight'):
        raise ValueError('Exclusive preflight numerical lease required')
    plan=read(args.plan);verify_sources(plan['sources'])
    if plan['config_sha256']!=identity: raise ValueError('Wrong campaign identity')
    for dependency in plan['dependencies']:
        if digest(dependency['path'])!=dependency['sha256']: raise ValueError('Historical dependency changed')
    calibration=read(reports.parent/'stage3i/calibration.json')
    diagnostic=read(reports.parent/'stage3i/posthoc-fractional-diagnostic.json')
    if not calibration['complete'] or not diagnostic['complete'] or not diagnostic['posthoc']:
        raise ValueError('Historical rationale incomplete')
    if [x['seed'] for x in calibration['seeds']]!=plan['training_seeds'] or [x['seed'] for x in diagnostic['seeds']]!=plan['training_seeds']:
        raise ValueError('Every saved seed must be retained')
    eligibility=[]
    for bundle,cal,diag in zip(plan['selected_bundles'],calibration['seeds'],diagnostic['seeds']):
        quarter=next(x for x in diag['grid'] if x['lambda']==.25)
        if (not cal['gate']['passed'] or cal['sha256']!=bundle['sha256']
                or quarter['change_rate']>.03 or quarter['reference_discard_change_rate']>.10):
            raise ValueError('Archived seed does not satisfy fixed design rationale')
        eligibility.append(dict(seed=bundle['seed'],predictive_gate_passed=True,quarter_strength=quarter,
                                evidence_role='reused post-hoc development rationale; no new tuning'))
    game=read(reports/'preflight-games.json')
    if not game['complete'] or game['completed_games']!=64 or game['attempted_games']!=64:
        raise ValueError('Full 64-game path must pass before admission')
    elapsed=game['elapsed_seconds']
    preflight_bytes=sum(p.stat().st_size for p in (root/'preflight-games').rglob('*') if p.is_file())
    current=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    seconds=elapsed*8192/64*1.5+30
    artifact=current+preflight_bytes*8192/64*1.5+500_000_000
    gates=dict(full_game_panel=True,time_forecast=seconds<=3500,artifact_forecast=artifact<config['artifact_bytes'])
    result=dict(schema='stage3j-measured-admission-v1',admitted=all(gates.values()),gates=gates,
        evaluation_forecast_seconds=seconds,artifact_forecast_bytes=artifact,
        preflight_elapsed_seconds=elapsed,preflight_bytes=preflight_bytes,
        preflight_sha256=digest(reports/'preflight-games.json'),
        eligibility=eligibility,
        method='Complete cold-started 64-game elapsed/bytes times128 times1.5; add30seconds and500MB; no ranks/returns/accuracy used for sizing or arm selection')
    write(args.report,result)
    if not result['admitted']: raise ValueError('Fixed full workload does not fit; no adaptive panel reduction')
    plan.update(execution_admitted=True,admission_path=str(Path(args.report).resolve()),admission_sha256=digest(args.report))
    write(args.output_plan,plan)
    print(json.dumps({k:result[k] for k in ('admitted','evaluation_forecast_seconds','artifact_forecast_bytes')}),flush=True)

if __name__=='__main__':main()
