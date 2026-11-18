"""Produce compact measured review tables; no extra model or game execution."""
import json,math,os
from pathlib import Path
from stage3i import read_config,digest

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def main():
 c,_=read_config();d=Path(c['data_root']);r=Path(c['reports_root']);active=[o for o in read(d/'budget-ledger.json')['operations'] if o['status']=='active']
 if len(active)!=1 or active[0]['name']!='audit-report' or active[0]['token']!=os.environ.get('RIICHI_STAGE3I_TOKEN'):raise ValueError('Exclusive report-analysis lease required')
 t=read(d/'training/training-report.json');cal=read(r/'calibration.json');e=read(r/'evaluation.json');a=read(r/'audit.json');replays=read(r/'replays.json');diag=read(r/'posthoc-fractional-diagnostic.json')
 if not all(x['complete'] for x in [t,cal,e,a,replays,diag]):raise ValueError('Incomplete evidence')
 seeds=[]
 for trained,cs in zip(t['seeds'],cal['seeds']):
  if trained['seed']!=cs['seed']:raise ValueError('Seed identity mismatch')
  mse=cs['gate']['row_mse'];one=next(row for row in cs['lambda_grid'] if row['lambda']==1)
  seeds.append(dict(seed=cs['seed'],selected_epochs=trained['selected_epochs'],checkpoint=trained['checkpoint'],sha256=trained['sha256'],gate_passed=cs['gate']['passed'],gate_row_mse=mse,gate_match_mse=cs['gate']['match_mse'],va_relative_mse_reduction_vs_v_percent=100*(1-mse['va']/mse['v']),formal_lambda=cs['calibration']['lambda_'],lambda_one_change_rate=one['change_rate'],lambda_one_changed=one['changed'],calibration_rows=one['rows'],maximum_changed_rows_at_3_percent=math.floor(.03*one['rows']),lambda_one_excess_changed_rows=one['changed']-math.floor(.03*one['rows'])))
 manifest=read(e['manifest_path']);records={x['match_id']:x for x in manifest['matches']};identical_games=identical_artifacts=0
 for block in range(470):
  for seat in range(4):
   prefix=f'block-{block:03d}-seat-{seat}-';raw=records[prefix+'reference_raw'];wrapped=records[prefix+'reference_wrapped']
   identical_games+=raw['game_sha256']==wrapped['game_sha256']
   # Artifact includes policy provenance, so identical canonical game events is
   # the useful paired trajectory comparison, not whole-file equality.
 games=[dict(comparison=k,mean_rank=v['mean_rank'],control_mean_rank=v['baseline_mean_rank'],statistics=v['statistics']) for k,v in e['statistics']['contrasts'].items()]
 output=dict(schema='stage3i-review-summary-v1',complete=True,training_device=t['device'],epochs=t['epochs'],optimizer_updates=t['optimizer_updates'],head_row_presentations=t['row_presentations'],unique_training_rows=t['training_cache']['rows'],training_elapsed_seconds=t['elapsed_seconds'],seeds=seeds,formal_games=e['completed_games'],formal_game_rows=e['rows'],formal_game_seconds=e['elapsed_seconds'],reference_pairs=1880,reference_raw_wrapped_identical_canonical_games=identical_games,learned_candidate_games=0,game_comparisons=games,independent_audit=a,replays=replays,posthoc_diagnostic_path=str(r/'posthoc-fractional-diagnostic.json'),policy_promotion=False,inputs={str(p):digest(p) for p in [d/'training/training-report.json',r/'calibration.json',r/'evaluation.json',r/'audit.json',r/'replays.json',r/'posthoc-fractional-diagnostic.json']})
 write(r/'summary.json',output)
 print(json.dumps({'training_seeds':seeds,'reference_pairs_identical_canonical_games':identical_games,'formal_games':e['completed_games'],'independent_audit_complete':a['complete']}),flush=True)
if __name__=='__main__':main()
