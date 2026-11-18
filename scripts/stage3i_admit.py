"""Outcome-blind measured resource admission for the fixed Stage 3I programme."""
import argparse,hashlib,json,math,os
from pathlib import Path
from stage3i import read_config,digest,verify_sources

def read(p):return json.loads(Path(p).read_text())
def write(p,value):
    with Path(p).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--output-plan',required=True);ap.add_argument('--report',required=True);args=ap.parse_args()
    c,ci=read_config();root=Path(c['data_root']);reports=Path(c['reports_root']);ledger=read(root/'budget-ledger.json');active=[o for o in ledger['operations'] if o['status']=='active']
    if len(active)!=1 or active[0]['token']!=os.environ.get('RIICHI_STAGE3I_TOKEN') or not active[0]['name'].startswith('preflight'):raise ValueError('Exclusive preflight numerical lease required')
    p=read(args.plan);verify_sources(p['sources']);f=read(reports/'preflight-features.json');t=read(root/'training-preflight/preflight.json');g=read(reports/'preflight-games.json')
    if not t['passed'] or not g['complete'] or f['max_probability_error']!=0:raise ValueError('Incomplete or failed prerequisites')
    cache=read(root/'preflight-cache/manifest.json');rows={k:sum(s['rows'] for s in cache['shards'] if s['partition']==k) for k in ['training','selection']}
    # Outcome-free: no losses, gate metrics, ranks, scores or candidate wins used.
    # Forecast uses the complete cold-started benchmark and a 50% margin.
    feature_seconds=f['full_4000_match_time_forecast_seconds']*1.5
    feature_bytes=f['full_4000_match_size_forecast_bytes']*1.25
    evaluation_seconds=g['elapsed_seconds']*9912/g['completed_games']*1.5+30
    game_bytes=sum(p.stat().st_size for p in (root/'preflight-games').rglob('*') if p.is_file())*9912/g['completed_games']*1.5
    current_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    artifact_forecast=feature_bytes+game_bytes+current_bytes+500_000_000
    # Conservative per-head training+selection forecast, plus full cache reads,
    # double measured timing and 60 seconds checkpoint/bookkeeping allowance.
    estimates={}
    for backend,heads in t['timings'].items():
      estimates[backend]={}
      for epochs in [12,8,4]:
        train=sum(v['step_seconds']*461 for v in heads.values())*epochs*3
        selection=sum(v['selection_seconds'] for v in heads.values())*(200/16)*epochs*3
        cache_load=t['cache_load_seconds']*(3200+200)/32
        estimates[backend][str(epochs)]=2*(train+selection+cache_load)+60
    options=[(ep,estimates[b][str(ep)],b) for ep in [12,8,4] for b in estimates if estimates[b][str(ep)]<=1750]
    if not options:raise ValueError('No predeclared training level fits')
    epochs,forecast,backend=sorted(options,key=lambda x:(-x[0],x[1],x[2]))[0]
    gates={'feature_time':feature_seconds<=580,'evaluation_time':evaluation_seconds<=1770,'artifact_size':artifact_forecast<c['artifact_bytes']}
    result=dict(schema='stage3i-measured-admission-v1',admitted=all(gates.values()),gates=gates,epochs=epochs,device=backend,training_forecasts_seconds=estimates,selected_training_forecast_seconds=forecast,feature_forecast_seconds=feature_seconds,evaluation_forecast_seconds=evaluation_seconds,artifact_forecast_bytes=artifact_forecast,feature_forecast_bytes=feature_bytes,game_forecast_bytes=game_bytes,methods='Cold-start inclusive 1.5x feature/game time;1.25x featurebytes;1.5x gamebytes;2x actual optimizer+selection/cache training time plus60s;500MB remaining-artifact margin. No outcome/accuracy quantities used.',inputs={str(x):digest(x) for x in [Path(args.plan),reports/'preflight-features.json',root/'training-preflight/preflight.json',reports/'preflight-games.json']})
    write(args.report,result)
    if not result['admitted']:raise ValueError('Measured workload does not fit frozen scope')
    p.update(feature_execution_admitted=True,epochs=epochs,device=backend,training_preflight_path=str(root/'training-preflight/preflight.json'),preflight_cache_manifest_sha256=digest(root/'preflight-cache/manifest.json'),admission_path=str(Path(args.report).resolve()),admission_sha256=digest(args.report))
    write(args.output_plan,p)
    print(json.dumps({k:v for k,v in result.items() if k not in ['inputs','training_forecasts_seconds']}),flush=True)
if __name__=='__main__':main()
