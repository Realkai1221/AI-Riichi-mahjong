"""Fixed block-level Stage 3J analysis, with every seed retained."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from stage3j import read_config,digest,verify_sources

PRIMARY_NAMES=('learned_tie_vs_reference','learned_tie_vs_hash','small_vs_reference','small_vs_learned_tie')

def analyze(execution,records,*,bootstrap_replicates=10000):
    """Average correlated seats/seeds inside blocks; jointly resample blocks."""
    seeds=execution['fixed_seed_family'];blocks=execution['blocks']
    names=[a['name'] for a in execution['arms']]
    if len(records)!=blocks*4*8 or len({r['match_id'] for r in records})!=len(records):
        raise ValueError('Complete unique eight-arm panel required')
    expected={j['match_id']:j for j in execution['matches']}
    table={}
    placements={a:[0]*4 for a in names}
    for row in records:
        if (row['match_id'] not in expected or any(row.get(k)!=v for k,v in expected[row['match_id']].items())
                or row['completed'] is not True or row['failure'] is not None
                or row['training_eligible'] is not False or row['ppo_eligible'] is not False):
            raise ValueError('Invalid panel membership or eligibility')
        rank=row['ranks'][row['focal_seat']];u=row['utilities'][row['focal_seat']]
        if type(rank) is not int or rank not in (1,2,3,4) or u!=2.5-rank:
            raise ValueError('Rank utility differs from declared objective')
        table[row['block'],row['focal_seat'],row['arm']]=u
        placements[row['arm']][rank-1]+=1
    utilities={a:np.asarray([sum(table[b,f,a] for f in range(4))/4 for b in range(blocks)],dtype=np.float64) for a in names}
    tie=np.mean([utilities[f'tie_seed_{s}'] for s in seeds],axis=0)
    small=np.mean([utilities[f'small_seed_{s}'] for s in seeds],axis=0)
    vectors=np.stack([tie-utilities['reference_wrapped'],tie-utilities['hash_tie'],small-utilities['reference_wrapped'],small-tie],axis=1)
    rng=np.random.default_rng(20261029031)
    indices=rng.integers(0,blocks,size=(bootstrap_replicates,blocks))
    samples=vectors[indices].mean(axis=1)
    half=6*math.sqrt(math.log(2*4/.05)/(2*blocks))
    primary={}
    for i,name in enumerate(PRIMARY_NAMES):
        vector=vectors[:,i];mean=float(vector.mean());sd=float(vector.std(ddof=1))
        primary[name]=dict(mean_utility_difference=mean,block_standard_deviation=sd,standard_error=sd/math.sqrt(blocks),
            bootstrap_9875_interval=np.quantile(samples[:,i],[.00625,.99375]).tolist(),
            simultaneous_95_hoeffding=[max(-3.,mean-half),min(3.,mean+half)],
            block_utility_differences=vector.tolist(),independent_blocks=blocks)
    return dict(schema='stage3j-fixed-statistics-v1',complete=True,independent_blocks=blocks,games=len(records),
        primary=primary,by_arm={a:dict(mean_rank=2.5-float(u.mean()),mean_utility=float(u.mean()),
            placement_counts=placements[a],games=blocks*4) for a,u in utilities.items()},
        pooled_mean_rank=dict(learned_tie=2.5-float(tie.mean()),small=2.5-float(small.mean())),
        inference=dict(bootstrap_seed=20261029031,bootstrap_replicates=bootstrap_replicates,
            confidence_per_contrast=.9875,family_size=4,joint_block_resampling=True,
            interpretation='Bonferroni percentile bootstrap is approximate; simultaneous Hoeffding bounds are conservative; no seed selection or promotion'),
        policy_promotion=False,seed_selection=False)

def read(path):return json.loads(Path(path).read_text())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--evaluation-report',required=True);ap.add_argument('--report',required=True);args=ap.parse_args()
    config,ci=read_config();active=[o for o in read(Path(config['data_root'])/'budget-ledger.json')['operations'] if o['status']=='active']
    if len(active)!=1 or active[0]['token']!=os.environ.get('RIICHI_STAGE3J_TOKEN') or not active[0]['name'].startswith('audit'):
        raise ValueError('Exclusive audit lease required')
    plan=read(args.plan);verify_sources(plan['sources'])
    result=read(args.evaluation_report)
    if not result['complete'] or digest(result['manifest_path'])!=result['manifest_sha256']:
        raise ValueError('Incomplete or changed evaluation')
    manifest=read(result['manifest_path']);execution=manifest['execution_plan']
    if execution['plan_sha256']!=digest(args.plan) or execution['phase']!='formal': raise ValueError('Wrong formal plan')
    statistics=analyze(execution,manifest['matches'])
    statistics.update(plan_sha256=digest(args.plan),evaluation_report_sha256=digest(args.evaluation_report),
                      evaluation_manifest_sha256=result['manifest_sha256'])
    with Path(args.report).open('x') as handle:json.dump(statistics,handle,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps({k:v['mean_utility_difference'] for k,v in statistics['primary'].items()}),flush=True)

if __name__=='__main__':main()
