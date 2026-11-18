"""Stdlib-only source/history freezing; never run models, games or statistics."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/stage3j'

def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False); handle.write('\n')

def extract_history(value, seeds, walls):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'seed' or key == 'seeds' or key.endswith('_seeds'):
                if type(item) is int: seeds.add(item)
                elif isinstance(item, list): seeds.update(x for x in item if type(x) is int)
            if key == 'ordinal_wall_fingerprints' and isinstance(item, list) and item:
                if isinstance(item[0], str): walls.add(item[0])
            extract_history(item, seeds, walls)
    elif isinstance(value, list):
        for item in value: extract_history(item, seeds, walls)

def history(external):
    paths = set(p for p in (ROOT/'reports').rglob('*.json') if not p.is_relative_to(REPORTS))
    for stage in external.parent.iterdir():
        if not stage.is_dir() or stage == external: continue
        paths.update(stage.glob('**/reports/*.json'))
        paths.update(stage.glob('**/execution-plan.json'))
        paths.update(stage.glob('*/manifest.json'))
    seeds, walls, records = set(), set(), []
    for path in sorted(paths):
        if path.is_symlink(): raise ValueError('Historical input is a symlink')
        raw = path.read_bytes(); value = json.loads(raw)
        extract_history(value, seeds, walls)
        records.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest()))
    return dict(schema='stage3j-historical-inputs-v1', inputs=records,
                extraction_rule="Recursively collect integer seed/seeds/*_seeds values (or integer list elements), and first ordinal_wall_fingerprints string; includes a conservative superset of model/RNG seeds",
                historical_seeds=sorted(seeds), historical_first_wall_fingerprints=sorted(walls))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name', required=True);args=parser.parse_args()
    if not args.name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in args.name):
        raise ValueError('Invalid release name')
    config_path=ROOT/'configs/stage3j.json';config=json.loads(config_path.read_text())
    external=Path(config['data_root']);external.mkdir(parents=True,exist_ok=True)
    history_path=REPORTS/'historical-inputs.json'
    if not history_path.exists(): write(history_path,history(external))
    h=json.loads(history_path.read_text())
    old=json.loads((ROOT/'reports/stage3i/plan.json').read_text())
    calibration=json.loads((ROOT/'reports/stage3i/calibration.json').read_text())
    bundles=[dict(seed=s['seed'],checkpoint=s['checkpoint'],sha256=s['sha256']) for s in calibration['seeds']]
    sources={}
    paths=sorted(set([* (ROOT/'src').rglob('*.py'), * (ROOT/'tests').rglob('*.py'),
        * (ROOT/'scripts').glob('stage3j*.py'), ROOT/'scripts/package_a_stability.py',
        ROOT/'configs/stage3j.json', ROOT/'docs/STAGE3J_PLAN.md', ROOT/'pyproject.toml']))
    # Preserve exact bytes for every numerical release, including failed preflights.
    archive=external/'source-archive';archive.mkdir(exist_ok=True)
    for path in paths:
        raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest();sources[str(path.relative_to(ROOT))]=sha
        dest=archive/(sha+'.bin')
        if dest.exists():
            if digest(dest)!=sha: raise ValueError('Archived source changed')
        else: dest.write_bytes(raw)
    dependencies=[]
    for path,role in [
        (ROOT/'reports/stage3i/manifest.json','historical_inventory'),
        (ROOT/'reports/stage3i/plan.json','source_protocol_identity'),
        (ROOT/'reports/stage3i/calibration.json','archived_predictive_gate'),
        (ROOT/'reports/stage3i/posthoc-fractional-diagnostic.json','posthoc_design_rationale'),
        (Path(old['cache_manifest_path']),'source_cache_manifest'),
        (Path(config['reference_checkpoint']),'original_reference'),
        (Path(config['call_checkpoint']),'original_explorer_call_heads'),
        (history_path,'historical_seed_wall_inventory')]:
        dependencies.append(dict(path=str(path),sha256=digest(path),role=role))
    for bundle in bundles:
        if digest(bundle['checkpoint'])!=bundle['sha256']: raise ValueError('Saved head changed')
        dependencies.append(dict(path=bundle['checkpoint'],sha256=bundle['sha256'],role='saved_residual_head'))
    evaluation_seeds=list(range(202610290000,202610290256));preflight=[202610289000,202610289001]
    if set(evaluation_seeds+preflight)&set(h['historical_seeds']): raise ValueError('Historical seed reuse')
    protocol=ROOT/'docs/STAGE3J_PLAN.md'
    plan=dict(schema='stage3j-plan-v1',date='2026-10-29',execution_admitted=False,
        config_sha256=digest(config_path),protocol_path=str(protocol),protocol_sha256=digest(protocol),
        training_seeds=config['seeds'],selected_bundles=bundles,
        source_identity={k:old[k] for k in ('protocol_sha256','cache_manifest_sha256','reference_sha256','call_sha256')},
        evaluation_seeds=evaluation_seeds,preflight_evaluation_seeds=preflight,
        forward_sample_blocks=[0,32,64,96,128,160,192,224],
        replay_selection=dict(salt='stage3j-audit-replay:',per_arm=4),
        freshness=dict(historical_seed_sources=[dict(path=str(history_path),sha256=digest(history_path))],
            historical_seeds=h['historical_seeds'],historical_first_wall_fingerprints=h['historical_first_wall_fingerprints']),
        dependencies=dependencies,sources=sources,
        reference_checkpoint=config['reference_checkpoint'],reference_sha256=config['reference_sha256'],
        call_checkpoint=config['call_checkpoint'],call_sha256=config['call_sha256'],
        policy_promotion=False,new_optimizer_updates=0,new_training_matches=0)
    write(REPORTS/(args.name+'-sources.json'),dict(sources=sources))
    write(REPORTS/(args.name+'-plan.json'),plan)
    print(json.dumps(dict(release=args.name,sources=len(sources),historical_inputs=len(h['inputs']),historical_seeds=len(h['historical_seeds']),historical_first_walls=len(h['historical_first_wall_fingerprints']))))

if __name__=='__main__':main()
