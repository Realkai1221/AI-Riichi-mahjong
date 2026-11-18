"""Outcome-independent saved-match selection for the bounded Stage 3H diagnosis."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ordered(names, split):
    return sorted(names, key=lambda name: hashlib.sha256(
        ('riichi-ai/stage3h/v1|' + split + '|' + name).encode()).hexdigest())


def prepare(output, sources):
    if not os.environ.get('RIICHI_STAGE3H_TOKEN'):
        raise RuntimeError('Selection must run inside the numerical ledger')
    import numpy as np
    from riichi_ai.stage3g_data import LearningDataset
    config = json.loads((ROOT / 'configs/stage3h.json').read_text())
    old = json.loads((ROOT / 'configs/stage3g.json').read_text())
    data = Path(old['data_root'])
    corpus = data / 'corpus'
    preflight = data / 'preflight'
    manifest = json.loads((corpus / 'manifest.json').read_text())
    selected = {split: ordered(manifest['splits'][split], split)[:128]
                for split in ('training', 'development')}
    models = [dict(id='reference', kind='reference', path=config['reference_checkpoint'],
        sha256=config['reference_sha256'], arm=None, update=0,
        reference_sha256=config['reference_sha256'])]
    initial = data / 'learning/mc_a0/initial.pt'
    models.append(dict(id='initial', kind='value', path=str(initial), sha256=sha(initial),
        arm='mc_a0', update=0, reference_sha256=config['reference_sha256']))
    arms = ('mc_a0', 'mc_a1', 'delta_a0', 'delta_a1')
    for arm in arms:
        for update in (100, 200, 300, 400):
            path = data / 'learning' / arm / ('final.pt' if update == 400 else f'update-{update:04d}.pt')
            models.append(dict(id=f'{arm}_u{update}', kind='value', path=str(path), sha256=sha(path),
                               arm=arm, update=update, reference_sha256=config['reference_sha256']))
    inputs = {model['path']: model['sha256'] for model in models}
    targets = data / 'grp/hand-targets.json'
    for path in (corpus / 'manifest.json', preflight / 'manifest.json', targets):
        inputs[str(path)] = sha(path)
    counts = {}
    for split, names in selected.items():
        dataset = LearningDataset(corpus, split)
        counts[split] = {}
        for name in names:
            actor, labels, _ = dataset._load(name)
            counts[split][name] = int(((labels['behavior_id'] == 3) &
                                     (np.diff(actor['candidate_offsets']) > 1)).sum())
            for area in ('actor', 'targets'):
                relative = f'{area}/{name}.npz'
                inputs[str(corpus / relative)] = dataset.reports[name]['artifacts'][relative]['sha256']
    dataset = LearningDataset(preflight, 'preflight')
    rows = []
    for name in sorted(dataset.match_ids):
        actor, labels, _ = dataset._load(name)
        eligible = np.flatnonzero((labels['behavior_id'] == 3) & (np.diff(actor['candidate_offsets']) > 1))
        chosen = sorted(map(int, eligible), key=lambda i: hashlib.sha256(
            f'riichi-ai/stage3h/preflight-v1|{name}|{i}'.encode()).hexdigest())[:64]
        rows.extend(dict(match_id=name, row_index=i) for i in sorted(chosen))
        for area in ('actor', 'targets'):
            relative = f'{area}/{name}.npz'
            inputs[str(preflight / relative)] = dataset.reports[name]['artifacts'][relative]['sha256']
    first = selected['training'][:64]
    conditions = [dict(id='initial_mc', checkpoint_id='initial', target='terminal_utility', trained_alpha=None),
                  dict(id='initial_delta', checkpoint_id='initial', target='one_hand_delta', trained_alpha=None)]
    for arm in arms:
        conditions.append(dict(id=arm, checkpoint_id=arm+'_u400',
            target='terminal_utility' if arm.startswith('mc') else 'one_hand_delta', trained_alpha=int(arm[-1])))
    plan = dict(schema='stage3h-diagnostic-plan-v1', execution_admitted=False,
        selection_rule='SHA256(riichi-ai/stage3h/v1|split|match_id), ascending; no outcomes used',
        config_sha256=sha(ROOT / 'configs/stage3h.json'), corpus_path=str(corpus),
        preflight_corpus_path=str(preflight), corpus_manifest_sha256=sha(corpus / 'manifest.json'),
        preflight_manifest_sha256=sha(preflight / 'manifest.json'), hand_targets_path=str(targets),
        hand_targets_sha256=sha(targets), models=models, selected_matches=selected,
        optional_rows_by_match=counts, candidate_split_sizes=[128, 96, 64],
        preflight_rows=rows, reference_model_id='reference', initial_model_id='initial', workers=14,
        gradient_groups=[first[i:i+16] for i in range(0,64,16)], gradient_conditions=conditions,
        inputs=inputs, sources=json.loads(Path(sources).read_text())['sources'],
        optimizer_steps=0, new_matches=0, policy_promotion=False)
    with Path(output).open('x') as handle:
        json.dump(plan, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(selected_matches={k:len(v) for k,v in selected.items()},
        optional_rows={k:sum(v.values()) for k,v in counts.items()}, preflight_rows=len(rows), models=len(models))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--source-manifest', required=True)
    args = p.parse_args()
    prepare(args.output, args.source_manifest)
