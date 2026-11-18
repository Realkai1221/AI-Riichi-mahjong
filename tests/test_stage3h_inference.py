"""Independent small fixtures for diagnostic choices, weighting and checkpoint gates."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from riichi_ai.dataset import _pack_arrays
from riichi_ai.model import _state_hash, collate_encoded
from riichi_ai.stage3g_data import LearningDataset
from riichi_ai.stage3g_learning import RECIPE, CHECKPOINT_SCHEMA, build_value_model
from riichi_ai.stage3h_inference import (KINDS, SLICE_NAMES, load_checkpoint, choices,
    opportunity_mask, probability_bucket, row_metrics, aggregate_matches,
    one_row_actor, evaluate_one, validate_plan, PLAN_SCHEMA, sha)
from test_stage3g_learning import corpus, reference


def test_probability_bucket_boundaries_and_invalid_values():
    actual = probability_bucket(np.asarray([0, .0001, .01, .0100001, .05, .05001, .2, .20001, 1]))
    np.testing.assert_array_equal(actual, [0, 1, 1, 2, 2, 3, 3, 4, 4])
    for value in (-.01, 1.01, float('nan')):
        with pytest.raises(ValueError): probability_bucket(np.asarray([value]))


def test_raw_argmax_and_first_win_wrapper_are_distinct():
    kinds = np.asarray([KINDS['PON'], KINDS['RON'], KINDS['PASS']])
    assert choices(np.asarray([5., 3., 0.]), kinds) == (0, 1)
    assert choices(np.asarray([1., 3., 0.]), kinds) == (1, 1)
    assert choices(np.asarray([1., 1.]), np.asarray([KINDS['DISCARD'], KINDS['PASS']])) == (0, 0)


def test_opportunity_slices_describe_legal_set_not_chosen_action():
    value = opportunity_mask(np.asarray([KINDS['PON'], KINDS['RON'], KINDS['PASS']]))
    present = {name for bit, name in enumerate(SLICE_NAMES) if value & (1 << bit)}
    assert present == {'all', 'call', 'win', 'call_and_noncall'}
    value = opportunity_mask(np.asarray([KINDS['ANKAN'], KINDS['DISCARD'], KINDS['RIICHI']]))
    present = {name for bit, name in enumerate(SLICE_NAMES) if value & (1 << bit)}
    assert present == {'all', 'discard', 'call', 'riichi', 'kan', 'call_and_noncall'}


def fixture_arrays():
    kinds = np.asarray([KINDS['DISCARD'], KINDS['PASS'], KINDS['PON'], KINDS['RON'], KINDS['PASS']], dtype=np.int8)
    scores = np.asarray([[2., 1., 3., 2., 0.], [1., 2., 1., 2., 3.], [.2, .8, .1, .3, .2]], dtype=np.float32)
    offsets = np.asarray([0, 2, 5], dtype=np.int64)
    probabilities = np.asarray([.75, .25, 0., 1., 0.])
    arrays = {'candidate_offsets': offsets, 'candidate_kinds': kinds, 'legal_scores': scores,
        'behavior_probabilities': probabilities, 'logged_index': np.asarray([0, 1]),
        'logged_kind': np.asarray([KINDS['DISCARD'], KINDS['RON']]),
        'terminal_targets': np.asarray([1.5, -.5]), 'delta_targets': np.asarray([.1, -.2]),
        'opportunity_mask': np.asarray([opportunity_mask(kinds[:2]), opportunity_mask(kinds[2:])], dtype=np.uint8)}
    for label in ('raw', 'wrapped', 'deployed'):
        arrays[label + '_index'] = np.zeros((3, 2), dtype=np.int16)
        arrays[label + '_kind'] = np.zeros((3, 2), dtype=np.int8)
        arrays[label + '_probability'] = np.zeros((3, 2), dtype=np.float64)
    for model in range(3):
        for row, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            raw, wrapped = choices(scores[model, start:end], kinds[start:end])
            deployed = raw if model == 0 else wrapped
            for label, index in (('raw', raw), ('wrapped', wrapped), ('deployed', deployed)):
                arrays[label + '_index'][model, row] = index
                arrays[label + '_kind'][model, row] = kinds[start + index]
                arrays[label + '_probability'][model, row] = probabilities[start + index]
    return arrays


def small_plan():
    return {'reference_model_id': 'reference', 'initial_model_id': 'initial', 'models': [
        {'id': 'reference', 'kind': 'reference', 'arm': None, 'update': 0},
        {'id': 'initial', 'kind': 'value', 'arm': 'mc_a0', 'update': 0},
        {'id': 'mc_a0_u400', 'kind': 'value', 'arm': 'mc_a0', 'update': 400}]}


def test_metrics_use_chosen_labels_and_do_not_assign_utility_to_reference_logits():
    arrays = fixture_arrays()
    plan = small_plan()
    ref = row_metrics(arrays, 0, plan['models'][0], reference_index=0, initial_index=1, delta_available=True)
    assert 'mc_chosen_mse' not in ref and 'chosen_ce' not in ref
    assert 'logit_spread' in ref and 'value_spread' not in ref
    result = row_metrics(arrays, 2, plan['models'][2], reference_index=0, initial_index=1, delta_available=True)
    np.testing.assert_allclose(result['mc_chosen_mse'], [1.69, .64], atol=1e-7)
    np.testing.assert_allclose(result['mc_zero_prediction_mse'], [2.25, .25])
    np.testing.assert_allclose(result['deployed_probability'], [.25, 1.])
    np.testing.assert_array_equal(result['deployed_matches_reference'], [0, 0])
    np.testing.assert_array_equal(result['deployed_matches_logged'], [0, 1])
    np.testing.assert_allclose(result['declared_objective'], result['mc_chosen_mse'])
    no_delta = row_metrics(arrays, 2, plan['models'][2], reference_index=0, initial_index=1, delta_available=False)
    assert 'delta_chosen_mse' not in no_delta


def test_aggregation_distinguishes_row_and_supporting_match_weighting(tmp_path):
    arrays = fixture_arrays()
    reports = []
    for name, keep in [('two-rows', [0, 1]), ('one-row', [0])]:
        packed = copy.deepcopy(arrays)
        if keep == [0]:
            packed['candidate_offsets'] = np.asarray([0, 2])
            packed['candidate_kinds'] = packed['candidate_kinds'][:2]
            packed['behavior_probabilities'] = packed['behavior_probabilities'][:2]
            packed['legal_scores'] = packed['legal_scores'][:, :2]
            for key in list(packed):
                if key.endswith('_index') and key != 'logged_index' or key.endswith('_kind') and key != 'logged_kind' or key.endswith('_probability'):
                    packed[key] = packed[key][:, :1]
                elif key in ('logged_index', 'logged_kind', 'terminal_targets', 'delta_targets', 'opportunity_mask'):
                    packed[key] = packed[key][:1]
        path = tmp_path / f'{name}.npz'
        path.write_bytes(_pack_arrays(packed))
        reports.append({'split': 'development', 'rows': len(keep), 'delta_target_available': True,
            'numeric': {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}})
    summary = aggregate_matches(small_plan(), reports)
    all_ = summary['by_split_model_slice']['development']['mc_a0_u400']['all']
    assert all_['row_weighted']['deployed_matches_logged'] == pytest.approx(1 / 3)
    assert all_['match_weighted']['deployed_matches_logged'] == pytest.approx(1 / 4)
    win = summary['by_split_model_slice']['development']['mc_a0_u400']['win']
    assert win['supporting_matches'] == win['rows'] == 1
    absent = summary['by_split_model_slice']['development']['mc_a0_u400']['kan']
    assert absent['supporting_matches'] == 0 and absent['match_weighted'] == {}


def test_one_row_tensor_and_score_path_matches_production_collation(tmp_path):
    corpus(tmp_path / 'data', matches=1)
    ds = LearningDataset(tmp_path / 'data')
    actor, targets, _ = ds._load(ds.match_ids[0])
    path = tmp_path / 'reference.pt'
    model, _ = reference(path)
    model.eval()
    batch, arrays = one_row_actor(actor, targets, 3)
    production = collate_encoded([{key: value[0] for key, value in arrays.items() if key != 'candidate_mask'}], 'cpu')
    assert set(batch) == set(production)
    assert all(torch.equal(batch[k], production[k]) and batch[k].dtype == production[k].dtype for k in batch)
    np.testing.assert_array_equal(evaluate_one(model, batch), evaluate_one(model, production))
    with pytest.raises(ValueError, match='one CPU row'):
        evaluate_one(model, {key: value.expand(2, *value.shape[1:]) for key, value in batch.items()})


def value_checkpoint(tmp_path, update=0):
    ref = tmp_path / 'reference.pt'
    _, ref_sha = reference(ref)
    model = build_value_model(ref, ref_sha)
    weights = _state_hash(model.state_dict())
    metadata = {'schema': CHECKPOINT_SCHEMA, 'weights_sha256': weights,
        'frozen_rank_sha256': _state_hash(model.rank_head.state_dict()),
        'initial_weights_sha256': weights, 'scheduled_updates': update, 'optimizer_updates': update,
        'complete': update == 400,
        'identity': {'purpose': 'formal-training', 'split': 'training', 'arm': 'mc_a0',
            'recipe': RECIPE, 'manifest_sha256': 'b' * 64, 'reference_sha256': ref_sha, 'hand_targets_sha256': None}}
    payload = {'model_config': model.config, 'model_state_dict': model.state_dict(), 'metadata': metadata}
    path = tmp_path / 'value.pt'
    torch.save(payload, path)
    record = {'id': 'initial' if update == 0 else f'mc_a0_u{update}', 'kind': 'value', 'arm': 'mc_a0',
        'update': update, 'path': str(path), 'sha256': sha(path), 'reference_sha256': ref_sha}
    return record, payload


@pytest.mark.parametrize('update', [0, 100, 200, 300, 400])
def test_strict_loader_accepts_immutable_intermediate_and_final_boundaries(tmp_path, update):
    record, _ = value_checkpoint(tmp_path, update)
    model, metadata = load_checkpoint(record)
    assert metadata['scheduled_updates'] == update
    assert not model.training
    assert all(not p.requires_grad for p in model.rank_head.parameters())
    assert all(p.requires_grad for name, p in model.named_parameters() if not name.startswith('rank_head.'))


@pytest.mark.parametrize('change', ['sha', 'arm', 'update', 'reference', 'corpus', 'rank', 'complete', 'recipe'])
def test_shared_loader_rejects_relabelled_checkpoint_identity(tmp_path, change):
    record, payload = value_checkpoint(tmp_path)
    if change == 'sha': record['sha256'] = 'a' * 64
    elif change == 'arm': record['arm'] = 'mc_a1'
    elif change == 'update': record['update'] = 100
    elif change == 'reference': record['reference_sha256'] = 'a' * 64
    elif change == 'corpus': record['corpus_manifest_sha256'] = 'a' * 64
    else:
        if change == 'rank': payload['metadata']['frozen_rank_sha256'] = 'a' * 64
        elif change == 'complete': payload['metadata']['complete'] = True
        else: payload['metadata']['identity']['recipe'] = {}
        torch.save(payload, record['path'])
        record['sha256'] = sha(record['path'])
    with pytest.raises(ValueError): load_checkpoint(record)
