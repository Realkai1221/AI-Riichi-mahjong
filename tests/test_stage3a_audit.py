"""Independent audit fault injection: stdlib fixtures, no model/engine import."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import struct
import tempfile
import unittest
import zipfile

SPEC = importlib.util.spec_from_file_location('independent_stage3a_audit',
    Path(__file__).resolve().parents[1] / 'scripts/stage3a_audit.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def npy(descr, shape, payload, *, fortran=False):
    header = repr({'descr': descr, 'fortran_order': fortran, 'shape': shape}).encode() + b'\n'
    return b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header + payload


def actor_row(probabilities=(.25, .75), draw=.5):
    keys = [json.dumps(['semantic-actions-v1', 'DISCARD', i, False, False, []], separators=(',', ':'))
            for i in range(len(probabilities))]
    selected = next((i for i in range(len(probabilities)) if draw < sum(probabilities[:i + 1])), len(probabilities) - 1)
    p = probabilities[selected]
    return dict(semantic_keys=keys, representative_tokens=[f'token{i}' for i in range(len(keys))],
        selected_index=selected, selected_semantic_key=keys[selected], selected_physical_token=f'token{selected}',
        probabilities=list(probabilities), probability=p, log_probability=math.log(p),
        entropy=-sum(x * math.log(x) for x in probabilities if x), uniform_draw=None if len(keys) == 1 else draw,
        forced=len(keys) == 1, eligible=True, fallback=False, overridden=False)


def actor_archive(path, rows, *, extra=None, malformed_tokens=False):
    with zipfile.ZipFile(path, 'w') as archive:
        for index, row in enumerate(rows):
            n = len(row['semantic_keys'])
            tokens = [x for key in row['semantic_keys'] for x in audit.semantic_tokens(key)]
            if malformed_tokens:
                tokens[0] += 1
            fields = {
                'tiles': ('<f4', 'float32', (34, 23), bytes(34 * 23 * 4)),
                'globals': ('<f4', 'float32', (45,), bytes(45 * 4)),
                'history': ('<i2', 'int16', (128, 10), bytes(128 * 10 * 2)),
                'history_mask': ('|b1', 'bool', (128,), bytes(128)),
                'action_tokens': ('<i2', 'int16', (n, 8), struct.pack('<' + 'h' * len(tokens), *tokens)),
                'candidate_efficiency': ('<f4', 'float32', (n, 4), bytes(n * 4 * 4))}
            result = hashlib.sha256()
            for name, (descr, dtype, shape, payload) in sorted(fields.items()):
                archive.writestr(f'row_{index:05d}__{name}.npy', npy(descr, shape, payload))
                result.update(audit.canonical([name, dtype, list(shape)]).encode())
                result.update(payload)
            row['encoded_sha256'] = result.hexdigest()
        if extra:
            archive.writestr(extra, b'not an actor feature')


def state(updates=0, *, weights='a' * 64):
    return dict(schema=audit.SCHEMA, protocol_sha256='0' * 64, reference_sha256=audit.REFERENCE,
        updates=updates, optimizer_steps_executed=updates, retired_batches=list(range(updates)), device='cpu',
        weights_sha256=weights, optimizer_sha256=('b' if updates == 0 else 'd') * 64,
        rank_head_sha256='c' * 64, policy_id=f'{audit.SCHEMA}:update-{updates}:{weights}',
        model_config={'feature_schema': 'actor-features-v2'},
        optimizer_config=dict(name='AdamW', learning_rate=1e-5, weight_decay=0., betas=[.9, .999],
            eps=1e-8, foreach=False, fresh_optimizer=True), sampler='python-random-float64-cdf-softmax-v1')


def update_fixture():
    before = state()
    matches = [dict(completed=True, optional_rows=i + 1, forced_rows=2,
        learner_utility=[1.5, .5, -.5, -1.5][i % 4], stored_log_probability_sum=-(i + 1) * .3,
        optional_entropy_sum=(i + 1) * .5) for i in range(16)]
    plans = [{'match_index': i} for i in range(16)]
    loss = -sum(m['learner_utility'] * m['stored_log_probability_sum'] for m in matches) / 16
    optional = sum(m['optional_rows'] for m in matches)
    update = dict(schema=audit.SCHEMA, batch_index=0, planned_matches=plans, completed_matches=16,
        loss_denominator_matches=16, optimizer_steps=1, baseline=0., discount=1., opponent_rows_in_loss=0,
        optional_learner_rows=optional, forced_learner_rows=32,
        match_utilities=[m['learner_utility'] for m in matches],
        stored_behavior_match_log_probability_sums=[m['stored_log_probability_sum'] for m in matches],
        stored_behavior_loss=loss, match_log_probability_sums=[m['stored_log_probability_sum'] for m in matches],
        selected_log_probability_replay_max_error=0., selected_log_probability_replay_atol=1e-4,
        behavior_reconstructed_loss=loss, differentiable_loss=loss, loss_numerical_difference=0.,
        probability_replay_atol=1e-6, probability_replay_rtol=1e-5, probability_replay_max_absolute_error=0.,
        before=before, after=state(1, weights='e' * 64), gradient_clip=1., gradients_finite=True,
        parameters_finite=True, rank_head_unchanged=True, gradient_norm_before_clip=.1,
        parameter_max_absolute_change=1e-5, parameter_l2_change=1e-4,
        distribution_max_absolute_change=1e-5, distribution_mean_l1_change=2e-5,
        mean_optional_entropy_before=.5, mean_optional_entropy_after=.51,
        distribution_diagnostic_rows=optional, distribution_diagnostic_complete=True, numerical_no_op=False)
    config = dict(matches_per_batch=16, replay_atol=1e-6, replay_rtol=1e-5, replay_log_atol=1e-4, gradient_clip=1.)
    return update, matches, before, plans, config


class ActorContractTests(unittest.TestCase):
    def test_safe_npy_byte_parsing_and_invalid_types(self):
        self.assertEqual(audit.parse_npy(npy('<f4', (2,), struct.pack('<ff', 1., 2.)))[:2], ('float32', (2,)))
        for value in (npy('|O', (1,), bytes(8)), npy('<f4', (1,), bytes(4), fortran=True),
                      npy('<f4', (2,), bytes(4)), b'not-npy'):
            with self.subTest(header=value[:12]), self.assertRaises((ValueError, SyntaxError)):
                audit.parse_npy(value)

    def test_encoded_hash_contract_and_hidden_field_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'a.npz'
            rows = [actor_row()]
            actor_archive(path, rows)
            audit.audit_encoded(path, rows, audit.Checks())
            rows[0]['encoded_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'byte hash'):
                audit.audit_encoded(path, rows, audit.Checks())
            actor_archive(path, rows, extra='row_00000__opponent_tiles.npy')
            with self.assertRaisesRegex(ValueError, 'unexpected fields'):
                audit.audit_encoded(path, rows, audit.Checks())

    def test_candidate_tensor_must_match_semantic_action(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'a.npz'
            rows = [actor_row()]
            actor_archive(path, rows, malformed_tokens=True)
            with self.assertRaisesRegex(ValueError, 'candidates and action tensors'):
                audit.audit_encoded(path, rows, audit.Checks())

    def test_sampler_forced_and_zero_bin_boundaries(self):
        for row in (actor_row(), actor_row((1.,), 0.), actor_row((0., .2, .8), 0.),
                    actor_row((.25, .75), .25)):
            audit.audit_row(row, audit.Checks())
        row = actor_row((1.,), 0.)
        row['uniform_draw'] = 0.
        with self.assertRaisesRegex(ValueError, 'Forced decision'):
            audit.audit_row(row, audit.Checks())

    def test_behavior_selection_and_logp_tampering(self):
        for key, value in [('log_probability', -9.), ('uniform_draw', .1), ('eligible', False), ('overridden', True)]:
            row = actor_row()
            row[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit.audit_row(row, audit.Checks())

    def test_semantic_bool_cannot_be_replaced_by_integer(self):
        with self.assertRaisesRegex(ValueError, 'Malformed semantic'):
            audit.semantic_tokens(json.dumps(['semantic-actions-v1', 'DISCARD', 1, False, 1, []]))


class TerminalTests(unittest.TestCase):
    def test_bare_terminal_reconstructs_repeated_dealer_hands_and_deposits(self):
        log = [dict(type='start_kyoku', scores=[25000] * 4, kyotaku=0),
            dict(type='reach_accepted', actor=1), dict(type='hora', deltas=[-8000, 9000, 0, 0]),
            dict(type='end_kyoku'), dict(type='start_kyoku', scores=[17000, 33000, 25000, 25000], kyotaku=0),
            dict(type='reach_accepted', actor=2), dict(type='ryukyoku', deltas=[0] * 4),
            dict(type='end_kyoku'), dict(type='end_game')]
        scores, evidence = audit.audit_score_chain(log, audit.Checks())
        self.assertEqual(scores, [17000, 34000, 24000, 25000])
        self.assertEqual(evidence['hands'], 2)
        self.assertEqual(evidence['terminal_deposit_award'], 1000)
        self.assertEqual(evidence['terminal_deposit_recipient'], 1)
        bad = deepcopy(log)
        bad[4]['scores'] = [18000, 32000, 25000, 25000]
        with self.assertRaisesRegex(ValueError, 'Successive hand'):
            audit.audit_score_chain(bad, audit.Checks())

    def test_exact_exhaustive_draw_log_quirk_and_no_generic_correction(self):
        log = [dict(type='start_kyoku', scores=[25000] * 4, kyotaku=0),
            dict(type='reach_accepted', actor=0),
            dict(type='ryukyoku', reason='exhaustive_draw', deltas=[-1000, 0, 0, 0]), dict(type='end_game')]
        scores, evidence = audit.audit_score_chain(log, audit.Checks())
        self.assertEqual(scores, [24000, 26000, 25000, 25000])
        self.assertEqual(len(evidence['upstream_log_delta_corrections']), 1)
        for reason, deltas in [('exhaustive_draw', [0, -1000, 0, 0]), ('abortive', [-1000, 0, 0, 0])]:
            bad = deepcopy(log)
            bad[2].update(reason=reason, deltas=deltas)
            with self.subTest(reason=reason), self.assertRaises(ValueError):
                audit.audit_score_chain(bad, audit.Checks())

    def test_nagashi_correction_preserves_separate_riichi_payment(self):
        log = [dict(type='start_kyoku', scores=[25000] * 4, kyotaku=0),
            dict(type='reach_accepted', actor=0),
            dict(type='ryukyoku', reason='nagashimangan', deltas=[5000, -2000, -2000, -2000]),
            dict(type='end_game')]
        scores, evidence = audit.audit_score_chain(log, audit.Checks())
        self.assertEqual(scores, [31000, 23000, 23000, 23000])
        self.assertEqual(evidence['upstream_log_delta_corrections'][0]['effective_deltas'], [6000, -2000, -2000, -2000])

    def test_terminal_marker_and_total_are_required(self):
        logs = [[dict(type='end_game')],
            [dict(type='start_kyoku', scores=[24000] * 4, kyotaku=0), dict(type='end_game')],
            [dict(type='start_kyoku', scores=[25000] * 4, kyotaku=0), dict(type='end_game'), dict(type='end_game')]]
        for log in logs:
            with self.subTest(log=log), self.assertRaises(ValueError):
                audit.audit_score_chain(log, audit.Checks())


class ObjectiveTests(unittest.TestCase):
    def test_exact_full_match_objective(self):
        fixture = update_fixture()
        result = audit.audit_update(*fixture, audit.Checks())
        self.assertEqual(result['optimizer_steps'], 1)
        self.assertFalse(result['numerical_no_op'])

    def test_partial_batches_or_decision_normalization_rejected(self):
        fixture = update_fixture()
        fixture[0]['loss_denominator_matches'] = fixture[0]['optional_learner_rows']
        with self.assertRaisesRegex(ValueError, 'average exactly'):
            audit.audit_update(*fixture, audit.Checks())
        fixture = update_fixture()
        fixture[1][-1]['completed'] = False
        with self.assertRaisesRegex(ValueError, 'Partial planned'):
            audit.audit_update(*fixture, audit.Checks())

    def test_stored_behavior_loss_and_opponent_actions_are_enforced(self):
        for key, value in [('stored_behavior_loss', 18.), ('differentiable_loss', 18.), ('opponent_rows_in_loss', 1),
                ('selected_log_probability_replay_max_error', 1e-3), ('rank_head_unchanged', False)]:
            fixture = update_fixture()
            fixture[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit.audit_update(*fixture, audit.Checks())

    def test_optimizer_step_and_rank_head_identity(self):
        for key, value in [('optimizer_steps_executed', 0), ('retired_batches', []), ('rank_head_sha256', 'f' * 64)]:
            fixture = update_fixture()
            fixture[0]['after'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit.audit_update(*fixture, audit.Checks())

    def test_incomplete_post_step_diagnostic_remains_update_but_not_complete(self):
        fixture = update_fixture()
        fixture[0].update(distribution_diagnostic_complete=False, distribution_diagnostic_rows=1)
        result = audit.audit_update(*fixture, audit.Checks())
        self.assertEqual(result['optimizer_steps'], 1)
        self.assertFalse(result['distribution_diagnostic_complete'])


class MatchProvenanceTests(unittest.TestCase):
    def fixture(self, folder):
        root = Path(folder)
        planned = dict(match_index=0, seed=17, learner_seat=0, sampling_seed=83)
        before = state()
        opponent = {'checkpoint_sha256': audit.REFERENCE, 'weights_sha256': before['weights_sha256']}
        rng = random.Random(83)
        start = rng.getstate()
        row = actor_row(draw=rng.random())
        row.update(row_index=0, match_index=0, batch_index=0, seat=0, hand_id=1,
            policy_id=before['policy_id'], step=0, player_view_sha256='1' * 64)
        views = {'0': row['player_view_sha256']}
        row['pre_response_id'] = audit.object_hash(dict(match_index=0, step=0, views=views))
        learner = dict(role='learner', policy_id=before['policy_id'], weights_sha256=before['weights_sha256'],
            update=0, reference_sha256=audit.REFERENCE, sampling_seed=83, feature_schema='actor-features-v2')
        frozen = {**opponent, 'role': 'frozen_opponent', 'policy_id': 'frozen-argmax:' + audit.REFERENCE}
        choices = {'0': dict(eligible=True, semantic_key=row['selected_semantic_key'],
            probability=row['probability'], physical_token=row['selected_physical_token'],
            policy_id=before['policy_id'], fallback=False, overridden=False, row_index=0)}
        next_views = {'1': '2' * 64}
        requests = [dict(step=0, pre_response_id=row['pre_response_id'], player_views=views, choices=choices),
            dict(step=1, pre_response_id=audit.object_hash(dict(match_index=0, step=1, views=next_views)),
                player_views=next_views, choices={'1': dict(eligible=False, probability=1.,
                    physical_token='pass', semantic_key='pass', policy_id=frozen['policy_id'],
                    fallback=False, overridden=False)})]
        log = [dict(type='start_kyoku', scores=[25000] * 4, kyotaku=0), dict(type='end_game')]
        _, reconstruction = audit.audit_score_chain(log, audit.Checks())
        array_path = root / 'match.npz'
        actor_archive(array_path, [row])
        record = dict(**planned, schema=audit.SCHEMA, protocol_sha256='0' * 64, batch_index=0,
            policy_id=before['policy_id'], sampler=before['sampler'],
            rule_profile_sha256=audit.RULE_PROFILE, view_hash_schema='riichi-ai-player-view-v2', max_steps=100000,
            encoded_artifact=dict(path=str(array_path), sha256=audit.digest(array_path), rows=1),
            policies=[learner, frozen, frozen, frozen], auditor_log=log, auditor_log_sha256=audit.object_hash(log),
            hands=1, completed=True, terminated=True, truncated=False, failure=None, stop_reason='completed',
            terminal_result_verified=True, conservation_verified=True, scores=[25000] * 4,
            terminal_event=log[-1], reconstructed_scores=[25000] * 4, terminal_score_reconstruction=reconstruction,
            ranks=[1, 2, 3, 4], reconstructed_ranks=[1, 2, 3, 4], utilities=[1.5, .5, -.5, -1.5],
            learner_utility=1.5, riichi_sticks=0, learner_rows=[row], requests=requests,
            sampling_rng_start=start, sampling_rng_end=rng.getstate(), steps=2,
            learner_last_step=0, opponent_steps_after_last_learner=1)
        return record, planned, before, opponent, array_path

    def run_record(self, fixture):
        record, planned, before, opponent, array_path = fixture
        path = array_path.with_suffix('.json')
        path.write_text(json.dumps(record))
        evidence = dict(json_path=str(path), json_sha256=audit.digest(path),
            npz_path=str(array_path), npz_sha256=audit.digest(array_path))
        return audit.audit_match(evidence, planned, 0, before, opponent, '0' * 64,
            array_path.parent, audit.Checks())

    def test_opponent_ended_match_keeps_earlier_learner_terminal_reward(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = self.fixture(folder)
            result = self.run_record(fixture)
            self.assertEqual(result['learner_utility'], 1.5)
            self.assertEqual(result['opponent_steps_after_last_learner'], 1)
            fixture[0]['learner_utility'] = 0
            with self.assertRaisesRegex(ValueError, 'rank utility'):
                self.run_record(fixture)

    def test_opponent_eligibility_and_mixed_pre_response_state_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = self.fixture(folder)
            fixture[0]['requests'][1]['choices']['1']['eligible'] = True
            with self.assertRaisesRegex(ValueError, 'Frozen-opponent action'):
                self.run_record(fixture)
            fixture[0]['requests'][1]['choices']['1']['eligible'] = False
            fixture[0]['requests'][0]['player_views']['0'] = '3' * 64
            with self.assertRaisesRegex(ValueError, 'Pre-response snapshot'):
                self.run_record(fixture)


class BudgetTests(unittest.TestCase):
    def fixture(self, folder):
        data, reports = Path(folder) / 'data', Path(folder) / 'reports'
        data.mkdir(); reports.mkdir()
        operations = []
        for i, (name, status, failure) in enumerate([
            ('check:initial', 'failed', {'type': 'RuntimeError', 'message': 'fixture failed'}),
            ('prepare', 'completed', None)]):
            operations.append(dict(id=i, name=name, status=status, reserved_seconds=30., elapsed_seconds=2.,
                started_unix=100 + i * 10, token=str(i), failure=failure, exceeded_reservation=False))
            budget = dict(schema='package-a2-budget-v1', identity='0' * 64, limit_seconds=600,
                operations=deepcopy(operations), charged_seconds=2. * (i + 1), remaining_seconds=600 - 2. * (i + 1))
            marker = dict(name=name, config_sha256='0' * 64, status=status, failure=failure, budget=budget,
                watchdog=dict(elapsed_seconds=1., sampled_peak_tree_rss_bytes=1, artifact_bytes_peak=1,
                    exit_code=0 if status == 'completed' else 1))
            (reports / ('prepare-attempt.json' if name == 'prepare' else 'checks-initial.json')).write_text(json.dumps(marker))
        saved = budget
        raw = {k: v for k, v in saved.items() if k not in ('charged_seconds', 'remaining_seconds')}
        (data / 'budget-ledger.json').write_text(json.dumps(raw))
        (reports / 'budget.json').write_text(json.dumps(saved))
        config = dict(test_allowance_seconds=90, prepare_allowance_seconds=30, per_batch_allowance_seconds=100,
            reserve_seconds=45, memory_bytes=32 * 1024**3, artifact_bytes=1_000_000_000)
        return config, reports, data

    def test_failed_attempt_is_charged_and_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            config, reports, data = self.fixture(folder)
            result, _ = audit.audit_budget(config, '0' * 64, reports, data, audit.Checks())
            self.assertEqual(result['charged_seconds'], 4.)
            self.assertEqual(result['failed_operations'], 1)

    def test_saved_ledger_must_match_live_and_no_uncharged_markers(self):
        with tempfile.TemporaryDirectory() as folder:
            config, reports, data = self.fixture(folder)
            (reports / 'checks-uncharged.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Uncharged'):
                audit.audit_budget(config, '0' * 64, reports, data, audit.Checks())
            (reports / 'checks-uncharged.json').unlink()
            saved = json.loads((reports / 'budget.json').read_text())
            saved['operations'][0]['elapsed_seconds'] = 0
            (reports / 'budget.json').write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, 'Live/saved'):
                audit.audit_budget(config, '0' * 64, reports, data, audit.Checks())


if __name__ == '__main__':
    unittest.main()
