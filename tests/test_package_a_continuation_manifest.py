"""Model-free fault injection at the final inventory packaging boundary."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location('continuation_manifest', SCRIPTS / 'package_a_continuation_manifest.py')
manifest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manifest)
sys.path.pop(0)


class PackagingBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.root, self.external = base / 'project', base / 'external'
        self.reports = self.root / 'reports/package_a_continuation'
        self.reports.mkdir(parents=True)
        self.external.mkdir()
        (self.root / 'configs').mkdir()
        (self.root / '.gitignore').write_text('Books/\n')
        self.config_path = self.root / 'configs/package_a_continuation.json'
        self.config = {'data_root': str(self.external), 'reports_root': 'reports/package_a_continuation',
            'limit_seconds': 900, 'maximum_attempts': 4,
            'attempts': [{'name': name} for name in ('same-1', 'lower-1', 'lower-2', 'same-2')],
            'diagnosis_snapshot_commit': 'prior-commit', 'diagnosis_manifest_sha256': 'prior-hash'}
        self.write(self.config_path, self.config)
        self.identity = manifest.digest(self.config_path)
        self.plan = {'config_sha256': self.identity}
        self.write(self.reports / 'plan.json', self.plan)
        self.ledger = {'schema': 'package-a2-budget-v1', 'identity': self.identity,
            'limit_seconds': 900, 'operations': [self.operation(i, 'continuation:' + a['name'])
                for i, a in enumerate(self.config['attempts'])]}
        self.audit = {'schema': 'package-a-continuation-audit-v1', 'valid': True,
            'config_sha256': self.identity, 'plan_sha256': manifest.digest(self.reports / 'plan.json'),
            'ledger_report_copy_verified': True, 'budget': {}}
        self.sync_ledger_and_audit()
        self.write(self.reports / 'prior-diagnosis-integrity.json', {'valid': True,
            'snapshot_commit': 'prior-commit', 'manifest_sha256': 'prior-hash'})
        self.inventory = self.reports / 'manifest.json'
        for key, value in [('ROOT', self.root), ('REPORTS', self.reports), ('MANIFEST', self.inventory)]:
            p = patch.object(manifest, key, value)
            p.start()
            self.addCleanup(p.stop)

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value, indent=2) + '\n')

    @staticmethod
    def operation(index, name, status='completed'):
        return {'id': index, 'name': name, 'status': status, 'elapsed_seconds': 2.0,
                'reserved_seconds': 20.0, 'exceeded_reservation': False}

    def sync_ledger_and_audit(self):
        for path in (self.external / 'budget-ledger.json', self.reports / 'budget-ledger.json'):
            self.write(path, self.ledger)
        ops = self.ledger['operations']
        charged = sum(op['elapsed_seconds'] for op in ops)
        self.audit['budget'] = {'charged_seconds': charged, 'limit_seconds': 900,
            'remaining_seconds': 900 - charged, 'operation_count': len(ops), 'active_leases': 0,
            'formal_attempts': sum(op['name'].startswith('continuation:') for op in ops)}
        self.write(self.reports / 'audit.json', self.audit)

    def gate(self):
        return manifest.packaging_state(self.config_path, self.config, self.external)

    def test_inventory_is_write_once_and_preserves_prior_record(self):
        historical = self.reports / 'prior-diagnosis-integrity.json'
        before = historical.read_bytes()
        with patch.object(manifest.subprocess, 'check_output', return_value='working-head\n'):
            result = manifest.create()
        self.assertTrue(result['created'])
        self.assertTrue(manifest.verify()['valid'])
        self.assertEqual(historical.read_bytes(), before)
        with self.assertRaises(FileExistsError):
            manifest.create()

    def test_changed_config_rejects_saved_audit(self):
        self.config['limit_seconds'] = 901
        self.write(self.config_path, self.config)
        with self.assertRaisesRegex(ValueError, 'live config'):
            self.gate()

    def test_changed_plan_rejects_saved_audit(self):
        self.plan['changed_after_audit'] = True
        self.write(self.reports / 'plan.json', self.plan)
        with self.assertRaisesRegex(ValueError, 'frozen plan'):
            self.gate()

    def test_configured_root_identity_and_missing_roots_fail(self):
        with self.assertRaisesRegex(ValueError, 'root identity'):
            manifest.packaging_state(self.config_path, self.config, self.root)
        (self.external / 'budget-ledger.json').unlink()
        self.external.rmdir()
        with self.assertRaisesRegex(ValueError, 'roots must exist'):
            self.gate()

    def test_ledger_copy_must_match_bytes(self):
        with (self.reports / 'budget-ledger.json').open('a') as stream:
            stream.write(' ')
        with self.assertRaisesRegex(ValueError, 'ledger bytes'):
            self.gate()

    def test_unverified_copy_is_rejected_even_when_bytes_match(self):
        self.audit['ledger_report_copy_verified'] = False
        self.write(self.reports / 'audit.json', self.audit)
        with self.assertRaisesRegex(ValueError, 'Verified report ledger'):
            self.gate()

    def test_active_lease_is_rejected(self):
        self.ledger['operations'][0]['status'] = 'active'
        self.sync_ledger_and_audit()
        with self.assertRaisesRegex(ValueError, 'active, invalid'):
            self.gate()

    def test_new_closed_operation_requires_fresh_audit(self):
        self.ledger['operations'].append(self.operation(4, 'later-check'))
        for path in (self.external / 'budget-ledger.json', self.reports / 'budget-ledger.json'):
            self.write(path, self.ledger)
        with self.assertRaisesRegex(ValueError, 'audit budget is stale'):
            self.gate()

    def test_each_audit_budget_component_is_bound(self):
        original = deepcopy(self.audit)
        for field in original['budget']:
            with self.subTest(field=field):
                self.audit = deepcopy(original)
                self.audit['budget'][field] += 1
                self.write(self.reports / 'audit.json', self.audit)
                with self.assertRaisesRegex(ValueError, 'audit budget is stale'):
                    self.gate()

    def test_fifth_formal_attempt_rejected_even_with_matching_totals(self):
        self.ledger['operations'].append(self.operation(4, 'continuation:replacement'))
        self.sync_ledger_and_audit()
        with self.assertRaisesRegex(ValueError, 'bounded sequence'):
            self.gate()

    def test_invalid_elapsed_times_fail_before_packaging(self):
        for elapsed in (float('nan'), -1, 21, True):
            with self.subTest(elapsed=elapsed):
                self.ledger['operations'][0]['elapsed_seconds'] = elapsed
                self.sync_ledger_and_audit()
                with self.assertRaisesRegex(ValueError, 'active, invalid'):
                    self.gate()

    def test_closed_failed_support_operation_is_retained_and_charged(self):
        self.ledger['operations'].append(self.operation(4, 'failed-compile', status='failed'))
        self.sync_ledger_and_audit()
        result = self.gate()
        self.assertEqual(result['budget']['operation_count'], 5)
        self.assertEqual(result['budget']['charged_seconds'], 10.0)


if __name__ == '__main__':
    unittest.main()
