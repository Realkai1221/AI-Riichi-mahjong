"""Model-free tests for continuation acceptance and safety reconstruction."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("continuation_audit", SCRIPTS / "package_a_continuation_audit.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
sys.path.pop(0)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.baseline = {cohort: {"smooth_kl_f64": .2, "discard_agreement": .98,
            "shanten_worsening_rate": .01} for cohort in
            ("training_teacher", "validation_teacher", "validation_student")}
        self.same = deepcopy(self.baseline)
        self.lower = deepcopy(self.baseline)
        self.lower["training_teacher"]["smooth_kl_f64"] = .1
        self.lower["validation_teacher"]["smooth_kl_f64"] = .1

    def test_every_prespecified_component_is_required(self):
        result = audit.compare_pair(self.baseline, self.same, self.lower, eligible=True)
        self.assertTrue(result["exploratory_passed"])
        self.assertEqual(len(result["components"]), 10)
        self.lower["validation_student"]["discard_agreement"] = .97
        self.assertFalse(audit.compare_pair(self.baseline, self.same, self.lower, eligible=True)["exploratory_passed"])

    def test_no_strict_kl_improvement_is_not_a_pass(self):
        self.lower["training_teacher"]["smooth_kl_f64"] = .2
        self.assertFalse(audit.compare_pair(self.baseline, self.same, self.lower, eligible=True)["exploratory_passed"])

    def test_prior_endpoint_guard_is_distinct_from_paired_control(self):
        self.same["validation_student"]["shanten_worsening_rate"] = .04
        self.lower["validation_student"]["shanten_worsening_rate"] = .02
        result = audit.compare_pair(self.baseline, self.same, self.lower, eligible=True)
        self.assertTrue(result["components"]["validation_student:shanten_rate_at_most_same"])
        self.assertFalse(result["components"]["validation_student:shanten_rate_at_most_baseline"])
        self.assertFalse(result["exploratory_passed"])

    def test_training_quality_is_descriptive_not_an_added_gate(self):
        self.lower["training_teacher"]["discard_agreement"] = .1
        self.assertTrue(audit.compare_pair(self.baseline, self.same, self.lower, eligible=True)["exploratory_passed"])

    def test_failed_or_missing_pair_cannot_pass(self):
        self.assertFalse(audit.compare_pair(self.baseline, self.same, self.lower, eligible=False)["exploratory_passed"])
        self.assertFalse(audit.compare_pair(self.baseline, None, self.lower, eligible=True)["exploratory_passed"])


class SafetyReplayTests(unittest.TestCase):
    def setUp(self):
        self.config = {"warmup_steps": 2, "scale_factor": 10, "scale_floor": 1,
            "raw_logit_absolute_floor": 100, "persistence": 3, "gradient_threshold": 50,
            "gradient_persistence": 3, "policy_ce_threshold": 10, "policy_ce_persistence": 50}

    @staticmethod
    def diagnostic(step, value):
        return {"step": step, **{population: {metric: {"mean": value, "maximum": value}
            for metric in ("context_rms", "legal_logit_abs", "centered_legal_logit_abs")}
            for population in ("sampled", "fixed")}}

    def test_split_resume_preserves_pending_streak_and_warmup(self):
        history = [{"step": n, "policy_ce": 1, "gradient_norm": 0} for n in range(1, 6)]
        diagnostics = [self.diagnostic(n, 2 if n <= 2 else 101) for n in range(1, 6)]
        start = audit.replay_safety(self.config, history[:4], diagnostics[:4])
        self.assertFalse(start["tripped"])
        self.assertEqual(start["streaks"]["scale:fixed:context_rms:mean"], 2)
        continued = audit.replay_safety(self.config, history[4:], diagnostics[4:], start)
        whole = audit.replay_safety(self.config, history, diagnostics)
        self.assertEqual(continued, whole)
        self.assertTrue(continued["tripped"])
        self.assertTrue(continued["scale_alarm"])
        self.assertEqual(continued["references"]["scale:fixed:context_rms:mean"], 2)
        self.assertFalse(start["tripped"], "Replay must not mutate input state")

    def test_nonfinite_and_missing_reference_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            audit.replay_safety(self.config, [{"step": 1, "policy_ce": float("nan"), "gradient_norm": 1}], [])
        with self.assertRaisesRegex(ValueError, "Missing historical"):
            audit.replay_safety(self.config, [{"step": 3, "policy_ce": 1, "gradient_norm": 1}], [self.diagnostic(3, 1)])

    def test_loss_streak_survives_split_and_resets_on_recovery(self):
        history = [{"step": n, "policy_ce": 1, "gradient_norm": 51 if n < 3 else 0} for n in range(1, 4)]
        state = audit.replay_safety(self.config, history[:2], [])
        self.assertEqual(state["streaks"]["gradient_norm"], 2)
        final = audit.replay_safety(self.config, history[2:], [], state)
        self.assertEqual(final["streaks"]["gradient_norm"], 0)
        self.assertFalse(final["tripped"])

    def test_duplicate_diagnostic_step_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            audit.replay_safety(self.config, [], [self.diagnostic(1, 1), self.diagnostic(1, 1)])

    def test_plain_state_digest_is_order_independent_but_type_sensitive(self):
        self.assertEqual(audit.plain_state_digest({"a": 1, "b": [2]}), audit.plain_state_digest({"b": [2], "a": 1}))
        self.assertNotEqual(audit.plain_state_digest({"a": 1}), audit.plain_state_digest({"a": 1.0}))


if __name__ == "__main__":
    unittest.main()
