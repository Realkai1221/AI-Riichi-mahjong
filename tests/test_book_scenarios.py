"""Stdlib-only intake checks; no simulator, model, or training is imported."""

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from riichi_ai.book_scenarios import exercise_inputs, load_jsonl, main, validate_record, validate_records


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/book_scenarios/example.jsonl"


class BookScenarioTests(unittest.TestCase):
    def setUp(self):
        self.record = load_jsonl(EXAMPLE)[0]

    def assert_invalid(self, fragment):
        report = validate_records([self.record])
        self.assertFalse(report["valid"], report)
        self.assertIn(fragment, " ".join(report["errors"]))

    def second_record(self, role="heldout"):
        second = copy.deepcopy(self.record)
        second["id"] = "second"
        second["split"]["role"] = role
        second["permissions"]["local_training"] = True
        return second

    def test_authored_example_is_valid_but_not_a_policy_training_row(self):
        report = validate_records([self.record])
        self.assertTrue(report["valid"], report)
        self.assertFalse(report["policy_training_ready"])
        self.assertEqual(report["state_kinds"], {"partial_exercise": 1})
        self.assertEqual(report["roles"], {"quarantine": 1})

    def test_empty_collection_is_not_a_valid_intake(self):
        self.assertFalse(validate_records([])["valid"])

    def test_actor_inputs_exclude_answer_explanation_and_source(self):
        actor = exercise_inputs(self.record)
        self.assertEqual(set(actor), {"schema", "concealed_hand", "drawn_tile"})
        baseline = copy.deepcopy(actor)
        self.record["source_answer"]["rationale"] = "Change the explanatory answer text"
        self.record["source_answer"]["acceptable_actions"] = self.record["source_answer"]["acceptable_actions"][:1]
        self.record["source"]["title"] = "A different title"
        self.assertEqual(exercise_inputs(self.record), baseline)
        actor["concealed_hand"].clear()
        self.assertEqual(len(self.record["state"]["observed"]["concealed_hand"]), 14)

    def test_actor_field_injection_is_rejected(self):
        self.record["state"]["observed"]["answer"] = "discard C"
        self.assert_invalid("state.observed")

    def test_missing_context_cannot_be_erased_from_partial_state(self):
        self.record["state"]["missing_context"] = []
        self.assert_invalid("missing_context")

    def test_red_and_normal_tiles_share_four_copy_limit(self):
        self.record["state"]["observed"]["concealed_hand"] = ["5pr", "5p", "5p", "5p", "5p", "C"]
        self.assert_invalid("more than four copies")

    def test_unknown_tiles_are_not_filled_in(self):
        self.record["state"]["observed"]["concealed_hand"][0] = "?"
        self.assert_invalid("invalid tile")

    def test_red_nonfive_is_rejected(self):
        self.record["state"]["observed"]["concealed_hand"][0] = "1mr"
        self.assert_invalid("invalid tile")

    def test_drawn_tile_must_be_present(self):
        self.record["state"]["observed"]["drawn_tile"] = "9m"
        self.assert_invalid("drawn_tile")

    def test_answer_tile_must_be_present(self):
        self.record["source_answer"]["acceptable_actions"][0]["tile"] = "9m"
        self.assert_invalid("absent")

    def test_drawn_and_held_discard_distinction_is_preserved(self):
        self.record["state"]["observed"]["drawn_tile"] = "5p"
        self.record["source_answer"]["acceptable_actions"][0]["discard_origin"] = "held"
        self.assert_invalid("no held copy")
        self.record["source_answer"]["acceptable_actions"][0]["discard_origin"] = "drawn"
        validate_record(self.record)
        self.record["source_answer"]["acceptable_actions"][1]["discard_origin"] = "drawn"
        self.assert_invalid("differs")

    def test_duplicate_actions_are_rejected(self):
        self.record["source_answer"]["acceptable_actions"].append(copy.deepcopy(self.record["source_answer"]["acceptable_actions"][0]))
        self.assert_invalid("duplicate acceptable action")

    def test_calls_require_a_future_schema(self):
        self.record["source_answer"]["acceptable_actions"][0]["kind"] = "PON"
        self.assert_invalid("discard exercises only")

    def test_training_role_requires_recorded_training_permission(self):
        self.record["split"]["role"] = "train"
        self.assert_invalid("local-training permission")

    def test_holdout_role_requires_recorded_evaluation_permission(self):
        self.record["split"]["role"] = "heldout"
        self.record["permissions"]["local_evaluation"] = False
        self.assert_invalid("local-evaluation permission")

    def test_permission_boolean_cannot_be_integer(self):
        self.record["permissions"]["local_training"] = 1
        self.assert_invalid("expected a boolean")

    def test_nonpending_label_needs_independent_evidence_record(self):
        self.record["label_review"]["status"] = "supported"
        self.assert_invalid("reviewer")

    def test_supported_label_preserves_original_answer_set(self):
        self.record["label_review"] = {"status": "supported", "reviewer": "fixture reviewer", "method": "fixture-only manual review", "evidence": "not a real validation result", "accepted_actions": copy.deepcopy(self.record["source_answer"]["acceptable_actions"])}
        validate_record(self.record)
        self.record["label_review"]["accepted_actions"].pop()
        self.assert_invalid("preserve the source")

    def test_disputed_review_preserves_both_answers_separately(self):
        original = copy.deepcopy(self.record["source_answer"])
        self.record["label_review"] = {"status": "disputed", "reviewer": "fixture reviewer", "method": "fixture-only manual review", "evidence": "not a real validation result", "accepted_actions": [copy.deepcopy(original["acceptable_actions"][0])]}
        validate_record(self.record)
        self.assertEqual(self.record["source_answer"], original)

    def test_whole_book_and_family_cannot_cross_splits(self):
        self.record["split"]["role"] = "validation"
        report = validate_records([self.record, self.second_record()])
        self.assertFalse(report["valid"])
        for group in ("book_group", "chapter_group", "problem_family", "exact_content"):
            self.assertIn(group, " ".join(report["errors"]))

    def test_reordered_identical_hand_cannot_bypass_exact_duplicate_check(self):
        self.record["split"]["role"] = "validation"
        second = self.second_record()
        second["source"]["book_id"] = "other-book"
        for key in ("book_group", "chapter_group", "problem_family"):
            second["split"][key] = "other-" + second["split"][key]
        second["state"]["observed"]["concealed_hand"].reverse()
        self.assertIn("exact_content", " ".join(validate_records([self.record, second])["errors"]))

    def test_book_identity_cannot_be_assigned_new_group_to_evade_split(self):
        second = self.second_record()
        second["split"]["book_group"] = "changed"
        self.assertIn("multiple book groups", " ".join(validate_records([self.record, second])["errors"]))

    def test_reordered_assumptions_cannot_bypass_exact_duplicate_check(self):
        self.record["split"]["role"] = "validation"
        second = self.second_record()
        second["source"]["book_id"] = "other-book"
        for key in ("book_group", "chapter_group", "problem_family"):
            second["split"][key] = "other-" + second["split"][key]
        second["rules"]["assumptions"].reverse()
        self.assertIn("exact_content", " ".join(validate_records([self.record, second])["errors"]))

    def test_variant_requires_existing_parent_and_matching_family(self):
        self.record["split"]["variant_of"] = "missing"
        self.assert_invalid("variant parent")
        self.record["split"]["variant_of"] = None
        second = self.second_record("quarantine")
        second["split"]["variant_of"] = self.record["id"]
        second["split"]["problem_family"] = "wrong-family"
        self.assertIn("parent's problem family", " ".join(validate_records([self.record, second])["errors"]))

    def test_variant_cycles_are_rejected(self):
        second = self.second_record("quarantine")
        self.record["split"]["variant_of"] = second["id"]
        second["split"]["variant_of"] = self.record["id"]
        self.assertIn("cycle", " ".join(validate_records([self.record, second])["errors"]))

    def test_duplicate_ids_are_rejected(self):
        self.assertIn("duplicate record id", " ".join(validate_records([self.record, copy.deepcopy(self.record)])["errors"]))

    def test_player_view_references_are_not_read_or_treated_as_certified(self):
        self.record["state"] = {"kind": "player_view_reference", "reference": {"path": "/does/not/exist.json", "sha256": "0" * 64, "schema": "future-canonical-player-view"}, "missing_context": []}
        self.record["rules"]["profile"] = "fixture profile"
        report = validate_records([self.record])
        self.assertTrue(report["valid"], report)
        self.assertFalse(report["policy_training_ready"])
        with self.assertRaisesRegex(ValueError, "audited adapter"):
            exercise_inputs(self.record)
        self.record["state"]["missing_context"] = ["scores"]
        self.assert_invalid("cannot omit")

    def test_duplicate_json_keys_and_nonstandard_numbers_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            for raw, fragment in (("{\"id\": 1, \"id\": 2}", "duplicate JSON key"), ("{\"id\": NaN}", "nonstandard JSON constant"), ("", "at least one")):
                with self.subTest(raw=raw):
                    path.write_text(raw)
                    with self.assertRaisesRegex(ValueError, fragment):
                        load_jsonl(path)

    def test_cli_exit_status_and_json_report(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main([str(EXAMPLE)]), 0)
        self.assertTrue(json.loads(output.getvalue())["valid"])
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["/does/not/exist.jsonl"]), 1)
        self.assertFalse(json.loads(output.getvalue())["valid"])

    def test_module_runs_without_site_packages(self):
        environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        process = subprocess.run([sys.executable, "-S", "-m", "riichi_ai.book_scenarios", str(EXAMPLE)], env=environment,
                                 capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertTrue(json.loads(process.stdout)["valid"])


if __name__ == "__main__":
    unittest.main()
