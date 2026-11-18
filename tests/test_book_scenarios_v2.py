"""Partial-schema invariants using project-authored synthetic fixtures only."""

import copy
import unittest

from riichi_ai.book_scenarios import validate_record as validate_v1
from riichi_ai.book_scenarios_v2 import (
    CONTEXT_FIELDS, SCHEMA, UNSUPPLIED_FIELDS, content_sha256, convert_packet,
    partial_observation, validate_record, validate_records,
)


REFERENCE = {"path": "fixture.jsonl", "bytes": 1, "sha256": "a" * 64}


def packet():
    hand = [f"{n}m" for n in range(1, 7)] + [f"{n}p" for n in range(1, 7)] + ["E", "E"]
    return {
        "schema": "raw-source-packet-v1",
        "packet_id": "project-authored-example",
        "source": {"book_id": "authored-fixture", "path": "fixture.pdf", "sha256": "b" * 64,
                   "question_id": "fixture-1", "problem_family": "authored-family"},
        "source_fields": {
            "concealed_hand_before_draw": hand[:-1], "drawn_tile": "E",
            "shown_hand_including_draw": hand, "round_text": "East 1", "seat_text": "East",
            "turn_text": None, "dora_text": {"label": "dora indicator", "tile_type": "9p"},
            "melds_text": "No declared melds shown.",
            "red_five_status": "Physical red identity unresolved.",
            "visual_red_five_candidates": [{
                "question_row_position_one_based": 5, "shape_tile": "5m",
                "observation": "Project-authored illustrative ambiguous colour.",
            }],
        },
        "source_answer": {
            "answer_text": "Discard E.",
            "explanation_paraphrase": "Synthetic explanation to check data boundaries.",
            "reported_acceptance_counts": [], "status": "unverified source recommendation",
        },
        "grouping": {
            "intended_use": "development", "whole_book_group": "authored-fixture",
            "problem_family": "authored-family", "split_assignment": "quarantine",
            "holdout_eligible": False, "cross_stage_family_links": [],
        },
        "evidence": {"method": "Synthetic test fixture, no source book copied."},
        "missing_context": ["No score vector, rivers or complete rules."],
        "count_conventions": {"interpretation": "Not established."},
        "rules_assumptions": ["Do not infer full observations."],
        "status": {"model_ready": False, "role": "quarantine"},
    }


class PartialBookScenarioTests(unittest.TestCase):
    def setUp(self):
        self.packet = packet()
        self.row = convert_packet(self.packet, input_reference=REFERENCE, arithmetic_reference=None)

    def assert_invalid(self, fragment):
        result = validate_records([self.row])
        self.assertFalse(result["valid"], result)
        self.assertIn(fragment, " ".join(result["errors"]))

    def test_round_trip_preserves_supplied_tiles_context_and_all_auditor_fields(self):
        self.assertEqual(self.row["schema"], SCHEMA)
        fields = self.packet["source_fields"]
        obs = self.row["observation"]
        self.assertEqual([tile["shape"] for tile in obs["hand_question_order"]], fields["shown_hand_including_draw"])
        self.assertEqual([tile["shape"] for tile in obs["before_draw"]["value"]], fields["concealed_hand_before_draw"])
        self.assertEqual(obs["drawn_tile"]["value"]["shape"], fields["drawn_tile"])
        for name in CONTEXT_FIELDS:
            self.assertEqual(obs["context"][name]["value"], fields.get(name))
        self.assertEqual(self.row["answer"]["source_transcription"], self.packet["source_answer"])
        self.assertEqual(self.row["answer"]["count_conventions"], self.packet["count_conventions"])
        self.assertEqual(self.row["provenance"]["source"], self.packet["source"])
        self.assertEqual(self.row["provenance"]["evidence"], self.packet["evidence"])
        self.assertEqual(self.row["provenance"]["packet_content_sha256"], content_sha256(self.packet))
        self.assertEqual(self.row["grouping"], self.packet["grouping"])

    def test_unknown_draw_is_retained_without_reconstructing_before_draw(self):
        self.packet["source_fields"]["drawn_tile"] = None
        self.packet["source_fields"]["concealed_hand_before_draw"] = None
        row = convert_packet(self.packet, input_reference=REFERENCE, arithmetic_reference=None)
        self.assertEqual(row["observation"]["drawn_tile"], {"status": "unknown", "value": None})
        self.assertEqual(row["observation"]["before_draw"], {"status": "unknown", "value": None})

    def test_unknown_is_not_a_default_zero_or_empty_array(self):
        for field in UNSUPPLIED_FIELDS:
            with self.subTest(field=field):
                self.row["observation"]["unsupplied"][field]["value"] = []
                self.assert_invalid("cannot be fabricated")
                self.row["observation"]["unsupplied"][field]["value"] = None
        self.row["observation"]["context"]["turn_text"] = {"status": "unknown", "value": "0"}
        self.assert_invalid("unknown must have null")

    def test_partial_export_excludes_answer_source_id_and_review_and_is_deep_copy(self):
        before = partial_observation(self.row)
        self.assertEqual(set(before), {"schema", *self.row["observation"]})
        self.row["answer"]["source_transcription"]["explanation_paraphrase"] = "Different explanation."
        self.row["provenance"]["source"]["question_id"] = "Changed metadata"
        self.assertEqual(partial_observation(self.row), before)
        self.row["answer"]["source_transcription"]["answer_text"] = "Discard 1m."
        self.row["answer"]["recommendation"]["components"][0]["tile"]["shape"] = "1m"
        self.assertEqual(partial_observation(self.row), before)
        before["hand_question_order"][0]["shape"] = "9s"
        self.assertEqual(self.row["observation"]["hand_question_order"][0]["shape"], "1m")

    def test_answer_injected_into_observation_is_rejected(self):
        self.row["observation"]["answer"] = "E"
        self.assert_invalid("observation")

    def test_invented_dora_indicator_and_false_red_identity_are_rejected(self):
        self.row["observation"]["hand_question_order"][4]["physical_red"] = "red"
        self.assert_invalid("cannot resolve")
        self.row["observation"]["hand_question_order"][4]["physical_red"] = "unknown"
        self.row["observation"]["context"]["dora_text"]["value"]["label"] = "inferred indicator"
        self.assert_invalid("retain source label")

    def test_dora_tile_and_indicator_remain_distinct(self):
        self.packet["source_fields"]["dora_text"] = {"label": "dora", "tile_type": "P"}
        row = convert_packet(self.packet, input_reference=REFERENCE, arithmetic_reference=None)
        self.assertEqual(row["observation"]["context"]["dora_text"]["value"], {"label": "dora", "tile_type": "P"})

    def test_colour_candidate_must_reference_matching_question_position(self):
        self.row["observation"]["red_notation"]["visual_candidates"][0]["question_row_position_one_based"] = 4
        self.assert_invalid("matching five")

    def test_drawn_tile_is_separate_from_sorted_answer_row(self):
        self.row["observation"]["hand_question_order"] = self.row["observation"]["hand_question_order"][::-1]
        self.assert_invalid("question order")

    def test_four_copy_limit_includes_uncertain_physical_tiles(self):
        self.row["observation"]["hand_question_order"][:5] = [{"shape": "5m", "physical_red": "unknown"}] * 5
        self.assert_invalid("more than four")

    def test_compound_riichi_is_one_semantic_recommendation(self):
        self.packet["source_answer"]["answer_text"] = "Discard E and declare riichi."
        row = convert_packet(self.packet, input_reference=REFERENCE, arithmetic_reference=None)
        action = row["answer"]["recommendation"]
        self.assertEqual(action["kind"], "compound_recommendation")
        self.assertEqual([part["kind"] for part in action["components"]], ["DISCARD", "DECLARE_RIICHI"])
        self.assertEqual(action["legality"], "not_established")
        row["answer"]["recommendation"]["components"].pop()
        self.row = row
        self.assert_invalid("invalid action kind")

    def test_source_answer_and_normalized_recommendation_cannot_diverge(self):
        self.row["answer"]["recommendation"]["components"][0]["tile"]["shape"] = "1m"
        self.assert_invalid("differs from source answer")

    def test_unsupported_free_text_answer_requires_review(self):
        self.packet["source_answer"]["answer_text"] = "Either E or 1m; decide somehow."
        with self.assertRaisesRegex(ValueError, "explicit human normalization"):
            convert_packet(self.packet, input_reference=REFERENCE, arithmetic_reference=None)

    def test_caveat_is_not_a_visibility_fact_or_a_dama_label(self):
        self.packet["packet_id"] = "statistics-check-problem-1-question-2"
        self.packet["source_answer"]["answer_text"] = "Discard E and declare riichi."
        self.packet["source_answer"]["explanation_paraphrase"] = "A choice unless at least three relevant tiles are visible."
        row = convert_packet(self.packet, input_reference=REFERENCE, arithmetic_reference=None)
        caveat = row["answer"]["conditions"][0]
        self.assertIsNone(caveat["recommendation"])
        self.assertEqual(caveat["condition_status"], "not_evaluated")
        self.assertEqual(row["observation"]["unsupplied"]["opponent_rivers"]["value"], None)
        row["answer"]["conditions"] = []
        self.row = row
        self.assert_invalid("required source caveat")

    def test_conditional_alternative_is_preserved_separately(self):
        self.packet["packet_id"] = "uzaku-301-q008"
        self.packet["source_answer"]["explanation_paraphrase"] = "Alternative when greater hand value is specifically needed."
        row = convert_packet(self.packet, input_reference=REFERENCE, arithmetic_reference=None)
        self.assertEqual(row["answer"]["conditions"][0]["recommendation"]["components"][0]["tile"]["shape"], "2p")
        self.assertEqual(row["answer"]["recommendation"]["components"][0]["tile"]["shape"], "E")
        self.assertNotIn("conditions", partial_observation(row))

    def test_condition_cannot_be_asserted_as_observed_or_detached_from_source(self):
        self.packet["packet_id"] = "uzaku-301-q008"
        with self.assertRaisesRegex(ValueError, "fixed source wording changed"):
            convert_packet(self.packet, input_reference=REFERENCE, arithmetic_reference=None)
        self.packet["source_answer"]["explanation_paraphrase"] = "Alternative when greater hand value is specifically needed."
        self.row = convert_packet(self.packet, input_reference=REFERENCE, arithmetic_reference=None)
        self.row["answer"]["conditions"][0]["condition_status"] = "observed"
        self.assert_invalid("cannot be asserted as observed")
        self.row["answer"]["conditions"][0]["condition_status"] = "not_evaluated"
        self.row["answer"]["conditions"][0]["condition_text"] = "The player is last by 10000."
        self.assert_invalid("required source caveat")

    def test_source_ev_remains_a_source_claim_not_rank_utility(self):
        self.packet["source_answer"]["expected_net_hand_result_claims"] = [
            {"choice": "Example A", "points": 300}, {"choice": "Example B", "points": -300},
        ]
        row = convert_packet(self.packet, input_reference=REFERENCE, arithmetic_reference=None)
        self.assertEqual(row["answer"]["source_transcription"]["expected_net_hand_result_claims"][1]["points"], -300)
        self.assertFalse(row["answer"]["project_objective"]["source_claims_calibrated_to_objective"])
        row["answer"]["project_objective"]["values"] = [300, -300]
        self.row = row
        self.assert_invalid("cannot replace rank utility")

    def test_arithmetic_reference_only_attaches_to_count_bearing_case(self):
        self.packet["source_answer"]["reported_acceptance_counts"] = [{"discard": "E", "reported_count": 3}]
        with self.assertRaisesRegex(ValueError, "only the audited"):
            convert_packet(self.packet, input_reference=REFERENCE, arithmetic_reference=None)
        row = convert_packet(self.packet, input_reference=REFERENCE, arithmetic_reference=REFERENCE)
        self.assertEqual(row["review"]["strategic_label_status"], "pending")

    def test_no_eligibility_promotion_in_this_schema(self):
        for field in ("training_eligible", "evaluation_eligible", "full_policy_ready"):
            with self.subTest(field=field):
                self.row["review"][field] = True
                self.assert_invalid("deliberately ineligible")
                self.row["review"][field] = False
        self.row["grouping"]["split_assignment"] = "train"
        self.assert_invalid("development/quarantine only")

    def test_duplicate_ids_and_inconsistent_family_groups_fail(self):
        self.assertFalse(validate_records([self.row, copy.deepcopy(self.row)])["valid"])
        other = copy.deepcopy(self.row)
        other["id"] = "different-id"
        other["grouping"]["whole_book_group"] = "different-book-group"
        result = validate_records([self.row, other])
        self.assertFalse(result["valid"])
        self.assertIn("inconsistent groups", " ".join(result["errors"]))

    def test_v1_cannot_silently_consume_v2_as_full_or_discard_only_data(self):
        with self.assertRaises(ValueError):
            validate_v1(self.row)
        self.assertFalse(validate_records([])["valid"])
        validate_record(self.row)

    def test_unrecognized_question_fields_cannot_be_silently_dropped(self):
        self.packet["source_fields"]["new_context"] = "New data needs a schema decision."
        with self.assertRaisesRegex(ValueError, "explicitly mapped"):
            convert_packet(self.packet, input_reference=REFERENCE, arithmetic_reference=None)


if __name__ == "__main__":
    unittest.main()
