"""Information-boundary tests use only project-authored synthetic questions."""
import copy

import pytest

from riichi_ai.book_calls import (CONTEXT, convert_packet, partial_observation, validate_export,
                                  validate_record, validate_records)

REFERENCE = {"path": "synthetic.jsonl", "sha256": "a" * 64, "bytes": 1}


def packet(conditional=False):
    region = {"pdf_page": 1, "printed_page": "1", "region": "Synthetic question"}
    answer_region = {**region, "region": "Synthetic explanation"}
    def slot(value=None, *, status=None):
        return {"status": status or ("unknown" if value is None else "supplied"),
                "value": value, "evidence": region if value is not None else None}
    def tile(shape, ref):
        return {"shape": shape, "physical_red": "unknown" if shape[0] == "5" else "not_applicable",
                "diagram_ref": ref, "physical_id": None}
    shapes = ["1m", "2m", "3m", "4m", "5m", "6p", "6p", "8p", "1s", "2s", "3s", "E", "E"]
    offered = "7p" if conditional else "E"
    target = {"group": "decline_call", "engine_action": None, "offered_shape": offered,
              "consumed_tiles": [], "consumption_status": "not_applicable_to_decline"}
    if conditional:
        target = {"group": "call_selection_conditional_on_call", "kind": "CHI", "offered_shape": offered,
                  "consumed_tiles": [{"shape": "6p", "count": 1, "possible_question_refs": ["h5", "h6"],
                                      "which_identical_copy": "unspecified"},
                                     {"shape": "8p", "count": 1, "possible_question_refs": ["h7"],
                                      "which_identical_copy": "single_depicted_copy"}],
                  "resulting_meld_shapes": ["6p", "7p", "8p"], "engine_action": None}
    return {"schema": "riichi-ai-raw-call-source-packet-v1", "packet_id": "synthetic-conditional" if conditional else "synthetic-gate",
            "source": {"question_region": region, "answer_region": answer_region},
            "grouping": {"role": "development_quarantine"}, "permissions": {}, "evidence": {},
            "observation": {"decision_phase": slot("discard_response"), "rules": slot(),
                "trigger": {"offered_tile": slot(tile(offered, "offered")),
                            "source_relation": slot("previous_player") if conditional else slot(),
                            "source_absolute_seat": slot(), "drawn_tile": slot(status="not_applicable"),
                            "kan_declaration": slot(status="not_applicable")},
                "actor_tiles": {"concealed_source_order": slot([tile(s, f"h{i}") for i, s in enumerate(shapes)]),
                                "declared_melds": slot(), "own_riichi": slot()},
                "public_context": {name: slot() for name in CONTEXT}, "source_offered_alternatives": slot(),
                "complete_legal_action_set": slot(), "engine_pass_availability": slot()},
            "source_answer": {"recommendation": target, "conditions": {"status": "not_evaluated_as_gate_decision"},
                              "explanation_paraphrase": "Synthetic reasoning.",
                              "additional_explanation_context": {"actual_dora": {"status": "supplied", "value": {
                                  "shape": "3p", "source_label": "dora", "indicator_not_supplied": True}, "evidence": answer_region}}},
            "completeness": {}, "review": {"training_ready": False, "full_policy_ready": False}}


@pytest.mark.parametrize("conditional", [False, True])
def test_source_round_trip_and_export(conditional):
    source = packet(conditional)
    row = convert_packet(source, input_reference=REFERENCE)
    validate_record(row)
    assert row["provenance"]["raw_packet_auditor_only"] == source
    export = partial_observation(row)
    validate_export(export, row)
    assert set(export) == {"schema", "observation"}
    assert export["observation"]["public_context"]["actual_dora"]["status"] == "unknown"
    assert row["target_auditor_only"]["gate_preference"] == (None if conditional else "decline_call")
    assert (row["target_auditor_only"]["conditional_call_selection"] is not None) == conditional
    assert row["target_auditor_only"]["exact_engine_action"] is None
    assert row["target_auditor_only"]["later_discard_target"] is None
    assert validate_records([row])["valid"]
    export["observation"]["rules"]["value"] = {}
    assert row["observation"]["rules"]["value"] is None


def test_answer_only_dora_cannot_become_question_context():
    source = packet()
    source["observation"]["public_context"]["actual_dora"] = source["source_answer"]["additional_explanation_context"]["actual_dora"]
    with pytest.raises(ValueError, match="question evidence"):
        convert_packet(source, input_reference=REFERENCE)


@pytest.mark.parametrize("field,value", [("scores", [25000] * 4), ("rivers", []), ("honba", 0),
                                         ("dora_indicators", ["2p"])])
def test_absent_public_context_cannot_be_defaulted(field, value):
    source = packet()
    source["observation"]["public_context"][field]["value"] = value
    with pytest.raises(ValueError, match="fabricated"):
        convert_packet(source, input_reference=REFERENCE)


@pytest.mark.parametrize("phase", ["own_turn", "kan_response", "post_call_discard"])
def test_other_phases_cannot_relabel_kan_decline_as_pass(phase):
    source = packet()
    source["observation"]["decision_phase"]["value"] = phase
    with pytest.raises(ValueError, match="phase"):
        convert_packet(source, input_reference=REFERENCE)


def test_identical_copy_ambiguity_is_preserved():
    source = packet(True)
    source["source_answer"]["recommendation"]["consumed_tiles"][0]["possible_question_refs"].pop()
    with pytest.raises(ValueError, match="ambiguity"):
        convert_packet(source, input_reference=REFERENCE)


def test_conditional_selection_cannot_gain_a_gate_label():
    row = convert_packet(packet(True), input_reference=REFERENCE)
    row["target_auditor_only"]["gate_preference"] = "call"
    with pytest.raises(ValueError, match="differs"):
        validate_record(row)


@pytest.mark.parametrize("field", ["source_answer", "packet_id", "page", "explanation", "review"])
def test_export_rejects_answer_or_metadata(field):
    row = convert_packet(packet(), input_reference=REFERENCE)
    export = partial_observation(row)
    export["observation"][field] = "injected"
    with pytest.raises(ValueError, match="export"):
        validate_export(export, row)


def test_red_identity_and_engine_pass_cannot_be_invented():
    source = packet()
    source["observation"]["actor_tiles"]["concealed_source_order"]["value"][4]["physical_red"] = "normal"
    with pytest.raises(ValueError, match="red identity"):
        convert_packet(source, input_reference=REFERENCE)
    source = packet()
    source["source_answer"]["recommendation"]["engine_action"] = "PASS"
    with pytest.raises(ValueError, match="engine action"):
        convert_packet(source, input_reference=REFERENCE)


def test_answer_changes_do_not_change_question_export():
    source = packet()
    old = partial_observation(convert_packet(source, input_reference=REFERENCE))
    source["source_answer"]["explanation_paraphrase"] = "Different synthetic reasoning."
    source["source_answer"]["additional_explanation_context"]["actual_dora"]["value"]["shape"] = "9s"
    assert partial_observation(convert_packet(source, input_reference=REFERENCE)) == old


def test_collection_and_promotion_guards():
    row = convert_packet(packet(), input_reference=REFERENCE)
    assert not validate_records([row, copy.deepcopy(row)])["valid"]
    assert not validate_records([])["valid"]
    row["review"]["training_eligible"] = True
    assert not validate_records([row])["valid"]
