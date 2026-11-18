"""Strict, deliberately partial call questions; never an actor or engine adapter.

Only the reviewed discard-response intake is supported. Own-turn kan questions
require a different schema, and declining a call does not certify engine PASS.
The retained raw packet is auditor-only; exports contain question fields alone.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from typing import Any

SCHEMA = "riichi-ai-partial-call-record-v1"
QUESTION_SCHEMA = "riichi-ai-partial-call-question-v1"
SHAPES = {f"{n}{s}" for s in "mps" for n in range(1, 10)} | set("ESWNPFC")
CONTEXT = {"round", "seat", "turn", "scores", "honba", "deposits", "dora_indicators",
           "actual_dora", "rivers", "opponent_melds", "opponent_riichi", "preceding_events", "remaining_wall"}
UNKNOWN = {"status": "unknown", "value": None}
REVIEW = {"training_eligible": False, "evaluation_eligible": False, "full_policy_ready": False,
          "holdout_eligible": False, "legality": "not_established", "strategy": "not_validated",
          "scope": "Structural and source-boundary validation only; no arithmetic or policy evaluation."}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def exact(value: Any, keys: set[str], where: str) -> dict:
    require(isinstance(value, dict) and set(value) == keys, f"{where}: unexpected fields")
    return value


def content_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _tile(value: Any) -> dict:
    exact(value, {"shape", "physical_red", "diagram_ref", "physical_id"}, "tile")
    require(isinstance(value["shape"], str) and value["shape"] in SHAPES, "tile: invalid shape")
    require(value["physical_id"] is None, "tile: physical IDs cannot be constructed")
    expected = "unknown" if value["shape"] in ("5m", "5p", "5s") else "not_applicable"
    require(value["physical_red"] == expected, "tile: unresolved red identity was changed")
    require(isinstance(value["diagram_ref"], str) and bool(value["diagram_ref"]), "tile: missing diagram reference")
    return {"shape": value["shape"], "physical_red": value["physical_red"]}


def _slot(value: Any, question_region: dict, where: str) -> dict:
    exact(value, {"status", "value", "evidence"}, where)
    status = value["status"]
    require(status in ("supplied", "unknown", "not_applicable"), f"{where}: unsupported status")
    if status == "supplied":
        require(value["value"] is not None, f"{where}: supplied cannot be null")
        require(value["evidence"] == question_region, f"{where}: question evidence required; answer context excluded")
    else:
        require(value["value"] is None and value["evidence"] is None, f"{where}: absent value cannot be fabricated")
    return {"status": status, "value": copy.deepcopy(value["value"])}


def _question(packet: dict) -> dict:
    raw = exact(packet["observation"], {"decision_phase", "trigger", "actor_tiles", "public_context", "rules",
                                     "source_offered_alternatives", "complete_legal_action_set", "engine_pass_availability"}, "observation")
    question = packet["source"]["question_region"]
    exact(question, {"pdf_page", "printed_page", "region"}, "question_region")
    require(type(question["pdf_page"]) is int and question["pdf_page"] > 0, "question_region: invalid page")
    require(question != packet["source"]["answer_region"], "question and answer regions must be distinct")
    obs = {name: _slot(raw[name], question, name) for name in (
        "decision_phase", "rules", "source_offered_alternatives", "complete_legal_action_set", "engine_pass_availability")}
    require(obs["decision_phase"] == {"status": "supplied", "value": "discard_response"},
            "phase: only discard_response supported; own-turn kan decline is not PASS")
    for name in ("rules", "complete_legal_action_set", "engine_pass_availability"):
        require(obs[name] == UNKNOWN, f"{name}: this intake does not certify rules or legal actions")
    alternatives = obs["source_offered_alternatives"]
    require(alternatives == UNKNOWN or alternatives == {"status": "supplied", "value": {
        "choices": ["call", "decline_call"], "exhaustive_engine_actions": False}}, "alternatives: cannot claim exhaustive actions")
    trigger = exact(raw["trigger"], {"offered_tile", "source_relation", "source_absolute_seat", "drawn_tile", "kan_declaration"}, "trigger")
    obs["trigger"] = {name: _slot(value, question, name) for name, value in trigger.items()}
    require(obs["trigger"]["offered_tile"]["status"] == "supplied", "offered tile required")
    obs["trigger"]["offered_tile"]["value"] = _tile(trigger["offered_tile"]["value"])
    relation = obs["trigger"]["source_relation"]
    require(relation == UNKNOWN or relation == {"status": "supplied", "value": "previous_player"}, "source relation unsupported")
    require(obs["trigger"]["source_absolute_seat"] == UNKNOWN, "absolute seat cannot be constructed")
    for name in ("drawn_tile", "kan_declaration"):
        require(obs["trigger"][name] == {"status": "not_applicable", "value": None}, f"{name}: wrong phase")
    actor = exact(raw["actor_tiles"], {"concealed_source_order", "declared_melds", "own_riichi"}, "actor_tiles")
    obs["actor_tiles"] = {name: _slot(value, question, name) for name, value in actor.items()}
    hand = obs["actor_tiles"]["concealed_source_order"]
    require(hand["status"] == "supplied" and isinstance(hand["value"], list) and len(hand["value"]) == 13,
            "hand: intake requires thirteen depicted concealed tiles")
    tiles = actor["concealed_source_order"]["value"]
    hand["value"] = [_tile(tile) for tile in tiles]
    require(len({tile["diagram_ref"] for tile in tiles}) == 13, "hand: duplicate diagram references")
    counts = Counter(tile["shape"] for tile in tiles)
    counts[obs["trigger"]["offered_tile"]["value"]["shape"]] += 1
    require(max(counts.values()) <= 4, "hand and offered tile: more than four copies")
    for name in ("declared_melds", "own_riichi"):
        require(obs["actor_tiles"][name] == UNKNOWN, f"{name}: absent context cannot be inferred")
    context = exact(raw["public_context"], CONTEXT, "public_context")
    obs["public_context"] = {name: _slot(value, question, name) for name, value in context.items()}
    for name, slot in obs["public_context"].items():
        if name in ("round", "seat", "turn"):
            require(slot == UNKNOWN or (slot["status"] == "supplied" and isinstance(slot["value"], str)
                                       and bool(slot["value"])), f"{name}: invalid text")
        elif name == "actual_dora" and slot["status"] == "supplied":
            value = exact(slot["value"], {"shape", "source_label", "indicator_not_supplied"}, "actual_dora")
            require(value["shape"] in SHAPES and value["source_label"] == "dora"
                    and value["indicator_not_supplied"] is True, "actual dora must not become indicator")
        else:
            require(slot == UNKNOWN, f"{name}: unavailable context must stay unknown")
    return obs


def _target(packet: dict, observation: dict) -> dict:
    source = packet["source_answer"]["recommendation"]
    require(isinstance(source, dict), "source recommendation must be an object")
    require(source.get("engine_action") is None, "source recommendation cannot certify engine action")
    offered = observation["trigger"]["offered_tile"]["value"]["shape"]
    require(source.get("offered_shape") == offered, "answer offered tile differs from question")
    base = {"gate_preference": None, "conditional_call_selection": None, "exact_engine_action": None,
            "legality": "not_established", "later_discard_target": None,
            "objective_calibration": "not_validated_for_average_rank"}
    if source.get("group") == "decline_call":
        exact(source, {"group", "engine_action", "offered_shape", "consumed_tiles", "consumption_status"}, "decline recommendation")
        require(source["consumed_tiles"] == [] and source["consumption_status"] == "not_applicable_to_decline",
                "decline cannot consume tiles")
        base["gate_preference"] = "decline_call"
    elif source.get("group") == "call_selection_conditional_on_call":
        exact(source, {"group", "kind", "offered_shape", "consumed_tiles", "resulting_meld_shapes", "engine_action"}, "conditional recommendation")
        require(source["kind"] == "CHI", "conditional intake currently supports CHI only")
        require(len(offered) == 2 and offered[1] in "mps", "CHI cannot use an honor tile")
        require(observation["trigger"]["source_relation"] == {"status": "supplied", "value": "previous_player"},
                "CHI requires a source-supplied previous-player relation")
        require(packet["source_answer"]["conditions"]["status"] == "not_evaluated_as_gate_decision",
                "conditional call cannot become a gate target")
        refs: dict[str, list[str]] = {}
        for tile in packet["observation"]["actor_tiles"]["concealed_source_order"]["value"]:
            refs.setdefault(tile["shape"], []).append(tile["diagram_ref"])
        consumed = source["consumed_tiles"]
        require(isinstance(consumed, list) and len(consumed) == 2, "CHI requires two consumed shapes")
        shapes = []
        for tile in consumed:
            exact(tile, {"shape", "count", "possible_question_refs", "which_identical_copy"}, "consumption")
            shape = tile["shape"]
            require(type(tile["count"]) is int and tile["count"] == 1 and shape in refs, "consumed shape absent")
            require(tile["possible_question_refs"] == refs[shape], "copy ambiguity must retain all depicted candidates")
            expected = "unspecified" if len(refs[shape]) > 1 else "single_depicted_copy"
            require(tile["which_identical_copy"] == expected, "identical physical copy cannot be guessed")
            shapes.append(shape)
        meld = sorted(shapes + [offered])
        require(all(len(shape) == 2 and shape[1] == offered[1] for shape in meld)
                and [int(shape[0]) for shape in meld] == list(range(int(meld[0][0]), int(meld[0][0]) + 3)),
                "CHI shapes do not form a sequence")
        require(source["resulting_meld_shapes"] == meld, "resulting meld differs")
        base["conditional_call_selection"] = {"kind": "CHI", "offered_shape": offered,
            "consumed_tiles": copy.deepcopy(consumed), "resulting_meld_shapes": meld,
            "condition": "call_already_assumed", "physical_action_resolved": False}
    else:
        raise ValueError("unsupported recommendation; own-turn kan decline must not become response PASS")
    return base


def convert_packet(packet: dict, *, input_reference: dict) -> dict:
    exact(packet, {"schema", "packet_id", "source", "grouping", "permissions", "observation", "source_answer",
                   "completeness", "review", "evidence"}, "raw packet")
    require(packet["schema"] == "riichi-ai-raw-call-source-packet-v1", "unsupported raw packet")
    require(isinstance(packet["packet_id"], str) and bool(packet["packet_id"]), "packet ID required")
    require(packet["grouping"].get("role") == "development_quarantine", "source must remain development quarantine")
    require(packet["review"].get("training_ready") is False and packet["review"].get("full_policy_ready") is False,
            "partial packets cannot be promoted")
    exact(input_reference, {"path", "sha256", "bytes"}, "input reference")
    require(isinstance(input_reference["path"], str) and bool(input_reference["path"]), "reference path required")
    require(isinstance(input_reference["sha256"], str) and len(input_reference["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in input_reference["sha256"]), "reference digest invalid")
    require(type(input_reference["bytes"]) is int and input_reference["bytes"] > 0, "reference bytes invalid")
    observation = _question(packet)
    return {"schema": SCHEMA, "id": packet["packet_id"],
            "provenance": {"input_reference": copy.deepcopy(input_reference), "packet_sha256": content_sha256(packet),
                           "raw_packet_auditor_only": copy.deepcopy(packet)},
            "observation": observation, "target_auditor_only": _target(packet, observation), "review": copy.deepcopy(REVIEW)}


def validate_record(record: dict) -> None:
    exact(record, {"schema", "id", "provenance", "observation", "target_auditor_only", "review"}, "record")
    provenance = exact(record["provenance"], {"input_reference", "packet_sha256", "raw_packet_auditor_only"}, "provenance")
    expected = convert_packet(provenance["raw_packet_auditor_only"], input_reference=provenance["input_reference"])
    require(record == expected, "record differs from validated source-derived representation")


def partial_observation(record: dict) -> dict:
    validate_record(record)
    return {"schema": QUESTION_SCHEMA, "observation": copy.deepcopy(record["observation"])}


def validate_export(export: dict, record: dict) -> None:
    """Source-bound validation: a standalone partial export cannot establish provenance."""
    require(export == partial_observation(record), "question export differs or includes answer/source metadata")


def validate_records(records: list[dict]) -> dict:
    errors = [] if records else ["collection is empty"]
    ids = []
    for index, record in enumerate(records):
        try:
            validate_record(record)
            ids.append(record["id"])
        except (ValueError, KeyError, TypeError, IndexError) as error:
            errors.append(f"record {index}: {error}")
    if len(ids) != len(set(ids)):
        errors.append("duplicate record IDs")
    return {"schema": "riichi-ai-partial-call-validation-v1", "valid": not errors, "errors": errors,
            "records": len(records), "training_eligible": False, "evaluation_eligible": False,
            "full_policy_ready": False, "scope": REVIEW["scope"]}
