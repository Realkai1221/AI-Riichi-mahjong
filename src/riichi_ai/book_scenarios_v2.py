"""Strict partial teaching scenarios; no simulator or full-policy adapter.

The v1 intake remains unchanged. This schema preserves shape-level uncertainty,
supplied context and compound recommendations without asserting a legal action.
The only observation export is explicitly partial and contains no answer fields.
"""

from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import re
from typing import Any


SCHEMA = "riichi-ai-partial-book-scenario-v2"
SHAPES = {f"{n}{s}" for s in "mps" for n in range(1, 10)} | set("ESWNPFC")
CONTEXT_FIELDS = {
    "round_text", "seat_text", "turn_text", "dora_text", "melds_text", "section_context",
}
UNSUPPLIED_FIELDS = {
    "scores", "honba", "kyotaku", "rule_profile", "opponent_rivers",
    "opponent_melds", "preceding_events", "legal_actions", "remaining_wall",
    "own_riichi_status",
}
INELIGIBILITY = [
    "Partial question context is not an audited full player observation.",
    "Physical red-five identity and the complete legal-action set are unresolved.",
    "Source recommendations have not been validated for average-rank utility.",
    "This intake is development/quarantine material, not untouched held-out data.",
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: Any, keys: set[str], where: str) -> dict:
    _require(isinstance(value, dict), f"{where}: expected an object")
    _require(set(value) == keys, f"{where}: expected exactly {sorted(keys)}")
    return value


def _text(value: Any, where: str) -> None:
    _require(isinstance(value, str) and bool(value.strip()), f"{where}: expected nonempty text")


def _texts(value: Any, where: str, *, empty: bool = False) -> None:
    _require(isinstance(value, list) and (empty or bool(value)), f"{where}: expected a text list")
    for item in value:
        _text(item, where)


def content_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _digest(value: Any, where: str) -> None:
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             f"{where}: expected lowercase SHA-256")


def _slot(value: Any) -> dict:
    return {"status": "unknown" if value is None else "supplied", "value": copy.deepcopy(value)}


def _validate_slot(value: Any, where: str) -> dict:
    slot = _object(value, {"status", "value"}, where)
    _require(slot["status"] in ("unknown", "supplied"), f"{where}: invalid status")
    _require((slot["value"] is None) == (slot["status"] == "unknown"),
             f"{where}: unknown must have null value and supplied must have a value")
    return slot


def _tile(shape: str) -> dict:
    return {"shape": shape, "physical_red": "unknown" if shape in {"5m", "5p", "5s"} else "not_applicable"}


def _validate_tile(value: Any, where: str) -> None:
    tile = _object(value, {"shape", "physical_red"}, where)
    _require(isinstance(tile["shape"], str) and tile["shape"] in SHAPES, f"{where}: invalid shape")
    expected = "unknown" if tile["shape"] in {"5m", "5p", "5s"} else "not_applicable"
    _require(tile["physical_red"] == expected, f"{where}: cannot resolve physical red identity in this schema")


def _hand(value: Any, length: int, where: str) -> list[str]:
    _require(isinstance(value, list) and len(value) == length, f"{where}: expected {length} tiles")
    for tile in value:
        _validate_tile(tile, where)
    shapes = [tile["shape"] for tile in value]
    _require(max(Counter(shapes).values()) <= 4, f"{where}: more than four copies")
    return shapes


def _recommendation(discard: str, riichi: bool) -> dict:
    components = [{"kind": "DISCARD", "tile": _tile(discard), "discard_origin": "unknown"}]
    if riichi:
        components.append({"kind": "DECLARE_RIICHI"})
    return {
        "kind": "compound_recommendation" if riichi else "discard_recommendation",
        "components": components,
        "semantics": "All listed components form one source recommendation; list order is not engine execution order.",
        "legality": "not_established",
    }


def _validate_recommendation(value: Any, hand: list[str], where: str) -> None:
    action = _object(value, {"kind", "components", "semantics", "legality"}, where)
    _text(action["semantics"], f"{where}.semantics")
    _require(action["legality"] == "not_established", f"{where}: cannot certify legality")
    components = action["components"]
    _require(isinstance(components, list) and len(components) in (1, 2), f"{where}: invalid components")
    discard = _object(components[0], {"kind", "tile", "discard_origin"}, where)
    _require(discard["kind"] == "DISCARD" and discard["discard_origin"] == "unknown",
             f"{where}: expected discard without inferred origin")
    _validate_tile(discard["tile"], where)
    _require(discard["tile"]["shape"] in hand, f"{where}: discarded shape is absent")
    if len(components) == 2:
        _require(components[1] == {"kind": "DECLARE_RIICHI"}, f"{where}: invalid second component")
        _require(action["kind"] == "compound_recommendation", f"{where}: compound action was flattened")
    else:
        _require(action["kind"] == "discard_recommendation", f"{where}: invalid action kind")


def _conditions(packet: dict) -> list[dict]:
    """These are review annotations, not executable conditions or facts."""
    if packet["packet_id"] == "uzaku-301-q008":
        _require("greater hand value is specifically needed" in packet["source_answer"]["explanation_paraphrase"],
                 "condition: fixed source wording changed; review annotation again")
        return [{
            "kind": "conditional_alternative",
            "condition_text": "Greater hand value is specifically needed.",
            "condition_status": "not_evaluated",
            "source_basis": "source_answer.explanation_paraphrase",
            "recommendation": _recommendation("2p", False),
            "interpretation": "Source mentions discarding 2p to reject a 1p draw; no actual score requirement is supplied.",
        }]
    if packet["packet_id"] == "statistics-check-problem-1-question-2":
        _require("unless at least three relevant tiles are visible" in packet["source_answer"]["explanation_paraphrase"],
                 "condition: fixed source wording changed; review annotation again")
        return [{
            "kind": "recommendation_caveat",
            "condition_text": "At least three relevant wait tiles are visible.",
            "condition_status": "not_evaluated",
            "source_basis": "source_answer.explanation_paraphrase",
            "recommendation": None,
            "interpretation": "The source flags this caveat; precise visibility convention and actual count remain unknown. No replacement action is inferred.",
        }]
    return []


def convert_packet(packet: dict, *, input_reference: dict, arithmetic_reference: dict | None) -> dict:
    """Convert a reviewed packet without recovering any absent game state.

    Callers must authenticate the input file separately. The original source,
    answer, evidence, conventions and revision are retained for lossless review.
    """
    _require(packet.get("schema") == "raw-source-packet-v1", "packet: unsupported schema")
    fields = packet["source_fields"]
    _require(set(fields) <= {
        "concealed_hand_before_draw", "drawn_tile", "shown_hand_including_draw",
        "red_five_status", "visual_red_five_candidates", *CONTEXT_FIELDS,
    }, "packet: unrecognized source fields must be explicitly mapped")
    match = re.fullmatch(r"Discard ([1-9][mps]|[ESWNPFC])( and declare riichi)?\.",
                         packet["source_answer"]["answer_text"])
    _require(match is not None, "packet: recommendation requires explicit human normalization")
    before = fields["concealed_hand_before_draw"]
    drawn = fields["drawn_tile"]
    record = {
        "schema": SCHEMA,
        "id": packet["packet_id"],
        "provenance": {
            "input_reference": copy.deepcopy(input_reference),
            "packet_content_sha256": content_sha256(packet),
            "source": copy.deepcopy(packet["source"]),
            "evidence": copy.deepcopy(packet["evidence"]),
            "revision": copy.deepcopy(packet.get("revision")),
        },
        "permissions": {
            "basis": "User supplied local books and authorized preparation of research scenarios.",
            "local_preparation": True,
            "training_permission_assessment": "not_assessed",
            "redistribution_permission_assessment": "not_assessed",
        },
        "grouping": copy.deepcopy(packet["grouping"]),
        "observation": {
            "kind": "partial_teaching_question",
            "hand_question_order": [_tile(shape) for shape in fields["shown_hand_including_draw"]],
            "before_draw": _slot(None if before is None else [_tile(shape) for shape in before]),
            "drawn_tile": _slot(None if drawn is None else _tile(drawn)),
            "context": {name: _slot(fields.get(name)) for name in sorted(CONTEXT_FIELDS)},
            "unsupplied": {name: _slot(None) for name in sorted(UNSUPPLIED_FIELDS)},
            "red_notation": {
                "status": "unresolved",
                "source_note": fields["red_five_status"],
                "visual_candidates": copy.deepcopy(fields["visual_red_five_candidates"]),
            },
        },
        "answer": {
            "source_transcription": copy.deepcopy(packet["source_answer"]),
            "recommendation": _recommendation(match[1], match[2] is not None),
            "conditions": _conditions(packet),
            "count_conventions": copy.deepcopy(packet["count_conventions"]),
            "project_objective": {
                "name": "average_rank_utility", "values": [1.5, 0.5, -0.5, -1.5],
                "source_claims_calibrated_to_objective": False,
            },
        },
        "review": {
            "source_missing_context": copy.deepcopy(packet["missing_context"]),
            "source_rule_assumptions": copy.deepcopy(packet["rules_assumptions"]),
            "source_status": copy.deepcopy(packet["status"]),
            "arithmetic_reference": copy.deepcopy(arithmetic_reference),
            "arithmetic_scope": "Prior conditional shape-count check only; no Stage 3B arithmetic rerun.",
            "strategic_label_status": "pending",
            "full_policy_ready": False,
            "training_eligible": False,
            "evaluation_eligible": False,
            "ineligibility_reasons": copy.deepcopy(INELIGIBILITY),
        },
    }
    validate_record(record)
    return record


def validate_record(record: Any) -> None:
    """Validate this deliberately partial schema; never follow a reference."""
    _object(record, {"schema", "id", "provenance", "permissions", "grouping", "observation", "answer", "review"}, "record")
    _require(record["schema"] == SCHEMA, "schema: unsupported version")
    _text(record["id"], "id")
    provenance = _object(record["provenance"], {
        "input_reference", "packet_content_sha256", "source", "evidence", "revision",
    }, "provenance")
    for ref in (provenance["input_reference"],):
        _reference(ref, "provenance.input_reference")
    _digest(provenance["packet_content_sha256"], "packet_content_sha256")
    source = provenance["source"]
    _require(isinstance(source, dict), "source: expected an object")
    for key in ("book_id", "path", "question_id", "problem_family"):
        _text(source.get(key), f"source.{key}")
    _digest(source.get("sha256"), "source.sha256")
    _require(isinstance(provenance["evidence"], dict), "evidence: expected an object")
    _require(provenance["revision"] is None or isinstance(provenance["revision"], dict), "revision: invalid metadata")
    permissions = _object(record["permissions"], {
        "basis", "local_preparation", "training_permission_assessment", "redistribution_permission_assessment",
    }, "permissions")
    _text(permissions["basis"], "permissions.basis")
    _require(permissions["local_preparation"] is True, "permissions: local preparation required")
    for key in ("training_permission_assessment", "redistribution_permission_assessment"):
        _require(permissions[key] == "not_assessed", f"permissions.{key}: unsupported assessment")
    grouping = _object(record["grouping"], {
        "intended_use", "whole_book_group", "problem_family", "split_assignment",
        "holdout_eligible", "cross_stage_family_links",
    }, "grouping")
    _require(grouping["intended_use"] == "development" and grouping["split_assignment"] == "quarantine",
             "grouping: this schema supports development/quarantine only")
    _require(grouping["holdout_eligible"] is False, "grouping: exposed cases are not held out")
    for key in ("whole_book_group", "problem_family"):
        _text(grouping[key], f"grouping.{key}")
    _require(grouping["problem_family"] == source["problem_family"], "grouping: source family differs")
    _texts(grouping["cross_stage_family_links"], "grouping.cross_stage_family_links", empty=True)
    _require(record["id"] not in grouping["cross_stage_family_links"], "grouping: cannot link to itself")
    observation = _object(record["observation"], {
        "kind", "hand_question_order", "before_draw", "drawn_tile", "context", "unsupplied", "red_notation",
    }, "observation")
    _require(observation["kind"] == "partial_teaching_question", "observation: only partial questions supported")
    hand = _hand(observation["hand_question_order"], 14, "hand_question_order")
    before = _validate_slot(observation["before_draw"], "before_draw")
    drawn = _validate_slot(observation["drawn_tile"], "drawn_tile")
    if before["status"] == "supplied":
        before_hand = _hand(before["value"], 13, "before_draw")
        _require(drawn["status"] == "supplied", "before_draw: supplied hand requires supplied draw")
        _validate_tile(drawn["value"], "drawn_tile")
        _require(before_hand + [drawn["value"]["shape"]] == hand,
                 "before_draw: question order must retain the separate drawn tile")
    elif drawn["status"] == "supplied":
        _validate_tile(drawn["value"], "drawn_tile")
        _require(drawn["value"]["shape"] in hand, "drawn_tile: absent from hand")
    context = _object(observation["context"], CONTEXT_FIELDS, "context")
    for name, value in context.items():
        slot = _validate_slot(value, f"context.{name}")
        if slot["status"] == "unknown":
            continue
        if name == "dora_text":
            dora = slot["value"]
            _require(isinstance(dora, dict) and set(dora) in (
                {"label", "tile_type"}, {"label", "tile_type", "visual_note"},
            ), "dora_text: invalid keys")
            _require(dora["label"] in ("dora", "dora indicator"), "dora_text: retain source label")
            _require(isinstance(dora["tile_type"], str) and dora["tile_type"] in SHAPES, "dora_text: invalid tile")
            if "visual_note" in dora:
                _text(dora["visual_note"], "dora_text.visual_note")
        else:
            _text(slot["value"], f"context.{name}")
    unsupplied = _object(observation["unsupplied"], UNSUPPLIED_FIELDS, "unsupplied")
    _require(all(value == {"status": "unknown", "value": None} for value in unsupplied.values()),
             "unsupplied: absent context cannot be fabricated")
    red = _object(observation["red_notation"], {"status", "source_note", "visual_candidates"}, "red_notation")
    _require(red["status"] == "unresolved", "red_notation: cannot resolve notation here")
    _text(red["source_note"], "red_notation.source_note")
    _require(isinstance(red["visual_candidates"], list), "visual_candidates: expected list")
    positions = set()
    for glyph in red["visual_candidates"]:
        _object(glyph, {"question_row_position_one_based", "shape_tile", "observation"}, "visual_candidate")
        position = glyph["question_row_position_one_based"]
        _require(type(position) is int and 1 <= position <= 14 and position not in positions,
                 "visual_candidate: invalid or repeated position")
        positions.add(position)
        _require(glyph["shape_tile"] in ("5m", "5p", "5s") and hand[position - 1] == glyph["shape_tile"],
                 "visual_candidate: does not identify a matching five")
        _text(glyph["observation"], "visual_candidate.observation")
    answer = _object(record["answer"], {
        "source_transcription", "recommendation", "conditions", "count_conventions", "project_objective",
    }, "answer")
    transcription = answer["source_transcription"]
    _require(isinstance(transcription, dict), "source_transcription: expected object")
    for key in ("answer_text", "explanation_paraphrase", "status"):
        _text(transcription.get(key), f"source_transcription.{key}")
    _validate_recommendation(answer["recommendation"], hand, "recommendation")
    match = re.fullmatch(r"Discard ([1-9][mps]|[ESWNPFC])( and declare riichi)?\.", transcription["answer_text"])
    _require(match is not None and answer["recommendation"] == _recommendation(match[1], match[2] is not None),
             "recommendation: differs from source answer")
    _require(isinstance(answer["conditions"], list), "conditions: expected list")
    for condition in answer["conditions"]:
        _object(condition, {"kind", "condition_text", "condition_status", "source_basis",
                            "recommendation", "interpretation"}, "condition")
        _require(condition["kind"] in ("conditional_alternative", "recommendation_caveat"), "condition: invalid kind")
        _require(condition["condition_status"] == "not_evaluated", "condition: source condition cannot be asserted as observed")
        _require(condition["source_basis"] == "source_answer.explanation_paraphrase", "condition: invalid source basis")
        for key in ("condition_text", "interpretation"):
            _text(condition[key], f"condition.{key}")
        if condition["kind"] == "conditional_alternative":
            _validate_recommendation(condition["recommendation"], hand, "condition.recommendation")
        else:
            _require(condition["recommendation"] is None, "condition: caveat does not establish a replacement action")
    _require(answer["conditions"] == _conditions({
        "packet_id": record["id"], "source_answer": transcription,
    }), "conditions: required source caveat or alternative was changed")
    _require(isinstance(answer["count_conventions"], dict), "count_conventions: expected object")
    _require(answer["project_objective"] == {
        "name": "average_rank_utility", "values": [1.5, 0.5, -0.5, -1.5],
        "source_claims_calibrated_to_objective": False,
    }, "objective: source values cannot replace rank utility")
    review = _object(record["review"], {
        "source_missing_context", "source_rule_assumptions", "source_status", "arithmetic_reference",
        "arithmetic_scope", "strategic_label_status", "full_policy_ready", "training_eligible",
        "evaluation_eligible", "ineligibility_reasons",
    }, "review")
    for key in ("source_missing_context", "source_rule_assumptions", "ineligibility_reasons"):
        _texts(review[key], f"review.{key}")
    _require(review["ineligibility_reasons"] == INELIGIBILITY, "review: eligibility blockers cannot be removed")
    _require(review["strategic_label_status"] == "pending", "review: strategic label remains pending")
    for key in ("full_policy_ready", "training_eligible", "evaluation_eligible"):
        _require(review[key] is False, f"review.{key}: this schema is deliberately ineligible")
    _require(isinstance(review["source_status"], dict), "review.source_status: expected object")
    _text(review["arithmetic_scope"], "review.arithmetic_scope")
    if review["arithmetic_reference"] is not None:
        _reference(review["arithmetic_reference"], "arithmetic_reference")
    counts = transcription.get("reported_acceptance_counts")
    _require(isinstance(counts, list), "reported_acceptance_counts: expected list")
    count_tiles = set()
    for count in counts:
        _object(count, {"discard", "reported_count"}, "reported_count")
        _require(isinstance(count["discard"], str) and count["discard"] in hand and count["discard"] not in count_tiles,
                 "reported_count: invalid or repeated discard")
        count_tiles.add(count["discard"])
        _require(type(count["reported_count"]) is int and count["reported_count"] >= 0,
                 "reported_count: expected nonnegative integer")
    _require(bool(counts) == (review["arithmetic_reference"] is not None),
             "arithmetic_reference: only the audited count-bearing cases carry a prior check")


def _reference(reference: Any, where: str) -> None:
    ref = _object(reference, {"path", "sha256", "bytes"}, where)
    _text(ref["path"], f"{where}.path")
    _digest(ref["sha256"], f"{where}.sha256")
    _require(type(ref["bytes"]) is int and ref["bytes"] > 0, f"{where}.bytes: expected positive integer")


def partial_observation(record: dict) -> dict:
    """Export question-only partial data, never a model-ready player view.

    No IDs, provenance, source answer, rationale, conditional alternatives, count
    tables, numerical values or review labels are exposed. This is not accepted
    by the full-policy encoder and must not be augmented with guessed context.
    """
    validate_record(record)
    return {"schema": "riichi-ai-partial-teaching-question-v2",
            **copy.deepcopy(record["observation"])}


def validate_records(records: list[dict]) -> dict:
    errors = [] if records else ["collection: at least one record is required"]
    valid = []
    for index, record in enumerate(records):
        try:
            validate_record(record)
            valid.append(record)
        except (ValueError, KeyError, TypeError) as error:
            errors.append(f"record {index + 1}: {error}")
    ids = [record["id"] for record in valid]
    if len(ids) != len(set(ids)):
        errors.append("collection: duplicate IDs")
    books: dict[str, set[str]] = {}
    family_books: dict[str, set[str]] = {}
    for record in valid:
        grouping = record["grouping"]
        books.setdefault(record["provenance"]["source"]["book_id"], set()).add(grouping["whole_book_group"])
        family_books.setdefault(grouping["problem_family"], set()).add(grouping["whole_book_group"])
    if any(len(groups) != 1 for groups in (*books.values(), *family_books.values())):
        errors.append("collection: book or family assigned to inconsistent groups")
    return {
        "schema": "riichi-ai-partial-book-validation-v2",
        "valid": not errors, "errors": errors, "records": len(records),
        "books": len(books), "families": len(family_books),
        "training_eligible": False, "evaluation_eligible": False,
        "full_policy_ready": False,
        "scope": "Structural consistency only; no new arithmetic, strategic validation or policy evaluation.",
    }
