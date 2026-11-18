"""Conservative, stdlib-only intake for sourced discard exercises.

This is a metadata and consistency validator, not an engine adapter, proof of
optimality, rights determination, training loader, or replay implementation.
Answers and provenance never enter :func:`exercise_inputs`.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Any


SCHEMA = "riichi-ai-book-exercise-v1"
ROLES = {"quarantine", "train", "validation", "heldout"}
TILES = {f"{rank}{suit}" for suit in "mps" for rank in range(1, 10)} | {
    "5mr", "5pr", "5sr", "E", "S", "W", "N", "P", "F", "C",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: Any, keys: set[str], where: str) -> dict:
    _require(isinstance(value, dict), f"{where}: expected an object")
    _require(set(value) == keys, f"{where}: expected exactly {sorted(keys)}")
    return value


def _text(value: Any, where: str) -> None:
    _require(isinstance(value, str) and bool(value.strip()), f"{where}: expected nonempty text")


def _texts(value: Any, where: str, *, allow_empty: bool = False) -> None:
    _require(isinstance(value, list) and (allow_empty or bool(value)), f"{where}: expected a text list")
    for item in value:
        _text(item, where)


def _actions(value: Any, state: dict, where: str) -> None:
    _require(isinstance(value, list) and bool(value), f"{where}: expected a nonempty action set")
    seen = set()
    for action in value:
        _object(action, {"kind", "tile", "discard_origin"}, where)
        _require(action["kind"] == "DISCARD", f"{where}: version 1 supports discard exercises only")
        _require(isinstance(action["tile"], str) and action["tile"] in TILES, f"{where}: invalid tile")
        _require(action["discard_origin"] in ("any", "drawn", "held"), f"{where}: invalid discard_origin")
        key = (action["tile"], action["discard_origin"])
        _require(key not in seen, f"{where}: duplicate acceptable action")
        seen.add(key)
        if state["kind"] == "partial_exercise":
            observed = state["observed"]
            hand, drawn = observed["concealed_hand"], observed["drawn_tile"]
            _require(action["tile"] in hand, f"{where}: discard tile is absent from the supplied hand")
            if drawn is not None and action["discard_origin"] == "drawn":
                _require(action["tile"] == drawn, f"{where}: drawn discard differs from supplied drawn tile")
            if drawn == action["tile"] and action["discard_origin"] == "held":
                _require(hand.count(drawn) > 1, f"{where}: no held copy exists besides the drawn tile")


def validate_record(record: Any) -> None:
    """Raise ValueError on a malformed record; do not execute its references."""
    _object(record, {"schema", "id", "source", "permissions", "split", "objective", "rules",
                     "state", "source_answer", "label_review"}, "record")
    _require(record["schema"] == SCHEMA, "schema: unsupported version")
    _text(record["id"], "id")
    source = _object(record["source"], {"kind", "book_id", "title", "authors", "edition",
                                           "chapter", "page", "problem"}, "source")
    _require(source["kind"] in ("book", "project_authored"), "source.kind: unsupported source")
    for key in ("book_id", "title", "edition", "chapter", "page", "problem"):
        _text(source[key], f"source.{key}")
    _texts(source["authors"], "source.authors")
    permissions = _object(record["permissions"], {"basis", "local_training", "local_evaluation",
                                                     "redistribution"}, "permissions")
    _text(permissions["basis"], "permissions.basis")
    for key in ("local_training", "local_evaluation", "redistribution"):
        _require(type(permissions[key]) is bool, f"permissions.{key}: expected a boolean")
    split = _object(record["split"], {"role", "book_group", "chapter_group", "problem_family",
                                         "variant_of"}, "split")
    _require(isinstance(split["role"], str) and split["role"] in ROLES, "split.role: unsupported role")
    for key in ("book_group", "chapter_group", "problem_family"):
        _text(split[key], f"split.{key}")
    if split["variant_of"] is not None:
        _text(split["variant_of"], "split.variant_of")
        _require(split["variant_of"] != record["id"], "split.variant_of: cannot reference itself")
    if split["role"] == "train":
        _require(permissions["local_training"], "permissions: training role lacks recorded local-training permission")
    if split["role"] in ("validation", "heldout"):
        _require(permissions["local_evaluation"], "permissions: evaluation role lacks recorded local-evaluation permission")
    _text(record["objective"], "objective")
    rules = _object(record["rules"], {"profile", "assumptions"}, "rules")
    if rules["profile"] is not None:
        _text(rules["profile"], "rules.profile")
    _texts(rules["assumptions"], "rules.assumptions", allow_empty=True)
    state = record["state"]
    _require(isinstance(state, dict), "state: expected an object")
    if state.get("kind") == "partial_exercise":
        _object(state, {"kind", "observed", "missing_context"}, "state")
        _texts(state["missing_context"], "state.missing_context")
        observed = _object(state["observed"], {"concealed_hand", "drawn_tile"}, "state.observed")
        hand = observed["concealed_hand"]
        _require(isinstance(hand, list) and 1 <= len(hand) <= 14, "state.observed.concealed_hand: expected 1–14 tiles")
        _require(all(isinstance(tile, str) and tile in TILES for tile in hand), "state.observed.concealed_hand: invalid tile")
        counts = Counter(tile.removesuffix("r") for tile in hand)
        _require(max(counts.values()) <= 4, "state.observed.concealed_hand: more than four copies of a tile type")
        drawn = observed["drawn_tile"]
        _require(drawn is None or isinstance(drawn, str) and drawn in hand, "state.observed.drawn_tile: must be null or a supplied tile")
    elif state.get("kind") == "player_view_reference":
        _object(state, {"kind", "reference", "missing_context"}, "state")
        _require(state["missing_context"] == [], "state: complete-view candidates cannot omit known missing context")
        _require(rules["profile"] is not None, "rules.profile: complete-view candidates require an explicit rule profile")
        reference = _object(state["reference"], {"path", "sha256", "schema"}, "state.reference")
        for key in ("path", "schema"):
            _text(reference[key], f"state.reference.{key}")
        _require(isinstance(reference["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", reference["sha256"]),
                 "state.reference.sha256: expected a lowercase SHA-256 digest")
    else:
        raise ValueError("state.kind: expected partial_exercise or player_view_reference")
    answer = _object(record["source_answer"], {"acceptable_actions", "rationale", "claim"}, "source_answer")
    _actions(answer["acceptable_actions"], state, "source_answer.acceptable_actions")
    for key in ("rationale", "claim"):
        _text(answer[key], f"source_answer.{key}")
    review = _object(record["label_review"], {"status", "reviewer", "method", "evidence", "accepted_actions"}, "label_review")
    _require(review["status"] in ("pending", "supported", "disputed"), "label_review.status: unsupported status")
    if review["status"] == "pending":
        _require(all(review[key] is None for key in ("reviewer", "method", "evidence", "accepted_actions")),
                 "label_review: pending reviews must keep review-result fields null")
    else:
        for key in ("reviewer", "method", "evidence"):
            _text(review[key], f"label_review.{key}")
        _actions(review["accepted_actions"], state, "label_review.accepted_actions")
        if review["status"] == "supported":
            _require(_action_set(review["accepted_actions"]) == _action_set(answer["acceptable_actions"]),
                     "label_review: supported status must preserve the source's acceptable-action set")


def _action_set(actions: list[dict]) -> set[str]:
    return {json.dumps(action, sort_keys=True) for action in actions}


def exercise_inputs(record: dict) -> dict:
    """Return only explicitly supplied tiles for a partial-exercise evaluator.

    This cannot be fed to the full policy encoder. It contains no source label,
    explanation, provenance, missing-context assumptions, or inferred tiles.
    """
    validate_record(record)
    _require(record["state"]["kind"] == "partial_exercise", "Player-view references require a separately audited adapter")
    return {"schema": "riichi-ai-partial-exercise-input-v1", **copy.deepcopy(record["state"]["observed"])}


def _fingerprint(record: dict) -> str:
    state = copy.deepcopy(record["state"])
    if state["kind"] == "partial_exercise":
        state["observed"]["concealed_hand"].sort()
        state["missing_context"].sort()
    else:
        state["reference"].pop("path")
    rules = copy.deepcopy(record["rules"])
    rules["assumptions"].sort()
    value = {"state": state, "objective": record["objective"], "rules": rules}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_records(records: list[dict]) -> dict:
    """Check intake consistency and conservative whole-book split isolation.

    Related editions/translations must share manually assigned groups. This
    cannot discover every semantic duplicate or verify source assertions.
    """
    errors = [] if records else ["collection: at least one record is required"]
    valid_records = []
    for index, record in enumerate(records, 1):
        try:
            validate_record(record)
            valid_records.append(record)
        except ValueError as error:
            errors.append(f"record {index}: {error}")
    by_id = {}
    groups: dict[tuple[str, str], set[str]] = {}
    book_groups: dict[str, set[str]] = {}
    chapter_parents: dict[str, set[str]] = {}
    for record in valid_records:
        identifier, split = record["id"], record["split"]
        if identifier in by_id:
            errors.append(f"duplicate record id: {identifier}")
        by_id[identifier] = record
        book_groups.setdefault(record["source"]["book_id"], set()).add(split["book_group"])
        chapter_parents.setdefault(split["chapter_group"], set()).add(split["book_group"])
        if split["role"] != "quarantine":
            for name in ("book_group", "chapter_group", "problem_family"):
                groups.setdefault((name, split[name]), set()).add(split["role"])
            groups.setdefault(("exact_content", _fingerprint(record)), set()).add(split["role"])
    for (name, group), roles in groups.items():
        if len(roles) > 1:
            errors.append(f"split leakage: {name} {group!r} spans {sorted(roles)}")
    for book, names in book_groups.items():
        if len(names) > 1:
            errors.append(f"book {book!r} maps to multiple book groups")
    for chapter, parents in chapter_parents.items():
        if len(parents) > 1:
            errors.append(f"chapter group {chapter!r} maps to multiple book groups")
    for record in valid_records:
        parent = record["split"]["variant_of"]
        if parent is None:
            continue
        if parent not in by_id:
            errors.append(f"{record['id']}: variant parent {parent!r} is missing from this collection")
        elif by_id[parent]["split"]["problem_family"] != record["split"]["problem_family"]:
            errors.append(f"{record['id']}: variants must share their parent's problem family")
        visited = {record["id"]}
        while parent in by_id:
            if parent in visited:
                errors.append(f"{record['id']}: variant references form a cycle")
                break
            visited.add(parent)
            parent = by_id[parent]["split"]["variant_of"]
    return {
        "schema": "riichi-ai-book-intake-report-v1", "valid": not errors,
        "records": len(records), "errors": errors,
        "roles": dict(sorted(Counter(record["split"]["role"] for record in valid_records).items())),
        "state_kinds": dict(sorted(Counter(record["state"]["kind"] for record in valid_records).items())),
        "policy_training_ready": False,
        "warnings": [
            "Structural validation only: source answers, permission statements, and review evidence are not independently verified.",
            "Partial exercises are not complete policy states; player-view references have not been opened or replayed.",
            "No training, evaluation, copying of source material, or redistribution is performed.",
            "Whole-book groups and problem families require manual review for translations, reprints, paraphrases, and variants.",
        ],
    }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON constant: {value}")


def load_jsonl(path: str | Path) -> list[dict]:
    """Load explicit JSON records only; never follow state or evidence paths."""
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line, object_pairs_hook=_unique_object, parse_constant=_reject_constant))
            except ValueError as error:
                raise ValueError(f"line {line_number}: {error}") from error
    _require(bool(records), "input: expected at least one JSONL record")
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    arguments = parser.parse_args(argv)
    try:
        report = validate_records(load_jsonl(arguments.path))
    except (ValueError, OSError) as error:
        report = {"valid": False, "errors": [str(error)], "policy_training_ready": False}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
