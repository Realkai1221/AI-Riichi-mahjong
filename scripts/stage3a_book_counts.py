#!/usr/bin/env python3
"""Audit all 47 printed options in the ten frozen Stage 3A Uzaku packets.

Execute only under the caller's numerical lease. Importing this module does not
load either calculator; exact input/source/helper identities are checked first.
Printed-count mismatches are retained observations, not execution failures.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib import metadata, util
import inspect
import json
from pathlib import Path
import platform
import re
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "Books/stage3a/source-packets.jsonl"
HELPER_PATH = ROOT / "scripts/book_count_conventions.py"
INPUT_SHA256 = "90234c0d343d5d2219ed5270e0cd886db3270c34893f6298725225fd85addf4a"
INPUT_BYTES = 59246
HELPER_SHA256 = "7e3e28d246f56377a10d7cb3d0410cc427f3ff820372a46ce1c662c7eb3ffd6d"
HELPER_BYTES = 15336
PACKET_OPTION_COUNTS = {
    "uzaku-301-q004": 5, "uzaku-301-q005": 5, "uzaku-301-q006": 5,
    "uzaku-301-q007": 6, "uzaku-301-q008": 6, "uzaku-301-q009": 6,
    "uzaku-300-q004": 4, "uzaku-300-q005": 4,
    "uzaku-300-q006": 3, "uzaku-300-q007": 3,
}
PACKET_IDS = tuple(PACKET_OPTION_COUNTS)
SOURCE_PINS = {
    "uzaku-301": {"sha256": "f89f6aec03c35e4d3adbadfe7eb74db9efa9727888bed19cff6feacc57cd3e94", "bytes": 36392855},
    "uzaku-300": {"sha256": "cb8e6ab80eacd53715e5f485f1f54e29064420afd0dc69c6a878cc9038443b3e", "bytes": 36680184},
}
TILES = {f"{rank}{suit}" for rank in range(1, 10) for suit in "mps"} | set("ESWNPFC")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path):
    path = Path(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON object key")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Nonstandard JSON constant: {value}")


def verify_input(path):
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    require(digest == INPUT_SHA256 and len(raw) == INPUT_BYTES,
            "Frozen Stage 3A input digest or byte count changed")
    records = [json.loads(line, object_pairs_hook=_object, parse_constant=_invalid_constant)
               for line in raw.decode("utf-8").splitlines() if line.strip()]
    return records, {"path": str(path.resolve()), "sha256": digest, "bytes": len(raw)}


def select_packets(records):
    require(isinstance(records, list), "Expected a list of raw packets")
    selected = {}
    for record in records:
        require(isinstance(record, dict), "Expected raw packet objects")
        identifier = record.get("packet_id")
        require(isinstance(identifier, str), "Expected a string packet identity")
        if identifier not in PACKET_IDS:
            continue
        require(identifier not in selected, "Duplicate selected packet identity")
        require(record.get("schema") == "raw-source-packet-v1", "Unsupported raw packet schema")
        source = record.get("source", {})
        book_id = identifier.rsplit("-q", 1)[0]
        require(isinstance(source, dict) and source.get("book_id") == book_id
                and source.get("sha256") == SOURCE_PINS[book_id]["sha256"]
                and isinstance(source.get("path"), str) and source["path"],
                "Changed or invalid source digest/identity")
        status = record.get("status", {})
        require(isinstance(status, dict) and status.get("role") == "quarantine"
                and status.get("transcription_only") is True and status.get("model_ready") is False
                and status.get("training_performed") is False
                and status.get("model_evaluation_performed") is False
                and status.get("strategic_label_review") == "pending",
                "Selected packets must remain quarantine with pending strategic labels")
        grouping = record.get("grouping", {})
        require(isinstance(grouping, dict) and grouping.get("intended_use") == "development"
                and grouping.get("holdout_eligible") is False,
                "Selected packets must remain development material")
        evidence = record.get("evidence", {})
        review = evidence.get("independent_review", {}) if isinstance(evidence, dict) else {}
        require(isinstance(review, dict) and review.get("status") == "passed_transcription"
                and review.get("reviewer") == "/root", "Missing recorded transcription review")
        fields = record.get("source_fields", {})
        require(isinstance(fields, dict), "Missing source fields")
        hand, before, drawn = (fields.get(key) for key in
                              ("shown_hand_including_draw", "concealed_hand_before_draw", "drawn_tile"))
        require(isinstance(hand, list) and len(hand) == 14
                and all(isinstance(tile, str) and tile in TILES for tile in hand)
                and max(Counter(hand).values()) <= 4, "Invalid fourteen-tile shape row")
        require(isinstance(before, list) and len(before) == 13
                and all(isinstance(tile, str) and tile in TILES for tile in before)
                and isinstance(drawn, str) and drawn in TILES
                and Counter(before + [drawn]) == Counter(hand), "Question draw does not match hand")
        indicator = fields.get("dora_text", {})
        require(isinstance(indicator, dict) and indicator.get("label") == "dora indicator"
                and isinstance(indicator.get("tile_type"), str) and indicator["tile_type"] in TILES,
                "Expected an explicit dora indicator")
        require(Counter(hand)[indicator["tile_type"]] < 4, "Indicator exceeds four physical copies")
        answer = record.get("source_answer", {})
        comparisons = answer.get("reported_acceptance_counts") if isinstance(answer, dict) else None
        require(isinstance(comparisons, list) and len(comparisons) == PACKET_OPTION_COUNTS[identifier],
                "All prespecified printed options are required")
        seen = set()
        for item in comparisons:
            require(isinstance(item, dict) and set(item) == {"discard", "reported_count"}
                    and isinstance(item["discard"], str) and item["discard"] in hand
                    and item["discard"] not in seen and type(item["reported_count"]) is int
                    and item["reported_count"] >= 0, "Invalid or duplicate printed option")
            seen.add(item["discard"])
        selected[identifier] = record
    require(set(selected) == set(PACKET_IDS), "All ten prespecified Uzaku packets are required")
    for book_id in SOURCE_PINS:
        sources = [record["source"] for record in selected.values() if record["source"]["book_id"] == book_id]
        require(len({source["path"] for source in sources}) == 1, "Same-book source paths disagree")
    return [selected[identifier] for identifier in PACKET_IDS]


def verify_sources(selected):
    verified = []
    for book_id, pin in SOURCE_PINS.items():
        source = next(record["source"] for record in selected if record["source"]["book_id"] == book_id)
        relative = Path(source["path"])
        require(not relative.is_absolute() and (ROOT / relative).resolve().is_relative_to(ROOT.resolve()),
                "Source path must remain inside the project")
        actual = identity(ROOT / relative)
        require(actual["sha256"] == pin["sha256"] and actual["bytes"] == pin["bytes"],
                "Source PDF digest or byte count changed")
        verified.append({"book_id": book_id, **actual, "verification": "Full local PDF bytes hashed before comparison"})
    return verified


def verify_helper():
    actual = identity(HELPER_PATH)
    require(actual["sha256"] == HELPER_SHA256 and actual["bytes"] == HELPER_BYTES,
            "Arithmetic helper digest or byte count changed")
    return actual


def load_helper():
    # This is the first numerical import, after all frozen-input checks.
    specification = util.spec_from_file_location("stage3a_existing_count_helper", HELPER_PATH)
    module = util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def findings(packets):
    comparisons = [row for packet in packets for row in packet["comparisons"]]
    return {
        "comparison_count": len(comparisons),
        "engine_convention_comparisons": sum(len(row["conventions"]) for row in comparisons),
        "all_calculators_agree": all(value["calculators_agree"] for row in comparisons for value in row["conventions"].values()),
        "all_printed_counts_match_book_convention": all(
            row["printed_vs_book_convention"]["project_matches"] and row["printed_vs_book_convention"]["independent_matches"]
            for row in comparisons),
        "printed_count_mismatches": [
            {"packet_id": packet["packet_id"], "discard_tile": row["discard_tile"],
             "comparison": deepcopy(row["printed_vs_book_convention"])}
            for packet in packets for row in packet["comparisons"]
            if not (row["printed_vs_book_convention"]["project_matches"] and row["printed_vs_book_convention"]["independent_matches"])],
        "calculator_disagreements": [
            {"packet_id": packet["packet_id"], "discard_tile": row["discard_tile"], "convention": convention}
            for packet in packets for row in packet["comparisons"]
            for convention, value in row["conventions"].items() if not value["calculators_agree"]],
        "count_differences_between_conventions": [
            {"packet_id": packet["packet_id"], "discard_tile": row["discard_tile"],
             "book_minus_known_hand_plus_indicator": deepcopy(row["book_minus_known_hand_plus_indicator"])}
            for packet in packets for row in packet["comparisons"]
            if any(row["book_minus_known_hand_plus_indicator"].values())],
    }


def audit_file(input_path=DEFAULT_INPUT):
    started = time.perf_counter()
    records, input_identity = verify_input(input_path)
    selected = select_packets(records)
    source_identities = verify_sources(selected)
    helper_identity = verify_helper()
    helper = load_helper()
    code_paths = [Path(__file__), ROOT / "src/riichi_ai/policies.py",
                  ROOT / "src/riichi_ai/book_scenarios.py", Path(inspect.getfile(helper.Shanten))]
    code_identities = [identity(path) for path in code_paths] + [helper_identity]
    packets = []
    for record in selected:
        # The unchanged helper's compare_packet uses hand_counts, tile_type,
        # and score_both for every option under both conventions. Do not fork
        # the already tested shape/counting algorithms here.
        result = helper.compare_packet(record)
        require([(row["discard_tile"], row["reported_count"]) for row in result["comparisons"]]
                == [(row["discard"], row["reported_count"]) for row in record["source_answer"]["reported_acceptance_counts"]],
                "Comparison implementation dropped or changed a source option")
        result["source_fields"] = deepcopy(record["source_fields"])
        result["source_independent_review"] = deepcopy(record["evidence"]["independent_review"])
        result["source_grouping"] = deepcopy(record["grouping"])
        packets.append(result)
    for original in [input_identity, *source_identities, *code_identities]:
        current = identity(original["path"])
        require(current["sha256"] == original["sha256"] and current["bytes"] == original["bytes"],
                "Input, source PDF or implementation changed during comparison")
    summary = findings(packets)
    require(summary["comparison_count"] == 47 and summary["engine_convention_comparisons"] == 94,
            "Expected all 47 printed options under both conventions")
    return {
        "schema": "riichi-ai-stage3a-book-counts-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": input_identity, "source_pdfs": source_identities, "implementation_files": code_identities,
        "selected_packet_ids": list(PACKET_IDS), "selection": "Ten frozen Uzaku packets; all 47 printed options; no result-dependent filtering.",
        "runtime": {"python": platform.python_version(), "riichienv": metadata.version("riichienv"), "mahjong": metadata.version("mahjong")},
        "conventions": deepcopy(helper.CONVENTIONS), "packets": packets, **summary,
        "elapsed_seconds": time.perf_counter() - started,
        "model_evaluations": 0, "optimizer_updates": 0, "new_matches": 0,
        "source_labels_modified": False, "policy_training_ready": False,
        "limitations": [
            "Each supplied row is conditionally treated as a closed fourteen-tile shape; no declared melds are modelled.",
            "Physical red-five identities remain unresolved. This shape-only arithmetic does not score red dora or select between red and ordinary fives.",
            "Both calculators use minimum regular/seven-pairs/thirteen-orphans shanten; the source may use additional pattern or improvement conventions.",
            "The second convention includes only the supplied hand and indicator; unshown public tiles and live-wall availability remain unknown.",
            "Agreement or disagreement does not certify wait quality, hand value, defence, legal actions, or the source's strategic recommendation.",
            "All packets remain development/quarantine with strategic labels pending. No learned policy is evaluated.",
        ],
        "scope": "Conditional arithmetic audit of all 47 printed options, retaining every mismatch and calculator disagreement.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Comparison output already exists; no silent replacement")
    report = audit_file(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"report": str(args.output), "comparisons": report["comparison_count"],
                      "all_calculators_agree": report["all_calculators_agree"],
                      "printed_count_mismatches": len(report["printed_count_mismatches"]),
                      "count_differences_between_conventions": len(report["count_differences_between_conventions"])}))
    return 0 if report["all_calculators_agree"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
