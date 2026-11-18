#!/usr/bin/env python3
"""Compare twelve printed Uzaku counts under two explicit known-copy conventions.

Run only under the caller's numerical lease. Printed-count agreement is an
arithmetic finding, not strategic label validation or policy evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import inspect
import json
from pathlib import Path
import platform
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from riichi_ai.book_scenarios import TILES, load_jsonl
from riichi_ai.policies import _discard_score
from mahjong.shanten import Shanten

SCHEMA = "riichi-ai-book-count-conventions-v1"
PACKET_IDS = ("uzaku-300-q001", "uzaku-300-q002", "uzaku-301-q001", "uzaku-301-q002")
HONORS = ("E", "S", "W", "N", "P", "F", "C")
CONVENTIONS = {
    "book_post_discard_13": {
        "known_tiles": "Only the thirteen tiles retained after the discard.",
        "discarded_tile_stays_known": False, "shown_indicator_counted": False,
        "interpretation": "The source's declared hypothetical counting convention; not physical visible-copy availability.",
    },
    "known_hand_14_plus_indicator": {
        "known_tiles": "All fourteen supplied hand tiles plus the single shown indicator; discard stays known.",
        "discarded_tile_stays_known": True, "shown_indicator_counted": True,
        "interpretation": "Unseen-copy upper bound from supplied hand and indicator only; unshown public tiles may reduce it further.",
    },
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tile_type(tile):
    if not isinstance(tile, str) or tile not in TILES:
        raise ValueError("Invalid explicit tile label")
    if tile in HONORS:
        return 27 + HONORS.index(tile)
    value = tile.removesuffix("r")
    return 9 * "mps".index(value[1]) + int(value[0]) - 1


def tile_name(kind):
    return HONORS[kind - 27] if kind >= 27 else f"{kind % 9 + 1}{'mps'[kind // 9]}"


def hand_counts(hand):
    if not isinstance(hand, list) or len(hand) != 14:
        raise ValueError("Expected fourteen explicit concealed tiles")
    result = [0] * 34
    for tile in hand:
        result[tile_type(tile)] += 1
    if max(result) > 4:
        raise ValueError("More than four copies of a hand tile type")
    return result


def known_counts(before, discard, indicator, convention):
    if (len(before) != 34 or sum(before) != 14
            or any(type(value) is not int or not 0 <= value <= 4 for value in before)
            or type(discard) is not int or not 0 <= discard < 34 or before[discard] < 1
            or type(indicator) is not int or not 0 <= indicator < 34):
        raise ValueError("Invalid hand/discard/indicator counts")
    physical = list(before)
    physical[indicator] += 1
    if max(physical) > 4:
        raise ValueError("Shown indicator plus hand exceeds four physical copies")
    if convention == "book_post_discard_13":
        result = list(before)
        result[discard] -= 1
        return result
    if convention == "known_hand_14_plus_indicator":
        return physical
    raise ValueError("Unknown counting convention")


def _format_score(shanten, improving, known):
    improving = list(improving)
    copies = [4 - known[kind] for kind in improving]
    return {"shanten": int(shanten), "improving_type_ids": improving,
            "improving_types": [tile_name(kind) for kind in improving],
            "copies_by_improving_type": [
                {"tile_type": tile_name(kind), "known_copies": known[kind], "counted_copies": count}
                for kind, count in zip(improving, copies)], "counted_ukeire": sum(copies)}


def score_both(before, discard, indicator, convention):
    """Apply identical shape and declared counting assumptions to both engines."""
    known = known_counts(before, discard, indicator, convention)
    # This raw helper accepts the explicitly artificial source convention. The
    # physical-only guards in book_efficiency_probe are intentionally untouched.
    project_raw = _discard_score(tuple(before), tuple(known), discard)
    project = _format_score(project_raw.shanten, project_raw.improving_types, known)
    if project["counted_ukeire"] != project_raw.ukeire:
        raise RuntimeError("Project per-type counts disagree with its aggregate")
    retained = list(before)
    retained[discard] -= 1
    baseline = Shanten.calculate_shanten(retained, use_chiitoitsu=True, use_kokushi=True)
    improving = []
    for kind in range(34):
        if known[kind] >= 4:
            continue
        retained[kind] += 1
        better = Shanten.calculate_shanten(retained, use_chiitoitsu=True, use_kokushi=True) < baseline
        retained[kind] -= 1
        if better:
            improving.append(kind)
    independent = _format_score(baseline, improving, known)
    return {"known_counts": known, "project": project, "independent": independent,
            "calculators_agree": project == independent}


def select_packets(records):
    packets = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Expected raw-source packet objects")
        identifier = record.get("packet_id")
        if identifier not in PACKET_IDS:
            continue
        if identifier in packets:
            raise ValueError("Duplicate selected packet identity")
        if record.get("schema") != "raw-source-packet-v1":
            raise ValueError("Selected record has unsupported raw packet schema")
        source = record["source"]
        if (source["book_id"] != identifier.rsplit("-q", 1)[0]
                or not isinstance(source["path"], str) or not source["path"]
                or not isinstance(source["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", source["sha256"])):
            raise ValueError("Invalid source book identity or digest")
        status = record["status"]
        if (status["role"] != "quarantine" or status["transcription_only"] is not True
                or status["model_ready"] is not False or status["strategic_label_review"] != "pending"):
            raise ValueError("Selected source packet must remain quarantined, transcription-only and strategically pending")
        fields = record["source_fields"]
        hand_counts(fields["shown_hand_including_draw"])
        before, drawn = fields["concealed_hand_before_draw"], fields["drawn_tile"]
        if (not isinstance(before, list) or len(before) != 13
                or not all(isinstance(tile, str) and tile in TILES for tile in before)
                or not isinstance(drawn, str) or drawn not in TILES
                or Counter(before + [drawn]) != Counter(fields["shown_hand_including_draw"])):
            raise ValueError("Question hand and separately shown drawn tile disagree")
        dora = fields["dora_text"]
        if not isinstance(dora, dict) or dora.get("label") != "dora indicator":
            raise ValueError("Expected an explicitly identified dora indicator, not a dora tile")
        tile_type(dora["tile_type"])
        comparisons = record["source_answer"]["reported_acceptance_counts"]
        if not isinstance(comparisons, list) or len(comparisons) != 3:
            raise ValueError("Each selected packet must retain all three printed candidate counts")
        seen = set()
        for item in comparisons:
            if (not isinstance(item, dict) or set(item) != {"discard", "reported_count"}
                    or item["discard"] not in fields["shown_hand_including_draw"]
                    or type(item["reported_count"]) is not int or item["reported_count"] < 0
                    or item["discard"] in seen):
                raise ValueError("Invalid or duplicate printed candidate count")
            seen.add(item["discard"])
        packets[identifier] = record
    if set(packets) != set(PACKET_IDS):
        raise ValueError("All four prespecified Uzaku packets are required")
    for left, right in ((PACKET_IDS[0], PACKET_IDS[1]), (PACKET_IDS[2], PACKET_IDS[3])):
        if any(packets[left]["source"][key] != packets[right]["source"][key] for key in ("book_id", "path", "sha256")):
            raise ValueError("Packets from the same book disagree on source identity")
    return [packets[identifier] for identifier in PACKET_IDS]


def compare_packet(record):
    fields = record["source_fields"]
    before = hand_counts(fields["shown_hand_including_draw"])
    indicator = tile_type(fields["dora_text"]["tile_type"])
    comparisons = []
    for item in record["source_answer"]["reported_acceptance_counts"]:
        discard = tile_type(item["discard"])
        conventions = {name: score_both(before, discard, indicator, name) for name in CONVENTIONS}
        printed = item["reported_count"]
        book = conventions["book_post_discard_13"]
        physical = conventions["known_hand_14_plus_indicator"]
        comparisons.append({"discard_tile": item["discard"], "reported_count": printed,
            "conventions": conventions,
            "printed_vs_book_convention": {
                "project_matches": book["project"]["counted_ukeire"] == printed,
                "independent_matches": book["independent"]["counted_ukeire"] == printed,
                "project_minus_printed": book["project"]["counted_ukeire"] - printed,
                "independent_minus_printed": book["independent"]["counted_ukeire"] - printed},
            "book_minus_known_hand_plus_indicator": {
                engine: book[engine]["counted_ukeire"] - physical[engine]["counted_ukeire"]
                for engine in ("project", "independent")}})
    return {"packet_id": record["packet_id"], "source": deepcopy(record["source"]),
        "source_packet_content_sha256": hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "source_status": deepcopy(record["status"]),
        "supplied_hand_including_draw": fields["shown_hand_including_draw"],
        "shown_indicator": fields["dora_text"]["tile_type"],
        "comparisons": comparisons, "comparison_count": len(comparisons),
        "all_calculators_agree": all(value["calculators_agree"] for row in comparisons for value in row["conventions"].values()),
        "all_printed_counts_match_book_convention": all(
            row["printed_vs_book_convention"]["project_matches"] and row["printed_vs_book_convention"]["independent_matches"]
            for row in comparisons),
        "strategic_label_review": "pending; unchanged by count comparison"}


def audit_file(input_path):
    started = time.perf_counter()
    input_path = Path(input_path)
    initial_sha = sha256(input_path)
    selected = select_packets(load_jsonl(input_path))
    packets = [compare_packet(packet) for packet in selected]
    if sha256(input_path) != initial_sha:
        raise RuntimeError("Input packets changed during the comparison")
    files = [Path(__file__), ROOT / "src/riichi_ai/policies.py",
             ROOT / "src/riichi_ai/book_scenarios.py", Path(inspect.getfile(Shanten))]
    comparisons = [row for packet in packets for row in packet["comparisons"]]
    return {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": {"path": str(input_path.resolve()), "sha256": initial_sha,
                  "bytes": input_path.stat().st_size},
        "selected_packet_ids": list(PACKET_IDS), "selection": "Four prespecified packet IDs, all three printed comparisons per packet; no result-dependent filtering.",
        "source_pdf_digests": [{"book_id": selected[index]["source"]["book_id"],
            "path": selected[index]["source"]["path"], "sha256": selected[index]["source"]["sha256"],
            "verification": "Attributed source identity from packet; PDF not reopened by this arithmetic probe."} for index in (0, 2)],
        "runtime": {"python": platform.python_version(), "riichienv": metadata.version("riichienv"),
                    "mahjong": metadata.version("mahjong")},
        "implementation_files": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in files],
        "conventions": deepcopy(CONVENTIONS), "packets": packets,
        "comparison_count": len(comparisons), "engine_convention_comparisons": 2 * len(comparisons),
        "all_calculators_agree": all(packet["all_calculators_agree"] for packet in packets),
        "all_printed_counts_match_book_convention": all(packet["all_printed_counts_match_book_convention"] for packet in packets),
        "printed_count_mismatches": [{"packet_id": packet["packet_id"], "discard_tile": row["discard_tile"],
            "comparison": row["printed_vs_book_convention"]} for packet in packets for row in packet["comparisons"]
            if not (row["printed_vs_book_convention"]["project_matches"] and row["printed_vs_book_convention"]["independent_matches"])],
        "elapsed_seconds": time.perf_counter() - started,
        "model_evaluations": 0, "optimizer_updates": 0, "new_matches": 0,
        "source_labels_modified": False, "policy_training_ready": False,
        "limitations": [
            "Each fourteen-tile row is conditionally treated as a complete closed hand; no declared melds are modelled.",
            "Both calculators use minimum regular/seven-pairs/thirteen-orphans shanten; source pattern conventions can differ.",
            "Printed counts are compared only with the source's declared post-discard-hand-only convention.",
            "The second convention counts only the shown hand and indicator, not unknown rivers/melds or actual remaining wall tiles.",
            "This does not independently validate transcription, hand scoring, wait quality, yaku, furiten, defense, legal-action availability or match strategy.",
            "Source recommendations remain attributed, strategically pending and quarantined. No learned policy is evaluated.",
        ], "scope": "Conditional arithmetic comparison of all twelve printed options, not strategic answer validation."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
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
        "all_printed_counts_match_book_convention": report["all_printed_counts_match_book_convention"]}))
    # A printed-count mismatch is a retained scientific observation, not a
    # failed execution. An inter-engine disagreement merits explicit attention.
    return 0 if report["all_calculators_agree"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
