#!/usr/bin/env python3
"""Conditional closed-hand arithmetic audit; run within a numerical lease.

This compares two shanten calculators on quarantined partial exercises. It does
not create a PlayerView, evaluate a neural policy, validate a strategic label,
produce training rows, or infer unshown public information.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import inspect
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from riichi_ai.book_scenarios import load_jsonl, validate_records
from riichi_ai.policies import _discard_score
from mahjong.shanten import Shanten

SCHEMA = "riichi-ai-book-conditional-efficiency-probe-v1"
CLOSED_HAND_ASSERTIONS = {
    "Closed hand with no declared melds.",
    "The displayed fourteen-tile row is treated as the complete concealed hand for this partial exercise; no meld is shown.",
}
HONORS = ("E", "S", "W", "N", "P", "F", "C")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tile_type(tile):
    if tile in HONORS:
        return 27 + HONORS.index(tile)
    value = tile.removesuffix("r")
    return 9 * "mps".index(value[1]) + int(value[0]) - 1


def tile_name(kind):
    return HONORS[kind - 27] if kind >= 27 else f"{kind % 9 + 1}{'mps'[kind // 9]}"


def _valid_counts(counts, visible, discard):
    if (len(counts) != 34 or len(visible) != 34 or sum(counts) != 14
            or any(type(c) is not int or not 0 <= c <= 4 for c in counts)
            or any(type(v) is not int or not c <= v <= 4 for c, v in zip(counts, visible))
            or type(discard) is not int or not 0 <= discard < 34 or counts[discard] < 1):
        raise ValueError("Expected a closed fourteen-tile hand and consistent known-copy counts")


def independent_discard_score(counts, visible, discard):
    """All three closed-hand patterns, independent of the project's calculator.

    ``visible`` deliberately remains unchanged after discarding: the discarded
    copy stays known. Improving copies are unseen-copy bounds, not live-wall EV.
    """
    _valid_counts(counts, visible, discard)
    hand = list(counts)
    hand[discard] -= 1
    baseline = Shanten.calculate_shanten(hand, use_chiitoitsu=True, use_kokushi=True)
    improving, availability = [], []
    for kind in range(34):
        available = 4 - visible[kind]
        if available <= 0:
            continue
        hand[kind] += 1
        improved = Shanten.calculate_shanten(hand, use_chiitoitsu=True, use_kokushi=True) < baseline
        hand[kind] -= 1
        if improved:
            improving.append(kind)
            availability.append(available)
    return {"shanten": int(baseline), "improving_type_ids": improving,
            "improving_types": [tile_name(k) for k in improving],
            "unseen_copies_per_improving_type": availability,
            "ukeire_unseen_copy_upper_bound": sum(availability)}


def project_discard_score(counts, visible, discard):
    _valid_counts(counts, visible, discard)
    score = _discard_score(tuple(counts), tuple(visible), discard)
    improving = list(score.improving_types)
    return {"shanten": int(score.shanten), "improving_type_ids": improving,
            "improving_types": [tile_name(k) for k in improving],
            "unseen_copies_per_improving_type": [4 - visible[k] for k in improving],
            "ukeire_unseen_copy_upper_bound": int(score.ukeire)}


def _best_sets(candidates, calculator):
    best_shanten = min(row[calculator]["shanten"] for row in candidates)
    minimum = [row for row in candidates if row[calculator]["shanten"] == best_shanten]
    best_ukeire = max(row[calculator]["ukeire_unseen_copy_upper_bound"] for row in minimum)
    return {"minimum_shanten": best_shanten,
            "minimum_shanten_discard_tiles": [row["discard_tile"] for row in minimum],
            "maximum_ukeire_among_minimum_shanten": best_ukeire,
            "coarse_best_discard_tiles": [row["discard_tile"] for row in minimum
                if row[calculator]["ukeire_unseen_copy_upper_bound"] == best_ukeire]}


def audit_record(record):
    # Collection-only variant checks are performed by audit_file.
    from riichi_ai.book_scenarios import validate_record
    validate_record(record)
    if record["split"]["role"] != "quarantine" or record["label_review"]["status"] != "pending":
        raise ValueError("This probe accepts quarantined, strategically unreviewed source records only")
    state = record["state"]
    if state["kind"] != "partial_exercise" or len(state["observed"]["concealed_hand"]) != 14:
        raise ValueError("Only partial exercises with exactly fourteen concealed tiles are supported")
    if not CLOSED_HAND_ASSERTIONS.intersection(record["rules"]["assumptions"]):
        raise ValueError("A supplied complete-concealed/no-meld assertion is required; open hands are unsupported")
    hand = state["observed"]["concealed_hand"]
    counts = [0] * 34
    for tile in hand:
        counts[tile_type(tile)] += 1
    # This is a declared conditional bound. No unshown public tile is invented.
    visible = list(counts)
    candidates = []
    for tile in sorted(set(hand), key=lambda value: (tile_type(value), value.endswith("r"))):
        kind = tile_type(tile)
        project = project_discard_score(counts, visible, kind)
        independent = independent_discard_score(counts, visible, kind)
        candidates.append({"discard_tile": tile, "discard_type_id": kind,
            "known_copy_counts_unchanged_after_discard": True,
            "project": project, "independent": independent,
            "calculators_agree": project == independent})
    project_sets, independent_sets = (_best_sets(candidates, calculator)
                                     for calculator in ("project", "independent"))
    memberships = [{"source_action": deepcopy(action),
        "in_project_minimum_shanten_set": action["tile"] in project_sets["minimum_shanten_discard_tiles"],
        "in_project_coarse_best_set": action["tile"] in project_sets["coarse_best_discard_tiles"],
        "in_independent_minimum_shanten_set": action["tile"] in independent_sets["minimum_shanten_discard_tiles"],
        "in_independent_coarse_best_set": action["tile"] in independent_sets["coarse_best_discard_tiles"]}
        for action in record["source_answer"]["acceptable_actions"]]
    return {"id": record["id"], "source": deepcopy(record["source"]),
        "source_objective": record["objective"],
        "input_role": record["split"]["role"], "input_label_review": record["label_review"]["status"],
        "input_permissions": deepcopy(record["permissions"]),
        "supplied_hand": list(hand), "supplied_drawn_tile": state["observed"]["drawn_tile"],
        "known_copy_counts_before_and_after_discard": counts,
        "conditional_assumptions": [
            "Closed fourteen-tile hand with no declared melds, conditional on the supplied row interpretation.",
            "Only supplied own-hand tiles are counted as known; unknown public tiles are not asserted absent.",
            "The discarded tile remains known, so it cannot be counted as an available new copy.",
            "Ukeire is an unseen-copy upper bound under this information restriction, not a live-wall count or winning probability.",
            "Regular hands, seven pairs, and thirteen orphans are all eligible for shape shanten; this may differ from a source's objective.",
        ],
        "candidates": candidates, "project_best_sets": project_sets,
        "independent_best_sets": independent_sets,
        "source_answer_membership": memberships,
        "all_calculators_agree": all(row["calculators_agree"] for row in candidates),
        "strategic_label_review": "pending; unchanged by this conditional arithmetic check",
        "policy_training_ready": False,
        "limitations": [
            "Candidate rows are shape-level discard tiles, not certified legal semantic actions; drawn/held identity is not resolved.",
            "Red and ordinary fives retain distinct labels but share shape scores; no red-value preference is scored.",
            "No defense, yaku legality, furiten, score, rank utility, future wait quality, or multi-step value is evaluated.",
            "Set membership does not validate the source's full strategic recommendation or promote the project heuristic.",
        ]}


def provenance(input_path, records):
    input_path = Path(input_path)
    source = {"input_jsonl": {"path": str(input_path.resolve()), "sha256": sha256(input_path),
                               "bytes": input_path.stat().st_size, "records": len(records)},
              "pilot_manifest": None, "book_pdf_declared_by_manifest": None}
    manifest_path = input_path.parent / "source-manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest["pilot"]["sha256"] != source["input_jsonl"]["sha256"]
                or manifest["pilot"]["bytes"] != source["input_jsonl"]["bytes"]
                or manifest["pilot"]["records"] != len(records)):
            raise ValueError("Pilot manifest does not match the input JSONL bytes/count")
        if any(record["source"]["book_id"] != manifest["source"]["book_id"] for record in records):
            raise ValueError("Pilot manifest book identity differs from input records")
        source["pilot_manifest"] = {"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path)}
        source["book_pdf_declared_by_manifest"] = deepcopy(manifest["source"])
        source["book_pdf_verified_by_this_probe"] = False
    return source


def audit_file(input_path):
    started = time.perf_counter()
    records = load_jsonl(input_path)
    intake = validate_records(records)
    if not intake["valid"]:
        raise ValueError("Invalid source collection: " + "; ".join(intake["errors"]))
    identities = provenance(input_path, records)
    rows = [audit_record(record) for record in records]
    files = [Path(__file__), ROOT / "src/riichi_ai/policies.py",
             ROOT / "src/riichi_ai/book_scenarios.py", Path(inspect.getfile(Shanten))]
    return {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        **identities, "runtime": {"python": platform.python_version(),
            "riichienv": metadata.version("riichienv"), "mahjong": metadata.version("mahjong")},
        "implementation_files": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in files],
        "source_intake_valid": intake["valid"], "records": rows,
        "record_count": len(rows), "candidate_count": sum(len(row["candidates"]) for row in rows),
        "all_calculators_agree": all(row["all_calculators_agree"] for row in rows),
        "elapsed_seconds": time.perf_counter() - started,
        "model_evaluations": 0, "optimizer_updates": 0, "new_matches": 0,
        "source_labels_modified": False, "policy_training_ready": False,
        "numerical_budget_management": "Caller must include this script and its tests in the numerical lease.",
        "scope": "Conditional hand-only arithmetic audit. No model predictions, complete player views, strategic label acceptance, or playing-strength inference."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Probe output already exists; no silent replacement")
    report = audit_file(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"report": str(args.output), "records": report["record_count"],
                      "candidates": report["candidate_count"], "all_calculators_agree": report["all_calculators_agree"]}))
    return 0 if report["all_calculators_agree"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
