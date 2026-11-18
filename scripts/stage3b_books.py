#!/usr/bin/env python3
"""Create or verify a fixed 12-case partial-scenario intake using stdlib only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from riichi_ai.book_scenarios import load_jsonl
from riichi_ai.book_scenarios_v2 import convert_packet, partial_observation, validate_records


INPUT = "Books/stage3a/source-packets-rev2.jsonl"
COUNT_REPORT = "reports/stage3a/book-counts-rev2.json"
PILOT = "Books/pilot/source-packets.jsonl"
PINS = {
    INPUT: "5deb9d3414d1ba094cb34c2ab897d6eff9958525cd3ba24a4cee26069cd40b9f",
    COUNT_REPORT: "59f187f2603c03923860f3ed6490d5456b49be7d29970c123846aef4ddc9592f",
    PILOT: "0a5b3b20ce4ada8de0bbc640afd42ae825db43b0b0a21f39a21fb375cdd9cccf",
}
OUTPUT = ROOT / "Books/stage3b"
IDS = [f"uzaku-301-q00{i}" for i in range(4, 10)] + [
    f"uzaku-300-q00{i}" for i in range(4, 8)
] + ["super-digital-hand-order-lesson-7-example-c", "statistics-check-problem-1-question-2"]


def reference(path: str) -> dict:
    payload = (ROOT / path).read_bytes()
    return {"path": path, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def _json(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def _jsonl(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
                   for row in rows).encode()


def prepare() -> tuple[dict[str, bytes], dict]:
    """Read exact prior evidence and construct bytes; no training dependencies."""
    inputs = [reference(path) for path in PINS]
    for ref in inputs:
        if ref["sha256"] != PINS[ref["path"]]:
            raise ValueError(f"Input changed: {ref['path']}")
    packets = load_jsonl(ROOT / INPUT)
    if [packet["packet_id"] for packet in packets] != IDS:
        raise ValueError("The exact fixed 12-case selection/order changed")
    arithmetic = json.loads((ROOT / COUNT_REPORT).read_text())
    if not (arithmetic["input"]["sha256"] == PINS[INPUT]
            and arithmetic["comparison_count"] == 47
            and arithmetic["all_calculators_agree"] is True
            and arithmetic["all_printed_counts_match_book_convention"] is True
            and arithmetic["printed_count_mismatches"] == []
            and arithmetic["calculator_disagreements"] == []):
        raise ValueError("The previous count audit does not support this conversion")
    audited = {packet["packet_id"]: packet for packet in arithmetic["packets"]}
    pilot = {row["packet_id"]: row for row in load_jsonl(ROOT / PILOT)}
    refs = {ref["path"]: ref for ref in inputs}
    scenarios, verified_links = [], []
    sources = {}
    evidence = {}
    for packet in packets:
        counts = packet["source_answer"]["reported_acceptance_counts"]
        if counts:
            report_counts = [{"discard": row["discard_tile"], "reported_count": row["reported_count"]}
                             for row in audited[packet["packet_id"]]["comparisons"]]
            if counts != report_counts:
                raise ValueError("Printed options differ from prior successful audit")
        for linked_id in packet["grouping"]["cross_stage_family_links"]:
            linked = pilot[linked_id]
            if (linked["source"]["problem_family"] != packet["grouping"]["problem_family"]
                    or linked["source"]["book_id"] != packet["source"]["book_id"]):
                raise ValueError("Cross-stage family does not match historical packet")
            verified_links.append({"id": packet["packet_id"], "related_id": linked_id,
                                   "problem_family": packet["grouping"]["problem_family"]})
        source = packet["source"]
        sources[source["path"]] = source["sha256"]
        for item in packet["evidence"]["page_renders"] + packet["evidence"].get("correction_crops", []):
            evidence[item["path"]] = item
        scenarios.append(convert_packet(
            packet, input_reference=refs[INPUT],
            arithmetic_reference=refs[COUNT_REPORT] if counts else None,
        ))
    # Authenticate source images/PDFs without rendering, OCR, engines or arithmetic.
    for path, expected in sources.items():
        actual = reference(path)
        if actual["sha256"] != expected:
            raise ValueError(f"Source PDF changed: {path}")
    for path, expected in evidence.items():
        actual = reference(path)
        if actual["sha256"] != expected["sha256"] or actual["bytes"] != expected["bytes"]:
            raise ValueError(f"Evidence changed: {path}")
    validation = validate_records(scenarios)
    if not validation["valid"]:
        raise ValueError(validation["errors"])
    observations = [partial_observation(row) for row in scenarios]
    payloads = {"partial-scenarios.jsonl": _jsonl(scenarios), "partial-observations.jsonl": _jsonl(observations)}
    implementation = [reference(path) for path in (
        "src/riichi_ai/book_scenarios_v2.py", "scripts/stage3b_books.py",
        "tests/test_book_scenarios_v2.py",
    )]
    report = {
        "schema": "riichi-ai-stage3b-book-preparation-v1",
        "python": platform.python_version(), "dependencies": "Python standard library only",
        "inputs": inputs, "implementation": implementation, "validation": validation,
        "source_pdfs_verified": len(sources), "source_images_verified": len(evidence),
        "prior_printed_counts_verified": 47,
        "arithmetic_rerun": False,
        "cross_stage_family_links_verified": verified_links,
        "compound_recommendations": sum(len(row["answer"]["recommendation"]["components"]) > 1 for row in scenarios),
        "conditional_annotations": sum(len(row["answer"]["conditions"]) for row in scenarios),
        "visual_red_five_candidates": sum(len(row["observation"]["red_notation"]["visual_candidates"]) for row in scenarios),
        "known_draws": sum(row["observation"]["drawn_tile"]["status"] == "supplied" for row in scenarios),
        "unknown_draws": sum(row["observation"]["drawn_tile"]["status"] == "unknown" for row in scenarios),
        "model_evaluations": 0, "optimizer_updates": 0, "new_matches": 0,
        "training_eligible_records": 0, "full_policy_ready_records": 0,
        "outputs": [
            {"path": "Books/stage3b/" + name, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
            for name, payload in payloads.items()
        ],
        "limitations": [
            "This is a structural conversion of the same 12 corrected Stage 3A cases, not 12 newly extracted cases.",
            "Visual interpretation and shape arithmetic are inherited from pinned prior evidence; neither was rerun.",
            "Source answers are attributed claims; no strategic optimality or average-rank calibration is established.",
            "Unknown context and red identities remain unresolved; no full-policy or training adapter is provided.",
            "Question-only export preserves row order but contains no record IDs; join through the authenticated record order only for review.",
        ],
    }
    payloads["preparation-report.json"] = _json(report)
    return payloads, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Compare existing output bytes without writing.")
    args = parser.parse_args(argv)
    payloads, report = prepare()
    if args.verify:
        for name, payload in payloads.items():
            if (OUTPUT / name).read_bytes() != payload:
                raise ValueError(f"Output mismatch: {name}")
    else:
        # Refuse overwrite; rerun with --verify after successful creation.
        if any((OUTPUT / name).exists() for name in payloads):
            raise FileExistsError("Outputs already exist; use --verify")
        OUTPUT.mkdir(parents=True, exist_ok=True)
        for name, payload in payloads.items():
            with (OUTPUT / name).open("xb") as stream:
                stream.write(payload)
    print(json.dumps({"valid": True, "mode": "verify" if args.verify else "create",
                      **report["validation"], "outputs": len(payloads),
                      "source_images_verified": report["source_images_verified"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
