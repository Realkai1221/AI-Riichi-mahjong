#!/usr/bin/env python3
"""Audit corrected Stage 3A source packets through the unchanged first runner.

Use only under the caller's numerical lease. Fixed revision, correction manifest,
and first-runner identities are verified before numerical helpers are imported.
The original input, arithmetic report, and runner are never replaced.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib import util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "Books/stage3a/source-packets-rev2.jsonl"
INPUT_SHA256 = "5deb9d3414d1ba094cb34c2ab897d6eff9958525cd3ba24a4cee26069cd40b9f"
INPUT_BYTES = 73664
PARENT_PATH = ROOT / "scripts/stage3a_book_counts.py"
PARENT_SHA256 = "6bfe29b6734e7859416c574cfc24dfdc9c4741f92506c23bea548fb809b600a7"
PARENT_BYTES = 15079
MANIFEST_PATH = ROOT / "reports/stage3a/book-corrections.json"
MANIFEST_SHA256 = "668697024e31b5e1666f8aec0bdfa242e2f8a5440eab69294c4615752253216a"
MANIFEST_BYTES = 28850
ORIGINAL_INPUT_SHA256 = "90234c0d343d5d2219ed5270e0cd886db3270c34893f6298725225fd85addf4a"
ORIGINAL_INPUT_BYTES = 59246


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path):
    path = Path(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}


def pinned(path, digest, size):
    observed = identity(path)
    require(observed["sha256"] == digest and observed["bytes"] == size,
            f"Frozen revision dependency changed: {path}")
    return observed


def audit_file(input_path=DEFAULT_INPUT):
    # These operations are stdlib-only. The unchanged runner imports numerical
    # helpers only after its own input, source, and helper validations.
    wrapper_identity = identity(__file__)
    parent_identity = pinned(PARENT_PATH, PARENT_SHA256, PARENT_BYTES)
    manifest_identity = pinned(MANIFEST_PATH, MANIFEST_SHA256, MANIFEST_BYTES)
    input_identity = pinned(input_path, INPUT_SHA256, INPUT_BYTES)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    require(manifest["revised_packets"]["sha256"] == INPUT_SHA256
            and manifest["revised_packets"]["bytes"] == INPUT_BYTES,
            "Correction manifest does not describe the pinned revision")
    review = manifest["independent_revision_review"]
    require(review["status"] == "passed_transcription" and review["reviewer"] == "/root",
            "Corrected fields lack the recorded independent transcription review")
    original = manifest["original_packets"]
    require(original["sha256"] == ORIGINAL_INPUT_SHA256
            and original["bytes"] == ORIGINAL_INPUT_BYTES,
            "Correction manifest changed the original input identity")
    original_identity = pinned(ROOT / original["path"], original["sha256"], original["bytes"])
    first_audit = manifest["triggering_arithmetic_report"]
    first_audit_identity = pinned(ROOT / first_audit["path"], first_audit["sha256"], first_audit["bytes"])

    specification = util.spec_from_file_location("stage3a_frozen_first_count_runner", PARENT_PATH)
    parent = util.module_from_spec(specification)
    specification.loader.exec_module(parent)
    require(parent.INPUT_SHA256 == ORIGINAL_INPUT_SHA256 and parent.INPUT_BYTES == ORIGINAL_INPUT_BYTES,
            "First-runner input declaration differs from its frozen original")
    # Override only the input pin in this isolated in-memory module instance.
    # The source file, validation logic, packet selection, and arithmetic remain
    # unchanged. The explicit input argument avoids a bound-default-path trap.
    parent.INPUT_SHA256 = INPUT_SHA256
    parent.INPUT_BYTES = INPUT_BYTES
    report = parent.audit_file(input_path)
    require(report["input"] == input_identity, "Parent returned a different revision identity")
    require(report["comparison_count"] == 47 and report["engine_convention_comparisons"] == 94,
            "The revised audit must retain all 47 options under both conventions")
    checked = [wrapper_identity, parent_identity, manifest_identity,
               input_identity, original_identity, first_audit_identity]
    for initial in checked:
        current = identity(initial["path"])
        require(current == initial, "Revision provenance changed during arithmetic audit")
    report["schema"] = "riichi-ai-stage3a-book-counts-rev2-v1"
    report["implementation_files"].append(wrapper_identity)
    report["revision_provenance"] = {
        "corrected_packet_input": input_identity,
        "corrections_manifest": manifest_identity,
        "preserved_original_input": original_identity,
        "preserved_first_arithmetic_report": first_audit_identity,
        "unchanged_parent_runner": parent_identity,
        "revision_wrapper": wrapper_identity,
        "independent_corrected_field_review": review,
        "first_audit_printed_mismatch_count": manifest["first_audit_result"]["printed_count_mismatch_count"],
        "note": "The original transcription and independent review missed glyph errors. This separate report audits the visually corrected revision; no previous report is replaced.",
    }
    report["selection"] = "The same ten Uzaku packet IDs and all 47 printed numeric values, with source-confirmed revision-2 tile glyphs; no result-dependent filtering."
    report["scope"] = "Conditional arithmetic audit of all 47 printed options in the corrected revision, retaining every mismatch and calculator disagreement. No strategic labels are validated."
    return report


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
