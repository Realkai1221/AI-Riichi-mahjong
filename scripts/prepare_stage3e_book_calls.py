#!/usr/bin/env python3
"""Migrate two authenticated Stage 3D packets; no extraction or policy work."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from riichi_ai.book_calls import convert_packet, partial_observation, validate_export, validate_records

INPUTS = {
    "Books/stage3d/call-source-packets.jsonl": "6bfcb12ceb9bf54d2984299cd1971eddfebd38e367454191857fa1914f2591e1",
    "Books/stage3d/call-observations.jsonl": "363d07be026c5267157afd08670ca79eb8d3772ed108d9b5db1a34a6abda07dc",
    "Books/stage3d/call-observation-map.json": "edac8335f9fda63765020ddb65eb2bfd42be4bf875b3da306e7fb97e8be11925",
}


def reference(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Books/stage3e")
    args = parser.parse_args()
    refs = {name: reference(ROOT / name) for name in INPUTS}
    for name, digest in INPUTS.items():
        if refs[name]["sha256"] != digest:
            raise ValueError(f"Authenticated Stage 3D input changed: {name}")
    raw_path = "Books/stage3d/call-source-packets.jsonl"
    packets = [json.loads(line) for line in (ROOT / raw_path).read_text().splitlines() if line.strip()]
    records = [convert_packet(packet, input_reference=refs[raw_path]) for packet in packets]
    validation = validate_records(records)
    if not validation["valid"] or len(records) != 2:
        raise ValueError(validation)
    exports = [partial_observation(record) for record in records]
    previous = [json.loads(line) for line in (ROOT / "Books/stage3d/call-observations.jsonl").read_text().splitlines() if line.strip()]
    if exports != previous:
        raise ValueError("Question-only export changed during migration")
    for export, record in zip(exports, records, strict=True):
        validate_export(export, record)
    mapping = {"schema": "riichi-ai-call-question-map-v1", "actor_input": False,
               "rows": [{"zero_based_row": i, "packet_id": row["id"]} for i, row in enumerate(records)]}
    if mapping != json.loads((ROOT / "Books/stage3d/call-observation-map.json").read_text()):
        raise ValueError("Question row order or source mapping changed")
    output = args.output_dir.resolve()
    if output != ROOT / "Books/stage3e":
        raise ValueError("Migration output is restricted to the new Books/stage3e directory")
    output.mkdir(parents=True, exist_ok=True)
    names = ["partial-call-records.jsonl", "call-observations.jsonl", "call-observation-map.json", "validation.json"]
    if any((output / name).exists() for name in names):
        raise FileExistsError("Preserve existing Stage 3E preparation; outputs are write-once")
    payloads = {
        names[0]: "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records),
        names[1]: "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in exports),
        names[2]: json.dumps(mapping, ensure_ascii=False, indent=2) + "\n",
    }
    for name, text in payloads.items():
        with (output / name).open("x", encoding="utf-8") as stream:
            stream.write(text)
    report = {**validation, "input_references": refs, "prior_exports_semantically_unchanged": True,
              "gate_preferences": 1, "conditional_call_selections": 1, "certified_engine_actions": 0,
              "artifacts": [reference(output / name) for name in payloads],
              "permissions": "Preparation only; training/evaluation/redistribution remain unassessed.",
              "no_new_extraction": True, "no_model_or_engine_work": True}
    with (output / names[3]).open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    if any(reference(ROOT / name) != refs[name] for name in INPUTS):
        raise ValueError("Prior input changed while preparing output")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
