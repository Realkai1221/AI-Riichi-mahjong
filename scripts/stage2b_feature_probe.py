"""Read-only exact-score feasibility probe on three Stage 2A training matches.

Run with .venv/bin/python scripts/stage2b_feature_probe.py. Only the first
three training actor/target shards are read; no test or auditor shard, model,
optimizer, or training routine is opened. Existing output is never overwritten.
"""

from __future__ import annotations

import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import platform
import time

import numpy as np

from riichi_ai.policies import _discard_score, shanten_cache_info


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "reports/stage2b-review/feature-feasibility.json"
MATCH_IDS = ("match-000000", "match-000001", "match-000002")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Preserving existing review evidence: {OUTPUT}")
    started = time.perf_counter()
    config_path = PROJECT / "configs/stage2_pilot.json"
    config_bytes = config_path.read_bytes()
    directory = Path(json.loads(config_bytes)["data_root"]) / "dataset"
    manifest_path = directory / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["valid"] is not True or tuple(manifest["splits"]["train"][:3]) != MATCH_IDS:
        raise ValueError("Expected the first three training matches of a valid Stage 2A dataset")
    source_paths = [Path(__file__).resolve()] + [PROJECT / "src/riichi_ai" / f"{name}.py"
                                              for name in ("policies", "player", "semantics", "features", "dataset")]
    source_hashes = {str(path.relative_to(PROJECT)): sha256(path.read_bytes()) for path in source_paths}
    by_id = {report["match_id"]: report for report in manifest["matches"]}
    cache_before = shanten_cache_info()._asdict()
    results = []
    for name in MATCH_IDS:
        match = by_id[name]
        if match["split"] != "train" or match["completed"] is not True:
            raise ValueError("Probe selection is not a completed training match")
        loaded, provenance = {}, {}
        for area in ("actor", "targets"):
            relative = f"{area}/{name}.npz"
            path = directory / relative
            payload = path.read_bytes()
            actual_hash = sha256(payload)
            expected = match["artifacts"][relative]
            if actual_hash != expected["sha256"] or len(payload) != expected["bytes"]:
                raise ValueError(f"Artifact differs from manifest: {relative}")
            provenance[area] = {"path": str(path), "sha256": actual_hash, "bytes": len(payload),
                                "matches_manifest_hash_and_size": True}
            fields = ("tiles", "candidate_offsets", "action_tokens") if area == "actor" else ("teacher_preference_keys",)
            with np.load(io.BytesIO(payload), allow_pickle=False) as shard:
                loaded[area] = {key: shard[key] for key in fields}
            provenance[area]["arrays_materialized"] = list(fields)
        actor, target = loaded["actor"], loaded["targets"]
        rows = len(actor["tiles"])
        if rows != match["decisions"] or len(actor["candidate_offsets"]) != rows + 1:
            raise ValueError("Row count disagrees with the manifest")
        result = {"match_id": name, "split": "train", "seed": match["seed"], "rows": rows,
                  "all_semantic_candidates": len(actor["action_tokens"]), "discard_candidates": 0,
                  "unique_discard_type_scores_computed": 0,
                  "count_decode_invalid_rows": 0, "shanten_mismatches": 0, "ukeire_mismatches": 0,
                  "improving_type_sum_mismatches": 0, "artifacts": provenance, "mismatch_examples": []}
        for row in range(rows):
            own = actor["tiles"][row, :, 0] * 4
            visible = actor["tiles"][row, :, 4] * 4
            valid_counts = (own.shape == visible.shape == (34,) and np.isfinite(own).all()
                            and np.isfinite(visible).all() and np.all((own >= 0) & (own <= 4))
                            and np.all((visible >= own) & (visible <= 4))
                            and np.array_equal(own, own.astype(np.int64))
                            and np.array_equal(visible, visible.astype(np.int64)))
            if not valid_counts:
                result["count_decode_invalid_rows"] += 1
                continue
            counts, seen = tuple(map(int, own)), tuple(map(int, visible))
            start, end = map(int, actor["candidate_offsets"][row:row + 2])
            per_type = {}
            for index in range(start, end):
                token = actor["action_tokens"][index]
                if int(token[0]) != 1:  # semantic-actions-v1 DISCARD token ID
                    continue
                kind = int(token[1]) - 1
                if not 0 <= kind < 34 or counts[kind] < 1:
                    raise ValueError("Discard type is absent from the decoded hand")
                if kind not in per_type:
                    # Visibility remains unchanged: moving a tile from one's
                    # hand to the river does not make its copy unseen.
                    per_type[kind] = _discard_score(counts, seen, kind)
                    result["unique_discard_type_scores_computed"] += 1
                score = per_type[kind]
                expected = target["teacher_preference_keys"][index]
                if int(expected[0]) != 2:
                    raise ValueError("A recorded discard has the wrong teacher branch")
                wrong_shanten = score.shanten != int(expected[2])
                wrong_ukeire = score.ukeire != -int(expected[3])
                wrong_sum = score.ukeire != sum(4 - seen[kind] for kind in score.improving_types)
                result["shanten_mismatches"] += int(wrong_shanten)
                result["ukeire_mismatches"] += int(wrong_ukeire)
                result["improving_type_sum_mismatches"] += int(wrong_sum)
                result["discard_candidates"] += 1
                if (wrong_shanten or wrong_ukeire or wrong_sum) and len(result["mismatch_examples"]) < 10:
                    result["mismatch_examples"].append({"row": row, "candidate_index_in_match": index,
                                                        "tile_type": kind, "shanten_mismatch": bool(wrong_shanten),
                                                        "ukeire_mismatch": bool(wrong_ukeire),
                                                        "improving_sum_mismatch": bool(wrong_sum)})
        results.append(result)
    sources_unchanged = source_hashes == {str(path.relative_to(PROJECT)): sha256(path.read_bytes()) for path in source_paths}
    inputs_unchanged = manifest_path.read_bytes() == manifest_bytes and config_path.read_bytes() == config_bytes
    total_keys = ("rows", "all_semantic_candidates", "discard_candidates", "unique_discard_type_scores_computed",
                  "count_decode_invalid_rows", "shanten_mismatches", "ukeire_mismatches", "improving_type_sum_mismatches")
    totals = {key: sum(match[key] for match in results) for key in total_keys}
    passed = sources_unchanged and inputs_unchanged and not any(totals[key] for key in total_keys[4:])
    report = {
        "schema": "riichi-ai-stage2b-feature-feasibility-v1", "passed": passed,
        "scope": "Only the first three Stage 2A training matches; no corpus-wide audit or feature implementation.",
        "matches": results, "totals": totals, "match_ids": list(MATCH_IDS),
        "derivation": {"own_counts": "4 * actor.tiles[row,:,0]",
                       "visible_counts": "4 * actor.tiles[row,:,4]; already includes own hand and deduplicates public aliases",
                       "discard_type": "action_tokens[candidate,1] - 1 for DISCARD token ID 1",
                       "score": "Existing pinned _discard_score(counts,visible,discard_type); visibility unchanged after discard",
                       "checks": "Exact finite integral count decoding in [0,4], own<=visible; exact shanten and ukeire comparisons, without tolerances",
                       "red_and_draw": "Candidates of the same tile type share the reconstructed shape score; existing semantic rows remain distinct",
                       "third_field": "DiscardScore.improving_types is a tuple; its proposed scalar feature is len(tuple). No recorded third-feature label exists."},
        "training_performed": False, "optimizer_used": False, "test_shards_opened": False,
        "auditor_shards_opened": False, "stage2a_files_modified": False,
        "cache": {"before": cache_before, "after": shanten_cache_info()._asdict()},
        "provenance": {"dataset_manifest_path": str(manifest_path),
                       "dataset_manifest_sha256": sha256(manifest_bytes),
                       "dataset_plan_sha256_recorded_in_manifest": manifest["plan_sha256"],
                       "campaign_config_path": str(config_path), "campaign_config_sha256": sha256(config_bytes),
                       "source_sha256": source_hashes, "sources_unchanged": sources_unchanged,
                       "input_manifests_unchanged": inputs_unchanged,
                       "python": platform.python_version(), "numpy": str(np.__version__),
                       "riichienv": version("riichienv")},
        "limitations": [
            "These selected matches do not establish correctness for every training/validation row, rare open-hand path, or future dataset.",
            "The same shanten/ukeire helper generated the teacher labels, so agreement is a serialization/reconstruction consistency check, not an independent Mahjong oracle.",
            "No hidden-information invariance or new model-quality claim follows from this probe; those remain separate implementation gates.",
            "Cache timings are a short warm-up feasibility measurement, not a sustained online-feature or self-play throughput benchmark.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"report": str(OUTPUT), "passed": passed, "totals": totals,
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2), flush=True)
    if not passed:
        raise RuntimeError("Feature feasibility check failed; inspect the retained report")


if __name__ == "__main__":
    main()
