"""Bounded dataset integrity checks and selected full-match semantic replays.

This audit deliberately opens privileged auditor files. The actor loader does
not use this module. Replay checks consistency with the pinned implementation,
not independent Mahjong rules conformance or teacher playing strength.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from .features import ACTION_FIELDS, GLOBAL_FEATURES, HISTORY_FIELDS, MAX_HISTORY, TILE_CHANNELS, encode_view
from .player import PlayerAdapter
from .policies import policy_manifest
from .rules import audit_physical_tiles, rule_profile_hash
from .semantics import ACTION_KINDS, teacher_decision
from .simulator import INITIAL_SCORES, create_env, score_total


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _encoding(actor: dict, row: int) -> dict:
    start, end = map(int, actor["candidate_offsets"][row:row + 2])
    hstart, length = int(actor["history_start"][row]), int(actor["history_length"][row])
    history = np.zeros((MAX_HISTORY, HISTORY_FIELDS), dtype=np.int16)
    mask = np.zeros(MAX_HISTORY, dtype=np.bool_)
    history[:length] = actor["history_tokens"][hstart:hstart + length]
    mask[:length] = True
    return {"tiles": actor["tiles"][row], "globals": actor["globals"][row],
            "history": history, "history_mask": mask,
            "action_tokens": actor["action_tokens"][start:end]}


def _validate_arrays(actor: dict, targets: dict, report: dict) -> int:
    rows = report["decisions"]
    _require(type(rows) is int and rows > 0, "Empty or invalid decision count")
    _require(set(actor) == {"tiles", "globals", "action_tokens", "candidate_offsets", "history_tokens",
                            "history_start", "history_length"}, "Unexpected actor array fields")
    _require(set(targets) == {"policy_target", "best_mask", "teacher_preference_keys", "chosen_index",
                              "final_rank", "seat", "step", "hand_id", "physical_candidate_count"},
             "Unexpected target array fields")
    for key, value in (*actor.items(), *targets.items()):
        _require(not value.dtype.hasobject and np.isfinite(value).all(), f"Nonfinite/non-numeric array: {key}")
        _require(value.dtype.kind in "biuf", f"Unsupported numeric dtype: {key}")
    _require(actor["tiles"].shape == (rows, 34, TILE_CHANNELS), "Tile feature shape mismatch")
    _require(actor["globals"].shape == (rows, GLOBAL_FEATURES), "Global feature shape mismatch")
    offsets = actor["candidate_offsets"]
    _require(offsets.dtype.kind in "iu" and offsets.shape == (rows + 1,), "Candidate offsets shape/dtype")
    _require(offsets[0] == 0 and np.all(np.diff(offsets) > 0), "Candidate offsets must increase from zero")
    candidates = int(offsets[-1])
    _require(actor["action_tokens"].shape == (candidates, ACTION_FIELDS), "Candidate token shape mismatch")
    _require(actor["action_tokens"].dtype.kind in "iu", "Candidate tokens must be integers")
    _require(np.all((actor["action_tokens"][:, 0] >= 1)
                    & (actor["action_tokens"][:, 0] <= len(ACTION_KINDS))), "Unknown candidate kind")
    _require(actor["history_tokens"].ndim == 2 and actor["history_tokens"].shape[1] == HISTORY_FIELDS,
             "History token shape mismatch")
    _require(actor["history_tokens"].dtype.kind in "iu", "History tokens must be integers")
    for key in ("history_start", "history_length"):
        _require(actor[key].shape == (rows,) and actor[key].dtype.kind in "iu", f"History reference shape/dtype: {key}")
    starts, lengths = actor["history_start"], actor["history_length"]
    _require(np.all((starts >= 0) & (lengths >= 0) & (lengths <= MAX_HISTORY)
                    & (starts + lengths <= len(actor["history_tokens"]))), "History reference out of range")
    for key in ("policy_target", "best_mask"):
        _require(targets[key].shape == (candidates,), f"Candidate target length mismatch: {key}")
    _require(targets["best_mask"].dtype == np.bool_, "Best mask must be Boolean")
    _require(targets["teacher_preference_keys"].shape == (candidates, 5), "Preference shape mismatch")
    _require(targets["teacher_preference_keys"].dtype.kind in "iu", "Preferences must be integers")
    for key in ("chosen_index", "final_rank", "seat", "step", "hand_id", "physical_candidate_count"):
        _require(targets[key].shape == (rows,) and targets[key].dtype.kind in "iu", f"Row target shape/dtype: {key}")
    seats = targets["seat"]
    _require(np.all((seats >= 0) & (seats < 4)), "Seat out of range")
    _require(sorted(report["final_ranks"]) == [1, 2, 3, 4], "Terminal ranks are not a permutation")
    _require(np.array_equal(targets["final_rank"], np.asarray(report["final_ranks"])[seats]),
             "Terminal rank labels differ from the match outcome")
    _require(np.all((targets["step"] >= 0) & (targets["step"] < report["steps"]))
             and np.all(targets["hand_id"] >= 1), "Step or hand ID out of range")
    _require(np.all(targets["physical_candidate_count"] >= np.diff(offsets)), "Semantic count exceeds physical count")
    for row in range(rows):
        start, end = map(int, offsets[row:row + 2])
        q, best = targets["policy_target"][start:end], targets["best_mask"][start:end]
        chosen = int(targets["chosen_index"][row])
        _require(0 <= chosen < end - start and best[chosen], f"Chosen index is not best at row {row}")
        _require(best.any() and np.all(q >= 0) and np.isclose(q.sum(), 1, atol=1e-6),
                 f"Policy target is not normalized at row {row}")
        expected_q = best.astype(np.float32) / best.sum()
        _require(np.allclose(q, expected_q, atol=1e-7, rtol=0), f"Nonuniform semantic tie target at row {row}")
        preferences = [tuple(map(int, item)) for item in targets["teacher_preference_keys"][start:end]]
        expected_best = np.asarray([item == min(preferences) for item in preferences])
        _require(np.array_equal(best, expected_best), f"Best mask disagrees with preferences at row {row}")
        _require(chosen == int(np.flatnonzero(best)[0]), f"Noncanonical teacher choice at row {row}")
    return rows


def _replay(report: dict, auditor: dict, actor: dict, targets: dict, teacher: str) -> int:
    env, observations = create_env(report["seed"])
    adapters = [PlayerAdapter(seat) for seat in range(4)]
    row = 0
    for step, batch in enumerate(auditor["policy_requests"]):
        _require(not env.done() and batch["step"] == step, "Unexpected replay request step")
        views = {seat: adapters[seat].build(observations[seat]) for seat in sorted(observations)}
        requested = [item["seat"] for item in batch["requests"]]
        _require(requested == [seat for seat, view in views.items() if view.legal_actions],
                 "Saved request set differs from actual simultaneous legal requests")
        chosen = {}
        for request in batch["requests"]:
            seat = request["seat"]
            view = views[seat]
            decision = teacher_decision(view, defense=teacher == "defense_v2")
            _require(request["row"] == row and row < report["decisions"], "Replay row mismatch")
            _require((int(targets["seat"][row]), int(targets["step"][row]), int(targets["hand_id"][row]))
                     == (seat, step, view.hand_id) and request["hand_id"] == view.hand_id,
                     f"Replay seat/step/hand mismatch at row {row}")
            encoded = encode_view(view, decision.actions)
            stored = _encoding(actor, row)
            for key in encoded:
                _require(np.array_equal(encoded[key], stored[key]), f"Actor replay differs at row {row}: {key}")
            _require(request["physical_candidates"] == [action.token for action in view.legal_actions],
                     f"Physical legal actions differ at row {row}")
            _require(request["semantic_keys"] == [action.key for action in decision.actions],
                     f"Semantic action order differs at row {row}")
            start, end = map(int, actor["candidate_offsets"][row:row + 2])
            for key, expected in (("best_mask", decision.best_mask),
                                  ("teacher_preference_keys", decision.preference_keys)):
                _require(np.array_equal(targets[key][start:end], expected), f"Teacher replay differs at row {row}: {key}")
            _require(np.array_equal(targets["policy_target"][start:end], np.asarray(decision.soft_targets, dtype=np.float32)),
                     f"Teacher probabilities differ at row {row}")
            _require(request["best_mask"] == list(decision.best_mask)
                     and request["preference_keys"] == [list(key) for key in decision.preference_keys],
                     f"Auditor teacher labels differ at row {row}")
            _require(request["chosen_index"] == int(targets["chosen_index"][row]) == decision.chosen_index
                     and request["behavior_probability"] == 1.0, f"Teacher behavior differs at row {row}")
            selected = decision.actions[decision.chosen_index].representative
            _require(selected.token == request["submitted_action"], f"Submitted physical action differs at row {row}")
            chosen[seat] = adapters[seat].resolve(selected, observations[seat])
            row += 1
        observations = env.step(chosen)
        audit_physical_tiles(env)
        _require(score_total(env) == sum(INITIAL_SCORES), "Replay point conservation failed")
    _require(env.done() and row == report["decisions"], "Replay did not reach the complete terminal match")
    _require(env.mjai_log == auditor["canonical_events"], "Replayed canonical events differ")
    _require(list(env.ranks()) == report["final_ranks"] == auditor["final_ranks"], "Replayed terminal ranks differ")
    _require(list(env.scores()) == report["final_scores"] == auditor["final_scores"], "Replayed terminal scores differ")
    return row


def verify_dataset(output_dir: str | Path, replay_matches: int = 3) -> dict[str, Any]:
    """Check every shard, then fully replay evenly spaced selected matches.

    Three requested replays select first/middle/last (or every match for a
    smaller dataset). Errors are returned explicitly; callers must require
    ``valid`` before using the data. Processing holds only one match at a time.
    """
    if type(replay_matches) is not int or replay_matches < 0:
        raise ValueError("replay_matches must be a nonnegative integer")
    from .dataset import AUDITOR_SCHEMA, validate_manifest

    started = time.perf_counter()
    directory = Path(output_dir).expanduser().resolve()
    result: dict[str, Any] = {"schema": "riichi-ai-dataset-verification-v1", "valid": False, "errors": [],
                             "checked_matches": 0, "checked_artifacts": 0, "checked_rows": 0,
                             "replayed_matches": 0, "replayed_rows": 0, "replayed_match_ids": []}
    try:
        manifest = validate_manifest(directory)
        _require(manifest["rule_profile_hash"] == rule_profile_hash(), "Pinned rule profile differs")
        teacher = manifest["teacher"]["policy"]
        _require(teacher in ("shanten_v2", "defense_v2"), "Unsupported teacher for replay")
        reports = manifest["matches"]
        replay_indices = set(np.linspace(0, len(reports) - 1, min(replay_matches, len(reports)), dtype=int).tolist())
        for index, report in enumerate(reports):
            name = report["match_id"]
            try:
                expected_paths = {f"actor/{name}.npz", f"targets/{name}.npz", f"auditor/{name}.json.gz"}
                _require(set(report["artifacts"]) == expected_paths, "Unexpected or missing match artifact")
                loaded = {}
                for relative in sorted(expected_paths):
                    metadata = report["artifacts"][relative]
                    raw = (directory / relative).read_bytes()
                    _require(len(raw) == metadata["bytes"] and hashlib.sha256(raw).hexdigest() == metadata["sha256"],
                             f"Artifact hash/length mismatch: {relative}")
                    result["checked_artifacts"] += 1
                    if relative.endswith(".npz"):
                        with np.load(io.BytesIO(raw), allow_pickle=False) as shard:
                            loaded[relative.split("/")[0]] = {key: shard[key] for key in shard.files}
                    else:
                        loaded["auditor"] = json.loads(gzip.decompress(raw))
                auditor = loaded["auditor"]
                _require(auditor["schema"] == AUDITOR_SCHEMA and auditor["completed"] and auditor["failure"] is None,
                         "Incomplete or invalid auditor record")
                plan = {key: report[key] for key in ("match_id", "index", "split", "seed", "independent_group")}
                _require(auditor["plan"] == plan and auditor["rules_hash"] == manifest["rule_profile_hash"],
                         "Auditor plan/rules identity differs")
                _require(auditor["policy_provenance"] == [policy_manifest(teacher, report["seed"] ^ seat)
                                                         for seat in range(4)], "Auditor teacher provenance differs")
                _require(auditor["final_ranks"] == report["final_ranks"]
                         and auditor["final_scores"] == report["final_scores"], "Auditor terminal labels differ")
                events_json = json.dumps(auditor["canonical_events"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
                _require(hashlib.sha256(events_json).hexdigest() == report["game_hash"], "Canonical game hash differs")
                _require(len(auditor["policy_requests"]) == report["steps"], "Auditor step count differs")
                _require(sum(len(batch["requests"]) for batch in auditor["policy_requests"]) == report["decisions"],
                         "Auditor decision count differs")
                result["checked_rows"] += _validate_arrays(loaded["actor"], loaded["targets"], report)
                result["checked_matches"] += 1
                if index in replay_indices:
                    result["replayed_rows"] += _replay(report, auditor, loaded["actor"], loaded["targets"], teacher)
                    result["replayed_matches"] += 1
                    result["replayed_match_ids"].append(name)
            except Exception as error:
                result["errors"].append({"match_id": name, "type": type(error).__name__, "message": str(error)})
        result["valid"] = not result["errors"] and result["checked_matches"] == len(reports)
    except Exception as error:
        result["errors"].append({"type": type(error).__name__, "message": str(error)})
    result["elapsed_seconds"] = time.perf_counter() - started
    result["scope"] = ("Every artifact hash and numeric row checked; selected whole matches replayed with the pinned engine/encoder/teacher. "
                       "This verifies internal consistency, not independent rules conformance or unobserved action coverage.")
    return result
