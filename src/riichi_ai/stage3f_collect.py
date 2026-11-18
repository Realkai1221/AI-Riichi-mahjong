"""Four-seat development collection with explicit mixed behavior propensities.

Only immutable PlayerView encodings enter policies. Auditor logs, terminal
labels and per-seat provenance have separate files. This module neither trains
nor declares its development corpus eligible for training or confirmation.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import gzip
import hashlib
import io
import math
import os
from pathlib import Path
import random
import re
import time
from typing import Any, Sequence

import numpy as np
import torch

from .auditing import PlayerViewHasher, VIEW_HASH_SCHEMA
from .dataset import _json, _pack_arrays, _rss
from .evaluation import _wall_fingerprint
from .features import encode_history_events, encode_view
from .model import NeuralPolicy, PolicyNetwork, _weights_hash, collate_encoded
from .player import PlayerAdapter, PlayerView
from .policies import policy_manifest
from .rules import audit_physical_tiles, rule_profile_hash
from .semantics import SEMANTIC_ACTION_VERSION, semantic_actions, teacher_decision
from .simulator import INITIAL_SCORES, create_env, score_total
from .stage3c import CallPolicy, CALL_KINDS, HEAD_CONFIG, SCHEMA as CALL_SCHEMA
from .stage3c import call_resolutions, sample_categorical, terminal_result

SCHEMA = "riichi-ai-stage3f-collection-v1"
ACTOR_SCHEMA = "riichi-ai-stage3f-compact-actor-v1"
TARGET_SCHEMA = "riichi-ai-stage3f-match-targets-v1"
BEHAVIORS = ("shanten_v2", "defense_v2", "reference", "explorer")
NAMESPACE = "riichi-ai/stage3f/development/v1"
EXPLORATION = .05
SAMPLER = "python-random-float64-latent-mixture-cdf-v1"


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value)).hexdigest()


def _seed(root: int, *labels: Any) -> int:
    return int(_hash([NAMESPACE, root, *labels])[:16], 16)


def derive_plan(root_seed: int, index: int) -> dict[str, Any]:
    """Derive one match assignment, independently of worker scheduling."""
    if type(root_seed) is not int or not 0 <= root_seed < 2**64 or type(index) is not int or index < 0:
        raise ValueError("Expected uint64 root and nonnegative index")
    names = list(BEHAVIORS)
    random.Random(_seed(root_seed, index, "seat-assignment")).shuffle(names)
    return {"match_id": f"match-{index:06d}", "index": index,
            "seed": _seed(root_seed, index, "wall"), "role": "development",
            "seat_behaviors": names,
            "sampling_seeds": [_seed(root_seed, index, "sampling", seat) for seat in range(4)],
            "independent_group": f"{NAMESPACE}:{root_seed}:{index}"}


def validate_plan(plan: dict[str, Any]) -> None:
    if (not isinstance(plan, dict) or not re.fullmatch(r"[A-Za-z0-9_-]+", str(plan.get("match_id", "")))
            or plan.get("role") != "development" or type(plan.get("index")) is not int or plan["index"] < 0
            or type(plan.get("seed")) is not int or not 0 <= plan["seed"] < 2**64
            or not isinstance(plan.get("seat_behaviors"), list)
            or sorted(plan["seat_behaviors"]) != sorted(BEHAVIORS)
            or not isinstance(plan.get("sampling_seeds"), list) or len(plan["sampling_seeds"]) != 4
            or any(type(value) is not int or not 0 <= value < 2**64 for value in plan["sampling_seeds"])
            or len(set(plan["sampling_seeds"])) != 4):
        raise ValueError("Expected a safe development plan with four distinct behaviors and sampling seeds")


def _distribution(values: Sequence[float], count: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if (result.shape != (count,) or count < 1 or not np.isfinite(result).all()
            or np.any(result < 0) or not math.isclose(float(result.sum()), 1., abs_tol=1e-12, rel_tol=0)):
        raise ValueError("Expected a finite normalized semantic probability vector")
    return result / result.sum()


def explore(actions: Sequence[Any], base_probabilities: Sequence[float], rng: random.Random) -> dict[str, Any]:
    """Sample a declared latent component and retain its marginal propensity.

    Wins use a collection heuristic, not a claim that declining a win can never
    be optimal. Otherwise the exact marginal is .95 * gated base + .05 *
    semantic uniform. One component draw precedes one conditional action draw.
    """
    base = _distribution(base_probabilities, len(actions))
    wins = [i for i, action in enumerate(actions) if action.kind in ("RON", "TSUMO")]
    if wins:
        selected = wins[0]
        probabilities = np.zeros(len(actions), dtype=np.float64)
        probabilities[selected] = 1.
        return {"selected_index": selected, "probabilities": probabilities.tolist(),
                "base_probabilities": base.tolist(), "uniform_probabilities": None,
                "selected_component": "win_heuristic", "component_probability": 1.,
                "conditional_probability": 1., "joint_probability": 1.,
                "component_draw": None, "uniform_draw": None, "probability": 1.}
    uniform = np.full(len(actions), 1. / len(actions), dtype=np.float64)
    probabilities = _distribution((1. - EXPLORATION) * base + EXPLORATION * uniform, len(actions))
    component_draw = rng.random()
    uniform_selected = component_draw < EXPLORATION
    component = uniform if uniform_selected else base
    selected, draw = sample_categorical(component, rng)
    weight = EXPLORATION if uniform_selected else 1. - EXPLORATION
    return {"selected_index": selected, "probabilities": probabilities.tolist(),
            "base_probabilities": base.tolist(), "uniform_probabilities": uniform.tolist(),
            "selected_component": "semantic_uniform" if uniform_selected else "gated_base",
            "component_probability": weight, "conditional_probability": float(component[selected]),
            "joint_probability": weight * float(component[selected]),
            "component_draw": component_draw, "uniform_draw": draw,
            "probability": float(probabilities[selected])}


@lru_cache(maxsize=2)
def _policies(reference_path: str, reference_sha: str, call_path: str, call_sha: str):
    """Process-local immutable inference cache; no optimizer is constructed."""
    torch.set_num_threads(1)
    reference_bytes, call_bytes = Path(reference_path).read_bytes(), Path(call_path).read_bytes()
    if hashlib.sha256(reference_bytes).hexdigest() != reference_sha or hashlib.sha256(call_bytes).hexdigest() != call_sha:
        raise ValueError("Frozen checkpoint identity mismatch")
    checkpoint = torch.load(io.BytesIO(call_bytes), map_location="cpu", weights_only=True)
    if (checkpoint.get("schema") != CALL_SCHEMA or checkpoint.get("head_config") != HEAD_CONFIG
            or checkpoint.get("reference_sha256") != reference_sha):
        raise ValueError("Call checkpoint does not identify the declared original base")
    with torch.random.fork_rng(devices=[]):
        reference = NeuralPolicy.from_checkpoint(reference_path, device="cpu")
        call = CallPolicy(PolicyNetwork(checkpoint["model_config"]), checkpoint["initialization_seed"])
    call.load_state_dict(checkpoint["model_state_dict"], strict=True)
    call.eval()
    if (reference.checkpoint_hash != reference_sha or call.config != reference.model.config
            or _weights_hash(call.base) != reference.weights_hash
            or _weights_hash(call) != checkpoint["weights_sha256"]
            or _weights_hash(call.heads) != checkpoint["heads_sha256"]):
        raise ValueError("Frozen reference and call-policy state do not match checkpoint metadata")
    return reference, call, {"checkpoint_sha256": call_sha, "weights_sha256": _weights_hash(call),
        "head_weights_sha256": _weights_hash(call.heads), "source_updates": checkpoint["updates"],
        "source_protocol_sha256": checkpoint["protocol_sha256"]}


def _decision(name: str, view: PlayerView, reference, call, rng: random.Random):
    actions = semantic_actions(view)
    encoded = encode_view(view, actions, feature_schema=reference.model.config["feature_schema"])
    if not actions:
        raise ValueError("No legal semantic candidate")
    if name == "explorer":
        if len(actions) == 1:
            q = np.ones(1, dtype=np.float64)
        else:
            with torch.inference_mode():
                q = call.distribution(encoded, "cpu")["probabilities"].numpy()
        result = explore(actions, q, rng)
    else:
        if name in ("shanten_v2", "defense_v2"):
            teacher = teacher_decision(view, defense=name == "defense_v2")
            if teacher.actions != actions:
                raise ValueError("Teacher semantic candidate ordering changed")
            selected = teacher.chosen_index
        elif name == "reference":
            selected = 0
            if len(actions) > 1:
                with torch.inference_mode():
                    logits, _ = reference.model(collate_encoded([encoded], "cpu"))
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError("Nonfinite reference logits")
                selected = int(logits[0].argmax())
        else:
            raise ValueError("Unknown behavior")
        probabilities = np.zeros(len(actions), dtype=np.float64)
        probabilities[selected] = 1.
        result = {"selected_index": selected, "probabilities": probabilities.tolist(),
                  "selected_component": name, "component_probability": 1.,
                  "conditional_probability": 1., "joint_probability": 1.,
                  "component_draw": None, "uniform_draw": None, "probability": 1.}
    return actions, encoded, result


def _write(path: Path, payload: bytes) -> None:
    directories = [path.parent]
    ancestor = path.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
        directories.append(ancestor)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    # Sync the new file entry and any newly created ancestor links.
    for directory in directories:
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def pack_actor(feature_rows, segments, history_refs) -> dict[str, np.ndarray]:
    """Compact histories retain verified per-row prefixes of seat/hand segments."""
    starts, lengths, history, cursor = {}, {}, [], 0
    for key, chunks in segments.items():
        values = np.concatenate(chunks, axis=0)
        starts[key], lengths[key] = cursor, len(values)
        cursor += len(values)
        history.append(values)
    offsets = np.cumsum([0] + [len(row["action_tokens"]) for row in feature_rows], dtype=np.int64)
    result = {"tiles": np.stack([row["tiles"] for row in feature_rows]).astype(np.float32),
        "globals": np.stack([row["globals"] for row in feature_rows]).astype(np.float32),
        "action_tokens": np.concatenate([row["action_tokens"] for row in feature_rows]).astype(np.int16),
        "candidate_offsets": offsets,
        "history_tokens": np.concatenate(history).astype(np.int16),
        "history_start": np.asarray([starts[key] + max(0, length - 128) for key, length in history_refs], dtype=np.int64),
        "history_length": np.asarray([min(128, length) for _, length in history_refs], dtype=np.int16),
        "history_segment_start": np.asarray([starts[key] for key, _ in history_refs], dtype=np.int64),
        "history_segment_length": np.asarray([lengths[key] for key, _ in history_refs], dtype=np.int32),
        "history_prefix_length": np.asarray([length for _, length in history_refs], dtype=np.int32)}
    if "candidate_efficiency" in feature_rows[0]:
        result["candidate_efficiency"] = np.concatenate([row["candidate_efficiency"] for row in feature_rows]).astype(np.float32)
    if any(array.dtype.hasobject or not np.isfinite(array).all() for array in result.values()):
        raise ValueError("Invalid actor array")
    return result


def run_one(plan: dict[str, Any], reference_path: str | Path, expected_reference_sha: str,
            call_checkpoint_path: str | Path, output_dir: str | Path, *, expected_call_sha256: str,
            deadline: float | None = None, max_steps: int = 100_000) -> dict[str, Any]:
    """Collect and durably persist one attempt; failures never produce training shards.

    The parent owns match-level admission, aggregate budgets and corpus closure.
    A start marker without a finish report remains evidence of a killed worker.
    """
    validate_plan(plan)
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be positive")
    started = time.monotonic()
    directory, match_id = Path(output_dir).resolve(), plan["match_id"]
    _write(directory / "lifecycle" / f"{match_id}.started.json", _json({"schema": SCHEMA, "plan": plan,
        "reference_sha256": expected_reference_sha, "call_checkpoint_sha256": expected_call_sha256,
        "max_steps": max_steps}))
    env = None
    failure = None
    steps, peak_rss = 0, _rss()
    rows, requests, feature_rows, fingerprints, hand_starts = [], [], [], [], []
    segments, history_lengths, history_refs = {}, {}, []
    timings = {key: 0. for key in ("load", "views", "decision", "encode_history", "step", "audit", "pack", "persist")}
    manifests = []
    rngs = [random.Random(seed) for seed in plan["sampling_seeds"]]
    rng_start = [rng.getstate() for rng in rngs]
    try:
        tick = time.monotonic()
        reference, call, call_manifest = _policies(str(Path(reference_path).resolve()), expected_reference_sha,
            str(Path(call_checkpoint_path).resolve()), expected_call_sha256)
        timings["load"] = time.monotonic() - tick
        for seat, name in enumerate(plan["seat_behaviors"]):
            details = (policy_manifest(name, plan["sampling_seeds"][seat]) if name in BEHAVIORS[:2]
                       else reference.manifest() if name == "reference" else {**call_manifest,
                           "collector_schema": SCHEMA, "sampler": SAMPLER, "epsilon": EXPLORATION,
                           "component_scope": "sampled independently at each non-win explorer decision",
                           "win_rule": "first legal semantic RON/TSUMO, probability 1",
                           "uniform_scope": "legal semantic non-win candidates",
                           "reference_sha256": expected_reference_sha,
                           "feature_schema": reference.model.config["feature_schema"]})
            manifests.append({"seat": seat, "behavior": name, "behavior_id": BEHAVIORS.index(name),
                "policy_id": name + ":" + _hash(details), "details": details,
                "sampling_seed": plan["sampling_seeds"][seat], "assignment_scope": "one fixed behavior for complete match"})
        env, observations = create_env(plan["seed"])
        adapters, hashers = [PlayerAdapter(seat) for seat in range(4)], [PlayerViewHasher() for _ in range(4)]
        while not env.done() and steps < max_steps:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Stage 3F cooperative match deadline reached")
            tick = time.monotonic()
            audit_physical_tiles(env)
            if score_total(env) != sum(INITIAL_SCORES):
                raise ValueError("Point/deposit conservation failure")
            timings["audit"] += time.monotonic() - tick
            tick = time.monotonic()
            # Build every immutable snapshot before invoking any actor.
            views = {seat: adapters[seat].build(observations[seat]) for seat in sorted(observations)}
            if not views or not any(view.legal_actions for view in views.values()):
                raise ValueError("Unfinished match has no legal requests")
            view_hashes = {str(seat): hashers[seat].digest(view) for seat, view in views.items()}
            pre_response_id = _hash({"step": steps, "views": view_hashes})
            hand_id = max(view.hand_id for view in views.values())
            if hand_id > len(fingerprints):
                if hand_id != len(fingerprints) + 1:
                    raise ValueError("Missing ordinal hand boundary")
                fingerprints.append(_wall_fingerprint(env))
                view = next(view for view in views.values() if view.hand_id == hand_id)
                start = next(event for event in view.history if event.kind == "start_kyoku")
                public = dict(start.data)
                hand_starts.append({"hand_id": hand_id, **{key: public[key]
                    for key in ("bakaze", "kyoku", "honba", "kyotaku", "oya", "scores")}})
            timings["views"] += time.monotonic() - tick
            choices, batch_rows = {}, []
            for seat, view in views.items():
                if not view.legal_actions:
                    continue
                tick = time.monotonic()
                actions, encoded, selection = _decision(plan["seat_behaviors"][seat], view, reference, call, rngs[seat])
                timings["decision"] += time.monotonic() - tick
                selected = actions[selection["selected_index"]]
                row = {"row_index": len(rows), "seat": seat, "step": steps, "hand_id": view.hand_id,
                    "behavior_id": BEHAVIORS.index(plan["seat_behaviors"][seat]),
                    "policy_id": manifests[seat]["policy_id"], "pre_response_id": pre_response_id,
                    "player_view_sha256": view_hashes[str(seat)],
                    "semantic_keys": [action.key for action in actions], "action_kinds": [action.kind for action in actions],
                    "physical_candidates": [action.token for action in view.legal_actions],
                    "representative_tokens": [action.representative.token for action in actions],
                    "selected_semantic_key": selected.key, "selected_kind": selected.kind,
                    "selected_physical_token": selected.representative.token,
                    "requested_call": selected.kind in CALL_KINDS, "requested_call_resolved": None,
                    "call_candidates": sum(action.kind in CALL_KINDS for action in actions),
                    "post_call_state": bool(view.melds[seat]), "open_hand_state": any(meld.opened for meld in view.melds[seat]),
                    "training_eligible": False, "ppo_eligible": False, **selection}
                rows.append(row)
                batch_rows.append(row["row_index"])
                feature_rows.append({key: value for key, value in encoded.items() if key not in ("history", "history_mask")})
                tick = time.monotonic()
                key = seat, view.hand_id
                previous = history_lengths.get(key, 0)
                if len(view.history) < previous:
                    raise ValueError("History shrank within one ordinal hand")
                segments.setdefault(key, []).append(encode_history_events(view.history[previous:], seat))
                history_lengths[key] = len(view.history)
                history_refs.append((key, len(view.history)))
                timings["encode_history"] += time.monotonic() - tick
                choices[seat] = selected.representative
            request = {"step": steps, "pre_response_id": pre_response_id, "player_views": view_hashes,
                       "rows": batch_rows, "submitted": False}
            requests.append(request)
            native = {seat: adapters[seat].resolve(action, observations[seat]) for seat, action in choices.items()}
            if any(rows[index]["requested_call"] for index in batch_rows):
                request["auditor_event_start"] = len(env.mjai_log)
            tick = time.monotonic()
            observations = env.step(native)
            timings["step"] += time.monotonic() - tick
            request["submitted"] = True
            if "auditor_event_start" in request:
                request["auditor_event_end"] = len(env.mjai_log)
            steps += 1
            if steps % 64 == 0:
                peak_rss = max(peak_rss, _rss())
        audit_physical_tiles(env)
        if score_total(env) != sum(INITIAL_SCORES):
            raise ValueError("Terminal point/deposit conservation failure")
        if not env.done():
            raise RuntimeError("TruncatedMatch: max_steps reached")
        if _weights_hash(reference.model) != reference.weights_hash or _weights_hash(call) != call_manifest["weights_sha256"]:
            raise ValueError("Frozen policy weights changed during collection")
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error), "step": steps}
    log = env.mjai_log if env is not None else []
    scores = list(env.scores()) if env is not None else None
    completed = bool(env is not None and env.done() and failure is None)
    try:
        terminal = terminal_result(scores or [], list(env.ranks()) if completed else [], log, completed=completed)
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error), "step": steps}
        completed = False
        terminal = terminal_result([], [], log, completed=False)
    resolutions = call_resolutions(log)
    for row in rows:
        if row["requested_call"]:
            request = requests[row["step"]] if row["step"] < len(requests) else {}
            matching = [item for item in resolutions if item["actor"] == row["seat"] and item["kind"] == row["selected_kind"]
                and request.get("auditor_event_start", len(log)) <= item["event_index"] < request.get("auditor_event_end", len(log))]
            row["call_resolution_evidence"] = matching
            row["requested_call_resolved"] = (None if not request.get("submitted") or any(item["resolved"] is None for item in matching)
                else any(item["resolved"] for item in matching))
    public_result = {"schema": TARGET_SCHEMA, "hand_starts": hand_starts, **terminal,
        "scores": scores, "training_eligible": False, "source_role": "development-only"}
    provenance = {"schema": SCHEMA, "plan": plan, "policies": manifests, "rows": rows,
        "requests": requests, "sampling_rng_start": rng_start, "sampling_rng_end": [rng.getstate() for rng in rngs],
        "exploration": {"epsilon": EXPLORATION, "component_scope": "sampled independently at each non-win explorer decision",
            "sampler": SAMPLER, "win_rule": "first legal semantic RON/TSUMO, probability 1; collection heuristic only"},
        "player_view_hash_schema": VIEW_HASH_SCHEMA, "semantic_action_schema": SEMANTIC_ACTION_VERSION,
        "feature_schema": reference.model.config["feature_schema"] if manifests else None,
        "completed": completed, "failure": failure, "training_eligible": False, "ppo_eligible": False}
    auditor = {"schema": SCHEMA, "plan": plan, "canonical_events": log,
        "ordinal_wall_fingerprints": fingerprints, "call_resolution_evidence": resolutions,
        "rule_profile_sha256": rule_profile_hash(), "completed": completed, "failure": failure}
    artifacts = {f"provenance/{match_id}.json.gz": gzip.compress(_json(provenance), mtime=0),
                 f"auditor/{match_id}.json.gz": gzip.compress(_json(auditor), mtime=0),
                 f"targets/{match_id}.json": _json(public_result)}
    actor_bytes = target_bytes = 0
    tick = time.monotonic()
    if completed:
        actor = pack_actor(feature_rows, segments, history_refs)
        targets = {"chosen_index": np.asarray([row["selected_index"] for row in rows], dtype=np.int16),
            "seat": np.asarray([row["seat"] for row in rows], dtype=np.int8),
            "step": np.asarray([row["step"] for row in rows], dtype=np.int32),
            "hand_id": np.asarray([row["hand_id"] for row in rows], dtype=np.int32),
            "behavior_id": np.asarray([row["behavior_id"] for row in rows], dtype=np.int8),
            "final_rank": np.asarray([terminal["ranks"][row["seat"]] for row in rows], dtype=np.int8),
            "final_utility": np.asarray([terminal["utilities"][row["seat"]] for row in rows], dtype=np.float32),
            "behavior_probability": np.asarray([row["probability"] for row in rows], dtype=np.float64),
            "behavior_probabilities": np.concatenate([np.asarray(row["probabilities"], dtype=np.float64) for row in rows])}
        actor_bytes = sum(value.nbytes for value in actor.values())
        target_bytes = sum(value.nbytes for value in targets.values())
        artifacts[f"actor/{match_id}.npz"] = _pack_arrays(actor)
        artifacts[f"targets/{match_id}.npz"] = _pack_arrays(targets)
    timings["pack"] = time.monotonic() - tick
    report = {**plan, "schema": SCHEMA, "actor_schema": ACTOR_SCHEMA, "target_schema": TARGET_SCHEMA,
        "completed": completed, "failure": failure, "truncated": bool(failure and "TruncatedMatch" in failure["message"]),
        "source_role": "development-only", "training_eligible": False, "ppo_eligible": False,
        "steps": steps, "rows": len(rows), "hands": len(hand_starts), "scores": scores,
        "ranks": terminal["ranks"], "utilities": terminal["utilities"], "terminal_result_verified": terminal["terminal_result_verified"],
        "physical_and_score_audits_passed": failure is None, "game_sha256": _hash(log),
        "reference_sha256": expected_reference_sha, "call_checkpoint_sha256": expected_call_sha256,
        "seat_rows": {str(seat): sum(row["seat"] == seat for row in rows) for seat in range(4)},
        "chosen_action_counts": dict(sorted(Counter(row["selected_kind"] for row in rows).items())),
        "requested_calls": {kind: sum(row["selected_kind"] == kind for row in rows) for kind in CALL_KINDS},
        "resolved_calls": {kind: sum(row["selected_kind"] == kind and row["requested_call_resolved"] is True for row in rows) for kind in CALL_KINDS},
        "post_call_rows": sum(row["post_call_state"] for row in rows), "open_hand_rows": sum(row["open_hand_state"] for row in rows),
        "artifacts": {name: {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)} for name, payload in sorted(artifacts.items())},
        "uncompressed_actor_bytes": actor_bytes, "uncompressed_target_bytes": target_bytes}
    report["payload_sha256"] = _hash(report)
    tick = time.monotonic()
    for name, payload in artifacts.items():
        _write(directory / name, payload)
    timings["persist"] = time.monotonic() - tick
    report.update(elapsed_seconds=time.monotonic() - started, peak_worker_rss_bytes=max(peak_rss, _rss()), timing_seconds=timings)
    _write(directory / "reports" / f"{match_id}.json", _json(report))
    _write(directory / "lifecycle" / f"{match_id}.finished.json", _json({"schema": SCHEMA, "match_id": match_id,
        "completed": completed, "payload_sha256": report["payload_sha256"], "report_sha256": _hash(report)}))
    return report
