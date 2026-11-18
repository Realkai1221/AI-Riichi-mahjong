"""Fixed-protocol continuation collection with durable lifecycle evidence.

The Stage 3C policy, sampler, actor features, simultaneous response ordering and
update function are reused unchanged. This local adaptation adds cooperative
cancellation and explicit completed-training eligibility to its collector.
"""
from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
import random
import time
from typing import Any

from .auditing import PlayerViewHasher, VIEW_HASH_SCHEMA
from .evaluation import _wall_fingerprint
from .model import NeuralPolicy, _weights_hash
from .player import PlayerAdapter
from .rules import audit_physical_tiles, rule_profile_hash
from .semantics import semantic_actions
from .simulator import INITIAL_SCORES, create_env, score_total
from .stage3c import (
    CALL_KINDS, SAMPLER, SCHEMA, MatchTrajectory, TrainingState,
    _deadline, _hash, actor_decision, call_resolutions, persist_match as _persist_match,
    terminal_result, update_batch,
)

COLLECTOR_SCHEMA = "stage3e-continuation-lifecycle-v1"


class CollectionCancelled(Exception):
    """An orderly stop observed between complete response steps."""


def collect_match(state: TrainingState, opponent: NeuralPolicy, *, collection_role: str, batch_index: int,
                             match_index: int, seed: int, learner_seat: int, sampling_seed: int,
                             max_steps: int = 100_000, deadline: float | None = None,
                             stop_requested: Callable[[], bool] | None = None) -> MatchTrajectory:
    """Collect one fixed-policy training or evaluation attempt with partial evidence.

    A cancellation check is made only at the original per-step deadline point,
    before building any requests for that step. A signal handler must set a flag;
    it must not raise an asynchronous exception inside a response or audit.
    The controller owns durable start/persist lifecycle markers and match limits.
    """
    if (learner_seat not in range(4) or type(batch_index) is not int or batch_index < 0 or max_steps < 1
            or collection_role not in ("training", "evaluation")
            or (collection_role == "training" and batch_index != state.updates)
            or (stop_requested is not None and not callable(stop_requested))):
        raise ValueError("Invalid learner seat, collection role, sequential batch index, max_steps or stop callback")
    if opponent.checkpoint_hash != state.reference_sha256:
        raise ValueError("Opponent is not the frozen original reference")
    policy_id = state.policy_id
    learner_manifest = {"policy_id": policy_id, "role": "learner", "sampler": SAMPLER,
                        "reference_sha256": state.reference_sha256,
                        "update": state.updates, "weights_sha256": _weights_hash(state.model),
                        "base_weights_sha256": state.immutable_base_sha256, "heads_sha256": _weights_hash(state.model.heads),
                        "sampling_seed": sampling_seed, "feature_schema": state.model.config["feature_schema"]}
    opponent_manifest = {**opponent.manifest(), "role": "frozen_opponent",
                         "policy_id": "frozen-argmax:" + str(opponent.checkpoint_hash)}
    manifests = [learner_manifest if seat == learner_seat else opponent_manifest for seat in range(4)]
    adapters = [PlayerAdapter(seat) for seat in range(4)]
    hashers = [PlayerViewHasher() for _ in range(4)]
    rng = random.Random(sampling_seed)
    rng_start = rng.getstate()
    env, observations = create_env(seed)
    rows, encoded_rows, trace = [], [], []
    fingerprints = []
    steps = 0
    failure = None
    interrupted = False
    started = time.monotonic()
    try:
        audit_physical_tiles(env)
        if score_total(env) != sum(INITIAL_SCORES):
            raise ValueError("Initial point conservation failure")
        while not env.done() and steps < max_steps:
            if stop_requested is not None and stop_requested():
                interrupted = True
                raise CollectionCancelled(f"{collection_role.capitalize()} collection cancellation requested")
            _deadline(deadline)
            # Build every immutable request from the SAME returned engine snapshot
            # before invoking any policy or submitting any response.
            views = {seat: adapters[seat].build(observations[seat]) for seat in sorted(observations)}
            if not views or not any(view.legal_actions for view in views.values()):
                raise ValueError("Unfinished match has no legal request")
            hand_id = max(view.hand_id for view in views.values())
            if hand_id > len(fingerprints):
                if hand_id != len(fingerprints) + 1:
                    raise ValueError("A hand boundary was omitted")
                fingerprints.append(_wall_fingerprint(env))
            view_hashes = {str(seat): hashers[seat].digest(view) for seat, view in views.items()}
            pre_response_id = _hash({"match_index": match_index, "step": steps, "views": view_hashes})
            choices, request = {}, {"step": steps, "pre_response_id": pre_response_id,
                                    "player_views": view_hashes, "choices": {}}
            for seat, view in views.items():
                if not view.legal_actions:
                    continue
                if seat == learner_seat:
                    chosen, row, encoded = actor_decision(
                        state.model, view, rng, device=state.device,
                        policy_id=policy_id, view_hash=view_hashes[str(seat)])
                    row.update({"step": steps, "pre_response_id": pre_response_id,
                                "row_index": len(rows), "match_index": match_index,
                                "batch_index": batch_index, "training_eligible": False})
                    rows.append(row)
                    encoded_rows.append(encoded)
                    choice = {"row_index": row["row_index"], "eligible": True,
                              "semantic_key": row["selected_semantic_key"],
                              "probability": row["probability"]}
                else:
                    chosen = opponent.choose(view)
                    semantic = next(a for a in semantic_actions(view) if chosen.token in a.physical_tokens)
                    choice = {"eligible": False, "semantic_key": semantic.key, "probability": 1.0}
                choices[seat] = chosen
                request["choices"][str(seat)] = {**choice, "physical_token": chosen.token,
                                                 "policy_id": manifests[seat]["policy_id"],
                                                 "fallback": False, "overridden": False}
            native = {seat: adapters[seat].resolve(action, observations[seat])
                      for seat, action in choices.items()}
            trace.append(request)
            log_start = len(env.mjai_log)
            observations = env.step(native)
            new_events = env.mjai_log[log_start:]
            request["declared_call_events"] = [dict(event) for event in new_events
                                               if event.get("type", "").upper() in CALL_KINDS]
            request["auditor_event_start"] = log_start
            request["auditor_event_end"] = len(env.mjai_log)
            if rows and rows[-1]["step"] == steps:
                row = rows[-1]
                requested_call = row["selected_kind"] in CALL_KINDS
                resolved = [event for event in request["declared_call_events"]
                            if event.get("actor") == learner_seat]
                row["requested_call"] = requested_call
                row["declared_call_events"] = resolved
                row["requested_call_resolved"] = (any(event["type"].upper() == row["selected_kind"]
                                                      for event in resolved) if requested_call else None)
            steps += 1
            audit_physical_tiles(env)
            if score_total(env) != sum(INITIAL_SCORES):
                raise ValueError("Points plus outstanding deposits do not conserve")
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error), "step": steps}
    completed = bool(env.done()) and failure is None
    log = env.mjai_log
    scores = list(env.scores())
    try:
        result = terminal_result(scores, list(env.ranks()), log, completed=completed)
        if _weights_hash(state.model.base) != state.immutable_base_sha256:
            raise ValueError("Frozen learner base changed")
        if state.policy_id != policy_id:
            raise ValueError("Learner weights changed during fixed-policy collection")
        if _weights_hash(opponent.model) != opponent.weights_hash:
            raise ValueError("Frozen opponent weights changed")
    except Exception as error:
        completed = False
        failure = {"type": type(error).__name__, "message": str(error), "step": steps}
        result = terminal_result(scores, (), log, completed=False)
    training_eligible = collection_role == "training" and completed
    resolutions = call_resolutions(log)
    for row in rows:
        row["training_eligible"] = training_eligible
        request = trace[row["step"]] if row["step"] < len(trace) else {}
        row["requested_call"] = row["selected_kind"] in CALL_KINDS
        declarations = [item for item in resolutions
                        if request.get("auditor_event_start", len(log)) <= item["event_index"] < request.get("auditor_event_end", len(log))
                        and item["actor"] == learner_seat]
        row["call_resolution_evidence"] = declarations
        if row["requested_call"]:
            matching = [item for item in declarations if item["kind"] == row["selected_kind"]]
            row["requested_call_resolved"] = (None if any(item["resolved"] is None for item in matching)
                                               else any(item["resolved"] for item in matching))
        else:
            row["requested_call_resolved"] = None
    record = {
        "schema": SCHEMA, "collector_schema": COLLECTOR_SCHEMA,
        "protocol_sha256": state.protocol_sha256,
        "batch_index": batch_index, "match_index": match_index, "seed": seed,
        "collection_role": collection_role, "training_eligible": training_eligible,
        "eligibility_scope": "Only completed training matches and their learner behavior rows may enter optimization.",
        "learner_seat": learner_seat, "sampling_seed": sampling_seed, "sampler": SAMPLER,
        "policy_id": policy_id, "policies": manifests,
        "rule_profile_sha256": rule_profile_hash(), "view_hash_schema": VIEW_HASH_SCHEMA,
        "completed": completed, "terminated": completed,
        "truncated": not completed and failure is None,
        "interrupted": interrupted,
        "stop_reason": ("cancelled" if failure and failure["type"] == "CollectionCancelled"
                        else "failure" if failure else "completed" if completed else "max_steps"),
        "failure": failure, "steps": steps, "max_steps": max_steps,
        "scores": scores, "riichi_sticks": env.riichi_sticks, **result,
        "learner_utility": result["utilities"][learner_seat] if completed else None,
        "learner_last_step": rows[-1]["step"] if rows else None,
        "opponent_steps_after_last_learner": steps - 1 - rows[-1]["step"] if rows else steps,
        "learner_rows": rows, "requests": trace,
        "sampling_rng_start": rng_start, "sampling_rng_end": rng.getstate(),
        "auditor_log": log, "auditor_log_sha256": _hash(log),
        "ordinal_wall_fingerprints": fingerprints, "call_resolution_evidence": resolutions,
        "hands": sum(event.get("type") == "start_kyoku" for event in log),
        "conservation_verified": failure is None,
        "requested_calls": {kind: sum(row["selected_kind"] == kind for row in rows) for kind in CALL_KINDS},
        "declared_calls": {kind: sum(item["kind"] == kind and item["actor"] == learner_seat
                                     for item in resolutions) for kind in CALL_KINDS},
        "resolved_calls": {kind: sum(item["kind"] == kind and item["actor"] == learner_seat and item["resolved"] is True
                                     for item in resolutions) for kind in CALL_KINDS},
        "post_call_rows": sum(row["post_call_state"] for row in rows),
        "open_hand_rows": sum(row["open_hand_state"] for row in rows),
        "win_competition_rows": sum(row["call_candidates"] > 0 and row["win_mass_without_gate"] > 0 for row in rows),
        "total_win_probability_mass_lost": sum(row["win_mass_lost"] for row in rows),
        "elapsed_seconds": time.monotonic() - started,
    }
    return MatchTrajectory(record, encoded_rows)


def persist_match(trajectory: MatchTrajectory, directory: str | Path) -> dict[str, Any]:
    """Persist once, then sync both artifacts and their parent before returning.

    A controller may write a durable "persisted" marker only after this returns.
    The original exclusive artifact writes retain collisions and partial writes as
    evidence; they are never overwritten or retried by this helper.
    """
    record = trajectory.record
    role = record.get("collection_role")
    expected_eligibility = role == "training" and record.get("completed") is True
    if (record.get("collector_schema") != COLLECTOR_SCHEMA
            or role not in ("training", "evaluation")
            or type(record.get("completed")) is not bool
            or record.get("training_eligible") is not expected_eligibility
            or any(row.get("training_eligible") is not expected_eligibility
                   for row in record.get("learner_rows", []))):
        raise ValueError("Expected a Stage 3E trajectory with completed-training eligibility")
    directory = Path(directory).resolve()
    directories = [directory]
    ancestor = directory
    while not ancestor.exists():
        ancestor = ancestor.parent
        directories.append(ancestor)
    artifacts = _persist_match(trajectory, directory)
    for key in ("npz_path", "json_path"):
        with Path(artifacts[key]).open("rb") as stream:
            os.fsync(stream.fileno())
    # Sync newly created directory links all the way to the first previously
    # existing parent, in addition to the directory entries for both files.
    for parent in directories:
        descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return artifacts
