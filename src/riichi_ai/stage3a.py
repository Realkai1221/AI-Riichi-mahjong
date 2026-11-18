"""Small complete-match REINFORCE engineering pilot, with explicit provenance.

The collector owns engine observations/logs. The actor only receives the existing
audited player encoding. This module makes no playing-strength claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .auditing import PlayerViewHasher, VIEW_HASH_SCHEMA
from .evaluation import RANK_UTILITY
from .features import encode_view
from .model import NeuralPolicy, PolicyNetwork, _weights_hash, collate_encoded
from .player import PlayerAdapter, PlayerView
from .rules import audit_physical_tiles, rule_profile_hash
from .semantics import semantic_actions
from .simulator import INITIAL_SCORES, create_env, score_total


SCHEMA = "stage3a-complete-match-reinforce-v1"
SAMPLER = "python-random-float64-cdf-softmax-v1"


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _tree_hash(value: Any) -> str:
    """Stable hash of optimizer/RNG structures, without pickle byte assumptions."""
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(_json(["tensor", str(tensor.dtype), list(tensor.shape)]).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"dict[")
            for key in sorted(item, key=lambda x: (type(x).__name__, str(x))):
                visit(key)
                visit(item[key])
            digest.update(b"]")
        elif isinstance(item, (list, tuple)):
            digest.update((type(item).__name__ + "[").encode())
            for child in item:
                visit(child)
            digest.update(b"]")
        else:
            digest.update(_json([type(item).__name__, item]).encode())

    visit(value)
    return digest.hexdigest()


def encoded_hash(encoded: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(encoded.items()):
        array = np.ascontiguousarray(value)
        digest.update(_json([key, str(array.dtype), list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Stage 3A cooperative operation deadline reached")


@dataclass
class TrainingState:
    model: PolicyNetwork
    optimizer: torch.optim.Optimizer
    protocol_sha256: str
    reference_sha256: str
    device: str = "cpu"
    updates: int = 0
    retired_batches: list[int] = field(default_factory=list)
    optimizer_config: dict[str, Any] = field(default_factory=dict)
    optimizer_steps_executed: int = 0

    @property
    def policy_id(self) -> str:
        return f"{SCHEMA}:update-{self.updates}:{_weights_hash(self.model)}"


@dataclass
class MatchTrajectory:
    record: dict[str, Any]
    encoded: list[dict[str, np.ndarray]]


def _freeze_rank_head(model: PolicyNetwork) -> None:
    for parameter in model.rank_head.parameters():
        parameter.requires_grad_(False)
    model.eval()


def state_manifest(state: TrainingState) -> dict[str, Any]:
    return {
        "schema": SCHEMA, "protocol_sha256": state.protocol_sha256,
        "reference_sha256": state.reference_sha256, "updates": state.updates,
        "optimizer_steps_executed": state.optimizer_steps_executed,
        "retired_batches": list(state.retired_batches), "device": state.device,
        "weights_sha256": _weights_hash(state.model),
        "optimizer_sha256": _tree_hash(state.optimizer.state_dict()),
        "rank_head_sha256": _tree_hash(state.model.rank_head.state_dict()),
        "policy_id": state.policy_id, "model_config": dict(state.model.config),
        "optimizer_config": dict(state.optimizer_config), "sampler": SAMPLER,
    }


def initialize(reference_path: str | Path, expected_sha256: str, *,
               protocol_sha256: str, device: str = "cpu", learning_rate: float = 1e-5,
               weight_decay: float = 0.0) -> TrainingState:
    if not math.isfinite(learning_rate) or learning_rate <= 0 or weight_decay != 0:
        raise ValueError("Stage 3A requires positive learning rate and zero weight decay")
    reference = NeuralPolicy.from_checkpoint(reference_path, device=device)
    if reference.checkpoint_hash != expected_sha256:
        raise ValueError("Reference checkpoint hash mismatch")
    _freeze_rank_head(reference.model)
    optimizer = torch.optim.AdamW(
        [p for p in reference.model.parameters() if p.requires_grad],
        lr=learning_rate, weight_decay=weight_decay, betas=(.9, .999), eps=1e-8,
        foreach=False,
    )
    return TrainingState(reference.model, optimizer, protocol_sha256,
                         expected_sha256, str(device), optimizer_config={
                             "name": "AdamW", "learning_rate": learning_rate,
                             "weight_decay": weight_decay, "betas": [.9, .999],
                             "eps": 1e-8, "foreach": False, "fresh_optimizer": True,
                         })


def save_boundary(state: TrainingState, path: str | Path) -> dict[str, Any]:
    """Exclusive immutable checkpoint; only initial/retired-batch boundaries."""
    if (state.retired_batches != list(range(state.updates))
            or state.optimizer_steps_executed != state.updates):
        raise ValueError("Checkpoint is not at a complete sequential batch boundary")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        **state_manifest(state),
        "model_state_dict": {k: v.detach().cpu().clone() for k, v in state.model.state_dict().items()},
        "optimizer_state_dict": state.optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(), "python_rng_state": random.getstate(),
        "device_rng_state": (torch.mps.get_rng_state() if state.device.startswith("mps")
                             else torch.cuda.get_rng_state(state.device) if state.device.startswith("cuda")
                             else None),
        "sampling_rng": "Each prespecified match has its own recorded random.Random seed and start/end state.",
    }
    with path.open("xb") as stream:
        torch.save(checkpoint, stream)
    return {**state_manifest(state), "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load_boundary(path: str | Path, *, expected_protocol_sha256: str,
                  device: str = "cpu") -> TrainingState:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (checkpoint.get("schema") != SCHEMA
            or checkpoint.get("protocol_sha256") != expected_protocol_sha256):
        raise ValueError("Boundary checkpoint protocol mismatch")
    if checkpoint.get("device") != str(device):
        raise ValueError("Exact boundary resume requires the declared original device")
    with torch.random.fork_rng(devices=[]):
        model = PolicyNetwork(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    _freeze_rank_head(model)
    settings = checkpoint["optimizer_config"]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=settings["learning_rate"], weight_decay=settings["weight_decay"],
                                 betas=tuple(settings["betas"]), eps=settings["eps"], foreach=False)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    state = TrainingState(model, optimizer, checkpoint["protocol_sha256"],
                          checkpoint["reference_sha256"], str(device), checkpoint["updates"],
                          checkpoint["retired_batches"], settings, checkpoint["optimizer_steps_executed"])
    if (state.retired_batches != list(range(state.updates))
            or state.optimizer_steps_executed != state.updates):
        raise ValueError("Checkpoint retired batches are not sequential")
    for key in ("weights_sha256", "optimizer_sha256", "rank_head_sha256", "policy_id"):
        if state_manifest(state)[key] != checkpoint[key]:
            raise ValueError(f"Boundary checkpoint {key} mismatch")
    torch.set_rng_state(checkpoint["torch_rng_state"])
    random.setstate(checkpoint["python_rng_state"])
    if state.device.startswith("mps"):
        torch.mps.set_rng_state(checkpoint["device_rng_state"])
    elif state.device.startswith("cuda"):
        torch.cuda.set_rng_state(checkpoint["device_rng_state"], state.device)
    return state


def normalized_probabilities(logits: torch.Tensor) -> np.ndarray:
    """Float64 CPU softmax from unmodified model logits; no temperature change."""
    if logits.ndim != 1 or not len(logits) or not bool(torch.isfinite(logits).all()):
        raise ValueError("Expected a finite nonempty legal-logit vector")
    values = torch.softmax(logits.detach().to("cpu", dtype=torch.float64), dim=0).numpy()
    total = float(values.sum())
    if not np.isfinite(values).all() or total <= 0:
        raise ValueError("Non-finite or empty categorical distribution")
    return values / total


def sample_categorical(probabilities: np.ndarray, rng: random.Random) -> tuple[int, float | None]:
    if (probabilities.ndim != 1 or not len(probabilities) or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0) or not math.isclose(float(probabilities.sum()), 1, abs_tol=1e-12)):
        raise ValueError("Invalid normalized categorical probabilities")
    if len(probabilities) == 1:
        return 0, None
    draw = rng.random()
    # right-side search prevents selecting zero-probability candidates at u=0.
    index = int(np.searchsorted(np.cumsum(probabilities), draw, side="right"))
    if index == len(probabilities):
        index = int(np.flatnonzero(probabilities > 0)[-1])
    if probabilities[index] <= 0:
        raise ValueError("Categorical sampler selected a zero-probability action")
    return index, draw


def actor_decision(model: PolicyNetwork, view: PlayerView, rng: random.Random,
                   *, device: str, policy_id: str, view_hash: str) -> tuple[Any, dict, dict]:
    actions = semantic_actions(view)
    if not actions:
        raise ValueError("Learner request has no legal semantic candidates")
    encoded = encode_view(view, actions, feature_schema=model.config["feature_schema"])
    if len(actions) == 1:
        probabilities = np.ones(1, dtype=np.float64)
    else:
        with torch.inference_mode():
            logits, _ = model(collate_encoded([encoded], device))
            probabilities = normalized_probabilities(logits[0])
    selected, draw = sample_categorical(probabilities, rng)
    nonzero = probabilities[probabilities > 0]
    metadata = {
        "seat": view.seat, "hand_id": view.hand_id, "policy_id": policy_id,
        "player_view_sha256": view_hash, "encoded_sha256": encoded_hash(encoded),
        "semantic_keys": [a.key for a in actions],
        "representative_tokens": [a.representative.token for a in actions],
        "selected_index": selected, "selected_semantic_key": actions[selected].key,
        "selected_physical_token": actions[selected].representative.token,
        "probabilities": probabilities.tolist(), "probability": float(probabilities[selected]),
        "log_probability": math.log(float(probabilities[selected])),
        "entropy": float(-(nonzero * np.log(nonzero)).sum()),
        "uniform_draw": draw, "forced": len(actions) == 1,
        "fallback": False, "overridden": False, "eligible": True,
    }
    return actions[selected].representative, metadata, encoded


def reconstruct_log_scores(log: Sequence[Mapping[str, Any]]) -> tuple[list[int], dict[str, Any]]:
    """Reconstruct pinned MJAI scoring events, including final riichi deposits.

    RiichiEnv 0.4.10 emits a bare end_game marker. Initial/start-hand scores,
    accepted riichi payments, and hand-result deltas carry the score evidence.
    Final outstanding deposits go to first place (the existing rule fixture
    test_final_draw_outstanding_deposit_goes_to_top covers this engine rule).
    """
    if (not log or log[-1].get("type") != "end_game"
            or sum(e.get("type") == "end_game" for e in log) != 1):
        raise ValueError("Terminal log must end in exactly one end_game marker")
    running = None
    deposits = 0
    hand_riichi = [0] * 4
    log_delta_corrections = []
    hands = payments = results = 0
    for event in log:
        kind = event.get("type")
        if kind == "start_kyoku":
            values, sticks = event.get("scores"), event.get("kyotaku")
            if (not isinstance(values, (list, tuple)) or len(values) != 4
                    or any(type(value) is not int for value in values)
                    or type(sticks) is not int or sticks < 0):
                raise ValueError("Malformed starting-hand score evidence")
            if running is not None and (running != list(values) or deposits != sticks * 1000):
                raise ValueError("Adjacent hand score/deposit evidence is inconsistent")
            running, deposits = list(values), sticks * 1000
            hand_riichi = [0] * 4
            hands += 1
        elif kind == "reach_accepted":
            actor = event.get("actor")
            if running is None or type(actor) is not int or actor not in range(4):
                raise ValueError("Malformed accepted riichi evidence")
            running[actor] -= 1000
            deposits += 1000
            if hand_riichi[actor]:
                raise ValueError("Repeated accepted riichi in one hand")
            hand_riichi[actor] = -1000
            payments += 1
        elif kind in ("hora", "ryukyoku"):
            deltas = event.get("deltas")
            if (running is None or not isinstance(deltas, (list, tuple)) or len(deltas) != 4
                    or any(type(value) is not int for value in deltas)):
                raise ValueError("Malformed hand-result score deltas")
            effective = list(deltas)
            if kind == "ryukyoku":
                # Pinned upstream state/mod.rs::_trigger_ryukyoku overwrites
                # score_delta for a 1--3 tenpai exchange, but leaves earlier
                # riichi deductions in it for a zero-exchange exhaustive draw;
                # nagashi adds payouts to that existing per-hand score_delta.
                # Keep raw logs intact and record this exact schema correction.
                reason = event.get("reason")
                if reason == "exhaustive_draw" and sum(deltas) < 0:
                    if list(deltas) != hand_riichi:
                        raise ValueError("Unexplained nonzero-sum exhaustive-draw deltas")
                    effective = [0] * 4
                elif reason == "nagashimangan":
                    effective = [delta - paid for delta, paid in zip(deltas, hand_riichi)]
                if sum(effective) != 0:
                    raise ValueError("Draw-result transfers must conserve player points")
                if effective != list(deltas):
                    log_delta_corrections.append({"hand": hands, "reason": reason,
                                                  "raw_deltas": list(deltas), "effective_deltas": effective})
            running = [score + delta for score, delta in zip(running, effective)]
            deposits -= sum(effective)
            if deposits < 0 or deposits % 1000:
                raise ValueError("Hand-result deltas disagree with outstanding deposits")
            results += 1
    if running is None:
        raise ValueError("Terminal log lacks starting-hand score evidence")
    recipient = min(range(4), key=lambda seat: (-running[seat], seat))
    terminal_deposit_award = deposits
    running[recipient] += deposits
    if "scores" in log[-1] and list(log[-1]["scores"]) != running:
        raise ValueError("Optional end_game scores disagree with reconstructed result")
    return running, {"method": "start-hand scores + accepted riichi + result deltas + final deposit award",
                     "hands": hands, "accepted_riichi": payments, "hand_results": results,
                     "upstream_log_delta_corrections": log_delta_corrections,
                     "terminal_deposit_award": terminal_deposit_award,
                     "terminal_deposit_recipient": recipient if terminal_deposit_award else None}


def terminal_result(scores: Sequence[int], engine_ranks: Sequence[int],
                    log: Sequence[Mapping[str, Any]], *, completed: bool) -> dict:
    if not completed:
        return {"ranks": None, "reconstructed_ranks": None, "utilities": None,
                "terminal_result_verified": False, "terminal_event": None,
                "reconstructed_scores": None, "terminal_score_reconstruction": None}
    if len(scores) != 4 or any(type(x) is not int for x in scores):
        raise ValueError("Invalid terminal scores")
    order = sorted(range(4), key=lambda seat: (-scores[seat], seat))
    reconstructed = [order.index(seat) + 1 for seat in range(4)]
    if list(engine_ranks) != reconstructed:
        raise ValueError("Engine ranks disagree with score/tie-priority reconstruction")
    reconstructed_scores, evidence = reconstruct_log_scores(log)
    if reconstructed_scores != list(scores):
        raise ValueError("Terminal log score chain does not confirm final scores")
    return {"ranks": list(engine_ranks), "reconstructed_ranks": reconstructed,
            "utilities": [RANK_UTILITY[rank - 1] for rank in reconstructed],
            "terminal_result_verified": True, "terminal_event": dict(log[-1]),
            "reconstructed_scores": reconstructed_scores, "terminal_score_reconstruction": evidence}


def collect_match(state: TrainingState, opponent: NeuralPolicy, *, batch_index: int,
                  match_index: int, seed: int, learner_seat: int, sampling_seed: int,
                  max_steps: int = 100_000, deadline: float | None = None) -> MatchTrajectory:
    if learner_seat not in range(4) or batch_index != state.updates or max_steps < 1:
        raise ValueError("Invalid learner seat, sequential batch index, or max_steps")
    if opponent.checkpoint_hash != state.reference_sha256:
        raise ValueError("Opponent is not the frozen original reference")
    policy_id = state.policy_id
    learner_manifest = {"policy_id": policy_id, "role": "learner", "sampler": SAMPLER,
                        "reference_sha256": state.reference_sha256,
                        "update": state.updates, "weights_sha256": _weights_hash(state.model),
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
    steps = 0
    failure = None
    started = time.monotonic()
    try:
        audit_physical_tiles(env)
        if score_total(env) != sum(INITIAL_SCORES):
            raise ValueError("Initial point conservation failure")
        while not env.done() and steps < max_steps:
            _deadline(deadline)
            # Build every immutable request from the SAME returned engine snapshot
            # before invoking any policy or submitting any response.
            views = {seat: adapters[seat].build(observations[seat]) for seat in sorted(observations)}
            if not views or not any(view.legal_actions for view in views.values()):
                raise ValueError("Unfinished match has no legal request")
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
                                "batch_index": batch_index})
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
            observations = env.step(native)
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
        if state.policy_id != policy_id:
            raise ValueError("Learner weights changed during fixed-policy collection")
        if _weights_hash(opponent.model) != opponent.weights_hash:
            raise ValueError("Frozen opponent weights changed")
    except Exception as error:
        completed = False
        failure = {"type": type(error).__name__, "message": str(error), "step": steps}
        result = terminal_result(scores, (), log, completed=False)
    record = {
        "schema": SCHEMA, "protocol_sha256": state.protocol_sha256,
        "batch_index": batch_index, "match_index": match_index, "seed": seed,
        "learner_seat": learner_seat, "sampling_seed": sampling_seed, "sampler": SAMPLER,
        "policy_id": policy_id, "policies": manifests,
        "rule_profile_sha256": rule_profile_hash(), "view_hash_schema": VIEW_HASH_SCHEMA,
        "completed": completed, "terminated": completed,
        "truncated": not completed and failure is None,
        "stop_reason": "failure" if failure else "completed" if completed else "max_steps",
        "failure": failure, "steps": steps, "max_steps": max_steps,
        "scores": scores, "riichi_sticks": env.riichi_sticks, **result,
        "learner_utility": result["utilities"][learner_seat] if completed else None,
        "learner_last_step": rows[-1]["step"] if rows else None,
        "opponent_steps_after_last_learner": steps - 1 - rows[-1]["step"] if rows else steps,
        "learner_rows": rows, "requests": trace,
        "sampling_rng_start": rng_start, "sampling_rng_end": rng.getstate(),
        "auditor_log": log, "auditor_log_sha256": _hash(log),
        "hands": sum(event.get("type") == "start_kyoku" for event in log),
        "conservation_verified": failure is None,
        "elapsed_seconds": time.monotonic() - started,
    }
    return MatchTrajectory(record, encoded_rows)


def persist_match(trajectory: MatchTrajectory, directory: str | Path) -> dict[str, Any]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"match-{trajectory.record['match_index']:03d}"
    arrays_path, record_path = directory / f"{stem}.npz", directory / f"{stem}.json"
    arrays = {f"row_{index:05d}__{key}": value for index, row in enumerate(trajectory.encoded)
              for key, value in row.items()}
    with arrays_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    array_hash = hashlib.sha256(arrays_path.read_bytes()).hexdigest()
    trajectory.record["encoded_artifact"] = {"path": str(arrays_path.resolve()),
                                              "sha256": array_hash, "rows": len(trajectory.encoded)}
    with record_path.open("x") as stream:
        stream.write(_json(trajectory.record) + "\n")
    return {"json_path": str(record_path.resolve()),
            "json_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
            "npz_path": str(arrays_path.resolve()), "npz_sha256": array_hash}


def _validate_batch(state: TrainingState, trajectories: Sequence[MatchTrajectory],
                    batch_index: int, planned_matches: Sequence[Mapping[str, Any]]) -> None:
    if batch_index != state.updates or batch_index in state.retired_batches:
        raise ValueError("Batch is out of order or already retired")
    if not planned_matches or len(trajectories) != len(planned_matches):
        raise ValueError("Every prespecified match is required; partial batches cannot update")
    if len({item["match_index"] for item in planned_matches}) != len(planned_matches):
        raise ValueError("Duplicate match in frozen plan")
    policy_id = state.policy_id
    for trajectory, plan in zip(trajectories, planned_matches):
        record = trajectory.record
        for key in ("match_index", "seed", "learner_seat", "sampling_seed"):
            if record[key] != plan[key]:
                raise ValueError(f"Match does not match frozen plan: {key}")
        if (record["batch_index"] != batch_index or record["protocol_sha256"] != state.protocol_sha256
                or record["policy_id"] != policy_id or not record["completed"]
                or not record["terminated"] or record["truncated"] or record["failure"] is not None
                or not record["terminal_result_verified"]):
            raise ValueError("Incomplete, invalid or wrong-behavior match cannot update")
        terminal = terminal_result(record["scores"], record["ranks"], record["auditor_log"], completed=True)
        if record["learner_utility"] != terminal["utilities"][record["learner_seat"]]:
            raise ValueError("Learner utility disagrees with independently reconstructed terminal reward")
        if len(record["learner_rows"]) != len(trajectory.encoded):
            raise ValueError("Encoded learner row count mismatch")
        for row, encoded in zip(record["learner_rows"], trajectory.encoded):
            if (row["seat"] != record["learner_seat"] or row["policy_id"] != policy_id
                    or not row["eligible"] or row["fallback"] or row["overridden"]
                    or row["match_index"] != record["match_index"] or row["batch_index"] != batch_index):
                raise ValueError("Only this learner's unmodified behavior actions are eligible")
            if row["encoded_sha256"] != encoded_hash(encoded):
                raise ValueError("Stored actor encoding changed")
            n = len(encoded["action_tokens"])
            selected = row["selected_index"]
            probabilities = np.asarray(row["probabilities"], dtype=np.float64)
            if (n != len(row["semantic_keys"]) or n != len(row["representative_tokens"])
                    or probabilities.shape != (n,) or not 0 <= selected < n
                    or row["forced"] != (n == 1) or not np.isfinite(probabilities).all()
                    or np.any(probabilities < 0) or not math.isclose(float(probabilities.sum()), 1, abs_tol=1e-12)
                    or row["probability"] != probabilities[selected] or row["probability"] <= 0
                    or row["log_probability"] != math.log(row["probability"])):
                raise ValueError("Malformed behavior distribution or legal action identity")
            if (row["selected_semantic_key"] != row["semantic_keys"][selected]
                    or row["selected_physical_token"] != row["representative_tokens"][selected]):
                raise ValueError("Selected semantic/physical action identity mismatch")


def update_batch(state: TrainingState, trajectories: Sequence[MatchTrajectory], *, batch_index: int,
                 planned_matches: Sequence[Mapping[str, Any]], gradient_clip: float = 1.0,
                 replay_atol: float = 1e-6, replay_rtol: float = 1e-5,
                 replay_log_atol: float = 1e-4,
                 microbatch_size: int = 64, deadline: float | None = None) -> dict[str, Any]:
    """One on-policy score-function update, summed over actions and averaged by matches."""
    _validate_batch(state, trajectories, batch_index, planned_matches)
    if gradient_clip <= 0 or microbatch_size < 1 or not math.isfinite(gradient_clip):
        raise ValueError("Invalid update clipping or microbatch size")
    before = state_manifest(state)
    state.model.eval()
    optional = [(match_index, row, encoded, trajectory.record["learner_utility"])
                for match_index, trajectory in enumerate(trajectories)
                for row, encoded in zip(trajectory.record["learner_rows"], trajectory.encoded)
                if not row["forced"]]
    match_sums = [0.0] * len(trajectories)
    stored_match_sums = [sum(row["log_probability"] for row in t.record["learner_rows"])
                         for t in trajectories]
    replay_max = 0.0
    replay_log_max = 0.0
    replayed = []
    # Verify ALL old probabilities before computing gradients or taking a step.
    with torch.inference_mode():
        for start in range(0, len(optional), microbatch_size):
            _deadline(deadline)
            chunk = optional[start:start + microbatch_size]
            logits, _ = state.model(collate_encoded([item[2] for item in chunk], state.device))
            for index, (match_index, row, encoded, _) in enumerate(chunk):
                n = len(encoded["action_tokens"])
                actual = normalized_probabilities(logits[index, :n])
                expected = np.asarray(row["probabilities"], dtype=np.float64)
                replay_max = max(replay_max, float(np.max(np.abs(actual - expected))))
                if not np.allclose(actual, expected, atol=replay_atol, rtol=replay_rtol):
                    raise ValueError("Pre-update behavior probability replay mismatch")
                selected = row["selected_index"]
                if actual[selected] <= 0:
                    raise ValueError("Replayed sampled action has zero probability")
                replay_log_max = max(replay_log_max, abs(math.log(float(actual[selected])) - row["log_probability"]))
                if replay_log_max > replay_log_atol:
                    raise ValueError("Pre-update selected log-probability replay mismatch")
                match_sums[match_index] += math.log(float(actual[selected]))
                replayed.append(actual)
    if not optional:
        raise ValueError("No optional learner actions; no policy-gradient update possible")
    old_parameters = {name: p.detach().cpu().clone() for name, p in state.model.named_parameters()}
    state.optimizer.zero_grad(set_to_none=True)
    numerical_loss = 0.0
    try:
        for start in range(0, len(optional), microbatch_size):
            _deadline(deadline)
            chunk = optional[start:start + microbatch_size]
            logits, _ = state.model(collate_encoded([item[2] for item in chunk], state.device))
            selected = torch.tensor([item[1]["selected_index"] for item in chunk])
            utilities = torch.tensor([item[3] for item in chunk], dtype=torch.float64)
            # The CPU float64 output transform is identical on all supported
            # devices; autograd carries this transfer back to model parameters.
            # Padding uses float32 minimum and therefore contributes zero mass.
            log_probability = torch.log_softmax(logits.to("cpu", dtype=torch.float64), dim=1).gather(1, selected[:, None]).squeeze(1)
            loss = -(log_probability * utilities).sum() / len(trajectories)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Non-finite policy-gradient loss")
            loss.backward()
            numerical_loss += float(loss.detach().cpu())
        parameters = [p for p in state.model.parameters() if p.requires_grad]
        if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters):
            raise ValueError("Non-finite learner gradient")
        norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip, error_if_nonfinite=True)
        gradient_norm = float(norm.detach().cpu())
        _deadline(deadline)
        state.optimizer.step()
        state.optimizer_steps_executed += 1
        if not all(bool(torch.isfinite(p).all()) for p in state.model.parameters()):
            raise ValueError("Non-finite learner parameter after update")
    except Exception:
        state.optimizer.zero_grad(set_to_none=True)
        raise
    state.updates += 1
    state.retired_batches.append(batch_index)
    after = state_manifest(state)
    deltas = [(p.detach().cpu().double() - old_parameters[name].double())
              for name, p in state.model.named_parameters() if p.requires_grad]
    max_delta = max(float(value.abs().max()) for value in deltas)
    l2_delta = math.sqrt(sum(float(value.square().sum()) for value in deltas))
    if before["rank_head_sha256"] != after["rank_head_sha256"]:
        raise ValueError("Frozen rank head changed")
    distribution_max = 0.0
    distribution_l1 = 0.0
    entropy_after = []
    # Final diagnostic is part of this operation; retain the completed update if
    # it cannot finish before the cooperative deadline, rather than pretending
    # that no optimizer step happened.
    diagnostic_complete = True
    with torch.inference_mode():
        for start in range(0, len(optional), microbatch_size):
            if deadline is not None and time.monotonic() >= deadline:
                diagnostic_complete = False
                break
            chunk = optional[start:start + microbatch_size]
            logits, _ = state.model(collate_encoded([item[2] for item in chunk], state.device))
            for index, (_, row, encoded, _) in enumerate(chunk):
                probabilities = normalized_probabilities(logits[index, :len(encoded["action_tokens"])])
                change = np.abs(probabilities - replayed[start + index])
                distribution_max = max(distribution_max, float(change.max()))
                distribution_l1 += float(change.sum())
                nonzero = probabilities[probabilities > 0]
                entropy_after.append(float(-(nonzero * np.log(nonzero)).sum()))
    behavioral_loss = -sum(t.record["learner_utility"] * total
                           for t, total in zip(trajectories, match_sums)) / len(trajectories)
    stored_loss = -sum(t.record["learner_utility"] * total
                      for t, total in zip(trajectories, stored_match_sums)) / len(trajectories)
    return {
        "schema": SCHEMA, "batch_index": batch_index, "completed_matches": len(trajectories),
        "planned_matches": [dict(plan) for plan in planned_matches], "optimizer_steps": 1,
        "baseline": 0.0, "discount": 1.0, "loss_denominator_matches": len(trajectories),
        "match_log_probability_sums": match_sums,
        "stored_behavior_match_log_probability_sums": stored_match_sums,
        "match_utilities": [t.record["learner_utility"] for t in trajectories],
        "behavior_reconstructed_loss": behavioral_loss, "differentiable_loss": numerical_loss,
        "stored_behavior_loss": stored_loss,
        "loss_reconstruction_note": "behavior_reconstructed_loss uses replayed probabilities; stored_behavior_loss uses original collection records.",
        "loss_numerical_difference": abs(numerical_loss - behavioral_loss),
        "optional_learner_rows": len(optional),
        "forced_learner_rows": sum(row["forced"] for t in trajectories for row in t.record["learner_rows"]),
        "opponent_rows_in_loss": 0, "probability_replay_max_absolute_error": replay_max,
        "probability_replay_atol": replay_atol, "probability_replay_rtol": replay_rtol,
        "selected_log_probability_replay_max_error": replay_log_max,
        "selected_log_probability_replay_atol": replay_log_atol,
        "gradient_norm_before_clip": gradient_norm, "gradient_clip": gradient_clip,
        "gradients_finite": True, "parameters_finite": True,
        "parameter_max_absolute_change": max_delta, "parameter_l2_change": l2_delta,
        "rank_head_unchanged": True, "distribution_diagnostic_complete": diagnostic_complete,
        "distribution_diagnostic_rows": len(entropy_after),
        "distribution_max_absolute_change": distribution_max,
        "distribution_mean_l1_change": distribution_l1 / len(entropy_after) if entropy_after else None,
        "mean_optional_entropy_before": float(np.mean([item[1]["entropy"] for item in optional])),
        "mean_optional_entropy_after": float(np.mean(entropy_after)) if entropy_after else None,
        "numerical_no_op": max_delta == 0 or distribution_max == 0,
        "before": before, "after": after,
        "interpretation": "Engineering update on complete-match utility; no strength inference.",
    }
