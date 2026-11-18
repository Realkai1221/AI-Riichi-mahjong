"""Frozen-base, trainable call selection; complete-match engineering pilot.

Only audited encoded player views enter the policy. Engine logs are collector
and terminal-auditor inputs, never gate or candidate-head features.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import stage3a as previous
from .auditing import PlayerViewHasher, VIEW_HASH_SCHEMA
from .features import FEATURE_SCHEMA_V2, TILE_CHANNELS, encode_view
from .evaluation import _wall_fingerprint
from .model import NeuralPolicy, PolicyNetwork, _weights_hash, collate_encoded
from .player import PlayerAdapter, PlayerView
from .rules import audit_physical_tiles, rule_profile_hash
from .semantics import ACTION_KINDS, semantic_actions
from .simulator import INITIAL_SCORES, create_env, score_total

SCHEMA = "stage3c-frozen-base-call-heads-v1"
SAMPLER = "python-random-float64-cdf-call-gate-v1"
CALL_KINDS = ("CHI", "PON", "DAIMINKAN", "ANKAN", "KAKAN")
CALL_IDS = tuple(ACTION_KINDS.index(kind) + 1 for kind in CALL_KINDS)
WIN_IDS = tuple(ACTION_KINDS.index(kind) + 1 for kind in ("RON", "TSUMO"))
HEAD_CONFIG = {"hidden_width": 64, "activation": "ReLU", "layer_norm_affine": False,
               "initial_call_mass": .05, "context_interface": "frozen-context-output",
               "candidate_interface": "frozen-action-encoding-plus-efficiency"}
_json, _hash, _tree_hash = previous._json, previous._hash, previous._tree_hash
encoded_hash, sample_categorical = previous.encoded_hash, previous.sample_categorical
terminal_result, persist_match = previous.terminal_result, previous.persist_match
MatchTrajectory = previous.MatchTrajectory


def _deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Stage 3C cooperative operation deadline reached")


def cpu_double(value: torch.Tensor) -> torch.Tensor:
    """Separate device transfer and dtype conversion (installed MPS build fix)."""
    return value.to(device="cpu").to(dtype=torch.float64)


class CallHeads(nn.Module):
    def __init__(self, context_width: int, candidate_width: int):
        super().__init__()
        self.context_norm = nn.LayerNorm(context_width, elementwise_affine=False)
        self.candidate_norm = nn.LayerNorm(candidate_width, elementwise_affine=False)
        self.gate = nn.Sequential(nn.Linear(context_width, 64), nn.ReLU(), nn.Linear(64, 1))
        self.call = nn.Sequential(nn.Linear(context_width + candidate_width, 64), nn.ReLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, math.log(.05 / .95))
        nn.init.zeros_(self.call[-1].weight)
        nn.init.zeros_(self.call[-1].bias)

    def forward(self, context: torch.Tensor, candidates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.context_norm(context.detach())
        candidates = self.candidate_norm(candidates.detach())
        combined = torch.cat((context.unsqueeze(1).expand(-1, candidates.shape[1], -1), candidates), dim=-1)
        return self.gate(context).squeeze(-1), self.call(combined).squeeze(-1)


class CallPolicy(nn.Module):
    def __init__(self, base: PolicyNetwork, initialization_seed: int):
        super().__init__()
        self.base = base
        self.config = dict(base.config)
        self.initialization_seed = initialization_seed
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initialization_seed)
            self.heads = CallHeads(self.config["context_width"], self.config["candidate_width"])
        self.heads.to(next(base.parameters()).device)
        self.eval()

    @property
    def rank_head(self):
        return self.base.rank_head

    def frozen_representations(self, batch: Mapping[str, torch.Tensor]):
        """Same operations as frozen PolicyNetwork.forward, exposing its two inputs.

        Deliberately explicit: no global mutation, observation access or hooks.
        A test compares the resulting base logits bit-for-bit to base.forward.
        """
        if batch["tiles"].shape[0] != 1:
            raise ValueError("Stage 3C requires single-row network execution")
        assisted = self.config["feature_schema"] == FEATURE_SCHEMA_V2
        if assisted != ("candidate_efficiency" in batch):
            raise ValueError("Batch feature schema differs from frozen model")
        base = self.base
        with torch.no_grad():
            tiles = batch["tiles"]
            count = tiles.shape[0]
            suits = base.suit_encoder(tiles[:, :27].reshape(count, 3, 9 * TILE_CHANNELS)).flatten(1)
            honors = base.honor_encoder(tiles[:, 27:].flatten(1))
            context = base.context(torch.cat((suits, honors, base.global_encoder(batch["globals"]),
                base._history(batch["history"], batch["history_mask"])), dim=-1))
            tokens = batch["action_tokens"]
            candidates = base.action_encoder(torch.cat((base.action_kind(tokens[..., 0]),
                base.tile_kind(tokens[..., 1]), base.red_flag(tokens[..., 2]),
                base.tsumogiri_flag(tokens[..., 3]), base.consumed_tile(tokens[..., 4:8]).flatten(-2)), dim=-1))
            if assisted:
                candidates = candidates + base.efficiency_encoder(batch["candidate_efficiency"])
            expanded = context.unsqueeze(1).expand(-1, candidates.shape[1], -1)
            logits = base.action_score(torch.cat((expanded, candidates), dim=-1)).squeeze(-1)
            logits = logits.masked_fill(~batch["candidate_mask"], torch.finfo(logits.dtype).min)
        return logits.detach(), context.detach(), candidates.detach()

    def distribution(self, encoded: Mapping[str, np.ndarray], device: str = "cpu") -> dict[str, torch.Tensor]:
        batch = collate_encoded([encoded], device)
        logits, context, candidates = self.frozen_representations(batch)
        gate, calls = self.heads(context, candidates)
        kinds = batch["action_tokens"][0, :, 0].to("cpu")
        return grouped_distribution(cpu_double(logits[0]), cpu_double(gate[0]),
                                    cpu_double(calls[0]), kinds)


def grouped_distribution(base_logits: torch.Tensor, gate_logit: torch.Tensor,
                         call_logits: torch.Tensor, kinds: torch.Tensor) -> dict[str, torch.Tensor]:
    """Stable legal-group normalization; actual q and differentiable log q."""
    if (base_logits.ndim != 1 or call_logits.shape != base_logits.shape or kinds.shape != base_logits.shape
            or not len(base_logits) or gate_logit.ndim != 0
            or not bool(torch.isfinite(base_logits).all()) or not bool(torch.isfinite(call_logits).all())
            or not bool(torch.isfinite(gate_logit))):
        raise ValueError("Expected finite aligned legal logits, kinds and a scalar gate")
    if any(value.device.type != "cpu" or value.dtype != torch.float64 for value in (base_logits, gate_logit, call_logits)):
        raise ValueError("Probability transforms require staged CPU float64 inputs")
    call_mask = torch.zeros_like(kinds, dtype=torch.bool)
    for kind in CALL_IDS:
        call_mask |= kinds == kind
    noncall = ~call_mask
    has_call, has_noncall = bool(call_mask.any()), bool(noncall.any())
    log_q = torch.empty_like(base_logits)
    conditional_noncall = torch.zeros_like(base_logits)
    conditional_calls = torch.zeros_like(base_logits)
    if has_call:
        values = F.log_softmax(call_logits[call_mask], dim=0)
        conditional_calls[call_mask] = values.exp()
        log_q[call_mask] = values + (F.logsigmoid(gate_logit) if has_noncall else 0.)
    if has_noncall:
        values = F.log_softmax(base_logits[noncall], dim=0)
        conditional_noncall[noncall] = values.exp()
        log_q[noncall] = values + (F.logsigmoid(-gate_logit) if has_call else 0.)
    # Renormalization is part of the declared sampler and differentiated rule.
    log_q = log_q - torch.logsumexp(log_q, dim=0)
    probabilities = log_q.exp()
    total = probabilities.sum()
    probabilities = probabilities / total
    # Retain stable finite logs even when an irrelevant noncall q underflows.
    log_q = log_q - torch.log(total)
    wins = torch.zeros_like(kinds, dtype=torch.bool)
    for kind in WIN_IDS:
        wins |= kinds == kind
    gate_mass = torch.sigmoid(gate_logit) if has_call and has_noncall else gate_logit.new_tensor(float(has_call))
    return {"probabilities": probabilities, "log_probabilities": log_q,
            "call_mask": call_mask, "conditional_noncall": conditional_noncall,
            "conditional_calls": conditional_calls, "call_mass": probabilities[call_mask].sum(),
            "gate_mass": gate_mass, "base_logits": base_logits,
            "win_mass_without_gate": conditional_noncall[wins].sum(),
            "win_mass_actual": probabilities[wins].sum(),
            "win_mass_lost": conditional_noncall[wins].sum() - probabilities[wins].sum()}


@dataclass
class TrainingState:
    model: CallPolicy
    optimizer: torch.optim.Optimizer
    protocol_sha256: str
    reference_sha256: str
    immutable_base_sha256: str
    device: str = "cpu"
    updates: int = 0
    retired_batches: list[int] = field(default_factory=list)
    optimizer_config: dict[str, Any] = field(default_factory=dict)
    optimizer_steps_executed: int = 0

    @property
    def policy_id(self) -> str:
        return f"{SCHEMA}:update-{self.updates}:{_weights_hash(self.model)}"


def state_manifest(state: TrainingState) -> dict[str, Any]:
    return {"schema": SCHEMA, "protocol_sha256": state.protocol_sha256,
        "reference_sha256": state.reference_sha256, "updates": state.updates,
        "optimizer_steps_executed": state.optimizer_steps_executed,
        "retired_batches": list(state.retired_batches), "device": state.device,
        "weights_sha256": _weights_hash(state.model), "heads_sha256": _weights_hash(state.model.heads),
        "base_weights_sha256": _weights_hash(state.model.base),
        "immutable_base_sha256": state.immutable_base_sha256,
        "base_unchanged": _weights_hash(state.model.base) == state.immutable_base_sha256,
        "optimizer_sha256": _tree_hash(state.optimizer.state_dict()),
        "rank_head_sha256": _tree_hash(state.model.rank_head.state_dict()),
        "policy_id": state.policy_id, "model_config": dict(state.model.config),
        "head_config": dict(HEAD_CONFIG), "initialization_seed": state.model.initialization_seed,
        "optimizer_config": dict(state.optimizer_config), "sampler": SAMPLER}


def _optimizer(model: CallPolicy, learning_rate: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.heads.parameters(), lr=learning_rate, weight_decay=0.,
                            betas=(.9, .999), eps=1e-8, foreach=False)


def initialize(reference_path: str | Path, expected_sha256: str, *, protocol_sha256: str,
               device: str = "cpu", initialization_seed: int = 310211,
               learning_rate: float = 1e-4, weight_decay: float = 0.) -> TrainingState:
    if learning_rate != 1e-4 or weight_decay != 0:
        raise ValueError("Stage 3C fixes AdamW learning rate 1e-4 and zero weight decay")
    if type(initialization_seed) is not int or not 0 <= initialization_seed < 2**63:
        raise ValueError("Invalid initialization seed")
    reference = NeuralPolicy.from_checkpoint(reference_path, device=device)
    if reference.checkpoint_hash != expected_sha256:
        raise ValueError("Reference checkpoint hash mismatch")
    model = CallPolicy(reference.model, initialization_seed)
    return TrainingState(model, _optimizer(model, learning_rate), protocol_sha256, expected_sha256,
        _weights_hash(model.base), str(device), optimizer_config={"name": "AdamW", "learning_rate": learning_rate,
        "weight_decay": 0., "betas": [.9, .999], "eps": 1e-8, "foreach": False,
        "fresh_optimizer": True, "parameter_scope": "new_heads_only"})


def save_boundary(state: TrainingState, path: str | Path) -> dict[str, Any]:
    manifest = state_manifest(state)
    if (state.retired_batches != list(range(state.updates)) or state.optimizer_steps_executed != state.updates
            or not manifest["base_unchanged"]):
        raise ValueError("Checkpoint requires complete sequential batches and immutable base")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {**manifest,
        "model_state_dict": {k: v.detach().cpu().clone() for k, v in state.model.state_dict().items()},
        "optimizer_state_dict": state.optimizer.state_dict(), "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "device_rng_state": (torch.mps.get_rng_state() if state.device.startswith("mps") else
                             torch.cuda.get_rng_state(state.device) if state.device.startswith("cuda") else None),
        "sampling_rng": "Each prespecified match records its separate random.Random start/end state."}
    with path.open("xb") as stream:
        torch.save(checkpoint, stream)
    return {**manifest, "path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load_boundary(path: str | Path, *, expected_protocol_sha256: str, device: str = "cpu") -> TrainingState:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (checkpoint.get("schema") != SCHEMA or checkpoint.get("protocol_sha256") != expected_protocol_sha256
            or checkpoint.get("head_config") != HEAD_CONFIG or checkpoint.get("device") != str(device)):
        raise ValueError("Boundary checkpoint protocol/architecture/device mismatch")
    with torch.random.fork_rng(devices=[]):
        model = CallPolicy(PolicyNetwork(checkpoint["model_config"]).to(device), checkpoint["initialization_seed"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    settings = checkpoint["optimizer_config"]
    optimizer = _optimizer(model, settings["learning_rate"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    state = TrainingState(model, optimizer, checkpoint["protocol_sha256"], checkpoint["reference_sha256"],
        checkpoint["immutable_base_sha256"], str(device), checkpoint["updates"], checkpoint["retired_batches"],
        settings, checkpoint["optimizer_steps_executed"])
    if (state.retired_batches != list(range(state.updates)) or state.optimizer_steps_executed != state.updates):
        raise ValueError("Checkpoint retired batches are not sequential")
    actual = state_manifest(state)
    if not actual["base_unchanged"]:
        raise ValueError("Frozen base checkpoint changed")
    for key in ("weights_sha256", "heads_sha256", "base_weights_sha256", "optimizer_sha256", "rank_head_sha256", "policy_id"):
        if actual[key] != checkpoint[key]:
            raise ValueError(f"Boundary checkpoint {key} mismatch")
    torch.set_rng_state(checkpoint["torch_rng_state"])
    random.setstate(checkpoint["python_rng_state"])
    if state.device.startswith("mps"):
        torch.mps.set_rng_state(checkpoint["device_rng_state"])
    elif state.device.startswith("cuda"):
        torch.cuda.set_rng_state(checkpoint["device_rng_state"], state.device)
    return state


def actor_decision(model: CallPolicy, view: PlayerView, rng: random.Random, *, device: str,
                   policy_id: str, view_hash: str) -> tuple[Any, dict, dict]:
    actions = semantic_actions(view)
    if not actions:
        raise ValueError("Learner request has no legal semantic candidates")
    encoded = encode_view(view, actions, feature_schema=model.config["feature_schema"])
    with torch.inference_mode():
        distribution = model.distribution(encoded, device)
    probabilities = distribution["probabilities"].numpy()
    selected, draw = sample_categorical(probabilities, rng)
    nonzero = probabilities[probabilities > 0]
    metadata = {"seat": view.seat, "hand_id": view.hand_id, "policy_id": policy_id,
        "player_view_sha256": view_hash, "encoded_sha256": encoded_hash(encoded),
        "semantic_keys": [a.key for a in actions], "action_kinds": [a.kind for a in actions],
        "representative_tokens": [a.representative.token for a in actions], "selected_index": selected,
        "selected_semantic_key": actions[selected].key, "selected_physical_token": actions[selected].representative.token,
        "selected_kind": actions[selected].kind, "probabilities": probabilities.tolist(),
        "probability": float(probabilities[selected]), "log_probability": math.log(float(probabilities[selected])),
        "entropy": float(-(nonzero * np.log(nonzero)).sum()), "uniform_draw": draw,
        "forced": len(actions) == 1, "fallback": False, "overridden": False, "eligible": True,
        "call_candidates": sum(a.kind in CALL_KINDS for a in actions),
        "call_mass": float(distribution["call_mass"]), "gate_mass": float(distribution["gate_mass"]),
        "win_mass_without_gate": float(distribution["win_mass_without_gate"]),
        "win_mass_actual": float(distribution["win_mass_actual"]), "win_mass_lost": float(distribution["win_mass_lost"]),
        "own_meld_count": len(view.melds[view.seat]),
        "own_open_meld_count": sum(meld.opened for meld in view.melds[view.seat]),
        "own_concealed_tile_count": len(view.hand),
        "post_call_state": bool(view.melds[view.seat]),
        "open_hand_state": any(meld.opened for meld in view.melds[view.seat])}
    return actions[selected].representative, metadata, encoded


def call_resolutions(log: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Auditor-only call outcomes; kan declarations can be robbed.

    A chi/pon event records an accepted meld. A kan counts as completed only
    once its caller receives the replacement draw before a win/hand boundary.
    This keeps a declaration/early meld upgrade from counting a robbed kakan.
    """
    result = []
    for index, event in enumerate(log):
        kind = event.get("type", "").upper()
        if kind not in CALL_KINDS:
            continue
        actor = event["actor"]
        completion = index if kind in ("CHI", "PON") else None
        resolved = True if completion is not None else None
        reason = "accepted meld event" if resolved else "pending replacement draw"
        if completion is None:
            for following in range(index + 1, len(log)):
                next_event = log[following]
                if next_event.get("type") == "tsumo" and next_event.get("actor") == actor:
                    completion, resolved, reason = following, True, "caller replacement draw"
                    break
                if next_event.get("type") in ("hora", "end_kyoku", "start_kyoku", "end_game"):
                    resolved, reason = False, "win or hand boundary before replacement draw"
                    break
        result.append({"event_index": index, "actor": actor, "kind": kind, "resolved": resolved,
                       "completion_event_index": completion, "reason": reason})
    return result


def collect_match(state: TrainingState, opponent: NeuralPolicy, *, batch_index: int,
                  match_index: int, seed: int, learner_seat: int, sampling_seed: int,
                  max_steps: int = 100_000, deadline: float | None = None,
                  collection_role: str = "training") -> MatchTrajectory:
    if (learner_seat not in range(4) or type(batch_index) is not int or batch_index < 0 or max_steps < 1
            or collection_role not in ("training", "evaluation")
            or (collection_role == "training" and batch_index != state.updates)):
        raise ValueError("Invalid learner seat, collection role, sequential batch index, or max_steps")
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
    resolutions = call_resolutions(log)
    for row in rows:
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
        "schema": SCHEMA, "protocol_sha256": state.protocol_sha256,
        "batch_index": batch_index, "match_index": match_index, "seed": seed,
        "collection_role": collection_role, "training_eligible": collection_role == "training",
        "eligibility_scope": "Learner behavior rows require match.training_eligible before optimization.",
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



def _gradient_report(model: CallPolicy) -> dict[str, Any]:
    components = {}
    for label, module in (("gate", model.heads.gate), ("call_head", model.heads.call)):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        components[label] = {"l2_norm": math.sqrt(sum(float(cpu_double(g).square().sum()) for g in gradients)),
                             "nonzero_elements": sum(int(torch.count_nonzero(g)) for g in gradients),
                             "finite": all(bool(torch.isfinite(g).all()) for g in gradients),
                             "hidden_layer_l2_norm": math.sqrt(sum(float(cpu_double(p.grad).square().sum())
                                 for p in module[0].parameters() if p.grad is not None))}
    return components


def update_batch(state: TrainingState, trajectories: Sequence[MatchTrajectory], *, batch_index: int,
                 planned_matches: Sequence[Mapping[str, Any]], gradient_clip: float = 1.,
                 replay_atol: float = 0., replay_rtol: float = 0., replay_log_atol: float = 0.,
                 microbatch_size: int = 1, deadline: float | None = None) -> dict[str, Any]:
    """One heads-only score-function update; sum actions, average full matches."""
    if any(trajectory.record.get("collection_role") != "training"
           or trajectory.record.get("training_eligible") is not True for trajectory in trajectories):
        raise ValueError("Evaluation or otherwise ineligible matches cannot train")
    previous._validate_batch(state, trajectories, batch_index, planned_matches)
    if (gradient_clip != 1. or microbatch_size != 1 or replay_atol != 0. or replay_rtol != 0.
            or replay_log_atol != 0.):
        raise ValueError("Stage 3C fixes single-row execution, exact replay and norm clip1")
    before = state_manifest(state)
    if not before["base_unchanged"]:
        raise ValueError("Frozen base changed before update")
    optional = [(match_index, row, encoded, trajectory.record["learner_utility"])
                for match_index, trajectory in enumerate(trajectories)
                for row, encoded in zip(trajectory.record["learner_rows"], trajectory.encoded)
                if not row["forced"]]
    if not optional:
        raise ValueError("No optional learner actions; no policy-gradient update possible")
    replayed = []
    match_sums = [0.] * len(trajectories)
    with torch.inference_mode():
        for match_index, row, encoded, _ in optional:
            _deadline(deadline)
            distribution = state.model.distribution(encoded, state.device)
            actual = distribution["probabilities"].numpy()
            expected = np.asarray(row["probabilities"], dtype=np.float64)
            if not np.array_equal(actual, expected):
                raise ValueError("Pre-update behavior probability replay mismatch")
            log_probability = math.log(float(actual[row["selected_index"]]))
            if log_probability != row["log_probability"]:
                raise ValueError("Pre-update selected log-probability replay mismatch")
            match_sums[match_index] += log_probability
            replayed.append(actual)
    old_parameters = {name: p.detach().cpu().clone() for name, p in state.model.heads.named_parameters()}
    state.optimizer.zero_grad(set_to_none=True)
    numerical_loss = 0.
    gradient_rows = 0
    try:
        for _, row, encoded, utility in optional:
            _deadline(deadline)
            distribution = state.model.distribution(encoded, state.device)
            loss = -distribution["log_probabilities"][row["selected_index"]] * utility / len(trajectories)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Non-finite policy-gradient loss")
            numerical_loss += float(loss.detach())
            if loss.requires_grad:
                loss.backward()
                gradient_rows += 1
        if not gradient_rows:
            raise ValueError("No legal-group decisions supply a trainable gradient")
        parameters = list(state.model.heads.parameters())
        components = _gradient_report(state.model)
        if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters):
            raise ValueError("Non-finite heads gradient")
        if any(p.grad is not None for p in state.model.base.parameters()):
            raise ValueError("Frozen base received gradients")
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True))
        _deadline(deadline)
        state.optimizer.step()
        state.optimizer_steps_executed += 1
        if not all(bool(torch.isfinite(p).all()) for p in parameters):
            raise ValueError("Non-finite heads parameter after update")
    except Exception:
        state.optimizer.zero_grad(set_to_none=True)
        raise
    state.updates += 1
    state.retired_batches.append(batch_index)
    after = state_manifest(state)
    if not after["base_unchanged"] or before["rank_head_sha256"] != after["rank_head_sha256"]:
        raise ValueError("Frozen base or rank head changed")
    deltas = {name: cpu_double(p.detach()) - old_parameters[name].double()
              for name, p in state.model.heads.named_parameters()}
    max_delta = max(float(value.abs().max()) for value in deltas.values())
    l2_delta = math.sqrt(sum(float(value.square().sum()) for value in deltas.values()))
    distribution_max = distribution_l1 = 0.
    entropy_after, call_mass_after = [], []
    diagnostic_complete = True
    with torch.inference_mode():
        for index, (_, row, encoded, _) in enumerate(optional):
            if deadline is not None and time.monotonic() >= deadline:
                diagnostic_complete = False
                break
            distribution = state.model.distribution(encoded, state.device)
            probabilities = distribution["probabilities"].numpy()
            change = np.abs(probabilities - replayed[index])
            distribution_max = max(distribution_max, float(change.max()))
            distribution_l1 += float(change.sum())
            nonzero = probabilities[probabilities > 0]
            entropy_after.append(float(-(nonzero * np.log(nonzero)).sum()))
            if row["call_candidates"]:
                call_mass_after.append(float(distribution["call_mass"]))
    loss_value = -sum(t.record["learner_utility"] * total
                      for t, total in zip(trajectories, match_sums)) / len(trajectories)
    return {"schema": SCHEMA, "batch_index": batch_index, "completed_matches": len(trajectories),
        "planned_matches": [dict(plan) for plan in planned_matches], "optimizer_steps": 1,
        "baseline": 0., "discount": 1., "loss_denominator_matches": len(trajectories),
        "match_log_probability_sums": match_sums,
        "stored_behavior_match_log_probability_sums": match_sums,
        "match_utilities": [t.record["learner_utility"] for t in trajectories],
        "behavior_reconstructed_loss": loss_value, "stored_behavior_loss": loss_value,
        "differentiable_loss": numerical_loss, "loss_numerical_difference": abs(numerical_loss - loss_value),
        "optional_learner_rows": len(optional), "gradient_eligible_rows": gradient_rows,
        "forced_learner_rows": sum(row["forced"] for t in trajectories for row in t.record["learner_rows"]),
        "opponent_rows_in_loss": 0, "probability_replay_max_absolute_error": 0.,
        "probability_replay_atol": 0., "probability_replay_rtol": 0.,
        "selected_log_probability_replay_max_error": 0., "selected_log_probability_replay_atol": 0.,
        "gradient_norm_before_clip": gradient_norm, "gradient_clip": 1., "gradient_components": components,
        "gradients_finite": True, "parameters_finite": True, "base_unchanged": True, "rank_head_unchanged": True,
        "parameter_max_absolute_change": max_delta, "parameter_l2_change": l2_delta,
        "parameter_component_l2_change": {prefix: math.sqrt(sum(float(value.square().sum())
            for name, value in deltas.items() if name.startswith(prefix))) for prefix in ("gate", "call")},
        "distribution_diagnostic_complete": diagnostic_complete, "distribution_diagnostic_rows": len(entropy_after),
        "distribution_max_absolute_change": distribution_max,
        "distribution_mean_l1_change": distribution_l1 / len(entropy_after) if entropy_after else None,
        "mean_optional_entropy_before": float(np.mean([item[1]["entropy"] for item in optional])),
        "mean_optional_entropy_after": float(np.mean(entropy_after)) if entropy_after else None,
        "mean_call_mass_after_on_call_rows": float(np.mean(call_mass_after)) if call_mass_after else None,
        "numerical_no_op": max_delta == 0 or distribution_max == 0, "before": before, "after": after,
        "interpretation": "Heads-only engineering update on complete-match utility; no strength inference."}


def prepare(state: TrainingState) -> dict[str, Any]:
    """Finite gradient/replay probe on one explicit-wall multiple-chi request.

    No optimizer update or full match. The wall is a targeted local fixture,
    unrelated to training/evaluation seed plans and never fed to the actor.
    """
    from .rules import wall_for_hands
    from .player import action_spec
    started = time.monotonic()
    before = state_manifest(state)
    env, _ = create_env(0)
    observations = env.reset(wall=wall_for_hands({1: "1240m123p456s123z"}, draws=["3m"]),
        seed=0, oya=0, round_wind=0, honba=0, kyotaku=0, scores=[25000] * 4)
    adapters = [PlayerAdapter(seat) for seat in range(4)]
    for seat, observation in observations.items():
        adapters[seat].build(observation)
    discard = next(action for action in observations[0].legal_actions()
                   if action_spec(action).kind == "DISCARD" and action.tile == env.drawn_tile)
    observations = env.step({0: discard})
    view = adapters[1].build(observations[1])
    encoded = encode_view(view, feature_schema=state.model.config["feature_schema"])
    batch = collate_encoded([encoded], state.device)
    with torch.inference_mode():
        original, _ = state.model.base(batch)
        exposed, _, _ = state.model.frozen_representations(batch)
        if not torch.equal(original, exposed):
            raise ValueError("Frozen representation path changes base logits")
        first = state.model.distribution(encoded, state.device)
        first_probabilities = first["probabilities"].numpy().copy()
        timings = []
        for _ in range(16):
            tick = time.monotonic()
            actual = state.model.distribution(encoded, state.device)["probabilities"].numpy()
            timings.append(time.monotonic() - tick)
            if not np.array_equal(first_probabilities, actual):
                raise ValueError("Preparation probability replay is not exact")
    call_indices = np.flatnonzero(first["call_mask"].numpy())
    if len(call_indices) < 2 or len(call_indices) == len(first_probabilities):
        raise ValueError("Preparation fixture needs multiple calls and a noncall")
    state.optimizer.zero_grad(set_to_none=True)
    distribution = state.model.distribution(encoded, state.device)
    (-distribution["log_probabilities"][int(call_indices[0])]).backward()
    components = _gradient_report(state.model)
    if not all(entry["finite"] and entry["l2_norm"] > 0 for entry in components.values()):
        raise ValueError("Direct selected-call gradient failed to reach both heads")
    if state.updates == 0 and any(entry["hidden_layer_l2_norm"] != 0 for entry in components.values()):
        raise ValueError("Zero final layers should initially block hidden-layer gradients")
    if any(parameter.grad is not None for parameter in state.model.base.parameters()):
        raise ValueError("Direct probe leaked a gradient into the frozen base")
    state.optimizer.zero_grad(set_to_none=True)
    after = state_manifest(state)
    if before != after:
        raise ValueError("Preparation mutated model/optimizer state")
    return {"schema": SCHEMA, "valid": True, "base_logit_path_exact": True,
        "probability_replay_max_absolute_error": 0., "repeated_rows": 16,
        "legal_candidates": len(first_probabilities), "call_candidates": len(call_indices),
        "initial_call_mass": float(first["call_mass"]),
        "probabilities": first_probabilities.tolist(), "gradient_components": components,
        "mean_single_row_seconds": float(np.mean(timings)), "maximum_single_row_seconds": max(timings),
        "optimizer_steps": 0, "new_complete_matches": 0, "state_unchanged": True,
        "elapsed_seconds": time.monotonic() - started, "state": before}
