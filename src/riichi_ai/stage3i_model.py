"""Immutable reference representations and explicitly bounded residual choices.

The new heads are predictive components, not identified causal advantages.
Only audited actor tensors enter the extractor; labels and provenance belong
to the separate trainer. The historical actor and call-head files are never
modified and no optimizer is constructed by this module.
"""
from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .features import FEATURE_SCHEMA_V2, TILE_CHANNELS
from .model import PolicyNetwork, _state_hash, _weights_hash
from .semantics import ACTION_KINDS
from .stage3c import CallPolicy, HEAD_CONFIG as CALL_HEAD_CONFIG, SCHEMA as CALL_SCHEMA
from .stage3c import cpu_double, grouped_distribution


SCHEMA = "riichi-ai-stage3i-frozen-reference-v1"
BUNDLE_SCHEMA = "riichi-ai-stage3i-head-bundle-v1"
HEAD_CONFIG = {"state_width": 256, "action_width": 128, "hidden_width": 64}
EXPLORATION = .05
ELIGIBILITY_THRESHOLD = .02
PRIOR_FLOOR = -8.
WIN_IDS = tuple(ACTION_KINDS.index(name) + 1 for name in ("RON", "TSUMO"))


def _freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return module


def _sha_bytes(path: str | Path, expected: str) -> bytes:
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Frozen checkpoint file hash mismatch")
    return raw


class FrozenReference(nn.Module):
    """One actor forward shared with the immutable collection call heads."""

    def __init__(self, base: PolicyNetwork, call_heads: nn.Module, metadata: Mapping[str, Any]):
        super().__init__()
        self.base = _freeze(base)
        self.call_heads = _freeze(call_heads)
        self.config = dict(base.config)
        self.metadata = dict(metadata)
        self.train(False)

    def train(self, mode: bool = True):
        # Calling train() on a containing model must never unfreeze the actor.
        return super().train(False)

    @property
    def actor(self) -> PolicyNetwork:
        return self.base

    @classmethod
    def from_checkpoints(cls, reference_path: str | Path, reference_sha256: str,
                         call_path: str | Path, call_sha256: str,
                         device: str = "cpu") -> "FrozenReference":
        reference_raw = _sha_bytes(reference_path, reference_sha256)
        call_raw = _sha_bytes(call_path, call_sha256)
        saved = torch.load(io.BytesIO(call_raw), map_location="cpu", weights_only=True)
        if (saved.get("schema") != CALL_SCHEMA or saved.get("head_config") != CALL_HEAD_CONFIG
                or saved.get("reference_sha256") != reference_sha256
                or saved.get("updates") != 0 or saved.get("optimizer_steps_executed") != 0
                or saved.get("retired_batches") != []):
            raise ValueError("Expected the declared unchanged Stage 3C initialization")
        # Restore caller CPU RNG; neither loading nor construction mutates it.
        with torch.random.fork_rng(devices=[]):
            reference_saved = torch.load(io.BytesIO(reference_raw), map_location="cpu", weights_only=True)
            base = PolicyNetwork(reference_saved["model_config"])
            base.load_state_dict(reference_saved["model_state_dict"], strict=True)
            call = CallPolicy(PolicyNetwork(saved["model_config"]), saved["initialization_seed"])
        call.load_state_dict(saved["model_state_dict"], strict=True)
        base_sha = _weights_hash(base)
        if (base.config != call.config or _weights_hash(call.base) != base_sha
                or _weights_hash(call) != saved.get("weights_sha256")
                or _weights_hash(call.heads) != saved.get("heads_sha256")
                or base_sha != saved.get("base_weights_sha256")
                or base_sha != saved.get("immutable_base_sha256")
                or base.config["context_width"] != HEAD_CONFIG["state_width"]
                or base.config["candidate_width"] != HEAD_CONFIG["action_width"]
                or base.config["feature_schema"] != FEATURE_SCHEMA_V2):
            raise ValueError("Frozen actor, collection heads or architecture mismatch")
        if any(not bool(torch.isfinite(p).all()) for p in (*base.parameters(), *call.heads.parameters())):
            raise ValueError("Nonfinite frozen model parameters")
        metadata = {"schema": SCHEMA, "reference_sha256": reference_sha256,
                    "call_sha256": call_sha256, "base_weights_sha256": base_sha,
                    "call_heads_sha256": _weights_hash(call.heads),
                    "call_initialization_seed": saved["initialization_seed"],
                    "source_updates": 0, "feature_schema": base.config["feature_schema"]}
        return cls(base, call.heads, metadata).to(device).eval()

    def frozen_representations(self, batch: Mapping[str, torch.Tensor]):
        """Historic forward operations, exposing detached context and candidates.

        Arbitrary batch size is available for feature preparation. Deployment
        and exact actor identity checks use CPU batch one.
        """
        assisted = self.config["feature_schema"] == FEATURE_SCHEMA_V2
        expected = {"tiles", "globals", "history", "history_mask", "action_tokens", "candidate_mask"}
        if assisted:
            expected.add("candidate_efficiency")
        if set(batch) != expected:
            raise ValueError("Extractor accepts audited actor tensor fields only")
        base = self.base
        with torch.no_grad():
            tiles = batch["tiles"]
            count = tiles.shape[0]
            if count < 1:
                raise ValueError("Empty actor batch")
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

    def explorer_probabilities(self, logits: torch.Tensor, context: torch.Tensor,
                               candidates: torch.Tensor, kinds: torch.Tensor,
                               mask: torch.Tensor) -> np.ndarray:
        """Exact collection marginal, staged through CPU float64 arithmetic.

        The existing call policy normalizes its two groups on CPU float64.
        Collection then applies NumPy normalization, the semantic-uniform
        mixture and another normalization. Keep every one of these operations.
        """
        if (logits.ndim != 2 or kinds.shape != logits.shape or mask.shape != logits.shape
                or mask.dtype != torch.bool or context.shape[0] != logits.shape[0]
                or candidates.shape[:2] != logits.shape):
            raise ValueError("Misaligned explorer feature tensors")
        with torch.no_grad():
            gate, calls = self.call_heads(context, candidates)
        legal = mask.detach().cpu().numpy()
        result = np.zeros(tuple(logits.shape), dtype=np.float64)
        kinds_cpu = kinds.detach().cpu()
        for row in range(len(result)):
            indices = np.flatnonzero(legal[row])
            if not len(indices):
                raise ValueError("Explorer row has no legal candidates")
            if len(indices) == 1:
                result[row, indices[0]] = 1.
                continue
            ix = torch.from_numpy(indices).to(logits.device)
            local_kinds = kinds_cpu[row, indices]
            distribution = grouped_distribution(cpu_double(logits[row, ix]), cpu_double(gate[row]),
                                                cpu_double(calls[row, ix]), local_kinds)
            base = distribution["probabilities"].detach().numpy()
            result[row, indices] = explorer_marginal(base, local_kinds.numpy())
        return result

    def assert_unchanged(self) -> None:
        if (self.training or self.base.training or self.call_heads.training
                or any(p.requires_grad or p.grad is not None for p in self.parameters())
                or _weights_hash(self.base) != self.metadata["base_weights_sha256"]
                or _weights_hash(self.call_heads) != self.metadata["call_heads_sha256"]):
            raise ValueError("Immutable reference or collection heads changed")


def _normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or np.any(values < 0) or not math.isclose(float(values.sum()), 1., rel_tol=0, abs_tol=1e-12)):
        raise ValueError("Expected normalized finite explorer probabilities")
    return values / values.sum()


def explorer_marginal(base_probabilities: np.ndarray, kinds: np.ndarray) -> np.ndarray:
    """Collection's exact first-win or 95/5 semantic-uniform marginal."""
    base = _normalized(base_probabilities)
    kinds = np.asarray(kinds)
    if kinds.shape != base.shape:
        raise ValueError("Explorer kinds differ from probabilities")
    wins = np.flatnonzero(np.isin(kinds, WIN_IDS))
    if len(wins):
        result = np.zeros_like(base)
        result[wins[0]] = 1.
        return result
    uniform = np.full(len(base), 1. / len(base), dtype=np.float64)
    return _normalized((1. - EXPLORATION) * base + EXPLORATION * uniform)


def first_win_index(kinds: np.ndarray, fallback: int) -> int:
    """The same first-legal-win wrapper for reference and corrected policies."""
    kinds = np.asarray(kinds)
    if kinds.ndim != 1 or type(fallback) is not int or not 0 <= fallback < len(kinds):
        raise ValueError("Invalid first-win wrapper input")
    wins = np.flatnonzero(np.isin(kinds, WIN_IDS))
    return int(wins[0]) if len(wins) else fallback


class _DetachedScalarHead(nn.Module):
    def __init__(self, input_width: int, hidden_width: int):
        super().__init__()
        if any(type(n) is not int or n < 1 for n in (input_width, hidden_width)):
            raise ValueError("Head dimensions must be positive integers")
        self.input_width = input_width
        self.layers = nn.Sequential(nn.Linear(input_width, hidden_width), nn.ReLU(), nn.Linear(hidden_width, 1))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.input_width:
            raise ValueError("Wrong frozen feature width")
        return self.layers(features.detach()).squeeze(-1)


class ValueHead(_DetachedScalarHead):
    def __init__(self, input_width: int = 256, hidden_width: int = 64):
        super().__init__(input_width, hidden_width)


class StateResidualHead(_DetachedScalarHead):
    def __init__(self, input_width: int = 256, hidden_width: int = 64):
        super().__init__(input_width, hidden_width)


class ActionResidualHead(_DetachedScalarHead):
    def __init__(self, input_width: int = 384, hidden_width: int = 64):
        super().__init__(input_width, hidden_width)


class ResidualHeads(nn.Module):
    def __init__(self, seed: int, state_width: int = 256, action_width: int = 128, hidden_width: int = 64):
        super().__init__()
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Head initialization seed must be a nonnegative int64")
        self.seed = seed
        self.config = {"state_width": state_width, "action_width": action_width, "hidden_width": hidden_width}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.value = ValueHead(state_width, hidden_width)
            self.state = StateResidualHead(state_width, hidden_width)
            self.action = ActionResidualHead(state_width + action_width, hidden_width)

    def predictions(self, context: torch.Tensor, candidates: torch.Tensor) -> dict[str, torch.Tensor]:
        if context.ndim != 2 or candidates.ndim != 3 or len(context) != len(candidates):
            raise ValueError("Expected aligned batched context and candidates")
        expanded = context.unsqueeze(1).expand(-1, candidates.shape[1], -1)
        return {"value": self.value(context), "state": self.state(context),
                "action": self.action(torch.cat((expanded, candidates), dim=-1))}


def correction_choices(logits: np.ndarray, residuals: np.ndarray, probabilities: np.ndarray,
                       kinds: np.ndarray, strength: float, wrapped: bool = True,
                       threshold: float = ELIGIBILITY_THRESHOLD,
                       prior_floor: float = PRIOR_FLOOR) -> dict[str, Any]:
    """Stable choices with exact lambda-zero identity and explicit eligibility.

    A raw-reference action remains eligible irrespective of its explorer mass.
    The first legal winning action is an explicit optional wrapper. A reference
    logit is a policy score, not a utility estimate.
    """
    logits, residuals = np.asarray(logits, dtype=np.float64), np.asarray(residuals, dtype=np.float64)
    probabilities, kinds = np.asarray(probabilities, dtype=np.float64), np.asarray(kinds)
    if (logits.ndim != 1 or not len(logits) or residuals.shape != logits.shape
            or probabilities.shape != logits.shape or kinds.shape != logits.shape
            or not np.isfinite(logits).all() or not np.isfinite(residuals).all()
            or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or np.any(probabilities > 1) or not math.isclose(float(probabilities.sum()), 1., rel_tol=0, abs_tol=1e-12)
            or not math.isfinite(strength) or strength < 0 or type(wrapped) is not bool
            or threshold != ELIGIBILITY_THRESHOLD or prior_floor != PRIOR_FLOOR):
        raise ValueError("Invalid correction inputs or undeclared eligibility/prior parameters")
    reference = int(logits.argmax())
    wrapped_reference = first_win_index(kinds, reference)
    eligible = probabilities >= threshold
    eligible[reference] = True
    prior = np.maximum(logits - logits[reference], prior_floor)
    with np.errstate(over="ignore", invalid="ignore"):
        scores = prior + strength * residuals
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite corrected scores")
    scores[~eligible] = -np.inf
    # An explicit branch protects identity even if the prior implementation
    # changes; zero output also preserves first-index tie ordering naturally.
    corrected = reference if strength == 0 else int(scores.argmax())
    selected = first_win_index(kinds, corrected) if wrapped else corrected
    return {"selected_index": selected, "raw_reference_index": reference,
            "wrapped_reference_index": wrapped_reference, "raw_corrected_index": corrected,
            "eligible": eligible, "prior": prior, "scores": scores}


def load_head_bundle(path: str | Path, expected_sha256: str | None = None,
                     expected_identity: Mapping[str, Any] | None = None,
                     device: str = "cpu") -> tuple[ResidualHeads, dict[str, Any]]:
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Head bundle file hash mismatch")
    saved = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    required_identity = {"reference_sha256", "call_sha256", "protocol_sha256", "cache_manifest_sha256"}
    identity = saved.get("identity", {})
    epochs = saved.get("selected_epochs", {})
    states = saved.get("head_state_dicts", {})
    if (saved.get("schema") != BUNDLE_SCHEMA or saved.get("head_config") != HEAD_CONFIG
            or not isinstance(identity, dict) or not required_identity <= identity.keys()
            or any(not isinstance(identity[key], str) or len(identity[key]) != 64 for key in required_identity)
            or set(epochs) != {"value", "state", "action"}
            or any(type(n) is not int or n < 0 for n in epochs.values())
            or set(states) != {"value", "state", "action"}):
        raise ValueError("Malformed Stage 3I head bundle identity or architecture")
    if expected_identity is not None and any(identity.get(k) != v for k, v in expected_identity.items()):
        raise ValueError("Head bundle identity differs from fixed protocol")
    heads = ResidualHeads(saved["seed"], **saved["head_config"])
    for name in ("value", "state", "action"):
        getattr(heads, name).load_state_dict(states[name], strict=True)
    if (_state_hash(heads.state_dict()) != saved.get("weights_sha256")
            or any(not bool(torch.isfinite(p).all()) for p in heads.parameters())):
        raise ValueError("Head bundle tensor hash or finiteness failure")
    _freeze(heads)
    metadata = {key: value for key, value in saved.items() if key != "head_state_dicts"}
    metadata["checkpoint_sha256"] = digest
    return heads.to(device), metadata
