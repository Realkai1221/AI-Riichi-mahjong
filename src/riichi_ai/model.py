"""Compact suit-aware actor and rank predictor with semantic legal actions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import io
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from .features import (
    ACTION_FIELDS, EVENT_KINDS, FEATURE_SCHEMA, FEATURE_SCHEMA_V2, GLOBAL_FEATURES, HISTORY_FIELDS,
    MAX_HISTORY, TILE_CHANNELS, encode_view, efficiency_from_encoded,
)
from .player import ActionSpec, PlayerView
from .semantics import ACTION_KINDS, SEMANTIC_ACTION_VERSION, semantic_actions


MODEL_SCHEMA = "suit-history-policy-v1"
MODEL_SCHEMA_V2 = "suit-history-efficiency-policy-v2"
_DEFAULT_CONFIG = {
    "model_schema": MODEL_SCHEMA, "feature_schema": FEATURE_SCHEMA,
    "semantic_action_schema": SEMANTIC_ACTION_VERSION,
    "suit_width": 192, "history_width": 64, "hidden_width": 512,
    "context_width": 256, "candidate_width": 128,
}


def model_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the complete, validated JSON-compatible architecture config."""
    result = dict(_DEFAULT_CONFIG)
    if config is not None:
        unknown = set(config) - set(result)
        if unknown:
            raise ValueError(f"Unknown model configuration keys: {sorted(unknown)}")
        result.update(config)
    if result["feature_schema"] not in (FEATURE_SCHEMA, FEATURE_SCHEMA_V2):
        raise ValueError(f"Unsupported feature_schema: {result['feature_schema']!r}")
    if result["model_schema"] not in (MODEL_SCHEMA, MODEL_SCHEMA_V2):
        raise ValueError(f"Unsupported model_schema: {result['model_schema']!r}")
    if (result["model_schema"], result["feature_schema"]) not in (
            (MODEL_SCHEMA, FEATURE_SCHEMA), (MODEL_SCHEMA_V2, FEATURE_SCHEMA_V2)):
        raise ValueError("Model and feature schemas must select the same supported version")
    if result["semantic_action_schema"] != SEMANTIC_ACTION_VERSION:
        raise ValueError(f"Unsupported semantic_action_schema: {result['semantic_action_schema']!r}")
    for name in ("suit_width", "history_width", "hidden_width", "context_width", "candidate_width"):
        if type(result[name]) is not int or result[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    return result


def efficiency_model_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Explicit additive-v2 configuration; model_config() remains raw v1."""
    return model_config({**dict(config or {}), "model_schema": MODEL_SCHEMA_V2,
                         "feature_schema": FEATURE_SCHEMA_V2})


def collate_encoded(samples: Sequence[Mapping[str, np.ndarray]], device: Any = None) -> dict[str, torch.Tensor]:
    """Pad only candidates; fixed state/history arrays preserve their masks."""
    if not samples:
        raise ValueError("Cannot collate an empty sample list")
    assisted = ["candidate_efficiency" in sample for sample in samples]
    if any(assisted) and not all(assisted):
        raise ValueError("Cannot mix v1 and v2 feature schemas in one batch")
    shapes = {"tiles": (34, TILE_CHANNELS), "globals": (GLOBAL_FEATURES,),
              "history": (MAX_HISTORY, HISTORY_FIELDS), "history_mask": (MAX_HISTORY,)}
    for sample in samples:
        for key, shape in shapes.items():
            if sample[key].shape != shape:
                raise ValueError(f"Invalid {key} shape; expected {shape}")
        actions = sample["action_tokens"]
        if actions.ndim != 2 or actions.shape[1] != ACTION_FIELDS or not len(actions):
            raise ValueError("Each decision needs at least one semantic candidate")
        if assisted[0]:
            efficiency = sample["candidate_efficiency"]
            if (efficiency.shape != (len(actions), 4) or efficiency.dtype != np.float32
                    or not np.isfinite(efficiency).all()):
                raise ValueError("Expected finite float32 candidate_efficiency with shape (N,4)")
            applicable = actions[:, 0] == ACTION_KINDS.index("DISCARD") + 1
            if (not np.array_equal(efficiency[:, 3], applicable.astype(np.float32))
                    or np.any(efficiency[~applicable] != 0)):
                raise ValueError("Efficiency applicability must match DISCARD candidates; other rows must be zero")
    batch = {
        key: torch.from_numpy(np.stack([sample[key] for sample in samples])).to(
            dtype=torch.bool if key == "history_mask" else torch.long if key == "history" else torch.float32
        ) for key in shapes
    }
    max_candidates = max(len(sample["action_tokens"]) for sample in samples)
    action_tokens = np.zeros((len(samples), max_candidates, ACTION_FIELDS), dtype=np.int64)
    mask = np.zeros((len(samples), max_candidates), dtype=np.bool_)
    for index, sample in enumerate(samples):
        count = len(sample["action_tokens"])
        action_tokens[index, :count] = sample["action_tokens"]
        mask[index, :count] = True
    batch["action_tokens"] = torch.from_numpy(action_tokens)
    batch["candidate_mask"] = torch.from_numpy(mask)
    if assisted[0]:
        efficiency = np.zeros((len(samples), max_candidates, 4), dtype=np.float32)
        for index, sample in enumerate(samples):
            efficiency[index, :len(sample["action_tokens"])] = sample["candidate_efficiency"]
        batch["candidate_efficiency"] = torch.from_numpy(efficiency)
    return batch if device is None else {key: value.to(device) for key, value in batch.items()}


class PolicyNetwork(nn.Module):
    """Suit-shared MLPs never convolve across suit or honor boundaries.

    A short history CNN preserves local event order; masked mean and final-event
    pooling compress it. Candidate scoring is permutation equivariant.
    """

    def __init__(self, config: Mapping[str, Any] | None = None):
        super().__init__()
        self.config = model_config(config)
        suit, history = self.config["suit_width"], self.config["history_width"]
        hidden, context = self.config["hidden_width"], self.config["context_width"]
        candidate = self.config["candidate_width"]
        self.suit_encoder = nn.Sequential(nn.Linear(9 * TILE_CHANNELS, suit), nn.ReLU(),
                                          nn.Linear(suit, suit), nn.ReLU())
        self.honor_encoder = nn.Sequential(nn.Linear(7 * TILE_CHANNELS, suit), nn.ReLU(),
                                           nn.Linear(suit, suit), nn.ReLU())
        self.global_encoder = nn.Sequential(nn.Linear(GLOBAL_FEATURES, 64), nn.ReLU(),
                                            nn.Linear(64, 64), nn.ReLU())
        self.event_kind = nn.Embedding(len(EVENT_KINDS) + 1, 16, padding_idx=0)
        self.relative_seat = nn.Embedding(5, 4, padding_idx=0)
        self.tile_kind = nn.Embedding(35, 16, padding_idx=0)
        self.red_flag = nn.Embedding(2, 4)
        self.tsumogiri_flag = nn.Embedding(3, 4, padding_idx=0)
        self.consumed_tile = nn.Embedding(69, 8, padding_idx=0)
        self.history_input = nn.Linear(80, history)
        self.history_position = nn.Embedding(MAX_HISTORY, history)
        self.history_conv = nn.Sequential(nn.Conv1d(history, history, 3, padding=1), nn.ReLU(),
                                          nn.Conv1d(history, history, 3, padding=1), nn.ReLU())
        self.history_pool = nn.Sequential(nn.Linear(history * 2, 128), nn.ReLU())
        self.context = nn.Sequential(nn.Linear(4 * suit + 64 + 128, hidden), nn.ReLU(),
                                     nn.Linear(hidden, context), nn.ReLU())
        self.action_kind = nn.Embedding(len(ACTION_KINDS) + 1, 16, padding_idx=0)
        self.action_encoder = nn.Sequential(nn.Linear(72, candidate), nn.ReLU(),
                                            nn.Linear(candidate, candidate), nn.ReLU())
        self.action_score = nn.Sequential(nn.Linear(context + candidate, 256), nn.ReLU(), nn.Linear(256, 1))
        self.rank_head = nn.Sequential(nn.Linear(context, 128), nn.ReLU(), nn.Linear(128, 4))
        # Added after every v1 module, preserving common parameter names,
        # shapes, initialization order and the historical forward path.
        if self.config["feature_schema"] == FEATURE_SCHEMA_V2:
            self.efficiency_encoder = nn.Linear(4, candidate, bias=False)

    def _history(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embedded = torch.cat((
            self.event_kind(tokens[..., 0]), self.relative_seat(tokens[..., 1]),
            self.relative_seat(tokens[..., 2]), self.tile_kind(tokens[..., 3]),
            self.red_flag(tokens[..., 4]), self.tsumogiri_flag(tokens[..., 5]),
            self.consumed_tile(tokens[..., 6:10]).flatten(-2),
        ), dim=-1)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = torch.relu(self.history_input(embedded) + self.history_position(positions))
        hidden = hidden * mask.unsqueeze(-1)
        hidden = self.history_conv(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = hidden * mask.unsqueeze(-1)
        lengths = mask.sum(dim=1)
        mean = hidden.sum(dim=1) / lengths.clamp(min=1).unsqueeze(-1)
        last_index = (lengths - 1).clamp(min=0)
        last = hidden[torch.arange(len(hidden), device=hidden.device), last_index]
        return self.history_pool(torch.cat((mean, last), dim=-1))

    def forward(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        assisted = self.config["feature_schema"] == FEATURE_SCHEMA_V2
        if assisted != ("candidate_efficiency" in batch):
            raise ValueError("Batch feature schema differs from model configuration")
        tiles = batch["tiles"]
        count = tiles.shape[0]
        suits = self.suit_encoder(tiles[:, :27].reshape(count, 3, 9 * TILE_CHANNELS)).flatten(1)
        honors = self.honor_encoder(tiles[:, 27:].flatten(1))
        context = self.context(torch.cat((suits, honors, self.global_encoder(batch["globals"]),
                                          self._history(batch["history"], batch["history_mask"])), dim=-1))
        tokens = batch["action_tokens"]
        actions = self.action_encoder(torch.cat((
            self.action_kind(tokens[..., 0]), self.tile_kind(tokens[..., 1]),
            self.red_flag(tokens[..., 2]), self.tsumogiri_flag(tokens[..., 3]),
            self.consumed_tile(tokens[..., 4:8]).flatten(-2),
        ), dim=-1))
        if assisted:
            actions = actions + self.efficiency_encoder(batch["candidate_efficiency"])
        expanded = context.unsqueeze(1).expand(-1, actions.shape[1], -1)
        logits = self.action_score(torch.cat((expanded, actions), dim=-1)).squeeze(-1)
        # Finite minimum avoids 0 * -inf in masked soft-target cross entropy.
        logits = logits.masked_fill(~batch["candidate_mask"], torch.finfo(logits.dtype).min)
        return logits, self.rank_head(context)


def _source_hashes() -> dict[str, str]:
    directory = Path(__file__).parent
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("features.py", "model.py", "semantics.py", "efficiency.py")}


def _weights_hash(model: nn.Module) -> str:
    return _state_hash(model.state_dict())


def _state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def matched_initialization(seed: int, raw_config: Mapping[str, Any] | None = None,
                           assisted_config: Mapping[str, Any] | None = None) -> tuple[dict, dict, dict]:
    """Return CPU state dicts with bit-identical common tensors and provenance.

    Fork only the CPU RNG and restore its state. The extra branch uses a
    separately derived stream; this helper never touches accelerator RNGs.
    Reuse the assisted state for both rank-loss weights in a paired comparison.
    """
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("Initialization seed must be an integer in [0,2**64)")
    raw_configuration = model_config(raw_config)
    if raw_configuration["feature_schema"] != FEATURE_SCHEMA:
        raise ValueError("Raw initialization must use v1")
    assisted_configuration = efficiency_model_config(assisted_config or raw_configuration)
    added_seed = int.from_bytes(hashlib.sha256(f"matched-efficiency-v1:{seed}".encode()).digest()[:8], "big")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        raw = PolicyNetwork(raw_configuration)
        torch.random.default_generator.manual_seed(added_seed)
        assisted = PolicyNetwork(assisted_configuration)
    raw_state = {key: value.detach().cpu().clone() for key, value in raw.state_dict().items()}
    assisted_state = {key: value.detach().cpu().clone() for key, value in assisted.state_dict().items()}
    for key, value in raw_state.items():
        if key not in assisted_state or assisted_state[key].shape != value.shape:
            raise ValueError("Matched initialization requires identical common parameter names and shapes")
        assisted_state[key] = value.clone()
    manifest = {"schema": "matched-efficiency-initialization-v1", "seed": seed,
                "added_branch_seed": added_seed, "raw_config": raw_configuration,
                "assisted_config": assisted_configuration, "common_tensors": len(raw_state),
                "common_weights_sha256": _state_hash(raw_state), "maximum_common_difference": 0.0,
                "raw_weights_sha256": _state_hash(raw_state), "assisted_weights_sha256": _state_hash(assisted_state),
                "added_tensors": sorted(set(assisted_state) - set(raw_state)),
                "rng": "Separate seeded CPU streams; caller CPU RNG restored; no accelerator RNG access."}
    return raw_state, assisted_state, manifest


class NeuralPolicy:
    """Deterministic semantic argmax with exact legal physical resolution."""

    def __init__(self, model: PolicyNetwork, *, device: Any = "cpu",
                 checkpoint_hash: str | None = None, source_manifest: Any = None):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.checkpoint_hash = checkpoint_hash
        self.weights_hash = _weights_hash(model)
        self.sources = _source_hashes() if source_manifest is None else source_manifest
        self._inference_counts = {"decisions": 0, "forced_decisions": 0, "network_calls": 0}
        self._inference_seconds = {key: 0.0 for key in (
            "semantic_actions", "encoding", "efficiency", "collation_and_transfer", "model", "total_choose")}

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: Any = "cpu") -> "NeuralPolicy":
        path = Path(path)
        # Hash exactly the bytes loaded, even if another process replaces the
        # checkpoint path while this policy is being constructed.
        payload = path.read_bytes()
        checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or not {"model_config", "model_state_dict"} <= checkpoint.keys():
            raise ValueError("Checkpoint needs model_config and model_state_dict")
        config = checkpoint["model_config"]
        if not isinstance(config, dict) or not set(_DEFAULT_CONFIG) <= config.keys():
            raise ValueError("Checkpoint must record the complete versioned model configuration")
        validated_config = model_config(config)
        if checkpoint.get("feature_schema", validated_config["feature_schema"]) != validated_config["feature_schema"]:
            raise ValueError("Checkpoint feature schema does not match the encoder")
        model = PolicyNetwork(config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return cls(model, device=device, checkpoint_hash=hashlib.sha256(payload).hexdigest(),
                   source_manifest=checkpoint.get("source_manifest"))

    def choose(self, view: PlayerView) -> ActionSpec:
        started = time.perf_counter()
        actions = semantic_actions(view)
        self._inference_seconds["semantic_actions"] += time.perf_counter() - started
        if not actions:
            raise ValueError("The player has no legal action to choose")
        self._inference_counts["decisions"] += 1
        if len(actions) == 1:
            self._inference_counts["forced_decisions"] += 1
            self._inference_seconds["total_choose"] += time.perf_counter() - started
            return actions[0].representative
        tick = time.perf_counter()
        encoded = encode_view(view, actions)
        self._inference_seconds["encoding"] += time.perf_counter() - tick
        if self.model.config["feature_schema"] == FEATURE_SCHEMA_V2:
            tick = time.perf_counter()
            encoded["candidate_efficiency"] = efficiency_from_encoded(encoded["tiles"], encoded["action_tokens"])
            self._inference_seconds["efficiency"] += time.perf_counter() - tick
        tick = time.perf_counter()
        batch = collate_encoded([encoded], self.device)
        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._inference_seconds["collation_and_transfer"] += time.perf_counter() - tick
        self.model.eval()
        self._inference_counts["network_calls"] += 1
        tick = time.perf_counter()
        with torch.inference_mode():
            logits, _ = self.model(batch)
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("Neural policy produced non-finite logits")
            index = int(logits[0].argmax().item())
        self._inference_seconds["model"] += time.perf_counter() - tick
        self._inference_seconds["total_choose"] += time.perf_counter() - started
        return actions[index].representative

    def inference_stats(self) -> dict[str, int]:
        """Per-wrapper counters, separate from the immutable policy manifest."""
        return dict(self._inference_counts)

    def inference_timing(self) -> dict[str, float]:
        """Cumulative wall seconds outside manifests/digests; device phases synchronize."""
        return dict(self._inference_seconds)

    def manifest(self) -> dict[str, Any]:
        return {"policy": "neural", "version": self.model.config["model_schema"],
                "feature_schema": self.model.config["feature_schema"],
                "semantic_action_schema": SEMANTIC_ACTION_VERSION,
                "model_config": dict(self.model.config), "checkpoint_sha256": self.checkpoint_hash,
                "weights_sha256": self.weights_hash, "source_manifest": self.sources,
                "device": str(self.device), "dtype": "float32", "decision_rule": "stable_semantic_argmax"}
