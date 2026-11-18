"""Bounded, manifest-bound development streaming for Stage 3F.

Actor tensors never contain outcome/provenance targets. Only explicitly named
actor and target shards are opened; auditor and full-engine logs are not read.
Whole matches are the shuffling/split unit. A single match's arrays and a single
batch are resident: this trades global row shuffling for bounded memory and
reproducible restart. This is infrastructure, not a trained value policy.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterator
import zipfile

import numpy as np

from .features import ACTION_FIELDS, GLOBAL_FEATURES, HISTORY_FIELDS, MAX_HISTORY, TILE_CHANNELS
from .semantics import ACTION_KINDS

CORPUS_SCHEMA = "riichi-ai-stage3f-corpus-v1"
ACTOR_SCHEMA = "riichi-ai-stage3f-compact-actor-v1"
TARGET_SCHEMA = "riichi-ai-stage3f-match-targets-v1"
CURSOR_SCHEMA = "riichi-ai-stage3f-stream-cursor-v1"
ACTOR_KEYS = frozenset(("tiles", "globals", "action_tokens", "candidate_efficiency",
    "candidate_offsets", "history_tokens", "history_start", "history_length",
    "history_segment_start", "history_segment_length", "history_prefix_length"))
TARGET_KEYS = frozenset(("chosen_index", "seat", "step", "hand_id", "final_rank",
    "final_utility", "behavior_probability", "behavior_id", "behavior_probabilities"))
RANK_UTILITY = np.asarray([1.5, .5, -.5, -1.5], dtype=np.float32)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _positive(value: int, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _integer(array: np.ndarray, name: str) -> None:
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must have integer dtype")


def validate_arrays(actor: dict[str, np.ndarray], targets: dict[str, np.ndarray],
                    rows: int) -> None:
    """Validate every loaded row, including within-seat/hand history prefixes.

    This catches structural corruption and future/cross-segment slices. It is
    not independent rule certification: replay/auditing remains separate.
    """
    if set(actor) != ACTOR_KEYS or set(targets) != TARGET_KEYS:
        raise ValueError("Shard array keys differ from the versioned schema")
    _positive(rows, "rows")
    for area in (actor, targets):
        for name, array in area.items():
            if array.dtype.hasobject or not np.isfinite(array).all():
                raise ValueError(f"Non-numeric or nonfinite {name}")
    if actor["tiles"].shape != (rows, 34, TILE_CHANNELS) or actor["globals"].shape != (rows, GLOBAL_FEATURES):
        raise ValueError("Actor state shape mismatch")
    for name in ("tiles", "globals", "candidate_efficiency"):
        if actor[name].dtype != np.float32:
            raise ValueError(f"{name} must be float32")
    offsets = actor["candidate_offsets"]
    _integer(offsets, "candidate_offsets")
    if offsets.shape != (rows + 1,) or offsets[0] != 0 or np.any(np.diff(offsets) <= 0):
        raise ValueError("Invalid candidate offsets")
    total = int(offsets[-1])
    tokens = actor["action_tokens"]
    _integer(tokens, "action_tokens")
    if tokens.shape != (total, ACTION_FIELDS):
        raise ValueError("Action token shape differs from candidate offsets")
    upper = np.asarray([len(ACTION_KINDS), 34, 1, 2, 68, 68, 68, 68])
    if np.any(tokens < 0) or np.any(tokens > upper) or np.any(tokens[:, 0] == 0):
        raise ValueError("Action token out of embedding range")
    efficiency = actor["candidate_efficiency"]
    if efficiency.shape != (total, 4):
        raise ValueError("Efficiency shape mismatch")
    discards = tokens[:, 0] == ACTION_KINDS.index("DISCARD") + 1
    if not np.array_equal(efficiency[:, 3], discards.astype(np.float32)) or np.any(efficiency[~discards] != 0):
        raise ValueError("Efficiency applicability mismatch")
    history = actor["history_tokens"]
    _integer(history, "history_tokens")
    if history.ndim != 2 or history.shape[1] != HISTORY_FIELDS:
        raise ValueError("History token shape mismatch")
    from .features import EVENT_KINDS
    history_upper = np.asarray([len(EVENT_KINDS), 4, 4, 34, 1, 2, 68, 68, 68, 68])
    if np.any(history < 0) or np.any(history > history_upper) or np.any(history[:, 0] == 0):
        raise ValueError("History token out of embedding range")
    for name in ("history_start", "history_length", "history_segment_start", "history_segment_length", "history_prefix_length"):
        _integer(actor[name], name)
        if actor[name].shape != (rows,):
            raise ValueError("History metadata shape mismatch")
    start, length = actor["history_start"], actor["history_length"]
    segment, segment_length = actor["history_segment_start"], actor["history_segment_length"]
    prefix = actor["history_prefix_length"]
    if (np.any(segment < 0) or np.any(segment > len(history))
            or np.any(segment_length <= 0) or np.any(segment_length > len(history)) or np.any(prefix < 0)
            or np.any(prefix > segment_length) or np.any(segment + segment_length > len(history))
            or not np.array_equal(start, segment + np.maximum(0, prefix - MAX_HISTORY))
            or not np.array_equal(length, np.minimum(MAX_HISTORY, prefix))):
        raise ValueError("Invalid history prefix or cross-segment slice")
    if not np.array_equal(np.rint(actor["globals"][:, 41] * MAX_HISTORY).astype(np.int64), prefix):
        raise ValueError("History prefix differs from visible encoded history length")
    for name in TARGET_KEYS - {"behavior_probabilities"}:
        if targets[name].shape != (rows,):
            raise ValueError(f"Target shape mismatch: {name}")
    for name in ("chosen_index", "seat", "step", "hand_id", "final_rank", "behavior_id"):
        _integer(targets[name], name)
    if (np.any(targets["chosen_index"] < 0) or np.any(targets["chosen_index"] >= np.diff(offsets))
            or np.any(targets["seat"] < 0) or np.any(targets["seat"] > 3)
            or np.any(targets["step"] < 0) or np.any(np.diff(targets["step"]) < 0)
            or np.any(targets["hand_id"] < 0) or np.any(targets["final_rank"] < 1)
            or np.any(targets["final_rank"] > 4) or np.any(targets["behavior_id"] < 0)
            or np.any(targets["behavior_id"] > 3)):
        raise ValueError("Invalid target identity or selected legal action")
    if (not np.array_equal(actor["globals"][:, 24:28], np.eye(4, dtype=np.float32)[targets["seat"]])
            or not np.allclose(actor["globals"][:, 38], targets["hand_id"].astype(np.float32) / 20, rtol=0, atol=1e-7)):
        raise ValueError("Actor seat or hand identity differs from targets")
    if not np.array_equal(targets["final_utility"], RANK_UTILITY[targets["final_rank"] - 1]):
        raise ValueError("Complete-match utility differs from declared rank objective")
    probabilities = targets["behavior_probabilities"]
    if probabilities.shape != (total,) or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Invalid behavior candidate probabilities")
    selected = probabilities[offsets[:-1] + targets["chosen_index"]]
    sums = np.add.reduceat(probabilities, offsets[:-1])
    if (not np.allclose(sums, 1, rtol=0, atol=1e-12)
            or not np.array_equal(selected, targets["behavior_probability"])
            or np.any(selected <= 0)):
        raise ValueError("Behavior probability or selected probability mismatch")
    segments: dict[tuple[int, int], tuple[int, int]] = {}
    previous: dict[int, tuple[int, int, int]] = {}
    seat_outcomes: dict[int, tuple[int, int]] = {}
    for row in range(rows):
        seat, hand, step = (int(targets[key][row]) for key in ("seat", "hand_id", "step"))
        identity = (int(segment[row]), int(segment_length[row]))
        key = (seat, hand)
        if key in segments and segments[key] != identity:
            raise ValueError("History segment identity changed within a hand")
        segments[key] = identity
        if seat in previous:
            old_hand, old_step, old_prefix = previous[seat]
            if step <= old_step or hand < old_hand or (hand == old_hand and prefix[row] < old_prefix):
                raise ValueError("History prefix or hand identity moved backwards")
        previous[seat] = (hand, step, int(prefix[row]))
        outcome = (int(targets["final_rank"][row]), int(targets["behavior_id"][row]))
        if seat in seat_outcomes and seat_outcomes[seat] != outcome:
            raise ValueError("Per-seat outcome or behavior identity changed")
        seat_outcomes[seat] = outcome
    if set(seat_outcomes) != {0, 1, 2, 3} or sorted(v[0] for v in seat_outcomes.values()) != [1, 2, 3, 4]:
        raise ValueError("Complete all-seat match outcomes are missing or invalid")
    ranges = sorted(segments.values())
    if len(set(ranges)) != len(ranges) or any(a[0] + a[1] != b[0] for a, b in zip(ranges, ranges[1:])):
        raise ValueError("History segments overlap, alias, or contain gaps")
    if not ranges or ranges[0][0] != 0 or ranges[-1][0] + ranges[-1][1] != len(history):
        raise ValueError("History segment coverage differs from stored tokens")


@dataclass(frozen=True)
class StreamBatch:
    """A batch entirely within one match, and a cursor after that batch."""
    actor: dict[str, np.ndarray]
    targets: dict[str, np.ndarray]
    match_id: str
    row_indices: np.ndarray
    cursor: dict[str, Any]
    working_array_bytes: int

    @property
    def rows(self) -> int:
        return len(self.row_indices)

    def torch_actor(self, device: str = "cpu") -> dict[str, Any]:
        """CPU-owned contiguous staging avoids historical MPS transfer aliasing."""
        import torch
        integers = {"history", "action_tokens"}
        booleans = {"history_mask", "candidate_mask"}
        staged = {}
        for name, array in self.actor.items():
            dtype = torch.long if name in integers else torch.bool if name in booleans else torch.float32
            value = torch.from_numpy(array.copy()).to(dtype=dtype).contiguous().clone()
            staged[name] = value.to(device=device, copy=True) if device != "cpu" else value
        return staged


class StreamDataset:
    """No preload or global row index; strict single-shard streaming."""
    def __init__(self, directory: str | Path, split: str = "development", *,
                 max_shard_bytes: int = 64 << 20, memory_limit_bytes: int = 256 << 20):
        self.directory = Path(directory).expanduser().resolve()
        self.max_shard_bytes = _positive(max_shard_bytes, "max_shard_bytes")
        self.memory_limit_bytes = _positive(memory_limit_bytes, "memory_limit_bytes")
        if memory_limit_bytes <= 2 * max_shard_bytes:
            raise ValueError("Memory limit must reserve two shard sizes plus a batch")
        path = self.directory / "manifest.json"
        raw = path.read_bytes()
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        manifest = json.loads(raw)
        if (manifest.get("schema") != CORPUS_SCHEMA or manifest.get("complete") is not True
                or manifest.get("source_role") != "development-only"):
            raise ValueError("Only finalized complete development manifests can be streamed")
        reports = manifest.get("matches")
        if not isinstance(reports, list) or not reports:
            raise ValueError("Manifest contains no matches")
        self.reports = {}
        for report in reports:
            name = report.get("match_id")
            if (not isinstance(name, str) or not name or Path(name).name != name
                    or name in self.reports or report.get("completed") is not True
                    or report.get("failure") is not None
                    or report.get("source_role") != "development-only"
                    or report.get("training_eligible") is not False or report.get("ppo_eligible") is not False
                    or report.get("actor_schema") != ACTOR_SCHEMA or report.get("target_schema") != TARGET_SCHEMA):
                raise ValueError("Incomplete or duplicate manifest match")
            _positive(report.get("rows"), "manifest rows")
            for area in ("actor", "targets"):
                relative = f"{area}/{name}.npz"
                artifact = report.get("artifacts", {}).get(relative)
                if (not isinstance(artifact, dict) or not isinstance(artifact.get("sha256"), str)
                        or len(artifact["sha256"]) != 64 or type(artifact.get("bytes")) is not int
                        or artifact["bytes"] <= 0):
                    raise ValueError("Missing manifested actor or target artifact")
                path = self.directory / relative
                if path.resolve().parent != self.directory / area or path.is_symlink():
                    raise ValueError("Shard escapes its declared directory")
            self.reports[name] = report
        splits = manifest.get("splits")
        if (not isinstance(splits, dict) or split not in splits
                or any(not isinstance(names, list) or any(not isinstance(name, str) for name in names) for names in splits.values())):
            raise ValueError("Unknown manifest split")
        assigned = [name for names in splits.values() for name in names]
        if len(assigned) != len(set(assigned)) or set(assigned) != set(self.reports):
            raise ValueError("Split overlap or unassigned manifest match")
        self.match_ids = tuple(splits[split])
        if not self.match_ids:
            raise ValueError("Requested split is empty")
        for area in ("actor", "targets"):
            actual = {path.name for path in (self.directory / area).glob("*.npz")}
            if actual != {f"{name}.npz" for name in self.reports}:
                raise ValueError("Unmanifested or missing numeric shard")
        self.split = split
        self.loaded_match: str | None = None
        self.peak_working_array_bytes = 0
        self.shards_opened = 0
        self.bytes_read = 0

    def _load(self, name: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
        report = self.reports[name]
        groups = []
        incoming = 0
        for area, keys in (("actor", ACTOR_KEYS), ("targets", TARGET_KEYS)):
            relative = f"{area}/{name}.npz"
            path, metadata = self.directory / relative, report["artifacts"][relative]
            if path.stat().st_size != metadata["bytes"] or _sha(path) != metadata["sha256"]:
                raise ValueError(f"Shard hash or byte count mismatch: {relative}")
            with zipfile.ZipFile(path) as archive:
                members = archive.infolist()
                if len(members) != len(keys) or {item.filename for item in members} != {f"{key}.npy" for key in keys}:
                    raise ValueError("Unexpected or duplicated NPZ member")
                if sum(item.file_size for item in members) + incoming > self.max_shard_bytes:
                    raise MemoryError("Shard decompression exceeds declared memory bound")
                for member in members:
                    with archive.open(member) as stream:
                        version = np.lib.format.read_magic(stream)
                        reader = { (1, 0): np.lib.format.read_array_header_1_0,
                                   (2, 0): np.lib.format.read_array_header_2_0 }.get(version)
                        if reader is None:
                            raise ValueError("Unsupported NPY format version")
                        shape, _, dtype = reader(stream, max_header_size=16384)
                        if dtype.hasobject or any(type(n) is not int or n < 0 for n in shape):
                            raise ValueError("Object or invalid-shape NPZ member")
                        declared = math.prod(shape) * dtype.itemsize
                        if declared > self.max_shard_bytes:
                            raise MemoryError("NPY header allocation exceeds memory bound")
                        if declared != member.file_size - stream.tell():
                            raise ValueError("NPY header size differs from stored numeric payload")
            with np.load(path, allow_pickle=False) as shard:
                arrays = {key: shard[key] for key in sorted(keys)}
            incoming += sum(array.nbytes for array in arrays.values())
            if incoming > self.max_shard_bytes:
                raise MemoryError("Uncompressed shard exceeds declared memory bound")
            expected = report.get("uncompressed_actor_bytes" if area == "actor" else "uncompressed_target_bytes")
            if expected != sum(array.nbytes for array in arrays.values()):
                raise ValueError("Uncompressed bytes differ from manifest")
            self.bytes_read += metadata["bytes"]
            groups.append(arrays)
        validate_arrays(groups[0], groups[1], report["rows"])
        for group in groups:
            for array in group.values():
                array.setflags(write=False)
        self.loaded_match = name
        self.shards_opened += 1
        return groups[0], groups[1], incoming

    def _cursor(self, *, epoch: int, seed: int, batch_rows: int, match_position: int, row_position: int) -> dict[str, Any]:
        return {"schema": CURSOR_SCHEMA, "manifest_sha256": self.manifest_sha256,
                "split": self.split, "epoch": epoch, "seed": seed, "batch_rows": batch_rows,
                "match_position": match_position, "row_position": row_position}

    def iter_batches(self, *, epoch: int = 0, seed: int = 0, batch_rows: int = 64,
                     cursor: dict[str, Any] | None = None) -> Iterator[StreamBatch]:
        _positive(batch_rows, "batch_rows")
        for value, label in ((epoch, "epoch"), (seed, "seed")):
            if type(value) is not int or not 0 <= value < 2**64:
                raise ValueError(f"{label} must be an integer in [0, 2**64)")
        match_order = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, epoch, 0]))).permutation(len(self.match_ids))
        match_position = row_position = 0
        if cursor is not None:
            expected = self._cursor(epoch=epoch, seed=seed, batch_rows=batch_rows,
                                    match_position=cursor.get("match_position"), row_position=cursor.get("row_position"))
            if cursor != expected:
                raise ValueError("Resume cursor differs from manifest or epoch settings")
            match_position, row_position = cursor["match_position"], cursor["row_position"]
            if (type(match_position) is not int or type(row_position) is not int or match_position < 0
                    or match_position > len(match_order) or row_position < 0
                    or (match_position == len(match_order) and row_position != 0)):
                raise ValueError("Invalid resume cursor position")
        for position in range(match_position, len(match_order)):
            name = self.match_ids[int(match_order[position])]
            actor, targets, shard_bytes = self._load(name)
            rows = self.reports[name]["rows"]
            name_seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")
            order = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, epoch, name_seed, 1]))).permutation(rows)
            first = row_position if position == match_position else 0
            if first >= rows or first % batch_rows:
                raise ValueError("Resume row is not a saved batch boundary")
            for begin in range(first, rows, batch_rows):
                indices = order[begin:begin + batch_rows]
                counts = actor["candidate_offsets"][indices + 1] - actor["candidate_offsets"][indices]
                maximum = int(counts.max())
                target_per_row = sum(a.dtype.itemsize for key, a in targets.items() if key != "behavior_probabilities")
                predicted_batch = len(indices) * (34 * TILE_CHANNELS * 4 + GLOBAL_FEATURES * 4
                    + MAX_HISTORY * HISTORY_FIELDS * 2 + MAX_HISTORY + maximum * (ACTION_FIELDS * 2 + 16 + 1 + 8)
                    + target_per_row + 8)
                if shard_bytes + order.nbytes + 2 * predicted_batch + self.max_shard_bytes > self.memory_limit_bytes:
                    raise MemoryError("Shard and padded batch exceed streaming memory bound")
                batch_actor, batch_targets = self._batch(actor, targets, indices)
                batch_bytes = sum(a.nbytes for a in (*batch_actor.values(), *batch_targets.values())) + indices.nbytes
                working_bytes = shard_bytes + order.nbytes + batch_bytes
                # Reserve one full shard for transient NPZ decompression copies.
                if working_bytes + self.max_shard_bytes > self.memory_limit_bytes:
                    raise MemoryError("Shard and padded batch exceed streaming memory bound")
                self.peak_working_array_bytes = max(self.peak_working_array_bytes, working_bytes)
                end = begin + len(indices)
                cursor_after = self._cursor(epoch=epoch, seed=seed, batch_rows=batch_rows,
                    match_position=position + int(end == rows), row_position=0 if end == rows else end)
                yield StreamBatch(batch_actor, batch_targets, name, indices.copy(), cursor_after, working_bytes)
            del actor, targets, order
            self.loaded_match = None

    @staticmethod
    def _batch(actor: dict[str, np.ndarray], targets: dict[str, np.ndarray],
               indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        offsets = actor["candidate_offsets"]
        counts = offsets[indices + 1] - offsets[indices]
        rows, candidates = len(indices), int(counts.max())
        output = {"tiles": actor["tiles"][indices], "globals": actor["globals"][indices],
                  "history": np.zeros((rows, MAX_HISTORY, HISTORY_FIELDS), dtype=np.int16),
                  "history_mask": np.zeros((rows, MAX_HISTORY), dtype=np.bool_),
                  "action_tokens": np.zeros((rows, candidates, ACTION_FIELDS), dtype=np.int16),
                  "candidate_efficiency": np.zeros((rows, candidates, 4), dtype=np.float32),
                  "candidate_mask": np.zeros((rows, candidates), dtype=np.bool_)}
        labels = {key: value[indices] for key, value in targets.items() if key != "behavior_probabilities"}
        labels["behavior_probabilities"] = np.zeros((rows, candidates), dtype=np.float64)
        for item, row in enumerate(indices):
            start, end = map(int, offsets[row:row + 2])
            hstart, length = int(actor["history_start"][row]), int(actor["history_length"][row])
            output["history"][item, :length] = actor["history_tokens"][hstart:hstart + length]
            output["history_mask"][item, :length] = True
            output["action_tokens"][item, :end - start] = actor["action_tokens"][start:end]
            output["candidate_efficiency"][item, :end - start] = actor["candidate_efficiency"][start:end]
            output["candidate_mask"][item, :end - start] = True
            labels["behavior_probabilities"][item, :end - start] = targets["behavior_probabilities"][start:end]
        return output, labels
