"""Stage 3G role-bound, single-match offline data adapter.

Only actor and target numerical shards are opened. Validation/decompression and
causal-history construction reuse the frozen Stage 3F primitives, but historical
Stage 3F manifests cannot become eligible training data through this adapter.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterator, Any

import numpy as np

from .stage3f_stream import (StreamDataset, StreamBatch, ACTOR_SCHEMA,
                             TARGET_SCHEMA, _positive)

CORPUS_SCHEMA = 'riichi-ai-stage3g-corpus-v1'
COLLECTION_SCHEMA = 'riichi-ai-stage3g-collection-v1'
BEHAVIORS = ('shanten_v2', 'defense_v2', 'reference', 'explorer')
TARGET_MAPPING_SCHEMA = 'riichi-ai-stage3g-hand-targets-v1'
ROLE_SOURCE = {'training': 'offline-training', 'development': 'development-only',
               'preflight': 'preflight-only'}


@dataclass
class ExplorerMatch:
    match_id: str
    actor: dict[str, np.ndarray]
    targets: dict[str, np.ndarray]
    indices: np.ndarray
    shard_bytes: int
    total_rows: int
    explorer_rows: int

    @property
    def optional_rows(self) -> int:
        return len(self.indices)


class LearningDataset(StreamDataset):
    """Read one complete match at a time; role gates precede every tensor load."""
    def __init__(self, directory: str | Path, split: str = 'training', *,
                 max_shard_bytes: int = 64 << 20, memory_limit_bytes: int = 256 << 20):
        self.directory = Path(directory).expanduser().resolve()
        self.max_shard_bytes = _positive(max_shard_bytes, 'max_shard_bytes')
        self.memory_limit_bytes = _positive(memory_limit_bytes, 'memory_limit_bytes')
        if memory_limit_bytes <= 2 * max_shard_bytes:
            raise ValueError('Memory limit must reserve two shards plus a batch')
        raw = (self.directory / 'manifest.json').read_bytes()
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        manifest = json.loads(raw)
        if (manifest.get('schema') != CORPUS_SCHEMA or manifest.get('complete') is not True
                or manifest.get('source_role') not in ('offline-learning-corpus', 'preflight-only')):
            raise ValueError('Only finalized Stage 3G learning/preflight manifests are accepted')
        if split not in ROLE_SOURCE:
            raise ValueError('Unknown learning split')
        if manifest['source_role'] == 'preflight-only' and split != 'preflight':
            raise ValueError('Preflight and formal roles cannot be interchanged')
        reports, splits = manifest.get('matches'), manifest.get('splits')
        if not isinstance(reports, list) or not reports or not isinstance(splits, dict) or split not in splits:
            raise ValueError('Missing matches or split')
        allowed_splits = {'preflight'} if split == 'preflight' else {'training', 'development'}
        if set(splits) != allowed_splits:
            raise ValueError('Unexpected corpus split roles')
        if any(not isinstance(names, list) or any(not isinstance(n, str) for n in names) for names in splits.values()):
            raise ValueError('Invalid split members')
        assigned = [n for names in splits.values() for n in names]
        if len(assigned) != len(set(assigned)):
            raise ValueError('Split overlap')
        roles = {name: role for role, names in splits.items() for name in names}
        self.reports = {}
        for report in reports:
            name = report.get('match_id')
            role = roles.get(name)
            if (not isinstance(name, str) or not name or Path(name).name != name or name in self.reports
                    or report.get('schema') != COLLECTION_SCHEMA
                    or role is None or report.get('completed') is not True or report.get('failure') is not None
                    or report.get('role') != role or report.get('source_role') != ROLE_SOURCE[role]
                    or report.get('training_eligible') is not (role == 'training')
                    or report.get('ppo_eligible') is not False
                    or report.get('actor_schema') != ACTOR_SCHEMA or report.get('target_schema') != TARGET_SCHEMA):
                raise ValueError('Invalid match role, eligibility or numerical schema')
            assigned_behaviors = report.get('seat_behaviors')
            if (not isinstance(assigned_behaviors, list)
                    or any(not isinstance(value, str) for value in assigned_behaviors)
                    or sorted(assigned_behaviors) != sorted(BEHAVIORS)):
                raise ValueError('Match must declare one of each fixed seat behavior')
            _positive(report.get('rows'), 'manifest rows')
            for area in ('actor', 'targets'):
                relative = f'{area}/{name}.npz'
                artifact = report.get('artifacts', {}).get(relative)
                if (not isinstance(artifact, dict) or not isinstance(artifact.get('sha256'), str)
                        or len(artifact['sha256']) != 64 or type(artifact.get('bytes')) is not int or artifact['bytes'] <= 0):
                    raise ValueError('Missing numeric artifact identity')
                path = self.directory / relative
                if path.resolve().parent != self.directory / area or path.is_symlink():
                    raise ValueError('Shard escapes its declared directory')
            self.reports[name] = report
        if set(assigned) != set(self.reports):
            raise ValueError('Split coverage mismatch')
        self.match_ids = tuple(splits[split])
        if not self.match_ids:
            raise ValueError('Requested split is empty')
        for area in ('actor', 'targets'):
            if {p.name for p in (self.directory / area).glob('*.npz')} != {f'{n}.npz' for n in self.reports}:
                raise ValueError('Unmanifested or missing numeric shard')
        self.split = split
        self.loaded_match = None
        self.peak_working_array_bytes = self.shards_opened = self.bytes_read = 0
        self.manifest = manifest

    def _load(self, name: str):
        actor, targets, size = super()._load(name)
        declared = np.asarray([BEHAVIORS.index(value)
            for value in self.reports[name]['seat_behaviors']], dtype=np.int8)
        if not np.array_equal(targets['behavior_id'], declared[targets['seat']]):
            raise ValueError('Numeric behavior IDs differ from the declared seat assignment')
        return actor, targets, size

    def match_order(self, epoch: int, seed: int) -> tuple[str, ...]:
        for value in (epoch, seed):
            if type(value) is not int or not 0 <= value < 2 ** 64:
                raise ValueError('Epoch and seed must be unsigned integers')
        order = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, epoch, 0]))).permutation(len(self.match_ids))
        return tuple(self.match_ids[int(i)] for i in order)

    def load_explorer(self, name: str, *, epoch: int, seed: int) -> ExplorerMatch:
        if name not in self.match_ids:
            raise ValueError('Match is outside the selected split')
        actor, targets, size = self._load(name)
        explorer = targets['behavior_id'] == 3
        seats = np.unique(targets['seat'][explorer])
        if len(seats) != 1:
            raise ValueError('Exactly one fixed explorer seat is required')
        eligible = np.flatnonzero(explorer & (np.diff(actor['candidate_offsets']) > 1))
        name_seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], 'big')
        order = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, epoch, name_seed, 1]))).permutation(len(eligible))
        indices = eligible[order]
        return ExplorerMatch(name, actor, targets, indices, size, len(explorer), int(explorer.sum()))

    def batches(self, match: ExplorerMatch, batch_rows: int = 256) -> Iterator[StreamBatch]:
        _positive(batch_rows, 'batch_rows')
        if batch_rows > 256:
            raise ValueError('Stage 3G batches are capped at 256 rows')
        for start in range(0, match.optional_rows, batch_rows):
            indices = match.indices[start:start + batch_rows]
            # Bound the padded expansion before allocation, using the same schema dimensions.
            counts = np.diff(match.actor['candidate_offsets'])[indices]
            from .features import TILE_CHANNELS, GLOBAL_FEATURES, MAX_HISTORY, HISTORY_FIELDS, ACTION_FIELDS
            predicted = len(indices) * (34 * TILE_CHANNELS * 4 + GLOBAL_FEATURES * 4
                + MAX_HISTORY * HISTORY_FIELDS * 2 + MAX_HISTORY
                + int(counts.max()) * (ACTION_FIELDS * 2 + 16 + 1 + 8) + 128)
            if match.shard_bytes + match.indices.nbytes + 2 * predicted + self.max_shard_bytes > self.memory_limit_bytes:
                raise MemoryError('Filtered batch exceeds fixed memory bound')
            actor, targets = self._batch(match.actor, match.targets, indices)
            size = match.shard_bytes + match.indices.nbytes + sum(a.nbytes for a in (*actor.values(), *targets.values()))
            if size + self.max_shard_bytes > self.memory_limit_bytes:
                raise MemoryError('Filtered batch exceeds fixed memory bound')
            self.peak_working_array_bytes = max(size, self.peak_working_array_bytes)
            yield StreamBatch(actor, targets, match.match_id, indices.copy(), {}, size)


class HandTargets:
    """Immutable explorer-only one-hand deltas; never an actor input."""
    def __init__(self, path: str | Path, dataset: LearningDataset):
        raw = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.payload = json.loads(raw)
        p = self.payload
        if (p.get('schema') != TARGET_MAPPING_SCHEMA or p.get('corpus_manifest_sha256') != dataset.manifest_sha256
                or p.get('gate', {}).get('passed') is not True or p.get('calibration_gate_passed') is not True
                or p.get('target') != 'one_hand_placement_potential_difference'):
            raise ValueError('Hand target identity or calibration gate failed')
        if not isinstance(p.get('matches'), dict) or set(p['matches']) != set(dataset.reports):
            raise ValueError('Hand target match coverage mismatch')
        for name, entry in p['matches'].items():
            digest = entry.get('predictor_sha256')
            seat, deltas = entry.get('explorer_seat'), entry.get('hand_deltas')
            if (entry.get('source_role') != dataset.reports[name]['source_role']
                    or type(seat) is not int or not 0 <= seat < 4 or not isinstance(digest, str) or len(digest) != 64
                    or not isinstance(deltas, dict) or not deltas):
                raise ValueError('Invalid hand target provenance')
            for hand, value in deltas.items():
                if (not hand.isdigit() or str(int(hand)) != hand or isinstance(value, bool)
                        or not isinstance(value, (int, float)) or not np.isfinite(value) or not -3 <= value <= 3):
                    raise ValueError('Invalid hand delta value or ordinal')

    def for_batch(self, batch: StreamBatch) -> np.ndarray:
        entry = self.payload['matches'][batch.match_id]
        if np.any(batch.targets['seat'] != entry['explorer_seat']) or np.any(batch.targets['behavior_id'] != 3):
            raise ValueError('Hand delta requested for a non-explorer decision')
        try:
            values = [entry['hand_deltas'][str(int(hand))] for hand in batch.targets['hand_id']]
        except KeyError as exc:
            raise ValueError('Missing explicit hand delta') from exc
        return np.asarray(values, dtype=np.float32)
