"""Synthetic match data with separate actor inputs, targets and audit records.

No external game corpus is read. Each independent full match has one split and
seed before any player view is extracted. Auditor files are never opened by
the default training loader.
"""

from __future__ import annotations

from collections import Counter, OrderedDict, deque
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
from importlib.metadata import version
import io
import json
import multiprocessing
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Iterator, Sequence
import zipfile

import numpy as np

from .player import PlayerAdapter
from .policies import policy_manifest
from .rules import audit_physical_tiles, resolved_rule_profile, rule_profile_hash
from .semantics import SEMANTIC_ACTION_VERSION, TEACHER_VERSION, teacher_decision
from .simulator import INITIAL_SCORES, create_env, score_total


DATASET_SCHEMA = "riichi-ai-synthetic-dataset-v1"
ACTOR_SCHEMA = "riichi-ai-actor-arrays-v1"
TARGET_SCHEMA = "riichi-ai-imitation-targets-v1"
AUDITOR_SCHEMA = "riichi-ai-auditor-match-v1"
SEED_NAMESPACE = "riichi-ai/stage2/synthetic-training/v1"
SPLITS = ("train", "validation", "test")
DEFAULT_DISK_LIMIT = 5_000_000_000
DEFAULT_MEMORY_LIMIT = 32_000_000_000


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _stage1_seeds() -> set[int]:
    """Reject recorded development roots and exact wall seeds, when present."""
    seeds: set[int] = set()
    directory = Path(__file__).resolve().parents[2] / "reports/stage1"
    for filename in ("shanten-v-random.json", "defense-v-shanten.json"):
        path = directory / filename
        if path.exists():
            report = json.loads(path.read_text())
            seeds.add(report["configuration"]["root_seed"])
            seeds.update(match["seed"] for match in report["matches"])
    return seeds


def plan_matches(matches: int, root_seed: int) -> list[dict[str, Any]]:
    """Preassign exact floor-80%/floor-10%/remainder splits and unique seeds."""
    _positive_integer(matches, "matches")
    if type(root_seed) is not int or not 0 <= root_seed < 2**64:
        raise ValueError("root_seed must be an integer in [0, 2**64)")
    forbidden = _stage1_seeds()
    if root_seed in forbidden:
        raise ValueError("Stage 1 development seed cannot be a dataset root")
    train_end, validation_end = matches * 8 // 10, matches * 8 // 10 + matches // 10
    result = []
    seen: set[int] = set()
    for index in range(matches):
        split = "train" if index < train_end else "validation" if index < validation_end else "test"
        digest = hashlib.sha256(_json([SEED_NAMESPACE, root_seed, split, index])).digest()
        seed = int.from_bytes(digest[:8], "big")
        if seed in forbidden or seed in seen:
            raise ValueError("Derived seed collides with development or another match")
        seen.add(seed)
        result.append({"match_id": f"match-{index:06d}", "index": index,
                       "split": split, "seed": seed,
                       "independent_group": f"{SEED_NAMESPACE}:{root_seed}:{index}"})
    return result


def _sources() -> dict[str, Any]:
    from .features import FEATURE_SCHEMA
    return {
        "python": platform.python_version(),
        "dependencies": {name: version(name) for name in ("numpy", "riichienv", "mahjong", "torch")},
        "sources": {path.name: _hash(path.read_bytes()) for path in sorted(Path(__file__).parent.glob("*.py"))},
        "feature_schema": FEATURE_SCHEMA, "semantic_action_schema": SEMANTIC_ACTION_VERSION,
        "teacher_version": TEACHER_VERSION,
    }


def _git_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                text=True, timeout=5, check=False)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                               capture_output=True, text=True, timeout=5, check=False)
        return {"commit": commit.stdout.strip() if commit.returncode == 0 else None,
                "tracked_worktree_dirty": bool(dirty.stdout.strip()),
                "status": "recorded" if commit.returncode == 0 else "no_commit"}
    except (OSError, subprocess.TimeoutExpired):
        return {"commit": None, "tracked_worktree_dirty": None, "status": "unavailable"}


def _pack_arrays(arrays: dict[str, np.ndarray]) -> bytes:
    """Write deterministic NPZ bytes (fixed ZIP timestamps, no pickle)."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for key, value in sorted(arrays.items()):
            if value.dtype.hasobject:
                raise ValueError("Object arrays are not allowed in dataset shards")
            payload = io.BytesIO()
            np.lib.format.write_array(payload, value, allow_pickle=False)
            info = zipfile.ZipInfo(key + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    return out.getvalue()


def _rare_paths(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Count explicit settlements only; never fabricate unavailable yaku labels."""
    counts = Counter(event.get("type", "unknown") for event in events)
    reasons = Counter(str(event["reason"]) for event in events
                      if event.get("type") == "ryukyoku" and "reason" in event)
    double_ron = 0
    wins = []
    for event in events:
        if event.get("type") == "start_kyoku":
            wins = []
        elif event.get("type") == "hora":
            wins.append(event)
        elif event.get("type") == "end_kyoku":
            double_ron += int(len(wins) == 2 and all(e.get("actor") != e.get("target") for e in wins))
    observed = {
        "double_ron": double_ron, "triple_ron_abort": reasons["sanchaho"],
        "ankan_declarations": counts["ankan"], "daiminkan_declarations": counts["daiminkan"],
        "kakan_declarations": counts["kakan"],
    }
    return {
        "observed_counts": observed, "ryukyoku_reasons": dict(sorted(reasons.items())),
        "event_counts": dict(sorted(counts.items())),
        "uninstrumented": {name: {"status": "not_instrumented", "count": None}
                           for name in ("haitei", "houtei", "rinshan", "chankan", "nagashi_mangan",
                                        "pao", "riichi_refused_for_wall_length")},
        "denominators": {"hands": counts["start_kyoku"], "winning_settlements": counts["hora"],
                         "independent_matches": 1},
        "limitation": "Counts describe this policy mixture; unobserved or uninstrumented paths are not validated.",
    }


def _rss() -> int:
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except ImportError:
        return 0


def _process_tree_rss() -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    try:
        process = psutil.Process()
        total = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        return total
    except (PermissionError, psutil.AccessDenied):
        # Sandboxed macOS may permit self RSS but deny process enumeration.
        # Preserve an explicit unknown rather than invent an aggregate bound.
        return None


def _bounded_results(executor: ProcessPoolExecutor, jobs: list[dict[str, Any]],
                     workers: int) -> Iterator[dict[str, Any]]:
    """Ordered, bounded results on every supported Python (including 3.11)."""
    pending = deque()
    source = iter(jobs)
    for _ in range(workers):
        job = next(source, None)
        if job is not None:
            pending.append(executor.submit(_generate_match, job))
    while pending:
        yield pending.popleft().result()
        job = next(source, None)
        if job is not None:
            pending.append(executor.submit(_generate_match, job))


def _generate_match(job: dict[str, Any]) -> dict[str, Any]:
    """Generate one match in isolation; parent writes returned bounded artifacts."""
    from .features import encode_history_events, encode_view
    plan, teacher = job["plan"], job["teacher"]
    started = time.perf_counter()
    env, observations = create_env(plan["seed"])
    adapters = [PlayerAdapter(seat) for seat in range(4)]
    policies = [policy_manifest(teacher, plan["seed"] ^ seat) for seat in range(4)]
    arrays: dict[str, list[Any]] = {key: [] for key in (
        "tiles", "globals", "action_tokens", "policy_target", "best_mask", "preference_keys",
        "chosen_index", "seat", "step", "hand_id", "physical_candidate_count",
    )}
    offsets = [0]
    history_segments: dict[tuple[int, int], list[np.ndarray]] = {}
    history_lengths: dict[tuple[int, int], int] = {}
    history_refs = []
    requests = []
    chosen_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    tie_sizes: Counter[int] = Counter()
    optional_tie_sizes: Counter[int] = Counter()
    physical_forced = semantic_forced = 0
    peak_rss = _rss()
    timings = {key: 0.0 for key in ("view", "teacher", "encode", "history", "step", "audit", "pack")}
    failure = None
    steps = 0
    try:
        while not env.done() and steps < job["max_steps"]:
            t = time.perf_counter()
            audit_physical_tiles(env)
            if score_total(env) != sum(INITIAL_SCORES):
                raise ValueError("Point/deposit conservation failure")
            timings["audit"] += time.perf_counter() - t
            t = time.perf_counter()
            # Consume every returned snapshot before any response is applied.
            views = {seat: adapters[seat].build(observations[seat]) for seat in sorted(observations)}
            timings["view"] += time.perf_counter() - t
            chosen = {}
            batch_requests = []
            for seat, view in views.items():
                if not view.legal_actions:
                    continue
                t = time.perf_counter()
                decision = teacher_decision(view, defense=teacher == "defense_v2")
                timings["teacher"] += time.perf_counter() - t
                t = time.perf_counter()
                encoding = encode_view(view, decision.actions)
                timings["encode"] += time.perf_counter() - t
                count = len(decision.actions)
                if count == 0 or not decision.best_mask[decision.chosen_index]:
                    raise ValueError("Teacher choice is absent from its best semantic set")
                if not np.isclose(sum(decision.soft_targets), 1):
                    raise ValueError("Policy target is not normalized")
                for key in ("tiles", "globals", "action_tokens"):
                    value = np.asarray(encoding[key])
                    if not np.isfinite(value).all():
                        raise ValueError("Nonfinite actor feature")
                    arrays[key].append(value)
                if encoding["action_tokens"].shape[0] != count:
                    raise ValueError("Encoder and teacher candidate order disagree")
                t = time.perf_counter()
                segment = (seat, view.hand_id)
                previous = history_lengths.get(segment, 0)
                if len(view.history) < previous:
                    raise ValueError("History shrank without a new hand ID")
                added = encode_history_events(view.history[previous:], seat)
                history_segments.setdefault(segment, []).append(added)
                history_lengths[segment] = len(view.history)
                history_refs.append((segment, len(view.history)))
                timings["history"] += time.perf_counter() - t
                arrays["policy_target"].append(np.asarray(decision.soft_targets, dtype=np.float32))
                arrays["best_mask"].append(np.asarray(decision.best_mask, dtype=np.bool_))
                arrays["preference_keys"].append(np.asarray(decision.preference_keys, dtype=np.int16))
                for key, value in (("chosen_index", decision.chosen_index), ("seat", seat),
                                   ("step", steps), ("hand_id", view.hand_id),
                                   ("physical_candidate_count", len(view.legal_actions))):
                    arrays[key].append(value)
                offsets.append(offsets[-1] + count)
                physical_forced += int(len(view.legal_actions) == 1)
                semantic_forced += int(count == 1)
                tie_sizes[sum(decision.best_mask)] += 1
                if count > 1:
                    optional_tie_sizes[sum(decision.best_mask)] += 1
                selected = decision.actions[decision.chosen_index]
                chosen_counts[selected.kind] += 1
                candidate_counts.update(action.kind for action in decision.actions)
                chosen[seat] = adapters[seat].resolve(selected.representative, observations[seat])
                batch_requests.append({
                    "row": len(history_refs) - 1, "seat": seat, "hand_id": view.hand_id,
                    "physical_candidates": [action.token for action in view.legal_actions],
                    "semantic_keys": [action.key for action in decision.actions],
                    "preference_keys": decision.preference_keys, "best_mask": decision.best_mask,
                    "chosen_index": decision.chosen_index, "submitted_action": selected.representative.token,
                    "behavior_probability": 1.0,
                })
            if not chosen:
                raise ValueError("Unfinished match supplied no legal requests")
            requests.append({"step": steps, "requests": batch_requests})
            t = time.perf_counter()
            observations = env.step(chosen)
            timings["step"] += time.perf_counter() - t
            steps += 1
            if steps % 64 == 0:
                peak_rss = max(peak_rss, _rss())
                if peak_rss > job["worker_memory_limit"]:
                    raise MemoryError("Worker exceeded its share of the aggregate memory target")
        audit_physical_tiles(env)
        if score_total(env) != sum(INITIAL_SCORES):
            raise ValueError("Terminal point/deposit conservation failure")
        if not env.done():
            failure = {"type": "TruncatedMatch", "message": "max_steps reached", "step": steps}
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error), "step": steps}
    completed = env.done() and failure is None
    events = env.mjai_log  # Export once; avoid quadratic repeated log serialization.
    ranks = list(env.ranks()) if completed else None
    if completed and sorted(ranks) != [1, 2, 3, 4]:
        failure = {"type": "InvalidTerminalRanks", "message": "Ranks are not a permutation", "step": steps}
        completed, ranks = False, None
    audit_record = {
        "schema": AUDITOR_SCHEMA, "plan": plan, "policy_provenance": policies,
        "rules_hash": rule_profile_hash(), "canonical_events": events,
        "policy_requests": requests, "completed": completed, "failure": failure,
        "final_ranks": ranks, "final_scores": list(env.scores()),
        "behavior_note": "Deterministic representative choice; soft tie targets are not behavior probabilities.",
    }
    raw_audit = _json(audit_record)
    artifacts = {f"auditor/{plan['match_id']}.json.gz": gzip.compress(raw_audit, mtime=0)}
    uncompressed_bytes = 0
    t = time.perf_counter()
    if completed:
        segments, starts = [], {}
        cursor = 0
        for key, chunks in history_segments.items():
            rows = np.concatenate(chunks, axis=0)
            starts[key] = cursor
            cursor += len(rows)
            segments.append(rows)
        actor = {
            "tiles": np.stack(arrays["tiles"]).astype(np.float32),
            "globals": np.stack(arrays["globals"]).astype(np.float32),
            "action_tokens": np.concatenate(arrays["action_tokens"]).astype(np.int16),
            "candidate_offsets": np.asarray(offsets, dtype=np.int64),
            "history_tokens": np.concatenate(segments).astype(np.int16),
            "history_start": np.asarray([starts[key] + max(0, length - 128) for key, length in history_refs], dtype=np.int64),
            "history_length": np.asarray([min(128, length) for _, length in history_refs], dtype=np.int16),
        }
        targets = {
            "policy_target": np.concatenate(arrays["policy_target"]),
            "best_mask": np.concatenate(arrays["best_mask"]),
            "teacher_preference_keys": np.concatenate(arrays["preference_keys"]),
            "chosen_index": np.asarray(arrays["chosen_index"], dtype=np.int16),
            "final_rank": np.asarray([ranks[seat] for seat in arrays["seat"]], dtype=np.int8),
            "seat": np.asarray(arrays["seat"], dtype=np.int8),
            "step": np.asarray(arrays["step"], dtype=np.int32),
            "hand_id": np.asarray(arrays["hand_id"], dtype=np.int32),
            "physical_candidate_count": np.asarray(arrays["physical_candidate_count"], dtype=np.int16),
        }
        artifacts[f"actor/{plan['match_id']}.npz"] = _pack_arrays(actor)
        artifacts[f"targets/{plan['match_id']}.npz"] = _pack_arrays(targets)
        uncompressed_bytes = sum(array.nbytes for array in (*actor.values(), *targets.values()))
    timings["pack"] = time.perf_counter() - t
    peak_rss = max(peak_rss, _rss())
    report = {
        **plan, "completed": completed, "failure": failure, "steps": steps,
        "decisions": len(history_refs), "hands": sum(e.get("type") == "start_kyoku" for e in events),
        "final_ranks": ranks, "final_scores": list(env.scores()), "game_hash": _hash(_json(events)),
        "physical_forced": physical_forced, "semantic_forced": semantic_forced,
        "chosen_action_counts": dict(sorted(chosen_counts.items())),
        "semantic_candidate_counts": dict(sorted(candidate_counts.items())),
        "tie_set_sizes": {str(k): v for k, v in sorted(tie_sizes.items())},
        "optional_tie_set_sizes": {str(k): v for k, v in sorted(optional_tie_sizes.items())},
        "seat_samples": {str(seat): arrays["seat"].count(seat) for seat in range(4)},
        "rare_paths": _rare_paths(events), "uncompressed_training_bytes": uncompressed_bytes,
        "peak_worker_rss_bytes": peak_rss, "elapsed_seconds": time.perf_counter() - started,
        "timing_seconds": timings,
        "artifacts": {name: {"sha256": _hash(content), "bytes": len(content)}
                      for name, content in sorted(artifacts.items())},
    }
    return {"report": report, "artifacts": artifacts}


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def generate_dataset(
    output_dir: str | Path, matches: int = 1000, root_seed: int = 2026101701,
    workers: int = 1, max_steps: int = 100000, *, teacher: str = "shanten_v2",
    disk_limit_bytes: int = DEFAULT_DISK_LIMIT, memory_limit_bytes: int = DEFAULT_MEMORY_LIMIT,
) -> dict[str, Any]:
    """Generate a bounded pilot; failure/truncation produces no training shard.

    Small spawn pools operate on independent matches. Results and manifests are
    written in planned order. Stop at the first failure or resource limit and
    explicitly list unexecuted matches. Existing nonempty output is rejected.
    """
    for value, name in ((workers, "workers"), (max_steps, "max_steps"),
                        (disk_limit_bytes, "disk_limit_bytes"), (memory_limit_bytes, "memory_limit_bytes")):
        _positive_integer(value, name)
    if workers > 8:
        raise ValueError("Pilot workers must be between 1 and 8; profile before increasing")
    if teacher not in ("shanten_v2", "defense_v2"):
        raise ValueError("Pilot teacher must be shanten_v2 or defense_v2")
    plan = plan_matches(matches, root_seed)
    directory = Path(output_dir).expanduser().resolve()
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Dataset output must be empty; historical runs are never overwritten")
    directory.mkdir(parents=True, exist_ok=True)
    provenance = _sources()
    plan_record = {"schema": DATASET_SCHEMA, "root_seed": root_seed, "seed_namespace": SEED_NAMESPACE,
                   "split_method": "ordered-independent-matches-floor80-floor10-remainder",
                   "matches": plan}
    plan_bytes = _json(plan_record)
    if len(plan_bytes) >= disk_limit_bytes:
        raise ValueError("Disk limit cannot accommodate the split plan")
    _write(directory / "plan.json", plan_bytes)
    bytes_written = len(plan_bytes)
    started = time.perf_counter()
    reports = []
    failure = None
    sampled_peak_process_tree_rss = _process_tree_rss()
    jobs = [{"plan": item, "teacher": teacher, "max_steps": max_steps,
             "worker_memory_limit": memory_limit_bytes // (workers + 1)} for item in plan]
    executor = None
    try:
        if workers == 1:
            results: Iterator[dict[str, Any]] = map(_generate_match, jobs)
        else:
            executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
            results = _bounded_results(executor, jobs, workers)
        for result in results:
            report, artifacts = result["report"], result["artifacts"]
            next_bytes = sum(len(content) for content in artifacts.values())
            # Reserve room for the final audit manifest rather than fill disk.
            manifest_reserve = max(100_000, len(plan) * 5000)
            if bytes_written + next_bytes + manifest_reserve > disk_limit_bytes:
                failure = {"type": "DiskBudgetExceeded", "match_id": report["match_id"],
                           "message": "Next artifacts would exceed disk cap including manifest reserve"}
                break
            for name, content in sorted(artifacts.items()):
                _write(directory / name, content)
            bytes_written += next_bytes
            reports.append(report)
            if not report["completed"]:
                failure = report["failure"]
                break
            sample = _process_tree_rss()
            if sample is not None:
                sampled_peak_process_tree_rss = max(sampled_peak_process_tree_rss or 0, sample)
            if sampled_peak_process_tree_rss is not None and sampled_peak_process_tree_rss > memory_limit_bytes:
                failure = {"type": "MemoryBudgetExceeded", "message": "Sampled process tree exceeded aggregate memory target"}
                break
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    sources_unchanged = provenance == _sources()
    counts = Counter(report["game_hash"] for report in reports if report["completed"])
    duplicate_hashes = [digest for digest, count in counts.items() if count > 1]
    chosen, candidates, ties, optional_ties, seats, rare, reasons, event_counts = (Counter() for _ in range(8))
    for report in reports:
        chosen.update(report["chosen_action_counts"])
        candidates.update(report["semantic_candidate_counts"])
        ties.update(report["tie_set_sizes"])
        optional_ties.update(report["optional_tie_set_sizes"])
        seats.update(report["seat_samples"])
        rare.update(report["rare_paths"]["observed_counts"])
        reasons.update(report["rare_paths"]["ryukyoku_reasons"])
        event_counts.update(report["rare_paths"]["event_counts"])
    by_id = {item["match_id"]: item for item in plan}
    actual_ids = [report["match_id"] for report in reports]
    actual_seeds = [report["seed"] for report in reports]
    reused_ids = len(actual_ids) != len(set(actual_ids))
    reused_seeds = len(actual_seeds) != len(set(actual_seeds))
    wrong_assignments = [report["match_id"] for report in reports
                         if any(report[key] != by_id[report["match_id"]][key]
                                for key in ("split", "seed", "independent_group"))]
    valid = (failure is None and len(reports) == matches and sources_unchanged and not duplicate_hashes
             and not reused_ids and not reused_seeds and not wrong_assignments)
    manifest = {
        "schema": DATASET_SCHEMA, "actor_schema": ACTOR_SCHEMA, "target_schema": TARGET_SCHEMA,
        "auditor_schema": AUDITOR_SCHEMA, "seed_namespace": SEED_NAMESPACE,
        "root_seed": root_seed, "requested_matches": matches, "workers": workers, "max_steps": max_steps,
        "teacher": policy_manifest(teacher), "source_mixture": {teacher: "all four seats"},
        "closed_hand_limitation": "Teachers pass optional calls and kans; absent paths are not validated.",
        "provenance": provenance, "git": _git_state(), "sources_unchanged": sources_unchanged,
        "rule_profile": resolved_rule_profile(), "rule_profile_hash": rule_profile_hash(),
        "plan_sha256": _hash(plan_bytes), "valid": valid, "failure": failure,
        "completed_matches": sum(report["completed"] for report in reports),
        "rejected_matches": [report["match_id"] for report in reports if not report["completed"]],
        "not_written_matches": [item["match_id"] for item in plan[len(reports):]],
        "splits": {split: [r["match_id"] for r in reports if r["split"] == split and r["completed"]]
                   for split in SPLITS},
        "split_leakage": {"shared_match_ids": reused_ids, "shared_seeds": reused_seeds,
                          "wrong_split_assignments": wrong_assignments,
                          "exact_duplicate_game_hashes": duplicate_hashes,
                          "development_seed_rejection": "Stage 1 report roots and exact match seeds checked"},
        "decisions": sum(report["decisions"] for report in reports),
        "hands": sum(report["hands"] for report in reports),
        "physical_forced": sum(report["physical_forced"] for report in reports),
        "semantic_forced": sum(report["semantic_forced"] for report in reports),
        "chosen_action_counts": dict(sorted(chosen.items())),
        "semantic_candidate_counts": dict(sorted(candidates.items())), "tie_set_sizes": dict(sorted(ties.items())),
        "optional_tie_set_sizes": dict(sorted(optional_ties.items())),
        "optional_multiple_best_fraction": (sum(v for k, v in optional_ties.items() if int(k) > 1)
                                              / sum(optional_ties.values()) if optional_ties else None),
        "seat_samples": dict(sorted(seats.items())), "observed_rare_path_counts": dict(sorted(rare.items())),
        "ryukyoku_reasons": dict(sorted(reasons.items())), "event_counts": dict(sorted(event_counts.items())),
        "uninstrumented_rare_paths": ["haitei", "houtei", "rinshan", "chankan", "nagashi_mangan",
                                       "pao", "riichi_refused_for_wall_length"],
        "belief_labels": "deferred; auditor secrets are separate and the actor loader never reads them",
        "disk_limit_bytes": disk_limit_bytes, "memory_target_bytes": memory_limit_bytes,
        "artifact_bytes_before_manifest": bytes_written,
        "uncompressed_training_bytes": sum(report["uncompressed_training_bytes"] for report in reports),
        "peak_worker_rss_bytes": max((report["peak_worker_rss_bytes"] for report in reports), default=0),
        "sampled_peak_process_tree_rss_bytes": sampled_peak_process_tree_rss,
        "memory_monitor": "Sampled psutil RSS; process-tree value null means unavailable. Worker self RSS and array byte limits remain checked; aggregate RSS is not then verified. Not an OS hard limit.",
        "elapsed_seconds": time.perf_counter() - started,
        "timing_seconds": {key: sum(r["timing_seconds"][key] for r in reports)
                           for key in ("view", "teacher", "encode", "history", "step", "audit", "pack")},
        "matches": reports,
    }
    payload = _json(manifest)
    if bytes_written + len(payload) > disk_limit_bytes:
        raise RuntimeError("Manifest reserve was insufficient; artifacts retained for audit")
    _write(directory / "manifest.json", payload)
    return manifest


class SplitDataset(Sequence[dict[str, Any]]):
    """Random-access numeric samples, optionally preloaded for shuffled batches.

    Each cache item holds one match's compact arrays, not Python sample objects.
    Preload is recommended for the bounded pilot; the byte limit is explicit.
    """

    def __init__(self, directory: Path, manifest: dict[str, Any], split: str, *,
                 max_matches: int | None, optional_only: bool, preload: bool,
                 cache_matches: int, memory_limit_bytes: int):
        self.directory = directory
        selected = manifest["splits"][split]
        if max_matches is not None:
            _positive_integer(max_matches, "max_matches")
            if max_matches > len(selected):
                raise ValueError(f"Requested {max_matches} matches but split has only {len(selected)}")
            selected = selected[:max_matches]
        self.match_ids = tuple(selected)
        self._reports = {r["match_id"]: r for r in manifest["matches"]}
        self._cache: OrderedDict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = OrderedDict()
        self.cache_matches = max(1, len(selected)) if preload else cache_matches
        self.preload = preload
        self.memory_limit_bytes = memory_limit_bytes
        self.loaded_bytes = 0
        estimate = sum(self._reports[name]["uncompressed_training_bytes"] for name in selected)
        if preload and estimate > memory_limit_bytes:
            raise MemoryError("Selected split exceeds preload byte limit")
        index_parts = []
        for match_index, name in enumerate(selected):
            actor, targets = self._load(name)
            counts = np.diff(actor["candidate_offsets"])
            rows = np.flatnonzero(counts > 1) if optional_only else np.arange(len(counts))
            index_parts.append(np.column_stack((np.full(len(rows), match_index, dtype=np.int32), rows.astype(np.int32))))
        self.index = np.concatenate(index_parts) if index_parts else np.empty((0, 2), dtype=np.int32)
        self.split, self.optional_only = split, optional_only

    def _load(self, name: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]
        values = []
        for area in ("actor", "targets"):
            relative = f"{area}/{name}.npz"
            raw = (self.directory / relative).read_bytes()
            if _hash(raw) != self._reports[name]["artifacts"][relative]["sha256"]:
                raise ValueError(f"Corrupted dataset artifact: {relative}")
            with np.load(io.BytesIO(raw), allow_pickle=False) as shard:
                arrays = {key: shard[key] for key in shard.files}
            for array in arrays.values():
                array.setflags(write=False)
            values.append(arrays)
        result = (values[0], values[1])
        incoming = sum(a.nbytes for group in result for a in group.values())
        while len(self._cache) >= self.cache_matches:
            _, dropped = self._cache.popitem(last=False)
            self.loaded_bytes -= sum(a.nbytes for group in dropped for a in group.values())
        if self.loaded_bytes + incoming > self.memory_limit_bytes:
            raise MemoryError("Numeric split cache exceeds byte limit")
        self.loaded_bytes += incoming
        self._cache[name] = result
        return result

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        match_index, row = self.index[index]
        name = self.match_ids[int(match_index)]
        actor, targets = self._load(name)
        row = int(row)
        start, end = map(int, actor["candidate_offsets"][row:row + 2])
        hstart, length = int(actor["history_start"][row]), int(actor["history_length"][row])
        history = np.zeros((128, 10), dtype=np.int16)
        mask = np.zeros(128, dtype=np.bool_)
        history[:length] = actor["history_tokens"][hstart:hstart + length]
        mask[:length] = True
        encoding = {"tiles": actor["tiles"][row], "globals": actor["globals"][row],
                    "history": history, "history_mask": mask,
                    "action_tokens": actor["action_tokens"][start:end]}
        return {
            "encoded": encoding, "policy_target": targets["policy_target"][start:end],
            "best_mask": targets["best_mask"][start:end],
            "chosen_index": int(targets["chosen_index"][row]),
            "rank_target": int(targets["final_rank"][row]) - 1,
            "match_id": name, "seat": int(targets["seat"][row]), "step": int(targets["step"][row]),
            "hand_id": int(targets["hand_id"][row]), "optional": end - start > 1,
        }


def validate_manifest(output_dir: str | Path, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate split identity against the committed plan without reading audits.

    Artifact content validation belongs to the numeric loader or full verifier.
    This validates assignment metadata even when a caller edits manifest.json
    while leaving its split-plan hash untouched.
    """
    from .features import FEATURE_SCHEMA
    directory = Path(output_dir).expanduser().resolve()
    if manifest is None:
        manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest.get("schema") != DATASET_SCHEMA or manifest.get("valid") is not True
            or manifest.get("actor_schema") != ACTOR_SCHEMA or manifest.get("target_schema") != TARGET_SCHEMA
            or manifest.get("auditor_schema") != AUDITOR_SCHEMA
            or manifest.get("provenance", {}).get("feature_schema") != FEATURE_SCHEMA):
        raise ValueError("Only complete, valid datasets with the current schema can be loaded")
    raw_plan = (directory / "plan.json").read_bytes()
    if _hash(raw_plan) != manifest.get("plan_sha256"):
        raise ValueError("Dataset split plan hash mismatch")
    plan = json.loads(raw_plan)
    if (plan.get("schema") != DATASET_SCHEMA or plan.get("seed_namespace") != SEED_NAMESPACE
            or manifest.get("seed_namespace") != SEED_NAMESPACE
            or plan.get("root_seed") != manifest.get("root_seed")):
        raise ValueError("Dataset plan namespace/root metadata mismatch")
    expected = plan_matches(manifest.get("requested_matches"), manifest.get("root_seed"))
    if plan.get("matches") != expected:
        raise ValueError("Dataset plan does not match deterministic match assignments")
    reports = manifest.get("matches")
    if not isinstance(reports, list) or len(reports) != len(expected):
        raise ValueError("Dataset match report count differs from split plan")
    for report, assignment in zip(reports, expected):
        if (not isinstance(report, dict) or report.get("completed") is not True
                or report.get("failure") is not None
                or any(report.get(key) != value for key, value in assignment.items())):
            raise ValueError("Dataset match report assignment differs from split plan")
        name = assignment["match_id"]
        paths = {f"actor/{name}.npz", f"targets/{name}.npz", f"auditor/{name}.json.gz"}
        if set(report.get("artifacts", {})) != paths:
            raise ValueError("Dataset artifact names differ from assigned match")
    expected_splits = {split: [r["match_id"] for r in expected if r["split"] == split] for split in SPLITS}
    if manifest.get("splits") != expected_splits:
        raise ValueError("Dataset split membership is not the exact disjoint plan assignment")
    # The canonical plan already proves seed/group uniqueness; reports must match
    # it exactly, while duplicate complete games are rejected independently.
    game_hashes = [r.get("game_hash") for r in reports]
    if any(not isinstance(value, str) or len(value) != 64 for value in game_hashes):
        raise ValueError("Dataset has missing or invalid complete-game hashes")
    if len(set(game_hashes)) != len(game_hashes):
        raise ValueError("Dataset contains exact duplicate complete games")
    if (manifest.get("completed_matches") != len(expected) or manifest.get("failure") is not None
            or manifest.get("rejected_matches") != [] or manifest.get("not_written_matches") != []
            or manifest.get("sources_unchanged") is not True):
        raise ValueError("Dataset completion/provenance metadata is inconsistent")
    return manifest


def load_split(
    output_dir: str | Path, split: str, *, max_matches: int | None = None,
    optional_only: bool = False, preload: bool = True, cache_matches: int = 2,
    memory_limit_bytes: int = 8_000_000_000, include_beliefs: bool = False,
) -> SplitDataset:
    """Load actor/target shards only; train prefixes support 100/300/800 curves."""
    if split == "val":
        split = "validation"
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    if include_beliefs:
        raise NotImplementedError("Auditor belief-label reader is deferred from the initial pilot")
    _positive_integer(cache_matches, "cache_matches")
    _positive_integer(memory_limit_bytes, "memory_limit_bytes")
    directory = Path(output_dir).expanduser().resolve()
    manifest = validate_manifest(directory)
    return SplitDataset(directory, manifest, split, max_matches=max_matches,
                        optional_only=optional_only, preload=preload, cache_matches=cache_matches,
                        memory_limit_bytes=memory_limit_bytes)
