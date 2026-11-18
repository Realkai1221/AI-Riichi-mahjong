"""Audited full-match execution and fixed-budget, paired development evaluation.

Only this runner sees simulator state. All simultaneous policy requests use
immutable views built before any choice is submitted to the engine.
"""

from __future__ import annotations

from dataclasses import asdict
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
from importlib.metadata import version
import json
import math
import multiprocessing
from pathlib import Path
import subprocess
import statistics
import time
from typing import Any, Sequence

import numpy as np

from .player import ActionSpec, PlayerAdapter
from .auditing import PlayerViewHasher, VIEW_HASH_SCHEMA
from .policies import make_policy, policy_manifest
from .rules import audit_physical_tiles, resolved_rule_profile, rule_profile_hash
from .simulator import INITIAL_SCORES, _validate_seed, create_env, score_total


SCHEMA = "riichi-ai-evaluation-v2"
RANK_UTILITY = (1.5, 0.5, -0.5, -1.5)


def _checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=4)
def _load_neural(path: str, expected_hash: str) -> Any:
    from .model import NeuralPolicy
    import torch
    torch.set_num_threads(1)
    policy = NeuralPolicy.from_checkpoint(path, device="cpu")
    if policy.checkpoint_hash != expected_hash:
        raise ValueError("Checkpoint changed while loading")
    return policy


def _evaluation_policy(name: str, seed: int) -> Any:
    if name.startswith("checkpoint:"):
        path = Path(name.removeprefix("checkpoint:")).resolve()
        cached = _load_neural(str(path), _checkpoint_hash(path))
        from .model import NeuralPolicy
        # Fresh policy wrapper per seat/match; the cached eval-mode network is
        # immutable and stateless during inference in this process.
        return NeuralPolicy(cached.model, device="cpu", checkpoint_hash=cached.checkpoint_hash,
                            source_manifest=cached.sources)
    return make_policy(name, seed)


def _evaluation_manifest(name: str, seed: int = 0) -> dict[str, Any]:
    if name.startswith("checkpoint:"):
        return _evaluation_policy(name, seed).manifest()
    return policy_manifest(name, seed)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _seed(root: int, *labels: Any) -> int:
    return int(_hash([root, *labels])[:16], 16)


def _latency(samples: list[float]) -> dict[str, Any]:
    return {
        "decisions": len(samples),
        "total_seconds": sum(samples),
        "p50_ms": float(np.percentile(samples, 50) * 1000) if samples else None,
        "p95_ms": float(np.percentile(samples, 95) * 1000) if samples else None,
        "max_ms": max(samples) * 1000 if samples else None,
    }


def _wall_fingerprint(env: Any) -> str:
    """Audit a newly dealt hand, normalized to relative dealer order.

    The full remaining wall plus physical initial allocations and first draw
    identify the same ordinal shuffle across paired matches. Never give this
    commitment or its inputs to a policy. This is not a portable wall format.
    """
    hands = [list(hand) for hand in env.hands]
    hands[env.oya].remove(env.drawn_tile)
    return _hash({
        "wall": list(env.wall), "draw": env.drawn_tile,
        "hands": [sorted(hands[(env.oya + i) % 4]) for i in range(4)],
    })


def run_match(
    seed: int, policy_names: Sequence[str], *, policy_seeds: Sequence[int] | None = None,
    max_steps: int = 100_000, failure_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run one whole match; save enough auditor data to reproduce failures.

    A truncated or failed match has no terminal rank utility. Timings are kept
    separate from the deterministic digest. Fresh adapters/policies prevent
    seat or match memory leakage. Physical and point audits are always enabled.
    """
    _validate_seed(seed)
    if len(policy_names) != 4:
        raise ValueError("Exactly four policies are required")
    if type(max_steps) is not int or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    seeds = list(policy_seeds) if policy_seeds is not None else [_seed(seed, "policy", i) for i in range(4)]
    if len(seeds) != 4:
        raise ValueError("Exactly four policy seeds are required")
    for policy_seed in seeds:
        _validate_seed(policy_seed)
    setup_start = time.perf_counter()
    policies = [_evaluation_policy(name, policy_seed) for name, policy_seed in zip(policy_names, seeds)]
    manifests = [policy.manifest() for policy in policies]
    adapters = [PlayerAdapter(seat) for seat in range(4)]
    view_hashers = [PlayerViewHasher() for _ in range(4)]
    env, observations = create_env(seed)
    trace: list[dict[str, str]] = []
    fingerprints: list[str] = []
    view_digest = hashlib.sha256()
    latencies: list[list[float]] = [[] for _ in range(4)]
    timings = {name: 0.0 for name in ("encode", "policy", "resolve", "step", "audit", "view_hash", "wall_hash")}
    setup_seconds = time.perf_counter() - setup_start
    steps = decisions = 0
    failure = None
    choices = {}
    start = time.perf_counter()
    try:
        t = time.perf_counter()
        audit_physical_tiles(env)
        if score_total(env) != sum(INITIAL_SCORES):
            raise ValueError("Initial scores/deposits do not conserve points")
        timings["audit"] += time.perf_counter() - t
        while not env.done() and steps < max_steps:
            t = time.perf_counter()
            # Observation creation consumes native event cursors. Use exactly
            # the snapshots returned by reset/step; never refetch observations.
            views = {seat: adapters[seat].build(observations[seat]) for seat in sorted(observations)}
            timings["encode"] += time.perf_counter() - t
            if not views or not any(view.legal_actions for view in views.values()):
                raise ValueError("Unfinished match has no legal decision")
            hand_id = max(view.hand_id for view in views.values())
            if hand_id > len(fingerprints):
                if hand_id != len(fingerprints) + 1:
                    raise ValueError("A hand boundary was omitted")
                t = time.perf_counter()
                fingerprints.append(_wall_fingerprint(env))
                timings["wall_hash"] += time.perf_counter() - t
            t = time.perf_counter()
            for seat, view in views.items():
                view_digest.update(f"{seat}:{view_hashers[seat].digest(view)}\n".encode())
            timings["view_hash"] += time.perf_counter() - t
            choices = {}
            for seat, view in views.items():
                if not view.legal_actions:
                    continue
                t = time.perf_counter()
                choices[seat] = policies[seat].choose(view)
                duration = time.perf_counter() - t
                latencies[seat].append(duration)
                timings["policy"] += duration
            t = time.perf_counter()
            native = {seat: adapters[seat].resolve(action, observations[seat]) for seat, action in choices.items()}
            timings["resolve"] += time.perf_counter() - t
            trace.append({str(seat): action.token for seat, action in choices.items()})
            t = time.perf_counter()
            observations = env.step(native)
            timings["step"] += time.perf_counter() - t
            steps += 1
            decisions += len(choices)
            t = time.perf_counter()
            audit_physical_tiles(env)
            if score_total(env) != sum(INITIAL_SCORES):
                raise ValueError("Scores plus outstanding riichi deposits do not conserve points")
            timings["audit"] += time.perf_counter() - t
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error), "step": steps}
    elapsed = time.perf_counter() - start
    finalization_start = time.perf_counter()
    completed = bool(env.done()) and failure is None
    scores, ranks = list(env.scores()), list(env.ranks())
    log = env.mjai_log
    if completed and sorted(ranks) != [1, 2, 3, 4]:
        failure = {"type": "ValueError", "message": "Terminal ranks are not a permutation", "step": steps}
        completed = False
    record = {
        "schema": SCHEMA, "seed": seed, "policies": manifests,
        "rule_profile_hash": rule_profile_hash(), "max_steps": max_steps,
        "completed": completed,
        "stop_reason": "failure" if failure else ("completed" if completed else "max_steps"),
        "steps": steps, "decisions": decisions, "events": len(log),
        "hands": sum(event.get("type") == "start_kyoku" for event in log),
        "scores": scores, "ranks": ranks if completed else None,
        "utilities": [RANK_UTILITY[rank - 1] for rank in ranks] if completed else None,
        "riichi_sticks": env.riichi_sticks, "failure": failure,
        "conservation_ok": failure is None,
        "trajectory_hash": _hash(trace), "player_views_hash": view_digest.hexdigest(),
        "player_view_hash_schema": VIEW_HASH_SCHEMA,
        "auditor_log_hash": _hash(log), "ordinal_wall_fingerprints": fingerprints,
    }
    record["digest"] = _hash(record)
    # Per-wrapper workload counters are report metadata, like latency. Keep
    # them outside frozen policy identity and the deterministic game digest.
    record["policy_inference_by_seat"] = [
        policy.inference_stats() if callable(getattr(policy, "inference_stats", None)) else None
        for policy in policies
    ]
    record["policy_inference_timing_by_seat"] = [
        policy.inference_timing() if callable(getattr(policy, "inference_timing", None)) else None
        for policy in policies
    ]
    record["elapsed_seconds"] = elapsed
    record["timing_seconds"] = timings
    record["policy_latency_by_seat"] = [_latency(samples) for samples in latencies]
    record["setup_seconds"] = setup_seconds
    record["finalization_seconds"] = time.perf_counter() - finalization_start
    record["unallocated_loop_seconds"] = max(0.0, elapsed - sum(timings.values()))
    if not completed and failure_dir is not None:
        directory = Path(failure_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"match-{seed}-{record['digest'][:12]}.json"
        path.write_text(json.dumps({"record": record, "submitted_actions": trace,
                                    "last_policy_requests": {
                                        str(seat): asdict(action) if isinstance(action, ActionSpec)
                                        else {"invalid_type": type(action).__name__, "repr": repr(action)}
                                        for seat, action in choices.items()
                                    },
                                    "auditor_log": log}, indent=2, allow_nan=False) + "\n")
        record["failure_record"] = str(path)
    return record


def summarize_blocks(differences: Sequence[float], *, bootstrap_seed: int = 0) -> dict[str, Any]:
    """One independent observation is a four-seat paired seed-block mean.

    Bootstrap intervals are descriptive development diagnostics. The bounded
    Hoeffding interval is conservative, fixed-budget, and assumes independent
    sampled blocks; neither interval is valid for optional stopping/reuse.
    """
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(np.abs(values) > 3):
        raise ValueError("Expected finite block differences within [-3, 3]")
    n, mean = len(values), float(values.mean())
    half = 6 * math.sqrt(math.log(40) / (2 * n))
    result = {
        "independent_blocks": n, "mean_utility_difference": mean,
        "mean_rank_improvement": mean,
        "standard_error": float(values.std(ddof=1) / math.sqrt(n)) if n > 1 else None,
        "fixed_budget_hoeffding_95": [max(-3.0, mean - half), min(3.0, mean + half)],
        "bootstrap_percentile_95_exploratory": None,
        "bootstrap_replicates": 0,
        "warning": "Development comparison, not a promotion test; intervals assume fresh independent seed blocks and a fixed budget.",
    }
    if n > 1:
        rng = np.random.default_rng(bootstrap_seed)
        # Chunking bounds memory if a later experiment supplies many blocks.
        samples = [float(rng.choice(values, size=n, replace=True).mean()) for _ in range(5000)]
        result["bootstrap_percentile_95_exploratory"] = np.percentile(samples, [2.5, 97.5]).tolist()
        result["bootstrap_replicates"] = 5000
    return result


def source_manifest() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
    return {
        "dependencies": {name: version(name) for name in ("riichienv", "mahjong", "numpy", "torch")},
        "python_rng": "random.Random; exact Python version recorded separately",
        "git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "git_dirty": bool(status.stdout.strip()),
        "sources": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(Path(__file__).parent.glob("*.py"))},
    }


def _evaluate_block(job: tuple[Any, ...]) -> tuple[list[dict], dict]:
    candidate, baseline, opponents, block, seed, max_steps, failure_dir = job
    wall_seed = _seed(seed, "block", block)
    seat_differences, records = [], []
    block_valid = True
    for focal in range(4):
        policy_seeds = [_seed(seed, "policy", block, focal, seat) for seat in range(4)]
        arms = []
        for arm, focal_policy in (("candidate", candidate), ("baseline", baseline)):
            lineup = [""] * 4
            lineup[focal] = focal_policy
            for offset, opponent in enumerate(opponents, start=1):
                lineup[(focal + offset) % 4] = opponent
            result = run_match(wall_seed, lineup, policy_seeds=policy_seeds,
                               max_steps=max_steps, failure_dir=failure_dir)
            result.update({"block": block, "arm": arm, "focal_seat": focal})
            arms.append(result)
            records.append(result)
        a, b = arms
        common = min(len(a["ordinal_wall_fingerprints"]), len(b["ordinal_wall_fingerprints"]))
        common_walls = a["ordinal_wall_fingerprints"][:common] == b["ordinal_wall_fingerprints"][:common]
        valid = a["completed"] and b["completed"] and common_walls
        block_valid = block_valid and valid
        seat_differences.append(a["utilities"][focal] - b["utilities"][focal] if valid else None)
        a["paired_common_walls_match"] = b["paired_common_walls_match"] = common_walls
        a["paired_common_hands"] = b["paired_common_hands"] = common
    return records, {
        "block": block, "wall_seed": wall_seed, "valid": block_valid,
        "seat_utility_differences": seat_differences,
        "mean_utility_difference": statistics.mean(seat_differences) if block_valid else None,
    }


def _record_matches_frozen_panel(record: dict[str, Any], panel: Sequence[dict], root_seed: int) -> bool:
    """Check actual per-game provenance, including its derived policy seeds.

    Before/after file checks alone miss a checkpoint temporarily replaced and
    restored during a campaign. The manifest recorded when each policy was
    instantiated must match the initially frozen role assigned to that seat.
    """
    focal, block, arm = record["focal_seat"], record["block"], record["arm"]
    if arm not in ("candidate", "baseline") or len(record["policies"]) != 4:
        return False
    for seat, actual in enumerate(record["policies"]):
        index = (0 if arm == "candidate" else 1) if seat == focal else 1 + (seat - focal) % 4
        expected = dict(panel[index])
        if "seed" in expected:
            expected["seed"] = _seed(root_seed, "policy", block, focal, seat)
        if actual != expected:
            return False
    return True


def _evaluation_resources(artifact_roots: Sequence[Path]) -> dict[str, int]:
    """Sample this process and descendants; count distinct existing artifact files."""
    import psutil
    process = psutil.Process()
    rss = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    files = {path.resolve() for root in artifact_roots for path in root.rglob("*") if path.is_file()}
    return {"process_tree_rss_bytes": rss, "artifact_bytes": sum(path.stat().st_size for path in files)}


def _terminate_evaluation_pool(pool: ProcessPoolExecutor) -> None:
    """Stop pending work rather than draining after a resource failure."""
    if hasattr(pool, "terminate_workers"):
        pool.terminate_workers()
        return
    # Python 3.11--3.13 compatibility; 3.14 exposes the public method above.
    processes = tuple(getattr(pool, "_processes", {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=1)
        if process.is_alive():
            process.kill()
            process.join(timeout=1)


def evaluate_paired(
    candidate: str, baseline: str, *, opponents: Sequence[str] = ("random", "random", "random"),
    blocks: int = 16, seed: int = 20261016, max_steps: int = 100_000,
    failure_dir: str | Path | None = None,
    workers: int = 1, memory_limit_bytes: int | None = None,
    artifact_roots: Sequence[str | Path] | None = None, disk_limit_bytes: int | None = None,
) -> dict[str, Any]:
    """Replace one focal seat, pair both arms, and rotate over all four seats.

    Opponent order is clockwise relative to the focal seat. Both arms share
    environment and per-seat policy RNG seeds. A fresh engine follows the same
    ordinal shuffle stream; dealer/context and match length may still diverge.
    No failed or truncated game is silently dropped from inference. Optional
    resource limits are cooperative samples before/between whole blocks and
    while waiting for workers; incomplete campaigns never produce statistics.
    """
    import platform
    _validate_seed(seed)
    if type(blocks) is not int or blocks < 1:
        raise ValueError("blocks must be a positive integer")
    if len(opponents) != 3:
        raise ValueError("Exactly three opponents are required")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("workers must be an integer from 1 through 16")
    for limit, name in ((memory_limit_bytes, "memory_limit_bytes"), (disk_limit_bytes, "disk_limit_bytes")):
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError(f"{name} must be a positive integer or None")
    roots = tuple(Path(root).expanduser().resolve() for root in (artifact_roots or ()))
    if disk_limit_bytes is not None and not roots:
        raise ValueError("disk_limit_bytes requires artifact_roots")
    for name in (candidate, baseline, *opponents):
        _evaluation_manifest(name, 0)
    frozen_policies = json.loads(_json([_evaluation_manifest(name, 0)
                                       for name in (candidate, baseline, *opponents)]))
    frozen_source = source_manifest()
    start = time.perf_counter()
    records, block_records = [], []
    resource_failure = None
    peaks = {"process_tree_rss_bytes": 0, "artifact_bytes": 0}

    def resources_ok() -> bool:
        nonlocal resource_failure
        if memory_limit_bytes is None and disk_limit_bytes is None:
            return True
        try:
            sample = _evaluation_resources(roots)
        except Exception as error:
            resource_failure = {"type": "ResourceInspectionError", "message": str(error),
                                "completed_blocks": len(block_records)}
            return False
        for key, value in sample.items():
            peaks[key] = max(peaks[key], value)
        for key, limit, kind in (("process_tree_rss_bytes", memory_limit_bytes, "MemoryBudgetExceeded"),
                                 ("artifact_bytes", disk_limit_bytes, "DiskBudgetExceeded")):
            if limit is not None and sample[key] > limit:
                resource_failure = {"type": kind, "measured_bytes": sample[key], "limit_bytes": limit,
                                    "completed_blocks": len(block_records)}
                return False
        return True

    jobs = [(candidate, baseline, tuple(opponents), block, seed, max_steps, failure_dir) for block in range(blocks)]
    if workers == 1:
        for job in jobs:
            if not resources_ok():
                break
            matches, block_record = _evaluate_block(job)
            records.extend(matches)
            block_records.append(block_record)
            if not resources_ok():
                break
    else:
        pool = None
        if resources_ok():
            pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
        try:
            if pool is not None:
                source, pending = iter(jobs), deque()
                for _ in range(min(workers, blocks)):
                    pending.append(pool.submit(_evaluate_block, next(source)))
                while pending and resources_ok():
                    try:
                        # Consume in plan order, regardless of completion order.
                        timeout = 1.0 if memory_limit_bytes is not None or disk_limit_bytes is not None else None
                        matches, block_record = pending[0].result(timeout=timeout)
                    except TimeoutError:
                        continue
                    pending.popleft()
                    records.extend(matches)
                    block_records.append(block_record)
                    if not resources_ok():
                        break
                    job = next(source, None)
                    if job is not None:
                        pending.append(pool.submit(_evaluate_block, job))
        finally:
            if pool is not None:
                if resource_failure is not None:
                    _terminate_evaluation_pool(pool)
                else:
                    pool.shutdown(wait=True)
    elapsed = time.perf_counter() - start
    sources_unchanged = frozen_source == source_manifest()
    policies_unchanged = frozen_policies == [_evaluation_manifest(name, 0) for name in (candidate, baseline, *opponents)]
    for match in records:
        match["policies_match_frozen_panel"] = _record_matches_frozen_panel(match, frozen_policies, seed)
    match_policies_match_frozen_panel = all(match["policies_match_frozen_panel"] for match in records)
    for block in block_records:
        block["policies_match_frozen_panel"] = all(match["policies_match_frozen_panel"]
                                                   for match in records if match["block"] == block["block"])
        block["valid"] = block["valid"] and block["policies_match_frozen_panel"]
        if not block["valid"]:
            block["mean_utility_difference"] = None
    valid = (resource_failure is None and len(block_records) == blocks
             and sources_unchanged and policies_unchanged and match_policies_match_frozen_panel
             and all(block["valid"] for block in block_records))
    completed = sum(match["completed"] for match in records)
    decisions = sum(match["decisions"] for match in records)
    arm_summaries = {}
    for arm in ("candidate", "baseline"):
        matches = [match for match in records if match["arm"] == arm and match["completed"]]
        arm_summaries[arm] = {
            "completed_games": len(matches),
            "mean_focal_rank": statistics.mean(match["ranks"][match["focal_seat"]] for match in matches) if matches else None,
            "mean_focal_score": statistics.mean(match["scores"][match["focal_seat"]] for match in matches) if matches else None,
            "focal_placement_counts": [sum(match["ranks"][match["focal_seat"]] == rank for match in matches) for rank in range(1, 5)],
        }
    return {
        "schema": SCHEMA, "purpose": "fixed_budget_development_comparison",
        "configuration": {"candidate": frozen_policies[0], "baseline": frozen_policies[1],
                          "opponents_clockwise": frozen_policies[2:],
                          "blocks": blocks, "root_seed": seed, "max_steps": max_steps,
                          "workers": workers,
                          "utility_by_rank": list(RANK_UTILITY), "matches_per_block": 8},
        "rule_profile": resolved_rule_profile(), "rule_profile_hash": rule_profile_hash(),
        "source_manifest": frozen_source, "sources_unchanged": sources_unchanged,
        "policies_unchanged": policies_unchanged,
        "match_policies_match_frozen_panel": match_policies_match_frozen_panel,
        "python_version": platform.python_version(),
        "pairing": "Fresh pinned-engine instances; shared ordinal walls verified by auditor fingerprints; all four focal seats, equal policy RNG seeds per paired arm. Rule context and hand counts may diverge. Not cross-engine seed equivalence.",
        "valid": valid, "completed_games": completed,
        "requested_blocks": blocks, "completed_blocks": len(block_records), "resource_failure": resource_failure,
        "resource_limits": {"memory_limit_bytes": memory_limit_bytes, "disk_limit_bytes": disk_limit_bytes,
                            "artifact_roots": [str(root) for root in roots],
                            "sampled_peaks": peaks,
                            "scope": "Cooperative sampled limits, not OS hard caps. Serial checks occur at block boundaries; parallel checks also occur while waiting. Returned partial blocks are excluded. Report serialization happens after these samples."},
        "failed_games": sum(match["failure"] is not None for match in records),
        "truncated_games": sum(match["stop_reason"] == "max_steps" for match in records),
        "elapsed_seconds": elapsed, "decisions": decisions,
        "hanchans_per_hour": completed * 3600 / elapsed,
        "decisions_per_second": decisions / elapsed,
        "timing_seconds": {key: sum(match["timing_seconds"][key] for match in records)
                           for key in ("encode", "policy", "resolve", "step", "audit", "view_hash", "wall_hash")},
        "match_setup_seconds": sum(match["setup_seconds"] for match in records),
        "match_finalization_seconds": sum(match["finalization_seconds"] for match in records),
        "unallocated_match_loop_seconds": sum(match["unallocated_loop_seconds"] for match in records),
        "arm_summaries": arm_summaries,
        "statistics": summarize_blocks([block["mean_utility_difference"] for block in block_records], bootstrap_seed=seed) if valid else None,
        "blocks": block_records, "matches": records,
        "limitations": "Local synthetic opponents; no public rank equivalence or world-class comparison. Timers include CPU audit/hash overhead and sum worker work when parallel; not a training speed estimate. Per-process shanten/model caches and candidate-first order make latency cache/order dependent. Bootstrap/Hoeffding summaries do not constitute an automatic promotion test.",
    }
