#!/usr/bin/env python3
"""Measured Stage 3F loader/value-computation pilot, with zero optimizer steps."""
from __future__ import annotations

import argparse
import copy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import psutil
import torch

from riichi_ai.model import PolicyNetwork, _state_hash
from riichi_ai.stage3f_stream import StreamDataset


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize(device)


def build_value_probe(checkpoint_path: str, expected_sha256: str, seed: int = 20261022031) -> PolicyNetwork:
    """Actual reference backbone; freshly reset scalar candidate head, no fit.

    All actor backbone parameters and the new scalar head receive gradients.
    The old rank head is frozen and unused by the chosen-action value objective.
    There is no optimizer, model checkpoint output, or deployed policy change.
    """
    import io
    payload = Path(checkpoint_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Reference checkpoint hash mismatch")
    saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = PolicyNetwork(saved["model_config"])
        model.load_state_dict(saved["model_state_dict"], strict=True)
        for module in model.action_score.modules():
            if isinstance(module, torch.nn.Linear):
                module.reset_parameters()
    for parameter in model.rank_head.parameters():
        parameter.requires_grad_(False)
    return model


def value_loss(model: PolicyNetwork, batch, device: str):
    actor = batch.torch_actor(device)
    chosen = torch.from_numpy(batch.targets["chosen_index"].astype(np.int64)).clone().to(device)
    utility = torch.from_numpy(batch.targets["final_utility"].astype(np.float32)).clone().to(device)
    values, _ = model(actor)
    selected = values.gather(1, chosen[:, None]).squeeze(1)
    mse = (selected - utility).square().mean()
    conservative = (torch.logsumexp(values, dim=-1) - selected).mean()
    return mse + conservative, values, actor


def numerical_snapshot(model: PolicyNetwork, batch, device: str) -> dict:
    model.zero_grad(set_to_none=True)
    loss, values, actor = value_loss(model, batch, device)
    loss.backward()
    synchronize(device)
    gradients = torch.cat([p.grad.detach().cpu().flatten() for p in model.parameters() if p.requires_grad and p.grad is not None])
    legal_values = values.detach().cpu()[torch.from_numpy(batch.actor["candidate_mask"])]
    return {"loss": float(loss.detach().cpu()), "values": legal_values.numpy().copy(),
            "gradients": gradients.numpy().copy(),
            "input_exact": all(np.array_equal(value.detach().cpu().numpy(), batch.actor[key]) for key, value in actor.items()),
            "finite": bool(torch.isfinite(legal_values).all() and torch.isfinite(gradients).all()),
            "gradient_l2": float(torch.linalg.vector_norm(gradients))}


def benchmark(args) -> dict:
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    deadline = started + args.max_seconds
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    dataset = StreamDataset(args.corpus, max_shard_bytes=args.max_shard_bytes, memory_limit_bytes=args.memory_limit_bytes)
    loader_started = time.perf_counter()
    rows = batches = 0
    identity = hashlib.sha256()
    first = None
    for batch in dataset.iter_batches(seed=args.seed, batch_rows=args.batch_rows):
        if first is None:
            first = batch
        identity.update(batch.match_id.encode())
        identity.update(batch.row_indices.tobytes())
        rows += batch.rows
        batches += 1
        peak_rss = max(peak_rss, process.memory_info().rss)
        if time.perf_counter() >= deadline:
            raise TimeoutError("Loader benchmark deadline reached")
        if rows >= args.max_rows:
            break
    loader_seconds = time.perf_counter() - loader_started
    if first is None:
        raise ValueError("No benchmark rows")
    initialized = build_value_probe(args.reference, args.reference_sha256)
    initial_hash = _state_hash(initialized.state_dict())
    devices = args.devices.split(",")
    if devices[0] != "cpu" or len(set(devices)) != len(devices):
        raise ValueError("Distinct devices must begin with the CPU reference")
    cpu_snapshot = None
    results = []
    for device in devices:
        available = device == "cpu" or (device == "mps" and torch.backends.mps.is_available()) or (device.startswith("cuda") and torch.cuda.is_available())
        if not available:
            results.append({"device": device, "available": False, "status": "unavailable"})
            continue
        if time.perf_counter() >= deadline:
            raise TimeoutError("Device benchmark deadline reached")
        model = copy.deepcopy(initialized).to(device)
        model.train()
        snapshot = numerical_snapshot(model, first, device)
        if device == "cpu":
            cpu_snapshot = snapshot
        reference = cpu_snapshot
        difference = snapshot["gradients"] - reference["gradients"]
        relative_l2 = float(np.linalg.norm(difference.astype(np.float64)) / max(np.linalg.norm(reference["gradients"].astype(np.float64)), 1e-12))
        numerical = {"loss": snapshot["loss"], "loss_abs_difference": abs(snapshot["loss"] - reference["loss"]),
            "legal_value_max_abs_difference": float(np.max(np.abs(snapshot["values"] - reference["values"]))),
            "gradient_max_abs_difference": float(np.max(np.abs(difference))),
            "gradient_relative_l2_difference": relative_l2,
            "gradient_l2": snapshot["gradient_l2"], "finite": snapshot["finite"], "input_exact": snapshot["input_exact"],
            "passed": bool(snapshot["finite"] and snapshot["input_exact"] and snapshot["gradient_l2"] > 0
                and relative_l2 <= .01 and np.allclose(snapshot["values"], reference["values"], rtol=.001, atol=.001)),
            "tolerances": {"gradient_relative_l2": .01, "values_rtol": .001, "values_atol": .001}}
        measured_rows = measured_batches = 0
        measured = StreamDataset(args.corpus, max_shard_bytes=args.max_shard_bytes, memory_limit_bytes=args.memory_limit_bytes)
        synchronize(device)
        tick = time.perf_counter()
        for batch in measured.iter_batches(seed=args.seed, batch_rows=args.batch_rows):
            model.zero_grad(set_to_none=True)
            loss, _, _ = value_loss(model, batch, device)
            loss.backward()
            synchronize(device)
            if not bool(torch.isfinite(loss.detach()).cpu()):
                raise ValueError("Nonfinite benchmark objective")
            measured_rows += batch.rows
            measured_batches += 1
            peak_rss = max(peak_rss, process.memory_info().rss)
            if time.perf_counter() >= deadline:
                raise TimeoutError("Value-computation benchmark deadline reached")
            if measured_batches >= args.model_batches:
                break
        seconds = time.perf_counter() - tick
        unchanged = _state_hash(model.state_dict()) == initial_hash
        if not unchanged:
            raise AssertionError("Read-only value benchmark mutated model weights")
        results.append({"device": device, "available": True, "status": "measured", "numerical": numerical,
            "rows": measured_rows, "batches": measured_batches, "seconds": seconds,
            "stream_plus_transfer_forward_backward_rows_per_second": measured_rows / seconds,
            "model_weights_unchanged": unchanged, "optimizer_steps": 0,
            "working_array_bytes": measured.peak_working_array_bytes,
            "device_allocated_bytes": int(torch.mps.current_allocated_memory()) if device == "mps" else None})
        model.zero_grad(set_to_none=True)
        del model
        if device == "mps":
            torch.mps.empty_cache()
    manifest = json.loads((Path(args.corpus) / "manifest.json").read_text())
    return {"schema": "riichi-ai-stage3f-stream-benchmark-v1", "complete": True,
        "manifest_sha256": dataset.manifest_sha256, "source_role": "development-only",
        "reference_checkpoint_sha256": args.reference_sha256, "initialized_probe_weights_sha256": initial_hash,
        "architecture": initialized.config,
        "parameters": sum(p.numel() for p in initialized.parameters()),
        "gradient_parameters": sum(p.numel() for p in initialized.parameters() if p.requires_grad),
        "architecture_scope": "Reference PolicyNetwork actor backbone plus freshly reset existing scalar candidate scorer; frozen, unused rank head; actor features only.",
        "objective_probe": "MSE to complete-match utility plus alpha=1 conservative chosen-action term; gradients only, no optimizer or training.",
        "limitations": "Throughput is an unoptimized representative computation probe, not trained-policy performance; includes bounded loader, transfer, forward and backward but excludes optimizer step, evaluation and corpus collection. Match-local shuffle and no asynchronous prefetch.",
        "versions": {"python": platform.python_version(), **{name: version(name) for name in ("numpy", "torch", "psutil", "riichienv", "mahjong")}},
        "threads": args.threads, "seed": args.seed, "batch_rows": args.batch_rows,
        "loader": {"rows": rows, "batches": batches, "seconds": loader_seconds, "rows_per_second": rows / loader_seconds,
            "numeric_compressed_bytes_read": dataset.bytes_read, "shards_opened": dataset.shards_opened,
            "stream_order_sha256": identity.hexdigest(), "working_array_bytes": dataset.peak_working_array_bytes,
            "all_corpus_uncompressed_bytes": sum(r["uncompressed_actor_bytes"] + r["uncompressed_target_bytes"] for r in manifest["matches"]),
            "memory_limit_bytes": args.memory_limit_bytes, "max_shard_bytes": args.max_shard_bytes},
        "devices": results, "numerical_checks_passed": all(r.get("numerical", {}).get("passed", True) for r in results),
        "peak_sampled_rss_bytes": peak_rss, "elapsed_seconds": time.perf_counter() - started,
        "memory_note": "RSS sampled at batch boundaries; array bound excludes model, tensor staging/backend allocator, Python overhead, and caller-retained batches. No OS memory hard limit."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--devices", default="cpu,mps")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=16384)
    parser.add_argument("--model-batches", type=int, default=16)
    parser.add_argument("--batch-rows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261022032)
    parser.add_argument("--max-shard-bytes", type=int, default=64 << 20)
    parser.add_argument("--memory-limit-bytes", type=int, default=256 << 20)
    parser.add_argument("--max-seconds", type=float, default=120)
    args = parser.parse_args()
    for key in ("threads", "max_rows", "model_batches", "batch_rows", "max_shard_bytes", "memory_limit_bytes", "max_seconds"):
        if getattr(args, key) <= 0:
            parser.error(f"{key} must be positive")
    path = Path(args.output)
    if path.exists():
        raise FileExistsError("Benchmark output already exists")
    report = benchmark(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"complete": report["complete"], "loader_rows_per_second": report["loader"]["rows_per_second"],
                      "numerical_checks_passed": report["numerical_checks_passed"], "elapsed_seconds": report["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
