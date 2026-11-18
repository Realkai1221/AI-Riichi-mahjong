"""Exact-state, single-factor continuations of the frozen Package A study.

Run only under the caller's numerical lease. This adds no actor features, data,
loss changes, public play, test-set access, or promotion decision.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .model import PolicyNetwork
from .package_a_stability import (
    ContextCapture, SafetyMonitor, activation_diagnostics, evaluate_probability,
    parameter_diagnostics, parameter_snapshot, stability_loss,
)
from .stage2b_training import _index_hash, evaluate_stage2b, stratified_indices
from .training import (
    _atomic_json, _atomic_torch, _cpu_state, _data_manifest, _restore_rng,
    _rng_state, _rss, _sources, _state_digest, _sync, collate_training,
)

SCHEMA = "package-a-lr-continuation-v1"


def state_digest(value: Any) -> str:
    """Stable recursive state identity, including all optimizer/RNG tensor bytes."""
    h = hashlib.sha256()

    def add(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous()
            h.update(json.dumps(["tensor", str(item.dtype), list(item.shape)]).encode())
            h.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            h.update(json.dumps(["array", str(item.dtype), list(item.shape)]).encode())
            h.update(item.tobytes())
        elif isinstance(item, Mapping):
            h.update(b"mapping{")
            for key in sorted(item, key=lambda x: (type(x).__name__, str(x))):
                add(key)
                add(item[key])
            h.update(b"}")
        elif isinstance(item, (list, tuple)):
            h.update(type(item).__name__.encode() + b"[")
            for child in item:
                add(child)
            h.update(b"]")
        else:
            h.update(json.dumps([type(item).__name__, item], allow_nan=False).encode())
    add(value)
    return h.hexdigest()


def optimizer_without_lr(state):
    result = deepcopy(state)
    for group in result["param_groups"]:
        group.pop("lr", None)
    return result


def safety_state(monitor: SafetyMonitor) -> dict:
    return {**deepcopy(monitor.report()), "streaks": dict(monitor.streaks)}


def restore_safety(report: Mapping, checkpoint: Mapping) -> SafetyMonitor:
    """Replay old logs to recover counters omitted from historical checkpoints.

    New continuations carry complete monitor state. Historical recovery requires
    every loss row and the original diagnostic schedule; it never resets warmup.
    """
    saved = checkpoint.get("safety_state")
    monitor = SafetyMonitor(report["safety"]["configuration"])
    if saved is not None:
        if {k: v for k, v in saved.items() if k != "streaks"} != report["safety"]:
            raise ValueError("Checkpoint safety state disagrees with source report")
        monitor.reference_samples = defaultdict(list, deepcopy(saved["warmup_reference_samples"]))
        monitor.references = deepcopy(saved["references"])
        monitor.streaks = defaultdict(int, saved["streaks"])
        monitor.events = deepcopy(saved["events"])
        monitor.tripped = bool(saved["tripped"])
        monitor.scale_alarm = bool(saved["scale_alarm"])
    else:
        steps = int(checkpoint["step"])
        history = report["history"]
        if [row["step"] for row in history] != list(range(1, steps + 1)):
            raise ValueError("Safety recovery requires complete ordered source history")
        diagnostics = report["diagnostics"]
        every = int(report["settings"]["diagnostic_every"])
        expected = sorted(set(range(every, steps + 1, every)) | {steps})
        if [row["step"] for row in diagnostics] != expected:
            raise ValueError("Safety recovery requires complete original diagnostics")
        by_step = {row["step"]: row for row in diagnostics}
        for row in history:
            monitor.update_loss(row["step"], row["policy_ce"], row["gradient_norm"])
            if row["step"] in by_step:
                monitor.update_scale(row["step"], by_step[row["step"]])
        if monitor.report() != report["safety"]:
            raise ValueError("Replayed safety state disagrees with historical report")
    if monitor.tripped or monitor.scale_alarm:
        raise ValueError("Cannot continue an already safety-stopped checkpoint")
    return monitor


def source_manifest():
    result = _sources()
    folder = Path(__file__).parent
    for name in ("package_a_stability.py", "package_a_continuation.py"):
        result["files"][name] = hashlib.sha256((folder / name).read_bytes()).hexdigest()
    return result


def _validate_source_manifest(saved, current):
    for key in ("python", "torch", "numpy"):
        if saved[key] != current[key]:
            raise ValueError(f"Source runtime changed: {key}")
    for name, sha in saved["files"].items():
        if current["files"].get(name) != sha:
            raise ValueError(f"Source implementation changed: {name}")


def continue_stability(dataset: Sequence, output_dir: str | Path, *, validation: Sequence,
                       checkpoint_path: str | Path, source_report: Mapping | str | Path,
                       lr: float, additional_steps=5000, device="mps", max_seconds=180,
                       memory_limit_bytes=32 * 1024**3, protocol_manifest=None,
                       progress_callback=None, artifact_guard=None, expected_start_step=None,
                       expected_checkpoint_sha256=None, fixed_indices=None,
                       diagnostic_indices=None, validation_per_stratum=5000,
                       diagnostic_per_stratum=32) -> dict[str, Any]:
    """Resume a terminal exactly, changing only optimizer learning rate.

    ``selected.pt`` means best scheduled *continuation* check, not deployment.
    All paths must be new. Caller owns cumulative time/storage and worker limits.
    """
    started = time.perf_counter()
    if not len(dataset) or not len(validation) or type(additional_steps) is not int or additional_steps < 1:
        raise ValueError("Nonempty data and positive additional_steps required")
    if not math.isfinite(lr) or lr <= 0 or max_seconds <= 0 or memory_limit_bytes <= 0:
        raise ValueError("Positive finite learning rate and resource ceilings required")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError("Continuation output must be empty; no silent retry")
    device = torch.device(device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    path = Path(checkpoint_path)
    checkpoint_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_checkpoint_sha256 is not None and checkpoint_sha != expected_checkpoint_sha256:
        raise ValueError("Starting checkpoint hash mismatch")
    if isinstance(source_report, (str, Path)):
        report_bytes = Path(source_report).read_bytes()
        source_report_sha = hashlib.sha256(report_bytes).hexdigest()
        parent = json.loads(report_bytes)
    else:
        parent = deepcopy(dict(source_report))
        source_report_sha = hashlib.sha256(json.dumps(parent, sort_keys=True, allow_nan=False).encode()).hexdigest()
    saved = torch.load(path, map_location="cpu", weights_only=True)
    required = {"optimizer_state_dict", "rng_state", "sampled_index_digest", "data_manifest", "validation_manifest"}
    if not required <= saved.keys():
        raise ValueError("Exact continuation requires terminal optimizer, RNG, and data state")
    step = start_step = int(saved["step"])
    if expected_start_step is not None and step != expected_start_step:
        raise ValueError("Unexpected starting step")
    if step != parent["completed_steps"] or not parent["complete"] or parent["stop_reason"] != "steps_completed":
        raise ValueError("Starting report must identify a complete terminal checkpoint")
    if saved["sampled_index_digest"] != parent["sampled_index_digest"]:
        raise ValueError("Checkpoint/report sample-stream identities differ")
    inherited = deepcopy(saved["package_a_settings"])
    if inherited != parent["settings"]:
        raise ValueError("Checkpoint settings disagree with source report")
    if str(device) != inherited["device"] or torch.get_num_threads() != inherited["cpu_threads"]:
        raise ValueError("Exact continuation requires unchanged device and CPU thread count")
    if inherited["epsilon"] != .01 or inherited["rank_weight"] != 0 or inherited["weight_decay"] != 1e-4 or inherited["gradient_clip"] != 1.:
        raise ValueError("Source is not the frozen smoothed rank-zero recipe")
    train_identity, val_identity = _data_manifest(dataset), _data_manifest(validation)
    if train_identity != saved["data_manifest"] or train_identity != parent["data_manifest"]:
        raise ValueError("Training data manifest changed")
    if val_identity != saved["validation_manifest"] or val_identity != parent["validation_manifest"]:
        raise ValueError("Validation data manifest changed")
    sources = source_manifest()
    if saved["source_manifest"] != parent["source_manifest"]:
        raise ValueError("Checkpoint/report source identities differ")
    _validate_source_manifest(saved["source_manifest"], sources)
    batch_size = int(inherited["batch_size"])
    validate_every, diagnostic_every = int(inherited["validate_every"]), int(inherited["diagnostic_every"])
    target_step = start_step + additional_steps
    fixed_indices = list(fixed_indices) if fixed_indices is not None else stratified_indices(
        validation, per_stratum=validation_per_stratum, seed=inherited["validation_seed"])
    diagnostic_indices = list(diagnostic_indices) if diagnostic_indices is not None else stratified_indices(
        validation, per_stratum=diagnostic_per_stratum, seed=inherited["validation_seed"] ^ 0xA201)
    for indices, key in ((fixed_indices, "validation_indices_sha256"), (diagnostic_indices, "diagnostic_indices_sha256")):
        if _index_hash(indices) != inherited[key]:
            raise ValueError(f"Frozen row identities changed: {key}")
    monitor = restore_safety(parent, saved)
    initial_safety = safety_state(monitor)
    model = PolicyNetwork(saved["model_config"]).to(device)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    initial_digest = _state_digest(_cpu_state(model))
    if initial_digest != _state_digest(saved["model_state_dict"]):
        raise ValueError("Starting model tensors changed during restore")
    optimizer = torch.optim.AdamW(model.parameters(), lr=inherited["lr"], weight_decay=inherited["weight_decay"])
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    initial_optimizer_digest = state_digest(optimizer.state_dict())
    if initial_optimizer_digest != state_digest(saved["optimizer_state_dict"]):
        raise ValueError("Optimizer tensors or parameter groups changed during restore")
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer_core_digest = state_digest(optimizer_without_lr(optimizer.state_dict()))
    if optimizer_core_digest != state_digest(optimizer_without_lr(saved["optimizer_state_dict"])):
        raise ValueError("An optimizer setting other than learning rate changed")
    sampler = np.random.default_rng()
    fixed_batch, fixed_labels = collate_training([validation[i] for i in diagnostic_indices], device)
    capture = ContextCapture(model)
    _restore_rng(saved["rng_state"], sampler, device)
    initial_rng_digest = state_digest(_rng_state(sampler, device))
    if initial_rng_digest != state_digest(saved["rng_state"]):
        capture.close()
        raise ValueError("RNG state changed during restore")
    sampled_digest = initial_sample_digest = saved["sampled_index_digest"]
    rank_initial = {k: v for k, v in saved["model_state_dict"].items() if k.startswith("rank_head.")}
    settings = {**inherited, "lr": lr, "steps": target_step, "protocol_manifest": protocol_manifest}
    history, diagnostics, validations, sampled_prefixes = [], [], [], []
    best_metrics = best_state = best_step = None
    stop_reason, failure = "steps_completed", None
    peak_memory, elapsed_training = 0, 0.
    terminal_validation = full_validation = probability = selected_full_validation = None

    def remaining():
        return max(0., max_seconds - (time.perf_counter() - started))

    def memory():
        nonlocal peak_memory
        allocated = _rss() + (torch.mps.driver_allocated_memory() if device.type == "mps" else
                              torch.cuda.memory_reserved(device) if device.type == "cuda" else 0)
        peak_memory = max(peak_memory, allocated)
        if allocated >= memory_limit_bytes:
            raise MemoryError("Sampled process plus accelerator memory ceiling reached")

    def checkpoint(destination, state, at_step, metrics):
        terminal = destination.name == "terminal.pt"
        _atomic_torch(destination, {"feature_schema": model.config["feature_schema"],
            "model_config": model.config, "model_state_dict": state, "step": at_step,
            "validation_metrics": metrics, "source_manifest": sources,
            "protocol_manifest": protocol_manifest, "initialization": saved.get("initialization"),
            "package_a_settings": settings, "continuation_source_checkpoint_sha256": checkpoint_sha,
            **({"optimizer_state_dict": optimizer.state_dict(), "rng_state": _rng_state(sampler, device),
                "sampled_index_digest": sampled_digest, "data_manifest": train_identity,
                "validation_manifest": val_identity, "safety_state": safety_state(monitor)} if terminal else {})})

    try:
        memory()
        if artifact_guard:
            artifact_guard()
        while step < target_step:
            if remaining() <= 20:
                stop_reason = "time_budget"
                break
            memory()
            tick = time.perf_counter()
            indices = np.asarray(dataset.sample_indices(sampler, batch_size)
                if callable(getattr(dataset, "sample_indices", None)) else sampler.integers(len(dataset), size=batch_size), dtype=np.int64)
            sampled_digest = hashlib.sha256(bytes.fromhex(sampled_digest) + indices.astype("<i8").tobytes()).hexdigest()
            batch, labels = collate_training([dataset[int(i)] for i in indices], device)
            sample_diag = (step + 1) % diagnostic_every == 0 or step + 1 == target_step
            before = parameter_snapshot(model) if sample_diag else None
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits, ranks = model(batch)
            context = capture.value
            objective, parts = stability_loss(logits, ranks, labels, batch["candidate_mask"], inherited["epsilon"])
            if not bool(torch.isfinite(objective)):
                raise FloatingPointError("Nonfinite training objective")
            objective.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), inherited["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            if not bool(torch.cat([p.detach().reshape(-1) for p in model.parameters()]).isfinite().all()):
                raise FloatingPointError("Nonfinite updated parameters")
            _sync(device)
            step += 1
            row = {"step": step, "loss": float(objective.detach().cpu()),
                   **{k: float(v.detach().cpu()) for k, v in parts.items()},
                   "gradient_norm": float(norm.detach().cpu())}
            if not all(math.isfinite(value) for value in row.values()):
                raise FloatingPointError("Nonfinite recorded loss/probe/gradient scalar")
            history.append(row)
            monitor.update_loss(step, row["policy_ce"], row["gradient_norm"])
            if sample_diag:
                diag = {"step": step, "sampled": activation_diagnostics(context, logits, ranks, batch, labels),
                        "parameters": parameter_diagnostics(model, before)}
                model.eval()
                with torch.inference_mode():
                    f_logits, f_ranks = model(fixed_batch)
                    diag["fixed"] = activation_diagnostics(capture.value, f_logits, f_ranks, fixed_batch, fixed_labels)
                monitor.update_scale(step, diag)
                diagnostics.append(diag)
                sampled_prefixes.append({"step": step, "sha256": sampled_digest})
            elapsed_training += time.perf_counter() - tick
            if step % validate_every == 0 or step == target_step:
                metric = evaluate_stage2b(model, validation, device=device, batch_size=batch_size,
                    indices=fixed_indices, max_seconds=max(0., remaining() - 20))
                validations.append({"step": step, **metric})
                if metric["complete"] and (best_metrics is None or tuple(metric["checkpoint_selection_key"]) > tuple(best_metrics["checkpoint_selection_key"])):
                    best_metrics, best_state, best_step = metric, _cpu_state(model), step
                if progress_callback:
                    progress_callback({"step": step, "additional_steps": step - start_step,
                        "elapsed_seconds": time.perf_counter() - started,
                        "equal_stratum_means": metric["equal_stratum_means"],
                        "scale_alarm": monitor.scale_alarm, "alarm_events": len(monitor.events)})
                _atomic_json(directory / "progress.json", {"step": step, "validation": metric,
                    "safety": monitor.report(), "elapsed_seconds": time.perf_counter() - started})
                if artifact_guard:
                    artifact_guard()
            if monitor.tripped:
                stop_reason = "safety_alarm"
                break
        if validations and validations[-1]["step"] == step:
            terminal_validation = validations[-1]
        elif remaining() > 20:
            terminal_validation = evaluate_stage2b(model, validation, device=device, batch_size=batch_size,
                indices=fixed_indices, max_seconds=max(0., remaining() - 20))
        current = _cpu_state(model)
        checkpoint(directory / "terminal.pt", current, step, terminal_validation)
        if best_state is not None:
            checkpoint(directory / "selected.pt", best_state, best_step, best_metrics)
        if remaining() > 15:
            full_validation = evaluate_stage2b(model, validation, device=device, batch_size=batch_size,
                max_seconds=max(0., remaining() - 15))
        if remaining() > 10:
            probability = evaluate_probability(model, validation, device=device, batch_size=batch_size,
                max_seconds=max(0., remaining() - 10))
        if best_state is not None and best_step != step and remaining() > 5:
            model.load_state_dict(best_state)
            try:
                selected_full_validation = evaluate_stage2b(model, validation, device=device, batch_size=batch_size,
                    max_seconds=max(0., remaining() - 5))
            finally:
                model.load_state_dict(current)
        elif best_step == step:
            selected_full_validation = full_validation
        memory()
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error), "step": step}
        stop_reason = "memory_budget" if isinstance(error, MemoryError) else "failure"
    finally:
        capture.close()
    current = _cpu_state(model)
    rank_frozen = all(torch.equal(v, current[k]) for k, v in rank_initial.items())
    actual = source_manifest()
    terminal_agreement = ((terminal_validation or {}).get("equal_stratum_means") or {}).get("discard_tie_agreement")
    expected_validation_steps = list(range((start_step // validate_every + 1) * validate_every,
                                          target_step + 1, validate_every))
    if not expected_validation_steps or expected_validation_steps[-1] != target_step:
        expected_validation_steps.append(target_step)
    schedule_complete = ([row["step"] for row in validations] == expected_validation_steps
                         and all(row["complete"] for row in validations))
    complete = bool(step == target_step and stop_reason == "steps_completed" and not failure
        and terminal_validation and terminal_validation["complete"] and full_validation and full_validation["complete"]
        and probability and probability["complete"] and rank_frozen and sources == actual and schedule_complete)
    durable = bool(complete and not monitor.tripped and not monitor.scale_alarm
                   and terminal_agreement is not None and terminal_agreement >= .999)
    report = {"schema": SCHEMA, "starting_step": start_step, "completed_steps": step,
        "additional_steps": step - start_step, "requested_additional_steps": additional_steps,
        "requested_steps": target_step, "stop_reason": stop_reason, "failure": failure,
        "complete": complete, "durability_passed": durable, "validation_schedule_complete": schedule_complete,
        "terminal_fixed_discard_agreement": terminal_agreement,
        "settings": settings, "initial_state_sha256": initial_digest,
        "starting_checkpoint_sha256": checkpoint_sha, "source_report_sha256": source_report_sha,
        "initial_optimizer_state_sha256": initial_optimizer_digest,
        "initial_optimizer_without_lr_sha256": optimizer_core_digest,
        "initial_rng_state_sha256": initial_rng_digest,
        "initial_safety_state": initial_safety, "initial_safety_state_sha256": state_digest(initial_safety),
        "initial_sampled_index_digest": initial_sample_digest,
        "sampled_index_digest": sampled_digest, "sampled_prefixes": sampled_prefixes,
        "data_manifest": train_identity, "validation_manifest": val_identity,
        "source_manifest": sources, "sources_unchanged": sources == actual,
        "frozen_rank_head_unchanged": rank_frozen, "history": history, "diagnostics": diagnostics,
        "validations": validations, "safety": monitor.report(), "safety_state": safety_state(monitor),
        "best_step": best_step, "selected_fixed_validation": best_metrics,
        "selected_full_validation": selected_full_validation,
        "terminal_fixed_validation": terminal_validation, "terminal_full_validation": full_validation,
        "terminal_probability_metrics": probability, "elapsed_training_seconds": elapsed_training,
        "elapsed_total_seconds": time.perf_counter() - started,
        "sampled_peak_conservative_memory_bytes": peak_memory,
        "terminal_checkpoint": str(directory / "terminal.pt"),
        "selected_checkpoint": str(directory / "selected.pt") if (directory / "selected.pt").exists() else None,
        "selection_scope": "Best complete scheduled continuation validation only; source terminal is not a selection candidate. No policy promotion.",
        "scope": "Paired single-factor learning-rate continuation; no test set, new matches, playing-strength conclusion, or deployment."}
    if not (directory / "terminal.pt").exists():
        checkpoint(directory / "terminal.pt", current, step, terminal_validation)
    _atomic_json(directory / "report.json", report)
    if artifact_guard:
        artifact_guard()
    return report
