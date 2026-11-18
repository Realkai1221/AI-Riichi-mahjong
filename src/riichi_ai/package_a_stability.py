"""Prospective rank-zero stability study without changing frozen Stage 2B code.

Actor inputs remain player-visible. Detached diagnostics may compare predictions
with target-only final outcomes; these never become actor inputs. Original
teacher labels remain immutable; smoothing is a separate training-only target.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .model import PolicyNetwork, matched_initialization
from .semantics import ACTION_KINDS
from .stage2b_training import evaluate_stage2b, stratified_indices, _index_hash
from .training import (_atomic_json, _atomic_torch, _cpu_state, _data_manifest,
                       _rss, _sources, _state_digest, _sync, _rng_state, collate_training, losses)

SCHEMA = "package-a2-stability-v1"
DEFAULT_SAFETY = {"warmup_steps": 1000, "scale_factor": 10., "scale_floor": 1.,
                  "raw_logit_absolute_floor": 100., "persistence": 3, "gradient_threshold": 50., "gradient_persistence": 3,
                  "policy_ce_threshold": 10., "policy_ce_persistence": 50}
CALL_KINDS = tuple(ACTION_KINDS.index(k) + 1 for k in ("CHI", "PON", "DAIMINKAN", "ANKAN", "KAKAN")
                   if k in ACTION_KINDS)
KAN_KINDS = tuple(ACTION_KINDS.index(k) + 1 for k in ("DAIMINKAN", "ANKAN", "KAKAN") if k in ACTION_KINDS)


def smoothed_targets(labels: Mapping[str, torch.Tensor], candidate_mask: torch.Tensor,
                     epsilon: float) -> torch.Tensor:
    """Smooth only over legal non-best semantic choices; never mutate q/best."""
    if not math.isfinite(epsilon) or not 0 <= epsilon < 1:
        raise ValueError("epsilon must be finite and in [0,1)")
    q = labels["q"]
    if epsilon == 0:
        return q
    nonbest = candidate_mask & ~labels["best"]
    count = nonbest.sum(-1, keepdim=True)
    active = (count > 0) & labels["optional"].unsqueeze(-1)
    smooth = (1 - epsilon) * q + epsilon * nonbest.to(q.dtype) / count.clamp_min(1)
    return torch.where(active, smooth, q)


def stability_loss(logits, ranks, labels, candidate_mask, epsilon):
    """Original rank-zero graph for controls; unsmoothed reported metrics always."""
    original, parts = losses(logits, ranks, labels, rank_weight=0)
    if epsilon == 0:
        return original, parts
    q_train = smoothed_targets(labels, candidate_mask, epsilon)
    per = -(q_train * logits.log_softmax(-1)).sum(-1)
    optional = labels["optional"].to(per.dtype)
    objective = (per * optional).sum() / optional.sum().clamp_min(1)
    return objective, parts


class ContextCapture:
    """A forward hook that retains only a detached view and consumes no RNG."""
    def __init__(self, model):
        self.value = None
        self.handle = model.context.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output):
        self.value = output.detach()

    def close(self):
        self.handle.remove()
        self.value = None


def _summary(values) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not array.size:
        return {"count": 0, "mean": None, "median": None, "p95": None, "p99": None, "maximum": None}
    if not np.isfinite(array).all():
        raise FloatingPointError("Nonfinite diagnostic value")
    return {"count": int(array.size), "mean": float(array.mean()), "median": float(np.median(array)),
            "p95": float(np.quantile(array, .95)), "p99": float(np.quantile(array, .99)),
            "maximum": float(array.max())}


def activation_diagnostics(context, logits, rank_logits, batch, labels) -> dict:
    """CPU float64 summaries avoid padded minima and squared float32 overflow."""
    c = context.detach().cpu().numpy().astype(np.float64)
    z = logits.detach().cpu().numpy().astype(np.float64)
    mask = batch["candidate_mask"].detach().cpu().numpy().astype(bool)
    best = labels["best"].detach().cpu().numpy().astype(bool)
    legal = z[mask]
    if not np.isfinite(c).all() or not np.isfinite(legal).all():
        raise FloatingPointError("Nonfinite context or legal logits")
    means = np.where(mask, z, 0).sum(-1) / mask.sum(-1)
    centered = (z - means[:, None])[mask]
    gaps = [float(row[b].max() - row[m & ~b].max())
            for row, m, b in zip(z, mask, best) if (m & ~b).any()]
    rank_ce = F.cross_entropy(rank_logits.detach(), labels["rank"], reduction="none").cpu().numpy()
    return {"context_l2": _summary(np.sqrt((c * c).sum(-1))),
            "context_rms": _summary(np.sqrt((c * c).mean(-1))),
            "context_abs": _summary(np.abs(c)), "legal_logit_abs": _summary(np.abs(legal)),
            "legal_logit_row_offset_abs": _summary(np.abs(means)),
            "centered_legal_logit_abs": _summary(np.abs(centered)),
            "best_minus_nonbest_max_gap": _summary(gaps), "frozen_rank_probe_ce": _summary(rank_ce),
            "padding_excluded": int((~mask).sum())}


def parameter_snapshot(model):
    return {name: value.detach().clone() for name, value in model.named_parameters()}


def parameter_diagnostics(model, before):
    totals = defaultdict(lambda: [0., 0., 0., 0.])
    for name, parameter in model.named_parameters():
        group = name.split(".")[0]
        value = parameter.detach().cpu().numpy().astype(np.float64)
        old = before[name].detach().cpu().numpy().astype(np.float64)
        diff = value - old
        if not np.isfinite(value).all() or not np.isfinite(diff).all():
            raise FloatingPointError("Nonfinite parameter/update diagnostic")
        totals[group][0] += float(np.square(value).sum())
        totals[group][1] += float(np.square(diff).sum())
        totals[group][2] = max(totals[group][2], float(np.abs(value).max()))
        totals[group][3] = max(totals[group][3], float(np.abs(diff).max()))
    return {name: {"parameter_l2": math.sqrt(x[0]), "update_l2": math.sqrt(x[1]),
                   "relative_update_l2": math.sqrt(x[1]) / max(math.sqrt(x[0]), 1e-30),
                   "parameter_abs_max": x[2], "update_abs_max": x[3]}
            for name, x in totals.items()}


class SafetyMonitor:
    """Prespecified exploratory operational alarms; not a causal diagnosis."""
    SCALE_METRICS = ("context_rms", "legal_logit_abs", "centered_legal_logit_abs")

    def __init__(self, config=None):
        self.config = {**DEFAULT_SAFETY, **dict(config or {})}
        if set(self.config) != set(DEFAULT_SAFETY):
            raise ValueError("Unknown safety option")
        if any(not math.isfinite(float(v)) or v <= 0 for v in self.config.values()):
            raise ValueError("Positive finite safety settings required")
        self.reference_samples = defaultdict(list)
        self.references = {}
        self.streaks = defaultdict(int)
        self.events = []
        self.tripped = False
        self.scale_alarm = False

    def _condition(self, step, key, value, threshold, persistence):
        exceeded = value > threshold
        self.streaks[key] = self.streaks[key] + 1 if exceeded else 0
        if exceeded:
            event = {"step": step, "metric": key, "value": value, "threshold": threshold,
                     "streak": self.streaks[key], "required_persistence": persistence,
                     "abort": self.streaks[key] >= persistence}
            self.events.append(event)
            if event["abort"]:
                self.tripped = True
                self.scale_alarm |= key.startswith("scale:")
        return self.tripped

    def update_loss(self, step, policy_ce, gradient_norm):
        if not math.isfinite(policy_ce) or not math.isfinite(gradient_norm):
            raise FloatingPointError("Nonfinite loss/gradient safety signal")
        self._condition(step, "gradient_norm", gradient_norm, self.config["gradient_threshold"],
                        self.config["gradient_persistence"])
        self._condition(step, "policy_ce", policy_ce, self.config["policy_ce_threshold"],
                        self.config["policy_ce_persistence"])
        return self.tripped

    def update_scale(self, step, diagnostics):
        # Fixed and sampled batch signals have separate warm-up references.
        for population in ("sampled", "fixed"):
            for metric in self.SCALE_METRICS:
                for statistic in ("mean", "maximum"):
                    key = f"scale:{population}:{metric}:{statistic}"
                    value = diagnostics[population][metric][statistic]
                    if not math.isfinite(value):
                        raise FloatingPointError("Nonfinite scale safety signal")
                    if step <= self.config["warmup_steps"]:
                        self.reference_samples[key].append(value)
                        continue
                    if key not in self.references:
                        samples = self.reference_samples[key]
                        if not samples:
                            raise RuntimeError("Scale alarm has no warm-up reference")
                        self.references[key] = max(self.config["scale_floor"], float(np.median(samples)))
                    threshold = self.config["scale_factor"] * self.references[key]
                    if metric == "legal_logit_abs":
                        threshold = max(self.config["raw_logit_absolute_floor"], threshold)
                    self._condition(step, key, value, threshold, self.config["persistence"])
        return self.tripped

    def report(self):
        return {"configuration": self.config, "warmup_reference_samples": dict(self.reference_samples),
                "references": self.references, "events": self.events, "tripped": self.tripped,
                "scale_alarm": self.scale_alarm,
                "interpretation": "Prospective exploratory safety thresholds; neither scale growth nor probe CE establishes a mechanism."}


def probability_rows(logits, batch, labels):
    """Per-row sampling statistics, retaining forced states as a separate group."""
    with torch.no_grad():
        mask = batch["candidate_mask"]
        p = logits.softmax(-1)
        logp = logits.log_softmax(-1)
        entropy = -(p * logp).sum(-1)
        count = mask.sum(-1)
        normalized = torch.where(count > 1, entropy / count.clamp_min(2).log(), torch.zeros_like(entropy))
        best_mass = (p * labels["best"]).sum(-1)
        kind = batch["action_tokens"][..., 0]
        call_mask = torch.zeros_like(mask)
        kan_mask = torch.zeros_like(mask)
        for value in CALL_KINDS:
            call_mask |= (kind == value) & mask
        for value in KAN_KINDS:
            kan_mask |= (kind == value) & mask
        values = torch.stack((entropy, normalized, best_mass, (p * (~labels["best"] & mask)).sum(-1),
                              (p * call_mask).sum(-1), (p * kan_mask).sum(-1)), -1).cpu().numpy()
        counts = count.cpu().tolist()
        calls = call_mask.any(-1).cpu().tolist()
        kans = kan_mask.any(-1).cpu().tolist()
    return values, counts, calls, kans


def evaluate_probability(model, dataset, *, device="cpu", batch_size=256, max_seconds=60):
    started = time.perf_counter()
    groups = defaultdict(lambda: {"rows": 0, "sums": np.zeros(6), "maxima": np.zeros(6),
                                  "minima": np.full(6, np.inf)})
    completed = 0
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(dataset), batch_size):
            if time.perf_counter() - started >= max_seconds:
                break
            records = [dataset[i] for i in range(offset, min(offset + batch_size, len(dataset)))]
            batch, labels = collate_training(records, device)
            logits, _ = model(batch)
            values, counts, calls, kans = probability_rows(logits, batch, labels)
            if not np.isfinite(values).all():
                raise FloatingPointError("Nonfinite probability metrics")
            for record, row, count, call, kan in zip(records, values, counts, calls, kans):
                kind = ACTION_KINDS[int(record["encoded"]["action_tokens"][record["chosen_index"], 0]) - 1]
                keys = ["all", "forced" if count == 1 else "optional", f"legal_count:{count}",
                        f"teacher_kind:{kind}", f"stratum:{record['stratum']}",
                        f"stratum:{record['stratum']}:{'forced' if count == 1 else 'optional'}",
                        f"call_available:{call}", f"kan_available:{kan}",
                        f"joint:{record['stratum']}:{kind}:legal={count}:call={call}:kan={kan}"]
                for key in keys:
                    target = groups[key]
                    target["rows"] += 1
                    target["sums"] += row
                    target["maxima"] = np.maximum(target["maxima"], row)
                    target["minima"] = np.minimum(target["minima"], row)
            completed += len(records)
    names = ("entropy_nats", "normalized_entropy", "teacher_best_mass", "nonbest_mass", "call_mass", "kan_mass")
    return {"complete": completed == len(dataset), "rows": completed, "requested_rows": len(dataset),
            "elapsed_seconds": time.perf_counter() - started,
            "groups": {key: {"rows": value["rows"], **{name: {
                "mean": float(value["sums"][i] / value["rows"]), "minimum": float(value["minima"][i]),
                "maximum": float(value["maxima"][i])} for i, name in enumerate(names)}} for key, value in groups.items()},
            "scope": "Actual untempered softmax on fixed validation; deterministic deployed argmax is separate. Forced normalized entropy reported as zero."}


def train_stability(dataset: Sequence, output_dir: str | Path, *, validation: Sequence,
                    seed: int, epsilon: float, steps=18500, batch_size=256, lr=3e-4,
                    device="mps", max_seconds=600, memory_limit_bytes=32 * 1024**3,
                    validate_every=500, validation_seed=2026101720,
                    validation_per_stratum=5000, diagnostic_every=100,
                    diagnostic_per_stratum=32, safety=None, raw_config=None,
                    protocol_manifest=None, progress_callback=None, artifact_guard=None) -> dict[str, Any]:
    """Four-attempt study primitive; caller leases cumulative time before loading.

    No convergence stop and no automatic resume. Each result separates selected
    from terminal weights. Safety stops are observations, never successful runs.
    """
    started = time.perf_counter()
    if not len(dataset) or not len(validation) or min(steps, batch_size, validate_every, diagnostic_every) < 1:
        raise ValueError("Nonempty data and positive training sizes required")
    if max_seconds <= 0 or not 0 <= epsilon < 1:
        raise ValueError("Invalid time allowance/epsilon")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError("Attempt output must be empty; no silent retry/resume")
    device = torch.device(device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    sampler = np.random.default_rng(seed)
    raw, initial, initialization = matched_initialization(seed, raw_config)
    del raw
    model = PolicyNetwork(initialization["assisted_config"]).to(device)
    model.load_state_dict(initial, strict=True)
    initial_digest = _state_digest(initial)
    rank_initial = {k: v for k, v in initial.items() if k.startswith("rank_head.")}
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    capture = ContextCapture(model)
    monitor = SafetyMonitor(safety)
    fixed_indices = stratified_indices(validation, per_stratum=validation_per_stratum, seed=validation_seed)
    diagnostic_indices = stratified_indices(validation, per_stratum=diagnostic_per_stratum, seed=validation_seed ^ 0xA201)
    fixed_batch, fixed_labels = collate_training([validation[i] for i in diagnostic_indices], device)
    sources = _sources()
    sources["files"]["package_a_stability.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    settings = {"seed": seed, "epsilon": epsilon, "steps": steps, "batch_size": batch_size, "lr": lr,
                "rank_weight": 0., "weight_decay": 1e-4, "gradient_clip": 1., "device": str(device),
                "cpu_threads": torch.get_num_threads(), "validate_every": validate_every,
                "diagnostic_every": diagnostic_every, "validation_seed": validation_seed,
                "validation_indices_sha256": _index_hash(fixed_indices),
                "diagnostic_indices_sha256": _index_hash(diagnostic_indices),
                "safety": monitor.config, "protocol_manifest": protocol_manifest}
    history, diagnostics, validations = [], [], []
    model.eval()
    with torch.inference_mode():
        initial_logits, initial_ranks = model(fixed_batch)
        initial_diagnostics = activation_diagnostics(capture.value, initial_logits, initial_ranks, fixed_batch, fixed_labels)
    best_metrics = best_state = best_step = None
    sampled_digest = hashlib.sha256(b"uniform-sample-stream-v1").hexdigest()
    sampled_prefixes = []
    step = 0
    stop_reason = "steps_completed"
    failure = None
    elapsed_training = 0.
    peak_memory = 0
    terminal_validation = full_validation = entropy = None
    selected_full_validation = None

    def memory():
        nonlocal peak_memory
        allocated = (_rss() + (torch.mps.driver_allocated_memory() if device.type == "mps" else
                               torch.cuda.memory_reserved(device) if device.type == "cuda" else 0))
        peak_memory = max(peak_memory, allocated)
        if allocated >= memory_limit_bytes:
            raise MemoryError("Sampled process plus accelerator memory ceiling reached")

    def remaining():
        return max(0., max_seconds - (time.perf_counter() - started))

    def checkpoint(path, state, at_step, metrics):
        _atomic_torch(path, {"feature_schema": model.config["feature_schema"], "model_config": model.config,
            "model_state_dict": state, "step": at_step, "validation_metrics": metrics,
            "source_manifest": sources, "protocol_manifest": protocol_manifest,
            "initialization": initialization, "package_a_settings": settings,
            **({"optimizer_state_dict": optimizer.state_dict(), "rng_state": _rng_state(sampler, device),
                "sampled_index_digest": sampled_digest, "data_manifest": _data_manifest(dataset),
                "validation_manifest": _data_manifest(validation)} if path.name == "terminal.pt" else {})})

    try:
        memory()
        while step < steps:
            if remaining() <= 20:
                stop_reason = "time_budget"
                break
            memory()
            tick = time.perf_counter()
            indices = np.asarray(dataset.sample_indices(sampler, batch_size)
                if callable(getattr(dataset, "sample_indices", None)) else sampler.integers(len(dataset), size=batch_size), dtype=np.int64)
            sampled_digest = hashlib.sha256(bytes.fromhex(sampled_digest) + indices.astype("<i8").tobytes()).hexdigest()
            batch, labels = collate_training([dataset[int(i)] for i in indices], device)
            sample_diag = (step + 1) % diagnostic_every == 0 or step + 1 == steps
            before = parameter_snapshot(model) if sample_diag else None
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits, ranks = model(batch)
            context = capture.value
            objective, parts = stability_loss(logits, ranks, labels, batch["candidate_mask"], epsilon)
            if not bool(torch.isfinite(objective)):
                raise FloatingPointError("Nonfinite training objective")
            objective.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
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
                # Diagnostics consume no random draws and never reuse their graph.
                monitor.update_scale(step, diag)
                diagnostics.append(diag)
                sampled_prefixes.append({"step": step, "sha256": sampled_digest})
            elapsed_training += time.perf_counter() - tick
            if step % validate_every == 0 or step == steps:
                metric = evaluate_stage2b(model, validation, device=device, batch_size=batch_size,
                                         indices=fixed_indices, max_seconds=max(0., remaining() - 20))
                validations.append({"step": step, **metric})
                if metric["complete"]:
                    if best_metrics is None or tuple(metric["checkpoint_selection_key"]) > tuple(best_metrics["checkpoint_selection_key"]):
                        best_metrics, best_state, best_step = metric, _cpu_state(model), step
                if progress_callback:
                    progress_callback({"step": step, "elapsed_seconds": time.perf_counter() - started,
                        "equal_stratum_means": metric["equal_stratum_means"], "scale_alarm": monitor.scale_alarm,
                        "alarm_events": len(monitor.events), "latest_gradient_norm": row["gradient_norm"]})
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
            entropy = evaluate_probability(model, validation, device=device, batch_size=batch_size,
                                           max_seconds=max(0., remaining() - 10))
        if best_state is not None and best_step != step and remaining() > 5:
            model.load_state_dict(best_state)
            selected_full_validation = evaluate_stage2b(model, validation, device=device, batch_size=batch_size,
                                                       max_seconds=max(0., remaining() - 5))
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
    actual = _sources()
    actual["files"]["package_a_stability.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    terminal_agreement = ((terminal_validation or {}).get("equal_stratum_means") or {}).get("discard_tie_agreement")
    complete = bool(step == steps and stop_reason == "steps_completed" and not failure
        and terminal_validation and terminal_validation["complete"] and full_validation and full_validation["complete"]
        and entropy and entropy["complete"] and rank_frozen and sources == actual)
    durable = bool(complete and not monitor.scale_alarm and not monitor.tripped
                   and terminal_agreement is not None and terminal_agreement >= .999)
    report = {"schema": SCHEMA, "completed_steps": step, "requested_steps": steps, "stop_reason": stop_reason,
        "failure": failure, "complete": complete, "durability_passed": durable,
        "terminal_fixed_discard_agreement": terminal_agreement,
        "settings": settings, "initialization": initialization, "initial_state_sha256": initial_digest,
        "sampled_index_digest": sampled_digest, "sampled_prefixes": sampled_prefixes,
        "data_manifest": _data_manifest(dataset), "validation_manifest": _data_manifest(validation),
        "source_manifest": sources, "sources_unchanged": sources == actual,
        "frozen_rank_head_unchanged": rank_frozen, "history": history, "diagnostics": diagnostics,
        "initial_fixed_diagnostics": initial_diagnostics,
        "validations": validations, "safety": monitor.report(), "best_step": best_step,
        "selected_fixed_validation": best_metrics, "selected_full_validation": selected_full_validation,
        "terminal_fixed_validation": terminal_validation, "terminal_full_validation": full_validation,
        "terminal_probability_metrics": entropy, "elapsed_training_seconds": elapsed_training,
        "elapsed_total_seconds": time.perf_counter() - started, "sampled_peak_conservative_memory_bytes": peak_memory,
        "terminal_checkpoint": str(directory / "terminal.pt"), "selected_checkpoint": str(directory / "selected.pt") if (directory / "selected.pt").exists() else None,
        "scope": "Prospective four-attempt stability study; no test-set tuning, playing-strength claim or policy deployment."}
    # Preserve even an aborted/nonfinite attempt's weights; files never replace a different attempt.
    if not (directory / "terminal.pt").exists():
        checkpoint(directory / "terminal.pt", current, step, terminal_validation)
    _atomic_json(directory / "report.json", report)
    if artifact_guard:
        artifact_guard()
    return report
