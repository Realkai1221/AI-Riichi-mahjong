"""Bounded imitation experiments, descriptive metrics, and resume checks.

Policy loss uses optional semantic choices only; rank loss retains forced rows.
No dataset secrets or teacher preference scores enter the network's inputs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import io
import inspect
import json
import math
from pathlib import Path
import platform
import random
import resource
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .features import FEATURE_SCHEMA
from .model import PolicyNetwork, collate_encoded, model_config
from .semantics import ACTION_KINDS


TRAINING_SCHEMA = "imitation-training-v1"


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _rss() -> int:
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except ImportError:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value if platform.system() == "Darwin" else value * 1024


def _sources() -> dict[str, Any]:
    folder = Path(__file__).parent
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": str(np.__version__), "files": {
                name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                for name in ("training.py", "stage2b_training.py", "model.py", "features.py", "efficiency.py", "semantics.py", "stage2b_data.py")
                if (folder / name).exists()}}


def _data_manifest(dataset: Sequence, supplied: Any = None) -> Any:
    if supplied is not None:
        return supplied
    if callable(getattr(dataset, "dataset_identity", None)):
        return dataset.dataset_identity()
    result = {"rows": len(dataset), "match_ids": list(getattr(dataset, "match_ids", ())),
              "split": getattr(dataset, "split", "provided_records")}
    directory = getattr(dataset, "directory", None)
    if directory is not None:
        path = Path(directory) / "manifest.json"
        result.update(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    else:
        digest = hashlib.sha256()
        for record in dataset:
            for key, array in sorted(record["encoded"].items()):
                value = np.asarray(array)
                digest.update(str((key, value.shape, str(value.dtype))).encode())
                digest.update(value.tobytes())
            for key in ("policy_target", "best_mask", "chosen_index", "rank_target", "teacher_preference_keys"):
                if key not in record:
                    continue
                value = np.asarray(record[key])
                digest.update(str((key, value.shape, str(value.dtype))).encode())
                digest.update(value.tobytes())
            digest.update(json.dumps({key: record[key] for key in ("stratum", "match_id", "seat", "step")
                                      if key in record}, sort_keys=True).encode())
        result["records_sha256"] = digest.hexdigest()
    return result


def _atomic_json(path: Path, record: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _atomic_torch(path: Path, record: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(record, temporary)
    temporary.replace(path)


def _cpu_state(model: PolicyNetwork) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _rng_state(sampler: np.random.Generator, device: torch.device) -> dict[str, Any]:
    legacy = np.random.get_state()
    return {"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
            "torch_mps": torch.mps.get_rng_state().cpu() if device.type == "mps" else None,
            "torch_cuda": torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None,
            "numpy_legacy": [legacy[0], legacy[1].tolist(), legacy[2], legacy[3], legacy[4]],
            "sampler": sampler.bit_generator.state}


def _restore_rng(saved: dict, sampler: np.random.Generator, device: torch.device) -> None:
    random.setstate(saved["python"])
    torch.set_rng_state(saved["torch_cpu"].cpu())
    legacy = saved["numpy_legacy"]
    np.random.set_state((legacy[0], np.asarray(legacy[1], dtype=np.uint32), *legacy[2:]))
    sampler.bit_generator.state = saved["sampler"]
    if device.type == "mps" and saved["torch_mps"] is not None:
        torch.mps.set_rng_state(saved["torch_mps"].cpu())
    if device.type == "cuda" and saved.get("torch_cuda") is not None:
        torch.cuda.set_rng_state(saved["torch_cuda"].cpu(), device)


def collate_training(records: Sequence[Mapping[str, Any]], device: Any = "cpu", target_mode: str = "soft") -> tuple[dict, dict]:
    if target_mode not in ("soft", "one_hot"):
        raise ValueError("target_mode must be soft or one_hot")
    batch = collate_encoded([record["encoded"] for record in records], device)
    width = batch["candidate_mask"].shape[1]
    q = np.zeros((len(records), width), dtype=np.float32)
    best = np.zeros_like(q, dtype=np.bool_)
    chosen, ranks, optional = [], [], []
    for index, record in enumerate(records):
        count = len(record["encoded"]["action_tokens"])
        target = np.asarray(record["policy_target"], dtype=np.float32)
        tied = np.asarray(record["best_mask"], dtype=np.bool_)
        choice, rank = int(record["chosen_index"]), int(record["rank_target"])
        if (target.shape != (count,) or tied.shape != (count,) or not np.isfinite(target).all()
                or np.any(target < 0) or not np.isclose(target.sum(), 1, atol=1e-5)
                or not 0 <= choice < count or not tied[choice] or not 0 <= rank < 4
                or np.any(target[~tied] != 0)):
            raise ValueError("Invalid semantic target/rank record")
        if bool(record.get("optional", count > 1)) != (count > 1):
            raise ValueError("Optional flag does not match semantic candidate count")
        if target_mode == "soft":
            q[index, :count] = target
        else:
            q[index, choice] = 1
        best[index, :count] = tied
        chosen.append(choice)
        ranks.append(rank)
        optional.append(count > 1)
    labels = {"q": torch.as_tensor(q, device=device), "best": torch.as_tensor(best, device=device),
              "chosen": torch.tensor(chosen, dtype=torch.long, device=device),
              "rank": torch.tensor(ranks, dtype=torch.long, device=device),
              "optional": torch.tensor(optional, dtype=torch.bool, device=device)}
    return batch, labels


def losses(logits: torch.Tensor, rank_logits: torch.Tensor, labels: Mapping[str, torch.Tensor], rank_weight: float = 0.1) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logp = logits.log_softmax(dim=-1)
    per_ce = -(labels["q"] * logp).sum(dim=-1)
    entropy = -(labels["q"] * labels["q"].clamp_min(1e-30).log()).sum(dim=-1)
    mask = labels["optional"].to(per_ce.dtype)
    policy = (per_ce * mask).sum() / mask.sum().clamp_min(1)
    rank = F.cross_entropy(rank_logits, labels["rank"])
    # Leaving the rank graph out entirely also keeps AdamW from decaying its
    # otherwise zero-gradient head in the rank-weight-zero ablation.
    total = policy if rank_weight == 0 else policy + rank_weight * rank
    return total, {"policy_ce": policy, "rank_ce": rank,
                                             "policy_kl": ((per_ce - entropy) * mask).sum() / mask.sum().clamp_min(1)}


def evaluate_model(model: PolicyNetwork, dataset: Sequence, *, device: Any = "cpu", batch_size: int = 256,
                   indices: Sequence[int] | None = None, max_samples: int | None = None,
                   seed: int = 0, target_mode: str = "soft", max_seconds: float | None = None) -> dict[str, Any]:
    """Fixed sampled descriptive metrics; rows are not independent matches."""
    if not len(dataset) or batch_size < 1:
        raise ValueError("Nonempty data and positive batch size required")
    if indices is None:
        indices = (np.random.default_rng(seed).choice(len(dataset), size=min(len(dataset), max_samples), replace=False).tolist()
                   if max_samples is not None else list(range(len(dataset))))
    indices = list(indices)
    if not indices:
        raise ValueError("No evaluation indices")
    device = torch.device(device)
    model.eval()
    totals = {key: 0.0 for key in ("ce", "kl", "tie", "exact", "random_tie", "rank_nll", "rank_brier", "rank_correct")}
    rows = optional = illegal = 0
    per_kind: dict[str, dict[str, float]] = {}
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(indices), batch_size):
            if max_seconds is not None and time.perf_counter() - started >= max_seconds:
                break
            records = [dataset[index] for index in indices[offset:offset + batch_size]]
            batch, labels = collate_training(records, device, target_mode)
            logits, rank_logits = model(batch)
            if not bool(torch.isfinite(logits).all() and torch.isfinite(rank_logits).all()):
                raise FloatingPointError("Non-finite evaluation logits")
            logp = logits.log_softmax(dim=-1)
            ce = -(labels["q"] * logp).sum(dim=-1)
            entropy = -(labels["q"] * labels["q"].clamp_min(1e-30).log()).sum(dim=-1)
            predictions = logits.argmax(dim=-1)
            selected = predictions.unsqueeze(-1)
            tie = labels["best"].gather(1, selected).squeeze(1)
            exact = predictions == labels["chosen"]
            chance = labels["best"].sum(dim=1) / batch["candidate_mask"].sum(dim=1)
            legal = batch["candidate_mask"].gather(1, selected).squeeze(1)
            probabilities = rank_logits.softmax(dim=-1)
            rank_nll = F.cross_entropy(rank_logits, labels["rank"], reduction="none")
            brier = (probabilities - F.one_hot(labels["rank"], 4)).square().sum(dim=-1)
            rank_correct = probabilities.argmax(dim=-1) == labels["rank"]
            data = torch.stack((ce, ce - entropy, tie, exact, chance, rank_nll, brier, rank_correct), dim=1).cpu().numpy()
            opt = labels["optional"].cpu().numpy()
            illegal += int((~legal).sum().item())
            rows += len(records)
            optional += int(opt.sum())
            for column, key in enumerate(("ce", "kl", "tie", "exact", "random_tie", "rank_nll", "rank_brier", "rank_correct")):
                totals[key] += float(data[opt if column < 5 else slice(None), column].sum(dtype=np.float64))
            for index, record in enumerate(records):
                if not opt[index]:
                    continue
                kind_id = int(record["encoded"]["action_tokens"][record["chosen_index"], 0])
                kind = ACTION_KINDS[kind_id - 1]
                summary = per_kind.setdefault(kind, {"rows": 0, "tie": 0.0, "exact": 0.0, "kl": 0.0, "random_tie": 0.0})
                summary["rows"] += 1
                for name, column in (("tie", 2), ("exact", 3), ("kl", 1), ("random_tie", 4)):
                    summary[name] += float(data[index, column])
    _sync(device)
    return {"complete": rows == len(indices), "rows": rows, "optional_rows": optional,
            "requested_rows": len(indices), "target_mode": target_mode,
            "policy_ce": totals["ce"] / optional if optional else None,
            "policy_kl": totals["kl"] / optional if optional else None,
            "tie_agreement": totals["tie"] / optional if optional else None,
            "exact_agreement": totals["exact"] / optional if optional else None,
            "random_legal_tie_agreement": totals["random_tie"] / optional if optional else None,
            "rank_nll": totals["rank_nll"] / rows if rows else None,
            "rank_brier": totals["rank_brier"] / rows if rows else None,
            "rank_accuracy": totals["rank_correct"] / rows if rows else None,
            "uniform_rank_baseline": {"nll": math.log(4), "brier": 0.75, "expected_random_tie_accuracy": 0.25},
            "illegal_choices": illegal,
            "per_action_kind": {kind: {"optional_rows": int(values["rows"]),
                **{key: value / values["rows"] for key, value in values.items() if key != "rows"}}
                for kind, values in sorted(per_kind.items())},
            "elapsed_seconds": time.perf_counter() - started,
            "scope": "Decision-weighted descriptive metrics; complete-match evaluation supplies strength evidence."}


def train(dataset: Sequence, output_dir: str | Path, *, validation: Sequence | None = None,
          device: Any = "cpu", steps: int = 2500, batch_size: int = 256, lr: float = 3e-4,
          seed: int = 20261017, max_seconds: float = 900, memory_limit_bytes: int = 32 * 1024**3,
          target_mode: str = "soft", rank_weight: float = 0.1, config: Mapping | None = None,
          resume_from: str | Path | None = None, data_manifest: Any = None, source_manifest: Any = None,
          validate_every: int = 250, validation_max_samples: int = 10_000,
          checkpoint_every: int = 250, stop_policy_kl: float | None = None,
          stop_tie_agreement: float = 1.0,
          initial_state_dict: Mapping[str, torch.Tensor] | None = None,
          initialization_manifest: Any = None, validation_indices: Sequence[int] | None = None,
          evaluation_fn: Any = None, evaluation_protocol: str = "standard-policy-v1",
          selection_rule: str = "highest_fixed_validation_tie_then_lowest_kl",
          protocol_manifest: Any = None, progress_callback: Any = None) -> dict[str, Any]:
    """Train to an absolute total step count, with cooperative resource limits.

    max_seconds includes initialization, validation, synchronization, and saves.
    A step/evaluation minibatch/save already underway can finish after the cap;
    final checkpoint/report writes are retained and their actual time reported.
    """
    started = time.perf_counter()
    if (not len(dataset) or steps < 1 or batch_size < 1 or lr <= 0 or max_seconds <= 0
            or memory_limit_bytes < 1 or validate_every < 1 or checkpoint_every < 1
            or validation_max_samples < 1 or rank_weight < 0 or not all(math.isfinite(v) for v in (lr, max_seconds, rank_weight))
            or target_mode not in ("soft", "one_hot")):
        raise ValueError("Invalid data, training configuration, or resource budget")
    if evaluation_fn is not None and evaluation_protocol == "standard-policy-v1":
        raise ValueError("A custom evaluator requires an explicit versioned protocol")
    evaluate = evaluate_model if evaluation_fn is None else evaluation_fn
    evaluator_identity = {"module": evaluate.__module__, "qualified_name": evaluate.__qualname__,
                          "source_sha256": hashlib.sha256(inspect.getsource(evaluate).encode()).hexdigest()}
    device = torch.device(device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable to this process")
    directory = Path(output_dir).expanduser().resolve()
    if directory.exists() and any(directory.iterdir()) and resume_from is None:
        raise FileExistsError("Use an empty output directory or an explicit resume checkpoint")
    directory.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    sampler = np.random.default_rng(seed)
    sources = _sources() if source_manifest is None else source_manifest
    actual_sources = _sources()
    data_identity = _data_manifest(dataset, data_manifest)
    validation_identity = _data_manifest(validation) if validation is not None else None
    initialization = ({"state_sha256": _state_digest(initial_state_dict), "manifest": initialization_manifest}
                      if initial_state_dict is not None else None)
    settings = {"batch_size": batch_size, "lr": lr, "seed": seed, "target_mode": target_mode,
                "rank_weight": rank_weight, "dataset_rows": len(dataset),
                "sampler": getattr(dataset, "sampling_protocol", "numpy.PCG64.uniform_with_replacement"),
                "device": str(device), "cpu_threads": torch.get_num_threads(),
                "validate_every": validate_every, "validation_max_samples": validation_max_samples,
                "checkpoint_every": checkpoint_every, "stop_policy_kl": stop_policy_kl,
                "stop_tie_agreement": stop_tie_agreement, "weight_decay": 1e-4, "gradient_clip": 1.0,
                "initialization": initialization, "evaluation_protocol": evaluation_protocol,
                "selection_rule": selection_rule, "protocol_manifest": protocol_manifest,
                "evaluator_identity": evaluator_identity}
    checkpoint = None
    if resume_from is not None:
        checkpoint = torch.load(io.BytesIO(Path(resume_from).read_bytes()), map_location="cpu", weights_only=True)
        if checkpoint.get("training_schema") != TRAINING_SCHEMA or "optimizer_state_dict" not in checkpoint:
            raise ValueError("Not a resumable training checkpoint")
        if checkpoint["training_config"] != settings or checkpoint["data_manifest"] != data_identity:
            raise ValueError("Resume data/configuration differs from the checkpoint")
        if checkpoint["feature_schema"] != checkpoint["model_config"]["feature_schema"]:
            raise ValueError("Resume feature/model schema mismatch")
        if config is not None and model_config(config) != checkpoint["model_config"]:
            raise ValueError("Requested resume model configuration differs")
        if checkpoint.get("implementation_manifest") != actual_sources:
            raise ValueError("Resume implementation/version differs")
        if checkpoint["source_manifest"] != sources:
            raise ValueError("Resume source provenance differs")
        if checkpoint.get("validation_manifest") != validation_identity:
            raise ValueError("Resume validation data differs")
        config = checkpoint["model_config"]
    model = PolicyNetwork(config).to(device)
    if initial_state_dict is not None:
        model.load_state_dict(initial_state_dict, strict=True)
    feature_schema = model.config["feature_schema"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history: list[dict] = []
    validations: list[dict] = []
    step = 0
    best_metrics = best_state = None
    best_step = None
    sampled_index_digest = hashlib.sha256(b"uniform-sample-stream-v1").hexdigest()
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        step, history, validations = checkpoint["step"], checkpoint["history"], checkpoint["validations"]
        if steps < step:
            raise ValueError("Requested total steps precede the resume checkpoint")
        best_metrics, best_state, best_step = checkpoint["best_metrics"], checkpoint["best_model_state_dict"], checkpoint["best_step"]
        sampled_index_digest = checkpoint["sampled_index_digest"]
        _restore_rng(checkpoint["rng_state"], sampler, device)
    if validation_indices is None:
        validation_indices = (np.random.default_rng(seed ^ 0x5A17).choice(
            len(validation), size=min(len(validation), validation_max_samples), replace=False).tolist()
            if validation is not None and len(validation) else [])
    else:
        validation_indices = list(validation_indices)
        if (validation is None or any(type(index) is not int or not 0 <= index < len(validation)
                                      for index in validation_indices)
                or len(set(validation_indices)) != len(validation_indices)):
            raise ValueError("Explicit validation indices must be unique, valid integers")
    if validation is not None and not validation_indices:
        raise ValueError("Validation set is empty")
    if checkpoint is not None and checkpoint["validation_indices"] != validation_indices:
        raise ValueError("Resume validation sample differs")
    peak_rss = _rss()
    peak_driver = torch.mps.driver_allocated_memory() if device.type == "mps" else 0
    peak_conservative_memory = peak_rss + peak_driver
    elapsed_training = 0.0
    stop_reason = "steps_completed"
    failure = None

    def save() -> None:
        _sync(device)
        current = _cpu_state(model)
        record = {"training_schema": TRAINING_SCHEMA, "feature_schema": feature_schema,
                  "model_config": dict(model.config), "model_state_dict": current,
                  "optimizer_state_dict": optimizer.state_dict(), "rng_state": _rng_state(sampler, device),
                  "training_config": settings, "step": step, "history": history, "validations": validations,
                  "validation_indices": validation_indices, "source_manifest": sources, "data_manifest": data_identity,
                  "sampled_index_digest": sampled_index_digest,
                  "validation_manifest": validation_identity, "implementation_manifest": actual_sources,
                  "best_model_state_dict": best_state, "best_metrics": best_metrics, "best_step": best_step}
        _atomic_torch(directory / "checkpoint.pt", record)
        selected = best_state if best_state is not None else current
        _atomic_torch(directory / "bestcheckpoint.pt", {
            "feature_schema": feature_schema, "model_config": dict(model.config), "model_state_dict": selected,
            "step": best_step if best_step is not None else step, "validation_metrics": best_metrics,
            "source_manifest": sources, "data_manifest": data_identity,
            "selection": selection_rule if best_state is not None else "last_no_complete_validation",
            "validation_manifest": validation_identity, "implementation_manifest": actual_sources,
            "initialization": initialization, "protocol_manifest": protocol_manifest})

    try:
        while step < steps:
            if time.perf_counter() - started >= max_seconds:
                stop_reason = "time_budget"
                break
            current_rss = _rss()
            driver = torch.mps.driver_allocated_memory() if device.type == "mps" else 0
            peak_rss, peak_driver = max(peak_rss, current_rss), max(peak_driver, driver)
            peak_conservative_memory = max(peak_conservative_memory, current_rss + driver)
            if current_rss + driver > memory_limit_bytes:
                stop_reason = "memory_budget"
                break
            tick = time.perf_counter()
            selected = (dataset.sample_indices(sampler, batch_size) if callable(getattr(dataset, "sample_indices", None))
                        else sampler.integers(len(dataset), size=batch_size))
            selected = np.asarray(selected, dtype=np.int64)
            if selected.shape != (batch_size,) or np.any(selected < 0) or np.any(selected >= len(dataset)):
                raise ValueError("Sampler returned invalid row indices")
            sampled_index_digest = hashlib.sha256(bytes.fromhex(sampled_index_digest)
                                                   + selected.astype("<i8").tobytes()).hexdigest()
            records = [dataset[int(index)] for index in selected]
            batch, labels = collate_training(records, device, target_mode)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            policy, rank = model(batch)
            loss, parts = losses(policy, rank, labels, rank_weight)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if not bool(torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()]).isfinite().all()):
                raise FloatingPointError("Optimizer update produced non-finite parameters")
            _sync(device)
            step += 1
            elapsed_training += time.perf_counter() - tick
            history.append({"step": step, "loss": float(loss.detach().cpu()),
                            **{key: float(value.detach().cpu()) for key, value in parts.items()},
                            "gradient_norm": float(norm.detach().cpu())})
            if validation_indices and (step % validate_every == 0 or step == steps):
                remaining = max_seconds - (time.perf_counter() - started)
                if remaining > 0:
                    metrics = evaluate(model, validation, device=device, batch_size=batch_size,
                                       indices=validation_indices, target_mode="soft", max_seconds=remaining)
                    validations.append({"step": step, **metrics})
                    if progress_callback is not None:
                        progress_callback({"step": step, "requested_steps": steps,
                                           "elapsed_seconds": time.perf_counter() - started,
                                           "validation": metrics})
                    if metrics["complete"] and metrics["optional_rows"]:
                        key = tuple(metrics.get("checkpoint_selection_key", (metrics["tie_agreement"], -metrics["policy_kl"])))
                        old = (tuple(best_metrics.get("checkpoint_selection_key", (best_metrics["tie_agreement"], -best_metrics["policy_kl"])))
                               if best_metrics else None)
                        if not all(math.isfinite(value) for value in key):
                            raise ValueError("Nonfinite checkpoint selection metric")
                        if old is None or key > old:
                            best_metrics, best_state, best_step = metrics, _cpu_state(model), step
                        if (stop_policy_kl is not None and metrics["policy_kl"] <= stop_policy_kl
                                and metrics["tie_agreement"] >= stop_tie_agreement):
                            stop_reason = "tiny_fit_target"
                            break
            if step % checkpoint_every == 0:
                save()
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error), "step": step}
        stop_reason = "failure"
    save()
    current_rss = _rss()
    driver = torch.mps.driver_allocated_memory() if device.type == "mps" else 0
    peak_rss, peak_driver = max(peak_rss, current_rss), max(peak_driver, driver)
    peak_conservative_memory = max(peak_conservative_memory, current_rss + driver)
    if failure is None and current_rss + driver > memory_limit_bytes:
        stop_reason = "memory_budget"
    sources_unchanged = actual_sources == _sources()
    report = {"schema": TRAINING_SCHEMA, "completed_steps": step, "requested_steps": steps,
              "stop_reason": stop_reason, "failure": failure, "sources_unchanged": sources_unchanged,
              "training_config": settings, "model_config": model.config,
              "parameters": sum(parameter.numel() for parameter in model.parameters()),
              "data_manifest": data_identity, "source_manifest": sources,
              "validation_manifest": validation_identity, "implementation_manifest": actual_sources,
              "checkpoint": str(directory / "checkpoint.pt"), "best_checkpoint": str(directory / "bestcheckpoint.pt"),
              "best_step": best_step, "best_validation": best_metrics, "history": history, "validations": validations,
              "sampled_index_digest": sampled_index_digest, "initialization": initialization,
              "elapsed_training_seconds": elapsed_training, "elapsed_total_seconds": time.perf_counter() - started,
              "max_seconds": max_seconds, "memory_limit_bytes": memory_limit_bytes, "sampled_peak_rss_bytes": peak_rss,
              "sampled_peak_mps_driver_bytes": peak_driver,
              "sampled_peak_conservative_memory_bytes": peak_conservative_memory,
              "memory_scope": "Sampled RSS plus MPS driver allocation is a conservative sum that can double-count unified memory; not an OS hard cap.",
              "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory() if device.type == "mps" else None,
              "resume_from": str(resume_from) if resume_from is not None else None,
              "scope": "Imitation training and fixed-development model selection; no playing-strength claim."}
    _atomic_json(directory / "report.json", report)
    return report


def tiny_overfit(dataset: Sequence, output_dir: str | Path, *, device: Any = "cpu", sample_count: int = 48,
                 seed: int = 37, max_steps: int = 1500, max_seconds: float = 120,
                 config: Mapping | None = None) -> dict[str, Any]:
    """Fit inspected optional states, including singleton and multiple-best ties."""
    rng = np.random.default_rng(seed)
    if sample_count < 2:
        raise ValueError("Tiny fit needs at least two states")
    count = min(sample_count, len(dataset))
    required = {"tied": count // 2, "unique": count - count // 2}
    indices = rng.choice(len(dataset), size=min(len(dataset), 10000), replace=False)
    pools: dict[str, list[tuple[int, dict]]] = {"tied": [], "unique": []}
    for index in indices:
        record = dataset[int(index)]
        candidates = len(record["encoded"]["action_tokens"])
        tied = int(np.asarray(record["best_mask"]).sum())
        kind_id = int(record["encoded"]["action_tokens"][record["chosen_index"], 0])
        if candidates < 2 or kind_id != ACTION_KINDS.index("DISCARD") + 1 or tied == candidates:
            continue
        group = "unique" if tied == 1 else "tied"
        if len(pools[group]) < required[group]:
            pools[group].append((int(index), record))
        if all(len(pools[group]) == required[group] for group in required):
            break
    if not all(len(pools[group]) == required[group] for group in required):
        raise ValueError("Not enough inspected optional discard states with unique and nontrivial tied targets")
    selected = pools["unique"] + pools["tied"]
    chosen_indices = [index for index, _ in selected]
    records = [record for _, record in selected]
    tie_sizes = [int(np.asarray(record["best_mask"]).sum()) for record in records]
    if min(tie_sizes) != 1 or max(tie_sizes) < 2:
        raise ValueError("Inspected tiny set must contain singleton and multiple-best targets; choose another seed/set")
    report = train(records, output_dir, validation=records, device=device, steps=max_steps,
                   batch_size=min(64, len(records)), lr=1e-3, seed=seed, max_seconds=max_seconds,
                   config=config, validate_every=25, validation_max_samples=len(records),
                   checkpoint_every=250, stop_policy_kl=0.02, stop_tie_agreement=1.0,
                   data_manifest={"kind": "inspected_tiny_optional_set", "source_indices": chosen_indices,
                                  "source": _data_manifest(dataset), "tie_sizes": tie_sizes,
                                  "candidate_counts": [len(record["encoded"]["action_tokens"]) for record in records]})
    report["tiny_fit_passed"] = report["stop_reason"] == "tiny_fit_target" and report["failure"] is None
    _atomic_json(Path(output_dir) / "tiny_report.json", report)
    return report


def check_cpu_mps_parity(records: Sequence, *, config: Mapping | None = None, seed: int = 41,
                         atol: float = 5e-5, rtol: float = 5e-4) -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        return {"status": "unavailable", "passed": False, "reason": "MPS is unavailable to this process"}
    torch.manual_seed(seed)
    cpu = PolicyNetwork(config)
    mps = copy.deepcopy(cpu).to("mps")
    left, labels = collate_training(records, "cpu")
    right, mps_labels = collate_training(records, "mps")
    cpu_logits, cpu_ranks = cpu(left)
    mps_logits, mps_ranks = mps(right)
    cpu_loss, _ = losses(cpu_logits, cpu_ranks, labels)
    mps_loss, _ = losses(mps_logits, mps_ranks, mps_labels)
    cpu_loss.backward()
    mps_loss.backward()
    _sync(torch.device("mps"))
    pairs = [("policy", cpu_logits[left["candidate_mask"]], mps_logits.detach().cpu()[left["candidate_mask"]]),
             ("rank", cpu_ranks, mps_ranks.detach().cpu()), ("loss", cpu_loss, mps_loss.detach().cpu())]
    pairs.extend(("gradient:" + name, param.grad, dict(mps.named_parameters())[name].grad.detach().cpu())
                 for name, param in cpu.named_parameters())
    differences = {name: float((a.detach() - b.detach()).abs().max()) for name, a, b in pairs}
    passed = all(torch.allclose(a, b, atol=atol, rtol=rtol) for _, a, b in pairs)
    return {"status": "passed" if passed else "failed", "passed": passed, "atol": atol, "rtol": rtol,
            "maximum_absolute_differences": differences, "scope": "One model-specific forward/backward fixture, not a full training-equivalence proof."}


def check_resume(records: Sequence, output_dir: str | Path, *, device: Any = "cpu", steps: int = 8,
                 config: Mapping | None = None, seed: int = 53, max_seconds: float = 120,
                 atol: float | None = None, rtol: float | None = None) -> dict[str, Any]:
    """Compare uninterrupted and interrupted/resumed updates on fixed records."""
    device = torch.device(device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        return {"status": "unavailable", "passed": False, "reason": "MPS unavailable to this process"}
    if steps < 2:
        raise ValueError("Resume check needs at least two updates")
    atol = (0.0 if device.type == "cpu" else 5e-5) if atol is None else atol
    rtol = (0.0 if device.type == "cpu" else 5e-4) if rtol is None else rtol
    folder = Path(output_dir)
    started = time.perf_counter()
    common = dict(device=device, config=config, seed=seed, batch_size=min(32, len(records)),
                  validate_every=steps + 1, checkpoint_every=steps + 1)
    full = train(records, folder / "uninterrupted", steps=steps, max_seconds=max_seconds / 3, **common)
    part = train(records, folder / "interrupted", steps=steps // 2, max_seconds=max_seconds / 3, **common)
    resumed = train(records, folder / "resumed", steps=steps, max_seconds=max_seconds / 3,
                    resume_from=part["checkpoint"], **common)
    left = torch.load(full["checkpoint"], weights_only=True, map_location="cpu")
    right = torch.load(resumed["checkpoint"], weights_only=True, map_location="cpu")
    weight_difference = max(float((left["model_state_dict"][key] - right["model_state_dict"][key]).abs().max())
                            for key in left["model_state_dict"])
    weights_equal = all(torch.allclose(left["model_state_dict"][key], right["model_state_dict"][key], atol=atol, rtol=rtol)
                        for key in left["model_state_dict"])
    loss_left = np.asarray([row["loss"] for row in full["history"]])
    loss_right = np.asarray([row["loss"] for row in resumed["history"]])
    complete = all(report["failure"] is None for report in (full, part, resumed)) and len(loss_left) == len(loss_right) == steps
    losses_equal = complete and bool(np.allclose(loss_left, loss_right, atol=atol, rtol=rtol))
    result = {"status": "passed" if weights_equal and losses_equal else "failed", "passed": bool(weights_equal and losses_equal),
              "device": str(device), "steps": steps, "atol": atol, "rtol": rtol,
              "maximum_parameter_difference": weight_difference,
              "maximum_loss_difference": float(np.max(np.abs(loss_left - loss_right))) if complete else None,
              "loss_curves_equal": losses_equal, "parameters_equal": weights_equal,
              "elapsed_total_seconds": time.perf_counter() - started,
              "scope": "Fixed local fixture, same implementation/backend/thread configuration; not cross-platform bitwise equivalence."}
    _atomic_json(folder / "resume_check.json", result)
    return result
