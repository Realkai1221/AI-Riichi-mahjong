"""Synthetic compute probe; this is not a trained Mahjong policy."""

from __future__ import annotations

import math
import platform
import resource
import statistics
import time

import torch
from torch import nn

from .diagnostics import runtime_report


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(width, width, 3, padding=1),
        )

    def forward(self, x):
        return torch.relu(x + self.layers(x))


class ComputeProbe(nn.Module):
    """Provisional 34-position CNN to measure approximately 5M-parameter compute."""

    def __init__(self, width: int = 256, blocks: int = 12):
        super().__init__()
        self.input = nn.Conv1d(64, width, 3, padding=1)
        self.blocks = nn.Sequential(*(ResidualBlock(width) for _ in range(blocks)))
        self.policy = nn.Linear(width, 46)
        self.value = nn.Linear(width, 1)

    def forward(self, x):
        features = self.blocks(torch.relu(self.input(x))).mean(dim=-1)
        return self.policy(features), self.value(features).squeeze(-1)


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _timing_summary(durations: list[float], batch_size: int) -> dict:
    durations_sorted = sorted(durations)
    return {
        "iterations": len(durations),
        "batch_size": batch_size,
        "median_batch_ms": 1000 * statistics.median(durations),
        "p95_batch_ms": 1000 * durations_sorted[math.ceil(0.95 * len(durations)) - 1],
        "examples_per_second": batch_size * len(durations) / sum(durations),
        "timed_seconds": sum(durations),
    }


def _validate_training_state(model: nn.Module, loss: torch.Tensor, before=None) -> dict:
    """Audit optimizer results outside timed sections, including the final update."""
    parameters = list(model.parameters())
    gradients = [p.grad for p in parameters]
    if any(gradient is None for gradient in gradients):
        raise RuntimeError("Missing gradients in synthetic training probe.")
    finite_gradients = torch.stack([g.isfinite().all() for g in gradients]).all().item()
    finite_parameters = torch.stack([p.isfinite().all() for p in parameters]).all().item()
    gradient_norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in gradients])).item()
    if not (torch.isfinite(loss).item() and finite_gradients and finite_parameters
            and math.isfinite(gradient_norm)):
        raise RuntimeError("Non-finite synthetic training step.")
    result = {"finite_gradients": finite_gradients, "finite_parameters": finite_parameters,
              "gradient_norm": gradient_norm}
    if before is not None:
        if gradient_norm <= 0:
            raise RuntimeError("Initial synthetic training step has zero gradients.")
        change = torch.linalg.vector_norm(torch.stack([
            (p.detach() - old).norm() for p, old in zip(parameters, before, strict=True)
        ])).item()
        if not math.isfinite(change) or change <= 0:
            raise RuntimeError("Synthetic optimizer did not produce a finite parameter change.")
        result["parameter_change_norm"] = change
    return result


def benchmark_compute(
    device_name: str,
    batch_sizes: list[int],
    iterations: int = 10,
    warmup: int = 3,
    width: int = 256,
    blocks: int = 12,
) -> dict:
    started = time.perf_counter()
    if not batch_sizes or min(batch_sizes) < 1 or iterations < 1 or warmup < 0:
        raise ValueError("Positive batch sizes/iterations and nonnegative warmup are required.")
    if width < 1 or blocks < 1:
        raise ValueError("Width and blocks must be positive.")
    device = torch.device(device_name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable to this process.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable to this process.")
    torch.manual_seed(20261016)
    model = ComputeProbe(width, blocks).to(device)
    generator = torch.Generator().manual_seed(20261016)
    results = []
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 64, 34, generator=generator).to(device)
        labels = torch.randint(0, 46, (batch_size,), generator=generator).to(device)
        target_values = torch.randn(batch_size, generator=generator).to(device)
        model.eval()
        with torch.inference_mode():
            for _ in range(warmup):
                model(x)
            _synchronize(device)
            durations = []
            for _ in range(iterations):
                start = time.perf_counter()
                model(x)
                _synchronize(device)
                durations.append(time.perf_counter() - start)
        inference = _timing_summary(durations, batch_size)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        def train_step():
            optimizer.zero_grad(set_to_none=True)
            logits, values = model(x)
            loss = nn.functional.cross_entropy(logits, labels)
            loss = loss + nn.functional.mse_loss(values, target_values)
            loss.backward()
            optimizer.step()
            return loss.detach()

        before = [p.detach().clone() for p in model.parameters()]
        initial_validation = _validate_training_state(model, train_step(), before)
        del before
        for _ in range(warmup):
            train_step()
        _synchronize(device)
        durations = []
        last_loss = None
        for _ in range(iterations):
            start = time.perf_counter()
            last_loss = train_step()
            _synchronize(device)
            durations.append(time.perf_counter() - start)
        final_validation = _validate_training_state(model, last_loss)
        results.append({
            "batch_size": batch_size,
            "inference": inference,
            "training_step": _timing_summary(durations, batch_size),
            "final_synthetic_loss": last_loss.item(),
            "initial_update_validation": initial_validation,
            "final_update_validation": final_validation,
        })
        del optimizer
    return {
        "runtime": runtime_report(),
        "device": str(device),
        "dtype": "float32",
        "model": {"kind": "synthetic_residual_cnn", "width": width, "blocks": blocks,
                  "parameters": sum(p.numel() for p in model.parameters()),
                  "input_shape": [64, 34], "policy_outputs": 46},
        "warmup_iterations": warmup,
        "untimed_validation_steps_per_batch": 1,
        "elapsed_seconds_including_warmup_and_validation": time.perf_counter() - started,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if platform.system() == "Darwin" else 1024),
        "mps_current_allocated_bytes": torch.mps.current_allocated_memory() if device.type == "mps" else None,
        "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory() if device.type == "mps" else None,
        "memory_scope": "Process lifetime peak resident memory; MPS fields are endpoint allocations, not GPU peaks.",
        "results": results,
        "scope": "Short synthetic compute measurement. Excludes simulation, encoding, data transfer, PPO bookkeeping and evaluation. No playing-strength or sustained-throughput claim.",
    }

