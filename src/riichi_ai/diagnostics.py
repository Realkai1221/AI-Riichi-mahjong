"""Small, reproducible checks of the local tensor runtime."""

from __future__ import annotations

import importlib.metadata
import math
import platform
import sys
from datetime import datetime, timezone

import torch


def runtime_report() -> dict:
    """Report capabilities without collecting machine serial numbers or IDs."""
    versions = {}
    for package in ("riichienv", "torch", "numpy", "pytest"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "os": platform.platform(),
        "architecture": platform.machine(),
        "packages": versions,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "torch_cpu_threads": torch.get_num_threads(),
    }


def check_backend(device_name: str) -> dict:
    """Check float32 forward and gradient agreement against a CPU reference."""
    device = torch.device(device_name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable to this process.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable to this process.")
    generator = torch.Generator().manual_seed(20261016)
    x = torch.randn(16, 32, generator=generator)
    w = torch.randn(32, 8, generator=generator)
    target = torch.randn(16, 8, generator=generator)
    cpu_w = w.clone().requires_grad_(True)
    cpu_output = (x @ cpu_w).tanh()
    cpu_loss = (cpu_output - target).square().mean()
    cpu_loss.backward()
    device_w = w.to(device).requires_grad_(True)
    device_output = (x.to(device) @ device_w).tanh()
    device_loss = (device_output - target.to(device)).square().mean()
    device_loss.backward()
    output = device_output.detach().cpu()
    gradient = device_w.grad.detach().cpu()
    forward_ok = torch.allclose(output, cpu_output.detach(), atol=2e-5, rtol=2e-4)
    gradient_ok = torch.allclose(gradient, cpu_w.grad, atol=2e-5, rtol=2e-4)
    output_error = (output - cpu_output.detach()).abs().max().item()
    gradient_error = (gradient - cpu_w.grad).abs().max().item()
    finite = math.isfinite(output_error) and math.isfinite(gradient_error)
    return {
        "device": str(device),
        "forward_max_absolute_error": output_error if math.isfinite(output_error) else None,
        "gradient_max_absolute_error": gradient_error if math.isfinite(gradient_error) else None,
        "forward_matches_cpu": forward_ok,
        "gradient_matches_cpu": gradient_ok,
        "finite": finite,
        "passed": forward_ok and gradient_ok and finite,
        "scope": "Float32 dense/tanh smoke check; not full-model backend certification.",
    }

