"""Focused dtype-transfer check under the Stage 3B numerical ledger.

Uses one frozen actor row, one checkpoint and no optimizer or simulator.
The original diagnostic implementation and frozen plan remain untouched.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

from package_a_stability import Ledger, atomic_json, digest, require_worker_lease
from stage3b import ROOT, implementation, settings


def clean(value, path="", nonfinite=None):
    nonfinite = [] if nonfinite is None else nonfinite
    if isinstance(value, float) and not math.isfinite(value):
        nonfinite.append({"path": path, "value": repr(value)})
        return None
    if isinstance(value, dict):
        return {key: clean(child, f"{path}.{key}", nonfinite) for key, child in value.items()}
    if isinstance(value, list):
        return [clean(child, f"{path}[{index}]", nonfinite) for index, child in enumerate(value)]
    return value


def main():
    _, config, identity = settings(ROOT / "configs/stage3b.json")
    ledger = Ledger(Path(config["data_root"]) / "budget-ledger.json", identity, 300)
    active = [item for item in ledger.snapshot()["operations"] if item["status"] == "active"]
    if len(active) != 1:
        raise RuntimeError("A single active Stage 3B lease is required")
    require_worker_lease(ledger, active[0]["token"], active[0]["reserved_seconds"],
                         "check:mps-transfer-probe")
    output = ROOT / "reports/stage3b/mps-transfer-probe.json"
    if output.exists():
        raise FileExistsError("Preserve the first transfer probe")
    source = implementation()
    plan_path = ROOT / "reports/stage3b/plan.json"
    plan = json.loads(plan_path.read_text())
    if plan["implementation"] != source:
        raise ValueError("Original diagnostic source changed")
    import torch
    from riichi_ai.model import _weights_hash, collate_encoded
    from riichi_ai.stage3b import _load_actor_rows, _load_model, distribution

    torch.set_num_threads(1)
    deadline = time.monotonic() + max(1.0, active[0]["reserved_seconds"] - 15)
    selection = plan["selection"]
    selected_id = selection["probe_ids"][0]
    row = next(item for item in selection["rows"] if item["id"] == selected_id)
    one = {**selection, "rows": [row]}
    encoded = _load_actor_rows(one, deadline)[selected_id]
    model, initial_hash = _load_model(config["checkpoints"][0], "mps")
    rng_before = digest_bytes(torch.get_rng_state().numpy().tobytes())
    mps_rng_before = digest_bytes(torch.mps.get_rng_state().numpy().tobytes())

    def stats(tensor):
        values = tensor.reshape(-1).tolist()
        finite = [value for value in values if math.isfinite(value)]
        return {
            "dtype": str(tensor.dtype), "device": str(tensor.device),
            "elements": tensor.numel(), "nonfinite_elements": len(values) - len(finite),
            "min": min(finite, default=None), "max": max(finite, default=None),
            "max_abs": max((abs(value) for value in finite), default=None),
            "stable_l2": math.hypot(*finite) if len(finite) == len(values) else None,
            "tensor_square_sum": float(tensor.square().sum()),
            "first_values": values[:8],
        }

    def transfer(tensor):
        raw = tensor.detach()
        # Preserve the direct route used by the original diagnostic.
        direct = raw.to("cpu", dtype=torch.float64)
        staged32 = raw.to("cpu")
        staged64 = staged32.to(torch.float64)
        torch.mps.synchronize()
        direct_after_sync = raw.to("cpu", dtype=torch.float64)
        left, right = direct.reshape(-1).tolist(), staged64.reshape(-1).tolist()
        differences = [abs(a - b) for a, b in zip(left, right)]
        return {
            "source_dtype": str(raw.dtype), "source_device": str(raw.device),
            "source_shape": list(raw.shape), "source_stride": list(raw.stride()),
            "source_contiguous": raw.is_contiguous(),
            "cpu_original_dtype": stats(staged32),
            "direct_cpu_float64": stats(direct),
            "staged_cpu_float64": stats(staged64),
            "direct_after_sync_cpu_float64": stats(direct_after_sync),
            "direct_equals_staged": left == right,
            "max_absolute_direct_staged_difference": max(differences, default=0.0),
        }

    known = torch.tensor([0.0, 1.0, -1.0, .125, 1e-20, 10000.0], device="mps", dtype=torch.float32)
    result = {
        "schema": "stage3b-mps-transfer-probe-v1", "config_sha256": identity,
        "plan_sha256": digest(plan_path), "script_sha256": digest(__file__),
        "checkpoint": config["checkpoints"][0], "row_id": row["id"],
        "target_index": row["target_index"], "target_kind": row["target_kind"],
        "known_tensor": transfer(known), "known_noncontiguous_tensor": transfer(known[::2]),
        "new_matches": 0, "optimizer_updates": 0, "checkpoint_writes": 0,
    }
    batch = collate_encoded([encoded], "mps")
    logits = model(batch)[0][0]
    result["logits"] = transfer(logits)
    dist = distribution(logits, batch["candidate_mask"][0])
    parameters = [(name, parameter) for name, parameter in model.named_parameters()
                  if parameter.requires_grad]
    gradients = torch.autograd.grad(dist["log_probabilities"][row["target_index"]],
                                    [logits, *[parameter for _, parameter in parameters]],
                                    allow_unused=True)
    result["logit_gradient"] = transfer(gradients[0])
    result["parameter_gradients"] = {}
    for (name, _), gradient in zip(parameters, gradients[1:]):
        if time.monotonic() >= deadline:
            raise TimeoutError("Transfer fixture deadline reached")
        result["parameter_gradients"][name] = None if gradient is None else transfer(gradient)
    entries = [entry for entry in result["parameter_gradients"].values() if entry is not None]
    result["parameter_transfer_summary"] = {
        "connected_parameters": len(entries),
        "direct_staged_different_parameters": sum(not entry["direct_equals_staged"] for entry in entries),
        "direct_float64_stable_l2": math.hypot(*[entry["direct_cpu_float64"]["stable_l2"]
                                                for entry in entries
                                                if entry["direct_cpu_float64"]["stable_l2"] is not None]),
        "staged_float64_stable_l2": math.hypot(*[entry["staged_cpu_float64"]["stable_l2"]
                                                for entry in entries
                                                if entry["staged_cpu_float64"]["stable_l2"] is not None]),
        "direct_float64_nonfinite_elements": sum(entry["direct_cpu_float64"]["nonfinite_elements"] for entry in entries),
        "staged_float64_nonfinite_elements": sum(entry["staged_cpu_float64"]["nonfinite_elements"] for entry in entries),
    }
    final_hash = _weights_hash(model)
    result.update(weights_before_sha256=initial_hash, weights_after_sha256=final_hash,
                  weights_unchanged=initial_hash == final_hash,
                  grad_fields_unchanged=all(parameter.grad is None for parameter in model.parameters()),
                  cpu_rng_unchanged=rng_before == digest_bytes(torch.get_rng_state().numpy().tobytes()),
                  mps_rng_unchanged=mps_rng_before == digest_bytes(torch.mps.get_rng_state().numpy().tobytes()),
                  frozen_implementation_unchanged=source == implementation())
    nonfinite = []
    safe = clean(result, nonfinite=nonfinite)
    safe["nonfinite_values"] = nonfinite
    atomic_json(output, safe)
    print(json.dumps({"output": str(output), "sha256": digest(output),
                      "nonfinite_values": len(nonfinite)}), flush=True)


def digest_bytes(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()


if __name__ == "__main__":
    main()
