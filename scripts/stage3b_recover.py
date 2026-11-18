"""Preserve the unchanged diagnostic's nonfinite findings as explicit JSON data.

The first formal attempt failed at strict JSON serialization. This recovery uses
the same frozen selection, checkpoint pair and numerical implementation. It does
not replace the failed attempt or silently turn nonfinite metrics into zeros.
Run only through the Stage 3B budgeted controller.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from package_a_stability import atomic_json, digest
from stage3b import settings, implementation, verify_inputs


def sanitize(value, path="$", anomalies=None):
    anomalies = [] if anomalies is None else anomalies
    if isinstance(value, float) and not math.isfinite(value):
        anomalies.append({"path": path, "value": repr(value)})
        return None
    if isinstance(value, dict):
        return {key: sanitize(child, f"{path}.{key}", anomalies) for key, child in value.items()}
    if isinstance(value, list):
        return [sanitize(child, f"{path}[{index}]", anomalies) for index, child in enumerate(value)]
    return value


def main():
    _, config, identity = settings(ROOT / "configs/stage3b.json")
    ledger = json.loads((Path(config["data_root"]) / "budget-ledger.json").read_text())
    active = [op for op in ledger["operations"] if op["status"] == "active"]
    if len(active) != 1 or active[0]["name"] != "check:serialization-recovery":
        raise RuntimeError("Recovery requires its matching active numerical lease")
    plan_path = ROOT / "reports/stage3b/plan.json"
    plan = json.loads(plan_path.read_text())
    if digest(plan_path) != "d7147bb10ab8ad13b82a7bdc7e8e824a092e81329082ea7ffc122f8a759c561d":
        raise ValueError("Original frozen selection plan changed")
    if plan["implementation"] != implementation() or plan["config_sha256"] != identity:
        raise ValueError("Recovery must use the original numerical implementation")
    verify_inputs(config)
    failed_path = ROOT / "reports/stage3b/diagnose-attempt.json"
    if json.loads(failed_path.read_text())["status"] != "failed":
        raise ValueError("Expected preserved failed first attempt")
    output = ROOT / "reports/stage3b/diagnostic-recovery.json"
    if output.exists():
        raise FileExistsError("Recovery evidence already exists")
    import torch
    from riichi_ai.stage3b import run_diagnostic
    torch.set_num_threads(1)
    remaining = active[0]["reserved_seconds"] - (time.time() - active[0]["started_unix"])
    result = run_diagnostic(plan["selection"], config["checkpoints"],
                            deadline=time.monotonic() + max(.1, remaining - 20), include_mps=True)
    anomalies = []
    result = sanitize(result, anomalies=anomalies)
    result.update(config_sha256=identity, plan_sha256=digest(plan_path),
        implementation=plan["implementation"], constraints=config["constraints"],
        recovery=dict(kind="serialization-only-wrapper", nonfinite_metrics=anomalies,
            wrapper_sha256=digest(__file__), preserved_failed_attempt_sha256=digest(failed_path),
            selection_and_grid_unchanged=True,
            numerical_measurements_repeated=True,
            interpretation="Nulls with listed paths preserve nonfinite findings; they are not passing numerical results."))
    atomic_json(output, result)
    verify_inputs(config)
    print(json.dumps({"report": str(output), "nonfinite_metrics": len(anomalies),
                      "cpu_complete": result["cpu_complete"], "mps_status": result["mps"]["status"]}))


if __name__ == "__main__":
    main()
