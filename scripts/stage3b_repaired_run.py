"""Repeat the fixed diagnostic after the evidenced MPS transfer repair.

The original plan, failed attempt, recovered invalid MPS measurements and
transfer fixture remain immutable. CPU results must match the recovered values.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from package_a_stability import atomic_json, digest
from stage3b import settings, implementation, verify_inputs


def main():
    _, config, identity = settings(ROOT / "configs/stage3b.json")
    ledger = json.loads((Path(config["data_root"]) / "budget-ledger.json").read_text())
    active = [op for op in ledger["operations"] if op["status"] == "active"]
    if len(active) != 1 or active[0]["name"] != "check:repaired-diagnostic":
        raise RuntimeError("Repaired run requires its matching active lease")
    reports = ROOT / "reports/stage3b"
    plan_path, repair_path = reports / "plan.json", reports / "repair-plan.json"
    plan, repair = json.loads(plan_path.read_text()), json.loads(repair_path.read_text())
    if repair["original_plan_sha256"] != digest(plan_path):
        raise ValueError("Original frozen plan changed")
    if repair["implementation_before"] != plan["implementation"]:
        raise ValueError("Repair does not preserve the original implementation identity")
    current = implementation()
    if repair["implementation_after"] != current:
        raise ValueError("Repaired implementation changed after its declaration")
    changed = [name for name in current if current[name] != plan["implementation"].get(name)]
    if changed != ["src/riichi_ai/stage3b.py"] or set(current) != set(plan["implementation"]):
        raise ValueError("Only the declared diagnostic transfer repair is allowed")
    if repair["config_sha256"] != identity or repair["wrapper_sha256"] != digest(__file__):
        raise ValueError("Protocol or wrapper changed")
    for item in repair["evidence"]:
        if digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("Preserved failure/transfer evidence changed")
    verify_inputs(config)
    output = reports / "diagnostic.json"
    if output.exists():
        raise FileExistsError("Do not replace the repaired diagnostic")
    import torch
    from riichi_ai.stage3b import run_diagnostic
    torch.set_num_threads(1)
    remaining = active[0]["reserved_seconds"] - (time.time() - active[0]["started_unix"])
    result = run_diagnostic(plan["selection"], config["checkpoints"],
                            deadline=time.monotonic() + max(.1, remaining - 20), include_mps=True)
    recovered = json.loads((reports / "diagnostic-recovery.json").read_text())
    # A transfer-path repair must not silently change the CPU experiment.
    cpu_equal = all(left[key] == right[key]
                    for left, right in zip(result["checkpoints"], recovered["checkpoints"])
                    for key in ("cpu_rows", "cpu_gradients", "cpu_summary"))
    result.update(config_sha256=identity, plan_sha256=digest(plan_path),
        repair_plan_sha256=digest(repair_path), implementation=current,
        constraints=config["constraints"], cpu_results_exactly_match_recovery=cpu_equal)
    atomic_json(output, result)
    if not cpu_equal:
        raise RuntimeError("CPU measurements changed; evidence preserved for investigation")
    verify_inputs(config)
    print(json.dumps({"report": str(output), "sha256": digest(output),
                      "cpu_unchanged": cpu_equal, "mps_status": result["mps"]["status"]}))


if __name__ == "__main__":
    main()
