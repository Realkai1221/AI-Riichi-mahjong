"""Bounded Stage 2B experiments; each formal run uses a fresh process.

No command is dispatched on import. Run from the repository's pinned venv.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Campaign:
    def __init__(self, config):
        from riichi_ai.stage2b_training import BudgetLedger
        self.config_path = Path(config).resolve()
        self.config_sha256 = digest(self.config_path)
        self.config = json.loads(self.config_path.read_text())
        self.data = Path(self.config["data_root"])
        self.reports = PROJECT / self.config["reports_root"]
        self.reports.mkdir(parents=True, exist_ok=True)
        self.training = self.config["training"]
        self.memory = self.config["limits"]["memory_gib"] * 1024**3
        self.ledger = BudgetLedger(self.data / "budget-ledger.json",
                                   limit_seconds=self.config["limits"]["total_training_related_seconds"],
                                   identity=self.config_sha256)

    def save(self, name, value, *, exclusive=False):
        from riichi_ai.training import _atomic_json
        path = self.reports / name
        if exclusive and path.exists():
            raise FileExistsError(path)
        _atomic_json(path, value)
        return path

    def read(self, name):
        return json.loads((self.reports / name).read_text())

    def disk_check(self):
        total = sum(path.stat().st_size for folder in (self.data, self.reports)
                    for path in folder.rglob("*") if path.is_file())
        if total >= self.config["limits"]["new_artifact_bytes"]:
            raise RuntimeError("Stage 2B artifact limit reached; retain evidence and stop")
        return total

    def memory_check(self, device):
        import torch
        from riichi_ai.training import _rss
        total = _rss() + (torch.mps.driver_allocated_memory() if device == "mps" else 0)
        if total >= self.memory:
            raise RuntimeError("Stage 2B conservative process memory limit reached")
        return total

    def load(self, split, assisted, *, old=False, max_matches=None, allow_test=False):
        from riichi_ai.stage2b_data import load_efficiency_split
        source = Path(self.config["stage2a_dataset"]) if old else self.data / "dataset"
        if old and digest(source / "manifest.json") != self.config["stage2a_manifest_sha256"]:
            raise RuntimeError("Original Stage 2A source manifest differs from the frozen protocol")
        if not old and json.loads((source / "manifest.json").read_text())["config"] != self.config:
            raise RuntimeError("Focal corpus configuration differs from the frozen protocol")
        derived = self.data / ("derived-stage2a" if old else "derived-stage2b")
        return load_efficiency_split(source, derived, split, assisted=assisted,
                                     max_matches=max_matches, memory_limit_bytes=self.memory // 4,
                                     allow_test=allow_test)

    def remaining(self, lease, *, reserve=0):
        result = lease["allowed_seconds"] - (time.perf_counter() - lease["started_monotonic"]) - reserve
        if result <= 0:
            raise RuntimeError("Operation's cumulative time allowance exhausted")
        return result

    def checks(self, device, allowance=360):
        import torch
        from riichi_ai.model import PolicyNetwork, NeuralPolicy, efficiency_model_config
        from riichi_ai.stage2b_training import evaluate_stage2b, stratified_indices
        from riichi_ai.training import check_cpu_mps_parity, check_resume, tiny_overfit, train, _data_manifest
        if (self.reports / "learning-checks.json").exists():
            raise FileExistsError("Checks already recorded; failed attempts must not be silently replaced")
        torch.set_num_threads(1)
        report = {"schema": "stage2b-learning-checks-v1", "config_sha256": self.config_sha256,
                  "device": device, "passed": False, "probes": {}, "failure": None}
        try:
            with self.ledger.operation("learning-checks-and-representative-probes", allowance_seconds=allowance,
                                       reserve_seconds=120) as lease:
                tick = time.perf_counter()
                assisted = self.load("train", True, old=True, max_matches=self.training["training_matches"])
                validation = self.load("validation", True)
                load_seconds = time.perf_counter() - tick
                self.memory_check(device)
                records = [assisted[int(index)] for index in __import__("numpy").linspace(0, len(assisted)-1, 32, dtype=int)]
                configuration = efficiency_model_config()
                report["parity"] = check_cpu_mps_parity(records[:8], config=configuration)
                for backend in ("cpu", "mps"):
                    report[backend + "_resume"] = check_resume(records, self.data / "checks" / (backend + "-resume"),
                        config=configuration, device=backend, steps=6,
                        max_seconds=min(45, self.remaining(lease, reserve=60)))
                report["tiny_fit"] = tiny_overfit(assisted, self.data / "checks" / "tiny-v2", device=device,
                    sample_count=32, seed=self.config["seed_namespaces"]["learning_checks"], config=configuration,
                    max_steps=1500, max_seconds=min(120, self.remaining(lease, reserve=40)))
                report["training_data"] = _data_manifest(assisted)
                report["validation_data"] = _data_manifest(validation)
                if digest(self.config["historical_checkpoint"]) != self.config["historical_checkpoint_sha256"]:
                    raise RuntimeError("Historical diagnostic checkpoint hash changed")
                historical = NeuralPolicy.from_checkpoint(self.config["historical_checkpoint"], device=device)
                diagnostic = self.load("diagnostic", False)
                report["historical_student_diagnostic"] = evaluate_stage2b(historical.model, diagnostic,
                    device=device, require_both_strata=False, max_seconds=self.remaining(lease, reserve=30))
                report["historical_student_diagnostic"]["checkpoint_sha256"] = self.config["historical_checkpoint_sha256"]
                report["historical_student_diagnostic"]["comparison_scope"] = "Descriptive historical student-state diagnostic; not a matched architecture control."
                if not report["historical_student_diagnostic"]["complete"]:
                    raise RuntimeError("Historical diagnostic evaluation was truncated")
                del historical, diagnostic
                gc.collect()
                for name, use_features in (("v2", True), ("v1", False)):
                    if use_features:
                        training_data, validation_data, loading = assisted, validation, load_seconds
                    else:
                        tick = time.perf_counter()
                        training_data = self.load("train", False, old=True, max_matches=self.training["training_matches"])
                        validation_data = self.load("validation", False)
                        loading = time.perf_counter() - tick
                    self.memory_check(device)
                    configuration = efficiency_model_config() if use_features else None
                    probe = train(training_data, self.data / "checks" / (name + "-probe"), config=configuration,
                                  device=device, steps=64, batch_size=self.training["batch_size"],
                                  max_seconds=min(60, self.remaining(lease, reserve=10)),
                                  memory_limit_bytes=self.memory, seed=self.training["seeds"][0])
                    if probe["completed_steps"] != 64 or probe["failure"] or not probe["sources_unchanged"]:
                        raise RuntimeError("Representative training probe did not finish cleanly")
                    model = PolicyNetwork(configuration).to(device)
                    fixed = stratified_indices(validation_data,
                        per_stratum=self.training["validation_samples_per_stratum"],
                        seed=self.config["seed_namespaces"]["learning_checks"])
                    timings = {}
                    for label, source, indices, both in (("sampled_validation", validation_data, fixed, True),
                            ("full_validation", validation_data, None, True),
                            ("frozen_training_sample", training_data,
                             stratified_indices(training_data, per_stratum=10000,
                                 seed=self.config["seed_namespaces"]["learning_checks"], require_both=False), False)):
                        metrics = evaluate_stage2b(model, source, device=device, indices=indices,
                            require_both_strata=both, max_seconds=self.remaining(lease))
                        if not metrics["complete"]:
                            raise RuntimeError("Representative evaluation probe was truncated")
                        timings[label] = {key: metrics[key] for key in ("elapsed_seconds", "rows", "indices_sha256")}
                    report["probes"][name] = {"batch_size": self.training["batch_size"], "steps": 64,
                        "seconds_per_step": probe["elapsed_training_seconds"] / 64,
                        "initialization_and_save_seconds": probe["elapsed_total_seconds"] - probe["elapsed_training_seconds"],
                        "load_seconds": loading, **timings}
                    del model, training_data, validation_data
                    gc.collect()
                report["passed"] = bool(report["parity"]["passed"] and report["cpu_resume"]["passed"]
                    and report["mps_resume"]["passed"] and report["tiny_fit"]["tiny_fit_passed"])
                self.memory_check(device)
                report["artifact_bytes"] = self.disk_check()
        except BaseException as error:
            report["passed"] = False
            report["failure"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            report["budget"] = self.ledger.snapshot()
            self.save("learning-checks.json", report, exclusive=True)
        if not report["passed"]:
            raise RuntimeError("Numerical learning checks did not all pass")
        print(json.dumps({"phase": "learning-checks", "passed": True, "probes": report["probes"]}), flush=True)

    def plan(self):
        with self.ledger.operation("freeze-common-update-plan", allowance_seconds=10, reserve_seconds=120):
            return self._plan()

    def _plan(self):
        checks = self.read("learning-checks.json")
        if not checks["passed"] or checks.get("failure") is not None or checks["config_sha256"] != self.config_sha256:
            raise RuntimeError("Current configuration requires passed numerical checks")
        if any(self.reports.glob("run-*.json")):
            raise RuntimeError("Common update ceiling cannot change after a formal attempt")
        probes = checks["probes"]
        step = max(row["seconds_per_step"] for row in probes.values())
        val = max(row["sampled_validation"]["elapsed_seconds"] for row in probes.values())
        fixed = max(row["load_seconds"] + row["initialization_and_save_seconds"] * 2
                    + row["full_validation"]["elapsed_seconds"] + row["frozen_training_sample"]["elapsed_seconds"]
                    for row in probes.values())
        reserve, margin = 120., 1.20
        count = len(self.training["recipes_in_execution_order"]) * len(self.training["seeds"])
        available = self.ledger.snapshot()["remaining_seconds"] - reserve
        interval = self.training["validate_every"]
        steps = self.training["steps"]
        def expected(number):
            # One second per periodic checkpoint conservatively covers I/O and
            # increasing JSON histories; startup/full evaluations measured above.
            return margin * (number * step + math.ceil(number / interval) * val + fixed
                             + math.ceil(number / self.training["checkpoint_every"]))
        while steps >= interval and count * expected(steps) > available:
            steps -= interval
        if steps < interval:
            raise RuntimeError("Required matched comparisons cannot fit the remaining budget")
        report = {"schema": "stage2b-frozen-update-plan-v1", "config_sha256": self.config_sha256,
            "checks_sha256": digest(self.reports / "learning-checks.json"), "common_steps": steps,
            "required_runs": count, "estimated_seconds_per_run": expected(steps),
            "required_seconds_estimate": count * expected(steps), "reserve_seconds": reserve,
            "margin": margin, "seconds_per_step_worst": step, "sampled_validation_seconds_worst": val,
            "fixed_overhead_seconds_worst": fixed, "budget_at_freeze": self.ledger.snapshot(),
            "scope": "Predeclared common ceiling based on representative batch256 and actual fixed/full validation costs; estimates are not guarantees."}
        self.save("training-plan.json", report, exclusive=True)
        print(json.dumps(report), flush=True)

    def budgeted(self, name, command, allowance=None, reserve=120):
        if command and command[0] == "--":
            command = command[1:]
        if not command:
            raise ValueError("An explicit subprocess command is required")
        with self.ledger.operation(name, allowance_seconds=allowance, reserve_seconds=reserve) as lease:
            timeout = self.remaining(lease, reserve=5)
            process = subprocess.Popen(command, cwd=PROJECT, start_new_session=True)
            try:
                code = process.wait(timeout=timeout)
                if code:
                    raise subprocess.CalledProcessError(code, command)
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                raise
            self.disk_check()

    def run(self, name, seed, device):
        import torch
        from riichi_ai.model import NeuralPolicy
        from riichi_ai.stage2b_training import (train_stage2b, evaluate_stage2b, stratified_indices,
                                               EqualRowMixture, complete_validation_schedule)
        from riichi_ai.training import _data_manifest
        torch.set_num_threads(1)
        plan = self.read("training-plan.json")
        if plan["config_sha256"] != self.config_sha256:
            raise RuntimeError("Frozen training plan configuration changed")
        recipes = self.training["recipes_in_execution_order"]
        contingent = None
        if (self.reports / "contingent-plan.json").exists():
            contingent = self.read("contingent-plan.json")
            if contingent["config_sha256"] != self.config_sha256:
                raise RuntimeError("Contingent decision configuration differs")
            recipes = recipes + ([contingent["recipe"]] if contingent.get("eligible_to_run") else [])
        matches = [recipe for recipe in recipes if recipe["name"] == name]
        if len(matches) != 1 or seed not in self.training["seeds"]:
            raise ValueError("Recipe/seed must belong to the frozen protocol")
        recipe = matches[0]
        report_name = f"run-{name}-{seed}.json"
        if (self.reports / report_name).exists():
            raise FileExistsError("A formal attempt already exists; never silently replace it")
        attempts = list(self.reports.glob("run-*.json"))
        if len(attempts) >= self.config["limits"]["maximum_full_training_runs"]:
            raise RuntimeError("Maximum formal run count reached")
        completed = sum(json.loads(path.read_text()).get("eligible", False) for path in attempts)
        mandatory_remaining = max(1, plan["required_runs"] - completed)
        if contingent and name == contingent.get("recipe", {}).get("name"):
            mandatory_remaining = 2 - sum(json.loads(path.read_text()).get("recipe") == name for path in attempts)
        available = self.ledger.snapshot()["remaining_seconds"] - plan["reserve_seconds"]
        allowance = min(plan["estimated_seconds_per_run"] * 1.15, available / mandatory_remaining)
        output = self.data / "models" / f"{name}-{seed}"
        report = {"schema": "stage2b-formal-run-v1", "config_sha256": self.config_sha256,
                  "recipe": name, "seed": seed, "eligible": False, "failure": None,
                  "common_steps": plan["common_steps"], "device": device}
        def progress(event):
            metrics = event["validation"]
            print(json.dumps({"recipe": name, "seed": seed, "step": event["step"],
                "elapsed_seconds": event["elapsed_seconds"], "equal_stratum_means": metrics["equal_stratum_means"]}), flush=True)
        try:
            with self.ledger.operation(f"formal:{name}:{seed}", allowance_seconds=allowance,
                                       reserve_seconds=plan["reserve_seconds"]) as lease:
                self.disk_check()
                assisted = recipe["features"] == "v2"
                dataset = self.load("train", assisted, old=True, max_matches=self.training["training_matches"])
                validation = self.load("validation", assisted)
                if recipe.get("aggregation"):
                    from riichi_ai.stage2b_data import load_efficiency_split
                    aggregation = json.loads((self.data / "aggregation" / "manifest.json").read_text())
                    conditional = aggregation["config"]["conditional_collection"]
                    inherited = {key: value for key, value in aggregation["config"].items()
                                 if key not in ("conditional_collection", "data_groups")}
                    if (conditional["checkpoint_sha256"] != contingent["base_checkpoint_sha256"]
                            or digest(conditional["checkpoint"]) != contingent["base_checkpoint_sha256"]
                            or Path(conditional["base_dataset_directory"]).resolve() != (self.data / "dataset").resolve()
                            or conditional["base_manifest_sha256"] != digest(self.data / "dataset" / "manifest.json")
                            or inherited != {key: value for key, value in self.config.items() if key != "data_groups"}):
                        raise RuntimeError("Aggregation is not bound to the frozen selected base/corpus/protocol")
                    student = load_efficiency_split(self.data / "aggregation", self.data / "derived-aggregation", "train",
                        assisted=assisted, memory_limit_bytes=self.memory // 4)
                    dataset = EqualRowMixture(dataset, student)
                self.memory_check(device)
                overhead = max(15., plan["fixed_overhead_seconds_worst"] * 1.3)
                result = train_stage2b(dataset, output, validation=validation, assisted=assisted, seed=seed,
                    rank_weight=recipe["rank_weight"], device=device, steps=plan["common_steps"],
                    batch_size=self.training["batch_size"], lr=self.training["learning_rate"],
                    validation_seed=self.config["seed_namespaces"]["learning_checks"],
                    validation_per_stratum=self.training["validation_samples_per_stratum"],
                    validate_every=self.training["validate_every"], checkpoint_every=self.training["checkpoint_every"],
                    max_seconds=self.remaining(lease, reserve=overhead), memory_limit_bytes=self.memory,
                    protocol_manifest={"config_sha256": self.config_sha256,
                                       "training_plan_sha256": digest(self.reports / "training-plan.json"),
                                       "recipe": recipe}, progress_callback=progress)
                report["training"] = result
                if (result["completed_steps"] != plan["common_steps"] or result["stop_reason"] != "steps_completed"
                        or result["failure"] or not result["sources_unchanged"] or result["best_validation"] is None
                        or not complete_validation_schedule(result, plan["common_steps"], self.training["validate_every"])):
                    raise RuntimeError("Formal comparison is incomplete; do not promote it")
                checkpoint = Path(result["best_checkpoint"])
                policy = NeuralPolicy.from_checkpoint(checkpoint, device=device)
                report["checkpoint"] = str(checkpoint)
                report["checkpoint_sha256"] = digest(checkpoint)
                report["full_validation"] = evaluate_stage2b(policy.model, validation, device=device,
                    max_seconds=self.remaining(lease))
                frozen_train = stratified_indices(dataset, per_stratum=10000,
                    seed=self.config["seed_namespaces"]["learning_checks"], require_both=False)
                report["frozen_training_sample"] = evaluate_stage2b(policy.model, dataset, device=device,
                    indices=frozen_train, require_both_strata=False, max_seconds=self.remaining(lease))
                report["data_identity"] = _data_manifest(dataset)
                report["validation_identity"] = _data_manifest(validation)
                report["sampled_index_digest"] = result["sampled_index_digest"]
                report["eligible"] = bool(report["full_validation"]["complete"] and report["frozen_training_sample"]["complete"])
                if not report["eligible"]:
                    raise RuntimeError("Frozen-model evaluation exhausted time; comparison remains incomplete")
                self.memory_check(device)
                report["artifact_bytes"] = self.disk_check()
        except BaseException as error:
            report["eligible"] = False
            report["failure"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            report["budget"] = self.ledger.snapshot()
            self.save(report_name, report, exclusive=True)
        print(json.dumps({"phase": "run-complete", "recipe": name, "seed": seed,
            "eligible": report["eligible"], "full_validation": report["full_validation"]["equal_stratum_means"],
            "budget_remaining_seconds": report["budget"]["remaining_seconds"]}), flush=True)

    def campaign(self, device):
        # Each run charges its own loading/training/evaluation; the parent does
        # no model work and does not reserve a conflicting nested lease.
        for recipe in self.training["recipes_in_execution_order"]:
            for seed in self.training["seeds"]:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "--config", str(self.config_path),
                    "run", "--recipe", recipe["name"], "--seed", str(seed), "--device", device], check=True)

    def ranking(self):
        from riichi_ai.stage2b_training import select_recipes
        runs = [json.loads(path.read_text()) for path in sorted(self.reports.glob("run-*.json"))]
        plan = self.read("training-plan.json")
        for run in runs:
            if run["config_sha256"] != self.config_sha256 or run["common_steps"] != plan["common_steps"]:
                raise RuntimeError("Run configuration/common ceiling mismatch")
            if run.get("eligible") and digest(run["checkpoint"]) != run["checkpoint_sha256"]:
                raise RuntimeError("Selected checkpoint changed")
        order = [recipe["name"] for recipe in self.training["recipes_in_execution_order"]]
        for seed in self.training["seeds"]:
            paired = [run for run in runs if run["seed"] == seed and run["recipe"] in order and run.get("eligible")]
            if paired:
                shared = {run["training"]["initialization"]["manifest"]["common_weights_sha256"] for run in paired}
                streams = {run["sampled_index_digest"] for run in paired}
                indices = {run["data_identity"]["index_sha256"] for run in paired}
                if len(shared) != 1 or len(streams) != 1 or len(indices) != 1:
                    raise RuntimeError("Matched base runs differ in shared initialization, sample stream or row ordering")
        if (self.reports / "contingent-plan.json").exists():
            contingent = self.read("contingent-plan.json")
            if contingent.get("eligible_to_run"):
                order.append(contingent["recipe"]["name"])
        result = select_recipes(runs, required_seeds=self.training["seeds"],
            primary_seed=self.training["primary_seed"], recipe_order=order)
        return result

    def contingent(self):
        with self.ledger.operation("contingent-trigger-analysis", allowance_seconds=30, reserve_seconds=120):
            ranking = self.ranking()
            base_names = {recipe["name"] for recipe in self.training["recipes_in_execution_order"]}
            if base_names.intersection(ranking["incomplete_recipes"]):
                raise RuntimeError("All six base comparisons must finish before conditional allocation")
            candidates = [item for item in ranking["eligible_recipes"] if item["recipe"].startswith("features_")]
            base = candidates[0]
            runs = base["per_seed"]
            gap = sum(run["full_validation"]["strata"]["teacher"]["discard_tie_agreement"]
                      - run["full_validation"]["strata"]["student"]["discard_tie_agreement"] for run in runs) / 2
            val_loss = sum(run["full_validation"]["equal_stratum_means"]["same_shanten_lost_ukeire_rate"] for run in runs) / 2
            train_loss = sum(run["frozen_training_sample"]["same_shanten_lost_ukeire_rate"] for run in runs) / 2
            checks = self.read("learning-checks.json")
            trigger = "aggregation" if gap > .02 else ("remaining_hand" if val_loss > .02 and train_loss > .01 and checks["passed"] else None)
            plan = self.read("training-plan.json")
            remaining = self.ledger.snapshot()["remaining_seconds"]
            # This operation's active reservation is charged; a conservative
            # extra minute covers conditional feature preparation/loading.
            affordable = remaining >= 2 * plan["estimated_seconds_per_run"] + 120 + 60
            base_recipe = next(recipe for recipe in self.training["recipes_in_execution_order"] if recipe["name"] == base["recipe"])
            report = {"schema": "stage2b-contingent-decision-v1", "config_sha256": self.config_sha256,
                "base_recipe": base["recipe"], "base_checkpoint": base["primary"]["checkpoint"],
                "base_checkpoint_sha256": base["primary"]["checkpoint_sha256"],
                "student_minus_teacher_discard_error": gap, "validation_lost_ukeire_rate": val_loss,
                "frozen_train_lost_ukeire_rate": train_loss, "trigger": trigger, "two_runs_affordable_estimate": affordable,
                "eligible_to_run": bool(trigger == "aggregation" and affordable),
                "recipe": {"name": base["recipe"] + "_aggregation", "features": "v2",
                           "rank_weight": base_recipe["rank_weight"], "aggregation": True} if trigger == "aggregation" else None,
                "status": ("no_trigger" if trigger is None else "skip_budget" if not affordable else
                           "prepare_frozen_aggregation_corpus" if trigger == "aggregation" else "requires_frozen_remaining_hand_implementation"),
                "scope": "Allocation thresholds are descriptive, not significance tests. At most one two-seed recipe; no controls or continuation added."}
            self.save("contingent-plan.json", report, exclusive=True)
        print(json.dumps(report), flush=True)

    def select(self):
        with self.ledger.operation("freeze-final-selection", allowance_seconds=60):
            if not (self.reports / "contingent-plan.json").exists():
                raise RuntimeError("Record the conditional allocation decision before final selection")
            contingent = self.read("contingent-plan.json")
            if contingent["status"] == "requires_frozen_remaining_hand_implementation":
                raise RuntimeError("Resolve declared remaining-hand experiment before final selection")
            ranking = self.ranking()
            base_names = {recipe["name"] for recipe in self.training["recipes_in_execution_order"]}
            if base_names.intersection(ranking["incomplete_recipes"]) or not ranking["winner"]:
                raise RuntimeError("Every declared eligible recipe needs two complete common-update runs")
            primary = ranking["winner"]["primary"]
            target = self.data / "models" / "selected.pt"
            if target.exists() or (self.reports / "selection.json").exists():
                raise FileExistsError("Final selection is already frozen")
            shutil.copy2(primary["checkpoint"], target)
            summary = {"schema": "stage2b-frozen-selection-v1", "config_sha256": self.config_sha256,
                "checkpoint": str(target), "checkpoint_sha256": digest(target),
                "recipe": primary["recipe"], "seed": self.training["primary_seed"],
                "source_checkpoint": primary["checkpoint"], "primary_fixed_before_test_and_development": True,
                "common_steps": self.read("training-plan.json")["common_steps"],
                "excluded_incomplete_recipes": ranking["incomplete_recipes"],
                "eligible_recipes": [{"name": item["recipe"], "mean_selection_key": item["mean_selection_key"],
                    "runs": [{key: run[key] for key in ("checkpoint", "checkpoint_sha256", "seed", "recipe")}
                             for run in item["per_seed"]]} for item in ranking["eligible_recipes"]],
                "selection_rule": self.training["recipe_selection"]}
            if summary["checkpoint_sha256"] != primary["checkpoint_sha256"]:
                raise RuntimeError("Frozen checkpoint copy hash differs")
            self.save("selection.json", summary, exclusive=True)
        print(json.dumps(summary), flush=True)

    def test(self, device):
        import torch
        from riichi_ai.model import NeuralPolicy
        from riichi_ai.stage2b_training import evaluate_stage2b
        torch.set_num_threads(1)
        if (self.reports / "action-test.json").exists():
            raise FileExistsError("Action test is one frozen evaluation, including failed attempts")
        selection = self.read("selection.json")
        if selection["config_sha256"] != self.config_sha256 or digest(selection["checkpoint"]) != selection["checkpoint_sha256"]:
            raise RuntimeError("Frozen selection identity changed")
        report = {"selection": selection, "failure": None, "complete": False}
        try:
            with self.ledger.operation("frozen-action-test") as lease:
                policy = NeuralPolicy.from_checkpoint(selection["checkpoint"], device=device)
                assisted = policy.model.config["feature_schema"] == "actor-features-v2"
                data = self.load("test", assisted, allow_test=True)
                self.memory_check(device)
                report["metrics"] = evaluate_stage2b(policy.model, data, device=device, max_seconds=self.remaining(lease))
                report["complete"] = report["metrics"]["complete"]
                self.disk_check()
        except BaseException as error:
            report["failure"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            report["budget"] = self.ledger.snapshot()
            self.save("action-test.json", report, exclusive=True)
        print(json.dumps({"phase": "frozen-action-test", "complete": report["complete"],
                          "equal_stratum_means": report["metrics"]["equal_stratum_means"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT / "configs/stage2b.json"))
    commands = parser.add_subparsers(dest="command", required=True)
    checks = commands.add_parser("checks")
    checks.add_argument("--device", default="mps", choices=("cpu", "mps"))
    checks.add_argument("--max-seconds", type=float, default=360)
    commands.add_parser("plan")
    run = commands.add_parser("run")
    run.add_argument("--recipe", required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--device", default="mps", choices=("cpu", "mps"))
    campaign = commands.add_parser("campaign")
    campaign.add_argument("--device", default="mps", choices=("cpu", "mps"))
    commands.add_parser("contingent")
    commands.add_parser("select")
    test = commands.add_parser("test")
    test.add_argument("--device", default="mps", choices=("cpu", "mps"))
    commands.add_parser("budget")
    account = commands.add_parser("account")
    account.add_argument("--name", required=True)
    account.add_argument("--seconds", type=float, required=True)
    account.add_argument("--failed", action="store_true")
    account.add_argument("--evidence")
    budgeted = commands.add_parser("budgeted")
    budgeted.add_argument("--name", required=True)
    budgeted.add_argument("--max-seconds", type=float)
    budgeted.add_argument("--reserve-seconds", type=float, default=120)
    budgeted.add_argument("subprocess_command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    campaign = Campaign(args.config)
    if args.command == "checks": campaign.checks(args.device, args.max_seconds)
    elif args.command == "plan": campaign.plan()
    elif args.command == "run": campaign.run(args.recipe, args.seed, args.device)
    elif args.command == "campaign": campaign.campaign(args.device)
    elif args.command == "contingent": campaign.contingent()
    elif args.command == "select": campaign.select()
    elif args.command == "test": campaign.test(args.device)
    elif args.command == "account":
        print(json.dumps(campaign.ledger.account(args.name, args.seconds, failed=args.failed, evidence=args.evidence)), flush=True)
    elif args.command == "budgeted":
        campaign.budgeted(args.name, args.subprocess_command, args.max_seconds, args.reserve_seconds)
    else: print(json.dumps(campaign.ledger.snapshot()), flush=True)


if __name__ == "__main__":
    main()
