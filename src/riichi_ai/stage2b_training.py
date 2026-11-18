"""Matched Stage 2B imitation experiments and stratified development metrics.

Teacher preferences remain diagnostic targets outside the encoded actor input.
All reported action metrics are descriptive; full matches determine strength.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .model import efficiency_model_config, matched_initialization, model_config
from .semantics import ACTION_KINDS
from .training import _atomic_json, _data_manifest, _sync, collate_training, train


PROTOCOL = "stage2b-equal-stratum-discard-v1"
STRATA = ("teacher", "student")
DISCARD = ACTION_KINDS.index("DISCARD") + 1
COMPONENTS = ("branch", "danger", "shanten", "negative_ukeire", "red")
SELECTION_RULE = "equal_stratum_discard_tie_then_lower_shanten_worsening_then_lower_optional_kl"


class EqualRowMixture(Sequence):
    """Exactly half each even minibatch, independently uniform within each pool."""
    sampling_protocol = "stage2b-half-original-half-student-uniform-replacement-v1"

    def __init__(self, original: Sequence, student: Sequence):
        if not len(original) or not len(student):
            raise ValueError("Both mixture populations must be nonempty")
        self.original, self.student = original, student
        self.stratum_indices = {"teacher": np.arange(len(original)),
                                "student": np.arange(len(original), len(self))}

    def __len__(self):
        return len(self.original) + len(self.student)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.original[index] if index < len(self.original) else self.student[index - len(self.original)]

    def sample_indices(self, rng, batch_size):
        if batch_size % 2:
            raise ValueError("An exact 50:50 minibatch requires an even batch size")
        half = batch_size // 2
        result = np.concatenate((rng.integers(len(self.original), size=half),
                                 len(self.original) + rng.integers(len(self.student), size=half)))
        return result[rng.permutation(batch_size)]

    def dataset_identity(self):
        return {"schema": self.sampling_protocol, "rows": len(self),
                "original": _data_manifest(self.original), "student": _data_manifest(self.student),
                "weights": [.5, .5], "minibatch_allocation": "equal, uniform replacement within each source"}


def _index_hash(indices: Sequence[int]) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def complete_validation_schedule(report: Mapping, steps: int, interval: int) -> bool:
    """A finished optimizer loop alone does not establish a complete comparison."""
    expected = list(range(interval, steps + 1, interval))
    if not expected or expected[-1] != steps:
        expected.append(steps)
    entries = report.get("validations", ())
    return ([item["step"] for item in entries] == expected
            and all(item.get("complete") for item in entries))


def stratified_indices(dataset: Sequence, *, per_stratum: int = 5000, seed: int = 2026101720,
                       require_both: bool = True) -> list[int]:
    """One fixed, seed-independent-of-training sample per named population."""
    if type(per_stratum) is not int or per_stratum < 1:
        raise ValueError("per_stratum must be a positive integer")
    declared = getattr(dataset, "stratum_indices", None)
    if callable(declared):
        declared = declared()
    pools = ({name: list(map(int, declared.get(name, ()))) for name in STRATA}
             if declared is not None else {name: [] for name in STRATA})
    if declared is None:
        for index, record in enumerate(dataset):
            if record.get("stratum") not in STRATA:
                raise ValueError("Every Stage 2B sample needs a teacher/student stratum")
            pools[record["stratum"]].append(index)
    if require_both and any(not pools[name] for name in STRATA):
        raise ValueError("Selection validation must contain both strata")
    all_indices = [index for name in STRATA for index in pools[name]]
    if len(set(all_indices)) != len(all_indices) or any(not 0 <= i < len(dataset) for i in all_indices):
        raise ValueError("Stratum indices must be disjoint valid rows")
    result = []
    for name in STRATA:
        name_seed = int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "big")
        pool = sorted(pools[name])
        if pool:
            result.extend(sorted(map(int, np.random.default_rng(name_seed).choice(
                pool, size=min(per_stratum, len(pool)), replace=False))))
    return result


class _Totals:
    def __init__(self) -> None:
        self.rows = self.optional = self.illegal = self.discards = self.discard_tie = self.discard_exact = 0
        self.sums = np.zeros(8, dtype=np.float64)
        self.shanten = Counter()
        self.ukeire = Counter()
        self.first_difference = Counter({key: 0 for key in COMPONENTS})
        self.action_kind_errors = 0
        self.bins = np.zeros((10, 3), dtype=np.float64)
        self.matches: dict[str, dict[str, int]] = {}

    def add(self, record: Mapping, values: np.ndarray, chosen: int, legal: bool,
            rank_confidence: float, rank_correct: bool) -> None:
        self.rows += 1
        self.illegal += int(not legal)
        optional = len(record["encoded"]["action_tokens"]) > 1
        self.optional += int(optional)
        self.sums[5:] += values[5:]
        if optional:
            self.sums[:5] += values[:5]
        bin_id = min(int(rank_confidence * 10), 9)
        self.bins[bin_id] += (1, rank_confidence, int(rank_correct))
        if not optional:
            return
        tokens = record["encoded"]["action_tokens"]
        teacher = int(record["chosen_index"])
        if int(tokens[teacher, 0]) != DISCARD:
            return
        preferences = np.asarray(record["teacher_preference_keys"])
        if preferences.shape != (len(tokens), 5):
            raise ValueError("Diagnostic teacher preferences must have shape [candidates,5]")
        best, predicted = preferences[teacher], preferences[chosen]
        correct = bool(record["best_mask"][chosen])
        if correct != bool(np.array_equal(best, predicted)):
            raise ValueError("Preference keys disagree with the supervised best mask")
        self.discards += 1
        self.discard_tie += int(correct)
        self.discard_exact += int(chosen == teacher)
        match = self.matches.setdefault(str(record.get("match_id", "provided_fixture")),
                                        {"optional_teacher_discard_rows": 0, "errors": 0,
                                         "worse_shanten_rows": 0, "lost_ukeire_rows": 0})
        match["optional_teacher_discard_rows"] += 1
        if correct:
            return
        match["errors"] += 1
        first = int(np.flatnonzero(best != predicted)[0])
        if predicted[first] <= best[first]:
            raise ValueError("Selected preference unexpectedly improves the teacher's minimum")
        self.first_difference[COMPONENTS[first]] += 1
        if int(tokens[chosen, 0]) != DISCARD:
            self.action_kind_errors += 1
            return
        delta = int(predicted[2]) - int(best[2])
        if delta > 0:
            self.shanten[delta] += 1
            match["worse_shanten_rows"] += 1
        if delta == 0:
            lost = int(predicted[3]) - int(best[3])
            if lost > 0:
                self.ukeire[lost] += 1
                match["lost_ukeire_rows"] += 1

    def finish(self) -> dict[str, Any]:
        optional, rows, discards = self.optional, self.rows, self.discards
        bins = [{"lower": i / 10, "upper": (i + 1) / 10, "upper_inclusive": i == 9,
                 "count": int(row[0]), "mean_confidence": float(row[1] / row[0]) if row[0] else None,
                 "accuracy": float(row[2] / row[0]) if row[0] else None}
                for i, row in enumerate(self.bins)]
        def severity(histogram: Counter) -> dict:
            count = sum(histogram.values())
            total = sum(value * frequency for value, frequency in histogram.items())
            return {"count": count, "total": total, "mean": total / count if count else None,
                    "maximum": max(histogram) if histogram else None,
                    "histogram": {str(k): int(v) for k, v in sorted(histogram.items())}}
        return {"rows": rows, "optional_rows": optional, "illegal_choices": self.illegal,
                **{key: float(self.sums[i] / optional) if optional else None
                   for i, key in enumerate(("policy_ce", "policy_kl", "tie_agreement", "exact_agreement", "random_legal_tie_agreement"))},
                **{key: float(self.sums[5 + i] / rows) if rows else None
                   for i, key in enumerate(("rank_nll", "rank_brier", "rank_accuracy"))},
                "rank_top_confidence_ece_10_bins": float(sum(abs(row[1] - row[2]) for row in self.bins) / rows) if rows else None,
                "rank_reliability": bins,
                "optional_discard_rows": discards,
                "discard_tie_agreement": self.discard_tie / discards if discards else None,
                "discard_exact_agreement": self.discard_exact / discards if discards else None,
                "shanten_worsening_rate": sum(self.shanten.values()) / discards if discards else None,
                "same_shanten_lost_ukeire_rate": sum(self.ukeire.values()) / discards if discards else None,
                "shanten_increments": severity(self.shanten), "same_shanten_lost_ukeire": severity(self.ukeire),
                "red_only_errors": self.first_difference["red"], "wrong_action_kind_errors": self.action_kind_errors,
                "first_difference_counts": dict(self.first_difference),
                "matches_with_discard_rows": len(self.matches), "by_match": self.matches}


def evaluate_stage2b(model: Any, dataset: Sequence, *, device: Any = "cpu", batch_size: int = 256,
                    indices: Sequence[int] | None = None, max_samples: int | None = None,
                    seed: int = 2026101720, target_mode: str = "soft", max_seconds: float | None = None,
                    require_both_strata: bool = True) -> dict[str, Any]:
    """One inference pass, equal-population selection, all-state rank metrics."""
    started = time.perf_counter()
    if not len(dataset) or batch_size < 1 or target_mode != "soft":
        raise ValueError("Stage 2B evaluation requires nonempty data and soft targets")
    if indices is None:
        indices = (stratified_indices(dataset, per_stratum=max(1, max_samples // 2), seed=seed,
                                      require_both=require_both_strata)
                   if max_samples is not None else list(range(len(dataset))))
    indices = list(indices)
    if (not indices or len(set(indices)) != len(indices)
            or any(type(index) is not int or not 0 <= index < len(dataset) for index in indices)):
        raise ValueError("Evaluation requires unique valid row indices")
    device = torch.device(device)
    model.eval()
    totals = _Totals()
    by_stratum = {name: _Totals() for name in STRATA}
    with torch.inference_mode():
        for offset in range(0, len(indices), batch_size):
            if max_seconds is not None and time.perf_counter() - started >= max_seconds:
                break
            records = [dataset[index] for index in indices[offset:offset + batch_size]]
            batch, labels = collate_training(records, device, "soft")
            logits, rank_logits = model(batch)
            if not bool(torch.isfinite(logits).all() and torch.isfinite(rank_logits).all()):
                raise FloatingPointError("Nonfinite Stage 2B evaluation logits")
            logp = logits.log_softmax(-1)
            ce = -(labels["q"] * logp).sum(-1)
            entropy = -(labels["q"] * labels["q"].clamp_min(1e-30).log()).sum(-1)
            selected = logits.argmax(-1)
            tie = labels["best"].gather(1, selected[:, None]).squeeze(1)
            exact = selected == labels["chosen"]
            legal = batch["candidate_mask"].gather(1, selected[:, None]).squeeze(1)
            random_tie = labels["best"].sum(1) / batch["candidate_mask"].sum(1)
            probabilities = rank_logits.softmax(-1)
            correct = probabilities.argmax(-1) == labels["rank"]
            nll = F.cross_entropy(rank_logits, labels["rank"], reduction="none")
            brier = (probabilities - F.one_hot(labels["rank"], 4)).square().sum(-1)
            values = torch.stack((ce, ce - entropy, tie, exact, random_tie, nll, brier, correct), 1).cpu().numpy()
            choices, legality = selected.cpu().tolist(), legal.cpu().tolist()
            confidence, correctness = probabilities.max(1).values.cpu().tolist(), correct.cpu().tolist()
            for row, record in enumerate(records):
                stratum = record.get("stratum")
                if stratum not in by_stratum:
                    raise ValueError("Evaluation sample lacks its teacher/student population")
                if not legality[row]:
                    raise ValueError("Policy chose a padded candidate")
                arguments = (record, values[row], choices[row], legality[row], confidence[row], correctness[row])
                totals.add(*arguments)
                by_stratum[stratum].add(*arguments)
    _sync(device)
    result = totals.finish()
    result.update(schema=PROTOCOL, complete=totals.rows == len(indices), requested_rows=len(indices),
                  target_mode="soft", strata={name: value.finish() for name, value in by_stratum.items()},
                  indices_sha256=_index_hash(indices), elapsed_seconds=time.perf_counter() - started,
                  scope="Within-stratum decision-weighted metrics; selection averages teacher/student rates equally. Rows within matches are correlated.")
    usable = all(by_stratum[name].discards and by_stratum[name].optional for name in STRATA)
    if result["complete"] and require_both_strata and not usable:
        raise ValueError("Selection requires optional discards and policy rows in both strata")
    if usable:
        means = {key: float(np.mean([result["strata"][name][key] for name in STRATA]))
                 for key in ("discard_tie_agreement", "shanten_worsening_rate", "policy_kl", "same_shanten_lost_ukeire_rate")}
        result["equal_stratum_means"] = means
        result["checkpoint_selection_key"] = [means["discard_tie_agreement"], -means["shanten_worsening_rate"], -means["policy_kl"]]
    else:
        result["equal_stratum_means"] = None
        result["checkpoint_selection_key"] = None
    return result


def train_stage2b(dataset: Sequence, output_dir: str | Path, *, validation: Sequence,
                 assisted: bool, seed: int, rank_weight: float = .1,
                 validation_seed: int = 2026101720, validation_per_stratum: int = 5000,
                 raw_config: Mapping | None = None, assisted_config: Mapping | None = None,
                 protocol_manifest: Any = None, **kwargs: Any) -> dict[str, Any]:
    """Shared raw tensors and paired sampler seeds are explicit artifacts."""
    raw, feature, initialization = matched_initialization(seed, raw_config, assisted_config)
    config = initialization["assisted_config" if assisted else "raw_config"]
    indices = stratified_indices(validation, per_stratum=validation_per_stratum, seed=validation_seed)
    protocol = {"schema": PROTOCOL, "validation_seed": validation_seed,
                "validation_per_stratum": validation_per_stratum,
                "validation_indices_sha256": _index_hash(indices), "declared": protocol_manifest}
    return train(dataset, output_dir, validation=validation, seed=seed, rank_weight=rank_weight,
                 config=config, initial_state_dict=feature if assisted else raw,
                 initialization_manifest=initialization, validation_indices=indices,
                 evaluation_fn=evaluate_stage2b, evaluation_protocol=PROTOCOL,
                 selection_rule=SELECTION_RULE, protocol_manifest=protocol, **kwargs)


def select_recipes(runs: Sequence[Mapping], *, required_seeds: Sequence[int], primary_seed: int,
                   recipe_order: Sequence[str]) -> dict[str, Any]:
    """Only complete two-seed recipes qualify; final seed cannot be cherry-picked."""
    required = tuple(sorted(required_seeds))
    if len(required) != 2 or len(set(required)) != 2 or primary_seed not in required:
        raise ValueError("Two distinct declared seeds and their fixed primary seed are required")
    eligible, incomplete = [], []
    for name in recipe_order:
        selected = [run for run in runs if run["recipe"] == name]
        if (len(selected) != len(required) or tuple(sorted(run["seed"] for run in selected)) != required
                or any(not run.get("eligible", False) or not run["full_validation"]["complete"] for run in selected)):
            incomplete.append(name)
            continue
        keys = np.asarray([run["full_validation"]["checkpoint_selection_key"] for run in selected], dtype=np.float64)
        if keys.shape != (2, 3) or not np.isfinite(keys).all():
            raise ValueError("Invalid full-validation selection metrics")
        eligible.append({"recipe": name, "mean_selection_key": keys.mean(0).tolist(),
                         "per_seed": sorted(selected, key=lambda run: run["seed"]),
                         "primary": next(run for run in selected if run["seed"] == primary_seed)})
    eligible.sort(key=lambda item: tuple(-x for x in item["mean_selection_key"]))
    return {"schema": PROTOCOL, "eligible_recipes": eligible, "incomplete_recipes": incomplete,
            "winner": eligible[0] if eligible else None, "primary_seed": primary_seed,
            "recipe_order_breaks_exact_ties": list(recipe_order)}


class BudgetLedger:
    """Persist actual operation durations; crashed reservations remain charged.

    An active operation reserves its entire allowance. Thus an unreported kill
    cannot silently refund spent time. Recovery of a crashed lease is an
    explicit evidence-based bookkeeping action, never an automatic reset.
    """
    def __init__(self, path: str | Path, *, limit_seconds: float = 3600, identity: Any = None):
        self.path = Path(path)
        self.limit = float(limit_seconds)
        self.identity = identity
        if not math.isfinite(self.limit) or self.limit <= 0:
            raise ValueError("Positive finite budget required")

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(self.path.suffix + ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = (json.loads(self.path.read_text()) if self.path.exists() else
                    {"schema": "stage2b-time-ledger-v1", "limit_seconds": self.limit,
                     "identity": self.identity, "operations": []})
            if data["limit_seconds"] != self.limit or data["identity"] != self.identity:
                raise ValueError("Budget ledger identity/limit differs")
            yield data
            _atomic_json(self.path, data)

    @staticmethod
    def charged(data: Mapping) -> float:
        return sum(item["reserved_seconds"] if item["status"] == "active" else item["elapsed_seconds"]
                   for item in data["operations"])

    def snapshot(self) -> dict:
        with self._locked() as data:
            result = dict(data)
            result["charged_seconds"] = self.charged(data)
            result["remaining_seconds"] = max(0., self.limit - result["charged_seconds"])
            return result

    def account(self, name: str, elapsed_seconds: float, *, failed: bool = False,
                evidence: Any = None) -> dict:
        """Append an externally measured duration, including failed attempts.

        This never authorizes new work or refunds a lease. Prefer operation()
        when a task has not started; account() records already incurred costs.
        """
        elapsed = float(elapsed_seconds)
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Measured elapsed seconds must be finite and nonnegative")
        with self._locked() as data:
            entry = {"id": len(data["operations"]), "name": name,
                     "status": "failed" if failed else "completed", "reserved_seconds": 0,
                     "elapsed_seconds": elapsed, "recorded_unix": time.time(),
                     "externally_measured": True, "evidence": evidence}
            data["operations"].append(entry)
        return self.snapshot()

    @contextmanager
    def operation(self, name: str, *, allowance_seconds: float | None = None, reserve_seconds: float = 0):
        if not math.isfinite(reserve_seconds) or reserve_seconds < 0:
            raise ValueError("Reserve must be finite and nonnegative")
        with self._locked() as data:
            available = self.limit - self.charged(data) - reserve_seconds
            allowance = available if allowance_seconds is None else min(float(allowance_seconds), available)
            if not math.isfinite(allowance) or allowance <= 0:
                raise RuntimeError("Cumulative Stage 2B training-related budget exhausted")
            identifier = len(data["operations"])
            entry = {"id": identifier, "name": name, "status": "active", "reserved_seconds": allowance,
                     "started_unix": time.time(), "elapsed_seconds": None}
            data["operations"].append(entry)
        started = time.perf_counter()
        failure = None
        try:
            yield {"id": identifier, "allowed_seconds": allowance, "started_monotonic": started}
        except BaseException as error:
            failure = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            elapsed = time.perf_counter() - started
            with self._locked() as data:
                data["operations"][identifier].update(status="failed" if failure else "completed",
                                                       elapsed_seconds=elapsed, failure=failure,
                                                       exceeded_reservation=elapsed > allowance)
