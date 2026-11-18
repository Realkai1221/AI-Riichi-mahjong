"""Public placement prediction and whole-match cross-fitted hand potentials.

The feature function accepts only a start-of-hand public summary and the
explorer's initial seat. Labels, final scores, behaviors and engine logs never
enter its feature array. Four relative scores, score gaps, round, continuation,
deposits, relative dealer and visible initial-seat tie order have fixed scales.

Every game has weight one; its start states each receive 1 / number_of_starts.
All five OOF models and the final train-only model have the same initialization,
80 fixed epochs, batch 512 and Adam lr .003. No validation early stopping occurs.

The calibration gate is fixed before fitting: development NLL and Brier must be
no worse than the unfitted score/10000 Plackett-Luce rank baseline. Overall and
each early (East 1-2), middle (East 3-South 2), late (South 3+) stratum must have
100 unique games. Utility bins [-1.5,-.9,-.3,.3,.9,1.5] are supported only with
50 unique games, must cover >=90% of stratum weight, have weighted absolute
calibration bias <=.15 and maximum supported-bin bias <=.30. Sparse occupied
bins are reported and count against coverage. Bootstrap intervals are descriptive
and never change the gate. Gate failure leaves delta targets audit-only.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn

CORPUS_SCHEMA = "riichi-ai-stage3g-corpus-v1"
TARGET_SCHEMA = "riichi-ai-stage3g-hand-targets-v1"
PUBLIC_TARGET_SCHEMA = "riichi-ai-stage3f-match-targets-v1"
FOLD_SEED = 20261023021
INITIALIZATION_SEED = 20261023022
BOOTSTRAP_SEED = 20261023023
FOLDS = 5
EPOCHS = 80
BATCH_ROWS = 512
LEARNING_RATE = .003
UTILITY = np.asarray([1.5, .5, -.5, -1.5], dtype=np.float64)
UTILITY_BINS = np.asarray([-1.5, -.9, -.3, .3, .9, 1.5], dtype=np.float64)
CALIBRATION = {"minimum_unique_games": 100, "minimum_bin_unique_games": 50,
               "minimum_supported_mass": .90, "maximum_weighted_absolute_bias": .15,
               "maximum_supported_bin_bias": .30, "utility_bins": UTILITY_BINS.tolist()}
FEATURE_NAMES = tuple([f"relative_score_{i}/10000" for i in range(4)]
    + [f"score_gap_to_self_{i}/10000" for i in range(4)]
    + [f"round_{v}" for v in "ESWN"] + [f"kyoku_{i}" for i in range(1, 5)]
    + ["honba/10", "kyotaku/10"] + [f"relative_dealer_{i}" for i in range(4)]
    + [f"initial_seat_{i}" for i in range(4)]
    + [f"relative_seat_tie_priority_{i}/3" for i in range(4)] + ["hand_ordinal/20"])


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _integer(value: Any, name: str, low: int, high: int | None = None) -> int:
    if type(value) is not int or value < low or (high is not None and value > high):
        raise ValueError(f"Invalid {name}")
    return value


def public_features(hand: dict[str, Any], seat: int) -> np.ndarray:
    """Whitelist public fields explicitly, ignoring any extra outcome metadata."""
    _integer(seat, "seat", 0, 3)
    wind = hand["bakaze"]
    if not isinstance(wind, str) or len(wind) != 1 or wind not in "ESWN":
        raise ValueError("Invalid public round wind")
    kyoku = _integer(hand["kyoku"], "kyoku", 1, 4)
    honba = _integer(hand["honba"], "honba", 0)
    sticks = _integer(hand["kyotaku"], "kyotaku", 0)
    dealer = _integer(hand["oya"], "dealer", 0, 3)
    ordinal = _integer(hand["hand_id"], "hand ordinal", 1)
    scores = hand["scores"]
    if (not isinstance(scores, list) or len(scores) != 4
            or any(type(score) is not int for score in scores)):
        raise ValueError("Invalid public scores")
    order = [(seat + i) % 4 for i in range(4)]
    values = [scores[i] / 10000 for i in order]
    values += [(scores[i] - scores[seat]) / 10000 for i in order]
    values += [float(wind == item) for item in "ESWN"]
    values += [float(kyoku == i) for i in range(1, 5)]
    values += [honba / 10, sticks / 10]
    values += [float((dealer - seat) % 4 == i) for i in range(4)]
    values += [float(seat == i) for i in range(4)]
    values += [i / 3 for i in order]  # Initial-seat ascending is the public tie rule.
    values += [ordinal / 20]
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError("Nonfinite or malformed public features")
    return result


def stage(hand: dict[str, Any]) -> int:
    round_number = "ESWN".index(hand["bakaze"]) * 4 + hand["kyoku"] - 1
    return 0 if round_number < 2 else 1 if round_number < 6 else 2


def score_baseline(hand: dict[str, Any], seat: int) -> np.ndarray:
    """Unfitted Plackett-Luce rank marginal, enumerating every permutation.

    Soft ranking intentionally does not deterministically break score ties; the
    exact initial-seat tie rule is nevertheless visible to the learned model.
    A shifted exponential avoids overflow without changing PL probabilities.
    """
    public_features(hand, seat)
    scores = np.asarray(hand["scores"], dtype=np.float64) / 10000
    scores -= scores.max()
    weights = np.exp(scores)
    result = np.zeros(4, dtype=np.float64)
    for permutation in itertools.permutations(range(4)):
        probability = 1.
        for rank, contender in enumerate(permutation[:-1]):
            probability *= weights[contender] / weights[list(permutation[rank:])].sum()
        result[permutation.index(seat)] += probability
    if not np.isfinite(result).all() or abs(float(result.sum()) - 1) > 1e-12:
        raise ValueError("Invalid baseline rank probabilities")
    return result


def assign_folds(match_ids: list[str]) -> dict[str, int]:
    """Deterministic balanced hash order; no outcomes or file order used."""
    if len(set(match_ids)) != len(match_ids) or len(match_ids) < FOLDS:
        raise ValueError("At least five distinct training matches are required")
    ordered = sorted(match_ids, key=lambda value: digest([FOLD_SEED, value]))
    return {name: index % FOLDS for index, name in enumerate(ordered)}


@dataclass
class PublicDataset:
    x: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    groups: np.ndarray
    stages: np.ndarray
    baseline: np.ndarray
    match_ids: list[str]
    matches: dict[str, dict[str, Any]]
    splits: dict[str, list[str]]
    manifest_sha256: str
    source_hashes: dict[str, str]

    def rows_for(self, names: list[str]) -> np.ndarray:
        wanted = set(names)
        if not wanted <= set(self.match_ids):
            raise ValueError("Unknown match requested")
        return np.flatnonzero(np.asarray([self.match_ids[int(i)] in wanted for i in self.groups]))


def load_public_dataset(directory: str | Path, *, expected_training: int = 3200,
                        expected_development: int = 800) -> PublicDataset:
    """Read exactly manifest and public target JSON; no actor/auditor files."""
    directory = Path(directory).resolve()
    raw_manifest = (directory / "manifest.json").read_bytes()
    manifest = json.loads(raw_manifest)
    if (manifest.get("schema") != CORPUS_SCHEMA or manifest.get("complete") is not True
            or manifest.get("source_role") != "offline-learning-corpus"):
        raise ValueError("Expected complete Stage 3G offline corpus")
    splits = manifest.get("splits", {})
    if set(splits) != {"training", "development"}:
        raise ValueError("Only declared training/development splits are allowed")
    train, dev = splits["training"], splits["development"]
    if (len(train) != expected_training or len(dev) != expected_development
            or len(set(train + dev)) != len(train) + len(dev)):
        raise ValueError("Unexpected match counts or overlapping splits")
    reports = manifest.get("matches", [])
    if len(reports) != len(train) + len(dev):
        raise ValueError("Manifest match count mismatch")
    by_id = {item["match_id"]: item for item in reports}
    if set(by_id) != set(train + dev) or len(by_id) != len(reports):
        raise ValueError("Manifest match identity mismatch")
    xs, ys, weights, groups, stages, baselines = [], [], [], [], [], []
    matches, sources = {}, {}
    ids = sorted(by_id)
    training = set(train)
    for group, name in enumerate(ids):
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ValueError("Unsafe match identity")
        report = by_id[name]
        split = "training" if name in training else "development"
        role = "offline-training" if split == "training" else "development-only"
        if (report.get("completed") is not True or report.get("failure") is not None
                or report.get("source_role") != role or report.get("role") != split
                or report.get("terminal_result_verified") is not True):
            raise ValueError("Ineligible match or source role")
        behaviors = report.get("seat_behaviors")
        if (not isinstance(behaviors, list) or len(behaviors) != 4
                or sorted(behaviors) != ["defense_v2", "explorer", "reference", "shanten_v2"]):
            raise ValueError("Invalid explorer identity")
        seat = behaviors.index("explorer")
        relative = f"targets/{name}.json"
        path = directory / relative
        if path.is_symlink() or path.resolve().parent != (directory / "targets").resolve():
            raise ValueError("Unsafe public target path")
        raw = path.read_bytes()
        identity = hashlib.sha256(raw).hexdigest()
        artifact = report.get("artifacts", {}).get(relative, {})
        if artifact.get("sha256") != identity or artifact.get("bytes") != len(raw):
            raise ValueError("Public target artifact identity mismatch")
        target = json.loads(raw)
        if (target.get("schema") != PUBLIC_TARGET_SCHEMA or target.get("source_role") != role
                or target.get("terminal_result_verified") is not True):
            raise ValueError("Invalid public target record")
        ranks, utilities = target.get("ranks"), target.get("utilities")
        if (not isinstance(ranks, list) or len(ranks) != 4 or any(type(rank) is not int for rank in ranks)
                or sorted(ranks) != [1, 2, 3, 4]
                or utilities != [float(UTILITY[rank - 1]) for rank in ranks]
                or report.get("ranks") != ranks or report.get("utilities") != utilities):
            raise ValueError("Invalid or mismatched terminal labels")
        hands = target.get("hand_starts")
        if not isinstance(hands, list) or not hands or [v["hand_id"] for v in hands] != list(range(1, len(hands) + 1)):
            raise ValueError("Missing explicit consecutive hand ordinals")
        if report.get("hands") != len(hands):
            raise ValueError("Public hand count mismatch")
        for hand in hands:
            xs.append(public_features(hand, seat))
            ys.append(ranks[seat] - 1)
            weights.append(1 / len(hands))
            groups.append(group)
            stages.append(stage(hand))
            baselines.append(score_baseline(hand, seat))
        # Only whitelisted public start fields and separately named final labels.
        matches[name] = {"split": split, "source_role": role, "explorer_seat": seat,
            "hand_ids": [hand["hand_id"] for hand in hands], "terminal_utility": utilities[seat]}
        sources[relative] = identity
    return PublicDataset(np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64),
        np.asarray(weights, dtype=np.float64), np.asarray(groups, dtype=np.int64),
        np.asarray(stages, dtype=np.int8), np.asarray(baselines, dtype=np.float64),
        ids, matches, {key: list(value) for key, value in splits.items()},
        hashlib.sha256(raw_manifest).hexdigest(), sources)


class PlacementPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(len(FEATURE_NAMES), 32), nn.Tanh(),
                                    nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def initialized_predictor() -> PlacementPredictor:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(INITIALIZATION_SEED)
        return PlacementPredictor()


def _deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("Stage 3G placement fit deadline; no partial fit accepted")


def fit_predictor(dataset: PublicDataset, train_ids: list[str], deadline: float,
                  *, epochs: int = EPOCHS) -> tuple[PlacementPredictor, dict[str, Any]]:
    """Fixed shuffled minibatch weighted objective; development is forbidden."""
    if not train_ids or len(set(train_ids)) != len(train_ids) or not set(train_ids) <= set(dataset.splits["training"]):
        raise ValueError("Predictor fit contains non-training or duplicate games")
    rows = dataset.rows_for(train_ids)
    x = torch.from_numpy(dataset.x[rows].copy())
    y = torch.from_numpy(dataset.y[rows].copy())
    # A uniform-row minibatch mean estimates mean_game mean_hand loss.
    w = torch.from_numpy(dataset.weights[rows].astype(np.float32)) * (len(rows) / len(train_ids))
    if not np.allclose([dataset.weights[rows][dataset.groups[rows] == group].sum()
                        for group in np.unique(dataset.groups[rows])], 1, atol=1e-12, rtol=0):
        raise ValueError("Each training game must have total state weight one")
    model = initialized_predictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator(device="cpu").manual_seed(INITIALIZATION_SEED)
    steps = 0
    tick = time.monotonic()
    for epoch in range(epochs):
        _deadline(deadline)
        order = torch.randperm(len(rows), generator=generator)
        for begin in range(0, len(rows), BATCH_ROWS):
            _deadline(deadline)
            selected = order[begin:begin + BATCH_ROWS]
            optimizer.zero_grad(set_to_none=True)
            loss = (nn.functional.cross_entropy(model(x[selected]), y[selected], reduction="none") * w[selected]).mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite placement fit objective")
            loss.backward()
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise ValueError("Nonfinite placement fit gradient")
            optimizer.step()
            steps += 1
    model.eval()
    with torch.inference_mode():
        losses = nn.functional.cross_entropy(model(x), y, reduction="none")
        train_nll = float((losses * torch.from_numpy(dataset.weights[rows].astype(np.float32))).sum() / len(train_ids))
    return model, {"epochs": epochs, "optimizer_steps": steps, "rows": len(rows),
        "training_matches": len(train_ids), "training_match_ids": sorted(train_ids),
        "training_match_ids_sha256": digest(sorted(train_ids)), "train_weighted_nll": train_nll,
        "elapsed_seconds": time.monotonic() - tick,
        "weighting": "Uniform row shuffle; minibatch mean of N_rows/N_games * (1/N_hand_states_in_game) * NLL"}


def predict(model: PlacementPredictor, x: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        probabilities = torch.softmax(model(torch.from_numpy(x.copy())), dim=-1).numpy().astype(np.float64)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("Nonfinite placement predictions")
    return probabilities


def proper_scores(probabilities: np.ndarray, ranks: np.ndarray) -> dict[str, np.ndarray]:
    if (probabilities.shape != (len(ranks), 4) or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-9)):
        raise ValueError("Invalid rank probability array")
    labels = np.eye(4, dtype=np.float64)[ranks]
    return {"nll": -np.log(np.maximum(probabilities[np.arange(len(ranks)), ranks], np.finfo(np.float64).tiny)),
            "brier": np.square(probabilities - labels).sum(axis=1),
            "utility_squared_error": np.square(probabilities @ UTILITY - UTILITY[ranks])}


def weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    if not len(values) or weights.sum() <= 0:
        raise ValueError("Empty weighted sample")
    return float(np.dot(values, weights) / weights.sum())


def calibration_stratum(values: np.ndarray, labels: np.ndarray, weights: np.ndarray,
                         groups: np.ndarray) -> dict[str, Any]:
    if not len(values):
        return {"passed": False, "unique_games": 0, "supported_mass": 0., "bins": [],
                "weighted_absolute_bias": None, "max_supported_bin_bias": None,
                "reason": "empty stratum"}
    indices = np.clip(np.searchsorted(UTILITY_BINS, values, side="right") - 1, 0, 4)
    total = float(weights.sum())
    supported_weight = 0.
    absolute = 0.
    maximum = 0.
    bins = []
    for index in range(5):
        rows = indices == index
        count = int(rows.sum())
        unique = len(np.unique(groups[rows]))
        mass = float(weights[rows].sum())
        supported = unique >= CALIBRATION["minimum_bin_unique_games"]
        predicted = weighted_average(values[rows], weights[rows]) if count else None
        observed = weighted_average(labels[rows], weights[rows]) if count else None
        bias = predicted - observed if count else None
        if supported:
            supported_weight += mass
            absolute += mass * abs(bias)
            maximum = max(maximum, abs(bias))
        bins.append({"index": index, "lower": float(UTILITY_BINS[index]), "upper": float(UTILITY_BINS[index + 1]),
            "right_closed": index == 4, "rows": count, "unique_games": unique,
            "weight": mass, "mass": mass / total, "supported": supported,
            "mean_prediction": predicted, "mean_outcome": observed, "bias": bias})
    coverage = supported_weight / total
    error = absolute / supported_weight if supported_weight else None
    unique = len(np.unique(groups))
    passed = (unique >= CALIBRATION["minimum_unique_games"] and coverage >= CALIBRATION["minimum_supported_mass"]
        and error is not None and error <= CALIBRATION["maximum_weighted_absolute_bias"]
        and maximum <= CALIBRATION["maximum_supported_bin_bias"])
    return {"passed": bool(passed), "rows": len(values), "unique_games": unique,
        "supported_mass": coverage, "weighted_absolute_bias": error,
        "max_supported_bin_bias": maximum if supported_weight else None,
        "overall_bias": weighted_average(values - labels, weights), "bins": bins}


def classwise_calibration(probabilities: np.ndarray, ranks: np.ndarray,
                          weights: np.ndarray, groups: np.ndarray) -> list[dict[str, Any]]:
    result = []
    edges = np.linspace(0, 1, 11)
    for rank in range(4):
        p, labels = probabilities[:, rank], (ranks == rank).astype(np.float64)
        indices = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 9)
        error, bins = 0., []
        for index in range(10):
            rows = indices == index
            mass = float(weights[rows].sum())
            prediction = weighted_average(p[rows], weights[rows]) if rows.any() else None
            outcome = weighted_average(labels[rows], weights[rows]) if rows.any() else None
            bias = prediction - outcome if rows.any() else None
            if rows.any():
                error += mass * abs(bias)
            bins.append({"index": index, "lower": float(edges[index]), "upper": float(edges[index + 1]),
                "rows": int(rows.sum()), "unique_games": len(np.unique(groups[rows])),
                "mass": mass / float(weights.sum()), "mean_prediction": prediction,
                "mean_outcome": outcome, "bias": bias})
        result.append({"rank": rank + 1, "weighted_ece": error / float(weights.sum()), "bins": bins})
    return result


def calibration_report(probabilities: np.ndarray, baseline: np.ndarray, ranks: np.ndarray,
                       weights: np.ndarray, groups: np.ndarray, stages: np.ndarray) -> dict[str, Any]:
    model_scores, base_scores = proper_scores(probabilities, ranks), proper_scores(baseline, ranks)
    scores = {name: weighted_average(value, weights) for name, value in model_scores.items()}
    base = {name: weighted_average(value, weights) for name, value in base_scores.items()}
    utility, labels = probabilities @ UTILITY, UTILITY[ranks]
    strata = {}
    for index, name in enumerate(("overall", "early", "middle", "late")):
        rows = np.ones(len(ranks), dtype=bool) if index == 0 else stages == index - 1
        strata[name] = calibration_stratum(utility[rows], labels[rows], weights[rows], groups[rows])
        if rows.any():
            strata[name]["proper_scores"] = {key: weighted_average(value[rows], weights[rows]) for key, value in model_scores.items()}
            strata[name]["baseline_proper_scores"] = {key: weighted_average(value[rows], weights[rows]) for key, value in base_scores.items()}
            strata[name]["classwise_calibration"] = classwise_calibration(probabilities[rows], ranks[rows], weights[rows], groups[rows])
    proper_pass = scores["nll"] <= base["nll"] and scores["brier"] <= base["brier"]
    return {"passed": bool(proper_pass and all(value["passed"] for value in strata.values())),
        "proper_score_gate_passed": bool(proper_pass), "proper_scores": scores,
        "baseline_proper_scores": base, "thresholds": CALIBRATION, "strata": strata,
        "classwise_calibration": classwise_calibration(probabilities, ranks, weights, groups),
        "baseline": "Unfitted Plackett-Luce(score/10000), exact 24-permutation rank marginals",
        "weighting": "Every match contributes unit total weight; every hand state has 1/N_hand_states",
        "uncertainty_use": "Descriptive only; point-estimate gate fixed before fitting"}


def bootstrap_report(probabilities: np.ndarray, baseline: np.ndarray, ranks: np.ndarray,
                     weights: np.ndarray, groups: np.ndarray, stages: np.ndarray,
                     gate: dict[str, Any], deadline: float, repetitions: int = 1000) -> dict[str, Any]:
    """Resample complete games and retain their within-game state weights."""
    unique, group_indices = np.unique(groups, return_inverse=True)
    count = len(unique)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.multinomial(count, np.full(count, 1 / count), size=repetitions).astype(np.float64)
    _deadline(deadline)
    def sums(row_values):
        return np.bincount(group_indices, weights=row_values, minlength=count)
    denominators = draws @ sums(weights)
    model, base = proper_scores(probabilities, ranks), proper_scores(baseline, ranks)
    samples = {}
    for name in model:
        samples[name] = (draws @ sums(model[name] * weights)) / denominators
        samples[f"baseline_{name}"] = (draws @ sums(base[name] * weights)) / denominators
        samples[f"difference_{name}"] = samples[name] - samples[f"baseline_{name}"]
    values, labels = probabilities @ UTILITY, UTILITY[ranks]
    bins = np.clip(np.searchsorted(UTILITY_BINS, values, side="right") - 1, 0, 4)
    for stage_index, stage_name in enumerate(("overall", "early", "middle", "late")):
        mask = np.ones(len(ranks), dtype=bool) if stage_index == 0 else stages == stage_index - 1
        supported = [entry["index"] for entry in gate["strata"][stage_name]["bins"] if entry["supported"]]
        absolute, mass = np.zeros(repetitions), np.zeros(repetitions)
        for index in supported:
            chosen = mask & (bins == index)
            absolute += np.abs(draws @ sums(weights * (values - labels) * chosen))
            mass += draws @ sums(weights * chosen)
        if supported and np.all(mass > 0):
            samples[f"{stage_name}_utility_calibration_error"] = absolute / mass
        # Classwise ECE is descriptive, with all fixed probability bins.
        stage_mass = draws @ sums(weights * mask)
        if np.all(stage_mass > 0):
            for rank in range(4):
                probability = probabilities[:, rank]
                probability_bin = np.clip((probability * 10).astype(np.int64), 0, 9)
                absolute_class = np.zeros(repetitions)
                for index in range(10):
                    selected = mask & (probability_bin == index)
                    absolute_class += np.abs(draws @ sums(weights * (probability - (ranks == rank)) * selected))
                samples[f"{stage_name}_rank_{rank + 1}_ece"] = absolute_class / stage_mass
        _deadline(deadline)
    return {"method": "Whole-game multinomial percentile bootstrap; state weights stay 1/N_hand_states",
        "repetitions": repetitions, "seed": BOOTSTRAP_SEED, "unique_games": count,
        "coverage": .95, "supported_bins": "Frozen at full-development gate support; no bootstrap gate selection",
        "intervals": {name: {"lower": float(np.quantile(value, .025)), "upper": float(np.quantile(value, .975))}
                      for name, value in samples.items()}}


def hand_targets(dataset: PublicDataset, predictions: np.ndarray, folds: dict[str, int],
                 predictor_ids: dict[str, str], gate: dict[str, Any]) -> dict[str, Any]:
    """Both potential boundaries use one held-out predictor per training game."""
    if predictions.shape != (len(dataset.y), 4) or set(folds) != set(dataset.splits["training"]):
        raise ValueError("Prediction shape or cross-fit assignment mismatch")
    proper_scores(predictions, dataset.y)
    values = predictions @ UTILITY
    matches = {}
    for group, name in enumerate(dataset.match_ids):
        match = dataset.matches[name]
        rows = np.flatnonzero(dataset.groups == group)
        hand_ids = match["hand_ids"]
        if len(rows) != len(hand_ids) or hand_ids != list(range(1, len(rows) + 1)):
            raise ValueError("Missing explicit hand boundaries")
        fold = folds[name] if match["split"] == "training" else None
        predictor = predictor_ids[f"fold-{fold}" if fold is not None else "final"]
        potentials = values[rows]
        deltas, hands = {}, {}
        for index, ordinal in enumerate(hand_ids):
            terminal = index + 1 == len(hand_ids)
            current = float(potentials[index])
            following = float(match["terminal_utility"] if terminal else potentials[index + 1])
            target = following - current
            if not -3 - 1e-12 <= target <= 3 + 1e-12:
                raise ValueError("Hand delta outside utility difference range")
            deltas[str(ordinal)] = target
            hands[str(ordinal)] = {"current_value": current, "next_value": following,
                "target": target, "terminal_next": terminal, "predictor_sha256": predictor}
        matches[name] = {"source_role": match["source_role"], "split": match["split"],
            "explorer_seat": match["explorer_seat"], "predictor_sha256": predictor,
            "fold": fold, "hand_deltas": deltas, "hands": hands}
    return {"schema": TARGET_SCHEMA, "target": "one_hand_placement_potential_difference",
        "corpus_manifest_sha256": dataset.manifest_sha256,
        "calibration_gate_passed": bool(gate["passed"]), "gate": gate,
        "eligible_for_delta_training": bool(gate["passed"]),
        "folds": folds, "crossfit_assignment_sha256": digest(folds),
        "splits": dataset.splits, "source_target_hashes": dataset.source_hashes,
        "matches": matches, "training_scope": "Explorer only; whole-match OOF potentials",
        "terminal_rule": "Actual average-rank utility replaces next value only at true match termination",
        "limitation": "One-hand surrogate, not decision return; failed gate leaves this artifact audit-only"}


def require_delta_gate(artifact: dict[str, Any]) -> None:
    if (artifact.get("schema") != TARGET_SCHEMA or artifact.get("calibration_gate_passed") is not True
            or artifact.get("gate", {}).get("passed") is not True
            or artifact.get("eligible_for_delta_training") is not True):
        raise ValueError("Placement calibration gate failed: hand-delta arms are disabled")
