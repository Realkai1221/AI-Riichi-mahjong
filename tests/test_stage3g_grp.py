"""Placement leakage, weighting, calibration and hand-boundary contract tests."""
from copy import deepcopy
import hashlib
import json
import time

import numpy as np
import pytest

from riichi_ai.stage3g_grp import (CALIBRATION, FEATURE_NAMES, PUBLIC_TARGET_SCHEMA,
    PublicDataset, UTILITY, assign_folds, bootstrap_report, calibration_report,
    calibration_stratum, canonical, fit_predictor, hand_targets, load_public_dataset,
    predict, proper_scores, public_features, require_delta_gate, score_baseline, stage)


def hand(ordinal=1, **kwargs):
    return dict(hand_id=ordinal, bakaze="E", kyoku=1, honba=0, kyotaku=0,
                oya=0, scores=[25000] * 4, **kwargs)


def dataset():
    ids = [f"match-{i:06d}" for i in range(6)]
    matches = {name: {"split": "training" if i < 5 else "development",
        "source_role": "offline-training" if i < 5 else "development-only",
        "explorer_seat": i % 4, "hand_ids": [1, 2, 3],
        "terminal_utility": float(UTILITY[i % 4])} for i, name in enumerate(ids)}
    return PublicDataset(np.stack([public_features(hand(j + 1), i % 4) for i in range(6) for j in range(3)]),
        np.repeat(np.arange(6) % 4, 3), np.full(18, 1 / 3), np.repeat(np.arange(6), 3),
        np.tile(np.arange(3), 6), np.full((18, 4), .25), ids, matches,
        {"training": ids[:5], "development": ids[5:]}, "manifesthash", {})


def fixture_corpus(tmp_path):
    data = dataset()
    reports = []
    (tmp_path / "targets").mkdir()
    for i, name in enumerate(data.match_ids):
        role = data.matches[name]["source_role"]
        target = {"schema": PUBLIC_TARGET_SCHEMA, "source_role": role,
            "terminal_result_verified": True, "ranks": [1, 2, 3, 4], "utilities": UTILITY.tolist(),
            "hand_starts": [hand(1), dict(hand(2), honba=1), dict(hand(3), kyoku=2, oya=1)]}
        raw = canonical(target)
        (tmp_path / "targets" / f"{name}.json").write_bytes(raw)
        reports.append({"match_id": name, "role": data.matches[name]["split"], "source_role": role,
            "completed": True, "failure": None, "terminal_result_verified": True, "hands": 3,
            "seat_behaviors": ["explorer", "reference", "shanten_v2", "defense_v2"],
            "ranks": target["ranks"], "utilities": target["utilities"],
            "artifacts": {f"targets/{name}.json": {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}}})
    manifest = {"schema": "riichi-ai-stage3g-corpus-v1", "complete": True,
        "source_role": "offline-learning-corpus", "splits": data.splits, "matches": reports}
    (tmp_path / "manifest.json").write_bytes(canonical(manifest))
    return manifest


def test_public_features_ignore_future_labels_behavior_and_auditor_content():
    base = hand()
    changed = dict(base, ranks=[4, 3, 2, 1], utilities=[-1.5, -.5, .5, 1.5],
        final_scores=[0, 0, 0, 100000], policy_id="secret", hidden_hands="secret",
        canonical_events=[{"future": True}], terminal_result_verified=True)
    np.testing.assert_array_equal(public_features(base, 2), public_features(changed, 2))
    assert len(public_features(base, 0)) == len(FEATURE_NAMES)


def test_public_features_show_absolute_seat_tie_priority_and_continuation():
    assert not np.array_equal(public_features(hand(), 0), public_features(hand(), 1))
    assert not np.array_equal(public_features(hand(), 0), public_features(dict(hand(2), honba=1), 0))
    assert public_features(hand(), 0)[-1] == pytest.approx(.05)


@pytest.mark.parametrize("field,value", [("hand_id", 0), ("honba", -1), ("kyotaku", True),
    ("oya", 4), ("kyoku", 0), ("scores", [25000] * 3), ("scores", [25000.] * 4), ("bakaze", "X")])
def test_invalid_public_fields_rejected(field, value):
    item = dict(hand(), **{field: value})
    with pytest.raises(ValueError):
        public_features(item, 0)


@pytest.mark.parametrize("wind,kyoku,wanted", [("E", 1, 0), ("E", 2, 0), ("E", 3, 1),
    ("E", 4, 1), ("S", 1, 1), ("S", 2, 1), ("S", 3, 2), ("W", 1, 2)])
def test_stage_definition(wind, kyoku, wanted):
    assert stage(dict(hand(), bakaze=wind, kyoku=kyoku)) == wanted


def test_baseline_uniform_at_equal_scores_and_monotone_at_high_score():
    np.testing.assert_allclose(score_baseline(hand(), 0), .25, atol=1e-12)
    ahead = dict(hand(), scores=[55000, 15000, 15000, 15000])
    assert score_baseline(ahead, 0)[0] > .8
    assert score_baseline(ahead, 0).sum() == pytest.approx(1)


def test_folds_balanced_order_invariant_and_whole_match():
    names = [f"game-{i}" for i in range(3200)]
    folds = assign_folds(names)
    assert folds == assign_folds(list(reversed(names)))
    assert [list(folds.values()).count(i) for i in range(5)] == [640] * 5


@pytest.mark.parametrize("names", [[], ["x"] * 5, ["a", "b", "c", "d"]])
def test_bad_fold_membership_rejected(names):
    with pytest.raises(ValueError):
        assign_folds(names)


def test_public_loader_opens_only_public_summary_and_manifest(tmp_path, monkeypatch):
    fixture_corpus(tmp_path)
    from pathlib import Path
    original = Path.read_bytes
    opened = []
    def guarded(path):
        opened.append(path)
        assert path.name == "manifest.json" or path.parent.name == "targets"
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", guarded)
    loaded = load_public_dataset(tmp_path, expected_training=5, expected_development=1)
    assert len(opened) == 7
    assert loaded.x.shape == (18, len(FEATURE_NAMES))
    for group in range(6):
        assert loaded.weights[loaded.groups == group].sum() == pytest.approx(1)


def test_label_mutation_does_not_change_features_or_folds(tmp_path):
    manifest = fixture_corpus(tmp_path)
    first = load_public_dataset(tmp_path, expected_training=5, expected_development=1)
    for report in manifest["matches"]:
        path = tmp_path / "targets" / f"{report['match_id']}.json"
        item = json.loads(path.read_text())
        item["ranks"] = report["ranks"] = [4, 3, 2, 1]
        item["utilities"] = report["utilities"] = UTILITY[::-1].tolist()
        raw = canonical(item)
        path.write_bytes(raw)
        report["artifacts"][f"targets/{report['match_id']}.json"] = {
            "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    (tmp_path / "manifest.json").write_bytes(canonical(manifest))
    second = load_public_dataset(tmp_path, expected_training=5, expected_development=1)
    np.testing.assert_array_equal(first.x, second.x)
    assert not np.array_equal(first.y, second.y)
    assert assign_folds(first.splits["training"]) == assign_folds(second.splits["training"])


@pytest.mark.parametrize("change", ["stage3f", "overlap", "incomplete", "bad_role", "artifact"])
def test_corpus_rejects_leakage_roles_partial_and_identity(tmp_path, change):
    manifest = fixture_corpus(tmp_path)
    if change == "stage3f":
        manifest["schema"] = "riichi-ai-stage3f-corpus-v1"
    elif change == "overlap":
        manifest["splits"]["development"] = manifest["splits"]["training"][:1]
    elif change == "incomplete":
        manifest["complete"] = False
    elif change == "bad_role":
        manifest["matches"][0]["source_role"] = "development-only"
    else:
        manifest["matches"][0]["artifacts"]["targets/match-000000.json"]["sha256"] = "wrong"
    (tmp_path / "manifest.json").write_bytes(canonical(manifest))
    with pytest.raises(ValueError):
        load_public_dataset(tmp_path, expected_training=5, expected_development=1)


def calibration_fixture():
    groups = np.repeat(np.arange(120), 3)
    ranks = np.repeat(np.arange(120) % 4, 3)
    weights = np.full(360, 1 / 3)
    probabilities = np.full((360, 4), .25)
    stages = np.tile(np.arange(3), 120)
    return probabilities, probabilities.copy(), ranks, weights, groups, stages


def test_calibration_gate_passes_supported_calibrated_baseline_tie():
    result = calibration_report(*calibration_fixture())
    assert result["passed"]
    assert result["proper_score_gate_passed"]
    assert all(item["unique_games"] == 120 for item in result["strata"].values())
    assert result["strata"]["early"]["supported_mass"] == pytest.approx(1)


def test_calibration_gate_rejects_mean_utility_bias():
    p, b, ranks, w, groups, stages = calibration_fixture()
    ranks[:] = 0
    result = calibration_report(p, b, ranks, w, groups, stages)
    assert not result["passed"]
    assert result["strata"]["overall"]["weighted_absolute_bias"] == pytest.approx(1.5)


def test_calibration_gate_rejects_missing_late_game_support():
    p, b, ranks, w, groups, stages = calibration_fixture()
    stages[:] = 0
    result = calibration_report(p, b, ranks, w, groups, stages)
    assert not result["passed"]
    assert result["strata"]["late"]["unique_games"] == 0


def test_sparse_occupied_bins_count_against_coverage():
    values = np.zeros(120)
    values[:45] = 1.2
    result = calibration_stratum(values, values.copy(), np.ones(120), np.arange(120))
    assert result["supported_mass"] == pytest.approx(75 / 120)
    assert not result["passed"]
    assert result["weighted_absolute_bias"] == 0


def test_repeated_states_cannot_fake_unique_game_support():
    result = calibration_stratum(np.zeros(500), np.zeros(500), np.ones(500), np.zeros(500))
    assert result["unique_games"] == 1
    assert result["supported_mass"] == 0
    assert not result["passed"]


def test_proper_score_failure_is_separate_from_utility_calibration():
    p, _, ranks, weights, groups, stages = calibration_fixture()
    baseline = np.eye(4)[ranks]
    result = calibration_report(p, baseline, ranks, weights, groups, stages)
    assert all(item["passed"] for item in result["strata"].values())
    assert not result["proper_score_gate_passed"]
    assert not result["passed"]


def test_targets_use_continuation_ordinals_same_fold_and_terminal_utility():
    data = dataset()
    probabilities = np.tile(np.asarray([[.25, .25, .25, .25], [.4, .3, .2, .1], [.1, .2, .3, .4]]), (6, 1))
    folds = assign_folds(data.splits["training"])
    identities = {**{f"fold-{i}": f"model{i}" for i in range(5)}, "final": "finalmodel"}
    target = hand_targets(data, probabilities, folds, identities, {"passed": True})
    require_delta_gate(target)
    for name, match in target["matches"].items():
        expected = identities[f"fold-{folds[name]}"] if name in folds else "finalmodel"
        assert match["predictor_sha256"] == expected
        assert set(match["hand_deltas"]) == {"1", "2", "3"}
        assert not match["hands"]["1"]["terminal_next"]
        assert not match["hands"]["2"]["terminal_next"]
        assert match["hands"]["3"]["terminal_next"]
        assert match["hands"]["3"]["next_value"] == data.matches[name]["terminal_utility"]
        assert sum(match["hand_deltas"].values()) == pytest.approx(data.matches[name]["terminal_utility"])
        assert len({h["predictor_sha256"] for h in match["hands"].values()}) == 1


def test_failed_gate_exports_audit_only_targets_and_refuses_training():
    data = dataset()
    target = hand_targets(data, data.baseline, assign_folds(data.splits["training"]),
        {**{f"fold-{i}": str(i) for i in range(5)}, "final": "final"}, {"passed": False})
    assert target["matches"]
    assert not target["eligible_for_delta_training"]
    with pytest.raises(ValueError, match="disabled"):
        require_delta_gate(target)


def test_fit_refuses_development_matches_before_optimization():
    data = dataset()
    with pytest.raises(ValueError, match="non-training"):
        fit_predictor(data, data.splits["development"], time.monotonic() + 30, epochs=1)


def test_fit_deadline_refuses_partial_fit():
    data = dataset()
    with pytest.raises(TimeoutError):
        fit_predictor(data, data.splits["training"], time.monotonic() - 1, epochs=1)


def test_fixed_fit_smoke_preserves_match_weighting_and_normalized_predictions():
    data = dataset()
    model, report = fit_predictor(data, data.splits["training"], time.monotonic() + 30, epochs=1)
    assert report["epochs"] == report["optimizer_steps"] == 1
    assert report["training_matches"] == 5
    assert report["rows"] == 15
    predictions = predict(model, data.x)
    np.testing.assert_allclose(predictions.sum(axis=1), 1, atol=1e-12)
    assert set(proper_scores(predictions, data.y)) == {"nll", "brier", "utility_squared_error"}


def test_bootstrap_clusters_games_and_preserves_fixed_gate_support():
    args = calibration_fixture()
    gate = calibration_report(*args)
    report = bootstrap_report(*args, gate, time.monotonic() + 30, repetitions=12)
    assert report["unique_games"] == 120
    assert report["repetitions"] == 12
    assert report["intervals"]["difference_nll"] == {"lower": 0., "upper": 0.}
    assert "late_utility_calibration_error" in report["intervals"]
