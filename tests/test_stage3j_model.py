"""Stage 3J choice invariants; synthetic checks are not playing-strength tests."""
import hashlib
import json

import numpy as np
import pytest

from riichi_ai.features import ACTION_FIELDS, GLOBAL_FEATURES, HISTORY_FIELDS, MAX_HISTORY, TILE_CHANNELS
from riichi_ai.semantics import ACTION_KINDS, SEMANTIC_ACTION_VERSION
from riichi_ai.stage3i_model import correction_choices
from riichi_ai.stage3j_model import (
    FRACTIONAL_STRENGTH, HASH_TIE_SALT, actor_feature_fingerprint, stage3j_choices,
)


def kinds(*names):
    return np.asarray([ACTION_KINDS.index(name) + 1 for name in names], dtype=np.int64)


def discard_keys(count):
    return [json.dumps([SEMANTIC_ACTION_VERSION, "DISCARD", index, False, False, []], separators=(",", ":"))
            for index in range(count)]


def encoded(count=3):
    result = {
        "tiles": np.zeros((34, TILE_CHANNELS), dtype=np.float32),
        "globals": np.zeros(GLOBAL_FEATURES, dtype=np.float32),
        "history": np.zeros((MAX_HISTORY, HISTORY_FIELDS), dtype=np.int16),
        "history_mask": np.zeros(MAX_HISTORY, dtype=np.bool_),
        "action_tokens": np.zeros((count, ACTION_FIELDS), dtype=np.int16),
        "candidate_efficiency": np.zeros((count, 4), dtype=np.float32),
    }
    result["action_tokens"][:, 0] = ACTION_KINDS.index("DISCARD") + 1
    result["action_tokens"][:, 1] = np.arange(count) + 1
    return result


def choice(logits=(1., 1., 0.), residuals=(0., 2., 100.), q=(.49, .49, .02), *, mode="learned_tie", **kwargs):
    return stage3j_choices(np.asarray(logits, dtype=np.float32),
                          None if residuals is None else np.asarray(residuals, dtype=np.float64),
                          np.asarray(q, dtype=np.float64), kinds(*(["DISCARD"] * len(logits))), mode=mode, **kwargs)


def test_learned_tie_ignores_even_a_huge_lower_logit_advantage_and_keeps_exact_sets():
    decision = choice()
    assert decision["selected_index"] == 1
    assert decision["raw_reference_index"] == decision["wrapped_reference_index"] == 0
    assert decision["exact_tie_mask"].tolist() == [True, True, False]
    assert decision["eligible_tie_mask"].tolist() == [True, True, False]
    assert decision["change_classification"] == "exact_tie"
    assert decision["selected_reference_gap"] == 0
    assert np.isneginf(decision["scores"][2])
    assert not decision["wrapper_active"] and not decision["wrapper_applied"]


def test_exact_tie_includes_zero_sign_but_not_one_float32_ulp():
    low = np.nextafter(np.float32(1.), np.float32(0.))
    exact = choice(logits=(1., low, 1.), residuals=(0., 1000., 1.))
    assert exact["exact_tie_mask"].tolist() == [True, False, True]
    assert exact["selected_index"] == 2
    zeros = choice(logits=(-0., 0., -1.), residuals=(0., 1., 100.))
    assert zeros["selected_index"] == 1 and zeros["change_classification"] == "exact_tie"


def test_eligibility_is_inclusive_at_threshold_and_always_retains_original_argmax():
    decision = choice(logits=(1., 1., 1.), residuals=(0., 1., 100.), q=(.961, .02, .019))
    assert decision["eligible_tie_mask"].tolist() == [True, True, False]
    assert decision["selected_index"] == 1
    kept = choice(logits=(1., 1., 0.), residuals=(5., 1., 100.), q=(0., .5, .5))
    assert kept["eligible"][0] and kept["selected_index"] == 0
    assert kept["change_classification"] == "unchanged"


def test_residual_ties_use_original_canonical_first_index_including_negative_residuals():
    decision = choice(logits=(1., 1., 1.), residuals=(-3., -3., -5.))
    assert decision["selected_index"] == 0
    assert decision["change_classification"] == "unchanged"


@pytest.mark.parametrize("mode", ["reference_wrapped", "learned_tie", "fractional", "hash_tie"])
def test_every_mode_preserves_first_legal_win_even_if_ineligible_in_supplied_q(mode):
    ids = kinds("DISCARD", "RON", "TSUMO")
    extra = {}
    if mode == "hash_tie":
        keys = [json.dumps([SEMANTIC_ACTION_VERSION, name, None, False, None, []], separators=(",", ":"))
                for name in ("DISCARD", "RON", "TSUMO")]
        extra = {"actor_fingerprint": "a" * 64, "semantic_keys": keys}
    decision = stage3j_choices(np.asarray([3., 0., -1.], dtype=np.float32),
                              np.asarray([0., 0., 100.]), np.asarray([.5, 0., .5]), ids, mode=mode, **extra)
    assert decision["selected_index"] == decision["wrapped_reference_index"] == 1
    assert decision["wrapper_active"] and decision["wrapper_applied"]
    assert decision["change_classification"] == "unchanged"
    # This is the raw-logit gap of a mandatory winning action, not an override.
    assert decision["selected_reference_gap"] == 3.


def test_reference_never_uses_the_residual_and_supports_forced_decisions():
    reference = choice(mode="reference_wrapped")
    assert reference["selected_index"] == 0 and reference["strength"] == 0.
    forced = choice(logits=(0.,), residuals=None, q=(1.,), mode="reference_wrapped")
    assert forced["selected_index"] == 0
    assert forced["exact_tie_mask"].tolist() == [True]


def test_fractional_mode_exactly_reuses_float64_correction_and_reports_positive_margin():
    logits = np.asarray([0., -.01, -.02], dtype=np.float32)
    residuals = np.asarray([0., .2, .3], dtype=np.float32)
    q = np.asarray([.49, .49, .02], dtype=np.float64)
    ids = kinds("DISCARD", "DISCARD", "DISCARD")
    actual = stage3j_choices(logits, residuals, q, ids, mode="fractional")
    expected = correction_choices(logits, residuals, q, ids, .25, wrapped=True)
    for name, value in expected.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(actual[name], value)
        else:
            assert actual[name] == value
    assert actual["strength"] == FRACTIONAL_STRENGTH == .25
    assert actual["selected_index"] == 2 and actual["change_classification"] == "positive_margin"
    assert actual["selected_reference_gap"] == -float(logits[2])
    assert actual["scores"].dtype == np.float64


def test_prior_subtraction_is_double_before_clipping_even_at_finite_float32_extremes():
    big = np.finfo(np.float32).max
    actual = choice(logits=(big, -big), residuals=(0., 33.), q=(.5, .5), mode="fractional")
    assert actual["prior"].tolist() == [0., -8.]
    assert actual["selected_index"] == 1
    assert actual["selected_reference_gap"] == 2 * float(big)


def test_hash_tie_matches_independent_bytes_and_never_uses_lower_or_ineligible_candidates():
    fingerprint = actor_feature_fingerprint(encoded())
    keys = discard_keys(3)
    expected = [hashlib.sha256(json.dumps([HASH_TIE_SALT, fingerprint, key], separators=(",", ":")).encode()).hexdigest()
                for key in keys]
    actual = choice(logits=(1., 1., 1.), residuals=None, q=(.49, .49, .02), mode="hash_tie",
                    actor_fingerprint=fingerprint, semantic_keys=keys)
    assert actual["tie_hashes"] == expected
    assert actual["selected_index"] == min(range(3), key=lambda index: (expected[index], index))
    again = choice(logits=(1., 1., 1.), residuals=(1e20, -1e20, 5.), q=(.49, .49, .02), mode="hash_tie",
                   actor_fingerprint=fingerprint, semantic_keys=keys)
    assert actual["selected_index"] == again["selected_index"]
    excluded = choice(logits=(1., 1., 0.), residuals=None, q=(.97, .01, .02), mode="hash_tie",
                      actor_fingerprint=fingerprint, semantic_keys=keys)
    assert excluded["selected_index"] == 0
    assert excluded["tie_hashes"] == [expected[0], None, None]


def test_hash_collision_uses_canonical_first_candidate(monkeypatch):
    class Collision:
        def hexdigest(self):
            return "b" * 64
    monkeypatch.setattr("riichi_ai.stage3j_model.hashlib.sha256", lambda _: Collision())
    actual = choice(mode="hash_tie", actor_fingerprint="a" * 64, semantic_keys=discard_keys(3))
    assert actual["selected_index"] == 0


def test_fingerprint_is_stable_over_dict_and_memory_layout_but_sensitive_to_visible_features():
    row = encoded()
    expected = actor_feature_fingerprint(row)
    reverse_fortran = {name: np.asfortranarray(row[name]) for name in reversed(row)}
    assert actor_feature_fingerprint(reverse_fortran) == expected
    row["globals"][0] = .5
    assert actor_feature_fingerprint(row) != expected


@pytest.mark.parametrize("field", ["wall", "seed", "player_view_sha256", "terminal_utility", "physical_tokens", "hand_id"])
def test_fingerprint_rejects_extra_auditor_or_provenance_fields(field):
    row = encoded()
    row[field] = np.asarray([0])
    with pytest.raises(ValueError, match="actor feature fields only"):
        actor_feature_fingerprint(row)


@pytest.mark.parametrize("change", ["missing", "float64", "int64", "shape", "nan", "empty", "list"])
def test_fingerprint_rejects_incorrect_schema_without_silent_conversion(change):
    row = encoded()
    if change == "missing":
        del row["history_mask"]
    elif change == "float64":
        row["tiles"] = row["tiles"].astype(np.float64)
    elif change == "int64":
        row["history"] = row["history"].astype(np.int64)
    elif change == "shape":
        row["candidate_efficiency"] = row["candidate_efficiency"][:2]
    elif change == "nan":
        row["globals"][0] = np.nan
    elif change == "empty":
        row["action_tokens"] = row["action_tokens"][:0]
    else:
        row["globals"] = row["globals"].tolist()
    with pytest.raises(ValueError):
        actor_feature_fingerprint(row)


@pytest.mark.parametrize("change", ["float64", "list", "nan", "inf", "empty", "rank2"])
def test_choices_require_original_float32_not_approximate_reconstruction(change):
    logits = np.asarray([1., 1.], dtype=np.float32)
    if change == "float64":
        logits = np.asarray([1., 1. + 1e-12], dtype=np.float64)
    elif change == "list":
        logits = logits.tolist()
    elif change in ("nan", "inf"):
        logits[0] = np.nan if change == "nan" else np.inf
    elif change == "empty":
        logits = logits[:0]
    else:
        logits = logits.reshape(1, 2)
    with pytest.raises(ValueError, match="original finite.*float32"):
        stage3j_choices(logits, np.zeros(2), np.asarray([.5, .5]), kinds("DISCARD", "DISCARD"), mode="learned_tie")


@pytest.mark.parametrize("change", ["bad_q", "negative_q", "nan_residual", "shape_residual", "missing_residual",
                                     "float_kind", "unknown_kind", "mode", "fingerprint_learned"])
def test_choices_fail_closed_for_malformed_inputs(change):
    args = dict(logits=np.ones(2, dtype=np.float32), residuals=np.zeros(2), probabilities=np.asarray([.5, .5]),
                kinds=kinds("DISCARD", "DISCARD"), mode="learned_tie")
    if change == "bad_q":
        args["probabilities"] = np.asarray([.5, .6])
    elif change == "negative_q":
        args["probabilities"] = np.asarray([-1., 2.])
    elif change == "nan_residual":
        args["residuals"] = np.asarray([np.nan, 0.])
    elif change == "shape_residual":
        args["residuals"] = np.zeros(3)
    elif change == "missing_residual":
        args["residuals"] = None
    elif change == "float_kind":
        args["kinds"] = np.ones(2)
    elif change == "unknown_kind":
        args["kinds"] = np.asarray([1, 99])
    elif change == "mode":
        args["mode"] = "adaptive_lambda"
    else:
        args["actor_fingerprint"] = "a" * 64
    with pytest.raises(ValueError):
        stage3j_choices(**args)


@pytest.mark.parametrize("change", ["missing_hash", "uppercase_hash", "bad_hash", "missing_keys", "physical_token",
                                     "duplicates", "wrong_kind", "reversed", "spaced_json"])
def test_hash_control_rejects_noncanonical_or_opaque_inputs(change):
    extra = {"actor_fingerprint": "a" * 64, "semantic_keys": discard_keys(3)}
    if change == "missing_hash":
        del extra["actor_fingerprint"]
    elif change == "uppercase_hash":
        extra["actor_fingerprint"] = "A" * 64
    elif change == "bad_hash":
        extra["actor_fingerprint"] = "hidden-state"
    elif change == "missing_keys":
        del extra["semantic_keys"]
    elif change == "physical_token":
        extra["semantic_keys"][0] = "DISCARD:42"
    elif change == "duplicates":
        extra["semantic_keys"][1] = extra["semantic_keys"][0]
    elif change == "wrong_kind":
        extra["semantic_keys"][0] = extra["semantic_keys"][0].replace("DISCARD", "PON")
    elif change == "reversed":
        extra["semantic_keys"] = extra["semantic_keys"][::-1]
    else:
        extra["semantic_keys"][0] = json.dumps(json.loads(extra["semantic_keys"][0]))
    with pytest.raises(ValueError):
        choice(mode="hash_tie", **extra)
