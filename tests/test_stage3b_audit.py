"""Fault injection for independent saved-evidence arithmetic and scope checks."""

from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("stage3b_independent_audit", ROOT / "scripts/stage3b_audit.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def row_fixture():
    selected = dict(id="row0", stratum="PON", action_kinds=["PON", "PASS"],
        semantic_keys=["pon-key", "pass-key"], target_index=0, target_kind="PON",
        policy_id="preserved-behavior", selected_index=1)
    row = {key: value for key, value in selected.items() if key not in ("policy_id", "selected_index")}
    row.update(recorded_policy_id=selected["policy_id"], recorded_selected_index=1,
        logits=[-1000., 0.], gaps_to_best_logit=[1000., 0.], logit_range=1000.,
        candidate_mask=[True, True], mask_all_saved_candidates_legal=True,
        by_kind={kind: dict(logits=audit.stats([z]), gap_to_best=audit.stats([-z]))
                 for kind, z in zip(selected["action_kinds"], [-1000., 0.])}, variants={})
    for option in audit.GRID:
        expected = audit.distribution(row["logits"], option["temperature"], option["epsilon"])
        p = expected["probabilities"]
        l1 = abs(p[0]) + abs(p[1] - 1)
        expected.pop("base_probabilities")
        expected.update(entropy_nats=-sum(value * math.log(value) for value in p if value),
            maximum_probability=max(p), legal_candidates=2, zero_probability_candidates=sum(value == 0 for value in p),
            underflow_with_finite_log_probability=sum(value == 0 for value in p),
            minimum_positive_probability=min(value for value in p if value),
            l1_from_same_checkpoint_unchanged=l1, total_variation_from_same_checkpoint_unchanged=l1 / 2,
            unchanged_argmax_index=1, unchanged_argmax_probability=p[1], argmax_index_preserved=True,
            positive_probabilities_below_2_minus53=sum(0 < value < 2**-53 for value in p),
            positive_probability_zero_width_float64_cdf_increments=0, call_kan_mass=p[0],
            by_kind={kind: dict(candidates=1, mass=value, zero_probability_candidates=int(value == 0))
                     for kind, value in zip(selected["action_kinds"], p)})
        row["variants"][option["name"]] = expected
    return selected, row


def gradient_fixture(row):
    result = dict(id=row["id"], target_index=0, target_kind="PON", variants={})
    for option in audit.GRID:
        dist = audit.distribution(row["logits"], option["temperature"], option["epsilon"])
        gradient = [dist["mixture_gradient_attenuation"][0] * ((i == 0) - p) / option["temperature"]
                    for i, p in enumerate(dist["base_probabilities"])]
        norm = 0. if option["epsilon"] else 4.
        result["variants"][option["name"]] = dict(
            target_probability=dist["probabilities"][0], target_log_probability=dist["log_probabilities"][0],
            target_base_log_probability=dist["base_log_probabilities"][0],
            target_mixture_gradient_attenuation=dist["mixture_gradient_attenuation"][0],
            logit_gradients=gradient, normalization_float64_logit_gradients=gradient,
            logit_gradient_l2=math.hypot(*gradient), normalization_float64_logit_gradient_l2=math.hypot(*gradient),
            logit_gradient_max_abs=max(abs(x) for x in gradient), parameter_gradient_l2=norm,
            gradients_finite=True, parameter_gradient_exactly_zero=norm == 0,
            parameter_gradient_below_1e_minus12=norm < 1e-12,
            parameters={"actor.weight": dict(connected=True, l2_norm=norm, max_abs=norm,
                                             nonzero_elements=int(norm > 0), elements=1)},
            parameter_groups={"actor": dict(l2_norm=norm, max_abs=norm,
                                            nonzero_elements=int(norm > 0), elements=1)})
    return result


def test_underflow_retains_finite_log_probability_and_zero_mixture_gradient():
    base = audit.distribution([-1000., 0.])
    mixed = audit.distribution([-1000., 0.], epsilon=.05)
    assert base["probabilities"] == [0., 1.]
    assert base["log_probabilities"] == [-1000., 0.]
    assert base["mixture_gradient_attenuation"] == [1., 1.]
    assert mixed["probabilities"] == [.025, .975]
    assert mixed["mixture_gradient_attenuation"][0] == 0


def test_saved_row_and_gradient_fixture_valid():
    selected, row = row_fixture()
    audit.audit_row(row, selected, audit.Checks())
    audit.audit_gradient(gradient_fixture(row), row, audit.Checks())


@pytest.mark.parametrize("mutation", [
    lambda row: row["candidate_mask"].__setitem__(0, False),
    lambda row: row.__setitem__("recorded_policy_id", "different-opponent"),
    lambda row: row["variants"]["uniform05"]["probabilities"].__setitem__(0, .2),
    lambda row: row["variants"]["unchanged"].__setitem__("zero_probability_candidates", 0),
    lambda row: row["variants"]["temperature100"].__setitem__("call_kan_mass", .1),
    lambda row: row["variants"]["temperature10"].__setitem__("positive_probability_zero_width_float64_cdf_increments", 1),
])
def test_bad_row_evidence_rejected(mutation):
    selected, row = row_fixture()
    mutation(row)
    with pytest.raises(ValueError):
        audit.audit_row(row, selected, audit.Checks())


@pytest.mark.parametrize("mutation", [
    lambda probe: probe["variants"]["uniform05"].__setitem__("logit_gradients", [1., -1.]),
    lambda probe: probe["variants"]["uniform05"].__setitem__("normalization_float64_logit_gradients", [1., -1.]),
    lambda probe: probe["variants"]["unchanged"].__setitem__("parameter_gradient_l2", 400.),
    lambda probe: probe["variants"]["unchanged"]["parameter_groups"]["actor"].__setitem__("l2_norm", 400.),
    lambda probe: probe["variants"]["unchanged"].__setitem__("parameter_gradient_exactly_zero", True),
    lambda probe: probe.__setitem__("target_index", 1),
])
def test_wrong_gradient_rule_or_aggregation_rejected(mutation):
    _, row = row_fixture()
    probe = gradient_fixture(row)
    mutation(probe)
    with pytest.raises(ValueError):
        audit.audit_gradient(probe, row, audit.Checks())


def test_nonfinite_gradient_is_not_passing_evidence():
    _, row = row_fixture()
    probe = gradient_fixture(row)
    probe["variants"]["unchanged"]["parameter_gradient_l2"] = None
    with pytest.raises(ValueError):
        audit.audit_gradient(probe, row, audit.Checks())


def test_tiny_positive_mass_cannot_be_replaced_with_consistent_zero_counters():
    selected, row = row_fixture()
    original = row["variants"]["temperature10"]
    changed = deepcopy(row["variants"]["unchanged"])
    # Preserve the correct stable log probabilities while changing the tiny
    # numerical mass and all associated summaries consistently to zero.
    changed["log_probabilities"] = original["log_probabilities"]
    changed["base_log_probabilities"] = original["base_log_probabilities"]
    row["variants"]["temperature10"] = changed
    with pytest.raises(ValueError, match="Exact float64 probability support"):
        audit.audit_row(row, selected, audit.Checks())


def test_round_robin_uses_identity_order_and_rare_stratum_priority():
    rows = [
        dict(id="d0", stratum="DISCARD", selection_key="a"),
        dict(id="p1", stratum="PON", selection_key="b"),
        dict(id="p0", stratum="PON", selection_key="a"),
        dict(id="a0", stratum="ANKAN", selection_key="z"),
    ]
    before = deepcopy(rows)
    assert [row["id"] for row in audit.round_robin(rows, 3)] == ["a0", "p0", "d0"]
    assert rows == before
