"""Fixed-budget NI planning and test direction; no real confirmation run."""

import math
from statistics import NormalDist

import pytest

from riichi_ai.statistics import noninferiority_test, plan_noninferiority


def test_planning_uses_positive_floor_even_for_zero_variance_pilot():
    plan = plan_noninferiority([0.] * 30)
    assert plan["pilot_zero_variance"]
    assert plan["planning_sd"] == plan["variance_floor_sd"]
    assert plan["confirmation_blocks"] == plan["required_blocks"] == 185
    assert plan["run_confirmation"] and plan["status"] == "ready"
    assert plan["approximate_power_at_required_blocks"] >= .8
    assert plan["assumed_true_utility_difference"] == 0


def test_planning_uses_inflated_pilot_sd_and_does_not_silently_cap_budget():
    plan = plan_noninferiority([-1., 1.] * 15)
    expected_sd = math.sqrt(30 / 29)
    assert plan["pilot_sd"] == pytest.approx(expected_sd)
    assert plan["planning_sd"] == pytest.approx(expected_sd * 1.25)
    assert plan["required_blocks"] > 256
    assert plan["confirmation_blocks"] is None and not plan["run_confirmation"]
    assert plan["status"] == "inconclusive_budget"
    assert plan["approximate_power_at_cap"] < .8


@pytest.mark.parametrize("pilot", [[-.1] * 30, [-.2] * 30])
def test_pilot_futility_at_or_below_margin_omits_confirmation(pilot):
    plan = plan_noninferiority(pilot)
    assert plan["status"] == "futility"
    assert not plan["run_confirmation"] and plan["confirmation_blocks"] is None


def test_planning_margin_and_error_assumptions_are_explicit():
    first = plan_noninferiority([0.] * 30)
    smaller_margin = plan_noninferiority([0.] * 30, margin=.05, max_blocks=1000)
    # Rounding accounts for the possible one-to-three-block difference.
    assert abs(smaller_margin["required_blocks"] - 4 * first["required_blocks"]) <= 3
    assert "normal approximation" in first["limitations"]
    assert "independent block" in first["limitations"]
    assert "not an exact" in first["limitations"]


def test_noninferiority_direction_lower_bound_and_p_value_agree():
    values = [-.5, .5] * 93
    result = noninferiority_test(values, planned_blocks=186)
    sd = math.sqrt(.25 * 186 / 185)
    se = sd / math.sqrt(186)
    expected = -NormalDist().inv_cdf(.95) * se
    assert result["valid"] and result["passed"]
    assert result["sample_sd"] == pytest.approx(sd)
    assert result["one_sided_lower_bound"] == pytest.approx(expected)
    assert result["one_sided_p_value"] == pytest.approx(1 - NormalDist().cdf(.1 / se))
    assert result["one_sided_lower_bound"] > -.1
    assert result["one_sided_p_value"] < .05
    weaker = noninferiority_test([value - .05 for value in values], planned_blocks=186)
    assert weaker["valid"] and not weaker["passed"]
    assert weaker["status"] == "not_demonstrated"
    assert weaker["one_sided_lower_bound"] < -.1
    assert "does not establish inferiority" in weaker["limitations"]


@pytest.mark.parametrize("count", [1, 184, 186])
def test_wrong_confirmation_count_never_produces_a_pass(count):
    result = noninferiority_test([.5] * count, planned_blocks=185)
    assert not result["valid"] and not result["passed"]
    assert result["status"] == "inconclusive_count"
    assert result["one_sided_lower_bound"] is None


@pytest.mark.parametrize("constant", [0., .5, -.5])
def test_zero_observed_variance_is_inconclusive_not_zero_uncertainty(constant):
    result = noninferiority_test([constant] * 185, planned_blocks=185)
    assert result["status"] == "inconclusive_zero_variance"
    assert not result["valid"] and not result["passed"]
    assert result["one_sided_lower_bound"] is None and result["one_sided_p_value"] is None


@pytest.mark.parametrize("bad", [[], [float("nan")] * 30, [float("inf")] * 30,
                                  [3.01] * 30, [True] * 30, ["0"] * 30])
def test_invalid_block_values_fail_explicitly(bad):
    with pytest.raises(ValueError):
        plan_noninferiority(bad)
    with pytest.raises(ValueError):
        noninferiority_test(bad, planned_blocks=185)


def test_too_small_pilot_and_invalid_plans_fail_explicitly():
    with pytest.raises(ValueError, match="30"):
        plan_noninferiority([0.] * 29)
    for kwargs in ({"margin": 0}, {"alpha": .5}, {"power": 1},
                   {"variance_floor_sd": 0}, {"pilot_sd_multiplier": .5},
                   {"min_blocks": 300, "max_blocks": 256}):
        with pytest.raises(ValueError):
            plan_noninferiority([0.] * 30, **kwargs)
    for planned in (184, 257, True):
        with pytest.raises(ValueError, match="planned_blocks"):
            noninferiority_test([0.] * 185, planned_blocks=planned)
