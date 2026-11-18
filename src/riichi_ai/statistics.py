"""Fixed-budget approximate non-inferiority for independent seed-block means.

The caller must freeze the candidate, opponent panel, margin, seed namespace,
and sample count before confirmation. Pilot and confirmation blocks must be
disjoint. These functions cannot establish independence from numeric deltas.
No result is an anytime-valid or finite-sample distribution-free guarantee.
"""

from __future__ import annotations

import math
from numbers import Real
from statistics import NormalDist
from typing import Sequence


NI_SCHEMA = "riichi-ai-normal-noninferiority-v1"
_NORMAL = NormalDist()
_LIMITATIONS = (
    "One-sided normal approximation using sample SD of independent block means; "
    "not an exact, distribution-free, sequential, or multiple-testing-adjusted guarantee. "
    "Requires a frozen candidate/panel and fresh confirmation blocks at a predeclared fixed count. "
    "Failure to demonstrate non-inferiority does not establish inferiority."
)


def _values(differences: Sequence[float]) -> tuple[float, ...]:
    if len(differences) == 0:
        raise ValueError("At least one independent block difference is required")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in differences):
        raise ValueError("Block differences must be real numbers")
    values = tuple(float(value) for value in differences)
    if any(not math.isfinite(value) or not -3 <= value <= 3 for value in values):
        raise ValueError("Block differences must be finite and within [-3, 3]")
    return values


def _moments(values: tuple[float, ...]) -> tuple[float, float | None]:
    mean = math.fsum(values) / len(values)
    sd = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)) if len(values) > 1 else None
    return mean, sd


def _parameters(margin: float, alpha: float, min_blocks: int, max_blocks: int) -> None:
    if not math.isfinite(margin) or not 0 < margin <= 3:
        raise ValueError("margin must be finite and within (0, 3]")
    if not math.isfinite(alpha) or not 0 < alpha < .5:
        raise ValueError("alpha must be within (0, 0.5)")
    if type(min_blocks) is not int or type(max_blocks) is not int or not 2 <= min_blocks <= max_blocks:
        raise ValueError("Require integer 2 <= min_blocks <= max_blocks")


def plan_noninferiority(
    pilot_differences: Sequence[float], *, margin: float = .1, alpha: float = .05,
    power: float = .8, variance_floor_sd: float = .5460174728387007,
    pilot_sd_multiplier: float = 1.25, min_blocks: int = 185,
    max_blocks: int = 256, minimum_pilot_blocks: int = 30,
) -> dict:
    """Plan fresh confirmation; never reduce the required count to fit a cap.

    Planning assumes true student-minus-teacher utility difference zero and
    sigma=max(Stage-1 floor, multiplier * pilot SD). The sample-size formula
    is ((z_(1-alpha)+z_power)*sigma/margin)^2, rounded up and subject to the
    minimum. A weak pilot or a required budget above the cap stops confirmation.
    """
    _parameters(margin, alpha, min_blocks, max_blocks)
    if not math.isfinite(power) or not .5 < power < 1:
        raise ValueError("power must be within (0.5, 1)")
    if not math.isfinite(variance_floor_sd) or variance_floor_sd <= 0:
        raise ValueError("variance_floor_sd must be finite and positive")
    if not math.isfinite(pilot_sd_multiplier) or pilot_sd_multiplier < 1:
        raise ValueError("pilot_sd_multiplier must be finite and at least one")
    if type(minimum_pilot_blocks) is not int or minimum_pilot_blocks < 2:
        raise ValueError("minimum_pilot_blocks must be an integer of at least two")
    values = _values(pilot_differences)
    if len(values) < minimum_pilot_blocks:
        raise ValueError(f"Need at least {minimum_pilot_blocks} development pilot blocks")
    mean, sd = _moments(values)
    assert sd is not None
    planning_sd = max(variance_floor_sd, pilot_sd_multiplier * sd)
    required = max(min_blocks, math.ceil(((_NORMAL.inv_cdf(1 - alpha) + _NORMAL.inv_cdf(power))
                                          * planning_sd / margin) ** 2))
    status = "futility" if mean <= -margin else "inconclusive_budget" if required > max_blocks else "ready"
    return {
        "schema": NI_SCHEMA, "phase": "planning", "status": status,
        "run_confirmation": status == "ready",
        "pilot_blocks": len(values), "pilot_mean_utility_difference": mean,
        "pilot_sd": sd, "pilot_zero_variance": sd == 0,
        "planning_sd": planning_sd, "variance_floor_sd": variance_floor_sd,
        "pilot_sd_multiplier": pilot_sd_multiplier,
        "margin": margin, "alpha": alpha, "planning_power": power,
        "assumed_true_utility_difference": 0.0,
        "minimum_blocks": min_blocks, "maximum_blocks": max_blocks,
        "required_blocks": required,
        "confirmation_blocks": required if status == "ready" else None,
        "approximate_power_at_required_blocks": _NORMAL.cdf(math.sqrt(required) * margin / planning_sd - _NORMAL.inv_cdf(1 - alpha)),
        "approximate_power_at_cap": _NORMAL.cdf(math.sqrt(max_blocks) * margin / planning_sd - _NORMAL.inv_cdf(1 - alpha)),
        "hypotheses": "H0: mean(student utility - teacher utility) <= -margin; H1: mean > -margin",
        "limitations": _LIMITATIONS + " A zero-variance pilot uses the positive SD floor; it is not evidence of zero population variance.",
    }


def noninferiority_test(
    differences: Sequence[float], *, planned_blocks: int,
    margin: float = .1, alpha: float = .05, min_blocks: int = 185,
    max_blocks: int = 256,
) -> dict:
    """Test the predeclared complete fixed sample; positive delta is better.

    Pass only when mean - z_(1-alpha)*sample_SD/sqrt(n) > -margin.
    Wrong sample counts and zero observed variance are explicitly inconclusive,
    never silently interpreted as a successful test with zero uncertainty.
    """
    _parameters(margin, alpha, min_blocks, max_blocks)
    if type(planned_blocks) is not int or not min_blocks <= planned_blocks <= max_blocks:
        raise ValueError("planned_blocks must satisfy the declared minimum and maximum")
    values = _values(differences)
    mean, sd = _moments(values)
    result = {
        "schema": NI_SCHEMA, "phase": "confirmation", "method": "one_sided_normal_approximation",
        "valid": False, "passed": False, "status": "inconclusive_count",
        "independent_blocks": len(values), "planned_blocks": planned_blocks,
        "mean_utility_difference": mean, "mean_rank_improvement": mean,
        "sample_sd": sd, "standard_error": None, "margin": margin,
        "alpha": alpha, "confidence_level": 1 - alpha,
        "one_sided_lower_bound": None, "one_sided_p_value": None,
        "hypotheses": "H0: mean(student utility - teacher utility) <= -margin; H1: mean > -margin",
        "limitations": _LIMITATIONS,
    }
    if len(values) != planned_blocks:
        return result
    if sd == 0:
        result["status"] = "inconclusive_zero_variance"
        return result
    assert sd is not None
    se = sd / math.sqrt(len(values))
    lower = mean - _NORMAL.inv_cdf(1 - alpha) * se
    passed = lower > -margin
    result.update({
        "valid": True, "passed": passed,
        "status": "noninferiority_demonstrated" if passed else "not_demonstrated",
        "standard_error": se, "one_sided_lower_bound": lower,
        "one_sided_p_value": 1 - _NORMAL.cdf((mean + margin) / se),
    })
    return result
