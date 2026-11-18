"""Fractional correction diagnostic: exact ties, thresholds and wrapper."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from stage3i_calibration_diagnostic import GRID, analyze_row
from riichi_ai.semantics import ACTION_KINDS
from riichi_ai.stage3i_model import correction_choices

D = ACTION_KINDS.index('DISCARD') + 1
W = ACTION_KINDS.index('RON') + 1


def test_fractional_grid_matches_frozen_deployed_formula():
    z = np.array([0., -.03, -9., -.1])
    a = np.array([.1, .4, 30., .8])
    q = np.array([.8, .1, .005, .095])
    kinds = np.array([D] * 4)
    result = analyze_row(z, a, q, kinds)
    for index, strength in enumerate(GRID):
        assert result['choices'][index] == correction_choices(z, a, q, kinds, strength)['selected_index']
    assert result['first_change_infimum'] == pytest.approx(.1)
    assert result['zero_plus'] == 0


def test_exact_reference_tie_causes_a_discontinuity_at_positive_lambda():
    result = analyze_row([2., 2., 1.], [0., 1., 10.], [.5, .4, .1], [D, D, D])
    assert result['choices'][0] == 0
    assert result['zero_plus'] == 1
    assert result['first_change_infimum'] == 0
    assert result['exact_eligible_prior_tie']
    assert result['choices'][1] == 1


def test_tied_residuals_keep_canonical_reference_and_no_positive_slope():
    result = analyze_row([0., 0.], [1., 1.], [.5, .5], [D, D])
    assert result['zero_plus'] == 0
    assert np.all(result['choices'] == 0)
    assert np.isinf(result['first_change_infimum'])


def test_ineligible_high_slope_cannot_create_breakpoint():
    result = analyze_row([0., -.01], [0., 100.], [.995, .005], [D, D])
    assert np.isinf(result['first_change_infimum'])
    assert np.all(result['choices'] == 0)


def test_win_wrapper_suppresses_all_changes_and_breakpoints():
    result = analyze_row([0., 0.], [0., 1.], [.5, .5], [W, D])
    assert result['zero_plus_raw'] == 1
    assert result['zero_plus'] == result['reference'] == 0
    assert np.all(result['choices'] == 0)
    assert np.isinf(result['first_change_infimum'])


def test_clipped_prior_controls_first_change_infimum():
    result = analyze_row([0., -100.], [0., 16.], [.5, .5], [D, D])
    assert result['first_change_infimum'] == .5
    assert result['choices'][-2] == 0  # canonical reference wins equality
    assert result['choices'][-1] == 1
