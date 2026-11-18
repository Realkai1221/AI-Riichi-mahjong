"""Run only within the parent diagnosis budget; inference and fixtures, no updates."""
import json

import numpy as np
import pytest
import torch

from riichi_ai.model import PolicyNetwork, efficiency_model_config
from riichi_ai.package_a_diagnosis_precision import compare_precision, summarize_outputs, prediction_error
from riichi_ai.training import collate_training
from test_training_v2 import records, SMALL


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_output_statistics_exclude_padding_and_separate_offsets():
    mask = np.asarray([[1, 1, 0], [1, 0, 0]], bool)
    best = np.asarray([[1, 0, 0], [1, 0, 0]], bool)
    logits = np.asarray([[100., 102., -3.4e38], [800., -3.4e38, -3.4e38]], np.float32)
    probabilities = np.asarray([[.2, .8, 0], [1, 0, 0]], np.float32)
    report, arrays = summarize_outputs(logits, probabilities, np.asarray([[3., 4.], [0., 0.]]), mask, best)
    assert report["padding_excluded"] == 3
    assert report["legal_logit_abs"]["maximum"] == 800
    assert report["centered_legal_logit_abs"]["maximum"] == 1
    assert report["top_two_margin_optional"]["mean"] == 2
    assert report["best_minus_nonbest_gap"]["mean"] == -2
    assert report["teacher_best_agreement_optional"] == 0
    assert report["argmax_illegal_count"] == 0
    assert np.array_equal(arrays["logits"][~mask], np.zeros(3))
    assert np.array_equal(arrays["has_nonbest"], [True, False])


def test_float64_normalization_recovers_underflow_but_never_raw_order():
    logits = torch.tensor([[0., -200.], [100., 100.]], dtype=torch.float32)
    mask = np.ones((2, 2), bool)
    best = np.asarray([[1, 0], [1, 1]], bool)
    context = np.zeros((2, 3), np.float32)
    left, a = summarize_outputs(logits.numpy(), logits.softmax(-1).numpy(), context, mask, best)
    right, b = summarize_outputs(logits.numpy(), logits.double().softmax(-1).numpy(), context, mask, best)
    assert a["probabilities"][0, 1] == 0 and b["probabilities"][0, 1] > 0
    assert np.array_equal(a["argmax"], b["argmax"])
    assert left["exact_top_tie_rows_optional"] == right["exact_top_tie_rows_optional"] == 1
    assert left["zero_probability_legal_candidates"] == 1
    assert right["zero_probability_legal_candidates"] == 0


def test_nonfinite_legal_output_is_recorded_and_invalid_masks_rejected():
    mask = np.ones((1, 2), bool)
    best = np.asarray([[1, 0]], bool)
    report, arrays = summarize_outputs(np.asarray([[np.nan, 1.]]), np.asarray([[.5, .5]]),
                                      np.zeros((1, 3)), mask, best)
    assert not report["finite"] and report["nonfinite_legal_logits"] == 1
    assert "argmax" not in arrays
    with pytest.raises(ValueError, match="legal best"):
        summarize_outputs(np.zeros((1, 2)), np.asarray([[1., 0.]]), np.zeros((1, 3)),
                          np.asarray([[1, 0]], bool), np.asarray([[0, 1]], bool))


@pytest.mark.parametrize("include_mps", [False, True])
def test_canonical_precision_comparison_preserves_caller_and_checks_padding(include_mps):
    if include_mps and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(732)
        model = PolicyNetwork(efficiency_model_config(SMALL))
    examples = records(True)[:3]
    batch, labels = collate_training(examples)
    # Preserve pre-existing gradients without running a backward or an update.
    first = next(model.parameters())
    first.grad = torch.ones_like(first)
    with torch.inference_mode():
        canonical, _ = model(batch)
    checks = []
    report, arrays = compare_precision(model, batch, labels, row_ids=[14, 29, 31],
                                      records=examples, include_mps=include_mps,
                                      check=lambda: checks.append(True))
    assert len(checks) == (4 if include_mps else 3)
    assert all(value is not False for value in report["checks"].values())
    assert report["checks"]["canonical_forward_used_without_adapter"]
    assert np.array_equal(arrays["cpu_float32__logits"], canonical.numpy())
    assert arrays["cpu_float64__logits"].dtype == np.float64
    assert model.training and first.dtype == torch.float32
    assert torch.equal(first.grad, torch.ones_like(first))
    for backend in (["cpu_float32", "mps_float32"] if include_mps else ["cpu_float32"]):
        pair = backend + "__vs__" + backend + "_softmax_float64"
        assert report["comparisons"][pair]["argmax_disagreement_rows"] == 0
    assert report["branches"]["cpu_float64"]["finite"]
    json.dumps(report, allow_nan=False)


def test_precision_comparison_refuses_invalid_best_mask_before_forward():
    model = PolicyNetwork(efficiency_model_config(SMALL))
    batch, labels = collate_training(records(True)[:2])
    labels["best"].zero_()
    with pytest.raises(ValueError, match="legal best"):
        compare_precision(model, batch, labels, include_mps=False)


def test_error_annotations_distinguish_shanten_and_ukeire():
    examples = records(True)
    shanten = prediction_error(examples[1], 0)
    ukeire = prediction_error(examples[2], 0)
    assert not shanten["teacher_best"] and shanten["shanten_increment"] == 1
    assert shanten["same_shanten_lost_ukeire"] is None
    assert not ukeire["teacher_best"] and ukeire["shanten_increment"] == 0
    assert ukeire["same_shanten_lost_ukeire"] == 3
