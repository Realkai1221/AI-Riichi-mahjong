"""Finite mathematical/read-only checks, not games or strength tests."""
import json
import math

import numpy as np
import pytest
import torch

from riichi_ai import stage3b
from riichi_ai.features import GLOBAL_FEATURES, HISTORY_FIELDS, MAX_HISTORY, TILE_CHANNELS
from riichi_ai.model import PolicyNetwork, _weights_hash, efficiency_model_config
from riichi_ai.semantics import ACTION_KINDS


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_underflow_is_distinct_from_finite_stable_log_probability():
    logits = torch.tensor([0., -1000.], requires_grad=True)
    result = stage3b.distribution(logits)
    assert result["probabilities"].tolist() == [1., 0.]
    assert result["log_probabilities"].tolist() == [0., -1000.]
    assert result["mixture_gradient_attenuation"].tolist() == [1., 1.]
    gradient, = torch.autograd.grad(result["log_probabilities"][1], logits)
    assert gradient.tolist() == [-1., 1.]


def test_legal_uniform_mixture_has_exact_mask_and_correct_logq_derivative():
    logits = torch.tensor([0.4, -0.8, 999.], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([True, True, False])
    result = stage3b.distribution(logits, mask, epsilon=.05)
    pi = torch.softmax(logits[:2], 0)
    expected_q = .95 * pi + .025
    assert torch.allclose(result["probabilities"][:2], expected_q, atol=1e-15, rtol=0)
    assert result["probabilities"][2] == 0
    assert result["log_probabilities"][2] == -math.inf
    gradient, = torch.autograd.grad(result["log_probabilities"][1], logits)
    attenuation = .95 * pi[1] / expected_q[1]
    expected_gradient = attenuation * torch.tensor([-pi[0], 1-pi[1]], dtype=torch.float64)
    assert torch.allclose(gradient[:2], expected_gradient, atol=1e-15, rtol=0)
    assert gradient[2] == 0
    assert not torch.allclose(gradient[:2], torch.tensor([-pi[0], 1-pi[1]], dtype=torch.float64))


def test_uniform_mixture_can_restore_support_but_have_zero_parameter_gradient():
    logits = torch.tensor([0., -1000.], requires_grad=True)
    result = stage3b.distribution(logits, epsilon=.05)
    assert result["probabilities"][1] == .025
    assert result["mixture_gradient_attenuation"][1] == 0
    gradient, = torch.autograd.grad(result["log_probabilities"][1], logits)
    assert torch.equal(gradient, torch.zeros(2))


def test_temperature_restores_support_and_has_inverse_temperature_derivative():
    logits = torch.tensor([0., -1000.], requires_grad=True)
    result = stage3b.distribution(logits, temperature=100)
    assert result["probabilities"][1] > 0
    gradient, = torch.autograd.grad(result["log_probabilities"][1], logits)
    probability = result["probabilities"][0]
    assert gradient[0] == pytest.approx(-float(probability) / 100, abs=1e-9)
    assert gradient[1] == pytest.approx(float(probability) / 100, abs=1e-9)


def test_padding_and_common_offset_do_not_change_probabilities():
    plain = stage3b.distribution(torch.tensor([4., 1.], dtype=torch.float64), epsilon=.05)
    padded = stage3b.distribution(torch.tensor([104., 101., float("nan")], dtype=torch.float64),
                                  torch.tensor([True, True, False]), epsilon=.05)
    assert torch.equal(plain["probabilities"], padded["probabilities"][:2])
    assert padded["probabilities"][-1] == 0


@pytest.mark.parametrize("temperature,epsilon", [(0, 0), (-1, 0), (math.inf, 0), (1, -1), (1, 1), (1, math.nan)])
def test_invalid_transform_rejected(temperature, epsilon):
    with pytest.raises(ValueError):
        stage3b.distribution(torch.tensor([1., 2.]), temperature=temperature, epsilon=epsilon)


def test_invalid_mask_and_legal_nan_rejected():
    for logits, mask in [(torch.tensor([1., 2.]), torch.tensor([False, False])),
                         (torch.tensor([1., 2.]), torch.tensor([True])),
                         (torch.tensor([1., float("nan")]), None)]:
        with pytest.raises(ValueError):
            stage3b.distribution(logits, mask)


def small_encoded():
    return {
        "tiles": np.zeros((34, TILE_CHANNELS), dtype=np.float32),
        "globals": np.zeros(GLOBAL_FEATURES, dtype=np.float32),
        "history": np.zeros((MAX_HISTORY, HISTORY_FIELDS), dtype=np.int16),
        "history_mask": np.zeros(MAX_HISTORY, dtype=np.bool_),
        "action_tokens": np.asarray([[ACTION_KINDS.index("CHI")+1, 1, 0, 0, 1, 2, 0, 0],
                                      [ACTION_KINDS.index("PASS")+1, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int16),
        "candidate_efficiency": np.zeros((2, 4), dtype=np.float32),
    }


def small_model():
    config = efficiency_model_config({"suit_width": 8, "history_width": 4, "hidden_width": 16,
                                      "context_width": 8, "candidate_width": 8})
    model = PolicyNetwork(config).eval()
    for parameter in model.rank_head.parameters():
        parameter.requires_grad_(False)
    return model


def test_gradient_probe_has_no_weight_rng_or_grad_mutations():
    model, encoded = small_model(), small_encoded()
    before, rng = _weights_hash(model), torch.get_rng_state().clone()
    results = stage3b.gradient_probe(model, encoded, target_index=0, device="cpu")
    assert set(results) == {item["name"] for item in stage3b.GRID}
    assert all(item["gradients_finite"] for item in results.values())
    assert any(item["parameter_gradient_l2"] > 0 for item in results.values())
    assert before == _weights_hash(model)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(parameter.grad is None for parameter in model.parameters())
    json.dumps(results, allow_nan=False)


def test_derived_norm_overflow_is_reported_invalid_and_json_safe():
    class HugeGradient(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))

        def forward(self, batch):
            logits = torch.stack([self.value * 1e200, self.value * 0])
            return logits.unsqueeze(0), None

    result = stage3b.gradient_probe(HugeGradient(), small_encoded(), target_index=0, device="cpu")
    for variant in result.values():
        assert not variant["gradients_finite"]
        assert variant["parameter_gradient_l2"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_staged_float64_distribution_forward_and_backward():
    known = [0.0, 1.0, -1.0, .125, 1e-20, 10000.0]
    logits = torch.tensor(known, device="mps", dtype=torch.float32, requires_grad=True)
    result = stage3b.distribution(logits, temperature=10000, epsilon=.05)
    cpu_logits = torch.tensor(known, dtype=torch.float32, requires_grad=True)
    reference = stage3b.distribution(cpu_logits, temperature=10000, epsilon=.05)
    assert result["normalization_logits"].tolist() == cpu_logits.detach().to(torch.float64).tolist()
    assert torch.equal(result["probabilities"], reference["probabilities"])
    assert result["probabilities"].max() - result["probabilities"].min() > .1
    mps_gradient, = torch.autograd.grad(result["log_probabilities"][1], logits)
    cpu_gradient, = torch.autograd.grad(reference["log_probabilities"][1], cpu_logits)
    copied_gradient = mps_gradient.to("cpu")
    assert torch.equal(copied_gradient, cpu_gradient)
    assert float(copied_gradient.abs().max()) > 1e-5


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_model_parameter_gradient_probe_is_nonzero_finite_and_read_only():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31415)
        cpu_model = small_model()
        mps_model = small_model()
    mps_model.load_state_dict(cpu_model.state_dict())
    mps_model.to("mps")
    encoded = small_encoded()
    cpu = stage3b.gradient_probe(cpu_model, encoded, target_index=0, device="cpu")
    before = _weights_hash(mps_model)
    mps = stage3b.gradient_probe(mps_model, encoded, target_index=0, device="mps")
    for name in cpu:
        assert mps[name]["gradients_finite"]
        assert mps[name]["parameter_gradient_l2"] > 1e-6
        assert mps[name]["parameter_gradient_l2"] == pytest.approx(
            cpu[name]["parameter_gradient_l2"], rel=.02, abs=1e-5)
        assert mps[name]["target_probability"] == pytest.approx(
            cpu[name]["target_probability"], rel=.001, abs=1e-5)
    assert _weights_hash(mps_model) == before
    assert all(parameter.grad is None for parameter in mps_model.parameters())
    json.dumps(mps, allow_nan=False)


def test_selection_order_is_deterministic_exclusive_and_covers_rare_strata():
    rows = [{"id": str(index), "stratum": stratum, "selection_key": str(index)}
            for index, stratum in enumerate(["CHI", "PON", "CHI", "ANKAN", "DISCARD"])]
    selected = stage3b._round_robin(rows, 4)
    assert [row["stratum"] for row in selected] == ["ANKAN", "CHI", "PON", "DISCARD"]
    assert selected == stage3b._round_robin(list(reversed(rows)), 4)
    assert len({row["id"] for row in selected}) == 4
    assert stage3b._stratum(["PON", "CHI", "PASS"]) == "CHI"
    assert stage3b._stratum(["DISCARD", "ANKAN"]) == "ANKAN"
    assert stage3b._stratum(["RON", "PON", "PASS"]) == "PON"
    assert stage3b._stratum(["RON", "PASS"]) == "WIN"
