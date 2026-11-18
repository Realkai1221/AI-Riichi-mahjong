"""Inference-only fixtures; run only inside the diagnosis numerical lease."""
from copy import deepcopy
import math
import random

import numpy as np
import pytest
import torch

from riichi_ai.model import PolicyNetwork, efficiency_model_config
from riichi_ai.package_a_diagnosis_fit import (
    METRIC_NAMES, _float64_probabilities, discard_error, evaluate_fit,
)
from riichi_ai.stage2b_training import evaluate_stage2b
from test_training_v2 import records, SMALL


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def zero_model():
    model = PolicyNetwork(efficiency_model_config(SMALL))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model


def mean(report, group, metric):
    return report["groups"][group]["metrics"][metric]["mean"]


def test_known_severity_and_agreement_match_existing_evaluator():
    data, model = records(True), zero_model()
    previous = evaluate_stage2b(model, data, batch_size=3)
    report, arrays = evaluate_fit(model, data, indices=list(range(16)), batch_size=3)
    assert report["complete"] and report["rows"] == 16
    assert mean(report, "optional_teacher_discard", "tie_agreement") == previous["discard_tie_agreement"] == 7 / 16
    assert mean(report, "optional", "original_ce_f32") == pytest.approx(previous["policy_ce"])
    teacher = report["groups"]["stratum:teacher:optional_teacher_discard"]
    assert teacher["severity"]["shanten_increments"]["histogram"] == {"1": 3}
    assert teacher["severity"]["same_shanten_lost_ukeire"]["histogram"] == {"3": 3}
    assert teacher["severity"]["first_difference_counts"]["red"] == 3
    assert teacher["severity"]["shanten_increments"]["denominator"] == 12
    assert arrays["candidate_offsets"].tolist() == list(range(0, 33, 2))
    assert arrays["logits"].shape == (32,)
    assert all(value.dtype != object for value in arrays.values())
    assert mean(report, "optional", "target_nonbest_mass") == pytest.approx(.01)
    assert mean(report, "optional", "smooth_kl_f64") == pytest.approx(math.log(2) + .99 * math.log(.99) + .01 * math.log(.01), abs=1e-8)


def test_fit_preserves_weights_modes_rng_and_stored_targets():
    data, model = records(True), zero_model()
    model.train()
    model.rank_head.eval()  # Preserve mixed modes too.
    saved_modes = [m.training for m in model.modules()]
    saved_weights = {name: value.clone() for name, value in model.state_dict().items()}
    saved_targets = [row["policy_target"].copy() for row in data]
    torch_rng, python_rng, numpy_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    calls = []
    report, arrays = evaluate_fit(model, data, indices=[3, 1, 14], batch_size=2,
        check=lambda: calls.append("check"), progress=lambda done, total: calls.append((done, total)))
    assert report["rows"] == 3 and arrays["indices"].tolist() == [3, 1, 14]
    assert calls == ["check", (2, 3), "check", (3, 3), "check"]
    assert [m.training for m in model.modules()] == saved_modes
    assert not model.context._forward_hooks
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in saved_weights.items())
    assert all(np.array_equal(row["policy_target"], value) for row, value in zip(data, saved_targets))
    assert torch.equal(torch_rng, torch.get_rng_state()) and random.getstate() == python_rng
    after_numpy = np.random.get_state()
    assert numpy_rng[0] == after_numpy[0] and np.array_equal(numpy_rng[1], after_numpy[1]) and numpy_rng[2:] == after_numpy[2:]
    assert all(parameter.grad is None for parameter in model.parameters())


def test_forced_and_all_best_denominators_and_padding_are_separate():
    data = [deepcopy(row) for row in records(True)[:3]]
    forced = data[0]
    for key in ("action_tokens", "candidate_efficiency"):
        forced["encoded"][key] = forced["encoded"][key][:1].copy()
    forced.update(policy_target=np.ones(1, dtype=np.float32), best_mask=np.ones(1, dtype=bool),
                  chosen_index=0, optional=False,
                  teacher_preference_keys=np.array([[0, 0, 2, -10, 0]]))
    data[1].update(policy_target=np.array([.5, .5], dtype=np.float32), best_mask=np.ones(2, dtype=bool),
                   chosen_index=0, teacher_preference_keys=np.array([[0, 0, 2, -10, 0]] * 2))
    report, arrays = evaluate_fit(zero_model(), data, indices=[0, 1, 2], batch_size=3)
    assert report["groups"]["forced"]["rows"] == 1
    assert report["groups"]["optional"]["rows"] == 2
    assert mean(report, "forced", "original_ce_f32") == 0
    assert mean(report, "forced", "target_nonbest_mass") == 0
    assert mean(report, "optional", "target_nonbest_mass") == pytest.approx(.005)
    gap = report["groups"]["all"]["metrics"]["best_minus_nonbest_max_gap"]
    assert gap["denominator"] == 1
    assert arrays["candidate_offsets"].tolist() == [0, 1, 3, 5]
    assert arrays["logits"].size == 5 and np.isfinite(arrays["logits"]).all()


def test_float64_normalization_exposes_underflow_without_changing_argmax():
    z = np.array([[0., -110., -np.finfo(np.float32).max]])
    mask = np.array([[True, True, False]])
    best = np.array([[True, False, False]])
    q = best.astype(np.float64)
    smooth = np.array([[.99, .01, 0.]])
    values = _float64_probabilities(z, mask, q, smooth, best, ~best & mask, np.zeros_like(mask))
    assert torch.tensor(z, dtype=torch.float32).softmax(-1)[0, 1] == 0
    assert values[0, 7] > 0  # Same logits, non-best mass in float64.
    assert values[0, 2] == pytest.approx(1.1)
    shifted = _float64_probabilities(z + 1000., mask, q, smooth, best, ~best & mask, np.zeros_like(mask))
    assert np.allclose(values, shifted, atol=1e-12, rtol=1e-12)
    assert z.argmax(1).tolist() == (z + 1000.).argmax(1).tolist()


def test_deadline_exception_restores_mode_and_removes_hook():
    model = zero_model()
    model.train()
    def stop():
        raise TimeoutError("lease fixture")
    with pytest.raises(TimeoutError, match="lease fixture"):
        evaluate_fit(model, records(True), indices=[0, 1], check=stop)
    assert model.training and not model.context._forward_hooks
    assert all(p.grad is None for p in model.parameters())


def test_explicit_indices_and_stratum_are_validated():
    model, data = zero_model(), records(True)
    for indices in ([], [0, 0], [-1], [len(data)], [True]):
        with pytest.raises(ValueError, match="indices"):
            evaluate_fit(model, data, indices=indices)
    bad = deepcopy(data[:1])
    bad[0]["stratum"] = "unknown"
    with pytest.raises(ValueError, match="stratum"):
        evaluate_fit(model, bad, indices=[0])


def test_preference_disagreement_fails_and_wrong_action_kind_is_not_ukeire():
    data = deepcopy(records(True))
    broken = data[1]
    broken["best_mask"] = np.array([True, True])
    with pytest.raises(ValueError, match="disagree"):
        discard_error(broken, 0)
    row = data[0]
    row["encoded"]["action_tokens"] = row["encoded"]["action_tokens"].copy()
    from riichi_ai.semantics import ACTION_KINDS
    row["encoded"]["action_tokens"][1, 0] = ACTION_KINDS.index("PON") + 1
    row["teacher_preference_keys"][1] = [1, 0, 99, 99, 0]
    assert discard_error(row, 1) == (0, 0, 0, 1, True)
