"""Policy masking, candidate permutation, numerical gradients, and checkpoints."""

from dataclasses import replace
import hashlib

import numpy as np
import pytest
import torch

from riichi_ai.features import FEATURE_SCHEMA, encode_view
from riichi_ai.model import NeuralPolicy, PolicyNetwork, collate_encoded, model_config
from riichi_ai.player import PlayerAdapter
from riichi_ai.semantics import semantic_actions
from riichi_ai.simulator import create_env


SMALL_CONFIG = {"suit_width": 32, "history_width": 16, "hidden_width": 64,
                "context_width": 32, "candidate_width": 16}


def sample_view():
    _, observations = create_env(91)
    return PlayerAdapter(0).build(observations[0])


def test_default_architecture_manifest_and_parameter_budget():
    model = PolicyNetwork()
    assert 900_000 <= sum(parameter.numel() for parameter in model.parameters()) <= 1_100_000
    assert model.config == model_config()
    assert model.config["feature_schema"] == FEATURE_SCHEMA
    with pytest.raises(ValueError, match="Unsupported feature_schema"):
        PolicyNetwork({"feature_schema": "different"})


def test_variable_candidates_mask_and_policy_rank_gradients():
    torch.manual_seed(17)
    view = sample_view()
    forced = replace(view, legal_actions=(view.legal_actions[0],))
    batch = collate_encoded([encode_view(view), encode_view(forced)])
    model = PolicyNetwork(SMALL_CONFIG)
    logits, ranks = model(batch)
    assert logits.shape == batch["candidate_mask"].shape
    assert ranks.shape == (2, 4)
    assert torch.all(logits[~batch["candidate_mask"]] == torch.finfo(torch.float32).min)
    probabilities = logits.softmax(dim=-1)
    assert probabilities[1, 0] == 1
    assert torch.all(probabilities[~batch["candidate_mask"]] == 0)
    targets = batch["candidate_mask"].float()
    targets /= targets.sum(dim=1, keepdim=True)
    loss = -(targets * logits.log_softmax(dim=-1)).sum(dim=-1).mean()
    loss = loss + torch.nn.functional.cross_entropy(ranks, torch.tensor([0, 2]))
    assert torch.isfinite(loss)
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    assert sum(float(parameter.grad.square().sum()) for parameter in model.parameters()) > 0


def test_candidate_permutation_equivariance_and_history_padding_invariance():
    torch.manual_seed(19)
    model = PolicyNetwork(SMALL_CONFIG).eval()
    view = sample_view()
    actions = semantic_actions(view)
    first = encode_view(view, actions)
    reverse = encode_view(view, actions[::-1])
    with torch.inference_mode():
        logits, ranks = model(collate_encoded([first]))
        swapped_logits, swapped_ranks = model(collate_encoded([reverse]))
    torch.testing.assert_close(logits.flip(-1), swapped_logits, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(ranks, swapped_ranks, rtol=0, atol=0)
    changed = {key: value.copy() for key, value in first.items()}
    changed["history"][~changed["history_mask"]] = 1
    with torch.inference_mode():
        masked_logits, masked_ranks = model(collate_encoded([changed]))
    torch.testing.assert_close(logits, masked_logits, rtol=0, atol=0)
    torch.testing.assert_close(ranks, masked_ranks, rtol=0, atol=0)


def test_no_cross_sample_or_padding_effect_on_rank_features():
    torch.manual_seed(20)
    model = PolicyNetwork(SMALL_CONFIG).eval()
    view = sample_view()
    first = encode_view(view)
    second = encode_view(replace(view, legal_actions=(view.legal_actions[0],)))
    with torch.inference_mode():
        logits, ranks = model(collate_encoded([first]))
        batch_logits, batch_ranks = model(collate_encoded([first, second]))
    torch.testing.assert_close(logits[0], batch_logits[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(ranks[0], batch_ranks[0], rtol=1e-5, atol=1e-6)


def test_neural_policy_stable_tie_resolution_and_forced_bypass(monkeypatch):
    model = PolicyNetwork(SMALL_CONFIG)
    for parameter in model.parameters():
        parameter.data.zero_()
    policy = NeuralPolicy(model)
    view = sample_view()
    actions = semantic_actions(view)
    expected = actions[0].representative
    assert policy.choose(view) == expected
    assert policy.choose(replace(view, legal_actions=view.legal_actions[::-1])) == expected
    monkeypatch.setattr(model, "forward", lambda _: (_ for _ in ()).throw(AssertionError("unexpected network call")))
    assert policy.choose(replace(view, legal_actions=(expected,))) == expected
    assert policy.inference_stats() == {"decisions": 3, "forced_decisions": 1, "network_calls": 2}


def test_weights_only_checkpoint_roundtrip_and_schema_validation(tmp_path):
    torch.manual_seed(21)
    model = PolicyNetwork(SMALL_CONFIG)
    path = tmp_path / "model.pt"
    checkpoint = {"model_config": model.config, "model_state_dict": model.state_dict(),
                  "feature_schema": FEATURE_SCHEMA, "source_manifest": {"test": "fixture"}}
    torch.save(checkpoint, path)
    restored = NeuralPolicy.from_checkpoint(path)
    view = sample_view()
    assert restored.choose(view) == NeuralPolicy(model).choose(view)
    assert restored.manifest()["checkpoint_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert restored.manifest()["source_manifest"] == {"test": "fixture"}
    checkpoint["feature_schema"] = "incompatible"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="feature schema"):
        NeuralPolicy.from_checkpoint(path)


def test_invalid_collation_and_empty_decisions_fail_closed():
    with pytest.raises(ValueError, match="empty"):
        collate_encoded([])
    view = sample_view()
    with pytest.raises(ValueError, match="at least one"):
        collate_encoded([encode_view(replace(view, legal_actions=()))])
    with pytest.raises(ValueError, match="no legal action"):
        NeuralPolicy(PolicyNetwork(SMALL_CONFIG)).choose(replace(view, legal_actions=()))
