"""Versioned feature assistance, historical compatibility and matched controls."""

from dataclasses import replace
import hashlib

import numpy as np
import pytest
import torch

from riichi_ai.features import FEATURE_SCHEMA, FEATURE_SCHEMA_V2, encode_view, feature_manifest
from riichi_ai.model import (
    MODEL_SCHEMA, MODEL_SCHEMA_V2, NeuralPolicy, PolicyNetwork, _weights_hash,
    collate_encoded, efficiency_model_config, matched_initialization, model_config,
)
from riichi_ai.player import PlayerAdapter
from riichi_ai.semantics import semantic_actions
from riichi_ai.simulator import create_env


SMALL = {"suit_width": 16, "history_width": 8, "hidden_width": 32,
         "context_width": 16, "candidate_width": 16}


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def view():
    _, observations = create_env(91)
    return PlayerAdapter(0).build(observations[0])


def test_v1_initialization_preserves_pre_v2_golden_weights():
    # Recorded by loading model.py directly from the pre-v2 commit
    # 18364394c8595c4f0312cde787ebea719baad63c under pinned CPU Torch 2.14.1.
    torch.manual_seed(204)
    raw = PolicyNetwork(SMALL)
    assert _weights_hash(raw) == "3e495d78a6fad685fc10ed84295ec0f59a550cf68c83683d2fea0b61c56d6a6d"
    assert raw.config["model_schema"] == MODEL_SCHEMA
    assert feature_manifest()["schema"] == FEATURE_SCHEMA
    old = encode_view(view())
    assisted = encode_view(view(), feature_schema=FEATURE_SCHEMA_V2)
    assert set(assisted) == set(old) | {"candidate_efficiency"}
    for key in old:
        np.testing.assert_array_equal(old[key], assisted[key])


def test_matched_initialization_copies_common_tensors_and_preserves_rng():
    torch.manual_seed(205)
    before = torch.get_rng_state().clone()
    raw, assisted, manifest = matched_initialization(206, SMALL)
    assert torch.equal(before, torch.get_rng_state())
    assert set(assisted) - set(raw) == {"efficiency_encoder.weight"}
    assert all(torch.equal(value, assisted[key]) for key, value in raw.items())
    assert manifest["maximum_common_difference"] == 0
    repeat_raw, repeat_assisted, repeat_manifest = matched_initialization(206, SMALL)
    assert manifest == repeat_manifest
    assert all(torch.equal(value, repeat_assisted[key]) for key, value in assisted.items())
    assert all(torch.equal(value, repeat_raw[key]) for key, value in raw.items())
    with pytest.raises(ValueError, match="identical common"):
        matched_initialization(206, SMALL, {**SMALL, "candidate_width": 8})


def test_zero_added_branch_reproduces_v1_logits_and_rank_exactly():
    raw_state, assisted_state, _ = matched_initialization(207, SMALL)
    raw = PolicyNetwork(SMALL).eval()
    assisted = PolicyNetwork(efficiency_model_config(SMALL)).eval()
    raw.load_state_dict(raw_state)
    assisted.load_state_dict(assisted_state)
    with torch.no_grad():
        assisted.efficiency_encoder.weight.zero_()
    state = view()
    with torch.inference_mode():
        old_policy, old_rank = raw(collate_encoded([encode_view(state)]))
        new_policy, new_rank = assisted(collate_encoded([encode_view(state, feature_schema=FEATURE_SCHEMA_V2)]))
    torch.testing.assert_close(old_policy, new_policy, atol=0, rtol=0)
    torch.testing.assert_close(old_rank, new_rank, atol=0, rtol=0)
    # The added branch is small, and it never changes the rank-only context.
    assert sum(p.numel() for p in assisted.parameters()) - sum(p.numel() for p in raw.parameters()) == 4 * SMALL["candidate_width"]


def test_v2_candidate_permutation_mask_and_numeric_gradients():
    torch.manual_seed(208)
    state = view()
    actions = semantic_actions(state)
    first = encode_view(state, actions, feature_schema=FEATURE_SCHEMA_V2)
    reverse = encode_view(state, actions[::-1], feature_schema=FEATURE_SCHEMA_V2)
    model = PolicyNetwork(efficiency_model_config(SMALL)).eval()
    with torch.inference_mode():
        logits, ranks = model(collate_encoded([first]))
        reversed_logits, reversed_ranks = model(collate_encoded([reverse]))
    torch.testing.assert_close(logits.flip(-1), reversed_logits, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(ranks, reversed_ranks, atol=0, rtol=0)
    forced = encode_view(replace(state, legal_actions=(actions[0].representative,)), feature_schema=FEATURE_SCHEMA_V2)
    batch = collate_encoded([first, forced])
    policy_logits, rank_logits = model(batch)
    assert torch.isfinite(policy_logits[batch["candidate_mask"]]).all()
    assert torch.all(policy_logits.softmax(-1)[~batch["candidate_mask"]] == 0)
    assert torch.all(batch["candidate_efficiency"][~batch["candidate_mask"]] == 0)
    loss = -policy_logits.log_softmax(-1)[:, 0].mean() + torch.nn.functional.cross_entropy(rank_logits, torch.tensor([0, 1]))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.efficiency_encoder.weight.grad is not None
    assert torch.isfinite(model.efficiency_encoder.weight.grad).all()
    assert model.efficiency_encoder.weight.grad.abs().sum() > 0


def test_collation_and_forward_reject_mixed_or_wrong_schemas():
    raw = encode_view(view())
    assisted = encode_view(view(), feature_schema=FEATURE_SCHEMA_V2)
    with pytest.raises(ValueError, match="mix v1 and v2"):
        collate_encoded([raw, assisted])
    with pytest.raises(ValueError, match="Batch feature schema"):
        PolicyNetwork(SMALL)(collate_encoded([assisted]))
    with pytest.raises(ValueError, match="Batch feature schema"):
        PolicyNetwork(efficiency_model_config(SMALL))(collate_encoded([raw]))
    broken = {**assisted, "candidate_efficiency": assisted["candidate_efficiency"].copy()}
    broken["candidate_efficiency"][0, 3] = 0
    with pytest.raises(ValueError, match="applicability"):
        collate_encoded([broken])
    with pytest.raises(ValueError, match="same supported version"):
        PolicyNetwork({"feature_schema": FEATURE_SCHEMA_V2})


@pytest.mark.parametrize("assisted", [False, True])
def test_checkpoint_dispatch_preserves_bytes_and_forced_bypass(tmp_path, monkeypatch, assisted):
    configuration = efficiency_model_config(SMALL) if assisted else model_config(SMALL)
    model = PolicyNetwork(configuration)
    path = tmp_path / ("v2.pt" if assisted else "historical-v1.pt")
    torch.save({"model_config": configuration, "model_state_dict": model.state_dict(),
                "feature_schema": configuration["feature_schema"], "source_manifest": {"frozen": True}}, path)
    before = path.read_bytes()
    policy = NeuralPolicy.from_checkpoint(path)
    manifest_before = policy.manifest()
    state = view()
    assert policy.choose(state) in state.legal_actions
    timings = policy.inference_timing()
    assert timings["total_choose"] > 0 and timings["encoding"] > 0 and timings["model"] > 0
    assert (timings["efficiency"] > 0) == assisted
    assert policy.manifest() == manifest_before
    assert policy.manifest()["version"] == (MODEL_SCHEMA_V2 if assisted else MODEL_SCHEMA)
    expected = semantic_actions(state)[0].representative
    monkeypatch.setattr(policy.model, "forward", lambda _: pytest.fail("Forced choice used network"))
    monkeypatch.setattr("riichi_ai.model.efficiency_from_encoded", lambda *args: pytest.fail("Forced choice computed efficiency"))
    assert policy.choose(replace(state, legal_actions=(expected,))) == expected
    assert policy.inference_stats() == {"decisions": 2, "forced_decisions": 1, "network_calls": 1}
    assert path.read_bytes() == before
    assert policy.checkpoint_hash == hashlib.sha256(before).hexdigest()
