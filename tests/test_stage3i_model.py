"""Synthetic feature tests: numerical/identity checks, not rule certification."""
from copy import deepcopy
import hashlib
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from riichi_ai.features import GLOBAL_FEATURES, HISTORY_FIELDS, MAX_HISTORY, TILE_CHANNELS
from riichi_ai.model import PolicyNetwork, _state_hash, _weights_hash, collate_encoded, efficiency_model_config
from riichi_ai.semantics import ACTION_KINDS
from riichi_ai.stage3c import CallPolicy, HEAD_CONFIG as CALL_HEAD_CONFIG, SCHEMA as CALL_SCHEMA
from riichi_ai.stage3g_collect import explore
from riichi_ai.stage3i_model import (
    BUNDLE_SCHEMA, HEAD_CONFIG, FrozenReference, ResidualHeads, correction_choices,
    explorer_marginal, first_win_index, load_head_bundle,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def encoded(names=("PASS", "CHI", "PON"), seed=1):
    rng = np.random.default_rng(seed)
    tokens = np.zeros((len(names), 8), dtype=np.int64)
    tokens[:, 0] = [ACTION_KINDS.index(name) + 1 for name in names]
    tokens[:, 1] = np.arange(len(names)) % 34 + 1
    efficiency = np.zeros((len(names), 4), dtype=np.float32)
    efficiency[:, 3] = np.asarray([name == "DISCARD" for name in names], dtype=np.float32)
    return {"tiles": rng.normal(size=(34, TILE_CHANNELS)).astype(np.float32),
            "globals": rng.normal(size=(GLOBAL_FEATURES,)).astype(np.float32),
            "history": np.zeros((MAX_HISTORY, HISTORY_FIELDS), dtype=np.int64),
            "history_mask": np.zeros(MAX_HISTORY, dtype=np.bool_),
            "action_tokens": tokens, "candidate_efficiency": efficiency}


@pytest.fixture
def frozen_files(tmp_path):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(36101)
        base = PolicyNetwork(efficiency_model_config({"suit_width": 8, "history_width": 4, "hidden_width": 16}))
        call = CallPolicy(deepcopy(base), 310211)
    reference = tmp_path / "reference.pt"
    torch.save({"model_config": base.config, "model_state_dict": base.state_dict()}, reference)
    reference_sha = hashlib.sha256(reference.read_bytes()).hexdigest()
    call_path = tmp_path / "call.pt"
    torch.save({"schema": CALL_SCHEMA, "head_config": CALL_HEAD_CONFIG,
                "reference_sha256": reference_sha, "updates": 0, "optimizer_steps_executed": 0,
                "retired_batches": [], "model_config": base.config, "initialization_seed": 310211,
                "model_state_dict": call.state_dict(), "weights_sha256": _weights_hash(call),
                "heads_sha256": _weights_hash(call.heads), "base_weights_sha256": _weights_hash(base),
                "immutable_base_sha256": _weights_hash(base)}, call_path)
    return reference, reference_sha, call_path, hashlib.sha256(call_path.read_bytes()).hexdigest(), call


def test_loading_freezes_every_actor_weight_preserves_rng_and_requires_hash(frozen_files):
    args = frozen_files[:4]
    before = torch.get_rng_state().clone()
    frozen = FrozenReference.from_checkpoints(*args)
    assert torch.equal(before, torch.get_rng_state())
    assert all(not p.requires_grad and p.grad is None for p in frozen.parameters())
    frozen.train(True)
    assert all(not part.training for part in frozen.modules())
    frozen.assert_unchanged()
    with pytest.raises(ValueError, match="file hash"):
        FrozenReference.from_checkpoints(args[0], "f" * 64, *args[2:])


@pytest.mark.parametrize("count", [1, 3])
def test_extractor_exact_historic_forward_with_padding(frozen_files, count):
    frozen = FrozenReference.from_checkpoints(*frozen_files[:4])
    samples = [encoded(("PASS", "CHI", "PON") if row % 2 == 0 else ("DISCARD",), row) for row in range(count)]
    batch = collate_encoded(samples)
    with torch.no_grad():
        expected, _ = frozen.base(batch)
    logits, context, candidates = frozen.frozen_representations(batch)
    assert torch.equal(logits, expected)
    assert context.shape == (count, 256) and candidates.shape == (count, 3, 128)
    assert all(not value.requires_grad for value in (logits, context, candidates))
    probabilities = frozen.explorer_probabilities(logits, context, candidates, batch["action_tokens"][..., 0], batch["candidate_mask"])
    assert probabilities.dtype == np.float64
    np.testing.assert_allclose(probabilities.sum(axis=1), 1., rtol=0, atol=1e-15)
    assert np.all(probabilities[~batch["candidate_mask"].numpy()] == 0)
    polluted = dict(batch, terminal_utility=torch.zeros(count))
    with pytest.raises(ValueError, match="actor tensor fields"):
        frozen.frozen_representations(polluted)


@pytest.mark.parametrize("names", [("PASS", "CHI", "PON"), ("PASS", "RON", "TSUMO"), ("CHI", "PON"), ("DISCARD",)])
def test_reconstructed_explorer_matches_existing_collection_exactly(frozen_files, names):
    frozen = FrozenReference.from_checkpoints(*frozen_files[:4])
    call = frozen_files[4]
    row = encoded(names)
    batch = collate_encoded([row])
    logits, context, candidates = frozen.frozen_representations(batch)
    actual = frozen.explorer_probabilities(logits, context, candidates, batch["action_tokens"][..., 0], batch["candidate_mask"])[0]
    with torch.no_grad():
        base = call.distribution(row)["probabilities"].numpy()
    expected = explore([SimpleNamespace(kind=name) for name in names], base, random.Random(310))
    assert np.array_equal(actual, np.asarray(expected["probabilities"]))


def test_heads_initialize_independently_zero_outputs_and_detach_inputs(frozen_files):
    before = torch.get_rng_state().clone()
    heads = ResidualHeads(123)
    assert torch.equal(before, torch.get_rng_state())
    other = ResidualHeads(123)
    assert _weights_hash(heads) == _weights_hash(other)
    context = torch.randn(4, 256, requires_grad=True)
    candidates = torch.randn(4, 3, 128, requires_grad=True)
    predictions = heads.predictions(context, candidates)
    assert all(torch.count_nonzero(value) == 0 for value in predictions.values())
    frozen = FrozenReference.from_checkpoints(*frozen_files[:4])
    batch = collate_encoded([encoded()])
    before_logits, _, _ = frozen.frozen_representations(batch)
    state_hash, action_hash = _weights_hash(heads.state), _weights_hash(heads.action)
    optimizer = torch.optim.Adam(heads.value.parameters(), lr=.001)
    optimizer.zero_grad(set_to_none=True)
    (predictions["value"] - 1).square().mean().backward()
    optimizer.step()
    assert context.grad is None and candidates.grad is None
    assert all(p.grad is None for p in heads.state.parameters())
    assert all(p.grad is None for p in heads.action.parameters())
    assert _weights_hash(heads.state) == state_hash and _weights_hash(heads.action) == action_hash
    after_logits, _, _ = frozen.frozen_representations(batch)
    assert torch.equal(before_logits, after_logits)
    frozen.assert_unchanged()


def test_correction_identity_threshold_boundary_and_stable_ties():
    kinds = np.full(3, ACTION_KINDS.index("DISCARD") + 1)
    logits = np.asarray([1., 0., -100.])
    probability = np.asarray([.001, .02, .979])
    residuals = np.asarray([-100., 100., 1000.])
    zero = correction_choices(logits, residuals, probability, kinds, 0.)
    assert zero["selected_index"] == zero["raw_reference_index"] == 0
    assert zero["eligible"].tolist() == [True, True, True]
    assert zero["prior"].tolist() == [0., -1., -8.]
    identity = correction_choices(logits, np.zeros(3), probability, kinds, 100.)
    assert identity["selected_index"] == 0
    tied = correction_choices(np.zeros(3), np.zeros(3), probability, kinds, 10.)
    assert tied["selected_index"] == 0
    excluded = correction_choices(logits, residuals, np.asarray([.97, .01, .02]), kinds, .001)
    assert excluded["eligible"].tolist() == [True, False, True]
    assert np.isneginf(excluded["scores"][1])


def test_shared_first_win_wrapper_keeps_raw_and_wrapped_identities_separate():
    kinds = np.asarray([ACTION_KINDS.index(k) + 1 for k in ("PASS", "RON", "TSUMO")])
    logits, residuals = np.asarray([4., 0., -1.]), np.asarray([0., 0., 100.])
    q = explorer_marginal(np.asarray([.8, .1, .1]), kinds)
    assert q.tolist() == [0., 1., 0.]
    raw = correction_choices(logits, residuals, q, kinds, 0., wrapped=False)
    wrapped = correction_choices(logits, residuals, q, kinds, 0., wrapped=True)
    assert raw["selected_index"] == 0 and wrapped["selected_index"] == 1
    assert first_win_index(kinds, 2) == 1
    corrected = correction_choices(logits, residuals, q, kinds, 100., wrapped=True)
    assert corrected["selected_index"] == 1


@pytest.mark.parametrize("change", ["negative_strength", "bad_q", "nonfinite", "threshold", "prior"])
def test_correction_rejects_undeclared_or_nonfinite_parameters(change):
    args = {"logits": np.asarray([1., 0.]), "residuals": np.zeros(2), "probabilities": np.asarray([.9, .1]),
            "kinds": np.ones(2), "strength": 1.}
    if change == "negative_strength":
        args["strength"] = -1.
    elif change == "bad_q":
        args["probabilities"] = np.asarray([.9, .2])
    elif change == "nonfinite":
        args["residuals"] = np.asarray([np.nan, 1.])
    elif change == "threshold":
        args["threshold"] = .01
    else:
        args["prior_floor"] = -10.
    with pytest.raises(ValueError, match="Invalid correction"):
        correction_choices(**args)


def test_strict_bundle_loader_rejects_identity_and_tensor_tampering(tmp_path):
    heads = ResidualHeads(99)
    identity = {name: "a" * 64 for name in ("reference_sha256", "call_sha256", "protocol_sha256", "cache_manifest_sha256")}
    saved = {"schema": BUNDLE_SCHEMA, "seed": heads.seed, "identity": identity,
             "selected_epochs": {name: 0 for name in ("value", "state", "action")},
             "head_config": HEAD_CONFIG, "head_state_dicts": {name: getattr(heads, name).state_dict() for name in ("value", "state", "action")},
             "weights_sha256": _state_hash(heads.state_dict())}
    path = tmp_path / "heads.pt"
    torch.save(saved, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded, metadata = load_head_bundle(path, digest, identity)
    assert metadata["checkpoint_sha256"] == digest
    assert _weights_hash(loaded) == _weights_hash(heads)
    assert all(not p.requires_grad for p in loaded.parameters())
    with pytest.raises(ValueError, match="identity differs"):
        load_head_bundle(path, expected_identity={"reference_sha256": "b" * 64})
    saved["head_state_dicts"]["action"]["layers.2.bias"] = torch.ones(1)
    torch.save(saved, path)
    with pytest.raises(ValueError, match="tensor hash"):
        load_head_bundle(path)
