# Bounded Stage 2 imitation training

[training.py](../src/riichi_ai/training.py) implements `imitation-training-v1`. It consumes records returned by `dataset.load_split`, with actor encodings and targets kept separate. The trainer does not load the auditor artifact. The caller owns dataset generation, cumulative experiment budgets, full-match comparisons, and final release decisions.

## Objective and sampling

Each update samples dataset rows uniformly with replacement using a local NumPy PCG64 generator. Every record contains an actor encoding, a semantic policy target, best-action mask, deterministic teacher choice, zero-based final-rank target, and optional/forced flag. Labels are checked against the number of semantic legal candidates. The model only receives the actor encoding.

The default loss is:

```text
mean_optional[-sum(q * log(policy_probability))]
    + 0.1 * mean_all_rows[cross_entropy(rank_logits, final_rank)]
```

The policy term averages over optional decisions only. An all-forced minibatch has a zero policy term while retaining rank learning. `target_mode="soft"` uses the provided uniform distribution over best semantic classes; `"one_hot"` uses the deterministic teacher choice. Neither target is a claim about the teacher's actual stochastic behavior probabilities. AdamW starts at learning rate `3e-4`, weight decay `1e-4`, and global gradient clipping at 1.0. Losses, gradient norms, and updated parameters are checked for finiteness.

Rank targets repeat across correlated decisions in a match. These are decision-weighted training/diagnostic samples, not independent placement observations. The rank head initially describes the behavior policies and opponent mixture in the dataset; it is not a guarantee of calibrated values for a subsequently changed policy.

## API and artifacts

```python
report = train(
    training_records, output_directory,
    validation=validation_records,
    device="mps", steps=2500, batch_size=256, seed=20261017,
    max_seconds=600, memory_limit_bytes=32 * 1024**3,
    target_mode="soft", rank_weight=0.1,
    validate_every=250, validation_max_samples=10000,
)
```

`steps` means the absolute total number of optimizer updates, including updates completed before a resume. The report records actual completed steps, losses, validation checks, sampled memory, runtime, source/data identities, resource stops, and failures. An existing nonempty output directory requires an explicit resume checkpoint.

- `checkpoint.pt` is the last resumable model, optimizer, RNG/sampler state, completed-step cursor, histories, selection state, and provenance.
- `bestcheckpoint.pt` is an inference checkpoint selected by highest fixed-validation tie agreement, then lowest soft-target KL. It intentionally lacks an optimizer and cannot be used as a resume checkpoint.
- `report.json` describes the run. A stop or numerical failure is explicit, not a successful learning claim.

Selection uses a fixed validation-index sample chosen once from a separate generator. Validation always uses **soft-target metrics**, including for one-hot-trained models. Incomplete, time-limited validation cannot select a new best checkpoint. With no completed validation, the inference artifact is explicitly labeled as the last model rather than a validated best model. The test split is not read by the trainer.

Checkpoint loading uses PyTorch `weights_only=True`. A resume must match the training data, validation data, optimizer/sampling settings, validation/checkpoint schedule, stopping configuration, implementation/runtime identity, and source provenance. Array content is hashed for supplied small record sequences; split datasets bind their manifest hash and selected match IDs. The saved state includes Python and NumPy RNGs, the local sampler, Torch CPU RNG, and the selected accelerator's MPS or CUDA RNG. This is a single-device trainer, not a distributed RNG implementation.

## Metrics

`evaluate_model` accepts an explicit index list or a fixed seeded sample. It reports optional-policy cross-entropy, `KL(q || policy)` computed as cross-entropy minus target entropy, tie-set agreement, exact deterministic-choice agreement, and random-legal tie agreement. The latter is the mean `number_of_best_semantic_actions / number_of_legal_semantic_actions`, rather than a guessed constant. Per-action-kind metrics classify rows by the teacher's chosen kind; `per_action_kind["DISCARD"]["tie"]` is optional-discard tie agreement.

Rank metrics include NLL, the summed four-class Brier score, and argmax accuracy. A uniform rank distribution has NLL `ln(4)` and Brier score `0.75`; uniform random tie resolution has expected accuracy `0.25`. A deterministic argmax on uniform logits instead depends on the observed rank frequency, so its measured accuracy is reported separately. Prediction legality is checked against the candidate mask.

These descriptive metrics do not have a match-independent confidence interval. Full-match utility and a prespecified paired comparison remain the route to non-inferiority or strength evidence. High tie agreement alone does not show that the whole target distribution was learned.

## Bounded checks before scaling

`tiny_overfit(dataset, output_dir, ...)` deliberately selects optional discard decisions: half with a unique best action and half with multiple tied best actions but at least one inferior candidate. It records source indices, candidate counts, and tie sizes. It fits this inspected set until soft KL is at most `0.02` and tie agreement is `1.0`, or its step/time limit is reached. A missing target category fails explicitly rather than replacing the check with forced actions or trivial all-candidate ties. This verifies learning machinery; it is not generalization evidence.

`check_resume(records, output_dir, device="cpu", ...)` compares an uninterrupted run with an interrupted-and-resumed run. CPU defaults require identical parameters and loss curves. MPS defaults use prespecified `atol=5e-5`, `rtol=5e-4`; the caller can fix other tolerances before execution. The check records the tolerances and maximum differences. It is scoped to the same runtime/backend/thread configuration, not arbitrary platforms.

`check_cpu_mps_parity(records, ...)` compares the actual model's policy/rank outputs, combined loss, and gradients using the same initial weights and fixture. It reports unavailable MPS as unavailable, never as a passing check. Default tolerances are `atol=5e-5`, `rtol=5e-4`. This single fixture does not establish long-run training equivalence; failed checks must be investigated before scaling.

The unit suite additionally checks forced-row rank gradients, analytically known uniform-prediction metrics, soft validation for one-hot training, checkpoint/schema use, changed-data resume rejection, and explicit resource stops.

## Resource and timing semantics

The per-run clock starts before initialization and includes training, device synchronization, validation, and checkpoints. `elapsed_training_seconds` isolates data sampling/collation and optimizer updates; `elapsed_total_seconds` includes the surrounding run. Limits are cooperative: an in-progress update, validation minibatch, or save can finish after the limit, and final diagnostic checkpoint/report writes are retained. The caller must reserve this small stopping overhead in the cumulative campaign cap.

Memory checks use process RSS plus MPS driver allocation as a conservative sum that can double-count unified memory. Both components and the sum are reported. If current RSS cannot be queried, the process-lifetime peak is the fallback. These sampled checks are not an operating-system hard limit and cannot guarantee catching a short allocation spike. Dataset preload, validation/model-loading work outside `train`, and campaign-level reporting still belong in the caller's resource ledger.

No external data acquisition, paid compute, automatic overnight run, or claim of playing strength is part of this module.
