# Package A diagnosis plan

Authorized 18 October 2026 after the user accepted the [proposed diagnosis](PACKAGE_A_REVIEW_GUIDE.md). This follow-up examines existing checkpoints and data. Its ceilings are **600 cumulative numerical seconds, 32 GiB sampled memory, and 250 MB of new artifacts**, with zero optimizer updates, new matches, or test-split loading. The [configuration](../configs/package_a_diagnosis.json) fixes the scope and checkpoint hashes.

## Fixed comparison

Evaluate seven checkpoints in the declared order: the original selected rank-0.1 reference; the two stopped rank-zero controls; and selected plus terminal smoothed checkpoints from each seed. These are descriptive endpoint comparisons. The original reference is not a matched smoothing control, and an early stopped control is not an observed collapse.

Freeze 10,000 distinct uniformly sampled training-row indices using seed 2026101801. The original 490,206-row training corpus contains teacher-generated positions only. Compare its fit with the teacher-generated validation stratum; report the student-generated stratum separately as a transfer population. Use all 16,024 validation rows. Rows share complete matches, so do not treat them as independent observations for significance tests.

Use the same 64 fixed validation positions as Package A's diagnostics: 32 per stratum, selected by its independent diagnostic RNG. Save actual indices, data identities, code hashes and checkpoint hashes before loading any model outputs. Load only the training and validation splits; prior manifests may name test artifacts, but this study must not load their arrays or predict their actions.

## Measurements

Measure cross-entropy and KL against both original teacher targets and epsilon-0.01 smoothed targets. Report best-set agreement, exact action agreement, shanten and ukeire errors, probability mass, entropy, raw/centered logits, row offsets and context norms. Separate forced decisions, optional teacher discards, teacher/student strata, legal-choice counts, and call/kan availability. Every mean retains its actual numerator and denominator. Compare observed non-best mass with the exact cohort target mean, since all-best or forced rows have zero non-best target mass.

Float32 losses/probabilities preserve actual backend arithmetic. Separately normalize the identical stored float32 logits in CPU float64, with copied targets normalized to unit mass for that KL calculation. This diagnoses normalization and underflow; it cannot recover upstream score differences or change the raw-logit argmax.

On the 64 positions, compare the unchanged canonical forward in CPU float32, MPS float32 and CPU float64. Float64 widens the saved weights and input features; it does not recreate high-precision training. Record raw arrays, action-order differences, best-set membership, margins and probability differences. Predictions that switch between equally good teacher actions must be distinguished from new teacher errors. Exclude padding throughout. Verify that inference preserves caller/model state, RNG, checkpoint files and source hashes.

Compare endpoint diagnostic scales with the original reference and the old control guard references. Threshold exceedance at an endpoint is not a sustained alarm, and endpoint comparisons cannot estimate prospective predictive value or false-positive rate. Restrict numerical conclusions to the tested positions and arithmetic.

## Accounting and execution

The controller reuses the tested persistent ledger and process watchdog with a separate identity and data root. Lease acquisition precedes numerical imports, data/model loading, tests and inference. Default allowances are 90 seconds for checks, 60 for preparation and at most 60 per checkpoint, retaining 30 seconds of cumulative reserve. Preparation freezes rows before model outcomes. Formal checkpoint markers are exclusive and ordered; a failed or partial checkpoint attempt is preserved, not silently retried or replaced.

Run focused inference-only tests, not historical training tests that perform optimizer updates. All numerical checks, failures, CPU/GPU forwards, serialization and finalization count. Source editing, document preparation and model-free hash checks are outside the numerical budget. Keep new compact arrays outside iCloud at the configured path and reports under `reports/package_a_diagnosis`. No historical inventory or checkpoint is rewritten.

## Decision after diagnosis

If teacher-training and teacher-validation both miss the target materially, investigate inadequate fitting or optimization before claiming a validation-only generalization failure. If training closely fits while teacher validation does not, investigate the generalization gap. Student-only degradation points to transfer weakness. These comparisons narrow hypotheses; they do not prove a unique cause or establish playing strength.

The final report should nominate one next training intervention with a fresh protocol, rather than combine changes. This authorization ends with diagnosis, accounting and review files. It does not launch that intervention, expand call coverage, resume retention testing or begin reinforcement learning.
