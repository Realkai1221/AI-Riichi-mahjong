# Stage 2B matched training protocol

The three required recipes are the assisted actor with placement loss weight 0.1, the raw actor with weight 0.1, and the assisted actor with weight 0. Both declared seeds run every recipe. The initial ceiling is 20,000 updates; representative measured costs can reduce a single common ceiling **before any formal attempt**. An incomplete comparison cannot qualify as a complete recipe.

`matched_initialization` copies every shared tensor exactly from the raw model into the assisted model for each seed. Its manifest records configurations, hashes and the independently seeded additional branch. The two assisted rank settings start from identical complete weights. The policy sampler uses the same NumPy PCG64 stream across matched runs; checkpoints retain a digest of every sampled index sequence. Model schemas and dataset identities differ where intended. A rank weight of zero disconnects the placement loss from backpropagation, so AdamW does not decay an otherwise unused placement head.

Policy cross entropy uses optional decisions and the audited soft tie targets. Placement cross entropy includes all rows. Teacher preference tuples remain outside `encoded` and are used only for loss labels and diagnostic calculations. Actor inputs never contain teacher preferences or terminal ranks. Full-state placement NLL, Brier score, accuracy and ten fixed top-confidence reliability bins are descriptive; correlated decisions do not supply independent strength evidence.

Checkpoint selection uses one fixed validation sample, up to 5,000 rows from each teacher-focal and student-focal population. Its seed is independent of training seeds. The lexicographic score is equal-population mean optional-DISCARD best-set agreement, then negative shanten-worsening rate, then negative optional-policy KL. Rates within each population are decision weighted. Shanten and ukeire errors refer to two DISCARD candidates and the teacher's canonical preferred candidate; wrong action kinds are reported separately. Red-only errors and integer severity histograms are retained. These diagnostics describe teacher imitation, not guaranteed play quality.

After each run, evaluate its selected checkpoint on full validation and a fixed frozen-model training sample. Recipe ranking averages each selection metric across both seeds. Exact ties follow the declared recipe order. The final checkpoint always uses the lower, declared primary seed of the winning recipe, including when the other seed scored better. Freeze that file and its hash before test predictions or development match outcomes. A failed optional recipe is excluded; failures in the six required runs leave the base comparison incomplete.

The conditional allocation first checks whether the best feature-assisted recipe's student-focal discard error exceeds teacher-focal error by more than two percentage points, averaged across seeds. If so, a frozen primary-seed actor may collect the separately authorized student corpus. Each additional minibatch then samples exactly 128 original rows and 128 student rows, uniformly with replacement within each source, with a paired shuffled order. The mixture binds both complete dataset identities. If the gap trigger is absent, the alternative ukeire trigger requires more than 2% on equal-stratum full validation and more than 1% on fixed training samples, plus passed alignment/tiny-fit checks. A remaining-hand encoder requires its separate frozen implementation before those two runs. The triggers allocate experiments; they are not significance tests. Neither optional route runs unless both seeds fit the remaining declared budget.

## Commands and artifacts

Run with the repository's pinned `.venv`, from the repository root. These commands are explicit stages and do not run merely by importing the script.

```bash
.venv/bin/python scripts/stage2b_train.py checks --device mps
.venv/bin/python scripts/stage2b_train.py plan
.venv/bin/python scripts/stage2b_train.py campaign --device mps
.venv/bin/python scripts/stage2b_train.py contingent
# Complete a triggered, affordable conditional experiment before selection.
.venv/bin/python scripts/stage2b_train.py select
.venv/bin/python scripts/stage2b_train.py test --device mps
.venv/bin/python scripts/stage2b_train.py budget
```

`campaign` starts one fresh subprocess for each of the six required runs. For individually scheduled runs use `run --recipe features_rank01 --seed 202610171 --device mps`. The optional aggregation recipe uses the name recorded in `contingent-plan.json`, the same two seeds, and the same common update ceiling. Its source directories are `aggregation` and `derived-aggregation` under the Stage 2B data root.

The checks include actual audited v2 tiny-fit states, forward/gradient CPU–MPS parity, CPU and MPS resume checks, representative batch-256 raw/assisted optimizer probes, fixed-sample validation, full validation and fixed training-sample timings. MPS unavailable means unavailable, never a fabricated pass. These probes train only disposable numerical-check models; they are not eligible formal runs. The plan uses the slower measured variant, a 20% margin, measured loading/evaluation overhead, conservative checkpoint overhead and at least 120 seconds reserved for final work.

Reports live under `reports/stage2b`; bulk models and the persistent ledger live under the configured local data root. `learning-checks.json`, `training-plan.json`, `run-RECIPE-SEED.json`, `contingent-plan.json`, `selection.json` and `action-test.json` preserve the decisions. Formal reports/checks/final selection refuse replacement. Progress is printed at each fixed validation interval. Every model has a last resumable checkpoint and a validation-selected checkpoint. Strict resume verifies model/features, actual implementation hashes including efficiency calculations, data and validation identities, initialization, sampler state, evaluator protocol, cadence and stopping settings.

## Resource accounting

`BudgetLedger.operation` reserves an allowance, measures initialization/loading/optimization/evaluation/save time, and records successful and failed attempts. An interrupted process with no completion leaves its entire reservation charged. No automatic refund or budget extension occurs. The ledger identity is the SHA-256 of the exact main configuration bytes. External measured work can be recorded with:

```bash
.venv/bin/python scripts/stage2b_train.py account --name feature-derivation --seconds 123.4 --evidence reports/stage2b/feature-audit.json
```

Add `--failed` for a failed attempt. This records already-incurred costs; it does not authorize extra work. Prefer wrapping future work in `BudgetLedger(...).operation(...)` so an interrupted operation remains accounted for. Avoid double charging work that already owns a lease.

For a command that does not own a ledger lease, `budgeted --name LABEL --max-seconds 300 -- COMMAND...` measures its startup and execution, preserves failed durations, and terminates its process group if the allowance expires. It retains five seconds for termination/bookkeeping and leaves the default 120-second final-work reserve. Planning and selection also own measured leases. The checks include a historical-checkpoint severity report on the 100 student diagnostic matches; this is descriptive and is not used as the matched raw-model control.

The one-hour ledger includes feature derivation/audit, learning checks, initialization/loading, all optimizer steps, validation/selection, action-test evaluation and failed attempts. Collection, complete-match evaluation and the separately capped runtime benchmark are reported separately. Non-learning implementation tests are outside the learning budget. RSS plus MPS driver allocation is a deliberately conservative sampled measure and may double-count unified memory; it is not an OS-enforced cap. Time and disk checks are cooperative; an in-flight minibatch or atomic checkpoint may finish after its allowance, and actual overruns remain recorded. A resource-limited comparison is explicitly incomplete.
