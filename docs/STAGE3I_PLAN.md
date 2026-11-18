# Stage 3I: frozen-reference residual learning

22 October 2026. The user authorized continuation after the Stage 3I review and asked for more ambitious training. This executable programme supersedes the review's then-pending authorization; it preserves the submitted proposal and all historical results.

## Scope and fixed decisions

The original Stage 2B actor, including its embeddings, efficiency encoder, backbone, action scorer and rank head, remains frozen and hash-verified. We train three independent seeds (20261024011, 20261024012, 20261024013). Each uses all 3,200 Stage 3G training matches and their 471,627 optional explorer decisions. This is offline supervised prediction of archived one-hand placement-potential differences, not PPO, causal advantage estimation or a claim of stronger play.

A state value head V (256–64–1) is fitted and selected first, then frozen. An action-conditioned residual head A (384–64–1) and a state-only residual control B (256–64–1) independently predict the same residual y−V. All output layers initialize to zero. Adam uses learning rate 0.0003, gradient norm cap 1, batch size up to 1,024, and residual ridge 0.01. The action ridge averages squared predictions over all legal actions within each decision; B uses its squared state prediction. Training losses average decisions. Validation reports both decision and whole-match means. The surrogate assigns the same label to decisions within a hand and may include earlier events; it is not a decision-specific return.

The measured throughput preflight selects the largest feasible fixed count from 12, 8 or 4 full passes per fitting phase, before any formal training outcome. There is no adaptive extra training. Each epoch is saved with optimizer and source identities. V, A and B independently select their lowest selection-set prediction MSE, breaking exact ties by earlier epoch. All three seeds are retained, with no winner chosen from game results.

## Data and inference boundaries

Only audited player-visible actor tensors enter the frozen representation. Cache features use CPU batch-one inference for exact deployment parity. Recorded outcomes and behavior provenance remain separately named labels; the heads consume only explicitly selected state/candidate arrays. The existing complete-match rank labels and explicit hand ordinals are preserved. No book answers or new external corpus enter this campaign.

The existing 800 development matches are sorted by match ID and divided into 200 checkpoint-selection, 200 correction-calibration and 400 predictive-gate matches. These are match-disjoint internal partitions of previously used development evidence, not a newly untouched test set.

Eligibility uses the exact archived explorer distribution, reconstructed from the frozen reference, original call heads and 5% semantic-uniform exploration. Every cached probability vector must exactly equal the recorded vector. A legal action is eligible when q≥0.02, or when it is the reference's canonical argmax. This is a behavior-probability restriction, not demonstrated empirical coverage.

For eligible actions the acting score is max(z_ref−max_legal z_ref, −8) + lambda*A. Ties use canonical legal order. Both raw and wrapped zero-correction identity are required. The wrapper always chooses the first legal win when available; the primary control uses the identical wrapper.

## Predictive gate and calibration

On the 400 gate matches, the selected V+A must have strictly lower decision-mean squared error than both V and V+B. Match-mean errors and slices are diagnostic. This weak predictive gate qualifies an experiment; it does not establish causal action value or strength.

On the separate 200 calibration matches, use lambda in [0,1,2,4,8,16,32,64,128,256,512,1024]. Select the value yielding the greatest observed wrapped change rate subject to at most 3% of all optional decisions and at most 10% of reference-discard decisions; ties choose smaller lambda. No minimum change rate is required. A zero-lambda seed is not admitted as a distinct candidate. Report per-kind and available public-context slices. Calibration is fixed before any game outcomes.

## Fresh local game panel

Use 470 fresh physical-wall blocks and all four focal-seat rotations. Each block includes raw reference and wrapped reference, plus every seed passing the declared predictive/calibration checks. No seed is selected after seeing game outcomes. Include reset-raw and reset-wrapped controls on the first 64 blocks for the initialization/wrapper diagnostic. Maximum formal workload is 9,912 complete games; two preflight blocks with all seven policies add at most 56, leaving 32 attempts within the 10,000 campaign ceiling for independent replay if admitted. Started/failed games count and are never silently replaced.

Policies face the fixed shanten, defense and original-reference opponent panel. Obtain simultaneous responses from the same pre-response state. Pair using explicit physical-tile walls by hand ordinal, including dealer continuations. Report each candidate against wrapped reference primarily, raw reference secondarily; report the equal-seed block contrast when defined. Bootstrap whole wall blocks (5,000 fixed replicates); distinguish descriptive intervals and multiplicity-adjusted bounds. The 470-block count is a fixed exploratory sample, not a guaranteed-power claim. No policy promotion occurs here; any encouraging result needs fresh confirmation.

The game panel executes only if measured full-deployment throughput and artifact forecasts fit. The default controls and reset diagnostic still execute if all candidates fail their gate, provided admission passes. Failure never automatically expands collection or reruns training.

## Resource ceiling and admission

All numerical work, tests, feature preparation, failed attempts, training, calibration, games, statistics and independent audit are charged by one exclusive Stage 3I ledger. The ceiling is 5,400 seconds of monitored numerical execution, not a promise about conversation wall time. Phase caps: preflight 450; feature preparation 600; training 1,800; calibration 300; evaluation 1,800; audit 300; protected shutdown reserve 150. Admission protects subsequent phases. Frozen source hashes are checked throughout each operation, with process-tree shutdown on failure.

Use project .venv, record exact versions, maintain CPU/MPS and portable CUDA code paths. Benchmark actual optimizer steps and deployed games, not synthetic matrix multiplication. Process-tree RSS cap is 32 GiB; preserve 10 GB free disk. New Stage 3I artifacts are capped at 8 GB, increased from the proposal's 4 GB to accommodate exact frozen features and the larger evaluation; benchmark size before formal preparation. Bulk artifacts remain outside iCloud under Application Support/RiichiAI/stage3i. No paid compute, new training games, external scraping, public bot accounts or policy deployment is authorized by this programme.

If admission or a formal execution fails, preserve evidence, stop dependent work, and finish the audit/report. Benchmarks and implementation fixes may be repeated within the preflight cap with separate receipts. Historical checkpoint and corpus identities must remain unchanged. Final deliverables include measured results, accounting, immutable inventory, limitations and a proposed next decision.

## Independent audit sample

Audit every cache shard's explorer selection, recorded probabilities and target provenance, all epoch/checkpoint selection and exposure records, all saved calibration/gate arithmetic, and the complete game panel's result and wall pairing. Additionally re-forward 64 cached rows selected by ascending SHA-256 of `stage3i-cache-audit|match_id|row_index`, without outcomes, and replay the first 32 formal games under ascending SHA-256 of `stage3i-audit-replay|match_id`. Replay attempts consume the global game ceiling. These checks establish the tested implementation and same-engine reproducibility scope, not independent full-rules certification.
