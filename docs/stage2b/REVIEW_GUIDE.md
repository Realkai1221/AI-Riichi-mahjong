# Reviewing Stage 2B

Start with the measured [Stage 2B report](../../reports/stage2b/REPORT.md). Its four headline findings are:

- Exact features improved full-validation discard agreement to 100% on both seeds, versus 73.14% and 64.23% for the raw controls.
- The frozen primary reached 99.9826% agreement on the untouched action test, with two +1-shanten errors.
- The original retention study was inconclusive under its 256-block cap. The separately authorized 319-block Package A1 confirmation is now complete; retention was not demonstrated. Keep these two studies distinct.
- Both rank-zero continuations deteriorated late; the selected earlier checkpoints do not demonstrate durable training. Placement prediction remains poorly calibrated.

A high action-agreement percentage alone is not a strength certificate. [Package A](../PACKAGE_A_PLAN.md) separated a ten-minute confirmation ceiling from a 45-minute, four-attempt stability study. It is now complete; see its [measured results](../../reports/package_a/REPORT.md) and [current review guide](../PACKAGE_A_REVIEW_GUIDE.md).

## What to inspect

| Question | Evidence | Interpretation |
|---|---|---|
| Did exact features help under matched conditions? | Two-seed full-validation table, shared-weight and sample-stream hashes | Compare assisted rank-0.1 against raw rank-0.1 at the same update count. Historical Stage 2A is background. |
| Did the rank loss help? | Assisted rank-0.1 against assisted rank-0, across both seeds | The primary comparison holds features and initial shared weights constant. A zero-weight rank head is not a trained critic. |
| Did the model merely fit the selected sample? | Full validation, frozen action test and error-severity counts | Test performance is opened only after the final primary checkpoint is frozen. |
| Did it retain the teacher's playing strength? | Final model's pilot, saved confirmation plan and any fresh confirmation | Development results and action agreement do not substitute for the fixed-budget retention test. An inconclusive result remains inconclusive. |
| Is local operation practical? | All-neural worker sweep and sustained benchmark | Audit overhead is included. CPU inference throughput is not a self-play RL training-rate estimate. |
| What remains outside the evidence? | Coverage and limitation sections | In particular, offered calls are not executed-call coverage; this stage does not establish world-class strength. |
| Was the approved scope respected? | Budget ledger, resource reports and final inventory | Check actual measured costs, incomplete/failed attempts, total new corpus count and any conditional allocation. |

## Files that control the work

- [Accepted plan](PLAN.md) and [executable configuration](../../configs/stage2b.json): scope, seeds, limits and selection rules.
- [Model contract](MODEL.md): actor-only inputs, old checkpoint compatibility and the exact feature branch.
- [Frozen model card](MODEL_CARD.md): selected checkpoint, measured behavior and limits on its use.
- [Data contract](DATA.md) and [collection observations](DATA_OBSERVATIONS.md): actual behavior versus teacher labels, splits and coverage.
- [Training protocol](TRAINING.md): matched initialization, sampling, eligibility and budget accounting.
- [Running guide](RUNNING.md): reproduction, runtime limits and the proposed routine for later longer runs.

The original proposals and historical evidence are preserved. `scripts/verify_stage2a_snapshot.py` and `scripts/verify_stage2b_snapshot.py` check their completed project snapshots and unchanged bulk artifacts. The older Stage 2B current-tree verifier is not appropriate after current source or documentation evolves; do not regenerate its historical inventory.

## Useful directions for the next instruction

The [first next-review proposal](NEXT_REVIEW.md) remains historical. The user subsequently authorized the revised [Package A](../PACKAGE_A_PLAN.md), including the completed 319-block confirmation. Its result cannot be changed by extending the sample after outcomes.

The next request can specify the scientific question, an additional local compute budget and the required acceptance evidence. The Package A review guide recommends a 600-second diagnosis-only follow-up. Keep any expansion explicit: open-hand/call coverage, reinforcement learning, longer histories, external data and paid compute are different changes and should not be bundled into an unexplained performance claim.

The present follow-up authorization ends with Package A results and review files. Reinforcement learning, public play, external contact and overnight runs remain outside that scope.
