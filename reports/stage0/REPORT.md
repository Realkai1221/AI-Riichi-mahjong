# Stage 0 foundation report

16 October 2026. The local simulator and tensor runtime are usable for the next implementation stage. The result is a reproducible foundation, not a trained agent or a certified rules engine.

## Environment and scope

Hardware inventory: Apple M5 Pro, 18 CPU cores, 20 GPU cores, 64 GB unified memory. The installed environment uses macOS 26.6 ARM64, Python 3.14.8, RiichiEnv 0.4.10, PyTorch 2.14.1, NumPy 2.5.4, and pytest 9.1.1. Exact package versions are in [requirements-macos-arm64.lock](../../requirements-macos-arm64.lock). Only prebuilt dependencies were needed; no Rust toolchain was installed.

MPS device visibility is blocked in the restricted sandbox and works in the host execution context. The initial CPU diagnostic records the sandbox context; MPS diagnostic records successful host GPU access. This difference is environmental, not evidence that the hardware lacks MPS support.

The numerical smoke check compares a float32 dense/tanh calculation and its weight gradients with a CPU reference. It passed, with maximum absolute errors below 4e-7 forward and 3e-8 for gradients. This does not replace parity testing of the eventual Mahjong model, legal masks, or PPO update.

## Verification results

Test results: **14 tests plus 17 subtests passed**, with no failures or skips.

Coverage includes:

- Deterministic fresh-environment initialization and action ordering.
- Matching continuation of independent state clones; advancing a clone does not change the original.
- Own-hand visibility and filtering of another seat's private draw.
- Two limited initial-deal information-set invariance families, constructed using complete alternate 136-tile walls: opponent hand/unseen live-wall changes, and another player's private drawn-tile identity.
- Complete-match score plus outstanding-deposit conservation, including a nonzero-deposit fixture.
- Reproducible trajectory digests, bounded truncation, and invalid input handling.
- Completed-match throughput accounting and whole-run seed-range preflight.
- Finite/nonzero initial gradients, changed parameters, and rejection of an intentionally corrupted final optimizer update.

Random-play smoke report: **100 complete hanchans**, seeds 20261016 through 20261115, **1,201 hands**, **106,343 decisions**, **zero truncations**, and **100/100 passing score/deposit audits**. Actions are selected from the engine's legal candidates. This checks integration; it does not independently prove the candidates are correct. Random play rarely reaches many important strategic and rule situations.

## Engine capability audit

| Capability | Evidence and remaining work |
|---|---|
| Full-match mode and legal candidates | Working with explicit `4p-red-half` and `GameRule.default_tenhou()` in the pinned wheel. Validate each semantic rule independently in Stage 1. |
| Clone/snapshot | `clone()` preserves tested continuations and independence. Cost and arbitrary-phase coverage still need measurement. |
| Explicit wall injection | `reset(wall=...)` works for complete initial-deal privacy fixtures. Cross-engine wall/dead-wall normalization remains unimplemented. |
| Seeded restart | Fresh adapter instances reproduce the tested trajectories. Repeated `reset(seed=...)` on one upstream instance retains a hand counter and is not a supported rewind operation. |
| Player information | Canonical snapshots keep a selected whitelist and own hand; a private draw is included only when the physical tile belongs to that hand. Full history/feature contracts remain Stage 1 work. |
| Arbitrary hidden-state construction | Not validated. State setters alone do not establish a legal determinization API. |
| Native multi-table batching | Not implemented in this adapter; the initial loop steps one table at a time. |
| Independent rule conformance | Pending targeted fixtures and independent scoring/transition comparison. |

Two upstream integration behaviors were reproduced locally. First, raw inactive-seat observations contain the current player's `drawn_tile`; the adapter masks this field. Second, resetting a seed does not reset the engine's per-hand shuffle counter; the adapter creates a fresh instance for reproducible restarts. These findings justify keeping a strict boundary between raw simulator objects and learned-policy inputs.

The upstream PPO worker was not imported. Its reward fallback and dealer-continuation hand-boundary logic remain separate audit tasks if that implementation is considered later.

## Compute measurements

MPS results and CPU reference use the same synthetic residual CNN: **4,786,223 parameters**, float32, 64-by-34 input, 12 residual blocks of width 256, and provisional 46-way/value heads. These are benchmark dimensions, not a gameplay representation. Every configuration has five warm-up steps and fifteen synchronized timed iterations, plus an untimed update validation.

| MPS batch size | Inference examples/s | Training examples/s | Median inference batch latency |
|---:|---:|---:|---:|
| 1 | 540 | 206 | 1.72 ms |
| 32 | 8,171 | 2,577 | 3.91 ms |
| 128 | 10,382 | 3,509 | 12.32 ms |

Training includes forward, loss, backward, and AdamW update. Inputs and targets are already on the selected device. Before and after timed training, checks confirm finite gradients/parameters; the initial step also confirms nonzero gradients and an actual parameter change. MPS process peak resident memory was approximately 502 MiB; endpoint driver allocation was approximately 247 MiB. These are different memory measurements and must not be added or described as a full training memory requirement.

These short probes exclude simulation, feature encoding, transfers, replay handling, PPO bookkeeping, and evaluation. They do not measure sustained thermal throughput or predict time to competitive strength. Their practical conclusion is to retain a compact batched MPS learner as the initial path and measure the complete pipeline before choosing a larger model or remote compute.

## Next stage

Stage 1 builds rule fixtures, independent scoring checks, a complete player-visible encoder/action adapter, heuristic opponents, and paired full-match evaluation. Large-scale training waits for those interfaces and checks. Current progress is maintained in [STATUS.md](../../docs/STATUS.md).
