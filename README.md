# AI Riichi Mahjong

A research project for a four-player Japanese Riichi Mahjong agent. The simulator, policies, feature encoder, data collection and training code, tests, and research reports are included. The project optimizes average placement across complete hanchans under a fixed ruleset.

In Stage 3J, the learned tie-breakers averaged rank 2.5049 over the fixed panel, compared with 2.5273 for the original reference and 2.5000 for a nonlearned tie-breaker. The experiment covered 8,192 games, with 256 independent wall blocks. Its four adjusted intervals included zero. [Stage 3J results](reports/stage3j/REPORT.md) and [experiment review](docs/STAGE3J_REVIEW.md).

## Run on macOS ARM64

The recorded setup used Python 3.14.8 and the pinned macOS ARM64 dependencies. Start with CPU for the portable smoke run. The agent CLI also exposes MPS and CUDA backend checks.

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements-macos-arm64.lock
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
.venv/bin/python -m riichi_ai doctor --device cpu
.venv/bin/python -m riichi_ai smoke --games 2 --seed 123
.venv/bin/python -m riichi_ai evaluate --candidate defense --baseline shanten --blocks 2 --seed 124
```

Run the test suite from the checkout root. The historical continuation verification requires the original archive and skips when its inputs are unavailable. Tests that require a tensor backend skip when that backend is unavailable.

```sh
.venv/bin/python -m pytest -q
```

## Research files

- Research direction and background. [Original plan](RIICHI_AI_RESEARCH_PLAN.md) · [Chinese translation](RIICHI_AI_RESEARCH_PLAN_ZH_CN.md)
- Implementation milestones and decisions. [Current status](docs/STATUS.md) · [Implementation plan](docs/IMPLEMENTATION_PLAN.md) · [Decision log](docs/DECISIONS.md)
- Playing and learning interfaces. [Player view](docs/PLAYER_INTERFACE.md) · [Data pipeline](docs/DATA_PIPELINE.md) · [Training](docs/TRAINING.md) · [Evaluation](docs/EVALUATION.md)
- Latest game experiment. [Stage 3J protocol](docs/STAGE3J_PLAN.md) · [Machine-readable statistics](reports/stage3j/summary.json) · [Evidence audit](reports/stage3j/audit.json)
- Book scenario tooling and a synthetic example. [Scenario format](docs/BOOK_SCENARIOS.md) · [Example input](examples/book_scenarios/example.jsonl)

## Snapshot scope

This GitHub snapshot comes from local research commit `bf3be0886a6f`. The project archive retains original Git history, model weights, generated corpora, detailed engine logs, and book material. The included reports record prior measurements. Seed-only records keep the early data generators from reusing recorded match seeds. Audit hashes refer to the archived files, and some original report references appear as plain text when the target is in that archive.

Historical campaign scripts and configurations describe the experiments that ran. Their machine-specific paths use placeholders such as `/Users/example/`. Re-running those campaigns requires the corresponding input checkpoints or other archived artifacts, real local paths, and fresh output directories. The CLI commands above run from a fresh checkout and need only the installed dependencies.
