# Project instructions

- Use English for current documentation and progress reports.
- Follow `docs/IMPLEMENTATION_PLAN.md` and record consequential changes in `docs/DECISIONS.md`.
- Keep `docs/STATUS.md` factual: distinguish implemented, measured, planned, and blocked work. Never describe smoke tests as full rule certification or synthetic compute as playing strength.
- The initial objective is average-rank utility `[1.5, 0.5, -0.5, -1.5]`. A venue-specific utility is a separate explicit experiment.
- Use the project `.venv`. Pin dependencies and record versions with benchmark outputs.
- Policies must consume audited player-visible features. Full simulator state and complete engine logs are auditor/critic-only inputs. Raw upstream observation serialization is not automatically trusted.
- Obtain all simultaneous responses from the same pre-response information state.
- Keep complete-match rewards and per-seat behavior-policy provenance intact. Do not train on frozen-opponent actions as on-policy PPO data.
- Do not infer hand boundaries from round/dealer index alone; dealer continuations must be represented.
- Differential engines require equivalent explicit physical-tile walls and actions, not merely equal numeric seeds.
- Keep research data provenance and permissions explicit. Do not scrape external game corpora, send messages, register bot accounts, or rent compute without the relevant user authorization.
- Maintain a local MPS/CPU path and a portable CUDA path. Benchmark before changing model scale or compute budget.
- Preserve the proposal and historical Chinese translation unless the user requests edits to them.
