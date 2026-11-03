# Changelog

## v2 scientific re-audit

- Trainer: correct stale rollout-size comment, retain default PPO and checkpoints, add opt-in `--callback-eval-seed` for a shared initial EvalCallback RNG seed and record its usage in training metadata.
- Evaluator: full-step metrics + output-trajectory sampling, strict time/reward/sensor checks, physical termination guard, training-condition consistency, safe staged outputs, CSV outputs at replica and scenario levels, physical force and control variation metrics, measured inference latencies.
- Plotter: timestamp-aligned continuous trajectories, piecewise-constant actions with final hold, SD undefined when fewer than two MC replicas remain, original-sample-first RMS envelopes, explicit finishing fixed-depth labeling, PDF/300dpi trajectory figures, grouped physical-outcome figure with sample-SD bars and scenario matching guard.
- Tests: dependency-light synthetic end-to-end and plotting tests plus numerical plotting/alignment tests; no physics claim.
