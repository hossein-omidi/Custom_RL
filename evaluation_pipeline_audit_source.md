# Evaluation pipeline audit — PPO face-milling project

## Scope

Reviewed and revised `eval_policy.py` and `plot_results.py`, with compatibility checked against the unified PPO trainer's output layout (`models/`, `logs/seed_<n>/training_config.json`, `evaluation/`). The training process itself is not modified by these two files.

## Critical corrections

1. **Learning-curve x-axis:** removed the reconstruction of global environment steps from parallel Monitor episode lengths and worker-relative wall-clock timestamps. The learning curve now reads SB3 `evaluations.npz` and uses its actual `timesteps` values.
2. **Statistical labels:** replaced the ambiguous `±3σ (~95%)` presentation with an explicitly labelled **mean ±1 sample SD**. The learning-curve band is across seed-level evaluation means; trajectory bands are across MC replicas within one test episode. Neither is labelled as a confidence interval.
3. **Scenario mixing:** trajectory plots are now generated separately for each training seed and test episode. MC replicas are averaged only after verifying that they share the same milling-line position and, in finishing mode, the same axial depth of cut.
4. **Finishing-mode parity:** evaluator defaults now load `env_id`, reward, time step, substeps, episode cap, disturbance settings, and `ap_max` from the unified trainer's per-seed `training_config.json`. Finishing defaults to `dense` when no saved config is available; roughing defaults to `productive`.
5. **MC scenario control:** evaluation reset seeds are independent of training-seed IDs and shared across evaluated policies. For each outer test episode, `y0` and finishing `ap` are fixed across its MC replicas. If `ap` is not supplied in finishing mode, one value is sampled per episode from the configured range.
6. **Checkpoint compatibility:** evaluator checks model/environment observation and action spaces, checks the configured finishing `ap_max` and episode cap, and refuses checkpoints detected as using `VecNormalize` because this raw-observation evaluator cannot restore their matching normalization statistics.
7. **Signal/action timing:** action timestamps are saved at action-application start; physical sensor/process timestamps remain post-step. Plotting uses the separate action timestamps.
8. **Artifact integrity:** non-finite values are converted to JSON `null` and written with strict JSON validation. Evaluation outputs include checkpoint paths/hashes and a manifest. Existing results are not silently overwritten; use a new output directory or `--overwrite`.
9. **Run-directory inference:** `plot_results.py --traj-dir <run>/evaluation` automatically finds the sibling `<run>/logs` and writes plots to `<run>/plots`, matching the unified trainer's printed command pattern.

## Recommended use

From the repository root, after the desired checkpoint exists:

```bash
python scripts/eval_policy.py \
  --model-dir "runs/ppo/<run-name>/models" \
  --out-dir "runs/ppo/<run-name>/evaluation" \
  --seeds 0 1 2 \
  --n-episodes 5 \
  --n-mc 5

python scripts/plot_results.py \
  --traj-dir "runs/ppo/<run-name>/evaluation"
```

For finishing mode, the unified run config supplies the mode, reward and `ap_max` automatically. If you want a fixed finishing depth, add `--ap <depth_mm>`. If you omit `--ap`, the evaluator samples one depth per outer episode and holds it fixed across that episode's MC replicas.

If the evaluation directory already contains trajectory files, choose a new `--out-dir` or explicitly pass `--overwrite`.

## How to interpret the figures

- `learning_curve.png`: mean evaluation return at the actual EvalCallback training timesteps. The shaded band is ±1 sample SD across seed-level evaluation means. The EvalCallback environments are stochastic, so this between-run spread also includes evaluation-episode variability; it is not a pure estimate of optimizer/training randomness and is not a 95% CI.
- `trajectory_*_seed<n>_episode<m>.png`: mean ±1 sample SD across MC replicas for that one test episode, conditional on its fixed `y0` and (for finishing) fixed `ap`. The band includes the stochastic/initialization channels enabled in the evaluation environment. With `n_mc=1`, the figure is a single rollout and has no SD band.

The final evaluation uses the same evaluation seed sequence across training seeds. This makes reset scenarios reproducible and improves comparability, but identical seeds alone do not guarantee perfectly time-aligned common disturbance forcing if controllers produce different episode lengths. Strict common-random-number comparisons would require pre-generated disturbance sequences indexed by physical time.

## Verification performed here

- Python syntax compilation passed for both revised files.
- Synthetic tests passed for strict JSON conversion, observation fallback scaling, config resolution, finishing `ap`/`y0` scenario locking, MC scalar aggregation, VecNormalize detection, and model/environment space checks.
- Synthetic plotting tests passed for `evaluations.npz` learning curves, per-episode MC sensor/action/process plots, and run-directory inference.
- Synthetic end-to-end evaluator runs passed for both roughing and finishing modes, including manifest and trajectory JSON generation.

**Not performed:** a real rollout against the project's actual `CustomODEPlate-v0` / `CustomODEPlateFinish-v0` implementation. The sandbox did not contain the full `custom_rl` package and its runtime dependencies, so real plant integration, numerical trajectories, and checkpoint loading must still be smoke-tested inside the user's project environment. The synthetic tests verify script logic, not physical-model correctness.
