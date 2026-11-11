# PPO Fast / Balanced / Precise Sweep — Additive Patch

This patch adds a reproducible comparison of three PPO training profiles to the `mpc-correction` branch. **It does not replace any existing repository script or module.** Copy only the seven new files listed below.

## 1. Files added

```text
configs/ppo_sweep/fast.yaml
configs/ppo_sweep/balanced.yaml
configs/ppo_sweep/precise.yaml
scripts/sweep_ppo.py
scripts/plot_ppo_sweep.py
tests/test_ppo_sweep.py
README_PPO_SWEEP.md
```

Keep the existing `scripts/experiment.py`, `scripts/train_ppo_unified.py`, `scripts/eval_policy.py`, `scripts/plot_results.py`, `scripts/run_paths.py`, and all `custom_rl/` modules unchanged.

## 2. PPO profiles

| Hyperparameter | Fast | Balanced | Precise |
|---|---:|---:|---:|
| Initial learning rate | 3.0e-4 | 2.5e-4 | 1.5e-4 |
| Final learning rate | 3.0e-5 | 2.5e-5 | 1.5e-5 |
| `n_steps` per environment | 1,024 | 2,048 | 4,096 |
| `n_envs` | 3 | 3 | 3 |
| Rollout size (`n_steps × n_envs`) | 3,072 | 6,144 | 12,288 |
| Minibatch size (`batch_size`) | 512 | 256 | 256 |
| PPO epochs (`n_epochs`) | 4 | 8 | 10 |
| Initial clip range | 0.20 | 0.20 | 0.15 |
| Final clip range | 0.10 | 0.10 | 0.075 |
| Discount factor (`gamma`) | 0.999995 | 0.999995 | 0.999995 |
| GAE lambda | 0.9995 | 0.9995 | 0.9995 |

Balanced preserves the current `mpc-correction` branch's exposed PPO baseline. The trainer's other PPO settings remain fixed across all profiles: entropy coefficient 0.002, value-function coefficient 0.5, gradient-norm limit 0.5, target KL 0.02, actor/critic network widths `[256, 256]`, initial policy log standard deviation -2.5, and Adam epsilon 1e-5. These values are retained in the trainer and recorded in each seed's `training_config.json`.

The three profiles are **bundled hyperparameter alternatives**. They are intended to compare training effort and learning behaviour; they do not isolate the causal effect of any single hyperparameter. Fast reduces nominal optimizer work per collected transition. Precise uses longer rollouts, more epochs, a lower learning rate, and a narrower clipping range. None is guaranteed to converge better or produce a better policy.

## 3. Matched training conditions

Defaults in each configuration:

- Training seeds: `0, 1, 2` (three independent training runs per profile).
- Vectorized environments: 3 `SubprocVecEnv` workers, using `spawn`.
- RK4 integration step: `1e-4 s`; 10 substeps; environment/controller sample time: `1e-3 s` when controller hold is 1.
- Dynamics-uncertainty standard deviation: 0.001.
- Randomized initial transverse cutting position: enabled.
- Evaluation episodes per callback: 10.
- Callback evaluation seed: 10000, common across training seeds and profiles.
- Requested evaluation frequency: 50,000 aggregate PPO transitions. With 3 workers, the current trainer converts this to 16,666 vector calls, i.e. 49,998 aggregate transitions.
- Requested checkpoint frequency: 200,000 aggregate transitions. With 3 workers, the actual interval is 199,998 transitions.

The same seeded evaluation episodes are replayed at each checkpoint and across training seeds. Each seed's evaluation episodes are averaged first; the shaded band is then calculated as **mean ± 1 sample standard deviation across independent training seeds**. This is not a standard deviation across evaluation episodes or a confidence interval. The fixed callback set is a paired screening set for checkpoint selection, not an independent test set. Use the existing `experiment.py eval` workflow with a larger independent evaluation set for final performance claims. With only three training seeds, the SD is descriptive; five or more seeds are preferable when computationally feasible.

### Equal transition budget

The rollout sizes are 3,072, 6,144, and 12,288 transitions. Their least common multiple is 12,288. The default budget is 1,007,616 aggregate PPO transitions **per training seed**, which is an exact multiple of all three rollout sizes. Each profile therefore receives 3,022,848 aggregate transitions across its three seeds. This avoids different rollout-related budget overshoots between profiles.

The nominal minibatch-update counts per seed are:

- Fast: 7,872.
- Balanced: 31,488.
- Precise: 39,360.

These are upper bounds. Stable-Baselines3 PPO can stop an epoch early when the approximate KL divergence exceeds `1.5 × target_kl`.

## 4. Configuration and implementation boundary

The sweep uses the existing `experiment.py` configuration resolver and training command. Although the config flattener lifts nested `timing`, `train`, and `noise` keys, the current trainer CLI exposes only a subset of them. In this sweep, the effective stochastic settings passed to the trainer are `dynamics_uncertainty_std` and `randomize_y0`; `process_noise_std`, `obs_noise_std`, `initial_eta_std`, and `initial_etad_std` in `base.yaml` are not forwarded by the current CLI. The complete nested `plant:` and `reward:` dictionaries are also **not** forwarded as arbitrary environment kwargs. Effective plant defaults come from the environment registration, while reward coefficients come from `REWARD_KWARGS` in `train_ppo_unified.py`. The mode-specific reward selector and finishing `--ap-max` are passed through the existing CLI.

Accordingly, the runner does not claim that every inherited YAML field is active. After each training run, it checks the actual `training_config.json` and `run_manifest.json` files, verifies the effective environment signature across seeds and profiles, verifies the PPO parameters actually used, checks the full transition budget and evaluation checkpoint schedule, and validates the requested finishing `ap_max` against the recorded arguments, environment kwargs, and plant probe. An invalid or incomplete run is not marked as a completed sweep. Always use each run's `training_config.json` as the record of the effective settings.

The sweep configuration sets `plant.ap_max: null` to prevent the inherited 12 mm base metadata from becoming a top-level roughing `--ap-max` argument. Finishing receives the explicit `--ap-max` argument (default 2.0 mm).

## 5. Critical existing Mode II reward inconsistency — resolve before finishing training

The current branch contains a reward-weight conflict that this additive sweep patch deliberately does not change:

- `custom_rl/envs/registration.py` registers the finishing environment with `DEFAULT_FINISH_PRODUCTIVITY_WEIGHT = 40.0`.
- `scripts/train_ppo_unified.py` applies the global `REWARD_KWARGS` to both `productive` and `dense` modes. That dictionary sets `productivity_weight = 240.0`; the finishing branch then overrides `ap_max` but does not restore the finishing-specific weight. Consequently, the current trainer actually sends **240.0** to finishing, overriding the registration default of **40.0**.
- The existing `configs/finishing.yaml` also declares `reward` twice (`reward: dense`, followed by a `reward:` mapping). Standard YAML loading keeps the latter value, so the scalar reward ID is lost. The sweep avoids this file, but it does not by itself resolve the trainer's 240-versus-40 conflict.

This changes the Mode II reward objective and should not be guessed. **Do not start the finishing sweep until you confirm whether 40.0 or 240.0 is the intended finishing productivity weight.** If 40.0 is intended, the trainer needs a small mode-specific override after applying `REWARD_KWARGS`, and the duplicate `reward` mapping in `configs/finishing.yaml` should be removed. If 240.0 is intended, the finishing registration/config comments should be reconciled with that choice and the duplicate YAML key should still be removed.

The roughing sweep is not affected by this specific conflict.

## 6. Install safely

From the repository root, copy only the seven files listed in Section 1, preserving their relative paths. Do not copy any older core scripts from any previous ZIP version.

Before training, run a dry-run:

```bash
python scripts/sweep_ppo.py --mode roughing --dry-run
python scripts/sweep_ppo.py --mode finishing --ap-max 2.0 --dry-run
# A dry-run checks command construction only; it does not resolve the reward-weight issue in Section 5.
```

Then run a sweep:

```bash
# Mode I: chatter-aware roughing
python scripts/sweep_ppo.py --mode roughing

# Mode II: run only after resolving the finishing reward-weight issue in Section 5
python scripts/sweep_ppo.py --mode finishing --ap-max 2.0
```

Sequential configuration execution is the default and is recommended for reliable wall-clock comparisons. Optional concurrent execution is available with `--parallel-configs 2`, but CPU/GPU contention can distort runtime measurements.

After a successful sweep, the runner prints the manifest path. Generate the cross-profile plots with:

```bash
python scripts/plot_ppo_sweep.py --manifest "runs/ppo_sweep/roughing/<sweep-id>/sweep_manifest.json"
```

The plotter requires a fully completed sweep, checks the recorded training configuration and summary for every seed, and requires the same evaluation checkpoint schedule across all profiles. It does not interpolate missing evaluation checkpoints or silently discard non-finite observations.

## 7. Plot outputs

1. `ppo_sweep_learning_curve.png/.pdf`: mean evaluation episode return, calculated by averaging evaluation episodes within each training seed and then plotting mean ± 1 sample SD across independent training seeds.
2. `ppo_sweep_training_diagnostics.png/.pdf`: when common TensorBoard tags are available, mean ± 1 sample SD for rolling training episode return, explained variance, approximate KL, clip fraction, and policy standard deviation (or entropy loss as a consistent fallback).
3. `ppo_sweep_reliability_metrics.png/.pdf`: pass-completion rate, safety-termination rate, time-limit truncation rate, completed-pass productivity reward, final pass progress, and episode-wise peak sensor displacement. The productivity panel is the across-seed mean of each seed's median cumulative productivity reward for completed episodes; it is a reward diagnostic, not a direct material-removal-rate measurement. The existing callback records this metric as zero when no episode has completed at a checkpoint, so interpret early zero values alongside the pass-completion-rate panel (the median is otherwise undefined).

The shaded bands are descriptive `mean ± 1 sample SD` bands across training seeds. They are **not** 95% confidence intervals. With three seeds, the SD estimate is based on a small sample; use additional seeds if computationally feasible before making strong claims about variability.

The plotter also writes raw evaluation-return data and aggregated CSV statistics. The `train/approx_kl` panel shows the logged epoch-averaged value. Because Stable-Baselines3 applies its `1.5 × target_kl` early-stop test to individual minibatches, the plot intentionally does not draw that threshold as a line on the epoch-averaged curve. The plotter does not generate or compare physical/action trajectories; those remain in the existing per-run evaluation workflow and belong to the later closed-loop performance subsection.

## 8. Tests

From the repository root:

```bash
python -m unittest tests.test_ppo_sweep -v
```

The tests cover configuration inheritance and validation, rollout geometry, equal-budget arithmetic, exact evaluation-grid handling, seed-level aggregation, completed-run validation, and synthetic plot generation. They do not replace the actual environment preflight or a real PPO training run. Run the dry-run and preflight in the same Python environment used by the project before starting the full sweep.
