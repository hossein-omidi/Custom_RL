# PPO Fast / Balanced / Precise Sweep

This patch adds three matched PPO configurations and a small sweep runner. It reuses the existing `scripts/experiment.py` and `scripts/train_ppo_unified.py`; it does not introduce a new RL algorithm or change the milling environment. The configs inherit the repository's timing and noise settings; plant and reward defaults continue to come from the existing trainer/environment registration.

## Configuration design

| Setting | Fast | Balanced | Precise |
|---|---:|---:|---:|
| Initial learning rate | 3e-4 | 2.5e-4 | 1.5e-4 |
| Final learning rate | 3e-5 | 2.5e-5 | 1.5e-5 |
| `n_steps` | 1,024 | 2,048 | 4,096 |
| `batch_size` | 512 | 256 | 256 |
| `n_epochs` | 4 | 8 | 10 |
| `gamma` | 0.9999 | 0.9999 | 0.9999 |
| `gae_lambda` | 0.98 | 0.98 | 0.98 |
| Initial clip range | 0.20 | 0.20 | 0.15 |
| Final clip range | 0.10 | 0.10 | 0.075 |

All other PPO settings remain fixed at the trainer's existing values: entropy coefficient, value-function coefficient, gradient clipping, target KL, network architecture, initial policy standard deviation, and Adam epsilon. These are deliberately not added as new tunable parameters in this simple sweep.

The Balanced configuration preserves the November 11 trainer defaults. Fast is a lower-compute candidate; Precise is a more conservative, higher-update-compute candidate. These are experimental candidates, not asserted optima.

## Fairness controls

- Same mode, environment, reward selector, physical timing, uncertainty, and reset randomization across the three configurations.
- Same training seeds: `0, 1, 2`.
- Same number of parallel environments: `3`.
- Same callback evaluation seed: `10000`.
- Same number of evaluation episodes per callback: `10`.
- Same requested training budget per seed. The runner rounds the requested budget up to the least common multiple of the selected rollout sizes so every configuration receives exactly the same number of transitions.
- Gamma and GAE lambda are explicitly fixed in YAML, rather than relying on version-specific trainer defaults.

With the default three configurations, the common budget is **1,007,616 transitions per seed** (3,022,848 transitions per configuration across three seeds). Rollout sizes are 3,072, 6,144, and 12,288, respectively. Batch sizes divide their rollout sizes exactly.

## Install

Unzip this package into the repository root, preserving its folders. The package includes the newer November 11 copies of `experiment.py`, `train_ppo_unified.py`, `eval_policy.py`, and `plot_results.py` that were supplied in this conversation. These replace the older copies on the `mpc-correction` GitHub branch. Keep the repository's existing `custom_rl/`, `scripts/run_paths.py`, `scripts/ppo_reliability.py`, `configs/base.yaml`, and other support files.

The three sweep YAML files extend the repository's existing `configs/base.yaml`. They set `plant.ap_max: null` intentionally: the existing config flattener otherwise promotes the base plant's 12 mm value to `--ap-max`, which the roughing trainer correctly rejects. For finishing, the runner passes `--ap-max 2.0` explicitly by default.

The trainer's reward weights are currently defined in `REWARD_KWARGS` inside `train_ppo_unified.py`; the existing `reward:` block in `base.yaml` is not forwarded by the current experiment config flattener. This sweep does not silently alter those reward weights. All three PPO configurations use the same trainer-defined reward settings.

## Run

From the repository root:

```bash
# Roughing (Mode I): preflight all configs, then train sequentially
python scripts/sweep_ppo.py --mode roughing

# Finishing (Mode II), default ap_max = 2.0 mm
python scripts/sweep_ppo.py --mode finishing --ap-max 2.0

# Preview commands without running anything
python scripts/sweep_ppo.py --mode roughing --dry-run

# Optional concurrent config jobs (not recommended for wall-clock comparisons)
python scripts/sweep_ppo.py --mode roughing --parallel-configs 2
```

The runner performs preflight checks for every selected configuration before starting any training. It trains only; it does not automatically launch the more expensive trajectory Monte Carlo evaluation.

When training finishes, it prints the path to `sweep_manifest.json`. Use that exact path:

```bash
python scripts/plot_ppo_sweep.py --manifest "runs/ppo_sweep/roughing/<sweep-id>/sweep_manifest.json"
```

The plotter creates a combined evaluation learning curve (mean ± 1 sample SD across training seeds), raw episode-return CSV, aligned learning-curve statistics CSV, and final summary CSV. It never pools Monte Carlo replicas as if they were independent training seeds.

For full physical trajectory evaluation after comparing learning curves, evaluate each run separately:

```bash
python scripts/experiment.py eval --run-dir "<run-directory-from-sweep-manifest>"
python scripts/experiment.py plot --run-dir "<run-directory-from-sweep-manifest>"
```

## Dependencies

No new runtime dependencies are introduced. The project environment should already contain PyYAML, NumPy, Matplotlib, Stable-Baselines3, Gymnasium, PyTorch, and TensorBoard as required by the existing pipeline.

## Tests

From the repository root:

```bash
python -m unittest tests.test_ppo_sweep -v
```
