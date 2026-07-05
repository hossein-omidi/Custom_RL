# Custom RL

Gymnasium-compatible ODE/RK4 control framework for the milling/plate vibration process with PPO-based reinforcement learning.

The current environment uses a nonlinear modal plate vibration plant for milling-like excitation:

```text
Environment ID: CustomODEPlate-v0
Plant: PlatePlant (face-milling on a cantilever flexible plate)
Internal state (modal, not observed directly): [eta1, eta1_dot, ..., etaK, etaK_dot]
Observation (physical, via modal->physical sensor projection):
  [w_sensor/w_obs_scale, wdot_sensor/wdot_obs_scale, cutter_x/L1, cutter_y/L2]
Action: normalized PPO action [u_omega, u_ap] in [-1, 1]^2
  (optionally [u_omega, u_ap, u_ae] if control_ae=True)
Physical action: [omega_rad_s, ap_mm] (axial depth of cut ap; radial
  immersion ae is a fixed process parameter unless control_ae=True)
Integrator: fixed-step RK4 with a regenerative-delay history buffer
RL algorithm: PPO actor-critic from Stable-Baselines3
```

## Quickstart

```bash
pip install -e .
```

## Register and create environment

```python
import gymnasium as gym
from custom_rl import register_envs

register_envs()

env = gym.make(
    "CustomODEPlate-v0",
    reward_id="dense",
    max_episode_steps=1000,
)

obs, info = env.reset(seed=42)
```

## Verify environment with random policy

```bash
python scripts/check_plate_random_policy_new.py
```

This checks environment registration, reset, step, observation/action dimensions, reward calculation, and RK4 integration (including the regenerative-delay history buffer).

## Fast PPO training check

Use this command to quickly verify that PPO training runs correctly:

```bash
python scripts/train_sb3_ppo.py \
  --seeds 0 \
  --reward dense \
  --total-timesteps 2048 \
  --n-envs 1 \
  --max-episode-steps 1000 \
  --n-eval-episodes 1
```

## Full PPO training

Recommended single-seed training:

```bash
python scripts/train_sb3_ppo.py \
  --seeds 0 \
  --reward dense \
  --total-timesteps 300000 \
  --n-envs 1 \
  --max-episode-steps 1000 \
  --n-eval-episodes 3
```

Recommended multi-seed training:

```bash
python scripts/train_sb3_ppo.py \
  --seeds 0 1 2 \
  --reward dense \
  --total-timesteps 300000 \
  --n-envs 1 \
  --max-episode-steps 1000 \
  --n-eval-episodes 3
```

For the productive-cutting reward:

```bash
python scripts/train_sb3_ppo.py \
  --seeds 0 \
  --reward productive \
  --total-timesteps 300000 \
  --n-envs 1 \
  --max-episode-steps 1000 \
  --n-eval-episodes 3
```

## Notes on parallel environments

Use `--n-envs 1` for the current milling/plate model because the nonlinear dynamics module uses module-level cached variables. Parallel training should only be used after confirming that the plant/dynamics cache is safe for multiple simultaneous environments.

Options:

```text
--n-envs N              number of vectorized environments
--vec-env dummy|subproc vectorized environment type
```

## Evaluate trained policy

Short evaluation:

```bash
python scripts/eval_policy.py \
  --seeds 0 \
  --reward dense \
  --n-episodes 5 \
  --max-episode-steps 1000
```

Longer evaluation for clearer trajectory analysis:

```bash
python scripts/eval_policy.py \
  --seeds 0 \
  --reward dense \
  --n-episodes 10 \
  --max-episode-steps 3000
```

Multi-seed evaluation:

```bash
python scripts/eval_policy.py \
  --seeds 0 1 2 \
  --reward dense \
  --n-episodes 10 \
  --max-episode-steps 3000
```

The evaluation script saves observations, normalized actions, physical actions `[omega_rad_s, ap_mm]`, rewards, reward-term decomposition, physical sensor signals, cutting-force diagnostics, and time values.

## Plot training and evaluation results

```bash
python scripts/plot_results.py \
  --seeds 0
```

For multi-seed plotting:

```bash
python scripts/plot_results.py \
  --seeds 0 1 2
```

To specify directories manually:

```bash
python scripts/plot_results.py \
  --log-dir logs/ppo_plate \
  --traj-dir eval_trajectories \
  --out-dir plots \
  --seeds 0 1 2
```

Generated plots include the learning curve, modal state trajectories, and physical control actions.

## Procedure: Train, Evaluate, Plot

### 1. Verify the environment

```bash
python scripts/check_plate_random_policy.py
```

### 2. Run fast PPO check

```bash
python scripts/train_sb3_ppo.py \
  --seeds 0 \
  --reward dense \
  --total-timesteps 2048 \
  --n-envs 1 \
  --max-episode-steps 1000 \
  --n-eval-episodes 1
```

### 3. Run full training

```bash
python scripts/train_sb3_ppo.py \
  --seeds 0 \
  --reward dense \
  --total-timesteps 300000 \
  --n-envs 1 \
  --max-episode-steps 1000 \
  --n-eval-episodes 3
```

### 4. Evaluate the trained policy

```bash
python scripts/eval_policy.py \
  --seeds 0 \
  --reward dense \
  --n-episodes 10 \
  --max-episode-steps 3000
```

### 5. Plot results

```bash
python scripts/plot_results.py \
  --seeds 0
```

## Generate publication-style figures

After training and evaluating a policy (trajectory JSON must exist first):

```bash
python scripts/plot_paper_figures.py \
  --log-dir logs/ppo_plate \
  --traj-dir eval_trajectories \
  --out-dir plots/paper
```

Produces the learning curve, closed-loop control/vibration time series, actuator heatmaps across runs, the (rpm, ap) operating-density map, the vibration field over pass progress, an evaluation summary (return distribution and pass-completion rate), a dense reward-term decomposition, cutting-force trajectories, and a vibration-robustness-vs-pass-line (y=a) figure.

## Stability lobe diagram (no-control, RL-independent)

```bash
python scripts/stability_lobe_new.py --rpm-min 1000 --rpm-max 40000 --ap-min 0 --ap-max 18
```

Sweeps spindle speed and axial depth of cut with no controller to trace the no-control stability boundary; supports 2D/3D and stochastic (Monte Carlo) variants and different milling pass lines. Independent of the RL training pipeline.

## Validate the stochastic/uncertainty setup

```bash
python scripts/validate_stochastic_setup.py
```

Sanity-checks deterministic vs. stochastic rollouts, reward-term decomposition, and milling pass-line (y0) randomization.

## Main files

```text
custom_rl/envs/
  ode_control_env.py               Generic Gymnasium ODE environment
  registration.py                  Registers CustomODEPlate-v0 (single source of
                                    truth for plant/reward defaults)

custom_rl/plants/
  base.py                          ODEPlant interface
  plate.py                         Face-milling flexible-plate plant (modal ODE,
                                    modal->physical sensor projection, termination)
  f_nonlinear2_face_milling.py     Nonlinear modal face-milling force dynamics
                                    with regenerative-delay history
  compute_mode_shapes_updated.py
  compute_natural_frequencies_updated.py
  compute_nonlinear_stiffness_updated.py

custom_rl/rewards/
  base.py                          Reward protocol
  plate_rewards.py                 Dense/productive/quadratic/sparse rewards

custom_rl/integration/
  rk4.py                           History-aware fixed-step RK4 integrator

custom_rl/eval/
  pipeline.py                      Shared env-kwargs/metadata helpers for
                                    train/eval/plot scripts
  monte_carlo.py                   Monte Carlo rollout aggregation/plotting

scripts/
  check_plate_random_policy_new.py Environment smoke test with random/fixed policy
  train_sb3_ppo.py                 Multi-seed PPO training
  eval_policy.py                   Policy evaluation and trajectory export
  plot_results.py                  Basic training/trajectory plots
  plot_paper_figures.py            Publication-quality figures (see above)
  stability_lobe_new.py            No-control stability lobe diagram (2D/3D/MC)
  validate_stochastic_setup.py     Stochastic-plant sanity checks
```

## Default output directories

```text
logs/ppo_plate/                 training monitor logs
models/ppo_plate/               saved PPO models
eval_trajectories/              evaluation trajectory JSON files
plots/                          generated plots
```
