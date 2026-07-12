# Face-milling chatter control — full workflow (RL + MPC)

End-to-end instructions for **training**, **evaluation**, the **stability-lobe
diagram**, and the **RL-vs-MPC comparison**, for **first-mode (roughing)** and
**second-mode (finishing)** control. Everything runs on the same plant, reward,
integrator, and sample times, so RL and MPC are directly comparable.

```bash
pip install -e .          # gymnasium, stable-baselines3, matplotlib, numpy
pip install casadi        # required only for the MPC controller
```

--------------------------------------------------------------------------
## 0. Sample times and consistency (read once)

All tools share one timing chain — do not change it between train / eval / MPC:

```
RK4 substep      dt          = 1e-4 s
substeps/step    n_substeps  = 10
env control step dt*n_substeps = 1e-3 s   (1 ms; one env.step, one RL action)
tooth delay      tau = 2*pi/(N*omega)  in [0.375 ms @40000rpm, 15 ms @1000rpm]
constraint       dt < tau_min           (0.1 ms < 0.375 ms)  -> RK4 resolves regeneration
MPC step         Ts = control_hold * 1 ms   (default control_hold=3 -> Ts=3 ms)
MPC horizon      Np * Ts                     (default Np=20 -> 60 ms lookahead)
```

`--dt 1e-4 --n-substeps 10` must be used everywhere (training, eval, MPC). The
MPC prints this chain at startup so you can verify it. The two control modes use
**identical** timing, integrator, plant, and reward — they differ only in the
action space.

--------------------------------------------------------------------------
## 1. First-mode (roughing) control — action = [omega, ap]

### 1a. Train PPO

```bash
python scripts/train_sb3_ppo.py \
  --env-id CustomODEPlate-v0 --seeds 0 1 2 --reward dense \
  --total-timesteps 300000 --n-envs 1 --vec-env dummy \
  --max-episode-steps 2000 --n-eval-episodes 3
# -> models/ppo_plate/best_<seed>/best_model.zip , models/ppo_plate/final_<seed>.zip
```

### 1b. Evaluate

```bash
python scripts/eval_policy.py \
  --env-id CustomODEPlate-v0 --seeds 0 1 2 --reward dense \
  --n-episodes 5 --max-episode-steps 2000 \
  --model-dir models/ppo_plate --out-dir eval_trajectories
# fixed milling line:  add --y0 0.5      cycle lines: --y-position 0.2 0.5 0.8
```

--------------------------------------------------------------------------
## 2. Second-mode (finishing) control — action = [omega], ap fixed

**Important:** the registered `CustomODEPlateFinish-v0` samples the fixed depth
`ap` uniformly over `[0, 18] mm`, but only `ap <~ 1 mm` keeps the plate under the
1 mm displacement limit. Training over `[0,18]` therefore fails on almost every
episode. Train over a **feasible finishing-depth range** by capping `ap_max`.
The action space (omega only) is unchanged, so the model still evaluates on the
standard env with a pinned `--ap`.

### 2a. Train PPO (feasible depth range)

Use the helper `scripts/train_finish.py` (identical PPO hyper-parameters to
`train_sb3_ppo.py`, but builds the finishing env with `ap_max=1.0 mm`):

```bash
python scripts/train_finish.py     # -> models/ppo_plate_finish/{best_0,final_0}
```

(Equivalently, if you extend `train_sb3_ppo.py` to pass `ap_max`, train with
`--env-id CustomODEPlateFinish-v0 ... ap_max=1.0`.)

### 2b. Evaluate at a chosen finishing depth

```bash
python scripts/eval_policy.py \
  --env-id CustomODEPlateFinish-v0 --seeds 0 --reward dense \
  --n-episodes 5 --max-episode-steps 2000 --ap 0.5 \
  --model-dir models/ppo_plate_finish --out-dir eval_trajectories_finish
```

--------------------------------------------------------------------------
## 3. Stability-lobe diagram (no controller, RL/MPC-independent)

```bash
# 2D lobe (ap vs rpm) at one milling line
python scripts/stability_lobe_new.py \
  --rpm-min 1000 --rpm-max 40000 --rpm-points 40 \
  --ap-min 0 --ap-max 3 --ap-points 30 --dt 1e-4 --n-substeps 10

# 3D surface over several milling lines
python scripts/stability_lobe_new.py --surface-3d \
  --rpm-min 1000 --rpm-max 40000 --rpm-points 30 --y-points 5 --dt 1e-4 --n-substeps 10
```

Use a fine `ap` grid near `[0, 2] mm` — that is where the stable boundary lives
for this plate (larger `ap` breaches the 1 mm limit everywhere).

--------------------------------------------------------------------------
## 4. MPC controller (CasADi) — both modes

### 4a. First-mode MPC

```bash
python scripts/mpc_face_milling.py \
  --env-id CustomODEPlate-v0 --seeds 0 --n-episodes 1 \
  --y0 0.5 --max-steps 900 --out-dir mpc_trajectories
```

### 4b. Second-mode MPC (fixed depth)

```bash
python scripts/mpc_face_milling.py \
  --env-id CustomODEPlateFinish-v0 --seeds 0 --n-episodes 1 \
  --y0 0.5 --ap 0.5 --max-steps 900 --out-dir mpc_trajectories_finish
```

The MPC solves an OCP each control step (multiple shooting + IPOPT): objective =
the dense reward; equality constraints = discretised modal EOM, measurement map
`w=Phi*eta`, and initial condition; inequality bounds = action box + soft
safe-region `|w|<=w_limit`. It runs slower than RL (an NLP per step), so
`--max-steps` caps the closed-loop length.

### 4c. Diagnostic: relax the limit to see the full actuator/state variation

If a run terminates early on instability and you want to *see* the unconstrained
behaviour, relax the displacement limit (plant termination **and** MPC
constraint) — e.g. `|w| <= 0.1 m`:

```bash
python scripts/mpc_face_milling.py --env-id CustomODEPlate-v0 --seeds 0 \
  --y0 0.5 --max-steps 900 --w-limit 0.1 --out-dir mpc_trajectories_relaxed
```

--------------------------------------------------------------------------
## 5. Compare MPC vs RL

Run RL eval and MPC on the **same** `--env-id`, `--y0` (and `--ap` for second
mode), then overlay (RL = solid blue, MPC = dashed red):

```bash
# first mode
python scripts/compare_mpc_rl.py \
  --rl-dir eval_trajectories --mpc-dir mpc_trajectories --out-dir plots/mpc_vs_rl

# second mode
python scripts/compare_mpc_rl.py \
  --rl-dir eval_trajectories_finish --mpc-dir mpc_trajectories_finish \
  --out-dir plots/mpc_vs_rl_finish
```

Each episode overlay shows `max|w(t)|` (with the limit), spindle speed, `ap`,
axial force `Fz`, cumulative reward, and feed progress; plus a summary bar chart
and a printed table (return / max|w| / pass / feed / termination reason).

--------------------------------------------------------------------------
## 6. MPC tunable parameters

| Flag | Default | Meaning |
|------|---------|---------|
| `--horizon` | 20 | prediction/control steps `Np` (longer = less myopic, slower) |
| `--control-hold` | 3 | env steps per MPC decision -> `Ts = control_hold * 1 ms` |
| `--n-rk` | 6 | RK4 substeps per shooting interval (integration accuracy) |
| `--w-lim-margin` | 0.75 | safe threshold = margin*w_limit (headroom for model under-prediction) |
| `--slack-weight` | 5e4 | penalty enforcing the soft safe-region constraint |
| `--terminal-weight` | 3.0 | terminal vibration penalty (prevents deferring a breach past the horizon) |
| `--omega-rate-weight` / `--ap-rate-weight` | 0.05 | smoothness of the control signal |
| `--prescreen-n-omega` / `--prescreen-n-ap` | 28 / 7 | global speed/depth pre-screen grid (warm start; handles the non-convex lobes) |
| `--delay-sigma` | 0.7 | softmax fractional-delay kernel width [Ts samples] |
| `--hessian` | limited-memory | IPOPT Hessian (`exact` = accurate but slow) |
| `--max-iter` | 40 | IPOPT iteration cap (primal is feasible well before this) |
| `--w-limit` | (env) | override plant/MPC displacement limit (diagnostic, see 4c) |
| `--feedback` | state | `state` = true modal state (ideal baseline); `observer` = min-norm estimate from sensors |

**If a run terminates on instability:** lower `--w-lim-margin` (e.g. 0.65),
and/or raise `--terminal-weight` (e.g. 6) and `--horizon` (e.g. 30). To simply
see the full trajectory without termination, use `--w-limit 0.1`.

--------------------------------------------------------------------------
## 7. Reward (identical for both modes)

`reward = productivity_gate * MRR - w_cost - wdot_cost - omega_cost (+ terminal)`

- `w_cost = 0.6 * mean((w/7.5e-4)^2)`, `wdot_cost = 0.02 * mean((wdot/1.0)^2)`
- productivity `= 30 * omega_score * ap_score * gate`,
  `gate = 1/(1 + (w_rms/5e-4)^2 + 0.05*(wdot_rms/1)^2)`
- terminal: `+10000` on pass completion, `-500` (progress-weighted) on failure.

Note on scale between modes: in first mode `ap_score = ap/18` is a decision, so
productivity is significant. In second mode `ap` is fixed and small
(`ap_score ~ 0.03` at 0.5 mm on the standard env), so the finishing objective is
dominated by chatter suppression (vibration cost) — which is the correct goal
for a finishing pass. RL and MPC use the **same** reward, so the comparison is
fair within each mode. The MPC reconstructs this reward symbolically with the
environment's exact weights.

--------------------------------------------------------------------------
## 8. Model, multivariable coupling, and verification

**Model / state / measurement.** The OCP state is the modal coordinate
`z=[eta; eta_dot]` (2K). The measured plate response is `w = Phi @ eta`
(equality E2). The **tool x-y position is NOT a controller state** — unlike the
RL observation, the MPC uses the full modal state. The cutter location is only a
**model parameter**: it is required to evaluate the mode shape `phi_z(xc,yc)` at
the tool for the force projection, so it is passed in and updated every control
step (frozen over the short horizon). This is the only place tool position enters.

**omega-ap coupling (predictive delay).** The regenerative delay
`tau = 2*pi/(N*omega)` is what makes stability depend on spindle speed. With
`--delay-mode predictive` (default) the OCP uses the per-node decision speed, so
the optimiser sees this coupling and trades speed against depth. Verify it:

```bash
python scripts/mpc_verify.py --out-dir plots/mpc_verification
```

This produces `omega_ap_interaction.png` (MPC-selected speed vs fixed depth: the
speed stays at 40 krpm while the depth is shallow and safe, then drops to
~8-17 krpm as the depth approaches the stability boundary — stability-lobe
following) and a feasibility table (100% solver success, finite/feasible, no NaN,
no terminations).

**Why the inputs look constant at shallow depth.** At small `ap` the whole speed
range is stable, so constant max speed is genuinely optimal and the MPC (correctly)
holds it — it beats the RL return there because RL's oscillation is suboptimal
policy noise, not required control. Active modulation appears when the process is
pushed to the stability boundary (higher `ap`), over the full pass (the cutter
moves into stiffer regions so `ap` ramps up), or under process disturbance
(`--dynamics-uncertainty-std 0.003`). Use those regimes to showcase responsive
multivariable control; use `--w-limit 0.1` to see the unconstrained response.

**Feasibility.** The safe-region constraint is soft (slack-penalised), so the NLP
is always feasible; the solver returns the best primal-feasible iterate and a
pre-screen fallback covers any non-convergence, so the applied control is never
NaN. Slew limits (`--ap-up-slew`, `--omega-slew`) keep the applied action from
jumping to a chattering point in one step.
