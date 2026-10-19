# MPC-only stability intervention — experimental research patch

Scope: **only** `scripts/MPC/mpc_face_milling_chatgpt_V3.py`. This patch builds on the MPC-only corrected V3 supplied in the prior ChatGPT turn (`mpc_face_milling_chatgpt_V3_mpc_fixed.py`). The plant, RL training/evaluation code, and solver/state-estimator dynamics are **untouched**. It does not change the safety limit, pass termination, delay discretization, or physical plant parameters.

## What the supplied plot supports

- RL completes the pass for the plotted case, while MPC terminates on `excessive_sensor_displacement` at `w_limit=0.003 m`.
- MPC depth approaches 6–7 mm around 10 s and the speed starts alternating strongly around 4 s. The plot **does not** disclose NLP status, prediction errors, MHE error, actual safety-guard state, or exact run settings; therefore it cannot uniquely diagnose solver nonconvergence.
- The uploaded `plate.py` constructor defaults to `w_limit=0.004`, whereas the GitHub registration factory defaults to `w_limit=0.003`. Compare *effective runtime settings*, not just constructor defaults.
- Comparing accumulated environment reward is confounded by termination time and by the potentially different environment-step periods used by phase-synchronous MPC and time-sampled RL. Prefer pass completed, violation rate, max sensor displacement, depth and speed variation, material removed, and compute cost.

## Theory / diagnosis

At four teeth, `dtheta_frac=0.25`, `mpc_horizon=8`, the current horizon is 2 tooth passages (`pi rad`). Its *physical-time* span is only `pi/omega` = **7.5 ms at 4000 rpm**, compared with about **59 ms** for the dominant ~17 Hz plate mode. This cannot guarantee that the optimizer observes the vibration-growth timescale at high rpm. Increasing the horizon is a possible later step but strongly increases the IPOPT burden.

The controller's existing MRR objective is ungated by vibration, unlike the repository's RL reward, which uses a smooth state-dependent productivity gate. A separate vibration cost does not enforce the same tradeoff in every state. Also, the former `omega_rate=2e4 rad/s2` is permissive enough for frequent command alternation. The current forward-feedback safety guard has no hysteresis. Delay history is reconstructed from two instantaneous displacement sensors for three modes; this mapping is underdetermined and may cause prediction errors.

## Additions (opt-in)

`--stability-first` applies the following research **starting values** to arguments that were not individually supplied on the CLI:

| Argument | Original corrected V3 | Experimental profile |
| --- | ---: | ---: |
| `--q-energy` | 8 | 18 |
| `--q-terminal` | 6 | 10 |
| `--prod-weight` | 60 | 45 |
| `--q-omega` | 10 | 14 |
| `--r-omega` | 4 | 16 |
| `--r-ap` | 1 | 5 |
| `--curvature-omega` | 8 | 20 |
| `--curvature-ap` | 2 | 6 |
| `--w-margin` | 0.50 | 0.40 |
| `--hard-w-margin` | 0.80 | 0.72 |
| `--safety-guard-margin` | 0.75 | 0.62 |
| `--guard-release-margin` | 0.75 (no hysteresis) | 0.52 |
| `--safety-guard-gain` | 1.0 | 1.5 |
| `--omega-rate` | 20000 | 3000 rad/s² |
| `--minimum-cut-depth` | 0 | 0.5 mm (roughing) |
| `--preview-gain` | 0 (disabled) | 1.0 |
| `--prod-gate-disp` | 0 (disabled) | 1.0 |
| `--prod-gate-vel` | 0 (disabled) | 0.1 |

The original `--ap-rate 20 mm/s` is deliberately unchanged so the controller can reduce depth promptly in an emergency. The chosen `0.5 mm` depth floor prevents gaming the completion metric by cutting at zero depth; replace it with a documented minimum effective depth if necessary. The profile does not alter `--dtheta-frac`, `--mpc-horizon`, `--mhe-horizon`, `--n-delay`, `--coll-deg`, or `--coll-ni`. These must be validated separately.

### Economic objective

Uses the previous dimensionless MRR proxy but optionally introduces

`P_gate = P_raw / (1 + q_d * D + q_v * V)`

with `D = disp_metric`, `V = vel_metric` already in the MPC formulation, and nonnegative `q_d`, `q_v`.

This term smoothly *reduces productivity incentive when predicted vibration grows*. It is NOT identical to the RL reward; the RL reward uses its own sensor-level scaling and a separate gate. This change should be identified in a research paper as an economic MPC design choice rather than an exact reward reproduction.

### Anticipatory vibration constraints

For each sensor and cutter-point displacement, adds the **soft** predictions

`w_preview = w + preview_gain * dtheta/omega * wdot`

with target `|w_preview| <= w_margin * w_limit` under the existing slack penalty. Original soft displacement and original hard displacement constraints remain. This is a local constant-velocity approximation (first-order Taylor expansion), **not** a rigorous future-state bound or a guarantee against collisions. It adds only simple algebra and soft constraints, but the IPOPT nonlinear program is modestly larger.

### Hysteretic safeguard

Activate emergency depth retreat when the measured conservative proxy `(|w| + gain*dt*|wdot|)/w_limit >= 0.62`. Continue retreat until the proxy drops below 0.52. The update obeys the same physical depth slew limit and respects the optional productive depth floor. The physical limit `w_limit` stays unchanged and active in `plate.py`.

### Diagnostics

Adds `measured_w_ratio`, `measured_velocity_ratio`, and `cutter_reconstruction_error_over_wlimit` to `mpc_diagnostics`. The last quantity uses the simulator's true modal state **for reporting only**; it is not used by the controller or optimization.

## Install

Back up the previous file first. From the repository root:

```bash
# copy the scripts/MPC/mpc_face_milling_chatgpt_V3.py file from this ZIP
# into scripts/MPC/mpc_face_milling_chatgpt_V3.py (overwrite only that file).
python -m pytest tests/test_mpc_stability_math.py -q
```

`mpc_stability_only.patch` is a unified diff relative to the **previous corrected V3**, not the unmodified GitHub branch version. If applying to the original GitHub V3, first apply the prior `mpc_only_changes.patch`.

## Short comparison command

For reproducible physical integration, set the RK substep to the same `1e-4 s` as RL, while MPC retains `n_substeps=1` to implement its phase-synchronous control timing.

```bash
python scripts/MPC/mpc_face_milling_chatgpt_V3.py \
  --env-id CustomODEPlate-v0 --seeds 0 --n-episodes 1 \
  --y0 0.519 --no-randomize-y0 \
  --dt 1e-4 --n-substeps 1 \
  --stability-first --compile-nlp \
  --max-env-steps 120000 --progress-every 250 \
  --out-dir mpc_stability_trial
```

If no C compiler is available, omit `--compile-nlp`. Run an identical baseline with `--stability-first` omitted, using the same plant settings, seed, sampling, and resource limits. This initial comparison covers approximately 12 s, enough to examine the ~4–10 s instability shown in the plot, but not the full 90% pass. Use the auto budget (omit `--max-env-steps`) for subsequent full-pass tests.

Test changes in stages: (A) baseline; (B) `--stability-first` with original horizon 8; (C) if the first 12 s are stable but the pass still fails, test `--mpc-horizon 12` with compile enabled; (D) change `--dtheta-frac` only after validating the phase-cell force and modal integration error at the lowest commanded rpm. More coarsening speeds up decisions but can alias high-frequency plate dynamics and does not preserve accuracy automatically.

## Acceptance conditions

- First: no termination on excessive displacement; complete 90% pass under unchanged physical limit and nonzero productive depth.
- Second: lower total variation and fewer direction reversals for both `omega` and `ap`.
- Third: no loss of physical prediction quality when changing phase resolution, and unchanged units.
- Fourth: report MPC/MHE solver-success fractions, mean and p95 wall times, realized actuator rates, safety-guard activity, and sensor/MHE/cutter-reconstruction errors.
- Fifth: compare normalized machining metrics in physical time, NOT raw accumulated reward when environment step periods differ.

## Limitations and further work

This patch has passed syntax and seven dependency-light mathematical/static tests but was **not** validated in a Gym/do-mpc closed-loop run here (those packages are unavailable in the execution container). It is not correct to claim convergence, zero failure, or improved runtime until the actual full model is run. Frequent switching can also reflect model/estimator mismatch or nonconvex stability lobes rather than only poor gains. If the profile fails, next investigate the delay-line pseudoinverse bias and MHE performance before increasing solver complexity.
