# Face-milling MPC–MHE architecture review (MPC-only, experimental)

**Base**: `mpc_followup_diagnostic_only.zip`, whose controller already contains
`--stability-first` and the earlier actuator-timing / objective / guarding patches.
The updated script in this archive replaces **only**
`scripts/MPC/mpc_face_milling_chatgpt_V3.py`. No changes to RL, environment,
plate, rewards, or the underlying do-mpc continuous phase-domain model.

## What the new run demonstrates

Your `--stability-first` seed-0 trial stopped with `environment_time_limit`, **not**
`excessive_sensor_displacement`: peak vibration 0.466 mm vs 3 mm limit, feed
progress 22.5%, 45,000 environment steps, MHE/MPC reported 100% accepted
solves, 4,566.5 s wall time. The controller was not given enough simulated
machining time to complete the nominal 90% pass.

The difference between 100% numerical solver success and validated observer
accuracy is important. The original controller unconditionally overwrote the
MHE's entire 26-state transport estimate with history assembled from a 2-sensor
pseudoinverse of a 3-mode system. Instantaneous sensor data cannot uniquely
recover three modal coordinates; the MHE uses a measurement *history* to help
resolve this. The overwrite is **not** justified by MHE convergence alone.

## New MHE history policies

`--mhe-history-policy pinv`: **legacy default**, reproduces the previous
post-MHE overwrite exactly. Preserve it for controlled comparison.

`--mhe-history-policy estimated`: retain the MHE's own dynamically constrained
pseudospectral delay state when the full estimation horizon is available and
IPOPT accepts the solution. On warmup or MHE failure revert to the existing
pseudoinverse-based history. **This is the recommended first A/B test**, not
an assertion that the MHE estimate is intrinsically accurate.

`--mhe-history-policy modal-history`: reconstruct a delay line from the past
MHE-estimated cutter displacements, which is another approximation. Compare it
only if the first two policies show an important discrepancy.

All policies use measured sensor observations and prior control commands only.
The full simulated modal state in `info['x_modal']` remains evaluation-only.

Each interval now logs:
- `mhe_history_disagreement_rms`: RMS(MHE delay - pinv delay)/w_limit before any overwrite.
- `mhe_meas_residual_rms`: RMS normalized difference between estimated and
  measured sensor displacement *and velocity*. It is an output-fit statistic,
  not sufficient to establish modal observability/accuracy.
- `mhe_disturbance_accel`: estimated `dF` input in acceleration units. Monitor
  for strong drift or saturation that could hide inaccurate force predictions.
- `mhe_history_policy_used`: actual per-decision source, including fallback.

Where simulation truth is available **for offline validation only**, also
compare `mhe_modal_err` and `cutter_reconstruction_error_over_wlimit` to the
new disagreement diagnostic; these should never be controller inputs.

## Balanced-pass MPC experiment

`--balanced-pass` is an opt-in, independent trial profile. It keeps joint
optimization and unchanged 400–4000 rpm / 0–12 mm machine bounds. The profile does not impose a positive minimum depth, speed floor,
or fixed-speed reference. If the machining process genuinely requires a
positive depth, set `--minimum-cut-depth` from that physical requirement rather
than to influence the optimizer's appearance. A nearly constant speed can still be a valid economic optimum.

| Quantity | V3 baseline | Overly conservative `--stability-first` | New `--balanced-pass` |
|---|---:|---:|---:|
| `q_energy` | 8 | 18 | 10 |
| `q_terminal` | 6 | 10 | 8 |
| `prod_weight` | 60 | 45 | 55 |
| `w_margin` | .50 | .40 | .55 |
| `hard_w_margin` | .80 | .72 | .80 |
| `r_omega` | 4 | 16 | 8 |
| `curvature_omega` | 8 | 20 | 10 |
| `omega_rate` (rad/s²) | 20000 | 3000 | 6000 |
| `prod_gate_disp` | 0 | 1 | 1 |
| `mhe_history_policy` | pinv | pinv | estimated |

At the same physical state the coefficient in the quadratic modal-energy
penalty scales approximately as `q_energy / w_margin^2`:
`(18/.4^2)/(8/.5^2) = 3.515625` for stability-first, but
`(10/.55^2)/(8/.5^2) ~ 1.033` for balanced-pass. This removes a strong
source of accidental over-penalization while keeping terminal headroom and
vibration-dependent productivity.

**These are starting experiments, not identified optimal weights.** The initial
horizon is only 8 * 0.25 tooth-pitch = 2 tooth passages: 7.5 ms at 4000 rpm,
compared with ~58.5 ms for the first 17.1 Hz plate mode. That is why short
predictions cannot by themselves certify future chatter avoidance. The
continuous-time output constraint is only checked on selected prediction nodes
(with phase-averaged engagement), another residual model risk.

## Controlled experiment commands (PowerShell, single line)

Use the same seed, milling line, and numerical configuration, and start with a
short interval. These runs do NOT demonstrate a completed pass.

```
python scripts/MPC/mpc_face_milling_chatgpt_V3.py --balanced-pass --mhe-history-policy pinv --y0 0.519 --no-randomize-y0 --seeds 0 --max-env-steps 45000 --progress-every 250 --out-dir ablation_balanced_pinv
python scripts/MPC/mpc_face_milling_chatgpt_V3.py --balanced-pass --mhe-history-policy estimated --y0 0.519 --no-randomize-y0 --seeds 0 --max-env-steps 45000 --progress-every 250 --out-dir ablation_balanced_mhe
```

Only if the history switch yields reasonable accuracy and the balanced profile
stays smooth, extend **the better candidate** through the previously unstable
region, e.g. `--max-env-steps 110000`, then use the AUTO full-pass budget. At
4000 rpm the 90% pass needs ~16.875 s of cutting and ~180,000 environment
steps for the 94-us environment step. A lower average RPM takes longer.

For **estimator diagnosis**, the strongest offline test is to use exactly the
same time-varying command history and state measurements to compare the
predicted one-step and multiple-step cutter displacement from each history
policy with the plant's true cutter displacement *offline*, not in the
controller. A successful MHE solve is not sufficient evidence.

## Required reporting and decision rules

1. Completion of 90% cut, termination reason, and physical peak sensor
   displacement less than the unchanged 3 mm safety limit.
2. Delivered depth, average spindle speed, removed-volume proxy, completion
   time, and cumulative reward **for complete passes**, not truncated returns.
3. Total variation and direction reversals of spindle speed and depth, realized
   actuator-rate constraints, fraction of post-MPC rate clips and guard actions.
4. Quantiles of MHE sensor residual, MHE-vs-pinv history disagreement,
   evaluation-only modal and cutter reconstruction error, `dF` drift,
   solver success and mean/p95 solve time.
5. Same environment seeds and initial conditions across ablations.

If the `estimated` history policy performs worse, **do not keep it blindly**.
That indicates model/measurement calibration or MHE transport-state tuning
requires investigation. The next engineering step would be an estimator
redesign that estimates modal state and regenerative history coherently without
optimizing an unnecessarily large unconstrained delay state. That is **not**
part of this minimal patch because it needs a validated replacement model.

## What has actually been verified here

- Python syntax and import-free unit tests of delay-policy selection and warmup
  fallback.
- Policy `pinv` reproduces the old override; policy `estimated` preserves
  the accepted MHE output; modal history is computed from MHE modal output.
- Balanced profile is opt-in, its weights are internally consistent in the
  original energy-normalization convention, and it does not restrict the
  original control bounds.

**Not verified here**: completed plant simulations, do-mpc solve behavior,
observer covariance calibration, real execution speed, improved vibration
suppression, or superiority to RL. The available container does not include
`do_mpc`/`gymnasium`. Do not claim a verified successful pass until the
Windows runs and offline diagnostics satisfy the criteria above.

## Optional offline finite-window observability diagnostic

```
python scripts/MPC/diagnose_mhe_observability.py --mhe-horizons 8 12 16 --rpm 400 1000 2000 4000
```

This uses the actual plant-provided sensor matrix and retained modal frequencies,
but a simplified **linear, unforced** plate model to compute the singular values
of the finite-window observation matrix. It cannot validate nonlinear MHE
accuracy or the 26 regenerative-delay states. A large condition number is a
warning that the modal state estimate will be sensitive to measurement noise,
model mismatch, and a short estimation horizon. Do not simply enlarge MHE until
the coupled estimator's runtime and accuracy have been compared.
