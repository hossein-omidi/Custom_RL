# Final Mathematical Audit — Plate Milling RL Simulator

**Date:** 2026-06-16  
**Model:** `displacement_model = "surface_normal_reduced"` (default)  
**Verdict:** **PASS** — internally consistent, Nasiri/Moradi-inspired reduced surface/plane regenerative model with honest generalizations. **Retrain from scratch** required (6-dim sensor obs + updated dynamics invalidate old checkpoints).

> A generalized reduced surface/plane milling regenerative model inspired by Nasiri/Moradi. It keeps nonlinear modal plate dynamics, tooth-period regenerative delay, directional engagement, chip-thickness feedback, surface-normal force projection into the modeled compliant direction, and a sensor-level RL interface. It is **not** the full two-component Nasiri/Moradi model unless independent feed and normal displacement fields are implemented.

See also [`DIRECTIONAL_MILLING.md`](DIRECTIONAL_MILLING.md).

---

## Section-by-section results

| # | Topic | Result | Evidence |
|---|--------|--------|----------|
| 1 | Structural dynamics | **PASS** | `f_nonlinear2.f_nonlinear2` L431–447; tests `test_ac_zero_*`, `test_structural_current_only` |
| 2 | Regenerative delay | **PASS** | Delay only in `Delta_q` (`milling_force.py` L251–268); `delay_time` L147–163; no `STATE_DELAY` |
| 3 | Directional reduced chip | **PASS** | `chip_thickness_surface_normal_reduced`; `Delta_f=0`; `h<=0` → zero force |
| 4 | Engagement / surface milling | **PASS** | Defaults π arc for surface/face/slotting; wrap-around in `engagement()`; metadata in `to_dict()` |
| 5 | Force law & projection | **PASS** | `F_projected = F_surface_normal`; no magnitude for excitation; diagnostics script |
| 6 | Units | **PASS** | `f_t = 2π·vf/(N·ω)`; `cf_units`; `ac`/`dz` in m; xi/delta not double-scaled |
| 7 | Depth of cut | **PASS** | Axial integration default; `ac<=0` → zero force |
| 8 | Mode ordering | **PASS** | `iter_mode_indices` / `mode_index_map` consistent across η, W_k, S, b_vec |
| 9 | Sensor / RL interface | **PASS** | obs `(6,)`; no η in obs/reward; clip on obs only |
| 10 | Action scaling | **PASS** | Single map in `PlatePlant._scale_action`; `info["action_phys"]` |
| 11 | Integrator / DDE | **PASS** | `dde_rk4` default; substage scratch history in `dde_rk4.py` L28–42 |
| 12 | Caching | **PASS** | Cache = geometry `b_vec` only; `F_projected(η)` every RHS call |
| 13 | Reward / termination | **PASS** | `SensorProductivePlateReward`; sensor displacement limits |
| 14 | Stability lobes | **PASS** | `scripts/stability_lobes.py`; empirical MC lobes, not Floquet |
| 15 | Fast config | **PASS** | `conf_fast` exists; training + lobes + pytest verified |
| 16 | Documentation | **PASS** | This report + `DIRECTIONAL_MILLING.md` |

---

## 1. Structural dynamics — PASS

**Verified:** Current-state-only modal ODE

```
η̈_k = -2ζ_k ω_k η̇_k - ω_k² η_k - λ_k η_k³ + F_k(t)
```

- No delayed η or η̇ in damping, stiffness, or cubic terms (`f_nonlinear2.py` L447).
- `ac=0` → zero cutting force → zero `Fk` → zero state remains zero (`test_ac_zero_zero_state`).
- `ac=0`, small IC decays with ζ>0 (`test_ac_zero_decay`, `test_free_vibration_decays_with_zero_ac`).
- Fake delayed history ignored at `ac=0` (`test_structural_current_only`).

**Integrators:** `dde_rk4` (recommended) and `rk4` (approximation) both call the same RHS; DDE adds substage history for delay interpolation.

---

## 2. Regenerative delay — PASS

**Verified:** Delay appears **only** in chip thickness via

```
Delta_q = q_c(t, x_c, y_c) - q_c(t_delay, x_c_delay, y_c_delay)
```

| Mode | Implementation |
|------|----------------|
| `constant_tau` | `τ = 2π/(N·ω_current)` (`tooth_period`) |
| `spindle_phase` | Bisection on `θ(t)-θ(t_delay)=2π/N` from ω history |

- No fixed `STATE_DELAY` in structural terms (grep: absent).
- Delayed history used only to reconstruct `q_c(t_delay)` (`_get_modal_state_at_delay` → `compute_directional_forces`).
- `Delta_q=0` when current and delayed `q_c` match (`test_delta_q_zero_when_q_c_equal`, `test_regenerative_vanishes_zero_delta_w`).
- **Startup:** for `t < τ`, no delayed state → `Delta_q=0` (documented approximation).
- **Variable ω:** use `delay_mode="spindle_phase"`; `constant_tau` assumes piecewise-constant ω per tooth period.

---

## 3. Directional reduced chip thickness — PASS

```
h_j = [ f_t·sin(φ_j) + Delta_q·cos(φ_j) ] · g(φ_j)
```

- `Delta_f = 0` in `surface_normal_reduced` (`milling_force.py` L269).
- `h_j ≤ 0` → `tangential_radial_increment` returns `(0,0)`.
- Tooth angle: `φ_j = θ + j·2π/N - 2z·tan(β)/D_c + φ_0`; helix term zero when `β=0`.
- `feed_normal_full` raises `NotImplementedError` on projection (`test_feed_normal_full_not_implemented`).

---

## 4. Engagement — PASS

- `g(φ)=1` on `(φ_st, φ_ex)` with wrap-around (`engagement()` L25–34; `test_wrap_around_engagement`).
- Surface/face/slotting default arc width π (`MillingForceConfig` defaults; `test_surface_default_engagement_pi`).
- Peripheral milling supported via `milling_type="peripheral"` (π/2 default arc).
- Metadata: `milling_type`, `phi_st`, `phi_ex`, `angle_convention`, `engaged_arc` via `to_dict()`.

---

## 5. Force law and projection — PASS

```
dF_t = poly_t(h)·dz,  dF_r = poly_r(h)·dz   (h>0)
F_sn = Σ (dF_t·sin φ + dF_r·cos φ)
F_projected = F_sn
F_k = W_k(x_c,y_c)/M_modal · F_projected
```

- In-plane `F_feed`, `F_normal_in_plane` are diagnostics only; they do **not** excite modal DOFs (`test_in_plane_forces_not_projected_into_modal`).
- `sqrt(F_feed² + F_normal²)` never used for excitation (`test_magnitude_never_used_for_modal_excitation`).
- Sign: `F_sn = dF_t·sin(φ) + dF_r·cos(φ)` (`surface_normal_force_component`; `test_surface_normal_component_from_ft_fr`).
- One-revolution plots: `python scripts/milling_force_diagnostics.py --omega 800 --ac 3`.

---

## 6. Units — PASS

| Quantity | Convention |
|----------|------------|
| ω | rad/s |
| f_t (from feed speed) | `2π·feed_speed/(N·ω)` [m] |
| cf | `m_per_tooth` or `mm_per_tooth` (×10⁻³) |
| h_j | metres (geometric chip) |
| xi, delta | N/m³ … N/m polynomial in h [m] → N via ×dz [m] |
| ac (RL action) | **mm** (`ac_units`, default) |
| ac, dz (internal) | metres (`ac_to_meters` before axial integration) |

- Axial integration: `dF ∝ dz`; xi/delta **not** multiplied by ac when `ac_via_axial_integration=True` (`test_ac_doubles_force_axial_integration`).

---

## 7. Depth of cut — PASS

- Default: `z ∈ [0, ac]`, quadrature sum (`milling_force.py` L243–245).
- `ac ≤ 0` → `_zero_force_result` (exactly zero force).
- If `ac_via_axial_integration=False`: coefficients scaled by ac (reduced lumped approximation; documented in config).

---

## 8. Mode ordering — PASS

Single ordering via `iter_mode_indices(m_max, n_max)` for:
- state vector `[η_k, η̇_k]`,
- `W_k` reconstruction,
- `omega_vec`, `lambda_vec`,
- `S_disp`, `b_vec`,
- force projection.

`test_mode_ordering_in_b_vec`, `test_S_disp_shape_and_matrix_definition`, `mode_index_map` on plant.

---

## 9. Sensor / RL interface — PASS

```
w_s = S_disp @ η,   ẇ_s = S_disp @ η̇
obs = [w_s/scale, ẇ_s/scale, prev_u_norm]  → shape (6,) for 2 sensors, 2 controls
```

- Agent never sees η (`test_plate_observation_is_sensor_based_not_modal`).
- Reward uses `sensor_obs_norm`, not η (`SensorProductivePlateReward`; `test_reward_depends_on_sensor_obs_norm_not_modal_state`).
- Clipping on PPO obs only; reward/termination use unclipped physical sensors.
- Old checkpoints: `result.py` `_validate_model_env` rejects obs-dim mismatch with retrain message.

---

## 10. Action scaling — PASS

```
u_norm ∈ [-1,1]²  →  [ω, a_c]  via PlatePlant._scale_action (once)
```

- Plant dynamics, delay, force, reward, stability lobes use physical values from same map.
- Previous action in obs is **normalized** (`ode_control_env.py` L130, L217+).
- No double rescaling in active `productive` reward path.

---

## 11. Integrator / DDE — PASS

- Default: `integrator="dde_rk4"` (`conf_fast`, `registration.py`).
- RK4 substages at `t`, `t+dt/2`, `t+dt` append scratch history before `k2`–`k4` (`dde_rk4.py`).
- `integrator="rk4"`: faster; delay sampled from accepted-step history only (documented approximation).
- History linear interpolation bounded (`_get_modal_state_at_delay`).

---

## 12. Caching — PASS

**Cached:** `b_vec_series` along tool path (geometry), keyed by rounded `(ω, ac)` + trajectory hash.

**Not cached:** `F_projected`, `Delta_q`, or any η-dependent force (`test_force_depends_on_delta_eta`).

**Warning:** global `f_nonlinear2` cache is process-wide; prefer `n_envs=1` or `SubprocVecEnv`.

---

## 13. Reward and termination — PASS

```
reward = productivity_weight·productivity(ω,ac)
       - vibration_weight·cost(w_s, ẇ_s)
       - smoothness + failure_penalty
```

- Termination: `max|w_s| > displacement_failure_limit` (primary physical guard).
- Optional internal η limit (`use_eta_internal_safety`) labeled numerical guard only.

---

## 14. Stability lobes — PASS

`scripts/stability_lobes.py` — **empirical stochastic lobes**, not analytic Floquet theory.

Outputs per controller mode:
- `stability_lobe_prob_heatmap.png`
- `stability_lobe_boundary.png`
- `stability_lobe_data.npz` + `stability_lobe_summary.csv`
- Optional `action_visit_heatmap.png` (trained controller)

Classification uses physical sensor RMS and failure termination.

---

## 15. Fast config — PASS

`configs/conf_fast.py`: short horizon, 1 seed, small net, `n_envs=1`, small lobe grid.

```bash
python training.py --config conf_fast
python result.py --config conf_fast          # after checkpoints exist
python scripts/stability_lobes.py --config conf_fast
python -m pytest tests/ -q -k "not test_env_checker"
```

---

## 16. Documentation — PASS

- [`DIRECTIONAL_MILLING.md`](DIRECTIONAL_MILLING.md) — reduced surface-normal model, explicit non-equivalence to full paper form.
- This report — end-to-end audit checklist.

---

## Files inspected

| Area | Files |
|------|-------|
| Dynamics / delay | `f_nonlinear2.py`, `dde_rk4.py`, `rk4.py` |
| Force | `milling_force.py`, `milling_config.py`, `mode_ordering.py` |
| Plant / sensors | `plate.py`, `ode_control_env.py` |
| Reward | `plate_rewards.py` |
| RL / eval | `registration.py`, `training.py`, `result.py`, `conf_fast.py` |
| Tools | `scripts/stability_lobes.py`, `scripts/milling_force_diagnostics.py` |
| Tests | `test_mathematical_audit.py`, `test_directional_milling.py`, `test_regenerative_chatter.py`, `test_plate_sensor_interface.py`, `test_end_to_end_consistency.py`, `test_system_audit.py` |
| Docs | `DIRECTIONAL_MILLING.md`, `MATHEMATICAL_AUDIT.md` |

**Changed in this audit:** strengthened audit tests; added `test_wrap_around_engagement`, `test_feed_normal_full_not_implemented`; fixed `f_nonlinear2.py` module docstring; fixed delay history lookup when `t_delayed` equals the first stored sample; fixed `record_modal_state` to update existing timestamps (keeps history valid for bisect).

---

## Remaining approximations (honest)

1. **Reduced one-field model** — not full `Delta_f`/`Delta_n` paper chip form.
2. **Thrust projection** — `F_sn = dF_t sin φ + dF_r cos φ`; in-plane feed force does not excite `q_c`.
3. **constant_tau delay** — local constant-ω over tooth period (default).
4. **Startup** — no regeneration for `t < τ`.
5. **DDE-RK4** — method-of-steps with linear history interpolation (not adaptive DDE solver).
6. **Global cache** — one process-level geometry cache.
7. **Stability lobes** — Monte Carlo simulation classification, not analytic lobes.
8. **Plate modal model** — 4-mode truncated plate; nonlinear cubic on η only.

---

## Reproduction commands

```bash
python -m pytest tests/ -q -k "not test_env_checker"
python scripts/milling_force_diagnostics.py --omega 800 --ac 3 --out plots/milling_diagnostics.png
python scripts/stability_lobes.py --config conf_fast
python training.py --config conf_fast
```

---

## Ready for retraining?

**Yes.** The simulator is mathematically consistent for the reduced surface-normal regenerative model. All pre-change PPO checkpoints are incompatible (6-dim obs, updated force/dynamics). Run full training on `conf1`/`conf2`/`conf3` for publication-quality results; use `conf_fast` for smoke tests.
