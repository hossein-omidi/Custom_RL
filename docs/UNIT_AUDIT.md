# Unit Audit Summary

**Overall verdict:** `WARN`

# Unit Consistency Audit

**Model:** `surface_normal_reduced`
**Verdict:** `WARN`

## Unit table (major variables)

| Variable | Symbol / location | Unit |
|----------|-------------------|------|
| Spindle speed | `action_phys[0]`, `omega` | rad/s |
| Axial depth of cut (RL) | `action_phys[1]`, `ac` | mm |
| Axial integration | `z`, `dz`, `ac_m` in `milling_force` | m |
| Plate lengths | `L1`, `L2`, `h` | m |
| Modal displacement | `eta`, `q`, `w_s` | m |
| Modal velocity | `eta_dot`, `w_dot_s` | m/s |
| Modal acceleration | `eta_ddot`, `F_k` | m/s^2 |
| Chip thickness | `h_j`, `f_t`, `Delta_*` | m (poly uses mm via `chip_thickness_for_force_polynomial`) |
| Cutting forces | `F_t`, `F_r`, `F_feed`, `F_normal` | N |
| Lumped modal mass | `M_modal`, `M_modal_f` | kg |
| Mode shapes | `W_k`, `V_k` | dimensionless |

## Modal equation (per mode k)

```
eta_ddot = -2*zeta*omega*eta_dot - omega^2*eta - lambda*eta^3 + F_k
F_k = W_k(x_c,y_c)/M_modal * F_scalar   [N/kg * N = m/s^2]
```

## Code conversion locations

- `custom_rl/plants/units.py` — `ac_to_meters`, `chip_thickness_for_force_polynomial`, coefficient docs
- `custom_rl/plants/milling_force.py` — `ac_to_meters` before axial quadrature; h_m→h_mm in `tangential_radial_increment`
- `custom_rl/plants/plate.py` — physical action bounds; `physical_action_units` metadata
- `custom_rl/rewards/plate_rewards.py` — normalized sensor obs; productivity uses physical `ac` in mm

## Findings

- **[PASS]** `modal_ode_convention` — eta[m], eta_dot[m/s], eta_ddot[m/s^2], omega[rad/s], zeta[-], lambda*eta^3[m/s^2], F_k[m/s^2]
- **[PASS]** `no_magnitude_excitation` — F_normal and F_feed are separate scalars
- **[PASS]** `Fk_units_m_s2` — W*F/M -> 3.149e+01 m/s^2
- **[PASS]** `f_t_from_feed_speed` — f_t=7.854e-05 m/tooth
- **[PASS]** `ac_rl_mm_internal_m` — ac=3.0 mm -> 0.0030 m
- **[PASS]** `ac_zero_zero_force` — 
- **[PASS]** `ac_doubles_force` — |F_n|(0.10mm)/|F_n|(0.05mm)=2.00
- **[PASS]** `cutting_poly_yields_N` — h=0.100 mm, dz=1 mm -> dFt=2.686e+06 N
- **[PASS]** `h_poly_uses_mm` — chip_thickness_for_force_polynomial(0.0001)=0.1
- **[WARN]** `force_clipping_moderate_ac` — MAX_FORCE=1e5 N clip hit for ac>0.05 mm
- **[PASS]** `physical_action_units` — ['rad/s', 'mm']
- **[PASS]** `termination_limits_si` — disp_limit=0.001 m, vel_limit=0.5 m/s
- **[PASS]** `sensor_norm_scales` — disp_norm_scale=0.001 m, vel_norm_scale=0.01 m/s
- **[PASS]** `reward_uses_normalized_sensors` — vibration cost on w_norm, w_dot_norm; productivity on physical omega[rad/s] and ac[mm]

## Modal projection diagnostics

- W_k at contact: [-2.4722, 1.1761] (dimensionless)
- V_k at contact: [2.0000, 3.4641] (dimensionless)
- M_n = M_f = 78.5000 kg
- |F_normal| @ 3 mm, 800 rad/s: 1.000e+05 N
- |F_feed| @ 3 mm, 800 rad/s: 1.000e+05 N
- max |F_k,n|: 3.124e+03 m/s^2
- Note: lumped plate mass L1*L2*rho*h [kg], not orthonormal modal mass

## Chip / feed diagnostics

- f_t: 7.854e-05 m/tooth (0.0785 mm/tooth)
- sample h_j: 0.000e+00 m (0.0000 mm)

## Cutting coefficient units

```
Cutting coefficient unit table (tangential xi, radial delta):
term     orig unit        h_poly dz   SI scale     SI unit    xi value
h^3      N/(mm^3·m)       mm     m    1.000e+00    N/m^1      2.706e+12
h^2      N/(mm^2·m)       mm     m    1.000e+03    N/m^2      -1.964e+09
h^1      N/(mm·m)         mm     m    1.000e+06    N/m^3      1.136e+06
const    N/m              mm     m    1.000e+09    N/m^4      5.280e+01
```

## Force magnitude sweep

| omega [rad/s] | ac [mm] | max h [mm] | |F_n| [N] | |F_feed| [N] | max|F_k,n| [m/s^2] |
|---:|---:|---:|---:|---:|---:|
| 400 | 0.0 | 0.0000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 400 | 1.0 | 0.1494 | 1.000e+05 | 1.000e+05 | 3.130e+03 |
| 400 | 3.0 | 0.1494 | 1.000e+05 | 1.000e+05 | 3.130e+03 |
| 400 | 6.0 | 0.1494 | 1.000e+05 | 1.000e+05 | 3.130e+03 |
| 800 | 0.0 | 0.0000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 800 | 1.0 | 0.0747 | 1.000e+05 | 1.000e+05 | 3.130e+03 |
| 800 | 3.0 | 0.0747 | 1.000e+05 | 1.000e+05 | 3.130e+03 |
| 800 | 6.0 | 0.0747 | 1.000e+05 | 1.000e+05 | 3.130e+03 |
| 1200 | 0.0 | 0.0000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 1200 | 1.0 | 0.0498 | 1.000e+05 | 1.000e+05 | 3.130e+03 |
| 1200 | 3.0 | 0.0498 | 1.000e+05 | 1.000e+05 | 3.130e+03 |
| 1200 | 6.0 | 0.0498 | 1.000e+05 | 1.000e+05 | 3.130e+03 |

## Remaining assumptions

- `M_modal` is a single lumped plate mass, not mode-specific orthonormal mass.
- `lambda_k` from nonlinear stiffness integral assumes `eta` in metres with dimensionless `W_k`.
- Cutting `xi`/`delta` follow legacy MATLAB hybrid (h in mm, dz in m).
- `ac_productive_target` in reward configs is in **mm** (same as physical action).


---

# Unit Consistency Audit

**Model:** `feed_normal_full`
**Verdict:** `WARN`

## Unit table (major variables)

| Variable | Symbol / location | Unit |
|----------|-------------------|------|
| Spindle speed | `action_phys[0]`, `omega` | rad/s |
| Axial depth of cut (RL) | `action_phys[1]`, `ac` | mm |
| Axial integration | `z`, `dz`, `ac_m` in `milling_force` | m |
| Plate lengths | `L1`, `L2`, `h` | m |
| Modal displacement | `eta`, `q`, `w_s` | m |
| Modal velocity | `eta_dot`, `w_dot_s` | m/s |
| Modal acceleration | `eta_ddot`, `F_k` | m/s^2 |
| Chip thickness | `h_j`, `f_t`, `Delta_*` | m (poly uses mm via `chip_thickness_for_force_polynomial`) |
| Cutting forces | `F_t`, `F_r`, `F_feed`, `F_normal` | N |
| Lumped modal mass | `M_modal`, `M_modal_f` | kg |
| Mode shapes | `W_k`, `V_k` | dimensionless |

## Modal equation (per mode k)

```
eta_ddot = -2*zeta*omega*eta_dot - omega^2*eta - lambda*eta^3 + F_k
F_k = W_k(x_c,y_c)/M_modal * F_scalar   [N/kg * N = m/s^2]
```

## Code conversion locations

- `custom_rl/plants/units.py` — `ac_to_meters`, `chip_thickness_for_force_polynomial`, coefficient docs
- `custom_rl/plants/milling_force.py` — `ac_to_meters` before axial quadrature; h_m→h_mm in `tangential_radial_increment`
- `custom_rl/plants/plate.py` — physical action bounds; `physical_action_units` metadata
- `custom_rl/rewards/plate_rewards.py` — normalized sensor obs; productivity uses physical `ac` in mm

## Findings

- **[PASS]** `modal_ode_convention` — eta[m], eta_dot[m/s], eta_ddot[m/s^2], omega[rad/s], zeta[-], lambda*eta^3[m/s^2], F_k[m/s^2]
- **[PASS]** `no_magnitude_excitation` — F_normal and F_feed are separate scalars
- **[PASS]** `Fk_units_m_s2` — W*F/M -> 3.149e+01 m/s^2
- **[PASS]** `f_t_from_feed_speed` — f_t=7.854e-05 m/tooth
- **[PASS]** `ac_rl_mm_internal_m` — ac=3.0 mm -> 0.0030 m
- **[PASS]** `ac_zero_zero_force` — 
- **[PASS]** `ac_doubles_force` — |F_n|(0.10mm)/|F_n|(0.05mm)=2.00
- **[PASS]** `cutting_poly_yields_N` — h=0.100 mm, dz=1 mm -> dFt=2.686e+06 N
- **[PASS]** `h_poly_uses_mm` — chip_thickness_for_force_polynomial(0.0001)=0.1
- **[WARN]** `force_clipping_moderate_ac` — MAX_FORCE=1e5 N clip hit for ac>0.05 mm
- **[PASS]** `physical_action_units` — ['rad/s', 'mm']
- **[PASS]** `termination_limits_si` — disp_limit=0.001 m, vel_limit=0.5 m/s
- **[PASS]** `sensor_norm_scales` — disp_norm_scale=0.001 m, vel_norm_scale=0.01 m/s
- **[PASS]** `reward_uses_normalized_sensors` — vibration cost on w_norm, w_dot_norm; productivity on physical omega[rad/s] and ac[mm]

## Modal projection diagnostics

- W_k at contact: [-2.4722, 1.1761] (dimensionless)
- V_k at contact: [2.0000, 3.4641] (dimensionless)
- M_n = M_f = 78.5000 kg
- |F_normal| @ 3 mm, 800 rad/s: 1.000e+05 N
- |F_feed| @ 3 mm, 800 rad/s: 1.000e+05 N
- max |F_k,n|: 3.124e+03 m/s^2
- Note: lumped plate mass L1*L2*rho*h [kg], not orthonormal modal mass

## Chip / feed diagnostics

- f_t: 7.854e-05 m/tooth (0.0785 mm/tooth)
- sample h_j: 4.643e-05 m (0.0464 mm)

## Cutting coefficient units

```
Cutting coefficient unit table (tangential xi, radial delta):
term     orig unit        h_poly dz   SI scale     SI unit    xi value
h^3      N/(mm^3·m)       mm     m    1.000e+00    N/m^1      2.706e+12
h^2      N/(mm^2·m)       mm     m    1.000e+03    N/m^2      -1.964e+09
h^1      N/(mm·m)         mm     m    1.000e+06    N/m^3      1.136e+06
const    N/m              mm     m    1.000e+09    N/m^4      5.280e+01
```

## Force magnitude sweep

| omega [rad/s] | ac [mm] | max h [mm] | |F_n| [N] | |F_feed| [N] | max|F_k,n| [m/s^2] |
|---:|---:|---:|---:|---:|---:|
| 400 | 0.0 | 0.0000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 400 | 1.0 | 0.1494 | 1.000e+05 | 1.000e+05 | 3.130e+03 |
| 400 | 3.0 | 0.1494 | 1.000e+05 | 1.000e+05 | 3.130e+03 |
| 400 | 6.0 | 0.1494 | 1.000e+05 | 1.000e+05 | 3.130e+03 |
| 800 | 0.0 | 0.0000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 800 | 1.0 | 0.0747 | 1.000e+05 | 1.000e+05 | 3.130e+03 |
| 800 | 3.0 | 0.0747 | 1.000e+05 | 1.000e+05 | 3.130e+03 |
| 800 | 6.0 | 0.0747 | 1.000e+05 | 1.000e+05 | 3.130e+03 |
| 1200 | 0.0 | 0.0000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 1200 | 1.0 | 0.0498 | 1.000e+05 | 1.000e+05 | 3.130e+03 |
| 1200 | 3.0 | 0.0498 | 1.000e+05 | 1.000e+05 | 3.130e+03 |
| 1200 | 6.0 | 0.0498 | 1.000e+05 | 1.000e+05 | 3.130e+03 |

## Remaining assumptions

- `M_modal` is a single lumped plate mass, not mode-specific orthonormal mass.
- `lambda_k` from nonlinear stiffness integral assumes `eta` in metres with dimensionless `W_k`.
- Cutting `xi`/`delta` follow legacy MATLAB hybrid (h in mm, dz in m).
- `ac_productive_target` in reward configs is in **mm** (same as physical action).
