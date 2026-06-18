# Two-Direction Regenerative Milling Force

## Scope

The project implements a **two-direction regenerative milling model inspired by Nasiri/Moradi**, adapted to **straight-line surface milling** through the middle of the plate. It uses:

- two modal displacement fields (W normal, V feed),
- tooth-period regenerative delay,
- directional chip thickness with `Delta_f` and `Delta_n`,
- engagement `g(phi)`,
- tangential/radial cutting forces,
- **separate** feed/normal force projections into V and W modal subsystems,
- sensor-level RL interface (agent observes physical W-field sensors).

Legacy `surface_normal_reduced` (single W-field) remains available.

---

## 1. Straight-line tool path

Default `trajectory_mode="middle_line"`:

```
x_c(t) = x_mid          (default L1/2)
y_c(t) = y_start + s_feed * feed_speed * t
```

- `s_feed = sign(y_end - y_start)`
- Path stays inside `0 <= x_c <= L1`, `0 <= y_c <= L2`
- `e_f`: feed direction along the path (default ±y)
- `e_n`: normal-to-feed in the plate plane (default x)

Alternative: `trajectory_mode="pass_grid"` for parallel passes across the plate width.

---

## 2. Two modal displacement fields

**Normal (W-field):**

```
q_n(t,x,y) = sum_k W_k(x,y) * eta_n,k(t)
```

**Feed (V-field):**

```
q_f(t,z,y) = sum_k V_k(z_c,y) * eta_f,k(t)
```

`z_c` defaults to plate thickness `h` (machined surface).

**State (feed_normal_full, 4K):**

```
x = [eta_n, eta_dot_n, eta_f, eta_dot_f]   # each subsystem 2K interleaved
```

---

## 3. Regenerative chip thickness

```
Delta_n = q_n(t, x_c, y_c) - q_n(t_delay, x_cd, y_cd)
Delta_f = q_f(t, z_c, y_c) - q_f(t_delay, z_c, y_cd)

h_j = [ f_t*sin(phi_j) + Delta_f*sin(phi_j) + Delta_n*cos(phi_j) ] * g(phi_j)
```

Delay: `tau = 2*pi/(N*omega)` or spindle-phase `theta(t)-theta(t_delay)=2*pi/N`.  
Delay appears **only** in chip thickness.

---

## 4. Force projection

```
dF_feed   = -dF_t*cos(phi) - dF_r*sin(phi)
dF_normal =  dF_t*sin(phi) - dF_r*cos(phi)

F_n,k = W_k(x_c,y_c)/M_n * F_normal_total
F_f,k = V_k(z_c,y_c)/M_f * F_feed_total
```

Never use `sqrt(F_feed^2 + F_normal^2)` for modal excitation.

---

## 5. Structural dynamics (current state only)

```
eta_ddot_n,k = -2*zeta_n,k*omega_n,k*eta_dot_n,k - omega_n,k^2*eta_n,k - lambda_n,k*eta_n,k^3 + F_n,k
eta_ddot_f,k = -2*zeta_f,k*omega_f,k*eta_dot_f,k - omega_f,k^2*eta_f,k - lambda_f,k*eta_f,k^3 + F_f,k
```

- `omega_n`, `lambda_n` from plate bending frequencies / nonlinear stiffness (W-field)
- `omega_f`, `lambda_f` from `compute_natural_frequencies_prim` / `lambda_prime_mn` (V-field)

---

## 6. Units (SI internal, mm at RL boundary)

| Quantity | Agent / physical action | Internal force integration |
|----------|-------------------------|----------------------------|
| ω | rad/s | rad/s |
| a_c | **mm** (`ac_min`…`ac_max`, default 0…10 mm) | converted to m via `ac_to_meters` |
| f_t | — | m (from feed speed or `cf_units`) |
| h_j, q, plate geometry | — | m |
| F_t, F_r, F_feed, F_normal | — | N |

Axial quadrature uses `z ∈ [0, a_c]` in metres after mm→m conversion.

---

## 7. Remaining approximations

- Straight-line middle-of-plate path (configurable entry/exit y)
- Surface milling engagement defaults (`phi_ex - phi_st = pi`, configurable)
- V-field dynamics from prime-frequency table
- Empirical stability lobes (not analytic Floquet)
- Agent observes W-field sensors only (feed modes hidden unless extended)

---

## Commands

```bash
python scripts/milling_force_diagnostics.py --omega 800 --ac 3
python -m pytest tests/test_feed_normal_full.py -q
python training.py --config conf_fast
```
