# Regenerative Milling Calibration

This model is a **two-direction regenerative surface-milling model inspired by Nasiri/Moradi**. It uses:

- current-state structural dynamics only (no delayed stiffness/damping),
- tooth-period delayed chip thickness (`Delta_f`, `Delta_n`),
- directional feed/normal force projection into W and V subsystems,
- straight middle-line tool pass (`x_c = L1/2`, monotonic `y_c`).

## Force calibration

Legacy MATLAB polynomial coefficients (`xi`, `delta`) are scaled by
`FORCE_COEFFICIENT_SCALE = 1e-2` with extra `HIGH_ORDER_FORCE_SCALE = 1e-2` on the
h^3/h^2 terms (legacy mm–m hybrid is overly stiff for small regenerative chip variation).

Integration sub-step dt is chosen from `custom_rl/plants/time_scales.py` (modal RK4
stability, tooth delay, tooth-pass resolution) — not a single hard-coded value.

Typical macro step: 2 ms with ~32 substeps for `feed_normal_full`.

- Typical cuts at `ac = 1–10 mm` produce **O(10²–10⁴) N** directional force.
- `MAX_FORCE_SAFETY_N = 1e6` is a numerical guard only (not active in normal operation).
- Stable cuts show **bounded oscillation**; unstable cuts show **regenerative chatter-like growth** — not force-clip saturation.

Chip thickness uses **m** internally; the force polynomial evaluates **h in mm** with **dz in m** (legacy hybrid convention).

## Verification

```bash
python custom_rl/plants/Test2.py --mode feed_normal_full
python -m pytest tests/test_force_calibration.py -q
```

The calibration sweep in `Test2.py` reports per `(omega, ac)`:

max `h_j`, `Delta_f`, `Delta_n`, forces, clip %, sensor displacement, and class (`stable` / `bounded` / `chatter-like` / `failed`).
