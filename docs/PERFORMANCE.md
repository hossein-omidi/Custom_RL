# Simulation performance — safe speedups (no accuracy loss)

The environment cost is dominated by the regenerative face-milling force, which is
evaluated 4x per RK4 substep x n_substeps per env step (default 40 force calls per
1 ms control step). Two classes of speedup were applied/identified; both respect
the **regenerative-delay constraint** (each RK substep must be smaller than the
minimum tooth-passing delay `tau_min = 2*pi/(N*omega_max)`).

## 1. Code-level optimizations — committed, BIT-IDENTICAL (1.6x)

Verified to reproduce the previous trajectories to `max|w diff| = 0.0` (exact):

- **O(1) delay-history lookup.** `_get_state_at_time` / the kinematic lookups used
  `bisect` on a freshly-rebuilt `[e[0] for e in history]` list *every* delay query
  — O(history), growing with episode length. Now `bisect(..., key=...)` runs
  directly on the history (O(log n), no rebuild). Operating on the same list keeps
  it consistent with the RK4 stage-history rollback.
- **Cache cutter mode shapes per force call.** `_phi_components(xc, yc)` was
  recomputed 3x at the *same* cutter point (displacement, velocity, force
  projection) and once at the delayed point — 4 evaluations, 2-3 redundant. It is
  now evaluated once per distinct point and reused. Mode-shape evaluation was ~47%
  of runtime (closures over cosh/cos/sinh/sin for K=6 modes).

Result: **42 -> 66 env-steps/s (1.57x)**, identical numerics.

## 2. Delay-aware substep reduction — 2x more, 0.55% error (opt-in)

With the realistic `rpm_max = 4000 rpm`, the minimum tooth delay is `tau_min =
3.75 ms`, so the delay constraint no longer forces the fine `dt = 0.1 ms`
(n_substeps = 10) that the old 40000 rpm window required (`tau_min = 0.375 ms`).
Holding the 1 ms control step fixed and coarsening the substep:

| n_substeps | dt [ms] | speedup | RMS error vs n=10 |
|-----------:|--------:|--------:|-------------------:|
| 10 (default) | 0.10 | 1.0x | reference |
| **5 (recommended)** | 0.20 | **2.0x** | **0.55%** |
| 4 | 0.25 | 2.4x | 0.62% |
| 2 | 0.50 | 4.5x | 1.91% |
| 1 | 1.00 | — | rejected by the delay guard |

The dominant plate modes are 17-41 Hz (period 24-59 ms), so dt = 0.2 ms still gives
>100 samples/period — the 0.55% error is far below the 2e-3 modal process noise.
`n_substeps = 1` (dt = 1 ms) is refused by the force module's delay guard, which is
the delay constraint correctly protecting accuracy: a delayed query must land in
already-committed history, not inside an uncommitted RK substep.

**Recommendation.** For new studies use `--n-substeps 5` everywhere (train, eval,
MPC, lobe) for a ~2x speedup at 0.55% error — combined with (1) that is **~3x**
overall (42 -> 131 steps/s). The repo default stays `n_substeps = 10` so existing
trained models keep their exact timing chain; switch the whole pipeline together
(the MPC `control_hold`/`Ts` chain and training/eval must share one n_substeps).

## Not changed (deliberately)
The defensive `nan_to_num` guards on forces/state are kept — they cost ~10% but
prevent silent NaN propagation, which matters for the stiff regenerative dynamics.
Removing them is not a "safe" speedup.
