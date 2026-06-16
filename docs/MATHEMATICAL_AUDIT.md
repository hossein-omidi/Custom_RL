# Mathematical Audit Report — Plate Milling RL Simulator

## a) Where delay is applied

Regenerative delay enters **only** in the cutting-force / chip-thickness path:

- Contact displacement: `w_c(t) = Σ_k W_k(x_c(t), y_c(t)) η_k(t)`
- Delayed displacement: `w_c(t_delay)` using modal state at `t_delay = t - delay`
- Dynamic chip component: `Δw_c = w_c(t) - w_c(t_delay)`
- Total chip thickness: `h_total = h_geom(t) + Δw_c` (geometry from tool path, cached)
- Cutting force: polynomial in `h_total` with coefficients scaled by `ac`

**Structural ODE uses current state only:**

```
η̈_k = -2ζ_k ω_k η̇_k - ω_k² η_k - λ_k η_k³ + F_k(t)
```

No delayed damping, stiffness, or nonlinearity.

## b) Why structural terms are not delayed

Delaying structural terms would model a **retarded oscillator**, not regenerative chatter. In milling theory, regeneration is a **surface / chip-thickness feedback** loop: the cut depth depends on the difference between the current and previously machined surface. That feedback appears in the **excitation** (cutting force), not in the plate's intrinsic modal parameters.

## c) How ω and a_c variation is handled

| Stage | Handling |
|-------|----------|
| Agent action | Normalized `u ∈ [-1,1]²` |
| Plant input | Affine map → physical `[ω, a_c]` once in `PlatePlant._scale_action` |
| Tooth delay | `constant_tau`: `τ = 2π/(N·ω_current)` (local constant-speed approx) |
| | `spindle_phase`: find `t_prev` with `θ(t)-θ(t_prev)=2π/N` from recorded ω history |
| Force | `ac` scales `ξ, δ` coefficients; `h≤0` or `ac≤0` → zero force |
| Cache | Keyed by rounded `(ω, ac)` + trajectory checksum; geometry cached, `η`-dependent force at runtime |
| Reward | `info["action_phys"]` from same scaled action as plant |
| Observation | Previous **normalized** action appended to sensor block |

**Approximation:** `constant_tau` assumes ω is piecewise constant over each tooth period. Use `delay_mode="spindle_phase"` when ω changes within an episode.

## d) Stability / instability classification

`scripts/stability_lobes.py` Monte Carlo grid over `(ω, a_c)`:

- **Uncontrolled:** fixed normalized action per grid point
- **Trained (optional):** PPO policy rollouts
- **Unstable if:** sensor failure termination OR post-transient RMS(`|w_s|`) > `rms_factor ×` initial RMS OR above displacement limit
- **Output:** `p_unstable(ω, a_c)`, critical `a_c` boundary at threshold (default 0.5)

## e) Approximations and limitations

1. **Scalar normal chip model:** `h_total = h_geom + Δw_c` with polynomial `F(h)`; full directional immersion `h(φ)` not yet implemented.
2. **Constant τ delay:** default mode; phase-based delay available for variable ω.
3. **DDE integration:** `dde_rk4` method-of-steps RK4 with history interpolation at each sub-stage; `rk4` available for comparison (delay from accepted-step history only).
4. **Global `f_nonlinear2` cache:** one cache per process; use `n_envs=1` or `SubprocVecEnv`.
5. **Stability lobes:** empirical classification from simulation, not analytic Floquet/lobe theory.

## Integrator selection

```python
# registration / env kwargs
integrator="dde_rk4"   # recommended (default)
integrator="rk4"       # faster, coarser delay at RK4 substages
delay_mode="constant_tau" | "spindle_phase"
```

## Quick commands

```bash
python training.py --config conf_fast
python result.py --config conf_fast
python scripts/stability_lobes.py --config conf_fast
python -m pytest tests/test_mathematical_audit.py -q
```
