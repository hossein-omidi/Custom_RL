# Fundamental stability-lobe diagram for face milling of the flexible plate

This note documents the **theoretically-accepted** way to compute the chatter
stability-lobe diagram (SLD) for our problem — face milling a thin cantilever
plate — and how it relates to (a) the cutting-force / modal model already in the
repository and (b) the three uploaded references. It accompanies
`scripts/stability_lobe_fundamental.py`.

The three references are:

* **[SDM]** Jia, Chen, Song, Huang, *Three-Dimensional Stability Lobe
  Construction for Face Milling of Thin-Wall Components with Position-Dependent
  Dynamics and Process Damping*, **Machines 13 (2025) 524**. Semi-discretization
  for a thin plate reduced to a single axial mode.
* **[ZOA]** Bortolanza & Polli, *Chatter prediction in thin-walled face milling
  considering variations in the workpiece's dynamic characteristics*,
  **J. Braz. Soc. Mech. Sci. 47 (2025) 285**. Frequency-domain zeroth-order
  (Altintas–Budak) SLD with axial direction + process damping.
* **[Review]** Qin, Jiang, Yin, Sun, Wang, *Chatter stability prediction methods
  in the machining processes: a review*, **Int. J. Adv. Manuf. Technol. 136
  (2025) 2945**. Canonical single-frequency (ZOA) and time-/frequency-domain
  method summary.

--------------------------------------------------------------------------------
## 1. Step 1 — verify the applied force and the modal response

The user's requirement is that *the force and modal response are verified before
building the lobe*. They already match the fundamental derivation. Below, each
line of the repository force model (`f_nonlinear2_face_milling.py`) is mapped to
the reference equations.

### 1.1 Cutting-force model (identical to [SDM] §2.1 and [ZOA] §2)

| Quantity | Repository (`f_nonlinear2`) | Reference |
|---|---|---|
| Chip thickness | `h_i = ft·sinθ_i + Δx·sinθ_i·cosγ_L + Δy·cosθ_i·cosγ_L − Δz·sinγ_L` | [SDM] Eq. (10) |
| Tangential/radial/axial force | `Ft=Kt·ap·h+Kte·ap`, `Fr=Kr·ap·h+Kre·ap`, `Fa=Ka·ap·h+Kae·ap` | [SDM] Eqs. (2)–(4) |
| CS1→CS0 projection | `Fx=−Ft·cosθ−Fr·sinθ`, `Fy=Ft·sinθ−Fr·cosθ`, `Fz=Fa` | [SDM] Eq. (8) |
| Engagement window | `θ_s<θ_i<θ_e` (up: `θ_s=0, θ_e=acos(1−2ae/D)`) | [SDM] Eqs. (32)–(33) |
| Process damping (optional) | `B(t)` in `process_damping_matrix()` | [SDM] Eqs. (11), (23)–(31) |
| Full-immersion mean axial force | `F̄z = −(N·ap·Ka/π)·ft − (N·ap·Kae/2)` | [SDM] Eq. (57) |

The lead angle `γ_L` (`gamma_L`, default 45°) is what couples the *axial* plate
motion into the chip thickness through the `−Δz·sinγ_L` term. This is the
physically important term for a plate whose dominant compliance is out-of-plane
(the `z`/axial direction), exactly the assumption of [SDM] §4.1 (their reduction
to a single axial mode) and [ZOA] §2 (their `Fa` axial-force branch).

### 1.2 Modal response (identical to the plate-modal reduction of [SDM] §4.1)

The structural side is a modal expansion of a clamped–free (cantilever) plate,
`w(x,y,t) = Σ_k φ_k(x,y) η_k(t)`, with per-mode equation

```
η_k'' + 2ζ_k ω_k η_k' + ω_k² η_k + λ_k η_k³ = Q_k ,   Q_k = φ_z,k(x_c,y_c)·Fz / M_k
```

* `ω_k`, `ζ_k`, `M_k` — natural frequency, damping ratio, modal mass
  (`compute_natural_frequencies_updated.py`, `compute_modal_mass_vector`).
* `φ_z,k(x_c,y_c)` — mode shape at the **cutter** (`compute_mode_shapes_updated.py`,
  clamped-free × free-free). It enters *twice*: once to project `Fz` into modal
  space, once to reconstruct the surface displacement that closes the
  regenerative loop. This is the `z(x,t)=Σφ_i(x)p_i(t)` reduction of [SDM] Eq. (52).
* `λ_k η_k³` — a Duffing hardening term. It is **not** part of the linear
  stability problem (see §4) and is dropped when forming the lobe.

For the default square AL plate the first six modal frequencies are
`[17.1, 41.2, 133.3, 104.2, 150.6, 263.3] Hz` (index order (0,0),(0,1),(0,2),
(1,0),(1,1),(1,2)); modal mass ≈ 56.2 kg (≈ total plate mass, i.e. the analytical
shapes are essentially mass-normalized). The lowest mode (17 Hz) carries ~5× the
cutter-point compliance of any other and dominates chatter.

**Conclusion of Step 1:** the repository force and modal model already *are* the
Altintas/[SDM]/[ZOA] face-milling model. No change is needed to the physics; the
lobe is built directly from it.

--------------------------------------------------------------------------------
## 2. Step 2 — linearise the regenerative loop

Chatter onset is a *linear* stability question: perturb the stationary cut and
ask whether the perturbation grows. Linearising the model of §1 about the
stationary chip load (drop the static `ft·sinθ` and edge `Kae·ap` parts, which do
not regenerate, and drop `λη³`):

Because `force_projection_mode='z'`, only `Fz=Fa` is projected and only the
`−Δz·sinγ_L` chip-thickness term is dynamic. With the cutter rigid,
`Δz[mm] = −1000·φ_z(x_c,y_c)ᵀ(η(t) − η(t−τ))` (`1000` = m→mm as in the plant), so

```
Q(t) = c(t) · R · ( η(t) − η(t−τ) )
c(t) = 1000 · sin(γ_L) · Ka · ap · g(t)          scalar, periodic through g(t)
R    = (φ_z / M) φ_zᵀ   (K×K, rank-1)            evaluated at the cutter
τ    = 2π /(N ω)         tooth period = regen delay = principal period
g(t) = number of teeth engaged at time t          (periodic, 0/1/… )
```

`g(t)` is the per-instant engaged-tooth count; its average is
`ḡ = N(θ_e−θ_s)/2π` (the ZOA "zeroth-order" directional coefficient). For the
defaults `ae/D = 28/63 = 0.44` (up-milling) → `ḡ = 0.93` teeth: a **highly
interrupted** cut, which matters for method choice in §3.

State-space form (`z=[η;η̇]`), matching [SDM] Eq. (34):

```
ż(t) = A(t) z(t) + B(t) z(t−τ)
A(t) = [[0, I], [−(Ω² − c(t)R), −2ZΩ]] ,   B(t) = [[0,0],[−c(t)R, 0]]
```

with `Ω²=diag(ω_k²)`, `2ZΩ=diag(2ζ_kω_k)`. This is exactly the delayed periodic
system that [SDM] Eqs. (53)–(55) write for the single-mode plate; here it is kept
for `K` modes and reduced to the dominant mode(s) by cutter-point compliance.

--------------------------------------------------------------------------------
## 3. Step 3 — two fundamental solvers

`scripts/stability_lobe_fundamental.py` implements both standard methods.

### 3.1 Semi-Discretization Method (primary) — [SDM], [Review] §4

First-order semi-discretization (Insperger–Stépán). One tooth period `T=τ` is
split into `k` intervals `Δt=τ/k`; because delay = period, the delay spans
exactly `k` steps and the first-order delayed weights are both `½` ([SDM]
Eqs. (41)–(42) with `τ=mΔt`). On each interval `A(t)≈A_i`, `B(t)≈B_i` (frozen at
the midpoint, using `g` there), and the interval map is

```
x_{i+1} = e^{A_iΔt} x_i + A_i⁻¹(e^{A_iΔt} − I) B_i · (½ x_{i−m+1} + ½ x_{i−m})
```

Stacking `[x_i; x_{i−1}; …; x_{i−m}]` gives a discrete map `u_{i+1}=D_i u_i`; the
monodromy (transition) matrix over one period is `Φ = D_{k−1}…D_0` ([SDM]
Eqs. (48)–(51)). By **Floquet theory the cut is stable iff the spectral radius
ρ(Φ) < 1**. The critical depth `ap_lim(rpm)` is found by bisection on `ap` at each
spindle speed; sweeping the cutter position `x_c` gives the **3D** surface
`ap_lim=f(rpm, x_c)` ([SDM] Fig. 10; position-dependence here enters through
`φ_z(x_c,y_c)` in `R`).

Correctness checks built into the code / verified in practice:

* `ρ(ap=0)` reduces to `exp(−ζω_dom τ)` (free damped-mode multiplier) to machine
  precision — a direct check that the monodromy is assembled correctly (observed
  `ρ(ap=0)=0.96831` vs `exp(−ζω₀τ)=0.96831` at 1000 rpm).
* `ρ(Φ)` is converged in `k` (identical lobes for `k=30…200`); default `k=60`.
* The predicted boundary is validated against the **nonlinear plant** (`--verify`,
  §5): small-seed runs grow when `ρ>1` and decay when `ρ<1`.  Measured at the free
  end (`xc=0.85, yc=0.20`), all four probe points agree:

  | rpm | ap [mm] | ρ(SDM) | prediction | nonlinear plant |
  |----:|-------:|------:|-----------|-----------------|
  | 2000 | 1.5 | 0.688 | stable   | decays to forced level |
  | 2000 | 8.0 | 1.578 | UNSTABLE | grows ×8.7 → 52 mm limit cycle |
  | 4000 | 1.5 | 0.751 | stable   | decays |
  | 4000 | 8.0 | 0.932 | stable   | decays to forced level |

  (For a clean test the verification env raises `w_limit` so the *amplitude*
  criterion does not truncate the run before the regenerative envelope reveals
  growth/decay — see §4.)

### 3.2 Zeroth-Order Analytical / single-frequency (cross-check) — [ZOA], [Review] Fig. 7

Using the oriented cutter-point receptance (all modes)

```
Φ_zz(iω) = Σ_k  (φ_z,k² / M_k) / (ω_k² − ω² + 2iζ_k ω_k ω)
```

and the averaged gain `κ = ḡ·Ka·1000·sin(γ_L)` (per unit `ap`), the closed-loop
characteristic equation `1 = ap·κ·(1−e^{−iωτ})·Φ_zz(iω)` gives the analytic lobe

```
ap_lim = 1 / (2 κ Re[Φ_zz(iω_c)])                       (real-axis condition)
ω_c τ  = 2·acot(−Im[Φ_zz]/Re[Φ_zz]) + 2πj  ,  j=0,1,…    (phase → spindle speed)
rpm    = 60 / (N τ)
```

This is the scalar specialisation of Altintas–Budak ([Review] Fig. 7, [ZOA]
Eqs. (16)–(20)). It is plotted as the light scallops + lower envelope.

### 3.3 Which to trust

For **highly interrupted** cutting (here `ḡ≈0.93` teeth engaged) the zeroth-order
frequency method loses accuracy and misses period-doubling ("flip") lobes — a
limitation stated explicitly in [Review] §2.1 and [ZOA] §2. **The
semi-discretization result is therefore the reference boundary**; the ZOA curve
is a fast sanity overlay and will agree only in the wide-engagement, well-damped
regions.

--------------------------------------------------------------------------------
## 4. Why this differs from `stability_lobe_new.py` (and which to use when)

`stability_lobe_new.py` is **not** the classical SLD. It runs the full
*nonlinear* plant at a fixed `[ω, ap]` and calls the trial "stable" if the
*sensor amplitude* stays below `w_limit` for the pass. That answers a different,
also-legitimate question:

| | `stability_lobe_new.py` | `stability_lobe_fundamental.py` |
|---|---|---|
| Question | Does the finished-pass **amplitude** stay under `w_limit`? | Is the stationary cut **linearly** stable (chatter onset)? |
| Method | Nonlinear time simulation + amplitude threshold | Floquet (SDM) / analytic (ZOA) eigenvalue boundary |
| Nonlinearity | Included (`λη³` saturates the limit cycle) | Excluded by definition (linear onset) |
| Literature class | "time-domain simulation + amplitude criterion" ([Review] §3, Table 2) | "time-domain discretization" + "frequency domain" ([Review] §2,§4) |
| Boundary meaning | Tool-safety / surface envelope | Classical stability lobe |
| Cost | High (sim × bisection) | Low (SDOF SDM ≈ ms/point) |

The two boundaries need not coincide: a linearly-unstable cut whose limit cycle
saturates (via `λη³`) below `w_limit` is "stable" to the amplitude test but
"unstable" to the classical test. **For a research-paper stability lobe, use
`stability_lobe_fundamental.py` (SDM).** Keep `stability_lobe_new.py` only if you
specifically want the nonlinear tool-safety amplitude envelope the RL reward uses.

Note also: chatter regenerates at the **cutter**, so the fundamental boundary is
a property of `φ_z` at the cutter — independent of where the observation sensors
sit. The old amplitude test, by contrast, depends on the sensor locations and on
`w_limit`.

--------------------------------------------------------------------------------
## 5. Usage

```bash
# 2D lobe (SDM + ZOA) at one cutter position (near the flexible free end)
python scripts/stability_lobe_fundamental.py --xc 0.85 --yc 0.20 \
       --rpm-min 1000 --rpm-max 40000 --rpm-points 80 \
       --out-dir plots/lobe_fundamental

# 3D position-dependent surface  ap_lim = f(rpm, cutter-x)
python scripts/stability_lobe_fundamental.py --surface-3d \
       --xc-min 0.30 --xc-max 0.98 --xc-points 12 --out-dir plots/lobe_fundamental

# validate the linear boundary against the true nonlinear plant
python scripts/stability_lobe_fundamental.py --verify \
       --verify-rpm 2000 4000 --verify-ap 0.5 6.0
```

Key options: `--n-modes` (modes kept in the SDM, dominant-compliance order;
default 1 = the primary flexible mode as in [SDM] §4.1); `--k-intervals` (SDM
resolution, default 60); `--ae`, `--feed`, `--zeta`, `--milling-mode`, `--E`
(process/material overrides, defaulting to the registered-env values); `--n-lobes`
(ZOA lobe count). All physical parameters default to the exact values used by
`CustomODEPlate-v0`, so the lobe is consistent with the RL/MPC environment.

--------------------------------------------------------------------------------
## 6. Summary

1. The repository's face-milling force and modal response were checked
   term-by-term against [SDM]/[ZOA]; they already implement the accepted model.
2. The classical lobe is obtained by linearising the regenerative loop (§2) and
   solving it with **semi-discretization/Floquet** (primary, [SDM]) and the
   **zeroth-order analytical** method (cross-check, [ZOA]).
3. This replaces the amplitude-threshold heuristic of `stability_lobe_new.py`
   with the theoretically-accepted chatter-onset boundary, at a fraction of the
   cost, and it is consistent with the plant the RL/MPC controllers act on.
