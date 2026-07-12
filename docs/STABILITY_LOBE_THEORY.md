# Fundamental stability-lobe diagram — theory, implementation, validation

Companion note for `scripts/stability_lobe_fundamental.py` (face milling of the
clamped-free plate). References: **[SDM]** Jia et al., Machines 13 (2025) 524;
**[ZOA]** Bortolanza & Polli, J.Braz.Soc.Mech.Sci. 47 (2025) 285; **[Review]**
Qin et al., Int.J.Adv.Manuf.Technol. 136 (2025) 2945.

## 1. Model verification (force + modal response)
The repo force model maps term-by-term onto the references: chip thickness
`h = ft sinθ + Δx sinθ cosγ_L + Δy cosθ cosγ_L − Δz sinγ_L` ([SDM] Eq.10),
forces `Ft,Fr,Fa = (Kt,Kr,Ka)·ap·h + edge` (Eqs.2–4), CS1→CS0 projection
(Eq.8), engagement window (Eqs.32–33), process damping (Eqs.11,23–31). The
modal side is `η_k''+2ζωη_k'+ω²η_k+λη_k³ = φ_k(x_c,y_c)Fz/M_k` ([SDM] Eq.52).
Plate modes: [17.1, 41.2, 133.3, 104.2, 150.6, 263.3] Hz, M ≈ 56.2 kg.

## 2. Numerical (black-box) linearisation
The regenerative coupling is extracted by central finite differences of the
plant's own `compute_face_milling_force` at k angles over one tooth pitch:
Jp=∂Q/∂η, Jv=∂Q/∂η̇ (process damping), Jpd=∂Q/∂η_delayed. Exact because the
periodic orbit satisfies x(t)=x(t−τ) (regenerative term vanishes on the orbit;
force piecewise-linear in states) and the force is linear in ap (unit-ap
Jacobians scale exactly). Cross-check vs the closed form
`Jp=1000 sinγ_L·Ka·ap·g(θ)·(φ/M)φᵀ`: agreement ~3e-12; `Jpd≈−Jp` to ~1e-3
(the difference is the delayed-cutter-position effect, captured automatically).

## 3. Solvers
**Semi-discretization/Floquet (primary).** First-order SDM over one tooth
period (delay = period; delayed weights ½,½); monodromy Φ; stable iff ρ(Φ)<1;
the SLD is the ρ=1 contour, shown as a HEAT MAP of ρ(rpm,ap) with the lobe
line. Low-rank delayed coupling (rank-1: rows ∝ φᵀ) shrinks the map dimension
to 2n+r·k (~15 ms per point, all modes). **ALL K plant modes are kept by
default** — compliance truncation is unsafe for mode-COUPLING flutter: modes
{0,1} alone predict instability at (2000 rpm, 1.5 mm) where the 6-mode system
and the true plant are stable. Built-in checks each run: FD-vs-analytic
Jacobian, ρ(ap=0)=exp(−ζωτ) to 6+ digits, k-convergence, reduced==dense
monodromy.

**ZOA (cross-check).** `ap_lim = 1/(2κ·Re Φ_zz(iω_c))`,
`ω_cτ = 2·atan2(1,−Im/Re)+2πj`, κ = ḡ·Ka·1000·sinγ_L. For this interrupted cut
(ḡ≈0.93) SDM is the reference; with all modes the two nearly coincide
(0.138 vs 0.115 mm at 1000 rpm).

## 4. Validation vs the nonlinear plant (6/6)
| rpm | ap[mm] | ρ | pred | plant |
|---|---|---|---|---|
|1000|0.05|0.949|stable|decays|
|1000|0.5|1.252|UNSTABLE|grows ×10.7|
|1000|3.0|2.078|UNSTABLE|grows ×23.4|
|2000|0.05|0.958|stable|decays|
|2000|0.5|0.915|stable|decays|
|2000|3.0|1.298|UNSTABLE|grows ×8.2|

Caveat: at deep cuts the *forced* response reaches mm level and the plant's
cubic (Duffing) hardening shifts tangent stiffness by tens of % — outside any
classical linear SLD's assumptions; this is where the nonlinear
amplitude-envelope tool (`stability_lobe_new.py`) legitimately differs.

## 5. Operating range and lobe structure
Scallops of a mode at f sit at rpm = 60f/(N·j). With the realistic range
400–4000 rpm, the dominant-mode scallops (17.1 Hz → ≤256 rpm-ish tails,
41.2 Hz → ≤618 rpm) fall INSIDE/below the window's lower part; cusps appear
near ~250/650/2500/3400 rpm in the zoom. Speed selection therefore genuinely
matters in this range (unlike the old 1000–40000 rpm window that sat beyond
the last lobe).

## 6. Position-dependent 3D surfaces (x and y0)
`--surface-3d` sweeps the cutter x (feed direction); `--surface-3d-y` sweeps
the milling line y0 at fixed xc. Both use the identical verified computation:
position enters ONLY through the cutter-point mode shape φ_z(x_c,y_c), exactly
as in the plant force projection. Correctness check: the y0 surface is
symmetric about y0=0.5 (free-free y-modes), observed exactly in the output.

## 7. Usage
```bash
python scripts/stability_lobe_fundamental.py                       # heat map + rho=1 line, 400-4000 rpm
python scripts/stability_lobe_fundamental.py --mode boundary       # bisection curve only
python scripts/stability_lobe_fundamental.py --surface-3d          # ap_lim = f(rpm, cutter-x)
python scripts/stability_lobe_fundamental.py --surface-3d-y        # ap_lim = f(rpm, milling line y0)
python scripts/stability_lobe_fundamental.py --verify              # validate vs nonlinear plant
```
Key flags: `--n-modes -1` (all modes, default) / `--k-intervals 48` (auto-raised
at low rpm) / `--coupling numerical|analytic` / `--ap-cap --ap-points
--rpm-points` / `--ae --feed --zeta --milling-mode --E`.
