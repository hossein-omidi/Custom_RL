"""Fundamental (Floquet/eigenvalue) stability-lobe diagram for face milling of the
flexible cantilever plate — numerical semi-discretization with a spectral-radius
HEAT MAP and the ZOA analytic cross-check.

What this computes
------------------
The classical chatter-onset boundary of the *linearised* regenerative process for
our case study (face milling across a clamped-free plate), using:

  1. Semi-Discretization Method (SDM, Insperger & Stepan) -> Floquet monodromy
     spectral radius rho(rpm, ap).  The cut is stable iff rho < 1; the stability
     lobe is the rho = 1 contour.  This is the method used for thin-wall face
     milling in Jia et al., Machines 13 (2025) 524, kept multi-mode here.
     The default output is a HEAT MAP of rho over the (rpm, ap) plane with the
     rho=1 boundary line drawn on it (plus a refined boundary curve in the CSV).

  2. Zeroth-Order Analytical (Altintas-Budak single-frequency) lobe from the
     oriented cutter-point FRF (Bortolanza & Polli 2025; Qin et al. review
     2025, Fig. 7) — plotted as an overlay for cross-checking.  For this highly
     interrupted cut (~0.93 teeth engaged on average) the SDM is the reference.

NUMERICAL (black-box) linearisation
-----------------------------------
The RL environment is registered as an (approximately) black box, so the
regenerative coupling is NOT hard-coded from hand-derived formulas.  Instead the
periodic Jacobians of the plant's own force routine
(`f_nonlinear2_face_milling.compute_face_milling_force`) are extracted by central
finite differences around the stationary cut:

    Jp(theta_i)  = d(modal force)/d(eta)          current positions
    Jv(theta_i)  = d(modal force)/d(eta_dot)      current velocities (process damping)
    Jpd(theta_i) = d(modal force)/d(eta_delayed)  regenerative (delayed) positions

sampled at k angles over one tooth pitch (the coefficient period).  This is
rigorous because (a) the periodic stationary orbit satisfies x(t)=x(t-tau), so
the Jacobian at the origin equals the Jacobian on the orbit while the chip stays
loaded (the force is piecewise-linear in the states), and (b) the force is exactly
linear in ap, so Jacobians extracted at ap_ref = 1 mm scale exactly with ap.
The hand-derived closed form  Jp = 1000*sin(gamma_L)*Ka*ap*g(theta) * (phi/M)phi^T
is retained as an automatic cross-check (`--coupling analytic` also available).

Efficiency (low-rank delayed coupling)
--------------------------------------
The delayed Jacobian has low rank r (rank-1 for the z-only force projection:
rows ~ phi^T).  The delayed history therefore only needs the r scalars
s_j = V^T eta(t - j*dt) instead of full states, shrinking the semi-discretization
state from 2n(k+1) to 2n + r*k (n = retained modes).
ALL K plant modes are kept by default: the plant IS a K-mode system, so this is
its exact linearisation, and compliance-based truncation is NOT safe for
mode-coupling flutter (verified here: modes {0,1} alone predict instability at
(2000 rpm, 1.5 mm) where the full 6-mode system -- and the true nonlinear plant --
are stable).  --n-modes >0 / 0(auto) remain available for speed studies.  The
SDM interval count adapts to resolve the fastest retained mode at low speeds.

Where the scalloped lobes are for THIS plate
--------------------------------------------
The classic scallops of a mode at f_n [Hz] sit at rpm = 60*f_n/(N*j).  Our plate
is dominated by 17.1 Hz (and 41.2 Hz) modes while N = 4, so their scallops sit
below ~620 rpm — inside the realistic 400-4000 rpm operating window used by the
project, whose upper part lies beyond the last lobe (smooth rising boundary).

Usage
-----
    # rho(rpm, ap) HEAT MAP + rho=1 boundary line (default mode)
    python scripts/stability_lobe_fundamental.py --xc 0.85 --out-dir plots/lobe_fundamental

    # bisection boundary curve only (fast, CSV)
    python scripts/stability_lobe_fundamental.py --mode boundary

    # 3D surface ap_lim = f(rpm, cutter x-position)   (feed-direction sweep)
    python scripts/stability_lobe_fundamental.py --surface-3d

    # 3D surface ap_lim = f(rpm, milling line y0)     (pass-line sweep)
    python scripts/stability_lobe_fundamental.py --surface-3d-y

    # validate the linear boundary against the true nonlinear plant
    python scripts/stability_lobe_fundamental.py --verify
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

# Make the package importable when the script is run directly (python scripts/..).
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from custom_rl.plants.plate import PlatePlant, omega_to_rpm, rpm_to_omega
from custom_rl.plants import f_nonlinear2_face_milling as f2
from custom_rl.plants.compute_mode_shapes_updated import build_mode_shape_matrix


# ---------------------------------------------------------------------------
# numpy-only matrix exponential (scipy is used if available, else Taylor s&s)
# ---------------------------------------------------------------------------
try:  # optional fast path
    from scipy.linalg import expm as _scipy_expm  # type: ignore

    def matrix_exp(A: np.ndarray) -> np.ndarray:
        return np.asarray(_scipy_expm(A), dtype=np.complex128)
except Exception:  # numpy-only scaling-and-squaring with a Taylor core

    def matrix_exp(A: np.ndarray) -> np.ndarray:
        A = np.asarray(A, dtype=np.complex128)
        norm = float(np.max(np.sum(np.abs(A), axis=1))) if A.size else 0.0
        s = max(0, int(np.ceil(np.log2(norm + 1.0))) + 1)
        As = A / (2.0**s)
        E = np.eye(A.shape[0], dtype=np.complex128)
        term = np.eye(A.shape[0], dtype=np.complex128)
        for kk in range(1, 24):
            term = term @ As / kk
            E = E + term
            if np.max(np.abs(term)) < 1e-18:
                break
        for _ in range(s):
            E = E @ E
        return E


# ---------------------------------------------------------------------------
# Dynamic model wrapper around the RL plant
# ---------------------------------------------------------------------------
@dataclass
class FaceMillingStabilityModel:
    """Linearised regenerative face-milling model, consistent with PlatePlant."""

    plant: Any
    omega_vec: np.ndarray = field(init=False)   # natural freqs [rad/s], (K,)
    zeta_vec: np.ndarray = field(init=False)    # damping ratios,   (K,)
    M_modal: np.ndarray = field(init=False)     # modal masses [kg], (K,)
    K: int = field(init=False)
    N: int = field(init=False)
    Ka: float = field(init=False)
    gamma_L: float = field(init=False)
    theta0: float = field(init=False)
    D_mm: float = field(init=False)
    ae_mm: float = field(init=False)
    milling_mode: str = field(init=False)
    disp_to_mm: float = field(init=False)

    def __post_init__(self) -> None:
        p = self.plant
        self.omega_vec = np.asarray(p.omega_vec, dtype=np.float64).reshape(-1)
        self.zeta_vec = np.asarray(p.zeta_vec, dtype=np.float64).reshape(-1)
        self.M_modal = np.asarray(p.M_modal, dtype=np.float64).reshape(-1)
        self.K = int(p.K)
        self.N = int(p.N)
        self.Ka = float(p.Ka)
        self.gamma_L = float(p.gamma_L)
        self.theta0 = float(getattr(p, "theta0", 0.0))
        self.D_mm = float(p.D_mm)
        self.ae_mm = float(p.ae_default)
        self.milling_mode = str(p.milling_mode)
        self.disp_to_mm = 1000.0  # f_nonlinear2.MODAL_DISPLACEMENT_TO_MM
        self._jac_cache: dict = {}

    # -- geometry / engagement ------------------------------------------------
    def phi_at(self, xc: float, yc: float) -> np.ndarray:
        """Cutter-point mode-shape vector phi_z (K,), identical to the plant."""
        return build_mode_shape_matrix(
            self.plant.W_mn, [(float(xc), float(yc))], self.plant.m_max, self.plant.n_max
        ).reshape(-1)

    def entry_exit_angles(self) -> tuple[float, float]:
        immersion = float(np.clip(self.ae_mm / max(self.D_mm, 1e-12), 0.0, 1.0))
        mode = self.milling_mode.lower()
        if mode in {"down", "climb", "climb_milling"}:
            return float(np.arccos(np.clip(2.0 * immersion - 1.0, -1.0, 1.0))), float(np.pi)
        return 0.0, float(np.arccos(np.clip(1.0 - 2.0 * immersion, -1.0, 1.0)))

    def engaged_count(self, t: float, omega: float) -> int:
        theta_s, theta_e = self.entry_exit_angles()
        psi = omega * t + self.theta0
        pitch = 2.0 * np.pi / self.N
        return sum(
            1 for i in range(self.N)
            if theta_s < (psi + i * pitch) % (2.0 * np.pi) < theta_e
        )

    def mean_engaged(self) -> float:
        theta_s, theta_e = self.entry_exit_angles()
        return self.N * (theta_e - theta_s) / (2.0 * np.pi)

    # -- oriented FRF at the cutter (all K modes), for the ZOA cross-check ----
    def oriented_frf(self, omega: float, phi: np.ndarray) -> complex:
        wn = self.omega_vec
        denom = (wn**2 - omega**2) + 2j * self.zeta_vec * wn * omega
        return complex(np.sum((phi**2 / self.M_modal) / denom))

    def kappa_per_ap(self) -> float:
        """Averaged regenerative gain per unit ap: gbar*Ka*1000*sin(gamma_L)."""
        return self.mean_engaged() * self.Ka * self.disp_to_mm * np.sin(self.gamma_L)

    # -- mode selection by cutter-point compliance -----------------------------
    def mode_indices(self, phi: np.ndarray, n_modes: int, rel_threshold: float) -> np.ndarray:
        """Retained-mode indices, ranked by static compliance phi^2/(M*wn^2).

        n_modes < 0 keeps all K modes (default; exact linearisation of the
        plant).  n_modes == 0 keeps every mode within rel_threshold of the
        maximum compliance (auto).  n_modes > 0 keeps the top n.
        """
        compliance = phi**2 / (self.M_modal * self.omega_vec**2)
        order = np.argsort(compliance)[::-1]
        if n_modes < 0:
            return np.arange(self.K)
        if n_modes > 0:
            return np.sort(order[: int(np.clip(n_modes, 1, self.K))])
        keep = compliance >= rel_threshold * float(compliance.max())
        keep[order[0]] = True
        return np.flatnonzero(keep)

    # ======================================================================
    # NUMERICAL (black-box) periodic Jacobians of the plant force routine
    # ======================================================================
    def periodic_jacobians(self, xc: float, yc: float, k: int, *,
                           coupling: str = "numerical", eps: float = 1e-8) -> dict:
        """Extract Jp/Jv/Jpd at k angles over one tooth pitch (unit ap = 1 mm).

        The plant force is exactly linear in ap, so these scale with ap; the
        cutting Jacobians are independent of omega (the angle grid is what
        matters), and the process-damping velocity Jacobian scales as 1/omega
        (handled via 'pd' + 'omega_ref').  Delayed-Jacobian low-rank factors
        (V, L) enable the reduced semi-discretization state.
        """
        key = (round(float(xc), 9), round(float(yc), 9), int(k), coupling)
        if key in self._jac_cache:
            return self._jac_cache[key]

        n = self.K
        p = self.plant
        omega_ref = 0.5 * (float(p.omega_min) + float(p.omega_max))
        tau_ref = 2.0 * np.pi / (self.N * omega_ref)
        pitch_dt = tau_ref  # one tooth pitch in time at omega_ref

        Jp = np.zeros((k, n, n))
        Jv = np.zeros((k, n, n))
        Jpd = np.zeros((k, n, n))
        jvd_max = 0.0
        g_samples = np.zeros(k)

        if coupling == "analytic":
            phi = self.phi_at(xc, yc)
            R = np.outer(phi / self.M_modal, phi)
            c_unit = self.disp_to_mm * np.sin(self.gamma_L) * self.Ka  # per mm ap, per tooth
            for i in range(k):
                t_eval = (i + 0.5) / k * pitch_dt + tau_ref
                g = self.engaged_count(t_eval, omega_ref)
                g_samples[i] = g
                Jp[i] = c_unit * g * R
                Jpd[i] = -Jp[i]
            check = {"coupling": "analytic"}
        elif coupling == "numerical":
            # save & pin the force-module state so the cutter sits at (xc, yc)
            saved = (f2.x0_cutter, f2.y_cutter,
                     getattr(f2, "USE_ACCEPTED_PATH_KINEMATICS", False))
            try:
                f2.USE_ACCEPTED_PATH_KINEMATICS = False
                f2.y_cutter = float(yc)
                feed = f2.feed_rate_m_s(omega_ref)
                x0v = np.zeros(2 * n)
                for i in range(k):
                    # evaluate one pitch later so t_delay = t - tau >= 0; the
                    # tooth pattern is pitch-periodic so the angles are identical
                    t_eval = (i + 0.5) / k * pitch_dt + tau_ref
                    f2.x0_cutter = float(p.L1 - float(xc) - feed * t_eval)

                    def Q(x, xd):
                        _, Fm, _ = f2.compute_face_milling_force(
                            t=t_eval, x_modal=x, x_delay_modal=xd,
                            omega_rad_s=omega_ref, ap_mm=1.0, ae_mm=self.ae_mm,
                            t_delay_s=t_eval - tau_ref, tau_s=tau_ref,
                            delay_available=True)
                        return np.asarray(Fm, dtype=np.float64)

                    for j in range(n):
                        for slot, J in ((2 * j, Jp), (2 * j + 1, Jv)):
                            xp = x0v.copy(); xp[slot] = +eps
                            xm = x0v.copy(); xm[slot] = -eps
                            J[i, :, j] = (Q(xp, x0v) - Q(xm, x0v)) / (2 * eps)
                        # delayed positions
                        xp = x0v.copy(); xp[2 * j] = +eps
                        xm = x0v.copy(); xm[2 * j] = -eps
                        Jpd[i, :, j] = (Q(x0v, xp) - Q(x0v, xm)) / (2 * eps)
                        # delayed velocities must be inert in this model
                        xp = x0v.copy(); xp[2 * j + 1] = +eps
                        xm = x0v.copy(); xm[2 * j + 1] = -eps
                        jvd_max = max(jvd_max, float(np.max(np.abs(
                            (Q(x0v, xp) - Q(x0v, xm)) / (2 * eps)))))
                    g_samples[i] = self.engaged_count(t_eval, omega_ref)
            finally:
                f2.x0_cutter, f2.y_cutter, f2.USE_ACCEPTED_PATH_KINEMATICS = saved

            # cross-check against the hand-derived closed form
            phi = self.phi_at(xc, yc)
            R = np.outer(phi / self.M_modal, phi)
            c_unit = self.disp_to_mm * np.sin(self.gamma_L) * self.Ka
            Jp_an = np.stack([c_unit * g_samples[i] * R for i in range(k)])
            scale = max(float(np.max(np.abs(Jp_an))), 1e-30)
            err_p = float(np.max(np.abs(Jp - Jp_an))) / scale
            err_d = float(np.max(np.abs(Jpd + Jp))) / max(float(np.max(np.abs(Jp))), 1e-30)
            check = {"coupling": "numerical", "rel_err_vs_analytic": err_p,
                     "rel_err_Jpd_plus_Jp": err_d, "jvd_max": jvd_max}
            if jvd_max > 1e-6 * scale:
                raise RuntimeError(
                    "Delayed-velocity Jacobian is non-negligible; extend the "
                    "semi-discretization history to velocities before using it.")
        else:
            raise ValueError("coupling must be 'numerical' or 'analytic'")

        # low-rank factorisation of the delayed coupling: Jpd_i ~= L_i V^T
        stack = Jpd.reshape(k * n, n)
        _, sv, Vt = np.linalg.svd(stack, full_matrices=False)
        r = int(np.sum(sv > 1e-9 * (sv[0] if sv.size else 1.0)))
        r = max(r, 1) if float(np.max(np.abs(Jpd))) > 0 else 0
        V = Vt[:r].T                                   # (n, r)
        L = np.einsum("inm,mr->inr", Jpd, V)           # (k, n, r)

        jac = {"k": k, "Jp": Jp, "Jv": Jv, "V": V, "L": L, "rank": r,
               "omega_ref": omega_ref, "pd": bool(getattr(p, "use_process_damping", False)),
               "g_samples": g_samples, "check": check, "xc": float(xc), "yc": float(yc)}
        self._jac_cache[key] = jac
        return jac


# ---------------------------------------------------------------------------
# Semi-discretization: Floquet spectral radius with reduced (low-rank) history
# ---------------------------------------------------------------------------
def sdm_spectral_radius(model: FaceMillingStabilityModel, jac: dict, rpm: float,
                        ap_mm: float, mode_idx: np.ndarray) -> float:
    """Largest |Floquet multiplier| of the linearised delayed modal system.

    First-order semi-discretization over one tooth period (= regen delay =
    coefficient period; the first-order delayed weights are both 1/2).  The
    history carries only the r low-rank coordinates s_j = V^T eta(t - j*dt), so
    the map dimension is 2n + r*k instead of 2n(k+1) — same theory, much faster.
    """
    idx = np.asarray(mode_idx, dtype=int)
    n = idx.size
    k = int(jac["k"])
    omega = float(rpm_to_omega(rpm))
    tau = 2.0 * np.pi / (model.N * max(omega, 1e-12))
    dt = tau / k

    wn = model.omega_vec[idx]
    ze = model.zeta_vec[idx]
    Om2 = np.diag(wn**2)
    Cd = np.diag(2.0 * ze * wn)
    Jp = jac["Jp"][np.ix_(range(k), idx, idx)]
    Jv = jac["Jv"][np.ix_(range(k), idx, idx)]
    vscale = (jac["omega_ref"] / omega) if jac["pd"] else 1.0
    r = int(jac["rank"])
    V = jac["V"][idx, :]                       # (n, r)
    L = jac["L"][np.ix_(range(k), idx, range(r))] if r else None  # (k, n, r)

    two_n = 2 * n
    d = two_n + r * k
    M = np.eye(d, dtype=np.complex128)
    In = np.eye(n)
    I2n = np.eye(two_n)

    for i in range(k):
        A = np.zeros((two_n, two_n))
        A[:n, n:] = In
        A[n:, :n] = -Om2 + ap_mm * Jp[i]
        A[n:, n:] = -Cd + ap_mm * vscale * Jv[i]
        P = matrix_exp(A * dt)
        T = np.zeros((d, d), dtype=np.complex128)
        T[:two_n, :two_n] = P
        if r > 0:
            Bc = np.zeros((two_n, r))
            Bc[n:, :] = ap_mm * L[i]
            try:
                Rd = np.linalg.solve(A, (P - I2n)) @ Bc
            except np.linalg.LinAlgError:
                Rd = dt * Bc
            # delayed term: x(t-tau) endpoints are history slots k-1 and k
            T[:two_n, two_n + r * (k - 2): two_n + r * (k - 1)] += 0.5 * Rd
            T[:two_n, two_n + r * (k - 1): two_n + r * k] += 0.5 * Rd
            # new s^(1) = V^T * (current positions);  s^(j) <- s^(j-1)
            T[two_n: two_n + r, :n] = V.T
            if k > 1:
                T[two_n + r:, two_n: two_n + r * (k - 1)] = np.eye(r * (k - 1))
        M = T @ M

    return float(np.max(np.abs(np.linalg.eigvals(M))))


def adaptive_k(model: FaceMillingStabilityModel, rpm: float, mode_idx: np.ndarray,
               k_base: int, samples_per_period: float = 6.0, k_max: int = 320) -> int:
    """Interval count that resolves the fastest retained mode at this speed."""
    omega = float(rpm_to_omega(rpm))
    tau = 2.0 * np.pi / (model.N * max(omega, 1e-12))
    f_max = float(np.max(model.omega_vec[np.asarray(mode_idx)])) / (2.0 * np.pi)
    k_need = int(np.ceil(samples_per_period * f_max * tau))
    k = int(np.clip(max(k_base, k_need), 8, k_max))
    return int(16 * np.ceil(k / 16))  # quantise so the Jacobian cache stays small


def sdm_critical_ap(model: FaceMillingStabilityModel, rpm: float, xc: float, yc: float,
                    mode_idx: np.ndarray, *, k_base: int = 48, coupling: str = "numerical",
                    ap_cap_mm: float = 30.0, tol_mm: float = 1e-3, max_iter: int = 40) -> float:
    """Critical depth ap_lim(rpm): spectral radius crosses 1 (bracket + bisection)."""
    k = adaptive_k(model, rpm, mode_idx, k_base)
    jac = model.periodic_jacobians(xc, yc, k, coupling=coupling)

    def rho(ap: float) -> float:
        return sdm_spectral_radius(model, jac, rpm, ap, mode_idx)

    ap_lo, ap_hi = 0.0, None
    ap = 0.02
    while ap <= ap_cap_mm:
        if rho(ap) > 1.0:
            ap_hi = ap
            break
        ap_lo = ap
        ap *= 1.7
    if ap_hi is None:
        return float(ap_cap_mm)
    for _ in range(max_iter):
        if ap_hi - ap_lo <= tol_mm:
            break
        mid = 0.5 * (ap_lo + ap_hi)
        if rho(mid) > 1.0:
            ap_hi = mid
        else:
            ap_lo = mid
    return 0.5 * (ap_lo + ap_hi)


# ---------------------------------------------------------------------------
# ZOA (Altintas-Budak) analytic lobe from the oriented FRF — cross-check
# ---------------------------------------------------------------------------
def zoa_lobe(model: FaceMillingStabilityModel, xc: float, yc: float, *,
             n_lobes: int = 12, freq_pts: int = 4000,
             ap_cap_mm: float = 50.0) -> dict[str, np.ndarray]:
    """Analytic scallops.  Exact for the averaged single-frequency model:
    ap = 1/(2*kappa*Re[Phi_zz(i*wc)]);  wc*tau = 2*theta + 2*pi*j."""
    phi = model.phi_at(xc, yc)
    kappa = model.kappa_per_ap()
    wn = model.omega_vec

    lobe_rpm: list[float] = []
    lobe_ap: list[float] = []
    for w0 in wn:
        for wc in np.linspace(0.6 * w0, 1.8 * w0, freq_pts):
            if wc <= 0:
                continue
            Phi = model.oriented_frf(wc, phi)
            G, H = Phi.real, Phi.imag
            if abs(G) < 1e-30:
                continue
            ap = 1.0 / (2.0 * kappa * G)
            if not np.isfinite(ap) or ap <= 0.0 or ap > ap_cap_mm:
                continue
            theta = np.arctan2(1.0, -H / G)  # psi/2 in (0, pi)
            for j in range(n_lobes):
                tau = (2.0 * theta + 2.0 * np.pi * j) / wc
                if tau > 0:
                    lobe_rpm.append(60.0 / (model.N * tau))
                    lobe_ap.append(ap)
    return {"rpm": np.asarray(lobe_rpm), "ap": np.asarray(lobe_ap)}


def zoa_envelope(lobe: dict[str, np.ndarray], rpm_grid: np.ndarray) -> np.ndarray:
    rpm, ap = lobe["rpm"], lobe["ap"]
    env = np.full(rpm_grid.shape, np.nan)
    if rpm.size == 0 or rpm_grid.size < 2:
        return env
    edges = np.concatenate([[rpm_grid[0] - 0.5 * (rpm_grid[1] - rpm_grid[0])],
                            0.5 * (rpm_grid[1:] + rpm_grid[:-1]),
                            [rpm_grid[-1] + 0.5 * (rpm_grid[-1] - rpm_grid[-2])]])
    idx = np.digitize(rpm, edges) - 1
    for gi in range(rpm_grid.size):
        sel = ap[idx == gi]
        if sel.size:
            env[gi] = float(np.min(sel))
    return env


# ---------------------------------------------------------------------------
# Heat map: rho(rpm, ap) grid + rho=1 boundary
# ---------------------------------------------------------------------------
def compute_heatmap(model: FaceMillingStabilityModel, args) -> dict[str, Any]:
    xc, yc = args.xc, args.yc
    phi = model.phi_at(xc, yc)
    mode_idx = model.mode_indices(phi, args.n_modes, args.mode_threshold)
    rpm_grid = np.linspace(args.rpm_min, args.rpm_max, args.rpm_points)
    ap_grid = np.linspace(args.ap_min_scan, args.ap_cap, args.ap_points)

    f_hz = model.omega_vec / (2 * np.pi)
    comp = phi**2 / (model.M_modal * model.omega_vec**2)
    print(f"[heatmap] cutter (xc={xc:.3f}, yc={yc:.3f}) m | N={model.N}, "
          f"ae/D={model.ae_mm / model.D_mm:.2f}, mean engaged={model.mean_engaged():.3f}")
    print(f"  modes kept {mode_idx.tolist()} of {model.K} "
          f"(f={np.round(f_hz[mode_idx], 1).tolist()} Hz, "
          f"compliance ratios={np.round(comp[mode_idx] / comp.max(), 4).tolist()})")
    f_dom = float(f_hz[int(np.argmax(comp))])
    last_lobe = 60.0 * f_dom / model.N
    if args.rpm_min > last_lobe:
        print(f"  note: the scallops of the dominant {f_dom:.1f} Hz mode lie below "
              f"{last_lobe:.0f} rpm (rpm=60*f/(N*j)); the requested range sits beyond "
              f"the last lobe, so the boundary there is smooth.")

    # self-check at ap=0: rho must equal the free-decay multiplier of the slowest mode
    k0 = adaptive_k(model, float(rpm_grid[0]), mode_idx, args.k_intervals)
    jac0 = model.periodic_jacobians(xc, yc, k0, coupling=args.coupling)
    if "rel_err_vs_analytic" in jac0["check"]:
        print(f"  linearisation check: |J_FD - J_analytic| = "
              f"{jac0['check']['rel_err_vs_analytic']:.2e} (rel), "
              f"|Jpd + Jp| = {jac0['check']['rel_err_Jpd_plus_Jp']:.2e} (rel), "
              f"delayed-coupling rank r={jac0['rank']}")
    tau0 = 2 * np.pi / (model.N * float(rpm_to_omega(rpm_grid[0])))
    rho0 = sdm_spectral_radius(model, jac0, float(rpm_grid[0]), 0.0, mode_idx)
    rho0_theory = float(np.exp(-np.min(model.zeta_vec[mode_idx] * model.omega_vec[mode_idx]) * tau0))
    print(f"  rho(ap=0) check: {rho0:.6f} vs exp(-zeta*w*tau) = {rho0_theory:.6f}")

    rho = np.empty((rpm_grid.size, ap_grid.size))
    t0 = time.time()
    for ri, rpm in enumerate(rpm_grid):
        k = adaptive_k(model, float(rpm), mode_idx, args.k_intervals)
        jac = model.periodic_jacobians(xc, yc, k, coupling=args.coupling)
        for ai, ap in enumerate(ap_grid):
            rho[ri, ai] = sdm_spectral_radius(model, jac, float(rpm), float(ap), mode_idx)
        if (ri + 1) % max(1, rpm_grid.size // 10) == 0 or ri == rpm_grid.size - 1:
            print(f"  [{ri + 1:>3}/{rpm_grid.size}] rpm={rpm:>7.0f} (k={k})  "
                  f"rho range [{rho[ri].min():.3f}, {rho[ri].max():.3f}]  "
                  f"({time.time() - t0:.0f}s)")

    # refined boundary: first upward rho=1 crossing per rpm, bisected between cells
    ap_bound = np.full(rpm_grid.size, np.nan)
    for ri, rpm in enumerate(rpm_grid):
        row = rho[ri]
        unstable = np.flatnonzero(row >= 1.0)
        if unstable.size == 0:
            ap_bound[ri] = args.ap_cap  # stable everywhere in scan
            continue
        i1 = int(unstable[0])
        if i1 == 0:
            ap_bound[ri] = ap_grid[0]
            continue
        k = adaptive_k(model, float(rpm), mode_idx, args.k_intervals)
        jac = model.periodic_jacobians(xc, yc, k, coupling=args.coupling)
        lo, hi = float(ap_grid[i1 - 1]), float(ap_grid[i1])
        for _ in range(24):
            if hi - lo <= 1e-3:
                break
            mid = 0.5 * (lo + hi)
            if sdm_spectral_radius(model, jac, float(rpm), mid, mode_idx) > 1.0:
                hi = mid
            else:
                lo = mid
        ap_bound[ri] = 0.5 * (lo + hi)

    zl = zoa_lobe(model, xc, yc, n_lobes=args.n_lobes, ap_cap_mm=args.ap_cap)
    return {"rpm": rpm_grid, "ap": ap_grid, "rho": rho, "ap_bound": ap_bound,
            "zoa_env": zoa_envelope(zl, rpm_grid), "zoa_points": zl,
            "mode_idx": mode_idx, "xc": xc, "yc": yc}


def plot_heatmap(res: dict[str, Any], model: FaceMillingStabilityModel, out: Path,
                 *, show_zoa: bool = True, fname: str = "stability_lobe_heatmap.png") -> None:
    rpm, ap, rho = res["rpm"], res["ap"], res["rho"]
    vmin = float(np.nanmin(rho)); vmax = float(np.nanmax(rho))
    norm = TwoSlopeNorm(vcenter=1.0, vmin=min(vmin, 1.0 - 1e-3), vmax=max(vmax, 1.0 + 1e-3))

    fig, ax = plt.subplots(figsize=(11, 6.5))
    pcm = ax.pcolormesh(rpm, ap, rho.T, cmap="RdBu_r", norm=norm, shading="auto",
                        rasterized=True)
    cb = fig.colorbar(pcm, ax=ax, pad=0.015)
    cb.set_label("Floquet spectral radius  ρ(Φ)   (stable ρ<1, chatter ρ>1)")
    try:
        ax.contour(rpm, ap, rho.T, levels=[0.9, 1.1], colors="k",
                   linewidths=0.8, linestyles=":", alpha=0.55)
        cs = ax.contour(rpm, ap, rho.T, levels=[1.0], colors="k", linewidths=2.4)
        ax.clabel(cs, fmt={1.0: "ρ = 1  (stability lobe)"}, fontsize=9)
    except Exception:
        pass
    if show_zoa and np.any(np.isfinite(res["zoa_env"])):
        ax.plot(rpm, res["zoa_env"], "--", color="#555555", lw=1.6,
                label="ZOA envelope (analytic cross-check)")
        ax.legend(loc="upper left", framealpha=0.85)
    ax.set_xlabel("Spindle speed  [rpm]")
    ax.set_ylabel("Axial depth of cut  ap  [mm]")
    ax.set_ylim(float(ap[0]), float(ap[-1]))
    ax.set_title(f"Face-milling stability heat map (semi-discretization, Floquet) — "
                 f"cutter xc={res['xc']:.2f} m, yc={res['yc']:.2f} m\n"
                 f"modes kept: {res['mode_idx'].tolist()}   "
                 f"(numerically linearised plant force)")
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / fname, dpi=160)
    plt.close(fig)
    print(f"saved {out / fname}")


# ---------------------------------------------------------------------------
# Bisection-only boundary mode (fast CSV) and 3D position sweeps (x or y0)
# ---------------------------------------------------------------------------
def compute_boundary(model: FaceMillingStabilityModel, args) -> dict[str, Any]:
    xc, yc = args.xc, args.yc
    phi = model.phi_at(xc, yc)
    mode_idx = model.mode_indices(phi, args.n_modes, args.mode_threshold)
    rpm_grid = np.linspace(args.rpm_min, args.rpm_max, args.rpm_points)
    print(f"[boundary] cutter ({xc:.3f},{yc:.3f}) m, modes kept {mode_idx.tolist()}")
    ap_sdm = np.empty(rpm_grid.size)
    for i, rpm in enumerate(rpm_grid):
        ap_sdm[i] = sdm_critical_ap(model, float(rpm), xc, yc, mode_idx,
                                    k_base=args.k_intervals, coupling=args.coupling,
                                    ap_cap_mm=args.ap_cap)
        print(f"  [{i + 1:>3}/{rpm_grid.size}] rpm={rpm:>7.0f}  ap_lim={ap_sdm[i]:.4g} mm")
    zl = zoa_lobe(model, xc, yc, n_lobes=args.n_lobes, ap_cap_mm=args.ap_cap)
    return {"rpm": rpm_grid, "ap_sdm": ap_sdm, "zoa_points": zl,
            "ap_zoa_env": zoa_envelope(zl, rpm_grid), "xc": xc, "yc": yc,
            "mode_idx": mode_idx}


def plot_boundary(res: dict[str, Any], out: Path) -> None:
    rpm = res["rpm"]
    fig, ax = plt.subplots(figsize=(10, 6))
    zl = res["zoa_points"]
    if zl["rpm"].size:
        m = (zl["rpm"] >= rpm[0]) & (zl["rpm"] <= rpm[-1])
        ax.scatter(zl["rpm"][m], zl["ap"][m], s=6, c="#bbbbbb", alpha=0.5,
                   label="ZOA scallops (analytic)")
    ax.plot(rpm, res["ap_zoa_env"], "--", color="#d62728", lw=1.8, label="ZOA lower envelope")
    ax.plot(rpm, res["ap_sdm"], "o-", color="#1f77b4", lw=2.2, ms=4,
            label="Semi-discretization (Floquet)")
    ax.fill_between(rpm, 0.0, res["ap_sdm"], alpha=0.10, color="#1f77b4")
    ax.set_xlabel("Spindle speed  [rpm]")
    ax.set_ylabel("Critical axial depth of cut  ap_lim  [mm]")
    ax.set_title(f"Fundamental face-milling stability lobe  "
                 f"(cutter xc={res['xc']:.2f} m, yc={res['yc']:.2f} m)")
    ax.set_ylim(0.0, float(np.nanmax([np.nanmax(res["ap_sdm"]), 1e-6]) * 1.3))
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "stability_lobe_fundamental_2d.png", dpi=160)
    plt.close(fig)


def compute_3d_surface(model: FaceMillingStabilityModel, args, *, axis: str = "x") -> dict[str, Any]:
    """3D stability surface: ap_lim = f(rpm, position), sweeping the cutter x
    (feed direction) or the milling line y0 (pass line).

    Both sweeps use identical, verified theory: the position enters the
    linearised regenerative loop ONLY through the cutter-point mode shape
    phi_z(xc, yc) — exactly as in the plant force projection — so sweeping y0
    at fixed xc is the same computation as sweeping xc at fixed y.
    """
    rpm_grid = np.linspace(args.rpm_min, args.rpm_max, args.rpm_points_3d)
    if axis == "y":
        pos_grid = np.linspace(args.yc_min, args.yc_max, args.yc_points)
        fixed = float(args.xc)
        lbl = "y0"
    else:
        pos_grid = np.linspace(args.xc_min, args.xc_max, args.xc_points)
        fixed = float(args.yc)
        lbl = "xc"
    ap = np.full((pos_grid.size, rpm_grid.size), np.nan)
    print(f"[3D surface, axis={axis}] {pos_grid.size} positions x {rpm_grid.size} speeds")
    for pi, pos in enumerate(pos_grid):
        xc, yc = (fixed, float(pos)) if axis == "y" else (float(pos), fixed)
        phi = model.phi_at(xc, yc)
        mode_idx = model.mode_indices(phi, args.n_modes, args.mode_threshold)
        for ri, rpm in enumerate(rpm_grid):
            ap[pi, ri] = sdm_critical_ap(model, float(rpm), xc, yc, mode_idx,
                                         k_base=args.k_intervals, coupling=args.coupling,
                                         ap_cap_mm=args.ap_cap)
        print(f"  {lbl}={pos:.3f} m  ap_lim range [{np.nanmin(ap[pi]):.3g}, "
              f"{np.nanmax(ap[pi]):.3g}] mm  (modes {mode_idx.tolist()})")
    return {"rpm": rpm_grid, "pos": pos_grid, "ap": ap, "axis": axis, "fixed": fixed}


def plot_3d_surface(res: dict[str, Any], out: Path) -> None:
    rpm, pos, ap = res["rpm"], res["pos"], res["ap"]
    axis = res.get("axis", "x")
    R, X = np.meshgrid(rpm, pos)
    fig = plt.figure(figsize=(11, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(R, X, ap, cmap="viridis", alpha=0.9, edgecolor="k", linewidth=0.2)
    ax.set_xlabel("Spindle speed  [rpm]")
    if axis == "y":
        ax.set_ylabel(f"Milling line y0  [m]   (fixed xc={res['fixed']:.2f} m)")
        title = "3D face-milling stability surface over the milling line y0"
        fname = "stability_lobe_fundamental_3d_y0.png"
    else:
        ax.set_ylabel(f"Cutter x-position  [m]  (0=clamped, L1=free; fixed y={res['fixed']:.2f} m)")
        title = "3D face-milling stability surface over the cutter x-position"
        fname = "stability_lobe_fundamental_3d.png"
    ax.set_zlabel("ap_lim  [mm]")
    ax.set_title(title + " (position-dependent chatter boundary)")
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="ap_lim [mm]")
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / fname, dpi=160)
    plt.close(fig)
    print(f"saved {out / fname}")


# ---------------------------------------------------------------------------
# Verification against the real nonlinear plant
# ---------------------------------------------------------------------------
def verify_against_plant(model: FaceMillingStabilityModel, args) -> list[dict[str, Any]]:
    """At probe (rpm, ap) points, integrate the true nonlinear plant from a small
    seed and compare observed regenerative growth/decay with the SDM prediction."""
    import gymnasium as gym
    from custom_rl import register_envs
    from custom_rl.eval.pipeline import plate_env_kwargs

    register_envs()
    xc, yc = args.xc, args.yc
    phi = model.phi_at(xc, yc)
    mode_idx = model.mode_indices(phi, args.n_modes, args.mode_threshold)

    kw = plate_env_kwargs(reward_id="dense", dt=1e-4, n_substeps=10,
                          max_episode_steps=8000, randomize_y0=False)
    kw["x0_cutter"] = float(model.plant.L1 - xc)
    kw["initial_eta_std"] = 1e-6
    kw["control_ap"] = False
    # Raise the displacement limit so the amplitude criterion does not truncate
    # the run before the regenerative envelope reveals growth/decay (a large
    # forced response tripping w_limit is NOT chatter -- see the theory doc).
    kw["w_limit"] = float(args.verify_w_limit)
    env = gym.make("CustomODEPlate-v0", **kw)

    rows = []
    print("\n=== Verification: linearised-SDM prediction vs nonlinear-plant growth ===")
    print(f"(modes kept {mode_idx.tolist()};"
          " a matching pair confirms the boundary is the true chatter onset)")
    print(f"{'rpm':>7} {'ap[mm]':>7} {'rho_SDM':>8} {'pred':>9} | {'growth/step':>11} "
          f"{'ratio':>6} {'max|w|um':>9} {'term':>5} {'obs':>9}  match")
    for rpm in [float(r) for r in args.verify_rpm]:
        omega = float(rpm_to_omega(rpm))
        k = adaptive_k(model, rpm, mode_idx, args.k_intervals)
        jac = model.periodic_jacobians(xc, yc, k, coupling=args.coupling)
        for ap in [float(a) for a in args.verify_ap]:
            rho = sdm_spectral_radius(model, jac, rpm, ap, mode_idx)
            pred = "UNSTABLE" if rho > 1.0 else "stable"
            obs, info = env.reset(seed=0, options={"y0": yc, "ap": ap})
            u = _phys_to_norm(env, omega, ap)
            w_env = []
            terminated_unstable = False
            for _ in range(int(args.verify_steps)):
                obs, r, term, trunc, info = env.step(u)
                if "w_sensor" in info:
                    w_env.append(float(np.max(np.abs(info["w_sensor"]))))
                if term or trunc:
                    if str(info.get("termination_reason", "")) in {
                            "excessive_sensor_displacement", "invalid_state"}:
                        terminated_unstable = True
                    break
            w_env = np.asarray(w_env)
            g_obs = _growth_per_step(w_env)
            ratio = _growth_ratio(w_env)
            max_w_um = float(np.max(w_env) * 1e6) if w_env.size else 0.0
            observed_unstable = (g_obs > 1.0 + 3e-4) or (ratio > 1.6)
            obs_state = "growing" if observed_unstable else (
                "amp>lim" if terminated_unstable else "decaying")
            match = "OK" if ((rho > 1.0) == observed_unstable) else "??"
            print(f"{rpm:>7.0f} {ap:>7.3f} {rho:>8.3f} {pred:>9} | {g_obs:>11.5f} "
                  f"{ratio:>6.2f} {max_w_um:>9.2f} {str(terminated_unstable):>5} "
                  f"{obs_state:>9}  {match}")
            rows.append({"rpm": rpm, "ap_mm": ap, "rho_sdm": rho, "pred": pred,
                         "growth_per_step": g_obs, "growth_ratio": ratio,
                         "max_w_um": max_w_um, "terminated_unstable": terminated_unstable,
                         "observed": obs_state, "match": match})
    env.close()
    n_ok = sum(1 for r in rows if r["match"] == "OK")
    print(f"\nagreement: {n_ok}/{len(rows)} points match the linear stability prediction")
    return rows


def _phys_to_norm(env, omega: float, ap: float) -> np.ndarray:
    p = env.unwrapped.plant
    lo, hi = p.physical_action_bounds()
    u_phys = np.array([omega, ap], dtype=np.float64)
    frac = (u_phys[: p.action_dim] - lo[: p.action_dim]) / np.maximum(
        hi[: p.action_dim] - lo[: p.action_dim], 1e-12)
    return np.clip(2.0 * frac - 1.0, -1.0, 1.0)


def _growth_per_step(w: np.ndarray) -> float:
    """Per-step envelope growth in the exponential regime (block maxima, late fit)."""
    w = np.asarray(w, dtype=np.float64)
    w = w[np.isfinite(w) & (w > 0)]
    if w.size < 40:
        return 1.0
    n_blocks = 24
    bs = max(1, w.size // n_blocks)
    centers, peaks = [], []
    for b in range(n_blocks):
        seg = w[b * bs:(b + 1) * bs]
        if seg.size:
            centers.append(b * bs + 0.5 * seg.size)
            peaks.append(float(np.max(seg)))
    centers, peaks = np.asarray(centers), np.asarray(peaks)
    if peaks.size < 6:
        return 1.0
    j0 = peaks.size // 3
    a = np.polyfit(centers[j0:], np.log(peaks[j0:] + 1e-30), 1)[0]
    return float(np.exp(a))


def _growth_ratio(w: np.ndarray) -> float:
    """Late-peak / post-transient-baseline ratio: catches fast chatter that
    saturates before a slope is measurable; forced responses give ratio ~ 1."""
    w = np.asarray(w, dtype=np.float64)
    w = w[np.isfinite(w) & (w > 0)]
    if w.size < 20:
        return 1.0
    i_a, i_b = int(0.15 * w.size), int(0.45 * w.size)
    baseline = float(np.median(w[i_a:i_b])) if i_b > i_a else float(np.median(w))
    late_peak = float(np.max(w[int(0.55 * w.size):]))
    return late_peak / max(baseline, 1e-30)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def save_heatmap_csv(res: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rpm", "ap_mm", "rho"])
        for ri in range(res["rpm"].size):
            for ai in range(res["ap"].size):
                w.writerow([f"{res['rpm'][ri]:.6g}", f"{res['ap'][ai]:.6g}",
                            f"{res['rho'][ri, ai]:.6g}"])


def save_boundary_csv(rpm: np.ndarray, ap_sdm: np.ndarray, ap_zoa: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rpm", "ap_lim_sdm_mm", "ap_lim_zoa_env_mm"])
        for i in range(rpm.size):
            w.writerow([f"{rpm[i]:.6g}", f"{ap_sdm[i]:.6g}",
                        f"{ap_zoa[i]:.6g}" if np.isfinite(ap_zoa[i]) else "nan"])


def save_csv_3d(res: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    col = "y0_m" if res.get("axis") == "y" else "xc_m"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([col, "rpm", "ap_lim_sdm_mm"])
        for pi in range(res["pos"].size):
            for ri in range(res["rpm"].size):
                w.writerow([f"{res['pos'][pi]:.6g}", f"{res['rpm'][ri]:.6g}",
                            f"{res['ap'][pi, ri]:.6g}"])


def build_model(args) -> FaceMillingStabilityModel:
    """Instantiate the RL plant (defaults = registered-env values) and wrap it."""
    kw = dict(ae_default=args.ae, feed_per_tooth_mm=args.feed,
              milling_mode=args.milling_mode, randomize_y0=False,
              modal_damping_ratio=args.zeta)
    if args.E is not None:
        kw["E"] = float(args.E)
    return FaceMillingStabilityModel(plant=PlatePlant(**kw))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("plots/lobe_fundamental"))
    ap.add_argument("--mode", choices=["heatmap", "boundary"], default="heatmap",
                    help="heatmap: rho(rpm,ap) field + rho=1 line (default); "
                         "boundary: bisection curve only")
    ap.add_argument("--rpm-min", type=float, default=400.0)
    ap.add_argument("--rpm-max", type=float, default=4000.0)
    ap.add_argument("--rpm-points", type=int, default=90)
    ap.add_argument("--rpm-points-3d", type=int, default=40)
    ap.add_argument("--ap-cap", type=float, default=30.0, help="upper ap of the scan [mm]")
    ap.add_argument("--ap-min-scan", type=float, default=0.02)
    ap.add_argument("--ap-points", type=int, default=70)
    ap.add_argument("--xc", type=float, default=0.85,
                    help="cutter x-position [m] (0=clamped, L1=free)")
    ap.add_argument("--yc", type=float, default=0.20, help="milling-line y [m]")
    ap.add_argument("--xc-min", type=float, default=0.30)
    ap.add_argument("--xc-max", type=float, default=0.98)
    ap.add_argument("--xc-points", type=int, default=12)
    ap.add_argument("--yc-min", type=float, default=0.10, help="y0 sweep lower bound [m]")
    ap.add_argument("--yc-max", type=float, default=0.90, help="y0 sweep upper bound [m]")
    ap.add_argument("--yc-points", type=int, default=9)
    ap.add_argument("--n-modes", type=int, default=-1,
                    help="-1 = ALL plant modes (default; exact linearisation -- "
                         "truncation can misjudge mode-COUPLING flutter: e.g. modes "
                         "{0,1} alone are unstable at (2000rpm,1.5mm) while the full "
                         "6-mode system is stable); 0 = auto by compliance threshold; "
                         ">0 = top-n by compliance (speed only)")
    ap.add_argument("--mode-threshold", type=float, default=1e-2,
                    help="auto mode-truncation: keep compliance >= threshold*max")
    ap.add_argument("--k-intervals", type=int, default=48,
                    help="base SDM subintervals per tooth period (auto-raised at "
                         "low rpm to resolve the fastest retained mode)")
    ap.add_argument("--coupling", choices=["numerical", "analytic"], default="numerical",
                    help="numerical: finite-difference Jacobians of the plant's own "
                         "force code (black-box, default); analytic: closed form")
    ap.add_argument("--n-lobes", type=int, default=12, help="ZOA lobe count")
    ap.add_argument("--no-zoa", action="store_true", help="skip the ZOA overlay")
    # process / model overrides (default = registered-env values)
    ap.add_argument("--ae", type=float, default=28.0, help="radial immersion ae [mm]")
    ap.add_argument("--feed", type=float, default=0.20, help="feed per tooth [mm]")
    ap.add_argument("--zeta", type=float, default=0.02, help="modal damping ratio")
    ap.add_argument("--milling-mode", default="up", choices=["up", "down"])
    ap.add_argument("--E", type=float, default=None, help="override Young's modulus [Pa]")
    # modes of operation
    ap.add_argument("--surface-3d", action="store_true",
                    help="3D surface ap_lim = f(rpm, cutter-x) at fixed y")
    ap.add_argument("--surface-3d-y", action="store_true",
                    help="3D surface ap_lim = f(rpm, milling line y0) at fixed xc")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--verify-rpm", nargs="+", type=float, default=[1000, 2000])
    ap.add_argument("--verify-ap", nargs="+", type=float, default=[0.05, 0.5, 3.0])
    ap.add_argument("--verify-steps", type=int, default=4000)
    ap.add_argument("--verify-w-limit", type=float, default=0.05,
                    help="displacement limit for the verification env [m]")
    args = ap.parse_args()

    out = Path(args.out_dir)
    model = build_model(args)
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "semi_discretization_floquet(low-rank, numerical Jacobians) + ZOA",
        "coupling": args.coupling,
        "natural_freq_hz": list(np.round(model.omega_vec / (2 * np.pi), 4)),
        "zeta": list(np.round(model.zeta_vec, 5)),
        "modal_mass_kg": list(np.round(model.M_modal, 4)),
        "N_teeth": model.N, "Ka_N_mm2": model.Ka,
        "gamma_L_deg": float(np.degrees(model.gamma_L)),
        "ae_mm": model.ae_mm, "D_mm": model.D_mm,
        "mean_engaged_teeth": model.mean_engaged(),
        "k_intervals_base": args.k_intervals, "n_modes": args.n_modes,
        "mode_threshold": args.mode_threshold, "xc": args.xc, "yc": args.yc,
    }

    if args.verify:
        rows = verify_against_plant(model, args)
        out.mkdir(parents=True, exist_ok=True)
        (out / "verification.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nsaved {out / 'verification.json'}")
        return

    if args.surface_3d or args.surface_3d_y:
        axis = "y" if args.surface_3d_y else "x"
        res = compute_3d_surface(model, args, axis=axis)
        plot_3d_surface(res, out)
        tag = "_y0" if axis == "y" else ""
        save_csv_3d(res, out / f"stability_lobe_fundamental_3d{tag}.csv")
        meta["surface_axis"] = axis
        (out / f"stability_lobe_fundamental_3d{tag}_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
        print(f"\nsaved 3D surface (axis={axis}) -> {out}")
        return

    if args.mode == "heatmap":
        res = compute_heatmap(model, args)
        plot_heatmap(res, model, out, show_zoa=not args.no_zoa)
        save_heatmap_csv(res, out / "stability_lobe_heatmap.csv")
        save_boundary_csv(res["rpm"], res["ap_bound"], res["zoa_env"],
                          out / "stability_lobe_boundary.csv")
        (out / "stability_lobe_heatmap_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
        print(f"saved heat map + boundary CSVs -> {out}")
    else:
        res = compute_boundary(model, args)
        plot_boundary(res, out)
        save_boundary_csv(res["rpm"], res["ap_sdm"], res["ap_zoa_env"],
                          out / "stability_lobe_fundamental_2d.csv")
        (out / "stability_lobe_fundamental_2d_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
        print(f"\nsaved 2D boundary -> {out}")


if __name__ == "__main__":
    main()
