"""Phase-domain pseudospectral output-feedback NMPC + MHE for face-milling chatter.

This replaces the frozen-delay do-mpc controller (mpc_mhe_dompc.py), whose
prediction model had two structural defects: (i) the spindle speed omega did not
enter the state dynamics at all (d f / d omega = 0), and (ii) the delayed cutter
displacement w_z(t-tau) was supplied as an externally frozen time-varying
parameter instead of being an internal predicted state.  Neither defect can be
repaired by tuning weights.

DERIVATION (all steps exact unless marked APPROX)
-------------------------------------------------
1. Phase as independent variable.  With omega(t) >= omega_min > 0 the spindle
   phase theta(t) = int omega dt is strictly increasing, so it is an admissible
   independent variable.  For any signal r,  dr/dtheta = (1/omega) dr/dt, hence

       eta' = v / omega
       v'   = (1/omega) [ -2 Z Om v - Om^2 eta - Lam eta^3 + q_f ]

   EXACT change of variables.  omega now multiplies the whole vector field, so
   the predicted vibration genuinely depends on the spindle speed.

2. The regenerative delay becomes CONSTANT in phase.  The plant defines the
   delayed instant t_d by the one-tooth PHASE separation
       theta(t) - theta(t_d) = p,   p = 2*pi/N,
   (the plant stores spindle-phase history and inverts it -- it does NOT use the
   instantaneous tau = 2 pi/(N omega) when the speed changes).  In phase
   coordinates that delay is exactly the constant p regardless of omega.

3. Delay -> transport PDE (EXACT).  Define the history field
       q(xi, theta) = w_c(theta - p xi),  xi in [0, 1].
   Then q(0,theta) = w_c(theta), q(1,theta) = w_c(theta - p) = w_d, and
       dq/dtheta + (1/p) dq/dxi = 0.
   This is an exact restatement of the delay, not a modelling assumption; it
   also carries the delayed CUTTER POSITION automatically, because w_c contains
   phi(x_c(theta)).

4. APPROX 1 -- pseudospectral collocation of the history field.  On the
   Chebyshev-Lobatto grid xi_j = (1 - cos(j pi/M))/2 with differentiation matrix
   D, collocating the transport equation at the interior nodes gives
       q_h' = -(1/p) [ D_II q_h + D_I0 w_c ],   w_d = e_M^T q_h.
   Convergence is spectral; the error is verified in the frequency domain
   against exp(-i nu p) by --verify-delay.

5. APPROX 2 -- tooth-wise force law with SMOOTHED switching.  The plant sums,
   over the N teeth, a force that is exactly linear in the effective chip
   thickness but switched twice: by the engagement indicator and by the
   chip-loss clamp max(h,0) (with the edge force gated off when h<=0, since the
   plant sets EDGE_FORCE_WHEN_ZERO_CHIP=False).  For the plant's configuration
   (scalar transverse mode family, so phi_x=phi_y=0) the chip thickness of
   tooth i is EXACTLY

       h_i = ft sin(theta_i) + kr ( w_c - w_d ),    kr = 1000 sin(gamma_L),

   theta_i = theta + theta0 + 2 pi i / N, and the axial force is
       F_z = a_p sum_i g_i [ Ka max(h_i,0) + Kae 1{h_i>0} ],   q_f = phi F_z/Mm.
   The model keeps the per-tooth structure and replaces only the two switches:

       1{engaged}  ->  g_i(theta) = sigmoid window of sharpness --sharp
       max(h,0)    ->  (h + sqrt(h^2 + eps^2))/2         (--chip-eps, in mm)
       1{h>0}      ->  (1 + h/sqrt(h^2 + eps^2))/2

   Both replacements converge to the exact law as sharp->inf, eps->0, and the
   residual is measured (not assumed) by --verify-force.  Dropping the chip-loss
   clamp instead -- the usual "linear regenerative model" -- is NOT acceptable
   here: with ft=0.2 mm and kr=707, teeth near entry fly over the surface for
   vibration levels well inside the displacement limit, and the affine model
   then predicts a large force where the plant produces exactly zero.

   Because every h_i depends on the state only through w_c = phi^T eta, the
   Jacobians retain the exact rank-one regenerative structure
   dq_f/d eta = -(dq_f/d w_d) phi^T   (i.e. J_d = -J_p), for any eps.

6. Known TVPs.  In phase coordinates BOTH theta and x_c advance deterministically
   (x_c' = -f_t N /(2 pi 1000) is independent of omega), so the periodic
   coefficients (Gc, Gs, phi_c) at every horizon node are known exactly in
   advance.  They are therefore legitimate do-mpc time-varying parameters --
   unlike the delayed displacement, which is now an internal state.

Augmented state:  z = [eta(K); v(K); q_h(M); d_F]  ->  n_z = 2K + M + 1.

IMPLEMENTATION NOTES (corrections applied vs. the source derivation)
--------------------------------------------------------------------
C1. STIFFNESS.  The pseudospectral operator is stiff: max|eig(-D_II/p)| ~ 182
    rad^-1 at M=40, so an explicit RK4 in phase would need ~100 substeps per
    tooth pitch merely for stability.  do-mpc's implicit (L-stable) Radau
    collocation is used instead, which damps the purely numerical high
    pseudospectral eigenvalues instead of amplifying them.
C2. MODE REDUCTION.  The required M is set by the highest RETAINED mode at the
    LOWEST speed (nu = omega_n/omega).  Keeping all K=6 modes forces M=40; the
    modal compliance participation phi^2/(M_m omega_n^2) printed at startup
    shows modes 3..6 carry <2% of the cutter-point compliance, so K=3 with M=20
    is both smaller AND more accurate.  Discarded modes appear only in the
    measurement, where they are covered by the measurement-noise weight and the
    force-bias state d_F.
C3. PHASE-SYNCHRONOUS SAMPLING (this is a real constraint, not a tuning knob).
    The plant is sampled in TIME with period dt_env, the model advances in
    PHASE with step dtheta.  A control interval of dtheta rad lasts dtheta/omega
    seconds, so it is only realizable while

        dtheta / omega >= dt_env    <=>    omega <= dtheta / dt_env.

    This is a Nyquist-type statement: you cannot close a per-tooth loop on a
    spindle whose tooth-passing period is shorter than your control period.
    The bound is therefore imposed as an explicit MPC input bound
    (--omega-sampling-cap, on by default) rather than being silently violated.
    The remaining integer-rounding of the hold length is not absorbed by
    assumption: the schedule is re-synchronised every step to the plant's OWN
    unwrapped spindle phase (info['spindle_phase_rad']), and the realized
    dtheta is measured and reported.
C4. INPUT-RATE LIMITS.  Limits are stated in PHYSICAL units -- spindle angular
    acceleration [rad/s^2] and axial-depth feed rate [mm/s] -- and converted to
    a per-interval bound with the interval's own duration dt = dtheta/omega.
    The bound on the FIRST input of the horizon (the one actually applied) is
    written into the NLP through lb_opt_x/ub_opt_x before every solve, so the
    applied command is feasible for the OCP that produced it.  Later stages are
    handled by the r-term.  The external clip is then provably inactive and is
    asserted as such (reported as 'rate_clip_active_frac').
C5. TERMINAL COST.  Vibration-energy weighting (no DARE).  Solving a Riccati
    equation for a 2K+M+1 state LPV linearisation every step is not real-time
    realistic; the terminal set of the source derivation is nominal theory.
C6. FORCE VERIFICATION is done against the plant's own tooth-by-tooth routine
    compute_face_milling_force(), not against a re-statement of the same
    formula, so the reported NRMSE genuinely bounds APPROX 2 (including the
    chip-loss max(h,0) and the edge-force gate that the affine model drops).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

try:
    import casadi as ca
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import do_mpc
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"do-mpc and CasADi required: pip install do-mpc casadi ({exc})")

import gymnasium as gym

from custom_rl import DEFAULT_TRAJ_DIR, register_envs
from custom_rl.eval.monte_carlo import face_milling_process_from_info
from custom_rl.eval.pipeline import plate_env_kwargs, plant_plot_metadata
from custom_rl.plants.plate import omega_to_rpm

MODAL_DISP_TO_MM = 1000.0
PROCESS_KEYS = (
    "omega_rad_s", "omega_rpm", "ap_mm", "ae_mm", "cutter_x", "cutter_y",
    "feed_progress", "feed_distance_m", "spindle_phase_rad", "mean_chip_mm",
    "max_chip_mm", "max_abs_w_m", "Fx_N", "Fy_N", "Fz_N", "F_mag_N",
)
PASS_COMPLETED_REASONS = {"pass_completed_90percent", "pass_completed"}


# ===========================================================================
# Chebyshev-Lobatto differentiation matrix on [0, 1], nodes ascending
# ===========================================================================
def cheb_lobatto_D(M: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (xi, D) with xi_0=0 < ... < xi_M=1 and D = d/dxi on that grid."""
    j = np.arange(M + 1)
    x = np.cos(np.pi * j / M)                      # [1 ... -1]
    c = np.ones(M + 1); c[0] = c[-1] = 2.0
    c = c * (-1.0) ** j
    X = np.tile(x, (M + 1, 1)).T
    dX = X - X.T + np.eye(M + 1)
    D = np.outer(c, 1.0 / c) / dX
    D -= np.diag(D.sum(axis=1))
    # map x in [-1,1] -> xi = (1-x)/2 in [0,1]:  d/dxi = -2 d/dx ; sort ascending
    idx = np.argsort((1.0 - x) / 2.0)
    return ((1.0 - x) / 2.0)[idx], (-2.0 * D)[np.ix_(idx, idx)]


# ===========================================================================
# Plant constants + analytic force coefficients
# ===========================================================================
class PhasePlant:
    """Constants from the gym env, plus the phase-domain force coefficients."""

    def __init__(self, env: gym.Env, n_modes: int | None = None) -> None:
        base = env.unwrapped
        plant = base.plant
        self.plant = plant
        self.base = base
        self.control_ap = bool(getattr(plant, "control_ap", True))

        K_full = int(plant.K)
        self.K_full = K_full
        wn_full = np.asarray(plant.omega_vec, dtype=np.float64).reshape(-1)
        Mm_full = np.asarray(plant.M_modal, dtype=np.float64).reshape(-1)

        # ---- model reduction by CUTTER-POINT COMPLIANCE PARTICIPATION --------
        # The regenerative loop gain is proportional to the receptance of the
        # workpiece AT THE CUTTER, sum_i phi_i(x_c)^2 / (M_i omega_i^2).  Ranking
        # modes by their (pass-averaged) share of that quantity is the correct
        # reduction criterion here -- ranking by frequency alone is a heuristic
        # that can retain a high-frequency mode with negligible participation.
        xs = np.linspace(0.0, float(plant.L1), 41)
        yc0 = float(plant.y_cutter)
        Pmat = np.array([self._phi_full(x, yc0) for x in xs])          # (41, K_full)
        part = np.mean(Pmat**2, axis=0) / (Mm_full * wn_full**2)
        self.participation = part / max(part.sum(), 1e-300)
        self.K = max(1, min(int(n_modes) if n_modes else K_full, K_full))
        self.sel = np.sort(np.argsort(-self.participation)[: self.K])
        self.participation_kept = float(self.participation[self.sel].sum())

        self.wn = wn_full[self.sel]
        self.zeta = np.asarray(plant.zeta_vec, dtype=np.float64).reshape(-1)[self.sel]
        self.lam = np.asarray(plant.lambda_vec, dtype=np.float64).reshape(-1)[self.sel]
        self.Mm = np.asarray(plant.M_modal, dtype=np.float64).reshape(-1)[self.sel]
        self.Phi = np.asarray(plant.Phi, dtype=np.float64)[:, self.sel]
        self.nS = int(self.Phi.shape[0])
        self.Phi_pinv = np.linalg.pinv(self.Phi)

        self.N = int(plant.N)
        self.Ka = float(plant.Ka)
        self.Kae = float(plant.Kae)
        self.ft = float(plant.feed_per_tooth_mm)
        self.kr = MODAL_DISP_TO_MM * float(np.sin(plant.gamma_L))
        self.D_mm = float(plant.D_mm)
        self.ae = float(plant.ae_default)
        self.mode = str(plant.milling_mode).lower()
        self.theta0 = float(getattr(plant, "theta0", 0.0))

        self.omega_min = float(plant.omega_min)
        self.omega_max = float(plant.omega_max)
        self.ap_min = float(plant.ap_min)
        self.ap_max = float(plant.ap_max)
        self.w_limit = float(plant.w_limit)
        self.w_obs = float(plant.w_obs_scale)
        self.wd_obs = float(plant.wdot_obs_scale)
        self.L1 = float(plant.L1)
        self.L2 = float(plant.L2)
        self.env_dt = float(base.dt) * int(base.n_substeps)

        # radial-immersion engagement window [theta_s, theta_e]
        imm = float(np.clip(self.ae / max(self.D_mm, 1e-12), 0.0, 1.0))
        if self.mode in {"down", "climb", "climb_milling"}:
            self.th_s = float(np.arccos(np.clip(2.0 * imm - 1.0, -1.0, 1.0)))
            self.th_e = float(np.pi)
        else:
            self.th_s = 0.0
            self.th_e = float(np.arccos(np.clip(1.0 - 2.0 * imm, -1.0, 1.0)))
        c = self.N / (2.0 * np.pi)
        self.S0 = float(c * (self.th_e - self.th_s))
        self.S1 = float(c * (np.cos(self.th_s) - np.cos(self.th_e)))

        self.p = 2.0 * np.pi / self.N          # one tooth pitch [rad]
        # smoothing actually used inside the NLP (overwritten from the CLI)
        self._vsharp = 40.0
        self._veps = 2.0e-3

    # ---- per-tooth periodic coefficients ----------------------------------
    def tooth_terms(self, theta: float, sharp: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Return (g, s): engagement weight and sin(theta_i) for each of N teeth.

        sharp>0 replaces the hard indicator by a product of sigmoids of that
        sharpness so the NLP stays differentiable; sharp=0 reproduces the
        plant's exact open-interval indicator (used for verification).
        """
        a = (theta + self.theta0
             + 2.0 * np.pi * np.arange(self.N) / self.N) % (2.0 * np.pi)
        if sharp > 0.0:
            g = (1.0 / (1.0 + np.exp(-sharp * (a - self.th_s)))) * \
                (1.0 / (1.0 + np.exp(sharp * (a - self.th_e))))
        else:
            g = ((a > self.th_s) & (a < self.th_e)).astype(np.float64)
        return g, np.sin(a)

    @staticmethod
    def n_quad_nodes(N: int, nq: int) -> int:
        """Fixed TVP length used by :meth:`tooth_grid` (N teeth x 2 segments x nq)."""
        return int(N) * 2 * max(1, int(nq))

    def tooth_grid(self, theta: float, dtheta: float, nq: int) -> tuple[np.ndarray, np.ndarray]:
        """Interval-averaging weights w_j and sin(theta_j) over [theta, theta+dtheta).

        do-mpc freezes a TVP across a control interval, so the correct
        piecewise-constant representative of a rapidly varying periodic
        coefficient is its interval AVERAGE; point-sampling aliases tooth
        entry/exit.  The average is taken of the full nonlinear tooth law (the
        chip-loss clamp stays inside the integral, where it belongs):

            (1/dtheta) int_theta^{theta+dtheta} sum_i 1{engaged}(a_i(s))
                                                 f(h_i(s)) ds  ~  sum_j w_j f(h_j)

        The quadrature is SPLIT AT THE DISCONTINUITIES: for each tooth the
        intersection of its swept arc with the engagement window is computed in
        closed form, and the nq midpoint nodes are placed inside that
        intersection.  The integrand is then smooth on every panel, so the rule
        converges at the midpoint rate instead of the O(1/nq) rate that
        sampling across a jump would give.

        Returns (w, s), both of fixed length N*2*nq (zero-weight padding), so
        the TVP dimension does not depend on the operating point.
        """
        nq = max(1, int(nq))
        dth = float(dtheta)
        w = np.zeros(self.N * 2 * nq)
        s = np.zeros(self.N * 2 * nq)
        two_pi = 2.0 * np.pi
        for i in range(self.N):
            lo = (theta + self.theta0 + two_pi * i / self.N) % two_pi
            hi = lo + dth
            for k, shift in enumerate((0.0, two_pi)):
                s0 = max(lo, self.th_s + shift)
                s1 = min(hi, self.th_e + shift)
                if s1 <= s0:
                    continue
                nodes = s0 + (np.arange(nq) + 0.5) * (s1 - s0) / nq
                j = (i * 2 + k) * nq
                w[j:j + nq] = (s1 - s0) / (nq * dth)
                s[j:j + nq] = np.sin(nodes)
        return w, s

    def engagement(self, theta: float, sharp: float = 0.0) -> tuple[float, float]:
        """Return the engaged-tooth sums (Gc, Gs) = (sum g_i, sum g_i sin th_i).

        These are the INSTANTANEOUS versions of S0, S1 (they average to them over
        one tooth pitch).  Kept for reporting only -- the model uses the
        per-tooth vectors from :meth:`tooth_terms`.
        """
        g, s = self.tooth_terms(theta, sharp)
        return float(g.sum()), float((g * s).sum())

    def _phi_full(self, xc: float, yc: float) -> np.ndarray:
        """Mode-shape vector at (xc, yc) for ALL K_full modes (plant convention)."""
        v = np.zeros(int(self.plant.K), dtype=np.float64)
        i = 0
        for m in range(self.plant.m_max):
            for n in range(self.plant.n_max):
                v[i] = float(self.plant.W_mn[m][n](float(xc), float(yc)))
                i += 1
        return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    def phi_at(self, xc: float, yc: float) -> np.ndarray:
        return self._phi_full(xc, yc)[self.sel]

    def cutter_from_obs(self, obs: np.ndarray) -> tuple[float, float]:
        a = np.asarray(obs, dtype=np.float64).reshape(-1)
        return (float(np.clip(a[2 * self.nS], 0.0, 1.0) * self.L1),
                float(np.clip(a[2 * self.nS + 1], 0.0, 1.0) * self.L2))

    def meas_from_obs(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, dtype=np.float64).reshape(-1)[: 2 * self.nS].copy()

    def modal_from_obs(self, obs: np.ndarray) -> np.ndarray:
        y = self.meas_from_obs(obs)
        return np.concatenate([self.Phi_pinv @ (y[: self.nS] * self.w_obs),
                               self.Phi_pinv @ (y[self.nS:] * self.wd_obs)])

    # ---- modal acceleration of the prediction model (numpy twin of build_model)
    def qf_model(self, theta, xc, yc, eta, wd, ap, sharp=0.0, eps=0.0,
                 dtheta=0.0, nq=1) -> np.ndarray:
        """Modal acceleration from the tooth-wise force law (numpy twin of the
        CasADi model in :func:`build_model` -- the two must agree bit-for-bit).

        dtheta=0, nq=1, eps=0, sharp=0 reproduce the plant's instantaneous law
        exactly; the values used inside the NLP are (--chip-eps, --n-quad).
        """
        if dtheta > 0.0:
            g, s = self.tooth_grid(theta, dtheta, nq)
        else:
            g, s = self.tooth_terms(theta, sharp)
        phi = self.phi_at(xc, yc)
        wc = float(phi @ eta)
        h = self.ft * s + self.kr * (wc - wd)                 # [mm], per node
        if eps > 0.0:
            root = np.sqrt(h * h + eps * eps)
            h_eff = 0.5 * (h + root)
            gate = 0.5 * (1.0 + h / root)
        else:
            h_eff = np.maximum(h, 0.0)
            gate = (h > 0.0).astype(np.float64)
        Fz = ap * float(np.sum(g * (self.Ka * h_eff + self.Kae * gate)))
        return phi * (Fz / self.Mm)


# ===========================================================================
# do-mpc phase-domain model
# ===========================================================================
def build_model(pp: PhasePlant, M: int, chip_eps: float,
                nq: int) -> tuple["do_mpc.model.Model", np.ndarray]:
    xi, D = cheb_lobatto_D(M)
    nq = max(1, int(nq))
    ng = PhasePlant.n_quad_nodes(pp.N, nq)
    DII = D[1:, 1:]
    DI0 = D[1:, 0]

    model = do_mpc.model.Model("continuous")   # independent variable = PHASE
    K = pp.K

    eta = model.set_variable("_x", "eta", (K, 1))
    v = model.set_variable("_x", "v", (K, 1))
    qh = model.set_variable("_x", "qh", (M, 1))
    dF = model.set_variable("_x", "dF")

    omega = model.set_variable("_u", "omega")
    ap = model.set_variable("_u", "ap")

    # Known TVPs: deterministic in phase (theta and x_c both advance
    # independently of the optimisation decisions).
    phi = model.set_variable("_tvp", "phi", (K, 1))
    gvec = model.set_variable("_tvp", "gvec", (ng, 1))     # quadrature weights
    svec = model.set_variable("_tvp", "svec", (ng, 1))     # sin(theta_i)

    wc = ca.mtimes(phi.T, eta)                 # cutter displacement now
    wd = qh[M - 1]                             # = q(xi=1) : one tooth back

    # Tooth-wise chip thickness [mm] at the quadrature nodes.  gvec already
    # carries the interval-averaging weights, so no extra 1/nq here (APPROX 2/3).
    eps = float(chip_eps)
    h = pp.ft * svec + pp.kr * (wc - wd)       # (N*2*nq,1), broadcast over nodes
    root = ca.sqrt(h * h + eps * eps)
    h_eff = 0.5 * (h + root)                   # -> max(h,0)   as eps -> 0
    gate = 0.5 * (1.0 + h / root)              # -> 1{h>0}     as eps -> 0
    Fz = ap * ca.sum1(gvec * (pp.Ka * h_eff + pp.Kae * gate))
    qf = phi * (Fz / ca.DM(pp.Mm.reshape(-1, 1)))

    wn = ca.DM(pp.wn.reshape(-1, 1))
    zt = ca.DM(pp.zeta.reshape(-1, 1))
    lm = ca.DM(pp.lam.reshape(-1, 1))
    acc = (-2.0 * zt * wn * v - wn**2 * eta - lm * eta**3 + qf + dF * phi)

    model.set_rhs("eta", v / omega, process_noise=False)
    model.set_rhs("v", acc / omega, process_noise=True)
    # deterministic transport (history shift) -- no independent process noise
    model.set_rhs("qh", -(ca.DM(DII) @ qh + ca.DM(DI0.reshape(-1, 1)) * wc) / pp.p,
                  process_noise=False)
    model.set_rhs("dF", ca.DM.zeros(1, 1), process_noise=True)

    Phi = ca.DM(pp.Phi)
    model.set_meas("y_w", ca.mtimes(Phi, eta) / pp.w_obs, meas_noise=True)
    model.set_meas("y_wd", ca.mtimes(Phi, v) / pp.wd_obs, meas_noise=True)
    model.set_meas("u_om", omega, meas_noise=False)
    model.set_meas("u_ap", ap, meas_noise=False)
    model.setup()
    return model, xi


# ===========================================================================
# TVP scheduling (phase is deterministic -> exactly known ahead and behind)
# ===========================================================================
class PhaseSchedule:
    """theta and x_c evolve deterministically, so TVPs are exactly computable."""

    def __init__(self, pp: PhasePlant, dtheta: float, sharp: float, nq: int = 8) -> None:
        self.pp = pp
        self.dtheta = float(dtheta)
        self.sharp = float(sharp)
        self.nq = max(1, int(nq))
        self.theta = 0.0
        self.xc = pp.L1
        self.yc = pp.L2 * 0.5
        # dx_c/dtheta is INDEPENDENT of spindle speed (constant feed per tooth)
        self.dxc = -pp.ft * pp.N / (2.0 * np.pi * 1000.0)
        # realized (theta, xc, yc) history, newest last -- the MHE window looks
        # BACKWARDS, where the true phases are known and need not be extrapolated
        self.hist: list[tuple[float, float, float]] = []

    def set_now(self, theta: float, xc: float, yc: float) -> None:
        self.theta, self.xc, self.yc = float(theta), float(xc), float(yc)
        self.hist.append((self.theta, self.xc, self.yc))
        if len(self.hist) > 512:
            del self.hist[:256]

    def at(self, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """TVPs k control intervals ahead (k<0 = behind) of the current phase.

        Backwards (k<0) the realized phase/position are read from the recorded
        history; only forwards is the nominal dtheta extrapolation used, where it
        is unavoidable because omega over the horizon is still being decided.
        """
        if k < 0 and len(self.hist) >= -k + 1:
            th, xc, yc = self.hist[k - 1]
        else:
            th = self.theta + k * self.dtheta
            xc = float(np.clip(self.xc + k * self.dtheta * self.dxc, 0.0, self.pp.L1))
            yc = self.yc
        g, s = self.pp.tooth_grid(th, self.dtheta, self.nq)
        return self.pp.phi_at(xc, yc), g, s


def _fill_tvp(tmpl, k: int, phi: np.ndarray, g: np.ndarray, s: np.ndarray) -> None:
    tmpl["_tvp", k, "phi"] = phi.reshape(-1, 1)
    tmpl["_tvp", k, "gvec"] = g.reshape(-1, 1)
    tmpl["_tvp", k, "svec"] = s.reshape(-1, 1)


def mpc_tvp_fun(sched: PhaseSchedule, mpc):
    tmpl = mpc.get_tvp_template()
    Np = int(mpc.settings.n_horizon)

    def fun(t_now):
        for k in range(Np + 1):
            _fill_tvp(tmpl, k, *sched.at(k))
        return tmpl
    return fun


def mhe_tvp_fun(sched: PhaseSchedule, mhe):
    tmpl = mhe.get_tvp_template()
    Ne = int(mhe.settings.n_horizon)

    def fun(t_now):
        # do-mpc MHE: index -1 (== Ne) is the CURRENT instant, 0 is the oldest.
        for k in range(Ne + 1):
            _fill_tvp(tmpl, k, *sched.at(k - Ne))
        return tmpl
    return fun


# ===========================================================================
# NMPC
# ===========================================================================
def build_mpc(model, pp: PhasePlant, sched: PhaseSchedule, M: int, a: argparse.Namespace,
              ap_fixed: float | None, omega_cap: float):
    mpc = do_mpc.controller.MPC(model)
    mpc.settings.t_step = sched.dtheta          # PHASE step
    mpc.settings.n_horizon = int(a.mpc_horizon)
    mpc.settings.n_robust = 0
    mpc.settings.open_loop = False
    # Implicit Radau collocation: the pseudospectral transport operator is stiff
    # (max|eig| ~ 1.8e2 rad^-1 at M=40); explicit integration would be unstable.
    mpc.settings.state_discretization = "collocation"
    mpc.settings.collocation_type = "radau"
    mpc.settings.collocation_deg = int(a.coll_deg)
    mpc.settings.collocation_ni = int(a.coll_ni)
    mpc.settings.store_full_solution = False
    mpc.settings.nlpsol_opts = {
        "ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": 0,
        "ipopt.max_iter": int(a.max_iter),
        "ipopt.max_wall_time": float(a.max_wall_time_s),
        # the first-stage input-rate bounds are written straight into lb/ub_opt_x;
        # IPOPT's default bound relaxation (1e-8 * |bound|) would let the applied
        # command drift outside them by ~1e-5 rad/s and make the external safety
        # clip look active when it is not.
        "ipopt.bound_relax_factor": 0.0,
    }

    eta = model.x["eta"]; v = model.x["v"]; qh = model.x["qh"]
    omega = model.u["omega"]; ap = model.u["ap"]
    Phi = ca.DM(pp.Phi)
    omega_hi = float(min(pp.omega_max, omega_cap))

    # ------------------------------------------------------------------
    # Economic objective.  Everything below is a physical quantity divided by a
    # constant reference, so the three weights are directly comparable and no
    # clipping, gating or lobe diagram is involved.
    #
    #   vibration energy   E(eta,v) = 1/2 sum M_i (v_i^2 + omega_i^2 eta_i^2)
    #   pass time          dt = dtheta / omega   (phase step is fixed, so a
    #                      faster spindle finishes the pass sooner)
    #   material removed   MRR dt = a_p a_e (ft N / 2 pi) omega dt
    #                             = a_p a_e (ft N / 2 pi) dtheta
    #                      -- with constant feed per tooth the volume removed per
    #                      unit PHASE depends on a_p only, which is why depth and
    #                      speed appear as separate, non-redundant objectives.
    #
    # Dividing the per-interval cost by the constant dtheta/omega_ref gives
    #
    #   l = q_time (omega_ref/omega) + q_energy (omega_ref/omega) E/E_ref
    #       - q_mrr (a_p/a_p_max)
    #
    # all three terms O(1).  1/omega is smooth on omega >= omega_min > 0.
    Mm = ca.DM(pp.Mm.reshape(-1, 1)); wn2 = ca.DM((pp.wn**2).reshape(-1, 1))
    E = 0.5 * ca.sum1(Mm * v**2) + 0.5 * ca.sum1(Mm * wn2 * eta**2)
    # Normalise the energy by the SAME vibration target the soft constraint uses,
    # so one knob (--w-margin) sets the operating point and the two terms cannot
    # disagree about what "acceptable vibration" means.
    w_target = float(a.w_margin) * pp.w_limit
    E_ref = max(0.5 * float(np.sum(pp.Mm * (pp.wn**2))) * w_target ** 2, 1e-12)

    om_ref = float(a.omega_ref if a.omega_ref else 0.5 * (pp.omega_min + omega_hi))
    om_rng = max(omega_hi - pp.omega_min, 1e-12)
    ap_rng = max(pp.ap_max - pp.ap_min, 1e-12)
    tscale = om_ref / omega                      # = dt / (dtheta/omega_ref)

    lterm = (a.q_time * tscale
             + a.q_energy * tscale * E / E_ref
             - a.q_mrr * (ap / max(pp.ap_max, 1e-12)))
    # Terminal cost must be commensurate with ONE stage of the same running cost
    # (a terminal weight applied to an un-scaled energy would swamp the horizon).
    mterm = a.q_energy * a.q_terminal * E / E_ref
    mpc.set_objective(lterm=lterm, mterm=mterm)
    mpc.set_rterm(omega=float(a.r_omega) / om_rng**2, ap=float(a.r_ap) / ap_rng**2)

    # Upper spindle bound = min(actuator limit, phase-synchronous sampling limit).
    # See note C3: a dtheta-rad control interval lasts dtheta/omega seconds and
    # cannot be shorter than the plant sampling period.
    mpc.bounds["lower", "_u", "omega"] = pp.omega_min
    mpc.bounds["upper", "_u", "omega"] = omega_hi
    if ap_fixed is not None:
        mpc.bounds["lower", "_u", "ap"] = float(ap_fixed)
        mpc.bounds["upper", "_u", "ap"] = float(ap_fixed)
    else:
        mpc.bounds["lower", "_u", "ap"] = pp.ap_min
        mpc.bounds["upper", "_u", "ap"] = pp.ap_max

    mpc.scaling["_x", "eta"] = 1e-3
    mpc.scaling["_x", "v"] = 1e-1
    mpc.scaling["_x", "qh"] = 1e-3

    # --- soft displacement constraints (sensors and cutter point) -----------
    # Expressed on w/w_limit so the slack is DIMENSIONLESS: rho_slack is then
    # directly comparable with the O(1) stage cost, and "penalty per unit
    # fractional overshoot" means the same thing whatever w_limit is.  do-mpc
    # penalises the slack linearly, so with the economic term bounded by q_mrr
    # the constraint dominates as soon as rho_slack >> q_mrr.
    # maximum_violation is deliberately generous: a slack cap that bites turns
    # into an infeasible NLP (and a fallback command) exactly when the
    # controller is most needed.
    w = ca.mtimes(Phi, eta) / pp.w_limit
    wc = ca.mtimes(model.tvp["phi"].T, eta) / pp.w_limit
    marg = float(a.w_margin)
    for nm, ex in (("w_up", w), ("w_dn", -w), ("wc_up", wc), ("wc_dn", -wc)):
        mpc.set_nl_cons(nm, ex, ub=marg, soft_constraint=True,
                        penalty_term_cons=float(a.rho_slack),
                        maximum_violation=float(a.slack_max))

    mpc.set_tvp_fun(mpc_tvp_fun(sched, mpc))
    mpc.setup()
    return mpc


# ===========================================================================
# MHE
# ===========================================================================
def build_mhe(model, pp: PhasePlant, sched: PhaseSchedule, M: int, a: argparse.Namespace):
    mhe = do_mpc.estimator.MHE(model, p_est_list=[])
    mhe.settings.t_step = sched.dtheta
    mhe.settings.n_horizon = int(a.mhe_horizon)
    mhe.settings.meas_from_data = True
    mhe.settings.state_discretization = "collocation"
    mhe.settings.collocation_type = "radau"
    mhe.settings.collocation_deg = int(a.coll_deg)
    mhe.settings.collocation_ni = int(a.coll_ni)
    mhe.settings.nlpsol_opts = {
        "ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": 0,
        "ipopt.max_iter": int(a.max_iter),
        "ipopt.max_wall_time": float(a.max_wall_time_s),
    }

    K = pp.K
    # Information matrices = inverse covariances (do-mpc wants the WEIGHTS).
    P_x = np.linalg.inv(np.diag([a.arr_eta**2] * K + [a.arr_v**2] * K
                                + [a.arr_qh**2] * M + [a.arr_dF**2]))
    P_v = np.linalg.inv(np.diag([a.meas_w**2] * pp.nS + [a.meas_wd**2] * pp.nS))
    P_w = np.linalg.inv(np.diag([a.proc_v**2] * K + [a.proc_dF**2]))
    mhe.set_default_objective(P_x=P_x, P_v=P_v, P_w=P_w)

    mhe.scaling["_x", "eta"] = 1e-3
    mhe.scaling["_x", "v"] = 1e-1
    mhe.scaling["_x", "qh"] = 1e-3

    big = 50.0 * pp.w_limit
    mhe.bounds["lower", "_x", "eta"] = -big
    mhe.bounds["upper", "_x", "eta"] = big
    mhe.bounds["lower", "_x", "qh"] = -big
    mhe.bounds["upper", "_x", "qh"] = big
    mhe.bounds["lower", "_u", "omega"] = pp.omega_min
    mhe.bounds["upper", "_u", "omega"] = pp.omega_max
    mhe.bounds["lower", "_u", "ap"] = pp.ap_min
    mhe.bounds["upper", "_u", "ap"] = pp.ap_max

    mhe.set_tvp_fun(mhe_tvp_fun(sched, mhe))
    mhe.setup()
    return mhe


# ===========================================================================
# Verification
# ===========================================================================
def delay_channel_error(pp: PhasePlant, M: int, n_om: int = 40) -> float:
    """max_nu |G_M(i nu) - exp(-i nu p)| over nu = omega_n/omega on the speed range."""
    _, D = cheb_lobatto_D(M)
    Aq = -D[1:, 1:] / pp.p
    Bq = -D[1:, 0] / pp.p
    eM = np.zeros(M); eM[-1] = 1.0
    oms = np.linspace(pp.omega_min, pp.omega_max, n_om)
    nus = np.unique(np.concatenate([pp.wn / om for om in oms]))
    return max(abs(eM @ np.linalg.solve(1j * nu * np.eye(M) - Aq, Bq)
                   - np.exp(-1j * nu * pp.p)) for nu in nus)


def auto_delay_order(pp: PhasePlant, tol: float, M_min: int = 6, M_max: int = 48) -> int:
    """Smallest even M whose delay-channel error meets `tol`.

    The required order is set by the highest RETAINED mode at the LOWEST speed
    (nu = omega_n/omega), so it is a property of the plant and the speed range,
    not a constant.  A plant whose retained modes are slow relative to the
    spindle needs far fewer nodes -- and every node is a state in the NLP, so
    guessing this high is paid for on every solve.
    """
    for M in range(M_min, M_max + 1, 2):
        if delay_channel_error(pp, M) <= tol:
            return M
    return M_max


def verify_delay(pp: PhasePlant, Ms=(16, 20, 24, 32, 40)) -> None:
    """Frequency-domain error of the pseudospectral delay vs exp(-i nu p)."""
    print("\n=== Delay-channel verification (|G_M(i nu) - e^{-i nu p}|) ===")
    print(f"  retained modes K={pp.K}: f = "
          f"{np.round(pp.wn/2/np.pi,1).tolist()} Hz;  p={pp.p:.4f} rad")
    oms = np.linspace(pp.omega_min, pp.omega_max, 60)
    nus = np.unique(np.concatenate([pp.wn / om for om in oms]))
    print(f"{'M':>4} | {'max err':>10} | {'max|eig(Aq)|':>12} (stiffness)")
    for M in Ms:
        _, D = cheb_lobatto_D(M)
        Aq = -D[1:, 1:] / pp.p
        Bq = -D[1:, 0] / pp.p
        eM = np.zeros(M); eM[-1] = 1.0
        err = max(abs(eM @ np.linalg.solve(1j * nu * np.eye(M) - Aq, Bq)
                      - np.exp(-1j * nu * pp.p)) for nu in nus)
        print(f"{M:>4} | {err:>10.3e} | {np.abs(np.linalg.eigvals(Aq)).max():>12.1f}")


def _plant_modal_force(pp: PhasePlant, theta: float, xc: float, omega: float, ap: float,
                       eta_full: np.ndarray, eta_del_full: np.ndarray):
    """Call the plant's OWN tooth-by-tooth force routine at a prescribed
    (spindle phase, cutter position, speed) operating point.

    The accepted-kinematics history is pinned at the current and one-tooth-back
    instants so that cutter_point()/spindle_phase() reproduce exactly the
    requested phase and position; this makes the comparison an apples-to-apples
    test of APPROX 2 alone.
    """
    from custom_rl.plants import f_nonlinear2_face_milling as f2
    p = 2.0 * np.pi / pp.N
    tau = p / omega
    t0 = 10.0
    t_del = t0 - tau
    feed0 = float(pp.L1 - float(pp.plant.x0_cutter) - xc)
    feed_del = feed0 - pp.ft / 1000.0        # exactly one tooth of table feed
    f2.reset_state_history()
    f2.set_accepted_kinematic_state(t_del, max(feed_del, 0.0), theta - p)
    f2.set_accepted_kinematic_state(t0, max(feed0, 0.0), theta)

    xm = np.zeros(2 * pp.K_full); xm[0::2] = eta_full
    xd = np.zeros(2 * pp.K_full); xd[0::2] = eta_del_full
    _, F_modal, info = f2.compute_face_milling_force(
        t0, xm, xd, omega, ap, pp.ae, t_delay_s=t_del, delay_available=True)
    return np.asarray(F_modal, dtype=np.float64).reshape(-1), info


def _force_sample(pp: PhasePlant, rng, amp, yc):
    """Draw one random operating point and return (q_ref, q_exact, q_smooth, lost)."""
    th = rng.uniform(0.0, 2.0 * np.pi)
    xc = rng.uniform(0.05 * pp.L1, pp.L1)
    om = rng.uniform(pp.omega_min, pp.omega_max)
    ap = rng.uniform(max(pp.ap_min, 1e-3), max(pp.ap_max, 1e-3))
    # only the RETAINED modes are excited, so this isolates the force law from
    # the (separately quantified) mode truncation
    ef = np.zeros(pp.K_full); ed = np.zeros(pp.K_full)
    ef[pp.sel] = rng.normal(0.0, amp, pp.K)
    ed[pp.sel] = rng.normal(0.0, amp, pp.K)
    q_ref_full, info = _plant_modal_force(pp, th, xc, om, ap, ef, ed)
    xc_d, yc_d = info.cutter_xy_delay_m
    wd = float(pp._phi_full(xc_d, yc_d) @ ed)
    lost = bool(np.any(info.engagement & (info.chip_raw_mm <= 0.0)))
    e = ef[pp.sel]
    return (q_ref_full[pp.sel],
            pp.qf_model(th, xc, yc, e, wd, ap, sharp=0.0, eps=0.0),
            pp.qf_model(th, xc, yc, e, wd, ap, sharp=pp._vsharp, eps=pp._veps),
            lost)


def verify_force(pp: PhasePlant, env, n=400, seed=0) -> None:
    """Prediction force law vs the plant's ACTUAL tooth-by-tooth routine.

    Two columns are reported:
      exact-switch  -- sharp=0, eps=0.  This must be ~machine precision; any
                       error here is an algebra bug, not an approximation.
      smoothed      -- the (sharp, chip_eps) actually used inside the NLP.  This
                       is the real APPROX 2 residual.
    """
    print("\n=== Force-law verification (APPROX 2) ===")
    print("  reference = custom_rl.plants.f_nonlinear2_face_milling."
          "compute_face_milling_force")
    print(f"  NLP switches: chip_eps={pp._veps:g} mm, sharp={pp._vsharp:g} "
          f"(feed mark ft={pp.ft:g} mm, kr={pp.kr:.1f} mm/m)")
    yc = float(pp.plant.y_cutter)
    print(f"{'w amplitude':>12} | {'exact switches':>15} | {'NLP switches':>13} | "
          f"{'chip loss':>10}")
    for frac in (0.05, 0.25, 0.60):
        rng = np.random.default_rng(seed)
        e_ex, e_sm, refs, nloss = [], [], [], 0
        for _ in range(n):
            q_ref, q_ex, q_sm, lost = _force_sample(pp, rng, frac * pp.w_limit, yc)
            e_ex.append(float(np.linalg.norm(q_ex - q_ref)))
            e_sm.append(float(np.linalg.norm(q_sm - q_ref)))
            refs.append(float(np.linalg.norm(q_ref)))
            nloss += int(lost)
        den = float(np.sqrt(np.mean(np.asarray(refs) ** 2))) + 1e-300
        print(f"{frac:>11.0%} | {np.sqrt(np.mean(np.asarray(e_ex)**2))/den:>15.2e} | "
              f"{np.sqrt(np.mean(np.asarray(e_sm)**2))/den:>13.2e} | {nloss:>6d}/{n}")
    print("  (NRMSE of the modal acceleration over the retained modes; the first")
    print("   column is an algebra check and must be at machine precision)")

    print(f"  engagement window = [{np.degrees(pp.th_s):.1f}, {np.degrees(pp.th_e):.1f}] deg "
          f"({pp.mode}-milling, immersion {pp.ae/pp.D_mm:.3f}); "
          f"pitch-averaged S0={pp.S0:.4f}, S1={pp.S1:.4f}")

    # rank-1 regenerative structure J_eta = -J_wd phi^T, by finite differences on
    # the smoothed model actually used by the NLP (must hold for any eps).
    th0, xc0, ap0 = 0.7, 0.5 * pp.L1, 2.0
    eta0 = np.full(pp.K, 0.2 * pp.w_limit); wd0 = 0.1 * pp.w_limit
    dh = 1e-10
    f = lambda e, w: pp.qf_model(th0, xc0, yc, e, w, ap0, sharp=pp._vsharp, eps=pp._veps)
    J_eta = np.column_stack([(f(eta0 + dh * np.eye(pp.K)[i], wd0)
                              - f(eta0 - dh * np.eye(pp.K)[i], wd0)) / (2 * dh)
                             for i in range(pp.K)])
    J_wd = (f(eta0, wd0 + dh) - f(eta0, wd0 - dh)) / (2 * dh)
    phi0 = pp.phi_at(xc0, yc)
    res = np.linalg.norm(J_eta + np.outer(J_wd, phi0)) / (np.linalg.norm(J_eta) + 1e-300)
    print(f"  rank-1 regenerative structure: ||J_eta + J_wd phi^T||/||J_eta|| = {res:.2e} "
          f"(J_d = -J_p, rank(J_eta)={np.linalg.matrix_rank(J_eta, tol=1e-12)})")


def verify_prediction(pp: PhasePlant, env, M: int, a: argparse.Namespace,
                      dtheta: float, n_pred=8, warm=4000, seed=0) -> None:
    """Open-loop multi-step prediction of the WHOLE model against the plant.

    This is the end-to-end test: phase change of variables, transport-equation
    delay, tooth-wise force and mode truncation all act together, with no
    feedback to hide a modelling error.  The plant is warmed up under a constant
    command so that a full tooth of history exists, the augmented state is read
    off the true trajectory (the history field q_h is filled from the logged
    cutter displacement at the Chebyshev nodes), and the model is then rolled
    forward in phase with that same constant command.
    """
    print("\n=== Open-loop multi-step prediction vs the environment ===")
    xi, _ = cheb_lobatto_D(M)
    model, _ = build_model(pp, M, a.chip_eps, a.n_quad)
    sim = do_mpc.simulator.Simulator(model)
    sim.settings.t_step = dtheta
    sim.settings.integration_tool = "idas"
    sched = PhaseSchedule(pp, dtheta, a.sharp, a.n_quad)
    tmpl = sim.get_tvp_template()

    def tvp_fun(t_now):
        phi, g, s = sched.at(0)
        tmpl["phi"] = phi.reshape(-1, 1)
        tmpl["gvec"] = g.reshape(-1, 1)
        tmpl["svec"] = s.reshape(-1, 1)
        return tmpl
    sim.set_tvp_fun(tvp_fun)
    sim.setup()

    plant = env.unwrapped.plant
    obs, rinfo = env.reset(seed=seed)
    low = np.asarray(plant._ctrl_low, dtype=np.float64)
    high = np.asarray(plant._ctrl_high, dtype=np.float64)
    om_c = float(np.clip(0.35 * (pp.omega_min + pp.omega_max), pp.omega_min, pp.omega_max))
    ap_c = float(np.clip(0.05 * pp.ap_max, pp.ap_min, pp.ap_max))
    u_c = np.array([om_c, ap_c])
    an = np.clip(2.0 * (u_c[: low.size] - low) / (high - low) - 1.0, -1.0, 1.0)

    th_hist, wc_hist, wcf_hist, w_hist, xm_hist, xc_hist = [], [], [], [], [], []

    def record(info):
        xm = np.asarray(info["x_modal"], dtype=np.float64).reshape(-1)
        xc = float(info.get("cutter_x", pp.L1))
        th_hist.append(float(info["spindle_phase_rad"]))
        xm_hist.append(xm); xc_hist.append(xc)
        phif = pp._phi_full(xc, plant.y_cutter)
        # The transport state must be seeded with the SAME cutter displacement the
        # model itself propagates, i.e. the truncated phi_sel^T eta_sel.  Seeding it
        # with the full-order value while the model computes w_c from K modes leaves
        # a systematic offset in (w_c - w_d) -- the regenerative term -- and that
        # offset is what the cutting force multiplies by k_r = 707 mm/m.
        wc_hist.append(float(phif[pp.sel] @ xm[0::2][pp.sel]))
        wcf_hist.append(float(phif @ xm[0::2]))
        w_hist.append(np.asarray(info["w_sensor"], dtype=np.float64).reshape(-1).copy())

    for _ in range(warm):
        obs, _, term, trunc, info = env.step(an)
        record(info)
        if term or trunc:
            break

    th0 = th_hist[-1]
    xm0 = xm_hist[-1]
    tha = np.asarray(th_hist)
    # augmented state at the snapshot: q_h(xi_j) = w_c(theta0 - p*xi_j)
    qh0 = np.interp(th0 - pp.p * xi[1:], tha, np.asarray(wc_hist))
    x0 = np.concatenate([xm0[0::2][pp.sel], xm0[1::2][pp.sel], qh0, [0.0]]).reshape(-1, 1)
    sched.set_now(th0, xc_hist[-1], plant.y_cutter)
    sim.x0 = x0
    sim.set_initial_guess()

    print(f"  constant command: omega={om_c:.1f} rad/s ({omega_to_rpm(om_c):.0f} rpm), "
          f"ap={ap_c:.3f} mm;  warm-up {len(th_hist)} env steps "
          f"({tha[-1]-tha[0]:.2f} rad = {(tha[-1]-tha[0])/pp.p:.1f} tooth pitches)")
    wcf = np.asarray(wcf_hist); wcs_ = np.asarray(wc_hist)
    print(f"  mode truncation at the cutter: w_c full {wcf[-1]:+.4e} m vs retained "
          f"{wcs_[-1]:+.4e} m ({abs(wcf[-1]-wcs_[-1])/max(abs(wcf[-1]),1e-30):.2%} of the "
          f"displacement, but only "
          f"{abs((wcf[-1]-wcf[0])-(wcs_[-1]-wcs_[0]))/max(abs(wcf[-1]-wcf[0]),1e-30):.2%} "
          f"of its variation)")
    print(f"{'k':>3} | {'phase [rad]':>11} | {'|w| plant':>11} | {'|w| model':>11} | "
          f"{'rel err':>9}")

    w_scale = max(float(np.max(np.abs(np.asarray(w_hist[-500:])))), 1e-12)
    for k in range(1, n_pred + 1):
        # NOTE: do-mpc's Simulator.make_step returns the MEASUREMENT y, not the
        # state; the propagated state is left in sim.x0.
        sim.make_step(u_c.reshape(-1, 1))
        xk = np.asarray(sim.x0.cat, dtype=np.float64).reshape(-1)
        sched.set_now(th0 + k * dtheta, sched.xc + dtheta * sched.dxc, plant.y_cutter)
        n_hold = max(1, int(round((dtheta / om_c) / pp.env_dt)))
        done = False
        for _ in range(n_hold):
            obs, _, term, trunc, info = env.step(an)
            record(info)
            if term or trunc:
                done = True
                break
        w_true = w_hist[-1]
        w_pred = pp.Phi @ xk[: pp.K]
        rel = float(np.linalg.norm(w_pred - w_true) / w_scale)
        print(f"{k:>3} | {th_hist[-1]:>11.3f} | {np.linalg.norm(w_true):>11.3e} | "
              f"{np.linalg.norm(w_pred):>11.3e} | {rel:>9.2e}")
        if done:
            break
    print(f"  (relative error normalised by the recent |w| amplitude {w_scale:.3e} m;")
    print(f"   a growing trend over the horizon is expected -- this is open loop)")


def verify_quadrature(pp: PhasePlant, dtheta: float, n=200, seed=0,
                      nqs=(1, 2, 4, 8, 16)) -> None:
    """Interval-averaging error of the periodic coefficients (APPROX 3).

    do-mpc freezes a TVP across a control interval, so the model can only see an
    interval-constant version of the engagement.  The reference here is the
    exact law averaged over the same interval with 400 nodes; the question is
    how many quadrature nodes the model needs to reproduce it.  This is a
    discretization statement about dtheta, independent of APPROX 2.
    """
    print(f"\n=== Interval-averaging of the periodic coefficients (APPROX 3) ===")
    print(f"  control interval dtheta = {dtheta:.4f} rad = {dtheta/pp.p:.3f} tooth pitch")
    yc = float(pp.plant.y_cutter)
    rng0 = np.random.default_rng(seed)
    pts = [(rng0.uniform(0, 2 * np.pi), rng0.uniform(0.05 * pp.L1, pp.L1),
            rng0.uniform(1e-3, max(pp.ap_max, 1e-3)),
            rng0.normal(0, 0.25 * pp.w_limit, pp.K),
            rng0.normal(0, 0.25 * pp.w_limit)) for _ in range(n)]
    ref = [np.mean([pp.qf_model(th + o, xc, yc, e, wd, ap, sharp=0.0, eps=0.0)
                    for o in (np.arange(400) + 0.5) * dtheta / 400], axis=0)
           for (th, xc, ap, e, wd) in pts]
    den = float(np.sqrt(np.mean([np.dot(r, r) for r in ref]))) + 1e-300
    print(f"{'nq':>4} | {'NRMSE vs interval mean':>24}")
    for nq in nqs:
        err = [np.linalg.norm(pp.qf_model(th, xc, yc, e, wd, ap, sharp=pp._vsharp,
                                          eps=pp._veps, dtheta=dtheta, nq=nq) - r)
               for (th, xc, ap, e, wd), r in zip(pts, ref)]
        print(f"{nq:>4} | {float(np.sqrt(np.mean(np.asarray(err)**2)))/den:>24.2e}")


# ===========================================================================
# helpers
# ===========================================================================
def _sf(v, d=float("nan")):
    try:
        o = float(v)
    except Exception:
        return float(d)
    return o if np.isfinite(o) else float(d)


def _json(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {str(k): _json(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json(x) for x in v]
    return v


def _proc(info, plant):
    pr = face_milling_process_from_info(info, plant)
    for k in ("cutter_x", "cutter_y", "feed_progress", "feed_distance_m", "spindle_phase_rad"):
        if k in info:
            pr[k] = _sf(info[k])
    if "w_sensor" in info:
        ws = np.asarray(info["w_sensor"], dtype=np.float64).reshape(-1)
        pr["max_abs_w_m"] = float(np.max(np.abs(ws))) if ws.size else float("nan")
    if "F_total_N" in info:
        f = np.asarray(info["F_total_N"], dtype=np.float64).reshape(-1)
        if f.size >= 3:
            pr["Fx_N"], pr["Fy_N"], pr["Fz_N"] = _sf(f[0]), _sf(f[1]), _sf(f[2])
            pr["F_mag_N"] = _sf(float(np.linalg.norm(f)))
    return {k: _sf(pr.get(k, np.nan)) for k in PROCESS_KEYS}


def _deint(xm, Kf):
    x = np.asarray(xm, dtype=np.float64).reshape(-1)
    return np.concatenate([x[0::2], x[1::2]])


# ===========================================================================
# Closed loop (phase-synchronous execution on the time-domain gym plant)
# ===========================================================================
def run_episode(env, pp: PhasePlant, model, M, a, *, seed, reset_options, metadata):
    plant = env.unwrapped.plant
    K = pp.K
    dtheta = a.dtheta_frac * pp.p

    obs, rinfo = env.reset(seed=seed, options=reset_options)
    oa = np.asarray(obs, dtype=np.float64).reshape(-1)
    ap_fixed = None
    if not pp.control_ap:
        ap_fixed = float(rinfo.get("ap_fixed_mm",
                         getattr(plant, "_episode_ap", 0.5 * (pp.ap_min + pp.ap_max))))

    # C3: a dtheta-rad interval lasts dtheta/omega s and cannot be shorter than
    # the plant sampling period -> omega <= dtheta/dt_env.
    omega_cap = dtheta / pp.env_dt if a.omega_sampling_cap else float("inf")
    omega_hi = float(min(pp.omega_max, omega_cap))

    sched = PhaseSchedule(pp, dtheta, a.sharp, a.n_quad)
    xc, yc = pp.cutter_from_obs(oa)
    theta_now = float(rinfo.get("spindle_phase_rad", 0.0))
    sched.set_now(theta_now, xc, yc)

    mpc = build_mpc(model, pp, sched, M, a, ap_fixed, omega_cap)
    mhe = build_mhe(model, pp, sched, M, a)

    z0m = pp.modal_from_obs(oa)
    phi0 = pp.phi_at(xc, yc)
    wc0 = float(phi0 @ z0m[:K])
    x0 = np.concatenate([z0m[:K], z0m[K:], np.full(M, wc0), [0.0]]).reshape(-1, 1)
    u0 = np.array([[float(np.clip(0.5 * (pp.omega_min + omega_hi), pp.omega_min, omega_hi))],
                   [ap_fixed if ap_fixed is not None else pp.ap_min]])
    mhe.x0 = x0; mhe.u0 = u0; mhe.set_initial_guess()
    mpc.x0 = x0; mpc.u0 = u0; mpc.set_initial_guess()

    u_prev = u0.reshape(-1).copy()

    logs = dict(times=[], observations=[], physical_signals=[], x_modal=[], actions=[],
                physical_actions=[], rewards=[], w_sensor=[], wdot_sensor=[],
                process={k: [] for k in PROCESS_KEYS})
    d = dict(feasible=[], fallback_used=[], solve_time_s=[], mhe_solve_ok=[],
             mhe_solve_time_s=[], mhe_output_err=[], mhe_modal_err=[], ekf_output_err=[],
             ekf_modal_err=[], dtheta_realized=[], rate_clip_active=[], n_hold=[],
             u_applied_omega=[], u_applied_ap=[], dt_interval_s=[])

    reason = None; term_f = trunc_f = passed = False
    step = 0; n_ctrl = 0; t0 = time.time(); x_hat = x0.copy()

    while step < a.max_env_steps:
        # ---- C4: rate limits on the APPLIED input, written into the NLP ------
        # The interval that the new command is held for is dtheta/omega_new, and
        # omega_new is what we are about to choose: raising the speed SHORTENS
        # the interval, so sizing the bound with omega_prev lets the realized
        # rate overshoot by omega_new/omega_prev.  Size it with the fastest speed
        # reachable this step, and with the integer hold that speed would give,
        # so the realized rate can only come in under the limit.
        om_next_hi = min(omega_hi,
                         u_prev[0] + float(a.omega_rate) * dtheta / max(u_prev[0], 1e-9))
        n_hold_min = max(1, int(round((dtheta / max(om_next_hi, 1e-9)) / pp.env_dt)))
        dt_int = n_hold_min * pp.env_dt
        d_om = float(a.omega_rate) * dt_int          # [rad/s^2] * [s]
        d_ap = float(a.ap_rate) * dt_int             # [mm/s]    * [s]
        om_lo = float(max(pp.omega_min, u_prev[0] - d_om))
        om_hi = float(min(omega_hi, u_prev[0] + d_om))
        try:
            mpc.lb_opt_x["_u", 0, 0, "omega"] = om_lo
            mpc.ub_opt_x["_u", 0, 0, "omega"] = om_hi
            if ap_fixed is None:
                mpc.lb_opt_x["_u", 0, 0, "ap"] = float(max(pp.ap_min, u_prev[1] - d_ap))
                mpc.ub_opt_x["_u", 0, 0, "ap"] = float(min(pp.ap_max, u_prev[1] + d_ap))
            rate_in_nlp = True
        except Exception:
            rate_in_nlp = False

        tA = time.perf_counter()
        try:
            u = np.asarray(mpc.make_step(x_hat), dtype=np.float64).reshape(-1)
            ok = bool(np.all(np.isfinite(u)))
        except Exception:
            u, ok = u_prev.copy(), False
        d["solve_time_s"].append(time.perf_counter() - tA)
        d["feasible"].append(ok); d["fallback_used"].append(not ok)
        if not ok:
            u = u_prev.copy()

        # Emergency net.  With rate_in_nlp the OCP already satisfies these, so a
        # non-zero rate_clip_active_frac means the NLP bound did NOT take effect.
        u_raw = u.copy()
        u[0] = float(np.clip(np.clip(u[0], om_lo, om_hi), pp.omega_min, omega_hi))
        u[1] = (float(ap_fixed) if ap_fixed is not None else
                float(np.clip(np.clip(u[1], u_prev[1] - d_ap, u_prev[1] + d_ap),
                              pp.ap_min, pp.ap_max)))
        tol = 1e-6 * np.maximum(1.0, np.abs(u_raw))
        d["rate_clip_active"].append(bool(ok and rate_in_nlp
                                          and np.any(np.abs(u - u_raw) > tol)))

        low = np.asarray(plant._ctrl_low, dtype=np.float64)
        high = np.asarray(plant._ctrl_high, dtype=np.float64)
        an = np.clip(2.0 * (u[: low.size] - low) / (high - low) - 1.0, -1.0, 1.0)

        # phase-synchronous hold: dt = dtheta / omega  ->  n env steps
        n_hold = max(1, int(round((dtheta / max(u[0], 1e-9)) / pp.env_dt)))
        d["n_hold"].append(int(n_hold))
        d["u_applied_omega"].append(float(u[0]))
        d["u_applied_ap"].append(float(u[1]))
        d["dt_interval_s"].append(float(n_hold * pp.env_dt))
        theta_before = theta_now
        done = False
        for _ in range(n_hold):
            obs, rew, term, trunc, info = env.step(an)
            oa = np.asarray(obs, dtype=np.float64).reshape(-1)
            logs["actions"].append(an.tolist())
            logs["physical_actions"].append([float(u[0]), float(u[1])])
            logs["observations"].append(oa.tolist())
            ws = np.asarray(info.get("w_sensor", np.zeros(pp.nS)), dtype=np.float64).reshape(-1)
            wds = np.asarray(info.get("wdot_sensor", np.zeros(pp.nS)), dtype=np.float64).reshape(-1)
            logs["w_sensor"].append(ws.tolist()); logs["wdot_sensor"].append(wds.tolist())
            logs["physical_signals"].append(np.concatenate([ws, wds]).tolist())
            if "x_modal" in info:
                logs["x_modal"].append(np.asarray(info["x_modal"], dtype=np.float64).reshape(-1).tolist())
            for k2, v2 in _proc(info, plant).items():
                logs["process"][k2].append(v2)
            logs["rewards"].append(float(rew))
            logs["times"].append(float(info.get("t", (step + 1) * pp.env_dt)))
            step += 1
            if term or trunc:
                term_f, trunc_f = bool(term), bool(trunc)
                reason = info.get("termination_reason")
                passed = bool(info.get("pass_completed") or reason in PASS_COMPLETED_REASONS)
                done = True
                break
            if step >= a.max_env_steps:
                done = True
                break

        # ---- re-synchronise the schedule to the plant's OWN spindle phase ----
        # (never integrate the nominal dtheta: the hold length is an integer
        #  number of environment steps, so the realized advance differs)
        xc, yc = pp.cutter_from_obs(oa)
        theta_now = float(_sf(info.get("spindle_phase_rad"), theta_before + dtheta))
        d["dtheta_realized"].append(float(theta_now - theta_before))
        sched.set_now(theta_now, xc, yc)

        y = np.concatenate([pp.meas_from_obs(oa), [u[0], u[1]]]).reshape(-1, 1)
        tB = time.perf_counter()
        try:
            x_mhe = np.asarray(mhe.make_step(y), dtype=np.float64).reshape(-1, 1)
            eok = bool(np.all(np.isfinite(x_mhe)))
        except Exception:
            x_mhe, eok = None, False
        d["mhe_solve_time_s"].append(time.perf_counter() - tB)
        n_ctrl += 1
        # do-mpc: "MHE will only work reliably once a full sequence of
        # measurements corresponding to the set horizon is available."
        warm = n_ctrl <= int(a.mhe_horizon) + 1
        if eok and not warm:
            x_hat = x_mhe
        else:
            zm = pp.modal_from_obs(oa)
            phi = pp.phi_at(xc, yc)
            x_hat = np.concatenate([zm[:K], zm[K:],
                                    np.full(M, float(phi @ zm[:K])), [0.0]]).reshape(-1, 1)
        d["mhe_solve_ok"].append(bool(eok and not warm))

        if "x_modal" in info:
            zt = _deint(info["x_modal"], pp.K_full)[np.concatenate([pp.sel, pp.K_full + pp.sel])]
            wt = pp.Phi @ zt[:K]; we = pp.Phi @ x_hat[:K, 0]
            oe = float(np.linalg.norm(we - wt) / (np.linalg.norm(wt) + 1e-12))
            me = float(np.linalg.norm(x_hat[:K, 0] - zt[:K]) / (np.linalg.norm(zt[:K]) + 1e-12))
            d["mhe_output_err"].append(oe); d["ekf_output_err"].append(oe)
            d["mhe_modal_err"].append(me); d["ekf_modal_err"].append(me)

        u_prev = u.copy()
        if a.progress_every and n_ctrl % int(a.progress_every) == 0:
            el = time.time() - t0
            wmax = float(np.max(np.abs(np.asarray(logs["w_sensor"])))) if logs["w_sensor"] else 0.0
            print(f"      [{n_ctrl:5d} ctrl / {step:6d} env steps] {el:6.1f}s "
                  f"({1e3*el/max(n_ctrl,1):.0f} ms/ctrl) omega={u[0]:7.1f} ap={u[1]:6.3f} "
                  f"max|w|={wmax:.2e} mpc_ok={100*np.mean(d['feasible']):.0f}%", flush=True)
        if done:
            break

    wall = time.time() - t0
    warr = np.asarray(logs["w_sensor"], dtype=np.float64)
    maxw = float(np.max(np.abs(warr))) if warr.size else float("nan")
    # The last hold is cut short whenever the episode ends mid-interval (env-step
    # cap or termination), so including it reports a phase-step error that the
    # controller never actually committed.  Judge the completed intervals.
    dth = np.asarray(d["dtheta_realized"], dtype=np.float64)
    dth_full = dth[:-1] if dth.size > 1 else dth
    dth_err = (float(np.max(np.abs(dth_full - dtheta)) / dtheta)
               if dth_full.size else float("nan"))
    oms = np.asarray([pa[0] for pa in logs["physical_actions"]], dtype=np.float64)
    aps = np.asarray([pa[1] for pa in logs["physical_actions"]], dtype=np.float64)
    # Realized actuator rates, differenced ACROSS CONTROL INTERVALS.  The command
    # is a zero-order hold: differentiating it at the environment sample rate
    # divides a finite step by dt_env and reports a spurious rate ~n_hold times
    # too large.  These are the physically meaningful numbers.
    uo = np.asarray(d["u_applied_omega"], dtype=np.float64)
    ua = np.asarray(d["u_applied_ap"], dtype=np.float64)
    dti = np.asarray(d["dt_interval_s"], dtype=np.float64)
    if uo.size > 1:
        om_rate = float(np.max(np.abs(np.diff(uo)) / np.maximum(dti[1:], 1e-30)))
        ap_rate = float(np.max(np.abs(np.diff(ua)) / np.maximum(dti[1:], 1e-30)))
    else:
        om_rate = ap_rate = float("nan")
    print(f"    seed {seed}: env_steps={len(logs['rewards'])} return={sum(logs['rewards']):.1f} "
          f"pass={'Y' if passed else 'N'} reason={reason} max|w|={maxw:.3e} "
          f"mpc_ok={100*np.mean(d['feasible']) if d['feasible'] else 0:.0f}% "
          f"mhe_ok={100*np.mean(d['mhe_solve_ok']) if d['mhe_solve_ok'] else 0:.0f}% wall={wall:.1f}s")
    print(f"      omega [rad/s] {oms.min():.1f}..{oms.max():.1f} (span {oms.max()-oms.min():.1f}); "
          f"ap [mm] {aps.min():.3f}..{aps.max():.3f}")
    print(f"      dtheta nominal {dtheta:.4f} rad, realized "
          f"{np.mean(dth_full) if dth_full.size else float('nan'):.4f} +- "
          f"{np.std(dth_full) if dth_full.size else float('nan'):.4f} "
          f"(max rel dev {dth_err:.1%}, completed intervals only); "
          f"n_hold {int(np.min(d['n_hold'])) if d['n_hold'] else 0}"
          f"..{int(np.max(d['n_hold'])) if d['n_hold'] else 0} env steps")
    print(f"      realized rates (per control interval): |d omega/dt|max "
          f"{om_rate:.3g} rad/s^2 (limit {a.omega_rate:g}), |d ap/dt|max "
          f"{ap_rate:.3g} mm/s (limit {a.ap_rate:g})")
    print(f"      external rate clip active on "
          f"{100*np.mean(d['rate_clip_active']) if d['rate_clip_active'] else 0:.1f}% of steps "
          f"(0% == rate limits fully enforced inside the OCP)")

    return {
        "controller": "mpc", "estimator": "mhe-phase", "metadata": metadata,
        "scenario_seed": int(seed), "reset_options": reset_options,
        "reset_info": _json(dict(rinfo)),
        "y_line_m": _sf(rinfo.get("y_cutter", np.nan)),
        "ap_fixed_mm": _sf(ap_fixed) if ap_fixed is not None else float("nan"),
        **{k: logs[k] for k in ("times", "observations", "physical_signals", "x_modal",
                                "actions", "physical_actions", "rewards", "w_sensor",
                                "wdot_sensor")},
        "process": logs["process"],
        "return": float(sum(logs["rewards"])), "length": int(len(logs["rewards"])),
        "terminated": term_f, "truncated": trunc_f, "pass_completed": passed,
        "termination_reason": reason,
        "mpc_diagnostics": d,
        "mhe": {"estimator": "mhe-phase", "n_modes": pp.K, "n_delay": M,
                "n_states": int(2 * pp.K + M + 1), "n_measurements": int(2 * pp.nS),
                "dtheta_rad": dtheta,
                "mean_output_err": float(np.mean(d["mhe_output_err"])) if d["mhe_output_err"] else float("nan"),
                "final_modal_err": float(d["mhe_modal_err"][-1]) if d["mhe_modal_err"] else float("nan")},
        "ekf": {"feedback": "mhe-phase", "n_measurements": int(2 * pp.nS),
                "n_states": int(2 * pp.K + M + 1), "observability_rank": int(2 * pp.K + M + 1),
                "observability_full": True, "process_noise_scale": float(a.proc_v),
                "measurement_noise_scale": float(a.meas_w), "mean_nis": float("nan"),
                "mean_output_err": float(np.mean(d["mhe_output_err"])) if d["mhe_output_err"] else float("nan"),
                "final_modal_err": float(d["mhe_modal_err"][-1]) if d["mhe_modal_err"] else float("nan")},
        "summary": {
            "return": float(sum(logs["rewards"])), "length": int(len(logs["rewards"])),
            "max_abs_w_m": maxw,
            "rms_w_m": float(np.sqrt(np.mean(warr**2))) if warr.size else float("nan"),
            "pass_completed": passed, "termination_reason": reason,
            "mpc_solve_feasible_frac": float(np.mean(d["feasible"])) if d["feasible"] else 0.0,
            "mpc_fallback_frac": float(np.mean(d["fallback_used"])) if d["fallback_used"] else 1.0,
            "mpc_mean_solve_time_s": float(np.mean(d["solve_time_s"])) if d["solve_time_s"] else float("nan"),
            "mhe_solve_ok_frac": float(np.mean(d["mhe_solve_ok"])) if d["mhe_solve_ok"] else 0.0,
            "mhe_mean_output_err": float(np.mean(d["mhe_output_err"])) if d["mhe_output_err"] else float("nan"),
            "mpc_wall_time_s": float(wall),
            "dtheta_nominal_rad": float(dtheta),
            "dtheta_realized_mean_rad": (float(np.mean(dth_full)) if dth_full.size
                                        else float("nan")),
            "dtheta_max_rel_dev": dth_err,
            "rate_clip_active_frac": (float(np.mean(d["rate_clip_active"]))
                                      if d["rate_clip_active"] else float("nan")),
            "realized_max_omega_rate_rad_s2": om_rate,
            "realized_max_ap_rate_mm_s": ap_rate,
            "omega_rate_limit_rad_s2": float(a.omega_rate),
            "ap_rate_limit_mm_s": float(a.ap_rate),
            "omega_span_rad_s": float(oms.max() - oms.min()) if oms.size else float("nan"),
            "modal_participation_kept": float(pp.participation_kept)},
    }


# ===========================================================================
# CLI
# ===========================================================================
def main() -> int:
    P = argparse.ArgumentParser(description="Phase-domain pseudospectral NMPC+MHE.")
    P.add_argument("--env-id", default="CustomODEPlate-v0",
                   choices=["CustomODEPlate-v0", "CustomODEPlateFinish-v0"])
    P.add_argument("--seeds", nargs="+", type=int, default=[0])
    P.add_argument("--n-episodes", type=int, default=1)
    P.add_argument("--out-dir", default="mpc_phase_trajectories")
    P.add_argument("--dt", type=float, default=None,
                   help="Plant RK substep [s].  Default AUTO: chosen so the control "
                        "interval spans at least --min-n-hold environment steps at the "
                        "maximum spindle speed (capped at 1e-4 s for integration "
                        "accuracy).  A fixed value here is plant-specific -- 1e-5 suits a "
                        "40000 rpm spindle and needlessly slows a 4000 rpm one by 10x.")
    P.add_argument("--min-n-hold", type=int, default=10,
                   help="Environment steps per control interval at omega_max; sets AUTO --dt.")
    P.add_argument("--n-substeps", type=int, default=1,
                   help="Environment steps hold dt*n_substeps seconds.  Phase-synchronous "
                        "control needs dt_env <= dtheta/omega_max, hence 1 (0.1 ms) here "
                        "rather than the RL default of 10.")
    P.add_argument("--max-episode-steps", type=int, default=None)
    P.add_argument("--max-env-steps", type=int, default=20000)
    P.add_argument("--reward", default="dense",
                   choices=["dense", "productive", "quadratic", "sparse"])
    P.add_argument("--dynamics-uncertainty-std", type=float, default=0.0)
    P.add_argument("--randomize-y0", action=argparse.BooleanOptionalAction, default=True)
    P.add_argument("--plant-override", default=None,
                   help="JSON dict of plant kwargs (e.g. '{\"omega_max\":418.879,"
                        "\"w_limit\":0.003}') for reproducing another machine's setup.")
    P.add_argument("--y0", type=float, default=None)
    P.add_argument("--ap", type=float, default=None)
    # model size
    P.add_argument("--n-modes", type=int, default=3, help="Retained modes K (default 3).")
    P.add_argument("--n-delay", type=int, default=None,
                   help="Collocation order M for the delay line.  Default AUTO: the "
                        "smallest order meeting --delay-tol over the actual retained "
                        "modes and speed range.  Every node is an NLP state.")
    P.add_argument("--delay-tol", type=float, default=1.0e-6,
                   help="Target delay-channel error used by AUTO --n-delay.")
    P.add_argument("--dtheta-frac", type=float, default=0.25,
                   help="Control interval as a fraction of one tooth pitch p.")
    P.add_argument("--sharp", type=float, default=0.0,
                   help="Engagement-window sigmoid sharpness.  Default 0 (exact "
                        "indicator): theta is a known TVP, not a decision variable, so "
                        "smoothing it buys the NLP nothing and only biases the force.")
    P.add_argument("--n-quad", type=int, default=2,
                   help="Midpoint nodes per control interval used to average the "
                        "periodic engagement coefficients (note APPROX 3).")
    P.add_argument("--chip-eps", type=float, default=2.0e-3,
                   help="Chip-loss smoothing eps [mm] in max(h,0) ~ (h+sqrt(h^2+eps^2))/2.")
    P.add_argument("--coll-deg", type=int, default=2,
                   help="Radau collocation degree per finite element.")
    P.add_argument("--coll-ni", type=int, default=1,
                   help="Finite elements per control interval.")
    P.add_argument("--omega-sampling-cap", action=argparse.BooleanOptionalAction, default=True,
                   help="Bound omega <= dtheta/dt_env so the phase-synchronous control "
                        "interval is realizable on the time-sampled plant (note C3).")
    # physical actuator rate limits (converted to per-interval bounds with dt=dtheta/omega)
    P.add_argument("--omega-rate", type=float, default=2.0e4,
                   help="Spindle angular acceleration limit [rad/s^2].")
    P.add_argument("--ap-rate", type=float, default=20.0,
                   help="Axial-depth (Z axis) feed-rate limit [mm/s].")
    # horizons
    P.add_argument("--mpc-horizon", type=int, default=6)
    P.add_argument("--mhe-horizon", type=int, default=6)
    # objective
    P.add_argument("--q-energy", type=float, default=1.0,
                   help="Weight on the (time-integrated) modal vibration energy.")
    P.add_argument("--q-time", type=float, default=1.0,
                   help="Weight on elapsed pass time (rewards higher spindle speed).")
    P.add_argument("--q-mrr", type=float, default=1.0,
                   help="Weight on removed volume (rewards higher axial depth).")
    P.add_argument("--q-terminal", type=float, default=3.0,
                   help="Terminal vibration-energy weight, in units of one stage cost.")
    P.add_argument("--r-omega", type=float, default=1.0e-2)
    P.add_argument("--r-ap", type=float, default=1.0e-2)
    P.add_argument("--omega-ref", type=float, default=None)
    P.add_argument("--w-margin", type=float, default=0.35,
                   help="VIBRATION TARGET as a fraction of w_limit.  This single knob "
                        "sets the productivity/vibration operating point: it is both the "
                        "soft displacement bound and the scale that normalises the "
                        "vibration-energy term.  The economic objective pushes depth and "
                        "speed up until this binds, so a plant whose w_limit is a safety "
                        "trip rather than a finish requirement wants a small value here.")
    P.add_argument("--rho-slack", type=float, default=1.0e4,
                   help="Linear penalty on the DIMENSIONLESS displacement slack "
                        "(units of |w|/w_limit).  Must exceed --q-mrr by orders of "
                        "magnitude for the limit to dominate the economic term.")
    P.add_argument("--slack-max", type=float, default=2.0,
                   help="Maximum admissible slack in units of w_limit.  Kept large on "
                        "purpose: a binding slack cap makes the NLP infeasible.")
    # MHE weights
    P.add_argument("--meas-w", type=float, default=2.0e-2)
    P.add_argument("--meas-wd", type=float, default=5.0e-2)
    P.add_argument("--proc-v", type=float, default=5.0e-2)
    P.add_argument("--proc-dF", type=float, default=1.0e-2)
    P.add_argument("--arr-eta", type=float, default=1.0e-3)
    P.add_argument("--arr-v", type=float, default=1.0e-1)
    P.add_argument("--arr-qh", type=float, default=1.0e-3)
    P.add_argument("--arr-dF", type=float, default=1.0e-1)
    # solver
    P.add_argument("--max-iter", type=int, default=200)
    P.add_argument("--max-wall-time-s", type=float, default=2.0)
    P.add_argument("--progress-every", type=int, default=100,
                   help="Print a closed-loop progress line every N control steps (0 = off).")
    # verification
    P.add_argument("--verify-delay", action="store_true")
    P.add_argument("--verify-force", action="store_true")
    P.add_argument("--verify-prediction", action="store_true",
                   help="Open-loop multi-step prediction of the full model vs the env.")
    P.add_argument("--verify-only", action="store_true")
    a = P.parse_args()

    register_envs()
    out = Path(a.out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    override = json.loads(a.plant_override) if a.plant_override else {}

    def _make(dt):
        kw = plate_env_kwargs(
            reward_id=a.reward, dt=dt, n_substeps=a.n_substeps,
            max_episode_steps=a.max_episode_steps,
            dynamics_uncertainty_std=a.dynamics_uncertainty_std,
            randomize_y0=a.randomize_y0)
        kw.update(override)
        return gym.make(a.env_id, **kw)

    # ---- AUTO sizing (both quantities are properties of the PLANT, not constants)
    # A probe env is needed because dtheta depends on the tooth count and the
    # required dt depends on the maximum spindle speed.
    probe = _make(1.0e-4)
    pr = PhasePlant(probe, n_modes=a.n_modes)
    dtheta = a.dtheta_frac * pr.p
    dt_auto = a.dt
    if dt_auto is None:
        # hold at least --min-n-hold env steps per control interval at omega_max
        dt_auto = min(1.0e-4, dtheta / (pr.omega_max * max(1, int(a.min_n_hold))
                                        * max(1, int(a.n_substeps))))
        dt_auto = float(f"{dt_auto:.1e}")          # a round number, not 9.37e-05
    probe.close()

    env = _make(dt_auto)
    pp = PhasePlant(env, n_modes=a.n_modes)
    pp._vsharp = float(a.sharp); pp._veps = float(a.chip_eps)

    M = int(a.n_delay) if a.n_delay else auto_delay_order(pp, float(a.delay_tol))

    dtheta = a.dtheta_frac * pp.p
    omega_cap = dtheta / pp.env_dt if a.omega_sampling_cap else float("inf")
    omega_hi = min(pp.omega_max, omega_cap)

    print(f"Phase-domain pseudospectral NMPC + MHE")
    print(f"  modes K={pp.K}/{pp.K_full} (f={np.round(pp.wn/2/np.pi,1).tolist()} Hz), "
          f"delay order M={M}{'' if a.n_delay else ' (AUTO)'} "
          f"[delay-channel error {delay_channel_error(pp, M):.1e}]  ->  n_z={2*pp.K+M+1}")
    if a.dt is None:
        print(f"  dt AUTO = {dt_auto:g} s (>= {a.min_n_hold} env steps per control "
              f"interval at omega_max)")
    fn_all = np.asarray(pp.plant.omega_vec).reshape(-1) / (2 * np.pi)
    print("  cutter-point compliance participation phi^2/(M wn^2), pass-averaged:")
    for i in np.argsort(-pp.participation):
        print(f"      mode {i}  {fn_all[i]:8.2f} Hz   {100*pp.participation[i]:6.2f} %"
              f"{'   [KEPT]' if i in pp.sel else ''}")
    print(f"    retained share = {100*pp.participation_kept:.3f} %  "
          f"(remainder enters only through the measurement and d_F)")
    print(f"  tooth pitch p={pp.p:.4f} rad, control interval dtheta={dtheta:.4f} rad "
          f"({a.dtheta_frac:.2f} p), dt_env={pp.env_dt*1e3:.3f} ms")
    print(f"  spindle {omega_to_rpm(pp.omega_min):.0f}-{omega_to_rpm(pp.omega_max):.0f} rpm, "
          f"ap {pp.ap_min:g}-{pp.ap_max:g} mm, |w|<={pp.w_limit:g} m (hard/termination)")
    print(f"  VIBRATION TARGET --w-margin {a.w_margin:g} -> {a.w_margin*pp.w_limit:g} m. "
          f"The economic objective raises depth and speed until this binds, so this "
          f"number -- not w_limit -- sets the operating point.")
    if np.isfinite(omega_cap):
        print(f"  phase-synchronous sampling limit (C3): omega <= dtheta/dt_env = "
              f"{omega_cap:.1f} rad/s ({omega_to_rpm(omega_cap):.0f} rpm) -> MPC upper "
              f"bound {omega_hi:.1f} rad/s ({omega_to_rpm(omega_hi):.0f} rpm, "
              f"{100*(omega_hi-pp.omega_min)/(pp.omega_max-pp.omega_min):.0f}% of the "
              f"actuator span retained)")
        if omega_cap < pp.omega_min:
            print("  ERROR: dtheta/dt_env < omega_min -- increase --dtheta-frac or "
                  "decrease --n-substeps/--dt.")
            env.close(); return 2
    print(f"  actuator rates: |d omega/dt| <= {a.omega_rate:g} rad/s^2, "
          f"|d ap/dt| <= {a.ap_rate:g} mm/s "
          f"(-> {a.omega_rate*dtheta/omega_hi:.1f} rad/s and "
          f"{a.ap_rate*dtheta/omega_hi:.4f} mm per interval at omega_max)")

    # The control interval is held for an INTEGER number of environment steps,
    # so the realized phase advance is omega*n_hold*dt_env, not exactly dtheta.
    # Show that rounding across the speed range instead of discovering it later.
    print("  phase-hold resolution (realized dtheta = omega * n_hold * dt_env):")
    worst = 0.0
    for om in np.linspace(pp.omega_min, omega_hi, 7):
        nh = max(1, int(round(dtheta / (om * pp.env_dt))))
        dev = abs(om * nh * pp.env_dt - dtheta) / dtheta
        worst = max(worst, dev)
        print(f"      omega {om:8.1f} rad/s ({omega_to_rpm(om):6.0f} rpm): "
              f"n_hold {nh:5d}  ->  dtheta {om*nh*pp.env_dt:.4f} rad ({dev:+.1%})")
    if worst > 0.10:
        print(f"    WARNING: worst-case phase-step rounding {worst:.0%}.  Reduce --dt "
              f"or raise --dtheta-frac; the prediction assumes a {dtheta:.4f} rad step.")
    else:
        print(f"    worst-case rounding over the admissible speed range: {worst:.1%}")

    # ---- how much of the pass this run will actually cover -------------------
    # x_c advances by |dxc| per radian of phase INDEPENDENTLY of spindle speed,
    # so the phase (and therefore the number of control steps) needed for a full
    # pass is a fixed property of the geometry.
    dxc = pp.ft * pp.N / (2.0 * np.pi * 1000.0)          # m of feed per rad
    th_pass = pp.L1 / max(dxc, 1e-30)
    n_ctrl_pass = th_pass / dtheta
    om_typ = 0.5 * (pp.omega_min + omega_hi)
    n_env_pass = th_pass / (om_typ * pp.env_dt)
    frac = min(1.0, a.max_env_steps / max(n_env_pass, 1e-30))
    print(f"  pass budget: {th_pass:.0f} rad of phase for the full {pp.L1:g} m pass "
          f"-> {n_ctrl_pass:.0f} control steps, ~{n_env_pass:.3g} env steps at "
          f"{omega_to_rpm(om_typ):.0f} rpm")
    print(f"    --max-env-steps {a.max_env_steps} covers about {frac:.1%} of the pass"
          + ("  <-- this is a short TRANSIENT, not steady state; the controller will "
             "still be ramping" if frac < 0.5 else ""))
    print(f"    a full pass is roughly {n_ctrl_pass * 0.3 / 3600:.1f} h at ~0.3 s/solve; "
          f"raise --dtheta-frac to trade intra-tooth resolution for wall time "
          f"(--dtheta-frac 1.0 costs 1/{1.0/max(a.dtheta_frac,1e-9):.0f} of that)")

    if a.verify_delay or a.verify_only:
        verify_delay(pp)
    if a.verify_force or a.verify_only:
        verify_force(pp, env)
        verify_quadrature(pp, dtheta)
    if a.verify_prediction or a.verify_only:
        verify_prediction(pp, env, M, a, dtheta)
    if a.verify_only:
        env.close(); return 0

    model, _ = build_model(pp, M, a.chip_eps, a.n_quad)
    meta = plant_plot_metadata(env)
    meta.update(controller="mpc", estimator="mhe-phase", reward_id=a.reward)
    meta["mpc"] = {"framework": "do-mpc phase-domain pseudospectral",
                   "n_modes": pp.K, "n_delay": M, "mpc_horizon": a.mpc_horizon,
                   "mhe_horizon": a.mhe_horizon, "dtheta_rad": a.dtheta_frac * pp.p,
                   "w_limit": pp.w_limit, "w_inner_limit": a.w_margin * pp.w_limit}

    for seed in a.seeds:
        trajs = []
        for ep in range(a.n_episodes):
            ro: dict[str, Any] = {}
            if a.y0 is not None:
                ro["y0"] = float(np.clip(a.y0, 0.0, pp.L2))
            if a.ap is not None and not pp.control_ap:
                ro["ap"] = float(a.ap)
            print(f"  [seed {seed} ep {ep}] phase-domain closed loop...")
            r = run_episode(env, pp, model, M, a, seed=int(seed * 1000 + ep),
                            reset_options=ro or None, metadata=meta)
            r["episode"] = int(ep); r["seed"] = int(seed)
            trajs.append(r)
        p = out / f"trajectories_seed{seed}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(_json(trajs), f, indent=2)
        print(f"  saved -> {p}")

    env.close()
    print("Done.")
    print(f"Compare: python scripts/compare_mpc_rl.py --rl-dir {DEFAULT_TRAJ_DIR} --mpc-dir {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
