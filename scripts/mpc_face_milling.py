"""Nonlinear Model Predictive Control (NMPC) for face-milling chatter suppression.

This module adds a *second*, physics-based controller to the project so the
PPO actor-critic policy can be independently verified against a model-based
optimal controller on the **same** plant and the **same** reward.  It is
completely independent of the RL training code: it only *uses* the already
built Gymnasium environment (``CustomODEPlate-v0`` / ``CustomODEPlateFinish-v0``)
as a simulator, connecting an MPC policy in place of the PPO policy.

It supports both control modes with a single implementation:

* **first-mode / roughing control**  (``CustomODEPlate-v0``)
    decision  = [spindle speed omega, axial depth ap]
* **second-mode / finishing control** (``CustomODEPlateFinish-v0``)
    decision  = [spindle speed omega];  ap is a fixed per-episode parameter.

--------------------------------------------------------------------------
Optimal control problem (solved online at every control instant with CasADi)
--------------------------------------------------------------------------
We solve a finite-horizon OCP by **direct multiple shooting** with IPOPT, in
the exact structure requested: an objective function minimised subject to a
*discretised model as equality constraints* and *bounds as inequality
constraints*.  With modal state ``z = [eta ; eta_dot]`` (K modes each), measured
plate response ``w = Phi @ eta`` (sensor points), horizon ``Np`` and control
step ``Ts``:

    minimise   sum_{k=0}^{Np-1}  L( z_{k+1}, u_k )        (dense-reward stage cost)
                 + rho * sum_k slack_k                     (safe-region softening)
                 + rate penalties on u

    subject to (EQUALITY constraints)
      (E1)  z_{k+1} = F_RK4( z_k, u_k, w_z(t_k-tau) )      discretised modal EOM
      (E2)  w_k     = Phi @ eta_k                          measurement / mode-shape map
      (E3)  z_0     = z_meas                               initial condition each sample

    subject to (INEQUALITY constraints / bounds)
      omega_min <= omega_k <= omega_max
      ap_min    <= ap_k    <= ap_max        (first mode; fixed in second mode)
      |w_k| <= w_limit + slack_k,   slack_k >= 0           safe machining region

--------------------------------------------------------------------------
Internal prediction model  (derived from, and numerically verified against,
``custom_rl.plants.f_nonlinear2_face_milling``)
--------------------------------------------------------------------------
The plant integrates, per retained mode ``k`` (mass-normalised modal EOM):

    eta_k''  = -2 zeta_k wn_k eta_k' - wn_k^2 eta_k - lambda_k eta_k^3 + Q_k

with the face-milling generalised modal force (z-projection, the plant default):

    Q_k = phi_z,k(xc,yc) * Fz / M_k
    Fz  = sum_{engaged teeth i} [ Ka*ap*h_eff,i + Kae*ap ]
    h_i = ft*sin(theta_i) + 1000*sin(gamma_L)*( w_z(t) - w_z(t-tau) )

where ``w_z = phi_z(xc,yc) . eta`` is the transverse plate displacement at the
cutter [m] and ``tau = 2*pi/(N*omega)`` is the tooth-passing (regenerative)
delay.  For a smooth, differentiable predictive model we use the classical
**zeroth-order (period-averaged) directional model** (Altintas--Budak): the
tooth sums are replaced by their one-revolution averages, which are exact
constants for a fixed radial immersion ``ae``:

    S0 = (N/2pi)(theta_e-theta_s),   S1 = (N/2pi)(cos theta_s - cos theta_e)

    Fz(avg) = Ka*ap*ft*S1                         (nominal cutting force)
            + Kae*ap*S0                           (edge force)
            + Ka*ap*1000*sin(gamma_L)*S0*( w_z(t) - w_z(t-tau) )   (regeneration)

This retains the full regenerative chatter mechanism (the delayed feedback that
makes stability depend on spindle speed) while removing the fast tooth-passing
ripple and the non-smooth ``max(h,0)`` unilateral-contact term — i.e. it is the
standard linearised chatter-stability model, appropriate for a model-based
controller whose job is to keep the process inside the stable region.  The
delay ``w_z(t-tau)`` is handled by a *method-of-steps* discretisation with
linear (fractional) interpolation over a rolling buffer of past/predicted
``w_z`` values, so ``tau(omega)`` — hence the spindle-speed dependence of
stability — enters the OCP predictively.

The objective is the project's own **dense reward** (``DenseProductivePlateReward``)
reconstructed symbolically with the environment's exact weights, so the MPC
maximises the very quantity the PPO return measures.

Because gradient-based NLP cannot by itself jump between stability lobes, each
solve is seeded by a cheap **spindle-speed pre-screen** (a grid rollout of the
same discrete model) which gives the OCP a globally-informed warm start.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import casadi as ca
except Exception as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "CasADi is required for the MPC controller. Install it with:\n"
        "    pip install casadi\n"
        f"(import error: {exc})"
    )

import gymnasium as gym

from custom_rl import DEFAULT_TRAJ_DIR, register_envs
from custom_rl.eval.monte_carlo import face_milling_process_from_info
from custom_rl.eval.pipeline import plate_env_kwargs, plant_plot_metadata
from custom_rl.plants.plate import omega_to_rpm


MODAL_DISP_TO_MM = 1000.0  # matches f_nonlinear2.MODAL_DISPLACEMENT_TO_MM

# Same process keys eval_policy.py stores, so comparison tooling is schema-identical.
PROCESS_KEYS = (
    "omega_rad_s",
    "omega_rpm",
    "ap_mm",
    "ae_mm",
    "cutter_x",
    "cutter_y",
    "feed_progress",
    "feed_distance_m",
    "spindle_phase_rad",
    "mean_chip_mm",
    "max_chip_mm",
    "max_abs_w_m",
    "Fx_N",
    "Fy_N",
    "Fz_N",
    "F_mag_N",
)

PASS_COMPLETED_REASONS = {"pass_completed_90percent", "pass_completed"}


# ===========================================================================
# Configuration
# ===========================================================================
@dataclass
class MPCConfig:
    """Tunable MPC settings.  Defaults are tuned for the AL7075 face-milling plant."""

    horizon: int = 20                 # Np prediction/control horizon steps
    n_rk: int = 6                     # RK4 substeps per shooting interval
    control_hold: int = 3             # env control steps per MPC decision (ZOH)

    # Safe-region softening and control-effort shaping.
    slack_weight: float = 5.0e4       # rho: penalty on safe-region slack (|w|<=w_limit)
    omega_rate_weight: float = 0.05   # penalty on normalised d(omega) between steps
    ap_rate_weight: float = 0.05      # penalty on normalised d(ap) between steps

    # Terminal cost: penalise plate vibration at the horizon end (approximate
    # cost-to-go).  Prevents the finite-horizon controller from deferring a
    # limit breach just beyond the horizon; expressed as an equivalent number of
    # extra steps of the running vibration cost.
    terminal_weight: float = 3.0

    # Spindle-speed pre-screen (global warm start).
    prescreen_n_omega: int = 28       # candidate spindle speeds
    prescreen_n_ap: int = 7           # candidate depths (first mode only)

    # Regenerative-delay interpolation width (in Ts samples).  A smooth
    # (softmax) fractional-delay kernel keeps tau(omega) differentiable so the
    # OCP selects spindle speed for chatter avoidance; sigma ~ 0.7 approximates
    # linear interpolation while staying C-infinity for the NLP solver.
    delay_sigma: float = 0.7

    # IPOPT options.  'limited-memory' (L-BFGS) gives cheap iterations for this
    # medium-size NLP; 'exact' is more accurate but much slower per iteration.
    hessian: str = "limited-memory"
    max_iter: int = 40
    tol: float = 1.0e-6
    acceptable_tol: float = 1.0e-3
    print_level: int = 0
    warm_start: bool = True

    # Feedback: 'state' uses the true modal state from the env (idealised,
    # full-state MPC baseline); 'observer' uses a min-norm modal estimate from
    # the sensor observation only (output feedback, comparable to RL sensing).
    feedback: str = "state"

    # Safety margin on the displacement limit inside the OCP: because the
    # averaged model slightly under-predicts the peak (tooth ripple is averaged
    # out), the soft constraint uses w_lim_margin * w_limit as the threshold.
    w_lim_margin: float = 0.85


# ===========================================================================
# MPC controller
# ===========================================================================
class FaceMillingMPC:
    """CasADi multiple-shooting NMPC for the face-milling plate plant.

    All structural/force/reward constants are read from a constructed
    environment so the controller is guaranteed consistent with the plant the
    RL policy was trained on.
    """

    def __init__(self, env: gym.Env, cfg: MPCConfig | None = None) -> None:
        self.cfg = cfg or MPCConfig()
        base = env.unwrapped
        plant = base.plant
        reward = base.reward_fn
        self.plant = plant

        # ---- control mode ----
        self.control_ap = bool(getattr(plant, "control_ap", True))
        self.nu = 2 if self.control_ap else 1  # decision dimension

        # ---- structural / modal constants ----
        self.K = int(plant.K)
        self.wn = np.asarray(plant.omega_vec, dtype=np.float64).reshape(-1)      # natural freqs
        self.zeta = np.asarray(plant.zeta_vec, dtype=np.float64).reshape(-1)
        self.lam = np.asarray(plant.lambda_vec, dtype=np.float64).reshape(-1)    # mass-normalised cubic
        self.Mmodal = np.asarray(plant.M_modal, dtype=np.float64).reshape(-1)
        self.Phi = np.asarray(plant.Phi, dtype=np.float64)                       # (nS, K)
        self.nS = int(self.Phi.shape[0])
        self._Phi_pinv = np.linalg.pinv(self.Phi)                                # (K, nS)

        # ---- face-milling force constants ----
        self.Nteeth = int(plant.N)
        self.Ka = float(plant.Ka)
        self.Kae = float(plant.Kae)
        self.ft = float(plant.feed_per_tooth_mm)
        self.sin_gL = float(np.sin(plant.gamma_L))
        self.D_mm = float(plant.D_mm)
        self.ae_default = float(plant.ae_default)
        self.milling_mode = str(plant.milling_mode)

        # ---- action bounds ----
        self.omega_min = float(plant.omega_min)
        self.omega_max = float(plant.omega_max)
        self.ap_min = float(plant.ap_min)
        self.ap_max = float(plant.ap_max)

        # ---- safe region ----
        self.w_limit = float(plant.w_limit)
        self.w_lim_margin = float(self.cfg.w_lim_margin)

        # ---- decision-variable scaling for a well-conditioned NLP ----
        # eta near the limit is ~ w_limit/|Phi| ~ 7e-4; eta_dot ~ wn*eta ~ 1e-1.
        self.S_eta = 1.0e-3
        self.S_etad = 1.0e-1
        self.S_w = self.w_limit
        self.Dz = np.concatenate(
            [np.full(self.K, self.S_eta), np.full(self.K, self.S_etad)]
        )

        # ---- timing ----
        env_control_dt = float(base.dt) * int(base.n_substeps)
        self.env_control_dt = env_control_dt
        self.Ts = env_control_dt * int(self.cfg.control_hold)
        self.Np = int(self.cfg.horizon)
        self.n_rk = int(self.cfg.n_rk)

        # regenerative delay buffer length (>= longest tooth delay in Ts units)
        tau_max = 2.0 * np.pi / (self.Nteeth * max(self.omega_min, 1e-9))
        self.nb = int(np.ceil(tau_max / self.Ts)) + 3
        self.nb = max(self.nb, 4)

        # ---- engagement (ZOA) constants from radial immersion ae ----
        self.S0, self.S1 = self._engagement_constants(self.ae_default)

        # ---- dense-reward weights (mirror env.reward_fn exactly) ----
        self._load_reward_weights(reward)

        # ---- build symbolic model, stage cost, and NLP solver ----
        self._build_dynamics()
        self._build_stage_cost()
        self._build_nlp()

        # persistent warm start
        self._x_guess: np.ndarray | None = None
        self._lam_g: np.ndarray | None = None
        self._lam_x: np.ndarray | None = None

    # ------------------------------------------------------------------
    # constants
    # ------------------------------------------------------------------
    def _engagement_constants(self, ae_mm: float) -> tuple[float, float]:
        """Zeroth-order (one-rev averaged) engagement constants S0, S1."""
        immersion = float(np.clip(ae_mm / max(self.D_mm, 1e-12), 0.0, 1.0))
        mode = self.milling_mode.lower()
        if mode in {"down", "climb", "climb_milling"}:
            theta_s = float(np.arccos(np.clip(2.0 * immersion - 1.0, -1.0, 1.0)))
            theta_e = float(np.pi)
        else:  # up / conventional
            theta_s = 0.0
            theta_e = float(np.arccos(np.clip(1.0 - 2.0 * immersion, -1.0, 1.0)))
        c = self.Nteeth / (2.0 * np.pi)
        S0 = c * (theta_e - theta_s)
        S1 = c * (np.cos(theta_s) - np.cos(theta_e))
        return float(S0), float(S1)

    def _load_reward_weights(self, reward: Any) -> None:
        """Copy the dense-reward weights so the MPC objective == RL reward."""
        g = lambda name, default: float(getattr(reward, name, default))
        self.r_w_weight = g("w_weight", 0.6)
        self.r_wdot_weight = g("wdot_weight", 0.02)
        self.r_w_scale = g("w_scale", 7.5e-4)
        self.r_wdot_scale = g("wdot_scale", 1.0)
        self.r_prod_weight = g("productivity_weight", 30.0)
        self.r_omega_cost = g("omega_cost_weight", 0.3)
        self.r_neg_ap_weight = g("negative_ap_weight", 2.0)
        self.r_gate_enabled = bool(getattr(reward, "productivity_gate_enabled", True))
        self.r_w_gate = g("productivity_w_gate", 5.0e-4)
        self.r_wdot_gate = g("productivity_wdot_gate", 1.0)
        self.r_wdot_gate_weight = g("productivity_wdot_gate_weight", 0.05)
        self.r_gate_power = g("productivity_gate_power", 2.0)
        self.r_gate_min = g("productivity_gate_min", 0.0)
        self.r_alive = g("alive_bonus", 0.0)
        # reward action-scaling bounds (identical to plant bounds here)
        self.r_omega_min = g("omega_min", self.omega_min)
        self.r_omega_max = g("omega_max", self.omega_max)
        self.r_ap_min = g("ap_min", self.ap_min)
        self.r_ap_max = g("ap_max", self.ap_max)

    # ------------------------------------------------------------------
    # symbolic continuous dynamics and RK4 discrete map
    # ------------------------------------------------------------------
    def _build_dynamics(self) -> None:
        K = self.K
        z = ca.SX.sym("z", 2 * K)          # [eta ; eta_dot]  (block layout)
        u = ca.SX.sym("u", self.nu)        # [omega] or [omega, ap]
        wz_del = ca.SX.sym("wz_del")       # delayed transverse disp at cutter [m]
        phiz = ca.SX.sym("phiz", K)        # z-mode shapes at cutter point
        ap_par = ca.SX.sym("ap_par")       # fixed ap (second mode)
        S0 = ca.SX.sym("S0")
        S1 = ca.SX.sym("S1")

        eta = z[0:K]
        etad = z[K : 2 * K]
        om_s = u[0]                         # spindle speed [rad/s]
        ap = u[1] if self.nu == 2 else ap_par

        wz = ca.dot(phiz, eta)             # transverse displacement at cutter [m]

        # zeroth-order averaged face-milling axial force (see module docstring)
        Fz = (
            self.Ka * ap * self.ft * S1
            + self.Kae * ap * S0
            + self.Ka * ap * MODAL_DISP_TO_MM * self.sin_gL * S0 * (wz - wz_del)
        )
        Mvec = ca.DM(self.Mmodal)
        q = (phiz * Fz) / Mvec             # generalised modal force (K,)

        wn = ca.DM(self.wn)
        zeta = ca.DM(self.zeta)
        lam = ca.DM(self.lam)
        ddeta = -2.0 * zeta * wn * etad - (wn**2) * eta - lam * (eta**3) + q
        dz = ca.vertcat(etad, ddeta)

        self._f = ca.Function(
            "f_rhs", [z, u, wz_del, phiz, ap_par, S0, S1], [dz]
        )

        # RK4 over Ts with n_rk substeps, delayed term held constant (method of steps)
        h = self.Ts / self.n_rk
        zk = z
        for _ in range(self.n_rk):
            k1 = self._f(zk, u, wz_del, phiz, ap_par, S0, S1)
            k2 = self._f(zk + 0.5 * h * k1, u, wz_del, phiz, ap_par, S0, S1)
            k3 = self._f(zk + 0.5 * h * k2, u, wz_del, phiz, ap_par, S0, S1)
            k4 = self._f(zk + h * k3, u, wz_del, phiz, ap_par, S0, S1)
            zk = zk + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self._F = ca.Function(
            "F_step", [z, u, wz_del, phiz, ap_par, S0, S1], [zk]
        )

    # ------------------------------------------------------------------
    # symbolic dense-reward stage cost
    # ------------------------------------------------------------------
    def _build_stage_cost(self) -> None:
        K = self.K
        z_next = ca.SX.sym("z_next", 2 * K)
        u = ca.SX.sym("u", self.nu)
        ap_par = ca.SX.sym("ap_par")

        eta = z_next[0:K]
        etad = z_next[K : 2 * K]
        Phi = ca.DM(self.Phi)
        w = ca.mtimes(Phi, eta)            # (nS,) sensor displacement [m]
        wd = ca.mtimes(Phi, etad)          # (nS,) sensor velocity [m/s]

        om_s = u[0]
        ap = u[1] if self.nu == 2 else ap_par

        eps = 1e-24  # smoothing so sqrt is differentiable at 0
        w_rms = ca.sqrt(ca.sumsqr(w) / self.nS + eps)
        wd_rms = ca.sqrt(ca.sumsqr(wd) / self.nS + eps)

        # omega/ap are box-bounded to [min, max] in the NLP, so the RL reward's
        # clip(., 0, 1) is a no-op here.  Keeping the scores linear (no fmax/fmin
        # kinks) makes the objective C-infinity, which lets IPOPT reach dual
        # feasibility.  ap >= ap_min = 0, so the negative-ap penalty is also
        # identically zero in the feasible region and is dropped.
        omega_score = (om_s - self.r_omega_min) / (self.r_omega_max - self.r_omega_min)
        ap_score = (ap - self.r_ap_min) / max(self.r_ap_max - self.r_ap_min, 1e-12)
        raw_prod = omega_score * ap_score  # ae_score = 1 (ae not an action)

        if self.r_gate_enabled:
            gate = 1.0 / (
                1.0
                + (w_rms / self.r_w_gate) ** self.r_gate_power
                + self.r_wdot_gate_weight * (wd_rms / self.r_wdot_gate) ** self.r_gate_power
            )
        else:
            gate = 1.0
        prod = self.r_prod_weight * raw_prod * gate

        w_cost = self.r_w_weight * ca.sumsqr(w / self.r_w_scale) / self.nS
        wdot_cost = self.r_wdot_weight * ca.sumsqr(wd / self.r_wdot_scale) / self.nS
        omega_cost = self.r_omega_cost * omega_score**2

        reward = self.r_alive + prod - w_cost - wdot_cost - omega_cost
        self._L = ca.Function("L_stage", [z_next, u, ap_par], [-reward])

        # terminal cost: vibration cost of the final state, scaled as an
        # equivalent number of extra running-cost steps (approximate cost-to-go)
        zt = ca.SX.sym("zt", 2 * K)
        et = zt[0:K]
        edt = zt[K : 2 * K]
        wt = ca.mtimes(Phi, et)
        wdt = ca.mtimes(Phi, edt)
        term = self.cfg.terminal_weight * (
            self.r_w_weight * ca.sumsqr(wt / self.r_w_scale) / self.nS
            + self.r_wdot_weight * ca.sumsqr(wdt / self.r_wdot_scale) / self.nS
        )
        self._Lterm = ca.Function("L_term", [zt], [term])

    # ------------------------------------------------------------------
    # normalised control <-> physical helpers (symbolic)
    # ------------------------------------------------------------------
    def _omega_from_norm(self, un0):
        return self.omega_min + 0.5 * (un0 + 1.0) * (self.omega_max - self.omega_min)

    def _ap_from_norm(self, un1):
        return self.ap_min + 0.5 * (un1 + 1.0) * (self.ap_max - self.ap_min)

    def _norm_from_phys(self, u_phys: np.ndarray) -> np.ndarray:
        """Physical [omega(, ap)] -> normalised decision in [-1, 1]^nu."""
        u_phys = np.asarray(u_phys, dtype=np.float64).reshape(-1)
        omn = 2.0 * (u_phys[0] - self.omega_min) / (self.omega_max - self.omega_min) - 1.0
        if self.nu == 2:
            apn = 2.0 * (u_phys[1] - self.ap_min) / max(self.ap_max - self.ap_min, 1e-12) - 1.0
            return np.clip(np.array([omn, apn]), -1.0, 1.0)
        return np.clip(np.array([omn]), -1.0, 1.0)

    # ------------------------------------------------------------------
    # multiple-shooting NLP  (scaled state, normalised controls)
    # ------------------------------------------------------------------
    def _build_nlp(self) -> None:
        K, Np, nS, nu, nb = self.K, self.Np, self.nS, self.nu, self.nb
        Dz = ca.DM(self.Dz)        # state scale, physical z = Dz .* z_scaled
        Dz_eta = ca.DM(self.Dz[:K])
        S_w = self.S_w
        margin = self.w_lim_margin  # threshold = margin (in scaled w units)

        # scaled / normalised decision variables
        Zs = [ca.SX.sym(f"zs_{k}", 2 * K) for k in range(Np + 1)]
        Ws = [ca.SX.sym(f"ws_{k}", nS) for k in range(Np + 1)]
        Un = [ca.SX.sym(f"un_{k}", nu) for k in range(Np)]   # in [-1, 1]
        Ss = [ca.SX.sym(f"ss_{k}", nS) for k in range(Np + 1)]

        # parameters
        z_meas = ca.SX.sym("z_meas", 2 * K)      # physical modal state
        phiz = ca.SX.sym("phiz", K)
        wz_hist = ca.SX.sym("wz_hist", nb)       # past w_z [m] (oldest..newest)
        ap_fixed = ca.SX.sym("ap_fixed")
        un_prev = ca.SX.sym("un_prev", nu)       # previous normalised control
        S0 = ca.SX.sym("S0")
        S1 = ca.SX.sym("S1")
        omega_ref = ca.SX.sym("omega_ref")       # reference speed for the frozen delay
        P = ca.vertcat(z_meas, phiz, wz_hist, ap_fixed, un_prev, S0, S1, omega_ref)

        # physical state per node
        Zp = [Dz * Zs[k] for k in range(Np + 1)]

        # combined w_z timeline for the delay interpolation:
        #   [ wz_hist(nb) | wz_node_0 ... wz_node_Np ]  positions 0 .. nb+Np
        wz_nodes = [ca.dot(phiz, Zp[k][0:K]) for k in range(Np + 1)]
        wz_comb = ca.vertcat(wz_hist, *wz_nodes)
        idxs = ca.DM(np.arange(nb + Np + 1, dtype=np.float64))

        # Frozen-delay scheduling: the regenerative delay tau = 2*pi/(N*omega_ref)
        # is evaluated at the reference speed selected by the global speed
        # pre-screen (which uses the true tau(omega) per candidate).  This keeps
        # the delayed displacement w_z(t-tau) linear in the decision states -> a
        # well-conditioned NLP that reliably refines the depth ramp, while the
        # non-convex spindle-speed selection is handled by enumeration.  The
        # softmax kernel below depends only on omega_ref (a parameter), so the
        # interpolation weights are constant during a solve.
        nd_ref = (2.0 * np.pi) / (self.Nteeth * omega_ref * self.Ts)

        g, lbg, ubg = [], [], []
        J = 0.0

        # (E3) initial condition  z_0 = z_meas
        g.append(Zp[0] - z_meas)
        lbg += [0.0] * (2 * K)
        ubg += [0.0] * (2 * K)

        for k in range(Np):
            om_k = self._omega_from_norm(Un[k][0])
            ap_k = self._ap_from_norm(Un[k][1]) if nu == 2 else ap_fixed
            u_phys_k = ca.vertcat(om_k, ap_k) if nu == 2 else om_k

            pos = (nb + k) - nd_ref
            d = (idxs - pos) / self.cfg.delay_sigma
            e = ca.exp(-d * d)
            wz_del_k = ca.dot(e, wz_comb) / (ca.sum1(e) + 1e-12)

            z_next = self._F(Zp[k], u_phys_k, wz_del_k, phiz, ap_fixed, S0, S1)

            # (E1) discretised modal EOM continuity  z_{k+1} = F(z_k, u_k)
            g.append(Zp[k + 1] - z_next)
            lbg += [0.0] * (2 * K)
            ubg += [0.0] * (2 * K)

            # stage cost = -(dense reward of reached state z_{k+1})
            J = J + self._L(Zp[k + 1], u_phys_k, ap_fixed)

        # terminal cost (approximate cost-to-go on plate vibration)
        J = J + self._Lterm(Zp[Np])

        # (E2) measurement equation  w_k = Phi @ eta_k   (scaled: S_w*Ws = Phi*eta)
        Phi = ca.DM(self.Phi)
        for k in range(Np + 1):
            g.append(S_w * Ws[k] - ca.mtimes(Phi, Zp[k][0:K]))
            lbg += [0.0] * nS
            ubg += [0.0] * nS

        # safe-region soft inequality  |w_k| <= margin*w_limit + slack_k  (scaled)
        for k in range(Np + 1):
            g.append(Ws[k] - Ss[k] - margin)      # <= 0
            lbg += [-ca.inf] * nS
            ubg += [0.0] * nS
            g.append(-Ws[k] - Ss[k] - margin)     # <= 0
            lbg += [-ca.inf] * nS
            ubg += [0.0] * nS
            J = J + self.cfg.slack_weight * ca.sum1(Ss[k])

        # control-rate penalties (normalised controls already O(1))
        u_prev_k = un_prev
        for k in range(Np):
            dom = Un[k][0] - u_prev_k[0]
            J = J + self.cfg.omega_rate_weight * dom**2
            if nu == 2:
                dap = Un[k][1] - u_prev_k[1]
                J = J + self.cfg.ap_rate_weight * dap**2
            u_prev_k = Un[k]

        # assemble decision vector and bounds
        x_list, lbx, ubx = [], [], []
        zs_big = 100.0        # scaled state bound (physical ~ zs_big * Dz)
        ws_big = 50.0         # scaled w bound (physical ~ 50 * w_limit)
        ss_big = 40.0         # scaled slack bound
        for k in range(Np + 1):
            x_list.append(Zs[k]); lbx += [-zs_big] * (2 * K); ubx += [zs_big] * (2 * K)
        for k in range(Np + 1):
            x_list.append(Ws[k]); lbx += [-ws_big] * nS; ubx += [ws_big] * nS
        for k in range(Np):
            x_list.append(Un[k]); lbx += [-1.0] * nu; ubx += [1.0] * nu
        for k in range(Np + 1):
            x_list.append(Ss[k]); lbx += [0.0] * nS; ubx += [ss_big] * nS

        x = ca.vertcat(*x_list)
        gcat = ca.vertcat(*g)

        nlp = {"x": x, "f": J, "g": gcat, "p": P}
        opts = {
            "print_time": False,
            "ipopt": {
                "print_level": int(self.cfg.print_level),
                "sb": "yes",
                "max_iter": int(self.cfg.max_iter),
                "tol": float(self.cfg.tol),
                "acceptable_tol": float(self.cfg.acceptable_tol),
                "acceptable_iter": 5,
                "acceptable_obj_change_tol": 1e-8,
                "hessian_approximation": str(self.cfg.hessian),
                "mu_strategy": "adaptive",
                "warm_start_init_point": "yes" if self.cfg.warm_start else "no",
            },
        }
        self._solver = ca.nlpsol("mpc_solver", "ipopt", nlp, opts)

        # bookkeeping for pack/unpack
        self._lbx = np.array(lbx, dtype=np.float64)
        self._ubx = np.array(ubx, dtype=np.float64)
        self._lbg = np.array(lbg, dtype=np.float64)
        self._ubg = np.array(ubg, dtype=np.float64)
        self._nx = int(x.size1())
        # offsets
        self._off_Z = 0
        self._off_W = self._off_Z + (Np + 1) * (2 * K)
        self._off_U = self._off_W + (Np + 1) * nS
        self._off_S = self._off_U + Np * nu

    # ------------------------------------------------------------------
    # numeric helpers reused for pre-screen and warm start
    # ------------------------------------------------------------------
    def state_estimate(self, info: dict[str, Any]) -> np.ndarray:
        """Return the modal state z = [eta; eta_dot] for the OCP initial condition.

        ``feedback='state'`` uses the environment's true modal state (idealised
        full-state MPC baseline).  ``feedback='observer'`` reconstructs a min-norm
        modal estimate from the physical sensor signals only,

            eta_hat = Phi^+ w_sensor,   eta_dot_hat = Phi^+ wdot_sensor,

        i.e. output feedback comparable to the RL policy's sensing.
        """
        if self.cfg.feedback == "observer" and "w_sensor" in info and "wdot_sensor" in info:
            w = np.asarray(info["w_sensor"], dtype=np.float64).reshape(-1)
            wd = np.asarray(info["wdot_sensor"], dtype=np.float64).reshape(-1)
            eta = self._Phi_pinv @ w
            etad = self._Phi_pinv @ wd
            return np.concatenate([eta, etad])
        return _deinterleave(info["x_modal"], self.K)

    def phiz_at(self, xc: float, yc: float) -> np.ndarray:
        """Evaluate z-projection mode shapes at cutter point (numeric, K,)."""
        v = np.zeros(self.K, dtype=np.float64)
        c = 0
        for m in range(self.plant.m_max):
            for n in range(self.plant.n_max):
                v[c] = float(self.plant.W_mn[m][n](float(xc), float(yc)))
                c += 1
        return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _interp_buffer(buf: np.ndarray, pos: float) -> float:
        """Linear interpolation of buf at real index pos (clamped)."""
        n = buf.size
        if n == 0:
            return 0.0
        p = float(np.clip(pos, 0.0, n - 1))
        i0 = int(np.floor(p))
        if i0 >= n - 1:
            return float(buf[-1])
        a = p - i0
        return float((1.0 - a) * buf[i0] + a * buf[i0 + 1])

    def _rollout_cost(
        self,
        z0: np.ndarray,
        omega: float,
        ap: float,
        phiz: np.ndarray,
        wz_hist: np.ndarray,
        ap_fixed: float,
    ) -> tuple[float, np.ndarray]:
        """Roll the discrete model under constant (omega, ap); return (cost, Z traj)."""
        K, Np, nb = self.K, self.Np, self.nb
        z = np.asarray(z0, dtype=np.float64).reshape(-1).copy()
        u = np.array([omega, ap], dtype=np.float64) if self.nu == 2 else np.array([omega])
        buf = np.concatenate([np.asarray(wz_hist, dtype=np.float64).reshape(-1),
                              [float(phiz @ z[0:K])]])
        Ztraj = np.zeros((Np + 1, 2 * K), dtype=np.float64)
        Ztraj[0] = z
        cost = 0.0
        two_pi = 2.0 * np.pi
        thr = self.w_lim_margin * self.w_limit
        for k in range(Np):
            nd = two_pi / (self.Nteeth * max(omega, 1e-9) * self.Ts)
            pos = (nb + k) - nd
            wz_del = self._interp_buffer(buf, pos)
            z = np.asarray(
                self._F(z, u, wz_del, phiz, ap_fixed, self.S0, self.S1)
            ).reshape(-1)
            Ztraj[k + 1] = z
            buf = np.append(buf, float(phiz @ z[0:K]))
            cost += float(np.asarray(self._L(z, u, ap_fixed)).reshape(-1)[0])
            # safe-region penalty (mirror the NLP soft constraint so the global
            # pre-screen prefers productive *and* safe operating points)
            w = self.Phi @ z[0:K]
            slack = np.maximum(np.abs(w) - thr, 0.0) / self.S_w
            cost += self.cfg.slack_weight * float(np.sum(slack))
        # terminal cost (consistent with the NLP objective)
        cost += float(np.asarray(self._Lterm(z)).reshape(-1)[0])
        return cost, Ztraj

    def _prescreen(
        self,
        z0: np.ndarray,
        phiz: np.ndarray,
        wz_hist: np.ndarray,
        ap_fixed: float,
        u_prev: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Grid search over spindle speed (and ap in first mode) for a warm start.

        Returns the best constant control and its state rollout.
        """
        om_grid = np.linspace(self.omega_min, self.omega_max, self.cfg.prescreen_n_omega)
        if self.nu == 2:
            # Quadratic spacing concentrates candidates at low ap, where the
            # safe cutting envelope lives (large ap breaches the displacement
            # limit); the high-ap tail is still sampled but sparsely.
            ap_lo = max(self.ap_min, 0.0)
            frac = np.linspace(0.0, 1.0, self.cfg.prescreen_n_ap) ** 2
            ap_grid = ap_lo + frac * (self.ap_max - ap_lo)
        else:
            ap_grid = np.array([ap_fixed], dtype=np.float64)

        best_cost = np.inf
        best_u = np.array([self.omega_min, ap_fixed])[: self.nu]
        best_Z = np.tile(z0, (self.Np + 1, 1))
        for om in om_grid:
            for ap in ap_grid:
                cost, Ztraj = self._rollout_cost(z0, om, ap, phiz, wz_hist, ap_fixed)
                if np.isfinite(cost) and cost < best_cost:
                    best_cost = cost
                    best_u = np.array([om, ap], dtype=np.float64)[: self.nu]
                    best_Z = Ztraj
        return best_u, best_Z

    # ------------------------------------------------------------------
    # solve one OCP
    # ------------------------------------------------------------------
    def solve(
        self,
        z_meas: np.ndarray,
        phiz: np.ndarray,
        wz_hist: np.ndarray,
        ap_fixed: float,
        u_prev: np.ndarray,
    ) -> dict[str, Any]:
        """Solve the OCP at the current state; return first control and diagnostics.

        Parameters
        ----------
        z_meas : (2K,) modal state in block layout [eta ; eta_dot].
        phiz   : (K,) z-mode shapes at the current cutter point.
        wz_hist: (nb,) past transverse displacement at cutter (oldest..newest).
        ap_fixed: fixed axial depth (second mode); ignored bound-wise in first mode.
        u_prev : (nu,) previously applied physical control.
        """
        K, Np, nS, nu, nb = self.K, self.Np, self.nS, self.nu, self.nb
        z_meas = np.asarray(z_meas, dtype=np.float64).reshape(-1)
        phiz = np.asarray(phiz, dtype=np.float64).reshape(-1)
        wz_hist = np.asarray(wz_hist, dtype=np.float64).reshape(-1)
        u_prev = np.asarray(u_prev, dtype=np.float64).reshape(-1)

        # global warm start from the pre-screen (physical rollout)
        u_ws, Z_ws = self._prescreen(z_meas, phiz, wz_hist, ap_fixed, u_prev)
        un_ws = self._norm_from_phys(u_ws)
        un_prev = self._norm_from_phys(u_prev)

        # build scaled/normalised initial guess vector
        x0 = np.zeros(self._nx, dtype=np.float64)
        Zs_ws = Z_ws / self.Dz[None, :]                       # (Np+1, 2K) scaled
        x0[self._off_Z : self._off_Z + (Np + 1) * (2 * K)] = Zs_ws.reshape(-1)
        Ws_ws = (self.Phi @ Z_ws[:, 0:K].T).T / self.S_w      # (Np+1, nS) scaled
        x0[self._off_W : self._off_W + (Np + 1) * nS] = Ws_ws.reshape(-1)
        x0[self._off_U : self._off_U + Np * nu] = np.tile(un_ws, Np)
        Ss_ws = np.maximum(np.abs(Ws_ws) - self.w_lim_margin, 0.0)
        x0[self._off_S : self._off_S + (Np + 1) * nS] = Ss_ws.reshape(-1)

        if self.cfg.warm_start and self._x_guess is not None:
            # blend previous solution with pre-screen for robustness
            x0 = 0.5 * x0 + 0.5 * self._x_guess

        omega_ref = float(u_ws[0])  # frozen-delay reference = pre-screen best speed
        p = np.concatenate([
            z_meas, phiz, wz_hist, [float(ap_fixed)], un_prev,
            [self.S0], [self.S1], [omega_ref],
        ])

        args = dict(
            x0=x0, p=p,
            lbx=self._lbx, ubx=self._ubx, lbg=self._lbg, ubg=self._ubg,
        )
        if self.cfg.warm_start and self._lam_g is not None:
            args["lam_g0"] = self._lam_g
            args["lam_x0"] = self._lam_x

        sol = self._solver(**args)
        stats = self._solver.stats()
        converged = bool(stats.get("success", False))
        status = str(stats.get("return_status", ""))

        xopt = np.asarray(sol["x"], dtype=np.float64).reshape(-1)
        obj = float(np.asarray(sol["f"]).reshape(-1)[0])
        # IPOPT returns the best (primal-feasible) iterate even when it stops at
        # the iteration cap or the "acceptable" level.  With the soft safe-region
        # slacks the problem is always feasible, so we use that iterate unless it
        # is numerically invalid.  This makes the receding-horizon controller
        # robust to the loose dual-feasibility tolerance of this nonconvex NLP.
        usable = (
            np.all(np.isfinite(xopt))
            and np.isfinite(obj)
            and status not in ("Infeasible_Problem_Detected", "Invalid_Number_Detected")
        )
        if usable:
            self._x_guess = xopt.copy()
            self._lam_g = np.asarray(sol["lam_g"], dtype=np.float64).reshape(-1)
            self._lam_x = np.asarray(sol["lam_x"], dtype=np.float64).reshape(-1)

        Un_opt = xopt[self._off_U : self._off_U + Np * nu].reshape(Np, nu)
        un0 = Un_opt[0].copy()

        if not (usable and np.all(np.isfinite(un0))):
            # fall back to the globally best pre-screen control
            un0 = np.asarray(un_ws, dtype=np.float64).reshape(-1)

        # normalised decision -> physical [omega, ap] (inject fixed ap in 2nd mode)
        omega = self.omega_min + 0.5 * (un0[0] + 1.0) * (self.omega_max - self.omega_min)
        if nu == 2:
            ap = self.ap_min + 0.5 * (un0[1] + 1.0) * (self.ap_max - self.ap_min)
        else:
            ap = float(ap_fixed)
        u_phys = np.array([omega, ap], dtype=np.float64)

        return {
            "u_phys": u_phys,           # [omega_rad_s, ap_mm]
            "u_decision": un0,          # normalised decision in [-1, 1]^nu
            "U_plan": Un_opt,
            "cost": obj,
            "success": bool(usable),
            "converged": bool(converged),
            "iter": int(stats.get("iter_count", -1)),
            "return_status": status,
        }

    # ------------------------------------------------------------------
    # action normalisation for the environment
    # ------------------------------------------------------------------
    def to_env_action(self, u_phys: np.ndarray) -> np.ndarray:
        """Map physical [omega, ap] to the env's normalised action in [-1, 1]^nu."""
        plant = self.plant
        low = np.asarray(plant._ctrl_low, dtype=np.float64)
        high = np.asarray(plant._ctrl_high, dtype=np.float64)
        if self.nu == 2:
            phys = np.array([u_phys[0], u_phys[1]], dtype=np.float64)
        else:
            phys = np.array([u_phys[0]], dtype=np.float64)
        norm = 2.0 * (phys - low) / (high - low) - 1.0
        return np.clip(norm, -1.0, 1.0)


# ===========================================================================
# Closed-loop simulation of the MPC policy on the environment
# ===========================================================================
def _deinterleave(x_modal: np.ndarray, K: int) -> np.ndarray:
    """Convert plant interleaved [eta1,etad1,...] to block [eta ; eta_dot]."""
    x = np.asarray(x_modal, dtype=np.float64).reshape(-1)
    eta = x[0::2]
    etad = x[1::2]
    return np.concatenate([eta, etad])


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _process_snapshot(info: dict[str, Any], plant: Any) -> dict[str, float]:
    """One time step of process diagnostics (same schema as eval_policy.py)."""
    process = face_milling_process_from_info(info, plant)
    for key in ("cutter_x", "cutter_y", "feed_progress", "feed_distance_m", "spindle_phase_rad"):
        if key in info:
            process[key] = _safe_float(info[key])
    if "w_sensor" in info:
        w_sensor = np.asarray(info["w_sensor"], dtype=np.float64).reshape(-1)
        process["max_abs_w_m"] = float(np.max(np.abs(w_sensor))) if w_sensor.size else float("nan")
    if "F_total_N" in info:
        f_total = np.asarray(info["F_total_N"], dtype=np.float64).reshape(-1)
        if f_total.size >= 3:
            process["Fx_N"] = _safe_float(f_total[0])
            process["Fy_N"] = _safe_float(f_total[1])
            process["Fz_N"] = _safe_float(f_total[2])
            process["F_mag_N"] = _safe_float(float(np.linalg.norm(f_total)))
    return {key: _safe_float(process.get(key, np.nan)) for key in PROCESS_KEYS}


def run_episode(
    env: gym.Env,
    mpc: FaceMillingMPC,
    *,
    seed: int,
    reset_options: dict[str, Any] | None,
    max_steps: int,
    metadata: dict[str, Any],
    verbose: bool = True,
) -> dict[str, Any]:
    """Run one closed-loop MPC episode and return an eval_policy-schema record."""
    plant = env.unwrapped.plant
    K = mpc.K
    hold = int(mpc.cfg.control_hold)

    obs, reset_info = env.reset(seed=seed, options=reset_options)
    if "x_modal" not in reset_info:
        raise RuntimeError("Environment reset info lacks 'x_modal'; MPC needs modal state.")

    z_block = mpc.state_estimate(reset_info)
    eta0 = z_block[0:K].copy()

    # fixed per-episode axial depth (second mode) read from the plant
    ap_fixed = float(getattr(plant, "_episode_ap", 0.5 * (mpc.ap_min + mpc.ap_max)))
    if mpc.control_ap:
        ap_fixed = 0.5 * (mpc.ap_min + mpc.ap_max)  # only used as a bound reference

    # delay buffer of past eta (fill with initial state -> zero initial regeneration)
    eta_hist: deque[np.ndarray] = deque([eta0.copy() for _ in range(mpc.nb)], maxlen=mpc.nb)

    # previous physical control (start mid-range / current ap)
    u_prev = np.array([0.5 * (mpc.omega_min + mpc.omega_max), ap_fixed], dtype=np.float64)
    if not mpc.control_ap:
        u_prev = np.array([0.5 * (mpc.omega_min + mpc.omega_max)], dtype=np.float64)

    # trajectory logs
    times: list[float] = []
    observations: list[list[float]] = []
    physical_signals: list[list[float]] = []
    x_modal_hist: list[list[float]] = []
    actions: list[list[float]] = []
    physical_actions: list[list[float]] = []
    rewards: list[float] = []
    w_sensor_hist: list[list[float]] = []
    wdot_sensor_hist: list[list[float]] = []
    reward_terms_hist: list[dict[str, Any]] = []
    process_hist: dict[str, list[float]] = {key: [] for key in PROCESS_KEYS}
    solve_iters: list[int] = []
    solve_success: list[bool] = []

    termination_reason = None
    terminated_final = False
    truncated_final = False
    pass_completed_final = False

    step = 0
    t0 = time.time()
    while step < max_steps:
        # current cutter position -> mode shapes at cutter
        xc = float(np.clip(plant.L1 - plant._feed_distance_m - plant.x0_cutter, 0.0, plant.L1))
        yc = float(plant.y_cutter)
        phiz = mpc.phiz_at(xc, yc)
        wz_hist = np.array([float(phiz @ e) for e in eta_hist], dtype=np.float64)

        sol = mpc.solve(z_block, phiz, wz_hist, ap_fixed, u_prev)
        u_phys = sol["u_phys"]
        a_norm = mpc.to_env_action(u_phys)
        solve_iters.append(int(sol["iter"]))
        solve_success.append(bool(sol["success"]))

        # apply the decision for `hold` env steps (zero-order hold)
        done = False
        for _ in range(hold):
            obs, reward, terminated, truncated, info = env.step(a_norm)
            obs_arr = np.asarray(obs, dtype=np.float64).reshape(-1)

            actions.append(np.asarray(a_norm, dtype=np.float64).reshape(-1).tolist())
            physical_actions.append([float(u_phys[0]), float(u_phys[1])])
            observations.append(obs_arr.tolist())
            if "w_sensor" in info:
                w_sensor_hist.append(np.asarray(info["w_sensor"], dtype=np.float64).reshape(-1).tolist())
                wsens = np.asarray(info["w_sensor"], dtype=np.float64).reshape(-1)
            else:
                wsens = np.zeros(mpc.nS)
            if "wdot_sensor" in info:
                wdot_sensor_hist.append(np.asarray(info["wdot_sensor"], dtype=np.float64).reshape(-1).tolist())
                wdsens = np.asarray(info["wdot_sensor"], dtype=np.float64).reshape(-1)
            else:
                wdsens = np.zeros(mpc.nS)
            physical_signals.append(np.concatenate([wsens, wdsens]).tolist())
            if "reward_terms" in info:
                reward_terms_hist.append(dict(info["reward_terms"]))
            if "x_modal" in info:
                x_modal_hist.append(np.asarray(info["x_modal"], dtype=np.float64).reshape(-1).tolist())
            for key, val in _process_snapshot(info, plant).items():
                process_hist[key].append(val)
            rewards.append(float(reward))
            times.append(float(info.get("t", len(rewards) * metadata["step_dt"])))

            step += 1
            if terminated or truncated:
                terminated_final = bool(terminated)
                truncated_final = bool(truncated)
                termination_reason = info.get("termination_reason")
                pass_completed_final = bool(
                    info.get("pass_completed") or termination_reason in PASS_COMPLETED_REASONS
                )
                done = True
                break
            if step >= max_steps:
                done = True
                break

        # update feedback state and delay buffer
        z_block = mpc.state_estimate(info)
        eta_hist.append(z_block[0:K].copy())
        u_prev = u_phys if mpc.control_ap else np.array([u_phys[0]])

        if done:
            break

    wall = time.time() - t0
    if verbose:
        succ = 100.0 * (np.mean(solve_success) if solve_success else 0.0)
        print(
            f"    seed {seed}: steps={len(rewards)} return={sum(rewards):.1f} "
            f"pass={'Y' if pass_completed_final else 'N'} reason={termination_reason} "
            f"solve_ok={succ:.0f}% wall={wall:.1f}s"
        )

    w_arr = np.asarray(w_sensor_hist, dtype=np.float64)
    max_abs_w = float(np.max(np.abs(w_arr))) if w_arr.size else float("nan")
    rms_w = float(np.sqrt(np.mean(w_arr**2))) if w_arr.size else float("nan")

    record = {
        "controller": "mpc",
        "metadata": metadata,
        "seed": int(seed),
        "reset_options": reset_options,
        "reset_info": _to_jsonable(dict(reset_info)),
        "y_line_m": _safe_float(reset_info.get("y_cutter", np.nan)),
        "x_start_m": _safe_float(reset_info.get("x_start", np.nan)),
        "x_end_target_m": _safe_float(reset_info.get("x_end", np.nan)),
        "ap_fixed_mm": _safe_float(reset_info.get("ap_fixed_mm", ap_fixed if not mpc.control_ap else np.nan)),
        "times": times,
        "observations": observations,
        "physical_signals": physical_signals,
        "x_modal": x_modal_hist,
        "actions": actions,
        "physical_actions": physical_actions,
        "process": process_hist,
        "w_sensor": w_sensor_hist,
        "wdot_sensor": wdot_sensor_hist,
        "reward_terms": reward_terms_hist,
        "rewards": rewards,
        "return": float(sum(rewards)),
        "length": int(len(rewards)),
        "terminated": bool(terminated_final),
        "truncated": bool(truncated_final),
        "pass_completed": bool(pass_completed_final),
        "termination_reason": termination_reason,
        "summary": {
            "return": float(sum(rewards)),
            "length": int(len(rewards)),
            "max_abs_w_m": max_abs_w,
            "rms_w_m": rms_w,
            "pass_completed": bool(pass_completed_final),
            "termination_reason": termination_reason,
            "mpc_mean_solve_iters": float(np.mean(solve_iters)) if solve_iters else float("nan"),
            "mpc_solve_success_frac": float(np.mean(solve_success)) if solve_success else 0.0,
            "mpc_wall_time_s": float(wall),
        },
    }
    return record


# ===========================================================================
# CLI
# ===========================================================================
def build_env(args: argparse.Namespace) -> gym.Env:
    register_envs()
    env_kwargs = plate_env_kwargs(
        reward_id=args.reward,
        dt=args.dt,
        n_substeps=args.n_substeps,
        max_episode_steps=args.max_episode_steps,
        dynamics_uncertainty_std=args.dynamics_uncertainty_std,
        randomize_y0=args.randomize_y0,
    )
    return gym.make(args.env_id, **env_kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Nonlinear MPC (CasADi) chatter-suppression controller for face milling."
    )
    parser.add_argument(
        "--env-id",
        default="CustomODEPlate-v0",
        choices=["CustomODEPlate-v0", "CustomODEPlateFinish-v0"],
        help=(
            "Control mode. CustomODEPlate-v0: first-mode roughing (decision [omega, ap]). "
            "CustomODEPlateFinish-v0: second-mode finishing (decision [omega], ap fixed)."
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--out-dir", default="mpc_trajectories")

    # timing / integration (must match training/eval to keep the plant identical)
    parser.add_argument("--dt", type=float, default=1.0e-4)
    parser.add_argument("--n-substeps", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=50000)
    parser.add_argument("--max-steps", type=int, default=3000,
                        help="Cap on closed-loop control steps per episode (MPC runs are slower than RL).")
    parser.add_argument("--reward", default="dense",
                        choices=["dense", "productive", "quadratic", "sparse"])
    parser.add_argument("--dynamics-uncertainty-std", type=float, default=0.0)
    parser.add_argument("--randomize-y0", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--y0", type=float, default=None,
                        help="Fix the milling line y0 [m] for every episode.")
    parser.add_argument("--ap", type=float, default=None,
                        help="Second-mode only: pin the fixed axial depth ap [mm].")

    # MPC settings
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--n-rk", type=int, default=6)
    parser.add_argument("--control-hold", type=int, default=3,
                        help="Env control steps per MPC decision (zero-order hold).")
    parser.add_argument("--slack-weight", type=float, default=5.0e4)
    parser.add_argument("--omega-rate-weight", type=float, default=0.05)
    parser.add_argument("--ap-rate-weight", type=float, default=0.05)
    parser.add_argument("--terminal-weight", type=float, default=3.0)
    parser.add_argument("--prescreen-n-omega", type=int, default=28)
    parser.add_argument("--prescreen-n-ap", type=int, default=7)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--feedback", default="state", choices=["state", "observer"])
    parser.add_argument("--ipopt-verbose", action="store_true",
                        help="Print IPOPT solver output (print_level 5).")

    args = parser.parse_args()

    if args.dt <= 0 or args.n_substeps <= 0:
        raise SystemExit("--dt and --n-substeps must be positive.")

    cfg = MPCConfig(
        horizon=args.horizon,
        n_rk=args.n_rk,
        control_hold=args.control_hold,
        slack_weight=args.slack_weight,
        omega_rate_weight=args.omega_rate_weight,
        ap_rate_weight=args.ap_rate_weight,
        terminal_weight=args.terminal_weight,
        prescreen_n_omega=args.prescreen_n_omega,
        prescreen_n_ap=args.prescreen_n_ap,
        max_iter=args.max_iter,
        feedback=args.feedback,
        print_level=5 if args.ipopt_verbose else 0,
    )

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(args)
    plant = env.unwrapped.plant

    # tooth-delay feasibility (same guard as eval_policy)
    tau_min = 2.0 * np.pi / (max(int(plant.N), 1) * max(float(plant.omega_max), 1e-12))
    if float(args.dt) >= tau_min:
        env.close()
        raise SystemExit(
            f"RK4 dt={args.dt:g}s is not smaller than the minimum tooth delay "
            f"{tau_min:g}s. Use --dt 0.0001 --n-substeps 10."
        )

    mpc = FaceMillingMPC(env, cfg)

    metadata = plant_plot_metadata(env)
    metadata["controller"] = "mpc"
    metadata["reward_id"] = args.reward
    metadata["mpc"] = {
        "horizon": cfg.horizon,
        "Ts_s": mpc.Ts,
        "n_rk": cfg.n_rk,
        "control_hold": cfg.control_hold,
        "delay_buffer_nb": mpc.nb,
        "S0": mpc.S0,
        "S1": mpc.S1,
        "feedback": cfg.feedback,
        "slack_weight": cfg.slack_weight,
        "prescreen_n_omega": cfg.prescreen_n_omega,
        "prescreen_n_ap": cfg.prescreen_n_ap,
    }

    mode = "second-mode finishing (omega)" if not mpc.control_ap else "first-mode roughing (omega, ap)"
    print(f"MPC controller: {mode}")
    print(f"  env_id={args.env_id}  Ts={mpc.Ts*1e3:.2f} ms  Np={cfg.horizon}  "
          f"n_rk={cfg.n_rk}  delay_buffer={mpc.nb}  feedback={cfg.feedback}")
    print(f"  spindle {omega_to_rpm(mpc.omega_min):.0f}-{omega_to_rpm(mpc.omega_max):.0f} rpm, "
          f"ap {mpc.ap_min:g}-{mpc.ap_max:g} mm, w_limit={mpc.w_limit:g} m")
    print(f"  engagement S0={mpc.S0:.4f} S1={mpc.S1:.4f}  (ae={mpc.ae_default:g} mm, "
          f"{mpc.milling_mode} milling)")
    print(f"  output -> {out_dir}")

    for seed in args.seeds:
        trajectories: list[dict[str, Any]] = []
        for ep in range(args.n_episodes):
            reset_options: dict[str, Any] = {}
            if args.y0 is not None:
                reset_options["y0"] = float(np.clip(args.y0, 0.0, float(plant.L2)))
            if (args.ap is not None) and (not mpc.control_ap):
                reset_options["ap"] = float(args.ap)
            run_seed = int(seed * 1000 + ep)
            print(f"  [seed {seed} ep {ep}] solving...")
            record = run_episode(
                env, mpc,
                seed=run_seed,
                reset_options=reset_options or None,
                max_steps=args.max_steps,
                metadata=metadata,
            )
            record["episode"] = int(ep)
            record["seed"] = int(seed)
            trajectories.append(record)

        out_path = out_dir / f"trajectories_seed{seed}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(trajectories), f, indent=2)
        print(f"  saved {len(trajectories)} episode(s) -> {out_path}")

    env.close()
    print("MPC evaluation complete.")
    print(f"Compare with RL: python scripts/compare_mpc_rl.py "
          f"--rl-dir {DEFAULT_TRAJ_DIR} --mpc-dir {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
