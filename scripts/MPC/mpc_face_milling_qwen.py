"""Nonlinear Model Predictive Control for face-milling chatter suppression."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import casadi as ca
except Exception as exc:
    raise SystemExit(f"CasADi is required. Install it with: pip install casadi\n(import error: {exc})")

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

@dataclass
class MPCConfig:
    horizon: int = 32
    n_rk: int = 6
    control_hold: int = 3
    slack_weight: float = 5.0e4
    omega_rate_weight: float = 5.0
    ap_rate_weight: float = 5.0
    terminal_weight: float = 3.0
    omega_slew: float = 600.0
    ap_up_slew: float = 0.25
    ap_down_slew: float = 1.5
    slew_frac_omega: float = 0.20
    slew_frac_ap_up: float = 0.15
    slew_frac_ap_down: float = 1.0
    wdot_limit: float | None = None
    fz_limit: float | None = None
    prescreen_n_omega: int = 28
    prescreen_n_ap: int = 7
    delay_sigma: float = 0.7
    delay_mode: str = "predictive"
    hessian: str = "limited-memory"
    max_iter: int = 40
    tol: float = 1.0e-6
    acceptable_tol: float = 1.0e-3
    primal_feas_tol: float = 1.0e-5
    print_level: int = 0
    warm_start: bool = True
    w_lim_margin: float = 0.8

class FaceMillingMPC:
    def __init__(self, env: gym.Env, cfg: MPCConfig | None = None) -> None:
        self.cfg = cfg or MPCConfig()
        base = env.unwrapped
        plant = base.plant
        reward = base.reward_fn
        self.plant = plant
        self.base_env = base

        if self.cfg.horizon <= 0 or self.cfg.n_rk <= 0 or self.cfg.control_hold <= 0:
            raise ValueError("horizon, n_rk and control_hold must be positive")
        if not (0.0 < float(self.cfg.w_lim_margin) < 1.0):
            raise ValueError("w_lim_margin must lie strictly between 0 and 1")
        if self.cfg.delay_sigma <= 0.0: raise ValueError("delay_sigma must be positive")
        if self.cfg.delay_mode not in {"predictive", "frozen"}: raise ValueError("delay_mode must be 'predictive' or 'frozen'")
        if self.cfg.primal_feas_tol <= 0.0: raise ValueError("primal_feas_tol must be positive")

        self.control_ap = bool(getattr(plant, "control_ap", True))
        self.nu = 2 if self.control_ap else 1

        self.K = int(plant.K)
        self.wn = np.asarray(plant.omega_vec, dtype=np.float64).reshape(-1)
        self.zeta = np.asarray(plant.zeta_vec, dtype=np.float64).reshape(-1)
        self.lam = np.asarray(plant.lambda_vec, dtype=np.float64).reshape(-1)
        self.Mmodal = np.asarray(plant.M_modal, dtype=np.float64).reshape(-1)
        self.Phi = np.asarray(plant.Phi, dtype=np.float64)
        self.nS = int(self.Phi.shape[0])
        self._Phi_pinv = np.linalg.pinv(self.Phi)

        self.Nteeth = int(plant.N)
        self.Ka = float(plant.Ka)
        self.Kae = float(plant.Kae)
        self.ft = float(plant.feed_per_tooth_mm)
        self.sin_gL = float(np.sin(plant.gamma_L))
        self.D_mm = float(plant.D_mm)
        self.ae_default = float(plant.ae_default)
        self.milling_mode = str(plant.milling_mode)

        self.omega_min = float(plant.omega_min)
        self.omega_max = float(plant.omega_max)
        self.ap_min = float(plant.ap_min)
        self.ap_max = float(plant.ap_max)
        if not self.omega_max > self.omega_min: raise ValueError("Invalid spindle-speed bounds")
        if self.control_ap and not self.ap_max > self.ap_min: raise ValueError("Invalid axial-depth bounds")

        om_range = self.omega_max - self.omega_min
        ap_range = max(self.ap_max - self.ap_min, 1.0e-12)
        self.cfg.omega_slew = min(max(float(self.cfg.omega_slew), 0.0), max(float(self.cfg.slew_frac_omega), 0.0) * om_range)
        self.cfg.ap_up_slew = min(max(float(self.cfg.ap_up_slew), 0.0), max(float(self.cfg.slew_frac_ap_up), 0.0) * ap_range)
        self.cfg.ap_down_slew = min(max(float(self.cfg.ap_down_slew), 0.0), max(float(self.cfg.slew_frac_ap_down), 0.0) * ap_range)
        self.du_omega_norm = 2.0 * self.cfg.omega_slew / om_range
        self.du_ap_up_norm = 2.0 * self.cfg.ap_up_slew / ap_range
        self.du_ap_down_norm = 2.0 * self.cfg.ap_down_slew / ap_range

        self.w_limit = float(plant.w_limit)
        if not np.isfinite(self.w_limit) or self.w_limit <= 0.0: raise ValueError("plant.w_limit must be finite and positive")
        self.w_lim_margin = float(self.cfg.w_lim_margin)

        def first_finite_attr(objects: tuple[Any, ...], names: tuple[str, ...]) -> float | None:
            for obj in objects:
                for name in names:
                    val = getattr(obj, name, None)
                    try: val_f = float(val)
                    except (TypeError, ValueError): continue
                    if np.isfinite(val_f) and val_f > 0.0: return val_f
            return None

        self.wdot_limit = float(self.cfg.wdot_limit) if self.cfg.wdot_limit is not None else first_finite_attr((base, plant), ("wdot_limit", "w_dot_limit", "sensor_velocity_limit", "wdot_sensor_limit"))
        if self.wdot_limit is not None and self.wdot_limit <= 0.0: raise ValueError("wdot_limit must be positive when supplied")
        
        self.fz_limit = float(self.cfg.fz_limit) if self.cfg.fz_limit is not None else first_finite_attr((base, plant), ("Fz_limit", "fz_limit", "axial_force_limit", "force_z_limit"))
        if self.fz_limit is not None and self.fz_limit <= 0.0: raise ValueError("fz_limit must be positive when supplied")

        self.S_eta = 1.0e-3
        self.S_etad = 1.0e-1
        self.S_w = self.w_limit
        self.Dz = np.concatenate([np.full(self.K, self.S_eta), np.full(self.K, self.S_etad)])

        self.env_control_dt = float(base.dt) * int(base.n_substeps)
        if self.env_control_dt <= 0.0: raise ValueError("Environment control time step must be positive")
        self.Ts = self.env_control_dt * int(self.cfg.control_hold)
        self.Np = int(self.cfg.horizon)
        self.n_rk = int(self.cfg.n_rk)

        tau_max = 2.0 * np.pi / (self.Nteeth * max(self.omega_min, 1.0e-12))
        self.nb = max(int(np.ceil(tau_max / self.env_control_dt)) + 3, 4)
        self.nb = min(self.nb, 1000)  # HARD CAP TO PREVENT MEMORY EXPLOSION
        
        self.S0, self.S1 = self._engagement_constants(self.ae_default)
        self._load_reward_weights(reward)
        self.reward_class = type(reward).__name__
        
        self._build_dynamics()
        self._build_stage_cost()
        self._build_nlp()

        self._U_last: np.ndarray | None = None
        self._n_screen = int(self.cfg.prescreen_n_omega) * (int(self.cfg.prescreen_n_ap) if self.nu == 2 else 1)
        if self._n_screen <= 0: raise ValueError("Pre-screen grid dimensions must be positive")
        
        # REMOVED .map() CALLS HERE TO PREVENT CASADI COMPILATION FREEZE

    def reset(self) -> None:
        self._U_last = None

    def _engagement_constants(self, ae_mm: float) -> tuple[float, float]:
        immersion = float(np.clip(ae_mm / max(self.D_mm, 1e-12), 0.0, 1.0))
        mode = self.milling_mode.lower()
        if mode in {"down", "climb", "climb_milling"}:
            theta_s = float(np.arccos(np.clip(2.0 * immersion - 1.0, -1.0, 1.0)))
            theta_e = float(np.pi)
        else:
            theta_s = 0.0
            theta_e = float(np.arccos(np.clip(1.0 - 2.0 * immersion, -1.0, 1.0)))
        c = self.Nteeth / (2.0 * np.pi)
        S0 = c * (theta_e - theta_s)
        S1 = c * (np.cos(theta_s) - np.cos(theta_e))
        return float(S0), float(S1)

    def _load_reward_weights(self, reward: Any) -> None:
        g = lambda name, default: float(getattr(reward, name, default))
        self.r_w_weight = g("w_weight", 0.6)
        self.r_wdot_weight = g("wdot_weight", 0.02)
        self.r_w_scale = g("w_scale", 7.5e-4)
        self.r_wdot_scale = g("wdot_scale", 1.0)
        self.r_prod_weight = g("productivity_weight", 30.0)
        self.r_omega_cost = g("omega_cost_weight", 3.0)
        self.r_neg_ap_weight = g("negative_ap_weight", 2.0)
        self.r_gate_enabled = bool(getattr(reward, "productivity_gate_enabled", True))
        self.r_w_gate = g("productivity_w_gate", 5.0e-4)
        self.r_wdot_gate = g("productivity_wdot_gate", 1.0)
        self.r_wdot_gate_weight = g("productivity_wdot_gate_weight", 0.05)
        self.r_gate_power = g("productivity_gate_power", 2.0)
        self.r_gate_min = g("productivity_gate_min", 0.0)
        self.r_alive = g("alive_bonus", 0.0)
        self.r_omega_min = g("omega_min", self.omega_min)
        self.r_omega_max = g("omega_max", self.omega_max)
        self.r_ap_min = g("ap_min", self.ap_min)
        self.r_ap_max = g("ap_max", self.ap_max)

    def _build_dynamics(self) -> None:
        K = self.K
        z = ca.SX.sym("z", 2 * K)
        u = ca.SX.sym("u", self.nu)
        wz_del = ca.SX.sym("wz_del")
        phiz = ca.SX.sym("phiz", K)
        ap_par = ca.SX.sym("ap_par")
        S0 = ca.SX.sym("S0")
        S1 = ca.SX.sym("S1")
        eta = z[0:K]
        etad = z[K : 2 * K]
        ap = u[1] if self.nu == 2 else ap_par
        wz = ca.dot(phiz, eta)
        Fz = self.Ka * ap * self.ft * S1 + self.Kae * ap * S0 + self.Ka * ap * MODAL_DISP_TO_MM * self.sin_gL * S0 * (wz - wz_del)
        q = (phiz * Fz) / ca.DM(self.Mmodal)
        wn = ca.DM(self.wn)
        zeta = ca.DM(self.zeta)
        lam = ca.DM(self.lam)
        ddeta = -2.0 * zeta * wn * etad - (wn**2) * eta - lam * (eta**3) + q
        dz = ca.vertcat(etad, ddeta)
        self._f = ca.Function("f_rhs", [z, u, wz_del, phiz, ap_par, S0, S1], [dz])
        self._Fz = ca.Function("Fz_avg", [z, u, wz_del, phiz, ap_par, S0, S1], [Fz])

        h = self.env_control_dt / self.n_rk
        zk = z
        for _ in range(self.n_rk):
            k1 = self._f(zk, u, wz_del, phiz, ap_par, S0, S1)
            k2 = self._f(zk + 0.5 * h * k1, u, wz_del, phiz, ap_par, S0, S1)
            k3 = self._f(zk + 0.5 * h * k2, u, wz_del, phiz, ap_par, S0, S1)
            k4 = self._f(zk + h * k3, u, wz_del, phiz, ap_par, S0, S1)
            zk = zk + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self._F = ca.Function("F_env_step", [z, u, wz_del, phiz, ap_par, S0, S1], [zk])

    def _build_stage_cost(self) -> None:
        K = self.K
        z_next = ca.SX.sym("z_next", 2 * K)
        u = ca.SX.sym("u", self.nu)
        ap_par = ca.SX.sym("ap_par")
        eta = z_next[0:K]
        etad = z_next[K : 2 * K]
        Phi = ca.DM(self.Phi)
        w = ca.mtimes(Phi, eta)
        wd = ca.mtimes(Phi, etad)
        om_s = u[0]
        ap = u[1] if self.nu == 2 else ap_par
        eps = 1.0e-8  # Increased eps for better Hessian approximation stability
        w_rms = ca.sqrt(ca.sumsqr(w) / self.nS + eps)
        wd_rms = ca.sqrt(ca.sumsqr(wd) / self.nS + eps)
        omega_score = (om_s - self.r_omega_min) / max(self.r_omega_max - self.r_omega_min, 1.0e-12)
        ap_score = (ap - self.r_ap_min) / max(self.r_ap_max - self.r_ap_min, 1.0e-12)
        raw_prod = omega_score * ap_score
        if self.r_gate_enabled:
            gate = 1.0 / (1.0 + (w_rms / self.r_w_gate) ** self.r_gate_power + self.r_wdot_gate_weight * (wd_rms / self.r_wdot_gate) ** self.r_gate_power)
            if self.r_gate_min > 0.0:
                delta = gate - self.r_gate_min
                gate = 0.5 * (gate + self.r_gate_min + ca.sqrt(delta * delta + 1.0e-12))
        else:
            gate = 1.0
        prod = self.r_prod_weight * raw_prod * gate
        w_cost = self.r_w_weight * ca.sumsqr(w / self.r_w_scale) / self.nS
        wdot_cost = self.r_wdot_weight * ca.sumsqr(wd / self.r_wdot_scale) / self.nS
        omega_cost = self.r_omega_cost * omega_score**2
        reward = self.r_alive + prod - w_cost - wdot_cost - omega_cost
        self._L = ca.Function("L_stage", [z_next, u, ap_par], [-reward])
        zt = ca.SX.sym("zt", 2 * K)
        wt = ca.mtimes(Phi, zt[0:K])
        wdt = ca.mtimes(Phi, zt[K : 2 * K])
        term = self.cfg.terminal_weight * (self.r_w_weight * ca.sumsqr(wt / self.r_w_scale) / self.nS + self.r_wdot_weight * ca.sumsqr(wdt / self.r_wdot_scale) / self.nS)
        self._Lterm = ca.Function("L_term", [zt], [term])

    def _omega_from_norm(self, un0): return self.omega_min + 0.5 * (un0 + 1.0) * (self.omega_max - self.omega_min)
    def _ap_from_norm(self, un1): return self.ap_min + 0.5 * (un1 + 1.0) * (self.ap_max - self.ap_min)
    def _norm_from_phys(self, u_phys: np.ndarray) -> np.ndarray:
        u_phys = np.asarray(u_phys, dtype=np.float64).reshape(-1)
        omn = 2.0 * (u_phys[0] - self.omega_min) / (self.omega_max - self.omega_min) - 1.0
        if self.nu == 2:
            apn = 2.0 * (u_phys[1] - self.ap_min) / max(self.ap_max - self.ap_min, 1e-12) - 1.0
            return np.clip(np.array([omn, apn]), -1.0, 1.0)
        return np.clip(np.array([omn]), -1.0, 1.0)

    def _build_nlp(self) -> None:
        K, Np, nS, nu, nb = self.K, self.Np, self.nS, self.nu, self.nb
        Dz = ca.DM(self.Dz)
        S_w = self.S_w
        margin = self.w_lim_margin
        Phi = ca.DM(self.Phi)
        Zs = [ca.SX.sym(f"zs_{k}", 2 * K) for k in range(Np + 1)]
        Ws = [ca.SX.sym(f"ws_{k}", nS) for k in range(Np + 1)]
        Un = [ca.SX.sym(f"un_{k}", nu) for k in range(Np)]
        Ss = [ca.SX.sym(f"ss_{k}", nS) for k in range(Np + 1)]
        z_meas = ca.SX.sym("z_meas", 2 * K)
        phiz = ca.SX.sym("phiz", K)
        wz_hist = ca.SX.sym("wz_hist", nb)
        ap_fixed = ca.SX.sym("ap_fixed")
        un_prev = ca.SX.sym("un_prev", nu)
        S0 = ca.SX.sym("S0")
        S1 = ca.SX.sym("S1")
        omega_ref = ca.SX.sym("omega_ref")
        P = ca.vertcat(z_meas, phiz, wz_hist, ap_fixed, un_prev, S0, S1, omega_ref)
        Zp = [Dz * Zs[k] for k in range(Np + 1)]
        g: list[Any] = []
        lbg: list[float] = []
        ubg: list[float] = []
        J = 0.0

        def add_constraint(expr: Any, lower: float | list[float], upper: float | list[float]) -> None:
            g.append(expr)
            n_expr = int(expr.numel())
            if np.isscalar(lower): lbg.extend([float(lower)] * n_expr)
            else: lbg.extend([float(v) for v in lower])
            if np.isscalar(upper): ubg.extend([float(upper)] * n_expr)
            else: ubg.extend([float(v) for v in upper])

        add_constraint(Zs[0] - z_meas / Dz, 0.0, 0.0)
        wz_timeline: list[Any] = [wz_hist[i] for i in range(nb - 1)]
        wz_timeline.append(ca.dot(phiz, Zp[0][0:K]))
        nd_ref = 2.0 * np.pi / (self.Nteeth * omega_ref * self.env_control_dt)

        for k in range(Np):
            om_k = self._omega_from_norm(Un[k][0])
            ap_k = self._ap_from_norm(Un[k][1]) if nu == 2 else ap_fixed
            u_phys_k = ca.vertcat(om_k, ap_k) if nu == 2 else ca.vertcat(om_k)
            z_micro = Zp[k]
            for _ in range(int(self.cfg.control_hold)):
                current_idx = len(wz_timeline) - 1
                if self.cfg.delay_mode == "predictive":
                    nd_k = 2.0 * np.pi / (self.Nteeth * om_k * self.env_control_dt)
                else:
                    nd_k = nd_ref
                pos = current_idx - nd_k
                
                # LIMIT SYMBOLIC GRAPH SIZE FOR GAUSSIAN INTERPOLATION
                M = 10 
                start_idx = max(0, current_idx + 1 - M)
                recent_timeline = wz_timeline[start_idx:]
                recent_idxs = np.arange(start_idx, current_idx + 1, dtype=np.float64)
                
                idxs = ca.DM(recent_idxs)
                timeline_vec = ca.vertcat(*recent_timeline)
                d = (idxs - pos) / self.cfg.delay_sigma
                weights = ca.exp(-(d * d))
                wz_del = ca.dot(weights, timeline_vec) / (ca.sum1(weights) + 1.0e-12)

                if self.fz_limit is not None:
                    fz = self._Fz(z_micro, u_phys_k, wz_del, phiz, ap_fixed, S0, S1)
                    add_constraint(fz / self.fz_limit, -1.0, 1.0)
                z_micro = self._F(z_micro, u_phys_k, wz_del, phiz, ap_fixed, S0, S1)
                J = J + self._L(z_micro, u_phys_k, ap_fixed)
                w_micro_scaled = ca.mtimes(Phi, z_micro[0:K]) / S_w
                add_constraint(w_micro_scaled, -1.0, 1.0)
                if self.wdot_limit is not None:
                    wd_micro_scaled = ca.mtimes(Phi, z_micro[K : 2 * K]) / self.wdot_limit
                    add_constraint(wd_micro_scaled, -1.0, 1.0)
                wz_timeline.append(ca.dot(phiz, z_micro[0:K]))
            add_constraint((Zp[k + 1] - z_micro) / Dz, 0.0, 0.0)

        J = J + self._Lterm(Zp[Np])
        for k in range(Np + 1):
            add_constraint(Ws[k] - ca.mtimes(Phi, Zp[k][0:K]) / S_w, 0.0, 0.0)
            add_constraint(Ws[k] - Ss[k] - margin, -ca.inf, 0.0)
            add_constraint(-Ws[k] - Ss[k] - margin, -ca.inf, 0.0)
            J = J + self.cfg.slack_weight * ca.sum1(Ss[k])
            if self.wdot_limit is not None:
                wd_scaled = ca.mtimes(Phi, Zp[k][K : 2 * K]) / self.wdot_limit
                add_constraint(wd_scaled, -1.0, 1.0)

        u_prev_k = un_prev
        for k in range(Np):
            dom = Un[k][0] - u_prev_k[0]
            add_constraint(dom, -self.du_omega_norm, self.du_omega_norm)
            J = J + self.cfg.omega_rate_weight * dom**2
            if nu == 2:
                dap = Un[k][1] - u_prev_k[1]
                add_constraint(dap, -self.du_ap_down_norm, self.du_ap_up_norm)
                J = J + self.cfg.ap_rate_weight * dap**2
            u_prev_k = Un[k]

        x_list: list[Any] = []
        lbx: list[float] = []
        ubx: list[float] = []
        zs_big = 100.0
        slack_max = max(1.0 - margin, 0.0)
        for k in range(Np + 1):
            x_list.append(Zs[k])
            lbx.extend([-zs_big] * (2 * K))
            ubx.extend([zs_big] * (2 * K))
        for k in range(Np + 1):
            x_list.append(Ws[k])
            lbx.extend([-1.0] * nS)
            ubx.extend([1.0] * nS)
        for k in range(Np):
            x_list.append(Un[k])
            lbx.extend([-1.0] * nu)
            ubx.extend([1.0] * nu)
        for k in range(Np + 1):
            x_list.append(Ss[k])
            lbx.extend([0.0] * nS)
            ubx.extend([slack_max] * nS)

        x = ca.vertcat(*x_list)
        gcat = ca.vertcat(*g)
        nlp = {"x": x, "f": J, "g": gcat, "p": P}
        opts = {
            "print_time": False,
            "ipopt": {
                "print_level": int(self.cfg.print_level), "sb": "yes", "max_iter": int(self.cfg.max_iter),
                "tol": float(self.cfg.tol), "acceptable_tol": float(self.cfg.acceptable_tol),
                "acceptable_iter": 5, "acceptable_obj_change_tol": 1.0e-8,
                "hessian_approximation": str(self.cfg.hessian), "mu_strategy": "adaptive", "warm_start_init_point": "no",
            },
        }
        self._solver = ca.nlpsol("mpc_solver", "ipopt", nlp, opts)
        self._g_fun = ca.Function("mpc_constraints", [x, P], [gcat])
        self._f_fun = ca.Function("mpc_objective", [x, P], [J])
        self._lbx = np.asarray(lbx, dtype=np.float64)
        self._ubx = np.asarray(ubx, dtype=np.float64)
        self._lbg = np.asarray(lbg, dtype=np.float64)
        self._ubg = np.asarray(ubg, dtype=np.float64)
        self._nx = int(x.size1())
        self._off_Z = 0
        self._off_W = self._off_Z + (Np + 1) * (2 * K)
        self._off_U = self._off_W + (Np + 1) * nS
        self._off_S = self._off_U + Np * nu

    def state_estimate(self, info: dict[str, Any]) -> np.ndarray:
        if self.cfg.feedback == "observer" and "w_sensor" in info and "wdot_sensor" in info:
            w = np.asarray(info["w_sensor"], dtype=np.float64).reshape(-1)
            wd = np.asarray(info["wdot_sensor"], dtype=np.float64).reshape(-1)
            eta = self._Phi_pinv @ w
            etad = self._Phi_pinv @ wd
            return np.concatenate([eta, etad])
        return _deinterleave(info["x_modal"], self.K)

    def phiz_at(self, xc: float, yc: float) -> np.ndarray:
        v = np.zeros(self.K, dtype=np.float64)
        c = 0
        for m in range(self.plant.m_max):
            for n in range(self.plant.n_max):
                v[c] = float(self.plant.W_mn[m][n](float(xc), float(yc)))
                c += 1
        return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _interp_buffer(buf: np.ndarray, pos: float) -> float:
        arr = np.asarray(buf, dtype=np.float64).reshape(-1)
        if arr.size == 0: return 0.0
        p = float(np.clip(pos, 0.0, arr.size - 1))
        i0 = int(np.floor(p))
        if i0 >= arr.size - 1: return float(arr[-1])
        alpha = p - i0
        return float((1.0 - alpha) * arr[i0] + alpha * arr[i0 + 1])

    def _delay_value_numeric(self, timeline: np.ndarray, delay_samples: float) -> float:
        values = np.asarray(timeline, dtype=np.float64).reshape(-1)
        if values.size == 0: return 0.0
        current_idx = values.size - 1
        pos = current_idx - float(delay_samples)
        idx = np.arange(values.size, dtype=np.float64)
        d = (idx - pos) / float(self.cfg.delay_sigma)
        weights = np.exp(np.clip(-(d * d), -745.0, 0.0))
        denom = float(np.sum(weights))
        if not np.isfinite(denom) or denom <= 1.0e-300: return self._interp_buffer(values, pos)
        return float(np.dot(weights, values) / denom)

    def _phys_from_norm(self, un: np.ndarray, ap_fixed: float) -> np.ndarray:
        un = np.asarray(un, dtype=np.float64).reshape(-1)
        omega = self.omega_min + 0.5 * (un[0] + 1.0) * (self.omega_max - self.omega_min)
        if self.nu == 2: ap = self.ap_min + 0.5 * (un[1] + 1.0) * (self.ap_max - self.ap_min)
        else: ap = float(ap_fixed)
        return np.array([omega, ap], dtype=np.float64)

    def project_physical_action(self, u_phys: np.ndarray, u_prev: np.ndarray) -> np.ndarray:
        u = np.asarray(u_phys, dtype=np.float64).reshape(-1).copy()
        prev = np.asarray(u_prev, dtype=np.float64).reshape(-1)
        u[0] = np.clip(u[0], max(self.omega_min, prev[0] - self.cfg.omega_slew), min(self.omega_max, prev[0] + self.cfg.omega_slew))
        if self.nu == 2:
            u[1] = np.clip(u[1], max(self.ap_min, prev[1] - self.cfg.ap_down_slew), min(self.ap_max, prev[1] + self.cfg.ap_up_slew))
        return u

    def _project_un_plan(self, U: np.ndarray, un_prev: np.ndarray) -> np.ndarray:
        plan = np.asarray(U, dtype=np.float64).reshape(self.Np, self.nu).copy()
        prev = np.asarray(un_prev, dtype=np.float64).reshape(self.nu)
        for k in range(self.Np):
            plan[k, 0] = np.clip(plan[k, 0], max(-1.0, prev[0] - self.du_omega_norm), min(1.0, prev[0] + self.du_omega_norm))
            if self.nu == 2:
                plan[k, 1] = np.clip(plan[k, 1], max(-1.0, prev[1] - self.du_ap_down_norm), min(1.0, prev[1] + self.du_ap_up_norm))
            prev = plan[k]
        return plan

    def _rollout_plan(self, z0: np.ndarray, U_norm: np.ndarray, phiz: np.ndarray, wz_hist: np.ndarray, ap_fixed: float, un_prev: np.ndarray, omega_ref: float) -> tuple[float, np.ndarray, bool]:
        z = np.asarray(z0, dtype=np.float64).reshape(2 * self.K).copy()
        phiz = np.asarray(phiz, dtype=np.float64).reshape(self.K)
        hist = np.asarray(wz_hist, dtype=np.float64).reshape(-1)
        if hist.size != self.nb:
            if hist.size == 0: hist = np.zeros(self.nb, dtype=np.float64)
            elif hist.size < self.nb: hist = np.pad(hist, (self.nb - hist.size, 0), mode="edge")
            else: hist = hist[-self.nb :]
        hist = hist.copy()
        hist[-1] = float(phiz @ z[: self.K])
        timeline = hist.tolist()
        U = self._project_un_plan(U_norm, un_prev)
        Z = np.zeros((self.Np + 1, 2 * self.K), dtype=np.float64)
        Z[0] = z
        cost = 0.0
        feasible = True
        w0 = self.Phi @ z[: self.K]
        if np.max(np.abs(w0)) > self.w_limit * (1.0 + 1.0e-9): feasible = False
        if self.wdot_limit is not None:
            wd0 = self.Phi @ z[self.K :]
            if np.max(np.abs(wd0)) > self.wdot_limit * (1.0 + 1.0e-9): feasible = False
        slack0 = np.maximum(np.abs(w0) / self.S_w - self.w_lim_margin, 0.0)
        cost += self.cfg.slack_weight * float(np.sum(slack0))
        prev = np.asarray(un_prev, dtype=np.float64).reshape(self.nu)
        for k in range(self.Np):
            un = U[k]
            dom = float(un[0] - prev[0])
            cost += self.cfg.omega_rate_weight * dom * dom
            if abs(dom) > self.du_omega_norm + 1.0e-9: feasible = False
            if self.nu == 2:
                dap = float(un[1] - prev[1])
                cost += self.cfg.ap_rate_weight * dap * dap
                if dap > self.du_ap_up_norm + 1.0e-9 or dap < -self.du_ap_down_norm - 1.0e-9: feasible = False
            prev = un
            u_phys_full = self._phys_from_norm(un, ap_fixed)
            u_model = u_phys_full if self.nu == 2 else u_phys_full[:1]
            om = float(u_phys_full[0])
            for _ in range(int(self.cfg.control_hold)):
                if self.cfg.delay_mode == "predictive": nd = 2.0 * np.pi / (self.Nteeth * max(om, 1.0e-12) * self.env_control_dt)
                else: nd = 2.0 * np.pi / (self.Nteeth * max(float(omega_ref), 1.0e-12) * self.env_control_dt)
                wz_del = self._delay_value_numeric(np.asarray(timeline), nd)
                if self.fz_limit is not None:
                    fz = float(np.asarray(self._Fz(z, u_model, wz_del, phiz, ap_fixed, self.S0, self.S1)).reshape(-1)[0])
                    if abs(fz) > self.fz_limit * (1.0 + 1.0e-9): feasible = False
                z = np.asarray(self._F(z, u_model, wz_del, phiz, ap_fixed, self.S0, self.S1), dtype=np.float64).reshape(-1)
                if not np.all(np.isfinite(z)): return float("inf"), Z, False
                cost += float(np.asarray(self._L(z, u_model, ap_fixed), dtype=np.float64).reshape(-1)[0])
                w = self.Phi @ z[: self.K]
                if np.max(np.abs(w)) > self.w_limit * (1.0 + 1.0e-9): feasible = False
                if self.wdot_limit is not None:
                    wd = self.Phi @ z[self.K :]
                    if np.max(np.abs(wd)) > self.wdot_limit * (1.0 + 1.0e-9): feasible = False
                timeline.append(float(phiz @ z[: self.K]))
            Z[k + 1] = z
            slack = np.maximum(np.abs(self.Phi @ z[: self.K]) / self.S_w - self.w_lim_margin, 0.0)
            cost += self.cfg.slack_weight * float(np.sum(slack))
        cost += float(np.asarray(self._Lterm(z), dtype=np.float64).reshape(-1)[0])
        if not np.isfinite(cost): feasible = False
        return float(cost), Z, bool(feasible)

    def _prescreen(self, z0: np.ndarray, phiz: np.ndarray, wz_hist: np.ndarray, ap_fixed: float, u_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        prev = np.asarray(u_prev, dtype=np.float64).reshape(-1)
        om_lo = max(self.omega_min, prev[0] - self.cfg.omega_slew)
        om_hi = min(self.omega_max, prev[0] + self.cfg.omega_slew)
        om_grid = np.linspace(om_lo, om_hi, int(self.cfg.prescreen_n_omega))
        if self.nu == 2:
            ap_lo = max(self.ap_min, prev[1] - self.cfg.ap_down_slew)
            ap_hi = min(self.ap_max, prev[1] + self.cfg.ap_up_slew)
            frac = np.linspace(0.0, 1.0, int(self.cfg.prescreen_n_ap)) ** 2
            ap_grid = ap_lo + frac * (ap_hi - ap_lo)
        else:
            ap_grid = np.array([float(ap_fixed)], dtype=np.float64)
        om_mesh, ap_mesh = np.meshgrid(om_grid, ap_grid, indexing="ij")
        om_flat = om_mesh.reshape(-1)
        ap_flat = ap_mesh.reshape(-1)
        n = om_flat.size
        K = self.K
        phiz_vec = np.asarray(phiz, dtype=np.float64).reshape(K)
        hist = np.asarray(wz_hist, dtype=np.float64).reshape(-1)
        if hist.size < self.nb: hist = np.pad(hist, (self.nb - hist.size, 0), mode="edge" if hist.size else "constant")
        hist = hist[-self.nb :].copy()
        hist[-1] = float(phiz_vec @ np.asarray(z0, dtype=np.float64).reshape(-1)[:K])
        total_micro = self.Np * int(self.cfg.control_hold)
        
        un_prev = self._norm_from_phys(prev)
        un_batch = np.vstack([2.0 * (om_flat - self.omega_min) / (self.omega_max - self.omega_min) - 1.0])
        if self.nu == 2:
            apn = 2.0 * (ap_flat - self.ap_min) / max(self.ap_max - self.ap_min, 1.0e-12) - 1.0
            un_batch = np.vstack([un_batch, apn])

        # PYTHON LOOP REPLACEMENT FOR CASADI .map() TO PREVENT COMPILATION FREEZE
        current_z = np.tile(np.asarray(z0, dtype=np.float64).reshape(-1, 1), (1, n))
        buf = np.zeros((n, self.nb + total_micro), dtype=np.float64)
        buf[:, :self.nb] = hist[None, :]
        for i in range(n):
            buf[i, self.nb - 1] = float(phiz_vec @ current_z[:K, i])
            
        costs = np.zeros(n, dtype=np.float64)
        Z_plans = np.zeros((n, self.Np + 1, 2 * self.K), dtype=np.float64)
        Z_plans[:, 0, :] = current_z.T
        infeasible = np.zeros(n, dtype=bool)
        violation_score = np.zeros(n, dtype=np.float64)
        
        for i in range(n):
            z_i = current_z[:, i]
            w0 = self.Phi @ z_i[:K]
            w0_ratio = np.max(np.abs(w0)) / self.w_limit
            if w0_ratio > 1.0 + 1.0e-9: infeasible[i] = True
            violation_score[i] = max(violation_score[i], max(w0_ratio - 1.0, 0.0))
            if self.wdot_limit is not None:
                wd0 = self.Phi @ z_i[K:]
                wd0_ratio = np.max(np.abs(wd0)) / self.wdot_limit
                if wd0_ratio > 1.0 + 1.0e-9: infeasible[i] = True
                violation_score[i] = max(violation_score[i], max(wd0_ratio - 1.0, 0.0))
            slack0 = np.maximum(np.abs(w0) / self.S_w - self.w_lim_margin, 0.0)
            costs[i] += self.cfg.slack_weight * float(np.sum(slack0))
            un_i = un_batch[:, i]
            dom = float(un_i[0] - un_prev[0])
            costs[i] += self.cfg.omega_rate_weight * dom * dom
            if self.nu == 2:
                dap = float(un_i[1] - un_prev[1])
                costs[i] += self.cfg.ap_rate_weight * dap * dap

        q = 0
        for k in range(self.Np):
            for _ in range(int(self.cfg.control_hold)):
                for i in range(n):
                    if not np.isfinite(costs[i]): continue
                    z_i = current_z[:, i]
                    u_i = un_batch[:, i]
                    u_phys_i = self._phys_from_norm(u_i, ap_fixed)
                    u_model = u_phys_i if self.nu == 2 else u_phys_i[:1]
                    
                    current_idx = self.nb - 1 + q
                    if self.cfg.delay_mode == "predictive":
                        nd = 2.0 * np.pi / (self.Nteeth * max(float(u_phys_i[0]), 1.0e-12) * self.env_control_dt)
                    else:
                        nd = 2.0 * np.pi / (self.Nteeth * max(float(u_phys_i[0]), 1.0e-12) * self.env_control_dt)
                    pos = current_idx - nd
                    
                    M = 20
                    start_idx = max(0, current_idx + 1 - M)
                    recent_buf = buf[i, start_idx:current_idx + 1]
                    recent_idxs = np.arange(start_idx, current_idx + 1, dtype=np.float64)
                    d = (recent_idxs - pos) / float(self.cfg.delay_sigma)
                    weights = np.exp(np.clip(-(d * d), -745.0, 0.0))
                    wz_del = float(np.sum(weights * recent_buf) / (np.sum(weights) + 1.0e-300))
                    
                    if self.fz_limit is not None:
                        fz = float(np.asarray(self._Fz(z_i, u_model, wz_del, phiz_vec, ap_fixed, self.S0, self.S1)).reshape(-1)[0])
                        fz_ratio = abs(fz) / self.fz_limit
                        if fz_ratio > 1.0 + 1.0e-9: infeasible[i] = True
                        violation_score[i] = max(violation_score[i], max(fz_ratio - 1.0, 0.0))
                        
                    z_next = np.asarray(self._F(z_i, u_model, wz_del, phiz_vec, ap_fixed, self.S0, self.S1), dtype=np.float64).reshape(-1)
                    if not np.all(np.isfinite(z_next)):
                        costs[i] = np.inf
                        continue
                    stage = float(np.asarray(self._L(z_next, u_model, ap_fixed), dtype=np.float64).reshape(-1)[0])
                    costs[i] += stage
                    w = self.Phi @ z_next[:K]
                    w_ratio = np.max(np.abs(w)) / self.w_limit
                    if w_ratio > 1.0 + 1.0e-9: infeasible[i] = True
                    violation_score[i] = max(violation_score[i], max(w_ratio - 1.0, 0.0))
                    if self.wdot_limit is not None:
                        wd = self.Phi @ z_next[K:]
                        wd_ratio = np.max(np.abs(wd)) / self.wdot_limit
                        if wd_ratio > 1.0 + 1.0e-9: infeasible[i] = True
                        violation_score[i] = max(violation_score[i], max(wd_ratio - 1.0, 0.0))
                    current_z[:, i] = z_next
                    buf[i, current_idx + 1] = float(phiz_vec @ z_next[:K])
                q += 1
            Z_plans[:, k + 1, :] = current_z.T
            for i in range(n):
                if np.isfinite(costs[i]):
                    slack = np.maximum(np.abs(self.Phi @ current_z[:K, i]) / self.S_w - self.w_lim_margin, 0.0)
                    costs[i] += self.cfg.slack_weight * float(np.sum(slack))

        for i in range(n):
            if np.isfinite(costs[i]):
                costs[i] += float(np.asarray(self._Lterm(current_z[:, i]), dtype=np.float64).reshape(-1)[0])

        finite_cost = np.isfinite(costs)
        feasible_cost = np.where(finite_cost & ~infeasible, costs, np.inf)
        if np.any(np.isfinite(feasible_cost)):
            winner = int(np.argmin(feasible_cost))
        else:
            safe_score = np.where(finite_cost, violation_score, np.inf)
            if not np.any(np.isfinite(safe_score)):
                fallback = self.project_physical_action(np.array([prev[0], prev[1] if self.nu == 2 else ap_fixed]), prev)
                un = self._norm_from_phys(fallback[: self.nu])
                U = np.tile(un, (self.Np, 1))
                c, Z, _ = self._rollout_plan(z0, U, phiz, wz_hist, ap_fixed, un_prev, float(fallback[0]))
                return fallback[: self.nu], Z, c, U
            min_violation = float(np.min(safe_score))
            candidates = np.flatnonzero(np.isclose(safe_score, min_violation, rtol=1.0e-10, atol=1.0e-12))
            winner = int(candidates[np.argmin(costs[candidates])])

        best_phys = np.array([om_flat[winner], ap_flat[winner]], dtype=np.float64)
        best_decision_phys = best_phys[: self.nu]
        best_un = self._norm_from_phys(best_decision_phys)
        U = np.tile(best_un, (self.Np, 1))
        best_cost, best_Z, _ = self._rollout_plan(z0, U, phiz, wz_hist, ap_fixed, un_prev, float(best_phys[0]))
        return best_decision_phys, best_Z, best_cost, U

    def _pack_initial_guess(self, Z: np.ndarray, U: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z, dtype=np.float64).reshape(self.Np + 1, 2 * self.K)
        U = np.asarray(U, dtype=np.float64).reshape(self.Np, self.nu)
        x0 = np.zeros(self._nx, dtype=np.float64)
        x0[self._off_Z : self._off_W] = (Z / self.Dz[None, :]).reshape(-1)
        W = (self.Phi @ Z[:, : self.K].T).T / self.S_w
        x0[self._off_W : self._off_U] = np.clip(W, -1.0, 1.0).reshape(-1)
        x0[self._off_U : self._off_S] = np.clip(U, -1.0, 1.0).reshape(-1)
        slack = np.maximum(np.abs(W) - self.w_lim_margin, 0.0)
        slack = np.clip(slack, 0.0, max(1.0 - self.w_lim_margin, 0.0))
        x0[self._off_S :] = slack.reshape(-1)
        return np.clip(x0, self._lbx, self._ubx)

    def _constraint_violation(self, x: np.ndarray, p: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size != self._nx or not np.all(np.isfinite(x)): return float("inf")
        gval = np.asarray(self._g_fun(x, p), dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(gval)): return float("inf")
        violations = [0.0]
        finite = np.isfinite(self._lbx)
        if np.any(finite): violations.append(float(np.max(np.maximum(self._lbx[finite] - x[finite], 0.0))))
        finite = np.isfinite(self._ubx)
        if np.any(finite): violations.append(float(np.max(np.maximum(x[finite] - self._ubx[finite], 0.0))))
        finite = np.isfinite(self._lbg)
        if np.any(finite): violations.append(float(np.max(np.maximum(self._lbg[finite] - gval[finite], 0.0))))
        finite = np.isfinite(self._ubg)
        if np.any(finite): violations.append(float(np.max(np.maximum(gval[finite] - self._ubg[finite], 0.0))))
        return max(violations)

    def solve(self, z_meas: np.ndarray, phiz: np.ndarray, wz_hist: np.ndarray, ap_fixed: float, u_prev: np.ndarray) -> dict[str, Any]:
        z_meas = np.asarray(z_meas, dtype=np.float64).reshape(2 * self.K)
        phiz = np.asarray(phiz, dtype=np.float64).reshape(self.K)
        u_prev = np.asarray(u_prev, dtype=np.float64).reshape(self.nu)
        hist = np.asarray(wz_hist, dtype=np.float64).reshape(-1)
        if hist.size == 0: hist = np.full(self.nb, float(phiz @ z_meas[: self.K]))
        elif hist.size < self.nb: hist = np.pad(hist, (self.nb - hist.size, 0), mode="edge")
        else: hist = hist[-self.nb :]
        hist = hist.copy()
        hist[-1] = float(phiz @ z_meas[: self.K])
        un_prev = self._norm_from_phys(u_prev)
        u_ws, Z_ws, cost_ws, U_ws = self._prescreen(z_meas, phiz, hist, ap_fixed, u_prev)
        U_init = U_ws
        Z_init = Z_ws
        cost_init = float(cost_ws)
        omega_ref = float(u_ws[0])
        warm_start_source = "prescreen"

        if self.cfg.warm_start and self._U_last is not None:
            U_shift = np.vstack([self._U_last[1:], self._U_last[-1:]])
            U_shift = self._project_un_plan(U_shift, un_prev)
            omega_ref_shift = float(self._phys_from_norm(U_shift[0], ap_fixed)[0])
            cost_shift, Z_shift, feasible_shift = self._rollout_plan(z_meas, U_shift, phiz, hist, ap_fixed, un_prev, omega_ref_shift)
            if feasible_shift and np.isfinite(cost_shift) and (not np.isfinite(cost_init) or cost_shift <= cost_init):
                U_init = U_shift
                Z_init = Z_shift
                cost_init = float(cost_shift)
                omega_ref = omega_ref_shift
                warm_start_source = "shifted_previous"

        x0 = self._pack_initial_guess(Z_init, U_init)
        p_vec = np.concatenate([z_meas, phiz, hist, [float(ap_fixed)], un_prev, [self.S0], [self.S1], [omega_ref]])
        converged = False
        status = "solver_not_run"
        iterations = -1
        solver_exception: str | None = None
        xopt: np.ndarray | None = None
        obj = float(cost_init)
        primal_violation = float("inf")
        solve_t0 = time.perf_counter()
        try:
            sol = self._solver(x0=x0, p=p_vec, lbx=self._lbx, ubx=self._ubx, lbg=self._lbg, ubg=self._ubg)
            stats = self._solver.stats()
            converged = bool(stats.get("success", False))
            status = str(stats.get("return_status", ""))
            iterations = int(stats.get("iter_count", -1))
            x_candidate = np.asarray(sol["x"], dtype=np.float64).reshape(-1)
            obj_candidate = float(np.asarray(sol["f"], dtype=np.float64).reshape(-1)[0])
            if x_candidate.size == self._nx and np.all(np.isfinite(x_candidate)) and np.isfinite(obj_candidate):
                xopt = x_candidate
                obj = obj_candidate
                primal_violation = self._constraint_violation(x_candidate, p_vec)
        except Exception as exc:
            solver_exception = f"{type(exc).__name__}: {exc}"
            status = "solver_exception"
        solve_time = time.perf_counter() - solve_t0
        feasible = bool(xopt is not None and np.isfinite(primal_violation) and primal_violation <= float(self.cfg.primal_feas_tol))
        fallback_used = not feasible
        if feasible:
            U_plan = xopt[self._off_U : self._off_S].reshape(self.Np, self.nu)
            U_plan = np.clip(U_plan, -1.0, 1.0)
            self._U_last = U_plan.copy()
        else:
            U_plan = np.asarray(U_init, dtype=np.float64).reshape(self.Np, self.nu)
            self._U_last = U_plan.copy()
            obj = float(cost_init)
        un0 = U_plan[0].copy()
        u_full = self._phys_from_norm(un0, ap_fixed)
        projected = self.project_physical_action(u_full, u_prev)
        projection_norm = float(np.linalg.norm(projected - u_full, ord=np.inf))
        u_full = projected
        return {
            "u_phys": u_full, "u_decision": un0, "U_plan": U_plan, "cost": float(obj),
            "success": bool(feasible), "feasible": bool(feasible), "converged": bool(converged),
            "fallback_used": bool(fallback_used), "warm_start_source": warm_start_source,
            "primal_violation": float(primal_violation), "iter": int(iterations), "return_status": status,
            "solver_exception": solver_exception, "solve_time_s": float(solve_time), "safety_projection_inf_norm": projection_norm,
        }

    def to_env_action(self, u_phys: np.ndarray) -> np.ndarray:
        plant = self.plant
        low = np.asarray(plant._ctrl_low, dtype=np.float64)
        high = np.asarray(plant._ctrl_high, dtype=np.float64)
        if self.nu == 2: phys = np.array([u_phys[0], u_phys[1]], dtype=np.float64)
        else: phys = np.array([u_phys[0]], dtype=np.float64)
        norm = 2.0 * (phys - low) / (high - low) - 1.0
        return np.clip(norm, -1.0, 1.0)

def _deinterleave(x_modal: np.ndarray, K: int) -> np.ndarray:
    x = np.asarray(x_modal, dtype=np.float64).reshape(-1)
    eta = x[0::2]
    etad = x[1::2]
    return np.concatenate([eta, etad])

def _safe_float(value: Any, default: float = float("nan")) -> float:
    try: out = float(value)
    except Exception: return float(default)
    return out if np.isfinite(out) else float(default)

def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_to_jsonable(v) for v in value]
    return value

def _process_snapshot(info: dict[str, Any], plant: Any) -> dict[str, float]:
    process = face_milling_process_from_info(info, plant)
    for key in ("cutter_x", "cutter_y", "feed_progress", "feed_distance_m", "spindle_phase_rad"):
        if key in info: process[key] = _safe_float(info[key])
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

def run_episode(env: gym.Env, mpc: FaceMillingMPC, *, seed: int, reset_options: dict[str, Any] | None, max_steps: int, metadata: dict[str, Any], verbose: bool = True) -> dict[str, Any]:
    if max_steps <= 0: raise ValueError("max_steps must be positive")
    plant = env.unwrapped.plant
    K = mpc.K
    hold = int(mpc.cfg.control_hold)
    mpc.reset()
    obs, reset_info = env.reset(seed=seed, options=reset_options)
    if "x_modal" not in reset_info: raise RuntimeError("Environment reset info lacks 'x_modal'; MPC needs modal state.")
    z_block = mpc.state_estimate(reset_info)
    ap_fixed = float(getattr(plant, "_episode_ap", 0.5 * (mpc.ap_min + mpc.ap_max)))
    if mpc.control_ap: ap_fixed = 0.5 * (mpc.ap_min + mpc.ap_max)

    def cutter_mode_shape() -> np.ndarray:
        xc_now = float(np.clip(plant.L1 - plant._feed_distance_m - plant.x0_cutter, 0.0, plant.L1))
        return mpc.phiz_at(xc_now, float(plant.y_cutter))

    phiz0 = cutter_mode_shape()
    wz0 = float(phiz0 @ z_block[:K])
    wz_hist: deque[float] = deque([wz0 for _ in range(mpc.nb)], maxlen=mpc.nb)
    om_start = 0.5 * (mpc.omega_min + mpc.omega_max)
    ap_start = mpc.ap_min + 0.01 * (mpc.ap_max - mpc.ap_min)
    if mpc.control_ap: u_prev = np.array([om_start, ap_start], dtype=np.float64)
    else: u_prev = np.array([om_start], dtype=np.float64)

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
    solve_feasible: list[bool] = []
    solve_converged: list[bool] = []
    fallback_used: list[bool] = []
    primal_violations: list[float] = []
    solve_times: list[float] = []
    projection_sizes: list[float] = []
    solve_statuses: list[str] = []
    termination_reason: str | None = None
    terminated_final = False
    truncated_final = False
    pass_completed_final = False
    env_step = 0
    mpc_solve_count = 0
    t0 = time.time()

    while env_step < max_steps:
        phiz = cutter_mode_shape()
        sol = mpc.solve(z_block, phiz, np.asarray(wz_hist, dtype=np.float64), ap_fixed, u_prev)
        u_phys = np.asarray(sol["u_phys"], dtype=np.float64).copy()
        a_norm = mpc.to_env_action(u_phys)
        mpc_solve_count += 1
        solve_iters.append(int(sol["iter"]))
        solve_feasible.append(bool(sol["feasible"]))
        solve_converged.append(bool(sol["converged"]))
        fallback_used.append(bool(sol["fallback_used"]))
        primal_violations.append(float(sol["primal_violation"]))
        solve_times.append(float(sol["solve_time_s"]))
        projection_sizes.append(float(sol["safety_projection_inf_norm"]))
        solve_statuses.append(str(sol["return_status"]))

        # PROGRESS LOGGING
        if env_step % 500 == 0 and env_step > 0:
            print(f"    [Progress] Step {env_step}/{max_steps}, Current Return: {sum(rewards):.1f}")

        done = False
        for _ in range(hold):
            if env_step >= max_steps:
                done = True
                break
            obs, reward, terminated, truncated, info = env.step(a_norm)
            obs_arr = np.asarray(obs, dtype=np.float64).reshape(-1)
            actions.append(np.asarray(a_norm, dtype=np.float64).reshape(-1).tolist())
            physical_actions.append([float(u_phys[0]), float(u_phys[1])])
            observations.append(obs_arr.tolist())
            if "w_sensor" in info: wsens = np.asarray(info["w_sensor"], dtype=np.float64).reshape(-1)
            else: wsens = np.zeros(mpc.nS, dtype=np.float64)
            if "wdot_sensor" in info: wdsens = np.asarray(info["wdot_sensor"], dtype=np.float64).reshape(-1)
            else: wdsens = np.zeros(mpc.nS, dtype=np.float64)
            w_sensor_hist.append(wsens.tolist())
            wdot_sensor_hist.append(wdsens.tolist())
            physical_signals.append(np.concatenate([wsens, wdsens]).tolist())
            if "reward_terms" in info: reward_terms_hist.append(dict(info["reward_terms"]))
            if "x_modal" in info: x_modal_hist.append(np.asarray(info["x_modal"], dtype=np.float64).reshape(-1).tolist())
            for key, val in _process_snapshot(info, plant).items(): process_hist[key].append(val)
            rewards.append(float(reward))
            times.append(float(info.get("t", (env_step + 1) * mpc.env_control_dt)))
            env_step += 1
            z_block = mpc.state_estimate(info)
            phiz_actual = cutter_mode_shape()
            wz_hist.append(float(phiz_actual @ z_block[:K]))
            if terminated or truncated:
                terminated_final = bool(terminated)
                truncated_final = bool(truncated)
                termination_reason = info.get("termination_reason")
                pass_completed_final = bool(info.get("pass_completed") or termination_reason in PASS_COMPLETED_REASONS)
                done = True
                break
        u_prev = u_phys[: mpc.nu].copy()
        if done: break

    if env_step >= max_steps and not terminated_final and not truncated_final:
        truncated_final = True
        termination_reason = "external_max_env_steps"
    wall = time.time() - t0
    if verbose:
        feasible_pct = 100.0 * (np.mean(solve_feasible) if solve_feasible else 0.0)
        conv_pct = 100.0 * (np.mean(solve_converged) if solve_converged else 0.0)
        fallback_pct = 100.0 * (np.mean(fallback_used) if fallback_used else 0.0)
        print(f"    scenario_seed {seed}: env_steps={len(rewards)} solves={mpc_solve_count} return={sum(rewards):.1f} pass={'Y' if pass_completed_final else 'N'} reason={termination_reason} feasible={feasible_pct:.0f}% converged={conv_pct:.0f}% fallback={fallback_pct:.0f}% wall={wall:.1f}s")

    w_arr = np.asarray(w_sensor_hist, dtype=np.float64)
    max_abs_w = float(np.max(np.abs(w_arr))) if w_arr.size else float("nan")
    rms_w = float(np.sqrt(np.mean(w_arr**2))) if w_arr.size else float("nan")
    finite_primal = np.asarray(primal_violations, dtype=np.float64)
    finite_primal = finite_primal[np.isfinite(finite_primal)]
    record = {
        "controller": "mpc", "metadata": metadata, "scenario_seed": int(seed), "reset_options": reset_options,
        "reset_info": _to_jsonable(dict(reset_info)), "y_line_m": _safe_float(reset_info.get("y_cutter", np.nan)),
        "x_start_m": _safe_float(reset_info.get("x_start", np.nan)), "x_end_target_m": _safe_float(reset_info.get("x_end", np.nan)),
        "ap_fixed_mm": _safe_float(reset_info.get("ap_fixed_mm", ap_fixed if not mpc.control_ap else np.nan)),
        "times": times, "observations": observations, "physical_signals": physical_signals, "x_modal": x_modal_hist,
        "actions": actions, "physical_actions": physical_actions, "process": process_hist, "w_sensor": w_sensor_hist,
        "wdot_sensor": wdot_sensor_hist, "reward_terms": reward_terms_hist, "rewards": rewards, "return": float(sum(rewards)),
        "length": int(len(rewards)), "terminated": bool(terminated_final), "truncated": bool(truncated_final),
        "pass_completed": bool(pass_completed_final), "termination_reason": termination_reason,
        "mpc_diagnostics": {"iterations": solve_iters, "feasible": solve_feasible, "converged": solve_converged, "fallback_used": fallback_used, "primal_violation": primal_violations, "solve_time_s": solve_times, "safety_projection_inf_norm": projection_sizes, "return_status": solve_statuses},
        "summary": {"return": float(sum(rewards)), "length": int(len(rewards)), "max_abs_w_m": max_abs_w, "rms_w_m": rms_w, "pass_completed": bool(pass_completed_final), "termination_reason": termination_reason, "mpc_solve_count": int(mpc_solve_count), "mpc_mean_solve_iters": float(np.mean(solve_iters)) if solve_iters else float("nan"), "mpc_solve_feasible_frac": float(np.mean(solve_feasible)) if solve_feasible else 0.0, "mpc_solve_converged_frac": float(np.mean(solve_converged)) if solve_converged else 0.0, "mpc_fallback_frac": float(np.mean(fallback_used)) if fallback_used else 0.0, "mpc_max_primal_violation": float(np.max(finite_primal)) if finite_primal.size else float("nan"), "mpc_mean_solve_time_s": float(np.mean(solve_times)) if solve_times else float("nan"), "mpc_p95_solve_time_s": float(np.percentile(solve_times, 95)) if solve_times else float("nan"), "mpc_max_safety_projection": float(np.max(projection_sizes)) if projection_sizes else 0.0, "mpc_wall_time_s": float(wall)},
    }
    return record

def build_env(args: argparse.Namespace) -> gym.Env:
    register_envs()
    env_kwargs = plate_env_kwargs(reward_id=args.reward, dt=args.dt, n_substeps=args.n_substeps, max_episode_steps=args.max_episode_steps, dynamics_uncertainty_std=args.dynamics_uncertainty_std, randomize_y0=args.randomize_y0)
    if args.w_limit is not None: env_kwargs["w_limit"] = float(args.w_limit)
    return gym.make(args.env_id, **env_kwargs)

def _load_reference_scenarios(directory: Path, seed: int) -> list[dict[str, Any]]:
    path = Path(directory) / f"trajectories_seed{seed}.json"
    if not path.is_file(): raise FileNotFoundError(f"Reference trajectory file not found: {path}")
    with open(path, "r", encoding="utf-8") as f: data = json.load(f)
    if isinstance(data, dict): data = [data]
    if not isinstance(data, list): raise ValueError(f"Reference trajectory must contain a record or list: {path}")
    scenarios: list[dict[str, Any]] = []
    for ep_idx, parent in enumerate(data):
        if not isinstance(parent, dict): continue
        runs = parent.get("runs")
        records = runs if isinstance(runs, list) and runs else [parent]
        for run_idx, child in enumerate(records):
            if not isinstance(child, dict): continue
            rec = {**parent, **child}
            options = dict(rec.get("reset_options") or {})
            y_line = rec.get("y_line_m", rec.get("reset_info", {}).get("y_cutter"))
            try: y_line_f = float(y_line)
            except (TypeError, ValueError): y_line_f = float("nan")
            if "y0" not in options and np.isfinite(y_line_f): options["y0"] = y_line_f
            ap_val = rec.get("ap_fixed_mm", rec.get("reset_info", {}).get("ap_fixed_mm"))
            try: ap_f = float(ap_val)
            except (TypeError, ValueError): ap_f = float("nan")
            if "ap" not in options and np.isfinite(ap_f): options["ap"] = ap_f
            explicit_seed = rec.get("scenario_seed", rec.get("reset_seed"))
            scenario_seed = int(explicit_seed) if explicit_seed is not None else int(seed * 1000 + ep_idx * 100 + run_idx)
            scenarios.append({"episode": int(ep_idx), "run_index": int(run_idx), "scenario_seed": scenario_seed, "reset_options": options or None, "reference_file": str(path)})
    return scenarios

def main() -> int:
    parser = argparse.ArgumentParser(description="Nonlinear MPC chatter-suppression controller for face milling.")
    parser.add_argument("--env-id", default="CustomODEPlate-v0", choices=["CustomODEPlate-v0", "CustomODEPlateFinish-v0"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--n-episodes", type=int, default=None)
    parser.add_argument("--match-rl-dir", default=None)
    parser.add_argument("--out-dir", default="mpc_trajectories")
    parser.add_argument("--dt", type=float, default=1.0e-4)
    parser.add_argument("--n-substeps", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--max-env-steps", "--max-steps", dest="max_env_steps", type=int, default=20000)
    parser.add_argument("--reward", default="productive", choices=["dense", "productive"])
    parser.add_argument("--w-limit", type=float, default=None)
    parser.add_argument("--dynamics-uncertainty-std", type=float, default=0.0)
    parser.add_argument("--randomize-y0", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--y0", type=float, default=None)
    parser.add_argument("--ap", type=float, default=None)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--n-rk", type=int, default=6)
    parser.add_argument("--control-hold", type=int, default=3)
    parser.add_argument("--slack-weight", type=float, default=5.0e4)
    parser.add_argument("--omega-rate-weight", type=float, default=5.0)
    parser.add_argument("--ap-rate-weight", type=float, default=5.0)
    parser.add_argument("--terminal-weight", type=float, default=3.0)
    parser.add_argument("--omega-slew", type=float, default=600.0)
    parser.add_argument("--ap-up-slew", type=float, default=0.25)
    parser.add_argument("--ap-down-slew", type=float, default=1.5)
    parser.add_argument("--wdot-limit", type=float, default=None)
    parser.add_argument("--fz-limit", type=float, default=None)
    parser.add_argument("--prescreen-n-omega", type=int, default=28)
    parser.add_argument("--prescreen-n-ap", type=int, default=7)
    parser.add_argument("--delay-sigma", type=float, default=0.7)
    parser.add_argument("--delay-mode", default="predictive", choices=["predictive", "frozen"])
    parser.add_argument("--w-lim-margin", type=float, default=0.8)
    parser.add_argument("--hessian", default="limited-memory", choices=["limited-memory", "exact"])
    parser.add_argument("--tol", type=float, default=1.0e-6)
    parser.add_argument("--primal-feas-tol", type=float, default=1.0e-5)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--feedback", default="state", choices=["state", "observer"])
    parser.add_argument("--ipopt-verbose", action="store_true")
    args = parser.parse_args()

    positive = {"dt": args.dt, "n_substeps": args.n_substeps, "max_env_steps": args.max_env_steps, "horizon": args.horizon, "n_rk": args.n_rk, "control_hold": args.control_hold, "prescreen_n_omega": args.prescreen_n_omega, "prescreen_n_ap": args.prescreen_n_ap}
    for name, value in positive.items():
        if float(value) <= 0.0: raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.n_episodes is not None and args.n_episodes <= 0: raise SystemExit("--n-episodes must be positive")

    cfg = MPCConfig(horizon=args.horizon, n_rk=args.n_rk, control_hold=args.control_hold, slack_weight=args.slack_weight, omega_rate_weight=args.omega_rate_weight, ap_rate_weight=args.ap_rate_weight, terminal_weight=args.terminal_weight, omega_slew=args.omega_slew, ap_up_slew=args.ap_up_slew, ap_down_slew=args.ap_down_slew, wdot_limit=args.wdot_limit, fz_limit=args.fz_limit, prescreen_n_omega=args.prescreen_n_omega, prescreen_n_ap=args.prescreen_n_ap, delay_sigma=args.delay_sigma, delay_mode=args.delay_mode, w_lim_margin=args.w_lim_margin, hessian=args.hessian, tol=args.tol, primal_feas_tol=args.primal_feas_tol, max_iter=args.max_iter, feedback=args.feedback, print_level=5 if args.ipopt_verbose else 0)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    env = build_env(args)
    plant = env.unwrapped.plant
    tau_min = 2.0 * np.pi / (max(int(plant.N), 1) * max(float(plant.omega_max), 1.0e-12))
    if float(args.dt) >= tau_min:
        env.close()
        raise SystemExit(f"RK4 dt={args.dt:g}s is not smaller than minimum tooth delay {tau_min:g}s")

    mpc = FaceMillingMPC(env, cfg)
    metadata = plant_plot_metadata(env)
    metadata["controller"] = "mpc"
    metadata["env_id"] = args.env_id
    metadata["reward_id"] = args.reward
    metadata["scenario_source"] = args.match_rl_dir
    metadata["mpc"] = {"horizon": cfg.horizon, "env_control_dt_s": mpc.env_control_dt, "mpc_Ts_s": mpc.Ts, "n_rk_per_env_step": cfg.n_rk, "control_hold": cfg.control_hold, "delay_buffer_nb_env_steps": mpc.nb, "delay_mode": cfg.delay_mode, "delay_sigma": cfg.delay_sigma, "feedback": cfg.feedback, "reward_class": mpc.reward_class, "w_limit": mpc.w_limit, "w_inner_limit": mpc.w_lim_margin * mpc.w_limit, "wdot_limit": mpc.wdot_limit, "fz_limit": mpc.fz_limit, "omega_slew": cfg.omega_slew, "ap_up_slew": cfg.ap_up_slew, "ap_down_slew": cfg.ap_down_slew, "primal_feas_tol": cfg.primal_feas_tol, "prescreen_n_omega": cfg.prescreen_n_omega, "prescreen_n_ap": cfg.prescreen_n_ap}
    mode = "finishing: omega" if not mpc.control_ap else "roughing: omega and ap"
    print(f"MPC controller ({mode})")
    print(f"  env step={mpc.env_control_dt*1e3:.3f} ms, MPC update={mpc.Ts*1e3:.3f} ms, horizon={mpc.Np*mpc.Ts*1e3:.1f} ms, prediction RK4 h={mpc.env_control_dt/mpc.n_rk*1e3:.4f} ms")
    print(f"  spindle={omega_to_rpm(mpc.omega_min):.0f}-{omega_to_rpm(mpc.omega_max):.0f} rpm, ap={mpc.ap_min:g}-{mpc.ap_max:g} mm, hard |w|<={mpc.w_limit:g} m, soft margin={mpc.w_lim_margin*mpc.w_limit:g} m")
    print(f"  hard increments: domega<={cfg.omega_slew:g} rad/s/update, dap=[-{cfg.ap_down_slew:g}, +{cfg.ap_up_slew:g}] mm/update; wdot_limit={mpc.wdot_limit}, fz_limit={mpc.fz_limit}")
    print(f"  output -> {out_dir}")

    for seed in args.seeds:
        if args.match_rl_dir:
            scenarios = _load_reference_scenarios(Path(args.match_rl_dir), seed)
            if args.n_episodes is not None: scenarios = scenarios[: args.n_episodes]
        else:
            count = args.n_episodes if args.n_episodes is not None else 1
            scenarios = [{"episode": ep, "run_index": 0, "scenario_seed": int(seed * 1000 + ep), "reset_options": None, "reference_file": None} for ep in range(count)]
        trajectories: list[dict[str, Any]] = []
        for scenario in scenarios:
            reset_options = dict(scenario.get("reset_options") or {})
            if args.y0 is not None: reset_options["y0"] = float(np.clip(args.y0, 0.0, float(plant.L2)))
            if args.ap is not None and not mpc.control_ap: reset_options["ap"] = float(args.ap)
            scenario_seed = int(scenario["scenario_seed"])
            ep_idx = int(scenario["episode"])
            run_idx = int(scenario.get("run_index", 0))
            print(f"  [seed {seed} ep {ep_idx} run {run_idx}] solving matched scenario {scenario_seed}...")
            record = run_episode(env, mpc, seed=scenario_seed, reset_options=reset_options or None, max_steps=args.max_env_steps, metadata=metadata)
            record["episode"] = ep_idx
            record["run_index"] = run_idx
            record["seed"] = int(seed)
            record["reference_file"] = scenario.get("reference_file")
            trajectories.append(record)
        out_path = out_dir / f"trajectories_seed{seed}.json"
        with open(out_path, "w", encoding="utf-8") as f: json.dump(_to_jsonable(trajectories), f, indent=2)
        print(f"  saved {len(trajectories)} episode(s) -> {out_path}")
    env.close()
    print("MPC evaluation complete.")
    print(f"Compare with RL: python scripts/compare_mpc_rl.py --rl-dir {DEFAULT_TRAJ_DIR} --mpc-dir {out_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())