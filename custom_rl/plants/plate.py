"""Nonlinear plate vibration ODE plant."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from gymnasium import spaces

from custom_rl.plants.base import ODEPlant
from custom_rl.plants import f_nonlinear2
from custom_rl.plants.compute_mode_shapes import compute_mode_shapes
from custom_rl.plants.compute_natural_frequencies import compute_natural_frequencies
from custom_rl.plants.compute_nonlinear_stiffness import compute_nonlinear_stiffness

# AL7075 cantilever plate (clamped at y = L2), abstract sensor locations [m].
DEFAULT_SENSOR_POINTS: tuple[tuple[float, float], ...] = (
    (0.83, 0.20),
    (0.17, 0.20),
)

DEFAULT_Y_CUTTER = 0.20

# Spindle speed operating range [rpm]; internal physics uses rad/s.
RPM_MIN = 50.0
RPM_MAX = 4000.0


def rpm_to_omega(rpm: float | np.ndarray) -> np.ndarray:
    """Convert spindle speed from rpm to rad/s (scalar or array)."""
    return np.asarray(rpm, dtype=np.float64) * 2.0 * np.pi / 60.0


def omega_to_rpm(omega: float | np.ndarray) -> np.ndarray:
    """Convert spindle speed from rad/s to rpm (scalar or array)."""
    return np.asarray(omega, dtype=np.float64) * 60.0 / (2.0 * np.pi)


OMEGA_MIN_RAD_S = float(rpm_to_omega(RPM_MIN))
OMEGA_MAX_RAD_S = float(rpm_to_omega(RPM_MAX))

# Reference pass speed for RL episode truncation (not slowest 50 rpm).
TRAINING_REFERENCE_RPM = 1000.0


def estimate_pass_duration(
    L1: float,
    cf: float,
    N: int,
    omega: float,
) -> float:
    """Seconds to complete a straight pass at constant spindle speed."""
    feed = (cf / 1000.0) * N * float(omega) / (2.0 * np.pi)
    return float(L1) / max(feed, 1e-12)


def estimate_pass_episode_steps(
    L1: float,
    cf: float,
    N: int,
    omega_min: float,
    step_dt: float,
    margin: float = 1.1,
) -> int:
    """Episode step budget for the slowest pass (omega_min)."""
    duration = estimate_pass_duration(L1, cf, N, omega_min) * margin
    return int(np.ceil(duration / max(step_dt, 1e-12))) + 200


def estimate_training_episode_steps(
    L1: float,
    cf: float,
    N: int,
    step_dt: float,
    *,
    reference_rpm: float = TRAINING_REFERENCE_RPM,
    margin: float = 1.35,
) -> int:
    """
    Practical RL episode cap from a typical pass duration (~reference_rpm).

    Using omega_min (50 rpm) can exceed 800k steps at dt=0.001 and makes PPO
    eval rollouts appear hung. Pass completion still terminates earlier when
    the tool finishes the cut.
    """
    omega_ref = float(rpm_to_omega(reference_rpm))
    return estimate_pass_episode_steps(L1, cf, N, omega_ref, step_dt, margin=margin)


def build_sensor_mode_matrix(
    W_mn: list,
    m_max: int,
    n_max: int,
    sensor_points: Sequence[tuple[float, float]],
) -> np.ndarray:
    """
    Build sensor mode-shape matrix Phi[i, k] = W_k(x_s_i, y_s_i).
    """
    n_sensors = len(sensor_points)
    k_modes = m_max * n_max
    phi = np.zeros((n_sensors, k_modes), dtype=np.float64)

    mode_index = 0
    for m in range(m_max):
        for n in range(n_max):
            w_func = W_mn[m][n]
            for i, (xs, ys) in enumerate(sensor_points):
                phi[i, mode_index] = float(w_func(xs, ys))
            mode_index += 1

    return phi


class PlatePlant(ODEPlant):
    """
    Nonlinear plate vibration plant.

    Internal modal state:
        x_modal = [eta1, eta1_dot, eta2, eta2_dot, ..., etaK, etaK_dot]

    Agent observation (physical sensors):
        [w_sensor / w_obs_scale, wdot_sensor / wdot_obs_scale]

    Normalized action:
        u = [u_omega, u_ac], each in [-1, 1]

    Physical action after scaling:
        omega [rad/s] (50-4000 rpm), ac [mm] depth of cut
    """

    def __init__(
        self,
        N: int = 5,
        L1: float = 1.0,
        L2: float = 1.0,
        h: float = 0.02,
        E: float = 71.7e9,
        nu: float = 0.33,
        rho: float = 2810.0,
        m_max: int = 3,
        n_max: int = 2,
        omega_min: float = OMEGA_MIN_RAD_S,
        omega_max: float = OMEGA_MAX_RAD_S,
        ac_min: float = 0,
        ac_max: float = 20.0,
        sensor_points: Sequence[tuple[float, float]] = DEFAULT_SENSOR_POINTS,
        w_limit: float = 0.01,
        w_obs_scale: float = 0.01,
        wdot_obs_scale: float = 1.0,
        wdot_limit: float = 10.0,
        eta_limit: float = 0.01,
        y_cutter: float = DEFAULT_Y_CUTTER,
        x0_cutter: float = 0.0,
        x_pass_end_tol: float = 0.01,
        dynamics_uncertainty_std: float = 0.0,
        y0_min: float = 0.05,
        y0_max: float = 0.45,
        randomize_y0: bool = True,
    ):
        self.N = N
        self.L1 = L1
        self.L2 = L2
        self.h = h

        self.E = E
        self.nu = nu
        self.rho = rho

        self.m_max = m_max
        self.n_max = n_max
        self.K = self.m_max * self.n_max
        self.state_dim = 2 * self.K

        self.omega_min = omega_min
        self.omega_max = omega_max
        self.ac_min = ac_min
        self.ac_max = ac_max

        self.sensor_points = tuple(
            (float(xs), float(ys)) for xs, ys in sensor_points
        )
        self.n_sensors = len(self.sensor_points)
        self.obs_dim = 2 * self.n_sensors

        self.w_limit = w_limit
        self.w_obs_scale = w_obs_scale
        self.wdot_obs_scale = wdot_obs_scale
        self.wdot_limit = wdot_limit
        self.eta_limit = eta_limit
        self.y_cutter = float(y_cutter)
        self.x0_cutter = float(x0_cutter)
        self.x_pass_end_tol = float(x_pass_end_tol)
        self.cf = 0.3
        self._last_omega = float(omega_min)
        self.dynamics_uncertainty_std = float(max(dynamics_uncertainty_std, 0.0))
        self.y0_min = float(y0_min)
        self.y0_max = float(y0_max)
        self.randomize_y0 = bool(randomize_y0)
        self._rng: np.random.Generator | None = None

        self.u_phys_low = np.array(
            [self.omega_min, self.ac_min],
            dtype=np.float64,
        )
        self.u_phys_high = np.array(
            [self.omega_max, self.ac_max],
            dtype=np.float64,
        )

        # Straight-line pass reference timeline (slowest feed at omega_min).
        pass_duration = estimate_pass_duration(
            self.L1, self.cf, self.N, self.omega_min
        )
        self.t_original = np.arange(
            0.0,
            pass_duration * 1.1,
            0.02,
            dtype=np.float64,
        )
        feed_slow = (self.cf / 1000.0) * self.N * self.omega_min / (2.0 * np.pi)
        self.x_traj = np.clip(
            self.L1 - feed_slow * self.t_original - self.x0_cutter,
            0.0,
            self.L1,
        )
        self.y_traj = np.full_like(self.t_original, self.y_cutter)

        # Modal mass: rho * h integrated over plate area (no double-counting of h).
        M_modal = self.L1 * self.L2 * self.rho * self.h

        W_mn, V_mn = compute_mode_shapes(
            self.L1,
            self.L2,
            self.h,
            self.m_max,
            self.n_max,
        )

        self.Phi = build_sensor_mode_matrix(
            W_mn,
            self.m_max,
            self.n_max,
            self.sensor_points,
        )

        omega_mn = compute_natural_frequencies(
            self.E,
            self.nu,
            self.rho,
            self.h,
            self.L1,
            self.L2,
            self.m_max,
            self.n_max,
        )

        lambda_mn, _lambda_prime_mn = compute_nonlinear_stiffness(
            self.E,
            self.nu,
            self.h,
            self.L1,
            self.L2,
            W_mn,
            V_mn,
            self.m_max,
            self.n_max,
        )

        omega_vec = np.asarray(omega_mn.T.reshape(self.K), dtype=np.float64)
        lambda_vec = np.asarray(lambda_mn.T.reshape(self.K), dtype=np.float64)

        zeta_vec = 0.05 * np.ones(self.K, dtype=np.float64)
        cf = self.cf

        xi_base = np.array(
            [6765e9, -4910e6, 2840e3, 132],
            dtype=np.float64,
        ) / 2.5

        delta_base = np.array(
            [12740e9, -7452e6, 1674e3, 246],
            dtype=np.float64,
        ) / 2.5

        f_nonlinear2.m_max = self.m_max
        f_nonlinear2.n_max = self.n_max
        f_nonlinear2.N = self.N
        f_nonlinear2.K = self.K

        f_nonlinear2.zeta_vec = zeta_vec
        f_nonlinear2.lambda_vec = lambda_vec
        f_nonlinear2.omega_vec = omega_vec

        f_nonlinear2.xi_base = xi_base
        f_nonlinear2.delta_base = delta_base
        f_nonlinear2.W_mn = W_mn
        f_nonlinear2.cf = cf

        f_nonlinear2.t_original = self.t_original
        f_nonlinear2.x_traj = self.x_traj
        f_nonlinear2.y_traj = self.y_traj

        f_nonlinear2.L1 = self.L1
        f_nonlinear2.x0_cutter = self.x0_cutter
        f_nonlinear2.y_cutter = self.y_cutter
        f_nonlinear2.x_pass_end_tol = self.x_pass_end_tol
        f_nonlinear2.USE_MOVING_FORCE_PROJECTION = True

        f_nonlinear2.M_modal = M_modal
        f_nonlinear2.decimal_places = 1

    def bind_rng(self, rng: np.random.Generator) -> None:
        """Bind Gymnasium RNG for y0 sampling and dynamics uncertainty."""
        self._rng = rng

    def _sync_f_nonlinear2_geometry(self) -> None:
        """Push milling path geometry into f_nonlinear2 module globals."""
        f_nonlinear2.y_traj = self.y_traj
        f_nonlinear2.y_cutter = self.y_cutter
        # Invalidate force cache so omega/ac trials pick up new y0 projection.
        f_nonlinear2.cache["initialized"] = False

    def set_milling_start_y(self, y0_m: float) -> None:
        """
        Set the straight-pass milling start y (meters).

        Updates the cutter path (f_nonlinear2.y_cutter, y_traj) and the modal
        force projection W_k(x_c, y0) used in _compute_b_vec_at. The 3D stability
        surface is only physically meaningful when y0 changes both path and force
        location; it is a process parameter, not an RL control action.
        """
        y0_m = float(np.clip(y0_m, 0.0, self.L2))
        self.y_cutter = y0_m
        self.y_traj = np.full_like(self.t_original, y0_m, dtype=np.float64)
        self._sync_f_nonlinear2_geometry()

    def modal_to_physical(
        self,
        x_modal: np.ndarray,
        *,
        clip: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map modal state to physical sensor displacement and velocity."""
        x_modal = np.asarray(x_modal, dtype=np.float64).reshape(-1)
        eta = x_modal[0::2]
        eta_dot = x_modal[1::2]

        w_sensor = self.Phi @ eta
        wdot_sensor = self.Phi @ eta_dot

        w_sensor = np.asarray(w_sensor, dtype=np.float64)
        wdot_sensor = np.asarray(wdot_sensor, dtype=np.float64)

        if clip:
            w_sensor = np.clip(w_sensor, -self.w_limit, self.w_limit)
            wdot_sensor = np.clip(wdot_sensor, -self.wdot_limit, self.wdot_limit)

        return w_sensor, wdot_sensor

    def get_observe_state(self, x_modal: np.ndarray) -> np.ndarray:
        """Return scaled physical observation for the RL agent."""
        return self.state_to_obs(x_modal)

    def _scale_action(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64).reshape(-1)

        if u.size != 2:
            raise ValueError(
                f"PlatePlant expects action with shape (2,), but received {u.shape}."
            )

        u = np.clip(u, -1.0, 1.0)

        u_phys = self.u_phys_low + 0.5 * (u + 1.0) * (
            self.u_phys_high - self.u_phys_low
        )

        return np.asarray(u_phys, dtype=np.float64)

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)

        if x.size != self.state_dim:
            raise ValueError(
                f"PlatePlant expects state with shape ({self.state_dim},), "
                f"but received {x.shape}."
            )

        u_phys = self._scale_action(u)
        self._last_omega = float(u_phys[0])
        x_dot = f_nonlinear2.f_nonlinear2(t, x, u_phys)
        x_dot = np.asarray(x_dot, dtype=np.float64).reshape(-1)

        if x_dot.size != self.state_dim:
            raise ValueError(
                f"f_nonlinear2 must return derivative with shape "
                f"({self.state_dim},), but returned {x_dot.shape}."
            )

        # Process/model disturbance on modal accelerations (before integration).
        if self.dynamics_uncertainty_std > 0.0 and self._rng is not None:
            x_dot = x_dot.copy()
            x_dot[1::2] += (
                self.dynamics_uncertainty_std
                * self._rng.standard_normal(self.K)
            )

        return x_dot

    def reset(
        self,
        rng,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        options = options or {}
        if "y0" in options:
            self.set_milling_start_y(float(options["y0"]))
        elif self.randomize_y0 and self._rng is not None:
            y0 = float(self._rng.uniform(self.y0_min, self.y0_max))
            self.set_milling_start_y(y0)

        f_nonlinear2.reset_state_history()
        self._last_omega = float(self.omega_min)

        eta0 = rng.uniform(0, 0, size=self.K)
        eta_dot0 = rng.uniform(0, 0, size=self.K)

        x0 = np.zeros(self.state_dim, dtype=np.float64)
        x0[0::2] = eta0
        x0[1::2] = eta_dot0

        return x0, {"y0": float(self.y_cutter)}

    def termination(self, t: float, x: np.ndarray) -> tuple[bool, bool, dict[str, Any]]:
        x = np.asarray(x, dtype=np.float64).reshape(-1)

        invalid_state = not np.all(np.isfinite(x))
        w_sensor, wdot_sensor = self.modal_to_physical(x)
        excessive_displacement = np.any(np.abs(w_sensor) > self.w_limit)
        excessive_velocity = np.any(np.abs(wdot_sensor) > self.wdot_limit)

        info: dict[str, Any] = {}

        if invalid_state:
            info["termination_reason"] = "invalid_state"
            return True, False, info

        if excessive_displacement:
            info["termination_reason"] = "excessive_sensor_displacement"
            return True, False, info

        if excessive_velocity:
            info["termination_reason"] = "excessive_sensor_velocity"
            return True, False, info

        cutter_x, cutter_y = f_nonlinear2.cutter_point(t, self._last_omega)
        info["cutter_x"] = float(cutter_x)
        info["cutter_y"] = float(cutter_y)
        info["feed_progress"] = float(1.0 - cutter_x / max(self.L1, 1e-12))

        if cutter_x <= self.x_pass_end_tol:
            info["pass_completed"] = True
            info["termination_reason"] = "pass_completed"
            return True, False, info

        return False, False, info

    def get_observation_space(self) -> spaces.Space:
        low = np.full(self.obs_dim, -np.inf, dtype=np.float64)
        high = np.full(self.obs_dim, np.inf, dtype=np.float64)

        low[: self.n_sensors] = -self.w_limit / self.w_obs_scale
        high[: self.n_sensors] = self.w_limit / self.w_obs_scale

        low[self.n_sensors :] = -self.wdot_limit / self.wdot_obs_scale
        high[self.n_sensors :] = self.wdot_limit / self.wdot_obs_scale

        return spaces.Box(
            low=low,
            high=high,
            shape=(self.obs_dim,),
            dtype=np.float64,
        )

    def get_action_space(self) -> spaces.Space:
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float64,
        )

    def state_to_obs(self, x: np.ndarray) -> np.ndarray:
        w_sensor, wdot_sensor = self.modal_to_physical(x)

        return np.concatenate(
            [
                w_sensor / self.w_obs_scale,
                wdot_sensor / self.wdot_obs_scale,
            ],
            dtype=np.float64,
        )
