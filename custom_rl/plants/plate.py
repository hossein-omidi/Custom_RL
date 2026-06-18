"""Nonlinear plate vibration ODE plant."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from custom_rl.plants.base import ODEPlant
from custom_rl.plants import f_nonlinear2
from custom_rl.plants.compute_mode_shapes import ModeShapeBasis, compute_mode_shapes
from custom_rl.plants.compute_natural_frequencies import compute_natural_frequencies
from custom_rl.plants.compute_nonlinear_stiffness import compute_nonlinear_stiffness
from custom_rl.plants.compute_natural_frequencies_prim import compute_natural_frequencies_prim
from custom_rl.plants.modal_state import split_modal_state, state_dim_for_model
from custom_rl.plants.mode_ordering import (
    build_sensor_displacement_matrix,
    flatten_mode_matrix,
    mode_index_map,
)
from custom_rl.plants.milling_config import MillingForceConfig
from custom_rl.plants.units import FORCE_COEFFICIENT_SCALE, calibrated_cutting_coefficients
from custom_rl.plants.pass_schedule import (
    build_middle_line_trajectory,
    build_straight_pass_trajectory,
    pass_line_x_positions,
)
from custom_rl.plants.stochasticity import (
    GeometryNominal,
    GeometryUncertaintyConfig,
    ProcessNoiseConfig,
    SensorUncertaintyConfig,
    absolute_to_relative_sensor_coords,
    build_process_noise_vector,
    process_noise_info,
    sample_episode_geometry,
    sample_sensor_coords,
)


class PlatePlant(ODEPlant):
    """
    Nonlinear plate vibration plant for milling chatter RL.

    **Internal dynamics (modal, hidden from agent)**

    For each mode k:
        η̈_k + 2ζ_k ω_k η̇_k + ω_k² η_k + λ_k η_k³ = F_k(t)

    Cutting force projection (moving tool on plate surface):
        F_k(t) = W_k(x_tool(t), y_tool(t)) / M_modal · F_scalar(t; ω, a_c, path, Δw)

    **Physical sensor model (agent-facing)**

        w_s(t)     = Σ_k W_k(x_s, y_s) η_k(t)  = S_disp @ η
        ẇ_s(t)     = S_disp @ η̇

    **Normalization**

        w_s_norm     = w_s / disp_norm_scale
        ẇ_s_norm     = ẇ_s / vel_norm_scale
        u_phys       = u_low + 0.5(u_norm + 1)(u_high - u_low)

    **PPO interface**

        Observation: [w_s_norm..., ẇ_s_norm..., prev_u_ω, prev_u_ac]
        Action:      [u_ω, u_ac] ∈ [-1, 1]²  →  physical [ω rad/s, a_c mm]
    """

    def __init__(
        self,
        N: int = 5,
        L1: float = 1.0,
        L2: float = 0.5,
        h: float = 0.02,
        E: float = 70e9,
        nu: float = 0.3,
        rho: float = 7850.0,
        m_max: int = 2,
        n_max: int = 2,
        omega_min: float = 50,
        omega_max: float = 2000.0,
        ac_min: float = 0,
        ac_max: float = 10.0,
        # Legacy modal safety (optional internal numerical guard)
        eta_limit: float = 0.01,
        eta_dot_limit: float = 1.0,
        eta_obs_limit: float = 1e6,  # unused for the PPO observation now
        eta_dot_obs_limit: float = 1e6,  # unused for the PPO observation now
        use_eta_internal_safety: bool = False,
        # Sensor model configuration
        sensor_coords: list[tuple[float, float]] | None = None,
        disp_norm_scale: float = 1e-3,
        vel_norm_scale: float = 1e-2,
        obs_clip: float | None = 10.0,
        displacement_failure_limit: float = 1e-3,
        velocity_failure_limit: float = 0.5,
        # Straight-line full-surface pass schedule (one pass per RL episode)
        n_pass_lines: int = 100,
        pass_margin: float = 0.01,
        feed_speed: float = 0.05,
        traj_dt: float = 0.02,
        pass_sampling: str = "random",
        pass_complete_tolerance: float = 1e-6,
        # Structured uncertainty (see custom_rl.plants.stochasticity)
        enable_geometry_uncertainty: bool = True,
        enable_sensor_uncertainty: bool = True,
        enable_process_noise: bool = True,
        geometry_uncertainty: GeometryUncertaintyConfig | None = None,
        sensor_uncertainty: SensorUncertaintyConfig | None = None,
        process_noise: ProcessNoiseConfig | None = None,
        sensor_coords_relative: list[tuple[float, float]] | None = None,
        # Directional milling force (see milling_config.py)
        milling_type: str = "surface",
        phi_st: float | None = None,
        phi_ex: float | None = None,
        phi_0: float = 0.0,
        cutter_diameter: float = 0.02,
        helix_angle: float = 0.0,
        axial_quadrature_points: int = 5,
        ac_via_axial_integration: bool = True,
        ac_units: str = "mm",
        displacement_model: str = "feed_normal_full",
        feed_per_tooth_source: str = "from_feed_speed",
        cf_units: str = "m_per_tooth",
        trajectory_mode: str = "middle_line",
        path_x_mid: float | None = None,
        path_y_start: float | None = None,
        path_y_end: float = 0.0,
    ):
        """
        Args:
            N: number of teeth / force-related model parameter used by f_nonlinear2
            L1: plate length in x direction
            L2: plate length in y direction
            h: plate thickness
            E: Young's modulus
            nu: Poisson's ratio
            rho: density
            m_max: maximum mode index in x direction
            n_max: maximum mode index in y direction
            omega_min: minimum physical spindle/angular input
            omega_max: maximum physical spindle/angular input
            ac_min: minimum physical axial depth of cut (mm by default; see ac_units)
            ac_max: maximum physical axial depth of cut (mm by default; see ac_units)
            eta_limit: optional internal modal safety limit (not used for main termination)
            eta_obs_limit: legacy unused observation bound
            eta_dot_obs_limit: legacy unused observation bound
            use_eta_internal_safety: if True, also terminate on |eta_k| > eta_limit
            sensor_coords: list of (x_s, y_s) sensor positions in the plate plane
            disp_norm_scale: normalization scale for sensor displacement in obs/reward
            vel_norm_scale: normalization scale for sensor velocity in obs/reward
            obs_clip: clip normalized sensor obs to [-obs_clip, obs_clip] (None disables)
            displacement_failure_limit: physical termination limit on |w_sensors| (m)
            velocity_failure_limit: physical termination limit on |w_dot_sensors| (m/s)
            n_pass_lines: number of parallel straight passes across plate width
            pass_margin: edge margin for pass-line x grid (m)
            feed_speed: tool feed along pass from upper to lower edge (m/s)
            traj_dt: sampling period for reference tool trajectory (s)
            pass_sampling: "random" or "sequential" pass-line selection each reset
            pass_complete_tolerance: time tolerance for pass-completion truncation (s)
            enable_geometry_uncertainty: sample L1,L2,h,E,rho once per episode
            enable_sensor_uncertainty: jitter sensor relative positions each reset
            enable_process_noise: diagonal modal noise after RK4 (unmodeled disturbance)
            geometry_uncertainty: optional GeometryUncertaintyConfig override
            sensor_uncertainty: optional SensorUncertaintyConfig override
            process_noise: optional ProcessNoiseConfig override
            sensor_coords_relative: optional (x/L1, y/L2) nominal sensor layout
        """
        self.N = N

        self._geometry_nominal = GeometryNominal(
            L1=float(L1),
            L2=float(L2),
            h=float(h),
            E=float(E),
            nu=float(nu),
            rho=float(rho),
        )

        geom_cfg = geometry_uncertainty or GeometryUncertaintyConfig()
        self.geometry_uncertainty = GeometryUncertaintyConfig(
            enable=bool(enable_geometry_uncertainty and geom_cfg.enable),
            rel_std_L1=geom_cfg.rel_std_L1,
            rel_std_L2=geom_cfg.rel_std_L2,
            rel_std_h=geom_cfg.rel_std_h,
            rel_std_E=geom_cfg.rel_std_E,
            rel_std_rho=geom_cfg.rel_std_rho,
            max_rel_deviation=geom_cfg.max_rel_deviation,
        )

        if sensor_coords_relative is not None:
            rel_positions = tuple(
                (float(x), float(y)) for x, y in sensor_coords_relative
            )
        elif sensor_coords is not None:
            coords_tmp = np.asarray(sensor_coords, dtype=np.float64)
            rel_positions = absolute_to_relative_sensor_coords(
                coords_tmp, L1, L2
            )
        else:
            rel_positions = SensorUncertaintyConfig().rel_positions

        sensor_cfg = sensor_uncertainty or SensorUncertaintyConfig()
        self.sensor_uncertainty = SensorUncertaintyConfig(
            enable=bool(enable_sensor_uncertainty and sensor_cfg.enable),
            rel_positions=rel_positions,
            rel_jitter_std=sensor_cfg.rel_jitter_std,
            edge_margin_frac=sensor_cfg.edge_margin_frac,
        )

        proc_cfg = process_noise or ProcessNoiseConfig()
        self.process_noise = ProcessNoiseConfig(
            enable=bool(enable_process_noise and proc_cfg.enable),
            eta_std_per_sqrt_s=proc_cfg.eta_std_per_sqrt_s,
            eta_dot_std_per_sqrt_s=proc_cfg.eta_dot_std_per_sqrt_s,
            clip_sigma=proc_cfg.clip_sigma,
        )

        self.L1 = float(L1)
        self.L2 = float(L2)
        self.h = float(h)
        self.E = float(E)
        self.nu = float(nu)
        self.rho = float(rho)

        self.m_max = m_max
        self.n_max = n_max
        self.K = self.m_max * self.n_max

        self.omega_min = omega_min
        self.omega_max = omega_max
        self.ac_min = ac_min
        self.ac_max = ac_max

        self.eta_limit = eta_limit
        self.eta_dot_limit = float(eta_dot_limit)
        self.use_eta_internal_safety = bool(use_eta_internal_safety)
        self.eta_obs_limit = eta_obs_limit
        self.eta_dot_obs_limit = eta_dot_obs_limit

        if sensor_coords is not None:
            coords = np.asarray(sensor_coords, dtype=np.float64)
            if coords.ndim != 2 or coords.shape[1] != 2:
                raise ValueError(
                    f"sensor_coords must have shape (n_sensors, 2), got {coords.shape}."
                )
            if np.any(coords[:, 0] < 0.0) or np.any(coords[:, 0] > self.L1):
                raise ValueError("All sensor x-coordinates must satisfy 0 <= x_s <= L1.")
            if np.any(coords[:, 1] < 0.0) or np.any(coords[:, 1] > self.L2):
                raise ValueError("All sensor y-coordinates must satisfy 0 <= y_s <= L2.")
            self.sensor_coords = coords
        else:
            self.sensor_coords, _ = sample_sensor_coords(
                np.random.default_rng(0),
                self.L1,
                self.L2,
                SensorUncertaintyConfig(
                    enable=False,
                    rel_positions=rel_positions,
                ),
            )

        self.n_sensors = int(self.sensor_coords.shape[0])
        self._episode_geometry: dict[str, float] = {}
        self._episode_sensor_info: dict[str, Any] = {}

        self.disp_norm_scale = float(disp_norm_scale)
        self.vel_norm_scale = float(vel_norm_scale)
        if self.disp_norm_scale <= 0.0 or self.vel_norm_scale <= 0.0:
            raise ValueError("disp_norm_scale and vel_norm_scale must be > 0.")

        self.obs_clip = obs_clip if obs_clip is None else float(obs_clip)
        self.displacement_failure_limit = float(displacement_failure_limit)
        self.velocity_failure_limit = float(velocity_failure_limit)

        if self.velocity_failure_limit <= 0.0:
            raise ValueError("velocity_failure_limit must be > 0.")

        self.n_pass_lines = int(n_pass_lines)
        if self.n_pass_lines < 1:
            raise ValueError("n_pass_lines must be >= 1.")

        self.pass_margin = float(pass_margin)
        self.feed_speed = float(feed_speed)
        self.traj_dt = float(traj_dt)
        if self.feed_speed <= 0.0 or self.traj_dt <= 0.0:
            raise ValueError("feed_speed and traj_dt must be > 0.")

        if pass_sampling not in {"random", "sequential"}:
            raise ValueError('pass_sampling must be "random" or "sequential".')
        self.pass_sampling = pass_sampling
        self.pass_complete_tolerance = float(pass_complete_tolerance)

        self.milling_config = MillingForceConfig(
            milling_type=milling_type,
            phi_st=phi_st,
            phi_ex=phi_ex,
            phi_0=phi_0,
            cutter_diameter=cutter_diameter,
            helix_angle=helix_angle,
            axial_quadrature_points=axial_quadrature_points,
            ac_via_axial_integration=ac_via_axial_integration,
            ac_units=ac_units,
            displacement_model=displacement_model,
            feed_per_tooth_source=feed_per_tooth_source,
            cf_units=cf_units,
        )

        if trajectory_mode not in {"middle_line", "pass_grid"}:
            raise ValueError('trajectory_mode must be "middle_line" or "pass_grid".')
        self.trajectory_mode = trajectory_mode
        self.path_x_mid = path_x_mid
        self.path_y_start = path_y_start
        self.path_y_end = float(path_y_end)

        self.state_dim = state_dim_for_model(self.K, self.milling_config)

        self._pass_counter = 0
        self.current_pass_line_index = 0
        self.current_pass_x = 0.0
        self.pass_duration = 0.0
        self.pass_x_positions = np.array([], dtype=np.float64)

        self.t_original = np.array([0.0], dtype=np.float64)
        self.x_traj = np.array([0.0], dtype=np.float64)
        self.y_traj = np.array([0.0], dtype=np.float64)
        self._modal_state_history: list[tuple[float, np.ndarray]] = []

        self.u_phys_low = np.array(
            [self.omega_min, self.ac_min],
            dtype=np.float64,
        )
        self.u_phys_high = np.array(
            [self.omega_max, self.ac_max],
            dtype=np.float64,
        )

        self._rebuild_modal_physics()
        self._rebuild_pass_grid()
        self._configure_pass_line(0)

    def _rebuild_pass_grid(self) -> None:
        """Rebuild pass-line x grid for current episode geometry L1."""
        self.pass_x_positions = pass_line_x_positions(
            self.n_pass_lines,
            self.L1,
            margin=self.pass_margin,
        )

    def _rebuild_sensor_matrix(self) -> None:
        """Rebuild S_disp from current mode shapes and sensor coordinates."""
        self.S_disp = build_sensor_displacement_matrix(
            self.W_mn,
            self.sensor_coords,
            self.m_max,
            self.n_max,
            mode_basis=self.mode_basis,
        )

    def _rebuild_modal_physics(self) -> None:
        """
        Recompute modal parameters and sync f_nonlinear2 for current geometry.

        Called at construction and whenever episode-constant Θ_geom is resampled.
        """
        M_modal = self.L1 * self.L2 * self.rho * self.h
        self.M_modal = M_modal

        W_mn, V_mn = compute_mode_shapes(
            self.L1,
            self.L2,
            self.h,
            self.m_max,
            self.n_max,
        )
        self.mode_basis = ModeShapeBasis.build(
            self.L1,
            self.L2,
            self.h,
            self.m_max,
            self.n_max,
        )
        self.W_mn = W_mn
        self.V_mn = V_mn

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

        lambda_mn, lambda_prime_mn = compute_nonlinear_stiffness(
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

        omega_prime_mn = compute_natural_frequencies_prim(
            self.E,
            self.nu,
            self.rho,
            self.h,
            self.L1,
            self.L2,
            self.m_max,
            self.n_max,
        )

        omega_vec = flatten_mode_matrix(omega_mn, self.m_max, self.n_max)
        lambda_vec = flatten_mode_matrix(lambda_mn, self.m_max, self.n_max)
        omega_f_vec = flatten_mode_matrix(omega_prime_mn, self.m_max, self.n_max)
        lambda_f_vec = flatten_mode_matrix(lambda_prime_mn, self.m_max, self.n_max)
        self.omega_vec = omega_vec
        self.lambda_vec = lambda_vec
        self.omega_f_vec = omega_f_vec
        self.lambda_f_vec = lambda_f_vec
        self.mode_index_map = mode_index_map(self.m_max, self.n_max)

        zeta_vec = 0.05 * np.ones(self.K, dtype=np.float64)
        zeta_f_vec = 0.05 * np.ones(self.K, dtype=np.float64)
        cf = 0.3

        z_contact = float(self.h if self.milling_config.z_contact is None else self.milling_config.z_contact)
        self.z_contact = z_contact
        self.milling_config.z_contact = z_contact

        M_modal_f = M_modal
        self.M_modal_f = M_modal_f

        xi_base, delta_base = calibrated_cutting_coefficients()
        self.force_coefficient_scale = FORCE_COEFFICIENT_SCALE
        self.xi_base = xi_base
        self.delta_base = delta_base

        f_nonlinear2.m_max = self.m_max
        f_nonlinear2.n_max = self.n_max
        f_nonlinear2.N = self.N
        f_nonlinear2.K = self.K
        f_nonlinear2.zeta_vec = zeta_vec
        f_nonlinear2.lambda_vec = lambda_vec
        f_nonlinear2.omega_vec = omega_vec
        f_nonlinear2.zeta_f_vec = zeta_f_vec
        f_nonlinear2.lambda_f_vec = lambda_f_vec
        f_nonlinear2.omega_f_vec = omega_f_vec
        f_nonlinear2.xi_base = xi_base
        f_nonlinear2.delta_base = delta_base
        f_nonlinear2.W_mn = W_mn
        f_nonlinear2.V_mn = V_mn
        f_nonlinear2.mode_basis = self.mode_basis
        f_nonlinear2.cf = cf
        f_nonlinear2.t_original = self.t_original
        f_nonlinear2.x_traj = self.x_traj
        f_nonlinear2.y_traj = self.y_traj
        f_nonlinear2.feed_speed = self.feed_speed
        f_nonlinear2.M_modal = M_modal
        f_nonlinear2.M_modal_f = M_modal_f
        f_nonlinear2.z_contact = z_contact
        f_nonlinear2.milling_cfg = self.milling_config
        f_nonlinear2.decimal_places = 1

        self.state_dim = state_dim_for_model(self.K, self.milling_config)

        self._rebuild_sensor_matrix()

    def apply_process_noise(
        self,
        x: np.ndarray,
        rng: np.random.Generator,
        step_dt: float,
    ) -> np.ndarray:
        """
        Add bounded diagonal modal process noise after deterministic integration.

        x_{k+1} = Φ(x_k, u_k) + w_k,  w_k ~ N(0, Q(Δt)),  Q ∝ Δt I (modal blocks).
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        noise = build_process_noise_vector(
            rng,
            self.state_dim,
            self.K,
            step_dt,
            self.process_noise,
        )
        return x + noise

    def get_process_noise_info(self, step_dt: float) -> dict[str, Any]:
        """Return process-noise metadata for logging and Monte Carlo analysis."""
        return process_noise_info(self.process_noise, step_dt)

    def _configure_pass_line(self, line_index: int) -> dict[str, Any]:
        """Install reference tool trajectory for the episode."""
        if self.trajectory_mode == "middle_line":
            x_mid = float(self.path_x_mid if self.path_x_mid is not None else 0.5 * self.L1)
            y_start = float(self.path_y_start if self.path_y_start is not None else self.L2)
            y_end = float(self.path_y_end)
            if not (0.0 <= x_mid <= self.L1):
                raise ValueError(f"path x_mid={x_mid} outside [0, L1={self.L1}].")
            if not (0.0 <= y_start <= self.L2) or not (0.0 <= y_end <= self.L2):
                raise ValueError("path y_start/y_end must lie in [0, L2].")
            t_original, x_traj, y_traj, pass_duration = build_middle_line_trajectory(
                x_mid=x_mid,
                y_start=y_start,
                y_end=y_end,
                feed_speed=self.feed_speed,
                dt=self.traj_dt,
            )
            self.current_pass_line_index = 0
            self.current_pass_x = x_mid
            pass_info = {
                "pass_line_index": 0,
                "pass_x": x_mid,
                "pass_duration": pass_duration,
                "trajectory_mode": "middle_line",
                "path_x_mid": x_mid,
                "path_y_start": y_start,
                "path_y_end": y_end,
                "feed_speed": self.feed_speed,
            }
        else:
            if line_index < 0 or line_index >= self.n_pass_lines:
                raise ValueError(
                    f"line_index must be in [0, {self.n_pass_lines}), got {line_index}."
                )
            self.current_pass_line_index = int(line_index)
            self.current_pass_x = float(self.pass_x_positions[line_index])
            t_original, x_traj, y_traj, pass_duration = build_straight_pass_trajectory(
                x_line=self.current_pass_x,
                L2=self.L2,
                feed_speed=self.feed_speed,
                dt=self.traj_dt,
                y_start=self.L2,
                y_end=0.0,
            )
            pass_info = {
                "pass_line_index": self.current_pass_line_index,
                "pass_x": self.current_pass_x,
                "pass_duration": pass_duration,
                "trajectory_mode": "pass_grid",
                "n_pass_lines": self.n_pass_lines,
                "feed_speed": self.feed_speed,
                "y_start": float(self.L2),
                "y_end": 0.0,
            }

        self.t_original = t_original
        self.x_traj = x_traj
        self.y_traj = y_traj
        self.pass_duration = float(pass_duration)

        f_nonlinear2.t_original = self.t_original
        f_nonlinear2.x_traj = self.x_traj
        f_nonlinear2.y_traj = self.y_traj
        f_nonlinear2.reset_episode_state(self._modal_state_history)

        return pass_info

    def recommended_max_episode_steps(self, step_dt: float, safety_margin: int = 10) -> int:
        """Episode horizon covering one full pass plus a small safety margin."""
        if step_dt <= 0.0:
            raise ValueError("step_dt must be > 0.")
        if self.pass_duration <= 0.0:
            return max(safety_margin, 1)
        return int(np.ceil(self.pass_duration / step_dt)) + int(safety_margin)

    def _scale_action(self, u: np.ndarray) -> np.ndarray:
        """
        Map normalized action from [-1, 1]^2 to physical action [omega, ac].

        Affine map (SB3-compatible):
            u_phys = u_low + 0.5 * (u_norm + 1) * (u_high - u_low)
        """
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

    def normalized_to_physical_action(self, u_norm: np.ndarray) -> np.ndarray:
        """Public alias for action denormalization (agent → plant)."""
        return self._scale_action(u_norm)

    def physical_to_normalized_action(self, u_phys: np.ndarray) -> np.ndarray:
        """
        Inverse affine map: physical [omega, ac] → normalized [-1, 1]^2.

        Used for logging, plotting, and verification of the action interface.
        """
        u_phys = np.asarray(u_phys, dtype=np.float64).reshape(-1)
        if u_phys.size != 2:
            raise ValueError(f"Expected physical action shape (2,), got {u_phys.shape}.")

        span = self.u_phys_high - self.u_phys_low
        u_norm = 2.0 * (u_phys - self.u_phys_low) / np.maximum(span, 1e-12) - 1.0
        return np.clip(u_norm, -1.0, 1.0).astype(np.float64, copy=False)

    def sensor_obs_norm_to_physical(
        self,
        sensor_obs_norm: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Denormalize sensor observation block to physical (w_s, w_dot_s) in SI units.

        Inverse of state_to_sensor_obs_norm (without clipping).
        """
        sensor_obs_norm = np.asarray(sensor_obs_norm, dtype=np.float64).reshape(-1)
        n = self.n_sensors
        if sensor_obs_norm.size < 2 * n:
            raise ValueError(
                f"sensor_obs_norm must have length >= {2 * n}, got {sensor_obs_norm.size}."
            )
        w = sensor_obs_norm[:n] * self.disp_norm_scale
        w_dot = sensor_obs_norm[n : 2 * n] * self.vel_norm_scale
        return w.astype(np.float64, copy=False), w_dot.astype(np.float64, copy=False)

    def get_interface_metadata(self) -> dict[str, Any]:
        """
        Contract summary for agent ↔ environment interface (paper / debugging).

        Internal state: modal η (hidden from agent).
        Agent observation: normalized physical sensors + previous normalized action.
        Agent action: normalized [u_ω, u_ac] ∈ [-1,1]².
        """
        return {
            "state_representation": "modal_eta_internal",
            "observation_layout": [
                "w_s1_norm",
                "w_s2_norm",
                "...",
                "w_dot_s1_norm",
                "w_dot_s2_norm",
                "...",
                "prev_u_omega_norm",
                "prev_u_ac_norm",
            ],
            "action_layout": ["u_omega_norm", "u_ac_norm"],
            "sensor_coords": self.sensor_coords.tolist(),
            "disp_norm_scale_m": float(self.disp_norm_scale),
            "vel_norm_scale_m_s": float(self.vel_norm_scale),
            "physical_action_low": self.u_phys_low.tolist(),
            "physical_action_high": self.u_phys_high.tolist(),
            "physical_action_units": ["rad/s", self.milling_config.ac_units],
            "displacement_failure_limit_m": float(self.displacement_failure_limit),
            "velocity_failure_limit_m_s": float(self.velocity_failure_limit),
            "n_modal_modes": int(self.K),
            "n_modal_subsystems": 2 if self.milling_config.is_feed_normal_full() else 1,
            "state_dim": int(self.state_dim),
            "trajectory_mode": self.trajectory_mode,
            "n_sensors": int(self.n_sensors),
            "milling_force": self.milling_config.to_dict(),
        }

    def tool_position_at(self, t: float) -> tuple[float, float]:
        """Reference cutter position (x, y) on the plate at simulation time t."""
        x_c = float(np.interp(float(t), self.t_original, self.x_traj))
        y_c = float(np.interp(float(t), self.t_original, self.y_traj))
        return x_c, y_c

    def record_modal_state(
        self,
        t: float,
        x: np.ndarray,
        omega: float | None = None,
    ) -> None:
        """Store accepted modal state and spindle speed for regenerative delay."""
        f_nonlinear2.record_modal_state(t, x, history=self._modal_state_history)
        if omega is not None:
            f_nonlinear2.record_omega(t, float(omega))

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        State derivative: dx/dt = dynamics(t, x, u).

        The environment supplies normalized action in [-1, 1]^2.
        This method scales it to physical [omega rad/s, ac mm] before calling f_nonlinear2.
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1)

        if x.size != self.state_dim:
            raise ValueError(
                f"PlatePlant expects state with shape ({self.state_dim},), "
                f"but received {x.shape}."
            )

        u_phys = self._scale_action(u)

        f_nonlinear2.bind_modal_history(self._modal_state_history)
        try:
            x_dot = f_nonlinear2.f_nonlinear2(t, x, u_phys)
        finally:
            f_nonlinear2.unbind_modal_history()
        x_dot = np.asarray(x_dot, dtype=np.float64).reshape(-1)

        if x_dot.size != self.state_dim:
            raise ValueError(
                f"f_nonlinear2 must return derivative with shape "
                f"({self.state_dim},), but returned {x_dot.shape}."
            )

        return x_dot

    def reset(self, rng) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Start a new straight-line pass episode with structured uncertainty.

        Episode-constant (until termination):
            Θ_geom = (L1, L2, h, E, ρ) — geometry/material scatter

        Per-reset layout (fixed within episode):
            pass line x-position, sensor coordinates on sampled plate

        Initial condition: stationary plate (η = 0, η̇ = 0).
        """
        geom = sample_episode_geometry(
            rng,
            self._geometry_nominal,
            self.geometry_uncertainty,
        )
        self._episode_geometry = geom
        self.L1 = geom["L1"]
        self.L2 = geom["L2"]
        self.h = geom["h"]
        self.E = geom["E"]
        self.rho = geom["rho"]

        self._rebuild_modal_physics()
        self._rebuild_pass_grid()

        sensor_coords, sensor_info = sample_sensor_coords(
            rng,
            self.L1,
            self.L2,
            self.sensor_uncertainty,
        )
        self.sensor_coords = sensor_coords
        self._episode_sensor_info = sensor_info
        self._rebuild_sensor_matrix()

        if self.pass_sampling == "random":
            line_index = int(rng.integers(0, self.n_pass_lines))
        else:
            line_index = int(self._pass_counter % self.n_pass_lines)
            self._pass_counter += 1

        pass_info = self._configure_pass_line(line_index)

        x0 = np.zeros(self.state_dim, dtype=np.float64)

        info: dict[str, Any] = {
            **geom,
            **sensor_info,
            **pass_info,
            "sensor_coords": self.sensor_coords.tolist(),
            "pass_sampling": self.pass_sampling,
            "geometry_uncertainty_enabled": self.geometry_uncertainty.enable,
            "sensor_uncertainty_enabled": self.sensor_uncertainty.enable,
            "process_noise_enabled": self.process_noise.enable,
        }
        return x0, info

    def termination(self, t: float, x: np.ndarray) -> tuple[bool, bool, dict[str, Any]]:
        """
        Episode end conditions for one straight pass.

        Truncation (non-failure):
            - pass_complete: tool reached the lower edge / pass duration elapsed

        Termination (failure):
            - invalid_state: NaN/Inf in modal state
            - excessive_sensor_displacement: |w_s| above physical limit
            - excessive_sensor_velocity: |w_dot_s| above physical limit
            - excessive_modal_displacement_internal: optional modal guard
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        invalid_state = not np.all(np.isfinite(x))

        info: dict[str, Any] = {
            "pass_line_index": self.current_pass_line_index,
            "pass_x": self.current_pass_x,
            "pass_duration": self.pass_duration,
            "sim_time": float(t),
        }

        # Successful completion of the milling pass (one grid line).
        if t >= self.pass_duration - self.pass_complete_tolerance:
            info["termination_reason"] = "pass_complete"
            return False, True, info

        w_sensors, w_dot_sensors = self.state_to_sensor_signals(x)

        excessive_sensor_displacement = bool(
            np.any(np.abs(w_sensors) > self.displacement_failure_limit)
        )
        excessive_sensor_velocity = bool(
            np.any(np.abs(w_dot_sensors) > self.velocity_failure_limit)
        )

        eta_n, eta_dot_n, _, _ = split_modal_state(
            x,
            self.K,
            two_field=self.milling_config.is_feed_normal_full(),
        )

        excessive_modal_displacement = bool(np.any(np.abs(eta_n) > self.eta_limit))
        excessive_modal_velocity = bool(np.any(np.abs(eta_dot_n) > self.eta_dot_limit))
        excessive_internal_modal = self.use_eta_internal_safety and (
            excessive_modal_displacement or excessive_modal_velocity
        )

        if invalid_state:
            info["termination_reason"] = "invalid_state"
            return True, False, info

        if excessive_sensor_displacement:
            info["termination_reason"] = "excessive_sensor_displacement"
            info["max_abs_sensor_displacement"] = float(np.max(np.abs(w_sensors)))
            return True, False, info

        if excessive_sensor_velocity:
            info["termination_reason"] = "excessive_sensor_velocity"
            info["max_abs_sensor_velocity"] = float(np.max(np.abs(w_dot_sensors)))
            return True, False, info

        if excessive_internal_modal:
            info["termination_reason"] = "excessive_modal_state_internal"
            return True, False, info

        return False, False, info

    def get_observation_space(self) -> spaces.Space:
        """
        Return Gymnasium observation space for the sensor-based PPO inputs.

        Note: this space covers *only* the sensor part.
        The env will append previous action commands.

        Sensor ordering (normalized):
            [w_s1_norm, w_s2_norm, ..., w_sN_norm,
             w_dot_s1_norm, w_dot_s2_norm, ..., w_dot_sN_norm]
        """
        if self.obs_clip is None:
            low = np.full((2 * self.n_sensors,), -np.inf, dtype=np.float64)
            high = np.full((2 * self.n_sensors,), np.inf, dtype=np.float64)
        else:
            low = np.full((2 * self.n_sensors,), -self.obs_clip, dtype=np.float64)
            high = np.full((2 * self.n_sensors,), self.obs_clip, dtype=np.float64)

        return spaces.Box(low=low, high=high, shape=(2 * self.n_sensors,), dtype=np.float64)

    def get_action_space(self) -> spaces.Space:
        """
        Return normalized Gymnasium action space.

        action[0]: normalized omega command
        action[1]: normalized ac command
        """
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float64,
        )

    def state_to_obs(self, x: np.ndarray) -> np.ndarray:
        """
        Map internal modal state to normalized sensor observation.
        """
        # For the agent observation, apply the optional clipping.
        return self.state_to_sensor_obs_norm(x, clip_for_observation=True)

    def state_to_sensor_obs_norm(
        self,
        x: np.ndarray,
        *,
        clip_for_observation: bool,
    ) -> np.ndarray:
        """
        Convert internal modal state to normalized sensor signals.

        clip_for_observation=True:
            Apply obs_clip to match the Gym observation bounds.
        clip_for_observation=False:
            Do not clip; intended for reward computation consistency.
        """
        w_sensors, w_dot_sensors = self.state_to_sensor_signals(x)

        w_norm = w_sensors / self.disp_norm_scale
        w_dot_norm = w_dot_sensors / self.vel_norm_scale

        if clip_for_observation and self.obs_clip is not None:
            w_norm = np.clip(w_norm, -self.obs_clip, self.obs_clip)
            w_dot_norm = np.clip(w_dot_norm, -self.obs_clip, self.obs_clip)

        # Ordering required:
        #   [w_s1, w_s2, ..., w_dot_s1, w_dot_s2, ...]
        return np.concatenate([w_norm, w_dot_norm]).astype(np.float64, copy=False)

    def state_to_sensor_signals(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct physical sensor signals from modal coordinates:
            w_sensors(t)      = S_disp @ eta(t)
            w_dot_sensors(t) = S_disp @ eta_dot(t)
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size != self.state_dim:
            raise ValueError(
                f"PlatePlant expects internal state shape ({self.state_dim},), got {x.shape}."
            )

        eta_n, eta_dot_n, _, _ = split_modal_state(
            x,
            self.K,
            two_field=self.milling_config.is_feed_normal_full(),
        )

        w_sensors = self.S_disp @ eta_n
        w_dot_sensors = self.S_disp @ eta_dot_n
        return w_sensors, w_dot_sensors