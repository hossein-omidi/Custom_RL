"""Face-milling flexible-plate ODE plant for RL environments.

This module replaces the old peripheral-milling plant wiring.  The structural
part remains modal, while the cutting-force module is now
``f_nonlinear2_face_milling`` and computes the face-milling force tooth-by-tooth:

    [Ft, Fr, Fa] -> [Fx, Fy, Fz] -> modal generalized force

Default geometric convention
----------------------------
The updated mode-shape module defaults to ``clamped_axis='x'``.  In that
convention the clamped side is x=0 and the free side is x=L1.  The milling pass
therefore starts at x=L1 and moves toward x=0, i.e. from the free side toward

the clamped side.  This is consistent with the current face-milling force
module's ``cutter_point`` function.

If your physical plate is clamped along y instead of x, the mode shapes and the
force-module path function must both be changed together.  Do not only change
one of them.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from gymnasium import spaces

from custom_rl.plants.base import ODEPlant
from custom_rl.plants import f_nonlinear2_face_milling as f_nonlinear2
from custom_rl.plants.compute_mode_shapes_updated import (
    build_mode_shape_matrix,
    compute_mode_shapes,
)
from custom_rl.plants.compute_natural_frequencies_updated import (
    compute_natural_frequencies,
)
from custom_rl.plants.compute_nonlinear_stiffness_updated import (
    compute_nonlinear_stiffness,
)


# ---------------------------------------------------------------------------
# Default sensor/observation locations [m].
# They are abstract displacement/velocity evaluation points, not necessarily
# physical accelerometer models.
# ---------------------------------------------------------------------------
DEFAULT_SENSOR_POINTS: tuple[tuple[float, float], ...] = (
    (0.83, 0.20),
    (0.83, 0.83),
)

# The face-milling force module currently supports a straight x-pass with
# constant y.  With clamped_axis='x', the pass starts at x=L1 and moves to x=0.
DEFAULT_Y_CUTTER = 0.20

# Spindle speed operating range [rpm]; internal physics uses rad/s.
RPM_MIN = 1000
RPM_MAX = 40000.0

# Practical reference speed for episode cap estimation.  This does not change
# pass termination; it only avoids extremely long RL rollout caps at very low rpm.
TRAINING_REFERENCE_RPM = 1000.0


def rpm_to_omega(rpm: float | np.ndarray) -> np.ndarray:
    """Convert spindle speed from rpm to rad/s."""
    return np.asarray(rpm, dtype=np.float64) * 2.0 * np.pi / 60.0


def omega_to_rpm(omega: float | np.ndarray) -> np.ndarray:
    """Convert spindle speed from rad/s to rpm."""
    return np.asarray(omega, dtype=np.float64) * 60.0 / (2.0 * np.pi)


OMEGA_MIN_RAD_S = float(rpm_to_omega(RPM_MIN))
OMEGA_MAX_RAD_S = float(rpm_to_omega(RPM_MAX))


def _resolve_areal_mass(rho: float, h: float, rho_type: str = "auto") -> tuple[float, str]:
    """Return areal mass [kg/m^2] from density or areal-mass input."""
    rho = float(rho)
    h = float(h)
    if not np.isfinite(rho) or rho <= 0.0:
        raise ValueError(f"rho must be positive and finite, got {rho!r}.")
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError(f"h must be positive and finite, got {h!r}.")

    kind = str(rho_type).lower().strip()
    if kind in {"volumetric", "density", "kg/m3", "kg/m^3"}:
        return rho * h, "volumetric"
    if kind in {"areal", "surface", "kg/m2", "kg/m^2"}:
        return rho, "areal"
    if kind == "auto":
        if rho > 500.0:
            return rho * h, "volumetric(auto)"
        return rho, "areal(auto)"
    raise ValueError("rho_type must be 'auto', 'volumetric', or 'areal'.")


def estimate_pass_duration(
    L1: float,
    feed_per_tooth_mm: float,
    n_teeth: int,
    omega: float,
) -> float:
    """Seconds to complete a straight free-side-to-clamped-side x-pass."""
    feed_m_s = (float(feed_per_tooth_mm) / 1000.0) * int(n_teeth) * float(omega) / (2.0 * np.pi)
    return float(L1) / max(feed_m_s, 1e-12)


def estimate_pass_episode_steps(
    L1: float,
    feed_per_tooth_mm: float,
    n_teeth: int,
    omega: float,
    step_dt: float,
    margin: float = 1.1,
) -> int:
    """Episode step budget from a pass duration estimate."""
    duration = estimate_pass_duration(L1, feed_per_tooth_mm, n_teeth, omega) * margin
    return int(np.ceil(duration / max(float(step_dt), 1e-12))) + 200


def estimate_training_episode_steps(
    L1: float,
    feed_per_tooth_mm: float,
    n_teeth: int,
    step_dt: float,
    *,
    reference_rpm: float = TRAINING_REFERENCE_RPM,
    margin: float = 1.35,
) -> int:
    """Practical RL episode cap from a typical pass duration."""
    omega_ref = float(rpm_to_omega(reference_rpm))
    return estimate_pass_episode_steps(
        L1=L1,
        feed_per_tooth_mm=feed_per_tooth_mm,
        n_teeth=n_teeth,
        omega=omega_ref,
        step_dt=step_dt,
        margin=margin,
    )




def _trapz_compat(y, x, *, axis: int):
    """NumPy-version-safe trapezoidal integration.

    np.trapezoid was introduced in newer NumPy versions.  Many Anaconda
    Python 3.12 installations still provide only np.trapz.  Keep one helper so
    the modal-mass calculation works on both versions without changing the
    theory: M_k = ∫∫ rho_areal W_k^2 dxdy.
    """
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return trapezoid(y, x=x, axis=axis)
    return np.trapz(y, x=x, axis=axis)

def compute_modal_mass_vector(
    W_mn,
    L1: float,
    L2: float,
    rho_areal: float,
    m_max: int,
    n_max: int,
    *,
    grid_points: int = 151,
) -> np.ndarray:
    """Compute true modal masses M_k = ∫∫ rho_areal * W_k(x,y)^2 dxdy.

    The analytical shape functions are close to mass-normalized on [0, 1]^2,
    so this often returns values close to the total plate mass.  Computing the
    vector explicitly keeps the modal force projection rigorous and safe if the
    mode-shape normalization changes later.
    """
    L1 = float(L1)
    L2 = float(L2)
    rho_areal = float(rho_areal)
    m_max = int(m_max)
    n_max = int(n_max)
    grid_points = max(int(grid_points), 25)

    xs = np.linspace(0.0, L1, grid_points, dtype=np.float64)
    ys = np.linspace(0.0, L2, grid_points, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    masses = np.zeros(m_max * n_max, dtype=np.float64)
    k = 0
    for m in range(m_max):
        for n in range(n_max):
            W = np.asarray(W_mn[m][n](X, Y), dtype=np.float64)
            W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
            integrand = rho_areal * W * W
            int_y = _trapz_compat(integrand, ys, axis=1)
            masses[k] = float(_trapz_compat(int_y, xs, axis=0))
            k += 1

    masses = np.nan_to_num(masses, nan=0.0, posinf=0.0, neginf=0.0)
    masses = np.where(masses > 1e-18, masses, 1e-18)
    return masses


class PlatePlant(ODEPlant):
    """Nonlinear flexible-plate plant driven by a face-milling force model.

    Internal modal state
        x_modal = [eta1, eta1_dot, eta2, eta2_dot, ..., etaK, etaK_dot]

    Agent observation
        [
            w_sensor / w_obs_scale,
            wdot_sensor / wdot_obs_scale,
            cutter_x / L1,
            cutter_y / L2,
        ]

    Default normalized action
        u = [u_omega, u_ap] in [-1, 1]^2

    Physical action after scaling
        [omega_rad_s, ap_mm]

    Optional action if ``control_ae=True``
        u = [u_omega, u_ap, u_ae]
        physical = [omega_rad_s, ap_mm, ae_mm]
    """

    def __init__(
        self,
        N: int = 4,
        L1: float = 1.0,
        L2: float = 1.0,
        h: float = 0.02,
        E: float = 71.7e9,
        nu: float = 0.33,
        rho: float = 2810.0,
        rho_type: str = "volumetric",
        m_max: int = 2,
        n_max: int = 3,
        mode_clamped_axis: str = "x",
        omega_min: float = OMEGA_MIN_RAD_S,
        omega_max: float = OMEGA_MAX_RAD_S,
        ap_min: float = 0.0,
        ap_max: float = 18.0,
        ae_min: float = 1.0,
        ae_max: float = 50.0,
        ae_default: float = 28.0,
        control_ae: bool = False,
        control_ap: bool = True,
        randomize_ap: bool = False,
        ap_fixed: float | None = None,
        D_mm: float = 63.0,
        feed_per_tooth_mm: float = 0.20,
        gamma_L_deg: float = 45.0,
        gamma_r_deg: float = 5.0,
        gamma_a_deg: float = 5.0,
        eta_c_deg: float = 0.0,
        # AL7075 face-milling coefficients for square inserts.
        # Units follow f_nonlinear2_face_milling:
        #   Kt, Kr, Ka     [N/mm^2]
        #   Kte, Kre, Kae  [N/mm]
        # The signs are kept from the identified local force convention.
        # With the current scalar plate projection mode, only Fz=Fa is projected
        # into the modal plate equation, so Ka/Kae directly set the transverse
        # excitation direction and magnitude.
        Kt: float = 538.127,
        Kr: float = 185.967,
        Ka: float = -691.297,
        Kte: float = 11.253,
        Kre: float = 6.991,
        Kae: float = -32.971,
        milling_mode: str = "up",
        theta0: float = 0.0,
        use_process_damping: bool = False,
        Ksp: float | None = None,
        mu: float = 0.3,
        VB: float = 0.0,
        lambda_L_deg: float | None = None,
        force_projection_mode: str = "z",
        sensor_points: Sequence[tuple[float, float]] = DEFAULT_SENSOR_POINTS,
        w_limit: float = 4.0e-3,
        w_obs_scale: float = 4.0e-3,
        wdot_limit: float = 10.0,
        wdot_obs_scale: float = 1.0,
        eta_limit: float = 1.0e-1,
        y_cutter: float = DEFAULT_Y_CUTTER,
        x0_cutter: float = 0.0,
        x_pass_end_tol: float | None = None,
        dynamics_uncertainty_std: float = 0.0,
        y0_min: float = 0.05,
        y0_max: float = 0.95,
        randomize_y0: bool = True,
        initial_eta_std: float = 0.0,
        initial_etad_std: float = 0.0,
        stiffness_grid_points: int = 100,
        modal_damping_ratio: float = 0.02,
    ):
        # ------------------------------------------------------------------
        # Basic structural parameters
        # ------------------------------------------------------------------
        self.N = int(N)
        self.L1 = float(L1)
        self.L2 = float(L2)
        self.h = float(h)
        self.E = float(E)
        self.nu = float(nu)
        self.rho = float(rho)
        self.rho_type = str(rho_type)
        self.rho_areal, self.rho_interpreted_as = _resolve_areal_mass(
            self.rho, self.h, self.rho_type
        )

        self.m_max = int(m_max)
        self.n_max = int(n_max)
        self.K = self.m_max * self.n_max
        self.state_dim = 2 * self.K
        self.mode_clamped_axis = str(mode_clamped_axis).lower().strip()

        if self.mode_clamped_axis != "x":
            raise ValueError(
                "This PlatePlant version is wired to f_nonlinear2_face_milling, "
                "whose cutter_point currently feeds along x from x=L1 to x=0. "
                "Use mode_clamped_axis='x' for a free-side-to-clamped-side pass, "
                "or update the force module path function before using a y-clamped model."
            )

        if self.N <= 0 or self.K <= 0:
            raise ValueError("N, m_max, and n_max must be positive.")

        # ------------------------------------------------------------------
        # Face-milling process parameters
        # ------------------------------------------------------------------
        self.omega_min = float(omega_min)
        self.omega_max = float(omega_max)
        self.ap_min = float(ap_min)
        self.ap_max = float(ap_max)
        # Backward-compatible aliases. Older scripts/configs used ac for the
        # second action. In the face-milling plant, that same second action is
        # axial depth of cut ap [mm]. Keep these aliases to avoid interface
        # breaks without changing the physical meaning.
        self.ac_min = self.ap_min
        self.ac_max = self.ap_max
        self.ae_min = float(ae_min)
        self.ae_max = float(ae_max)
        self.ae_default = float(np.clip(ae_default, self.ae_min, self.ae_max))
        self.control_ae = bool(control_ae)

        # Second-mode / finishing control: when control_ap is False, axial depth
        # of cut ap is NOT an RL action.  It is a fixed process parameter held
        # constant within an episode.  Like the milling line y0, it can be
        # randomized between episodes (randomize_ap) over [ap_min, ap_max], or
        # held at a single constant value ap_fixed.
        self.control_ap = bool(control_ap)
        self.randomize_ap = bool(randomize_ap)
        if ap_fixed is not None:
            self._ap_fixed = float(np.clip(ap_fixed, self.ap_min, self.ap_max))
        else:
            self._ap_fixed = 0.5 * (self.ap_min + self.ap_max)
        # Current per-episode axial depth used when ap is a fixed parameter.
        self._episode_ap = float(self._ap_fixed)

        self.D_mm = float(D_mm)
        self.feed_per_tooth_mm = float(feed_per_tooth_mm)
        self.cf = self.feed_per_tooth_mm  # compatibility alias: cf = ft [mm/tooth]

        self.gamma_L = float(np.deg2rad(gamma_L_deg))
        self.gamma_r = float(np.deg2rad(gamma_r_deg))
        self.gamma_a = float(np.deg2rad(gamma_a_deg))
        self.eta_c = float(np.deg2rad(eta_c_deg))

        self.Kt = float(Kt)
        self.Kr = float(Kr)
        self.Ka = float(Ka)
        self.Kte = float(Kte)
        self.Kre = float(Kre)
        self.Kae = float(Kae)

        self.milling_mode = str(milling_mode).lower().strip()
        self.theta0 = float(theta0)

        self.use_process_damping = bool(use_process_damping)
        self.Ksp = None if Ksp is None else float(Ksp)
        self.mu = float(mu)
        self.VB = float(VB)
        self.lambda_L = (
            None if lambda_L_deg is None else float(np.deg2rad(lambda_L_deg))
        )
        self.force_projection_mode = str(force_projection_mode).lower().strip()

        # ------------------------------------------------------------------
        # Observation, path, and RL helper parameters
        # ------------------------------------------------------------------
        self.sensor_points = tuple((float(xs), float(ys)) for xs, ys in sensor_points)
        self.n_sensors = len(self.sensor_points)
        # Observation = physical sensor displacement/velocity plus normalized
        # cutter location.  The path coordinates are part of the state observed
        # by the agent because the face-milling excitation and modal projection
        # depend on where the cutter is on the plate.
        self.obs_dim = 2 * self.n_sensors + 2

        self.w_limit = float(w_limit)
        self.w_obs_scale = float(w_obs_scale)
        self.wdot_limit = float(wdot_limit)
        self.wdot_obs_scale = float(wdot_obs_scale)
        self.eta_limit = float(eta_limit)

        self.y_cutter = float(y_cutter)
        self.x0_cutter = float(x0_cutter)
        # The milling pass ends after 90% travel: x = 0.1*L1.  Older code
        # used the name x_pass_end_tol; keep it as a backward-compatible
        # absolute end-position override, but default to the theory value.
        self.x_pass_end_m = 0.1 * self.L1 if x_pass_end_tol is None else float(x_pass_end_tol)
        self.x_pass_end_m = float(np.clip(self.x_pass_end_m, 0.0, self.L1))
        self.x_pass_end_tol = self.x_pass_end_m
        self._last_omega = float(self.omega_min)
        self._last_ap = float(self.ap_min)
        self._last_ac = self._last_ap  # compatibility alias for old diagnostics
        self._last_ae = float(self.ae_default)

        # Accepted pass kinematics. The modal state remains unchanged; these
        # variables only keep the cutter x-position and spindle phase continuous
        # when omega changes between RL steps.
        self._feed_distance_m = 0.0
        self._spindle_phase_rad = 0.0
        self._kinematic_step_t0 = 0.0
        self._kinematic_step_feed0_m = 0.0
        self._kinematic_step_phase0_rad = 0.0

        self.dynamics_uncertainty_std = float(max(dynamics_uncertainty_std, 0.0))
        self.y0_min = float(np.clip(y0_min, 0.0, self.L2))
        self.y0_max = float(np.clip(y0_max, 0.0, self.L2))
        if self.y0_max <= self.y0_min:
            raise ValueError(
                f"y0_max must be larger than y0_min after clipping to [0, L2]; "
                f"got y0_min={self.y0_min}, y0_max={self.y0_max}."
            )
        self.randomize_y0 = bool(randomize_y0)
        self.initial_eta_std = float(max(initial_eta_std, 0.0))
        self.initial_etad_std = float(max(initial_etad_std, 0.0))
        self.modal_damping_ratio = float(modal_damping_ratio)
        if not np.isfinite(self.modal_damping_ratio) or self.modal_damping_ratio < 0.0:
            raise ValueError(
                f"modal_damping_ratio must be finite and non-negative, got {modal_damping_ratio!r}."
            )
        self._rng: np.random.Generator | None = None

        # Explicit regenerative-history module handle.  ODEControlEnv can use
        # this to pass history_module=f_nonlinear2_face_milling into RK4.
        self.history_module = f_nonlinear2
        self.dynamics_history_module = f_nonlinear2
        self.force_module = f_nonlinear2
        self.face_milling_force_module = f_nonlinear2

        # Full physical action bounds [omega, ap] or [omega, ap, ae].  These are
        # kept full-width regardless of which entries are RL-controlled, so that
        # physical_action_bounds()/metadata and the physical action logged during
        # evaluation always describe [omega, ap(, ae)] consistently.
        if self.control_ae:
            self.u_phys_low = np.array(
                [self.omega_min, self.ap_min, self.ae_min], dtype=np.float64
            )
            self.u_phys_high = np.array(
                [self.omega_max, self.ap_max, self.ae_max], dtype=np.float64
            )
        else:
            self.u_phys_low = np.array(
                [self.omega_min, self.ap_min], dtype=np.float64
            )
            self.u_phys_high = np.array(
                [self.omega_max, self.ap_max], dtype=np.float64
            )

        # Controlled-action layout for the normalized RL action.  omega is always
        # controlled; ap only if control_ap; ae only if control_ae.  These bounds
        # drive _scale_action; uncontrolled entries are injected as fixed process
        # parameters (ap from the per-episode value, ae from ae_default).
        ctrl_low = [self.omega_min]
        ctrl_high = [self.omega_max]
        if self.control_ap:
            self._ap_action_idx = len(ctrl_low)
            ctrl_low.append(self.ap_min)
            ctrl_high.append(self.ap_max)
        else:
            self._ap_action_idx = None
        if self.control_ae:
            self._ae_action_idx = len(ctrl_low)
            ctrl_low.append(self.ae_min)
            ctrl_high.append(self.ae_max)
        else:
            self._ae_action_idx = None
        self._ctrl_low = np.array(ctrl_low, dtype=np.float64)
        self._ctrl_high = np.array(ctrl_high, dtype=np.float64)
        self.action_dim = int(self._ctrl_low.size)

        # Straight x-pass reference timeline, used for diagnostics and y path.
        pass_duration = estimate_pass_duration(
            self.L1, self.feed_per_tooth_mm, self.N, self.omega_min
        )
        self.t_original = np.arange(0.0, pass_duration * 1.1, 0.02, dtype=np.float64)
        feed_slow = (
            (self.feed_per_tooth_mm / 1000.0)
            * self.N
            * self.omega_min
            / (2.0 * np.pi)
        )
        self.x_traj = np.clip(
            self.L1 - feed_slow * self.t_original - self.x0_cutter,
            0.0,
            self.L1,
        )
        self.y_traj = np.full_like(self.t_original, self.y_cutter, dtype=np.float64)

        # ------------------------------------------------------------------
        # Structural mode shapes, frequencies, nonlinear stiffness
        # ------------------------------------------------------------------
        W_mn, V_mn = compute_mode_shapes(
            self.L1,
            self.L2,
            self.h,
            self.m_max,
            self.n_max,
            clamped_axis=self.mode_clamped_axis,
        )
        self.W_mn = W_mn
        self.V_mn = V_mn

        # True modal mass vector: M_k = ∫∫ rho_areal * W_k(x,y)^2 dxdy.
        # For the current normalized analytical shapes this is numerically close
        # to total plate mass for each mode, but using a vector is the rigorous
        # modal-coordinate formulation and works for non-normalized shapes too.
        self.M_modal = compute_modal_mass_vector(
            W_mn,
            self.L1,
            self.L2,
            self.rho_areal,
            self.m_max,
            self.n_max,
            grid_points=max(int(stiffness_grid_points), 151),
        )
        self.total_mass = self.L1 * self.L2 * self.rho_areal

        self.Phi = build_mode_shape_matrix(
            W_mn,
            self.sensor_points,
            self.m_max,
            self.n_max,
        )

        omega_result = compute_natural_frequencies(
            self.E,
            self.nu,
            self.rho,
            self.h,
            self.L1,
            self.L2,
            self.m_max,
            self.n_max,
            rho_type=self.rho_type,
            return_info=True,
        )
        omega_mn, omega_info = omega_result
        self.frequency_info = omega_info

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
            grid_points=stiffness_grid_points,
        )
        self.lambda_prime_mn = lambda_prime_mn

        # Preserve the original ordering convention used by f_nonlinear2:
        # for m in range(m_max): for n in range(n_max)
        omega_vec = np.asarray(omega_mn.reshape(self.K), dtype=np.float64)
        lambda_structural_vec = np.asarray(lambda_mn.reshape(self.K), dtype=np.float64)

        # Unit-consistency fix:
        # compute_nonlinear_stiffness returns structural cubic coefficients K3_k
        # with force scaling. f_nonlinear2 integrates an acceleration equation,
        # so the cubic term must be mass-normalized exactly like the cutting
        # force projection Fk = Q_k / M_k.
        modal_mass_safe = np.maximum(np.asarray(self.M_modal, dtype=np.float64), 1e-18)
        lambda_vec = lambda_structural_vec / modal_mass_safe
        zeta_vec = self.modal_damping_ratio * np.ones(self.K, dtype=np.float64)

        self.omega_vec = omega_vec
        self.lambda_structural_vec = lambda_structural_vec
        self.lambda_vec = lambda_vec
        self.zeta_vec = zeta_vec

        self._initialize_face_milling_dynamics_module()

    # ------------------------------------------------------------------
    # Module/global synchronization
    # ------------------------------------------------------------------
    def _initialize_face_milling_dynamics_module(self) -> None:
        """Push all structural and face-milling parameters into the dynamics module."""
        f_nonlinear2.m_max = self.m_max
        f_nonlinear2.n_max = self.n_max
        f_nonlinear2.N = self.N
        f_nonlinear2.K = self.K

        f_nonlinear2.zeta_vec = self.zeta_vec
        f_nonlinear2.lambda_vec = self.lambda_vec
        f_nonlinear2.omega_vec = self.omega_vec

        f_nonlinear2.W_mn = self.W_mn
        f_nonlinear2.W_mn_x = None
        f_nonlinear2.W_mn_y = None
        f_nonlinear2.W_mn_z = None
        f_nonlinear2.M_modal = self.M_modal

        f_nonlinear2.L1 = self.L1
        f_nonlinear2.L2 = self.L2
        f_nonlinear2.x0_cutter = self.x0_cutter
        f_nonlinear2.y_cutter = self.y_cutter

        f_nonlinear2.cf = self.feed_per_tooth_mm
        f_nonlinear2.ACTION_OMEGA_UNIT = "rad/s"

        f_nonlinear2.D_mm = self.D_mm
        f_nonlinear2.ae_default_mm = self.ae_default
        f_nonlinear2.gamma_L = self.gamma_L
        f_nonlinear2.gamma_r = self.gamma_r
        f_nonlinear2.gamma_a = self.gamma_a
        f_nonlinear2.eta_c = self.eta_c

        f_nonlinear2.Kt = self.Kt
        f_nonlinear2.Kr = self.Kr
        f_nonlinear2.Ka = self.Ka
        f_nonlinear2.Kte = self.Kte
        f_nonlinear2.Kre = self.Kre
        f_nonlinear2.Kae = self.Kae

        f_nonlinear2.milling_mode = self.milling_mode
        f_nonlinear2.theta0 = self.theta0

        f_nonlinear2.USE_PROCESS_DAMPING = self.use_process_damping
        f_nonlinear2.Ksp = self.Ksp
        f_nonlinear2.mu = self.mu
        f_nonlinear2.VB = self.VB
        f_nonlinear2.lambda_L = self.lambda_L
        f_nonlinear2.FORCE_PROJECTION_MODE = self.force_projection_mode

        # The scalar W_mn projection represents transverse/axial plate bending.
        f_nonlinear2.MODAL_DISPLACEMENT_TO_MM = 1000.0
        f_nonlinear2.USE_DELAYED_CUTTER_POSITION_FOR_REGEN = True
        f_nonlinear2.EDGE_FORCE_WHEN_ZERO_CHIP = False

        # Use accepted feed distance/spindle phase rather than omega*t. This is
        # important when the RL policy changes omega during an episode.
        if hasattr(f_nonlinear2, "USE_ACCEPTED_PATH_KINEMATICS"):
            f_nonlinear2.USE_ACCEPTED_PATH_KINEMATICS = True

    def _sync_face_milling_geometry(self) -> None:
        """Synchronize pass-line geometry with the force module."""
        f_nonlinear2.y_cutter = self.y_cutter
        f_nonlinear2.x0_cutter = self.x0_cutter
        f_nonlinear2.L1 = self.L1
        f_nonlinear2.L2 = self.L2

    def set_milling_line_y(self, y_m: float) -> None:
        """Set the constant y-location of the straight x-direction face-milling pass."""
        y_m = float(np.clip(y_m, 0.0, self.L2))
        self.y_cutter = y_m
        self.y_traj = np.full_like(self.t_original, y_m, dtype=np.float64)
        self._sync_face_milling_geometry()

    # Backward-compatible name used by earlier experiments.
    def set_milling_start_y(self, y0_m: float) -> None:
        self.set_milling_line_y(y0_m)

    def bind_rng(self, rng: np.random.Generator) -> None:
        """Bind Gymnasium RNG for random y-line and model uncertainty."""
        self._rng = rng

    # ------------------------------------------------------------------
    # Accepted cutter-path / spindle kinematics
    # ------------------------------------------------------------------
    def _feed_rate_from_omega(self, omega_rad_s: float) -> float:
        """Return table/feed speed [m/s] from feed per tooth and spindle speed."""
        omega_rad_s = max(float(omega_rad_s), 1e-12)
        return (self.feed_per_tooth_mm / 1000.0) * self.N * omega_rad_s / (2.0 * np.pi)

    def _sync_accepted_kinematics_to_force_module(self, t: float, *, append: bool = True) -> None:
        """Push accepted feed distance and spindle phase into the force module."""
        if hasattr(f_nonlinear2, "set_accepted_kinematic_state"):
            f_nonlinear2.set_accepted_kinematic_state(
                float(t),
                float(self._feed_distance_m),
                float(self._spindle_phase_rad),
                append=append,
            )

    def begin_step(self, t: float, u: np.ndarray) -> None:
        """Prepare continuous path/phase kinematics before RK integration.

        ODEControlEnv calls this once per accepted environment step. The action
        is still held constant during the RK substeps, but the cutter position
        and tooth phase are propagated from the last accepted values instead of
        recomputing them as omega*t from the start of the episode.
        """
        u_phys = self._scale_action(u)
        self._last_omega = float(u_phys[0])
        self._last_ap = float(u_phys[1])
        self._last_ac = self._last_ap
        self._last_ae = float(u_phys[2]) if self.control_ae else self.ae_default

        self._kinematic_step_t0 = float(t)
        self._kinematic_step_feed0_m = float(self._feed_distance_m)
        self._kinematic_step_phase0_rad = float(self._spindle_phase_rad)
        self._sync_accepted_kinematics_to_force_module(float(t), append=True)

    def end_step(self, t: float, u: np.ndarray) -> None:
        """Commit accepted cutter feed distance and spindle phase after integration."""
        u_phys = self._scale_action(u)
        omega = float(u_phys[0])
        dt_step = max(float(t) - float(self._kinematic_step_t0), 0.0)

        self._feed_distance_m = self._kinematic_step_feed0_m + self._feed_rate_from_omega(omega) * dt_step
        self._feed_distance_m = float(np.clip(self._feed_distance_m, 0.0, self.L1 + abs(self.x0_cutter)))
        self._spindle_phase_rad = self._kinematic_step_phase0_rad + omega * dt_step

        self._last_omega = omega
        self._last_ap = float(u_phys[1])
        self._last_ac = self._last_ap
        self._last_ae = float(u_phys[2]) if self.control_ae else self.ae_default
        self._sync_accepted_kinematics_to_force_module(float(t), append=True)

    # ------------------------------------------------------------------
    # Observation wrapper: modal -> physical displacement/velocity
    # ------------------------------------------------------------------
    def modal_to_physical(
        self,
        x_modal: np.ndarray,
        *,
        clip: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map modal coordinates to displacement/velocity at observation points."""
        x_modal = np.asarray(x_modal, dtype=np.float64).reshape(-1)
        if x_modal.size != self.state_dim:
            raise ValueError(
                f"Expected modal state shape ({self.state_dim},), got {x_modal.shape}."
            )

        eta = x_modal[0::2]
        eta_dot = x_modal[1::2]

        w_sensor = np.asarray(self.Phi @ eta, dtype=np.float64)
        wdot_sensor = np.asarray(self.Phi @ eta_dot, dtype=np.float64)

        if clip:
            w_sensor = np.clip(w_sensor, -self.w_limit, self.w_limit)
            wdot_sensor = np.clip(wdot_sensor, -self.wdot_limit, self.wdot_limit)

        return w_sensor, wdot_sensor

    def get_observe_state(self, x_modal: np.ndarray) -> np.ndarray:
        """Return scaled physical observation for the RL agent."""
        return self.state_to_obs(x_modal)

    def _normalized_cutter_position_obs(self) -> np.ndarray:
        """Return [cutter_x/L1, cutter_y/L2] from accepted pass kinematics."""
        cutter_x = self.L1 - self._feed_distance_m - self.x0_cutter
        cutter_x = float(np.clip(cutter_x, 0.0, self.L1))
        cutter_y = float(np.clip(self.y_cutter, 0.0, self.L2))

        x_norm = cutter_x / max(self.L1, 1e-12)
        y_norm = cutter_y / max(self.L2, 1e-12)
        return np.asarray([x_norm, y_norm], dtype=np.float64)

    def state_to_obs(self, x: np.ndarray) -> np.ndarray:
        w_sensor, wdot_sensor = self.modal_to_physical(x)
        obs = np.concatenate(
            [
                w_sensor / self.w_obs_scale,
                wdot_sensor / self.wdot_obs_scale,
                self._normalized_cutter_position_obs(),
            ]
        )
        return np.asarray(obs, dtype=np.float64)

    # ------------------------------------------------------------------
    # Action scaling and dynamics
    # ------------------------------------------------------------------
    def _scale_action(self, u: np.ndarray) -> np.ndarray:
        """Map the normalized RL action to the full physical action.

        The returned vector is always the full face-milling action
        ``[omega, ap]`` (or ``[omega, ap, ae]`` when control_ae=True), regardless
        of which entries are RL-controlled.  Uncontrolled ap is injected from the
        per-episode value ``self._episode_ap``; uncontrolled ae from ae_default.
        This keeps begin_step/end_step/dynamics and the force module unchanged.
        """
        u = np.asarray(u, dtype=np.float64).reshape(-1)
        if u.size != self.action_dim:
            raise ValueError(
                f"PlatePlant expects action shape ({self.action_dim},), got {u.shape}."
            )
        u = np.clip(u, -1.0, 1.0)
        ctrl = self._ctrl_low + 0.5 * (u + 1.0) * (self._ctrl_high - self._ctrl_low)

        omega = float(ctrl[0])
        ap = float(ctrl[self._ap_action_idx]) if self.control_ap else float(self._episode_ap)
        if self.control_ae:
            ae = float(ctrl[self._ae_action_idx])
            return np.array([omega, ap, ae], dtype=np.float64)
        return np.array([omega, ap], dtype=np.float64)

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size != self.state_dim:
            raise ValueError(
                f"PlatePlant expects state shape ({self.state_dim},), got {x.shape}."
            )

        u_phys = self._scale_action(u)
        self._last_omega = float(u_phys[0])
        self._last_ap = float(u_phys[1])
        self._last_ac = self._last_ap  # compatibility alias for old diagnostics
        self._last_ae = float(u_phys[2]) if self.control_ae else self.ae_default

        # If ae is not controlled, pass [omega, ap].  The force module then uses
        # ae_default_mm.  If ae is controlled, pass [omega, ap, ae].
        x_dot = f_nonlinear2.f_nonlinear2(t, x, u_phys)
        x_dot = np.asarray(x_dot, dtype=np.float64).reshape(-1)

        if x_dot.size != self.state_dim:
            raise ValueError(
                f"f_nonlinear2_face_milling must return shape ({self.state_dim},), "
                f"but returned {x_dot.shape}."
            )

        if self.dynamics_uncertainty_std > 0.0 and self._rng is not None:
            x_dot = x_dot.copy()
            x_dot[1::2] += (
                self.dynamics_uncertainty_std
                * self._rng.standard_normal(self.K)
            )

        return np.nan_to_num(x_dot, nan=0.0, posinf=1e8, neginf=-1e8)

    # ------------------------------------------------------------------
    # Reset / termination / spaces
    # ------------------------------------------------------------------
    def reset(
        self,
        rng: np.random.Generator | None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        options = options or {}

        # 1. Determine the y-location of the milling pass
        # FIX: Removed trailing spaces in dictionary keys to ensure proper matching
        if "y0" in options:
            self.set_milling_line_y(float(options["y0"]))
        elif "y_cutter" in options:
            self.set_milling_line_y(float(options["y_cutter"]))
        elif self.randomize_y0:
            # Use the passed `rng` for reproducibility, falling back to self._rng if needed
            randomizer = rng if rng is not None else self._rng
            if randomizer is not None:
                y0 = float(randomizer.uniform(self.y0_min, self.y0_max))
                self.set_milling_line_y(y0)

        # 1b. Determine the per-episode axial depth of cut when ap is a fixed
        # (non-action) process parameter (second-mode / finishing control).
        # Mirrors the y0 convention: an explicit options["ap"] pins it (for
        # evaluation), otherwise it is randomized over [ap_min, ap_max] when
        # randomize_ap is set, else held at the constant ap_fixed value.
        if not self.control_ap:
            if "ap" in options:
                self._episode_ap = float(np.clip(float(options["ap"]), self.ap_min, self.ap_max))
            elif self.randomize_ap:
                randomizer = rng if rng is not None else self._rng
                if randomizer is not None:
                    self._episode_ap = float(randomizer.uniform(self.ap_min, self.ap_max))
            else:
                self._episode_ap = float(self._ap_fixed)

        # 2. Reset force module history and internal kinematic states
        f_nonlinear2.reset_state_history()
        self._last_omega = float(self.omega_min)
        self._last_ap = float(self._episode_ap) if not self.control_ap else float(self.ap_min)
        self._last_ac = self._last_ap  # compatibility alias for old diagnostics
        self._last_ae = float(self.ae_default)
        
        self._feed_distance_m = 0.0
        self._spindle_phase_rad = 0.0
        self._kinematic_step_t0 = 0.0
        self._kinematic_step_feed0_m = 0.0
        self._kinematic_step_phase0_rad = 0.0
        
        self._sync_accepted_kinematics_to_force_module(0.0, append=True)

        # 3. Initialize modal state
        x0 = np.zeros(self.state_dim, dtype=np.float64)
        if self.initial_eta_std > 0.0 and rng is not None:
            x0[0::2] = rng.normal(0.0, self.initial_eta_std, size=self.K)
        if self.initial_etad_std > 0.0 and rng is not None:
            x0[1::2] = rng.normal(0.0, self.initial_etad_std, size=self.K)

        # 4. Return initial state and info dictionary
        info = {
            "y_cutter": float(self.y_cutter),
            "x_start": float(self.L1 - self.x0_cutter),
            "x_end": float(self.x_pass_end_m),
            "path_direction": "x=L1 free side -> x=0.1*L1 (90% pass)",
            "feed_distance_m": float(self._feed_distance_m),
            "spindle_phase_rad": float(self._spindle_phase_rad),
            "rho_interpreted_as": self.rho_interpreted_as,
            "control_ap": bool(self.control_ap),
        }
        if not self.control_ap:
            # Fixed axial depth held constant for this episode (finishing mode).
            info["ap_fixed_mm"] = float(self._episode_ap)
        return x0, info

    def termination(self, t: float, x: np.ndarray) -> tuple[bool, bool, dict[str, Any]]:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        info: dict[str, Any] = {}

        # 1. Catch numerical explosions (NaN/Inf) immediately
        invalid_state = not np.all(np.isfinite(x))
        if invalid_state:
            info["termination_reason"] = "invalid_state"
            return True, False, info

        # 2. Check physical instability based SOLELY on displacement
        w_sensor, wdot_sensor = self.modal_to_physical(x)
        if np.any(np.abs(w_sensor) > self.w_limit):
            info["termination_reason"] = "excessive_sensor_displacement"
            info["max_displacement_m"] = float(np.max(np.abs(w_sensor)))
            return True, False, info
        
        # Note: Velocity termination removed. Displacement is the true physical 
        # limit for machining (chip thickness/tool crash). Velocity is implicitly bounded.

        # 3. Gather kinematic and force diagnostics
        cutter_x, cutter_y = f_nonlinear2.cutter_point(t, self._last_omega)
        info["cutter_x"] = float(cutter_x)
        info["cutter_y"] = float(cutter_y)
        info["feed_progress"] = float(1.0 - cutter_x / max(self.L1, 1e-12))
        info["feed_distance_m"] = float(self._feed_distance_m)
        info["spindle_phase_rad"] = float(self._spindle_phase_rad)
        info["omega_rad_s"] = float(self._last_omega)
        info["omega_rpm"] = float(omega_to_rpm(self._last_omega))
        info["ap_mm"] = float(self._last_ap)
        info["ac_mm"] = float(self._last_ap)  # backward-compatible alias
        info["ae_mm"] = float(self._last_ae)

        force_info = getattr(f_nonlinear2, "last_force_info", None)
        if force_info is not None:
            info["F_total_N"] = np.asarray(force_info.F_total_N, dtype=np.float64)
            info["F_cut_N"] = np.asarray(force_info.F_cut_N, dtype=np.float64)
            info["mean_chip_mm"] = float(np.mean(force_info.chip_eff_mm))
            info["max_chip_mm"] = float(np.max(force_info.chip_eff_mm))

        # 4. Terminate at the configured 90% pass end (default x = 0.1 * L1).
        # This avoids fixture interference and steep mode-shape gradients at x=0.
        x_termination_threshold = self.x_pass_end_m
        if cutter_x <= x_termination_threshold:
            info["pass_completed"] = True
            info["termination_reason"] = "pass_completed_90percent"
            info["final_cutter_x"] = float(cutter_x)
            return True, False, info

        return False, False, info

    def get_observation_space(self) -> spaces.Space:
        low = np.full(self.obs_dim, -np.inf, dtype=np.float64)
        high = np.full(self.obs_dim, np.inf, dtype=np.float64)

        low[: self.n_sensors] = -self.w_limit / self.w_obs_scale
        high[: self.n_sensors] = self.w_limit / self.w_obs_scale
        low[self.n_sensors : 2 * self.n_sensors] = (
            -self.wdot_limit / self.wdot_obs_scale
        )
        high[self.n_sensors : 2 * self.n_sensors] = (
            self.wdot_limit / self.wdot_obs_scale
        )

        # Normalized cutter coordinates [cutter_x/L1, cutter_y/L2].
        low[2 * self.n_sensors :] = 0.0
        high[2 * self.n_sensors :] = 1.0

        return spaces.Box(low=low, high=high, shape=(self.obs_dim,), dtype=np.float64)

    def get_action_space(self) -> spaces.Space:
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float64,
        )

    def physical_action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return physical action bounds: [omega, ap] or [omega, ap, ae]."""
        return self.u_phys_low.copy(), self.u_phys_high.copy()


__all__ = [
    "PlatePlant",
    "rpm_to_omega",
    "omega_to_rpm",
    "estimate_pass_duration",
    "estimate_pass_episode_steps",
    "estimate_training_episode_steps",
    "compute_modal_mass_vector",
    "DEFAULT_SENSOR_POINTS",
]
