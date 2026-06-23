"""Nonlinear plate vibration ODE plant."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from custom_rl.plants.base import ODEPlant
from custom_rl.plants import f_nonlinear2
from custom_rl.plants.compute_mode_shapes import compute_mode_shapes
from custom_rl.plants.compute_natural_frequencies import compute_natural_frequencies
from custom_rl.plants.compute_nonlinear_stiffness import compute_nonlinear_stiffness


class PlatePlant(ODEPlant):
    """
    Nonlinear plate vibration plant.

    State:
        [eta1, eta1_dot, eta2, eta2_dot, ..., etaK, etaK_dot]

    Normalized action:
        u = [u_omega, u_ac], each in [-1, 1]

    Physical action passed to f_nonlinear2:
        omega in [omega_min, omega_max]
        ac    in [ac_min, ac_max]
    """

    def __init__(
        self,
        N: int = 4,
        L1: float = 1.0,
        L2: float = 1,
        h: float = 0.02,
        E: float = 71.7e9,
        nu: float = 0.33,
        rho: float = 2810.0,
        m_max: int = 3,
        n_max: int = 2,
        omega_min: float = 50,
        omega_max: float = 2000.0,
        ac_min: float = 0,
        ac_max: float = 20.0,
        eta_limit: float = 0.01,
        eta_obs_limit: float = 1e6,
        eta_dot_obs_limit: float = 1e6,
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
            ac_min: minimum physical control coefficient/input
            ac_max: maximum physical control coefficient/input
            eta_limit: modal displacement safety limit for termination
            eta_obs_limit: observation bound for modal displacement
            eta_dot_obs_limit: observation bound for modal velocity
        """
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

        self.eta_limit = eta_limit
        self.eta_obs_limit = eta_obs_limit
        self.eta_dot_obs_limit = eta_dot_obs_limit

        self.u_phys_low = np.array(
            [self.omega_min, self.ac_min],
            dtype=np.float64,
        )
        self.u_phys_high = np.array(
            [self.omega_max, self.ac_max],
            dtype=np.float64,
        )

        # Trajectory input
        self.t_original = np.arange(0.0, 20.0, 0.02, dtype=np.float64)
        self.x_traj = 0.1 * np.sin(2.0 * np.pi * 0.2 * self.t_original)
        self.y_traj = 0.1 * np.cos(2.0 * np.pi * 0.2 * self.t_original)

        # Modal mass
        M_modal = self.L1 * self.L2 * self.rho * self.h

        # Mode shapes
        W_mn, V_mn = compute_mode_shapes(
            self.L1,
            self.L2,
            self.h,
            self.m_max,
            self.n_max,
        )

        # Natural frequencies
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

        # Nonlinear stiffness
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

        cf = 0.3

        xi_base = np.array(
            [6765e9, -4910e6, 2840e3, 132],
            dtype=np.float64,
        ) / 2.5

        delta_base = np.array(
            [12740e9, -7452e6, 1674e3, 246],
            dtype=np.float64,
        ) / 2.5

        # Set required variables inside the existing f_nonlinear2 module.
        # This preserves your original nonlinear dynamics implementation.
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

        f_nonlinear2.M_modal = M_modal
        f_nonlinear2.decimal_places = 1

    def _scale_action(self, u: np.ndarray) -> np.ndarray:
        """
        Map normalized action from [-1, 1]^2 to physical action [omega, ac].
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

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        State derivative: dx/dt = dynamics(t, x, u).

        The environment supplies normalized action in [-1, 1]^2.
        This method scales it to physical [omega, ac] before calling f_nonlinear2.
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1)

        if x.size != self.state_dim:
            raise ValueError(
                f"PlatePlant expects state with shape ({self.state_dim},), "
                f"but received {x.shape}."
            )

        u_phys = self._scale_action(u)

        x_dot = f_nonlinear2.f_nonlinear2(t, x, u_phys)
        x_dot = np.asarray(x_dot, dtype=np.float64).reshape(-1)

        if x_dot.size != self.state_dim:
            raise ValueError(
                f"f_nonlinear2 must return derivative with shape "
                f"({self.state_dim},), but returned {x_dot.shape}."
            )

        return x_dot

    def reset(self, rng) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Sample initial modal displacement and velocity.
        """
        f_nonlinear2.reset_state_history()

        #eta0 = rng.uniform(-1e-4, 1e-4, size=self.K)
        #eta_dot0 = rng.uniform(-1e-4, 1e-4, size=self.K)
        eta0 = rng.uniform(0, 0, size=self.K)
        eta_dot0 = rng.uniform(0, 0, size=self.K)

        x0 = np.zeros(self.state_dim, dtype=np.float64)
        x0[0::2] = eta0
        x0[1::2] = eta_dot0

        return x0, {}

    def termination(self, t: float, x: np.ndarray) -> tuple[bool, bool, dict[str, Any]]:
        """
        Terminate if modal displacement becomes unsafe or state becomes invalid.
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        eta = x[0::2]

        invalid_state = not np.all(np.isfinite(x))
        excessive_displacement = np.any(np.abs(eta) > self.eta_limit)

        terminated = bool(invalid_state or excessive_displacement)

        info: dict[str, Any] = {}

        if invalid_state:
            info["termination_reason"] = "invalid_state"

        if excessive_displacement:
            info["termination_reason"] = "excessive_modal_displacement"

        return terminated, False, info

    def get_observation_space(self) -> spaces.Space:
        """
        Return Gymnasium observation space.

        Observation:
            [eta1, eta1_dot, eta2, eta2_dot, ..., etaK, etaK_dot]
        """
        low = np.zeros(self.state_dim, dtype=np.float64)
        high = np.zeros(self.state_dim, dtype=np.float64)

        low[0::2] = -self.eta_obs_limit
        high[0::2] = self.eta_obs_limit

        low[1::2] = -self.eta_dot_obs_limit
        high[1::2] = self.eta_dot_obs_limit

        return spaces.Box(
            low=low,
            high=high,
            shape=(self.state_dim,),
            dtype=np.float64,
        )

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
        Map internal state to observation.

        Here, observation is equal to the full modal state.
        """
        return np.asarray(x, dtype=np.float64).reshape(-1)