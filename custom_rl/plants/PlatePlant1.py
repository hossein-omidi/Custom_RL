# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 11:43:26 2026

@author: Acer
"""

import numpy as np
from scipy.integrate import solve_ivp

import f_nonlinear2
from compute_mode_shapes import compute_mode_shapes
from compute_natural_frequencies import compute_natural_frequencies
from compute_natural_frequencies_prim import compute_natural_frequencies_prim
from compute_nonlinear_stiffness import compute_nonlinear_stiffness

#from custom_rl.plants.base import ODEPlant


#class PlatePlant(ODEPlant):
class PlatePlant:
    """
    Plate vibration nonlinear plant, same structure as CartPolePlant.
    State: [eta1, eta1_dot, eta2, eta2_dot, ...]
    Action: u = [omega, ac]
    """

    def __init__(self):
        # فقط پارامترهای اولیه + محاسبات یک‌بار
        self.N = 5
        self.L1 = 1.0
        self.L2 = 0.5
        self.h = 0.02

        self.E = 70e9
        self.nu = 0.3
        self.rho = 7850

        self.m_max = 2
        self.n_max = 2
        self.K = self.m_max * self.n_max

        # trajectory input
        self.t_original = np.arange(0, 10.0, 0.02)
        self.x_traj = 0.1 * np.sin(2 * np.pi * 0.2 * self.t_original)
        self.y_traj = 0.1 * np.cos(2 * np.pi * 0.2 * self.t_original)

        # modal mass
        M_modal = self.L1 * self.L2 * self.rho * self.h

        # mode shapes + natural freq + nonlinear stiffness
        W_mn, V_mn = compute_mode_shapes(self.L1, self.L2, self.h, self.m_max, self.n_max)
        omega_mn = compute_natural_frequencies(self.E, self.nu, self.rho, self.h, self.L1, self.L2, self.m_max, self.n_max)
        lambda_mn, lambda_prime_mn = compute_nonlinear_stiffness(self.E, self.nu, self.h, self.L1, self.L2,W_mn, V_mn, self.m_max, self.n_max)
        omega_vec = omega_mn.T.reshape(self.K)
        lambda_vec = lambda_mn.T.reshape(self.K)

        # damping
        zeta_vec = 0.05 * np.ones(self.K)

        # force coefficients
        cf = 0.3    # اسکالر

        xi_base = np.array([6765e9, -4910e6, 2840e3, 132]) / 2.5
        delta_base = np.array([12740e9, -7452e6, 1674e3, 246]) / 2.5

        # ---- set globals in f_nonlinear2 ----
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


    def dynamics(self, t, x, u):
        """
        u = [omega, ac]
        دقیقا f_nonlinear2 را صدا می‌زند
        """
        return f_nonlinear2.f_nonlinear2(t, x, u)


    def reset(self, rng):
        """
        مثل CartPole: حالت اولیه را random می‌کند
        """
        eta0 = rng.uniform(-0.0001, 0.0001, size=self.K)
        etad0 = rng.uniform(-0.0001, 0.0001, size=self.K)

        x0 = np.zeros(2*self.K)
        x0[0::2] = eta0
        x0[1::2] = etad0

        return x0, {}


    def termination(self, t, x):
        """
        اگر جابجایی زیاد شود   terminate = True
        """
        eta = x[0::2]
        terminated = np.any(np.abs(eta) > 0.1)
        return terminated, False, {}
    '''
    def get_observation_space(self):
        low = -1e6 * np.ones(2*self.K)
        high = 1e6 * np.ones(2*self.K)
        return spaces.Box(low=low, high=high, dtype=np.float64)


    def get_action_space(self):
        # u = [omega, ac]
        return spaces.Box(low=np.array([0.0, -10.0]),
                          high=np.array([2000.0, 10.0]),
                          dtype=np.float64)
    '''
    
