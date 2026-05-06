# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 22:26:24 2026

@author: Acer
"""

import numpy as np

def compute_mode_shapes(L1, L2, h, m_max, n_max):
    """
    پورت مستقیم از نسخه MATLAB:
    function [W_mn, V_mn] = compute_mode_shapes(L1, L2, h, m_max, n_max)
    W_mn و V_mn آرایه‌های دوبعدی از توابع هستند.
    """

    # ریشه‌ها برای تیر گیردار-آزاد در جهت x/z
    alpha = np.array([1.875, 4.694, 7.854, 10.995, 14.137, 17.279])

    # ریشه‌ها برای تیر آزاد-آزاد در جهت y
    beta = np.array([4.73, 7.853, 10.995, 14.137, 17.279, 20.420])

    # در پایتون برای آرایه 2D از توابع، از لیست تو در تو استفاده می‌کنیم
    W_mn = [[None for _ in range(n_max)] for _ in range(m_max)]
    V_mn = [[None for _ in range(n_max)] for _ in range(m_max)]

    for m in range(m_max):  # m = 0..m_max-1 متناظر m=1..m_max متلب
        alpha_m = alpha[m]
        mu_m = (np.cosh(alpha_m) + np.cos(alpha_m)) / (np.sinh(alpha_m) * np.sin(alpha_m))
        v_m = (np.sinh(alpha_m) - np.sin(alpha_m)) / (np.sinh(alpha_m) * np.sin(alpha_m))

        def make_phi(alpha_m, mu_m, v_m):
            def phi_m(y_normalized):
                y_normalized = np.array(y_normalized, dtype=float)
                return (
                    mu_m * (np.cosh(alpha_m * y_normalized) - np.cos(alpha_m * y_normalized))
                    - v_m * (np.sinh(alpha_m * y_normalized) - np.sin(alpha_m * y_normalized))
                )
            return phi_m

        phi_m_fun = make_phi(alpha_m, mu_m, v_m)

        for n in range(n_max):  # n = 0..n_max-1 متناظر n=1..n_max متلب
            if n == 0:
                def psi_n_fun(s_normalized):
                    return np.ones_like(np.array(s_normalized, dtype=float))
            elif n == 1:
                def psi_n_fun(s_normalized):
                    s_normalized = np.array(s_normalized, dtype=float)
                    return np.sqrt(3.0) * (2.0 * s_normalized - 1.0)
            else:
                beta_n = beta[n - 2]
                mu_n = (np.cosh(beta_n) - np.cos(beta_n)) / (np.sinh(beta_n) * np.sin(beta_n))
                v_n = (np.sinh(beta_n) + np.sin(beta_n)) / (np.sinh(beta_n) * np.sin(beta_n))

                def make_psi(beta_n, mu_n, v_n):
                    def psi_n(s_normalized):
                        s_normalized = np.array(s_normalized, dtype=float)
                        return (
                            mu_n * (np.cosh(beta_n * s_normalized) + np.cos(beta_n * s_normalized))
                            - v_n * (np.sinh(beta_n * s_normalized) + np.sin(beta_n * s_normalized))
                        )
                    return psi_n

                psi_n_fun = make_psi(beta_n, mu_n, v_n)

            # تعریف W_{mn}(x,y) = φ_m(x/L1) * ψ_n(y/L2)
            def make_W(phi_m_local, psi_n_local):
                def W_func(x, y):
                    x_arr = np.array(x, dtype=float)
                    y_arr = np.array(y, dtype=float)
                    return phi_m_local(x_arr / L1) * psi_n_local(y_arr / L2)
                return W_func

            # تعریف V_{mn}(z,y) = φ_m(z/h) * ψ_n(y/L2)
            def make_V(phi_m_local, psi_n_local):
                def V_func(z, y):
                    z_arr = np.array(z, dtype=float)
                    y_arr = np.array(y, dtype=float)
                    return phi_m_local(z_arr / h) * psi_n_local(y_arr / L2)
                return V_func

            W_mn[m][n] = make_W(phi_m_fun, psi_n_fun)
            V_mn[m][n] = make_V(phi_m_fun, psi_n_fun)

    return W_mn, V_mn
