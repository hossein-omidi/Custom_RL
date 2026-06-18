# -*- coding: utf-8 -*-
"""
Mode shapes for the plate plant (MATLAB port + fast NumPy evaluator).

W_{mn}(x, y) = phi_m(x/L1) * psi_n(y/L2)   — normal / out-of-plane field
V_{mn}(z, y) = phi_m(z/h)  * psi_n(y/L2)   — in-plane feed field
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Clamped-free beam roots (x / z direction)
_ALPHA = np.array([1.875, 4.694, 7.854, 10.995, 14.137, 17.279], dtype=np.float64)
# Free-free beam roots (y direction), used for n >= 2
_BETA = np.array([4.73, 7.853, 10.995, 14.137, 17.279, 20.420], dtype=np.float64)


def _phi_beam(s: np.ndarray, alpha: float, mu: float, v: float) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    return (
        mu * (np.cosh(alpha * s) - np.cos(alpha * s))
        - v * (np.sinh(alpha * s) - np.sin(alpha * s))
    )


def _psi_beam(s: np.ndarray, beta: float, mu: float, v: float) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    return (
        mu * (np.cosh(beta * s) + np.cos(beta * s))
        - v * (np.sinh(beta * s) + np.sin(beta * s))
    )


@dataclass(frozen=True)
class ModeShapeBasis:
    """Vectorized mode-shape evaluator (same formulas as ``compute_mode_shapes``)."""

    L1: float
    L2: float
    h: float
    m_max: int
    n_max: int
    alpha: np.ndarray
    mu_m: np.ndarray
    v_m: np.ndarray
    psi_kind: np.ndarray  # 0=const, 1=linear, 2=beam
    beta_n: np.ndarray
    mu_n: np.ndarray
    v_n: np.ndarray

    @property
    def K(self) -> int:
        return int(self.m_max * self.n_max)

    @classmethod
    def build(cls, L1: float, L2: float, h: float, m_max: int, n_max: int) -> ModeShapeBasis:
        alpha = _ALPHA[:m_max].copy()
        mu_m = (np.cosh(alpha) + np.cos(alpha)) / (np.sinh(alpha) * np.sin(alpha))
        v_m = (np.sinh(alpha) - np.sin(alpha)) / (np.sinh(alpha) * np.sin(alpha))

        psi_kind = np.zeros(n_max, dtype=np.int8)
        beta_n = np.zeros(n_max, dtype=np.float64)
        mu_n = np.zeros(n_max, dtype=np.float64)
        v_n = np.zeros(n_max, dtype=np.float64)
        for n in range(n_max):
            if n == 0:
                psi_kind[n] = 0
            elif n == 1:
                psi_kind[n] = 1
            else:
                psi_kind[n] = 2
                b = float(_BETA[n - 2])
                beta_n[n] = b
                mu_n[n] = (np.cosh(b) - np.cos(b)) / (np.sinh(b) * np.sin(b))
                v_n[n] = (np.sinh(b) + np.sin(b)) / (np.sinh(b) * np.sin(b))

        return cls(
            L1=float(L1),
            L2=float(L2),
            h=float(h),
            m_max=int(m_max),
            n_max=int(n_max),
            alpha=alpha,
            mu_m=mu_m,
            v_m=v_m,
            psi_kind=psi_kind,
            beta_n=beta_n,
            mu_n=mu_n,
            v_n=v_n,
        )

    def _psi_values(self, sy: np.ndarray) -> np.ndarray:
        """psi_n(sy) for all n; shape (n_max, *sy.shape)."""
        sy = np.asarray(sy, dtype=np.float64)
        out = np.empty((self.n_max,) + sy.shape, dtype=np.float64)
        for n in range(self.n_max):
            kind = int(self.psi_kind[n])
            if kind == 0:
                out[n] = 1.0
            elif kind == 1:
                out[n] = np.sqrt(3.0) * (2.0 * sy - 1.0)
            else:
                out[n] = _psi_beam(sy, self.beta_n[n], self.mu_n[n], self.v_n[n])
        return out

    def w_values(self, x_c: float, y_c: float) -> np.ndarray:
        """W_k(x_c, y_c) for k = m*n_max + n (m-major). Shape (K,)."""
        sx = np.float64(x_c) / self.L1
        sy = np.float64(y_c) / self.L2
        psi = self._psi_values(np.asarray([sy]))[:, 0]
        out = np.empty(self.K, dtype=np.float64)
        for m in range(self.m_max):
            phi = float(_phi_beam(sx, self.alpha[m], self.mu_m[m], self.v_m[m]))
            out[m * self.n_max : (m + 1) * self.n_max] = phi * psi
        return out

    def v_values(self, z_c: float, y_c: float) -> np.ndarray:
        """V_k(z_c, y_c); shape (K,)."""
        sz = np.float64(z_c) / self.h
        sy = np.float64(y_c) / self.L2
        psi = self._psi_values(np.asarray([sy]))[:, 0]
        out = np.empty(self.K, dtype=np.float64)
        for m in range(self.m_max):
            phi = float(_phi_beam(sz, self.alpha[m], self.mu_m[m], self.v_m[m]))
            out[m * self.n_max : (m + 1) * self.n_max] = phi * psi
        return out

    def w_values_batch(self, x_c: np.ndarray, y_c: np.ndarray) -> np.ndarray:
        """W_k at P tool positions; shape (K, P)."""
        x_c = np.asarray(x_c, dtype=np.float64).reshape(-1)
        y_c = np.asarray(y_c, dtype=np.float64).reshape(-1)
        if x_c.size != y_c.size:
            raise ValueError("x_c and y_c must have the same length.")
        P = int(x_c.size)
        sx = x_c / self.L1
        sy = y_c / self.L2
        psi = self._psi_values(sy)  # (n_max, P)
        out = np.zeros((self.K, P), dtype=np.float64)
        for m in range(self.m_max):
            phi = _phi_beam(sx, self.alpha[m], self.mu_m[m], self.v_m[m])  # (P,)
            out[m * self.n_max : (m + 1) * self.n_max, :] = phi * psi
        return out

    def v_values_batch_z(self, z_c: float, y_c: np.ndarray) -> np.ndarray:
        """V_k at fixed z_c and P y positions; shape (K, P)."""
        y_c = np.asarray(y_c, dtype=np.float64).reshape(-1)
        P = int(y_c.size)
        sz = np.float64(z_c) / self.h
        sy = y_c / self.L2
        psi = self._psi_values(sy)
        out = np.zeros((self.K, P), dtype=np.float64)
        for m in range(self.m_max):
            phi = float(_phi_beam(sz, self.alpha[m], self.mu_m[m], self.v_m[m]))
            out[m * self.n_max : (m + 1) * self.n_max, :] = phi * psi
        return out

    def projection_w(self, x_c: float, y_c: float, M_n: float) -> np.ndarray:
        """b_n,k = W_k / M_n for generalized force projection."""
        return self.w_values(x_c, y_c) / float(M_n)

    def projection_v(self, z_c: float, y_c: float, M_f: float) -> np.ndarray:
        return self.v_values(z_c, y_c) / float(M_f)


def compute_mode_shapes(L1, L2, h, m_max, n_max):
    """
    پورت مستقیم از نسخه MATLAB:
    function [W_mn, V_mn] = compute_mode_shapes(L1, L2, h, m_max, n_max)
    W_mn و V_mn آرایه‌های دوبعدی از توابع هستند.
    """

    # ریشه‌ها برای تیر گیردار-آزاد در جهت x/z
    alpha = _ALPHA

    # ریشه‌ها برای تیر آزاد-آزاد در جهت y
    beta = _BETA

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
