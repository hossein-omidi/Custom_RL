# -*- coding: utf-8 -*-
import numpy as np

def compute_second_derivatives(f_values, X, Y, L1, L2):
    """
    معادل دقیق تابع MATLAB:
    function [f_xx, f_yy] = compute_second_derivatives(f_values, X, Y, L1, L2)

    در MATLAB:
        dx = L1 / (size(X, 2) - 1);
        dy = L2 / (size(Y, 1) - 1);
        [f_x, f_y] = gradient(f_values, dx, dy);      % f_x: ∂f/∂x (ستون‌ها)، f_y: ∂f/∂y (ردیف‌ها)
        [f_xx, ~]   = gradient(f_x, dx, dy);          % ∂²f/∂x²
        [~,   f_yy] = gradient(f_y, dx, dy);          % ∂²f/∂y²

    در NumPy:
        np.gradient(f, dy, dx) برمی‌گرداند (∂f/∂y, ∂f/∂x).
    """
    f_values = np.asarray(f_values, dtype=float)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    dx = L1 / (X.shape[1] - 1)  # طول در راستای ستون‌ها
    dy = L2 / (Y.shape[0] - 1)  # طول در راستای ردیف‌ها

    # ∂f/∂y و ∂f/∂x
    f_y, f_x = np.gradient(f_values, dy, dx, edge_order=2)

    # ∂²f/∂y² و ∂²f/∂x²
    f_y_y, _ = np.gradient(f_y, dy, dx, edge_order=2)
    _, f_x_x = np.gradient(f_x, dy, dx, edge_order=2)

    return f_x_x, f_y_y


def compute_nonlinear_stiffness(E, nu, h, L1, L2, W_mn, V_mn, m_max, n_max):
    """
    پورت مستقیم:
    [lambda_mn, lambda_prime_mn] = compute_nonlinear_stiffness(E, nu, h, L1, L2, W_mn, V_mn, m_max, n_max)
    """

    lambda_mn = np.zeros((m_max, n_max), dtype=float)
    lambda_prime_mn = np.zeros((m_max, n_max), dtype=float)

    prefactor = (E * h) / (2.0 * (1.0 - nu ** 2))

    # Discretize the plate
    x = np.linspace(0.0, L1, 100)
    y = np.linspace(0.0, L2, 100)
    z = np.linspace(0.0, h, 100)

    X, Y = np.meshgrid(x, y, indexing="xy")   # شکل: (len(y), len(x))
    Z, Y2 = np.meshgrid(z, y, indexing="xy")  # شکل: (len(y), len(z))

    for m in range(m_max):        # MATLAB: m = 1..m_max
        for n in range(n_max):    # MATLAB: n = 1..n_max
            # فرض: W_mn[m][n] و V_mn[m][n] توابع قابل فراخوانی هستند
            W_values = W_mn[m][n](X, Y)
            V_values = V_mn[m][n](Z, Y2)

            W_xx, W_yy = compute_second_derivatives(W_values, X, Y, L1, L2)
            V_zz, V_yy = compute_second_derivatives(V_values, Z, Y2, h, L2)

            integrand = (W_xx + W_yy) ** 2
            inner = np.trapz(integrand, x, axis=1)       # انتگرال نسبت به x (ستون‌ها)
            lambda_mn[m, n] = prefactor * np.trapz(inner, y)  # انتگرال نسبت به y (ردیف‌ها)

            integrand_prime = (V_zz + V_yy) ** 2
            inner_p = np.trapz(integrand_prime, z, axis=1)    # نسبت به z
            lambda_prime_mn[m, n] = prefactor * np.trapz(inner_p, y)

    return lambda_mn, lambda_prime_mn
