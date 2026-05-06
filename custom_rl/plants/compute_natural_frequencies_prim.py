# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 22:27:50 2026

@author: Acer
"""

import numpy as np

def compute_lambda_star(b_a, mode, mode_type):
    """
    پورت مستقیم از تابع داخلی compute_lambda_star در MATLAB.
    mode: عدد صحیح 1..4 یا 1..5
    """

    b_a_values = np.array(
        [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55,
         0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00]
    )

    lambda_star_symmetric = np.array([
        [3.472, 21.29, 27.2, 54.3, 61.3],
        [3.132, 19.30, 26.6, 51.5, 55.5],
        [2.809, 17.36, 26.1, 48.3, 50.4],
        [2.504, 15.75, 25.6, 43.8, 47.0],
        [2.217, 13.75, 25.2, 39.0, 44.3],
        [1.946, 12.09, 24.7, 34.3, 41.8],
        [1.694, 10.53, 24.2, 30.0, 39.6],
        [1.459, 9.08, 23.5, 26.3, 37.3],
        [1.242, 7.73, 21.4, 24.3, 35.2],
        [1.042, 6.49, 18.2, 23.7, 33.2],
        [0.861, 5.37, 15.03, 23.3, 29.6],
        [0.696, 4.34, 12.18, 22.7, 24.5],
        [0.549, 3.42, 9.61, 18.9, 23.0],
        [0.419, 2.62, 7.36, 14.47, 22.5],
        [0.307, 1.92, 5.39, 10.63, 17.6],
        [0.213, 1.33, 3.73, 7.36, 12.20],
        [0.135, 0.85, 2.39, 4.70, 7.79],
        [0.076, 0.47, 1.34, 2.64, 4.36],
        [0.034, 0.21, 0.59, 1.16, 1.92],
        [0.008, 0.05, 0.15, 0.29, 0.47],
        [0.000, 0.00, 0.00, 0.00, 0.00]
    ])

    lambda_star_antisymmetric = np.array([
        [8.55, 31.1, 64.2, 71.1],
        [8.01, 28.8, 62.3, 66.6],
        [7.49, 26.6, 57.9, 64.8],
        [6.98, 24.6, 52.8, 64.2],
        [6.47, 22.5, 47.9, 63.7],
        [5.99, 20.6, 43.0, 63.2],
        [5.51, 18.8, 38.6, 62.5],
        [5.04, 17.0, 34.3, 58.7],
        [4.59, 15.3, 30.4, 51.7],
        [4.15, 13.7, 26.6, 44.7],
        [3.71, 12.1, 23.2, 38.3],
        [3.39, 10.7, 19.9, 32.5],
        [2.87, 9.21, 17.0, 27.0],
        [2.48, 7.86, 14.2, 22.3],
        [2.09, 6.56, 11.7, 18.0],
        [1.72, 5.34, 9.36, 14.05],
        [1.35, 4.16, 7.19, 10.60],
        [0.997, 3.06, 5.20, 7.52],
        [0.66, 1.99, 3.36, 4.78],
        [0.33, 0.98, 1.64, 2.30],
        [0.00, 0.00, 0.00, 0.00]
    ])

    if mode_type == 'symmetric':
        lambda_star_table = lambda_star_symmetric
        max_mode = 5
    elif mode_type == 'antisymmetric':
        lambda_star_table = lambda_star_antisymmetric
        max_mode = 4
    else:
        raise ValueError('Invalid mode type. Use "symmetric" or "antisymmetric".')

    if mode < 1 or mode > max_mode:
        raise ValueError(f"Mode must be between 1 and {max_mode} for {mode_type} modes.")

    col = mode - 1
    y = lambda_star_table[:, col]
    lambda_star_val = np.interp(b_a, b_a_values, y)
    return lambda_star_val


def compute_natural_frequencies_prim(E, nu, rho, h, a, b, m_max, n_max):
    """
    پورت مستقیم از compute_natural_frequencies_prim.m
    """

    D = E * h ** 3 / (12.0 * (1.0 - nu ** 2))
    b_a = h / b

    omega_prime_mn = np.zeros((m_max, n_max))

    for m in range(m_max):
        for n in range(n_max):
            # در نسخه MATLAB: mode = n (1..n_max)؛ اینجا n پایتون 0..n_max-1 است، پس mode = n+1
            mode_index = n + 1

            lambda_star_symmetric = compute_lambda_star(b_a, mode_index, 'symmetric')
            lambda_star_antisymmetric = compute_lambda_star(b_a, mode_index, 'antisymmetric')

            omega_symmetric = (lambda_star_symmetric / h ** 2) * np.sqrt(D / rho)
            omega_antisymmetric = (lambda_star_antisymmetric / h ** 2) * np.sqrt(D / rho)

            omega_prime_mn[m, n] = 0.5 * (omega_symmetric + omega_antisymmetric)

    return omega_prime_mn
