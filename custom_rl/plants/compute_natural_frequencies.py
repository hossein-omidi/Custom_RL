# -*- coding: utf-8 -*-
"""
Port دقیق از compute_natural_frequencies.m با تطابق کامل با MATLAB.
"""

import numpy as np

def compute_natural_frequencies(E, nu, rho, h, a, b, m_max, n_max):
    """
    محاسبه فرکانس‌های طبیعی صفحه، معادل compute_natural_frequencies.m در MATLAB.
    """

    # داده‌های تجربی (exp_data) دقیقاً مطابق MATLAB
    exp_data = np.array([
        [0, 0, 3.50,  3.50,  3.50,  3.45],
        [0, 1, 21.7,  21.6,  21.5,  21.1],
        [0, 2, 60.5,  60.4,  59.8,  59.3],
        [0, 3, 118.7, 117.5, 116.5, 115.2],
        [0, 4, 196.0, 195.0, 190.0, 188.0],
        [0, 5, 292.0, 285.0, 283.0, 281.0],
        [1, 0, 14.5,  17.3,  22.5,  32.0],
        [1, 1, 48.1,  54.8,  69.6,  98.0],
        [1, 2, 92.3,  101.5, 125.0, 169.0],
        [1, 3, 154.0, 170.0, 187.0, 248.0],
        [1, 4, 228.0, np.nan, np.nan, np.nan],
        [1, 5, np.nan, np.nan, np.nan, np.nan],
        [2, 0, 92.8,  139.1, 246.0, 321.0],
        [2, 1, 125.1, np.nan, np.nan, np.nan],
        [2, 2, 176.0, np.nan, np.nan, np.nan],
        [2, 3, 244.0, np.nan, np.nan, np.nan],
        [3, 0, 246.0, np.nan, np.nan, np.nan],
        [3, 1, 274.0, np.nan, np.nan, np.nan],
        [3, 2, np.nan, np.nan, np.nan, np.nan]
    ], dtype=float)

    # نسبت‌های طول صفحه متناظر با ستون‌های سوم تا ششم
    aspect_ratios_exp = np.array([2.00, 2.50, 3.33, 5.00], dtype=float)

    # سختی خمشی صفحه
    D = E * h**3 / (12.0 * (1.0 - nu**2))

    # نسبت طول صفحه فعلی
    aspect_ratio = float(a) / float(b)

    # ---- بازسازی دقیق interp1(aspect_ratios, exp_data(:,3:end)', aspect_ratio, 'linear', 'extrap')' ----
    # exp_data(:,3:end) در MATLAB: 19×4 → در پایتون:
    exp_values = exp_data[:, 2:6]     # shape: (19, 4)

    num_rows = exp_values.shape[0]    # باید 19 باشد
    interpolated_data = np.zeros(num_rows, dtype=float)  # معادل 19×1

    for i in range(num_rows):
        # این دقیقاً exp_data(i, 3:6) در MATLAB است
        y_row = exp_values[i, :]     # شکل (4,)

        # حذف مقادیر NaN برای تقلید رفتار interp1
        valid_mask = ~np.isnan(y_row)
        if not np.any(valid_mask):
            interpolated_data[i] = np.nan
        else:
            x_valid = aspect_ratios_exp[valid_mask]
            y_valid = y_row[valid_mask]
            # np.interp با ورودی خارج از بازه، به صورت خطی extrapolate می‌کند
            interpolated_data[i] = np.interp(aspect_ratio, x_valid, y_valid)

    # ---- ساختن ماتریس فرکانس‌های طبیعی ω_mn ----
    omega_mn = np.zeros((m_max, n_max), dtype=float)

    for m in range(m_max):      # m = 0..m_max-1
        for n in range(n_max):  # n = 0..n_max-1

            # معادل: idx = find((exp_data(:,1)==n) & (exp_data(:,2)==m))
            mask = (exp_data[:, 0] == float(n)) & (exp_data[:, 1] == float(m))
            idx_arr = np.where(mask)[0]

            if idx_arr.size > 0:
                idx = idx_arr[0]
                val = interpolated_data[idx]

                if np.isnan(val):
                    omega_mn[m, n] = 500.0
                else:
                    omega_mn[m, n] = np.sqrt(D / (rho * a**4)) * val
            else:
                omega_mn[m, n] = 500.0

    return omega_mn
