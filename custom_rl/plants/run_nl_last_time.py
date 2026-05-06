import numpy as np
from scipy.integrate import solve_ivp

from compute_mode_shapes import compute_mode_shapes
from compute_natural_frequencies import compute_natural_frequencies
from compute_natural_frequencies_prim import compute_natural_frequencies_prim
from compute_nonlinear_stiffness import compute_nonlinear_stiffness
import f_nonlinear2


def run_nl_last_time():
    """
    نسخه کاملاً معادل run_nl_last_time.m

    خروجی‌ها:
        t_sol  : آرایه زمان
        x_sol  : ماتریس حالت‌ها (time × 8)
        y_sol  : خروجی اسکالر y(t) = C * x(t)
    """

    # ---------------- 1) پارامترها ----------------
    N = 5
    L1 = 1.0
    L2 = 0.5
    h = 0.02

    E = 70e9
    nu = 0.3
    rho = 7850

    m_max = 2
    n_max = 2
    K = m_max * n_max   # 4 mode

    cf_scalar = 0.3
    zeta_mn = 0.05
    decimal_places = 1

    # ---------------- 2) مسیر ابزار ----------------
    t_original = np.arange(0.0, 10.0 + 1e-12, 0.05)
    x_traj = 0.1 * np.sin(2 * np.pi * 0.2 * t_original)
    y_traj = 0.1 * np.cos(2 * np.pi * 0.2 * t_original)

    # ---------------- 3) جرم مودال ----------------
    M_modal = L1 * L2 * rho * h

    # ---------------- 4) محاسبه مودها ----------------
    W_mn, V_mn = compute_mode_shapes(L1, L2, h, m_max, n_max)

    omega_mn = compute_natural_frequencies(E, nu, rho, h, L1, L2, m_max, n_max)
    omega_prime_mn = compute_natural_frequencies_prim(E, nu, rho, h, L1, L2, m_max, n_max)

    lambda_mn, lambda_prime_mn = compute_nonlinear_stiffness(
        E, nu, h, L1, L2, W_mn, V_mn, m_max, n_max
    )

    # ---------------- 5) فلت‌کردن بردارها ----------------
    omega_vec = omega_mn.T.reshape(K)
    lambda_vec = lambda_mn.T.reshape(K)
    zeta_vec = zeta_mn * np.ones(K)
    #cf = cf_scalar * np.ones(K)
    cf = cf_scalar 

    xi_base = np.array([6765e9, -4910e6, 2840e3, 132]) / 2.5
    delta_base = np.array([12740e9, -7452e6, 1674e3, 246]) / 2.5

    # ---------------- 6) تنظیم متغیرهای global ----------------
    f_nonlinear2.m_max = m_max
    f_nonlinear2.n_max = n_max
    f_nonlinear2.N = N
    f_nonlinear2.K = K

    f_nonlinear2.zeta_vec = zeta_vec
    f_nonlinear2.lambda_vec = lambda_vec
    f_nonlinear2.omega_vec = omega_vec

    f_nonlinear2.xi_base = xi_base
    f_nonlinear2.delta_base = delta_base

    f_nonlinear2.W_mn = W_mn
    f_nonlinear2.cf = cf

    f_nonlinear2.t_original = t_original
    f_nonlinear2.x_traj = x_traj
    f_nonlinear2.y_traj = y_traj

    f_nonlinear2.M_modal = M_modal
    f_nonlinear2.decimal_places = decimal_places

    # ---------------- 7) ورودی u و شرایط اولیه ----------------
    u = np.array([650.0, 4.5])
    f_nonlinear2.ac = u[1]

    x0 = np.array([0.01, 0, 0.005, 0, 0.01, 0, 0.003, 0])
    t_span = (0.0, 2.0)

    # MATLAB step-size constraints
    tau = (2 * np.pi) / (N * u[0])
    safeMax = min(5e-2, tau)

    # ---------------- 8) حل ODE (معادل ode15s) ----------------
    sol = solve_ivp(
        fun=lambda T, X: f_nonlinear2.f_nonlinear2(T, X, u),
        t_span=t_span,
        y0=x0,
        method="BDF",
        max_step=safeMax,
        first_step=1e-3,
        rtol=1e-1,
        atol=1e-1,
    )

    t_sol = sol.t
    x_sol = sol.y.T

    # ---------------- 9) ساخت ماتریس C (دقیقاً مثل MATLAB) ----------------
    xstar = -0.0985
    ystar = -0.0941

    C = np.zeros(2 * K)
    cnt = 0

    for m in range(m_max):
        for n in range(n_max):
            cnt += 1
            Wk = W_mn[m][n]              # تابع شکل مود
            C[2 * (cnt - 1)] = Wk(xstar, ystar)
            C[2 * (cnt - 1) + 1] = 0.0   # سرعت‌ها صفر

    # ---------------- 10) خروجی y(t) = C·x(t) ----------------
    y_sol = C @ x_sol.T
    y_sol = y_sol.reshape(-1, 1)   # مشابه MATLAB: خروجی یک ستون

    return t_sol, x_sol, y_sol


if __name__ == "__main__":
    t, x, y = run_nl_last_time()
    print("t shape:", t.shape)
    print("x shape:", x.shape)
    print("y shape:", y.shape)
