# =============================================================================
# Publication figure: spindle-phase-synchronous MHE–NMPC execution
#
# Uses ACTUAL controller-level diagnostics saved by mpc_face_milling_DS_2.py.
#
# Implemented timing architecture:
#
#   theta_{k+1} - theta_k = Delta_theta                (nominal optimizer grid)
#
# while the physical simulator advances with a fixed integration step dt_env.
# Each controller command is held for an integer number of simulator steps,
# producing
#
#   Delta t_k = n_hold,k * dt_env
#
# and therefore a realized phase increment
#
#   Delta theta_k^real ~= Delta theta.
#
# After every controller interval, the scheduler is re-synchronized to the
# phase reported by the plant. The final episode interval is excluded because
# it may terminate before the planned hold interval has completed; this matches
# the verification convention used in mpc_face_milling_DS_2.py.
# =============================================================================

from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# =============================================================================
# USER OPTIONS
# =============================================================================

SEED = 0
EPISODE_INDEX = 0

# Number of consecutive completed controller intervals displayed.
N_SHOW = 20

# Optional global controller index at which the displayed window begins.
# If None, a deterministic post-MHE-warm-up window is used.
START_CTRL = 50

OUTPUT_FOLDER = "paper_figures"
FIGURE_NAME = "phase_synchronous_MHE_NMPC"


# =============================================================================
# 1. PATHS
# =============================================================================

# Expected location:
#   Custom_RL/scripts/phase_trigger_plot.py
ROOT = Path(__file__).resolve().parents[1]

TRAJECTORY_FILE = (
    ROOT
    / "mpc_phase_trajectories"
    / f"trajectories_seed{SEED}.json"
)

if not TRAJECTORY_FILE.exists():
    raise FileNotFoundError(
        "\nTrajectory file was not found:\n"
        f"    {TRAJECTORY_FILE}\n\n"
        "Run mpc_face_milling_DS_2.py first, or edit TRAJECTORY_FILE.\n"
    )


# =============================================================================
# 2. LOAD EPISODE
# =============================================================================

with TRAJECTORY_FILE.open("r", encoding="utf-8") as f:
    raw = json.load(f)

if isinstance(raw, list):
    if not raw:
        raise RuntimeError("The trajectory JSON contains no episodes.")
    if not (0 <= EPISODE_INDEX < len(raw)):
        raise IndexError(
            f"EPISODE_INDEX={EPISODE_INDEX}, but the file contains "
            f"{len(raw)} episode(s)."
        )
    episode = raw[EPISODE_INDEX]
else:
    episode = raw


# =============================================================================
# 3. EXTRACT CONTROLLER-LEVEL DIAGNOSTICS
# =============================================================================

diag = episode.get("mpc_diagnostics", {})
summary = episode.get("summary", {})
mhe_info = episode.get("mhe", {})
metadata = episode.get("metadata", {})
mpc_meta = metadata.get("mpc", {}) if isinstance(metadata, dict) else {}

omega_ctrl = np.asarray(
    diag.get("u_applied_omega", []),
    dtype=float,
)

dt_interval = np.asarray(
    diag.get("dt_interval_s", []),
    dtype=float,
)

dtheta_realized = np.asarray(
    diag.get("dtheta_realized", []),
    dtype=float,
)

dtheta_nominal = float(
    mhe_info.get(
        "dtheta_rad",
        summary.get(
            "dtheta_nominal_rad",
            mpc_meta.get("dtheta_rad", np.nan),
        ),
    )
)

mhe_horizon = int(
    mpc_meta.get(
        "mhe_horizon",
        0,
    )
)


# =============================================================================
# 4. CONSISTENCY CHECKS
# =============================================================================

if omega_ctrl.size == 0:
    raise RuntimeError(
        "Missing mpc_diagnostics['u_applied_omega']."
    )

if dt_interval.size == 0:
    raise RuntimeError(
        "Missing mpc_diagnostics['dt_interval_s']."
    )

if dtheta_realized.size == 0:
    raise RuntimeError(
        "Missing mpc_diagnostics['dtheta_realized']."
    )

if not np.isfinite(dtheta_nominal) or dtheta_nominal <= 0.0:
    raise RuntimeError(
        "A valid nominal phase increment Delta theta was not found."
    )

n_common = min(
    omega_ctrl.size,
    dt_interval.size,
    dtheta_realized.size,
)

if n_common < 3:
    raise RuntimeError(
        "At least three logged controller intervals are required."
    )

# --------------------------------------------------------------------------
# IMPORTANT:
# The MPC code itself evaluates dtheta accuracy using dtheta_realized[:-1].
# The final logged interval may be truncated by episode termination. Exclude
# it consistently from ALL timing quantities used in this publication figure.
# --------------------------------------------------------------------------

n_complete = n_common - 1

omega_ctrl = omega_ctrl[:n_complete]
dt_interval = dt_interval[:n_complete]
dtheta_realized = dtheta_realized[:n_complete]

# Do not silently remove isolated invalid samples: that would join intervals
# that were not consecutive in time.
valid = (
    np.isfinite(omega_ctrl)
    & np.isfinite(dt_interval)
    & np.isfinite(dtheta_realized)
    & (omega_ctrl > 0.0)
    & (dt_interval > 0.0)
)

if not np.all(valid):
    bad = np.flatnonzero(~valid)
    raise RuntimeError(
        "Invalid controller-level data were found at completed interval "
        f"indices {bad.tolist()}. The sequence is not filtered because doing "
        "so would destroy temporal continuity."
    )

n_intervals = omega_ctrl.size


# =============================================================================
# 5. SELECT A DETERMINISTIC PUBLICATION WINDOW
# =============================================================================
#
# Avoid choosing the window by maximizing signal variation; such a data-driven
# selection can look like cherry-picking in a paper. By default, start after
# the MHE warm-up period used in the closed-loop implementation.
# =============================================================================

N_SHOW = int(min(max(N_SHOW, 2), n_intervals))

if START_CTRL is None:
    # In the implementation the MHE estimate is used only after:
    #     n_ctrl > mhe_horizon + 1
    # Hence this is a natural deterministic starting point.
    i0 = min(
        mhe_horizon + 1,
        max(0, n_intervals - N_SHOW),
    )
else:
    i0 = int(START_CTRL)
    if i0 < 0:
        raise ValueError("START_CTRL must be non-negative.")
    i0 = min(
        i0,
        max(0, n_intervals - N_SHOW),
    )

i1 = i0 + N_SHOW

omega_show = omega_ctrl[i0:i1]
dt_show = dt_interval[i0:i1]
dtheta_show = dtheta_realized[i0:i1]


# =============================================================================
# 6. LOCAL TIME AND PHASE COORDINATES
# =============================================================================
#
# The selected window is shifted to t=0 and theta-theta_start=0 only for
# presentation. No physical information is lost.
# =============================================================================

t_trigger = np.concatenate(
    ([0.0], np.cumsum(dt_show))
)

theta_real = np.concatenate(
    ([0.0], np.cumsum(dtheta_show))
)

theta_nominal = (
    np.arange(N_SHOW + 1, dtype=float)
    * dtheta_nominal
)


# =============================================================================
# 7. INTERPOLATED PHASE CURVE FOR DISPLAY ONLY
# =============================================================================
#
# The JSON stores controller-level update points, not a continuous encoder
# trace. Linear interpolation is used only to connect actual phase-update
# points visually; all update markers themselves are logged values.
# =============================================================================

t_dense_parts = []
theta_dense_parts = []

for k in range(N_SHOW):
    tt = np.linspace(
        t_trigger[k],
        t_trigger[k + 1],
        80,
        endpoint=False,
    )

    tau = (
        (tt - t_trigger[k])
        / (t_trigger[k + 1] - t_trigger[k])
    )

    th = (
        theta_real[k]
        + tau
        * (theta_real[k + 1] - theta_real[k])
    )

    t_dense_parts.append(tt)
    theta_dense_parts.append(th)

t_dense_parts.append(
    np.array([t_trigger[-1]])
)

theta_dense_parts.append(
    np.array([theta_real[-1]])
)

t_dense = np.concatenate(t_dense_parts)
theta_dense = np.concatenate(theta_dense_parts)


# =============================================================================
# 8. NUMERICAL PHASE-STEP VERIFICATION
# =============================================================================

relative_phase_error = (
    (dtheta_show - dtheta_nominal)
    / dtheta_nominal
)

max_rel_phase_error = float(
    np.max(np.abs(relative_phase_error))
)


# =============================================================================
# 9. PUBLICATION STYLE
# =============================================================================

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(7.1, 4.9),
    sharex=True,
    gridspec_kw={
        "height_ratios": [1.05, 1.0],
        "hspace": 0.10,
    },
)


# =============================================================================
# PANEL (a): PHASE-SYNCHRONOUS UPDATE GRID
# =============================================================================

ax1.plot(
    t_dense,
    theta_dense,
    linewidth=1.6,
    label=r"realized phase $\theta-\theta_{\rm start}$",
)

ax1.scatter(
    t_trigger,
    theta_real,
    s=22,
    zorder=3,
    label="MHE–NMPC updates",
)

for tk in t_trigger:
    ax1.axvline(
        tk,
        linestyle=":",
        linewidth=0.65,
        alpha=0.30,
    )

# A single nominal phase-step annotation is sufficient.
k_phase = min(1, N_SHOW - 1)

x_phase = (
    t_trigger[k_phase]
    + 0.18
    * (
        t_trigger[k_phase + 1]
        - t_trigger[k_phase]
    )
)

ax1.annotate(
    "",
    xy=(
        x_phase,
        theta_real[k_phase + 1],
    ),
    xytext=(
        x_phase,
        theta_real[k_phase],
    ),
    arrowprops=dict(
        arrowstyle="<->",
        linewidth=0.9,
    ),
)

ax1.text(
    x_phase,
    0.5
    * (
        theta_real[k_phase]
        + theta_real[k_phase + 1]
    ),
    r"  $\Delta\theta_k^{\rm real}\approx\Delta\theta$",
    ha="left",
    va="center",
    fontsize=8,
)

# Compact verification note instead of a nearly overlapping nominal curve.
ax1.text(
    0.985,
    0.055,
    (
        rf"nominal $\Delta\theta={dtheta_nominal:.4g}$ rad"
        "\n"
        rf"max. displayed deviation $={100*max_rel_phase_error:.2f}\%$"
    ),
    transform=ax1.transAxes,
    ha="right",
    va="bottom",
    fontsize=7.5,
)

ax1.set_ylabel(
    r"relative spindle phase [rad]"
)

ax1.text(
    0.015,
    0.94,
    "(a)",
    transform=ax1.transAxes,
    ha="left",
    va="top",
    fontweight="bold",
)

ax1.grid(
    True,
    alpha=0.20,
)

ax1.legend(
    frameon=False,
    loc="upper left",
)


# =============================================================================
# PANEL (b): VARIABLE PHYSICAL INTER-UPDATE INTERVAL
# =============================================================================

omega_step = np.concatenate(
    (
        omega_show,
        [omega_show[-1]],
    )
)

ax2.step(
    t_trigger,
    omega_step,
    where="post",
    linewidth=1.6,
    label=r"applied $\omega_k$ (ZOH)",
)

ax2.scatter(
    t_trigger[:-1],
    omega_show,
    s=18,
    zorder=3,
)

for tk in t_trigger:
    ax2.axvline(
        tk,
        linestyle=":",
        linewidth=0.65,
        alpha=0.30,
    )


# -----------------------------------------------------------------------------
# Annotate shortest and longest displayed physical intervals BELOW the curve.
# This avoids crossing the speed trajectory and other annotations.
# -----------------------------------------------------------------------------

k_short = int(np.argmin(dt_show))
k_long = int(np.argmax(dt_show))

if k_short == k_long and N_SHOW > 1:
    k_long = (k_short + 1) % N_SHOW

omega_min = float(np.min(omega_show))
omega_max = float(np.max(omega_show))
omega_span = max(
    omega_max - omega_min,
    1.0,
)

y_short = omega_min - 0.12 * omega_span
y_long = omega_min - 0.28 * omega_span


def annotate_dt(ax, k_interval, y, text):
    x0 = t_trigger[k_interval]
    x1 = t_trigger[k_interval + 1]

    ax.annotate(
        "",
        xy=(x1, y),
        xytext=(x0, y),
        arrowprops=dict(
            arrowstyle="<->",
            linewidth=0.9,
        ),
    )

    ax.text(
        0.5 * (x0 + x1),
        y - 0.025 * omega_span,
        text,
        ha="center",
        va="top",
        fontsize=7.5,
    )


annotate_dt(
    ax2,
    k_short,
    y_short,
    rf"$\Delta t_{{{i0+k_short}}}"
    rf"={1000*dt_show[k_short]:.2f}\,$ms",
)

annotate_dt(
    ax2,
    k_long,
    y_long,
    rf"$\Delta t_{{{i0+k_long}}}"
    rf"={1000*dt_show[k_long]:.2f}\,$ms",
)

ax2.set_xlabel(
    r"physical time from start of displayed window [ms]"
)

ax2.set_ylabel(
    r"spindle speed $\omega$ [rad/s]"
)

ax2.xaxis.set_major_formatter(
    FuncFormatter(
        lambda x, pos:
        f"{1000*x:.1f}"
    )
)

ax2.set_ylim(
    omega_min - 0.40 * omega_span,
    omega_max + 0.10 * omega_span,
)

ax2.text(
    0.015,
    0.94,
    "(b)",
    transform=ax2.transAxes,
    ha="left",
    va="top",
    fontweight="bold",
)

ax2.grid(
    True,
    alpha=0.20,
)

ax2.legend(
    frameon=False,
    loc="upper right",
)

ax2.set_xlim(
    t_trigger[0],
    t_trigger[-1],
)

fig.align_ylabels(
    (ax1, ax2)
)


# =============================================================================
# 10. SAVE VECTOR AND HIGH-RESOLUTION OUTPUTS
# =============================================================================

OUTPUT_DIR = ROOT / OUTPUT_FOLDER
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

pdf_path = (
    OUTPUT_DIR
    / f"{FIGURE_NAME}.pdf"
)

svg_path = (
    OUTPUT_DIR
    / f"{FIGURE_NAME}.svg"
)

png_path = (
    OUTPUT_DIR
    / f"{FIGURE_NAME}.png"
)

fig.savefig(
    pdf_path,
    bbox_inches="tight",
)

fig.savefig(
    svg_path,
    bbox_inches="tight",
)

fig.savefig(
    png_path,
    dpi=600,
    bbox_inches="tight",
)


# =============================================================================
# 11. NUMERICAL VERIFICATION
# =============================================================================

print("=" * 72)
print("Phase-synchronous MHE–NMPC timing verification")
print("=" * 72)

print(
    f"Trajectory file             : {TRAJECTORY_FILE}"
)

print(
    f"Completed intervals         : {n_intervals}"
)

print(
    f"Displayed global indices    : {i0} ... {i1-1}"
)

print(
    f"Nominal Delta theta         : {dtheta_nominal:.8f} rad"
)

print(
    f"Mean realized Delta theta   : {np.mean(dtheta_show):.8f} rad"
)

print(
    f"Std realized Delta theta    : {np.std(dtheta_show):.3e} rad"
)

print(
    f"Max relative deviation      : "
    f"{100*max_rel_phase_error:.3f} %"
)

print(
    f"Minimum displayed Delta t   : "
    f"{1000*np.min(dt_show):.4f} ms"
)

print(
    f"Maximum displayed Delta t   : "
    f"{1000*np.max(dt_show):.4f} ms"
)

print(
    f"Displayed omega range       : "
    f"{np.min(omega_show):.4f} ... "
    f"{np.max(omega_show):.4f} rad/s"
)

# Under ZOH, omega_k * Delta t_k is the nominal phase advance implied by the
# held command and integer physical-time hold. Compare it against the phase
# increment actually reported by the plant after re-synchronization.
hold_phase = (
    omega_show
    * dt_show
)

hold_vs_real_error = (
    hold_phase
    - dtheta_show
)

print(
    f"Mean |omega_k*Delta t_k - Delta theta_real| : "
    f"{np.mean(np.abs(hold_vs_real_error)):.6e} rad"
)

print("\nInterpretation:")
print(
    "  - The optimization grid uses a fixed nominal spindle-phase step."
)
print(
    "  - Physical inter-update time varies with the applied spindle speed."
)
print(
    "  - The realized phase step is close to the nominal value because the"
)
print(
    "    physical simulator can hold the input only for an integer number"
)
print(
    "    of fixed-time integration steps."
)
print(
    "  - The scheduler is re-synchronized to the plant-reported phase after"
)
print(
    "    every completed controller interval."
)

print("\nSaved:")
print(f"  PDF : {pdf_path}")
print(f"  SVG : {svg_path}")
print(f"  PNG : {png_path}")

plt.show()
