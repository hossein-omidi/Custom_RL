#!/usr/bin/env python3
"""
Final journal-style schematic for the differentiable cutting-force model.

(a) Chip-loss regularisation

(b) Unified geometric engagement and phase-cell integration:
    - engagement support E_i = [theta_s, theta_e]
    - binary engagement indicator g_i(theta)
    - uniform phase-cell decomposition over 0 <= theta < 2*pi
    - natural entry / full / exit intersection conditions
    - selected phase cell C_k
    - exact active interval A_{i,r,k} = C_k ∩ E_i
    - attached midpoint-quadrature zoom

The layout is intentionally compact and annotation-light:
every symbol is placed next to the geometric object it represents.
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle, ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# ============================================================
# Journal style
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "DejaVu Serif"
    ],
    "mathtext.fontset": "stix",

    "font.size": 7.2,
    "axes.labelsize": 7.8,
    "axes.titlesize": 8.5,

    "xtick.labelsize": 6.6,
    "ytick.labelsize": 6.6,

    "legend.fontsize": 6.0,

    "axes.linewidth": 0.60,

    "figure.dpi": 200,
    "savefig.dpi": 600
})


# ============================================================
# Plotting helpers
# ============================================================

def style_axis(ax, grid=False):

    ax.tick_params(
        direction="in",
        length=2.8,
        width=0.48
    )

    if grid:
        ax.grid(
            True,
            linestyle=":",
            linewidth=0.28,
            alpha=0.18
        )


def open_axis(ax):

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def thin_spines(ax, lw=0.50):

    for spine in ax.spines.values():
        spine.set_linewidth(lw)


# ============================================================
# Chip-loss regularisation
# ============================================================

eps = 2e-3


def psi_eps(h):

    return 0.5 * (
        h + np.sqrt(h*h + eps*eps)
    )


def chi_eps(h):

    return 0.5 * (
        1.0
        +
        h / np.sqrt(h*h + eps*eps)
    )


# ============================================================
# Engagement geometry
# ============================================================

two_pi = 2.0 * np.pi

# Reference-tooth physical engagement window
theta_entry = 1.58
theta_exit  = 3.05


# ============================================================
# Uniform phase-cell discretisation
# ============================================================

n_cells = 10

cell_edges = np.linspace(
    0.0,
    two_pi,
    n_cells + 1
)

delta_theta = (
    cell_edges[1] - cell_edges[0]
)


# Cell containing tooth entry
entry_cell = (
    np.searchsorted(
        cell_edges,
        theta_entry,
        side="right"
    )
    - 1
)


# Cell containing tooth exit
exit_cell = (
    np.searchsorted(
        cell_edges,
        theta_exit,
        side="right"
    )
    - 1
)


# ============================================================
# Exact active interval
# ============================================================

def active_interval_in_cell(k):
    """
    Exact intersection between the numerical phase cell

        C_k = [theta_k, theta_k + Delta theta]

    and the physical engagement support

        E_i = [theta_s, theta_e].

    If the intersection is non-empty,

        A_{i,r,k}
        =
        [a_{i,r,k}, b_{i,r,k}].
    """

    c0 = cell_edges[k]
    c1 = cell_edges[k + 1]

    a = max(
        c0,
        theta_entry
    )

    b = min(
        c1,
        theta_exit
    )

    active = (
        b > a
    )

    return (
        c0,
        c1,
        a,
        b,
        active
    )


# ============================================================
# Engagement indicator
# ============================================================

theta = np.linspace(
    0.0,
    two_pi,
    3000
)


g_ref = np.where(
    (theta >= theta_entry)
    &
    (theta <= theta_exit),
    1.0,
    0.0
)


# ============================================================
# Figure construction
# ============================================================

fig = plt.figure(
    figsize=(7.35, 3.15)
)


outer = fig.add_gridspec(
    1,
    2,
    width_ratios=[
        1.00,
        1.86
    ],
    wspace=0.27
)


ax1 = fig.add_subplot(
    outer[0, 0]
)


ax2 = fig.add_subplot(
    outer[0, 1]
)


# ============================================================
# PANEL (a)
# Chip-loss regularisation
# ============================================================

h = np.linspace(
    -0.01,
    0.01,
    1000
)


ax1.plot(
    h,
    np.maximum(h, 0),
    color="black",
    linewidth=1.00,
    label=r"$\max(h,0)$"
)


ax1.plot(
    h,
    psi_eps(h),
    linestyle=(0, (4, 2)),
    color="tab:blue",
    linewidth=1.10,
    label=r"$\psi_\epsilon(h)$"
)


ax1.axhline(
    0,
    linestyle=":",
    color="0.55",
    linewidth=0.48
)


ax1.axvline(
    0,
    linestyle=":",
    color="0.55",
    linewidth=0.48
)


ax1.scatter(
    [0],
    [eps / 2],
    s=15,
    color="tab:red",
    zorder=5
)


ax1.set_title(
    "(a) Chip-loss regularisation"
)


ax1.set_xlabel(
    r"Chip thickness $h$ [mm]"
)


ax1.set_ylabel(
    r"Effective thickness [mm]"
)


ax1.set_xlim(
    -0.01,
    0.01
)


ax1.set_ylim(
    -0.0005,
    0.0105
)


style_axis(
    ax1,
    grid=True
)


ax1.legend(
    frameon=False,
    loc="upper right",
    handlelength=1.6
)


# ------------------------------------------------------------
# Activation inset
# ------------------------------------------------------------

ins1 = inset_axes(
    ax1,
    width="31%",
    height="24%",
    loc="upper left",
    borderpad=1.45
)


ins1.step(
    h,
    (h > 0).astype(float),
    where="post",
    color="tab:blue",
    linewidth=0.78
)


ins1.plot(
    h,
    chi_eps(h),
    linestyle=(0, (4, 2)),
    color="tab:red",
    linewidth=0.88
)


ins1.set_xlim(
    -0.006,
    0.006
)


ins1.set_ylim(
    -0.05,
    1.05
)


ins1.set_xticks([])

ins1.set_yticks(
    [0, 1]
)


ins1.tick_params(
    direction="in",
    labelsize=5.0,
    length=1.8,
    width=0.42,
    pad=1.0
)


thin_spines(
    ins1,
    0.48
)


# ============================================================
# PANEL (b)
# Geometric engagement + phase-cell integration
# ============================================================


# ============================================================
# Engagement-support shading
# ============================================================

ax2.axvspan(
    theta_entry,
    theta_exit,
    ymin=0.30,
    ymax=0.91,
    color="tab:blue",
    alpha=0.08,
    zorder=0
)


# ============================================================
# Uniform phase-cell boundaries
#
# These vertical lines deliberately continue through the
# engagement curve and the lower phase-cell ribbon.
# ============================================================

for edge in cell_edges:

    ax2.vlines(
        edge,
        -0.34,
        1.02,
        color="0.84",
        linewidth=0.38,
        zorder=0.4
    )


# ============================================================
# Binary engagement indicator
# ============================================================

ax2.step(
    theta,
    g_ref,
    where="post",
    color="tab:blue",
    linewidth=1.12,
    zorder=4
)


# ============================================================
# Entry / exit boundaries
# ============================================================

for angle in [
    theta_entry,
    theta_exit
]:

    ax2.axvline(
        angle,
        linestyle="--",
        color="0.38",
        linewidth=0.62,
        zorder=3
    )


# ============================================================
# Engagement support E_i
# ============================================================

engagement_bar_y = 1.085


ax2.plot(
    [
        theta_entry,
        theta_exit
    ],
    [
        engagement_bar_y,
        engagement_bar_y
    ],
    color="0.35",
    linewidth=0.55,
    clip_on=False,
    zorder=6
)


ax2.vlines(
    [
        theta_entry,
        theta_exit
    ],
    engagement_bar_y - 0.025,
    engagement_bar_y + 0.025,
    color="0.35",
    linewidth=0.55,
    clip_on=False,
    zorder=6
)


ax2.text(
    0.5 * (
        theta_entry
        +
        theta_exit
    ),
    engagement_bar_y + 0.018,
    r"$E_i$",
    fontsize=6.0,
    ha="center",
    va="bottom"
)


ax2.text(
    theta_entry,
    1.015,
    r"$\theta_s$",
    fontsize=5.8,
    ha="right",
    va="bottom"
)


ax2.text(
    theta_exit,
    1.015,
    r"$\theta_e$",
    fontsize=5.8,
    ha="left",
    va="bottom"
)


# ============================================================
# Phase-cell ribbon
#
# Grey:
# complete numerical phase cells
#
# Blue:
# exact engaged part C_k ∩ E_i
#
# Therefore entry / full / exit cases appear naturally.
# ============================================================

strip_y = -0.30
strip_h = 0.145


for k in range(n_cells):

    c0 = cell_edges[k]
    c1 = cell_edges[k + 1]


    # Alternating cell shading improves discretisation visibility.
    face = (
        "0.89"
        if k % 2 == 0
        else "0.95"
    )


    # Cells involved in engagement receive a slightly
    # stronger outline.
    important = (
        k in {
            entry_cell,
            entry_cell + 1,
            exit_cell
        }
    )


    edge_color = (
        "0.48"
        if important
        else "0.70"
    )


    edge_width = (
        0.60
        if important
        else 0.35
    )


    # Complete numerical cell C_k
    ax2.add_patch(
        Rectangle(
            (
                c0,
                strip_y
            ),
            c1 - c0,
            strip_h,
            facecolor=face,
            edgecolor=edge_color,
            linewidth=edge_width,
            zorder=2
        )
    )


    # Exact active portion
    _, _, a_int, b_int, active = (
        active_interval_in_cell(k)
    )


    if active:

        ax2.add_patch(
            Rectangle(
                (
                    a_int,
                    strip_y
                ),
                b_int - a_int,
                strip_h,
                facecolor="tab:blue",
                alpha=0.36,
                edgecolor="none",
                zorder=3
            )
        )


# ============================================================
# Compact case labels
# ============================================================

for text, k in [
    ("entry", entry_cell),
    ("full",  entry_cell + 1),
    ("exit",  exit_cell)
]:

    x_mid = 0.5 * (
        cell_edges[k]
        +
        cell_edges[k + 1]
    )

    ax2.text(
        x_mid,
        strip_y + strip_h + 0.032,
        text,
        fontsize=5.2,
        color="0.35",
        ha="center",
        va="bottom"
    )


# ============================================================
# Phase-cell width Delta theta
#
# It is shown on an inactive cell to avoid interference
# with the engagement geometry.
# ============================================================

delta_y = -0.08


ax2.annotate(
    "",
    xy=(
        cell_edges[1],
        delta_y
    ),
    xytext=(
        cell_edges[0],
        delta_y
    ),
    arrowprops=dict(
        arrowstyle="<->",
        color="0.45",
        lw=0.55
    )
)


ax2.text(
    0.5 * (
        cell_edges[0]
        +
        cell_edges[1]
    ),
    delta_y + 0.018,
    r"$\Delta\theta$",
    fontsize=5.7,
    color="0.35",
    ha="center",
    va="bottom"
)


# ============================================================
# Select the exit-crossing phase cell
#
# This choice is deliberate:
#     0 < b-a < Delta theta
#
# so the geometric extraction is immediately visible.
# ============================================================

c0_q, c1_q, a_q, b_q, active_q = (
    active_interval_in_cell(
        exit_cell
    )
)


assert active_q


# Complete selected phase cell
ax2.add_patch(
    Rectangle(
        (
            c0_q,
            strip_y
        ),
        c1_q - c0_q,
        strip_h,
        facecolor="none",
        edgecolor="0.25",
        linewidth=0.85,
        zorder=5
    )
)


# Phase-cell notation
ax2.text(
    0.5 * (
        c0_q
        +
        c1_q
    ),
    strip_y - 0.025,
    r"$C_k$",
    fontsize=5.8,
    color="0.25",
    ha="center",
    va="top"
)


# Active-interval notation
ax2.text(
    0.5 * (
        a_q
        +
        b_q
    ),
    strip_y + 0.5 * strip_h,
    r"$A_{i,r,k}$",
    fontsize=5.1,
    color="tab:blue",
    ha="center",
    va="center"
)


# ============================================================
# Attached quadrature inset
#
# Positioned entirely in the disengaged upper-right region.
# ============================================================

ins2 = inset_axes(
    ax2,
    width="100%",
    height="100%",
    bbox_to_anchor=(
        0.595,
        0.415,
        0.375,
        0.42
    ),
    bbox_transform=ax2.transAxes,
    loc="lower left",
    borderpad=0
)


# ============================================================
# Midpoint quadrature
# ============================================================

nq = 4


delta_vartheta = (
    b_q - a_q
) / nq


quad_edges = (
    a_q
    +
    np.arange(nq + 1)
    * delta_vartheta
)


quad_x = (
    a_q
    +
    (
        np.arange(nq)
        + 0.5
    )
    * delta_vartheta
)


# ============================================================
# Smooth schematic force integrand
# ============================================================

xq = np.linspace(
    a_q,
    b_q,
    600
)


s = (
    xq - a_q
) / (
    b_q - a_q
)


fq = (
    0.205
    +
    0.035
    * np.sin(
        np.pi
        * (
            0.12
            +
            0.72 * s
        )
    )
    -
    0.008 * s
)


sq = (
    quad_x - a_q
) / (
    b_q - a_q
)


quad_y = (
    0.205
    +
    0.035
    * np.sin(
        np.pi
        * (
            0.12
            +
            0.72 * sq
        )
    )
    -
    0.008 * sq
)


# ============================================================
# Full selected phase cell
# ============================================================

ins2.axvspan(
    c0_q,
    c1_q,
    color="0.55",
    alpha=0.07,
    zorder=0
)


# ============================================================
# Extracted active interval
# ============================================================

ins2.axvspan(
    a_q,
    b_q,
    color="tab:blue",
    alpha=0.095,
    zorder=0.5
)


# ============================================================
# Small geometric ribbon inside zoom
#
# Repeats the exact geometry of the main phase-cell ribbon.
# ============================================================

mask_y = 0.300
mask_h = 0.018


# Full C_k
ins2.add_patch(
    Rectangle(
        (
            c0_q,
            mask_y
        ),
        c1_q - c0_q,
        mask_h,
        facecolor="0.87",
        edgecolor="0.60",
        linewidth=0.32,
        zorder=6
    )
)


# Active A_{i,r,k}
ins2.add_patch(
    Rectangle(
        (
            a_q,
            mask_y
        ),
        b_q - a_q,
        mask_h,
        facecolor="tab:blue",
        alpha=0.55,
        edgecolor="none",
        zorder=7
    )
)


# ============================================================
# Non-engaged remainder of selected exit cell
# ============================================================

ins2.hlines(
    0.0,
    b_q,
    c1_q,
    color="0.60",
    linewidth=0.70,
    zorder=2
)


# ============================================================
# Quadrature subinterval boundaries
# ============================================================

for edge in quad_edges:

    ins2.axvline(
        edge,
        color="0.68",
        linewidth=0.38,
        zorder=1
    )


# ============================================================
# Midpoint guides
# ============================================================

for xm in quad_x:

    ins2.axvline(
        xm,
        linestyle=":",
        color="0.56",
        linewidth=0.28,
        zorder=1.2
    )


# ============================================================
# Midpoint rectangles
# ============================================================

for j in range(nq):

    ins2.add_patch(
        Rectangle(
            (
                a_q
                +
                j * delta_vartheta,
                0.0
            ),
            delta_vartheta,
            quad_y[j],
            facecolor="tab:red",
            alpha=0.09,
            edgecolor="0.55",
            linewidth=0.28,
            zorder=2
        )
    )


# ============================================================
# Smooth integrand
# ============================================================

ins2.plot(
    xq,
    fq,
    color="tab:blue",
    linewidth=1.08,
    zorder=4
)


# ============================================================
# Midpoint samples
# ============================================================

ins2.scatter(
    quad_x,
    quad_y,
    marker="x",
    s=17,
    linewidth=0.82,
    color="tab:red",
    zorder=5
)


# ============================================================
# Exact active-interval boundaries
# ============================================================

for bound in [
    a_q,
    b_q
]:

    ins2.axvline(
        bound,
        linestyle="--",
        color="tab:blue",
        linewidth=0.68,
        zorder=4
    )


# ============================================================
# Representative midpoint notation
# ============================================================

ins2.text(
    quad_x[1],
    quad_y[1] + 0.015,
    r"$\vartheta_j$",
    fontsize=4.9,
    color="tab:red",
    ha="center",
    va="bottom"
)


# ============================================================
# Quadrature subinterval width
# ============================================================

ins2.text(
    0.50,
    0.055,
    r"$\Delta\vartheta=(b-a)/n_q$",
    transform=ins2.transAxes,
    fontsize=4.9,
    color="0.35",
    ha="center",
    va="bottom"
)


# ============================================================
# Inset formatting
# ============================================================

pad_q = (
    0.025 * delta_theta
)


ins2.set_xlim(
    c0_q - pad_q,
    c1_q + pad_q
)


ins2.set_ylim(
    -0.012,
    0.335
)


# Only the true active bounds are labelled.
# The complete C_k is already identified in the parent plot.
ins2.set_xticks(
    [
        a_q,
        b_q
    ],
    [
        r"$a_{i,r,k}$",
        r"$b_{i,r,k}$"
    ]
)


ins2.set_yticks([])


ins2.set_xlabel(
    r"$\vartheta$",
    fontsize=5.5,
    labelpad=0.45
)


ins2.tick_params(
    direction="in",
    length=1.9,
    width=0.40,
    labelsize=5.0,
    pad=1.0
)


thin_spines(
    ins2,
    0.48
)


# ============================================================
# Geometric connection:
# selected C_k -> quadrature zoom
#
# Dotted neutral lines are deliberately used instead of arrows.
# ============================================================

connection_left = ConnectionPatch(
    xyA=(
        c0_q,
        strip_y + strip_h
    ),
    coordsA=ax2.transData,

    xyB=(
        0.0,
        0.0
    ),
    coordsB=ins2.transAxes,

    color="0.58",
    linewidth=0.42,
    linestyle=":",
    alpha=0.68,
    zorder=1
)


connection_right = ConnectionPatch(
    xyA=(
        c1_q,
        strip_y + strip_h
    ),
    coordsA=ax2.transData,

    xyB=(
        1.0,
        0.0
    ),
    coordsB=ins2.transAxes,

    color="0.58",
    linewidth=0.42,
    linestyle=":",
    alpha=0.68,
    zorder=1
)


fig.add_artist(
    connection_left
)


fig.add_artist(
    connection_right
)


# ============================================================
# Panel (b) formatting
# ============================================================

ax2.set_title(
    "(b) Geometric engagement and phase-cell integration",
    pad=5
)


ax2.set_xlabel(
    r"Spindle phase $\theta$ [rad]"
)


ax2.set_ylabel(
    r"$g_i(\theta)$"
)


ax2.set_xlim(
    0,
    two_pi
)


ax2.set_ylim(
    -0.42,
    1.15
)


ax2.set_yticks(
    [0, 1]
)


ax2.set_xticks(
    [
        0,
        np.pi,
        two_pi
    ],
    [
        r"$0$",
        r"$\pi$",
        r"$2\pi$"
    ]
)


style_axis(
    ax2,
    grid=False
)


open_axis(
    ax2
)


# ============================================================
# Final layout
# ============================================================

fig.subplots_adjust(
    left=0.070,
    right=0.988,
    bottom=0.19,
    top=0.88
)


plt.savefig(
    "figure_differentiable_cutting_force_FINAL.pdf",
    bbox_inches="tight"
)


plt.savefig(
    "figure_differentiable_cutting_force_FINAL.png",
    dpi=600,
    bbox_inches="tight"
)


plt.show()