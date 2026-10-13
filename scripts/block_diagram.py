#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ppo_architecture_figure.py   (v2.0)
=====================================================================
Publication-quality block diagram of the observation-based PPO training
architecture for the milling chatter-suppression feedback controller.

v2.0 changelog (inspired by the reference PPO diagram)
------------------------------------------------------
* neural-network schematics drawn inside the Actor and Critic boxes
  (layered nodes with input o_t -> output a_t / V_phi), making the
  "network" nature explicit;
* dashed "PPO Agent" boundary grouping actor + critic;
* numbered steps (1)-(6) on the data flow, referenced by the caption;
* reward symbol switched to R_t (capital) following the reference
  convention -- removes the r_t(theta) ratio / r_t reward collision.
  ONE-LINE switch: REWARD = r"r" restores lowercase if your manuscript
  requires it (every equation follows automatically);
* arrow-colour legend (policy / value / interaction pathways);
* synchronization note theta_old <- theta at iteration start.

Dependencies : matplotlib >= 3.4 (numpy ships with it). No LaTeX needed.
Outputs      : ppo_architecture.pdf   vector, fonts embedded -> submit this
               ppo_architecture.png   600 dpi raster preview
               ppo_architecture.svg   optional (SAVE_SVG = True)
Run          : python ppo_architecture_figure.py
"""

import math

import numpy as np                     # guaranteed present (matplotlib dep.)

import matplotlib
matplotlib.use("Agg")                  # headless-safe

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------
W_CM, H_CM = 16.5, 16.0      # final print size (double-column full width)
DPI        = 600
SAVE_SVG   = False
MONOCHROME = False

OUT_PDF = "ppo_architecture.pdf"
OUT_PNG = "ppo_architecture.png"
OUT_SVG = "ppo_architecture.svg"

# --- symbol switches (figure follows the manuscript automatically) --------
# Probability ratio: r_t(theta) as in the manuscript; switch to \rho if you
# later decouple it from the reward symbol.
RATIO  = r"r"
RT     = RATIO + r"_t"
# Reward: R_t (capital, reference convention, no clash with the ratio).
# If your manuscript uses lowercase r_t for reward, set REWARD = r"r".
REWARD = r"R"
RTW    = REWARD + r"_t"

INK, GRAY, C_NEU = "#1A1A1A", "#6E6E6E", "#333333"

if MONOCHROME:
    C_BAN, F_BAN = "#000000", "#FFFFFF"
    C_POL, F_POL = "#111111", "#FFFFFF"
    C_VAL, F_VAL = "#555555", "#F2F2F2"
    C_SIM, F_SIM = "#333333", "#F2F2F2"
    C_BUF, F_BUF = "#444444", "#FFFFFF"
    C_GAE, F_GAE = "#666666", "#FFFFFF"
else:                        # Okabe-Ito palette (colour-vision-deficiency safe)
    C_BAN, F_BAN = "#3B3B3B", "#F5F5F2"   # neutral   : objective banner
    C_POL, F_POL = "#D55E00", "#FCE8DC"   # vermillion: policy path
    C_VAL, F_VAL = "#0072B2", "#E4F0F9"   # blue      : value path
    C_SIM, F_SIM = "#009E73", "#E2F3ED"   # green     : environment
    C_BUF, F_BUF = "#6B6B6B", "#F1F1F1"   # gray      : data storage
    C_GAE, F_GAE = "#A95CA8", "#F7ECF6"   # purple    : GAE / targets

# ---------------------------------------------------------------------------
# 2. GLOBAL STYLE
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype":     42,        # embed TrueType fonts (no Type 3)
    "ps.fonttype":      42,
    "svg.fonttype":     "none",
})

# ---------------------------------------------------------------------------
# 3. CANVAS -- 1 data unit = 1 cm
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(W_CM / 2.54, H_CM / 2.54))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W_CM)
ax.set_ylim(0, H_CM)
ax.set_aspect("equal")
ax.axis("off")

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def box(cx, cy, w, h, edge, face, lw=1.0, rounding=0.10):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        linewidth=lw, edgecolor=edge, facecolor=face, zorder=2))


def arrow(p0, p1, color, lw=1.1):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=13, linewidth=lw,
        color=color, shrinkA=0, shrinkB=0, zorder=3))


def tag(x, y, s, size=6.8, ha="center", color=INK, style="normal",
        rotation=0, bbox_on=False, weight="normal"):
    kw = dict(fontsize=size, ha=ha, va="center", color=color, zorder=7,
              fontstyle=style, fontweight=weight, rotation=rotation,
              linespacing=1.35)
    if bbox_on:
        kw["bbox"] = dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none", alpha=0.92)
    ax.text(x, y, s, **kw)


def mini_net(cx, cy, layers, color, in_label, out_label):
    """Layered neural-network schematic: nodes + sparse connections.
    layers : tuple of ints, nodes per layer (input ... output)."""
    n = len(layers)
    xs = np.linspace(cx - 0.95, cx + 0.95, n)
    coords = []
    for li, k in enumerate(layers):
        span = 0.28 * (k - 1)
        ys = cy + np.linspace(-span / 2, span / 2, k)
        coords.append((xs[li], ys))
    for l in range(n - 1):                       # connections (below nodes)
        x1, ys1 = coords[l]
        x2, ys2 = coords[l + 1]
        for y1 in ys1:
            for y2 in ys2:
                ax.plot([x1, x2], [y1, y2], color=color, lw=0.30,
                        alpha=0.45, zorder=5, solid_capstyle="butt")
    for x, ys in coords:                         # nodes
        for y in ys:
            ax.add_patch(Circle((x, y), 0.078, facecolor="white",
                                edgecolor=color, lw=0.8, zorder=6))
    # input / output mini arrows + labels
    ax.add_patch(FancyArrowPatch((xs[0] - 0.55, cy), (xs[0] - 0.14, cy),
                 arrowstyle="-|>", mutation_scale=8, lw=0.8,
                 color=INK, zorder=6))
    ax.text(xs[0] - 0.62, cy, in_label, fontsize=6.3, ha="right",
            va="center", zorder=6)
    ax.add_patch(FancyArrowPatch((xs[-1] + 0.14, cy), (xs[-1] + 0.55, cy),
                 arrowstyle="-|>", mutation_scale=8, lw=0.8,
                 color=INK, zorder=6))
    ax.text(xs[-1] + 0.62, cy, out_label, fontsize=6.3, ha="left",
            va="center", zorder=6)

# ---------------------------------------------------------------------------
# 4. EQUATIONS (all raw strings; symbol audit: \mathrm \frac \left \right
#    \hat \; \, \mid \varepsilon \gamma \lambda \delta \phi \theta \pi
#    \sum \geq \{ \} \rightarrow \leftarrow \sim \cdot -- all supported.
#    \ge / \le abbreviations are NOT: always use \geq / \leq.)
# ---------------------------------------------------------------------------
EQ_CLIP = (r"$\mathrm{max}_{\theta}\;L^{\mathrm{CLIP}}(\theta)\;=\;"
           r"\hat{E}_{t}\left[\;\mathrm{min}\left(\;"
           + RT +
           r"(\theta)\,\hat{A}_{t},\;\;\mathrm{clip}\left(\;"
           + RT +
           r"(\theta),\,1-\varepsilon,\,1+\varepsilon\right)\hat{A}_{t}"
           r"\;\right)\;\right]$")

EQ_RATIO = (r"$" + RT + r"(\theta)\;=\;"
            r"\frac{\pi_{\theta}(a_t\mid o_t)}"
            r"{\pi_{\theta_{\mathrm{old}}}(a_t\mid o_t)}$")

EQ_CRITIC = (r"$\mathrm{min}_{\phi}\;L^{V}(\phi)\;=\;\hat{E}_{t}"
             r"\left[\left(V_{\phi}(o_t)-V_{\mathrm{target}}\right)^{2}\right]$")

EQ_TD     = (r"$\delta_{t}\;=\;" + RTW +
             r"+\gamma V_{\phi}(o_{t+1})-V_{\phi}(o_{t})$")
EQ_GAE    = r"$\hat{A}_{t}\;=\;\sum_{k\geq 0}(\gamma\lambda)^{k}\,\delta_{t+k}$"
EQ_TGT    = r"$V_{\mathrm{target}}\;=\;\hat{A}_{t}+V_{\phi}(o_{t})$"
EQ_BUFFER = (r"$\{\,o_t,\;a_t,\;" + RTW + r",\;o_{t+1}\,\}$")

# ---------------------------------------------------------------------------
# 5. BLOCKS
# ---------------------------------------------------------------------------
# ---- objective banner -----------------------------------------------------
box(8.25, 14.80, 15.9, 2.2, C_BAN, F_BAN, lw=1.1)
tag(8.25, 15.47, "PPO Policy Optimization", size=9.0, weight="bold")
tag(8.25, 15.00, EQ_CLIP,  size=7.8)
tag(8.25, 14.32, EQ_RATIO, size=7.5)

# ---- actor ----------------------------------------------------------------
box(3.60, 9.90, 6.6, 3.4, C_POL, F_POL)
tag(3.60, 11.22, "Actor Network", size=8.6, weight="bold")
mini_net(3.60, 10.02, (4, 5, 5, 2), C_POL, r"$o_t$", r"$a_t$")
tag(3.60, 9.02, r"$\pi_{\theta}(a_t\mid o_t)$   (stochastic policy)", size=7.3)
tag(3.60, 8.55, r"$\pi_{\theta_{\mathrm{old}}}\;\rightarrow\;\pi_{\theta}$"
                r"   (clipped update)", size=7.1)

# ---- critic ---------------------------------------------------------------
box(12.90, 9.90, 6.6, 3.4, C_VAL, F_VAL)
tag(12.90, 11.22, "Critic Network", size=8.6, weight="bold")
mini_net(12.90, 10.02, (4, 5, 5, 1), C_VAL, r"$o_t$", r"$V_{\phi}$")
tag(12.90, 9.02, r"$V_{\phi}(o_t)$   (value function)", size=7.3)
tag(12.90, 8.55, EQ_CRITIC, size=7.0)

# ---- agent boundary (dashed container around actor + critic) --------------
ax.add_patch(FancyBboxPatch(
    (0.15, 7.75), 16.20, 4.30,
    boxstyle="round,pad=0,rounding_size=0.25",
    linewidth=0.9, edgecolor=GRAY, facecolor="none",
    linestyle=(0, (5, 3)), zorder=1))
tag(8.25, 11.83, "PPO Agent", size=7.6, style="italic", color=GRAY)

# ---- milling simulator ----------------------------------------------------
box(3.60, 5.30, 6.6, 2.0, C_SIM, F_SIM)
tag(3.60, 5.78, "Milling Simulator", size=8.4, weight="bold")
tag(3.60, 5.32, "regenerative chatter dynamics", size=6.8,
    style="italic", color=GRAY)
# optional -- replace the line above with your observation vector, e.g.:
# tag(3.60, 5.32, r"$o_t=[\,n(t),\,x(t),\,\dot{x}(t),\ldots\,]$", size=6.4,
#     style="italic", color=GRAY)

# ---- rollout buffer -------------------------------------------------------
box(3.60, 2.20, 6.6, 1.8, C_BUF, F_BUF)
tag(3.60, 2.62, "Rollout Buffer", size=8.2, weight="bold")
tag(3.60, 2.20, EQ_BUFFER, size=7.4)
tag(3.60, 1.82, "frozen during optimization", size=6.4,
    style="italic", color=GRAY)

# ---- GAE ------------------------------------------------------------------
box(12.90, 2.60, 6.6, 2.9, C_GAE, F_GAE)
tag(12.90, 3.62, "GAE Advantage Estimation", size=8.2, weight="bold")
tag(12.90, 3.05, EQ_TD,  size=7.3)
tag(12.90, 2.50, EQ_GAE, size=7.3)
tag(12.90, 1.95, EQ_TGT, size=7.3)

# ---------------------------------------------------------------------------
# 6. ARROWS
# ---------------------------------------------------------------------------
ARROWS = [
    (( 2.60, 13.70), ( 2.60, 11.60), C_POL),  # (5) update theta
    ((13.50, 13.70), (13.50, 11.60), C_VAL),  # (6) update phi
    (( 5.50,  8.20), ( 5.50,  6.30), C_NEU),  # (2) a_t
    (( 1.60,  6.30), ( 1.60,  8.20), C_NEU),  # (1) o_t
    (( 3.60,  4.30), ( 3.60,  3.10), C_NEU),  # (3) R_t, o_{t+1}
    (( 6.90,  2.20), ( 9.60,  2.20), C_NEU),  # (4) batch
    ((11.60,  8.20), (11.60,  4.05), C_VAL),  # V_phi -> GAE
    ((14.30,  4.05), (14.30,  8.20), C_VAL),  # V_target -> critic
    (( 9.60,  3.90), ( 6.90,  8.50), C_POL),  # advantages -> actor
]
for p0, p1, c in ARROWS:
    arrow(p0, p1, c)

# ---------------------------------------------------------------------------
# 7. LABELS
# ---------------------------------------------------------------------------
# parameter updates + synchronization
tag( 2.85, 12.65, r"(5)  update $\theta$  ($L^{\mathrm{CLIP}}$)",
    ha="left", size=6.9)
tag(13.25, 12.65, r"(6)  update $\phi$  ($L^{V}$)",
    ha="right", size=6.9)
tag( 8.25, 12.65,
    r"$\theta_{\mathrm{old}}\leftarrow\theta$ at start of each iteration",
    style="italic", color=GRAY, size=6.6)

# control loop
tag(5.32, 7.25, r"(2)  $a_t\sim\pi_{\theta}(\cdot\mid o_t)$"
                 + "\n" + "control action", ha="right")
tag(1.42, 7.25, r"(1)  $o_t$" + "\n" + "observation", ha="right")

# simulator -> buffer
tag(3.85, 3.70, r"(3)  " + RTW + r"  (reward),$\;o_{t+1}$", ha="left")

# buffer -> GAE
tag(8.25, 2.50, "(4)  trajectory batch", size=6.6, style="italic",
    bbox_on=True)

# critic <-> GAE
tag(11.28, 6.10, r"$V_{\phi}(o_t),\;V_{\phi}(o_{t+1})$", size=6.6,
    rotation=90)
tag(14.62, 6.10, r"$V_{\mathrm{target}}$", size=6.6, rotation=90)

# GAE -> actor (advantages, along the diagonal)
_d0, _d1 = (9.60, 3.90), (6.90, 8.50)
_ang = math.degrees(math.atan2(_d1[1] - _d0[1], _d1[0] - _d0[0]))
if _ang > 90:
    _ang -= 180
tag((_d0[0] + _d1[0]) / 2, (_d0[1] + _d1[1]) / 2,
    r"$\hat{A}_{t}$  (advantages)", size=6.8, rotation=_ang, bbox_on=True)

# temporal structure notes
tag(3.60, 0.85, r"$N$ transitions per iteration", style="italic",
    color=GRAY, size=6.4)
tag(12.90, 0.85, r"$K$ epochs over frozen buffer", style="italic",
    color=GRAY, size=6.4)

# legend
_handles = [
    Line2D([0], [0], color=C_POL, lw=1.6),
    Line2D([0], [0], color=C_VAL, lw=1.6),
    Line2D([0], [0], color=C_NEU, lw=1.6),
]
ax.legend(_handles,
          ["policy pathway ($\\theta$)", "value pathway ($\\phi$)",
           "agent-environment data"],
          loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=3,
          frameon=False, fontsize=6.2, handlelength=1.5,
          columnspacing=1.3, handletextpad=0.5)

# ---------------------------------------------------------------------------
# 7b. PRE-FLIGHT VALIDATION -- parse every math string through the SAME
#     mathtext engine savefig uses, BEFORE export. Named failures, no trace.
# ---------------------------------------------------------------------------
def validate_math_strings(fig, ax):
    renderer = fig.canvas.get_renderer()
    checked, failures = 0, []
    for t in ax.texts:
        for line in t.get_text().split("\n"):
            if "$" not in line:
                continue
            checked += 1
            try:
                renderer.get_text_width_height_descent(
                    line, t.get_fontproperties(), True)
            except Exception as err:
                failures.append(f"  string : {line}\n  reason : {err}")
    if failures:
        raise SystemExit(
            "\n[FAIL] mathtext pre-flight -- fix these strings:\n\n"
            + "\n\n".join(failures) + "\n")
    print(f"[ok] {checked} math strings pre-validated "
          f"against matplotlib {matplotlib.__version__}")

validate_math_strings(fig, ax)

# ---------------------------------------------------------------------------
# 8. EXPORT
# ---------------------------------------------------------------------------
fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=DPI)
if SAVE_SVG:
    fig.savefig(OUT_SVG)
plt.close(fig)

print(f"[ok] {OUT_PDF}  ({W_CM:.1f} x {H_CM:.1f} cm vector, embedded fonts)")
print(f"[ok] {OUT_PNG}  ({DPI} dpi preview)")
if SAVE_SVG:
    print(f"[ok] {OUT_SVG}")