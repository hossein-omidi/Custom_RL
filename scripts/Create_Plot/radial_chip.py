"""
figure_chip_thickness_intuitive.py

A simple, intuitive figure showing the face-milling chip-thickness
mechanism from the top view (looking down the tool axis).

Left panel:  the cutter and workpiece as seen from above
Right panel: a zoomed view of the cutting zone showing
             two consecutive tooth passes and the chip between them
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Arc


# ==============================================================
# Figure setup
# ==============================================================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 6))


# ==============================================================
# LEFT PANEL: overview of the operation (top view)
# ==============================================================
# Cutter circle
R = 1.0
theta = np.linspace(0, 2*np.pi, 400)
axL.plot(R*np.cos(theta), R*np.sin(theta), 'k-', linewidth=2.0)

# Cutter center
axL.plot(0, 0, 'ko', markersize=6)

# Teeth shown as small circles on the perimeter
n_teeth = 6
for k in range(n_teeth):
    ang = 2*np.pi * k / n_teeth + np.pi/6
    axL.plot(R*np.cos(ang), R*np.sin(ang), 'ks', markersize=10,
             markerfacecolor='gray', markeredgecolor='k')

# A few of the teeth labeled
axL.text(R*np.cos(np.pi/6)+0.05, R*np.sin(np.pi/6)+0.05,
         "tooth $i$", fontsize=11, color='k')
axL.text(R*np.cos(3*np.pi/2)+0.05, R*np.sin(3*np.pi/2)-0.15,
         "tooth $i-1$", fontsize=11, color='k')

# Workpiece: a shaded rectangle that the cutter sweeps through
wp_x = [-1.6, 1.6, 1.6, -1.6, -1.6]
wp_y = [-1.5, -1.5, 1.5, 1.5, -1.5]
axL.fill(wp_x, wp_y, color='lightgray', alpha=0.5, zorder=0)
axL.text(-1.4, 1.2, "workpiece", fontsize=11, color='0.3')

# Feed direction
axL.annotate("", xy=(1.5, -1.8), xytext=(0.5, -1.8),
             arrowprops=dict(arrowstyle="->", lw=2.0, color='k'))
axL.text(1.0, -2.05, "feed", fontsize=11, ha='center')

# Rotation arrow
arc_rot = Arc((0, 0), 2*R*1.25, 2*R*1.25,
              angle=0, theta1=30, theta2=120,
              color='k', linewidth=1.5)
axL.add_patch(arc_rot)
axL.annotate("", xy=(-R*1.25*np.sin(np.deg2rad(30)),
                     R*1.25*np.cos(np.deg2rad(30))),
             xytext=(-R*1.25*np.sin(np.deg2rad(35)),
                     R*1.25*np.cos(np.deg2rad(35))),
             arrowprops=dict(arrowstyle="->", lw=1.5, color='k'))
axL.text(-R*1.1, R*1.35, r"$\Omega$", fontsize=14)

# Cutting zone indication: small dashed box
axL.plot([0.15, 1.05, 1.05, 0.15, 0.15],
         [-0.5, -0.5, 0.5, 0.5, -0.5],
         'r--', linewidth=1.5)
axL.text(0.55, 0.62, "cutting\nzone", fontsize=10, color='r',
         ha='center', va='bottom')

axL.set_xlim(-2.0, 2.0)
axL.set_ylim(-2.2, 2.0)
axL.set_aspect('equal')
axL.axis('off')
axL.set_title("(a) Top view of face milling", fontsize=13, pad=10)


# ==============================================================
# RIGHT PANEL: zoomed view of the cutting zone
# ==============================================================
# The zoom shows the arc traced by two consecutive teeth.
# Because the plate vibrates, the arc traces are wavy.
# The chip is the material between them.

# Use polar-like coordinates: the cutting edge sweeps an arc
# Parameterize the angular position from theta_s to theta_e

theta_s = np.deg2rad(10)
theta_e = np.deg2rad(130)
theta = np.linspace(theta_s, theta_e, 400)

# Radius of the tool tip
r_tip = 1.0

# Nominal position of the current cut
# Add sinusoidal waviness due to vibration
wavelength = np.deg2rad(55)
amp = 0.05
phase_curr = 0.0
phase_prev = np.pi  # opposite phase to make difference visible

# Current surface (tooth i just cut)
r_curr = r_tip - amp * np.sin((theta - theta_s)/wavelength * 2*np.pi + phase_curr)

# Previous surface (tooth i-1 cut one tooth period earlier)
r_prev = r_tip - amp * np.sin((theta - theta_s)/wavelength * 2*np.pi + phase_prev)

# Compute Cartesian coordinates
x_curr = r_curr * np.cos(theta)
y_curr = r_curr * np.sin(theta)
x_prev = r_prev * np.cos(theta)
y_prev = r_prev * np.sin(theta)

# ---- Chip: shaded region between the two wavy arcs ----
poly_x = np.concatenate([x_curr, x_prev[::-1], [x_curr[0]]])
poly_y = np.concatenate([y_curr, y_prev[::-1], [y_curr[0]]])
axR.fill(poly_x, poly_y, color='#f0a860', alpha=0.55,
         edgecolor='none', zorder=1)

# ---- Two wavy arcs ----
axR.plot(x_curr, y_curr, 'k-', linewidth=2.2, zorder=3,
         label=r"surface left by tooth $i$ (current)")
axR.plot(x_prev, y_prev, 'b--', linewidth=2.0, zorder=3,
         label=r"surface left by tooth $i-1$ (previous)")

# ---- Radial lines at the two ends ----
axR.plot([0, r_tip*np.cos(theta_s)*1.02],
         [0, r_tip*np.sin(theta_s)*1.02], 'k-', linewidth=0.6, zorder=0)
axR.plot([0, r_tip*np.cos(theta_e)*1.02],
         [0, r_tip*np.sin(theta_e)*1.02], 'k-', linewidth=0.6, zorder=0)

# ---- Chip thickness arrow ----
i_ann = 180  # index for annotation
axR.annotate("", xy=(x_curr[i_ann], y_curr[i_ann]),
             xytext=(x_prev[i_ann], y_prev[i_ann]),
             arrowprops=dict(arrowstyle="<->", lw=1.8, color='k'),
             zorder=4)
mid_x = 0.5*(x_curr[i_ann] + x_prev[i_ann])
mid_y = 0.5*(y_curr[i_ann] + y_prev[i_ann])
axR.text(mid_x + 0.06, mid_y - 0.02, r"$h_i$", fontsize=16,
         color='k', zorder=5)

# ---- Radius direction arrow (label "r") ----
axR.annotate("", xy=(1.15, 0.0), xytext=(0.0, 0.0),
             arrowprops=dict(arrowstyle="->", lw=0.8, color='0.4',
                             linestyle=':'))
axR.text(1.02, -0.07, "r", fontsize=12, color='0.4')

# ---- Entry/exit angles ----
axR.text(r_tip*np.cos(theta_s) + 0.02,
         r_tip*np.sin(theta_s) - 0.15,
         r"$\theta_s$", fontsize=14, color='k')
axR.text(r_tip*np.cos(theta_e) + 0.02,
         r_tip*np.sin(theta_e) + 0.05,
         r"$\theta_e$", fontsize=14, color='k')

# ---- Cutter center ----
axR.plot(0, 0, 'ko', markersize=6, zorder=5)

# ---- Rotation arrow ----
arc_rot2 = Arc((0, 0), 0.55, 0.55, angle=0,
               theta1=20, theta2=110, color='k', linewidth=1.2)
axR.add_patch(arc_rot2)
axR.annotate("", xy=(-0.05, 0.29), xytext=(0.02, 0.27),
             arrowprops=dict(arrowstyle="->", lw=1.2, color='k'))
axR.text(0.28, 0.30, r"$\Omega$", fontsize=14)

# ---- Zoom-in box on the chip thickness ----
axR.text(0.85, 0.75,
         "chip",
         fontsize=12, color='#a05010', ha='center',
         fontweight='bold')

# ---- Legend ----
axR.legend(loc='upper left', fontsize=10, framealpha=0.9)

# ---- Labels for the two passes ----
axR.text(0.05, -0.35, "tooth $i$ cuts here\nnow",
         fontsize=10, color='k', ha='left')
axR.text(0.05, -0.55, "tooth $i-1$ cut here\none period earlier",
         fontsize=10, color='b', ha='left')

axR.set_xlim(-0.3, 1.3)
axR.set_ylim(-0.7, 1.2)
axR.set_aspect('equal')
axR.axis('off')
axR.set_title("(b) Zoomed view of the cutting zone",
              fontsize=13, pad=10)


# ==============================================================
# Save and show
# ==============================================================
plt.tight_layout()
plt.savefig("fig_chip_thickness_intuitive.pdf", bbox_inches='tight')
plt.savefig("fig_chip_thickness_intuitive.png", dpi=220,
            bbox_inches='tight')
plt.show()