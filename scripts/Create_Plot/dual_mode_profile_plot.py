import numpy as np
import matplotlib.pyplot as plt

# Set publication-standard rcParams for journal style (IEEE / Elsevier)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Computer Modern Roman"],
    "axes.labelsize": 11,
    "font.size": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "text.usetex": False,
})

# 1. Reproducible Synthetic Profile Generation
np.random.seed(42)
n_points = 600
x = np.linspace(0, 1, n_points)

base_trend = 0.2 + 0.65 * (x ** 1.3)
chatter_profile = (
    base_trend 
    + 0.03 * np.sin(40 * x) 
    + 0.02 * np.cos(90 * x) 
    + 0.015 * np.random.randn(n_points)
)
roughing_profile = np.clip(chatter_profile, 0.05, 0.98)

# 2. Create Figure Layout (Two stacked, perfectly aligned subplots)
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, sharey=True)

# -------------------------------------------------------------------------
# Subplot (a): Chatter-Aware Roughing Control (Mode 1)
# -------------------------------------------------------------------------
ax1 = axes[0]
ax1.axhline(1.0, color='k', linestyle='--', linewidth=1.2)
ax1.plot(x, roughing_profile, color='#0047AB', linewidth=1.5)
ax1.fill_between(x, roughing_profile, 1.0, color='#0047AB', alpha=0.06)

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1.05)
ax1.set_ylabel('Normalized depth ($a_p / a_{max}$)')
ax1.set_title('(a)', loc='left', fontweight='bold', fontsize=12)
ax1.set_yticks([0, 1])
ax1.set_yticklabels(['0', '1'])
ax1.set_xticks([0, 1])

# Annotations for (a)
ax1.annotate('Initial surface', xy=(0.8, 1.0), xytext=(0.75, 0.85),
             arrowprops=dict(arrowstyle="->", lw=1.0, color='k'),
             fontsize=9, fontweight='bold')
ax1.annotate('Roughing profile', xy=(0.65, roughing_profile[int(0.65 * n_points)]), 
             xytext=(0.68, 0.45),
             arrowprops=dict(arrowstyle="->", lw=1.0, color='k'),
             fontsize=9, fontweight='bold')

# -------------------------------------------------------------------------
# Subplot (b): Chatter-Suppressed Finishing Control (Mode 2)
# -------------------------------------------------------------------------
ax2 = axes[1]
ax2.axhline(1.0, color='k', linestyle='--', linewidth=1.2)

# Segmented finishing levels ($a_{p,c}$) - 7 total lines
finishing_levels = np.linspace(0.15, 0.85, 7)
active_level = finishing_levels[5]  # Selecting the 6th line (index 5) for partial flattening

# Modify profile to show Mode 2 material removal: 
# Flatten/cut off the profile smoothly at the active finishing level from right to left
finishing_profile = np.minimum(roughing_profile, active_level)

# Plot remaining profile and removed region
ax2.plot(x, finishing_profile, color='#0047AB', linewidth=1.5)
ax2.fill_between(x, 0.05, finishing_profile, color='#B22222', alpha=0.08)

# Draw finishing lines
for i, level in enumerate(finishing_levels):
    ax2.axhline(level, color='#B22222', linestyle='--', linewidth=0.9)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1.05)
ax2.set_ylabel('Normalized depth ($a_p / a_{max}$)')
ax2.set_xlabel('Normalized length ($x / L$)')
ax2.set_title('(b)', loc='left', fontweight='bold', fontsize=12)
ax2.set_yticks([0, 1])
ax2.set_yticklabels(['0', '1'])
ax2.set_xticks([0, 1])

# Clear and bold annotations for (b)
ax2.annotate('Segmented finishing levels', xy=(0.15, finishing_levels[4]), xytext=(0.12, 0.92),
             arrowprops=dict(arrowstyle="->", lw=1.0, color='k'),
             fontsize=9, fontweight='bold')
ax2.annotate('Target finishing depth\n(Mode 2 flattened pass)', xy=(0.5, active_level), xytext=(0.45, 0.35),
             arrowprops=dict(arrowstyle="->", lw=1.0, color='k'),
             fontsize=9, fontweight='bold')

plt.tight_layout()

# Save as high-resolution vector and raster formats
plt.savefig('dual_mode_control_strategy_modified.pdf', format='pdf', dpi=300)
plt.savefig('dual_mode_control_strategy_modified.png', format='png', dpi=300)
plt.show()