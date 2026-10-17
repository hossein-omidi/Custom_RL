import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

# Set publication-standard rcParams for journal style (IEEE / Elsevier)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Computer Modern Roman"],
    "axes.labelsize": 11,
    "font.size": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "text.usetex": False,
})

# 1. Reproducible Simulation Setup (Cutting from Right to Left: x=1 to x=0)
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

# Tool engagement positions (cutting from x=1.0 moving left towards x=0.0)
tool_x_a = 0.35  # Current roughing cutter position
tool_idx_a = int(tool_x_a * n_points)

tool_x_b = 0.45  # Current finishing cutter position
tool_idx_b = int(tool_x_b * n_points)

# 2. Create Figure Layout
fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, sharey=True)

# -------------------------------------------------------------------------
# Subplot (a): Chatter-Aware Roughing Control (Mode 1)
# -------------------------------------------------------------------------
ax1 = axes[0]
ax1.axhline(1.0, color='k', linestyle='--', linewidth=1.2)

# Uncut raw stock region (from x=0 to tool_x_a)
ax1.fill_between(x[:tool_idx_a], 0, 1.0, color='#D3D3D3', alpha=0.5, label='Uncut Stock')
ax1.plot([tool_x_a, tool_x_a], [roughing_profile[tool_idx_a], 1.0], color='k', linewidth=1.5) # Vertical uncut wall

# Machined roughing profile (from tool_x_a to x=1.0)
ax1.plot(x[tool_idx_a:], roughing_profile[tool_idx_a:], color='#0047AB', linewidth=1.5)
ax1.fill_between(x[tool_idx_a:], roughing_profile[tool_idx_a:], 1.0, color='#0047AB', alpha=0.06)

# Professional Face Milling Cutter Symbol at cutting front
shank_a = Rectangle((tool_x_a - 0.015, 0.95), 0.03, 0.12, facecolor='#4F4F4F', edgecolor='k', linewidth=0.8, zorder=5)
body_a = Rectangle((tool_x_a - 0.025, roughing_profile[tool_idx_a]), 0.05, 0.15, facecolor='#708090', edgecolor='k', linewidth=1, zorder=5)
ax1.add_patch(shank_a)
ax1.add_patch(body_a)
ax1.text(tool_x_a, 1.12, 'Milling Tool ($\leftarrow$ Feed)', fontsize=8, fontweight='bold', ha='center')

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1.2)
ax1.set_ylabel('Normalized depth ($a_p / a_{max}$)')
ax1.set_title('(a) Mode 1: Chatter-Aware Roughing', loc='left', fontweight='bold', fontsize=11)
ax1.set_yticks([0, 1])
ax1.set_yticklabels(['0', '1'])

# Clean, non-overlapping annotations
ax1.annotate('Initial surface', xy=(0.85, 1.0), xytext=(0.65, 1.08),
             arrowprops=dict(arrowstyle="->", lw=0.9, color='k'),
             fontsize=9, fontweight='bold')
ax1.annotate('Roughing profile', xy=(0.70, roughing_profile[int(0.70 * n_points)]), 
             xytext=(0.75, 0.55),
             arrowprops=dict(arrowstyle="->", lw=0.9, color='k'),
             fontsize=9, fontweight='bold')

# -------------------------------------------------------------------------
# Subplot (b): Chatter-Suppressed Finishing Control (Mode 2)
# -------------------------------------------------------------------------
ax2 = axes[1]
ax2.axhline(1.0, color='k', linestyle='--', linewidth=1.2)

# Segmented finishing levels
finishing_levels = np.linspace(0.15, 0.85, 7)
target_depth = finishing_levels[0]
removal_threshold = finishing_levels[4]  # Active finishing line

for level in finishing_levels:
    ax2.axhline(level, color='#B22222', linestyle='--', linewidth=0.9)

# Bulk substrate below target depth
ax2.fill_between(x, 0, target_depth, color='#B0C4DE', alpha=0.4)

# Uncut roughing profile zone (from x=0 to tool_x_b)
ax2.plot(x[:tool_idx_b], roughing_profile[:tool_idx_b], color='#0047AB', linewidth=1.5)

# Finished profile zone (from tool_x_b to x=1.0, flattened at removal threshold)
finished_segment = np.copy(roughing_profile[tool_idx_b:])
finished_segment[finished_segment > removal_threshold] = removal_threshold
ax2.plot(x[tool_idx_b:], finished_segment, color='#0047AB', linewidth=1.5)

# Vertical finishing step starting precisely from the active red line up to roughing profile
ax2.plot([tool_x_b, tool_x_b], [removal_threshold, roughing_profile[tool_idx_b]], color='#B22222', linestyle='-', linewidth=1.5)

ax2.fill_between(x[tool_idx_b:], target_depth, finished_segment, color='#B22222', alpha=0.08)

# Professional Finishing Tool Symbol
shank_b = Rectangle((tool_x_b - 0.015, 0.90), 0.03, 0.12, facecolor='#4F4F4F', edgecolor='k', linewidth=0.8, zorder=5)
body_b = Rectangle((tool_x_b - 0.025, removal_threshold), 0.05, 0.15, facecolor='#708090', edgecolor='k', linewidth=1, zorder=5)
ax2.add_patch(shank_b)
ax2.add_patch(body_b)
ax2.text(tool_x_b, 1.07, 'Finishing Tool ($\leftarrow$ Feed)', fontsize=8, fontweight='bold', ha='center')

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1.15)
ax2.set_ylabel('Normalized depth ($a_p / a_{max}$)')
ax2.set_xlabel('Normalized length ($x / L$)')
ax2.set_title('(b) Mode 2: Chatter-Suppressed Finishing', loc='left', fontweight='bold', fontsize=11)
ax2.set_yticks([0, 1])
ax2.set_yticklabels(['0', '1'])

# Clean, non-overlapping annotations
ax2.annotate('Finishing levels', xy=(0.20, finishing_levels[3]), xytext=(0.08, 0.95),
             arrowprops=dict(arrowstyle="->", lw=0.9, color='k'),
             fontsize=9, fontweight='bold')
ax2.annotate('Target depth', xy=(0.75, target_depth), xytext=(0.72, 0.04),
             arrowprops=dict(arrowstyle="->", lw=0.9, color='k'),
             fontsize=9, fontweight='bold')

plt.tight_layout()

# Save final publication-ready vector files
plt.savefig('dual_mode_control_strategy_perfect.pdf', format='pdf', dpi=300)
plt.savefig('dual_mode_control_strategy_perfect.png', format='png', dpi=300)
plt.show()