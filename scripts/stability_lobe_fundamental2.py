#!/usr/bin/env python3
"""
Stability Lobe Diagram for Face Milling on Flexible Cantilever Plate
====================================================================

Numerical time-domain discretization approach treating the milling dynamics
as a "black box" defined by modal parameters (as per your requirement).

Methods:
1. Semi-Discretization Method (SDM) - Insperger & Stepan (2004)
   - First-order with linear interpolation of delayed terms
   - Floquet monodromy eigenvalue analysis
   - Spectral radius ρ(Φ) < 1 → stable
   
2. Zeroth-Order Analytical (ZOA/SFM) - Altintas & Budak (1995)
   - Single-frequency approximation for validation

Outputs:
- Heatmap: spectral radius ρ(rpm, ap) with stability boundary contour
- 2D stability lobe: ap_lim vs rpm
- 3D stability surface: ap_lim vs (rpm, cutter_x)

References:
- Qin et al., Int. J. Adv. Manuf. Technol. 136 (2025) 2945-2985
- Insperger & Stepan, J. Comput. Nonlinear Dyn. 1 (2006) 71-87
- Altintas, Manufacturing Automation, 2nd Ed. (2012)
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple, Optional, List
import sys
import warnings

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Try to import scipy for fast matrix exponential
try:
    from scipy.linalg import expm as scipy_expm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not found, using numpy-only matrix exponential (slower)")

# Make package importable
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from custom_rl.plants.plate import PlatePlant, omega_to_rpm, rpm_to_omega
from custom_rl.plants.compute_mode_shapes_updated import build_mode_shape_matrix


# =============================================================================
# Matrix Exponential (numpy fallback if no scipy)
# =============================================================================
def matrix_exp(A: np.ndarray) -> np.ndarray:
    """Compute matrix exponential with scipy if available, else scaling-and-squaring."""
    if HAS_SCIPY:
        return np.asarray(scipy_expm(A), dtype=np.complex128)
    
    A = np.asarray(A, dtype=np.complex128)
    # Scaling and squaring with Padé approximation
    n = A.shape[0]
    norm = np.linalg.norm(A, ord=1)
    
    # Choose scaling factor
    s = max(0, int(np.ceil(np.log2(norm / 5.371920351148152)) + 1))
    As = A / (2.0 ** s)
    
    # Padé [6/6] approximation for exp(As)
    I = np.eye(n, dtype=np.complex128)
    As2 = As @ As
    As4 = As2 @ As2
    As6 = As2 @ As4
    
    U = As @ (As6 + 42*As4 + 840*As2 + 5040*I)
    V = As6 + 30*As4 + 420*As2 + 3024*I
    
    # exp(As) ≈ (V - U)^{-1} (V + U)
    E = np.linalg.solve(V - U, V + U)
    
    # Squaring
    for _ in range(s):
        E = E @ E
    
    return E


# =============================================================================
# Dynamic Model for Face Milling on Flexible Cantilever
# =============================================================================
@dataclass
class FaceMillingStabilityModel:
    """
    Linearised regenerative face-milling model extracted from PlatePlant.
    
    The dynamic model (per mode k):
        η̈_k + 2ζ_kω_k η̇_k + ω_k² η_k = Q_k
    
    Face-milling z-force regeneration:
        h_z = -Δz · sin(γ_L)     (regenerative perturbation)
        F_z = K_a · a_p · h_z
        Δz = -1000 · φ_z^T (η(t) - η(t-τ))  [mm conversion]
        Q_k = φ_k · F_z / M_k
    
    Linearised regenerative coupling:
        Q(t) = κ · a_p · g(t) · R · (η(t) - η(t-τ))
    where:
        κ = 1000 · sin(γ_L) · K_a
        g(t) = number of engaged teeth
        R_{kj} = φ_k · φ_j / M_k  (rank-1 coupling matrix)
        τ = 2π/(N·ω) = tooth period = regenerative delay
    """
    plant: Any
    omega_vec: np.ndarray = field(init=False)
    zeta_vec: np.ndarray = field(init=False)
    M_modal: np.ndarray = field(init=False)
    K: int = field(init=False)
    N: int = field(init=False)
    Ka: float = field(init=False)
    gamma_L: float = field(init=False)
    theta0: float = field(init=False)
    D_mm: float = field(init=False)
    ae_mm: float = field(init=False)
    milling_mode: str = field(init=False)
    disp_to_mm: float = 1000.0

    def __post_init__(self) -> None:
        p = self.plant
        self.omega_vec = np.asarray(p.omega_vec, dtype=np.float64).ravel()
        self.zeta_vec = np.asarray(p.zeta_vec, dtype=np.float64).ravel()
        self.M_modal = np.asarray(p.M_modal, dtype=np.float64).ravel()
        self.K = int(p.K)
        self.N = int(p.N)
        self.Ka = float(p.Ka)
        self.gamma_L = float(p.gamma_L)
        self.theta0 = float(getattr(p, "theta0", 0.0))
        self.D_mm = float(p.D_mm)
        self.ae_mm = float(p.ae_default)
        self.milling_mode = str(p.milling_mode)

    def phi_at(self, xc: float, yc: float) -> np.ndarray:
        """Cutter-point mode shape vector φ_z (K,)."""
        return build_mode_shape_matrix(
            self.plant.W_mn, [(float(xc), float(yc))], 
            self.plant.m_max, self.plant.n_max
        ).ravel()

    def entry_exit_angles(self) -> Tuple[float, float]:
        """Face-milling entry/exit angles [rad]."""
        immersion = np.clip(self.ae_mm / max(self.D_mm, 1e-12), 0.0, 1.0)
        mode = self.milling_mode.lower()
        if mode in {"down", "climb", "climb_milling"}:
            return float(np.arccos(np.clip(2.0 * immersion - 1.0, -1.0, 1.0))), float(np.pi)
        return 0.0, float(np.arccos(np.clip(1.0 - 2.0 * immersion, -1.0, 1.0)))

    def engaged_teeth_at_angle(self, psi: float) -> int:
        """Number of teeth engaged at spindle angle psi."""
        theta_s, theta_e = self.entry_exit_angles()
        pitch = 2.0 * np.pi / self.N
        count = 0
        for i in range(self.N):
            th = (psi + i * pitch) % (2.0 * np.pi)
            if theta_s < th < theta_e:
                count += 1
        return count

    def engaged_count(self, t: float, omega: float) -> int:
        """Number of teeth engaged at time t."""
        psi = omega * t + self.theta0
        return self.engaged_teeth_at_angle(psi)

    def mean_engaged(self) -> float:
        """Average number of engaged teeth over one revolution."""
        theta_s, theta_e = self.entry_exit_angles()
        return self.N * (theta_e - theta_s) / (2.0 * np.pi)

    def oriented_frf(self, omega: float, phi: np.ndarray) -> complex:
        """
        Cutter-point receptance Φ_zz(iω) = Σ φ_k² / (M_k · (ω_k² - ω² + 2iζ_kω_kω))
        
        This is the key FRF that determines chatter stability.
        """
        wn = self.omega_vec
        denom = (wn**2 - omega**2) + 2j * self.zeta_vec * wn * omega
        return complex(np.sum((phi**2 / self.M_modal) / denom))

    def kappa_per_ap(self) -> float:
        """Averaged regenerative gain per unit a_p [N/m per mm]."""
        return self.mean_engaged() * self.Ka * self.disp_to_mm * np.sin(self.gamma_L)

    def dominant_mode_indices(self, phi: np.ndarray, n_modes: int) -> np.ndarray:
        """
        Indices of n_modes most chatter-relevant modes at the cutter.
        
        Ranking by static compliance φ_k²/(M_k·ω_k²) - the peak height
        of each mode in the cutter-point FRF.
        """
        compliance = phi**2 / (self.M_modal * self.omega_vec**2)
        order = np.argsort(compliance)[::-1]
        return np.sort(order[:int(np.clip(n_modes, 1, self.K))])


# =============================================================================
# 1. SEMI-DISCRETIZATION METHOD (SDM) - Numerical Floquet Analysis
# =============================================================================
def sdm_spectral_radius(
    model: FaceMillingStabilityModel,
    rpm: float,
    ap_mm: float,
    phi: np.ndarray,
    mode_idx: np.ndarray,
    k_intervals: int = 40
) -> float:
    """
    Compute spectral radius using First-Order Semi-Discretization Method.
    
    The DDE is discretized into k_intervals over the tooth period τ.
    The monodromy matrix Φ is constructed using the matrices multiplication
    scheme (MMS) as described in Insperger & Stepan (2004).
    
    Returns:
        float: Maximum absolute eigenvalue of the monodromy matrix (ρ < 1 → stable)
    """
    omega = float(rpm_to_omega(rpm))
    tau = 2.0 * np.pi / (model.N * max(omega, 1e-12))
    dt = tau / k_intervals
    k = k_intervals
    
    # Extract modal parameters for selected modes
    n = len(mode_idx)
    wn = model.omega_vec[mode_idx]
    zeta = model.zeta_vec[mode_idx]
    Mk = model.M_modal[mode_idx]
    ph = phi[mode_idx]
    
    # Pre-compute structural matrices (constant)
    K_diag = wn**2
    C_diag = 2.0 * zeta * wn
    I_n = np.eye(n)
    Z_n = np.zeros((n, n))
    
    # Coupling matrix R_{kj} = φ_k · φ_j / M_k
    R = np.outer(ph / Mk, ph)
    
    # Coefficient scaling
    kappa = model.disp_to_mm * np.sin(model.gamma_L) * model.Ka * ap_mm
    
    # State space dimension (position + velocity for each mode)
    two_n = 2 * n
    
    # Pre-compute engagement for all intervals (avoid repeated calculations)
    g_vals = np.array([model.engaged_count((i + 0.5) * dt, omega) for i in range(k)])
    c_vals = kappa * g_vals
    
    # Build monodromy matrix using full augmented state
    # State: z_i = [x_i; x_{i-1}; ...; x_{i-k}] where x = [η; η̇]
    # Dimension: d = two_n * (k + 1)
    d = two_n * (k + 1)
    
    # Initialize monodromy as identity
    Phi = np.eye(d, dtype=np.complex128)
    
    I2n = np.eye(two_n, dtype=np.complex128)
    
    for i in range(k):
        c_i = c_vals[i]
        
        # Effective state matrix A_eff = A + B_current
        # A = [[0, I], [-ω²I, -2ζωI]]  (structural)
        # B_current = [[0, 0], [+cR, 0]]  (current regenerative)
        A_eff = np.zeros((two_n, two_n), dtype=np.complex128)
        A_eff[:n, n:] = I_n
        A_eff[n:, :n] = -np.diag(K_diag) + c_i * R  # Note: +cR (stiffness modification)
        A_eff[n:, n:] = -np.diag(C_diag)
        
        # Delayed state matrix B_delay
        # B_delay = [[0, 0], [-cR, 0]]
        B_delay = np.zeros((two_n, two_n), dtype=np.complex128)
        B_delay[n:, :n] = -c_i * R
        
        # Matrix exponential P_i = exp(A_eff · dt)
        P_i = matrix_exp(A_eff * dt)
        
        # Particular solution matrix for delayed term
        # R_i = A_eff^{-1} (P_i - I) B_delay
        try:
            Ri = np.linalg.solve(A_eff, P_i - I2n) @ B_delay
        except np.linalg.LinAlgError:
            Ri = dt * B_delay  # Fallback for singular A_eff
        
        half_Ri = 0.5 * Ri
        
        # Build interval transition matrix D_i
        # D_i maps z_i to z_{i+1}:
        #   x_{i+1} = P_i x_i + half_Ri x_{i-k} + half_Ri x_{i-k+1}
        #   x_j -> x_{j-1} for j = i, i-1, ..., i-k+1 (shift)
        D_i = np.zeros((d, d), dtype=np.complex128)
        
        # First block row: new state
        D_i[:two_n, :two_n] = P_i
        D_i[:two_n, (k-1)*two_n : k*two_n] += half_Ri   # x_{i-k}
        D_i[:two_n, k*two_n : (k+1)*two_n] += half_Ri   # x_{i-k+1}
        
        # Shift rows: history shifts down
        D_i[two_n:, :k*two_n] = np.eye(two_n * k, dtype=np.complex128)
        
        # Multiply monodromy
        Phi = D_i @ Phi
    
    # Compute spectral radius (max absolute eigenvalue)
    eigs = np.linalg.eigvals(Phi)
    return float(np.max(np.abs(eigs)))


def sdm_critical_ap(
    model: FaceMillingStabilityModel,
    rpm: float,
    xc: float,
    yc: float,
    n_modes: int = 1,
    k_intervals: int = 40,
    ap_cap_mm: float = 30.0,
    tol_mm: float = 1e-3,
    max_iter: int = 50
) -> float:
    """
    Find critical axial depth a_p_lim(rpm) where spectral radius crosses 1.
    
    Uses bisection search for robust convergence.
    """
    phi = model.phi_at(xc, yc)
    mode_idx = model.dominant_mode_indices(phi, n_modes)
    
    def rho(ap: float) -> float:
        return sdm_spectral_radius(model, rpm, ap, phi, mode_idx, k_intervals)
    
    # Check stability at zero depth (should be stable)
    if rho(0.0) > 1.0:
        warnings.warn(f"System unstable at a_p=0 for rpm={rpm}, returning 0")
        return 0.0
    
    # Find upper bound where system becomes unstable
    ap_lo = 0.0
    ap_hi = None
    ap = 0.01 * ap_cap_mm
    
    while ap <= ap_cap_mm:
        if rho(ap) > 1.0:
            ap_hi = ap
            break
        ap_lo = ap
        ap *= 1.5
    
    if ap_hi is None:
        return ap_cap_mm  # Stable up to cap
    
    # Bisection
    for _ in range(max_iter):
        if ap_hi - ap_lo <= tol_mm:
            break
        ap_mid = 0.5 * (ap_lo + ap_hi)
        if rho(ap_mid) > 1.0:
            ap_hi = ap_mid
        else:
            ap_lo = ap_mid
    
    return 0.5 * (ap_lo + ap_hi)


def compute_stability_heatmap(
    model: FaceMillingStabilityModel,
    xc: float,
    yc: float,
    rpm_grid: np.ndarray,
    ap_grid: np.ndarray,
    n_modes: int = 1,
    k_intervals: int = 40,
    verbose: bool = True
) -> np.ndarray:
    """
    Compute spectral radius on a 2D grid for heatmap visualization.
    
    Returns:
        rho_map: array of shape (len(ap_grid), len(rpm_grid))
    """
    phi = model.phi_at(xc, yc)
    mode_idx = model.dominant_mode_indices(phi, n_modes)
    
    n_rpm = len(rpm_grid)
    n_ap = len(ap_grid)
    rho_map = np.zeros((n_ap, n_rpm))
    
    for j, rpm in enumerate(rpm_grid):
        if verbose:
            print(f"  Heatmap: rpm {j+1}/{n_rpm} = {rpm:.0f}")
        for i, ap in enumerate(ap_grid):
            rho_map[i, j] = sdm_spectral_radius(
                model, rpm, ap, phi, mode_idx, k_intervals
            )
    
    return rho_map


# =============================================================================
# 2. ZERO-ORDER ANALYTICAL (ZOA/SFM) - Altintas & Budak
# =============================================================================
def zoa_lobe(
    model: FaceMillingStabilityModel,
    xc: float,
    yc: float,
    n_lobes: int = 15,
    freq_pts: int = 5000,
    ap_cap_mm: float = 50.0
) -> dict[str, np.ndarray]:
    """
    Analytical stability lobes using Single-Frequency Method (ZOA).
    
    Characteristic equation:
        -1/(κ·a_p·ḡ) = G(iω_c) · (1 - e^{-iω_c τ})
    
    Solution:
        a_p_lim = -1 / (2·κ·ḡ·Re[G(iω_c)])
        ω_c·τ = 2·atan2(-G_R, G_I) + 2πk
    
    where G = G_R + i·G_I is the oriented FRF at chatter frequency ω_c.
    """
    phi = model.phi_at(xc, yc)
    kappa_g = model.kappa_per_ap()  # κ·ḡ
    
    wn = model.omega_vec
    lobe_rpm: List[float] = []
    lobe_ap: List[float] = []
    
    # Scan frequency bands around each natural frequency
    for w0 in wn:
        # Wider frequency range to capture all lobes
        w_lo = 0.3 * w0
        w_hi = 2.5 * w0
        
        for wc in np.linspace(w_lo, w_hi, freq_pts):
            if wc <= 0:
                continue
            
            G = model.oriented_frf(wc, phi)
            G_R, G_I = G.real, G.imag
            
            # Critical depth requires G_R < 0
            if G_R >= -1e-15:
                continue
            
            a_lim = -1.0 / (2.0 * kappa_g * G_R)
            
            if not np.isfinite(a_lim) or a_lim <= 0 or a_lim > ap_cap_mm:
                continue
            
            # Phase condition: ψ/2 = atan2(-G_R, G_I)
            # ψ = 2·atan2(-G_R, G_I) + 2πk
            psi_half = np.arctan2(-G_R, G_I)
            
            for k in range(n_lobes):
                psi = 2.0 * psi_half + 2.0 * np.pi * k
                
                if psi <= 0:
                    continue
                
                # Verify G_I·sin(ψ) < 0 for consistency
                if G_I * np.sin(psi) >= 0:
                    continue
                
                tau = psi / wc
                rpm = 60.0 / (model.N * tau)
                
                if rpm > 0 and rpm < 100000:  # Sanity check
                    lobe_rpm.append(rpm)
                    lobe_ap.append(a_lim)
    
    return {"rpm": np.asarray(lobe_rpm), "ap": np.asarray(lobe_ap)}


def zoa_envelope(
    lobe: dict[str, np.ndarray],
    rpm_grid: np.ndarray
) -> np.ndarray:
    """Compute lower envelope of ZOA scallops on regular rpm grid."""
    rpm = lobe["rpm"]
    ap = lobe["ap"]
    env = np.full(rpm_grid.shape, np.nan)
    
    if rpm.size == 0:
        return env
    
    # Bin scallop points and keep minimum ap (most conservative)
    for i, r in enumerate(rpm_grid):
        mask = np.abs(rpm - r) < (rpm_grid[1] - rpm_grid[0]) * 0.6
        if np.any(mask):
            env[i] = float(np.min(ap[mask]))
    
    return env


# =============================================================================
# 3D Stability Surface Computation
# =============================================================================
def compute_3d_surface(
    model: FaceMillingStabilityModel,
    rpm_grid: np.ndarray,
    xc_grid: np.ndarray,
    yc: float,
    n_modes: int = 1,
    k_intervals: int = 40,
    ap_cap_mm: float = 30.0,
    verbose: bool = True
) -> np.ndarray:
    """Compute 3D stability surface a_p_lim(rpm, x_cutter)."""
    ap = np.full((len(xc_grid), len(rpm_grid)), np.nan)
    
    for xi, xc in enumerate(xc_grid):
        if verbose:
            print(f"  3D surface: position {xi+1}/{len(xc_grid)}, xc={xc:.3f}m")
        for ri, rpm in enumerate(rpm_grid):
            ap[xi, ri] = sdm_critical_ap(
                model, rpm, xc, yc,
                n_modes=n_modes, k_intervals=k_intervals,
                ap_cap_mm=ap_cap_mm
            )
    
    return ap


# =============================================================================
# Plotting Functions
# =============================================================================
def plot_stability_heatmap(
    rho_map: np.ndarray,
    rpm_grid: np.ndarray,
    ap_grid: np.ndarray,
    sdm_ap: np.ndarray,
    zoa_data: dict,
    xc: float,
    yc: float,
    out: Path
) -> None:
    """
    Plot stability heatmap with contour lines showing spectral radius.
    
    - Background: colored by log10(ρ)
    - Contour line: ρ = 1 (stability boundary)
    - Overlaid: SDM boundary (points) and ZOA scallops
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    R, A = np.meshgrid(rpm_grid, ap_grid)
    
    # Left: Heatmap with log scale
    ax1 = axes[0]
    rho_clipped = np.clip(rho_map, 1e-3, 100)
    
    im = ax1.pcolormesh(
        R, A, rho_clipped,
        norm=LogNorm(vmin=0.1, vmax=10),
        cmap='RdYlGn_r',  # Red=unstable, Green=stable
        shading='auto'
    )
    
    # Contour at ρ = 1
    contour = ax1.contour(
        R, A, rho_map, 
        levels=[1.0], 
        colors='black', 
        linewidths=2.0
    )
    ax1.clabel(contour, fmt='ρ=1', fontsize=10)
    
    # Additional contours
    ax1.contour(
        R, A, rho_map,
        levels=[0.5, 0.8, 1.2, 2.0, 5.0],
        colors='gray', linewidths=0.5, linestyles='--', alpha=0.7
    )
    
    # SDM boundary points
    valid = np.isfinite(sdm_ap)
    ax1.plot(rpm_grid[valid], sdm_ap[valid], 'b-o', markersize=3, linewidth=1.5,
             label='SDM boundary', zorder=5)
    
    # ZOA scallops
    if zoa_data["rpm"].size > 0:
        mask = (zoa_data["rpm"] >= rpm_grid[0]) & (zoa_data["rpm"] <= rpm_grid[-1])
        ax1.scatter(zoa_data["rpm"][mask], zoa_data["ap"][mask], 
                   s=4, c='blue', alpha=0.3, label='ZOA scallops')
    
    ax1.set_xlabel("Spindle Speed [rpm]", fontsize=11)
    ax1.set_ylabel("Axial Depth of Cut a_p [mm]", fontsize=11)
    ax1.set_title(f"Stability Heatmap (log₁₀ ρ)  |  xc={xc:.2f}m, yc={yc:.2f}m", fontsize=12)
    ax1.set_ylim(0, float(np.nanmax(ap_grid)))
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    cbar = fig.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label("Spectral Radius ρ", fontsize=10)
    
    # Right: Clean stability lobe with filled regions
    ax2 = axes[1]
    
    # Fill stable region (below SDM boundary)
    ax2.fill_between(rpm_grid[valid], 0, sdm_ap[valid], 
                     alpha=0.3, color='green', label='Stable region')
    
    # SDM boundary
    ax2.plot(rpm_grid[valid], sdm_ap[valid], 'g-', linewidth=2.5, 
             label='SDM (Floquet)')
    
    # ZOA envelope
    zoa_env = zoa_envelope(zoa_data, rpm_grid)
    valid_zoa = np.isfinite(zoa_env)
    if np.any(valid_zoa):
        ax2.plot(rpm_grid[valid_zoa], zoa_env[valid_zoa], 'r--', linewidth=1.5,
                label='ZOA envelope')
    
    ax2.set_xlabel("Spindle Speed [rpm]", fontsize=11)
    ax2.set_ylabel("Critical Depth a_p_lim [mm]", fontsize=11)
    ax2.set_title("Stability Lobe Diagram", fontsize=12)
    ax2.set_ylim(0, float(max(np.nanmax(sdm_ap[valid]) * 1.3, 1)))
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Add text annotations
    ax2.text(0.02, 0.95, "STABLE", transform=ax2.transAxes, 
             fontsize=14, fontweight='bold', color='green', alpha=0.7,
             verticalalignment='top')
    ax2.text(0.02, 0.05, "UNSTABLE", transform=ax2.transAxes,
             fontsize=14, fontweight='bold', color='red', alpha=0.7)
    
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "stability_lobe_heatmap.png", dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out / 'stability_lobe_heatmap.png'}")


def plot_3d_surface(
    rpm_grid: np.ndarray,
    xc_grid: np.ndarray,
    ap: np.ndarray,
    yc: float,
    out: Path
) -> None:
    """Plot 3D stability surface."""
    R, X = np.meshgrid(rpm_grid, xc_grid)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Replace NaN with 0 for plotting
    ap_plot = np.nan_to_num(ap, nan=0.0)
    
    surf = ax.plot_surface(
        R, X, ap_plot,
        cmap='viridis', alpha=0.9,
        edgecolor='k', linewidth=0.1
    )
    
    ax.set_xlabel("Spindle Speed [rpm]", fontsize=11, labelpad=10)
    ax.set_ylabel("Cutter x-position [m]\n(0=clamped, L=free)", fontsize=11, labelpad=10)
    ax.set_zlabel("a_p_lim [mm]", fontsize=11, labelpad=10)
    ax.set_title(f"3D Stability Surface (yc={yc:.2f}m)", fontsize=13)
    
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="a_p_lim [mm]")
    
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "stability_lobe_3d.png", dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out / 'stability_lobe_3d.png'}")


# =============================================================================
# I/O Helpers
# =============================================================================
def save_csv_2d(rpm: np.ndarray, ap_sdm: np.ndarray, ap_zoa: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['rpm', 'ap_lim_sdm_mm', 'ap_lim_zoa_env_mm'])
        for i in range(len(rpm)):
            writer.writerow([f"{rpm[i]:.4g}", f"{ap_sdm[i]:.6g}", f"{ap_zoa[i]:.6g}"])


def save_csv_heatmap(rho_map: np.ndarray, rpm_grid: np.ndarray, ap_grid: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ap_mm'] + [f"{r:.2f}" for r in rpm_grid])
        for i, ap in enumerate(ap_grid):
            writer.writerow([f"{ap:.4f}"] + [f"{rho_map[i,j]:.6g}" for j in range(len(rpm_grid))])


def save_csv_3d(xc_grid: np.ndarray, rpm_grid: np.ndarray, ap: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['xc_m', 'rpm', 'ap_lim_mm'])
        for xi, xc in enumerate(xc_grid):
            for ri, rpm in enumerate(rpm_grid):
                writer.writerow([f"{xc:.6g}", f"{rpm:.4g}", f"{ap[xi, ri]:.6g}"])


def build_model(args) -> FaceMillingStabilityModel:
    """Instantiate the stability model from plant parameters."""
    kw = dict(
        ae_default=args.ae,
        feed_per_tooth_mm=args.feed,
        milling_mode=args.milling_mode,
        randomize_y0=False,
        modal_damping_ratio=args.zeta
    )
    if args.E is not None:
        kw["E"] = float(args.E)
    
    return FaceMillingStabilityModel(plant=PlatePlant(**kw))


# =============================================================================
# Main Entry Point
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Output options
    parser.add_argument("--out-dir", type=Path, default=Path("plots/lobe_fundamental"))
    
    # Spindle speed range
    parser.add_argument("--rpm-min", type=float, default=1000.0)
    parser.add_argument("--rpm-max", type=float, default=50000.0)
    parser.add_argument("--rpm-points", type=int, default=100, help="RPM grid points for 2D lobe")
    parser.add_argument("--rpm-points-3d", type=int, default=40)
    
    # Cutter position
    parser.add_argument("--xc", type=float, default=0.99, help="Cutter x-position [m] (0=clamped, L=free)")
    parser.add_argument("--yc", type=float, default=0.20, help="Milling line y [m]")
    parser.add_argument("--xc-min", type=float, default=0.30)
    parser.add_argument("--xc-max", type=float, default=0.98)
    parser.add_argument("--xc-points", type=int, default=15)
    
    # SDM parameters
    parser.add_argument("--n-modes", type=int, default=2, help="Modes for SDM (dominant modes)")
    parser.add_argument("--k-intervals", type=int, default=40, help="SDM intervals per tooth period")
    
    # ZOA parameters  
    parser.add_argument("--n-lobes", type=int, default=20, help="Number of ZOA lobe orders")
    
    # Depth of cut range
    parser.add_argument("--ap-cap", type=float, default=30.0, help="Max axial depth [mm]")
    parser.add_argument("--ap-min", type=float, default=0.01, help="Min axial depth for heatmap [mm]")
    parser.add_argument("--ap-points-heatmap", type=int, default=60, help="AP grid points for heatmap")
    
    # Process/model parameters (defaults match registered env)
    parser.add_argument("--ae", type=float, default=28.0, help="Radial immersion [mm]")
    parser.add_argument("--feed", type=float, default=0.20, help="Feed per tooth [mm]")
    parser.add_argument("--zeta", type=float, default=0.02, help="Modal damping ratio")
    parser.add_argument("--milling-mode", default="up", choices=["up", "down"])
    parser.add_argument("--E", type=float, default=None, help="Override Young's modulus [Pa]")
    
    # Visualization modes
    parser.add_argument("--heatmap", action="store_true", help="Generate stability heatmap")
    parser.add_argument("--surface-3d", action="store_true", help="Generate 3D stability surface")
    parser.add_argument("--no-zoa", action="store_true", help="Skip ZOA computation")
    
    args = parser.parse_args()
    
    out = Path(args.out_dir)
    model = build_model(args)
    
    # Print model info
    print("\n" + "="*60)
    print("STABILITY LOBE DIAGRAM - Face Milling on Flexible Cantilever")
    print("="*60)
    print(f"Natural frequencies [Hz]: {np.round(model.omega_vec/(2*np.pi), 1)}")
    print(f"Damping ratios: {model.zeta_vec}")
    print(f"Modal masses [kg]: {np.round(model.M_modal, 4)}")
    print(f"Teeth: {model.N}, Ka: {model.Ka} N/mm², γ_L: {np.degrees(model.gamma_L):.1f}°")
    print(f"Immersion: ae={model.ae_mm}mm, D={model.D_mm}mm, mean engaged={model.mean_engaged():.2f}")
    print(f"SDM: {args.n_modes} modes, {args.k_intervals} intervals/tooth period")
    print("="*60 + "\n")
    
    # Save metadata
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "SDM_Floquet + ZOA_analytical",
        "natural_freq_hz": list(np.round(model.omega_vec / (2 * np.pi), 4)),
        "zeta": list(np.round(model.zeta_vec, 5)),
        "modal_mass_kg": list(np.round(model.M_modal, 4)),
        "N_teeth": model.N,
        "Ka_N_mm2": model.Ka,
        "gamma_L_deg": float(np.degrees(model.gamma_L)),
        "ae_mm": model.ae_mm,
        "D_mm": model.D_mm,
        "mean_engaged_teeth": float(model.mean_engaged()),
        "n_modes_sdm": args.n_modes,
        "k_intervals": args.k_intervals,
    }
    
    # =====================================================================
    # 3D Surface Mode
    # =====================================================================
    if args.surface_3d:
        print("[3D Surface] Computing position-dependent stability...")
        rpm_grid = np.linspace(args.rpm_min, args.rpm_max, args.rpm_points_3d)
        xc_grid = np.linspace(args.xc_min, args.xc_max, args.xc_points)
        
        ap = compute_3d_surface(
            model, rpm_grid, xc_grid, args.yc,
            n_modes=args.n_modes, k_intervals=args.k_intervals,
            ap_cap_mm=args.ap_cap
        )
        
        plot_3d_surface(rpm_grid, xc_grid, ap, args.yc, out)
        save_csv_3d(xc_grid, rpm_grid, ap, out / "stability_lobe_3d.csv")
        
        meta["xc_range"] = [args.xc_min, args.xc_max]
        meta["rpm_range"] = [args.rpm_min, args.rpm_max]
        (out / "stability_lobe_3d_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"\nSaved 3D surface to {out}")
        return
    
    # =====================================================================
    # 2D Lobe + Heatmap Mode (default)
    # =====================================================================
    rpm_grid = np.linspace(args.rpm_min, args.rpm_max, args.rpm_points)
    phi = model.phi_at(args.xc, args.yc)
    mode_idx = model.dominant_mode_indices(phi, args.n_modes)
    
    print(f"[2D Lobe] Cutter position: xc={args.xc}m, yc={args.yc}m")
    print(f"  Dominant modes: {mode_idx}")
    print(f"  Mode shapes at cutter: {np.round(phi[mode_idx], 4)}")
    print()
    
    # Compute SDM boundary
    print("[SDM] Computing critical depth at each RPM...")
    ap_sdm = np.empty(rpm_grid.size)
    for i, rpm in enumerate(rpm_grid):
        ap_sdm[i] = sdm_critical_ap(
            model, rpm, args.xc, args.yc,
            n_modes=args.n_modes, k_intervals=args.k_intervals,
            ap_cap_mm=args.ap_cap
        )
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1:>3}/{rpm_grid.size}] rpm={rpm:>7.0f}  a_p_lim={ap_sdm[i]:.4g} mm")
    
    # Compute ZOA (optional)
    zoa_data = {"rpm": np.array([]), "ap": np.array([])}
    zoa_env = np.full(rpm_grid.size, np.nan)
    if not args.no_zoa:
        print("\n[ZOA] Computing analytical lobes...")
        zoa_data = zoa_lobe(
            model, args.xc, args.yc,
            n_lobes=args.n_lobes, ap_cap_mm=args.ap_cap
        )
        zoa_env = zoa_envelope(zoa_data, rpm_grid)
        print(f"  Generated {len(zoa_data['rpm'])} scallop points")
    
    # Compute heatmap if requested
    if args.heatmap:
        print("\n[Heatmap] Computing spectral radius grid...")
        ap_grid = np.linspace(args.ap_min, args.ap_cap, args.ap_points_heatmap)
        rho_map = compute_stability_heatmap(
            model, args.xc, args.yc,
            rpm_grid, ap_grid,
            n_modes=args.n_modes, k_intervals=args.k_intervals
        )
        
        # Plot combined figure
        plot_stability_heatmap(rho_map, rpm_grid, ap_grid, ap_sdm, zoa_data, 
                              args.xc, args.yc, out)
        save_csv_heatmap(rho_map, rpm_grid, ap_grid, out / "stability_heatmap.csv")
    
    # Save results
    save_csv_2d(rpm_grid, ap_sdm, zoa_env, out / "stability_lobe_2d.csv")
    
    meta["xc"] = args.xc
    meta["yc"] = args.yc
    meta["rpm_range"] = [args.rpm_min, args.rpm_max]
    (out / "stability_lobe_2d_meta.json").write_text(json.dumps(meta, indent=2))
    
    print(f"\nSaved results to {out}")
    
    # Print summary
    valid = np.isfinite(ap_sdm)
    if np.any(valid):
        print(f"\nSummary:")
        print(f"  Max stable depth: {np.max(ap_sdm[valid]):.3g} mm at {rpm_grid[valid][np.argmax(ap_sdm[valid])]:.0f} rpm")
        print(f"  Min stable depth: {np.min(ap_sdm[valid]):.3g} mm")
        
        # Find stability pockets (local maxima)
        from scipy.signal import find_peaks
        if HAS_SCIPY:
            peaks, _ = find_peaks(ap_sdm[valid], height=0.1*np.max(ap_sdm[valid]))
            if len(peaks) > 0:
                print(f"  Stability pockets found at: {rpm_grid[valid][peaks].astype(int)} rpm")


if __name__ == "__main__":
    main()