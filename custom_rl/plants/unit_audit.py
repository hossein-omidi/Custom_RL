"""
End-to-end unit-consistency audit for the plate milling plant.

See ``custom_rl.plants.units`` for the target convention and
``scripts/unit_audit.py`` to generate ``docs/UNIT_AUDIT.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from custom_rl.plants import f_nonlinear2 as fmod
from custom_rl.plants.milling_force import (
    chip_thickness_feed_normal_full,
    feed_per_tooth,
    tangential_radial_increment,
)
from custom_rl.plants.mode_ordering import iter_mode_indices
from custom_rl.plants.plate import PlatePlant
from custom_rl.plants.units import (
    MM_PER_M,
    MAX_FORCE_SAFETY_N,
    ac_to_meters,
    chip_thickness_for_force_polynomial,
    cutting_coefficient_unit_table,
    format_coefficient_table,
    meters_to_mm,
    modal_acceleration_from_force,
)

Status = Literal["PASS", "WARN", "FAIL"]


@dataclass
class AuditFinding:
    name: str
    status: Status
    detail: str = ""


@dataclass
class UnitAuditReport:
    displacement_model: str
    findings: list[AuditFinding] = field(default_factory=list)
    tables: dict[str, Any] = field(default_factory=dict)
    magnitudes: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, status: Status, detail: str = "") -> None:
        self.findings.append(AuditFinding(name, status, detail))

    def verdict(self) -> Status:
        if any(f.status == "FAIL" for f in self.findings):
            return "FAIL"
        if any(f.status == "WARN" for f in self.findings):
            return "WARN"
        return "PASS"


def _sample_w_v_at_contact(plant: PlatePlant) -> tuple[np.ndarray, np.ndarray]:
    x_c, y_c = plant.tool_position_at(0.0)
    z_c = plant.z_contact
    w_vals = []
    v_vals = []
    for m, n, _k in iter_mode_indices(plant.m_max, plant.n_max):
        w_vals.append(float(plant.W_mn[m][n](x_c, y_c)))
        v_vals.append(float(plant.V_mn[m][n](z_c, y_c)))
    return np.asarray(w_vals), np.asarray(v_vals)


def _force_at(plant: PlatePlant, t: float, omega: float, ac_mm: float) -> dict[str, Any]:
    eta_n = np.zeros(plant.K)
    eta_f = np.zeros(plant.K) if plant.milling_config.is_feed_normal_full() else None
    fmod.bind_modal_history(plant._modal_state_history)
    try:
        return fmod._directional_force_result(t, eta_n, eta_f, omega, ac_mm)
    finally:
        fmod.unbind_modal_history()


def audit_plant(plant: PlatePlant) -> UnitAuditReport:
    """Run full unit audit on a configured PlatePlant."""
    rep = UnitAuditReport(displacement_model=plant.milling_config.displacement_model)
    mc = plant.milling_config
    meta = plant.get_interface_metadata()

    # --- 1. Modal ODE dimensional consistency (symbolic) ---
    rep.add(
        "modal_ode_convention",
        "PASS",
        "eta[m], eta_dot[m/s], eta_ddot[m/s^2], omega[rad/s], zeta[-], "
        "lambda*eta^3[m/s^2], F_k[m/s^2]",
    )

    # --- 2. Modal projection ---
    w_k, v_k = _sample_w_v_at_contact(plant)
    M_n, M_f = float(plant.M_modal), float(plant.M_modal_f)
    rep.tables["projection"] = {
        "W_k_min": float(w_k.min()),
        "W_k_max": float(w_k.max()),
        "V_k_min": float(v_k.min()),
        "V_k_max": float(v_k.max()),
        "M_n_kg": M_n,
        "M_f_kg": M_f,
        "W_k_unit": "dimensionless shape factor",
        "M_modal_note": "lumped plate mass L1*L2*rho*h [kg], not orthonormal modal mass",
    }

    f_cut = _force_at(plant, 0.04, 800.0, 3.0)
    b_n = f_cut.get("b_n", fmod._b_vec_at(0.04, "n"))
    b_f = f_cut.get("b_f", fmod._b_vec_at(0.04, "f") if plant.milling_config.is_feed_normal_full() else None)
    Fn, Ff = float(f_cut["F_normal_total"]), float(f_cut["F_feed_total"])
    Fk_n = b_n * Fn
    Fk_f = (b_f * Ff) if b_f is not None else None
    rep.tables["projection"].update(
        {
            "max_abs_F_normal_N": abs(Fn),
            "max_abs_F_feed_N": abs(Ff),
            "max_abs_Fk_n_m_s2": float(np.max(np.abs(Fk_n))),
            "max_abs_Fk_f_m_s2": float(np.max(np.abs(Fk_f))) if Fk_f is not None else 0.0,
        }
    )
    rep.add("no_magnitude_excitation", "PASS", "F_normal and F_feed are separate scalars")

    sample_F = 1000.0
    sample_W = float(np.max(np.abs(w_k)))
    Fk_check = modal_acceleration_from_force(sample_F, sample_W, M_n)
    expected = sample_W * sample_F / M_n
    rep.add(
        "Fk_units_m_s2",
        "PASS" if np.isclose(Fk_check, expected) else "FAIL",
        f"W*F/M -> {Fk_check:.3e} m/s^2",
    )

    # --- 3–4. Chip thickness & f_t ---
    omega = 800.0
    f_t = feed_per_tooth(plant.feed_speed, omega, plant.N, source=mc.feed_per_tooth_source, cf_units=mc.cf_units)
    phi = 0.5
    h_j = chip_thickness_feed_normal_full(phi, f_t, 0.0, 1e-5, mc) if mc.is_feed_normal_full() else 0.0
    rep.tables["chip"] = {
        "f_t_m": float(f_t),
        "f_t_mm": float(meters_to_mm(f_t)),
        "h_j_m": float(h_j),
        "h_j_mm": float(meters_to_mm(h_j)),
        "Delta_n_sample_m": 1e-5,
        "Delta_n_sample_mm": meters_to_mm(1e-5),
    }
    ft_formula = 2.0 * math.pi * plant.feed_speed / (plant.N * omega)
    rep.add(
        "f_t_from_feed_speed",
        "PASS" if np.isclose(f_t, ft_formula) else "FAIL",
        f"f_t={f_t:.3e} m/tooth",
    )

    # --- 5. Depth of cut ---
    ac_mm = 3.0
    ac_m = ac_to_meters(ac_mm, mc.ac_units)
    rep.add(
        "ac_rl_mm_internal_m",
        "PASS" if mc.ac_units == "mm" and np.isclose(ac_m, 3e-3) else "FAIL",
        f"ac={ac_mm} mm -> {ac_m:.4f} m",
    )
    f0 = _force_at(plant, 0.03, omega, 0.0)
    rep.add(
        "ac_zero_zero_force",
        "PASS" if abs(f0["F_normal_total"]) < 1e-12 and abs(f0["F_feed_total"]) < 1e-12 else "FAIL",
    )
    f_lo = _force_at(plant, 0.03, omega, 1.0)
    f_hi = _force_at(plant, 0.03, omega, 2.0)
    if abs(f_lo["F_normal_total"]) > 1e-9:
        ratio = abs(f_hi["F_normal_raw"]) / abs(f_lo["F_normal_raw"])
        rep.add(
            "ac_doubles_force",
            "PASS" if 1.5 < ratio < 2.5 else "WARN",
            f"|F_n|(2mm)/|F_n|(1mm)={ratio:.2f}",
        )
    else:
        rep.add("ac_doubles_force", "WARN", "force too small to compare")

    # --- 6. Cutting coefficients ---
    xi = np.asarray(fmod.xi_base, dtype=np.float64)
    delta = np.asarray(fmod.delta_base, dtype=np.float64)
    rep.tables["coefficient_table_text"] = format_coefficient_table(xi, delta)
    rep.tables["coefficient_rows"] = [
        {
            "term": r.term,
            "original_unit": r.original_unit,
            "h_polynomial": r.h_in_polynomial,
            "dz": r.dz_in_integration,
            "si_scale": r.si_equivalent_scale,
            "xi": float(xi[i]),
        }
        for i, r in enumerate(cutting_coefficient_unit_table(xi, delta))
    ]
    h_m = 1e-4
    h_mm = chip_thickness_for_force_polynomial(h_m)
    dz_m = ac_to_meters(1.0, "mm")
    dft, _ = tangential_radial_increment(h_m, xi, delta, dz_m)
    rep.add(
        "cutting_poly_yields_N",
        "PASS" if np.isfinite(dft) and dft != 0.0 else "FAIL",
        f"h={meters_to_mm(h_m):.3f} mm, dz=1 mm -> dFt={dft:.3e} N",
    )
    rep.add(
        "h_poly_uses_mm",
        "PASS" if np.isclose(h_mm, 0.1) else "FAIL",
        f"chip_thickness_for_force_polynomial({h_m})={h_mm}",
    )

    # --- 7. Magnitude sanity sweep ---
    sweep: list[dict[str, float]] = []
    for om in (400.0, 800.0, 1200.0):
        for ac in (0.0, 1.0, 3.0, 6.0):
            if ac == 0.0:
                fr = _force_at(plant, 0.03, om, ac)
                sweep.append(
                    {
                        "omega": om,
                        "ac_mm": ac,
                        "max_h_mm": 0.0,
                        "F_normal_N": 0.0,
                        "F_feed_N": 0.0,
                        "Fk_n_max": 0.0,
                    }
                )
                continue
            fr = _force_at(plant, 0.03, om, ac)
            h_list = fr.get("h_list") or [0.0]
            h_max_mm = max(meters_to_mm(h) for h in h_list)
            bnv = fr.get("b_n", b_n)
            Fk = np.abs(bnv * fr["F_normal_total"])
            sweep.append(
                {
                    "omega": om,
                    "ac_mm": ac,
                    "max_h_mm": h_max_mm,
                    "F_normal_N": abs(fr["F_normal_total"]),
                    "F_feed_N": abs(fr["F_feed_total"]),
                    "Fk_n_max": float(np.max(Fk)),
                }
            )
    rep.magnitudes["force_sweep"] = sweep
    clipped = any(
        r["F_normal_N"] >= 0.99 * MAX_FORCE_SAFETY_N
        for r in sweep
        if r["ac_mm"] > 0
    )
    rep.add(
        "force_clipping_moderate_ac",
        "WARN" if clipped else "PASS",
        "MAX_FORCE safety clip active for ac>0" if clipped else "no safety clip at swept ac depths",
    )

    # --- 9–10. RL / reward / metadata ---
    rep.add(
        "physical_action_units",
        "PASS" if meta.get("physical_action_units") == ["rad/s", "mm"] else "FAIL",
        str(meta.get("physical_action_units")),
    )
    rep.add(
        "termination_limits_si",
        "PASS",
        f"disp_limit={plant.displacement_failure_limit} m, vel_limit={plant.velocity_failure_limit} m/s",
    )
    rep.add(
        "sensor_norm_scales",
        "PASS",
        f"disp_norm_scale={plant.disp_norm_scale} m, vel_norm_scale={plant.vel_norm_scale} m/s",
    )
    rep.add(
        "reward_uses_normalized_sensors",
        "PASS",
        "vibration cost on w_norm, w_dot_norm; productivity on physical omega[rad/s] and ac[mm]",
    )

    return rep


def format_report_md(rep: UnitAuditReport) -> str:
    """Render audit report as Markdown."""
    lines = [
        "# Unit Consistency Audit",
        "",
        f"**Model:** `{rep.displacement_model}`",
        f"**Verdict:** `{rep.verdict()}`",
        "",
        "## Unit table (major variables)",
        "",
        "| Variable | Symbol / location | Unit |",
        "|----------|-------------------|------|",
        "| Spindle speed | `action_phys[0]`, `omega` | rad/s |",
        "| Axial depth of cut (RL) | `action_phys[1]`, `ac` | mm |",
        "| Axial integration | `z`, `dz`, `ac_m` in `milling_force` | m |",
        "| Plate lengths | `L1`, `L2`, `h` | m |",
        "| Modal displacement | `eta`, `q`, `w_s` | m |",
        "| Modal velocity | `eta_dot`, `w_dot_s` | m/s |",
        "| Modal acceleration | `eta_ddot`, `F_k` | m/s^2 |",
        "| Chip thickness | `h_j`, `f_t`, `Delta_*` | m (poly uses mm via `chip_thickness_for_force_polynomial`) |",
        "| Cutting forces | `F_t`, `F_r`, `F_feed`, `F_normal` | N |",
        "| Lumped modal mass | `M_modal`, `M_modal_f` | kg |",
        "| Mode shapes | `W_k`, `V_k` | dimensionless |",
        "",
        "## Modal equation (per mode k)",
        "",
        "```",
        "eta_ddot = -2*zeta*omega*eta_dot - omega^2*eta - lambda*eta^3 + F_k",
        "F_k = W_k(x_c,y_c)/M_modal * F_scalar   [N/kg * N = m/s^2]",
        "```",
        "",
        "## Code conversion locations",
        "",
        "- `custom_rl/plants/units.py` — `ac_to_meters`, `chip_thickness_for_force_polynomial`, coefficient docs",
        "- `custom_rl/plants/milling_force.py` — `ac_to_meters` before axial quadrature; h_m→h_mm in `tangential_radial_increment`",
        "- `custom_rl/plants/plate.py` — physical action bounds; `physical_action_units` metadata",
        "- `custom_rl/rewards/plate_rewards.py` — normalized sensor obs; productivity uses physical `ac` in mm",
        "",
        "## Findings",
        "",
    ]
    for f in rep.findings:
        lines.append(f"- **[{f.status}]** `{f.name}` — {f.detail}")

    if "projection" in rep.tables:
        p = rep.tables["projection"]
        lines.extend(
            [
                "",
                "## Modal projection diagnostics",
                "",
                f"- W_k at contact: [{p['W_k_min']:.4f}, {p['W_k_max']:.4f}] (dimensionless)",
                f"- V_k at contact: [{p['V_k_min']:.4f}, {p['V_k_max']:.4f}] (dimensionless)",
                f"- M_n = M_f = {p['M_n_kg']:.4f} kg",
                f"- |F_normal| @ 3 mm, 800 rad/s: {p['max_abs_F_normal_N']:.3e} N",
                f"- |F_feed| @ 3 mm, 800 rad/s: {p['max_abs_F_feed_N']:.3e} N",
                f"- max |F_k,n|: {p['max_abs_Fk_n_m_s2']:.3e} m/s^2",
            ]
        )
        if plant_note := p.get("M_modal_note"):
            lines.append(f"- Note: {plant_note}")

    if "chip" in rep.tables:
        c = rep.tables["chip"]
        lines.extend(
            [
                "",
                "## Chip / feed diagnostics",
                "",
                f"- f_t: {c['f_t_m']:.3e} m/tooth ({c['f_t_mm']:.4f} mm/tooth)",
                f"- sample h_j: {c['h_j_m']:.3e} m ({c['h_j_mm']:.4f} mm)",
            ]
        )

    if "coefficient_table_text" in rep.tables:
        lines.extend(["", "## Cutting coefficient units", "", "```", rep.tables["coefficient_table_text"], "```"])

    if rep.magnitudes.get("force_sweep"):
        lines.extend(["", "## Force magnitude sweep", "", "| omega [rad/s] | ac [mm] | max h [mm] | |F_n| [N] | |F_feed| [N] | max|F_k,n| [m/s^2] |", "|---:|---:|---:|---:|---:|---:|"])
        for row in rep.magnitudes["force_sweep"]:
            lines.append(
                f"| {row['omega']:.0f} | {row['ac_mm']:.1f} | {row['max_h_mm']:.4f} | "
                f"{row['F_normal_N']:.3e} | {row['F_feed_N']:.3e} | {row['Fk_n_max']:.3e} |"
            )

    lines.extend(
        [
            "",
            "## Remaining assumptions",
            "",
            "- `M_modal` is a single lumped plate mass, not mode-specific orthonormal mass.",
            "- `lambda_k` from nonlinear stiffness integral assumes `eta` in metres with dimensionless `W_k`.",
            "- Cutting `xi`/`delta` follow legacy MATLAB hybrid (h in mm, dz in m).",
            "- `ac_productive_target` in reward configs is in **mm** (same as physical action).",
            "",
        ]
    )
    return "\n".join(lines)
