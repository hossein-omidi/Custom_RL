"""Configurable parameters for directional regenerative milling force."""

from __future__ import annotations

import math
from dataclasses import dataclass


_MILLING_DEFAULTS: dict[str, dict[str, float]] = {
    "peripheral": {"phi_st": 0.0, "phi_ex": math.pi / 2.0},
    "slotting": {"phi_st": 0.0, "phi_ex": math.pi},
    "surface": {"phi_st": 0.0, "phi_ex": math.pi},
    "face": {"phi_st": 0.0, "phi_ex": math.pi},
}

_APPROXIMATIONS_REDUCED = (
    "single_modal_displacement_field_q_c",
    "Delta_f_not_modeled",
    "not_full_feed_normal_two_component_paper_form",
    "piecewise_constant_omega_for_phase_delay",
    "thrust_via_dFt_sin_phi_plus_dFr_cos_phi",
)

_APPROXIMATIONS_FULL = (
    "two_modal_displacement_fields_W_and_V",
    "straight_line_middle_of_plate_tool_path",
    "separate_feed_and_normal_force_projection",
    "piecewise_constant_omega_for_phase_delay",
    "empirical_stability_lobes_not_floquet",
    "V_field_omega_from_prime_frequency_table",
)


@dataclass
class MillingForceConfig:
    """
  Directional chip-thickness / engagement configuration.

    ``surface_normal_reduced``: one W-field, Delta_q only.
    ``feed_normal_full``: Nasiri/Moradi-style two-component chip thickness with
    separate W (normal) and V (feed) modal fields and force projections.
    """

    milling_type: str = "surface"
    phi_st: float | None = None
    phi_ex: float | None = None
    phi_0: float = 0.0
    angle_convention: str = "phi_from_feed_ccw"
    cutter_diameter: float = 0.02
    helix_angle: float = 0.0
    axial_quadrature_points: int = 5
    ac_via_axial_integration: bool = True
    ac_units: str = "mm"
    displacement_model: str = "feed_normal_full"
    feed_per_tooth_source: str = "from_feed_speed"
    cf_units: str = "m_per_tooth"
    z_contact: float | None = None

    def __post_init__(self) -> None:
        mtype = str(self.milling_type).lower().strip()
        if mtype not in _MILLING_DEFAULTS:
            raise ValueError(
                f"milling_type must be one of {list(_MILLING_DEFAULTS)}, got {self.milling_type!r}."
            )
        self.milling_type = mtype
        defaults = _MILLING_DEFAULTS[mtype]
        if self.phi_st is None:
            self.phi_st = float(defaults["phi_st"])
        if self.phi_ex is None:
            self.phi_ex = float(defaults["phi_ex"])
        if self.axial_quadrature_points < 1:
            raise ValueError("axial_quadrature_points must be >= 1.")
        if self.cutter_diameter <= 0.0:
            raise ValueError("cutter_diameter must be > 0.")
        self.displacement_model = self._normalize_displacement_model(self.displacement_model)
        if self.displacement_model not in {"surface_normal_reduced", "feed_normal_full"}:
            raise ValueError(
                "displacement_model must be 'surface_normal_reduced' or 'feed_normal_full'."
            )
        if self.feed_per_tooth_source not in {"from_feed_speed", "from_cf"}:
            raise ValueError("feed_per_tooth_source must be 'from_feed_speed' or 'from_cf'.")
        if self.cf_units not in {"m_per_tooth", "mm_per_tooth"}:
            raise ValueError("cf_units must be 'm_per_tooth' or 'mm_per_tooth'.")
        if self.ac_units not in {"mm", "m"}:
            raise ValueError("ac_units must be 'mm' or 'm'.")

    @staticmethod
    def _normalize_displacement_model(name: str) -> str:
        aliases = {
            "out_of_plane_only": "surface_normal_reduced",
            "reduced_one_direction": "surface_normal_reduced",
            "feed_normal": "feed_normal_full",
        }
        key = str(name).lower().strip()
        return aliases.get(key, key)

    def is_surface_normal_reduced(self) -> bool:
        return self.displacement_model == "surface_normal_reduced"

    def is_feed_normal_full(self) -> bool:
        return self.displacement_model == "feed_normal_full"

    def engaged_arc_width(self) -> float:
        st, ex = float(self.phi_st), float(self.phi_ex)
        if st <= ex:
            return ex - st
        return (2.0 * math.pi - st) + ex

    def approximations(self) -> tuple[str, ...]:
        if self.is_feed_normal_full():
            return _APPROXIMATIONS_FULL
        return _APPROXIMATIONS_REDUCED

    def to_dict(self) -> dict:
        return {
            "milling_type": self.milling_type,
            "phi_st": self.phi_st,
            "phi_ex": self.phi_ex,
            "phi_0": self.phi_0,
            "angle_convention": self.angle_convention,
            "cutter_diameter": self.cutter_diameter,
            "helix_angle": self.helix_angle,
            "axial_quadrature_points": self.axial_quadrature_points,
            "ac_via_axial_integration": self.ac_via_axial_integration,
            "ac_units": self.ac_units,
            "displacement_model": self.displacement_model,
            "feed_per_tooth_source": self.feed_per_tooth_source,
            "cf_units": self.cf_units,
            "z_contact": self.z_contact,
            "approximations": list(self.approximations()),
            "is_surface_normal_reduced": self.is_surface_normal_reduced(),
            "is_feed_normal_full": self.is_feed_normal_full(),
        }
