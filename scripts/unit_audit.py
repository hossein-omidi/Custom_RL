#!/usr/bin/env python3
"""Generate end-to-end unit consistency audit report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from custom_rl.plants import f_nonlinear2 as fmod
from custom_rl.plants.plate import PlatePlant
from custom_rl.plants.unit_audit import audit_plant, format_report_md


def main() -> int:
    parser = argparse.ArgumentParser(description="Unit consistency audit")
    parser.add_argument(
        "--mode",
        choices=["surface_normal_reduced", "feed_normal_full", "both"],
        default="both",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "docs" / "UNIT_AUDIT.md",
    )
    args = parser.parse_args()

    modes = (
        ["surface_normal_reduced", "feed_normal_full"]
        if args.mode == "both"
        else [args.mode]
    )

    sections: list[str] = []
    verdicts: list[str] = []
    for mode in modes:
        plant = PlatePlant(
            enable_geometry_uncertainty=False,
            enable_sensor_uncertainty=False,
            enable_process_noise=False,
            displacement_model=mode,
            trajectory_mode="middle_line",
        )
        plant.reset(__import__("numpy").random.default_rng(0))
        fmod.reset_episode_state(plant._modal_state_history)
        rep = audit_plant(plant)
        verdicts.append(rep.verdict())
        sections.append(format_report_md(rep))
        print(f"{mode}: {rep.verdict()} ({sum(1 for f in rep.findings if f.status=='PASS')} pass, "
              f"{sum(1 for f in rep.findings if f.status=='WARN')} warn, "
              f"{sum(1 for f in rep.findings if f.status=='FAIL')} fail)")

    overall = "FAIL" if "FAIL" in verdicts else ("WARN" if "WARN" in verdicts else "PASS")
    body = "\n\n---\n\n".join(sections)
    header = f"# Unit Audit Summary\n\n**Overall verdict:** `{overall}`\n\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(header + body, encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0 if overall != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
