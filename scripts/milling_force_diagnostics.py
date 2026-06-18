#!/usr/bin/env python3
"""Plot directional milling diagnostics for one spindle revolution."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from custom_rl.plants import f_nonlinear2 as fmod
from custom_rl.plants.plate import PlatePlant


def main() -> int:
    parser = argparse.ArgumentParser(description="Milling force one-revolution diagnostics")
    parser.add_argument("--omega", type=float, default=800.0)
    parser.add_argument("--ac", type=float, default=3.0)
    parser.add_argument("--delta-f", type=float, default=0.0)
    parser.add_argument("--delta-n", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=Path("plots/milling_diagnostics.png"))
    args = parser.parse_args()

    plant = PlatePlant(
        enable_geometry_uncertainty=False,
        enable_sensor_uncertainty=False,
        enable_process_noise=False,
        trajectory_mode="middle_line",
        displacement_model="feed_normal_full",
        milling_type="slotting",
    )
    plant.reset(__import__("numpy").random.default_rng(0))

    data = fmod.get_revolution_diagnostics(
        0.0,
        args.omega,
        args.ac,
        delta_f=args.delta_f,
        delta_n=args.delta_n,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
    t = data["t"]
    axes[0, 0].plot(t, data["phi"])
    axes[0, 0].set_ylabel("phi_j")
    axes[0, 1].plot(t, data["g"])
    axes[0, 1].set_ylabel("g(phi)")
    axes[1, 0].plot(t, np.asarray(data["h"]) * 1e3)
    axes[1, 0].set_ylabel("h_j [mm]")
    axes[1, 1].plot(t, np.asarray(data["Delta_f"]) * 1e3, label="Delta_f")
    axes[1, 1].plot(t, np.asarray(data["Delta_n"]) * 1e3, label="Delta_n")
    axes[1, 1].set_ylabel("Delta [mm]")
    axes[1, 1].legend()
    axes[2, 0].plot(t, data["Ft"], label="Ft")
    axes[2, 0].plot(t, data["Fr"], label="Fr")
    axes[2, 0].set_ylabel("Ft, Fr [N]")
    axes[2, 0].legend()
    axes[2, 1].plot(t, data["F_feed"], label="F_feed")
    axes[2, 1].plot(t, data["F_normal"], label="F_normal")
    axes[2, 1].set_ylabel("F_feed, F_normal [N]")
    axes[2, 1].legend()
    axes[3, 0].plot(t, data["theta"])
    axes[3, 0].set_ylabel("theta")
    axes[3, 0].set_xlabel("t [s]")
    axes[3, 1].axis("off")
    axes[3, 1].text(
        0.1,
        0.5,
        f"model={fmod.milling_cfg.displacement_model}\n"
        f"milling_type={fmod.milling_cfg.milling_type}\n"
        f"phi_st={fmod.milling_cfg.phi_st:.3f}\n"
        f"phi_ex={fmod.milling_cfg.phi_ex:.3f}\n"
        f"ac={args.ac} mm",
        fontsize=10,
        family="monospace",
    )
    fig.suptitle("Two-direction milling — one revolution")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
