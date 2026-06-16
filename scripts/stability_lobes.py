#!/usr/bin/env python3
"""
Stability-lobe analysis: p_unstable(omega, ac) over a grid.

Usage:
    python scripts/stability_lobes.py --config conf_fast
    python scripts/stability_lobes.py --config conf1 --controller uncontrolled trained
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs import list_configs, load_config
from custom_rl import register_envs

ENV_ID = "CustomODEPlate-v0"


def _build_env_kwargs(cfg: dict) -> dict:
    kw = dict(cfg["env"])
    kw.update(cfg["reward"])
    return kw


def _classify_rollout(
    sensor_w: np.ndarray,
    transient_fraction: float,
    displacement_limit: float,
    rms_factor: float,
    terminated: bool,
    term_reason: str,
) -> tuple[bool, float]:
    """
    Classify stable vs unstable from physical sensor displacement.

    Unstable if: failure termination OR post-transient RMS exceeds rms_factor * baseline.
    """
    if terminated and term_reason.startswith("excessive"):
        return True, float(np.nanmax(np.abs(sensor_w)))

    n = sensor_w.shape[0]
    start = int(np.floor(transient_fraction * n))
    tail = sensor_w[start:]
    if tail.size == 0:
        return False, 0.0

    rms = float(np.sqrt(np.mean(tail**2)))
    head = sensor_w[: max(start, 1)]
    baseline = float(np.sqrt(np.mean(head**2)) + 1e-12)
    unstable = rms > rms_factor * baseline or rms > displacement_limit
    return unstable, rms


def _run_fixed_point_rollout(
    env,
    omega: float,
    ac: float,
    seed: int,
    horizon: int,
    policy: PPO | None,
) -> dict:
    plant = env.unwrapped.plant
    u_fixed = plant.physical_to_normalized_action(np.array([omega, ac], dtype=np.float64))

    obs, _ = env.reset(seed=seed)
    sensor_ws = []
    actions_phys = []
    terminated = False
    term_reason = "pass_complete"

    for _ in range(horizon):
        if policy is not None:
            action, _ = policy.predict(obs, deterministic=True)
        else:
            action = u_fixed
        obs, _, term, trunc, info = env.step(action)
        w = np.asarray(info.get("sensor_w", []), dtype=np.float64)
        if w.size:
            sensor_ws.append(float(np.max(np.abs(w))))
        ap = info.get("action_phys")
        if ap is not None:
            actions_phys.append(np.asarray(ap, dtype=np.float64))
        if term or trunc:
            terminated = term
            term_reason = str(info.get("termination_reason", "truncated"))
            break

    sw = np.asarray(sensor_ws, dtype=np.float64) if sensor_ws else np.zeros(1)
    ap_arr = np.asarray(actions_phys, dtype=np.float64) if actions_phys else np.zeros((0, 2))
    return {
        "sensor_w_max": sw,
        "terminated": terminated,
        "term_reason": term_reason,
        "actions_phys": ap_arr,
    }


def run_stability_grid(
    cfg: dict,
    controller: str = "uncontrolled",
    model_path: Path | None = None,
) -> dict:
    sl = cfg["stability_lobe"]
    omega_grid = np.asarray(sl["omega_grid"], dtype=np.float64)
    ac_grid = np.asarray(sl["ac_grid"], dtype=np.float64)
    seeds = list(sl.get("seeds", cfg["seeds"]))
    n_rollouts = int(sl["n_rollouts"])
    horizon = int(sl["horizon_steps"])

    register_envs()
    env_kwargs = _build_env_kwargs(cfg)
    env = gym.make(ENV_ID, **env_kwargs)
    plant = env.unwrapped.plant
    disp_limit = float(plant.displacement_failure_limit)

    policy = None
    if controller == "trained":
        if model_path is None or not model_path.exists():
            raise FileNotFoundError(f"Trained policy required: {model_path}")
        policy = PPO.load(str(model_path), device="cpu")

    n_omega, n_ac = len(omega_grid), len(ac_grid)
    p_unstable = np.zeros((n_omega, n_ac), dtype=np.float64)
    rms_mean = np.zeros((n_omega, n_ac), dtype=np.float64)
    rms_std = np.zeros((n_omega, n_ac), dtype=np.float64)
    visit_density = np.zeros((n_omega, n_ac), dtype=np.float64)

    for i, omega in enumerate(omega_grid):
        for j, ac in enumerate(ac_grid):
            flags = []
            rms_vals = []
            for rollout in range(n_rollouts):
                seed = int(seeds[rollout % len(seeds)] + 1000 * i + 10 * j + rollout)
                out = _run_fixed_point_rollout(
                    env, float(omega), float(ac), seed, horizon, policy
                )
                unstable, rms = _classify_rollout(
                    out["sensor_w_max"],
                    float(sl["transient_fraction"]),
                    disp_limit,
                    float(sl["rms_unstable_factor"]),
                    out["terminated"],
                    out["term_reason"],
                )
                flags.append(unstable)
                rms_vals.append(rms)
                if policy is not None and out["actions_phys"].size:
                    for ap in out["actions_phys"]:
                        oi = int(np.argmin(np.abs(omega_grid - ap[0])))
                        aj = int(np.argmin(np.abs(ac_grid - ap[1])))
                        visit_density[oi, aj] += 1.0

            p_unstable[i, j] = float(np.mean(flags))
            rms_mean[i, j] = float(np.mean(rms_vals))
            rms_std[i, j] = float(np.std(rms_vals))

    env.close()

    threshold = float(sl["unstable_threshold"])
    critical_ac = np.full(n_omega, np.nan, dtype=np.float64)
    for i in range(n_omega):
        row = p_unstable[i, :]
        cross = np.where(row >= threshold)[0]
        if cross.size:
            critical_ac[i] = float(ac_grid[cross[0]])
        else:
            above = np.where(row > 0.0)[0]
            if above.size:
                j0 = int(above[-1])
                if j0 + 1 < n_ac:
                    t = (threshold - row[j0]) / max(row[j0 + 1] - row[j0], 1e-12)
                    critical_ac[i] = float(ac_grid[j0] + t * (ac_grid[j0 + 1] - ac_grid[j0]))

    return {
        "omega_grid": omega_grid,
        "ac_grid": ac_grid,
        "p_unstable": p_unstable,
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "critical_ac": critical_ac,
        "visit_density": visit_density,
        "controller": controller,
        "threshold": threshold,
    }


def _plot_heatmaps(data: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    og = data["omega_grid"]
    ag = data["ac_grid"]
    pu = data["p_unstable"]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        pu.T,
        origin="lower",
        aspect="auto",
        extent=[og[0], og[-1], ag[0], ag[-1]],
        vmin=0.0,
        vmax=1.0,
        cmap="RdYlGn_r",
    )
    plt.colorbar(im, ax=ax, label="p_unstable")
    ax.set_xlabel("Spindle speed omega [rad/s]")
    ax.set_ylabel("Depth of cut ac")
    ax.set_title(f"Stability lobe probability ({data['controller']})")
    fig.tight_layout()
    fig.savefig(out_dir / "stability_lobe_prob_heatmap.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(og, data["critical_ac"], "b-", lw=2, label="p_unstable threshold")
    ax.fill_between(
        og,
        np.maximum(data["critical_ac"] - data["rms_std"].mean(axis=1), 0.0),
        data["critical_ac"] + data["rms_std"].mean(axis=1),
        alpha=0.25,
        color="blue",
    )
    ax.set_xlabel("omega [rad/s]")
    ax.set_ylabel("critical ac")
    ax.set_title("Stability boundary")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "stability_lobe_boundary.png", dpi=150)
    plt.close(fig)

    vd = data["visit_density"]
    if np.any(vd > 0):
        fig, ax = plt.subplots(figsize=(8, 5))
        im2 = ax.imshow(
            vd.T,
            origin="lower",
            aspect="auto",
            extent=[og[0], og[-1], ag[0], ag[-1]],
            cmap="Blues",
        )
        plt.colorbar(im2, ax=ax, label="visit count")
        ax.set_xlabel("omega [rad/s]")
        ax.set_ylabel("ac")
        ax.set_title("RL action visitation density")
        fig.tight_layout()
        fig.savefig(out_dir / "action_visit_heatmap.png", dpi=150)
        plt.close(fig)


def _save_data(data: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "stability_lobe_data.npz",
        omega_grid=data["omega_grid"],
        ac_grid=data["ac_grid"],
        p_unstable=data["p_unstable"],
        rms_mean=data["rms_mean"],
        rms_std=data["rms_std"],
        critical_ac=data["critical_ac"],
        visit_density=data["visit_density"],
        threshold=data["threshold"],
        controller=data["controller"],
    )
    with open(out_dir / "stability_lobe_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["omega", "ac", "p_unstable", "rms_mean", "rms_std"])
        for i, om in enumerate(data["omega_grid"]):
            for j, ac in enumerate(data["ac_grid"]):
                writer.writerow([
                    om, ac,
                    data["p_unstable"][i, j],
                    data["rms_mean"][i, j],
                    data["rms_std"][i, j],
                ])


def main() -> int:
    available = ", ".join(list_configs()) or "conf_fast"
    parser = argparse.ArgumentParser(description="Stability-lobe Monte Carlo analysis")
    parser.add_argument("--config", default="conf_fast", help=f"Config ({available})")
    parser.add_argument(
        "--controller",
        nargs="+",
        default=None,
        choices=["uncontrolled", "trained"],
        help="Controller modes to evaluate",
    )
    parser.add_argument("--model", type=Path, default=None, help="Path to trained PPO zip")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_base = Path(cfg["stability_dir"])
    modes = args.controller or cfg["stability_lobe"].get("controller_modes", ["uncontrolled"])

    model_path = args.model
    if model_path is None and "trained" in modes:
        seed = cfg["seeds"][0]
        model_path = Path(cfg["model_dir"]) / f"final_{seed}.zip"

    for mode in modes:
        print(f"[stability] controller={mode} config={args.config}")
        data = run_stability_grid(cfg, controller=mode, model_path=model_path)
        sub = out_base / mode
        _save_data(data, sub)
        _plot_heatmaps(data, sub)
        print(f"  saved -> {sub}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
