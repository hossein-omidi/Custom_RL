#!/usr/bin/env python3
"""
Stability-lobe analysis for the regenerative surface-milling plant.

Uncontrolled mode: conventional stability lobe  critical_ac(omega)  from fixed
(omega, ac) with Test2-style response classification and Monte Carlo seeds.

Trained mode: closed-loop *performance map* at grid initial conditions (agent may
change omega/ac) plus action-visitation heatmap — not a conventional stability lobe.

Usage:
    python scripts/stability_lobes.py --config conf_fast
    python scripts/stability_lobes.py --config conf_fast --controller uncontrolled trained
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
from custom_rl.analysis.milling_rollout import (
    rollout_env_trained,
    rollout_uncontrolled,
)
from custom_rl.analysis.response_classification import (
    classify_milling_response,
    metrics_from_rollout_dict,
)
from custom_rl.analysis.stability_lobe import (
    critical_ac_curve_with_mc_std,
    critical_ac_from_probability,
    monotonicity_violations,
    start_position_sensitivity,
)
from custom_rl.plants.plate import PlatePlant
from custom_rl.plants.time_scales import recommend_integration_dt, tooth_period

ENV_ID = "CustomODEPlate-v0"


def _build_env_kwargs(cfg: dict) -> dict:
    kw = dict(cfg["env"])
    kw.update(cfg["reward"])
    return kw


def _resolve_t_final(sl: dict, plant: PlatePlant, omega_grid: np.ndarray) -> float:
    if sl.get("t_final_s") is not None:
        return float(sl["t_final_s"])
    horizon = sl.get("horizon_steps")
    macro_dt = float(sl.get("macro_dt_s", 0.002))
    if horizon is not None:
        return float(horizon) * macro_dt
    om_min = float(np.min(omega_grid))
    om_max = float(np.max(omega_grid))
    tau_max = tooth_period(om_max, plant.N)
    t_rev_min = 2.0 * math.pi / om_min
    return max(2.0, 40.0 * tau_max, 5.0 * t_rev_min)


def _plant_kwargs_from_cfg(cfg: dict) -> dict:
    import inspect

    env_kw = _build_env_kwargs(cfg)
    params = inspect.signature(PlatePlant.__init__).parameters
    return {k: v for k, v in env_kw.items() if k in params and k != "self"}


def _make_plant(cfg: dict) -> PlatePlant:
    return PlatePlant(**_plant_kwargs_from_cfg(cfg))


def run_uncontrolled_lobe(cfg: dict) -> dict:
    sl = cfg["stability_lobe"]
    omega_grid = np.asarray(sl["omega_grid"], dtype=np.float64)
    ac_grid = np.asarray(sl["ac_grid"], dtype=np.float64)
    seeds = list(sl.get("seeds", cfg["seeds"]))
    n_rollouts = int(sl["n_rollouts"])
    macro_dt = float(sl.get("macro_dt_s", cfg["env"].get("dt", 0.002)))
    transient_fraction = float(sl["transient_fraction"])
    threshold = float(sl["unstable_threshold"])
    growth_factor = float(sl.get("growth_factor", sl.get("rms_unstable_factor", 3.0)))
    perturb = float(sl.get("initial_perturb_m", 1e-6))

    plant = _make_plant(cfg)
    if sl.get("pass_sampling"):
        plant.pass_sampling = str(sl["pass_sampling"])
    plant.reset(np.random.default_rng(0))
    t_final = _resolve_t_final(sl, plant, omega_grid)
    disp_limit = float(plant.displacement_failure_limit)

    print(f"  t_final={t_final:.3f}s  macro_dt={macro_dt}s  rollouts/point={n_rollouts}")
    scales = recommend_integration_dt(plant, macro_dt=macro_dt, training_mode=True)
    print(f"  sub_dt={scales['dt_recommended_training_substep_s']:.2e}s  "
          f"n_substeps={scales['n_substeps']}")

    n_omega, n_ac = len(omega_grid), len(ac_grid)
    p_chatter = np.zeros((n_omega, n_ac), dtype=np.float64)
    p_failed = np.zeros((n_omega, n_ac), dtype=np.float64)
    p_bounded = np.zeros((n_omega, n_ac), dtype=np.float64)
    rms_mean = np.zeros((n_omega, n_ac), dtype=np.float64)
    rms_std = np.zeros((n_omega, n_ac), dtype=np.float64)
    all_records: list[dict] = []

    instability = np.zeros((n_omega, n_ac, n_rollouts), dtype=bool)
    for i, omega in enumerate(omega_grid):
        for j, ac in enumerate(ac_grid):
            classes: list[str] = []
            rms_vals: list[float] = []
            for rollout in range(n_rollouts):
                seed = int(seeds[rollout % len(seeds)] + 1000 * i + 10 * j + rollout)
                rng = np.random.default_rng(seed)
                raw = rollout_uncontrolled(
                    plant, float(omega), float(ac),
                    t_final=t_final, macro_dt=macro_dt, rng=rng, perturb=perturb,
                )
                raw["transient_fraction"] = transient_fraction
                metrics = metrics_from_rollout_dict(raw)
                klass = classify_milling_response(
                    metrics,
                    transient_fraction=transient_fraction,
                    displacement_limit_m=disp_limit,
                    growth_factor=growth_factor,
                )
                classes.append(klass)
                rms_vals.append(metrics.rms_w_post_m)
                instability[i, j, rollout] = klass == "chatter-like"
                all_records.append({**raw, "class": klass, "seed": seed})

            p_chatter[i, j] = float(np.mean([c == "chatter-like" for c in classes]))
            p_failed[i, j] = float(np.mean([c == "failed" for c in classes]))
            p_bounded[i, j] = float(np.mean([c == "bounded" for c in classes]))
            rms_mean[i, j] = float(np.mean(rms_vals))
            rms_std[i, j] = float(np.std(rms_vals))

    critical_ac, critical_unc, critical_mc_std, p_mono = critical_ac_curve_with_mc_std(
        omega_grid, ac_grid, instability, threshold=threshold,
    )
    start_sens = start_position_sensitivity(all_records, omega_grid, ac_grid)
    mono_v = monotonicity_violations(p_mono)

    return {
        "map_type": "uncontrolled_stability_lobe",
        "omega_grid": omega_grid,
        "ac_grid": ac_grid,
        "p_chatter": p_chatter,
        "p_failed": p_failed,
        "p_bounded": p_bounded,
        "p_mono": p_mono,
        "p_unstable": p_chatter,
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "critical_ac": critical_ac,
        "critical_ac_uncertainty": critical_unc,
        "critical_ac_std": critical_mc_std,
        "threshold": threshold,
        "t_final_s": t_final,
        "macro_dt_s": macro_dt,
        "start_position": start_sens,
        "monotonicity_violations": mono_v,
        "records": all_records,
    }


def run_trained_performance_map(cfg: dict, model_path: Path) -> dict:
    sl = cfg["stability_lobe"]
    omega_grid = np.asarray(sl["omega_grid"], dtype=np.float64)
    ac_grid = np.asarray(sl["ac_grid"], dtype=np.float64)
    seeds = list(sl.get("seeds", cfg["seeds"]))
    n_rollouts = int(sl["n_rollouts"])
    transient_fraction = float(sl["transient_fraction"])
    growth_factor = float(sl.get("growth_factor", sl.get("rms_unstable_factor", 3.0)))
    perturb = float(sl.get("initial_perturb_m", 1e-6))

    if not model_path.exists():
        raise FileNotFoundError(f"Trained policy required: {model_path}")

    register_envs()
    env = gym.make(ENV_ID, **_build_env_kwargs(cfg))
    plant = env.unwrapped.plant
    step_dt = float(env.unwrapped._step_dt)
    t_final = _resolve_t_final(sl, plant, omega_grid)
    n_steps = max(int(math.ceil(t_final / step_dt)), 10)
    disp_limit = float(plant.displacement_failure_limit)
    policy = PPO.load(str(model_path), device="cpu")

    n_omega, n_ac = len(omega_grid), len(ac_grid)
    p_chatter = np.zeros((n_omega, n_ac), dtype=np.float64)
    p_failed = np.zeros((n_omega, n_ac), dtype=np.float64)
    rms_mean = np.zeros((n_omega, n_ac), dtype=np.float64)
    reward_mean = np.zeros((n_omega, n_ac), dtype=np.float64)
    visit_density = np.zeros((n_omega, n_ac), dtype=np.float64)

    for i, omega in enumerate(omega_grid):
        for j, ac in enumerate(ac_grid):
            classes: list[str] = []
            rms_vals: list[float] = []
            rew_vals: list[float] = []
            for rollout in range(n_rollouts):
                seed = int(seeds[rollout % len(seeds)] + 2000 * i + 10 * j + rollout)
                raw = rollout_env_trained(
                    env, float(omega), float(ac),
                    seed=seed, n_steps=n_steps, policy=policy, perturb=perturb,
                )
                raw["transient_fraction"] = transient_fraction
                metrics = metrics_from_rollout_dict(raw)
                klass = classify_milling_response(
                    metrics,
                    transient_fraction=transient_fraction,
                    displacement_limit_m=disp_limit,
                    growth_factor=growth_factor,
                )
                classes.append(klass)
                rms_vals.append(metrics.rms_w_post_m)
                rew_vals.append(float(raw.get("mean_reward", 0.0)))
                for ap in raw.get("actions_phys", np.zeros((0, 2))):
                    oi = int(np.argmin(np.abs(omega_grid - ap[0])))
                    aj = int(np.argmin(np.abs(ac_grid - ap[1])))
                    visit_density[oi, aj] += 1.0

            p_chatter[i, j] = float(np.mean([c == "chatter-like" for c in classes]))
            p_failed[i, j] = float(np.mean([c == "failed" for c in classes]))
            rms_mean[i, j] = float(np.mean(rms_vals))
            reward_mean[i, j] = float(np.mean(rew_vals))

    env.close()
    return {
        "map_type": "closed_loop_performance_map",
        "omega_grid": omega_grid,
        "ac_grid": ac_grid,
        "p_chatter": p_chatter,
        "p_failed": p_failed,
        "p_unstable": p_chatter,
        "rms_mean": rms_mean,
        "reward_mean": reward_mean,
        "visit_density": visit_density,
        "t_final_s": t_final,
        "step_dt_s": step_dt,
    }


def _plot_uncontrolled(data: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    og, ag = data["omega_grid"], data["ac_grid"]
    pc = data["p_chatter"]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        pc.T, origin="lower", aspect="auto",
        extent=[og[0], og[-1], ag[0], ag[-1]], vmin=0.0, vmax=1.0, cmap="RdYlGn_r",
    )
    plt.colorbar(im, ax=ax, label="P(chatter-like)")
    ax.set_xlabel("omega [rad/s]")
    ax.set_ylabel("ac [mm]")
    ax.set_title("Uncontrolled stability lobe (Monte Carlo)")
    fig.tight_layout()
    fig.savefig(out_dir / "stability_lobe_prob_heatmap.png", dpi=150)
    plt.close(fig)

    pf = data["p_failed"]
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        pf.T, origin="lower", aspect="auto",
        extent=[og[0], og[-1], ag[0], ag[-1]], vmin=0.0, vmax=1.0, cmap="Oranges",
    )
    plt.colorbar(im, ax=ax, label="P(failed)")
    ax.set_xlabel("omega [rad/s]")
    ax.set_ylabel("ac [mm]")
    ax.set_title("Numerical / clip failure probability")
    fig.tight_layout()
    fig.savefig(out_dir / "failure_prob_heatmap.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(og, data["critical_ac"], "b-", lw=2, label=f"P(chatter)={data['threshold']}")
    unc = data["critical_ac_uncertainty"]
    ax.fill_between(og, data["critical_ac"] - unc, data["critical_ac"] + unc, alpha=0.25)
    mc_std = data.get("critical_ac_std")
    if mc_std is not None and np.any(np.isfinite(mc_std)):
        ax.fill_between(
            og,
            data["critical_ac"] - mc_std,
            data["critical_ac"] + mc_std,
            alpha=0.15,
            color="orange",
            label="MC std",
        )
    ax.set_xlabel("omega [rad/s]")
    ax.set_ylabel("critical ac [mm]")
    ax.set_title("Conventional stability boundary (uncontrolled)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "stability_lobe_boundary.png", dpi=150)
    plt.close(fig)


def _plot_trained(data: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    og, ag = data["omega_grid"], data["ac_grid"]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        data["p_chatter"].T, origin="lower", aspect="auto",
        extent=[og[0], og[-1], ag[0], ag[-1]], vmin=0.0, vmax=1.0, cmap="RdYlGn_r",
    )
    plt.colorbar(im, ax=ax, label="P(chatter-like at initial op.)")
    ax.set_xlabel("initial omega [rad/s]")
    ax.set_ylabel("initial ac [mm]")
    ax.set_title("Closed-loop performance map (NOT conventional stability lobe)")
    fig.tight_layout()
    fig.savefig(out_dir / "closed_loop_performance_map.png", dpi=150)
    plt.close(fig)

    vd = data["visit_density"]
    if np.any(vd > 0):
        fig, ax = plt.subplots(figsize=(8, 5))
        im2 = ax.imshow(
            vd.T, origin="lower", aspect="auto",
            extent=[og[0], og[-1], ag[0], ag[-1]], cmap="Blues",
        )
        plt.colorbar(im2, ax=ax, label="action visit count")
        ax.set_xlabel("omega [rad/s]")
        ax.set_ylabel("ac [mm]")
        ax.set_title("RL action visitation density")
        fig.tight_layout()
        fig.savefig(out_dir / "action_visit_heatmap.png", dpi=150)
        plt.close(fig)


def _save_outputs(data: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    og, ag = data["omega_grid"], data["ac_grid"]

    np.savez_compressed(
        out_dir / "stability_lobe_data.npz",
        map_type=np.array(data["map_type"]),
        omega_grid=og,
        ac_grid=ag,
        p_chatter=data["p_chatter"],
        p_failed=data["p_failed"],
        p_bounded=data.get("p_bounded", np.zeros_like(data["p_chatter"])),
        p_unstable=data["p_unstable"],
        rms_mean=data["rms_mean"],
        rms_std=data.get("rms_std", np.zeros_like(data["rms_mean"])),
        critical_ac=data.get("critical_ac", np.full(len(og), np.nan)),
        critical_ac_uncertainty=data.get("critical_ac_uncertainty", np.zeros(len(og))),
        critical_ac_std=data.get("critical_ac_std", np.zeros(len(og))),
        threshold=np.array(data.get("threshold", 0.5)),
        omega_units=np.array("rad/s"),
        ac_units=np.array("mm"),
        rms_units=np.array("m"),
        t_final_s=np.array(data.get("t_final_s", 0.0)),
    )

    with open(out_dir / "stability_lobe_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "omega_rad_s", "ac_mm", "p_chatter", "p_failed", "p_bounded",
            "rms_mean_m", "rms_std_m",
        ])
        for i, om in enumerate(og):
            for j, ac in enumerate(ag):
                w.writerow([
                    om, ac,
                    data["p_chatter"][i, j],
                    data["p_failed"][i, j],
                    data.get("p_bounded", np.zeros_like(data["p_chatter"]))[i, j],
                    data["rms_mean"][i, j],
                    data.get("rms_std", np.zeros_like(data["rms_mean"]))[i, j],
                ])

    if "critical_ac" in data:
        with open(out_dir / "critical_ac_curve.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["omega_rad_s", "critical_ac_mm", "uncertainty_mm", "mc_std_mm"])
            for om, ac_crit, unc, mcs in zip(
                og, data["critical_ac"], data["critical_ac_uncertainty"],
                data.get("critical_ac_std", np.zeros(len(og))),
            ):
                w.writerow([om, ac_crit, unc, mcs])

    report = {
        "map_type": data["map_type"],
        "stability_criterion": (
            "Monte Carlo classification per grid point: stable / bounded / "
            "chatter-like / failed using sensor vibration, Delta_f, force-clip %, "
            "finite-state check, and post-transient RMS growth (Test2 philosophy). "
            "Lobe boundary: critical ac where P(chatter-like) crosses threshold."
        ),
        "stochastic_sources": (
            "Episode geometry/sensor/pass-line resampling via plant.reset(seed); "
            "enable_geometry_uncertainty, enable_sensor_uncertainty, "
            "enable_process_noise, pass_sampling=random from env config."
        ),
        "t_final_s": data.get("t_final_s"),
        "threshold": data.get("threshold"),
    }
    if "start_position" in data:
        report["start_position_sensitivity"] = data["start_position"]
    if "monotonicity_violations" in data:
        report["monotonicity_violations_ac_scan"] = data["monotonicity_violations"]

    with open(out_dir / "stability_lobe_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main() -> int:
    available = ", ".join(list_configs()) or "conf_fast"
    parser = argparse.ArgumentParser(description="Regenerative milling stability-lobe analysis")
    parser.add_argument("--config", default="conf_fast", help=f"Config ({available})")
    parser.add_argument(
        "--controller", nargs="+", default=None,
        choices=["uncontrolled", "trained"],
    )
    parser.add_argument("--model", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_base = Path(cfg["stability_dir"])
    modes = args.controller or cfg["stability_lobe"].get("controller_modes", ["uncontrolled"])

    model_path = args.model
    if model_path is None and "trained" in modes:
        seed = cfg["seeds"][0]
        model_path = Path(cfg["model_dir"]) / f"final_{seed}.zip"

    for mode in modes:
        print(f"[stability] mode={mode} config={args.config}")
        if mode == "uncontrolled":
            data = run_uncontrolled_lobe(cfg)
            sub = out_base / "uncontrolled"
            _save_outputs(data, sub)
            _plot_uncontrolled(data, sub)
            sp = data.get("start_position", {})
            print(f"  critical_ac range: {np.nanmin(data['critical_ac']):.2f}-"
                  f"{np.nanmax(data['critical_ac']):.2f} mm")
            print(f"  start-position 3D lobe recommended: {sp.get('recommend_3d_lobe', False)}")
        else:
            data = run_trained_performance_map(cfg, model_path)
            sub = out_base / "trained"
            _save_outputs(data, sub)
            _plot_trained(data, sub)
        print(f"  saved -> {sub}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
