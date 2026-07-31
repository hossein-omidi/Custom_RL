"""Compare the MPC controller against the trained PPO (RL) policy.

Both controllers are evaluated on the *same* face-milling plate environment and
write trajectories in the same JSON schema:

    RL  : ``scripts/eval_policy.py``        -> eval_trajectories/trajectories_seed{N}.json
    MPC : ``scripts/mpc_face_milling.py``   -> mpc_trajectories/trajectories_seed{N}.json

This script loads both sets, matches episodes, and overlays the key machining
signals so the RL policy can be verified against the model-based optimal
controller:

    * plate displacement max|w(t)|         (with the safety limit)
    * spindle speed  [rpm]
    * axial depth of cut ap  [mm]
    * axial cutting force Fz  [N]
    * cumulative reward (return)
    * feed progress along the pass

plus a summary bar chart (return, max|w|, pass completion, feed progress).

Usage
-----
    python scripts/compare_mpc_rl.py \
        --rl-dir eval_trajectories \
        --mpc-dir mpc_trajectories \
        --out-dir plots/mpc_vs_rl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RL_COLOR = "#1f77b4"     # blue
MPC_COLOR = "#d62728"    # red
RL_STYLE = dict(color=RL_COLOR, linestyle="-", linewidth=1.5)
MPC_STYLE = dict(color=MPC_COLOR, linestyle="--", linewidth=2.0)
PASS_REASONS = {"pass_completed_90percent", "pass_completed"}


# ---------------------------------------------------------------------------
# loading / schema extraction (robust to both eval_policy and MPC records)
# ---------------------------------------------------------------------------
def _load_seed_files(directory: Path) -> dict[int, list[dict[str, Any]]]:
    """Return {seed: [episode records]} from trajectories_seed*.json in a dir."""
    directory = Path(directory)
    out: dict[int, list[dict[str, Any]]] = {}
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("trajectories_seed*.json")):
        stem = path.stem.replace("trajectories_seed", "")
        try:
            seed = int(stem)
        except ValueError:
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        out[seed] = data
    return out


def _primary(ep: dict[str, Any]) -> dict[str, Any]:
    """Return the primary rollout: first MC run if present, else the record."""
    runs = ep.get("runs")
    if runs:
        return runs[0]
    return ep


def _series(ep: dict[str, Any]) -> dict[str, np.ndarray]:
    """Extract aligned time series from an episode record (RL or MPC)."""
    rec = _primary(ep)
    times = np.asarray(rec.get("times", []), dtype=np.float64)

    w = np.asarray(rec.get("w_sensor", []), dtype=np.float64)
    if w.ndim == 2 and w.size:
        max_abs_w = np.max(np.abs(w), axis=1)
    else:
        max_abs_w = np.full(times.shape, np.nan)

    process = rec.get("process", {}) or {}

    def proc(key: str) -> np.ndarray:
        vals = process.get(key, [])
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size == 0:
            return np.full(times.shape, np.nan)
        return arr

    rewards = np.asarray(rec.get("rewards", []), dtype=np.float64)
    cum_reward = np.cumsum(rewards) if rewards.size else np.full(times.shape, np.nan)

    n = times.shape[0] if times.size else max_abs_w.shape[0]

    def fit(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
        if arr.shape[0] == n:
            return arr
        out = np.full(n, np.nan)
        m = min(n, arr.shape[0])
        out[:m] = arr[:m]
        return out

    if not times.size:
        times = np.arange(n, dtype=np.float64)

    return {
        "times": times,
        "max_abs_w": fit(max_abs_w),
        "omega_rpm": fit(proc("omega_rpm")),
        "ap_mm": fit(proc("ap_mm")),
        "Fz_N": fit(proc("Fz_N")),
        "F_mag_N": fit(proc("F_mag_N")),
        "feed_progress": fit(proc("feed_progress")),
        "cum_reward": fit(cum_reward),
    }


def _meta(ep: dict[str, Any]) -> dict[str, Any]:
    return ep.get("metadata", {}) or {}


def _summary(ep: dict[str, Any]) -> dict[str, Any]:
    rec = _primary(ep)
    s = rec.get("summary", {}) or {}
    reason = rec.get("termination_reason", s.get("termination_reason"))
    w = np.asarray(rec.get("w_sensor", []), dtype=np.float64)
    max_abs_w = float(np.max(np.abs(w))) if w.size else float(s.get("max_abs_w_m", np.nan))
    process = rec.get("process", {}) or {}
    fp = process.get("feed_progress", [])
    final_fp = float(fp[-1]) if fp else float("nan")
    return {
        "return": float(rec.get("return", s.get("return", np.nan))),
        "max_abs_w_m": max_abs_w,
        "pass_completed": bool(rec.get("pass_completed", reason in PASS_REASONS)),
        "final_feed_progress": final_fp,
        "length": int(rec.get("length", 0)),
        "termination_reason": reason,
    }


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def _plot_episode_overlay(
    rl_ep: dict[str, Any] | None,
    mpc_ep: dict[str, Any] | None,
    *,
    out_path: Path,
    title: str,
) -> None:
    """Overlay RL vs MPC time-series for one matched episode."""
    panels = [
        ("max_abs_w", "max |w(t)|  [m]", True),
        ("omega_rpm", "Spindle speed  [rpm]", False),
        ("ap_mm", "Axial depth ap  [mm]", False),
        ("Fz_N", "Axial force Fz  [N]", False),
        ("cum_reward", "Cumulative reward", False),
        ("feed_progress", "Feed progress", False),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    axes = axes.reshape(-1)

    w_limit = None
    for ep in (rl_ep, mpc_ep):
        if ep is not None:
            w_limit = _meta(ep).get("w_limit", _meta(ep).get("w_limit_m"))
            if w_limit is not None:
                break

    # (series, style, end-label)  RL solid blue, MPC dashed red
    rl_ser = _series(rl_ep) if rl_ep is not None else None
    mpc_ser = _series(mpc_ep) if mpc_ep is not None else None
    rl_lbl = _controller_label("RL", rl_ep)
    mpc_lbl = _controller_label("MPC", mpc_ep)
    series = [
        (rl_ser, RL_STYLE, rl_lbl),
        (mpc_ser, MPC_STYLE, mpc_lbl),
    ]

    for idx, (key, ylabel, is_w) in enumerate(panels):
        ax = axes[idx]
        for ser, style, label in series:
            if ser is None:
                continue
            t = ser["times"]
            y = ser[key]
            good = np.isfinite(t) & np.isfinite(y)
            if np.count_nonzero(good) >= 1:
                ax.plot(t[good], y[good], label=label, **style)
        if is_w and w_limit is not None and np.isfinite(float(w_limit)):
            ax.axhline(float(w_limit), ls=":", lw=1.2, color="k", alpha=0.7,
                       label=f"w_limit={float(w_limit):g} m")
            ax.set_ylim(bottom=0.0)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="best", fontsize=9, framealpha=0.9)

    axes[-1].set_xlabel("Time  [s]")
    axes[-2].set_xlabel("Time  [s]")
    fig.suptitle(title + "     (RL: solid blue   |   MPC: dashed red)", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _controller_label(name: str, ep: dict[str, Any] | None) -> str:
    """Legend label annotated with return / pass-status / termination reason."""
    if ep is None:
        return f"{name} (absent)"
    s = _summary(ep)
    tag = "pass" if s["pass_completed"] else (s["termination_reason"] or "cap")
    return f"{name}  (R={s['return']:.0f}, {tag})"


def _plot_summary_bars(
    rows: list[dict[str, Any]],
    *,
    out_path: Path,
) -> None:
    """Grouped bar chart comparing RL vs MPC summary metrics across episodes."""
    if not rows:
        return
    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))
    width = 0.38

    metrics = [
        ("return", "Return", False),
        ("max_abs_w_m", "max |w|  [m]", True),
        ("final_feed_progress", "Final feed progress", False),
        ("pass_completed", "Pass completed", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.reshape(-1)

    for idx, (key, title, hline_w) in enumerate(metrics):
        ax = axes[idx]
        rl_vals = [float(r["rl"].get(key, np.nan)) if r["rl"] else np.nan for r in rows]
        mpc_vals = [float(r["mpc"].get(key, np.nan)) if r["mpc"] else np.nan for r in rows]
        ax.bar(x - width / 2, rl_vals, width, label="RL", color=RL_COLOR)
        ax.bar(x + width / 2, mpc_vals, width, label="MPC", color=MPC_COLOR)
        if hline_w:
            w_limit = rows[0].get("w_limit")
            if w_limit is not None and np.isfinite(float(w_limit)):
                ax.axhline(float(w_limit), ls="--", color="k", alpha=0.6, label="w_limit")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        if idx == 0:
            ax.legend(loc="best", fontsize=9)

    fig.suptitle("MPC vs RL — episode summary", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Compare MPC and RL controllers.")
    parser.add_argument("--rl-dir", default="eval_trajectories",
                        help="Directory with RL trajectories from eval_policy.py")
    parser.add_argument("--mpc-dir", default="mpc_trajectories",
                        help="Directory with MPC trajectories from mpc_face_milling.py")
    parser.add_argument("--out-dir", default="plots/mpc_vs_rl")
    parser.add_argument("--seeds", nargs="*", type=int, default=None,
                        help="Seeds to compare (default: intersection of both dirs).")
    args = parser.parse_args()

    rl = _load_seed_files(Path(args.rl_dir))
    mpc = _load_seed_files(Path(args.mpc_dir))
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"RL  seeds : {sorted(rl)} ({args.rl_dir})")
    print(f"MPC seeds : {sorted(mpc)} ({args.mpc_dir})")

    if not rl and not mpc:
        print("No trajectories found in either directory. Run eval_policy.py and "
              "mpc_face_milling.py first.")
        return 1

    if args.seeds:
        seeds = list(args.seeds)
    else:
        seeds = sorted(set(rl) | set(mpc))

    summary_rows: list[dict[str, Any]] = []
    n_plots = 0
    for seed in seeds:
        rl_eps = rl.get(seed, [])
        mpc_eps = mpc.get(seed, [])
        n_ep = max(len(rl_eps), len(mpc_eps))
        for ep_idx in range(n_ep):
            rl_ep = rl_eps[ep_idx] if ep_idx < len(rl_eps) else None
            mpc_ep = mpc_eps[ep_idx] if ep_idx < len(mpc_eps) else None
            if rl_ep is None and mpc_ep is None:
                continue

            title = f"seed {seed}  episode {ep_idx}"
            ref = rl_ep if rl_ep is not None else mpc_ep
            y_line = _primary(ref).get("y_line_m")
            if y_line is not None and np.isfinite(float(y_line)):
                title += f"   (milling line y={float(y_line):.3f} m)"

            out_path = out_dir / f"compare_seed{seed}_ep{ep_idx}.png"
            _plot_episode_overlay(rl_ep, mpc_ep, out_path=out_path, title=title)
            n_plots += 1

            w_limit = None
            for ep in (rl_ep, mpc_ep):
                if ep is not None:
                    w_limit = _meta(ep).get("w_limit", _meta(ep).get("w_limit_m"))
                    if w_limit is not None:
                        break
            summary_rows.append({
                "label": f"s{seed}e{ep_idx}",
                "rl": _summary(rl_ep) if rl_ep is not None else None,
                "mpc": _summary(mpc_ep) if mpc_ep is not None else None,
                "w_limit": w_limit,
            })

    if summary_rows:
        _plot_summary_bars(summary_rows, out_path=out_dir / "summary_bars.png")

    # text summary table
    print("\n=== Summary (RL vs MPC) ===")
    header = f"{'episode':>10} | {'controller':>10} | {'return':>10} | {'max|w| m':>10} | {'pass':>5} | {'feed':>6} | reason"
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        for ctrl in ("rl", "mpc"):
            s = row[ctrl]
            if s is None:
                continue
            print(f"{row['label']:>10} | {ctrl.upper():>10} | {s['return']:>10.1f} | "
                  f"{s['max_abs_w_m']:>10.2e} | {str(s['pass_completed']):>5} | "
                  f"{s['final_feed_progress']:>6.3f} | {s['termination_reason']}")

    print(f"\nSaved {n_plots} episode overlay(s) and a summary bar chart to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
