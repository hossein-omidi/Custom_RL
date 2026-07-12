"""One-shot pipeline: stability lobe -> train (both modes) -> MPC -> RL-vs-MPC compare.

Runs the whole study in order by invoking the project's CLIs as subprocesses,
each writing to its own directory.  Ready to run:

    python scripts/run_pipeline.py                 # full study (hours; do it locally)
    python scripts/run_pipeline.py --quick         # fast smoke test of every stage
    python scripts/run_pipeline.py --stages train mpc compare
    python scripts/run_pipeline.py --skip lobe     # everything except the lobe diagram

Stages (run in this order): lobe, train, eval, mpc, compare.
Both control modes are handled:
    first  mode  = CustomODEPlate-v0        (action [omega, ap])   -> models/ppo_plate
    second mode  = CustomODEPlateFinish-v0  (action [omega], ap fixed) -> models/ppo_plate_finish
Each RL-vs-MPC comparison uses the model of its own mode.

The whole study is stochastic: training, RL evaluation, MPC closed loop, and the
comparison all use the same modal-acceleration disturbance (--uncertainty) and a
randomized milling line during training; the comparison is run on one fixed
milling line so RL and MPC face the same nominal conditions.

Output layout
-------------
    plots/stability_lobe/{y*,surface3d,stochastic}/   stability lobe diagrams
    models/ppo_plate/            logs/ppo_plate/       first-mode  RL
    models/ppo_plate_finish/     logs/ppo_plate_finish/second-mode RL
    results/rl_first/  results/rl_finish/              RL trajectories (per mode)
    results/mpc_first/ results/mpc_finish/             MPC trajectories (per mode)
    plots/compare_first/  plots/compare_finish/        RL-vs-MPC comparison figures
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = [sys.executable, "-u"]

# --------------------------------------------------------------------------
# directory layout (all relative to the repo root)
# --------------------------------------------------------------------------
LOBE_DIR = "plots/stability_lobe"
MODELS_FIRST, LOGS_FIRST = "models/ppo_plate", "logs/ppo_plate"
MODELS_FIN, LOGS_FIN = "models/ppo_plate_finish", "logs/ppo_plate_finish"
RL_FIRST, RL_FIN = "results/rl_first", "results/rl_finish"
MPC_FIRST, MPC_FIN = "results/mpc_first", "results/mpc_finish"
CMP_FIRST, CMP_FIN = "plots/compare_first", "plots/compare_finish"


def sh(script: str, *args: str) -> None:
    """Run `python -u scripts/<script> <args>` from the repo root, echoing it."""
    cmd = PY + [str(REPO / "scripts" / script), *map(str, args)]
    print("\n" + "=" * 90)
    print(">>>", " ".join(cmd))
    print("=" * 90, flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(REPO))
    dt = time.time() - t0
    if r.returncode != 0:
        raise SystemExit(f"[pipeline] stage command failed (exit {r.returncode}) after {dt:.0f}s:\n  {' '.join(cmd)}")
    print(f"[pipeline] ok ({dt:.0f}s)")


# ==========================================================================
# stages
# ==========================================================================
def stage_lobe(a):
    """Stability lobe: 2D per milling line, 3D surface, and stochastic (MC)."""
    # Cap the per-trial simulation length: regenerative chatter always develops
    # within a few hundred control steps, so a few-thousand-step cap gives the
    # same stability boundary while avoiding the ~70k-step passes at low rpm.
    common = ["--rpm-min", a.rpm_min, "--rpm-max", a.rpm_max, "--ap-min", 0.0,
              "--ap-max", a.lobe_ap_max, "--ap-tol", a.lobe_ap_tol,
              "--binary-iters", a.lobe_bin_iters, "--dt", a.dt, "--n-substeps", a.n_substeps,
              "--max-sim-steps", a.lobe_max_steps, "--max-episode-steps", a.lobe_max_steps]
    # 2D deterministic lobe at several milling lines y0 (shows y-dependence)
    for y in a.lobe_y_lines:
        sh("stability_lobe_new.py", *common, "--rpm-points", a.lobe_rpm_pts_2d,
           "--line-y", y, "--out-dir", f"{LOBE_DIR}/y{y}")
    # 3D surface ap = f(rpm, milling-line-y)
    sh("stability_lobe_new.py", *common, "--surface-3d",
       "--rpm-points", a.lobe_rpm_pts_3d, "--y-min", a.lobe_y_min, "--y-max", a.lobe_y_max,
       "--y-points", a.lobe_y_points, "--out-dir", f"{LOBE_DIR}/surface3d")
    # 2D stochastic (Monte Carlo) lobe over milling lines
    sh("stability_lobe_new.py", *common, "--stochastic-lobe",
       "--rpm-points", a.lobe_rpm_pts_3d, "--n-mc", a.lobe_n_mc,
       "--dynamics-uncertainty-std", a.uncertainty,
       "--y-position", *[str(y) for y in a.lobe_y_lines],
       "--out-dir", f"{LOBE_DIR}/stochastic")


def stage_train(a):
    """Train both control modes into separate directories (stochastic plant)."""
    # first mode: action [omega, ap]
    sh("train_sb3_ppo.py", "--env-id", "CustomODEPlate-v0", "--seeds", *map(str, a.seeds),
       "--reward", "dense", "--total-timesteps", a.timesteps_first,
       "--n-envs", 1, "--vec-env", "dummy", "--dt", a.dt, "--n-substeps", a.n_substeps,
       "--max-episode-steps", a.train_max_steps, "--n-eval-episodes", 2, "--eval-freq", 20000,
       "--dynamics-uncertainty-std", a.uncertainty, "--randomize-y0",
       "--save-dir", MODELS_FIRST, "--log-dir", LOGS_FIRST)
    # second mode: action [omega], ap fixed & randomized over a FEASIBLE range
    sh("train_finish.py", "--seeds", *map(str, a.seeds),
       "--total-timesteps", a.timesteps_second, "--ap-max", a.finish_train_ap_max,
       "--dt", a.dt, "--n-substeps", a.n_substeps, "--max-episode-steps", a.train_max_steps,
       "--dynamics-uncertainty-std", a.uncertainty, "--randomize-y0",
       "--save-dir", MODELS_FIN, "--log-dir", LOGS_FIN)


def stage_eval(a):
    """Evaluate each trained RL policy on the comparison milling line (stochastic)."""
    seed = a.seeds[0]
    sh("eval_policy.py", "--env-id", "CustomODEPlate-v0", "--seeds", seed, "--reward", "dense",
       "--n-episodes", 1, "--dt", a.dt, "--n-substeps", a.n_substeps,
       "--max-episode-steps", a.cmp_steps, "--y0", a.cmp_y0,
       "--dynamics-uncertainty-std", a.uncertainty,
       "--model-dir", MODELS_FIRST, "--out-dir", RL_FIRST)
    sh("eval_policy.py", "--env-id", "CustomODEPlateFinish-v0", "--seeds", seed, "--reward", "dense",
       "--n-episodes", 1, "--dt", a.dt, "--n-substeps", a.n_substeps,
       "--max-episode-steps", a.cmp_steps, "--y0", a.cmp_y0, "--ap", a.cmp_ap,
       "--dynamics-uncertainty-std", a.uncertainty,
       "--model-dir", MODELS_FIN, "--out-dir", RL_FIN)


def stage_mpc(a):
    """Run the MPC closed loop for each mode on the comparison milling line (stochastic)."""
    seed = a.seeds[0]
    sh("mpc_face_milling.py", "--env-id", "CustomODEPlate-v0", "--seeds", seed, "--n-episodes", 1,
       "--dt", a.dt, "--n-substeps", a.n_substeps, "--y0", a.cmp_y0, "--max-steps", a.cmp_steps,
       "--delay-mode", "predictive", "--dynamics-uncertainty-std", a.uncertainty,
       "--horizon", a.mpc_horizon, "--control-hold", a.mpc_hold, "--out-dir", MPC_FIRST)
    sh("mpc_face_milling.py", "--env-id", "CustomODEPlateFinish-v0", "--seeds", seed, "--n-episodes", 1,
       "--dt", a.dt, "--n-substeps", a.n_substeps, "--y0", a.cmp_y0, "--ap", a.cmp_ap,
       "--max-steps", a.cmp_steps, "--delay-mode", "predictive",
       "--dynamics-uncertainty-std", a.uncertainty,
       "--horizon", a.mpc_horizon, "--control-hold", a.mpc_hold, "--out-dir", MPC_FIN)


def stage_compare(a):
    """Overlay RL vs MPC for each mode, each using its own model's trajectories."""
    sh("compare_mpc_rl.py", "--rl-dir", RL_FIRST, "--mpc-dir", MPC_FIRST, "--out-dir", CMP_FIRST)
    sh("compare_mpc_rl.py", "--rl-dir", RL_FIN, "--mpc-dir", MPC_FIN, "--out-dir", CMP_FIN)


STAGES = {"lobe": stage_lobe, "train": stage_train, "eval": stage_eval,
          "mpc": stage_mpc, "compare": stage_compare}
ORDER = ["lobe", "train", "eval", "mpc", "compare"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stages", nargs="+", choices=ORDER, default=None,
                   help="Subset of stages to run (default: all, in order).")
    p.add_argument("--skip", nargs="+", choices=ORDER, default=[], help="Stages to skip.")
    p.add_argument("--quick", action="store_true", help="Tiny settings to smoke-test every stage fast.")

    # shared physics / stochasticity
    p.add_argument("--dt", type=float, default=1e-4)
    p.add_argument("--n-substeps", type=int, default=10)
    p.add_argument("--uncertainty", type=float, default=0.001,
                   help="Modal-acceleration disturbance std (stochastic plant), used everywhere.")
    p.add_argument("--seeds", nargs="+", type=int, default=[0])

    # stability lobe (strong grid by default)
    p.add_argument("--rpm-min", type=float, default=1000.0)
    p.add_argument("--rpm-max", type=float, default=40000.0)
    p.add_argument("--lobe-ap-max", type=float, default=3.0)
    p.add_argument("--lobe-ap-tol", type=float, default=0.02)
    p.add_argument("--lobe-bin-iters", type=int, default=12)
    p.add_argument("--lobe-rpm-pts-2d", type=int, default=60)
    p.add_argument("--lobe-rpm-pts-3d", type=int, default=40)
    p.add_argument("--lobe-y-lines", nargs="+", type=float, default=[0.2, 0.5, 0.8])
    p.add_argument("--lobe-y-min", type=float, default=0.1)
    p.add_argument("--lobe-y-max", type=float, default=0.9)
    p.add_argument("--lobe-y-points", type=int, default=6)
    p.add_argument("--lobe-n-mc", type=int, default=5)
    p.add_argument("--lobe-max-steps", type=int, default=4000,
                   help="Cap on control steps per stability trial (chatter develops much sooner).")

    # training (efficient-but-sufficient)
    p.add_argument("--timesteps-first", type=int, default=400000)
    p.add_argument("--timesteps-second", type=int, default=300000)
    p.add_argument("--train-max-steps", type=int, default=2000)
    p.add_argument("--finish-train-ap-max", type=float, default=1.0)

    # comparison (RL vs MPC) settings
    p.add_argument("--cmp-y0", type=float, default=0.5, help="Milling line for the RL-vs-MPC comparison.")
    p.add_argument("--cmp-ap", type=float, default=0.5, help="Fixed finishing depth for second-mode comparison [mm].")
    p.add_argument("--cmp-steps", type=int, default=1200, help="Closed-loop length for eval/MPC.")
    p.add_argument("--mpc-horizon", type=int, default=20)
    p.add_argument("--mpc-hold", type=int, default=3)
    args = p.parse_args()

    if args.quick:   # tiny settings to verify the whole chain quickly
        args.lobe_rpm_pts_2d = 6; args.lobe_rpm_pts_3d = 5; args.lobe_y_points = 3
        args.lobe_bin_iters = 6; args.lobe_ap_tol = 0.1; args.lobe_n_mc = 2
        args.lobe_y_lines = [0.2, 0.5]; args.lobe_max_steps = 1200
        args.timesteps_first = 4096; args.timesteps_second = 4096
        args.cmp_steps = 90; args.train_max_steps = 500

    to_run = [s for s in (args.stages or ORDER) if s not in args.skip]
    print(f"[pipeline] stages: {to_run}")
    print(f"[pipeline] uncertainty(std)={args.uncertainty}  seeds={args.seeds}  "
          f"cmp_y0={args.cmp_y0}  cmp_ap={args.cmp_ap}  cmp_steps={args.cmp_steps}"
          + ("   [QUICK]" if args.quick else ""))
    t0 = time.time()
    for s in to_run:
        STAGES[s](args)
    print(f"\n[pipeline] ALL DONE in {time.time()-t0:.0f}s.")
    print("  lobe    -> plots/stability_lobe/")
    print("  models  -> models/ppo_plate , models/ppo_plate_finish")
    print("  compare -> plots/compare_first , plots/compare_finish")


if __name__ == "__main__":
    main()
