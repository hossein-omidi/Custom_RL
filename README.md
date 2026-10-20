# Research-grade RL vs MPC paired evaluation patch

## Scope

This package corrects **evaluation and reporting only**, not RL training, plant equations, MPC nonlinear optimization, or MHE mathematics. It was designed against branch `mpc-correction` of `hossein-omidi/Custom_RL` and checked against the previously supplied `mpc_mhe_regenerative_history_fix.zip` controller. It fails safely if an expected source pattern has changed.

The package includes:

- `scripts/compare_mpc_rl.py`: complete replacement comparator with **strict** scenario checks, physically integrated nominal removed-volume proxy, time-weighted vibration RMS, final-feed fallback, sensor peak, input variation, no ranking by raw cumulative reward, and explicit *unmatched* exploratory plotting.
- `tools/install_fair_comparison.py`: idempotent installer for small edits in the RL and MPC runners, plus backup of the old comparator. It leaves all equations unchanged.
- `tests/test_fair_comparison.py`: standalone synthetic regression tests (NumPy, Matplotlib, standard library).

## Main problems corrected

1. **Unmatched plant step and safety-check frequency.** Original RL evaluation uses `dt=1e-4, n_substeps=10` (1 ms environment step), but phase MPC typically uses `dt≈9.4e-5, n_substeps=1` (~0.094 ms step). The plant tests termination once per environment step. A defensible evaluation evaluates *both* at `dt=1e-4, n_substeps=1` (0.1 ms safety checks), while the RL action is held for **10 environment steps** (original 1 ms policy decision cadence). The MPC remains phase-triggered; different controller decision intervals are allowed, but reported.
2. **Different disturbance defaults.** RL evaluation uses `dynamics_uncertainty_std=0.001`, MPC defaults to `0`; run deterministic paired tests with `0` first. The new metadata captures initial-state std and observation/process noise as well.
3. **Unmatched episode seeds and milling lines.** Same filename seed does **not** mean same env reset seed. New `--reset-seed-offset` flag lets both runners reset e.g. seed=1000 when file seed=0. Use the same `--y0` and `--no-randomize-y0` for both. Comparator checks actual reset seed and line.
4. **Raw rewards cannot be ranked.** Dense reward is accrued every environment step and is not scaled by elapsed physical time. Raw sums are saved **only as a labeled diagnostic**, never as a comparable performance metric. Instead the comparator uses measured safety and physical-time metrics, including a *nominal* removed-volume proxy (`a_e * a_p * f_t * N * Omega/(2*pi)` integrated in seconds). This proxy is not an exact measurement of chip-removal volume under partial engagement or chatter.
5. **Failure progress was often NaN.** Comparator reconstructs terminal progress from cutter position or normalized cutter position observation when the plant omits terminal info.
6. **Misleading time/progress labels.** MPC console distinguishes wall-clock time from physical simulation time and explicitly names the historical maximum displacement so far.
7. **Plot mislabeled scenario.** Comparator verifies the actual y-line (and other setup) *for both* records and refuses to generate apparently paired plots for mismatches unless `--allow-mismatch` is explicitly set. Plots are marked exploratory if unmatched.

## Installation (PowerShell from your Custom_RL root)

Unzip the package to a temporary directory, for example `fair_patch` in your repository root.

```powershell
python fair_patch/tools/install_fair_comparison.py --repo . --dry-run
python fair_patch/tools/install_fair_comparison.py --repo .
python -m unittest discover -s fair_patch/tests -v
```

Installer saves `*.before_fair_eval.bak` alongside the three edited files. Make a git commit first, or back up the project. Rerunning the installer detects its own marker and refuses to apply twice. If source versions differ, the dry run fails rather than guessing.

## Matched deterministic evaluation (seed 0, milling line 0.519 m)

Run BOTH controllers at the same plant RK4 and environment step of 0.1 ms. This changes the **RL evaluation** grid, *not RL training or the already learned policy*. PPO still chooses one action every 1 ms via `--policy-hold-env-steps 10`.

```powershell
python scripts/eval_policy.py --seeds 0 --n-episodes 1 --n-mc 1 --dt 0.0001 --n-substeps 1 --policy-hold-env-steps 10 --reset-seed-offset 1000 --reward productive --dynamics-uncertainty-std 0 --y0 0.519 --no-randomize-y0 --max-episode-steps 2200000 --out-dir eval_trajectories_paired

python scripts/MPC/mpc_face_milling_chatgpt_V3.py --seeds 0 --n-episodes 1 --dt 0.0001 --n-substeps 1 --reset-seed-offset 1000 --reward productive --dynamics-uncertainty-std 0 --y0 0.519 --no-randomize-y0 --max-env-steps 2200000 --max-episode-steps 2200000 --out-dir mpc_trajectories_paired

python scripts/compare_mpc_rl.py --rl-dir eval_trajectories_paired --mpc-dir mpc_trajectories_paired --out-dir plots/paired_comparison
```

The full-pass MPC may require many wall-clock hours. First smoke-test with short (e.g. 2,000 step) budgets, but **do not compare partial-run returns or claim pass performance from smoke tests**. Two identical budget caps do not imply equal final physical time if an episode terminates early.

For more robust deterministic initialization, both scripts also support `--plant-override` (MPC as already implemented, RL restricted to initial displacement / velocity standard deviations and noise parameters). If using an override, apply *identical relevant initial/noise settings* on both. Quoting JSON arguments varies by PowerShell version; avoiding overrides is simpler when seed and initial parameters already match.

## Scientific interpretation

- This **does not correct** an actual `excessive_sensor_displacement` plant shutdown. If it remains under matched conditions, repair controller prediction, estimator/history, constraints, and command execution separately.
- The MPC and RL decision times need not match; they are different algorithms. The physical integration, safety checks, task, and uncertainty conditions *do* need to match. The report makes their cadence distinction explicit.
- Under stochastic uncertainty, identical seeds only help pairing if random-number draws correspond. Validate random-stream synchronization; otherwise compare distributions across a predeclared set of seeds and use confidence intervals. The recommended deterministic baseline avoids this issue.
- Comparing performance across **unequal machining durations** using cumulative reward is fundamentally misleading. Use pass completion, safety, time, nominal removed volume and actual actuator variation. Incomplete runs must be identified and not ranked as if they finished.
- The phase-domain MPC approximations, three-mode MHE, and original 3 mm sensor termination remain unchanged. Neither MPC feasibility nor global chatter stability is certified by this comparison fix.
- A full fairness study should also verify initial physical modal states (or use identical zero initial conditions), hardware/actuator model equivalence, and numerical convergence when checking transient displacement peaks.

## Verification

`python -m unittest discover -s fair_patch/tests -v` checks scenario mismatches, matched metadata, units/time integration, progress fallback, strict vs exploratory plotting, and fail-closed installation. Syntax checks pass for the new files. The installer modifications were syntax-checked against the previously supplied MPC history-repair controller. Full Gym/do-mpc/RL end-to-end runs require your configured environment and **have not been performed here**.
