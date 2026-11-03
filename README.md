# PPO face-milling research pipeline — re-audited version 2

This package replaces three existing scripts under `scripts/` while keeping their CLI entrypoints and trajectory JSON structure. It revises **data integrity, statistical interpretation, runtime safety, and figure generation**. It does **not** change PPO network architecture, optimization hyperparameters, reward, the physical plant, or existing checkpoint formats.

## Contents

- `scripts/train_ppo_unified.py`: compatible trainer, optional shared initial callback evaluation seed; documented evaluation RNG.
- `scripts/eval_policy.py`: held-out evaluation and full-step physical outcome metrics, conservative output handling, configurable **storage sampling**, CSV exports.
- `scripts/plot_results.py`: learning-curve and TensorBoard diagnostics, physically time-aligned MC trajectories, original-sample rolling RMS, publication-resolution PNG + vector PDF, physical outcome figure.
- `tests/test_pipeline_contracts.py`: seven synthetic contract tests. They use mock objects and do **not** claim to validate milling physics.
- `RESEARCH_AUDIT.md`: scientific interpretation, remaining gaps, and acceptance protocol.

## Installation

Copy the three script files into the repository's existing `scripts/` folder, keeping your existing `custom_rl/` package and training data. Back up the old scripts first; never replace your physical plant with a synthetic mock. Run from the repository root, in your environment with Gymnasium, Stable-Baselines3, NumPy, Matplotlib and TensorBoard installed.

### Training: three independent PPO policies

```bash
python scripts/train_ppo_unified.py --seeds 0 1 2
```

This preserves the historical evaluation callback seed rule (`training_seed + 10000`) unless you **explicitly** select a common initial evaluation seed:

```bash
python scripts/train_ppo_unified.py --seeds 0 1 2 --callback-eval-seed 10000
```

The shared seed improves initial comparison conditions but **does not** guarantee identical stochastic disturbances or test cases at every training checkpoint. Unlike fixed trial schedules, EvalCallback advances its environment RNG while episodes run; the path of the RNG can depend on policy behavior and episode length. Checkpoint-selection reward is only a selection criterion, **not** a held-out performance estimate.

For a **finishing-mode** training run, include `--env-id CustomODEPlateFinish-v0` (and optionally `--ap-max ...`). In finishing mode PPO controls **spindle speed only**; the physical command vector includes a fixed per-episode depth to drive the simulator, but that depth is **not** an actor output.

### Held-out physical evaluation

```bash
python scripts/eval_policy.py \
  --model-dir "runs/ppo/<run>/models" \
  --out-dir "runs/ppo/<run>/evaluation" \
  --seeds 0 1 2 \
  --n-episodes 5 --n-mc 5
```

`--n-episodes 5 --n-mc 5` means **five physical scenarios**, with five stochastic replicas per scenario, **per training policy**. The replicas are not 25 independent physical scenarios. The evaluation uses the same scenario index, `y0`, finishing `ap`, and reset-seed sequence across training seeds, verifies saved evaluation settings, and refuses to combine seeds whose training/plant settings differ. Outputs are staged until evaluation succeeds; existing results are not deleted before replacement. Use `--overwrite` deliberately if output files already exist.

**Potentially very large output:** In the worst case, `5 × 5 × 250000 = 6.25 million` simulation steps **per policy**. Storing every trajectory channel as JSON can use enormous memory/disk space. For a broad robustness batch where only aggregate physical metrics matter, choose e.g. `--record-every 20` to save each twentieth trajectory sample. The **return, episode length, physical displacement RMS/peak, force RMS/peak and action total variation are computed over all simulation steps**, not the saved subset. **Subsampled trajectories and plots MUST NOT be used for frequency spectra, chatter-frequency evidence, or transient-peak plots.** For detailed representative waveforms or spectral analysis, use `--record-every 1` on a smaller selection of scenarios; the environment must sample physical sensors sufficiently fast for the frequencies of interest.

Note: Full-step metrics are exact relative to the simulator **output cadence**, not a validation that the simulator resolves all physical modes or regenerative delays. The reference default environment timestep is 1 ms (RK4 integration substep 0.1 ms × 10), implying a nominal **500 Hz Nyquist limit for saved 1-ms sensor signals**. Higher-frequency chatter cannot be assessed reliably from those saved samples; a finer observation/recording cadence and a numerical-convergence study are needed.

### Generate figures

```bash
python scripts/plot_results.py \
  --log-dir "runs/ppo/<run>/logs" \
  --traj-dir "runs/ppo/<run>/evaluation" \
  --out-dir "runs/ppo/<run>/plots" \
  --seeds 0 1 2
```

**Main research figures**:

- `figure1_ppo_learning_curve.pdf/.png`: each seed's mean EvalCallback episode return against actual global environment timesteps, individual episode-return points, descriptive *across-seed sample* SD band (not a confidence interval).
- `ppo_training_diagnostics.pdf/.png`: rolling training episode return, explained variance, approximate KL + clip fraction, policy standard deviation (or clearly named entropy-loss fallback). These diagnose training but do **not** prove chatter suppression.
- `figure2_ppo_physical_outcomes.pdf/.png`: MC-mean physical RMS displacement, MC-mean peak absolute displacement, final feed progress and simulated pass completion. Each point is one outer scenario. Error bars, when present, are ±1 **within-scenario MC sample SD**, not 95% CIs. All seeds are visually offset by scenario rather than falsely connected as a continuous curve. Verify pairing status.
- `trajectory_displacement_rms_seed*_episode*.pdf/.png`: trailing **50 ms** RMS displacement computed from the original saved physical sensor samples **before visualization decimation**, then MC-averaged. Skipped when physical sampling is too coarse or episodes are too short. The 50-ms window is illustrative; justify/adapt it for your system dynamics.
- `trajectory_physical_sensor_response_*`, `trajectory_actions_*`, `trajectory_process_signals_*`: scenario-specific time histories and MC descriptive spreads. Raw oscillatory trajectories can cancel in the mean; inspect the RMS envelope and scalar metrics as well.

All figures export as **300 dpi PNG and vector PDF**; the graphing step may coarsen a large time grid purely for display. Never use such a coarsened time trace as a PSD estimate. Action curves use **zero-order hold** and are aligned to action-application times. Sensors/process signals use the physical post-step time. Values after early termination are **not extrapolated**; uncertainty is not falsely shown where fewer than two MC realizations exist. Finishing fixed depth is explicitly labeled **not a PPO-controlled action**.

**Numerical CSVs**:

- `figure1_ppo_learning_curve_data.csv`: individual evaluation episodes and mean per seed/checkpoint.
- `figure1_ppo_learning_curve_across_seed_stats.csv`: aligned descriptive seed statistics.
- `physical_outcomes_replicas.csv`: one row per *actual MC rollout* with complete physical metrics; includes realized reset seed, physical scenario, feed progress, returns, duration, displacement, cutting-force magnitudes, control variation and inference-time diagnostics.
- `physical_outcomes_scenarios.csv`: one row per *outer scenario and trained policy*, with mean and descriptive sample SD over its MC replicas. This is the correct observational unit for a scenario-level analysis.

## Validations and known scientific limitations

- **Done:** Python syntax check, seven synthetic regression tests including mocked roughing-mode end-to-end evaluator, scenario consistency checks, physical-time interpolation, zero-order-hold action semantics, exact full-step physical summaries when trajectories are sampled, CSV schema, and new PNG/PDF plots. A three-seed synthetic Figure 1 and physical outcomes figure were rendered to PDF and inspected visually. The previews are **synthetic**, not research findings.
- **Source-level checked:** `custom_rl/envs/ode_control_env.py`, `custom_rl/plants/plate.py`, `custom_rl/eval/pipeline.py`, and `custom_rl/envs/registration.py` on the `hossein-omidi/Custom_RL` `mpc-correction` branch. No claim is made that these are the precise versions on your local computer.
- **Not done:** real full PPO training, loading your genuine trained checkpoints, reading your genuine TensorBoard event files, running real plant rollouts or checking full NMPC+MHE pairing. The complete project runtime and data were not available in the sandbox.
- **Research caveat:** Before claiming PPO improves chatter, include consistent numerical integration/convergence, physical force/displacement/PSD where adequately sampled, full simulated pass completion, and appropriately matched fixed-policy and NMPC+MHE controls. The simulator's force/displacement are **simulated outputs**, not lab measurements or surface roughness (`Ra`).

