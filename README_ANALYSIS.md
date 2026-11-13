# PPO sweep scientific plotting and selection — reviewed statistical add-on (v5)

**Scope:** Additive post-processing only. The package does not train PPO, change rewards, evaluate physical trajectories, change `LATEST`, or modify the NMPC/MHE plant. Only replace the two scripts listed below if you previously installed older versions of this add-on. Do not replace your repository's `experiment.py`, `train_ppo_unified.py`, `eval_policy.py`, or `plot_results.py`.

## Install

Unzip into the root of your `Custom_RL` repository, preserving paths:

```
scripts/analyze_ppo_sweep.py
scripts/plot_ppo_sweep.py
tests/test_sweep_analysis.py
README_ANALYSIS.md
```

Back up any local custom changes to `scripts/plot_ppo_sweep.py` before replacement. Python dependencies for plotting: `numpy`, `matplotlib`. Reading logged PPO optimization metrics requires `tensorboard` (`python -m pip install tensorboard` in the same active environment used for the commands).

## Minimal commands — no paths needed

From the repository root, after **all three training configurations have completed**:

```powershell
python scripts/analyze_ppo_sweep.py plot --latest --mode roughing
python scripts/analyze_ppo_sweep.py plot --latest --mode finishing
```

Each command identifies the most recent **complete** three-profile sweep group in `runs/ppo_sweep/<mode>/`, prints its exact Fast/Balanced/Precise run directories, and produces aggregated figures under that sweep group's `plots/` directory. An incomplete newer sweep is ignored; completed runs from different sweep groups are **never mixed**. Sweep directory timestamps take precedence over copied/retouched file modification times.

If only one operating mode has complete sweep results, `--latest` alone also works. If **both** have completed sweeps, `--mode` is required for safety.

Check which runs would be used **without plotting**:

```powershell
python scripts/analyze_ppo_sweep.py plot --latest --mode roughing --resolve-only
```

Request all essential optimization diagnostics (fail clearly if unavailable or incomplete):

```powershell
python scripts/analyze_ppo_sweep.py plot --latest --mode roughing --require-diagnostics
```

When your sweep root is elsewhere, use `--sweep-root "C:\path\to\runs\ppo_sweep"`. To place figures elsewhere, add `--out-dir "C:\path\to\paper_figures"`.

## Original explicit options are retained

**Completed version-3 sweep manifest:**

```powershell
python scripts/analyze_ppo_sweep.py plot --manifest "runs/ppo_sweep/roughing/<sweep-id>/sweep_manifest.json"
```

**Earlier run directories without a compatible manifest:**

```powershell
python scripts/analyze_ppo_sweep.py plot `
  --fast "C:\path\to\fast\roughing_<id>" `
  --balanced "C:\path\to\balanced\roughing_<id>" `
  --precise "C:\path\to\precise\roughing_<id>"
```

The original version-3 command remains available:

```powershell
python scripts/plot_ppo_sweep.py --manifest "runs/ppo_sweep/roughing/<sweep-id>/sweep_manifest.json"
```

It continues to require a completed schema-version-3 sweep manifest; the **new automatic discovery and older-run compatibility** are provided by `analyze_ppo_sweep.py`.

## Generated figures (vector PDF + 300-dpi PNG)

- `ppo_sweep_learning_curve`: Three configuration-specific **mean evaluation returns** with **±1 sample SD** shaded envelopes and markers at actual recorded evaluation checkpoints. Each seed first averages its evaluation episodes; then mean and sample SD are taken across independent training seeds (`ddof=1`). No evaluation checkpoint interpolation or smoothing.
- `ppo_sweep_reliability_metrics`: Four-panel main figure — pass completion, safety termination, completed-pass productivity reward, and feed progress. Rates are shown in percent; bands are display-clipped to physical bounds **without modifying the underlying statistics**. Productivity reward is a reward term, **not a physical MRR measurement**; the code assigns zero if no pass completes.
- `ppo_sweep_reliability_supplement`: Time-limit truncation rate and peak displacement.
- `ppo_sweep_training_diagnostics` **if available**: Training episode return, explained variance, approximate KL divergence, clip fraction, and policy Gaussian SD (`train/std`) or entropy loss. The PPO `train/std` is **pre-clipping policy-distribution SD**, not the physical control-signal SD. Because different `n_steps` yield different optimizer-update times across configurations, **each configuration keeps its own observed update-step x-axis**. Within each configuration, only checkpoints actually recorded for all its seeds are aggregated. No interpolation, extrapolation, or smoothing. An epoch-averaged `train/approx_kl` is not compared to an individual-minibatch early-stopping threshold.

Accompanying CSV files contain raw evaluation returns, seed-resolved values, configuration-level mean and sample SD, and final training summary. `ppo_sweep_plot_provenance.json` records selected paths, seeds, number of evaluation episodes, statistical conventions, and which diagnostic tags were actually available.

**Interpretation:** The bands are descriptive **sample SD across independent training seeds**, not 95% confidence intervals. Monte Carlo replicas and evaluation episodes are **not** counted as additional independent training seeds. These plots show empirical optimization behaviour, not guaranteed mathematical convergence.

## Checkpoint screening and configuration selection

A per-seed `models/best_<seed>/best_model.zip` is a reliability-selected **checkpoint within that seed**, not a best sweep-wide configuration. `experiment.py plot --run latest` uses a run pointer; it is **not** a model-selection command.

Preliminary screening using already-recorded training callback metrics (no new simulations):

```powershell
python scripts/analyze_ppo_sweep.py screen --latest --mode roughing
```

To select a configuration more rigorously, first evaluate **each** of the three profiles on the same independent scenario set, using `experiment.py eval --run-dir ...` or the existing evaluator. The default `evaluation_seed=1000` is distinct from the usual checkpoint-screening seed base `10000`, but for important comparisons verify these have not been changed. This add-on does **not** run the expensive evaluations automatically.

Then:

```powershell
python scripts/analyze_ppo_sweep.py select --latest --mode roughing
```

`select` checks pairing of reset seeds, scenario index, initial milling line, fixed finishing depth, and evaluation settings. It selects a **configuration based on aggregate results across training seeds**, and identifies a representative median-ranked seed only for an illustrative trajectory. The selected configuration must be evaluated again, alongside NMPC–MHE, on separate final test scenarios. Never use the selection set as an unbiased final test.

Both `screen` and `select` also retain `--manifest` and explicit run-path options.

## Validation and limitations

Run tests from the repository root:

```powershell
python -m unittest discover -s tests -p test_sweep_analysis.py -v
```

The reviewed add-on was checked with Python syntax compilation, 10 regression tests, mock TensorBoard scalar series, synthetic three-profile learning curves, PDF/PNG output inspection, automatic discovery with old/new layouts, and failure cases for incomplete/mismatched data. **Real TensorBoard event files and full project PPO training were not available in the review environment.** Real event-file reading, actual evaluation performance, and deployment on your Windows environment must still be validated locally. An absent TensorBoard dependency/log prints a warning rather than fabricating a diagnostic figure; `--require-diagnostics` makes it a hard failure.

## Interpretation of statistical randomness

For each profile and checkpoint: `evaluations.npz["results"]` is shaped
`[n_checkpoints, n_evaluation_episodes]` within one training seed. One policy
contributes its **mean across the evaluation episodes**. The plotted solid
line is the mean of these per-policy observations across independent training
seeds and the shaded region is **±1 sample SD (`ddof=1`) across those policies**,
not SD over individual evaluation episodes. The callback uses deterministic
policy actions but randomized milling conditions and configured disturbances.
The callback reuses a common set of evaluation scenario seeds at every
checkpoint and across training seeds; the resulting band is conditional on
that finite test set and may also contain evaluation-estimation noise.

**Important separate upstream issue:** for trained runs with
`controller_hold_env_steps > 1`, the current `ControllerSampleTimeWrapper`
sums the rewards of fine environment steps but forwards only the final step's
`reward_terms` dictionary. Therefore the callback's completed-pass
productivity and vibration/omega component sums can be undercounted. The
primary total episode-return learning curves are not affected by this issue.
Do not interpret productivity-component reliability plots or related
checkpoint-ranking tie-breakers as full held-interval sums until that upstream
aggregation is corrected and the metrics are regenerated. This post-processing
add-on intentionally does not modify a trainer that may already have trained
checkpoints.

## v5 statistical review and upgrade (minimal changes)

- The primary evaluation learning curve is unchanged: `results[K, M]` is first
  averaged along the evaluation-episode dimension *within each training seed*,
  then the cross-seed sample mean and SD are calculated at exactly matching
  saved evaluation checkpoints. It never interpolates evaluation returns.
- Training TensorBoard diagnostic plots now also use **only exactly shared,
  actually recorded update steps** within each configuration. Since Fast,
  Balanced and Precise have distinct rollout schedules, their curves retain
  their own native x-axis coordinates. No invented intermediate points.
- When training metadata reports `controller_hold_env_steps > 1`, the plotted
  reliability diagnostics **do not show** `median_completed_productivity` or
  `mean_max_displacement_m`; those historical callback metrics omit information
  from intermediate fine steps of the held-action interval. Main total-return,
  pass-completion, safety-termination, and progress figures are still available.
  For corrected reward-component summaries, fix the upstream wrapper/callback
  and re-evaluate checkpoints rather than retroactively altering recorded data.
- Figures are descriptive results conditional on the callback's shared finite
  evaluation scenario set. Three seeds are sufficient to compute sample SD but
  not to establish a precise distribution of expected performance.

Run `python -m unittest discover -s tests -p test_sweep_analysis.py -v`
for the offline regression suite. Training/real event files are not bundled.
