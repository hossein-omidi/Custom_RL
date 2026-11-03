# Synthetic regression verification

Python 3.13, NumPy and Matplotlib available; actual Gymnasium/SB3/`custom_rl` runtime was mocked for pipeline integration. Seven tests were executed and all passed:

1. Held-out roughing-mode mocked end-to-end evaluator; full-step metrics, MC and scenario CSV, and PNG/PDF outputs.
2. Explicit refusal to pool MC rollouts with different physical test scenarios.
3. Undefined (rather than zero) SD when only one MC realization supports a timepoint.
4. 50-ms RMS computed from full saved sensor samples before display decimation, with vector PDF output.
5. Invariance of physical metrics and returns when the file-storage stride is changed from 1 to 3.
6. Correct physical-time alignment for sensor signals and zero-order-hold actions; no extrapolation beyond termination.
7. Four-panel PPO training-diagnostic figure from synthetic TensorBoard scalar data, including KL/clip metrics.

**Important:** These are implementation/format regression tests, not a finite-element, cutting-force, delayed-dynamics, hardware timing, training-success, or real-checkpoint validation. Original real TensorBoard event files were unavailable. The screenshot/plot previews used to inspect typography were generated from synthetic data and should not be included as research observations.
