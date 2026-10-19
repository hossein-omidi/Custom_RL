# MPC-only patch: mathematical review and validation protocol

**Source:** `hossein-omidi/Custom_RL`, branch `mpc-correction`, original `scripts/MPC/mpc_face_milling_chatgpt_V3.py` Git blob `b866bd42a03a8cd8abc058eb5fecc90082c12391`. The uploaded file matched this exact Git blob after conversion of CRLF to LF. The patch changes this file only; the included separate test does not modify the RL plant or its evaluation.

## What was changed

1. **Discrete slew-limit timing corrected.** The original OCP used `|u_k-u_(k-1)| * omega_k <= rate*dtheta`. A command change happens after the **preceding** held command, so the discrete approximation should use `omega_(k-1)`: `|u_k-u_(k-1)| * omega_(k-1) <= rate*dtheta`. At runtime the projection uses the **actually elapsed preceding hold** `dt_(k-1)`, so `|u_k-u_(k-1)| <= rate*dt_(k-1)` applies to executed commands. The first step uses a nominal initialization interval because no preceding controlled hold is available. These are *sampled-command* slew limits under a zero-order-held actuator; a truly continuous velocity limit requires modeling the actuator state and ramp, not ZOH switching.
2. **Control curvature regularization added** using `sum_(k=1)^(Np-1) (s_omega*((omega_k-2*omega_(k-1)+omega_(k-2))/delta_omega)^2 + s_ap*((ap_k-2*ap_(k-1)+ap_(k-2))/delta_ap)^2)` with the available previous input at k=1. This suppresses alternating speed/depth requests but preserves the existing hard bounds and model. Flags: `--curvature-omega` (default 8), `--curvature-ap` (default 2); both `0` recover the original smoothness objective. The original `set_rterm` still penalizes first differences.
3. **Material removal normalization corrected** for the case `ap_min != 0`: physical MRR with fixed feed per tooth is proportional to `omega*ap`, not `omega*(ap-ap_min)`. For `ap_min=0` results are unchanged. The previously unused `--q-mrr` can now override `--prod-weight` when positive; default `--q-mrr 0` preserves `--prod-weight`.
4. **Optional CasADi NLP compilation** (`--compile-nlp`) calls documented `MPC.compile_nlp` and `MHE.compile_nlp` for the *unchanged symbolic NLPs* and writes the compiler output only into an automatically removed temporary directory. This can accelerate repeated derivative/evaluation calls but has an upfront compilation cost and depends on a working C compiler. Default is OFF; measure whether it helps. It is not an acceleration guarantee.
5. **Realized actuator-rate diagnostics corrected** to use `dt_i` for the difference between `u_i` and `u_(i+1)` rather than the next interval's elapsed time.

## Formulation represented in this MPC code

The phase-domain structural/transport state is `x=(eta,v,qh,dF)` with `d eta/dtheta=v/omega`, `d v/dtheta=( -2*zeta*wn*v - wn^2*eta -lambda*eta^3+phi*Fz/M + dF*phi )/omega` and `d qh/dtheta=-(DII*qh+DI0*wc)/p`. The last delay-line node approximates `wc(theta-p)`. The force uses a smoothed positive chip thickness and known interval-averaged engagement coefficients. The physical-time economic running objective is transformed with the `1/omega` Jacobian. The stage objective includes productivity, preferred speed, energy/output vibration, and elapsed time; the terminal objective penalizes vibration. Hard input/displacement limits, soft interior displacement targets, quadratic first differences, and quadratic second differences complete the patch's MPC OCP.

This is **nonconvex**, so IPOPT produces a local solution, not a guaranteed global optimum. The model is a structurally consistent *control approximation*, not an exact infinite-dimensional nonlinear/stochastic plate model. In particular:

- The stored full-plant validation supports nominal reduced-model use but does not prove closed-loop convergence or correctness for every spindle speed, cutting depth, or disturbance.
- The two output sensors do not algebraically identify all three retained modal displacements. MHE provides temporal estimates; overwriting its transport delay line with a pseudoinverse-based reconstructed history can introduce noise and bias and should be examined in estimator sensitivity experiments.
- The `safety_guard` is a first-order extrapolation based on measured velocity, **not** a rigorous bound on future displacement.
- Hard vibration constraints are checked on sensor and cutter outputs at finite prediction nodes (by default endpoints), not at every position and time. Interpret any claims of guaranteed plate-wide safety accordingly.
- With a fixed MHE process-noise weight, the implemented phase-domain process-noise penalty is an estimator tuning convention, not a full physical-time Itô diffusion transformation. A Wiener diffusion transformed from time to phase scales its noise amplitude as `1/sqrt(omega)`.
- The current force TVPs are averaged over each phase cell. Increasing `--dtheta-frac` requires quantitative prediction and closed-loop tests before asserting that no accuracy is lost.

## Suggested experiment matrix (do not change the RL baseline)

Use the same seeds, plant, controller scenarios, fixed `y0`/`ap`, software version, and hardware. Compare three cases:

| Experiment | Controls | Purpose |
|---|---|---|
| A: baseline-equivalent | `--curvature-omega 0 --curvature-ap 0 --no-compile-nlp` | Reference after the rate-timing correction |
| B: smooth | `--curvature-omega 8 --curvature-ap 2 --no-compile-nlp` | Switching vs. economic/constraint performance |
| C: compiled | `--curvature-omega 8 --curvature-ap 2 --compile-nlp` | Test the benefit of compiling exactly the same OCP |

Examples (from repository root after applying):

```bash
python scripts/MPC/mpc_face_milling_chatgpt_V3.py --env-id CustomODEPlate-v0 --no-randomize-y0 --y0 0.2 --max-env-steps 3000 --curvature-omega 8 --curvature-ap 2 --compile-nlp --out-dir mpc_benchmark
python -m unittest discover -s tests -p test_mpc_patch_math.py -v
```

Do not select a faster horizon/discretization by speed alone. Record **median/p95 MPC and MHE solve times, solver-failure/fallback percentage, full-run wall time, maximum and RMS sensor displacement, control total variation, number of direction reversals, achieved pass progress, elapsed cutting time, and tracking/model error**. For time-grid accuracy, use matched command profiles and compare `dtheta_frac=0.25, 0.5, 1.0`, including lower-speed cases where even the current degree-2 single-element collocation can under-resolve the highest retained mode. Consider increasing collocation elements if needed, despite added computational cost.

Potential future high-impact changes, **not included** because they could alter fidelity: estimator replacement (EKF/UKF plus delay-history observer), sparse multiple-shooting with direct integration, SQP real-time iteration, control move blocking, a larger phase interval, reduced MHE horizon or mode/delay order. Benchmark these against the current full-order plant and retained modal output, especially disturbance and stability outcomes, before making research accuracy claims.

## Validation status

- Uploaded V3 **verified identical** to GitHub branch file by SHA after newline normalization.
- `ast.parse`/`py_compile` succeeds on patched script.
- Five dependency-light `unittest` checks pass, including polynomial differentiation, sampled-command slew bounds and delay-channel transfer error versus `exp(-i nu p)` for the selected 3-mode test case.
- **Not validated here:** construction/execution of do-mpc NLP (do-mpc and Gymnasium unavailable in this execution environment), compiled-solver benefit, closed-loop response, speedup, or decrease in actual switching. These require running the provided tests and a matched episode in your project environment.
