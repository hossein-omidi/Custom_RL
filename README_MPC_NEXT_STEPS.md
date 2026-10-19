# Face-milling MPC: diagnostic follow-up after the 18 October failure

## What the supplied log actually establishes

The user ran `python scripts/MPC/mpc_face_milling_chatgpt_V3.py` with **no CLI flags**.
The log prints `w-margin=0.50`, `hard-w-margin=0.80`, `omega-rate=20000` and
never prints `Stability-first MPC profile`. Consequently, the opt-in stability
profile supplied in the earlier patch was NOT active in that log. Do not treat
this as a failed experimental test of the patched profile.

The run terminated after 84,826 environment steps with an observed displacement
of 3.008 mm (limit 3.000 mm), despite reported 100% accepted MPC and MHE
optimizations, in 7,661.7 wall-clock seconds. NLP success does not imply that
an approximate plant model predicts the actual continuous trajectory safely.

## Main numerical concern

At `dtheta=0.392699` radians and the fastest retained mode f=104.18 Hz,

theta-frequency advance per MPC interval = (2*pi*f)*dtheta/omega.

At 400 rpm this is 6.14 rad, or only ~1.02 samples per vibration period.
At 2200 rpm this is 1.12 rad, or ~5.6 samples per period.
At 4000 rpm this is 0.614 rad, or ~10.2 samples per period.

The Radau scheme uses one finite element of degree two per control interval.
Hard vibration constraints are enforced at interval endpoints by default.
Consequently, severe between-node excursions and inaccurate low-speed
predictions are plausible, though not proven as the sole cause. The independent
continuous plant limit is evaluated more often.

## Scope and installation

This ZIP contains ONLY the replacement `scripts/MPC/mpc_face_milling_chatgpt_V3.py`
based on the PREVIOUS stability-first patch, a diff **relative to that previous
patch**, and a dependency-light math test. No plant, environment, RL, reward,
MHE model, or evaluation scripts are modified. Back up the existing V3 script
before replacing it. Unzip at repository root.

The controller must print `Stability-first MPC profile` when the `--stability-first`
flag is passed, and `DIAGNOSTIC ONLY controller bounds` when the new experimental
floor/cap are passed. Otherwise you are not running this update.

## Stage 0: quick validation (NO full pass)

```
python scripts/MPC/mpc_face_milling_chatgpt_V3.py --help
python scripts/MPC/mpc_face_milling_chatgpt_V3.py --verify-only --verify-prediction \
    --y0 0.519 --no-randomize-y0
```

Check output of both. The verify-only step tests model fidelity, not a closed
loop. On Windows PowerShell use a single line or backticks instead of `\`.

## Stage 1: use the previously provided stability profile, without diagnostic limits

```
python scripts/MPC/mpc_face_milling_chatgpt_V3.py --stability-first \
    --y0 0.519 --no-randomize-y0 --seeds 0 --n-episodes 1 \
    --max-env-steps 45000 --progress-every 250 --out-dir trial_stability
```

Expected profile: `w_margin=0.40`, `hard_w_margin=0.72`, `omega_rate=3000`,
`q_energy=18`, `curvature_omega=20`, `minimum_cut_depth=0.5`. Keep phase step
0.25 pitch, K=3, and delay M selected automatically for fair comparison.
The 45k-step run tests the transition at about 4 seconds (time depends on dt).
It is not a pass-completion test.

## Stage 2: diagnose low-speed interpolation/constraint losses if Stage 1 is unstable

```
python scripts/MPC/mpc_face_milling_chatgpt_V3.py --stability-first \
    --mpc-omega-floor-rpm 2200 --mpc-ap-ceiling-mm 3.0 \
    --y0 0.519 --no-randomize-y0 --seeds 0 --n-episodes 1 \
    --max-env-steps 45000 --progress-every 250 --out-dir trial_stability_bounded
```

These optional limits affect only the MPC's admissible decisions and applied
command projection. Plant parameters stay identical. A better result in this
trial suggests sensitivity to the low-speed numerical approximation, depth
pressure or nonconvex control switching; it does NOT establish which by itself.
Test one bound at a time if isolating individual effects is important.

If this works, extend to 95,000 env steps to cross the previous failure region;
then only if that succeeds, do a full 90% pass. Do not claim that a partial run
is a completed pass. Set a minimum depth appropriate to the machining objective.

## Stage 3: reintroduce generality after finding a safe regime

Remove the `--mpc-ap-ceiling-mm` and `--mpc-omega-floor-rpm` restrictions, one
at a time, and compare. If low-speed prediction is the culprit, the principled
long-term solution is finer/adaptive physical-time integration or more Radau
finite elements at low speed, with constraints checked at meaningful
intermediate points and a verified model-to-plant error margin. Merely adjusting
weights cannot resolve numerical aliasing.

## What changed in this patch

* Optional CLI-only MPC speed floor and depth ceiling, with initial command and
  physical actuator rate projection kept consistent with these diagnostic bounds.
* Measured peak displacement accumulated at **all** environment steps in a
  control interval and incorporated into next control decision's hysteretic
  safety ratio. It does not override the original environment limit.
* Diagnostics now report the measured prior-interval peak relative to w_limit.
* No change to the structural ODE, transport PDE, optimizer, MHE model, or RL.

## Scientific qualification

Neither the revised measured guard nor the previous constant-velocity preview
is a mathematical guarantee of constraint satisfaction between grid points.
A certified robust constraint would require a validated disturbance/model-error
set and sufficient intra-interval constraint enforcement. The optional speed
floor and depth cap are experimental diagnostics, not final general limits.

The existing controller forces the delay-line state using reconstructed cutter
displacement from two sensors for three retained modes. That reconstruction is
underdetermined instantaneously and can bias delayed force predictions. If the
bounded experiment still fails, the next item is a *prediction-residual study*
using logged true plant states for diagnostics (never as controller input).

For comparisons report at least pass completion, peak sensor displacement,
productive cutting depth, feed progress, total variation and reversals of
spindle speed, MPC/MHE acceptance, rate-clipping fraction, and CPU time.
