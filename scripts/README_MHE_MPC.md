# Unified do-mpc MHE + NMPC for face-milling chatter suppression

Requires: `pip install do-mpc==5.1.1 casadi` (plus the custom_rl package installed, `pip install -e .`).

Place `mpc_mhe_dompc.py` in `scripts/`.

## Run (finishing example, pinned depth and milling line)
    python scripts/mpc_mhe_dompc.py --env-id CustomODEPlateFinish-v0 --ap 0.5 --y0 0.2 --no-randomize-y0 --seeds 0 --max-env-steps 2000 --out-dir mpc_mhe_finish

## Run (roughing)
    python scripts/mpc_mhe_dompc.py --env-id CustomODEPlate-v0 --seeds 0 --max-env-steps 20000 --out-dir mpc_mhe_rough

## Compare with RL (same schema as the CasADi MPC output)
    python scripts/compare_mpc_rl.py --rl-dir eval_rough --mpc-dir mpc_mhe_rough --out-dir plots/compare_mhe

## Architecture (do-mpc 5.1.1 standard)
    gym obs -> MHE.make_step(y) -> x_hat -> MPC.make_step(x_hat) -> slew clip -> env action

- One shared continuous do-mpc model (modal plate + averaged regenerative face-milling force).
- Cutter mode shapes phiz and delayed displacement wz_del enter as time-varying parameters:
  MHE horizon maps them backward onto recorded history; MPC holds newest values (frozen delay).
- Applied inputs are declared as noise-free measurements (documented do-mpc MHE pattern).
- MHE objective: set_default_objective with information matrices P_x=P0^-1, P_v=R^-1, P_w=Q^-1.
- MPC objective: verified polynomial economic cost (bilinear MRR - quadratic vibration),
  rterm input-move penalty, soft |w| margin via set_nl_cons, per-solve IPOPT wall cap.
- The controller sees ONLY the RL observation (sensors + cutter position); ground truth
  x_modal is logged for diagnostics only.

## Verified (closed loop, CustomODEPlateFinish-v0)
- MPC feasible 100%, MHE solved 100%, no NaN.
- max|w| = 5.1e-4 m under the hard limit; soft margin respected.
- Smooth spindle command (rterm + range-capped slew clip).
- MHE modal-state error converges 3.96 -> 0.11; output error -> 9e-2.
- ~0.04 s/MPC solve + ~0.03 s/MHE solve (much faster than the hand-rolled CasADi NMPC).
