# Minimal-change audit

| File | Change | Unaffected |
|---|---|---|
| `scripts/eval_policy.py` | Optional RL policy action hold with same fine safety-check step, adjustable reset seed, restricted init/noise overrides, metadata | Trained PPO policy, training, reward function, plant |
| `scripts/MPC/mpc_face_milling_chatgpt_V3.py` | Optional paired reset seed, complete noise metadata, clarified console time labels | do-mpc objective, MHE, history states, prediction equations, sampling logic |
| `scripts/compare_mpc_rl.py` | Validated reporting & safe, physically meaningful comparison | Controller inputs/dynamics |

There is deliberately **no optimizer or environment modification**. The branch-local physical failure remains an independent controller problem if replicated with a matched simulator/safety-check grid.
