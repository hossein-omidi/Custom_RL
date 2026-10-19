#!/usr/bin/env python3
"""Linearized finite-window observability diagnostic for the phase-domain MHE.

This is an OFFLINE, plant-read-only analysis. It uses the actual retained mode
frequencies, damping, and measured sensor-mode matrix. The cutting-force
feedback and transport dynamics are not linearized here, so full rank is a
necessary sanity check, not sufficient evidence for nonlinear MHE accuracy.

Run from the repository root:
    python scripts/MPC/diagnose_mhe_observability.py
"""

from __future__ import annotations

import argparse
import numpy as np
from scipy.linalg import expm
import gymnasium as gym

from custom_rl import register_envs
from custom_rl.eval.pipeline import plate_env_kwargs
from mpc_face_milling_chatgpt_V3 import PhasePlant


def finite_window_observability(pp, n_horizon: int, rpm: float,
                                dtheta_frac: float = 0.25) -> dict:
    """Return scaled sensor-state observability singular values over an MHE window."""
    K = pp.K
    A = np.block([
        [np.zeros((K, K)), np.eye(K)],
        [-np.diag(pp.wn**2), -np.diag(2.0*pp.zeta*pp.wn)]
    ])
    # Measurements are precisely the normalized displacement/velocity channels.
    C = np.block([
        [pp.Phi / pp.w_obs, np.zeros_like(pp.Phi)],
        [np.zeros_like(pp.Phi), pp.Phi / pp.wd_obs]
    ])
    # State scaling makes the singular values dimensionless and comparable:
    # 1 mm modal displacement, 0.1 m/s modal velocity are reference amplitudes.
    state_ref = np.diag([1.0e-3]*K + [1.0e-1]*K)
    om = float(rpm) * 2.0*np.pi/60.0
    dt = float(dtheta_frac) * pp.p / om
    O = np.vstack([C @ expm(A*(j*dt)) @ state_ref
                   for j in range(int(n_horizon)+1)])
    singular = np.linalg.svd(O, compute_uv=False)
    rank = int(np.linalg.matrix_rank(O, tol=singular[0]*1e-9))
    return dict(rpm=float(rpm), seconds=n_horizon*dt, rank=rank,
                total_dim=2*K, condition=float(singular[0]/max(singular[-1],1e-30)),
                relative_smallest=float(singular[-1]/max(singular[0],1e-30)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--env-id',default='CustomODEPlate-v0')
    p.add_argument('--n-modes',type=int,default=3)
    p.add_argument('--dtheta-frac',type=float,default=0.25)
    p.add_argument('--mhe-horizons',type=int,nargs='+',default=[8,12,16])
    p.add_argument('--rpm',type=float,nargs='+',default=[400,1000,2000,4000])
    a=p.parse_args()
    register_envs()
    kw=plate_env_kwargs(reward_id='dense',dt=1e-4,n_substeps=1,
                        max_episode_steps=10000,randomize_y0=False)
    env=gym.make(a.env_id,**kw)
    try:
        pp=PhasePlant(env,n_modes=a.n_modes)
        print(f'Modes {pp.sel.tolist()}, frequencies Hz {(pp.wn/(2*np.pi)).round(2).tolist()}, '
              f'Phi shape={tuple(pp.Phi.shape)}')
        print('Warning: this is only the sensor/linear-plate observability diagnostic, '
              'not the full nonlinear delayed-plant MHE observability.')
        for H in a.mhe_horizons:
            for rpm in a.rpm:
                r=finite_window_observability(pp,H,rpm,a.dtheta_frac)
                print(f'H={H:2d}, rpm={rpm:5.0f}, window={r["seconds"]*1e3:8.3f} ms, '
                      f'rank={r["rank"]}/{r["total_dim"]}, '
                      f'cond={r["condition"]:.3e}, s_min/s_max={r["relative_smallest"]:.3e}')
    finally:
        env.close()


if __name__=='__main__':
    main()
