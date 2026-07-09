"""Fast interface/behaviour tests for the CasADi face-milling MPC controller.

These tests avoid long closed-loop rollouts.  They check that the OCP builds for
both control modes, that the internal averaged model reproduces the plant's
one-step response, and that a (tiny-horizon) solve returns an admissible action.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("casadi")

import gymnasium as gym

from custom_rl import register_envs
from custom_rl.eval.pipeline import plate_env_kwargs

from scripts.mpc_face_milling import FaceMillingMPC, MPCConfig, _deinterleave


def _make_env(env_id: str):
    register_envs()
    kw = plate_env_kwargs(
        reward_id="dense", dt=1e-4, n_substeps=10,
        max_episode_steps=50000, randomize_y0=False,
    )
    return gym.make(env_id, **kw)


def _small_cfg(**over):
    base = dict(horizon=6, control_hold=3, n_rk=4, prescreen_n_omega=8,
               prescreen_n_ap=4, max_iter=12)
    base.update(over)
    return MPCConfig(**base)


def test_first_mode_dimensions():
    env = _make_env("CustomODEPlate-v0")
    try:
        mpc = FaceMillingMPC(env, _small_cfg())
        assert mpc.control_ap is True
        assert mpc.nu == 2                      # [omega, ap]
        assert mpc.K == 6 and mpc.nS == 2
        assert mpc.Ts == pytest.approx(3e-3)    # control_hold * dt * n_substeps
        assert mpc.nb >= 4
    finally:
        env.close()


def test_second_mode_dimensions():
    env = _make_env("CustomODEPlateFinish-v0")
    try:
        mpc = FaceMillingMPC(env, _small_cfg())
        assert mpc.control_ap is False
        assert mpc.nu == 1                      # [omega] only, ap fixed
    finally:
        env.close()


def test_averaged_model_matches_plant_one_step():
    """The MPC internal averaged model must reproduce the plant's one control
    step to good accuracy from a quiescent state (validates signs/scaling)."""
    env = _make_env("CustomODEPlate-v0")
    try:
        mpc = FaceMillingMPC(env, _small_cfg())
        plant = env.unwrapped.plant
        obs, info = env.reset(seed=0, options={"y0": 0.2})
        z = mpc.state_estimate(info)

        rpm, ap = 20000.0, 1.0
        om = rpm * 2 * np.pi / 60.0
        a = 2 * (np.array([om, ap]) - plant._ctrl_low) / (plant._ctrl_high - plant._ctrl_low) - 1

        # one MPC control step of the true plant
        for _ in range(mpc.cfg.control_hold):
            obs, r, term, trunc, info = env.step(a)
        w_true = float(np.max(np.abs(info["w_sensor"])))

        # one MPC control step of the internal averaged model
        xc = float(np.clip(plant.L1 - 0.0 - plant.x0_cutter, 0.0, plant.L1))
        phiz = mpc.phiz_at(xc, plant.y_cutter)
        wz0 = float(phiz @ z[: mpc.K])
        wz_del = wz0  # quiescent history -> zero initial regeneration
        u = np.array([om, ap])
        z_next = np.asarray(mpc._F(z, u, wz_del, phiz, ap, mpc.S0, mpc.S1)).reshape(-1)
        w_model = float(np.max(np.abs(mpc.Phi @ z_next[: mpc.K])))

        # both are tiny (~1e-4) and should agree to within 40 %
        assert w_model == pytest.approx(w_true, rel=0.4, abs=2e-5)
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ["CustomODEPlate-v0", "CustomODEPlateFinish-v0"])
def test_solve_returns_admissible_action(env_id):
    env = _make_env(env_id)
    try:
        mpc = FaceMillingMPC(env, _small_cfg())
        opts = {"y0": 0.2} if "Finish" not in env_id else {"y0": 0.2, "ap": 0.5}
        obs, info = env.reset(seed=0, options=opts)
        z = mpc.state_estimate(info)
        xc = float(np.clip(mpc.plant.L1 - 0.0 - mpc.plant.x0_cutter, 0.0, mpc.plant.L1))
        phiz = mpc.phiz_at(xc, mpc.plant.y_cutter)
        wz_hist = np.full(mpc.nb, float(phiz @ z[: mpc.K]))
        ap_fixed = float(getattr(mpc.plant, "_episode_ap", 0.5))
        u_prev = np.array([2000.0, 0.3])[: mpc.nu]

        sol = mpc.solve(z, phiz, wz_hist, ap_fixed, u_prev)
        a = mpc.to_env_action(sol["u_phys"])

        assert a.shape == (mpc.nu,)
        assert np.all(np.isfinite(a))
        assert np.all(a >= -1.0 - 1e-6) and np.all(a <= 1.0 + 1e-6)
        assert mpc.omega_min - 1 <= sol["u_phys"][0] <= mpc.omega_max + 1
        assert mpc.ap_min - 1e-6 <= sol["u_phys"][1] <= mpc.ap_max + 1e-6
    finally:
        env.close()


def test_observer_feedback_shape():
    env = _make_env("CustomODEPlate-v0")
    try:
        mpc = FaceMillingMPC(env, _small_cfg(feedback="observer"))
        obs, info = env.reset(seed=0, options={"y0": 0.2})
        z = mpc.state_estimate(info)
        assert z.shape == (2 * mpc.K,)
        assert np.all(np.isfinite(z))
    finally:
        env.close()
