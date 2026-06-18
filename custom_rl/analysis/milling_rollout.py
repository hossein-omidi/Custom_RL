"""Fixed-action and env rollouts for stability-lobe analysis."""

from __future__ import annotations

import numpy as np

from custom_rl.integration import get_integrator
from custom_rl.plants import f_nonlinear2 as fmod
from custom_rl.plants.modal_state import split_modal_state
from custom_rl.plants.plate import PlatePlant
from custom_rl.plants.time_scales import recommend_integration_dt


def _apply_initial_perturbation(plant: PlatePlant, x: np.ndarray, perturb: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1).copy()
    if perturb <= 0.0:
        return x
    two_field = plant.milling_config.is_feed_normal_full()
    x[0] = perturb
    if two_field and x.size > 2 * plant.K:
        x[2 * plant.K] = perturb * 0.1
    return x


def rollout_uncontrolled(
    plant: PlatePlant,
    omega: float,
    ac_mm: float,
    *,
    t_final: float,
    macro_dt: float,
    rng: np.random.Generator,
    perturb: float = 1e-6,
    integrator: str = "dde_rk4",
) -> dict:
    """
    Deterministic fixed-(omega, ac) plant rollout with Test2-style sub-stepping.

    Resamples geometry / sensor / pass line via ``plant.reset(rng)``.
    """
    plant.reset(rng)
    fmod.reset_episode_state(plant._modal_state_history)
    integrate_fn = get_integrator(integrator)

    scales = recommend_integration_dt(plant, macro_dt=macro_dt)
    sub_dt = scales["dt_recommended_training_substep_s"]
    substeps = scales["n_substeps"]

    u_norm = plant.physical_to_normalized_action(
        np.array([omega, max(ac_mm, plant.u_phys_low[1])], dtype=np.float64)
    )
    x = _apply_initial_perturbation(plant, np.zeros(plant.state_dim), perturb if ac_mm > 0 else 0.0)
    fmod.bind_modal_history(plant._modal_state_history)
    plant.record_modal_state(0.0, x.copy(), omega=omega)

    n_macro = max(int(round(t_final / macro_dt)), 1)
    sensor_ws: list[float] = []
    force_clipped: list[bool] = []
    delta_f: list[float] = []
    delta_n: list[float] = []
    f_normal_raw: list[float] = []
    f_feed_raw: list[float] = []
    terminated = truncated = False
    term_reason = "complete"

    t = 0.0
    two_field = plant.milling_config.is_feed_normal_full()
    try:
        for _ in range(n_macro):
            w_s, _ = plant.state_to_sensor_signals(x)
            sensor_ws.append(float(np.max(np.abs(w_s))))

            eta_n, _, eta_f, _ = split_modal_state(x, plant.K, two_field=two_field)
            fr = fmod._directional_force_result(t, eta_n, eta_f, omega, ac_mm)
            force_clipped.append(bool(fr.get("force_clipped", False)))
            delta_f.append(float(fr["Delta_f"]))
            delta_n.append(float(fr["Delta_n"]))
            f_normal_raw.append(float(fr.get("F_normal_raw", fr["F_normal_total"])))
            f_feed_raw.append(float(fr.get("F_feed_raw", fr["F_feed_total"])))

            for _ in range(substeps):
                x = integrate_fn(plant.dynamics, t, x, u_norm, sub_dt, n_steps=1)
                t += sub_dt
                plant.record_modal_state(t, x, omega=omega)

            terminated, truncated, term_info = plant.termination(t, x)
            if terminated or truncated:
                term_reason = str(term_info.get("termination_reason", "truncated"))
                break
    finally:
        fmod.unbind_modal_history()

    sw = np.asarray(sensor_ws, dtype=np.float64)
    clip_pct = 100.0 * float(np.mean(force_clipped)) if force_clipped else 0.0
    path_y = getattr(plant, "path_y_start", None)
    return {
        "omega": omega,
        "ac_mm": ac_mm,
        "finite": bool(np.all(np.isfinite(x))),
        "clip_pct": clip_pct,
        "max_w_m": float(np.max(sw)) if sw.size else 0.0,
        "max_delta_f_m": float(np.max(np.abs(delta_f))) if delta_f else 0.0,
        "max_delta_n_m": float(np.max(np.abs(delta_n))) if delta_n else 0.0,
        "max_F_normal_N": float(np.max(np.abs(f_normal_raw))) if f_normal_raw else 0.0,
        "max_F_feed_N": float(np.max(np.abs(f_feed_raw))) if f_feed_raw else 0.0,
        "sensor_w_series": sw,
        "terminated": terminated,
        "term_reason": term_reason,
        "path_y_start_m": float(path_y) if path_y is not None else None,
        "sim_time_s": t,
    }


def rollout_env_fixed_action(
    env,
    omega: float,
    ac_mm: float,
    *,
    seed: int,
    n_steps: int,
    perturb: float = 1e-6,
) -> dict:
    """Gym rollout with fixed normalized action each step (uncontrolled via env)."""
    plant = env.unwrapped.plant
    u_fixed = plant.physical_to_normalized_action(
        np.array([omega, max(ac_mm, plant.u_phys_low[1])], dtype=np.float64)
    )
    obs, info = env.reset(seed=seed)
    path_y = info.get("path_y_start", info.get("y_start"))

    if ac_mm > 0 and perturb > 0:
        state = _apply_initial_perturbation(plant, env.unwrapped._state, perturb)
        env.unwrapped._state = state
        if hasattr(plant, "record_modal_state"):
            plant.record_modal_state(0.0, state, omega=omega)

    sensor_ws: list[float] = []
    terminated = False
    term_reason = "pass_complete"

    for _ in range(n_steps):
        obs, _, term, trunc, info = env.step(u_fixed)
        w = np.asarray(info.get("sensor_w", []), dtype=np.float64)
        if w.size:
            sensor_ws.append(float(np.max(np.abs(w))))
        if term or trunc:
            terminated = term
            term_reason = str(info.get("termination_reason", "truncated"))
            break

    sw = np.asarray(sensor_ws, dtype=np.float64)
    x = env.unwrapped._state
    return {
        "omega": omega,
        "ac_mm": ac_mm,
        "finite": bool(np.all(np.isfinite(x))),
        "clip_pct": 0.0,
        "max_w_m": float(np.max(sw)) if sw.size else 0.0,
        "max_delta_f_m": 0.0,
        "max_delta_n_m": 0.0,
        "max_F_normal_N": 0.0,
        "max_F_feed_N": 0.0,
        "sensor_w_series": sw,
        "terminated": terminated,
        "term_reason": term_reason,
        "path_y_start_m": float(path_y) if path_y is not None else None,
        "sim_time_s": env.unwrapped._t,
    }


def rollout_env_trained(
    env,
    omega: float,
    ac_mm: float,
    *,
    seed: int,
    n_steps: int,
    policy,
    perturb: float = 1e-6,
) -> dict:
    """
    Closed-loop rollout: grid (omega, ac) is the *initial* operating point only;
    the policy may change actions each step.
    """
    plant = env.unwrapped.plant
    u_init = plant.physical_to_normalized_action(
        np.array([omega, max(ac_mm, plant.u_phys_low[1])], dtype=np.float64)
    )
    obs, info = env.reset(seed=seed)
    path_y = info.get("path_y_start", info.get("y_start"))

    if ac_mm > 0 and perturb > 0:
        state = _apply_initial_perturbation(plant, env.unwrapped._state, perturb)
        env.unwrapped._state = state
        if hasattr(plant, "record_modal_state"):
            plant.record_modal_state(0.0, state, omega=omega)

    env.unwrapped._prev_action_norm[:] = plant.physical_to_normalized_action(u_init)

    sensor_ws: list[float] = []
    actions_phys: list[np.ndarray] = []
    rewards: list[float] = []
    terminated = False
    term_reason = "pass_complete"

    for _ in range(n_steps):
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        rewards.append(float(reward))
        w = np.asarray(info.get("sensor_w", []), dtype=np.float64)
        if w.size:
            sensor_ws.append(float(np.max(np.abs(w))))
        ap = info.get("action_phys")
        if ap is not None:
            actions_phys.append(np.asarray(ap, dtype=np.float64))
        if term or trunc:
            terminated = term
            term_reason = str(info.get("termination_reason", "truncated"))
            break

    sw = np.asarray(sensor_ws, dtype=np.float64)
    x = env.unwrapped._state
    ap_arr = np.asarray(actions_phys, dtype=np.float64) if actions_phys else np.zeros((0, 2))
    return {
        "omega": omega,
        "ac_mm": ac_mm,
        "finite": bool(np.all(np.isfinite(x))),
        "clip_pct": 0.0,
        "max_w_m": float(np.max(sw)) if sw.size else 0.0,
        "max_delta_f_m": 0.0,
        "max_delta_n_m": 0.0,
        "max_F_normal_N": 0.0,
        "max_F_feed_N": 0.0,
        "sensor_w_series": sw,
        "terminated": terminated,
        "term_reason": term_reason,
        "path_y_start_m": float(path_y) if path_y is not None else None,
        "sim_time_s": env.unwrapped._t,
        "actions_phys": ap_arr,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "initial_omega": omega,
        "initial_ac_mm": ac_mm,
    }
