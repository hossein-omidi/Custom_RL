"""Reward functions for the face-milling flexible-plate RL plant.

This file is a minimal update of the previous plate reward module.  The reward
logic remains intentionally simple:

    reward = vibration-gated productivity - vibration cost - action/speed regularization

with generalized terminal penalties. The dense productivity term is safety-gated:
productive cutting is rewarded only when the physical vibration level is controlled

The important consistency change is naming and scaling: the second action is now
interpreted as axial depth of cut ``ap`` [mm], not the old peripheral-milling
``ac`` variable.  Backward-compatible ``ac_*`` keyword aliases are still
accepted so existing experiment configs do not break.

Expected normalized actions
---------------------------
Default face-milling plant action:
    u = [u_omega, u_ap] in [-1, 1]^2

Optional if the plant was created with control_ae=True:
    u = [u_omega, u_ap, u_ae] in [-1, 1]^3

The reward relies strictly on physical displacement/velocity signals provided
in ``info``:
    info["w_sensor"], info["wdot_sensor"]

By default, missing physical sensor signals are treated as a wiring error.  This
prevents silent use of modal coordinates as if they were physical sensor
responses.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # Prefer the new face-milling plant constants when this module exists.
    from custom_rl.plants.plate import (
        OMEGA_MAX_RAD_S,
        OMEGA_MIN_RAD_S,
    )
except Exception:  # Fall back to the package's plate.py if the new file is renamed to plate.py.
    from custom_rl.plants.plate import OMEGA_MAX_RAD_S, OMEGA_MIN_RAD_S

from custom_rl.rewards.base import RewardFn


def _physical_vibration_cost(
    info: dict[str, Any],
    x_next: np.ndarray,
    w_weight: float,
    wdot_weight: float,
    w_scale: float,
    wdot_scale: float,
    w_clip: float | None = None,
    wdot_clip: float | None = None,
    require_physical_info: bool = True,
) -> tuple[float, float, str, float, float]:
    """
    Return (displacement_cost, velocity_cost, source, w_rms, wdot_rms).

    Required source by default:
        info["w_sensor"], info["wdot_sensor"]

    Optional fallback, only when require_physical_info=False:
        x_next modal coordinates [eta, eta_dot]

    For this face-milling project, keep require_physical_info=True so the reward
    remains synchronized with observation and termination, both of which are
    based on physical sensor displacement/velocity.
    """
    x_next = np.asarray(x_next, dtype=np.float64).reshape(-1)

    if "w_sensor" in info and "wdot_sensor" in info:
        w_sensor = np.asarray(info["w_sensor"], dtype=np.float64).reshape(-1)
        wdot_sensor = np.asarray(info["wdot_sensor"], dtype=np.float64).reshape(-1)
        source = "physical_info"
    elif require_physical_info:
        raise KeyError(
            "Reward requires physical sensor signals info['w_sensor'] and "
            "info['wdot_sensor']; refusing modal-state fallback."
        )
    else:
        # Legacy-only fallback. Do not use for the current face-milling plant.
        w_sensor = x_next[0::2]
        wdot_sensor = x_next[1::2]
        source = "modal_fallback"

    if not (np.all(np.isfinite(w_sensor)) and np.all(np.isfinite(wdot_sensor))):
        return float("inf"), float("inf"), source, float("inf"), float("inf")

    w_scale = max(float(w_scale), 1e-12)
    wdot_scale = max(float(wdot_scale), 1e-12)

    if w_clip is not None:
        w_sensor = np.clip(w_sensor, -float(w_clip), float(w_clip))
    if wdot_clip is not None:
        wdot_sensor = np.clip(wdot_sensor, -float(wdot_clip), float(wdot_clip))

    w_rms = float(np.sqrt(np.mean(w_sensor**2)))
    wdot_rms = float(np.sqrt(np.mean(wdot_sensor**2)))

    w_norm = w_sensor / w_scale
    wdot_norm = wdot_sensor / wdot_scale

    w_cost = float(np.mean(w_norm**2))
    wdot_cost = float(np.mean(wdot_norm**2))

    return float(w_weight) * w_cost, float(wdot_weight) * wdot_cost, source, w_rms, wdot_rms


def _productivity_vibration_gate(
    *,
    w_rms: float,
    wdot_rms: float,
    enabled: bool,
    w_gate: float,
    wdot_gate: float,
    wdot_gate_weight: float,
    power: float,
    gate_min: float,
) -> float:
    """Return a smooth safety gate in [gate_min, 1].

    The gate is generalized and state-based, not stage-based:

        G = 1 / (1 + (w_rms/w_gate)^q + beta*(wdot_rms/wdot_gate)^q)

    Thus material-removal productivity is rewarded strongly only while the
    measured physical vibration is controlled.  The gate never changes the hard
    plant termination limit; it only prevents dense productivity reward from
    encouraging cutting while chatter is already developing.
    """
    if not enabled:
        return 1.0

    w_gate = max(float(w_gate), 1e-12)
    wdot_gate = max(float(wdot_gate), 1e-12)
    power = max(float(power), 1.0)
    beta = max(float(wdot_gate_weight), 0.0)
    gate_min = float(np.clip(gate_min, 0.0, 1.0))

    if not (np.isfinite(w_rms) and np.isfinite(wdot_rms)):
        return gate_min

    displacement_term = (abs(float(w_rms)) / w_gate) ** power
    velocity_term = beta * (abs(float(wdot_rms)) / wdot_gate) ** power
    gate = 1.0 / (1.0 + displacement_term + velocity_term)
    return float(np.clip(gate, gate_min, 1.0))


def _safe_progress(info: dict[str, Any]) -> float:
    """Return clipped machining progress in [0, 1].

    The plant reports feed_progress = 0 at the beginning of the pass and
    approximately 0.9 at the configured 90% pass-completion point.  If the
    value is missing, treat the terminal event as an early failure.
    """
    try:
        progress = float(info.get("feed_progress", 0.0))
    except Exception:
        progress = 0.0
    if not np.isfinite(progress):
        progress = 0.0
    return float(np.clip(progress, 0.0, 1.0))


def _pass_completed(info: dict[str, Any]) -> bool:
    """Return True for all pass-completion reason strings used by the plant."""
    reason = str(info.get("termination_reason", ""))
    return bool(info.get("pass_completed") or reason in {"pass_completed", "pass_completed_90percent"})


def _terminal_adjustment(
    *,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
    termination_penalty: float,
    truncation_penalty: float,
    pass_completion_bonus: float,
    failure_progress_penalty_weight: float,
) -> tuple[float, dict[str, float | str]]:
    """Return terminal reward adjustment and diagnostics.

    The terminal penalty is weighted by continuous feed progress, not by
    manually defined machining stages. This avoids rewarding immediate failure
    while keeping the reward generalized over the whole line pass.
    """
    if not (terminated or truncated):
        return 0.0, {
            "terminal_adjustment": 0.0,
            "terminal_bonus": 0.0,
            "terminal_penalty": 0.0,
            "terminal_progress": _safe_progress(info),
            "terminal_status": "running",
        }

    progress = _safe_progress(info)
    if _pass_completed(info):
        bonus = float(pass_completion_bonus)
        return bonus, {
            "terminal_adjustment": bonus,
            "terminal_bonus": bonus,
            "terminal_penalty": 0.0,
            "terminal_progress": progress,
            "terminal_status": "pass_completed",
        }

    base_penalty = float(termination_penalty if terminated else truncation_penalty)
    multiplier = 1.0 + float(failure_progress_penalty_weight) * (1.0 - progress)
    multiplier = max(multiplier, 1.0)
    penalty = base_penalty * multiplier
    status = "terminated_failure" if terminated else "truncated_failure"
    return -penalty, {
        "terminal_adjustment": -penalty,
        "terminal_bonus": 0.0,
        "terminal_penalty": penalty,
        "terminal_progress": progress,
        "terminal_failure_multiplier": multiplier,
        "terminal_status": status,
    }


class DenseProductivePlateReward:
    """Dense reward for productive face milling with vibration suppression.

    The second physical action is axial depth of cut:
        ap [mm]

    For the default 2D action, productivity is a normalized material-removal
    proxy proportional to spindle speed and axial depth:
        raw_productivity_score = omega_score * ap_score
        productivity_score = raw_productivity_score * vibration_gate

    If a 3D action is used, the third action is radial immersion/depth ae [mm],
    and productivity can include ae in the same material-removal proxy:
        raw_productivity_score = omega_score * ap_score * ae_score
        productivity_score = raw_productivity_score * vibration_gate
    """

    def __init__(
        self,
        w_weight: float = 0.6,
        wdot_weight: float = 0.02,
        action_weight: float = 0.0,
        productivity_weight: float = 30.0,
        negative_ap_weight: float = 2.0,
        omega_cost_weight: float = 0.3,
        ap_action_weight: float = 0.0,
        w_scale: float = 7.5e-4,
        wdot_scale: float = 1.0,
        omega_min: float = OMEGA_MIN_RAD_S,
        omega_max: float = OMEGA_MAX_RAD_S,
        ap_min: float = 0.0,
        ap_max: float = 1.0,
        ap_productive_target: float = 0.5,
        ae_min: float = 1.0,
        ae_max: float = 50.0,
        ae_default: float = 25.0,
        include_ae_in_productivity: bool = True,
        productivity_gate_enabled: bool = True,
        productivity_w_gate: float | None = None,
        productivity_wdot_gate: float | None = None,
        productivity_wdot_gate_weight: float = 0.05,
        productivity_gate_power: float = 2.0,
        productivity_gate_min: float = 0.0,
        alive_bonus: float = 0.0,
        termination_penalty: float = 500.0,
        pass_completion_bonus: float = 10000.0,
        truncation_penalty: float | None = None,
        failure_progress_penalty_weight: float = 1.0,
        w_clip: float | None = None,
        wdot_clip: float | None = None,
        require_physical_info: bool = True,
        # Backward-compatible aliases for old configs.
        ac_min: float | None = None,
        ac_max: float | None = None,
        ac_productive_target: float | None = None,
        negative_ac_weight: float | None = None,
        ac_action_weight: float | None = None,
        eta_weight: float | None = None,
        eta_dot_weight: float | None = None,
        eta_scale: float | None = None,
        eta_dot_scale: float | None = None,
    ):
        # Old names -> new face-milling names.
        if ac_min is not None:
            ap_min = ac_min
        if ac_max is not None:
            ap_max = ac_max
        if ac_productive_target is not None:
            ap_productive_target = ac_productive_target
        if negative_ac_weight is not None:
            negative_ap_weight = negative_ac_weight
        if ac_action_weight is not None:
            ap_action_weight = ac_action_weight

        # Old modal names -> physical sensor names.
        if eta_weight is not None:
            w_weight = eta_weight
        if eta_dot_weight is not None:
            wdot_weight = eta_dot_weight
        if eta_scale is not None:
            w_scale = eta_scale
        if eta_dot_scale is not None:
            wdot_scale = eta_dot_scale

        self.w_weight = float(w_weight)
        self.wdot_weight = float(wdot_weight)
        self.action_weight = float(action_weight)
        self.productivity_weight = float(productivity_weight)
        self.negative_ap_weight = float(negative_ap_weight)
        self.omega_cost_weight = float(omega_cost_weight)
        self.ap_action_weight = float(ap_action_weight)

        self.w_scale = float(w_scale)
        self.wdot_scale = float(wdot_scale)
        # Do not clip vibration costs by default.  Clipping at the scale value
        # hides the difference between moderate chatter and severe chatter.
        self.w_clip = None if w_clip is None else float(w_clip)
        self.wdot_clip = None if wdot_clip is None else float(wdot_clip)
        self.require_physical_info = bool(require_physical_info)

        self.omega_min = float(omega_min)
        self.omega_max = float(omega_max)
        self.ap_min = float(ap_min)
        self.ap_max = float(ap_max)
        self.ap_productive_target = float(ap_productive_target)

        self.ae_min = float(ae_min)
        self.ae_max = float(ae_max)
        self.ae_default = float(ae_default)
        self.include_ae_in_productivity = bool(include_ae_in_productivity)

        self.productivity_gate_enabled = bool(productivity_gate_enabled)
        self.productivity_w_gate = float(self.w_scale if productivity_w_gate is None else productivity_w_gate)
        self.productivity_wdot_gate = float(self.wdot_scale if productivity_wdot_gate is None else productivity_wdot_gate)
        self.productivity_wdot_gate_weight = float(productivity_wdot_gate_weight)
        self.productivity_gate_power = float(productivity_gate_power)
        self.productivity_gate_min = float(productivity_gate_min)

        self.alive_bonus = float(alive_bonus)
        self.termination_penalty = float(termination_penalty)
        self.pass_completion_bonus = float(pass_completion_bonus)
        self.truncation_penalty = (
            float(termination_penalty) if truncation_penalty is None else float(truncation_penalty)
        )
        self.failure_progress_penalty_weight = float(failure_progress_penalty_weight)
        self.last_reward_terms: dict[str, float | str] = {}

        if self.omega_max <= self.omega_min:
            raise ValueError("omega_max must be greater than omega_min.")
        if self.ap_max <= self.ap_min:
            raise ValueError("ap_max must be greater than ap_min.")
        if self.ap_productive_target <= 0.0:
            raise ValueError("ap_productive_target must be positive.")
        if self.ae_max <= self.ae_min:
            raise ValueError("ae_max must be greater than ae_min.")
        if self.productivity_w_gate <= 0.0:
            raise ValueError("productivity_w_gate must be positive.")
        if self.productivity_wdot_gate <= 0.0:
            raise ValueError("productivity_wdot_gate must be positive.")
        if self.productivity_gate_power < 1.0:
            raise ValueError("productivity_gate_power must be >= 1.")
        if not (0.0 <= self.productivity_gate_min <= 1.0):
            raise ValueError("productivity_gate_min must be in [0, 1].")

    def _scale_action(self, u: np.ndarray) -> tuple[float, float, float, bool]:
        """Scale normalized action to physical [omega, ap, ae]."""
        u = np.asarray(u, dtype=np.float64).reshape(-1)
        has_ae_action = u.size >= 3

        if u.size < 2:
            u_safe = np.zeros(2, dtype=np.float64)
            u_safe[: u.size] = u
            u = u_safe

        # Always scale first two controls as [omega, ap].
        u2 = np.clip(u[:2], -1.0, 1.0)
        low2 = np.array([self.omega_min, self.ap_min], dtype=np.float64)
        high2 = np.array([self.omega_max, self.ap_max], dtype=np.float64)
        omega, ap = low2 + 0.5 * (u2 + 1.0) * (high2 - low2)

        if has_ae_action:
            u_ae = float(np.clip(u[2], -1.0, 1.0))
            ae = self.ae_min + 0.5 * (u_ae + 1.0) * (self.ae_max - self.ae_min)
        else:
            ae = self.ae_default

        return float(omega), float(ap), float(ae), bool(has_ae_action)

    def __call__(
        self,
        t: float,
        x: np.ndarray,
        u: np.ndarray,
        x_next: np.ndarray,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> float:
        x_next = np.asarray(x_next, dtype=np.float64).reshape(-1)
        u = np.asarray(u, dtype=np.float64).reshape(-1)

        if not np.all(np.isfinite(x_next)):
            return -float(self.termination_penalty)

        w_cost, wdot_cost, signal_source, w_rms, wdot_rms = _physical_vibration_cost(
            info,
            x_next,
            self.w_weight,
            self.wdot_weight,
            self.w_scale,
            self.wdot_scale,
            w_clip=self.w_clip,
            wdot_clip=self.wdot_clip,
            require_physical_info=self.require_physical_info,
        )
        if not np.isfinite(w_cost) or not np.isfinite(wdot_cost):
            return -float(self.termination_penalty)

        omega, ap, ae, has_ae_action = self._scale_action(u)

        omega_score = (omega - self.omega_min) / (self.omega_max - self.omega_min)
        omega_score = float(np.clip(omega_score, 0.0, 1.0))

        # Normalized MRR proxy: for fixed feed/tooth and fixed cutter diameter,
        # face-milling material removal is proportional to omega * ap * ae.
        # Use the full admissible ap range instead of saturating at a target.
        ap_score = (ap - self.ap_min) / max(self.ap_max - self.ap_min, 1e-12)
        ap_score = float(np.clip(ap_score, 0.0, 1.0))

        if has_ae_action and self.include_ae_in_productivity:
            ae_score = (ae - self.ae_min) / (self.ae_max - self.ae_min)
            ae_score = float(np.clip(ae_score, 0.0, 1.0))
        else:
            ae_score = 1.0

        # Normalized MRR proxy for face milling. With fixed ae, this reduces
        # to the omega*ap tradeoff.  The dense productivity term is then gated
        # by the measured physical vibration, so high MRR is rewarded only when
        # chatter is controlled.  This is a continuous state-based gate, not a
        # staged early/late-process rule.
        raw_productivity_score = omega_score * ap_score * ae_score
        productivity_gate = _productivity_vibration_gate(
            w_rms=w_rms,
            wdot_rms=wdot_rms,
            enabled=self.productivity_gate_enabled,
            w_gate=self.productivity_w_gate,
            wdot_gate=self.productivity_wdot_gate,
            wdot_gate_weight=self.productivity_wdot_gate_weight,
            power=self.productivity_gate_power,
            gate_min=self.productivity_gate_min,
        )
        productivity_score = raw_productivity_score * productivity_gate
        productivity_term = self.productivity_weight * productivity_score

        omega_cost = self.omega_cost_weight * (omega_score**2)
        negative_ap = max(-ap, 0.0)
        negative_ap_cost = (negative_ap / max(self.ap_max, 1e-12)) ** 2

        u_clipped = np.clip(u, -1.0, 1.0)
        ap_u = float(u_clipped[1]) if u_clipped.size > 1 else 0.0
        ap_action_cost = self.ap_action_weight * (ap_u**2)
        action_cost = float(np.sum(u_clipped**2))

        reward = (
            self.alive_bonus
            + productivity_term
            - w_cost
            - wdot_cost
            - omega_cost
            - self.negative_ap_weight * negative_ap_cost
            - ap_action_cost
            - self.action_weight * action_cost
        )

        terminal_adjustment, terminal_terms = _terminal_adjustment(
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
            termination_penalty=self.termination_penalty,
            truncation_penalty=self.truncation_penalty,
            pass_completion_bonus=self.pass_completion_bonus,
            failure_progress_penalty_weight=self.failure_progress_penalty_weight,
        )
        reward += terminal_adjustment

        self.last_reward_terms = {
            "vibration_w_cost": float(w_cost),
            "vibration_wdot_cost": float(wdot_cost),
            "vibration_signal_source": signal_source,
            "w_scale_m": float(self.w_scale),
            "wdot_scale_m_s": float(self.wdot_scale),
            "w_rms_m": float(w_rms),
            "wdot_rms_m_s": float(wdot_rms),
            "raw_productivity_score": float(raw_productivity_score),
            "productivity_gate": float(productivity_gate),
            "productivity_score": float(productivity_score),
            "productivity_w_gate_m": float(self.productivity_w_gate),
            "productivity_wdot_gate_m_s": float(self.productivity_wdot_gate),
            "productivity": float(productivity_term),
            "omega_cost": float(omega_cost),
            "negative_ap_cost": float(self.negative_ap_weight * negative_ap_cost),
            "ap_action_cost": float(ap_action_cost),
            "action_cost": float(self.action_weight * action_cost),
            "omega_rad_s": float(omega),
            "ap_mm": float(ap),
            "ae_mm": float(ae),
            **terminal_terms,
        }

        if not np.isfinite(reward):
            reward = -float(self.termination_penalty)

        return float(np.clip(reward, -1e4, 1e4))


class DenseQuadraticPlateReward:
    """Quadratic physical-sensor vibration suppression reward."""

    def __init__(
        self,
        w_weight: float = 1.0,
        wdot_weight: float = 0.1,
        action_weight: float = 0.01,
        w_scale: float = 1e-4,
        wdot_scale: float = 1.0,
        alive_bonus: float = 0.0,
        termination_penalty: float = 500.0,
        pass_completion_bonus: float = 2000.0,
        truncation_penalty: float | None = None,
        failure_progress_penalty_weight: float = 1.0,
        w_clip: float | None = None,
        wdot_clip: float | None = None,
        require_physical_info: bool = True,
        eta_weight: float | None = None,
        eta_dot_weight: float | None = None,
        eta_scale: float | None = None,
        eta_dot_scale: float | None = None,
        **_unused_kwargs: Any,
    ):
        if eta_weight is not None:
            w_weight = eta_weight
        if eta_dot_weight is not None:
            wdot_weight = eta_dot_weight
        if eta_scale is not None:
            w_scale = eta_scale
        if eta_dot_scale is not None:
            wdot_scale = eta_dot_scale

        self.w_weight = float(w_weight)
        self.wdot_weight = float(wdot_weight)
        self.action_weight = float(action_weight)
        self.w_scale = float(w_scale)
        self.wdot_scale = float(wdot_scale)
        self.w_clip = None if w_clip is None else float(w_clip)
        self.wdot_clip = None if wdot_clip is None else float(wdot_clip)
        self.require_physical_info = bool(require_physical_info)
        self.alive_bonus = float(alive_bonus)
        self.termination_penalty = float(termination_penalty)
        self.pass_completion_bonus = float(pass_completion_bonus)
        self.truncation_penalty = (
            float(termination_penalty) if truncation_penalty is None else float(truncation_penalty)
        )
        self.failure_progress_penalty_weight = float(failure_progress_penalty_weight)
        self.last_reward_terms: dict[str, float | str] = {}

    def __call__(
        self,
        t: float,
        x: np.ndarray,
        u: np.ndarray,
        x_next: np.ndarray,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> float:
        x_next = np.asarray(x_next, dtype=np.float64).reshape(-1)
        u = np.asarray(u, dtype=np.float64).reshape(-1)

        if not np.all(np.isfinite(x_next)):
            return -float(self.termination_penalty)

        w_cost, wdot_cost, signal_source, w_rms, wdot_rms = _physical_vibration_cost(
            info,
            x_next,
            self.w_weight,
            self.wdot_weight,
            self.w_scale,
            self.wdot_scale,
            w_clip=self.w_clip,
            wdot_clip=self.wdot_clip,
            require_physical_info=self.require_physical_info,
        )
        if not np.isfinite(w_cost) or not np.isfinite(wdot_cost):
            return -float(self.termination_penalty)

        action_cost = float(np.sum(np.clip(u, -1.0, 1.0) ** 2))
        reward = self.alive_bonus - w_cost - wdot_cost - self.action_weight * action_cost

        terminal_adjustment, terminal_terms = _terminal_adjustment(
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
            termination_penalty=self.termination_penalty,
            truncation_penalty=self.truncation_penalty,
            pass_completion_bonus=self.pass_completion_bonus,
            failure_progress_penalty_weight=self.failure_progress_penalty_weight,
        )
        reward += terminal_adjustment

        self.last_reward_terms = {
            "vibration_w_cost": float(w_cost),
            "vibration_wdot_cost": float(wdot_cost),
            "vibration_signal_source": signal_source,
            "w_scale_m": float(self.w_scale),
            "wdot_scale_m_s": float(self.wdot_scale),
            "w_rms_m": float(w_rms),
            "wdot_rms_m_s": float(wdot_rms),
            "action_cost": float(self.action_weight * action_cost),
            **terminal_terms,
        }

        if not np.isfinite(reward):
            reward = -float(self.termination_penalty)

        return float(np.clip(reward, -1e4, 1e4))


class SparseStablePlateReward:
    """Sparse reward for simple stability testing."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def __call__(
        self,
        t: float,
        x: np.ndarray,
        u: np.ndarray,
        x_next: np.ndarray,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> float:
        return 0.0 if terminated else 1.0


_PLATE_REWARD_REGISTRY: dict[str, type] = {
    "dense": DenseProductivePlateReward,
    "productive": DenseProductivePlateReward,
    "quadratic": DenseQuadraticPlateReward,
    "sparse": SparseStablePlateReward,
}


def register_plate_reward(reward_id: str, reward_cls: type) -> None:
    """Register a custom Plate reward."""
    _PLATE_REWARD_REGISTRY[str(reward_id)] = reward_cls


def get_plate_reward(reward_id: str, **kwargs: Any) -> RewardFn:
    """Return Plate reward by id."""
    if reward_id not in _PLATE_REWARD_REGISTRY:
        raise ValueError(
            f"Unknown reward_id: {reward_id}. Known reward ids: {list(_PLATE_REWARD_REGISTRY)}"
        )
    return _PLATE_REWARD_REGISTRY[reward_id](**kwargs)
