"""Modal state layout for single- and two-field plate plants."""

from __future__ import annotations

import numpy as np

from custom_rl.plants.milling_config import MillingForceConfig


def state_dim_for_model(K: int, cfg: MillingForceConfig) -> int:
    """Return internal state dimension (2K reduced, 4K two-direction)."""
    if cfg.is_feed_normal_full():
        return 4 * int(K)
    return 2 * int(K)


def split_modal_state(
    x: np.ndarray,
    K: int,
    *,
    two_field: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Split state into normal and feed modal components.

    Two-field layout (4K):
        [eta_n, eta_dot_n, eta_f, eta_dot_f] each 2K interleaved.
    Reduced layout (2K):
        returns eta_n, eta_dot_n only; eta_f and eta_dot_f are None.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    K = int(K)
    if two_field:
        block = 2 * K
        if x.size != 4 * K:
            raise ValueError(f"Expected state ({4 * K},), got {x.shape}.")
        x_n = x[:block]
        x_f = x[block:]
        return x_n[0::2], x_n[1::2], x_f[0::2], x_f[1::2]
    if x.size != 2 * K:
        raise ValueError(f"Expected state ({2 * K},), got {x.shape}.")
    return x[0::2], x[1::2], None, None


def pack_modal_derivative(
    etad_n: np.ndarray,
    ddeta_n: np.ndarray,
    etad_f: np.ndarray | None,
    ddeta_f: np.ndarray | None,
    *,
    two_field: bool,
) -> np.ndarray:
    """Assemble dx/dt from subsystem accelerations."""
    dx_n = np.empty(2 * len(etad_n), dtype=np.float64)
    dx_n[0::2] = etad_n
    dx_n[1::2] = ddeta_n
    if not two_field:
        return dx_n
    if etad_f is None or ddeta_f is None:
        raise ValueError("Feed subsystem derivatives required for two-field model.")
    dx_f = np.empty(2 * len(etad_f), dtype=np.float64)
    dx_f[0::2] = etad_f
    dx_f[1::2] = ddeta_f
    return np.concatenate([dx_n, dx_f])
