"""Reward function protocol and utilities."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class RewardFn(Protocol):
    """
    Protocol for reward functions.
    Callable: (t, x, u, x_next, terminated, truncated, info) -> reward
    """

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
        ...
