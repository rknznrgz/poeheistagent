"""Temporal tracking: adaptive velocity EMA and per-pixel running EMA."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VelocityTracker:
    """Adaptive-alpha EMA velocity tracker."""

    dx: float = 0.0
    dy: float = 0.0
    min_alpha: float = 0.15
    max_alpha: float = 0.85
    n: int = 0

    def update(self, new_dx: float, new_dy: float) -> None:
        if self.n == 0:
            self.dx = new_dx
            self.dy = new_dy
            self.n = 1
            return
        err = float(((new_dx - self.dx) ** 2 + (new_dy - self.dy) ** 2) ** 0.5)
        norm_err = err / max(float((self.dx ** 2 + self.dy ** 2) ** 0.5), 1.0)
        t = min(norm_err * 2.0, 3.0)
        alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * (t / (1.0 + t))
        self.dx = (1 - alpha) * self.dx + alpha * new_dx
        self.dy = (1 - alpha) * self.dy + alpha * new_dy
        self.n += 1

    @property
    def speed(self) -> float:
        return float((self.dx ** 2 + self.dy ** 2) ** 0.5)


class RunningEMA:
    """Per-pixel exponential moving average.

    Replaces temporal-median buffer. Zero allocation after init.
    alpha=1.0 means no smoothing (raw frame), alpha=0.3 means strong smoothing.
    """

    __slots__ = ("acc", "alpha", "n")

    def __init__(self, h: int, w: int, alpha: float = 0.4):
        self.acc = np.zeros((h, w), dtype=np.float32)
        self.alpha = alpha
        self.n = 0

    def update(self, obs_f32: np.ndarray) -> np.ndarray:
        if self.n == 0:
            np.copyto(self.acc, obs_f32)
        else:
            cv2.accumulateWeighted(obs_f32, self.acc, self.alpha)
        self.n += 1
        return self.acc