"""Spatial helpers: donut mask, shift clamping, canvas view extraction."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def make_donut_mask_u8(h: int, w: int, inner_ratio: float, outer_ratio: float) -> np.ndarray:
    cy, cx = (h - 1) * 0.5, (w - 1) * 0.5
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = min(h, w) * 0.5
    m = (r >= inner_ratio * rmax) & (r <= outer_ratio * rmax)
    return (m.astype(np.uint8) * 255)


def make_donut_rings(
    h: int, w: int, inner_ratio: float, outer_ratio: float, thickness: int = 2
) -> Tuple[np.ndarray, float, float]:
    cy, cx = (h - 1) * 0.5, (w - 1) * 0.5
    rmax = min(h, w) * 0.5
    r_in, r_out = inner_ratio * rmax, outer_ratio * rmax
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(overlay, (int(round(cx)), int(round(cy))), int(round(r_in)),
               (255, 255, 0), thickness, cv2.LINE_AA)
    cv2.circle(overlay, (int(round(cx)), int(round(cy))), int(round(r_out)),
               (255, 0, 255), thickness, cv2.LINE_AA)
    cv2.circle(overlay, (int(round(cx)), int(round(cy))), 3,
               (255, 255, 255), -1, cv2.LINE_AA)
    return overlay, r_in, r_out


def clamp_shift(dx: float, dy: float, max_shift: float) -> Tuple[float, float]:
    mag = float((dx * dx + dy * dy) ** 0.5)
    if max_shift > 0 and mag > max_shift and mag > 1e-9:
        s = max_shift / mag
        return dx * s, dy * s
    return dx, dy


def safe_extract_view(
    canvas: np.ndarray,
    center_xy: Tuple[float, float],
    size_wh: Tuple[int, int],
) -> np.ndarray:
    Hc, Wc = canvas.shape
    w, h = size_wh
    cx, cy = center_xy
    x0 = int(round(cx - w / 2))
    y0 = int(round(cy - h / 2))
    out = np.zeros((h, w), dtype=canvas.dtype)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(Wc, x0 + w), min(Hc, y0 + h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy0 - y0 + sy1 - sy0,
            sx0 - x0:sx0 - x0 + sx1 - sx0] = canvas[sy0:sy1, sx0:sx1]
    return out