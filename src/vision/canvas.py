"""Map canvas: accumulation with donut-scoped decay, cleanup, re-anchoring."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from ..config import CanvasConfig
from ..utils.geometry import safe_extract_view


class MapCanvas:
    """Persistent map canvas with donut-scoped decay and integer-snap integration."""

    def __init__(self, cfg: CanvasConfig):
        self.cfg = cfg
        self.data = np.zeros((cfg.canvas_h, cfg.canvas_w), dtype=np.float32)
        self.confidence = np.zeros((cfg.canvas_h, cfg.canvas_w), dtype=np.float32)
        self.center = (cfg.canvas_w / 2.0, cfg.canvas_h / 2.0)
        self.n_cleanups = 0

    def integrate(
        self, obs_f32: np.ndarray, donut_f32: np.ndarray,
        pos_x: float, pos_y: float, sw: int, sh: int,
    ) -> None:
        """Integer-snap donut-scoped integration. No bilinear blur."""
        Hc, Wc = self.data.shape
        cfg = self.cfg
        ix0 = int(round(self.center[0] + pos_x - sw * 0.5))
        iy0 = int(round(self.center[1] + pos_y - sh * 0.5))
        cx0, cy0 = max(0, ix0), max(0, iy0)
        cx1, cy1 = min(Wc, ix0 + sw), min(Hc, iy0 + sh)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        rx0, ry0 = cx0 - ix0, cy0 - iy0
        rw, rh = cx1 - cx0, cy1 - cy0
        patch = self.data[cy0:cy1, cx0:cx1]
        conf_patch = self.confidence[cy0:cy1, cx0:cx1]
        obs_crop = obs_f32[ry0:ry0 + rh, rx0:rx0 + rw]
        donut_crop = donut_f32[ry0:ry0 + rh, rx0:rx0 + rw]
        # Donut-scoped decay: only fade where donut covers
        patch *= 1.0 - donut_crop * (1.0 - cfg.decay)
        patch += obs_crop * cfg.add_weight
        np.clip(patch, 0.0, cfg.clip_max, out=patch)
        conf_patch += donut_crop * cfg.conf_boost

    def cleanup(self, stronger: bool = False) -> None:
        cfg = self.cfg
        ct = cfg.conf_threshold * (1.2 if stronger else 1.0)
        if ct > 0:
            self.data[self.confidence < ct] *= 0.5
        if cfg.morph_ksize >= 2:
            tmp = np.clip(self.data * 255.0, 0, 255).astype(np.uint8)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (cfg.morph_ksize, cfg.morph_ksize))
            self.data[:] = cv2.morphologyEx(tmp, cv2.MORPH_OPEN, k).astype(np.float32) / 255.0
        if cfg.gauss_sigma > 0:
            ks = max(3, int(cfg.gauss_sigma * 4) | 1)
            nz = self.data > 0.01
            if nz.any():
                ys, xs = np.where(nz)
                y0 = max(0, ys.min() - ks)
                y1 = min(self.data.shape[0], ys.max() + ks + 1)
                x0 = max(0, xs.min() - ks)
                x1 = min(self.data.shape[1], xs.max() + ks + 1)
                self.data[y0:y1, x0:x1] = cv2.GaussianBlur(
                    self.data[y0:y1, x0:x1], (ks, ks), cfg.gauss_sigma)
        self.n_cleanups += 1

    def maybe_cleanup(self, frame_idx: int) -> bool:
        if (self.cfg.cleanup_every > 0
                and frame_idx > 0
                and frame_idx % self.cfg.cleanup_every == 0
                and float(self.data.sum()) > 100):
            self.cleanup()
            return True
        return False

    def reanchor(self, pos_x: float, pos_y: float,
                 curr_edges: np.ndarray, sw: int, sh: int
    ) -> Tuple[float, float, bool]:
        cfg = self.cfg
        view = safe_extract_view(
            self.data, (self.center[0] + pos_x, self.center[1] + pos_y), (sw, sh))
        view_u8 = np.clip(view * 255.0, 0, 255).astype(np.uint8)
        ce = cv2.Canny(view_u8, 25, 90)
        if int(ce.sum()) < 500:
            return pos_x, pos_y, False
        h, w = ce.shape
        pad = cfg.reanchor_pad
        search = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.uint8)
        search[pad:pad + h, pad:pad + w] = curr_edges
        res = cv2.matchTemplate(search, ce, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(res)
        if mv < cfg.reanchor_min_score:
            return pos_x, pos_y, False
        pos_x -= float(ml[0] - pad) * cfg.reanchor_damping
        pos_y -= float(ml[1] - pad) * cfg.reanchor_damping
        return pos_x, pos_y, True

    def maybe_reanchor(self, frame_idx: int, pos_x: float, pos_y: float,
                       curr_edges: np.ndarray, sw: int, sh: int
    ) -> Tuple[float, float, bool]:
        if (self.cfg.reanchor_every > 0
                and frame_idx > 0
                and frame_idx % self.cfg.reanchor_every == 0
                and float(self.data.sum()) > 100):
            return self.reanchor(pos_x, pos_y, curr_edges, sw, sh)
        return pos_x, pos_y, False

    def to_u8(self) -> np.ndarray:
        return np.clip(self.data * 255.0, 0, 255).astype(np.uint8)

    def view_at(self, pos_x: float, pos_y: float, sw: int, sh: int) -> np.ndarray:
        return safe_extract_view(
            self.data, (self.center[0] + pos_x, self.center[1] + pos_y), (sw, sh))