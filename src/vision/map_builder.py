"""MapBuilder: full flow-tracking pipeline that builds a map from minimap video."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np

from ..config import MapBuilderConfig, FlowConfig, CanvasConfig, VelocityConfig
from ..utils.geometry import clamp_shift, make_donut_mask_u8
from .canvas import MapCanvas
from .flow import build_edges_u8, multi_pass_flow, MultiPassResult
from .tracker import RunningEMA, VelocityTracker


def _ensure_gray_u8(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame.astype(np.uint8, copy=False)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _threshold_mask(gray_u8: np.ndarray, thr: int) -> np.ndarray:
    _, bw = cv2.threshold(gray_u8, thr, 255, cv2.THRESH_BINARY)
    return bw


def _denoise_mask(bw_u8: np.ndarray, morph_ksize: int) -> np.ndarray:
    if morph_ksize < 2:
        return bw_u8
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
    return cv2.morphologyEx(cv2.morphologyEx(bw_u8, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)


# -- Frame preloading ------------------------------------------------

def preload_frames(
    input_path: str, frame_w: int, frame_h: int, crop: str, mask_thr: int,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], float]:
    """Load all frames into memory as (roi_gray, roi_bw) pairs. Returns (frames, fps)."""
    x0, y0, x1, y1 = map(int, crop.split(","))
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {input_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames: List[Tuple[np.ndarray, np.ndarray]] = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if frame.shape[1] != frame_w or frame.shape[0] != frame_h:
            frame = cv2.resize(frame, (frame_w, frame_h), interpolation=cv2.INTER_AREA)
        gray = _ensure_gray_u8(frame)
        bw = _threshold_mask(gray, mask_thr)
        frames.append((gray[y0:y1, x0:x1].copy(), bw[y0:y1, x0:x1].copy()))
    cap.release()
    return frames, fps


# -- MapBuilder -------------------------------------------------------

@dataclass
class MapBuildResult:
    canvas_u8: np.ndarray
    ok_flow: int
    fail: int
    total_frames: int
    flow_rate: float


class MapBuilder:
    """Runs the full flow-tracking map-accumulation pipeline.

    Can be used in two ways:
      1. Feed pre-loaded (roi_gray, roi_bw) tuples via run().
      2. Feed raw frames one at a time via step() for streaming use.
    """

    def __init__(self, cfg: MapBuilderConfig):
        self.cfg = cfg
        fc = cfg.flow
        cc = cfg.canvas
        proc = cfg.processing

        # Crop
        cr = proc.crop
        self.roi_w = cr.w
        self.roi_h = cr.h

        # Scaled dimensions
        self.sw = max(1, int(round(self.roi_w * fc.scale)))
        self.sh = max(1, int(round(self.roi_h * fc.scale)))

        # Donut mask
        self.donut = make_donut_mask_u8(self.sh, self.sw, fc.inner_ratio, fc.outer_ratio)
        self.donut_f32 = self.donut.astype(np.float32) / 255.0

        # Max shift at scale
        self.max_shift_scaled = fc.max_shift_full * fc.scale

        # Temporal EMA buffer
        self.ema_buf: Optional[RunningEMA] = (
            RunningEMA(self.sh, self.sw, alpha=cc.ema_alpha)
            if cc.ema_alpha < 1.0 else None
        )

        # Canvas
        self.canvas = MapCanvas(cc)

        # Velocity tracker
        self.vel = VelocityTracker(
            min_alpha=cfg.velocity.min_alpha, max_alpha=cfg.velocity.max_alpha,
        )

        # State
        self.prev_flow_img: Optional[np.ndarray] = None
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.ok_flow = 0
        self.fail = 0
        self.consecutive_fails = 0
        self.frame_idx = 0

    def _prepare_frame(self, roi_gray: np.ndarray, roi_bw: np.ndarray):
        """Scale and mask a single ROI pair. Returns (flow_img, edges, bw_clean)."""
        fc = self.cfg.flow
        cc = self.cfg.canvas
        sw, sh = self.sw, self.sh

        gray_small = cv2.bitwise_and(
            cv2.resize(roi_gray, (sw, sh), interpolation=cv2.INTER_AREA), self.donut)
        bw_small = cv2.bitwise_and(
            cv2.resize(roi_bw, (sw, sh), interpolation=cv2.INTER_AREA), self.donut)
        bw_clean = _denoise_mask(bw_small, cc.input_morph_ksize)

        if fc.flow_input == "gray":
            flow_img = gray_small
        elif fc.flow_input == "dist_transform":
            _, bw_bin = cv2.threshold(bw_clean, 127, 255, cv2.THRESH_BINARY)
            dt = cv2.distanceTransform(bw_bin, cv2.DIST_L2, 5)
            dt = cv2.normalize(dt, None, 0, 255, cv2.NORM_MINMAX)
            flow_img = cv2.bitwise_and(dt.astype(np.uint8), self.donut)
        else:
            flow_img = bw_clean

        edges = build_edges_u8(
            flow_img, self.donut, fc.blur, fc.edge_low, fc.edge_high, fc.edge_dilate)
        return flow_img, edges, bw_clean

    def step(self, roi_gray: np.ndarray, roi_bw: np.ndarray) -> Optional[MultiPassResult]:
        """Process one frame. Returns the MultiPassResult (or None for first frame)."""
        fc = self.cfg.flow
        cc = self.cfg.canvas
        i = self.frame_idx

        flow_img, edges, bw_clean = self._prepare_frame(roi_gray, roi_bw)

        integrate = False
        used_dx = 0.0
        used_dy = 0.0
        mp_res: Optional[MultiPassResult] = None

        if self.prev_flow_img is not None:
            pdx, pdy = self.vel.dx, self.vel.dy
            if self.consecutive_fails > 2:
                df = max(0.0, 1.0 - self.consecutive_fails * 0.2)
                pdx *= df
                pdy *= df

            mp_res = multi_pass_flow(
                self.prev_flow_img, flow_img, edges, pdx, pdy,
                fc, self.max_shift_scaled,
            )

            if mp_res.ok:
                used_dx, used_dy = mp_res.dx, mp_res.dy
                integrate = True
                self.ok_flow += 1
                self.consecutive_fails = 0
            else:
                self.consecutive_fails += 1
                if self.cfg.on_fail == "predict" and self.vel.n > 0:
                    df = max(0.1, 1.0 - self.consecutive_fails * 0.15)
                    used_dx, used_dy = clamp_shift(
                        self.vel.dx * df, self.vel.dy * df, self.max_shift_scaled)
                self.fail += 1

        # Update position
        if self.cfg.invert_update:
            self.pos_x -= used_dx
            self.pos_y -= used_dy
        else:
            self.pos_x += used_dx
            self.pos_y += used_dy

        # Update velocity & integrate into canvas
        if integrate:
            self.vel.update(used_dx, used_dy)
            obs_raw = bw_clean.astype(np.float32) / 255.0
            obs = self.ema_buf.update(obs_raw) if self.ema_buf else obs_raw
            self.canvas.integrate(obs, self.donut_f32, self.pos_x, self.pos_y, self.sw, self.sh)

        # Canvas cleanup
        self.canvas.maybe_cleanup(i)

        # Re-anchor
        re_edges = build_edges_u8(
            bw_clean, self.donut, fc.blur, fc.edge_low, fc.edge_high, fc.edge_dilate)
        self.pos_x, self.pos_y, _ = self.canvas.maybe_reanchor(
            i, self.pos_x, self.pos_y, re_edges, self.sw, self.sh)

        self.prev_flow_img = flow_img
        self.frame_idx += 1
        return mp_res

    def run(self, raw_frames: List[Tuple[np.ndarray, np.ndarray]]) -> MapBuildResult:
        """Run pipeline on a list of pre-loaded (roi_gray, roi_bw) frames."""
        for roi_gray, roi_bw in raw_frames:
            self.step(roi_gray, roi_bw)

        # Final cleanup
        if float(self.canvas.data.sum()) > 100:
            self.canvas.cleanup(stronger=True)

        total = len(raw_frames)
        return MapBuildResult(
            canvas_u8=self.canvas.to_u8(),
            ok_flow=self.ok_flow,
            fail=self.fail,
            total_frames=total,
            flow_rate=self.ok_flow / max(total - 1, 1),
        )