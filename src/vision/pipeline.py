from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from ..config import BenchmarkConfig, ClusterConfig, ColorFilterConfig, OutputConfig, ProcessingConfig
from .color_distance import (
    DistBuffers,
    ensure_dist_buffers,
    make_luts_bgr,
    parse_hex_list,
    rgb_targets_to_bgr,
    compute_min_dist2_lut_gate_bgr,
)
from .clustering import cluster_cc_then_expand, cluster_dbscan_then_expand, ClusterResult


@dataclass
class FrameMetrics:
    proc_ms: float
    dist_ms: float
    cluster_ms: float
    strict_points: int
    used_points: int
    method: str


class FilterPipeline:
    """Frame processing pipeline (BGR in -> BGR out + metrics)."""

    def __init__(
        self,
        processing: ProcessingConfig,
        color: ColorFilterConfig,
        cluster: ClusterConfig,
        output: OutputConfig,
        bench: BenchmarkConfig | None = None,
    ):
        self.processing = processing
        self.color = color
        self.cluster_cfg = cluster
        self.output = output
        self.bench = bench or BenchmarkConfig()

        self.proc_w, self.proc_h = processing.proc_size
        self.crop = processing.crop
        self.scale = float(processing.proc_scale)
        if not (0 < self.scale <= 1.0):
            raise ValueError("proc_scale must be in (0,1]")

        targets_rgb = parse_hex_list(color.targets_hex)
        self.targets_bgr = rgb_targets_to_bgr(targets_rgb)
        self.luts = make_luts_bgr(self.targets_bgr)

        self.threshold = float(color.threshold)
        self.mult = float(color.dbscan_multiplier)
        self.gate = self.threshold * self.mult

        self.crop = self.crop.clamp(self.proc_w, self.proc_h)
        self.roi_w = self.crop.w
        self.roi_h = self.crop.h
        self.small_w = max(1, int(round(self.roi_w * self.scale)))
        self.small_h = max(1, int(round(self.roi_h * self.scale)))

        self.eps_spatial_small = (self.threshold * self.mult) * self.scale
        self.roi_pad_small = max(1, int(round(cluster.roi_pad * self.scale)))

        ps2 = self.scale * self.scale
        self.cc_fallback_scaled = int(cluster.cc_fallback_points * ps2)
        self.cc_min_area_scaled = max(1, int(round(cluster.cc_min_area * ps2)))
        self.min_samples_scaled = max(2, int(round(cluster.min_samples * ps2)))
        self.target_points_scaled = max(200, int(round(cluster.dbscan_target_points * ps2)))
        self.max_points_scaled = max(500, int(round(cluster.dbscan_max_points * ps2)))

        self.dist_buffers: DistBuffers = ensure_dist_buffers(self.small_h, self.small_w)

        self.mask_full_u8 = np.zeros((self.proc_h, self.proc_w), dtype=np.uint8)
        out_w, out_h = self.output.output_size
        self.disp_left = np.empty((out_h, out_w // 2, 3), dtype=np.uint8)
        self.disp_right = np.empty((out_h, out_w // 2, 3), dtype=np.uint8)
        self.out_frame = np.empty((out_h, out_w, 3), dtype=np.uint8)

        self.rng = np.random.default_rng(12345)

    def _ensure_proc_size(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        if (w, h) == (self.proc_w, self.proc_h):
            return frame_bgr
        return cv2.resize(frame_bgr, (self.proc_w, self.proc_h), interpolation=cv2.INTER_AREA)

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, FrameMetrics]:
        t0 = cv2.getTickCount()

        frame = self._ensure_proc_size(frame_bgr)

        x0, y0, x1, y1 = self.crop.x0, self.crop.y0, self.crop.x1, self.crop.y1
        roi = frame[y0:y1, x0:x1]

        if self.scale != 1.0:
            roi_small = cv2.resize(roi, (self.small_w, self.small_h), interpolation=cv2.INTER_AREA)
        else:
            roi_small = roi

        t_dist0 = cv2.getTickCount()
        min_dist2 = compute_min_dist2_lut_gate_bgr(
            roi_small,
            targets_bgr=self.targets_bgr,
            luts=self.luts,
            gate=self.gate,
            buffers=self.dist_buffers,
        )
        t_dist1 = cv2.getTickCount()
        dist_ms = (t_dist1 - t_dist0) * 1000.0 / cv2.getTickFrequency()

        thr2 = self.threshold * self.threshold
        strict_points = int((min_dist2 <= thr2).sum())

        t_cl0 = cv2.getTickCount()
        if self.cluster_cfg.use_cc_fallback and strict_points > self.cc_fallback_scaled:
            cr: ClusterResult = cluster_cc_then_expand(
                min_dist2_roi=min_dist2,
                threshold=self.threshold,
                mult=self.mult,
                roi_pad=self.roi_pad_small,
                min_component_area=self.cc_min_area_scaled,
            )
        else:
            cr = cluster_dbscan_then_expand(
                min_dist2_roi=min_dist2,
                threshold=self.threshold,
                mult=self.mult,
                eps_spatial=self.eps_spatial_small,
                min_samples=self.min_samples_scaled,
                roi_pad=self.roi_pad_small,
                base_stride=self.cluster_cfg.dbscan_stride,
                target_points=self.target_points_scaled,
                max_points=self.max_points_scaled,
                dbscan_algorithm=self.cluster_cfg.dbscan_algorithm,
                leaf_size=self.cluster_cfg.leaf_size,
                rng=self.rng,
            )
        t_cl1 = cv2.getTickCount()
        cluster_ms = (t_cl1 - t_cl0) * 1000.0 / cv2.getTickFrequency()

        mask_small_bool = cr.mask

        if self.scale != 1.0:
            mask_roi_u8 = cv2.resize(
                (mask_small_bool.astype(np.uint8) * 255),
                (self.roi_w, self.roi_h),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            mask_roi_u8 = (mask_small_bool.astype(np.uint8) * 255)

        self.mask_full_u8.fill(0)
        self.mask_full_u8[y0:y1, x0:x1] = mask_roi_u8

        out = self._render_output(frame)

        t1 = cv2.getTickCount()
        proc_ms = (t1 - t0) * 1000.0 / cv2.getTickFrequency()

        metrics = FrameMetrics(
            proc_ms=proc_ms,
            dist_ms=dist_ms,
            cluster_ms=cluster_ms,
            strict_points=strict_points,
            used_points=cr.used_points,
            method=cr.method,
        )
        return out, metrics

    def _render_output(self, frame_proc_bgr: np.ndarray) -> np.ndarray:
        mode = self.output.output_mode
        out_w, out_h = self.output.output_size

        if mode == "mask":
            mask_bgr = cv2.cvtColor(self.mask_full_u8, cv2.COLOR_GRAY2BGR)
            if (mask_bgr.shape[1], mask_bgr.shape[0]) != (out_w, out_h):
                cv2.resize(mask_bgr, (out_w, out_h), dst=self.out_frame, interpolation=cv2.INTER_NEAREST)
                return self.out_frame
            self.out_frame[:] = mask_bgr
            return self.out_frame

        if mode == "overlay":
            base = frame_proc_bgr
            if (base.shape[1], base.shape[0]) != (out_w, out_h):
                base = cv2.resize(base, (out_w, out_h), interpolation=cv2.INTER_AREA)
            out = self.out_frame
            out[:] = base

            m = self.mask_full_u8
            if m.shape[:2] != (out_h, out_w):
                m = cv2.resize(m, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            out[m > 0, 1] = 255  # green channel
            return out

        # sbs
        half_w = out_w // 2
        cv2.resize(frame_proc_bgr, (half_w, out_h), dst=self.disp_left, interpolation=cv2.INTER_AREA)
        mask_half = cv2.resize(self.mask_full_u8, (half_w, out_h), interpolation=cv2.INTER_NEAREST)
        cv2.cvtColor(mask_half, cv2.COLOR_GRAY2BGR, dst=self.disp_right)

        self.out_frame[:, :half_w] = self.disp_left
        self.out_frame[:, half_w : half_w * 2] = self.disp_right
        return self.out_frame