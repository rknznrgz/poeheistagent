"""Optical flow computation: Farneback, robust global flow, multi-pass strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from ..config import FlowConfig
from ..utils.geometry import clamp_shift


def median_mad(arr: np.ndarray) -> Tuple[float, float]:
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return med, mad


@dataclass
class FlowResult:
    ok: bool
    dx: float
    dy: float
    inlier_frac: float
    n_pts: int
    reason: str
    inlier_xs: Optional[np.ndarray] = None
    inlier_ys: Optional[np.ndarray] = None
    inlier_vx: Optional[np.ndarray] = None
    inlier_vy: Optional[np.ndarray] = None


@dataclass
class MultiPassResult:
    ok: bool
    dx: float
    dy: float
    inlier_frac: float
    n_pts: int
    pass_name: str
    predict_dx: float
    predict_dy: float
    residual_dx: float
    residual_dy: float
    flow_res: Optional[FlowResult] = None


# -- Low-level helpers ------------------------------------------------

def warp_translate(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(
        img, M, (w, h),
        borderMode=cv2.BORDER_CONSTANT, borderValue=0, flags=cv2.INTER_LINEAR,
    )


def build_edges_u8(
    img_u8: np.ndarray, donut_u8: np.ndarray,
    blur: int, edge_low: int, edge_high: int, dilate_it: int,
) -> np.ndarray:
    x = cv2.GaussianBlur(img_u8, (blur, blur), 0) if blur >= 3 and blur % 2 == 1 else img_u8
    e = cv2.bitwise_and(cv2.Canny(x, edge_low, edge_high), donut_u8)
    if dilate_it > 0:
        e = cv2.dilate(e, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=dilate_it)
    return e


def calc_flow_fb(
    prev_u8, curr_u8,
    pyr_scale, levels, winsize, iterations, poly_n, poly_sigma,
    init_flow=None,
):
    flags = 0
    flow_init = None
    if init_flow is not None:
        flags = cv2.OPTFLOW_USE_INITIAL_FLOW
        flow_init = init_flow.astype(np.float32, copy=True)
    return cv2.calcOpticalFlowFarneback(
        prev_u8, curr_u8, flow_init,
        pyr_scale=pyr_scale, levels=levels, winsize=winsize,
        iterations=iterations, poly_n=poly_n, poly_sigma=poly_sigma,
        flags=flags,
    )


# -- Robust global flow -----------------------------------------------

def robust_global_flow(
    flow: np.ndarray, sel_mask_u8: np.ndarray,
    max_points: int, k_mad: float, base_tol: float,
    min_points: int, min_inlier_frac: float, max_mag: float,
    keep_inliers: bool = False,
) -> FlowResult:
    ys, xs = np.where(sel_mask_u8 > 0)
    n = xs.size
    if n < min_points:
        return FlowResult(False, 0, 0, 0, n, "few_points")
    if max_points > 0 and n > max_points:
        idx = np.random.choice(n, size=max_points, replace=False)
        ys, xs = ys[idx], xs[idx]
        n = xs.size
    v = flow[ys, xs]
    vx, vy = v[:, 0], v[:, 1]
    mx, my = float(np.median(vx)), float(np.median(vy))
    dev = np.sqrt((vx - mx) ** 2 + (vy - my) ** 2)
    _, dmad = median_mad(dev)
    tol = max(base_tol, k_mad * dmad)
    inl = dev <= tol
    inlier_frac = float(inl.mean())
    if inlier_frac < min_inlier_frac:
        return FlowResult(False, 0, 0, inlier_frac, n, "low_inlier_frac")
    vx_i, vy_i = vx[inl], vy[inl]
    if vx_i.size < min_points:
        return FlowResult(False, 0, 0, inlier_frac, n, "few_inliers")
    dx, dy = clamp_shift(float(np.median(vx_i)), float(np.median(vy_i)), max_mag)
    i_xs = i_ys = i_vx = i_vy = None
    if keep_inliers:
        i_xs, i_ys, i_vx, i_vy = xs[inl], ys[inl], vx_i, vy_i
    return FlowResult(True, dx, dy, inlier_frac, n, "ok", i_xs, i_ys, i_vx, i_vy)


def _try_flow_pass(
    prev_u8, curr_u8, edges,
    pyr_scale, levels, winsize, iterations, poly_n, poly_sigma,
    max_points, k_mad, base_tol, min_points, min_inlier_frac, max_mag,
    keep_inliers, init_flow=None,
) -> FlowResult:
    flow = calc_flow_fb(
        prev_u8, curr_u8, pyr_scale, levels, winsize,
        iterations, poly_n, poly_sigma, init_flow,
    )
    return robust_global_flow(
        flow, edges, max_points, k_mad, base_tol,
        min_points, min_inlier_frac, max_mag, keep_inliers,
    )


# -- Multi-pass flow --------------------------------------------------

def multi_pass_flow(
    prev_img: np.ndarray,
    curr_img: np.ndarray,
    edges: np.ndarray,
    predict_dx: float,
    predict_dy: float,
    cfg: FlowConfig,
    max_shift_scaled: float,
    keep_inliers: bool = False,
) -> MultiPassResult:
    """Three-pass strategy: warped -> seeded -> boosted -> fail."""
    h, w = curr_img.shape
    pmag = float((predict_dx ** 2 + predict_dy ** 2) ** 0.5)
    has_pred = pmag > 0.3

    fa = (
        cfg.fb_pyr_scale, cfg.fb_levels, cfg.fb_winsize,
        cfg.fb_iterations, cfg.fb_poly_n, cfg.fb_poly_sigma,
        cfg.max_points, cfg.k_mad, cfg.base_tol,
        cfg.min_points, cfg.min_inlier_frac, max_shift_scaled, keep_inliers,
    )

    # Pass 1: warped
    if has_pred:
        cw = warp_translate(curr_img, predict_dx, predict_dy)
        ew = warp_translate(edges, predict_dx, predict_dy)
        res = _try_flow_pass(prev_img, cw, ew, *fa)
        if res.ok:
            tdx, tdy = clamp_shift(predict_dx + res.dx, predict_dy + res.dy, max_shift_scaled)
            return MultiPassResult(
                True, tdx, tdy, res.inlier_frac, res.n_pts,
                "warped", predict_dx, predict_dy, res.dx, res.dy, res,
            )

    # Pass 2: seeded
    if has_pred:
        init_flow = np.full((h, w, 2), [predict_dx, predict_dy], dtype=np.float32)
        res = _try_flow_pass(prev_img, curr_img, edges, *fa, init_flow=init_flow)
        if res.ok:
            return MultiPassResult(
                True, res.dx, res.dy, res.inlier_frac, res.n_pts,
                "seeded", predict_dx, predict_dy,
                res.dx - predict_dx, res.dy - predict_dy, res,
            )

    # Pass 3: boosted (wider window, more levels)
    bl = min(cfg.fb_levels + 2, 7)
    bws = min(cfg.fb_winsize + 10, 41) | 1
    res = _try_flow_pass(
        prev_img, curr_img, edges,
        cfg.fb_pyr_scale, bl, bws, cfg.fb_iterations + 1,
        cfg.fb_poly_n, cfg.fb_poly_sigma,
        cfg.max_points, cfg.k_mad, cfg.base_tol,
        cfg.min_points, cfg.min_inlier_frac, max_shift_scaled, keep_inliers,
    )
    if res.ok:
        return MultiPassResult(
            True, res.dx, res.dy, res.inlier_frac, res.n_pts,
            "boosted", predict_dx, predict_dy, res.dx, res.dy, res,
        )

    return MultiPassResult(
        False, 0, 0, 0, res.n_pts if res else 0,
        "fail", predict_dx, predict_dy, 0, 0, res,
    )