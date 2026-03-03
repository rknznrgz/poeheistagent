from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    s = hex_str.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) != 6:
        raise ValueError(f"Bad hex color '{hex_str}'. Expected like #RRGGBB")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def parse_hex_list(s: str) -> List[Tuple[int, int, int]]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("No target colors parsed")
    return [hex_to_rgb(p) for p in parts]


def rgb_targets_to_bgr(targets_rgb: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    return [(b, g, r) for (r, g, b) in targets_rgb]


def make_luts_bgr(targets_bgr: List[Tuple[int, int, int]]):
    x = np.arange(256, dtype=np.int32)
    luts = []
    for (bt, gt, rt) in targets_bgr:
        lB = (x - int(bt)) * (x - int(bt))
        lG = (x - int(gt)) * (x - int(gt))
        lR = (x - int(rt)) * (x - int(rt))
        luts.append((lB, lG, lR))
    return luts


@dataclass
class DistBuffers:
    out: np.ndarray
    cand: np.ndarray
    ab: np.ndarray


def ensure_dist_buffers(h: int, w: int, existing: DistBuffers | None = None) -> DistBuffers:
    if existing is not None:
        if (
            existing.out.shape == (h, w)
            and existing.out.dtype == np.int32
            and existing.cand.shape == (h, w)
            and existing.cand.dtype == np.bool_
            and existing.ab.shape == (h, w)
            and existing.ab.dtype == np.int16
        ):
            return existing
    return DistBuffers(
        out=np.empty((h, w), dtype=np.int32),
        cand=np.empty((h, w), dtype=bool),
        ab=np.empty((h, w), dtype=np.int16),
    )


def compute_min_dist2_lut_gate_bgr(
    roi_bgr_u8: np.ndarray,
    targets_bgr: List[Tuple[int, int, int]],
    luts,
    gate: float,
    buffers: DistBuffers,
) -> np.ndarray:
    """
    Exact min squared distance to any target in BGR, with per-channel gating.
    Non-candidate pixels get BIG value. Reuses buffers to avoid allocations.
    """
    H, W = roi_bgr_u8.shape[:2]
    BIG = np.int32(2_147_483_647)
    gate_i = int(np.ceil(gate))

    out = buffers.out
    cand = buffers.cand
    ab = buffers.ab

    out[:] = BIG
    cand[:] = False

    ch_b, ch_g, ch_r = cv2.split(roi_bgr_u8)
    B16 = ch_b.astype(np.int16, copy=False)
    G16 = ch_g.astype(np.int16, copy=False)
    R16 = ch_r.astype(np.int16, copy=False)

    for (bt, gt, rt) in targets_bgr:
        np.subtract(B16, bt, out=ab)
        np.abs(ab, out=ab)
        mask = ab <= gate_i

        np.subtract(G16, gt, out=ab)
        np.abs(ab, out=ab)
        mask &= (ab <= gate_i)

        np.subtract(R16, rt, out=ab)
        np.abs(ab, out=ab)
        mask &= (ab <= gate_i)

        cand |= mask

    ys, xs = np.where(cand)
    if xs.size == 0:
        return out

    bc = ch_b[ys, xs]
    gc = ch_g[ys, xs]
    rc = ch_r[ys, xs]

    lB, lG, lR = luts[0]
    min_d2 = lB[bc] + lG[gc] + lR[rc]
    for (lB, lG, lR) in luts[1:]:
        d2 = lB[bc] + lG[gc] + lR[rc]
        np.minimum(min_d2, d2, out=min_d2)

    out[ys, xs] = min_d2.astype(np.int32, copy=False)
    return out