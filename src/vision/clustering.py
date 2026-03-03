from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from sklearn.cluster import DBSCAN


def adaptive_subsample(xs: np.ndarray, ys: np.ndarray, base_stride: int, target_points: int):
    """Safe subsample: keep points where (x+y)%stride==0, stride adapts by point count."""
    n = xs.size
    stride = max(1, int(base_stride))
    if target_points > 0 and n > target_points:
        stride = max(stride, int(np.ceil(n / target_points)))
    if stride == 1:
        return xs, ys, stride
    keep = ((xs + ys) % stride) == 0
    return xs[keep], ys[keep], stride


def expand_from_bboxes(min_dist2: np.ndarray, expanded: np.ndarray, bboxes: List[Tuple[int, int, int, int]], loose_thr2: float, pad: int):
    H, W = min_dist2.shape
    for (x, y, w, h) in bboxes:
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(W, x + w + pad)
        y1 = min(H, y + h + pad)
        roi = min_dist2[y0:y1, x0:x1]
        expanded[y0:y1, x0:x1] |= (roi <= loose_thr2)


@dataclass
class ClusterResult:
    mask: np.ndarray  # bool mask
    strict_points: int
    used_points: int
    method: str
    cluster_ms: float


def cluster_dbscan_then_expand(
    min_dist2_roi: np.ndarray,
    threshold: float,
    mult: float,
    eps_spatial: float,
    min_samples: int,
    roi_pad: int,
    base_stride: int,
    target_points: int,
    max_points: int,
    dbscan_algorithm: str,
    leaf_size: int,
    rng: np.random.Generator,
) -> ClusterResult:
    thr2 = threshold * threshold
    strict = (min_dist2_roi <= thr2)
    ys, xs = np.where(strict)
    strict_points = int(xs.size)
    if strict_points == 0:
        return ClusterResult(mask=strict, strict_points=0, used_points=0, method="DBSCAN", cluster_ms=0.0)

    xs_s, ys_s, _ = adaptive_subsample(xs, ys, base_stride=base_stride, target_points=target_points)

    if max_points > 0 and xs_s.size > max_points:
        idx = rng.choice(xs_s.size, size=max_points, replace=False)
        xs_s, ys_s = xs_s[idx], ys_s[idx]

    used_points = int(xs_s.size)
    if used_points == 0:
        return ClusterResult(mask=strict, strict_points=strict_points, used_points=0, method="DBSCAN", cluster_ms=0.0)

    coords = np.column_stack([xs_s, ys_s]).astype(np.float32)

    db = DBSCAN(
        eps=float(eps_spatial),
        min_samples=int(min_samples),
        metric="euclidean",
        n_jobs=-1,
        algorithm=str(dbscan_algorithm),
        leaf_size=int(leaf_size),
    )

    t0 = cv2.getTickCount()
    labels = db.fit_predict(coords)
    t1 = cv2.getTickCount()
    cluster_ms = (t1 - t0) * 1000.0 / cv2.getTickFrequency()

    loose_thr2 = (threshold * mult) ** 2
    expanded = strict.copy()

    labs = [lab for lab in np.unique(labels) if lab != -1]
    if not labs:
        return ClusterResult(mask=expanded, strict_points=strict_points, used_points=used_points, method="DBSCAN", cluster_ms=cluster_ms)

    bboxes = []
    for lab in labs:
        idx = np.where(labels == lab)[0]
        if idx.size == 0:
            continue
        x_min = int(xs_s[idx].min()); x_max = int(xs_s[idx].max())
        y_min = int(ys_s[idx].min()); y_max = int(ys_s[idx].max())
        bboxes.append((x_min, y_min, x_max - x_min + 1, y_max - y_min + 1))

    expand_from_bboxes(min_dist2_roi, expanded, bboxes, loose_thr2=loose_thr2, pad=int(roi_pad))
    return ClusterResult(mask=expanded, strict_points=strict_points, used_points=used_points, method="DBSCAN", cluster_ms=cluster_ms)


def cluster_cc_then_expand(
    min_dist2_roi: np.ndarray,
    threshold: float,
    mult: float,
    roi_pad: int,
    min_component_area: int,
) -> ClusterResult:
    thr2 = threshold * threshold
    strict = (min_dist2_roi <= thr2)
    strict_points = int(strict.sum())
    if strict_points == 0:
        return ClusterResult(mask=strict, strict_points=0, used_points=0, method="CC", cluster_ms=0.0)

    strict_u8 = (strict.astype(np.uint8) * 255)

    t0 = cv2.getTickCount()
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(strict_u8, connectivity=8)
    t1 = cv2.getTickCount()
    cluster_ms = (t1 - t0) * 1000.0 / cv2.getTickFrequency()

    loose_thr2 = (threshold * mult) ** 2
    expanded = strict.copy()

    bboxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if int(area) >= int(min_component_area):
            bboxes.append((int(x), int(y), int(w), int(h)))

    if bboxes:
        expand_from_bboxes(min_dist2_roi, expanded, bboxes, loose_thr2=loose_thr2, pad=int(roi_pad))

    return ClusterResult(mask=expanded, strict_points=strict_points, used_points=strict_points, method="CC", cluster_ms=cluster_ms)