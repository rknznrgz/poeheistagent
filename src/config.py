from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple


@dataclass(frozen=True)
class Crop:
    x0: int
    y0: int
    x1: int
    y1: int

    def clamp(self, width: int, height: int) -> "Crop":
        x0 = max(0, min(width, self.x0))
        x1 = max(0, min(width, self.x1))
        y0 = max(0, min(height, self.y0))
        y1 = max(0, min(height, self.y1))
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Invalid crop after clamp: ({x0},{y0})-({x1},{y1})")
        return Crop(x0=x0, y0=y0, x1=x1, y1=y1)

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0


@dataclass(frozen=True)
class ProcessingConfig:
    proc_size: Tuple[int, int] = (1920, 1080)  # (width, height)
    crop: Crop = Crop(490, 100, 1500, 900)
    proc_scale: float = 0.5


@dataclass(frozen=True)
class ColorFilterConfig:
    targets_hex: str = "#88829F,#6B8490"
    threshold: float = 12.0
    dbscan_multiplier: float = 2.25


@dataclass(frozen=True)
class ClusterConfig:
    use_cc_fallback: bool = True
    cc_fallback_points: int = 5000
    cc_min_area: int = 30

    # DBSCAN
    min_samples: int = 10
    roi_pad: int = 40
    dbscan_stride: int = 2
    dbscan_target_points: int = 900
    dbscan_max_points: int = 2500
    dbscan_algorithm: Literal["auto", "brute", "kd_tree", "ball_tree"] = "brute"
    leaf_size: int = 40


@dataclass(frozen=True)
class OutputConfig:
    output_mode: Literal["sbs", "mask", "overlay"] = "sbs"
    output_size: Tuple[int, int] = (1920, 1080)  # (width, height)


@dataclass(frozen=True)
class BenchmarkConfig:
    print_every: int = 60
    time_io: bool = False  # if True, includes read+write in per-frame time