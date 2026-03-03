"""Grid search: exhaustive parameter sweep for map quality optimization."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from ..config import CanvasConfig, FlowConfig, MapBuilderConfig, ProcessingConfig, VelocityConfig
from .map_builder import MapBuilder, preload_frames


DEFAULT_PARAM_GRID: Dict[str, list] = {
    "add_weight":        [0.08, 0.20, 0.35],
    "decay":             [0.993, 0.999, 1.000],
    "ema_alpha":         [0.3, 1.0],
    "input_morph_ksize": [0, 3],
    "scale":             [0.5, 0.7],
    "flow_input":        ["gray", "dist_transform"],
    "k_mad":             [3.0, 5.0],
    "reanchor_damping":  [0.2, 0.5],
}


_FLOW_KEYS = {
    "scale", "flow_input", "blur", "inner_ratio", "outer_ratio",
    "edge_low", "edge_high", "edge_dilate",
    "fb_pyr_scale", "fb_levels", "fb_winsize", "fb_iterations",
    "fb_poly_n", "fb_poly_sigma", "max_points", "k_mad", "base_tol",
    "min_points", "min_inlier_frac", "max_shift_full",
}

_CANVAS_KEYS = {
    "canvas_w", "canvas_h", "add_weight", "decay", "clip_max", "conf_boost",
    "ema_alpha", "input_morph_ksize",
    "cleanup_every", "morph_ksize", "conf_threshold", "gauss_sigma",
    "reanchor_every", "reanchor_min_score", "reanchor_damping", "reanchor_pad",
}

_VELOCITY_KEYS = {"min_alpha", "max_alpha"}
_TOP_KEYS = {"invert_update", "on_fail"}


def _build_config(base: MapBuilderConfig, overrides: Dict[str, Any]) -> MapBuilderConfig:
    from dataclasses import replace
    flow_ov = {k: overrides[k] for k in overrides if k in _FLOW_KEYS}
    canvas_ov = {k: overrides[k] for k in overrides if k in _CANVAS_KEYS}
    vel_ov = {k: overrides[k] for k in overrides if k in _VELOCITY_KEYS}
    top_ov = {k: overrides[k] for k in overrides if k in _TOP_KEYS}
    flow = replace(base.flow, **flow_ov) if flow_ov else base.flow
    canvas = replace(base.canvas, **canvas_ov) if canvas_ov else base.canvas
    vel = replace(base.velocity, **vel_ov) if vel_ov else base.velocity
    return replace(base, flow=flow, canvas=canvas, velocity=vel, **top_ov)


def make_label(varied_params: Dict[str, Any]) -> str:
    abbrev = {
        "input_morph_ksize": "morph", "add_weight": "aw", "ema_alpha": "ema",
        "decay": "dc", "scale": "sc", "flow_input": "fi",
        "k_mad": "km", "reanchor_damping": "rd",
    }
    parts = []
    for k, v in sorted(varied_params.items()):
        sk = abbrev.get(k, k)
        if isinstance(v, float):
            parts.append(f"{sk}{v:.3f}")
        elif isinstance(v, str):
            sv = v.replace("dist_transform", "dt").replace("gray", "gr")
            parts.append(f"{sk}-{sv}")
        else:
            parts.append(f"{sk}{v}")
    return "_".join(parts)


@dataclass
class GridSearchRunner:
    """Runs exhaustive parameter grid search over the MapBuilder pipeline."""

    base_config: MapBuilderConfig
    param_grid: Dict[str, list] = field(default_factory=lambda: dict(DEFAULT_PARAM_GRID))

    def run(self, input_path: str, outdir: str = "grid_results") -> Path:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)

        proc = self.base_config.processing
        pw, ph = proc.proc_size
        cr = proc.crop
        crop_str = f"{cr.x0},{cr.y0},{cr.x1},{cr.y1}"

        keys = sorted(self.param_grid.keys())
        combos = list(itertools.product(*[self.param_grid[k] for k in keys]))
        n = len(combos)

        print(f"Grid search: {n} combinations")
        for k in keys:
            print(f"  {k}: {self.param_grid[k]}")
        print()

        print("Pre-loading frames...", flush=True)
        t0 = time.perf_counter()
        raw_frames, _ = preload_frames(input_path, pw, ph, crop_str, 127)
        print(f"  {len(raw_frames)} frames in {time.perf_counter() - t0:.1f}s\n")

        summary = [f"Grid search: {n} combos, {len(raw_frames)} frames", f"Params: {keys}", "", "-" * 110]

        for idx, combo in enumerate(combos):
            varied = dict(zip(keys, combo))
            label = make_label(varied)
            cfg = _build_config(self.base_config, varied)
            fname = f"{idx:03d}_{label}.png"

            print(f"[{idx + 1:3d}/{n}] {label} ...", end="", flush=True)
            t0 = time.perf_counter()

            builder = MapBuilder(cfg)
            result = builder.run(raw_frames)
            cv2.imwrite(str(out / fname), result.canvas_u8)

            elapsed = time.perf_counter() - t0
            pct = result.flow_rate * 100
            print(f"  flow={pct:5.1f}%  fail={result.fail:4d}  {elapsed:6.1f}s")
            summary.append(
                f"{idx:4d}  {label:<55}  {pct:5.1f}%  {result.fail:5d}  {elapsed:6.1f}s  {fname}"
            )

        (out / "_summary.txt").write_text("\n".join(summary) + "\n")
        print(f"\nDone. {n} images in {out}/")
        return out