from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2


def fourcc(code: str) -> int:
    return cv2.VideoWriter_fourcc(*code)


def pick_codec_for_path(path: str) -> str:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".mp4":
        return "mp4v"
    return "XVID"


class VideoSink:
    def __init__(self, path: str | Path, fps: float, size_wh: Tuple[int, int], codec: str | None = None):
        self.path = str(path)
        w, h = size_wh
        if codec is None:
            codec = pick_codec_for_path(self.path)
        self.codec = codec
        self.writer = cv2.VideoWriter(self.path, fourcc(codec), float(fps), (int(w), int(h)), True)
        if not self.writer.isOpened():
            raise RuntimeError(
                f"Could not open VideoWriter for {self.path} (codec={codec}). Try .avi or a different codec."
            )

    def write(self, frame_bgr) -> None:
        self.writer.write(frame_bgr)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()