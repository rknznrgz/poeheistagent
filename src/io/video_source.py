from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int


class FileVideoSource:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.path}")

        self.info = VideoInfo(
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            fps=float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0),
            frame_count=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def read(self):
        if self.cap is None:
            return False, None
        return self.cap.read()