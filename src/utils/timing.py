from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunningStats:
    n: int = 0
    total: float = 0.0
    max_v: float = 0.0

    def add(self, v: float) -> None:
        self.n += 1
        self.total += v
        if v > self.max_v:
            self.max_v = v

    @property
    def avg(self) -> float:
        return self.total / self.n if self.n else 0.0