"""Simple temporal detector for ASL J and Z motions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import hypot
from typing import Deque


Point = tuple[float, float]


@dataclass
class MotionLetterDetector:
    """Detect J and Z from normalized index-fingertip trajectories."""

    max_points: int = 18
    min_points: int = 6
    min_distance: float = 0.18
    cooldown_frames: int = 12
    points: Deque[Point] = field(default_factory=deque)
    _cooldown: int = 0

    def update(self, point: Point | None) -> str | None:
        """Add a point and return "J" or "Z" when a motion is recognized."""
        if point is None:
            self.reset()
            return None

        if self._cooldown > 0:
            self._cooldown -= 1

        self.points.append(point)
        while len(self.points) > self.max_points:
            self.points.popleft()

        if self._cooldown > 0 or len(self.points) < self.min_points:
            return None

        path = list(self.points)
        if self._detect_z(path):
            self._trigger()
            return "Z"
        if self._detect_j(path):
            self._trigger()
            return "J"
        return None

    def reset(self) -> None:
        self.points.clear()
        self._cooldown = 0

    def _trigger(self) -> None:
        self.points.clear()
        self._cooldown = self.cooldown_frames

    def _path_distance(self, path: list[Point]) -> float:
        return sum(hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))

    def _detect_j(self, path: list[Point]) -> bool:
        start = path[0]
        end = path[-1]
        total_distance = self._path_distance(path)
        downward = end[1] - start[1] > self.min_distance
        hook_left = start[0] - end[0] > 0.12
        has_curve = min(point[0] for point in path[-3:]) < min(point[0] for point in path[:3])
        return total_distance > self.min_distance and downward and hook_left and has_curve

    def _detect_z(self, path: list[Point]) -> bool:
        if len(path) < 6 or self._path_distance(path) < self.min_distance * 1.8:
            return False

        first = path[: len(path) // 3]
        middle = path[len(path) // 3 : 2 * len(path) // 3]
        last = path[2 * len(path) // 3 :]
        if not first or not middle or not last:
            return False

        first_dx = first[-1][0] - first[0][0]
        first_dy = abs(first[-1][1] - first[0][1])
        mid_dx = middle[-1][0] - middle[0][0]
        mid_dy = middle[-1][1] - middle[0][1]
        last_dx = last[-1][0] - last[0][0]
        last_dy = abs(last[-1][1] - last[0][1])

        top_horizontal = first_dx > 0.15 and first_dy < 0.10
        diagonal_down_left = mid_dx < -0.08 and mid_dy > 0.12
        bottom_horizontal = last_dx > 0.15 and last_dy < 0.12
        return top_horizontal and diagonal_down_left and bottom_horizontal
