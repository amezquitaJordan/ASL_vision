"""Hand tracking adapters for MediaPipe and an OpenCV-only fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np


BBox = tuple[int, int, int, int]
Point = tuple[float, float]


@dataclass
class HandDetection:
    bbox: BBox
    crop: np.ndarray
    motion_point: Point | None
    status: str


class HandTracker(Protocol):
    status: str

    def detect(self, frame: np.ndarray) -> HandDetection | None:
        ...

    def draw(self, frame: np.ndarray, detection: HandDetection | None) -> None:
        ...

    def close(self) -> None:
        ...


def fixed_roi_bbox(frame_width: int, frame_height: int) -> BBox:
    """Return a centered square ROI that fits inside the frame."""
    size = int(min(frame_width, frame_height) * 0.50)
    x1 = max(0, (frame_width - size) // 2)
    y1 = max(0, (frame_height - size) // 2 - int(frame_height * 0.08))
    x2 = min(frame_width, x1 + size)
    y2 = min(frame_height, y1 + size)
    return x1, y1, x2, y2


def fingertip_from_mask(mask: np.ndarray, roi_origin: tuple[int, int], frame_size: tuple[int, int]) -> Point | None:
    """Estimate a fingertip point from the topmost point of the largest contour."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 120:
        return None

    topmost = tuple(largest[largest[:, :, 1].argmin()][0])
    frame_width, frame_height = frame_size
    x = (roi_origin[0] + int(topmost[0])) / frame_width
    y = (roi_origin[1] + int(topmost[1])) / frame_height
    return x, y


class MediaPipeHandTracker:
    def __init__(self) -> None:
        import mediapipe as mp

        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._hands = self._mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65,
        )
        self.status = "MediaPipe"
        self._last_landmarks = None

    def detect(self, frame: np.ndarray) -> HandDetection | None:
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        self._last_landmarks = None
        if not result.multi_hand_landmarks:
            return None

        landmarks = result.multi_hand_landmarks[0]
        self._last_landmarks = landmarks
        xs = [landmark.x for landmark in landmarks.landmark]
        ys = [landmark.y for landmark in landmarks.landmark]
        padding = 35
        x1 = max(0, int(min(xs) * width) - padding)
        y1 = max(0, int(min(ys) * height) - padding)
        x2 = min(width, int(max(xs) * width) + padding)
        y2 = min(height, int(max(ys) * height) + padding)
        index_tip = landmarks.landmark[8]
        return HandDetection(
            bbox=(x1, y1, x2, y2),
            crop=frame[y1:y2, x1:x2],
            motion_point=(index_tip.x, index_tip.y),
            status=self.status,
        )

    def draw(self, frame: np.ndarray, detection: HandDetection | None) -> None:
        if detection:
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (39, 174, 96), 2)
        if self._last_landmarks is not None:
            self._mp_drawing.draw_landmarks(frame, self._last_landmarks, self._mp_hands.HAND_CONNECTIONS)

    def close(self) -> None:
        self._hands.close()


class FixedRoiHandTracker:
    """OpenCV-only fallback when MediaPipe cannot load native graph assets."""

    def __init__(self, reason: str = "") -> None:
        self.status = "ROI fijo"
        self.reason = reason

    def detect(self, frame: np.ndarray) -> HandDetection | None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = fixed_roi_bbox(width, height)
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 25, 35], dtype=np.uint8)
        upper_skin = np.array([25, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        point = fingertip_from_mask(mask, roi_origin=(x1, y1), frame_size=(width, height))
        return HandDetection(
            bbox=(x1, y1, x2, y2),
            crop=crop,
            motion_point=point,
            status=self.status,
        )

    def draw(self, frame: np.ndarray, detection: HandDetection | None) -> None:
        if not detection:
            return
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (92, 225, 230), 2)
        cv2.putText(frame, "Coloca la mano aqui", (x1 + 12, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (92, 225, 230), 2)

    def close(self) -> None:
        return None


def create_hand_tracker() -> HandTracker:
    """Create MediaPipe tracker, falling back to a fixed ROI if native assets fail."""
    try:
        return MediaPipeHandTracker()
    except Exception as exc:
        print(f"MediaPipe no pudo iniciar; usando ROI fijo. Detalle: {exc}")
        return FixedRoiHandTracker(reason=str(exc))
