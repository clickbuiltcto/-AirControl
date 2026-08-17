"""
hand_tracker.py
================
Wraps MediaPipe Hands behind a small, stable interface so the rest of the
codebase never touches the MediaPipe API directly. This isolation is what
lets a future version swap in a custom/AI gesture-recognition backend
(per the roadmap) without touching gesture_controller.py's logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np

from aircontrol.config import HandTrackerConfig
from aircontrol.utils.logger import get_logger

log = get_logger("hand_tracker")


# Landmark index constants (MediaPipe Hands topology) — exposed so other
# modules never hard-code magic numbers.
class Landmark:
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


@dataclass
class HandResult:
    """A single detected hand, in a framework-agnostic shape."""
    label: str                      # "Left" or "Right" (as reported by MediaPipe,
                                     # i.e. from the camera's point of view of the user)
    score: float                    # handedness classification confidence
    landmarks: np.ndarray           # shape (21, 3) normalized x, y, z in [0,1]
    landmarks_px: np.ndarray        # shape (21, 2) pixel coordinates for the current frame

    def point(self, idx: int) -> np.ndarray:
        return self.landmarks[idx]

    def point_px(self, idx: int) -> np.ndarray:
        return self.landmarks_px[idx]


@dataclass
class TrackingFrame:
    hands: List[HandResult] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.hands)

    def get(self, label: str) -> Optional[HandResult]:
        for h in self.hands:
            if h.label == label:
                return h
        return None


class HandTracker:
    """Thin adapter around mediapipe.solutions.hands."""

    def __init__(self, config: HandTrackerConfig):
        self.config = config
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles

        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.max_num_hands,
            model_complexity=config.model_complexity,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        log.info("MediaPipe Hands initialized "
                  f"(max_hands={config.max_num_hands}, complexity={config.model_complexity})")

    def process(self, frame_bgr: np.ndarray) -> TrackingFrame:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)

        out = TrackingFrame()
        if not results.multi_hand_landmarks:
            return out

        handedness_list = results.multi_handedness or []
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
            pts_px = np.array([[int(p[0] * w), int(p[1] * h)] for p in pts], dtype=np.int32)

            label, score = "Unknown", 0.0
            if i < len(handedness_list):
                classification = handedness_list[i].classification[0]
                label, score = classification.label, classification.score

            out.hands.append(HandResult(label=label, score=score, landmarks=pts, landmarks_px=pts_px))

        return out

    def draw(self, frame_bgr: np.ndarray, raw_results) -> None:  # pragma: no cover - visual only
        """Optional helper retained for callers that want raw MediaPipe drawing."""
        pass

    def draw_hand(self, frame_bgr: np.ndarray, hand: HandResult, color=(0, 255, 170)) -> None:
        """Draw landmarks + connections for a single HandResult using pixel coords."""
        connections = self._mp_hands.HAND_CONNECTIONS
        pts = hand.landmarks_px
        for a, b in connections:
            cv2.line(frame_bgr, tuple(pts[a]), tuple(pts[b]), (60, 60, 60), 2, cv2.LINE_AA)
        for i, p in enumerate(pts):
            radius = 5 if i in (Landmark.THUMB_TIP, Landmark.INDEX_TIP) else 3
            cv2.circle(frame_bgr, tuple(p), radius, color, -1, cv2.LINE_AA)

    def close(self) -> None:
        self._hands.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
