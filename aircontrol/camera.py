"""
camera.py
=========
Owns the webcam device: discovery, opening, reading frames, and clean
shutdown. Designed to "just work" on an HP Victus PC's built-in webcam
(normally device index 0) without requiring an external camera, while
still gracefully probing a couple of fallback indices.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from aircontrol.config import CameraConfig
from aircontrol.utils.logger import get_logger

log = get_logger("camera")


class CameraUnavailableError(RuntimeError):
    """Raised when no usable webcam can be opened."""


@dataclass
class FrameResult:
    ok: bool
    frame: Optional[np.ndarray]
    timestamp: float


class Camera:
    """
    Thin, safe wrapper around cv2.VideoCapture with:
      * automatic device discovery (built-in webcam first)
      * resolution / fps configuration
      * a clear, actionable error if no camera is found
      * context-manager support for guaranteed release()
    """

    def __init__(self, config: CameraConfig):
        self.config = config
        self._cap: Optional[cv2.VideoCapture] = None
        self.active_index: Optional[int] = None
        self._last_frame_time = time.time()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def open(self) -> None:
        """Open the best available camera or raise CameraUnavailableError."""
        indices_to_try = [self.config.preferred_index, *self.config.fallback_indices]
        backend = self._preferred_backend()

        for index in indices_to_try:
            log.info(f"Attempting to open camera index {index} ...")
            cap = cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)

            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            cap.set(cv2.CAP_PROP_FPS, self.config.requested_fps)

            # Confirm the camera actually delivers frames — isOpened() alone
            # can lie on some platforms/drivers.
            if self._confirm_frames(cap):
                self._cap = cap
                self.active_index = index
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                log.info(f"Camera ready: index={index} resolution={w}x{h} fps={fps:.1f}")
                return

            cap.release()

        raise CameraUnavailableError(
            "AirControl could not find a usable webcam.\n"
            "  - Checked device index(es): "
            f"{indices_to_try}\n"
            "  - Make sure your HP Victus built-in webcam is enabled "
            "(Windows Camera privacy settings, or Device Manager if disabled).\n"
            "  - Close any other application that might be using the camera "
            "(Zoom, Teams, Windows Camera app, etc.).\n"
            "  - If you use an external webcam, unplug it or set "
            "CameraConfig.preferred_index to its device number."
        )

    def _confirm_frames(self, cap: cv2.VideoCapture, attempts: Optional[int] = None) -> bool:
        attempts = attempts or self.config.max_init_attempts
        for _ in range(attempts):
            ok, frame = cap.read()
            if ok and frame is not None:
                return True
            time.sleep(0.02)
        return False

    @staticmethod
    def _preferred_backend():
        """Pick a sane default backend per-OS; None lets OpenCV auto-select."""
        system = platform.system()
        if system == "Windows":
            # DirectShow is the most reliable backend for built-in laptop/
            # desktop webcams on Windows (incl. HP Victus machines).
            return cv2.CAP_DSHOW
        return None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            log.info("Camera released.")

    # Context manager sugar -------------------------------------------------
    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Frame access
    # ------------------------------------------------------------------ #
    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read(self) -> FrameResult:
        if not self.is_open:
            return FrameResult(ok=False, frame=None, timestamp=time.time())

        ok, frame = self._cap.read()
        now = time.time()
        if not ok or frame is None:
            return FrameResult(ok=False, frame=None, timestamp=now)

        if self.config.flip_horizontal:
            frame = cv2.flip(frame, 1)

        self._last_frame_time = now
        return FrameResult(ok=True, frame=frame, timestamp=now)
