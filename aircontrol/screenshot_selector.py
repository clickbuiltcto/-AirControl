"""
screenshot_selector.py
=======================
Implements the two-hand pinch screenshot gesture:

  * Right hand thumb+index pinch  -> first corner
  * Left hand thumb+index pinch   -> opposite corner
  * Both corners are mapped to screen coordinates using the same mapping
    MouseController uses, so the rectangle you see corresponds to real
    screen real-estate.
  * While both hands pinch, a live rectangle is drawn on the camera
    overlay (see ui.py) between the two corners.
  * Holding both pinches for ``hold_seconds_to_capture`` confirms the
    capture and a PNG is written to ``screenshots/``.
  * ESC cancels a selection in progress.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import pyautogui

from aircontrol.config import ScreenshotConfig
from aircontrol.utils.logger import get_logger

log = get_logger("screenshot")


@dataclass
class SelectionState:
    active: bool = False
    corner_a: Optional[Tuple[int, int]] = None  # right hand, screen px
    corner_b: Optional[Tuple[int, int]] = None  # left hand, screen px
    hold_start: Optional[float] = None
    progress: float = 0.0  # 0..1 toward capture
    captured_path: Optional[str] = None
    just_captured: bool = False


class ScreenshotSelector:
    def __init__(self, config: ScreenshotConfig, screen_to_px_fn):
        """
        screen_to_px_fn: callable(norm_x, norm_y) -> (screen_x, screen_y)
        Reuses MouseController's coordinate mapping so the drawn rectangle
        matches real screen coordinates.
        """
        self.config = config
        self._map = screen_to_px_fn
        self.state = SelectionState()
        self._awaiting_release = False  # prevents multi-capture spam while pinches stay held
        os.makedirs(self.config.output_dir, exist_ok=True)

    def reset(self) -> None:
        self.state = SelectionState()
        self._awaiting_release = False

    def update(self, right_pinch_point: Optional[Tuple[float, float]],
               left_pinch_point: Optional[Tuple[float, float]]) -> SelectionState:
        """
        Call every frame with the normalized (x, y) pinch point for each
        hand, or None if that hand isn't currently pinching.
        """
        state = self.state
        state.just_captured = False
        both_pinching = right_pinch_point is not None and left_pinch_point is not None

        if not both_pinching:
            if state.active:
                # released early — cancel silently, ready for a new attempt
                log.info("Two-hand pinch released before hold time - selection reset.")
            self.reset()
            return self.state

        if self._awaiting_release:
            # A capture just fired; require the user to release both pinches
            # before a new selection can start (prevents rapid-fire repeats).
            return self.state

        state.active = True
        state.corner_a = self._map(*right_pinch_point)
        state.corner_b = self._map(*left_pinch_point)

        if state.hold_start is None:
            state.hold_start = time.time()

        elapsed = time.time() - state.hold_start
        state.progress = min(1.0, elapsed / self.config.hold_seconds_to_capture)

        if state.progress >= 1.0:
            self._capture(state)

        return self.state

    def cancel(self) -> None:
        if self.state.active:
            log.info("Screenshot selection cancelled by user (ESC).")
        self.reset()

    # ------------------------------------------------------------------ #
    def _capture(self, state: SelectionState) -> None:
        x1, y1 = state.corner_a
        x2, y2 = state.corner_b
        left, top = min(x1, x2), min(y1, y2)
        right, bottom = max(x1, x2), max(y1, y2)
        width, height = right - left, bottom - top

        if width < self.config.min_region_px or height < self.config.min_region_px:
            log.info("Selected region too small - ignoring capture.")
            self.reset()
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.filename_prefix}_{timestamp}.png"
        path = os.path.join(self.config.output_dir, filename)

        try:
            shot = pyautogui.screenshot(region=(left, top, width, height))
            shot.save(path)
            log.info(f"Screenshot saved: {path} ({width}x{height})")
            state.captured_path = path
            state.just_captured = True
            self._awaiting_release = True
        except Exception as exc:  # pragma: no cover - depends on display env
            log.error(f"Failed to capture screenshot: {exc}")
        finally:
            # Deactivate the selection but KEEP this same state object (with
            # captured_path / just_captured intact) so the caller processing
            # this frame can still see the result. self.reset() would swap
            # in a brand-new object and silently drop that one-frame signal.
            state.active = False
            state.corner_a = None
            state.corner_b = None
            state.hold_start = None
            state.progress = 0.0
