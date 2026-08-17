"""
mouse_controller.py
====================
Everything that touches the actual OS mouse cursor lives here, and only
here. Centralizing it means:
  * Every safety rule (bounds clamping, enable/disable gate, fail-safe)
    is enforced in exactly one place.
  * Coordinate mapping / smoothing / dead-zone logic is reusable for
    future features (e.g. drawing mode) without duplicating math.

AirControl NEVER moves the mouse unless ``MouseController.enabled`` is True
and a caller explicitly asks it to — there is no background timer or
implicit motion.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pyautogui

from aircontrol.config import MouseConfig
from aircontrol.utils.logger import get_logger
from aircontrol.utils.smoothing import PointSmoother

log = get_logger("mouse")

# pyautogui safety net: if the controlled cursor is ever driven into a
# physical screen corner, pyautogui raises FailSafeException, which we
# treat as an automatic emergency stop upstream.
pyautogui.FAILSAFE = True
# We manage our own frame-rate pacing; disable pyautogui's built-in delay
# for responsiveness.
pyautogui.PAUSE = 0.0


@dataclass
class ClickState:
    is_pinching: bool = False
    pinch_start_time: float = 0.0
    stable_frames: int = 0
    last_click_time: float = 0.0


class MouseController:
    """
    Maps normalized hand-landmark coordinates to smooth, safe on-screen
    cursor motion, and exposes discrete click/scroll actions.
    """

    def __init__(self, config: MouseConfig):
        self.config = config
        self.enabled = True  # master safety gate — set False to hard-disable all motion

        self.screen_w, self.screen_h = pyautogui.size()
        self._smoother = PointSmoother(
            min_cutoff=config.one_euro_min_cutoff,
            beta=config.one_euro_beta,
            d_cutoff=config.one_euro_d_cutoff,
        )
        self._last_screen_pos: Optional[Tuple[int, int]] = None
        self._click_state = ClickState()
        self._scroll_ref_y: Optional[float] = None

        log.info(f"MouseController ready. Screen resolution: {self.screen_w}x{self.screen_h}")

    # ------------------------------------------------------------------ #
    # Coordinate mapping
    # ------------------------------------------------------------------ #
    def map_to_screen(self, norm_x: float, norm_y: float) -> Tuple[int, int]:
        """
        Map a normalized (0-1) camera-space point to screen pixel coords,
        using a shrunken "active region" so the user doesn't need to reach
        the physical edges of the camera frame to hit the screen edges.
        """
        cfg = self.config
        mx, my = cfg.active_region_margin_x, cfg.active_region_margin_y

        # Re-scale [mx, 1-mx] -> [0, 1]
        x = (norm_x - mx) / max(1e-6, (1 - 2 * mx))
        y = (norm_y - my) / max(1e-6, (1 - 2 * my))

        # Apply sensitivity around the center (0.5, 0.5)
        x = 0.5 + (x - 0.5) * cfg.sensitivity
        y = 0.5 + (y - 0.5) * cfg.sensitivity

        x = float(np.clip(x, 0.0, 1.0))
        y = float(np.clip(y, 0.0, 1.0))

        screen_x = int(x * (self.screen_w - 1))
        screen_y = int(y * (self.screen_h - 1))
        return screen_x, screen_y

    # ------------------------------------------------------------------ #
    # Cursor motion
    # ------------------------------------------------------------------ #
    def move_to(self, norm_x: float, norm_y: float) -> Optional[Tuple[int, int]]:
        """Smoothly move the OS cursor toward the mapped target. Returns the
        actual screen position moved to, or None if suppressed (disabled /
        inside dead zone)."""
        if not self.enabled:
            return None

        target_x, target_y = self.map_to_screen(norm_x, norm_y)
        smoothed = self._smoother.update(target_x, target_y)
        sx, sy = int(smoothed.x), int(smoothed.y)

        # clamp defensively — never allow an out-of-bounds cursor jump
        sx = int(np.clip(sx, 0, self.screen_w - 1))
        sy = int(np.clip(sy, 0, self.screen_h - 1))

        if self._last_screen_pos is not None:
            dx = sx - self._last_screen_pos[0]
            dy = sy - self._last_screen_pos[1]
            if abs(dx) < self.config.dead_zone_px and abs(dy) < self.config.dead_zone_px:
                return self._last_screen_pos  # inside dead zone — do not move

        try:
            pyautogui.moveTo(sx, sy, _pause=False)
        except pyautogui.FailSafeException:
            log.warning("pyautogui fail-safe triggered - forcing disable.")
            self.enabled = False
            return None

        self._last_screen_pos = (sx, sy)
        return self._last_screen_pos

    def reset_smoothing(self) -> None:
        """Call when the hand is (re)acquired to avoid a big smoothing jump."""
        self._smoother.reset()
        self._last_screen_pos = None

    # ------------------------------------------------------------------ #
    # Click (pinch) handling — includes debounce + hysteresis
    # ------------------------------------------------------------------ #
    def update_pinch(self, pinch_distance: float) -> bool:
        """
        Feed the current normalized thumb-index distance in. Returns True
        exactly on the frame a click should fire.
        """
        cfg = self.config
        state = self._click_state
        now = time.time()

        if not state.is_pinching:
            if pinch_distance < cfg.pinch_distance_threshold:
                state.stable_frames += 1
                if state.stable_frames >= cfg.click_hold_frames:
                    if now - state.last_click_time >= cfg.click_debounce_seconds:
                        state.is_pinching = True
                        state.pinch_start_time = now
                        state.last_click_time = now
                        state.stable_frames = 0
                        return True
            else:
                state.stable_frames = 0
        else:
            # hysteresis: require a bigger gap to "release" the pinch so we
            # don't chatter click/no-click around the threshold
            if pinch_distance > cfg.pinch_release_threshold:
                state.is_pinching = False
                state.stable_frames = 0

        return False

    def click(self) -> None:
        if not self.enabled:
            return
        pyautogui.click(_pause=False)
        log.info("Left click")

    # ------------------------------------------------------------------ #
    # Scroll
    # ------------------------------------------------------------------ #
    def start_scroll_reference(self, norm_y: float) -> None:
        self._scroll_ref_y = norm_y

    def scroll_from(self, norm_y: float) -> None:
        if not self.enabled or self._scroll_ref_y is None:
            return
        delta = self._scroll_ref_y - norm_y  # moving hand up -> scroll up (positive)
        if abs(delta) < self.config.scroll_dead_zone:
            return
        amount = int(delta * self.config.scroll_sensitivity * 10)
        if amount != 0:
            pyautogui.scroll(amount, _pause=False)
        self._scroll_ref_y = norm_y

    def end_scroll(self) -> None:
        self._scroll_ref_y = None

    # ------------------------------------------------------------------ #
    # Safety
    # ------------------------------------------------------------------ #
    def emergency_stop(self) -> None:
        """Immediately halt all mouse influence. No motion, no clicks."""
        self.enabled = False
        self._scroll_ref_y = None
        self._click_state = ClickState()
        log.warning("EMERGENCY STOP - mouse control disabled.")

    def set_enabled(self, value: bool) -> None:
        self.enabled = value
        if value:
            self.reset_smoothing()
        log.info(f"Mouse control {'ENABLED' if value else 'DISABLED'}")
