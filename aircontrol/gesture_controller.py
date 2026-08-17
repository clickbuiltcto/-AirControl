"""
gesture_controller.py
======================
The brain of AirControl: turns raw hand landmarks (from hand_tracker.py)
into classified gestures, applies debouncing/stability rules, and drives
mouse_controller.py / screenshot_selector.py accordingly.

This module intentionally knows nothing about OpenCV windows or key
capture — it is pure gesture-to-action logic, which makes it unit
testable with synthetic landmark arrays (see tests/) and reusable by
future modes (drawing, presentation control, custom gestures, etc.).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List, Optional, Tuple

import numpy as np
import pyautogui

from aircontrol.config import AppConfig
from aircontrol.hand_tracker import HandResult, Landmark, TrackingFrame
from aircontrol.mouse_controller import MouseController
from aircontrol.screenshot_selector import ScreenshotSelector
from aircontrol.utils.logger import get_logger

log = get_logger("gesture")


# --------------------------------------------------------------------------- #
# Enums / state shapes
# --------------------------------------------------------------------------- #
class Gesture(str, Enum):
    NONE = "None"
    POINT = "Point (Move)"
    PINCH = "Pinch (Click)"
    TWO_FINGER_SCROLL = "Two-Finger (Scroll)"
    FIST = "Fist (Pause)"
    OPEN_PALM = "Open Palm (Neutral)"
    SWIPE_LEFT = "Swipe Left"
    SWIPE_RIGHT = "Swipe Right"
    TWO_HAND_PINCH = "Two-Hand Pinch (Screenshot)"
    NO_HAND = "No Hand Detected"


class Mode(str, Enum):
    DISABLED = "Disabled"
    PAUSED_NO_HAND = "Paused (No Hand)"
    PAUSED_FIST = "Paused (Fist)"
    TRACKING = "Tracking"
    SCROLLING = "Scrolling"
    SCREENSHOT_SELECTING = "Screenshot Selecting"


@dataclass
class GestureState:
    mode: Mode = Mode.DISABLED
    gesture: Gesture = Gesture.NONE
    hands_detected: int = 0
    hand_confident: bool = False
    master_enabled: bool = True
    cursor_pos: Optional[Tuple[int, int]] = None
    screenshot_progress: float = 0.0
    screenshot_active: bool = False
    screenshot_rect: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    last_screenshot_path: Optional[str] = None
    message: str = ""


@dataclass
class _SwipeTracker:
    history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=20))
    last_trigger_time: float = 0.0

    def add(self, x: float, t: float) -> None:
        self.history.append((t, x))

    def clear(self) -> None:
        self.history.clear()


# --------------------------------------------------------------------------- #
# Finger-state helpers
# --------------------------------------------------------------------------- #
def fingers_up(hand: HandResult, margin: float = 0.02) -> List[bool]:
    """Return [thumb, index, middle, ring, pinky] booleans (True = extended)."""
    lm = hand.landmarks
    fingers = []

    # Thumb: direction depends on which hand it is (mirrored selfie view).
    if hand.label == "Right":
        fingers.append(lm[Landmark.THUMB_TIP][0] < lm[Landmark.THUMB_IP][0])
    else:
        fingers.append(lm[Landmark.THUMB_TIP][0] > lm[Landmark.THUMB_IP][0])

    tip_pip_pairs = [
        (Landmark.INDEX_TIP, Landmark.INDEX_PIP),
        (Landmark.MIDDLE_TIP, Landmark.MIDDLE_PIP),
        (Landmark.RING_TIP, Landmark.RING_PIP),
        (Landmark.PINKY_TIP, Landmark.PINKY_PIP),
    ]
    for tip, pip in tip_pip_pairs:
        fingers.append(bool(lm[tip][1] < lm[pip][1] - margin))

    return fingers


def hand_scale(hand: HandResult) -> float:
    """A rough, camera-distance-invariant scale reference for this hand
    (wrist-to-middle-MCP distance). Used to normalize pinch distances."""
    lm = hand.landmarks
    wrist = lm[Landmark.WRIST][:2]
    mid_mcp = lm[Landmark.MIDDLE_MCP][:2]
    return float(np.linalg.norm(wrist - mid_mcp)) or 1e-6


def pinch_distance(hand: HandResult) -> float:
    """Normalized thumb-tip to index-tip distance (scale invariant)."""
    lm = hand.landmarks
    tip_a = lm[Landmark.THUMB_TIP][:2]
    tip_b = lm[Landmark.INDEX_TIP][:2]
    raw = float(np.linalg.norm(tip_a - tip_b))
    return raw / hand_scale(hand)


def pinch_midpoint(hand: HandResult) -> Tuple[float, float]:
    lm = hand.landmarks
    tip_a = lm[Landmark.THUMB_TIP][:2]
    tip_b = lm[Landmark.INDEX_TIP][:2]
    mid = (tip_a + tip_b) / 2.0
    return float(mid[0]), float(mid[1])


# --------------------------------------------------------------------------- #
# Navigation action execution (configurable, extensible)
# --------------------------------------------------------------------------- #
def _run_action(action: str) -> None:
    try:
        if action == "prev":
            pyautogui.hotkey("alt", "left")
        elif action == "next":
            pyautogui.hotkey("alt", "right")
        elif "+" in action:
            pyautogui.hotkey(*action.split("+"))
        else:
            pyautogui.press(action)
        # Hotkeys/keypresses go to whichever window currently has keyboard
        # focus (not wherever the mouse cursor is) - if that's the
        # AirControl window itself, this fires but has no visible effect.
        log.info(f"Navigation action '{action}' sent to the focused window.")
    except Exception as exc:  # pragma: no cover - depends on display env
        log.error(f"Failed to execute navigation action '{action}': {exc}")


# --------------------------------------------------------------------------- #
# Main controller
# --------------------------------------------------------------------------- #
class GestureController:
    def __init__(self, config: AppConfig, mouse: MouseController, screenshot: ScreenshotSelector):
        self.config = config
        self.mouse = mouse
        self.screenshot = screenshot

        self.master_enabled = config.start_enabled
        self._frames_without_confident_hand = 0
        self._swipe = _SwipeTracker()
        self._last_gesture_raw: Gesture = Gesture.NONE
        self._gesture_stable_count = 0
        self._stable_gesture: Gesture = Gesture.NONE
        self._scrolling = False

    # ------------------------------------------------------------------ #
    # Master enable/disable + emergency stop
    # ------------------------------------------------------------------ #
    def toggle_enabled(self) -> None:
        self.set_enabled(not self.master_enabled)

    def set_enabled(self, value: bool) -> None:
        self.master_enabled = value
        self.mouse.set_enabled(value)
        if not value:
            self.screenshot.reset()
            self._swipe.clear()
        log.info(f"AirControl {'ENABLED' if value else 'DISABLED'}")

    def emergency_stop(self) -> None:
        self.master_enabled = False
        self.mouse.emergency_stop()
        self.screenshot.reset()
        self._swipe.clear()
        log.warning("EMERGENCY STOP triggered.")

    # ------------------------------------------------------------------ #
    # Frame processing
    # ------------------------------------------------------------------ #
    def process(self, tracking: TrackingFrame) -> GestureState:
        state = GestureState(master_enabled=self.master_enabled, hands_detected=tracking.count)

        if not self.master_enabled:
            state.mode = Mode.DISABLED
            state.gesture = Gesture.NONE
            return state

        # --- Two-hand pinch screenshot takes priority over everything else
        right = tracking.get("Right")
        left = tracking.get("Left")
        screenshot_result = self._update_screenshot(right, left)
        if screenshot_result is not None:
            return screenshot_result

        # --- No hand at all -> auto pause (safety requirement)
        confident_hands = [h for h in tracking.hands if h.score >= self.config.hand_tracker.confident_threshold]
        if not confident_hands:
            self._frames_without_confident_hand += 1
            if self._frames_without_confident_hand >= self.config.hand_tracker.frames_lost_before_pause:
                state.mode = Mode.PAUSED_NO_HAND
                state.gesture = Gesture.NO_HAND
                self.mouse.reset_smoothing()
                self.mouse.end_scroll()
                self._swipe.clear()
                return state
            # Brief dropout (no *confident* hand yet, though an unconfident
            # one may be present): never drive the mouse from an unconfident
            # detection, but don't fully pause until frames_lost_before_pause
            # is reached either.
            state.mode = Mode.TRACKING
            state.gesture = Gesture.NONE
            return state

        else:
            self._frames_without_confident_hand = 0

        primary = self._pick_primary_hand(tracking)
        if primary is None:
            state.mode = Mode.TRACKING
            state.gesture = Gesture.NONE
            return state

        state.hand_confident = primary.score >= self.config.hand_tracker.confident_threshold

        raw_gesture = self._classify(primary)
        stable_gesture = self._debounce(raw_gesture)
        state.gesture = stable_gesture

        # Swipe detection runs alongside the primary classification.
        swipe = self._check_swipe(primary, stable_gesture)
        if swipe is not None:
            state.gesture = swipe
            state.mode = Mode.TRACKING
            state.message = f"Navigation action: {swipe.value}"
            return state

        if stable_gesture == Gesture.FIST:
            state.mode = Mode.PAUSED_FIST
            self.mouse.reset_smoothing()
            self.mouse.end_scroll()
            self._scrolling = False
            return state

        if stable_gesture == Gesture.OPEN_PALM:
            state.mode = Mode.TRACKING
            self.mouse.end_scroll()
            self._scrolling = False
            return state

        if stable_gesture == Gesture.TWO_FINGER_SCROLL:
            state.mode = Mode.SCROLLING
            idx_tip = primary.point(Landmark.INDEX_TIP)
            if not self._scrolling:
                self.mouse.start_scroll_reference(float(idx_tip[1]))
                self._scrolling = True
            else:
                self.mouse.scroll_from(float(idx_tip[1]))
            return state
        else:
            if self._scrolling:
                self.mouse.end_scroll()
                self._scrolling = False

        if stable_gesture in (Gesture.POINT, Gesture.PINCH):
            state.mode = Mode.TRACKING
            idx_tip = primary.point(Landmark.INDEX_TIP)
            pos = self.mouse.move_to(float(idx_tip[0]), float(idx_tip[1]))
            state.cursor_pos = pos

            dist = pinch_distance(primary)
            if self.mouse.update_pinch(dist):
                self.mouse.click()
                state.gesture = Gesture.PINCH
                state.message = "Click!"
            return state

        state.mode = Mode.TRACKING
        return state

    # ------------------------------------------------------------------ #
    # Screenshot (two-hand pinch)
    # ------------------------------------------------------------------ #
    def _update_screenshot(self, right: Optional[HandResult], left: Optional[HandResult]) -> Optional[GestureState]:
        right_pinching = right is not None and pinch_distance(right) < self.config.screenshot.pinch_distance_threshold
        left_pinching = left is not None and pinch_distance(left) < self.config.screenshot.pinch_distance_threshold

        if not (right_pinching and left_pinching):
            if self.screenshot.state.active:
                self.screenshot.reset()
            return None

        right_point = pinch_midpoint(right)
        left_point = pinch_midpoint(left)
        sel = self.screenshot.update(right_point, left_point)

        state = GestureState(master_enabled=self.master_enabled, hands_detected=2)
        state.mode = Mode.SCREENSHOT_SELECTING
        state.gesture = Gesture.TWO_HAND_PINCH
        state.screenshot_active = sel.active
        state.screenshot_progress = sel.progress
        if sel.corner_a and sel.corner_b:
            state.screenshot_rect = (sel.corner_a, sel.corner_b)
        if sel.just_captured:
            state.last_screenshot_path = sel.captured_path
            state.message = f"Screenshot saved: {sel.captured_path}"
        return state

    # ------------------------------------------------------------------ #
    # Classification
    # ------------------------------------------------------------------ #
    def _pick_primary_hand(self, tracking: TrackingFrame) -> Optional[HandResult]:
        if not tracking.hands:
            return None
        # Prefer the Right hand as the dominant control hand; fall back to
        # whichever hand is present / most confident.
        right = tracking.get("Right")
        if right is not None:
            return right
        return max(tracking.hands, key=lambda h: h.score)

    def _classify(self, hand: HandResult) -> Gesture:
        margin = self.config.gesture.finger_up_margin
        thumb, index, middle, ring, pinky = fingers_up(hand, margin)

        if not any([thumb, index, middle, ring, pinky]):
            return Gesture.FIST

        if all([thumb, index, middle, ring, pinky]):
            return Gesture.OPEN_PALM

        if index and middle and not ring and not pinky:
            return Gesture.TWO_FINGER_SCROLL

        if index and not middle and not ring and not pinky:
            dist = pinch_distance(hand)
            if dist < self.config.mouse.pinch_distance_threshold:
                return Gesture.PINCH
            return Gesture.POINT

        # Fallback: if thumb+index are close together regardless of other
        # fingers, still treat as a pinch (people rarely keep a perfect
        # "index only" pose while pinching).
        if pinch_distance(hand) < self.config.mouse.pinch_distance_threshold:
            return Gesture.PINCH

        return Gesture.POINT

    def _debounce(self, raw: Gesture) -> Gesture:
        needed = self.config.gesture.gesture_stability_frames
        if raw == self._last_gesture_raw:
            self._gesture_stable_count += 1
        else:
            self._gesture_stable_count = 1
            self._last_gesture_raw = raw

        if self._gesture_stable_count >= needed:
            self._stable_gesture = raw
        return self._stable_gesture if self._stable_gesture != Gesture.NONE else raw

    # ------------------------------------------------------------------ #
    # Swipe navigation
    # ------------------------------------------------------------------ #
    def _check_swipe(self, hand: HandResult, gesture: Gesture) -> Optional[Gesture]:
        cfg = self.config.mouse
        now = time.time()

        # Only accumulate swipe history while the hand is "open" (flat) or
        # mid-point/neutral — not while pinching, scrolling, or fisted —
        # to avoid confusing gestures.
        if gesture in (Gesture.FIST, Gesture.PINCH, Gesture.TWO_FINGER_SCROLL):
            self._swipe.clear()
            return None

        wrist_x = float(hand.point(Landmark.WRIST)[0])
        self._swipe.add(wrist_x, now)

        if now - self._swipe.last_trigger_time < cfg.swipe_cooldown:
            return None

        history = self._swipe.history
        if len(history) < 3:
            return None

        # Trim to the swipe time window
        window = [p for p in history if now - p[0] <= cfg.swipe_max_duration]
        if len(window) < 3:
            return None

        xs = [p[1] for p in window]
        delta = xs[-1] - xs[0]
        spread = max(xs) - min(xs)

        if spread < cfg.swipe_min_distance:
            return None

        self._swipe.last_trigger_time = now
        self._swipe.clear()

        if delta > 0:
            _run_action(cfg.swipe_right_action)
            return Gesture.SWIPE_RIGHT
        else:
            _run_action(cfg.swipe_left_action)
            return Gesture.SWIPE_LEFT
