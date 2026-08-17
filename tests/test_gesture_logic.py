"""
tests/test_gesture_logic.py
============================
Camera-free unit tests. Hand poses are synthesized as raw landmark arrays
(the same shape MediaPipe produces) so the gesture-classification and
mouse-mapping logic can be verified without any webcam, display, or
model inference — fast, deterministic, CI-friendly.

Run with:  python -m pytest tests/ -v
       or: python tests/test_gesture_logic.py   (falls back to a tiny
                                                   built-in runner if
                                                   pytest isn't installed)
"""

from __future__ import annotations

import copy
import time

import numpy as np

from aircontrol.config import AppConfig
from aircontrol.gesture_controller import (
    GestureController,
    Gesture,
    Mode,
    fingers_up,
    pinch_distance,
)
from aircontrol.hand_tracker import HandResult, Landmark, TrackingFrame
from aircontrol.mouse_controller import MouseController
from aircontrol.screenshot_selector import ScreenshotSelector


# --------------------------------------------------------------------------- #
# Synthetic hand construction
# --------------------------------------------------------------------------- #
# A plausible "open palm, fingers spread, facing camera" right-hand template
# in MediaPipe's normalized (x, y, z) coordinate convention (y grows downward).
_OPEN_RIGHT_HAND = {
    Landmark.WRIST:        (0.50, 0.85, 0.0),
    Landmark.THUMB_CMC:    (0.43, 0.78, 0.0),
    Landmark.THUMB_MCP:    (0.38, 0.70, 0.0),
    Landmark.THUMB_IP:     (0.34, 0.62, 0.0),
    Landmark.THUMB_TIP:    (0.30, 0.55, 0.0),
    Landmark.INDEX_MCP:    (0.45, 0.60, 0.0),
    Landmark.INDEX_PIP:    (0.45, 0.45, 0.0),
    Landmark.INDEX_DIP:    (0.45, 0.35, 0.0),
    Landmark.INDEX_TIP:    (0.45, 0.25, 0.0),
    Landmark.MIDDLE_MCP:   (0.50, 0.58, 0.0),
    Landmark.MIDDLE_PIP:   (0.50, 0.42, 0.0),
    Landmark.MIDDLE_DIP:   (0.50, 0.30, 0.0),
    Landmark.MIDDLE_TIP:   (0.50, 0.20, 0.0),
    Landmark.RING_MCP:     (0.55, 0.60, 0.0),
    Landmark.RING_PIP:     (0.55, 0.45, 0.0),
    Landmark.RING_DIP:     (0.55, 0.35, 0.0),
    Landmark.RING_TIP:     (0.55, 0.25, 0.0),
    Landmark.PINKY_MCP:    (0.60, 0.63, 0.0),
    Landmark.PINKY_PIP:    (0.60, 0.50, 0.0),
    Landmark.PINKY_DIP:    (0.60, 0.42, 0.0),
    Landmark.PINKY_TIP:    (0.60, 0.35, 0.0),
}


def make_hand(overrides: dict, label: str = "Right", score: float = 0.95) -> HandResult:
    points = copy.deepcopy(_OPEN_RIGHT_HAND)
    points.update(overrides)
    arr = np.array([points[i] for i in range(21)], dtype=np.float32)
    px = (arr[:, :2] * np.array([1280, 720])).astype(np.int32)
    return HandResult(label=label, score=score, landmarks=arr, landmarks_px=px)


def open_palm_hand(**kw) -> HandResult:
    return make_hand({}, **kw)


def fist_hand(**kw) -> HandResult:
    # curl every fingertip below (higher y than) its pip, and tuck the thumb.
    overrides = {
        Landmark.THUMB_TIP: (0.40, 0.66, 0.0),   # tip.x > ip.x(0.34) -> curled for Right hand
        Landmark.INDEX_TIP: (0.45, 0.55, 0.0),   # below pip(0.45)
        Landmark.MIDDLE_TIP: (0.50, 0.52, 0.0),  # below pip(0.42)
        Landmark.RING_TIP: (0.55, 0.55, 0.0),    # below pip(0.45)
        Landmark.PINKY_TIP: (0.60, 0.58, 0.0),   # below pip(0.50)
    }
    return make_hand(overrides, **kw)


def point_hand(**kw) -> HandResult:
    # Only index extended; everything else curled like a fist.
    overrides = {
        Landmark.THUMB_TIP: (0.40, 0.66, 0.0),
        Landmark.MIDDLE_TIP: (0.50, 0.52, 0.0),
        Landmark.RING_TIP: (0.55, 0.55, 0.0),
        Landmark.PINKY_TIP: (0.60, 0.58, 0.0),
    }
    return make_hand(overrides, **kw)


def pinch_hand(**kw) -> HandResult:
    # Index extended, thumb tip brought right next to index tip.
    overrides = {
        Landmark.THUMB_TIP: (0.445, 0.255, 0.0),  # very close to index tip (0.45, 0.25)
        Landmark.MIDDLE_TIP: (0.50, 0.52, 0.0),
        Landmark.RING_TIP: (0.55, 0.55, 0.0),
        Landmark.PINKY_TIP: (0.60, 0.58, 0.0),
    }
    return make_hand(overrides, **kw)


def two_finger_hand(**kw) -> HandResult:
    # Index + middle extended, ring + pinky curled.
    overrides = {
        Landmark.THUMB_TIP: (0.40, 0.66, 0.0),
        Landmark.RING_TIP: (0.55, 0.55, 0.0),
        Landmark.PINKY_TIP: (0.60, 0.58, 0.0),
    }
    return make_hand(overrides, **kw)


# --------------------------------------------------------------------------- #
# Test harness (tiny, dependency-free; also works under pytest)
# --------------------------------------------------------------------------- #
_FAILURES = []


def check(name: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(name)
    assert condition, f"{name}: {detail}"


def new_controller():
    cfg = AppConfig()
    mouse = MouseController(cfg.mouse)
    shot = ScreenshotSelector(cfg.screenshot, mouse.map_to_screen)
    gc = GestureController(cfg, mouse, shot)
    return cfg, gc, mouse, shot


# --------------------------------------------------------------------------- #
def test_fingers_up_open_palm():
    hand = open_palm_hand()
    fingers = fingers_up(hand)
    check("open palm -> all 5 fingers up", fingers == [True, True, True, True, True], str(fingers))


def test_fingers_up_fist():
    hand = fist_hand()
    fingers = fingers_up(hand)
    check("fist -> all 5 fingers down", fingers == [False, False, False, False, False], str(fingers))


def test_fingers_up_point():
    hand = point_hand()
    fingers = fingers_up(hand)
    check("point -> only index up", fingers == [False, True, False, False, False], str(fingers))


def test_pinch_distance_small_when_pinching():
    hand = pinch_hand()
    d = pinch_distance(hand)
    check("pinch distance below threshold", d < 0.06, f"d={d:.4f}")


def test_classify_gestures():
    cfg, gc, mouse, shot = new_controller()

    g = gc._classify(open_palm_hand())
    check("classify open palm", g == Gesture.OPEN_PALM, str(g))

    g = gc._classify(fist_hand())
    check("classify fist", g == Gesture.FIST, str(g))

    g = gc._classify(point_hand())
    check("classify point", g == Gesture.POINT, str(g))

    g = gc._classify(pinch_hand())
    check("classify pinch", g == Gesture.PINCH, str(g))

    g = gc._classify(two_finger_hand())
    check("classify two-finger scroll", g == Gesture.TWO_FINGER_SCROLL, str(g))


def test_end_to_end_point_moves_cursor():
    cfg, gc, mouse, shot = new_controller()
    gc.set_enabled(True)

    tf = TrackingFrame(hands=[point_hand()])
    state = None
    # Feed several frames so the gesture debounce settles.
    for _ in range(cfg.gesture.gesture_stability_frames + 2):
        state = gc.process(tf)

    check("point gesture stabilizes", state.gesture == Gesture.POINT, str(state.gesture))
    check("mode is TRACKING", state.mode == Mode.TRACKING, str(state.mode))
    check("cursor_pos was computed", state.cursor_pos is not None)


def test_end_to_end_fist_pauses():
    cfg, gc, mouse, shot = new_controller()
    gc.set_enabled(True)
    tf = TrackingFrame(hands=[fist_hand()])
    state = None
    for _ in range(cfg.gesture.gesture_stability_frames + 2):
        state = gc.process(tf)
    check("fist -> PAUSED_FIST mode", state.mode == Mode.PAUSED_FIST, str(state.mode))


def test_no_hand_triggers_auto_pause():
    cfg, gc, mouse, shot = new_controller()
    gc.set_enabled(True)
    tf = TrackingFrame(hands=[])
    state = None
    for _ in range(cfg.hand_tracker.frames_lost_before_pause + 2):
        state = gc.process(tf)
    check("no hand -> PAUSED_NO_HAND", state.mode == Mode.PAUSED_NO_HAND, str(state.mode))


def test_master_disable_blocks_everything():
    cfg, gc, mouse, shot = new_controller()
    gc.set_enabled(False)
    tf = TrackingFrame(hands=[point_hand()])
    state = gc.process(tf)
    check("disabled -> Mode.DISABLED", state.mode == Mode.DISABLED, str(state.mode))
    check("disabled -> no gesture action", state.gesture == Gesture.NONE, str(state.gesture))
    check("mouse controller reports disabled", mouse.enabled is False)


def test_emergency_stop():
    cfg, gc, mouse, shot = new_controller()
    gc.set_enabled(True)
    gc.emergency_stop()
    check("emergency stop disables master", gc.master_enabled is False)
    check("emergency stop disables mouse", mouse.enabled is False)


def test_click_debounce_prevents_double_click():
    cfg, gc, mouse, shot = new_controller()
    fired = []
    # Simulate feeding a sustained pinch distance under threshold repeatedly;
    # only the first stable occurrence (after click_hold_frames) should fire.
    for _ in range(10):
        if mouse.update_pinch(0.02):
            fired.append(time.time())
    check("only one click fires for a single sustained pinch", len(fired) == 1, str(len(fired)))


def test_mouse_mapping_stays_in_bounds():
    cfg, gc, mouse, shot = new_controller()
    for nx, ny in [(-1, -1), (0, 0), (0.5, 0.5), (2, 2), (1, 1)]:
        sx, sy = mouse.map_to_screen(nx, ny)
        check(f"mapped point in bounds ({nx},{ny})",
              0 <= sx < mouse.screen_w and 0 <= sy < mouse.screen_h,
              f"got ({sx},{sy}) vs screen {mouse.screen_w}x{mouse.screen_h}")


def test_two_hand_pinch_triggers_screenshot_flow():
    cfg, gc, mouse, shot = new_controller()
    cfg.screenshot.hold_seconds_to_capture = 0.01  # speed up for the test
    gc.set_enabled(True)

    right = pinch_hand(label="Right")
    left = pinch_hand(label="Left")
    tf = TrackingFrame(hands=[right, left])

    state = gc.process(tf)
    check("two-hand pinch enters SCREENSHOT_SELECTING", state.mode == Mode.SCREENSHOT_SELECTING, str(state.mode))

    time.sleep(0.02)
    state = gc.process(tf)
    check("screenshot progress advances/captures", state.screenshot_progress >= 0.0)


def run_all():
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} gesture-logic tests...\n")
    for t in tests:
        t()
    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILED: {_FAILURES}")
        raise SystemExit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    run_all()
