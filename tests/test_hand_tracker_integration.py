"""
tests/test_hand_tracker_integration.py
========================================
Integration test that exercises the REAL MediaPipe pipeline through
HandTracker (aircontrol/hand_tracker.py), instead of the hand-built
landmark arrays used in tests/test_gesture_logic.py.

test_gesture_logic.py verifies gesture *logic* using synthetic HandResult
objects and never instantiates mediapipe.solutions.hands.Hands at all -
that's a real gap: it means the actual parsing code in HandTracker.process()
(color conversion, model inference, multi_hand_landmarks / multi_handedness
-> HandResult/TrackingFrame conversion) is never run by the test suite, so
a mediapipe/numpy/opencv version mismatch, an API shape change, or a bug in
the parsing itself would not be caught.

This test needs no webcam, no display, and no real hand in frame - it feeds
synthetic image arrays straight into the real HandTracker and asserts the
output has exactly the shape/dtype/interface the rest of the codebase
(gesture_controller.py) assumes, and that a real (empty) TrackingFrame
integrates correctly with GestureController.

Run with:  python -m pytest tests/ -v
       or: python tests/test_hand_tracker_integration.py
"""

from __future__ import annotations

import numpy as np

from aircontrol.config import AppConfig
from aircontrol.gesture_controller import GestureController, Mode
from aircontrol.hand_tracker import HandTracker, Landmark, TrackingFrame, HandResult
from aircontrol.mouse_controller import MouseController
from aircontrol.screenshot_selector import ScreenshotSelector

# --------------------------------------------------------------------------- #
# Test harness (same tiny dependency-free runner as test_gesture_logic.py;
# also works under pytest)
# --------------------------------------------------------------------------- #
_FAILURES = []


def check(name: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(name)
    assert condition, f"{name}: {detail}"


def black_frame(w: int = 640, h: int = 480) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def noise_frame(w: int = 640, h: int = 480, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


# --------------------------------------------------------------------------- #
def test_real_hand_tracker_initializes_and_closes():
    cfg = AppConfig()
    tracker = HandTracker(cfg.hand_tracker)
    tracker.close()
    check("real HandTracker initializes and closes without error", True)


def test_real_mediapipe_processes_blank_frame():
    cfg = AppConfig()
    with HandTracker(cfg.hand_tracker) as tracker:
        result = tracker.process(black_frame())
        check("blank frame returns a TrackingFrame", isinstance(result, TrackingFrame))
        check("blank frame -> no hands detected", result.count == 0, str(result.count))
        check("hands is a list", isinstance(result.hands, list))


def test_real_mediapipe_processes_noise_frames_at_multiple_sizes():
    cfg = AppConfig()
    with HandTracker(cfg.hand_tracker) as tracker:
        for w, h in [(320, 240), (640, 480), (1280, 720)]:
            result = tracker.process(noise_frame(w, h, seed=w + h))
            check(f"noise frame {w}x{h} processed without crash",
                  isinstance(result, TrackingFrame) and result.count >= 0,
                  f"count={result.count}")
            # If the real model ever *does* claim a detection on noise, the
            # resulting HandResult must still satisfy the shape/dtype
            # contract gesture_controller.py relies on.
            for hand in result.hands:
                check("landmarks shape is (21, 3)", hand.landmarks.shape == (21, 3), str(hand.landmarks.shape))
                check("landmarks_px shape is (21, 2)", hand.landmarks_px.shape == (21, 2), str(hand.landmarks_px.shape))
                check("label is Left/Right/Unknown", hand.label in ("Left", "Right", "Unknown"), hand.label)
                check("score is in [0, 1]", 0.0 <= hand.score <= 1.0, str(hand.score))


def test_real_hand_tracker_draw_hand_uses_real_connections():
    """
    draw_hand() iterates mediapipe's real HAND_CONNECTIONS constant against
    our HandResult/Landmark indices. This is never exercised by the
    synthetic gesture-logic tests since they don't touch HandTracker at
    all - if mediapipe ever changed its landmark topology/count, this would
    catch it (an IndexError) instead of only surfacing live, on camera.
    """
    cfg = AppConfig()
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[:, 0] = np.linspace(0.3, 0.7, 21)
    lm[:, 1] = np.linspace(0.3, 0.7, 21)
    px = (lm[:, :2] * np.array([640, 480])).astype(np.int32)
    hand = HandResult(label="Right", score=0.9, landmarks=lm, landmarks_px=px)

    with HandTracker(cfg.hand_tracker) as tracker:
        frame = black_frame()
        tracker.draw_hand(frame, hand)
        check("draw_hand runs against real HAND_CONNECTIONS without error", True)
        check("draw_hand does not resize the frame", frame.shape == (480, 640, 3), str(frame.shape))


def test_real_empty_tracking_frame_integrates_with_gesture_controller():
    """
    End-to-end: a real (empty) TrackingFrame produced by the actual
    HandTracker.process() pipeline, fed into the actual GestureController -
    not a hand-built TrackingFrame like the rest of the suite uses. Confirms
    the two real modules agree on the TrackingFrame/HandResult interface.
    """
    cfg = AppConfig()
    mouse = MouseController(cfg.mouse)
    shot = ScreenshotSelector(cfg.screenshot, mouse.map_to_screen)
    gc = GestureController(cfg, mouse, shot)
    gc.set_enabled(True)

    with HandTracker(cfg.hand_tracker) as tracker:
        state = None
        for _ in range(cfg.hand_tracker.frames_lost_before_pause + 2):
            tracking = tracker.process(black_frame())
            state = gc.process(tracking)

        check("real empty TrackingFrame drives GestureController to PAUSED_NO_HAND",
              state.mode == Mode.PAUSED_NO_HAND, str(state.mode))
        check("no cursor motion from an empty real tracking frame", state.cursor_pos is None)


def run_all():
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} hand-tracker integration tests...\n")
    for t in tests:
        t()
    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILED: {_FAILURES}")
        raise SystemExit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    run_all()
