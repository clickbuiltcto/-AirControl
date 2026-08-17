"""
main.py
=======
Application entry point. Wires together camera -> hand_tracker ->
gesture_controller -> mouse_controller / screenshot_selector -> ui, and
owns the real-time loop + keyboard safety controls.

Run with:  python -m aircontrol.main
       or: python run.py   (from the project root)
"""

from __future__ import annotations

import sys
import time

import cv2

from aircontrol.camera import Camera, CameraUnavailableError
from aircontrol.config import CONFIG
from aircontrol.gesture_controller import GestureController, Mode
from aircontrol.hand_tracker import HandTracker
from aircontrol.mouse_controller import MouseController
from aircontrol.screenshot_selector import ScreenshotSelector
from aircontrol.ui import FPSCounter, OverlayRenderer
from aircontrol.utils.logger import get_logger

log = get_logger("main")

# OpenCV key codes for special keys (varies slightly per platform/build;
# 27 (ESC) is standard everywhere).
KEY_ESC = 27

# cv2.waitKeyEx() codes for F8 vary by platform/OpenCV build, so we accept
# several known candidates. 'P' is always offered as a guaranteed, portable
# in-window fallback for the enable/disable toggle.
F8_CANDIDATES = {66, 119, 65470 + 8 - 1, 0xFFC5, 1114047}  # best-effort coverage


class GlobalHotkeys:
    """
    Best-effort *global* (works even if the AirControl window isn't
    focused) ESC / F8 listener using pynput. On platforms/sessions where a
    global hook isn't available (e.g. some headless/Wayland/sandboxed
    environments), this degrades gracefully — the in-window cv2 key
    handling in the main loop still provides ESC/F8 safety whenever the
    AirControl window has focus.
    """

    def __init__(self, on_stop, on_toggle):
        self.on_stop = on_stop
        self.on_toggle = on_toggle
        self._listener = None
        self._available = False
        try:
            from pynput import keyboard  # imported lazily; optional dependency at runtime

            def _on_press(key):
                try:
                    if key == keyboard.Key.esc:
                        self.on_stop()
                    elif key == keyboard.Key.f8:
                        self.on_toggle()
                except Exception as exc:  # pragma: no cover
                    log.error(f"Hotkey handler error: {exc}")

            self._listener = keyboard.Listener(on_press=_on_press)
            self._listener.start()
            self._available = True
            log.info("Global hotkeys active (ESC / F8 work even when window isn't focused).")
        except Exception as exc:
            log.warning(
                "Global hotkey listener unavailable in this environment "
                f"({exc}). Falling back to in-window ESC/F8 only - "
                "make sure the AirControl window is focused to use them."
            )

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()


class AirControlApp:
    def __init__(self):
        self.config = CONFIG
        self.camera = Camera(self.config.camera)
        self.tracker = HandTracker(self.config.hand_tracker)
        self.mouse = MouseController(self.config.mouse)
        self.screenshot = ScreenshotSelector(self.config.screenshot, self.mouse.map_to_screen)
        self.gestures = GestureController(self.config, self.mouse, self.screenshot)
        self.ui = OverlayRenderer(self.config.ui)
        self.fps_counter = FPSCounter()

        self._running = False
        self._hotkeys = GlobalHotkeys(on_stop=self._emergency_stop, on_toggle=self._toggle_enabled)

    # ------------------------------------------------------------------ #
    def _emergency_stop(self) -> None:
        self.gestures.emergency_stop()

    def _toggle_enabled(self) -> None:
        self.gestures.toggle_enabled()

    # ------------------------------------------------------------------ #
    def run(self) -> int:
        try:
            self.camera.open()
        except CameraUnavailableError as exc:
            log.error(str(exc))
            print("\n" + str(exc) + "\n", file=sys.stderr)
            return 1

        self.gestures.set_enabled(self.config.start_enabled)
        self._running = True

        window = self.config.ui.window_name
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        log.info("AirControl running. Press ESC for emergency stop, F8 to toggle, Q to quit.")

        try:
            while self._running:
                result = self.camera.read()
                fps = self.fps_counter.tick()

                if not result.ok:
                    log.warning("Camera frame read failed - retrying...")
                    time.sleep(0.05)
                    continue

                frame = result.frame
                tracking = self.tracker.process(frame)

                if self.config.ui.show_landmarks:
                    for hand in tracking.hands:
                        self.tracker.draw_hand(frame, hand)

                state = self.gestures.process(tracking)
                self.ui.draw(frame, state, camera_ok=self.camera.is_open, fps=fps)

                cv2.imshow(window, frame)

                raw_key = cv2.waitKeyEx(1)
                key = raw_key & 0xFF if raw_key != -1 else -1

                if key == KEY_ESC:
                    self._emergency_stop()
                elif key in (ord("q"), ord("Q")):
                    log.info("Quit requested.")
                    break
                elif key in (ord("p"), ord("P")) or raw_key in F8_CANDIDATES:
                    # 'P' is the guaranteed portable toggle key; F8 is
                    # honored too whenever the OpenCV build reports a code
                    # we recognize (window must be focused either way —
                    # the pynput listener above covers true global F8).
                    self._toggle_enabled()

                # Window closed via the OS 'X' button
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break

        except KeyboardInterrupt:
            log.info("Interrupted by user (Ctrl+C).")
        finally:
            self.shutdown()

        return 0

    # ------------------------------------------------------------------ #
    def shutdown(self) -> None:
        self._running = False
        self.gestures.emergency_stop()
        self._hotkeys.stop()
        self.tracker.close()
        self.camera.close()
        cv2.destroyAllWindows()
        log.info("AirControl shut down cleanly.")


def main() -> int:
    app = AirControlApp()
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
