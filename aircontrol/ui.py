"""
ui.py
=====
Renders the professional on-camera HUD overlay: status panel, gesture
label, mode, screenshot progress ring, FPS counter, and the live
two-hand screenshot selection rectangle.

Kept fully separate from gesture logic so the visual design can evolve
(themes, layouts, future "presentation mode" HUD, etc.) without ever
touching gesture_controller.py.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from aircontrol.config import UIConfig
from aircontrol.gesture_controller import GestureState, Mode, Gesture

_MODE_COLOR = {
    Mode.DISABLED: (110, 110, 110),
    Mode.PAUSED_NO_HAND: (0, 165, 255),
    Mode.PAUSED_FIST: (0, 165, 255),
    Mode.TRACKING: (0, 255, 170),
    Mode.SCROLLING: (255, 210, 0),
    Mode.SCREENSHOT_SELECTING: (255, 120, 255),
}


class FPSCounter:
    def __init__(self, smoothing: float = 0.9):
        self._smoothing = smoothing
        self._fps = 0.0
        self._last_t = time.time()

    def tick(self) -> float:
        now = time.time()
        dt = now - self._last_t
        self._last_t = now
        if dt > 0:
            inst_fps = 1.0 / dt
            self._fps = self._smoothing * self._fps + (1 - self._smoothing) * inst_fps if self._fps else inst_fps
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps


class OverlayRenderer:
    def __init__(self, config: UIConfig):
        self.config = config

    # ------------------------------------------------------------------ #
    def draw(self, frame: np.ndarray, state: GestureState, camera_ok: bool, fps: float,
              hands_drawn: bool = True) -> np.ndarray:
        h, w = frame.shape[:2]
        cfg = self.config

        self._draw_top_panel(frame, state, camera_ok, fps)
        self._draw_active_region_guide(frame, state)

        if state.mode == Mode.SCREENSHOT_SELECTING and state.screenshot_rect:
            self._draw_screenshot_rect(frame, state)

        if state.message:
            self._draw_toast(frame, state.message)

        self._draw_footer(frame)
        return frame

    # ------------------------------------------------------------------ #
    def _panel(self, frame: np.ndarray, x: int, y: int, w: int, h: int, alpha: Optional[float] = None) -> None:
        alpha = self.config.panel_alpha if alpha is None else alpha
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def _draw_top_panel(self, frame: np.ndarray, state: GestureState, camera_ok: bool, fps: float) -> None:
        cfg = self.config
        h, w = frame.shape[:2]
        panel_h = 118
        self._panel(frame, 0, 0, w, panel_h)

        # Title
        cv2.putText(frame, "AirControl", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    cfg.accent_color, 2, cv2.LINE_AA)

        # Camera status dot
        cam_color = cfg.accent_color if camera_ok else cfg.danger_color
        cv2.circle(frame, (150, 22), 6, cam_color, -1, cv2.LINE_AA)
        cv2.putText(frame, "Camera", (162, 27), cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale,
                    cfg.text_color, 1, cv2.LINE_AA)

        # Master enabled status
        en_color = cfg.accent_color if state.master_enabled else cfg.danger_color
        en_text = "ENABLED" if state.master_enabled else "DISABLED (F8 to enable)"
        cv2.putText(frame, en_text, (16, 55), cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale,
                    en_color, 2, cv2.LINE_AA)

        # Mode
        mode_color = _MODE_COLOR.get(state.mode, cfg.text_color)
        cv2.putText(frame, f"Mode: {state.mode.value}", (16, 78), cv2.FONT_HERSHEY_SIMPLEX,
                    cfg.font_scale, mode_color, 1, cv2.LINE_AA)

        # Gesture
        cv2.putText(frame, f"Gesture: {state.gesture.value}", (16, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    cfg.font_scale, cfg.text_color, 1, cv2.LINE_AA)

        # Hands detected + confidence, right-aligned-ish block
        hands_txt = f"Hands: {state.hands_detected}"
        conf_txt = "Confident" if state.hand_confident else "Low-confidence"
        cv2.putText(frame, hands_txt, (w - 260, 28), cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale,
                    cfg.text_color, 1, cv2.LINE_AA)
        conf_color = cfg.accent_color if state.hand_confident else cfg.warn_color
        cv2.putText(frame, conf_txt, (w - 260, 50), cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale,
                    conf_color, 1, cv2.LINE_AA)

        # FPS
        fps_color = cfg.accent_color if fps >= 20 else cfg.warn_color
        cv2.putText(frame, f"FPS: {fps:4.1f}", (w - 260, 72), cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale,
                    fps_color, 1, cv2.LINE_AA)

        # Mouse control status
        mouse_status = "MOUSE: ACTIVE" if state.mode in (Mode.TRACKING, Mode.SCROLLING) and state.master_enabled \
            else "MOUSE: HOLD"
        mouse_color = cfg.accent_color if "ACTIVE" in mouse_status else cfg.warn_color
        cv2.putText(frame, mouse_status, (w - 260, 94), cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale,
                    mouse_color, 1, cv2.LINE_AA)

        # Screenshot status
        if state.screenshot_active:
            pct = int(state.screenshot_progress * 100)
            cv2.putText(frame, f"SCREENSHOT: HOLD {pct}%", (w - 260, 112), cv2.FONT_HERSHEY_SIMPLEX,
                        cfg.font_scale, (255, 120, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "SCREENSHOT: idle", (w - 260, 112), cv2.FONT_HERSHEY_SIMPLEX,
                        cfg.font_scale, cfg.text_color, 1, cv2.LINE_AA)

    def _draw_active_region_guide(self, frame: np.ndarray, state: GestureState) -> None:
        """Faint rectangle showing the active control region mapped to the screen."""
        h, w = frame.shape[:2]
        # These margins mirror MouseConfig defaults visually; purely a UX aid.
        mx, my = 0.18, 0.15
        x1, y1 = int(w * mx), int(h * my)
        x2, y2 = int(w * (1 - mx)), int(h * (1 - my))
        color = (90, 90, 90)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    def _draw_screenshot_rect(self, frame: np.ndarray, state: GestureState) -> None:
        # screenshot_rect corners are in SCREEN pixel space; we only have
        # frame pixel space here, so draw a stylized progress indicator +
        # dimension readout instead of a 1:1 rectangle (frame != screen).
        (ax, ay), (bx, by) = state.screenshot_rect
        w_screen = abs(bx - ax)
        h_screen = abs(by - ay)

        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        radius = 46
        color = self.config.accent_color

        cv2.circle(frame, (cx, cy), radius, (60, 60, 60), 4, cv2.LINE_AA)
        end_angle = int(360 * state.screenshot_progress) - 90
        cv2.ellipse(frame, (cx, cy), (radius, radius), 0, -90, end_angle, color, 4, cv2.LINE_AA)
        cv2.putText(frame, "HOLD BOTH PINCHES", (cx - 110, cy - radius - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{w_screen}x{h_screen} px  |  ESC to cancel", (cx - 130, cy + radius + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.config.text_color, 1, cv2.LINE_AA)

    def _draw_toast(self, frame: np.ndarray, message: str) -> None:
        h, w = frame.shape[:2]
        (tw, th), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        x = (w - tw) // 2
        y = h - 50
        self._panel(frame, x - 16, y - th - 12, tw + 32, th + 24, alpha=0.6)
        cv2.putText(frame, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    self.config.accent_color, 2, cv2.LINE_AA)

    def _draw_footer(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        text = "ESC: Emergency Stop   |   F8 / P: Toggle AirControl   |   Q: Quit"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        self._panel(frame, 0, h - 34, w, 34, alpha=0.5)
        cv2.putText(frame, text, ((w - tw) // 2, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    self.config.text_color, 1, cv2.LINE_AA)
