"""
config.py
=========
Single source of truth for every tunable value in AirControl.

Keeping every knob in one dataclass-driven module means:
  * V1 behaviour can be tuned without touching logic code.
  * Future versions (drawing mode, presentation mode, per-app profiles,
    user-custom gestures, etc.) can layer new config sections here or
    load alternate profiles at runtime without breaking existing modules.

Nothing in here talks to hardware — it is pure data.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #
@dataclass
class CameraConfig:
    # HP Victus (and most laptops/desktops with a built-in webcam) expose the
    # integrated camera as device index 0. We try this first, then probe a
    # few more indices as a fallback in case an external camera grabbed 0.
    preferred_index: int = 0
    fallback_indices: tuple = (1, 2, 3)

    width: int = 1280
    height: int = 720
    requested_fps: int = 30

    # If the camera drops below this many consecutive successful reads before
    # first success, we consider it unavailable and raise a clear error.
    max_init_attempts: int = 30

    flip_horizontal: bool = True  # mirror view feels natural ("selfie mode")


# --------------------------------------------------------------------------- #
# Hand tracking (MediaPipe)
# --------------------------------------------------------------------------- #
@dataclass
class HandTrackerConfig:
    max_num_hands: int = 2
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.6
    model_complexity: int = 1  # 0=fast/light, 1=balanced, good for real-time

    # A hand must sustain at least this detection confidence to be considered
    # "confidently tracked" for the purposes of driving the mouse.
    confident_threshold: float = 0.65

    # Number of consecutive frames without a confident hand before AirControl
    # auto-pauses mouse control (safety requirement).
    frames_lost_before_pause: int = 8


# --------------------------------------------------------------------------- #
# Mouse control
# --------------------------------------------------------------------------- #
@dataclass
class MouseConfig:
    # --- coordinate mapping -------------------------------------------------
    # The camera frame is mirrored & mapped from an "active region" (a
    # smaller rectangle in the middle of the frame) to the full screen.
    # Shrinking the active region relative to the frame means the user does
    # not need to reach the physical edges of the camera view to reach the
    # edges of the screen — this is the "sensitivity" control.
    active_region_margin_x: float = 0.18   # fraction of frame width cropped off each side
    active_region_margin_y: float = 0.15   # fraction of frame height cropped off each side
    sensitivity: float = 1.0               # multiplier applied on top of the mapping

    # --- smoothing ------------------------------------------------------
    smoothing_factor: float = 0.55         # EMA weight for new sample (0-1, higher = snappier)
    one_euro_min_cutoff: float = 1.0
    one_euro_beta: float = 0.4
    one_euro_d_cutoff: float = 1.0

    # --- dead zone --------------------------------------------------------
    dead_zone_px: int = 4                  # ignore sub-pixel jitter smaller than this (screen px)

    # --- pinch / click ------------------------------------------------------
    pinch_distance_threshold: float = 0.055   # normalized (relative to hand size) distance
    pinch_release_threshold: float = 0.075    # hysteresis: release needs a larger gap than engage
    click_debounce_seconds: float = 0.35      # minimum time between two clicks
    click_hold_frames: int = 2                # frames pinch must be sustained before firing click

    # --- scrolling ------------------------------------------------------
    scroll_sensitivity: float = 18.0        # scroll "clicks" per unit of normalized movement
    scroll_dead_zone: float = 0.01

    # --- swipe navigation -------------------------------------------------
    swipe_min_distance: float = 0.22        # normalized horizontal distance to count as a swipe
    swipe_max_duration: float = 0.6         # seconds — must happen faster than this
    swipe_cooldown: float = 0.8             # seconds before another swipe can trigger

    # Configurable actions — future versions can remap these per-application.
    swipe_left_action: str = "prev"         # "prev" / "next" / custom hotkey string
    swipe_right_action: str = "next"


# --------------------------------------------------------------------------- #
# Screenshot (two-hand pinch selection)
# --------------------------------------------------------------------------- #
@dataclass
class ScreenshotConfig:
    output_dir: str = "screenshots"
    hold_seconds_to_capture: float = 1.0
    pinch_distance_threshold: float = 0.055
    min_region_px: int = 12                 # ignore accidental 1px selections
    countdown_ring_color: tuple = (0, 255, 170)
    rect_color: tuple = (0, 255, 170)
    filename_prefix: str = "aircontrol"


# --------------------------------------------------------------------------- #
# Gesture engine
# --------------------------------------------------------------------------- #
@dataclass
class GestureConfig:
    # A gesture must be classified identically for this many consecutive
    # frames before it is treated as "active" (debouncing / anti-flicker).
    gesture_stability_frames: int = 3

    # Finger "up" detection tolerance (normalized y-distance) between the
    # fingertip and the pip joint.
    finger_up_margin: float = 0.02


# --------------------------------------------------------------------------- #
# UI / overlay
# --------------------------------------------------------------------------- #
@dataclass
class UIConfig:
    show_landmarks: bool = True
    show_fps: bool = True
    panel_alpha: float = 0.55
    accent_color: tuple = (0, 255, 170)     # BGR
    warn_color: tuple = (0, 165, 255)
    danger_color: tuple = (60, 60, 255)
    text_color: tuple = (235, 235, 235)
    font_scale: float = 0.55
    window_name: str = "AirControl - Touchless Control"


# --------------------------------------------------------------------------- #
# Keyboard / safety
# --------------------------------------------------------------------------- #
@dataclass
class KeyBindings:
    emergency_stop: str = "esc"
    toggle_enable: str = "f8"


# --------------------------------------------------------------------------- #
# Top-level app config
# --------------------------------------------------------------------------- #
@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    hand_tracker: HandTrackerConfig = field(default_factory=HandTrackerConfig)
    mouse: MouseConfig = field(default_factory=MouseConfig)
    screenshot: ScreenshotConfig = field(default_factory=ScreenshotConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    keys: KeyBindings = field(default_factory=KeyBindings)

    start_enabled: bool = True
    target_fps: int = 30


# Singleton-style default config used across the app. Future versions can
# replace this with a profile loader (JSON/YAML per user or per application).
CONFIG = AppConfig()
