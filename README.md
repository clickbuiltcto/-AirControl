# AirControl

**Touchless computer control using natural hand gestures — no external hardware required.**

AirControl turns your PC's built-in webcam into a full mouse-replacement input device. Point to move the cursor, pinch to click, two-finger scroll, make a fist to pause, swipe to navigate, and pinch with both hands to select and capture a screenshot — all hands-free.

Built for an **HP Victus gaming PC's built-in webcam** (device index `0`), and designed to run comfortably at **30+ FPS** on that kind of hardware.

---

## ✨ V1 Features

| Gesture | Action |
|---|---|
| ☝️ Index finger extended | Move the mouse cursor |
| 🤏 Thumb + index pinch | Left click |
| ✌️ Index + middle finger up | Scroll (move hand up/down) |
| ✊ Closed fist | Pause / disable control (held) |
| 🖐️ Open palm | Neutral / stop (cursor holds still) |
| 👋 Fast hand swipe (left/right) | Configurable navigation action (default: browser/app back-forward) |
| 🤏🤏 Two-hand pinch | Screenshot region selection |

### Two-hand screenshot selection

1. Pinch with your **right hand** (thumb + index) → sets the **first corner**.
2. Pinch with your **left hand** (thumb + index) → sets the **opposite corner**.
3. While both hands pinch, a live progress ring + region size is shown on screen.
4. **Hold both pinches for ~1 second** to capture.
5. The screenshot is saved as a PNG in `screenshots/`.
6. Press **ESC** at any time to cancel.
7. Release both pinches after a capture before starting a new one (prevents accidental repeat captures).

---

## 🛡️ Safety — you are always in control

- **ESC** → **Emergency stop**. Immediately disables all mouse influence.
- **F8** (global, works even if the AirControl window isn't focused) or **P** (in-window fallback) → toggle AirControl ON/OFF.
- **Automatic pause** if no hand is confidently detected for a few frames — the cursor freezes instead of drifting.
- The mouse is **never** moved unless AirControl is explicitly enabled **and** a hand is confidently tracked **and** you are actively pointing — there is no background timer or idle drift.
- All cursor coordinates are clamped to the screen bounds; `pyautogui`'s corner fail-safe is also enabled as a second safety net.
- Click and swipe detection use debouncing + hysteresis so a shaky hand can't spam clicks or accidental navigation.

---

## 🖥️ Hardware requirements

- A Windows PC with a **built-in webcam** (e.g. HP Victus). No external webcam needed.
- AirControl automatically tries camera index `0` first (the standard built-in webcam index), then probes a couple of fallback indices if needed.
- If AirControl can't find a working camera, it prints a clear, actionable error explaining what to check (Windows camera privacy settings, other apps using the camera, Device Manager, etc.) instead of crashing silently.

---

## 📦 Installation

```bash
# 1. Clone / copy the AirControl folder, then from inside it:
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

**Note on the `mediapipe` version pin:** `requirements.txt` pins `mediapipe==0.10.14`. Newer MediaPipe releases removed the classic `mp.solutions.hands` API in favor of a Tasks API that requires downloading a separate `.task` model file over the network on first run. Version `0.10.14` ships its hand-tracking model **inside the pip wheel**, so AirControl works immediately after installation with **no extra downloads and no internet access required at runtime**. (See "Upgrading MediaPipe" below if you want to move to the newer API later.)

### Run it

```bash
python run.py
```

or

```bash
python -m aircontrol.main
```

A window titled **"AirControl — Touchless Control"** will open showing your camera feed with the gesture overlay. Press **Q** to quit, **ESC** for emergency stop, **F8**/**P** to toggle control on/off.

---

## ⌨️ Keyboard controls

| Key | Action |
|---|---|
| `ESC` | Emergency stop (disables all mouse control immediately) |
| `F8` | Toggle AirControl on/off (global — works even if window isn't focused, via a background hotkey listener) |
| `P` | Toggle AirControl on/off (guaranteed in-window fallback if global hotkeys aren't available on your platform/session) |
| `Q` | Quit the application |

---

## 🧠 How it works (architecture)

```
AirControl/
├── run.py                      # Entry point
├── requirements.txt
├── README.md
├── screenshots/                # Saved two-hand-pinch screenshots land here
├── tests/
│   └── test_gesture_logic.py   # Camera-free unit tests (synthetic landmarks)
└── aircontrol/
    ├── config.py                # Every tunable value — sensitivity, thresholds, colors, keybinds
    ├── camera.py                 # Webcam discovery / open / read / clean shutdown
    ├── hand_tracker.py           # MediaPipe Hands wrapper — framework isolation layer
    ├── gesture_controller.py     # Landmark -> gesture classification, debouncing, orchestration
    ├── mouse_controller.py       # Coordinate mapping, smoothing, dead zones, click/scroll, safety
    ├── screenshot_selector.py    # Two-hand pinch screenshot state machine
    ├── ui.py                     # On-camera HUD overlay (status, gesture, FPS, progress ring)
    ├── main.py                   # App wiring + real-time loop + keyboard safety
    └── utils/
        ├── smoothing.py          # One-Euro filter + EMA smoothing primitives
        └── logger.py             # Consistent timestamped logging
```

**Design principles that make this a foundation, not a demo:**

- **Strict separation of concerns.** `hand_tracker.py` is the *only* file that imports MediaPipe. `mouse_controller.py` is the *only* file that moves the OS cursor. This means the hand-tracking backend can be swapped (e.g. for a custom AI gesture model) or the mouse backend can be swapped (e.g. for a different automation library) without touching gesture logic.
- **`gesture_controller.py` has no OpenCV/window dependency**, so it's fully unit-testable with synthetic landmark arrays (see `tests/`) — no webcam or display needed to verify gesture logic.
- **`config.py` is the single source of truth** for every tunable number. Sensitivity, thresholds, colors, and key bindings can all be tuned there without touching logic code, and it's structured so a future version can load per-user or per-application profiles.
- **The screenshot selector reuses the mouse controller's coordinate mapping**, so the selection rectangle always corresponds to real screen pixels, not arbitrary camera-frame pixels.

### Mouse control internals

- **Coordinate mapping** — a shrunken "active region" in the middle of the camera frame maps to the full screen, so you don't need to reach the physical edges of the frame to reach the screen edges (`MouseConfig.active_region_margin_x/y`).
- **Smoothing** — a speed-adaptive **One-Euro filter** smooths cursor motion: heavy smoothing when your hand is nearly still (kills jitter), light smoothing during fast intentional motion (kills lag).
- **Dead zone** — sub-pixel jitter below `dead_zone_px` is ignored entirely.
- **Pinch-distance threshold + hysteresis** — click-engage and click-release use different thresholds so the pinch doesn't "chatter" around the boundary.
- **Debouncing** — gestures must be classified consistently for several consecutive frames before they're treated as active, and clicks have a minimum time between them (`click_debounce_seconds`).

---

## 🎛️ Tuning

Everything is in `aircontrol/config.py`. A few knobs you'll likely want to adjust first:

- `MouseConfig.sensitivity` — overall cursor responsiveness.
- `MouseConfig.active_region_margin_x/y` — how much of the frame edge you need to reach the screen edge.
- `MouseConfig.pinch_distance_threshold` — how close thumb+index must get to register a click.
- `HandTrackerConfig.min_detection_confidence` / `min_tracking_confidence` — raise these if you get false detections, lower them if tracking drops out too easily.
- `MouseConfig.swipe_left_action` / `swipe_right_action` — set to `"prev"`, `"next"`, or any `pyautogui`-style hotkey string like `"ctrl+left"`.

---

## 🧪 Testing

```bash
python -m pytest tests/ -v
# or, without pytest installed:
python tests/test_gesture_logic.py
```

The test suite synthesizes MediaPipe-shaped hand landmarks for each canonical pose (open palm, fist, point, pinch, two-finger) and verifies:
- finger-up detection
- gesture classification
- end-to-end state transitions (pause-on-fist, auto-pause-on-no-hand, master disable, emergency stop)
- click debounce (a single sustained pinch fires exactly one click)
- mouse coordinate mapping stays within screen bounds
- the two-hand pinch screenshot flow enters `SCREENSHOT_SELECTING` and captures correctly

No webcam or display is required to run these tests.

---

## 🚀 Roadmap — where this is headed

AirControl V1 is built as the foundation for a broader touchless-computing platform. The modular architecture is specifically designed so these can be added without rewriting existing modules:

- **Virtual drawing mode** — reuse `mouse_controller.py`'s smoothing/mapping to draw on a canvas overlay.
- **Presentation control** — swipe gestures already fire configurable actions; a presentation profile just remaps them to slide-next/prev.
- **Accessibility features** — dwell-clicking, adjustable sensitivity profiles, larger/higher-contrast HUD modes.
- **AI-based custom gesture recognition** — `hand_tracker.py` is an isolation layer; a learned gesture classifier can plug in behind the same `TrackingFrame` interface.
- **User-defined custom gestures** — `gesture_controller.py`'s classification step is centralized and can be extended with a user-trained gesture set.
- **Voice commands** — a new `voice_controller.py` module can drive the same `MouseController` / action dispatch used by gestures today.
- **Per-application control profiles** — `config.py`'s structure supports loading alternate profiles (e.g. a browser profile, a slides profile, a media-player profile).

---

## 🔧 Troubleshooting

- **"AirControl could not find a usable webcam"** — Check Windows camera privacy settings (Settings → Privacy → Camera), close other apps using the camera (Zoom, Teams, Camera app), and confirm the webcam is enabled in Device Manager.
- **Cursor feels jittery** — increase `MouseConfig.smoothing` values (`one_euro_beta` lower = smoother but more lag) or increase `dead_zone_px`.
- **Clicks fire too easily / not easily enough** — adjust `MouseConfig.pinch_distance_threshold` and `pinch_release_threshold`.
- **F8 doesn't work globally** — this depends on OS-level permissions for global key hooks. The in-window **P** key always works as a guaranteed fallback while the AirControl window is focused.

### Upgrading MediaPipe

If you want to move to a newer MediaPipe version that uses the Tasks API (`HandLandmarker`) instead of the legacy `mp.solutions.hands` used here, you'll need to: (1) download a `hand_landmarker.task` model file, (2) update `hand_tracker.py` to use `mediapipe.tasks.python.vision.HandLandmarker` instead of `mp.solutions.hands.Hands`, and (3) adjust the result-parsing code, since the Tasks API returns a slightly different result object shape. This is a contained change isolated entirely to `hand_tracker.py`.

---

## License

Internal project foundation — add your preferred license here before distribution.
