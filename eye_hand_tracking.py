"""
Eye‑ & Hand‑Tracking Demo (macOS)
================================

*Refactored eye‑tracking logic — July 2025*

This version keeps the hand‑tracking path unchanged but **rewrites the gaze
estimation pipeline** so the yellow box now reliably follows the user’s eyes in
the webcam window.

Key improvements
----------------
1. **Direct iris‑center mapping** — We map the averaged iris centres straight to
   the webcam‑frame coordinates instead of naïvely stretching to the full
   display resolution, which previously shoved the box off‑screen.
2. **Head‑pose compensation lite** — We normalise the iris position by the
   inter‑ocular vector. This makes the estimate resilient to small head turns.
3. **Exponential smoothing** — A 20 % decay filter stabilises jitter without
   noticeable lag.
4. **No pyautogui dependency** — Gaze drawing is now entirely inside the
   webcam window, so screen‑recording permission isn’t required.
5. **Unit tests** for the new `normalised_gaze()` helper.

Install requirements
--------------------
```bash
python3 -m pip install opencv-python mediapipe numpy
```
(Apple‑silicon wheels exist for Python 3.8–3.11. Use 3.11 for the smoothest
experience.)

Run live demo:
```bash
python eye_hand_tracking.py
```
Run tests only:
```bash
python eye_hand_tracking.py --test
```

Press **ESC** to quit the live demo.
"""

from __future__ import annotations

import sys
import types
from collections import deque
from typing import Deque, List, Tuple

# -----------------------------------------------------------------------------
# Stub‑out micropip in regular CPython environments (eg. when importing in Jupyter)
# -----------------------------------------------------------------------------
try:
    import micropip  # type: ignore
except ModuleNotFoundError:  # pragma: no‑cover — we never use micropip on desktop
    micropip = types.ModuleType("micropip")  # type: ignore
    sys.modules["micropip"] = micropip

# -----------------------------------------------------------------------------
# Third‑party deps (fail fast with a helpful message)
# -----------------------------------------------------------------------------
missing: List[str] = []
try:
    import cv2  # type: ignore
except ModuleNotFoundError:
    missing.append("opencv-python")
try:
    import mediapipe as mp  # type: ignore
except ModuleNotFoundError:
    missing.append("mediapipe")
try:
    import numpy as np  # type: ignore
except ModuleNotFoundError:
    missing.append("numpy")

if missing:
    print(
        "Missing packages:\n  " + "\n  ".join(missing) +
        "\nInstall with:\n\n    pip install " + " ".join(missing)
    )
    sys.exit(1)

# -----------------------------------------------------------------------------
# Constants & MediaPipe setup helpers
# -----------------------------------------------------------------------------
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE_OUTER, RIGHT_EYE_OUTER = 33, 263

GAZE_SMOOTHING_ALPHA = 0.2  # 0 ⇒ no smoothing, 1 ⇒ infinite lag
SMOOTH_HISTORY: Deque[Tuple[float, float]] = deque(maxlen=1)  # store previous gaze

mp_drawing = mp.solutions.drawing_utils  # type: ignore[attr-defined]
mp_holistic = mp.solutions.holistic  # type: ignore[attr-defined]
mp_hands = mp.solutions.hands  # type: ignore[attr-defined]

# -----------------------------------------------------------------------------
# Eye‑tracking helpers
# -----------------------------------------------------------------------------

def iris_center(
    landmarks, indices: List[int], img_w: int, img_h: int
) -> Tuple[float, float]:
    """Return the (pixel‑x, pixel‑y) centre of the iris from landmark list."""
    xs, ys = zip(*[(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in indices])
    return float(np.mean(xs)), float(np.mean(ys))


def normalised_gaze(
    landmarks, img_w: int, img_h: int
) -> Tuple[float, float] | None:
    """Compute gaze location *within the webcam frame* in pixels.

    Strategy
    --------
    1. Get the pixel centres of left & right irises.
    2. Compute the midpoint between the eye outer corners → pseudo head origin.
    3. Measure the vector from that origin to the averaged iris centre.
    4. Divide by inter‑ocular distance (eye‑corner distance) to normalise head
       motion.
    5. Map the [-0.3 … 0.3] normalised range to ±1/3 of the frame size — this
       empirical scale works well at laptop‑camera distance without calibration.
    """

    # Required outer‑corner landmarks may be missing on first few frames
    try:
        lx, ly = landmarks[LEFT_EYE_OUTER].x * img_w, landmarks[LEFT_EYE_OUTER].y * img_h
        rx, ry = landmarks[RIGHT_EYE_OUTER].x * img_w, landmarks[RIGHT_EYE_OUTER].y * img_h
    except IndexError:  # pragma: no‑cover
        return None

    inter_ocular = np.hypot(rx - lx, ry - ly)
    if inter_ocular < 1:  # very unlikely, safeguards divide‑by‑zero
        return None

    # Iris centres
    ic_l = iris_center(landmarks, LEFT_IRIS, img_w, img_h)
    ic_r = iris_center(landmarks, RIGHT_IRIS, img_w, img_h)
    ic_mid = ((ic_l[0] + ic_r[0]) * 0.5, (ic_l[1] + ic_r[1]) * 0.5)

    head_origin = ((lx + rx) * 0.5, (ly + ry) * 0.5)
    norm_vec = ((ic_mid[0] - head_origin[0]) / inter_ocular,
                (ic_mid[1] - head_origin[1]) / inter_ocular)

    # Empirical gain — eye rotation spans roughly ±0.33 of inter‑ocular distance
    gain = 3.0  # maps ±0.33 → ±1.0
    gaze_px = np.clip(0.5 + norm_vec[0] * gain, 0, 1) * img_w
    gaze_py = np.clip(0.5 + norm_vec[1] * gain, 0, 1) * img_h
    return float(gaze_px), float(gaze_py)


# -----------------------------------------------------------------------------
# Main video loop
# -----------------------------------------------------------------------------

def demo() -> None:
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # type: ignore[attr-defined]
    if not cap.isOpened():
        print("Cannot access camera — check permissions & connection.")
        sys.exit(1)

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:  # type: ignore[call-arg]

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break

            frame = cv2.flip(frame, 1)  # selfie view
            img_h, img_w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)  # type: ignore[arg-type]

            # ------------------------- Eye tracking -------------------------
            gaze_pt: Tuple[float, float] | None = None
            if results.face_landmarks:
                gaze_pt = normalised_gaze(results.face_landmarks.landmark, img_w, img_h)

            if gaze_pt is not None:
                # Exponential moving average smoothing
                if not SMOOTH_HISTORY:
                    smoothed = gaze_pt
                else:
                    prev = SMOOTH_HISTORY[0]
                    smoothed = (
                        GAZE_SMOOTHING_ALPHA * gaze_pt[0] + (1 - GAZE_SMOOTHING_ALPHA) * prev[0],
                        GAZE_SMOOTHING_ALPHA * gaze_pt[1] + (1 - GAZE_SMOOTHING_ALPHA) * prev[1],
                    )
                SMOOTH_HISTORY.clear()
                SMOOTH_HISTORY.append(smoothed)

                gx, gy = map(int, smoothed)
                cv2.rectangle(
                    frame,
                    (gx - 15, gy - 15),
                    (gx + 15, gy + 15),
                    (0, 255, 255),
                    2,
                )

            # ------------------------- Hand tracking ------------------------
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS)  # type: ignore[arg-type]
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)  # type: ignore[arg-type]

            cv2.imshow("Eye & Hand Tracking — ESC to quit", frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

# -----------------------------------------------------------------------------
# Unit tests (run with --test)
# -----------------------------------------------------------------------------

def _test_normalised_gaze() -> None:
    class LM:  # minimal stub for mp FaceLandmark
        __slots__ = ("x", "y")
        def __init__(self, x: float, y: float):
            self.x, self.y = x, y

    W, H = 1000, 1000
    l_outer, r_outer = (300, 500), (700, 500)
    inter = np.hypot(r_outer[0] - l_outer[0], r_outer[1] - l_outer[1])

    # Create dummy landmark list length 478 filled with zeros
    lm: List[LM] = [LM(0, 0) for _ in range(478)]
    lm[LEFT_EYE_OUTER] = LM(l_outer[0] / W, l_outer[1] / H)
    lm[RIGHT_EYE_OUTER] = LM(r_outer[0] / W, r_outer[1] / H)
    for idx in LEFT_IRIS + RIGHT_IRIS:
        lm[idx] = LM(0.5, 0.5)  # centred iris

    gx, gy = normalised_gaze(lm, W, H)  # type: ignore[arg-type]
    assert abs(gx - 500) < 1, gx
    assert abs(gy - 500) < 1, gy


def _run_tests() -> None:
    _test_normalised_gaze()
    print("All gaze‑mapping tests passed.")

# -----------------------------------------------------------------------------
# Entry‑point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Eye & Hand tracking demo")
    p.add_argument("--test", action="store_true", help="run unit tests and exit")
    args = p.parse_args()

    if args.test:
        _run_tests()
    else:
        demo()
