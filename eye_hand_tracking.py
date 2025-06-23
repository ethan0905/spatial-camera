"""
Eye‑ & Hand‑Tracking Demo (macOS)
================================

*Calibration update — July 2025*

This release adds an **interactive eye‑calibration phase** so the yellow square
follows your gaze far more accurately. Calibration is triggered automatically
at startup and takes ~15 s:

1. A series of **red dots** appear one‑by‑one around the webcam window (corners
   + edges + centre).
2. **Look at the dot** and perform a **pinch gesture** (touch thumb & index‑tip)
   with either hand to validate.
3. Repeat until all dots are confirmed. The program fits an affine transform
   and switches to live gaze tracking.

Hand tracking and pinch detection reuse MediaPipe Hands, so no extra packages
are required. Screen‑recording permission is still **not** needed.

Install requirements
--------------------
```bash
python3 -m pip install opencv-python mediapipe numpy
```

Run live demo:
```bash
python eye_hand_tracking.py
```
Run tests only:
```bash
python eye_hand_tracking.py --test
```

Press **ESC** at any time to quit.
"""

from __future__ import annotations

import sys
import types
from collections import deque
from typing import Deque, List, Tuple

# -----------------------------------------------------------------------------
# Stub‑out micropip for plain CPython
# -----------------------------------------------------------------------------
try:
    import micropip  # type: ignore
except ModuleNotFoundError:  # pragma: no‑cover
    sys.modules["micropip"] = types.ModuleType("micropip")  # type: ignore

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
# Constants & MediaPipe helpers
# -----------------------------------------------------------------------------
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE_OUTER, RIGHT_EYE_OUTER = 33, 263

PINCH_THRESHOLD = 0.04  # distance in normalised coords to register a pinch
GAZE_SMOOTHING_ALPHA = 0.25
SMOOTH_HISTORY: Deque[Tuple[float, float]] = deque(maxlen=1)

mp_drawing = mp.solutions.drawing_utils  # type: ignore[attr-defined]
mp_holistic = mp.solutions.holistic  # type: ignore[attr-defined]
mp_hands = mp.solutions.hands  # type: ignore[attr-defined]

# -----------------------------------------------------------------------------
# Eye‑tracking helpers
# -----------------------------------------------------------------------------

def iris_center(lm, idxs: List[int], w: int, h: int) -> Tuple[float, float]:
    xs, ys = zip(*[(lm[i].x * w, lm[i].y * h) for i in idxs])
    return float(np.mean(xs)), float(np.mean(ys))


def raw_gaze_vector(lm, w: int, h: int) -> Tuple[float, float] | None:
    """Return raw normalised gaze vector (no gain/offset)."""
    try:
        lx, ly = lm[LEFT_EYE_OUTER].x * w, lm[LEFT_EYE_OUTER].y * h
        rx, ry = lm[RIGHT_EYE_OUTER].x * w, lm[RIGHT_EYE_OUTER].y * h
    except IndexError:
        return None

    inter = np.hypot(rx - lx, ry - ly)
    if inter < 1:
        return None

    ic_l = iris_center(lm, LEFT_IRIS, w, h)
    ic_r = iris_center(lm, RIGHT_IRIS, w, h)
    ic_mid = ((ic_l[0] + ic_r[0]) * 0.5, (ic_l[1] + ic_r[1]) * 0.5)
    origin = ((lx + rx) * 0.5, (ly + ry) * 0.5)
    vec = ((ic_mid[0] - origin[0]) / inter, (ic_mid[1] - origin[1]) / inter)
    return vec


# -----------------------------------------------------------------------------
# Pinch detection helper
# -----------------------------------------------------------------------------

INDEX_TIP, THUMB_TIP = 8, 4

def is_pinched(hand, thresh: float = PINCH_THRESHOLD) -> bool:
    if not hand:
        return False
    try:
        d = np.hypot(
            hand.landmark[INDEX_TIP].x - hand.landmark[THUMB_TIP].x,
            hand.landmark[INDEX_TIP].y - hand.landmark[THUMB_TIP].y,
        )
        return d < thresh
    except IndexError:
        return False


# -----------------------------------------------------------------------------
# Calibration utilities
# -----------------------------------------------------------------------------

def calibration_targets(w: int, h: int) -> List[Tuple[int, int]]:
    """Return list of pixel positions to show calibration dots (9‑point grid)."""
    return [
        (int(0.1 * w), int(0.1 * h)),
        (int(0.5 * w), int(0.1 * h)),
        (int(0.9 * w), int(0.1 * h)),
        (int(0.9 * w), int(0.5 * h)),
        (int(0.9 * w), int(0.9 * h)),
        (int(0.5 * w), int(0.9 * h)),
        (int(0.1 * w), int(0.9 * h)),
        (int(0.1 * w), int(0.5 * h)),
        (int(0.5 * w), int(0.5 * h)),
    ]


def fit_affine(xs: List[float], ys: List[float], x_targets: List[float], y_targets: List[float]):
    """Solve pixel = a * raw + b for x & y independently."""
    A = np.vstack([xs, np.ones(len(xs))]).T
    (a_x, b_x), _ = np.linalg.lstsq(A, x_targets, rcond=None)[0], None  # type: ignore
    (a_y, b_y), _ = np.linalg.lstsq(A, y_targets, rcond=None)[0], None  # type: ignore
    return (float(a_x), float(b_x)), (float(a_y), float(b_y))


# -----------------------------------------------------------------------------
# Main demo with calibration
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
    ) as holistic:

        # --------- Calibration phase ---------
        calib_raw: List[Tuple[float, float]] = []
        calib_targets = []
        affine_x: Tuple[float, float] | None = None
        affine_y: Tuple[float, float] | None = None
        calib_done = False

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)  # type: ignore[arg-type]

            # Initialise target list once we know frame size
            if not calib_targets:
                calib_targets = calibration_targets(w, h)

            # Draw current calibration dot or gaze square depending on phase
            if not calib_done:
                tgt_idx = len(calib_raw)
                tx, ty = calib_targets[tgt_idx]
                cv2.circle(frame, (tx, ty), 10, (0, 0, 255), -1)  # red dot
            
            # Gaze vector and pinch detection available in both phases
            raw_vec: Tuple[float, float] | None = None
            if res.face_landmarks:
                raw_vec = raw_gaze_vector(res.face_landmarks.landmark, w, h)

            pinch = is_pinched(res.left_hand_landmarks) or is_pinched(res.right_hand_landmarks)

            if not calib_done:
                # Wait for pinch to capture sample
                if pinch and raw_vec is not None:
                    calib_raw.append(raw_vec)
                    cv2.circle(frame, (tx, ty), 14, (0, 255, 0), 2)  # green flash
                    if len(calib_raw) == len(calib_targets):
                        # Fit affine transform
                        xs, ys = zip(*calib_raw)
                        xt, yt = zip(*calib_targets)
                        affine_x, affine_y = fit_affine(xs, ys, xt, yt)
                        calib_done = True
                        print("Calibration complete! Switching to live gaze mode…")
            else:
                # ---------- Live tracking ----------
                if raw_vec is not None and affine_x and affine_y:
                    gx = affine_x[0] * raw_vec[0] + affine_x[1]
                    gy = affine_y[0] * raw_vec[1] + affine_y[1]

                    # Smooth
                    if SMOOTH_HISTORY:
                        prev = SMOOTH_HISTORY[0]
                        gx = GAZE_SMOOTHING_ALPHA * gx + (1 - GAZE_SMOOTHING_ALPHA) * prev[0]
                        gy = GAZE_SMOOTHING_ALPHA * gy + (1 - GAZE_SMOOTHING_ALPHA) * prev[1]
                        SMOOTH_HISTORY.clear()
                    SMOOTH_HISTORY.append((gx, gy))

                    cv2.rectangle(
                        frame,
                        (int(gx) - 15, int(gy) - 15),
                        (int(gx) + 15, int(gy) + 15),
                        (0, 255, 255),
                        2,
                    )

            # Draw hand landmarks regardless of phase
            if res.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, res.left_hand_landmarks, mp_hands.HAND_CONNECTIONS)  # type: ignore[arg-type]
            if res.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, res.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)  # type: ignore[arg-type]

            cv2.imshow("Eye & Hand Tracking — ESC to quit", frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

# -----------------------------------------------------------------------------
# Unit tests
# -----------------------------------------------------------------------------

def _test_affine() -> None:
    xs = [0, 1]
    xt = [100, 300]
    ys = [0, 1]
    yt = [200, 600]
    (a_x, b_x), (a_y, b_y) = fit_affine(xs, ys, xt, yt)
    assert abs(a_x * 0 + b_x - 100) < 1e-5
    assert abs(a_x * 1 + b_x - 300) < 1e-5
    assert abs(a_y * 0 + b_y - 200) < 1e-5
    assert abs(a_y * 1 + b_y - 600) < 1e-5


def _run_tests() -> None:
    _test_affine()
    print("All calibration helpers passed.")

# -----------------------------------------------------------------------------
# Entry‑point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Eye & Hand tracking demo with calibration")
    p.add_argument("--test", action="store_true", help="run unit tests and exit")
    args = p.parse_args()

    if args.test:
        _run_tests()
    else:
        demo()
