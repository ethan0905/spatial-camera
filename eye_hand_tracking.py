"""
Eye‑ & Hand‑Tracking Demo (macOS)
================================

*Cooldown & calibration update — July 2025*

Adds a **pinch‑cooldown** so one sustained pinch validates **exactly one**
calibration dot. This prevents rapid‑fire confirmations and gives you time to
move to the next target.

How it works
------------
* We detect the *rising edge* of a pinch ( `False → True`), not the pinch state
  itself. A dot is recorded **only** when the pinch first appears.
* After you release, the code is ready for the next pinch.
* No external timers are needed, but you can tweak `COOLDOWN_FRAMES` for extra
  margin.

Otherwise the workflow is unchanged: look at the red dot, pinch once, repeat.

Install / run / test commands stay the same.
"""

from __future__ import annotations

import sys
import time
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

PINCH_THRESHOLD = 0.04  # normalised dist between thumb & index tips
GAZE_SMOOTHING_ALPHA = 0.25
COOLDOWN_FRAMES = 3      # ignore extra pinch frames after a registration

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
    """Return normalised gaze vector (no gain/offset)."""
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
    return ((ic_mid[0] - origin[0]) / inter, (ic_mid[1] - origin[1]) / inter)

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
    """Pixel positions for 9‑point grid."""
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


def fit_affine(xs: List[float], ys: List[float], xt: List[float], yt: List[float]):
    A = np.vstack([xs, np.ones(len(xs))]).T
    (a_x, b_x) = np.linalg.lstsq(A, xt, rcond=None)[0]  # type: ignore
    (a_y, b_y) = np.linalg.lstsq(A, yt, rcond=None)[0]  # type: ignore
    return (float(a_x), float(b_x)), (float(a_y), float(b_y))

# -----------------------------------------------------------------------------
# Main demo with cooldown
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

        calib_raw: List[Tuple[float, float]] = []
        calib_targets: List[Tuple[int, int]] = []
        affine_x: Tuple[float, float] | None = None
        affine_y: Tuple[float, float] | None = None
        calib_done = False

        prev_pinched = False
        cooldown = 0  # frame countdown after a registration

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            if not calib_targets:
                calib_targets = calibration_targets(w, h)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)  # type: ignore[arg-type]

            # Draw red dot during calibration
            if not calib_done:
                tx, ty = calib_targets[len(calib_raw)]
                cv2.circle(frame, (tx, ty), 10, (0, 0, 255), -1)

            # Compute gaze and pinch state
            raw_vec = None
            if res.face_landmarks:
                raw_vec = raw_gaze_vector(res.face_landmarks.landmark, w, h)

            pinched_now = (
                is_pinched(res.left_hand_landmarks) or is_pinched(res.right_hand_landmarks)
            )
            just_pinched = pinched_now and not prev_pinched and cooldown == 0
            prev_pinched = pinched_now

            # Cooldown decrement
            if cooldown > 0:
                cooldown -= 1

            if not calib_done:
                if just_pinched and raw_vec is not None:
                    calib_raw.append(raw_vec)
                    cv2.circle(frame, (tx, ty), 14, (0, 255, 0), 2)
                    cooldown = COOLDOWN_FRAMES  # prevent multi‑hits
                    if len(calib_raw) == len(calib_targets):
                        xs, ys = zip(*calib_raw)
                        xt, yt = zip(*calib_targets)
                        affine_x, affine_y = fit_affine(xs, ys, xt, yt)
                        calib_done = True
                        print("Calibration complete — live gaze active…")
            else:
                if raw_vec is not None and affine_x and affine_y:
                    gx = affine_x[0] * raw_vec[0] + affine_x[1]
                    gy = affine_y[0] * raw_vec[1] + affine_y[1]

                    if SMOOTH_HISTORY:
                        px, py = SMOOTH_HISTORY[0]
                        gx = GAZE_SMOOTHING_ALPHA * gx + (1 - GAZE_SMOOTHING_ALPHA) * px
                        gy = GAZE_SMOOTHING_ALPHA * gy + (1 - GAZE_SMOOTHING_ALPHA) * py
                        SMOOTH_HISTORY.clear()
                    SMOOTH_HISTORY.append((gx, gy))

                    cv2.rectangle(
                        frame,
                        (int(gx) - 15, int(gy) - 15),
                        (int(gx) + 15, int(gy) + 15),
                        (0, 255, 255),
                        2,
                    )

            # Hand landmarks
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
# Simple tests
# -----------------------------------------------------------------------------

def _test_affine() -> None:
    xs = [0, 1]
    ys = [0, 1]
    xt = [100, 300]
    yt = [200, 600]
    (ax, bx), (ay, by) = fit_affine(xs, ys, xt, yt)
    assert abs(ax * 0 + bx - 100) < 1e-5
    assert abs(ax * 1 + bx - 300) < 1e-5
    assert abs(ay * 0 + by - 200) < 1e-5
    assert abs(ay * 1 + by - 600) < 1e-5


def _run_tests() -> None:
    _test_affine()
    print("All tests passed.")

# -----------------------------------------------------------------------------
# Entry‑point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Eye & Hand tracking demo with cooldown")
    parser.add_argument("--test", action="store_true", help="run unit tests and exit")
    args = parser.parse_args()

    if args.test:
        _run_tests()
    else:
        demo()
