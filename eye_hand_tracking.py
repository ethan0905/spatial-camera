"""
Quick eye‑tracking + hand‑tracking demo for macOS

This version includes:
  • Graceful fallback when optional dependencies (pyautogui, micropip) are missing.
  • A stubbed ``micropip`` module so environments that implicitly import it (e.g. Pyodide/PyScript) no longer crash.
  • A ``--test`` flag with a minimal unit test for ``iris_center``.
  • A proper ``main`` guard so importing this file elsewhere does not instantly open the webcam.

Recommended requirements:
  pip install opencv-python mediapipe pyautogui numpy

Run live demo:
  python eye_hand_tracking.py

Run tests only:
  python eye_hand_tracking.py --test

Press **ESC** to quit the live demo.

Notes:
 * macOS will ask for camera **and** screen‑recording permissions the first time.
 * Accuracy improves if you sit ~50‑70 cm from the camera with good lighting.
"""

from __future__ import annotations

import sys
import types
from typing import List, Tuple

# -----------------------------------------------------------------------------
# Handle environments where 'micropip' is absent (e.g., outside Pyodide/PyScript)
# -----------------------------------------------------------------------------
try:
    import micropip  # type: ignore
except ModuleNotFoundError:
    micropip = types.ModuleType("micropip")  # type: ignore

    def _unavailable(*_args, **_kwargs):  # noqa: D401  (simple function)
        """Raise helpful error if ``micropip.install`` is called."""
        raise ModuleNotFoundError(
            "micropip is not available in this environment. "
            "Run this script with standard CPython and install packages via pip."
        )

    micropip.install = _unavailable  # type: ignore[attr-defined]
    sys.modules["micropip"] = micropip  # type: ignore[assignment]

# -----------------------------------------------------------------------------
# Third‑party libraries (catch ImportError early with helpful messages)
# -----------------------------------------------------------------------------
missing_deps: List[str] = []

try:
    import cv2  # type: ignore
except ModuleNotFoundError:
    missing_deps.append("opencv-python (cv2)")

try:
    import mediapipe as mp  # type: ignore
except ModuleNotFoundError:
    missing_deps.append("mediapipe")

try:
    import numpy as np  # type: ignore
except ModuleNotFoundError:
    missing_deps.append("numpy")

try:
    import pyautogui  # type: ignore
except ModuleNotFoundError:
    pyautogui = None  # type: ignore[assignment]
    missing_deps.append("pyautogui (optional)")

if missing_deps:
    print(
        "Missing required packages:\n  "
        + "\n  ".join(missing_deps)
        + "\nInstall them with, e.g.:\n\n    pip install "
        + " ".join(pkg.split()[0] for pkg in missing_deps)
    )
    sys.exit(1)

# -----------------------------------------------------------------------------
# Constants & helpers
# -----------------------------------------------------------------------------
SCREEN_W, SCREEN_H = (pyautogui.size() if pyautogui else (1280, 800))

# Face‑mesh landmark indices (MediaPipe)
LEFT_EYE_OUTER, LEFT_EYE_INNER = 33, 133
RIGHT_EYE_OUTER, RIGHT_EYE_INNER = 263, 362
LEFT_EYE_TOP, LEFT_EYE_BOTTOM = 159, 145
RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM = 386, 374
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

mp_drawing = mp.solutions.drawing_utils  # type: ignore[attr-defined]
mp_holistic = mp.solutions.holistic  # type: ignore[attr-defined]
mp_hands = mp.solutions.hands  # type: ignore[attr-defined]


def iris_center(
    landmarks, indices: List[int], img_w: int, img_h: int
) -> Tuple[float, float]:
    """Return (x, y) of the iris centre in image pixel coordinates."""
    xs, ys = zip(
        *[(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in indices]
    )
    return float(np.mean(xs)), float(np.mean(ys))


# -----------------------------------------------------------------------------
# Main video loop
# -----------------------------------------------------------------------------

def demo() -> None:
    cap = cv2.VideoCapture(
        0, cv2.CAP_AVFOUNDATION  # type: ignore[attr-defined]
    )  # AVFOUNDATION back‑end is fast on macOS

    if not cap.isOpened():
        print("Unable to access the camera. Is it connected and permitted?")
        sys.exit(1)

    with mp_holistic.Holistic(  # type: ignore[attr-defined]
        static_image_mode=False,
        model_complexity=1,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:  # type: ignore[call-arg]

        while True:
            success, frame = cap.read()
            if not success:
                print("Unable to read from camera.")
                break

            frame = cv2.flip(frame, 1)  # Mirror for a selfie‑view
            img_h, img_w, _ = frame.shape

            # Mediapipe expects RGB
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # type: ignore[arg-type]

            # ------------------------------------------------------------
            # Eye tracking → yellow box where the user is looking
            # ------------------------------------------------------------
            ratios = []  # (x_ratio, y_ratio) per eye
            if results.face_landmarks:
                lm = results.face_landmarks.landmark
                for outer, inner, top, bottom, iris in [
                    (
                        LEFT_EYE_OUTER,
                        LEFT_EYE_INNER,
                        LEFT_EYE_TOP,
                        LEFT_EYE_BOTTOM,
                        LEFT_IRIS,
                    ),
                    (
                        RIGHT_EYE_OUTER,
                        RIGHT_EYE_INNER,
                        RIGHT_EYE_TOP,
                        RIGHT_EYE_BOTTOM,
                        RIGHT_IRIS,
                    ),
                ]:
                    iris_cx, iris_cy = iris_center(
                        lm, iris, img_w, img_h
                    )
                    eye_w = (lm[inner].x - lm[outer].x) * img_w
                    eye_h = (lm[bottom].y - lm[top].y) * img_h
                    if eye_w <= 0 or eye_h <= 0:
                        continue

                    x_ratio = (iris_cx - lm[outer].x * img_w) / eye_w
                    y_ratio = (iris_cy - lm[top].y * img_h) / eye_h
                    ratios.append((x_ratio, y_ratio))

            if ratios:
                gaze_x_ratio = float(
                    np.clip(np.mean([r[0] for r in ratios]), 0, 1)
                )
                gaze_y_ratio = float(
                    np.clip(np.mean([r[1] for r in ratios]), 0, 1)
                )
                gaze_x = int(gaze_x_ratio * SCREEN_W)
                gaze_y = int(gaze_y_ratio * SCREEN_H)
                # Draw a 40×40 yellow box at gaze point
                cv2.rectangle(
                    frame,
                    (gaze_x - 20, gaze_y - 20),
                    (gaze_x + 20, gaze_y + 20),
                    (0, 255, 255),
                    2,
                )

            # ------------------------------------------------------------
            # Hand tracking overlays
            # ------------------------------------------------------------
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS  # type: ignore[attr-defined]
                )
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.right_hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,  # type: ignore[attr-defined]
                )

            cv2.imshow("Eye & Hand Tracking — ESC to quit", frame)
            if cv2.waitKey(5) & 0xFF == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()


# -----------------------------------------------------------------------------
# Simple unit tests
# -----------------------------------------------------------------------------

def _test_iris_center() -> None:
    class _LM:  # minimal stub
        def __init__(self, x: float, y: float):
            self.x = x
            self.y = y

    landmarks = [_LM(0, 0) for _ in range(478)]
    for idx in LEFT_IRIS:
        landmarks[idx] = _LM(0.5, 0.5)
    cx, cy = iris_center(landmarks, LEFT_IRIS, 1000, 1000)
    assert abs(cx - 500) < 1e-3 and abs(cy - 500) < 1e-3, (cx, cy)


def _run_tests() -> None:
    _test_iris_center()
    print("All unit tests passed.")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Eye & Hand tracking demo")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run simple built‑in unit tests and exit",
    )
    args = parser.parse_args()

    if args.test:
        _run_tests()
    else:
        demo()
