"""
Eye‑ & Hand‑Tracking Demo (macOS)
================================

Center‑Edge Calibration + Live Metrics
--------------------------------------

* **Alternating dots:** centre → edge → centre → next edge … so you can judge
  drift instantly.
* **Keyboard ‘P’** acts as a virtual pinch, **‘R’** spawns an extra refine dot.
* **Metrics overlay** shows total samples (`N`) and overall RMSE in pixels.
"""

from __future__ import annotations

import math, random, sys, types
from collections import deque
from pathlib import Path
from typing import List, Tuple, Deque

try:
    import cv2  # type: ignore
    import mediapipe as mp  # type: ignore
    import numpy as np  # type: ignore
except ModuleNotFoundError as e:
    sys.exit(f"Missing dependency: {e.name}. Install with pip.")

# --------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE_OUTER, RIGHT_EYE_OUTER = 33, 263
INDEX_TIP, THUMB_TIP = 8, 4

PINCH_THRESHOLD = 0.04
SMOOTH_WINDOW = 10
COOLDOWN_FRAMES = 3
PINCH_DUPLICATES = 10
MIN_GRID_SAMPLES = 20
DATA_PATH = Path("calibration_data.npz")

KEY_ESC, KEY_PINCH, KEY_REFINE = 27, ord("p"), ord("r")
FONT = cv2.FONT_HERSHEY_SIMPLEX  # type: ignore[attr-defined]

SMOOTH_HISTORY: Deque[Tuple[float, float]] = deque(maxlen=SMOOTH_WINDOW)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# --------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------

def iris_center(lm, idxs: List[int], w: int, h: int):
    xs, ys = zip(*[(lm[i].x * w, lm[i].y * h) for i in idxs])
    return float(np.mean(xs)), float(np.mean(ys))


def roll_angle(lm, w: int, h: int):
    lx, ly = lm[LEFT_EYE_OUTER].x * w, lm[LEFT_EYE_OUTER].y * h
    rx, ry = lm[RIGHT_EYE_OUTER].x * w, lm[RIGHT_EYE_OUTER].y * h
    return math.atan2(ry - ly, rx - lx)


def raw_gaze(lm, w: int, h: int):
    try:
        lx, ly = lm[LEFT_EYE_OUTER].x * w, lm[LEFT_EYE_OUTER].y * h
        rx, ry = lm[RIGHT_EYE_OUTER].x * w, lm[RIGHT_EYE_OUTER].y * h
    except IndexError:
        return None
    inter = math.hypot(rx - lx, ry - ly)
    if inter < 1:
        return None
    ic_l = iris_center(lm, LEFT_IRIS, w, h)
    ic_r = iris_center(lm, RIGHT_IRIS, w, h)
    ic_mid = ((ic_l[0] + ic_r[0]) * .5, (ic_l[1] + ic_r[1]) * .5)
    origin = ((lx + rx) * .5, (ly + ry) * .5)
    vx, vy = (ic_mid[0] - origin[0]) / inter, (ic_mid[1] - origin[1]) / inter
    th = -roll_angle(lm, w, h)
    c, s = math.cos(th), math.sin(th)
    return vx * c - vy * s, vx * s + vy * c


def is_pinched(hand):
    if not hand:
        return False
    try:
        d = math.hypot(hand.landmark[INDEX_TIP].x - hand.landmark[THUMB_TIP].x,
                       hand.landmark[INDEX_TIP].y - hand.landmark[THUMB_TIP].y)
        return d < PINCH_THRESHOLD
    except IndexError:
        return False

# quadratic mapping

def _phi(x, y):
    return [1, x, y, x * x, x * y, y * y]


def fit_quadratic(raw: np.ndarray, tgt: np.ndarray):
    F = np.array([_phi(*v) for v in raw])
    bx, *_ = np.linalg.lstsq(F, tgt[:, 0], rcond=None)
    by, *_ = np.linalg.lstsq(F, tgt[:, 1], rcond=None)
    return bx.astype(float), by.astype(float)


def map_quadratic(bx, by, vec):
    f = _phi(*vec)
    return float(np.dot(bx, f)), float(np.dot(by, f))


def rmse(bx, by, raw, tgt):
    if raw.size == 0:
        return 0.0
    pred = np.array([map_quadratic(bx, by, v) for v in raw])
    err = np.hypot(pred[:, 0] - tgt[:, 0], pred[:, 1] - tgt[:, 1])
    return float(np.sqrt(np.mean(err ** 2)))

# targets

def center_edge_seq(w, h):
    cols = np.linspace(0.1, 0.9, 5)
    rows = np.linspace(0.15, 0.85, 4)
    pts = [(int(c * w), int(r * h)) for r in rows for c in cols]
    centre = min(pts, key=lambda p: abs(p[0] - w/2) + abs(p[1] - h/2))
    pts.remove(centre)
    ordered = []
    for p in pts:
        ordered.append(centre)
        ordered.append(p)
    return ordered


def random_target(w, h):
    margin = 0.1
    return int(random.uniform(margin, 1 - margin) * w), int(random.uniform(margin, 1 - margin) * h)

# persistence

def load_data():
    if DATA_PATH.exists():
        d = np.load(DATA_PATH)
        return d['raw'], d['tgt']
    return np.empty((0, 2)), np.empty((0, 2))


def save_data(raw, tgt):
    np.savez(DATA_PATH, raw=raw, tgt=tgt)

# --------------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------------

def demo():
    raw_ds, tgt_ds = load_data()
    calib_ready = raw_ds.shape[0] >= MIN_GRID_SAMPLES
    bx = by = None
    current_rmse = 0.0
    if calib_ready:
        bx, by = fit_quadratic(raw_ds, tgt_ds)
        current_rmse = rmse(bx, by, raw_ds, tgt_ds)

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # type: ignore
    if not cap.isOpened():
        sys.exit("Camera not available")

    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1,
                              refine_face_landmarks=True,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holo:
        prev_pinched = False
        cooldown = 0
        seq, idx = [], 0
        refine_mode, current_target = False, None

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            if not seq:
                seq = center_edge_seq(w, h)

            res = holo.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # type: ignore
            raw_vec = raw_gaze(res.face_landmarks.landmark, w, h) if res.face_landmarks else None

            key = cv2.waitKey(5) & 0xFF
            if key == KEY_ESC:
                break
            if key == KEY_REFINE and calib_ready and not refine_mode:
                current_target = random_target(w, h)
                refine_mode = True

            hand_pinch = is_pinched(res.left_hand_landmarks) or is_pinched(res.right_hand_landmarks)
            pinched = hand_pinch or (key == KEY_PINCH)
            new_pinch = pinched and not prev_pinched and cooldown == 0
            prev_pinched = pinched
            if cooldown:
                cooldown -= 1

            # draw dot
            if not calib_ready:
                tx, ty = seq[idx]
                cv2.circle(frame, (tx, ty), 10, (0, 0, 255), -1)
            elif refine_mode and current_target:
                cv2.circle(frame, current_target, 10, (0, 0, 255), -1)

            # handle pinch actions
            if new_pinch and raw_vec is not None:
                if not calib_ready:
                    tgt = seq[idx]
                    idx += 1
                    if idx == len(seq):
                        calib_ready = True
                    add_sample = True
                elif refine_mode and current_target:
                    tgt = current_target
                    refine_mode = False
                    current_target = None
                    add_sample = True
                else:
                    add_sample = False
                if add_sample:
                    raw_dup = np.tile(raw_vec, (PINCH_DUPLICATES, 1))
                    tgt_dup = np.tile(tgt, (PINCH_DUPLICATES, 1))
                    raw_ds = np.vstack([raw_ds, raw_dup])
                    tgt_ds = np.vstack([tgt_ds, tgt_dup])
                    bx, by = fit_quadratic(raw_ds, tgt_ds)
                    save_data(raw_ds, tgt_ds)
                    current_rmse = rmse(bx, by, raw_ds, tgt_ds)
                    cooldown = COOLDOWN_FRAMES

            # live tracking
            if calib_ready and raw_vec is not None and bx is not None:
                gx, gy = map_quadratic(bx, by, raw_vec)
                SMOOTH_HISTORY.append((gx, gy))
                sx, sy = np.mean(SMOOTH_HISTORY, axis=0)
                cv2.rectangle(frame, (int(sx) - 15, int(sy) - 15), (int(sx) + 15, int(sy) + 15), (0, 255, 255), 2)

            # metrics overlay
            cv2.putText(frame, f"N:{raw_ds.shape[0]}  RMSE:{current_rmse:.1f}px", (10, 25), FONT, .6, (0, 255, 0), 2)

            # hand overlay
            if res.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, res.left_hand_landmarks, mp_hands.HAND_CONNECTIONS)  # type: ignore
            if res.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, res.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)  # type: ignore

            cv2.imshow("Eye/Hand Tracking", frame)

    cap.release()
    cv2.destroyAllWindows()

# --------------------------------------------------------------------------------
if __name__ == "__main__":
    demo()
