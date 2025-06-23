#!/usr/bin/env python3
# eye_hand_tracking.py  –  smooth eye- & hand-tracking demo for macOS
# -------------------------------------------------------------------
import cv2, math, time, os, itertools
import numpy as np
from collections import deque

from filterpy.kalman import KalmanFilter          # pip install filterpy
from skimage import feature                       # pip install scikit-image
import mediapipe as mp
mp_holistic = mp.solutions.holistic

# ─────────────── tweakables ───────────────────────────────────────────────
PINCH_DUPLICATES = 10          # weight each pinch N×
COOLDOWN_FRAMES  = 3           # frames to ignore after a pinch
DATA_FILE        = "calibration_data.npz"

EDGE_PTS = [                   # clockwise edge points (norm. coords)
    (0.05,0.05),(0.5,0.05),(0.95,0.05),
    (0.95,0.5 ),(0.95,0.95),(0.5,0.95),
    (0.05,0.95),(0.05,0.5 )
]
CENTRE = (0.5,0.5)

# ─────────────── helpers ─────────────────────────────────────────────────
def subpixel_pupil(gray, cx, cy, patch=40):
    h, w = gray.shape
    x0, y0 = int(max(cx - patch/2, 0)), int(max(cy - patch/2, 0))
    patch_img = gray[y0:y0+patch, x0:x0+patch]
    edges = cv2.Canny(patch_img, 40, 80)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return cx, cy
    cnts = max(cnts, key=cv2.contourArea)
    (xc, yc), _, _ = cv2.fitEllipse(cnts)
    return x0 + xc, y0 + yc

def raw_gaze(lm, w, h, gray):
    lx, ly = lm[33].x * w,  lm[33].y * h      # outer eye corners
    rx, ry = lm[263].x * w, lm[263].y * h
    inter  = math.hypot(rx - lx, ry - ly)
    if inter < 5:                             # face too small / lost
        return None

    # coarse iris centre from iris landmarks
    idx = (474,475,476,477,469,470,471,472)
    cx = np.mean([lm[i].x for i in idx]) * w
    cy = np.mean([lm[i].y for i in idx]) * h
    cx, cy = subpixel_pupil(gray, cx, cy)     # refine

    vx, vy = (cx - (lx + rx)/2)/inter, (cy - (ly + ry)/2)/inter
    roll   = math.atan2(ry - ly, rx - lx)
    c, s   = math.cos(-roll), math.sin(-roll)
    return vx * c - vy * s, vx * s + vy * c   # de-roll, normalised

# ---- 3ʳᵈ-order polynomial mapping with RANSAC ---------------------------
def phi3(x, y):
    return [1, x, y, x*x, x*y, y*y, x**3, x**2*y, x*y**2, y**3]

def _solve3(raw, tgt):
    F = np.array([phi3(*v) for v in raw])
    bx,*_ = np.linalg.lstsq(F, tgt[:,0], rcond=None)
    by,*_ = np.linalg.lstsq(F, tgt[:,1], rcond=None)
    return bx, by

def _predict(raw, bx, by):
    F = np.array([phi3(*v) for v in raw])
    return np.c_[F @ bx, F @ by]

def fit_poly3_ransac(raw, tgt, iters=150, thr=25):
    if len(raw) < 12:                  # not enough data yet
        return _solve3(raw, tgt)
    best_inliers = np.ones(len(raw), bool)
    for _ in range(iters):
        samp = np.random.choice(len(raw), 12, replace=False)
        bx, by = _solve3(raw[samp], tgt[samp])
        err = np.linalg.norm(_predict(raw, bx, by) - tgt, axis=1)
        inl = err < thr
        if inl.sum() > best_inliers.sum():
            best_inliers = inl
    return _solve3(raw[best_inliers], tgt[best_inliers])

# ---- Kalman smoother (x, y, vx, vy) -------------------------------------
def make_kf(dt=1/60., q=1e-3, r=5):
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1,0,dt,0],
                     [0,1,0,dt],
                     [0,0,1,0 ],
                     [0,0,0,1 ]])
    kf.H = np.array([[1,0,0,0],
                     [0,1,0,0]])
    kf.Q = np.eye(4) * q
    kf.R = np.eye(2) * r
    kf.P *= 500
    return kf

# ---- calibration data ---------------------------------------------------
def load_dataset():
    if os.path.exists(DATA_FILE):
        d = np.load(DATA_FILE)
        raw = d['raw']
        tgt = d['tgt']
        ts  = d['ts'] if 'ts' in d.files else np.empty(0)
        return raw, tgt, ts
    return np.empty((0,2)), np.empty((0,2)), np.empty(0)

def save_dataset(raw, tgt, ts):
    np.savez(DATA_FILE, raw=raw, tgt=tgt, ts=ts)

# ───────────── main demo ─────────────────────────────────────────────────
def demo():
    # Camera ----------------------------------------------------------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ Could not open webcam")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH , 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    # Mediapipe -------------------------------------------------------------
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        refine_face_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=.5,
        min_tracking_confidence=.5
    )

    # State -----------------------------------------------------------------
    raw_ds, tgt_ds, ts_ds = load_dataset()
    bx = by = None
    if len(raw_ds) >= 12:
        bx, by = fit_poly3_ransac(raw_ds, tgt_ds)

    kf = make_kf()
    cooldown = 0
    pts_cycle = itertools.cycle(sum(zip(EDGE_PTS, itertools.repeat(CENTRE)), ()))
    current_dot = None
    metrics = deque(maxlen=100)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # ---------- gaze vector -------------------------------------------
        raw = None
        if res.face_landmarks:
            raw = raw_gaze(res.face_landmarks.landmark, w, h, gray)

        if raw is not None and bx is not None:
            kf.predict()
            kf.update(np.array(raw))
            pos = kf.x[:2].flatten()                     # ← flatten here
            gx, gy = _predict(np.array([pos]), bx, by)[0]
            if current_dot is not None:
                metrics.append(
                    np.hypot(gx - current_dot[0]*w, gy - current_dot[1]*h)
                )

        # ---------- draw dot ----------------------------------------------
        if current_dot is None:
            current_dot = next(pts_cycle)
        cx, cy = int(current_dot[0]*w), int(current_dot[1]*h)
        cv2.circle(frame, (cx, cy), 12, (0,0,255), -1)

        # ---------- pinch or key 'p' --------------------------------------
        pinch = False
        if res.left_hand_landmarks:
            lm = res.left_hand_landmarks.landmark
            pinch = np.hypot((lm[4].x - lm[8].x)*w,
                             (lm[4].y - lm[8].y)*h) < 40
        if cv2.waitKey(1) & 0xFF == ord('p'):
            pinch = True

        # ---------- handle pinch ------------------------------------------
        if pinch and cooldown == 0 and raw is not None:
            for _ in range(PINCH_DUPLICATES):
                raw_ds = np.vstack([raw_ds, np.array(raw)])
                tgt_ds = np.vstack([tgt_ds, np.array([cx, cy])])
                ts_ds  = np.hstack([ts_ds, time.time()])
            cooldown = COOLDOWN_FRAMES

            if len(raw_ds) >= 12:      # Fit only when enough samples
                bx, by = fit_poly3_ransac(raw_ds, tgt_ds)
            current_dot = None         # next dot

        cooldown = max(0, cooldown-1)

        # ---------- overlay metrics ---------------------------------------
        if metrics:
            rmse = (np.mean(np.square(metrics))) ** 0.5
            last = metrics[-1]
            cv2.putText(frame,
                        f"N:{len(raw_ds):3d}  RMSE:{rmse:4.1f}px  LAST:{last:4.1f}px",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,0), 2)

        cv2.imshow("Eye & Hand Tracking Demo  –  ESC quits", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:               # ESC
            break
        elif key == ord('r'):       # spawn refine dot
            current_dot = None

    save_dataset(raw_ds, tgt_ds, ts_ds)
    cap.release()
    cv2.destroyAllWindows()

# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo()

