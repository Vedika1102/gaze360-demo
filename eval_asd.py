"""Quantitative eval of the mouth-motion active-speaker detector.

Builds an in-memory clip from a still face (no mp4 round-trip, so codec noise
does not confound the measurement) with known speaking segments, runs the
detector + MouthMotionASD, and reports frame-level accuracy / precision /
recall / F1 of raw "is speaking" vs ground truth. Onset lag from the rolling
window is expected at segment boundaries and is reported honestly.

    python eval_asd.py --source test_face.jpg
"""
import argparse

import cv2
import numpy as np

from asd import MouthMotionASD
from detectors import build_detector

# ground-truth speaking segments over N frames
N = 75
SPEAK = set(range(15, 30)) | set(range(45, 60))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="test_face.jpg")
    ap.add_argument("--max-side", type=int, default=640)
    ap.add_argument("--speak-thresh", type=float, default=6.0)
    ap.add_argument("--noise", type=float, default=40.0)
    args = ap.parse_args()

    img = cv2.imread(args.source)
    if img is None:
        raise SystemExit(f"cannot read {args.source}")
    h, w = img.shape[:2]
    s = args.max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)))

    det = build_detector("yunet")
    boxes, _ = det.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if len(boxes) == 0:
        raise SystemExit("no face detected in source")
    box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    roi = MouthMotionASD.mouth_roi(img, box)  # to locate injection region
    x1, y1, x2, y2 = [int(v) for v in box]
    mw, mh = int(0.25 * (x2 - x1)), int(0.60 * (y2 - y1))
    mx1, my1 = x1 + mw, y1 + mh
    mx2, my2 = x1 + int(0.75 * (x2 - x1)), y1 + int(0.95 * (y2 - y1))

    asd = MouthMotionASD(speak_thresh=args.speak_thresh)
    rng = np.random.default_rng(0)
    y_true, y_pred = [], []
    for i in range(N):
        frame = img.copy()
        gt = i in SPEAK
        if gt:
            patch = frame[my1:my2, mx1:mx2].astype(np.float32)
            patch += rng.normal(0, args.noise, patch.shape)
            frame[my1:my2, mx1:mx2] = np.clip(patch, 0, 255).astype(np.uint8)
        _, speaking = asd.update(0, frame, box)
        y_true.append(gt)
        y_pred.append(speaking)

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    acc = (tp + tn) / N
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    print(f"frames={N}  speaking_gt={int(y_true.sum())}")
    print(f"confusion: TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"accuracy={acc:.3f}  precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}")
    print("(boundary FN/FP expected from rolling-window onset lag)")


if __name__ == "__main__":
    main()
