"""Build a synthetic test clip from a still face photo with scripted speaking.

A real face (real gaze) + controllable mouth motion lets us validate the
gaze x ASD x addressee fusion without a webcam/mic. Speaking segments get
noise injected into the mouth ROI (high motion energy); quiet segments stay
static. Expected states: ATTENTIVE (quiet) -> ADDRESSING_ROBOT (speaking).
"""
import cv2
import numpy as np

from asd import MouthMotionASD
from detectors import build_detector

SRC, OUT = "test_face.jpg", "test_convo.mp4"
MAX_SIDE, FPS, N = 640, 15, 60
SPEAK = set(range(20, 40))  # frames where the person is "speaking"

img = cv2.imread(SRC)
h, w = img.shape[:2]
s = MAX_SIDE / max(h, w)
img = cv2.resize(img, (int(w * s), int(h * s)))
h, w = img.shape[:2]

det = build_detector("mtcnn")
boxes, _ = det.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
x1, y1, x2, y2 = [int(v) for v in box]
mw, mh = int(0.25 * (x2 - x1)), int(0.60 * (y2 - y1))
mx1, my1 = x1 + mw, y1 + mh
mx2, my2 = x1 + int(0.75 * (x2 - x1)), y1 + int(0.95 * (y2 - y1))

rng = np.random.default_rng(0)
writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
for i in range(N):
    frame = img.copy()
    if i in SPEAK:  # inject mouth motion
        roi = frame[my1:my2, mx1:mx2].astype(np.float32)
        roi += rng.normal(0, 40, roi.shape)
        frame[my1:my2, mx1:mx2] = np.clip(roi, 0, 255).astype(np.uint8)
    writer.write(frame)
writer.release()
print(f"wrote {OUT}: {N} frames, speaking frames {min(SPEAK)}-{max(SPEAK)}")
