"""Lightweight visual active-speaker detection (ASD) from mouth-region motion.

An interpretable, model-free stand-in for audio-visual ASD (e.g. TalkNet):
sustained temporal change in the lower-face (mouth) region indicates speaking.
Per-track rolling motion energy keeps it stable and cheap on CPU. This is a
Stage-1 proxy; a later stage can swap in TalkNet for audio-visual accuracy.
"""
from collections import defaultdict, deque

import cv2
import numpy as np


class MouthMotionASD:
    def __init__(self, patch: int = 32, window: int = 8, speak_thresh: float = 6.0):
        self.patch = patch
        self.window = window
        self.speak_thresh = speak_thresh
        self._prev = {}                       # track_id -> last mouth patch
        self._energy = defaultdict(lambda: deque(maxlen=window))

    @staticmethod
    def mouth_roi(frame, box):
        """Lower-center region of the face box (approx. the mouth area)."""
        x1, y1, x2, y2 = [int(v) for v in box]
        w, h = x2 - x1, y2 - y1
        mx1, mx2 = x1 + int(0.25 * w), x1 + int(0.75 * w)
        my1, my2 = y1 + int(0.60 * h), y1 + int(0.95 * h)
        mx1, my1 = max(mx1, 0), max(my1, 0)
        return frame[my1:my2, mx1:mx2]

    def update(self, track_id, frame, box):
        """Return (speak_score, is_speaking) for this track this frame."""
        roi = self.mouth_roi(frame, box)
        if roi.size == 0:
            return 0.0, False
        g = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY),
                       (self.patch, self.patch)).astype(np.float32)
        prev = self._prev.get(track_id)
        self._prev[track_id] = g
        if prev is None:
            return 0.0, False
        self._energy[track_id].append(float(np.mean(np.abs(g - prev))))
        score = float(np.mean(self._energy[track_id]))
        return score, score >= self.speak_thresh
