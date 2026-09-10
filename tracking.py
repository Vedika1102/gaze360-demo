"""Minimal IoU tracker for assigning stable per-face IDs across frames.

Greedy IoU matching with track aging. Enough to key per-person temporal
smoothing; not a full multi-object tracker.
"""
import numpy as np


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class IoUTracker:
    def __init__(self, iou_thresh: float = 0.3, max_age: int = 15):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self._tracks = {}   # id -> {"box": box, "age": frames_since_seen}
        self._next_id = 0

    def update(self, boxes):
        """Match boxes to tracks; returns a list of ids aligned with `boxes`."""
        for t in self._tracks.values():
            t["age"] += 1

        assigned = [-1] * len(boxes)
        used = set()
        # Greedy: best (detection, track) IoU pairs first.
        pairs = []
        for di, box in enumerate(boxes):
            for tid, t in self._tracks.items():
                iou = _iou(box, t["box"])
                if iou >= self.iou_thresh:
                    pairs.append((iou, di, tid))
        pairs.sort(reverse=True)
        for iou, di, tid in pairs:
            if assigned[di] == -1 and tid not in used:
                assigned[di] = tid
                used.add(tid)
                self._tracks[tid] = {"box": boxes[di], "age": 0}

        for di, box in enumerate(boxes):
            if assigned[di] == -1:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {"box": box, "age": 0}
                assigned[di] = tid

        # Drop stale tracks.
        for tid in [t for t, v in self._tracks.items() if v["age"] > self.max_age]:
            del self._tracks[tid]
        return assigned
