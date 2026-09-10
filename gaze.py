"""Live gaze-estimation demo built on L2CS-Net (Gaze360-trained), hardened.

Pipeline per frame:
    1. (optional) low-light CLAHE correction
    2. detect faces with MTCNN, filtered by confidence + min size
    3. crop each face, resize + ImageNet-normalize
    4. L2CS-Net -> 90-bin yaw/pitch -> soft-argmax angles + entropy confidence
    5. IoU tracking -> per-track One-Euro smoothing of yaw/pitch
    6. draw a gaze arrow (grayed when low-confidence) + box + id + FPS

Usage:
    python gaze.py --source 0                       # webcam window (q quits)
    python gaze.py --source clip.mp4 --output out.mp4
    python gaze.py --source face.jpg --output out.jpg
    python gaze.py --source clip.mp4 --log run.jsonl # structured per-frame log
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch

from facenet_pytorch import MTCNN

from filters import OneEuroFilter
from gaze_utils import (auto_lowlight, decode_angles, gaze_confidence,
                        preprocess)
from l2cs_model import load_l2cs
from tracking import IoUTracker

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def draw_gaze(frame, box, yaw, pitch, tid, conf, low_conf):
    """Draw face box + gaze arrow (grayed if low-confidence) + id/conf label."""
    x1, y1, x2, y2 = [int(v) for v in box]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    length = max(x2 - x1, 1)
    dx = -length * np.sin(yaw) * np.cos(pitch)
    dy = -length * np.sin(pitch)
    box_col = (128, 128, 128) if low_conf else (0, 200, 0)
    arr_col = (140, 140, 140) if low_conf else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_col, 2)
    cv2.arrowedLine(frame, (cx, cy), (int(cx + dx), int(cy + dy)),
                    arr_col, 3, tipLength=0.25)
    tag = f"#{tid} {np.degrees(yaw):+.0f}/{np.degrees(pitch):+.0f} c{conf:.2f}"
    if low_conf:
        tag += " low"
    cv2.putText(frame, tag, (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_col, 1, cv2.LINE_AA)


class GazePipeline:
    def __init__(self, weights, device="cpu", det_size=224, input_size=224,
                 min_prob=0.90, min_size=24, conf_thresh=0.55,
                 smooth=True, lowlight=True):
        self.device = device
        self.input_size = input_size
        self.min_prob = min_prob
        self.min_size = min_size
        self.conf_thresh = conf_thresh
        self.smooth = smooth
        self.lowlight = lowlight
        self.detector = MTCNN(keep_all=True, device=device, image_size=det_size,
                              post_process=False, select_largest=False)
        self.model = load_l2cs(weights, device=device)
        self.idx = torch.arange(90, dtype=torch.float32, device=device)
        self.tracker = IoUTracker()
        self._filts = {}  # tid -> (OneEuroFilter yaw, OneEuroFilter pitch)
        self.timings = {"detect_ms": 0.0, "estimate_ms": 0.0}

    def _smooth(self, tid, yaw, pitch, t):
        if tid not in self._filts:
            self._filts[tid] = (OneEuroFilter(), OneEuroFilter())
        fy, fp = self._filts[tid]
        return fy(yaw, t), fp(pitch, t)

    @torch.no_grad()
    def infer(self, frame_bgr, t=None):
        """Return list of per-face result dicts; records stage timings."""
        t = time.time() if t is None else t
        frame = frame_bgr
        lowlight_applied = False
        if self.lowlight:
            frame, lowlight_applied = auto_lowlight(frame_bgr)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        t0 = time.perf_counter()
        boxes, probs = self.detector.detect(rgb)
        self.timings["detect_ms"] = (time.perf_counter() - t0) * 1000

        kept_boxes, crops = [], []
        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if prob is None or prob < self.min_prob:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box]
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, w), min(y2, h)
                if x2 - x1 < self.min_size or y2 - y1 < self.min_size:
                    continue
                kept_boxes.append((x1, y1, x2, y2))
                crops.append(preprocess(rgb[y1:y2, x1:x2], self.input_size))

        self.timings["estimate_ms"] = 0.0
        if not crops:
            return [], lowlight_applied

        t1 = time.perf_counter()
        yaw_l, pitch_l = self.model(torch.cat(crops).to(self.device))
        yaw, pitch = decode_angles(yaw_l, pitch_l, self.idx)
        conf = gaze_confidence(yaw_l, pitch_l)
        self.timings["estimate_ms"] = (time.perf_counter() - t1) * 1000

        yaw = yaw.cpu().numpy()
        pitch = pitch.cpu().numpy()
        conf = conf.cpu().numpy()
        ids = self.tracker.update(kept_boxes)

        results = []
        for box, tid, y, p, c in zip(kept_boxes, ids, yaw, pitch, conf):
            if not (np.isfinite(y) and np.isfinite(p)):  # NaN guard
                continue
            if self.smooth:
                y, p = self._smooth(tid, float(y), float(p), t)
            results.append({
                "id": int(tid), "box": [int(v) for v in box],
                "yaw": float(y), "pitch": float(p), "conf": float(c),
                "low_conf": bool(c < self.conf_thresh),
            })
        return results, lowlight_applied

    def annotate(self, frame, results):
        for r in results:
            draw_gaze(frame, r["box"], r["yaw"], r["pitch"],
                      r["id"], r["conf"], r["low_conf"])


def _log_line(fp, frame_idx, t, results, timings):
    if fp is None:
        return
    fp.write(json.dumps({
        "frame": frame_idx, "t": round(t, 4),
        "detect_ms": round(timings["detect_ms"], 2),
        "estimate_ms": round(timings["estimate_ms"], 2),
        "faces": results,
    }) + "\n")


def run_image(pipe, src, out, log_fp):
    frame = cv2.imread(src)
    if frame is None:
        raise SystemExit(f"Could not read image: {src}")
    results, _ = pipe.infer(frame)
    pipe.annotate(frame, results)
    out = out or "gaze_out.jpg"
    cv2.imwrite(out, frame)
    _log_line(log_fp, 0, 0.0, results, pipe.timings)
    print(f"faces: {len(results)}  ->  {out}")


def run_stream(pipe, src, out, log_fp):
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {src}")
    writer = None
    if out:
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    show = writer is None
    t0, frames = time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        results, _ = pipe.infer(frame, t=now)
        pipe.annotate(frame, results)
        _log_line(log_fp, frames, now - t0, results, pipe.timings)
        frames += 1
        fps_now = frames / (now - t0 + 1e-9)
        cv2.putText(frame, f"{fps_now:4.1f} FPS  det {pipe.timings['detect_ms']:.0f}ms"
                    f"  est {pipe.timings['estimate_ms']:.0f}ms", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        if writer is not None:
            writer.write(frame)
        if show:
            cv2.imshow("L2CS gaze (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if writer is not None:
        writer.release()
        print(f"wrote {frames} frames -> {out}")
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="L2CS-Net live gaze demo (hardened)")
    ap.add_argument("--source", default="0", help="webcam index, video, or image")
    ap.add_argument("--weights", default="gaze360_model.pth.tar")
    ap.add_argument("--output", default=None, help="output file (video/image)")
    ap.add_argument("--log", default=None, help="write per-frame JSONL here")
    ap.add_argument("--det-size", type=int, default=224)
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--min-prob", type=float, default=0.90, help="MTCNN conf thresh")
    ap.add_argument("--min-size", type=int, default=24, help="min face px")
    ap.add_argument("--conf-thresh", type=float, default=0.55, help="gaze conf gate")
    ap.add_argument("--no-smooth", action="store_true")
    ap.add_argument("--no-lowlight", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    pipe = GazePipeline(
        args.weights, device=args.device, det_size=args.det_size,
        input_size=args.input_size, min_prob=args.min_prob,
        min_size=args.min_size, conf_thresh=args.conf_thresh,
        smooth=not args.no_smooth, lowlight=not args.no_lowlight)

    log_fp = open(args.log, "w") if args.log else None
    try:
        ext = os.path.splitext(args.source)[1].lower()
        if ext in _IMG_EXTS:
            run_image(pipe, args.source, args.output, log_fp)
        else:
            run_stream(pipe, args.source, args.output, log_fp)
    finally:
        if log_fp:
            log_fp.close()


if __name__ == "__main__":
    main()
