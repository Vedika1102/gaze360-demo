"""Live gaze-estimation demo built on L2CS-Net (Gaze360-trained).

Pipeline per frame:
    1. detect faces with MTCNN (facenet-pytorch)
    2. crop each face, resize + ImageNet-normalize
    3. L2CS-Net -> 90-bin yaw/pitch logits -> soft-argmax expectation (degrees)
    4. draw a 3D gaze arrow (projected to 2D) + face box + FPS

Usage:
    # webcam (opens a window; press q to quit)
    python gaze.py --source 0

    # a video file -> annotated .mp4
    python gaze.py --source clip.mp4 --output out.mp4

    # a single image -> annotated .jpg (headless-friendly, used for validation)
    python gaze.py --source face.jpg --output out.jpg
"""
import argparse
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN

from l2cs_model import load_l2cs

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def preprocess(face_rgb: np.ndarray, size: int) -> torch.Tensor:
    """RGB uint8 HxWx3 -> normalized 1x3xSxS float tensor."""
    img = cv2.resize(face_rgb, (size, size)).astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)


def decode_angles(yaw_logits, pitch_logits, idx):
    """Soft-argmax over 90 bins (4 deg each, centered at 0) -> radians."""
    yaw = torch.sum(F.softmax(yaw_logits, dim=1) * idx, dim=1) * 4 - 180
    pitch = torch.sum(F.softmax(pitch_logits, dim=1) * idx, dim=1) * 4 - 180
    return torch.deg2rad(yaw), torch.deg2rad(pitch)


def draw_gaze(frame, box, yaw, pitch):
    """Draw face box + a gaze arrow from the face center (yaw/pitch in rad)."""
    x1, y1, x2, y2 = [int(v) for v in box]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    length = max(x2 - x1, 1)
    dx = -length * np.sin(yaw) * np.cos(pitch)
    dy = -length * np.sin(pitch)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
    cv2.arrowedLine(frame, (cx, cy), (int(cx + dx), int(cy + dy)),
                    (0, 0, 255), 3, tipLength=0.25)
    label = f"yaw {np.degrees(yaw):+.0f} pitch {np.degrees(pitch):+.0f}"
    cv2.putText(frame, label, (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)


class GazePipeline:
    def __init__(self, weights, device, det_size, input_size):
        self.device = device
        self.input_size = input_size
        self.detector = MTCNN(keep_all=True, device=device, image_size=det_size,
                              post_process=False, select_largest=False)
        self.model = load_l2cs(weights, device=device)
        self.idx = torch.arange(90, dtype=torch.float32, device=device)

    @torch.no_grad()
    def process(self, frame_bgr):
        """Annotate a BGR frame in place; returns number of faces found."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes, _ = self.detector.detect(rgb)
        if boxes is None:
            return 0
        h, w = rgb.shape[:2]
        batch, kept = [], []
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, w), min(y2, h)
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            batch.append(preprocess(rgb[y1:y2, x1:x2], self.input_size))
            kept.append((x1, y1, x2, y2))
        if not batch:
            return 0
        yaw_l, pitch_l = self.model(torch.cat(batch).to(self.device))
        yaw, pitch = decode_angles(yaw_l, pitch_l, self.idx)
        for box, y, p in zip(kept, yaw.cpu().numpy(), pitch.cpu().numpy()):
            draw_gaze(frame_bgr, box, float(y), float(p))
        return len(kept)


def run_image(pipe, src, out):
    frame = cv2.imread(src)
    if frame is None:
        raise SystemExit(f"Could not read image: {src}")
    n = pipe.process(frame)
    out = out or "gaze_out.jpg"
    cv2.imwrite(out, frame)
    print(f"faces: {n}  ->  {out}")


def run_stream(pipe, src, out):
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {src}")
    writer = None
    if out:
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    show = writer is None  # display live only when not writing a file
    t0, frames = time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pipe.process(frame)
        frames += 1
        fps_now = frames / (time.time() - t0 + 1e-9)
        cv2.putText(frame, f"{fps_now:4.1f} FPS", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
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
    ap = argparse.ArgumentParser(description="L2CS-Net live gaze demo")
    ap.add_argument("--source", default="0", help="webcam index, video, or image")
    ap.add_argument("--weights", default="gaze360_model.pth.tar")
    ap.add_argument("--output", default=None, help="output file (video/image)")
    ap.add_argument("--det-size", type=int, default=224, help="MTCNN image_size")
    ap.add_argument("--input-size", type=int, default=224, help="L2CS input size")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    pipe = GazePipeline(args.weights, args.device, args.det_size, args.input_size)
    ext = os.path.splitext(args.source)[1].lower()
    if ext in _IMG_EXTS:
        run_image(pipe, args.source, args.output)
    else:
        run_stream(pipe, args.source, args.output)


if __name__ == "__main__":
    main()
