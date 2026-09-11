"""Addressee-aware perception runner: who is talking TO the robot?

Fuses gaze (Gaze360/L2CS) + visual active-speaker detection into an
interpretable per-person addressee state and a robot conversational role,
overlaid on video and logged per-frame.

    python converse.py --source clip.mp4 --output out.mp4 --log run.jsonl
    python converse.py --source 0                      # webcam
"""
import argparse
import json
import os
import time

import cv2

from addressee import AddresseeFusion
from asd import MouthMotionASD
from gaze import GazePipeline

_STATE_COLOR = {
    "ADDRESSING_ROBOT": (0, 0, 255),   # red   - talking to the robot
    "ATTENTIVE": (0, 200, 0),          # green - looking, not speaking
    "BYSTANDER": (150, 150, 150),      # gray  - not engaged
}
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def annotate(frame, results, states, role, primary):
    for r in results:
        s = states.get(r["id"])
        if not s:
            continue
        x1, y1, x2, y2 = r["box"]
        col = _STATE_COLOR[s["state"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        star = "*" if r["id"] == primary else ""
        cv2.putText(frame, f"#{r['id']}{star} {s['state']}", (x1, max(y1 - 24, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
        cv2.putText(frame, f"cone {s['cone_deg']:.0f}d spk {s['speak_score']:.1f}",
                    (x1, max(y1 - 8, 26)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1,
                    cv2.LINE_AA)
    banner = f"ROBOT: {role}" + (f"  (primary #{primary})" if primary is not None else "")
    cv2.putText(frame, banner, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255) if role == "ACTIVE" else (200, 200, 0), 2, cv2.LINE_AA)


def process_frame(frame, pipe, asd, fusion, t):
    results, _ = pipe.infer(frame, t=t)
    per_person = []
    for r in results:
        score, speaking = asd.update(r["id"], frame, r["box"])
        per_person.append({**r, "speaking": speaking, "speak_score": score})
    role, primary, states = fusion.update(per_person)
    annotate(frame, results, states, role, primary)
    return role, primary, states


def main():
    ap = argparse.ArgumentParser(description="Addressee-aware perception runner")
    ap.add_argument("--source", default="0")
    ap.add_argument("--weights", default="gaze360_model.pth.tar")
    ap.add_argument("--output", default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--detector", default="mtcnn", choices=["mtcnn", "yunet"])
    ap.add_argument("--cone-deg", type=float, default=25.0)
    ap.add_argument("--speak-thresh", type=float, default=6.0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    pipe = GazePipeline(args.weights, device=args.device, smooth=True,
                        lowlight=True, detector=args.detector, min_prob=0.6)
    asd = MouthMotionASD(speak_thresh=args.speak_thresh)
    fusion = AddresseeFusion(cone_deg=args.cone_deg)
    log_fp = open(args.log, "w") if args.log else None

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")
    writer = None
    if args.output:
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (w, h))
    show = writer is None
    t0, idx = time.time(), 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.time()
            role, primary, states = process_frame(frame, pipe, asd, fusion, now - t0)
            if log_fp:
                log_fp.write(json.dumps({"frame": idx, "t": round(now - t0, 3),
                                         "role": role, "primary": primary,
                                         "people": states}) + "\n")
            idx += 1
            if writer is not None:
                writer.write(frame)
            if show:
                cv2.imshow("addressee (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"wrote {idx} frames -> {args.output}")
        if log_fp:
            log_fp.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
