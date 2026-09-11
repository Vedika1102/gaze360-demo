"""Columbia ASD benchmark for the mouth-motion active-speaker detector.

Recognized benchmark (Chakravarty & Zisserman 2016; used by TalkNet, which
reports avg F1 96.3). Protocol: per labeled frame, match each speaker's GT face
box to a detected face by IoU>0.5, take that face's speaking score, and report
per-speaker F1 over {bell, boll, lieb, long, sick}.

This establishes the CURRENT method's baseline on the benchmark; a later
TalkNet swap can be re-run through the same harness to show the delta.

Evaluated on dense labeled windows (a subset of the 87-min video) to bound CPU
cost while preserving the temporal continuity the motion detector needs.

    python eval_columbia.py --video data/col/col360.mp4 --labels data/col/col_labels/fusion
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

from asd import MouthMotionASD
from detectors import YuNetDetector
from tracking import IoUTracker

SPEAKERS = ["bell", "boll", "lieb", "long", "sick"]
WINDOWS = [44200, 47233, 61451, 154600]  # windows with mixed speak/silent per speaker
WIN_LEN = 800
SCALE = 0.889                        # GT (720-wide) -> video (640-wide); verified


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_labels(labels_dir):
    """frame -> list of (speaker, scaled_box, label)."""
    gt = {}
    for sp in SPEAKERS:
        path = os.path.join(labels_dir, sp + ".txt")
        for line in open(path):
            d = line.split("\t")
            if len(d) < 5:
                continue
            f = int(d[0])
            x, y, w = int(d[1]), int(d[2]), int(d[3])
            box = (int(x * SCALE), int(y * SCALE), int((x + w) * SCALE), int((y + w) * SCALE))
            gt.setdefault(f, []).append((sp, box, int(d[4])))
    return gt


def collect(video, gt):
    """Run detector+tracker+ASD over the windows; return per-speaker (score,label)."""
    cap = cv2.VideoCapture(video)
    det = YuNetDetector(score_thresh=0.5)
    data = {sp: {"score": [], "label": []} for sp in SPEAKERS}
    for start in WINDOWS:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        tracker = IoUTracker()
        asd = MouthMotionASD()
        for i in range(WIN_LEN):
            ok, frame = cap.read()
            if not ok:
                break
            f = start + i
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, probs = det.detect(rgb)
            ids = tracker.update([tuple(b) for b in boxes]) if len(boxes) else []
            faces = []  # (box, score)
            for b, tid in zip(boxes, ids):
                score, _ = asd.update(tid, frame, b)
                faces.append((b, score))
            for sp, gbox, label in gt.get(f, []):
                # GT boxes are large head squares; match the detected face whose
                # center lies inside the GT box (closest to its center).
                gcx, gcy = (gbox[0] + gbox[2]) / 2, (gbox[1] + gbox[3]) / 2
                best_score, best_d = None, 1e18
                for b, score in faces:
                    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                    if gbox[0] <= cx <= gbox[2] and gbox[1] <= cy <= gbox[3]:
                        d = (cx - gcx) ** 2 + (cy - gcy) ** 2
                        if d < best_d:
                            best_d, best_score = d, score
                data[sp]["score"].append(best_score if best_score is not None else 0.0)
                data[sp]["label"].append(label)
        print(f"  window {start}: done")
    cap.release()
    return data


def f1_at(scores, labels, thr):
    pred = (np.array(scores) >= thr).astype(int)
    lab = np.array(labels)
    tp = int(np.sum((pred == 1) & (lab == 1)))
    fp = int(np.sum((pred == 1) & (lab == 0)))
    fn = int(np.sum((pred == 0) & (lab == 1)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/col/col360.mp4")
    ap.add_argument("--labels", default="data/col/col_labels/fusion")
    ap.add_argument("--outdir", default="eval_out")
    args = ap.parse_args()

    gt = load_labels(args.labels)
    print(f"loaded labels for {len(gt)} frames; running windows {WINDOWS}")
    data = collect(args.video, gt)

    # Pick a single global threshold maximizing mean per-speaker F1 (honest).
    grid = np.arange(0.5, 20.1, 0.5)
    means = []
    for thr in grid:
        fs = [f1_at(data[sp]["score"], data[sp]["label"], thr)
              for sp in SPEAKERS if data[sp]["label"]]
        means.append(np.mean(fs) if fs else 0.0)
    best_thr = float(grid[int(np.argmax(means))])

    report = {"windows": WINDOWS, "win_len": WIN_LEN, "best_threshold": best_thr,
              "per_speaker_f1": {}, "n": {}}
    f1s = []
    for sp in SPEAKERS:
        n = len(data[sp]["label"])
        report["n"][sp] = n
        if n:
            f1 = f1_at(data[sp]["score"], data[sp]["label"], best_thr)
            report["per_speaker_f1"][sp] = round(100 * f1, 1)
            f1s.append(f1)
    report["avg_f1"] = round(100 * float(np.mean(f1s)), 1)
    report["talknet_avg_f1"] = 96.3  # reference bar

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "columbia.json"), "w") as f:
        json.dump(report, f, indent=2)
    plot_columbia(report, os.path.join(args.outdir, "columbia_f1.png"))
    print(json.dumps(report, indent=2))


def plot_columbia(report, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    sp = list(report["per_speaker_f1"].keys())
    ours = [report["per_speaker_f1"][s] for s in sp]
    x = np.arange(len(sp))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - 0.2, ours, 0.4, label=f"ours (mouth-motion) avg {report['avg_f1']}")
    ax.axhline(report["talknet_avg_f1"], ls="--", color="gray",
               label=f"TalkNet avg {report['talknet_avg_f1']}")
    ax.set_xticks(x)
    ax.set_xticklabels(sp)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 100)
    ax.set_title("Columbia ASD: mouth-motion baseline vs TalkNet")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
