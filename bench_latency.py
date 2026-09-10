"""Latency benchmark across detector x backend combinations (CPU).

Measures per-stage timing (detect / estimate) and reports p50/p95 total ms
and FPS for each configuration, so the speedups from swapping MTCNN->YuNet and
PyTorch->ONNX-INT8 are quantified.

    python export_onnx.py            # produce models/l2cs*.onnx first
    python bench_latency.py --source test_face.jpg
"""
import argparse
import json
import os

import cv2
import numpy as np

from gaze import GazePipeline

CONFIGS = [
    {"name": "mtcnn + torch",      "detector": "mtcnn", "backend": "torch", "min_prob": 0.90},
    {"name": "yunet + torch",      "detector": "yunet", "backend": "torch", "min_prob": 0.60},
    {"name": "yunet + onnx-fp32",  "detector": "yunet", "backend": "onnx",  "min_prob": 0.60,
     "onnx": "models/l2cs.onnx"},
    {"name": "yunet + onnx-int8",  "detector": "yunet", "backend": "onnx",  "min_prob": 0.60,
     "onnx": "models/l2cs.int8.onnx"},
]


def cap(img, max_side=640):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return img if s >= 1 else cv2.resize(img, (int(w * s), int(h * s)))


def bench_one(cfg, weights, img, n, warmup):
    pipe = GazePipeline(weights, device="cpu", smooth=False, lowlight=False,
                        detector=cfg["detector"], backend=cfg["backend"],
                        onnx_path=cfg.get("onnx", "models/l2cs.int8.onnx"),
                        min_prob=cfg["min_prob"])
    for _ in range(warmup):
        pipe.infer(img.copy())
    det, est, faces = [], [], 0
    for _ in range(n):
        res, _ = pipe.infer(img.copy())
        faces = len(res)
        det.append(pipe.timings["detect_ms"])
        est.append(pipe.timings["estimate_ms"])
    total = [d + e for d, e in zip(det, est)]
    p = lambda xs, q: round(float(np.percentile(xs, q)), 2)
    return {
        "name": cfg["name"], "faces": faces,
        "detect_ms_p50": p(det, 50), "estimate_ms_p50": p(est, 50),
        "total_ms_p50": p(total, 50), "total_ms_p95": p(total, 95),
        "fps_p50": round(1000.0 / p(total, 50), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="test_face.jpg")
    ap.add_argument("--weights", default="gaze360_model.pth.tar")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--outdir", default="eval_out")
    args = ap.parse_args()

    img = cap(cv2.imread(args.source))
    if img is None:
        raise SystemExit(f"Could not read {args.source}")

    rows = []
    for cfg in CONFIGS:
        if cfg["backend"] == "onnx" and not os.path.exists(cfg.get("onnx", "")):
            print(f"skip {cfg['name']}: run export_onnx.py first")
            continue
        r = bench_one(cfg, args.weights, img, args.n, args.warmup)
        rows.append(r)
        print(f"{r['name']:20} det {r['detect_ms_p50']:7.1f}  est {r['estimate_ms_p50']:7.1f}"
              f"  total {r['total_ms_p50']:7.1f}ms (p95 {r['total_ms_p95']:.0f})  ~{r['fps_p50']} FPS")

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "latency.json"), "w") as f:
        json.dump(rows, f, indent=2)

    if rows:
        base = rows[0]["total_ms_p50"]
        print(f"\nspeedup vs baseline ({rows[0]['name']}):")
        for r in rows:
            print(f"  {r['name']:20} {base / r['total_ms_p50']:.1f}x")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4))
            names = [r["name"] for r in rows]
            det = [r["detect_ms_p50"] for r in rows]
            est = [r["estimate_ms_p50"] for r in rows]
            ax.bar(names, det, label="detect")
            ax.bar(names, est, bottom=det, label="estimate")
            ax.set_ylabel("latency p50 (ms)")
            ax.set_title("Gaze pipeline latency by configuration (CPU)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(args.outdir, "latency.png"), dpi=130)
        except Exception:
            pass


if __name__ == "__main__":
    main()
