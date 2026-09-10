"""Label-free robustness + performance evaluation for the gaze pipeline.

Three self-contained evals (no ground-truth dataset required):

  1. Corruption sweep  - degrade each input at 5 severities and measure the
     gaze *angular deviation* (deg) from the clean prediction, plus detection
     success and mean confidence. Answers "how gracefully does it degrade?"
  2. Temporal stability - synthesize a static-gaze clip (photometric jitter +
     sensor noise) and measure per-frame angular jitter with One-Euro
     smoothing OFF vs ON. Answers "how steady is the output?"
  3. Latency            - p50/p95 detect / estimate / total ms and FPS.

Outputs eval_out/robustness.json and eval_out/degradation.png.

Usage:
    python eval_robustness.py                 # uses test_face.jpg
    python eval_robustness.py --images data   # all images in a folder
"""
import argparse
import glob
import json
import os
import statistics as stats

import cv2
import numpy as np

from gaze import GazePipeline
from gaze_utils import angular_error_deg

SEVERITIES = [1, 2, 3, 4, 5]


# --- corruptions (severity 1..5) --------------------------------------------
def c_blur(img, s):
    k = [3, 5, 7, 11, 15][s - 1]
    return cv2.GaussianBlur(img, (k, k), 0)


def c_darken(img, s):
    f = [0.8, 0.6, 0.45, 0.3, 0.2][s - 1]
    return np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)


def c_noise(img, s):
    sigma = [5, 10, 20, 35, 50][s - 1]
    n = np.random.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def c_lowres(img, s):
    f = [2, 3, 4, 6, 8][s - 1]
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(w // f, 1), max(h // f, 1)))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def c_jpeg(img, s):
    q = [50, 35, 25, 15, 8][s - 1]
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img


def c_motion(img, s):
    k = [5, 9, 13, 17, 21][s - 1]
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kernel)


CORRUPTIONS = {"blur": c_blur, "darken": c_darken, "noise": c_noise,
               "lowres": c_lowres, "jpeg": c_jpeg, "motion": c_motion}


def top_face(results):
    """Highest-confidence face result, or None."""
    return max(results, key=lambda r: r["conf"], default=None)


def cap_resolution(img, max_side):
    """Downscale so the longer side <= max_side (representative webcam size)."""
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def eval_corruptions(pipe, images):
    np.random.seed(0)
    out = {}
    for name, fn in CORRUPTIONS.items():
        rows = []
        for sev in SEVERITIES:
            devs, confs, detected = [], [], 0
            for img in images:
                clean = top_face(pipe.infer(img.copy())[0])
                if clean is None:
                    continue
                cor = top_face(pipe.infer(fn(img.copy(), sev))[0])
                if cor is None:
                    continue
                detected += 1
                confs.append(cor["conf"])
                devs.append(angular_error_deg(
                    clean["yaw"], clean["pitch"], cor["yaw"], cor["pitch"]))
            rows.append({
                "severity": sev,
                "detect_rate": round(detected / max(len(images), 1), 3),
                "mean_angular_dev_deg": round(float(np.mean(devs)), 2) if devs else None,
                "mean_conf": round(float(np.mean(confs)), 3) if confs else None,
            })
        out[name] = rows
    return out


def eval_stability(weights, device, base_img, n=60, fps=30.0):
    """Synthesize a static-gaze clip; compare jitter smoothing OFF vs ON."""
    np.random.seed(1)
    frames = []
    for _ in range(n):
        flick = 1.0 + np.random.uniform(-0.03, 0.03)
        noisy = base_img.astype(np.float32) * flick
        noisy += np.random.normal(0, 3.0, base_img.shape)
        frames.append(np.clip(noisy, 0, 255).astype(np.uint8))

    result = {}
    for label, smooth in [("smoothing_off", False), ("smoothing_on", True)]:
        pipe = GazePipeline(weights, device=device, smooth=smooth, lowlight=False)
        yaws, pitches = [], []
        for i, f in enumerate(frames):
            r = top_face(pipe.infer(f, t=i / fps)[0])
            if r:
                yaws.append(np.degrees(r["yaw"]))
                pitches.append(np.degrees(r["pitch"]))
        result[label] = {
            "yaw_std_deg": round(float(np.std(yaws)), 3) if yaws else None,
            "pitch_std_deg": round(float(np.std(pitches)), 3) if pitches else None,
        }
    off = result["smoothing_off"]["yaw_std_deg"]
    on = result["smoothing_on"]["yaw_std_deg"]
    if off and on:
        result["yaw_jitter_reduction_pct"] = round(100 * (off - on) / off, 1)
    return result


def eval_latency(pipe, img, n=30):
    det, est = [], []
    for _ in range(n):
        pipe.infer(img.copy())
        det.append(pipe.timings["detect_ms"])
        est.append(pipe.timings["estimate_ms"])
    total = [d + e for d, e in zip(det, est)]

    def pct(xs, q):
        return round(float(np.percentile(xs, q)), 2)
    return {
        "detect_ms": {"p50": pct(det, 50), "p95": pct(det, 95)},
        "estimate_ms": {"p50": pct(est, 50), "p95": pct(est, 95)},
        "total_ms": {"p50": pct(total, 50), "p95": pct(total, 95)},
        "fps_p50": round(1000.0 / pct(total, 50), 1),
    }


def plot_degradation(corr, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, rows in corr.items():
        xs = [r["severity"] for r in rows if r["mean_angular_dev_deg"] is not None]
        ys = [r["mean_angular_dev_deg"] for r in rows if r["mean_angular_dev_deg"] is not None]
        if xs:
            ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel("corruption severity")
    ax.set_ylabel("mean gaze angular deviation (deg)")
    ax.set_title("Gaze robustness under input corruption")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return path


def main():
    ap = argparse.ArgumentParser(description="Label-free robustness eval")
    ap.add_argument("--images", default="test_face.jpg",
                    help="image file or a folder of images")
    ap.add_argument("--weights", default="gaze360_model.pth.tar")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--outdir", default="eval_out")
    ap.add_argument("--max-side", type=int, default=640,
                    help="downscale inputs so longer side <= this (webcam-like)")
    args = ap.parse_args()

    paths = ([args.images] if os.path.isfile(args.images)
             else sorted(sum([glob.glob(os.path.join(args.images, f"*{e}"))
                              for e in (".jpg", ".jpeg", ".png")], [])))
    images = [cv2.imread(p) for p in paths]
    images = [cap_resolution(im, args.max_side) for im in images if im is not None]
    if not images:
        raise SystemExit(f"No readable images at: {args.images}")
    print(f"evaluating on {len(images)} image(s)")

    pipe = GazePipeline(args.weights, device=args.device, smooth=False,
                        lowlight=False)

    report = {
        "n_images": len(images),
        "corruptions": eval_corruptions(pipe, images),
        "temporal_stability": eval_stability(args.weights, args.device, images[0]),
        "latency": eval_latency(pipe, images[0]),
    }

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "robustness.json"), "w") as f:
        json.dump(report, f, indent=2)
    plot = plot_degradation(report["corruptions"],
                            os.path.join(args.outdir, "degradation.png"))

    # concise console summary
    print("\n=== corruption robustness (mean angular deviation, deg) ===")
    print(f"{'corruption':10} " + " ".join(f"s{s}" for s in SEVERITIES))
    for name, rows in report["corruptions"].items():
        cells = [f"{(r['mean_angular_dev_deg'] or 0):5.1f}" for r in rows]
        print(f"{name:10} " + " ".join(cells))
    ts = report["temporal_stability"]
    print(f"\ntemporal jitter (yaw std deg): off={ts['smoothing_off']['yaw_std_deg']}"
          f"  on={ts['smoothing_on']['yaw_std_deg']}"
          f"  reduction={ts.get('yaw_jitter_reduction_pct')}%")
    lat = report["latency"]
    print(f"latency total p50={lat['total_ms']['p50']}ms p95={lat['total_ms']['p95']}ms"
          f"  (~{lat['fps_p50']} FPS)")
    print(f"\nwrote {args.outdir}/robustness.json" + (f" and {plot}" if plot else ""))


if __name__ == "__main__":
    main()
