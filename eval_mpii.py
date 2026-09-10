"""Cross-dataset gaze benchmark on MPIIFaceGaze + uncertainty analysis.

The L2CS-Net checkpoint here is Gaze360-trained, so evaluating on MPIIFaceGaze
measures *cross-dataset generalization*. Ground-truth gaze is the camera-frame
vector (gaze_target - face_center) from each subject's annotation file; our
model also predicts gaze in the camera frame, so the two are directly
comparable (no head-pose normalization needed).

Reports:
  - mean / median angular error (deg), optional test-time augmentation (hflip)
  - uncertainty: correlation(confidence, error) + a selective-prediction
    risk-coverage curve and "error @ coverage" (the novel result: does the
    model know when it is wrong?)

Usage:
    python eval_mpii.py --data data/MPIIFaceGaze --per-subject 60
    python eval_mpii.py --data data/MPIIFaceGaze --per-subject 60 --tta
    python eval_mpii.py --data data/MPIIFaceGaze --check-convention   # one-time
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

from gaze import GazePipeline
from gaze_utils import angles_to_vector, angular_error_vectors

# Axis map from our drawing convention to the MPII camera frame, chosen once by
# --check-convention (see README). Applied to the predicted gaze vector.
AXIS_SIGN = np.array([1.0, 1.0, 1.0])


def parse_annotation(txt_path):
    """Yield (image_relpath, gt_gaze_vector) for each line of pXX.txt.

    Columns: 0=path, 1-2=screen px, 3-14=6 landmarks, 15-20=head pose,
    21-23=face center (mm), 24-26=3D gaze target (mm), 27=eye.
    """
    with open(txt_path) as f:
        for line in f:
            t = line.split()
            if len(t) < 27:
                continue
            fc = np.array(t[21:24], dtype=np.float64)
            gt = np.array(t[24:27], dtype=np.float64)
            yield t[0], (gt - fc)


def predict_vector(pipe, img, tta=False):
    """Highest-confidence face -> (gaze_vector, confidence) or (None, None)."""
    res = pipe.infer(img)[0]
    if not res:
        return None, None
    r = max(res, key=lambda d: d["conf"])
    v = angles_to_vector(r["yaw"], r["pitch"]) * AXIS_SIGN
    if tta:
        res_f = pipe.infer(cv2.flip(img, 1))[0]
        if res_f:
            rf = max(res_f, key=lambda d: d["conf"])
            vf = angles_to_vector(rf["yaw"], rf["pitch"]) * AXIS_SIGN
            vf[0] *= -1  # undo horizontal flip
            v = v + vf
    return v / (np.linalg.norm(v) + 1e-9), r["conf"]


def collect(pipe, data_dir, per_subject, tta):
    """Run the model over sampled images; return arrays of (error, confidence)."""
    rng = np.random.default_rng(0)
    errors, confs = [], []
    subjects = sorted(glob.glob(os.path.join(data_dir, "p*")))
    subjects = [s for s in subjects if os.path.isdir(s)]
    for sub in subjects:
        txt = os.path.join(sub, os.path.basename(sub) + ".txt")
        if not os.path.exists(txt):
            continue
        items = list(parse_annotation(txt))
        if per_subject and len(items) > per_subject:
            items = [items[i] for i in rng.choice(len(items), per_subject, replace=False)]
        for relpath, gt_vec in items:
            img = cv2.imread(os.path.join(sub, relpath.replace("/", os.sep)))
            if img is None:
                continue
            pred, conf = predict_vector(pipe, img, tta=tta)
            if pred is None:
                continue
            errors.append(angular_error_vectors(pred, gt_vec))
            confs.append(conf)
        print(f"  {os.path.basename(sub)}: {len(errors)} samples so far")
    return np.array(errors), np.array(confs)


def check_convention(pipe, data_dir, n=120):
    """One-time: report mean error for each axis-sign variant to align frames."""
    global AXIS_SIGN
    rng = np.random.default_rng(0)
    samples = []
    for sub in sorted(glob.glob(os.path.join(data_dir, "p*")))[:4]:
        txt = os.path.join(sub, os.path.basename(sub) + ".txt")
        if os.path.exists(txt):
            for relpath, gt in list(parse_annotation(txt))[:n // 4]:
                samples.append((sub, relpath, gt))
    variants = {"(+,+,+)": [1, 1, 1], "(-,+,+)": [-1, 1, 1],
                "(+,-,+)": [1, -1, 1], "(+,+,-)": [1, 1, -1],
                "(-,-,+)": [-1, -1, 1], "(-,+,-)": [-1, 1, -1],
                "(+,-,-)": [1, -1, -1], "(-,-,-)": [-1, -1, -1]}
    print("axis-sign variant -> mean angular error (deg):")
    best, best_err = None, 1e9
    for name, sign in variants.items():
        AXIS_SIGN = np.array(sign, dtype=float)
        errs = []
        for sub, relpath, gt in samples:
            img = cv2.imread(os.path.join(sub, relpath.replace("/", os.sep)))
            if img is None:
                continue
            pred, _ = predict_vector(pipe, img)
            if pred is not None:
                errs.append(angular_error_vectors(pred, gt))
        m = float(np.mean(errs)) if errs else 1e9
        print(f"  {name}: {m:6.2f}")
        if m < best_err:
            best, best_err = name, m
    print(f"best: {best} ({best_err:.2f} deg) -> set AXIS_SIGN accordingly")


def risk_coverage(errors, confs):
    """Mean error when keeping the most-confident fraction (coverage)."""
    order = np.argsort(-confs)  # high confidence first
    e = errors[order]
    curve = []
    for cov in np.linspace(0.1, 1.0, 10):
        k = max(int(round(cov * len(e))), 1)
        curve.append({"coverage": round(float(cov), 2),
                      "mean_error_deg": round(float(np.mean(e[:k])), 2)})
    return curve


def pearson(a, b):
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return round(float(np.corrcoef(a, b)[0, 1]), 3)


def main():
    ap = argparse.ArgumentParser(description="MPIIFaceGaze cross-dataset eval")
    ap.add_argument("--data", default="data/MPIIFaceGaze")
    ap.add_argument("--per-subject", type=int, default=60)
    ap.add_argument("--weights", default="gaze360_model.pth.tar")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tta", action="store_true", help="hflip test-time aug")
    ap.add_argument("--check-convention", action="store_true")
    ap.add_argument("--outdir", default="eval_out")
    args = ap.parse_args()

    if not os.path.isdir(args.data):
        raise SystemExit(f"Dataset not found at {args.data}")
    pipe = GazePipeline(args.weights, device=args.device, smooth=False,
                        lowlight=False, min_prob=0.80)

    if args.check_convention:
        check_convention(pipe, args.data)
        return

    errors, confs = collect(pipe, args.data, args.per_subject, args.tta)
    if len(errors) == 0:
        raise SystemExit("No predictions collected.")

    rc = risk_coverage(errors, confs)
    report = {
        "n_samples": int(len(errors)),
        "tta": args.tta,
        "mean_angular_error_deg": round(float(np.mean(errors)), 2),
        "median_angular_error_deg": round(float(np.median(errors)), 2),
        "corr_confidence_vs_error": pearson(confs, errors),
        "risk_coverage": rc,
        "error_at_coverage": {
            "1.0": rc[-1]["mean_error_deg"],
            "0.8": next(r["mean_error_deg"] for r in rc if r["coverage"] == 0.8),
            "0.5": next(r["mean_error_deg"] for r in rc if r["coverage"] == 0.5),
        },
    }
    os.makedirs(args.outdir, exist_ok=True)
    suffix = "_tta" if args.tta else ""
    with open(os.path.join(args.outdir, f"mpii{suffix}.json"), "w") as f:
        json.dump(report, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cov = [r["coverage"] for r in rc]
        err = [r["mean_error_deg"] for r in rc]
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.plot([c * 100 for c in cov], err, marker="o")
        ax.set_xlabel("coverage (% most-confident kept)")
        ax.set_ylabel("mean angular error (deg)")
        ax.set_title("Selective prediction: error vs coverage (MPIIFaceGaze)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, f"risk_coverage{suffix}.png"), dpi=130)
    except Exception:
        pass

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
