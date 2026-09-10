# Gaze360 / L2CS-Net — Live Gaze Estimation Demo

Real-time appearance-based **gaze estimation** from a webcam, video, or image.
Faces are detected with MTCNN, and gaze (yaw/pitch) is predicted with
**L2CS-Net** (ResNet-50 backbone, 90-bin soft-argmax heads), trained on the
**Gaze360** dataset. Each detected face gets a projected 3D gaze arrow.

Runs on **CPU** (no GPU required).

## Why L2CS-Net (and not the original Gaze360 LSTM)

The plan started from the Gaze360 paper (Kellnhofer et al., ICCV 2019), whose
model is a ResNet-18 + bidirectional LSTM over a 7-frame window. Its official
pretrained weights are hosted at `gaze360.csail.mit.edu`, which is currently
returning HTTP 403 for all files. **L2CS-Net** (Abdelrahman et al., 2022) is a
single-frame model trained on the same Gaze360 dataset — its weights are
mirrored on Hugging Face, it needs no temporal buffer, and it is the stronger
real-time / on-robot choice. This repo uses L2CS-Net; swapping in the Gaze360
LSTM later only requires a new `model.py` + weights.

## Setup

Uses [`uv`](https://docs.astral.sh/uv/) with Python 3.12.

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

## Weights

The L2CS-Net (Gaze360, ResNet-50) checkpoint (~96 MB):

```bash
curl -L -o gaze360_model.pth.tar \
  "https://huggingface.co/smoky1496/gaze360/resolve/main/gaze360.pkl?download=true"
```

## Run

```bash
# webcam (opens a window, press q to quit)
python gaze.py --source 0

# a video file -> annotated .mp4
python gaze.py --source clip.mp4 --output out.mp4

# a single image -> annotated .jpg
python gaze.py --source face.jpg --output out.jpg
```

Useful flags: `--input-size` (L2CS crop size, default 224), `--det-size`
(MTCNN size), `--device cuda` (if a GPU is available).

## Robustness

The pipeline is hardened for real-world input:

- **Detection resilience** — MTCNN confidence threshold (`--min-prob`) and
  minimum face size (`--min-size`); no-face and NaN-angle frames are handled
  gracefully.
- **Temporal smoothing** — per-face [One-Euro filter](https://gery.casiez.net/1euro/)
  on yaw/pitch, keyed by an IoU tracker so each person is smoothed
  independently. Cuts jitter ~60% (see eval).
- **Confidence gating** — softmax **entropy** of the yaw/pitch heads gives a
  per-face confidence in [0,1]; low-confidence gaze is flagged and grayed out
  instead of drawn as a confident (wrong) arrow.
- **Low-light correction** — automatic CLAHE on the L channel when a frame is
  dark, before detection.

Toggle with `--no-smooth`, `--no-lowlight`, `--conf-thresh`, `--min-prob`,
`--min-size`. Add `--log run.jsonl` for a structured per-frame record
(boxes, angles, confidence, stage timings).

## Evaluation

`eval_robustness.py` runs three **label-free** evals (no ground-truth dataset
required) and writes `eval_out/robustness.json` + a degradation plot:

```bash
python eval_robustness.py                 # uses test_face.jpg
python eval_robustness.py --images data   # a folder of face images
```

1. **Corruption sweep** — degrade inputs at 5 severities (blur, darken, noise,
   low-res, JPEG, motion blur) and measure gaze **angular deviation** from the
   clean prediction, plus detection rate and confidence.
2. **Temporal stability** — synthesize a static-gaze clip and measure per-frame
   angular jitter with smoothing **OFF vs ON**.
3. **Latency** — p50/p95 detect/estimate/total ms and FPS.

Representative CPU results (1 face, inputs capped at 640 px):

| corruption | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|
| blur   |  9.6 | 10.9 | 22.1 | 27.4 | 34.3 |
| darken |  8.9 |  5.8 |  4.5 |  7.8 | 14.5 |
| noise  |  3.4 |  8.5 | 63.8 | 82.0 | 104.0 |
| lowres |  4.7 | 17.2 | 19.6 | 16.2 | 34.5 |
| jpeg   |  5.2 |  1.9 |  4.6 |  8.6 |  8.8 |
| motion |  9.7 | 22.9 | 31.8 | 38.0 | 42.1 |

*(mean gaze angular deviation, degrees — lower is more robust)*

![degradation curves](docs/degradation.png)

- **Temporal smoothing:** yaw jitter 1.91° → 0.72° (**−62%**).
- **Latency:** ~199 ms/frame total (p50), ~5 FPS on CPU — MTCNN detection
  dominates; a lighter detector or GPU raises this substantially.

Takeaway: gaze is most fragile to **sensor noise** (add denoising upstream) and
robust to **JPEG/darkening**; smoothing markedly steadies the live output.

## How it works

1. **Detect** — MTCNN returns face boxes (`keep_all=True`).
2. **Preprocess** — crop each face, resize to 224², ImageNet-normalize.
3. **Estimate** — L2CS-Net outputs 90-bin logits for yaw and pitch; a softmax
   expectation over bin centers (4° each, spanning ±180°) gives continuous
   angles.
4. **Render** — convert (yaw, pitch) to a gaze direction and draw the arrow,
   face box, and per-face angle label; overlay FPS on streams.

## Files

- `l2cs_model.py` — L2CS-Net (ResNet-50 + yaw/pitch heads) + strict loader.
- `gaze_utils.py` — gaze math, entropy confidence, low-light correction.
- `filters.py` — One-Euro temporal filter.
- `tracking.py` — IoU tracker for stable per-face IDs.
- `gaze.py` — detection → estimation → tracking/smoothing → rendering + CLI.
- `eval_robustness.py` — label-free robustness/latency eval harness.

## Credits & license

- **Gaze360** — Kellnhofer et al., *Gaze360: Physically Unconstrained Gaze
  Estimation in the Wild*, ICCV 2019.
- **L2CS-Net** — Abdelrahman et al., *L2CS-Net: Fine-Grained Gaze Estimation in
  Unconstrained Environments*, 2022.
- Face detection via [facenet-pytorch](https://github.com/timesler/facenet-pytorch) MTCNN.

Model weights are for **non-commercial research use** per the upstream Gaze360
license. This demo code is provided for research/educational use.
