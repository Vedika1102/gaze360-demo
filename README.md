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
- `gaze.py` — detection → estimation → rendering pipeline and CLI.

## Credits & license

- **Gaze360** — Kellnhofer et al., *Gaze360: Physically Unconstrained Gaze
  Estimation in the Wild*, ICCV 2019.
- **L2CS-Net** — Abdelrahman et al., *L2CS-Net: Fine-Grained Gaze Estimation in
  Unconstrained Environments*, 2022.
- Face detection via [facenet-pytorch](https://github.com/timesler/facenet-pytorch) MTCNN.

Model weights are for **non-commercial research use** per the upstream Gaze360
license. This demo code is provided for research/educational use.
