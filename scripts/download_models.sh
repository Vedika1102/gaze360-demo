#!/usr/bin/env bash
# Download the model weights needed to RUN the system, then export ONNX.
# Datasets (MPIIFaceGaze, Columbia) are separate — see README "Reproducing evals".
#
#   bash scripts/download_models.sh
#
# Requires: curl, and the project deps installed (gdown for the TalkNet weight).
set -e
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
PY="${PYTHON:-python}"   # override with PYTHON=./.venv/Scripts/python.exe on Windows/uv

mkdir -p models talknet

echo "[1/4] L2CS gaze weights (Gaze360-trained) -> gaze360_model.pth.tar"
[ -f gaze360_model.pth.tar ] || curl -L -o gaze360_model.pth.tar \
  "https://huggingface.co/smoky1496/gaze360/resolve/main/gaze360.pkl?download=true"

echo "[2/4] YuNet face detector -> models/"
[ -f models/face_detection_yunet_2023mar.onnx ] || curl -L -o \
  models/face_detection_yunet_2023mar.onnx \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

echo "[3/4] TalkNet ASD weights -> talknet/pretrain_TalkSet.model"
[ -f talknet/pretrain_TalkSet.model ] || "$PY" -m gdown \
  "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea" -O talknet/pretrain_TalkSet.model

echo "[4/4] Export L2CS -> ONNX (fp32 + int8) for the fast CPU backend"
[ -f models/l2cs.onnx ] || "$PY" export_onnx.py

echo "done. run:  $PY gaze.py --source 0            (live gaze)"
echo "            $PY converse.py --source 0 --controller   (full stack)"
