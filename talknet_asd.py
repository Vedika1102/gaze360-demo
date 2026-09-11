"""TalkNet audio-visual active-speaker detection wrapper (inference only).

Vendored model in talknet/; pretrained TalkSet weights. Given a face track
(sequence of frames) and the corresponding audio, returns per-frame speaking
logits (>=0 => speaking, per TalkNet convention).

Face preprocessing follows TalkNet's demo: gray -> resize 224 -> center 112.
Audio is 16 kHz MFCC (13-dim) with 4 audio frames per video frame; winstep is
set from the video fps so the 4:1 ratio holds at native 29.97 fps.
"""
import cv2
import numpy as np
import python_speech_features
import torch

from talknet.loss import lossAV
from talknet.talkNetModel import talkNetModel


class TalkNetASD:
    def __init__(self, weights="talknet/pretrain_TalkSet.model", device="cpu",
                 fps=29.97, chunk=200):
        self.device = device
        self.fps = fps
        self.chunk = chunk
        self.model = talkNetModel().to(device).eval()
        self.lossAV = lossAV().to(device).eval()
        sd = torch.load(weights, map_location="cpu", weights_only=False)
        self.model.load_state_dict(
            {k[6:]: v for k, v in sd.items() if k.startswith("model.")}, strict=True)
        self.lossAV.load_state_dict(
            {k[7:]: v for k, v in sd.items() if k.startswith("lossAV.")}, strict=True)

    @staticmethod
    def face_crop(frame_bgr, box, margin=0.4):
        """Detected box -> TalkNet-style 112x112 gray crop (or None)."""
        x1, y1, x2, y2 = [int(v) for v in box]
        w, h = x2 - x1, y2 - y1
        x1, y1 = max(x1 - int(margin * w), 0), max(y1 - int(margin * h), 0)
        x2 = min(x2 + int(margin * w), frame_bgr.shape[1])
        y2 = min(y2 + int(margin * h), frame_bgr.shape[0])
        face = frame_bgr[y1:y2, x1:x2]
        if face.size == 0:
            return None
        g = cv2.resize(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), (224, 224))
        return g[56:168, 56:168]

    def mfcc(self, wav_16k):
        return python_speech_features.mfcc(
            wav_16k, 16000, numcep=13, winlen=0.025, winstep=1.0 / (self.fps * 4))

    @torch.no_grad()
    def score(self, faces_112, mfcc_feat):
        """faces_112: list[T] of 112x112 arrays; mfcc_feat: (>=4T,13). -> (L,) logits."""
        T = len(faces_112)
        length = min(mfcc_feat.shape[0] // 4, T)
        if length < 1:
            return np.zeros(T, dtype=np.float32)
        out_scores = []
        for s in range(0, length, self.chunk):
            e = min(s + self.chunk, length)
            v = torch.FloatTensor(np.array(faces_112[s:e])).unsqueeze(0).to(self.device)
            a = torch.FloatTensor(mfcc_feat[s * 4:e * 4]).unsqueeze(0).to(self.device)
            ea = self.model.forward_audio_frontend(a)
            ev = self.model.forward_visual_frontend(v)
            ea, ev = self.model.forward_cross_attention(ea, ev)
            o = self.model.forward_audio_visual_backend(ea, ev)
            out_scores.extend(np.asarray(self.lossAV.forward(o, labels=None)).reshape(-1))
        scores = np.array(out_scores, dtype=np.float32)
        if len(scores) < T:  # pad tail (unscored trailing frames)
            scores = np.concatenate([scores, np.full(T - len(scores), scores[-1])])
        return scores
