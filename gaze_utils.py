"""Shared gaze math, confidence, and image ops used by the demo and evals."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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


def gaze_confidence(yaw_logits, pitch_logits):
    """Confidence in [0,1] from mean normalized softmax entropy of both heads.

    Low entropy (peaked distribution) -> high confidence.
    """
    def norm_conf(logits):
        p = F.softmax(logits, dim=1)
        ent = -(p * torch.log(p + 1e-9)).sum(dim=1)
        return 1.0 - ent / np.log(logits.shape[1])
    return ((norm_conf(yaw_logits) + norm_conf(pitch_logits)) / 2).clamp(0, 1)


def angles_to_vector(yaw, pitch):
    """(yaw, pitch) in radians -> 3D unit gaze direction (camera coords)."""
    return np.array([
        -np.cos(pitch) * np.sin(yaw),
        -np.sin(pitch),
        -np.cos(pitch) * np.cos(yaw),
    ])


def angular_error_deg(y1, p1, y2, p2):
    """Angle (degrees) between two gaze directions given as (yaw, pitch) rad."""
    v1, v2 = angles_to_vector(y1, p1), angles_to_vector(y2, p2)
    cos = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def auto_lowlight(bgr: np.ndarray, thresh: float = 90.0):
    """CLAHE on the L channel when the frame is dark. Returns (image, applied)."""
    if float(bgr.mean()) >= thresh:
        return bgr, False
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR), True
