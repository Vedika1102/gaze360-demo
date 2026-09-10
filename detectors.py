"""Pluggable face detectors with a common interface.

Each detector exposes:  detect(rgb) -> (boxes_xyxy: np.ndarray[N,4], probs: np.ndarray[N])

- MTCNNDetector: facenet-pytorch MTCNN (accurate, heavier on CPU).
- YuNetDetector : OpenCV YuNet ONNX (very fast on CPU).
"""
import numpy as np


class MTCNNDetector:
    def __init__(self, device="cpu", image_size=224):
        from facenet_pytorch import MTCNN
        self.det = MTCNN(keep_all=True, device=device, image_size=image_size,
                         post_process=False, select_largest=False)

    def detect(self, rgb):
        boxes, probs = self.det.detect(rgb)
        if boxes is None:
            return np.empty((0, 4)), np.empty((0,))
        probs = np.array([0.0 if p is None else p for p in probs])
        return np.asarray(boxes, dtype=np.float32), probs


class YuNetDetector:
    def __init__(self, model_path="models/face_detection_yunet_2023mar.onnx",
                 score_thresh=0.6):
        import cv2
        self.cv2 = cv2
        self.det = cv2.FaceDetectorYN.create(model_path, "", (320, 320),
                                             score_thresh, 0.3, 5000)

    def detect(self, rgb):
        bgr = self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(bgr)
        if faces is None or len(faces) == 0:
            return np.empty((0, 4)), np.empty((0,))
        boxes = faces[:, :4].copy()
        boxes[:, 2] += boxes[:, 0]  # x+w
        boxes[:, 3] += boxes[:, 1]  # y+h
        return boxes.astype(np.float32), faces[:, 14].astype(np.float32)


def build_detector(name, device="cpu"):
    if name == "yunet":
        return YuNetDetector()
    return MTCNNDetector(device=device)
