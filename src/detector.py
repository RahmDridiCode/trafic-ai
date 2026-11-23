"""Vehicle detection backends: custom/pretrained Ultralytics YOLO weights or MOG2 motion baseline."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# Class names of the project's trained models (informational; the real names come from model.names).
EXPECTED_CLASSES = {0: "bus", 1: "car", 2: "motorcycle", 3: "truck"}
# Used only to filter generic (e.g. COCO) models; matching is done by name, never by COCO id.
VEHICLE_NAMES = {"bus", "car", "motorbike", "motorcycle", "truck"}
# YOLO boxes smaller than this fraction of the frame are too small to be useful (distant clutter).
MIN_BOX_AREA_RATIO = 0.0005


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # x, y, w, h (top-left origin, pixels)
    centroid: Tuple[int, int]
    class_name: str
    confidence: float
    class_id: int = -1


def resolve_device(device: str = "auto") -> str:
    """'auto' -> first CUDA GPU if available else CPU; anything else is passed through."""
    device = str(device).strip().lower() if device is not None else "auto"
    if device == "auto":
        try:
            import torch
            return "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    if device == "cuda":
        return "0"
    return device


def _xyxy_to_detection(x1, y1, x2, y2, name, conf, cls_id, width, height):
    x1 = int(max(0, min(width - 1, round(x1))))
    y1 = int(max(0, min(height - 1, round(y1))))
    x2 = int(max(0, min(width - 1, round(x2))))
    y2 = int(max(0, min(height - 1, round(y2))))
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    return Detection((x1, y1, w, h), (x1 + w // 2, y1 + h // 2), name, float(conf), int(cls_id))


class VehicleDetector:
    def __init__(self, backend="yolo", model_path="yolo11n.pt", confidence=0.35,
                 image_size=640, device="cpu", min_area=1800):
        if backend not in ("yolo", "mog2"):
            raise ValueError(f"Unknown detector backend: {backend}")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        self.backend = backend
        self.confidence = confidence
        self.image_size = int(image_size)
        self.min_area = min_area
        self.device = resolve_device(device)
        self.names = {}
        self.model = None
        self.allowed_ids = None

        if backend == "yolo":
            self._load_yolo(model_path)
        else:
            self.subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=32, detectShadows=True)
            self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def _load_yolo(self, model_path):
        from ultralytics import YOLO

        p = Path(model_path)
        # Bare names such as "yolo11n.pt" may be auto-downloaded by Ultralytics; paths must exist.
        if (p.parent != Path(".") or p.is_absolute()) and not p.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {model_path}")
        self.model = YOLO(str(p))
        names = self.model.names
        self.names = dict(names) if isinstance(names, dict) else dict(enumerate(names))
        lowered = {i: str(n).lower() for i, n in self.names.items()}
        if VEHICLE_NAMES & set(lowered.values()):
            self.allowed_ids = {i for i, n in lowered.items() if n in VEHICLE_NAMES}
        if self.names != EXPECTED_CLASSES:
            print(f"  [Detector Notice] Model classes {self.names} differ from project classes {EXPECTED_CLASSES}")

    def detect(self, frame) -> List[Detection]:
        return self._detect_yolo(frame) if self.backend == "yolo" else self._detect_mog2(frame)

    def _detect_yolo(self, frame) -> List[Detection]:
        h_img, w_img = frame.shape[:2]
        result = self.model.predict(
            frame, conf=self.confidence, imgsz=self.image_size,
            device=self.device, verbose=False,
        )[0]
        detections = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), conf, cid in zip(xyxy, confs, clss):
            if self.allowed_ids is not None and cid not in self.allowed_ids:
                continue
            name = str(self.names.get(int(cid), cid))
            det = _xyxy_to_detection(x1, y1, x2, y2, name, conf, cid, w_img, h_img)
            if det is None or det.bbox[2] * det.bbox[3] < MIN_BOX_AREA_RATIO * w_img * h_img:
                continue
            detections.append(det)
        return detections

    def _detect_mog2(self, frame) -> List[Detection]:
        h_img, w_img = frame.shape[:2]
        mask = self.subtractor.apply(frame)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.dilate(mask, self.kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            det = _xyxy_to_detection(x, y, x + w, y + h, "moving_object", 0.5, -1, w_img, h_img)
            if det is not None:
                detections.append(det)
        return detections
