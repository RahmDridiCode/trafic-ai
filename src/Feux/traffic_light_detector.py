"""Détection de l'état du feu par ratio de pixels rouges dans sa ROI."""

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from ..detector import resolve_device


STATE_ALIASES = {
    "red": "RED",
    "red_light": "RED",
    "redlight": "RED",
    "traffic_light_red": "RED",
    "green": "GREEN",
    "green_light": "GREEN",
    "greenlight": "GREEN",
    "traffic_light_green": "GREEN",
    "yellow": "YELLOW",
    "amber": "YELLOW",
    "yellow_light": "YELLOW",
    "traffic_light_yellow": "YELLOW",
}
STATE_COLORS = {
    "RED": (0, 0, 255),
    "GREEN": (0, 255, 0),
    "YELLOW": (0, 220, 255),
    "UNKNOWN": (180, 180, 180),
}

HSV_RED_LOW_1 = (0, 70, 70)
HSV_RED_HIGH_1 = (12, 255, 255)
HSV_RED_LOW_2 = (168, 70, 70)
HSV_RED_HIGH_2 = (180, 255, 255)


@dataclass(frozen=True)
class TrafficLightDetection:
    class_name: str
    state: str
    confidence: float
    bbox: tuple[int, int, int, int]


class TrafficLightDetector:
    """Estime RED/GREEN par couleur; YOLO ne sert qu'aux cadres facultatifs."""

    def __init__(
        self,
        model_path,
        light_polygon,
        confidence=0.5,
        image_size=640,
        device="cpu",
        min_state_detections=3,
        model=None,
        red_ratio_threshold=0.01,
        history_size=5,
        min_red_detections=3,
        min_green_detections=3,
        show_yolo_boxes=False,  # Désactive les cadres YOLO par défaut.
    ):
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Traffic-light confidence must be within [0, 1].")
        if image_size <= 0 or min_state_detections < 1:
            raise ValueError("Traffic-light image size and confirmation count must be positive.")
        if not 0.0 <= red_ratio_threshold <= 1.0:
            raise ValueError("red_ratio_threshold must be within [0, 1].")
        if history_size < 1:
            raise ValueError("history_size must be positive.")
        if not 1 <= min_red_detections <= history_size:
            raise ValueError("min_red_detections must be between 1 and history_size.")
        if not 1 <= min_green_detections <= history_size:
            raise ValueError("min_green_detections must be between 1 and history_size.")

        self.model_path = str(model_path)
        self.confidence = float(confidence)
        self.image_size = int(image_size)
        self.device = resolve_device(device)
        self.light_polygon = np.asarray(light_polygon, dtype=np.int32)
        self.model = model
        if self.model is None:
            from ultralytics import YOLO

            self.model = YOLO(self.model_path)

        model_names = self.model.names
        if isinstance(model_names, dict):
            self.names = {int(class_id): str(name) for class_id, name in model_names.items()}
        else:
            self.names = dict(enumerate(str(name) for name in model_names))
        self.state_class_ids = {}
        for class_id, name in self.names.items():
            normalized = name.strip().lower().replace(" ", "_").replace("-", "_")
            state = STATE_ALIASES.get(normalized)
            if state is not None:
                self.state_class_ids[class_id] = state

        self.red_ratio_threshold = float(red_ratio_threshold)
        self.history_size = int(history_size)
        self.min_red_detections = int(min_red_detections)
        self.min_green_detections = int(min_green_detections)
        if min_state_detections != 3:
            self.min_red_detections = int(min_state_detections)
            self.min_green_detections = int(min_state_detections)
        if not 1 <= self.min_red_detections <= self.history_size:
            raise ValueError("min_red_detections must be between 1 and history_size.")
        if not 1 <= self.min_green_detections <= self.history_size:
            raise ValueError("min_green_detections must be between 1 and history_size.")
        self.red_history = deque(maxlen=self.history_size)
        self.green_history = deque(maxlen=self.history_size)
        self.show_yolo_boxes = bool(show_yolo_boxes)  # Mémorise l'option d'affichage YOLO.
        self.red_ratio = 0.0
        self.state = "UNKNOWN"

    def detect(self, frame):
        """Calcule le ratio rouge masqué par la ROI, puis applique l'hystérésis."""
        height, width = frame.shape[:2]
        if self.light_polygon.size == 0 or (
            np.any(self.light_polygon[:, 0] < 0)
            or np.any(self.light_polygon[:, 0] >= width)
            or np.any(self.light_polygon[:, 1] < 0)
            or np.any(self.light_polygon[:, 1] >= height)
        ):
            raise ValueError(
                f"Traffic-light polygon does not fit the {width}x{height} frame."
            )

        x0 = int(np.min(self.light_polygon[:, 0]))
        y0 = int(np.min(self.light_polygon[:, 1]))
        x1 = int(np.max(self.light_polygon[:, 0])) + 1
        y1 = int(np.max(self.light_polygon[:, 1])) + 1
        crop = frame[y0:y1, x0:x1]
        local_polygon = self.light_polygon - np.array([x0, y0], dtype=np.int32)
        roi_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.fillPoly(roi_mask, [local_polygon.reshape((-1, 1, 2))], 255)
        roi_pixels = cv2.countNonZero(roi_mask)

        if roi_pixels == 0:
            self.red_ratio = 0.0
        else:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            red_mask_1 = cv2.inRange(
                hsv,
                np.array(HSV_RED_LOW_1, dtype=np.uint8),
                np.array(HSV_RED_HIGH_1, dtype=np.uint8),
            )
            red_mask_2 = cv2.inRange(
                hsv,
                np.array(HSV_RED_LOW_2, dtype=np.uint8),
                np.array(HSV_RED_HIGH_2, dtype=np.uint8),
            )
            red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
            red_in_roi = cv2.bitwise_and(red_mask, roi_mask)
            self.red_ratio = cv2.countNonZero(red_in_roi) / roi_pixels

        frame_is_red = self.red_ratio >= self.red_ratio_threshold
        self.red_history.append(frame_is_red)
        self.green_history.append(not frame_is_red)

        if sum(self.red_history) >= self.min_red_detections:
            self.state = "RED"
        elif sum(self.green_history) >= self.min_green_detections:
            self.state = "GREEN"

        detections = (  # N'exécute YOLO que si ses cadres sont demandés.
            self._detect_boxes(crop, x0, y0, width, height)
            if self.show_yolo_boxes else []  # Sinon, aucun appel YOLO par frame.
        )
        return detections, self.state

    def _detect_boxes(self, crop, x0, y0, width, height):
        """Retourne les cadres YOLO reconnus, sans les utiliser pour l'état."""
        if not self.state_class_ids:
            return []

        results = self.model.predict(
            crop,
            classes=list(self.state_class_ids),
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        detections = []
        if not results or results[0].boxes is None:
            return detections

        polygon = self.light_polygon.reshape((-1, 1, 2))
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            state = self.state_class_ids.get(class_id)
            if state is None:
                continue
            bx0, by0, bx1, by1 = (float(value) for value in box.xyxy[0])
            center = (
                int(round((bx0 + bx1) / 2)) + x0,
                int(round((by0 + by1) / 2)) + y0,
            )
            if cv2.pointPolygonTest(
                polygon, (float(center[0]), float(center[1])), False
            ) < 0:
                continue
            detections.append(
                TrafficLightDetection(
                    class_name=self.names[class_id],
                    state=state,
                    confidence=float(box.conf[0]),
                    bbox=(
                        max(0, int(round(bx0)) + x0),
                        max(0, int(round(by0)) + y0),
                        min(width, int(round(bx1)) + x0),
                        min(height, int(round(by1)) + y0),
                    ),
                )
            )
        return detections

    def draw(self, frame, detections, state):
        """Dessine la ROI, les cadres YOLO et le diagnostic en haut à droite."""
        polygon = self.light_polygon.reshape((-1, 1, 2))
        cv2.polylines(frame, [polygon], True, (255, 255, 0), 2)
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            color = STATE_COLORS[detection.state]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{detection.state} {detection.confidence:.2f}",
                (x1, max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        x = max(0, frame.shape[1] - 520)
        color = STATE_COLORS.get(state, STATE_COLORS["UNKNOWN"])
        cv2.putText(
            frame,
            f"TRAFFIC LIGHT: {state}",
            (x, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"red ratio: {self.red_ratio:.3f}",
            (x, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    @property
    def class_names(self):
        return dict(self.names)
