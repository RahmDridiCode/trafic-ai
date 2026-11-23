"""Analyse les franchissements de la ligne avec les IDs du tracker principal."""

import csv
from collections import deque
from pathlib import Path

import cv2
import numpy as np


CSV_FIELDS = [
    "track_id", "vehicle_class", "frame", "timestamp", "light_state",
    "confidence", "speed_m_s", "x", "y", "width", "height",
]


class RedLightAnalyzer:
    """Enregistre une seule infraction par véhicule franchissant vers le haut au rouge."""

    def __init__(
        self,
        vehicle_polygon,
        stop_line,
        frame_step=1,
        violation_threshold=20,
        max_past_line=350,
        min_upward_displacement=10,
        position_history_size=5,
    ):
        self.vehicle_polygon = np.asarray(vehicle_polygon, dtype=np.int32)
        self.stop_line = tuple(int(value) for value in stop_line)
        if len(self.stop_line) != 4:
            raise ValueError("stop_line must contain [x1, y1, x2, y2].")
        if position_history_size < 3:
            raise ValueError("position_history_size must be at least 3.")
        self.frame_step = max(1, int(frame_step))
        self.violation_threshold = float(violation_threshold)
        self.max_past_line = float(max_past_line)
        self.min_upward_displacement = float(min_upward_displacement)
        self.position_history_size = int(position_history_size)
        self._position_history = {}
        self._last_frames = {}
        self.crossing_red_state = {}
        self._recorded_track_ids = set()
        self.violating_ids = set()
        self.violations = []

    def _stop_line_y_at_x(self, x):
        x1, y1, x2, y2 = self.stop_line
        if x1 == x2:
            return (y1 + y2) / 2.0
        ratio = max(0.0, min(1.0, (x - x1) / (x2 - x1)))
        return y1 + ratio * (y2 - y1)

    def update(self, tracks, light_state, frame_no, fps, speeds=None, fresh_ids=None):
        """Vérifie le mouvement ascendant et n'enregistre qu'un état RED confirmé."""
        speeds = speeds or {}
        fresh_ids = fresh_ids or set()
        new_violations = []

        for track in tracks:
            if track.missed != 0:
                continue

            track_id = track.track_id
            x, y, width, height = track.bbox
            point_x = x + width / 2.0
            point_y = y + height
            past = self._stop_line_y_at_x(point_x) - point_y

            history = self._position_history.setdefault(
                track_id, deque(maxlen=self.position_history_size)
            )
            previous = history[-1] if history else None
            last_frame = self._last_frames.get(track_id)
            if last_frame is not None and frame_no - last_frame != self.frame_step:
                history.clear()
                self.crossing_red_state.pop(track_id, None)
                previous = None

            current_is_red = light_state == "RED"
            history.append((point_x, point_y, past, current_is_red))
            self._last_frames[track_id] = frame_no

            dx = history[-1][0] - history[0][0]
            dy = history[-1][1] - history[0][1]
            moving_up = (
                len(history) > 2
                and dy <= -self.min_upward_displacement
                and abs(dy) > abs(dx)
            )

            if past <= self.violation_threshold:
                self.crossing_red_state.pop(track_id, None)
            elif track_id not in self.crossing_red_state:
                if previous is None:
                    self.crossing_red_state[track_id] = (
                        current_is_red and past <= self.max_past_line
                    )
                elif previous[2] <= self.violation_threshold:
                    self.crossing_red_state[track_id] = (
                        current_is_red
                        and point_y < previous[1]
                        and abs(dy) > abs(dx)
                    )

            inside_roi = cv2.pointPolygonTest(
                self.vehicle_polygon.reshape((-1, 1, 2)),
                (float(point_x), float(point_y)),
                False,
            ) >= 0
            is_violation = (
                light_state == "RED"
                and self.crossing_red_state.get(track_id, False)
                and inside_roi
                and self.violation_threshold <= past <= self.max_past_line
                and moving_up
                and track_id not in self._recorded_track_ids
            )
            if not is_violation:
                continue

            self._recorded_track_ids.add(track_id)
            self.violating_ids.add(track_id)
            speed = speeds.get(track_id) if track_id in fresh_ids else None
            violation = {
                "track_id": int(track_id),
                "vehicle_class": str(track.class_name),
                "frame": int(frame_no),
                "timestamp": round(frame_no / fps, 3) if fps else None,
                "light_state": "RED",
                "confidence": round(float(track.confidence), 4),
                "speed_m_s": "" if speed is None else round(float(speed), 4),
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height),
            }
            self.violations.append(violation)
            new_violations.append(violation)
        return new_violations

    def draw(self, frame, tracks):
        """Dessine la ligne d'arrêt ; le marquage des infractions est intégré
        à l'étiquette du véhicule (visualizer.draw_tracks, red_light_ids)."""
        x1, y1, x2, y2 = self.stop_line
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)

    def write_csv(self, path):
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(self.violations)
