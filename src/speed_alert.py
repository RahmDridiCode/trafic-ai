import csv
import math
from collections import deque
from pathlib import Path

KMH_PER_MS = 3.6
CSV_FIELDS = [
    "track_id", "vehicle_class", "frame", "speed_kmh", "speed_limit", "confidence",
    "x", "y", "width", "height",
]
DIAGNOSTIC_FIELDS = [
    "frame", "track_id", "vehicle_class", "fresh", "calibration_valid", "speed_kmh",
    "confidence", "x", "y", "width", "height", "reasons",
]


class _TrackInfo:
    def __init__(self, position_window):
        self.first_frame = None
        self.last_frame = None
        self.streak_start_frame = None
        self.conf_sum = 0.0
        self.frames_seen = 0
        self.last_width = None
        self.positions = deque(maxlen=position_window)


class SpeedAlertManager:
    """Produces conservative, calibrated speeding alerts from fresh speed estimates."""

    def __init__(
        self,
        limit_kmh=50.0,
        kmh_valid=True,
        margin=0.10,
        min_speeding_seconds=0.6,
        min_track_seconds=1.0,
        min_conf=0.4,
        max_plausible_kmh=150.0,
        edge_margin=3,
        max_width_variation=0.25,
        frame_step=1,
        fps=30.0,
        dedup_seconds=3.0,
        movement_window_seconds=0.6,
    ):
        if limit_kmh <= 0:
            raise ValueError("limit_kmh must be > 0")
        if fps <= 0:
            raise ValueError("fps must be > 0")
        if min_speeding_seconds <= 0 or min_track_seconds <= 0:
            raise ValueError("Alert durations must be > 0")
        if dedup_seconds < 0 or movement_window_seconds <= 0:
            raise ValueError("Deduplication duration must be >= 0 and movement window > 0")
        self.limit_kmh = float(limit_kmh)
        self.kmh_valid = bool(kmh_valid)
        self.threshold_kmh = self.limit_kmh * (1.0 + margin)
        self.min_speeding_seconds = float(min_speeding_seconds)
        self.min_track_seconds = float(min_track_seconds)
        self.min_conf = float(min_conf)
        self.max_plausible_kmh = float(max_plausible_kmh)
        self.edge_margin = max(0, int(edge_margin))
        self.max_width_variation = float(max_width_variation)
        self.frame_step = max(1, int(frame_step))
        self.fps = float(fps)
        self.dedup_frames = self.fps * dedup_seconds
        self.movement_frames = self.fps * movement_window_seconds
        position_window = math.ceil(
            self.fps * (movement_window_seconds + 1.0) / self.frame_step
        ) + 2
        self._tracks = {}
        self._alerted = set()
        self.alerts = []
        self.diagnostics = []
        self._position_window = position_window

    @property
    def alerted_ids(self):
        return set(self._alerted)

    def _touches_edge(self, box, img_size):
        x, y, width, height = box
        img_w, img_h = img_size
        margin = self.edge_margin
        return (
            x <= margin
            or x + width >= img_w - margin
            or y + height >= img_h - margin
        )

    def _has_moved_enough(self, info, frame_no, box):
        x, y, width, height = box
        current = (x + width / 2.0, y + height)
        target_frame = frame_no - self.movement_frames
        previous = next(
            (point for point in reversed(info.positions) if point[0] <= target_frame),
            None,
        )
        if previous is None or frame_no - previous[0] > self.movement_frames + self.frame_step:
            return False
        distance = math.hypot(current[0] - previous[1], current[1] - previous[2])
        return distance > width

    def _is_recent_duplicate(self, class_name, frame_no, box):
        x, y, width, height = box
        center = (x + width / 2.0, y + height / 2.0)
        for alert in reversed(self.alerts):
            elapsed_frames = frame_no - alert["frame"]
            if elapsed_frames > self.dedup_frames:
                break
            if elapsed_frames < 0 or alert["vehicle_class"] != class_name:
                continue
            old_center = (
                alert["x"] + alert["width"] / 2.0,
                alert["y"] + alert["height"] / 2.0,
            )
            if math.hypot(center[0] - old_center[0], center[1] - old_center[1]) < alert["width"]:
                return True
        return False

    def _record_diagnostic(self, track_id, class_name, frame_no, speed_ms, confidence,
                           box, fresh, reasons):
        x, y, width, height = box
        speed_kmh = (
            round(speed_ms * KMH_PER_MS, 2)
            if self.kmh_valid and speed_ms is not None
            else ""
        )
        self.diagnostics.append({
            "frame": frame_no,
            "track_id": track_id,
            "vehicle_class": class_name,
            "fresh": bool(fresh),
            "calibration_valid": self.kmh_valid,
            "speed_kmh": speed_kmh,
            "confidence": round(float(confidence), 4),
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "reasons": ";".join(reasons) if reasons else "alert_emitted",
        })

    def update(self, track_id, speed_ms, frame_no, box, img_size, confidence, fresh=True,
               class_name="unknown"):
        """Return an alert only for a fresh, sustained, physically calibrated measurement."""
        info = self._tracks.get(track_id)
        if info is None:
            info = self._tracks[track_id] = _TrackInfo(self._position_window)

        previous_frame = info.last_frame
        if info.first_frame is None:
            info.first_frame = frame_no
        frame_delta = None if previous_frame is None else frame_no - previous_frame
        if frame_delta is not None and frame_delta > self.frame_step:
            info.streak_start_frame = None
            info.conf_sum = 0.0
            info.frames_seen = 0
            info.first_frame = frame_no
        info.last_frame = frame_no

        x, y, width, height = box
        info.positions.append((frame_no, x + width / 2.0, y + height))
        width_jump = (
            frame_delta == self.frame_step
            and info.last_width is not None
            and info.last_width > 0
            and abs(width - info.last_width) / info.last_width > self.max_width_variation
        )
        info.last_width = width

        touches_edge = self._touches_edge(box, img_size)
        reasons = []
        if not self.kmh_valid:
            reasons.append("calibration_invalid")
        if not fresh:
            reasons.append("not_fresh")
        if speed_ms is None:
            reasons.append("speed_unavailable")
        if touches_edge:
            reasons.append("edge_box")
        if width_jump:
            reasons.append("width_jump")

        if (
            not self.kmh_valid
            or not fresh
            or speed_ms is None
            or touches_edge
            or width_jump
        ):
            info.streak_start_frame = None
            info.conf_sum = 0.0
            info.frames_seen = 0
            self._record_diagnostic(
                track_id, class_name, frame_no, speed_ms, confidence, box, fresh, reasons
            )
            return None

        kmh = speed_ms * KMH_PER_MS
        if kmh <= self.threshold_kmh:
            reasons.append("speed_below_threshold")
            info.streak_start_frame = None
            info.conf_sum = 0.0
            info.frames_seen = 0
            self._record_diagnostic(
                track_id, class_name, frame_no, speed_ms, confidence, box, fresh, reasons
            )
            return None

        if kmh >= self.max_plausible_kmh:
            reasons.append("speed_implausible")
            info.streak_start_frame = None
            info.conf_sum = 0.0
            info.frames_seen = 0
            self._record_diagnostic(
                track_id, class_name, frame_no, speed_ms, confidence, box, fresh, reasons
            )
            return None

        if info.streak_start_frame is None:
            info.streak_start_frame = frame_no
            info.conf_sum = 0.0
            info.frames_seen = 0
        info.frames_seen += 1
        info.conf_sum += confidence

        speeding_duration = (frame_no - info.streak_start_frame) / self.fps
        tracked_duration = (frame_no - info.first_frame) / self.fps
        mean_conf = info.conf_sum / info.frames_seen
        if track_id in self._alerted:
            reasons.append("already_alerted")
        if speeding_duration < self.min_speeding_seconds:
            reasons.append("insufficient_speed_duration")
        if tracked_duration < self.min_track_seconds:
            reasons.append("insufficient_track_duration")
        if mean_conf < self.min_conf:
            reasons.append("low_mean_confidence")
        if not self._has_moved_enough(info, frame_no, box):
            reasons.append("insufficient_movement")
        if not reasons and self._is_recent_duplicate(class_name, frame_no, box):
            reasons.append("recent_duplicate")
            self._alerted.add(track_id)

        if reasons:
            self._record_diagnostic(
                track_id, class_name, frame_no, speed_ms, confidence, box, fresh, reasons
            )
            return None

        alert = {
            "track_id": track_id,
            "vehicle_class": class_name,
            "frame": frame_no,
            "speed_kmh": round(kmh, 2),
            "speed_limit": self.limit_kmh,
            "confidence": round(mean_conf, 4),
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        }
        self._alerted.add(track_id)
        self.alerts.append(alert)
        self._record_diagnostic(
            track_id, class_name, frame_no, speed_ms, confidence, box, fresh, []
        )
        return alert

    def write_csv(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(self.alerts)

    def write_diagnostic_csv(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=DIAGNOSTIC_FIELDS)
            writer.writeheader()
            writer.writerows(self.diagnostics)
