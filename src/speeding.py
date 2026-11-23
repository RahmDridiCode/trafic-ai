import csv
from pathlib import Path

KMH_PER_MS = 3.6

CSV_FIELDS = [
    "track_id", "vehicle_class", "measured_speed", "speed_limit",
    "excess_speed", "frame_number", "timestamp",
]


class SpeedingDetector:
    """Flags vehicles that exceed a speed limit for several consecutive measurements.

    Speeds come from SpeedEstimator (m/s when calibrated, px/s otherwise). Only a REAL calibration
    gives physical km/h: with `kmh_valid=False` (uncalibrated or demo calibration) nothing is ever
    flagged, a px/s value is never read as km/h.

    - a violation needs `min_consecutive` consecutive measurements above `limit_kmh`;
    - a missing (None) or non-exceeding measurement resets the streak;
    - one violation per episode: the episode ends only when the speed is measured at or below the
      limit, so the same vehicle is not recorded again while it keeps speeding.
    """

    def __init__(self, limit_kmh=50.0, min_consecutive=5, kmh_valid=True):
        if limit_kmh <= 0:
            raise ValueError("limit_kmh must be > 0")
        if min_consecutive < 1:
            raise ValueError("min_consecutive must be >= 1")
        self.limit_kmh = float(limit_kmh)
        self.min_consecutive = int(min_consecutive)
        self.kmh_valid = bool(kmh_valid)
        self._streak = {}
        self._episode = set()
        self.violations = []

    @property
    def speeding_ids(self):
        return set(self._episode)

    def update(self, track_id, class_name, speed_ms, frame_no, fps):
        """speed_ms: estimator speed in m/s (None if unavailable). Returns the new violation or None."""
        if not self.kmh_valid or speed_ms is None:
            self._streak[track_id] = 0
            return None
        kmh = speed_ms * KMH_PER_MS
        if kmh <= self.limit_kmh:
            self._streak[track_id] = 0
            self._episode.discard(track_id)
            return None
        self._streak[track_id] = self._streak.get(track_id, 0) + 1
        if track_id in self._episode or self._streak[track_id] < self.min_consecutive:
            return None
        self._episode.add(track_id)
        violation = {
            "track_id": track_id,
            "vehicle_class": class_name,
            "measured_speed": round(kmh, 2),
            "speed_limit": self.limit_kmh,
            "excess_speed": round(kmh - self.limit_kmh, 2),
            "frame_number": frame_no,
            "timestamp": round(frame_no / fps, 3) if fps else None,
        }
        self.violations.append(violation)
        return violation

    def forget(self, active_ids):
        """Drop state of tracks that are gone (a lost track ends its episode)."""
        active = set(active_ids)
        for tid in list(self._streak):
            if tid not in active:
                self._streak.pop(tid)
                self._episode.discard(tid)

    def write_csv(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(self.violations)
