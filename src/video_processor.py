"""
Headless Video Processor Pipeline Orchestrator.
Processes traffic video streams in strict CLI mode:
  - Decodes frames without GUI display server dependencies
  - Runs detection, multi-object tracking, and kinematic estimation
  - Streams annotated video to MP4 container and observations to CSV/JSON
"""

from collections import Counter
import csv
import json
import os
from pathlib import Path
from typing import Dict, Optional

# Enforce strict offscreen/headless execution for OpenCV/Qt
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import cv2
from .speed_alert import SpeedAlertManager
from .speeding import SpeedingDetector
from .visualizer import KMH_PER_MS, draw_hud, draw_tracks
from .Feux.red_light_analyzer import CSV_FIELDS as RED_LIGHT_CSV_FIELDS


class VideoProcessor:
    """
    Coordinates batch frame execution across detector, tracker, speed estimator,
    and statistical analyzer subsystems in a headless environment.
    """

    def __init__(self, detector, tracker, estimator, analyzer, calibration_mode=None,
                 speed_limit_kmh=50.0, speeding_min_measures=5,
                 speeding_seconds=0.6, speeding_track_seconds=1.0,
                 speeding_dedup_seconds=3.0, speeding_movement_seconds=0.6,
                 speeding_min_confidence=0.4, speeding_margin=0.10,
                 speeding_max_plausible_kmh=150.0, speeding_edge_margin=3,
                 speeding_max_width_jump=0.25, traffic_light_detector=None,
                 red_light_analyzer=None, traffic_light_config=None):
        # calibration_mode: "real", "demo" or "disabled" (derived from the transformer if omitted).
        if calibration_mode is None:
            calibration_mode = "real" if estimator.transformer is not None else "disabled"
        self.calibration_mode = calibration_mode
        self.detector = detector
        self.tracker = tracker
        self.estimator = estimator
        self.analyzer = analyzer
        self.speed_limit_kmh = speed_limit_kmh
        self.speeding_min_measures = speeding_min_measures
        self.speeding_seconds = speeding_seconds
        self.speeding_track_seconds = speeding_track_seconds
        self.speeding_dedup_seconds = speeding_dedup_seconds
        self.speeding_movement_seconds = speeding_movement_seconds
        self.speeding_min_confidence = speeding_min_confidence
        self.speeding_margin = speeding_margin
        self.speeding_max_plausible_kmh = speeding_max_plausible_kmh
        self.speeding_edge_margin = speeding_edge_margin
        self.speeding_max_width_jump = speeding_max_width_jump
        self.traffic_light_detector = traffic_light_detector
        self.red_light_analyzer = red_light_analyzer
        self.traffic_light_config = traffic_light_config

    def _calibration_label(self):
        if self.calibration_mode == "real":
            return "Speed: Calibrated (Road Homography -> km/h)"
        if self.calibration_mode == "demo":
            return "Speed: DEMO calibration (km/h NOT physical)"
        return "Speed: Pixel displacement (px/s) - NOT calibrated"

    def run(
        self,
        input_path: str,
        video_output: str,
        csv_output: str,
        json_output: str,
        max_frames: Optional[int] = None,
        frame_skip: int = 0,
        violations_output: Optional[str] = None,
    ) -> Dict:
        """
        Executes the video processing pipeline in strict headless CLI mode.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video input: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if violations_output is None:
            violations_output = str(Path(csv_output).with_name("speeding_violations.csv"))
        # Only a real calibration yields physical km/h; px/s and demo values are never flagged.
        speeding = SpeedingDetector(
            self.speed_limit_kmh, self.speeding_min_measures,
            kmh_valid=(self.calibration_mode == "real" and self.estimator.transformer is not None))

        alerts = SpeedAlertManager(
            limit_kmh=self.speed_limit_kmh,
            kmh_valid=speeding.kmh_valid,
            margin=self.speeding_margin,
            frame_step=frame_skip + 1,
            fps=fps,
            min_speeding_seconds=self.speeding_seconds,
            min_track_seconds=self.speeding_track_seconds,
            min_conf=self.speeding_min_confidence,
            max_plausible_kmh=self.speeding_max_plausible_kmh,
            edge_margin=self.speeding_edge_margin,
            max_width_variation=self.speeding_max_width_jump,
            dedup_seconds=self.speeding_dedup_seconds,
            movement_window_seconds=self.speeding_movement_seconds,
        )
        alerts_csv = Path(csv_output).with_name("speed_alerts.csv")
        alert_diagnostic_csv = Path(csv_output).with_name("alert_diagnostic.csv")
        alerts_dir = Path(csv_output).with_name("speed_alerts")
        red_light_csv = Path(csv_output).with_name("red_light_violations.csv")
        red_light_images_dir = Path(csv_output).with_name("red_light_violations")
        light_state = "UNKNOWN"
        light_detection_counts = Counter()

        # Ensure target directories exist
        Path(video_output).parent.mkdir(parents=True, exist_ok=True)
        Path(csv_output).parent.mkdir(parents=True, exist_ok=True)
        Path(json_output).parent.mkdir(parents=True, exist_ok=True)

        writer = cv2.VideoWriter(
            video_output,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps / (frame_skip + 1),
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            writer.release()
            raise RuntimeError(f"Unable to open output video for writing: {video_output}")

        with open(csv_output, "w", newline="", encoding="utf-8") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                "frame", "vehicle_id", "class", "confidence", "x", "y", "width", "height",
                "speed", "speed_unit", "speed_kmh", "speed_status",
                "traffic_light_state", "red_light_violation",
            ])

            frame_no = 0
            processed = 0

            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    # Frame decimation for performance acceleration if requested
                    if frame_skip and frame_no % (frame_skip + 1) != 0:
                        frame_no += 1
                        continue

                    light_detections = []
                    light_state = "UNKNOWN"
                    if self.traffic_light_detector is not None:
                        light_detections, light_state = self.traffic_light_detector.detect(frame)
                        light_detection_counts.update(
                            detection.state for detection in light_detections
                        )

                    # 1. Semantic / Motion Detection
                    detections = self.detector.detect(frame)

                    # 2. Multi-Object Tracking Association
                    tracks = self.tracker.update(detections)

                    # 3. Kinematic Displacement & Speed Estimation
                    speeds = {}
                    display_speeds = {}
                    speed_statuses = {}
                    confirmed_fresh_speeds = {}
                    for t in tracks:
                        if t.missed == 0:
                            x, y, w, h = t.bbox
                            # Bottom-centre (wheels on the road plane) is the stable ground reference.
                            speeds[t.track_id] = self.estimator.update(
                                t.track_id,
                                (x + w / 2.0, y + h),
                                frame_no,
                                ref_size=w,
                                box_size=(w, h),
                                box=t.bbox,
                                img_size=(width, height),
                            )
                            status = self.estimator.speed_status(t.track_id)
                            speed_statuses[t.track_id] = status
                            if status == "provisional":
                                display_speeds[t.track_id] = self.estimator.provisional_speed(t.track_id)
                            elif speeds[t.track_id] is not None:
                                display_speeds[t.track_id] = speeds[t.track_id]
                            if self.estimator.is_fresh(t.track_id):
                                confirmed_fresh_speeds[t.track_id] = speeds[t.track_id]

                    for t in tracks:
                        if t.missed == 0:
                            fresh = self.estimator.is_fresh(t.track_id)
                            measured_speed = speeds.get(t.track_id) if fresh else None
                            speeding.update(t.track_id, t.class_name, measured_speed, frame_no, fps)
                    speeding.forget(t.track_id for t in tracks if t.missed == 0)

                    # 4. Statistical Aggregation
                    self.analyzer.update(tracks, confirmed_fresh_speeds)
                    active_tracks = [t for t in tracks if t.missed == 0]
                    counts = Counter(t.class_name for t in active_tracks)
                    new_alerts = []
                    for t in active_tracks:
                        alert = alerts.update(
                            t.track_id,
                            speeds.get(t.track_id),
                            frame_no,
                            t.bbox,
                            (width, height),
                            t.confidence,
                            fresh=self.estimator.is_fresh(t.track_id),
                            class_name=t.class_name,
                        )
                        if alert:
                            new_alerts.append(alert)

                    red_light_violations = []
                    if self.red_light_analyzer is not None:
                        fresh_ids = {
                            track_id for track_id in confirmed_fresh_speeds
                        }
                        red_light_violations = self.red_light_analyzer.update(
                            active_tracks,
                            light_state,
                            frame_no,
                            fps,
                            speeds=confirmed_fresh_speeds,
                            fresh_ids=fresh_ids,
                        )
                    red_light_violation_ids = {
                        violation["track_id"] for violation in red_light_violations
                    }

                    # 5. Graphical Annotation Rendering
                    calibrated = (
                        self.calibration_mode == "real"
                        and self.estimator.transformer is not None
                    )
                    demo_calibration = (
                        self.calibration_mode == "demo"
                        and self.estimator.transformer is not None
                    )
                    annotated = draw_tracks(
                        frame,
                        tracks,
                        display_speeds,
                        calibrated,
                        alerts.alerted_ids,
                        demo_calibration=demo_calibration,
                        speed_statuses=speed_statuses,
                        red_light_ids=(
                            self.red_light_analyzer.violating_ids
                            if self.red_light_analyzer is not None else None
                        ),
                    )
                    annotated = draw_hud(
                        annotated, frame_no, len(active_tracks), counts, calibrated,
                        self._calibration_label(),
                    )
                    if self.traffic_light_detector is not None:
                        self.traffic_light_detector.draw(
                            annotated, light_detections, light_state
                        )
                    if self.red_light_analyzer is not None:
                        self.red_light_analyzer.draw(annotated, active_tracks)
                    for alert in new_alerts:
                        alerts_dir.mkdir(parents=True, exist_ok=True)
                        image_path = alerts_dir / f"alert_id{alert['track_id']}_frame{alert['frame']}.jpg"
                        if not cv2.imwrite(str(image_path), annotated):
                            raise RuntimeError(f"Unable to write speeding alert image: {image_path}")
                    for violation in red_light_violations:
                        red_light_images_dir.mkdir(parents=True, exist_ok=True)
                        image_path = (
                            red_light_images_dir
                            / f"red_light_id{violation['track_id']}_frame{violation['frame']}.jpg"
                        )
                        if not cv2.imwrite(str(image_path), annotated):
                            raise RuntimeError(f"Unable to write red-light violation image: {image_path}")
                    writer.write(annotated)

                    # 6. Tabular Observation Persistence
                    if self.calibration_mode == "demo" and self.estimator.transformer is not None:
                        unit = "km/h (DEMO; unverified calibration)"
                    elif calibrated:
                        unit = "m/s"
                    else:
                        unit = "pixels/s"
                    for t in active_tracks:
                        x, y, w, h = t.bbox
                        speed_val = display_speeds.get(t.track_id)
                        status = speed_statuses.get(t.track_id, "na")
                        if speed_val is None:
                            speed_str, speed_kmh_str = "N/A", "N/A"
                        else:
                            if self.calibration_mode == "demo" and self.estimator.transformer is not None:
                                speed_kmh = speed_val * KMH_PER_MS
                                speed_str = f"{speed_kmh:.2f}"
                                speed_kmh_str = "N/A"
                            elif calibrated:
                                speed_str = f"{speed_val:.4f}"
                                speed_kmh_str = f"{speed_val * KMH_PER_MS:.2f}"
                            else:
                                speed_str = f"{speed_val:.4f}"
                                speed_kmh_str = "N/A"
                        csv_writer.writerow([
                            frame_no, t.track_id, t.class_name, f"{t.confidence:.4f}",
                            t.centroid[0], t.centroid[1], w, h,
                            speed_str, unit, speed_kmh_str, status, light_state,
                            t.track_id in red_light_violation_ids,
                        ])

                    processed += 1
                    frame_no += 1

                    # Live headless progress feedback to stdout
                    if processed % 50 == 0 or (max_frames is not None and processed >= max_frames):
                        target_str = f"/{max_frames}" if max_frames else f"/{total_frames}"
                        print(f"  [CLI Engine] Processed {processed}{target_str} frames | Active vehicles: {len(active_tracks)}")

                    if max_frames is not None and processed >= max_frames:
                        break
            finally:
                cap.release()
                writer.release()

        speeding.write_csv(violations_output)
        alerts.write_csv(alerts_csv)
        alerts.write_diagnostic_csv(alert_diagnostic_csv)
        if self.red_light_analyzer is not None:
            self.red_light_analyzer.write_csv(red_light_csv)
        else:
            with red_light_csv.open("w", newline="", encoding="utf-8") as file:
                csv.DictWriter(file, fieldnames=RED_LIGHT_CSV_FIELDS).writeheader()

        # Compile and export comprehensive JSON analytics summary
        calibrated = (
            self.calibration_mode == "real"
            and self.estimator.transformer is not None
        )
        speed_unit = (
            "km/h (DEMO; unverified calibration)"
            if self.calibration_mode == "demo" and self.estimator.transformer is not None
            else "m/s" if calibrated else "pixels/s"
        )
        speed_scale = (
            KMH_PER_MS
            if self.calibration_mode == "demo" and self.estimator.transformer is not None
            else 1.0
        )
        summary = self.analyzer.summary(
            calibrated=calibrated,
            speed_unit=speed_unit,
            speed_scale=speed_scale,
        )
        summary["calibration_mode"] = self.calibration_mode
        summary["speed_estimate_is_physical"] = calibrated
        if self.calibration_mode == "demo" and self.estimator.transformer is not None:
            summary["speed_estimate_note"] = (
                "Displayed as km/h after multiplying the demo m/s estimate by 3.6; "
                "not a physically calibrated speed."
            )
        summary["frame_skip"] = frame_skip
        summary["speeding"] = {
            "enabled": speeding.kmh_valid,
            "speed_limit_kmh": self.speed_limit_kmh,
            "min_consecutive_measures": self.speeding_min_measures,
            "violation_count": len(speeding.violations),
            "violations": speeding.violations,
            "output_csv": str(Path(violations_output).resolve()),
        }
        summary["speed_alerts"] = {
            "enabled": alerts.kmh_valid,
            "alert_count": len(alerts.alerts),
            "alerts": alerts.alerts,
            "output_csv": str(alerts_csv.resolve()),
            "images_dir": str(alerts_dir.resolve()),
            "diagnostic_csv": str(alert_diagnostic_csv.resolve()),
            "diagnostic_reason_counts": dict(Counter(
                reason
                for row in alerts.diagnostics
                for reason in row["reasons"].split(";")
            )),
            "conditions": {
                "speeding_seconds": self.speeding_seconds,
                "track_seconds": self.speeding_track_seconds,
                "movement_seconds": self.speeding_movement_seconds,
                "minimum_mean_confidence": self.speeding_min_confidence,
                "limit_margin": self.speeding_margin,
                "max_plausible_kmh": self.speeding_max_plausible_kmh,
                "edge_margin_pixels": self.speeding_edge_margin,
                "max_consecutive_width_jump": self.speeding_max_width_jump,
                "dedup_seconds": self.speeding_dedup_seconds,
            },
        }
        summary["traffic_lights"] = {
            "enabled": self.traffic_light_detector is not None,
            "state": light_state,
            "model": (
                self.traffic_light_detector.model_path
                if self.traffic_light_detector is not None else None
            ),
            "model_classes": (
                self.traffic_light_detector.class_names
                if self.traffic_light_detector is not None else None
            ),
            "roi_config": (
                self.traffic_light_config["path"]
                if self.traffic_light_config is not None else None
            ),
            "detections_by_state": dict(light_detection_counts),
            "red_light_violation_count": (
                len(self.red_light_analyzer.violations)
                if self.red_light_analyzer is not None else 0
            ),
            "red_light_violations": (
                self.red_light_analyzer.violations
                if self.red_light_analyzer is not None else []
            ),
            "violations_csv": str(red_light_csv.resolve()),
            "images_dir": str(red_light_images_dir.resolve()),
        }
        summary["input_resolution"] = f"{width}x{height}"
        summary["video_fps"] = round(fps, 3)
        summary["output_video"] = str(Path(video_output).resolve())
        summary["output_csv"] = str(Path(csv_output).resolve())

        with open(json_output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary
