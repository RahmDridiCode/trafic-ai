"""
Intelligent Traffic Surveillance & Vehicle Analytics (CSE3010)
Command-Line Interface (CLI) Entry Point & Orchestrator.
Designed for strict headless execution in automated evaluation environments.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Enforce offscreen headless execution for GUI libraries
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import cv2
from src.detector import VehicleDetector
from src.homography import HomographyTransformer
from src.speed_estimator import SpeedEstimator
from src.tracker import MultiObjectTracker
from src.traffic_analyzer import TrafficAnalyzer
from src.video_processor import VideoProcessor
from src.Feux.roi_config import load_roi_config
from src.Feux.red_light_analyzer import RedLightAnalyzer
from src.Feux.traffic_light_detector import TrafficLightDetector


def parse_args():
    p = argparse.ArgumentParser(
        description="Intelligent Traffic Surveillance & Vehicle Analytics — Headless CLI Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", default="dataset/test/videos/traffic.mp4", help="Path to input traffic video file")
    p.add_argument("--output-dir", default="outputs", help="Directory where generated MP4, CSV, and JSON outputs are saved")
    p.add_argument("--detector", choices=["yolo", "mog2"], default="yolo", help="Detection engine: deep semantic YOLO or course-baseline MOG2")
    p.add_argument("--model", default="models/yolov11n_best.pt", help="Path to YOLO weights (custom .pt or a pretrained name)")
    traffic_lights = p.add_mutually_exclusive_group()
    traffic_lights.add_argument(
        "--traffic-lights", dest="traffic_lights", action="store_true",
        help="Enable separate traffic-light detection and red-light crossing analysis",
    )
    traffic_lights.add_argument(
        "--no-traffic-lights", dest="traffic_lights", action="store_false",
        help="Disable all traffic-light processing",
    )
    p.set_defaults(traffic_lights=False)
    p.add_argument(
        "--traffic-light-model", default="models/Feux/best_n.pt",
        help="Separate YOLO weights used only for traffic-light state detection",
    )
    p.add_argument(
        "--traffic-light-roi", default="config/roi_config_polygon_2pts.json",
        help="ROI polygons and stop-line configuration for the input video",
    )
    p.add_argument("--traffic-light-conf", type=float, default=0.5, help="Traffic-light detection confidence")
    p.add_argument("--traffic-light-imgsz", type=int, default=640, help="Traffic-light model inference size")
    p.add_argument("--conf", type=float, default=0.35, help="Confidence detection threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference resolution for object detection")
    p.add_argument("--device", default="cpu", help="Compute device: 'auto', 'cpu', 'cuda', '0', etc.")
    p.add_argument("--max-distance", type=float, default=90.0, help="Maximum centroid distance gating for tracker association")
    p.add_argument("--max-missed", type=int, default=12, help="Number of consecutive frames a track persists through occlusion")
    p.add_argument("--iou-threshold", type=float, default=0.05, help="Minimum IoU overlap threshold for track association")
    p.add_argument("--min-area", type=int, default=1800, help="Minimum contour area filter for MOG2 background subtractor")
    p.add_argument("--calibration", help="Path to road geometry calibration JSON file (defaults to config/calibration.json)")
    p.add_argument("--uncalibrated", action="store_true", help="Force uncalibrated pixel kinematics mode (reporting in pixels/s)")
    p.add_argument("--allow-demo-calibration", action="store_true", help="Use an unverified calibration for demonstration only (km/h are NOT physical)")
    p.add_argument("--max-frames", type=int, help="Optional limit on number of frames to process (useful for fast verification)")
    p.add_argument("--speed-limit", type=float, default=50.0, help="Speed limit in km/h (speeding is only detected with a REAL calibration)")
    p.add_argument("--speeding-measures", type=int, default=5, help="Consecutive measurements above the limit needed to record a violation")
    p.add_argument("--speeding-seconds", type=float, default=0.6, help="Continuous fresh measured speeding duration required for an alert")
    p.add_argument("--speeding-track-seconds", type=float, default=1.0, help="Minimum track lifetime required for an alert")
    p.add_argument("--speeding-dedup-seconds", type=float, default=3.0, help="Nearby same-class alert deduplication interval")
    p.add_argument("--speeding-movement-seconds", type=float, default=0.6, help="Movement window used to confirm real displacement")
    p.add_argument("--speeding-min-confidence", type=float, default=0.4, help="Minimum mean detection confidence for an alert")
    p.add_argument("--speeding-margin", type=float, default=0.10, help="Fraction added to the speed limit before alerting")
    p.add_argument("--speeding-max-plausible-kmh", type=float, default=150.0, help="Maximum plausible calibrated speed")
    p.add_argument("--speeding-edge-margin", type=int, default=3, help="Pixels from image edges where alerts are suppressed")
    p.add_argument("--speeding-max-width-jump", type=float, default=0.25, help="Maximum relative box-width change between consecutive observations")
    p.add_argument("--skip", type=int, default=0, help="Process every (N+1)th frame to accelerate throughput on CPU")
    return p.parse_args()


def _parse_resolution(text):
    try:
        w, h = str(text).lower().split("x")
        resolution = int(w), int(h)
        return resolution if all(value > 0 for value in resolution) else None
    except Exception:
        return None


def load_calibration(path=None, width=0, height=0, force_uncalibrated=False, allow_demo=False):
    """
    Returns (transformer, mode, status) where mode is one of:
      "real"     - calibration file explicitly marked "calibration_type": "real" and matching the video
                   resolution: speeds are physical (m/s -> km/h)
      "demo"     - unverified/demo calibration, only used with --allow-demo-calibration; NOT physical
      "disabled" - no valid real calibration: speeds are in pixels/s
    Nothing is ever synthesised from the image size.
    """
    if force_uncalibrated:
        return None, "disabled", "DISABLED (forced via --uncalibrated; reporting in pixels/s)"

    explicit_path = path is not None
    cal_path = Path(path) if path else PROJECT_ROOT / "config" / "calibration.json"
    if not cal_path.is_absolute() and not cal_path.is_file():
        cal_path = PROJECT_ROOT / cal_path
    if not cal_path.is_file():
        if explicit_path:
            raise ValueError(f"Calibration file not found: '{cal_path}'")
        return None, "disabled", f"DISABLED (calibration file '{cal_path}' not found; reporting in pixels/s)"

    try:
        cfg = json.loads(cal_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read calibration file '{cal_path}': {exc}") from exc
    if not isinstance(cfg, dict):
        raise ValueError(f"Calibration file '{cal_path}' must contain a JSON object.")

    resolution = _parse_resolution(cfg.get("resolution", ""))
    if resolution is None:
        raise ValueError(
            f"Calibration file '{cal_path}' must declare a positive resolution such as '854x480'."
        )
    if resolution != (width, height):
        raise ValueError(
            f"Calibration resolution is {resolution[0]}x{resolution[1]}, "
            f"but the video is {width}x{height}."
        )

    image_points = cfg.get("image_points")
    if not isinstance(image_points, list) or len(image_points) != 4:
        raise ValueError(f"Calibration '{cal_path}' must contain four image_points.")
    for index, point in enumerate(image_points):
        if (
            not isinstance(point, list)
            or len(point) != 2
            or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in point)
        ):
            raise ValueError(f"Calibration image_points[{index}] must be a finite (x, y) pair.")
        if not (0 <= point[0] < width and 0 <= point[1] < height):
            raise ValueError(
                f"Calibration image_points[{index}]={point} is outside the declared "
                f"{width}x{height} image."
            )
    try:
        transformer = HomographyTransformer(image_points, cfg["world_points"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid point data in calibration file '{cal_path}': {exc}") from exc

    kind = str(cfg.get("calibration_type", "unverified")).lower()
    if kind == "real":
        return transformer, "real", f"ACTIVE - REAL calibration ('{cal_path}')"
    if allow_demo:
        return transformer, "demo", f"DEMO (unverified calibration '{cal_path}'; km/h are NOT physical)"
    return None, "disabled", (
        f"DISABLED ('{cal_path}' is not marked \"calibration_type\": \"real\"; reporting in pixels/s. "
        "Use --allow-demo-calibration for demonstration only)")

def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"\n[CLI Error] Input video file not found: {args.input}", file=sys.stderr)
        print("Please provide a valid video path via --input\n", file=sys.stderr)
        sys.exit(1)

    out = Path(args.output_dir)
    try:
        out.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"\n[CLI Error] Unable to create output directory '{args.output_dir}': {exc}\n", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"\n[CLI Error] OpenCV could not open video container: {args.input}\n", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    traffic_light_detector = None
    red_light_analyzer = None
    traffic_light_config = None
    if args.traffic_lights:
        roi_path = Path(args.traffic_light_roi)
        if not roi_path.is_absolute():
            roi_path = PROJECT_ROOT / roi_path
        try:
            traffic_light_config = load_roi_config(roi_path, width, height)
            light_model_path = Path(args.traffic_light_model)
            if not light_model_path.is_absolute():
                light_model_path = PROJECT_ROOT / light_model_path
            traffic_light_detector = TrafficLightDetector(
                model_path=light_model_path,
                light_polygon=traffic_light_config["light_polygon"],
                confidence=args.traffic_light_conf,
                image_size=args.traffic_light_imgsz,
                device=args.device,
            )
            red_light_analyzer = RedLightAnalyzer(
                vehicle_polygon=traffic_light_config["vehicle_polygon"],
                stop_line=traffic_light_config["stop_line"],
                frame_step=args.skip + 1,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"\n[CLI Error] Traffic-light initialization failed: {exc}\n", file=sys.stderr)
            sys.exit(1)

    # Fault-tolerant calibration initialization
    try:
        transformer, calib_mode, calib_status = load_calibration(
            path=args.calibration,
            width=width,
            height=height,
            force_uncalibrated=args.uncalibrated,
            allow_demo=args.allow_demo_calibration,
        )
    except ValueError as exc:
        print(f"\n[CLI Error] Invalid calibration: {exc}\n", file=sys.stderr)
        sys.exit(1)

    model_path = args.model
    if args.detector == "yolo" and not Path(model_path).is_file() and (PROJECT_ROOT / model_path).is_file():
        model_path = str(PROJECT_ROOT / model_path)

    try:
        detector = VehicleDetector(
            backend=args.detector,
            model_path=model_path,
            confidence=args.conf,
            image_size=args.imgsz,
            device=args.device,
            min_area=args.min_area,
        )
    except Exception as exc:
        print(f"\n[CLI Error] Detector initialization failed: {exc}\n", file=sys.stderr)
        sys.exit(1)

    tracker = MultiObjectTracker(
        max_distance=args.max_distance,
        max_missed=args.max_missed,
        iou_threshold=args.iou_threshold,
    )
    max_physical_speed = 38.0 if transformer else None  # m/s, same unit as the calibrated speed
    estimator = SpeedEstimator(
        fps=fps,
        transformer=transformer,
        max_speed=max_physical_speed,
        frame_step=args.skip + 1,
    )
    analyzer = TrafficAnalyzer(fps=fps, road_length_metres=40.0, num_lanes=4, frame_step=args.skip + 1)
    processor = VideoProcessor(detector, tracker, estimator, analyzer, calibration_mode=calib_mode,
                              speed_limit_kmh=args.speed_limit,
                              speeding_min_measures=args.speeding_measures,
                              speeding_seconds=args.speeding_seconds,
                              speeding_track_seconds=args.speeding_track_seconds,
                              speeding_dedup_seconds=args.speeding_dedup_seconds,
                              speeding_movement_seconds=args.speeding_movement_seconds,
                              speeding_min_confidence=args.speeding_min_confidence,
                              speeding_margin=args.speeding_margin,
                              speeding_max_plausible_kmh=args.speeding_max_plausible_kmh,
                              speeding_edge_margin=args.speeding_edge_margin,
                              speeding_max_width_jump=args.speeding_max_width_jump,
                              traffic_light_detector=traffic_light_detector,
                              red_light_analyzer=red_light_analyzer,
                              traffic_light_config=traffic_light_config)

    print("=" * 72)
    print("  INTELLIGENT TRAFFIC SURVEILLANCE & VEHICLE ANALYTICS (CSE3010)")
    print("=" * 72)
    print(f"  Input Video  : {args.input}")
    print(f"  Resolution   : {width}x{height} @ {fps:.2f} FPS")
    print(f"  Total Frames : {frame_count}")
    print(f"  Execution    : Strict Headless CLI Engine (No GUI)")
    print(f"  Detector     : {args.detector.upper()}")
    if args.detector == "yolo":
        print(f"  Model        : {model_path} ({args.imgsz}px inference on {detector.device})")
        print(f"  Confidence   : {args.conf:.2f}")
    if traffic_light_detector is not None:
        print(f"  Light Model  : {traffic_light_detector.model_path}")
        print(f"  Light Classes: {traffic_light_detector.class_names}")
        print(f"  Light ROI    : {traffic_light_config['path']}")
    else:
        print("  Traffic Lights: disabled")
    print(f"  Calibration  : {calib_status}")
    if args.skip > 0:
        print(f"  Frame Skip   : Process every {args.skip + 1}th frame")
    if args.max_frames:
        print(f"  Frame Limit  : Processing first {args.max_frames} frames")
    print("-" * 72)
    print("  [CLI Engine] Starting pipeline processing...\n")

    summary = processor.run(
        input_path=args.input,
        video_output=str(out / "annotated_video.mp4"),
        csv_output=str(out / "vehicle_data.csv"),
        json_output=str(out / "traffic_summary.json"),
        max_frames=args.max_frames,
        frame_skip=args.skip,
    )

    print("\n" + "=" * 72)
    print("  TRAFFIC SURVEILLANCE & KINEMATICS SUMMARY")
    print("=" * 72)
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"  {k:34s}:")
            for sub_k, sub_v in v.items():
                print(f"    - {sub_k:20s}: {sub_v}")
        else:
            print(f"  {k:34s}: {v}")
    print("-" * 72)
    print(f"  All outputs generated successfully in: {out.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()
