"""Détection des véhicules, des feux rouges et export vidéo annoté."""

import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_WEIGHTS = BASE_DIR / "best_n.pt"
VIDEO_PATH = BASE_DIR / "video.mp4"
CONFIG_FILE = BASE_DIR / "roi_config_polygon_2pts.json"
VIOLATION_OUTPUT_DIR = BASE_DIR / "Output"
ANNOTATED_VIDEO_PATH = VIOLATION_OUTPUT_DIR / f"traffic_annotated_{time.strftime('%Y%m%d_%H%M%S')}.mp4"  # Nom unique par exécution.

# Seuils et classes
INFERENCE_SIZE = 640
LIGHT_CONF_THRESHOLD = 0.5
VEHICLE_CONF_THRESHOLD = 0.6
CONFIDENCE_THRESHOLD = min(LIGHT_CONF_THRESHOLD, VEHICLE_CONF_THRESHOLD)
CAR_CLASS_ID = 0
MOTORBIKE_CLASS_ID = 1
GREEN_LIGHT_CLASS_ID = 2
RED_LIGHT_CLASS_ID = 3
CLASSES_TO_TRACK = [CAR_CLASS_ID, MOTORBIKE_CLASS_ID, RED_LIGHT_CLASS_ID, GREEN_LIGHT_CLASS_ID]
ALL_CLASS_NAMES = ["car", "motorbike", "green", "red", "yellow"]

# Détection HSV du feu rouge
HSV_RED_LOW_1 = (0, 70, 70)
HSV_RED_HIGH_1 = (12, 255, 255)
HSV_RED_LOW_2 = (168, 70, 70)
HSV_RED_HIGH_2 = (180, 255, 255)
RED_RATIO_THRESHOLD = 0.01

# Détection des infractions
VIOLATION_THRESHOLD = 20
MAX_PAST_LINE = 350
MIN_UPWARD_DISPLACEMENT = 10
POSITION_HISTORY_SIZE = 5
DEBUG = False  # Masque les informations techniques des véhicules.
SHOW_ZONES = False  # Masque les zones ROI, la ligne et les cadres des feux.
SHOW_STATUS = True  # Affiche le statut du feu et le ratio rouge.
CAPTURE_PADDING = 50
VIDEO_CODEC = "mp4v"

# Hystérésis du feu
RED_LIGHT_DETECTION_HISTORY_SIZE = 5
RED_LIGHT_MIN_DETECTIONS = 3
GREEN_LIGHT_MIN_DETECTIONS = 3


def preprocess_frame_for_infer(frame_bgr: np.ndarray, target: int = INFERENCE_SIZE):
    """Redimensionne l'image et applique un léger lissage et accentuation."""
    orig_h, orig_w = frame_bgr.shape[:2]
    r = min(target / orig_h, target / orig_w)
    new_w, new_h = int(round(orig_w * r)), int(round(orig_h * r))
    img = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    top = (target - new_h) // 2
    bottom = target - new_h - top
    left = (target - new_w) // 2
    right = target - new_w - left

    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             borderType=cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    img = cv2.filter2D(img, -1, kernel)

    return img, r, (left, top), (orig_h, orig_w)


def red_ratio_in_roi(frame: np.ndarray, polygon: np.ndarray) -> float:
    """Retourne la proportion de pixels rouges dans le polygone du feu."""
    if polygon.size < 6:
        return 0.0

    points = polygon.reshape((-1, 2))
    frame_height, frame_width = frame.shape[:2]
    x_min = max(0, int(np.min(points[:, 0])))
    y_min = max(0, int(np.min(points[:, 1])))
    x_max = min(frame_width, int(np.max(points[:, 0])) + 1)
    y_max = min(frame_height, int(np.max(points[:, 1])) + 1)
    if x_max <= x_min or y_max <= y_min:
        return 0.0

    roi_frame = frame[y_min:y_max, x_min:x_max]
    roi_points = points - np.array([x_min, y_min], dtype=np.int32)
    roi_mask = np.zeros(roi_frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(roi_mask, [roi_points.reshape((-1, 1, 2))], 255)
    roi_pixels = cv2.countNonZero(roi_mask)
    if roi_pixels == 0:
        return 0.0

    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    red_low_1 = cv2.inRange(
        hsv, np.array(HSV_RED_LOW_1, dtype=np.uint8), np.array(HSV_RED_HIGH_1, dtype=np.uint8)
    )
    red_low_2 = cv2.inRange(
        hsv, np.array(HSV_RED_LOW_2, dtype=np.uint8), np.array(HSV_RED_HIGH_2, dtype=np.uint8)
    )
    red_mask = cv2.bitwise_or(red_low_1, red_low_2)
    red_in_roi = cv2.bitwise_and(red_mask, roi_mask)
    return cv2.countNonZero(red_in_roi) / roi_pixels


def stop_line_y_at_x(x: int, stop_line_coords: list[int]) -> float:
    """Interpole la position verticale de la ligne à l'abscisse x."""
    x1, y1, x2, y2 = stop_line_coords
    if x1 == x2:
        return float((y1 + y2) / 2)
    ratio = min(1.0, max(0.0, (x - x1) / (x2 - x1)))
    return y1 + ratio * (y2 - y1)


def upward_motion(history: deque) -> tuple[bool, int, int]:
    """Vérifie le mouvement vers le haut et renvoie dx, dy sur l'historique."""
    if len(history) < 2:
        return False, 0, 0
    dx = history[-1][0] - history[0][0]
    dy = history[-1][1] - history[0][1]
    moving_up = dy <= -MIN_UPWARD_DISPLACEMENT and abs(dy) > abs(dx)
    return moving_up, dx, dy


def capture_violation_image(frame: np.ndarray, track_id: int, x1: int, y1: int, x2: int, y2: int, output_dir: Path):
    """Enregistre une capture recadrée et l'image complète."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename_base = f"id{track_id}_{timestamp}"

    x1_crop = max(0, x1 - CAPTURE_PADDING)
    y1_crop = max(0, y1 - CAPTURE_PADDING)
    x2_crop = min(frame.shape[1], x2 + CAPTURE_PADDING)
    y2_crop = min(frame.shape[0], y2 + CAPTURE_PADDING)

    cropped_img = frame[y1_crop:y2_crop, x1_crop:x2_crop]

    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / f"{filename_base}_crop.jpg"), cropped_img)
    cv2.imwrite(str(output_dir / f"{filename_base}_full.jpg"), frame)

    print(f"Captured VIOLATION snapshot for vehicle track ID #{track_id}")


def main():
    print("--- Starting Traffic Light & Red Light Violation Detection ---")

    if not CONFIG_FILE.exists():
        print(f"Error: Config file {CONFIG_FILE} not found. Please run 05_roi_config.py first.")
        sys.exit(1)

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    vehicle_roi_polygon = np.array(config_data.get("VEHICLE_DETECT_ROI_POLYGON", []), np.int32)
    red_light_roi_polygon = np.array(config_data.get("RED_LIGHT_DETECT_ROI_POLYGON", []), np.int32)
    stop_line_coords = config_data.get("STOP_LINE_COORDS", [200, 440, 800, 440])
    VIOLATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        model = YOLO(str(MODEL_WEIGHTS))
    except Exception as e:
        print(f"Error loading model weights: {e}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"Error: Cannot open video file at {VIDEO_PATH}")
        sys.exit(1)

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not np.isfinite(source_fps) or source_fps <= 0 or frame_width <= 0 or frame_height <= 0:
        cap.release()
        print("Error: Source video has invalid FPS or frame dimensions.")
        sys.exit(1)

    video_writer = cv2.VideoWriter(
        str(ANNOTATED_VIDEO_PATH),
        cv2.VideoWriter_fourcc(*VIDEO_CODEC),
        source_fps,
        (frame_width, frame_height),
    )
    if not video_writer.isOpened():
        cap.release()
        video_writer.release()
        print(f"Error: Cannot create annotated video at {ANNOTATED_VIDEO_PATH}")
        sys.exit(1)

    red_light_active = False
    red_light_history = deque(maxlen=RED_LIGHT_DETECTION_HISTORY_SIZE)
    green_light_history = deque(maxlen=RED_LIGHT_DETECTION_HISTORY_SIZE)
    captured_violations = set()
    violating_vehicles = set()
    vehicle_position_history: dict[int, deque] = {}
    crossing_red_state: dict[int, bool] = {}
    preview_enabled = False
    frames_written = 0

    print(f"Writing complete annotated video to {ANNOTATED_VIDEO_PATH}")
    print("Running detection pipeline. Press 'q' to close the preview.")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_pre, r, (left, top), (orig_h, orig_w) = preprocess_frame_for_infer(
                frame, target=INFERENCE_SIZE
            )

            results = model.track(frame_pre, persist=True, classes=CLASSES_TO_TRACK,
                                  conf=CONFIDENCE_THRESHOLD, verbose=False, imgsz=INFERENCE_SIZE)

            boxes = results[0].boxes if len(results) > 0 else []
            red_ratio = red_ratio_in_roi(frame, red_light_roi_polygon)

            # L'état du feu dépend uniquement du ratio rouge dans sa ROI.
            current_frame_red = red_ratio >= RED_RATIO_THRESHOLD
            current_frame_green = not current_frame_red
            red_light_history.append(current_frame_red)
            green_light_history.append(current_frame_green)

            if sum(red_light_history) >= RED_LIGHT_MIN_DETECTIONS:
                red_light_active = True
            elif sum(green_light_history) >= GREEN_LIGHT_MIN_DETECTIONS:
                red_light_active = False

            # L'affichage est optionnel ; les zones restent actives pour les calculs.
            if SHOW_ZONES:  # Affiche les zones et la ligne seulement si demandé.
                if vehicle_roi_polygon.size >= 6:
                    cv2.polylines(frame, [vehicle_roi_polygon.reshape((-1, 1, 2))], True, (255, 0, 255), 2)
                if red_light_roi_polygon.size >= 6:
                    cv2.polylines(frame, [red_light_roi_polygon.reshape((-1, 1, 2))], True, (255, 255, 0), 2)
                if len(stop_line_coords) == 4:
                    cv2.line(frame, (stop_line_coords[0], stop_line_coords[1]),
                             (stop_line_coords[2], stop_line_coords[3]), (0, 0, 255), 2)

            # Affiche la classe du feu détectée par YOLO, corrigée par l'état combiné.
            if SHOW_ZONES and boxes is not None and len(boxes) > 0:  # Affiche les boîtes YOLO du feu.
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = box.conf[0].item()
                    if cls_id in [RED_LIGHT_CLASS_ID, GREEN_LIGHT_CLASS_ID] and conf >= LIGHT_CONF_THRESHOLD:
                        x1_box, y1_box, x2_box, y2_box = box.xyxy[0].tolist()
                        x1 = max(0, min(frame_width - 1, int((x1_box - left) / r)))
                        y1 = max(0, min(frame_height - 1, int((y1_box - top) / r)))
                        x2 = max(0, min(frame_width - 1, int((x2_box - left) / r)))
                        y2 = max(0, min(frame_height - 1, int((y2_box - top) / r)))
                        detected_red = red_light_active
                        light_color = (0, 0, 255) if detected_red else (0, 255, 0)
                        light_label = "RED LIGHT" if detected_red else "GREEN LIGHT"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), light_color, 2)
                        cv2.putText(frame, light_label, (x1, max(0, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, light_color, 2)

            # Suit le point bas-centre et mémorise l'état du feu au passage.
            if boxes is not None and len(boxes) > 0 and boxes.id is not None:
                for box, track_id in zip(boxes, boxes.id.tolist()):
                    cls_id = int(box.cls[0])
                    conf = box.conf[0].item()

                    if cls_id in [CAR_CLASS_ID, MOTORBIKE_CLASS_ID] and conf >= VEHICLE_CONF_THRESHOLD:
                        # Convertit les coordonnées vers l'image originale.
                        x1_box, y1_box, x2_box, y2_box = box.xyxy[0].tolist()
                        x1 = max(0, min(frame_width - 1, int((x1_box - left) / r)))
                        y1 = max(0, min(frame_height - 1, int((y1_box - top) / r)))
                        x2 = max(0, min(frame_width - 1, int((x2_box - left) / r)))
                        y2 = max(0, min(frame_height - 1, int((y2_box - top) / r)))

                        point = (int((x1 + x2) / 2), y2)
                        y_line = stop_line_y_at_x(point[0], stop_line_coords)
                        past = y_line - point[1]

                        history = vehicle_position_history.setdefault(
                            track_id, deque(maxlen=POSITION_HISTORY_SIZE)
                        )
                        previous_past = history[-1][2] if history else None
                        history.append((point[0], point[1], past, red_light_active))
                        moving_up, dx, dy = upward_motion(history)

                        # Si le suivi commence déjà après la ligne, la première
                        # observation sert d'estimation du moment du franchissement.
                        if past <= VIOLATION_THRESHOLD:
                            crossing_red_state.pop(track_id, None)
                        elif track_id not in crossing_red_state:
                            if previous_past is None:
                                crossing_red_state[track_id] = (
                                    red_light_active and past <= MAX_PAST_LINE
                                )
                            elif previous_past <= VIOLATION_THRESHOLD:
                                crossing_red_state[track_id] = (
                                    red_light_active and point[1] < history[-2][1]
                                    and abs(dy) > abs(dx)
                                )

                        inside_vehicle_roi = (
                            vehicle_roi_polygon.size >= 6
                            and cv2.pointPolygonTest(
                                vehicle_roi_polygon.reshape((-1, 1, 2)),
                                (float(point[0]), float(point[1])),
                                False,
                            ) >= 0
                        )
                        is_violating = (
                            red_light_active
                            and crossing_red_state.get(track_id, False)
                            and inside_vehicle_roi
                            and VIOLATION_THRESHOLD <= past <= MAX_PAST_LINE
                            and moving_up
                        )
                        if is_violating:
                            violating_vehicles.add(track_id)
                        is_violating = track_id in violating_vehicles

                        box_color = (0, 0, 255) if is_violating else (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                        label = f"ID:{int(track_id)} {ALL_CLASS_NAMES[cls_id]} {conf:.2f}"
                        if is_violating:
                            label += " VIOLATION"
                        if DEBUG:
                            label += f" yL:{y_line:.0f} past:{past:.0f} dx:{dx} dy:{dy}"
                        cv2.putText(frame, label, (x1, max(0, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                        if is_violating and track_id not in captured_violations:
                            captured_violations.add(track_id)
                            capture_violation_image(
                                frame, int(track_id), x1, y1, x2, y2, VIOLATION_OUTPUT_DIR
                            )

            # Affiche l'état du feu et le ratio rouge pour le réglage HSV.
            light_status = "RED LIGHT ACTIVE" if red_light_active else "GREEN / NORMAL LIGHT"
            status_color = (0, 0, 255) if red_light_active else (0, 255, 0)
            # Garde les diagnostics du feu visibles si demandé.
            if SHOW_STATUS:  # Affiche le statut et le ratio si demandé.
                cv2.putText(frame, f"STATUS: {light_status}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                cv2.putText(
                    frame,
                    f"red ratio: {red_ratio:.3f}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255) if red_ratio >= RED_RATIO_THRESHOLD else (255, 255, 255),
                    2,
                )

            video_writer.write(frame)
            frames_written += 1

            if preview_enabled:
                cv2.imshow("Red Light Violation Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    preview_enabled = False
                    cv2.destroyWindow("Red Light Violation Detection")
    finally:
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()

    output_check = cv2.VideoCapture(str(ANNOTATED_VIDEO_PATH))
    output_is_playable = False
    if output_check.isOpened():
        output_read, output_frame = output_check.read()
        output_fps = output_check.get(cv2.CAP_PROP_FPS)
        output_width = int(output_check.get(cv2.CAP_PROP_FRAME_WIDTH))
        output_height = int(output_check.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_frame_count = int(output_check.get(cv2.CAP_PROP_FRAME_COUNT))
        output_is_playable = (
            output_read
            and output_frame is not None
            and output_frame.shape[1] == frame_width
            and output_frame.shape[0] == frame_height
            and np.isclose(output_fps, source_fps, rtol=1e-3)
            and output_frame_count == frames_written
        )
    output_check.release()
    if not output_is_playable:
        print(f"Error: Annotated video is incomplete or has invalid playback metadata: {ANNOTATED_VIDEO_PATH}")
        sys.exit(1)

    print(f"Annotated video saved to {ANNOTATED_VIDEO_PATH}")
    print("Detection pipeline terminated.")


if __name__ == "__main__":
    main()
