"""
05_roi_config.py
================
Outil interactif pour sélectionner les zones de détection et la ligne d'arrêt.
Les coordonnées enregistrées restent celles de la vidéo originale.
"""

import json
import argparse
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO_PATH = PROJECT_ROOT / "dataset" / "test" / "videos" / "traffic.mp4"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "roi_config_polygon_2pts.json"

MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 700
SCALE = 1.0
REFERENCE_WIDTH = 1280
REFERENCE_HEIGHT = 720

# État de la sélection souris
roi_polygon_points = []
stop_line_coords = []
current_roi_type = None


def to_display_point(point: tuple) -> tuple:
    """Convertit un point source vers les coordonnées affichées."""
    return tuple(int(round(value * SCALE)) for value in point)


def to_original_point(x: int, y: int, size: tuple) -> tuple:
    """Convertit un clic affiché vers les coordonnées de la vidéo."""
    width, height = size
    original_x = max(0, min(width - 1, int(round(x / SCALE))))
    original_y = max(0, min(height - 1, int(round(y / SCALE))))
    return original_x, original_y


def mouse_callback(event, x, y, flags, param):
    global stop_line_coords, roi_polygon_points, current_roi_type

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    frame_copy = param["frame_copy"].copy()
    window_name = param["window_name"]
    point = to_original_point(x, y, param["original_size"])

    if current_roi_type == "STOP_LINE":
        if len(stop_line_coords) < 2:
            stop_line_coords.append(point)
            print(f"Point de ligne d'arrêt {len(stop_line_coords)} : {point}")

            display_points = [to_display_point(p) for p in stop_line_coords]
            for coord in display_points:
                cv2.circle(frame_copy, coord, 5, (0, 0, 255), -1)

            if len(display_points) == 2:
                cv2.line(frame_copy, display_points[0], display_points[1], (0, 255, 255), 2)
                cv2.putText(frame_copy, "STOP LINE SELECTED",
                            (50, max(20, display_points[0][1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow(window_name, frame_copy)

    elif current_roi_type in ["VEHICLE_ROI", "RED_LIGHT_ROI"]:
        roi_polygon_points.append(point)
        print(f"Point {len(roi_polygon_points)} (coordonnées originales) : {point}")

        # Redessine tous les sommets pour garder les clics précédents visibles.
        display_points = [to_display_point(p) for p in roi_polygon_points]
        if len(display_points) > 1:
            pts = np.array(display_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame_copy, [pts], False, (0, 255, 0), 2)

        for coord in display_points:
            cv2.circle(frame_copy, coord, 5, (0, 0, 255), -1)

        cv2.imshow(window_name, frame_copy)


def select_polygon_roi(window_name: str, frame: np.ndarray, roi_type: str,
                       original_size: tuple) -> list | None:
    global roi_polygon_points, current_roi_type
    roi_polygon_points = []
    current_roi_type = roi_type

    print(f"\n--- Sélection : {roi_type} ---")
    print("Cliquez pour ajouter les sommets du polygone, puis appuyez sur 'q'.")

    temp_frame = frame.copy()
    cv2.imshow(window_name, temp_frame)
    cv2.setMouseCallback(
        window_name,
        mouse_callback,
        {"frame_copy": temp_frame, "window_name": window_name, "original_size": original_size},
    )

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyWindow(window_name)

    if len(roi_polygon_points) >= 3:
        final_frame = frame.copy()
        display_points = [to_display_point(p) for p in roi_polygon_points]
        pts = np.array(display_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(final_frame, [pts], True, (0, 255, 0), 2)
        cv2.imshow("Final Selected ROI", final_frame)
        cv2.waitKey(500)
        cv2.destroyWindow("Final Selected ROI")
        print(f"Zone {roi_type} configurée avec {len(roi_polygon_points)} sommets.")
        return roi_polygon_points

    print(f"Zone {roi_type} invalide : au moins 3 points sont nécessaires.")
    return None


def default_stop_line(width: int, height: int) -> list:
    """Garde la ligne par défaut à la même position relative dans l'image."""
    x1 = round(200 * width / REFERENCE_WIDTH)
    y = round(440 * height / REFERENCE_HEIGHT)
    x2 = round(800 * width / REFERENCE_WIDTH)
    return [x1, y, x2, y]


def select_roi_and_save(video_path: Path, output_json_path: Path):
    global stop_line_coords, current_roi_type, SCALE

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Erreur : impossible d'ouvrir la vidéo {video_path}")
        return

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Erreur : impossible de lire la première image de la vidéo.")
        return

    original_height, original_width = frame.shape[:2]
    SCALE = min(
        1.0,
        MAX_DISPLAY_WIDTH / original_width,
        MAX_DISPLAY_HEIGHT / original_height,
    )
    display_width = max(1, int(round(original_width * SCALE)))
    display_height = max(1, int(round(original_height * SCALE)))
    frame_display = cv2.resize(frame, (display_width, display_height),
                               interpolation=cv2.INTER_AREA)

    print(f"Résolution originale : {original_width} x {original_height}")
    print(f"Résolution affichée : {display_width} x {display_height} (échelle {SCALE:.4f})")

    config_data = {}

    # 1. Zone des véhicules
    vehicle_roi_points = select_polygon_roi(
        "Select VEHICLE DETECT ROI", frame_display, "VEHICLE_ROI",
        (original_width, original_height),
    )
    if vehicle_roi_points is None:
        print("Sélection de la zone véhicule annulée.")
        return
    config_data["VEHICLE_DETECT_ROI_POLYGON"] = vehicle_roi_points

    # 2. Zone du feu
    red_light_roi_points = select_polygon_roi(
        "Select RED LIGHT DETECT ROI", frame_display, "RED_LIGHT_ROI",
        (original_width, original_height),
    )
    if red_light_roi_points is None:
        print("Zone du feu obligatoire non sélectionnée. Configuration annulée.")
        return
    else:
        config_data["RED_LIGHT_DETECT_ROI_POLYGON"] = red_light_roi_points

    # 3. Ligne d'arrêt (deux clics)
    print("\nCliquez deux fois pour tracer la ligne d'arrêt. Appuyez sur 'q' pour la valeur par défaut.")
    stop_line_coords = []
    current_roi_type = "STOP_LINE"

    temp_frame = frame_display.copy()
    vehicle_display = np.array(
        [to_display_point(p) for p in config_data["VEHICLE_DETECT_ROI_POLYGON"]],
        np.int32,
    ).reshape((-1, 1, 2))
    cv2.polylines(temp_frame, [vehicle_display], True, (255, 0, 255), 2)

    red_light_display = np.array(
        [to_display_point(p) for p in config_data["RED_LIGHT_DETECT_ROI_POLYGON"]],
        np.int32,
    ).reshape((-1, 1, 2))
    cv2.polylines(temp_frame, [red_light_display], True, (255, 255, 0), 2)

    window_name = "Select STOP LINE (2 Points)"
    cv2.imshow(window_name, temp_frame)
    cv2.setMouseCallback(
        window_name,
        mouse_callback,
        {
            "frame_copy": temp_frame,
            "window_name": window_name,
            "original_size": (original_width, original_height),
        },
    )

    while len(stop_line_coords) < 2:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyWindow(window_name)

    if len(stop_line_coords) < 2:
        config_data["STOP_LINE_COORDS"] = default_stop_line(original_width, original_height)
        print(f"Ligne d'arrêt par défaut (coordonnées originales) : {config_data['STOP_LINE_COORDS']}")
    else:
        x1, y1 = stop_line_coords[0]
        x2, y2 = stop_line_coords[1]
        config_data["STOP_LINE_COORDS"] = [x1, y1, x2, y2]

    with open(output_json_path, "w", encoding="utf-8") as f:
        config_data["resolution"] = f"{original_width}x{original_height}"
        json.dump(config_data, f, indent=4)

    print(f"\nConfiguration sauvegardée dans {output_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select vehicle ROI, traffic-light ROI and stop line for a video."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO_PATH, help="Video to configure")
    parser.add_argument("--output", type=Path, default=DEFAULT_CONFIG_PATH, help="ROI JSON output path")
    options = parser.parse_args()
    select_roi_and_save(options.video, options.output)
