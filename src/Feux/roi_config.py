"""Validation and loading for traffic-light and stop-line ROI configuration."""

import json
import math
from pathlib import Path

import cv2
import numpy as np


def _polygon(config, key, width, height):
    points = config.get(key)
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError(f"ROI configuration must contain at least three points in '{key}'.")
    parsed = []
    for index, point in enumerate(points):
        if (
            not isinstance(point, (list, tuple))
            or len(point) != 2
            or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in point)
        ):
            raise ValueError(f"'{key}[{index}]' must be a finite (x, y) point.")
        x, y = point
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(
                f"'{key}[{index}]'={point} is outside the {width}x{height} video."
            )
        parsed.append((int(round(x)), int(round(y))))
    polygon = np.asarray(parsed, dtype=np.int32)
    if abs(cv2.contourArea(polygon)) < 1:
        raise ValueError(f"'{key}' must enclose a non-zero area.")
    return parsed


def load_roi_config(path, width, height):
    """Load both required polygons and the stop line, validating against video size."""
    config_path = Path(path)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read traffic-light ROI config '{config_path}': {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Traffic-light ROI config '{config_path}' must contain a JSON object.")

    declared_resolution = config.get("resolution")
    if declared_resolution is not None:
        try:
            declared_width, declared_height = (
                int(value) for value in str(declared_resolution).lower().split("x")
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"ROI config resolution must look like '1920x1080', got {declared_resolution!r}."
            ) from exc
        if (declared_width, declared_height) != (width, height):
            raise ValueError(
                f"ROI config resolution is {declared_width}x{declared_height}, "
                f"but the video is {width}x{height}."
            )

    vehicle_polygon = _polygon(config, "VEHICLE_DETECT_ROI_POLYGON", width, height)
    light_polygon = _polygon(config, "RED_LIGHT_DETECT_ROI_POLYGON", width, height)
    line = config.get("STOP_LINE_COORDS")
    if (
        not isinstance(line, list)
        or len(line) != 4
        or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in line)
    ):
        raise ValueError("'STOP_LINE_COORDS' must contain four finite values [x1, y1, x2, y2].")
    x1, y1, x2, y2 = (int(round(value)) for value in line)
    if not (0 <= x1 < width and 0 <= x2 < width and 0 <= y1 < height and 0 <= y2 < height):
        raise ValueError(f"'STOP_LINE_COORDS'={line} is outside the {width}x{height} video.")
    if (x1, y1) == (x2, y2):
        raise ValueError("'STOP_LINE_COORDS' must define a non-zero-length line.")

    return {
        "resolution": f"{width}x{height}",
        "vehicle_polygon": vehicle_polygon,
        "light_polygon": light_polygon,
        "stop_line": [x1, y1, x2, y2],
        "path": str(config_path),
    }
