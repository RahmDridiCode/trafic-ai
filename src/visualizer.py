import cv2

KMH_PER_MS = 3.6

# BGR colours per class (bright, high contrast on dark label plates).
CLASS_COLORS = {
    "bus": (0, 165, 255),
    "car": (0, 255, 0),
    "motorcycle": (255, 0, 255),
    "truck": (0, 215, 255),
}
DEFAULT_COLOR = (255, 200, 0)
PLATE_BORDER_COLOR = (255, 255, 255)
NORMAL_BOX_COLOR = (0, 255, 0)
SPEEDING_BOX_COLOR = (0, 0, 255)
SPEEDING_TEXT_COLOR = (90, 90, 255)
RED_LIGHT_COLOR = (0, 165, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def frame_scale(frame):
    """Text/line scale from the frame resolution (geometric mean works for portrait and landscape)."""
    h, w = frame.shape[:2]
    return max(0.7, min(3.0, (w * h) ** 0.5 / 1500.0))


def speed_to_display(speed, calibrated, demo_calibration=False):
    """Return speed in km/h only for real or explicitly marked demo metric calibrations."""
    unit = "km/h" if calibrated else "km/h (DEMO; unverified)" if demo_calibration else "px/s"
    if speed is None:
        return None, unit
    return (speed * KMH_PER_MS if calibrated or demo_calibration else speed), unit


def _overlaps(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _text_sizes(lines, font_scale, thickness):
    return [cv2.getTextSize(t, FONT, font_scale, thickness) for t in lines]


def _place_plate(box, pw, ph, img_w, img_h, placed, gap):
    """Pick a plate rectangle fully inside the image, avoiding earlier plates when possible."""
    x, y, w, h = box
    candidates = []
    for px in (x, x + w - pw):
        candidates.append((px, y - ph - gap))        # above
        candidates.append((px, y + h + gap))        # below
        candidates.append((px, y + gap))            # inside top
        candidates.append((px, y + h - ph - gap))   # inside bottom
    fallback = None
    for px, py in candidates:
        px = int(max(0, min(img_w - pw, px)))
        py = int(max(0, min(img_h - ph, py)))
        rect = (px, py, px + pw, py + ph)
        if fallback is None:
            fallback = rect
        if not any(_overlaps(rect, r) for r in placed):
            return rect
    return fallback


def draw_tracks(frame, tracks, speeds, calibrated=False, speeding_ids=None,
                demo_calibration=False, speed_statuses=None, red_light_ids=None):
    """Draw boxes, trails and readable ID/class/confidence/speed plates.

    `speeds` are raw estimator values (m/s when calibrated, px/s otherwise).
    """
    # speeding_ids: tracks in a speeding episode get a red box + SPEEDING line; others a green box.
    speeding_ids = speeding_ids or ()
    speed_statuses = speed_statuses or {}
    red_light_ids = red_light_ids or ()
    img_h, img_w = frame.shape[:2]
    scale = frame_scale(frame)
    box_thick = max(3, int(round(3 * scale)))
    font_scale = 0.75 * scale
    font_thick = max(2, int(round(2 * scale)))
    pad = max(4, int(round(6 * scale)))
    gap = max(2, int(round(3 * scale)))

    active = [t for t in tracks if t.missed == 0]
    for t in active:
        color = NORMAL_BOX_COLOR
        if t.track_id in red_light_ids:
            color = RED_LIGHT_COLOR
        if t.track_id in speeding_ids:
            color = SPEEDING_BOX_COLOR
        x, y, w, h = t.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, box_thick)

    placed = []
    # Larger (closer) vehicles first: they get the preferred label positions.
    for t in sorted(active, key=lambda tr: -(tr.bbox[2] * tr.bbox[3])):
        speeding = t.track_id in speeding_ids
        value, unit = speed_to_display(
            speeds.get(t.track_id), calibrated, demo_calibration=demo_calibration
        )
        speed_prefix = "~" if speed_statuses.get(t.track_id) in ("provisional", "held") else ""
        lines = [
            f"ID: {t.track_id} | {t.class_name.upper()} | Conf: {t.confidence:.2f}",
            "Speed: N/A" if value is None else f"Speed: {speed_prefix}{value:.1f} {unit}",
        ]
        red_light = t.track_id in red_light_ids
        if speeding:
            lines.append("SPEEDING")
        if red_light:
            lines.append("RED LIGHT")
        sizes = _text_sizes(lines, font_scale, font_thick)
        line_h = max(s[0][1] + s[1] for s in sizes) + pad // 2
        pw = min(img_w, max(s[0][0] for s in sizes) + 2 * pad)
        ph = min(img_h, line_h * len(lines) + pad)
        x1, y1, x2, y2 = _place_plate(t.bbox, pw, ph, img_w, img_h, placed, gap)
        placed.append((x1, y1, x2, y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 20, 20), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), SPEEDING_BOX_COLOR if speeding else RED_LIGHT_COLOR if red_light else PLATE_BORDER_COLOR,
                      2 if speeding or red_light else 1)
        for i, (text, s) in enumerate(zip(lines, sizes)):
            baseline_y = y1 + pad // 2 + (i + 1) * line_h - s[1]
            text_color = (255, 255, 255) if i == 0 else (80, 255, 120)
            if i >= 1 and text == "RED LIGHT":
                text_color = RED_LIGHT_COLOR
            elif speeding and i >= 1:
                text_color = SPEEDING_TEXT_COLOR
            cv2.putText(frame, text, (x1 + pad, baseline_y), FONT, font_scale, text_color,
                        font_thick, cv2.LINE_AA)
    return frame


def draw_hud(frame, frame_no, active, counts, calibrated, calibration_label=None):
    scale = frame_scale(frame)
    font_scale = 0.75 * scale
    font_thick = max(2, int(round(2 * scale)))
    pad = int(round(12 * scale))

    if calibration_label is None:
        calibration_label = (
            "Speed: Calibrated (Road Homography -> km/h)" if calibrated
            else "Speed: Pixel displacement (px/s) - NOT calibrated"
        )
    lines = [
        f"Frame: {frame_no}",
        f"Active Vehicles: {active}",
        "Class Breakdown: " + (", ".join(f"{k}={v}" for k, v in sorted(counts.items())) if counts else "None"),
        calibration_label,
    ]
    sizes = _text_sizes(lines, font_scale, font_thick)
    line_h = max(s[0][1] + s[1] for s in sizes) + int(round(8 * scale))
    plate_w = min(frame.shape[1] - 20, max(s[0][0] for s in sizes) + 2 * pad)
    plate_h = line_h * len(lines) + pad

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + plate_w, 10 + plate_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    y = 10 + pad // 2
    for text, s in zip(lines, sizes):
        y += line_h
        cv2.putText(frame, text, (10 + pad, y - s[1]), FONT, font_scale,
                    (255, 255, 255), font_thick, cv2.LINE_AA)
    return frame
