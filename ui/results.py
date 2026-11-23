"""Reading and comparing run outputs. Only files that exist are reported; nothing is estimated."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

META_FILE = "ui_run.json"
LOG_FILE = "run.log"
CSV_FILES = {
    "vehicles": "vehicle_data.csv",
    "speeding": "speeding_violations.csv",
    "alerts": "speed_alerts.csv",
    "red_light": "red_light_violations.csv",
    "diagnostic": "alert_diagnostic.csv",
}
# Config entries of the summary that make two runs not strictly comparable when they differ.
COMPARABLE_CONFIG = [
    ("input_resolution", "Résolution de la vidéo"),
    ("video_fps", "FPS de la vidéo"),
    ("frame_skip", "Images ignorées (--skip)"),
    ("calibration_mode", "Mode de calibration"),
    ("traffic_lights.enabled", "Détection des feux"),
    ("traffic_lights.model", "Modèle des feux"),
    ("traffic_lights.roi_config", "Fichier ROI"),
    ("speeding.speed_limit_kmh", "Limite de vitesse (km/h)"),
]


@dataclass
class Run:
    path: Path
    summary: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.path.name

    def file(self, key: str) -> Path:
        return self.path / CSV_FILES[key]

    def table(self, key: str) -> pd.DataFrame | None:
        p = self.file(key)
        if not p.is_file():
            return None
        try:
            return pd.read_csv(p)
        except (pd.errors.EmptyDataError, UnicodeError):
            return pd.DataFrame()

    @property
    def lights_enabled(self) -> bool | None:
        value = get_path(self.summary, "traffic_lights.enabled")
        return None if value is None else bool(value)

    @property
    def video(self) -> Path:
        return self.path / "annotated_video.mp4"

    def images(self, folder: str) -> list[Path]:
        d = self.path / folder
        return sorted(d.glob("*.jpg")) if d.is_dir() else []


def get_path(data: dict[str, Any], dotted: str) -> Any:
    current: Any = data
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def load_run(path: str | Path) -> Run | None:
    path = Path(path)
    summary_file = path / "traffic_summary.json"
    if not summary_file.is_file():
        return None
    try:
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    meta: dict[str, Any] = {}
    meta_file = path / META_FILE
    if meta_file.is_file():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
    return Run(path, summary, meta)


def flatten(data: Any, prefix: str = "") -> dict[str, Any]:
    """Scalar leaves of the summary with dotted keys (event lists are excluded)."""
    out: dict[str, Any] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            out.update(flatten(value, f"{prefix}{key}."))
    elif not isinstance(data, list):
        out[prefix[:-1]] = data
    return out


def headline_stats(run: Run) -> list[tuple[str, Any]]:
    """Most useful statistics that are actually present in this run's summary."""
    s = run.summary
    wanted = [
        ("Images traitées", "frames_processed"), ("Véhicules uniques suivis", "unique_vehicle_tracks"),
        ("Détections totales", "total_vehicle_detections"),
        ("Véhicules actifs / image (moy.)", "average_active_vehicles_per_frame"),
        ("Pic de véhicules actifs", "peak_active_vehicles"),
        ("Niveau de service", "level_of_service"), ("Congestion", "congestion_level"),
        ("Débit horaire estimé (veh/h)", "estimated_hourly_flow_rate_vph"),
        ("% véhicules lourds", "heavy_vehicle_percentage"),
    ]
    if get_path(s, "speed_calibrated"):
        wanted += [("Vitesse moyenne (km/h)", "average_speed_kmh"), ("Vitesse max (km/h)", "maximum_speed_kmh"),
                   ("V85 (km/h)", "speed_percentile_85_kmh")]
    else:
        unit = s.get("speed_unit", "")
        wanted += [(f"Vitesse moyenne ({unit})", "average_speed"), (f"Vitesse max ({unit})", "maximum_speed")]
    wanted += [("Infractions de vitesse", "speeding.violation_count"),
               ("Alertes de vitesse", "speed_alerts.alert_count")]
    if run.lights_enabled:
        wanted.append(("Infractions feu rouge", "traffic_lights.red_light_violation_count"))
    return [(label, get_path(s, key)) for label, key in wanted if get_path(s, key) is not None]


def light_state_counts(run: Run) -> pd.Series | None:
    df = run.table("vehicles")
    if df is None or "traffic_light_state" not in df.columns or df.empty:
        return None
    per_frame = df.drop_duplicates("frame")["traffic_light_state"]
    return per_frame.value_counts()


def events(run: Run) -> pd.DataFrame:
    """Chronological list of detected events (speeding, speed alerts, red-light crossings)."""
    fps = float(run.summary.get("video_fps") or 0)
    rows: list[dict[str, Any]] = []
    speeding = run.table("speeding")
    if speeding is not None and not speeding.empty:
        for r in speeding.itertuples():
            rows.append({"temps_s": r.timestamp, "image": r.frame_number, "type": "Excès de vitesse",
                         "id": r.track_id, "classe": r.vehicle_class,
                         "détail": f"{r.measured_speed} km/h pour une limite de {r.speed_limit}"})
    alerts = run.table("alerts")
    if alerts is not None and not alerts.empty:
        for r in alerts.itertuples():
            rows.append({"temps_s": round(r.frame / fps, 3) if fps else None, "image": r.frame,
                         "type": "Alerte vitesse", "id": r.track_id, "classe": r.vehicle_class,
                         "détail": f"{r.speed_kmh} km/h (limite {r.speed_limit})"})
    red = run.table("red_light")
    if red is not None and not red.empty:
        for r in red.itertuples():
            rows.append({"temps_s": r.timestamp, "image": r.frame, "type": "Feu rouge franchi",
                         "id": r.track_id, "classe": r.vehicle_class, "détail": f"feu {r.light_state}"})
    df = pd.DataFrame(rows, columns=["temps_s", "image", "type", "id", "classe", "détail"])
    return df.sort_values(["temps_s", "image"], na_position="last").reset_index(drop=True)


# ---------------------------------------------------------------- comparison
@dataclass
class Comparison:
    metrics: pd.DataFrame
    config: pd.DataFrame
    warnings: list[str]
    red_light_overlap: dict[str, list[int]] | None
    class_counts: pd.DataFrame


def _fmt(value: Any) -> Any:
    return value if value is not None else "—"


def compare_runs(a: Run, b: Run) -> Comparison:
    fa, fb = flatten(a.summary), flatten(b.summary)
    rows = []
    for key in sorted(set(fa) | set(fb)):
        va, vb = fa.get(key), fb.get(key)
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (va, vb))
        if key.startswith(("vehicle_detections_by_class", "fleet_modal_split")) or key in ("output_video", "output_csv"):
            continue
        if key.endswith(("output_csv", "images_dir", "diagnostic_csv", "violations_csv")):
            continue
        rows.append({"indicateur": key, "A": _fmt(va), "B": _fmt(vb),
                     "écart (B − A)": round(vb - va, 3) if numeric else None})
    metrics = pd.DataFrame(rows)

    config_rows, warnings = [], []
    for key, label in COMPARABLE_CONFIG:
        va, vb = get_path(a.summary, key), get_path(b.summary, key)
        config_rows.append({"paramètre": label, "A": _fmt(va), "B": _fmt(vb), "identique": va == vb})
    for key in ("input", "model"):
        va, vb = a.meta.get("values", {}).get(key), b.meta.get("values", {}).get(key)
        if va is not None or vb is not None:
            config_rows.append({"paramètre": f"{key} (lancement via l'interface)", "A": _fmt(va), "B": _fmt(vb),
                                "identique": va == vb})
    config = pd.DataFrame(config_rows)
    for row in config_rows:
        if not row["identique"] and row["paramètre"] in ("Résolution de la vidéo", "FPS de la vidéo",
                                                         "Images ignorées (--skip)"):
            warnings.append(f"« {row['paramètre']} » diffère ({row['A']} ≠ {row['B']}) : "
                            "les deux exécutions ne portent peut-être pas sur la même vidéo/échantillonnage.")
    if a.summary.get("frames_processed") != b.summary.get("frames_processed"):
        warnings.append("Le nombre d'images traitées diffère : comparez les valeurs absolues avec prudence.")

    classes_a = a.summary.get("vehicle_detections_by_class") or {}
    classes_b = b.summary.get("vehicle_detections_by_class") or {}
    class_counts = pd.DataFrame(
        [{"classe": c, "A": classes_a.get(c, 0), "B": classes_b.get(c, 0)} for c in sorted(set(classes_a) | set(classes_b))])

    overlap = None
    ra, rb = a.table("red_light"), b.table("red_light")
    if a.lights_enabled or b.lights_enabled:
        ids_a = set(ra["track_id"]) if ra is not None and "track_id" in ra else set()
        ids_b = set(rb["track_id"]) if rb is not None and "track_id" in rb else set()
        overlap = {"communs": sorted(ids_a & ids_b), "seulement A": sorted(ids_a - ids_b),
                   "seulement B": sorted(ids_b - ids_a)}
        if a.lights_enabled != b.lights_enabled:
            warnings.append("Une seule des deux exécutions a la détection des feux activée : "
                            "l'autre n'a produit aucune infraction feu rouge par construction.")
    return Comparison(metrics, config, warnings, overlap, class_counts)
