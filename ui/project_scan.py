"""Discovery and validation of the project's real assets (videos, models, calibrations, ROI files, runs)."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STATE_DIR = PROJECT_ROOT / ".ui_state"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".flv", ".webm"}
SKIP_DIRS = {".venv", "venv", ".git", ".idea", "__pycache__", ".ui_state", "node_modules"}
JSON_SKIP_DIRS = SKIP_DIRS | {"dataset", "cleaned", "raw", "runs", "outputs", "reports"}
ROI_KEYS = ("VEHICLE_DETECT_ROI_POLYGON", "RED_LIGHT_DETECT_ROI_POLYGON", "STOP_LINE_COORDS")


def rel(path: str | Path) -> str:
    """Project-relative POSIX path when possible, otherwise the absolute path."""
    p = Path(path)
    try:
        return p.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(p)


def resolve_path(path: str | Path) -> Path:
    """Same lookup as the pipeline: as given, else relative to the project root."""
    p = Path(path)
    if p.is_absolute() or p.is_file() or p.is_dir():
        return p
    return PROJECT_ROOT / p


def _walk(root: Path, patterns: tuple[str, ...], skip: set[str]):
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir():
                if entry.name not in skip:
                    stack.append(entry)
            elif entry.suffix.lower() in patterns:
                yield entry


# ---------------------------------------------------------------- videos
@dataclass
class VideoInfo:
    path: str
    readable: bool = False
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frames: int = 0
    size_mb: float = 0.0
    error: str | None = None

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"

    @property
    def duration_s(self) -> float:
        return self.frames / self.fps if self.fps else 0.0


def scan_videos() -> list[str]:
    return sorted((rel(p) for p in _walk(PROJECT_ROOT, tuple(VIDEO_EXTENSIONS), SKIP_DIRS | {"outputs", "runs", "cleaned", "raw"})),
                  key=str.lower)


@lru_cache(maxsize=256)
def _video_info_cached(path: str, mtime: float) -> VideoInfo:
    import cv2

    info = VideoInfo(path=path)
    p = Path(path)
    info.size_mb = round(p.stat().st_size / 1e6, 2)
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            info.error = "OpenCV ne peut pas ouvrir ce fichier (codec ou conteneur non supporté)."
            return info
        info.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        info.fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        info.frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, _ = cap.read()
        info.readable = bool(ok and info.width > 0 and info.height > 0)
        if not info.readable:
            info.error = "Aucune image ne peut être lue dans cette vidéo."
    finally:
        cap.release()
    return info


def video_info(path: str | Path) -> VideoInfo:
    p = resolve_path(path)
    if not p.is_file():
        return VideoInfo(path=str(path), error="Fichier introuvable.")
    return _video_info_cached(str(p), p.stat().st_mtime)


# ---------------------------------------------------------------- JSON assets
def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


@dataclass
class JsonAsset:
    path: str
    kind: str  # "calibration" | "roi"
    resolution: str | None = None
    calibration_type: str | None = None
    camera: str | None = None
    notes: str = ""


def scan_json_assets() -> tuple[list[JsonAsset], list[JsonAsset]]:
    calibrations, rois = [], []
    for p in sorted(_walk(PROJECT_ROOT, (".json",), JSON_SKIP_DIRS), key=lambda x: str(x).lower()):
        data = _read_json(p)
        if not isinstance(data, dict):
            continue
        if "image_points" in data and "world_points" in data:
            calibrations.append(JsonAsset(
                rel(p), "calibration", str(data.get("resolution")) if data.get("resolution") else None,
                str(data.get("calibration_type", "unverified")).lower(), data.get("camera"),
                str(data.get("notes", ""))))
        elif any(k in data for k in ROI_KEYS):
            rois.append(JsonAsset(rel(p), "roi", str(data.get("resolution")) if data.get("resolution") else None))
    return calibrations, rois


@dataclass
class Compat:
    path: str
    ok: bool
    message: str
    mode: str | None = None  # calibration only: real | demo | disabled
    recommended: bool = False


def check_calibration(path: str, video: VideoInfo, allow_demo: bool = False) -> Compat:
    """Runs the project's own ``load_calibration`` so the verdict is exactly the pipeline's."""
    from src.main import load_calibration

    try:
        _, mode, status = load_calibration(path=path, width=video.width, height=video.height,
                                           force_uncalibrated=False, allow_demo=allow_demo)
        return Compat(path, True, status, mode)
    except ValueError as exc:
        return Compat(path, False, str(exc))


def check_roi(path: str, video: VideoInfo) -> Compat:
    from src.Feux.roi_config import load_roi_config

    try:
        load_roi_config(resolve_path(path), video.width, video.height)
        return Compat(path, True, "ROI valide pour cette vidéo")
    except ValueError as exc:
        return Compat(path, False, str(exc))


def compatible_calibrations(video: VideoInfo, assets: list[JsonAsset]) -> list[Compat]:
    results = []
    stem = Path(video.path).name.lower()
    for asset in assets:
        c = check_calibration(asset.path, video, allow_demo=True)
        if c.ok:
            c.mode = (asset.calibration_type or "unverified")
            c.message = f"type={asset.calibration_type}, résolution={asset.resolution}"
            c.recommended = bool(asset.camera and Path(str(asset.camera)).name.lower() == stem)
        results.append(c)
    return sorted(results, key=lambda c: (not c.ok, not c.recommended, c.mode != "real", c.path))


def compatible_rois(video: VideoInfo, assets: list[JsonAsset]) -> list[Compat]:
    results = [check_roi(a.path, video) for a in assets]
    for c, a in zip(results, assets):
        c.recommended = bool(c.ok and a.resolution)
    return sorted(results, key=lambda c: (not c.ok, not c.recommended, c.path))


# ---------------------------------------------------------------- models
@dataclass
class ModelInfo:
    path: str
    size_mb: float
    classes: dict[int, str] = field(default_factory=dict)
    task: str | None = None
    error: str | None = None
    analysed: bool = False
    train_imgsz: int | None = None

    def class_names(self) -> list[str]:
        return [str(n) for n in self.classes.values()]


def scan_models() -> list[str]:
    return sorted((rel(p) for p in _walk(PROJECT_ROOT, (".pt",), SKIP_DIRS)), key=str.lower)


def _model_cache() -> dict:
    return _read_json(STATE_DIR / "model_cache.json") or {}


def model_info(path: str, analyse: bool = False) -> ModelInfo:
    """Class names come from the checkpoint itself (cached by size+mtime); never guessed from the file name."""
    p = resolve_path(path)
    if not p.is_file():
        return ModelInfo(path=path, size_mb=0.0, error="Fichier introuvable.")
    stat = p.stat()
    info = ModelInfo(path=rel(p), size_mb=round(stat.st_size / 1e6, 2))
    key = rel(p)
    cache = _model_cache()
    entry = cache.get(key)
    if entry and entry.get("v") == 2 and entry.get("size") == stat.st_size and entry.get("mtime") == stat.st_mtime:
        info.classes = {int(k): v for k, v in entry.get("classes", {}).items()}
        info.task, info.error, info.analysed = entry.get("task"), entry.get("error"), True
        info.train_imgsz = entry.get("train_imgsz")
        return info
    if not analyse:
        return info
    try:
        from ultralytics import YOLO

        model = YOLO(str(p))
        names = model.names
        info.classes = {int(k): str(v) for k, v in (names.items() if isinstance(names, dict) else enumerate(names))}
        info.task = getattr(model, "task", None)
        imgsz = ((getattr(model, "ckpt", None) or {}).get("train_args") or {}).get("imgsz")
        info.train_imgsz = int(imgsz) if isinstance(imgsz, (int, float)) else None
    except Exception as exc:  # corrupted / unsupported checkpoint
        info.error = f"{type(exc).__name__}: {str(exc)[:300]}"
    info.analysed = True
    cache[key] = {"v": 2, "size": stat.st_size, "mtime": stat.st_mtime, "classes": info.classes,
                  "task": info.task, "train_imgsz": info.train_imgsz, "error": info.error}
    STATE_DIR.mkdir(exist_ok=True)
    (STATE_DIR / "model_cache.json").write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return info


def vehicle_names() -> set[str]:
    from src.detector import VEHICLE_NAMES

    return set(VEHICLE_NAMES)


def light_state_names() -> set[str]:
    from src.Feux.traffic_light_detector import STATE_ALIASES

    return set(STATE_ALIASES)


def is_vehicle_model(info: ModelInfo) -> bool:
    return bool(vehicle_names() & {n.lower() for n in info.class_names()})


def is_light_model(info: ModelInfo) -> bool:
    normalized = {n.strip().lower().replace(" ", "_").replace("-", "_") for n in info.class_names()}
    return bool(light_state_names() & normalized)


# ---------------------------------------------------------------- environment
@lru_cache(maxsize=1)
def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


# ---------------------------------------------------------------- runs
def scan_runs(output_root: str | Path = "outputs") -> list[Path]:
    """Output folders holding a traffic_summary.json (newest first)."""
    root = resolve_path(output_root)
    if not root.is_dir():
        return []
    runs = [d for d in root.iterdir() if d.is_dir() and (d / "traffic_summary.json").is_file()]
    return sorted(runs, key=lambda d: (d / "traffic_summary.json").stat().st_mtime, reverse=True)

