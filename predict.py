"""
Compare several YOLO models on the same image, image folder or video (detection only, no tracking).

Examples:
    python predict.py --model all --device cpu
    python predict.py --model yolov8m_1280 --device 0
    python predict.py --model all --source dataset/test/videos/traffic.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_SOURCE = PROJECT_ROOT / "dataset" / "test" / "videos" / "traffic.mp4"
DEFAULT_PROJECT = PROJECT_ROOT / "runs" / "traffic_video_prediction"

# Central model configuration: name -> weights, imgsz, conf.
MODELS = {
    "yolov8m_1280": {"weights": "models/yolov8m_1280_best.pt", "imgsz": 1280, "conf": 0.25},
    "yolov8s_1280": {"weights": "models/yolov8s_1280_best.pt", "imgsz": 1280, "conf": 0.25},
    "yolov10m_1280": {"weights": "models/yolov10m_1280_best.pt", "imgsz": 1280, "conf": 0.25},
}


def resolve_path(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (Path.cwd() / p if (Path.cwd() / p).exists() else PROJECT_ROOT / p)


def resolve_device(device: str) -> str:
    """'auto' -> GPU 0 if CUDA is available, else CPU. Otherwise the value is used as given."""
    device = str(device).strip().lower()
    if device == "auto":
        try:
            import torch
            return "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device


def select_models(selection) -> list[str]:
    names = []
    for item in selection:
        for name in item.split(","):
            name = name.strip()
            if not name:
                continue
            if name == "all":
                names.extend(MODELS)
            elif name in MODELS:
                names.append(name)
            else:
                raise ValueError(f"Unknown model '{name}'. Available: all, {', '.join(MODELS)}")
    unique = list(dict.fromkeys(names))
    if not unique:
        raise ValueError("No model selected")
    return unique


def validate_params(imgsz: int, conf: float) -> None:
    if imgsz <= 0 or imgsz % 32 != 0:
        raise ValueError(f"imgsz must be a positive multiple of 32 (got {imgsz})")
    if not 0.0 <= conf <= 1.0:
        raise ValueError(f"conf must be within [0, 1] (got {conf})")


def unique_run_name(project: Path, name: str) -> str:
    """Never reuse an existing run folder: yolov8m_1280, yolov8m_1280_2, yolov8m_1280_3, ..."""
    candidate, i = name, 2
    while (project / candidate).exists():
        candidate = f"{name}_{i}"
        i += 1
    return candidate


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", nargs="+", default=["all"],
                   help=f"Model name(s) or 'all'. Available: {', '.join(MODELS)}")
    p.add_argument("--source", default=str(DEFAULT_SOURCE), help="Image, folder of images or video")
    p.add_argument("--device", default="auto", help="'auto' (GPU if available else CPU), 'cpu', '0', '0,1', ...")
    p.add_argument("--imgsz", type=int, help="Override imgsz for all selected models")
    p.add_argument("--conf", type=float, help="Override conf for all selected models")
    p.add_argument("--project", default=str(DEFAULT_PROJECT), help="Output root directory")
    return p.parse_args(argv)


def run_model(name, cfg, source, device, project):
    from ultralytics import YOLO

    run_name = unique_run_name(project, name)
    print(f"\n=== {name}: weights={cfg['weights']} imgsz={cfg['imgsz']} conf={cfg['conf']} "
          f"device={device} -> {project / run_name}")
    model = YOLO(str(cfg["weights"]))
    # stream=True avoids keeping every video frame result in memory.
    n = 0
    for _ in model.predict(
        source=str(source),
        imgsz=cfg["imgsz"],
        conf=cfg["conf"],
        device=device,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(project),
        name=run_name,
        exist_ok=True,
        stream=True,
        verbose=False,
    ):
        n += 1
    print(f"    done: {n} frame(s)/image(s) processed")
    return project / run_name


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        names = select_models(args.model)
        configs = {}
        for name in names:
            cfg = dict(MODELS[name])
            if args.imgsz is not None:
                cfg["imgsz"] = args.imgsz
            if args.conf is not None:
                cfg["conf"] = args.conf
            validate_params(cfg["imgsz"], cfg["conf"])
            weights = resolve_path(cfg["weights"])
            if not weights.is_file():
                raise FileNotFoundError(f"Weights not found for '{name}': {weights}")
            cfg["weights"] = weights
            configs[name] = cfg
        source = resolve_path(args.source)
        if not source.exists():
            raise FileNotFoundError(f"Source not found: {source}")
    except (ValueError, FileNotFoundError) as exc:
        print(f"[predict] Error: {exc}", file=sys.stderr)
        return 1

    device = resolve_device(args.device)
    project = Path(args.project)
    if not project.is_absolute():
        project = Path.cwd() / project
    print(f"Source: {source}\nDevice: {device}\nModels: {', '.join(names)}")

    failures = 0
    for name, cfg in configs.items():
        try:
            run_model(name, cfg, source, device, project)
        except Exception as exc:
            failures += 1
            print(f"[predict] {name} failed: {exc}", file=sys.stderr)
    print("\nResults:", project)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
