"""Evaluate YOLO weights on validation and/or test and save metrics to reports/."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml



PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "data.yaml"
DEFAULT_WEIGHTS = (
    PROJECT_ROOT / "runs" / "yolov8n_baseline" / "weights" / "best.pt"
)
REPORTS_DIR = PROJECT_ROOT / "reports"
CLASS_NAMES = {0: "bus", 1: "car", 2: "motorcycle", 3: "truck"}


def parse_arguments() -> argparse.Namespace:
    """Parse model, dataset, split, and execution device options."""
    parser = argparse.ArgumentParser(
        description="Évalue un modèle YOLO sur validation et/ou test."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Fichier .pt (défaut : best.pt du baseline YOLOv8n).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_YAML,
        help="Fichier data.yaml du projet.",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test", "both"),
        default="both",
        help="Ensemble à évaluer (défaut : val et test séparément).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, ou index GPU tel que 0.",
    )
    return parser.parse_args()


def choose_device(requested_device: str) -> str:
    """Choose GPU 0 when available, otherwise CPU, unless explicitly requested."""
    if requested_device.lower() != "auto":
        return requested_device

    import torch

    return "0" if torch.cuda.is_available() else "cpu"


def validate_data_yaml(data_yaml: Path, splits: tuple[str, ...]) -> None:
    """Check YAML classes and image/label directories for requested splits."""
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Fichier data.yaml introuvable : {data_yaml}")

    with data_yaml.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration YAML invalide : {data_yaml}")

    configured_root = Path(str(config.get("path", ".")))
    dataset_root = (
        configured_root.resolve()
        if configured_root.is_absolute()
        else (data_yaml.parent / configured_root).resolve()
    )

    raw_names = config.get("names")
    names = (
        {int(class_id): str(name) for class_id, name in raw_names.items()}
        if isinstance(raw_names, dict)
        else {}
    )
    if names != CLASS_NAMES:
        raise ValueError(
            f"Classes attendues : {CLASS_NAMES}; classes du YAML : {names}."
        )

    for split in splits:
        relative_images = config.get(split)
        if not isinstance(relative_images, str):
            raise ValueError(f"Chemin du split '{split}' absent de {data_yaml}.")
        image_dir = Path(relative_images)
        if not image_dir.is_absolute():
            image_dir = (dataset_root / image_dir).resolve()
        label_dir = image_dir.parent / "labels"
        if not image_dir.is_dir():
            raise FileNotFoundError(
                f"Dossier d'images du split '{split}' introuvable : {image_dir}"
            )
        if not label_dir.is_dir():
            raise FileNotFoundError(
                f"Dossier de labels du split '{split}' introuvable : {label_dir}"
            )


def class_metric(metric: Any, attribute: str, class_id: int) -> float | None:
    """Get one per-class metric from Ultralytics, or None if the class is absent."""
    values = getattr(metric, attribute, [])
    class_indices = getattr(metric, "ap_class_index", [])
    for index, found_class_id in enumerate(class_indices):
        if int(found_class_id) == class_id and index < len(values):
            return float(values[index])
    return None


def metrics_to_rows(split: str, metrics: Any) -> list[dict[str, Any]]:
    """Convert the Ultralytics result object into global and class-level CSV rows."""
    box_metrics = metrics.box
    rows: list[dict[str, Any]] = [
        {
            "split": split,
            "class_id": "",
            "class_name": "all",
            "precision": float(box_metrics.mp),
            "recall": float(box_metrics.mr),
            "map50": float(box_metrics.map50),
            "map50_95": float(box_metrics.map),
        }
    ]

    for class_id, class_name in CLASS_NAMES.items():
        rows.append(
            {
                "split": split,
                "class_id": class_id,
                "class_name": class_name,
                "precision": class_metric(box_metrics, "p", class_id),
                "recall": class_metric(box_metrics, "r", class_id),
                "map50": class_metric(box_metrics, "ap50", class_id),
                "map50_95": class_metric(box_metrics, "ap", class_id),
            }
        )
    return rows


def evaluate(
    weights: Path,
    data_yaml: Path = DATA_YAML,
    split_option: str = "both",
    requested_device: str = "auto",
) -> int:
    """Run separate Ultralytics evaluations and write CSV/plots to a new folder."""
    weights = weights.resolve()
    data_yaml = data_yaml.resolve()
    reports_root = REPORTS_DIR.resolve()
    splits = ("val", "test") if split_option == "both" else (split_option,)

    if not weights.is_file():
        raise FileNotFoundError(f"Poids du modèle introuvables : {weights}")
    validate_data_yaml(data_yaml, splits)
    device = choose_device(requested_device)

    # Create a unique directory so previous reports and confusion matrices remain.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = reports_root / f"evaluation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    from ultralytics import YOLO

    model = YOLO(str(weights))
    rows: list[dict[str, Any]] = []

    for split in splits:
        # Each split is evaluated independently; training is never called here.
        results = model.val(
            data=str(data_yaml),
            split=split,
            plots=True,
            project=str(output_dir),
            name=split,
            exist_ok=False,
            device=device,
            verbose=True,
        )
        split_rows = metrics_to_rows(split, results)
        rows.extend(split_rows)

        overall = split_rows[0]
        print(
            f"{split.upper()} — Precision={overall['precision']:.4f}, "
            f"Recall={overall['recall']:.4f}, "
            f"mAP50={overall['map50']:.4f}, "
            f"mAP50-95={overall['map50_95']:.4f}"
        )
        for class_row in split_rows[1:]:
            print(
                f"  {class_row['class_name']}: "
                f"P={class_row['precision']}, R={class_row['recall']}, "
                f"mAP50={class_row['map50']}, "
                f"mAP50-95={class_row['map50_95']}"
            )

    csv_path = output_dir / "metrics.csv"
    fieldnames = [
        "split",
        "class_id",
        "class_name",
        "precision",
        "recall",
        "map50",
        "map50_95",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV des métriques : {csv_path}")
    print(f"Résultats YOLO et matrices de confusion : {output_dir}")
    return 0


def main() -> int:
    """Command-line entry point."""
    arguments = parse_arguments()
    return evaluate(
        weights=arguments.weights,
        data_yaml=arguments.data,
        split_option=arguments.split,
        requested_device=arguments.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())