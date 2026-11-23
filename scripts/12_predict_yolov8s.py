from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BEST_WEIGHTS = PROJECT_ROOT / "models" / "yolov8s_1280_best.pt"

DEFAULT_SOURCE = PROJECT_ROOT / "dataset" / "test" / "my_images"

RUNS_DIR = PROJECT_ROOT / "runs"
PREDICTION_NAME = "yolov8s_1280_predictions2"


def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Prédit les véhicules avec YOLOv8s 1280."
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Image ou dossier d'images à prédire.",
    )

    parser.add_argument(
        "--weights",
        type=Path,
        default=BEST_WEIGHTS,
        help="Fichier .pt du modèle entraîné.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Seuil de confiance.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Taille des images.",
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu ou GPU comme 0.",
    )

    return parser.parse_args()


def choose_device(requested_device: str) -> str:

    if requested_device.lower() != "auto":
        return requested_device

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch est requis pour détecter automatiquement le GPU."
        ) from exc

    return "0" if torch.cuda.is_available() else "cpu"


def predict(
    source: Path = DEFAULT_SOURCE,
    weights: Path = BEST_WEIGHTS,
    confidence: float = 0.15,
    image_size: int = 1280,
    requested_device: str = "auto",
) -> int:

    source = source.resolve()
    weights = weights.resolve()
    output_dir = RUNS_DIR / PREDICTION_NAME

    if not source.exists():
        print(
            f"ERREUR : source introuvable : {source}",
            file=sys.stderr,
        )
        return 1

    if not weights.is_file():
        print(
            f"ERREUR : modèle introuvable : {weights}",
            file=sys.stderr,
        )
        return 1

    if not 0 <= confidence <= 1:
        print(
            "ERREUR : --conf doit être compris entre 0 et 1.",
            file=sys.stderr,
        )
        return 1

    if image_size <= 0:
        print(
            "ERREUR : --imgsz doit être positif.",
            file=sys.stderr,
        )
        return 1

    if output_dir.exists():
        print(
            f"ERREUR : le dossier existe déjà : {output_dir}",
            file=sys.stderr,
        )
        print(
            "Supprime-le ou change PREDICTION_NAME.",
            file=sys.stderr,
        )
        return 1

    try:
        device = choose_device(requested_device)
    except RuntimeError as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "ERREUR : ultralytics n'est pas installé.",
            file=sys.stderr,
        )
        print(
            "Utilise : pip install ultralytics",
            file=sys.stderr,
        )
        return 1

    print("=" * 60)
    print("YOLOv8s - PREDICTION")
    print("=" * 60)

    print(f"Modèle  : {weights}")
    print(f"Source  : {source}")
    print(f"ImgSize : {image_size}")
    print(f"Conf    : {confidence}")
    print(f"Device  : {device}")

    print("=" * 60)

    model = YOLO(str(weights))

    results = model.predict(
        source=str(source),
        conf=confidence,
        imgsz=image_size,
        device=device,
        project=str(RUNS_DIR),
        name=PREDICTION_NAME,
        exist_ok=False,
        save=True,
        save_txt=True,
        save_conf=True,
        verbose=True,
    )

    print()
    print("=" * 60)
    print("PREDICTION TERMINÉE")
    print("=" * 60)

    print(f"Images traitées : {len(results)}")
    print(f"Résultats       : {output_dir}")
    print(f"Labels YOLO     : {output_dir / 'labels'}")

    return 0


def main() -> int:

    arguments = parse_arguments()

    return predict(
        source=arguments.source,
        weights=arguments.weights,
        confidence=arguments.conf,
        image_size=arguments.imgsz,
        requested_device=arguments.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())