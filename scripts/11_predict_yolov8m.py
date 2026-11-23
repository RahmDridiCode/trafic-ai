

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BEST_WEIGHTS = (
    PROJECT_ROOT
    / "models"
    / "yolov8m_1280_best.pt"
)

RUNS_DIR = PROJECT_ROOT / "runs"

PREDICTION_NAME = "predictions_yolov8m_1280"

DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "dataset"
    / "test"
    / "my_images"
)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Test YOLOv8m on an image, directory, "
            "or video."
        )
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=(
            "Image, dossier d'images ou vidéo "
            "à tester."
        ),
    )

    parser.add_argument(
        "--weights",
        type=Path,
        default=BEST_WEIGHTS,
        help="Chemin vers le modèle YOLOv8m.",
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
        help=(
            "auto, cpu ou index GPU "
            "(exemple : 0)."
        ),
    )

    return parser.parse_args()


# ============================================================
# DEVICE
# ============================================================

def choose_device(
    requested_device: str,
) -> str:

    if requested_device.lower() != "auto":
        return requested_device

    try:
        import torch

    except ImportError as exc:

        raise RuntimeError(
            "PyTorch est requis."
        ) from exc

    if torch.cuda.is_available():
        return "0"

    return "cpu"


# ============================================================
# PREDICTION
# ============================================================

def predict(
    source: Path,
    weights: Path,
    confidence: float,
    image_size: int,
    requested_device: str,
) -> int:

    source = source.resolve()
    weights = weights.resolve()

    output_dir = (
        RUNS_DIR
        / PREDICTION_NAME
    )

    # --------------------------------------------------------
    # CHECK SOURCE
    # --------------------------------------------------------

    if not source.exists():

        print(
            f"ERREUR : source introuvable : {source}",
            file=sys.stderr,
        )

        return 1


    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if not weights.is_file():

        print(
            f"ERREUR : modèle introuvable : {weights}",
            file=sys.stderr,
        )

        return 1


    # --------------------------------------------------------
    # CHECK CONFIDENCE
    # --------------------------------------------------------

    if not 0 <= confidence <= 1:

        print(
            "ERREUR : conf doit être compris entre 0 et 1.",
            file=sys.stderr,
        )

        return 1


    # --------------------------------------------------------
    # CHECK IMAGE SIZE
    # --------------------------------------------------------

    if image_size <= 0:

        print(
            "ERREUR : imgsz doit être positif.",
            file=sys.stderr,
        )

        return 1


    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    if output_dir.exists():

        print(
            f"ERREUR : le dossier existe déjà : "
            f"{output_dir}\n"
            "Supprime-le ou change PREDICTION_NAME.",
            file=sys.stderr,
        )

        return 1


    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    try:

        device = choose_device(
            requested_device
        )

    except RuntimeError as exc:

        print(
            f"ERREUR : {exc}",
            file=sys.stderr,
        )

        return 1


    # --------------------------------------------------------
    # ULTRALYTICS
    # --------------------------------------------------------

    try:

        from ultralytics import YOLO

    except ImportError:

        print(
            "ERREUR : ultralytics n'est pas installé.\n"
            "Lance : pip install ultralytics",
            file=sys.stderr,
        )

        return 1


    # --------------------------------------------------------
    # DISPLAY CONFIGURATION
    # --------------------------------------------------------

    print("=" * 60)

    print(
        "YOLOv8m PREDICTION"
    )

    print(
        f"Model      : {weights}"
    )

    print(
        f"Source     : {source}"
    )

    print(
        f"Confidence : {confidence}"
    )

    print(
        f"Image size : {image_size}"
    )

    print(
        f"Device     : {device}"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = YOLO(
        str(weights)
    )


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    results = model.predict(

        source=str(source),

        conf=confidence,

        imgsz=image_size,

        device=device,

        project=str(
            RUNS_DIR
        ),

        name=PREDICTION_NAME,

        exist_ok=False,

        save=True,

        save_txt=True,

        save_conf=True,

        verbose=True,
    )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 60)

    print(
        "PREDICTION TERMINÉE ✅"
    )

    print(
        f"Images/frames traités : {len(results)}"
    )

    print(
        f"Résultats : {output_dir}"
    )

    print("=" * 60)

    return 0


# ============================================================
# MAIN
# ============================================================

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

    raise SystemExit(
        main()
    )