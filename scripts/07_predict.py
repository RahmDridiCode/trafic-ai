

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Les chemins par défaut sont ancrés à la racine du projet.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BEST_WEIGHTS = PROJECT_ROOT / "runs" / "yolov8n_baseline" / "weights" / "best.pt"
DEFAULT_SOURCE = PROJECT_ROOT /"dataset" / "test" / "my_images"
RUNS_DIR = PROJECT_ROOT / "runs"
PREDICTION_NAME = "predictions_1280"


def parse_arguments() -> argparse.Namespace:
    """Lit la source à prédire et les réglages de prédiction."""
    parser = argparse.ArgumentParser(
        description="Prédit les classes de véhicules sur une image ou un dossier."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Image ou dossier d'images (défaut : <projet>/test/my_images).",
    )
    parser.add_argument("--weights", type=Path, default=BEST_WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto (GPU si disponible, sinon CPU), cpu, ou index GPU comme 0.",
    )
    return parser.parse_args()


def choose_device(requested_device: str) -> str:
    """Choisit automatiquement un GPU disponible ou conserve le choix explicite."""
    if requested_device.lower() != "auto":
        return requested_device
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch est requis pour choisir automatiquement le device.") from exc
    return "0" if torch.cuda.is_available() else "cpu"


def predict(
    source: Path = DEFAULT_SOURCE,
    weights: Path = BEST_WEIGHTS,
    confidence: float = 0.25,
    image_size: int = 640,
    requested_device: str = "auto",
) -> int:
    """Enregistre les images annotées et les labels/classes avec confiance."""
    source = source.resolve()
    weights = weights.resolve()
    output_dir = RUNS_DIR / PREDICTION_NAME

    if not source.exists():
        print(f"ERREUR : source introuvable : {source}", file=sys.stderr)
        return 1
    if source.is_dir() and not any(path.is_file() for path in source.iterdir()):
        print(f"ERREUR : le dossier source ne contient aucun fichier : {source}", file=sys.stderr)
        return 1
    if not weights.is_file():
        print(f"ERREUR : poids YOLO introuvables : {weights}", file=sys.stderr)
        return 1
    if not 0 <= confidence <= 1:
        print("ERREUR : --conf doit être compris entre 0 et 1.", file=sys.stderr)
        return 1
    if image_size <= 0:
        print("ERREUR : --imgsz doit être strictement positif.", file=sys.stderr)
        return 1
    if output_dir.exists():
        print(
            f"ERREUR : le dossier de prédictions existe déjà : {output_dir}. "
            "Il ne sera pas écrasé; déplacez-le avant de relancer.",
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
            "ERREUR : ultralytics n'est pas installé. "
            "Installez-le avec `pip install ultralytics`.",
            file=sys.stderr,
        )
        return 1

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

    # Les images sauvegardées incluent les boîtes, noms de classe et scores.
    # Les fichiers labels contiennent l'ID de classe et le score (save_conf).
    print(f"Images traitées : {len(results)}")
    print(f"Images annotées et labels : {output_dir}")
    return 0


def main() -> int:
    """Point d'entrée du script de prédiction."""
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
