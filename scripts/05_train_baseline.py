"""Train a YOLOv8n baseline using train/val only, leaving test untouched."""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml


# Les chemins par défaut sont calculés depuis la racine du projet.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "data.yaml"
RUNS_DIR = PROJECT_ROOT / "runs"
RUN_NAME = "yolov8n_baseline"
MODEL_NAME = "yolov8n.pt"
EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 16
PATIENCE = 10
RANDOM_SEED = 42
CLASS_NAMES = {0: "bus", 1: "car", 2: "motorcycle", 3: "truck"}
IMAGE_EXTENSIONS = {".bmp", ".jfif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_arguments() -> argparse.Namespace:
    """Lit les réglages d'entraînement, avec des valeurs raisonnables par défaut."""
    parser = argparse.ArgumentParser(description="Entraîne un modèle baseline YOLOv8n.")
    parser.add_argument("--data", type=Path, default=DATA_YAML, help="Fichier data.yaml.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto (GPU si disponible, sinon CPU), cpu, ou index GPU comme 0.",
    )
    return parser.parse_args()


def load_dataset_config(data_yaml: Path) -> tuple[Path, dict[str, Any]]:
    """Charge data.yaml et résout les dossiers déclarés."""
    data_yaml = data_yaml.resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Fichier de configuration introuvable : {data_yaml}")

    with data_yaml.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Le fichier YAML doit contenir un dictionnaire : {data_yaml}")

    configured_root = Path(str(config.get("path", ".")))
    dataset_root = (
        configured_root.resolve()
        if configured_root.is_absolute()
        else (data_yaml.parent / configured_root).resolve()
    )
    return dataset_root, config


def count_and_validate_split(
    dataset_root: Path,
    config: dict[str, Any],
    split: str,
) -> int:
    """Vérifie chaque image et son label YOLO, puis renvoie le nombre d'images."""
    image_value = config.get(split)
    if not isinstance(image_value, str):
        raise ValueError(f"Le chemin du split '{split}' manque dans data.yaml.")

    image_dir = Path(image_value)
    if not image_dir.is_absolute():
        image_dir = (dataset_root / image_dir).resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Dossier d'images '{split}' introuvable : {image_dir}")

    label_dir = image_dir.parent / "labels"
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Dossier de labels '{split}' introuvable : {label_dir}")

    images = {
        path.stem.casefold(): path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    labels = {
        path.stem.casefold(): path
        for path in label_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt"
    }
    if not images:
        raise ValueError(f"Aucune image prise en charge dans le split '{split}'.")
    if images.keys() != labels.keys():
        missing_labels = sorted(images.keys() - labels.keys())
        orphan_labels = sorted(labels.keys() - images.keys())
        raise ValueError(
            f"Paires incorrectes dans '{split}'; images sans label : "
            f"{missing_labels[:10]}, labels sans image : {orphan_labels[:10]}."
        )

    class_ids = set(CLASS_NAMES)
    for label_path in labels.values():
        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"Lecture impossible pour {label_path}: {exc}") from exc

        for line_number, line in enumerate(lines, start=1):
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(
                    f"{label_path.name}, ligne {line_number}: 5 valeurs attendues, "
                    f"{len(fields)} trouvée(s)."
                )
            try:
                class_id = int(fields[0])
                coordinates = [float(value) for value in fields[1:]]
            except ValueError as exc:
                raise ValueError(
                    f"{label_path.name}, ligne {line_number}: valeurs non numériques."
                ) from exc
            if class_id not in class_ids:
                raise ValueError(
                    f"{label_path.name}, ligne {line_number}: class_id {class_id} "
                    "n'appartient pas aux classes 0, 1, 2, 3."
                )
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in coordinates):
                raise ValueError(
                    f"{label_path.name}, ligne {line_number}: coordonnées hors de [0, 1]."
                )
            if coordinates[2] <= 0 or coordinates[3] <= 0:
                raise ValueError(
                    f"{label_path.name}, ligne {line_number}: width/height "
                    "doivent être strictement positives."
                )

    return len(images)


def ensure_splits_are_disjoint(
    dataset_root: Path,
    config: dict[str, Any],
) -> None:
    """Refuse tout nom d'image présent dans plusieurs splits."""
    stems_by_split: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        image_value = config.get(split)
        if not isinstance(image_value, str):
            raise ValueError(f"Le chemin du split '{split}' manque dans data.yaml.")
        image_dir = Path(image_value)
        if not image_dir.is_absolute():
            image_dir = (dataset_root / image_dir).resolve()
        stems_by_split[split] = {
            path.stem.casefold()
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }

    overlap = (
        stems_by_split["train"] & stems_by_split["val"]
        | stems_by_split["train"] & stems_by_split["test"]
        | stems_by_split["val"] & stems_by_split["test"]
    )
    if overlap:
        raise ValueError(
            "Les splits se chevauchent; aucun entraînement lancé. "
            f"Exemples : {', '.join(sorted(overlap)[:10])}"
        )


def choose_device(requested_device: str) -> str:
    """Choisit automatiquement un GPU disponible ou conserve le choix explicite."""
    if requested_device.lower() != "auto":
        return requested_device
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch n'est pas installé dans l'environnement courant. "
            "Installez les dépendances du projet avant l'entraînement."
        ) from exc
    return "0" if torch.cuda.is_available() else "cpu"


def copy_best_weights(source: Path, destination: Path) -> None:
    """Copie les meilleurs poids sans jamais remplacer un modèle déjà présent."""
    if not source.is_file():
        raise FileNotFoundError(f"Le meilleur modèle n'a pas été créé : {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Le mode exclusif empêche d'écraser un fichier créé en parallèle.
    with source.open("rb") as source_file, destination.open("xb") as destination_file:
        shutil.copyfileobj(source_file, destination_file)
    shutil.copystat(source, destination)


def train_baseline(
    data_yaml: Path = DATA_YAML,
    epochs: int = EPOCHS,
    imgsz: int = IMAGE_SIZE,
    batch: int = BATCH_SIZE,
    patience: int = PATIENCE,
    requested_device: str = "auto",
) -> int:
    """Vérifie train/val/test, entraîne sur train/val, puis copie best.pt."""
    try:
        dataset_root, config = load_dataset_config(data_yaml)
        train_count = count_and_validate_split(dataset_root, config, "train")
        val_count = count_and_validate_split(dataset_root, config, "val")
        test_count = count_and_validate_split(dataset_root, config, "test")
        ensure_splits_are_disjoint(dataset_root, config)
        names = config.get("names")
        normalized_names = (
            {int(key): str(value) for key, value in names.items()}
            if isinstance(names, dict)
            else {}
        )
        if normalized_names != CLASS_NAMES:
            raise ValueError(
                f"Les noms des classes de data.yaml doivent être {CLASS_NAMES}; "
                f"valeur actuelle : {normalized_names}."
            )
        if epochs <= 0 or imgsz <= 0 or batch <= 0 or patience < 0:
            raise ValueError("epochs, imgsz et batch doivent être positifs; patience >= 0.")
        device = choose_device(requested_device)
    except (OSError, ValueError, RuntimeError, yaml.YAMLError) as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        return 1

    run_dir = RUNS_DIR / RUN_NAME
    best_weights = run_dir / "weights" / "best.pt"
    copied_weights = PROJECT_ROOT / "models" / "baseline_yolov8n_best.pt"
    if run_dir.exists():
        print(
            f"ERREUR : le dossier d'entraînement existe déjà : {run_dir}. "
            "Il ne sera pas écrasé.",
            file=sys.stderr,
        )
        return 1
    if copied_weights.exists():
        print(
            f"ERREUR : les poids de destination existent déjà : {copied_weights}. "
            "Ils ne seront pas remplacés.",
            file=sys.stderr,
        )
        return 1

    # Afficher les paramètres et les effectifs avant de lancer un calcul long.
    print(f"Images train : {train_count}")
    print(f"Images val : {val_count}")
    print(f"Images test (non utilisées pour l'entraînement) : {test_count}")
    print(f"Nombre de classes : {len(normalized_names)}")
    print(f"Noms des classes : {normalized_names}")
    print(f"Modèle : {MODEL_NAME}")
    print(f"Epochs : {epochs}")
    print(f"Taille des images : {imgsz}")
    print(f"Batch : {batch}")
    print(f"Patience : {patience}")
    print(f"Device : {device}")

    # Charger Ultralytics seulement lors de l'entraînement pour que les
    # vérifications de configuration restent possibles sans importer YOLO.
    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "ERREUR : ultralytics n'est pas installé dans cet environnement. "
            "Installez-le avec `pip install ultralytics` puis relancez le script.",
            file=sys.stderr,
        )
        return 1

    model = YOLO(MODEL_NAME)
    model.train(
        data=str(data_yaml.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        project=str(RUNS_DIR),
        name=RUN_NAME,
        device=device,
        seed=RANDOM_SEED,
        exist_ok=False,
    )
    copy_best_weights(best_weights, copied_weights)

    print(f"Meilleurs poids : {best_weights}")
    print(f"Copie du modèle : {copied_weights}")
    return 0


def main() -> int:
    """Point d'entrée du script d'entraînement."""
    arguments = parse_arguments()
    return train_baseline(
        data_yaml=arguments.data,
        epochs=arguments.epochs,
        imgsz=arguments.imgsz,
        batch=arguments.batch,
        patience=arguments.patience,
        requested_device=arguments.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
