
"""Train YOLOv8m at 1280 px using train/val only, leaving test untouched."""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_YAML = PROJECT_ROOT / "data.yaml"
RUNS_DIR = PROJECT_ROOT / "runs"

# YOLOv8m
RUN_NAME = "yolov8m_1280"
MODEL_NAME = "yolov8m.pt"
COPIED_WEIGHTS_NAME = "yolov8m_1280_best.pt"

# Training parameters
EPOCHS = 100
IMAGE_SIZE = 1280
BATCH_SIZE = 8
PATIENCE = 20
RANDOM_SEED = 42

# Classes
CLASS_NAMES = {
    0: "bus",
    1: "car",
    2: "motorcycle",
    3: "truck",
}

IMAGE_EXTENSIONS = {
    ".bmp",
    ".jfif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8m at 1280 px using train/val only."
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_YAML,
        help="Path to data.yaml.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=IMAGE_SIZE,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=PATIENCE,
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, or GPU index such as 0.",
    )

    return parser.parse_args()


# ============================================================
# LOAD DATASET CONFIGURATION
# ============================================================

def load_dataset_config(
    data_yaml: Path,
) -> tuple[Path, dict[str, Any]]:

    data_yaml = data_yaml.resolve()

    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"Fichier data.yaml introuvable : {data_yaml}"
        )

    with data_yaml.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError(
            "Le fichier YAML doit contenir un dictionnaire."
        )

    configured_root = Path(
        str(config.get("path", "."))
    )

    if configured_root.is_absolute():
        dataset_root = configured_root.resolve()
    else:
        dataset_root = (
            data_yaml.parent / configured_root
        ).resolve()

    return dataset_root, config


# ============================================================
# VALIDATE TRAIN / VAL / TEST
# ============================================================

def count_and_validate_split(
    dataset_root: Path,
    config: dict[str, Any],
    split: str,
) -> int:

    image_value = config.get(split)

    if not isinstance(image_value, str):
        raise ValueError(
            f"Le chemin du split '{split}' manque dans data.yaml."
        )

    image_dir = Path(image_value)

    if not image_dir.is_absolute():
        image_dir = (
            dataset_root / image_dir
        ).resolve()

    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Dossier '{split}' introuvable : {image_dir}"
        )

    label_dir = image_dir.parent / "labels"

    if not label_dir.is_dir():
        raise FileNotFoundError(
            f"Dossier de labels '{split}' introuvable : {label_dir}"
        )

    images = {
        path.stem.casefold(): path
        for path in image_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    }

    labels = {
        path.stem.casefold(): path
        for path in label_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() == ".txt"
        )
    }

    if not images:
        raise ValueError(
            f"Aucune image trouvée dans le split '{split}'."
        )

    if images.keys() != labels.keys():

        missing_labels = sorted(
            images.keys() - labels.keys()
        )

        orphan_labels = sorted(
            labels.keys() - images.keys()
        )

        raise ValueError(
            f"Paires incorrectes dans '{split}'. "
            f"Images sans label : {missing_labels[:10]}. "
            f"Labels sans image : {orphan_labels[:10]}."
        )

    class_ids = set(CLASS_NAMES)

    for label_path in labels.values():

        try:
            lines = label_path.read_text(
                encoding="utf-8"
            ).splitlines()

        except (OSError, UnicodeError) as exc:

            raise ValueError(
                f"Lecture impossible pour {label_path}: {exc}"
            ) from exc

        for line_number, line in enumerate(
            lines,
            start=1,
        ):

            fields = line.split()

            if len(fields) != 5:
                raise ValueError(
                    f"{label_path.name}, ligne {line_number}: "
                    "5 valeurs attendues."
                )

            try:

                class_id = int(fields[0])

                coordinates = [
                    float(value)
                    for value in fields[1:]
                ]

            except ValueError as exc:

                raise ValueError(
                    f"{label_path.name}, ligne {line_number}: "
                    "valeurs non numériques."
                ) from exc

            if class_id not in class_ids:

                raise ValueError(
                    f"{label_path.name}, ligne {line_number}: "
                    f"class_id {class_id} invalide."
                )

            if any(
                not math.isfinite(value)
                or not 0 <= value <= 1
                for value in coordinates
            ):

                raise ValueError(
                    f"{label_path.name}, ligne {line_number}: "
                    "coordonnées hors de [0,1]."
                )

            if (
                coordinates[2] <= 0
                or coordinates[3] <= 0
            ):

                raise ValueError(
                    f"{label_path.name}, ligne {line_number}: "
                    "width/height doivent être positives."
                )

    return len(images)


# ============================================================
# CHECK SPLIT OVERLAP
# ============================================================

def ensure_splits_are_disjoint(
    dataset_root: Path,
    config: dict[str, Any],
) -> None:

    stems_by_split: dict[str, set[str]] = {}

    for split in (
        "train",
        "val",
        "test",
    ):

        image_value = config.get(split)

        if not isinstance(image_value, str):
            raise ValueError(
                f"Le chemin du split '{split}' manque."
            )

        image_dir = Path(image_value)

        if not image_dir.is_absolute():
            image_dir = (
                dataset_root / image_dir
            ).resolve()

        stems_by_split[split] = {
            path.stem.casefold()
            for path in image_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        }

    overlap = (
        (
            stems_by_split["train"]
            & stems_by_split["val"]
        )
        |
        (
            stems_by_split["train"]
            & stems_by_split["test"]
        )
        |
        (
            stems_by_split["val"]
            & stems_by_split["test"]
        )
    )

    if overlap:

        raise ValueError(
            "Les splits se chevauchent. "
            f"Exemples : {sorted(overlap)[:10]}"
        )


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
            "PyTorch n'est pas installé."
        ) from exc

    if torch.cuda.is_available():
        return "0"

    return "cpu"


# ============================================================
# COPY BEST MODEL
# ============================================================

def copy_best_weights(
    source: Path,
    destination: Path,
) -> None:

    if not source.is_file():

        raise FileNotFoundError(
            f"best.pt introuvable : {source}"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists():
        destination.unlink()

    shutil.copy2(
        source,
        destination,
    )


# ============================================================
# TRAINING
# ============================================================

def train_yolov8m(
    data_yaml: Path = DATA_YAML,
    epochs: int = EPOCHS,
    imgsz: int = IMAGE_SIZE,
    batch: int = BATCH_SIZE,
    patience: int = PATIENCE,
    requested_device: str = "auto",
) -> int:

    try:

        dataset_root, config = (
            load_dataset_config(data_yaml)
        )

        train_count = count_and_validate_split(
            dataset_root,
            config,
            "train",
        )

        val_count = count_and_validate_split(
            dataset_root,
            config,
            "val",
        )

        test_count = count_and_validate_split(
            dataset_root,
            config,
            "test",
        )

        ensure_splits_are_disjoint(
            dataset_root,
            config,
        )

        names = config.get("names")

        normalized_names = (
            {
                int(key): str(value)
                for key, value in names.items()
            }
            if isinstance(names, dict)
            else {}
        )

        if normalized_names != CLASS_NAMES:

            raise ValueError(
                f"Classes incorrectes dans data.yaml.\n"
                f"Attendu : {CLASS_NAMES}\n"
                f"Actuel : {normalized_names}"
            )

        if (
            epochs <= 0
            or imgsz <= 0
            or batch <= 0
            or patience < 0
        ):

            raise ValueError(
                "epochs, imgsz et batch doivent être "
                "positifs et patience >= 0."
            )

        device = choose_device(
            requested_device
        )

    except (
        OSError,
        ValueError,
        RuntimeError,
        yaml.YAMLError,
    ) as exc:

        print(
            f"ERREUR : {exc}",
            file=sys.stderr,
        )

        return 1


    # ========================================================
    # PATHS
    # ========================================================

    run_dir = (
        RUNS_DIR / RUN_NAME
    )

    best_weights = (
        run_dir
        / "weights"
        / "best.pt"
    )

    copied_weights = (
        PROJECT_ROOT
        / "models"
        / COPIED_WEIGHTS_NAME
    )


    # ========================================================
    # PREVENT ACCIDENTAL OVERWRITE
    # ========================================================

    if run_dir.exists():

        print(
            f"ERREUR : le dossier existe déjà : {run_dir}",
            file=sys.stderr,
        )

        return 1


    if copied_weights.exists():

        print(
            f"ERREUR : le modèle existe déjà : "
            f"{copied_weights}",
            file=sys.stderr,
        )

        return 1


    # ========================================================
    # DISPLAY CONFIGURATION
    # ========================================================

    print("=" * 60)

    print(
        f"Images train : {train_count}"
    )

    print(
        f"Images val : {val_count}"
    )

    print(
        f"Images test : {test_count} "
        "(NON utilisées pour l'entraînement)"
    )

    print(
        f"Classes : {normalized_names}"
    )

    print(
        f"Modèle : {MODEL_NAME}"
    )

    print(
        f"Epochs : {epochs}"
    )

    print(
        f"Image size : {imgsz}"
    )

    print(
        f"Batch : {batch}"
    )

    print(
        f"Patience : {patience}"
    )

    print(
        f"Device : {device}"
    )

    print("=" * 60)


    # ========================================================
    # ULTRALYTICS
    # ========================================================

    try:

        from ultralytics import YOLO

    except ImportError:

        print(
            "ERREUR : ultralytics n'est pas installé.\n"
            "Lance : pip install ultralytics",
            file=sys.stderr,
        )

        return 1


    # ========================================================
    # LOAD YOLOv8m
    # ========================================================

    print(
        "\nChargement de YOLOv8m..."
    )

    model = YOLO(
        MODEL_NAME
    )


    # ========================================================
    # TRAIN
    # ========================================================

    print(
        "\nDébut de l'entraînement..."
    )

    model.train(

        data=str(
            data_yaml.resolve()
        ),

        epochs=epochs,

        imgsz=imgsz,

        batch=batch,

        patience=patience,

        project=str(
            RUNS_DIR
        ),

        name=RUN_NAME,

        device=device,

        seed=RANDOM_SEED,

        exist_ok=False,
    )


    # ========================================================
    # COPY BEST MODEL
    # ========================================================

    copy_best_weights(
        best_weights,
        copied_weights,
    )


    print("\n" + "=" * 60)

    print(
        "ENTRAÎNEMENT TERMINÉ ✅"
    )

    print(
        f"Best model : {best_weights}"
    )

    print(
        f"Copied model : {copied_weights}"
    )

    print("=" * 60)


    return 0


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    arguments = parse_arguments()

    return train_yolov8m(

        data_yaml=arguments.data,

        epochs=arguments.epochs,

        imgsz=arguments.imgsz,

        batch=arguments.batch,

        patience=arguments.patience,

        requested_device=arguments.device,
    )


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
