"""Validate YOLO detection labels in a cleaned image/label dataset."""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path


# Les chemins sont ancrés à la racine du projet, quel que soit le dossier courant.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "cleaned"
REPORT_PATH = PROJECT_ROOT / "annotation_report.txt"
CLASS_IDS = {0, 1, 2, 3}
SUPPORTED_IMAGE_EXTENSIONS = {
    ".bmp",
    ".jfif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def is_inside(path: Path, parent: Path) -> bool:
    """Indique si path est parent ou l'un de ses sous-dossiers."""
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def parse_arguments() -> argparse.Namespace:
    """Lit les chemins du dataset et du rapport depuis la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Vérifie le format des annotations YOLO dans cleaned/."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help="Dossier contenant images/ et labels/ (défaut : <projet>/cleaned).",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPORT_PATH,
        help="Chemin du rapport (défaut : <projet>/annotation_report.txt).",
    )
    return parser.parse_args()


def validate_label_file(label_path: Path) -> tuple[int, list[str]]:
    """Vérifie chaque ligne d'un fichier, en gardant la ligne dans les erreurs."""
    errors: list[str] = []
    valid_objects = 0

    try:
        contents = label_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return 0, [f"{label_path.name}: lecture impossible ({exc})"]

    # Un fichier vide représente valablement une image sans objet annoté.
    for line_number, line in enumerate(contents.splitlines(), start=1):
        values = line.split()
        if len(values) != 5:
            errors.append(
                f"{label_path.name}, ligne {line_number}: "
                f"5 valeurs attendues, {len(values)} trouvée(s)."
            )
            continue

        try:
            class_id = int(values[0])
        except ValueError:
            errors.append(
                f"{label_path.name}, ligne {line_number}: "
                f"class_id doit être un entier, valeur reçue : {values[0]!r}."
            )
            continue

        if class_id not in CLASS_IDS:
            errors.append(
                f"{label_path.name}, ligne {line_number}: "
                f"class_id {class_id} invalide; valeurs autorisées : 0, 1, 2, 3."
            )
            continue

        try:
            x_center, y_center, width, height = (
                float(value) for value in values[1:]
            )
        except ValueError:
            errors.append(
                f"{label_path.name}, ligne {line_number}: "
                "x_center, y_center, width et height doivent être numériques."
            )
            continue

        coordinates = {
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height,
        }
        invalid_coordinate = False
        for coordinate_name, coordinate_value in coordinates.items():
            if not math.isfinite(coordinate_value):
                errors.append(
                    f"{label_path.name}, ligne {line_number}: "
                    f"{coordinate_name} doit être une valeur finie."
                )
                invalid_coordinate = True
            elif not 0 <= coordinate_value <= 1:
                errors.append(
                    f"{label_path.name}, ligne {line_number}: "
                    f"{coordinate_name}={coordinate_value} est hors de [0, 1]."
                )
                invalid_coordinate = True

        if width <= 0:
            errors.append(
                f"{label_path.name}, ligne {line_number}: "
                f"width doit être strictement positive (valeur : {width})."
            )
            invalid_coordinate = True
        if height <= 0:
            errors.append(
                f"{label_path.name}, ligne {line_number}: "
                f"height doit être strictement positive (valeur : {height})."
            )
            invalid_coordinate = True

        if not invalid_coordinate:
            valid_objects += 1

    return valid_objects, errors


def write_report(
    report_path: Path,
    dataset_dir: Path,
    image_count: int,
    label_count: int,
    valid_label_count: int,
    valid_object_count: int,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Écrit un rapport lisible même lorsque des erreurs sont présentes."""
    lines = [
        "RAPPORT DE VALIDATION DES ANNOTATIONS",
        "=" * 40,
        f"Dossier analysé : {dataset_dir}",
        f"Images détectées : {image_count}",
        f"Fichiers label .txt détectés : {label_count}",
        f"Fichiers label sans erreur : {valid_label_count}",
        f"Objets annotés valides : {valid_object_count}",
        f"Nombre d'erreurs : {len(errors)}",
        f"Nombre d'avertissements : {len(warnings)}",
        "",
        "ERREURS",
        "-------",
    ]
    lines.extend(f"- {message}" for message in errors)
    if not errors:
        lines.append("- Aucune erreur.")
    lines.extend(["", "AVERTISSEMENTS", "--------------"])
    lines.extend(f"- {message}" for message in warnings)
    if not warnings:
        lines.append("- Aucun avertissement.")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_dataset(dataset_dir: Path, report_path: Path) -> int:
    """Valide les labels et la correspondance image/label sans rien modifier."""
    dataset_dir = dataset_dir.resolve()
    report_path = report_path.resolve()
    image_dir = dataset_dir / "images"
    label_dir = dataset_dir / "labels"

    # Refuser les chemins manquants ou une destination du rapport dans le dataset.
    if not image_dir.is_dir() or not label_dir.is_dir():
        missing = []
        if not image_dir.is_dir():
            missing.append(f"dossier d'images introuvable : {image_dir}")
        if not label_dir.is_dir():
            missing.append(f"dossier de labels introuvable : {label_dir}")
        for message in missing:
            print(f"ERREUR : {message}", file=sys.stderr)
        return 1

    if is_inside(report_path, dataset_dir):
        print(
            f"ERREUR : le rapport ne peut pas être écrit dans le dataset "
            f"({report_path}). Choisissez un chemin en dehors de {dataset_dir}.",
            file=sys.stderr,
        )
        return 1

    errors: list[str] = []
    warnings: list[str] = []

    # Indexer les images reconnues par nom de base pour apparier les fichiers .txt.
    images_by_stem: dict[str, list[Path]] = defaultdict(list)
    image_count = 0
    for path in sorted(image_dir.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file():
            warnings.append(f"Entrée non fichier ignorée dans images/: {path.name}")
        elif path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            image_count += 1
            images_by_stem[path.stem.casefold()].append(path)
        else:
            warnings.append(f"Fichier ignoré dans images/ (extension non reconnue) : {path.name}")

    # Lire tous les labels .txt, signaler les noms ambigus et valider leur contenu.
    labels_by_stem: dict[str, list[Path]] = defaultdict(list)
    label_paths: list[Path] = []
    for path in sorted(label_dir.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file():
            warnings.append(f"Entrée non fichier ignorée dans labels/: {path.name}")
        elif path.suffix.lower() == ".txt":
            label_paths.append(path)
            labels_by_stem[path.stem.casefold()].append(path)
        else:
            warnings.append(f"Fichier ignoré dans labels/ (extension non .txt) : {path.name}")

    for stem, paths in sorted(images_by_stem.items()):
        if len(paths) > 1:
            errors.append(
                f"Plusieurs images ont le même nom de base '{stem}' : "
                f"{', '.join(path.name for path in paths)}."
            )
    for stem, paths in sorted(labels_by_stem.items()):
        if len(paths) > 1:
            errors.append(
                f"Plusieurs labels ont le même nom de base '{stem}' : "
                f"{', '.join(path.name for path in paths)}."
            )

    image_stems = set(images_by_stem)
    label_stems = set(labels_by_stem)
    for stem, paths in sorted(images_by_stem.items()):
        if stem not in label_stems:
            for image_path in paths:
                errors.append(f"Image sans label correspondant : {image_path.name}")
    for stem, paths in sorted(labels_by_stem.items()):
        if stem not in image_stems:
            for label_path in paths:
                errors.append(f"Label sans image correspondante : {label_path.name}")

    valid_label_count = 0
    valid_object_count = 0
    for label_path in label_paths:
        object_count, label_errors = validate_label_file(label_path)
        valid_object_count += object_count
        if not label_errors:
            valid_label_count += 1
        errors.extend(label_errors)

    try:
        write_report(
            report_path=report_path,
            dataset_dir=dataset_dir,
            image_count=image_count,
            label_count=len(label_paths),
            valid_label_count=valid_label_count,
            valid_object_count=valid_object_count,
            errors=errors,
            warnings=warnings,
        )
    except OSError as exc:
        print(f"ERREUR : impossible d'écrire le rapport {report_path}: {exc}", file=sys.stderr)
        return 1

    print(f"Images détectées : {image_count}")
    print(f"Fichiers label .txt : {len(label_paths)}")
    print(f"Fichiers label sans erreur : {valid_label_count}")
    print(f"Objets annotés valides : {valid_object_count}")
    print(f"Erreurs : {len(errors)}")
    print(f"Avertissements : {len(warnings)}")
    print(f"Rapport : {report_path}")

    for message in warnings:
        print(f"AVERTISSEMENT : {message}")
    for message in errors:
        print(f"ERREUR : {message}", file=sys.stderr)

    return 1 if errors else 0


def main() -> int:
    """Point d'entrée du validateur."""
    arguments = parse_arguments()
    return validate_dataset(arguments.dataset_dir, arguments.report_path)


if __name__ == "__main__":
    raise SystemExit(main())
