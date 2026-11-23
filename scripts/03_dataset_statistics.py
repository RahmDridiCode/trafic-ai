"""Calculate class and annotation statistics for the cleaned vehicle dataset."""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

# Sélectionner un backend sans interface graphique afin que le script fonctionne
# aussi depuis un terminal ou sur une machine sans écran.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Les chemins sont ancrés à la racine du projet, quel que soit le dossier courant.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "cleaned"
REPORTS_DIR = PROJECT_ROOT / "reports"
CLASS_NAMES = {
    0: "bus",
    1: "car",
    2: "motorcycle",
    3: "truck",
}
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
    """Lit les chemins du dataset et des rapports depuis la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Calcule les statistiques des annotations YOLO de cleaned/."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help="Dossier contenant images/ et labels/ (défaut : <projet>/cleaned).",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Dossier des résultats (défaut : <projet>/reports).",
    )
    return parser.parse_args()


def validate_and_count_label(
    label_path: Path,
) -> tuple[dict[int, int], list[str]]:
    """Valide le format YOLO du fichier et compte les objets valides par classe."""
    class_counts = {class_id: 0 for class_id in CLASS_NAMES}
    errors: list[str] = []

    # Ouvrir le fichier en UTF-8 pour rendre les erreurs de lecture explicites.
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return class_counts, [f"{label_path.name}: lecture impossible ({exc})"]

    # Un fichier vide est autorisé : cette image ne contient aucun objet annoté.
    for line_number, line in enumerate(lines, start=1):
        values = line.split()
        if len(values) != 5:
            errors.append(
                f"{label_path.name}, ligne {line_number}: "
                f"5 valeurs attendues, {len(values)} trouvée(s)."
            )
            continue

        # Vérifier l'identifiant de classe avant de compter l'objet.
        try:
            class_id = int(values[0])
        except ValueError:
            errors.append(
                f"{label_path.name}, ligne {line_number}: "
                f"class_id doit être un entier, valeur reçue : {values[0]!r}."
            )
            continue
        if class_id not in CLASS_NAMES:
            errors.append(
                f"{label_path.name}, ligne {line_number}: "
                f"class_id {class_id} invalide; valeurs autorisées : 0, 1, 2, 3."
            )
            continue

        # Vérifier les coordonnées normalisées et les dimensions de la boîte.
        try:
            x_center, y_center, width, height = (
                float(value) for value in values[1:]
            )
        except ValueError:
            errors.append(
                f"{label_path.name}, ligne {line_number}: "
                "les coordonnées et dimensions doivent être numériques."
            )
            continue

        coordinate_values = {
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height,
        }
        line_is_valid = True
        for name, value in coordinate_values.items():
            if not math.isfinite(value):
                errors.append(
                    f"{label_path.name}, ligne {line_number}: "
                    f"{name} doit être une valeur finie."
                )
                line_is_valid = False
            elif not 0 <= value <= 1:
                errors.append(
                    f"{label_path.name}, ligne {line_number}: "
                    f"{name}={value} est hors de [0, 1]."
                )
                line_is_valid = False

        if width <= 0:
            errors.append(
                f"{label_path.name}, ligne {line_number}: "
                f"width doit être strictement positive (valeur : {width})."
            )
            line_is_valid = False
        if height <= 0:
            errors.append(
                f"{label_path.name}, ligne {line_number}: "
                f"height doit être strictement positive (valeur : {height})."
            )
            line_is_valid = False

        if line_is_valid:
            class_counts[class_id] += 1

    return class_counts, errors


def collect_dataset_statistics(
    dataset_dir: Path,
) -> tuple[
    int,
    dict[int, int],
    dict[int, int],
    list[str],
    list[str],
]:
    """Compte images et objets et rapporte toute incohérence de paire/label."""
    image_dir = dataset_dir / "images"
    label_dir = dataset_dir / "labels"
    errors: list[str] = []
    warnings: list[str] = []
    images_by_stem: dict[str, list[Path]] = defaultdict(list)
    labels_by_stem: dict[str, list[Path]] = defaultdict(list)

    # Recenser les images lisibles par extension et signaler les autres entrées.
    image_count = 0
    for path in sorted(image_dir.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file():
            warnings.append(f"Entrée non fichier ignorée dans images/: {path.name}")
        elif path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            image_count += 1
            images_by_stem[path.stem.casefold()].append(path)
        else:
            warnings.append(
                f"Fichier ignoré dans images/ (extension non reconnue) : {path.name}"
            )

    # Recenser les labels et signaler les fichiers hors format .txt.
    for path in sorted(label_dir.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file():
            warnings.append(f"Entrée non fichier ignorée dans labels/: {path.name}")
        elif path.suffix.lower() == ".txt":
            labels_by_stem[path.stem.casefold()].append(path)
        else:
            warnings.append(
                f"Fichier ignoré dans labels/ (extension non .txt) : {path.name}"
            )

    # Des noms de base ambigus ne permettent pas d'associer une paire sûrement.
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

    # Toute image et tout label doivent avoir un correspondant.
    for stem, paths in sorted(images_by_stem.items()):
        if stem not in labels_by_stem:
            errors.extend(
                f"Image sans label correspondant : {path.name}" for path in paths
            )
    for stem, paths in sorted(labels_by_stem.items()):
        if stem not in images_by_stem:
            errors.extend(
                f"Label sans image correspondante : {path.name}" for path in paths
            )

    # Compter les objets valides et le nombre d'images contenant chaque classe.
    object_counts = {class_id: 0 for class_id in CLASS_NAMES}
    image_counts = {class_id: 0 for class_id in CLASS_NAMES}
    for stem, image_paths in sorted(images_by_stem.items()):
        label_paths = labels_by_stem.get(stem, [])
        if len(image_paths) != 1 or len(label_paths) != 1:
            continue

        file_counts, label_errors = validate_and_count_label(label_paths[0])
        errors.extend(label_errors)
        for class_id, count in file_counts.items():
            object_counts[class_id] += count
            if count > 0:
                image_counts[class_id] += 1

    return image_count, object_counts, image_counts, errors, warnings


def write_class_distribution_chart(
    output_path: Path,
    object_counts: dict[int, int],
) -> None:
    """Crée un graphique simple du nombre d'objets annotés pour chaque classe."""
    class_ids = list(CLASS_NAMES)
    class_labels = [CLASS_NAMES[class_id] for class_id in class_ids]
    counts = [object_counts[class_id] for class_id in class_ids]

    figure, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(class_labels, counts, color="#3572A5")
    axis.set_title("Distribution des objets annotés par classe")
    axis.set_xlabel("Classe")
    axis.set_ylabel("Nombre d'objets")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.bar_label(bars, padding=3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def write_statistics_report(
    report_path: Path,
    dataset_dir: Path,
    image_count: int,
    object_counts: dict[int, int],
    image_counts: dict[int, int],
    errors: list[str],
    warnings: list[str],
) -> None:
    """Enregistre les comptes, proportions et problèmes rencontrés."""
    total_objects = sum(object_counts.values())
    mean_objects = total_objects / image_count if image_count else 0.0

    lines = [
        "STATISTIQUES DU DATASET",
        "=" * 24,
        f"Dossier analysé : {dataset_dir}",
        f"Nombre total d'images : {image_count}",
        f"Nombre total d'objets annotés valides : {total_objects}",
        f"Moyenne d'objets par image : {mean_objects:.3f}",
        "",
        "STATISTIQUES PAR CLASSE",
        "-----------------------",
        "Classe | Objets | Distribution des objets | Images contenant la classe",
    ]
    for class_id, class_name in CLASS_NAMES.items():
        object_count = object_counts[class_id]
        percentage = object_count / total_objects * 100 if total_objects else 0.0
        lines.append(
            f"{class_id} - {class_name} | {object_count} | "
            f"{percentage:.2f}% | {image_counts[class_id]}"
        )
    lines.extend(
        [
            "",
            f"Nombre d'erreurs : {len(errors)}",
            f"Nombre d'avertissements : {len(warnings)}",
            "",
            "ERREURS",
            "-------",
        ]
    )
    lines.extend(f"- {message}" for message in errors)
    if not errors:
        lines.append("- Aucune erreur.")
    lines.extend(["", "AVERTISSEMENTS", "--------------"])
    lines.extend(f"- {message}" for message in warnings)
    if not warnings:
        lines.append("- Aucun avertissement.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def calculate_statistics(dataset_dir: Path, reports_dir: Path) -> int:
    """Valide, calcule les statistiques et enregistre les rapports."""
    dataset_dir = dataset_dir.resolve()
    reports_dir = reports_dir.resolve()
    image_dir = dataset_dir / "images"
    label_dir = dataset_dir / "labels"

    # Ne pas créer de résultats dans le dataset ni accepter des sources absentes.
    if is_inside(reports_dir, dataset_dir):
        print(
            f"ERREUR : le dossier de rapports ne peut pas être dans le dataset "
            f"({reports_dir}).",
            file=sys.stderr,
        )
        return 1
    if not image_dir.is_dir() or not label_dir.is_dir():
        if not image_dir.is_dir():
            print(f"ERREUR : dossier d'images introuvable : {image_dir}", file=sys.stderr)
        if not label_dir.is_dir():
            print(f"ERREUR : dossier de labels introuvable : {label_dir}", file=sys.stderr)
        return 1

    # Établir les comptes sans modifier les images ni les fichiers d'annotation.
    image_count, object_counts, image_counts, errors, warnings = (
        collect_dataset_statistics(dataset_dir)
    )
    report_path = reports_dir / "dataset_statistics.txt"
    chart_path = reports_dir / "class_distribution.png"

    try:
        reports_dir.mkdir(parents=True, exist_ok=True)
        write_statistics_report(
            report_path=report_path,
            dataset_dir=dataset_dir,
            image_count=image_count,
            object_counts=object_counts,
            image_counts=image_counts,
            errors=errors,
            warnings=warnings,
        )
    except OSError as exc:
        print(f"ERREUR : impossible d'écrire le rapport {report_path}: {exc}", file=sys.stderr)
        return 1

    total_objects = sum(object_counts.values())
    mean_objects = total_objects / image_count if image_count else 0.0
    print(f"Nombre total d'images : {image_count}")
    print(f"Nombre total d'objets annotés valides : {total_objects}")
    for class_id, class_name in CLASS_NAMES.items():
        percentage = (
            object_counts[class_id] / total_objects * 100 if total_objects else 0.0
        )
        print(
            f"{class_name} : {object_counts[class_id]} objets "
            f"({percentage:.2f} %), dans {image_counts[class_id]} images"
        )
    print(f"Moyenne d'objets par image : {mean_objects:.3f}")
    print(f"Erreurs : {len(errors)}")
    print(f"Avertissements : {len(warnings)}")
    print(f"Rapport : {report_path}")

    for message in warnings:
        print(f"AVERTISSEMENT : {message}")
    for message in errors:
        print(f"ERREUR : {message}", file=sys.stderr)

    # Éviter de publier une distribution partielle si les annotations sont invalides.
    if errors:
        return 1

    try:
        write_class_distribution_chart(chart_path, object_counts)
    except (OSError, ValueError) as exc:
        print(f"ERREUR : impossible de créer le graphique {chart_path}: {exc}", file=sys.stderr)
        return 1

    print(f"Graphique : {chart_path}")
    return 0


def main() -> int:
    """Point d'entrée du script de statistiques."""
    arguments = parse_arguments()
    return calculate_statistics(arguments.dataset_dir, arguments.reports_dir)


if __name__ == "__main__":
    raise SystemExit(main())
