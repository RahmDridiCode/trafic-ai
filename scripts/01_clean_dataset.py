"""Create a cleaned, RGB/JPEG copy of a YOLO image/label dataset."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps


# Les chemins par défaut sont calculés depuis la racine du projet, pas depuis
# le dossier courant depuis lequel le script est lancé.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "raw"
CLEANED_DIR = PROJECT_ROOT / "cleaned"
REPORT_PATH = PROJECT_ROOT / "cleaning_report.txt"

# Seules les extensions que Pillow sait normalement décoder sont acceptées.
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
    """Permet d'utiliser les chemins du projet ou de les remplacer au besoin."""
    parser = argparse.ArgumentParser(
        description=(
            "Vérifie les fichiers du dossier raw et crée une copie nettoyée "
            "dans cleaned, sans modifier raw."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help="Dossier source contenant images/ et labels/ (défaut : <projet>/raw).",
    )
    parser.add_argument(
        "--cleaned-dir",
        type=Path,
        default=CLEANED_DIR,
        help="Dossier de sortie (défaut : <projet>/cleaned).",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPORT_PATH,
        help="Chemin du rapport (défaut : <projet>/cleaning_report.txt).",
    )
    return parser.parse_args()


def get_pixel_hash(image: Image.Image) -> str:
    """Calcule un hash des pixels RGB pour trouver les doublons visuels exacts."""
    digest = hashlib.sha256()
    digest.update(f"{image.width}x{image.height}:".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def write_report(
    report_path: Path,
    raw_dir: Path,
    counts: dict[str, int],
    errors: list[str],
    warnings: list[str],
    cleaned_dir: Path,
) -> None:
    """Enregistre le bilan complet, y compris les problèmes non bloquants."""
    lines = [
        "RAPPORT DE NETTOYAGE DU DATASET",
        "=" * 34,
        f"Dossier source : {raw_dir}",
        f"Dossier nettoyé : {cleaned_dir}",
        "",
        f"Fichiers image détectés (toutes extensions) : {counts['total_image_files']}",
        f"Images valides copiées : {counts['valid_images']}",
        f"Images invalides ou non copiées : {counts['invalid_images']}",
        f"Fichiers label .txt détectés : {counts['label_files']}",
        f"Fichiers ignorés (extension/type non pris en charge) : {counts['ignored_files']}",
        f"Labels sans image correspondante : {counts['orphan_labels']}",
        "",
        "ERREURS",
        "-------",
    ]
    lines.extend(f"- {message}" for message in errors)
    if not errors:
        lines.append("- Aucune erreur.")
    lines.extend(["", "FICHIERS IGNORÉS / AVERTISSEMENTS", "---------------------------------"])
    lines.extend(f"- {message}" for message in warnings)
    if not warnings:
        lines.append("- Aucun fichier ignoré.")
    lines.extend(
        [
            "",
            "Note : les doublons correspondent aux images dont les pixels RGB "
            "décodés sont identiques.",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def clean_dataset(
    raw_dir: Path,
    cleaned_dir: Path,
    report_path: Path,
) -> int:
    """Vérifie les paires image/label et copie les éléments valides."""
    raw_dir = raw_dir.resolve()
    cleaned_dir = cleaned_dir.resolve()
    report_path = report_path.resolve()

    counts = {
        "total_image_files": 0,
        "valid_images": 0,
        "invalid_images": 0,
        "label_files": 0,
        "ignored_files": 0,
        "orphan_labels": 0,
    }
    errors: list[str] = []
    warnings: list[str] = []
    report_is_inside_raw = is_inside(report_path, raw_dir)

    # Garde-fou : ni les sorties ni le rapport ne peuvent être créés dans raw/.
    if is_inside(cleaned_dir, raw_dir):
        errors.append(
            f"Le dossier de sortie {cleaned_dir} est dans raw/; "
            "choisissez une destination en dehors de raw/."
        )
    if report_is_inside_raw:
        errors.append(
            f"Le rapport {report_path} est dans raw/; "
            "choisissez un chemin en dehors de raw/."
        )

    # Refuser de mélanger de nouveaux résultats avec une sortie précédente.
    if cleaned_dir.exists() and (
        not cleaned_dir.is_dir() or any(cleaned_dir.iterdir())
    ):
        errors.append(
            f"Le dossier de sortie existe déjà et n'est pas vide : {cleaned_dir}. "
            "Déplacez-le ou choisissez un autre --cleaned-dir; aucun fichier "
            "de sortie existant n'a été remplacé."
        )

    image_dir = raw_dir / "images"
    label_dir = raw_dir / "labels"
    if not image_dir.is_dir():
        errors.append(f"Dossier d'images introuvable : {image_dir}")
    if not label_dir.is_dir():
        errors.append(f"Dossier de labels introuvable : {label_dir}")

    # Si la destination est dangereuse ou existe déjà, générer le rapport puis
    # s'arrêter avant toute création du dataset nettoyé.
    if errors:
        if not report_is_inside_raw:
            try:
                write_report(
                    report_path, raw_dir, counts, errors, warnings, cleaned_dir
                )
            except OSError as exc:
                print(
                    f"Impossible d'écrire le rapport {report_path}: {exc}",
                    file=sys.stderr,
                )
        for message in errors:
            print(f"ERREUR : {message}", file=sys.stderr)
        return 1

    image_entries = sorted(image_dir.iterdir(), key=lambda item: item.name.casefold())
    label_entries = sorted(label_dir.iterdir(), key=lambda item: item.name.casefold())
    counts["total_image_files"] = sum(item.is_file() for item in image_entries)

    # Indexer les images prises en charge et les labels par nom de base.
    images_by_stem: dict[str, list[Path]] = defaultdict(list)
    for image_path in image_entries:
        if not image_path.is_file():
            counts["ignored_files"] += 1
            warnings.append(f"Entrée non fichier ignorée dans images/: {image_path.name}")
        elif image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            counts["ignored_files"] += 1
            warnings.append(
                f"Extension d'image non prise en charge, fichier ignoré : {image_path.name}"
            )
        else:
            images_by_stem[image_path.stem.casefold()].append(image_path)

    labels_by_stem: dict[str, list[Path]] = defaultdict(list)
    for label_path in label_entries:
        if not label_path.is_file():
            counts["ignored_files"] += 1
            warnings.append(f"Entrée non fichier ignorée dans labels/: {label_path.name}")
        elif label_path.suffix.lower() != ".txt":
            counts["ignored_files"] += 1
            warnings.append(
                f"Fichier de label non .txt ignoré : {label_path.name}"
            )
        else:
            counts["label_files"] += 1
            labels_by_stem[label_path.stem.casefold()].append(label_path)

    # Les noms ambigus ne sont pas choisis arbitrairement.
    ambiguous_image_stems = {
        stem for stem, paths in images_by_stem.items() if len(paths) > 1
    }
    for stem in sorted(ambiguous_image_stems):
        paths = images_by_stem[stem]
        counts["invalid_images"] += len(paths)
        errors.append(
            "Plusieurs images ont le même nom de base "
            f"'{stem}' : {', '.join(path.name for path in paths)}"
        )

    ambiguous_label_stems = {
        stem for stem, paths in labels_by_stem.items() if len(paths) > 1
    }
    for stem in sorted(ambiguous_label_stems):
        errors.append(
            "Plusieurs fichiers .txt ont le même nom de base "
            f"'{stem}' : {', '.join(path.name for path in labels_by_stem[stem])}"
        )

    # Identifier les labels orphelins et créer une destination vide.
    image_stems = set(images_by_stem)
    for stem, paths in sorted(labels_by_stem.items()):
        if stem not in image_stems:
            counts["orphan_labels"] += len(paths)
            for path in paths:
                errors.append(f"Label sans image correspondante : {path.name}")

    cleaned_images_dir = cleaned_dir / "images"
    cleaned_labels_dir = cleaned_dir / "labels"
    try:
        cleaned_images_dir.mkdir(parents=True, exist_ok=True)
        cleaned_labels_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        errors.append(f"Impossible de créer les dossiers de sortie : {exc}")
        try:
            write_report(
                report_path, raw_dir, counts, errors, warnings, cleaned_dir
            )
        except OSError as report_exc:
            print(f"Impossible d'écrire le rapport {report_path}: {report_exc}", file=sys.stderr)
        print(f"ERREUR : {errors[-1]}", file=sys.stderr)
        return 1

    # Convertir chaque image valide en JPG/RGB et copier son label sans le modifier.
    seen_pixel_hashes: dict[str, str] = {}
    for stem, image_paths in sorted(images_by_stem.items()):
        if stem in ambiguous_image_stems:
            continue

        image_path = image_paths[0]
        label_paths = labels_by_stem.get(stem, [])
        if not label_paths:
            counts["invalid_images"] += 1
            errors.append(f"Image sans label correspondant : {image_path.name}")
            continue
        if stem in ambiguous_label_stems:
            counts["invalid_images"] += 1
            errors.append(
                f"Image ignorée car plusieurs labels correspondent : {image_path.name}"
            )
            continue

        cleaned_image_path = cleaned_images_dir / f"{image_path.stem}.jpg"
        cleaned_label_path = cleaned_labels_dir / f"{image_path.stem}.txt"

        try:
            # verify() vérifie le fichier; une seconde ouverture charge réellement
            # les pixels avant conversion (et détecte les images tronquées).
            with Image.open(image_path) as source_image:
                source_image.verify()
            with Image.open(image_path) as source_image:
                source_image.load()
                rgb_image = ImageOps.exif_transpose(source_image).convert("RGB")

            pixel_hash = get_pixel_hash(rgb_image)
            if pixel_hash in seen_pixel_hashes:
                counts["invalid_images"] += 1
                errors.append(
                    f"Doublon d'image ignoré : {image_path.name} est identique à "
                    f"{seen_pixel_hashes[pixel_hash]} (pixels RGB)."
                )
                continue

            rgb_image.save(cleaned_image_path, format="JPEG", quality=95, optimize=True)
            shutil.copy2(label_paths[0], cleaned_label_path)
            seen_pixel_hashes[pixel_hash] = image_path.name
            counts["valid_images"] += 1
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
            # Si une moitié de la paire a été écrite avant l'erreur, la retirer
            # uniquement de cleaned/ pour ne pas laisser une paire incomplète.
            for output_path in (cleaned_image_path, cleaned_label_path):
                if output_path.exists():
                    try:
                        output_path.unlink()
                    except OSError as cleanup_exc:
                        errors.append(
                            f"Impossible de retirer la sortie partielle {output_path}: "
                            f"{cleanup_exc}"
                        )
            counts["invalid_images"] += 1
            errors.append(f"Image/label non copiés pour {image_path.name}: {exc}")

    # Le rapport et la console exposent explicitement tous les problèmes trouvés.
    try:
        write_report(report_path, raw_dir, counts, errors, warnings, cleaned_dir)
    except OSError as exc:
        print(f"Impossible d'écrire le rapport {report_path}: {exc}", file=sys.stderr)
        return 1

    print(f"Images détectées : {counts['total_image_files']}")
    print(f"Images valides copiées : {counts['valid_images']}")
    print(f"Images invalides ou non copiées : {counts['invalid_images']}")
    print(f"Labels .txt détectés : {counts['label_files']}")
    print(f"Fichiers ignorés : {counts['ignored_files']}")
    print(f"Labels sans image : {counts['orphan_labels']}")
    print(f"Rapport : {report_path}")

    for message in warnings:
        print(f"AVERTISSEMENT : {message}")
    for message in errors:
        print(f"ERREUR : {message}", file=sys.stderr)

    return 1 if errors else 0


def main() -> int:
    """Lance le nettoyage avec les chemins fournis par l'utilisateur."""
    arguments = parse_arguments()
    return clean_dataset(
        raw_dir=arguments.raw_dir,
        cleaned_dir=arguments.cleaned_dir,
        report_path=arguments.report_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
