"""Create reproducible train/validation/test copies of the cleaned dataset."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


# Les chemins par défaut sont ancrés à la racine du projet.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEANED_DIR = PROJECT_ROOT / "cleaned"
DATASET_DIR = PROJECT_ROOT / "dataset"
RANDOM_SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
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
SPLIT_NAMES = ("train", "val", "test")


def is_inside(path: Path, parent: Path) -> bool:
    """Indique si path est parent ou l'un de ses sous-dossiers."""
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def parse_arguments() -> argparse.Namespace:
    """Lit les chemins sources/sorties et la seed depuis la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Répartit cleaned/ en train, val et test de manière reproductible."
    )
    parser.add_argument(
        "--cleaned-dir",
        type=Path,
        default=CLEANED_DIR,
        help="Source contenant images/ et labels/ (défaut : <projet>/cleaned).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help="Destination (défaut : <projet>/dataset); elle ne doit pas exister.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Seed du mélange aléatoire (défaut : {RANDOM_SEED}).",
    )
    return parser.parse_args()


def calculate_split_sizes(total: int) -> dict[str, int]:
    """Calcule des tailles proches des ratios demandés et non vides si possible."""
    if total < len(SPLIT_NAMES):
        raise ValueError(
            "Il faut au moins 3 paires image/label pour créer trois ensembles non vides."
        )

    ratios = (TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    raw_sizes = [total * ratio for ratio in ratios]
    sizes = [int(value) for value in raw_sizes]
    remaining = total - sum(sizes)

    # Attribuer les éléments restants selon les plus grandes parties décimales.
    remainders = sorted(
        range(len(SPLIT_NAMES)),
        key=lambda index: (raw_sizes[index] - sizes[index], -index),
        reverse=True,
    )
    for index in remainders[:remaining]:
        sizes[index] += 1

    # Pour les très petits datasets, garantir un élément minimum par split.
    for empty_index, size in enumerate(sizes):
        if size == 0:
            donor_index = max(range(len(sizes)), key=sizes.__getitem__)
            if sizes[donor_index] <= 1:
                raise ValueError("Impossible de répartir les données sans split vide.")
            sizes[donor_index] -= 1
            sizes[empty_index] = 1

    return dict(zip(SPLIT_NAMES, sizes, strict=True))


def find_image_label_pairs(
    cleaned_dir: Path,
) -> tuple[list[tuple[Path, Path]], list[str], list[str]]:
    """Associe les images aux labels et rapporte les données ambiguës ou orphelines."""
    image_dir = cleaned_dir / "images"
    label_dir = cleaned_dir / "labels"
    errors: list[str] = []
    warnings: list[str] = []
    images_by_stem: dict[str, list[Path]] = defaultdict(list)
    labels_by_stem: dict[str, list[Path]] = defaultdict(list)

    # Recenser les images prises en charge et les entrées inattendues.
    for path in sorted(image_dir.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file():
            warnings.append(f"Entrée non fichier ignorée dans images/: {path.name}")
        elif path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            images_by_stem[path.stem.casefold()].append(path)
        else:
            warnings.append(
                f"Fichier ignoré dans images/ (extension non reconnue) : {path.name}"
            )

    # Les annotations doivent être des fichiers texte .txt.
    for path in sorted(label_dir.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file():
            warnings.append(f"Entrée non fichier ignorée dans labels/: {path.name}")
        elif path.suffix.lower() == ".txt":
            labels_by_stem[path.stem.casefold()].append(path)
        else:
            warnings.append(
                f"Fichier ignoré dans labels/ (extension non .txt) : {path.name}"
            )

    # Signaler les correspondances absentes et les noms de base ambigus.
    for stem, paths in sorted(images_by_stem.items()):
        if len(paths) > 1:
            errors.append(
                f"Plusieurs images ont le même nom de base '{stem}' : "
                f"{', '.join(path.name for path in paths)}."
            )
        if stem not in labels_by_stem:
            errors.extend(
                f"Image sans label correspondant : {path.name}" for path in paths
            )
    for stem, paths in sorted(labels_by_stem.items()):
        if len(paths) > 1:
            errors.append(
                f"Plusieurs labels ont le même nom de base '{stem}' : "
                f"{', '.join(path.name for path in paths)}."
            )
        if stem not in images_by_stem:
            errors.extend(
                f"Label sans image correspondante : {path.name}" for path in paths
            )

    pairs: list[tuple[Path, Path]] = []
    for stem, image_paths in sorted(images_by_stem.items()):
        label_paths = labels_by_stem.get(stem, [])
        if len(image_paths) == 1 and len(label_paths) == 1:
            pairs.append((image_paths[0], label_paths[0]))

    return pairs, errors, warnings


def copy_split(
    pairs: list[tuple[Path, Path]],
    split_root: Path,
) -> None:
    """Copie les paires image/label dans le sous-dossier de split prévu."""
    for image_path, label_path in pairs:
        target_image = split_root / "images" / image_path.name
        target_label = split_root / "labels" / label_path.name
        shutil.copy2(image_path, target_image)
        shutil.copy2(label_path, target_label)


def create_dataset_split(
    cleaned_dir: Path,
    dataset_dir: Path,
    seed: int = RANDOM_SEED,
) -> int:
    """Vérifie les paires puis construit dataset/ sans écraser une sortie existante."""
    cleaned_dir = cleaned_dir.resolve()
    dataset_dir = dataset_dir.resolve()
    image_dir = cleaned_dir / "images"
    label_dir = cleaned_dir / "labels"

    # Refuser les sources absentes et tout chevauchement source/destination.
    if not image_dir.is_dir() or not label_dir.is_dir():
        if not image_dir.is_dir():
            print(f"ERREUR : dossier d'images introuvable : {image_dir}", file=sys.stderr)
        if not label_dir.is_dir():
            print(f"ERREUR : dossier de labels introuvable : {label_dir}", file=sys.stderr)
        return 1
    if is_inside(dataset_dir, cleaned_dir) or is_inside(cleaned_dir, dataset_dir):
        print(
            f"ERREUR : les dossiers source et destination ne doivent pas se "
            f"chevaucher ({cleaned_dir}, {dataset_dir}).",
            file=sys.stderr,
        )
        return 1
    if dataset_dir.exists():
        print(
            f"ERREUR : la destination existe déjà : {dataset_dir}. "
            "Aucun fichier ne sera remplacé; choisissez un autre --dataset-dir "
            "ou déplacez l'ancien dossier.",
            file=sys.stderr,
        )
        return 1

    # Vérifier toutes les paires avant de commencer les copies.
    pairs, errors, warnings = find_image_label_pairs(cleaned_dir)
    for message in warnings:
        print(f"AVERTISSEMENT : {message}")
    if errors:
        for message in errors:
            print(f"ERREUR : {message}", file=sys.stderr)
        print("Aucun split n'a été créé car les paires ne sont pas toutes valides.")
        return 1
    if not pairs:
        print("ERREUR : aucune paire image/label à répartir.", file=sys.stderr)
        return 1

    try:
        split_sizes = calculate_split_sizes(len(pairs))
    except ValueError as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        return 1

    # Mélanger une liste triée pour produire le même split avec la même seed.
    shuffled_pairs = sorted(pairs, key=lambda pair: pair[0].name.casefold())
    random.Random(seed).shuffle(shuffled_pairs)
    train_end = split_sizes["train"]
    val_end = train_end + split_sizes["val"]
    pairs_by_split = {
        "train": shuffled_pairs[:train_end],
        "val": shuffled_pairs[train_end:val_end],
        "test": shuffled_pairs[val_end:],
    }

    # Vérifier l'absence de chevauchement avant toute copie.
    stems_by_split = {
        split_name: {
            image_path.stem.casefold() for image_path, _ in split_pairs
        }
        for split_name, split_pairs in pairs_by_split.items()
    }
    overlaps = (
        stems_by_split["train"] & stems_by_split["val"]
        | stems_by_split["train"] & stems_by_split["test"]
        | stems_by_split["val"] & stems_by_split["test"]
    )
    if overlaps:
        overlap_list = ", ".join(sorted(overlaps))
        print(
            f"ERREUR : chevauchement entre les ensembles : {overlap_list}",
            file=sys.stderr,
        )
        return 1

    # Copier dans un dossier temporaire du même parent et publier dataset/ seulement
    # lorsque toutes les paires sont copiées avec succès.
    try:
        dataset_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{dataset_dir.name}_staging_",
            dir=dataset_dir.parent,
        ) as temporary_directory:
            staging_dir = Path(temporary_directory)
            for split_name in SPLIT_NAMES:
                split_root = staging_dir / split_name
                (split_root / "images").mkdir(parents=True)
                (split_root / "labels").mkdir()
                copy_split(pairs_by_split[split_name], split_root)

            # Ne jamais remplacer une destination qui serait apparue entre-temps.
            if dataset_dir.exists():
                raise FileExistsError(
                    f"La destination est apparue pendant la copie : {dataset_dir}"
                )
            staging_dir.rename(dataset_dir)
    except (OSError, shutil.Error) as exc:
        print(f"ERREUR : création du split impossible : {exc}", file=sys.stderr)
        return 1

    # Vérifier le résultat publié et récapituler le split effectivement créé.
    for split_name in SPLIT_NAMES:
        split_images = dataset_dir / split_name / "images"
        split_labels = dataset_dir / split_name / "labels"
        image_stems = {
            path.stem.casefold()
            for path in split_images.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        }
        label_stems = {
            path.stem.casefold()
            for path in split_labels.iterdir()
            if path.is_file() and path.suffix.lower() == ".txt"
        }
        if image_stems != label_stems or len(image_stems) != split_sizes[split_name]:
            print(
                f"ERREUR : vérification après copie échouée pour {split_name}: "
                f"{len(image_stems)} image(s), {len(label_stems)} label(s).",
                file=sys.stderr,
            )
            return 1

    print(f"Seed aléatoire : {seed}")
    print(f"Total : {len(pairs)} images")
    print(f"Train: {split_sizes['train']} images")
    print(f"Val: {split_sizes['val']} images")
    print(f"Test: {split_sizes['test']} images")
    print(f"Chevauchements entre ensembles : {len(overlaps)}")
    print(f"Dataset créé : {dataset_dir}")
    return 0


def main() -> int:
    """Point d'entrée du script de split."""
    arguments = parse_arguments()
    return create_dataset_split(
        cleaned_dir=arguments.cleaned_dir,
        dataset_dir=arguments.dataset_dir,
        seed=arguments.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
