"""Log interpretation: each rule matches a message that the project's code (or its libraries) really emits."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Diagnosis:
    level: str  # "error" | "warning" | "info"
    title: str
    cause: str
    solution: str
    evidence: str


# (regex, level, title, cause, solution) - sources are noted next to each rule.
RULES: list[tuple[str, str, str, str, str]] = [
    # src/main.py
    (r"Input video file not found", "error", "Vidéo introuvable",
     "Le chemin passé à --input ne désigne aucun fichier (src/main.py).",
     "Sélectionnez une vidéo existante dans l'étape « Vidéo »."),
    (r"OpenCV could not open video container|Unable to open video input", "error", "Vidéo illisible",
     "OpenCV ne sait pas décoder ce conteneur ou ce codec.",
     "Ré-encodez la vidéo en MP4/H.264 ou choisissez un autre fichier."),
    (r"Traffic-light initialization failed:.*ROI config resolution is (\d+x\d+), but the video is (\d+x\d+)",
     "error", "ROI incompatible avec la résolution de la vidéo",
     "Le fichier ROI a été dessiné pour une autre résolution que celle de la vidéo (src/Feux/roi_config.py).",
     "Choisissez un ROI compatible (l'interface ne propose que ceux qui passent la validation) ou créez-en un "
     "avec l'éditeur de ROI sur cette vidéo."),
    (r"Traffic-light initialization failed:.*is outside the", "error", "Points du ROI hors de l'image",
     "Au moins un point du ROI ou de la ligne d'arrêt sort des dimensions de la vidéo.",
     "Recréez le ROI avec l'éditeur de ROI en utilisant cette vidéo."),
    (r"Cannot read traffic-light ROI config|ROI configuration must contain|must contain a JSON object|"
     r"STOP_LINE_COORDS", "error", "Fichier ROI invalide",
     "Le JSON du ROI est absent, mal formé ou incomplet (VEHICLE_DETECT_ROI_POLYGON, "
     "RED_LIGHT_DETECT_ROI_POLYGON, STOP_LINE_COORDS).",
     "Régénérez le fichier avec l'éditeur de ROI."),
    (r"Traffic-light initialization failed", "error", "Initialisation des feux impossible",
     "Le modèle des feux ou le ROI n'a pas pu être chargé.",
     "Vérifiez le modèle des feux et le ROI dans l'étape « Feux rouges »."),
    (r"Invalid calibration:.*Calibration resolution is (\d+x\d+), but the video is (\d+x\d+)", "error",
     "Calibration incompatible avec la vidéo",
     "La résolution déclarée dans la calibration diffère de celle de la vidéo (load_calibration).",
     "Choisissez une calibration de même résolution ou lancez sans calibration."),
    (r"Calibration file not found", "error", "Fichier de calibration introuvable",
     "Le chemin donné à --calibration n'existe pas.", "Sélectionnez une calibration existante."),
    (r"Invalid calibration", "error", "Calibration invalide",
     "Le JSON de calibration est incomplet ou hors de l'image (image_points, world_points, resolution).",
     "Corrigez le fichier ou choisissez-en un autre."),
    (r"Detector initialization failed.*(not found|No such file)|YOLO weights not found", "error",
     "Modèle de détection introuvable", "Le fichier de poids indiqué n'existe pas.",
     "Choisissez un modèle détecté dans le projet."),
    (r"Detector initialization failed", "error", "Chargement du modèle impossible",
     "Le checkpoint est corrompu, incompatible avec la version d'Ultralytics ou le paramètre est invalide.",
     "Essayez un autre modèle ou vérifiez la version d'ultralytics (requirements.txt)."),
    (r"Unable to create output directory", "error", "Dossier de sortie inaccessible",
     "Droits insuffisants ou chemin invalide.", "Choisissez un dossier de sortie accessible en écriture."),
    # src/video_processor.py
    (r"Unable to open output video for writing", "error", "Impossible d'écrire la vidéo annotée",
     "Le codec mp4v est indisponible ou le dossier n'est pas accessible en écriture.",
     "Vérifiez les droits d'écriture et l'installation d'OpenCV."),
    (r"Unable to write (speeding alert|red-light violation) image", "error", "Écriture d'image impossible",
     "Le dossier de sortie n'est pas accessible en écriture ou le disque est plein.",
     "Libérez de l'espace ou changez de dossier de sortie."),
    # libraries
    (r"Invalid CUDA|Torch not compiled with CUDA|CUDA is not available|no CUDA", "error",
     "CUDA indisponible", "Un périphérique GPU a été demandé mais PyTorch ne le voit pas.",
     "Choisissez « cpu » ou « auto »."),
    (r"CUDA out of memory|OutOfMemoryError", "error", "Mémoire GPU insuffisante",
     "Le modèle ou la taille d'inférence dépasse la mémoire GPU.",
     "Réduisez la taille d'inférence, utilisez un modèle plus petit ou passez sur « cpu »."),
    (r"MemoryError|DefaultCPUAllocator: not enough memory", "error", "Mémoire insuffisante",
     "La mémoire vive est saturée.", "Réduisez la taille d'inférence ou fermez d'autres applications."),
    (r"ModuleNotFoundError: No module named '([^']+)'", "error", "Dépendance Python manquante",
     "Un module requis n'est pas installé dans l'environnement qui exécute le pipeline.",
     "Installez les dépendances : pip install -r requirements.txt."),
    (r"PermissionError", "error", "Accès refusé",
     "Un fichier est verrouillé (lecteur vidéo ouvert ?) ou non accessible en écriture.",
     "Fermez les programmes qui utilisent les fichiers de sortie."),
    (r"Traceback \(most recent call last\)", "error", "Exception Python non gérée",
     "Le pipeline s'est arrêté sur une exception.", "Consultez la dernière ligne de la trace ci-dessous."),
]

PROGRESS_RE = re.compile(r"Processed (\d+)/(\d+) frames")
SUMMARY_MARK = "TRAFFIC SURVEILLANCE & KINEMATICS SUMMARY"
DONE_MARK = "All outputs generated successfully"


def parse_progress(text: str) -> tuple[int, int] | None:
    matches = PROGRESS_RE.findall(text)
    if not matches:
        return None
    done, total = matches[-1]
    return int(done), int(total)


def analyze_log(text: str, return_code: int | None = None) -> list[Diagnosis]:
    """Match the log against known messages; unmatched failures still surface the last meaningful line."""
    found: list[Diagnosis] = []
    seen: set[str] = set()
    for pattern, level, title, cause, solution in RULES:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and title not in seen:
            line = next((ln for ln in text.splitlines() if re.search(pattern.split(".*")[0], ln, re.IGNORECASE)),
                        match.group(0))
            seen.add(title)
            specific = cause
            if title.startswith(("ROI incompatible", "Calibration incompatible")) and match.lastindex == 2:
                specific += f" Résolution du fichier : {match.group(1)} ; résolution de la vidéo : {match.group(2)}."
            found.append(Diagnosis(level, title, specific, solution, line.strip()[:400]))
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if return_code not in (None, 0) and not any(d.level == "error" for d in found):
        found.append(Diagnosis("error", f"Le pipeline s'est terminé avec le code {return_code}",
                               "Aucune cause connue n'a été reconnue dans le log.",
                               "Lisez les dernières lignes du log.", lines[-1][:400] if lines else ""))
    if return_code == 0 and DONE_MARK not in text and text.strip():
        found.append(Diagnosis("warning", "Fin de log inhabituelle",
                               "Le message final de réussite n'apparaît pas dans le log.",
                               "Vérifiez les fichiers produits dans l'onglet Résultats.", ""))
    return found
