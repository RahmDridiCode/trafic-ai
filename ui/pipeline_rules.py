"""Rules, validation, presets and command building for the traffic pipeline (``src/main.py``).

Everything here is derived from the project: options come from the parser AST, activation rules from
the way the code reads each option, validity from the project's own loaders
(``load_calibration``, ``load_roi_config``) and model checkpoints. Rules that cannot be read
automatically are listed in ``CURATED`` / ``RANGES`` with the source file that justifies them.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import project_scan as ps
from .cli_introspect import CliOption, ParserSpec, argument_usage_guards, extract_parser

MAIN_SCRIPT = ps.PROJECT_ROOT / "src" / "main.py"
OUTPUT_FILES = ("annotated_video.mp4", "vehicle_data.csv", "traffic_summary.json")

# dest -> (category, French label). Options not listed are still shown (category "Autres options").
CURATED: dict[str, tuple[str, str]] = {
    "input": ("1. Entrée / sortie", "Vidéo d'entrée"),
    "output_dir": ("1. Entrée / sortie", "Dossier de sortie"),
    "detector": ("2. Détection des véhicules", "Moteur de détection"),
    "model": ("2. Détection des véhicules", "Modèle YOLO véhicules"),
    "conf": ("2. Détection des véhicules", "Seuil de confiance"),
    "imgsz": ("2. Détection des véhicules", "Taille d'inférence (px)"),
    "device": ("2. Détection des véhicules", "Périphérique de calcul"),
    "min_area": ("2. Détection des véhicules", "Aire minimale MOG2 (px²)"),
    "max_distance": ("3. Suivi (tracker)", "Distance max. d'association (px)"),
    "max_missed": ("3. Suivi (tracker)", "Images manquées tolérées"),
    "iou_threshold": ("3. Suivi (tracker)", "Seuil IoU minimal"),
    "traffic_lights": ("4. Feux rouges", "Détection des feux et franchissement au rouge"),
    "traffic_light_model": ("4. Feux rouges", "Modèle YOLO des feux"),
    "traffic_light_roi": ("4. Feux rouges", "Fichier ROI (zone feu, zone véhicules, ligne d'arrêt)"),
    "traffic_light_conf": ("4. Feux rouges", "Confiance détection des feux"),
    "traffic_light_imgsz": ("4. Feux rouges", "Taille d'inférence des feux (px)"),
    "calibration": ("5. Calibration de vitesse", "Fichier de calibration"),
    "uncalibrated": ("5. Calibration de vitesse", "Forcer le mode sans calibration (pixels/s)"),
    "allow_demo_calibration": ("5. Calibration de vitesse", "Autoriser une calibration de démonstration"),
    "speed_limit": ("6. Excès de vitesse", "Limite de vitesse (km/h)"),
    "speeding_measures": ("6. Excès de vitesse", "Mesures consécutives au-dessus de la limite"),
    "speeding_seconds": ("6. Excès de vitesse", "Durée d'excès continue (s)"),
    "speeding_track_seconds": ("6. Excès de vitesse", "Durée de vie minimale de la trace (s)"),
    "speeding_dedup_seconds": ("6. Excès de vitesse", "Intervalle de déduplication (s)"),
    "speeding_movement_seconds": ("6. Excès de vitesse", "Fenêtre de confirmation du mouvement (s)"),
    "speeding_min_confidence": ("6. Excès de vitesse", "Confiance moyenne minimale"),
    "speeding_margin": ("6. Excès de vitesse", "Marge ajoutée à la limite (fraction)"),
    "speeding_max_plausible_kmh": ("6. Excès de vitesse", "Vitesse max. plausible (km/h)"),
    "speeding_edge_margin": ("6. Excès de vitesse", "Marge aux bords de l'image (px)"),
    "speeding_max_width_jump": ("6. Excès de vitesse", "Variation relative max. de largeur"),
    "skip": ("7. Performance", "Images ignorées entre deux traitées"),
    "max_frames": ("7. Performance", "Limiter le nombre d'images traitées"),
}

# dest -> (minimum, minimum_is_inclusive, maximum or None, justification in the code)
RANGES: dict[str, tuple[float, bool, float | None, str]] = {
    "conf": (0.0, True, 1.0, "src/detector.py: confidence must be within [0, 1]"),
    "imgsz": (0, False, None, "src/detector.py: image_size must be positive"),
    "traffic_light_conf": (0.0, True, 1.0, "src/Feux/traffic_light_detector.py: confidence within [0, 1]"),
    "traffic_light_imgsz": (0, False, None, "src/Feux/traffic_light_detector.py: image size positive"),
    "speed_limit": (0, False, None, "src/speeding.py: limit_kmh must be > 0"),
    "speeding_measures": (1, True, None, "src/speeding.py: min_consecutive must be >= 1"),
    "speeding_seconds": (0, False, None, "src/speed_alert.py: alert durations must be > 0"),
    "speeding_track_seconds": (0, False, None, "src/speed_alert.py: alert durations must be > 0"),
    "speeding_movement_seconds": (0, False, None, "src/speed_alert.py: movement window must be > 0"),
    "speeding_dedup_seconds": (0, True, None, "src/speed_alert.py: dedup duration must be >= 0"),
    "skip": (0, True, None, "src/main.py: frame_step = skip + 1 must be >= 1"),
    "max_frames": (1, True, None, "src/video_processor.py: a limit below 1 would stop at the first frame"),
}

DEVICE_RE = re.compile(r"^(auto|cpu|cuda|mps|\d+(,\d+)*)$", re.IGNORECASE)


@dataclass
class Issue:
    level: str  # "error" | "warning" | "info"
    message: str
    dest: str | None = None
    fix: str | None = None


@dataclass
class Analysis:
    issues: list[Issue] = field(default_factory=list)
    inactive: dict[str, str] = field(default_factory=dict)
    video: ps.VideoInfo | None = None
    calib_mode: str | None = None
    calib_status: str | None = None
    estimated_frames: int | None = None

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors


# ---------------------------------------------------------------- spec access
_SPEC_CACHE: dict[str, tuple[float, ParserSpec, dict[str, str]]] = {}


def load_spec(script: Path = MAIN_SCRIPT) -> tuple[ParserSpec, dict[str, str]]:
    mtime = script.stat().st_mtime
    cached = _SPEC_CACHE.get(str(script))
    if cached and cached[0] == mtime:
        return cached[1], cached[2]
    spec = extract_parser(script)
    guards = argument_usage_guards(script) if script == MAIN_SCRIPT else {}
    _SPEC_CACHE[str(script)] = (mtime, spec, guards)
    return spec, guards


def logical_options(spec: ParserSpec) -> list[list[CliOption]]:
    """One entry per destination (``--x`` / ``--no-x`` pairs are merged), in declaration order."""
    seen: dict[str, list[CliOption]] = {}
    for opt in spec.options:
        seen.setdefault(opt.dest, []).append(opt)
    return list(seen.values())


def category_of(dest: str) -> str:
    if dest in CURATED:
        return CURATED[dest][0]
    if dest.startswith("traffic_light"):
        return "4. Feux rouges"
    if dest.startswith("speed"):
        return "6. Excès de vitesse"
    return "8. Autres options"


def label_of(dest: str, group: list[CliOption]) -> str:
    return CURATED[dest][1] if dest in CURATED else group[0].flag


def default_values(spec: ParserSpec) -> dict[str, Any]:
    return {group[0].dest: group[0].default for group in logical_options(spec)}


def is_bool_pair(group: list[CliOption]) -> bool:
    return group[0].kind == "flag"


# ---------------------------------------------------------------- analysis
def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nearest_existing(path: Path) -> Path:
    for candidate in [path, *path.parents]:
        if candidate.exists():
            return candidate
    return path


def analyze(values: dict[str, Any], spec: ParserSpec | None = None) -> Analysis:
    """Validate a configuration against the real project state before any execution."""
    if spec is None:
        spec, _ = load_spec()
    guards = load_spec()[1]
    result = Analysis()
    add = lambda level, msg, dest=None, fix=None: result.issues.append(Issue(level, msg, dest, fix))  # noqa: E731

    # --- video
    input_value = str(values.get("input") or "").strip()
    if not input_value:
        add("error", "Aucune vidéo sélectionnée.", "input", "Choisissez une vidéo du projet ou saisissez un chemin.")
    else:
        video = ps.video_info(input_value)
        result.video = video
        if video.error:
            add("error", f"Vidéo inutilisable : {video.error}", "input",
                "Choisissez un autre fichier ou ré-encodez la vidéo (H.264/MP4).")
        else:
            total = video.frames
            step = int(values.get("skip") or 0) + 1
            planned = -(-total // step) if total > 0 else 0
            max_frames = values.get("max_frames")
            result.estimated_frames = min(planned, int(max_frames)) if max_frames else planned
            if video.fps <= 0:
                add("warning", "La cadence (FPS) n'est pas lisible ; le pipeline utilisera 30 FPS.", "input")

    video = result.video if result.video and not result.video.error else None

    # --- output directory
    out_value = str(values.get("output_dir") or "").strip()
    if not out_value:
        add("error", "Dossier de sortie vide.", "output_dir")
    else:
        out = ps.resolve_path(out_value)
        if out.exists() and not out.is_dir():
            add("error", f"Le dossier de sortie « {out_value} » est un fichier.", "output_dir")
        else:
            anchor = _nearest_existing(out)
            if not anchor.is_dir() or not os.access(anchor, os.W_OK):
                add("error", f"Impossible d'écrire dans « {anchor} ».", "output_dir")
            existing = [name for name in OUTPUT_FILES if (out / name).exists()] if out.is_dir() else []
            if existing:
                add("warning", f"Des résultats existent déjà dans « {out_value} » et seront écrasés : "
                               f"{', '.join(existing)}.", "output_dir",
                    "Utilisez le nom de dossier suggéré pour conserver l'exécution précédente.")

    # --- detector / model
    detector = values.get("detector")
    if detector == "yolo":
        model_value = str(values.get("model") or "").strip()
        model_path = ps.resolve_path(model_value) if model_value else None
        if not model_value:
            add("error", "Aucun modèle YOLO indiqué.", "model")
        elif not model_path.is_file():
            if Path(model_value).parent == Path(".") and not Path(model_value).is_absolute():
                add("warning", f"« {model_value} » n'est pas un fichier local : Ultralytics tentera de le "
                               "télécharger (connexion Internet requise).", "model")
            else:
                add("error", f"Modèle introuvable : {model_value}", "model",
                    "Sélectionnez un des modèles détectés dans le projet.")
        else:
            info = ps.model_info(model_value, analyse=True)
            if info.error:
                add("error", f"Modèle illisible ({model_value}) : {info.error}", "model")
            elif not ps.is_vehicle_model(info):
                add("warning", f"Aucune classe véhicule connue dans ce modèle (classes : "
                               f"{', '.join(info.class_names()) or 'aucune'}). Le pipeline ne filtrera pas les "
                               "classes et détectera tout ce que le modèle propose.", "model")
    elif detector == "mog2" and values.get("traffic_lights"):
        add("info", "MOG2 détecte des objets en mouvement (classe « moving_object ») : les classes de véhicules "
                    "ne sont pas distinguées dans les résultats.", "detector")

    # --- device
    device = str(values.get("device") or "").strip()
    if not DEVICE_RE.match(device):
        add("error", f"Périphérique invalide : « {device} ».", "device", "Valeurs : auto, cpu, cuda, 0, 0,1 …")
    elif device.lower() not in ("cpu", "auto", "mps") and not ps.cuda_available():
        add("error", f"Périphérique « {device} » demandé mais CUDA n'est pas disponible sur cette machine.",
            "device", "Utilisez « cpu » ou « auto ».")

    # --- traffic lights
    if values.get("traffic_lights"):
        light_value = str(values.get("traffic_light_model") or "").strip()
        light_path = ps.resolve_path(light_value) if light_value else None
        if not light_value or not light_path.is_file():
            add("error", f"Modèle des feux introuvable : {light_value or '(vide)'}", "traffic_light_model",
                "Sélectionnez un modèle compatible (classes red/green/yellow).")
        else:
            info = ps.model_info(light_value, analyse=True)
            if info.error:
                add("error", f"Modèle des feux illisible : {info.error}", "traffic_light_model")
            elif not ps.is_light_model(info):
                add("warning", "Ce modèle ne contient aucune classe de feu reconnue "
                               f"(classes : {', '.join(info.class_names())}).", "traffic_light_model")
        roi_value = str(values.get("traffic_light_roi") or "").strip()
        if not roi_value or not ps.resolve_path(roi_value).is_file():
            add("error", f"Fichier ROI introuvable : {roi_value or '(vide)'}", "traffic_light_roi",
                "Créez-le avec l'outil « Éditeur de ROI » de l'onglet Outils.")
        elif video:
            compat = ps.check_roi(roi_value, video)
            if not compat.ok:
                add("error", f"ROI incompatible avec la vidéo : {compat.message}", "traffic_light_roi",
                    "Choisissez un ROI compatible ou créez-en un avec l'éditeur de ROI.")

    # --- calibration
    uncal = bool(values.get("uncalibrated"))
    calib_value = values.get("calibration")
    if uncal:
        result.calib_mode, result.calib_status = "disabled", "Mode sans calibration forcé (vitesses en pixels/s)."
    elif video:
        default_cal = ps.PROJECT_ROOT / "config" / "calibration.json"
        if not calib_value and not default_cal.is_file():
            result.calib_mode = "disabled"
            result.calib_status = ("Aucune calibration : vitesses en pixels/s "
                                   "(le fichier par défaut config/calibration.json n'existe pas).")
        else:
            path = calib_value or "config/calibration.json"
            compat = ps.check_calibration(path, video, allow_demo=bool(values.get("allow_demo_calibration")))
            if not compat.ok:
                add("error", f"Calibration invalide : {compat.message}", "calibration",
                    "Choisissez une calibration compatible avec la résolution de la vidéo.")
            else:
                result.calib_mode, result.calib_status = compat.mode, compat.message
                if compat.mode == "demo":
                    add("warning", "Calibration de démonstration : les km/h ne sont PAS physiques.", "calibration")
                elif compat.mode == "disabled":
                    add("warning", "Calibration non marquée « real » : elle sera ignorée (vitesses en pixels/s).",
                        "calibration", "Activez « Autoriser une calibration de démonstration » pour l'utiliser "
                                       "à titre indicatif.")

    # --- numeric ranges
    for group in logical_options(spec):
        opt = group[0]
        if opt.kind != "value" or opt.type_name not in ("int", "float"):
            continue
        raw = values.get(opt.dest)
        if raw is None:
            continue
        number = _num(raw)
        if number is None:
            add("error", f"{label_of(opt.dest, group)} : valeur numérique attendue.", opt.dest)
            continue
        if opt.dest in RANGES:
            low, inclusive, high, source = RANGES[opt.dest]
            if number < low or (number == low and not inclusive) or (high is not None and number > high):
                bounds = f"{'≥' if inclusive else '>'} {low:g}" + (f" et ≤ {high:g}" if high is not None else "")
                add("error", f"{label_of(opt.dest, group)} doit être {bounds} ({source}).", opt.dest)
        elif number < 0:
            add("error", f"{label_of(opt.dest, group)} ne peut pas être négatif.", opt.dest)

    # --- inactive options (after the calibration verdict is known)
    result.inactive = inactive_options(spec, values, guards, result.calib_mode, video, calib_value)
    if result.calib_mode in ("disabled", "demo") and not uncal and video:
        add("info", "Les options d'excès de vitesse sont sans effet : seule une calibration « real » "
                    "permet de détecter des infractions (src/video_processor.py).")
    return result


def inactive_options(spec: ParserSpec, values: dict[str, Any], guards: dict[str, str],
                     calib_mode: str | None, video: ps.VideoInfo | None, calib_value: Any) -> dict[str, str]:
    """dest -> reason why the option has no effect (and is therefore not transmitted)."""
    dests = {g[0].dest for g in logical_options(spec)}
    flag_of = {g[0].dest: g[0].flag for g in logical_options(spec)}
    reasons: dict[str, str] = {}
    for dest, guard in guards.items():
        if dest in dests and not values.get(guard):
            reasons[dest] = (f"Sans effet : le code ne lit cette option que lorsque {flag_of.get(guard, guard)} "
                             "est activé (src/main.py).")
    detector = values.get("detector")
    if detector and detector != "yolo":
        for dest in ("model", "conf", "imgsz"):
            reasons.setdefault(dest, "Sans effet avec le détecteur MOG2 (src/detector.py).")
    if detector and detector != "mog2":
        reasons.setdefault("min_area", "Utilisée uniquement par le détecteur MOG2 (src/detector.py).")
    if values.get("uncalibrated"):
        reasons.setdefault("calibration", "Ignorée : « sans calibration » est prioritaire (load_calibration).")
        reasons.setdefault("allow_demo_calibration", "Sans objet en mode sans calibration.")
    elif not calib_value:
        reasons.setdefault("allow_demo_calibration", "Aucune calibration sélectionnée.")
    if calib_mode is not None and calib_mode != "real":
        for dest in dests:
            if category_of(dest).startswith("6."):
                reasons.setdefault(dest, "Sans effet : les excès de vitesse ne sont détectés qu'avec une "
                                         "calibration « real » (src/video_processor.py).")
    return reasons


# ---------------------------------------------------------------- command
def _format(value: Any) -> str:
    return repr(value) if isinstance(value, float) else str(value)


def build_args(spec: ParserSpec, values: dict[str, Any], inactive: dict[str, str] | None = None,
               explicit: bool = False, extra: list[str] | None = None) -> list[str]:
    """CLI arguments (without interpreter / script). Inactive options are never transmitted.

    By default only values that differ from the code's default are passed; ``explicit`` passes every
    active option.
    """
    inactive = inactive or {}
    args: list[str] = []
    for group in logical_options(spec):
        opt = group[0]
        dest = opt.dest
        if dest in inactive or dest not in values:
            continue
        value = values[dest]
        if opt.kind == "flag":
            on = next((o for o in group if o.action == "store_true"), None)
            off = next((o for o in group if o.action == "store_false"), None)
            chosen = on if value else off
            if chosen is not None and (explicit or bool(value) != bool(opt.default)):
                args.append(chosen.flag)
            continue
        if value is None or value == "" or value == []:
            continue
        if opt.default_resolved and not explicit and not opt.required and _same(value, opt.default):
            continue
        items = value if isinstance(value, list) else [value]
        items = [int(v) if opt.type_name == "int" and _num(v) is not None else v for v in items]
        if opt.is_positional:
            args.extend(_format(v) for v in items)
        else:
            args.append(opt.flag)
            args.extend(_format(v) for v in items)
    return args + list(extra or [])


def _same(a: Any, b: Any) -> bool:
    na, nb = _num(a), _num(b)
    if na is not None and nb is not None and not isinstance(a, bool):
        return na == nb
    return str(a).replace("\\", "/") == str(b).replace("\\", "/")


def full_command(script: Path, args: list[str], for_display: bool = False) -> list[str]:
    interpreter = "python" if for_display else sys.executable
    return [interpreter, ps.rel(script) if for_display else str(script), *args]


def display_command(script: Path, args: list[str]) -> str:
    return subprocess.list2cmdline(full_command(script, args, for_display=True))


def check_extra_args(spec: ParserSpec, extra: list[str]) -> list[str]:
    """Flags typed in the free-form advanced box must exist in the parser."""
    known = {f for opt in spec.options for f in opt.flags}
    return [tok for tok in extra if tok.startswith("--") and tok.split("=")[0] not in known]


def parse_extra(text: str) -> list[str]:
    return shlex.split(text, posix=False) if text.strip() else []


def suggest_output_dir(values: dict[str, Any]) -> str:
    video = Path(str(values.get("input") or "video")).stem
    model = Path(str(values.get("model") or "")).stem if values.get("detector") == "yolo" else "mog2"
    tag = "feux" if values.get("traffic_lights") else "sans_feux"
    base = f"outputs/{video}_{model}_{tag}"
    candidate, n = base, 2
    while ps.resolve_path(candidate).exists():
        candidate, n = f"{base}_{n}", n + 1
    return candidate


# ---------------------------------------------------------------- presets
@dataclass
class Preset:
    key: str
    label: str
    description: str
    overrides: dict[str, Any]
    available: bool = True
    reason: str = ""


def build_presets(spec: ParserSpec, values: dict[str, Any], video: ps.VideoInfo | None,
                  calibrations: list[ps.Compat], rois: list[ps.Compat]) -> list[Preset]:
    defaults = default_values(spec)
    best_cal = next((c for c in calibrations if c.ok and c.mode == "real"), None)
    best_roi = next((r for r in rois if r.ok), None)
    models = [ps.model_info(m, analyse=True) for m in ps.scan_models()]
    vehicle_models = [m for m in models if ps.is_vehicle_model(m)]
    light_models = [m for m in models if ps.is_light_model(m)]
    smallest = min(vehicle_models, key=lambda m: m.size_mb, default=None)
    default_light = defaults.get("traffic_light_model")
    light_model = default_light if default_light and ps.resolve_path(default_light).is_file() else (
        light_models[0].path if light_models else None)

    calibration_overrides = ({"calibration": best_cal.path, "uncalibrated": False} if best_cal
                             else {"calibration": None, "uncalibrated": True})
    calibration_note = ("calibration réelle compatible : " + best_cal.path) if best_cal else \
        "aucune calibration « real » compatible avec cette vidéo : vitesses en pixels/s"
    lights_ok = bool(video and best_roi and light_model)
    lights_reason = "" if lights_ok else (
        "Sélectionnez d'abord une vidéo valide." if not video else
        "Aucun fichier ROI compatible avec la résolution de cette vidéo." if not best_roi else
        "Aucun modèle de feux détecté dans le projet.")
    lights = {"traffic_lights": True, "traffic_light_roi": best_roi.path if best_roi else defaults.get(
        "traffic_light_roi"), "traffic_light_model": light_model}

    presets = [
        Preset("defaults", "Valeurs par défaut du code", "Aucun paramètre modifié (hors vidéo).",
               {**defaults, "input": values.get("input"), "output_dir": values.get("output_dir")}),
        Preset("full", "Analyse complète", f"Feux rouges + vitesse ; {calibration_note}.",
               {**calibration_overrides, **lights}, lights_ok, lights_reason),
        Preset("lights", "Analyse avec feux", "Feux rouges activés, vitesses en pixels/s (sans calibration).",
               {**lights, "calibration": None, "uncalibrated": True}, lights_ok, lights_reason),
        Preset("no_lights", "Analyse sans feux", f"Détection des feux désactivée ; {calibration_note}.",
               {"traffic_lights": False, **calibration_overrides}),
    ]
    fast = {"traffic_lights": False, "skip": 2, "uncalibrated": True, "calibration": None}
    if smallest:
        fast.update(detector="yolo", model=smallest.path)
    presets.append(Preset(
        "fast", "Analyse rapide",
        "Sans feux ni calibration, 1 image sur 3 traitée (--skip 2)"
        + (f", plus petit modèle véhicules du projet ({smallest.path}, {smallest.size_mb} Mo)." if smallest else "."),
        fast))
    return presets


def apply_preset(preset: Preset, spec: ParserSpec, current: dict[str, Any]) -> dict[str, Any]:
    merged = {**default_values(spec), "input": current.get("input"), "output_dir": current.get("output_dir")}
    merged.update({k: v for k, v in preset.overrides.items()})
    return merged
