"""Traffic AI - interface adaptative.  Lancer :  streamlit run ui/app.py"""

from __future__ import annotations

import json
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ui import history, log_analysis, pipeline_rules as pr, project_scan as ps, results  # noqa: E402
from ui.cli_introspect import CliOption  # noqa: E402
from ui.runner import RunHandle  # noqa: E402

st.set_page_config(page_title="Traffic AI", page_icon="🚦", layout="wide")

NONE_LABEL = "— aucune (comportement par défaut du code)"
ICONS = {"error": "🔴", "warning": "🟠", "info": "🔵"}


# =============================================================== cached project data
@st.cache_data(show_spinner="Analyse du projet…", ttl=30)
def project_data() -> dict[str, Any]:
    calibrations, rois = ps.scan_json_assets()
    models = [ps.model_info(m, analyse=True) for m in ps.scan_models()]
    return {"videos": ps.scan_videos(), "calibrations": calibrations, "rois": rois, "models": models}


def models_for(kind: str, data: dict[str, Any]) -> list[ps.ModelInfo]:
    keep = ps.is_vehicle_model if kind == "vehicle" else ps.is_light_model
    items = [m for m in data["models"] if not m.error and keep(m)]
    if kind == "vehicle":  # project's own vehicle models first, light models and training checkpoints last
        items.sort(key=lambda m: (m.path.startswith("runs/"), m.path.startswith("models/Feux/"),
                                  not m.path.startswith("models/"), m.path.lower()))
    return items


def model_label(m: ps.ModelInfo) -> str:
    return f"{m.path}  ·  {m.size_mb} Mo"


# =============================================================== state helpers
def key(dest: str, prefix: str = "v") -> str:
    return f"{prefix}_{dest}"


def init_state(spec, prefix: str = "v") -> None:
    for group in pr.logical_options(spec):
        opt = group[0]
        st.session_state.setdefault(key(opt.dest, prefix), _widget_default(opt))
        if opt.kind == "value" and opt.type_name in ("int", "float") and opt.default is None:
            st.session_state.setdefault(key(opt.dest, prefix) + "_on", False)


def _widget_default(opt: CliOption) -> Any:
    if opt.kind == "flag":
        return bool(opt.default)
    if opt.nargs in ("+", "*") or isinstance(opt.default, list):
        return shlex.join(str(x) for x in (opt.default or []))
    if opt.default is None and opt.type_name in ("int", "float"):
        return 1
    if opt.type_name == "int":
        return int(opt.default) if opt.default is not None else 0
    if opt.type_name == "float":
        return float(opt.default) if opt.default is not None else 0.0
    return opt.default


def collect(spec, prefix: str = "v") -> dict[str, Any]:
    values: dict[str, Any] = {}
    for group in pr.logical_options(spec):
        opt = group[0]
        value = st.session_state.get(key(opt.dest, prefix), _widget_default(opt))
        if opt.kind == "value" and opt.type_name in ("int", "float") and opt.default is None:
            value = value if st.session_state.get(key(opt.dest, prefix) + "_on") else None
        elif opt.nargs in ("+", "*"):
            value = shlex.split(value, posix=False) if str(value).strip() else []
        values[opt.dest] = value
    return values


def push_values(spec, values: dict[str, Any], prefix: str = "v") -> None:
    for group in pr.logical_options(spec):
        opt = group[0]
        if opt.dest not in values:
            continue
        value = values[opt.dest]
        if opt.kind == "value" and opt.type_name in ("int", "float") and opt.default is None:
            st.session_state[key(opt.dest, prefix) + "_on"] = value is not None
            if value is not None:
                st.session_state[key(opt.dest, prefix)] = value
        elif opt.nargs in ("+", "*"):
            st.session_state[key(opt.dest, prefix)] = shlex.join(str(x) for x in value) if isinstance(
                value, list) else value
        else:
            st.session_state[key(opt.dest, prefix)] = value


def cb_apply_values(spec, values, auto_reset: bool = True):
    push_values(spec, values)
    if auto_reset:
        st.session_state["_auto_out"] = None


# =============================================================== option widgets
def option_help(opt: CliOption, group: list[CliOption]) -> str:
    flags = " / ".join(o.flag for o in group)
    default = "(aucun)" if opt.default is None else opt.default
    return f"{opt.help}\n\n`{flags}` · défaut du code : `{default}`"


def select_with_values(label, widget_key, options: list[tuple[Any, str]], help_, disabled, allow_none=False):
    values = [o[0] for o in options]
    labels = {o[0]: o[1] for o in options}
    if allow_none:
        values = [None] + values
        labels[None] = NONE_LABEL
    current = st.session_state.get(widget_key)
    if current not in values:
        values.append(current)
        labels[current] = f"{current}  (hors liste)"
    return st.selectbox(label, values, key=widget_key, format_func=lambda v: labels.get(v, str(v)),
                        help=help_, disabled=disabled, accept_new_options=not allow_none)


def render_option(group: list[CliOption], spec, data: dict[str, Any] | None, disabled: bool,
                  prefix: str = "v", analysis: pr.Analysis | None = None) -> None:
    opt = group[0]
    dest, wkey = opt.dest, key(opt.dest, prefix)
    label = pr.label_of(dest, group) if prefix == "v" else opt.flag
    help_ = option_help(opt, group)
    curated = data is not None and prefix == "v"

    if opt.kind == "flag":
        st.toggle(label, key=wkey, help=help_, disabled=disabled)
    elif curated and dest == "input":
        select_with_values(label, wkey, [(v, f"{v}") for v in data["videos"]], help_, disabled)
    elif curated and dest == "model":
        select_with_values(label, wkey, [(m.path, model_label(m)) for m in models_for("vehicle", data)],
                           help_, disabled)
    elif curated and dest == "traffic_light_model":
        select_with_values(label, wkey, [(m.path, model_label(m)) for m in models_for("light", data)],
                           help_, disabled)
    elif curated and dest == "traffic_light_roi":
        video = analysis.video if analysis and analysis.video and not analysis.video.error else None
        if video:
            compat = ps.compatible_rois(video, data["rois"])
            options = [(c.path, f"{'✅' if c.ok else '❌'} {c.path}" + ("" if c.ok else f" — {c.message[:70]}"))
                       for c in compat]
        else:
            options = [(a.path, a.path) for a in data["rois"]]
        select_with_values(label, wkey, options, help_, disabled)
    elif curated and dest == "calibration":
        video = analysis.video if analysis and analysis.video and not analysis.video.error else None
        if video:
            compat = ps.compatible_calibrations(video, data["calibrations"])
            options = [(c.path, f"{'✅' if c.ok else '❌'} {c.path}"
                        + (f" — {c.mode}{' · recommandée' if c.recommended else ''}" if c.ok
                           else f" — {c.message[:70]}")) for c in compat]
        else:
            options = [(a.path, a.path) for a in data["calibrations"]]
        select_with_values(label, wkey, options, help_, disabled, allow_none=True)
    elif opt.choices:
        choices = list(opt.choices)
        if st.session_state.get(wkey) not in choices:
            st.session_state[wkey] = opt.default if opt.default in choices else choices[0]
        st.selectbox(label, choices, key=wkey, help=help_, disabled=disabled)
    elif opt.type_name in ("int", "float"):
        is_int = opt.type_name == "int"
        low = pr.RANGES.get(dest, (0, True, None, ""))
        optional = opt.default is None
        if optional:
            st.checkbox(f"Activer : {label}", key=wkey + "_on", help=help_, disabled=disabled)
        number_args: dict[str, Any] = {
            "min_value": int(low[0]) if is_int else float(low[0]),
            "step": 1 if is_int else (0.05 if abs(float(opt.default or 0)) <= 1 else 1.0),
        }
        if low[2] is not None:
            number_args["max_value"] = int(low[2]) if is_int else float(low[2])
        if not is_int:
            number_args["format"] = "%g"
        st.number_input(label if not optional else "Valeur", key=wkey, help=None if optional else help_,
                        disabled=disabled or (optional and not st.session_state.get(wkey + "_on")), **number_args)
    else:
        st.text_input(label, key=wkey, help=help_, disabled=disabled,
                      placeholder="(aucune valeur : défaut du code)" if opt.default in (None, "") else None)


# =============================================================== run monitoring
def show_diagnosis(text: str, return_code: int | None) -> None:
    diagnoses = log_analysis.analyze_log(text, return_code)
    for d in diagnoses:
        box = st.error if d.level == "error" else st.warning if d.level == "warning" else st.info
        box(f"**{d.title}**\n\n*Cause probable :* {d.cause}\n\n*Solution :* {d.solution}"
            + (f"\n\n`{d.evidence}`" if d.evidence else ""))


def monitor(state_key: str, total_hint: int | None = None) -> None:
    handle: RunHandle | None = st.session_state.get(state_key)
    if handle is None:
        return
    text = handle.text
    progress = log_analysis.parse_progress(text)
    running = handle.running
    head = st.columns([3, 1])
    head[0].markdown(f"**{'⏳ En cours' if running else ('✅ Terminée' if handle.return_code == 0 else '❌ Échec')}**"
                     f" · {handle.elapsed:.0f} s · code retour : `{handle.return_code}`")
    if running and head[1].button("⏹ Arrêter", key=f"stop_{state_key}"):
        handle.stop()
    if progress:
        done, total = progress
        st.progress(min(done / total, 1.0) if total else 0.0, text=f"{done}/{total} images")
    elif running:
        st.progress(0.0, text="Chargement des modèles et de la vidéo…")
    st.code("\n".join(handle.lines[-25:]) or "(en attente de sortie…)", language="text")
    if not running:
        show_diagnosis(text, handle.return_code)
        if not st.session_state.get(f"{state_key}_done"):
            st.session_state[f"{state_key}_done"] = True
            history.update_latest(handle.display, "terminée" if handle.return_code == 0 else "échec")
            st.cache_data.clear()
            st.rerun()


def run_monitor(state_key: str) -> None:
    handle = st.session_state.get(state_key)
    every = 1.0 if handle is not None and handle.running else None
    st.fragment(run_every=every)(monitor)(state_key)


# =============================================================== TAB: configuration
def tab_config(spec, data) -> None:
    init_state(spec)
    values = collect(spec)
    analysis = pr.analyze(values, spec)

    # smart values that follow the video / model / lights selection
    video_changed = st.session_state.get("_last_video") != values["input"]
    if video_changed and analysis.video and not analysis.video.error:
        compat_cal = ps.compatible_calibrations(analysis.video, data["calibrations"])
        best_cal = next((c for c in compat_cal if c.ok and c.recommended and c.mode == "real"),
                        next((c for c in compat_cal if c.ok and c.mode == "real"), None))
        compat_roi = ps.compatible_rois(analysis.video, data["rois"])
        best_roi = next((r for r in compat_roi if r.ok), None)
        st.session_state[key("calibration")] = best_cal.path if best_cal else None
        if best_roi:
            st.session_state[key("traffic_light_roi")] = best_roi.path
        st.session_state["_auto_msg"] = (
            f"Suggestions pour {values['input']} ({analysis.video.resolution}) : calibration = "
            f"{best_cal.path if best_cal else 'aucune compatible'} ; ROI feux = "
            f"{best_roi.path if best_roi else 'aucun compatible'}.")
    st.session_state["_last_video"] = values["input"]
    suggestion = pr.suggest_output_dir(values)
    if st.session_state.get("_auto_out") is None or values["output_dir"] in (None, "", "outputs",
                                                                              st.session_state.get("_auto_out")):
        if values["output_dir"] != suggestion:
            st.session_state[key("output_dir")] = suggestion
            st.session_state["_auto_out"] = suggestion
            st.rerun()
        st.session_state["_auto_out"] = suggestion

    if st.session_state.get("_auto_msg"):
        st.info("💡 " + st.session_state["_auto_msg"])

    # ---- presets
    calib_list = ps.compatible_calibrations(analysis.video, data["calibrations"]) if (
        analysis.video and not analysis.video.error) else []
    roi_list = ps.compatible_rois(analysis.video, data["rois"]) if (
        analysis.video and not analysis.video.error) else []
    presets = pr.build_presets(spec, values, analysis.video if analysis.video and not analysis.video.error else None,
                               calib_list, roi_list)
    st.subheader("Préconfigurations")
    cols = st.columns(len(presets))
    for col, preset in zip(cols, presets):
        col.button(preset.label, key=f"preset_{preset.key}", disabled=not preset.available, width="stretch",
                   help=preset.description if preset.available else preset.reason, on_click=cb_apply_values,
                   args=(spec, pr.apply_preset(preset, spec, values)))
        col.caption(preset.description if preset.available else f"⛔ {preset.reason}")

    # ---- options by category
    st.subheader("Paramètres")
    st.caption("Les paramètres sans effet (d'après le code) sont repliés et ne sont jamais transmis.")
    by_cat: dict[str, list[list[CliOption]]] = {}
    for group in pr.logical_options(spec):
        by_cat.setdefault(pr.category_of(group[0].dest), []).append(group)
    advanced_cats = {"3. Suivi (tracker)", "8. Autres options"}
    for cat in sorted(by_cat):
        groups = by_cat[cat]
        active = [g for g in groups if g[0].dest not in analysis.inactive]
        inactive = [g for g in groups if g[0].dest in analysis.inactive]
        if not active and not inactive:
            continue
        container = st.expander(cat + (" (avancé)" if cat in advanced_cats else ""),
                                expanded=cat not in advanced_cats and cat.startswith(("1.", "2.", "4.", "5.")))
        with container:
            for group in active:
                render_option(group, spec, data, False, analysis=analysis)
                _hints(group[0].dest, values, analysis, data)
            if inactive:
                st.markdown("**Options sans effet dans cette configuration** (non transmises)")
                for group in inactive:
                    st.caption(f"{group[0].flag} — {analysis.inactive[group[0].dest]}")
                    render_option(group, spec, data, True, analysis=analysis)

    # ---- diagnostics
    st.subheader("Vérifications avant lancement")
    values = collect(spec)
    analysis = pr.analyze(values, spec)
    if analysis.video and not analysis.video.error:
        v = analysis.video
        st.caption(f"Vidéo : {v.resolution} · {v.fps:.2f} FPS · {v.frames} images · {v.duration_s:.1f} s · "
                   f"{v.size_mb} Mo" + (f" · ≈ {analysis.estimated_frames} images à traiter"
                                         if analysis.estimated_frames is not None else ""))
    if analysis.calib_status:
        st.caption(f"Calibration : {analysis.calib_status}")
    if not analysis.issues:
        st.success("Aucun problème détecté.")
    for issue in analysis.issues:
        box = st.error if issue.level == "error" else st.warning if issue.level == "warning" else st.info
        box(issue.message + (f"\n\n➡️ {issue.fix}" if issue.fix else ""))

    # ---- command
    st.subheader("Commande générée")
    explicit = st.toggle("Commande explicite (inclure toutes les options actives, même aux valeurs par défaut)")
    extra_text = st.text_input("Arguments supplémentaires (avancé, ajoutés tels quels)", key="v__extra",
                               placeholder="ex. --max-frames 200")
    extra = pr.parse_extra(extra_text)
    unknown = pr.check_extra_args(spec, extra)
    if unknown:
        st.error(f"Option(s) inconnue(s) du pipeline : {', '.join(unknown)}")
    args = pr.build_args(spec, values, analysis.inactive, explicit, extra)
    st.code(pr.display_command(pr.MAIN_SCRIPT, args), language="powershell")

    name = st.text_input("Nom de cette configuration (facultatif)", key="v__name")
    left, right = st.columns(2)
    launch = left.button("🚀 Lancer l'analyse", type="primary", disabled=not analysis.ok or bool(unknown),
                         width="stretch")
    if right.button("💾 Enregistrer la configuration sans lancer", width="stretch"):
        history.save_config(values, pr.display_command(pr.MAIN_SCRIPT, args), name=name or None, status="enregistrée")
        st.toast("Configuration enregistrée")
    if not analysis.ok:
        st.caption("Corrigez les erreurs 🔴 ci-dessus pour activer le lancement.")

    current = st.session_state.get("run")
    if launch:
        if current is not None and current.running:
            st.warning("Une analyse est déjà en cours.")
        else:
            out_dir = ps.resolve_path(values["output_dir"])
            command = pr.full_command(pr.MAIN_SCRIPT, args)
            display = pr.display_command(pr.MAIN_SCRIPT, args)
            meta = {"values": values, "name": name or None, "lights": bool(values["traffic_lights"])}
            history.save_config(values, display, name=name or None, status="en cours", run_dir=ps.rel(out_dir))
            st.session_state["run"] = RunHandle(command, display, out_dir / results.LOG_FILE,
                                                out_dir / results.META_FILE, meta)
            st.session_state["run_done"] = False
            st.rerun()
    if st.session_state.get("run") is not None:
        st.subheader("Suivi de l'exécution")
        run_monitor("run")


def _hints(dest: str, values, analysis: pr.Analysis, data) -> None:
    """Context hints taken from real project data (never invented)."""
    if dest == "device":
        st.caption("CUDA détecté : « auto » ou « cuda » possibles." if ps.cuda_available()
                   else "Aucun GPU CUDA détecté par PyTorch : seul « cpu » (ou « auto ») fonctionnera.")
    elif dest == "imgsz" and values.get("detector") == "yolo" and values.get("model"):
        info = ps.model_info(str(values["model"]), analyse=False)
        if info.train_imgsz and info.train_imgsz != values.get("imgsz"):
            st.button(f"Ce modèle a été entraîné à {info.train_imgsz}px → utiliser {info.train_imgsz}",
                      key="hint_imgsz", on_click=lambda v=info.train_imgsz: st.session_state.update({key("imgsz"): v}))
    elif dest == "traffic_light_roi" and analysis.video and not analysis.video.error:
        compat = ps.compatible_rois(analysis.video, data["rois"])
        if not any(c.ok for c in compat):
            st.warning(f"Aucun ROI du projet n'est compatible avec la résolution {analysis.video.resolution}. "
                       "Créez-en un dans l'onglet Outils → « Éditeur de ROI ».")
    elif dest == "calibration" and analysis.video and not analysis.video.error:
        compat = ps.compatible_calibrations(analysis.video, data["calibrations"])
        if not any(c.ok for c in compat):
            st.warning(f"Aucune calibration du projet ne correspond à {analysis.video.resolution} : "
                       "les vitesses seront en pixels/s.")


# =============================================================== TAB: history
def tab_history(spec) -> None:
    entries = history.load_history()
    if not entries:
        st.info("Aucune configuration mémorisée pour l'instant.")
        return
    for i, entry in enumerate(entries):
        title = f"{entry['timestamp']} · {entry.get('name') or Path(str(entry['values'].get('input', ''))).name}" \
                f" · {entry.get('status', '')}"
        with st.expander(title):
            st.code(entry.get("command", ""), language="powershell")
            if entry.get("script", "src/main.py") == "src/main.py":
                st.button("Recharger dans la configuration (puis modifiable)", key=f"hist_{i}",
                          on_click=cb_apply_values, args=(spec, entry["values"], False))
                if entry.get("run_dir"):
                    st.caption(f"Dossier de résultats : {entry['run_dir']}")
            else:
                st.caption("Commande d'un outil : relancez-la depuis l'onglet Outils.")


# =============================================================== TAB: results
def run_label(run: results.Run) -> str:
    lights = {True: "avec feux", False: "sans feux", None: "?"}[run.lights_enabled]
    stamp = time.strftime("%d/%m %H:%M", time.localtime((run.path / "traffic_summary.json").stat().st_mtime))
    return f"{run.name} — {lights} — {stamp}"


def load_all_runs(root: str) -> list[results.Run]:
    return [r for r in (results.load_run(p) for p in ps.scan_runs(root)) if r is not None]


def tab_results() -> None:
    root = st.text_input("Dossier racine des résultats", "outputs", key="res_root")
    runs = load_all_runs(root)
    if not runs:
        st.info("Aucune exécution trouvée (un dossier doit contenir traffic_summary.json).")
        return
    latest = st.session_state.get("run")
    default_index = 0
    if latest is not None and latest.meta_path:
        for i, r in enumerate(runs):
            if r.path == latest.meta_path.parent:
                default_index = i
    run = st.selectbox("Exécution", runs, index=default_index, format_func=run_label)
    s = run.summary
    badge = []
    badge.append("🚦 feux activés" if run.lights_enabled else "🚦 feux désactivés" if run.lights_enabled is False
                 else "🚦 feux : inconnu (ancienne exécution)")
    badge.append(f"calibration : {s.get('calibration_mode', '?')}")
    badge.append(f"résolution : {s.get('input_resolution', '?')} @ {s.get('video_fps', '?')} FPS")
    badge.append(f"--skip : {s.get('frame_skip', '?')}")
    st.caption(" · ".join(badge))
    if run.meta.get("command"):
        st.code(run.meta["command"], language="powershell")
    if not s.get("speed_estimate_is_physical", False):
        st.warning(f"Les vitesses sont exprimées en {s.get('speed_unit', 'unité inconnue')} : "
                   "elles ne sont pas des km/h physiques. " + s.get("speed_estimate_note", ""))

    stats = results.headline_stats(run)
    for start in range(0, len(stats), 4):
        cols = st.columns(4)
        for col, (label, value) in zip(cols, stats[start:start + 4]):
            col.metric(label, value)

    left, right = st.columns(2)
    classes = s.get("vehicle_detections_by_class")
    if classes:
        left.markdown("**Détections par classe**")
        left.bar_chart(pd.Series(classes, name="détections"))
    reasons = (s.get("speed_alerts") or {}).get("diagnostic_reason_counts")
    if reasons:
        right.markdown("**Raisons de rejet des alertes de vitesse** (alert_diagnostic.csv)")
        right.bar_chart(pd.Series(reasons, name="occurrences"))

    if run.lights_enabled:
        tl = s.get("traffic_lights", {})
        st.markdown("### 🚦 Feux rouges")
        c1, c2 = st.columns(2)
        c1.metric("Infractions feu rouge", tl.get("red_light_violation_count", 0))
        c1.caption(f"Modèle : {tl.get('model')}  \nROI : {tl.get('roi_config')}")
        counts = results.light_state_counts(run)
        if counts is not None:
            c2.markdown("**État du feu (images avec au moins un véhicule)**")
            c2.bar_chart(counts)

    ev = results.events(run)
    st.markdown("### Événements détectés")
    if ev.empty:
        st.info("Aucun événement (excès de vitesse, alerte, feu rouge) dans cette exécution.")
    else:
        kinds = st.multiselect("Types", sorted(ev["type"].unique()), default=sorted(ev["type"].unique()))
        shown = ev[ev["type"].isin(kinds)]
        st.dataframe(shown, hide_index=True, width="stretch")
        timed = shown.dropna(subset=["temps_s"])
        if not timed.empty:
            timed = timed.assign(seconde=(timed["temps_s"] // 5 * 5).astype(int))
            st.markdown("**Chronologie (événements par tranche de 5 s)**")
            st.bar_chart(timed.pivot_table(index="seconde", columns="type", values="id", aggfunc="count").fillna(0))
        st.download_button("Télécharger les événements (CSV)", shown.to_csv(index=False).encode("utf-8"),
                           file_name=f"{run.name}_evenements.csv")

    for folder, title in (("red_light_violations", "Captures feu rouge"), ("speed_alerts", "Captures alertes vitesse")):
        images = run.images(folder)
        if images:
            with st.expander(f"{title} ({len(images)})"):
                cols = st.columns(3)
                for i, img in enumerate(images[:30]):
                    cols[i % 3].image(str(img), caption=img.stem)

    with st.expander("Journal d'exécution"):
        log = run.path / results.LOG_FILE
        if log.is_file():
            text = log.read_text(encoding="utf-8", errors="replace")
            show_diagnosis(text, run.meta.get("return_code"))
            st.code(text[-6000:], language="text")
        else:
            st.caption("Pas de run.log : exécution lancée hors de l'interface.")
    with st.expander("Fichiers produits"):
        rows = [{"fichier": p.name, "taille (Mo)": round(p.stat().st_size / 1e6, 3)}
                for p in sorted(run.path.iterdir()) if p.is_file()]
        st.dataframe(pd.DataFrame(rows).astype(str), hide_index=True)
    if run.video.is_file() and st.checkbox("Charger la vidéo annotée "
                                           f"({run.video.stat().st_size / 1e6:.0f} Mo)", key="load_video"):
        st.video(str(run.video))
    with st.expander("traffic_summary.json (brut)"):
        st.json(s, expanded=False)


# =============================================================== TAB: comparison
def tab_compare() -> None:
    runs = load_all_runs(st.session_state.get("res_root", "outputs"))
    if len(runs) < 2:
        st.info("Il faut au moins deux exécutions pour comparer.")
        return

    def signature(r: results.Run):
        return (r.summary.get("input_resolution"), r.summary.get("video_fps"), r.summary.get("frame_skip"))

    pairs = []
    for i, a in enumerate(runs):
        for b in runs[i + 1:]:
            if signature(a) == signature(b) and a.lights_enabled != b.lights_enabled \
                    and None not in (a.lights_enabled, b.lights_enabled):
                pairs.append((b, a) if a.lights_enabled else (a, b))
    st.caption("Paires suggérées : même résolution/FPS/--skip, l'une avec et l'autre sans détection des feux "
               "(A = sans feux, B = avec feux).")
    if pairs:
        pair = st.selectbox("Paire suggérée", pairs, index=None, placeholder="Choisir…",
                            format_func=lambda p: f"{p[0].name}  ↔  {p[1].name}")
    else:
        pair = None
        st.caption("Aucune paire détectée automatiquement.")
    c1, c2 = st.columns(2)
    ia = runs.index(pair[0]) if pair else 0
    ib = runs.index(pair[1]) if pair else 1
    a = c1.selectbox("Exécution A", runs, index=ia, format_func=run_label, key=f"cmp_a_{ia}")
    b = c2.selectbox("Exécution B", runs, index=ib, format_func=run_label, key=f"cmp_b_{ib}")
    if a.path == b.path:
        st.warning("Choisissez deux exécutions différentes.")
        return
    cmp = results.compare_runs(a, b)
    for w in cmp.warnings:
        st.warning(w)
    st.markdown("### Configuration")
    st.dataframe(cmp.config.astype(str), hide_index=True, width="stretch")
    st.markdown("### Indicateurs")
    only_diff = st.checkbox("Afficher uniquement les écarts", value=True)
    metrics = cmp.metrics
    if only_diff:
        metrics = metrics[metrics["A"].astype(str) != metrics["B"].astype(str)]
    st.dataframe(metrics.astype(str), hide_index=True, width="stretch")
    if not cmp.class_counts.empty:
        st.markdown("### Détections par classe")
        st.bar_chart(cmp.class_counts.set_index("classe"))
    if cmp.red_light_overlap is not None:
        st.markdown("### Infractions feu rouge (par identifiant de trace)")
        st.json(cmp.red_light_overlap)
        st.caption("Les identifiants de trace ne sont comparables que si le détecteur et le tracker sont "
                   "identiques dans les deux exécutions.")
    st.download_button("Télécharger la comparaison (CSV)", cmp.metrics.to_csv(index=False).encode("utf-8"),
                       file_name=f"comparaison_{a.name}_vs_{b.name}.csv")


# =============================================================== TAB: options coverage
def tab_options(spec, guards) -> None:
    values = collect(spec)
    analysis = pr.analyze(values, spec)
    args = pr.build_args(spec, values, analysis.inactive, explicit=True)
    transmitted = {a for a in args if a.startswith("--")}
    rows = []
    for group in pr.logical_options(spec):
        opt = group[0]
        rows.append({
            "option": " / ".join(o.flag for o in group), "type": "booléen" if opt.kind == "flag" else opt.type_name,
            "défaut du code": opt.default, "choix": ", ".join(map(str, opt.choices)) if opt.choices else "",
            "catégorie": pr.category_of(opt.dest),
            "état actuel": ("sans effet — " + analysis.inactive[opt.dest]) if opt.dest in analysis.inactive else "actif",
            "transmis": any(o.flag in transmitted for o in group), "aide": opt.help,
        })
    st.markdown(f"**{len(spec.options)} options** détectées dans `src/main.py` (analyse statique du parser) ; "
                "toutes sont accessibles dans l'onglet Configuration.")
    st.dataframe(pd.DataFrame(rows).astype(str), hide_index=True, width="stretch")
    st.markdown("**Dépendances détectées automatiquement dans le code** (options lues uniquement dans un bloc "
                "`if args.X:`)")
    st.json(guards)


# =============================================================== main
def main() -> None:
    spec, guards = pr.load_spec()
    data = project_data()
    st.title("🚦 Traffic AI — interface intelligente")
    if spec.error:
        st.error(f"Impossible d'analyser src/main.py : {spec.error}")
        return
    cols = st.columns(4)
    cols[0].metric("Vidéos", len(data["videos"]))
    cols[1].metric("Modèles (.pt)", len(data["models"]))
    cols[2].metric("Calibrations", len(data["calibrations"]))
    cols[3].metric("ROI feux", len(data["rois"]))
    tabs = st.tabs(["⚙️ Configuration & lancement", "🕘 Historique", "📊 Résultats", "⚖️ Comparaison",
                    "📋 Options CLI"])
    with tabs[0]:
        tab_config(spec, data)
    with tabs[1]:
        tab_history(spec)
    with tabs[2]:
        tab_results()
    with tabs[3]:
        tab_compare()
    with tabs[4]:
        tab_options(spec, guards)


main()
