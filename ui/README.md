# Interface Traffic AI

```
.\.venv\Scripts\python.exe -m pip install -r requirements-ui.txt
.\.venv\Scripts\python.exe -m streamlit run ui\app.py
```

- Les options CLI sont lues par analyse AST des `argparse` du projet (`src/main.py`, `predict.py`, `src/Feux/05_roi_config.py`, `scripts/*.py`) : toutes sont exposées (onglet « Options CLI » = table de couverture).
- Les options sans effet (ex. ROI/modèle feux quand les feux sont désactivés) ne sont jamais transmises.
- Fichiers ROI/calibration validés avec `load_roi_config` / `load_calibration` du projet.
- Mémoire des configurations et métadonnées dans `.ui_state/` ; chaque exécution écrit `ui_run.json` et `run.log` dans son dossier de sortie.
- Limite technique : `05_roi_config.py` ouvre une fenêtre OpenCV ; l'interface en fournit la commande à lancer.
- Tests : `python -m unittest tests.test_ui`.
