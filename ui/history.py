"""Persistence of the last configurations (``.ui_state/history.json``)."""

from __future__ import annotations

import json
import time
from typing import Any

from .project_scan import STATE_DIR

HISTORY_FILE = STATE_DIR / "history.json"
MAX_ENTRIES = 50


def load_history() -> list[dict[str, Any]]:
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, UnicodeError, json.JSONDecodeError):
        return []


def _write(entries: list[dict[str, Any]]) -> None:
    STATE_DIR.mkdir(exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(entries[:MAX_ENTRIES], indent=1, ensure_ascii=False), encoding="utf-8")


def save_config(values: dict[str, Any], command: str, script: str = "src/main.py", name: str | None = None,
                status: str = "lancée", run_dir: str | None = None) -> dict[str, Any]:
    entry = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "name": name, "script": script, "values": values,
             "command": command, "status": status, "run_dir": run_dir}
    _write([entry, *load_history()])
    return entry


def update_latest(command: str, status: str) -> None:
    entries = load_history()
    for entry in entries:
        if entry.get("command") == command:
            entry["status"] = status
            break
    _write(entries)
