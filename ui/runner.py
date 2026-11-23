"""Background execution of CLI commands with live log capture."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from .project_scan import PROJECT_ROOT, STATE_DIR


class RunHandle:
    def __init__(self, command: list[str], display: str, log_path: Path, meta_path: Path | None = None,
                 meta: dict[str, Any] | None = None):
        self.command, self.display = command, display
        self.log_path, self.meta_path = log_path, meta_path
        self.meta = dict(meta or {})
        self.lines: list[str] = []
        self.started = time.time()
        self.ended: float | None = None
        self.stopped = False
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        self.process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", bufsize=1, env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0)
        self._write_meta(status="running")
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _write_meta(self, **extra: Any) -> None:
        if not self.meta_path:
            return
        data = {**self.meta, "command": self.display, "started": time.strftime("%Y-%m-%d %H:%M:%S",
                time.localtime(self.started)), **extra}
        try:
            self.meta_path.parent.mkdir(parents=True, exist_ok=True)
            self.meta_path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass

    def _pump(self) -> None:
        with open(self.log_path, "w", encoding="utf-8") as log:
            for line in self.process.stdout:
                self.lines.append(line.rstrip("\n"))
                log.write(line)
                log.flush()
        self.process.wait()
        self.ended = time.time()
        status = "arrêtée" if self.stopped else ("terminée" if self.process.returncode == 0 else "échec")
        self._write_meta(status=status, return_code=self.process.returncode,
                         duration_s=round(self.ended - self.started, 1))

    @property
    def running(self) -> bool:
        return self.process.poll() is None or self._thread.is_alive()

    @property
    def return_code(self) -> int | None:
        return self.process.poll()

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def elapsed(self) -> float:
        return (self.ended or time.time()) - self.started

    def stop(self) -> None:
        if self.process.poll() is None:
            self.stopped = True
            self.process.terminate()


def tool_log_path(script: str) -> Path:
    return STATE_DIR / "logs" / f"{Path(script).stem}_{time.strftime('%Y%m%d_%H%M%S')}.log"
