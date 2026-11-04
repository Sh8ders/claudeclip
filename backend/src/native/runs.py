"""Run folders: one directory per source video under ``runs/<name>/``."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_WHISPER_MODEL = "small"
DEFAULT_WHISPER_BACKEND = "auto"


def repo_root() -> Optional[Path]:
    """The checkout this package lives in (``<repo>/backend/src/native``)."""
    candidate = Path(__file__).resolve().parents[3]
    if (candidate / "backend").is_dir():
        return candidate
    return None


def default_runs_dir() -> Path:
    configured = os.getenv("CLIPPER_RUNS_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    root = repo_root()
    return (root / "runs") if root else (Path.cwd() / "runs").resolve()


def default_broll_dir() -> Optional[Path]:
    configured = os.getenv("CLIPPER_BROLL_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    root = repo_root()
    return (root / "broll") if root else None


def default_whisper_model() -> str:
    return os.getenv("CLIPPER_WHISPER_MODEL") or DEFAULT_WHISPER_MODEL


def default_whisper_backend() -> str:
    return os.getenv("CLIPPER_WHISPER_BACKEND") or DEFAULT_WHISPER_BACKEND


def configure_environment(runs_dir: Path) -> None:
    """Process-wide settings for native runs. Explicit environment wins."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TEMP_DIR", str(runs_dir / ".cache"))
    Path(os.environ["TEMP_DIR"]).mkdir(parents=True, exist_ok=True)
    # The web pipeline's provider switch; native mode always transcribes locally.
    os.environ.setdefault("TRANSCRIPTION_PROVIDER", "whisper")


def slugify(value: str, fallback: str = "video") -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value or "").strip("-._")
    return slug[:80] or fallback


def resolve_run_dir(run: str, runs_dir: Optional[Path] = None) -> Path:
    """Accept a run name or a path to a run folder."""
    candidate = Path(run).expanduser()
    if candidate.is_dir() and (candidate / "meta.json").is_file():
        return candidate.resolve()
    folder = (runs_dir or default_runs_dir()) / run
    if not (folder / "meta.json").is_file():
        raise FileNotFoundError(
            f"No run named {run!r} (looked for {folder / 'meta.json'}). Run `clipper prepare` first."
        )
    return folder.resolve()


@dataclass
class Run:
    path: Path

    @property
    def meta_path(self) -> Path:
        return self.path / "meta.json"

    def meta(self) -> Dict[str, Any]:
        return json.loads(self.meta_path.read_text())

    def write_meta(self, meta: Dict[str, Any]) -> None:
        write_json(self.meta_path, meta)

    @property
    def video(self) -> Path:
        return self.path / self.meta()["video"]

    @property
    def transcript_path(self) -> Path:
        return self.path / "transcript.txt"

    @property
    def signals_path(self) -> Path:
        return self.path / "signals.txt"

    @property
    def selection_path(self) -> Path:
        return self.path / "selection.json"

    @property
    def segments_path(self) -> Path:
        return self.path / "segments.json"

    @property
    def report_path(self) -> Path:
        return self.path / "validate_report.txt"

    @property
    def clips_dir(self) -> Path:
        return self.path / "clips"

    @property
    def clips_json(self) -> Path:
        return self.path / "clips.json"

    @property
    def broll_json(self) -> Path:
        return self.path / "broll.json"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def mmss(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"
