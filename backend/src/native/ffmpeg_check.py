"""Make sure ``clipper`` runs an ffmpeg that can burn in captions.

Captions and hook titles are drawn with ffmpeg's ``subtitles`` filter, which
needs libass. Homebrew's ``ffmpeg`` formula is built without libass, so on a
Mac every render fails with "No such filter: 'subtitles'". Homebrew ships the
full build as ``ffmpeg-full``, but keg-only: it is not on PATH.

Every part of the pipeline calls ``ffmpeg``/``ffprobe`` by name, so when the
ffmpeg on PATH lacks the filter and a full build is installed, its ``bin``
directory is put first on this process's PATH (child processes inherit it).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional

FULL_BUILD_DIRS = (
    "/opt/homebrew/opt/ffmpeg-full/bin",  # Homebrew on Apple Silicon
    "/usr/local/opt/ffmpeg-full/bin",  # Homebrew on Intel Macs
)

MISSING_FILTER_HELP = (
    "Your ffmpeg can't draw captions (it was built without libass, so the "
    "'subtitles' filter is missing). On a Mac run: brew install ffmpeg-full "
    "(clipper finds it automatically). Elsewhere, install an ffmpeg built with "
    "libass, or set CLIPPER_FFMPEG_DIR to the folder that holds one."
)


def has_subtitles_filter(ffmpeg: Optional[str]) -> bool:
    """True when this ffmpeg binary lists the libass ``subtitles`` filter."""
    if not ffmpeg:
        return False
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return any(
        len(parts) >= 2 and parts[1] == "subtitles"
        for parts in (line.split() for line in result.stdout.splitlines())
    )


def _candidate_dirs() -> Iterable[str]:
    configured = os.getenv("CLIPPER_FFMPEG_DIR", "").strip()
    if configured:
        yield str(Path(configured).expanduser())
    yield from FULL_BUILD_DIRS


def ensure_caption_ffmpeg() -> bool:
    """Put an ffmpeg with the ``subtitles`` filter first on PATH if one exists.

    Returns True when the ffmpeg on PATH (possibly after the switch) can burn
    captions, False otherwise.
    """
    for directory in _candidate_dirs():
        candidate = Path(directory) / "ffmpeg"
        if candidate.is_file() and os.access(candidate, os.X_OK) and has_subtitles_filter(str(candidate)):
            current = shutil.which("ffmpeg")
            if not current or Path(current).resolve() != candidate.resolve():
                os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")
            return True
    return has_subtitles_filter(shutil.which("ffmpeg"))
