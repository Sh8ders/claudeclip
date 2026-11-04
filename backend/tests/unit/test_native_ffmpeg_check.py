"""clipper switches to an ffmpeg that can burn captions, or stops with a clear fix.

Homebrew's ``ffmpeg`` is built without libass, so its ``subtitles`` filter is
missing and every render fails. Homebrew's keg-only ``ffmpeg-full`` has it.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from src.native import cli, ffmpeg_check


def _fake_ffmpeg(directory: Path, with_subtitles: bool) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    filters = [" ... scale             V->V       Scale the input video size."]
    if with_subtitles:
        filters.append(" ... subtitles         V->V       Render text subtitles onto input video using the libass library.")
    script = directory / "ffmpeg"
    script.write_text("#!/bin/sh\ncat <<'EOF'\nFilters:\n" + "\n".join(filters) + "\nEOF\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return script


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(ffmpeg_check, "FULL_BUILD_DIRS", ())
    monkeypatch.delenv("CLIPPER_FFMPEG_DIR", raising=False)
    return tmp_path


def test_detects_the_subtitles_filter(isolated):
    assert ffmpeg_check.has_subtitles_filter(str(_fake_ffmpeg(isolated / "full", True)))
    assert not ffmpeg_check.has_subtitles_filter(str(_fake_ffmpeg(isolated / "slim", False)))
    assert not ffmpeg_check.has_subtitles_filter(None)
    assert not ffmpeg_check.has_subtitles_filter(str(isolated / "missing" / "ffmpeg"))


def test_switches_path_to_a_full_build(isolated, monkeypatch):
    slim = _fake_ffmpeg(isolated / "slim", False).parent
    full = _fake_ffmpeg(isolated / "full", True).parent
    monkeypatch.setenv("PATH", str(slim) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("CLIPPER_FFMPEG_DIR", str(full))

    assert ffmpeg_check.ensure_caption_ffmpeg() is True
    assert os.environ["PATH"].split(os.pathsep)[0] == str(full)


def test_slim_ffmpeg_without_a_full_build_is_reported(isolated, monkeypatch):
    slim = _fake_ffmpeg(isolated / "slim", False).parent
    monkeypatch.setenv("PATH", str(slim) + os.pathsep + os.environ["PATH"])

    assert ffmpeg_check.ensure_caption_ffmpeg() is False


def test_capable_ffmpeg_on_path_is_left_alone(isolated, monkeypatch):
    full = _fake_ffmpeg(isolated / "full", True).parent
    path = str(full) + os.pathsep + os.environ["PATH"]
    monkeypatch.setenv("PATH", path)

    assert ffmpeg_check.ensure_caption_ffmpeg() is True
    assert os.environ["PATH"] == path


def test_render_stops_before_any_work_without_caption_support(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(ffmpeg_check, "ensure_caption_ffmpeg", lambda: False)
    monkeypatch.setenv("TEMP_DIR", str(tmp_path / "cache"))

    code = cli.main(["--runs-dir", str(tmp_path / "runs"), "render", "nothing-here"])

    assert code == 1
    assert "brew install ffmpeg-full" in capsys.readouterr().err


def test_schema_needs_no_ffmpeg(monkeypatch, tmp_path):
    def fail():
        raise AssertionError("schema must not probe ffmpeg")

    monkeypatch.setattr(ffmpeg_check, "ensure_caption_ffmpeg", fail)
    assert cli.main(["--runs-dir", str(tmp_path / "runs"), "schema", "selection"]) == 0
