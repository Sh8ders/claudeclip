"""Claude-native mode end to end, with every AI SDK and the network blocked.

Each ``clipper`` command runs in its own Python process behind an import
guard: importing pydantic_ai, anthropic, openai, assemblyai (or another
hosted-model SDK) is recorded and fails, and any non-local socket connection
is refused. The run must still produce playable 9:16 clips with burned
captions, hook titles and local B-roll.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND / "scripts"))

import make_native_fixture as fixture  # noqa: E402

FORBIDDEN = (
    "pydantic_ai", "anthropic", "openai", "assemblyai", "google.genai",
    "google.generativeai", "groq", "mistralai", "cohere", "xai_sdk",
)

GUARD = textwrap.dedent(
    """
    import atexit, importlib.abc, json, os, socket, sys

    FORBIDDEN = {forbidden!r}
    attempts = []

    class Guard(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if any(name == f or name.startswith(f + ".") for f in FORBIDDEN):
                attempts.append(name)
                raise ImportError("Claude-native mode must not import " + name)
            return None

    sys.meta_path.insert(0, Guard())

    LOCAL = ("127.0.0.1", "::1", "localhost", "0.0.0.0")
    def _refuse(host):
        return isinstance(host, str) and host not in LOCAL and not host.startswith("/")
    _connect = socket.socket.connect
    def connect(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if _refuse(host):
            attempts.append("network:" + str(host))
            raise OSError("network blocked in Claude-native test: " + str(address))
        return _connect(self, address)
    socket.socket.connect = connect
    _getaddrinfo = socket.getaddrinfo
    def getaddrinfo(host, *args, **kwargs):
        if _refuse(host):
            attempts.append("dns:" + str(host))
            raise OSError("network blocked in Claude-native test: " + str(host))
        return _getaddrinfo(host, *args, **kwargs)
    socket.getaddrinfo = getaddrinfo

    @atexit.register
    def report():
        with open(os.environ["GUARD_REPORT"], "a") as handle:
            for name in attempts:
                handle.write(name + "\\n")

    sys.path.insert(0, {backend!r})
    from src.native.cli import main
    code = main(sys.argv[1:])
    sys.modules_loaded = [m for m in sys.modules if any(m == f or m.startswith(f + ".") for f in FORBIDDEN)]
    if sys.modules_loaded:
        attempts.extend("loaded:" + m for m in sys.modules_loaded)
    sys.exit(code)
    """
)


def _clean_env(tmp_path: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY") and k not in {"LLM", "RERANK_LLM"}}
    env.update(TEMP_DIR=str(tmp_path / "cache"), GUARD_REPORT=str(tmp_path / "guard.txt"), PEXELS_API_KEY="")
    env.pop("CLIPPER_RUNS_DIR", None)
    return env


def _clipper(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    guard = tmp_path / "guarded_clipper.py"
    if not guard.exists():
        guard.write_text(GUARD.format(forbidden=FORBIDDEN, backend=str(BACKEND)))
    return subprocess.run(
        [sys.executable, str(guard), "--runs-dir", str(tmp_path / "runs"), *args],
        capture_output=True, text=True, env=_clean_env(tmp_path), timeout=900,
    )


def _probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,width,height",
         "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)


def _white_share(frame: np.ndarray, top: float, bottom: float) -> float:
    """Share of near-white pixels (caption/title text) in a horizontal band."""
    height = frame.shape[0]
    band = frame[int(height * top): int(height * bottom)]
    white = (band[..., 0] > 215) & (band[..., 1] > 215) & (band[..., 2] > 215)
    return float(white.mean())


def _frame(path: Path, second: float) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(second), "-i", str(path), "-frames:v", "1",
         "-vf", "scale=216:384", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(384, 216, 3)


def test_native_package_has_no_ai_sdk_imports():
    """Static check: no module under src/native imports a hosted-model SDK."""
    for path in sorted((BACKEND / "src" / "native").glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            elif isinstance(node, ast.ImportFrom) and node.level and node.module:
                names = [node.module]
            for name in names:
                assert not any(name == f or name.startswith(f + ".") for f in FORBIDDEN), (path, name)
                assert name not in {"ai", "services.social_caption_service"}, (path, name)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_full_native_run_without_ai_sdks_or_network(tmp_path):
    files = fixture.build(tmp_path / "fixture")

    result = _clipper(tmp_path, "prepare", str(files["video"]), "--name", "demo",
                      "--transcript-json", str(files["transcript"]))
    assert result.returncode == 0, result.stderr
    run = tmp_path / "runs" / "demo"

    # Claude's picks: two short clips on exact transcript lines.
    picks = fixture.selection()
    picks["most_relevant_segments"] = picks["most_relevant_segments"][:2]
    picks["most_relevant_segments"][0].update(start_time="00:00", end_time="00:18")
    picks["most_relevant_segments"][1].update(start_time="00:36", end_time="00:54")
    (run / "selection.json").write_text(json.dumps(picks))

    result = _clipper(tmp_path, "validate", "demo")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Status: OK" in result.stdout

    result = _clipper(tmp_path, "broll-context", "demo")
    assert result.returncode == 0, result.stderr
    assert (run / "broll" / "01.md").is_file()

    # Claude's B-roll cue, and a local footage library matched by filename.
    (run / "broll.json").write_text(json.dumps({
        "1": {"cues": [{"start": 6.2, "duration": 3.0, "query": "hands typing laptop", "fallback_query": "laptop"}]},
        "2": {"cues": []},
    }))
    library = tmp_path / "library"
    library.mkdir()
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "mandelbrot=size=540x960:rate=25",
         "-t", "5", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast",
         str(library / "laptop-typing-desk.mp4")],
        check=True,
    )

    for kind in ("selection", "ranking", "broll", "edit"):
        assert _clipper(tmp_path, "schema", kind).returncode == 0

    result = _clipper(tmp_path, "render", "demo", "--broll", "--broll-dir", str(library), "--template", "hormozi")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "| 1 | 00:00-00:18 |" in result.stdout

    guard_report = tmp_path / "guard.txt"
    attempts = guard_report.read_text().split() if guard_report.exists() else []
    assert attempts == [], f"Claude-native mode reached for an AI SDK or the network: {attempts}"

    clips = json.loads((run / "clips.json").read_text())["clips"]
    assert [clip["clip"] for clip in clips] == [1, 2]
    for clip in clips:
        path = run / clip["file"]
        probe = _probe(path)
        video = next(s for s in probe["streams"] if s["codec_type"] == "video")
        assert (video["codec_name"], video["width"], video["height"]) == ("h264", 1080, 1920)
        assert any(s["codec_type"] == "audio" for s in probe["streams"])
        duration = float(probe["format"]["duration"])
        assert duration == pytest.approx(clip["duration"], abs=0.15)
        assert 15 <= duration <= 21
        assert clip["hook_title"] and clip["virality_score"] > 0

        # Burned text: the hook title near the top while it shows, captions
        # in the lower band. The fixture's background has no white at all.
        early = _frame(path, 1.0)
        assert _white_share(early, 0.03, 0.30) > 0.004, "hook title not visible"
        assert _white_share(early, 0.66, 0.90) > 0.002, "captions not visible"

    assert clips[0]["broll"] and clips[0]["broll"][0]["provider"] == "local"
    assert clips[0]["broll"][0]["local_path"].endswith("laptop-typing-desk.mp4")
    assert clips[1]["broll"] == []

    # Phase 2: the studio's AI edit, with Claude writing the EditPlan.
    result = _clipper(tmp_path, "edit-context", "demo", "1", "make the captions yellow and cut the first sentence")
    assert result.returncode == 0, result.stderr
    assert "You are the editing assistant inside SupoClip's clip editor" in result.stdout
    assert "Request:\nmake the captions yellow and cut the first sentence" in result.stdout
    word_lines = [line.split() for line in result.stdout.splitlines() if line.startswith("word-")]
    second_sentence = next(float(parts[1]) for parts in word_lines if parts[-1] == "I")
    ops = tmp_path / "ops.json"
    ops.write_text(json.dumps({
        "summary": "Made the captions yellow and cut the first sentence.",
        "operations": [
            {"type": "set", "setting": "captions.color", "value": "#FACC15"},
            {"type": "cut_ranges", "ranges": [{"start": 0.0, "end": second_sentence}]},
            {"type": "set", "setting": "captions.font", "value": "Comic Sans"},
        ],
    }))
    result = _clipper(tmp_path, "edit-apply", "demo", "1", str(ops))
    assert result.returncode == 0, result.stdout + result.stderr
    outcome = json.loads(result.stdout)
    assert outcome["changed"] and outcome["revision"] == 1
    assert outcome["notes"] == ["Font 'Comic Sans' isn't available"]
    draft = json.loads((run / "edits" / "clip_01" / "draft.json").read_text())
    assert draft["document"]["captions"]["color"] == "#FACC15"
    edited = run / outcome["file"]
    probe = _probe(edited)
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (1080, 1920)
    assert float(probe["format"]["duration"]) == pytest.approx(outcome["length"], abs=0.15)
    assert outcome["length"] < clips[0]["duration"] - 4
    assert json.loads((run / "clips.json").read_text())["clips"][0]["edits"][0]["file"] == outcome["file"]

    # Square output goes through the studio renderer.
    result = _clipper(tmp_path, "render", "demo", "--format", "square", "--clips", "2")
    assert result.returncode == 0, result.stdout + result.stderr
    square = json.loads((run / "clips.json").read_text())["clips"][1]
    probe = _probe(run / square["file"])
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (1080, 1080)
    assert _white_share(_frame(run / square["file"], 1.0), 0.66, 0.90) > 0.002, "captions not visible"

    attempts = guard_report.read_text().split() if guard_report.exists() else []
    assert attempts == [], f"Claude-native mode reached for an AI SDK or the network: {attempts}"
