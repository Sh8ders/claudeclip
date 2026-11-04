"""``clipper prepare``: download, normalize, transcribe locally, write Claude's inputs."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .runs import Run, mmss, slugify, write_json

logger = logging.getLogger(__name__)


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def fetch_source(source: str) -> tuple[Path, str, Optional[str]]:
    """Return (local file, source type, title) for a file path or YouTube URL."""
    from ..youtube_utils import download_youtube_video, get_youtube_video_id, get_youtube_video_title

    if get_youtube_video_id(source):
        title = None
        try:
            title = get_youtube_video_title(source)
        except Exception as exc:  # metadata is optional
            logger.warning("Could not read the YouTube title: %s", exc)
        path = download_youtube_video(source)
        if not path:
            raise RuntimeError(f"Could not download {source} with yt-dlp")
        return Path(path), "youtube", title
    path = Path(source).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{source} is not a file or a YouTube URL")
    return path.resolve(), "file", None


def default_run_name(source: str, local: Path, title: Optional[str]) -> str:
    from ..youtube_utils import get_youtube_video_id

    if title:
        return slugify(title)
    video_id = get_youtube_video_id(source)
    if video_id:
        return slugify(f"youtube-{video_id}")
    return slugify(local.stem)


def transcribe(video: Path, model: str, backend: str, language: Optional[str]) -> Dict[str, Any]:
    """Local Whisper with word timings, cached beside the video for the renderer."""
    from ..media.transcription import (
        cache_transcript_data,
        format_transcript_for_analysis,
        resolve_whisper_backend,
        transcribe_with_whisper,
    )

    resolved = resolve_whisper_backend(backend)
    started = time.monotonic()
    result = transcribe_with_whisper(video, model, language, resolved)
    cache_transcript_data(video, result)
    lines = format_transcript_for_analysis(result)
    return {
        "transcript": "\n".join(lines),
        "backend": resolved,
        "model": model,
        "language": result.get("language") or language,
        "seconds": round(time.monotonic() - started, 1),
    }


def import_transcript(video: Path, transcript_json: Path) -> Dict[str, Any]:
    """Use an existing Whisper-format JSON (segments with word timings) instead of transcribing.

    This is what ``whisper --word_timestamps True --output_format json`` writes,
    and what faster-whisper/mlx-whisper results look like as dicts.
    """
    from ..media.transcription import cache_transcript_data, format_transcript_for_analysis

    result = json.loads(Path(transcript_json).read_text(encoding="utf-8"))
    if not isinstance(result, dict) or not isinstance(result.get("segments"), list):
        raise ValueError(f"{transcript_json} is not a Whisper JSON result (no 'segments' list)")
    if not any(segment.get("words") for segment in result["segments"]):
        raise ValueError(
            f"{transcript_json} has no word timings; captions need them (Whisper: --word_timestamps True)"
        )
    cache_transcript_data(video, result)
    return {
        "transcript": "\n".join(format_transcript_for_analysis(result)),
        "backend": "imported",
        "model": None,
        "language": result.get("language"),
        "imported_from": str(Path(transcript_json).resolve()),
    }


def selection_prompt(
    transcript: str,
    signals: Optional[str],
    clip_length: str,
    output_file: str,
    window: Optional[tuple[int, int]] = None,
    window_label: str = "",
) -> str:
    """The web app's exact selection prompt (system + task), as one Markdown file."""
    from ..clip_selection import build_transcript_analysis_prompt, build_transcript_system_prompt

    task = build_transcript_analysis_prompt(
        transcript=transcript, clip_signals=signals, window=window, clip_length=clip_length
    )
    return (
        f"# Clip selection{window_label}\n\n"
        f"Write your answer as JSON to `{output_file}` (schema: `clipper schema selection`).\n"
        "Then run `clipper validate <run>`.\n\n"
        "## System prompt\n\n"
        f"{build_transcript_system_prompt(clip_length)}\n\n"
        "## Task\n\n"
        f"{task}\n"
    )


def write_selection_inputs(run: Run, transcript: str, signals: Optional[str], clip_length: str) -> List[Dict[str, Any]]:
    """prompt.md for one window, or windows/NN.{txt,md} for long sources."""
    from ..clip_selection import _filter_clip_signals_for_window, split_transcript_into_windows

    windows = split_transcript_into_windows(transcript)
    windows_dir = run.path / "windows"
    if windows_dir.is_dir():
        for stale in list(windows_dir.glob("*.txt")) + list(windows_dir.glob("*.md")):
            stale.unlink()
    if len(windows) == 1:
        (run.path / "prompt.md").write_text(
            selection_prompt(transcript, signals, clip_length, "selection.json"), encoding="utf-8"
        )
        return [{"index": 1, "start": windows[0]["start"], "end": windows[0]["end"],
                 "prompt": "prompt.md", "selection": "selection.json"}]

    (run.path / "prompt.md").unlink(missing_ok=True)
    windows_dir.mkdir(exist_ok=True)
    described = []
    for index, window in enumerate(windows, start=1):
        stem = f"{index:02d}"
        (windows_dir / f"{stem}.txt").write_text(window["transcript"] + "\n", encoding="utf-8")
        window_signals = _filter_clip_signals_for_window(signals, window["start"], window["end"])
        label = f" — window {index} of {len(windows)} ({mmss(window['start'])}-{mmss(window['end'])})"
        (windows_dir / f"{stem}.md").write_text(
            selection_prompt(
                window["transcript"], window_signals, clip_length,
                f"windows/{stem}.selection.json", (window["start"], window["end"]), label,
            ),
            encoding="utf-8",
        )
        described.append({
            "index": index, "start": window["start"], "end": window["end"],
            "transcript": f"windows/{stem}.txt", "prompt": f"windows/{stem}.md",
            "selection": f"windows/{stem}.selection.json",
        })
    return described


async def prepare(
    source: str,
    runs_dir: Path,
    name: Optional[str] = None,
    whisper_model: str = "small",
    whisper_backend: str = "auto",
    language: Optional[str] = None,
    clip_length: str = "auto",
    cleanup: Optional[Dict[str, Any]] = None,
    force: bool = False,
    transcript_json: Optional[Path] = None,
) -> Run:
    from ..clip_cleanup import new_task_cleanup_settings
    from ..clip_duration import validate_clip_length
    from ..clip_selection import max_final_clips_for_duration
    from ..config import get_config
    from ..media.ffmpeg import ffprobe_duration, ffprobe_video_size
    from ..media.timeline import build_clip_signal_summary
    from ..media.transcription import load_cached_transcript_data
    from ..services.video_service import VideoService

    validate_clip_length(clip_length)
    local, source_type, title = await asyncio.to_thread(fetch_source, source)
    run = Run(runs_dir / slugify(name or default_run_name(source, local, title)))
    run.path.mkdir(parents=True, exist_ok=True)
    previous = run.meta() if run.meta_path.is_file() else {}

    video = run.path / f"source{local.suffix.lower() or '.mp4'}"
    same_source = (
        previous.get("source") == source
        and video.is_file()
        and previous.get("source_bytes") == local.stat().st_size
    )
    if force or not same_source:
        for stale in run.path.glob("source.*"):
            stale.unlink()
        _link_or_copy(local, video)
        video = await VideoService.prepare_source(video)

    cached = load_cached_transcript_data(video) or {}
    previous_whisper = previous.get("whisper") or {}
    reuse_transcript = (
        not force
        and same_source
        and run.transcript_path.is_file()
        and cached.get("words")
        and previous_whisper.get("model") == whisper_model
        and (previous_whisper.get("requested_language") or None) == (language or None)
    )
    if transcript_json is not None:
        info = await asyncio.to_thread(import_transcript, video, transcript_json)
        transcript = info.pop("transcript")
        whisper_info = {**info, "requested_language": language}
        run.transcript_path.write_text(transcript + "\n", encoding="utf-8")
    elif reuse_transcript:
        transcript = run.transcript_path.read_text(encoding="utf-8").rstrip("\n")
        whisper_info = previous_whisper
        logger.info("Reusing the cached transcript (%s)", whisper_model)
    else:
        info = await asyncio.to_thread(transcribe, video, whisper_model, whisper_backend, language)
        transcript = info.pop("transcript")
        whisper_info = {**info, "requested_language": language}
        run.transcript_path.write_text(transcript + "\n", encoding="utf-8")
    if not transcript.strip():
        raise RuntimeError("Whisper found no speech in this video, so there is nothing to clip.")

    try:
        signals = await asyncio.to_thread(build_clip_signal_summary, video, transcript)
    except Exception as exc:
        logger.warning("Clip signal extraction failed: %s", exc)
        signals = ""
    run.signals_path.write_text((signals or "") + "\n", encoding="utf-8")

    windows = write_selection_inputs(run, transcript, signals or None, clip_length)
    duration = await asyncio.to_thread(ffprobe_duration, video)
    try:
        width, height = await asyncio.to_thread(ffprobe_video_size, video)
    except Exception:
        width = height = None
    transcript_data = load_cached_transcript_data(video) or {}
    words = transcript_data.get("words") or []
    max_clips = get_config().max_clips
    meta = {
        "name": run.path.name,
        "source": source,
        "source_type": source_type,
        "source_bytes": local.stat().st_size,
        "title": title,
        "video": video.name,
        "duration_seconds": round(float(duration), 3),
        "width": width,
        "height": height,
        "clip_length": clip_length,
        "max_clips": max_clips,
        "clip_budget": max_final_clips_for_duration(float(duration), max_clips),
        "cleanup": cleanup if cleanup is not None else new_task_cleanup_settings({}),
        "windows": windows,
        "transcript_lines": len([line for line in transcript.splitlines() if line.strip()]),
        "words": len(words),
        "speaker_labels": any(word.get("speaker") for word in words),
        "whisper": whisper_info,
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    run.write_meta(meta)
    return run
