"""``clipper broll-context`` and ``clipper render``: clips from segments.json, no DB or Redis."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .runs import Run, mmss, write_json

logger = logging.getLogger(__name__)

RENDER_FORMATS = ("vertical", "vertical_fit", "vertical_pan", "vertical_split", "original")


def load_segments(run: Run) -> Dict[str, Any]:
    if not run.segments_path.is_file():
        raise FileNotFoundError("segments.json is missing: run `clipper validate` first.")
    return json.loads(run.segments_path.read_text(encoding="utf-8"))


def _keep_ranges(run: Run, segment: Dict[str, Any], meta: Dict[str, Any]):
    from ..clip_duration import clip_duration
    from ..services.video_service import VideoService

    return VideoService.resolve_clip_keep_ranges(
        run.video,
        segment,
        meta.get("cleanup"),
        clip_duration(meta.get("clip_length", "auto")).maximum,
    )


def write_broll_context(run: Run) -> List[Path]:
    """broll/NN.md per clip, on the exact timeline render will produce."""
    from .broll_local import broll_prompt, clip_words

    meta = run.meta()
    segments = load_segments(run)["segments"]
    folder = run.path / "broll"
    if folder.is_dir():
        shutil.rmtree(folder)
    folder.mkdir()
    written = []
    ranges_by_clip = {}
    for number, segment in enumerate(segments, start=1):
        resolved = _keep_ranges(run, segment, meta)
        if not resolved:
            continue
        words, clip_seconds = clip_words(run.video, resolved["keep_ranges"])
        path = folder / f"{number:02d}.md"
        path.write_text(broll_prompt(words, clip_seconds, number), encoding="utf-8")
        ranges_by_clip[str(number)] = [list(r) for r in resolved["keep_ranges"]]
        written.append(path)
    write_json(folder / "keep_ranges.json", ranges_by_clip)
    return written


def load_broll_plans(run: Run) -> Dict[str, Any]:
    from .schemas import BRollFile

    if not run.broll_json.is_file():
        return {}
    return BRollFile.model_validate_json(run.broll_json.read_text(encoding="utf-8")).root


def _clear_clips(run: Run, only: Optional[Sequence[int]]) -> None:
    run.clips_dir.mkdir(parents=True, exist_ok=True)
    for path in run.clips_dir.iterdir():
        if only is None or any(path.name.startswith(f"clip_{n}_") for n in only):
            if path.is_file():
                path.unlink()


async def render_run(
    run: Run,
    template: str = "default",
    font: Optional[str] = None,
    font_size: Optional[int] = None,
    font_color: Optional[str] = None,
    output_format: str = "vertical",
    subtitles: bool = True,
    broll: bool = False,
    broll_dir: Optional[Path] = None,
    only: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    from ..clip_duration import clip_duration
    from ..media.ffmpeg import ffprobe_video_size
    from ..services.video_service import VideoService
    from .broll_local import NativeBRoll

    if output_format not in RENDER_FORMATS:
        raise ValueError(f"--format must be one of {', '.join(RENDER_FORMATS)}")
    meta = run.meta()
    plan = load_segments(run)
    if plan.get("status") != "OK":
        logger.warning("segments.json status is %s; rendering anyway", plan.get("status"))
    segments = plan["segments"]
    wanted = set(only or range(1, len(segments) + 1))

    native_broll = None
    if broll:
        plans = load_broll_plans(run)
        if not plans:
            logger.warning("--broll given but broll.json is missing or empty; rendering without B-roll")
        native_broll = NativeBRoll(run.video, plans, broll_dir, output_format)
        if native_broll.source is None:
            logger.info("No PEXELS_API_KEY and no local B-roll folder; rendering without B-roll")
        stored = run.path / "broll" / "keep_ranges.json"
        if stored.is_file():
            expected = json.loads(stored.read_text())
            for number, segment in enumerate(segments, start=1):
                resolved = _keep_ranges(run, segment, meta)
                if resolved and str(number) in expected and [
                    list(r) for r in resolved["keep_ranges"]
                ] != expected[str(number)]:
                    logger.warning(
                        "Clip %s changed since broll-context ran; its cue times may be off", number
                    )

    previous = []
    if only and run.clips_json.is_file():
        previous = [
            clip for clip in json.loads(run.clips_json.read_text()).get("clips", [])
            if clip.get("clip") not in wanted
        ]
    _clear_clips(run, None if only is None else sorted(wanted))

    rendered: List[Dict[str, Any]] = []
    for index, segment in enumerate(segments):
        number = index + 1
        if number not in wanted:
            continue
        logger.info("Rendering clip %s/%s (%s-%s)", number, len(segments), segment["start_time"], segment["end_time"])
        info = await VideoService.create_single_clip(
            run.video,
            segment,
            index,
            run.clips_dir,
            font_family=font,
            font_size=font_size,
            font_color=font_color,
            caption_template=template,
            output_format=output_format,
            add_subtitles=subtitles,
            cleanup_settings=meta.get("cleanup"),
            max_clip_seconds=clip_duration(meta.get("clip_length", "auto")).maximum,
            include_broll=native_broll is not None,
            broll_session=native_broll.session if native_broll else None,
            broll_planner=native_broll.planner(number) if native_broll else None,
        )
        if not info:
            logger.error("Clip %s failed to render", number)
            rendered.append({"clip": number, "error": "render failed", **_scores(segment)})
            continue
        path = Path(info["path"])
        try:
            width, height = ffprobe_video_size(path)
        except Exception:
            width = height = None
        broll_sidecar = _read_broll_sidecar(path)
        rendered.append(
            {
                "clip": number,
                "file": f"clips/{path.name}",
                "start_time": info["start_time"],
                "end_time": info["end_time"],
                "duration": round(float(info["duration"]), 3),
                "width": width,
                "height": height,
                "hook_title": info.get("hook_title"),
                "text": info.get("text"),
                **_scores(info),
                "reasoning": info.get("reasoning"),
                "keep_ranges": [list(r) for r in info.get("keep_ranges") or []],
                "broll": [
                    {k: v for k, v in placement.items() if k in {"start", "duration", "query", "provider", "local_path", "pexels_url"}}
                    for placement in broll_sidecar
                ],
                "settings": {
                    "template": template, "font": font, "font_size": font_size, "font_color": font_color,
                    "format": output_format, "subtitles": subtitles,
                },
            }
        )

    clips = sorted(previous + rendered, key=lambda clip: clip["clip"])
    write_json(run.clips_json, {"name": meta["name"], "source": meta["source"], "clips": clips})
    return clips


def _scores(source: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "virality_score": source.get("virality_score", 0),
        "hook_score": source.get("hook_score", 0),
        "engagement_score": source.get("engagement_score", 0),
        "value_score": source.get("value_score", 0),
        "shareability_score": source.get("shareability_score", 0),
        "hook_type": source.get("hook_type"),
        "relevance_score": source.get("relevance_score"),
    }


def _read_broll_sidecar(clip_path: Path) -> List[Dict[str, Any]]:
    try:
        from ..clip_source_map import load_clip_broll

        return list(load_clip_broll(clip_path) or [])
    except Exception:
        return []


def summary_table(clips: List[Dict[str, Any]]) -> str:
    rows = ["| # | Time | Length | Virality | Hook title | File |", "|---|---|---|---|---|---|"]
    for clip in clips:
        if clip.get("error"):
            rows.append(f"| {clip['clip']} | — | — | {clip.get('virality_score', 0)} | (render failed) | — |")
            continue
        rows.append(
            f"| {clip['clip']} | {clip['start_time']}-{clip['end_time']} | {mmss(clip['duration'])} "
            f"| {clip['virality_score']} | {clip.get('hook_title') or ''} | {clip['file']} |"
        )
    return "\n".join(rows)
