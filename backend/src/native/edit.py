"""Studio "AI edit" for Claude-native runs, and square renders through the editor.

``clipper edit-context`` builds the clip studio's draft (an ``EditDocument``
on the clip's own timeline, exactly like ``editor_service.prepare_editor``
but from the run folder instead of the database) and prints the studio's AI
edit prompt with ``ai_edit_service.build_context``. Claude answers with an
``EditPlan`` (ops.json). ``clipper edit-apply`` applies it with the studio's
own ``apply_plan`` (every value clamped and validated) and renders the draft
with ``render_document``. Each applied plan is a new revision; earlier ones
stay in ``edits/clip_NN/history/``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .runs import Run, write_json

logger = logging.getLogger(__name__)


def _clip_entry(run: Run, number: int) -> Dict[str, Any]:
    if not run.clips_json.is_file():
        raise FileNotFoundError("clips.json is missing: run `clipper render` first.")
    for clip in json.loads(run.clips_json.read_text()).get("clips", []):
        if clip.get("clip") == number and not clip.get("error"):
            return clip
    raise ValueError(f"Clip {number} is not in clips.json (or failed to render).")


def _editor_task(settings: Dict[str, Any]) -> Dict[str, Any]:
    """The task fields the studio reads its initial caption look from."""
    return {
        "caption_template": settings.get("template") or "default",
        "font_family": settings.get("font"),
        "font_size": settings.get("font_size"),
        "font_color": settings.get("font_color"),
    }


def build_editor_source(
    video: Path, keep_ranges: Sequence[Tuple[float, float]], directory: Path, output_format: str
) -> Tuple[Dict[str, Any], List[float]]:
    """clean.mp4 (the clip's uncropped picture) plus the studio's asset metadata."""
    from ..media.ffmpeg import output_offsets_for_ranges, render_source_ranges_ffmpeg
    from ..services.editor_service import bake_editor_layout, make_assets
    from ..video_utils import create_optimized_clip

    ranges = [tuple(r) for r in keep_ranges]
    offsets = output_offsets_for_ranges(ranges)
    directory.mkdir(parents=True, exist_ok=True)
    clean = directory / "clean.mp4"
    if len(ranges) > 1:
        rendered = render_source_ranges_ffmpeg(video, ranges, clean, offsets)
    else:
        rendered = create_optimized_clip(
            video, ranges[0][0], ranges[-1][1], clean, add_subtitles=False,
            output_format="original", keep_ranges=ranges, hook_title=None, extend_to_sentence=False,
        )
    if not rendered:
        raise RuntimeError("Could not prepare the clip for editing")
    if output_format in {"vertical", "vertical_split"}:
        # Face-tracked (or split) framing, baked like the studio does.
        bake_editor_layout(clean, output_format, video, ranges, True)
    metadata = make_assets(directory)
    if (directory / "layout.mp4").exists():
        metadata["hasLayout"] = True
        metadata.update(json.loads((directory / "layout.json").read_text()))
    return metadata, offsets


def initial_document(
    video: Path,
    keep_ranges: Sequence[Tuple[float, float]],
    offsets: List[float],
    metadata: Dict[str, Any],
    text: str,
    hook_title: Optional[str],
    settings: Dict[str, Any],
    caption_settings: Optional[Dict[str, Any]] = None,
    broll_placements: Optional[List[Dict[str, Any]]] = None,
    aspect: Optional[str] = None,
):
    """The studio's starting draft for a generated clip (see ``prepare_editor``)."""
    from ..clip_editor import _caption_words_with_timings
    from ..editor_document import EditDocument
    from ..media.timeline import get_words_for_keep_ranges
    from ..media.transcription import load_cached_transcript_data
    from ..services.editor_service import editor_broll_from_placements, generated_emphasis, initial_caption_style

    duration = metadata["duration"]
    transcript = load_cached_transcript_data(video)
    timed = get_words_for_keep_ranges(transcript, list(keep_ranges), offsets) if transcript else []
    words = _caption_words_with_timings((text or "").split(), timed, duration)
    task = _editor_task(settings)
    emphasis = generated_emphasis(words, task)
    output_format = settings.get("format") or "vertical"
    original_format = output_format == "original"
    aspect = aspect or ("original" if original_format else "vertical")
    clip_width, clip_height = (metadata["width"], metadata["height"]) if original_format else (1080, 1920)
    # Local B-roll has no place in the studio's draft (it only carries Pexels footage).
    pexels = [p for p in (broll_placements or []) if p.get("provider", "pexels") == "pexels" and p.get("pexels_id")]
    return EditDocument.model_validate(
        {
            "segments": [{"id": "original", "start": 0, "end": duration}],
            "framing": {
                "aspect": aspect,
                "fit": "auto" if metadata.get("hasLayout") and aspect == "vertical" else "cover",
            },
            "captions": initial_caption_style(
                task, caption_settings or {}, settings.get("subtitles", True), clip_width, clip_height
            ),
            "hookTitle": (hook_title or "")[:200],
            "broll": editor_broll_from_placements(pexels, duration),
            "words": [
                {
                    "id": f"word-{i}",
                    "start": max(0, min(w["start"], duration - 0.01)),
                    "end": max(0.01, min(w["end"], duration)),
                    "text": w["text"][:120],
                    "highlight": i in emphasis,
                }
                for i, w in enumerate(words)
                if w["start"] < duration and w["end"] > 0
            ],
        }
    ).validate_duration(duration)


def edit_dir(run: Run, number: int) -> Path:
    return run.path / "edits" / f"clip_{number:02d}"


async def ensure_draft(run: Run, number: int, reset: bool = False):
    """(directory, draft document, revision, asset metadata) for one rendered clip."""
    from ..clip_source_map import load_clip_broll, load_clip_caption_settings
    from ..editor_document import EditDocument

    directory = edit_dir(run, number)
    if reset and directory.exists():
        shutil.rmtree(directory)
    draft_path, asset_path = directory / "draft.json", directory / "asset.json"
    if draft_path.is_file() and asset_path.is_file():
        draft = json.loads(draft_path.read_text())
        return directory, EditDocument.model_validate(draft["document"]), draft["revision"], json.loads(asset_path.read_text())

    clip = _clip_entry(run, number)
    settings = clip.get("settings") or {}
    ranges = [tuple(r) for r in clip["keep_ranges"]]
    clip_path = run.path / clip["file"]
    metadata, offsets = await asyncio.to_thread(
        build_editor_source, run.video, ranges, directory, settings.get("format") or "vertical"
    )
    document = initial_document(
        run.video, ranges, offsets, metadata, clip.get("text") or "", clip.get("hook_title"), settings,
        load_clip_caption_settings(clip_path), load_clip_broll(clip_path),
    )
    write_json(asset_path, metadata)
    write_json(directory / "original.json", document.model_dump())
    write_json(draft_path, {"revision": 0, "document": document.model_dump()})
    return directory, document, 0, metadata


async def edit_context(run: Run, number: int, instruction: str, reset: bool = False) -> str:
    from ..services.ai_edit_service import SYSTEM_PROMPT, build_context

    directory, document, revision, metadata = await ensure_draft(run, number, reset)
    context = build_context(document, metadata["duration"], None, instruction or "(the creator's request)")
    text = (
        f"# AI edit — clip {number} (draft revision {revision})\n\n"
        f"Write one JSON object matching `clipper schema edit` to a file (e.g. `{directory / 'ops.json'}`), "
        f"then run `clipper edit-apply {run.path.name} {number} <that file>`.\n\n"
        "## System prompt\n\n"
        f"{SYSTEM_PROMPT}\n\n"
        "## Draft and request\n\n"
        f"{context}\n"
    )
    (directory / "context.md").write_text(text, encoding="utf-8")
    return text


async def _render_draft(directory: Path, job_id: str, document) -> Path:
    from ..editor_document import render_document
    from ..services.editor_service import document_footage

    broll_files = await document_footage(document)
    await asyncio.to_thread(render_document, directory, job_id, document, "tiktok", broll_files)
    return directory / f"export-{job_id}.mp4"


async def edit_apply(run: Run, number: int, ops_path: Path, render: bool = True) -> Dict[str, Any]:
    from ..services.ai_edit_service import EditPlan, apply_plan

    plan = EditPlan.model_validate_json(Path(ops_path).read_text(encoding="utf-8"))
    directory, document, revision, metadata = await ensure_draft(run, number)
    result, notes = await apply_plan(
        document, plan, metadata["duration"], int(metadata.get("width") or 1080), int(metadata.get("height") or 1920)
    )
    changed = result.model_dump() != document.model_dump()
    outcome: Dict[str, Any] = {
        "clip": number,
        "summary": plan.summary.strip()[:600] if changed or not plan.operations else "The draft is unchanged.",
        "notes": notes,
        "changed": changed,
        "revision": revision,
        "length": round(sum(s.end - s.start for s in result.segments), 2),
    }
    if changed:
        revision += 1
        history = directory / "history"
        write_json(history / f"rev-{revision:02d}.plan.json", plan.model_dump())
        write_json(history / f"rev-{revision:02d}.json", result.model_dump())
        write_json(directory / "draft.json", {"revision": revision, "document": result.model_dump()})
        outcome["revision"] = revision
    if render:
        exported = await _render_draft(directory, f"rev{revision:02d}", result)
        final = run.path / "edits" / f"clip_{number:02d}_v{revision}.mp4"
        exported.replace(final)
        outcome["file"] = str(final.relative_to(run.path))
        _record_edit(run, number, outcome)
    return outcome


def _record_edit(run: Run, number: int, outcome: Dict[str, Any]) -> None:
    data = json.loads(run.clips_json.read_text())
    for clip in data.get("clips", []):
        if clip.get("clip") == number:
            edits = [e for e in clip.get("edits", []) if e.get("revision") != outcome["revision"]]
            edits.append({k: outcome[k] for k in ("revision", "file", "summary", "notes", "length")})
            clip["edits"] = edits
    write_json(run.clips_json, data)


def _square_anchor(clean: Path, duration: float) -> float:
    """Horizontal anchor (0-1) that keeps the dominant face in a square crop."""
    from ..media.ffmpeg import ffprobe_video_size
    from ..media.reframing import detect_optimal_crop_region

    try:
        width, height = ffprobe_video_size(clean)
        x, _y, crop_width, _crop_height = detect_optimal_crop_region(clean, 0, duration, target_ratio=1.0)
        overflow = width - crop_width
        return min(1.0, max(0.0, x / overflow)) if overflow > 0 else 0.5
    except Exception as exc:
        logger.warning("Face position unavailable for the square crop (%s); centring it", exc)
        return 0.5


async def render_square(
    run: Run,
    template: str = "default",
    font: Optional[str] = None,
    subtitles: bool = True,
    only: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    """1080x1080 clips through the studio renderer, with a face-anchored square crop."""
    from .render import _clear_clips, _keep_ranges, _scores, load_segments

    meta = run.meta()
    segments = load_segments(run)["segments"]
    wanted = set(only or range(1, len(segments) + 1))
    settings = {"template": template, "font": font, "format": "square", "subtitles": subtitles}
    previous = []
    if only and run.clips_json.is_file():
        previous = [c for c in json.loads(run.clips_json.read_text()).get("clips", []) if c.get("clip") not in wanted]
    _clear_clips(run, None if only is None else sorted(wanted))
    rendered = []
    for number, segment in enumerate(segments, start=1):
        if number not in wanted:
            continue
        logger.info("Rendering square clip %s/%s (%s-%s)", number, len(segments), segment["start_time"], segment["end_time"])
        try:
            resolved = _keep_ranges(run, segment, meta)
            if not resolved:
                raise ValueError("invalid duration")
            directory = run.path / "edits" / f"square_{number:02d}"
            shutil.rmtree(directory, ignore_errors=True)
            ranges = resolved["keep_ranges"]
            metadata, offsets = await asyncio.to_thread(build_editor_source, run.video, ranges, directory, "original")
            document = initial_document(
                run.video, ranges, offsets, metadata, resolved["text"], segment.get("hook_title"),
                {**settings, "format": "vertical"}, aspect="square",
            )
            anchor = await asyncio.to_thread(_square_anchor, directory / "clean.mp4", metadata["duration"])
            document = document.model_copy(update={"framing": document.framing.model_copy(update={"x": round(anchor, 3)})})
            exported = await _render_draft(directory, "square", document)
            name = f"clip_{number}_{segment['start_time'].replace(':', '')}-{segment['end_time'].replace(':', '')}_square.mp4"
            final = run.clips_dir / name
            exported.replace(final)
        except Exception as exc:
            logger.error("Square clip %s failed: %s", number, exc)
            rendered.append({"clip": number, "error": "render failed", **_scores(segment)})
            continue
        from ..media.ffmpeg import ffprobe_duration

        rendered.append({
            "clip": number,
            "file": f"clips/{final.name}",
            "start_time": segment["start_time"],
            "end_time": segment["end_time"],
            "duration": round(ffprobe_duration(final), 3),
            "width": 1080,
            "height": 1080,
            "hook_title": segment.get("hook_title"),
            "text": resolved["text"],
            **_scores(segment),
            "reasoning": segment.get("reasoning"),
            "keep_ranges": [list(r) for r in ranges],
            "broll": [],
            "settings": settings,
        })
    clips = sorted(previous + rendered, key=lambda clip: clip["clip"])
    write_json(run.clips_json, {"name": meta["name"], "source": meta["source"], "clips": clips})
    return clips
