"""B-roll for Claude-native runs: Claude writes the cues, footage needs no key.

Cues come from ``broll.json`` (one ``BRollPlan`` per clip, in clip seconds,
written by Claude from ``broll/NN.md``). They go through the web pipeline's
own ``select_cues`` (snap to words, spacing, coverage cap). Footage comes
from Pexels when ``PEXELS_API_KEY`` is set, otherwise from a local folder of
video files matched on filename keywords, otherwise the clip renders without
B-roll.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}


def clip_words(video: Path, keep_ranges: Sequence[tuple]) -> tuple[List[Dict[str, Any]], float]:
    """Words on the clip timeline and the clip's rendered length (as the web planner sees them)."""
    from ..media.ffmpeg import rendered_ranges_duration
    from ..media.timeline import get_words_for_keep_ranges
    from ..media.transcription import load_cached_transcript_data

    transcript = load_cached_transcript_data(video)
    words = get_words_for_keep_ranges(transcript, list(keep_ranges)) if transcript else []
    return words, rendered_ranges_duration(list(keep_ranges))


def broll_prompt(words: List[Dict[str, Any]], clip_seconds: float, clip_number: int) -> str:
    """The web app's B-roll prompt (system + task) for one clip."""
    from ..broll import EARLIEST_START_SECONDS, max_cues_for_duration, timed_lines_from_words
    from ..clip_selection import broll_plan_system_prompt, build_broll_plan_prompt

    task = build_broll_plan_prompt(
        timed_lines_from_words(words), clip_seconds, max_cues_for_duration(clip_seconds), EARLIEST_START_SECONDS
    )
    return (
        f"# B-roll cues — clip {clip_number}\n\n"
        f'Add this clip\'s plan to `broll.json` under the key "{clip_number}" '
        '(schema: `clipper schema broll`). Use {"cues": []} when nothing fits.\n\n'
        "## System prompt\n\n"
        f"{broll_plan_system_prompt}\n\n"
        "## Task\n\n"
        f"{task}\n"
    )


def _file_words(path: Path) -> set[str]:
    from ..broll import _content_words

    return _content_words(re.sub(r"[_\-.]+", " ", path.stem))


def local_footage(broll_dir: Optional[Path]) -> List[Path]:
    if not broll_dir or not broll_dir.is_dir():
        return []
    return sorted(
        path for path in broll_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS and not path.name.startswith(".")
    )


def match_local_footage(query: str, files: Sequence[Path], used: set[str]) -> Optional[Path]:
    """Best filename match for a query: most shared content words, unused files first."""
    from ..broll import _content_words, _same_word

    wanted = _content_words(query)
    if not wanted:
        return None
    best: Optional[tuple] = None
    for path in files:
        tokens = _file_words(path)
        score = sum(1 for word in wanted if any(_same_word(word, token) for token in tokens))
        if score == 0:
            continue
        key = (str(path) not in used, score, score / max(1, len(tokens)), -len(path.name))
        if best is None or key > best[0]:
            best = (key, path)
    return best[1] if best else None


class NativeBRoll:
    """Turns Claude's cues for one run into placements, clip by clip."""

    def __init__(
        self,
        video: Path,
        plans: Dict[str, Any],
        broll_dir: Optional[Path] = None,
        output_format: str = "vertical",
    ) -> None:
        from ..broll import BRollSession, broll_configured

        self.video = video
        self.plans = plans
        self.output_format = output_format
        self.session = BRollSession()
        self.used_local: set[str] = set()
        self.use_pexels = broll_configured()
        self.files = [] if self.use_pexels else local_footage(broll_dir)
        self.source = "pexels" if self.use_pexels else ("local" if self.files else None)

    def planner(self, clip_number: int):
        async def plan(keep_ranges: List[tuple]) -> List[Dict[str, Any]]:
            try:
                return await self._plan(clip_number, keep_ranges)
            except Exception as exc:  # B-roll never fails a clip
                logger.warning("B-roll for clip %s skipped: %s", clip_number, exc)
                return []

        return plan

    async def _target(self) -> tuple[str, int]:
        import asyncio

        from ..media.ffmpeg import ffprobe_video_size

        if self.output_format != "original":
            return "portrait", 1080
        width, height = await asyncio.to_thread(ffprobe_video_size, self.video)
        return ("landscape" if width > height else "portrait"), min(1080, min(width, height) or 1080)

    async def _plan(self, clip_number: int, keep_ranges: List[tuple]) -> List[Dict[str, Any]]:
        import asyncio

        from ..broll import (
            EARLIEST_START_SECONDS,
            END_GUARD_SECONDS,
            MIN_CUE_SECONDS,
            _source_cue,
            cue_context,
            ensure_broll_file,
            max_cues_for_duration,
            select_cues,
        )
        from ..media.ffmpeg import ffprobe_duration

        plan = self.plans.get(str(clip_number))
        if not plan or not plan.cues or self.source is None:
            return []
        words, clip_seconds = await asyncio.to_thread(clip_words, self.video, keep_ranges)
        if not words or clip_seconds < EARLIEST_START_SECONDS + MIN_CUE_SECONDS + END_GUARD_SECONDS:
            return []
        cues = select_cues(plan.cues, words, clip_seconds, max_cues_for_duration(clip_seconds))
        for cue in cues:
            cue["context"] = cue_context(words, cue["start"], cue["duration"])

        placements: List[Dict[str, Any]] = []
        if self.use_pexels:
            orientation, short_side = await self._target()
            for cue in cues:
                placement = await _source_cue(cue, orientation, short_side, self.session)
                if placement and await ensure_broll_file(placement):
                    placements.append(placement)
            return placements

        for cue in cues:
            path = None
            matched_query = None
            for query in dict.fromkeys(q for q in (cue["query"], cue.get("fallback_query")) if q):
                path = match_local_footage(query, self.files, self.used_local)
                if path:
                    matched_query = query
                    break
            if not path:
                logger.info("No local footage matches B-roll query %r", cue["query"])
                continue
            try:
                footage_seconds = await asyncio.to_thread(ffprobe_duration, path)
            except Exception:
                logger.warning("Local B-roll %s is not a readable video; skipping it", path.name)
                continue
            duration = min(cue["duration"], max(0.0, footage_seconds - 0.05))
            if duration < MIN_CUE_SECONDS:
                continue
            self.used_local.add(str(path))
            placements.append(
                {
                    "start": cue["start"],
                    "duration": round(duration, 3),
                    "query": matched_query,
                    "provider": "local",
                    "local_path": str(path),
                    "footage_offset": 0.0,
                    "footage_duration": float(footage_seconds),
                }
            )
        return placements
