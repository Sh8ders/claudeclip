"""``clipper validate``: ground Claude's picks with the web pipeline's own checks.

Runs, in order, exactly what the web app runs on a model's answer:
``_validate_segments`` (snap to transcript lines, repair length, ground the
text), ``dedupe_overlapping_segments``, the cross-window rerank
(``apply_candidate_ranking``, from Claude's ranking.json), the opening review
(``VideoService._refine_segment_openings``), the still-picture check
(``VideoService._drop_static_segments``) and the clip budget. Every drop or
change is written to validate_report.txt with the reason, so Claude can fix
selection.json and run it again.
"""

from __future__ import annotations

import hashlib
import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from pydantic import ValidationError

from .runs import Run, write_json

OK = "OK"
NEEDS_FIXES = "NEEDS_FIXES"
NEEDS_RANKING = "NEEDS_RANKING"
ERROR = "ERROR"

EXIT_CODES = {OK: 0, NEEDS_FIXES: 2, NEEDS_RANKING: 2, ERROR: 1}


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.messages: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.messages.append(record.getMessage())
        except Exception:
            self.messages.append(str(record.msg))


@contextmanager
def capture_logs() -> Iterator[List[str]]:
    """Collect the pipeline's own log messages (they carry each drop reason)."""
    package_logger = logging.getLogger(__name__.split(".")[0])
    handler = _Capture()
    previous_level, previous_propagate = package_logger.level, package_logger.propagate
    package_logger.addHandler(handler)
    if package_logger.getEffectiveLevel() > logging.INFO:
        package_logger.setLevel(logging.INFO)
    # The report carries these messages; keep them off the console.
    package_logger.propagate = False
    try:
        yield handler.messages
    finally:
        package_logger.removeHandler(handler)
        package_logger.setLevel(previous_level)
        package_logger.propagate = previous_propagate


@dataclass
class SegmentNote:
    source: str
    index: int
    start: str
    end: str
    title: Optional[str]
    status: str = "KEPT"
    changes: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    clip: Optional[int] = None

    def label(self) -> str:
        title = f' "{self.title}"' if self.title else ""
        return f"{self.source} #{self.index} {self.start}-{self.end}{title}"


@dataclass
class ValidationResult:
    status: str
    report: str
    segments: List[Dict[str, Any]]

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.status]


def _segment_payload(segment: Any) -> Dict[str, Any]:
    """The web pipeline's segment dict (``process_video_complete``)."""
    virality = segment.virality.model_dump() if segment.virality else {}
    return {
        "start_time": segment.start_time,
        "end_time": segment.end_time,
        "text": segment.text,
        "relevance_score": segment.relevance_score,
        "reasoning": segment.reasoning,
        "virality_score": virality.get("total_score", 0),
        "hook_score": virality.get("hook_score", 0),
        "engagement_score": virality.get("engagement_score", 0),
        "value_score": virality.get("value_score", 0),
        "shareability_score": virality.get("shareability_score", 0),
        "hook_type": virality.get("hook_type"),
        "hook_title": getattr(segment, "hook_title", None),
        "hook_start_time": getattr(segment, "hook_start_time", None),
        "hook_end_time": getattr(segment, "hook_end_time", None),
    }


def _diff_segment(before: Any, after: Any) -> List[str]:
    changes = []
    if (before.start_time, before.end_time) != (after.start_time, after.end_time):
        changes.append(
            f"time {before.start_time}-{before.end_time} -> {after.start_time}-{after.end_time}"
        )
    if before.text.strip() != after.text.strip():
        changes.append("text replaced with the transcript's words for that span")
    if (before.hook_title or None) != (after.hook_title or None):
        changes.append(f"hook_title {before.hook_title!r} -> {after.hook_title!r}")
    if before.virality.total_score != after.virality.total_score:
        changes.append(
            f"virality total {before.virality.total_score} -> {after.virality.total_score} (sum of subscores)"
        )
    if (before.hook_start_time, before.hook_end_time) != (after.hook_start_time, after.hook_end_time):
        changes.append("cold-open hook range removed (malformed or outside the clip)")
    return changes


def candidates_fingerprint(candidates: List[Any]) -> str:
    payload = json.dumps(
        [[c.start_time, c.end_time, c.hook_title, c.text] for c in candidates], ensure_ascii=False
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _load_analysis(path: Path):
    from ..clip_selection import TranscriptAnalysis

    return TranscriptAnalysis.model_validate_json(path.read_text(encoding="utf-8"))


def _format_validation_error(error: ValidationError) -> List[str]:
    lines = []
    for item in error.errors()[:20]:
        where = ".".join(str(part) for part in item.get("loc", ())) or "(top level)"
        lines.append(f"{where}: {item.get('msg')}")
    return lines


def validate_run(
    run: Run,
    max_clips: Optional[int] = None,
    visual_check: bool = True,
) -> ValidationResult:
    from ..clip_selection import (
        _overlap_ratio,
        _parse_transcript_lines,
        _parse_transcript_spans,
        _segment_bounds_seconds,
        _validate_segments,
        apply_candidate_ranking,
        build_candidate_rerank_prompt,
        candidate_rerank_system_prompt,
        dedupe_overlapping_segments,
        max_final_clips_for_duration,
    )
    from ..media.transcription import load_cached_transcript_data
    from ..services.video_service import VideoService
    from .schemas import NativeRanking

    meta = run.meta()
    clip_length = meta.get("clip_length", "auto")
    transcript = run.transcript_path.read_text(encoding="utf-8")
    video = run.video
    windows = meta.get("windows") or [{"index": 1, "selection": "selection.json"}]
    windowed = len(windows) > 1
    budget_ceiling = max_clips if max_clips is not None else meta.get("max_clips", 10)
    clip_limit = max_final_clips_for_duration(float(meta["duration_seconds"]), budget_ceiling)

    errors: List[str] = []
    notes: List[SegmentNote] = []
    lines = [f"clipper validate — {meta['name']}"]

    transcript_lines = _parse_transcript_lines(transcript)
    transcript_spans = _parse_transcript_spans(transcript)

    # 1. Load and ground every selection file.
    candidates: List[Any] = []
    note_for: Dict[int, SegmentNote] = {}
    summaries: List[str] = []
    for window in windows:
        path = run.path / window["selection"]
        if not path.is_file():
            errors.append(f"{window['selection']} is missing" + (f" (window {window['index']})" if windowed else ""))
            continue
        try:
            analysis = _load_analysis(path)
        except ValidationError as error:
            errors.append(f"{window['selection']} does not match `clipper schema selection`:")
            errors.extend(f"    {line}" for line in _format_validation_error(error))
            continue
        except (OSError, ValueError) as error:
            errors.append(f"{window['selection']} is not valid JSON: {error}")
            continue
        if analysis.summary:
            summaries.append(analysis.summary.strip())
        if not analysis.most_relevant_segments:
            errors.append(f"{window['selection']} has no segments")
        for index, segment in enumerate(analysis.most_relevant_segments, start=1):
            before = segment.model_copy(deep=True)
            note = SegmentNote(
                window["selection"], index, before.start_time, before.end_time, before.hook_title
            )
            notes.append(note)
            with capture_logs() as messages:
                kept = _validate_segments([segment], transcript_lines, transcript_spans, clip_length)
            if not kept:
                note.status = "DROPPED"
                note.reasons = [m for m in messages if "Skipping" in m] or messages or [
                    "rejected by grounding"
                ]
                continue
            note.changes = _diff_segment(before, segment)
            if any("Repaired segment duration" in m for m in messages):
                note.changes.append("length repaired to fit the clip-length bounds")
            candidates.append(segment)
            note_for[id(segment)] = note

    # 2. Overlapping picks: keep the higher-scoring one.
    deduped = dedupe_overlapping_segments(candidates)
    kept_ids = {id(segment) for segment in deduped}
    for segment in candidates:
        if id(segment) in kept_ids:
            continue
        note = note_for[id(segment)]
        note.status = "DROPPED"
        bounds = _segment_bounds_seconds(segment)
        winner = max(
            deduped,
            key=lambda other: _overlap_ratio(bounds, _segment_bounds_seconds(other)),
            default=None,
        )
        note.reasons = [
            "overlaps a higher-scoring pick by more than half"
            + (f" ({note_for[id(winner)].label()})" if winner is not None else "")
        ]

    # 3. Long sources: Claude's cross-window rerank.
    status_reasons: List[str] = []
    ranked = deduped
    fingerprint = candidates_fingerprint(deduped)
    if windowed and len(deduped) >= 2 and not errors:
        write_json(
            run.path / "candidates.json",
            {
                "candidates_fingerprint": fingerprint,
                "candidates": [
                    {"candidate_id": i, **_segment_payload(s)} for i, s in enumerate(deduped)
                ],
            },
        )
        (run.path / "rerank.md").write_text(
            "# Cross-window rerank\n\n"
            f"Write `ranking.json` (schema: `clipper schema ranking`) with "
            f'"candidates_fingerprint": "{fingerprint}" and every candidate_id once, best first.\n'
            "Then run `clipper validate <run>` again.\n\n"
            "## System prompt\n\n"
            f"{candidate_rerank_system_prompt}\n\n"
            "## Task\n\n"
            f"{build_candidate_rerank_prompt(deduped)}\n",
            encoding="utf-8",
        )
        ranking_path = run.path / "ranking.json"
        if not ranking_path.is_file():
            status_reasons.append("ranking.json is missing: rank the candidates in rerank.md")
        else:
            try:
                ranking = NativeRanking.model_validate_json(ranking_path.read_text(encoding="utf-8"))
            except (ValidationError, ValueError) as error:
                ranking = None
                status_reasons.append(f"ranking.json is invalid: {error}")
            if ranking is not None:
                if ranking.candidates_fingerprint and ranking.candidates_fingerprint != fingerprint:
                    ranking = None
                    status_reasons.append(
                        "ranking.json is stale (the candidates changed): rank rerank.md again"
                    )
            if ranking is not None:
                ids = [item.candidate_id for item in ranking.ranking]
                missing = sorted(set(range(len(deduped))) - set(ids))
                unknown = sorted({i for i in ids if not 0 <= i < len(deduped)})
                if unknown:
                    lines.append(f"Note: ranking.json has unknown candidate ids {unknown}; ignored.")
                if missing:
                    lines.append(
                        f"Note: ranking.json skipped candidates {missing}; they keep first-pass order after the ranked ones."
                    )
                try:
                    ranked = apply_candidate_ranking(deduped, ranking)
                except ValueError as error:
                    status_reasons.append(f"ranking.json is unusable: {error}")

    # 4. The web pipeline's post-analysis steps on segment dicts.
    transcript_data = load_cached_transcript_data(video)
    payloads: List[Dict[str, Any]] = []
    note_by_payload: Dict[int, SegmentNote] = {}
    for segment in ranked:
        payload = _segment_payload(segment)
        payload["text"] = VideoService._ground_segment_text(payload, transcript_data)
        payloads.append(payload)
        note_by_payload[id(payload)] = note_for[id(segment)]

    before_openings = {id(p): (p["start_time"], p["hook_score"]) for p in payloads}
    try:
        with capture_logs():
            payloads = VideoService._refine_segment_openings(payloads, transcript_data, clip_length)
    except Exception as error:  # a heuristic never blocks the run
        lines.append(f"Note: opening review failed ({error}); openings kept as written.")
    for payload in payloads:
        note = note_by_payload[id(payload)]
        start, hook = before_openings[id(payload)]
        if payload["start_time"] != start:
            note.changes.append(
                f"opening trimmed: start {start} -> {payload['start_time']} (lead-in or fragment removed)"
            )
        if payload["hook_score"] != hook:
            note.changes.append(
                f"WARN opening still leans on missing context: hook_score {hook} -> {payload['hook_score']}"
                " (consider starting on a self-contained sentence)"
            )

    reserve = payloads[: min(24, max(1, clip_limit) * 2)]
    for payload in payloads[len(reserve):]:
        note = note_by_payload[id(payload)]
        note.status = "NOT RENDERED"
        note.reasons = [f"beyond the clip budget ({clip_limit})"]
    if visual_check and reserve and video.is_file():
        try:
            with capture_logs() as messages:
                visible = VideoService._drop_static_segments(video, reserve)
        except Exception as error:
            visible = reserve
            lines.append(f"Note: still-picture check failed ({error}); every segment kept.")
        else:
            visible_ids = {id(p) for p in visible}
            for payload in reserve:
                if id(payload) not in visible_ids:
                    note = note_by_payload[id(payload)]
                    note.status = "DROPPED"
                    note.reasons = [
                        "one still picture (slate, card or black) covers most of it"
                    ] + [m for m in messages if payload["start_time"] in m and "Dropping" in m]
        reserve = visible
    final = reserve[: max(1, clip_limit)]
    for payload in reserve[len(final):]:
        note = note_by_payload[id(payload)]
        note.status = "NOT RENDERED"
        note.reasons = [f"beyond the clip budget ({clip_limit})"]

    for number, payload in enumerate(final, start=1):
        payload["clip"] = number
        note_by_payload[id(payload)].clip = number
        note_by_payload[id(payload)].status = "CHANGED" if note_by_payload[id(payload)].changes else "KEPT"

    # 5. Status and report.
    grounding_drops = [n for n in notes if n.status == "DROPPED" and any("Skipping" in r for r in n.reasons)]
    if errors or not final:
        status = ERROR
        if not final and not errors:
            errors.append("No segment survived validation; write new picks and run validate again.")
    elif status_reasons:
        status = NEEDS_RANKING
    elif grounding_drops:
        status = NEEDS_FIXES
    else:
        status = OK

    lines.append(f"Status: {status}")
    lines.append(
        f"Clip budget: {clip_limit} (source {meta['duration_seconds'] / 60:.1f} min, clip length "
        f"{clip_length}, MAX_CLIPS {budget_ceiling})"
    )
    if errors:
        lines.append("")
        lines.append("Errors:")
        lines.extend(f"  {line}" for line in errors)
    if status_reasons:
        lines.append("")
        lines.append("Needs ranking:")
        lines.extend(f"  {line}" for line in status_reasons)
    if notes:
        lines.append("")
        lines.append("Segments as written:")
        for note in notes:
            outcome = f"clip {note.clip}" if note.clip else note.status
            lines.append(f"  {note.label()} -> {outcome}")
            for change in note.changes:
                lines.append(f"      changed: {change}")
            for reason in note.reasons:
                lines.append(f"      reason: {reason}")
    if final:
        lines.append("")
        lines.append("Final segments (segments.json):")
        for payload in final:
            lines.append(
                f"  {payload['clip']:>2}. {payload['start_time']}-{payload['end_time']} "
                f"virality {payload['virality_score']:>3} | {payload.get('hook_title') or '(no hook title)'}"
            )
    lines.append("")
    if status == OK:
        lines.append("Next: optional `clipper broll-context <run>`, then `clipper render <run>`.")
    elif status == NEEDS_FIXES:
        lines.append(
            "Next: fix or replace the DROPPED segments in the selection file (use exact transcript "
            "line timestamps and the clip-length bounds), then run validate again."
        )
    elif status == NEEDS_RANKING:
        lines.append("Next: write ranking.json from rerank.md, then run validate again.")
    else:
        lines.append("Next: fix the errors above, then run validate again.")

    report = "\n".join(lines) + "\n"
    run.report_path.write_text(report, encoding="utf-8")
    if final:
        write_json(
            run.segments_path,
            {
                "status": status,
                "summary": " ".join(summaries),
                "clip_length": clip_length,
                "segments": final,
            },
        )
    else:
        run.segments_path.unlink(missing_ok=True)
    return ValidationResult(status, report, final)
