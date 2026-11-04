"""Pure clip-selection logic shared by the web pipeline and Claude-native mode.

Everything here is deterministic: the structured output models, the prompt
builders, and the grounding/repair that turns a model's (or Claude's) picks
into transcript-aligned segments. It must never import ``pydantic_ai`` or any
hosted AI SDK; ``src/ai.py`` re-exports every name for existing callers.
"""

from typing import List, Dict, Any, Optional, Literal
import logging
import math
import re

from pydantic import AliasChoices, BaseModel, Field, field_validator

from .clip_duration import clip_duration
from .prompts import load_prompt

# Keep the historical logger name so log filters and dashboards still match.
logger = logging.getLogger(__name__.rsplit(".", 1)[0] + ".ai" if "." in __name__ else "ai")

IDEAL_CLIP_MIN_SECONDS = 25
IDEAL_CLIP_MAX_SECONDS = 50
MIN_ACCEPTED_CLIP_SECONDS = 15
MAX_ACCEPTED_CLIP_SECONDS = 60

TRANSCRIPT_ANALYSIS_CACHE_VERSION = "windowed-rerank-v6"

# Long transcripts are analysed in overlapping windows so every part of the
# source gets the model's full attention; candidates are then reranked together.
ANALYSIS_WINDOW_SECONDS = 12 * 60
ANALYSIS_WINDOW_OVERLAP_SECONDS = 2 * 60
ANALYSIS_SINGLE_WINDOW_MAX_SECONDS = 15 * 60
ANALYSIS_WINDOW_CONCURRENCY = 3
ANALYSIS_WINDOW_SEGMENT_TARGET = "3-5"

CANDIDATE_DEDUPE_OVERLAP_RATIO = 0.5
# How far a model timestamp may sit from a transcript line boundary and still be
# snapped onto it instead of the whole clip being dropped.
SEGMENT_SNAP_TOLERANCE_SECONDS = 8
# Final clip count scales with source length: one clip per five minutes,
# never fewer than the single-pass prompt's old maximum and never more than 20.
MIN_FINAL_CLIPS = 5
MAX_FINAL_CLIPS = 20
SECONDS_PER_FINAL_CLIP = 5 * 60

HOOK_TITLE_MAX_CHARS = 64
HOOK_TITLE_MAX_WORDS = 10
# The transcript formatter does not wrap minutes into hours, so sources of 100+
# minutes produce labels such as "100:05".
TRANSCRIPT_SPAN_RE = re.compile(
    r"^\[(?P<start>\d{1,3}:\d{2}(?::\d{2})?)\s*-\s*"
    r"(?P<end>\d{1,3}:\d{2}(?::\d{2})?)\]\s*(?P<text>.*)$"
)
TRANSCRIPT_LINE_PATTERN = re.compile(
    r"^\[(\d{1,3}:\d{2}(?::\d{2})?)\s*-\s*(\d{1,3}:\d{2}(?::\d{2})?)\]\s*(.*)$"
)
SIGNAL_LINE_PATTERN = re.compile(r"^- \[(?P<start>\d{1,3}:\d{2}(?::\d{2})?)\s*-")
SPEAKER_PREFIX_PATTERN = re.compile(r"^Speaker [^:]+:\s*")

def _normalize_transcript_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9']+", " ", value.lower())).strip()


def _parse_transcript_lines(transcript: str) -> List[Dict[str, Any]]:
    lines: List[Dict[str, Any]] = []
    for raw_line in transcript.splitlines():
        match = TRANSCRIPT_LINE_PATTERN.match(raw_line.strip())
        if not match:
            continue
        line_text = SPEAKER_PREFIX_PATTERN.sub("", match.group(3).strip())
        lines.append(
            {
                "start_time": match.group(1),
                "end_time": match.group(2),
                "start_seconds": _parse_transcript_timestamp_seconds(match.group(1)),
                "end_seconds": _parse_transcript_timestamp_seconds(match.group(2)),
                "text": line_text,
            }
        )
    return lines


def _snap_bounds_to_transcript_lines(
    transcript_lines: List[Dict[str, Any]],
    start_seconds: int,
    end_seconds: int,
    tolerance_seconds: int = SEGMENT_SNAP_TOLERANCE_SECONDS,
) -> Optional[tuple[Dict[str, Any], Dict[str, Any]]]:
    """Snap model timestamps onto the nearest transcript line start and end.

    Models often pick a second that sits inside a line rather than on its
    boundary. Grounding needs exact line labels, so move each bound to the
    closest boundary within the tolerance instead of discarding the clip.
    """
    if not transcript_lines:
        return None

    start_line = min(
        transcript_lines,
        key=lambda line: (abs(line["start_seconds"] - start_seconds), line["start_seconds"]),
    )
    if abs(start_line["start_seconds"] - start_seconds) > tolerance_seconds:
        return None

    end_candidates = [
        line
        for line in transcript_lines
        if line["end_seconds"] > start_line["start_seconds"]
    ]
    if not end_candidates:
        return None
    end_line = min(
        end_candidates,
        key=lambda line: (abs(line["end_seconds"] - end_seconds), -line["end_seconds"]),
    )
    if abs(end_line["end_seconds"] - end_seconds) > tolerance_seconds:
        return None
    return start_line, end_line


def _extract_transcript_text_for_segment(
    transcript_lines: List[Dict[str, str]],
    start_time: str,
    end_time: str,
) -> Optional[str]:
    for start_index, line in enumerate(transcript_lines):
        if line["start_time"] != start_time:
            continue

        collected_parts = [line["text"]] if line["text"] else []
        if line["end_time"] == end_time:
            return " ".join(part for part in collected_parts if part).strip()

        for next_line in transcript_lines[start_index + 1 :]:
            if next_line["text"]:
                collected_parts.append(next_line["text"])
            if next_line["end_time"] == end_time:
                return " ".join(part for part in collected_parts if part).strip()

        return None

    return None


class ViralityAnalysis(BaseModel):
    """Detailed virality breakdown for a segment."""

    hook_score: int = Field(
        default=15,
        description="How strong is the opening hook (0-25)",
        ge=0,
        le=25,
    )
    engagement_score: int = Field(
        default=15,
        description="How engaging/entertaining is the content (0-25)",
        ge=0,
        le=25,
    )
    value_score: int = Field(
        default=15,
        description="Educational/informational value (0-25)",
        ge=0,
        le=25,
    )
    shareability_score: int = Field(
        default=15,
        description="Likelihood of being shared (0-25)",
        ge=0,
        le=25,
    )
    total_score: int = Field(
        default=60,
        description="Combined virality score (0-100)",
        ge=0,
        le=100,
    )
    hook_type: Optional[
        Literal["question", "statement", "statistic", "story", "contrast", "none"]
    ] = Field(
        default="none",
        description="Type of hook: question, statement, statistic, story, contrast, or none",
    )
    virality_reasoning: str = Field(
        default="The model did not provide a detailed virality breakdown.",
        description="Explanation of the virality score",
    )


def _default_virality_analysis() -> ViralityAnalysis:
    return ViralityAnalysis()


class TranscriptSegment(BaseModel):
    """Represents a relevant segment of transcript with precise timing and virality analysis."""

    start_time: str = Field(description="Start timestamp in MM:SS format")
    end_time: str = Field(description="End timestamp in MM:SS format")
    text: str = Field(
        validation_alias=AliasChoices("text", "segment"),
        description=(
            "Transcript text taken only from the selected timestamp range. "
            "Keep it verbatim or near-verbatim, and do not paraphrase or merge non-contiguous lines."
        )
    )
    relevance_score: float = Field(
        default=0.75,
        description="Relevance score from 0.0 to 1.0", ge=0.0, le=1.0
    )
    reasoning: str = Field(
        default="Selected by the AI model as a clip candidate.",
        description=(
            "Brief factual explanation of why this exact segment works as a clip. "
            "Base it only on the provided transcript content."
        )
    )
    virality: ViralityAnalysis = Field(
        default_factory=_default_virality_analysis,
        description="Detailed virality score breakdown",
    )
    hook_title: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("hook_title", "title", "headline"),
        description=(
            "Short punchy on-screen title for the clip (3-9 words). Grounded in "
            "the segment content, no hashtags, no emojis, no surrounding quotes."
        ),
    )

    hook_start_time: Optional[str] = Field(
        default=None,
        description=(
            "Optional MM:SS start of the single strongest 2-5 second line inside "
            "the segment, replayed as a cold-open teaser. Null when the segment "
            "already opens on its best line."
        ),
    )
    hook_end_time: Optional[str] = Field(
        default=None,
        description="Optional MM:SS end of that hook line (null with hook_start_time).",
    )

    @field_validator("relevance_score", mode="before")
    @classmethod
    def _coerce_percent_relevance_score(cls, value: Any) -> Any:
        if value is None:
            return value
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return value
        if numeric_value > 1 and numeric_value <= 100:
            return numeric_value / 100
        return value


class TranscriptAnalysis(BaseModel):
    """Analysis result for transcript segments with virality scores."""

    most_relevant_segments: List[TranscriptSegment]
    summary: str = Field(description="Brief summary of the video content")
    key_topics: List[str] = Field(description="List of main topics discussed")


# Enhanced system prompt with virality scoring (text: prompts/transcript_analysis_system.md)
transcript_analysis_system_prompt = load_prompt("transcript_analysis_system")

def build_transcript_system_prompt(clip_length: str = "auto") -> str:
    bounds = clip_duration(clip_length)
    return transcript_analysis_system_prompt.format(
        minimum=bounds.minimum, maximum=bounds.maximum,
        ideal_minimum=bounds.ideal_minimum, ideal_maximum=bounds.ideal_maximum,
    )


class RankedCandidate(BaseModel):
    """One candidate's place in the comparative rerank."""

    candidate_id: int = Field(description="The id of the candidate being ranked")
    score: int = Field(
        description="Comparative virality score (0-100) relative to the other candidates",
        ge=0,
        le=100,
    )
    reason: str = Field(
        default="",
        description="One short sentence on why it ranks here versus the others",
    )


class CandidateRanking(BaseModel):
    """Best-first ranking of clip candidates from one source video."""

    ranking: List[RankedCandidate] = Field(
        description="Every candidate, ordered from strongest to weakest clip"
    )


candidate_rerank_system_prompt = load_prompt("candidate_rerank_system")

class BRollCue(BaseModel):
    """One stock-footage cutaway inside a finished clip."""

    start: float = Field(
        description="Clip second where the cutaway starts: the timestamp of the line that names the subject"
    )
    duration: float = Field(description="Seconds to show the footage, 2 to 4")
    query: str = Field(
        description="2-4 word English stock-footage search describing what the footage shows"
    )
    fallback_query: str = Field(
        default="",
        description="Broader 1-2 word English search used when the first finds nothing",
    )


class BRollPlan(BaseModel):
    """Stock-footage cutaways for one clip, in play order."""

    cues: List[BRollCue] = Field(description="Cutaways in play order; empty when nothing fits")


broll_plan_system_prompt = load_prompt("broll_plan_system")

def build_broll_plan_prompt(
    timed_lines: str, clip_duration: float, max_cues: int, earliest_start: float
) -> str:
    """Ask for cutaways over one clip's transcript (timestamps in clip seconds)."""
    return (
        f"Clip length: {clip_duration:.1f} seconds.\n"
        f"Return at most {max_cues} cutaways. None may start before second "
        f"{earliest_start:.1f} or run past second {max(0.0, clip_duration - 1):.1f}.\n\n"
        "Transcript, one line per phrase, prefixed with the clip second it starts at:\n"
        f"{timed_lines}"
    )

def _performance_history_section(performance_context: str | None) -> str:
    """The creator's measured audience history, framed as a tie-breaker only."""
    if not performance_context:
        return ""
    return (
        "\n\nCreator performance history (real audience data from clips this creator "
        "already published):\n"
        f"{performance_context}\n\n"
        "Use this history to break ties and to bias hook_type, clip length and hook "
        "style toward what has demonstrably worked for this creator. It never justifies "
        "fabricating content, stretching timestamps, or picking a weak moment."
    )


def build_candidate_rerank_prompt(
    candidates: List[TranscriptSegment], performance_context: str | None = None
) -> str:
    """Describe every candidate in full so the reranker can compare them."""
    blocks = []
    for candidate_id, segment in enumerate(candidates):
        duration = _parse_transcript_timestamp_seconds(
            segment.end_time
        ) - _parse_transcript_timestamp_seconds(segment.start_time)
        blocks.append(
            f"candidate_id: {candidate_id}\n"
            f"time: {segment.start_time} - {segment.end_time} ({duration}s)\n"
            f"hook_title: {segment.hook_title or '(none)'}\n"
            f"text: {segment.text}"
        )
    return (
        f"Rank these {len(candidates)} clip candidates from the same video, best first.\n\n"
        + "\n\n".join(blocks)
        + _performance_history_section(performance_context)
    )


def build_transcript_analysis_prompt(
    transcript: str,
    clip_signals: str | None = None,
    window: tuple[int, int] | None = None,
    performance_context: str | None = None,
    clip_length: str = "auto",
) -> str:
    """Build the grounded task prompt for transcript analysis.

    ``window`` marks the prompt as one (start, end) slice of a longer source.
    ``performance_context`` is the creator's measured audience history, if any.
    """
    bounds = clip_duration(clip_length)
    segment_target = ANALYSIS_WINDOW_SEGMENT_TARGET if window else "2-5"
    window_note = ""
    if window:
        window_note = (
            f"\nThis transcript is one window ({_format_minutes_label(window[0])} - "
            f"{_format_minutes_label(window[1])}) of a longer video. Other windows are "
            "analysed separately, so pick the best clips that exist inside this window only.\n"
        )
    performance_section = _performance_history_section(performance_context)
    signal_section = ""
    if clip_signals:
        signal_section = (
            "\n\nAdditional deterministic signals from transcript/audio analysis:\n"
            f"{clip_signals}\n\n"
            "Use these as hints only. They should influence ranking, but every final segment "
            "must still be a coherent contiguous transcript range."
        )

    return f"""Analyze this video transcript and identify the most engaging segments for short-form content.
{window_note}
The transcript is formatted as one line per timestamped span, for example:
[00:12 - 00:21] Spoken text here
[00:21 - 00:35] More spoken text here

Follow this workflow:
1. Read the transcript as a sequence of timestamped spans.
2. Select only contiguous ranges that already exist in the transcript.
3. Prefer moments with a strong hook, clear payoff, emotional charge, or concrete value.
4. For each chosen segment, use the earliest timestamp in the selected range as start_time and the latest timestamp in the selected range as end_time.
5. start_time must be the first word of a self-contained sentence: never a mid-thought fragment, a filler, a bare agreement, or a reference to something this clip does not contain. end_time must finish that thought's last sentence and must not include the next thought's opening words.

Selection target:
- Choose {segment_target} segments total.
- Most selected clips should be {bounds.ideal_minimum}-{bounds.ideal_maximum} seconds.
- Every clip must contain a full setup and payoff within {bounds.minimum}-{bounds.maximum} seconds.
- If a strong moment is shorter than {bounds.ideal_minimum} seconds, first try expanding to nearby contiguous transcript lines that add useful context.
- Skip weak standalone picks: intros, sponsor reads, CTAs, contextless quotes, repeated points, vague setup, and answer fragments that require prior context.
- Before returning a segment, ask whether a viewer would understand and care without seeing the rest of the source video.

Critical accuracy requirements:
- Do not fabricate or embellish content.
- Do not use timestamps that are not present in the transcript.
- Do not merge separate non-contiguous moments into one segment.
- segment.text must reflect only the spoken content inside the selected time range.
- If a span lacks enough context to stand alone, expand to nearby contiguous lines rather than guessing.
- If there is a tradeoff between "viral" and "accurate", choose accuracy.
- Do not reject or penalize a segment simply because of the subject matter; stay content-neutral and assess clip quality only.
{signal_section}{performance_section}

JSON-only output requirements:
- Return one valid JSON object and nothing else.
- No Markdown, headings, bullets, code fences, or explanatory text outside JSON.
- Top-level keys: "most_relevant_segments", "summary", "key_topics".
- Segment keys: "start_time", "end_time", "text", "relevance_score", "reasoning", "virality", "hook_title".
- Optional segment keys "hook_start_time"/"hook_end_time": null unless a later 2-5 second line stands alone and the real opening is already a complete setup. The clip is not replayed from that line by default.
- "hook_title" is a 3-9 word plain-text headline for the clip, in the transcript's language, grounded in the segment (no hashtags, emojis, or quotes).
- Virality keys: "hook_score", "engagement_score", "value_score", "shareability_score", "total_score", "hook_type", "virality_reasoning".
- Do not return segments shorter than {bounds.minimum} seconds or longer than {bounds.maximum} seconds.

Transcript:
{transcript}"""


def sanitize_hook_title(raw: Optional[str]) -> Optional[str]:
    """Normalize an AI-provided hook title for on-screen rendering.

    Strips wrapping quotes/markdown, collapses whitespace, drops hashtags, and
    trims to a word-boundary length cap. Returns None when nothing usable is
    left so callers can simply skip the overlay.
    """
    if not raw:
        return None
    title = str(raw).strip()
    title = title.strip("\"'`“”‘’").strip()
    title = re.sub(r"#\w+", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    # Drop trailing sentence punctuation but keep ?/! (they carry the hook).
    title = title.rstrip(".,;:-–— ").strip()
    if not title:
        return None

    words = title.split()
    if len(words) > HOOK_TITLE_MAX_WORDS:
        words = words[:HOOK_TITLE_MAX_WORDS]
        title = " ".join(words)
    if len(title) > HOOK_TITLE_MAX_CHARS:
        clipped = title[: HOOK_TITLE_MAX_CHARS + 1]
        cut = clipped.rfind(" ")
        title = (clipped[:cut] if cut > 20 else title[:HOOK_TITLE_MAX_CHARS]).rstrip(
            ".,;:-–— "
        )
    return title or None


def _sanitize_hook_span(
    segment: TranscriptSegment, start_seconds: int, end_seconds: int
) -> None:
    """Drop a cold-open hook range that is malformed or outside the segment.

    Word-level snapping and length checks happen at render time.
    """
    try:
        hook_start = _parse_transcript_timestamp_seconds(segment.hook_start_time or "")
        hook_end = _parse_transcript_timestamp_seconds(segment.hook_end_time or "")
    except ValueError:
        hook_start = hook_end = None
    if (
        hook_start is None
        or hook_end is None
        or not start_seconds < hook_start < hook_end <= end_seconds
        or hook_end - hook_start > 8
    ):
        segment.hook_start_time = None
        segment.hook_end_time = None


def _parse_transcript_timestamp_seconds(timestamp: str) -> int:
    """Parse MM:SS or HH:MM:SS transcript timestamps into seconds."""
    parts = [int(part) for part in timestamp.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"Unsupported timestamp format: {timestamp}")


def _format_transcript_timestamp(seconds: int) -> str:
    """Format seconds as a transcript timestamp."""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _format_minutes_label(seconds: int) -> str:
    """Format seconds the way transcript lines do: MM:SS with unbounded minutes."""
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    return f"{minutes:02d}:{seconds:02d}"


def _parse_transcript_spans(transcript: str) -> list[dict[str, Any]]:
    """Parse timestamped transcript lines into spans."""
    spans = []
    for line in transcript.splitlines():
        match = TRANSCRIPT_SPAN_RE.match(line.strip())
        if not match:
            continue
        try:
            start_seconds = _parse_transcript_timestamp_seconds(match.group("start"))
            end_seconds = _parse_transcript_timestamp_seconds(match.group("end"))
        except ValueError:
            continue
        if end_seconds <= start_seconds:
            continue
        spans.append(
            {
                "start": start_seconds,
                "end": end_seconds,
                "text": match.group("text").strip(),
            }
        )
    return spans


def _extract_transcript_text(
    transcript_spans: list[dict[str, Any]], start_seconds: int, end_seconds: int
) -> str:
    """Return transcript text overlapping a selected time range."""
    selected_text = [
        span["text"]
        for span in transcript_spans
        if span["text"]
        and span["end"] > start_seconds
        and span["start"] < end_seconds
    ]
    return " ".join(selected_text).strip()


def _choose_repaired_bounds(
    transcript_spans: list[dict[str, Any]], start_seconds: int, end_seconds: int,
    clip_length: str = "auto",
) -> tuple[int, int] | None:
    """Repair model-selected bounds to the nearest acceptable contiguous range."""
    bounds = clip_duration(clip_length)
    if not transcript_spans:
        return None

    starts = sorted({span["start"] for span in transcript_spans})
    ends = sorted({span["end"] for span in transcript_spans})
    current_duration = end_seconds - start_seconds

    if current_duration > bounds.maximum:
        target_end = start_seconds + bounds.ideal_maximum
        candidate_ends = [
            candidate
            for candidate in ends
            if start_seconds + bounds.minimum
            <= candidate
            <= min(target_end, end_seconds)
        ]
        if candidate_ends:
            return start_seconds, max(candidate_ends)
        if start_seconds + bounds.minimum <= target_end:
            return start_seconds, target_end
        return None

    if current_duration < bounds.minimum:
        candidate_ranges: list[tuple[int, int, int]] = []
        for candidate_start in starts:
            if candidate_start > start_seconds:
                continue
            for candidate_end in ends:
                if candidate_end < end_seconds:
                    continue
                duration = candidate_end - candidate_start
                if bounds.minimum <= duration <= bounds.maximum:
                    extra_context = (start_seconds - candidate_start) + (
                        candidate_end - end_seconds
                    )
                    ideal_penalty = 0
                    if duration < bounds.ideal_minimum:
                        ideal_penalty = bounds.ideal_minimum - duration
                    elif duration > bounds.ideal_maximum:
                        ideal_penalty = duration - bounds.ideal_maximum
                    candidate_ranges.append(
                        (ideal_penalty * 1000 + extra_context, candidate_start, candidate_end)
                    )
        if candidate_ranges:
            _, repaired_start, repaired_end = min(candidate_ranges)
            return repaired_start, repaired_end

    return None


def _repair_segment_bounds(
    segment: TranscriptSegment,
    transcript_spans: list[dict[str, Any]],
    start_seconds: int,
    end_seconds: int,
    clip_length: str = "auto",
) -> tuple[int, int] | None:
    """Adjust near-miss model ranges to usable transcript-aligned bounds."""
    repaired_bounds = _choose_repaired_bounds(
        transcript_spans,
        start_seconds,
        end_seconds,
        clip_length,
    )
    if not repaired_bounds:
        return None

    repaired_start, repaired_end = repaired_bounds
    segment.start_time = _format_transcript_timestamp(repaired_start)
    segment.end_time = _format_transcript_timestamp(repaired_end)
    repaired_text = _extract_transcript_text(
        transcript_spans,
        repaired_start,
        repaired_end,
    )
    if repaired_text:
        segment.text = repaired_text
    logger.info(
        "Repaired segment duration: %s-%s -> %s-%s",
        _format_transcript_timestamp(start_seconds),
        _format_transcript_timestamp(end_seconds),
        segment.start_time,
        segment.end_time,
    )
    return repaired_start, repaired_end

def split_transcript_into_windows(
    transcript: str,
    window_seconds: int = ANALYSIS_WINDOW_SECONDS,
    overlap_seconds: int = ANALYSIS_WINDOW_OVERLAP_SECONDS,
    single_window_max_seconds: int = ANALYSIS_SINGLE_WINDOW_MAX_SECONDS,
) -> List[Dict[str, Any]]:
    """Split a formatted transcript into overlapping time windows.

    Transcripts up to ``single_window_max_seconds`` stay one window holding the
    original text unchanged. A short trailing window is folded into the one
    before it so no window is analysed with too little material.
    """
    timed_lines: List[tuple[int, str]] = []
    total_seconds = 0
    for raw_line in transcript.splitlines():
        line = raw_line.strip()
        match = TRANSCRIPT_SPAN_RE.match(line)
        if not match:
            continue
        try:
            start_seconds = _parse_transcript_timestamp_seconds(match.group("start"))
            end_seconds = _parse_transcript_timestamp_seconds(match.group("end"))
        except ValueError:
            continue
        timed_lines.append((start_seconds, line))
        total_seconds = max(total_seconds, end_seconds, start_seconds)

    if not timed_lines or total_seconds <= single_window_max_seconds:
        return [{"start": 0, "end": total_seconds, "transcript": transcript}]

    step = max(1, window_seconds - overlap_seconds)
    bounds: List[List[int]] = []
    window_start = 0
    while True:
        window_end = window_start + window_seconds
        if window_end >= total_seconds:
            bounds.append([window_start, total_seconds])
            break
        bounds.append([window_start, window_end])
        window_start += step
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < window_seconds // 2:
        bounds.pop()
        bounds[-1][1] = total_seconds

    windows: List[Dict[str, Any]] = []
    for index, (window_start, window_end) in enumerate(bounds):
        is_last = index == len(bounds) - 1
        lines = [
            line
            for start_seconds, line in timed_lines
            if window_start <= start_seconds
            and (start_seconds < window_end or (is_last and start_seconds <= window_end))
        ]
        if lines:
            windows.append(
                {"start": window_start, "end": window_end, "transcript": "\n".join(lines)}
            )
    return windows


def _filter_clip_signals_for_window(
    clip_signals: Optional[str], window_start: int, window_end: int
) -> Optional[str]:
    """Keep only the signal lines that fall inside one analysis window."""
    if not clip_signals:
        return None
    header: List[str] = []
    kept: List[str] = []
    for line in clip_signals.splitlines():
        match = SIGNAL_LINE_PATTERN.match(line.strip())
        if not match:
            if not kept:
                header.append(line)
            continue
        try:
            signal_start = _parse_transcript_timestamp_seconds(match.group("start"))
        except ValueError:
            continue
        if window_start <= signal_start < window_end:
            kept.append(line)
    if not kept:
        return None
    return "\n".join(header + kept)


def _segment_bounds_seconds(segment: TranscriptSegment) -> tuple[int, int]:
    return (
        _parse_transcript_timestamp_seconds(segment.start_time),
        _parse_transcript_timestamp_seconds(segment.end_time),
    )


def _segment_rank_key(segment: TranscriptSegment) -> tuple[int, float]:
    return (
        segment.virality.total_score if segment.virality else 0,
        segment.relevance_score,
    )


def _overlap_ratio(first: tuple[int, int], second: tuple[int, int]) -> float:
    """Overlap as a share of the shorter range (1.0 = one contains the other)."""
    shorter = min(first[1] - first[0], second[1] - second[0])
    if shorter <= 0:
        return 0.0
    overlap = min(first[1], second[1]) - max(first[0], second[0])
    return max(0, overlap) / shorter


def dedupe_overlapping_segments(
    segments: List[TranscriptSegment],
    max_overlap_ratio: float = CANDIDATE_DEDUPE_OVERLAP_RATIO,
) -> List[TranscriptSegment]:
    """Sort candidates best-first and drop ones that mostly repeat a stronger pick.

    Overlapping windows can surface the same moment twice; keep the higher score.
    """
    kept: List[tuple[TranscriptSegment, tuple[int, int]]] = []
    for segment in sorted(segments, key=_segment_rank_key, reverse=True):
        bounds = _segment_bounds_seconds(segment)
        if any(
            _overlap_ratio(bounds, kept_bounds) > max_overlap_ratio
            for _, kept_bounds in kept
        ):
            logger.info(
                "Dropping overlapping candidate %s-%s", segment.start_time, segment.end_time
            )
            continue
        kept.append((segment, bounds))
    return [segment for segment, _ in kept]


def max_final_clips_for_duration(
    source_seconds: float, max_clips: int = MAX_FINAL_CLIPS
) -> int:
    """Scale the clip budget with source length (about one clip per 5 minutes),
    never above ``max_clips`` (the configured MAX_CLIPS)."""
    ceiling = max(1, min(MAX_FINAL_CLIPS, max_clips))
    return max(
        min(MIN_FINAL_CLIPS, ceiling),
        min(ceiling, math.ceil(max(0.0, source_seconds) / SECONDS_PER_FINAL_CLIP)),
    )


def _rescale_virality(virality: ViralityAnalysis, new_total: int) -> None:
    """Set a new total while keeping the subscores' proportions and 0-25 bounds."""
    new_total = max(0, min(100, int(new_total)))
    subscores = [
        virality.hook_score,
        virality.engagement_score,
        virality.value_score,
        virality.shareability_score,
    ]
    old_total = sum(subscores)
    if old_total <= 0:
        scaled = [new_total / 4] * 4
    else:
        scaled = [score * new_total / old_total for score in subscores]
    rounded = [min(25, max(0, int(value))) for value in scaled]
    remainder = new_total - sum(rounded)
    by_fraction = sorted(range(4), key=lambda i: scaled[i] - int(scaled[i]), reverse=True)
    while remainder > 0:
        progressed = False
        for index in by_fraction:
            if remainder <= 0:
                break
            if rounded[index] < 25:
                rounded[index] += 1
                remainder -= 1
                progressed = True
        if not progressed:
            break
    (
        virality.hook_score,
        virality.engagement_score,
        virality.value_score,
        virality.shareability_score,
    ) = rounded
    virality.total_score = sum(rounded)


def apply_candidate_ranking(
    candidates: List[TranscriptSegment], ranking: CandidateRanking
) -> List[TranscriptSegment]:
    """Reorder candidates by the rerank and give them its calibrated scores.

    Scores are clamped to be non-increasing down the ranking. Candidates the
    reranker skipped keep their first-pass order after the ranked ones.
    """
    ordered: List[tuple[TranscriptSegment, int]] = []
    seen: set[int] = set()
    for item in ranking.ranking:
        candidate_id = item.candidate_id
        if candidate_id in seen or not 0 <= candidate_id < len(candidates):
            continue
        seen.add(candidate_id)
        ordered.append((candidates[candidate_id], item.score))
    if not ordered:
        raise ValueError("Rerank returned no usable candidate ids")

    reranked: List[TranscriptSegment] = []
    ceiling = 100
    for segment, score in ordered:
        ceiling = min(score, ceiling)
        _rescale_virality(segment.virality, ceiling)
        reranked.append(segment)
    for index, segment in enumerate(candidates):
        if index in seen:
            continue
        ceiling = min(segment.virality.total_score, ceiling)
        _rescale_virality(segment.virality, ceiling)
        reranked.append(segment)
    return reranked

def _snap_segment_bounds(
    segment: TranscriptSegment,
    transcript_lines: List[Dict[str, Any]],
    start_seconds: int,
    end_seconds: int,
) -> tuple[int, int]:
    """Move a segment onto transcript line boundaries when they are close."""
    snapped = _snap_bounds_to_transcript_lines(transcript_lines, start_seconds, end_seconds)
    if not snapped:
        return start_seconds, end_seconds
    start_line, end_line = snapped
    if (start_line["start_time"], end_line["end_time"]) != (
        segment.start_time,
        segment.end_time,
    ):
        logger.info(
            "Snapped segment %s-%s to transcript lines %s-%s",
            segment.start_time,
            segment.end_time,
            start_line["start_time"],
            end_line["end_time"],
        )
        segment.start_time = start_line["start_time"]
        segment.end_time = end_line["end_time"]
    return start_line["start_seconds"], end_line["end_seconds"]


def _validate_segments(
    segments: List[TranscriptSegment],
    transcript_lines: List[Dict[str, Any]],
    transcript_spans: List[Dict[str, Any]],
    clip_length: str = "auto",
) -> List[TranscriptSegment]:
    """Ground model segments onto real transcript lines and drop unusable ones."""
    bounds = clip_duration(clip_length)
    validated_segments: List[TranscriptSegment] = []
    for segment in segments:
        # Validate text content
        if not segment.text.strip() or len(segment.text.split()) < 3:
            logger.warning(
                f"Skipping segment with insufficient content: '{segment.text[:50]}...'"
            )
            continue

        # Validate timestamps - CRITICAL: start and end must be different
        if segment.start_time == segment.end_time:
            logger.warning(
                f"Skipping segment with identical start/end times: {segment.start_time}"
            )
            continue

        # Parse timestamps to validate duration
        try:
            start_seconds = _parse_transcript_timestamp_seconds(segment.start_time)
            end_seconds = _parse_transcript_timestamp_seconds(segment.end_time)
            start_seconds, end_seconds = _snap_segment_bounds(
                segment, transcript_lines, start_seconds, end_seconds
            )
            duration = end_seconds - start_seconds

            if duration < bounds.minimum or duration > bounds.maximum:
                repaired_bounds = _repair_segment_bounds(
                    segment,
                    transcript_spans,
                    start_seconds,
                    end_seconds,
                    clip_length,
                )
                if repaired_bounds:
                    # Repairs format hours as HH:MM:SS; snap back onto line labels.
                    start_seconds, end_seconds = _snap_segment_bounds(
                        segment, transcript_lines, *repaired_bounds
                    )
                    duration = end_seconds - start_seconds

            if duration <= 0:
                logger.warning(
                    f"Skipping segment with invalid duration: {segment.start_time} to {segment.end_time} = {duration}s"
                )
                continue

            if duration < bounds.minimum:
                logger.warning(
                    f"Skipping segment too short: {duration}s (min {bounds.minimum}s required)"
                )
                continue

            if duration > bounds.maximum:
                logger.warning(
                    f"Skipping segment too long: {duration}s (max {bounds.maximum}s allowed)"
                )
                continue

            grounded_text = _extract_transcript_text_for_segment(
                transcript_lines,
                segment.start_time,
                segment.end_time,
            )
            if grounded_text is None:
                logger.warning(
                    "Skipping segment with timestamps not aligned to transcript lines: %s-%s",
                    segment.start_time,
                    segment.end_time,
                )
                continue
            if _normalize_transcript_text(segment.text) != _normalize_transcript_text(
                grounded_text
            ):
                logger.warning(
                    "Adjusting segment text to transcript-backed span for %s-%s",
                    segment.start_time,
                    segment.end_time,
                )
                segment.text = grounded_text

            # Validate virality scores
            if segment.virality:
                # Ensure total score is sum of subscores
                calculated_total = (
                    segment.virality.hook_score
                    + segment.virality.engagement_score
                    + segment.virality.value_score
                    + segment.virality.shareability_score
                )
                if segment.virality.total_score != calculated_total:
                    logger.warning(
                        f"Correcting virality total: {segment.virality.total_score} -> {calculated_total}"
                    )
                    segment.virality.total_score = calculated_total

            segment.hook_title = sanitize_hook_title(segment.hook_title)
            # Bounds are final here (after snapping and repair): keep the teaser
            # only when it still sits inside the clip.
            _sanitize_hook_span(segment, start_seconds, end_seconds)

            validated_segments.append(segment)
            virality_info = (
                f", virality={segment.virality.total_score}"
                if segment.virality
                else ""
            )
            logger.info(
                f"Validated segment: {segment.start_time}-{segment.end_time} ({duration}s){virality_info}"
            )

        except (ValueError, IndexError) as e:
            logger.warning(
                f"Skipping segment with invalid timestamp format: {segment.start_time}-{segment.end_time}: {e}"
            )
            continue
    return validated_segments


def _merge_key_topics(analyses: List[TranscriptAnalysis], limit: int = 12) -> List[str]:
    topics: List[str] = []
    seen: set[str] = set()
    for analysis in analyses:
        for topic in analysis.key_topics or []:
            key = topic.strip().lower()
            if key and key not in seen:
                seen.add(key)
                topics.append(topic.strip())
    return topics[:limit]
