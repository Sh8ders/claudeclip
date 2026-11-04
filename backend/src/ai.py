"""
AI-related functions for transcript analysis with enhanced precision and virality scoring.
"""

from typing import List, Dict, Any, Optional
import asyncio
import logging
import random

import httpx
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.profiles.anthropic import AnthropicModelProfile
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider

from .config import Config, get_config
from .runtime_settings import apply_settings_to_process_env

logger = logging.getLogger(__name__)

from .clip_selection import (  # noqa: F401 - re-exported for existing callers
    IDEAL_CLIP_MIN_SECONDS,
    IDEAL_CLIP_MAX_SECONDS,
    MIN_ACCEPTED_CLIP_SECONDS,
    MAX_ACCEPTED_CLIP_SECONDS,
    TRANSCRIPT_ANALYSIS_CACHE_VERSION,
    ANALYSIS_WINDOW_SECONDS,
    ANALYSIS_WINDOW_OVERLAP_SECONDS,
    ANALYSIS_SINGLE_WINDOW_MAX_SECONDS,
    ANALYSIS_WINDOW_CONCURRENCY,
    ANALYSIS_WINDOW_SEGMENT_TARGET,
    CANDIDATE_DEDUPE_OVERLAP_RATIO,
    SEGMENT_SNAP_TOLERANCE_SECONDS,
    MIN_FINAL_CLIPS,
    MAX_FINAL_CLIPS,
    SECONDS_PER_FINAL_CLIP,
    HOOK_TITLE_MAX_CHARS,
    HOOK_TITLE_MAX_WORDS,
    TRANSCRIPT_SPAN_RE,
    TRANSCRIPT_LINE_PATTERN,
    SIGNAL_LINE_PATTERN,
    SPEAKER_PREFIX_PATTERN,
    _normalize_transcript_text,
    _parse_transcript_lines,
    _snap_bounds_to_transcript_lines,
    _extract_transcript_text_for_segment,
    ViralityAnalysis,
    _default_virality_analysis,
    TranscriptSegment,
    TranscriptAnalysis,
    transcript_analysis_system_prompt,
    build_transcript_system_prompt,
    RankedCandidate,
    CandidateRanking,
    candidate_rerank_system_prompt,
    BRollCue,
    BRollPlan,
    broll_plan_system_prompt,
    build_broll_plan_prompt,
    _performance_history_section,
    build_candidate_rerank_prompt,
    build_transcript_analysis_prompt,
    sanitize_hook_title,
    _sanitize_hook_span,
    _parse_transcript_timestamp_seconds,
    _format_transcript_timestamp,
    _format_minutes_label,
    _parse_transcript_spans,
    _extract_transcript_text,
    _choose_repaired_bounds,
    _repair_segment_bounds,
    split_transcript_into_windows,
    _filter_clip_signals_for_window,
    _segment_bounds_seconds,
    _segment_rank_key,
    _overlap_ratio,
    dedupe_overlapping_segments,
    max_final_clips_for_duration,
    _rescale_virality,
    apply_candidate_ranking,
    _snap_segment_bounds,
    _validate_segments,
    _merge_key_topics,
)

TRANSCRIPT_ANALYSIS_MAX_ATTEMPTS = 3
TRANSCRIPT_ANALYSIS_TIMEOUT_SECONDS = 600
# Wall-clock budget for all windows together. Each window additionally keeps
# its own TRANSCRIPT_ANALYSIS_TIMEOUT_SECONDS budget once it starts running.
ANALYSIS_TOTAL_TIMEOUT_SECONDS = 1800
RERANK_TIMEOUT_SECONDS = 240
TRANSIENT_MODEL_STATUS_CODES = {408, 429, 500, 502, 503, 504, 529}


# Lazy-loaded agent to avoid import-time failures when API keys aren't set
_transcript_agent: Optional[Agent[None, TranscriptAnalysis]] = None
_transcript_agent_signature: Optional[tuple[str | None, ...]] = None

SUPPORTED_LLM_PROVIDERS = {"google", "google-gla", "openai", "openrouter", "anthropic", "ollama"}


def _split_llm_name(model_name: str) -> tuple[str, str | None]:
    if ":" not in model_name:
        return model_name.strip().lower(), None

    provider, provider_model_name = model_name.split(":", 1)
    return provider.strip().lower(), provider_model_name.strip() or None


def _get_missing_llm_key_error(model_name: str, runtime_config: Config) -> Optional[str]:
    """Return a clear configuration error when the selected LLM key is missing."""
    provider, provider_model_name = _split_llm_name(model_name)

    if provider not in SUPPORTED_LLM_PROVIDERS:
        return (
            f"Unsupported LLM provider '{provider}'. "
            "Use google-gla:*, openai:*, openrouter:*, anthropic:*, or ollama:*."
        )

    if not provider_model_name:
        return (
            "Selected LLM is missing a model name. "
            "Use the format provider:model, for example ollama:gpt-oss:20b."
        )

    if provider in {"google", "google-gla"} and not runtime_config.google_api_key:
        return (
            "Selected LLM provider is Google, but GOOGLE_API_KEY is not set. "
            "Set GOOGLE_API_KEY or set LLM to openai:* / anthropic:* / ollama:* with the matching API key."
        )

    if provider == "openrouter" and not runtime_config.openrouter_api_key:
        return "Selected LLM provider is OpenRouter, but OPENROUTER_API_KEY is not set."

    if provider == "openrouter" and "/" not in provider_model_name:
        return "OpenRouter model must include its vendor, for example openrouter:anthropic/claude-sonnet-5.5."

    if provider == "openai" and not runtime_config.openai_api_key:
        return (
            "Selected LLM provider is OpenAI, but OPENAI_API_KEY is not set. "
            "Set OPENAI_API_KEY or choose another provider with a matching API key."
        )

    if provider == "anthropic" and not runtime_config.anthropic_api_key:
        return (
            "Selected LLM provider is Anthropic, but ANTHROPIC_API_KEY is not set. "
            "Set ANTHROPIC_API_KEY or choose another provider with a matching API key."
        )

    if provider == "ollama":
        # Ollama can run locally without an API key. OLLAMA_BASE_URL/OLLAMA_API_KEY
        # are optional and passed through as environment variables.
        return None

    return None


def _uses_native_output(model_name: str) -> bool:
    provider, name = _split_llm_name(model_name)
    return provider == "openrouter" or (
        provider == "anthropic" and name == "claude-sonnet-5-5"
    )


def _build_transcript_model(
    runtime_config: Config, model_name: str | None = None
) -> Model | str:
    model_name = model_name or runtime_config.llm
    provider, provider_model_name = _split_llm_name(model_name)
    if provider == "anthropic" and _uses_native_output(model_name):
        # The installed SDK predates Sonnet 5.5. This model requires native
        # structured output rather than forced tool calls.
        profile = AnthropicProvider.model_profile(provider_model_name).update(
            AnthropicModelProfile(supports_json_schema_output=True)
        )
        return AnthropicModel(
            provider_model_name,
            provider=AnthropicProvider(api_key=runtime_config.anthropic_api_key),
            profile=profile,
            settings={"max_tokens": 8192},
        )
    if provider == "openrouter":
        fallback = runtime_config.openrouter_fallback_model
        if not provider_model_name or "/" not in provider_model_name or "/" not in fallback:
            raise RuntimeError("OpenRouter primary and fallback models must use vendor/model IDs.")
        # The installed SDK predates Sonnet 5.5. Both selected models advertise
        # native JSON schema output; require supporting endpoints when routing.
        profile = OpenRouterProvider.model_profile(provider_model_name).update(
            OpenAIModelProfile(supports_json_schema_output=True)
        )
        return OpenRouterModel(
            provider_model_name,
            provider=OpenRouterProvider(api_key=runtime_config.openrouter_api_key),
            profile=profile,
            settings={
                "openrouter_models": [fallback] if fallback != provider_model_name else [],
                "openrouter_provider": {"allow_fallbacks": True, "require_parameters": True, "data_collection": "deny"},
                "openrouter_reasoning": {"effort": "low"},
                "max_tokens": 8192,
            },
        )
    if provider != "ollama":
        return model_name

    if not provider_model_name:
        raise RuntimeError(
            "Selected LLM provider is Ollama, but no model name was provided. "
            "Use the format ollama:<model>, for example ollama:gpt-oss:20b."
        )

    return OllamaModel(
        provider_model_name,
        provider=OllamaProvider(
            base_url=runtime_config.resolve_ollama_base_url(),
            api_key=runtime_config.ollama_api_key,
        ),
    )


def get_transcript_agent(clip_length: str = "auto") -> Agent[None, TranscriptAnalysis]:
    """Get or create the transcript analysis agent (lazy initialization)."""
    global _transcript_agent, _transcript_agent_signature
    runtime_config = get_config()
    provider, _ = _split_llm_name(runtime_config.llm)
    signature = (
        clip_length,
        runtime_config.llm,
        runtime_config.openai_api_key,
        runtime_config.openrouter_api_key,
        runtime_config.openrouter_fallback_model,
        runtime_config.google_api_key,
        runtime_config.anthropic_api_key,
        runtime_config.ollama_base_url,
        runtime_config.ollama_api_key,
    )
    if _transcript_agent is None or _transcript_agent_signature != signature:
        apply_settings_to_process_env(runtime_config.as_runtime_settings())
        config_error = _get_missing_llm_key_error(runtime_config.llm, runtime_config)
        if config_error:
            raise RuntimeError(config_error)

        _transcript_agent = Agent[None, TranscriptAnalysis](
            model=_build_transcript_model(runtime_config),
            output_type=NativeOutput(TranscriptAnalysis, strict=True) if _uses_native_output(runtime_config.llm) else TranscriptAnalysis,
            system_prompt=build_transcript_system_prompt(clip_length),
            # Some local Ollama/OpenAI-compatible endpoints can return formatted
            # prose before settling on schema-valid JSON. Keep retries limited
            # while still allowing enough repair attempts for local models.
            output_retries=2 if provider == "ollama" else 2,
        )
        _transcript_agent_signature = signature
    return _transcript_agent

_rerank_agent: Optional[Agent[None, CandidateRanking]] = None
_rerank_agent_signature: Optional[tuple[str | None, ...]] = None


def get_rerank_agent() -> Agent[None, CandidateRanking]:
    """Get or create the candidate rerank agent (lazy initialization)."""
    global _rerank_agent, _rerank_agent_signature
    runtime_config = get_config()
    model_name = getattr(runtime_config, "rerank_llm", None) or runtime_config.llm
    provider, _ = _split_llm_name(model_name)
    signature = (
        model_name,
        runtime_config.openai_api_key,
        runtime_config.openrouter_api_key,
        runtime_config.openrouter_fallback_model,
        runtime_config.google_api_key,
        runtime_config.anthropic_api_key,
        runtime_config.ollama_base_url,
        runtime_config.ollama_api_key,
    )
    if _rerank_agent is None or _rerank_agent_signature != signature:
        apply_settings_to_process_env(runtime_config.as_runtime_settings())
        config_error = _get_missing_llm_key_error(model_name, runtime_config)
        if config_error:
            raise RuntimeError(config_error)

        _rerank_agent = Agent[None, CandidateRanking](
            model=_build_transcript_model(runtime_config, model_name),
            output_type=NativeOutput(CandidateRanking, strict=True) if _uses_native_output(model_name) else CandidateRanking,
            system_prompt=candidate_rerank_system_prompt,
            output_retries=2,
        )
        _rerank_agent_signature = signature
    return _rerank_agent

_broll_agent: Optional[Agent[None, BRollPlan]] = None
_broll_agent_signature: Optional[tuple[str | None, ...]] = None
BROLL_PLAN_TIMEOUT_SECONDS = 60


def get_broll_agent() -> Agent[None, BRollPlan]:
    """Get or create the B-roll planning agent (lazy initialization)."""
    global _broll_agent, _broll_agent_signature
    runtime_config = get_config()
    model_name = runtime_config.llm
    signature = (
        model_name,
        runtime_config.openai_api_key,
        runtime_config.openrouter_api_key,
        runtime_config.openrouter_fallback_model,
        runtime_config.google_api_key,
        runtime_config.anthropic_api_key,
        runtime_config.ollama_base_url,
        runtime_config.ollama_api_key,
    )
    if _broll_agent is None or _broll_agent_signature != signature:
        apply_settings_to_process_env(runtime_config.as_runtime_settings())
        config_error = _get_missing_llm_key_error(model_name, runtime_config)
        if config_error:
            raise RuntimeError(config_error)
        _broll_agent = Agent[None, BRollPlan](
            model=_build_transcript_model(runtime_config, model_name),
            output_type=NativeOutput(BRollPlan, strict=True) if _uses_native_output(model_name) else BRollPlan,
            system_prompt=broll_plan_system_prompt,
            output_retries=2,
        )
        _broll_agent_signature = signature
    return _broll_agent

async def plan_broll_cues(
    timed_lines: str, clip_duration: float, max_cues: int, earliest_start: float
) -> List[BRollCue]:
    """Pick stock-footage moments for one clip. Raises when the model is unavailable."""
    result = await _run_transcript_analysis(
        get_broll_agent(),
        build_broll_plan_prompt(timed_lines, clip_duration, max_cues, earliest_start),
        timeout_seconds=BROLL_PLAN_TIMEOUT_SECONDS,
    )
    return list(result.output.cues)


def _is_transient_model_error(error: Exception) -> bool:
    """Recognize provider overload and transport failures, including SDK wrappers."""
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ModelHTTPError):
            return current.status_code in TRANSIENT_MODEL_STATUS_CODES
        if isinstance(
            current,
            (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError, TimeoutError),
        ):
            return True
        current = current.__cause__
    return False


async def _run_transcript_analysis(
    agent: Agent,
    prompt: str,
    timeout_seconds: Optional[float] = None,
) -> Any:
    """Retry temporary failures on the configured model within one overall budget.

    Agent output retries handle schema repair, not HTTP failures. Retrying only
    the model step preserves completed downloads and transcription. SDK retries
    also count against the wall-clock budget, and cancellation passes through.
    """
    if timeout_seconds is None:
        timeout_seconds = TRANSCRIPT_ANALYSIS_TIMEOUT_SECONDS
    try:
        async with asyncio.timeout(timeout_seconds):
            for attempt in range(1, TRANSCRIPT_ANALYSIS_MAX_ATTEMPTS + 1):
                try:
                    result = await agent.run(prompt)
                    if hasattr(result, "response"):
                        logger.info("AI analysis completed using model %s", result.response.model_name)
                    return result
                except Exception as error:
                    if not _is_transient_model_error(error):
                        raise
                    if attempt == TRANSCRIPT_ANALYSIS_MAX_ATTEMPTS:
                        logger.error(
                            "AI analysis unavailable after %s attempts (%s, HTTP status %s)",
                            attempt,
                            type(error).__name__,
                            getattr(error, "status_code", None),
                        )
                        raise RuntimeError(
                            "The AI analysis service is temporarily unavailable after "
                            "several attempts. Please try again shortly."
                        ) from None

                    delay = 5 * 2 ** (attempt - 1) + random.uniform(0, 1)
                    # Do not log provider bodies: they may contain transcript data.
                    logger.warning(
                        "Temporary AI analysis failure (%s, HTTP status %s); "
                        "retrying attempt %s/%s in %.1fs",
                        type(error).__name__,
                        getattr(error, "status_code", None),
                        attempt + 1,
                        TRANSCRIPT_ANALYSIS_MAX_ATTEMPTS,
                        delay,
                    )
                    await asyncio.sleep(delay)
    except TimeoutError:
        logger.error("AI analysis exceeded the %ss time budget", timeout_seconds)
        raise RuntimeError(
            "The AI analysis service took too long to respond. Please try again shortly."
        ) from None


async def rerank_candidates(
    candidates: List[TranscriptSegment], performance_context: str | None = None
) -> List[TranscriptSegment]:
    """Compare all candidates in one call; keep first-pass order if that fails.

    ``performance_context`` lets the creator's audience history shape the
    final selection too, not just each window's first pass.
    """
    if len(candidates) < 2:
        return candidates
    try:
        agent = get_rerank_agent()
        result = await _run_transcript_analysis(
            agent,
            build_candidate_rerank_prompt(candidates, performance_context),
            timeout_seconds=RERANK_TIMEOUT_SECONDS,
        )
        reranked = apply_candidate_ranking(candidates, result.output)
    except Exception as error:
        # Do not log the error text: provider bodies may contain transcript data.
        logger.warning(
            "Candidate rerank failed (%s); keeping first-pass order", type(error).__name__
        )
        return candidates
    logger.info("Reranked %d clip candidates", len(reranked))
    return reranked


async def _analyze_transcript_windows(
    agent: Agent,
    windows: List[Dict[str, Any]],
    clip_signals: Optional[str],
    performance_context: Optional[str] = None,
    clip_length: str = "auto",
) -> List[TranscriptAnalysis]:
    """Analyse windows with bounded concurrency, tolerating some failures.

    Raises the first window error only when no window produced an analysis.
    """
    semaphore = asyncio.Semaphore(ANALYSIS_WINDOW_CONCURRENCY)

    async def run_window(window: Dict[str, Any]) -> TranscriptAnalysis:
        async with semaphore:
            prompt = build_transcript_analysis_prompt(
                transcript=window["transcript"],
                clip_signals=_filter_clip_signals_for_window(
                    clip_signals, window["start"], window["end"]
                ),
                window=(window["start"], window["end"]),
                performance_context=performance_context,
                clip_length=clip_length,
            )
            result = await _run_transcript_analysis(agent, prompt)
            return result.output

    tasks = [asyncio.create_task(run_window(window)) for window in windows]
    try:
        _, pending = await asyncio.wait(tasks, timeout=ANALYSIS_TOTAL_TIMEOUT_SECONDS)
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    if pending:
        logger.warning(
            "%d of %d analysis windows exceeded the %ss budget",
            len(pending),
            len(windows),
            ANALYSIS_TOTAL_TIMEOUT_SECONDS,
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    analyses: List[TranscriptAnalysis] = []
    errors: List[BaseException] = []
    for index, task in enumerate(tasks):
        if task in pending:
            continue
        error = task.exception()
        if error is not None:
            logger.warning(
                "Analysis window %d/%d failed (%s)", index + 1, len(windows), type(error).__name__
            )
            errors.append(error)
            continue
        analyses.append(task.result())

    if not analyses:
        if errors:
            raise errors[0]
        raise RuntimeError(
            "The AI analysis service took too long to respond. Please try again shortly."
        )
    return analyses


async def get_most_relevant_parts_by_transcript(
    transcript: str,
    clip_signals: str | None = None,
    performance_context: str | None = None,
    clip_length: str = "auto",
    include_reserve: bool = False,
) -> TranscriptAnalysis:
    """Get the most relevant parts of a transcript with virality scoring.

    Short transcripts are analysed in one call. Long ones are split into
    overlapping windows analysed concurrently, then deduplicated and reranked
    in one comparative call so scores are comparable across windows.
    """
    logger.info(
        f"Starting AI analysis of transcript ({len(transcript)} chars)"
    )

    try:
        agent = get_transcript_agent(clip_length)
        transcript_lines = _parse_transcript_lines(transcript)
        transcript_spans = _parse_transcript_spans(transcript)
        windows = split_transcript_into_windows(transcript)

        if len(windows) == 1:
            result = await _run_transcript_analysis(
                agent,
                build_transcript_analysis_prompt(
                    transcript=transcript,
                    clip_signals=clip_signals,
                    performance_context=performance_context,
                    clip_length=clip_length,
                )
            )
            analyses = [result.output]
        else:
            logger.info("Analysing transcript in %d overlapping windows", len(windows))
            analyses = await _analyze_transcript_windows(
                agent, windows, clip_signals, performance_context, clip_length
            )

        candidates: List[TranscriptSegment] = []
        for analysis in analyses:
            logger.info(
                f"AI analysis found {len(analysis.most_relevant_segments)} segments"
            )
            candidates.extend(
                _validate_segments(
                    analysis.most_relevant_segments, transcript_lines, transcript_spans, clip_length
                )
            )

        # Best-first by virality (primary) then relevance (secondary), without
        # near-duplicates; then compare across windows in one rerank call.
        candidates = dedupe_overlapping_segments(candidates)
        if len(windows) > 1:
            candidates = await rerank_candidates(candidates, performance_context)

        source_seconds = max(
            (line["end_seconds"] for line in transcript_lines), default=0
        )
        limit = max_final_clips_for_duration(source_seconds, get_config().max_clips)
        # Retain already-ranked alternatives until picture suitability checks
        # finish. No extra model call is needed when a slate is rejected.
        validated_segments = candidates[: min(24, limit * 2) if include_reserve else limit]

        final_analysis = TranscriptAnalysis(
            most_relevant_segments=validated_segments,
            summary=" ".join(
                analysis.summary.strip() for analysis in analyses if analysis.summary
            ),
            key_topics=_merge_key_topics(analyses),
        )

        logger.info(f"Selected {len(validated_segments)} segments for processing")
        if validated_segments:
            top = validated_segments[0]
            logger.info(
                f"Top segment - relevance: {top.relevance_score:.2f}, virality: {top.virality.total_score if top.virality else 'N/A'}"
            )

        return final_analysis

    except Exception as e:
        logger.error(f"Error in transcript analysis: {e}")
        raise RuntimeError(f"Transcript analysis failed: {str(e)}") from e


def get_most_relevant_parts_sync(transcript: str) -> TranscriptAnalysis:
    """Synchronous wrapper for the async function."""
    return asyncio.run(get_most_relevant_parts_by_transcript(transcript))
