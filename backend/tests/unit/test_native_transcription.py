"""Local Whisper backends, speaker-less transcripts, and keyless B-roll sourcing."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.media import transcription
from src.media.speaker_framing import build_clip_speaker_turns, build_speaker_turns


# --------------------------------------------------------------------------
# Backend selection and result conversion
# --------------------------------------------------------------------------


def test_default_backend_is_unchanged_openai_whisper():
    assert transcription.resolve_whisper_backend(None) == "openai"
    assert transcription.resolve_whisper_backend("faster-whisper") == "faster"
    with pytest.raises(ValueError):
        transcription.resolve_whisper_backend("cloud")


@pytest.mark.parametrize(
    "apple, available, expected",
    [
        (True, {"mlx_whisper", "faster_whisper"}, "mlx"),
        (False, {"mlx_whisper", "faster_whisper"}, "faster"),
        (True, {"faster_whisper"}, "faster"),
        (False, set(), "openai"),
    ],
)
def test_auto_backend_prefers_mlx_then_faster(monkeypatch, apple, available, expected):
    monkeypatch.setattr(transcription, "_is_apple_silicon", lambda: apple)
    monkeypatch.setattr(transcription, "_module_available", lambda name: name in available)
    assert transcription.resolve_whisper_backend("auto") == expected


def _word(text, start, end, probability=0.9):
    return SimpleNamespace(word=text, start=start, end=end, probability=probability)


def test_faster_whisper_output_becomes_the_standard_cache(tmp_path):
    segments = [
        SimpleNamespace(start=0.0, end=2.5, text=" Hello there.", words=[_word(" Hello", 0.0, 0.6), _word(" there.", 0.7, 1.2)]),
        SimpleNamespace(start=2.5, end=5.0, text=" Second line here.", words=[_word(" Second", 2.6, 3.0), _word(" line", 3.1, 3.4), _word(" here.", 3.5, 4.0)]),
    ]
    result = transcription._faster_whisper_result(segments, SimpleNamespace(language="en"))
    assert result["text"] == " Hello there. Second line here."
    assert result["language"] == "en"

    video = tmp_path / "clip.mp4"
    transcription.cache_transcript_data(video, result)
    cache = json.loads(video.with_suffix(".transcript_cache.json").read_text())
    assert [w["start"] for w in cache["words"]] == [0, 700, 2600, 3100, 3500]
    assert all(w["speaker"] is None for w in cache["words"])
    assert transcription.format_transcript_for_analysis(result) == [
        "[00:00 - 00:02] Hello there.",
        "[00:02 - 00:05] Second line here.",
    ]


def test_transcribe_dispatches_to_the_selected_backend(monkeypatch, tmp_path):
    audio = tmp_path / "a.mp3"
    monkeypatch.setattr(transcription, "_prepare_audio_for_transcription", lambda path: audio)
    calls = []

    class FakeModel:
        def transcribe(self, path, word_timestamps, language):
            calls.append((path, word_timestamps, language))
            return iter([SimpleNamespace(start=0.0, end=1.0, text=" Hi.", words=[_word(" Hi.", 0.1, 0.4)])]), SimpleNamespace(language="de")

    monkeypatch.setattr(transcription, "_get_faster_whisper_model", lambda name: FakeModel())
    result = transcription.transcribe_with_whisper(tmp_path / "v.mp4", "small", "de_de", "faster")
    assert calls == [(str(audio), True, "de")]
    assert result["segments"][0]["words"][0]["start"] == 0.1


def test_mlx_backend_maps_model_names_and_plain_floats(monkeypatch, tmp_path):
    seen = {}

    def fake_transcribe(path, path_or_hf_repo, word_timestamps, language):
        seen.update(repo=path_or_hf_repo, words=word_timestamps)
        import numpy as np

        return {"text": " Hi.", "language": "en", "segments": [
            {"start": np.float64(0.0), "end": np.float64(1.0), "text": " Hi.",
             "words": [{"word": " Hi.", "start": np.float64(0.25), "end": np.float64(0.5), "probability": np.float32(0.9)}]},
        ]}

    monkeypatch.setitem(sys.modules, "mlx_whisper", types.SimpleNamespace(transcribe=fake_transcribe))
    monkeypatch.setattr(transcription, "_prepare_audio_for_transcription", lambda path: tmp_path / "a.mp3")
    result = transcription.transcribe_with_whisper(tmp_path / "v.mp4", "small", None, "mlx")
    assert seen == {"repo": "mlx-community/whisper-small-mlx", "words": True}
    word = result["segments"][0]["words"][0]
    assert type(word["start"]) is float and type(word["probability"]) is float


def test_web_whisper_provider_passes_the_backend(monkeypatch, tmp_path):
    seen = {}

    def fake(video, model, language, backend):
        seen.update(model=model, backend=backend)
        return {"text": "", "language": "en", "segments": [{"start": 0.0, "end": 1.0, "text": " Hi.", "words": []}]}

    monkeypatch.setattr(transcription, "transcribe_with_whisper", fake)
    config = SimpleNamespace(whisper_model="base", transcription_language="", whisper_backend="faster")
    transcription._get_transcript_with_whisper(tmp_path / "v.mp4", config)
    assert seen == {"model": "base", "backend": "faster"}


# --------------------------------------------------------------------------
# Whisper has no speaker labels: speaker framing falls back to face tracking
# --------------------------------------------------------------------------


def test_speaker_none_words_produce_no_speaker_turns(tmp_path):
    words = [
        {"text": "Hello", "start": 0.0, "end": 0.4, "speaker": None},
        {"text": "world.", "start": 0.5, "end": 0.9, "speaker": None},
        {"text": "Again", "start": 2.0, "end": 2.3},
    ]
    assert build_speaker_turns(words) == []

    video = tmp_path / "talk.mp4"
    transcription.cache_transcript_data(video, {"text": " Hello world.", "language": "en", "segments": [
        {"start": 0.0, "end": 3.0, "text": " Hello world.", "words": [
            {"word": " Hello", "start": 0.0, "end": 0.4, "probability": 0.9},
            {"word": " world.", "start": 0.5, "end": 0.9, "probability": 0.9},
        ]},
    ]})
    assert build_clip_speaker_turns(video, [(0.0, 3.0)]) == ([], [])


def test_whisper_transcript_lines_have_no_speaker_prefix():
    result = {"segments": [{"start": 1.0, "end": 4.0, "text": " No labels here."}]}
    assert transcription.format_transcript_for_analysis(result) == ["[00:01 - 00:04] No labels here."]


# --------------------------------------------------------------------------
# Imported transcripts and keyless B-roll
# --------------------------------------------------------------------------


def test_import_transcript_requires_word_timings(tmp_path):
    from src.native.prepare import import_transcript

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"segments": [{"start": 0, "end": 1, "text": "x"}]}))
    with pytest.raises(ValueError, match="no word timings"):
        import_transcript(tmp_path / "v.mp4", bad)
    bad.write_text(json.dumps(["not", "whisper"]))
    with pytest.raises(ValueError, match="not a Whisper JSON"):
        import_transcript(tmp_path / "v.mp4", bad)


def test_local_footage_matches_filename_keywords(tmp_path):
    from src.native.broll_local import local_footage, match_local_footage

    for name in ("laptop-typing-desk.mp4", "city_street_night.mov", "mountain-hikers.mp4", "notes.txt", ".hidden.mp4"):
        (tmp_path / name).write_bytes(b"x")
    files = local_footage(tmp_path)
    assert [f.name for f in files] == ["city_street_night.mov", "laptop-typing-desk.mp4", "mountain-hikers.mp4"]
    assert match_local_footage("hands typing laptop", files, set()).name == "laptop-typing-desk.mp4"
    assert match_local_footage("mountain summit hikers", files, set()).name == "mountain-hikers.mp4"
    assert match_local_footage("busy city streets", files, set()).name == "city_street_night.mov"
    assert match_local_footage("ocean waves", files, set()) is None
    assert local_footage(tmp_path / "missing") == []


def test_unused_footage_is_preferred(tmp_path):
    from src.native.broll_local import match_local_footage

    first, second = tmp_path / "laptop-a.mp4", tmp_path / "laptop-b.mp4"
    assert match_local_footage("laptop", [first, second], {str(first)}) == second


def test_local_placements_become_overlays_without_downloading(tmp_path):
    from src.broll import broll_overlays_for

    footage = tmp_path / "laptop.mp4"
    footage.write_bytes(b"x")
    placements = [
        {"start": 3.0, "duration": 2.5, "provider": "local", "local_path": str(footage), "footage_offset": 0.0},
        {"start": 9.0, "duration": 2.0, "provider": "local", "local_path": str(tmp_path / "gone.mp4")},
    ]
    overlays = asyncio.run(broll_overlays_for(placements))
    assert overlays == [{"path": footage, "start": 3.0, "duration": 2.5, "offset": 0.0}]


def test_native_broll_without_pexels_or_folder_is_skipped(monkeypatch, tmp_path):
    from src.clip_selection import BRollPlan
    from src.native.broll_local import NativeBRoll

    monkeypatch.setenv("PEXELS_API_KEY", "")
    plans = {"1": BRollPlan.model_validate({"cues": [{"start": 5, "duration": 3, "query": "laptop"}]})}
    broll = NativeBRoll(tmp_path / "v.mp4", plans, broll_dir=tmp_path / "missing")
    assert broll.source is None
    assert asyncio.run(broll.planner(1)([(0.0, 20.0)])) == []
