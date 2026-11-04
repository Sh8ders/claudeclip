"""Claude-native mode: prepare, validate (incl. the cross-window rerank), B-roll inputs.

Rendering is exercised once, end to end and with AI SDKs blocked, in
test_native_purity.py. Transcription is imported from a hand-written
Whisper-format JSON (``--transcript-json``): this sandbox cannot download
Whisper models, and tests must not depend on them anyway.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import make_native_fixture as fixture  # noqa: E402

from src.native import cli  # noqa: E402
from src.native.runs import Run, write_json  # noqa: E402
from src.native.validate import NEEDS_FIXES, NEEDS_RANKING, OK, validate_run  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")


@pytest.fixture(scope="module")
def fixture_files(tmp_path_factory):
    return fixture.build(tmp_path_factory.mktemp("native-fixture"))


@pytest.fixture
def prepared(fixture_files, tmp_path, monkeypatch):
    monkeypatch.setenv("TEMP_DIR", str(tmp_path / "cache"))
    runs = tmp_path / "runs"
    code = cli.main([
        "--runs-dir", str(runs), "prepare", str(fixture_files["video"]), "--name", "demo",
        "--transcript-json", str(fixture_files["transcript"]),
    ])
    assert code == 0
    return Run(runs / "demo"), runs


def test_prepare_writes_claude_inputs(prepared):
    run, _ = prepared
    meta = run.meta()
    assert meta["duration_seconds"] == pytest.approx(100, abs=0.5)
    assert meta["clip_budget"] == 5
    assert meta["speaker_labels"] is False
    assert meta["words"] == sum(len(sentence.split()) for _, _, sentence in fixture.SCRIPT)
    # Word timings sit where the renderer looks for them, in milliseconds.
    cache = json.loads((run.path / "source.transcript_cache.json").read_text())
    assert cache["words"][0]["start"] == 200 and cache["words"][0]["speaker"] is None
    lines = run.transcript_path.read_text().splitlines()
    assert lines[0] == "[00:00 - 00:06] Most people think productivity is about working longer hours."
    assert len(lines) == len(fixture.SCRIPT)
    assert run.signals_path.read_text().startswith("Deterministic clip-worthiness signals")
    prompt = (run.path / "prompt.md").read_text()
    # The web app's own prompt text, rendered for this run's clip length.
    assert "You are an expert transcript analyst" in prompt
    assert "Every returned segment must be 15-60 seconds long" in prompt
    assert lines[3] in prompt
    assert not (run.path / "windows").exists()


def test_validate_reports_drops_then_passes_once_fixed(prepared, fixture_files):
    run, runs = prepared
    shutil.copy(fixture_files["selection"], run.selection_path)

    result = validate_run(run)
    assert result.status == NEEDS_FIXES and result.exit_code == 2
    assert "#3 05:00-05:30" in result.report and "DROPPED" in result.report
    assert "not aligned to transcript lines" in result.report
    assert [s["start_time"] for s in result.segments] == ["00:00", "00:36"]

    picks = json.loads(run.selection_path.read_text())
    picks["most_relevant_segments"].pop()
    # A near-miss end time (inside a line) is snapped, reported, and kept.
    picks["most_relevant_segments"][1]["end_time"] = "01:10"
    write_json(run.selection_path, picks)
    assert cli.main(["--runs-dir", str(runs), "validate", "demo"]) == 0

    result = validate_run(run)
    assert result.status == OK
    assert "time 00:36-01:10 -> 00:36-01:12" in result.report
    plan = json.loads(run.segments_path.read_text())
    assert plan["status"] == OK
    first = plan["segments"][0]
    assert first["clip"] == 1 and first["virality_score"] == 83
    assert first["hook_title"] == "Three hours beat a ten hour day"
    assert first["text"].startswith("Most people think productivity")


def test_validate_rejects_schema_errors(prepared):
    run, _ = prepared
    run.selection_path.write_text(json.dumps({"most_relevant_segments": [{"start_time": "00:00"}]}))
    result = validate_run(run)
    assert result.status == "ERROR" and result.exit_code == 1
    assert "does not match `clipper schema selection`" in result.report
    assert not run.segments_path.exists()


def test_validate_trims_a_lead_in_opening(prepared):
    run, _ = prepared
    picks = fixture.selection()
    del picks["most_relevant_segments"][1:]
    # Opening on a back-reference ("So here is the system...") gets reviewed.
    picks["most_relevant_segments"][0].update(start_time="00:30", end_time="01:00")
    write_json(run.selection_path, picks)
    result = validate_run(run)
    assert result.status == OK
    assert len(result.segments) == 1


def _long_run(tmp_path: Path) -> Run:
    """A 24-minute transcript (two analysis windows), no video needed."""
    from src.clip_selection import split_transcript_into_windows

    run = Run(tmp_path / "long")
    run.path.mkdir()
    lines = []
    for index in range(0, 24 * 60, 10):
        minutes, seconds = divmod(index, 60)
        end_minutes, end_seconds = divmod(index + 10, 60)
        lines.append(
            f"[{minutes:02d}:{seconds:02d} - {end_minutes:02d}:{end_seconds:02d}] "
            f"Sentence number {index // 10} explains one concrete idea in plain words."
        )
    transcript = "\n".join(lines)
    run.transcript_path.write_text(transcript + "\n")
    windows = split_transcript_into_windows(transcript)
    assert len(windows) == 2
    run.write_meta({
        "name": "long", "source": "x", "video": "missing.mp4", "duration_seconds": 1440,
        "clip_length": "auto", "max_clips": 10,
        "windows": [
            {"index": i, "start": w["start"], "end": w["end"], "selection": f"windows/{i:02d}.selection.json"}
            for i, w in enumerate(windows, start=1)
        ],
    })
    picks = fixture.selection()
    template = picks["most_relevant_segments"][0]
    for index, (start, end, total) in enumerate([("01:00", "01:30", 70), ("15:00", "15:40", 90)], start=1):
        segment = json.loads(json.dumps(template))
        segment.update(start_time=start, end_time=end, text="placeholder words here")
        segment["virality"].update(hook_score=total // 4, engagement_score=total // 4,
                                   value_score=total // 4, shareability_score=total - 3 * (total // 4))
        write_json(run.path / f"windows/{index:02d}.selection.json", {**picks, "most_relevant_segments": [segment]})
    return run


def test_long_source_needs_claudes_rerank(tmp_path):
    run = _long_run(tmp_path)
    result = validate_run(run, visual_check=False)
    assert result.status == NEEDS_RANKING and result.exit_code == 2
    candidates = json.loads((run.path / "candidates.json").read_text())
    fingerprint = candidates["candidates_fingerprint"]
    # First pass: window 2's pick scored higher, so it is candidate 0.
    assert [c["start_time"] for c in candidates["candidates"]] == ["15:00", "01:00"]
    rerank = (run.path / "rerank.md").read_text()
    assert fingerprint in rerank and "senior short-form video editor" in rerank

    write_json(run.path / "ranking.json", {
        "candidates_fingerprint": fingerprint,
        "ranking": [{"candidate_id": 1, "score": 88, "reason": "better"}, {"candidate_id": 0, "score": 61, "reason": "weaker"}],
    })
    result = validate_run(run, visual_check=False)
    assert result.status == OK
    assert [s["start_time"] for s in result.segments] == ["01:00", "15:00"]
    assert [s["virality_score"] for s in result.segments] == [88, 61]

    write_json(run.path / "ranking.json", {
        "candidates_fingerprint": "stale", "ranking": [{"candidate_id": 0, "score": 50}],
    })
    assert validate_run(run, visual_check=False).status == NEEDS_RANKING


def test_broll_context_matches_the_render_timeline(prepared, fixture_files):
    run, runs = prepared
    picks = fixture.selection()
    del picks["most_relevant_segments"][2]
    write_json(run.selection_path, picks)
    assert validate_run(run).status == OK
    assert cli.main(["--runs-dir", str(runs), "broll-context", "demo"]) == 0
    prompt = (run.path / "broll" / "01.md").read_text()
    assert "stock-footage cutaways" in prompt
    assert "Clip length:" in prompt and "] Most people think productivity" in prompt
    ranges = json.loads((run.path / "broll" / "keep_ranges.json").read_text())
    assert set(ranges) == {"1", "2"}


def test_schema_command_prints_valid_json(capsys):
    for kind in ("selection", "ranking", "broll", "edit"):
        assert cli.main(["schema", kind]) == 0
        schema = json.loads(capsys.readouterr().out)
        assert schema.get("properties") or schema.get("additionalProperties") or schema.get("$defs")


def test_missing_run_is_a_clear_error(tmp_path, capsys):
    assert cli.main(["--runs-dir", str(tmp_path), "validate", "nope"]) == 1
    assert "Run `clipper prepare` first" in capsys.readouterr().err
