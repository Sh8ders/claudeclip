# Claude-native mode — plan and progress

This file is the working memory for the Claude-native work. If a session is
compacted or restarted, read this first, then `git log --oneline claude-native`.

## Goal

Give Claude Code a video file or YouTube URL and have Claude Code itself do
every AI step (clip selection, cross-window rerank, B-roll cues, social
captions, studio AI edits). No LLM, AssemblyAI or other paid AI keys.
Download, local Whisper transcription, grounding/repair, cropping, captions
and rendering stay in the existing Python code.

## Ground rules (from the request)

- Branch `claude-native`; PRs only against `Sh8ders/claudeclip` `main`.
  Nothing on `FujiwaraChoki/supoclip`.
- Additive: web app, workers, Docker and the test suite keep working.
  No provider is removed.
- The native path never imports or calls `pydantic_ai`, `anthropic`,
  `openai`, `assemblyai` or a hosted AI API. A test enforces it.
- Commit and push after every working milestone.
- Unit tests: `cd backend && uv run pytest tests/unit -q` (`--no-cov` for
  focused runs).

## Environment notes (this sandbox)

- `download.pytorch.org`, `huggingface.co` and the OpenAI Whisper model host
  are blocked by the network policy. Install here with
  `uv sync --extra cpu --no-install-package torch` and run commands with
  `uv run --no-sync ...`. torch / Whisper models cannot be downloaded, so a
  real transcription cannot run here: transcription is mocked in tests.
- Baseline before any change: `tests/unit` = 1644 passed, 1 skipped.
- Original prompt sha256 (for the byte-identity test):
  - transcript_analysis_system `d8a65515c200cb3e…` (10520 chars)
  - candidate_rerank_system `24949a641273fa88…` (1647)
  - broll_plan_system `75b0288d4071845a…` (2012)
  - ai_edit_system `419aeb66ec1763f5…` (2274)
  - social_caption `4d5cae38d8f4cf6c…` (571)

## Design

### Import hygiene (needed so the native path is AI-SDK-free)

- `src/ai.py` imports `pydantic_ai` at module top, and `video_service.py`
  imports `ai`. `media/common.py` and `media/transcription.py` import
  `assemblyai` at module top, and every renderer module imports `common`.
- Fix, behaviour-preserving:
  1. New `src/clip_selection.py` holds everything in `ai.py` that is pure:
     the Pydantic output models (`TranscriptSegment`, `TranscriptAnalysis`,
     `CandidateRanking`, `BRollPlan`, ...), prompt builders, grounding and
     repair (`_validate_segments`, `dedupe_overlapping_segments`,
     `apply_candidate_ranking`, `split_transcript_into_windows`,
     `max_final_clips_for_duration`, ...). `ai.py` re-imports every name, so
     `from src.ai import X` and tests that monkeypatch `ai.*` are unchanged.
     Logger name stays `src.ai`.
  2. `assemblyai` becomes a lazy module proxy in `media/common.py`
     (`aai.Transcript` etc. still work; import happens on first use).
  3. `video_service.py` takes `max_final_clips_for_duration` from
     `clip_selection` and wraps `get_most_relevant_parts_by_transcript` in a
     module-level function that imports `ai` lazily (still patchable).

### Prompts as files (single source of truth)

`backend/src/prompts/*.md`, loaded by `src/prompts/__init__.py:load_prompt`:
`transcript_analysis_system.md`, `candidate_rerank_system.md`,
`broll_plan_system.md`, `ai_edit_system.md`, `social_caption.md`. The Python
constants (`ai.transcript_analysis_system_prompt`, `ai_edit_service.SYSTEM_PROMPT`,
`social_caption_service.CAPTION_PROMPT`, ...) are assigned from them. A test
checks the loaded text is byte-identical to the original (sha256 above).

### Transcription backends

`media/transcription.py` gains a `WHISPER_BACKEND` setting
(`openai` default = current behaviour, `faster`, `mlx`, `auto`).
faster-whisper / mlx-whisper results are converted into the same
Whisper-shaped dict, so caching (`<video>.transcript_cache.json`) and
`format_transcript_for_analysis` are unchanged. New optional extras:
`native` (faster-whisper) and `mlx` (mlx-whisper, Apple Silicon only).
Native default model: `small` (flag `--whisper-model`, env
`CLIPPER_WHISPER_MODEL`), backend `auto` (mlx on Apple Silicon if installed,
else faster-whisper if installed, else openai-whisper).

### `clipper` CLI (`backend/src/native/`)

Entry point `clipper = "clipper_entry:main"` (`backend/src/clipper_entry.py`
puts `backend/` on `sys.path`, because the editable install exposes
`backend/src` as the import root). Runs live in `<repo>/runs/<name>/`
(gitignored; override with `--runs-dir` or `CLIPPER_RUNS_DIR`). `TEMP_DIR`
defaults to `<runs>/.cache` for the process.

Run folder:

```
runs/<name>/
  meta.json            duration, clip budget, clip_length, windows, whisper info
  source.mp4           normalized source (prepare_source)
  source.transcript_cache.json   word timings (where the renderer looks)
  transcript.txt       format_transcript_for_analysis output
  signals.txt          build_clip_signal_summary
  prompt.md            full selection prompt (system + task) for short sources
  windows/NN.txt       per-window transcript (sources > 15 min)
  windows/NN.md        per-window full prompt; subagent writes windows/NN.selection.json
  selection.json       Claude's TranscriptAnalysis (short sources)
  candidates.json / rerank.md / ranking.json   long sources: Claude's rerank
  segments.json        validated, render-ready segments
  validate_report.txt  every drop / change and why
  broll/NN.md          B-roll prompt per clip (clip timeline)
  broll.json           Claude's cues {"1": {"cues": [...]}, ...}
  clips/*.mp4 + clips.json
  posts.md             Claude's social captions
```

Commands:
- `prepare <file|url> [--name] [--whisper-model] [--whisper-backend] [--language] [--clip-length]`
- `schema <selection|ranking|broll|edit>`
- `validate <run> [--max-clips]` — exit 0 clean, 2 needs action (rerank or fixes), 1 error
- `broll-context <run>` — writes `broll/NN.md` from the same keep ranges render uses
- `render <run> [--template] [--font] [--font-size] [--font-color] [--format] [--no-subtitles] [--broll] [--broll-dir]`
- `list <templates|fonts>`
- Phase 2: `edit-context <run> <clip#> "<instruction>"`, `edit-apply <run> <clip#> ops.json`

### Render

`VideoService.create_single_clip` is reused as-is, with two additive hooks:
keep-range resolution is factored into `VideoService.resolve_clip_keep_ranges`
(so `broll-context` sees the exact ranges render will use) and an optional
`broll_planner` callback replaces the LLM B-roll planner. No Postgres/Redis.

### B-roll without keys

Claude writes cues per clip (clip seconds, `BRollPlan`). `select_cues`
validates them. Footage: Pexels if `PEXELS_API_KEY` is set, else a local
`broll/` folder matched by filename keywords (`--broll-dir`,
`CLIPPER_BROLL_DIR`, default `<repo>/broll`), else skipped quietly.
`broll_overlays_for` learns `provider: "local"` placements (`local_path`).

### Formats

The clip renderer supports `vertical`, `vertical_fit`, `vertical_pan`,
`vertical_split`, `original`. `--format square` renders through the studio
renderer (`render_document`, `framing.aspect = square`) with the horizontal
anchor from `detect_optimal_crop_region(target_ratio=1)`.

### Trade-offs

- Whisper has no speaker labels: words carry `speaker: None`, speaker
  framing returns no turns and face tracking is used. Covered by a test.
- Whisper output is sentence-segmented by the model, not by AssemblyAI
  utterances, so transcript lines differ in length from the web app's.

## Milestones / progress

- [x] Read code, baseline tests green, plan written
- [x] M1 Import hygiene + prompts as files (`test_prompt_files.py`)
- [x] M2 Whisper backends (faster/mlx) + `native`/`mlx` extras. `uv.lock` NOT
      regenerated (pytorch index blocked here); `uv sync` re-locks on the Mac.
- [x] M3 `clipper prepare` / `schema` / `validate` (`test_native_pipeline.py`).
      `prepare --transcript-json` imports an existing Whisper JSON (used by tests).
- [x] M4 `clipper render` + `broll-context` + local/Pexels B-roll
      (`test_native_purity.py` renders end to end, `test_native_transcription.py`)
- [x] M5 Skill `.claude/skills/clip/SKILL.md`, `docs/claude-native/README.md`, CLAUDE.md section
- [x] M6 Purity test (imports + network blocked in a subprocess), full suite
      1678 passed / 1 skipped
- [x] PR against Sh8ders/claudeclip main: https://github.com/Sh8ders/claudeclip/pull/1
- [x] Phase 2: `edit-context` / `edit-apply` (`src/native/edit.py`) + `/clip-edit`
      skill + `render --format square` (studio renderer, face-anchored crop).
      Drafts live in `runs/<name>/edits/clip_NN/` (draft.json, history/);
      covered by the guarded end-to-end test in `test_native_purity.py`.

## Verified by hand (sandbox)

- Fixture render: 1080x1920 H.264/AAC, 30.0 s, hook title + word captions
  burned (no face in the test pattern -> blurred-fit layout).
- Guard negative control: importing `src.ai` and connecting to
  api.anthropic.com are both caught by the purity guard.
