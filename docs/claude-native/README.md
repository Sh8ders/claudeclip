# Claude-native mode

Clip a video with SupoClip's pipeline where **Claude Code is the AI**. You
give Claude Code a video file or a YouTube URL; it picks the clips, ranks
them, writes hook titles, plans B-roll and drafts social captions itself.
No LLM, AssemblyAI or other AI API keys are needed.

Download (yt-dlp), transcription (local Whisper), grounding and repair,
face-tracked 9:16 cropping, word-synced captions, hook titles and rendering
are the existing Python code, run locally by the `clipper` CLI. The web
app, workers and Docker setup are unchanged.

## Quickstart (3 steps)

1. **Install** (once, macOS):

   ```bash
   brew install ffmpeg uv
   cd backend
   uv sync --extra cpu --extra native --extra mlx   # Intel Mac: leave out --extra mlx
   ```

2. **Open Claude Code in the repository** and run:

   ```
   /clip ~/Movies/interview.mp4
   ```

   or `/clip https://www.youtube.com/watch?v=...`. Options:
   `--clip-length short|medium|long`, `--template hormozi`, `--broll`,
   `--format original`, `--platform tiktok`, `--whisper-model turbo`.

3. **Collect the results** in `runs/<name>/`: `clips/*.mp4` (the clips),
   `clips.json` (times, scores, hook titles, text) and `posts.md` (one
   ready-to-edit social caption per clip). To change a clip afterwards:
   `/clip-edit <name> 2 "cut the filler words and make the captions yellow"`.

The first run downloads the Whisper model (a few hundred MB for `small`),
so it takes longer.

## Mac setup in more detail

- **Python and packages:** `uv` installs Python 3.11+ and everything in
  `backend/.venv`. `--extra cpu` installs PyTorch (needed by Whisper),
  `--extra native` installs faster-whisper, `--extra mlx` installs
  mlx-whisper (Apple Silicon GPU). If `uv sync` says the lockfile needs
  updating, that is expected on this branch (see "Notes" below); it updates
  `uv.lock` for you.
- **ffmpeg** must be on `PATH` (`brew install ffmpeg`).
- **Check it works:** `cd backend && uv run --no-sync clipper --help`.
- **Try it without a real video:** the synthetic fixture below renders two
  clips end to end in about a minute.

### Choosing a Whisper model

`clipper prepare --whisper-model M` or `export CLIPPER_WHISPER_MODEL=M`.
The default is `small`.

| Model | Good for | Speed on a laptop |
|---|---|---|
| `tiny`, `base` | quick tests, clear studio audio | fastest; word timings are rougher |
| `small` (default) | most talking-head content | ~real time or faster on a modern CPU |
| `medium` | accents, noisy audio | several times slower on CPU |
| `turbo` (`large-v3-turbo`) | best accuracy | fast on Apple Silicon with mlx; slow on CPU |
| `large-v3` | maximum accuracy | slow without a GPU |

The backend is picked automatically (`--whisper-backend auto`, or
`CLIPPER_WHISPER_BACKEND`): **mlx** on Apple Silicon when installed, else
**faster-whisper**, else **openai-whisper**. Models are cached after the
first download (`~/.cache/huggingface` for mlx/faster-whisper,
`~/.cache/whisper` for openai-whisper). `--language en` skips language
detection. `--transcript-json file.json` uses an existing Whisper JSON with
word timings instead of transcribing (e.g. `whisper --word_timestamps True
--output_format json`).

## What happens in a run

```
runs/<name>/
  meta.json            duration, clip budget, windows, Whisper info, cleanup settings
  source.mp4           the normalized source (max 1080p/60fps)
  source.transcript_cache.json   word timings used for captions and cuts
  transcript.txt       timestamped transcript lines
  signals.txt          deterministic hints (trigger phrases, audio peaks)
  prompt.md            the selection prompt Claude follows (short sources)
  windows/NN.md        per-window prompts (sources over 15 minutes)
  selection.json       Claude's picks
  rerank.md, ranking.json        long sources: Claude's cross-window ranking
  segments.json        validated segments, render-ready
  validate_report.txt  every segment validation dropped or changed, and why
  broll/NN.md, broll.json        optional B-roll cues
  clips/*.mp4, clips.json        rendered clips
  posts.md             social captions
```

`/clip` drives these commands (run from `backend/`):

```bash
uv run --no-sync clipper prepare <file-or-url> [--name N] [--whisper-model M] [--clip-length auto]
uv run --no-sync clipper schema selection          # also: ranking, broll, edit
uv run --no-sync clipper validate <name>           # exit 0 clean, 2 needs fixes/ranking, 1 error
uv run --no-sync clipper broll-context <name>      # optional
uv run --no-sync clipper render <name> [--template ID] [--font F] [--format vertical|original] [--no-subtitles] [--broll]
uv run --no-sync clipper list templates            # or: fonts
uv run --no-sync clipper edit-context <name> <clip#> "<request>"
uv run --no-sync clipper edit-apply <name> <clip#> ops.json
```

`validate` runs exactly what the web app runs on a model's answer: snap to
transcript lines, repair lengths, ground the text, drop overlaps, apply the
rerank, review openings, drop slates/still pictures, apply the clip budget
(about one clip per 5 minutes, `MAX_CLIPS` default 10).

The prompts are the web app's own and live in `backend/src/prompts/*.md`;
the web app loads the same files, so improving a prompt improves both.

### B-roll without keys

With `--broll`, Claude writes cues per clip (`broll.json`). Footage comes from
Pexels when `PEXELS_API_KEY` is set; otherwise from video files in `broll/` at
the repository root (or `--broll-dir`, `CLIPPER_BROLL_DIR`) matched on
filename words, so `broll/hands-typing-laptop.mp4` matches the query
"hands typing laptop"; otherwise clips render without B-roll.

### Settings

- `CLIPPER_RUNS_DIR`: where run folders go (default `runs/` in the repo).
- `MAX_CLIPS`: ceiling for the clip budget (default 10).
- Cleanup: long pauses and filler words are cut by default, like new web
  tasks. `prepare --no-cut-pauses --no-remove-fillers` keeps them.
- `TEMP_DIR` defaults to `runs/.cache` (YouTube downloads, B-roll cache).

## Known trade-offs

- **No speaker labels.** Whisper does not diarize, so every word has
  `speaker: None`. Active-speaker framing (which needs labels) is skipped and
  the crop falls back to face tracking; podcasts with two people on camera
  will not hard-cut between speakers like AssemblyAI-transcribed web tasks do.
- **Transcript lines differ from the web app's.** Whisper segments its
  output itself, so line lengths and timestamps are not identical to an
  AssemblyAI transcript of the same video.
- **Formats:** the clip renderer supports `vertical` (default; face-tracked,
  blurred fill when no face is found), `vertical_fit`, `vertical_pan`,
  `vertical_split` and `original`. `--format square` goes through the
  editor renderer (centre framing, no face tracking).
- **Speed:** rendering is CPU-bound, roughly real time per clip.

## Try it on a synthetic fixture

No real video needed (this is what the tests use):

```bash
cd backend
uv run --no-sync python scripts/make_native_fixture.py /tmp/clipper-fixture
uv run --no-sync clipper prepare /tmp/clipper-fixture/demo.mp4 --name demo --transcript-json /tmp/clipper-fixture/demo.whisper.json
cp /tmp/clipper-fixture/selection.json ../runs/demo/selection.json
uv run --no-sync clipper validate demo     # NEEDS_FIXES: the third pick is deliberately ungroundable
uv run --no-sync clipper render demo       # renders the two valid picks
```

## Notes

- `uv.lock` on this branch was not regenerated for the new `native`/`mlx`
  extras, because the environment that built it could not reach the
  PyTorch index. `uv sync` updates it on first use; commit the result.
  Docker and CI use `uv sync --frozen` and are unaffected.
- Tests: `cd backend && uv run pytest tests/unit -q` (`test_native_*.py`
  cover this mode; `test_native_purity.py` runs the whole CLI with every AI
  SDK import and all non-local network access blocked).
