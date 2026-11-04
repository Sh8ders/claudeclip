# Claude-native mode: clip videos with Claude Code

Turn a long video into short, captioned, vertical clips from inside
[Claude Code](https://code.claude.com/docs), with **no AI API keys**.

You type one command:

```
/clip ~/Movies/interview.mp4
```

Claude Code then does what the hosted product pays an LLM to do. It reads the
transcript, picks the strongest moments, scores them, writes hook titles,
plans B-roll and drafts a social caption for every clip. Everything else
runs on your own machine: download, transcription with local Whisper,
face-tracked 9:16 cropping, word-synced captions and rendering. That's the
same Python pipeline the SupoClip web app uses.

| | Web app (Docker) | Claude-native mode |
|---|---|---|
| Who picks the clips | An LLM API (Gemini, GPT, Claude, Ollama) | Claude Code, in your session |
| Transcription | AssemblyAI (or Whisper) | Local Whisper |
| Keys needed | AssemblyAI + an LLM key | None (Pexels optional, for stock B-roll) |
| Needs Docker, Postgres, Redis | Yes | No |
| Interface | Browser | Claude Code chat + files on disk |

---

## Contents

1. [Install](#1-install)
2. [Clip your first video](#2-clip-your-first-video)
3. [Edit a clip](#3-edit-a-clip-with-clip-edit)
4. [`/clip` options](#4-clip-options)
5. [What you get](#5-what-you-get-the-run-folder)
6. [B-roll](#6-b-roll)
7. [Choosing a Whisper model](#7-choosing-a-whisper-model)
8. [How it works](#8-how-it-works)
9. [Using `clipper` without Claude Code](#9-using-clipper-by-hand)
10. [Settings](#10-settings-and-environment-variables)
11. [Troubleshooting](#11-troubleshooting)
12. [Limitations](#12-known-limitations)

---

## 1. Install

You need **Claude Code**, **ffmpeg**, **uv** and this repository. A GPU is
not required.

### macOS

```bash
# 1. Tools. Use ffmpeg-full, not ffmpeg: Homebrew's plain ffmpeg is built
#    without libass, so it can't burn in captions (clipper finds ffmpeg-full
#    on its own; you don't need to change PATH).
brew install ffmpeg-full uv git

# 2. This repository
git clone https://github.com/Sh8ders/claudeclip.git
cd claudeclip/backend

# 3. Python environment (Apple Silicon)
uv sync --extra cpu --extra native --extra mlx
#    Intel Mac: uv sync --extra cpu --extra native

# 4. Check it works
uv run --no-sync clipper --help
```

### Linux

```bash
sudo apt-get install -y ffmpeg git      # or your distro's equivalent
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/Sh8ders/claudeclip.git
cd claudeclip/backend
uv sync --extra cpu --extra native
uv run --no-sync clipper --help
```

Windows: use WSL (Ubuntu) and follow the Linux steps.

What the extras install:

| Extra | Installs | Why |
|---|---|---|
| `cpu` | PyTorch (CPU build on Linux) | Needed by Whisper |
| `native` | faster-whisper | Fast local transcription on any CPU |
| `mlx` | mlx-whisper | Transcription on the Apple Silicon GPU (Macs with M-series chips only) |

The first `uv sync` may report that it updated `uv.lock`. That's expected,
because the new extras were added without re-locking. Commit the updated
lockfile if you track this branch.

### Claude Code

Install Claude Code ([setup guide](https://code.claude.com/docs)) and sign
in. Then start it from the **repository root**, not `backend/`, so it finds
the skills in `.claude/skills/`:

```bash
cd claudeclip
claude
```

Type `/` and you should see `clip` and `clip-edit` in the list.

> **Tip: fewer permission prompts.** Claude Code asks before running shell
> commands. When it asks to run `uv run --no-sync clipper ...`, choose the
> option to always allow it for this project.

---

## 2. Clip your first video

In Claude Code:

```
/clip ~/Movies/interview.mp4
```

or a YouTube link:

```
/clip https://www.youtube.com/watch?v=VIDEO_ID
```

Claude will:

1. **Prepare** the video: download or copy it, scale it to at most 1080p,
   and transcribe it with Whisper. The first run also downloads the Whisper
   model, so it takes longer.
2. **Pick the clips** by reading the transcript. Videos over 15 minutes are
   split into windows, and each window goes to its own sub-agent in parallel.
3. **Validate** its picks with SupoClip's own checks, fix anything rejected,
   and run validation again.
4. **Rank** candidates across windows on long videos, so the best clips from
   the whole video win.
5. **Render** each clip as a 1080×1920 MP4 with captions and a hook title.
6. **Write captions**: one social post per clip, saved to `posts.md`.
7. **Report** a table of clips with scores, hook titles and file paths.

A typical result:

```
| # | Time        | Length | Virality | Hook title                         | File                       |
|---|-------------|--------|----------|------------------------------------|----------------------------|
| 1 | 04:12-04:48 | 00:36  | 86       | The $40k mistake I keep seeing     | runs/interview/clips/...   |
| 2 | 17:03-17:41 | 00:38  | 81       | Why nobody tells you this about VC | runs/interview/clips/...   |
```

Open the clips from `runs/<name>/clips/`, or ask Claude to open the folder
for you.

You can also just ask in plain words, for example *"clip this podcast into
short clips with the hormozi captions: ~/Downloads/ep12.mp4"*. Claude uses
the same skill.

---

## 3. Edit a clip with `/clip-edit`

`/clip-edit` works like the "AI edit" box in SupoClip's clip studio. Describe
the change and Claude plans and applies it:

```
/clip-edit interview 2 "cut the filler words and make the captions yellow"
/clip-edit interview 1 "trim it to the best 20 seconds"
/clip-edit interview 3 "remove the long pauses and change the headline to 'Stop doing this'"
/clip-edit interview 2 "start over and just make the captions bigger" --reset
```

The arguments are the run name (the folder in `runs/`), the clip number, and
the request.

What it can change:

- **Cuts:** remove specific words (fillers, stumbles), trim, keep the best
  part, reorder, tighten pauses.
- **Captions:** colour, accent, font, size, position, background box, words
  per line, karaoke highlight, uppercase, template, spelling fixes, emphasis.
- **Hook title:** change or remove it.
- **Framing:** vertical, square or original aspect, zoom, position.
- **Look and sound:** brightness, contrast, saturation, blur, hue, volume,
  mute.
- **B-roll:** add, move or remove stock footage. Adding needs `PEXELS_API_KEY`.

Each edit is saved as a new version, `runs/<name>/edits/clip_02_v1.mp4`,
`_v2.mp4` and so on, and builds on the previous one. Anything the editor
can't do (music, voiceover, speed changes, transitions) is reported back
instead of silently ignored.

---

## 4. `/clip` options

Add these after the video path or URL. Examples:
`/clip talk.mp4 --clip-length short --template hormozi --broll`, or
`/clip talk.mp4 --format square --platform linkedin`.

| Option | Values | Default | What it does |
|---|---|---|---|
| `--name` | text | from the title or file name | Name of the run folder |
| `--clip-length` | `auto`, `short` (15-30 s), `medium` (30-60 s), `long` (60-120 s) | `auto` (15-60 s, ideally 25-50) | Target clip length |
| `--template` | `default`, `hormozi`, `mrbeast`, `minimal`, `tiktok`, `neon`, `podcast` | `default` | Caption style |
| `--font` | a font from `clipper list fonts` | template's font | Caption font |
| `--format` | `vertical`, `square`, `original`, `vertical_fit`, `vertical_pan`, `vertical_split` | `vertical` | Output shape (see below) |
| `--no-subtitles` | flag | off | No captions (the hook title stays) |
| `--broll` | flag | off | Add stock-footage cutaways ([B-roll](#6-b-roll)) |
| `--platform` | e.g. `tiktok`, `instagram`, `youtube`, `linkedin` | all short-form | Tone of the captions in `posts.md` |
| `--whisper-model` | `tiny` … `turbo` | `small` | Transcription accuracy and speed ([guide](#7-choosing-a-whisper-model)) |
| `--language` | e.g. `en`, `de`, `es` | auto-detect | Force the spoken language |
| `--max-clips` | number | 10 | Upper limit on clips. The real count is about 1 per 5 minutes of video. |

Formats:

- **`vertical`**: 1080×1920. The speaker's face is tracked. With no face,
  the full picture is shown over a blurred fill.
- **`square`**: 1080×1080, with the crop placed on the speaker's face.
- **`original`**: keeps the source's shape.
- **`vertical_fit`**: the whole picture over a blurred background.
- **`vertical_pan`**: a smooth pan that follows the action.
- **`vertical_split`**: two speakers stacked.

---

## 5. What you get: the run folder

Every video gets its own folder in `runs/` at the repository root. The
folder is git-ignored.

```
runs/interview/
├── clips/                    ← your finished clips (.mp4)
├── clips.json                ← every clip: times, length, scores, hook title, text, B-roll
├── posts.md                  ← one ready-to-edit social caption per clip
├── edits/                    ← /clip-edit versions (clip_02_v1.mp4, ...) and their drafts
│
├── source.mp4                ← the prepared source video
├── transcript.txt            ← timestamped transcript
├── meta.json                 ← duration, clip budget, Whisper model, settings
├── validate_report.txt       ← which picks were kept, changed or dropped, and why
└── (working files: prompt.md, selection.json, segments.json, signals.txt, windows/, broll/, ...)
```

Running `/clip` on the same video again reuses the cached transcript, so the
second run is much faster. Delete the folder to start completely fresh.

---

## 6. B-roll

With `--broll`, Claude picks moments where the speaker names something you
can film ("biking to work", "a stack of cash") and plans a 2-4 second
cutaway for each. The footage comes from the first of these that's
available:

1. **Pexels.** If `PEXELS_API_KEY` is set (a free key from
   [pexels.com/api](https://www.pexels.com/api/)), matching stock videos are
   downloaded automatically.
2. **Your own library.** Video files in `broll/` at the repository root, or
   in `--broll-dir` / `CLIPPER_BROLL_DIR`. Files are matched by the words in
   their **file names**, so name them descriptively:

   ```
   broll/
   ├── hands-typing-laptop.mp4
   ├── city-street-night.mov
   └── mountain-hikers-summit.mp4
   ```

3. **Nothing.** The clips render without B-roll, and Claude tells you so.

---

## 7. Choosing a Whisper model

Pass `--whisper-model` in `/clip`, or set it once with
`export CLIPPER_WHISPER_MODEL=turbo`.

| Model | Best for | Speed on a laptop |
|---|---|---|
| `tiny`, `base` | Quick tests, clean studio audio | Fastest; word timings are rougher |
| **`small`** (default) | Most talking-head videos | About real time or faster on a modern CPU |
| `medium` | Accents, noisy rooms | Several times slower on CPU |
| `turbo` | Best accuracy for the time | Fast on Apple Silicon (mlx); slow on CPU |
| `large-v3` | Maximum accuracy | Slow without a GPU |

The engine is chosen automatically. Apple Silicon uses **mlx-whisper** (GPU)
when installed, other machines use **faster-whisper**, and the fallback is
**openai-whisper**. You can force one with
`CLIPPER_WHISPER_BACKEND=mlx|faster|openai`. Models are downloaded once and
cached in `~/.cache/huggingface` or `~/.cache/whisper`.

Already have a transcript? Pass a Whisper JSON with word timings to
`clipper prepare --transcript-json file.json` and transcription is skipped.
Whisper writes one with `whisper video.mp4 --word_timestamps True --output_format json`.

---

## 8. How it works

```
 you: /clip video.mp4
        │
        ▼
 clipper prepare ── download/copy ─ normalize ─ Whisper ─► transcript.txt, prompt.md
        │
        ▼
 Claude reads prompt.md ─────────────────────────────────► selection.json
        │
        ▼
 clipper validate ── snap to transcript lines, fix lengths, drop overlaps,
        │            review openings, drop still frames, apply clip budget
        │  ▲
        │  └── report says something was dropped → Claude fixes it, runs again
        ▼
 (long videos) Claude ranks all candidates together ────► ranking.json
        │
        ▼
 (--broll) Claude plans cutaways per clip ───────────────► broll.json
        │
        ▼
 clipper render ── face-tracked crop, captions, hook title, B-roll ─► clips/*.mp4
        │
        ▼
 Claude writes one caption per clip ─────────────────────► posts.md
```

**The `clipper` CLI** (`backend/src/native/`) does every deterministic step
with SupoClip's existing code: yt-dlp downloads, source normalization, the
grounding and repair that keeps clips on real sentence boundaries,
`VideoService.create_single_clip` for rendering, and the clip studio's
`apply_plan` / `render_document` for edits. It never calls an AI service.
A test (`test_native_purity.py`) runs the whole flow with every AI SDK
import and all internet access blocked.

**The skills** (`.claude/skills/clip/` and `.claude/skills/clip-edit/`) tell
Claude Code how to drive `clipper` and what to write at each step.

**The prompts** Claude follows are the web app's own, stored in
`backend/src/prompts/*.md`. The web app loads the same files, so improving a
prompt there improves both.

---

## 9. Using `clipper` by hand

Everything the skills do, you can run yourself from `backend/`. This is
handy for scripting or for re-rendering with different styles:

```bash
cd backend
alias clipper="uv run --no-sync clipper"

clipper prepare ~/Movies/talk.mp4 --name talk      # → runs/talk/
# write runs/talk/selection.json yourself (or let Claude do it)
clipper validate talk                              # exit 0 = clean, 2 = needs fixes/ranking, 1 = error
clipper render talk --template hormozi             # → runs/talk/clips/
clipper render talk --clips 2 --format square      # re-render just clip 2
```

All commands:

| Command | Purpose |
|---|---|
| `clipper prepare <file-or-url> [--name] [--whisper-model] [--whisper-backend] [--language] [--clip-length] [--no-cut-pauses] [--no-remove-fillers] [--transcript-json] [--force]` | Download or copy, normalize, transcribe, write Claude's inputs |
| `clipper schema selection\|ranking\|broll\|edit` | JSON schema of each file Claude writes |
| `clipper validate <run> [--max-clips N] [--no-visual-check]` | Check picks, write `segments.json` and `validate_report.txt` |
| `clipper broll-context <run>` | Write per-clip B-roll prompts (`broll/NN.md`) |
| `clipper render <run> [--template] [--font] [--font-size] [--font-color] [--format] [--no-subtitles] [--broll] [--broll-dir] [--clips 1,3]` | Render clips and `clips.json` |
| `clipper edit-context <run> <clip#> "<request>" [--reset]` | Studio edit prompt and current draft for one clip |
| `clipper edit-apply <run> <clip#> ops.json [--no-render]` | Apply an edit plan and render a new version |
| `clipper list templates\|fonts` | Valid caption templates and fonts |

Add `-v` to any command to see the pipeline's progress logs.

**Try it without a real video.** A synthetic fixture renders two clips in
about a minute:

```bash
cd backend
uv run --no-sync python scripts/make_native_fixture.py /tmp/clipper-fixture
uv run --no-sync clipper prepare /tmp/clipper-fixture/demo.mp4 --name demo \
  --transcript-json /tmp/clipper-fixture/demo.whisper.json
cp /tmp/clipper-fixture/selection.json ../runs/demo/selection.json
uv run --no-sync clipper validate demo   # NEEDS_FIXES: pick #3 is deliberately invalid
uv run --no-sync clipper render demo     # renders the two valid picks
```

---

## 10. Settings and environment variables

All of these are optional.

| Variable | Default | Effect |
|---|---|---|
| `CLIPPER_WHISPER_MODEL` | `small` | Default Whisper model |
| `CLIPPER_WHISPER_BACKEND` | `auto` | `mlx`, `faster` or `openai` |
| `WHISPER_COMPUTE_TYPE` | `int8` | faster-whisper precision (`int8`, `float16`, `float32`) |
| `CLIPPER_RUNS_DIR` | `<repo>/runs` | Where run folders go |
| `CLIPPER_BROLL_DIR` | `<repo>/broll` | Local B-roll library |
| `CLIPPER_FFMPEG_DIR` | unset | Folder holding an ffmpeg with libass, if not Homebrew's `ffmpeg-full` |
| `PEXELS_API_KEY` | unset | Stock B-roll from Pexels |
| `MAX_CLIPS` | `10` | Upper limit on clips per video |
| `TEMP_DIR` | `runs/.cache` | YouTube download and B-roll cache |

Long pauses and filler words ("um", "uh") are cut from clips by default, as
in the web app. To keep them, add `--no-cut-pauses` and
`--no-remove-fillers` to `clipper prepare`, or ask Claude to keep them when
you run `/clip`.

---

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `/clip` doesn't appear in Claude Code | Start `claude` from the repository root (the folder that has `.claude/`), not from `backend/` |
| `clipper: command not found` / `.venv/bin/clipper` missing | Run `cd backend && uv sync --extra cpu --extra native` again |
| `ffmpeg: not found` | `brew install ffmpeg-full` (macOS) or `sudo apt-get install ffmpeg` |
| "Your ffmpeg can't draw captions" or `No such filter: 'subtitles'` | Homebrew's plain `ffmpeg` has no libass. Run `brew install ffmpeg-full`; `clipper` uses it automatically. Elsewhere, install an ffmpeg built with libass or set `CLIPPER_FFMPEG_DIR` to its folder |
| "Whisper is not installed" / torch errors | Re-run `uv sync` with `--extra cpu --extra native` (and `--extra mlx` on Apple Silicon) |
| First run is very slow | The Whisper model is downloading. Later runs reuse it. For speed, try `--whisper-model base`, or `turbo` on Apple Silicon. |
| YouTube download fails | Update yt-dlp (`cd backend && uv lock --upgrade-package yt-dlp && uv sync --extra cpu --extra native`), or download the video yourself and pass the file |
| "Whisper found no speech" | The audio has no speech, or the wrong language was forced. Drop `--language`. |
| A clip starts or ends mid-sentence | Ask `/clip-edit <run> <n> "start on the sentence about X"`, or re-run `/clip` |
| Captions are out of sync | Use a larger Whisper model (`--whisper-model medium` or `turbo`), then re-run with `--force` |
| Validation keeps dropping picks | Read `runs/<name>/validate_report.txt`: every drop lists its reason |
| No B-roll appeared | No `PEXELS_API_KEY` and no matching file names in `broll/`. See [B-roll](#6-b-roll). |

---

## 12. Known limitations

- **No speaker labels.** Whisper doesn't tell speakers apart, so multi-person
  podcasts don't hard-cut between speakers the way AssemblyAI-transcribed web
  tasks can. Cropping falls back to face tracking.
- **Square crops don't pan.** The square crop is placed once per clip, on
  the main face.
- **B-roll from your own folder stays out of edits.** Footage from `broll/`
  is not carried into `/clip-edit` drafts, which only hold Pexels footage.
- **Rendering speed.** Rendering uses the CPU and takes roughly as long as
  the clip itself.
- **Transcript lines.** Whisper splits sentences differently from AssemblyAI,
  so clip boundaries can differ slightly from a web-app run on the same video.

The web app, Docker setup and every existing provider are unaffected by
this mode. You can use both.
