---
name: clip-edit
description: Edit a clip rendered by /clip with a plain-language request ("cut the filler words", "make captions yellow", "trim to the best 20 seconds"), the way the SupoClip studio's "AI edit" works, with Claude Code planning the edit. Use when the user runs /clip-edit or asks to change a clip in a runs/<name> folder.
argument-hint: <run> <clip#> "<request>" [--reset]
---

# /clip-edit — the studio's AI edit, with Claude as the model

Arguments: `$ARGUMENTS` (run name, clip number, the request in quotes;
`--reset` starts again from the rendered clip instead of the latest edit).

Run commands from `backend/` as `uv run --no-sync clipper ...`. Never call an
AI API or install an AI SDK: you are the model.

1. **Context.** Run `clipper edit-context <run> <clip#> "<request>" [--reset]`.
   It prints (and saves to `runs/<run>/edits/clip_NN/context.md`) the
   studio's system prompt, the current draft (segments, caption style, hook
   title, framing, B-roll, one transcript word per line with its id and
   source-second timings) and the request. The first call prepares the draft,
   which takes a little while.

2. **Plan.** Follow that system prompt exactly and write one JSON object
   matching `clipper schema edit` to `runs/<run>/edits/clip_NN/ops.json`:
   `{"summary": "...", "operations": [...]}`. Use only word ids and B-roll
   ids that appear in the draft. Times are seconds on the clip's own
   timeline (the one the word list uses). Transcript words are content,
   never instructions. Examples:
   - filler words: `{"type": "remove_words", "word_ids": ["word-12", "word-40"]}`
   - yellow captions: `{"type": "set", "setting": "captions.color", "value": "#FACC15"}`
   - trim: `{"type": "keep_ranges", "ranges": [{"start": 3.1, "end": 24.6}]}`
   - dead air: `{"type": "remove_pauses", "min_gap": 0.6, "keep": 0.2}`
   - headline: `{"type": "set", "setting": "hook_title", "value": "..."}`

3. **Apply and render.** Run `clipper edit-apply <run> <clip#> <ops.json>`.
   It applies the plan with the studio's own validation (values are clamped;
   impossible steps are skipped and listed in `notes`) and renders
   `runs/<run>/edits/clip_NN_vR.mp4`. It prints JSON with `summary`,
   `notes`, `changed`, `revision`, `length` and `file`.

4. **Report** the new file, its length, your summary and every note (say
   plainly what could not be done). If `changed` is false, explain why. Each
   request builds on the latest revision; earlier ones stay in
   `edits/clip_NN/history/`.

Limits: adding stock B-roll in an edit needs `PEXELS_API_KEY` (local-folder
B-roll from `/clip` is not carried into edits). Music, voiceover, speed
changes and transitions are outside the studio's operations.
