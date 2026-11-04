"""``clipper``: SupoClip's pipeline with Claude Code doing the AI steps.

    clipper prepare <file|youtube-url> [--name N] [--whisper-model M]
    clipper schema <selection|ranking|broll|edit>
    clipper validate <run>
    clipper broll-context <run>
    clipper render <run> [--template ID] [--font F] [--format vertical|original] [--no-subtitles] [--broll]
    clipper edit-context <run> <clip#> "<instruction>"
    clipper edit-apply <run> <clip#> ops.json
    clipper list <templates|fonts>

See docs/claude-native/README.md and the /clip skill.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .runs import (
    Run,
    configure_environment,
    default_broll_dir,
    default_runs_dir,
    default_whisper_backend,
    default_whisper_model,
    resolve_run_dir,
)


def _parse_clip_numbers(value: Optional[str]) -> Optional[List[int]]:
    if not value:
        return None
    numbers = []
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            first, last = part.split("-", 1)
            numbers.extend(range(int(first), int(last) + 1))
        elif part:
            numbers.append(int(part))
    return sorted(set(numbers))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clipper",
        description="Claude-native SupoClip: deterministic steps here, AI steps in Claude Code.",
    )
    parser.add_argument("--runs-dir", type=Path, default=None, help="Folder holding run folders (default: <repo>/runs, or $CLIPPER_RUNS_DIR)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show the pipeline's progress logs")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Download/copy, normalize, transcribe locally, write Claude's inputs")
    prepare.add_argument("source", help="Video file path or YouTube URL")
    prepare.add_argument("--name", help="Run folder name (default: from the title or file name)")
    prepare.add_argument("--whisper-model", default=None, help="tiny, base, small (default), medium, large-v3, turbo... ($CLIPPER_WHISPER_MODEL)")
    prepare.add_argument("--whisper-backend", default=None, choices=("auto", "faster", "mlx", "openai"), help="Default auto: mlx on Apple Silicon, else faster-whisper, else openai-whisper ($CLIPPER_WHISPER_BACKEND)")
    prepare.add_argument("--language", default=None, help="Force the spoken language (e.g. en, de); default auto-detect")
    prepare.add_argument("--clip-length", default="auto", choices=("auto", "short", "medium", "long"))
    prepare.add_argument("--no-cut-pauses", action="store_true", help="Keep long pauses inside clips")
    prepare.add_argument("--no-remove-fillers", action="store_true", help="Keep filler words (um, uh) inside clips")
    prepare.add_argument("--force", action="store_true", help="Re-copy and re-transcribe even when cached")
    prepare.add_argument("--transcript-json", type=Path, default=None, help="Use an existing Whisper JSON with word timings instead of transcribing")

    schema = sub.add_parser("schema", help="Print the JSON schema of a file Claude writes")
    schema.add_argument("kind", choices=("selection", "ranking", "broll", "edit"))

    validate = sub.add_parser("validate", help="Ground selection.json; write segments.json and validate_report.txt")
    validate.add_argument("run")
    validate.add_argument("--max-clips", type=int, default=None, help="Override MAX_CLIPS for the clip budget")
    validate.add_argument("--no-visual-check", action="store_true", help="Skip the still-picture (slate) check")

    broll = sub.add_parser("broll-context", help="Write broll/NN.md prompts on each clip's final timeline")
    broll.add_argument("run")

    render = sub.add_parser("render", help="Render clips/*.mp4 and clips.json from segments.json")
    render.add_argument("run")
    render.add_argument("--template", default="default", help="Caption template id (`clipper list templates`)")
    render.add_argument("--font", default=None, help="Caption font family (`clipper list fonts`)")
    render.add_argument("--font-size", type=int, default=None)
    render.add_argument("--font-color", default=None, help="#RRGGBB")
    render.add_argument("--format", default="vertical", choices=("vertical", "vertical_fit", "vertical_pan", "vertical_split", "original", "square"), help="square renders through the studio renderer with a face-anchored crop")
    render.add_argument("--no-subtitles", action="store_true", help="No burned captions (the hook title stays)")
    render.add_argument("--broll", action="store_true", help="Add B-roll from broll.json (Pexels if PEXELS_API_KEY, else --broll-dir)")
    render.add_argument("--broll-dir", type=Path, default=None, help="Local footage folder (default: <repo>/broll, or $CLIPPER_BROLL_DIR)")
    render.add_argument("--clips", default=None, help="Only these clip numbers, e.g. 1,3 or 2-4")

    listing = sub.add_parser("list", help="List caption templates or fonts")
    listing.add_argument("what", choices=("templates", "fonts"))

    edit_context = sub.add_parser("edit-context", help="Print the studio AI-edit context for one clip")
    edit_context.add_argument("run")
    edit_context.add_argument("clip", type=int)
    edit_context.add_argument("instruction", nargs="?", default="", help="The edit request")
    edit_context.add_argument("--reset", action="store_true", help="Discard earlier edits and start from the rendered clip")

    edit_apply = sub.add_parser("edit-apply", help="Apply an EditPlan (ops.json) to one clip and render it")
    edit_apply.add_argument("run")
    edit_apply.add_argument("clip", type=int)
    edit_apply.add_argument("ops", type=Path)
    edit_apply.add_argument("--no-render", action="store_true", help="Only update the draft")
    return parser


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s" if verbose else "%(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    if not verbose:
        # The pipeline logs every step at INFO; keep only our own progress.
        logging.getLogger(__name__.rsplit(".", 1)[0]).setLevel(logging.INFO)


def _print(text: str = "") -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def run_command(args: argparse.Namespace) -> int:
    runs_dir = (args.runs_dir or default_runs_dir()).expanduser().resolve()
    configure_environment(runs_dir)

    if args.command == "schema":
        from .schemas import SCHEMA_NOTES, render_schema

        sys.stderr.write(SCHEMA_NOTES[args.kind] + "\n")
        _print(render_schema(args.kind))
        return 0

    if args.command == "list":
        if args.what == "templates":
            from ..caption_templates import CAPTION_TEMPLATES

            for template_id, template in CAPTION_TEMPLATES.items():
                _print(f"{template_id}\t{template.get('name', template_id)}")
        else:
            from ..font_registry import get_available_fonts

            for font in get_available_fonts():
                _print(font.get("name") or str(font))
        return 0

    if args.command == "prepare":
        from ..clip_cleanup import new_task_cleanup_settings
        from .prepare import prepare

        cleanup = new_task_cleanup_settings({})
        if args.no_cut_pauses:
            cleanup["cut_long_pauses"] = False
        if args.no_remove_fillers:
            cleanup["remove_filler_words"] = False
        run = asyncio.run(
            prepare(
                args.source,
                runs_dir,
                name=args.name,
                whisper_model=args.whisper_model or default_whisper_model(),
                whisper_backend=args.whisper_backend or default_whisper_backend(),
                language=args.language,
                clip_length=args.clip_length,
                cleanup=cleanup,
                force=args.force,
                transcript_json=args.transcript_json,
            )
        )
        meta = run.meta()
        _print(f"Run: {run.path}")
        _print(
            f"Source: {meta['duration_seconds'] / 60:.1f} min, {meta['transcript_lines']} transcript lines, "
            f"{meta['words']} words (Whisper {meta['whisper'].get('model')} via {meta['whisper'].get('backend')}, "
            f"language {meta['whisper'].get('language') or 'auto'})"
        )
        _print(f"Clip budget: {meta['clip_budget']} clips ({meta['clip_length']} length)")
        if len(meta["windows"]) > 1:
            _print(f"Long source: {len(meta['windows'])} windows. For each, read windows/NN.md and write windows/NN.selection.json.")
        else:
            _print("Next: read prompt.md and write selection.json, then `clipper validate`.")
        return 0

    run = Run(resolve_run_dir(args.run, runs_dir))

    if args.command == "validate":
        from .validate import validate_run

        result = validate_run(run, max_clips=args.max_clips, visual_check=not args.no_visual_check)
        _print(result.report.rstrip("\n"))
        return result.exit_code

    if args.command == "broll-context":
        from .render import write_broll_context

        written = write_broll_context(run)
        _print(f"Wrote {len(written)} B-roll prompts in {run.path / 'broll'}. Write broll.json, then `clipper render --broll`.")
        return 0

    if args.command == "render":
        from .render import render_run, summary_table

        if args.format == "square":
            from .edit import render_square

            clips = asyncio.run(
                render_square(
                    run, template=args.template, font=args.font, subtitles=not args.no_subtitles,
                    only=_parse_clip_numbers(args.clips),
                )
            )
        else:
            clips = asyncio.run(
                render_run(
                    run,
                    template=args.template,
                    font=args.font,
                    font_size=args.font_size,
                    font_color=args.font_color,
                    output_format=args.format,
                    subtitles=not args.no_subtitles,
                    broll=args.broll,
                    broll_dir=(args.broll_dir or default_broll_dir()),
                    only=_parse_clip_numbers(args.clips),
                )
            )
        _print(summary_table(clips))
        failed = [clip["clip"] for clip in clips if clip.get("error")]
        if failed:
            _print(f"Failed clips: {failed}")
            return 1
        return 0

    if args.command == "edit-context":
        from .edit import edit_context

        _print(asyncio.run(edit_context(run, args.clip, args.instruction, reset=args.reset)))
        return 0

    if args.command == "edit-apply":
        from .edit import edit_apply

        result = asyncio.run(edit_apply(run, args.clip, args.ops, render=not args.no_render))
        _print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    raise AssertionError(args.command)


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    try:
        return run_command(args)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        sys.stderr.write(f"clipper: {error}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
