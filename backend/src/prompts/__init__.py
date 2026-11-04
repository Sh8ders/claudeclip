"""Prompt texts, kept as Markdown files so every caller reads one source of truth.

The web app's model calls and Claude-native mode (the ``clipper`` CLI and the
``/clip`` skill) load the same files. Each file holds the exact prompt text
followed by a single trailing newline, which ``load_prompt`` removes.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent

PROMPT_NAMES = (
    "transcript_analysis_system",
    "candidate_rerank_system",
    "broll_plan_system",
    "ai_edit_system",
    "social_caption",
)


def prompt_path(name: str) -> Path:
    if name not in PROMPT_NAMES:
        raise KeyError(f"Unknown prompt {name!r}")
    return PROMPTS_DIR / f"{name}.md"


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    text = prompt_path(name).read_text(encoding="utf-8")
    return text[:-1] if text.endswith("\n") else text
