"""The prompt Markdown files are the single source of truth for every prompt.

The hashes pin the exact text the prompts had when they were moved out of the
Python modules, so the web app's model calls stay byte-for-byte identical.
If you change a prompt on purpose, update its hash here in the same commit.
"""

import hashlib

import pytest

from src import ai
from src.prompts import PROMPT_NAMES, load_prompt, prompt_path
from src.services import ai_edit_service
from src.services import social_caption_service

EXPECTED_SHA256 = {
    "transcript_analysis_system": "d8a65515c200cb3ebff18d4a2964e1da51d61f2301f3eadf6741b233df052c48",
    "candidate_rerank_system": "24949a641273fa881ed5c49cd0daa4570d7c63e68eb4840e8fde345d562a03ea",
    "broll_plan_system": "75b0288d4071845a1107f4a1220931cca3bedbb319456cbeaf91e08725c05fb9",
    "ai_edit_system": "419aeb66ec1763f514fb73b15e563e6769dedb9e6bac7daca015ba751b436389",
    "social_caption": "4d5cae38d8f4cf6cbfed01a8670bc9b8a3b6374bf49bc5899a645f576108702f",
}


@pytest.mark.parametrize("name", PROMPT_NAMES)
def test_prompt_file_text_is_unchanged(name):
    text = load_prompt(name)
    assert hashlib.sha256(text.encode()).hexdigest() == EXPECTED_SHA256[name]
    # One trailing newline in the file, none in the loaded prompt.
    raw = prompt_path(name).read_text(encoding="utf-8")
    assert raw == text + "\n"


def test_python_callers_use_the_prompt_files():
    assert ai.transcript_analysis_system_prompt == load_prompt("transcript_analysis_system")
    assert ai.candidate_rerank_system_prompt == load_prompt("candidate_rerank_system")
    assert ai.broll_plan_system_prompt == load_prompt("broll_plan_system")
    assert ai_edit_service.SYSTEM_PROMPT == load_prompt("ai_edit_system")
    assert social_caption_service.CAPTION_PROMPT == load_prompt("social_caption")


def test_transcript_system_prompt_still_formats():
    rendered = ai.build_transcript_system_prompt("short")
    assert "{minimum}" not in rendered
    assert "Every returned segment must be 15-30 seconds long" in rendered


def test_unknown_prompt_is_rejected():
    with pytest.raises(KeyError):
        load_prompt("nope")
