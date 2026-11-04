"""Build a synthetic Claude-native fixture: a test video, its transcript, a selection.

    uv run --no-sync python scripts/make_native_fixture.py /tmp/fixture

writes ``demo.mp4`` (ffmpeg test pattern + tone, 960x540), ``demo.whisper.json``
(a hand-written transcript in Whisper's JSON shape, with word timings) and
``selection.json`` (two good picks and one that validation rejects). Then:

    clipper prepare /tmp/fixture/demo.mp4 --name demo --transcript-json /tmp/fixture/demo.whisper.json
    cp /tmp/fixture/selection.json ../runs/demo/selection.json
    clipper validate demo && clipper render demo

The tests use the same builders (tests/unit/test_native_pipeline.py).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# (start, end, sentence) on whole seconds, so transcript lines are predictable.
SCRIPT: List[Tuple[int, int, str]] = [
    (0, 6, "Most people think productivity is about working longer hours."),
    (6, 12, "I tracked every hour I worked for one full year."),
    (12, 18, "The results completely changed how I plan my week."),
    (18, 24, "My best output came from just three focused hours a day."),
    (24, 30, "Everything after that was busywork that felt like progress."),
    (30, 36, "So here is the system I use now instead of a to-do list."),
    (36, 42, "Every morning I pick one task that would make the day a win."),
    (42, 48, "I block ninety minutes for it before I open my email."),
    (48, 54, "Then I take a real break and walk outside for ten minutes."),
    (54, 60, "The second block is for the hardest problem on my plate."),
    (60, 66, "Meetings only happen after lunch, never in the morning."),
    (66, 72, "This one rule gave me back almost ten hours every week."),
    (72, 78, "The biggest mistake is treating every task as equally urgent."),
    (78, 84, "If everything is important then nothing actually is."),
    (84, 90, "Try this for two weeks and write down what you finish."),
    (90, 96, "You will be surprised how much less you need to work."),
]

DURATION_SECONDS = 100


def whisper_result() -> Dict[str, Any]:
    """The transcript as openai-whisper's JSON: segments with timed words."""
    segments = []
    for start, end, sentence in SCRIPT:
        tokens = sentence.split()
        step = (end - start - 0.4) / len(tokens)
        words = []
        for index, token in enumerate(tokens):
            word_start = round(start + 0.2 + index * step, 3)
            words.append(
                {
                    "word": f" {token}",
                    "start": float(word_start),
                    "end": float(round(word_start + step * 0.85, 3)),
                    "probability": 0.98,
                }
            )
        segments.append({"start": float(start), "end": float(end), "text": f" {sentence}", "words": words})
    return {"text": "".join(s["text"] for s in segments), "segments": segments, "language": "en"}


def selection() -> Dict[str, Any]:
    """Two grounded picks (exact line timestamps) and one that cannot be grounded."""

    def segment(start: str, end: str, text: str, title: str, scores: Tuple[int, int, int, int], hook_type: str):
        hook, engagement, value, share = scores
        return {
            "start_time": start,
            "end_time": end,
            "text": text,
            "relevance_score": 0.9,
            "reasoning": "Self-contained claim with a concrete payoff.",
            "virality": {
                "hook_score": hook,
                "engagement_score": engagement,
                "value_score": value,
                "shareability_score": share,
                "total_score": hook + engagement + value + share,
                "hook_type": hook_type,
                "virality_reasoning": "Contrarian opener backed by a personal experiment.",
            },
            "hook_title": title,
            "hook_start_time": None,
            "hook_end_time": None,
        }

    text = {start: sentence for start, _end, sentence in SCRIPT}
    return {
        "most_relevant_segments": [
            segment(
                "00:00", "00:30",
                " ".join(text[s] for s in (0, 6, 12, 18, 24)),
                "Three hours beat a ten hour day", (22, 20, 21, 20), "contrast",
            ),
            segment(
                "00:36", "01:12",
                " ".join(text[s] for s in (36, 42, 48, 54, 60, 66)),
                "The morning rule that saves ten hours", (20, 19, 22, 19), "statement",
            ),
            segment(
                "05:00", "05:30",
                "This span is not in the transcript at all.",
                "A pick validation must reject", (10, 10, 10, 10), "none",
            ),
        ],
        "summary": "A creator explains how tracking a year of work led to a three-hour focus routine.",
        "key_topics": ["productivity", "focus", "time blocking"],
    }


def make_video(path: Path, seconds: int = DURATION_SECONDS) -> Path:
    """A moving test pattern with a tone: never a still picture, always audio."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc2=size=960x540:rate=25:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=220:sample_rate=48000:duration={seconds}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "96k", "-shortest", str(path),
        ],
        check=True,
    )
    return path


def build(directory: Path) -> Dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    video = make_video(directory / "demo.mp4")
    transcript = directory / "demo.whisper.json"
    transcript.write_text(json.dumps(whisper_result(), indent=2))
    picks = directory / "selection.json"
    picks.write_text(json.dumps(selection(), indent=2))
    return {"video": video, "transcript": transcript, "selection": picks}


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "native-fixture")
    for kind, path in build(target).items():
        print(f"{kind}: {path}")
