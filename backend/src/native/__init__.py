"""Claude-native mode: run SupoClip's pipeline with Claude Code as the AI.

The ``clipper`` CLI (``src/native/cli.py``) does every deterministic step:
download, local Whisper transcription, grounding and repair, rendering. It
writes plain files into a run folder, and Claude Code (the ``/clip`` skill)
reads the transcript and writes the AI decisions back as JSON.

Nothing in this package may import ``pydantic_ai``, ``anthropic``, ``openai``,
``assemblyai`` or call a hosted AI API; ``tests/unit/test_native_purity.py``
enforces that.
"""
