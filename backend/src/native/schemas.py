"""JSON files Claude writes in a run folder, and their schemas (``clipper schema``)."""

from __future__ import annotations

import json
from typing import Dict

from pydantic import Field, RootModel

from ..clip_selection import BRollPlan, CandidateRanking, TranscriptAnalysis


class NativeRanking(CandidateRanking):
    """``ranking.json``: the cross-window rerank, tied to one candidates.json."""

    candidates_fingerprint: str = Field(
        default="",
        description="The fingerprint printed in rerank.md / candidates.json, so a stale ranking is caught",
    )


class BRollFile(RootModel[Dict[str, BRollPlan]]):
    """``broll.json``: clip number (as a string, "1", "2", ...) -> that clip's cutaways."""


def schema_for(kind: str) -> dict:
    if kind == "selection":
        return TranscriptAnalysis.model_json_schema()
    if kind == "ranking":
        return NativeRanking.model_json_schema()
    if kind == "broll":
        return BRollFile.model_json_schema()
    if kind == "edit":
        from ..services.ai_edit_service import EditPlan

        return EditPlan.model_json_schema()
    raise KeyError(kind)


SCHEMA_KINDS = ("selection", "ranking", "broll", "edit")

SCHEMA_NOTES = {
    "selection": "selection.json (or windows/NN.selection.json): one TranscriptAnalysis object.",
    "ranking": "ranking.json: every candidate_id from candidates.json exactly once, best first; copy candidates_fingerprint.",
    "broll": 'broll.json: {"1": {"cues": [...]}, "2": {"cues": []}} keyed by clip number; times in clip seconds.',
    "edit": "ops.json for `clipper edit-apply`: one EditPlan object (summary + operations).",
}


def render_schema(kind: str) -> str:
    """Valid JSON only; ``SCHEMA_NOTES`` holds the one-line usage note."""
    return json.dumps(schema_for(kind), indent=2)
