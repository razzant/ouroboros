"""The dialogue transcript and the reclaim request the context-compaction suites build their cases on.

Moved verbatim out of ``tests/test_context_attention_mind.py`` (``_tool``, ``_owner``, ``_transcript`` and the
words they carry) and ``tests/test_context_reclaim_materializer.py`` (``_request``), which import them back: a
suite that reads them through those modules keeps the shapes it was written against, and
``tests/_noted_transcript_fixture.py`` builds on them without importing a test module.
"""

from __future__ import annotations

import json

from ouroboros import context_compaction as cc
from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, OWNER_WORDS_KIND, ContextReclaimRequest

OWNER = ("Owner: keep the old API alive until the migration is verified. " * 20).strip()
HOST = "[Context view receipt] status=applied " + "detail " * 400
ASSISTANT = "I concluded the migration is safe because both code paths are covered. " * 30
# _transcript() rows: 0 system, 1 assignment, 2-3 tool "one", 4 own prose, 5 owner words, 6 host prose, 7-8 tool "two"


def _tool(call_id="read", text="exact evidence " * 400):
    return [{"role": "assistant", "content": "", "tool_calls": [{"id": call_id, "type": "function", "function": {
        "name": "read_file", "arguments": json.dumps({"path": f"{call_id}.py"})}}]},
        {"role": "tool", "tool_call_id": call_id, "content": text}]


def _owner(text):
    """The owner's words as delivery types them (``context_budget.OWNER_WORDS_KIND``)."""
    return {"role": "user", HOST_CONTEXT_KIND_KEY: OWNER_WORDS_KIND, "content": text}


def _transcript():
    return [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "Assignment: migrate the API."},
            *_tool("one"), {"role": "assistant", "content": ASSISTANT}, _owner(OWNER),
            {"role": "user", "content": HOST}, *_tool("two")]


def _request(
    messages: list[dict],
    goal: int,
    *,
    allow_partial: bool = True,
    measurement_basis: str = "cold_estimate",
    measurement_density: float = 1.0,
):
    return ContextReclaimRequest(
        route_fp="main-route",
        round_id="round-1",
        transcript_sha256=cc.context_reclaim_transcript_sha256(messages),
        measurement_basis=measurement_basis,
        measurement_density=measurement_density,
        reclaim_goal_tokens=goal,
        allow_partial_shrink=allow_partial,
    )
