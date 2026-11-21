"""The mind's working-view notes as a fold leaves them, for the context-compaction suites and their neighbours.

Moved verbatim out of ``tests/test_compaction.py``, which imports them back: a tool unit (``_unit``,
``_large_unit``), the reclaim request and the mind's authored fold over it (``_request``, ``_authored``, ``_fold``),
one earlier note that fold leaves (``_note``) and the same note as older releases wrote it (``_unattributed``: no
authorship; ``_undated``: no write time). ``tests/test_arrival_notes.py`` and ``tests/test_context_view_tool.py``
read them here, never through a test module.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace

from ouroboros import context_compaction as cc
from ouroboros.context_budget import ContextReclaimRequest


def _unit(call_id: str, *, argument: str = "a", result: str = "r") -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": f"reasoning-{call_id}",
            "tool_calls": [{
                "id": call_id,
                "function": {"name": "read_file", "arguments": argument},
            }],
        },
        {"role": "tool", "tool_call_id": call_id, "content": result},
    ]


def _large_unit(call_id: str, marker: str = "x") -> list[dict]:
    return _unit(
        call_id,
        argument=json.dumps({"content": marker * 4_000 + f"-arg-tail-{call_id}"}),
        result=marker * 4_000 + f"-result-tail-{call_id}",
    )


def _request(messages: list[dict], *, goal: int = 1_000_000) -> ContextReclaimRequest:
    return ContextReclaimRequest(
        route_fp="main-route",
        round_id="round-7",
        transcript_sha256=cc.context_reclaim_transcript_sha256(messages),
        measurement_basis="fresh_route_usage",
        measurement_density=1.0,
        reclaim_goal_tokens=goal,
        allow_partial_shrink=True,
    )


def _authored(messages: list[dict], *, note: str = "I read the evidence; the conclusion holds.", keep=()) -> ContextReclaimRequest:
    """The mind's own fold of the observed view: a note, the units it keeps raw."""
    return replace(_request(messages), working_note=note,
                   expected_view_revision=cc.context_reclaim_transcript_sha256(messages), keep_unit_ids=keep)


def _fold(messages: list[dict], request: ContextReclaimRequest, tmp_path):
    return cc.compact_tool_history_llm(
        messages, request=request, observed_messages=messages, tool_schemas=[],
        fit_candidate=lambda *_: {"accepted": True}, drive_root=tmp_path, task_id="compaction")


def _note(tmp_path, text: str, call_id: str = "noted") -> dict:
    """One earlier note of the mind: the capsule an applied fold left in place of its source."""
    source = _large_unit(call_id)
    folded, receipt = _fold(source, _authored(source, note=text), tmp_path)
    assert receipt.status == "applied" and len(folded) == 1
    return folded[0]


def _unattributed(capsule: dict) -> dict:
    """The same capsule as a release before authorship metadata wrote it: still a valid note, attributed to no one."""
    legacy = copy.deepcopy(capsule)
    meta = legacy["content"][0]["_context_capsule"]
    del meta["authorship"]
    meta["summary_contract_digest"] = cc._LEGACY_CAPSULE_CONTRACT_DIGEST
    assert cc._capsule_metadata(legacy)[1] is not None
    return legacy


def _undated(capsule: dict) -> dict:
    """The same note as a release before notes carried their write time wrote it: its old label, no ``written_at``."""
    legacy = copy.deepcopy(capsule)
    block = legacy["content"][0]
    block["text"] = "[Actor-authored working view; exact source retained by checkpoint]\n" + block["text"].partition("\n")[2]
    del block["_context_capsule"]["written_at"]
    block["_context_capsule"]["visible_sha256"] = cc._sha256(block["text"])
    assert cc._capsule_metadata(legacy)[1] is not None
    return legacy
