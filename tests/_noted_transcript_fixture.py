"""A real transcript with one authored working note (the mind's own fold), then later work: the context renewal's
fixture (tests/test_context_renewal.py), shared with the memory capture's working-notes reader
(tests/test_memory_capture_reads.py)."""

from __future__ import annotations

from dataclasses import replace

from ouroboros import context_compaction as cc
from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, bound_host_row
from ouroboros.loop_messages import CONTEXT_FACTS_NAME
from ouroboros.main_context_authority import PREDECESSOR_NOTES_NAME
from tests._context_reclaim_fixtures import _owner, _request, _tool, _transcript

OWNER_LATER = "Owner: also keep the CLI flags."
NOTE = "I verified the migration on both code paths; the old API stays."
FEEDBACK = {"role": "user", "content": "[Task acceptance feedback] Review verdict: REVISE.",
            "review_feedback": [{"task_id": "renewal", "run_index": 0, "binding_hash": "b", "settled": True}]}
LEGACY_FACTS = {"role": "user", HOST_CONTEXT_KIND_KEY: CONTEXT_FACTS_NAME,  # written before binding, merged later
                "content": "[CONTEXT_FACTS] round 5\n\n---\n\n[Message from my human]: B"}
PROSE_CALL = {"role": "assistant", "content": "Checking the flag parser next.", "tool_calls": [{
    "id": "four", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"cli.py\"}"}}]}
# A repetition loop cut at the output limit: arguments nesting past the JSON decoder's depth chose no read.
# Python 3.14 bounds the decoder by its C stack (about 115k levels on macOS), not the 1000-frame limit.
DEEP_CALL = [{"role": "assistant", "content": "", "tool_calls": [{"id": "deep", "type": "function", "function": {
    "name": "read_file", "arguments": '{"path": ' + "[" * 1_000_000}}]},
    {"role": "tool", "tool_call_id": "deep", "content": "⚠️ TOOL_ARG_ERROR: Could not parse arguments for 'read_file'"}]


def _noted(tmp_path):
    """A real transcript with one authored note (the mind's own fold), then later work."""
    messages = _transcript()
    request = replace(_request(messages, 0), working_note=NOTE,
                      expected_view_revision=cc.context_reclaim_transcript_sha256(messages), keep_unit_ids=())
    folded, receipt = cc.compact_tool_history_llm(messages, request=request, observed_messages=messages,
                                                     tool_schemas=[], drive_root=tmp_path, task_id="renewal",
                                                     fit_candidate=lambda _m, _t: {"accepted": True})
    assert receipt.status == "applied"
    return [*folded,
            {"role": "user", HOST_CONTEXT_KIND_KEY: PREDECESSOR_NOTES_NAME, "content": "[Predecessor notes] earlier work"},
            LEGACY_FACTS, bound_host_row(CONTEXT_FACTS_NAME, "[CONTEXT_FACTS] round 9"),
            *_tool("three", "big body " * 2000),
            {"role": "assistant", "content": "My reasoning since the note."},
            PROSE_CALL, {"role": "tool", "tool_call_id": "four", "content": "def parse(): ..."}, *DEEP_CALL,
            _owner(OWNER_LATER), FEEDBACK,
            {"role": "user", "content": "A plain notice delivered before typing."}]
