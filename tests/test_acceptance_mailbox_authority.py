"""Mailbox wakeups and unread owner authority are separate facts at the root admission seal.

Context-only task mail wakes the mind but never reopens a seal or hides the
owner-source observation; unread owner or control input keeps finalization open.
"""
from __future__ import annotations

import pytest

from ouroboros.delivery_seal import _end_task_acceptance_fence
from ouroboros.loop_messages import (
    acknowledge_acceptance_observation,
    capture_acceptance_observation,
    owner_source_sha256,
)
from ouroboros.loop_transport import _owner_signal_pending
from ouroboros.owner_mailbox import (
    OwnerMailboxPeek,
    drain_owner_entries,
    write_owner_message,
    write_task_message,
)
from tests.test_acceptance_source_ack_truth import case as _owner_source_case

case = _owner_source_case  # noqa: F811 - pytest fixture re-export


def _fence(tool_ctx):
    tool_ctx._task_acceptance_fence_token = "final-fence"
    tool_ctx._task_acceptance_owner_generation = tool_ctx.owner_message_admission_agent._owner_message_generation
    calls = []

    def end(**kwargs):
        calls.append(kwargs)
        return {"ok": True, "status": "released" if kwargs["outcome"] == "revision" else "sealed"}

    tool_ctx.end_acceptance_fence = end
    return calls


@pytest.mark.parametrize("provenance", ["system", "descendant_task", "independent_task"])
@pytest.mark.parametrize("cached", [False, True])
def test_context_mail_wakes_without_reopening_acceptance(case, provenance, cached):
    tool_ctx, _tools, ctx, _trace, _candidate = case
    before = owner_source_sha256(tool_ctx)
    calls = _fence(tool_ctx)
    assert write_task_message(tool_ctx.drive_root, "Review slot answered PASS.", "root",
                              source_task_id="source", provenance=provenance, msg_id="context")
    peek = OwnerMailboxPeek() if cached else None
    assert _owner_signal_pending(ctx.incoming_messages, tool_ctx.drive_root, "root", set(), 1, peek)
    assert _end_task_acceptance_fence(tool_ctx, outcome="terminal")
    assert calls[-1]["outcome"] == "terminal"
    assert not tool_ctx._task_acceptance_fence_generation_mismatch
    assert owner_source_sha256(tool_ctx) == before
    assert tool_ctx._loop_mailbox_seen_ids == set()
    assert drain_owner_entries(tool_ctx.drive_root, "root", set(), 1)[0]["msg_id"] == "context"


@pytest.mark.parametrize("provenance", ["system", "descendant_task", "independent_task"])
def test_context_mail_does_not_hide_observation_or_refuse_ack(case, provenance):
    tool_ctx, _tools, ctx, trace, _candidate = case
    observed = capture_acceptance_observation(tool_ctx, trace, ctx.incoming_messages)
    assert write_task_message(tool_ctx.drive_root, "Context only.", "root", source_task_id="source",
                              provenance=provenance, msg_id="context")
    assert acknowledge_acceptance_observation(tool_ctx, observed["owner_source_sha256"])
    assert capture_acceptance_observation(tool_ctx, trace, ctx.incoming_messages) == observed
    assert tool_ctx._loop_mailbox_seen_ids == set()


@pytest.mark.parametrize("kind", ["owner_text", "quiz_answer", "hurry", "finalize_now",
                                 "ancestor_task", "peer_via_ancestor", "legacy"])
def test_unread_owner_or_control_keeps_finalization_open(case, kind):
    tool_ctx, _tools, ctx, trace, _candidate = case
    observed = capture_acceptance_observation(tool_ctx, trace, ctx.incoming_messages)
    calls = _fence(tool_ctx)
    if kind in {"ancestor_task", "peer_via_ancestor"}:
        assert write_task_message(tool_ctx.drive_root, "Principal input.", "root",
                                  source_task_id="principal", provenance=kind, msg_id="authority")
    elif kind == "legacy":
        from ouroboros.owner_mailbox import _mailbox_path
        from ouroboros.utils import append_jsonl

        assert append_jsonl(_mailbox_path(tool_ctx.drive_root, "root"),
                            {"msg_id": "authority", "text": "Legacy owner text."})
    else:
        assert write_owner_message(tool_ctx.drive_root, "Owner input.", "root",
                                   msg_id="authority", kind=kind)
    assert not acknowledge_acceptance_observation(tool_ctx, observed["owner_source_sha256"])
    assert capture_acceptance_observation(tool_ctx, trace, ctx.incoming_messages) == {}
    assert _end_task_acceptance_fence(tool_ctx, outcome="terminal")
    assert calls[-1]["outcome"] == "revision"
    assert tool_ctx._task_acceptance_fence_generation_mismatch
    assert tool_ctx._loop_mailbox_seen_ids == set()


def test_direct_incoming_owner_input_still_prevents_final_seal(case):
    tool_ctx, _tools, ctx, _trace, _candidate = case
    calls = _fence(tool_ctx)
    ctx.incoming_messages.put("One more owner message.")
    assert _end_task_acceptance_fence(tool_ctx, outcome="terminal")
    assert calls[-1]["outcome"] == "revision"
    assert not ctx.incoming_messages.empty()
