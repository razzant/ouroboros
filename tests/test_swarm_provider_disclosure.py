"""Swarm host admission joins the real managed provider-death delivery path.

This is the executable P5 K transfer, with no removed router actor or fabricated
terminal notice. Admission/queue/result/annotation writers and forced provider
finalization are real; process startup and paid transport are the unit boundary.
The full websocket/server boundary is covered by test_swarm_routing_browser.
"""

from types import SimpleNamespace

import pytest

import server
from ouroboros import agent_task_pipeline as pipeline, loop, loop_llm_call
from ouroboros.task_finalization import send_provider_death_notice
from ouroboros.task_results import load_task_result
from supervisor.terminal_delivery import build_completed_result_event
from tests.test_delivery_forced_finalization import _forced_test_context
from tests.test_swarm_host_admission import host as host_fixture, incoming_case, rows

host = host_fixture
pytestmark = pytest.mark.serial


def test_last_assistant_salvage_keeps_bytes_and_skips_empty_nonassistant_messages():
    from ouroboros.loop_transport import last_assistant_text

    raw = "  Retained intermediate work.  \n\n"
    empty = [{"role": "assistant", "content": value} for value in (None, "", " \n", [])]
    empty.append({"role": "assistant", "_host_context_kind": "owner_dialogue",
                  "content": "[Owner-directed dialogue]\nRetained delivery metadata"})
    other = [{"role": "user", "content": "New owner input"},
             {"role": "tool", "content": "Tool result"}]
    assert last_assistant_text(empty + other) == ""
    assert last_assistant_text([{"role": "assistant", "content": raw}, *empty, *other]) == raw


@pytest.mark.parametrize("current", [False, True], ids=["old-work", "current-candidate"])
def test_host_admitted_swarm_root_has_one_live_and_rebuilt_provider_disclosure(host, monkeypatch, current):
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **k: None)
    case = incoming_case(host, "main")
    server._route_owner_message(host.bridge, host.ctx, case.incoming)
    assert len(host.pending) == 1
    task = host.pending[0]
    assert task["id"] == case.task_id and task["root_task_id"] == case.task_id
    assert task["metadata"]["force_plan"] is True
    assert not task.get("_is_direct_chat") and not task.get("_ephemeral_turn")
    assert load_task_result(host.root, case.task_id)["status"] == "scheduled"
    assert len(rows(host.root / "logs/chat_annotations.jsonl")) == 1
    assert host.attempts == []

    usage = {
        "_last_llm_error_kind": "provider_outcome_unknown",
        loop_llm_call.TRANSPORT_DEATHS_KEY: {
            "round_id": "managed-first-round", "count": 1,
            "error_kind": "provider_outcome_unknown",
        },
    }
    _, registry, ctx, trace = _forced_test_context(host.root, usage=usage)
    registry._ctx.task_id = ctx.task_id = case.task_id
    registry._ctx.task_contract = dict(task["task_contract"])
    registry._ctx.task_metadata.update(task["metadata"], root_task_id=case.task_id)
    raw = "Useful exact intermediate answer.\n\nRetain these source facts.  \n"
    if current:
        loop._replace_delivery_candidate(registry, ctx, trace, raw, control="replace")
    else:
        ctx.messages.append({"role": "assistant", "content": raw})
    from ouroboros.tools.owner_delivery import _retain_dialogue, publish_pending_owner_dialogue
    registry._ctx.messages = ctx.messages
    _retain_dialogue(registry._ctx, {"type": "send_message", "text": "Progress sent to the owner",
                                   "task_id": case.task_id, "chat_id": case.chat_id}, "deferred")
    publish_pending_owner_dialogue(registry._ctx, ctx.messages)
    assert ctx.messages[-1].get("_host_context_kind") == "owner_dialogue"
    text, usage, trace = loop._handle_provider_unavailable(
        ctx, error_kind="provider_outcome_unknown",
        wait_cause="transport_unavailable", waited_sec=125.0,
    )
    assert usage["terminal_provider_notice"]
    delivered = []
    pipeline.emit_task_results(
        SimpleNamespace(drive_root=host.root, repo_dir=host.root), None, None,
        delivered, task, text, usage, trace, start_time=0.0,
        drive_logs=host.root / "logs",
    )
    sent = next(event for event in delivered if event["type"] == "send_message")
    stored = load_task_result(host.root, case.task_id)
    assert stored["result"] == raw
    assert stored["status"] == "failed"
    rebuilt = build_completed_result_event(host.root, task, case.task_id, stored)
    assert rebuilt["text"] == sent["text"]
    incidents = []
    separate = send_provider_death_notice(
        SimpleNamespace(send_with_budget=lambda *a, **k: incidents.append(a[1])),
        case.chat_id, case.task_id, stored,
    )
    assert separate is current
    for projection in (sent, rebuilt):
        all_text = "\n".join([projection["text"], *incidents])
        assert all_text.count("no terminal provider outcome") == 1
        assert "Swarm reached the task-wide rail" not in all_text
    assert task["origin_message_ref"] == case.ref
    assert task["origin_message_text"] == case.incoming["text"]
    assert len(rows(host.root / "logs/chat_annotations.jsonl")) == 1
    assert host.attempts == []
