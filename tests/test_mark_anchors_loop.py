"""A mark tied to ``delegate_wait`` leaves the host's wake readers alone, through the real loop.

The mind sees the mark beside the wake, while the terminal-wake acknowledgement and the
exposure of owner mail riding the wake read the plain delivered result: the wake is
acknowledged once and the owner's words reach the next model input once.
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from ouroboros import delegate_custody as custody
from ouroboros import loop
from ouroboros.tools.chronicle import _chronicle_write, _memory_mark
from tests._stub_delegate_engines import stub_delegate_engines as stub_delegate_engines  # noqa: F401
from tests.test_configured_session_restore import InputReached
from tests.test_delegated_run_ack_consumers import (  # noqa: F401  (autouse fixtures ride the import)
    OWN_LINE,
    TASK,
    _acks,
    _fresh_messages,
    _loop_kwargs,
    _owned_gateway_uses_each_test_transport,
    _registry,
    _settle,
    _system_text,
    no_provider,
)

pytestmark = pytest.mark.serial


def _mark_delegate_wait(ctx):
    node = json.loads(_chronicle_write(ctx, kind="note", room_id="1", text="a lesson about delegated runs"))
    assert node["ok"], node
    made = json.loads(_memory_mark(ctx, text="DELEGATE_WAIT_LESSON: check output_delivery before trusting it.",
                                   node_id=node["node_id"], room_id="1", scope="global",
                                   anchor={"tool": "delegate_wait"}))
    assert made["ok"], made


def test_a_marked_terminal_wake_is_shown_with_its_mark_and_still_acknowledged(tmp_path, monkeypatch):
    from ouroboros.delegate_supervision import supervision_checkpoint
    from ouroboros.gateways import claudexor as gw

    engine = _settle(tmp_path, task_id=TASK)
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: engine)
    live = _registry(tmp_path)
    _mark_delegate_wait(live._ctx)
    first = _fresh_messages(tmp_path, live)
    assert OWN_LINE in _system_text(first)
    rounds = []

    def model(call, _disposition, **_kwargs):
        rounds.append(len(call.messages))
        if len(rounds) == 1:
            return {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call-wait", "type": "function",
                "function": {"name": "delegate_wait", "arguments": json.dumps({"run_id": "run-1"})}}]}, 0.0
        return {"role": "assistant", "content": "parking for the owner", "tool_calls": []}, 0.0

    monkeypatch.setattr(loop, "_dispatch_round_model", model)
    loop.run_llm_loop(**_loop_kwargs(tmp_path, live, first))
    rows = [row for row in live._ctx.messages if row.get("role") == "tool" and row.get("tool_call_id") == "call-wait"]
    assert len(rows) == 1 and '"status": "terminal"' in rows[0]["content"]
    assert "DELEGATE_WAIT_LESSON" in rows[0]["content"]
    run = supervision_checkpoint(live._ctx)["runs"]["run-1"]
    assert _acks(tmp_path) == [("run-1", TASK, "1")] and run.get("last_acknowledged_wake")
    assert not run.get("pending_wake")
    assert [row for row in custody.custody_rows(tmp_path) if row["type"] == custody.OUTPUT_CONSUMED]
    assert OWN_LINE not in _system_text(_fresh_messages(tmp_path, live))


def test_owner_mail_in_a_marked_wake_reaches_the_next_model_input_once(tmp_path, monkeypatch):
    from ouroboros import delegate_supervision as ds
    from ouroboros.owner_mailbox import write_owner_message
    from ouroboros.tools import delegate
    from tests.test_loop_transport_wait import _loop_kwargs as transport_loop_kwargs
    from tests.test_working_checkpoint import _registry as nanny_registry

    registry = nanny_registry(tmp_path, monkeypatch, "nanny", 2)
    ctx = registry._ctx
    _mark_delegate_wait(ctx)
    text, msg_id, run_id = "Unique owner correction carried by the wake", "correction-mail", "bound-run"

    def wait_once(_ctx, rid, *_args, **_kwargs):
        assert write_owner_message(tmp_path, text, ctx.task_id, msg_id=msg_id)
        return json.dumps({"status": "terminal", "run_id": rid, "state": "succeeded"})

    def wait_entry(ctx, run_id):
        return ds.supervised_wait(ctx, run_id, wait_once=wait_once)

    entry = registry._entries["delegate_wait"]
    monkeypatch.setitem(registry._entries, "delegate_wait", replace(
        entry, handler=delegate._published_entry(wait_entry)))
    rounds, observed = [], {}

    def model_input(call, *_args, **_kwargs):
        rounds.append(call.messages)
        if len(rounds) == 1:
            return {"role": "assistant", "content": "", "tool_calls": [{
                "id": "wait", "type": "function", "function": {
                    "name": "delegate_wait", "arguments": json.dumps({"run_id": run_id})}}]}, 0.0
        contents = [str(row.get("content") or "") for row in call.messages]
        observed.update(copies=sum(item.count(text) for item in contents),
                        mark_shown=any("DELEGATE_WAIT_LESSON" in item for item in contents))
        raise InputReached

    monkeypatch.setattr(loop, "_dispatch_round_model", model_input)
    kwargs = transport_loop_kwargs(tmp_path, registry, [])
    kwargs.update(task_id=ctx.task_id)
    with pytest.raises(InputReached):
        loop.run_llm_loop(**kwargs)
    assert len(rounds) == 2
    assert observed == {"copies": 1, "mark_shown": True}, observed
