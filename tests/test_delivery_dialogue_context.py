"""Explicit outward words survive as dialogue, separately from raw tool batches."""
from copy import deepcopy
from dataclasses import replace
import json
import queue
from types import SimpleNamespace

import pytest

from ouroboros import context_compaction as cc
from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY
from ouroboros.owner_wait import continuation_state, restore_continuation_state, store_continuation_source
from ouroboros.tools.owner_delivery import (
    deliver_owner_event, pending_owner_dialogue, publish_pending_owner_dialogue,
)
from ouroboros.working_checkpoint import close_unanswered_calls
from tests.test_context_reclaim_materializer import _request, _SPEC
from tests.test_main_authored_context import call, main_loop as _main_loop

main_loop = _main_loop


def _dialogue(messages):
    return [m for m in messages if m.get(HOST_CONTEXT_KIND_KEY) == "owner_dialogue"]


def _facts(row):
    return json.loads(row["content"].split("\n", 2)[1])


def _batch():
    calls = [call("send_user_message", {"text": "I will keep the old route.\nChoose A or B?"}, "say"),
             call("read_file", {"path": "evidence.txt"}, "large")]
    return {"role": "assistant", "content": "", "tool_calls": [c["tool_calls"][0] for c in calls]}


def test_real_send_producer_is_a_separate_exact_row_after_the_completed_batch(main_loop):
    f = main_loop
    f.ctx.current_chat_id = 1
    f.run([_batch(), {"content": "done"}])
    before, after = f.inputs[0]["messages"], f.inputs[1]["messages"]
    rows = _dialogue(after)
    assert len(rows) == 1 and not _dialogue(before)
    row = rows[0]
    assert [m.get("tool_call_id") for m in after[after.index(row) - 2:after.index(row)]] == ["say", "large"]
    facts = _facts(row)
    exact = "I will keep the old route.\nChoose A or B?"
    assert row["content"].endswith("[Exact addressed text]\n" + exact)
    assert read_actor_source_bytes(f.ctx.drive_root, f.ctx.task_id, facts["source_ref"]).decode("utf-8") == exact
    assert facts["author"]["kind"] == "ouroboros"
    assert facts["transport_mode"] == "live"
    assert facts["delivery_confirmation"] == facts["read_confirmation"] == "unknown"
    assert not pending_owner_dialogue(f.ctx)
    assert all(exact not in str(d) for d in getattr(f.ctx, "_owner_directives", []))
    # Appending explicit dialogue does not rewrite any earlier byte of history.
    assert after[:len(before)] == before


@pytest.mark.parametrize("recovery", ["authored", "helper", "emergency"])
def test_outward_speech_survives_while_the_unrelated_large_tool_body_folds(main_loop, monkeypatch, recovery):
    f = main_loop
    f.ctx.current_chat_id = 1
    (f.ctx.repo_dir / "evidence.txt").write_text("large unrelated result\n" * 3000, encoding="utf-8")
    f.run([_batch(), {"content": "done"}])
    messages = deepcopy(f.ctx.messages)
    rows = _dialogue(messages)
    assert len(rows) == 1
    # This working draft was never sent to the human; authored selection may fold it.
    internal = {"role": "assistant", "content": "Internal working draft. " * 1000}
    messages.append(internal)
    request = _request(messages, 1)
    if recovery == "emergency":
        from ouroboros.context_source_view import emergency_address_view
        rebuilt, receipt = emergency_address_view(messages, request, rung="unseen_bodies",
                                                  drive_root=f.ctx.drive_root, task_id=f.ctx.task_id)
    else:
        if recovery == "helper":
            monkeypatch.setattr(cc, "_summarizer_spec", lambda: dict(_SPEC))
            monkeypatch.setattr(cc, "_call_summarizer", lambda parts, **_kw: {
                part.source_id: "Attributed helper account of the tool result." for part in parts})
        else:
            request = replace(request, working_note="The large source is recorded; the question remains explicit.",
                              expected_view_revision=request.transcript_sha256, keep_unit_ids=())
        rebuilt, receipt, _ = cc.compact_tool_history_llm(messages, keep_recent=0, request=request,
            observed_messages=messages, tool_schemas=[], fit_candidate=lambda *_: {"accepted": True},
            drive_root=f.ctx.drive_root, task_id=f.ctx.task_id)
    assert receipt.status == "applied"
    assert all(row in rebuilt for row in rows)
    assert not any(m.get("tool_call_id") == "large" for m in rebuilt)
    if recovery == "authored":
        assert internal not in rebuilt
    checkpoint = json.loads(read_actor_source_bytes(f.ctx.drive_root, f.ctx.task_id, receipt.checkpoint_ref))
    assert checkpoint["messages"] == messages


def test_real_escalate_keeps_question_options_details_and_assumption(main_loop):
    from ouroboros.task_results import write_task_result
    f = main_loop
    f.ctx.current_chat_id = 1
    write_task_result(f.ctx.drive_root, "authored-main", "running")
    args = {"question": "Which exact destination?", "options": [
        {"label": "A: keep", "detail": "Preserve the existing route.", "recommended": True},
        {"label": "B: change", "detail": "Move to the new route."}],
        "stake": "The public behavior changes.", "assumption": "Keep the old route while waiting."}
    f.run([call("escalate", args, "question"), {"content": "done"}])
    rows = _dialogue(f.ctx.messages)
    assert len(rows) == 1
    facts = _facts(rows[0])
    assert facts["event_type"] == "send_quiz"
    spoken = json.loads(rows[0]["content"].split("[Exact addressed text]\n", 1)[1])
    assert spoken == args
    assert json.loads(read_actor_source_bytes(f.ctx.drive_root, f.ctx.task_id, facts["source_ref"])) == args


def test_post_batch_cold_checkpoint_keeps_words_without_resending(main_loop):
    f = main_loop
    f.ctx.current_chat_id = 1
    f.run([_batch(), {"content": "done"}])
    state = continuation_state(f.ctx, f.ctx.messages, {}, {}, 2, [], set())
    ref = store_continuation_source(f.ctx, state, "dialogue-cold")
    saved = json.loads(read_actor_source_bytes(f.ctx.drive_root, f.ctx.task_id, ref))
    restored, events_before = [], f.events.qsize()
    restore_continuation_state(f.registry, saved, restored, {}, {}, set())
    publish_pending_owner_dialogue(f.ctx, restored)
    assert _dialogue(restored) == _dialogue(state["messages"])
    assert f.events.qsize() == events_before


@pytest.mark.parametrize("live", [True, False])
def test_budget_abort_after_real_outward_send_preserves_pending_speech_on_restore(main_loop, monkeypatch, live):
    from ouroboros.loop_tool_execution import handle_tool_calls, StatefulToolExecutor
    from ouroboros.usage_accounting import BudgetExceeded

    f = main_loop
    f.ctx.task_id, f.ctx.current_chat_id = "dialogue-abort", 1
    f.ctx.messages = f.messages
    f.ctx.event_queue = f.events if live else None
    # The first real handler sends; the second handler aborts the batch before
    # process_tool_results can adopt its remembered speech.
    execute = f.registry.execute_result
    def execute_or_refuse(name, args):
        if name == "read_file":
            assert pending_owner_dialogue(f.ctx), "The outward send must precede the refusal"
            raise BudgetExceeded("test: budget admission refused")
        return execute(name, args)

    monkeypatch.setattr(f.registry, "execute_result", execute_or_refuse)
    batch = _batch()
    f.messages.append(batch)
    executor = StatefulToolExecutor()
    try:
        with pytest.raises(BudgetExceeded):
            handle_tool_calls(batch["tool_calls"], f.registry, f.ctx.drive_root / "logs", f.ctx.task_id,
                              executor, f.messages, {"tool_calls": []}, lambda *_a, **_kw: None)
    finally:
        executor.shutdown()
    pending = pending_owner_dialogue(f.ctx)
    assert len(pending) == 1 and not _dialogue(f.messages)
    state = continuation_state(f.ctx, f.messages, {}, {}, 1, [], set())
    ref = store_continuation_source(f.ctx, state, "after-budget-abort")
    saved = json.loads(read_actor_source_bytes(f.ctx.drive_root, f.ctx.task_id, ref))
    f.ctx._pending_owner_dialogue = []
    restored, count = [], f.events.qsize()
    restore_continuation_state(f.registry, saved, restored, {}, {}, set())
    closed = close_unanswered_calls(restored, "budget pause")
    from ouroboros.loop_round_limits import _run_round_compaction
    from tests.test_context_view_tool import _context
    restored, _ = _run_round_compaction(restored, _context(f.ctx.drive_root, f.ctx, [], monkeypatch))
    assert closed == ["say", "large"]
    assert _dialogue(restored) == pending and not pending_owner_dialogue(f.ctx)
    assert f.events.qsize() == count  # restoring words never dispatches a send
    position = restored.index(pending[0])
    assert [r.get("tool_call_id") for r in restored[position-2:position]] == ["say", "large"]
    facts = _facts(restored[position])
    assert facts["transport_mode"] == ("live" if live else "deferred")
    assert facts["delivery_confirmation"] == facts["read_confirmation"] == "unknown"


def test_progress_and_child_or_host_authorship_are_not_protected_dialogue(tmp_path):
    ctx = SimpleNamespace(messages=[], pending_events=[], drive_root=tmp_path, task_id="dialogue-origin",
                          task_metadata={}, event_queue=queue.Queue())
    for event in ({"is_progress": True}, {"role": "system"}, {"parent_task_id": "parent"}):
        deliver_owner_event(ctx, {"type": "send_message", "chat_id": 1, "text": "Not root-authored speech", **event})
    assert not pending_owner_dialogue(ctx)
    deliver_owner_event(ctx, {"type": "send_message", "chat_id": 1, "text": "Explicitly addressed to you."})
    assert len(pending_owner_dialogue(ctx)) == 1


def test_only_spoken_attachment_fields_are_retained(tmp_path):
    ctx = SimpleNamespace(messages=[], pending_events=[], drive_root=tmp_path, task_id="dialogue-caption",
                          task_metadata={}, event_queue=None)
    deliver_owner_event(ctx, {"type": "send_photo", "chat_id": 1,
                              "caption": "These are the measured pixels.", "image_base64": "binary" * 10000})
    [row] = pending_owner_dialogue(ctx)
    assert "These are the measured pixels." in row["content"]
    assert "binary" not in row["content"]
    assert _facts(row)["transport_mode"] == "deferred"


def test_failed_batch_measurement_keeps_pending_words_without_reemitting(tmp_path):
    from ouroboros.loop_tool_execution import process_tool_results
    ctx = SimpleNamespace(messages=[], pending_events=[], drive_root=tmp_path, task_id="dialogue-fit",
                          task_metadata={}, event_queue=queue.Queue())
    deliver_owner_event(ctx, {"type": "send_message", "chat_id": 1, "text": "The exact question stays here."})
    pending = pending_owner_dialogue(ctx)
    result = {"tool_call_id": "say", "fn_name": "send_user_message", "result": "OK", "is_error": False,
              "tool_args": {}, "args_for_log": {}, "result_meta": {}}
    def failed_measurement(*_a):
        raise RuntimeError("test: measurement interrupted")
    with pytest.raises(RuntimeError, match="measurement interrupted"):
        process_tool_results([result], ctx.messages, {"tool_calls": []}, lambda *_: None,
                             SimpleNamespace(_ctx=ctx), fit_candidate=failed_measurement)
    assert not ctx.messages and pending_owner_dialogue(ctx) == pending
    process_tool_results([result], ctx.messages, {"tool_calls": []}, lambda *_: None, SimpleNamespace(_ctx=ctx))
    assert _dialogue(ctx.messages) == pending and not pending_owner_dialogue(ctx)
    assert ctx.event_queue.qsize() == 1
