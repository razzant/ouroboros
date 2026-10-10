"""The rolling working checkpoint (#1563, consolidated #1543) through its real consumers.

One serializer (``owner_wait.continuation_state``), one atomically replaced file
per attempt, the four loop boundaries, the pre-effect HOLD, the delayed content
ACK, the recovery freeze and its precedence rules, and the real loop restoring a
frozen state for the same and a new execution id.
"""
from __future__ import annotations

import json
import queue
import time
from types import SimpleNamespace

import pytest

from ouroboros import budget_pause, working_checkpoint as wc
from ouroboros.task_results import write_task_result
from tests._budget_pause_exact_helpers import _loop_ctx


def _file(root, task_id="pause-task", attempt=1):
    return wc.checkpoint_path(root, task_id, attempt)


def test_each_boundary_replaces_one_attempt_file_with_the_one_serializer(tmp_path):
    ctx, limit_ctx = _loop_ctx(tmp_path)
    assert wc.save_round(limit_ctx, "pre_effect") is True
    first = json.loads(_file(tmp_path).read_bytes())
    assert first["task_id"] == "pause-task" and first["task_attempt"] == 1
    assert first["working"]["boundary"] == "pre_effect" and first["working"]["seq"] == 1
    assert first["working"]["pending_tool_call_ids"] == ["call_b"], "the unanswered call is named"
    assert first["messages"] == limit_ctx.messages and first["round_idx"] == 4
    assert "execution_status" not in first["usage"], "a rail's terminal projection never travels"

    limit_ctx.messages.append({"role": "tool", "tool_call_id": "call_b", "content": "done b"})
    assert wc.save_round(limit_ctx, "post_batch") is True
    second = json.loads(_file(tmp_path).read_bytes())
    assert second["working"]["seq"] == 2 and second["working"]["pending_tool_call_ids"] == []
    assert wc.save_round(limit_ctx, "ready") is False, "nothing changed since the batch was saved"
    limit_ctx.messages.append({"role": "user", "content": "[owner] also check B"})
    assert wc.save_round(limit_ctx, "ready") is True
    stats = limit_ctx.accumulated_usage["working_checkpoint"]
    assert stats["saves"] == 3 and stats["max_bytes"] > 0 and stats["last_boundary"] == "ready"
    assert sorted(p.name for p in _file(tmp_path).parent.glob("working_checkpoint-a*")) == [
        "working_checkpoint-a1.json"], "one file per attempt, replaced, never one per boundary"

    ctx.task_attempt = 2  # a later attempt is a different single writer
    wc.save_round(limit_ctx, "ready")
    assert _file(tmp_path, attempt=2).is_file() and _file(tmp_path, attempt=1).is_file()
    wc.discard(tmp_path, "pause-task")
    assert not list(_file(tmp_path).parent.glob("working_checkpoint-a*"))


def test_a_failed_pre_effect_write_holds_visibly_and_only_the_task_controls_end_it(tmp_path, monkeypatch):
    _ctx, limit_ctx = _loop_ctx(tmp_path)
    monkeypatch.setattr(budget_pause, "_HOLD_POLL_SEC", 0.01)
    published = []
    monkeypatch.setattr(budget_pause, "_publish_hold", lambda _ctx, row: published.append(row))
    failures = {"left": 3}
    import ouroboros.utils as utils

    real_write = utils.write_bytes_atomic

    def flaky(path, data, **kw):
        if failures["left"]:
            failures["left"] -= 1
            raise OSError(28, "No space left on device")
        return real_write(path, data, **kw)

    monkeypatch.setattr(utils, "write_bytes_atomic", flaky)
    monkeypatch.setattr(budget_pause, "_hold_control_reason", lambda _ctx: "")
    wc.save_before_effects(limit_ctx)  # returns only once the state is durable
    assert json.loads(_file(tmp_path).read_bytes())["working"]["boundary"] == "pre_effect"
    assert published and published[0]["hold_reason"] == wc.HOLD_UNWRITABLE
    assert published[0]["rail"] == "working_checkpoint"
    assert "budget_pause_hold" not in limit_ctx.accumulated_usage

    from ouroboros.model_wait import ModelWaitInterrupted

    failures["left"] = 10**6
    controls = iter(["", "", "cancelled"])
    monkeypatch.setattr(budget_pause, "_hold_control_reason", lambda _ctx: next(controls))
    limit_ctx.messages.append({"role": "assistant", "tool_calls": [{"id": "call_c"}]})
    with pytest.raises(ModelWaitInterrupted) as stopped:
        wc.save_before_effects(limit_ctx)
    assert stopped.value.control_reason == "cancelled", "Stop ends the hold; no tool of the batch ran"


def test_content_mail_is_acknowledged_only_after_the_ready_state_that_holds_it(tmp_path):
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_mailbox import drain_owner_entries, write_owner_message

    ctx, limit_ctx = _loop_ctx(tmp_path)
    write_task_result(tmp_path, "pause-task", "running")
    assert write_owner_message(tmp_path, "please also check B", "pause-task", msg_id="m-b")
    messages = list(limit_ctx.messages)
    seen: set = set()
    _drain_incoming_messages(messages, queue.Queue(), tmp_path, "pause-task", None, seen,
                             owner_ctx=ctx, defer_content_ack=True)
    assert any("please also check B" in str(row.get("content")) for row in messages)
    # A crash here: nothing saved held this mail, and nothing acknowledged it.
    assert [e["msg_id"] for e in drain_owner_entries(tmp_path, "pause-task", set(), 1)] == ["m-b"]
    limit_ctx.messages, limit_ctx.owner_msg_seen = messages, seen
    wc.save_ready(limit_ctx)
    saved = json.loads(_file(tmp_path).read_bytes())
    assert "m-b" in saved["seen"] and any("please also check B" in str(r.get("content")) for r in saved["messages"])
    assert drain_owner_entries(tmp_path, "pause-task", set(), 1) == [], "acknowledged once its state is saved"


def test_the_round_drain_without_deferral_still_acknowledges_at_once(tmp_path):
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_mailbox import drain_owner_entries, write_owner_message

    ctx, _limit = _loop_ctx(tmp_path)
    write_task_result(tmp_path, "pause-task", "running")
    write_owner_message(tmp_path, "now", "pause-task", msg_id="m-now")
    _drain_incoming_messages([], queue.Queue(), tmp_path, "pause-task", None, set(), owner_ctx=ctx)
    assert drain_owner_entries(tmp_path, "pause-task", set(), 1) == []


def test_recovery_freezes_exact_bytes_but_a_live_park_or_an_owed_final_wins(tmp_path):
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.owner_wait import checkpoint_owner_wait, set_owner_wait

    ctx, limit_ctx = _loop_ctx(tmp_path)
    write_task_result(tmp_path, "pause-task", "running")
    wc.save_round(limit_ctx, "pre_effect")
    exact = _file(tmp_path).read_bytes()
    task: dict = {"id": "pause-task"}
    assert wc.attach_recovery(tmp_path, task, source_task_id="pause-task", from_attempt=1, cause="worker_crash")
    handoff = task["_working_recovery"]
    assert handoff["boundary"] == "pre_effect" and handoff["from_attempt"] == 1 and handoff["cause"] == "worker_crash"
    assert read_actor_source_bytes(tmp_path, "pause-task", handoff["source_ref"]) == exact
    # Parent 2026-10-08: the locator is in memory until the retry's own publication;
    # a crash before it finds the SAME file and freezes the same bytes again.
    assert _file(tmp_path).read_bytes() == exact, "the rolling file stays until a consumer captured it"
    again: dict = {"id": "pause-task"}
    assert wc.attach_recovery(tmp_path, again, source_task_id="pause-task", from_attempt=1, cause="worker_crash")
    assert again["_working_recovery"] == handoff and handoff["source_kind"] == wc.SOURCE_WORKING
    consumer = SimpleNamespace(task_id="pause-task", task_attempt=2, drive_root=tmp_path, budget_drive_root=tmp_path)
    loaded = wc.load_recovery(consumer, handoff)
    wc.consume_recovery(consumer, loaded["_working_handoff"])
    assert not _file(tmp_path).exists(), "removed only after the consumer loaded the frozen copy"
    assert not wc.attach_recovery(tmp_path, {}, source_task_id="pause-task", from_attempt=1, cause="x")

    wc.save_round(limit_ctx, "post_batch")  # a newer park's source is the authority
    wait = checkpoint_owner_wait(ctx, limit_ctx.messages, {}, {}, 4, [], set())
    set_owner_wait(tmp_path, "pause-task", {**wait, "state": "waiting"})
    assert wc.prepare_recovery(tmp_path, "pause-task", from_attempt=1, cause="restart") == {}

    write_task_result(tmp_path, "owed-task", "running")
    owed_ctx, owed_limit = _loop_ctx(tmp_path, "owed-task")
    wc.save_round(owed_limit, "candidate")
    from supervisor.terminal_delivery import register_pending_delivery

    register_pending_delivery(tmp_path, {"delivery_id": "final:owed-task:0123456789abcdef", "task_id": "owed-task",
                                         "chat_id": 1, "text": "the answer", "type": "send_message"})
    assert wc.prepare_recovery(tmp_path, "owed-task", from_attempt=1, cause="restart") == {}, \
        "an answer the host already owes is never answered twice"


def test_a_sourceless_pausing_seed_is_completed_from_the_same_attempts_working_state(tmp_path):
    from ouroboros.artifacts import read_actor_source_bytes

    _ctx, limit_ctx = _loop_ctx(tmp_path)
    write_task_result(tmp_path, "pause-task", "running")
    wc.save_round(limit_ctx, "pre_effect")
    seed = {"pause_id": "p1", "state": budget_pause.STATE_PAUSING, "reason": "budget", "rail": "global_exhausted",
            "pause_generation": 1, "scope": "global", "task_attempt": 1, "source_ref": None}
    budget_pause.set_budget_pause(tmp_path, "pause-task", seed)
    row = wc.complete_pause_from_working(tmp_path, "pause-task", seed, 1)
    assert row["source_ref"] and row["exact_continuation"] and row["resume_point"]["unanswered_tool_call_ids"] == ["call_b"]
    state = json.loads(read_actor_source_bytes(tmp_path, "pause-task", row["source_ref"]))
    assert state["pause_id"] == "p1" and state["messages"] == limit_ctx.messages
    assert budget_pause.budget_pause_row(tmp_path, "pause-task")["source_ref"] == row["source_ref"]


# --- the real loop continues a frozen state ----------------------------------------------

def _registry(tmp_path, monkeypatch, task_id, attempt):
    from ouroboros import context, task_pacing
    from ouroboros.tools.registry import ToolRegistry
    from tests.test_context_fit_v664 import _plan
    from types import SimpleNamespace

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    ctx = registry._ctx
    ctx.task_id, ctx.task_attempt = task_id, attempt
    ctx.active_model, ctx.active_effort = "same-model", "high"
    ctx.active_use_local, ctx.active_context_mode = False, "max"
    ctx._cost_ceiling = task_pacing.CostCeiling(state="disabled")
    ctx.owner_wait_callback = lambda *_: None
    ctx.context_fit_plan = _plan(window=500_000, known=True)
    monkeypatch.setattr(context, "_context_fit_route", lambda task, **_: (
        {"model": task["model"], "provider": "openai", "use_local": False},
        SimpleNamespace(status="confirmed", stale=False, window_tokens=500_000, route_fp="route-a"),
    ))
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "max")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    return registry


def _saved_working_state(tmp_path, monkeypatch, task_id):
    """The dead attempt 1: a batch accepted, one call answered, one not."""
    from types import SimpleNamespace

    from tests.test_context_fit_v664 import _plan

    write_task_result(tmp_path, task_id, "running")
    registry = _registry(tmp_path, monkeypatch, task_id, 1)
    messages = _plan().messages_for("max") + [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "done", "type": "function", "function": {"name": "save", "arguments": "{}"}},
            {"id": "inflight", "type": "function", "function": {"name": "save", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "done", "content": "Saved object 42"},
    ]
    limit = SimpleNamespace(tools=registry, messages=messages, llm_trace={"tool_calls": []},
                            accumulated_usage={"cost": 3.0}, round_idx=7, tool_schemas=[], owner_msg_seen=set(),
                            budget_tail="tool")
    wc.save_round(limit, "pre_effect")
    task: dict = {"id": task_id}
    assert wc.attach_recovery(tmp_path, task, source_task_id=task_id, from_attempt=1, cause="worker_crash")
    return task["_working_recovery"]


@pytest.mark.parametrize("new_id", [False, True])
def test_the_real_loop_continues_the_frozen_state_and_never_reruns_the_unknown_call(tmp_path, monkeypatch, new_id):
    from ouroboros import loop
    from tests.test_loop_transport_wait import _loop_kwargs

    monkeypatch.setattr("ouroboros.llm.LLMClient.chat", lambda *a, **k: pytest.fail("no provider call"))
    handoff = _saved_working_state(tmp_path, monkeypatch, "t-wait")
    task_id = "t-retry" if new_id else "t-wait"
    if new_id:
        write_task_result(tmp_path, task_id, "running")
    registry = _registry(tmp_path, monkeypatch, task_id, 2)
    registry._ctx.working_recovery = handoff
    seen_messages, ran = [], []
    monkeypatch.setattr(registry, "execute_result", lambda *a, **k: ran.append(a) or pytest.fail("no tool"),
                        raising=False)

    def dispatch(call, disposition, *, candidate_predicate=None, **kwargs):
        seen_messages.extend(call.messages)
        return {"role": "assistant", "content": "Continued from the saved state", "tool_calls": []}, 0.0

    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    kwargs = _loop_kwargs(tmp_path, registry, [])
    kwargs["task_id"] = task_id
    result, usage, _trace = loop.run_llm_loop(**kwargs)
    assert result == "Continued from the saved state" and not ran
    contents = [str(row.get("content") or "") for row in seen_messages]
    assert "Saved object 42" in contents, "completed results are restored"
    unknown = [row for row in seen_messages if row.get("tool_call_id") == "inflight"]
    assert unknown and "UNKNOWN" in unknown[0]["content"] and "NOT re-executed" in unknown[0]["content"]
    notice = next(text for text in contents if "continued from its working checkpoint" in text)
    assert "pre_effect" in notice and "Delegated runs of this task" in notice
    assert ("new execution of task t-wait" in notice) is new_id
    assert registry._ctx.working_recovery is None
    assert usage.get("cost", 0.0) == (0.0 if new_id else 3.0), "a new id keeps its own accounting"
    assert not wc.checkpoint_path(tmp_path, "t-wait", 1).exists(), "the consumer retired the captured file"


def test_an_unusable_frozen_source_starts_over_disclosed_never_silently(tmp_path, monkeypatch):
    from ouroboros import loop
    from tests.test_loop_transport_wait import _loop_kwargs

    monkeypatch.setattr("ouroboros.llm.LLMClient.chat", lambda *a, **k: pytest.fail("no provider call"))
    handoff = _saved_working_state(tmp_path, monkeypatch, "t-wait")
    registry = _registry(tmp_path, monkeypatch, "t-wait", 2)
    registry._ctx.working_recovery = {**handoff, "from_attempt": 9}  # identity mismatch
    seen = []

    def dispatch(call, disposition, *, candidate_predicate=None, **kwargs):
        seen.extend(call.messages)
        return {"role": "assistant", "content": "fresh", "tool_calls": []}, 0.0

    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    loop.run_llm_loop(**_loop_kwargs(tmp_path, registry, []))
    assert any("could not be restored" in str(row.get("content")) for row in seen)


def test_representative_checkpoint_io_is_measured_not_promised(tmp_path):
    """Disclosure only (owner F7): bytes and encode+replace time at ~12 MB, no threshold."""
    _ctx, limit_ctx = _loop_ctx(tmp_path)
    blob = "x" * 4000
    limit_ctx.messages = [{"role": "tool", "tool_call_id": f"c{i}", "content": blob} for i in range(3000)]
    started = time.perf_counter()
    wc.save_round(limit_ctx, "post_batch")
    elapsed = time.perf_counter() - started
    stats = limit_ctx.accumulated_usage["working_checkpoint"]
    assert stats["last_bytes"] > 11_000_000
    print(f"WORKING_CHECKPOINT_IO bytes={stats['last_bytes']} encode_ms={stats['encode_ms']} "
          f"write_ms={stats['write_ms']} total_s={elapsed:.3f}")


def test_a_held_question_wait_continues_through_its_own_reader_after_quit_and_resume(tmp_path, monkeypatch):
    """Parent 2026-10-08: a REAL question park → application stop → actual boot restore
    → Resume → the real loop. The wait's own continuation source is named as such
    (``source_kind=owner_wait``, bound to its ``wait_id``), never posing as a rolling
    state; the consumed wait reads ``resumed`` (an answer is still accepted)."""
    import json as _json

    from ouroboros import loop
    from ouroboros.owner_wait import checkpoint_owner_wait, set_owner_wait
    from supervisor.events_budget import HOLD_SAVED_WORK, budget_hold_fact
    from tests._budget_pause_exact_helpers import _install_queue
    from tests.test_context_fit_v664 import _plan
    from tests.test_loop_transport_wait import _loop_kwargs
    from tests.test_restart_retention import _pool_events

    monkeypatch.setattr("ouroboros.llm.LLMClient.chat", lambda *a, **k: pytest.fail("no provider call"))
    queue_mod, state_mod, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    write_task_result(tmp_path, "asker", "running", chat_id=0, root_task_id="asker")
    asker = _registry(tmp_path, monkeypatch, "asker", 1)
    asker._ctx._owner_wait_requested = "quiz-1"
    messages = _plan().messages_for("max") + [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "ask", "type": "function", "function": {"name": "ask_owner", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "ask", "content": "Question posted; waiting for the owner"}]
    wait = checkpoint_owner_wait(asker._ctx, messages, {"tool_calls": []}, {"cost": 1.0}, 5, [], set())
    set_owner_wait(tmp_path, "asker", {**wait, "state": "waiting"})
    workers.RUNNING["asker"] = {"task": {"id": "asker", "type": "task", "chat_id": 0, "root_task_id": "asker",
                                         "_attempt": 1}, "worker_id": 0, "attempt": 1, "owner_wait": wait}
    workers.kill_workers(force=True, terminal_status="cancelled", result_reason="Server shutdown.",
                         stop_source="server_shutdown", retain_saved_work=True)
    workers.PENDING[:] = []
    workers.RUNNING.clear()
    queue_mod.restore_pending_from_snapshot()

    [row] = [task for task in workers.PENDING if task["id"] == "asker"]
    assert budget_hold_fact(row)["reason"] == HOLD_SAVED_WORK
    handoff = row["_working_recovery"]
    assert handoff["source_kind"] == wc.SOURCE_OWNER_WAIT and handoff["wait_id"] == wait["wait_id"]
    monkeypatch.setattr(state_mod, "budget_remaining", lambda *_a, **_k: 5.0)
    assert queue_mod.resume_budget_paused_task("asker")["ok"]

    registry = _registry(tmp_path, monkeypatch, "asker", row["_attempt"])
    registry._ctx.working_recovery = handoff
    seen = []

    def dispatch(call, disposition, *, candidate_predicate=None, **kwargs):
        seen.extend(call.messages)
        return {"role": "assistant", "content": "Continued after the stop", "tool_calls": []}, 0.0

    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    kwargs = _loop_kwargs(tmp_path, registry, [])
    kwargs["task_id"] = "asker"
    result, _usage, _trace = loop.run_llm_loop(**kwargs)
    assert result == "Continued after the stop"
    contents = [str(item.get("content") or "") for item in seen]
    assert "Question posted; waiting for the owner" in contents
    notice = next(text for text in contents if "continued from its working checkpoint" in text)
    assert "owner_wait" in notice and "that question stays open" in notice
    stored = _json.loads((tmp_path / "task_results" / "asker.json").read_text())["owner_wait"]
    assert stored["wait_id"] == wait["wait_id"] and stored["state"] == "resumed"


@pytest.mark.parametrize("new_id", [False, True])
def test_s3_integration_pin_working_recovery_restores_its_saved_cost_ceiling(tmp_path, monkeypatch, new_id):
    """INTEGRATION PIN (S3 money seam, parent 2026-10-08). A working recovery must restore
    its SAVED cost ceiling through the common ``task_pacing.restore_cost_ceiling(ctx, saved)``
    (the restore the owner-wait and budget-pause continuations use), assign
    ``ctx._cost_ceiling`` and return it from ``loop._resume_continuation`` — an explicit
    saved point (7) stays 7 although the wallet changed. Skipped while this baseline
    lacks S3's symbol; no S3 code is copied here."""
    from types import SimpleNamespace as NS

    from ouroboros import loop, task_pacing

    if not hasattr(task_pacing, "restore_cost_ceiling"):
        pytest.skip("S3 task_pacing.restore_cost_ceiling is not in this baseline: wire it at integration")
    from tests.test_context_fit_v664 import _plan

    write_task_result(tmp_path, "t-wait", "running")
    dead = _registry(tmp_path, monkeypatch, "t-wait", 1)
    dead._ctx._cost_ceiling = task_pacing.CostCeiling(state="active", ceiling_usd=7.0, basis="explicit_saved")
    limit = NS(tools=dead, messages=_plan().messages_for("max"), llm_trace={"tool_calls": []},
               accumulated_usage={"cost": 1.0}, round_idx=3, tool_schemas=[], owner_msg_seen=set(),
               budget_tail="tool")
    wc.save_round(limit, "post_batch")
    task: dict = {"id": "t-wait"}
    assert wc.attach_recovery(tmp_path, task, source_task_id="t-wait", from_attempt=1, cause="worker_crash")
    task_id = "t-retry" if new_id else "t-wait"
    if new_id:
        write_task_result(tmp_path, task_id, "running")
    registry = _registry(tmp_path, monkeypatch, task_id, 2)
    registry._ctx.task_contract = {"budget_profile": {"cost_hard_stop_pct": 50}}
    registry._ctx.working_recovery = task["_working_recovery"]
    saved = wc.load_recovery(registry._ctx)
    fresh = task_pacing.CostCeiling(state="active", ceiling_usd=3.0, basis="current_wallet")
    route = ("same-model", "high", False, "max", 0, registry._ctx.context_fit_plan)
    resumed = loop._resume_continuation(registry, ({}, {}, saved), [], {}, {}, set(), route, fresh, 1.0)
    assert resumed[6].ceiling_usd == 7.0 and registry._ctx._cost_ceiling is resumed[6]
