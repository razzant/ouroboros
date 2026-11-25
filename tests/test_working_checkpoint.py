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
from tests._store_shared import read_result


def _file(root, task_id="pause-task", attempt=1):
    return wc.checkpoint_path(root, task_id, attempt)


def test_each_boundary_replaces_one_attempt_file_with_the_one_serializer(tmp_path):
    ctx, limit_ctx = _loop_ctx(tmp_path)
    assert wc.save_round(limit_ctx, "pre_effect") is True
    first = wc.read_working(tmp_path, "pause-task", 1)[1]  # the file is a skeleton over CAS parts: read through the reader
    assert first["task_id"] == "pause-task" and first["task_attempt"] == 1
    assert first["working"]["boundary"] == "pre_effect" and first["working"]["seq"] == 1
    assert first["working"]["pending_tool_call_ids"] == ["call_b"], "the unanswered call is named"
    assert first["messages"] == limit_ctx.messages and first["round_idx"] == 4
    assert "execution_status" not in first["usage"], "a rail's terminal projection never travels"

    limit_ctx.messages.append({"role": "tool", "tool_call_id": "call_b", "content": "done b"})
    assert wc.save_round(limit_ctx, "post_batch") is True
    second = wc.read_working(tmp_path, "pause-task", 1)[1]
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


def test_checkpoint_for_fallback_exhaustion_keeps_one_selected_frame(tmp_path):
    ctx, limit = _loop_ctx(tmp_path)
    ctx.route_wait_on_primary = True
    ctx.route_choice = {'kind': 'following', 'source': 'main'}
    limit.tool_schemas = [{'function': {'name': 'primary_only'}}]
    rows = [{'role': 'system', 'content': 'nano'}, {'role': 'user', 'content': 'Reply was length limited.'}]
    schemas = [{'function': {'name': 'nano_only'}}]
    call = SimpleNamespace(messages=rows, tool_schemas=schemas, model_role='fallback:0', active_model='alternate',
        active_use_local=False, active_context_mode='nano', fallback_is_primary=False,
        fallback_entry_failure={'failure_code': 'quota_exhausted'})
    with wc.model_call_scope(limit):
        wc.save_model_call(model_call=call)
    saved = wc.read_working(tmp_path, 'pause-task', 1)[1]
    route = saved['route']
    assert saved['messages'] == rows and saved['tool_schemas'] == schemas
    assert saved['context_model_role'] == route['_lead_role'] == 'fallback:0'
    assert route['active_context_mode'] == 'nano' and route['active_model'] == 'alternate'
    assert not route['route_wait_on_primary'] and route['route_choice']['source'] == 'fallback'
    assert ctx.route_wait_on_primary and limit.tool_schemas == [{'function': {'name': 'primary_only'}}]


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
    assert wc.read_working(tmp_path, "pause-task", 1)[1]["working"]["boundary"] == "pre_effect"
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


def test_a_disappeared_saved_root_holds_the_cut_batch_until_write_or_control(tmp_path, monkeypatch):
    from ouroboros.loop_output_exhaustion import intermediate_reply
    from ouroboros.model_wait import ModelWaitInterrupted

    finish = {"finish_reason_present": True, "finish_reason": "length"}
    reply = {"role": "assistant", "content": "Saving a note.", "tool_calls": [
        {"id": "fixture-call", "function": {"name": "note_fixture", "arguments": "{}"}}]}
    real_write = wc._write
    for outcome in ("restored", "cancelled", "deadline"):
        root = tmp_path / outcome
        root.mkdir()
        moved = tmp_path / f"{outcome}-away"
        ctx, limit = _loop_ctx(root)
        ctx._round_response_facts = finish
        limit.messages = [{"role": "user", "content": "Save a note."}]
        limit.llm_trace = {"reasoning_notes": []}
        assert wc.save_round(limit, "ready")
        controls, updates = [], []

        def fail_then_disappear(ctx, limit, boundary):
            if not controls:
                root.rename(moved)
                raise OSError("fixture write failed as the root disappeared")
            return real_write(ctx, limit, boundary)

        def control(_ctx):
            assert not updates and not root.exists()
            controls.append(True)
            if len(controls) == 2:
                if outcome != "restored":
                    return outcome
                moved.rename(root)
            return ""

        def progress(_text, **_meta):
            updates.append(wc.read_working(root, ctx.task_id, 1)[1])

        with monkeypatch.context() as held:
            held.setattr(wc, "_write", fail_then_disappear)
            held.setattr(budget_pause, "_HOLD_POLL_SEC", 0)
            held.setattr(budget_pause, "_publish_hold", lambda *_: None)
            held.setattr(budget_pause, "_hold_control_reason", control)
            if outcome == "restored":
                assert intermediate_reply(limit, reply, progress)
                [state] = updates
                assert state["working"]["boundary"] == "post_batch"
                assert state["working"]["pending_tool_call_ids"] == []
                assert state["messages"][-1]["content"].startswith("[HOST NOTICE] NOT STARTED:")
            else:
                with pytest.raises(ModelWaitInterrupted) as stopped:
                    intermediate_reply(limit, reply, progress)
                assert stopped.value.control_reason == outcome and not updates
                assert wc.read_working(moved, ctx.task_id, 1)[1]["working"]["boundary"] == "ready"
        assert len(controls) == 2

    # A named task that never had a durable directory does not inherit this hold.
    root = tmp_path / "never-provisioned"
    ctx, limit = _loop_ctx(root)
    ctx._round_response_facts = finish
    limit.llm_trace = {"reasoning_notes": []}
    emitted = []
    assert intermediate_reply(limit, reply, lambda text, **_meta: emitted.append(text))
    assert emitted == ["Saving a note."] and not root.exists()


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
    saved = wc.read_working(tmp_path, "pause-task", 1)[1]
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
    exact, rolling = wc.read_working(tmp_path, "pause-task", 1)[0], _file(tmp_path).read_bytes()
    task: dict = {"id": "pause-task"}
    assert wc.attach_recovery(tmp_path, task, source_task_id="pause-task", from_attempt=1, cause="worker_crash")
    handoff = task["_working_recovery"]
    assert handoff["boundary"] == "pre_effect" and handoff["from_attempt"] == 1 and handoff["cause"] == "worker_crash"
    assert read_actor_source_bytes(tmp_path, "pause-task", handoff["source_ref"]) == exact
    # Parent 2026-10-08: the locator is in memory until the retry's own publication;
    # a crash before it finds the SAME file and freezes the same bytes again.
    assert _file(tmp_path).read_bytes() == rolling, "the rolling file stays until a consumer captured it"
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


def test_a_retained_wait_parks_only_while_the_pause_that_retained_it_is_live(tmp_path):
    """A stop moves a warm wait into an exact pause and marks the wait ``retained``. Its Resume
    continues the SAME attempt, so once that pause is consumed the attempt's newer working
    state is the source: restart and retry recovery freeze it instead of refusing (19.10.2026,
    two updates in a row cancelled work the restart had promised to return)."""
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.owner_wait import checkpoint_owner_wait, set_owner_wait

    ctx, limit_ctx = _loop_ctx(tmp_path)
    write_task_result(tmp_path, "pause-task", "running")
    wait = checkpoint_owner_wait(ctx, limit_ctx.messages, {}, {}, 4, [], set())
    set_owner_wait(tmp_path, "pause-task", {**wait, "state": "retained"})
    pause = {"pause_id": "owner-pause-1", "pause_generation": 1, "task_attempt": 1, "reason": "owner",
             "rail": "owner_pause", "scope": "root", "source_ref": wait["source_ref"],
             "retained_owner_wait_id": wait["wait_id"], "state": budget_pause.STATE_PAUSED}
    budget_pause.set_budget_pause(tmp_path, "pause-task", pause)
    wc.save_round(limit_ctx, "post_batch")
    assert wc.live_park(tmp_path, "pause-task", 1) == "owner_wait", "the live pause's own source still wins"
    assert wc.prepare_recovery(tmp_path, "pause-task", from_attempt=1, cause="restart") == {}

    budget_pause.set_budget_pause(tmp_path, "pause-task", {**pause, "state": budget_pause.STATE_RESUMED})
    assert wc.live_park(tmp_path, "pause-task", 1) == ""
    limit_ctx.messages.append({"role": "user", "content": "[owner] work after the Resume"})
    wc.save_round(limit_ctx, "ready")
    handoff = wc.prepare_recovery(tmp_path, "pause-task", from_attempt=1, cause="restart")
    assert handoff["source_kind"] == wc.SOURCE_WORKING
    frozen = json.loads(read_actor_source_bytes(tmp_path, "pause-task", handoff["source_ref"]))
    assert frozen["messages"][-1]["content"] == "[owner] work after the Resume"

    # A newer live pause that never retained this wait is the park; the old wait is history.
    budget_pause.set_budget_pause(tmp_path, "pause-task", {
        **pause, "pause_id": "budget-2", "pause_generation": 2, "reason": "budget", "rail": "global_exhausted",
        "retained_owner_wait_id": "", "state": budget_pause.STATE_PAUSED})
    assert wc.live_park(tmp_path, "pause-task", 1) == "budget_pause"


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


def _mail_sources(call):
    """The send seam's own reading of this request's task-mail facts: each fact's source hash, beside the text of
    the one prepared row it located ("" where it located none)."""
    from ouroboros.context_assembly_trace import bind_contribution_input, verified_contributions
    from ouroboros.llm_messages import project_wire_messages
    from ouroboros.loop_messages import _extract_plain_text_from_content
    from ouroboros.physical_attempt_context import PhysicalAttemptContext

    fit = PhysicalAttemptContext("owner_max", "max", "cold_estimate", "", "", None, None, False, False)
    wire = {"messages": project_wire_messages(call.messages)}
    bound = bind_contribution_input(fit, wire, call.messages, call.tools._ctx)
    rows = [_extract_plain_text_from_content(row.get("content")) for row in wire["messages"]]
    return [(fact["source_id_sha256"], rows[int(fact["path"][11:-1])] if "path" in fact else "")
            for fact in verified_contributions(wire, bound.contribution_input)["entries"] if fact["family"] == "task_mail"]


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
    noted = []  # a continued task's view was captured at its start: no second Low-start line
    monkeypatch.setattr(loop, "_note_low_start", lambda *a, **kw: noted.append(a))
    kwargs = _loop_kwargs(tmp_path, registry, [])
    kwargs["task_id"] = task_id
    result, usage, _trace = loop.run_llm_loop(**kwargs)
    assert result == "Continued from the saved state" and not ran and noted == []
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


@pytest.mark.parametrize("start", ["first", "restored", "unfrozen", "unusable"])
def test_same_task_retry_rebuilds_acknowledged_ordinary_mail_without_redelivery(tmp_path, monkeypatch, start):
    from ouroboros import loop
    from ouroboros.context_assembly_trace import digest
    from ouroboros.owner_mailbox import _ack_path, PROVENANCE_INDEPENDENT_TASK, write_task_message
    from tests.test_loop_transport_wait import _loop_kwargs

    task_id = "t-wait"  # the existing loop fixture supplies this same admitted id
    write_task_result(tmp_path, task_id, "running")
    assert write_task_message(tmp_path, "exact ordinary note", task_id,
                              source_task_id="parent-task", msg_id="ordinary")
    assert write_task_message(tmp_path, "exact relayed note", task_id, source_task_id="parent-task",
                              provenance="peer_via_ancestor", relayed_from_task_id="peer-task", msg_id="relayed")
    assert write_task_message(tmp_path, "exact independent note", task_id, source_task_id="other-root",
                              provenance=PROVENANCE_INDEPENDENT_TASK, msg_id="independent")
    task = {"id": task_id}
    ack_path = _ack_path(tmp_path, task_id)
    if start != "first":
        dead = _registry(tmp_path, monkeypatch, task_id, 1)
        messages, ids = _loop_kwargs(tmp_path, dead, [])["messages"], set()
        loop._drain_incoming_messages(messages, queue.Queue(), tmp_path, task_id, None, ids,
                                      owner_ctx=dead._ctx, defer_content_ack=True)
        wc.save_ready(SimpleNamespace(tools=dead, messages=messages, llm_trace={"tool_calls": []},
                                     accumulated_usage={}, round_idx=1, tool_schemas=[], owner_msg_seen=ids,
                                     budget_tail="no_tool"))
        assert {json.loads(line)["msg_id"] for line in ack_path.read_bytes().splitlines()} == {
            "ordinary", "relayed", "independent"}
        if start == "unfrozen":
            wc.checkpoint_path(tmp_path, task_id, 1).unlink()
        assert wc.attach_recovery(tmp_path, task, source_task_id=task_id, from_attempt=1,
                                  cause="worker_crash") is (start != "unfrozen")
        if start == "unusable":
            task["_working_recovery"]["from_attempt"] = 9  # identity mismatch
    before = ack_path.read_bytes() if ack_path.exists() else b""
    registry = _registry(tmp_path, monkeypatch, task_id, 1 if start == "first" else 2)
    registry._ctx.working_recovery = task.get("_working_recovery")
    seen, events, mail_sources = [], queue.Queue(), []

    def dispatch(call, disposition, *, candidate_predicate=None, **kwargs):
        seen.append("\n".join(loop._extract_plain_text_from_content(row.get("content"))
                              for row in call.messages if row.get("role") == "user"))
        mail_sources.append(_mail_sources(call))
        call.tools._ctx._round_response_facts = {
            "finish_reason_present": True, "finish_reason": "commentary" if len(seen) == 1 else "stop",
        }
        return {"role": "assistant", "content": "working" if len(seen) == 1 else "done"}, 0.0

    monkeypatch.setattr("ouroboros.llm.LLMClient.chat", lambda *a, **k: pytest.fail("no provider call"))
    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    # The real emitter (OuroborosAgent._emit_progress) also takes narration=; the borrowed fixture's lambda takes incident= only.
    shown = []
    loop_kwargs = dict(_loop_kwargs(tmp_path, registry, []),
                       emit_progress=lambda text, **meta: shown.append((text, meta.get("narration"))))
    result, _usage, _trace = loop.run_llm_loop(**loop_kwargs, task_type="task", event_queue=events)
    assert result == "done"
    assert ("working", True) in shown, "the commentary round is shown as the model's own words"
    ordinary = "[Message from ancestor task parent-task]\nexact ordinary note"
    relayed = "[Message from task peer-task, relayed by ancestor parent-task]\nexact relayed note"
    history = "[Previous-attempt mailbox history; not a new delivery]\n"
    assert len(seen) == 2
    for text in seen:
        assert text.count(ordinary) == text.count(relayed) == 1
        assert text.index(ordinary) < text.index(relayed)
        assert text.count(history) == 2 * int(start in {"unfrozen", "unusable"})
        if start in {"unfrozen", "unusable"}:
            assert history + ordinary in text and history + relayed in text
    independent = "[Message from independent task other-root]\nexact independent note"
    assert [text.count(independent) for text in seen] == [int(start in {"first", "restored"})] * 2
    # Each delivery this run made, live or recovered, is one producer fact on the exact row that carries the letter;
    # the restored checkpoint's rows carry no receipt of the dead attempt's deliveries, so none is invented.
    letters = {"ordinary": ordinary, "relayed": relayed, "independent": independent}
    delivered = {"first": ("ordinary", "relayed", "independent"), "restored": ()}.get(start, ("ordinary", "relayed"))
    for sources in mail_sources:
        assert sorted(source for source, _row in sources) == sorted(digest(msg_id) for msg_id in delivered)
        for msg_id in delivered:
            (row,) = [row for source, row in sources if source == digest(msg_id)]
            assert letters[msg_id] in row and row.startswith(history) is (start != "first")
    assert ("could not be restored" in seen[0]) is (start == "unusable")
    assert ("continued from its working checkpoint" in seen[0]) is (start == "restored")
    injections = [row["source_task_id"] for row in events.queue if row.get("type") == "task_message_injected"]
    if start == "first":
        assert injections == ["parent-task", "parent-task", "other-root"]
        assert len(ack_path.read_bytes().splitlines()) == 3
    else:
        assert injections == [] and ack_path.read_bytes() == before


def test_representative_checkpoint_io_is_measured_not_promised(tmp_path):
    """Disclosure only (owner F7): bytes and encode+replace time at ~12 MB, no threshold."""
    _ctx, limit_ctx = _loop_ctx(tmp_path)
    blob = "x" * 4000
    limit_ctx.messages = [{"role": "tool", "tool_call_id": f"c{i}", "content": blob} for i in range(3000)]
    started = time.perf_counter()
    wc.save_round(limit_ctx, "post_batch")
    elapsed = time.perf_counter() - started
    stats = limit_ctx.accumulated_usage["working_checkpoint"]
    assert stats["last_whole_bytes"] > 11_000_000, "the state is representative; its file is a skeleton over CAS parts"
    print(f"WORKING_CHECKPOINT_IO whole_bytes={stats['last_whole_bytes']} file_bytes={stats['last_bytes']} "
          f"parts={stats['parts_written']} encode_ms={stats['encode_ms']} write_ms={stats['write_ms']} total_s={elapsed:.3f}")


@pytest.mark.parametrize("start", ["first", "restored", "emptied", "unfrozen", "unusable", "busy"])
def test_a_retry_that_restores_nothing_starts_from_the_rows_its_own_record_holds(tmp_path, monkeypatch, start):
    """#3555: attempt 2 that restores nothing builds its held rows and open relays from its own record before its first
    send, so the checkpoint it saves parks them; a first start and a restore (an empty map too) read nothing. After
    one unavailable read, the first saved checkpoint must still respect the record's closed address."""
    from ouroboros import loop, owner_relay

    task_id, kept, relay = "t-held", {"row:1@a": {"text": "Please keep the report short."}}, {"relay-1": {"note": "inert"}}
    parked = ({"row:1@b": {"text": "Use the second draft."}}, {"relay-2": {"note": "inert too"}}) if start == "restored" else ({}, {})
    write_task_result(tmp_path, task_id, "running")
    previous = _registry(tmp_path, monkeypatch, task_id, 1)
    previous._ctx._held_owner_messages, previous._ctx._open_relays = dict(kept), dict(relay)
    if start == "busy":
        previous._ctx._open_relays = {}  # attempt 1 closed the inherited relay's return address
    owner_relay.persist_held_state(previous._ctx)  # the record keeps what attempt 1 retained
    task = {"id": task_id}
    if start not in {"first", "busy"}:
        previous._ctx._held_owner_messages, previous._ctx._open_relays = map(dict, parked)  # its checkpoint parks another view
        wc.save_ready(SimpleNamespace(tools=previous, messages=[{"role": "user", "content": "go"}], llm_trace={"tool_calls": []},
                                     accumulated_usage={}, round_idx=1, tool_schemas=[], owner_msg_seen=set(), budget_tail="no_tool"))
        if start == "unfrozen":
            wc.checkpoint_path(tmp_path, task_id, 1).unlink()
        assert wc.attach_recovery(tmp_path, task, source_task_id=task_id, from_attempt=1,
                                  cause="worker_crash") is (start != "unfrozen")
        if start == "unusable":
            task["_working_recovery"]["from_attempt"] = 9  # identity mismatch
    registry = _registry(tmp_path, monkeypatch, task_id, 1 if start == "first" else 2)
    registry._ctx.working_recovery = task.get("_working_recovery")
    inherited = {"row:1@b": {"text": "Use the second draft."}}
    if start == "busy":
        registry._ctx.task_metadata = dict(getattr(registry._ctx, "task_metadata", None) or {},
                                           continuation={"owner_sources": {
                                               "held_owner_messages": inherited, "open_relays": dict(relay)}})
    reads, first = [], {}
    own_state = owner_relay._own_held_state

    def _read(ctx):
        reads.append(1)
        if start == "busy" and len(reads) == 1:
            return None  # the first read is unavailable, not an empty record
        return own_state(ctx)

    monkeypatch.setattr(owner_relay, "_own_held_state", _read)
    persists, real_persist = [], owner_relay.persist_held_state
    monkeypatch.setattr(owner_relay, "persist_held_state",  # leave the writer real; witness its invocation
                        lambda ctx: persists.append(ctx) or real_persist(ctx))
    monkeypatch.setattr("ouroboros.llm.LLMClient.chat", lambda *a, **k: pytest.fail("no provider call"))

    def dispatch(call, disposition, *, candidate_predicate=None, **kwargs):
        ctx = call.tools._ctx  # raw attributes: asking the seeder would build the very state under test
        if not first:
            snap = lambda value: None if value is None else dict(value)
            first.update(maps=(snap(getattr(ctx, "_held_owner_messages", None)), snap(getattr(ctx, "_open_relays", None))),
                         reads=len(reads))
            assert not persists, "restoring the maps must not invoke the held-map writer before the first send"
            # The real loop's pre-send save already ran. Read exactly what it parked.
            first["parked"] = tuple(map(wc.read_working(tmp_path, task_id, ctx.task_attempt)[1].get, ("held_owner_messages", "open_relays")))
        ctx._round_response_facts = {"finish_reason_present": True, "finish_reason": "stop"}
        return {"role": "assistant", "content": "done"}, 0.0

    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    result, _usage, _trace = loop.run_llm_loop(
        messages=[{"role": "user", "content": "go"}], tools=registry, llm=SimpleNamespace(default_model=lambda: "test-model"),
        drive_logs=tmp_path, emit_progress=lambda text, **meta: None, incoming_messages=queue.Queue(), task_id=task_id,
        drive_root=tmp_path, task_type="task", event_queue=queue.Queue())
    assert result == "done"
    held, relays = first["maps"]
    if start in {"unfrozen", "unusable"}:
        assert (held, relays) == first["parked"] == (kept, relay) and held is not kept and first["reads"]
    elif start == "first":
        assert held is None and relays is None and first["reads"] == 0  # a healthy first start builds and reads nothing
    elif start == "busy":
        assert (held, relays) == first["parked"] == ({**kept, **inherited}, {})
        assert first["reads"] == 2  # the retry settled the inherited view against its own record
    else:
        assert (held, relays) == first["parked"] == parked and first["reads"] == 0  # the checkpoint's own maps, empty ones too


def test_an_unreadable_record_keeps_a_retry_view_and_no_return_address(tmp_path, monkeypatch):
    """The inherited view stays provisional until the task's own record can be read."""
    from ouroboros import owner_relay, task_store
    from ouroboros.store import StoreError

    def unreadable(*args, **kwargs):
        raise StoreError("reader convoy")

    ctx = _registry(tmp_path, monkeypatch, "t-blind", 2)._ctx
    ctx.task_metadata = {"continuation": {"owner_sources": {
        "held_owner_messages": {"row:1@a": {"text": "the inherited row"}},
        "open_relays": {"r1": {"relay_id": "r1", "source": {"chat_id": 1}}}}}}
    monkeypatch.setattr(task_store, "read_document", unreadable)
    held = owner_relay.held_owner_messages(ctx, defer_persist=True)
    assert held == {"row:1@a": {"text": "the inherited row"}} and list(ctx._open_relays) == ["r1"]
    assert ctx._held_state_seed == {"r1"} and ctx._held_state_dirty is True
    assert owner_relay.select_return(ctx, "") == (
        {}, "RELAY_RETURN_UNRESOLVED: this start's own record could not be read; no return address is confirmed")


def test_a_held_question_wait_continues_through_its_own_reader_after_quit_and_resume(tmp_path, monkeypatch):
    """Parent 2026-10-08: a REAL question park → application stop → actual boot restore
    → Resume → the real loop. The wait's own continuation source is named as such
    (``source_kind=owner_wait``, bound to its ``wait_id``), never posing as a rolling
    state; the consumed wait reads ``resumed`` (an answer is still accepted)."""

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
    stored = read_result(tmp_path, "asker")["owner_wait"]
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


def cold_continuation(ctx, trace, monkeypatch, *, writer="working", new_id=False, legacy=False):
    """``ctx`` saved through a real writer — the rolling working checkpoint (its encoder turns non-JSON values into
    opaque markers) or a parked owner wait (strict JSON) — attached to a retry by the supervisor's own carrier
    (``restart_retention._recovery_source``) and continued as a new worker process does: a fresh context, the
    loop's turn reset, then ``resume_from_working`` for the same task or a new execution id. Only the route
    rebind, which needs a live model plan, is stubbed. ``legacy`` drops the result review's shown facts from
    the saved bytes, as a source written before they were carried. Returns ``(fresh context, restored trace)``."""
    from ouroboros import loop, owner_wait
    from ouroboros.tools.registry import ToolContext
    from supervisor.restart_retention import _recovery_source

    root, messages = ctx.drive_root, [{"role": "user", "content": "task"}]
    ctx.task_attempt = 1  # a worker's context always names its attempt
    write_task_result(root, ctx.task_id, "running")
    if writer == "owner_wait":
        wait = owner_wait.checkpoint_owner_wait(ctx, messages, trace, {}, 3, [], set())
        owner_wait.set_owner_wait(root, ctx.task_id, {**wait, "state": "waiting"})
    else:
        wc.save_round(SimpleNamespace(tools=SimpleNamespace(_ctx=ctx), messages=messages, llm_trace=trace,
                                      accumulated_usage={}, round_idx=3, tool_schemas=[], owner_msg_seen=set(),
                                      budget_tail="no_tool"), "candidate")
    if legacy:
        path = wc.checkpoint_path(root, ctx.task_id, 1)
        state = wc.read_working(root, ctx.task_id, 1)[1]  # written back whole: the reader takes both forms
        state["acceptance"] = {key: value for key, value in state["acceptance"].items()
                               if not key.startswith("_result_review_")}
        path.write_text(json.dumps(state), encoding="utf-8")
    task = {"id": ctx.task_id, "_attempt": 1}
    assert _recovery_source(root, task, 1, "restart")
    fresh = ToolContext(repo_dir=ctx.repo_dir, system_repo_dir=ctx.system_repo_dir, drive_root=root,
                        workspace_root=ctx.workspace_root, workspace_mode=ctx.workspace_mode,
                        task_id="task-retry" if new_id else ctx.task_id)
    fresh.task_attempt, fresh.working_recovery = 2, task["_working_recovery"]
    loop._reset_turn_state(fresh)
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_a, **_k: (None, "max"))
    restored: dict = {}
    wc.resume_from_working(SimpleNamespace(_ctx=fresh), wc.load_recovery(fresh), [], restored, {}, set())
    return fresh, restored
