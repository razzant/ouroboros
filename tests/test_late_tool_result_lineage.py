"""A late tool result owed to a Continue lineage is never lost, and its readers keep their bounds (#1376).

Continue admission holds while a predecessor's timed-out call still runs (its ``launch_handoffs``
claim), so that result lands before any successor exists or never. A successor's FIRST request must
still carry every such result that landed before it: one older than any recent-call window, one owed
through five ancestors that died before their first turn, past a malformed record; one whose ancestor
could not be read arrives in the next round. An ancestor's call still unsettled when its successor
first looks is probed once. Real dispatcher, loop, queue custody check, evidence packer, frozen
Presence registry and warm owner Pause; scaled times. Fixtures: ``tests/test_late_tool_result_delivery.py``.
"""
from __future__ import annotations

import json
import time
from dataclasses import replace

import ouroboros.loop_tool_execution as lte
from ouroboros.task_results import load_task_result, write_task_result
from tests.test_configured_session_restore import InputReached
from tests.test_late_tool_result_delivery import (
    BODY, TASK, Slow, _call, _late_text, _locators, _loop_registry, _rows, _run, _settled, _tools,
)


def _timed_out(tmp_path, monkeypatch, task_id="pred-task", guard=20.0):
    """A real root whose slow call returned TOOL_TIMEOUT through the dispatcher; the call still runs."""
    slow = Slow(guard)
    registry = _loop_registry(tmp_path, monkeypatch, task_id=task_id, slow=slow)
    result = lte._execute_with_timeout(registry, _call("slow_probe")["tool_calls"][0], tmp_path / "logs",
                                       lte._get_tool_timeout(registry, "slow_probe", {}), task_id=task_id)
    assert result["tool_result"].code == "TOOL_TIMEOUT"
    return registry, slow


def _settle(tmp_path, slow, predecessor="pred-task"):
    slow.release.set()
    ref = _settled(tmp_path / "logs")["result_ref"]
    write_task_result(tmp_path, predecessor, "failed", reason_code="provider_unavailable")
    return ref


def _successor_requests(tmp_path, monkeypatch, task_id, predecessor, slow, rounds=1):
    """The successor's real loop; returns the late-result notes of each assembled request."""
    metadata = {"continuation": {"predecessor_task_id": predecessor}}
    write_task_result(tmp_path, task_id, "running", metadata=metadata)
    registry = _loop_registry(tmp_path, monkeypatch, task_id=task_id, slow=slow)
    registry._ctx.task_metadata = metadata
    seen = []

    def dispatch(call, *_args, **_kwargs):
        seen.append(_locators(call.messages))
        registry.watch = json.loads(json.dumps((registry._ctx._execution_trace or {}).get("late_tool_watch")))
        if len(seen) < rounds:
            return _call("fast_probe", f"{task_id}-{len(seen)}"), 0.0
        raise InputReached

    _run(tmp_path, registry, task_id, dispatch, monkeypatch)
    return seen, registry


def test_a_successor_gets_a_result_older_than_two_hundred_newer_calls(tmp_path, monkeypatch):
    from ouroboros.tool_call_log import logical_calls

    registry, slow = _timed_out(tmp_path, monkeypatch, guard=240.0)  # pending through 201 calls under load
    try:
        for index in range(201):
            call = _call("fast_probe", f"newer-{index}")["tool_calls"][0]
            lte._execute_with_timeout(registry, call, tmp_path / "logs", 1, task_id="pred-task")
        own = [row for row in _rows(tmp_path / "logs") if row.get("task_id") == "pred-task"]
        assert len(logical_calls(own)) == 202, "the timed-out call is older than 201 newer ones"
    finally:
        slow.release.set()  # an early failure must not strand the held worker for its 240 s guard
    ref = _settle(tmp_path, slow)
    seen, successor = _successor_requests(tmp_path, monkeypatch, "succ-task", "pred-task", slow)
    (note,) = seen[0]
    assert "task_id='pred-task'" in note and f"call_id={ref['call_id']!r}" in note
    assert f"manifest_sha256={ref['sha256']!r}" in note
    assert _late_text(successor._ctx, "pred-task", ref) == BODY and slow.calls == 1


def test_an_owed_result_reaches_the_next_successor_when_the_first_never_took_a_turn(tmp_path, monkeypatch):
    _registry, slow = _timed_out(tmp_path, monkeypatch)
    ref = _settle(tmp_path, slow)
    # The first successor was admitted after the settlement and died before its first model request.
    write_task_result(tmp_path, "succ-1", "failed", reason_code="worker_crash_signal",
                      metadata={"continuation": {"predecessor_task_id": "pred-task"}})
    seen, second = _successor_requests(tmp_path, monkeypatch, "succ-2", "succ-1", slow)
    (note,) = seen[0]
    assert "task_id='pred-task'" in note and f"call_id={ref['call_id']!r}" in note
    assert second.watch["walked"] == ["succ-1", "pred-task"] and second.watch["pending"] == []
    assert _late_text(second._ctx, "pred-task", ref) == BODY


def _died_before_a_turn(tmp_path, count):
    """``succ-1`` .. ``succ-<count>``, each the Continue of the one before, none took a model turn."""
    for index in range(1, count + 1):
        previous = "pred-task" if index == 1 else f"succ-{index - 1}"
        write_task_result(tmp_path, f"succ-{index}", "failed", reason_code="worker_crash_signal",
                          metadata={"continuation": {"predecessor_task_id": previous}})


def test_the_first_request_carries_a_result_owed_through_five_ancestors(tmp_path, monkeypatch):
    _registry, slow = _timed_out(tmp_path, monkeypatch)
    ref = _settle(tmp_path, slow)
    _died_before_a_turn(tmp_path, 4)
    seen, fifth = _successor_requests(tmp_path, monkeypatch, "succ-5", "succ-4", slow)
    (note,) = seen[0]  # the walk finished before the first send; nothing waits for another round
    assert "task_id='pred-task'" in note and f"call_id={ref['call_id']!r}" in note
    assert fifth.watch["walked"] == ["succ-4", "succ-3", "succ-2", "succ-1", "pred-task"]
    assert fifth.watch["pending"] == [] and _late_text(fifth._ctx, "pred-task", ref) == BODY


def test_a_wait_end_record_that_is_no_manifest_does_not_stall_the_walk(tmp_path, monkeypatch):
    from ouroboros.observability import call_manifest_path

    _registry, slow = _timed_out(tmp_path, monkeypatch)
    ref = _settle(tmp_path, slow)
    _died_before_a_turn(tmp_path, 1)
    malformed = call_manifest_path(tmp_path, "succ-1", "tool_timeout_slow_probe_" + "e" * 32)
    malformed.parent.mkdir(parents=True, exist_ok=True)
    malformed.write_text("[]", encoding="utf-8")  # parses, but pairs with nothing
    seen, second = _successor_requests(tmp_path, monkeypatch, "succ-2", "succ-1", slow, rounds=2)
    (note,) = seen[0]
    assert f"call_id={ref['call_id']!r}" in note and second.watch["walked"] == ["succ-1", "pred-task"]
    assert "retry" not in second.watch, "a permanent malformation is no transient read failure"


def test_an_ancestors_call_that_never_settled_is_probed_once(tmp_path, monkeypatch):
    from ouroboros import late_tool_results

    _registry, slow = _timed_out(tmp_path, monkeypatch)
    write_task_result(tmp_path, "pred-task", "failed", reason_code="provider_unavailable")
    real, probes = late_tool_results.late_settlement, []
    monkeypatch.setattr(late_tool_results, "late_settlement", lambda *args: probes.append(args[1:]) or real(*args))
    try:  # unsettled when its successor first looks: Continue admission held while it ran, so it never will
        seen, successor = _successor_requests(tmp_path, monkeypatch, "succ-task", "pred-task", slow, rounds=3)
        assert seen == [[], [], []] and len(probes) == 1, probes
        assert successor.watch["walked"] == ["pred-task"] and successor.watch["invocations"] == []
    finally:
        slow.release.set()
    # Were it to settle anyway (admission bypassed), the next Continue context still finds it.
    ref = _settle(tmp_path, slow)
    later, _third = _successor_requests(tmp_path, monkeypatch, "succ-2", "succ-task", slow)
    (note,) = later[0]
    assert f"call_id={ref['call_id']!r}" in note and slow.calls == 1


def test_an_ancestors_reciprocal_retry_predecessor_is_walked_too(tmp_path, monkeypatch):
    _registry, slow = _timed_out(tmp_path, monkeypatch)
    ref = _settle(tmp_path, slow)
    # The interrupted root was retried under a new id (exact reciprocal links); the retry died before a turn.
    write_task_result(tmp_path, "pred-task", "failed", reason_code="provider_unavailable", retry_task_id="retry-task")
    write_task_result(tmp_path, "retry-task", "failed", reason_code="worker_crash_signal",
                      timeout_retry_from="pred-task", original_task_id="pred-task")
    seen, successor = _successor_requests(tmp_path, monkeypatch, "succ-task", "retry-task", slow)
    (note,) = seen[0]
    assert "task_id='pred-task'" in note and f"call_id={ref['call_id']!r}" in note
    assert _late_text(successor._ctx, "pred-task", ref) == BODY


def test_an_unreadable_ancestor_is_read_again_and_its_result_still_arrives(tmp_path, monkeypatch):
    from ouroboros import late_tool_results
    from ouroboros import loop

    notices = []
    append = loop._append_or_merge_user_message

    def observe(messages, text, **kwargs):
        notices.append(text)
        return append(messages, text, **kwargs)

    monkeypatch.setattr(loop, "_append_or_merge_user_message", observe)

    _registry, slow = _timed_out(tmp_path, monkeypatch)
    ref = _settle(tmp_path, slow)
    real, failed = late_tool_results._own_timeouts, []

    def unreadable_once(roots, task_id):
        if task_id == "pred-task" and not failed:
            failed.append(task_id)
            raise PermissionError("the predecessor's call folder is unreadable for a moment")
        return real(roots, task_id)

    monkeypatch.setattr(late_tool_results, "_own_timeouts", unreadable_once)
    seen, _successor = _successor_requests(tmp_path, monkeypatch, "succ-task", "pred-task", slow, rounds=2)
    assert failed == ["pred-task"] and seen[0] == [], "nothing is claimed while the ancestor is unreadable"
    gap, = [text for text in notices if "[LATE_TOOL_RESULT_UNAVAILABLE]" in text]
    assert "pred-task" in gap and "unknown" in gap
    (note,) = seen[1]
    assert f"call_id={ref['call_id']!r}" in note


def test_continue_admission_holds_while_the_predecessors_timed_out_call_still_runs(tmp_path, monkeypatch):
    from supervisor.continuation_admission import CONTINUE_PURPOSE, conflicting_writers
    from tests._budget_pause_exact_helpers import _install_queue

    queue, _state, _workers = _install_queue(tmp_path, monkeypatch)
    _registry, slow = _timed_out(tmp_path, monkeypatch)
    write_task_result(tmp_path, "pred-task", "failed", reason_code="provider_unavailable")

    def tool_holds():
        return [row for row in conflicting_writers(queue, "pred-task", purpose=CONTINUE_PURPOSE)
                if row["kind"] == "tool_handoff"]

    assert tool_holds(), "a successor cannot be admitted while the timed-out call still runs"
    slow.release.set()
    _settled(tmp_path / "logs")
    deadline = time.monotonic() + 15
    while tool_holds() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not tool_holds(), "the settled call releases the hold; its result already exists"


def test_evidence_packing_keeps_late_manifests_loose_and_the_source_readable(tmp_path, monkeypatch):
    from ouroboros import evidence_packer
    from ouroboros.late_tool_results import late_settlement
    from ouroboros.observability import call_manifest_path

    registry, slow = _timed_out(tmp_path, monkeypatch, task_id="packed-task")
    slow.release.set()
    settled = _settled(tmp_path / "logs")
    ref = settled["result_ref"]
    write_task_result(tmp_path, "packed-task", "completed", result="done")
    manifest = json.loads(call_manifest_path(tmp_path, "packed-task", ref["call_id"]).read_text(encoding="utf-8"))
    blob = manifest["redacted_projection_ref"]
    loose_blob = tmp_path / "observability" / "blobs" / f"{blob['sha256']}.{blob['kind']}.gz"
    assert loose_blob.is_file()
    report = evidence_packer.run_pass(tmp_path)
    assert report["packed"] >= 1 and not loose_blob.exists(), report
    timeout_id = "tool_timeout_slow_probe_" + settled["invocation_id"]
    for call_id in (ref["call_id"], timeout_id):
        assert call_manifest_path(tmp_path, "packed-task", call_id).is_file(), "call manifests are never packed"
    found = late_settlement([tmp_path], "packed-task", "slow_probe", settled["invocation_id"])
    assert (found["call_id"], found["manifest_sha256"]) == (ref["call_id"], ref["sha256"])
    assert _late_text(registry._ctx, "packed-task", ref) == BODY, "the payload is read from its pack"


def test_presence_reads_its_own_late_result_only_inside_its_ceiling(tmp_path, monkeypatch):
    from ouroboros.loop_nudges import _inject_round_checkpoints
    from ouroboros.loop_tool_execution import StatefulToolExecutor, handle_tool_calls
    from ouroboros.presence_authority import PresenceResourceGrant, presence_ceiling_payload
    from ouroboros.tools.tool_resolution import _root_containing_absolute_path
    from tests.test_presence_own_work import _ceiling, _registry as presence_registry

    registry, ctx = presence_registry(tmp_path, task_id="presence-late")
    write_task_result(tmp_path, ctx.task_id, "running", metadata=ctx.task_metadata)
    slow = Slow()
    _tools(registry, monkeypatch, slow)
    monkeypatch.setitem(registry._entries, "recent_tasks",  # granted by the own-work baseline
                        replace(registry._entries["recent_tasks"], handler=slow, timeout_sec=1))
    logs, trace = tmp_path / "logs", {"tool_calls": []}
    call = _call("recent_tasks", "presence-call")
    handle_tool_calls(call["tool_calls"], registry, logs, ctx.task_id, StatefulToolExecutor(),
                      [{"role": "user", "content": "go"}, call], trace, lambda *_a, **_k: None)
    assert trace["tool_calls"][-1]["tool_result_code"] == "TOOL_TIMEOUT"
    slow.release.set()
    ref = _settled(logs, "recent_tasks")["result_ref"]
    arguments = {"task_id": ctx.task_id, "call_id": ref["call_id"], "manifest_sha256": ref["sha256"]}

    def announce(llm_trace):  # the loop's own per-round seam
        messages = [{"role": "user", "content": "go"}]
        _inject_round_checkpoints(round_idx=2, messages=messages, accumulated_usage={}, tools=registry,
                                  event_queue=None, task_id=ctx.task_id, drive_logs=logs, llm_trace=llm_trace)
        (note,) = _locators(messages)
        return note

    # The note states the exact address and no verdict on reach: the own-work ceiling grants the reader
    # tool but not this read, and the reader refuses it as it does every call record of this turn.
    bare = announce(trace)
    assert f"task_id={ctx.task_id!r}, call_id={ref['call_id']!r}, manifest_sha256={ref['sha256']!r}" in bare
    assert registry.execute("get_task_result", arguments) == "the Presence resource ceiling does not include this read"
    root = _root_containing_absolute_path(ctx, "read_file", ref["path"])
    ceiling = replace(_ceiling(), resource_grants=(*_ceiling().resource_grants,
                                                   PresenceResourceGrant(root, ("read",), "observability")))
    ceiling = replace(ceiling, digest=presence_ceiling_payload(ceiling)["digest"])
    ctx.task_contract = {"capability_ceiling": presence_ceiling_payload(ceiling)}
    assert announce({"tool_calls": trace["tool_calls"]}) == bare, "one address whatever the grant"
    meta = json.loads(registry.execute("get_task_result", arguments))["call_source"]  # host-bound own_binding
    page = json.loads(registry.execute("get_task_result", {**arguments, "source_start_char": 0,
                                                            "source_end_char": meta["complete_chars"]}))
    assert json.loads(page["call_source"]["text"])["result"] == BODY
    stranger, _ = presence_registry(tmp_path, ceiling, binding="b" * 32, task_id="stranger")
    assert "PRESENCE_CAPABILITY_BLOCKED" in stranger.execute("get_task_result", arguments)
    assert slow.calls == 1


def test_a_warm_owner_resume_carries_the_locator_into_the_resumed_round(tmp_path, monkeypatch):
    from ouroboros import owner_pause, review_pause

    slow = Slow()
    registry = _loop_registry(tmp_path, monkeypatch, slow=slow)
    real_timeout = lte._make_timeout_result

    def owner_pauses_as_the_wait_ends(*args, **kwargs):
        result = real_timeout(*args, **kwargs)
        owner_pause.install_fence(tmp_path, TASK, request_id="owner-pause")
        return result

    parked = []

    def warm_owner_wait(_ctx, checkpoint):  # the worker holds this same stack until the owner's Resume
        parked.append((checkpoint.get("owner_pause") or {}).get("fence_id"))
        slow.release.set()
        _settled(tmp_path / "logs")  # the late result lands while the stack is parked warm
        owner_pause.release_fence(tmp_path, TASK, reason="owner_resume")
        return "control:owner_resume"

    monkeypatch.setattr(lte, "_make_timeout_result", owner_pauses_as_the_wait_ends)
    monkeypatch.setattr(review_pause, "live_detached_operations", lambda _task_id: ["review-op"])
    registry._ctx.owner_wait_callback = warm_owner_wait
    rounds = []

    def dispatch(call, *_args, **_kwargs):
        rounds.append([dict(row) for row in call.messages])
        if len(rounds) == 1:
            return _call("slow_probe"), 0.0
        raise InputReached

    _run(tmp_path, registry, TASK, dispatch, monkeypatch)
    ref = _settled(tmp_path / "logs")["result_ref"]
    assert parked and parked[0] and len(rounds) == 2, (parked, len(rounds))
    (note,) = _locators(rounds[1])
    assert f"call_id={ref['call_id']!r}" in note and slow.calls == 1
    assert load_task_result(tmp_path, TASK)["status"] == "running", "the arrival resumed nothing by itself"
