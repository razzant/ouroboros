"""A timed-out call's later settlement reaches the next authorized turn (#1376, owner decision A).

One slow handler (its sentinel sits after 2000 characters) times out at a scaled 1 s and settles
later, exactly once, through the real dispatcher. The next assembled model request of the same
running task, of its Resume from the saved working state and of a Continue successor carries one
exact locator; the standard ``get_task_result`` reads the complete result; another task, attempt,
invocation, digest, a reused provider call id and an unbound manifest are refused or kept apart;
arrival changes no status and schedules nothing; the source survives its child drive's disposal.
Every settled call reaches one request; an unreadable record neither ends the round nor counts as
absent; a handler's own timely TOOL_TIMEOUT is never watched. A forced wrap-up is such a request
too, priced with its locator; a rail that sends nothing leaves the locator for the next one.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

import ouroboros.loop_tool_execution as lte
from ouroboros import loop
from ouroboros.observability import call_manifest_path
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.tools.registry import ToolEntry
from tests.test_configured_session_restore import InputReached

SENTINEL = "SENTINEL_AFTER_2000"
BODY = "A" * 2100 + SENTINEL + "Z" * 40
TASK = "late-task"


class Slow:
    """A slow handler held until released; it counts its executions."""

    def __init__(self, guard=20.0):
        self.release, self.calls, self.guard = threading.Event(), 0, guard

    def __call__(self, ctx, **_kwargs):
        self.calls += 1
        assert self.release.wait(self.guard), "the fixture never released the slow handler"
        return BODY


def _entry(name, handler):
    return ToolEntry(name, {"name": name, "description": "fixture",
                            "parameters": {"type": "object", "properties": {}, "required": []}},
                     handler, timeout_sec=1, safety_policy="skip")


def _fast(name, handler):
    """A fast fixture with a window no load reaches: a call past its window would be a real late result."""
    return replace(_entry(name, handler), timeout_sec=60)


def _tools(registry, monkeypatch, slow):
    """The saved tool timeout and the slow entry's window are scaled to one second."""
    for module in ("ouroboros.loop_tool_execution", "ouroboros.tools.shell_process"):
        monkeypatch.setattr(f"{module}.load_settings", lambda: {"OUROBOROS_TOOL_TIMEOUT_SEC": 1})
    monkeypatch.delenv("OUROBOROS_TOOL_TIMEOUT_SEC", raising=False)
    registry.register(_entry("slow_probe", slow))
    registry.register(_fast("fast_probe", lambda ctx, **_: "FAST_RESULT"))
    return registry


def _rows(logs):
    path = logs / "tools.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _settled(logs, tool="slow_probe", timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rows = [row for row in _rows(logs) if row["type"] == "tool_call" and row["tool"] == tool]
        if rows:
            return rows[-1]
        time.sleep(0.05)
    raise AssertionError(f"{tool} never settled")


def _read(ctx, task_id, ref, **ranges):
    from ouroboros.tools.control_task_results import _get_task_result

    return _get_task_result(ctx, task_id=task_id, call_id=ref["call_id"], manifest_sha256=ref["sha256"], **ranges)


def _late_text(ctx, task_id, ref):
    """The complete late source through the standard reader: length first, then the exact text."""
    meta = json.loads(_read(ctx, task_id, ref))["call_source"]
    text = json.loads(_read(ctx, task_id, ref, source_start_char=0,
                            source_end_char=meta["complete_chars"]))["call_source"]["text"]
    return json.loads(text)["result"]


def _call(name, call_id="call-late"):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}]}


def _loop_registry(tmp_path, monkeypatch, task_id=TASK, attempt=1, slow=None):
    from tests.test_working_checkpoint import _registry

    monkeypatch.setattr("ouroboros.llm.LLMClient.chat", lambda *a, **k: pytest.fail("no provider call"))
    if load_task_result(tmp_path, task_id) is None:
        write_task_result(tmp_path, task_id, "running")
    return _tools(_registry(tmp_path, monkeypatch, task_id, attempt), monkeypatch, slow or Slow())


def _run(tmp_path, registry, task_id, dispatch, monkeypatch):
    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    with pytest.raises(InputReached):
        loop.run_llm_loop(messages=[{"role": "user", "content": "go"}], tools=registry,
                          llm=SimpleNamespace(default_model=lambda: "fixture-model"), drive_logs=tmp_path / "logs",
                          emit_progress=lambda *_a, **_k: None, incoming_messages=queue.Queue(), task_id=task_id,
                          drive_root=tmp_path)


def _locators(messages):
    return [str(row.get("content") or "") for row in messages if "[LATE_TOOL_RESULT]" in str(row.get("content") or "")]


def _handled(registry, logs, *calls):
    """``(tool, provider id)`` calls through the loop's real tool handling; returns their trace."""
    from ouroboros.loop_tool_execution import StatefulToolExecutor, handle_tool_calls

    message = {"role": "assistant", "content": "", "tool_calls": [
        _call(name, call_id)["tool_calls"][0] for name, call_id in calls]}
    trace = {"tool_calls": []}
    handle_tool_calls(message["tool_calls"], registry, logs, TASK, StatefulToolExecutor(),
                      [{"role": "user", "content": "go"}, message], trace, lambda *_a, **_k: None)
    return trace


def _round(registry, trace, logs):
    """The late-result notes the loop's own per-round seam adds to the next request."""
    from ouroboros.loop_nudges import _inject_round_checkpoints

    messages = [{"role": "user", "content": "go"}]
    _inject_round_checkpoints(round_idx=2, messages=messages, accumulated_usage={}, tools=registry,
                              event_queue=None, task_id=TASK, drive_logs=logs, llm_trace=trace)
    return _locators(messages)


@pytest.mark.parametrize("settle_first", [False, True], ids=["settles-after-the-timeout-record",
                                                              "settles-before-the-timeout-record"])
def test_the_next_round_of_the_running_task_carries_one_exact_locator(tmp_path, monkeypatch, settle_first):
    slow = Slow()
    registry = _loop_registry(tmp_path, monkeypatch, slow=slow)
    logs, rounds = tmp_path / "logs", []
    if settle_first:  # the worker settles before the wait-end is even recorded
        real = lte._make_timeout_result

        def timeout_after_settlement(*args, **kwargs):
            slow.release.set()
            _settled(logs)
            return real(*args, **kwargs)

        monkeypatch.setattr(lte, "_make_timeout_result", timeout_after_settlement)

    def dispatch(call, *_args, **_kwargs):
        rounds.append([dict(row) for row in call.messages])
        if len(rounds) == 1:
            return _call("slow_probe"), 0.0
        if len(rounds) == 2:
            assert bool(_locators(call.messages)) is settle_first
            slow.release.set()
            _settled(logs)
            return _call("fast_probe"), 0.0  # a later call reusing the provider's id
        if len(rounds) == 3:
            return _call("fast_probe", "call-next"), 0.0
        raise InputReached

    _run(tmp_path, registry, TASK, dispatch, monkeypatch)
    settled = _settled(logs)
    ref = settled["result_ref"]
    seen = rounds[1] if settle_first else rounds[2]
    (note,) = _locators(seen)
    assert f"call_id={ref['call_id']!r}" in note and f"manifest_sha256={ref['sha256']!r}" in note
    assert f"task_id={TASK!r}" in note and "provider call call-late" in note and "slow_probe" in note
    assert all(len(_locators(messages)) == 1 for messages in rounds[rounds.index(seen):]), "announced once"
    assert slow.calls == 1, "the handler ran once; nothing re-ran it"
    same_id = [row["content"] for row in rounds[-1] if row.get("role") == "tool" and row.get("tool_call_id") == "call-late"]
    assert len(same_id) == 2 and "TOOL_TIMEOUT" in same_id[0], "the recorded timeout is never rewritten"
    assert "FAST_RESULT" in same_id[1] and SENTINEL not in same_id[1], "the reused id keeps its own result"
    timeout_rows = [row for row in _rows(logs) if row["type"] == "tool_call_timeout"]
    assert [row["invocation_id"] for row in timeout_rows] == [settled["invocation_id"]]
    assert _late_text(registry._ctx, TASK, ref) == BODY and BODY.index(SENTINEL) > 2000


def test_the_reader_admits_only_the_bound_late_settlement(tmp_path, monkeypatch):
    slow = Slow()
    registry = _loop_registry(tmp_path, monkeypatch, slow=slow)
    logs = tmp_path / "logs"
    outer = lte._get_tool_timeout(registry, "slow_probe", {})
    result = lte._execute_with_timeout(registry, _call("slow_probe")["tool_calls"][0], logs, outer, task_id=TASK)
    assert result["tool_result"].code == "TOOL_TIMEOUT" and outer == 1
    fast = lte._execute_with_timeout(registry, _call("fast_probe", "call-late")["tool_calls"][0], logs,
                                     lte._get_tool_timeout(registry, "fast_probe", {}), task_id=TASK)
    assert fast["result"] == "FAST_RESULT"
    slow.release.set()
    ref = _settled(logs)["result_ref"]
    ctx = registry._ctx
    assert _late_text(ctx, TASK, ref) == BODY

    write_task_result(tmp_path, "other-task", "running")
    refusals = {
        "other task": _read(ctx, "other-task", ref),
        "other digest": _read(ctx, TASK, {**ref, "sha256": "0" * 64}),
        "other invocation (a call that never timed out)": _read(ctx, TASK, _settled(logs, "fast_probe")["result_ref"]),
    }
    path = call_manifest_path(tmp_path, TASK, ref["call_id"])
    original = path.read_bytes()
    for label, change in (("other attempt", {"task_attempt": 7}), ("unbound (no invocation id)", {"invocation_id": None})):
        forged = {**json.loads(original), **change}
        path.write_text(json.dumps(forged), encoding="utf-8")
        refusals[label] = _read(ctx, TASK, {**ref, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    path.write_bytes(original)
    refused = ("call identity is neither published by this task result "
               "nor the later settlement of one of its timed-out calls")
    assert {label: text for label, text in refusals.items() if text != refused} == {}, refusals
    assert _late_text(ctx, TASK, ref) == BODY, "the genuine pair still reads after the forgeries"
    assert slow.calls == 1


def test_resume_from_the_saved_working_state_announces_a_result_that_settled_while_down(tmp_path, monkeypatch):
    from ouroboros import working_checkpoint as wc

    slow = Slow()
    first = _loop_registry(tmp_path, monkeypatch, slow=slow)

    def attempt_one(call, *_args, **_kwargs):
        if not any(row.get("role") == "tool" for row in call.messages):
            return _call("slow_probe"), 0.0
        assert not _locators(call.messages)
        raise InputReached  # the worker dies here; its saved state names the timed-out call

    _run(tmp_path, first, TASK, attempt_one, monkeypatch)
    write_task_result(tmp_path, TASK, "paused")
    slow.release.set()
    ref = _settled(tmp_path / "logs")["result_ref"]
    assert load_task_result(tmp_path, TASK)["status"] == "paused", "arrival changes no status, starts nothing"
    write_task_result(tmp_path, TASK, "running")  # the owner's Resume, not the arrival
    task: dict = {"id": TASK}
    assert wc.attach_recovery(tmp_path, task, source_task_id=TASK, from_attempt=1, cause="worker_crash")
    resumed = _loop_registry(tmp_path, monkeypatch, attempt=2, slow=slow)
    resumed._ctx.working_recovery = task["_working_recovery"]
    seen = []

    def attempt_two(call, *_args, **_kwargs):
        seen.append(_locators(call.messages))
        raise InputReached

    _run(tmp_path, resumed, TASK, attempt_two, monkeypatch)
    (note,) = seen[0]
    assert f"call_id={ref['call_id']!r}" in note and "attempt 1" in note
    assert _late_text(resumed._ctx, TASK, ref) == BODY and slow.calls == 1


def test_a_continue_successor_gets_its_predecessors_late_result(tmp_path, monkeypatch):
    slow = Slow()
    predecessor = _loop_registry(tmp_path, monkeypatch, task_id="pred-task", slow=slow)

    def interrupted(call, *_args, **_kwargs):
        if not any(row.get("role") == "tool" for row in call.messages):
            return _call("slow_probe"), 0.0
        raise InputReached

    _run(tmp_path, predecessor, "pred-task", interrupted, monkeypatch)
    write_task_result(tmp_path, "pred-task", "failed", reason_code="provider_unavailable")
    # Continue admission holds until the timed-out call settles (``test_late_tool_result_lineage``).
    slow.release.set()
    ref = _settled(tmp_path / "logs")["result_ref"]
    successor = _loop_registry(tmp_path, monkeypatch, task_id="succ-task", slow=slow)
    successor._ctx.task_metadata = {"continuation": {"predecessor_task_id": "pred-task"}}
    seen = []

    def successor_rounds(call, *_args, **_kwargs):
        seen.append(_locators(call.messages))
        if len(seen) == 1:
            return _call("fast_probe", "succ-call"), 0.0
        raise InputReached

    _run(tmp_path, successor, "succ-task", successor_rounds, monkeypatch)
    (note,) = seen[0]
    assert "task_id='pred-task'" in note and f"manifest_sha256={ref['sha256']!r}" in note
    assert seen[1] == [note], "announced in the first request, once"
    assert _late_text(successor._ctx, "pred-task", ref) == BODY
    assert load_task_result(tmp_path, "pred-task")["status"] == "failed" and slow.calls == 1


def test_the_late_source_survives_its_child_drive_disposal(tmp_path, monkeypatch):
    from ouroboros import headless
    from ouroboros.task_custody import settle_child_drive
    from ouroboros.tools.registry import ToolRegistry

    parent = tmp_path / "parent"
    child = parent / "task_drives" / TASK
    (child / "logs").mkdir(parents=True)
    write_task_result(parent, TASK, "completed", result="done", artifact_status="ready",
                      headless_child_drive_root=str(child),
                      child_ref_promotion={"schema_version": 1, "status": "complete", "pending_refs": []})
    slow = Slow()
    registry = _tools(ToolRegistry(repo_dir=tmp_path, drive_root=child), monkeypatch, slow)
    registry._ctx.task_id, registry._ctx.budget_drive_root = TASK, parent
    result = lte._execute_with_timeout(registry, _call("slow_probe")["tool_calls"][0], child / "logs",
                                       lte._get_tool_timeout(registry, "slow_probe", {}), task_id=TASK)
    assert result["tool_result"].code == "TOOL_TIMEOUT"
    slow.release.set()
    settled = _settled(child / "logs")
    ref = settled["result_ref"]
    assert _late_text(registry._ctx, TASK, ref) == BODY, "the live child reads its own drive's record"
    assert settle_child_drive(parent, TASK, child, live=lambda _: False)["reason"] == "call_inventory_pending"
    headless.retry_child_task_refs(parent, child, TASK)
    assert settle_child_drive(parent, TASK, child, live=lambda _: False)["status"] == "removed"
    assert not child.exists()
    registry._ctx.drive_root = parent  # a reader of the canonical store, the child drive gone
    assert _late_text(registry._ctx, TASK, ref) == BODY, "the original digest reads the retained version"
    from ouroboros.late_tool_results import late_settlement

    found = late_settlement([parent], TASK, "slow_probe", settled["invocation_id"])
    assert found and found["call_id"] == ref["call_id"], "the canonical copy is found by its own address"


def test_every_settled_call_reaches_one_request(tmp_path, monkeypatch):
    slow = Slow()
    registry = _loop_registry(tmp_path, monkeypatch, slow=slow)
    monkeypatch.setattr(lte, "_get_tool_timeout", lambda *_a, **_k: 0.2)  # nine scaled waits
    logs = tmp_path / "logs"
    trace = _handled(registry, logs, *[("slow_probe", f"call-{index}") for index in range(9)])
    assert [row["tool_result_code"] for row in trace["tool_calls"]] == ["TOOL_TIMEOUT"] * 9
    slow.release.set()
    deadline = time.monotonic() + 15
    while len([row for row in _rows(logs) if row["type"] == "tool_call"]) < 9 and time.monotonic() < deadline:
        time.sleep(0.05)
    refs = [row["result_ref"] for row in _rows(logs) if row["type"] == "tool_call"]
    (note,) = _round(registry, trace, logs)  # none waits for a model round that may never come
    assert len(refs) == 9 and note.count("\n- slow_probe") == 9
    assert all(f"call_id={ref['call_id']!r}, manifest_sha256={ref['sha256']!r}" in note for ref in refs)
    assert _round(registry, trace, logs) == [] and slow.calls == 9, "announced once"


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason="needs POSIX permissions")
def test_an_unreadable_record_keeps_its_watch_and_the_round_goes_on(tmp_path, monkeypatch):
    notices = []
    append = loop._append_or_merge_user_message

    def observe(messages, text, **kwargs):
        notices.append(text)
        return append(messages, text, **kwargs)

    monkeypatch.setattr(loop, "_append_or_merge_user_message", observe)
    slow = Slow()
    registry = _loop_registry(tmp_path, monkeypatch, slow=slow)
    logs = tmp_path / "logs"
    trace = _handled(registry, logs, ("slow_probe", "call-late"))
    assert trace["tool_calls"][-1]["tool_result_code"] == "TOOL_TIMEOUT"
    slow.release.set()
    ref = _settled(logs)["result_ref"]
    folder = call_manifest_path(tmp_path, TASK, ref["call_id"]).parent
    folder.chmod(0)  # the file system refuses the probe; no reader admission is involved
    try:
        assert _round(registry, trace, logs) == [], "unreadable now: no locator, no absence, no crash"
        assert _round(registry, trace, logs) == [], "unchanged gaps do not flood the next context"
        gaps = [text for text in notices if "[LATE_TOOL_RESULT_UNAVAILABLE]" in text]
        assert len(gaps) == 1 and TASK in gaps[0] and "unknown" in gaps[0]
        assert trace["tool_calls"][-1]["late_watch"]["invocation_id"] in gaps[0]
    finally:
        folder.chmod(0o700)
    wait_end = folder / ("tool_timeout_slow_probe_" + ref["call_id"].rsplit("_", 1)[1] + ".json")
    wait_end.chmod(0)
    try:  # the reader too names the failure: neither an access block nor a refused identity
        assert _read(registry._ctx, TASK, ref) == "Retained call source is unreadable or failed verification."
    finally:
        wait_end.chmod(0o600)
    (note,) = _round(registry, trace, logs)
    assert f"call_id={ref['call_id']!r}" in note and slow.calls == 1


def test_a_handlers_own_timely_timeout_is_never_watched(tmp_path, monkeypatch):
    from ouroboros import late_tool_results

    slow = Slow()
    registry = _loop_registry(tmp_path, monkeypatch, slow=slow)
    registry.register(_fast("timely_probe", lambda ctx, **_: "⚠️ TOOL_TIMEOUT (timely_probe): its own child timed out"))
    logs = tmp_path / "logs"
    trace = _handled(registry, logs, ("timely_probe", "call-timely"), ("slow_probe", "call-late"))
    real, probes = late_tool_results._manifest_at, []

    def probe(roots, task_id, call_id, *args):
        probes.append(call_id)
        return real(roots, task_id, call_id, *args)

    try:
        timely, waited = trace["tool_calls"]
        assert timely["tool_result_code"] == waited["tool_result_code"] == "TOOL_TIMEOUT"
        assert "late_watch" not in timely and waited["late_watch"]["invocation_id"], "only an ended outer wait"
        monkeypatch.setattr(late_tool_results, "_manifest_at", probe)
        assert [_round(registry, trace, logs) for _ in range(3)] == [[], [], []]
        assert len(probes) == 3 and not [call_id for call_id in probes if "timely_probe" in call_id], probes
    finally:
        slow.release.set()
    _settled(logs)


def _forced_send(sent):
    """``call_llm_with_retry`` at a forced rail: keeps the request it was handed to price, seal and send."""

    def send(_llm, messages, *_args, initial_messages=None, **_kwargs):
        sent.append(json.loads(json.dumps(messages if initial_messages is None else initial_messages)))
        raise InputReached

    return send


def _settled_watch(registry, logs, slow):
    """One call whose wait ended in the loop's real tool handling and that settled since:
    ``(trace, transcript, result ref)``."""
    from ouroboros.loop_tool_execution import StatefulToolExecutor, handle_tool_calls

    messages, trace = [{"role": "user", "content": "go"}, _call("slow_probe")], {"tool_calls": []}
    handle_tool_calls(messages[-1]["tool_calls"], registry, logs, TASK, StatefulToolExecutor(), messages, trace,
                      lambda *_a, **_k: None)
    assert trace["tool_calls"][-1]["late_watch"], "the wait ended"
    slow.release.set()
    return trace, messages, _settled(logs)["result_ref"]


def _limit_ctx(registry, trace, messages, logs):
    """The round-limit context the loop hands a forced rail (``_finalize_limit_ctx``)."""
    ctx = loop._RoundLimitContext(
        messages, SimpleNamespace(default_model=lambda: "fixture-model"), "fixture-model", "medium", 1, logs,
        TASK, 2, None, {}, "task", False, 1, drive_root=registry._ctx.drive_root, incoming_messages=queue.Queue(),
        owner_msg_seen=set())
    return loop._finalize_limit_ctx(ctx, registry, trace)


def test_a_round_limit_wrap_up_is_sent_with_the_locator_of_an_already_settled_result(tmp_path, monkeypatch):
    """With OUROBOROS_MAX_ROUNDS=1 the next authorized turn is the forced wrap-up, which takes the
    forced rail before the round's own notices: it still carries the locator, ahead of its prompt."""
    from ouroboros.late_tool_results import DELIVERED_KEY

    slow = Slow(60)
    registry = _loop_registry(tmp_path, monkeypatch, slow=slow)
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "1")
    logs, sent = tmp_path / "logs", []
    real = lte._make_timeout_result

    def timeout_after_settlement(*args, **kwargs):  # the worker settles as the loop's wait ends
        slow.release.set()
        _settled(logs)
        return real(*args, **kwargs)

    forced, delivered = _forced_send(sent), []

    def send(*args, **kwargs):  # the live trace as the request leaves
        delivered.extend(registry._ctx._execution_trace.get(DELIVERED_KEY) or [])
        return forced(*args, **kwargs)

    monkeypatch.setattr(lte, "_make_timeout_result", timeout_after_settlement)
    monkeypatch.setattr(loop, "call_llm_with_retry", send)
    try:
        _run(tmp_path, registry, TASK, lambda *_a, **_k: (_call("slow_probe"), 0.0), monkeypatch)
    finally:
        slow.release.set()
    settled = _settled(logs)
    ref = settled["result_ref"]
    (request,) = sent
    (note,) = _locators(request)
    assert f"call_id={ref['call_id']!r}" in note and f"manifest_sha256={ref['sha256']!r}" in note
    contents = [str(row.get("content") or "") for row in request]
    assert "[ROUND_LIMIT]" in contents[-1] and contents.index(note) < len(contents) - 1, "the prompt stays the tail"
    assert any(row.get("role") == "tool" and "TOOL_TIMEOUT" in str(row.get("content")) for row in request)
    assert [row["invocation_id"] for row in delivered] == [settled["invocation_id"]], "announced once"
    assert _late_text(registry._ctx, TASK, ref) == BODY and slow.calls == 1


def test_an_admitted_budget_wrap_up_is_priced_with_the_locator_it_sends(tmp_path, monkeypatch):
    """A budget wrap-up is priced before it is sent: the priced copy carries the locator, the admitted
    send is those bytes with no second one, and the live transcript a drifted resend uses agrees."""
    from ouroboros import budget_pause, task_pacing
    from ouroboros.late_tool_results import DELIVERED_KEY
    from tests.test_tree_cost_ceiling import EXPLICIT_50

    slow = Slow()
    registry = _loop_registry(tmp_path, monkeypatch, slow=slow)
    logs = tmp_path / "logs"
    trace, messages, ref = _settled_watch(registry, logs, slow)
    priced, sent, request = [], [], object()

    def price(_ctx, prospective, **_kwargs):
        priced.append(json.loads(json.dumps(prospective)))
        return request, prospective

    monkeypatch.setattr(budget_pause, "request_pause", lambda *_a, **_k: None)  # the pause seed is not the subject
    monkeypatch.setattr(loop, "_prepare_forced_prompt", lambda _ctx, prompt, _trace: prompt)
    monkeypatch.setattr(task_pacing, "prepared_wrapup_candidate", price)
    monkeypatch.setattr(task_pacing, "wrapup_reservation_fits", lambda **_k: True)
    monkeypatch.setattr(loop, "_loop_tree_accounting", lambda **_k: None)
    monkeypatch.setattr(loop, "call_llm_with_retry", _forced_send(sent))
    ctx = _limit_ctx(registry, trace, messages, logs)
    with pytest.raises(InputReached):
        loop._soft_land_exhausted_ceiling(ctx, task_pacing.resolve_cost_ceiling(100.0, EXPLICIT_50, root_cap_usd=0.5))
    assert sent == priced and "[BUDGET LIMIT]" in str(sent[0][-1]["content"])
    (note,) = _locators(sent[0])
    assert f"call_id={ref['call_id']!r}" in note and f"manifest_sha256={ref['sha256']!r}" in note
    assert _locators(ctx.messages) == [note]
    assert [row["call_id"] for row in trace[DELIVERED_KEY]] == [ref["call_id"]]


def test_a_refused_forced_send_keeps_the_locator_and_its_bookkeeping_together(tmp_path, monkeypatch):
    """The dispatch fence refuses the forced send before a byte leaves and the rail hands the refusal to
    the loop's authority rail: the transcript and trace any saved state takes hold the locator and its
    once-per-context record together, never the record alone."""
    from ouroboros import usage_accounting
    from ouroboros.late_tool_results import DELIVERED_KEY

    slow = Slow()
    registry = _loop_registry(tmp_path, monkeypatch, slow=slow)
    logs = tmp_path / "logs"
    trace, messages, ref = _settled_watch(registry, logs, slow)

    def refused(*_a, **_k):
        raise usage_accounting.ControlAuthorityUnavailable("unreadable fence", cause="dispatch_document_unreadable")

    monkeypatch.setattr(loop, "call_llm_with_retry", refused)
    ctx = _limit_ctx(registry, trace, messages, logs)
    with pytest.raises(usage_accounting.ControlAuthorityUnavailable):
        loop._handle_round_limit(ctx)
    (note,) = _locators(ctx.messages)
    assert f"call_id={ref['call_id']!r}" in note
    assert [row["call_id"] for row in trace[DELIVERED_KEY]] == [ref["call_id"]] and slow.calls == 1


@pytest.mark.parametrize("rail,source", [("owner_stop_now", "owner_stopped_direct_turn"),
                                         ("deadline_elapsed", "round_limit_window_elapsed"),
                                         ("timeout_repeat_spent", "timeout_repeat_spent_no_resend")])
def test_a_forced_rail_that_sends_nothing_leaves_the_locator_for_the_next_authorized_turn(
        tmp_path, monkeypatch, rail, source):
    from ouroboros.late_tool_results import DELIVERED_KEY
    from ouroboros.loop_llm_call import TIMEOUT_REPEAT_KEY
    from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN

    slow = Slow()
    registry = _loop_registry(tmp_path, monkeypatch, slow=slow)
    logs = tmp_path / "logs"
    trace, messages, ref = _settled_watch(registry, logs, slow)
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_a, **_k: pytest.fail("a no-call rail sent a request"))
    ctx = _limit_ctx(registry, trace, messages, logs)
    if rail == "owner_stop_now":
        loop._handle_forced_finalization(ctx, REASON_OWNER_STOPPED_DIRECT_TURN)
    else:
        if rail == "deadline_elapsed":
            ctx.deadline_ts = time.time() - 1
        else:
            ctx.accumulated_usage[TIMEOUT_REPEAT_KEY] = {"state": "spent", "repeat": {"failure_code": "timeout"}}
        loop._handle_round_limit(ctx)
    assert trace["forced_finalization"]["source"] == source, "the rail's own no-call guard ended it"
    assert not _locators(ctx.messages) and DELIVERED_KEY not in trace, "nothing was sent, nothing was announced"
    (note,) = _round(registry, trace, logs)
    assert f"call_id={ref['call_id']!r}" in note and slow.calls == 1


@pytest.mark.parametrize("admitted,fits", [(False, True), (True, True), (True, False)])
def test_forced_warm_resume_refreshes_and_reprices_the_late_result(tmp_path, monkeypatch, admitted, fits):
    from ouroboros import task_pacing
    from ouroboros.loop_tool_execution import StatefulToolExecutor, handle_tool_calls
    from ouroboros.model_wait import ModelWaitInterrupted

    slow = Slow(60)
    registry = _loop_registry(tmp_path, monkeypatch, slow=slow)
    logs = tmp_path / "logs"
    messages, trace = [{"role": "user", "content": "go"}, _call("slow_probe")], {"tool_calls": []}
    handle_tool_calls(messages[-1]["tool_calls"], registry, logs, TASK, StatefulToolExecutor(), messages, trace,
                      lambda *_a, **_k: None)
    assert trace["tool_calls"][-1]["late_watch"]
    sent, priced, checked = [], [], []
    balances = {"root_cap_usd": 1.0, "deciding_usd": 0.5}
    request = object()

    def send(_llm, live, *_a, initial_messages=None, **_k):
        sent.append(json.loads(json.dumps(live if initial_messages is None else initial_messages)))
        if len(sent) == 1:
            raise ModelWaitInterrupted("pause")
        raise InputReached

    def resumed(_ctx, _error):
        slow.release.set()
        _settled(logs)
        return None  # only the existing control owner authorizes this Resume

    def price(_ctx, prospective, **_k):
        priced.append(json.loads(json.dumps(prospective)))
        return request, prospective

    def fits_budget(**kwargs):
        checked.append(kwargs)
        return fits

    monkeypatch.setattr(loop, "call_llm_with_retry", send)
    monkeypatch.setattr(loop, "_handle_model_wait_control", resumed)
    monkeypatch.setattr(task_pacing, "prepared_wrapup_candidate", price)
    monkeypatch.setattr(task_pacing, "wrapup_reservation_fits", fits_budget)
    ctx = _limit_ctx(registry, trace, messages, logs)
    kwargs = {"prompt": "[ROUND_LIMIT] Wrap up", "fallback_text": "retained", "reason_code": "round_limit",
              "_prompt_prepared": True}
    if admitted:
        kwargs.update(_admitted_request=object(), _initial_messages=list(messages), _admission=balances)
    try:
        if fits:
            with pytest.raises(InputReached):
                loop._forced_final_answer(ctx, **kwargs)
        else:
            loop._forced_final_answer(ctx, **kwargs)
    finally:
        slow.release.set()
    assert not _locators(sent[0])
    if fits:
        assert len(sent) == 2 and len(_locators(sent[1])) == 1
    else:
        assert len(sent) == 1 and trace["forced_finalization"]["source"] == "budget_wrapup_unaffordable"
    if admitted:
        assert len(priced) == len(checked) == 1 and len(_locators(priced[0])) == 1
        assert checked[0] == {"request": request, **balances}
        if fits:
            assert sent[1] == priced[0]
    assert len(_locators(ctx.messages)) == 1 and slow.calls == 1
