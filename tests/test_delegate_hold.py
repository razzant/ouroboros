"""Incident-shaped contracts for the unknown-provider hold (nanny-leaf D1-min).

A configured-session nanny whose metered round dies ``provider_outcome_unknown``
while EXACTLY one delegated leaf is alive must hold on the LEAF (zero provider
calls) and resume with a wake-bearing NEW round — the unknown request is never
resent. Control wakes and every ineligible shape keep today's no-resend
terminal, and the terminal cleanup (leaf cancellation) fires only on terminals.
"""

from __future__ import annotations

import json
import queue
import time
from types import SimpleNamespace

import pytest

import ouroboros.delegate_hold as delegate_hold
import ouroboros.loop as loop_mod
from ouroboros import delegate_custody as custody
from ouroboros.delegate_supervision import read_unknown_hold, write_unknown_hold
from ouroboros.loop import run_llm_loop
from ouroboros.tools.registry import ToolRegistry


def _read_hold_events(tmp_path):
    path = tmp_path / "events.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [row for row in rows if row.get("type") == "delegate_hold"]


def _configured_registry(tmp_path, task_id="t-hold"):
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = task_id
    registry._ctx.exact_model_route = True
    registry._ctx.task_metadata = {"configured_subagent": {"config_fingerprint": "fp"}}
    return registry


def _start_leaf(tmp_path, task_id="t-hold", run_id="run-leaf"):
    custody._CUSTODY.pop(run_id, None)
    row = custody.RunCustody(run_id=run_id, task_id=task_id, route_id="claude", model="m")
    assert custody.record_started(tmp_path, row)
    return run_id


@pytest.fixture(autouse=True)
def _quiet_probe(monkeypatch):
    """Default leaf probe: read-only poll sees a live engine state; releases are
    recorded, not executed."""
    import ouroboros.claudexor_daemon as daemon_mod
    import ouroboros.delegate_progress as progress_mod

    monkeypatch.setattr(daemon_mod, "ensure_owned_gateway",
                        lambda **_k: SimpleNamespace(close=lambda: None), raising=False)
    monkeypatch.setattr(
        progress_mod, "bounded_poll",
        lambda _gw, _run, _sec, **_k: {"summary": {"state": "running"}, "lastSeq": 1},
    )
    released = []
    monkeypatch.setattr(custody, "release_task_runs", lambda root, tid: released.append(tid))
    yield released


def _loop_kwargs(tmp_path, registry, notes):
    return dict(
        messages=[{"role": "user", "content": "supervise"}],
        tools=registry,
        llm=SimpleNamespace(default_model=lambda: "test-model"),
        drive_logs=tmp_path,
        emit_progress=lambda text, *, incident=None: notes.append(text),
        incoming_messages=queue.Queue(),
        task_id=str(registry._ctx.task_id),
        drive_root=tmp_path,
    )


def _unknown_then_check_call(check):
    calls = {"n": 0}

    def fake_call(_llm, messages, _model, _tools, _effort, _max_retries, _drive_logs,
                  _task_id, _round_idx, _event_queue, accumulated_usage, *_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1:
            accumulated_usage["_last_llm_error_kind"] = "provider_outcome_unknown"
            accumulated_usage.update(execution_status="infra_failed", reason_code="llm_api_error")
            return None, 0.0
        return check(messages, accumulated_usage)

    return fake_call, calls


def test_unknown_with_live_leaf_holds_and_resumes_with_wake(tmp_path, monkeypatch, _quiet_probe):
    wake_payload = {"status": "succeeded", "run_id": "run-leaf", "supervision_wake_id": "w1"}
    monkeypatch.setattr(delegate_hold, "supervised_wait",
                        lambda _ctx, _run: json.dumps(wake_payload))
    acks = []
    monkeypatch.setattr(delegate_hold, "acknowledge_pending_wake",
                        lambda _ctx, delivered=None: acks.append(delivered) or True)

    def check(messages, accumulated_usage):
        assert "[DELEGATED LEAF WAKE / UNKNOWN-HOLD RESUME]" in messages[-1]["content"]
        assert "run-leaf" in messages[-1]["content"]
        accumulated_usage.pop("_last_llm_error_kind", None)
        return {"role": "assistant", "content": "integrated"}, 0.0

    fake_call, calls = _unknown_then_check_call(check)
    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    registry = _configured_registry(tmp_path)
    _start_leaf(tmp_path)
    notes = []
    result, usage, _trace = run_llm_loop(**_loop_kwargs(tmp_path, registry, notes))

    assert result == "integrated"
    assert usage.get("reason_code") != "provider_unavailable"
    assert calls["n"] == 2  # the unknown request itself was never resent
    phases = [(row["phase"], row.get("detail", "")) for row in _read_hold_events(tmp_path)]
    assert ("entered", "") in phases
    assert any(p == "resumed" for p, _d in phases)
    assert acks and acks[0]["supervision_wake_id"] == "w1"
    assert not read_unknown_hold(registry._ctx).get("run_id")  # inactive tombstone
    assert _quiet_probe == ["t-hold"]  # release only at the (successful) terminal
    assert any("holding on the leaf" in note for note in notes)


def test_terminal_leaf_never_enters_hold(tmp_path, monkeypatch, _quiet_probe):
    import ouroboros.delegate_progress as progress_mod

    monkeypatch.setattr(
        progress_mod, "bounded_poll",
        lambda _gw, _run, _sec, **_k: {"summary": {"state": "succeeded"}},
    )
    monkeypatch.setattr(delegate_hold, "supervised_wait",
                        lambda *_a, **_k: pytest.fail("terminal leaf must not hold"))
    fake_call, calls = _unknown_then_check_call(lambda *_: pytest.fail("no second dial"))
    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    registry = _configured_registry(tmp_path)
    _start_leaf(tmp_path)
    notes = []
    _r, usage, trace = run_llm_loop(**_loop_kwargs(tmp_path, registry, notes))

    assert calls["n"] == 1
    assert usage.get("execution_status") == "infra_failed"
    assert trace.get("forced_finalization", {}).get("source") == "provider_outcome_unknown_no_resend"
    assert _read_hold_events(tmp_path) == []


def test_control_wake_exits_through_no_call_terminal(tmp_path, monkeypatch, _quiet_probe):
    monkeypatch.setattr(
        delegate_hold, "supervised_wait",
        lambda _ctx, _run: json.dumps({
            "status": "progress",
            "wake_events": [{"type": "cancellation_intent"}],
            "supervision_wake_id": "w2",
        }),
    )
    fake_call, calls = _unknown_then_check_call(lambda *_: pytest.fail("no dial after Stop"))
    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    registry = _configured_registry(tmp_path)
    _start_leaf(tmp_path)
    notes = []
    _r, usage, trace = run_llm_loop(**_loop_kwargs(tmp_path, registry, notes))

    assert calls["n"] == 1  # zero provider calls after the control wake
    assert usage.get("execution_status") == "infra_failed"
    assert trace.get("forced_finalization", {}).get("source") == "provider_outcome_unknown_no_resend"
    details = [row.get("detail") for row in _read_hold_events(tmp_path) if row["phase"] == "ended"]
    assert "control_wake" in details
    assert _quiet_probe == ["t-hold"]  # the terminal cleanup owns the leaf now


def test_finalize_now_mid_hold_takes_no_call_terminal(tmp_path, monkeypatch, _quiet_probe):
    from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, write_owner_message

    def waiting_forever(_ctx, _run):
        time.sleep(30)
        pytest.fail("supervised_wait should have been pre-empted by finalize_now")

    # finalize_now lands BETWEEN the failing round and the next round top, so
    # the latched hold sees it in controls before any wait starts.
    monkeypatch.setattr(delegate_hold, "supervised_wait", waiting_forever)

    def check(_messages, _usage):
        pytest.fail("no dial")

    fake_call, calls = _unknown_then_check_call(check)
    orig_fake = fake_call

    def fake_with_mailbox(*args, **kwargs):
        out = orig_fake(*args, **kwargs)
        if calls["n"] == 1:
            write_owner_message(tmp_path, "wrap up", "t-hold", kind=KIND_FINALIZE_NOW)
        return out

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_with_mailbox)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    registry = _configured_registry(tmp_path)
    _start_leaf(tmp_path)
    notes = []
    _r, usage, trace = run_llm_loop(**_loop_kwargs(tmp_path, registry, notes))

    assert calls["n"] == 1
    assert trace.get("forced_finalization", {}).get("source") == "provider_outcome_unknown_no_resend"
    details = [row.get("detail") for row in _read_hold_events(tmp_path) if row["phase"] == "ended"]
    assert "finalize_now" in details


def test_generic_task_and_multi_run_never_hold(tmp_path, monkeypatch, _quiet_probe):
    monkeypatch.setattr(delegate_hold, "supervised_wait",
                        lambda *_a, **_k: pytest.fail("ineligible shapes must not hold"))
    fake_call, calls = _unknown_then_check_call(lambda *_: pytest.fail("no second dial"))
    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)

    # Generic task: no configured_subagent snapshot.
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = "t-generic"
    _start_leaf(tmp_path, task_id="t-generic", run_id="run-g")
    _r, usage, _t = run_llm_loop(**_loop_kwargs(tmp_path, registry, []))
    assert calls["n"] == 1 and usage.get("execution_status") == "infra_failed"

    # Configured but TWO live leaves.
    calls["n"] = 0
    registry2 = _configured_registry(tmp_path, task_id="t-multi")
    _start_leaf(tmp_path, task_id="t-multi", run_id="run-m1")
    _start_leaf(tmp_path, task_id="t-multi", run_id="run-m2")
    _r, usage2, _t = run_llm_loop(**_loop_kwargs(tmp_path, registry2, []))
    assert calls["n"] == 1 and usage2.get("execution_status") == "infra_failed"
    assert _read_hold_events(tmp_path) == []


def test_recovered_latch_reenters_hold_before_any_dispatch(tmp_path, monkeypatch, _quiet_probe):
    """The durable latch (worker-crash adoption) parks the successor's FIRST
    round in the hold before any LLM call — sol #5 contract."""
    wake_payload = {"status": "attention", "run_id": "run-leaf", "supervision_wake_id": "w3"}
    order = []
    monkeypatch.setattr(
        delegate_hold, "supervised_wait",
        lambda _ctx, _run: order.append("wait") or json.dumps(wake_payload),
    )
    monkeypatch.setattr(delegate_hold, "acknowledge_pending_wake", lambda *_a, **_k: True)

    def fake_call(_llm, messages, *_a, **_k):
        order.append("dispatch")
        assert "[DELEGATED LEAF WAKE / UNKNOWN-HOLD RESUME]" in messages[-1]["content"]
        return {"role": "assistant", "content": "resumed"}, 0.0

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    registry = _configured_registry(tmp_path)
    _start_leaf(tmp_path)
    write_unknown_hold(registry._ctx, "run-leaf", {
        "run_id": "run-leaf", "entered_at": "2026-08-30T00:00:00Z", "hold_cycles": 1,
    })
    result, _u, _t = run_llm_loop(**_loop_kwargs(tmp_path, registry, []))
    assert result == "resumed"
    assert order == ["wait", "dispatch"]


def test_repeated_unknown_reholds_with_backoff_floor(tmp_path, monkeypatch, _quiet_probe):
    wake_payload = {"status": "progress_report", "run_id": "run-leaf", "supervision_wake_id": "w4"}
    monkeypatch.setattr(delegate_hold, "supervised_wait",
                        lambda _ctx, _run: json.dumps(wake_payload))
    monkeypatch.setattr(delegate_hold, "acknowledge_pending_wake", lambda *_a, **_k: True)
    sleeps = []
    monkeypatch.setattr(delegate_hold.time, "sleep", lambda sec: sleeps.append(sec))
    calls = {"n": 0}

    def fake_call(_llm, _messages, _model, _tools, _effort, _max_retries, _drive_logs,
                  _task_id, _round_idx, _event_queue, accumulated_usage, *_a, **_k):
        calls["n"] += 1
        if calls["n"] <= 2:
            accumulated_usage["_last_llm_error_kind"] = "provider_outcome_unknown"
            accumulated_usage.update(execution_status="infra_failed", reason_code="llm_api_error")
            return None, 0.0
        accumulated_usage.pop("_last_llm_error_kind", None)
        return {"role": "assistant", "content": "done"}, 0.0

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    registry = _configured_registry(tmp_path)
    _start_leaf(tmp_path)
    result, _u, _t = run_llm_loop(**_loop_kwargs(tmp_path, registry, []))

    assert result == "done"
    assert calls["n"] == 3
    cycle_counts = [row["hold_cycles"] for row in _read_hold_events(tmp_path)
                    if row["phase"] == "entered"]
    assert cycle_counts == [1, 2]
    # The patch is on the GLOBAL time.sleep, so a daemon thread leaked by an earlier test on the
    # same xdist worker (a 0.5 s poll loop; windows-latest, rc.13 dispatch) lands in ``sleeps``
    # too: pin the backoff floor of the second cycle by presence, not by position.
    assert any(4.0 <= sec <= 15.0 for sec in sleeps), sleeps  # backoff floor on the second cycle


def test_refused_probe_and_state_less_payload_never_hold(tmp_path, monkeypatch, _quiet_probe):
    """A daemon refusal or a state-less payload is not evidence of a live leaf
    (grok #1/#2, fable F3): the probe fails closed to today's terminal."""
    import ouroboros.delegate_progress as progress_mod

    monkeypatch.setattr(delegate_hold, "supervised_wait",
                        lambda *_a, **_k: pytest.fail("refused probe must not hold"))
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)

    def raising_poll(_gw, _run, _sec, **_k):
        raise RuntimeError("daemon unreachable")

    monkeypatch.setattr(progress_mod, "bounded_poll", raising_poll)
    fake_call, calls = _unknown_then_check_call(lambda *_: pytest.fail("no second dial"))
    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)
    registry = _configured_registry(tmp_path, task_id="t-refused")
    _start_leaf(tmp_path, task_id="t-refused", run_id="run-r1")
    _r, usage, _t = run_llm_loop(**_loop_kwargs(tmp_path, registry, []))
    assert calls["n"] == 1 and usage.get("execution_status") == "infra_failed"

    monkeypatch.setattr(progress_mod, "bounded_poll", lambda _gw, _run, _sec, **_k: {})
    calls["n"] = 0
    registry2 = _configured_registry(tmp_path, task_id="t-stateless")
    _start_leaf(tmp_path, task_id="t-stateless", run_id="run-r2")
    _r, usage2, _t = run_llm_loop(**_loop_kwargs(tmp_path, registry2, []))
    assert calls["n"] == 1 and usage2.get("execution_status") == "infra_failed"
    assert _read_hold_events(tmp_path) == []


def test_refused_wait_takes_terminal_not_paid_resume(tmp_path, monkeypatch, _quiet_probe):
    """A refused/fault wait status is a daemon statement, not a leaf wake
    (fable F3): no paid resume round is bought on it."""
    monkeypatch.setattr(delegate_hold, "supervised_wait",
                        lambda _ctx, _run: json.dumps({"status": "refused"}))
    fake_call, calls = _unknown_then_check_call(lambda *_: pytest.fail("no dial on refusal"))
    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    registry = _configured_registry(tmp_path)
    _start_leaf(tmp_path)
    _r, usage, trace = run_llm_loop(**_loop_kwargs(tmp_path, registry, []))

    assert calls["n"] == 1
    assert trace.get("forced_finalization", {}).get("source") == "provider_outcome_unknown_no_resend"
    details = [row.get("detail") for row in _read_hold_events(tmp_path) if row["phase"] == "ended"]
    assert "wait_refused" in details


def test_ack_failure_fails_closed_without_dispatch(tmp_path, monkeypatch, _quiet_probe):
    """One wake = one dispatch (sol CRITICAL #1): a wake that cannot be durably
    acknowledged is never dispatched — the appended receipt is removed and the
    task takes the honest no-resend terminal."""
    wake_payload = {"status": "succeeded", "run_id": "run-leaf", "supervision_wake_id": "w5"}
    monkeypatch.setattr(delegate_hold, "supervised_wait",
                        lambda _ctx, _run: json.dumps(wake_payload))
    monkeypatch.setattr(delegate_hold, "acknowledge_pending_wake", lambda *_a, **_k: False)
    monkeypatch.setattr(delegate_hold.time, "sleep", lambda _s: None)
    seen_messages = []

    def fake_call(_llm, messages, _model, _tools, _effort, _max_retries, _drive_logs,
                  _task_id, _round_idx, _event_queue, accumulated_usage, *_a, **_k):
        seen_messages.append(list(messages))
        accumulated_usage["_last_llm_error_kind"] = "provider_outcome_unknown"
        accumulated_usage.update(execution_status="infra_failed", reason_code="llm_api_error")
        return None, 0.0

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    registry = _configured_registry(tmp_path)
    _start_leaf(tmp_path)
    _r, _u, trace = run_llm_loop(**_loop_kwargs(tmp_path, registry, []))

    assert len(seen_messages) == 1  # only the original unknown round dialed
    assert trace.get("forced_finalization", {}).get("source") == "provider_outcome_unknown_no_resend"
    details = [row.get("detail") for row in _read_hold_events(tmp_path) if row["phase"] == "ended"]
    assert "ack_failed" in details


def test_owner_input_resumes_without_wait(tmp_path, monkeypatch, _quiet_probe):
    """An owner message drained at the round top IS material new input (sol
    HIGH #4): the hold resumes on it without entering supervised_wait."""
    monkeypatch.setattr(delegate_hold, "supervised_wait",
                        lambda *_a, **_k: pytest.fail("owner input must resume without waiting"))

    def fake_call(_llm, messages, *_a, **_k):
        assert any("please integrate" in str(m.get("content")) for m in messages)
        return {"role": "assistant", "content": "resumed-on-owner-input"}, 0.0

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    registry = _configured_registry(tmp_path)
    _start_leaf(tmp_path)
    write_unknown_hold(registry._ctx, "run-leaf", {
        "run_id": "run-leaf", "entered_at": "2026-08-30T00:00:00Z", "hold_cycles": 1,
    })
    kwargs = _loop_kwargs(tmp_path, registry, [])
    # The transcript tail is LONGER than the owner message: an appended short
    # message must still read as new input (final-pair fable F1 — a naive
    # length-sum signature missed exactly this common shape).
    kwargs["messages"] = [
        {"role": "user", "content": "supervise"},
        {"role": "assistant", "content": "long tool result " * 50},
    ]
    kwargs["incoming_messages"].put("please integrate")
    result, _u, _t = run_llm_loop(**kwargs)
    assert result == "resumed-on-owner-input"
    details = [row.get("detail") for row in _read_hold_events(tmp_path) if row["phase"] == "resumed"]
    assert "owner_input" in details


def test_recovered_latch_control_wake_stays_no_call(tmp_path, monkeypatch, _quiet_probe):
    """Fable F1: after crash recovery the usage record is fresh — a control
    wake must still exit through the no-call unknown terminal, never a paid
    [PROVIDER_UNAVAILABLE] forced final."""
    monkeypatch.setattr(
        delegate_hold, "supervised_wait",
        lambda _ctx, _run: json.dumps({
            "status": "progress", "wake_events": [{"type": "cancellation_intent"}],
        }),
    )
    monkeypatch.setattr(loop_mod, "call_llm_with_retry",
                        lambda *_a, **_k: pytest.fail("Stop after recovery must not dial"))
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    registry = _configured_registry(tmp_path)
    _start_leaf(tmp_path)
    write_unknown_hold(registry._ctx, "run-leaf", {
        "run_id": "run-leaf", "entered_at": "2026-08-30T00:00:00Z", "hold_cycles": 1,
    })
    _r, _u, trace = run_llm_loop(**_loop_kwargs(tmp_path, registry, []))
    assert trace.get("forced_finalization", {}).get("source") == "provider_outcome_unknown_no_resend"


def test_round_limit_with_live_hold_takes_no_call_terminal(tmp_path, monkeypatch, _quiet_probe):
    """Sol CRITICAL #2 / fable F2: an unknown on the last legal round must not
    buy a paid [ROUND_LIMIT] dial — the hold closes into the no-call unknown
    terminal and the latch does not dangle."""
    monkeypatch.setattr(delegate_hold, "supervised_wait",
                        lambda *_a, **_k: pytest.fail("round-limit hold must not wait"))
    calls = {"n": 0}

    def fake_call(_llm, _messages, _model, _tools, _effort, _max_retries, _drive_logs,
                  _task_id, _round_idx, _event_queue, accumulated_usage, *_a, **_k):
        calls["n"] += 1
        accumulated_usage["_last_llm_error_kind"] = "provider_outcome_unknown"
        accumulated_usage.update(execution_status="infra_failed", reason_code="llm_api_error")
        return None, 0.0

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "1")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    registry = _configured_registry(tmp_path)
    _start_leaf(tmp_path)
    _r, _u, trace = run_llm_loop(**_loop_kwargs(tmp_path, registry, []))

    assert calls["n"] == 1  # the [ROUND_LIMIT] wrap-up never dialed
    assert trace.get("forced_finalization", {}).get("source") == "provider_outcome_unknown_no_resend"
    details = [row.get("detail") for row in _read_hold_events(tmp_path) if row["phase"] == "ended"]
    assert "round_limit" in details
    assert not read_unknown_hold(registry._ctx)  # no stale latch left behind


def test_latch_survives_real_supervised_wait_state_reset(tmp_path, monkeypatch, _quiet_probe):
    """Final-pair CRITICAL (sol #1 / fable F2): the REAL supervised_wait's
    _load_state rebuild for a new run id must carry the durable latch — a
    worker crash mid-wait must find it, or the successor resends."""
    import ouroboros.delegate_supervision as sup

    registry = _configured_registry(tmp_path, task_id="t-reset")
    _start_leaf(tmp_path, task_id="t-reset", run_id="run-reset")
    write_unknown_hold(registry._ctx, "run-reset", {
        "run_id": "run-reset", "entered_at": "2026-08-30T00:00:00Z", "hold_cycles": 1,
    })

    def wait_once(_ctx, _run, _sec, _seq):
        # Mid-wait crash shape: the durable file must STILL carry the latch
        # after supervised_wait's entry persisted its (rebuilt) state.
        data = json.loads(sup._state_path(registry._ctx).read_text())
        assert data.get("unknown_provider_hold", {}).get("run_id") == "run-reset"
        return json.dumps({"status": "succeeded", "run_id": "run-reset", "last_seq": 2})

    raw = sup.supervised_wait(registry._ctx, "run-reset", wait_once=wait_once)
    assert json.loads(raw).get("status") == "succeeded"
    assert read_unknown_hold(registry._ctx).get("run_id") == "run-reset"


def test_unreadable_latch_fails_closed_to_terminal(tmp_path, monkeypatch, _quiet_probe):
    """Final-pair sol #2: an existing-but-corrupt latch file must not read as
    'no hold' — dispatching there could resend the unknown request."""
    import ouroboros.delegate_supervision as sup

    registry = _configured_registry(tmp_path, task_id="t-corrupt")
    _start_leaf(tmp_path, task_id="t-corrupt", run_id="run-c")
    path = sup._state_path(registry._ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{corrupt json", encoding="utf-8")
    monkeypatch.setattr(delegate_hold, "supervised_wait",
                        lambda *_a, **_k: pytest.fail("corrupt latch must not wait"))
    monkeypatch.setattr(loop_mod, "call_llm_with_retry",
                        lambda *_a, **_k: pytest.fail("corrupt latch must not dispatch"))
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    _r, _u, trace = run_llm_loop(**_loop_kwargs(tmp_path, registry, []))
    assert trace.get("forced_finalization", {}).get("source") == "provider_outcome_unknown_no_resend"
    details = [row.get("detail") for row in _read_hold_events(tmp_path) if row["phase"] == "ended"]
    assert "latch_unreadable" in details


def test_eligibility_probe_closes_its_gateway(tmp_path, monkeypatch, _quiet_probe):
    """Final-pair sol #5 / fable F3: the probe owns close() on the gateway."""
    import ouroboros.claudexor_daemon as daemon_mod

    closed = []

    class _Gw:
        def close(self):
            closed.append(True)

    monkeypatch.setattr(daemon_mod, "ensure_owned_gateway",
                        lambda **_k: _Gw(), raising=False)
    registry = _configured_registry(tmp_path, task_id="t-gw")
    _start_leaf(tmp_path, task_id="t-gw", run_id="run-gw")
    assert delegate_hold._single_live_run(registry._ctx) == "run-gw"
    assert closed == [True]
