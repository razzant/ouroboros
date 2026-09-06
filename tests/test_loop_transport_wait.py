"""Contracts for the remote transport-outage wait episode (net-resilience sprint).

Covers: typed classification of released pre-dispatch transport failures
(remote vs local provider), the one-physical-attempt-per-call contract, the
round-level wait episode (free redials, recovery, deterministic no-resend
terminal, the interactive turns' idle-timeout bound, local-only fallback pass),
the owner-signal-interruptible sleep, and durable ``network_wait`` evidence.
Interactive-episode contracts continue in ``test_loop_transport_wait_interactive.py``.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest

import ouroboros.loop as loop_mod
import ouroboros.loop_transport as loop_transport
from ouroboros import usage_accounting as ua
from ouroboros.loop import run_llm_loop
from ouroboros.loop_llm_call import call_llm_with_retry, classify_llm_exception
from ouroboros.tools.registry import ToolRegistry


def _released_capture(provider: str = "openrouter", state: str = "released") -> ua.PhysicalAttemptCapture:
    return ua.PhysicalAttemptCapture(
        attempt_id="pa-test", model="test-model", provider=provider, state=state,
        candidate_measurement_kind="opaque",
    )


def _typed_transport_exc(provider: str = "openrouter", exc_cls=httpx.ConnectError, state: str = "released"):
    """A provider-wrapper exception with typed transport provenance + custody capture."""
    cause = exc_cls("connection failed")
    try:
        raise RuntimeError("Connection error.") from cause
    except RuntimeError as exc:
        exc.physical_attempt_capture = _released_capture(provider=provider, state=state)
        return exc


def _read_network_wait_events(tmp_path):
    path = tmp_path / "events.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [row for row in rows if row.get("type") == "network_wait"]


# --------------------------------------------------------------- classification

def test_released_remote_transport_failure_classifies_transport_unavailable():
    result = classify_llm_exception(_typed_transport_exc())
    assert result.kind == "transport_unavailable"
    assert result.retry_same_request is True


def test_released_connect_timeout_now_classifies_transport_unavailable():
    """Contract flip pinned explicitly: a RELEASED typed ConnectTimeout used to
    ride the generic "timeout" marker into provider_transient; the typed custody
    fact now wins."""
    exc = httpx.ConnectTimeout("connect timed out")
    exc.physical_attempt_capture = _released_capture()
    assert classify_llm_exception(exc).kind == "transport_unavailable"


def test_released_local_provider_failure_stays_generic():
    """A stopped LOCAL model server is not a network outage worth waiting out."""
    result = classify_llm_exception(_typed_transport_exc(provider="local"))
    assert result.kind != "transport_unavailable"


def test_dispatched_capture_stays_provider_outcome_unknown():
    result = classify_llm_exception(_typed_transport_exc(state="dispatched"))
    assert result.kind == "provider_outcome_unknown"
    assert result.retry_same_request is False


def test_released_capture_without_typed_transport_cause_stays_generic():
    """Preparation failures release custody too but carry no typed transport fact."""
    try:
        raise RuntimeError("payload preparation failed")
    except RuntimeError as exc:
        exc.physical_attempt_capture = _released_capture()
        assert classify_llm_exception(exc).kind != "transport_unavailable"


def test_connect_timeout_without_capture_keeps_legacy_transient_path():
    assert classify_llm_exception(httpx.ConnectTimeout("connect timed out")).kind == "provider_transient"


def test_transport_unavailable_is_not_a_transient_or_cooldown_kind():
    """§4e-1 negative membership pin: the kind lives in NO retry frozenset —
    no in-helper burst (_TRANSIENT_RETRY_KINDS) and no model cooldown
    (_COOLDOWN_ERROR_KINDS: the egress is shared, cooling a model is
    meaningless and would poison the fallback chain after recovery)."""
    from ouroboros.loop_llm_call import _COOLDOWN_ERROR_KINDS, _TRANSIENT_RETRY_KINDS

    assert "transport_unavailable" not in _TRANSIENT_RETRY_KINDS
    assert "transport_unavailable" not in _COOLDOWN_ERROR_KINDS


def test_review_actor_keeps_its_bounded_retry_contract():
    """Review custody consults the same classifier: released transport stays
    retryable (its existing 1-2 physical-send cap), and never inherits the
    round-level wait."""
    from ouroboros.review_custody import retryable_review_exception

    assert retryable_review_exception(_typed_transport_exc(), None) is True


# ------------------------------------------------ one physical attempt per call

class _RaisingLLM:
    def __init__(self, exc_factory):
        self.exc_factory = exc_factory
        self.calls = 0

    def chat(self, **_kwargs):
        self.calls += 1
        raise self.exc_factory()


def test_call_llm_with_retry_makes_exactly_one_attempt_on_transport_unavailable(tmp_path):
    llm = _RaisingLLM(_typed_transport_exc)
    usage = {}
    msg, _cost = call_llm_with_retry(
        llm, [{"role": "user", "content": "hi"}], "test-model", None, "low", 3,
        tmp_path, "t-one", 1, None, usage,
    )
    assert msg is None
    assert llm.calls == 1
    assert usage.get("_last_llm_error_kind") == "transport_unavailable"


def test_call_llm_with_retry_keeps_bounded_burst_for_local_released_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    llm = _RaisingLLM(lambda: _typed_transport_exc(provider="local"))
    usage = {}
    msg, _cost = call_llm_with_retry(
        llm, [{"role": "user", "content": "hi"}], "test-model", None, "low", 3,
        tmp_path, "t-local", 1, None, usage,
    )
    assert msg is None
    assert llm.calls > 1  # the legacy bounded burst, not the one-attempt return


# ----------------------------------------------------------- round-episode loop

class _FakeLoopLLM:
    def default_model(self):
        return "test-model"


def _loop_kwargs(tmp_path, registry, notes, llm=None):
    return dict(
        messages=[{"role": "user", "content": "go"}],
        tools=registry,
        llm=llm or _FakeLoopLLM(),
        drive_logs=tmp_path,
        # emit_progress honors the ``incident=`` keyword (OuroborosAgent._emit_progress).
        emit_progress=lambda text, *, incident=None: notes.append(text),
        incoming_messages=queue.Queue(),
        task_id="t-wait",
        drive_root=tmp_path,
    )


class _FakeClock:
    """loop_transport-local monotonic clock that only faked sleeps advance, so
    a bound measured from episode entry is exercised deterministically; the
    wall clock (owner deadlines) stays real."""

    def __init__(self, monkeypatch, start: float = 1000.0):
        self.now = start
        self.sleeps: list = []
        monkeypatch.setattr(
            loop_transport, "time", SimpleNamespace(monotonic=lambda: self.now, time=time.time),
        )
        monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", self.sleep)

    def sleep(self, sec, _wake):
        self.sleeps.append(sec)
        self.now += sec
        return False


def _transport_failing_call(fail_times: int, final_content: str = "done"):
    calls = {"n": 0, "routes": []}

    def fake_call(_llm, _messages, model, _tools, _effort, _max_retries, _drive_logs,
                  _task_id, _round_idx, _event_queue, accumulated_usage, *_a, **kwargs):
        calls["n"] += 1
        calls["routes"].append((model, bool(kwargs.get("use_local"))))
        if calls["n"] <= fail_times:
            accumulated_usage["_last_llm_error_kind"] = "transport_unavailable"
            accumulated_usage["_last_llm_error"] = "Connection error."
            return None, 0.0
        accumulated_usage.pop("_last_llm_error_kind", None)
        return {"role": "assistant", "content": final_content}, 0.0

    return fake_call, calls


def test_transport_outage_waits_redials_free_rounds_and_recovers(tmp_path, monkeypatch):
    fake_call, calls = _transport_failing_call(fail_times=3)
    sleeps = []
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep",
                        lambda sec, _wake: (sleeps.append(sec), False)[1])
    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "1")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    notes = []
    result, usage, _trace = run_llm_loop(**_loop_kwargs(tmp_path, ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path), notes))

    # Recovery, not a round-limit forced finalization: the 3 redials were FREE
    # rounds (MAX_ROUNDS=1), and the recovered dispatch finished the task.
    assert result == "done"
    assert usage.get("reason_code") is None
    assert calls["n"] == 4
    assert len(sleeps) == 3
    assert all(0.0 < sec <= 60.0 for sec in sleeps)
    assert sleeps == sorted(sleeps)  # backoff grows monotonically toward the cap

    phases = [row["phase"] for row in _read_network_wait_events(tmp_path)]
    assert phases[0] == "entered"
    assert phases.count("waiting") == 3
    assert phases[-1] == "recovered"
    assert any("provider connection" in note for note in notes)  # first note, immediate
    assert any("restored" in note for note in notes)  # mandatory recovery note


@pytest.mark.parametrize("flag", ["is_direct_chat", "is_ephemeral_turn"])
def test_interactive_turns_wait_redial_free_and_terminalize_at_the_idle_bound(tmp_path, monkeypatch, flag):
    """Direct-chat and ephemeral decision turns are wait-eligible: they redial
    for free until the RAW configured task idle timeout — their only rail, as
    they carry no deadline and no queue rails — is spent, then take the
    deterministic no-resend terminal whose detail names that bound and whose
    wording calls it "this turn", never a task."""
    fake_call, calls = _transport_failing_call(fail_times=99)
    clock = _FakeClock(monkeypatch)
    monkeypatch.setattr(loop_transport, "get_task_idle_timeout_sec", lambda: 120)
    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    setattr(registry._ctx, flag, True)
    notes = []
    result, usage, trace = run_llm_loop(**_loop_kwargs(tmp_path, registry, notes))

    assert calls["n"] >= 3  # free redials, not one honest attempt
    assert calls["n"] == len(clock.sleeps) + 1  # every dispatch after the first followed a wait
    assert 0.0 < sum(clock.sleeps) <= 120.0  # the raw idle bound, not the queue's effective rail
    assert usage.get("execution_status") == "infra_failed"
    assert usage.get("reason_code") == "provider_unavailable"
    assert trace.get("forced_finalization", {}).get("source") == "transport_unavailable_no_resend"
    assert "this turn waited and redialed for" in result
    assert "the task" not in result
    assert "fails fast" not in result
    events = _read_network_wait_events(tmp_path)
    assert events[0]["phase"] == "entered"
    assert events[-1]["phase"] == "ended"
    assert events[-1]["detail"] == "interactive_wait_window_exhausted"
    assert notes and all("Stop cancels" not in note for note in notes)


def test_deadline_bounds_wait_with_one_last_free_redial_then_no_resend(tmp_path, monkeypatch):
    fake_call, calls = _transport_failing_call(fail_times=99)
    sleeps = []
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep",
                        lambda sec, _wake: (sleeps.append(sec), False)[1])
    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)

    def _chain_must_not_run(**_kwargs):
        raise AssertionError("remote fallback chain must not dial during a transport outage")

    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", _chain_must_not_run)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "other/model")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    from ouroboros.config import get_finalization_grace_sec

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    deadline = datetime.now(timezone.utc) + timedelta(seconds=get_finalization_grace_sec() + 8)
    registry._ctx.task_metadata = {"deadline_at": deadline.isoformat()}
    notes = []
    result, usage, trace = run_llm_loop(**_loop_kwargs(tmp_path, registry, notes))

    # Bounded by the deadline: a couple of redials, the last one granted just
    # before the admission window closes, then the deterministic no-resend
    # terminal — never a forced-final provider call, never deadline_local.
    assert calls["n"] >= 2
    assert calls["n"] == len(sleeps) + 1  # every dispatch after the first followed a wait
    assert usage.get("execution_status") == "infra_failed"
    assert usage.get("reason_code") == "provider_unavailable"
    assert trace.get("forced_finalization", {}).get("source") == "transport_unavailable_no_resend"
    assert "waited and redialed" in result  # honest waited-out terminal text
    phases = [row["phase"] for row in _read_network_wait_events(tmp_path)]
    assert phases[-1] == "ended"


def test_deadline_refusal_during_episode_takes_transport_no_resend_terminal(tmp_path, monkeypatch):
    """The admission gate overwriting the mutable kind with deadline_exhausted
    must not fork the terminal story: the episode's latched cause wins (no
    [DEADLINE] forced provider call)."""
    calls = {"n": 0}

    def fake_call(_llm, _messages, _model, _tools, _effort, _max_retries, _drive_logs,
                  _task_id, _round_idx, _event_queue, accumulated_usage, *_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1:
            accumulated_usage["_last_llm_error_kind"] = "transport_unavailable"
        else:
            accumulated_usage["_last_llm_error_kind"] = "deadline_exhausted"
        return None, 0.0

    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", lambda _sec, _wake: False)
    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    notes = []
    _result, usage, trace = run_llm_loop(**_loop_kwargs(tmp_path, ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path), notes))

    assert calls["n"] == 2  # the refused redial is terminal; no third (forced) call
    assert usage.get("execution_status") == "infra_failed"
    assert usage.get("reason_code") == "provider_unavailable"
    assert trace.get("forced_finalization", {}).get("source") == "transport_unavailable_no_resend"


def test_scheduled_swarm_handoff_stays_truthful_on_transport_terminal(tmp_path, monkeypatch):
    """An ephemeral router turn waits out its idle bound on an outage, but the
    requested managed work was already durably admitted: the no-resend terminal
    stamp must not clobber the router's deliberate execution_status/reason_code
    clear — the successful handoff stays truthful."""
    calls = {"n": 0}
    _FakeClock(monkeypatch)
    monkeypatch.setattr(loop_transport, "get_task_idle_timeout_sec", lambda: 60)

    def fake_call(_llm, _messages, _model, _tools, _effort, _max_retries, _drive_logs,
                  _task_id, _round_idx, _event_queue, accumulated_usage, *_a, **_k):
        calls["n"] += 1
        accumulated_usage["_last_llm_error_kind"] = "transport_unavailable"
        accumulated_usage["_last_llm_error"] = "Connection error."
        # Mirror _record_llm_call_error's stamps so the test proves the router
        # pop clears them rather than them never having been set.
        accumulated_usage.update(execution_status="infra_failed", reason_code="llm_api_error")
        return None, 0.0

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.is_ephemeral_turn = True
    registry._ctx.task_metadata = {"force_plan": True, "force_plan_source": "swarm"}
    registry._ctx._swarm_handoff_attempt = {
        "task_id": "swarm-task-1",
        "routing_token": "route-token",
        "status": "scheduled",
        "reason": "",
        "response": "OK: task swarm-task-1 accepted and durably scheduled",
    }
    notes = []
    result, usage, _trace = run_llm_loop(**_loop_kwargs(tmp_path, registry, notes))

    assert "Swarm admitted managed task swarm-task-1" in result
    assert calls["n"] >= 2  # the ephemeral turn waited and redialed before the terminal
    assert usage.get("execution_status") is None
    assert usage.get("reason_code") is None


@pytest.mark.parametrize("turn_flag", [None, "is_direct_chat", "is_ephemeral_turn"])
def test_outage_first_observed_mid_chain_latches_episode_and_recovers(tmp_path, monkeypatch, turn_flag):
    """Primary fails generically (429-class), the chain walks, and a REMOTE
    candidate dies pre-dispatch: the post-chain reconcile must latch an episode
    from the FRESH kind — wait, redial, recover — instead of the generic
    terminal dialing a forced-final call over the proven-dead egress. The latch
    is turn-kind independent: a direct-chat or ephemeral turn waits and
    recovers the same way, and never remote-fallbacks over the dead egress."""
    calls = {"n": 0}

    def fake_call(_llm, _messages, _model, _tools, _effort, _max_retries, _drive_logs,
                  _task_id, _round_idx, _event_queue, accumulated_usage, *_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1:
            accumulated_usage["_last_llm_error_kind"] = "provider_transient"
            return None, 0.0
        accumulated_usage.pop("_last_llm_error_kind", None)
        return {"role": "assistant", "content": "done"}, 0.0

    chain_calls = {"n": 0}

    def chain_breaks_on_transport(**kwargs):
        chain_calls["n"] += 1
        kwargs["accumulated_usage"]["_last_llm_error_kind"] = "transport_unavailable"
        return (
            None, kwargs["active_model"], kwargs["active_use_local"],
            kwargs["context_fit_plan"], kwargs["active_context_mode"],
        )

    sleeps = []
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep",
                        lambda sec, _wake: (sleeps.append(sec), False)[1])
    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", chain_breaks_on_transport)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    if turn_flag:
        setattr(registry._ctx, turn_flag, True)
    notes = []
    result, usage, _trace = run_llm_loop(**_loop_kwargs(tmp_path, registry, notes))

    assert result == "done"
    assert chain_calls["n"] == 1
    assert calls["n"] == 2  # the redial went back to the primary and recovered
    assert len(sleeps) == 1
    assert usage.get("reason_code") is None
    phases = [row["phase"] for row in _read_network_wait_events(tmp_path)]
    assert phases[0] == "entered"
    assert phases[-1] == "recovered"


@pytest.mark.parametrize("turn_flag", [None, "is_direct_chat", "is_ephemeral_turn"])
def test_mid_chain_latch_that_never_recovers_takes_the_no_resend_terminal(tmp_path, monkeypatch, turn_flag):
    """The mid-chain latch drives the same terminal as a primary-first outage:
    free redials until the binding window (a managed task's deadline, an
    interactive turn's idle bound) closes, then the deterministic no-resend
    terminal — never a forced-final provider call over the dead egress."""
    calls = {"n": 0}

    def fake_call(_llm, _messages, _model, _tools, _effort, _max_retries, _drive_logs,
                  _task_id, _round_idx, _event_queue, accumulated_usage, *_a, **_k):
        calls["n"] += 1
        accumulated_usage["_last_llm_error_kind"] = (
            "provider_transient" if calls["n"] == 1 else "transport_unavailable"
        )
        return None, 0.0

    chain_calls = {"n": 0}

    def chain_breaks_on_transport(**kwargs):
        chain_calls["n"] += 1
        kwargs["accumulated_usage"]["_last_llm_error_kind"] = "transport_unavailable"
        return (
            None, kwargs["active_model"], kwargs["active_use_local"],
            kwargs["context_fit_plan"], kwargs["active_context_mode"],
        )

    clock = _FakeClock(monkeypatch)
    monkeypatch.setattr(loop_transport, "get_task_idle_timeout_sec", lambda: 60)
    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", chain_breaks_on_transport)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    if turn_flag:
        setattr(registry._ctx, turn_flag, True)
    else:
        from ouroboros.config import get_finalization_grace_sec

        deadline = datetime.now(timezone.utc) + timedelta(seconds=get_finalization_grace_sec() + 8)
        registry._ctx.task_metadata = {"deadline_at": deadline.isoformat()}
    notes = []
    _result, usage, trace = run_llm_loop(**_loop_kwargs(tmp_path, registry, notes))

    assert chain_calls["n"] == 1  # the chain never re-dials over the dead egress
    assert calls["n"] >= 2
    assert calls["n"] == len(clock.sleeps) + 1  # every dispatch after the first followed a wait
    assert usage.get("execution_status") == "infra_failed"
    assert usage.get("reason_code") == "provider_unavailable"
    assert trace.get("forced_finalization", {}).get("source") == "transport_unavailable_no_resend"
    events = _read_network_wait_events(tmp_path)
    assert events[0]["phase"] == "entered"
    assert events[-1]["phase"] == "ended"
    expected = "interactive_wait_window_exhausted" if turn_flag else "deadline_after_final_redial"
    assert events[-1]["detail"] == expected


def test_exact_model_route_waits_and_redials_its_own_pin(tmp_path, monkeypatch):
    """Q13: an exact route waits and redials its OWN pinned model — the chain
    never runs (even with a configured local fallback) and recovery adopts the
    pinned route's response."""
    fake_call, calls = _transport_failing_call(fail_times=2)
    sleeps = []
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep",
                        lambda sec, _wake: (sleeps.append(sec), False)[1])
    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)

    def _chain_must_not_run(**_kwargs):
        raise AssertionError("exact_model_route must never walk the fallback chain")

    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", _chain_must_not_run)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "other/model")
    monkeypatch.setenv("USE_LOCAL_FALLBACK", "true")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.exact_model_route = True
    notes = []
    result, _usage, _trace = run_llm_loop(**_loop_kwargs(tmp_path, registry, notes))

    assert result == "done"
    assert calls["n"] == 3
    assert len(sleeps) == 2
    assert all(0.0 < sec <= 60.0 for sec in sleeps)
    assert {route[0] for route in calls["routes"]} == {"test-model"}  # only the pin dialed


def test_local_fallback_pass_adopts_local_route_when_configured(tmp_path, monkeypatch):
    fake_call, calls = _transport_failing_call(fail_times=99)
    chain_calls = {"n": 0}

    def fake_chain(**kwargs):
        chain_calls["n"] += 1
        return (
            {"role": "assistant", "content": "local-ok"}, "local/candidate", True,
            kwargs["context_fit_plan"], kwargs["active_context_mode"],
        )

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", fake_chain)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "local/candidate")
    monkeypatch.setenv("USE_LOCAL_FALLBACK", "true")
    notes = []
    result, _usage, _trace = run_llm_loop(**_loop_kwargs(tmp_path, ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path), notes))

    assert result == "local-ok"
    assert chain_calls["n"] == 1  # Q4: the LOCAL chain dialed, exactly once
    events = _read_network_wait_events(tmp_path)
    assert [row["phase"] for row in events] == ["entered", "ended"]
    assert events[-1].get("detail") == "local_fallback_adopted"


def test_failed_local_pass_keeps_remote_cause_and_runs_at_most_once(tmp_path, monkeypatch):
    """A failed local walk overwrites the mutable error kind, but the latched
    remote cause keeps the episode waiting — and the local pass never re-dials."""
    fake_call, calls = _transport_failing_call(fail_times=3)
    chain_calls = {"n": 0}

    def failing_chain(**kwargs):
        chain_calls["n"] += 1
        kwargs["accumulated_usage"]["_last_llm_error_kind"] = "provider_error"
        return (
            None, kwargs["active_model"], kwargs["active_use_local"],
            kwargs["context_fit_plan"], kwargs["active_context_mode"],
        )

    sleeps = []
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep",
                        lambda sec, _wake: (sleeps.append(sec), False)[1])
    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", failing_chain)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "local/candidate")
    monkeypatch.setenv("USE_LOCAL_FALLBACK", "true")
    notes = []
    result, _usage, _trace = run_llm_loop(**_loop_kwargs(tmp_path, ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path), notes))

    assert result == "done"  # the episode kept waiting and recovered
    assert chain_calls["n"] == 1  # one local pass per episode
    assert calls["n"] == 4
    assert len(sleeps) == 3
    phases = [row["phase"] for row in _read_network_wait_events(tmp_path)]
    assert phases[-1] == "recovered"


def _unknown_outcome_chain():
    chain_calls = {"n": 0}

    def failing_chain(**kwargs):
        chain_calls["n"] += 1
        kwargs["accumulated_usage"]["_last_llm_error_kind"] = "provider_outcome_unknown"
        return (
            None, kwargs["active_model"], kwargs["active_use_local"],
            kwargs["context_fit_plan"], kwargs["active_context_mode"],
        )

    return failing_chain, chain_calls


def test_failed_local_pass_with_unknown_outcome_keeps_episode_waiting(tmp_path, monkeypatch):
    """Hardening for the refuted local-pass finding: a local pass that dies
    with provider_outcome_unknown overwrites the mutable kind, but the latched
    remote cause keeps the episode waiting until the egress recovers."""
    fake_call, calls = _transport_failing_call(fail_times=3)
    failing_chain, chain_calls = _unknown_outcome_chain()
    sleeps = []
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep",
                        lambda sec, _wake: (sleeps.append(sec), False)[1])
    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", failing_chain)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "local/candidate")
    monkeypatch.setenv("USE_LOCAL_FALLBACK", "true")
    notes = []
    result, _usage, _trace = run_llm_loop(**_loop_kwargs(tmp_path, ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path), notes))

    assert result == "done"  # the episode kept waiting and recovered
    assert chain_calls["n"] == 1  # one local pass per episode
    assert calls["n"] == 4
    assert len(sleeps) == 3
    phases = [row["phase"] for row in _read_network_wait_events(tmp_path)]
    assert phases[-1] == "recovered"


def test_unknown_outcome_local_pass_then_deadline_takes_transport_no_resend(tmp_path, monkeypatch):
    """Same shape, but the deadline expires while the egress stays dead: the
    terminal must key on the episode's latched cause — transport no-resend —
    not on the local pass's provider_outcome_unknown overwrite."""
    fake_call, calls = _transport_failing_call(fail_times=99)
    failing_chain, chain_calls = _unknown_outcome_chain()
    sleeps = []
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep",
                        lambda sec, _wake: (sleeps.append(sec), False)[1])
    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", failing_chain)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "local/candidate")
    monkeypatch.setenv("USE_LOCAL_FALLBACK", "true")
    from ouroboros.config import get_finalization_grace_sec

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    deadline = datetime.now(timezone.utc) + timedelta(seconds=get_finalization_grace_sec() + 8)
    registry._ctx.task_metadata = {"deadline_at": deadline.isoformat()}
    notes = []
    _result, usage, trace = run_llm_loop(**_loop_kwargs(tmp_path, registry, notes))

    assert chain_calls["n"] == 1
    assert calls["n"] >= 2
    assert usage.get("execution_status") == "infra_failed"
    assert usage.get("reason_code") == "provider_unavailable"
    assert trace.get("forced_finalization", {}).get("source") == "transport_unavailable_no_resend"


class _FlakyChatLLM:
    """Raises a typed released transport failure, then answers."""

    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.calls = 0

    def default_model(self):
        return "test-model"

    def chat(self, **_kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise _typed_transport_exc()
        return (
            {"role": "assistant", "content": "done"},
            {"prompt_tokens": 1, "completion_tokens": 1},
        )


def test_end_to_end_episode_through_real_retry_helper(tmp_path, monkeypatch):
    """Classification -> one physical attempt per call -> wait episode -> recovery,
    with the durable exception-class evidence riding the existing llm_api_error rows."""
    sleeps = []
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep",
                        lambda sec, _wake: (sleeps.append(sec), False)[1])
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    llm = _FlakyChatLLM(fail_times=2)
    notes = []
    result, _usage, _trace = run_llm_loop(
        **_loop_kwargs(tmp_path, ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path), notes, llm=llm))

    assert result == "done"
    assert llm.calls == 3  # one physical attempt per invocation, two free redials
    assert len(sleeps) == 2
    phases = [row["phase"] for row in _read_network_wait_events(tmp_path)]
    assert phases == ["entered", "waiting", "waiting", "recovered"]
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines() if line.strip()]
    api_errors = [row for row in rows if row.get("type") == "llm_api_error"]
    assert api_errors and all(row.get("error_kind") == "transport_unavailable" for row in api_errors)


# ------------------------------------------------------- interruptible sleeping

def test_interruptible_sleep_reacts_to_owner_signal_within_slices():
    incoming = queue.Queue()
    wake = lambda: not incoming.empty()  # noqa: E731
    threading.Timer(0.2, lambda: incoming.put("owner message")).start()
    start = time.monotonic()
    interrupted = loop_transport.interruptible_wait_sleep(30.0, wake)
    elapsed = time.monotonic() - start
    assert interrupted is True
    assert elapsed < 5.0  # slice-level reaction, not the full sleep


def test_interruptible_sleep_completes_quiet_short_sleep():
    assert loop_transport.interruptible_wait_sleep(0.05, lambda: False) is False


def test_wait_step_caps_sleep_at_note_interval_for_low_idle_timeouts(tmp_path, monkeypatch):
    """Owner-lowered idle timeouts shrink the sleep so waiting notes keep the
    idle rail alive (effective interval = min(constant, idle/2))."""
    sleeps = []
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep",
                        lambda sec, _wake: (sleeps.append(sec), False)[1])
    monkeypatch.setattr(loop_transport, "get_task_idle_timeout_sec", lambda: 60)
    episode = loop_transport.TransportWaitEpisode(
        started_monotonic=time.monotonic(), wait_iterations=10,
    )
    tools = SimpleNamespace(_ctx=SimpleNamespace(task_metadata={}, task_attempt=None))
    notes = []
    redial = loop_transport.transport_wait_step(
        episode, tools=tools, error_kind="transport_unavailable",
        drive_root=None, drive_logs=tmp_path, task_id="t-idle", model="m",
        emit_progress=lambda text, *, incident=None: notes.append(text),
        incoming_messages=None, owner_msg_seen=set(),
    )
    assert redial is True
    assert sleeps == [30.0]  # min(60s backoff cap, 60/2 note interval)
    assert len(notes) == 1  # the periodic note fired for the lowered interval


# ------------------------------------------------- route locality (A9, loopback)

def test_released_loopback_route_failure_stays_generic():
    """A loopback OPENAI_COMPATIBLE_BASE_URL install (Ollama / LM Studio /
    vLLM) stamps a remote-shaped provider name, but a stopped LOCAL server is
    not a network outage worth waiting out."""
    exc = httpx.ConnectError("connection refused")
    exc.physical_attempt_capture = ua.PhysicalAttemptCapture(
        attempt_id="pa-lb", model="m", provider="openai-compatible",
        state="released", candidate_measurement_kind="opaque",
        route_is_loopback=True,
    )
    assert classify_llm_exception(exc).kind != "transport_unavailable"


def test_released_remote_compatible_route_still_classifies_transport_unavailable():
    """The additive default keeps every remote route on the wait path."""
    exc = httpx.ConnectError("connection refused")
    exc.physical_attempt_capture = ua.PhysicalAttemptCapture(
        attempt_id="pa-rc", model="m", provider="openai-compatible",
        state="released", candidate_measurement_kind="opaque",
    )
    assert classify_llm_exception(exc).kind == "transport_unavailable"


def test_attempt_request_carries_route_locality_from_target():
    from ouroboros.llm import _attempt_request

    loopback = _attempt_request(
        {"provider": "openai-compatible", "usage_model": "m",
         "base_url": "http://localhost:11434/v1"},
        {"model": "m", "messages": []},
    )
    assert loopback.route_is_loopback is True
    remote = _attempt_request(
        {"provider": "openrouter", "usage_model": "m",
         "base_url": "https://openrouter.ai/api/v1"},
        {"model": "m", "messages": []},
    )
    assert remote.route_is_loopback is False


def test_attempt_capture_propagates_route_locality(tmp_path):
    reservation = ua.AttemptReservation(
        attempt_id="pa-cap", drive_root=tmp_path, model="m",
        provider="openai-compatible", reservation_upper_bound_usd=None,
    )
    request = ua.AttemptRequest(
        model="m", provider="openai-compatible", route_is_loopback=True,
    )
    capture = ua._record_attempt_capture(reservation, request, "released")
    assert capture.route_is_loopback is True


# ------------------------------------------ finalize_now during an episode (A7)

def test_finalize_now_during_episode_takes_no_resend_terminal_via_mailbox(tmp_path, monkeypatch):
    """finalize_now (deadline / ceiling / owner stop flavors share this exit)
    landing through the REAL owner mailbox while an episode is active: the
    episode's wake check sees it before sleeping, the round top drains it, and
    the terminal is the transport no-resend — zero further provider dials,
    never a forced-final paid call over the dead egress. The control is
    written from inside the first failing dispatch, so it lands after that
    round's drain and before the episode's first wait whatever the host's
    speed (a timer raced a cold process's setup)."""
    from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, write_owner_message

    calls = {"n": 0}

    def fake_call(_llm, _messages, _model, _tools, _effort, _max_retries, _drive_logs,
                  _task_id, _round_idx, _event_queue, accumulated_usage, *_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1:
            write_owner_message(tmp_path, "budget ceiling reached", "t-wait", kind=KIND_FINALIZE_NOW)
        accumulated_usage["_last_llm_error_kind"] = "transport_unavailable"
        accumulated_usage["_last_llm_error"] = "Connection error."
        return None, 0.0

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    notes = []
    start = time.monotonic()
    result, usage, trace = run_llm_loop(**_loop_kwargs(tmp_path, registry, notes))
    elapsed = time.monotonic() - start

    assert elapsed < 3.5  # the wake check saw the control before the 4 s backoff: no sleep ran
    assert calls["n"] == 1  # the woken redial exited at the round top: zero further dials
    assert usage.get("execution_status") == "infra_failed"
    assert usage.get("reason_code") == "provider_unavailable"
    assert trace.get("forced_finalization", {}).get("source") == "transport_unavailable_no_resend"
    assert "ended as a provider outage" in result
    events = _read_network_wait_events(tmp_path)
    assert events[-1]["phase"] == "ended"
    assert events[-1].get("detail") == "finalize_now"


# --------------------------------------------------- episode exit contracts (A12)

def test_redial_failing_with_different_kind_ends_episode_and_resumes_fallback(tmp_path, monkeypatch):
    """A redial that gets past the connect phase but fails differently proves
    the transport passable: the episode ends and the ORDINARY fallback chain
    resumes for the fresh kind."""
    calls = {"n": 0}

    def fake_call(_llm, _messages, _model, _tools, _effort, _max_retries, _drive_logs,
                  _task_id, _round_idx, _event_queue, accumulated_usage, *_a, **_k):
        calls["n"] += 1
        accumulated_usage["_last_llm_error_kind"] = (
            "transport_unavailable" if calls["n"] == 1 else "provider_transient"
        )
        return None, 0.0

    chain_calls = {"n": 0}

    def fake_chain(**kwargs):
        chain_calls["n"] += 1
        kwargs["accumulated_usage"].pop("_last_llm_error_kind", None)
        return (
            {"role": "assistant", "content": "fallback-ok"}, "other/model", False,
            kwargs["context_fit_plan"], kwargs["active_context_mode"],
        )

    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", lambda _sec, _wake: False)
    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", fake_chain)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    notes = []
    result, usage, _trace = run_llm_loop(**_loop_kwargs(tmp_path, ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path), notes))

    assert result == "fallback-ok"
    assert calls["n"] == 2  # primary, then the one free redial
    assert chain_calls["n"] == 1  # ordinary policy resumed after the episode ended
    assert usage.get("reason_code") is None
    events = _read_network_wait_events(tmp_path)
    assert events[-1]["phase"] == "ended"
    assert events[-1].get("detail") == "error_kind_changed:provider_transient"


def test_redial_with_unknown_outcome_takes_unknown_no_resend_terminal(tmp_path, monkeypatch):
    """A main-dispatch redial whose socket outcome is unknown ends the episode
    AND the round: provider_outcome_unknown keeps its own no-resend terminal —
    zero further dials of any kind."""
    calls = {"n": 0}

    def fake_call(_llm, _messages, _model, _tools, _effort, _max_retries, _drive_logs,
                  _task_id, _round_idx, _event_queue, accumulated_usage, *_a, **_k):
        calls["n"] += 1
        accumulated_usage["_last_llm_error_kind"] = (
            "transport_unavailable" if calls["n"] == 1 else "provider_outcome_unknown"
        )
        # Mirror _record_llm_call_error's stamps (the real failure path).
        accumulated_usage.update(execution_status="infra_failed", reason_code="llm_api_error")
        return None, 0.0

    def _chain_must_not_run(**_kwargs):
        raise AssertionError("no fallback chain after an unknown-outcome redial")

    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", lambda _sec, _wake: False)
    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", _chain_must_not_run)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "other/model")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    notes = []
    _result, usage, trace = run_llm_loop(**_loop_kwargs(tmp_path, ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path), notes))

    assert calls["n"] == 2  # zero dials after the unknown outcome
    assert usage.get("execution_status") == "infra_failed"
    assert trace.get("forced_finalization", {}).get("source") == "provider_outcome_unknown_no_resend"
    events = _read_network_wait_events(tmp_path)
    assert events[-1]["phase"] == "ended"
    assert events[-1].get("detail") == "error_kind_changed:provider_outcome_unknown"


def test_episode_ledger_evidence_has_zero_dispatched_paid_attempts(tmp_path, monkeypatch):
    """Between episode entry and the last wait, every durable failure row is
    the typed $0 transport kind and no usage row records a dispatched paid
    attempt (released rows are the legitimate evidence)."""
    sleeps = []
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep",
                        lambda sec, _wake: (sleeps.append(sec), False)[1])
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    llm = _FlakyChatLLM(fail_times=3)
    notes = []
    result, _usage, _trace = run_llm_loop(
        **_loop_kwargs(tmp_path, ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path), notes, llm=llm))

    assert result == "done"
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines() if line.strip()]
    entered_idx = next(i for i, r in enumerate(rows) if r.get("type") == "network_wait" and r.get("phase") == "entered")
    last_wait_idx = max(i for i, r in enumerate(rows) if r.get("type") == "network_wait" and r.get("phase") == "waiting")
    window = rows[entered_idx:last_wait_idx + 1]
    api_errors = [r for r in window if r.get("type") == "llm_api_error"]
    assert api_errors, "the episode's failures must leave durable evidence"
    assert all(r.get("error_kind") == "transport_unavailable" for r in api_errors)
    assert not any(r.get("type") == "llm_usage" for r in window)


def test_nonstandard_transient_retry_max_keeps_one_attempt_transport_contract(tmp_path, monkeypatch):
    """OUROBOROS_TRANSIENT_RETRY_MAX tunes the transient burst, never the
    one-physical-attempt transport contract."""
    monkeypatch.setenv("OUROBOROS_TRANSIENT_RETRY_MAX", "12")
    llm = _RaisingLLM(_typed_transport_exc)
    usage = {}
    msg, _cost = call_llm_with_retry(
        llm, [{"role": "user", "content": "hi"}], "test-model", None, "low", 3,
        tmp_path, "t-max", 1, None, usage,
    )
    assert msg is None
    assert llm.calls == 1
    assert usage.get("_last_llm_error_kind") == "transport_unavailable"


def test_review_actor_physical_send_rail_stays_bounded_at_two():
    """Review actors (P3 / scope / acceptance) dispatch under
    physical_attempt_limit(2) — the review_substrate contract — so a transport
    outage can never turn a review slot into an unbounded redial loop."""
    from ouroboros.usage_accounting import (
        PhysicalAttemptLimitExceeded,
        _claim_physical_dispatch,
        physical_attempt_limit,
    )

    with physical_attempt_limit(2):
        _claim_physical_dispatch()
        _claim_physical_dispatch()
        with pytest.raises(PhysicalAttemptLimitExceeded):
            _claim_physical_dispatch()


# ------------------------------------------------ terminal wordings + Q14 margin

def test_terminal_wordings_cover_four_wait_outcomes():
    """Four honest terminal texts keyed on the two typed facts: a managed task
    that waited says outage without the supervisor's lifecycle term
    INTERRUPTED; a managed zero-wait task names the spent window; an
    interactive turn is "this turn" (never "the task") that either names how
    long it waited or says no window was left; no wording claims a fast fail."""
    kwargs = dict(is_context_overflow=False, is_transport_wait=True, is_deadline_exhausted=False)
    waited = loop_transport.provider_terminal_fallback_text({}, waited_sec=610.0, **kwargs)
    assert "the task waited and redialed" in waited
    assert "ended as a provider outage, not completed" in waited
    assert "INTERRUPTED" not in waited
    zero_wait = loop_transport.provider_terminal_fallback_text({}, waited_sec=0.0, **kwargs)
    assert "left no time to wait" in zero_wait
    chat_waited = loop_transport.provider_terminal_fallback_text(
        {}, waited_sec=610.0, interactive=True, **kwargs)
    assert "this turn waited and redialed for 10.2 min" in chat_waited
    assert "ended as a provider outage, not completed" in chat_waited
    chat_zero = loop_transport.provider_terminal_fallback_text(
        {}, waited_sec=0.0, interactive=True, **kwargs)
    assert "no wait window was left" in chat_zero
    assert "this turn ended as a provider outage" in chat_zero
    for text in (chat_waited, chat_zero):
        assert "task" not in text.lower()
        assert "INTERRUPTED" not in text
    assert all("fails fast" not in text for text in (waited, zero_wait, chat_waited, chat_zero))


def test_final_redial_reserves_named_margin_before_admission_close(tmp_path, monkeypatch):
    """Q14/A11: the last free redial sleeps to remaining minus the named 3s
    margin (round-top overhead routinely eats ~1s), then the next step
    terminalizes deterministically."""
    from ouroboros.config import get_finalization_grace_sec
    from ouroboros.deadline_utils import dispatch_window_remaining_sec

    assert loop_transport._FINAL_REDIAL_MARGIN_SEC == 3.0
    sleeps = []
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep",
                        lambda sec, _wake: (sleeps.append(sec), False)[1])
    deadline = datetime.now(timezone.utc) + timedelta(seconds=get_finalization_grace_sec() + 40)
    tools = SimpleNamespace(_ctx=SimpleNamespace(
        task_metadata={"deadline_at": deadline.isoformat()}, task_attempt=None,
    ))
    episode = loop_transport.TransportWaitEpisode(
        started_monotonic=time.monotonic(), wait_iterations=10,  # backoff at the 60s cap
    )
    remaining = dispatch_window_remaining_sec(
        deadline_ts=deadline.timestamp(), reserve_sec=get_finalization_grace_sec(),
    )
    redial = loop_transport.transport_wait_step(
        episode, tools=tools, error_kind="transport_unavailable",
        drive_root=None, drive_logs=tmp_path, task_id="t-margin", model="m",
        emit_progress=lambda _n, *, incident=None: None, incoming_messages=None, owner_msg_seen=set(),
    )
    assert redial is True
    assert episode.final_redial_done is True
    assert sleeps[0] == pytest.approx(
        remaining - loop_transport._FINAL_REDIAL_MARGIN_SEC, abs=1.0)
    assert loop_transport.transport_wait_step(
        episode, tools=tools, error_kind="transport_unavailable",
        drive_root=None, drive_logs=tmp_path, task_id="t-margin", model="m",
        emit_progress=lambda _n, *, incident=None: None, incoming_messages=None, owner_msg_seen=set(),
    ) is False  # deadline_after_final_redial


# ------------------------------------------------ final-review regression pins

def test_released_httpx_proxy_error_classifies_transport_unavailable():
    """A CONNECT/SOCKS tunnel failure (httpx.ProxyError) happens before any
    provider request exists — typed pre-dispatch, same wait entry as connects."""
    from ouroboros.transport_custody import is_pre_dispatch_transport_failure

    assert is_pre_dispatch_transport_failure(httpx.ProxyError("503 from proxy"))
    result = classify_llm_exception(_typed_transport_exc(exc_cls=httpx.ProxyError))
    assert result.kind == "transport_unavailable"


def test_proxy_error_outage_enters_wait_and_recovers(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep",
                        lambda sec, _wake: (sleeps.append(sec), False)[1])
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)

    class ProxyFlakyLLM(_FlakyChatLLM):
        def chat(self, **kwargs):
            self.calls += 1
            if self.calls <= self.fail_times:
                raise _typed_transport_exc(exc_cls=httpx.ProxyError)
            return (
                {"role": "assistant", "content": "done"},
                {"prompt_tokens": 1, "completion_tokens": 1},
            )

    llm = ProxyFlakyLLM(fail_times=2)
    notes = []
    result, _usage, _trace = run_llm_loop(
        **_loop_kwargs(tmp_path, ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path), notes, llm=llm))
    assert result == "done"
    assert llm.calls == 3
    phases = [row["phase"] for row in _read_network_wait_events(tmp_path)]
    assert phases == ["entered", "waiting", "waiting", "recovered"]


def test_budget_exceeded_mid_wait_closes_episode_with_ended_event(tmp_path, monkeypatch):
    """The budget rail firing between free redials must not leave the episode's
    durable story open: entered/waiting rows get their ended(budget_exhausted)."""
    calls = {"n": 0}

    def fake_call(_llm, _messages, _model, _tools, _effort, _max_retries, _drive_logs,
                  _task_id, _round_idx, _event_queue, accumulated_usage, *_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1:
            accumulated_usage["_last_llm_error_kind"] = "transport_unavailable"
            return None, 0.0
        raise ua.BudgetExceeded("root budget exhausted by a concurrent consumer")

    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", lambda _s, _w: False)
    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    notes = []
    # replay_safe budget exits re-raise to the supervisor (base pause-and-retry
    # semantics) — the episode's durable story must still be closed first.
    with pytest.raises(ua.BudgetExceeded):
        run_llm_loop(
            **_loop_kwargs(tmp_path, ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path), notes))

    assert calls["n"] == 2
    events = _read_network_wait_events(tmp_path)
    assert [row["phase"] for row in events][0] == "entered"
    assert events[-1]["phase"] == "ended"
    assert events[-1].get("detail") == "budget_exhausted"
