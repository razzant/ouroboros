"""Contracts for the bounded paid repeat after a post-dispatch transport death.

A DISPATCHED request whose socket died with a typed transport death (httpx
ReadError / WriteError / RemoteProtocolError, or the requests ProtocolError /
RemoteDisconnected shape) is `provider_outcome_unknown`; an inline Presence
turn's PRIMARY round dispatch alone may repeat it at most twice per round (every
other turn's new generation after an unknown outcome is the round recovery,
``tests/test_unknown_fallback_first.py``), each repeat a NEW
physical attempt with its own ledger lifecycle (the earlier rows stay unresolved
at their upper bound). Every other surface keeps the no-resend doctrine, the
classifier is unchanged, and every durable `llm_api_error` row tells the truth
about what happens next.
"""

from __future__ import annotations

import json
import queue
from types import SimpleNamespace

import httpx
import pytest

import ouroboros.loop as loop_mod
import ouroboros.loop_llm_call as call_mod
import ouroboros.loop_transport as loop_transport
from ouroboros import usage_accounting as ua
from ouroboros.loop import run_llm_loop
from ouroboros.loop_llm_call import (
    RETRY_WALL_EXHAUSTED_KEY,
    TRANSPORT_DEATHS_KEY,
    _TRANSPORT_DEATH_RETRIES,
    call_llm_with_retry,
    classify_llm_exception,
    provider_no_call_source,
)
from ouroboros.tools.registry import ToolRegistry
from tests._usage_store_testing import ledger_rows

MESSAGES = [{"role": "user", "content": "hi"}]
OK_RESPONSE = ({"role": "assistant", "content": "done"}, {"prompt_tokens": 1, "completion_tokens": 1})


def _capture(state: str = "unresolved", provider: str = "openrouter") -> ua.PhysicalAttemptCapture:
    return ua.PhysicalAttemptCapture(
        attempt_id=f"pa-{state}", model="test-model", provider=provider, state=state,
        candidate_measurement_kind="opaque",
    )


def _death(exc_cls=httpx.ReadError, state: str = "unresolved", provider: str = "openrouter"):
    """The OpenAI SDK shape: ``raise APIConnectionError(request=request) from err``
    with the custody capture execute_physical_attempt attaches."""
    try:
        raise RuntimeError("Connection error.") from exc_cls("socket died after dispatch")
    except RuntimeError as exc:
        exc.physical_attempt_capture = _capture(state=state, provider=provider)
        return exc


def _status_failure(status: int):
    """A provider status on the repeat (the capture already holds the death's row)."""
    exc = RuntimeError(f"HTTP {status} from the provider")
    exc.status_code = status
    exc.physical_attempt_capture = ua.PhysicalAttemptCapture(
        attempt_id="pa-status", model="test-model", provider="openrouter", state="unresolved",
        candidate_measurement_kind="opaque", provider_status_code=status,
    )
    return exc


EMPTY_RESPONSE = ({"role": "assistant", "content": "", "tool_calls": []}, {"response_finish_reason": None})


def _released_connect():
    try:
        raise RuntimeError("Connection error.") from httpx.ConnectError("connection refused")
    except RuntimeError as exc:
        exc.physical_attempt_capture = _capture(state="released")
        return exc


class _ScriptedLLM:
    """chat() follows a script: an exception factory raises, anything else is returned."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls = 0

    def default_model(self):
        return "test-model"

    def chat(self, **_kwargs):
        self.calls += 1
        step = self.script.pop(0) if self.script else OK_RESPONSE
        if callable(step):
            raise step()
        return step


def _events(drive_logs, kind: str):
    path = drive_logs / "events.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [row for row in rows if row.get("type") == kind]


@pytest.fixture
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(call_mod, "_sleep_within_deadline", lambda sec, _dl, **_kw: (sleeps.append(sec), True)[1])
    return sleeps


def _primary_call(llm, drive_logs, usage, *, round_idx=1, max_retries=3, **kwargs):
    return call_llm_with_retry(
        llm, MESSAGES, "test-model", None, "low", max_retries, drive_logs, "t-death",
        round_idx, None, usage, transport_death_retries=_TRANSPORT_DEATH_RETRIES, **kwargs,
    )


# ------------------------------------------------------------------ ledger rail

@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    (root / "state").mkdir(parents=True)
    return root


def _ledger(root):
    return [row for row in ledger_rows(root) if row.get("kind") == "attempt"]


class _LedgerLLM:
    """Every chat() is a REAL execute_physical_attempt (reserve → dispatched →
    unresolved/settled) whose send follows the script — the production shape."""

    def __init__(self, root, *script, reservation_usd: float = 1.0):
        self.root = root
        self.script = list(script)
        self.calls = 0
        self.reservation_usd = reservation_usd

    def default_model(self):
        return "test-model"

    def chat(self, **_kwargs):
        self.calls += 1
        step = self.script.pop(0) if self.script else "ok"

        def send():
            if callable(step):
                raise RuntimeError("Connection error.") from step()
            return {"content": "done"}

        request = ua.AttemptRequest(
            model="test-model", provider="openrouter", reservation_usd=self.reservation_usd,
            drive_root=self.root, task_id="t-death", root_task_id="t-death", source="test.death",
        )
        ua.execute_physical_attempt(
            request, send, extractor=lambda _resp: ({"prompt_tokens": 1, "completion_tokens": 1}, 0.01, True),
        )
        return OK_RESPONSE


def test_two_deaths_then_success_are_three_ledger_lifecycles(data_root, tmp_path, no_sleep):
    """Each physical send is its own reservation/dispatch/terminal lifecycle; the
    dead ones stay unresolved at their upper bound; exactly one durable error row
    per FAILED send, each truthfully announcing the repeat; the success emits none."""
    llm = _LedgerLLM(data_root, lambda: httpx.ReadError("died"), lambda: httpx.RemoteProtocolError("eof"))
    usage = {}
    msg, _cost = _primary_call(llm, tmp_path, usage)

    assert msg == OK_RESPONSE[0]
    assert llm.calls == 3
    rows = _ledger(data_root)
    by_attempt = {}
    for row in rows:
        by_attempt.setdefault(row["attempt_id"], []).append(row["state"])
    assert len(by_attempt) == 3  # never a reused reservation
    assert list(by_attempt.values()) == [
        ["unresolved"],
        ["unresolved"],
        ["settled"],
    ]
    assert ua.usage_projection(data_root)["unresolved_upper_bound_usd"] == 2.0
    api_errors = _events(tmp_path, "llm_api_error")
    assert [row["retry_same_request"] for row in api_errors] == [True, True]
    assert {row["error_kind"] for row in api_errors} == {"provider_outcome_unknown"}
    assert [row["transport_cause_type"] for row in api_errors] == ["ReadError", "RemoteProtocolError"]
    assert {row["attempt_custody_state"] for row in api_errors} == {"unresolved"}
    unresolved_ids = [aid for aid, states in by_attempt.items() if states[-1] == "unresolved"]
    assert [row["physical_attempt_id"] for row in api_errors] == unresolved_ids
    assert _events(tmp_path, "llm_non_retryable_same_request") == []
    assert "_last_llm_error_kind" not in usage
    assert TRANSPORT_DEATHS_KEY not in usage  # the round is over: a success clears its repeat record


def test_budget_refusal_on_the_second_send_propagates_untouched(data_root, tmp_path, no_sleep, monkeypatch):
    """A BudgetExceeded from reserve_attempt on the repeat propagates as-is and no
    further physical attempt is dispatched (sol s9). The dead send's unresolved bound
    is exposure, not spending (#1487); the repeat is refused because another task's
    KNOWN charge reached the $1.50 wallet while the first send died."""
    monkeypatch.setenv("TOTAL_BUDGET", "1.5")

    def death_while_known_spend_lands():
        ua.record_subscription_session("other-session", drive_root=data_root, route="subscription",
                                       task_id="other", root_task_id="other", spend_usd=1.5)
        return httpx.ReadError("died")

    llm = _LedgerLLM(data_root, death_while_known_spend_lands, reservation_usd=1.0)
    usage = {}
    with pytest.raises(ua.BudgetExceeded) as raised:
        _primary_call(llm, tmp_path, usage)

    assert raised.value.limit_scope == "global"
    assert llm.calls == 2
    rows = [row for row in _ledger(data_root) if row.get("task_id") == "t-death"]
    assert [row["state"] for row in rows] == ["unresolved"]
    assert len(_events(tmp_path, "llm_api_error")) == 1  # only the death itself was a provider failure
    assert _events(tmp_path, "llm_non_retryable_same_request") == []
    # A budget refusal proves only that a reservation was refused, not that the granted
    # repeat never left the host (llm.chat retries on the wire before a later reservation
    # can hit the wall), so the round record keeps the attempt booked, the flag stays as
    # the grant left it, and the budget terminal, not the provider terminal, ends the round.
    assert usage[TRANSPORT_DEATHS_KEY]["count"] == 1
    assert usage["_last_llm_retry_same_request"] is True
    assert usage["_last_llm_error_kind"] == "provider_outcome_unknown"  # the sticky kind is untouched


# --------------------------------------------------------- bounded, truthful flags

def test_third_death_exhausts_the_round_budget(tmp_path, no_sleep):
    llm = _ScriptedLLM(_death, _death, _death)
    usage = {}
    msg, _cost = _primary_call(llm, tmp_path, usage)

    assert msg is None
    assert llm.calls == 3  # 1 + _TRANSPORT_DEATH_RETRIES
    api_errors = _events(tmp_path, "llm_api_error")
    assert [row["retry_same_request"] for row in api_errors] == [True, True, False]
    non_retryable = _events(tmp_path, "llm_non_retryable_same_request")
    assert len(non_retryable) == 1
    assert non_retryable[0]["error_kind"] == "provider_outcome_unknown"
    assert non_retryable[0]["attempt"] == 3
    assert usage["_last_llm_error_kind"] == "provider_outcome_unknown"
    assert usage["_last_llm_retry_same_request"] is False
    assert RETRY_WALL_EXHAUSTED_KEY not in usage  # the unknown terminal outranks the wall
    assert provider_no_call_source(usage, False) == ("provider_outcome_unknown_no_resend", False)
    assert no_sleep == [4.0, 8.0]  # backoff by death ordinal; none after exhaustion


def test_deadline_window_refuses_the_repeat_before_it_is_promised(tmp_path, no_sleep):
    """The grant re-checks the admission the NEXT iteration would apply (backoff
    plus the finalization reserve): inside that window the death's own row
    already says no repeat, the non-retryable row is written, nothing is
    counted, nothing sleeps, and the unknown no-resend terminal follows."""
    import time

    llm = _ScriptedLLM(_death, _death)
    usage = {}
    msg, _cost = _primary_call(llm, tmp_path, usage, deadline_ts=time.time() + 32, transport_reserve_sec=30.0)

    assert msg is None
    assert llm.calls == 1
    assert [row["retry_same_request"] for row in _events(tmp_path, "llm_api_error")] == [False]
    assert len(_events(tmp_path, "llm_non_retryable_same_request")) == 1
    assert TRANSPORT_DEATHS_KEY not in usage
    assert no_sleep == []
    assert usage["_last_llm_error_kind"] == "provider_outcome_unknown"
    assert usage["_last_llm_retry_same_request"] is False
    assert provider_no_call_source(usage, False) == ("provider_outcome_unknown_no_resend", False)


def test_backoff_sleep_refusal_race_stops_without_a_resend(tmp_path, monkeypatch):
    """Residual race (the sleep gate refusing after the grant, e.g. a laptop
    that slept between the two checks): the loop stops on the durable
    deadline-exhausted event, keeps the unknown kind, never resends, and takes
    the never-sent repeat back off the round record so the hint stays true."""
    monkeypatch.setattr(call_mod, "_sleep_within_deadline", lambda _sec, _dl: False)
    llm = _ScriptedLLM(_death, _death)
    usage = {}
    msg, _cost = _primary_call(llm, tmp_path, usage)

    assert msg is None
    assert llm.calls == 1
    deadline_rows = _events(tmp_path, "llm_retry_deadline_exhausted")
    assert [row["error_kind"] for row in deadline_rows] == ["provider_outcome_unknown"]
    assert usage["_last_llm_error_kind"] == "provider_outcome_unknown"
    assert RETRY_WALL_EXHAUSTED_KEY not in usage
    assert provider_no_call_source(usage, False) == ("provider_outcome_unknown_no_resend", False)
    assert TRANSPORT_DEATHS_KEY not in usage
    assert "were repeated" not in loop_transport.provider_recovery_hint(usage)


def test_sleep_refusal_after_the_second_grant_keeps_only_the_real_repeat(tmp_path, monkeypatch):
    """Attempt 1 was really repeated (attempt 2 died too); the second grant is
    refused by the sleep gate: the record keeps exactly one real repeat."""
    gates = iter([True, False])
    monkeypatch.setattr(call_mod, "_sleep_within_deadline", lambda _sec, _dl: next(gates))
    llm = _ScriptedLLM(_death, _death, _death)
    usage = {}
    msg, _cost = _primary_call(llm, tmp_path, usage)

    assert msg is None
    assert llm.calls == 2
    assert usage[TRANSPORT_DEATHS_KEY]["count"] == 1
    hint = loop_transport.provider_recovery_hint(usage)
    assert "1 earlier physical attempt(s) of the last dispatched round" in hint
    assert provider_no_call_source(usage, False) == ("provider_outcome_unknown_no_resend", False)


def test_admission_refusal_after_a_granted_repeat_keeps_the_unknown_fence(tmp_path, monkeypatch, no_sleep):
    """If the loop-top admission gate refuses the granted repeat anyway (clock
    moved between the grant and the re-check), the refusal must not relabel the
    unresolved request as `deadline_exhausted`: the forced-final rail would then
    dial a NEW paid request over a possibly live one."""
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", _no_chain)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    checks = {"n": 0}

    def exhausted_after_the_grant(**_kwargs):
        checks["n"] += 1
        return checks["n"] > 3  # loop-top, pre-send, the grant's re-check pass; the next admission refuses

    monkeypatch.setattr(call_mod, "owner_deadline_exhausted", exhausted_after_the_grant)
    llm = _ScriptedLLM(_death, _death, _death)
    notes = []
    kwargs = _loop_kwargs(tmp_path, llm, notes)
    _presence_turn(kwargs)  # Inline Presence alone retains the bounded repeat rail.
    _result, usage, trace = run_llm_loop(**kwargs)

    assert llm.calls == 1  # the refused repeat and the forced-final rail both dispatched nothing
    assert usage["_last_llm_error_kind"] == "provider_outcome_unknown"
    assert usage["_last_llm_retry_same_request"] is False
    assert len(_events(tmp_path, "llm_not_dispatched")) == 1
    assert trace.get("forced_finalization", {}).get("source") == "provider_outcome_unknown_no_resend"
    assert TRANSPORT_DEATHS_KEY not in usage  # the granted-but-unsent repeat is un-counted
    assert "were repeated" not in _result  # the hint names only real repeats


def test_attempt_loop_ceiling_stays_the_outer_bound_and_the_flag_stays_truthful(tmp_path, no_sleep, monkeypatch):
    """The transient attempt budget is the outer ceiling of the same loop: at its
    last slot the death's row already says no repeat (no true-then-nothing)."""
    monkeypatch.setenv("OUROBOROS_TRANSIENT_RETRY_MAX", "2")
    llm = _ScriptedLLM(_death, _death, _death)
    usage = {}
    msg, _cost = _primary_call(llm, tmp_path, usage, max_retries=1)

    assert msg is None
    assert llm.calls == 2
    assert [row["retry_same_request"] for row in _events(tmp_path, "llm_api_error")] == [True, False]
    assert len(_events(tmp_path, "llm_non_retryable_same_request")) == 1


@pytest.mark.parametrize("non_death", [
    lambda: _death(httpx.ReadTimeout),
    lambda: _death(httpx.ReadError, provider="local"),
])
def test_unknown_outcomes_that_are_not_typed_transport_deaths_are_never_resent(tmp_path, no_sleep, non_death):
    llm = _ScriptedLLM(non_death, non_death)
    usage = {}
    msg, _cost = _primary_call(llm, tmp_path, usage)

    assert msg is None
    assert llm.calls == 1
    assert usage["_last_llm_error_kind"] == "provider_outcome_unknown"
    assert usage["_last_llm_retry_same_request"] is False
    assert len(_events(tmp_path, "llm_non_retryable_same_request")) == 1
    assert no_sleep == []


def test_requests_lane_death_repeats_on_the_same_rail(tmp_path, no_sleep):
    """The Anthropic-native requests/urllib3 shape rides the same bounded rail."""
    import http.client

    import requests
    import urllib3

    def requests_death():
        exc = requests.exceptions.ConnectionError(urllib3.exceptions.ProtocolError(
            "Connection aborted.", http.client.RemoteDisconnected("closed without response"),
        ))
        exc.physical_attempt_capture = _capture(provider="anthropic")
        return exc

    llm = _ScriptedLLM(requests_death)
    usage = {}
    msg, _cost = _primary_call(llm, tmp_path, usage)
    assert msg == OK_RESPONSE[0]
    assert llm.calls == 2
    api_errors = _events(tmp_path, "llm_api_error")
    assert [(row["error_kind"], row["retry_same_request"]) for row in api_errors] == [("provider_outcome_unknown", True)]
    assert api_errors[0]["transport_cause_type"] == "RemoteDisconnected"


def test_requests_chunked_body_disconnect_repeats_on_the_same_rail(data_root, tmp_path, no_sleep):
    """The Anthropic-native POST is non-streaming, so the body is read inside
    ``requests.post`` and a socket that dies mid-body raises a BARE
    ``ChunkedEncodingError`` — no ``ConnectionError`` in its MRO — out of the
    physical send. Through the REAL ledger: one bounded repeat, two physical
    lifecycles (the dead one left unresolved at its upper bound), and the durable
    row names the innermost typed cause."""
    import http.client

    import requests
    import urllib3

    def chunked_body_death():
        raise requests.exceptions.ChunkedEncodingError(urllib3.exceptions.ProtocolError(
            "Connection broken: IncompleteRead(0 bytes read)",
            http.client.RemoteDisconnected("closed without response"),
        ))

    llm = _LedgerLLM(data_root, chunked_body_death)
    usage = {}
    msg, _cost = _primary_call(llm, tmp_path, usage)

    assert msg == OK_RESPONSE[0]
    assert llm.calls == 2
    by_attempt = {}
    for row in _ledger(data_root):
        by_attempt.setdefault(row["attempt_id"], []).append(row["state"])
    assert list(by_attempt.values()) == [
        ["unresolved"],
        ["settled"],
    ]
    assert ua.usage_projection(data_root)["unresolved_upper_bound_usd"] == 1.0
    api_errors = _events(tmp_path, "llm_api_error")
    assert [(row["error_kind"], row["retry_same_request"]) for row in api_errors] == [
        ("provider_outcome_unknown", True),
    ]
    assert api_errors[0]["transport_cause_type"] == "RemoteDisconnected"
    assert _events(tmp_path, "llm_non_retryable_same_request") == []


# ------------------------------------------------------- round-keyed counter

def test_counter_is_keyed_by_round_and_survives_re_entry_of_the_same_round(tmp_path, no_sleep):
    """A re-entry with the SAME round id after a FAILED invocation (the wait
    episode's free redial) keeps the spent count; the next round starts from zero."""
    usage = {}
    llm = _ScriptedLLM(_death, _death, _death)
    msg, _cost = _primary_call(llm, tmp_path, usage, round_idx=1)
    assert msg is None and llm.calls == 3
    assert usage[TRANSPORT_DEATHS_KEY] == {  # the exhausted repeat's own class rides the record
        "round_id": f"{usage['execution_id']}:round:1", "count": 2, "backoff_sec": 8.0,
        "error_kind": "provider_outcome_unknown",
    }

    redial = _ScriptedLLM(_death, _death)
    msg, _cost = _primary_call(redial, tmp_path, usage, round_idx=1)
    assert msg is None
    assert redial.calls == 1  # the round's budget was already spent
    assert usage["_last_llm_retry_same_request"] is False

    next_round = _ScriptedLLM(_death, _death)
    msg, _cost = _primary_call(next_round, tmp_path, usage, round_idx=2)
    assert msg == OK_RESPONSE[0] and next_round.calls == 3  # re-armed: two repeats granted again
    assert TRANSPORT_DEATHS_KEY not in usage  # and the successful round cleared its record


def test_stale_counter_from_an_earlier_round_is_dropped_by_a_later_unknown(tmp_path, no_sleep):
    """A later round's NON-transport unknown must not inherit an earlier round's
    repeat count (the terminal hint reads the record)."""
    usage = {}
    _primary_call(_ScriptedLLM(_death, _death, _death), tmp_path, usage, round_idx=1)
    assert usage[TRANSPORT_DEATHS_KEY]["count"] == 2
    _primary_call(_ScriptedLLM(lambda: _death(httpx.ReadTimeout)), tmp_path, usage, round_idx=3)
    assert TRANSPORT_DEATHS_KEY not in usage
    assert "earlier physical attempt" not in loop_transport.provider_recovery_hint(usage)


def test_round_success_clears_the_repeat_record_for_later_no_call_terminals(tmp_path, no_sleep):
    """A round that spent repeats and then SUCCEEDED leaves no record: a later
    unknown terminal stamped without a dispatch (the delegate-hold paths) must
    not read "repeated" for a request that was never sent."""
    usage = {}
    _primary_call(_ScriptedLLM(_death, _death), tmp_path, usage, round_idx=1)
    assert TRANSPORT_DEATHS_KEY not in usage
    usage["_last_llm_error_kind"] = "provider_outcome_unknown"
    hint = loop_transport.provider_recovery_hint(usage)
    assert "earlier physical attempt" not in hint
    assert "no retry or paid fallback was sent" in hint


# --------------------------------------------------------------- round gate

def _loop_kwargs(tmp_path, llm, notes):
    return dict(
        messages=[{"role": "user", "content": "go"}],
        tools=ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path),
        llm=llm,
        drive_logs=tmp_path,
        # The wait episode's owner notes carry the ``incident=`` keyword
        # (``OuroborosAgent._emit_progress``); a bare list.append rejects it.
        emit_progress=lambda text, *, incident=None: notes.append(text),
        incoming_messages=queue.Queue(),
        task_id="t-death",
        drive_root=tmp_path,
    )


def _no_chain(**_kwargs):
    raise AssertionError("unknown physical work must stop the paid fallback chain")


def _presence_turn(kwargs):
    """Inline Presence, the one caller that keeps the bounded paid-repeat rail."""
    kwargs["task_type"] = "presence"
    kwargs["tools"]._ctx.is_direct_chat = True
    kwargs["tools"]._ctx.current_task_type = "presence"
    return kwargs


def test_primary_round_dispatch_exhaustion_takes_the_unknown_no_resend_terminal(tmp_path, monkeypatch, no_sleep):
    """End to end through the real round dispatcher: three deaths, then the
    forced-final rail ships the salvage WITHOUT a provider call and the terminal
    text names the repeats."""
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", _no_chain)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "other/model")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    llm = _ScriptedLLM(_death, _death, _death, _death)
    notes = []
    kwargs = _loop_kwargs(tmp_path, llm, notes)
    _presence_turn(kwargs)  # Inline Presence alone retains the bounded repeat rail.
    result, usage, trace = run_llm_loop(**kwargs)

    assert llm.calls == 3  # zero further dials of any kind: no forced-final resend
    assert usage.get("execution_status") == "infra_failed"
    assert usage.get("reason_code") == "provider_unavailable"
    assert trace.get("forced_finalization", {}).get("source") == "provider_outcome_unknown_no_resend"
    assert "2 earlier physical attempt(s) of the last dispatched round" in result
    assert "no further retry or paid fallback was sent" in result


def test_primary_round_dispatch_recovers_after_two_deaths(tmp_path, monkeypatch, no_sleep):
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", _no_chain)
    monkeypatch.setattr(loop_transport, "upstream_transport_reachable", lambda *a, **kw: {"kind": "upstream_http", "status_code": 200})
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", lambda *a: False)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    llm = _ScriptedLLM(_death, _death)
    notes = []
    result, usage, _trace = run_llm_loop(**_loop_kwargs(tmp_path, llm, notes))

    assert result == "done"
    assert llm.calls == 3
    assert usage.get("reason_code") is None
    assert TRANSPORT_DEATHS_KEY not in usage
    assert no_sleep == []  # Managed attempts wait for upstream proof, never use the paid-repeat backoff.
    assert usage["transport_recovery"]["old_outcome"] == "unknown"


def _tool_round(name, args, call_id):
    return (
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }]},
        {"prompt_tokens": 11, "completion_tokens": 2},
    )


def test_unexpected_loop_error_carries_accumulated_evidence_to_owner_projection(
    tmp_path, monkeypatch,
):
    """The outer agent catch must not turn a failed multi-round loop into 0 calls.

    Round 1 runs the REAL tool executor, so ``llm_trace`` holds a recorded call
    with its durable trace ref; round 2's executor then dies. The loop owns that
    accumulated evidence, so it must reach the raiser instead of being replaced
    by an empty projection.
    """
    real_handle_tool_calls = loop_mod.handle_tool_calls
    executed = {"count": 0}

    def explode_after_the_first_batch(*args, **kwargs):
        executed["count"] += 1
        if executed["count"] == 1:
            return real_handle_tool_calls(*args, **kwargs)
        raise RuntimeError("tool executor crashed after the provider response")

    monkeypatch.setattr(loop_mod, "handle_tool_calls", explode_after_the_first_batch)
    llm = _ScriptedLLM(
        _tool_round("write_file", {"root": "task_drive", "path": "a.txt", "content": "a"}, "call-1"),
        _tool_round("write_file", {"root": "task_drive", "path": "b.txt", "content": "b"}, "call-2"),
    )

    with pytest.raises(RuntimeError) as caught:
        run_llm_loop(**_loop_kwargs(tmp_path, llm, []))

    # The ORIGINAL exception propagates: same object, same traceback, no
    # wrapper that would hide where the lifecycle actually failed.
    assert str(caught.value) == "tool executor crashed after the provider response"
    assert [entry.name for entry in caught.traceback][-1] == "explode_after_the_first_batch"
    usage = getattr(caught.value, "_ouroboros_loop_usage")
    trace = getattr(caught.value, "_ouroboros_loop_trace")
    assert usage["rounds"] == 2
    assert usage["prompt_tokens"] == 22
    assert usage["completion_tokens"] == 4
    assert TRANSPORT_DEATHS_KEY not in usage
    # Durable evidence for the completed batch survives the failure.
    assert [call["tool_call_id"] for call in trace["tool_calls"]] == ["call-1"]
    assert trace["tool_calls"][0]["trace_ref"]["call_id"]


@pytest.mark.parametrize("turn_flag", ["presence"])
def test_counter_survives_the_wait_episodes_free_redial_of_the_same_round(tmp_path, monkeypatch, no_sleep, turn_flag):
    """death → released ConnectError → wait episode → free redial → death →
    death: the round stays bounded by two paid repeats in total (sol s1). The
    same holds for an interactive turn, whose episode is bounded by the task
    idle timeout instead of the queue rails: the episode's free redial of the
    same round neither clears nor re-arms the round record."""
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", lambda _sec, _wake: False)
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", _no_chain)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "other/model")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    llm = _ScriptedLLM(_death, _released_connect, _death, _death, _death)
    notes = []
    kwargs = _loop_kwargs(tmp_path, llm, notes)
    if turn_flag:
        _presence_turn(kwargs)
    _result, usage, trace = run_llm_loop(**kwargs)

    assert llm.calls == 4  # death, connect (free), death, death — the fifth script step never runs
    api_errors = _events(tmp_path, "llm_api_error")
    assert [(row["error_kind"], row["retry_same_request"]) for row in api_errors] == [
        ("provider_outcome_unknown", True),
        ("transport_unavailable", True),
        ("provider_outcome_unknown", True),
        ("provider_outcome_unknown", False),
    ]
    assert len(_events(tmp_path, "llm_non_retryable_same_request")) == 1
    assert [row["phase"] for row in _events(tmp_path, "network_wait")] == ["entered", "waiting", "ended"]
    waiting = [row for row in _events(tmp_path, "network_wait") if row["phase"] == "waiting"]
    assert ("window_remaining_sec" in waiting[0]) is bool(turn_flag)  # the idle bound was live for the interactive kinds only
    assert trace.get("forced_finalization", {}).get("source") == "provider_outcome_unknown_no_resend"
    assert usage[TRANSPORT_DEATHS_KEY]["count"] == 2


# ------------------------------------------------ zero repeats everywhere else

def test_forced_final_call_never_repeats_a_transport_death(tmp_path, no_sleep):
    llm = _ScriptedLLM(_death, _death)
    ctx = SimpleNamespace(
        llm=llm, messages=MESSAGES, active_model="test-model", active_effort="low",
        max_retries=3, drive_logs=tmp_path, task_id="t-forced", round_idx=1,
        event_queue=None, accumulated_usage={}, task_type="task", active_use_local=False,
        deadline_ts=None,
    )
    text = loop_mod._call_forced_model_once(ctx)

    assert text == ""
    assert llm.calls == 1
    assert ctx.accumulated_usage["_last_llm_error_kind"] == "provider_outcome_unknown"
    assert ctx.accumulated_usage["_last_llm_retry_same_request"] is False
    assert no_sleep == []


@pytest.mark.parametrize("attempt_cap,expected_calls", [(None, 3), (2, 1)])
def test_round_dispatcher_opts_in_only_the_primary(tmp_path, no_sleep, attempt_cap, expected_calls):
    """attempt_cap is set only for fallback-chain candidates (the primary passes
    None): candidates get zero paid repeats, an inline Presence primary its bounded two."""
    from copy import deepcopy

    llm = _ScriptedLLM(_death, _death)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    ctx = SimpleNamespace(
        llm=llm, messages=deepcopy(MESSAGES), tools=registry, active_model="test-model", tool_schemas=[],
        context_fit_plan=None, active_context_mode="max", drive_root=tmp_path,
        active_effort="low", max_retries=3, drive_logs=tmp_path, task_id="t-cand", round_idx=1,
        event_queue=None, accumulated_usage={}, task_type="presence", active_use_local=False,
    )
    msg, _cost = loop_mod._dispatch_round_model(ctx, None, attempt_cap=attempt_cap)

    assert llm.calls == expected_calls
    assert (msg is not None) is (attempt_cap is None)


def test_default_budget_is_zero_for_every_direct_caller(tmp_path, no_sleep):
    llm = _ScriptedLLM(_death, _death)
    msg, _cost = call_llm_with_retry(llm, MESSAGES, "test-model", None, "low", 3, tmp_path, "t-default", 1, None, {})
    assert msg is None
    assert llm.calls == 1


def test_classifier_and_review_custody_are_unchanged_by_the_rail():
    """The global classifier still says no-resend for a dispatched death, and
    review custody (which consults it) keeps refusing a second paid send."""
    from ouroboros.review_custody import retryable_review_exception
    from ouroboros.transport_custody import is_retryable_transport_death

    exc = _death()
    assert is_retryable_transport_death(exc) is True
    classification = classify_llm_exception(exc)
    assert classification.kind == "provider_outcome_unknown"
    assert classification.retry_same_request is False
    assert retryable_review_exception(exc, None) is False


def test_recovery_hint_names_the_spent_repeats():
    usage = {"_last_llm_error_kind": "provider_outcome_unknown", TRANSPORT_DEATHS_KEY: {"round_id": "r", "count": 2}}
    hint = loop_transport.provider_recovery_hint(usage)
    assert "2 earlier physical attempt(s) of the last dispatched round" in hint
    assert "no terminal provider outcome" in hint
    assert "no further retry or paid fallback was sent" in hint
    mixed = loop_transport.provider_recovery_hint({"_last_llm_error_kind": "bad_request", TRANSPORT_DEATHS_KEY: {"round_id": "r", "count": 1}})
    assert "1 earlier physical attempt(s) of the last dispatched round" in mixed
    assert "the repeat failed as bad_request" in mixed
    assert "no further retry or paid fallback was sent" in mixed
    bare = loop_transport.provider_recovery_hint({"_last_llm_error_kind": "provider_outcome_unknown"})
    assert "no retry or paid fallback was sent" in bare


def test_recovery_hint_names_the_class_on_the_record_over_the_sticky_kind():
    """The record carries the class the repeat itself failed with; the sticky kind
    may by then belong to a later free redial (or a refusal) of the same round.
    A record without the stamp keeps reading the sticky kind."""
    record = {"round_id": "r", "count": 1, "backoff_sec": 4.0, "error_kind": "transport_unavailable"}
    released = loop_transport.provider_recovery_hint({"_last_llm_error_kind": "bad_request", TRANSPORT_DEATHS_KEY: record})
    assert "the repeat failed as transport_unavailable" in released
    assert "bad_request" not in released
    exhausted = loop_transport.provider_recovery_hint({
        "_last_llm_error_kind": "deadline_exhausted",
        TRANSPORT_DEATHS_KEY: {**record, "count": 2, "error_kind": "provider_outcome_unknown"},
    })
    assert "2 earlier physical attempt(s) of the last dispatched round" in exhausted
    assert "no terminal provider outcome" in exhausted and "deadline_exhausted" not in exhausted
    unstamped = loop_transport.provider_recovery_hint({"_last_llm_error_kind": "bad_request", TRANSPORT_DEATHS_KEY: {"round_id": "r", "count": 1}})
    assert "the repeat failed as bad_request" in unstamped


def test_round_record_carries_the_class_the_repeat_itself_failed_with(tmp_path, no_sleep):
    """death → granted repeat RELEASED: the record names `transport_unavailable`.
    The round's later free redial failing as a 400 leaves that stamp alone (the
    sticky kind moves on to `bad_request`): only the repeat's own failure writes
    the class, and the hint names it whichever terminal reads the record."""
    usage = {}
    msg, _cost = _primary_call(_ScriptedLLM(_death, _released_connect), tmp_path, usage, round_idx=1)
    assert msg is None and no_sleep == [4.0]
    assert usage[TRANSPORT_DEATHS_KEY] == {
        "round_id": f"{usage['execution_id']}:round:1", "count": 1, "backoff_sec": 4.0,
        "error_kind": "transport_unavailable",
    }

    redial = _ScriptedLLM(lambda: _status_failure(400))
    msg, _cost = _primary_call(redial, tmp_path, usage, round_idx=1)
    assert msg is None and redial.calls == 1  # fenced on the record, no burst
    assert usage["_last_llm_error_kind"] == "bad_request"  # the free redial's class is the sticky kind
    assert usage[TRANSPORT_DEATHS_KEY]["error_kind"] == "transport_unavailable"
    assert usage[TRANSPORT_DEATHS_KEY]["count"] == 1
    hint = loop_transport.provider_recovery_hint(usage)
    assert "the repeat failed as transport_unavailable" in hint
    assert "bad_request" not in hint


# ------------------------------------ the fence: nothing else after a granted repeat

@pytest.mark.parametrize("status,kind", [(400, "bad_request"), (503, "provider_transient")])
def test_repeat_failing_with_another_class_ends_the_round_on_the_unknown_terminal(
    tmp_path, monkeypatch, no_sleep, status, kind,
):
    """A granted repeat that fails with ANY other class does not re-open the
    same-model burst, the forced-final dial or the paid chain while attempt #1
    of the round is still unresolved: exactly two sends, the unknown no-resend
    terminal, a truthful row for the repeat, and a hint that names both facts."""
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", _no_chain)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "other/model")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    llm = _ScriptedLLM(_death, lambda: _status_failure(status), _death, _death)
    notes = []
    kwargs = _loop_kwargs(tmp_path, llm, notes)
    _presence_turn(kwargs)  # Inline Presence alone retains the bounded repeat rail.
    result, usage, trace = run_llm_loop(**kwargs)

    assert llm.calls == 2  # no burst, no forced-final dial, no chain candidate
    assert [(row["error_kind"], row["retry_same_request"]) for row in _events(tmp_path, "llm_api_error")] == [
        ("provider_outcome_unknown", True), (kind, False),
    ]
    non_retryable = _events(tmp_path, "llm_non_retryable_same_request")
    assert [row["error_kind"] for row in non_retryable] == [kind]
    assert trace.get("forced_finalization", {}).get("source") == "provider_outcome_unknown_no_resend"
    assert usage.get("execution_status") == "infra_failed"
    assert usage.get("reason_code") == "provider_unavailable"
    assert usage["_last_llm_error_kind"] == kind  # the sticky kind stays real
    assert RETRY_WALL_EXHAUSTED_KEY not in usage
    assert usage[TRANSPORT_DEATHS_KEY]["count"] == 1
    assert "1 earlier physical attempt(s) of the last dispatched round" in result
    assert f"the repeat failed as {kind}" in result
    assert "no further retry or paid fallback was sent" in result


def test_repeat_returning_an_empty_response_ends_the_round_and_keeps_the_record(tmp_path, monkeypatch, no_sleep):
    """The empty/incomplete-response retry is fenced the same way, and the
    record survives the response object (it clears only on a USABLE reply)."""
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", _no_chain)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "other/model")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    llm = _ScriptedLLM(_death, EMPTY_RESPONSE, _death, _death, _death)
    notes = []
    kwargs = _loop_kwargs(tmp_path, llm, notes)
    _presence_turn(kwargs)  # Inline Presence alone retains the bounded repeat rail.
    _result, usage, trace = run_llm_loop(**kwargs)

    assert llm.calls == 2  # the empty repeat is not retried; the third death never happens
    assert len(_events(tmp_path, "provider_incomplete_response")) == 1
    assert usage[TRANSPORT_DEATHS_KEY]["count"] == 1
    assert trace.get("forced_finalization", {}).get("source") == "provider_outcome_unknown_no_resend"
    assert usage.get("reason_code") == "provider_unavailable"


def test_no_call_rail_fences_on_the_round_record_whatever_the_sticky_kind():
    record = {"round_id": "r", "count": 1, "backoff_sec": 4.0}
    fenced = {"_last_llm_error_kind": "bad_request", TRANSPORT_DEATHS_KEY: record}
    assert provider_no_call_source(fenced, False) == ("provider_outcome_unknown_no_resend", False)
    routable = SimpleNamespace(exact_model_route=False)
    assert loop_transport.fallback_chain_allowed(routable, "bad_request", None, fenced) is False
    assert loop_transport.fallback_chain_allowed(routable, "bad_request", None, {"_last_llm_error_kind": "bad_request"}) is True
    assert loop_transport.fallback_chain_allowed(routable, "bad_request", None) is True
    assert provider_no_call_source({"_last_llm_error_kind": "bad_request"}, False) == ("", False)
    assert provider_no_call_source({"_last_llm_error_kind": "provider_outcome_unknown"}, False) == (
        "provider_outcome_unknown_no_resend", False,
    )


def test_stale_round_record_never_fences_a_later_round(tmp_path, no_sleep):
    """A record from an exhausted earlier round is dropped by the next round's
    invocation, so a later transient burst runs as the base contract says."""
    usage = {}
    _primary_call(_ScriptedLLM(_death, _death, _death), tmp_path, usage, round_idx=1)
    assert usage[TRANSPORT_DEATHS_KEY]["count"] == 2
    later = _ScriptedLLM(lambda: _status_failure(503), lambda: _status_failure(503))
    msg, _cost = _primary_call(later, tmp_path, usage, round_idx=3)
    assert msg == OK_RESPONSE[0]
    assert later.calls == 3  # the burst was NOT fenced
    assert TRANSPORT_DEATHS_KEY not in usage
    assert provider_no_call_source(usage, False) == ("", False)


def test_control_without_a_record_keeps_the_base_contracts(tmp_path, no_sleep):
    """No grant → no fence: a 400 stops as before and a transient burst keeps its
    budget, with `transport_death_retries=0` and with the default."""
    usage = {}
    bad = _ScriptedLLM(lambda: _status_failure(400))
    msg, _cost = call_llm_with_retry(bad, MESSAGES, "test-model", None, "low", 3, tmp_path, "t-ctl", 1, None, usage)
    assert msg is None and bad.calls == 1
    assert usage["_last_llm_error_kind"] == "bad_request"
    assert provider_no_call_source(usage, False) == ("", False)
    burst = _ScriptedLLM(lambda: _status_failure(503), lambda: _status_failure(503))
    msg, _cost = call_llm_with_retry(burst, MESSAGES, "test-model", None, "low", 3, tmp_path, "t-ctl", 2, None, usage)
    assert msg == OK_RESPONSE[0] and burst.calls == 3


def test_deadline_refusal_without_a_round_record_keeps_the_base_stamps(tmp_path, no_sleep):
    """The unknown-preserving refusal is scoped to a round that holds a record;
    a sticky unknown without one gets the base deadline stamps."""
    import time

    usage = {"_last_llm_error_kind": "provider_outcome_unknown"}
    llm = _ScriptedLLM(_death)
    msg, _cost = _primary_call(llm, tmp_path, usage, deadline_ts=time.time() - 1)
    assert msg is None and llm.calls == 0
    assert usage["_last_llm_error_kind"] == "deadline_exhausted"
    assert usage["reason_code"] == "deadline_exhausted"


def test_stop_path_without_a_record_stops_instead_of_crashing(tmp_path):
    ctx = call_mod._LlmErrorContext(
        task_id="t", task_type="task", execution_id="e", round_id="e:round:1", llm_call_id="c",
        round_idx=1, attempt=0, model="m", request_ref=None, drive_logs=tmp_path, event_queue=None,
        accumulated_usage={"_last_llm_error_kind": "provider_outcome_unknown"},
    )
    assert call_mod._stop_after_llm_error(ctx) is True
    assert RETRY_WALL_EXHAUSTED_KEY not in ctx.accumulated_usage


def test_native_subagent_child_primary_dispatch_opts_in(tmp_path, monkeypatch, no_sleep):
    """A native API subagent child runs the ordinary run_llm_loop: its rounds are
    primary rounds of its own loop and get the bounded transport-death rail."""
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", _no_chain)
    monkeypatch.setattr(loop_transport, "upstream_transport_reachable", lambda *a, **kw: {"kind": "upstream_http", "status_code": 200})
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", lambda *a: False)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    llm = _ScriptedLLM(_death, _death)
    kwargs = _loop_kwargs(tmp_path, llm, [])
    kwargs["tools"]._ctx.task_metadata = {"delegation_role": "subagent", "parent_task_id": "t-parent"}
    result, usage, _trace = run_llm_loop(**kwargs)

    assert result == "done"
    assert llm.calls == 3
    assert usage.get("reason_code") is None
    assert TRANSPORT_DEATHS_KEY not in usage
    assert no_sleep == []  # Managed attempts wait for upstream proof, never use the paid-repeat backoff.
    assert usage["transport_recovery"]["old_outcome"] == "unknown"


def _overflow_failure():
    """The repeat rejected as a context overflow (a structured provider code)."""
    exc = RuntimeError("upstream window is smaller than the routed payload")
    exc.code = "context_length_exceeded"
    exc.physical_attempt_capture = ua.PhysicalAttemptCapture(
        attempt_id="pa-overflow", model="test-model", provider="openrouter", state="unresolved",
        candidate_measurement_kind="opaque", provider_status_code=400, provider_code="context_length_exceeded",
    )
    return exc


def test_repeat_failing_as_context_overflow_takes_the_unknown_terminal(tmp_path, monkeypatch, no_sleep):
    """A granted repeat rejected as a context overflow must not open the
    compaction retry or the overflow-salvage terminal while attempt #1 of the
    round is unresolved: the unknown no-resend rail wins and the hint names
    both facts."""
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", _no_chain)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "other/model")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    llm = _ScriptedLLM(_death, _overflow_failure, _death, _death)
    notes = []
    kwargs = _loop_kwargs(tmp_path, llm, notes)
    _presence_turn(kwargs)  # Inline Presence alone retains the bounded repeat rail.
    result, usage, trace = run_llm_loop(**kwargs)

    assert llm.calls == 2
    assert usage["_last_llm_error_kind"] == "context_overflow"
    assert usage[TRANSPORT_DEATHS_KEY]["count"] == 1
    assert trace.get("forced_finalization", {}).get("source") == "provider_outcome_unknown_no_resend"
    assert usage.get("execution_status") == "infra_failed"
    assert usage.get("reason_code") == "provider_unavailable"
    assert "the repeat failed as context_overflow" in result
    assert "no further retry or paid fallback was sent" in result


def test_refused_free_redial_keeps_the_base_deadline_stamps(tmp_path, no_sleep):
    """death → granted repeat RELEASED → wait episode → free redial refused by
    the deadline: the sticky kind is `transport_unavailable`, so the refusal
    keeps the base `deadline_exhausted` stamps (the episode attributes the
    refusal), the record is not un-counted, and it still fences the no-call
    rail whatever the kind says."""
    import time

    usage = {}
    first = _ScriptedLLM(_death, _released_connect)
    msg, _cost = _primary_call(first, tmp_path, usage, round_idx=1)
    assert msg is None and first.calls == 2
    assert usage["_last_llm_error_kind"] == "transport_unavailable"
    assert usage[TRANSPORT_DEATHS_KEY]["count"] == 1

    redial = _ScriptedLLM(_death)
    msg, _cost = _primary_call(redial, tmp_path, usage, round_idx=1, deadline_ts=time.time() - 1)
    assert msg is None and redial.calls == 0
    assert usage["_last_llm_error_kind"] == "deadline_exhausted"
    assert usage["reason_code"] == "deadline_exhausted"
    assert usage[TRANSPORT_DEATHS_KEY]["count"] == 1
    assert provider_no_call_source(usage, True) == ("provider_outcome_unknown_no_resend", False)


def test_proxy_tunnel_failure_keeps_the_base_unknown_terminal(data_root, tmp_path, monkeypatch, no_sleep):
    """Through the REAL ledger and the round gate: a requests ProxyError whose
    nested error is a typed death (the tunnel to the proxy died) is neither the
    free pre-dispatch class nor a paid repeat — custody stays unresolved, the
    kind is `provider_outcome_unknown`, one attempt, no repeat, no episode."""
    import http.client

    import requests
    import urllib3

    def proxy_tunnel_death():
        return requests.exceptions.ProxyError(urllib3.exceptions.MaxRetryError(
            None, "/messages", reason=urllib3.exceptions.ProxyError(
                "Unable to connect to proxy", http.client.RemoteDisconnected("closed without response"),
            ),
        ))

    class _ProxyTunnelLLM:
        calls = 0

        def default_model(self):
            return "test-model"

        def chat(self, **_kwargs):
            self.calls += 1

            def send():
                raise proxy_tunnel_death()

            request = ua.AttemptRequest(
                model="test-model", provider="anthropic", reservation_usd=1.0,
                drive_root=data_root, task_id="t-proxy", root_task_id="t-proxy", source="test.proxy",
            )
            ua.execute_physical_attempt(request, send)
            return OK_RESPONSE

    llm = _ProxyTunnelLLM()
    usage = {}
    msg, _cost = _primary_call(llm, tmp_path, usage)

    assert msg is None
    assert llm.calls == 1
    assert [row["state"] for row in _ledger(data_root)] == ["unresolved"]
    assert ua.usage_projection(data_root)["unresolved_upper_bound_usd"] == 1.0
    assert usage["_last_llm_error_kind"] == "provider_outcome_unknown"
    assert usage["_last_llm_retry_same_request"] is False
    assert TRANSPORT_DEATHS_KEY not in usage
    assert no_sleep == []
    assert provider_no_call_source(usage, False) == ("provider_outcome_unknown_no_resend", False)

    # And through run_llm_loop: no wait episode, the base unknown no-resend terminal, no forced dial.
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", _no_chain)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)

    def scripted():
        exc = proxy_tunnel_death()
        exc.physical_attempt_capture = _capture(provider="anthropic")
        return exc

    loop_llm = _ScriptedLLM(scripted, scripted)
    kwargs = _presence_turn(_loop_kwargs(tmp_path, loop_llm, []))
    _result, loop_usage, trace = run_llm_loop(**kwargs)
    assert loop_llm.calls == 1
    assert _events(tmp_path, "network_wait") == []
    assert trace.get("forced_finalization", {}).get("source") == "provider_outcome_unknown_no_resend"


# ------------------------------------------- rejected complete wire: no unknown continuation

def test_rejected_stream_settles_and_walks_the_fallback_chain_without_a_wait_episode(
    data_root, tmp_path, monkeypatch, no_sleep,
):
    """A stream that reached its terminal frame but assembled into an unusable body is
    a deterministic local verdict, not an unknown provider outcome. Through the REAL
    ledger and the real round gate: the usage frame settles the primary attempt
    (``reserved, dispatched, settled``), the classifier files it as ``provider_error``
    with no same-model repeat, the configured cross-model fallback is dialed and answers,
    and neither a ``network_wait`` episode nor a ``[SYSTEM NOTICE]`` continuation exists."""
    from ouroboros import fallback_cooldown
    from ouroboros.llm_stream import RejectedProviderStream

    fallback_cooldown.reset_for_tests()
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "other/model")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    # The REAL chain runs; only the agent-built immutable context core (absent in a bare
    # loop) is stubbed, exactly as the chain's own tests do (test_resolved_model_target).
    monkeypatch.setattr(loop_mod, "_rebind_context_fit_plan", lambda *a, **k: (None, "max"))

    class _RejectingPrimaryLLM:
        """The primary's physical send raises the assembler's verdict WITH the usage it
        read; the fallback model answers — both through execute_physical_attempt."""

        def __init__(self, root):
            self.root = root
            self.models = []

        def default_model(self):
            return "test-model"

        def chat(self, **kwargs):
            model = kwargs["model"]
            self.models.append(model)

            def send():
                if model == "test-model":
                    raise RejectedProviderStream(
                        "Stream rejected after terminal framing: choice 0 tool call 0: function.arguments is not text",
                        usage={"prompt_tokens": 40, "completion_tokens": 9, "cost": 0.02},
                    )
                return {"content": "done"}

            request = ua.AttemptRequest(
                model=model, provider="openrouter", reservation_usd=1.0,
                drive_root=self.root, task_id="t-death", root_task_id="t-death", source="test.rejected",
            )
            ua.execute_physical_attempt(
                request, send, extractor=lambda _resp: ({"prompt_tokens": 1, "completion_tokens": 1}, 0.01, True),
            )
            return OK_RESPONSE

    llm = _RejectingPrimaryLLM(data_root)
    notes = []
    kwargs = _loop_kwargs(tmp_path, llm, notes)
    messages = kwargs["messages"]
    result, usage, _trace = run_llm_loop(**kwargs)

    assert result == "done"
    assert llm.models == ["test-model", "other/model"]  # one verdict, one fallback dial, no same-model repeat
    by_attempt = {}
    for row in _ledger(data_root):
        by_attempt.setdefault(row["attempt_id"], []).append(row)
    assert [[row["state"] for row in rows] for rows in by_attempt.values()] == [
        ["settled"],
        ["settled"],
    ]
    rejected = list(by_attempt.values())[0][-1]
    assert rejected["cost_usd"] == 0.02 and rejected["prompt_tokens"] == 40 and rejected["completion_tokens"] == 9
    assert ua.usage_projection(data_root)["unresolved_upper_bound_usd"] == 0.0
    api_errors = _events(tmp_path, "llm_api_error")
    assert [(row["error_kind"], row["retry_same_request"], row["attempt_custody_state"]) for row in api_errors] == [
        ("provider_error", False, "settled"),
    ]
    assert [row["error_kind"] for row in _events(tmp_path, "llm_non_retryable_same_request")] == ["provider_error"]
    assert _events(tmp_path, "network_wait") == []
    assert "transport_recovery" not in usage and "_pending_transport_outcome" not in usage
    assert TRANSPORT_DEATHS_KEY not in usage and usage.get("reason_code") is None
    for transcript in (messages, kwargs["tools"]._ctx.messages):
        assert not any("[SYSTEM NOTICE]" in str(m.get("content") or "") for m in transcript)
    assert any("⚡ Fallback: test-model → other/model" in note and "provider_error" in note for note in notes)


def test_code_less_stream_error_without_usage_keeps_the_upper_bound_and_still_walks_the_chain(
    data_root, tmp_path, monkeypatch, no_sleep,
):
    """An explicit SSE error frame without an HTTP-shaped code (the native overload shape) read
    before any usage frame is the provider's own terminal verdict: ``provider_error`` with no
    same-model repeat, the cross-model fallback dialed, no wait episode — while the first attempt
    keeps its unresolved upper bound (message_start's snapshot is deliberately not settled), so the
    budget fence still counts that money."""
    from ouroboros import fallback_cooldown
    from ouroboros.llm_stream import ProviderStreamError

    fallback_cooldown.reset_for_tests()
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "other/model")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    monkeypatch.setattr(loop_mod, "_rebind_context_fit_plan", lambda *a, **k: (None, "max"))

    class _OverloadedPrimaryLLM:
        def __init__(self, root):
            self.root = root
            self.models = []

        def default_model(self):
            return "test-model"

        def chat(self, **kwargs):
            model = kwargs["model"]
            self.models.append(model)

            def send():
                if model == "test-model":
                    raise ProviderStreamError({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})
                return {"content": "done"}

            request = ua.AttemptRequest(
                model=model, provider="anthropic", reservation_usd=1.0,
                drive_root=self.root, task_id="t-death", root_task_id="t-death", source="test.overloaded",
            )
            ua.execute_physical_attempt(
                request, send, extractor=lambda _resp: ({"prompt_tokens": 1, "completion_tokens": 1}, 0.01, True),
            )
            return OK_RESPONSE

    llm = _OverloadedPrimaryLLM(data_root)
    notes = []
    kwargs = _loop_kwargs(tmp_path, llm, notes)
    result, usage, _trace = run_llm_loop(**kwargs)

    assert result == "done"
    assert llm.models == ["test-model", "other/model"]
    by_attempt = {}
    for row in _ledger(data_root):
        by_attempt.setdefault(row["attempt_id"], []).append(row)
    assert [[row["state"] for row in rows] for rows in by_attempt.values()] == [
        ["unresolved"],
        ["settled"],
    ]
    assert ua.usage_projection(data_root)["unresolved_upper_bound_usd"] == 1.0
    api_errors = _events(tmp_path, "llm_api_error")
    assert [(row["error_kind"], row["retry_same_request"], row["attempt_custody_state"]) for row in api_errors] == [
        ("provider_error", False, "unresolved"),
    ]
    assert _events(tmp_path, "network_wait") == []
    assert "transport_recovery" not in usage and "_pending_transport_outcome" not in usage
