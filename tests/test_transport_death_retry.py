"""Current transport recovery and the remaining custody of historical round checkpoints."""

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
    call_llm_with_retry,
    classify_llm_exception,
    provider_no_call_source,
)
from ouroboros.tools.registry import ToolRegistry
from tests._usage_store_testing import ledger_rows
from tests._shared import saved_settings
from tests._transport_death_fixtures import MESSAGES, OK_RESPONSE, _ScriptedLLM, _primary_call


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


def test_budget_refusal_on_the_second_send_propagates_untouched(data_root, tmp_path, no_sleep, monkeypatch):
    """A restored round still consults the wallet before its first ordinary dispatch."""
    monkeypatch.setenv("TOTAL_BUDGET", "1.5")
    llm, usage = _LedgerLLM(data_root, lambda: httpx.ReadError("socket closed")), {"execution_id": "restored"}
    assert _primary_call(llm, tmp_path, usage)[0] is None
    assert [row["state"] for row in _ledger(data_root)] == ["unresolved"]
    # The old writer stored this pair independently of the pending outcome.
    usage[TRANSPORT_DEATHS_KEY] = {"round_id": "restored:round:1", "count": 1}
    ua.record_subscription_session("other-session", drive_root=data_root, route="subscription",
                                   task_id="other", root_task_id="other", spend_usd=1.5)
    with pytest.raises(ua.BudgetExceeded) as raised:
        _primary_call(llm, tmp_path, usage)
    assert raised.value.limit_scope == "global"
    assert [row["state"] for row in _ledger(data_root)] == ["unresolved"]
    assert usage[TRANSPORT_DEATHS_KEY]["count"] == 1


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("ending", ["unknown", "bad_request", "transient", "empty", "overflow", "success", "deadline"])
def test_legacy_count_permits_the_original_first_call_only(tmp_path, no_sleep, count, ending):
    """The original {round_id,count} has no operation identity; the ordinary call is the same on either tree."""
    usage = {"execution_id": "saved", "_last_llm_error_kind": "provider_outcome_unknown",
             TRANSPORT_DEATHS_KEY: {"round_id": "saved:round:1", "count": count}}
    llm = _ScriptedLLM({"unknown": _death, "bad_request": lambda: _status_failure(400),
                        "transient": lambda: _status_failure(503), "empty": EMPTY_RESPONSE,
                        "overflow": _overflow_failure, "success": OK_RESPONSE, "deadline": OK_RESPONSE}[ending])
    reply, _ = _primary_call(llm, tmp_path, usage, deadline_ts=1 if ending == "deadline" else None)
    assert llm.calls == (0 if ending == "deadline" else 1)
    assert (reply is not None) == (ending == "success")
    assert no_sleep == []
    if ending == "success":
        assert TRANSPORT_DEATHS_KEY not in usage
    else:
        assert usage[TRANSPORT_DEATHS_KEY]["count"] == count
        assert provider_no_call_source(usage, False)[0] == "provider_outcome_unknown_no_resend"
        if ending != "unknown":
            assert not usage.get("_pending_transport_outcome")
        if ending == "deadline":
            # A round that holds a record keeps its unknown kind through the deadline refusal: no deadline stamps.
            assert usage["_last_llm_error_kind"] == "provider_outcome_unknown"
            assert usage["_last_llm_retry_same_request"] is False
            assert usage.get("reason_code") != "deadline_exhausted"
            assert len(_events(tmp_path, "llm_not_dispatched")) == 1
            assert provider_no_call_source(usage, True)[0] == "provider_outcome_unknown_no_resend"


def test_stale_legacy_record_leaves_the_next_round_its_normal_transient_retry(tmp_path, no_sleep):
    usage = {"execution_id": "saved", TRANSPORT_DEATHS_KEY: {"round_id": "saved:round:1", "count": 1}}
    llm = _ScriptedLLM(lambda: _status_failure(503), OK_RESPONSE)
    assert _primary_call(llm, tmp_path, usage, round_idx=2)[0] == OK_RESPONSE[0]
    assert llm.calls == 2 and TRANSPORT_DEATHS_KEY not in usage


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
    """Stamp the production inline Presence caller class."""
    kwargs["task_type"] = "presence"
    kwargs["tools"]._ctx.is_direct_chat = True
    kwargs["tools"]._ctx.current_task_type = "presence"
    return kwargs


def test_presence_unknown_wait_exhaustion_never_forces_another_generation(tmp_path, monkeypatch, no_sleep):
    """A spent interactive wait keeps the original unknown attempt and dispatches no summary."""
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    saved_settings(monkeypatch, tmp_path, OUROBOROS_MODEL_FALLBACKS="")  # a saved blank: no chain
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    monkeypatch.setattr(loop_transport, "get_task_idle_timeout_sec", lambda: 0)
    monkeypatch.setattr(loop_transport, "upstream_transport_reachable", lambda *_a, **_k: pytest.fail("spent wait"))
    llm = _ScriptedLLM(_death, _death)
    result, usage, trace = run_llm_loop(**_presence_turn(_loop_kwargs(tmp_path, llm, [])))
    assert llm.calls == 1 and no_sleep == [] and TRANSPORT_DEATHS_KEY not in usage
    assert usage["execution_status"] == "infra_failed" and usage["reason_code"] == "provider_unavailable"
    assert trace["forced_finalization"]["source"] == "provider_outcome_unknown_no_resend"
    assert "no terminal provider outcome" in result.lower()



def test_primary_round_dispatch_recovers_after_two_deaths(tmp_path, monkeypatch, no_sleep):
    saved_settings(monkeypatch, tmp_path, OUROBOROS_MODEL_FALLBACKS="")  # a saved blank: no chain
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


@pytest.mark.parametrize("attempt_cap", [None, 2])
def test_round_dispatcher_leaves_unknown_recovery_to_the_shared_round(tmp_path, no_sleep, attempt_cap):
    """Presence primary and fallback calls both return unknown to the shared recovery owner."""
    from copy import deepcopy
    llm = _ScriptedLLM(_death, _death)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    ctx = SimpleNamespace(
        llm=llm, messages=deepcopy(MESSAGES), tools=registry, active_model="test-model", tool_schemas=[],
        context_fit_plan=None, active_context_mode="max", drive_root=tmp_path,
        active_effort="low", max_retries=3, drive_logs=tmp_path, task_id="t-cand", round_idx=1,
        event_queue=None, accumulated_usage={}, task_type="presence", active_use_local=False,
        physical_send_claim=None,
    )
    msg, _cost = loop_mod._dispatch_round_model(ctx, None, attempt_cap=attempt_cap)
    assert msg is None and llm.calls == 1 and no_sleep == []
    assert TRANSPORT_DEATHS_KEY not in ctx.accumulated_usage


def test_direct_call_returns_a_lost_response_without_resending(tmp_path, no_sleep):
    llm = _ScriptedLLM(_death, _death)
    msg, _cost = call_llm_with_retry(llm, MESSAGES, "test-model", None, "low", 3, tmp_path, "t-default", 1, None, {})
    assert msg is None
    assert llm.calls == 1


def test_classifier_and_review_custody_are_unchanged_by_the_rail():
    """The global classifier still says no-resend for a dispatched death, and
    review custody (which consults it) keeps refusing a second paid send."""
    from ouroboros.review_custody import retryable_review_exception

    exc = _death()
    classification = classify_llm_exception(exc)
    assert classification.kind == "provider_outcome_unknown"
    assert classification.retry_same_request is False
    assert retryable_review_exception(exc, None) is False


def test_recovery_hint_does_not_infer_dispatch_from_an_old_count():
    usage = {"_last_llm_error_kind": "provider_outcome_unknown", TRANSPORT_DEATHS_KEY: {"round_id": "r", "count": 2}}
    hint = loop_transport.provider_recovery_hint(usage)
    assert "no confirmed provider outcome" in hint
    assert "permits no further retry or paid fallback" in hint
    mixed = loop_transport.provider_recovery_hint({"_last_llm_error_kind": "bad_request", TRANSPORT_DEATHS_KEY: {"round_id": "r", "count": 1}})
    assert mixed == hint
    bare = loop_transport.provider_recovery_hint({"_last_llm_error_kind": "provider_outcome_unknown"})
    assert "no retry or paid fallback was sent" in bare


def test_recovery_hint_names_the_class_on_the_record_over_the_sticky_kind():
    """The record carries the class the repeat itself failed with; the sticky kind
    may by then belong to a later free redial (or a refusal) of the same round.
    A count without a failure stamp cannot attribute an outcome."""
    record = {"round_id": "r", "count": 1, "backoff_sec": 4.0, "error_kind": "transport_unavailable"}
    released = loop_transport.provider_recovery_hint({"_last_llm_error_kind": "bad_request", TRANSPORT_DEATHS_KEY: record})
    assert "checkpoint records transport_unavailable" in released
    assert "bad_request" not in released
    exhausted = loop_transport.provider_recovery_hint({
        "_last_llm_error_kind": "deadline_exhausted",
        TRANSPORT_DEATHS_KEY: {**record, "count": 2, "error_kind": "provider_outcome_unknown"},
    })
    assert "checkpoint records provider_outcome_unknown" in exhausted and "deadline_exhausted" not in exhausted
    unstamped = loop_transport.provider_recovery_hint({"_last_llm_error_kind": "bad_request", TRANSPORT_DEATHS_KEY: {"round_id": "r", "count": 1}})
    assert "bad_request" not in unstamped


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


def test_control_without_a_record_keeps_the_base_contracts(tmp_path, no_sleep):
    """Without a legacy record a 400 stops and a transient burst keeps its ordinary budget."""
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
    """A native API child uses the same paced unknown recovery as every turn."""
    saved_settings(monkeypatch, tmp_path, OUROBOROS_MODEL_FALLBACKS="")  # a saved blank: no chain
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

    # The ordinary loop can recover this unknown; an exhausted wait still sends nothing.
    saved_settings(monkeypatch, tmp_path, OUROBOROS_MODEL_FALLBACKS="")  # a saved blank: no chain
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", _no_chain)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)

    def scripted():
        exc = proxy_tunnel_death()
        exc.physical_attempt_capture = _capture(provider="anthropic")
        return exc

    monkeypatch.setattr(loop_transport, "get_task_idle_timeout_sec", lambda: 0)
    loop_llm = _ScriptedLLM(scripted, scripted)
    kwargs = _presence_turn(_loop_kwargs(tmp_path, loop_llm, []))
    _result, loop_usage, trace = run_llm_loop(**kwargs)
    assert loop_llm.calls == 1
    assert [row["phase"] for row in _events(tmp_path, "network_wait")] == ["entered", "ended"]
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
    # The host's own verdict on the stream is no provider sentence: the class alone is named.
    assert "⚡ Fallback: trying other/model after a provider error on test-model." in notes
    assert "⚡ Fallback: other/model is now the active route; it answered." in notes


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
