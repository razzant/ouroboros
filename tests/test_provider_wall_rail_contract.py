"""Contract tests for the no-repay provider rail (OB-01), pinned against the
accumulated_usage stamps the transport ACTUALLY leaves behind (reason_code
"llm_api_error" / "llm_empty_response"), not a bare marker dict:

1. A wall-exhausted provider death must terminalize with the typed
   ``reason_code="provider_unavailable"`` + ``infra_failed`` pair — exactly what
   fires the supervisor's "provider outage — NOT completed" owner notice
   (supervisor/events.py keys on that reason_code).
2. A scheduled Presence handoff remains addressable through its work_ref,
   while failure of the current turn retains its honest infrastructure outcome.
"""
import json
import time
from types import SimpleNamespace

import pytest

import ouroboros.loop as loop_mod
from ouroboros.loop_llm_call import RETRY_WALL_EXHAUSTED_KEY, call_llm_with_retry
from tests._processing_transport_fixture import physical_transport as physical_transport


class _TransientFailingLLM:
    def chat(self, **kwargs):
        error = Exception("upstream 502")
        error.status_code = 502
        raise error


def _rail_ctx(tmp_path, accumulated, tools=None):
    kwargs = dict(
        messages=[
            {"role": "user", "content": "do the thing"},
            {"role": "assistant", "content": "PARTIAL RESULT."},
        ],
        llm=SimpleNamespace(), active_model="openai/gpt-5.5", active_effort="medium",
        max_retries=3, drive_logs=tmp_path, task_id="task-wall-contract",
        round_idx=7, event_queue=None, accumulated_usage=accumulated,
        task_type="task", active_use_local=False, max_rounds=200,
        drive_root=tmp_path,
    )
    if tools is not None:
        kwargs["tools"] = tools
    return loop_mod._RoundLimitContext(**kwargs)


def test_wall_exhausted_rail_carries_the_typed_provider_unavailable_reason(
    tmp_path, monkeypatch,
):
    """Use the REAL transport stamps: after a genuinely exhausted transient wall,
    accumulated carries reason_code="llm_api_error". The no-repay rail must still
    terminalize with the typed provider_unavailable + infra_failed pair, or the
    supervisor's provider-death owner notice never fires for the most common
    provider-death shape (transient retries exhausted)."""
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    accumulated = {}
    msg, _ = call_llm_with_retry(
        _TransientFailingLLM(), [{"role": "user", "content": "hi"}],
        "openai/gpt-5.5", None, "medium", 2, tmp_path, "task-wall-contract", 1,
        None, accumulated, "task", False,
    )
    assert msg is None and accumulated[RETRY_WALL_EXHAUSTED_KEY] is True
    monkeypatch.setattr(
        loop_mod, "_call_forced_model_once",
        lambda ctx: (_ for _ in ()).throw(AssertionError("no forced call")),
    )

    _text, usage, trace = loop_mod._handle_provider_unavailable(_rail_ctx(tmp_path, accumulated))

    assert trace["forced_finalization"]["source"] == "retry_wall_exhausted_no_repay"
    assert usage["execution_status"] == "infra_failed"
    assert usage["reason_code"] == "provider_unavailable"  # owner-notice trigger


def test_wall_exhausted_body_429_empty_is_infra_not_a_model_failure(tmp_path, monkeypatch):
    """The empty-response wall shape (body-429 with finish_reason="stop") leaves
    execution_status="failed"/reason_code="llm_empty_response" in accumulated; the
    no-repay rail must upgrade it to the typed infra pair — a rate-limited-out
    provider is an outage, not a model that answered badly."""
    monkeypatch.setattr(
        loop_mod, "_call_forced_model_once",
        lambda ctx: (_ for _ in ()).throw(AssertionError("no forced call")),
    )
    accumulated = {
        RETRY_WALL_EXHAUSTED_KEY: True,
        "execution_status": "failed",
        "reason_code": "llm_empty_response",
        "_last_llm_error_kind": "rate_limit",
    }

    _text, usage, _trace = loop_mod._handle_provider_unavailable(_rail_ctx(tmp_path, accumulated))

    assert usage["execution_status"] == "infra_failed"
    assert usage["reason_code"] == "provider_unavailable"


def test_scheduled_presence_handoff_survives_the_no_call_rail(tmp_path):
    """Presence keeps the admitted child without sending host-salvaged text."""
    from ouroboros.presence_runner import build_presence_result_event
    tools_ctx = SimpleNamespace(
        task_metadata={"presence": {"binding_id": "presence-binding"}},
        _presence_completion={"outcome": "deferred"},
        _swarm_handoff_attempt={"status": "scheduled", "task_id": "t-child-1"},
    )
    accumulated = {
        "_last_llm_error_kind": "provider_outcome_unknown",
        "execution_status": "infra_failed",
        "reason_code": "llm_api_error",
    }

    text, usage, _trace = loop_mod._handle_provider_unavailable(
        _rail_ctx(tmp_path, accumulated, tools=SimpleNamespace(_ctx=tools_ctx)),
    )

    assert text == "PARTIAL RESULT."
    assert usage["execution_status"] == "infra_failed"
    assert usage["reason_code"] == "provider_unavailable"
    terminal = build_presence_result_event(
        {"id": "presence-turn"}, text, tools_ctx,
        terminal_origin=usage["terminal_origin"],
        retain_scheduled_handoff=True,
    )
    assert terminal["work_ref"] == "t-child-1"
    assert terminal["outcome"] == "deferred"  # admitted work retains its polling custody after the parent failure
    assert usage["terminal_origin"] == "host_salvage"
    assert terminal["text"] == ""


@pytest.mark.parametrize("scenario,blocked", [
    ("same", True), ("all_dialects", True), ("different_repair_error", True),
    ("unsupported_value_same", True), ("unsupported_value_changed", False),
    ("required_option", True), ("required_option_new_tools", True), ("absent_option", False),
    ("acting_refused", True), ("only_spare_refused", False),
    ("user_tail", False), ("nested_param", False), ("unknown_400", False), ("other_model", False),
    ("other_account", False), ("other_endpoint", False), ("different_tools", False),
    ("removed_tools", False), ("new_tool_choice_scope", False), ("previous_round", False), ("429", False),
])
def test_forced_send_compares_the_acting_routes_refused_parameter(transport, monkeypatch, scenario, blocked):
    from tests._usage_store_testing import ledger_rows

    root, client, _ = transport
    usage, sent = {}, []
    ctx = _rail_ctx(root / "logs", usage)
    ctx.drive_logs.mkdir(exist_ok=True)
    # An explicit none sends one dialect. With nonzero effort the pre-existing dialect ladder
    # may repair the rejected parameter and return an answer before a forced call is needed.
    ctx.llm, ctx.active_model = client, "openai::fixture"
    repairing = scenario in {"all_dialects", "different_repair_error"}
    ctx.active_effort = "high" if repairing else "none"
    ctx.tool_schemas = [{"type": "function", "function": {"name": "inspect", "parameters": {"type": "object"}}}]
    route, resolve = {}, client._resolve_remote_target
    monkeypatch.setattr(client, "_resolve_remote_target", lambda model: {**resolve(model), **route})
    monkeypatch.setattr(loop_mod, "_server_web_allowed_by_task", lambda _: False)
    monkeypatch.setattr("ouroboros.llm_fallback.consume_stream", lambda response, **_: response)

    def create(**candidate):
        sent.append(candidate)
        if len(sent) == 1 or repairing:
            error = RuntimeError("Request refused")
            error.status_code = 429 if scenario == "429" else 400
            if scenario == "different_repair_error" and len(sent) > 1:
                error.code = "invalid_request"
                error.body = {"code": error.code}
                raise error
            error.code = ("invalid_request" if scenario == "unknown_400" else
                          "unsupported_value" if scenario.startswith("unsupported_value_") else "unsupported_parameter")
            param = {"user_tail": "messages", "nested_param": "tools[0]", "required_option": "model",
                     "required_option_new_tools": "model", "new_tool_choice_scope": "tool_choice",
                     "unsupported_value_same": "reasoning_effort", "unsupported_value_changed": "reasoning_effort",
                     "absent_option": "not_sent"}.get(scenario, "tools")
            error.body = {"code": error.code, "param": param}
            raise error
        return SimpleNamespace(model_dump=lambda: {"choices": [{"message": {"role": "assistant", "content": "done"},
            "finish_reason": "stop"}], "usage": {"prompt_tokens": 8, "completion_tokens": 1, "cost": 0}})
    monkeypatch.setattr(client, "_get_remote_client", lambda target: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    msg, _ = call_llm_with_retry(client, ctx.messages, ctx.active_model, ctx.tool_schemas, ctx.active_effort,
        1, ctx.drive_logs, ctx.task_id, ctx.round_idx, None, usage, ctx.task_type, attempt_cap=1)
    normal_count = len(sent)
    assert msg is None
    if repairing:
        assert 2 <= normal_count <= 3
        assert len({(row["tools"][0]["type"], row.get("reasoning_effort")) for row in sent}) == normal_count
    else:
        assert normal_count == 1
    failure = usage["_last_llm_failure"]
    if scenario == "429":
        assert not failure.get("refused_option_bindings")  # its normal rate-limit policy owns any later attempt
        return
    assert failure["failure_code"] == "bad_request"
    if scenario == "different_repair_error":
        assert normal_count == 2 and failure["provider_code"] != "unsupported_parameter"
        assert len(failure["refused_option_bindings"]) == 1  # from the first send, not invented for the second
    elif scenario == "all_dialects":
        # Switching only the effort does not mint a new tools value.
        assert len(failure["refused_option_bindings"]) == len({json.dumps(row["tools"], sort_keys=True) for row in sent})
    else:
        ignored = scenario in {"user_tail", "nested_param", "absent_option", "unknown_400"}
        assert len(failure.get("refused_option_bindings", ())) == (0 if ignored else 1)
    if scenario in {"acting_refused", "only_spare_refused"}:
        other = {**failure, "refused_option_bindings": [], "model": "other"}
        usage["_acting_route_failure"], usage["_last_llm_failure"] = (
            (failure, other) if scenario == "acting_refused" else (other, failure))
    if scenario == "previous_round":
        ctx.round_idx += 1
    if scenario == "other_model":
        ctx.active_model = "openai::other"
    if scenario == "other_account":
        route["api_key"] = "a-different-fixture-key"
    if scenario == "other_endpoint":
        route["base_url"] = "https://another-endpoint.invalid/v1"
    if scenario in {"different_tools", "required_option_new_tools", "new_tool_choice_scope"}:
        ctx.tool_schemas = [{"type": "function", "function": {"name": "read_summary", "parameters": {"type": "object"}}}]
    if scenario == "removed_tools":
        ctx.tool_schemas = None
    if scenario == "unsupported_value_changed":
        ctx.active_effort = "low"
    prior = {key: usage[key] for key in ("_acting_route_failure", "_last_llm_failure") if key in usage}
    _, terminal, trace = loop_mod._handle_provider_unavailable(ctx, error_kind="bad_request")
    assert len(sent) == len(ledger_rows(root)) == normal_count + (0 if blocked else 1)
    assert terminal["execution_status"] == "infra_failed"
    if blocked:
        assert all(terminal[key] is value for key, value in prior.items())
        assert trace["forced_finalization"]["source"] == "bad_request_no_resend"
    else:
        assert sent[-1]["messages"] != sent[0]["messages"]
        if scenario == "unsupported_value_changed":
            assert [request["reasoning_effort"] for request in sent] == ["none", "low"]
        if scenario in {"different_tools", "new_tool_choice_scope"}:
            assert sent[-1]["tools"] != sent[0]["tools"] and sent[-1]["tool_choice"] == sent[0]["tool_choice"]
