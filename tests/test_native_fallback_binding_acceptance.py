"""Independent #1407/#1409 native request proofs, proposed for the frozen candidate.

Uses existing main_call/setup fixtures and the real loop, rebind, LLMClient,
Claudexor serializer and physical ledger. No provider calls. Not run by the scout.
"""

from copy import deepcopy
from dataclasses import replace
import json
import queue
from types import SimpleNamespace

import pytest
import httpx

from ouroboros import context, fallback_cooldown, llm_claudexor, loop
from ouroboros import loop_transport
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from ouroboros.model_slots import MODEL_ACCOUNTS_KEY
from ouroboros.model_wait import ModelWaitInterrupted
from ouroboros.tools.control_runtime import _switch_model
from tests._usage_store_testing import attempt_rows_in_start_order
from tests.test_llm_claudexor import MODEL, ROUTE, ledger, result
from tests.test_llm_claudexor import setup as gateway_fixture
from tests.test_model_wait import live_wait as wait_fixture
from tests.test_model_wait_controls import _loop_tools
from tests.test_subscription_main_wait import main_call as main_call_fixture

setup = gateway_fixture
live_wait = wait_fixture
main_call = main_call_fixture
ROUTE_B = {**ROUTE, "credentialProfileId": "account-b", "accountFingerprint": "fingerprint-b"}
ROUTE_C = {**ROUTE, "credentialProfileId": "account-c", "accountFingerprint": "fingerprint-c"}


def _reply(route, message):
    row = result(route=route)
    row["message"] = message
    return row


def _bounded_creates(monkeypatch, gateway, maximum):
    create = gateway.create_model_operation

    def checked(*args, **kwargs):
        if len(gateway.creates) >= maximum:
            pytest.fail("unexpected extra model operation; the test must not retry into success")
        return create(*args, **kwargs)

    monkeypatch.setattr(gateway, "create_model_operation", checked)


@pytest.mark.parametrize("role,direct", [("main", False), ("main", True), ("consciousness", True)])
@pytest.mark.parametrize("primary_account", ["account-a", ""], ids=["pinned-A", "Auto"])
def test_native_loop_fallback_then_symbolic_return_uses_real_role_account_and_fit(
        main_call, monkeypatch, role, direct, primary_account):
    """A -> B -> primary: equal model names do not make equal account/role bindings.

    The fallback MODEL is deliberately identical to the primary MODEL. A wrong
    fallback:0 role on return would send pin B, and a global-Main return from
    consciousness would send the decoy model/account. Both fail physical assertions.
    """
    ctx, gateway, owner, events, _decide, _observations = main_call
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "max")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "4")
    monkeypatch.setenv("MCP_ENABLED", "false")
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "inline")
    monkeypatch.setenv("OUROBOROS_MODEL", MODEL if role == "main" else "claudexor::codex=decoy-main")
    monkeypatch.setenv("OUROBOROS_MODEL_CONSCIOUSNESS", MODEL)
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", MODEL)
    monkeypatch.delenv("USE_LOCAL_MAIN", raising=False)
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    pins = {"main": "decoy-main-pin", "consciousness": "decoy-consciousness-pin", "fallback": ["account-b"]}
    pins[role] = primary_account
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps(pins))
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_args: False)
    monkeypatch.setattr(loop_transport, "upstream_transport_reachable",
                        lambda *_args, **_kwargs: pytest.fail("healthy configured routes require no recovery probe"))

    # Metadata is the only route observation stub. _rebind_context_fit_plan,
    # task_model_binding and every physical preparation/send stay real.
    def route_evidence(task, **_kwargs):
        actual_role = task["model_role"]
        assert actual_role in (role, "fallback:0")
        assert task["model"] == MODEL and not task["use_local_model"]
        observed = task.get("model_route") or {}
        explicit = task.get("credential_profile_id")
        configured = "account-b" if actual_role == "fallback:0" else primary_account
        profile = (explicit if explicit is not None else configured) or observed.get("credentialProfileId") or "account-c"
        window = 240_000 if profile == "account-b" else 900_000
        return ({"model": MODEL, "provider": "claudexor"}, SimpleNamespace(
            route_fp=f"capacity-{profile}", status="confirmed", stale=False, window_tokens=window,
            source_id="codex", source="advertised", credential_profile_id=profile,
            account_fingerprint=f"fingerprint-{profile.removeprefix('account-')}"))

    monkeypatch.setattr(context, "_context_fit_route", route_evidence)
    ctx.context_fit_plan = replace(ctx.context_fit_plan, model_role=role, model_route=deepcopy(ROUTE))
    tools = _loop_tools(ctx, owner)
    tools._ctx.task_metadata = {"model_role": role}
    tools._ctx.is_direct_chat = direct
    tools._ctx.model_wait_context = owner
    owner.worker_slot_held = not direct
    original_tools = deepcopy([row for row in ctx.messages if row.get("role") == "tool"])
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}}
    ctx.messages.append({"role": "user", "content": [{"type": "text", "text": "Keep the diagram"}, image]})
    return_call = {"id": "return-primary", "type": "function", "function": {
        "name": "switch_model", "arguments": json.dumps({"primary": "return"})}}
    answered_route = ROUTE if primary_account else ROUTE_C
    gateway.results = [
        result(outcome="unknown", route=ROUTE),
        _reply(ROUTE_B, {"role": "assistant", "content": "Returning to the original role.",
                         "tool_calls": [return_call]}),
        _reply(answered_route, {"role": "assistant", "content": "Primary answered after return."}),
    ]
    gateway.dispatch = ["unknown", "response_received", "response_received"]
    _bounded_creates(monkeypatch, gateway, 3)

    text, usage, trace = loop.run_llm_loop(
        ctx.messages, tools, ctx.llm, ctx.drive_logs, lambda *_args, **_kwargs: None,
        queue.Queue(), task_id="task-one", drive_root=ctx.drive_root, event_queue=events,
        initial_effort="high",
    )

    assert text == "Primary answered after return."
    assert len(gateway.uploads) == len(gateway.creates) == len(gateway.accepted_operations) == 3
    payloads = [payload for payload, _key in gateway.uploads]
    assert all(payload["model"] == "exact-model" for payload in payloads)
    assert payloads[1]["account"] == {"mode": "pin", "profileId": "account-b"}
    for index in (0, 2):
        account = payloads[index]["account"]
        if primary_account:
            assert account == {"mode": "pin", "profileId": "account-a"}
        else:
            assert account["mode"] == "auto" and "profileId" not in account
    # A returned Auto account is observation, not permission to convert Auto to pin C.
    assert tools._ctx.context_fit_plan.model_role == role
    assert tools._ctx.context_fit_plan.model_route["credentialProfileId"] == answered_route["credentialProfileId"]
    assert tools._ctx.primary_route == {"model": MODEL, "use_local": False, "role": role}
    request_manifests = [json.loads((ctx.drive_root / "observability" / "calls" / "task-one" /
                         f"{attempt}_model_request.json").read_text(encoding="utf-8"))
                         for attempt in gateway.creates]
    assert [manifest["model_role"] for manifest in request_manifests] == [role, "fallback:0", role]
    # One current row per attempt, in start order; each keeps its dispatch context.
    rows = attempt_rows_in_start_order(ctx.drive_root)
    assert [(row["state"], row["revision"]) for row in rows] == [("unresolved", 4), ("settled", 3), ("settled", 3)]
    assert rows[0]["physical_failure"]["stage"] == "raised_exception"
    dispatched = rows
    assert len({row["attempt_id"] for row in dispatched}) == 3
    assert [row["physical_context"]["capacity_total_tokens"] for row in dispatched] == [900_000, 240_000, 900_000]
    assert [row["physical_context"]["route_fp"] for row in dispatched] == [
        "capacity-account-a", "capacity-account-b", f"capacity-{answered_route['credentialProfileId']}"]
    assert usage["transport_recovery"]["previous_attempt"]["physical_attempt_id"] == dispatched[0]["attempt_id"]
    for payload in payloads:
        sent = payload["messages"]
        for completed in original_tools:
            matches = [row for row in sent if row.get("role") == "tool" and
                       row.get("tool_call_id") == completed["tool_call_id"]]
            assert len(matches) == 1 and matches[0]["content"] == completed["content"]
        assert any(image in row.get("content", []) for row in sent if isinstance(row.get("content"), list))
    returned_tool = [row for row in payloads[2]["messages"] if row.get("role") == "tool" and
                     row.get("tool_call_id") == "return-primary"]
    assert len(returned_tool) == 1 and "OK: switching to primary route" in returned_tool[0]["content"]
    assert [call["tool"] for call in trace["tool_calls"]] == ["switch_model"]


@pytest.mark.parametrize("direct", [False, True], ids=["queued", "direct"])
def test_accepted_get_404_recovers_same_native_operation_for_both_turns(live_wait, monkeypatch, direct):
    root, gateway, client, owner, _events, _decide = live_wait
    owner.tool_context = SimpleNamespace(task_id="task-one", is_direct_chat=direct, task_metadata={})
    owner.worker_slot_held = not direct
    gateway.pending = True
    original_get = gateway.get_model_operation
    observed_reads = []

    def read(operation_id, **kwargs):
        observed_reads.append(operation_id)
        if len(observed_reads) <= 2:
            raise ClaudexorUnavailable("http_404", "accepted operation temporarily not readable", status_code=404)
        gateway.pending = False
        return original_get(operation_id, **kwargs)

    monkeypatch.setattr(gateway, "get_model_operation", read)
    monkeypatch.setattr(llm_claudexor, "read_owned_gateway", lambda: gateway)
    now = [0.0]
    monkeypatch.setattr(llm_claudexor, "time", SimpleNamespace(
        monotonic=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0] + seconds)))
    _bounded_creates(monkeypatch, gateway, 1)
    message, usage = client.chat([{"role": "user", "content": "recover this response"}], MODEL,
                                 model_role="main", timeout=1)
    assert message == result()["message"]
    assert observed_reads == ["op-0"] * 3
    assert len(gateway.uploads) == len(gateway.creates) == len(gateway.accepted_operations) == 1
    assert [(row["state"], row["revision"]) for row in ledger(root)] == [("settled", 3)]
    assert len(usage["ledger_attempt_ids"]) == 1 and not gateway.cancels
    events = [json.loads(line) for line in (root / "logs/events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event.get("detail") == "same_model_operation_rejoined" for event in events)


def test_lost_create_reply_without_operation_id_cannot_authorize_new_generation(main_call, monkeypatch):
    """A generic create 4xx is not model_request_invalid: accepted custody may exist.

    Same-key rereads are allowed. After the second lost reply the scripted owner
    stops observation; neither another physical id nor a fallback send is allowed.
    An immediate honest recoverable-unknown terminal is also no-new-generation.
    """
    ctx, gateway, owner, events, _decide, _observations = main_call
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "max")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", MODEL)
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"main": "account-a", "fallback": ["account-b"]}))
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_args: False)
    monkeypatch.setattr(loop_transport, "upstream_transport_reachable",
                        lambda *_args, **_kwargs: pytest.fail("metadata cannot authorize a new generation"))
    tools = _loop_tools(ctx, owner)
    tools._ctx.is_direct_chat = True
    owner.worker_slot_held = False
    create = gateway.create_model_operation
    original_key = []

    def lost_reply(ref, *, idempotency_key, **kwargs):
        if original_key and idempotency_key != original_key[0]:
            pytest.fail("lost create reply caused a NEW physical id/fallback generation")
        original_key[:] = [idempotency_key]
        create(ref, idempotency_key=idempotency_key, **kwargs)  # engine accepted; host never sees its id
        raise ClaudexorUnavailable("http_404", "create reply lost after admission", status_code=404)

    monkeypatch.setattr(gateway, "create_model_operation", lost_reply)
    monkeypatch.setattr(llm_claudexor, "read_owned_gateway", lambda: gateway)
    owner.owner_control = lambda: "cancelled" if len(gateway.creates) >= 2 else None
    now = [0.0]
    monkeypatch.setattr(llm_claudexor, "time", SimpleNamespace(
        monotonic=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0] + seconds)))
    with pytest.raises(ModelWaitInterrupted, match="cancelled"):
        loop.run_llm_loop(
            ctx.messages, tools, ctx.llm, ctx.drive_logs, lambda *_args, **_kwargs: None,
            queue.Queue(), task_id="task-one", drive_root=ctx.drive_root, event_queue=events, initial_effort="high",
        )
    assert len(gateway.accepted_operations) == len(gateway.uploads) == 1
    assert set(gateway.creates) == {original_key[0]}
    assert gateway.uploads[0][0]["account"] == {"mode": "pin", "profileId": "account-a"}
    rows = ledger(ctx.drive_root)
    assert [(row["state"], row["revision"]) for row in rows] == [("unresolved", 4)]
    assert rows[0]["physical_failure"]["stage"] == "raised_exception"
    assert len({row["attempt_id"] for row in rows}) == 1
    assert not gateway.acks


def test_duplicate_model_slots_keep_their_distinct_native_accounts(main_call, monkeypatch):
    ctx, gateway, owner, events, _decide, _observations = main_call
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "max")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", f"{MODEL},{MODEL}")
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"main": "account-a", "fallback": ["account-b", "account-c"]}))
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_args: False)
    tools = _loop_tools(ctx, owner)
    gateway.results = [result(outcome="unknown", route=ROUTE), result(outcome="unknown", route=ROUTE_B),
                       _reply(ROUTE_C, {"role": "assistant", "content": "Third account answered."})]
    gateway.dispatch = ["unknown", "unknown", "response_received"]
    _bounded_creates(monkeypatch, gateway, 3)
    text, _usage, _trace = loop.run_llm_loop(
        ctx.messages, tools, ctx.llm, ctx.drive_logs, lambda *_args, **_kwargs: None,
        queue.Queue(), task_id="task-one", drive_root=ctx.drive_root, event_queue=events)
    assert text == "Third account answered."
    assert [payload["account"] for payload, _ in gateway.uploads] == [
        {"mode": "pin", "profileId": pin} for pin in ("account-a", "account-b", "account-c")]
    assert tools._ctx.context_fit_plan.model_role == "fallback:1"
    assert [row["state"] for row in ledger(ctx.drive_root)].count("unresolved") == 2


@pytest.mark.parametrize("status,reason", [(401, "auth"), (402, "quota"), (400, "")])
@pytest.mark.parametrize("wire", ["http", "sse"])
def test_primary_wait_honors_a_direct_api_access_refusal(main_call, monkeypatch, status, reason, wire):
    """A declared wait must not be undone by a paid fallback merely because access uses an API key."""
    from ouroboros import usage_accounting as ua
    from ouroboros.llm_attempt import _attempt_request, _candidate_before_dispatch

    ctx, gateway, owner, events, _decide, _observations = main_call
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", MODEL)
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-key")
    ctx.active_model = "openai::primary-test"
    ctx.context_fit_plan = replace(ctx.context_fit_plan, model=ctx.active_model, provider="openai", model_route={})
    ctx.tools._ctx.context_fit_plan = ctx.context_fit_plan
    ctx.tools._ctx.primary_route = {"model": ctx.active_model, "use_local": False, "role": "main"}
    ctx.tools._ctx.model_wait_context = owner
    _switch_model(ctx.tools._ctx, primary="wait")
    sent = []
    original_send = ctx.llm._chat_remote

    def send(target, messages, schemas, *args, **kwargs):
        if target["provider"] != "openai":
            return original_send(target, messages, schemas, *args, **kwargs)
        assert target["provider"] == "openai"
        sent.append(target["provider"])
        body = {"messages": deepcopy(messages), "tools": schemas or [], "model": "primary-test"}
        request = _attempt_request(target, body)

        def refused():
            if wire == "sse":
                from ouroboros.llm_stream import ChatAccumulator
                ChatAccumulator().accept("message", json.dumps({"error": {"code": status, "message": "Access refused"}}))
            raise httpx.HTTPStatusError("Access refused", request=httpx.Request("POST", "https://fixture.invalid"),
                                       response=httpx.Response(status))

        return ua.execute_physical_attempt(request, refused, before_dispatch=_candidate_before_dispatch(body, request))

    monkeypatch.setattr(ctx.llm, "_chat_remote", send)
    monkeypatch.setattr(ctx.llm, "claudexor_model_sources", lambda: pytest.fail("API wait has no subscription observer"))
    owner.owner_control = lambda: "cancelled" if any(row.get("state") == "waiting" for row in owner.waits.values()) else None
    if not reason:  # Invalid request is not unavailable access: ordinary fallback still works.
        from ouroboros.loop_model_call import _recover_failed_round
        assert loop._call_round_model(ctx)[0] is None
        gateway.results = [_reply(ROUTE_B, {"role": "assistant", "content": "Alternative accepted the request."})]
        gateway.dispatch = ["response_received"]
        fallback_cooldown.reset_for_tests()
        reply, *_ = _recover_failed_round(ctx, ctx.tools, None, None, context_fit_plan=ctx.context_fit_plan,
            active_context_mode="max", emit_progress=lambda *_a, **_k: None)
        assert reply["content"] == "Alternative accepted the request."
        assert not owner.waits and sent == ["openai"] and len(gateway.creates) == 1
        return
    with pytest.raises(ModelWaitInterrupted, match="cancelled"):
        loop._call_round_model(ctx)
    waits = [row for row in list(events.queue) if row.get("type") == "task_model_wait"]
    assert waits[0]["state"] == "waiting" and waits[0]["reason"] == reason
    assert waits[0]["availability_observation"] == "unavailable" and waits[0]["auto_continue"] is False
    assert waits[0]["credential_harness"] == "" and waits[-1]["resolution"] == "cancelled"
    assert sent == ["openai"] and gateway.creates == []


@pytest.mark.parametrize("primary_account,fallback_account", [("account-a", "account-b"), ("", "account-b"), ("account-a", "")])
def test_transient_cooldown_keeps_other_native_account_available(main_call, monkeypatch, primary_account, fallback_account):
    from ouroboros import loop_model_call

    ctx, gateway, owner, _events, _decide, _observations = main_call
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", MODEL)
    monkeypatch.setenv("OUROBOROS_FALLBACK_COOLDOWN_ENABLED", "true")
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"main": primary_account, "fallback": ["decoy"]}))
    owner.overrides["fallback:0"] = {"model_account_override": fallback_account}
    inner = ctx.tools._ctx
    inner.primary_route = {"model": MODEL, "use_local": False, "role": "main"}
    ctx.accumulated_usage["_last_llm_error_kind"] = "provider_transient"
    gateway.results = [_reply(ROUTE_B, {"role": "assistant", "content": "Other account answered."})]
    gateway.dispatch = ["response_received"]
    _bounded_creates(monkeypatch, gateway, 1)
    fallback_cooldown.reset_for_tests()
    try:
        reply, *_ = loop_model_call._run_cross_model_fallback_chain(
            llm=ctx.llm, ctx=inner, tools=ctx.tools, messages=ctx.messages, active_model=MODEL,
            active_use_local=False, tool_schemas=[], active_effort="high", max_retries=1,
            drive_logs=ctx.drive_logs, task_id=inner.task_id, round_idx=1, event_queue=ctx.event_queue,
            accumulated_usage=ctx.accumulated_usage, task_type="task", emit_progress=lambda *_a, **_k: None,
            context_fit_plan=ctx.context_fit_plan, active_context_mode="max")
        assert reply["content"] == "Other account answered."
        sent = gateway.uploads[0][0]["account"]
        if fallback_account:
            assert sent == {"mode": "pin", "profileId": fallback_account}
        else:
            assert sent["mode"] == "auto" and "profileId" not in sent
        assert fallback_cooldown.is_cooling_down(MODEL, False, primary_account)
        assert not fallback_cooldown.is_cooling_down(MODEL, False, fallback_account)
        # Resource-refusal deferral must use the same account key as the actual walk.
        assert loop_model_call._route_follows([(MODEL, "fallback:0", False, False)])
        assert not loop_model_call._route_follows([(MODEL, "main", False, True)])
    finally:
        fallback_cooldown.reset_for_tests()


@pytest.mark.parametrize("choice", ["return", "wait"])
@pytest.mark.parametrize("primary_account", ["account-a", ""])
def test_primary_choice_survives_shared_cold_continuation(main_call, monkeypatch, choice, primary_account):
    from ouroboros import budget_pause, loop_model_call, owner_wait
    from ouroboros.model_slots import route_binding

    ctx, _gateway, owner, _events, _decide, _observations = main_call
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"main": "decoy", "fallback": ["account-b"]}))
    owner.overrides["main"] = {"model_account_override": primary_account}
    inner = ctx.tools._ctx
    inner.model_wait_context = owner
    inner.active_effort, inner.active_use_local = "high", False
    inner.context_fit_plan = replace(ctx.context_fit_plan, model_role="fallback:0")
    inner.primary_route = {"model": MODEL, "use_local": False, "role": "main"}
    loop._reset_turn_state(inner)
    assert "OK:" in _switch_model(inner, primary=choice)
    inner._route_facts_pending = "dated refusal fact"
    state = owner_wait.continuation_state(inner, ctx.messages, {}, {}, 1, [], set())
    # JSON storage and a fresh turn can replace the initial role before restore.
    state = json.loads(json.dumps(state))
    loop._reset_turn_state(inner)
    inner.primary_route = {"model": "decoy", "use_local": False, "role": "fallback:0"}
    inner.budget_drive_root = inner.drive_root = ctx.drive_root
    state["_pause_row"] = {"pause_id": "primary-pause", "state": budget_pause.STATE_RESUME_GRANTED,
                           "rail": budget_pause.RAIL_GLOBAL_EXHAUSTED, "grant": {"grant_id": "owner-resume"}}
    budget_pause.set_budget_pause(ctx.drive_root, inner.task_id, state["_pause_row"])
    model, effort, local, mode, _round, restored_plan = budget_pause.resume_paused_loop(
        ctx.tools, state, ctx.messages, {}, {}, set(), budget_remaining_usd=5.0)
    assert budget_pause.budget_pause_row(ctx.drive_root, inner.task_id)["state"] == budget_pause.STATE_RESUMED
    assert any("explicit owner Resume" in str(row.get("content")) for row in ctx.messages)
    model, local, effort, plan, _mode = loop_model_call._apply_round_route_overrides(
        inner, ctx.tools, ctx.messages, (model, local, effort), restored_plan, mode, "max", [])
    assert plan.model_role == "main" and inner.route_wait_on_primary is (choice == "wait")
    assert route_binding(model, local, plan.model_role, overrides=owner.overrides) == (MODEL, False, primary_account)
    assert inner.primary_route == {"model": MODEL, "use_local": False, "role": "main"}
    assert inner._route_facts_pending == "dated refusal fact" and effort == "high"


@pytest.mark.parametrize("wire,status,ending", [
    (wire, status, ending) for wire in ("managed", "http", "sse")
    for status in (429, 503, 529) for ending in ("answer", "stop", "deadline")
] + [("managed", 503, "wrap")])
def test_primary_refusal_wait_paces_real_requests_without_catalog_recovery(
        main_call, monkeypatch, wire, status, ending):
    from datetime import datetime, timedelta, timezone
    from ouroboros import loop_llm_call, owner_mailbox, usage_accounting as ua
    from ouroboros.llm_attempt import _attempt_request, _candidate_before_dispatch
    from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN

    ctx, gateway, owner, events, _decide, _observations = main_call
    primary = MODEL if wire == "managed" else "openai::primary-test"
    for key, value in {"OUROBOROS_CONTEXT_MODE": "max", "OUROBOROS_TASK_REVIEW_MODE": "off",
                       "OUROBOROS_SAFETY_MODE": "off", "MCP_ENABLED": "false",
                       "OUROBOROS_TRANSIENT_RETRY_MAX": "6", "OUROBOROS_MODEL_FALLBACKS": MODEL,
                       "OPENAI_API_KEY": "fixture-key"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"main": "account-a", "fallback": ["account-b"]}))
    ctx.context_fit_plan = replace(ctx.context_fit_plan, model=primary,
        provider="claudexor" if wire == "managed" else "openai", model_route=ROUTE if wire == "managed" else {})
    tools = _loop_tools(ctx, owner)
    tools._ctx.task_model_override, tools._ctx.is_direct_chat = primary, ending != "wrap"
    tools._ctx.model_wait_context = owner
    owner.worker_slot_held = ending == "wrap"
    wait_call = {"id": "wait-primary", "type": "function", "function": {
        "name": "switch_model", "arguments": json.dumps({"primary": "wait"})}}
    first = {"role": "assistant", "content": "", "tool_calls": [wait_call]}
    final = {"role": "assistant", "content": "The primary answered."}
    failures = 12
    gateway.results = [_reply(ROUTE, first)] + [result(outcome="failed", problem={
        "code": "provider_failed", "message": "Upstream refused", "retryable": True,
        "context": {"httpStatus": status}}) for _ in range(failures)] + [_reply(ROUTE, final)]
    gateway.dispatch = ["response_received"] * (failures + 2)
    _bounded_creates(monkeypatch, gateway, failures + 2)
    sent = []
    if wire != "managed":
        def send(target, messages, schemas, *args, **kwargs):
            assert target["provider"] == "openai", "a chosen primary wait must not dispatch a fallback"
            sent.append(target["provider"])
            body = {"messages": deepcopy(messages), "tools": schemas or [], "model": "primary-test"}
            request = _attempt_request(target, body)
            def physical():
                if len(sent) in (1, failures + 2):
                    return (first if len(sent) == 1 else final), {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.01, "cost_final": True}
                if wire == "sse":
                    from ouroboros.llm_stream import ChatAccumulator
                    ChatAccumulator().accept("message", json.dumps({"error": {"code": status, "message": "Temporary refusal"}}))
                raise httpx.HTTPStatusError("Temporary refusal", request=httpx.Request("POST", "https://fixture.invalid"),
                                           response=httpx.Response(status))
            return ua.execute_physical_attempt(request, physical,
                extractor=lambda value: (value[1], 0.01, True), before_dispatch=_candidate_before_dispatch(body, request))
        monkeypatch.setattr(ctx.llm, "_chat_remote", send)
    def no_probe(*_a, **_kw):
        pytest.fail("an available catalog does not prove generation recovery")
    monkeypatch.setattr(ctx.llm, "claudexor_model_catalog", no_probe)
    monkeypatch.setattr(ctx.llm, "claudexor_model_sources", no_probe)
    bursts, waits, notes = [], [], []
    monkeypatch.setattr(loop_llm_call, "_sleep_within_deadline", lambda seconds, *_a, **_k: bursts.append(seconds) or True)
    def pause(seconds, _wake):
        waits.append(seconds)
        if ending in {"stop", "wrap"}:
            from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION
            reason = REASON_OWNER_STOPPED_DIRECT_TURN if ending == "stop" else REASON_OWNER_REQUESTED_FINALIZATION
            msg_id = "stop-primary-wait"
            if ending == "wrap":
                from ouroboros import cancel_intents
                from supervisor.owner_stop import owner_stop_control_id
                intent = cancel_intents.request_cancel(ctx.drive_root, "task-one",
                    requested_stop_policy=cancel_intents.STOP_POLICY_FINALIZE)
                msg_id = owner_stop_control_id(intent)
            owner_mailbox.write_owner_message(ctx.drive_root, reason, "task-one",
                msg_id=msg_id, kind=owner_mailbox.KIND_FINALIZE_NOW)
        elif ending == "deadline":
            tools._ctx.task_metadata["deadline_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        return ending != "answer"
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", pause)
    text, usage, _trace = loop.run_llm_loop(
        ctx.messages, tools, ctx.llm, ctx.drive_logs, lambda text, **_kw: notes.append(text), queue.Queue(),
        task_id="task-one", drive_root=ctx.drive_root, event_queue=events)
    attempts = len(gateway.creates) if wire == "managed" else len(sent)
    assert attempts == (failures + 2 if ending == "answer" else 7)
    assert waits == ([4.0, 8.0] if ending == "answer" else [4.0])
    assert bursts == [4.0, 8.0, 16.0, 32.0, 60.0] * (2 if ending == "answer" else 1)
    assert not owner.waits  # no immediate resource_available row/new paid call loop
    if wire == "managed":
        assert all(payload["account"] == {"mode": "pin", "profileId": "account-a"} for payload, _ in gateway.uploads)
    rows = ledger(ctx.drive_root)  # one current row per attempt: every sent attempt
    assert sum(row["state"] in {"dispatched", "settled", "unresolved"} for row in rows) == attempts
    assert not any(row["state"] == "released" for row in rows)
    assert any("primary provider temporarily refused" in note for note in notes)
    assert not any("$0" in note or "connection restored" in note.lower() for note in notes)
    if ending == "answer":
        assert text == final["content"]
    else:
        notice = usage.get("terminal_provider_notice", text)
        if ending == "stop":
            assert "owner stopped this chat turn" in notice
        else:
            assert "No new summary request was sent" in notice
        if ending == "wrap":
            assert "owner requested Wrap up while the primary provider was refusing requests" in notice
            assert any(note.endswith("Stop cancels.") for note in notes)
        assert "provider connection was unavailable" not in notice and "$0" not in notice
