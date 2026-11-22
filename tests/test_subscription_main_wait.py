"""The live Main call rebinds routes without replaying its completed work."""

import asyncio
from copy import deepcopy
from dataclasses import replace
import json
import queue
from types import SimpleNamespace

import httpx
import pytest

from ouroboros import fallback_cooldown, loop, loop_forced_finalization as forced, model_wait, owner_model
from ouroboros import owner_wait, pricing, provider_models, task_pacing, usage_accounting as ua, working_checkpoint
from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY
from ouroboros.llm_attempt import _attempt_request, _candidate_before_dispatch
from ouroboros.llm_claudexor import cache_key_for_model
from ouroboros.loop_llm_call import confirmed_api_resource_refusal
from ouroboros.loop_model_call import RESOURCE_REFUSAL_KEY, _recover_failed_round, _reprepare_waiting_main
from ouroboros.loop_round_limits import _RoundLimitContext
from ouroboros.model_slots import MODEL_ACCOUNTS_KEY
from ouroboros.send_clock import CLOCK_NOTE_PREFIX
from ouroboros.tool_capabilities import META_TOOL_NAMES
from ouroboros.tool_policy import route_tool_limit_notice
from tests.test_context_fit_integration import _plan
from tests.test_llm_claudexor import MODEL, ROUTE, result, ledger, setup as gateway_fixture
from tests.test_model_wait import live_wait as wait_fixture
from tests._usage_store_testing import dispatched_attempts
from tests._shared import saved_settings

setup = gateway_fixture
live_wait = wait_fixture
ROUTE_B = {**ROUTE, "credentialProfileId": "account-b", "accountFingerprint": "fingerprint-b"}


@pytest.fixture
def main_call(live_wait, monkeypatch):
    root, gateway, client, controller, events, decide = live_wait
    plan = replace(_plan(preferred="max", window=900_000), model=MODEL,
                   provider="claudexor", model_role="main", model_route=ROUTE,
                   route_fp="capacity-account-a")
    messages = [*plan.messages_for("max"), result()["message"],
                {"role": "tool", "tool_call_id": "a", "content": "verified read A"},
                {"role": "tool", "tool_call_id": "b", "content": "completed review B"}]
    inner = SimpleNamespace(task_id="task-one", task_attempt=1, task_metadata={},
                            active_model=MODEL, active_context_mode="max", messages=messages,
                            context_fit_plan=plan, event_queue=events, drive_logs=lambda: root / "logs")
    controller.tool_context = inner
    # A Main registry: both discovery selectors are callable (``tool_policy.usable_selectors``).
    tools = SimpleNamespace(_ctx=inner, callable_rows=lambda namespace="": [{"name": n} for n in META_TOOL_NAMES])
    ctx = loop._RoundModelCallContext(
        llm=client, messages=messages, tools=tools, context_fit_plan=plan,
        active_model=MODEL, tool_schemas=[], active_effort="medium", max_retries=3,
        drive_logs=root / "logs", task_id="task-one", round_idx=1, event_queue=events,
        accumulated_usage={}, task_type="task", active_use_local=False,
        active_context_mode="max", drive_root=root)
    observations = []

    def route_evidence(task, **_kwargs):
        observations.append(deepcopy(task))
        observed = task.get("model_route") or {}
        account = task.get("credential_profile_id") or observed.get("credentialProfileId") or "account-b"
        subscription = task["model"] == MODEL and not task["use_local_model"]
        return ({"model": task["model"], "provider": "claudexor" if subscription else "local" if task["use_local_model"] else "openai"},
                SimpleNamespace(route_fp=f"capacity-{account}" if subscription else "other-route", status="confirmed",
                                stale=False, window_tokens=240_000 if subscription else 300_000,
                                source_id="codex" if subscription else "", source="advertised",
                                credential_profile_id=account if subscription else "",
                                account_fingerprint="fingerprint-b" if subscription else ""))

    from ouroboros import context
    monkeypatch.setattr(context, "_context_fit_route", route_evidence)
    monkeypatch.setattr(loop, "_server_web_allowed_by_task", lambda _ctx: False)
    return ctx, gateway, controller, events, decide, observations


def _without_context_facts(messages, *, physical=False):
    """Compare protocol/source bytes independently of the required, validated facts tail."""
    from ouroboros.loop_messages import CONTEXT_FACTS_HEADER, CONTEXT_FACTS_NAME

    facts = ([row for row in messages if row.get("role") == "user" and isinstance(row.get("content"), str)
              and row["content"].startswith(CONTEXT_FACTS_HEADER + " ")] if physical else
             [row for row in messages if row.get(HOST_CONTEXT_KIND_KEY) == CONTEXT_FACTS_NAME])
    assert facts, "Main dispatch/reprepare must include its own context facts"
    assert all(row["role"] == "user" and isinstance(row["content"], str)
               and row["content"].startswith(CONTEXT_FACTS_HEADER + " ") for row in facts)
    assert all("name" not in row for row in facts)
    if physical:
        assert all(HOST_CONTEXT_KIND_KEY not in row for row in facts)
    return [row for row in messages if row not in facts]


def _failed(code, route=ROUTE):
    return result(outcome="failed", route=route, problem={"code": code, "message": code})


def _answer(text, route=ROUTE):
    return {**result(route=route), "message": {"role": "assistant", "content": text}}


def _switch_wait(decide, event, model, *, account="", use_local=False):
    return decide({
        "request_id": "switch-once", "decision_id": f"model_wait:{event['task_id']}:{event['wait_id']}",
        "revision": event["revision"], "action": "switch", "model": model,
        "credential_profile_id": account, "use_local": use_local, "persist_role": False,
    })


def _dispatch(ctx):
    return loop._call_round_model(ctx)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_native_account_repair_rebinds_real_physical_candidate_before_send(main_call, asynchronous):
    ctx, gateway, controller, _events, _decide, observations = main_call
    original = deepcopy(ctx.messages)
    gateway.results = [_failed("invalid_continuation", ROUTE_B), result(route=ROUTE_B)]
    gateway.dispatch = ["not_started", "response_received"]
    if asynchronous:
        disposition = loop._measure_round_main_fit(ctx, automatic_pass_used=False)
        with controller.register_reprepare("main", lambda values: _reprepare_waiting_main(ctx, values)):
            with ua.bind_physical_attempt_context(loop._physical_context_for_fit(disposition)):
                asyncio.run(ctx.llm.chat_async(ctx.messages, MODEL, model_role="main"))
    else:
        answer, _cost, _mode = _dispatch(ctx)
        assert answer["content"] == result()["message"]["content"]
    rows = ledger(ctx.drive_root)  # one current row per attempt: the released send, then the answered one
    assert [(row["state"], row["revision"]) for row in rows] == [("released", 4), ("settled", 3)]
    assert rows[0]["physical_failure"]["stage"] == "raised_exception"
    dispatched = dispatched_attempts(ctx.drive_root)
    assert dispatched[0]["physical_context"]["route_fp"] == "capacity-account-a"
    assert dispatched[1]["physical_context"]["route_fp"] == "capacity-account-b"
    # The round's frame for a stated 240,000 total nothing has proved: less the reply floor, plus the reserve.
    assert dispatched[1]["physical_context"]["capacity_total_tokens"] == 240_000 - 8_192 + 65_536
    assert observations[0]["model_route"] == ROUTE_B
    assert len(gateway.accepted_operations) == 2 and gateway.creates[0] != gateway.creates[1]
    resent = _without_context_facts(gateway.uploads[1][0]["messages"], physical=True)
    assert "nativeContinuation" not in resent[2]
    # A Main round's repaired send is a new host preparation: canonical rows, then its own
    # clock line. The bare async driver binds no Main clock.
    clocked = resent[-1]["content"].startswith(CLOCK_NOTE_PREFIX)
    assert clocked is (not asynchronous)
    assert resent[2]["tool_calls"] == original[2]["tool_calls"] and resent[3:len(resent) - clocked] == original[3:]
    assert "nativeContinuation" not in ctx.messages[2]
    assert ctx.context_fit_plan.core_sha256 == "a" * 64


def test_quota_auto_wait_rejoins_same_round_then_repairs_changed_account(main_call, monkeypatch):
    ctx, gateway, _controller, events, _decide, observations = main_call
    saved_settings(monkeypatch, ctx.drive_root, OUROBOROS_MODEL_FALLBACKS="")  # no configured route: the wait is in place
    gateway.results = [_failed("subscription_window_exhausted"),
                       _failed("invalid_continuation", ROUTE_B), result(route=ROUTE_B)]
    gateway.dispatch = ["not_started", "not_started", "response_received"]
    answer, cost, mode = _dispatch(ctx)
    assert answer and cost is None and mode == "max"
    assert len(gateway.accepted_operations) == 3
    assert ctx.accumulated_usage["rounds"] == 1
    assert ctx.accumulated_usage["_model_route"] == ROUTE_B
    assert ctx.accumulated_usage["_context_route_fp"] == "capacity-account-b"
    assert any(item["model_route"] == ROUTE_B for item in observations)
    waits = [event for event in list(events.queue) if event.get("type") == "task_model_wait"]
    assert [event["state"] for event in waits] == ["waiting", "waiting", "resolved"]
    assert waits[-1]["resolution"] == "resource_available"
    assert [row["state"] for row in ledger(ctx.drive_root)].count("settled") == 1


@pytest.mark.parametrize("image_mode", ["off", "caption"])
def test_wait_reprepares_vision_from_canonical_images(main_call, monkeypatch, image_mode):
    from ouroboros import vision_routing

    saved_settings(monkeypatch, main_call[0].drive_root, OUROBOROS_MODEL_FALLBACKS="")  # the wait is in place

    ctx, gateway, _controller, _events, _decide, _observations = main_call
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}}
    ctx.messages[-1]["content"] = [{"type": "text", "text": "Original visual evidence"}, image]
    original = deepcopy(ctx.messages)
    monkeypatch.setattr(vision_routing, "get_image_input_mode", lambda: image_mode)
    monkeypatch.setattr(vision_routing, "resolve_vision_caption_model", lambda *a, **kw: "fixture/vision")
    captions = []

    def caption(*args, **kwargs):
        captions.append(True)
        return "The exact diagram caption", {"prompt_tokens": 2, "completion_tokens": 3, "cost": 0.01}

    monkeypatch.setattr(ctx.llm, "vision_query", caption)
    gateway.results = [_failed("subscription_window_exhausted"), result()]
    gateway.dispatch = ["not_started", "response_received"]
    answer, _, _ = _dispatch(ctx)
    assert answer and len(gateway.accepted_operations) == 2
    # The consumed clock line of the answered send now closes the canonical transcript.
    assert ctx.messages[-1]["content"] == gateway.uploads[1][0]["messages"][-1]["content"]
    canonical = _without_context_facts(ctx.messages)
    assert canonical[:-1] == original  # only the consumed clock may follow the exact source rows
    assert canonical[-2]["content"] == original[-1]["content"]
    sent = [_without_context_facts(item[0]["messages"], physical=True)[-2]["content"] for item in gateway.uploads]
    assert sent[0] == sent[1] and "image_url" not in str(sent)
    assert len(captions) == (1 if image_mode == "caption" else 0)
    assert canonical[-3]["content"] == "verified read A"


def test_main_authored_checkpoint_after_a_real_projected_image_wait(main_call, monkeypatch, tmp_path):
    from ouroboros import vision_routing
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.tools.registry import ToolRegistry
    from tests.test_main_authored_context import call

    ctx, gateway, controller, events, _decide, _observations = main_call
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "max")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")
    monkeypatch.setenv("MCP_ENABLED", "false")
    monkeypatch.setattr(loop, "_maybe_inject_finalization_nudges", lambda *_a: False)
    monkeypatch.setattr(vision_routing, "get_image_input_mode", lambda: "off")
    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=ctx.drive_root)
    registry._ctx.repo_dir.mkdir()
    (registry._ctx.repo_dir / "source.txt").write_text("Complete evidence.\n" * 160, encoding="utf-8")
    registry._ctx.context_fit_plan = ctx.context_fit_plan
    registry._ctx.task_model_override = MODEL
    controller.tool_context = registry._ctx
    image_turn = {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}}]}
    def response(message):
        row = result()
        row["message"] = message
        return row
    gateway.results = [response(call("read_file", {"path": "source.txt"}, "read")),
        _failed("subscription_window_exhausted"),
        response(call("compact_context", {"working_note": "The image and source remain available.", "keep_unit_ids": []}, "compact")),
        response({"role": "assistant", "content": "done"})]
    gateway.dispatch = ["response_received", "not_started", "response_received", "response_received"]
    answer, _, _ = loop.run_llm_loop(
        messages=[*ctx.context_fit_plan.messages_for("max"), deepcopy(image_turn)], tools=registry,
        llm=ctx.llm, drive_root=ctx.drive_root, drive_logs=ctx.drive_logs,
        incoming_messages=queue.Queue(), emit_progress=lambda *a, **kw: None,
        event_queue=events, task_id="task-one")
    assert answer == "done" and len(gateway.accepted_operations) == 4
    receipt = registry._ctx._context_view_receipt
    assert receipt["status"] == "applied"
    checkpoint = json.loads(read_actor_source_bytes(ctx.drive_root, "task-one", receipt["checkpoint_ref"]))
    assert image_turn in checkpoint["messages"] and image_turn in registry._ctx.messages
    assert any(m.get("tool_call_id") == "read" for m in checkpoint["messages"])
    assert not any(m.get("tool_call_id") == "read" for m in gateway.uploads[-1][0]["messages"])
    assert all("image omitted" in str(payload[0]["messages"]) for payload in gateway.uploads)


@pytest.mark.parametrize("inline_first", [True, False])
def test_reprepare_selected_vision_route_preserves_source_and_accounts_caption(main_call, monkeypatch, inline_first):
    from ouroboros import vision_routing
    from ouroboros.loop_llm_call import _prepare_main_messages
    from ouroboros.llm_attempt import single_physical_send, single_physical_send_active, track_physical_attempts
    from ouroboros.transport_custody import physical_attempt_tracker

    ctx, _gateway, _controller, _events, _decide, _observations = main_call
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}}
    ctx.messages[-1]["content"] = [deepcopy(image)]
    monkeypatch.setattr(vision_routing, "get_image_input_mode", lambda: "auto")
    monkeypatch.setattr(vision_routing, "supports_vision", lambda model, **kw: (model == MODEL) == inline_first)
    monkeypatch.setattr(vision_routing, "resolve_vision_caption_model", lambda *a, **kw: "fixture/vision")
    captions = []
    def caption(*args, **kwargs):
        assert ua.current_physical_attempt_context() is None
        captions.append((single_physical_send_active(), physical_attempt_tracker()))
        return "A complete caption", {"prompt_tokens": 2, "completion_tokens": 3, "cost": 0.01}
    monkeypatch.setattr(ctx.llm, "vision_query", caption)
    sent = _prepare_main_messages(ctx.messages, model=MODEL, model_role="main", model_account_override=None,
        llm=ctx.llm, accumulated_usage=ctx.accumulated_usage, drive_root=ctx.drive_root,
        task_id=ctx.task_id, event_queue=ctx.event_queue, use_local=False, task_attempt=1, deadline_ts=None)
    assert (sent[-1]["content"] == [image]) is inline_first
    physical = loop._physical_context_for_fit(loop._measure_round_main_fit(ctx, automatic_pass_used=False))
    tracker = object()  # no Main send happens in this probe; only its scope is read
    with ua.bind_physical_attempt_context(physical), single_physical_send(lambda _reservation: None), track_physical_attempts(tracker):
        prepared = _reprepare_waiting_main(ctx, {"messages": sent, "model": "openai::vision-route",
                                                "model_role": "main", "tools": []})
        assert single_physical_send_active() and physical_attempt_tracker() is tracker  # Main's scopes resume after helper preparation
    prepared_source = _without_context_facts(prepared.kwargs["messages"])
    assert (prepared_source[-1]["content"] == [image]) is not inline_first
    assert "image caption" in str(prepared_source[-1]["content"] if inline_first else sent[-1]["content"])
    assert _without_context_facts(ctx.messages)[-1]["content"] == [image] and captions == [(False, None)]
    assert ctx.accumulated_usage["cost"] == 0.01


@pytest.mark.parametrize("destination,ceiling", [
    ("openai::gpt-5.6-terra", 128), ("openai/gpt-5.6-luna", None), (MODEL, None),
])
def test_a_wait_card_switch_sends_the_list_fitted_to_the_chosen_routes_ceiling(main_call, destination, ceiling):
    from tests.test_route_tool_schema_limit import PINNED, _catalog, _names

    ctx = main_call[0]
    ctx.tool_schemas[:] = _catalog(129)  # extras first, then core and meta
    catalog = _names(ctx.tool_schemas)
    physical = loop._physical_context_for_fit(loop._measure_round_main_fit(ctx, automatic_pass_used=False))
    with ua.bind_physical_attempt_context(physical):
        prepared = _reprepare_waiting_main(ctx, {"messages": deepcopy(ctx.messages), "model": destination,
                                                "model_role": "main", "tools": deepcopy(ctx.tool_schemas)})
    sent, notice = _names(prepared.kwargs["tools"]), "at most 128 tool schemas" in str(prepared.kwargs["messages"])
    # The switched send, its measurement and discovery carry one list: fitted on a ceiling, untouched without one.
    assert sent == _names(ctx.tool_schemas) and notice is (ceiling is not None)
    assert ("and enable_tools loads one" in str(prepared.kwargs["messages"])) is notice  # a selector Main can call
    last_extra = catalog[-len(PINNED) - 1]
    assert sent == ([name for name in catalog if name != last_extra] if ceiling else catalog)


def test_processing_repair_does_not_authorize_native_source_reset(main_call, monkeypatch):
    from ouroboros import llm_claudexor
    from tests.test_processing_claudexor import refusal

    ctx, gateway, _controller, _events, _decide, observations = main_call
    monkeypatch.setenv("OUROBOROS_PROCESSING_PREFERENCE", "economy")
    monkeypatch.setattr(llm_claudexor, "model_sources", lambda **kw: {
        "sources": [{"id": "codex", "processingPreferences": ["standard", "economy"]}]})
    original = deepcopy(ctx.messages[2]["nativeContinuation"])
    failed = refusal()
    failed["route"] = ROUTE_B
    gateway.results = [failed, result(route=ROUTE_B)]
    gateway.dispatch = ["response_received", "response_received"]
    answer, _, _ = _dispatch(ctx)
    assert answer and len(gateway.accepted_operations) == 2
    assert ctx.messages[2]["nativeContinuation"] == original
    assert gateway.uploads[1][0]["messages"][2]["nativeContinuation"] == original
    assert observations[0]["model_route"] == ROUTE_B
    assert gateway.uploads[1][0]["options"]["processingPreference"] == "standard"


@pytest.mark.parametrize("destination,use_local,pin", [
    ("openai::alternate", False, ""), ("local-model", True, ""), (MODEL, False, "account-b"),
])
def test_manual_switch_updates_only_waiting_role_and_continues_current_main_call(
    main_call, monkeypatch, destination, use_local, pin,
):
    ctx, gateway, controller, events, decide, _observations = main_call
    saved_settings(monkeypatch, ctx.drive_root, OUROBOROS_MODEL_FALLBACKS="")  # the wait is in place
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"main": "account-a", "light": "light-account"}))
    actor = ctx.tools._ctx
    actor.primary_route = {"model": MODEL, "use_local": False, "role": "main"}
    actor.model_wait_context = controller
    gateway.results = [_failed("subscription_window_exhausted"), result(route=ROUTE_B), result(route=ROUTE_B)]
    gateway.dispatch = ["not_started", "response_received", "response_received"]
    if destination == MODEL:
        gateway.results.insert(1, _failed("invalid_continuation", ROUTE_B))
        gateway.dispatch.insert(1, "not_started")
    decisions = []

    def catalog(*_args, **_kwargs):
        wait = next(event for event in reversed(list(events.queue)) if event.get("type") == "task_model_wait")
        response = _switch_wait(decide, wait, destination, account=pin, use_local=use_local)
        assert response.status_code == 202
        decisions.append(json.loads(response.body))
        return {}

    monkeypatch.setattr(ctx.llm, "claudexor_model_catalog", catalog)
    remote = ctx.llm._chat_remote

    def direct(messages, model, tools):
        # The replacement did not answer yet, and there may not be another round.
        published = [row["lead"] for row in events.queue if row.get("type") == "work_lead"]
        assert len(published) == 1 and published[0]["model"] == destination and published[0]["basis"] == "owner"
        target = {"provider": "local" if use_local else "openai", "usage_model": model}
        payload = {"messages": deepcopy(messages), "tools": tools or [], "model": model}
        request = _attempt_request(target, payload)
        usage = {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0 if use_local else 0.2,
                 "provider": target["provider"], "resolved_model": model, "cost_final": True}
        return ua.execute_physical_attempt(
            request, lambda: ({"role": "assistant", "content": "switched answer"}, usage),
            extractor=lambda value: (value[1], value[1]["cost"], True),
            before_dispatch=_candidate_before_dispatch(payload, request))

    def remote_call(target, messages, tools, *args, **kwargs):
        if target["provider"] == "claudexor":
            return remote(target, messages, tools, *args, **kwargs)
        return direct(messages, destination, tools)

    monkeypatch.setattr(ctx.llm, "_chat_remote", remote_call)
    monkeypatch.setattr(ctx.llm, "_chat_local", lambda messages, tools, *_args, **_kwargs: direct(messages, destination, tools))
    answer, cost, _mode = _dispatch(ctx)
    assert answer and len(decisions) == 1 and decisions[0]["saved"] is False
    assert ctx.active_model == destination and ctx.active_use_local is use_local
    (lead,) = [row["lead"] for row in events.queue if row.get("type") == "work_lead"]
    assert (lead["model"], lead["use_local"], lead["basis"], lead["cause"]) == (destination, use_local, "owner", "wait_card")
    assert lead["primary"] == {"model": MODEL, "use_local": False, "basis": "main"} and "pending" not in lead
    assert controller.overrides == {"main": {"model": destination, "use_local": use_local, "model_account_override": pin}}
    assert json.loads(__import__("os").environ[MODEL_ACCOUNTS_KEY])["light"] == "light-account"
    assert cost == (None if destination == MODEL else 0 if use_local else 0.2)
    rows = [row for row in _without_context_facts(ctx.messages)
            if not str(row.get("content") or "").startswith(CLOCK_NOTE_PREFIX)]
    assert rows[-2]["content"] == "verified read A" and rows[-1]["content"] == "completed review B"
    # Only a real stamping lane seals — and so replays — a clock line; this fixture's direct lane does not.
    assert str(ctx.messages[-1]["content"]).startswith(CLOCK_NOTE_PREFIX) is (destination == MODEL)
    # The same model used by Light remains pinned to its own account after Main switches.
    ctx.llm.chat([], MODEL, model_role="light")
    assert gateway.uploads[-1][0]["account"] == {"mode": "pin", "profileId": "light-account"}


def test_main_control_interrupt_is_typed_no_retry_and_keeps_operation_custody(main_call, monkeypatch):
    ctx, gateway, controller, _events, _decide, _observations = main_call
    gateway.pending = True
    monkeypatch.setattr(controller, "control_reason", lambda **_kw: "cancelled" if gateway.accepted_operations else None)
    with pytest.raises(model_wait.ModelWaitInterrupted) as raised:
        _dispatch(ctx)
    error = raised.value
    assert error.control_reason == "cancelled" and error.operation_id == "op-0"
    assert error.physical_attempt_capture.state == "unresolved"
    assert error.model_role_route["role"] == "main"
    assert len(gateway.accepted_operations) == 1 and gateway.cancels == [("op-0", "host_cancelled")]
    assert not controller.waits and not ctx.accumulated_usage.get("_last_llm_retry_same_request")


def test_cancel_after_result_keeps_settled_usage_and_exact_result(main_call, monkeypatch):
    ctx, gateway, controller, _events, _decide, _observations = main_call
    monkeypatch.setattr(controller, "control_reason", lambda **_kw: "cancelled" if gateway.acks else None)
    with pytest.raises(model_wait.ModelWaitInterrupted) as raised:
        _dispatch(ctx)
    assert raised.value.model_result == result()
    assert raised.value.usage["prompt_tokens"] == 20
    assert raised.value.physical_attempt_capture.state == "settled"
    assert raised.value.route == ROUTE
    assert len(gateway.accepted_operations) == 1


def test_native_repair_without_main_callback_refuses_stale_physical_fit(setup):
    _root, gateway, client = setup
    gateway.results = [_failed("invalid_continuation", ROUTE_B), result(route=ROUTE_B)]
    gateway.dispatch = ["not_started", "response_received"]
    physical = ua.PhysicalAttemptContext(
        profile="owner_max", route_fp="old-account", rendered_mode="max",
        measurement_basis="cold_estimate", round_id="round-one", target_total_tokens=None,
        capacity_total_tokens=900_000, context_target_miss=False, automatic_pass_used=False)
    with ua.bind_physical_attempt_context(physical):
        with pytest.raises(model_wait.ModelWaitInterrupted, match="model_wait_reprepare_required"):
            client.chat([result()["message"]], MODEL, model_role="main")
    assert len(gateway.accepted_operations) == 1


def test_unknown_main_outcome_never_retries_or_waits_for_quota(main_call):
    ctx, gateway, controller, _events, _decide, _observations = main_call
    gateway.results, gateway.dispatch = [result(outcome="unknown")], ["unknown"]
    answer, _cost, _mode = _dispatch(ctx)
    assert answer is None and ctx.accumulated_usage["_last_llm_error_kind"] == "provider_outcome_unknown"
    assert len(gateway.accepted_operations) == 1 and not controller.waits
    assert ledger(ctx.drive_root)[-1]["state"] == "unresolved"


def _api_error(status, code, *, settled=True, extra=None):
    body = {"error": {"message": "Fixture upstream detail", "code": code, **(extra or {})}}
    if settled:
        body["usage"] = {"cost": 0.0}
    return httpx.HTTPStatusError("API refusal", request=httpx.Request("POST", "https://fixture.invalid"),
                                 response=httpx.Response(status, json=body))


def _refuse_api_attempt(target, messages, schemas, error):
    body = {"model": target["resolved_model"], "messages": messages, "tools": schemas or []}
    request = _attempt_request(target, body)

    def refused():
        raise error

    return ua.execute_physical_attempt(request, refused, before_dispatch=_candidate_before_dispatch(body, request))


@pytest.mark.parametrize("provider,status,code,reason", [
    ("openai", 401, None, ""), ("openai", 403, "invalid_api_key", "auth"),
    ("openai", 402, None, ""), ("openai", 429, "insufficient_quota", "quota"),
    ("zai", 429, 1113, "quota"), ("openai", 429, 1113, ""), ("zai", 429, None, ""),
    ("openai", 429, "rate_limit_exceeded", ""), ("openrouter", 402, "openrouter_in_flight_budget", ""),
    ("openai", 403, None, ""), ("openai", 403, "unsupported_country_region_territory", ""),
    ("openai", 403, "content_policy_violation", ""), ("openai", 401, "unsupported_country", ""),
    ("local", 401, "invalid_api_key", ""), ("claudexor", 401, "invalid_api_key", ""),
])
def test_only_this_api_attempts_settled_resource_refusal_enters_a_wait(provider, status, code, reason):
    error = _api_error(status, code)
    capture = ua.PhysicalAttemptCapture(attempt_id="api-refusal", model="fixture", provider=provider,
        state="settled", candidate_measurement_kind="opaque", provider_status_code=status, provider_code=str(code or ""))
    error.physical_attempt_capture = capture
    assert confirmed_api_resource_refusal(error) == reason
    # A received HTTP refusal with no token/price receipt is still a response.
    error.physical_attempt_capture = replace(capture, state="unresolved", route_is_loopback=True)
    assert confirmed_api_resource_refusal(error) == reason
    for flag in ("receiver_abandoned", "explicit_timeout"):
        setattr(error, flag, True)
        assert confirmed_api_resource_refusal(error) == ""
        delattr(error, flag)
    for state in ("dispatched", "released"):
        error.physical_attempt_capture = replace(capture, state=state)
        assert confirmed_api_resource_refusal(error) == ""
    error.physical_attempt_capture = replace(capture, attempt_id="")
    assert confirmed_api_resource_refusal(error) == ""
    error.physical_attempt_capture = capture
    error.stream_incomplete = True
    assert confirmed_api_resource_refusal(error) == ""
    del error.stream_incomplete
    error.stream_receipt = {"model_output_received": True}
    assert confirmed_api_resource_refusal(error) == ""


@pytest.mark.parametrize("status,body,headers,expected", [
    (401, b"", {}, "auth"), (402, b"", {}, "quota"), (403, b"", {}, ""),
    (401, b'{"error":{"message":"Account unavailable"}}', {"WWW-Authenticate": "Bearer"}, "auth"),
    (402, b'{"error":{"metadata":{"limit_source":"openrouter_credits"}}}', {}, "quota"),
    (402, b'{"error":{"metadata":{"limit_source":"openrouter_key_limit"}}}', {}, "quota"),
    (402, b'{"error":{"code":402,"metadata":{"limit_source":"openrouter_in_flight_budget"}}}', {}, ""),
])
def test_direct_api_wait_uses_the_received_status_and_its_own_limit_source(status, body, headers, expected):
    response = httpx.Response(status, content=body, headers=headers)
    error = httpx.HTTPStatusError("ignored", response=response, request=httpx.Request("POST", "http://localhost"))
    error.physical_attempt_capture = ua.PhysicalAttemptCapture(attempt_id="local-api", model="fixture", provider="openrouter",
        state="unresolved", candidate_measurement_kind="opaque", provider_status_code=status, route_is_loopback=True,
        provider_code="402" if b'"code":402' in body else "")
    assert confirmed_api_resource_refusal(error) == expected


@pytest.mark.parametrize("source", ["upstream", "earlier_attempt", "conflicting_type", "lost_connection",
                                    "local_policy", "region", "content"])
def test_foreign_evidence_does_not_open_an_api_key_wait(source):
    code = None if source in {"region", "content"} else "invalid_api_key"
    error = _api_error(401, code, extra={
        "upstream": {"metadata": {"provider_name": "Fixture", "raw": {"code": "invalid_api_key"}}},
        "earlier_attempt": {"attempts": [{"code": "invalid_api_key"}]},
        "region": {"message": "Service unavailable in this region"},
        "content": {"message": "Rejected under the content policy"},
    }.get(source))
    if source == "lost_connection":
        error = httpx.ReadError("fixture connection lost")
    elif source == "local_policy":
        error.code = "provider_policy_refusal"
    error.physical_attempt_capture = ua.PhysicalAttemptCapture(
        attempt_id="outer", model="fixture", provider="openrouter", state="settled", candidate_measurement_kind="opaque",
        provider_status_code=401, provider_code=code or "",
        provider_error_type="unsupported_country" if source == "conflicting_type" else "")
    assert confirmed_api_resource_refusal(error) == ""


@pytest.mark.parametrize("status,ending", [(401, "wait"), (402, "wait"), (401, "cooled"), (401, "rejected"),
                                          (401, "alternative"), (402, "alternative"), (403, "alternative"),
                                          (401, "exact"), (402, "presence"), (401, "unknown_cost"), (401, "old_count")])
def test_ordinary_api_refusal_reaches_the_owner_only_after_configured_routes(main_call, monkeypatch, status, ending):
    ctx, gateway, owner, events, _, _ = main_call
    api = "openai::primary-test"
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-key")
    monkeypatch.setattr(pricing, "_fetch_live_rows", lambda *_a, **_kw: {})
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", MODEL)
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_: ending == "cooled")
    ctx.active_model = ctx.tools._ctx.active_model = api
    ctx.tools._ctx.primary_route = {"model": api, "use_local": False, "role": "main"}
    ctx.tools._ctx.model_wait_context = owner
    ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = replace(ctx.context_fit_plan, model=api,
        provider="openai", model_route={}, model_role="main")
    ctx.tools._ctx.exact_model_route = ending == "exact"
    ctx.task_type = "presence" if ending == "presence" else "task"
    if ending == "presence":
        owner.task["_presence_turn"], owner.worker_slot_held = True, False
        ctx.tools._ctx.is_direct_chat, ctx.tools._ctx.current_task_type = True, "presence"
    elif ending == "old_count":
        ctx.accumulated_usage.update(execution_id="old", _transport_deaths={"round_id": "old:round:1", "count": 1})
    calls = []
    send = ctx.llm._chat_remote

    def remote(target, messages, schemas, *args, **kwargs):
        calls.append(target["provider"])
        if target["provider"] != "openai":
            return send(target, messages, schemas, *args, **kwargs)
        error = _api_error(status, {401: "invalid_api_key", 402: "billing_not_active"}.get(status),
                           settled=ending != "unknown_cost")
        return _refuse_api_attempt(target, messages, schemas, error)

    monkeypatch.setattr(ctx.llm, "_chat_remote", remote)
    exhausted = result(outcome="failed", problem={"code": "invalid_request", "message": "Invalid argument",
                                                   "context": {"httpStatus": 400}})
    if ending == "rejected":
        exhausted = result(outcome="failed", cash=0.17, knowledge="exact", problem={
            "code": "response_rejected", "message": "The operation returned no usable model reply."})
        exhausted["message"] = None
    gateway.results = [_answer("Alternative answer") if ending == "alternative" else exhausted]
    gateway.dispatch = ["response_received"]
    # Interrupt a real durable wait on its first control poll.
    owner.owner_control = lambda: "cancelled" if any(row["state"] == "waiting" for row in owner.waits.values()) else None
    if ending == "cooled":
        with pytest.raises(model_wait.ModelWaitInterrupted, match="cancelled"):
            loop._call_round_model(ctx)
        assert calls == ["openai"] and ledger(ctx.drive_root)[0]["state"] == "settled"
        assert [r["reason"] for r in events.queue if r.get("type") == "task_model_wait" and r["state"] == "waiting"] == ["auth"]
        assert [r["access_reason"] for r in events.queue if r.get("type") == "task_model_wait"] == ["api_key", "api_key"]
        return
    assert loop._call_round_model(ctx)[0] is None
    assert calls == ["openai"] and not owner.waits
    assert ledger(ctx.drive_root)[0]["state"] == ("unresolved" if ending == "unknown_cost" else "settled")

    def recover():
        return _recover_failed_round(ctx, ctx.tools, None, None, context_fit_plan=ctx.context_fit_plan,
                                     active_context_mode="max", emit_progress=lambda *_a, **_kw: None)[0]

    if ending in {"wait", "unknown_cost"}:
        with pytest.raises(model_wait.ModelWaitInterrupted, match="cancelled"):
            recover()
        rows = [row for row in events.queue if row.get("type") == "task_model_wait"]
        assert (rows[0]["model"], rows[0]["reason"], rows[0]["availability_observation"]) == (
            api, "auth" if status == 401 else "quota", "unavailable")
        assert {row["access_reason"] for row in rows} == {"api_key" if status == 401 else "api_billing"}
        assert rows[-1]["state"] == "resolved"
        assert "Fixture upstream detail" not in json.dumps(rows)
    else:
        reply = recover()
        assert bool(reply) == (ending == "alternative")
        assert not owner.waits
        if ending == "alternative":
            (lead,) = [row["lead"] for row in events.queue if row.get("type") == "work_lead"]
            assert (lead["model"], lead["basis"], lead["cause"]) == (
                MODEL, "fallback", {401: "api_key", 402: "api_billing", 403: "auth_error"}[status])
            assert lead["primary"]["model"] == api
        elif ending == "rejected":
            assert len(gateway.creates) == 1
            assert not any(row.get("type") in {"task_model_wait", "work_lead"} for row in events.queue)
            assert ctx.accumulated_usage["_last_llm_failure"]["rejected_receipt"] is True
            assert RESOURCE_REFUSAL_KEY not in ctx.accumulated_usage
            assert "_acting_route_failure" not in ctx.accumulated_usage
            assert ledger(ctx.drive_root)[-1]["state"] == "settled" and ctx.accumulated_usage["cost"] == 0.17
    assert calls == (["openai"] if ending in {"exact", "old_count"} else ["openai", "claudexor"])


@pytest.mark.parametrize("shared", [True, False], ids=["same-family", "cross-family"])
@pytest.mark.parametrize("ending", ["first_candidate", "deferred_owner", "after_choice", "budget_after_choice", "accepted"])
def test_fallback_choice_and_first_request_from_its_saved_round(main_call, monkeypatch, shared, ending):
    ctx, gateway, owner, events, decide, _ = main_call
    actor = ctx.tools._ctx
    actor.drive_root, actor.active_use_local, actor.active_effort = ctx.drive_root, False, ctx.active_effort
    actor.model_wait_context = owner
    actor.primary_route = {"model": MODEL, "use_local": False, "role": "main"}
    primary = deepcopy(ctx.messages)
    original_plan = ctx.context_fit_plan
    original_route = deepcopy(ROUTE)
    ctx.accumulated_usage.update(_model_route=deepcopy(ROUTE), _context_route_fp=original_plan.route_fp,
                                 _last_llm_error_kind="bad_request")
    catalog = [{"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
               for name in ("read", "lookup", "inspect")]
    ctx.tool_schemas[:] = deepcopy(catalog)
    ctx.tools.schemas = lambda: deepcopy(catalog)
    actor._execution_trace = {"tool_calls": [{"tool": "read", "id": call} for call in ("a", "b")]}
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-key")
    monkeypatch.setenv("OUROBOROS_MODEL", MODEL)
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "max")
    candidate = MODEL if shared else "openai::other"
    notice = route_tool_limit_notice(candidate, 2, 3, ["inspect"], selectors=(), schemas=catalog)
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", f"{candidate},openai::last")
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"main": "account-a", "fallback": ["account-b", ""]}))
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_: False)
    monkeypatch.setattr(pricing, "_fetch_live_rows", lambda *_a, **_kw: {})
    gateway.results = [_failed("auth_required", ROUTE_B)]
    gateway.dispatch = ["not_started"]
    chosen = {**ROUTE, "credentialProfileId": "account-c", "accountFingerprint": "fingerprint-c"}
    original_send = ctx.llm._chat_remote

    def send(target, messages, schemas, *args, **kwargs):
        if ending == "budget_after_choice" and polls:
            raise ua.BudgetExceeded("Insufficient budget for the selected account", limit_scope="root")
        if ending == "accepted" and polls:
            gateway.results[len(gateway.creates):] = [_answer("Accepted on the selected account.", chosen)]
            gateway.dispatch[len(gateway.creates):] = ["response_received"]
        if target["provider"] != "openai":
            return original_send(target, messages, schemas, *args, **kwargs)
        error = _api_error(401, "invalid_api_key") if target["resolved_model"] == "other" else _api_error(400, "invalid_value")
        return _refuse_api_attempt(target, messages, schemas, error)

    def check_original():
        assert ctx.messages == primary and actor.messages is ctx.messages
        assert ctx.tool_schemas == catalog and not getattr(actor, "_route_left_out_tool_names", None)
        assert actor.context_fit_plan is original_plan and actor.active_context_mode == "max"
        assert actor.active_model == MODEL and actor.active_use_local is False
        assert ctx.accumulated_usage["_context_route_fp"] == original_plan.route_fp
        assert ctx.accumulated_usage["_model_route"] == original_route
        assert not getattr(actor, "_pending_model_wait_handover", None)

    polls = []

    def while_waiting(_owner):
        row = next((r for r in reversed(list(events.queue))
                    if r.get("type") == "task_model_wait" and r["state"] == "waiting"), None)
        if not row:
            return None
        assert not polls and row["role"] == "fallback:0"
        polls.append(row["role"])
        check_original()  # the durable wait may save the task here
        if ending == "deferred_owner":
            return "owner_model"
        assert _switch_wait(decide, row, MODEL, account="account-c").status_code == 202
        return None

    calls = []
    original_dispatch = loop._dispatch_round_model

    def dispatch(call, *args, **kwargs):
        calls.append(call.model_role)
        if call.model_role == "fallback:0":
            if calls.count(call.model_role) == 1:
                assert len(call.tool_schemas) == 2
                assert (call.messages is ctx.messages) is shared
                assert any(notice in str(row.get("content", "")) for row in call.messages)
                if ending == "first_candidate":
                    _reprepare_waiting_main(call, {"model": "local-fixture", "model_role": call.model_role,
                                                  "use_local": True, "tools": call.tool_schemas})
                    assert actor.active_use_local and actor.active_model == "local-fixture"
                    assert actor._pending_model_wait_handover
            else:
                assert call.tool_schemas == catalog
                assert call.active_model == MODEL and call.context_fit_plan.route_fp == "capacity-account-c"
                assert owner.overrides[call.model_role]["model_account_override"] == "account-c"
        return original_dispatch(call, *args, **kwargs)

    with monkeypatch.context() as staged:
        staged.setattr(ctx.llm, "_chat_remote", send)
        staged.setattr(loop, "_dispatch_round_model", dispatch)
        # The chosen account can serve the full catalog; a retried copy of the
        # refused request would still carry its ceiling notice and two schemas.
        staged.setattr(provider_models, "tool_schema_limit", lambda *_a, **_kw: None if polls else 2)
        staged.setattr(owner_model, "waiting_wake", while_waiting)
        # A missing wake binding would otherwise leave the offline test in a wait.
        wait_polls = []

        def control():
            if any(r["state"] == "waiting" for r in owner.waits.values()):
                wait_polls.append(True)
                assert len(wait_polls) <= 4, "deferred wait did not poll its owner's model change"
            elif calls == ["fallback:0"] and ending == "first_candidate":
                return "owner_pause"
            elif calls.count("fallback:0") == 2 and ending == "after_choice":
                return "owner_pause"

        staged.setattr(owner, "owner_control", control)

        def recover():
            return _recover_failed_round(ctx, ctx.tools, None, None, context_fit_plan=original_plan,
                                         active_context_mode="max", emit_progress=lambda *_a, **_kw: None)

        if ending == "accepted":
            assert recover()[0]["content"] == "Accepted on the selected account."
        elif ending == "budget_after_choice":
            with pytest.raises(ua.BudgetExceeded) as refused:
                recover()
        else:
            reason = "owner_model" if ending == "deferred_owner" else "owner_pause"
            with pytest.raises(model_wait.ModelWaitInterrupted, match=reason):
                recover()
    assert calls == (["fallback:0"] if ending == "first_candidate" else
                     ["fallback:0", "fallback:1"] + (["fallback:0"] if polls and ending != "deferred_owner" else []))
    if ending == "accepted":
        assert actor.context_fit_plan.model_role == "fallback:0" and ctx.accumulated_usage["_model_route"] == chosen
        # Only this round ran. The choice differs by account in the same-family case.
        (lead,) = [row["lead"] for row in events.queue if row.get("type") == "work_lead"]
        assert (lead["model"], lead["basis"], lead["cause"]) == (MODEL, "owner", "wait_card")
        assert lead["primary"]["model"] == MODEL and "pending" not in lead
        packet = gateway.uploads[-1][0]
        assert packet["account"] == {"mode": "pin", "profileId": "account-c"} and packet["tools"] == catalog
        assert not any(notice in str(m.get("content", "")) for m in packet["messages"])
        assert [m["tool_call_id"] for m in packet["messages"] if m["role"] == "tool"] == ["a", "b"]
        return
    assert not any(row.get("type") == "work_lead" for row in events.queue), "an abandoned candidate never started acting"
    check_original()
    limit = SimpleNamespace(tools=ctx.tools, messages=ctx.messages, tool_schemas=ctx.tool_schemas, round_idx=1,
                            llm_trace=actor._execution_trace, accumulated_usage=ctx.accumulated_usage, owner_msg_seen=set())
    if ending == "budget_after_choice":
        from ouroboros import budget_pause, loop_budget
        from ouroboros.artifacts import read_actor_source_bytes

        actor.budget_drive_root, actor.owner_wait_callback = ctx.drive_root, lambda *_: None
        exit_ctx = loop_budget._LoopExitContext(ctx.tools, ctx.drive_root, ctx.task_id, events, ctx.drive_logs,
                                              ctx.accumulated_usage, actor._execution_trace)
        try:
            with pytest.raises(budget_pause.BudgetPauseRequested) as paused:
                loop_budget._handle_budget_exceeded(refused.value, exit_ctx, limit_ctx=limit)
            row = paused.value.pause
            assert row["rail"] == budget_pause.RAIL_DISPATCH_REFUSED
            saved = json.loads(read_actor_source_bytes(ctx.drive_root, ctx.task_id, row["source_ref"]))
        finally:
            budget_pause.end_dispatch_fence(ctx.task_id)
    else:
        assert working_checkpoint.save_round(limit, "ready")
        _, saved = working_checkpoint.read_working(ctx.drive_root, ctx.task_id, 1)
    assert saved["messages"] == primary and saved["tool_schemas"] == catalog
    assert saved["context_model_role"] == "main" and saved["route"]["active_use_local"] is False
    assert saved["usage"]["_model_route"] == original_route
    choices = saved["model_wait"]["overrides"]
    assert ("fallback:0" in choices) == (ending in {"after_choice", "budget_after_choice"})
    if choices:
        assert choices["fallback:0"]["model_account_override"] == "account-c"
    if ending == "budget_after_choice":
        attempts = {row["attempt_id"] for row in ledger(ctx.drive_root)}
        assert len(attempts) == 2
        assert {ref["physical_attempt_id"] for ref in saved["usage"]["llm_call_refs"]} == attempts
    owner.overrides.clear()
    owner.restore_continuation(saved["model_wait"], started_at=None)
    assert owner.overrides == choices
    if ending == "budget_after_choice":
        return  # A durable budget pause still needs its separate Resume grant.
    restored, trace, usage = [], {}, {}
    owner_wait.restore_continuation_state(ctx.tools, saved, restored, trace, usage, set())
    plan, mode = owner_wait.rebind_restored_route(ctx.tools, saved, restored)
    resumed = replace(ctx, messages=restored, tool_schemas=saved["tool_schemas"], accumulated_usage=usage,
                      active_model=actor.active_model, active_use_local=actor.active_use_local,
                      active_context_mode=mode, context_fit_plan=plan)
    first = len(gateway.creates)
    gateway.results[first:] = [_answer("Work continued.")]
    gateway.dispatch[first:] = ["response_received"]
    assert loop._call_round_model(resumed)[0]["content"] == "Work continued."
    assert len(gateway.creates) == first + 1
    packet = gateway.uploads[first][0]
    assert plan.model_role == "main" and mode == "max"
    assert packet["account"] == {"mode": "pin", "profileId": "account-a"} and packet["tools"] == catalog
    assert any("nativeContinuation" in row for row in packet["messages"])
    assert [row for row in packet["messages"] if row["role"] == "tool"] == [row for row in primary if row["role"] == "tool"]
    assert not any(notice in str(row.get("content", "")) for row in packet["messages"])
    assert [row["id"] for row in trace["tool_calls"]] == ["a", "b"]


@pytest.mark.parametrize("local", [True, False], ids=["switch-to-builtin", "switch-subscription-account"])
def test_forced_wait_switch_reprices_the_request_that_is_sent(main_call, monkeypatch, local):
    from ouroboros import local_model

    ctx, gateway, owner, events, decide, _ = main_call
    destination = "local-model" if local else MODEL
    slot = ctx.tools._ctx
    slot.primary_route = {"model": MODEL, "use_local": False, "role": "main"}
    slot.model_wait_context = owner
    ctx.tool_schemas[:] = [{"type": "function", "function": {"name": f"inspect_fixture_{idx}",
        "parameters": {"type": "object", "properties": {}}}} for idx in range(3)]
    monkeypatch.setattr(provider_models, "tool_schema_limit", lambda model, *, use_local: 2 if use_local else None)
    saved_settings(monkeypatch, ctx.drive_root, OUROBOROS_MODEL_FALLBACKS="")
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"main": "account-a"}))
    gateway.results = [_failed("subscription_window_exhausted"), _answer("Finished.", ROUTE_B)]
    gateway.dispatch = ["not_started", "response_received"]
    def choose(*_args, **_kwargs):
        wait = next(e for e in reversed(list(events.queue)) if e.get("type") == "task_model_wait")
        assert wait["state"] == "waiting" and wait["role"] == "main"
        response = _switch_wait(decide, wait, destination, use_local=local, account="" if local else "account-b")
        assert response.status_code == 202
        return {}

    monkeypatch.setattr(ctx.llm, "claudexor_model_catalog", choose)
    local_packets = []
    def complete(**kwargs):
        local_packets.append(kwargs)
        return SimpleNamespace(model_dump=lambda: {"choices": [{"finish_reason": "stop",
            "message": {"role": "assistant", "content": "Finished."}}], "usage": {"prompt_tokens": 5, "completion_tokens": 1}})

    monkeypatch.setattr(local_model, "get_manager", lambda: SimpleNamespace(
        serving_context_evidence=lambda *, bind_endpoint=False: {}))
    monkeypatch.setattr(ctx.llm, "_get_local_client", lambda: SimpleNamespace(
        timeout=None, chat=SimpleNamespace(completions=SimpleNamespace(create=complete))))
    final = _RoundLimitContext(messages=ctx.messages, llm=ctx.llm, active_model=MODEL, active_use_local=False,
        active_effort=ctx.active_effort, max_retries=1, drive_logs=ctx.drive_logs, task_id=ctx.task_id,
        round_idx=1, event_queue=events, accumulated_usage={}, task_type="task", max_rounds=1,
        drive_root=ctx.drive_root, tools=ctx.tools, tool_schemas=ctx.tool_schemas)
    admitted = []
    make_guard = task_pacing.forced_admission_predicate

    def guard(price, admission, **options):
        check = make_guard(price, admission, **options)

        def check_send(actual):
            accepted = check(actual)
            if accepted:
                admitted.append((price, actual))
            return accepted

        return check_send

    monkeypatch.setattr(task_pacing, "forced_admission_predicate", guard)
    original, prepared = task_pacing.prepared_wrapup_candidate(final, deepcopy(final.messages), allow_server_web_search=False)
    assert forced._call_forced_model_once(final, initial_messages=prepared, admitted_request=original) == "Finished."
    before, after = ledger(ctx.drive_root)
    assert [before["state"], after["state"]] == ["released", "settled"] and len(admitted) == 2
    price, sent = admitted[-1]
    assert after["provider"] == price.provider == sent.provider == ("local" if local else "claudexor")
    assert after["candidate_raw_sha256"] == sent.candidate_raw_sha256
    assert price.candidate_clock_free_sha256 == sent.candidate_clock_free_sha256 is not None
    assert len(gateway.creates) == (1 if local else 2) and len(local_packets) == int(local)
    assert final.active_use_local is local and owner.overrides["main"]["model"] == destination
    [lead] = [event["lead"] for event in list(events.queue) if event.get("type") == "work_lead"]
    assert (lead["model"], lead["use_local"], lead["basis"]) == (destination, local, "owner")
    assert slot.route_choice == {"kind": "explicit", "source": "wait_card"}
    if local:
        assert len(local_packets[0]["tools"]) == len(final.tool_schemas) == 2
        assert len(slot._route_left_out_tool_names) == 1
        assert "accepts at most 2 tool schemas" in json.dumps(local_packets[0]["messages"])
    if not local:
        assert gateway.uploads[-1][0]["account"] == {"mode": "pin", "profileId": "account-b"}


@pytest.mark.parametrize("priced,ending", [(False, "answer"), (True, "answer"), (True, "unaffordable"),
                                           (True, "drift"), (False, "refused_again"), (False, "unknown")])
def test_forced_repair_binds_its_own_role_after_a_separate_settled_400(main_call, monkeypatch, priced, ending):
    ctx, gateway, owner, _, _, observations = main_call
    role = "fallback:0"
    ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = replace(ctx.context_fit_plan, model_role=role)
    ctx.tools._ctx.model_wait_context = owner
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"main": "wrong-role", "fallback": [""]}))
    bad_request = result(outcome="failed", problem={"code": "invalid_request", "message": "Invalid argument",
                                                    "context": {"httpStatus": 400, "vendorCode": "invalid_value"}})
    invalid = _failed("invalid_continuation", ROUTE_B)
    good = _answer("Verified final answer", ROUTE_B)
    gateway.results = [bad_request, invalid, invalid if ending == "refused_again" else good]
    if ending == "unknown":
        gateway.results[1] = result(outcome="unknown", route=ROUTE_B, problem=invalid["problem"])
    gateway.dispatch = ["response_received", "unknown" if ending == "unknown" else "not_started",
                        "not_started" if ending == "refused_again" else "response_received"]
    assert loop._call_round_model(ctx)[0] is None
    assert ctx.accumulated_usage["_last_llm_error_kind"] == "bad_request"
    assert [row["state"] for row in ledger(ctx.drive_root)] == ["settled"]

    ctx.messages.append({"role": "user", "content": "Use the late child result to finish."})
    final = _RoundLimitContext(messages=ctx.messages, llm=ctx.llm, active_model=MODEL, active_use_local=False,
        active_effort=ctx.active_effort, max_retries=1, drive_logs=ctx.drive_logs, task_id=ctx.task_id,
        round_idx=ctx.round_idx, event_queue=ctx.event_queue, accumulated_usage=ctx.accumulated_usage,
        task_type="task", max_rounds=1, drive_root=ctx.drive_root, tools=ctx.tools, tool_schemas=ctx.tool_schemas)
    prices = []
    if priced:
        expected, initial = task_pacing.prepared_wrapup_candidate(final, deepcopy(final.messages), allow_server_web_search=False)
        balances = {"root_cap_usd": 3.0, "deciding_usd": 0.0, "global_remaining_usd": 3.0, "use_local": False}

        def fits(*, request, **given):
            assert given == balances
            prices.append(request.candidate_clock_free_sha256)
            return ending != "unaffordable" or len(gateway.creates) < 2

        monkeypatch.setattr(task_pacing, "wrapup_reservation_fits", fits)
        if ending == "drift":
            build = task_pacing.prospective_wrapup_attempt_request

            def changed(**kwargs):
                return replace(build(**kwargs), candidate_clock_free_sha256="0" * 64)

            monkeypatch.setattr(task_pacing, "prospective_wrapup_attempt_request", changed)

        def finish():
            return forced._send_admitted_forced_candidate(final, initial, expected, "round_limit", balances)
    else:
        def finish():
            return forced._call_forced_model_once(final)

    if ending in {"unaffordable", "drift"}:
        failure = forced.ForcedCandidateUnaffordable if ending == "unaffordable" else forced.ForcedRepairDrift
        with pytest.raises(failure):
            finish()
    else:
        assert finish() == ("Verified final answer" if ending == "answer" else "")
    wanted = 3 if ending in {"answer", "refused_again"} else 2
    assert len(gateway.creates) == len(gateway.accepted_operations) == wanted
    assert all(payload["account"]["mode"] == "auto" for payload, _ in gateway.uploads)
    rows = ledger(ctx.drive_root)
    if ending != "answer":
        assert [row["state"] for row in rows] == {
            "drift": ["settled", "released", "released"],
            "unaffordable": ["settled", "released"],
            "refused_again": ["settled", "released", "released"],
            "unknown": ["settled", "unresolved"],
        }[ending]
        if ending == "unknown":
            pending = ctx.accumulated_usage["_pending_transport_outcome"]
            assert pending["physical_attempt_id"] == rows[-1]["attempt_id"]
            assert pending["operation_id"] == gateway.detail(1)["id"]
        if priced:
            assert len(prices) == 2
    if ending == "answer":
        assert [row["state"] for row in rows] == ["settled", "released", "settled"]
        first, repaired = [payload for payload, _ in gateway.uploads][1:]
        assert any("nativeContinuation" in m for m in first["messages"])
        assert not any("nativeContinuation" in m for m in repaired["messages"])
        for payload in (first, repaired):
            assert [m["content"] for m in payload["messages"] if m["role"] == "tool"] == [
                "verified read A", "completed review B"]
            assert any("late child result" in str(m["content"]) for m in payload["messages"])
            assert payload["tools"] == final.tool_schemas
        assert any(o["model_role"] == role and o["model_route"] == ROUTE_B for o in observations)
        repaired_fit = dispatched_attempts(ctx.drive_root)[-1]["physical_context"]
        assert repaired_fit["route_fp"] == "capacity-account-b"
        assert ctx.tools._ctx.context_fit_plan.core_sha256 == ctx.context_fit_plan.core_sha256
        if priced:
            assert len(prices) == 3 and prices[0] != prices[1] == prices[2]


def test_original_fallback_ordinal_keeps_account_after_filters(tmp_path, monkeypatch):
    from tests.test_loop_compaction import _ctx

    ctx = _ctx(tmp_path)
    configured = [ctx.active_model, MODEL, MODEL, "openai::third"]
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", ",".join(configured))
    seen = []
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda _plan, *_args, **kwargs: (seen.append(kwargs["model_role"]) or _plan, "max"))
    monkeypatch.setattr(loop, "_call_round_model", lambda call: (None, 0, "max"))
    from ouroboros import fallback_cooldown
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_args: False)
    loop._run_cross_model_fallback_chain(
        llm=ctx.llm, ctx=ctx.tools._ctx, tools=ctx.tools, messages=ctx.messages,
        active_model=ctx.active_model, active_use_local=False, tool_schemas=[], active_effort="medium",
        max_retries=1, drive_logs=ctx.drive_logs, task_id=ctx.task_id, round_idx=1,
        event_queue=None, accumulated_usage={}, task_type="task",
        emit_progress=lambda _text, *, incident=None: None,
        context_fit_plan=ctx.context_fit_plan, active_context_mode="max")
    assert seen == ["fallback:1", "fallback:3"]


def test_same_round_delivery_uses_the_route_changed_inside_model_call(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolRegistry

    class Client:
        def default_model(self):
            return MODEL

    def switched(call):
        call.active_model, call.active_use_local = "local-destination", True
        return {"role": "assistant", "content": "finished"}, 0, "max"

    def final(_content, limit, trace, tools, *_args, explicit_candidate=False, resume_candidate=False):
        assert not explicit_candidate and not resume_candidate  # This fixture's first ordinary reply stays plain.
        assert limit.active_model == tools._ctx.active_model == "local-destination"
        assert limit.active_use_local is tools._ctx.active_use_local is True
        return "finished", limit.accumulated_usage, trace

    monkeypatch.setattr(loop, "_call_round_model", switched)
    monkeypatch.setattr(loop, "_no_tool_final_answer", final)
    value, _usage, _trace = loop.run_llm_loop(
        messages=[{"role": "user", "content": "go"}],
        tools=ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path), llm=Client(),
        drive_logs=tmp_path / "logs", emit_progress=lambda _text, **_kwargs: None,
        incoming_messages=queue.Queue(), task_id="route-delivery", drive_root=tmp_path)
    assert value == "finished"


TURN = {"route": ROUTE, "format": "codex.turn.v1", "payload": {"turnState": "live-turn"}}


@pytest.fixture
def turn_engine(monkeypatch):
    """A serving engine whose strict request schema accepts the active-turn field."""
    from ouroboros import config, llm_claudexor

    monkeypatch.setattr(llm_claudexor, "owned_engine_version",
                        lambda: config.CLAUDEXOR_MODEL_TURN_STATE_MIN_VERSION)


def _slot(envelope=None):
    from ouroboros.llm_claudexor import ModelTurnState

    return ModelTurnState(deepcopy(envelope) if envelope else None)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_same_route_wait_and_reprepare_update_the_original_slot(main_call, turn_engine, asynchronous, monkeypatch):
    """Both real reprepare routes deep-copy their kwargs and must keep ONE owner.

    The synchronous case goes through the live quota wait, the asynchronous one
    through the proven-un-sent account repair and its thread offload.
    """
    ctx, gateway, controller, _events, _decide, _observations = main_call
    saved_settings(monkeypatch, ctx.drive_root, OUROBOROS_MODEL_FALLBACKS="")  # the wait is in place
    gateway.operation_catalog = [{"method": "POST", "path": "/v2/model-operations", "parameters": [
        {"name": "typedRefusals", "location": "query", "enum": ["true"]}]}]
    landed = {**result(route=ROUTE_B if asynchronous else ROUTE), "nativeContinuation": deepcopy(TURN)}
    gateway.results = [_failed("invalid_continuation", ROUTE_B) if asynchronous
                       else _failed("subscription_window_exhausted"), landed]
    gateway.dispatch = ["not_started", "response_received"]
    slot = _slot()
    ctx.tools._ctx.model_turn_state = slot
    if asynchronous:
        disposition = loop._measure_round_main_fit(ctx, automatic_pass_used=False)
        with controller.register_reprepare("main", lambda values: _reprepare_waiting_main(ctx, values)):
            with ua.bind_physical_attempt_context(loop._physical_context_for_fit(disposition)):
                asyncio.run(ctx.llm.chat_async(ctx.messages, MODEL, model_role="main", model_turn_state=slot))
    else:
        assert _dispatch(ctx)[0]
    # The slot the loop still owns is the one the durable result must have replaced.
    assert ctx.tools._ctx.model_turn_state is slot and slot.envelope == TURN
    assert len(gateway.accepted_operations) == 2 and gateway.uploads[-1][0]["nativeContinuation"] is None
    # The loop opted in. Direct chat_async used its strict default, even with the same Main reprepare.
    assert [request.get("typed_refusals") for request in gateway.capture_requests] == [None if asynchronous else True] * 2


def test_a_helper_call_cannot_overwrite_the_running_loop_slot(setup, turn_engine):
    _root, gateway, client = setup
    gateway.results = [{**result(), "nativeContinuation": deepcopy(TURN)}]
    slot = _slot({"route": ROUTE, "format": "codex.turn.v1", "payload": {"turnState": "main-turn"}})
    client.chat([{"role": "user", "content": "compact this"}], MODEL, model_role="light")
    assert slot.envelope["payload"]["turnState"] == "main-turn"
    assert "nativeContinuation" not in gateway.uploads[-1][0]


@pytest.mark.parametrize("destination,use_local", [("openai::alternate", False), ("local-model", True)])
def test_leaving_this_transport_ends_the_turn_and_returning_does_not_revive_it(
    setup, turn_engine, monkeypatch, destination, use_local,
):
    _root, gateway, client = setup
    answer = ({"role": "assistant", "content": "elsewhere"}, {"cost": None, "provider": "other"})
    subscription, remote = {"on": False}, client._chat_remote
    monkeypatch.setattr(client, "_chat_remote",
                        lambda *args, **kwargs: remote(*args, **kwargs) if subscription["on"] else deepcopy(answer))
    monkeypatch.setattr(client, "_chat_local", lambda *_args, **_kwargs: deepcopy(answer))
    slot = _slot(TURN)
    client.chat([{"role": "user", "content": "hi"}], destination, use_local=use_local, model_turn_state=slot)
    assert slot.envelope is None
    subscription["on"] = True
    client.chat([{"role": "user", "content": "hi"}], MODEL, model_turn_state=slot)
    assert gateway.uploads[-1][0]["nativeContinuation"] is None


def _run_loop(tmp_path, monkeypatch, rounds, registry=None):
    """Drive one real run_llm_loop invocation, recording the slot every round saw."""
    from ouroboros.tools.registry import ToolRegistry

    seen, replies = [], iter(rounds)

    def call_round(call):
        slot = call.tools._ctx.model_turn_state
        seen.append((slot, deepcopy(slot.envelope)))
        reply = next(replies)
        return reply(call) if callable(reply) else reply

    def tools_then_steering(calls, _tools, _logs, _task, _executor, messages, *_args,
                            fit_candidate=None, tool_schemas=None):
        messages.append({"role": "tool", "tool_call_id": calls[0]["id"], "content": "done"})
        # Owner steering and host notices arrive as user turns INSIDE one loop;
        # they must never be read as the start of a new transport turn (P5).
        messages.append({"role": "user", "content": "[SYSTEM NOTICE]\nkeep going"})

    monkeypatch.setattr(loop, "_call_round_model", call_round)
    monkeypatch.setattr(loop, "_no_tool_final_answer",
                        lambda _content, limit, trace, *_args, explicit_candidate=False, resume_candidate=False:
                        ("finished", limit.accumulated_usage, trace))
    monkeypatch.setattr(loop, "handle_tool_calls", tools_then_steering)
    registry = registry if registry is not None else ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    loop.run_llm_loop(
        messages=[{"role": "user", "content": "go"}], tools=registry,
        llm=SimpleNamespace(default_model=lambda: MODEL), drive_logs=tmp_path / "logs",
        emit_progress=lambda _text, **_kwargs: None, incoming_messages=queue.Queue(),
        task_id="turn-loop", drive_root=tmp_path)
    return registry, seen


def test_every_round_of_one_loop_shares_its_slot_and_a_next_loop_starts_empty(tmp_path, monkeypatch):
    def tool_round(call):
        call.tools._ctx.model_turn_state.envelope = deepcopy(TURN)  # the engine answered
        return ({"role": "assistant", "content": "", "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]}, 0, "max")

    final_round = ({"role": "assistant", "content": "finished"}, 0, "max")
    registry, seen = _run_loop(tmp_path, monkeypatch, [tool_round, final_round])
    (first_slot, first_value), (second_slot, second_value) = seen
    assert first_slot is second_slot and first_value is None and second_value == TURN
    # A next loop over the SAME context and history — the shape a cold restart
    # takes — opens a new turn instead of replaying the finished one.
    _registry, again = _run_loop(tmp_path, monkeypatch, [final_round], registry=registry)
    assert again[0][0] is not first_slot and again[0][1] is None


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_legacy_shaped_exchange_leaves_a_live_turn_untouched(setup, monkeypatch, asynchronous):
    """A legacy result is SILENCE about the turn, not a disclaimer (BIBLE P1).

    The version floor can be closed while the caller already holds a token: an
    engine this process has never proven, a slot armed by an earlier proven one.
    Such a request asks nothing about the turn and its result answers nothing,
    so adopting its absent field would discard a token the engine never dropped.
    """
    from ouroboros import llm_claudexor

    _root, gateway, client = setup
    monkeypatch.setattr(llm_claudexor, "owned_engine_version", lambda: "")
    slot, messages = _slot(TURN), [{"role": "user", "content": "hi"}]
    if asynchronous:
        asyncio.run(client.chat_async(messages, MODEL, model_turn_state=slot))
    else:
        client.chat(messages, MODEL, model_turn_state=slot)
    assert "nativeContinuation" not in gateway.uploads[-1][0]
    assert slot.envelope == TURN


@pytest.mark.parametrize("asynchronous", [False, True])
def test_an_opted_in_exchange_still_adopts_and_then_clears_the_turn(setup, turn_engine, asynchronous):
    """Both entrypoints keep the opt-in contract the legacy guard sits beside."""
    _root, gateway, client = setup
    gateway.results = [{**result(), "nativeContinuation": deepcopy(TURN)}, result()]
    gateway.dispatch = ["response_received"] * 2
    slot, messages = _slot(), [{"role": "user", "content": "hi"}]

    def send():
        if asynchronous:
            asyncio.run(client.chat_async(messages, MODEL, model_turn_state=slot))
        else:
            client.chat(messages, MODEL, model_turn_state=slot)

    send()
    assert slot.envelope == TURN and gateway.uploads[0][0]["nativeContinuation"] is None
    send()
    # The engine answered without an envelope: the turn is stateless, not stale.
    assert gateway.uploads[-1][0]["nativeContinuation"] == TURN and slot.envelope is None


# Main execution affinity follows the actual route after a live role override.
CACHE_REPREPARE_API = "openrouter::openai/fixture-model"
CACHE_REPREPARE_EXECUTION = "execution-reprepare-proof"


@pytest.mark.parametrize(
    "destination,use_local",
    [(CACHE_REPREPARE_API, False), ("local-fixture", True), (MODEL, False)],
    ids=["claudexor-to-openrouter", "claudexor-to-local", "same-claudexor-control"],
)
def test_live_owner_wait_reprojects_affinity(main_call, monkeypatch, destination, use_local):
    ctx, gateway, controller, events, decide, _ = main_call
    saved_settings(monkeypatch, ctx.drive_root, OUROBOROS_MODEL_FALLBACKS="")  # the wait is in place
    ctx.accumulated_usage["execution_id"] = CACHE_REPREPARE_EXECUTION
    gateway.results = [_failed("subscription_window_exhausted"), result(route=ROUTE_B)]
    gateway.dispatch = ["not_started", "response_received"]
    # The selected API fixture uses an explicit synthetic tariff, never a live pricing lookup.
    monkeypatch.setattr(ua, "estimate_cost_optional", lambda *a, **kw: 0.0)
    decisions = []
    wire = []
    prepared = []
    from ouroboros import loop_model_call

    reprepare = loop_model_call._reprepare_waiting_main

    def observed_reprepare(call, values):
        answer = reprepare(call, values)
        prepared.append(
            {
                "model": answer.kwargs["model"],
                "use_local": answer.kwargs.get("use_local"),
                "cache_affinity": answer.kwargs.get("cache_affinity"),
            }
        )
        return answer

    monkeypatch.setattr(loop_model_call, "_reprepare_waiting_main", observed_reprepare)

    def catalog(*args, **kwargs):
        wait = next(x for x in reversed(list(events.queue)) if x.get("type") == "task_model_wait")
        response = _switch_wait(decide, wait, destination, use_local=use_local)
        assert response.status_code == 202
        decisions.append(json.loads(response.body))
        return {}

    monkeypatch.setattr(ctx.llm, "claudexor_model_catalog", catalog)
    remote = ctx.llm._chat_remote

    def direct(messages, model, tools):
        target = {"provider": "local" if use_local else "openrouter", "usage_model": model}
        payload = {"messages": deepcopy(messages), "tools": tools or [], "model": model}
        request = _attempt_request(target, payload)
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "cost": 0,
            "provider": target["provider"],
            "resolved_model": model,
            "cost_final": True,
        }
        return ua.execute_physical_attempt(
            request,
            lambda: ({"role": "assistant", "content": "switched"}, usage),
            extractor=lambda value: (value[1], 0, True),
            before_dispatch=_candidate_before_dispatch(payload, request),
        )

    def remote_call(target, messages, tools, *args, **kwargs):
        wire.append(
            {
                "provider": target["provider"],
                "model": target.get("resolved_model"),
                "cache_affinity": kwargs.get("cache_affinity"),
            }
        )
        if target["provider"] == "claudexor":
            return remote(target, messages, tools, *args, **kwargs)
        return direct(messages, destination, tools)

    monkeypatch.setattr(ctx.llm, "_chat_remote", remote_call)
    monkeypatch.setattr(
        ctx.llm, "_chat_local", lambda messages, tools, *args, **kwargs: direct(messages, destination, tools)
    )
    answer, _, _ = _dispatch(ctx)
    assert answer and len(decisions) == 1 and prepared
    facts = {
        "route": destination,
        "use_local": use_local,
        "prepared": prepared,
        "remote_boundaries": wire,
        "execution_id": ctx.accumulated_usage["execution_id"],
        "owner_switch_saved": decisions[0]["saved"],
        "completed_tool_texts": [x["content"] for x in ctx.messages if x.get("role") == "tool"],
        "gateway_operations": len(gateway.accepted_operations),
    }
    assert facts["completed_tool_texts"] == ["verified read A", "completed review B"]
    assert ctx.accumulated_usage["execution_id"] == CACHE_REPREPARE_EXECUTION
    assert prepared[-1]["cache_affinity"] == ("" if use_local or destination != MODEL else cache_key_for_model(MODEL)), (
        facts
    )
    if not use_local:
        assert wire[-1]["cache_affinity"] == prepared[-1]["cache_affinity"]


@pytest.mark.parametrize(
    "initial_model,initial_local",
    [(CACHE_REPREPARE_API, False), ("local-fixture", True)],
    ids=["openrouter-to-claudexor", "local-to-claudexor"],
)
def test_recorded_wait_override_reprojects_affinity_before_send(main_call, initial_model, initial_local):
    ctx, gateway, controller, events, decide, _ = main_call
    ctx.accumulated_usage["execution_id"] = CACHE_REPREPARE_EXECUTION
    ctx.active_model = ctx.tools._ctx.active_model = initial_model
    ctx.active_use_local = initial_local
    ctx.context_fit_plan = replace(
        ctx.context_fit_plan,
        model=initial_model,
        provider="local" if initial_local else "openrouter",
        model_route={},
        route_fp="prior-api-local",
    )
    ctx.tools._ctx.context_fit_plan = ctx.context_fit_plan
    # Existing task-local override is the ordinary model_wait.prepare branch;
    # API/local have no subscription quota wait of their own.
    controller.overrides["main"] = {"model": MODEL, "use_local": False, "model_account_override": ""}
    answer, _, _ = _dispatch(ctx)
    assert answer and len(gateway.accepted_operations) == 1
    payload = gateway.uploads[0][0]
    facts = {
        "initial_model": initial_model,
        "initial_local": initial_local,
        "active_model": ctx.active_model,
        "cache_key": payload["options"].get("cacheKey"),
        "execution_id": ctx.accumulated_usage["execution_id"],
        "gateway_operations": len(gateway.accepted_operations),
        "completed_tool_texts": [x["content"] for x in ctx.messages if x.get("role") == "tool"],
    }
    assert facts["completed_tool_texts"] == ["verified read A", "completed review B"]
    assert ctx.accumulated_usage["execution_id"] == CACHE_REPREPARE_EXECUTION
    # The override re-prepared the send for the subscription route, so the wire
    # carries the install-scoped Codex key that route shares across executions.
    assert payload["options"].get("cacheKey") == cache_key_for_model(MODEL), facts
