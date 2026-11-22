"""Synthetic transport/usage checks; no Go subscription, network, or recorded wire fixture."""

from __future__ import annotations

import asyncio
import json
import queue
import time
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from ouroboros import __version__, net_transport, pricing
from ouroboros import usage_accounting as ua
from ouroboros.llm import LLMClient
from ouroboros.llm_routing import resolve_target
from ouroboros.llm_stream import ProviderStreamError
from ouroboros.opencode_go import GO_CHAT_BASE_URL, GO_MODELS_URL, compatible_models_url, go_chat_headers
from ouroboros.provider_models import CONNECTIONS_KEY
from tests._openai_sse import streamed_reply

MODEL = "openai-compatible@go-connection::example-model"
SESSION = "x-opencode-session"
MESSAGE = {"role": "user", "content": "Please measure two plus two."}
TOOLS = [{"type": "function", "function": {"name": "measure", "parameters": {"type": "object"}}}]


def connection_settings(url=GO_CHAT_BASE_URL):
    return {CONNECTIONS_KEY: [{"kind": "openai-compatible", "name": "go-connection",
                              "api_key": "synthetic-key", "base_url": url}]}


def response(message=None, *, usage=None, finish_reason="stop"):
    return httpx.Response(200, json={"id": "reply", "object": "chat.completion", "created": 0,
                                    "model": "example-model", "usage": usage,
                                    "choices": [{"index": 0, "finish_reason": finish_reason,
                                                 "message": message or {"role": "assistant", "content": "4"}}]})


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    import ouroboros.request_wire_contract as wire
    from ouroboros.install_identity import acting_as

    monkeypatch.setattr(wire, "canonical_wire_evidence_root", lambda: tmp_path)
    # The real estimator has a price for the same bare model on another service.
    monkeypatch.setattr(pricing, "_cached_pricing", {"openrouter": {"example-model": (2.0, None, None, 4.0)}})
    monkeypatch.setattr(pricing, "_pricing_fetched_at", {"openrouter": time.time()})
    monkeypatch.setattr(net_transport, "env_proxies_configured", lambda: False)
    monkeypatch.setattr(net_transport, "extra_ca_bundle", lambda: None)
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    with acting_as(tmp_path):
        yield ua.UsageScope(drive_root=tmp_path, task_id="go-task", root_task_id="same-root", source="agent.task",
                            billing_group_id="same-root", billing_group_limit_source="synthetic_fixture")


def client_for(monkeypatch, handler, url=GO_CHAT_BASE_URL):
    monkeypatch.setattr(net_transport, "remote_httpx_transport", lambda *_a, **_k: httpx.MockTransport(handler))
    client = LLMClient()
    monkeypatch.setattr(client, "_resolve_remote_target",
                        lambda model, settings=None: resolve_target(model, settings or connection_settings(url)))
    return client


def assert_durable_price(scope, expected, final):
    records = ua.read_usage_records(scope.drive_root)
    assert len(records) == 1
    row = records[0]
    assert row["state"] == "settled" and not row["pricing_known"]
    assert (row.get("cost_usd"), bool(row["cost_final"])) == (expected, final)


@pytest.mark.parametrize("delivery", ["sync", "async", "no_proxy_sync", "no_proxy_async"])
def test_same_host_conversation_through_chat_tool_roundtrip(monkeypatch, isolated, delivery):
    sent = []
    tool = {"id": "one", "type": "function", "function": {"name": "measure", "arguments": "{}"}}

    def peer(request):
        sent.append(request)
        messages = json.loads(request.content)["messages"]
        if messages[-1]["role"] == "tool":
            assert messages[-1]["tool_call_id"] == tool["id"]
            return response()
        return response({"role": "assistant", "tool_calls": [tool]}, finish_reason="tool_calls")

    client = client_for(monkeypatch, peer, GO_CHAT_BASE_URL + "/")

    def chat(messages, affinity, identity):
        args = dict(messages=messages, model=MODEL, tools=TOOLS, max_tokens=128,
                    cache_affinity=affinity, chat_conversation_id=identity, no_proxy=delivery.startswith("no_proxy"))
        if delivery.endswith("async"):
            return asyncio.run(client.chat_async(**args))[0]
        return client.chat(**args)[0]

    with ua.usage_scope(isolated):
        assistant = chat([MESSAGE], "first-cache-key", "acting-turn")
        assert assistant["tool_calls"][0]["id"] == tool["id"]
    # A re-entered task keeps its conversation even if its usage category changes.
    with ua.usage_scope(replace(isolated, category="chat")):
        assert chat([MESSAGE, assistant, {"role": "tool", "tool_call_id": "one", "content": "4"}],
                    "another-cache-key", "acting-turn")["content"] == "4"
    with ua.usage_scope(replace(isolated, task_id="another-child")):
        chat([MESSAGE], "first-cache-key", "child-turn")
    assert len(sent) == 3
    for req in sent:
        assert str(req.url) == GO_CHAT_BASE_URL + "/chat/completions"
        assert req.headers["User-Agent"] == f"Ouroboros/{__version__}"
    assert sent[0].headers[SESSION] == sent[1].headers[SESSION] != sent[2].headers[SESSION]


def test_light_acts_through_a_tool_round_on_the_same_conversation(monkeypatch, isolated, tmp_path):
    from ouroboros.loop import run_llm_loop
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.registry import ToolRegistry

    sent = []
    tool = {"id": "inspect", "type": "function", "function": {
        "name": "read_file", "arguments": '{"path":"result.txt"}'}}
    (tmp_path / "result.txt").write_text("Synthetic result: four.\n", encoding="utf-8")

    def peer(request):
        sent.append(request.headers[SESSION])
        if len(sent) == 1:
            return streamed_reply(tool_call=tool)
        assert len(sent) == 2
        messages = json.loads(request.content)["messages"]
        assert any(m["role"] == "tool" and m["tool_call_id"] == tool["id"] and "Synthetic result: four." in m["content"]
                   for m in messages)
        return streamed_reply()

    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", MODEL)
    monkeypatch.setenv("USE_LOCAL_LIGHT", "false")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_metadata = {"model_role": "light"}
    write_task_result(tmp_path, isolated.task_id, "running", root_task_id=isolated.root_task_id, _attempt=1)
    with ua.usage_scope(isolated):
        text, usage, trace = run_llm_loop([MESSAGE.copy()], registry, client_for(monkeypatch, peer), tmp_path / "logs",
                                         lambda _text, **_kw: None, queue.Queue(), task_id=isolated.task_id, drive_root=tmp_path)
    assert registry._ctx.primary_route["role"] == "light" and usage["initial_model_request"]["model"] == MODEL
    assert text == "4" and [entry["tool"] for entry in trace["tool_calls"]] == ["read_file"]
    assert len(sent) == 2 and sent[0] == sent[1]


def test_acting_conversation_survives_role_change_but_not_task_or_model_change(monkeypatch, isolated):
    from ouroboros.llm_observability import chat_observed
    from ouroboros.loop_llm_call import call_llm_with_retry

    sent = []

    def peer(request):
        sent.append(request.headers[SESSION])
        return streamed_reply() if json.loads(request.content).get("stream") else response()

    client = client_for(monkeypatch, peer)
    usage = {}

    def turn(task_id, role, round_idx, model=MODEL):
        msg, _ = call_llm_with_retry(client, [MESSAGE.copy()], model, None, "medium", 1,
                                     isolated.drive_root / "logs", task_id, round_idx, None, usage,
                                     model_role=role, max_tokens=32)
        assert msg["content"] == "4"

    with ua.usage_scope(isolated):
        turn(isolated.task_id, "light", 1)
        for _ in range(2):
            # Safety's exchanges share task attribution and model, not the task's conversation.
            chat_observed(client, drive_root=isolated.drive_root, task_id=isolated.task_id,
                          call_type="safety_supervisor", messages=[MESSAGE], model=MODEL,
                          model_role="safety", max_tokens=32)
        turn(isolated.task_id, "fallback:0", 2)
        turn(isolated.task_id, "main", 3, MODEL.replace("example-model", "other-model"))
        turn("continue-successor", "light", 1)
    assert sent[0] == sent[3] and len(set(sent)) == 5


@pytest.mark.parametrize("role", ["main", "light", "vision"])
def test_auxiliary_role_without_a_bound_conversation_gets_fresh_sessions(monkeypatch, isolated, role):
    sent = []

    def peer(request):
        sent.append(request.headers[SESSION])
        return response()

    client = client_for(monkeypatch, peer)
    with ua.usage_scope(isolated):
        client.chat([MESSAGE], MODEL, model_role=role, max_tokens=32)
        asyncio.run(client.chat_async([MESSAGE], MODEL, model_role=role, max_tokens=32))
    assert len(set(sent)) == 2


@pytest.mark.parametrize("last_cash,subtotal,partial,estimated,measured", [
    ({"cost": 0}, 0.25, False, False, 0.25),
    pytest.param("refused_with_zero", 0.25, False, False, 0.25, id="refusal-settled-at-zero"),
    ({}, 0.25, True, False, None),
    (None, 0.25, True, False, None),  # last send crossed the boundary, but no bill arrived
    ({"cost": 0.125, "opencode": {"estimated": True}}, 0.375, False, True, None),
])
def test_native_review_keeps_conversation_and_price_truth_through_history(monkeypatch, isolated, tmp_path,
                                                                        last_cash, subtotal, partial, estimated, measured):
    from ouroboros import deep_self_review, review_ledger, reviewer_slot_config
    from ouroboros.tools import review_change
    from ouroboros.tools.registry import ToolContext

    root = tmp_path / "body"
    root.mkdir()
    (root / "BIBLE.md").write_text("# Fixture constitution\n", encoding="utf-8")
    sent = []
    report = "The inspection read the fixture constitution."
    tool = {"id": "read", "type": "function", "function": {
        "name": "read_file", "arguments": '{"path":"BIBLE.md"}'}}
    report_delivered = isinstance(last_cash, dict)

    def peer(request):
        messages = json.loads(request.content)["messages"]
        sent.append(request.headers[SESSION])
        counts = {"prompt_tokens": 23, "completion_tokens": 4}
        if any(message["role"] == "tool" for message in messages):
            assert "Fixture constitution" in messages[-1]["content"]
            if last_cash is None:
                raise httpx.ReadError("Synthetic reply was not available", request=request)
            if last_cash == "refused_with_zero":
                # The SDK keeps this top-level error body, including the explicit cash receipt.
                return httpx.Response(400, json={"message": "Synthetic review refused",
                                                  "usage": {**counts, "cost": 0}})
            return response({"role": "assistant", "content": report}, usage={**counts, **last_cash})
        return response({"role": "assistant", "tool_calls": [tool]}, usage={**counts, "cost": 0.25}, finish_reason="tool_calls")

    client = client_for(monkeypatch, peer)
    row = reviewer_slot_config.ConfiguredReviewerSlot("fixture-seat", "api_chat", MODEL, use_local=False)
    monkeypatch.setattr(review_change, "_system_repo", lambda _ctx: root)
    monkeypatch.setattr(review_change, "system_review_row", lambda *_a: row)
    monkeypatch.setattr(deep_self_review, "deep_review_route", lambda row: ("", row.target_id))
    monkeypatch.setattr(reviewer_slot_config, "_last_execution_path", lambda: tmp_path / "seat-executions.json")
    ctx = ToolContext(repo_dir=root, drive_root=tmp_path)
    ctx.task_id = isolated.task_id
    check_session_isolation = last_cash == {"cost": 0}
    # A standalone system inspection has no admitted task to give its registry Pause custody.
    with ua.usage_scope(replace(isolated, non_task_operation=True)):
        result = review_change.run_system_review(ctx, llm=client)
        assert result["report_delivered"] is report_delivered
        if report_delivered:
            assert result["report"].endswith(report)
        usage = result["usage"]
        assert usage["native_rounds"] == 2 and len(set(usage["ledger_attempt_ids"])) == 2
        assert (usage["cost"], usage["cost_partial"], usage["cost_estimated"], usage["cost_final"]) == (
            subtotal, partial, estimated, not (partial or estimated))
        record = review_ledger.load_record(tmp_path, result["record_id"])
        seat, = record["rows"]
        assert seat["cost_usd"] == measured
        assert record["cost"] == {"usd": measured or 0.0, "unknown": measured is None}
        public = reviewer_slot_config.public_slot_executions(reviewer_slot_config.reviewer_slot_last_executions())
        assert public[row.slot_id]["cost"] == {"kind": "usd" if measured is not None else "unknown", "usd": measured}
        if check_session_isolation:
            client.chat([MESSAGE], MODEL, chat_conversation_id="test-acting", max_tokens=32)
            assert review_change.run_system_review(ctx, llm=client)["report_delivered"]
    assert sent[0] == sent[1]
    if check_session_isolation:
        first, _, main, second, again = sent
        assert second == again and len({main, first, second}) == 3


def test_wait_reprepare_keeps_acting_conversation_but_helper_has_its_own(monkeypatch, isolated, tmp_path):
    from ouroboros.loop_llm_call import call_llm_with_retry
    from ouroboros.model_wait import task_model_wait_scope
    from ouroboros.task_results import write_task_result

    sent = []

    def peer(request):
        sent.append(request.headers[SESSION])
        return streamed_reply() if json.loads(request.content).get("stream") else response()

    client = client_for(monkeypatch, peer)
    usage = {}

    def turn(round_idx):
        reply, _ = call_llm_with_retry(client, [MESSAGE], MODEL, None, "medium", 1, tmp_path / "logs",
                                       isolated.task_id, round_idx, None, usage, model_role="light", max_tokens=32)
        assert reply["content"] == "4"

    def reprepare(kwargs):
        client.chat([MESSAGE], MODEL, model_role="main", max_tokens=32)
        return kwargs

    write_task_result(tmp_path, isolated.task_id, "running", root_task_id=isolated.task_id, _attempt=1)
    scope = replace(isolated, root_task_id=isolated.task_id)
    with ua.usage_scope(scope), task_model_wait_scope(task={"id": scope.task_id, "_attempt": 1},
            drive_root=tmp_path, event_queue=None, worker_slot_held=False) as waiter:
        turn(1)
        # Rebuild on a same-model owner override; the helper uses the same task's accounting context.
        waiter.overrides["light"] = {"model": MODEL, "model_account_override": ""}
        with waiter.register_reprepare("light", reprepare):
            turn(2)
    first, helper, reprepared = sent
    assert first == reprepared != helper


def test_provider_probe_is_a_fresh_conversation(monkeypatch, isolated):
    sent = []

    def peer(request):
        sent.append(request)
        return response()

    client = client_for(monkeypatch, peer)
    with ua.usage_scope(isolated):
        client.chat([MESSAGE], MODEL, chat_conversation_id="ordinary", max_tokens=32)
        assert client.probe_provider_readiness(MODEL, settings=connection_settings())["ok"]
        assert client.probe_provider_readiness(MODEL, settings=connection_settings())["ok"]
    assert len(sent) == 3 and all(req.headers["User-Agent"] == f"Ouroboros/{__version__}" for req in sent)
    assert len({req.headers[SESSION] for req in sent}) == 3


@pytest.mark.parametrize("url", [GO_CHAT_BASE_URL, GO_CHAT_BASE_URL + "/", "HTTPS://OpenCode.ai/inference/go/openai/v1"])
def test_go_route_matches_its_scheme_and_host_in_any_letter_case(monkeypatch, isolated, url):
    sent = []

    def peer(request):
        sent.append(request)
        return response()

    client = client_for(monkeypatch, peer, url)
    with ua.usage_scope(isolated):
        client.chat([MESSAGE], MODEL, chat_conversation_id="test-acting", max_tokens=32)
    request, = sent
    # The transport already folds this spelling onto the one Go address, so it owes that address's headers.
    assert str(request.url) == GO_CHAT_BASE_URL + "/chat/completions"
    assert request.headers["User-Agent"] == f"Ouroboros/{__version__}" and request.headers[SESSION]
    assert compatible_models_url("openai-compatible", url) == GO_MODELS_URL


@pytest.mark.parametrize("url", ["http://opencode.ai/inference/go/openai/v1", "https://example.test/inference/go/openai/v1",
                                 "https://opencode.ai/inference/openai/v1", "https://opencode.ai/inference/go/openai/v1/else",
                                 GO_CHAT_BASE_URL + "//", GO_CHAT_BASE_URL + "?", GO_CHAT_BASE_URL + "#",
                                 # Only the scheme and host ignore letter case: another path, a port or userinfo is not Go.
                                 "HTTPS://OpenCode.ai/zen/go/v1", "HTTPS://OpenCode.ai/Inference/Go/OpenAI/v1",
                                 "https://OpenCode.ai:443/inference/go/openai/v1",
                                 "https://user@OpenCode.ai/inference/go/openai/v1",
                                 "https://openrouter.ai/api/v1"])
def test_other_routes_do_not_get_go_headers(url):
    target = {**resolve_target(MODEL, connection_settings(url)), "chat_conversation_id": "test-acting"}
    assert go_chat_headers(target) == {}
    assert compatible_models_url(target["provider"], url) != GO_MODELS_URL
    assert go_chat_headers({**target, "provider": "openrouter", "base_url": GO_CHAT_BASE_URL}) == {}
    if "openrouter.ai" in url:
        assert target["default_headers"]["HTTP-Referer"]


@pytest.mark.parametrize("no_proxy", [False, True])
def test_go_client_stops_at_redirect(monkeypatch, isolated, no_proxy):
    sent = []

    def peer(request):
        sent.append(request)
        return httpx.Response(307, headers={"Location": "https://other.test/v1/chat/completions"})

    client = client_for(monkeypatch, peer)
    with ua.usage_scope(isolated), pytest.raises(Exception) as denial:
        client.chat([MESSAGE], MODEL, max_tokens=128, no_proxy=no_proxy)
    assert getattr(denial.value, "status_code", None) == 307
    assert [str(req.url) for req in sent] == [GO_CHAT_BASE_URL + "/chat/completions"]


@pytest.mark.parametrize("reported,expected,estimated", [({}, None, False), ({"cost": False}, None, False),
    ({"cost": 0}, 0.0, False), ({"cost": 0.002}, 0.002, False),
    ({"cost": 0.003, "opencode": {"billing_source": "byok", "estimated": True}}, 0.003, True)])
def test_reported_cost_controls_return_and_settlement(monkeypatch, isolated, reported, expected, estimated):
    raw = {"prompt_tokens": 23, "completion_tokens": 4, "total_cost": 18,
           "opencode": {"billing_source": "go", "allowance": 0.05, "list_cost": 0.04,
                        "tokens": {"cache_read": 80}}, **reported}
    observed = []
    settle = ua.settle_attempt

    def record(reservation, usage, *, cost_usd=None, cost_final=False):
        settle(reservation, usage, cost_usd=cost_usd, cost_final=cost_final)
        observed.append((reservation.opencode_go, usage["prompt_tokens"], cost_usd, cost_final))

    monkeypatch.setattr(ua, "settle_attempt", record)
    client = client_for(monkeypatch, lambda _request: response(usage=raw))
    with ua.usage_scope(isolated):
        _, usage = client.chat([MESSAGE], MODEL, max_tokens=128)
    final = expected is not None and not estimated
    assert (usage["cost"], usage["cost_estimated"], usage["cost_final"]) == (expected, estimated, final)
    assert observed == [(True, 23, expected, final)]
    assert usage["prompt_tokens"] == 23 and usage["opencode"] == raw["opencode"]
    loop_cost, _, _, loop_estimated = pricing.normalize_usage_cost(usage, model=MODEL, use_local=False)
    assert (loop_cost, loop_estimated) == (expected, estimated)
    assert_durable_price(isolated, expected, final)


@pytest.mark.parametrize("url", ["https://other.test/v1", "https://openrouter.ai/api/v1"])
def test_other_client_ignores_go_marker_and_keeps_ordinary_cost(monkeypatch, isolated, url):
    def peer(request):
        assert SESSION not in request.headers
        if "openrouter.ai" in url:
            assert request.headers["HTTP-Referer"]
        return response(usage={"prompt_tokens": 23, "completion_tokens": 4, "total_cost": 0.008,
                               "opencode": {"estimated": True}})

    client = client_for(monkeypatch, peer, url)
    with ua.usage_scope(isolated):
        _, usage = client.chat([MESSAGE], MODEL, max_tokens=128)
    assert "opencode_go" not in usage and usage["cost"] == 0.008 and usage["cost_final"]
    row, = ua.read_usage_records(isolated.drive_root)
    assert row["cost_usd"] == 0.008 and row["cost_final"]


@pytest.mark.parametrize("variant", ["named", "dedicated", "legacy"])
def test_unpriced_go_does_not_borrow_a_same_id_model_tariff(monkeypatch, isolated, variant):
    settings = {
        "named": connection_settings(),
        "dedicated": {"OPENAI_COMPATIBLE_API_KEY": "synthetic-key", "OPENAI_COMPATIBLE_BASE_URL": GO_CHAT_BASE_URL},
        "legacy": {"OPENAI_API_KEY": "synthetic-key", "OPENAI_BASE_URL": GO_CHAT_BASE_URL},
    }[variant]
    model = MODEL if variant == "named" else "openai-compatible::example-model"

    def peer(request):
        assert request.headers[SESSION]
        return response(usage={"prompt_tokens": 23, "completion_tokens": 4, "total_cost": 8})

    assert pricing.estimate_cost_optional("example-model", 23, 4, provider="openrouter") == pytest.approx(0.000062)
    client = client_for(monkeypatch, peer)
    monkeypatch.setattr(client, "_resolve_remote_target", lambda model: resolve_target(model, settings))
    with ua.usage_scope(isolated):
        _, usage = client.chat([MESSAGE], model, max_tokens=32)
    assert (usage["cost"], usage["cost_final"]) == (None, False)
    assert pricing.normalize_usage_cost(usage, model=model, use_local=False)[0] is None
    assert_durable_price(isolated, None, False)


@pytest.mark.parametrize("cash,estimated", [(0.02, True), (0, False)])
def test_review_slots_keep_distinct_sessions_and_cash_knowledge_in_live_and_saved_events(monkeypatch, isolated, cash,
                                                                                        estimated):
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request
    from supervisor.events_budget import _handle_llm_usage

    received = []

    def peer(request):
        received.append(request.headers[SESSION])
        return response({"role": "assistant", "content": "[]\nNO_FINDINGS"},
                        usage={"cost": cash, "opencode": ({"billing_source": "byok", "estimated": True}
                                                          if estimated else {"billing_source": "go"}),
                               "prompt_tokens": 23, "completion_tokens": 4})

    client = client_for(monkeypatch, peer)
    ctx = SimpleNamespace(task_id=isolated.task_id, pending_events=[])
    request = ReviewRequest(surface="multi_model_review", goal="Review the fixture", task_id=isolated.task_id,
                            messages=[MESSAGE], max_tokens=32, usage_attribution={"review_wave_id": "fixture-wave"})
    with ua.usage_scope(isolated):
        main = go_chat_headers(resolve_target(MODEL, connection_settings()))[SESSION]
        result = run_review_request(request, slots=[ReviewSlot("one", MODEL), ReviewSlot("two", MODEL)],
                                    drive_root=isolated.drive_root, llm=client, usage_ctx=ctx)
    assert [actor["status"] for actor in result.actors] == ["ok", "ok"]
    assert len({main, *received}) == 3
    rows = {row["attempt_id"]: row for row in ua.read_usage_records(isolated.drive_root)}
    live = []
    host = SimpleNamespace(DRIVE_ROOT=isolated.drive_root, bridge=SimpleNamespace(push_log=live.append))
    for event in ctx.pending_events:
        if event["type"] == "llm_usage":
            _handle_llm_usage(event, host)
    persisted = [json.loads(line) for line in (isolated.drive_root / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row for row in persisted if row["type"] == "llm_usage"] == live and len(live) == 2
    for event in live:
        attempt_id, = event["ledger_attempt_ids"]
        row = rows[attempt_id]
        assert (row["cost_usd"], row["cost_estimated"], row["cost_final"]) == (cash, estimated, not estimated)
        assert (event["cost"], event["cost_estimated"], event["cost_final"]) == (cash, estimated, not estimated)


@pytest.mark.parametrize("delivery", ["sync", "async"])
@pytest.mark.parametrize("reported,complete,expected,final", [
    ({"cost": 0}, True, 0.0, True),
    ({}, False, None, False),
    ({"cost": 0}, False, 0.0, True),
    ({"cost": 0.02, "opencode": {"billing_source": "byok", "estimated": True}}, False, 0.02, False),
])
def test_sdk_stream_policy_matches_durable_settlement(monkeypatch, isolated, delivery, reported, complete,
                                                       expected, final):
    sent, prices = [], []
    raw = {"prompt_tokens": 23, "completion_tokens": 4, "total_cost": 8, **reported}
    chunk = {"id": "synthetic-reply", "object": "chat.completion.chunk", "created": 0, "model": "example-model"}
    body = {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "4"},
                                 "finish_reason": "stop" if complete else None}]}
    trailer = ({**chunk, "choices": [], "usage": raw} if complete else
               {"error": {"type": "synthetic_error"}, "usage": raw})
    sse = "".join(f"data: {json.dumps(frame)}\n\n" for frame in (body, trailer))
    if complete:
        sse += "data: [DONE]\n\n"

    def peer(request):
        sent.append(request)
        assert json.loads(request.content)["stream_options"] == {"include_usage": True}
        return httpx.Response(200, content=sse, headers={"Content-Type": "text/event-stream"})

    settle = ua.settle_attempt

    def record(reservation, usage, *, cost_usd=None, cost_final=False, estimate_cache_ttl=None):
        settle(reservation, usage, cost_usd=cost_usd, cost_final=cost_final,
               estimate_cache_ttl=estimate_cache_ttl)
        prices.append((reservation.opencode_go, usage["prompt_tokens"], cost_usd, cost_final))

    monkeypatch.setattr(ua, "settle_attempt", record)
    client = client_for(monkeypatch, peer)

    def chat():
        kwargs = dict(max_tokens=128, stream=True)
        return (asyncio.run(client.chat_async([MESSAGE], MODEL, **kwargs)) if delivery == "async"
                else client.chat([MESSAGE], MODEL, **kwargs))

    with ua.usage_scope(isolated):
        if complete:
            message, usage = chat()
            assert message["content"] == "4" and usage["cost"] == expected
        else:
            with pytest.raises(ProviderStreamError):
                chat()
    assert len(sent) == 1 and sent[0].headers[SESSION]
    assert prices == [(True, 23, expected, final)]
    assert_durable_price(isolated, expected, final)


@pytest.mark.parametrize("url,address,expected", [(GO_CHAT_BASE_URL + "/", GO_MODELS_URL, (120000, 16000)),
    ("https://other.test/v1", "https://other.test/v1/models", (64000, 8000))])
def test_existing_catalog_sets_capacity_and_output_limit(monkeypatch, tmp_path, url, address, expected):
    from ouroboros.response_limits import resolve_response_limit
    from ouroboros.server_window import SOURCE_PROVIDER_METADATA, read_metadata_window

    def get(request_url, *, headers, **_kwargs):
        assert request_url == address
        assert (headers.get("User-Agent") == f"Ouroboros/{__version__}") == (address == GO_MODELS_URL)
        assert SESSION not in headers and headers["Authorization"] == "Bearer synthetic"
        return httpx.Response(200, request=httpx.Request("GET", request_url), json={"data": [
            {"id": "different-model", "limit": {"context": 500000, "output": 50000}},
            {"id": "example-model", "limit": {"context": 120000, "output": 16000},
             "context_length": 64000, "max_output_tokens": 8000},
        ]})

    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.setattr(net_transport, "verify_kwargs", lambda: {})
    monkeypatch.setattr("ouroboros.vision_routing.record_catalog_image_input", lambda *_a, **_k: None)
    binding = {"provider": "openai-compatible", "base_url": url, "model": MODEL}
    assert read_metadata_window(**binding, api_key="synthetic", evidence_root=tmp_path) == (
        expected[0], SOURCE_PROVIDER_METADATA,
    )
    assert resolve_response_limit(tmp_path, **binding).ceiling(50000) == expected[1]


@pytest.mark.parametrize("url,address", [(GO_CHAT_BASE_URL + "/", GO_MODELS_URL),
                                           ("https://other.test/v1", "https://other.test/v1/models")])
def test_named_picker_and_provider_test_use_the_connection_catalog(url, address):
    from ouroboros.gateway.models import _provider_specs

    def peer(request):
        assert request.method == "GET" and str(request.url) == address
        assert request.headers["Authorization"] == "Bearer synthetic-key" and SESSION not in request.headers
        assert (request.headers.get("User-Agent") == f"Ouroboros/{__version__}") == (address == GO_MODELS_URL)
        return httpx.Response(200, json={"data": [{"id": "example-model", "name": "Example"}]})

    async def load():
        loader, = (loader for source, loader in _provider_specs(connection_settings(url))
                   if source["id"] == "openai-compatible@go-connection")
        async with httpx.AsyncClient(transport=httpx.MockTransport(peer)) as client:
            return await loader(client)

    model, = asyncio.run(load())
    assert model["value"] == MODEL


def test_non_generating_go_connectivity_probe_identifies_the_client_only(monkeypatch):
    from ouroboros.llm_probe import upstream_transport_reachable

    def peer(request):
        assert request.method == "HEAD" and str(request.url) == GO_CHAT_BASE_URL
        assert request.headers["User-Agent"] == f"Ouroboros/{__version__}"
        assert SESSION not in request.headers and "Authorization" not in request.headers
        return httpx.Response(204)

    client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client(transport=httpx.MockTransport(peer), **kwargs))
    llm = SimpleNamespace(_resolve_remote_target=lambda _: resolve_target(MODEL, connection_settings()),
                          _no_proxy_timeout=lambda timeout: httpx.Timeout(timeout))
    assert upstream_transport_reachable(llm, MODEL, timeout=1)["status_code"] == 204
