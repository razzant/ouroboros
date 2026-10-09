"""Processing reaches the exact physical send without changing cognitive input."""

import asyncio
import copy
import json
import sys
from types import SimpleNamespace

import pytest

from ouroboros import config, model_wait, pricing, usage_accounting as ua
from ouroboros.llm import LLMClient
from ouroboros.llm_attempt import apply_processing_preference, processing_contract_headers
from tests._usage_store_testing import ledger_rows


@pytest.fixture
def transport(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setattr(config, "DATA_DIR", root)
    monkeypatch.setattr(config, "SETTINGS_PATH", root / "settings.json")
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_PROCESSING_PREFERENCE", "")
    monkeypatch.setenv("OUROBOROS_MODEL_PROCESSING_PREFERENCES", "{}")
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(pricing, "estimate_cost_optional", lambda *a, **kw: None)
    monkeypatch.setattr(ua, "estimate_cost_optional", lambda *a, **kw: None)
    monkeypatch.setattr("ouroboros.llm.in_worker_process", lambda: False)
    monkeypatch.setattr(LLMClient, "_SUPPORTED_PARAMS_FETCHED", True)
    monkeypatch.setattr(LLMClient, "_SUPPORTED_PARAMS_CACHE", {})
    monkeypatch.setattr(LLMClient, "_get_supported_parameters", lambda *a: None)
    monkeypatch.setattr(LLMClient, "_fetch_generation_cost", lambda *a: None)
    sent = []

    class Response:
        status_code = 200

        def __init__(self, candidate):
            self.candidate = candidate

        def model_dump(self):
            if "speed" in self.candidate:
                return self.json()
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 1, "cost": 0},
                    "service_tier": self.candidate.get("service_tier")}

        def json(self):
            return {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 1,
                              "speed": self.candidate.get("speed")}}

    def create(**candidate):
        sent.append(copy.deepcopy(candidate))
        return Response(candidate)

    async def create_async(**candidate):
        return create(**candidate)

    client = LLMClient(api_key="test-key")
    monkeypatch.setattr(client, "_get_remote_client", lambda target: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    monkeypatch.setattr(client, "_get_async_remote_client", lambda target: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_async))))

    def post(url, *, headers, json, **kwargs):
        sent.append({"payload": copy.deepcopy(json), "headers": copy.deepcopy(headers)})
        return Response(json)

    monkeypatch.setattr("requests.post", post)
    with ua.usage_scope(ua.UsageScope(drive_root=root, task_id="processing", root_task_id="processing")):
        yield root, client, sent


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("provider", ["openai", "openrouter", "anthropic"])
@pytest.mark.parametrize("preference", ["standard", "fast", "economy"])
def test_native_processing_is_in_the_exact_accounted_request(transport, provider, preference, asynchronous):
    root, client, sent = transport
    model = f"{provider}::test-model"
    kwargs = dict(messages=[{"role": "user", "content": "unchanged input"}], model=model,
                  model_role="light", processing_preference=preference, max_tokens=123,
                  reasoning_effort="high")
    message, usage = asyncio.run(client.chat_async(**kwargs)) if asynchronous else client.chat(**kwargs)
    assert message["content"] == "ok"
    assert len(sent) == 1
    expected = ({"standard": "standard", "fast": "fast", "economy": "standard"}
                if provider == "anthropic" else
                {"standard": "default", "fast": "priority", "economy": "flex"})[preference]
    candidate = sent[0]["payload"] if provider == "anthropic" else sent[0]
    assert candidate["speed" if provider == "anthropic" else "service_tier"] == expected
    assert "processing_preference" not in candidate
    assert candidate.get("max_completion_tokens", candidate.get("max_tokens")) == 123
    if provider == "anthropic":
        assert ("fast-mode-2026-02-01" in sent[0]["headers"].get("anthropic-beta", "")) == (preference == "fast")
    rows = ledger_rows(root)
    assert [row["state"] for row in rows] == ["settled"]
    assert all(row["processing_preference"] == preference for row in rows)
    assert all(row["submitted_processing_mode"] == expected for row in rows)
    assert usage["processing"]["requested"] == preference
    assert usage["processing"]["submittedNative"] == expected
    assert usage["processing"]["observedNative"] == [expected]
    assert rows[-1]["processing"] == usage["processing"]
    from ouroboros.llm_attempt import _canonical_candidate_bytes
    import hashlib
    assert rows[-1]["candidate_raw_sha256"] == hashlib.sha256(_canonical_candidate_bytes(candidate)).hexdigest()


def test_explicit_standard_does_not_inherit_global_fast(transport, monkeypatch):
    _root, client, sent = transport
    monkeypatch.setenv("OUROBOROS_PROCESSING_PREFERENCE", "fast")
    monkeypatch.setenv("OUROBOROS_MODEL_PROCESSING_PREFERENCES", '{"light":"standard"}')
    for role in ("main", "light"):
        client.chat([{"role": "user", "content": "same"}], "openai::same-model", model_role=role)
    client.chat([{"role": "user", "content": "same"}], "openai::same-model", processing_preference="")
    assert [item.get("service_tier") for item in sent] == ["priority", "default", None]


@pytest.mark.parametrize("provider,payload", [
    ("openai", {"service_tier": "default"}),
    ("openrouter", {"extra_body": {"service_tier": "flex"}}),
    ("anthropic", {"speed": "standard"}),
    ("openai-compatible", {"model": "custom"}),
])
def test_native_override_and_unknown_transport_shape_are_preserved(provider, payload):
    before = copy.deepcopy(payload)
    apply_processing_preference({"provider": provider, "processing_preference": "fast"}, payload)
    assert payload == before


def test_fast_headers_preserve_other_beta_intent():
    target = {"provider": "anthropic", "contract_headers": {"anthropic-beta": "another-beta"}}
    assert processing_contract_headers(target, {"speed": "fast"})["anthropic-beta"] == "another-beta,fast-mode-2026-02-01"
    assert target["contract_headers"]["anthropic-beta"] == "another-beta"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_quota_reentry_keeps_captured_preference_and_same_owner(transport, monkeypatch, asynchronous):
    from ouroboros.llm_claudexor import ClaudexorModelNotDispatched

    root, _client, _sent = transport
    seen = []
    monkeypatch.setenv("OUROBOROS_PROCESSING_PREFERENCE", "fast")

    def invoke(self, *, model, model_role, processing_preference=None, model_poll_control=None):
        seen.append(processing_preference)
        if len(seen) == 1:
            error = ClaudexorModelNotDispatched({"code": "subscription_window_exhausted"})
            error.physical_attempt_capture = SimpleNamespace(state="released", attempt_id="attempt-one")
            raise error
        return {}, {}

    async def invoke_async(self, *, model, model_role, processing_preference=None, model_poll_control=None):
        return invoke(self, model=model, model_role=model_role, processing_preference=processing_preference,
                      model_poll_control=model_poll_control)

    wrapped = model_wait.model_waitable(invoke_async if asynchronous else invoke)
    with model_wait.task_model_wait_scope(task={"id": "processing"}, drive_root=root,
            event_queue=None, worker_slot_held=False, owner_control=lambda: None) as owner:
        def wait(llm, error, values, *args):
            monkeypatch.setenv("OUROBOROS_PROCESSING_PREFERENCE", "standard")
            assert model_wait.current_model_wait() is owner
            return values
        monkeypatch.setattr(owner, "wait", wait)
        result = wrapped(object(), model="claudexor::codex/model", model_role="light")
        if asynchronous:
            result = asyncio.run(result)
    assert seen == ["fast", "fast"]
    assert result[1]["ledger_attempt_ids"] == ["attempt-one"]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("refusal", ["typed", "capacity", "unsupported"])
def test_typed_no_start_reprices_standard_without_changing_custom_tools(transport, monkeypatch, asynchronous, refusal):
    from ouroboros import send_clock
    from ouroboros.llm_attempt import ProcessingNotStarted
    from ouroboros.request_wire_recovery import current_wire_candidate

    root, client, _sent = transport
    reservations, candidates, catalogs = [], [], []
    monkeypatch.setattr(ua, "_reservation_cost", lambda request: (
        reservations.append((request.submitted_processing_mode, request.candidate_raw_sha256)) or
        (0.01 if request.submitted_processing_mode == "flex" else 0.04)))
    target = {**client._resolve_remote_target("openai::same-model"), "processing_preference": "economy"}
    payload = client._build_remote_kwargs(target, [{"role": "user", "content": "unchanged"}],
        "high", 123, "required", None, [{"type": "function", "function": {
            "name": "inspect", "description": "Read the requested value", "parameters": {
                "type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}}}])

    def create(**candidate):
        candidates.append(copy.deepcopy(candidate))
        catalogs.append(current_wire_candidate().custom_catalog_sha256)
        if len(candidates) == 1:
            if refusal == "typed":
                raise ProcessingNotStarted(RuntimeError("documented no-start receipt"), reason="capacity")
            error = RuntimeError("Structured provider refusal")
            error.status_code = 429 if refusal == "capacity" else 400
            error.body = {"error": {"code": "resource_unavailable" if refusal == "capacity" else "unsupported_parameter",
                                    "param": "service_tier"}}
            raise error
        return SimpleNamespace(model_dump=lambda: {"choices": [{"message": {
            "role": "assistant", "content": "ok"}}], "usage": {"cost": 0.04}})

    async def create_async(**candidate):
        return create(**candidate)

    from datetime import datetime, timedelta, timezone
    samples = iter(datetime(2026, 9, 26, tzinfo=timezone.utc) + timedelta(minutes=i) for i in range(10))
    monkeypatch.setattr(send_clock, "_now", lambda: next(samples))
    with send_clock.MainSendClock(send_clock.SendClockPolicy()).bound() as clock:
        if asynchronous:
            asyncio.run(client._create_chat_completion_with_retries_async(create_async, payload, target))
        else:
            client._create_chat_completion_with_retries(create, payload, target)
        assert len(candidates) == 2 and len(clock.notes) == 2 and clock.notes[0] != clock.notes[1]
        first, second = [send_clock.split_clock_note(p)[1] for p in candidates]
    assert {**first, "service_tier": "default"} == second
    assert candidates[1]["tools"][0]["type"] == "custom"
    assert catalogs[0] and catalogs[0] == catalogs[1]
    assert [mode for mode, _digest in reservations] == ["flex", "default"]
    assert reservations[0][1] != reservations[1][1]
    rows = ledger_rows(root)
    finals = list({row["attempt_id"]: row for row in rows}.values())
    assert [row["state"] for row in finals] == ["released", "settled"]
    assert [row["reservation_upper_bound_usd"] for row in finals] == [0.01, 0.04]
    assert not (root / "state/request_wire_compatibility.json").exists()
    assert payload["service_tier"] == "flex"


@pytest.mark.parametrize("error_kind", ["timeout", "bare_429", "stream"])
def test_unknown_or_unqualified_failure_never_falls_back(transport, error_kind):
    root, client, _sent = transport
    calls = []
    target = {**client._resolve_remote_target("openai::same-model"), "processing_preference": "economy"}
    payload = client._build_remote_kwargs(target, [{"role": "user", "content": "input"}],
        "high", 123, "auto", None, None)

    def create(**candidate):
        calls.append(candidate)
        error = TimeoutError("outcome unknown")
        if error_kind == "bare_429":
            error.status_code = 429
        if error_kind == "stream":
            error.stream_incomplete = True
        raise error

    with pytest.raises(TimeoutError):
        client._create_chat_completion_with_retries(create, payload, target)
    assert len(calls) == 1
    rows = ledger_rows(root)
    assert rows[-1]["state"] == "unresolved"


def test_anthropic_fast_rate_refusal_releases_then_sends_standard(transport, monkeypatch):
    root, client, sent = transport
    success = __import__("requests").post

    def post(url, **kwargs):
        if not sent:
            sent.append({"payload": copy.deepcopy(kwargs["json"]), "headers": kwargs["headers"]})
            body = {"error": {"type": "rate_limit_error", "message": "Fast rate limit"}}
            return SimpleNamespace(status_code=429, text=json.dumps(body), reason="Too Many Requests",
                                   url=url, json=lambda: body)
        return success(url, **kwargs)

    monkeypatch.setattr("requests.post", post)
    message, _usage = client.chat([{"role": "user", "content": "same input"}],
                                  "anthropic::test-model", processing_preference="fast")
    assert message["content"] == "ok"
    assert [entry["payload"]["speed"] for entry in sent] == ["fast", "standard"]
    assert "fast-mode-2026-02-01" in sent[0]["headers"].get("anthropic-beta", "")
    assert "fast-mode-2026-02-01" not in sent[1]["headers"].get("anthropic-beta", "")
    rows = ledger_rows(root)
    assert [row["state"] for row in {r["attempt_id"]: r for r in rows}.values()] == ["released", "settled"]


def _known_sibling_charge_before_the_retry(monkeypatch, root, task_id, first_mode):
    """Price the first mode at $0.01; before the standard retry is priced, a sibling's
    $0.02 lands at a final price: the tree's KNOWN spend reaches the $0.02 cap (#1487)."""
    landed = []

    def price(request):
        if request.submitted_processing_mode == first_mode:
            return 0.01
        if not landed:
            landed.append(True)
            with ua.usage_scope(ua.UsageScope(drive_root=root, task_id="sibling", root_task_id=task_id)):
                held = ua.reserve_attempt(ua.AttemptRequest(model="m", provider="p", reservation_usd=0.02))
                ua.mark_dispatched(held)
                ua.settle_attempt(held, {}, cost_usd=0.02, cost_final=True)
        return 0.04

    monkeypatch.setattr(ua, "_reservation_cost", price)


def test_standard_retry_is_refused_when_known_spend_reached_the_budget(transport, monkeypatch):
    """The retry gets its own admission: known spend that reached the cap after the
    first attempt was admitted refuses the retry's reservation; nothing is sent."""
    from ouroboros.llm_attempt import ProcessingNotStarted

    root, client, _sent = transport
    calls = []
    _known_sibling_charge_before_the_retry(monkeypatch, root, "processing", "flex")
    target = {**client._resolve_remote_target("openai::same-model"), "processing_preference": "economy"}
    payload = client._build_remote_kwargs(target, [{"role": "user", "content": "input"}],
        "high", 123, "auto", None, None)

    def create(**candidate):
        calls.append(candidate)
        raise ProcessingNotStarted(RuntimeError("typed no-start fixture"), reason="capacity")

    with ua.usage_scope(ua.UsageScope(drive_root=root, task_id="processing", root_task_id="processing", root_limit_usd=0.02)):
        with pytest.raises(ua.BudgetExceeded):
            client._create_chat_completion_with_retries(create, payload, target)
    assert len(calls) == 1
    finals = {row["attempt_id"]: row for row in ledger_rows(root)}.values()
    assert [(row["task_id"], row["state"]) for row in finals] == [("processing", "released"), ("sibling", "settled")]


@pytest.mark.parametrize("provider,expected", [("openrouter", "default"), ("anthropic", "standard")])
def test_direct_web_helper_carries_its_own_role_preference(transport, monkeypatch, provider, expected):
    from ouroboros import llm

    root, _client, _sent = transport
    sent, clients = [], []
    monkeypatch.setenv("OUROBOROS_PROCESSING_PREFERENCE", "fast")
    monkeypatch.setenv("OUROBOROS_MODEL_PROCESSING_PREFERENCES", '{"websearch":"standard"}')

    def create(**candidate):
        sent.append(candidate)
        return {"usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0}}

    def factory(**kwargs):
        clients.append(kwargs)
        def messages_create(*, model, messages, max_tokens, tools, extra_body=None):
            return create(model=model, messages=messages, max_tokens=max_tokens,
                          tools=tools, **(extra_body or {}))
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                               messages=SimpleNamespace(create=messages_create))

    if provider == "openrouter":
        monkeypatch.setattr("ouroboros.net_transport.web_search_openai_client", factory)
        llm.openrouter_web_search_server_tool(api_key="test", model="model", query="query", search_context_size="low")
    else:
        monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=factory))
        llm.anthropic_web_search_server_tool(api_key="test", model="model", query="query")
    assert len(sent) == 1
    assert sent[0]["speed" if provider == "anthropic" else "service_tier"] == expected
    rows = ledger_rows(root)
    assert rows[-1]["submitted_processing_mode"] == expected


@pytest.mark.parametrize("provider,preference,field,native", [
    ("openai", "economy", "service_tier", "flex"),
    ("anthropic", "fast", "speed", "fast"),
])
def test_explicit_native_option_is_not_downgraded_by_preference_fallback(transport, provider, preference, field, native):
    from ouroboros.llm_attempt import ProcessingNotStarted, _finalized_physical_candidate, _attempt_request
    from ouroboros.request_wire_contract import physical_candidate_sha256
    from ouroboros.request_wire_recovery import plan_next_wire_retry, request_wire_call_scope

    _root, client, _sent = transport
    target = {**client._resolve_remote_target(f"{provider}::same-model"), "processing_preference": preference}
    payload = {"model": "same-model", "messages": [{"role": "user", "content": "input"}], field: native}
    apply_processing_preference(target, payload)
    assert target["processing_native_origin"] == "native_override"
    with request_wire_call_scope():
        candidate = _finalized_physical_candidate(target, payload, "messages" if provider == "anthropic" else "chat.completions")
        request = _attempt_request(target, candidate)
        error = ProcessingNotStarted(RuntimeError("typed no-start fixture"), reason="capacity")
        error.physical_attempt_capture = SimpleNamespace(state="released", candidate_raw_sha256=physical_candidate_sha256(candidate))
        assert plan_next_wire_retry(candidate, error=error, target=target) is None
        assert request.submitted_processing_mode == native


def test_foreign_capture_never_supplies_submitted_mode_to_display(transport):
    _root, client, _sent = transport
    ua.adopt_physical_attempt_capture(ua.PhysicalAttemptCapture(
        attempt_id="foreign", model="openai/other-model", provider="openai", state="settled",
        candidate_measurement_kind="canonical_json_v1", processing_preference="economy", submitted_processing_mode="flex"))
    target = {**client._resolve_remote_target("openai::same-model"), "processing_preference": "fast"}
    _, usage = client._normalize_remote_response({"choices": [{"message": {"content": "ok"}}],
        "usage": {"cost": 0}, "service_tier": "default"}, target)
    assert usage["processing"]["requested"] == "fast"
    assert usage["processing"]["submittedNative"] is None
    assert usage["processing"]["observed"] == "standard"
