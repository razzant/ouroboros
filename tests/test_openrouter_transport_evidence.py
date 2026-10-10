"""Received price and generation identity belong to the exact physical send."""
from __future__ import annotations

import asyncio
import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest

from ouroboros import usage_accounting as ua
from ouroboros.llm import LLMClient
from ouroboros.llm_stream import AssembledResponse, consume_stream, consume_stream_async
from tests._usage_store_testing import ledger_rows


def target(provider="openrouter"):
    return {"provider": provider, "resolved_model": "vendor/fixture", "usage_model": "vendor/fixture",
            "base_url": "https://provider.invalid/v1", "api_key": "fixture-key",
            "supports_openrouter_extensions": provider == "openrouter", "supports_generation_cost": True}


def frame(*, generation="gen-fixture", usage=None, finish=None):
    body = {"id": generation, "choices": [{"index": 0, "delta": {"content": "answer"},
                                           "finish_reason": finish}]}
    if usage is not None:
        body["usage"] = usage
    return ("data: " + json.dumps(body) + "\n\n").encode()


class Wire:
    def __init__(self, chunks, *, read_error=None, close_error=None, header="", step=None):
        self.chunks, self.read_error, self.close_error = chunks, read_error, close_error
        self.headers = {"x-generation-id": header} if header else {}
        self.step = step
        self.close_calls = 0

    def iter_bytes(self):
        for index, chunk in enumerate(self.chunks):
            if self.step:
                self.step(index)
            yield chunk
        if self.read_error is not None:
            raise self.read_error

    def iter_content(self, **_kwargs):
        return self.iter_bytes()

    async def aiter_bytes(self):
        for chunk in self.iter_bytes():
            yield chunk

    def close(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error

    async def aclose(self):
        self.close()


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    import ouroboros.pricing as pricing
    import ouroboros.request_wire_contract as wire

    monkeypatch.setattr(wire, "canonical_wire_evidence_root", lambda: tmp_path)
    monkeypatch.setattr(pricing, "estimate_cost_optional", lambda *a, **k: None)
    monkeypatch.setattr(ua, "_reservation_cost", lambda request: 2.0)
    monkeypatch.setattr(LLMClient, "_get_supported_parameters", lambda *a, **k: None)
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="money-task", root_task_id="money-task")):
        yield tmp_path


def run_driver(wire, *, asynchronous=False, provider="openrouter"):
    client = LLMClient()
    kwargs = {"model": "vendor/fixture", "messages": [{"role": "user", "content": "hello"}],
              "max_tokens": 100, "stream": True}
    if asynchronous:
        async def create(**_kwargs):
            return wire
        return asyncio.run(client._create_chat_completion_with_retries_async(create, kwargs, target(provider)))
    return client._create_chat_completion_with_retries(lambda **kw: wire, kwargs, target(provider))


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("phase", ["read", "close", "cancel"])
@pytest.mark.parametrize("cost", [0.0, 1.25])
def test_received_price_survives_stream_failure(isolated, asynchronous, phase, cost):
    failure = asyncio.CancelledError("owner stop") if phase == "cancel" else RuntimeError("wire failed")
    chunks = [frame(usage={"cost": cost, "prompt_tokens": 11, "completion_tokens": 3}, finish="stop")]
    if phase == "close":
        chunks.append(b"data: [DONE]\n\n")
    wire = Wire(chunks, read_error=failure if phase != "close" else None,
                close_error=failure if phase == "close" else RuntimeError("cleanup failed too"))
    with pytest.raises(type(failure)) as caught:
        run_driver(wire, asynchronous=asynchronous)
    if not (asynchronous and phase == "cancel"):
        assert caught.value is failure
    # The async accounting custody wrapper re-emits cancellation; the original
    # physical failure retains its usage and no caller receives a partial reply.
    assert failure.stream_usage["cost"] == cost
    rows = ledger_rows(isolated)
    assert len(rows) == 1  # No repeated generation after partial response or Stop.
    assert rows[0]["cost_usd"] == cost and rows[0]["cost_final"] is True
    assert rows[0]["prompt_tokens"] == 11 and rows[0]["completion_tokens"] == 3
    assert rows[0]["provider_receipt_binding"]["generation_id"] == "gen-fixture"
    assert wire.close_calls >= 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("header", ["", "gen-fixture"])
def test_generation_is_bound_before_stream_finishes_once(isolated, monkeypatch, asynchronous, header):
    observations = []
    bind = ua.bind_provider_generation

    def observe(generation_id, **kwargs):
        observations.append(generation_id)
        return bind(generation_id, **kwargs)

    monkeypatch.setattr(ua, "bind_provider_generation", observe)

    def step(index):
        if index or header:
            row = ledger_rows(isolated)[0]
            assert row["state"] == "dispatched"
            binding = row["provider_receipt_binding"]
            assert binding["generation_id"] == "gen-fixture"
            assert binding["endpoint"] == "https://provider.invalid/v1"
            assert binding["credential_sha256"] == hashlib.sha256(b"fixture-key").hexdigest()
            assert "fixture-key" not in json.dumps(row)

    wire = Wire([frame(), frame(usage={"cost": 0.2}, finish="stop"), b"data: [DONE]\n\n"],
                header=header, step=step)
    assert run_driver(wire, asynchronous=asynchronous).model_dump()["choices"][0]["message"]["content"]
    # Header/chunk repetitions and post-response accounting do not open another
    # generation-binding transaction once the first observation was retained.
    assert observations == ["gen-fixture"]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("header", ["gen-header", ""])
def test_generation_conflict_remains_visible_without_losing_answer(isolated, asynchronous, header):
    chunks = [frame(generation="gen-first"), frame(generation="gen-second", usage={"cost": 0.3}, finish="stop"),
              b"data: [DONE]\n\n"]
    result = run_driver(Wire(chunks, header=header), asynchronous=asynchronous).model_dump()
    assert result["choices"][0]["message"]["content"] == "answeranswer"
    assert result["_stream_receipt"]["generation_conflict"] is True
    row = ledger_rows(isolated)[0]
    assert row["provider_receipt_binding"]["conflict"]
    assert row["cost_usd"] == 0.3


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("header", ["gen-first", ""])
def test_conflicting_error_frame_preserves_binding_conflict(isolated, asynchronous, header):
    from ouroboros.llm_stream import ProviderStreamError
    from ouroboros.openrouter_cost import generation_binding, fetch_generation_receipt

    error_frame = ("data: " + json.dumps({"id": "gen-second", "error": {
        "code": 400, "message": "synthetic conflicting generation"}, "usage": {"cost": 0.3}}) + "\n\n").encode()
    wire = Wire([frame(generation="gen-first"), error_frame], header=header)
    with pytest.raises(ProviderStreamError) as caught:
        run_driver(wire, asynchronous=asynchronous)
    rows = ledger_rows(isolated)
    assert len(rows) == 1
    assert rows[0]["provider_receipt_binding"]["conflict"] == {"observed_generation_id": "gen-second"}
    assert generation_binding(rows[0]) is None
    assert fetch_generation_receipt(isolated, rows[0], target())["status"] == "binding_unavailable"
    assert caught.value.stream_receipt["generation_conflict"] is True
    assert rows[0]["physical_failure"]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_public_chat_without_price_never_fetches_generation(isolated, monkeypatch, asynchronous):
    import requests

    client = LLMClient()
    calls = []

    def unexpected_get(*args, **kwargs):
        pytest.fail("ordinary chat must not make an out-of-band generation GET")

    def create(**kwargs):
        calls.append(kwargs)
        return AssembledResponse({"id": "gen-json", "choices": [{"message": {"role": "assistant", "content": "usable"},
                                                                 "finish_reason": "stop"}],
                                  "usage": {"prompt_tokens": 10, "completion_tokens": 2}})

    async def async_create(**kwargs):
        return create(**kwargs)

    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=async_create if asynchronous else create)))
    monkeypatch.setattr(requests, "get", unexpected_get)
    monkeypatch.setattr(client, "_resolve_remote_target", lambda model: target())
    monkeypatch.setattr(client, "_get_remote_client", lambda route: sdk)
    monkeypatch.setattr(client, "_get_async_remote_client", lambda route: sdk)
    kwargs = {"messages": [{"role": "user", "content": "hello"}], "model": "vendor/fixture", "max_tokens": 100}
    message, usage = asyncio.run(client.chat_async(**kwargs)) if asynchronous else client.chat(**kwargs)
    assert message["content"] == "usable" and len(calls) == 1
    assert usage["cost"] is None and usage["cost_final"] is False
    row = ledger_rows(isolated)[0]
    assert row["state"] == "settled" and row["cost_final"] is False
    assert row["provider_receipt_binding"]["generation_id"] == "gen-json"


@pytest.mark.parametrize("final_counters", [False, True])
def test_native_partial_counters_do_not_become_money(isolated, final_counters):
    start = {"type": "message_start", "message": {"id": "msg-native", "type": "message", "content": [],
                                                 "usage": {"input_tokens": 10, "output_tokens": 1}}}
    events = [start]
    if final_counters:
        events.append({"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                       "usage": {"output_tokens": 9, "cost": 0.4}})
    chunks = [("data: " + json.dumps(event) + "\n\n").encode() for event in events]
    failure = RuntimeError("read failed")
    request = ua.AttemptRequest(model="anthropic::fixture", provider="anthropic", reservation_usd=2.0,
                                drive_root=isolated)
    with pytest.raises(RuntimeError) as caught:
        ua.execute_physical_attempt(request, lambda: consume_stream(Wire(chunks, read_error=failure), native=True))
    assert caught.value is failure
    row = ledger_rows(isolated)[0]
    assert "provider_receipt_binding" not in row
    if final_counters:
        assert row["cost_usd"] == 0.4 and failure.stream_usage["output_tokens"] == 9
    else:
        assert row["state"] == "unresolved" and row.get("cost_usd") is None
        assert getattr(failure, "stream_usage", None) is None


def test_clean_eof_after_finish_still_returns_received_answer(isolated):
    result = run_driver(Wire([frame(usage={"cost": 0.25}, finish="stop")])).model_dump()
    assert result["choices"][0]["message"]["content"] == "answer"
    assert ledger_rows(isolated)[0]["cost_usd"] == 0.25


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("usage", [{"prompt_tokens": 2}, {"cost": 0.125, "prompt_tokens": 2}])
def test_other_provider_preserves_price_but_not_partial_counter_settlement(isolated, asynchronous, usage):
    error = RuntimeError("socket read failed")
    with pytest.raises(RuntimeError) as caught:
        run_driver(Wire([frame(usage=usage)], read_error=error), asynchronous=asynchronous, provider="example-provider")
    assert caught.value is error
    row = ledger_rows(isolated)[0]
    assert "provider_receipt_binding" not in row
    if "cost" in usage:
        assert row["cost_usd"] == 0.125 and row["cost_final"] is True
    else:
        assert row["state"] == "unresolved" and row.get("cost_usd") is None


@pytest.mark.parametrize("cost,expected", [(True, None), (-1, None), (float("nan"), None), ("0", 0.0), ("1.25", 1.25)])
def test_public_usage_projects_only_valid_received_money(isolated, monkeypatch, cost, expected):
    monkeypatch.setattr("ouroboros.pricing.estimate_cost_optional",
                        lambda *a, **kw: pytest.fail("received price must not use tariffs"))
    client = LLMClient()
    response = {"id": "gen-json", "choices": [{"message": {"role": "assistant", "content": "usable"}}],
                "usage": {"cost": cost, "prompt_tokens": 10, "completion_tokens": 2}}
    message, usage = client._normalize_remote_response(response, target())
    assert message["content"] == "usable"
    assert usage["cost"] == expected and usage["cost_final"] is (expected is not None)
    assert bool(usage.get("cost_invalid")) is (expected is None)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("cost", [True, -1, float("nan"), float("inf"), "invalid"])
@pytest.mark.parametrize("field", ["cost", "total_cost", "total_cost_usd"])
def test_public_chat_invalid_price_stays_unknown_with_available_tariff(
        isolated, monkeypatch, asynchronous, cost, field):
    from ouroboros import loop_llm_call as main
    import requests

    estimates, sends = [], []

    def tariff(*args, **kwargs):
        estimates.append((args, kwargs))
        return 0.75

    body = {"id": "gen-invalid", "choices": [{"message": {"role": "assistant", "content": "usable"},
                                               "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
    (body if field == "total_cost_usd" else body["usage"])[field] = cost

    def create(**kwargs):
        sends.append(kwargs)
        return AssembledResponse(body)

    async def async_create(**kwargs):
        return create(**kwargs)

    client = LLMClient()
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=async_create if asynchronous else create)))
    for owner in ("ouroboros.pricing", "ouroboros.usage_accounting", "ouroboros.loop_llm_call"):
        monkeypatch.setattr(f"{owner}.estimate_cost_optional", tariff)
    monkeypatch.setattr(requests, "get", lambda *a, **kw: pytest.fail("chat made a metadata GET"))
    monkeypatch.setattr(client, "_resolve_remote_target", lambda model: target())
    monkeypatch.setattr(client, "_get_remote_client", lambda route: sdk)
    monkeypatch.setattr(client, "_get_async_remote_client", lambda route: sdk)
    kwargs = {"messages": [{"role": "user", "content": "hello"}], "model": "vendor/fixture", "max_tokens": 100}
    message, usage = asyncio.run(client.chat_async(**kwargs)) if asynchronous else client.chat(**kwargs)
    assert message["content"] == "usable" and len(sends) == 1
    assert usage["cost"] is None and usage["cost_final"] is False and usage["cost_invalid"] is True
    assert main._normalize_usage_cost(usage, model="vendor/fixture", use_local=False)[0] is None
    row, = ledger_rows(isolated)
    assert row["state"] == "settled" and row["cost_usd"] is None and row["cost_final"] is False
    assert row["prompt_tokens"] == 10 and row["completion_tokens"] == 2
    assert estimates == []


@pytest.mark.parametrize("price", [0.0, 1.25])
@pytest.mark.parametrize("field", ["total_cost", "total_cost_usd"])
def test_valid_alternate_price_outranks_invalid_cost(isolated, monkeypatch, price, field):
    body = {"choices": [{"message": {"content": "usable"}}],
            "usage": {"cost": True, "prompt_tokens": 10, "cost_invalid": True}}
    (body if field == "total_cost_usd" else body["usage"])[field] = price
    monkeypatch.setattr("ouroboros.pricing.estimate_cost_optional",
                        lambda *a, **kw: pytest.fail("valid alternate price must not use tariffs"))
    _, usage = LLMClient()._normalize_remote_response(body, target())
    normalized, cost, final = ua.usage_from_response(body)
    assert usage["cost"] == cost == price and usage["cost_final"] is final is True
    assert "cost_invalid" not in usage and "cost_invalid" not in normalized


def test_absent_price_still_permits_tariff_estimate(isolated, monkeypatch):
    body = {"choices": [{"message": {"content": "usable"}}],
            "usage": {"prompt_tokens": 10, "cost": None, "cost_invalid": True}}
    monkeypatch.setattr("ouroboros.pricing.estimate_cost_optional", lambda *a, **kw: 0.75)
    _, usage = LLMClient()._normalize_remote_response(body, target())
    normalized, cost, final = ua.usage_from_response(body)
    assert usage["cost"] == 0.75 and usage["cost_estimated"] is True and usage["cost_final"] is False
    assert cost is None and final is False
    assert "cost_invalid" not in usage and "cost_invalid" not in normalized


def test_generation_retention_failure_does_not_destroy_response(isolated, monkeypatch):
    def fail(_generation, **_kwargs):
        raise OSError("synthetic binding write failure")

    monkeypatch.setattr(ua, "bind_provider_generation", fail)
    result = run_driver(Wire([frame(usage={"cost": 0.25}, finish="stop"), b"data: [DONE]\n\n"])).model_dump()
    assert result["choices"][0]["message"]["content"] == "answer"
    assert result["_stream_receipt"]["generation_binding_error"] == "OSError"
    assert ledger_rows(isolated)[0]["cost_usd"] == 0.25


@pytest.mark.parametrize("asynchronous", [False, True])
def test_stream_evidence_uses_explicit_attempt_root_before_ambient_scope(isolated, asynchronous):
    from ouroboros.observability import read_call_payload
    from ouroboros.openrouter_cost import binding_for_target

    selected_root = isolated / "selected-attempt"
    request = ua.AttemptRequest(model="vendor/fixture", provider="openrouter", reservation_usd=2.0,
                                drive_root=selected_root, provider_receipt_binding=binding_for_target(target()))
    wire = Wire([frame(usage={"cost": 0.125}, finish="stop"), b"data: [DONE]\n\n"])
    if asynchronous:
        response = asyncio.run(ua.execute_physical_attempt_async(request, lambda: consume_stream_async(wire)))
    else:
        response = ua.execute_physical_attempt(request, lambda: consume_stream(wire))
    row = ledger_rows(selected_root)[0]
    assert row["provider_receipt_binding"]["generation_id"] == "gen-fixture"
    receipt = response.model_dump()["_stream_receipt"]
    assert receipt["attempt_id"] == row["attempt_id"] and "retention_error" not in receipt
    _, payload, _ = read_call_payload(selected_root, task_id="money-task", call_id=f"physical_{row['attempt_id']}_stream")
    assert payload["attempt_id"] == row["attempt_id"]
    assert ledger_rows(isolated) == []


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("cost", [0.0, 1.25])
@pytest.mark.parametrize("phase", ["read", "close"])
@pytest.mark.parametrize("error_type", [httpx.ReadError, httpx.ReadTimeout])
def test_main_does_not_resend_incomplete_response_with_settled_price(
        isolated, monkeypatch, asynchronous, cost, phase, error_type):
    from ouroboros import loop_llm_call as main

    calls = []
    failure = error_type("socket response incomplete")
    chunks = [frame(usage={"cost": cost}, finish="stop" if phase == "close" else None)]
    if phase == "close":
        chunks.append(b"data: [DONE]\n\n")
    wire = Wire(chunks, read_error=failure if phase == "read" else None,
                close_error=failure if phase == "close" else None)

    class Consumer:
        def chat(self, **kwargs):
            calls.append(kwargs)
            return run_driver(wire, asynchronous=asynchronous)

    monkeypatch.setattr(main, "_prepare_main_messages", lambda messages, **kwargs: messages)
    monkeypatch.setattr(main, "_sleep_within_deadline", lambda *args, **kwargs: True)
    logs = isolated / "logs"
    logs.mkdir(exist_ok=True)
    usage = {}
    message, _ = main.call_llm_with_retry(
        Consumer(), [{"role": "user", "content": "hello"}], "vendor/fixture", [], "low", 2,
        logs, "money-task", 1, None, usage, "task",
    )
    assert message is None and len(calls) == 1
    assert usage["_last_llm_error_kind"] == "provider_outcome_unknown"
    assert usage["_last_llm_retry_same_request"] is False
    rows = ledger_rows(isolated)
    assert len(rows) == 1 and rows[0]["cost_usd"] == cost and rows[0]["cost_final"] is True
    assert rows[0]["physical_failure"]
    assert failure.physical_attempt_capture.state == "settled"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_failed_first_stream_binding_recovers_from_failure_receipt(isolated, monkeypatch, asynchronous):
    original, observed = ua.bind_provider_generation, []

    def fail_first(generation, **kwargs):
        observed.append(generation)
        if len(observed) == 1:
            raise OSError("first binding unavailable")
        return original(generation, **kwargs)

    monkeypatch.setattr(ua, "bind_provider_generation", fail_first)
    failure = RuntimeError("socket read failed")
    with pytest.raises(RuntimeError) as caught:
        run_driver(Wire([frame()], read_error=failure), asynchronous=asynchronous)
    assert caught.value is failure
    row = ledger_rows(isolated)[0]
    assert row["provider_receipt_binding"]["generation_id"] == "gen-fixture"
    assert observed == ["gen-fixture", "gen-fixture"]
    assert failure.stream_receipt["generation_binding_error"] == "OSError"
    assert row["state"] == "unresolved" and row["physical_failure"]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_raised_json_error_binds_explicit_reservation_without_ambient_capture(isolated, asynchronous):
    from ouroboros.openrouter_cost import binding_for_target

    class BodyError(RuntimeError):
        body = {"id": "gen-raised-json", "error": {"code": 500}}

    selected = isolated / "explicit-root"
    request = ua.AttemptRequest(model="vendor/fixture", provider="openrouter", reservation_usd=2,
                                drive_root=selected, provider_receipt_binding=binding_for_target(target()))
    def send():
        raise BodyError("provider rejected")
    async def send_async():
        send()
    with pytest.raises(BodyError):
        if asynchronous:
            asyncio.run(ua.execute_physical_attempt_async(request, send_async))
        else:
            ua.execute_physical_attempt(request, send)
    row = ledger_rows(selected)[0]
    assert row["provider_receipt_binding"]["generation_id"] == "gen-raised-json"
    assert row["physical_failure"] and ledger_rows(isolated) == []


@pytest.mark.parametrize("cancel", [False, True])
def test_async_first_binding_is_off_loop_joined_and_not_repeated(isolated, monkeypatch, cancel):
    import threading

    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    original, observed = ua.bind_provider_generation, []

    def held_binding(generation, **kwargs):
        observed.append(generation)
        entered.set()
        assert release.wait(2), "binding blocked the event loop"
        original(generation, **kwargs)
        finished.set()

    monkeypatch.setattr(ua, "bind_provider_generation", held_binding)

    async def exercise():
        from ouroboros.openrouter_cost import binding_for_target

        request = ua.AttemptRequest(model="vendor/fixture", provider="openrouter", reservation_usd=2,
                                    drive_root=isolated, provider_receipt_binding=binding_for_target(target()))
        wire = Wire([frame(usage={"cost": 0.25}, finish="stop"), b"data: [DONE]\n\n"], header="gen-fixture")
        task = asyncio.create_task(ua.execute_physical_attempt_async(request, lambda: consume_stream_async(wire)))
        for _ in range(200):
            if entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert entered.is_set()
        if cancel:
            task.cancel()
            await asyncio.sleep(0.02)
            assert not task.done()  # The in-flight binding remains owned.
        release.set()
        if cancel:
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            assert (await task).model_dump()["choices"][0]["message"]["content"] == "answer"
        assert finished.is_set()

    asyncio.run(exercise())
    assert observed == ["gen-fixture"]
    assert ledger_rows(isolated)[0]["provider_receipt_binding"]["generation_id"] == "gen-fixture"
