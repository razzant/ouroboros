"""Light consumers reach real client preparation/accounting, with in-memory senders."""

import asyncio
import contextlib
import threading
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from ouroboros import deadline_utils, model_concurrency, model_wait, project_naming, ui_translation
from ouroboros import usage_accounting as ua
from ouroboros.llm import LLMClient
from ouroboros.llm_stream import AssembledResponse
from tests._usage_store_testing import ledger_rows


@pytest.fixture
def call_env(tmp_path, monkeypatch):
    from ouroboros import observability, pricing, request_wire_contract

    now, sends = [100.0], []
    clock = SimpleNamespace(time=lambda: now[0], monotonic=lambda: now[0],
                            sleep=lambda seconds: now.__setitem__(0, now[0] + seconds))
    # The old naming module had no clock import; keep the negative control runnable.
    monkeypatch.setattr(project_naming, "time", clock, raising=False)
    monkeypatch.setattr(ui_translation, "time", clock)
    monkeypatch.setattr(model_concurrency, "time", clock)
    monkeypatch.setattr(deadline_utils, "utc_now", lambda: datetime.fromtimestamp(now[0], timezone.utc))
    monkeypatch.setattr(model_wait, "monotonic_now", lambda slot=None: now[0])
    monkeypatch.setattr(project_naming, "_naming_timeout_sec", lambda: 10.0)
    monkeypatch.setattr(ui_translation, "_timeout_sec", lambda: 10.0)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    monkeypatch.setenv("OUROBOROS_PER_TASK_COST_USD", "100")
    monkeypatch.setenv("OUROBOROS_MODEL_MAX_CONCURRENCY", "1")
    monkeypatch.setattr(request_wire_contract, "canonical_wire_evidence_root", lambda: tmp_path)
    monkeypatch.setattr(pricing, "estimate_cost_optional", lambda *a, **k: None)
    monkeypatch.setattr(ua, "_reservation_cost", lambda request: 1.0)
    monkeypatch.setattr(LLMClient, "_get_supported_parameters", lambda *a, **k: None)
    route = {"provider": "openai-compatible", "resolved_model": "unseen-light", "usage_model": "unseen-light",
             "base_url": "https://light.invalid/v1", "api_key": "fixture",
             "supports_openrouter_extensions": False, "supports_generation_cost": False}

    def send(**kwargs):
        sends.append(kwargs)
        return AssembledResponse({"id": "fixture-generation", "choices": [{
            "message": {"role": "assistant", "content": "Useful title"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.25}})

    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=send)))
    client = LLMClient()
    monkeypatch.setattr(client, "_resolve_remote_target", lambda model: route)
    monkeypatch.setattr(client, "_get_remote_client", lambda target: sdk)
    monkeypatch.setattr(project_naming, "_light_naming_model", lambda: "openai-compatible::unseen-light")
    monkeypatch.setattr(ui_translation, "_light_route", lambda: ("openai-compatible::unseen-light", False, True))
    return SimpleNamespace(root=tmp_path, now=now, sends=sends, client=client, sdk=sdk, route=route,
                           observability=observability)


def invoke(env, consumer):
    if consumer == "naming":
        return project_naming.llm_project_name("Owner request", llm_client=env.client, use_local=False,
                                               drive_root=env.root, task_id="title-task")
    return ui_translation._call_light(env.root, [{"role": "user", "content": "Translate"}], client=env.client)[0]


def refused(env, consumer):
    if consumer == "naming":
        assert invoke(env, consumer) == "Owner request"
    else:
        with pytest.raises(Exception):
            invoke(env, consumer)


def spend_preparation(env, monkeypatch, until):
    original = env.observability.persist_physical_candidate

    def persist(*args, **kwargs):
        result = original(*args, **kwargs)
        env.now[0] = until
        return result

    monkeypatch.setattr(env.observability, "persist_physical_candidate", persist)


@pytest.mark.parametrize("consumer", ["naming", "translation"])
@pytest.mark.parametrize("limiter", [0, 1])
def test_preparation_expiry_prevents_a_physical_send(call_env, monkeypatch, consumer, limiter):
    monkeypatch.setenv("OUROBOROS_MODEL_MAX_CONCURRENCY", str(limiter))
    spend_preparation(call_env, monkeypatch, 111)
    refused(call_env, consumer)
    assert not call_env.sends
    assert [row["state"] for row in ledger_rows(call_env.root)] == ["released"]


@pytest.mark.parametrize("consumer", ["naming", "translation"])
def test_slot_wait_spends_the_same_deadline(call_env, monkeypatch, consumer):
    # Hold the actual per-route slot; its queue and expiry check remain production code.
    monkeypatch.setattr(model_concurrency, "_max_slot_wait_sec", lambda: 12)
    with model_concurrency.model_call_slot("unseen-light", route=call_env.route):
        refused(call_env, consumer)
    assert call_env.now[0] == pytest.approx(110)
    assert not call_env.sends
    assert ledger_rows(call_env.root)[-1]["state"] == "released"


@pytest.mark.parametrize("consumer", ["naming", "translation"])
@pytest.mark.parametrize("spent", [3, 11])
def test_recovery_has_remaining_allowance_and_never_renews_it(call_env, consumer, spent):
    original = call_env.sdk.chat.completions.create

    def reject_once(**kwargs):
        if not call_env.sends:
            call_env.sends.append(kwargs)
            call_env.now[0] += spent
            error = RuntimeError("reasoning_effort unsupported")
            error.status_code = 400
            error.body = {"error": {"type": "invalid_request_error", "message": str(error)}}
            raise error
        return original(**kwargs)

    call_env.sdk.chat.completions.create = reject_once
    if spent > 10:
        refused(call_env, consumer)
        assert len(call_env.sends) == 1
        assert [row["state"] for row in ledger_rows(call_env.root)] == ["unresolved"]
    else:
        assert invoke(call_env, consumer) == "Useful title"
        assert [send["timeout"] for send in call_env.sends] == [10, 7]
        assert [row["state"] for row in ledger_rows(call_env.root)] == ["unresolved", "settled"]
        assert ledger_rows(call_env.root)[-1]["cost_usd"] == 0.25


@pytest.mark.parametrize("consumer", ["naming", "translation"])
@pytest.mark.parametrize("inherited", ["none", "calendar", "execution"])
def test_earlier_inherited_bound_wins_and_unbounded_helpers_still_work(call_env, consumer, inherited):
    scopes = {"none": contextlib.nullcontext, "calendar": lambda: model_wait.calendar_scope(
        datetime.fromtimestamp(104, timezone.utc).isoformat()),
        "execution": lambda: model_wait.execution_deadline_scope(104)}
    with scopes[inherited]():
        assert invoke(call_env, consumer) == "Useful title"
    assert call_env.sends[0]["timeout"] == (10 if inherited == "none" else 4)
    row = ledger_rows(call_env.root)[-1]
    assert row["state"] == "settled" and row["cost_usd"] == 0.25
    assert row["category"] == ("project_naming" if consumer == "naming" else "ui_translation")


@pytest.mark.parametrize("delay,inline_bound", [(9, 8), (11, 20)])
def test_delayed_inline_worker_cannot_mint_a_new_window(call_env, monkeypatch, delay, inline_bound):
    real_to_thread = asyncio.to_thread

    async def delayed(function, *args, **kwargs):
        call_env.now[0] += delay
        return await real_to_thread(function, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", delayed)
    title = asyncio.run(project_naming.llm_project_name_async(
        "Owner request", timeout_sec=inline_bound, llm_client=call_env.client,
        use_local=False, drive_root=call_env.root, task_id="inline-title"))
    assert title == "Owner request"
    assert not call_env.sends


def test_delayed_background_namer_keeps_the_scheduling_deadline(call_env, monkeypatch):
    pending = []
    original = project_naming.llm_project_name

    def name(*args, **kwargs):
        return original(*args, llm_client=call_env.client, **kwargs)

    class Thread:
        def __init__(self, *, target, args, name, daemon):
            self.target, self.args, self.name = target, args, name

        def start(self):
            if self.name.startswith("namer-call-"):
                self.target(*self.args)
            else:
                pending.append(self)

        def join(self):
            pass

    monkeypatch.setattr(project_naming, "llm_project_name", name)
    monkeypatch.setattr(project_naming, "threading", SimpleNamespace(
        Thread=Thread, Event=threading.Event, Lock=threading.Lock))
    monkeypatch.setattr(project_naming, "_refresh_root_cost_after_naming", lambda *args: None)
    project_naming.spawn_turn_namer(call_env.root, "background-title", "Owner request")
    call_env.now[0] = 111
    pending[0].target(*pending[0].args)
    assert not call_env.sends


def test_expired_translation_batch_requeues_without_spending_an_answer_attempt(call_env, monkeypatch):
    from ouroboros import i18n_memory as memory

    memory.update_memory(call_env.root, "ru", lambda doc: None, create=True)
    memory.record_missing(call_env.root, "ru", [{"key": "Settings", "context": {}}])
    spend_preparation(call_env, monkeypatch, 111)
    facts = ui_translation.translate_batch(call_env.root, "ru", client=call_env.client)
    assert facts["error"] and facts["requeued"] == 1 and facts["dropped"] == 0
    pending = dict(memory.take_pending(call_env.root, "ru", 10))
    assert set(pending) == {"Settings"} and not pending["Settings"].get("attempts")
    assert memory.load_memory(call_env.root, "ru")["refused"] == {}
    assert not call_env.sends


@pytest.mark.parametrize("consumer", ["naming", "translation"])
@pytest.mark.parametrize("expired", [False, True])
def test_local_consumer_checks_preparation_deadline(call_env, monkeypatch, consumer, expired):
    from ouroboros import local_model

    original = call_env.sdk.chat.completions.create

    def local_send(**kwargs):
        response = original(**kwargs).model_dump()
        response["usage"]["cost"] = 0
        return AssembledResponse(response)

    call_env.sdk.chat.completions.create = local_send
    monkeypatch.setattr(local_model, "get_manager", lambda: SimpleNamespace(serving_context_evidence=lambda: {}))
    monkeypatch.setattr(call_env.client, "_get_local_client", lambda: call_env.sdk)
    monkeypatch.setattr(project_naming, "_light_use_local", lambda explicit: True)
    monkeypatch.setattr(ui_translation, "_light_route", lambda: ("unseen-local", True, True))
    if expired:
        spend_preparation(call_env, monkeypatch, 111)
        refused(call_env, consumer)
        assert not call_env.sends
    else:
        assert invoke(call_env, consumer) == "Useful title"
        assert len(call_env.sends) == 1
        assert ledger_rows(call_env.root)[-1]["cost_usd"] == 0


@pytest.mark.parametrize("expired", [False, True])
def test_managed_naming_checks_preparation_before_model_create(call_env, monkeypatch, expired):
    from ouroboros import llm_claudexor as transport
    from tests.test_llm_claudexor import Gateway, MODEL, result

    gateway = Gateway(results=[result(cash=0, knowledge="exact")])
    monkeypatch.setattr(transport, "ensure_owned_gateway", lambda **kwargs: gateway)
    monkeypatch.setattr(call_env.client, "_resolve_remote_target", LLMClient()._resolve_remote_target)
    monkeypatch.setattr(project_naming, "_light_naming_model", lambda: MODEL)
    monkeypatch.setattr(ui_translation, "_light_route", lambda: (MODEL, False, True))
    if expired:
        spend_preparation(call_env, monkeypatch, 111)
        refused(call_env, "naming")
        assert not gateway.creates
    else:
        assert invoke(call_env, "naming") == "Ответ 🐍"
        assert len(gateway.creates) == 1 and len(gateway.acks) == 1
        assert ledger_rows(call_env.root)[-1]["state"] == "settled"


def test_managed_translation_keeps_existing_unsupported_format_refusal(call_env, monkeypatch):
    from ouroboros import llm_claudexor as transport
    from tests.test_llm_claudexor import MODEL

    monkeypatch.setattr(call_env.client, "_resolve_remote_target", LLMClient()._resolve_remote_target)
    monkeypatch.setattr(ui_translation, "_light_route", lambda: (MODEL, False, True))
    monkeypatch.setattr(transport, "ensure_owned_gateway", lambda **kwargs: pytest.fail("must not create"))
    with pytest.raises(transport.ClaudexorModelError, match="response_format") as caught:
        invoke(call_env, "translation")
    assert caught.value.code == "unsupported_parameter"
    assert not ledger_rows(call_env.root)


def test_relative_socket_timeout_alone_still_creates_no_operation_deadline(call_env, monkeypatch):
    spend_preparation(call_env, monkeypatch, 111)
    with ua.usage_scope(ua.UsageScope(drive_root=call_env.root, task_id="bare", root_task_id="bare")):
        message, _ = call_env.client.chat([{"role": "user", "content": "hello"}],
                                        "openai-compatible::unseen-light", timeout=5)
    assert message["content"] == "Useful title" and len(call_env.sends) == 1
    assert call_env.sends[0]["timeout"] == 5
