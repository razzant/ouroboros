"""Light consumers reach real client preparation/accounting, with in-memory senders."""

import asyncio
import contextlib
import threading
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

# loop_llm_call and review_verdict_extraction load before any fixture replaces model_wait.monotonic_now, which
# modules they import bind at import.
from ouroboros import (
    deadline_utils,
    loop_llm_call,
    model_concurrency,
    model_wait,
    project_naming,
    review_verdict_extraction,
    safety,
    ui_translation,
    vision_routing,
)
from ouroboros import usage_accounting as ua
from ouroboros.llm import LLMClient
from ouroboros.llm_stream import AssembledResponse
from tests._safety_fixtures import supervisor as supervisor_fixture
from tests._usage_store_testing import ledger_rows

supervisor = supervisor_fixture
REAL_SAFETY_CALL = safety._safety_model_call  # the shared supervisor fixture fakes it; these checks need the real one


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


@pytest.mark.parametrize(("limiter", "consumer"), [(0, "naming"), (1, "translation")])
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


@pytest.mark.parametrize(("spent", "consumer"), [(3, "naming"), (11, "translation")])
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


@pytest.mark.parametrize(("inherited", "consumer"), [
    ("none", "naming"),
    ("calendar", "translation"),
    ("execution", "naming"),
])
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


@pytest.mark.parametrize(("expired", "consumer"), [(False, "naming"), (True, "translation")])
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


def light_lane(call_env, monkeypatch, lane):
    """Route both light consumers to ``lane``: its slot identity and the send timeouts it records."""
    from ouroboros import local_model
    from tests.test_local_gigachat_send_deadlines import gigachat_network

    if lane == "local":
        timeouts, original = [], call_env.sdk.chat.completions.create

        def local_send(**kwargs):
            timeouts.append(kwargs.get("timeout"))
            response = original(**kwargs).model_dump()
            response["usage"]["cost"] = 0
            return AssembledResponse(response)

        call_env.sdk.chat.completions.create = local_send
        monkeypatch.setattr(local_model, "get_manager", lambda: SimpleNamespace(serving_context_evidence=lambda: {}))
        monkeypatch.setattr(call_env.client, "_get_local_client", lambda: call_env.sdk)
        monkeypatch.setattr(project_naming, "_light_use_local", lambda explicit: True)
        monkeypatch.setattr(ui_translation, "_light_route", lambda: ("unseen-local", True, True))
        return ("local-model", {"provider": "local"}), lambda: timeouts
    seen = gigachat_network(monkeypatch, call_env, content="Useful title")
    route = {"provider": "gigachat", "resolved_model": "GigaChat-2", "usage_model": "gigachat::GigaChat-2",
             "access_token": "token", "base_url": "https://giga.test/api/v1"}
    monkeypatch.setattr(call_env.client, "_resolve_remote_target", lambda model: dict(route))
    monkeypatch.setattr(project_naming, "_light_naming_model", lambda: route["usage_model"])
    monkeypatch.setattr(ui_translation, "_light_route", lambda: (route["usage_model"], False, True))
    return (route["usage_model"], route), lambda: [item.timeout["read"] for item in seen]


@pytest.mark.parametrize("consumer", ["naming", "translation"])
@pytest.mark.parametrize("lane", ["local", "gigachat"])
@pytest.mark.parametrize("expired", [False, True])
def test_local_and_gigachat_sends_get_the_window_left_after_preparation(call_env, monkeypatch, consumer, lane,
                                                                        expired):
    """Both lanes used to fix the 10 s socket before preparation and the slot wait (#572)."""
    _slot, timeouts = light_lane(call_env, monkeypatch, lane)
    spend_preparation(call_env, monkeypatch, 111 if expired else 103)
    if expired:
        refused(call_env, consumer)
        assert ledger_rows(call_env.root)[-1]["state"] == "released"
    else:
        assert invoke(call_env, consumer) == "Useful title"
        assert ledger_rows(call_env.root)[-1]["state"] == "settled"
    assert timeouts() == ([] if expired else [7])


@pytest.mark.parametrize("lane", ["local", "gigachat"])
def test_local_and_gigachat_sends_get_the_window_left_after_a_slot_wait(call_env, monkeypatch, lane):
    (model, route), timeouts = light_lane(call_env, monkeypatch, lane)
    with slot_held_until(call_env, monkeypatch, model, route, 103) as released:
        assert invoke(call_env, "naming") == "Useful title"
    assert timeouts() == [pytest.approx(110 - released[0])] and released[0] >= 103  # not the stale 10 s


@contextlib.contextmanager
def slot_held_until(call_env, monkeypatch, model, route, release_at):
    """Another sender holds the lane's real slot until ``release_at`` on the controlled clock."""
    held, released = contextlib.ExitStack(), []
    held.enter_context(model_concurrency.model_call_slot(model, route=route))

    def sleep(seconds):  # the queue's real poll
        call_env.now[0] += seconds
        if call_env.now[0] >= release_at and not released:
            released.append(call_env.now[0])
            held.close()

    monkeypatch.setattr(model_concurrency, "time", SimpleNamespace(
        time=lambda: call_env.now[0], monotonic=lambda: call_env.now[0], sleep=sleep))
    try:
        yield released
    finally:
        held.close()


def spend_until(call_env, monkeypatch, elapsed, until):
    """The window from 100 to ``until`` goes to preparation or to another sender's slot."""
    if elapsed == "preparation":
        spend_preparation(call_env, monkeypatch, until)
        return contextlib.nullcontext([until])
    return slot_held_until(call_env, monkeypatch, "unseen-light", call_env.route, until)


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


def at(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def no_proxy_timeouts(monkeypatch, content="Useful title"):
    """The production no-proxy client with only its network transport in memory: each request's HTTP timeout."""
    import httpx

    from ouroboros import net_transport

    timeouts = []

    def respond(request):
        timeouts.append(request.extensions["timeout"])
        return httpx.Response(200, json={
            "id": "fixture-generation", "object": "chat.completion", "created": 1, "model": "unseen-light",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.25}})

    monkeypatch.setattr(net_transport, "remote_httpx_transport", lambda *a, **k: httpx.MockTransport(respond))
    return timeouts


def test_place_reply_safety_check_ends_by_the_task_deadline_safety_already_uses(call_env, monkeypatch, supervisor):
    """The reply check runs on the loop thread, outside any tool window. Safety already ends its slot wait and
    its retry at the task deadline; the send itself used to keep the whole safety timeout (#572)."""
    from ouroboros.config import get_safety_call_timeout_sec

    monkeypatch.setattr(supervisor.module, "_safety_model_call", REAL_SAFETY_CALL)
    monkeypatch.setattr(LLMClient, "_resolve_remote_target", lambda self, model: dict(call_env.route))
    monkeypatch.setattr(LLMClient, "_get_remote_client", lambda self, target: call_env.sdk)
    call_env.sdk.chat.completions.create = lambda **kwargs: call_env.sends.append(kwargs) or AssembledResponse({
        "id": "verdict", "choices": [{"message": {"role": "assistant", "content": '{"status":"SAFE","reason":"ok"}'},
                                      "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.01}})
    supervisor.ctx.task_metadata["deadline_at"] = at(120)
    spend_preparation(call_env, monkeypatch, 105)
    reply = {"text": "Here is the answer", "destination": "room", "place_id": "place"}
    assert supervisor.module.check_safety("place_reply", reply, ctx=supervisor.ctx) == (True, "")
    assert get_safety_call_timeout_sec() == 60 and [send["timeout"] for send in call_env.sends] == [15]


@pytest.mark.parametrize("elapsed", ["preparation", "slot"])
def test_verdict_extraction_keeps_its_finalization_reserve_after_preparation_or_a_slot_wait(
        call_env, monkeypatch, elapsed):
    """Extraction refuses inside deadline − grace and projects its timeout from it, once, before the send."""
    from ouroboros import config
    from ouroboros import review_verdict_extraction as extraction

    timeouts = no_proxy_timeouts(monkeypatch)
    monkeypatch.setattr(config, "get_light_model", lambda: "openai-compatible::unseen-light")
    with ua.usage_scope(ua.UsageScope(drive_root=call_env.root, task_id="review", root_task_id="review")), \
            model_wait.calendar_scope(at(300)), \
            spend_until(call_env, monkeypatch, elapsed, 110) as started:  # review custody's raw request deadline
        extraction._extract_verdict_via_light_model("The change looks fine.", llm=call_env.client, deadline_at=at(300))
    # Projected at 100: 300 − 120 (grace) − 100 = 80 s; about 10 s later only 70 s of that window remain.
    assert [(timeout["read"], timeout["connect"]) for timeout in timeouts] == [(pytest.approx(180 - started[0]), 30)]
    assert started[0] >= 110 and ledger_rows(call_env.root)[-1]["state"] == "settled"


@pytest.mark.parametrize("elapsed", ["preparation", "slot"])
def test_vision_caption_keeps_its_reserve_after_image_preparation_or_a_slot_wait(call_env, monkeypatch, elapsed):
    """The caption refuses inside the loop deadline − grace and projects its timeout from it before preparing."""
    from ouroboros import vision_routing

    timeouts = no_proxy_timeouts(monkeypatch, content="A red square.")
    monkeypatch.setattr(deadline_utils, "time", SimpleNamespace(time=lambda: call_env.now[0]))  # epoch deadline_ts
    image = "data:image/png;base64,iVBORw0KGgo="
    monkeypatch.setattr(vision_routing, "prepare_caption_image", lambda *a: (
        "openai-compatible::unseen-light", image, "pixel-digest", "", ""))
    monkeypatch.setattr(vision_routing, "prepare_messages_for_send", lambda messages, **kwargs: messages)
    with ua.usage_scope(ua.UsageScope(drive_root=call_env.root, task_id="loop", root_task_id="loop")), \
            spend_until(call_env, monkeypatch, elapsed, 110) as started:
        caption, failure = vision_routing._caption_for_block(
            {"type": "image_url", "image_url": {"url": image}}, ctx=SimpleNamespace(deadline_ts=300.0),
            llm=call_env.client, accumulated_usage={})
    assert (caption, failure) == ("A red square.", "")
    # min(90, 300 − 120 − 100) = 80 s projected; about 10 s later the send has 70 s.
    assert [(timeout["read"], timeout["connect"]) for timeout in timeouts] == [(pytest.approx(180 - started[0]), 30)]
    assert started[0] >= 110


def light_extraction(call_env, monkeypatch, *, until, owner=True, cancel=False):
    """The review substrate's real verdict extraction, as a review inside a task calls it: its window is the
    review deadline less the grace, 300 − 120 = 180, and its preparation runs to ``until``."""
    from ouroboros import cancel_intents, config
    from ouroboros.task_results import write_task_result

    write_task_result(call_env.root, "review", "running", root_task_id="review")  # the owner's Pause authority
    timeouts = no_proxy_timeouts(monkeypatch, content="[]")
    monkeypatch.setattr(config, "get_light_model", lambda: "openai-compatible::unseen-light")
    spend_preparation(call_env, monkeypatch, until)
    task = {"id": "review", "root_task_id": "review", "deadline_at": at(300)}
    waiter = (model_wait.task_model_wait_scope(task=task, drive_root=call_env.root, event_queue=None,
                                               worker_slot_held=False) if owner else contextlib.nullcontext())
    with ua.usage_scope(ua.UsageScope(drive_root=call_env.root, task_id="review", root_task_id="review")), waiter:
        if cancel:
            cancel_intents.request_cancel(call_env.root, "review", source="extraction-test")
        result = review_verdict_extraction.canonicalize_session_verdict(
            "The change looks fine; I found nothing to fix.", conformance_passed=False,
            llm=call_env.client, deadline_at=at(300))
    return result, timeouts


@pytest.mark.parametrize("owner", [True, False], ids=["wait_owner", "bare"])
def test_verdict_extraction_whose_window_ends_in_preparation_returns_its_own_deadline_result(call_env, monkeypatch,
                                                                                               owner):
    """Spent while it records its candidate, the extraction sends nothing and gives the deadline_exhausted result it
    already gives before admission; the review keeps its own unparsed text. A task's wait owner used to turn that
    refusal into the task's deadline control, and without one its reason was lost."""
    (_text, method, usage), timeouts = light_extraction(call_env, monkeypatch, until=181, owner=owner)
    assert (method, usage) == ("unparsed", {"model": "openai-compatible::unseen-light",
                                            "reason_code": "deadline_exhausted", "dispatch": "not_dispatched"})
    assert timeouts == [] and [row["state"] for row in ledger_rows(call_env.root)] == ["released"]


@pytest.mark.parametrize("until,cancel,reason", [(301, False, "deadline"), (110, True, "cancelled")],
                         ids=["task_deadline", "stop"])
def test_verdict_extraction_keeps_its_tasks_own_controls(call_env, monkeypatch, until, cancel, reason):
    with pytest.raises(model_wait.ModelWaitInterrupted) as interrupted:
        light_extraction(call_env, monkeypatch, until=until, cancel=cancel)
    assert interrupted.value.control_reason == reason


PIXEL = ("data:image/png;base64,"
         "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGNgAAIAAAUAAXpeqz8AAAAASUVORK5CYII=")


@pytest.mark.parametrize("image", [True, False], ids=["captioned", "text_only"])
def test_a_send_time_caption_refused_at_its_window_ends_mains_round_as_mains_own_send_does(call_env, monkeypatch,
                                                                                          image):
    """Captions are made only while Main prepares its own send, whose window is the same deadline − grace. So a
    caption refused there costs nothing Main's own refusal does not: both rounds end in the same interruption,
    and neither sends. Without the caption's bound that round still ended so, after a late paid caption."""
    from ouroboros.task_results import write_task_result

    write_task_result(call_env.root, "main", "running", root_task_id="main")  # the owner's Pause authority
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "caption")
    timeouts = no_proxy_timeouts(monkeypatch, content="A red square.")
    monkeypatch.setattr(deadline_utils, "time", SimpleNamespace(time=lambda: call_env.now[0]))  # epoch deadline_ts
    monkeypatch.setattr(vision_routing, "prepare_caption_image", lambda *a: (
        "openai-compatible::unseen-light", PIXEL, "pixel-digest", "", ""))
    spend_preparation(call_env, monkeypatch, 181)
    content = [{"type": "text", "text": "What is on the picture?"},
               *([{"type": "image_url", "image_url": {"url": PIXEL}}] if image else [])]
    task = {"id": "main", "root_task_id": "main", "deadline_at": at(300)}
    with ua.usage_scope(ua.UsageScope(drive_root=call_env.root, task_id="main", root_task_id="main")), \
            model_wait.task_model_wait_scope(task=task, drive_root=call_env.root, event_queue=None,
                                             worker_slot_held=False), \
            pytest.raises(model_wait.ModelWaitInterrupted) as interrupted:
        loop_llm_call.call_llm_with_retry(
            call_env.client, [{"role": "user", "content": content}], "openai-compatible::unseen-light", None,
            "medium", 1, call_env.root / "logs", "main", 1, None, {}, "task",
            deadline_ts=300.0, transport_reserve_sec=120.0)
    assert interrupted.value.control_reason == "deadline"
    assert (timeouts, call_env.sends) == ([], [])


def accepted_operation(call_env, monkeypatch, *, done_at, lose="", stop_at=None):
    """The engine gateway alone: its operation, accepted at 100, runs on the controlled clock. Each read moves it on
    30 s and it finishes at ``done_at``; ``lose`` drops the create reply (the window ends meanwhile) or one read past
    the window; ``stop_at`` is when the owner presses Stop. Cancel requests record when they came."""
    from ouroboros import cancel_intents
    from ouroboros import llm_claudexor as transport
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from tests.test_llm_claudexor import Gateway, result

    gateway, lost = Gateway(results=[result(cash=0, knowledge="exact")]), []
    gateway.pending, gateway.cancelled_at = True, []
    create, read, cancel = gateway.create_model_operation, gateway.get_model_operation, gateway.cancel_model_operation

    def creating(*args, **kwargs):
        if lose == "create" and not lost:
            lost.append(call_env.now[0])
            call_env.now[0], gateway.lose_create = 190, True
        return create(*args, **kwargs)

    def reading(operation_id, **kwargs):
        call_env.now[0] += 30
        if stop_at is not None and call_env.now[0] >= stop_at:
            cancel_intents.request_cancel(call_env.root, call_env.task, source="accepted-operation-test")
        gateway.pending = call_env.now[0] < done_at
        if lose == "read" and not lost and call_env.now[0] > 180:
            lost.append(call_env.now[0])
            raise ClaudexorUnavailable("daemon_unreachable", "controlled lost read")
        return read(operation_id, **kwargs)

    def cancelling(operation_id, **kwargs):
        gateway.cancelled_at.append(call_env.now[0])
        return cancel(operation_id, **kwargs)

    gateway.create_model_operation, gateway.get_model_operation = creating, reading
    gateway.cancel_model_operation = cancelling
    monkeypatch.setattr(transport, "ensure_owned_gateway", lambda **kwargs: gateway)
    monkeypatch.setattr(transport.config, "CLAUDEXOR_MODEL_POLL_INTERVAL_SEC", 0.001)
    monkeypatch.setattr(call_env.client, "_resolve_remote_target", LLMClient()._resolve_remote_target)
    return gateway


@contextlib.contextmanager
def task_wait_owner(call_env, task_id):
    """The task's real wait owner, its deadline at 300, and its Stop/Pause authority on disk."""
    from ouroboros.task_results import write_task_result

    call_env.task = task_id
    write_task_result(call_env.root, task_id, "running", root_task_id=task_id)
    task = {"id": task_id, "root_task_id": task_id, "deadline_at": at(300)}
    with ua.usage_scope(ua.UsageScope(drive_root=call_env.root, task_id=task_id, root_task_id=task_id)), \
            model_wait.task_model_wait_scope(task=task, drive_root=call_env.root, event_queue=None,
                                             worker_slot_held=False):
        yield


def managed_caption(call_env, monkeypatch):
    """Main's preparation captions its image on a managed route; the caption's window is 300 − 120 = 180."""
    from tests.test_llm_claudexor import MODEL

    monkeypatch.setattr(deadline_utils, "time", SimpleNamespace(time=lambda: call_env.now[0]))  # epoch deadline_ts
    monkeypatch.setattr(vision_routing, "prepare_caption_image", lambda *a: (MODEL, PIXEL, "pixel-digest", "", ""))
    monkeypatch.setattr(vision_routing, "prepare_messages_for_send", lambda messages, **kwargs: messages)
    with task_wait_owner(call_env, "main"):
        return vision_routing._caption_for_block({"type": "image_url", "image_url": {"url": PIXEL}},
                                                 ctx=SimpleNamespace(deadline_ts=300.0), llm=call_env.client,
                                                 accumulated_usage={})


def collected_once(call_env, gateway):
    """One upload, one generation under one idempotency key, no cancel request, one ACK, one settled row."""
    assert gateway.cancels == [] and len(gateway.uploads) == 1 and len(gateway.accepted_operations) == 1
    assert set(gateway.creates) == {gateway.uploads[0][1]} and len(gateway.acks) == 1
    assert [row["state"] for row in ledger_rows(call_env.root)] == ["settled"]


@pytest.mark.parametrize("lose", ["", "create", "read"], ids=["running", "lost_create", "lost_read"])
def test_an_accepted_caption_outlives_its_own_window_on_the_same_operation(call_env, monkeypatch, lose):
    """Accepted at 100, the caption is still running when its window ends at 180 and finishes at 250, before the
    task's own deadline at 300. That window only admitted the send: the task's wait owner used to read it as a
    deadline and request the operation's cancellation. A lost create reply or read past 180 rejoins it too."""
    gateway = accepted_operation(call_env, monkeypatch, done_at=250, lose=lose)
    assert managed_caption(call_env, monkeypatch) == ("Ответ 🐍", "")
    collected_once(call_env, gateway)
    assert call_env.now[0] >= 250 and len(gateway.creates) == (2 if lose == "create" else 1)


@pytest.mark.parametrize("done_at,stop_at,reason,cancelled_at", [
    (400, None, "deadline", 310),  # the first read at or past the task's own deadline
    (400, 220, "cancelled", 220),  # the owner's Stop, after the caption's window ended
], ids=["task_deadline", "stop"])
def test_the_tasks_own_deadline_and_stop_still_cancel_an_accepted_caption(call_env, monkeypatch, done_at, stop_at,
                                                                          reason, cancelled_at):
    gateway = accepted_operation(call_env, monkeypatch, done_at=done_at, stop_at=stop_at)
    with pytest.raises(model_wait.ModelWaitInterrupted) as interrupted:
        managed_caption(call_env, monkeypatch)
    assert interrupted.value.control_reason == reason
    assert gateway.cancels == [("op-0", "host_cancelled")] and gateway.cancelled_at == [cancelled_at]
    assert len(gateway.creates) == 1 and gateway.acks == []


def test_a_caption_window_spent_before_its_create_still_sends_nothing(call_env, monkeypatch):
    """Preparation runs to 181: the same window refuses the first create, positively unsent, as before."""
    gateway = accepted_operation(call_env, monkeypatch, done_at=250)
    spend_preparation(call_env, monkeypatch, 181)
    with pytest.raises(model_wait.ModelWaitInterrupted) as interrupted:
        managed_caption(call_env, monkeypatch)
    assert interrupted.value.control_reason == "deadline"
    assert (gateway.uploads, gateway.creates, gateway.cancels) == ([], [], [])
    assert [row["state"] for row in ledger_rows(call_env.root)] == ["released"]


def test_an_accepted_verdict_extraction_outlives_its_own_window(call_env, monkeypatch):
    """The same class: the review deadline 300 less the 120 s grace admits the managed extraction at 100; it
    finishes at 250, before the review's own deadline, on its one operation."""
    from ouroboros import config
    from tests.test_llm_claudexor import MODEL

    gateway = accepted_operation(call_env, monkeypatch, done_at=250)
    monkeypatch.setattr(config, "get_light_model", lambda: MODEL)
    with task_wait_owner(call_env, "review"):
        review_verdict_extraction.canonicalize_session_verdict(
            "The change looks fine; I found nothing to fix.", conformance_passed=False,
            llm=call_env.client, deadline_at=at(300))
    collected_once(call_env, gateway)


@pytest.mark.parametrize(("bound", "cancelled_at", "asynchronous"), [
    pytest.param("caller_deadline_ts", [190], False, id='callers_own_deadline-sync'),
    pytest.param("caller_dispatch_deadline_ts", [], True, id='helper_send_window-async'),
])
def test_only_a_callers_own_deadline_ends_the_wait_for_an_accepted_operation(call_env, monkeypatch, bound,
                                                                             cancelled_at, asynchronous):
    """The same managed chat, wait owner and running operation, both bounds at 180. Main's own deadline less its
    finalization reserve still ends its wait there and requests the cancellation; a helper's send window does not."""
    from tests.test_llm_claudexor import MODEL

    gateway = accepted_operation(call_env, monkeypatch, done_at=250)
    with task_wait_owner(call_env, "main"), contextlib.ExitStack() as stack:
        if cancelled_at:
            interrupted = stack.enter_context(pytest.raises(model_wait.ModelWaitInterrupted))
        arguments = {"messages": [{"role": "user", "content": "hi"}], "model": MODEL, bound: 180.0}
        message, _usage = (asyncio.run(call_env.client.chat_async(**arguments)) if asynchronous
                           else call_env.client.chat(**arguments))
    assert gateway.cancelled_at == cancelled_at and len(gateway.creates) == 1
    if cancelled_at:
        assert interrupted.value.control_reason == "deadline" and gateway.acks == []
    else:
        assert message["content"] == "Ответ 🐍"
        collected_once(call_env, gateway)
