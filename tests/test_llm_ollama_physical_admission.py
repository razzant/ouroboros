"""Every physical send to a private compatible connection has its own decision (G7): a retry, a send after a slot
wait and a Settings Test are each decided by the read made for them, the decision that admitted a send is the one
its caller reports, and a redirect never carries a private chat elsewhere.

Synthetic transports and fake request dispatch only: no socket, endpoint, key or model is used.
"""
import asyncio

import httpx
import pytest

from tests._ollama import TARGET, MESSAGES, GatewayHTTP, LlamaCppHTTP, OllamaHTTP, _chat_completion
from tests._ollama import install_http as install_http
from tests.test_llm_ollama_gateway import GATEWAY, MODEL, _shown, _store, _test_settings
from tests.test_llm_ollama_gateway import _unstreamed_client, world as world
from tests._usage_store_testing import attempt_rows_in_start_order


def _answer(content):
    from openai.types.chat import ChatCompletion

    return ChatCompletion.model_validate(_chat_completion(content).json())


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("meanwhile", ["unchanged", "another program named", "consent withdrawn", "ollama seen there"])
def test_a_repaired_resend_is_decided_by_its_own_read(install_http, world, tmp_path, asynchronous, meanwhile):
    """The call's read admits its first send by consent; the server refuses an optional parameter and the ladder
    resends at once, with no slot wait. That resend reads again: the owner's consent as it stands then and what the
    address shows then decide it. An unchanged admitted server still receives the repaired request."""
    from ouroboros import llm_ollama
    from ouroboros.llm import LLMClient
    from ouroboros.llm_ollama_guard import UnidentifiedRouteWithheld, note_ollama
    from ouroboros.llm_routing import resolve_target

    world.settings["OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED"] = GATEWAY
    gateway, cpp, named = GatewayHTTP(), LlamaCppHTTP(), []
    install_http(lambda request: (cpp if named else gateway)(request))
    target = resolve_target(MODEL, settings=world.settings)
    assert llm_ollama.is_ollama(target, before_send=True) is False
    sent = []

    def change():
        if meanwhile == "another program named":
            named.append(True)
        elif meanwhile == "consent withdrawn":  # the owner clears the box between the two sends
            world.settings["OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED"] = ""
        elif meanwhile == "ollama seen there":  # another connection's read at this address
            assert note_ollama({"base_url": GATEWAY}, "0.40.16", "/api/version+/api/ps+/api/show")

    def create(**candidate):
        sent.append(candidate)
        if len(sent) == 1:
            change()
            raise RuntimeError("unsupported parameter: cache")
        return _answer("repaired")

    async def create_async(**candidate):
        return create(**candidate)

    request = {"model": "mock-model", "messages": MESSAGES, "extra_body": {"cache": {"type": "ephemeral"}}}
    client = LLMClient()

    def run():
        from ouroboros import usage_accounting as ua

        with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="resend", root_task_id="resend")):
            if asynchronous:
                return asyncio.run(client._create_chat_completion_with_retries_async(create_async, request, target))
            return client._create_chat_completion_with_retries(create, request, target)

    if meanwhile in ("consent withdrawn", "ollama seen there"):
        with pytest.raises(UnidentifiedRouteWithheld) as withheld:
            run()
        assert withheld.value.code == {"consent withdrawn": "ollama_guard_consent",
                                       "ollama seen there": "ollama_guard_held"}[meanwhile]
        assert len(sent) == 1, "the repaired request was not sent"
        assert "nothing was sent" in withheld.value.route_caution, "the round hears why, in the refusal's own words"
        assert "by your consent" in withheld.value.route_caution_used[0], "and of the route the first send used"
        rows = attempt_rows_in_start_order(tmp_path)
        assert len(rows) == 2 and rows[-1]["state"] == "released", "and its prepared attempt is given back"
    else:
        assert run().choices[0].message.content == "repaired"
        assert len(sent) == 2 and "cache" not in (sent[1].get("extra_body") or {})
        assert "by your consent" in target["route_caution"], "the call keeps the caution of its first send"
    reads = [path for method, path in gateway.paths + cpp.paths if path == "/api/version"]
    assert len(reads) == 2, "the call's read decided the first send, the resend read again"
    assert all(request.headers.get("authorization") == "Bearer key-one"
               for request in gateway.requests if request.method == "GET"), "each read carries the connection's key"


def _queue_once(monkeypatch, tmp_path, during):
    """The next send waits once for its model slot; ``during`` runs while it waits."""
    from ouroboros import model_concurrency as slots

    take, queued = slots._Waiter.step, []

    def wait_once(waiter):
        if not queued:
            queued.append(waiter)
            during()
            return False
        return take(waiter)

    monkeypatch.setattr(slots._Waiter, "step", wait_once)
    monkeypatch.setattr(slots, "_queue_root", lambda: tmp_path / "slots")
    monkeypatch.setattr(slots, "_cap", lambda local: 1)
    return queued


def test_a_settings_test_reports_the_decision_its_physical_send_was_made_by(install_http, monkeypatch, tmp_path):
    """llama.cpp names itself to the Test's read, then stops answering its metadata while the Test waits for its
    slot: the read after the wait finds no program, the draft's consent admits the send, and the answer says so."""
    import ouroboros.gateway.models as provider_api
    from ouroboros.llm_ollama_guard import CHECKS, PRIOR_ROUTES, TEST_CAUTION, TOLD

    cpp, gateway, closed = LlamaCppHTTP(), GatewayHTTP(), []
    install_http(lambda request: (gateway if closed else cpp)(request))
    queued = _queue_once(monkeypatch, tmp_path, lambda: closed.append(True))
    result = provider_api._run_provider_test_with_settings("openai-compatible", _test_settings(
        OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED="http://draft.lan/v1"))
    assert queued and result == {"ok": True, "caution": TEST_CAUTION}, result
    assert gateway.chats == ["/v1/chat/completions"] and ("POST", "/v1/chat/completions") not in cpp.paths
    assert not {PRIOR_ROUTES, CHECKS, TOLD} & set(_store()), "the Test admitted and recorded nothing"


def test_a_settings_test_names_what_kept_its_send_back(install_http, world):
    """``code`` with a failed Test whose compatible send was withheld; none for a pass or another failure."""
    import ouroboros.gateway.models as provider_api

    def test(**more):
        return provider_api._run_provider_test_with_settings("openai-compatible", _test_settings(**more))

    consent = "http://draft.lan/v1"
    gateway = install_http(GatewayHTTP())
    assert test()["code"] == "ollama_guard_consent" and gateway.chats == []
    assert "code" not in test(OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED=consent)
    install_http(GatewayHTTP(chat=lambda request: httpx.Response(401, json={"error": {"message": "bad key"}})))
    assert test(OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED=consent) == {
        "ok": False, "error": "Invalid key", "caution": "Server type not confirmed: Ouroboros cannot tell whether "
                                                      "it keeps the whole history."}
    native = OllamaHTTP()
    install_http(lambda request: httpx.Response(200, json={"version": "0.12.5"})
                 if request.url.path == "/api/version" else native(request))
    old = test(OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED=consent)
    assert old["code"] == "ollama_guard_unavailable" and "Upgrade to 0.12.6" in old["error"] and native.posts == []
    silent = install_http(GatewayHTTP(httpx.Response(401, text="no")))
    assert test(OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED=consent)["code"] == "ollama_guard_held" and silent.chats == []


def test_the_first_use_is_shown_after_a_retried_empty_reply_and_told_only_when_shown(
        install_http, world, tmp_path, monkeypatch):
    """The first reply on an unnamed route is empty and retried: the caution of that first use still reaches the
    round's line, whatever the retry answers, and counts as told only once a line carried it."""
    from ouroboros import loop_llm_call
    from ouroboros.llm_ollama_guard import TOLD
    from ouroboros.loop_transport import emit_route_notice

    monkeypatch.setattr(loop_llm_call, "_sleep_within_deadline", lambda *args, **kwargs: True)
    client = _unstreamed_client(monkeypatch)
    world.settings["OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED"] = GATEWAY
    replies = iter(["", "after the retry", "next task"])
    server = install_http(GatewayHTTP(chat=lambda request: _chat_completion(next(replies))))

    def call(usage):
        message, _ = loop_llm_call.call_llm_with_retry(client, MESSAGES, MODEL, None, "low", 3, tmp_path / "logs",
                                                       "root", 1, None, usage, initial_messages=MESSAGES)
        return message

    usage = {}
    assert call(usage)["content"] == "after the retry" and len(server.chats) == 2
    assert TOLD not in _store(), "nothing counts as told before a line carries it"
    emit_route_notice(usage, None)  # a round with no line keeps the words
    assert TOLD not in _store()
    words = _shown(usage)
    assert "did not show which program" in words and "by your consent" in words and len(_store()[TOLD]) == 1
    later = {}
    assert call(later)["content"] == "next task" and _shown(later) is None, "a told route is not told again"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_private_chat_is_never_carried_to_a_redirect_destination(install_http, world, asynchronous):
    asked = []

    def server(request):
        asked.append(request.url.host)
        if request.url.host != "gateway.lan":
            pytest.fail("the redirect destination received the chat")
        if request.method == "POST":
            return httpx.Response(307, headers={"Location": "http://elsewhere.lan/v1/chat/completions"})
        return GatewayHTTP()(request)

    install_http(server)
    world.settings["OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED"] = GATEWAY
    with pytest.raises(Exception):
        world(asynchronous=asynchronous)
    assert set(asked) == {"gateway.lan"}


def test_a_public_compatible_route_is_read_and_held_by_nothing(install_http, world):
    """The contrast: a public endpoint gets no metadata read, no admission and keeps its client's redirects."""
    from ouroboros.llm_routing import resolve_target

    public = "https://api.example.com/v1"
    world.settings.update(OPENAI_COMPATIBLE_BASE_URL=public)
    assert "follow_redirects" not in resolve_target(MODEL, settings=world.settings)
    assert resolve_target(MODEL, settings={**world.settings, "OPENAI_COMPATIBLE_BASE_URL": GATEWAY})[
        "follow_redirects"] is False
    seen = []

    def server(request):
        seen.append((request.method, request.url.path))
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(307, headers={"Location": "https://api.example.com/v2/chat/completions"})
        return _chat_completion("public")

    install_http(server)
    assert world()[0]["content"] == "public"
    assert seen == [("POST", "/v1/chat/completions"), ("POST", "/v2/chat/completions")], "no read, redirect as before"


def test_a_private_connection_without_its_key_is_asked_nothing(install_http):
    from ouroboros import llm_ollama

    install_http(lambda request: pytest.fail(f"asked {request.url}"))
    target = {**TARGET, "base_url": GATEWAY, "api_key": "", "keyless": False}
    assert llm_ollama.is_ollama(target, before_send=True) is False and "ollama_probe_no_proxy" not in target


def test_a_connections_own_read_records_what_it_found_again(install_http, world):
    """Settings' check follows this connection's own reads: Ollama found again after a read that named nothing, the
    program that replaced it, and an Ollama too old for its guard, with its version."""
    from ouroboros.llm_ollama import OllamaUpgradeRequired
    from ouroboros.llm_ollama_guard import UnidentifiedRouteWithheld

    def checked():
        [row] = _store()["compatible_checks"].values()
        return row["type"], row.get("version", "")

    install_http(OllamaHTTP())
    assert world()[0]["content"] == "391" and checked() == ("ollama", "0.40.16")
    install_http(GatewayHTTP(httpx.Response(403, text="Forbidden")))
    with pytest.raises(UnidentifiedRouteWithheld):
        world()
    assert checked() == ("unknown", "")
    install_http(OllamaHTTP())
    assert world()[0]["content"] == "391" and checked() == ("ollama", "0.40.16"), "found again natively"
    cpp = install_http(LlamaCppHTTP())
    message, usage = world()
    assert message["content"] == "from llama.cpp" and "no longer answers as Ollama" in usage["route_notice"]
    assert checked() == ("llama.cpp", "") and cpp.paths.count(("GET", "/api/version")) == 1, "one read, one send"
    native = OllamaHTTP()
    install_http(lambda request: httpx.Response(200, json={"version": "0.12.5"})
                 if request.url.path == "/api/version" else native(request))
    with pytest.raises(OllamaUpgradeRequired):
        world()
    assert checked() == ("ollama", "0.12.5") and native.posts == []


def test_a_route_switch_to_an_old_ollama_leaves_the_refusal_to_its_send(monkeypatch):
    """A configured route on an Ollama too old for its guard is one route's failure: the switch keeps unknown evidence
    (the send's own read then refuses it before any generation, typed as a bad request no repeat retries), and the
    walk to the next configured route is not fenced by it."""
    from types import SimpleNamespace
    from ouroboros import context, context_fit, loop_model_call
    from ouroboros.llm_ollama import OllamaUpgradeRequired
    from ouroboros.loop_llm_call import classify_llm_exception

    refusal = OllamaUpgradeRequired("Ollama 0.12.5 can silently truncate history. Upgrade to 0.12.6 or later.")

    class UnknownEvidence(BaseException):
        """Stops the switch where it takes unknown evidence."""

    def ask(task, *, allow_fetch):
        raise refusal

    def unknown(task):
        raise UnknownEvidence(task)

    monkeypatch.setattr(context, "_context_fit_route", ask)
    monkeypatch.setattr(context_fit, "_failed_route_evidence", unknown)
    plan = SimpleNamespace(max_projection=None, low_projection=None, core_sha256="", model_role="main")
    with pytest.raises(UnknownEvidence):
        loop_model_call._rebind_context_fit_plan(plan, SimpleNamespace(_ctx=SimpleNamespace()), [],
                                                 model="openai-compatible::old", use_local=False, tool_schemas=[])
    verdict = classify_llm_exception(refusal)
    assert (verdict.kind, verdict.retry_same_request) == ("bad_request", False)
    assert not loop_model_call._walk_fenced(SimpleNamespace(), {"_last_llm_error_kind": verdict.kind})


def test_a_round_hears_both_the_route_its_first_send_used_and_why_a_later_one_was_withheld(install_http, tmp_path):
    """A call whose first send went on an unnamed route by consent and whose repaired resend was then withheld says
    both, each once: the first use, and the refusal's own words."""
    from types import SimpleNamespace
    from ouroboros.llm_ollama_guard import UnidentifiedRouteWithheld
    from ouroboros.loop_llm_call import call_llm_with_retry
    from ouroboros.loop_transport import emit_route_notice

    refusal = UnidentifiedRouteWithheld("ollama_guard_consent", "The server at http://gateway.lan:8080 does not show "
                                        "which program it runs ...; nothing was sent.")
    refusal.route_caution_used = ("The server at http://gateway.lan:8080 did not show which program it runs. "
                                  "This request uses OpenAI-compatible /v1 by your consent ...", "route-digest")

    def chat(**kwargs):
        raise refusal

    usage, said = {}, []
    message, _ = call_llm_with_retry(SimpleNamespace(chat=chat), MESSAGES, MODEL, None, "low", 3, tmp_path / "logs",
                                     "root", 1, None, usage, initial_messages=MESSAGES)
    emit_route_notice(usage, said.append)
    assert message is None and len(said) == 2
    assert "by your consent" in said[0] and "nothing was sent" in said[1]


def test_an_accounting_wait_outlives_the_prepared_verdict_and_the_send_reads_again(install_http, world, monkeypatch):
    """The read just before the call admits its first send; ledger contention then delays a later dispatch and
    the owner withdraws consent during that wait. The delayed send re-reads — the verdict expired with the wait
    it did not survive — while an undelayed send keeps the decision its own read made."""
    from ouroboros import _usage_wait, llm_ollama, model_concurrency
    from ouroboros.llm_ollama_guard import UnidentifiedRouteWithheld
    from ouroboros.llm_routing import resolve_target

    monkeypatch.setattr(model_concurrency, "limit_physical_send", lambda route, send, after_wait=None: send)
    monkeypatch.setattr(_usage_wait, "_LAST_PRESEND_WAIT", 0.0)
    world.settings["OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED"] = GATEWAY
    server = install_http(GatewayHTTP())

    def resolve(_route):
        return resolve_target(MODEL, settings=world.settings)

    def reads():
        return sum(1 for _method, path in server.paths if path == "/api/version")

    target = resolve_target(MODEL, settings=world.settings)
    assert llm_ollama.is_ollama(target, before_send=True) is False
    before, sent = reads(), []
    llm_ollama.limit_compatible_send(target, lambda: sent.append(1), resolve=resolve)()
    assert sent == [1] and reads() == before, "an undelayed send keeps the verdict its own read made"

    delayed = resolve_target(MODEL, settings=world.settings)
    assert llm_ollama.is_ollama(delayed, before_send=True) is False  # the verdict is prepared under the consent
    world.settings["OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED"] = ""    # withdrawn before the physical dispatch
    _usage_wait._note_presend_wait()                                 # ... which the ledger contention delays
    with pytest.raises(UnidentifiedRouteWithheld) as withheld:
        llm_ollama.limit_compatible_send(delayed, lambda: sent.append(1), resolve=resolve)()
    assert withheld.value.code == "ollama_guard_consent" and len(sent) == 1, "the expired verdict admits nothing"


def test_a_caution_counts_as_told_only_once_a_line_carried_it(install_http, world):
    """An emitter that reports a failed enqueue leaves the caution pending and its route untold; the line that
    carries it tells it once — the first-use warning is never marked shown before it reached the chat."""
    from ouroboros.llm_ollama_guard import TOLD
    from ouroboros.loop_llm_call import note_route_caution
    from ouroboros.loop_transport import emit_route_notice

    usage, failed = {}, []

    def broken(line):
        failed.append(line)
        return False

    note_route_caution(usage, "words of an unnamed route", route="route-digest")
    emit_route_notice(usage, broken)
    assert failed and usage.get("_route_cautions_pending"), "an enqueue nobody took keeps the words"
    assert TOLD not in _store(), "and its route stays untold"
    said = []
    emit_route_notice(usage, said.append)
    assert said == ["🌐 words of an unnamed route"] and not usage.get("_route_cautions_pending")
    assert len(_store()[TOLD]) == 1, "told once, by the line that carried it"
