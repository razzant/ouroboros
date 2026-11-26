"""Native Ollama contracts for loop_llm_call."""
import pytest
from tests._ollama import (
    TARGET, TOOLS, completed, OllamaHTTP,
    chat, install_http as install_http,
)


@pytest.mark.parametrize("make,kind,retry", [
    (lambda w: w.OllamaIntegrityError("no terminal frame"), "provider_error", False),
    (lambda w: w.OllamaUnsupportedRequest("forced tool_choice"), "bad_request", False),
    (lambda w: w.OllamaRouteUnsettledError("could not be saved"), "provider_transient", True),
])
def test_native_verdicts_carry_their_own_repeat_class(make, kind, retry):
    from ouroboros import llm_ollama_wire as wire
    from ouroboros.loop_llm_call import classify_llm_exception

    error = make(wire)
    verdict = classify_llm_exception(error)
    assert (verdict.kind, verdict.retry_same_request, verdict.provider_code) == (kind, retry, error.code)


@pytest.mark.parametrize("case", ["incomplete", "forced_tool", "unsaved"])
def test_each_native_failure_form_is_classified_from_the_real_raise(install_http, monkeypatch, case):
    """The exceptions the adapter actually raises, not hand-built ones."""
    from ouroboros import capability_evidence as ce, llm_ollama
    from ouroboros.loop_llm_call import classify_llm_exception

    reply, kwargs = completed(), {}
    if case == "incomplete":
        reply, kwargs = [{"message": {"role": "assistant", "content": "Part"}, "done": False}], {"stream": True}
    elif case == "forced_tool":
        kwargs = {"tools": TOOLS, "tool_choice": "required"}
    server = install_http(OllamaHTTP(reply))
    with pytest.raises(Exception) as raised:
        if case == "unsaved":
            monkeypatch.setattr(ce, "_store_evidence", lambda *args, **kwargs: False)
            llm_ollama.is_ollama({**TARGET, "base_url": "http://127.0.0.1:11461/v1"})
        else:
            chat(**kwargs)
    verdict = classify_llm_exception(raised.value)
    assert verdict.retry_same_request is (case == "unsaved"), (case, raised.value)
    if case == "forced_tool":
        assert verdict.kind == "bad_request" and server.posts == []


@pytest.mark.parametrize("unfinished_first_frame", [False, True])
def test_incomplete_native_round_retains_partial_and_blocks_configured_fallback(install_http, tmp_path, monkeypatch, unfinished_first_frame):
    from types import SimpleNamespace
    from ouroboros.llm import LLMClient
    from ouroboros.loop_llm_call import call_llm_with_retry
    from ouroboros.loop_transport import fallback_chain_allowed
    from tests._ollama import MESSAGES, _compatible_target

    _compatible_target(monkeypatch, TARGET["base_url"])
    client = LLMClient()
    native_chat = client.chat
    monkeypatch.setattr(client, "chat", lambda **kw: native_chat(**{**kw, "stream": True}))
    import httpx
    frames = [{"message": {"content": "partial"}, "done": False}]
    server = install_http(OllamaHTTP(httpx.Response(200, content=b'{"message":') if unfinished_first_frame else frames))
    usage = {}
    message, _ = call_llm_with_retry(client, MESSAGES, TARGET["requested_model"], None, "low", 3,
        tmp_path / "logs", "root", 1, None, usage, initial_messages=MESSAGES, transport_death_retries=2)
    assert message is None and len(server.posts) == 1
    assert usage["_last_llm_error_kind"] == "provider_outcome_unknown"
    assert usage["_pending_transport_outcome"]["retain_partial_reply"]
    assert not fallback_chain_allowed(SimpleNamespace(task_id="root"), usage["_last_llm_error_kind"], None, usage)


def test_route_notice_reaches_the_round_as_a_shown_fact():
    """Said at the first round whose reply reports it, again only after a round without it. A retry inside one call
    (an empty first reply) keeps it for the line the round shows after the call (``loop_transport.emit_route_notice``)."""
    from ouroboros.loop_llm_call import _take_response_facts
    from ouroboros.loop_transport import emit_route_notice

    accumulated = {}
    notice = "The program at http://127.0.0.1:11464 no longer answers as Ollama"

    def round_(*replies):
        for facts in replies:  # one call's replies: no line between them
            _take_response_facts(facts, {"role": "assistant", "content": "ok"}, accumulated)
        said = []
        emit_route_notice(accumulated, said.append)
        return said

    assert round_({"route_notice": notice}, {"route_notice": notice}) == ["🌐 " + notice]
    assert round_({"route_notice": notice}) == [] and round_({}) == []
    assert round_({"route_notice": notice}) == ["🌐 " + notice]


def test_a_route_notice_is_said_once_and_consumed_even_when_no_line_can_carry_it():
    from ouroboros.loop_transport import emit_route_notice

    said = []
    usage = {"_route_notice": "The program at http://127.0.0.1:11464 no longer answers as Ollama"}
    emit_route_notice(usage, said.append)
    assert said == ["🌐 The program at http://127.0.0.1:11464 no longer answers as Ollama"] and usage == {}
    emit_route_notice(usage, said.append)  # nothing pending: nothing said
    assert len(said) == 1
    usage = {"_route_notice": "again"}
    emit_route_notice(usage, None)
    assert usage == {} and len(said) == 1
