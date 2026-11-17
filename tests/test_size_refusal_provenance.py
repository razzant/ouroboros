"""Panel 5 regressions: numeric facts belong to the recognized provider answer."""
import io
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ouroboros import capability_evidence as ce
from ouroboros import size_refusal as sr
from ouroboros.context_window import record_refusal
from ouroboros.llm_attempt import body_error_facts
from ouroboros.loop_transport import stamp_owner_provider_message
from ouroboros.window_learning import learn_window_from_refusal
from tests._size_refusal import _sdk_error


def test_engine_display_recognizes_size_but_never_supplies_numbers(tmp_path, monkeypatch):
    from ouroboros.llm_claudexor import ClaudexorModelError
    from ouroboros.loop_llm_call import classify_llm_exception

    exc = ClaudexorModelError({"code": "context_length_exceeded", "message":
        "This model's maximum context length is 272000 tokens. However, your messages resulted in 352113 tokens."})
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    usage = {}
    verdict = classify_llm_exception(exc)
    assert verdict.kind == "context_overflow"
    stamp_owner_provider_message(usage, exc, verdict)
    sr.stamp(usage, exc)
    assert not usage.get("_last_llm_provider_message")
    fact = sr.for_usage(usage)
    assert (fact.window, fact.requested, fact.reply, fact.requested_total) == (0, 0, 0, 0)
    plan = SimpleNamespace(provider="claudexor", route_fp="account-a", window_record=None,
                           window_witness_bound=True, account_pin="account-a")
    record_refusal(usage, plan)
    assert not ce._read(tmp_path).get("main_windows")
    # The same nonnumeric fact cannot trigger early learning on any eligible transport either.
    loop = SimpleNamespace(_rebind_context_fit_plan=Mock(side_effect=AssertionError("unexpected early rebind")))
    ctx = SimpleNamespace(accumulated_usage=usage, context_fit_plan=SimpleNamespace(provider="openai-compatible"),
                          active_use_local=False)
    assert not learn_window_from_refusal(ctx, loop)
    loop._rebind_context_fit_plan.assert_not_called()


def _structured(window=4000):
    return {"type": sr.LLAMA_CPP_TYPE, "n_ctx": window, "n_prompt_tokens": 90000,
            "message": f"request (90000 tokens) exceeds the available context size ({window} tokens)"}


@pytest.mark.parametrize("location", ["previous_errors", "sibling"])
def test_structured_numbers_belong_to_recognized_object(location):
    current = {"code": "context_length_exceeded", "type": "invalid_request_error",
               "message": "prompt is too long: 90000 tokens > 80000 maximum"}
    body = {"code": 400, "message": "Provider returned error", "metadata": {
        "provider_name": "current", "raw": json.dumps({"error": current})}}
    if location == "previous_errors":
        body["metadata"]["previous_errors"] = [{"code": 400, "message": "Provider returned error",
            "provider_name": "previous", "raw": json.dumps({"error": _structured()})}]
    else:
        body["diagnostic"] = _structured()
    exc = _sdk_error(400, json.dumps(body).encode())
    found = sr.from_exception(exc)
    assert (found.window, found.requested, found.meaning) == (80000, 90000, "input")


def test_numberless_typed_http200_refusal_never_borrows_sibling_sentence():
    from ouroboros._usage_response import provider_answer
    from ouroboros.context_budget import size_refusal

    error = {"code": 400, "message": "Provider returned error",
             "current": {"provider_name": "current", "code": "context_length_exceeded"},
             "diagnostic": {"provider_name": "unrelated",
                            "message": "prompt is too long: 90000 tokens > 4000 maximum"}}
    answer = provider_answer(error)
    assert size_refusal(answer, 400) == ("typed", "context_length_exceeded", "current", "")
    found = sr.from_answer(answer, 400)
    assert (found.window, found.requested, found.upstream) == (0, 0, "")
    usage = {}
    facts = body_error_facts(error, False)
    stamp_owner_provider_message(usage, facts)
    # The owner quotation keeps upstream reporting, but cannot recreate rejected numbers.
    assert "4000" in usage["_last_llm_provider_message"]
    consumed = sr.for_usage(usage)
    assert (consumed.window, consumed.requested, consumed.upstream) == (0, 0, "")


def test_printed_only_recognition_never_selects_between_provider_sentences():
    from ouroboros._usage_response import provider_answer
    from ouroboros.context_budget import size_refusal

    error = {"code": 400, "type": "BadRequestError", "message":
             "'max_tokens' or 'max_completion_tokens' is too large: 90000. This model's maximum context length "
             "is 80000 tokens and your request has 1000 input tokens (90000 > 80000 - 1000).",
             "detail": "Additional provider diagnostics"}
    exc = _sdk_error(400, json.dumps({"error": error}).encode())
    answer = provider_answer(exc)
    assert len([row for row in answer.said if not row[2]]) == 2
    assert size_refusal(answer, 400, printed=repr(exc)) == ("frozen", "", "", "")
    found = sr.from_exception(exc)
    assert not found.has_numbers and found.upstream == ""


def test_recognized_wrapper_can_read_its_own_structured_child():
    body = {"code": "context_length_exceeded", "message": "context exhausted",
            "metadata": {"raw": json.dumps({"error": _structured(80000)})}}
    found = sr.from_exception(_sdk_error(400, json.dumps(body).encode()))
    assert (found.window, found.requested) == (80000, 90000)


@pytest.mark.parametrize("message,expected", [
    ("input length and max_tokens exceed context limit: 90000 + 8192 > 80000", (80000, 90000, 8192)),
    ("prompt is too long: 90000 tokens > 80000 maximum", (80000, 90000, 0)),
])
def test_http200_keeps_each_recognized_numeric_fact(message, expected):
    facts = body_error_facts({"code": 400, "type": "invalid_request_error", "message": message}, False)
    found = facts["answer"].get("size")
    assert found is not None
    assert (found["window"], found["requested"], found["reply"]) == expected


@pytest.mark.parametrize("basis", ["status_413", "number_equals_declared"])
@pytest.mark.parametrize("via", ["stamp", "answer"])
def test_numberless_fact_does_not_shadow_unitless_recognition(basis, via):
    fact = sr.SizeRefusal(source="typed").to_json()
    answer = {"basis": basis, "limit": 80000, "limit_kind": "input"}
    usage = {"_last_llm_provider_answer": answer}
    if via == "stamp":
        usage[sr.USAGE_KEY] = fact
    else:
        answer["size"] = fact
    found = sr.for_usage(usage)
    assert (found.window, found.units, found.meaning, found.source) == (80000, "unstated", "input", basis)
    assert found.token_window == 0


def test_http200_text_refusal_rebinds_real_sdk_request_to_nano(tmp_path, monkeypatch):
    from tests._first_contact import ANSWER, _episode, _run
    from tests._first_contact_server import FakeLlamaServer

    def refusal(count):
        return 200, {"error": {"code": 400, "type": "invalid_request_error", "message":
            f"input length and max_tokens exceed context limit: {count} + 65536 > 80000"}}

    with FakeLlamaServer(n_ctx=80000, publish_window=False, refuse_with=refusal) as server:
        base = server._server.RequestHandlerClass

        class Handler(base):
            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                if json.loads(raw).get("stream"):
                    return self._raw(400, {"error": {"code": 400, "type": "invalid_request_error",
                        "param": "stream", "message": "Unsupported parameter: stream"}})
                original, self.rfile = self.rfile, io.BytesIO(raw)
                try:
                    return super().do_POST()
                finally:
                    self.rfile = original

        server._server.RequestHandlerClass = Handler
        episode = _episode(tmp_path, monkeypatch, server)
        text, usage, _trace = _run(episode)
    main = [post for post in server.posts if post["tools"]]
    assert ANSWER in str(text), (text, server.posts)
    assert main[0]["refused"] and main[0]["tools"] > 3
    assert len(main) == 2 and main[1]["tools"] == 3 and not main[1].get("refused")
    assert episode.tools._ctx.active_context_mode == "nano"
