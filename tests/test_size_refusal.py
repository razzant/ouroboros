"""Size refusals of real servers, read as one typed fact (P1, 15.11.2026), and where its numbers may come from.

Each body in ``fixtures/size_refusals`` names its server, version and how it was obtained (``manifest.json``):
llama.cpp bodies captured on the stand, vLLM/Ollama/LM Studio built from their sources, OpenRouter's nested
upstream error from #3518, and negative controls. The exception is made by the real OpenAI SDK over an
``httpx.MockTransport`` and classified by the host's own classifier, as a Main call meets it.

Numbers belong to the recognised object of the current attempt. A cause the classifier types as another kind,
an output or body limit, an earlier upstream attempt and a sibling diagnostic state none.
"""

from __future__ import annotations

import io
import json
import pathlib
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import openai
import pytest

from ouroboros import capability_evidence as ce, size_refusal as sr
from ouroboros.context_anchor import refused_request_density
from ouroboros.context_window import record_refusal, resolve_window
from ouroboros.llm_attempt import body_error_facts
from ouroboros.loop_llm_call import classify_llm_exception
from ouroboros.loop_model_call import _refusal_target
from ouroboros.loop_transport import stamp_owner_provider_message
from ouroboros.window_learning import learn_window_from_refusal
from tests._size_refusal import _sdk_error


ROOT = pathlib.Path(__file__).parent / "fixtures" / "size_refusals"
CASES = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))




@pytest.mark.parametrize("case", CASES, ids=[c["file"].split("/")[-1].removesuffix(".body.json") for c in CASES])
def test_every_body_is_read_as_its_server_states_it(case):
    assert case["server"] and case["version"] and case["provenance"]  # every body says where it came from
    raw = (ROOT / case["file"]).read_bytes()
    exc = _sdk_error(case["status"], raw)
    found = sr.parse(case["status"], json.loads(raw), message=repr(exc))
    expected = case["expected_refusal"]
    if expected is None:
        assert found is None
    else:
        assert found is not None
        assert {key: getattr(found, key) for key in expected} == expected

    # The same body through the SDK and the host's classifier, as a Main call meets it.
    from ouroboros.loop_llm_call import classify_llm_exception

    exc = _sdk_error(case["status"], raw)
    assert classify_llm_exception(exc).kind == case["expected_class"]
    via_sdk = sr.from_exception(exc)
    assert (via_sdk is None) == (expected is None)
    if expected is not None:
        assert (via_sdk.window, via_sdk.requested, via_sdk.meaning) == (
            expected["window"], expected["requested"], expected["meaning"])


@pytest.mark.parametrize("case", [c for c in CASES if (c["expected_refusal"] or {}).get("window")
                                  and c["expected_refusal"]["units"] == "tokens"], ids=lambda c: c["file"][7:-10])
def test_the_numbers_reach_the_hosts_text_readers(case):
    """The readers recovery and the window record use (``_parse_ctx_limit_number``, ``_parse_ctx_requested``) see
    the same window and input in the sentence the usage carries, the gateway's upstream one included."""
    from ouroboros.capability_evidence import _parse_ctx_requested
    from ouroboros.context_budget import _parse_ctx_limit_number
    from ouroboros.loop_transport import owner_provider_message

    exc = _sdk_error(case["status"], (ROOT / case["file"]).read_bytes())
    usage = {"_last_llm_provider_message": owner_provider_message(exc)}
    sr.stamp(usage, exc)
    sentence = sr.refusal_sentence(usage)
    expected = case["expected_refusal"]
    assert _parse_ctx_limit_number(sentence) == expected["window"]
    if expected["requested"]:
        stated, reply, total = _parse_ctx_requested(sentence)
        assert (stated or total - reply) == expected["requested"]


@pytest.mark.parametrize("case", [c for c in CASES if c['recognition'] == 'number_equals_declared'],
                         ids=lambda c: c['file'])
def test_new_wording_with_and_without_declared_limit(case, tmp_path, monkeypatch):
    _assert_number_recognition(case, tmp_path, monkeypatch)


def _assert_number_recognition(case, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ouroboros import capability_evidence as ce, pricing
    from ouroboros.context_anchor import cited_limit_refusal
    from ouroboros.context_window import record_refusal
    from ouroboros.loop_llm_call import classify_llm_exception
    from ouroboros.loop_transport import stamp_owner_provider_message, provider_failure_hint

    monkeypatch.setattr(ce, 'canonical_evidence_root', lambda: tmp_path)
    catalog = {}
    monkeypatch.setattr(pricing, 'get_pricing', lambda **_: catalog)
    exc = _sdk_error(case['status'], (ROOT / case['file']).read_bytes())
    verdict = classify_llm_exception(exc)
    assert verdict.kind == case['expected_class'] and verdict.untyped
    usage = {}
    stamp_owner_provider_message(usage, exc, verdict)
    sentence = usage['_last_llm_provider_message']
    assert sentence in provider_failure_hint(usage)
    plan = SimpleNamespace(provider='openrouter', model='fixture', route_fp='r', window_record=None)
    capture = SimpleNamespace(physical_context=SimpleNamespace(route_fp='r'), max_completion_tokens=8192)
    measurement = SimpleNamespace(route_fp='r', reply_allowance_tokens=8192)
    assert not cited_limit_refusal(usage, capture, measurement, plan)
    key = 'max_prompt_tokens' if case['declared_kind'] == 'input' else 'context_length'
    catalog['endpoint'] = pricing.PricingSchedule((1., None, None, 2.), source={key: case['declared_limit']})
    assert cited_limit_refusal(usage, capture, measurement, plan)
    answer = usage['_last_llm_provider_answer']
    assert (answer['basis'], answer['limit'], answer['limit_kind']) == (
        'number_equals_declared', case['declared_limit'], case['declared_kind'])
    assert sentence in provider_failure_hint(usage)
    assert sr.for_usage(usage).token_window == 0
    record_refusal(usage, plan, refused_input=case['declared_limit'] + 1)
    assert not ce._read(tmp_path).get('main_windows')


def test_mlx_unrecognised_refusal_reaches_the_person():
    from ouroboros.loop_llm_call import classify_llm_exception
    from ouroboros.loop_transport import stamp_owner_provider_message, provider_terminal_fallback_text
    exc = _sdk_error(400, (ROOT / 'bodies/lmstudio-mlx.body.json').read_bytes())
    verdict = classify_llm_exception(exc)
    assert verdict.kind == 'bad_request' and sr.from_exception(exc) is None
    usage = {'_last_llm_error_kind': verdict.kind}
    stamp_owner_provider_message(usage, exc, verdict)
    text = provider_terminal_fallback_text(usage, is_context_overflow=False,
        is_transport_wait=False, waited_sec=0., is_deadline_exhausted=False)
    assert 'Context size has been exceeded' in text


def test_a_later_failure_clears_the_stamp():
    usage = {}
    sr.stamp(usage, _sdk_error(400, (ROOT / "bodies/stand-llama-b12421-first-contact.body.json").read_bytes()))
    assert sr.stamped(usage).window == 80_000
    sr.stamp(usage, _sdk_error(400, (ROOT / "bodies/ctl-llama-invalid-tokens.body.json").read_bytes()))
    assert sr.stamped(usage) is None and sr.USAGE_KEY not in usage
    sr.stamp(usage, ConnectionError("connection reset by peer"))  # a dropped connection states nothing
    assert sr.stamped(usage) is None


def test_numberless_zai_form_does_not_shadow_a_stated_limit():
    """"Prompt too long" carries no numbers, but a wording that continues to state a token limit
    ("Prompt too long; maximum context length is 80000 tokens") must keep it: the numberless form is
    anchored to the whole sentence, so the shared vocabulary — not the form — reads the numbers.
    Both bodies pass through the real SDK and the host classifier, as a Main call meets them."""
    import json as _json
    from ouroboros.loop_llm_call import classify_llm_exception

    verbatim = (ROOT / "bodies/zai-1261.body.json").read_bytes()
    exc = _sdk_error(400, verbatim)
    found = sr.from_exception(exc)
    assert found is not None and found.source == "text:zai/prompt-too-long" and not found.has_numbers
    assert classify_llm_exception(exc).kind == "context_overflow"

    combined_body = _json.dumps({"error": {"code": "1261",
                                "message": "Prompt too long; maximum context length is 80000 tokens"}}).encode()
    exc = _sdk_error(400, combined_body)
    found = sr.from_exception(exc)
    assert found is not None and found.source != "text:zai/prompt-too-long"
    assert found.token_window == 80_000 and found.meaning == "total"
    assert classify_llm_exception(exc).kind == "context_overflow"
    assert ce._parse_ctx_limit_number(sr.refusal_sentence({"_last_llm_provider_message": found.upstream})) == 80_000


ZAI = {"error": {"code": "1261", "message": "Prompt too long"}}
ZAI_LIMIT = {"error": {"code": "1261", "message": "Prompt too long; maximum context length is 80000 tokens"}}
ZAI_RAW, ZAI_LIMIT_RAW = json.dumps(ZAI), json.dumps(ZAI_LIMIT)


def _forwarded(raw: str, code="1214", message: str = "Invalid parameter") -> dict:
    """A gateway's own error carrying an upstream provider's document verbatim (OpenRouter's ``metadata.raw``)."""
    return {"error": {"code": code, "message": message, "metadata": {"provider_name": "Z.AI", "raw": raw}}}


def _sse_frame(raw: str) -> str:
    return "event: error\ndata: %s\n\ndata: [DONE]\n\n" % raw


@pytest.mark.parametrize(("status", "body", "kind"), [
    # The pair counts on one current object: the refusal itself, its numeric code, a bare JSON forward, an SSE frame.
    (400, ZAI, "context_overflow"),
    (400, {"error": {"code": 1261, "message": "Prompt too long"}}, "context_overflow"),
    (400, _forwarded(ZAI_RAW, code=400, message="Provider returned error"), "context_overflow"),
    (400, _forwarded(_sse_frame(ZAI_RAW)), "context_overflow"),
    # ... and never on a document a message cites, the same frame quoted in prose, a neighbouring string, a
    # ``type`` field or an earlier attempt; no other status is read for size whatever its words say.
    (400, {"error": {"code": "1214", "message": "Invalid parameter; previous response: " + ZAI_RAW}}, "bad_request"),
    (400, {"error": {"code": "1214", "message": "Invalid parameter; previous response:\n" + _sse_frame(ZAI_RAW)}},
     "bad_request"),
    (400, {"error": {"code": "1261", "message": "see " + ZAI_RAW}}, "bad_request"),
    (400, {"error": {"code": "1261", "message": "Invalid parameter", "detail": "Prompt too long"}}, "bad_request"),
    (400, {"error": {"type": "1261", "message": "Prompt too long"}}, "bad_request"),
    (400, {"error": {"code": "1214", "message": "Invalid parameter",
                     "metadata": {"previous_errors": [ZAI["error"]]}}}, "bad_request"),
    (400, {"error": {"code": "1214", "message": "Parameter `messages[0].content.str` is invalid. "
                                                "Please check the documentation."}}, "bad_request"),
    (401, ZAI, "auth_error"),
    # A previous answer the parameter refusal cites states a window of its own (the observed Z.ai 1214 form):
    # the number is never this attempt's. The same document forwarded bare — the wire's own body, SSE included —
    # still speaks, and so does the numberless pair.
    (400, {"error": {"code": "1214", "message": "Invalid parameter; previous response: " + ZAI_LIMIT_RAW}},
     "bad_request"),
    (400, {"error": {"code": "1214", "message": "Invalid parameter; previous response:\n" + _sse_frame(ZAI_LIMIT_RAW)}},
     "bad_request"),
    (400, _forwarded("previous response: " + ZAI_LIMIT_RAW), "bad_request"),
    (400, _forwarded(ZAI_LIMIT_RAW, code=400, message="Provider returned error"), "context_overflow"),
    (400, _forwarded(_sse_frame(ZAI_LIMIT_RAW)), "context_overflow"),
])
def test_the_zai_1261_pair_counts_on_one_current_object(status, body, kind):
    """#3858: "Prompt too long" types nothing alone, and its code completes a refusal only beside its own whole
    ``message`` on one current, uncited object — a gateway forward included, a citation never."""
    assert classify_llm_exception(_sdk_error(status, json.dumps(body).encode())).kind == kind


@pytest.mark.parametrize(("name", "body"), [
    ("cited-prose-json-80k",
     {"error": {"code": "1214", "message": "Invalid parameter; previous response: " + ZAI_LIMIT_RAW}}),
    ("cited-sse-80k",
     {"error": {"code": "1214", "message": "Invalid parameter; previous response:\n" + _sse_frame(ZAI_LIMIT_RAW)}}),
    ("cited-prose-json-80k-forwarded", _forwarded("previous response: " + ZAI_LIMIT_RAW)),
    ("documented-1214",
     {"error": {"code": "1214", "message": "Parameter `messages[0].content.str` is invalid. "
                                           "Please check the documentation."}}),
])
def test_a_previous_answer_a_parameter_refusal_cites_teaches_no_window(tmp_path, monkeypatch, name, body):
    """#3858: Z.ai's 1214 names a malformed request in its own words, and the answer its ``previous response:``
    carries states a window of that other answer's — so this attempt is no size refusal at any measured size, and
    the number is not taken even when the route declares that very limit (``context_anchor.cited_limit_refusal``,
    the secondary step an untyped client refusal reaches). Nothing is stamped as a size fact, so the early rebind
    (``window_learning``) has nothing to record either."""
    from ouroboros.context_anchor import cited_limit_refusal
    from ouroboros.context_window import WindowLimit, WindowRecord

    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    exc = _sdk_error(400, json.dumps(body).encode())
    verdict = classify_llm_exception(exc)
    assert verdict.kind == "bad_request" and verdict.untyped is False, name
    usage = {}
    sr.stamp_failure(usage, exc, verdict)
    fact = sr.for_usage(usage)
    assert not fact.has_numbers and fact.token_window == 0, name
    assert sr.stamped(usage) is None and "size" not in (usage.get("_last_llm_provider_answer") or {}), name
    plan = _plan()
    capture = SimpleNamespace(physical_context=SimpleNamespace(route_fp="r"), max_completion_tokens=8192)
    measurement = SimpleNamespace(route_fp="r", reply_allowance_tokens=8192)
    assert not cited_limit_refusal(usage, capture, measurement, plan), name
    plan.window_record = WindowRecord("r", declared=(WindowLimit(80_000, "total", "server_metadata"),))
    assert not cited_limit_refusal(usage, capture, measurement, plan), name
    record_refusal(usage, plan)  # the early rebind's own call shape (``learn_window_from_refusal``)
    assert not ce._read(tmp_path).get("main_windows"), name


@pytest.mark.parametrize(("name", "error", "event_type"), [
    ("cited-prose-json-80k",
     {"code": "1214", "message": "Invalid parameter; previous response: " + ZAI_LIMIT_RAW}, "provider_body_error"),
    ("documented-1214",
     {"code": "1214", "message": "Parameter `messages[0].content.str` is invalid. "
                                 "Please check the documentation."}, "provider_body_error"),
    ("bare-raw-own-limit",
     {"code": "1214", "message": "Invalid parameter",
      "metadata": {"provider_name": "Z.AI", "raw": ZAI_LIMIT_RAW}}, "remote_context_overflow"),
    ("direct-numberless", {"code": "1261", "message": "Prompt too long"}, "remote_context_overflow"),
])
def test_an_http200_parameter_refusal_is_no_context_overflow(name, error, event_type):
    """An HTTP 200 body's own error reaches the loop as ``usage['provider_error']`` (``body_error_facts``), and the
    second point (``loop_llm_call._classify_empty_response``) decides on the same recognition: a window only the
    cited previous answer states makes no overflow, a document forwarded bare still does."""
    from ouroboros.llm_attempt import is_size_refusal
    from ouroboros.loop_llm_call import _classify_empty_response

    facts = body_error_facts(error, False)
    assert _classify_empty_response({"provider_error": facts}, {})[0] == event_type, name
    assert is_size_refusal(facts) == (event_type == "remote_context_overflow"), name
    if event_type == "provider_body_error":
        assert not facts["answer"].get("untyped"), name  # the typed cause leaves no untyped refusal behind


@pytest.mark.parametrize(("name", "body", "window"), [
    ("direct", ZAI, 0),
    ("bare-raw", _forwarded(ZAI_RAW, code=400, message="Provider returned error"), 0),
    ("sse", _forwarded(_sse_frame(ZAI_RAW)), 0),
    ("child-diagnostic", {"error": {"code": "1261", "message": "Prompt too long",
                                    "diagnostic": {"type": sr.LLAMA_CPP_TYPE, "n_ctx": 4000,
                                                   "n_prompt_tokens": 90_000}}}, 0),
    ("bare-raw-own-limit", _forwarded(ZAI_LIMIT_RAW), 80_000),
])
def test_a_zai_1261_pair_keeps_only_its_own_stated_limit(tmp_path, monkeypatch, name, body, window):
    """The pair is numberless by its form: no child diagnostic, citation or earlier attempt states its window, so the
    numberless refusal is retry-only and records no observation at any measured size. The same message stating its
    own limit keeps that limit, and it is recorded."""
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    exc = _sdk_error(400, json.dumps(body).encode())
    verdict = classify_llm_exception(exc)
    assert verdict.kind == "context_overflow" and not verdict.untyped, name
    usage = {}
    sr.stamp_failure(usage, exc, verdict)
    fact = sr.for_usage(usage)
    assert fact.token_window == window and (fact.has_numbers is bool(window)), name
    record_refusal(usage, _plan(), refused_input=90_000)
    rows = (ce._read(tmp_path).get("main_windows", {}).get("r") or {}).get("observations", [])
    assert [row.get("refused") for row in rows] == ([80_000] if window else []), name


#: The fourth review's distinguishing scenarios (#3858): the decision about the current refusal must be the same
#: on the SDK path, the HTTP 200 error body, the owner display and the stored window — never re-read from a quote.
ZAI_SHORT = "Prompt too long"
ZAI_CAP = "maximum context length is 80000 tokens"
ZAI_OLD_CAP = "maximum context length is 4000 tokens"
ZAI_INVALID = "Parameter `messages[0].content.str` is invalid. Please check the documentation."
ZAI_OTHER = "Invalid parameter"


def _zai_wire(code, message, **other):
    return json.dumps({"error": {"code": code, "message": message, **other}})


def _zai_frames(document):
    return 'data: {"id":"chatcmpl-fake","choices":[]}\n\nevent: error\ndata: ' + document + '\n\ndata: [DONE]\n\n'


@pytest.mark.parametrize(("name", "error", "overflow", "limit", "observed"), [
    # A plain-text citation after "previous response:" is no party of the 1214 answer, words or numbers.
    ("plain-reference", {"code": "1214", "message": f"{ZAI_OTHER}; previous response: {ZAI_SHORT}; {ZAI_CAP}"},
     False, 0, []),
    ("multiframe-citation", {"code": "1214", "message": ZAI_INVALID + " previous response:\n" + _zai_frames(
        _zai_wire("1261", ZAI_SHORT + "; " + ZAI_CAP))}, False, 0, []),
    ("typed-citation", {"code": "1214", "message": ZAI_INVALID + " previous response: "
                         + _zai_wire("context_length_exceeded", ZAI_CAP)}, False, 0, []),
    # The same words an earlier attempt spoke change nothing here.
    ("historic-speaker-same-words", {"code": "1214", "message": ZAI_INVALID + " previous response: "
     + _zai_wire("1261", ZAI_SHORT + "; " + ZAI_CAP, provider_name="Named"),
     "metadata": {"provider_name": "Z.AI", "previous_errors": [
         {"code": "1261", "provider_name": "Named", "message": ZAI_SHORT + "; " + ZAI_CAP}]}}, False, 0, []),
    # Both observed forms of the 1214 field, backticked or not.
    ("unbackticked", {"code": "1214", "message": ZAI_INVALID.replace("`", "") + " previous response: "
     + _zai_wire("1261", ZAI_SHORT + "; " + ZAI_CAP)}, False, 0, []),
    # A gateway's normalized negative wrapper forwarding the same cited 1214 document.
    ("gateway-normalized-negative", {"code": 400, "message": "Provider returned error", "metadata": {
        "provider_name": "Z.AI", "raw": _zai_wire("1214", ZAI_INVALID + " previous response: "
                                                  + _zai_wire("1261", ZAI_SHORT + "; " + ZAI_CAP))}}, False, 0, []),
    # The wire's own bare forward (raw JSON, SSE included) still speaks, and beats the quoted other number.
    ("true-bare-forward-plus-quoted-other", {"code": "1214",
     "message": f"{ZAI_OTHER}; previous response: {ZAI_SHORT}; {ZAI_OLD_CAP}",
     "metadata": {"provider_name": "Z.AI", "raw": _zai_wire("1261", ZAI_SHORT + "; " + ZAI_CAP)}}, True, 80_000, [80_000]),
    ("true-current-typed-no-known-phrase", {"code": "1214", "message": ZAI_INVALID, "metadata": {
        "provider_name": "Z.AI", "raw": _zai_wire("context_length_exceeded", "Input range exceeded")}},
     True, 0, [90_000]),
    ("true-current-beats-history", {"code": 400, "message": "Provider returned error", "metadata": {
        "provider_name": "Z.AI", "raw": _zai_frames(_zai_wire("1261", ZAI_SHORT + "; " + ZAI_CAP)),
        "previous_errors": [{"type": "context_length_exceeded", "provider_name": "Earlier",
                             "message": ZAI_OLD_CAP}]}}, True, 80_000, [80_000]),
    # The 1261 pair reads a limit from its own normalized message or its own detail — never an inline diagnostic.
    ("zai-short-plus-inline-diagnostic", {"code": "1261", "message": ZAI_SHORT + " " + json.dumps(
        {"note": "Payload too large; " + ZAI_OLD_CAP})}, True, 0, []),
    ("zai-explicit-plus-inline-diagnostic", {"code": "1261", "message": ZAI_SHORT + "; " + ZAI_CAP + " "
     + json.dumps({"note": "Payload too large; " + ZAI_OLD_CAP})}, True, 80_000, [80_000]),
    ("zai-short-message-real-detail", {"code": "1261", "message": ZAI_SHORT, "detail": ZAI_CAP}, True, 80_000, [80_000]),
    # Retry-only by the actual pair: an OpenAI-compatible typed code with the same words keeps the old anchor.
    ("ordinary-openai-typed-anchor", {"code": "context_length_exceeded", "message": ZAI_SHORT}, True, 0, [90_000]),
    ("ordinary-typed-citing-1261", {"code": "context_length_exceeded", "message": ZAI_SHORT,
     "detail": "previous response: " + _zai_wire("1261", ZAI_SHORT + "; " + ZAI_OLD_CAP)}, True, 0, [90_000]),
    ("forwarded-sse-citing-zai", {"code": 400, "message": "Provider returned error", "metadata": {
        "provider_name": "Z.AI", "raw": _zai_frames(_zai_wire(
            "1214", ZAI_INVALID + " previous response: " + _zai_wire("1261", ZAI_SHORT + "; " + ZAI_CAP)))}},
     False, 0, []),
    ("genuine-sse-with-diagnostic", {"code": 400, "message": "Provider returned error", "metadata": {
        "provider_name": "Z.AI", "raw": _zai_frames(_zai_wire("1261", ZAI_SHORT, detail="previous diagnostic: none"))}},
     True, 0, []),
    ("cloudflare-vllm-envelope", {"code": 400, "message": "Provider returned error", "metadata": {
        "provider_name": "Cloudflare", "raw": json.dumps({"errors": [{"code": 8007, "message": "AiError: AiError: "
            + (ROOT / "bodies/vllm-0.11.2-input.body.json").read_text(encoding="utf-8").strip() + " (test-only)"}],
            "success": False})}}, True, 80_000, [80_000]),
    ("zai-requested-count-only", {"code": "1261", "message": ZAI_SHORT, "type": sr.LLAMA_CPP_TYPE,
     "n_prompt_tokens": 91_000}, True, 0, []),
    ("typed-forward-keeps-own-limit", {"code": "1214", "message": ZAI_INVALID + " previous response: "
     + _zai_wire("context_length_exceeded", ZAI_OLD_CAP, provider_name="Quoted"), "metadata": {
         "provider_name": "Z.AI", "raw": _zai_wire("context_length_exceeded", ZAI_CAP, provider_name="Current")}},
     True, 80_000, [80_000]),
    ("parameter-citing-body-limit", {"code": "1214",
     "message": ZAI_INVALID + " previous response: Request body too large"}, False, 0, []),
    ("explicit-zai-versus-child-count", {"code": "1261", "message": ZAI_SHORT + "; " + ZAI_CAP,
     "diagnostic": {"type": sr.LLAMA_CPP_TYPE, "n_ctx": 4_000, "n_prompt_tokens": 91_000}}, True, 80_000, [80_000]),
])
@pytest.mark.parametrize("surface", ["sdk-400", "http200"])
def test_r4_current_decision_is_one_on_every_surface(tmp_path, monkeypatch, surface, name, error, overflow, limit, observed):
    from ouroboros.context_anchor import cited_limit_refusal
    from ouroboros.context_window import WindowLimit, WindowRecord
    from ouroboros.loop_llm_call import _classify_empty_response

    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    usage = {}
    if surface == "sdk-400":
        exc = _sdk_error(400, json.dumps({"error": error}).encode())
        verdict = classify_llm_exception(exc)
        event, expected = verdict.kind, ("context_overflow" if overflow else "bad_request")
        sr.stamp_failure(usage, exc, verdict)
    else:
        facts = body_error_facts(error, False)
        event = _classify_empty_response({"provider_error": facts}, {})[0]
        expected = "remote_context_overflow" if overflow else "provider_body_error"
        stamp_owner_provider_message(usage, facts)
    assert event == expected, name
    fact = sr.for_usage(usage)
    assert fact.token_window == limit, name
    plan = _plan()
    plan.window_record = WindowRecord("r", declared=(WindowLimit(80_000, "total", "server_metadata"),
                                                     WindowLimit(4_000, "total", "server_metadata")))
    capture = SimpleNamespace(physical_context=SimpleNamespace(route_fp="r"), max_completion_tokens=8192)
    measurement = SimpleNamespace(route_fp="r", reply_allowance_tokens=8192)
    assert not (not overflow and cited_limit_refusal(usage, capture, measurement, plan)), name
    record_refusal(usage, plan, refused_input=90_000 if overflow else 0)
    rows = (ce._read(tmp_path).get("main_windows", {}).get("r") or {}).get("observations", [])
    assert [row.get("refused") for row in rows] == observed, name


def test_a_dropped_connection_is_no_size_refusal():
    from ouroboros.loop_llm_call import classify_llm_exception

    def drop(request):
        raise httpx.ReadError("connection reset", request=request)

    client = openai.OpenAI(api_key="x", base_url="http://127.0.0.1:9/v1", max_retries=0,
                           http_client=httpx.Client(transport=httpx.MockTransport(drop)))
    with pytest.raises(openai.APIConnectionError) as caught:
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
    assert sr.from_exception(caught.value) is None
    assert classify_llm_exception(caught.value).kind != "context_overflow"


SIZE = "request (90068 tokens) exceeds the available context size (80128 tokens), try increasing it"
COMBINED = "input length and `max_tokens` exceed context limit: 75000 + 8192 > 80000"


#: Recognised by the frozen wording alone, without a typed size code.
FROZEN = ("This model's maximum context length is 80000 tokens. However, you requested 98192 tokens "
          "(90000 in the messages, 8192 in the completion).")


@pytest.mark.parametrize("code", ["invalid_api_key", "insufficient_quota", "rate_limit_exceeded", "authentication_error",
                                  "quota_exceeded", "invalid_schema", "content_policy_violation", "payload_too_large"])
@pytest.mark.parametrize("wrapping", ["direct", "raw", "twice", "outer"])
def test_negative_typed_cause_survives_every_wrapper(tmp_path, monkeypatch, code, wrapping):
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    error = {"code": code, "message": FROZEN}
    if wrapping == "outer":
        error["metadata"] = {"raw": json.dumps({"error": {"type": sr.LLAMA_CPP_TYPE,
                              "n_ctx": 80128, "n_prompt_tokens": 90068, "message": SIZE}})}
    else:
        for _ in range({"direct": 0, "raw": 1, "twice": 2}[wrapping]):
            error = {"code": 400, "message": "Provider returned error",
                     "metadata": {"raw": json.dumps({"error": error}), "provider_name": "fixture"}}
    exc = _sdk_error(400, json.dumps({"error": error}).encode())
    usage = {}
    stamp_owner_provider_message(usage, exc, classify_llm_exception(exc))
    sr.stamp(usage, exc)
    # A non-context cause must not even mint numbers another recovery entry could store.
    assert not (sr.stamped(usage) or sr.SizeRefusal()).has_numbers and sr.for_usage(usage).token_window == 0
    record_refusal(usage, _plan())
    assert not ce._read(tmp_path).get("main_windows")


@pytest.mark.parametrize("status,text", [(413, "Request body too large. " + SIZE),
    (400, "Request body too large. " + COMBINED), (400, "max_tokens: " + SIZE),
    (400, "input length and `max_tokens` exceed context limit: 75000 + 90000 > 80000")])
def test_body_and_output_limits_outrank_size_quotations(status, text):
    exc = _sdk_error(status, json.dumps({"error": {"message": text}}).encode())
    assert sr.from_exception(exc) is None


@pytest.mark.parametrize("text", [COMBINED,
    "'max_tokens' or 'max_completion_tokens' is too large: 8192. This model's maximum context length is "
    "80000 tokens and your request has 75000 input tokens (8192 > 80000 - 75000)."])
def test_input_plus_reply_remains_a_context_refusal(text):
    exc = _sdk_error(400, json.dumps({"error": {"code": 400, "message": text}}).encode())
    assert classify_llm_exception(exc).kind == "context_overflow"
    found = sr.from_exception(exc)
    assert (found.window, found.requested, found.reply, found.meaning) == (80000, 75000, 8192, "input_plus_reply")


@pytest.mark.parametrize("message", [
    "request (120000 tokens) exceeds the available context size (200000 tokens)",
    "context exhausted", "",
])
def test_structured_counts_survive_learning_storage_and_recovery(tmp_path, monkeypatch, message):
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    body = {"error": {"type": sr.LLAMA_CPP_TYPE, "n_ctx": 80000, "n_prompt_tokens": 90000, "message": message}}
    exc = _sdk_error(400, json.dumps(body).encode())
    usage = {"_last_llm_provider_message": message}
    sr.stamp(usage, exc)
    plan = SimpleNamespace(route_fp="refusal-fixture", provider="openai-compatible", model="fixture",
                           window_record=None, window_witness_bound=True)
    ctx = SimpleNamespace(round_idx=1, accumulated_usage=usage, context_fit_plan=plan, active_use_local=False,
                          active_context_mode="max", active_model="fixture", messages=[], tools=None,
                          tool_schemas=[], model_role="main")
    loop = SimpleNamespace(_rebind_context_fit_plan=lambda *args, **kwargs: (plan, "max"))
    learn_window_from_refusal(ctx, loop)
    evidence = ce.CapabilityEvidence(0, "unprobeable", "none", "refusal-fixture")
    assert resolve_window(evidence, 8192, root=tmp_path).total.tokens == 80000
    before = ce._store_path(tmp_path).read_bytes()
    record_refusal(usage, plan)
    assert ce._store_path(tmp_path).read_bytes() == before
    assert resolve_window(evidence, 8192, root=tmp_path).total.tokens == 80000
    measurement = SimpleNamespace(estimated_input_tokens=45000, anchor_input_tokens=None,
        accepted_input_tokens=None, capacity_total_tokens=None, window_reading_tokens=None,
        reply_allowance_tokens=8192, probes_reply_reservation=False, measurement_density=1.0,
        route_fp="refusal-fixture")
    refused = SimpleNamespace(prompt_tokens_bounded_estimate=45000,
                              physical_context=SimpleNamespace(route_fp="refusal-fixture"))
    assert refused_request_density(usage, refused, measurement) == 2.0
    assert _refusal_target(ctx, SimpleNamespace(measurement=measurement), refused) == (28192, 10000)
    final = resolve_window(evidence, 8192, root=tmp_path)
    assert final.total.tokens == 80000 and final.admitted == 71808
    assert final.total.basis == "provider_stated"
    from ouroboros.context_view_facts import remember_size_refusal, SIZE_REFUSAL_KEY
    # A misleading sentence cannot replace the structured numbers in the round slot.
    remember_size_refusal(ctx, None, measured=False)
    assert usage[SIZE_REFUSAL_KEY]["tokens"] == 90000
    assert usage[SIZE_REFUSAL_KEY]["basis"] == "provider_stated"
    assert usage[SIZE_REFUSAL_KEY]["round"] == 1


@pytest.mark.parametrize("fields,expected", [
    ({"n_ctx": 80000}, (80000, 90068)), ({"n_prompt_tokens": 90000}, (80128, 90000)),
])
def test_text_supplies_only_missing_structured_counts(fields, expected):
    exc = _sdk_error(400, json.dumps({"error": {"type": sr.LLAMA_CPP_TYPE, "message": SIZE, **fields}}).encode())
    found = sr.from_exception(exc)
    assert (found.window, found.requested) == expected


def test_non_size_classes_match_the_recorded_baseline():
    from pathlib import Path
    table = json.loads((Path(__file__).parent / 'fixtures/size_refusals/non_size_baseline.json').read_text())
    mismatches = []
    for row in table['cases']:
        exc = _sdk_error(row['status'], json.dumps(row['body']).encode())
        if row['kind'] != 'context_overflow':
            assert sr.from_exception(exc) is None, row['id']
        actual = classify_llm_exception(exc).kind
        if actual != row['kind']:
            mismatches.append((row['id'], actual, row['kind']))
    assert not mismatches


@pytest.mark.parametrize("provider,learns", [("openrouter", False), ("openai", False),
                                           ("openai-compatible", True), ("local", True)])
def test_smaller_refusal_is_server_restart_evidence_only_for_single_server(tmp_path, monkeypatch, provider, learns):
    from ouroboros.context_window import record_observation
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    record_observation(tmp_path, "route", accepted=70000)
    usage = {}
    body = {"error": {"type": sr.LLAMA_CPP_TYPE, "n_ctx": 40000, "n_prompt_tokens": 60000}}
    sr.stamp(usage, _sdk_error(400, json.dumps(body).encode()))
    plan = SimpleNamespace(route_fp="route", provider=provider, window_record=None)
    record_refusal(usage, plan)
    rows = ce._read(tmp_path)["main_windows"]["route"]["observations"]
    assert any(row.get("refused") == 40000 for row in rows) == learns


def test_local_wrapper_keeps_structured_counts_from_server_cause():
    from ouroboros.context_budget import LocalContextTooLargeError
    cause = _sdk_error(500, json.dumps({"error": {"type": sr.LLAMA_CPP_TYPE,
                       "message": "context exhausted", "n_ctx": 40000, "n_prompt_tokens": 60000}}).encode())
    wrapper = LocalContextTooLargeError("local server refused")
    wrapper.__cause__ = cause
    found = sr.from_exception(wrapper)
    assert (found.window, found.requested) == (40000, 60000)


def test_http200_error_keeps_structured_counts_for_round_and_learning():
    from ouroboros.llm_attempt import body_error_facts
    from ouroboros.loop_transport import stamp_owner_provider_message
    error = {"code": 400, "type": sr.LLAMA_CPP_TYPE, "message": "context exhausted",
             "n_ctx": 40000, "n_prompt_tokens": 60000}
    usage = {}
    stamp_owner_provider_message(usage, body_error_facts(error, False))
    found = sr.for_usage(usage)
    assert (found.window, found.requested) == (40000, 60000)


@pytest.mark.parametrize("name,limit", [
    ("vllm-919d-renderer", 80000), ("vllm-919d-renderer-atleast", 80000),
    ("vllm-919d-decoder", 80000), ("vllm-919d-decoder-equal", 80000),
    ("issue-3518-openrouter-alibaba", 983616), ("alibaba-direct-compat", 983616),
    ("lmstudio-mlx", 80000),
])
def test_new_wording_needs_upstream_declaration_match(tmp_path, monkeypatch, name, limit):
    from pathlib import Path
    from ouroboros import pricing
    from ouroboros.context_anchor import cited_limit_refusal
    from ouroboros.context_window import WindowLimit, WindowRecord
    from ouroboros.loop_transport import stamp_owner_provider_message
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    monkeypatch.setattr(pricing, "declared_token_limits", lambda *args: {})
    raw = (Path(__file__).parent / "fixtures/size_refusals/bodies" / (name + ".body.json")).read_bytes()
    exc = _sdk_error(400, raw)
    verdict = classify_llm_exception(exc)
    assert verdict.kind != "context_overflow" and verdict.untyped
    usage = {}
    stamp_owner_provider_message(usage, exc, verdict)
    plan = SimpleNamespace(route_fp="r", provider="openai-compatible", model="fixture", window_record=None)
    capture = SimpleNamespace(physical_context=SimpleNamespace(route_fp="r"), max_completion_tokens=8192)
    measurement = SimpleNamespace(route_fp="r", reply_allowance_tokens=8192)
    assert not cited_limit_refusal(usage, capture, measurement, plan)
    plan.window_record = WindowRecord("r", declared=(WindowLimit(limit, "total", "server_metadata"),))
    assert cited_limit_refusal(usage, capture, measurement, plan) == (name != "lmstudio-mlx")
    if name != "lmstudio-mlx":
        assert usage["_last_llm_provider_answer"]["limit"] == limit
        record_refusal(usage, plan, refused_input=limit + 1)
        assert not ce._read(tmp_path).get("main_windows")  # number equality teaches no ceiling


@pytest.mark.parametrize("structured", [False, True])
def test_early_rebind_requires_this_attempts_normalized_size_fact(monkeypatch, structured):
    """Legacy words still size recovery passes; only a fresh numeric receipt starts early learning."""
    from ouroboros import context_window
    from ouroboros.llm_attempt import body_error_facts
    from ouroboros.loop_transport import stamp_owner_provider_message
    usage = {}
    error = {"message": "maximum context length is 40000 tokens"}
    if structured:
        error.update(type=sr.LLAMA_CPP_TYPE, n_ctx=40000, n_prompt_tokens=60000)
    stamp_owner_provider_message(usage, body_error_facts(error, False) if structured else error)
    plan = SimpleNamespace(provider="openai-compatible", model="fixture", route_fp="r", window_record=None)
    ctx = SimpleNamespace(accumulated_usage=usage, context_fit_plan=plan, active_use_local=False,
                          active_model="fixture", active_context_mode="low", messages=["shown"],
                          tools=object(), tool_schemas=[], model_role="main")
    calls = []
    monkeypatch.setattr(context_window, "record_refusal", lambda *args: None)
    def rebind(*args, **kwargs):
        calls.append(1)
        return plan, "low"
    learn_window_from_refusal(ctx, SimpleNamespace(_rebind_context_fit_plan=rebind))
    assert bool(calls) is structured


@pytest.mark.parametrize('owner', [262000, 300000])
def test_server_read_is_a_declaration_even_when_settings_is_smaller(tmp_path, monkeypatch, owner):
    from ouroboros import context_window, pricing
    from ouroboros.context_anchor import cited_limit_refusal
    from ouroboros.loop_transport import stamp_owner_provider_message
    from pathlib import Path
    monkeypatch.setattr(ce, 'canonical_evidence_root', lambda: tmp_path)
    monkeypatch.setattr(pricing, 'declared_token_limits', lambda *_: {})
    class Response:
        status_code = 200
        headers = {}
        def raise_for_status(self):
            pass
        def json(self):
            return {'data': [{'id': 'fixture', 'max_model_len': 262144}]}
    monkeypatch.setattr('httpx.get', lambda *_, **__: Response())
    evidence = ce.probe(tmp_path, provider='openai-compatible', model='fixture',
        base_url='http://127.0.0.1:9/v1', api_key='', force=True)
    evidence = ce.with_role_window(evidence, owner)
    record = context_window.resolve_window(evidence, 8192, root=tmp_path)
    assert record.total.tokens == min(owner, 262144)
    assert record.contradicted_by.tokens == 262144
    raw = (Path(__file__).parent / 'fixtures/size_refusals/bodies/vllm-919d-renderer.body.json').read_bytes()
    exc = _sdk_error(400, raw.replace(b'80000', b'262144').replace(b'90000', b'280000'))
    usage = {}
    stamp_owner_provider_message(usage, exc, classify_llm_exception(exc))
    plan = SimpleNamespace(provider='openai-compatible', model='fixture', window_record=record)
    capture = SimpleNamespace(physical_context=SimpleNamespace(route_fp='wire'), max_completion_tokens=8192)
    measurement = SimpleNamespace(route_fp='wire', reply_allowance_tokens=8192)
    before = ce._read(tmp_path)
    assert cited_limit_refusal(usage, capture, measurement, plan)
    assert usage['_last_llm_provider_answer']['limit'] == 262144
    assert 'server_metadata' in usage['_last_llm_provider_answer']['declaration']
    record_refusal(usage, plan, refused_input=280000)
    assert ce._read(tmp_path) == before


@pytest.mark.parametrize("standing,history", [(False, False), (False, True), (True, False)])
def test_window_learning_uses_the_actors_nano_schema_set(tmp_path, monkeypatch, standing, history):
    from ouroboros.tool_policy import initial_tool_schemas
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx._nano_standing = standing
    expected = initial_tool_schemas(registry, "nano", ["read_file"] if history else [], standing=standing)
    if not standing and not history:
        assert {s["function"]["name"] for s in expected} == {
            "compact_context", "enable_tools", "list_available_tools"}
    usage = {}
    sr.stamp(usage, _sdk_error(400, json.dumps({"error": {
        "type": sr.LLAMA_CPP_TYPE, "n_ctx": 80000, "n_prompt_tokens": 90000}}).encode()))
    plan = SimpleNamespace(route_fp="call-fixture", provider="openai-compatible", model="fixture",
                           window_record=None, window_witness_bound=True)
    messages = [{"role": "system", "content": "Low view"}]
    if history:
        messages += [{"role": "assistant", "content": None, "tool_calls": [{"id": "r1", "type": "function",
                      "function": {"name": "read_file", "arguments": "{}"}}]}]
    ctx = SimpleNamespace(accumulated_usage=usage, context_fit_plan=plan, active_use_local=False,
                          active_context_mode="low", active_model="fixture", messages=messages, tools=registry,
                          tool_schemas=initial_tool_schemas(registry, "low"), model_role="main", task_id="call1")

    def rebind(*args, **kwargs):
        messages[0] = {"role": "system", "content": "Nano view"}
        return plan, "nano"

    assert learn_window_from_refusal(ctx, SimpleNamespace(_rebind_context_fit_plan=rebind))
    assert ctx.tool_schemas == expected  # same canonical bytes as a Nano start, including history's tools


def test_a_tool_loaded_but_not_yet_called_survives_the_rebind_into_nano(tmp_path, monkeypatch):
    """Review 8 (Fable): enable_tools loaded a tool, a size refusal rebinds the request into Nano before it is called;
    the loaded tool's schema is still sent."""
    from ouroboros.tool_policy import initial_tool_schemas
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx._actor_loaded_tool_names = {"query_code"}  # what enable_tools(["query_code"]) leaves
    usage = {}
    sr.stamp(usage, _sdk_error(400, json.dumps({"error": {
        "type": sr.LLAMA_CPP_TYPE, "n_ctx": 80000, "n_prompt_tokens": 90000}}).encode()))
    plan = SimpleNamespace(route_fp="loaded-fixture", provider="openai-compatible", model="fixture",
                           window_record=None, window_witness_bound=True)
    messages = [{"role": "system", "content": "Low view"}]
    ctx = SimpleNamespace(accumulated_usage=usage, context_fit_plan=plan, active_use_local=False,
                          active_context_mode="low", active_model="fixture", messages=messages, tools=registry,
                          tool_schemas=initial_tool_schemas(registry, "low"), model_role="main", task_id="loaded1")

    def rebind(*args, **kwargs):
        messages[0] = {"role": "system", "content": "Nano view"}
        return plan, "nano"

    assert learn_window_from_refusal(ctx, SimpleNamespace(_rebind_context_fit_plan=rebind))
    names = [schema["function"]["name"] for schema in ctx.tool_schemas]
    assert "query_code" in names and "finish_task" in names


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
    from ouroboros.tool_policy import NANO_RESIDENT_SCHEMA_NAMES
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
    sent = [body for kind, body in server.calls if body.get("tools") and kind != "safety"]
    names = {tool["function"]["name"] for tool in sent[-1]["tools"]}
    assert ANSWER in str(text), (text, server.posts)
    assert main[0]["refused"] and main[0]["tools"] > len(NANO_RESIDENT_SCHEMA_NAMES)
    assert len(main) == 2 and not main[1].get("refused")
    assert "finish_task" in names and names <= NANO_RESIDENT_SCHEMA_NAMES and len(names) == main[1]["tools"]
    assert episode.tools._ctx.active_context_mode == "nano"


def _plan(route="r"):
    return SimpleNamespace(route_fp=route, provider="openai-compatible", model="m", window_record=None,
                           window_witness_bound=True)


def test_an_earlier_attempts_rate_limit_leaves_the_current_refusals_numbers():
    current = ("This endpoint's maximum context length is 80000 tokens. However, you requested about 98192 tokens "
               "(90000 of text input, 8192 in the output).")
    body = {"error": {"code": "context_length_exceeded", "message": current, "metadata": {
        "previous_errors": [{"code": 429, "message": "Rate limit exceeded upstream", "provider_name": "Earlier"}]}}}
    exc = _sdk_error(400, json.dumps(body).encode())
    verdict = classify_llm_exception(exc)
    usage = {}
    stamp_owner_provider_message(usage, exc, verdict)
    sr.stamp(usage, exc)
    assert verdict.kind == "context_overflow"
    assert (sr.stamped(usage).window, sr.stamped(usage).requested) == (80000, 90000)


@pytest.mark.parametrize("via", ["exception", "http200"])
@pytest.mark.parametrize("status,text", [
    (413, "Request body too large. " + SIZE),
    (400, "max_tokens: " + SIZE),
])
def test_an_excluded_size_quotation_teaches_no_window(tmp_path, monkeypatch, status, text, via):
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    usage = {}
    if via == "http200":
        stamp_owner_provider_message(usage, body_error_facts({"code": status, "message": text}, False))
    else:
        exc = _sdk_error(status, json.dumps({"error": {"message": text}}).encode())
        stamp_owner_provider_message(usage, exc, classify_llm_exception(exc))
        sr.stamp(usage, exc)
        assert sr.from_exception(exc) is None
    assert sr.for_usage(usage).token_window == 0
    record_refusal(usage, _plan())
    rows = (ce._read(tmp_path).get("main_windows", {}).get("r") or {}).get("observations", [])
    assert not any(row.get("refused") for row in rows)


def test_an_earlier_attempts_sentence_states_no_numbers(tmp_path, monkeypatch):
    earlier = ("This model's maximum context length is 80000 tokens. However, you requested 98192 tokens "
               "(90000 in the messages, 8192 in the completion).")
    body = {"error": {"code": 400, "message": "Provider returned error", "metadata": {
        "provider_name": "Current", "raw": json.dumps({"error": {"message": "Upstream request failed"}}),
        "previous_errors": [{"code": 400, "message": "Provider returned error", "provider_name": "Earlier",
                             "raw": json.dumps({"error": {"message": earlier}})}]}}}
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    exc = _sdk_error(400, json.dumps(body).encode())
    assert classify_llm_exception(exc).kind == "context_overflow"  # recognition is unchanged
    found = sr.from_exception(exc)
    assert found is not None and not found.has_numbers
    usage = {}
    stamp_owner_provider_message(usage, exc, classify_llm_exception(exc))
    sr.stamp(usage, exc)
    record_refusal(usage, _plan())
    assert not ce._read(tmp_path).get("main_windows")


@pytest.mark.parametrize("shape", ["sibling", "structured_sibling", "history"])
def test_a_numberless_typed_refusal_borrows_no_other_partys_limit(shape):
    sentence = "prompt is too long: 90000 tokens > 4000 maximum"
    if shape != "history":
        body = {"code": 400, "message": "Provider returned error",
                "current": {"provider_name": "current", "code": "context_length_exceeded"},
                "diagnostic": {"provider_name": "unrelated", "message": sentence}}
        if shape == "structured_sibling":
            body["diagnostic"] = {**_structured(), "provider_name": "unrelated"}
    else:
        body = {"code": 400, "message": "Provider returned error", "metadata": {
            "provider_name": "current", "raw": json.dumps({"error": {
                "code": 400, "type": "invalid_request_error", "message": "The provider refused this request"}}),
            "previous_errors": [{"provider_name": "previous", "raw": json.dumps({"error": {
                "code": "context_length_exceeded", "message": sentence}})}]}}
    exc = _sdk_error(400, json.dumps({"error": body}).encode())
    assert classify_llm_exception(exc).kind == "context_overflow"
    assert not sr.from_exception(exc).has_numbers
    assert not (body_error_facts(body, False)["answer"].get("size") or {}).get("window")


@pytest.mark.parametrize("current_body_limit", [False, True])
def test_an_earlier_attempts_body_limit_leaves_the_current_refusals_numbers(current_body_limit):
    body = {"code": "context_length_exceeded", "message": FROZEN + (" Request body too large." if current_body_limit else ""),
            "metadata": {"previous_errors": [{"code": 413, "message": "Request body too large", "provider_name": "Earlier"}]}}
    exc = _sdk_error(400, json.dumps({"error": body}).encode())
    assert classify_llm_exception(exc).kind == "context_overflow"  # recognition is unchanged
    windows = {(sr.from_exception(exc) or sr.SizeRefusal()).window,
               (body_error_facts(body, False)["answer"].get("size") or {}).get("window", 0)}
    assert windows == ({0} if current_body_limit else {80000})  # only the current object's own body limit excludes


def test_a_siblings_other_cause_leaves_the_recognised_numbers_on_both_paths():
    size = "This model's maximum context length is 8192 tokens. However, your request has 9000 input tokens."
    body = {"code": 400, "message": "Provider returned error",
            "current": {"provider_name": "cur", "code": "context_length_exceeded", "message": size},
            "diagnostic": {"provider_name": "other", "code": "rate_limit_exceeded", "message": "an unrelated note"}}
    exc = _sdk_error(400, json.dumps({"error": body}).encode())
    via_exception = sr.from_exception(exc)
    via_body = body_error_facts(body, False)["answer"].get("size") or {}
    assert (via_exception.window, via_exception.requested) == (via_body.get("window"), via_body.get("requested")) == (8192, 9000)


@pytest.mark.parametrize("earlier", [
    {"code": "context_length_exceeded", "provider_name": "Earlier",
     "message": "This model's maximum context length is 80000 tokens. However, you requested 98192 tokens."},
    {"code": sr.LLAMA_CPP_TYPE, "type": sr.LLAMA_CPP_TYPE, "n_ctx": 80000, "n_prompt_tokens": 91000,
     "provider_name": "llama.cpp", "message": "Upstream request failed"},
])
@pytest.mark.parametrize("current, expected_kind, refused", [
    ({"code": 400, "message": "Provider returned error"}, "context_overflow", True),
    ({"code": "rate_limit_exceeded", "message": "Rate limit exceeded upstream"}, "provider_transient", None),
])
def test_a_named_earlier_size_code_is_last_resort_for_a_generic_wrapper(earlier, current, expected_kind, refused):
    """The earlier attempt's named/local code may type an otherwise unexplained generic wrapper, and its subtree
    never states a number; a known current cause of another kind keeps its classification and takes no number."""
    body = {"error": {**current, "metadata": {"provider_name": "Current", "previous_errors": [earlier]}}}
    exc = _sdk_error(400, json.dumps(body).encode())
    assert classify_llm_exception(exc).kind == expected_kind
    found = sr.from_exception(exc)
    if refused is not None:
        assert (found is not None) is refused
    assert found is None or found.window == 0  # the earlier attempt's sentence states no number
    size = body_error_facts(body["error"], False)["answer"].get("size") or {}
    assert size.get("window", 0) == 0


def test_a_normalized_excluded_size_is_no_recognition():
    """A stamped projection whose only size row is the negative reason is no size refusal; a recognised basis
    still is one."""
    from ouroboros.loop_llm_call import _classify_empty_response
    excluded = {"provider_error": {"answer": {"basis": None, "size": {"source": "excluded"}}}}
    assert _classify_empty_response(excluded, {})[0] == "provider_body_error"
    recognised = {"provider_error": {"answer": {"basis": "context_overflow", "size": {"source": "excluded"}}}}
    assert _classify_empty_response(recognised, {})[0] == "remote_context_overflow"
