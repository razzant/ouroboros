"""Size refusals of real servers, read as one typed fact (P1, 15.11.2026).

Each body in ``fixtures/size_refusals`` names its server, version and how it was obtained (``manifest.json``):
llama.cpp bodies captured on the stand, vLLM/Ollama/LM Studio built from their sources, OpenRouter's nested
upstream error from #3518, and negative controls. The exception is made by the real OpenAI SDK over an
``httpx.MockTransport`` and classified by the host's own classifier, as a Main call meets it.
"""
from __future__ import annotations

import json
import pathlib

import httpx
import openai
import pytest

from ouroboros import size_refusal as sr
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
