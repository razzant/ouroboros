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
