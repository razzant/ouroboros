"""Crossed SDK errors and one end-to-end refusal fact: F1 review regressions."""
import json
from types import SimpleNamespace

import pytest

from ouroboros import capability_evidence as ce, size_refusal as sr
from ouroboros.context_anchor import refused_request_density
from ouroboros.context_window import record_refusal, resolve_window
from ouroboros.loop_llm_call import classify_llm_exception
from ouroboros.loop_model_call import _refusal_target
from ouroboros.window_learning import learn_window_from_refusal
from tests._size_refusal import _sdk_error


SIZE = "request (90068 tokens) exceeds the available context size (80128 tokens), try increasing it"
COMBINED = "input length and `max_tokens` exceed context limit: 75000 + 8192 > 80000"


@pytest.mark.parametrize("code,kind", [
    ("invalid_api_key", "auth_error"), ("insufficient_quota", "quota_exhausted"),
    ("authentication_error", "auth_error"), ("quota_exceeded", "quota_exhausted"),
    ("invalid_schema", "bad_request"), ("content_policy_violation", "content_filter"),
    ("payload_too_large", "request_too_large"), ("rate_limit_exceeded", "provider_transient"),
])
@pytest.mark.parametrize("wrapping", ["direct", "raw", "twice", "outer"])
def test_negative_typed_cause_survives_every_wrapper(code, kind, wrapping):
    error = {"code": code, "message": SIZE}
    if wrapping == "outer":
        error["metadata"] = {"raw": json.dumps({"error": {"type": sr.LLAMA_CPP_TYPE,
                              "n_ctx": 80128, "n_prompt_tokens": 90068, "message": SIZE}})}
    else:
        for _ in range({"direct": 0, "raw": 1, "twice": 2}[wrapping]):
            error = {"code": 400, "message": "Provider returned error",
                     "metadata": {"raw": json.dumps({"error": error}), "provider_name": "fixture"}}
    exc = _sdk_error(400, json.dumps({"error": error}).encode())
    usage = {}
    sr.stamp(usage, exc)
    # A non-context cause must not even mint a fact that another recovery entry could store.
    assert sr.stamped(usage) is None


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


def test_error_classes_match_3f8120b4():
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
