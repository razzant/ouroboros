"""The transport finalizer sets a rendered-Nano candidate's wire allowance (T2/T3).

One rule (``context_budget.reply_allowance_tokens``) on the candidate about to be sealed:
its own bounded count times Main's density against the bound capacity. Every lane with a
numeric field gets it through the existing mock transports; a payload without the field
stays without it; Low and Max payloads are byte-identical to an unbound send.
"""
import copy
import json
import math
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from ouroboros import llm_attempt, usage_accounting as ua
from ouroboros.context_budget import NANO_MIN_HEADROOM_TOKENS, OWNER_NANO_TARGET_TOKENS, reply_allowance_tokens
from ouroboros.context_fit import bounded_prompt_tokens_for_payload
from ouroboros.request_wire_recovery import current_wire_candidate
from tests.test_processing_transport import transport as _transport

transport = _transport
C = 65_536
INCIDENT_DENSITY = 0.835284798376117
TOOL = {"type": "function", "function": {
    "name": "inspect", "description": "read " * 40, "parameters": {"type": "object", "properties": {"name": {"type": "string"}}}}}


def _physical(profile="owner_nano", *, mode="nano", capacity=128_000, density=None):
    return ua.PhysicalAttemptContext(
        profile=profile, rendered_mode=mode, measurement_basis="fresh_route_usage" if density else "cold_estimate",
        route_fp="r", round_id="x:round:1", target_total_tokens=OWNER_NANO_TARGET_TOKENS if profile == "owner_nano" else None,
        capacity_total_tokens=capacity, context_target_miss=False, automatic_pass_used=False, measurement_density=density)


def _sent_payload(sent):
    return sent["payload"] if "payload" in sent else sent


def _rule(payload, physical, field):
    """The rule on a candidate's own system/messages/tools (what the finalizer measures)."""
    raw = bounded_prompt_tokens_for_payload({key: payload[key] for key in ("system", "messages", "tools", "functions")
                                             if key in payload}, 0)
    return reply_allowance_tokens(
        caller_max_tokens=C, nano=True, owner_nano=physical.profile == "owner_nano",
        input_tokens=math.ceil(raw * (physical.measurement_density or 1.0)), raw_input_tokens=raw,
        window_tokens=physical.capacity_total_tokens)


def _spy_measured_candidates(monkeypatch):
    """The candidates the finalizer measured, before the wire projection (direct OpenAI's custom tool
    dialect adds bytes after the measurement; the window's slack absorbs that)."""
    measured, real = [], llm_attempt.bound_reply_allowance

    def spy(target, payload):
        measured.append(copy.deepcopy(payload))
        return real(target, payload)

    monkeypatch.setattr(llm_attempt, "bound_reply_allowance", spy)
    return measured


@pytest.mark.parametrize("model,field", [("openai/test-model", "max_tokens"), ("openai::test-model", "max_completion_tokens"),
                                         ("anthropic::test-model", "max_tokens")])
def test_the_wire_field_equals_the_rule_on_every_lane_with_a_numeric_field(transport, monkeypatch, model, field):
    _root, client, sent = transport
    measured = _spy_measured_candidates(monkeypatch)
    physical = _physical(density=0.9)
    messages = [{"role": "system", "content": "govern " * 2_000}, {"role": "user", "content": "complete source " * 18_000}]
    with ua.bind_physical_attempt_context(physical):
        message, _usage = client.chat(messages, model, max_tokens=C, tools=[TOOL])
    assert message["content"] == "ok" and len(sent) == 1 and len(measured) == 1
    payload = _sent_payload(sent[0])
    expected = _rule(measured[0], physical, field)
    assert NANO_MIN_HEADROOM_TOKENS < expected < C, expected  # the window-bound regime, not a floor or ceiling clamp
    assert payload[field] == expected
    if model != "openai::test-model":  # no wire dialect after the measurement: the sent bytes ARE the measured ones
        assert _rule(payload, physical, field) == expected


def test_a_large_messages_system_field_lowers_the_anthropic_allowance(transport, monkeypatch):
    _root, client, sent = transport
    measured = _spy_measured_candidates(monkeypatch)
    physical = _physical()
    user = [{"role": "user", "content": "complete source " * 16_000}]
    with ua.bind_physical_attempt_context(physical):
        client.chat([{"role": "system", "content": "short"}, *user], "anthropic::test-model", max_tokens=C)
        client.chat([{"role": "system", "content": "governance " * 8_000}, *user], "anthropic::test-model", max_tokens=C)
    short, large = (_sent_payload(item) for item in sent)
    assert "system" in large and "system" in measured[1]
    assert large["max_tokens"] == _rule(measured[1], physical, "max_tokens") == _rule(large, physical, "max_tokens")
    assert short["max_tokens"] - large["max_tokens"] > 15_000


def test_the_gigachat_lane_applies_the_same_rule(transport, monkeypatch):
    _root, client, sent = transport
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", function_call=None), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=1, precached_prompt_tokens=0))
    monkeypatch.setattr(client, "_get_gigachat_client", lambda target, timeout=None: SimpleNamespace(
        chat=lambda candidate: sent.append(copy.deepcopy(candidate)) or completion))
    target = {"provider": "gigachat", "resolved_model": "GigaChat-2", "usage_model": "gigachat::GigaChat-2"}
    physical = _physical(density=1.1)
    with ua.bind_physical_attempt_context(physical):
        client._chat_gigachat(target, [{"role": "user", "content": "привет " * 30_000}], [TOOL], "high", C, "auto")
    assert "functions" in sent[0] and sent[0]["max_tokens"] == _rule(sent[0], physical, "max_tokens") < C
    client._chat_gigachat(target, [{"role": "user", "content": "привет " * 30_000}], [TOOL], "high", C, "auto")
    assert sent[1]["max_tokens"] == C  # unbound (no Main context): the caller's ceiling, as before


def test_a_payload_without_a_numeric_field_stays_without_one(transport):
    """The web-search Responses body carries no max_tokens; the finalizer never creates the field."""
    from ouroboros.request_wire_recovery import request_wire_call_scope

    target = {"provider": "openai", "resolved_model": "gpt-test", "usage_model": "openai/gpt-test", "base_url": "https://x"}
    payload = {"model": "gpt-test", "tools": [{"type": "web_search", "search_context_size": "low"}],
               "reasoning": {"effort": "low"}, "tool_choice": "auto", "input": "query " * 30_000, "stream": True}
    with ua.bind_physical_attempt_context(_physical()), request_wire_call_scope():
        candidate = llm_attempt._finalized_physical_candidate(target, payload, "responses")
    assert "max_tokens" not in candidate and "max_completion_tokens" not in candidate


@pytest.mark.parametrize("profile,mode", [("owner_max", "max"), ("owner_low", "low"), ("task_local_low", "low")])
def test_low_and_max_payloads_are_byte_identical_to_an_unbound_send(transport, profile, mode):
    _root, client, sent = transport
    messages = [{"role": "system", "content": "govern " * 2_000}, {"role": "user", "content": "complete source " * 6_000}]
    with ua.bind_physical_attempt_context(_physical(profile, mode=mode, capacity=90_000)):
        client.chat(messages, "openai::test-model", max_tokens=C, tools=[TOOL])
    client.chat(messages, "openai::test-model", max_tokens=C, tools=[TOOL])
    assert sent[0] == sent[1] and sent[0]["max_completion_tokens"] == C


def test_a_nano_the_window_chose_answers_to_the_window_alone(transport):
    _root, client, sent = transport
    messages = [{"role": "user", "content": "complete source " * 6_000}]
    for profile in ("task_local_nano", "owner_nano"):
        with ua.bind_physical_attempt_context(_physical(profile, capacity=None)):
            client.chat(messages, "openai::test-model", max_tokens=C)
    assert sent[0]["max_completion_tokens"] == C  # no window, no owner target: the ceiling
    raw = bounded_prompt_tokens_for_payload(sent[1], 0)
    assert sent[1]["max_completion_tokens"] == OWNER_NANO_TARGET_TOKENS - raw  # the owner's target stands in


def test_output_rebind_preserves_the_same_canonical_tool_catalog(transport):
    from ouroboros.request_wire_recovery import request_wire_call_scope

    _, client, _ = transport
    target = client._resolve_remote_target("openai::test-model")
    payload = client._build_remote_kwargs(target, [{"role": "user", "content": "all input " * 30_000}],
                                          "high", C, "auto", None, [TOOL])
    with ua.bind_physical_attempt_context(_physical()), request_wire_call_scope():
        first = llm_attempt._finalized_physical_candidate(target, payload, "chat.completions")
        catalog = current_wire_candidate().custom_catalog_sha256
        # A registered wire form re-finalized (a clock refresh, a rejoin) is measured on its canonical
        # source, not on its projected dialect's extra bytes: the allowance and the registration hold.
        second = llm_attempt._finalized_physical_candidate(target, first, "chat.completions")
        assert first == second and current_wire_candidate().custom_catalog_sha256 == catalog
        assert first["max_completion_tokens"] < C


def _payload_with_bounded_estimate(estimate, tools):
    """A synthetic candidate whose bounded count is exactly ``estimate`` (the estimator is chars/4)."""
    payload = {"model": "openai/test-model", "messages": [{"role": "user", "content": ""}], "tools": tools, "max_tokens": C}
    base = bounded_prompt_tokens_for_payload(payload, 0)
    assert estimate >= base
    payload["messages"][0]["content"] = "x" * (4 * (estimate - base))
    assert bounded_prompt_tokens_for_payload(payload, 0) == estimate
    return payload


def test_the_six_incident_requests_keep_the_whole_ceiling_on_their_known_million_token_window(transport):
    """Scout S2: six retries of one round, bounded counts 84,784 .. 84,923 (each stacked clock line +28),
    fresh density 0.835…, window 1,000,000. The incident sent 216 .. 77; the rule sends 65,536."""
    from ouroboros.request_wire_recovery import request_wire_call_scope

    tools = [dict(TOOL, function=dict(TOOL["function"], name=f"tool_{index}")) for index in range(32)]
    target = {"provider": "openrouter", "resolved_model": "openai/test-model", "usage_model": "openai/test-model",
              "base_url": "https://openrouter.ai/api/v1", "supports_openrouter_extensions": True}
    for estimate in (84_784, 84_812, 84_840, 84_868, 84_895, 84_923):
        with ua.bind_physical_attempt_context(_physical(capacity=1_000_000, density=INCIDENT_DENSITY)), request_wire_call_scope():
            candidate = llm_attempt._finalized_physical_candidate(target, _payload_with_bounded_estimate(estimate, tools), "chat.completions")
        assert candidate["max_tokens"] == C
        with ua.bind_physical_attempt_context(_physical(capacity=None, density=INCIDENT_DENSITY)), request_wire_call_scope():
            hidden = llm_attempt._finalized_physical_candidate(target, _payload_with_bounded_estimate(estimate, tools), "chat.completions")
        assert hidden["max_tokens"] == OWNER_NANO_TARGET_TOKENS - math.ceil(estimate * INCIDENT_DENSITY)  # 14,270 .. 14,065


def test_the_128k_class_passes_a_provider_that_counts_above_the_host_estimate(transport, monkeypatch):
    """Memory-sprint acceptance row 9.1: a mature Nano input (~91.7K) on a known 128K window used to send
    65,536 and be refused. The mock provider counts 0.25 % above the host (fable R4 П8), so a sum that
    only fits on the host's own count still fails here."""
    _root, client, sent = transport
    window, refusals = 128_000, []

    def create(**candidate):
        sent.append(copy.deepcopy(candidate))
        counted = math.ceil(bounded_prompt_tokens_for_payload(candidate, 0) * 1.0025)
        if counted + candidate["max_completion_tokens"] > window:
            refusals.append(counted + candidate["max_completion_tokens"])
            error = RuntimeError(f"This model's maximum context length is {window} tokens. However, you requested "
                                 f"{refusals[-1]} tokens ({counted} in the messages, {candidate['max_completion_tokens']} "
                                 "in the completion). Please reduce the length of the messages or completion.")
            error.code = "context_length_exceeded"
            raise error
        return SimpleNamespace(model_dump=lambda: {"choices": [{"message": {"role": "assistant", "content": "ok"}}],
                                                   "usage": {"prompt_tokens": counted, "completion_tokens": 1, "cost": 0}})

    monkeypatch.setattr(client, "_get_remote_client", lambda target: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    payload = _payload_with_bounded_estimate(91_677, [TOOL])
    with ua.bind_physical_attempt_context(_physical(capacity=window)):
        message, _usage = client.chat(payload["messages"], "openai::test-model", max_tokens=C, tools=[TOOL])
    assert message["content"] == "ok" and not refusals
    assert sent[0]["max_completion_tokens"] == 20_323  # 128,000 - 91,677 - 16,000; the ceiling would sum to 157,213


def test_local_serving_window_uses_live_arguments_not_training_or_fallback(monkeypatch):
    from ouroboros.local_model import LocalModelManager

    monkeypatch.delenv("OUROBOROS_IN_WORKER", raising=False)
    manager = LocalModelManager.__new__(LocalModelManager)
    manager._proc = None
    manager._status = "offline"
    manager._context_length = 4096
    manager._serving_context_length = 0
    assert manager.serving_context_evidence()["context_window"] is None
    manager._proc = SimpleNamespace(poll=lambda: None, pid=123)
    manager._status = "ready"
    manager._serving_context_length = 16384
    manager._context_length = 131072  # Training metadata returned by /models.
    assert manager.serving_context_evidence() == {
        "context_window": 16384, "confirmed": True, "source": "owned_server_arguments", "process_id": 123}
    manager._proc = SimpleNamespace(poll=lambda: 0, pid=123)
    assert manager.serving_context_evidence()["context_window"] is None


def test_the_physical_context_literals_are_unchanged_by_the_density_field():
    """The ledger-validated ``measurement_basis`` literals stay; the witness basis string is another field."""
    physical = _physical(density=INCIDENT_DENSITY)
    assert json.loads(json.dumps(asdict(physical)))["measurement_basis"] == "fresh_route_usage"


@pytest.mark.parametrize('maximum', [4096, 8192])
@pytest.mark.parametrize('mode', ['low', 'max', 'nano', None])
@pytest.mark.parametrize('model,field', [('openai::test-model', 'max_completion_tokens'),
    ('openai/test-model', 'max_tokens'), ('anthropic::test-model', 'max_tokens'),
    ('openai-compatible::unlisted-future-model', 'max_tokens')])
def test_confirmed_output_maximum_reaches_real_wire_capture_and_reservation(transport, monkeypatch, maximum, mode, model, field):
    from ouroboros.response_limits import record_response_ack
    from tests._usage_store_testing import ledger_rows

    root, client, sent = transport
    monkeypatch.setenv('OPENAI_COMPATIBLE_BASE_URL', 'https://compatible.test/v1')
    monkeypatch.setenv('OPENAI_COMPATIBLE_API_KEY', 'test-key')
    target = client._resolve_remote_target(model)
    record_response_ack(root, provider=target['provider'], model=model,
                        base_url=target['base_url'], max_output_tokens=maximum)
    import contextlib
    bound = ua.bind_physical_attempt_context(_physical(mode=mode)) if mode else contextlib.nullcontext()
    with bound:
        message, usage = client.chat([{'role': 'user', 'content': 'small actual input'}], model,
                                     max_tokens=C, model_role='reviewer:one')
    assert message['content'] == 'ok'
    assert _sent_payload(sent[0])[field] == maximum
    rows = ledger_rows(root)
    assert ua.last_physical_attempt_capture().max_completion_tokens == maximum
    assert rows[-1]['candidate_raw_sha256']
    # Same endpoint, different model has no borrowed owner assertion.
    client.chat([{'role': 'user', 'content': 'same'}], model + '-other', max_tokens=C)
    assert _sent_payload(sent[-1])[field] == C


@pytest.mark.parametrize('provider,acked,sent_model,expected', [
    # Direct Anthropic dispatch sends (and labels usage with) the canonical id, so the
    # owner's maximum holds whichever alias the selection or the ack spelled.
    ('anthropic', 'anthropic::claude-opus-4.8', 'anthropic::claude-opus-4.8', 4096),
    ('anthropic', 'anthropic::claude-opus-4.8', 'anthropic::claude-opus-4-8', 4096),
    ('anthropic', 'anthropic::claude-opus-4-8', 'anthropic::claude-opus-4.8', 4096),
    ('anthropic', 'anthropic/claude-opus-4.8', 'anthropic::claude-opus-4-8', 4096),
    # OpenRouter ids stay exact: no alias folding there, and no direct-route borrow.
    ('openrouter', 'anthropic/claude-opus-4.8', 'anthropic/claude-opus-4.8', 4096),
    ('openrouter', 'anthropic/claude-opus-4.8', 'anthropic/claude-opus-4-8', C),
    ('anthropic', 'anthropic::claude-opus-4.8', 'anthropic/claude-opus-4-8', C),
])
def test_a_direct_anthropic_alias_keeps_the_owner_maximum_on_the_real_wire(transport, provider, acked, sent_model, expected):
    from ouroboros.response_limits import record_response_ack
    from tests._usage_store_testing import ledger_rows

    root, client, sent = transport
    record_response_ack(root, provider=provider, model=acked, max_output_tokens=4096)
    message, _usage = client.chat([{'role': 'user', 'content': 'small actual input'}], sent_model,
                                  max_tokens=C, model_role='reviewer:one')
    assert message['content'] == 'ok'
    assert _sent_payload(sent[0])['max_tokens'] == expected
    assert ua.last_physical_attempt_capture().max_completion_tokens == expected
    assert ledger_rows(root)[-1]['candidate_raw_sha256']


def test_context_ack_does_not_suppress_output_metadata_or_refresh_its_clock(tmp_path, monkeypatch):
    from ouroboros import capability_evidence as ce
    from ouroboros.response_limits import resolve_response_limit

    root = tmp_path / 'evidence'
    route = dict(provider='openai-compatible', model='openai-compatible::new-model', base_url='https://compat.test/v1')
    ce.record_owner_ack(root, **route, window_tokens=32768)
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'data': [
            {'id': 'new-model', 'max_output_tokens': 4096, 'max_tokens': 99999, 'context_length': 50000}]})
    monkeypatch.setattr('httpx.get', get)
    evidence = ce.probe(root, **route)
    assert evidence.window_tokens == 32768 and evidence.source == 'owner_ack'
    assert evidence.response_limit['max_output_tokens'] == 4096
    stamp = evidence.response_limit['observed_at']
    ce.record_owner_ack(root, **route, window_tokens=65536)
    assert ce.probe(root, **route).response_limit['observed_at'] == stamp
    assert calls == ['https://compat.test/v1/models']
    assert resolve_response_limit(root, **{**route, 'base_url': 'https://other.test/v1'}).ceiling(C) == C
