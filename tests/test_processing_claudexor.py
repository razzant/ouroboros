"""Consume the engine's Processing contract without a second generation owner."""

import asyncio
import copy
from types import SimpleNamespace

import pytest

from ouroboros import llm_claudexor as cx
from tests.test_llm_claudexor import MODEL, ROUTE, ledger, result, setup as setup_fixture

setup = setup_fixture
_MODEL_SOURCES = cx.model_sources


@pytest.fixture(autouse=True)
def source_view(monkeypatch):
    monkeypatch.setenv("OUROBOROS_PROCESSING_PREFERENCE", "")
    monkeypatch.setenv("OUROBOROS_MODEL_PROCESSING_PREFERENCES", "{}")
    monkeypatch.setattr(cx, "model_sources", lambda **kwargs: {
        "sources": [{"id": "codex", "processingPreferences": ["standard", "fast", "economy"]}]})


def receipt(preference="standard", native="default", observed="standard"):
    return {"requested": preference, "submitted": preference, "submittedNative": native,
            "observed": observed, "observedNative": [native], "reason": None,
            "source": "codex.service_tier"}


def refusal(preference="economy", reason="capacity"):
    # Engine9bd378ae fixture: response_received is the refusal envelope; these
    # independent fields prove generation never started. Not raw HTTP inference.
    native = "flex" if preference == "economy" else "priority"
    return {**result(outcome="failed"), "message": None, "processing": receipt(preference, native, "unknown"),
            "problem": {"code": "processing_unavailable", "message": "Engine refused processing",
                "context": {"httpStatus": 429 if reason == "capacity" else 400,
                    "vendorCode": "resource_unavailable" if reason == "capacity" else "unsupported_parameter",
                    "parameter": "service_tier", "generationStarted": False,
                    "processingFallback": "standard", "processingRefusal": reason}}}


@pytest.mark.parametrize(("preference", "asynchronous"), [("standard", False), ("fast", True), ("economy", False)])
def test_advertised_preference_is_frozen_in_body_and_result_survives_settlement(setup, preference, asynchronous):
    root, gateway, client = setup
    observed = receipt(preference, "priority" if preference == "fast" else "flex" if preference == "economy" else "default", preference)
    gateway.results = [{**result(), "processing": observed}]
    kwargs = dict(messages=[{"role": "user", "content": "input"}], model=MODEL, processing_preference=preference)
    _message, usage = asyncio.run(client.chat_async(**kwargs)) if asynchronous else client.chat(**kwargs)
    assert gateway.uploads[0][0]["options"]["processingPreference"] == preference
    assert usage["processing"] == observed
    assert ledger(root)[-1]["processing"] == observed
    assert ledger(root)[0]["reservation_upper_bound_usd"] is None


@pytest.mark.parametrize(("preference", "reason", "asynchronous"), [
    ("economy", "capacity", False),
    ("economy", "unsupported", True),
    ("fast", "unsupported", False),
])
def test_response_received_with_proved_no_generation_reprepares_standard(setup, monkeypatch, preference, reason, asynchronous):
    root, gateway, client = setup
    calls = []
    monkeypatch.setattr(cx, "model_sources", lambda **kwargs: calls.append(kwargs) or {
        "sources": [{"id": "codex", "processingPreferences": ["standard", "fast", "economy"]}]})
    gateway.results = [refusal(preference, reason), {**result(), "processing": receipt()}]
    gateway.dispatch = ["response_received", "response_received"]
    kwargs = dict(messages=[{"role": "user", "content": "input"}], model=MODEL, processing_preference=preference)
    _message, usage = asyncio.run(client.chat_async(**kwargs)) if asynchronous else client.chat(**kwargs)
    first, second = [row[0] for row in gateway.uploads]
    assert {**first, "options": {**first["options"], "processingPreference": "standard"}} == second
    assert len(calls) == 1 and len(gateway.accepted_operations) == 2 and len(gateway.acks) == 2
    finals = list({row["attempt_id"]: row for row in ledger(root)}.values())
    assert [row["state"] for row in finals] == ["released", "settled"]
    assert finals[0]["candidate_raw_sha256"] != finals[1]["candidate_raw_sha256"]
    assert all(row["processing_preference"] == preference for row in finals)
    assert usage["processing"] == receipt()


@pytest.mark.parametrize("axis", ["unknown_outcome", "exact_native", "standard"])
def test_only_explicit_no_start_advisory_proof_allows_retry(setup, axis):
    root, gateway, client = setup
    response = refusal()
    kwargs = {"processing_preference": "economy"}
    if axis == "unknown_outcome":
        response["outcome"] = "unknown"
    if axis == "unknown_dispatch":
        gateway.dispatch = ["unknown"]
    if axis == "no_proof":
        response["problem"]["context"].pop("generationStarted")
    if axis == "standard":
        kwargs["processing_preference"] = "standard"
    gateway.results = [response]
    with pytest.raises(cx.ClaudexorModelError) as raised:
        if axis == "exact_native":
            target = {**client._resolve_remote_target(MODEL), "processing_preference": "economy"}
            cx.chat_claudexor(target, [], None, service_tier="flex")
        else:
            client.chat([], MODEL, **kwargs)
    assert len(gateway.accepted_operations) == 1
    if axis.startswith("unknown"):
        assert raised.value.code == "model_outcome_unknown" and ledger(root)[-1]["state"] == "unresolved"
    if axis == "exact_native":
        assert gateway.uploads[0][0]["options"]["serviceTier"] == "flex"


def test_missing_transport_support_keeps_strict_legacy_body_and_discloses(setup, monkeypatch):
    root, gateway, client = setup
    monkeypatch.setattr(cx, "model_sources", lambda **kwargs: {"sources": [{"id": "codex"}]})
    _message, usage = client.chat([], MODEL, processing_preference="fast")
    assert "processingPreference" not in gateway.uploads[0][0]["options"]
    assert usage["processing"]["requested"] == "fast"
    assert usage["processing"]["submitted"] is None and usage["processing"]["observed"] == "unknown"
    assert ledger(root)[-1]["processing"] == usage["processing"]


def test_legacy_empty_and_captured_target_do_not_probe_again(setup, monkeypatch):
    _root, gateway, client = setup
    monkeypatch.setattr(cx, "model_sources", lambda **kwargs: pytest.fail("No metadata read required"))
    client.chat([], MODEL, processing_preference="")
    target = {"source": "codex", "processing_preference": "fast", "processing_preferences": ["fast"]}
    assert cx.prepare_processing_target(target) is target
    assert "processingPreference" not in gateway.uploads[0][0]["options"]


def test_processing_retry_does_not_replace_owner_turn_before_success(setup, monkeypatch):
    _root, gateway, client = setup
    monkeypatch.setattr(cx, "owned_engine_version", lambda: "3.10.4")
    turn = {"source": "codex", "payload": {"turnState": "private-original"}}
    slot = cx.ModelTurnState(copy.deepcopy(turn))
    gateway.results = [refusal(), {**result(), "nativeContinuation": copy.deepcopy(turn), "processing": receipt()}]
    gateway.dispatch = ["response_received", "response_received"]
    client.chat([], MODEL, processing_preference="economy", model_turn_state=slot)
    assert [row[0]["nativeContinuation"] for row in gateway.uploads] == [turn, turn]
    assert slot.envelope == turn


@pytest.mark.parametrize("order", ["native_first", "processing_first"])
def test_one_repair_on_each_axis_uses_the_same_bounded_preparation_loop(setup, order):
    root, gateway, client = setup
    native = result(outcome="failed", route={**ROUTE, "credentialProfileId": "account-b"},
                    problem={"code": "invalid_continuation", "message": "Account changed"})
    pair = [(native, "not_started"), (refusal(), "response_received")]
    if order == "processing_first":
        pair.reverse()
    gateway.results = [value for value, _dispatch in pair] + [{**result(), "processing": receipt()}]
    gateway.dispatch = [dispatch for _value, dispatch in pair] + ["response_received"]
    client.chat([result()["message"]], MODEL, processing_preference="economy")
    assert len(gateway.accepted_operations) == 3
    assert [row["state"] for row in {r["attempt_id"]: r for r in ledger(root)}.values()] == ["released", "released", "settled"]


def test_modern_source_view_uses_negotiated_facade_only(monkeypatch):
    from ouroboros.gateways import claudexor as facade

    calls = []
    gateway = SimpleNamespace(operations=lambda: ["fixture"], close=lambda: calls.append("close"),
        list_model_sources=lambda **kwargs: calls.append(kwargs) or {"sources": []})
    monkeypatch.setattr(cx, "read_owned_gateway", lambda: gateway)
    monkeypatch.setattr(facade, "account_catalog_supported", lambda operations, path:
        operations == ["fixture"] and path == "/v2/model-sources", raising=False)
    _MODEL_SOURCES(processing_view=True)
    assert calls == [{"view": "accounts"}, "close"]


def test_standard_must_be_supported_before_emitting_a_processing_retry(setup, monkeypatch):
    _root, gateway, client = setup
    monkeypatch.setattr(cx, "model_sources", lambda **kwargs: {
        "sources": [{"id": "codex", "processingPreferences": ["economy"]}]})
    gateway.results = [refusal()]
    with pytest.raises(cx.ClaudexorModelNotDispatched):
        client.chat([], MODEL, processing_preference="economy")
    assert len(gateway.accepted_operations) == 1, "Do not retry by omitting unsupported Standard and inheriting native premium"


def test_old_facade_does_not_send_new_query_parameter(monkeypatch):
    from ouroboros.gateways import claudexor as facade

    calls = []
    gateway = SimpleNamespace(close=lambda: calls.append("close"),
        list_model_sources=lambda: calls.append("legacy") or {"sources": []})
    monkeypatch.setattr(cx, "read_owned_gateway", lambda: gateway)
    monkeypatch.delattr(facade, "account_catalog_supported", raising=False)
    _MODEL_SOURCES(processing_view=True)
    assert calls == ["legacy", "close"]
