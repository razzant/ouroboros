"""Consume the engine's Processing contract without a second generation owner."""

import asyncio
import copy
from types import SimpleNamespace

import pytest

from ouroboros import llm_claudexor as cx
from ouroboros import usage_accounting as ua
from ouroboros.loop_llm_call import classify_llm_exception
from ouroboros.observability import read_call_payload
from ouroboros.transport_custody import is_pre_dispatch_transport_failure
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
    # Claudexor 6.2.0's emptyModelResult + processingAdmissionProblem after a non-OK POST.
    refused = result(outcome="failed", route={**ROUTE, "model": None})
    native = {"economy": "flex", "fast": "priority", "ultrafast": "ultrafast"}[preference]
    return {**refused, "message": None, "appliedOptions": {}, "usage": dict.fromkeys(refused["usage"]),
            "cost": {**refused["cost"], "valuationUsd": None, "valuationKnowledge": "unknown"},
            "processing": {**receipt(preference, native, "unknown"), "observedNative": []},
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


@pytest.mark.parametrize(("preference", "reason", "asynchronous", "counters", "cash"), [
    ("economy", "capacity", False, None, None),
    ("economy", "unsupported", True, 0, None),
    ("fast", "unsupported", False, 0, 0.0),
])
def test_response_received_with_proved_no_generation_reprepares_standard(
        setup, monkeypatch, preference, reason, asynchronous, counters, cash):
    root, gateway, client = setup
    calls = []
    monkeypatch.setattr(cx, "model_sources", lambda **kwargs: calls.append(kwargs) or {
        "sources": [{"id": "codex", "processingPreferences": ["standard", "fast", "economy"]}]})
    refused = refusal(preference, reason)
    refused["usage"] = dict.fromkeys(refused["usage"], counters)
    if cash is not None:
        refused["cost"].update(knowledge="exact", cashUsd=cash)
    gateway.results = [refused, {**result(), "processing": receipt()}]
    gateway.dispatch = ["response_received", "response_received"]

    ack, ack_observations = gateway.acknowledge_model_result, []

    def observe_ack(operation_id, digest):
        key = gateway.uploads[int(operation_id.removeprefix("op-"))][1]
        row = next(row for row in ledger(root) if row["attempt_id"] == key)
        manifest, _, _ = read_call_payload(root, task_id=row["task_id"], call_id=f"{key}_model_response")
        ack_observations.append((operation_id, row["state"], manifest["dispatch_state"]))
        return ack(operation_id, digest)

    monkeypatch.setattr(gateway, "acknowledge_model_result", observe_ack)
    kwargs = dict(messages=[{"role": "user", "content": "input"}], model=MODEL, processing_preference=preference,
                  model_account_override="account-a")
    _message, usage = asyncio.run(client.chat_async(**kwargs)) if asynchronous else client.chat(**kwargs)
    first, second = [row[0] for row in gateway.uploads]
    assert {**first, "options": {**first["options"], "processingPreference": "standard"}} == second
    assert len(calls) == 1 and len(gateway.accepted_operations) == 2 and len(gateway.acks) == 2
    assert ack_observations == [("op-0", "settled", "response_received"), ("op-1", "settled", "response_received")]
    finals = ledger(root)
    assert usage["ledger_attempt_ids"] == [row["attempt_id"] for row in finals] == [key for _, key in gateway.uploads]
    assert finals[0]["candidate_raw_sha256"] != finals[1]["candidate_raw_sha256"]
    assert all(row["processing_preference"] == preference for row in finals)
    first_row = finals[0]
    assert first_row["cost_usd"] == cash and first_row["cost_final"] is (cash is not None)
    assert first_row["prompt_tokens"] == first_row["completion_tokens"] == counters
    assert first_row["processing"] == refused["processing"]
    recovered, recovered_usage, recovered_cost, recovered_final = cx.recover_model_attempt(
        root, first_row, gateway_factory=lambda: pytest.fail("Read the retained refusal, never rejoin a settled one"))
    assert (recovered, recovered_cost, recovered_final) == ("settled", cash, cash is not None)
    assert recovered_usage["processing"] == first_row["processing"]
    # The engine's receipt answers the Standard retry; the call asked the work's tier, which its receipt keeps.
    assert usage["processing"] == {**receipt(), "requested": preference, "reason": "submitted_mode_differs"}
    assert finals[1]["processing"] == receipt()


@pytest.mark.parametrize("axis", ["unknown_outcome", "unknown_dispatch", "no_proof", "exact_native", "standard"])
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
        assert not gateway.acks
    else:
        assert raised.value.code == "processing_unavailable" and ledger(root)[-1]["state"] == "settled"
        assert gateway.acks and not classify_llm_exception(raised.value).retry_same_request
    assert not is_pre_dispatch_transport_failure(raised.value)
    if axis == "exact_native":
        assert gateway.uploads[0][0]["options"]["serviceTier"] == "flex"


@pytest.mark.parametrize(("contradiction", "asynchronous", "cash", "knowledge"), [
    ("usage", False, None, "unknown"),
    ("price", True, .12, "exact"),
    ("price", False, .34, "estimated"),
    ("output", True, None, "unknown"),
    ("substitution", False, None, "unknown"),
])
def test_a_conflicting_tier_refusal_settles_its_evidence_but_cannot_reask(setup, contradiction, asynchronous, cash, knowledge):
    root, gateway, client = setup
    response = refusal()
    if contradiction == "usage":
        response["usage"].update(input_tokens=20, output_tokens=7, cached_input_tokens=12)
    elif contradiction == "price":
        response["cost"].update(knowledge=knowledge, **{"cashUsd" if knowledge == "exact" else "estimatedUsd": cash})
    elif contradiction == "output":
        response["message"] = result()["message"]
    else:
        response["outcome"] = "incomplete"
        response["modelMismatch"] = {"requested": "exact-model", "observed": "another-model"}
    # A conflicting account retry grant can't send either.
    response["problem"]["context"].update(retryOnOtherAccount=True, operationId="op-0")
    gateway.results = [response]
    kwargs = dict(messages=[], model=MODEL, processing_preference="economy")
    with pytest.raises(cx.ClaudexorModelError) as raised:
        asyncio.run(client.chat_async(**kwargs)) if asynchronous else client.chat(**kwargs)
    row, = ledger(root)
    assert row["state"] == "settled" and row["cost_usd"] == cash and row["cost_final"] is (knowledge == "exact")
    assert row["cost_estimated"] is (knowledge == "estimated")
    assert (row["prompt_tokens"], row["completion_tokens"], row["cached_tokens"]) == tuple(
        response["usage"][key] for key in ("input_tokens", "output_tokens", "cached_input_tokens"))
    assert row["cost_evidence"] == response["cost"]
    assert raised.value.usage["claudexor"]["result_custody"]["state"] == "acknowledged"
    assert raised.value.processing_refusal_conflict and classify_llm_exception(raised.value).kind == "provider_outcome_unknown"
    assert len(gateway.creates) == len(gateway.acks) == 1


@pytest.mark.parametrize(("budget", "count"), [(1, 1), (2, 2)])
def test_refusal_consumes_a_send_and_standard_refusal_cannot_retry(setup, budget, count):
    root, gateway, client = setup
    gateway.results = [refusal(), refusal()]
    gateway.dispatch = ["response_received"] * 2
    with ua.physical_attempt_limit(budget), pytest.raises(cx.ClaudexorModelError) as raised:
        client.chat([], MODEL, processing_preference="economy")
    rows = ledger(root)
    assert [row["state"] for row in rows] == ["settled"] * count
    assert len(gateway.creates) == len(gateway.acks) == count
    assert raised.value.code == "processing_unavailable" and not classify_llm_exception(raised.value).retry_same_request
    assert [row[0]["options"]["processingPreference"] for row in gateway.uploads] == ["economy", "standard"][:count]


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


def test_a_found_source_list_serves_its_engine_build_and_anything_else_is_asked_again(monkeypatch):
    """A source's declared list is a fact of the exact engine build: once a read found it, later calls on that build
    and source get the same list without asking; a failed read, a missing list, another source, another build or an
    unknown build asks the engine again and remembers nothing it did not find."""
    reads, build = [], {"engine": ("6.1.2", "sha-a")}

    def sources(**kwargs):
        reads.append(kwargs)
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return {"sources": [answer]}

    monkeypatch.setattr(cx, "_PROCESSING_PREFERENCES", {})
    monkeypatch.setattr(cx, "current_engine", lambda: build["engine"])
    monkeypatch.setattr(cx, "model_sources", sources)
    codex, claude = ({"source": name, "processing_preference": "fast"} for name in ("codex", "claude"))
    listed = lambda name, *values: {"id": name, "processingPreferences": list(values)}  # noqa: E731
    answers = [cx.ClaudexorUnavailable("daemon_unreachable", "down"), {"id": "codex"}, listed("codex", "standard", "fast"),
               listed("claude", "standard"), listed("codex", "standard"), listed("codex", "fast"), listed("codex", "fast")]
    assert [cx.prepare_processing_target(codex)["processing_preferences"] for _ in range(4)] == [
        [], [], ["standard", "fast"], ["standard", "fast"]]
    assert len(reads) == 3 and reads[-1] == {"processing_view": True}, "the found list served the fourth call"
    assert cx.prepare_processing_target(claude)["processing_preferences"] == ["standard"] and len(reads) == 4
    build["engine"] = ("6.1.2", "sha-b")
    assert cx.prepare_processing_target(codex)["processing_preferences"] == ["standard"] and len(reads) == 5
    build["engine"] = None
    assert [cx.prepare_processing_target(codex)["processing_preferences"] for _ in range(2)] == [["fast"], ["fast"]]
    assert len(reads) == 7 and not answers


def test_a_found_list_is_kept_only_for_the_build_that_answered_it(monkeypatch):
    """A list is kept under the engine build that answered it: a replacement whose handshake is remembered while the
    read is in flight keeps nothing, so A's list never serves B, which is asked and then serves itself."""
    reads, build = [], {"engine": ("6.1.2", "sha-a")}

    def sources(**_kwargs):
        reads.append(build["engine"])
        answer = {"id": "codex", "processingPreferences": ["standard"] if build["engine"][1] == "sha-a" else
                  ["standard", "fast"]}
        if len(reads) == 1:
            build["engine"] = ("6.1.2", "sha-b")  # announced by another caller before A's answer is kept
        return {"sources": [answer]}

    monkeypatch.setattr(cx, "_PROCESSING_PREFERENCES", {})
    monkeypatch.setattr(cx, "current_engine", lambda: build["engine"])
    monkeypatch.setattr(cx, "model_sources", sources)
    codex = {"source": "codex", "processing_preference": "fast"}
    assert cx.prepare_processing_target(codex)["processing_preferences"] == ["standard"], "A's answer, for this call"
    assert [cx.prepare_processing_target(codex)["processing_preferences"] for _ in range(2)] == [["standard", "fast"]] * 2
    assert reads == [("6.1.2", "sha-a"), ("6.1.2", "sha-b")], "B was asked once; its own list then served it"


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
    assert [row["state"] for row in ledger(root)] == (
        ["released", "settled", "settled"] if order == "native_first" else ["settled", "released", "settled"])


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
    with pytest.raises(cx.ClaudexorModelError) as raised:
        client.chat([], MODEL, processing_preference="economy")
    assert raised.value.physical_attempt_capture.state == "settled"
    assert not classify_llm_exception(raised.value).retry_same_request
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
