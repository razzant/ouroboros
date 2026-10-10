"""A round another model answered is discarded and asked again, never accepted."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from ouroboros import llm_claudexor as transport
from ouroboros import usage_accounting as ua
from ouroboros.llm import LLMClient
from ouroboros.llm_messages import drop_source_native_messages
from ouroboros.loop_llm_call import classify_llm_exception, provider_no_call_source
from ouroboros.loop_llm_call import _COOLDOWN_ERROR_KINDS
from ouroboros.loop_transport import (
    emit_model_substitution, fallback_chain_allowed, provider_recovery_hint,
)
from ouroboros.model_slots import MODEL_ACCOUNTS_KEY

from tests.test_llm_claudexor import Gateway, MODEL, ROUTE, result
from tests._usage_store_testing import ledger_rows

OTHER_ACCOUNT = {**ROUTE, "credentialProfileId": "account-b", "accountFingerprint": "fingerprint-b"}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """The transport harness of `test_llm_claudexor`, with its own data root."""
    root = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    monkeypatch.delenv(MODEL_ACCOUNTS_KEY, raising=False)
    monkeypatch.setattr(transport.config, "CLAUDEXOR_MODEL_POLL_INTERVAL_SEC", 0.001)
    gateway = Gateway()
    monkeypatch.setattr(transport, "ensure_owned_gateway", lambda: gateway)
    with ua.usage_scope(ua.UsageScope(drive_root=root, task_id="task-one", root_task_id="task-one")):
        yield root, gateway, LLMClient()


def substituted(route=None, *, served="cheaper-model", outcome="completed"):
    """One terminal generation the engine says came from another model."""
    value = result(outcome=outcome, route={**(route or ROUTE), "model": served})
    value["modelMismatch"] = {"requested": "exact-model", "observed": served}
    return value


def events(root, kind):
    path = root / "logs" / "events.jsonl"
    rows = [json.loads(row) for row in path.read_text().splitlines()] if path.exists() else []
    return [row for row in rows if row.get("type") == kind]


def call(client, messages=None):
    return client.chat(messages or [{"role": "user", "content": "go"}], MODEL, None, "high")


@pytest.mark.parametrize("asynchronous", [False, True])
def test_substituted_round_is_discarded_and_asked_again(setup, asynchronous):
    root, gateway, client = setup
    gateway.results = [substituted(), result()]
    gateway.dispatch = ["response_received"] * 2
    messages = [{"role": "user", "content": "go"}]
    message, usage = (asyncio.run(client.chat_async(messages, MODEL, None, "high")) if asynchronous
                      else call(client, messages))
    # The accepted answer is the second generation, and the first one is gone.
    assert message["content"] == "Ответ 🐍"
    assert usage["claudexor"]["route"]["model"] == "exact-model"
    assert len(gateway.creates) == 2 and len(set(gateway.creates)) == 2
    assert len(gateway.acks) == 2  # the discarded bytes are acknowledged like any other
    disclosed, = usage["claudexor"]["substituted"]
    assert {key: disclosed[key] for key in ("requested", "observed", "account", "disposition")} == {
        "requested": "exact-model", "observed": "cheaper-model",
        "account": "account-a", "disposition": "redo"}
    # The discarded bytes are only evidence while custody says they are there.
    assert disclosed["result_custody"]["state"] == "acknowledged"
    row, = events(root, "model_served_mismatch")
    assert row["result_custody"]["state"] == "acknowledged"
    assert row["requested"] == "exact-model" and row["observed"] == "cheaper-model"
    assert row["disposition"] == "redo" and row["redo"] == 1 and row["redos"] == 2
    assert row["discarded_usage"] == {"input_tokens": 20, "output_tokens": 7, "cached_input_tokens": 12}
    assert row["task_id"] == "task-one" and row["physical_attempt_id"]
    # Two settled physical attempts: the discarded generation was paid for too.
    rows = ledger_rows(root)
    assert len(rows) >= 2 and all(row["state"] == "settled" for row in rows)


def test_second_request_drops_the_substituting_account_preference(setup):
    root, gateway, client = setup
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "earlier",
         "nativeContinuation": {"route": dict(ROUTE), "format": "codex.responses.v1",
                                "payload": [{"type": "reasoning"}]}},
        {"role": "user", "content": "second"},
    ]
    gateway.results = [substituted(), result(route=OTHER_ACCOUNT)]
    gateway.dispatch = ["response_received"] * 2
    client.chat(history, MODEL, None, "high")
    first, second = (payload for payload, _ in gateway.uploads)
    # The first request prefers the account the conversation ran on; the redo
    # names none, so the engine alone decides where it lands.
    assert first["account"] == {"mode": "auto", "preferredProfileId": "account-a"}
    assert second["account"] == {"mode": "auto"}


def test_exhausted_redos_refuse_instead_of_returning_the_other_model(setup, monkeypatch):
    root, gateway, client = setup
    monkeypatch.setattr(transport.config, "get_model_substitution_redos", lambda: 1)
    gateway.results = [substituted(), substituted(OTHER_ACCOUNT)]
    gateway.dispatch = ["response_received"] * 2
    with pytest.raises(transport.ClaudexorModelError) as raised:
        call(client)
    error = raised.value
    assert error.code == "model_substituted" and error.retryable is False
    assert error.problem["context"] == {"requested_model": "exact-model",
                                        "observed_model": "cheaper-model",
                                        "reason": "redos_exhausted", "redos": 1}
    assert "cheaper-model" in str(error) and error.route["credentialProfileId"] == "account-b"
    assert len(gateway.creates) == 2  # one redo, then the refusal
    assert [row["disposition"] for row in events(root, "model_served_mismatch")] == ["redo", "redos_exhausted"]


def test_no_redo_is_spent_on_a_pinned_account(setup, monkeypatch):
    root, gateway, client = setup
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"main": "account-a"}))
    gateway.results = [substituted()]
    with pytest.raises(transport.ClaudexorModelError) as raised:
        client.chat([{"role": "user", "content": "go"}], MODEL, None, "high", model_role="main")
    assert raised.value.problem["context"]["reason"] == "pinned_account"
    assert len(gateway.creates) == 1
    assert events(root, "model_served_mismatch")[0]["disposition"] == "pinned_account"


def test_no_redo_changes_the_bytes_of_an_admitted_candidate(setup):
    root, gateway, client = setup
    gateway.results = [substituted()]
    with ua.bind_physical_attempt_context(None, lambda request: True):
        with pytest.raises(transport.ClaudexorModelError) as raised:
            call(client)
    assert raised.value.problem["context"]["reason"] == "admitted_candidate"
    assert len(gateway.creates) == 1


def test_a_redo_never_spends_a_send_its_caller_still_needs(setup):
    root, gateway, client = setup
    gateway.results = [substituted()]
    # A packet review or a density probe owns its sends for its own rail: the
    # last one must stay available for the repair that rail is going to make.
    with ua.physical_attempt_limit(1):
        with pytest.raises(transport.ClaudexorModelError) as raised:
            call(client)
    assert raised.value.problem["context"]["reason"] == "send_budget_spent"
    assert len(gateway.creates) == 1
    assert events(root, "model_served_mismatch")[0]["disposition"] == "send_budget_spent"


def test_a_redo_uses_a_send_the_caller_can_still_spare(setup):
    root, gateway, client = setup
    gateway.results = [substituted(), result()]
    gateway.dispatch = ["response_received"] * 2
    with ua.physical_attempt_limit(2):
        message, _usage = call(client)
    assert message["content"] == "Ответ 🐍" and len(gateway.creates) == 2


def test_every_redo_of_one_call_carries_no_account_preference(setup, monkeypatch):
    """Not just the first redo: a single remembered profile let the second ask
    for the account that had already substituted."""
    root, gateway, client = setup
    monkeypatch.setattr(transport.config, "get_model_substitution_redos", lambda: 2)
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "earlier",
         "nativeContinuation": {"route": dict(ROUTE), "format": "codex.responses.v1",
                                "payload": [{"type": "reasoning"}]}},
        {"role": "user", "content": "second"},
    ]
    gateway.results = [substituted(), substituted(), result(route=OTHER_ACCOUNT)]
    gateway.dispatch = ["response_received"] * 3
    client.chat(history, MODEL, None, "high")
    accounts = [payload["account"] for payload, _ in gateway.uploads]
    assert accounts == [{"mode": "auto", "preferredProfileId": "account-a"},
                        {"mode": "auto"}, {"mode": "auto"}]


def test_a_redo_that_changes_account_spends_the_one_continuation_repair_and_recovers(setup, monkeypatch):
    """The real engine refuses a redo it routed to another account before sending it.

    The history's continuations are bound to the account that substituted, so the
    engine answers the redo with a no-start `invalid_continuation` naming the new
    account. That is the transport's ordinary one-shot repair, not a second
    substitution: it must neither spend a redo nor bring the old preference back,
    and the loop bound must still leave room for the redo after it.
    """
    root, gateway, client = setup
    monkeypatch.setattr(transport.config, "get_model_substitution_redos", lambda: 2)
    third = {**ROUTE, "credentialProfileId": "account-c", "accountFingerprint": "fingerprint-c"}
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "earlier",
         "nativeContinuation": {"route": dict(ROUTE), "format": "codex.responses.v1",
                                "payload": [{"type": "reasoning"}]}},
        {"role": "user", "content": "second"},
    ]
    refused = result(outcome="failed", route=OTHER_ACCOUNT,
                     problem={"code": "invalid_continuation", "message": "different account"})
    gateway.results = [substituted(), refused, substituted(OTHER_ACCOUNT), result(route=third)]
    gateway.dispatch = ["response_received", "not_started", "response_received", "response_received"]
    message, usage = client.chat(history, MODEL, None, "high")
    assert message["content"] == "Ответ 🐍" and len(gateway.creates) == 4
    sent = [payload for payload, _ in gateway.uploads]
    assert [payload["account"] for payload in sent] == [
        {"mode": "auto", "preferredProfileId": "account-a"}, {"mode": "auto"}, {"mode": "auto"}, {"mode": "auto"}]
    # The repair dropped the old account's continuation and kept the words.
    assert "nativeContinuation" in sent[1]["messages"][1]
    assert all("nativeContinuation" not in row for row in sent[2]["messages"] + sent[3]["messages"])
    assert sent[3]["messages"][1]["content"] == "earlier"
    assert [row["disposition"] for row in usage["claudexor"]["substituted"]] == ["redo", "redo"]
    assert [row["account"] for row in usage["claudexor"]["substituted"]] == ["account-a", "account-b"]
    assert len(events(root, "native_continuation_reset")) == 1
    assert len(events(root, "model_served_mismatch")) == 2


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_repair_after_a_redo_never_brings_the_account_preference_back(setup, monkeypatch, asynchronous):
    """A no-start repair rebuilds the request from the caller's parameters.

    Without the redo's decision carried on those parameters, the rebuilt request
    would read the history again and prefer the account that had just answered
    with the wrong model, and the engine honours a preference before its ranking.
    """
    from tests.test_processing_claudexor import receipt, refusal

    root, gateway, client = setup
    monkeypatch.setenv("OUROBOROS_PROCESSING_PREFERENCE", "")
    monkeypatch.setenv("OUROBOROS_MODEL_PROCESSING_PREFERENCES", "{}")
    monkeypatch.setattr(transport, "model_sources", lambda **kwargs: {
        "sources": [{"id": "codex", "processingPreferences": ["standard", "fast", "economy"]}]})
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "earlier",
         "nativeContinuation": {"route": dict(ROUTE), "format": "codex.responses.v1",
                                "payload": [{"type": "reasoning"}]}},
        {"role": "user", "content": "second"},
    ]
    gateway.results = [substituted(), refusal("fast", "unsupported"), {**result(), "processing": receipt()}]
    gateway.dispatch = ["response_received"] * 3
    kwargs = dict(messages=history, model=MODEL, processing_preference="fast")
    asyncio.run(client.chat_async(**kwargs)) if asynchronous else client.chat(**kwargs)
    accounts = [payload["account"] for payload, _ in gateway.uploads]
    assert accounts == [{"mode": "auto", "preferredProfileId": "account-a"}, {"mode": "auto"}, {"mode": "auto"}]
    assert gateway.uploads[2][0]["options"]["processingPreference"] == "standard"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_the_account_preference_is_dropped_on_both_transports(setup, asynchronous):
    root, gateway, client = setup
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "earlier",
         "nativeContinuation": {"route": dict(ROUTE), "format": "codex.responses.v1",
                                "payload": [{"type": "reasoning"}]}},
        {"role": "user", "content": "second"},
    ]
    gateway.results = [substituted(), result(route=OTHER_ACCOUNT)]
    gateway.dispatch = ["response_received"] * 2
    # The async loop offloads its blocking work to a thread, whose context is a
    # copy: a preference dropped there would never reach the next request.
    if asynchronous:
        asyncio.run(client.chat_async(history, MODEL, None, "high"))
    else:
        client.chat(history, MODEL, None, "high")
    first, second = (payload for payload, _ in gateway.uploads)
    assert first["account"] == {"mode": "auto", "preferredProfileId": "account-a"}
    assert second["account"] == {"mode": "auto"}


def test_only_the_generation_its_own_model_produced_teaches_a_density(setup):
    """The guard must fire on the discarded answer AND stay quiet on the real one.

    A negative assertion alone would pass with the whole guard deleted, and did:
    keying it off a host-side model comparison silenced EVERY Claudexor answer,
    because the request names the qualified route and the result names the bare
    wire id. The engine's own fact is the only honest key.
    """
    from ouroboros import capability_evidence

    root, gateway, client = setup
    gateway.results = [substituted(), result()]
    gateway.dispatch = ["response_received"] * 2
    stamped, original = [], capability_evidence.observe_token_density
    try:
        capability_evidence.observe_token_density = lambda request, usage, **kw: stamped.append(
            (usage.get("claudexor") or {}).get("served_other_model"))
        call(client)
    finally:
        capability_evidence.observe_token_density = original
    # The transport marks the discarded generation and only that one; the guard
    # reads the engine's own fact, never a model string the host compared.
    assert stamped == [True, None]

    recorded = []
    request = SimpleNamespace(provider="claudexor", model="claudexor::codex=exact-model",
                              prompt_tokens_bounded_estimate=500, prompt_tokens_estimate=500,
                              drive_root=root, physical_context=None)
    monkey = capability_evidence.record_token_density
    try:
        capability_evidence.record_token_density = lambda *a, **kw: recorded.append(kw["prompt_tokens"])
        capability_evidence.observe_token_density(
            request, {"input_tokens": 2000}, drive_root_resolver=lambda value: value)
        capability_evidence.observe_token_density(
            request, {"input_tokens": 2000, "claudexor": {"served_other_model": True}},
            drive_root_resolver=lambda value: value)
    finally:
        capability_evidence.record_token_density = monkey
    # The guard must fire on the substituted row AND stay quiet on the ordinary
    # one: a negative assertion alone passes with the whole guard deleted.
    assert recorded == [2000]


def test_a_failed_acknowledgement_is_disclosed_rather_than_assumed(setup):
    root, gateway, client = setup
    gateway.results = [substituted(), result()]
    gateway.dispatch = ["response_received"] * 2
    gateway.ack_error = RuntimeError("ACK reply lost")
    message, usage = call(client)
    assert message["content"] == "Ответ 🐍"  # the round still recovers
    custody = usage["claudexor"]["substituted"][0]["result_custody"]
    assert custody["state"] == "pending" and custody["reason"] == "RuntimeError"
    assert events(root, "model_served_mismatch")[0]["result_custody"]["reason"] == "RuntimeError"


def test_a_redo_never_starts_once_the_owner_window_is_spent(setup, monkeypatch):
    from ouroboros import model_wait

    root, gateway, client = setup
    gateway.results = [substituted()]
    now = [100.0]
    monkeypatch.setattr(model_wait, "monotonic_now", lambda slot=None: now[0])
    read_result = gateway.get_model_result

    def expire_on_response(*args, **kwargs):
        response = read_result(*args, **kwargs)
        now[0] = 111.0  # the first paid response consumes the window, regardless of control polls
        return response

    monkeypatch.setattr(gateway, "get_model_result", expire_on_response)
    with model_wait.execution_deadline_scope(110.0):
        with pytest.raises(transport.ClaudexorModelError) as raised:
            call(client)
    assert raised.value.problem["context"]["reason"] == "deadline_spent"
    assert len(gateway.creates) == 1
    assert len(gateway.acks) == 1
    assert events(root, "model_served_mismatch")[0]["disposition"] == "deadline_spent"


def test_an_unknown_outcome_is_never_treated_as_a_substitution(setup):
    root, gateway, client = setup
    substitute = substituted()
    gateway.results = [substitute]
    gateway.dispatch = ["unknown"]
    with pytest.raises(transport.ClaudexorModelError) as raised:
        call(client)
    # An operation whose outcome nobody knows must not become a discarded
    # answer: the rail would be re-asking a round that may still be running.
    assert raised.value.code == "model_outcome_unknown"
    assert not events(root, "model_served_mismatch") and len(gateway.creates) == 1


def test_no_outer_paid_repeat_can_nest_another_redo_budget_on_this_transport():
    """The rail's budget cannot be multiplied by the transport-death repeat.

    A reviewer read the two bounded budgets as a product. They never meet: this
    transport's route is loopback by construction, and a loopback route is
    excluded from the paid repeat class, so no outer repeat starts a second
    call whose redos begin again.
    """
    from ouroboros.transport_custody import is_loopback_base_url, is_retryable_transport_death

    target = LLMClient()._resolve_remote_target(MODEL)
    assert target["provider"] == "claudexor" and is_loopback_base_url(target["base_url"])
    import httpx

    def died(loopback):
        # A typed transport death on a dispatched send: the ONLY thing left to
        # decide the verdict is the route's locality.
        error = transport.ClaudexorModelError({"code": "model_operation_failed", "message": "died"})
        error.__cause__ = httpx.ReadError("connection reset")
        error.physical_attempt_capture = SimpleNamespace(
            state="dispatched", provider="claudexor", route_is_loopback=loopback)
        return error

    assert is_retryable_transport_death(died(False)) is True
    assert is_retryable_transport_death(died(True)) is False


def test_a_discarded_generation_never_becomes_the_live_turn(setup, monkeypatch):
    root, gateway, client = setup
    monkeypatch.setattr(transport, "owned_engine_version", lambda: "3.12.5")
    slot = transport.ModelTurnState()
    gateway.results = [substituted(), result()]
    gateway.dispatch = ["response_received"] * 2
    gateway.results[0]["nativeContinuation"] = {
        "route": dict(ROUTE), "format": "codex.turn.v1", "payload": {"turnState": "discarded"}}
    gateway.results[1]["nativeContinuation"] = {
        "route": dict(ROUTE), "format": "codex.turn.v1", "payload": {"turnState": "accepted"}}
    client.chat([{"role": "user", "content": "go"}], MODEL, None, "high", model_turn_state=slot)
    # Only the accepted generation owns the live turn; the discarded one would
    # have carried the substituting route straight back into the redo.
    assert slot.envelope == gateway.results[1]["nativeContinuation"]


def test_an_incomplete_answer_from_another_model_is_refused_too(setup, monkeypatch):
    root, gateway, client = setup
    monkeypatch.setattr(transport.config, "get_model_substitution_redos", lambda: 0)
    gateway.results = [substituted(outcome="incomplete")]
    gateway.dispatch = ["response_received"]
    with pytest.raises(transport.ClaudexorModelError) as raised:
        call(client)
    assert raised.value.code == "model_substituted"
    assert raised.value.problem["context"]["reason"] == "redos_exhausted"


def test_an_ordinary_answer_is_untouched_and_an_unknown_model_is_not_a_mismatch(setup):
    root, gateway, client = setup
    plain = result()
    plain["modelMismatch"] = None
    gateway.results = [plain]
    message, usage = call(client)
    assert message["content"] == "Ответ 🐍" and "substituted" not in usage["claudexor"]
    assert len(gateway.creates) == 1 and not events(root, "model_served_mismatch")


def test_the_refusal_is_its_own_kind_that_rotates_without_cooling_the_model():
    error = transport.ClaudexorModelError(
        {"code": "model_substituted", "message": "another model answered",
         "context": {"requested_model": "a", "observed_model": "b"}})
    classified = classify_llm_exception(error)
    assert classified.kind == "model_substituted" and classified.retry_same_request is False
    assert classified.provider_code == "model_substituted"
    # A substituting account must never cool the model the owner asked for, and
    # the configured chain stays available once the redos are spent.
    assert "model_substituted" not in _COOLDOWN_ERROR_KINDS
    assert fallback_chain_allowed(object(), "model_substituted", None, {}) is True


@pytest.mark.parametrize("usage, expected", [
    ({"_last_llm_error_kind": "model_substituted"}, "different model than the one requested"),
    ({"_last_llm_error_kind": "model_substituted"}, "ranks this one lower"),
    ({"_last_llm_error_kind": "bad_request", "_last_llm_provider_code": "invalid_continuation"},
     "refused the stored continuation"),
    ({"_last_llm_error_kind": "bad_request", "_last_llm_provider_code": "unsupported_parameter"},
     "rejected the request shape"),
])
def test_the_owner_terminal_names_the_real_refusal(usage, expected):
    assert expected in provider_recovery_hint(usage)


def test_a_same_route_refusal_is_not_asked_a_second_time():
    for usage in ({"_last_llm_error_kind": "model_substituted"},
                  {"_last_llm_error_kind": "bad_request", "_last_llm_provider_code": "invalid_continuation"}):
        source, wall = provider_no_call_source(usage, deadline_exhausted=False)
        assert source == "same_route_refusal_no_resend" and wall is False
    # An ordinary refusal keeps its one salvage call.
    assert provider_no_call_source({"_last_llm_error_kind": "bad_request"}, deadline_exhausted=False) == ("", False)


def test_the_owner_row_names_what_happened_once_per_task_and_model():
    notes = []
    usage = {"_model_substitutions": [
        {"requested": "exact-model", "observed": "cheaper-model", "account": "account-a",
         "disposition": "redo"},
        {"requested": "exact-model", "observed": "cheaper-model", "account": "account-b",
         "disposition": "redo"},
    ]}
    for _ in range(2):
        emit_model_substitution(usage, task_id="task-one",
                                emit_progress=lambda text, **meta: notes.append((text, meta)))
    text, meta = notes[0]
    assert len(notes) == 1
    assert "cheaper-model answered instead of the requested exact-model" in text
    assert "Claudexor account account-a" in text
    # The host stops preferring an account; it never claims the round moved.
    assert "asked again without naming an account" in text
    assert "another account" not in text
    assert meta == {"card_row": "timeline", "card_row_id": "task-one:model_substitution:exact-model"}


def test_a_refusal_row_does_not_claim_the_round_was_redone():
    notes = []
    emit_model_substitution(
        {"_model_substitutions": [{"requested": "a", "observed": "b", "account": "",
                                   "disposition": "pinned_account"}]},
        task_id="task-one", emit_progress=lambda text, **meta: notes.append(text))
    assert "the account is pinned, so the round was not asked again" in notes[0]
    assert "Claudexor account" not in notes[0]


def test_dropping_one_source_continuation_keeps_every_other_message_intact():
    messages = [
        {"role": "user", "content": "ask"},
        {"role": "assistant", "content": "said", "tool_calls": [{"id": "a"}],
         "nativeContinuation": {"route": {"source": "codex"}, "payload": [{"type": "reasoning"}]}},
        {"role": "assistant", "content": "other engine",
         "nativeContinuation": {"route": {"source": "other"}, "payload": []}},
    ]
    original = deepcopy(messages)
    prepared, changed = drop_source_native_messages(messages, source="codex")
    assert messages == original  # the caller's history is never mutated in place
    assert len(changed) == 1 and changed[0]["old_route"] == {"source": "codex"}
    assert "nativeContinuation" not in prepared[1] and prepared[1]["tool_calls"] == [{"id": "a"}]
    assert prepared[1]["content"] == "said" and "nativeContinuation" in prepared[2]
    assert drop_source_native_messages(prepared, source="codex")[1] == []
