"""A released ``auth_changed`` receipt is asked once more as a new operation on the same route.

The engine proves the request never left (``dispatch.state=not_started``) because the account's
authentication changed. The transport sends the same request ONCE more as a new operation (new
attempt, idempotency key and operation id; same route, account choice, pin and effort) inside the
existing no-start repair chain. A second refusal, an unknown outcome, an unreleased attempt and every
other code keep the caller's previous path: the classifier and the configured routes.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from ouroboros import llm_claudexor as transport
from ouroboros import usage_accounting as ua
from ouroboros.loop_llm_call import classify_llm_exception
from tests.test_llm_claudexor import MODEL, ROUTE, ledger, result, setup as setup_fixture
from tests.test_model_substitution_rail import substituted
from tests.test_processing_claudexor import receipt, refusal

setup = setup_fixture
AUTH = {"code": "auth_changed", "message": "The account's authentication changed; retry the operation.",
        "context": {"generationStarted": False}}


def auth_refusal():
    return result(outcome="failed", problem=AUTH)


def send(client, asynchronous, **kwargs):
    messages = [{"role": "user", "content": "hi"}]
    if asynchronous:
        return asyncio.run(client.chat_async(messages, MODEL, **kwargs))
    return client.chat(messages, MODEL, **kwargs)


def events(root, kind):
    path = root / "logs" / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    return [row for row in rows if row.get("type") == kind]


@pytest.mark.parametrize(("pin", "asynchronous"), [("", False), ("account-a", True)])
def test_released_auth_change_answers_from_one_new_operation_on_the_same_route(setup, asynchronous, pin):
    root, gateway, client = setup
    gateway.results, gateway.dispatch = [auth_refusal(), result()], ["not_started", "response_received"]
    message, usage = send(client, asynchronous, reasoning_effort="high", model_account_override=pin)
    assert message["content"] == "Ответ 🐍"
    first, second = (payload for payload, _key in gateway.uploads)
    assert first == second and "retryOfOperationId" not in second
    assert second["account"] == ({"mode": "pin", "profileId": pin} if pin else {"mode": "auto"})
    assert second["options"]["reasoningEffort"] == "high" and second["model"] == "exact-model"
    assert gateway.creates[0] != gateway.creates[1] and len(gateway.accepted_operations) == 2
    assert len(set(usage["ledger_attempt_ids"])) == 2 and usage["claudexor"]["operation_id"] == "op-1"
    assert [row["state"] for row in ledger(root)] == ["released", "settled"]
    [event] = events(root, "model_operation_reissued")
    assert (event["cause"], event["operation_id"], event["task_id"]) == ("auth_changed", "op-0", "task-one")


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_reissue_whose_next_attempt_cannot_be_prepared_is_not_recorded(setup, monkeypatch, asynchronous):
    """The reissue is recorded once its next attempt is prepared: a preparation that refuses (a helper without a wait
    owner inside a physical attempt, ``model_wait_reprepare_required``) leaves no second operation and no
    ``model_operation_reissued``."""
    from ouroboros.model_wait import ModelWaitInterrupted

    root, gateway, client = setup
    gateway.results, gateway.dispatch = [auth_refusal(), result()], ["not_started", "response_received"]

    def refuse(_target, _payload, parameters, error):
        raise ModelWaitInterrupted("model_wait_reprepare_required", role=parameters.get("model_role", ""), cause=error)

    monkeypatch.setattr(transport, "_unstarted_retry_preparation", refuse)
    with pytest.raises(ModelWaitInterrupted) as raised:
        send(client, asynchronous)
    assert raised.value.control_reason == "model_wait_reprepare_required"
    assert len(gateway.accepted_operations) == 1 and not events(root, "model_operation_reissued")


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_second_auth_change_keeps_the_previous_refusal(setup, asynchronous):
    root, gateway, client = setup
    gateway.results, gateway.dispatch = [auth_refusal()] * 3, ["not_started"] * 3
    with pytest.raises(transport.ClaudexorModelNotDispatched) as raised:
        send(client, asynchronous)
    assert raised.value.code == "auth_changed" and raised.value.operation_id == "op-1"
    assert raised.value.presence_all_operations_not_started is True
    assert len(gateway.accepted_operations) == 2
    assert [row["state"] for row in ledger(root)] == ["released", "released"]
    classified = classify_llm_exception(raised.value)
    assert (classified.kind, classified.retry_same_request) == ("bad_request", False)


@pytest.mark.parametrize(("case", "asynchronous"), [
    ("unknown_outcome", False),
    ("unknown_dispatch", True),
    ("unreleased", False),
    ("other_code", True),
])
def test_only_a_released_typed_auth_change_is_asked_again(setup, monkeypatch, asynchronous, case):
    root, gateway, client = setup
    value, dispatch = auth_refusal(), "not_started"
    if case == "unknown_outcome":
        value["outcome"], dispatch = "unknown", "response_received"
    if case == "unknown_dispatch":
        dispatch = "unknown"
    if case == "unreleased":
        monkeypatch.setattr(ua, "release_pre_dispatch_attempt", lambda *_: False)
    if case == "other_code":
        value["problem"] = {**AUTH, "code": "auth_unavailable"}
    gateway.results, gateway.dispatch = [value, result()], [dispatch, "response_received"]
    with pytest.raises(transport.ClaudexorModelError) as raised:
        send(client, asynchronous)
    assert len(gateway.accepted_operations) == 1 and not events(root, "model_operation_reissued")
    if case.startswith("unknown"):
        assert raised.value.code == "model_outcome_unknown"
    if case == "unreleased":
        assert raised.value.physical_attempt_capture.state == "unresolved"


def test_a_pause_after_the_refusal_sends_nothing_more(setup):
    _root, gateway, client = setup
    gateway.results, gateway.dispatch = [auth_refusal(), result()], ["not_started", "response_received"]
    with pytest.raises(transport.ClaudexorModelNotDispatched) as raised:
        send(client, False, model_poll_control=lambda: "owner_paused" if gateway.accepted_operations else None)
    assert raised.value.control_reason == "owner_paused" and len(gateway.accepted_operations) == 1


@pytest.mark.parametrize("order", [("auth", "native"), ("native", "auth"), ("processing", "native", "auth")])
def test_each_cause_keeps_its_one_repair_in_the_same_bounded_loop(setup, monkeypatch, order):
    root, gateway, client = setup
    monkeypatch.setattr(transport, "model_sources", lambda **_kw: {
        "sources": [{"id": "codex", "processingPreferences": ["standard", "fast", "economy"]}]})
    native = result(outcome="failed", route={**ROUTE, "credentialProfileId": "account-b"},
                    problem={"code": "invalid_continuation", "message": "Account changed"})
    steps = {"auth": (auth_refusal(), "not_started"), "native": (native, "not_started"),
             "processing": (refusal(), "response_received")}
    gateway.results = [steps[cause][0] for cause in order] + [{**result(), "processing": receipt()}]
    gateway.dispatch = [steps[cause][1] for cause in order] + ["response_received"]
    client.chat([result()["message"]], MODEL, processing_preference="economy")
    assert len(gateway.accepted_operations) == len(order) + 1
    states = [row["state"] for row in {row["attempt_id"]: row for row in ledger(root)}.values()]
    assert states == ["released"] * len(order) + ["settled"]


def test_a_discarded_answer_ends_presence_no_start_but_not_the_repair(setup, monkeypatch):
    """Presence's all-not-started fact only accumulates; the repair reads this refusal's own receipt."""
    _root, gateway, client = setup
    monkeypatch.setenv("OUROBOROS_SERVED_MODEL_REDOS", "1")
    gateway.results = [substituted(), auth_refusal(), auth_refusal()]
    gateway.dispatch = ["response_received", "not_started", "not_started"]
    with pytest.raises(transport.ClaudexorModelNotDispatched) as raised:
        send(client, False)
    assert len(gateway.accepted_operations) == 3
    assert raised.value.presence_all_operations_not_started is False
