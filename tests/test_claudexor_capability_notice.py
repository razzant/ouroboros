"""The engine-granted capability retry and the owner's account/model notice, from typed facts.

Pure host behavior: problem shapes follow the engine contract
(``docs/MODEL_CAPABILITY_ROUTING.md`` in Claudexor); the recorded-wire and
real-engine suites cover the same paths end to end.
"""
from __future__ import annotations

import copy
import queue
from types import SimpleNamespace

import pytest

from ouroboros import model_wait
from ouroboros.llm_substitution import CapabilityRetry, capability_notice

SINCE_A, SINCE_B = "2026-10-21T02:01:00.000Z", "2026-10-21T02:03:00.000Z"


def refusal(*, code="model_capability_refused", mode="auto", own="a", remains=False, marks=("a", "b"), **context):
    rows = [{"credentialProfileId": pid, "accountFingerprint": f"fp-{pid}",
             "providerReason": f"{pid} cannot serve this model", "since": {"a": SINCE_A, "b": SINCE_B}[pid],
             "state": "provisional"} for pid in marks]
    return {"code": code, "message": "refused", "retryable": False, "context": {
        "source": "codex", "requestedModel": "gpt-6.1-sol", "accountMode": mode,
        "credentialProfileId": own, "accountFingerprint": f"fp-{own}", "providerReason": f"{own} cannot serve this model",
        "generationStarted": False, "retryOnOtherAccount": False, "operationId": "op-1",
        "nextStep": "Check the requested model and account plan, refresh account catalogs, or choose another model.",
        "accountsTried": [{"credentialProfileId": own, "providerReason": f"{own} cannot serve this model"}],
        "modelMarks": rows, **({"alternativeRemains": remains} if mode == "auto" else {}), **context}}


class TestNoticeGate:
    def test_every_account_marked_speaks_once_with_model_accounts_and_next_step(self):
        delivery_id, text = capability_notice(refusal())
        assert delivery_id.startswith("model-capability:")
        lines = text.splitlines()
        assert lines[0] == "No account can serve gpt-6.1-sol right now."
        assert lines[1].startswith("a: a cannot serve this model Temporary mark since 2026-10-21")
        assert lines[2].startswith("b: b cannot serve this model Temporary mark since 2026-10-21")
        assert lines[3].startswith("Check the requested model")

    def test_a_pinned_capability_refusal_names_its_account(self):
        delivery_id, text = capability_notice(refusal(mode="pin", marks=()))
        assert text.splitlines()[:2] == ["The pinned account a cannot serve gpt-6.1-sol.", "a: a cannot serve this model"]
        assert capability_notice(refusal(mode="pin", marks=("b",)))[0] == delivery_id

    @pytest.mark.parametrize("case", [
        dict(remains=True),                       # another account may serve
        dict(own="c", marks=("a", "b")),          # the refusing account itself is unmarked
        dict(marks=()),                           # no account is blamed
        dict(mode="pin", code="invalid_request", marks=()),  # a pin's ordinary refusal
        dict(requestedModel=None),
    ], ids=["alternative", "unmarked_refuser", "no_marks", "pin_generic", "no_model"])
    def test_silent_unless_typed_facts_say_no_account_can_serve(self, case):
        assert capability_notice(refusal(**case)) is None

    def test_an_unrecognized_refusal_counts_once_its_account_is_marked(self):
        assert capability_notice(refusal(code="invalid_request", capabilityKind="unclassified")) is not None

    def test_a_step_addressed_to_the_sender_becomes_the_owners(self):
        from ouroboros.llm_substitution import OWNER_STEP

        sender = "The history carries native continuation bound to this account; start a new conversation."
        text = capability_notice(refusal(nextStep=sender, retryBlockedBy="native_continuation"))[1]
        assert text.splitlines()[-1] == OWNER_STEP and sender not in text
        assert capability_notice(refusal(nextStep=sender))[1].splitlines()[-1] == sender


class TestEpisodeIdentity:
    def test_order_and_reason_text_do_not_change_the_episode(self):
        first = refusal()
        second = refusal(marks=("b", "a"))
        for mark in second["context"]["modelMarks"]:
            mark["providerReason"] += " (reworded)"
        assert capability_notice(first)[0] == capability_notice(second)[0]

    def test_a_new_account_or_a_returning_failure_is_a_new_episode(self):
        base = capability_notice(refusal())[0]
        assert capability_notice(refusal(marks=("a",)))[0] != base
        returned = refusal()
        returned["context"]["modelMarks"][0]["since"] = "2026-10-21T07:00:00.000Z"
        assert capability_notice(returned)[0] != base


def invocation(payload=None, operation_id="op-1"):
    return SimpleNamespace(payload=payload or {"account": {"mode": "auto"}, "messages": [{"role": "user", "content": "x"}]},
                           operation_id=operation_id, root=None)


def error(problem, *, state="settled"):
    from ouroboros.llm_claudexor import ClaudexorModelError

    fault = ClaudexorModelError(problem, operation_id=problem["context"].get("operationId"))
    fault.physical_attempt_capture = SimpleNamespace(state=state, attempt_id="attempt-1")
    return fault


class TestRetryGrant:
    @pytest.mark.parametrize("code", ["model_capability_refused", "invalid_request", "provider_refused"])
    def test_the_typed_grant_alone_authorizes_one_identical_auto_operation(self, code, monkeypatch):
        policy = CapabilityRetry()
        monkeypatch.setattr(policy, "announce", lambda *_: pytest.fail("a granted retry announces nothing"))
        assert not policy.ends_chain(error(refusal(code="rate_limited")))
        call = invocation({"account": {"mode": "auto", "preferredProfileId": "a"},
                           "messages": [{"role": "user", "content": "x"}]})
        updated = policy.refused(call, error(refusal(code=code, retryOnOtherAccount=True)))
        assert updated == {**copy.deepcopy(call.payload), "account": {"mode": "auto"}, "retryOfOperationId": "op-1"}
        # After the one correlated operation nothing else re-asks: no rotation stacks on it.
        assert policy.ends_chain(error(refusal(code="rate_limited")))

    @pytest.mark.parametrize("bound", [lambda request: None, lambda request: False], ids=["fit", "rejecting"])
    def test_a_bound_send_predicate_does_not_withhold_it(self, bound, monkeypatch):
        """Every task round binds its context-fit check for the whole chat; the identical
        resend is judged by that check again at its own dispatch, so it is not withheld."""
        from ouroboros.usage_accounting import bind_physical_attempt_context

        policy = CapabilityRetry()
        monkeypatch.setattr(policy, "announce", lambda *_: pytest.fail("a granted retry announces nothing"))
        with bind_physical_attempt_context(None, candidate_predicate=bound):
            assert policy.refused(invocation(), error(refusal(retryOnOtherAccount=True)))["retryOfOperationId"] == "op-1"

    @pytest.mark.parametrize("guard", ["no_grant", "generation_started", "unsettled", "unknown",
                                       "already_correlated", "pin", "other_operation", "second"])
    def test_host_custody_facts_withhold_it(self, guard, monkeypatch):
        problem = refusal(code="invalid_request", retryOnOtherAccount=guard != "no_grant",
                          generationStarted=guard == "generation_started")
        payload = {"account": {"mode": "pin", "profileId": "a"} if guard == "pin" else {"mode": "auto"},
                   "messages": [], **({"retryOfOperationId": "op-0"} if guard == "already_correlated" else {})}
        policy = CapabilityRetry()
        if guard == "second":
            policy.original = error(refusal())
        announced = []
        monkeypatch.setattr(policy, "announce", lambda _call, fault: announced.append(fault))
        fault = error(problem, state="unresolved" if guard == "unsettled" else "settled")
        if guard == "unknown":  # an unknown outcome may have generated: never re-sent
            fault.code = "model_outcome_unknown"
        call = invocation(payload, operation_id="op-9" if guard == "other_operation" else "op-1")
        assert policy.refused(call, fault) is None
        assert announced == [fault]

    def test_an_invalid_correlation_returns_the_original_refusal_unannounced(self, monkeypatch):
        policy = CapabilityRetry()
        monkeypatch.setattr(policy, "announce", lambda *_: pytest.fail("no notice on model_retry_invalid"))
        first = error(refusal(retryOnOtherAccount=True))
        assert policy.refused(invocation(), first) is not None
        with pytest.raises(type(first)) as raised:
            policy.refused(invocation(operation_id="op-2"), error(refusal(code="model_retry_invalid")))
        assert raised.value is first and first.ledger_attempt_ids == ["attempt-1"]


class TestOwnerDelivery:
    def scope(self, tmp_path, **task):
        from ouroboros.task_results import write_task_result

        write_task_result(tmp_path, "helper", "running")
        events = queue.Queue()
        return events, model_wait.task_model_wait_scope(
            task={"id": "helper", "chat_id": 777, "title": "Draft the outline", **task},
            drive_root=tmp_path, event_queue=events, worker_slot_held=False)

    def test_one_owed_row_to_the_owner_chat_whatever_the_helper_room(self, tmp_path, monkeypatch):
        from supervisor import schedule_notes
        from supervisor.terminal_delivery import ENQUEUE_ALREADY_DELIVERED, pending_deliveries, register_delivery

        # The owner's bound chat (Main for a web owner), resolved by the reminders' owner.
        monkeypatch.setattr(schedule_notes, "owner_chat_id", lambda root: 42 if root == tmp_path else 0)
        events, scope = self.scope(tmp_path)
        with scope:
            for _ in range(16):  # a burst of helpers with the same episode
                model_wait.announce_capability_refusal(error(refusal()), drive_root=tmp_path)
        sent = [events.get_nowait() for _ in range(events.qsize())]
        assert {event["delivery_id"] for event in sent} == {capability_notice(refusal())[0]}
        event = sent[0]
        assert (event["chat_id"], event["task_id"], event["role"], event["system_type"]) == (
            42, "", "system", "model_capability_notice")
        assert event["text"].endswith("\nFirst seen in Draft the outline.")
        # Owed before the live send: a supervisor restart replays the same row.
        assert [row["delivery_id"] for row in pending_deliveries(tmp_path)] == [event["delivery_id"]]
        register_delivery(tmp_path, event["delivery_id"], emitted=event)
        assert pending_deliveries(tmp_path) == []
        from supervisor.terminal_delivery import enqueue_terminal_delivery_outcome
        assert enqueue_terminal_delivery_outcome(tmp_path, event, event_queue=queue.Queue()) == ENQUEUE_ALREADY_DELIVERED

    def test_a_project_helper_is_named_by_project_and_task_in_the_owner_chat(self, tmp_path, monkeypatch):
        from ouroboros.contracts.chat_id_policy import project_chat_id
        from ouroboros.projects_registry import create_project
        from supervisor import schedule_notes

        monkeypatch.setattr(schedule_notes, "owner_chat_id", lambda root: 1)
        create_project(tmp_path, "launch", name="Launch Plan")
        events, scope = self.scope(tmp_path, project_id="launch", chat_id=project_chat_id("launch"))
        with scope:
            model_wait.announce_capability_refusal(error(refusal()), drive_root=tmp_path)
        event = events.get_nowait()
        assert (event["chat_id"], event["task_id"]) == (1, "")
        assert event["text"].endswith("\nFirst seen in Launch Plan › Draft the outline.")

    def test_a_label_that_cannot_be_read_does_not_drop_the_notice(self, tmp_path, monkeypatch):
        from ouroboros import projects_registry
        from supervisor import schedule_notes

        def unreadable(*_args, **_kwargs):
            raise OSError("registry unreadable")

        monkeypatch.setattr(schedule_notes, "owner_chat_id", lambda root: 1)
        monkeypatch.setattr(projects_registry, "task_presentation_snapshot", unreadable)
        events, scope = self.scope(tmp_path)
        with scope:
            model_wait.announce_capability_refusal(error(refusal()), drive_root=tmp_path)
        event = events.get_nowait()
        assert event["delivery_id"] == capability_notice(refusal())[0] and "First seen" not in event["text"]

    def test_a_bare_library_caller_has_no_chat_side_effect(self, tmp_path):
        from supervisor.terminal_delivery import pending_deliveries

        model_wait.announce_capability_refusal(error(refusal()), drive_root=tmp_path)
        assert pending_deliveries(tmp_path) == []
