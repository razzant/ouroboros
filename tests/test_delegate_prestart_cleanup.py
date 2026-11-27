"""Fresh pre-POST interruptions retire only their own disposable registration."""
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import json

import pytest

from ouroboros import delegate_custody as custody, owner_pause, queue_observation
from ouroboros.gateways import claudexor as gateway_module
from ouroboros.llm_attempt import PhysicalDispatchInterrupted
from ouroboros.model_wait import task_model_wait_scope
from ouroboros.task_results import write_task_result
from ouroboros.tools import delegate
from tests._bound_registration import StartRegistry, enable_start_registry
from tests._delegated_transport_shared import (
    _LiveRunStub, _nanny_ctx, _select_transport_access,
    _owned_gateway_uses_each_test_transport as _owned_gateway_fixture,
)
from tests._review_session_route_shared import (
    FakeGateway, _run_session_directly, fake_route as _fake_route,
)
from tests._store_journal_shared import use_journal
from tests.test_delegated_start_contention import hold_request

_owned_gateway_uses_each_test_transport = _owned_gateway_fixture
fake_route = _fake_route


@pytest.fixture(autouse=True)
def bound_review_registry(monkeypatch):
    enable_start_registry(monkeypatch, FakeGateway)


def legacy_retry_store(monkeypatch):
    for gateway in (FakeGateway, ToolGateway):
        monkeypatch.setattr(gateway, "store_identity", "")


class ToolGateway(StartRegistry, _LiveRunStub):
    def __init__(self, *, shared=False):
        super().__init__()
        self.shared = shared
        self.registrations, self.removals, self.start_requests = [], [], []
        self.start_error = None

    def find_project_id(self, root):
        return "proj-existing" if self.shared else ""

    def register_project(self, root):
        self.registrations.append(root)
        return "proj-new"

    def start_run(self, request, *, idempotency_key=""):
        self.start_requests.append((request, idempotency_key))
        if self.start_error:
            raise self.start_error
        return {"runId": self.run_id}


@pytest.fixture(params=["review", "tool"])
def start_case(request, tmp_path, monkeypatch, fake_route):
    write_task_result(tmp_path, "t-b", "running", root_task_id="t-b")
    repo = tmp_path / "repo"
    repo.mkdir()
    state = {}
    if request.param == "review":
        FakeGateway.project_unregistered = True
        return (
            lambda: _run_session_directly(tmp_path, root=str(repo), retry_state=state),
            lambda: FakeGateway.instances[-1],
            lambda: setattr(FakeGateway, "project_unregistered", False),
            lambda error: setattr(FakeGateway, "start_error", error),
            state,
        )
    stub = ToolGateway()
    _select_transport_access(monkeypatch, "readonly")
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setattr(gateway_module, "ClaudexorGateway", lambda *a, **k: stub)
    ctx = _nanny_ctx(tmp_path, "t-b")

    def start():
        result = delegate._delegate_start(ctx, "inspect the source", **(
            {"retry_of": state["pending_invocation_id"]} if state else {}))
        payload = json.loads(result.text)
        if payload.get("pending_invocation_id"):
            state["pending_invocation_id"] = payload["pending_invocation_id"]
        return result

    return start, lambda: stub, lambda: setattr(stub, "shared", True), lambda error: setattr(stub, "start_error", error), state


def failures(root):
    path = custody.event_log_path(root)
    return [row for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and (row := json.loads(line)).get("type") == custody.START_FAILED]


@pytest.mark.parametrize(("control", "journal"), [("stop", "WAL"), ("deadline", "WAL"), ("closed", "PERSIST")])
def test_interrupted_unsent_claim_retires_fresh_registration(start_case, tmp_path, monkeypatch, control, journal):
    use_journal(monkeypatch, journal)  # a held write fails its COMMIT (rollback journal), its BEGIN (WAL)
    start, gateway, _shared, _fail, _state = start_case
    cancelled = []
    with task_model_wait_scope(task={"id": "t-b", "_attempt": 1}, drive_root=tmp_path,
            event_queue=None, worker_slot_held=False,
            owner_control=lambda: "cancelled" if cancelled else None) as owner, ExitStack() as stack:
        release, events = hold_request(monkeypatch, stack, tmp_path)

        def wait(_seconds):
            assert gateway().start_requests == []
            release[0]()
            if control == "stop":
                cancelled.append(True)
            elif control == "deadline":
                owner.task["deadline_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
            else:
                owner.closed = True

        monkeypatch.setattr(queue_observation, "_sleep", wait)
        with pytest.raises((PhysicalDispatchInterrupted, owner_pause.OwnerPauseRefused)):
            start()
    assert len(events) == 1 and gateway().start_requests == []
    assert gateway().removals == ["proj-new"]
    row = failures(tmp_path)[-1]
    assert row["definite"] is True
    [release] = [r for r in custody.custody_rows(tmp_path) if r["type"] == custody.PROJECT_RETIRED]
    assert release["project_id"] == row["project_id"] and release["removal_outcome"] == "removed"
    assert row["invocation_id"] == events[0]["invocation_id"]


def test_preclaim_exception_keeps_shared_registration(start_case, tmp_path, monkeypatch):
    start, gateway, shared, _fail, _state = start_case
    shared()
    error = PhysicalDispatchInterrupted("cancelled")
    monkeypatch.setattr(custody, "record_start_requested", lambda *a, **k: (_ for _ in ()).throw(error))
    with pytest.raises(PhysicalDispatchInterrupted) as caught:
        start()
    assert caught.value is error
    assert gateway().removals == [] and gateway().start_requests == []
    row = failures(tmp_path)[-1]
    # The row names its task on both start paths (review registration and the delegate tool), so
    # the task feed can show it; the review path wrote it anonymously until the 05–08.11 audit.
    assert row["definite"] is True and row["task_id"] == "t-b" and row["invocation_id"]


def test_recovered_unknown_start_is_not_retired_by_new_preclaim_error(start_case, tmp_path, monkeypatch):
    legacy_retry_store(monkeypatch)
    start, gateway, _shared, fail, state = start_case
    fail(gateway_module.ClaudexorUnavailable("daemon_unreachable", "POST response lost"))
    try:
        start()
    except gateway_module.ClaudexorUnavailable:
        pass
    token = state["pending_invocation_id"]
    original = custody.invocation_record(tmp_path, token)
    assert original["state"] == "pending"
    error = PhysicalDispatchInterrupted("cancelled")
    monkeypatch.setattr(custody, "record_start_requested", lambda *a, **k: (_ for _ in ()).throw(error))
    with pytest.raises(PhysicalDispatchInterrupted) as caught:
        start()
    assert caught.value is error and gateway().removals == []
    assert custody.invocation_record(tmp_path, token)["request"] == original["request"]
    assert failures(tmp_path)[-1]["definite"] is False


def test_cleanup_failure_does_not_replace_original_control(start_case, monkeypatch):
    from ouroboros import review_execution
    start, _gateway, _shared, _fail, _state = start_case
    error = PhysicalDispatchInterrupted("cancelled")
    monkeypatch.setattr(custody, "record_start_requested", lambda *a, **k: (_ for _ in ()).throw(error))

    def fail_cleanup(*_args, **_kwargs):
        raise OSError("cleanup unavailable")

    monkeypatch.setattr(review_execution, "_retire_orphaned_review_registration", fail_cleanup)
    monkeypatch.setattr(delegate, "_retire_orphaned_registration", fail_cleanup)
    with pytest.raises(PhysicalDispatchInterrupted) as caught:
        start()
    assert caught.value is error


@pytest.mark.parametrize("start_case", ["tool"], indirect=True)
def test_cleanup_failure_keeps_typed_pause_refusal(start_case, monkeypatch):
    start, gateway, _shared, _fail, _state = start_case
    error = owner_pause.OwnerPauseRefused("owner_pause")
    monkeypatch.setattr(delegate, "run_operation", lambda *a, **k: (_ for _ in ()).throw(error))
    monkeypatch.setattr(delegate, "_retire_orphaned_registration",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("custody unavailable")))
    result = json.loads(start().text)
    assert result["reason"] == "owner_pause" and result["definitely_unrun"] is True
    assert result["project_retired"] is False
    assert result["project_retention_reason"] == "cleanup_unconfirmed"
    assert gateway().start_requests == []


@pytest.mark.parametrize("start_case", ["tool"], indirect=True)
@pytest.mark.parametrize("recovering", [False, True])
def test_pause_before_callable_reports_only_a_fresh_start_as_unrun(start_case, tmp_path, monkeypatch, recovering):
    """The callable's refusal says nothing about a prior POST whose reply was lost."""
    start, gateway, _shared, fail, state = start_case
    original = None
    if recovering:
        legacy_retry_store(monkeypatch)
        fail(gateway_module.ClaudexorUnavailable("daemon_unreachable", "POST response lost"))
        start()
        original = custody.invocation_record(tmp_path, state["pending_invocation_id"])
        assert original["state"] == "pending" and len(gateway().start_requests) == 1

    def refuse(_source, _function, *_args, **_kwargs):
        raise owner_pause.OwnerPauseRefused("owner_pause")

    monkeypatch.setattr(delegate, "run_operation", refuse)
    payload = json.loads(start().text)
    assert payload["reason"] == "owner_pause"
    assert payload.get("definitely_unrun", False) is not recovering
    assert len(gateway().start_requests) == int(recovering), "the fenced call sends no new POST"
    assert gateway().removals == ([] if recovering else ["proj-new"])
    row = failures(tmp_path)[-1]
    assert row["definite"] is not recovering
    record = custody.invocation_record(tmp_path, row["invocation_id"])
    if recovering:
        assert payload["pending_invocation_id"] == row["invocation_id"] == state["pending_invocation_id"]
        assert record["state"] == "pending" and record["request"] == original["request"]
        assert record["project_id"] == original["project_id"] == "proj-new"
        assert payload["project_retention_reason"] == "start_outcome_unknown_run_may_exist"
    else:
        from ouroboros.delegate_custody_current import owed_registrations

        assert "pending_invocation_id" not in payload and record["state"] == "failed_definite"
        assert payload["project_retired"] is True and not owed_registrations(tmp_path)
        retired, = (r for r in custody.custody_rows(tmp_path) if r.get("type") == custody.PROJECT_RETIRED)
        assert retired["removal_outcome"] == "removed"


@pytest.mark.parametrize("journal", ["PERSIST", "WAL"])
def test_unowned_claim_wait_expiry_retires_without_post(start_case, tmp_path, monkeypatch, journal):
    from ouroboros import _usage_wait, delegate_custody_current, store
    from tests.test_admission_contention import database_reader

    use_journal(monkeypatch, journal)  # a held write fails its COMMIT (rollback journal), its BEGIN (WAL)
    start, gateway, _shared, _fail, _state = start_case
    original = delegate_custody_current.publish
    events = []

    def held(root, event, **kwargs):
        if event["type"] != custody.START_REQUESTED:
            return original(root, event, **kwargs)
        events.append(event)
        # Held through the claim's write only and released before cleanup.
        with database_reader(tmp_path, write=True):
            return original(root, event, **kwargs)

    monkeypatch.setattr(delegate_custody_current, "publish", held)
    monkeypatch.setattr(_usage_wait, "USAGE_LOCK_TIMEOUT_SEC", 0)
    with pytest.raises(store.StoreBusy):
        start()
    assert len(events) == 1 and gateway().start_requests == []
    assert gateway().removals == ["proj-new"]
    row = failures(tmp_path)[-1]
    assert row["definite"] is True
    [release] = [r for r in custody.custody_rows(tmp_path) if r["type"] == custody.PROJECT_RETIRED]
    assert release["project_id"] == row["project_id"] and release["removal_outcome"] == "removed"


def test_untyped_transport_failure_keeps_possible_run_registration(start_case, tmp_path):
    from ouroboros.delegate_pending import request_body

    start, gateway, _shared, fail, _state = start_case
    error = MemoryError("POST outcome unknown")
    fail(error)
    with pytest.raises(MemoryError) as caught:
        start()
    assert caught.value is error
    assert len(gateway().start_requests) == 1 and gateway().removals == []
    pending = custody.pending_invocations(tmp_path)
    assert len(pending) == 1 and pending[0]["request"] is None and pending[0]["request_ref"]
    sent = gateway().start_requests[0]
    body, key = (sent, gateway().start_keys[0]) if isinstance(sent, dict) else sent
    assert request_body(tmp_path, pending[0]) == body
    assert pending[0]["invocation_id"] == key


def test_successful_start_keeps_its_registration(start_case):
    start, gateway, _shared, _fail, _state = start_case
    start()
    assert len(gateway().start_requests) == 1 and gateway().removals == []


def test_control_after_claim_before_callable_retires_fresh_project(start_case, tmp_path, monkeypatch):
    from ouroboros import review_execution

    start, gateway, _shared, _fail, _state = start_case
    error = PhysicalDispatchInterrupted("cancelled")

    def refuse(_source, _function, *_args, **_kwargs):
        assert len(custody.pending_invocations(tmp_path)) == 1
        raise error

    monkeypatch.setattr(review_execution, "run_operation", refuse)
    monkeypatch.setattr(delegate, "run_operation", refuse)
    with pytest.raises(PhysicalDispatchInterrupted) as caught:
        start()
    assert caught.value is error and gateway().start_requests == []
    assert gateway().removals == ["proj-new"]
    assert len(failures(tmp_path)) == 1
    assert failures(tmp_path)[0]["definite"] is True
    assert custody.pending_invocations(tmp_path) == []


@pytest.mark.parametrize("control", ["budget", "deadline"])
@pytest.mark.parametrize("recovering", [False, True])
def test_review_preclaim_control_cleans_only_fresh_registration(tmp_path, monkeypatch, fake_route, control, recovering):
    from ouroboros import budget_pause, review_execution

    FakeGateway.project_unregistered = True
    state = {}
    invocation = review_execution.SessionInvocation(
        task_id="t-b", surface="scope_review", slot_id="scope_slot_1", timeout_sec=30,
        retry_state=state, owner_deadline_at="2099-01-01T00:00:00+00:00")

    def start():
        return review_execution.run_delegated_review_session(
            prompt="review this", root="/tmp/fake-repo", custody_drive=tmp_path, invocation=invocation)

    if recovering:
        legacy_retry_store(monkeypatch)
        FakeGateway.start_error = gateway_module.ClaudexorUnavailable("daemon_unreachable", "lost response")
        with pytest.raises(gateway_module.ClaudexorUnavailable):
            start()
        token = state["pending_invocation_id"]
        saved = custody.invocation_record(tmp_path, token)["request"]
    if control == "budget":
        monkeypatch.setattr(budget_pause, "dispatch_fenced", lambda _task: True)
    else:
        monkeypatch.setattr(review_execution, "owner_deadline_exhausted", lambda **_kwargs: True)
    if control == "deadline" and recovering:
        # The old request owns its physical deadline; a new local deadline cannot retire it.
        result = start()
        assert result["idempotent_recovery"] is True
        assert FakeGateway.instances[-1].removals == []
        return
    with pytest.raises(review_execution.ReviewRouteUnavailable):
        start()
    gateway = FakeGateway.instances[-1]
    assert gateway.start_requests == []
    assert gateway.removals == ([] if recovering else ["proj-new"])
    assert failures(tmp_path)[-1]["definite"] is not recovering
    if recovering:
        assert state["pending_invocation_id"] == token
        assert custody.invocation_record(tmp_path, token)["request"] == saved


@pytest.mark.parametrize("fence", [{"state": owner_pause.FENCE_PAUSED, "reason": "owner_pause"},
                                   {"state": "unknown", "reason": "owner_pause_authority_unreadable"}])
def test_only_a_read_pause_is_a_held_launch(fence, tmp_path, monkeypatch):
    monkeypatch.setattr(owner_pause, "member_fence", lambda *_a, **_k: dict(fence))
    with pytest.raises(owner_pause.OwnerPauseRefused) as caught:
        owner_pause._validate_launch(SimpleNamespace(task_id="t-b", root_task_id="t-b", drive_root=tmp_path))
    assert type(caught.value) is owner_pause.OwnerPauseRefused, "callers keep the refusal type they already read"
    assert caught.value.held is owner_pause.fence_closed(fence)
    assert str(caught.value) == fence["reason"]


@pytest.mark.parametrize("refusal", [owner_pause.OwnerPauseRefused("owner_pause", held=True),
                                     owner_pause.OwnerPauseRefused("owner_pause_authority_unreadable")])
def test_a_fenced_replay_of_an_ordinary_lost_start_stays_lost(tmp_path, monkeypatch, fake_route, refusal):
    """Only a selected continuation's send is held for Resume: an ordinary recovered start keeps its old refusal,
    its key and bytes, and its seat's custody stays lost (never re-posted), whether the Pause was read or not."""
    from ouroboros import review_custody, review_execution

    legacy_retry_store(monkeypatch)
    state = {}
    invocation = review_execution.SessionInvocation(
        task_id="t-b", surface="scope_review", slot_id="scope_slot_1", timeout_sec=30,
        retry_state=state, owner_deadline_at="2099-01-01T00:00:00+00:00")

    def start():
        return review_execution.run_delegated_review_session(
            prompt="review this", root="/tmp/fake-repo", custody_drive=tmp_path, invocation=invocation)

    FakeGateway.start_error = gateway_module.ClaudexorUnavailable("daemon_unreachable", "lost response")
    with pytest.raises(gateway_module.ClaudexorUnavailable):
        start()
    token = state["pending_invocation_id"]
    saved = custody.invocation_record(tmp_path, token)["request"]
    monkeypatch.setattr(review_execution, "run_operation", lambda *_a, **_k: (_ for _ in ()).throw(refusal))
    with pytest.raises(review_execution.ReviewRouteUnavailable) as caught:
        start()
    assert caught.value.held_before_send is False and FakeGateway.instances[-1].start_requests == []
    assert review_custody._worker_exception_operation_state(caught.value, state) == "custody_lost"
    record = custody.invocation_record(tmp_path, token)
    assert record["state"] == "pending" and record["request"] == saved and state["pending_invocation_id"] == token





def _deciding_operation(**facts):
    """The bound review operation of a selected continuation: the deciding task owns its controls."""
    author = SimpleNamespace(task_id="deciding-task", root_task_id="deciding-task", drive_root=facts.pop("root"))
    return SimpleNamespace(continuation_author=author, continuation_pause_seen=facts.get("paused", False),
                           control_state=facts.get("state", ""), control=lambda: facts.get("control"))


@pytest.mark.parametrize("case", ["budget_pause", "owner_pause_at_launch", "pause_and_stop", "wave_task_fenced"])
def test_a_paused_replay_of_a_start_whose_post_was_entered_keeps_its_key_and_stays_unknown(tmp_path, monkeypatch,
                                                                                            fake_route, case):
    """A continuation start whose first POST was entered (its handoff marked) and lost its reply: the deciding task's
    budget pause, or its read Pause at launch, refuses the replay before any POST and keeps its key and bytes pending,
    but its fate stays unknown (lost, re-sent only by an explicit choice's exact replay). A coinciding Stop ends it as
    Stop. The wave task's own fence does not decide whether the deciding task may send."""
    from ouroboros import budget_pause, review_custody, review_execution
    from ouroboros.review_continuation import PREFIX
    from ouroboros.review_operation import bind_review_operation

    legacy_retry_store(monkeypatch)
    state = {}
    invocation = review_execution.SessionInvocation(
        task_id="wave-task", surface="scope_review", slot_id="scope_slot_1", timeout_sec=30, retry_state=state,
        owner_deadline_at="2099-01-01T00:00:00+00:00", operation_id=f"{PREFIX}synthetic")

    def start():
        return review_execution.run_delegated_review_session(
            prompt="review this", root="/tmp/fake-repo", custody_drive=tmp_path, invocation=invocation)

    FakeGateway.start_error = gateway_module.ClaudexorUnavailable("daemon_unreachable", "lost response")
    with pytest.raises(gateway_module.ClaudexorUnavailable):
        start()
    token = state["pending_invocation_id"]
    saved = custody.invocation_record(tmp_path, token)["request"]
    assert custody.invocation_record(tmp_path, token)["handoff"] == "entered", "its first POST was entered"
    paused = {"wave_task_fenced": "wave-task", "owner_pause_at_launch": ""}.get(case, "deciding-task")
    monkeypatch.setattr(budget_pause, "dispatch_fenced", lambda task: bool(paused) and task == paused)
    if case == "owner_pause_at_launch":
        monkeypatch.setattr(review_execution, "run_operation", lambda *_a, **_k: (_ for _ in ()).throw(
            owner_pause.OwnerPauseRefused("owner_pause", held=True)))
    operation = _deciding_operation(root=tmp_path, **({"control": "cancelled"} if case == "pause_and_stop" else {}))
    with bind_review_operation(operation):
        if case == "wave_task_fenced":
            FakeGateway.start_error = gateway_module.ClaudexorUnavailable("daemon_unreachable", "lost again")
            with pytest.raises(gateway_module.ClaudexorUnavailable):
                start()
            assert FakeGateway.instances[-1].start_keys == [token], "the same recorded start, sent by the deciding task"
        else:
            with pytest.raises(review_execution.ReviewRouteUnavailable) as caught:
                start()
            code = {"budget_pause": "budget_pausing_no_send", "owner_pause_at_launch": "owner_pause",
                    "pause_and_stop": "review_continuation_author_fenced"}[case]
            assert (caught.value.code, caught.value.held_before_send) == (code, False)
            assert FakeGateway.instances[-1].start_requests == []
            assert review_custody._worker_exception_operation_state(caught.value, state) == "custody_lost"
    record = custody.invocation_record(tmp_path, token)
    assert record["state"] == "pending" and record["request"] == saved and state["pending_invocation_id"] == token


def _selected_start(root, token="selected-token", **facts):
    """One newly selected Continue's recorded START (synthetic), as its choice writes it (``handoff``: unsent)."""
    from ouroboros.review_continuation import PREFIX

    assert custody.emit(root, custody.START_REQUESTED, {
        "invocation_id": token, "task_id": "wave-task", "root_task_id": "wave-task", "surface": "scope_review",
        "slot_id": "scope_slot_1", "operation_id": f"{PREFIX}synthetic", "request": {"prompt": "continue"},
        **({"handoff": "unsent"} | facts)})
    return SimpleNamespace(operation_id=f"{PREFIX}synthetic", task_id="wave-task", surface="scope_review",
                           slot_id="scope_slot_1", retry_state={"pending_invocation_id": token})


@pytest.mark.parametrize("control", ["budget_pause", "owner_pause", "stop", "durable_stop", "unknown",
                                     "pause_and_window_closed", "pause_and_budget_reached",
                                     "pause_after_a_handoff", "pause_on_an_older_start"])
def test_a_pause_holds_a_continuation_send_only_while_its_start_was_never_handed_off(tmp_path, monkeypatch, control):
    """At the send callback, on one synthetic selected START: the deciding task's read Pause or entering budget pause
    holds it for Resume (in flight, key kept) while its record still says it was never handed off. Its Stop, durable or
    latched, an unreadable control, a closed window and a reached budget fence refuse as themselves even when a pause
    coincides. After one admitted send marked the handoff, or on a START recorded without the fact, a pause holds
    nothing: its fate stays unknown."""
    from ouroboros import budget_pause, review_custody
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.review_execution import ReviewRouteUnavailable
    from ouroboros.review_operation import bind_review_operation
    from ouroboros.review_session_custody import continuation_send_admitted
    from ouroboros.tools import review_helpers

    invocation = _selected_start(tmp_path, **({"handoff": None} if control == "pause_on_an_older_start" else {}))
    pausing = {"on": control in {"budget_pause", "pause_and_window_closed", "pause_and_budget_reached",
                                 "pause_after_a_handoff", "pause_on_an_older_start"}}
    monkeypatch.setattr(budget_pause, "dispatch_fenced", lambda task: pausing["on"] and task == "deciding-task")
    if control == "pause_after_a_handoff":  # an earlier admitted send marked it, then its POST lost its reply
        pausing["on"] = False
        with bind_review_operation(_deciding_operation(root=tmp_path)):
            continuation_send_admitted(invocation, None, {"prompt": "continue"}, "fake-model", tmp_path)
        assert custody.invocation_record(tmp_path, "selected-token")["handoff"] == "entered"
        pausing["on"] = True
    if control == "durable_stop":
        request_cancel(tmp_path, "deciding-task", reason="owner Stop")
    if control == "pause_and_budget_reached":
        monkeypatch.setattr(review_helpers, "review_wave_budget_gate", lambda *_a, **_k: {
            "binding_axis": "root", "limit_usd": 1.0, "known_usd": 1.0})
    facts = {"owner_pause": {"paused": True}, "stop": {"control": "cancelled"}, "unknown": {"state": "unknown"}}.get(control, {})
    end = -1.0 if control == "pause_and_window_closed" else None
    with bind_review_operation(_deciding_operation(root=tmp_path, **facts)), \
            pytest.raises(ReviewRouteUnavailable) as caught:
        continuation_send_admitted(invocation, end, {"prompt": "continue"}, "fake-model", tmp_path)
    held = control in {"budget_pause", "owner_pause"}
    assert caught.value.held_before_send is held
    assert caught.value.code == {
        "stop": "review_continuation_author_fenced", "durable_stop": "review_continuation_author_fenced",
        "unknown": "review_continuation_author_fenced", "pause_and_window_closed": "continuation_window_ended",
        "pause_and_budget_reached": "continuation_budget_insufficient"}.get(control, "review_continuation_author_paused")
    assert review_custody._worker_exception_operation_state(
        caught.value, {"pending_invocation_id": "selected-token"}) == ("in_flight" if held else "custody_lost")
    record = custody.invocation_record(tmp_path, "selected-token")
    assert record["state"] == "pending" and record["request"] == {"prompt": "continue"}
    assert record.get("handoff") == {"pause_after_a_handoff": "entered", "pause_on_an_older_start": None}.get(control, "unsent")


def test_a_handoff_mark_that_did_not_land_sends_nothing_and_claims_nothing(tmp_path, monkeypatch):
    """The mark precedes the transport call: when it does not land, the send is refused before any POST, unknown (not
    held), and the record keeps the fact it had."""
    from ouroboros.review_execution import ReviewRouteUnavailable
    from ouroboros.review_operation import bind_review_operation
    from ouroboros.review_session_custody import continuation_send_admitted

    invocation = _selected_start(tmp_path)
    real = custody.emit
    monkeypatch.setattr(custody, "emit", lambda root, kind, payload, **kw: False if kind == custody.START_HANDOFF
                        else real(root, kind, payload, **kw))
    with bind_review_operation(_deciding_operation(root=tmp_path)), pytest.raises(ReviewRouteUnavailable) as caught:
        continuation_send_admitted(invocation, None, {"prompt": "continue"}, "fake-model", tmp_path)
    assert (caught.value.code, caught.value.held_before_send) == ("review_continuation_handoff_unrecorded", False)
    assert custody.invocation_record(tmp_path, "selected-token")["handoff"] == "unsent"
