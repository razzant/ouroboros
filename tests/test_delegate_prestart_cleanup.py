"""Fresh pre-POST interruptions retire only their own disposable registration."""
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import json

import pytest

from ouroboros import delegate_custody as custody, owner_pause, queue_observation
from ouroboros.gateways import claudexor as gateway_module
from ouroboros.llm_attempt import PhysicalDispatchInterrupted
from ouroboros.model_wait import task_model_wait_scope
from ouroboros.task_results import write_task_result
from ouroboros.tools import delegate
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
pytestmark = pytest.mark.serial


class ToolGateway(_LiveRunStub):
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

    def remove_project(self, project_id):
        self.removals.append(project_id)

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


@pytest.mark.parametrize("journal", ["PERSIST", "WAL"])
@pytest.mark.parametrize("control", ["stop", "deadline", "closed"])
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
    assert row["definite"] is True and row["project_retired"] is True
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
    assert failures(tmp_path)[-1]["definite"] is True


def test_recovered_unknown_start_is_not_retired_by_new_preclaim_error(start_case, tmp_path, monkeypatch):
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
    assert row["definite"] is True and row["project_retired"] is True


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
