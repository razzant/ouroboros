"""A real review adapter retains its exact unsent custody claim while SQLite is busy."""
from contextlib import ExitStack, contextmanager
from datetime import datetime, timedelta, timezone

import pytest

from ouroboros import delegate_custody as custody, delegate_custody_current as current
from ouroboros import owner_pause, queue_observation, store
from ouroboros.llm_attempt import PhysicalDispatchInterrupted
from ouroboros.model_wait import task_model_wait_scope
from ouroboros.review_execution import ReviewRouteUnavailable
from ouroboros.task_results import write_task_result
from tests.test_admission_contention import database_reader
from tests._store_journal_shared import use_journal
from tests._review_session_route_shared import (
    FakeGateway, _run_session_directly,
    fake_route as _fake_route,
    _owned_gateway_uses_each_test_transport as _owned_gateway_fixture,
)

fake_route = _fake_route
_owned_gateway_uses_each_test_transport = _owned_gateway_fixture
pytestmark = pytest.mark.serial


@pytest.fixture
def start(tmp_path, fake_route):
    write_task_result(tmp_path, "t-b", "running", root_task_id="t-b")
    repo = tmp_path / "repo"
    repo.mkdir()
    return lambda: _run_session_directly(tmp_path, root=str(repo))


def posts():
    return [(request, key) for gateway in FakeGateway.instances
            for request, key in zip(gateway.start_requests, gateway.start_keys)]


def hold_request(monkeypatch, stack, root, *, exclusive=False):
    """Contend the first start claim: every access (``exclusive``), else its write, as the store's
    journal mode contends one (a reader holding a rollback COMMIT, a writer holding a WAL one)."""
    publish = current.publish
    release, events = [], []

    def held(root_arg, event, **kwargs):
        if event["type"] == custody.START_REQUESTED:
            events.append(dict(event))
            if not release:
                release.append(stack.enter_context(database_reader(root, exclusive=exclusive, write=not exclusive)))
        return publish(root_arg, event, **kwargs)

    monkeypatch.setattr(current, "publish", held)
    return release, events


@pytest.mark.parametrize("exclusive", [False, True], ids=["reader-commit", "exclusive-acquisition"])
def test_actual_review_start_retains_one_request_and_post(start, tmp_path, monkeypatch, exclusive):
    if not exclusive:  # a reader holds only a rollback-journal COMMIT; a WAL COMMIT waits for no reader
        use_journal(monkeypatch, "PERSIST")
    waited = []
    with ExitStack() as stack:
        release, events = hold_request(monkeypatch, stack, tmp_path, exclusive=exclusive)

        def wait(_seconds):
            assert posts() == []
            with owner_pause.launch_lock(tmp_path, "t-b", wait=False):
                waited.append(True)
            release[0]()
            assert custody.invocation_record(tmp_path, events[0]["invocation_id"]) is None

        monkeypatch.setattr(queue_observation, "_sleep", wait)
        facts = start()
    assert facts["run_id"] == "run-1" and waited == [True]
    assert len(events) == 2 and events[0] == events[1]
    assert len(posts()) == 1 and posts()[0][1] == events[0]["invocation_id"]
    assert custody.invocation_record(tmp_path, events[0]["invocation_id"])["request"] == posts()[0][0]


@pytest.mark.parametrize("journal", ["PERSIST", "WAL"])
@pytest.mark.parametrize("control", ["pause", "stop", "deadline", "closed"])
def test_current_controls_end_the_unsent_wait_without_post(start, tmp_path, monkeypatch, control, journal):
    use_journal(monkeypatch, journal)  # a held write fails its COMMIT (rollback journal), its BEGIN (WAL)
    stopped, waited = [], []
    with task_model_wait_scope(task={"id": "t-b", "_attempt": 1}, drive_root=tmp_path,
            event_queue=None, worker_slot_held=False,
            owner_control=lambda: "cancelled" if stopped else None) as owner, ExitStack() as stack:
        release, events = hold_request(monkeypatch, stack, tmp_path)

        def wait(_seconds):
            assert posts() == []
            with owner_pause.launch_lock(tmp_path, "t-b", wait=False):
                waited.append(True)
            release[0]()
            if control == "pause":
                owner_pause.install_fence(tmp_path, "t-b", request_id="during-start-claim")
            elif control == "stop":
                stopped.append(True)
            elif control == "deadline":
                owner.task["deadline_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
            else:
                owner.closed = True

        monkeypatch.setattr(queue_observation, "_sleep", wait)
        with pytest.raises((owner_pause.OwnerPauseRefused, PhysicalDispatchInterrupted)) as caught:
            start()
        if control in {"stop", "deadline"}:
            assert caught.value.control_reason == ("cancelled" if control == "stop" else "deadline")
    assert waited == [True] and len(events) == 1 and posts() == []
    assert custody.invocation_record(tmp_path, events[0]["invocation_id"]) is None


def uncertain_rollback():
    error = store.StoreBusy("outcome unknown", reason="contention")
    error.rollback_confirmed = False
    return error


@pytest.mark.parametrize("error", [store.StoreBusy("unknown"),
    store.StoreBusy("uninspectable", reason="uninspectable"), uncertain_rollback(),
    store.StoreLost("lost"), PermissionError("denied")])
def test_hard_unknown_and_uncertain_claim_fail_once(start, monkeypatch, error):
    publish, calls = current.publish, []

    @contextmanager
    def unavailable(*_args, **_kwargs):
        raise error
        yield

    def fail_claim(root, event, **kwargs):
        if event["type"] != custody.START_REQUESTED:
            return publish(root, event, **kwargs)
        calls.append(event)
        with monkeypatch.context() as local:
            local.setattr(store, "open_store", unavailable)
            return publish(root, event, **kwargs)

    monkeypatch.setattr(current, "publish", fail_claim)
    monkeypatch.setattr(queue_observation, "_sleep", lambda _: pytest.fail("hard failure retried"))
    with pytest.raises(ReviewRouteUnavailable) as caught:
        start()
    assert caught.value.code == "start_request_row_unwritable"
    assert len(calls) == 1 and posts() == []


def test_default_emit_retains_boolean_failure(tmp_path):
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    with database_reader(tmp_path, write=True), store.admission_acquisition():
        assert custody.emit(tmp_path, custody.START_REQUESTED, {
            "task_id": "root", "invocation_id": "default-bool", "idempotency_key": "one"}) is False


def test_postcommit_flush_contention_never_replays_claim(start, tmp_path, monkeypatch):
    publish, flush, events = current.publish, current.flush_outbox, []
    active = []

    def capture(root, event, **kwargs):
        if event["type"] != custody.START_REQUESTED:
            return publish(root, event, **kwargs)
        events.append(dict(event))
        active.append(True)
        try:
            return publish(root, event, **kwargs)
        finally:
            active.pop()

    def fail_flush(root):
        if active:
            error = store.StoreBusy("after custody committed", reason="contention")
            error.rollback_confirmed = True  # A different transaction's rollback is not this claim's.
            raise error
        return flush(root)

    monkeypatch.setattr(current, "publish", capture)
    monkeypatch.setattr(current, "flush_outbox", fail_flush)
    monkeypatch.setattr(queue_observation, "_sleep", lambda _: pytest.fail("committed claim replayed"))
    with pytest.raises(ReviewRouteUnavailable) as caught:
        start()
    assert caught.value.code == "start_request_row_unwritable"
    assert len(events) == 1 and posts() == []
    assert custody.invocation_record(tmp_path, events[0]["invocation_id"]) is not None


def test_committed_outbox_allows_one_post_when_chain_append_is_owed(start, tmp_path, monkeypatch):
    monkeypatch.setattr(current, "_append_durable", lambda *_args: False)
    monkeypatch.setattr(queue_observation, "_sleep", lambda _: pytest.fail("committed outbox replayed"))
    facts = start()
    assert facts["run_id"] == "run-1" and len(posts()) == 1
    assert current.outbox_depth(tmp_path) > 0
    assert custody.invocation_record(tmp_path, posts()[0][1]) is not None


def test_post_failure_never_retries_claim_or_transport(start, tmp_path, monkeypatch):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    FakeGateway.start_error = ClaudexorUnavailable("transport_error", "response lost after POST")
    monkeypatch.setattr(queue_observation, "_sleep", lambda _: pytest.fail("POST replayed"))
    with pytest.raises(ClaudexorUnavailable):
        start()
    assert len(posts()) == 1
    assert custody.invocation_record(tmp_path, posts()[0][1]) is not None
