"""Real contention defers the same preparation, never repeats its physical copy."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import contextlib
import queue as stdqueue
import threading
import time

import pytest

from ouroboros import store
from ouroboros.task_results import STATUS_SCHEDULED
from supervisor import queue, task_admission, workspace_preparation as prep
from tests.test_admission_contention import database_reader
from tests.test_workspace_preparation import env as env, spy_provisions, _WAIT_SEC

pytestmark = pytest.mark.serial


def test_held_store_at_entry_claims_in_memory_and_admits_once_when_free(env, monkeypatch):
    """The event loop touches no storage: a request arriving while the store is held is claimed in
    memory, never re-put, and its lane writes the marker and admits once the store frees."""
    bus = env.ctx.event_queue = stdqueue.Queue()
    event = env.request("entry-busy")
    copies = spy_provisions(monkeypatch)
    with database_reader(env.data, exclusive=True):
        assert env.dispatch_returns(event)
        [claim] = [row for row in prep._LIVE.values() if row.task_id == "entry-busy"]
        assert queue.ADMISSION_RESERVATIONS == {"entry-busy": claim.token}
        assert env.dispatch_returns(event) and env.dispatch_returns(dict(event))  # duplicates coalesce
        assert env.dispatch_returns({**event, "root_task_id": "foreign", "parent_task_id": "foreign"})
        assert bus.empty() and copies == [] and len(prep._LIVE) == 1
        assert not env.gen.wait_idle(0.5)  # retrying at the head of its own lane
    assert env.gen.wait_idle(_WAIT_SEC)
    assert copies == ["entry-busy"]
    assert env.pending_ids() == ["entry-busy"] and env.row("entry-busy")["status"] == STATUS_SCHEDULED
    assert env.row("entry-busy")["root_task_id"] == "root" and env.row("entry-busy")[prep.MARKER]["token"] == claim.token
    assert bus.empty() and not prep._LIVE and queue.ADMISSION_RESERVATIONS == {}
    assert not any("not scheduled" in text for _, text, _ in env.ctx.sent)


@pytest.mark.parametrize("published", [False, True])
def test_contended_first_marker_retries_on_its_lane_with_the_same_token(env, monkeypatch, published):
    event = env.request("marker-busy")
    real_write, real_read = prep.write_task_result, prep.load_task_result
    armed, attempts = [True], []

    def write(*args, **kwargs):
        if kwargs.get("_field_projector") and armed[0]:
            attempts.append(threading.current_thread().name)
            if published:
                real_write(*args, **kwargs)
            raise store.StoreBusy("held marker", reason="contention")
        return real_write(*args, **kwargs)

    def read(*args, **kwargs):
        if armed[0]:
            raise store.StoreBusy("readback held", reason="contention")
        return real_read(*args, **kwargs)

    monkeypatch.setattr(prep, "write_task_result", write)
    monkeypatch.setattr(prep, "load_task_result", read)
    assert env.dispatch_returns(event)
    deadline = time.monotonic() + _WAIT_SEC
    while len(attempts) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    [claim] = [row for row in prep._LIVE.values() if row.task_id == "marker-busy"]
    assert len(attempts) >= 2 and set(attempts) == {"workspace-preparation"}  # the lane, never the loop
    assert queue.ADMISSION_RESERVATIONS == {"marker-busy": claim.token} and env.pending_ids() == []
    armed[0] = False
    assert env.gen.wait_idle(_WAIT_SEC)
    assert env.pending_ids() == ["marker-busy"]
    assert env.row("marker-busy")[prep.MARKER]["token"] == claim.token


@pytest.mark.parametrize("seam", ["preflight", "depth", "ready", "final"])
def test_owned_job_waits_off_q_then_uses_its_one_copy(env, monkeypatch, seam):
    event = env.request("owned-busy")
    copies = spy_provisions(monkeypatch)
    reached, proceed = threading.Event(), threading.Event()
    under_q, opened = [], threading.Event()
    real_connection = store.connection

    @contextlib.contextmanager
    def connection(*args, **kwargs):  # every lease, a thread's kept connection included
        if queue._queue_lock._is_owned():
            under_q.append(kwargs.get("wait_sec", 0))
        opened.set()
        with real_connection(*args, **kwargs) as conn:
            yield conn

    if seam in {"preflight", "depth", "final"}:
        name = {"preflight": "subagent_schedule_preflight", "depth": "parse_schedule_task_depth",
                "final": "enqueue_subagent_with_scheduled_result"}[seam]
        original = getattr(task_admission, name)

        def gated(*args, **kwargs):
            if not reached.is_set():
                reached.set()
                assert proceed.wait(_WAIT_SEC)
            return original(*args, **kwargs)

        monkeypatch.setattr(task_admission, name, gated)
    else:
        original = prep._transition

        def gated(root, tid, token, marker, **kwargs):
            if marker["phase"] == prep.READY and not reached.is_set():
                reached.set()
                assert proceed.wait(_WAIT_SEC)
            return original(root, tid, token, marker, **kwargs)

        monkeypatch.setattr(prep, "_transition", gated)
    assert env.dispatch_returns(event) and reached.wait(_WAIT_SEC)
    with database_reader(env.data, exclusive=True):
        monkeypatch.setattr(store, "connection", connection)
        proceed.set()
        # The job's storage acquisitions wait OFF Q, in slices; Q stays free for a control.
        assert opened.wait(3) and not env.gen.wait_idle(0.5)
        with ThreadPoolExecutor(max_workers=1) as pool:
            def control():
                with queue._queue_lock:
                    return True
            assert pool.submit(control).result(timeout=3)
        assert env.pending_ids() == [] and under_q == []
    assert env.gen.wait_idle(_WAIT_SEC)
    assert under_q == [], "a prepared-child owner opened SQLite while holding Q"
    assert copies == ["owned-busy"] and env.pending_ids() == ["owned-busy"]
    assert env.row("owned-busy")["status"] == STATUS_SCHEDULED


@pytest.mark.parametrize("reason", ["contention", "uninspectable"])
def test_nested_queue_refusal_retries_only_positive_contention(env, monkeypatch, reason):
    from ouroboros import projects_registry

    event = env.request("inner-busy")
    copies, reads, tokens = spy_provisions(monkeypatch), [], []
    original = projects_registry.task_project_membership

    def membership(root, task, *, bindings_snapshot=None):
        if task.get("id") == "inner-busy":
            reads.append(task["id"])
            tokens.append(queue.ADMISSION_RESERVATIONS.get("inner-busy"))
            if len(reads) == 1:
                raise store.StoreBusy("held Project read", reason=reason)
        return original(root, task, bindings_snapshot=bindings_snapshot)

    monkeypatch.setattr(projects_registry, "task_project_membership", membership)
    assert env.dispatch_returns(event) and env.gen.wait_idle(_WAIT_SEC)
    assert copies == ["inner-busy"]
    if reason == "contention":
        assert len(reads) == 2 and tokens[0] == tokens[1] and tokens[0]
        assert env.pending_ids() == ["inner-busy"] and env.row("inner-busy")["status"] == STATUS_SCHEDULED
    else:
        assert len(reads) == 1 and env.pending_ids() == []
        assert env.row("inner-busy")["admission_outcome"] == "never_admitted"


@pytest.mark.parametrize("control", ["root_stop", "owner_pause", "generation_close"])
def test_control_during_preflight_contention_prevents_later_copy(env, monkeypatch, control):
    from supervisor import task_lifecycle
    from supervisor.events_budget import _set_root_budget_pause_locked

    event = env.request("controlled-busy")
    copies = spy_provisions(monkeypatch)
    reached, proceed, busy_seen = threading.Event(), threading.Event(), threading.Event()
    original = task_admission.subagent_schedule_preflight

    def preflight(*args, **kwargs):
        reached.set()
        assert proceed.wait(_WAIT_SEC)
        try:
            return original(*args, **kwargs)
        except store.StoreBusy:
            busy_seen.set()
            raise

    monkeypatch.setattr(task_admission, "subagent_schedule_preflight", preflight)
    assert env.dispatch_returns(event) and reached.wait(_WAIT_SEC)
    with database_reader(env.data, exclusive=True):
        proceed.set()
        assert busy_seen.wait(3)
        with queue._queue_lock:
            if control == "root_stop":
                task_lifecycle.CANCELLED_ROOT_FENCES["root"] = "2026-10-13T00:00:00Z"
            elif control == "owner_pause":
                _set_root_budget_pause_locked("root", {"cause": "owner_pause", "fence_id": "pause-busy"})
            else:
                prep.close_generation(env.gen)
    assert env.gen.wait_idle(_WAIT_SEC)
    prep.recover_interrupted_preparations(env.data)
    assert copies == [] and env.pending_ids() == []
    assert env.row("controlled-busy")["admission_outcome"] == "never_admitted"
    expected = {"root_stop": "root_cancelled", "owner_pause": "root_owner_paused",
                "generation_close": prep.INTERRUPTED_BY_GENERATION_EXIT}[control]
    assert env.row("controlled-busy")["reason_code"] == expected


def test_possible_enqueue_is_not_retried_on_typed_contention(env, monkeypatch):
    event = env.request("uncertain")
    calls = []
    real_enqueue = env.ctx.enqueue_task

    def enqueue(task):
        calls.append(task["id"])
        real_enqueue(task)
        raise store.StoreBusy("after enqueue, outcome unknown", reason="contention")

    monkeypatch.setattr(env.ctx, "enqueue_task", enqueue)
    assert env.dispatch_returns(event)
    assert env.gen.wait_idle(_WAIT_SEC)
    assert calls == ["uncertain"] and env.pending_ids() == ["uncertain"]
    assert env.row("uncertain")[prep.MARKER]["phase"] == prep.ADMITTING
    assert env.row("uncertain").get("admission_outcome") != "never_admitted"


def test_queue_refusal_flag_after_actual_append_never_authorizes_retry(env, monkeypatch):
    from ouroboros import projects_registry

    event = env.request("post-append")
    real_membership = projects_registry.task_project_membership
    real_view = projects_registry.project_admission_view
    calls = []
    # Use the real queue's guarded append but inject its failing context exit.
    monkeypatch.setattr(projects_registry, "task_project_membership", lambda root, task, *, bindings_snapshot=None: (
        ("bound", False) if task.get("id") == "post-append"
        else real_membership(root, task, bindings_snapshot=bindings_snapshot)))
    monkeypatch.setattr(projects_registry, "project_admission_view", lambda root, project, **kw: (
        {"project_id": "bound", "project": {"id": "bound"}} if project == "bound"
        else real_view(root, project, **kw)))

    @contextlib.contextmanager
    def guard(*args, **kwargs):
        calls.append("enter")
        yield
        assert env.pending_ids() == ["post-append"]
        raise store.StoreBusy("commit window after append", reason="contention")

    monkeypatch.setattr(projects_registry, "project_admission_guard", guard)
    assert env.dispatch_returns(event) and env.gen.wait_idle(_WAIT_SEC)
    assert calls == ["enter"] and env.pending_ids() == ["post-append"]
    admitted = queue.PENDING[0]
    assert admitted.get("_admission_uncertain") and not admitted.get("_admission_never_admitted")
    # The receipt preceded the put and the put happened: the claim is closed, never refused or retried.
    row = env.row("post-append")
    assert row["status"] == STATUS_SCHEDULED and row[prep.MARKER]["phase"] == prep.ADMITTED


def test_refusal_committed_before_busy_readback_rejoins_its_same_receipt(env, monkeypatch):
    from supervisor import task_lifecycle

    event = env.request("refusal-readback")
    writes, readback_gap = [], [False]
    original_write, original_read = task_admission.write_task_result, task_admission.load_task_result

    def write(root, tid, status, **kwargs):
        result = original_write(root, tid, status, **kwargs)
        if tid == "refusal-readback" and status == "failed":
            writes.append(kwargs["_admission_refusal_token"])
            readback_gap[0] = True
        return result

    def read(root, tid, **kwargs):
        if tid == "refusal-readback" and readback_gap[0]:
            readback_gap[0] = False
            raise store.StoreBusy("refusal already committed, readback busy", reason="contention")
        return original_read(root, tid, **kwargs)

    monkeypatch.setattr(task_admission, "write_task_result", write)
    monkeypatch.setattr(task_admission, "load_task_result", read)
    with queue._queue_lock:
        task_lifecycle.CANCELLED_ROOT_FENCES["root"] = "2026-10-13T00:00:00Z"
    assert env.dispatch_returns(event) and env.gen.wait_idle(_WAIT_SEC)
    stored = env.row("refusal-readback")
    assert stored["status"] == "failed" and stored["reason_code"] == "root_cancelled"
    assert writes == [stored["_admission_refusal_token"]]
    assert env.pending_ids() == [] and queue.ADMISSION_RESERVATIONS == {} and not prep._LIVE


@pytest.mark.parametrize("seam", ["entry", "depth", "final"])
def test_readonly_child_retries_real_contention_on_the_admission_lane(env, monkeypatch, seam):
    bus = env.ctx.event_queue = stdqueue.Queue()
    event = env.readonly("readonly-busy")
    copies = spy_provisions(monkeypatch)
    reached, proceed = threading.Event(), threading.Event()
    if seam != "entry":
        name = "parse_schedule_task_depth" if seam == "depth" else "enqueue_subagent_with_scheduled_result"
        original = getattr(task_admission, name)

        def gate(*args, **kwargs):
            if not reached.is_set():
                reached.set()
                assert proceed.wait(_WAIT_SEC)
            return original(*args, **kwargs)

        monkeypatch.setattr(task_admission, name, gate)
        assert env.dispatch_returns(event) and reached.wait(_WAIT_SEC)
    with database_reader(env.data, exclusive=True):
        if seam == "entry":
            assert env.dispatch_returns(event)  # the queued marker is the lane's first contended write
        proceed.set()
        assert not env.gen.wait_idle(0.5)  # retried at its own owner: one live claim, no bus copy
        assert bus.empty() and env.pending_ids() == [] and copies == []
        assert [claim.task_id for claim in prep._LIVE.values()] == ["readonly-busy"]
    assert env.gen.wait_idle(_WAIT_SEC)
    assert env.pending_ids() == ["readonly-busy"] and copies == [] and bus.empty()
    row = env.row("readonly-busy")
    assert row["status"] == STATUS_SCHEDULED and row[prep.MARKER]["phase"] == prep.ADMITTED
    assert not any("not scheduled" in text for _, text, _ in env.ctx.sent)


@pytest.mark.parametrize("owner", ["cancelled", "foreign_marker"])
def test_readonly_claim_retrying_contention_never_displaces_new_custody(env, monkeypatch, owner):
    from ouroboros import cancel_intents
    from ouroboros.task_results import write_task_result
    from supervisor import workers

    monkeypatch.setattr(workers, "WORKERS", {})
    bus = env.ctx.event_queue = stdqueue.Queue()
    event = env.readonly("readonly-held")
    reached, proceed = threading.Event(), threading.Event()
    original = task_admission.parse_schedule_task_depth

    def gate(*args, **kwargs):
        if not reached.is_set():
            reached.set()
            assert proceed.wait(_WAIT_SEC)
        return original(*args, **kwargs)

    monkeypatch.setattr(task_admission, "parse_schedule_task_depth", gate)
    assert env.dispatch_returns(event) and reached.wait(_WAIT_SEC)
    if owner == "cancelled":
        cancel_intents.request_cancel(env.data, "readonly-held", reason="owner Stop", source="test")
        queue.cancel_task_custody("readonly-held")
        assert env.row("readonly-held")["status"] == "cancelled"
    else:
        write_task_result(env.data, "readonly-held", "requested",
                          workspace_preparation={"token": "foreign-token", "phase": "queued"})
    before = env.row("readonly-held")
    proceed.set()
    assert env.gen.wait_idle(_WAIT_SEC)
    assert env.pending_ids() == [] and env.row("readonly-held") == before and bus.empty()
    assert queue.ADMISSION_RESERVATIONS == {} and not prep._LIVE


def test_closed_generation_retains_its_chosen_refusal_across_real_contention(env, monkeypatch):
    reached, proceed, busy_seen = threading.Event(), threading.Event(), threading.Event()
    original_refuse = prep.PreparationClaim.refuse
    original_persist = task_admission.persist_never_admitted_refusal

    def refuse(claim, detail, reason_code, **kwargs):
        if reason_code == prep.INTERRUPTED_BY_GENERATION_EXIT:
            reached.set()
            assert proceed.wait(_WAIT_SEC)
        return original_refuse(claim, detail, reason_code, **kwargs)

    def persist(*args, **kwargs):
        try:
            return original_persist(*args, **kwargs)
        except store.StoreBusy:
            busy_seen.set()
            raise

    monkeypatch.setattr(prep.PreparationClaim, "refuse", refuse)
    monkeypatch.setattr(task_admission, "persist_never_admitted_refusal", persist)
    with env.copy_blocked():
        assert env.dispatch_returns(env.request("closed-busy"))
        env.wait_phase("closed-busy", prep.COPYING)
        prep.close_generation(env.gen)
    assert reached.wait(_WAIT_SEC)
    with database_reader(env.data, exclusive=True):
        proceed.set()
        assert busy_seen.wait(3)
        assert not env.gen.wait_idle(0), "refusal ownership was dropped while the store was busy"
        assert env.pending_ids() == []
    assert env.gen.wait_idle(_WAIT_SEC)
    stored = env.row("closed-busy")
    assert stored["status"] == "failed"
    assert stored["reason_code"] == prep.INTERRUPTED_BY_GENERATION_EXIT
    assert stored[prep.MARKER]["phase"] == prep.REFUSED
