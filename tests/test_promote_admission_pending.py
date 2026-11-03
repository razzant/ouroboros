"""#1160: an emitted promote is a durable pending fact, never "unknown".

A promote whose admission the supervisor has not confirmed inside the
confirmation window answered `PROMOTE_UNCONFIRMED`; asking `get_task_result`
for that id then said "unknown or not yet registered", which reads as "your
promote never happened" and invites a second promote — that is how one
conversation produced duplicate roots here and on a foreign Windows install.
"""

from __future__ import annotations

import types

import pytest

from tests.test_swarm_host_admission import host as _swarm_host

swarm_host = _swarm_host  # the real supervisor handler needs a real host


class _DeadQueue:
    """A supervisor that accepted the event and has not answered yet."""

    def __init__(self):
        self.events = []

    def put_nowait(self, event):
        self.events.append(event)


def _promote_ctx(tmp_path, event_queue):
    return types.SimpleNamespace(
        pending_events=[], event_queue=event_queue, current_chat_id=1,
        drive_root=tmp_path, budget_drive_root=str(tmp_path), project_id="",
        task_metadata={}, task_id="",
    )


@pytest.fixture
def _short_confirmation_window(tmp_path, monkeypatch):
    """The busy-supervisor case, without spending its real window in a test."""
    from ouroboros.tools import control_events

    monkeypatch.setattr(control_events, "_PROMOTE_CONFIRM_TIMEOUT_SEC", 0.05)
    monkeypatch.setattr(control_events, "_PROMOTE_CONFIRM_POLL_SEC", 0.005)
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)


def test_an_unconfirmed_promote_reads_as_a_pending_admission(tmp_path, _short_confirmation_window):
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.control_routing import _start_task
    from ouroboros.tools.control_task_results import _get_task_result

    ctx = _promote_ctx(tmp_path, _DeadQueue())
    out = _start_task(ctx, "Continue the racer", title="Racer continuation", workspace="none")
    assert out.startswith("Accepted, waiting to start:")
    task_id = ctx.event_queue.events[0]["task_id"]
    assert f"get_task_result({task_id})" in out

    stub = load_task_result(tmp_path, task_id)
    assert stub["promotion_admission"]["status"] == "emitted"
    assert stub["promotion_admission"]["routing_token"] == ctx.event_queue.events[0]["routing_token"]
    emitted_at = stub["promotion_admission"]["prepared_at"]
    assert emitted_at

    read = _get_task_result(ctx, task_id)
    assert f"admission pending since {emitted_at}" in read
    assert "unknown or not yet registered" not in read
    assert f"get_task_result({task_id})" in read

    # The quiet direction: the pending sentence belongs to an emitted stub alone,
    # so an id nobody ever promoted is still honestly unknown.
    assert "unknown or not yet registered" in _get_task_result(ctx, "never-promoted")


def test_a_confirmed_admission_reads_as_the_ordinary_result(tmp_path, _short_confirmation_window):
    """The quiet direction: once the supervisor answers, the pending sentence is
    gone and the result reads exactly as before."""
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.control_routing import _start_task
    from ouroboros.tools.control_task_results import _get_task_result

    ctx = _promote_ctx(tmp_path, _DeadQueue())
    _start_task(ctx, "Continue the racer", title="Racer continuation",
                          workspace="none")
    event = ctx.event_queue.events[0]
    task_id = event["task_id"]

    write_task_result(
        tmp_path, task_id, "scheduled", root_task_id=task_id, delegation_role="root",
        description="Continue the racer",
        promotion_admission={"status": "scheduled", "routing_token": event["routing_token"],
                             "confirmed_at": "2026-09-21T00:00:00Z"},
        result="Task accepted and durably scheduled.",
    )

    read = _get_task_result(ctx, task_id)
    assert "admission pending since" not in read
    assert "Task accepted and durably scheduled." in read


def test_a_fast_admission_leaves_no_emitted_stub_behind(tmp_path, _short_confirmation_window):
    """Guard quiet: the stub only ever initializes ABSENCE, so a supervisor that
    answered before the tool wrote sees its scheduled receipt untouched."""
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.tools.control_routing import _start_task

    class _FastSupervisor:
        def __init__(self):
            self.events = []

        def put_nowait(self, event):
            self.events.append(event)
            write_task_result(
                tmp_path, event["task_id"], "scheduled", root_task_id=event["task_id"],
                delegation_role="root", description=event["objective"],
                promotion_admission={"status": "scheduled",
                                     "routing_token": event["routing_token"],
                                     "confirmed_at": "2026-09-21T00:00:00Z"},
                result="Task accepted and durably scheduled.",
            )

    ctx = _promote_ctx(tmp_path, _FastSupervisor())
    out = _start_task(ctx, "Build the racer", title="Racer build", workspace="none")
    assert out.startswith("OK: task")

    stored = load_task_result(tmp_path, ctx.event_queue.events[0]["task_id"])
    assert stored["status"] == "scheduled"
    assert stored["promotion_admission"]["status"] == "scheduled"
    assert "emitted_at" not in stored["promotion_admission"]


def test_a_late_admission_replaces_the_stub_it_finds(tmp_path, _short_confirmation_window):
    """The other side of the race: the admission receipt is a whole field, so an
    emitted stub written first leaves nothing in the scheduled row."""
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.tools.control_routing import _start_task

    ctx = _promote_ctx(tmp_path, _DeadQueue())
    _start_task(ctx, "Build the racer", title="Racer build", workspace="none")
    event = ctx.event_queue.events[0]

    write_task_result(
        tmp_path, event["task_id"], "scheduled", root_task_id=event["task_id"],
        delegation_role="root",
        promotion_admission={"status": "scheduled", "routing_token": event["routing_token"],
                             "confirmed_at": "2026-09-21T00:00:00Z"},
    )
    stored = load_task_result(tmp_path, event["task_id"])
    assert stored["promotion_admission"] == {
        "status": "scheduled", "routing_token": event["routing_token"],
        "confirmed_at": "2026-09-21T00:00:00Z",
    }


def test_the_emitted_stub_never_owns_its_own_admissions_id(tmp_path, monkeypatch, _short_confirmation_window):
    """Positive scheduling authority stays with the supervisor: the stub must not
    make the promote it belongs to look like a duplicate id at any of the three
    admission gates, while a real durable row still does."""
    import supervisor.queue as supervisor_queue
    import supervisor.workers as workers
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.control_routing import _start_task

    ctx = _promote_ctx(tmp_path, _DeadQueue())
    _start_task(ctx, "Build the racer", title="Racer build", workspace="none")
    event = ctx.event_queue.events[0]
    task_id, token = event["task_id"], event["routing_token"]

    monkeypatch.setattr(supervisor_queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(supervisor_queue, "PENDING", [])
    monkeypatch.setattr(supervisor_queue, "RUNNING", {})
    monkeypatch.setattr(supervisor_queue, "ADMISSION_RESERVATIONS", {})

    assert supervisor_queue.reserve_task_admission(
        task_id, token, drive_root=tmp_path,
    )["status"] == "reserved"
    assert workers._promote_duplicate_reason(
        task_id, types.SimpleNamespace(DRIVE_ROOT=tmp_path, PENDING=[], RUNNING={}),
        admission_token=token,
    ) == ""
    queued = supervisor_queue.enqueue_task({
        "id": task_id, "type": "task", "_require_unique_task_id": True,
        "_admission_token": token,
    })
    assert "_admission_blocked" not in queued

    write_task_result(tmp_path, "other-root", "completed", description="someone else's work")
    assert supervisor_queue.reserve_task_admission(
        "other-root", token, drive_root=tmp_path,
    ) == {"status": "blocked", "reason": "duplicate_task_id"}
    assert workers._promote_duplicate_reason(
        "other-root", types.SimpleNamespace(DRIVE_ROOT=tmp_path, PENDING=[], RUNNING={}),
        admission_token=token,
    ) == "duplicate_task_id"


def test_a_stub_is_read_around_only_by_the_token_that_wrote_it(tmp_path, monkeypatch, _short_confirmation_window):
    """The other direction of the three gates: a promote's stub belongs to its own
    routing token. Any OTHER admission meets it as a row that owns the id, and a
    `requested` row that carries no emitted admission is never a stub at all."""
    import supervisor.queue as supervisor_queue
    import supervisor.workers as workers
    from ouroboros.routing_wait import is_emitted_admission_stub, is_own_admission_stub
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.control_routing import _start_task

    ctx = _promote_ctx(tmp_path, _DeadQueue())
    _start_task(ctx, "Build the racer", title="Racer build", workspace="none")
    task_id = ctx.event_queue.events[0]["task_id"]
    monkeypatch.setattr(supervisor_queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(supervisor_queue, "PENDING", [])
    monkeypatch.setattr(supervisor_queue, "RUNNING", {})
    monkeypatch.setattr(supervisor_queue, "ADMISSION_RESERVATIONS", {})
    host = types.SimpleNamespace(DRIVE_ROOT=tmp_path, PENDING=[], RUNNING={})

    assert supervisor_queue.reserve_task_admission(
        task_id, "other-token", drive_root=tmp_path) == {"status": "blocked", "reason": "duplicate_task_id"}
    assert workers._promote_duplicate_reason(task_id, host, admission_token="other-token") == "duplicate_task_id"
    queued = supervisor_queue.enqueue_task({
        "id": task_id, "type": "task", "_require_unique_task_id": True, "_admission_token": "other-token"})
    assert queued.get("_admission_blocked") == "duplicate_task_id"

    stub = load_task_result(tmp_path, task_id)
    assert is_emitted_admission_stub(stub) and not is_emitted_admission_stub(stub, "other-token")
    # The gates' form: no token is no claim, so a tokenless admission never reads around it.
    assert is_own_admission_stub(stub, stub["promotion_admission"]["routing_token"])
    assert not is_own_admission_stub(stub, "") and not is_own_admission_stub(stub, "other-token")
    assert supervisor_queue.enqueue_task({
        "id": task_id, "type": "task", "_require_unique_task_id": True,
    }).get("_admission_blocked") == "duplicate_task_id"
    assert not is_emitted_admission_stub({**stub, "status": "completed"})
    assert not is_emitted_admission_stub({**stub, "promotion_admission": {"status": "scheduled"}})


def test_one_runtime_limit_bounds_both_promote_confirmation_waits():
    """The two confirmation windows were byte-identical literals in two modules."""
    from ouroboros import routing_wait, runtime_limits
    from ouroboros.tools import control_events

    assert runtime_limits.get_promote_confirm_wait_sec() == 15.0
    # The routing leaf reads the bound at the wait itself (no import-time edge): its
    # default IS the getter, and an explicit caller bound still wins, floored at zero.
    assert routing_wait._confirm_wait_sec(None) == runtime_limits.get_promote_confirm_wait_sec()
    assert routing_wait._confirm_wait_sec(0.05) == 0.05 and routing_wait._confirm_wait_sec(-1) == 0.0
    assert control_events._PROMOTE_CONFIRM_TIMEOUT_SEC == runtime_limits.get_promote_confirm_wait_sec()


def _emit_into(root, monkeypatch):
    """A promote emitted at a supervisor that has not answered: the stub is on disk."""
    from ouroboros.tools import control_events
    from ouroboros.tools.control_routing import _start_task

    monkeypatch.setattr(control_events, "_PROMOTE_CONFIRM_TIMEOUT_SEC", 0.05)
    monkeypatch.setattr(control_events, "_PROMOTE_CONFIRM_POLL_SEC", 0.005)
    ctx = _promote_ctx(root, _DeadQueue())
    out = _start_task(ctx, "Build the racer", title="Racer build", workspace="none")
    assert out.startswith("Accepted, waiting to start:"), out
    return ctx, ctx.event_queue.events[0]


def _attachments_rejected(task_id):
    return {"status": "needs_manual_target", "reason": "attachment_admission_rejected",
            "detail": "- a.png: rejected (reason=staging_unavailable, ordinal=0)",
            "task_id": task_id, "attachment_manifest": []}


def test_a_refusal_that_writes_no_result_of_its_own_still_replaces_the_stub(swarm_host, monkeypatch):
    """The attachment refusal persists no task result by design. Over an emitted
    stub that silence left "admission pending" on disk for ever, so the model was
    told not to promote again and the owner's work never started."""
    import supervisor.workers as workers
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.control_task_results import _get_task_result
    from supervisor.events_project_routing import _handle_promote_chat_to_task

    ctx, event = _emit_into(swarm_host.root, monkeypatch)
    task_id = event["task_id"]
    assert load_task_result(swarm_host.root, task_id)["promotion_admission"]["status"] == "emitted"
    monkeypatch.setattr(workers, "promote_chat_to_task", lambda evt, c: _attachments_rejected(task_id))

    _handle_promote_chat_to_task(event, swarm_host.ctx)

    stored = load_task_result(swarm_host.root, task_id)
    assert stored["status"] == "failed"
    assert stored["promotion_admission"]["status"] == "rejected"
    assert stored["promotion_admission"]["reason"] == "attachment_admission_rejected"
    read = _get_task_result(ctx, task_id)
    assert "admission pending" not in read and "attachment_admission_rejected" in read


def test_the_attachment_refusal_without_a_stub_still_writes_no_result(swarm_host, monkeypatch):
    """The quiet direction: that refusal's own contract is untouched. With no
    emitted stub on disk (a host-issued promote) it persists nothing, as before."""
    import supervisor.workers as workers
    from ouroboros.task_results import load_task_result
    from supervisor.events_project_routing import _handle_promote_chat_to_task

    event = {"type": "promote_chat_to_task", "task_id": "hostpromote000001", "routing_token": "tok-host",
             "objective": "Build the racer", "chat_id": 1, "workspace": "none"}
    monkeypatch.setattr(workers, "promote_chat_to_task",
                        lambda evt, c: _attachments_rejected("hostpromote000001"))

    _handle_promote_chat_to_task(event, swarm_host.ctx)

    assert not load_task_result(swarm_host.root, "hostpromote000001")


def test_the_pending_sentence_never_forbids_a_new_promote_for_ever(tmp_path, _short_confirmation_window):
    """The row proves one thing: emitted, no receipt yet. An event can be lost (a
    supervisor restart drops its in-memory queue) and a source can take minutes to
    prepare, and the row cannot tell the two apart; so the sentence gives the facts
    and hands the decision back, with neither a standing ban nor an invented cause."""
    from ouroboros.tools.control_routing import _start_task
    from ouroboros.tools.control_task_results import _get_task_result

    ctx = _promote_ctx(tmp_path, _DeadQueue())
    _start_task(ctx, "Continue the racer", title="Racer continuation",
                          workspace="none")
    task_id = ctx.event_queue.events[0]["task_id"]

    read = _get_task_result(ctx, task_id)
    assert "admission pending since" in read and f"get_task_result({task_id})" in read
    assert "do not promote the same work a second time" not in read
    # Facts only: the row cannot tell a lost event from a source still being prepared,
    # so the sentence names the longest honest wait and leaves the judgement to the model.
    assert "did not land" not in read and "within seconds" not in read
    assert "elapsed time alone cannot prove" in read and "NEW task id" in read
