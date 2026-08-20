"""Focused durability coverage for abnormal terminal and replay-safe paths."""

from __future__ import annotations

import inspect
import pathlib
import tempfile
from types import SimpleNamespace

import pytest


def _available_cost_fields(*, calls: int = 0, degraded: bool = False) -> dict:
    """The REAL projection over an empty ledger, not a hand-copied mirror of it.

    This used to be a ten-key dict literal. It was the reason a projection key
    the terminal emitter could not bind stayed invisible in a green suite: every
    test that reached the emitter substituted this stale copy for the real
    `reconstruct_task_cost`, so the shape under test was the one the test author
    remembered rather than the one production builds."""
    from supervisor.state import reconstruct_task_cost

    fields = reconstruct_task_cost(
        "no-such-task", fields=True, drive_root=pathlib.Path(tempfile.mkdtemp()),
    )
    assert fields["cost_accounting_status"] == "available"
    fields["total_rounds"] = calls
    fields["ledger_integrity_degraded"] = degraded
    return fields


def test_headless_worker_crash_emits_task_done_without_main_chat_reroute(tmp_path, monkeypatch):
    from supervisor import queue, workers

    class DeadProc:
        pid = None
        exitcode = -11

        @staticmethod
        def is_alive():
            return False

        @staticmethod
        def join(timeout=None):
            del timeout

    metadata = {"managed_update": {"authority_fingerprint": "host-bound"}}
    task = {
        "id": "headless-crash", "type": "task", "chat_id": 0, "_attempt": 1,
        "metadata": metadata,
    }
    worker = SimpleNamespace(wid=0, busy_task_id=task["id"], proc=DeadProc(), reaping=False)
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "WORKERS", {0: worker})
    monkeypatch.setattr(workers, "RUNNING", {
        task["id"]: {
            "task": task,
            "started_at": 1.0,
            "last_heartbeat_at": 1.0,
            "attempt": 1,
        },
    })
    monkeypatch.setattr(workers, "QUEUE_MAX_RETRIES", 1)
    monkeypatch.setattr(workers, "_LAST_SPAWN_TIME", 0)
    monkeypatch.setattr(workers, "CRASH_TS", [])
    events = []
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=events.append))
    monkeypatch.setattr(workers, "reconstruct_task_cost", lambda *_a, **_k: _available_cost_fields())
    monkeypatch.setattr(workers, "respawn_worker", lambda _wid: None)
    monkeypatch.setattr(workers, "send_with_budget", lambda *_a, **_k: None)
    monkeypatch.setattr(workers, "load_state", lambda: {})
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": None)
    monkeypatch.setattr(queue, "enqueue_task", lambda *_a, **_k: None)
    monkeypatch.setattr("ouroboros.tools.services.archive_task_service_logs", lambda *_a, **_k: None)
    monkeypatch.setattr("ouroboros.task_results.load_task_result", lambda *_a, **_k: None)
    monkeypatch.setattr("ouroboros.task_results.write_task_result", lambda *_a, **_k: None)

    workers.ensure_workers_healthy()

    terminal = [event for event in events if event.get("type") == "task_done"]
    assert len(terminal) == 1
    assert terminal[0]["task_id"] == task["id"]
    assert terminal[0]["chat_id"] == 0
    assert terminal[0]["metadata"] == metadata


def test_terminal_helper_preserves_task_metadata(monkeypatch):
    from supervisor import workers

    metadata = {"managed_update": {"authority_fingerprint": "host-bound"}}
    events = []
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=events.append))

    assert workers._emit_task_done_terminal(
        {"id": "resolver", "type": "task", "metadata": metadata}, "resolver"
    )
    assert events[0]["metadata"] == metadata


def test_headless_pending_cancel_still_emits_task_done(tmp_path, monkeypatch):
    from supervisor import queue, workers

    task = {"id": "headless-cancel", "type": "task", "chat_id": 0}
    events = []
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "PENDING", [task])
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=events.append))
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": None)

    assert queue.cancel_task_by_id(task["id"]) is True

    terminal = [event for event in events if event.get("type") == "task_done"]
    assert len(terminal) == 1
    assert terminal[0]["task_id"] == task["id"]
    assert terminal[0]["chat_id"] == 0
    assert terminal[0]["status"] == "cancelled"


def _patch_reaper(tmp_path, monkeypatch):
    from supervisor import queue, workers

    events = []
    enqueued = []
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(queue, "QUEUE_MAX_RETRIES", 1)
    monkeypatch.setattr(queue, "reconstruct_task_cost", lambda *_a, **_k: _available_cost_fields())
    monkeypatch.setattr(queue, "enqueue_task", lambda task, front=False: enqueued.append((dict(task), front)))
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": None)
    monkeypatch.setattr(queue, "_kept_service_pids", lambda: set(), raising=False)
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=events.append))
    monkeypatch.setattr("ouroboros.tools.services.archive_task_service_logs", lambda *_a, **_k: None)
    monkeypatch.setattr("ouroboros.headless.copy_child_task_result", lambda *_a, **_k: None)
    monkeypatch.setattr("ouroboros.observability.latest_llm_response_text", lambda *_a, **_k: "")
    monkeypatch.setattr("ouroboros.owner_mailbox.cleanup_task_mailbox", lambda *_a, **_k: None)
    return events, enqueued


def test_headless_reaper_still_emits_task_done(tmp_path, monkeypatch):
    from supervisor.task_reaper import reap_timed_out_task

    events, enqueued = _patch_reaper(tmp_path, monkeypatch)
    reap_timed_out_task({
        "worker_id": 4,
        "proc": None,
        "task_id": "headless-reaped",
        "task": {"id": "headless-reaped", "type": "task", "chat_id": 0},
        "task_type": "task",
        "terminal_reason": "idle_timeout",
        "attempt": 1,
        "owner_chat_id": 0,
        "will_retry": False,
    })

    terminal = [event for event in events if event.get("type") == "task_done"]
    assert enqueued == []
    assert len(terminal) == 1
    assert terminal[0]["task_id"] == "headless-reaped"
    assert terminal[0]["chat_id"] == 0


def test_self_finalized_reaper_preserves_evolution_transaction_metadata(
    tmp_path, monkeypatch,
):
    from ouroboros.task_results import STATUS_CANCELLED, write_task_result
    from supervisor.task_reaper import reap_timed_out_task

    events, enqueued = _patch_reaper(tmp_path, monkeypatch)
    tx = {"campaign_id": "camp", "transaction_id": "tx", "task_id": "evo-reaped"}
    task = {
        "id": "evo-reaped",
        "type": "evolution",
        "chat_id": 0,
        "metadata": {
            "evolution_transaction": tx,
            "secret_sentinel": "must-not-reach-terminal-event",
            "workspace_root": "/private/workspace",
        },
    }
    write_task_result(tmp_path, task["id"], STATUS_CANCELLED, result="cancelled")

    reap_timed_out_task({
        "worker_id": 4,
        "proc": None,
        "task_id": task["id"],
        "task": task,
        "task_type": "evolution",
        "terminal_reason": "idle_timeout",
        "attempt": 1,
        "owner_chat_id": 0,
        "will_retry": False,
    })

    terminal = [event for event in events if event.get("type") == "task_done"]
    assert enqueued == []
    assert len(terminal) == 1
    assert terminal[0]["metadata"] == {"evolution_transaction": tx}


def test_self_finalized_reaper_event_preserves_managed_update_metadata(
    tmp_path, monkeypatch,
):
    """The assisted-merge watchdog in _handle_task_done releases the writer gate
    from the terminal event's metadata when the task already left RUNNING, so a
    reaped resolver task must forward managed_update — and nothing else."""
    from ouroboros.task_results import write_task_result
    from supervisor.task_reaper import reap_timed_out_task

    events, enqueued = _patch_reaper(tmp_path, monkeypatch)
    managed = {"authority_fingerprint": "host-bound"}
    write_task_result(tmp_path, "self-finalized", "completed", result="done")

    reap_timed_out_task({
        "worker_id": 4,
        "proc": None,
        "task_id": "self-finalized",
        "task": {
            "id": "self-finalized",
            "type": "task",
            "metadata": {
                "managed_update": managed,
                "secret_sentinel": "must-not-reach-terminal-event",
            },
        },
        "task_type": "task",
        "terminal_reason": "idle_timeout",
        "attempt": 1,
        "owner_chat_id": 0,
        "will_retry": False,
    })

    terminal = [event for event in events if event.get("type") == "task_done"]
    assert enqueued == []
    assert terminal[-1]["metadata"] == {"managed_update": managed}


def test_top_level_retry_preserves_logical_root_and_typed_attempt_lineage(
    tmp_path, monkeypatch,
):
    from ouroboros.task_results import resolve_task_lineage
    from ouroboros.task_results import STATUS_SCHEDULED, load_task_result
    from supervisor.task_reaper import reap_timed_out_task

    _events, enqueued = _patch_reaper(tmp_path, monkeypatch)
    reap_timed_out_task({
        "worker_id": 4,
        "proc": None,
        "task_id": "old-root",
        "task": {
            "id": "old-root",
            "type": "task",
            "chat_id": 0,
            "root_task_id": "old-root",
            "parent_task_id": "",
            "delegation_role": "root",
            "metadata": {
                "task_id": "old-root",
                "root_task_id": "old-root",
                "parent_task_id": "stale-parent",
                "delegation_role": "root",
            },
        },
        "task_type": "task",
        "terminal_reason": "idle_timeout",
        "attempt": 1,
        "owner_chat_id": 0,
        "will_retry": True,
        "retry_task_id": "new-root",
    })

    assert len(enqueued) == 1
    queued, front = enqueued[0]
    assert front is True
    assert queued["id"] == "new-root"
    assert queued["root_task_id"] == "old-root"
    assert queued["parent_task_id"] == ""
    assert queued["original_task_id"] == "old-root"
    assert queued["timeout_retry_from"] == "old-root"
    assert resolve_task_lineage(
        queued["id"],
        metadata=queued["metadata"],
        root_task_id=queued["root_task_id"],
        parent_task_id=queued["parent_task_id"],
        delegation_role=queued["delegation_role"],
        original_task_id=queued["original_task_id"],
        timeout_retry_from=queued["timeout_retry_from"],
    )["is_root_task"] is True

    scheduled = load_task_result(tmp_path, "new-root")
    assert scheduled["status"] == STATUS_SCHEDULED
    assert scheduled["root_task_id"] == "old-root"
    assert not scheduled.get("parent_task_id")
    assert scheduled["original_task_id"] == "old-root"
    assert scheduled["timeout_retry_from"] == "old-root"
    assert scheduled["delegation_role"] == "root"


def test_same_id_subagent_retry_preserves_parent_and_root_lineage(
    tmp_path, monkeypatch,
):
    from supervisor.task_reaper import reap_timed_out_task

    _events, enqueued = _patch_reaper(tmp_path, monkeypatch)
    child = {
        "id": "child-retry",
        "type": "task",
        "chat_id": 0,
        "root_task_id": "root",
        "parent_task_id": "parent",
        "delegation_role": "subagent",
        "metadata": {
            "task_id": "child-retry",
            "root_task_id": "root",
            "parent_task_id": "parent",
            "delegation_role": "subagent",
        },
    }
    reap_timed_out_task({
        "worker_id": 4,
        "proc": None,
        "task_id": "child-retry",
        "task": child,
        "task_type": "task",
        "terminal_reason": "idle_timeout",
        "attempt": 1,
        "owner_chat_id": 0,
        "will_retry": True,
        "retry_task_id": "child-retry",
    })

    queued, front = enqueued[0]
    assert front is True
    assert queued["id"] == "child-retry"
    assert queued["root_task_id"] == "root"
    assert queued["parent_task_id"] == "parent"
    assert queued["metadata"]["root_task_id"] == "root"
    assert queued["metadata"]["parent_task_id"] == "parent"


def test_retry_terminal_cost_uses_logical_root_authority(tmp_path, monkeypatch):
    from supervisor import events

    monkeypatch.setattr(
        "supervisor.state.reconstruct_task_cost",
        lambda *_args, **_kwargs: {
            "cost_accounting_status": "available",
            "cost_usd": 0.75,
            "cost_final": True,
        },
    )
    seen = []
    monkeypatch.setattr(
        "ouroboros.usage_accounting.usage_breakdown",
        lambda root, *, root_task_id="", **_kwargs: (
            seen.append((root, root_task_id))
            or {"accounted_usd": 2.0, "cost_final": True}
        ),
    )
    task = {
        "id": "retry-2",
        "root_task_id": "logical-root",
        "parent_task_id": "",
        "delegation_role": "root",
        "original_task_id": "retry-1",
        "timeout_retry_from": "retry-1",
        "budget_drive_root": str(tmp_path),
    }

    projection = events._authoritative_terminal_cost(
        "retry-2", task, dict(task), {}, tmp_path,
    )

    assert seen == [(tmp_path, "logical-root")]
    assert projection["cost_usd_with_children"] == 2.0
    assert projection["cost_final"] is True


def test_a_root_narrowed_by_a_childs_open_row_reports_that_rows_count(tmp_path, monkeypatch):
    """THIRD site of one class: `non_final_rows` is `cost_final`'s DISCLOSED CAUSE and
    rides with it by contract (`task_results.py`: "a projection reporting cost_final:
    false with every dollar bucket at zero ... is a flag no reader can reconstruct").

    The root branch re-derives `cost_final` against the SUBTREE — so a root whose own
    rows are all settled goes non-final purely because a CHILD still has one open. It
    left the count describing this task alone, which for exactly that root is 0: the
    projection said "not final, caused by nothing"."""
    from supervisor import events

    monkeypatch.setattr(
        "supervisor.state.reconstruct_task_cost",
        lambda *_args, **_kwargs: {
            # This task's OWN rows are complete and final.
            "cost_accounting_status": "available", "cost_usd": 0.75,
            "cost_final": True, "non_final_rows": 0,
        },
    )
    monkeypatch.setattr(
        "ouroboros.usage_accounting.usage_breakdown",
        lambda root, *, root_task_id="", **_kwargs: {
            "accounted_usd": 2.0, "cost_final": False, "non_final_rows": 3,
        },
    )
    task = {"id": "root-1", "root_task_id": "root-1", "parent_task_id": "",
            "delegation_role": "root", "budget_drive_root": str(tmp_path)}

    projection = events._authoritative_terminal_cost("root-1", task, dict(task), {}, tmp_path)

    assert projection["cost_final"] is False
    assert projection["cost_with_children_partial"] is True
    # The cause travels with the flag it explains, and it is the SUBTREE's count.
    assert projection["non_final_rows"] == 3


def test_reaper_admission_block_terminalizes_retry(tmp_path, monkeypatch):
    from ouroboros.task_results import STATUS_FAILED, load_task_result
    from supervisor import queue
    from supervisor.task_reaper import reap_timed_out_task

    events, _ = _patch_reaper(tmp_path, monkeypatch)
    monkeypatch.setattr(
        queue,
        "enqueue_task",
        lambda *_args, **_kwargs: {"_admission_blocked": "task_acceptance_fence"},
    )

    reap_timed_out_task({
        "worker_id": 4,
        "proc": None,
        "task_id": "fenced-retry",
        "task": {"id": "fenced-retry", "type": "task", "chat_id": 0},
        "task_type": "task",
        "terminal_reason": "idle_timeout",
        "attempt": 1,
        "owner_chat_id": 0,
        "will_retry": True,
    })

    result = load_task_result(tmp_path, "fenced-retry")
    assert result["status"] == STATUS_FAILED
    assert result["reason_code"] == "idle_timeout_retry_admission_blocked"
    terminal = [event for event in events if event.get("type") == "task_done"]
    assert terminal and terminal[-1]["status"] == "failed"


def test_assign_keeps_unsafe_pending_when_terminal_write_is_not_durable(tmp_path, monkeypatch):
    from supervisor import queue, state, workers

    task = {
        "id": "unsafe-write-failure",
        "type": "task",
        "chat_id": 0,
        "_attempt": 2,
        "original_task_id": "first-attempt",
    }
    events = []
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "PENDING", [task])
    monkeypatch.setattr(workers, "RUNNING", {})
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(workers, "load_state", lambda: {"owner_chat_id": 0})
    monkeypatch.setattr(workers, "reconstruct_task_cost", lambda *_a, **_k: _available_cost_fields(calls=1))
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=events.append))
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 0.0)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": None)
    monkeypatch.setattr(
        "ouroboros.task_results.write_task_result",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("durable write failed")),
    )

    workers.assign_tasks()

    assert [item["id"] for item in workers.PENDING] == [task["id"]]
    assert "_budget_pause" not in workers.PENDING[0]
    assert not any(event.get("type") == "task_done" for event in events)


@pytest.mark.parametrize(
    ("corruption", "expected_error"),
    (("quarantined_tail", "replay_unsafe"), ("midstream", "accounting_unavailable")),
)
def test_corrupt_or_integrity_degraded_ledger_never_permits_budget_resume(
    tmp_path, monkeypatch, corruption, expected_error,
):
    from ouroboros import usage_accounting as accounting
    from supervisor import queue, state, workers

    state.init(tmp_path, total_budget_limit=10.0)
    queue.init(tmp_path)
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "PENDING", [{
        "id": "replay-risk",
        "type": "task",
        "chat_id": 0,
        "_budget_pause": {
            "status": "paused_before_dispatch",
            "physical_calls": 0,
            "replay_safe": True,
            "auto_resume": False,
        },
    }])
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": None)

    reservation = accounting.reserve_attempt(accounting.AttemptRequest(
        model="test/model",
        provider="test",
        drive_root=tmp_path,
        task_id="replay-risk",
        root_task_id="replay-risk",
        reservation_usd=0.01,
        global_limit_usd=10.0,
    ))
    accounting.release_attempt(reservation, "test_setup")
    ledger = tmp_path / accounting.LEDGER_REL
    if corruption == "quarantined_tail":
        with ledger.open("ab") as handle:
            handle.write(b'{"seq":')
    else:
        lines = ledger.read_text(encoding="utf-8").splitlines()
        ledger.write_text(lines[0] + "\nnot-json\n" + lines[1] + "\n", encoding="utf-8")

    result = queue.resume_budget_paused_task("replay-risk")

    assert result == {
        "ok": False,
        "error": expected_error,
        "action": "cancel_or_new_run",
    }
    assert "_budget_pause" in queue.PENDING[0]


def test_reaper_suppresses_retry_when_terminal_result_write_fails(tmp_path, monkeypatch):
    from supervisor.task_reaper import reap_timed_out_task

    events, enqueued = _patch_reaper(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "ouroboros.task_results.write_task_result",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("durable write failed")),
    )

    reap_timed_out_task({
        "worker_id": 7,
        "proc": None,
        "task_id": "retry-needs-terminal",
        "task": {"id": "retry-needs-terminal", "type": "task", "chat_id": 0},
        "task_type": "task",
        "terminal_reason": "idle_timeout",
        "attempt": 1,
        "owner_chat_id": 0,
        "will_retry": True,
        "retry_task_id": "retry-needs-terminal",
    })

    assert enqueued == []
    terminal = [event for event in events if event.get("type") == "task_done"]
    assert len(terminal) == 1
    assert terminal[0]["task_id"] == "retry-needs-terminal"
    assert terminal[0]["status"] == "failed"


# ---------------------------------------------------------------------------
# The projection/consumer seam.
#
# Three times in this range a key was added to a cost projection and a consumer
# that could not carry it was missed. The suite stayed green each time because
# every test that reached a consumer handed it a HAND-WRITTEN dict, so no test
# ever asked whether the object the consumer receives in production is an object
# the consumer can accept. These drive the real functions with the real
# projection instead.
# ---------------------------------------------------------------------------


def _real_projection(tmp_path):
    from supervisor.state import reconstruct_task_cost

    projection = reconstruct_task_cost("no-such-task", fields=True, drive_root=tmp_path)
    assert projection["cost_accounting_status"] == "available"
    return projection


def test_every_splat_consumer_binds_the_real_cost_projection(tmp_path):
    """`reconstruct_task_cost(fields=True)` is splatted whole into these callables.

    A key the callee's signature cannot bind is a TypeError on a live teardown
    path, not a dropped field -- and on the budget-stop path it is a SILENT one,
    swallowed by a try/except that then never removes the task from PENDING."""
    from ouroboros.task_results import write_task_result

    projection = _real_projection(tmp_path)
    unbindable = {}
    for name, fn, leading in (
        ("write_task_result", write_task_result, (tmp_path, "t1", "failed")),
    ):
        try:
            inspect.signature(fn).bind(*leading, **projection)
        except TypeError as exc:
            unbindable[name] = str(exc)
    assert unbindable == {}, (
        "a consumer cannot bind the projection its call site splats into it: "
        f"{unbindable}"
    )


def test_terminal_emitter_names_no_cost_field_and_forwards_an_unknown_one(tmp_path):
    """The emitter left the splat-consumer list above by taking the projection
    whole, the way `queue._emit_cancel_task_done` already does. That is what
    retires the defect class here: a field added upstream tomorrow reaches the
    card without an edit, so there is no mirror left to forget to update."""
    from types import SimpleNamespace

    from supervisor import workers

    signature = inspect.signature(workers._emit_task_done_terminal)
    named_cost_fields = sorted(
        name for name in signature.parameters
        if name in _real_projection(tmp_path)
    )
    assert named_cost_fields == [], (
        "the emitter re-declares projection fields by name again: "
        f"{named_cost_fields} -- that is the mirror that drifted three times"
    )

    events = []
    _orig_get_event_q = workers.get_event_q
    workers.get_event_q = lambda: SimpleNamespace(put=events.append)
    try:
        future = dict(_real_projection(tmp_path), a_field_invented_after_this_commit=7)
        assert workers._emit_task_done_terminal(None, "t1", "failed", cost_fields=future) is True
        assert events[-1]["a_field_invented_after_this_commit"] == 7
        assert events[-1]["non_final_rows"] == 0
        # The projection carries `ledger_integrity_degraded: False` on a healthy
        # task; forwarding it would put a "degraded" key on every clean terminal
        # event. The flag is a disclosure, so it appears only when it discloses.
        assert "ledger_integrity_degraded" not in events[-1]
    finally:
        # Direct assignment (not monkeypatch) polluted every later test in the
        # process — the manager-backed event-bus tests received this namespace.
        workers.get_event_q = _orig_get_event_q


def test_terminal_emitter_still_withholds_an_unavailable_projection(tmp_path, monkeypatch):
    """The opaque hand-off must not turn `cost_accounting_status: unavailable`
    into published numbers: the projection's `None` placeholders are absence,
    and a card that shows them as $0.00 is a false accounting claim (P1).

    The unavailable projection is produced by breaking the real ledger rather
    than hand-written here -- a fixture is one more copy of the field set, and
    copies of this field set are what the pass above went in to delete."""
    from types import SimpleNamespace

    import ouroboros.usage_accounting as usage_accounting
    from supervisor import workers
    from supervisor.state import reconstruct_task_cost

    def _ledger_down(*_a, **_k):
        raise RuntimeError("ledger down")

    monkeypatch.setattr(usage_accounting, "usage_breakdown", _ledger_down)
    unavailable = reconstruct_task_cost("t1", fields=True, drive_root=tmp_path)
    assert unavailable["cost_accounting_status"] == "unavailable"
    assert unavailable["cost_usd"] is None

    events = []
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=events.append))
    assert workers._emit_task_done_terminal(None, "t1", "failed", cost_fields=unavailable) is True
    emitted = events[-1]
    assert emitted["cost_accounting_status"] == "unavailable"
    assert emitted["cost_accounting_error"] == "ledger_unavailable"
    assert emitted["ledger_integrity_degraded"] is True
    assert [k for k, v in emitted.items() if v is None] == []
    assert "cost_usd" not in emitted and "non_final_rows" not in emitted


def _install_supervisor(tmp_path, monkeypatch):
    from supervisor import queue, state, workers

    state.init(tmp_path, total_budget_limit=10.0)
    queue.init(tmp_path)
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    workers.PENDING[:] = []
    workers.RUNNING.clear()
    workers.WORKERS.clear()
    queue.BUDGET_ROOT_FENCES.clear()
    queue.init_queue_refs(workers.PENDING, workers.RUNNING, workers.QUEUE_SEQ_COUNTER_REF)
    events = []
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=events.append))
    monkeypatch.setattr(workers, "send_with_budget", lambda *_a, **_k: None)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": None)
    return queue, state, workers, events


def test_budget_stop_terminalizes_instead_of_stranding_the_task(tmp_path, monkeypatch):
    """The silent one: the emitter call sits immediately before
    `terminal_ids.append`, inside a try/except that only logs. A raise there
    leaves the durable row `failed` while the task stays in PENDING forever --
    the card spins, every later tick re-plans it, and a budget top-up dispatches
    a task whose result already says failed."""
    from supervisor import state as state_mod

    _queue, _state, workers, events = _install_supervisor(tmp_path, monkeypatch)
    monkeypatch.setattr(workers, "load_state", lambda: {"owner_chat_id": 0})
    monkeypatch.setattr(state_mod, "budget_remaining", lambda _st, **_kw: 0.0)
    # A retry lineage is not replay-safe, which is what selects the terminal branch.
    workers.PENDING.append({"id": "stopped-task", "type": "task", "chat_id": 0, "_attempt": 2})

    workers.assign_tasks()

    assert workers.PENDING == [], "task stranded in PENDING: the terminal projection raised"
    terminal = [event for event in events if event.get("type") == "task_done"]
    assert len(terminal) == 1
    assert terminal[0]["task_id"] == "stopped-task"
    assert terminal[0]["reason_code"] == "budget_exhausted"
    assert "non_final_rows" in terminal[0]


class _DeadProc:
    exitcode = 0

    @staticmethod
    def is_alive():
        return False

    @staticmethod
    def join(timeout=None):
        del timeout


def _crashed_worker(monkeypatch, workers, task):
    workers.WORKERS[0] = SimpleNamespace(
        wid=0, busy_task_id=task["id"], proc=_DeadProc(), reaping=False,
    )
    workers.RUNNING[task["id"]] = {
        "task": task, "started_at": 1.0, "last_heartbeat_at": 1.0, "attempt": 1,
    }
    monkeypatch.setattr(workers, "QUEUE_MAX_RETRIES", 3)
    monkeypatch.setattr(workers, "_LAST_SPAWN_TIME", 0)
    monkeypatch.setattr(workers, "CRASH_TS", [])
    monkeypatch.setattr(workers, "respawn_worker", lambda _wid: None)
    monkeypatch.setattr("ouroboros.tools.services.archive_task_service_logs", lambda *_a, **_k: None)


def test_evolution_stop_cleanup_finishes_the_health_sweep(tmp_path, monkeypatch):
    """Uncaught: a raise propagates out of `_ensure_workers_healthy_locked`
    mid-sweep, so this tick's respawns and the crash-storm bookkeeping below it
    are lost and the cancelled task never gets its terminal event."""
    _queue, _state, workers, events = _install_supervisor(tmp_path, monkeypatch)
    task = {"id": "evo-task", "type": "evolution", "chat_id": 0, "_attempt": 1}
    _crashed_worker(monkeypatch, workers, task)
    monkeypatch.setattr(workers, "load_state", lambda: {"evolution_mode_enabled": False})

    workers.ensure_workers_healthy()

    terminal = [event for event in events if event.get("type") == "task_done"]
    assert len(terminal) == 1, f"health sweep aborted before emitting: {events}"
    assert terminal[0]["task_id"] == "evo-task"
    assert terminal[0]["status"] == "cancelled"
    assert "non_final_rows" in terminal[0]


def test_admission_blocked_crash_retry_finishes_the_health_sweep(tmp_path, monkeypatch):
    """Same uncaught propagation on the crash-retry admission fence."""
    queue, _state, workers, events = _install_supervisor(tmp_path, monkeypatch)
    task = {"id": "blocked-task", "type": "task", "chat_id": 0, "_attempt": 1}
    _crashed_worker(monkeypatch, workers, task)
    monkeypatch.setattr(workers, "load_state", lambda: {})
    monkeypatch.setattr(
        queue, "enqueue_task", lambda *_a, **_k: {"_admission_blocked": "root_budget_fence"},
    )

    workers.ensure_workers_healthy()

    terminal = [event for event in events if event.get("type") == "task_done"]
    assert len(terminal) == 1, f"health sweep aborted before emitting: {events}"
    assert terminal[0]["task_id"] == "blocked-task"
    assert terminal[0]["reason_code"] == "worker_crash_retry_admission_blocked"
    assert "non_final_rows" in terminal[0]
