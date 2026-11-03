"""Separate-work tool emission through admission and boot, without replay."""

from types import SimpleNamespace

import pytest

from ouroboros.task_results import load_task_result
from ouroboros.tools.control_routing import _start_task
from tests.test_swarm_host_admission import host  # noqa: F401

pytestmark = pytest.mark.serial


class Crash(BaseException):
    """Leave the durable prefix intact, bypassing ordinary exception recovery."""


def start_context(host):  # noqa: F811
    class Transport(list):
        def append(self, event):
            # The producer's real emit boundary must already have complete bytes.
            stored = load_task_result(host.root, event["task_id"], strict=True)
            assert stored["promotion_intent"] == event
            assert stored["promotion_admission"]["phase"] == "prepared"
            super().append(event)

    return SimpleNamespace(task_id="caller", task_metadata={}, task_contract={}, project_id="",
                           current_chat_id=1, drive_root=host.root, budget_drive_root=host.root,
                           pending_events=Transport(), event_queue=None)


@pytest.mark.parametrize("cut", ["emitted", "admitting", "admitted"])
def test_start_crash_retains_intent_and_boot_never_admits_it(host, monkeypatch, cut):  # noqa: F811
    from ouroboros import server_maintenance
    from ouroboros.startup_migrations import prepare_startup_state
    from ouroboros.tools.control_task_results import _get_task_result
    from supervisor.events_project_routing import _handle_promote_chat_to_task

    prepare_startup_state(host.root)
    ctx = start_context(host)
    reply = _start_task(ctx, "Separate work", title="Separate", workspace="none")
    [event] = ctx.pending_events
    tid = event["task_id"]
    assert "Accepted, waiting to start" in reply and not host.pending
    if cut == "admitting":
        def crash_before_enqueue(*args, **kwargs):
            assert load_task_result(host.root, tid)["promotion_admission"]["phase"] == "admitting"
            raise Crash

        with monkeypatch.context() as faults:
            faults.setattr(host.ctx, "enqueue_task", crash_before_enqueue)
            with pytest.raises(Crash):
                _handle_promote_chat_to_task(event, host.ctx)
    elif cut == "admitted":
        assert _handle_promote_chat_to_task(event, host.ctx)["status"] == "scheduled"
        assert [task["id"] for task in host.pending] == [tid]

    # A new host loses the transport and volatile admission reservations.
    from supervisor import queue, workers

    ctx.pending_events.clear()
    queue.ADMISSION_RESERVATIONS.clear()
    before = list(host.pending)
    for _ in range(2):
        report = server_maintenance._run_startup_task_recovery(
            host.root, workers.REPO_DIR, skip_live_data=False, prior_worker_pids=set())
        assert report["errors"] == [], report
        assert host.pending == before and not host.running
    stored = load_task_result(host.root, tid, strict=True)
    assert stored["promotion_intent"] == event
    admission = stored["promotion_admission"]
    if cut == "emitted":
        assert admission["status"] == "not_started" and admission["retry_available"] is True
        assert "explicit retry is available: start_task(" in _get_task_result(ctx, tid)
        # Explicit retry is a new request/id, admitted once through the real consumer.
        _start_task(ctx, "Separate work", title="Separate", workspace="none")
        [retry] = ctx.pending_events
        assert retry["task_id"] != tid
        assert _handle_promote_chat_to_task(retry, host.ctx)["status"] == "scheduled"
        assert [row["id"] for row in host.pending] == [retry["task_id"]]
        assert load_task_result(host.root, tid)["promotion_admission"]["status"] == "not_started"
    elif cut == "admitting":
        assert admission["status"] == "unconfirmed" and admission["retry_available"] is False
        assert "held for reconciliation" in _get_task_result(ctx, tid)
    else:
        assert admission["status"] == "scheduled" and stored["status"] == "scheduled"


def test_start_cannot_emit_until_intent_write_succeeds(host, monkeypatch):  # noqa: F811
    from ouroboros import promotion_intent

    ctx = start_context(host)

    def disk_full(*args, **kwargs):
        raise OSError("test intent disk full")

    with monkeypatch.context() as faults:
        faults.setattr(promotion_intent, "write_task_result", disk_full)
        result = _start_task(ctx, "Separate work", title="Separate", workspace="none")
    assert "intent_persist_failed" in result and not ctx.pending_events and not host.pending
    _start_task(ctx, "Separate work", title="Separate", workspace="none")
    assert len(ctx.pending_events) == 1
