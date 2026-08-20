"""The mandatory E2E class: a live split-drive worker through the REAL kill path.

Split out of ``tests/test_cancel_intents_phase_a.py`` by theme. These tests spawn real OS
processes and belong to the serial lane; the ``_LiveProc`` scaffolding and its autouse
reaper live in ``tests/_cancel_intents_shared.py`` and are imported here so a leaked
sleeper never outlives its test.
"""

from __future__ import annotations

import types

import pytest

from ouroboros import cancel_intents as ci
from ouroboros.task_results import (
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_RUNNING,
    load_task_result,
    write_task_result,
)

from tests._cancel_intents_shared import (
    _CaptureQueue,
    _LiveProc,
    _live_split_drive_task,
    _seed_llm_response,
)
from tests._cancel_intents_shared import (  # noqa: F401  (autouse fixture applies on import)
    _reap_spawned_live_procs,
)
from tests._cancel_intents_shared import qenv as _qenv

# The fixture is requested by name as a test parameter, so it is re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
qenv = _qenv


@pytest.mark.serial
def test_e2e_tool_cancel_kills_live_worker_and_settles_with_cost(qenv, monkeypatch):
    """tool cancel → durable intent → custody kills a LIVE worker process →
    post-kill copy runs → settled cancelled result with reconstructed cost and
    salvage → intent settled → typed task_done."""
    from ouroboros.tools.join_ledger import _cancel_task

    task_id = "e2e-cancel"
    task, child_drive, proc = _live_split_drive_task(qenv, task_id)
    write_task_result(qenv.drive, task_id, STATUS_RUNNING, result="working",
                      parent_task_id="parent-e2e", root_task_id="parent-e2e",
                      delegation_role="subagent")
    write_task_result(child_drive, task_id, STATUS_RUNNING, result="child mirror")
    _seed_llm_response(child_drive, task_id, "the partial answer so far")

    done_events: list = []
    monkeypatch.setattr(
        qenv.q, "_emit_cancel_task_done",
        lambda t, tid, cost_fields=None, status="cancelled": done_events.append(
            {"task_id": tid, "status": status, **(cost_fields or {})},
        ),
    )

    # Ingress through the TOOL (the one request_cancel seam).
    ctx = types.SimpleNamespace(
        task_depth=0, pending_events=[], event_queue=_CaptureQueue(),
        drive_root=qenv.drive, task_id="parent-e2e",
        task_metadata={"root_task_id": "parent-e2e"},
        is_direct_chat=False, is_workspace_mode=lambda: False,
    )
    assert "Cancel requested" in _cancel_task(ctx, task_id, reason="no longer needed")
    assert ci.active_intent(qenv.drive, task_id) is not None
    assert load_task_result(qenv.drive, task_id)["status"] == STATUS_RUNNING

    outcome = qenv.tl.cancel_task_custody(task_id)

    assert outcome == qenv.tl.CANCEL_CANCELLED
    assert not proc.is_alive(), "custody must actually kill the live worker"
    stored = load_task_result(qenv.drive, task_id)
    assert stored["status"] == STATUS_CANCELLED
    assert "the partial answer so far" in stored["result"]  # salvage in the result
    assert stored["parent_decision"] == "cancelled"          # stamped at OUTCOME
    assert stored.get("cost_accounting_status") == "available"  # reconstructed
    assert ci.active_intent(qenv.drive, task_id) is None
    assert not child_drive.exists(), "cancelled subagent drive is cleaned up"
    # task_done carries the reconstructed accounting — never a fabricated final $0
    # (an empty ledger reconstructs to a CONFIRMED zero, which is fine).
    (done,) = done_events
    assert done["status"] == STATUS_CANCELLED
    assert done["cost_accounting_status"] == "available"


@pytest.mark.serial
def test_e2e_child_finishing_before_the_kill_keeps_its_completed_result(qenv, monkeypatch):
    """The race the incident erased: the child wrote its COMPLETED result on the
    split child drive before the kill — custody copies it back, publishes it,
    and the completed payload + artifacts + cost survive."""
    task_id = "e2e-race"
    task, child_drive, proc = _live_split_drive_task(qenv, task_id)
    write_task_result(qenv.drive, task_id, STATUS_RUNNING, result="working",
                      parent_task_id="parent-e2e", root_task_id="parent-e2e",
                      delegation_role="subagent")
    write_task_result(
        child_drive, task_id, STATUS_COMPLETED,
        result="the finished child answer",
        final_answer="the finished child answer",
        trace_summary="did the work",
        cost_usd=0.42, cost_final=True, cost_accounting_status="available",
    )

    done_events: list = []
    monkeypatch.setattr(
        qenv.q, "_emit_cancel_task_done",
        lambda t, tid, cost_fields=None, status="cancelled": done_events.append(
            {"task_id": tid, "status": status, **(cost_fields or {})},
        ),
    )
    ci.request_cancel(qenv.drive, task_id, reason="late cancel", requested_by="parent-e2e")

    outcome = qenv.tl.cancel_task_custody(task_id)

    assert outcome == qenv.tl.CANCEL_ALREADY_SETTLED
    assert not proc.is_alive()
    stored = load_task_result(qenv.drive, task_id)
    assert stored["status"] == STATUS_COMPLETED
    assert stored["result"] == "the finished child answer"
    assert stored["final_answer"] == "the finished child answer"
    assert stored["cost_usd"] == 0.42
    # Completion wins WITHOUT a parent_decision overwrite of the kept result.
    assert "parent_decision" not in stored
    assert ci.active_intent(qenv.drive, task_id) is None
    (done,) = done_events
    assert done["status"] == STATUS_COMPLETED
    assert done["cost_usd"] == 0.42


@pytest.mark.serial
def test_kill_path_registers_the_owed_answer_before_the_intent_settles(qenv, monkeypatch):
    """GR2-4 crash order: the owner's terminal answer is durably OWED before the
    intent settles — a crash between the two replays instead of losing both the
    watchdog trigger and the answer."""
    from supervisor import terminal_delivery as td

    order: list = []
    real_register = td.register_pending_delivery
    monkeypatch.setattr(
        "supervisor.terminal_delivery.register_pending_delivery",
        lambda root, evt: order.append(("owed", str(evt.get("task_id") or ""))) or real_register(root, evt),
    )
    real_settle = ci.settle_intent
    monkeypatch.setattr(
        "ouroboros.cancel_intents.settle_intent",
        lambda root, tid, **kw: order.append(("settle", tid)) or real_settle(root, tid, **kw),
    )
    monkeypatch.setattr(qenv.q, "_emit_cancel_task_done", lambda *_a, **_kw: None)

    task_id = "owed-order"
    task = {"id": task_id, "chat_id": 9}
    proc = _LiveProc()
    qenv.workers.WORKERS[0] = types.SimpleNamespace(
        wid=0, proc=proc, busy_task_id=task_id, reaping=False,
    )
    qenv.q.RUNNING[task_id] = {"task": task, "worker_id": 0}
    write_task_result(qenv.drive, task_id, STATUS_RUNNING, result="working", chat_id=9)
    _seed_llm_response(qenv.drive, task_id, "the salvaged partial answer")
    ci.request_cancel(qenv.drive, task_id, reason="stop")

    try:
        assert qenv.tl.cancel_task_custody(task_id) == qenv.tl.CANCEL_CANCELLED
    finally:
        proc.terminate()

    owed_at = order.index(("owed", task_id))
    settle_at = order.index(("settle", task_id))
    assert owed_at < settle_at, f"owed must precede the settle: {order}"
    owed_rows = td.pending_deliveries(qenv.drive)
    assert any(row.get("task_id") == task_id for row in owed_rows), (
        "the durable outbox holds the answer until a send confirms"
    )
