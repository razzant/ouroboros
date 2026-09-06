"""The real kill path, end to end: a live split-drive worker, killed and settled.

Split out of ``tests/test_cancel_intents_phase_a.py`` by theme: the agent-tool cancel
kills a live worker and settles with honest cost, a child that finishes first keeps its
completed result, and the owed answer is registered before the intent settles.
"""

from __future__ import annotations
import json
import subprocess
import sys
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

from tests._cancel_intents_shared import _CaptureQueue, _LiveProc, _live_split_drive_task, _seed_llm_response
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
    # ABI-3 fix-round-2: persisted and relayed under the honest name only
    # (the child writer above used the legacy kwarg — honored, then stripped).
    assert stored["accounted_upper_bound_usd"] == 0.42
    assert "cost_usd" not in stored
    # Completion wins WITHOUT a parent_decision overwrite of the kept result.
    assert "parent_decision" not in stored
    assert ci.active_intent(qenv.drive, task_id) is None
    (done,) = done_events
    assert done["status"] == STATUS_COMPLETED
    assert done["accounted_upper_bound_usd"] == 0.42
    assert "cost_usd" not in done

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


@pytest.mark.serial
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="the os.kill(pid, 0) liveness probe is not non-mutating on Windows "
    "(TerminateProcess); the outcomes-level pin still covers the contract there",
)
def test_e2e_cancel_of_inflight_run_command_child_never_reads_as_tool_failure(
    qenv, monkeypatch, tmp_path,
):
    """T7 / Q2-2=A EXECUTED-path pin (triad round 1, G-B2): custody cancels a
    worker whose REAL in-flight run_command child (a live grandchild process)
    dies WITH the worker via kill_pid_tree, and the stored verdict's execution
    axis is ``cancelled`` — never a red ``tool_failure`` derived from the -9
    the child would have rendered. This executes the claim the outcomes-level
    pin (tests/test_observability_outcomes_v2.py) previously argued from code
    reading; while it stays green, the narrow active-cancel-intent classifier
    filter of Q2-2=A remains deliberately unimplemented.
    """
    import os
    import time

    from ouroboros.outcomes import normalize_outcome_axes

    task_id = "e2e-cancel-tree"
    pid_file = tmp_path / "child.pid"
    # A worker double whose REAL process spawns a REAL long-lived grandchild
    # (the in-flight run_command stand-in) and parks — a two-level tree.
    tree = _LiveProc.__new__(_LiveProc)
    tree._proc = subprocess.Popen([
        sys.executable, "-c",
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        "open(sys.argv[1], 'w').write(str(child.pid))\n"
        "time.sleep(120)",
        str(pid_file),
    ])
    tree.pid = tree._proc.pid
    _LiveProc._SPAWNED.append(tree._proc)
    deadline = time.time() + 10
    while not pid_file.exists() and time.time() < deadline:
        time.sleep(0.05)
    child_pid = int(pid_file.read_text())

    worker = types.SimpleNamespace(wid=0, proc=tree, busy_task_id=task_id, reaping=False)
    qenv.workers.WORKERS[0] = worker
    qenv.q.RUNNING[task_id] = {"task": {"id": task_id, "chat_id": 1}, "worker_id": 0}
    write_task_result(qenv.drive, task_id, STATUS_RUNNING, result="working")
    monkeypatch.setattr(
        qenv.q, "_emit_cancel_task_done",
        lambda *a, **kw: None,
    )
    ci.request_cancel(qenv.drive, task_id, reason="operator stop")

    try:
        outcome = qenv.tl.cancel_task_custody(task_id)

        assert outcome == qenv.tl.CANCEL_CANCELLED
        assert not tree.is_alive(), "custody must kill the live worker"
        child_deadline = time.time() + 5
        child_alive = True
        while time.time() < child_deadline:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                child_alive = False
                break
            time.sleep(0.1)
        assert not child_alive, "the in-flight run_command child must die with the worker"

        stored = load_task_result(qenv.drive, task_id)
        assert stored["status"] == STATUS_CANCELLED
        axes = normalize_outcome_axes(stored)
        assert axes["execution"]["status"] == "cancelled"
        assert "tool_failure" not in json.dumps(stored)
        assert ci.active_intent(qenv.drive, task_id) is None
    finally:
        for pid in (child_pid, tree.pid):
            try:
                os.kill(pid, 9)
            except (ProcessLookupError, PermissionError):
                pass
