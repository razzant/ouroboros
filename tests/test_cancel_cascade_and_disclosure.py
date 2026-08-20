"""The cancel tool, the cascade it mints, and what the cancelled result discloses.

Split out of ``tests/test_cancel_intents_phase_a.py`` by theme: the tool answers for a
settled task and for an id that was never scheduled, the child promotion before
cancelling, the refusal to fabricate a completed row, the cascade scope and its replay,
the teardown paths that refuse when the intent write fails, and the open delegated runs
and scoped homes disclosed on the cancelled result.
"""

from __future__ import annotations

import json
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
)
from tests._cancel_intents_shared import (  # noqa: F401  (autouse fixture applies on import)
    _reap_spawned_live_procs,
)
from tests._cancel_intents_shared import qenv as _qenv

# The fixture is requested by name as a test parameter, so it is re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
qenv = _qenv


def test_cancel_tool_reports_a_settled_task_instead_of_requesting(tmp_path, monkeypatch):
    """A-F8 at the ingress the agent actually calls.

    GR7-1a: "Nothing to cancel" now requires a FRESH queue snapshot that
    positively proves no live ownership — a missing/stale snapshot fails OPEN
    and mints (see test_gate_round7_fixes)."""
    from ouroboros.tools.join_ledger import _cancel_task
    from ouroboros.utils import utc_now_iso

    write_task_result(tmp_path, "settled-child", STATUS_COMPLETED, result="done")
    snap = tmp_path / "state" / "queue_snapshot.json"
    snap.parent.mkdir(parents=True, exist_ok=True)
    snap.write_text(
        json.dumps({"ts": utc_now_iso(), "running": [], "pending": []}),
        encoding="utf-8",
    )
    ctx = types.SimpleNamespace(
        task_depth=0, pending_events=[], event_queue=_CaptureQueue(),
        drive_root=tmp_path, task_id="parent1",
        task_metadata={"root_task_id": "parent1"},
        is_direct_chat=False, is_workspace_mode=lambda: False,
    )
    monkeypatch.setattr("ouroboros.tools.control._emit_control_event", lambda *_a, **_k: "live")
    reply = _cancel_task(ctx, "settled-child")
    assert "Nothing to cancel" in reply and STATUS_COMPLETED in reply
    assert ci.active_intent(tmp_path, "settled-child") is None


def test_cancel_of_a_never_scheduled_id_is_not_found_not_a_fabricated_row(qenv):
    """A-F22: no phantom cancelled task with a fabricated $0."""
    ci.request_cancel(qenv.drive, "ghost-typo", reason="mistyped id")
    assert qenv.tl.cancel_task_custody("ghost-typo") == qenv.tl.CANCEL_NOT_FOUND
    assert load_task_result(qenv.drive, "ghost-typo") in (None, {})
    assert ci.active_intent(qenv.drive, "ghost-typo") is None
    trail = (qenv.drive / "logs" / "supervisor.jsonl").read_text(encoding="utf-8")
    assert '"outcome": "not_found"' in trail


def test_finalize_on_miss_promotes_a_child_result_before_cancelling(qenv):
    """A-F23: a crash mid-custody must not bury a completed child result."""
    task_id = "miss-with-child"
    child_drive = qenv.drive / "child-of-miss"
    write_task_result(child_drive, task_id, STATUS_COMPLETED,
                      result="the child's finished answer", final_answer="answer")
    write_task_result(qenv.drive, task_id, STATUS_RUNNING, result="mirror",
                      child_drive_root=str(child_drive), delegation_role="subagent")
    ci.request_cancel(qenv.drive, task_id, reason="late cancel")

    assert qenv.tl.cancel_task_custody(task_id) == qenv.tl.CANCEL_ALREADY_SETTLED
    stored = load_task_result(qenv.drive, task_id)
    assert stored["status"] == STATUS_COMPLETED
    assert stored["result"] == "the child's finished answer"


@pytest.mark.serial
def test_cancel_of_a_task_with_no_durable_result_never_fabricates_completed(qenv, monkeypatch):
    """A-F5, PROVEN class: killed inside the spawn→RUNNING-write window.

    The artifact capture used to default-stamp ``completed`` on a workspace task
    with no durable row, after which the monotonic guard defended the invented
    completion against the real ``cancelled`` write — and it was published AND
    delivered to the owner."""
    task_id = "no-result-yet"
    workspace = qenv.drive / "ws"
    workspace.mkdir()
    child_drive = qenv.drive / "state" / "headless_tasks" / task_id / "data"
    child_drive.mkdir(parents=True)
    task = {
        "id": task_id, "chat_id": 4, "workspace_root": str(workspace),
        "child_drive_root": str(child_drive),
    }
    proc = _LiveProc()
    qenv.workers.WORKERS[0] = types.SimpleNamespace(
        wid=0, proc=proc, busy_task_id=task_id, reaping=False,
    )
    qenv.q.RUNNING[task_id] = {"task": task, "worker_id": 0}
    assert load_task_result(qenv.drive, task_id) in (None, {}), "no durable row yet"
    monkeypatch.setattr(qenv.q, "_emit_cancel_task_done", lambda *_a, **_kw: None)
    ci.request_cancel(qenv.drive, task_id, reason="kill it")

    try:
        outcome = qenv.tl.cancel_task_custody(task_id)
    finally:
        proc.terminate()

    assert outcome == qenv.tl.CANCEL_CANCELLED
    stored = load_task_result(qenv.drive, task_id)
    assert stored["status"] == STATUS_CANCELLED, "never a fabricated completed"
    # AR2-9 (§8-A4: провал capture = failed, не missing): the capture was OWED —
    # a RUNNING workspace task was killed — and could not run; that is a capture
    # FAILURE, never an honest "nothing was ever due".
    assert stored["artifact_status"] == "failed"
    assert "owed" in str(stored.get("artifact_error") or "")


def test_cascade_over_a_settled_root_with_live_children_still_delivers(qenv, monkeypatch):
    """A-F6, the incident's exact ending: root dead on budget, children live,
    ZERO chat messages. The routing chat comes from a live descendant."""
    delivered: list = []
    monkeypatch.setattr(
        "supervisor.terminal_delivery.deliver_unreviewed_salvage",
        lambda drive, task, tid, **kw: delivered.append({"task": task, "task_id": tid, **kw}),
    )
    monkeypatch.setattr(qenv.q, "_emit_cancel_task_done", lambda *_a, **_kw: None)
    # Root already settled (budget hard stop) and gone from both live maps.
    write_task_result(qenv.drive, "root-dead", "failed", reason_code="budget_exhausted",
                      result="root died on budget")
    qenv.q.PENDING[:] = [
        {"id": "kid1", "chat_id": 77, "parent_task_id": "root-dead", "root_task_id": "root-dead"},
    ]
    write_task_result(qenv.drive, "kid1", "scheduled")

    assert qenv.tl.cancel_task_by_id("root-dead", cascade=True) is True

    assert delivered, "a settled root with live children must still report to chat"
    (row,) = delivered
    from supervisor.terminal_delivery import lineage_chat_id
    assert lineage_chat_id(qenv.drive, row["task"], row["task_id"]) == 77
    # A-F21: the root's REAL status, never "cancelled" over a failed root.
    assert "failed" in row["outcome"]


def test_cascade_mints_child_intents_and_records_scope(qenv, monkeypatch):
    """A-F9: a crash mid-cascade leaves every live descendant fenced, and the
    root intent replays as a CASCADE."""
    monkeypatch.setattr(qenv.q, "cancel_task_custody",
                        lambda tid, **_kw: qenv.q.CANCEL_FAILED)
    qenv.q.PENDING[:] = [
        {"id": "c-root", "chat_id": 2},
        {"id": "c-kid", "chat_id": 2, "parent_task_id": "c-root", "root_task_id": "c-root"},
    ]
    ci.request_cancel(qenv.drive, "c-root", reason="stop the tree")

    assert qenv.tl.cancel_task_by_id("c-root", cascade=True) is False  # custody refused

    intents = ci.active_intents(qenv.drive)
    assert "c-kid" in intents, "every captured descendant carries its own intent"
    assert intents["c-kid"]["requested_by"] == "c-root"
    assert intents["c-root"]["scope"] == ci.SCOPE_CASCADE


def test_watchdog_replays_a_cascade_intent_as_a_cascade(qenv, monkeypatch):
    """A-F9: replaying a cascade as a single cancel would leave descendants live."""
    calls: list = []
    monkeypatch.setattr(qenv.tl, "cancel_task_by_id",
                        lambda tid, **kw: calls.append((tid, kw)) or True)
    monkeypatch.setattr(qenv.tl, "cancel_task_custody",
                        lambda tid, **kw: calls.append((tid, "single")) or "cancelled")
    ci.request_cancel(qenv.drive, "casc-root", scope=ci.SCOPE_CASCADE)
    store = qenv.drive / "state" / "cancel_intents.json"
    data = json.loads(store.read_text(encoding="utf-8"))
    from datetime import datetime, timezone
    data["intents"]["casc-root"]["requested_at"] = datetime.fromtimestamp(
        1_000_000 - 600, tz=timezone.utc,
    ).isoformat()
    store.write_text(json.dumps(data), encoding="utf-8")

    qenv.tl.sweep_cancel_intents(now=1_000_000.0)

    assert calls == [("casc-root", {"cascade": True})]


@pytest.mark.serial
def test_unreconciled_delegated_runs_are_disclosed_on_the_cancelled_result(
    qenv, monkeypatch,
):
    """A-F12: 'cancelled + salvage' while a workspace_write run may still mutate."""
    task_id = "delegating"
    task, child_drive, proc = _live_split_drive_task(qenv, task_id)
    write_task_result(qenv.drive, task_id, STATUS_RUNNING, result="working")
    ci.request_cancel(qenv.drive, task_id)
    monkeypatch.setattr(qenv.q, "_emit_cancel_task_done", lambda *_a, **_kw: None)
    monkeypatch.setattr("ouroboros.delegate_custody.reconcile_task_runs",
                        lambda *_a, **_kw: [])
    monkeypatch.setattr(
        "ouroboros.delegate_custody.open_runs",
        lambda *_a, **_kw: [types.SimpleNamespace(task_id=task_id, run_id="run-abc")],
    )
    notes: list = []
    monkeypatch.setattr(
        "supervisor.terminal_delivery.deliver_unreviewed_salvage",
        lambda *_a, **kw: notes.append(kw),
    )

    try:
        assert qenv.tl.cancel_task_custody(task_id) == qenv.tl.CANCEL_CANCELLED
    finally:
        proc.terminate()

    stored = load_task_result(qenv.drive, task_id)
    assert stored["delegated_runs_unreconciled"] == ["run-abc"]
    rows = [
        json.loads(line)
        for line in (qenv.drive / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [r for r in rows if r.get("type") == "delegated_runs_unreconciled"]


def test_nested_scoped_home_is_disclosed_even_with_an_os_boundary(tmp_path, monkeypatch):
    """A-F13: a nested home + recorded boundary was promoted to verified=true and
    its durable unconfined row was suppressed."""
    from ouroboros.gateways.claudexor import AttemptContainment
    from ouroboros.tools import delegate as dg

    operator_home = tmp_path / "home"
    nested = operator_home / ".claudexor" / "v3" / "scoped" / "a01"
    nested.mkdir(parents=True)
    attempts = [AttemptContainment(
        attempt_id="a01", home_isolated=True, home_dir=str(nested),
        boundary_mechanism="seatbelt",
    )]
    monkeypatch.setattr("ouroboros.gateways.claudexor.attempt_containment",
                        lambda run_dir: attempts)
    monkeypatch.setattr("ouroboros.gateways.claudexor.operator_home",
                        lambda: str(operator_home))
    detail = {"summary": {"runDir": str(tmp_path / "run")}}

    evidence = dg._containment_evidence(detail)

    assert evidence["nested_under_operator_home"] is True
    assert evidence["verified"] is False, "a nested home is not isolation"
    assert "not isolation from the operator's home" in evidence["note"]
    assert "seatbelt boundary WAS applied" in evidence["note"]

    # And the durable unconfined row is still emitted for that shape.
    emitted: list = []
    monkeypatch.setattr(dg, "_emit", lambda ctx, kind, payload: emitted.append((kind, payload)))
    dg._record_containment(None, None, {"containment": evidence, "state": "succeeded"})
    assert emitted and emitted[0][1]["nested_under_operator_home"] is True


def test_evolution_stop_refuses_teardown_when_the_intent_write_fails(qenv, monkeypatch):
    """AR2-1 (owner 1=A) + GR2-13: no cancel without a durable intent — the task
    is KEPT (pending rows stay queued, nothing is killed) and the failure is in
    the caller's typed view instead of vanishing behind a clean 'stopped'."""
    qenv.q.RUNNING["evo1"] = {"task": {"id": "evo1", "chat_id": 1, "type": "evolution"},
                              "worker_id": 0}
    qenv.q.PENDING[:] = [{"id": "evo-queued", "chat_id": 1, "type": "evolution"}]
    killed: list = []
    monkeypatch.setattr(qenv.q, "cancel_task_custody",
                        lambda tid, **_kw: killed.append(tid) or qenv.q.CANCEL_CANCELLED)
    monkeypatch.setattr(
        "ouroboros.cancel_intents.request_cancel",
        lambda *_a, **_kw: (_ for _ in ()).throw(OSError("intent store io")),
    )

    out = qenv.q.stop_evolution_tasks("owner stop")

    assert out["cancelled"] == []
    assert sorted(out["intent_write_failed"]) == ["evo-queued", "evo1"]
    assert killed == [], "no unfenced teardown"
    assert [t["id"] for t in qenv.q.PENDING] == ["evo-queued"], "the task is kept"
    lines, incomplete = qenv.q.evolution_stop_report(out)
    assert incomplete is True and any("INCOMPLETE" in line for line in lines)


def test_project_delete_refuses_teardown_when_the_intent_write_fails(qenv, monkeypatch):
    """AR2-1: the project-delete ingress fails CLOSED — the task stays live and
    the deletion fails visibly instead of tearing down without a durable fence."""
    from supervisor import queue_transitions as qt

    monkeypatch.setattr(
        qt, "_live_project_task_ids",
        lambda root, pid, roots_only=False, covering=None: ["p-task1"],
    )
    failed: list = []
    monkeypatch.setattr("ouroboros.projects_registry.fail_project_deletion",
                        lambda root, pid, err: failed.append((pid, err)))
    monkeypatch.setattr(
        "ouroboros.projects_registry.complete_project_deletion",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("must not complete")),
    )
    killed: list = []
    monkeypatch.setattr(qenv.q, "cancel_task_by_id",
                        lambda tid, **_kw: killed.append(tid) or True)
    monkeypatch.setattr(
        "ouroboros.cancel_intents.request_cancel",
        lambda *_a, **_kw: (_ for _ in ()).throw(OSError("intent store io")),
    )

    qt.run_project_deletion(qenv.drive, "proj1", 1)

    assert killed == [], "no unfenced teardown"
    assert failed and "cancel_intent_write_failed" in failed[0][1]


def test_cascade_descendant_intent_failure_is_surfaced_not_silent(qenv, monkeypatch):
    """AR2-1: a child whose per-descendant intent write fails is still cancelled
    THIS sweep, and the failure is a typed forensic row — never a debug line."""
    calls: list = []

    def _mock_custody(tid, **_kw):
        calls.append(tid)
        qenv.q.PENDING[:] = [t for t in qenv.q.PENDING if str(t.get("id")) != tid]
        write_task_result(qenv.drive, tid, STATUS_CANCELLED, result="cancelled")
        ci.settle_intent(qenv.drive, tid, outcome="cancelled")
        return qenv.q.CANCEL_CANCELLED

    monkeypatch.setattr(qenv.q, "cancel_task_custody", _mock_custody)
    monkeypatch.setattr("supervisor.terminal_delivery.deliver_cascade_summary",
                        lambda *_a, **_kw: None)
    real_request = ci.request_cancel

    def _flaky(root, tid, **kw):
        if tid == "d-kid":
            raise OSError("intent store io")
        return real_request(root, tid, **kw)

    monkeypatch.setattr("ouroboros.cancel_intents.request_cancel", _flaky)
    qenv.q.PENDING[:] = [
        {"id": "d-root", "chat_id": 2},
        {"id": "d-kid", "chat_id": 2, "parent_task_id": "d-root", "root_task_id": "d-root"},
    ]
    real_request(qenv.drive, "d-root", reason="stop the tree")

    assert qenv.tl.cancel_task_by_id("d-root", cascade=True) is True
    assert "d-kid" in calls, "custody still runs on the child this sweep"
    trail = (qenv.drive / "logs" / "supervisor.jsonl").read_text(encoding="utf-8")
    assert "cascade_descendant_intent_write_failed" in trail


def test_cascade_over_settled_root_keeps_the_intent_until_the_postcondition(qenv, monkeypatch):
    """GR2-1b/1e: a settled root with a live child keeps its durable cascade
    intent through a failed sweep (the crash-mid-sweep shape) — per-task custody
    defers the settle while descendants remain — and the intent settles only
    when a later cascade's no-live postcondition passes."""
    delivered: list = []
    monkeypatch.setattr(
        "supervisor.terminal_delivery.deliver_unreviewed_salvage",
        lambda drive, task, tid, **kw: delivered.append(tid),
    )
    monkeypatch.setattr(qenv.q, "_emit_cancel_task_done", lambda *_a, **_kw: None)
    write_task_result(qenv.drive, "sr1", "failed", reason_code="budget_exhausted",
                      result="root died on budget")
    qenv.q.PENDING[:] = [
        {"id": "sr1-kid", "chat_id": 9, "parent_task_id": "sr1", "root_task_id": "sr1"},
    ]
    write_task_result(qenv.drive, "sr1-kid", "scheduled")
    ci.request_cancel(qenv.drive, "sr1", scope=ci.SCOPE_CASCADE, allow_settled_target=True)

    # Sweep 1: the child's custody FAILS (simulated crash / stubborn teardown).
    real_custody = qenv.tl.cancel_task_custody
    monkeypatch.setattr(
        qenv.q, "cancel_task_custody",
        lambda tid, **kw: qenv.q.CANCEL_FAILED if tid == "sr1-kid" else real_custody(tid, **kw),
    )
    assert qenv.tl.cancel_task_by_id("sr1", cascade=True) is False
    row = ci.active_intent(qenv.drive, "sr1")
    assert row is not None and row["scope"] == ci.SCOPE_CASCADE, (
        "the durable cascade intent must survive a failed sweep — it is the "
        "watchdog's replay trigger for the live descendants"
    )

    # The watchdog replay converges: custody works now, postcondition settles it.
    monkeypatch.setattr(qenv.q, "cancel_task_custody", real_custody)
    assert qenv.tl.cancel_task_by_id("sr1", cascade=True) is True
    assert ci.active_intent(qenv.drive, "sr1") is None
    assert load_task_result(qenv.drive, "sr1-kid")["status"] == STATUS_CANCELLED
    assert delivered, "the tree's summary still reaches chat"


def test_reconcile_discloses_open_runs_even_when_outcomes_are_nonempty(qenv, monkeypatch):
    """GR2-7: a non-empty reconcile outcome list proves an ATTEMPT, not a
    settlement — unreadable/requested/failed outcomes and raising transports
    must still surface every run the durable custody rows say is open."""
    monkeypatch.setattr(
        "ouroboros.delegate_custody.reconcile_task_runs",
        lambda *_a, **_kw: [{"outcome": "unreadable", "run_id": "run-open"}],
    )
    monkeypatch.setattr(
        "ouroboros.delegate_custody.open_runs",
        lambda *_a, **_kw: [types.SimpleNamespace(task_id="rt1", run_id="run-open")],
    )
    assert qenv.tl._reconcile_delegated_runs_on_kill(qenv.q, "rt1") == ["run-open"]
    rows = [
        json.loads(line)
        for line in (qenv.drive / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [r for r in rows if r.get("type") == "delegated_runs_unreconciled"]

    # A RAISING reconcile is audited the same way, never swallowed into [].
    monkeypatch.setattr(
        "ouroboros.delegate_custody.reconcile_task_runs",
        lambda *_a, **_kw: (_ for _ in ()).throw(ConnectionError("daemon gone")),
    )
    assert qenv.tl._reconcile_delegated_runs_on_kill(qenv.q, "rt1") == ["run-open"]
