"""The agent-facing schedule and cancel tools, and the durable rows they mint.

Split out of ``tests/test_task_status_flow.py`` by theme: the ``schedule_task`` contract
(live emission, the pending-events fallback, memory modes, the closed options mapping,
workspace inheritance) and the ``cancel_task`` intent, including the natural completion
that wins a late cancel.
"""

import json
import pathlib
from types import SimpleNamespace


class _FakeEventQueue:
    def __init__(self, fail=False, status_root=None):
        self.fail = fail
        self.status_root = status_root
        self.events = []

    def put_nowait(self, evt):
        if self.fail:
            raise RuntimeError("queue unavailable")
        if self.status_root is not None:
            path = pathlib.Path(self.status_root) / "task_results" / f"{evt['task_id']}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["status"] == "requested"
        self.events.append(dict(evt))


def test_schedule_task_live_emits_strict_contract_and_requested_status(tmp_path):
    from ouroboros.tools.control import _schedule_task
    from ouroboros.task_results import STATUS_REQUESTED

    event_queue = _FakeEventQueue(status_root=tmp_path)
    ctx = SimpleNamespace(
        task_depth=0,
        pending_events=[],
        event_queue=event_queue,
        drive_root=tmp_path,
        task_id="parent123",
        task_metadata={"root_task_id": "root123", "session_id": "sess123"},
        current_chat_id=777,
        is_direct_chat=False,
        is_workspace_mode=lambda: False,
    )

    result = _schedule_task(
        ctx,
        objective="Do the thing",
        expected_output="A concise handoff",
        role="architecture",
        context="Model focus A",
    )

    assert "Subagent request queued" in result
    assert ctx.pending_events == []
    assert len(event_queue.events) == 1
    evt = event_queue.events[0]
    task_id = evt["task_id"]
    assert evt["description"] == "Do the thing"
    assert evt["expected_output"] == "A concise handoff"
    assert evt["role"] == "architecture"
    assert evt["parent_task_id"] == "parent123"
    assert evt["root_task_id"] == "root123"
    assert evt["session_id"] == "sess123"
    assert evt["chat_id"] == 777
    assert evt["delegation_role"] == "subagent"
    assert evt["memory_mode"] == "forked"
    assert pathlib.Path(evt["drive_root"]).parts[-3:] == ("headless_tasks", task_id, "data")
    assert evt["child_drive_root"] == evt["drive_root"]
    assert evt["budget_drive_root"] == str(tmp_path)
    assert evt["task_constraint"]["mode"] == "local_readonly_subagent"
    path = tmp_path / "task_results" / f"{task_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["status"] == STATUS_REQUESTED
    assert data["description"] == "Do the thing"
    assert data["expected_output"] == "A concise handoff"
    assert data["role"] == "architecture"
    assert data["context"] == "Model focus A"
    assert data["chat_id"] == 777
    assert data["memory_mode"] == "forked"
    assert data["child_drive_root"] == evt["drive_root"]


def test_schedule_task_falls_back_to_pending_events_when_live_queue_unavailable(tmp_path, monkeypatch):
    from ouroboros.tools import control_scheduling as control_mod
    from ouroboros.tools.control import _schedule_task

    ctx = SimpleNamespace(
        task_depth=0,
        pending_events=[],
        event_queue=_FakeEventQueue(fail=True),
        drive_root=tmp_path,
        task_id="parent123",
        task_metadata={},
        is_direct_chat=False,
        is_workspace_mode=lambda: False,
    )

    result = _schedule_task(ctx, objective="Fallback child", expected_output="Result")

    assert "Subagent request queued" in result
    assert len(ctx.pending_events) == 1
    assert ctx.pending_events[0]["objective"] == "Fallback child"

    event_queue = _FakeEventQueue()
    ctx.pending_events = []
    ctx.event_queue = event_queue
    monkeypatch.setattr(control_mod, "write_task_result", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("disk full")))
    result = _schedule_task(ctx, objective="No status", expected_output="No child")
    assert "SUBTASK_STATUS_ERROR" in result
    assert ctx.pending_events == []
    assert event_queue.events == []


def test_cancel_task_writes_durable_intent_and_emits_live(tmp_path):
    """Phase A: the cancel_task tool records a DURABLE intent, never a status."""
    from ouroboros.cancel_intents import active_intent
    from ouroboros.tools.join_ledger import _cancel_task
    from ouroboros.task_results import (
        STATUS_RUNNING, load_task_result, write_task_result,
    )
    from ouroboros.task_status import load_effective_task_result

    write_task_result(tmp_path, "child42", STATUS_RUNNING, result="working")
    event_queue = _FakeEventQueue()
    ctx = SimpleNamespace(
        task_depth=0, pending_events=[], event_queue=event_queue,
        drive_root=tmp_path, task_id="parent123", task_metadata={},
        is_direct_chat=False, is_workspace_mode=lambda: False,
    )

    result = _cancel_task(ctx, "child42", reason="not needed")

    assert "Cancel requested" in result
    # The canonical status is NOT touched — intent lives in the projection.
    assert load_task_result(tmp_path, "child42")["status"] == STATUS_RUNNING
    intent = active_intent(tmp_path, "child42")
    assert intent is not None and intent["state"] == "requested"
    assert intent["reason"] == "not needed"
    # The typed public projection rides every effective read.
    effective = load_effective_task_result(tmp_path, "child42")
    assert effective["status"] == STATUS_RUNNING
    assert effective["cancel_state"] == "pending"
    # And the cancel is emitted live (not buffered to round end).
    assert any(e.get("type") == "cancel_task" and e.get("task_id") == "child42" for e in event_queue.events)
    # Idempotent: a second request reuses the intent instead of re-minting.
    again = _cancel_task(ctx, "child42")
    assert "idempotent" in again
    assert active_intent(tmp_path, "child42")["request_id"] == intent["request_id"]


def test_natural_completion_wins_a_late_cancel(tmp_path, monkeypatch):
    """Phase A (owner 4=A): a child that finished before the teardown KEEPS its
    completed result and artifacts; the cancel settles as already_settled and
    the durable intent is closed — never the old completed-overwrite."""
    from ouroboros.cancel_intents import active_intent
    from ouroboros.outcomes import public_task_result
    from ouroboros.task_results import (
        STATUS_COMPLETED,
        load_task_result,
        write_task_result,
    )
    from ouroboros.tools.join_ledger import _cancel_task
    from supervisor import queue as queue_module
    from supervisor import workers
    from supervisor import task_lifecycle

    write_task_result(
        tmp_path,
        "fast-child",
        STATUS_COMPLETED,
        parent_task_id="parent123",
        root_task_id="parent123",
        delegation_role="subagent",
        result="finished in the cancellation race",
        final_answer="kept answer",
        trace_summary="kept trace",
        artifacts=[{"name": "kept.txt"}],
        artifact_bundle={"status": "ready"},
        outcome_axes={
            "execution": {"status": "ok"},
            "artifacts": {"status": "complete"},
            "objective": {"status": "solved"},
            "review": {"status": "pass"},
        },
        cost_usd=0.75,
    )
    event_queue = _FakeEventQueue()
    ctx = SimpleNamespace(
        task_depth=0,
        pending_events=[],
        event_queue=event_queue,
        drive_root=tmp_path,
        task_id="parent123",
        task_metadata={"root_task_id": "parent123"},
        is_direct_chat=False,
        is_workspace_mode=lambda: False,
    )

    # GR7-1a: "Nothing to cancel" needs a FRESH snapshot that positively
    # proves no live ownership — a missing snapshot fails OPEN and mints.
    from ouroboros.utils import atomic_write_json, utc_now_iso

    atomic_write_json(
        tmp_path / "state" / "queue_snapshot.json",
        {"ts": utc_now_iso(), "running": [], "pending": []},
    )
    # The child had ALREADY finished, so the tool mints no intent at all: an
    # intent on a settled task would show a "Cancelling…" badge on a finished
    # card until the watchdog cleaned it up, and there is nothing to tear down.
    assert "Nothing to cancel" in _cancel_task(ctx, "fast-child")
    assert active_intent(tmp_path, "fast-child") is None
    monkeypatch.setattr(queue_module, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue_module, "PENDING", [])
    monkeypatch.setattr(queue_module, "RUNNING", {})
    monkeypatch.setattr(workers, "WORKERS", {}, raising=False)
    monkeypatch.setattr(queue_module, "persist_queue_snapshot", lambda reason="": None)
    monkeypatch.setattr(queue_module, "_emit_cancel_task_done", lambda *_args, **_kwargs: None)

    assert queue_module.cancel_task_by_id("fast-child") is True
    stored = load_task_result(tmp_path, "fast-child")
    assert stored["status"] == STATUS_COMPLETED
    assert stored["cost_usd"] == 0.75
    assert stored["result"] == "finished in the cancellation race"
    assert stored["final_answer"] == "kept answer"
    assert stored["artifacts"] == [{"name": "kept.txt"}]
    # Completion wins WITHOUT a parent_decision stamp: discarding a kept result
    # stays a separate explicit action (discard_child_result).
    assert "parent_decision" not in stored
    # The durable intent settled (already_settled) — nothing left pending.
    assert active_intent(tmp_path, "fast-child") is None
    public = public_task_result(stored)
    assert public["outcome_axes"]["execution"]["status"] == "ok"
    # The typed custody outcome (not the boolean facade) reports already_settled.
    assert task_lifecycle.cancel_task_custody("fast-child") == task_lifecycle.CANCEL_ALREADY_SETTLED


def test_cancel_workspace_task_records_terminal_artifact_state(tmp_path, monkeypatch):
    from supervisor import queue as queue_module
    from supervisor import workers
    from ouroboros.headless import ARTIFACT_STATUS_MISSING, ARTIFACT_STATUS_PENDING
    from ouroboros.task_results import (
        STATUS_CANCELLED,
        STATUS_SCHEDULED,
        load_task_result,
        write_task_result,
    )
    from ouroboros.task_status import load_effective_task_result, wait_for_effective_tasks

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task = {
        "id": "workspacecancel",
        "chat_id": 0,
        "workspace_root": str(workspace),
        "metadata": {"workspace_root": str(workspace)},
    }
    monkeypatch.setattr(queue_module, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue_module, "PENDING", [task])
    monkeypatch.setattr(queue_module, "RUNNING", {})
    monkeypatch.setattr(workers, "WORKERS", {}, raising=False)
    monkeypatch.setattr(queue_module, "persist_queue_snapshot", lambda reason="": None)
    write_task_result(
        tmp_path,
        "workspacecancel",
        STATUS_SCHEDULED,
        workspace_root=str(workspace),
        artifact_status=ARTIFACT_STATUS_PENDING,
        artifact_bundle={"schema_version": 1, "status": ARTIFACT_STATUS_PENDING, "artifacts": [], "errors": []},
        result="queued",
    )

    assert queue_module.cancel_task_by_id("workspacecancel") is True

    stored = load_task_result(tmp_path, "workspacecancel")
    assert stored["status"] == STATUS_CANCELLED
    assert stored["artifact_status"] == ARTIFACT_STATUS_MISSING
    assert stored["artifact_bundle"]["status"] == ARTIFACT_STATUS_MISSING
    assert stored["outcome_axes"]["artifacts"]["status"] == ARTIFACT_STATUS_MISSING
    effective = load_effective_task_result(tmp_path, "workspacecancel")
    waited = wait_for_effective_tasks(tmp_path, ["workspacecancel"], timeout_sec=0)
    assert effective["status"] == STATUS_CANCELLED
    assert effective["artifact_status"] == ARTIFACT_STATUS_MISSING
    assert effective["artifact_bundle"]["status"] == ARTIFACT_STATUS_MISSING
    assert waited["all_terminal"] is True


def test_effective_cancelled_workspace_with_stale_bundle_is_terminal(tmp_path):
    from ouroboros.headless import ARTIFACT_STATUS_MISSING, ARTIFACT_STATUS_PENDING
    from ouroboros.task_results import STATUS_CANCELLED, write_task_result
    from ouroboros.task_status import load_effective_task_result, wait_for_effective_tasks

    write_task_result(
        tmp_path,
        "workspacecancel2",
        STATUS_CANCELLED,
        workspace_root=str(tmp_path / "workspace"),
        artifact_bundle={"schema_version": 1, "status": ARTIFACT_STATUS_PENDING, "artifacts": [], "errors": []},
        result="cancelled before finalization",
    )

    effective = load_effective_task_result(tmp_path, "workspacecancel2")
    waited = wait_for_effective_tasks(tmp_path, ["workspacecancel2"], timeout_sec=0)

    assert effective["status"] == STATUS_CANCELLED
    assert effective["artifact_status"] == ARTIFACT_STATUS_MISSING
    assert effective["artifact_bundle"]["status"] == ARTIFACT_STATUS_MISSING
    assert waited["all_terminal"] is True


def test_schedule_task_memory_modes_prepare_declared_drive_shape(tmp_path):
    from ouroboros.tools.control import _schedule_task

    parent_memory = tmp_path / "memory"
    (parent_memory / "knowledge").mkdir(parents=True)
    (parent_memory / "identity.md").write_text("stable identity", encoding="utf-8")
    (parent_memory / "scratchpad.md").write_text("working scratch", encoding="utf-8")
    (parent_memory / "knowledge" / "pattern.md").write_text("stable pattern", encoding="utf-8")

    event_queue = _FakeEventQueue()
    ctx = SimpleNamespace(
        task_depth=0,
        pending_events=[],
        event_queue=event_queue,
        drive_root=tmp_path,
        task_id="parent123",
        task_metadata={},
        is_direct_chat=False,
        is_workspace_mode=lambda: False,
    )

    _schedule_task(ctx, objective="Fork child", expected_output="Result", memory_mode="forked")
    forked_drive = tmp_path / "state" / "headless_tasks" / event_queue.events[-1]["task_id"] / "data"
    assert event_queue.events[-1]["drive_root"] == str(forked_drive)
    assert (forked_drive / "memory" / "identity.md").read_text(encoding="utf-8") == "stable identity"
    assert not (forked_drive / "memory" / "scratchpad.md").exists()
    assert (forked_drive / "memory" / "knowledge" / "pattern.md").is_file()

    _schedule_task(ctx, objective="Empty child", expected_output="Result", memory_mode="empty")
    empty_drive = tmp_path / "state" / "headless_tasks" / event_queue.events[-1]["task_id"] / "data"
    assert event_queue.events[-1]["drive_root"] == str(empty_drive)
    assert not (empty_drive / "memory" / "identity.md").exists()

    before_shared = len(event_queue.events)
    shared_result = _schedule_task(ctx, objective="Shared child", expected_output="Result", memory_mode="shared")
    assert "TOOL_ARG_ERROR" in shared_result
    assert "memory_mode=shared is disabled" in shared_result
    assert len(event_queue.events) == before_shared


def test_schedule_task_rejects_legacy_description_schema(tmp_path):
    from ouroboros.tools.control import _schedule_task

    ctx = SimpleNamespace(
        task_depth=0,
        pending_events=[],
        event_queue=None,
        drive_root=tmp_path,
        task_id="parent123",
        task_metadata={},
        is_direct_chat=False,
        is_workspace_mode=lambda: False,
    )

    result = _schedule_task(ctx, description="legacy", context="old", parent_task_id="p1")

    assert "TOOL_ARG_ERROR" in result
    assert "description" in result
    assert ctx.pending_events == []
    assert not (tmp_path / "task_results").exists()

    # `deadline_at` is a PUBLIC parameter as of v6.87.7 — the parent LLM is what knows when a
    # child's handoff stops being useful — so a model emitting it is accepted, not refused.
    from datetime import timedelta

    from ouroboros.deadline_utils import utc_now

    future = (utc_now() + timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ")
    accepted = _schedule_task(ctx, objective="o", expected_output="e", deadline_at=future)
    assert "TOOL_ARG_ERROR" not in accepted
    assert ctx.pending_events
    ctx.pending_events.clear()

    # An option the schema does not expose is still refused with the strict v6 message.
    unknown_as_kwarg = _schedule_task(ctx, objective="o", expected_output="e", nonesuch="x")
    assert "TOOL_ARG_ERROR" in unknown_as_kwarg and "nonesuch" in unknown_as_kwarg
    assert ctx.pending_events == []


def test_schedule_task_internal_options_mapping_is_closed(tmp_path):
    """The private seam is closed: a typo in an internal option must fail loudly rather than be
    silently ignored (the failure mode a free-form mapping invites)."""
    import pytest

    from ouroboros.tools.control import _schedule_task

    ctx = SimpleNamespace(
        task_depth=0, pending_events=[], event_queue=None, drive_root=tmp_path,
        task_id="parent123", task_metadata={}, is_direct_chat=False,
        is_workspace_mode=lambda: False,
    )
    with pytest.raises(TypeError, match="deadline_ats"):
        _schedule_task(ctx, {"deadline_ats": "typo"}, objective="o", expected_output="e")


def test_schedule_task_workspace_mode_inherits_context_and_enqueues(tmp_path):
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.tools.control import _get_task_result, _schedule_task, _wait_for_task

    budget_root = tmp_path / "root-data"
    ctx = SimpleNamespace(
        task_depth=0,
        pending_events=[],
        event_queue=_FakeEventQueue(),
        drive_root=tmp_path,
        task_id="parent123",
        task_metadata={"budget_drive_root": str(budget_root)},
        is_direct_chat=False,
        is_workspace_mode=lambda: True,
        workspace_root=tmp_path / "workspace",
        workspace_mode="external",
    )

    result = _schedule_task(ctx, objective="Inspect workspace", expected_output="Findings")

    assert "Subagent request queued" in result
    assert ctx.pending_events == []
    assert len(ctx.event_queue.events) == 1
    evt = ctx.event_queue.events[0]
    task_id = evt["task_id"]
    assert evt["workspace_root"] == str(tmp_path / "workspace")
    assert evt["budget_drive_root"] == str(budget_root)
    assert str(evt["child_drive_root"]).startswith(str(budget_root))
    assert not (tmp_path / "task_results" / f"{task_id}.json").exists()
    data = json.loads((budget_root / "task_results" / f"{task_id}.json").read_text(encoding="utf-8"))
    assert data["budget_drive_root"] == str(budget_root)
    assert data["child_drive_root"] == evt["child_drive_root"]

    write_task_result(budget_root, task_id, STATUS_COMPLETED, result="child handoff")
    assert "child handoff" in _get_task_result(ctx, task_id)
    assert "child handoff" in _wait_for_task(ctx, task_id, timeout_sec=0)
