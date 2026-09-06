import json
import pathlib
import threading
import time
from types import SimpleNamespace

_TEST_SUBAGENTS = '{"enabled":true,"items":[{"subagent_id":"api-scout","name":"API scout","recommended_use":"Tests","route":{"kind":"api_model","target_id":"openai/gpt-5.6-sol"},"effort":"high"}]}'


def _configure_test_subagent(monkeypatch):
    monkeypatch.setenv("OUROBOROS_SUBAGENTS", _TEST_SUBAGENTS)


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


def test_schedule_task_live_emits_strict_contract_and_requested_status(tmp_path, monkeypatch):
    from ouroboros.tools.control import _schedule_task
    from ouroboros.task_results import STATUS_REQUESTED

    _configure_test_subagent(monkeypatch)
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
        subagent_id="api-scout",
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

    _configure_test_subagent(monkeypatch)
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

    result = _schedule_task(ctx, subagent_id="api-scout", objective="Fallback child", expected_output="Result")

    assert "Subagent request queued" in result
    assert len(ctx.pending_events) == 1
    assert ctx.pending_events[0]["objective"] == "Fallback child"

    event_queue = _FakeEventQueue()
    ctx.pending_events = []
    ctx.event_queue = event_queue
    monkeypatch.setattr(control_mod, "write_task_result", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("disk full")))
    result = _schedule_task(ctx, subagent_id="api-scout", objective="No status", expected_output="No child")
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
    """A child that completed before teardown keeps its result and artifacts."""
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
        accounted_upper_bound_usd=0.75,
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

    from ouroboros.utils import atomic_write_json, utc_now_iso

    atomic_write_json(
        tmp_path / "state" / "queue_snapshot.json",
        {"ts": utc_now_iso(), "running": [], "pending": []},
    )
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
    assert stored["accounted_upper_bound_usd"] == 0.75
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


def test_schedule_task_memory_modes_prepare_declared_drive_shape(tmp_path, monkeypatch):
    from ouroboros.tools.control import _schedule_task

    _configure_test_subagent(monkeypatch)
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

    _schedule_task(ctx, subagent_id="api-scout", objective="Fork child", expected_output="Result", memory_mode="forked")
    forked_drive = tmp_path / "state" / "headless_tasks" / event_queue.events[-1]["task_id"] / "data"
    assert event_queue.events[-1]["drive_root"] == str(forked_drive)
    assert (forked_drive / "memory" / "identity.md").read_text(encoding="utf-8") == "stable identity"
    assert not (forked_drive / "memory" / "scratchpad.md").exists()
    assert (forked_drive / "memory" / "knowledge" / "pattern.md").is_file()

    _schedule_task(ctx, subagent_id="api-scout", objective="Empty child", expected_output="Result", memory_mode="empty")
    empty_drive = tmp_path / "state" / "headless_tasks" / event_queue.events[-1]["task_id"] / "data"
    assert event_queue.events[-1]["drive_root"] == str(empty_drive)
    assert not (empty_drive / "memory" / "identity.md").exists()

    before_shared = len(event_queue.events)
    shared_result = _schedule_task(ctx, subagent_id="api-scout", objective="Shared child", expected_output="Result", memory_mode="shared")
    assert "TOOL_ARG_ERROR" in shared_result
    assert "memory_mode=shared is disabled" in shared_result
    assert len(event_queue.events) == before_shared


def test_configured_session_child_materializes_initial_and_steered_attachments(tmp_path, monkeypatch):
    from ouroboros.artifacts import stage_task_attachments
    from ouroboros.owner_mailbox import write_owner_message
    from ouroboros.subagent_work_order import compile_external_work_order
    from ouroboros.tools.control import _schedule_task
    from ouroboros.tools.core import _read_file
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setenv(
        "OUROBOROS_SUBAGENTS",
        '{"enabled":true,"items":[{"subagent_id":"session-scout","name":"Session scout",'
        '"recommended_use":"Tests","route":{"kind":"agent_session",'
        '"target_id":"codex=gpt-5.6-sol"},"effort":"high"}]}',
    )
    source = tmp_path / "parent-input.txt"
    source.write_text("session-child-readable", encoding="utf-8")
    parent_manifest = stage_task_attachments(
        tmp_path, "parent-attachments", [{"path": str(source), "label": "parent input"}],
    )
    later = tmp_path / "steered-input.txt"
    later.write_text("steered-child-readable", encoding="utf-8")
    steered_manifest = stage_task_attachments(
        tmp_path, "parent-attachments", [{"path": str(later), "label": "steered input"}],
    )
    assert write_owner_message(
        tmp_path, "Use the later input", "parent-attachments", msg_id="steer-1",
        attachment_manifest=steered_manifest,
    )
    event_queue = _FakeEventQueue()
    ctx = SimpleNamespace(
        task_depth=0, pending_events=[], event_queue=event_queue,
        drive_root=tmp_path, task_id="parent-attachments",
        task_contract={"attachment_manifest": [dict(row) for row in parent_manifest]},
        task_metadata={}, is_direct_chat=False, is_workspace_mode=lambda: False,
    )

    result = _schedule_task(
        ctx, subagent_id="session-scout", objective="Inspect inherited input",
        expected_output="Report", memory_mode="forked",
    )

    assert "Subagent request queued" in result
    event = event_queue.events[0]
    child_manifest = event["task_contract"]["attachment_manifest"]
    assert [row["label"] for row in child_manifest] == ["parent input", "steered input"]
    assert str(parent_manifest[0]["abs_path"]) not in str(child_manifest)
    assert str(steered_manifest[0]["abs_path"]) not in str(child_manifest)
    child_ctx = ToolContext(
        repo_dir=tmp_path, drive_root=pathlib.Path(event["drive_root"]),
        task_id=event["task_id"],
    )
    assert "session-child-readable" in _read_file(
        child_ctx, child_manifest[0]["relpath"], root="artifact_store",
    )
    assert "steered-child-readable" in _read_file(
        child_ctx, child_manifest[1]["relpath"], root="artifact_store",
    )
    work_order = compile_external_work_order(event)
    # The canonical work-order renderer serializes the manifest with Python's
    # repr; assert the exact serialized value on both POSIX and Windows.
    assert all(repr(row["abs_path"]) in work_order for row in child_manifest)
    assert repr(parent_manifest[0]["abs_path"]) not in work_order
    assert repr(steered_manifest[0]["abs_path"]) not in work_order


def test_schedule_task_rejects_legacy_description_schema(tmp_path, monkeypatch):
    from ouroboros.tools.control import _schedule_task

    _configure_test_subagent(monkeypatch)
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

    from datetime import timedelta

    from ouroboros.deadline_utils import utc_now

    future = (utc_now() + timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ")
    accepted = _schedule_task(ctx, subagent_id="api-scout", objective="o", expected_output="e", deadline_at=future)
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


def test_schedule_task_workspace_mode_inherits_context_and_enqueues(tmp_path, monkeypatch):
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.tools.control import _get_task_result, _schedule_task, _wait_for_task

    _configure_test_subagent(monkeypatch)
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

    result = _schedule_task(ctx, subagent_id="api-scout", objective="Inspect workspace", expected_output="Findings")

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


def test_get_task_result_returns_full_completed_output(tmp_path):
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.tools.control import _get_task_result

    full_text = ("hello\n" * 1200) + "TAIL_MARKER"
    write_task_result(
        tmp_path,
        "abc123",
        STATUS_COMPLETED,
        result=full_text,
        accounted_upper_bound_usd=1.23,
        trace_summary="trace",
    )

    ctx = SimpleNamespace(drive_root=tmp_path)
    output = _get_task_result(ctx, "abc123")

    assert "TAIL_MARKER" in output
    assert full_text in output
    assert "[SUBTASK_OUTCOME]" in output
    assert '"outcome_axes"' in output
    assert "[BEGIN_SUBTASK_OUTPUT]" in output


def test_get_task_result_carries_bounded_per_receipt_rows(tmp_path):
    """W2: the FULL single-child handoff (get_task_result/wait_task) shows WHICH
    checks passed as bounded identity rows — OUTSTANDING first, then newest, hard
    cap 10, exact omitted count — while the wait_tasks batch projection stays
    counts-compact.

    The bound must not be able to bury the fact the parent's absorption decision
    turns on: a child that failed a check early and then produced ten greens for
    OTHER criteria used to hand up an affirmatively all-green list."""
    import json as _json

    from ouroboros.outcomes import append_verification_receipt
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.tools.control import _get_task_result

    write_task_result(tmp_path, "abc123", STATUS_COMPLETED, result="done", accounted_upper_bound_usd=0.1)
    for idx in range(12):
        append_verification_receipt(tmp_path, "abc123", {
            "status": "pass" if idx else "fail",
            "check": f"pytest tests/x{idx}.py",
            "criterion_id": f"claim_{idx}",
        })

    output = _get_task_result(SimpleNamespace(drive_root=tmp_path), "abc123")
    summary = _json.loads(
        output.split("[SUBTASK_OUTCOME]\n", 1)[1].split("\n[/SUBTASK_OUTCOME]", 1)[0]
    )

    rows = summary["verification_receipts"]
    assert len(rows) == 10                                   # hard cap
    assert summary["verification_receipts_omitted"] == 2     # disclosed, exact
    # The still-unreconciled RED is carried FIRST and says why, even though ten
    # newer greens exist — no green of another criterion clears it.
    assert rows[0]["criterion_id"] == "claim_0"
    assert rows[0]["status"] == "fail"
    assert rows[0]["outstanding"] == "unreconciled_failed"
    # ...the rest of the cap is the newest remaining receipts, and only the OLDEST
    # greens are the ones left out.
    assert [row["criterion_id"] for row in rows[1:]] == [
        f"claim_{idx}" for idx in range(11, 2, -1)
    ]
    assert all("outstanding" not in row for row in rows[1:])
    assert "check" in rows[0] and "reconciliation_identity" in rows[0]

    # A red that a LATER green for the same criterion reconciles is not carried:
    # the rule is the shared unreconciled-set SSOT, not "always float failures".
    write_task_result(tmp_path, "closed", STATUS_COMPLETED, result="done", accounted_upper_bound_usd=0.1)
    append_verification_receipt(tmp_path, "closed", {
        "status": "fail", "check": "pytest tests/a.py", "criterion_id": "claim_a",
    })
    for idx in range(11):
        append_verification_receipt(tmp_path, "closed", {
            "status": "pass", "check": "pytest tests/a.py", "criterion_id": "claim_a"
            if idx == 0 else f"claim_b{idx}",
        })
    closed = _json.loads(
        _get_task_result(SimpleNamespace(drive_root=tmp_path), "closed")
        .split("[SUBTASK_OUTCOME]\n", 1)[1].split("\n[/SUBTASK_OUTCOME]", 1)[0]
    )
    assert all("outstanding" not in row for row in closed["verification_receipts"])
    assert closed["verification_receipts"][0]["criterion_id"] == "claim_b10"
    # No receipts -> no rows key at all (the wave1 zero-receipt shape stays visible
    # through the ledger counts, not an empty list).
    write_task_result(tmp_path, "noreceipts", STATUS_COMPLETED, result="done")
    bare = _get_task_result(SimpleNamespace(drive_root=tmp_path), "noreceipts")
    assert "verification_receipts_omitted" not in bare


def _receipt_rows_of(output):
    import json as _json

    summary = _json.loads(
        output.split("[SUBTASK_OUTCOME]\n", 1)[1].split("\n[/SUBTASK_OUTCOME]", 1)[0]
    )
    return summary.get("verification_receipts")


def test_default_receipt_read_stays_all_or_nothing_on_invalid_utf8(tmp_path):
    from ouroboros.outcomes import (
        read_verification_receipts,
        verification_receipts_path,
    )

    path = verification_receipts_path(tmp_path, "invalid-utf8", create=True)
    path.write_bytes(b'\xff\n{"status":"pass","criterion_id":"partial-green"}\n')

    # Existing observational consumers must not receive a partial green view.
    assert read_verification_receipts(tmp_path, "invalid-utf8") == []

    gaps: set[str] = set()
    rows = read_verification_receipts(
        tmp_path, "invalid-utf8", gap_reasons=gaps,
    )
    assert [row["criterion_id"] for row in rows] == ["partial-green"]
    assert "unreadable_bytes" in gaps


def test_child_finalization_publishes_receipts_to_canonical_root(tmp_path):
    """S3 seam (a): every real schedule_subagent child runs memory_mode forked|empty
    on an ISOLATED drive, so verify_and_record writes its receipts under the CHILD
    drive while the parent-side W2 reader resolves them against the canonical root.
    Child finalization (headless.copy_child_task_result) must publish
    verification_receipts.jsonl to the canonical root alongside the artifact rebase
    — WITHOUT any parent read in between (the opportunistic effective-read artifact
    sync must not be the only carrier: it dies with the child drive, which the
    cancel path and the startup prune both delete)."""
    from ouroboros.headless import copy_child_task_result, prepare_task_drive
    from ouroboros.outcomes import append_verification_receipt, read_verification_receipts
    from ouroboros.task_results import STATUS_COMPLETED, STATUS_SCHEDULED, write_task_result
    from ouroboros.tools.control import _get_task_result

    tid = "childsplit"
    child_drive = prepare_task_drive(tmp_path, tid, "forked")
    assert child_drive == tmp_path / "state" / "headless_tasks" / tid / "data"

    # Parent-side scheduled record (the shape schedule_subagent writes); the child
    # self-finalizes and records receipts ONLY on its isolated drive.
    write_task_result(
        tmp_path, tid, STATUS_SCHEDULED,
        drive_root=str(child_drive), child_drive_root=str(child_drive),
    )
    write_task_result(child_drive, tid, STATUS_COMPLETED, result="child split done", accounted_upper_bound_usd=0.2)
    append_verification_receipt(child_drive, tid, {
        "status": "fail", "check": "pytest tests/red.py", "criterion_id": "claim_red",
    })
    append_verification_receipt(child_drive, tid, {
        "status": "pass", "check": "pytest tests/green.py", "criterion_id": "claim_green",
    })
    append_verification_receipt(tmp_path, tid, {
        "status": "declared", "contract_kind": "delegation_zero_run",
        "zero_run": True, "zero_run_decision": "complete",
    })

    # Finalization copy-back publishes the receipts file to the canonical root
    # (no parent-side read has happened yet — the publish alone must carry them).
    copied = copy_child_task_result(tmp_path, {"id": tid, "drive_root": str(child_drive)})
    assert copied is not None
    canonical = read_verification_receipts(tmp_path, tid)
    assert [r.get("criterion_id") for r in canonical] == [
        "claim_red", "claim_green", None,
    ]
    assert canonical[-1]["contract_kind"] == "delegation_zero_run"

    # Durability: the receipts survive child-drive pruning (retention GC / the
    # cancel path delete the drive; the canonical copy is the durable record).
    import shutil as _shutil

    _shutil.rmtree(child_drive)
    rows = _receipt_rows_of(_get_task_result(SimpleNamespace(drive_root=tmp_path), tid))
    assert rows is not None and len(rows) == 3
    assert rows[0]["criterion_id"] == "claim_red"
    assert rows[0]["outstanding"] == "unreconciled_failed"


def test_child_receipt_republish_is_idempotent_refresh(tmp_path):
    """S3 seam (a) re-entry: copy_child_task_result runs more than once per child
    (task_done + reaper/cancel re-checks). The publish refreshes the union of the
    append-only child and canonical stores — newer rows land, nothing duplicates."""
    from ouroboros.headless import copy_child_task_result, prepare_task_drive
    from ouroboros.outcomes import append_verification_receipt, read_verification_receipts
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result

    tid = "childagain"
    child_drive = prepare_task_drive(tmp_path, tid, "forked")
    write_task_result(child_drive, tid, STATUS_COMPLETED, result="done")
    append_verification_receipt(child_drive, tid, {
        "status": "fail", "check": "pytest tests/red.py", "criterion_id": "claim_red",
    })
    task = {"id": tid, "drive_root": str(child_drive)}
    copy_child_task_result(tmp_path, task)
    copy_child_task_result(tmp_path, task)  # re-entry: no duplication
    assert [r["criterion_id"] for r in read_verification_receipts(tmp_path, tid)] == ["claim_red"]

    append_verification_receipt(child_drive, tid, {
        "status": "pass", "check": "pytest tests/red.py", "criterion_id": "claim_red",
    })
    copy_child_task_result(tmp_path, task)
    assert [r["criterion_id"] for r in read_verification_receipts(tmp_path, tid)] == [
        "claim_red", "claim_red",
    ]


def test_materializing_child_read_cannot_overwrite_canonical_zero_run_receipt(tmp_path):
    """Generic artifact refresh must never own the receipt authority stream.

    A running parent's ``wait_tasks`` performs materializing reads.  Before this
    regression, the first read registered the child receipt file as an ordinary
    artifact; a later read reused that source/destination manifest mapping and
    copied the stale child bytes over a newer canonical-only zero-run row.
    """
    from ouroboros.headless import copy_child_task_result, prepare_task_drive
    from ouroboros.outcomes import append_verification_receipt, read_verification_receipts
    from ouroboros.task_results import (
        STATUS_COMPLETED,
        STATUS_SCHEDULED,
        load_task_result,
        write_task_result,
    )
    from ouroboros.task_status import effective_task_result

    tid = "child-materialized-receipts"
    child_drive = prepare_task_drive(tmp_path, tid, "empty")
    assert child_drive is not None
    write_task_result(
        tmp_path, tid, STATUS_SCHEDULED,
        drive_root=str(child_drive), child_drive_root=str(child_drive),
    )
    assert append_verification_receipt(child_drive, tid, {
        "status": "pass", "criterion_id": "child-check",
        "ts": "2026-01-01T00:00:01+00:00",
    })
    from ouroboros.outcomes import verification_receipts_path

    child_receipts = verification_receipts_path(child_drive, tid)
    write_task_result(
        child_drive,
        tid,
        STATUS_COMPLETED,
        result="done",
        # Historical/current pre-fix task rows may already carry the authority
        # file as an ordinary artifact even though collection now excludes it.
        artifacts=[{
            "kind": "task_artifact",
            "name": child_receipts.name,
            "path": str(child_receipts),
        }],
    )

    from ouroboros.artifacts import collect_task_artifact_records

    assert collect_task_artifact_records(child_drive, tid) == []

    # The first wait/detail read observes the child while its local receipt file
    # exists.  It must not register that authority stream as a generic artifact.
    effective_task_result(tmp_path, load_task_result(tmp_path, tid) or {})
    assert append_verification_receipt(tmp_path, tid, {
        "status": "declared",
        "contract_kind": "delegation_zero_run",
        "zero_run": True,
        "zero_run_decision": "complete",
        "zero_run_basis": "completed useful direct child",
        "physical_run_started": False,
        "ts": "2026-01-01T00:00:02+00:00",
    })

    # Repeated polling must preserve the canonical-only row.  Final copy-back
    # then unions the ordinary child check into that same authority file.
    effective_task_result(tmp_path, load_task_result(tmp_path, tid) or {})
    assert [
        row.get("contract_kind")
        for row in read_verification_receipts(tmp_path, tid)
    ] == [None, "delegation_zero_run"]

    copied = copy_child_task_result(
        tmp_path, {"id": tid, "drive_root": str(child_drive)},
    )
    assert copied is not None
    rows = read_verification_receipts(tmp_path, tid)
    assert [row.get("criterion_id") for row in rows] == ["child-check", None]
    assert rows[-1]["contract_kind"] == "delegation_zero_run"
    assert all(
        str(item.get("name") or "") != "verification_receipts.jsonl"
        for item in copied.get("artifacts") or []
    )


def test_child_receipt_publish_preserves_corrupt_canonical_authority(tmp_path):
    from ouroboros.headless import _publish_child_verification_receipts, prepare_task_drive
    from ouroboros.outcomes import append_verification_receipt, verification_receipts_path

    tid = "child-corrupt-authority"
    child_drive = prepare_task_drive(tmp_path, tid, "forked")
    assert append_verification_receipt(child_drive, tid, {
        "status": "pass",
        "check": "pytest tests/ordinary.py",
        "criterion_id": "ordinary",
    })
    canonical = verification_receipts_path(tmp_path, tid, create=True)
    corrupt = b'{"contract_kind":"delegation_zero_run","zero_run":true\n'
    canonical.write_bytes(corrupt)

    _publish_child_verification_receipts(tmp_path, tid, child_drive)

    assert canonical.read_bytes() == corrupt


def test_child_receipt_publish_serializes_late_canonical_append(tmp_path, monkeypatch):
    """An append arriving after union-read must survive the atomic refresh."""
    import ouroboros.outcome_receipt_store as receipt_store
    import ouroboros.utils as utils
    from ouroboros.outcomes import append_verification_receipt, read_verification_receipts

    tid = "child-late-canonical-append"
    child_drive = tmp_path / "child"
    assert append_verification_receipt(child_drive, tid, {
        "status": "pass", "criterion_id": "child-check",
    })
    assert append_verification_receipt(tmp_path, tid, {
        "status": "declared", "criterion_id": "canonical-before",
    })

    publisher_at_replace = threading.Event()
    allow_replace = threading.Event()
    real_write = receipt_store.write_text_atomic

    def paused_write(path, content, **kwargs):
        publisher_at_replace.set()
        assert allow_replace.wait(timeout=2.0)
        return real_write(path, content, **kwargs)

    monkeypatch.setattr(receipt_store, "write_text_atomic", paused_write)
    append_blocked_on_union = threading.Event()
    real_sleep = utils.time.sleep

    def observed_sleep(seconds):
        if (
            threading.current_thread().name == "late-receipt-append"
            and seconds == 0.01
        ):
            append_blocked_on_union.set()
        return real_sleep(seconds)

    monkeypatch.setattr(utils.time, "sleep", observed_sleep)
    publish_result = []
    publish_thread = threading.Thread(
        target=lambda: publish_result.append(
            receipt_store.publish_verification_receipt_union(
                tmp_path, tid, child_drive,
            )
        )
    )
    publish_thread.start()
    assert publisher_at_replace.wait(timeout=2.0)

    append_result = []
    append_thread = threading.Thread(
        name="late-receipt-append",
        target=lambda: append_result.append(
            append_verification_receipt(tmp_path, tid, {
                "status": "declared", "criterion_id": "canonical-late",
            })
        )
    )
    append_thread.start()
    assert append_blocked_on_union.wait(timeout=2.0)
    assert append_result == [], "late append must wait on the union writer's sidecar"

    allow_replace.set()
    publish_thread.join(timeout=2.0)
    append_thread.join(timeout=2.0)
    assert not publish_thread.is_alive()
    assert not append_thread.is_alive()
    assert publish_result == [True]
    assert append_result == [True]
    assert [
        row.get("criterion_id")
        for row in read_verification_receipts(tmp_path, tid)
    ] == ["child-check", "canonical-before", "canonical-late"]


def test_verification_receipt_append_refuses_unlocked_timeout(tmp_path, monkeypatch):
    import ouroboros.platform_layer as platform
    from ouroboros.outcome_receipt_store import (
        append_verification_receipt,
        verification_receipts_path,
    )

    monkeypatch.setattr(
        platform, "acquire_exclusive_file_lock", lambda *_args, **_kwargs: None,
    )

    assert append_verification_receipt(
        tmp_path, "locked-receipt", {"status": "pass"},
    ) is False
    assert not verification_receipts_path(
        tmp_path, "locked-receipt", create=False,
    ).exists()


def test_authority_lock_never_steals_from_live_owner_by_age(tmp_path):
    import os

    from ouroboros.platform_layer import (
        acquire_exclusive_file_lock,
        release_exclusive_file_lock,
    )
    from ouroboros.utils import jsonl_append_lock_path

    receipt_path = tmp_path / "verification_receipts.jsonl"
    lock_path = jsonl_append_lock_path(receipt_path)
    owner_fd = acquire_exclusive_file_lock(lock_path)
    assert owner_fd is not None
    old = time.time() - 60.0
    os.utime(lock_path, (old, old))
    contender_fd = acquire_exclusive_file_lock(
        lock_path, timeout_sec=0.05, stale_sec=0.01,
        poll_sec=0.005, owner_aware_stale=True,
    )
    try:
        assert contender_fd is None
        assert lock_path.exists()
    finally:
        release_exclusive_file_lock(lock_path, contender_fd)
        release_exclusive_file_lock(lock_path, owner_fd)


def test_get_task_result_merges_child_and_canonical_receipts(tmp_path):
    """S3 seam (b): before copy-back, a canonical actor lifecycle receipt and
    child-local ordinary checks are one effective evidence view rather than
    competing fallback stores."""
    from ouroboros.agent_startup_checks import task_result_authority_projection
    from ouroboros.headless import prepare_task_drive
    from ouroboros.outcomes import append_verification_receipt
    from ouroboros.task_results import STATUS_SCHEDULED, load_task_result, write_task_result
    from ouroboros.tools.control import _get_task_result

    tid = "childlive"
    child_drive = prepare_task_drive(tmp_path, tid, "forked")
    write_task_result(
        tmp_path, tid, STATUS_SCHEDULED,
        drive_root=str(child_drive), child_drive_root=str(child_drive),
    )
    # The child has recorded receipts but NO result yet (still running): nothing
    # exists canonically and the effective read has no child result to sync from.
    append_verification_receipt(child_drive, tid, {
        "status": "fail", "check": "pytest tests/red.py", "criterion_id": "claim_red",
    })
    append_verification_receipt(tmp_path, tid, {
        "status": "declared", "contract_kind": "delegation_zero_run",
        "zero_run": True, "zero_run_decision": "unknown",
    })

    rows = _receipt_rows_of(_get_task_result(SimpleNamespace(drive_root=tmp_path), tid))
    assert rows is not None and len(rows) == 2
    assert rows[0]["criterion_id"] == "claim_red"
    assert rows[0]["outstanding"] == "unreconciled_failed"
    assert rows[1]["status"] == "declared"

    authority = task_result_authority_projection(
        load_task_result(tmp_path, tid), drive_root=tmp_path,
    )
    assert [row.get("criterion_id") for row in authority["verification_receipts"]] == [
        "claim_red", None,
    ]


def test_get_task_result_uses_child_terminal_over_stale_parent(tmp_path):
    from ouroboros.task_results import STATUS_COMPLETED, STATUS_SCHEDULED, write_task_result
    from ouroboros.tools.control import _get_task_result

    child_drive = tmp_path / "state" / "headless_tasks" / "child123" / "data"
    child_drive.mkdir(parents=True)
    write_task_result(
        tmp_path,
        "child123",
        STATUS_SCHEDULED,
        child_drive_root=str(child_drive),
        result="stale parent handoff",
    )
    write_task_result(
        child_drive,
        "child123",
        STATUS_COMPLETED,
        result="child terminal handoff",
        accounted_upper_bound_usd=0.42,
        trace_summary="child trace",
    )

    ctx = SimpleNamespace(drive_root=tmp_path)
    output = _get_task_result(ctx, "child123")

    assert "child terminal handoff" in output
    assert "stale parent handoff" not in output
    assert "[SUBTASK_TRACE]" in output


def test_wait_for_tasks_returns_compact_structural_batch(tmp_path):
    from ouroboros.task_results import STATUS_COMPLETED, STATUS_SCHEDULED, write_task_result
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.tools.control import _wait_for_tasks
    from ouroboros.tools.join_ledger import _child_result_sha256

    child_drive = tmp_path / "state" / "headless_tasks" / "childdone" / "data"
    child_drive.mkdir(parents=True)
    write_task_result(
        tmp_path,
        "parentdone",
        STATUS_COMPLETED,
        result="parent finished",
        accounted_upper_bound_usd=1.25,
        loop_outcome={"result_status": "succeeded", "compat_result_status": "succeeded"},
        verification_ledger={"entries": [{"kind": "objective_outcome"}]},
        trace_refs=[{"path": "logs/trace.jsonl"}],
    )
    write_task_result(tmp_path, "childdone", STATUS_SCHEDULED, child_drive_root=str(child_drive), result="queued")
    write_task_result(child_drive, "childdone", STATUS_COMPLETED, result="child finished", trace_summary="trace")

    ctx = SimpleNamespace(drive_root=tmp_path)
    payload = json.loads(_wait_for_tasks(ctx, ["parentdone", "childdone"], timeout_sec=0))

    # Wait-envelope keys are preserved unchanged.
    assert payload["all_terminal"] is True
    assert payload["timed_out"] is False
    assert payload["mode"] == "all_terminal"
    assert "elapsed_sec" in payload and "timeout_sec" in payload
    # Disclosed omission: the note points at the full on-disk envelope.
    assert "get_task_result" in payload["tasks_note"]

    parent = payload["tasks"]["parentdone"]
    assert parent["task_id"] == "parentdone"
    assert parent["status"] == STATUS_COMPLETED
    assert parent["result"] == "parent finished"
    assert parent["accounted_upper_bound_usd"] == 1.25
    assert parent["outcome_axes"]["lifecycle"]["status"] == STATUS_COMPLETED
    # Forensics stay on disk — not inlined into the batch projection.
    assert "loop_outcome" not in parent
    assert "verification_ledger" not in parent
    assert "trace_refs" not in parent
    assert "duplicate_of" not in parent

    # child_result_sha256 reuses the join-ledger SSOT hash over the effective result.
    assert parent["child_result_sha256"] == _child_result_sha256(
        load_effective_task_result(tmp_path, "parentdone")
    )

    child = payload["tasks"]["childdone"]
    assert child["result"] == "child finished"
    assert child["trace_summary"] == "trace"
    assert child["accounted_upper_bound_usd"] is None  # absent accounting -> honest null, not $0
    assert child["child_result_sha256"] == _child_result_sha256(
        load_effective_task_result(tmp_path, "childdone")
    )


def test_wait_for_tasks_projects_execution_evidence_for_harness_children(tmp_path):
    # Q1A (2026-08-10 amendments): the batch projection is the surface a fan-out
    # parent absorbs its children through, and it used to hide whether a
    # harness-dispatched child ever actually delegated (the e9108a09 shape:
    # nine "harness" children, zero delegated runs, invisible in the batch).
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.tools.control import _wait_for_tasks

    write_task_result(
        tmp_path, "harnesskid", STATUS_COMPLETED, result="done",
        effective_executor="harness", executor_route="codex",
        actual_substrate="native_only",
        subagent_envelope={
            "actual_substrate": "native_only",
            "execution_evidence": {
                "delegated_runs_started": 0, "delegated_runs_settled": 0,
                "delegated_runs_succeeded": 0, "delegated_run_failure_states": [],
                "delegated_runs_source_unresolved": 0,
                "evidence_read_failed": False, "subscription_cost_usd": None,
                "subscription_cost_estimated": False, "harness_models": [],
            },
        },
    )
    write_task_result(tmp_path, "nativekid", STATUS_COMPLETED, result="done")

    ctx = SimpleNamespace(drive_root=tmp_path)
    payload = json.loads(_wait_for_tasks(ctx, ["harnesskid", "nativekid"], timeout_sec=0))

    assert payload["tasks"]["harnesskid"]["execution_evidence"] == {
        "delegated_runs_settled": 0,
        "delegated_runs_failed": 0,
        "native_contribution": "unknown",
        "dispatch_executor": "harness",
        "actual_substrate": "native_only",
        "delegated_runs_started": 0,
        "delegated_runs_succeeded": 0,
        "delegated_runs_source_unresolved": 0,
    }
    # A native child with no custody evidence stays compact — no evidence block.
    assert "execution_evidence" not in payload["tasks"]["nativekid"]


def test_wait_for_tasks_projection_marks_unreadable_evidence(tmp_path):
    # v6.94.0 landing-gate scope fix: unreadable custody evidence means the
    # counts are UNKNOWN — the projection carries ONLY dispatch_executor and
    # the typed evidence_read_failed marker. Emitting the raw zeros beside the
    # marker fabricated a "no runs" receipt for a log that was never read; the
    # substrate claim is likewise dropped even when the stored record carries
    # one (same omission rule subagents.envelope_from_task applies).
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.tools.control import _wait_for_tasks

    write_task_result(
        tmp_path, "blindkid", STATUS_COMPLETED, result="done",
        effective_executor="harness", executor_route="codex",
        actual_substrate="native_only",
        subagent_envelope={
            "actual_substrate": "native_only",
            "execution_evidence": {
                "delegated_runs_started": 0, "delegated_runs_succeeded": 0,
                "evidence_read_failed": True,
            },
        },
    )
    ctx = SimpleNamespace(drive_root=tmp_path)
    payload = json.loads(_wait_for_tasks(ctx, ["blindkid"], timeout_sec=0))
    assert payload["tasks"]["blindkid"]["execution_evidence"] == {
        "dispatch_executor": "harness",
        "evidence_read_failed": True,
    }


def test_wait_for_tasks_projection_omits_counts_without_envelope_evidence(tmp_path):
    # 6c03c24e corrective wave (LOW b): a stored harness child with NO envelope
    # evidence at all (pre-6.94 records) must not read as a zero-run receipt —
    # absence means "no evidence yet", so no counts and no substrate claim.
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.tools.control import _wait_for_tasks

    write_task_result(
        tmp_path, "oldkid", STATUS_COMPLETED, result="done",
        effective_executor="harness", executor_route="codex",
    )
    ctx = SimpleNamespace(drive_root=tmp_path)
    payload = json.loads(_wait_for_tasks(ctx, ["oldkid"], timeout_sec=0))
    assert payload["tasks"]["oldkid"]["execution_evidence"] == {
        "dispatch_executor": "harness",
    }


def test_wait_for_tasks_any_terminal_early_return_projects_pending_child(tmp_path):
    from ouroboros.task_results import STATUS_COMPLETED, STATUS_SCHEDULED, write_task_result
    from ouroboros.tools.control import _wait_for_tasks

    write_task_result(tmp_path, "fastchild", STATUS_COMPLETED, result="done first", accounted_upper_bound_usd=0.10)
    write_task_result(tmp_path, "slowchild", STATUS_SCHEDULED, result="")

    ctx = SimpleNamespace(drive_root=tmp_path)
    payload = json.loads(_wait_for_tasks(ctx, ["fastchild", "slowchild"], timeout_sec=0, mode="any_terminal"))

    assert payload["mode"] == "any_terminal"
    assert payload["all_terminal"] is False
    assert payload["timed_out"] is False
    assert payload["tasks"]["fastchild"]["status"] == STATUS_COMPLETED
    assert payload["tasks"]["fastchild"]["accounted_upper_bound_usd"] == 0.10
    # The still-pending child gets the same compact shape with cost present.
    assert payload["tasks"]["slowchild"]["status"] == STATUS_SCHEDULED
    assert "accounted_upper_bound_usd" in payload["tasks"]["slowchild"]
    assert "child_result_sha256" in payload["tasks"]["slowchild"]


def test_wait_for_tasks_cost_present_on_cancelled_and_failed(tmp_path):
    from ouroboros.task_results import STATUS_CANCELLED, STATUS_FAILED, write_task_result
    from ouroboros.tools.control import _wait_for_tasks

    write_task_result(tmp_path, "cancelledchild", STATUS_CANCELLED, result="best-effort partial handoff", accounted_upper_bound_usd=0.42)
    write_task_result(tmp_path, "failedchild", STATUS_FAILED, result="provider exploded")

    ctx = SimpleNamespace(drive_root=tmp_path)
    payload = json.loads(_wait_for_tasks(ctx, ["cancelledchild", "failedchild"], timeout_sec=0))

    cancelled = payload["tasks"]["cancelledchild"]
    assert cancelled["status"] == STATUS_CANCELLED
    assert cancelled["accounted_upper_bound_usd"] == 0.42
    assert cancelled["result"] == "best-effort partial handoff"
    failed = payload["tasks"]["failedchild"]
    assert failed["status"] == STATUS_FAILED
    # Absent accounting projects an honest null — never a confirmed-looking $0
    # (triad v6.71.2 r1; mirrors the ledger's unknown-cost discipline).
    assert "accounted_upper_bound_usd" in failed and failed["accounted_upper_bound_usd"] is None
    assert "child_result_sha256" in failed


def test_wait_for_tasks_rejected_duplicate_carries_duplicate_of(tmp_path):
    from ouroboros.task_results import STATUS_REJECTED_DUPLICATE, write_task_result
    from ouroboros.tools.control import _wait_for_tasks

    write_task_result(
        tmp_path,
        "dupechild",
        STATUS_REJECTED_DUPLICATE,
        result="duplicate of original123",
        duplicate_of="original123",
    )

    ctx = SimpleNamespace(drive_root=tmp_path)
    payload = json.loads(_wait_for_tasks(ctx, ["dupechild"], timeout_sec=0))

    dupe = payload["tasks"]["dupechild"]
    assert dupe["status"] == STATUS_REJECTED_DUPLICATE
    assert dupe["duplicate_of"] == "original123"
    assert "accounted_upper_bound_usd" in dupe


# --- v6.91 wait terminality: cancel_requested is a latch, not a settled record


def test_wait_for_effective_tasks_keeps_polling_cancel_requested(tmp_path):
    """The cancel-INTENT latch is not settled: the worker may still be exiting
    and the supervisor finalizes to `cancelled` shortly after. Returning
    "completed after 0.0s" here (pre-v6.91 FINAL_STATUSES) disagreed with the
    acceptance fence's SETTLED_STATUSES quiescence and looped the parent on the
    gap (wave3's $1.64 endgame loop). The wait stays bounded by its timeout."""
    from ouroboros.task_results import STATUS_CANCEL_REQUESTED, STATUS_CANCELLED, write_task_result
    from ouroboros.task_status import wait_for_effective_tasks

    write_task_result(tmp_path, "cancelling1", STATUS_CANCEL_REQUESTED, result="cancel pending")

    waited = wait_for_effective_tasks(tmp_path, ["cancelling1"], timeout_sec=0)
    assert waited["all_terminal"] is False
    assert waited["timed_out"] is True
    # A pending cancellation is reported as the typed state — never terminal/unknown.
    assert waited["live_child_status"]["cancelling1"] == "cancel_pending"

    # Once the supervisor settles it, the same wait completes normally.
    write_task_result(tmp_path, "cancelling1", STATUS_CANCELLED, result="cancelled")
    waited = wait_for_effective_tasks(tmp_path, ["cancelling1"], timeout_sec=0)
    assert waited["all_terminal"] is True


def test_wait_task_does_not_claim_completion_on_cancel_requested(tmp_path):
    from ouroboros.task_results import STATUS_CANCEL_REQUESTED, write_task_result
    from ouroboros.tools.control import _wait_for_task

    write_task_result(tmp_path, "cancelling2", STATUS_CANCEL_REQUESTED, result="cancel pending")

    output = _wait_for_task(SimpleNamespace(drive_root=tmp_path), "cancelling2", timeout_sec=0)
    assert output.startswith("Task wait timed out")
    assert not output.startswith("Task wait completed")


def test_wait_tools_surface_preentry_beacons_once_per_actor_context(tmp_path, monkeypatch):
    from ouroboros import task_tree_ledger
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from ouroboros.tools.control import _wait_for_task, _wait_for_tasks

    monkeypatch.setattr(task_tree_ledger, "DATA_DIR", str(tmp_path))
    for task_id in ("waitingchild1", "waitingchild2"):
        write_task_result(
            tmp_path, task_id, STATUS_RUNNING,
            parent_task_id="waitparent2", root_task_id="waitparent2",
            delegation_role="subagent",
        )
        assert task_tree_ledger.tree_ledger_append(
            "waitparent2", "question", f"preentry-{task_id}", task_id=task_id,
        ).startswith("OK:")

    ctx = SimpleNamespace(
        drive_root=tmp_path,
        task_id="waitparent2",
        task_metadata={"root_task_id": "waitparent2"},
    )
    single = _wait_for_task(ctx, "waitingchild1", timeout_sec=0)
    assert single.startswith("Task wait interrupted by a child attention beacon")
    assert "preentry-waitingchild1" in single

    batch = json.loads(_wait_for_tasks(ctx, ["waitingchild2"], timeout_sec=0))
    assert batch["early_return"]["reason"] == "child_attention_beacon"
    assert [row["text"] for row in batch["early_return"]["beacons"]] == [
        "preentry-waitingchild2"
    ]

    # A later wait in this same actor context does not replay either row.
    assert _wait_for_task(ctx, "waitingchild1", timeout_sec=0).startswith("Task wait timed out")
    assert "early_return" not in json.loads(
        _wait_for_tasks(ctx, ["waitingchild2"], timeout_sec=0)
    )


def test_wait_surfaces_preentry_beacon_before_terminal_fast_path(tmp_path, monkeypatch):
    from ouroboros import task_tree_ledger
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.tools.control import _wait_for_task

    monkeypatch.setattr(task_tree_ledger, "DATA_DIR", str(tmp_path))
    write_task_result(
        tmp_path, "terminal-beacon-child", STATUS_COMPLETED,
        parent_task_id="terminal-beacon-parent", root_task_id="terminal-beacon-parent",
        delegation_role="subagent",
    )
    assert task_tree_ledger.tree_ledger_append(
        "terminal-beacon-parent", "question", "answer me before completion",
        task_id="terminal-beacon-child",
    ).startswith("OK:")
    ctx = SimpleNamespace(
        drive_root=tmp_path, task_id="terminal-beacon-parent",
        task_metadata={"root_task_id": "terminal-beacon-parent"},
    )
    first = _wait_for_task(ctx, "terminal-beacon-child", timeout_sec=0)
    assert first.startswith("Task wait interrupted by a child attention beacon")
    assert "answer me before completion" in first
    assert _wait_for_task(ctx, "terminal-beacon-child", timeout_sec=0).startswith(
        "Task wait completed"
    )


# --- v6.91 wait_tasks typed unknown ids + children roster ---------------------


def test_wait_for_tasks_flags_unknown_ids_and_attaches_children_roster(tmp_path):
    import json as _json

    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.tools.control import _wait_for_tasks

    # A READABLE queue snapshot that does not know the phantom: a MISSING
    # snapshot fail-softs to "known" (never brand a real child unknown on an
    # unreadable surface), so the unknown verdict needs all surfaces present.
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "queue_snapshot.json").write_text(
        _json.dumps({"pending": [], "running": []}), encoding="utf-8"
    )
    write_task_result(
        tmp_path,
        "realchild1",
        STATUS_COMPLETED,
        result="real child finished",
        accounted_upper_bound_usd=0.55,
        parent_task_id="waitparent1",
        root_task_id="waitparent1",
        delegation_role="subagent",
    )

    ctx = SimpleNamespace(
        drive_root=tmp_path,
        task_id="waitparent1",
        task_metadata={"root_task_id": "waitparent1"},
    )
    payload = json.loads(_wait_for_tasks(ctx, ["realchild1", "phantomid9"], timeout_sec=0))

    # The phantom id gets a TYPED marker row, not a silent empty projection.
    phantom = payload["tasks"]["phantomid9"]
    assert phantom["unknown_task_id"] is True
    assert "not yet registered or never scheduled" in phantom["note"]
    assert payload["unknown_task_ids"] == ["phantomid9"]

    # The real child still projects the normal compact row.
    real = payload["tasks"]["realchild1"]
    assert real["status"] == STATUS_COMPLETED
    assert "unknown_task_id" not in real

    # The repair surface: the ACTUAL direct children, compact v6.71.2 field set
    # only — no result/trace envelope fields, absent accounting projects null.
    roster = payload["children_roster"]
    assert [row["task_id"] for row in roster] == ["realchild1"]
    assert set(roster[0]) == {"task_id", "status", "accounted_upper_bound_usd",
                              "child_result_sha256", "outcome_axes"}
    assert roster[0]["accounted_upper_bound_usd"] == 0.55
    # Nothing was capped away, and the projection SAYS so (BIBLE P1).
    assert payload["children_roster_omitted"] == 0


def test_children_roster_projection_discloses_the_capped_tail(tmp_path):
    """A parent with MORE direct children than the roster cap: the repair surface
    stays bounded, but the bound is disclosed — `children_roster_omitted` carries
    the exact count of real children the cap hid. A silent [:30] here could hide
    the very replacement id wait_tasks' unknown-id repair exists to surface."""
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.tools.control import _children_roster_projection

    total = 33
    for idx in range(total):
        write_task_result(
            tmp_path,
            f"bigchild{idx:03d}",
            STATUS_COMPLETED,
            result=f"child {idx} finished",
            parent_task_id="bigparent1",
            root_task_id="bigparent1",
            delegation_role="subagent",
        )

    ctx = SimpleNamespace(
        drive_root=tmp_path,
        task_id="bigparent1",
        task_metadata={"root_task_id": "bigparent1"},
    )
    projected = _children_roster_projection(ctx, tmp_path)
    roster = projected["children_roster"]
    assert len(roster) == 30  # the cap holds — the surface stays compact
    assert projected["children_roster_omitted"] == total - 30  # …and is disclosed
    assert all(
        set(row) == {"task_id", "status", "accounted_upper_bound_usd",
                     "child_result_sha256", "outcome_axes"}
        for row in roster
    )


def test_wait_for_tasks_phantom_only_set_short_circuits_the_window(tmp_path, monkeypatch):
    """A wait set in which NOTHING was ever minted ends after the registration
    grace instead of blocking the whole requested window — and says so."""
    import json as _json

    from ouroboros.tools import control
    # The wait internals live in the extracted owner leaf (v7 D07 split):
    # _wait_for_tasks reads the grace knob from its own module, so the
    # interception targets the leaf, not the re-exporting facade.
    from ouroboros.tools import control_task_results

    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "queue_snapshot.json").write_text(
        _json.dumps({"pending": [], "running": []}), encoding="utf-8"
    )
    monkeypatch.setattr(control_task_results, "_UNMINTED_WAIT_GRACE_SEC", 0.1)

    ctx = SimpleNamespace(drive_root=tmp_path, task_id="waitparent3", task_metadata={})
    started = time.monotonic()
    payload = json.loads(control._wait_for_tasks(ctx, ["phantomid7", "phantomid8"], timeout_sec=600))
    elapsed = time.monotonic() - started

    assert elapsed < 30, "phantom-only wait must not block for the requested window"
    short = payload["wait_short_circuited"]
    assert short["reason"] == "all_task_ids_unminted"
    assert short["requested_timeout_sec"] == 600.0
    assert sorted(payload["unknown_task_ids"]) == ["phantomid7", "phantomid8"]


def test_wait_for_tasks_id_minted_during_grace_keeps_waiting(tmp_path, monkeypatch):
    """The grace is for the registration race: an id that becomes real during it
    is a genuine child, so the wait resumes with the remaining window."""
    import json as _json

    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.tools import control
    # The wait internals live in the extracted owner leaf (v7 D07 split):
    # _wait_for_tasks reads the grace knob and the minted-ids probe from its
    # own module, so the interceptions target the leaf, not the facade.
    from ouroboros.tools import control_task_results

    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "queue_snapshot.json").write_text(
        _json.dumps({"pending": [], "running": []}), encoding="utf-8"
    )
    monkeypatch.setattr(control_task_results, "_UNMINTED_WAIT_GRACE_SEC", 0.1)

    real_calls = {"n": 0}
    original = control_task_results._unminted_wait_ids

    def _mint_after_grace(ctx, drive_root, task_ids):
        real_calls["n"] += 1
        if real_calls["n"] > 1:
            # The child registered during the grace window.
            write_task_result(
                tmp_path, "latechild1", STATUS_COMPLETED, result="registered late",
                parent_task_id="waitparent4", root_task_id="waitparent4",
                delegation_role="subagent",
            )
        return original(ctx, drive_root, task_ids)

    monkeypatch.setattr(control_task_results, "_unminted_wait_ids", _mint_after_grace)
    ctx = SimpleNamespace(drive_root=tmp_path, task_id="waitparent4", task_metadata={})
    payload = json.loads(control._wait_for_tasks(ctx, ["latechild1"], timeout_sec=5))

    assert "wait_short_circuited" not in payload
    assert payload["timeout_sec"] == 5.0
    assert payload["tasks"]["latechild1"]["status"] == STATUS_COMPLETED


def test_wait_for_tasks_queue_scheduled_id_is_not_unknown(tmp_path):
    """An id with a queue-snapshot row but no task result yet is a REAL child
    (just-scheduled), never a phantom — and without unknowns the roster is not
    attached (the compact batch stays compact, v6.71.2)."""
    import json as _json

    from ouroboros.tools.control import _wait_for_tasks

    snapshot = {"pending": [{"id": "queuedonly1", "task": {}}], "running": []}
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "queue_snapshot.json").write_text(_json.dumps(snapshot), encoding="utf-8")

    ctx = SimpleNamespace(drive_root=tmp_path, task_id="waitparent2", task_metadata={})
    payload = json.loads(_wait_for_tasks(ctx, ["queuedonly1"], timeout_sec=0))

    assert "unknown_task_ids" not in payload
    assert "children_roster" not in payload
    assert "unknown_task_id" not in payload["tasks"]["queuedonly1"]


def test_recent_tasks_includes_outcome_contract_and_ledger(tmp_path):
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.tools.recent_tasks import _handle_recent_tasks

    write_task_result(
        tmp_path,
        "recent1",
        STATUS_COMPLETED,
        result="done",
        task_contract={"schema_version": 1, "objective": "Do work"},
        outcome_axes={"execution": {"status": "ok"}, "objective": {"status": "not_evaluated"}},
        artifact_bundle={"schema_version": 1, "status": "ready_no_changes", "artifacts": [], "errors": []},
        verification_ledger={"schema_version": 2, "entries": [{"kind": "objective_outcome"}], "summary": {"entry_count": 1}},
    )

    payload = json.loads(_handle_recent_tasks(SimpleNamespace(drive_root=tmp_path), limit=1))
    record = payload["tasks"][0]

    assert record["outcome_axes"]["execution"]["status"] == "ok"
    assert record["task_contract"]["objective"] == "Do work"
    assert record["artifact_bundle"]["status"] == "ready_no_changes"
    assert record["verification_ledger"]["entry_count"] == 1

    # A ledger above the inline threshold rides as a stub with NO entries: its
    # own summary is the count authority, so the row must not report zero.
    write_task_result(
        tmp_path,
        "recent2",
        STATUS_COMPLETED,
        result="done",
        verification_ledger={
            "schema_version": 1, "omitted_to_artifact": True,
            "summary": {"entry_count": 9, "has_failures": True},
        },
    )
    payload = json.loads(_handle_recent_tasks(SimpleNamespace(drive_root=tmp_path), limit=2))
    stub_record = next(row for row in payload["tasks"] if row["task_id"] == "recent2")
    assert stub_record["verification_ledger"]["entry_count"] == 9
    assert stub_record["verification_ledger"]["summary"]["has_failures"] is True


def test_subtask_outcome_summary_reports_a_stub_ledger_count(tmp_path):
    """The same count authority on the parent-visible child summary: a stub
    ledger used to report ``0 entries / no failures`` to its parent."""
    from ouroboros.tools.control import _subtask_outcome_summary

    summary = json.loads(_subtask_outcome_summary({
        "status": "completed", "result": "done",
        "verification_ledger": {
            "schema_version": 1, "omitted_to_artifact": True,
            "summary": {"entry_count": 9, "has_failures": True},
        },
    }))
    assert summary["verification_ledger"]["entry_count"] == 9
    assert summary["verification_ledger"]["summary"]["has_failures"] is True


def test_effective_status_keeps_workspace_finalization_nonterminal_without_child_drive(tmp_path):
    from ouroboros.headless import ARTIFACT_STATUS_FINALIZING
    from ouroboros.task_results import STATUS_COMPLETED, STATUS_RUNNING, write_task_result
    from ouroboros.task_status import load_effective_task_result, wait_for_effective_tasks

    write_task_result(
        tmp_path,
        "workspace1",
        STATUS_COMPLETED,
        workspace_root=str(tmp_path / "workspace"),
        artifact_status=ARTIFACT_STATUS_FINALIZING,
        result="worker finished but artifacts are still pending",
    )

    effective = load_effective_task_result(tmp_path, "workspace1")
    waited = wait_for_effective_tasks(tmp_path, ["workspace1"], timeout_sec=0)

    assert effective["status"] == STATUS_RUNNING
    assert effective["child_status"] == STATUS_COMPLETED
    assert effective["artifact_status"] == ARTIFACT_STATUS_FINALIZING
    assert waited["all_terminal"] is False
    assert waited["timed_out"] is True


def test_effective_status_repairs_stale_running_infra_failure_when_queue_empty(tmp_path):
    from ouroboros.headless import ARTIFACT_STATUS_FINALIZING, ARTIFACT_STATUS_FAILED
    from ouroboros.task_results import STATUS_FAILED, STATUS_RUNNING, write_task_result
    from ouroboros.task_status import load_effective_task_result

    write_task_result(
        tmp_path,
        "providerfail",
        STATUS_RUNNING,
        workspace_root=str(tmp_path / "workspace"),
        artifact_status=ARTIFACT_STATUS_FINALIZING,
        result_status="infra_failed",
        reason_code="provider_failure",
        result="provider error",
        artifact_bundle={
            "status": ARTIFACT_STATUS_FINALIZING,
            "artifacts": [
                {"name": "deck.html", "status": ARTIFACT_STATUS_FINALIZING, "errors": []},
            ],
        },
    )
    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "state" / "queue_snapshot.json").write_text('{"pending": [], "running": []}', encoding="utf-8")

    effective = load_effective_task_result(tmp_path, "providerfail")

    assert effective["status"] == STATUS_FAILED
    assert effective["status_reconciled_from"] == STATUS_RUNNING
    assert effective["artifact_status"] == ARTIFACT_STATUS_FAILED
    assert effective["artifact_bundle"]["status"] == ARTIFACT_STATUS_FAILED
    assert effective["artifact_bundle"]["artifacts"][0]["status"] == ARTIFACT_STATUS_FAILED
    assert "task ended before artifact finalization" in effective["artifact_bundle"]["artifacts"][0]["errors"]


def test_effective_status_does_not_repair_running_when_queue_snapshot_missing(tmp_path):
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from ouroboros.task_status import load_effective_task_result

    write_task_result(
        tmp_path,
        "providerfail",
        STATUS_RUNNING,
        result_status="infra_failed",
        reason_code="provider_failure",
        result="provider error",
    )

    effective = load_effective_task_result(tmp_path, "providerfail")

    assert effective["status"] == STATUS_RUNNING
    assert effective["queue_reconciliation_warning"] == "queue snapshot missing or invalid"


def test_effective_status_repairs_orphan_running_after_worker_restart(tmp_path, monkeypatch):
    from ouroboros.headless import ARTIFACT_STATUS_FINALIZING, ARTIFACT_STATUS_FAILED
    from ouroboros.task_results import STATUS_FAILED, STATUS_RUNNING, write_task_result
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.utils import append_jsonl

    monkeypatch.setattr(time, "time", lambda: 1_800_000_000.0)
    write_task_result(
        tmp_path,
        "cc4db6fa",
        STATUS_RUNNING,
        result="Task is running.",
        ts="2026-05-28T00:00:00+00:00",
        artifact_status=ARTIFACT_STATUS_FINALIZING,
        artifact_bundle={
            "status": ARTIFACT_STATUS_FINALIZING,
            "artifacts": [
                {"name": "presentation.html", "status": ARTIFACT_STATUS_FINALIZING, "errors": []},
            ],
        },
    )
    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "state" / "queue_snapshot.json").write_text('{"pending": [], "running": []}', encoding="utf-8")
    events = tmp_path / "logs" / "events.jsonl"
    append_jsonl(events, {"ts": "2026-05-28T00:00:01+00:00", "type": "llm_round", "task_id": "cc4db6fa"})
    append_jsonl(events, {"ts": "2026-05-28T00:00:02+00:00", "type": "worker_boot"})

    effective = load_effective_task_result(tmp_path, "cc4db6fa")

    assert effective["status"] == STATUS_FAILED
    assert effective["status_reconciled_from"] == STATUS_RUNNING
    assert effective["outcome_axes"]["execution"]["status"] == "infra_failed"
    assert effective["reason_code"] == "orphaned_running_after_worker_restart"
    assert "TASK_ORPHAN_RECONCILED" in effective["result"]
    assert effective["artifact_status"] == ARTIFACT_STATUS_FAILED
    assert effective["artifact_bundle"]["artifacts"][0]["status"] == ARTIFACT_STATUS_FAILED
    assert "task interrupted before artifact finalization" in effective["artifact_bundle"]["artifacts"][0]["errors"]


def test_reconcile_durably_finalizes_orphaned_running_task(tmp_path, monkeypatch):
    # C5: the durable sweep persists what the read projection already decides, so
    # a headless/no-UI run that never re-reads the result no longer keeps a zombie
    # `running` record on disk.
    from ouroboros.task_results import (
        STATUS_FAILED,
        STATUS_RUNNING,
        load_task_result,
        write_task_result,
    )
    from ouroboros.task_status import reconcile_orphaned_running_tasks
    from ouroboros.utils import append_jsonl

    monkeypatch.setattr(time, "time", lambda: 1_800_000_000.0)
    write_task_result(
        tmp_path, "orphan1", STATUS_RUNNING,
        result="Task is running.", ts="2026-05-28T00:00:00+00:00",
    )
    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "state" / "queue_snapshot.json").write_text('{"pending": [], "running": []}', encoding="utf-8")
    events = tmp_path / "logs" / "events.jsonl"
    append_jsonl(events, {"ts": "2026-05-28T00:00:01+00:00", "type": "llm_round", "task_id": "orphan1"})
    append_jsonl(events, {"ts": "2026-05-28T00:00:02+00:00", "type": "worker_boot"})

    healed = reconcile_orphaned_running_tasks(tmp_path)

    assert healed == 1
    on_disk = load_task_result(tmp_path, "orphan1")
    assert on_disk["status"] == STATUS_FAILED
    assert on_disk["reason_code"] == "orphaned_running_after_worker_restart"


def test_best_effort_outcome_is_not_a_terminal_failure(tmp_path):
    # ...and the effective-status projection must NOT flip a best_effort
    # completion to failed: it is the documented non-failed, non-clean shelf.
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.task_status import load_effective_task_result

    write_task_result(
        tmp_path, "besteffort1", STATUS_COMPLETED,
        result="Partial best-effort answer.",
        outcome_axes={
            "execution": {"status": "best_effort", "reason_code": "round_limit_reached"},
            "objective": {"status": "not_evaluated"},
        },
    )
    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "state" / "queue_snapshot.json").write_text('{"pending": [], "running": []}', encoding="utf-8")

    effective = load_effective_task_result(tmp_path, "besteffort1")

    assert effective["status"] == STATUS_COMPLETED  # never reconciled to failed
    assert effective["outcome_axes"]["execution"]["status"] == "best_effort"


def test_reconcile_skips_running_when_queue_snapshot_missing(tmp_path):
    # Liveness gate: a missing/invalid queue snapshot means we cannot prove the
    # task is orphaned, so the sweep must leave the durable `running` untouched.
    from ouroboros.task_results import STATUS_RUNNING, load_task_result, write_task_result
    from ouroboros.task_status import reconcile_orphaned_running_tasks

    write_task_result(tmp_path, "live1", STATUS_RUNNING, result="still running")

    healed = reconcile_orphaned_running_tasks(tmp_path)

    assert healed == 0
    assert load_task_result(tmp_path, "live1")["status"] == STATUS_RUNNING


def test_find_child_tasks_does_not_regress_terminal_or_running_from_stale_queue_snapshot(tmp_path):
    from ouroboros.task_results import STATUS_COMPLETED, STATUS_RUNNING, write_task_result
    from ouroboros.task_status import find_child_tasks, load_effective_task_result

    write_task_result(
        tmp_path,
        "childdone",
        STATUS_COMPLETED,
        parent_task_id="parent1",
        root_task_id="parent1",
        delegation_role="subagent",
        result="terminal handoff",
    )
    write_task_result(
        tmp_path,
        "childrun",
        STATUS_RUNNING,
        parent_task_id="parent1",
        root_task_id="parent1",
        delegation_role="subagent",
        result="still working",
    )
    snapshot = {
        "pending": [
            {"id": "childdone", "task": {"id": "childdone", "parent_task_id": "parent1", "root_task_id": "parent1", "delegation_role": "subagent"}},
            {"id": "childrun", "task": {"id": "childrun", "parent_task_id": "parent1", "root_task_id": "parent1", "delegation_role": "subagent"}},
        ],
        "running": [],
    }
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "queue_snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")

    effective_done = load_effective_task_result(tmp_path, "childdone")
    effective_running = load_effective_task_result(tmp_path, "childrun")
    children = {row["task_id"]: row for row in find_child_tasks(tmp_path, parent_task_id="parent1", root_task_id="parent1")}

    assert effective_done["status"] == STATUS_COMPLETED
    assert effective_running["status"] == STATUS_RUNNING
    assert children["childdone"]["status"] == STATUS_COMPLETED
    assert children["childrun"]["status"] == STATUS_RUNNING


def test_effective_status_preserves_parent_retry_status_over_stale_child_running(tmp_path):
    from ouroboros.task_results import STATUS_INTERRUPTED, STATUS_RUNNING, STATUS_SCHEDULED, write_task_result
    from ouroboros.task_status import load_effective_task_result

    child_drive = tmp_path / "state" / "headless_tasks" / "childretry" / "data"
    child_drive.mkdir(parents=True)
    write_task_result(
        tmp_path,
        "childretry",
        STATUS_INTERRUPTED,
        child_drive_root=str(child_drive),
        parent_task_id="parent1",
        root_task_id="parent1",
        delegation_role="subagent",
        result="parent marked retry",
        error="worker interrupted",
        ts="2026-01-01T00:00:02Z",
    )
    write_task_result(
        child_drive,
        "childretry",
        STATUS_RUNNING,
        result="stale child still running",
        error="",
        ts="2026-01-01T00:00:01Z",
    )
    snapshot = {
        "pending": [
            {
                "id": "childretry",
                "task": {
                    "id": "childretry",
                    "parent_task_id": "parent1",
                    "root_task_id": "parent1",
                    "delegation_role": "subagent",
                },
            }
        ],
        "running": [],
    }
    (tmp_path / "state" / "queue_snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")

    effective = load_effective_task_result(tmp_path, "childretry")

    assert effective["status"] == STATUS_SCHEDULED
    assert effective["result"] == "parent marked retry"
    assert effective["error"] == "worker interrupted"


def test_find_child_tasks_requires_subagent_role_and_can_exclude_current_task(tmp_path):
    from ouroboros.task_results import STATUS_COMPLETED, STATUS_RUNNING, write_task_result
    from ouroboros.task_status import find_child_tasks

    write_task_result(
        tmp_path,
        "forgedroot",
        STATUS_COMPLETED,
        parent_task_id="parent1",
        root_task_id="parent1",
        delegation_role="root",
        result="should not be treated as child",
    )
    write_task_result(
        tmp_path,
        "child1",
        STATUS_RUNNING,
        parent_task_id="parent1",
        root_task_id="parent1",
        delegation_role="subagent",
        role="reviewer",
        result="x" * 2000,
        trace_summary="trace" * 500,
    )

    children = find_child_tasks(tmp_path, parent_task_id="parent1", root_task_id="parent1")
    excluded = find_child_tasks(tmp_path, parent_task_id="parent1", root_task_id="parent1", exclude_task_id="child1")

    assert [row["task_id"] for row in children] == ["child1"]
    assert excluded == []


def test_wait_for_task_times_out_when_child_is_not_terminal(tmp_path):
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from ouroboros.tools.control import _wait_for_task

    write_task_result(tmp_path, "stillrunning", STATUS_RUNNING, result="working")

    ctx = SimpleNamespace(drive_root=tmp_path)
    output = _wait_for_task(ctx, "stillrunning", timeout_sec=0)

    assert "Task wait timed out" in output
    assert "stillrunning [running]" in output


def test_wait_tools_reject_invalid_ids_and_cap_batch(tmp_path):
    from ouroboros.tools.control import _wait_for_task, _wait_for_tasks

    ctx = SimpleNamespace(drive_root=tmp_path)

    assert "TOOL_ARG_ERROR" in _wait_for_task(ctx, "../settings", timeout_sec=0)
    assert "TOOL_ARG_ERROR" in _wait_for_tasks(ctx, ["ok123", "../bad"], timeout_sec=0)
    from ouroboros.config import MAX_ACTIVE_SUBAGENTS_HARD_CAP
    assert MAX_ACTIVE_SUBAGENTS_HARD_CAP == 500
    assert "capped at 500" in _wait_for_tasks(
        ctx, [f"task{i}" for i in range(MAX_ACTIVE_SUBAGENTS_HARD_CAP + 1)], timeout_sec=0
    )


def test_wait_for_task_reports_rejected_duplicate(tmp_path):
    from ouroboros.task_results import STATUS_REJECTED_DUPLICATE, write_task_result
    from ouroboros.tools.control import _wait_for_task

    write_task_result(
        tmp_path,
        "dup123",
        STATUS_REJECTED_DUPLICATE,
        duplicate_of="orig999",
        result="Task was rejected as semantically similar to already active task orig999.",
    )

    ctx = SimpleNamespace(drive_root=tmp_path)
    output = _wait_for_task(ctx, "dup123")

    assert "rejected_duplicate" in output
    assert "duplicate_of=orig999" in output


def test_handle_schedule_task_duplicate_writes_rejected_status(tmp_path, monkeypatch):
    from supervisor import events as ev_module
    from supervisor import events_schedule_task as schedule_module
    from ouroboros.task_results import STATUS_REJECTED_DUPLICATE

    captured_identity = {}

    def _duplicate(*args, **kwargs):
        captured_identity.update(kwargs.get("dedupe_identity") or {})
        return "orig111"

    monkeypatch.setattr(schedule_module, "_find_duplicate_task", _duplicate)

    sent = []

    class FakeCtx:
        DRIVE_ROOT = tmp_path
        PENDING = []
        RUNNING = {}
        WORKERS = {0: SimpleNamespace(busy_task_id=None)}

        def load_state(self):
            return {"owner_chat_id": 1}

        def send_with_budget(self, chat_id, text, **kwargs):
            sent.append((chat_id, text, kwargs))

    ev_module._handle_schedule_task(
        {
            "type": "schedule_subagent",
            "task_id": "dup222",
            "objective": "Do the thing",
            "expected_output": "Duplicate verdict",
            "context": "Model focus B",
            "depth": 1,
            "memory_mode": "forked",
            "parent_task_id": "parent111",
            "root_task_id": "root111",
            "drive_root": str(tmp_path / "state" / "headless_tasks" / "dup222" / "data"),
            "child_drive_root": str(tmp_path / "state" / "headless_tasks" / "dup222" / "data"),
            "budget_drive_root": str(tmp_path),
        },
        FakeCtx(),
    )

    path = tmp_path / "task_results" / "dup222.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["status"] == STATUS_REJECTED_DUPLICATE
    assert data["duplicate_of"] == "orig111"
    assert sent and "semantically similar" in sent[0][1]
    assert sent[0][2]["is_progress"] is True
    assert sent[0][2]["progress_meta"]["delegation_role"] == "subagent"
    assert sent[0][2]["progress_meta"]["parent_task_id"] == "parent111"
    assert sent[0][2]["progress_meta"]["status"] == STATUS_REJECTED_DUPLICATE
    assert captured_identity == {
        "delegation_role": "subagent",
        "task_id": "dup222",
        "parent_task_id": "parent111",
        "root_task_id": "root111",
        "budget_drive_root": str(tmp_path),
    }


def test_find_duplicate_task_includes_subagent_handoff_fields(monkeypatch):
    from supervisor import events as ev_module
    import ouroboros.config as config_module
    import ouroboros.llm as llm_module

    captured = {}

    class FakeClient:
        def chat(self, messages, **kwargs):
            captured["prompt"] = messages[0]["content"]
            return {"content": "NONE"}, {}

    monkeypatch.setattr(config_module, "get_light_model", lambda: "test-light")
    monkeypatch.setattr(llm_module, "LLMClient", lambda: FakeClient())

    result = ev_module._find_duplicate_task(
        "Review shared surface",
        "same context",
        [
            {
                "id": "pending1",
                "description": "Review shared surface",
                "context": "same context",
                "expected_output": "Docs table",
                "constraints": "docs only",
                "role": "docs reviewer",
            }
        ],
        {},
        expected_output="Security table",
        constraints="security only",
        role="security reviewer",
    )

    assert result is None
    prompt = captured["prompt"]
    assert "Expected output:\nSecurity table" in prompt
    assert "Expected output:\nDocs table" in prompt
    assert "Constraints:\nsecurity only" in prompt
    assert "Constraints:\ndocs only" in prompt
    assert "Role:\nsecurity reviewer" in prompt
    assert "Role:\ndocs reviewer" in prompt


def test_find_duplicate_task_allows_distinct_subagent_roles(monkeypatch):
    from supervisor import events as ev_module
    import ouroboros.config as config_module
    import ouroboros.llm as llm_module

    calls = []

    class FakeClient:
        def chat(self, messages, **kwargs):
            calls.append(messages[0]["content"])
            return {"content": "pending1"}, {}

    monkeypatch.setattr(config_module, "get_light_model", lambda: "test-light")
    monkeypatch.setattr(llm_module, "LLMClient", lambda: FakeClient())

    result = ev_module._find_duplicate_task(
        "Run nested smoke slot",
        "",
        [
            {
                "id": "pending1",
                "description": "Run nested smoke slot",
                "expected_output": "Smoke handoff",
                "role": "l1-alpha-coordinator",
                "delegation_role": "subagent",
                "parent_task_id": "root1",
                "root_task_id": "root1",
            }
        ],
        {},
        expected_output="Smoke handoff",
        role="l1-beta-coordinator",
        dedupe_identity={
            "delegation_role": "subagent",
            "parent_task_id": "root1",
            "root_task_id": "root1",
        },
    )

    assert result is None
    assert calls == []


def test_find_duplicate_task_keeps_same_role_subagent_dedupe(monkeypatch):
    from supervisor import events as ev_module
    import ouroboros.config as config_module
    import ouroboros.llm as llm_module

    class FakeClient:
        def chat(self, messages, **kwargs):
            return {"content": "pending1"}, {}

    monkeypatch.setattr(config_module, "get_light_model", lambda: "test-light")
    monkeypatch.setattr(llm_module, "LLMClient", lambda: FakeClient())

    result = ev_module._find_duplicate_task(
        "Run nested smoke slot",
        "",
        [
            {
                "id": "pending1",
                "description": "Run nested smoke slot",
                "expected_output": "Smoke handoff",
                "role": "l1-alpha-coordinator",
                "delegation_role": "subagent",
                "parent_task_id": "root1",
                "root_task_id": "root1",
            }
        ],
        {},
        expected_output="Smoke handoff",
        role="l1-alpha-coordinator",
        dedupe_identity={
            "delegation_role": "subagent",
            "parent_task_id": "root1",
            "root_task_id": "root1",
        },
    )

    assert result == "pending1"


def test_find_duplicate_task_allows_distinct_subagent_parent_branches(monkeypatch):
    from supervisor import events as ev_module
    import ouroboros.config as config_module
    import ouroboros.llm as llm_module

    calls = []

    class FakeClient:
        def chat(self, messages, **kwargs):
            calls.append(messages[0]["content"])
            return {"content": "pending1"}, {}

    monkeypatch.setattr(config_module, "get_light_model", lambda: "test-light")
    monkeypatch.setattr(llm_module, "LLMClient", lambda: FakeClient())

    result = ev_module._find_duplicate_task(
        "Run nested branch smoke slot",
        "",
        [
            {
                "id": "pending1",
                "description": "Run nested branch smoke slot",
                "expected_output": "Smoke handoff",
                "role": "shared-l2-role",
                "delegation_role": "subagent",
                "parent_task_id": "l1-alpha",
                "root_task_id": "root1",
            }
        ],
        {},
        expected_output="Smoke handoff",
        role="shared-l2-role",
        dedupe_identity={
            "delegation_role": "subagent",
            "parent_task_id": "l1-beta",
            "root_task_id": "root1",
        },
    )

    assert result is None
    assert calls == []


def test_find_duplicate_task_allows_subagent_against_running_root_ancestor(monkeypatch):
    from supervisor import events as ev_module
    import ouroboros.config as config_module
    import ouroboros.llm as llm_module

    calls = []

    class FakeClient:
        def chat(self, messages, **kwargs):
            calls.append(messages[0]["content"])
            return {"content": "root1"}, {}

    monkeypatch.setattr(config_module, "get_light_model", lambda: "test-light")
    monkeypatch.setattr(llm_module, "LLMClient", lambda: FakeClient())

    result = ev_module._find_duplicate_task(
        "You are l1-alpha-coordinator; schedule L2 smoke agents",
        "",
        [],
        {
            "root1": {
                "task": {
                    "id": "root1",
                    "description": "Root coordinator: schedule l1-alpha, l1-beta, l1-gamma subagents",
                    "delegation_role": "root",
                    "parent_task_id": "",
                    "root_task_id": "root1",
                }
            }
        },
        expected_output="L1 handoff",
        role="l1-alpha-coordinator",
        dedupe_identity={
            "delegation_role": "subagent",
            "parent_task_id": "root1",
            "root_task_id": "root1",
        },
    )

    assert result is None
    assert calls == []


def test_find_duplicate_task_allows_subagent_against_pending_parent_ancestor(monkeypatch):
    from supervisor import events as ev_module
    import ouroboros.config as config_module
    import ouroboros.llm as llm_module

    calls = []

    class FakeClient:
        def chat(self, messages, **kwargs):
            calls.append(messages[0]["content"])
            return {"content": "parent1"}, {}

    monkeypatch.setattr(config_module, "get_light_model", lambda: "test-light")
    monkeypatch.setattr(llm_module, "LLMClient", lambda: FakeClient())

    result = ev_module._find_duplicate_task(
        "You are l1-alpha-coordinator-l2-1; return a smoke handoff",
        "",
        [
            {
                "id": "parent1",
                "description": "You are l1-alpha-coordinator; schedule three L2 smoke subagents",
                "role": "l1-alpha-coordinator",
                "delegation_role": "subagent",
                "parent_task_id": "root1",
                "root_task_id": "root1",
            }
        ],
        {},
        expected_output="L2 handoff",
        role="l1-alpha-coordinator-l2-1",
        dedupe_identity={
            "delegation_role": "subagent",
            "parent_task_id": "parent1",
            "root_task_id": "root1",
        },
    )

    assert result is None
    assert calls == []


def test_handle_schedule_task_accepts_unique_subagent_with_lineage_and_constraint(tmp_path, monkeypatch):
    from supervisor import events as ev_module
    from supervisor import events_schedule_task as schedule_module
    from ouroboros.task_results import STATUS_SCHEDULED

    monkeypatch.setattr(schedule_module, "_find_duplicate_task", lambda *args, **kwargs: None)
    enqueued = []
    sent = []

    class FakeCtx:
        DRIVE_ROOT = tmp_path
        PENDING = []
        RUNNING = {}
        WORKERS = {0: SimpleNamespace(busy_task_id=None)}

        def load_state(self):
            return {"owner_chat_id": 1}

        def send_with_budget(self, chat_id, text, **kwargs):
            sent.append((chat_id, text, kwargs))

        def enqueue_task(self, task):
            enqueued.append(task)

        def persist_queue_snapshot(self, reason=""):
            self.snapshot_reason = reason

    ev_module._handle_schedule_task(
        {
            "type": "schedule_subagent",
            "task_id": "child123",
            "objective": "Inspect scheduling",
            "expected_output": "Findings table",
            "constraints": "No writes",
            "role": "reviewer",
            "context": "Parent facts",
            "depth": 1,
            "parent_task_id": "parent123",
            "root_task_id": "root123",
            "session_id": "sess123",
            "actor_id": "subagent:reviewer",
            "delegation_role": "subagent",
            "memory_mode": "forked",
            "drive_root": str(tmp_path / "state" / "headless_tasks" / "child123" / "data"),
            "child_drive_root": str(tmp_path / "state" / "headless_tasks" / "child123" / "data"),
            "budget_drive_root": str(tmp_path),
            "task_constraint": {"mode": "skill_repair", "allow_enable": True, "allow_review": True},
        },
        FakeCtx(),
    )

    assert len(enqueued) == 1
    task = enqueued[0]
    assert task["id"] == "child123"
    assert task["parent_task_id"] == "parent123"
    assert task["root_task_id"] == "root123"
    assert task["session_id"] == "sess123"
    assert task["role"] == "reviewer"
    assert task["memory_mode"] == "forked"
    assert task["child_drive_root"] == task["drive_root"]
    assert task["task_constraint"]["mode"] == "local_readonly_subagent"
    assert task["task_constraint"]["allow_enable"] is False
    assert task["task_constraint"]["allow_review"] is False
    assert "[EXPECTED_OUTPUT]" in task["text"]
    assert "[BEGIN_PARENT_CONTEXT" in task["text"]
    data = json.loads((tmp_path / "task_results" / "child123.json").read_text(encoding="utf-8"))
    assert data["status"] == STATUS_SCHEDULED
    assert data["expected_output"] == "Findings table"
    assert data["child_drive_root"] == task["drive_root"]
    assert data["task_constraint"]["mode"] == "local_readonly_subagent"
    assert "Do not delegate further" not in task["text"]
    assert "Nested readonly delegation is allowed only through schedule_subagent" in task["text"]
    assert sent and sent[0][2].get("is_progress") is True


def test_handle_schedule_task_rejects_internal_subagent_without_child_drive_contract(tmp_path, monkeypatch):
    from supervisor import events as ev_module
    from supervisor import events_schedule_task as schedule_module
    from ouroboros.task_results import STATUS_FAILED

    monkeypatch.setattr(schedule_module, "_find_duplicate_task", lambda *args, **kwargs: None)
    sent = []

    class FakeCtx:
        DRIVE_ROOT = tmp_path
        PENDING = []
        RUNNING = {}
        WORKERS = {0: SimpleNamespace(busy_task_id=None)}

        def load_state(self):
            return {"owner_chat_id": 1}

        def send_with_budget(self, chat_id, text, **kwargs):
            sent.append((chat_id, text, kwargs))

        def enqueue_task(self, task):
            raise AssertionError("invalid internal subagent should not enqueue")

    ev_module._handle_schedule_task(
        {
            "type": "schedule_subagent",
            "task_id": "badchild",
            "objective": "Inspect invalid event",
            "expected_output": "Nothing",
            "depth": 1,
            "delegation_role": "subagent",
            "memory_mode": "shared",
        },
        FakeCtx(),
    )

    data = json.loads((tmp_path / "task_results" / "badchild.json").read_text(encoding="utf-8"))
    assert data["status"] == STATUS_FAILED
    assert "memory_mode=forked or empty" in data["result"]
    assert sent and sent[0][2]["progress_meta"]["subagent_event"] == "rejected"
    assert sent[0][2]["progress_meta"]["delegation_role"] == "subagent"
    assert sent[0][2]["progress_meta"]["parent_task_id"] == ""
    assert sent[0][2]["progress_meta"]["status"] == STATUS_FAILED


def test_handle_schedule_task_uses_event_chat_id_without_owner(tmp_path, monkeypatch):
    from supervisor import events as ev_module
    from supervisor import events_schedule_task as schedule_module
    from ouroboros.task_results import STATUS_SCHEDULED

    monkeypatch.setattr(schedule_module, "_find_duplicate_task", lambda *args, **kwargs: None)
    enqueued = []
    sent = []

    class FakeCtx:
        DRIVE_ROOT = tmp_path
        PENDING = []
        RUNNING = {}
        WORKERS = {0: SimpleNamespace(busy_task_id=None)}

        def load_state(self):
            return {}

        def send_with_budget(self, chat_id, text, **kwargs):
            sent.append((chat_id, text, kwargs))

        def enqueue_task(self, task):
            enqueued.append(task)

        def persist_queue_snapshot(self, reason=""):
            self.snapshot_reason = reason

    ev_module._handle_schedule_task(
        {
            "type": "schedule_subagent",
            "task_id": "headless1",
            "objective": "Inspect no-owner path",
            "expected_output": "Findings",
            "depth": 1,
            "chat_id": 44,
            "delegation_role": "subagent",
            "memory_mode": "forked",
            "drive_root": str(tmp_path / "state" / "headless_tasks" / "headless1" / "data"),
            "child_drive_root": str(tmp_path / "state" / "headless_tasks" / "headless1" / "data"),
        },
        FakeCtx(),
    )

    assert len(enqueued) == 1
    assert enqueued[0]["chat_id"] == 44
    scheduled = json.loads((tmp_path / "task_results" / "headless1.json").read_text(encoding="utf-8"))
    assert scheduled["status"] == STATUS_SCHEDULED
    assert scheduled["chat_id"] == 44
    assert sent and sent[0][0] == 44

    ev_module._handle_schedule_task(
        {
            "type": "schedule_subagent",
            "task_id": "headless2",
            "objective": "Inspect missing chat target",
            "expected_output": "Findings",
            "depth": 1,
            "delegation_role": "subagent",
            "memory_mode": "forked",
            "drive_root": str(tmp_path / "state" / "headless_tasks" / "headless2" / "data"),
            "child_drive_root": str(tmp_path / "state" / "headless_tasks" / "headless2" / "data"),
        },
        FakeCtx(),
    )

    # B1 (v6.33.0): a headless subagent with no chat target is no longer
    # rejected — it is enqueued and runs. The durable record now keeps its real
    # address instead of recording the hidden partition as "no chat" (that is the
    # truthiness class this sprint closed), while the LIVE toast is still skipped:
    # a progress notice needs a reader, the hidden partition has none, and a
    # headless run's progress log is read back as its benchmark trajectory.
    assert len(enqueued) == 2
    assert enqueued[1]["id"] == "headless2"
    scheduled2 = json.loads((tmp_path / "task_results" / "headless2.json").read_text(encoding="utf-8"))
    assert scheduled2["status"] == STATUS_SCHEDULED
    assert scheduled2["chat_id"] == 0
    assert len(sent) == 1


def test_handle_schedule_task_depth_rejection_writes_failed_status(tmp_path, monkeypatch):
    from supervisor import events as ev_module
    from supervisor import events_schedule_task as schedule_module
    from ouroboros.config import get_max_subagent_depth
    from ouroboros.task_results import STATUS_FAILED

    monkeypatch.setattr(schedule_module, "_find_duplicate_task", lambda *args, **kwargs: None)
    sent = []

    class FakeCtx:
        DRIVE_ROOT = tmp_path
        PENDING = []
        RUNNING = {}
        WORKERS = {0: SimpleNamespace(busy_task_id=None)}

        def load_state(self):
            return {"owner_chat_id": 1}

        def send_with_budget(self, chat_id, text, **kwargs):
            sent.append((chat_id, text, kwargs))

        def enqueue_task(self, task):
            raise AssertionError("depth-rejected task should not enqueue")

    ev_module._handle_schedule_task(
        {
            "type": "schedule_subagent",
            "task_id": "deep1",
            "objective": "Too deep",
            "expected_output": "Nothing",
            "depth": get_max_subagent_depth() + 1,
            "delegation_role": "subagent",
            "memory_mode": "forked",
            "drive_root": str(tmp_path / "state" / "headless_tasks" / "deep1" / "data"),
            "child_drive_root": str(tmp_path / "state" / "headless_tasks" / "deep1" / "data"),
        },
        FakeCtx(),
    )

    data = json.loads((tmp_path / "task_results" / "deep1.json").read_text(encoding="utf-8"))
    assert data["status"] == STATUS_FAILED
    assert "depth limit" in data["result"]
    assert sent and "depth limit" in sent[0][1]
    assert sent[0][2]["is_progress"] is True
    assert sent[0][2]["progress_meta"]["delegation_role"] == "subagent"
    assert sent[0][2]["progress_meta"]["status"] == STATUS_FAILED


def test_configured_zero_subagent_depth_truly_disables_delegation(tmp_path, monkeypatch):
    """A configured depth of zero disables child delegation, not the root task."""
    from supervisor import events as ev_module
    from supervisor import events_schedule_task as schedule_module
    from ouroboros.config import get_max_subagent_depth
    from ouroboros.task_results import STATUS_FAILED

    monkeypatch.setenv("OUROBOROS_MAX_SUBAGENT_DEPTH", "0")
    _configure_test_subagent(monkeypatch)
    assert get_max_subagent_depth() == 0

    import ouroboros.tools.control as control
    from ouroboros.tools.registry import ToolContext

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "root-no-swarm"
    ctx.task_depth = 0
    out = control._schedule_task(
        ctx, subagent_id="api-scout", objective="Delegate", expected_output="Something")
    assert "depth limit (0) exceeded" in out
    assert "subtask_depth_limit" in out
    refused_id = out.split("task_id=", 1)[1].split(";", 1)[0]
    refused = json.loads(
        (tmp_path / "task_results" / f"{refused_id}.json").read_text(encoding="utf-8")
    )
    assert refused["status"] == STATUS_FAILED
    assert refused["delegation_admission"]["reason_code"] == "subtask_depth_limit"
    assert refused["depth_provenance"] == {
        "requested_depth": None,
        "permitted_depth": 0,
        "attempted_depth": 1,
        "achieved_depth": None,
    }

    monkeypatch.setattr(schedule_module, "_find_duplicate_task", lambda *args, **kwargs: None)
    enqueued = []

    class FakeCtx:
        DRIVE_ROOT = tmp_path
        PENDING = []
        RUNNING = {}
        WORKERS = {0: SimpleNamespace(busy_task_id=None)}

        def load_state(self):
            return {"owner_chat_id": 1}

        def send_with_budget(self, chat_id, text, **kwargs):
            pass

        def enqueue_task(self, task):
            enqueued.append(task)

        def persist_queue_snapshot(self, reason=""):
            pass

    def _event(task_id: str, depth: int) -> dict:
        return {
            "type": "schedule_subagent",
            "task_id": task_id,
            "objective": "work",
            "expected_output": "result",
            "depth": depth,
            "delegation_role": "subagent" if depth else "",
            "memory_mode": "forked",
            "chat_id": 1,
            "drive_root": str(tmp_path / "state" / "headless_tasks" / task_id / "data"),
            "child_drive_root": str(tmp_path / "state" / "headless_tasks" / task_id / "data"),
        }

    ev_module._handle_schedule_task(_event("child-at-1", 1), FakeCtx())
    child = json.loads((tmp_path / "task_results" / "child-at-1.json").read_text(encoding="utf-8"))
    assert child["status"] == STATUS_FAILED and "depth limit (0)" in child["result"]
    assert not enqueued

    ev_module._handle_schedule_task(_event("root-at-0", 0), FakeCtx())
    root = json.loads((tmp_path / "task_results" / "root-at-0.json").read_text(encoding="utf-8"))
    assert root["status"] != STATUS_FAILED
    assert enqueued and enqueued[0]["id"] == "root-at-0"


def test_other_bounded_int_settings_keep_their_min_of_one(monkeypatch):
    """``min_value`` defaults to 1, so the depth fix does not leak into sibling settings."""
    from ouroboros.config import get_max_active_subagents_per_root, SETTINGS_DEFAULTS

    monkeypatch.setenv("OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT", "0")
    assert get_max_active_subagents_per_root() == int(
        SETTINGS_DEFAULTS["OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT"]
    )


def test_settings_ui_carries_a_configured_zero_subagent_depth():
    """The runtime honouring 0 is worthless if the Settings page silently reverts it: 0 is FALSY
    in JS, so a stored 0 read through the plain `if (value)` branch displayed the fallback 3, and
    the next Save (which posts every number field unconditionally) wrote 3 back — re-enabling three
    levels of delegation through the UI. All three carriers of the owner's 0 are pinned: the input
    can reach it, the depth entry is falsy-tolerant, and the load path still honours that flag
    (without which the flag is inert)."""
    root = pathlib.Path(__file__).resolve().parents[1]
    settings_js = (root / "web" / "modules" / "settings.js").read_text(encoding="utf-8")
    # The input moved from Advanced -> Runtime Limits to Agents -> Delegation
    # (D-10): the counts bound the agents, not the process pool. Same invariant,
    # new address.
    settings_ui = (root / "web" / "modules" / "subagents_settings.js").read_text(encoding="utf-8")
    assert 'id="s-subagent-depth" type="number" min="0"' in settings_ui
    # The 4th tuple element is the falsy-tolerant flag consumed by the load path below.
    assert "['s-subagent-depth', 'OUROBOROS_MAX_SUBAGENT_DEPTH', 3, true]" in settings_js
    assert (
        "if (allowFalsy ? value !== null && value !== undefined : value) byId(id).value = value;"
        in settings_js
    ), "the load path no longer honours the falsy-tolerant flag, so the entry is inert"


def test_handle_schedule_task_rejects_legacy_subagent_event_schema(tmp_path, monkeypatch):
    from supervisor import events as ev_module
    from supervisor import events_schedule_task as schedule_module
    from ouroboros.task_results import STATUS_FAILED

    monkeypatch.setattr(schedule_module, "_find_duplicate_task", lambda *args, **kwargs: None)
    enqueued = []
    sent = []

    class FakeCtx:
        DRIVE_ROOT = tmp_path
        PENDING = []
        RUNNING = {}
        WORKERS = {0: SimpleNamespace(busy_task_id=None)}

        def load_state(self):
            return {"owner_chat_id": 1}

        def send_with_budget(self, chat_id, text, **kwargs):
            sent.append((chat_id, text, kwargs))

        def enqueue_task(self, task):
            enqueued.append(task)

        def persist_queue_snapshot(self, reason=""):
            return None

    ev_module._handle_schedule_task(
        {
            "type": "schedule_subagent",
            "task_id": "legacy123",
            "description": "Old child form",
            "context": "old reference",
            "parent_task_id": "parent123",
            "delegation_role": "subagent",
        },
        FakeCtx(),
    )

    assert enqueued == []
    data = json.loads((tmp_path / "task_results" / "legacy123.json").read_text(encoding="utf-8"))
    assert data["status"] == STATUS_FAILED
    assert "objective and expected_output" in data["result"]
    assert sent and "objective and expected_output" in sent[0][1]
    assert sent[0][2]["is_progress"] is True
    assert sent[0][2]["progress_meta"]["delegation_role"] == "subagent"
    assert sent[0][2]["progress_meta"]["parent_task_id"] == "parent123"
    assert sent[0][2]["progress_meta"]["status"] == STATUS_FAILED


def test_handle_schedule_task_queues_when_active_subagent_cap_is_full(tmp_path, monkeypatch):
    from supervisor import events as ev_module
    from supervisor import events_schedule_task as schedule_module
    from ouroboros.task_results import STATUS_COMPLETED, STATUS_FAILED, STATUS_SCHEDULED, load_task_result, write_task_result

    monkeypatch.setattr(schedule_module, "_find_duplicate_task", lambda *args, **kwargs: None)
    monkeypatch.setenv("OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT", "3")  # pin cap (v6.20.0 raised default to 6)
    sent = []
    enqueued = []

    class FakeCtx:
        DRIVE_ROOT = tmp_path
        PENDING = [{"id": f"p{i}", "root_task_id": "root123", "delegation_role": "subagent"} for i in range(2)]
        RUNNING = {"r1": {"task": {"id": "r1", "root_task_id": "root123", "delegation_role": "subagent"}}}
        WORKERS = {0: SimpleNamespace(busy_task_id=None)}

        def load_state(self):
            return {"owner_chat_id": 1}

        def send_with_budget(self, chat_id, text, **kwargs):
            sent.append((chat_id, text, kwargs))

        def enqueue_task(self, task):
            enqueued.append(task)

        def persist_queue_snapshot(self, reason=""):
            pass

    ev_module._handle_schedule_task(
        {
            "type": "schedule_subagent",
            "task_id": "child999",
            "objective": "Too many",
            "expected_output": "Nothing",
            "depth": 1,
            "root_task_id": "root123",
            "delegation_role": "subagent",
            "memory_mode": "forked",
            "drive_root": str(tmp_path / "state" / "headless_tasks" / "child999" / "data"),
            "child_drive_root": str(tmp_path / "state" / "headless_tasks" / "child999" / "data"),
        },
        FakeCtx(),
    )

    data = json.loads((tmp_path / "task_results" / "child999.json").read_text(encoding="utf-8"))
    assert data["status"] == STATUS_SCHEDULED
    assert enqueued and enqueued[0]["id"] == "child999"
    assert sent and "queued behind active subagent cap" in sent[0][1]
    assert sent[0][2]["is_progress"] is True
    assert sent[0][2]["progress_meta"]["delegation_role"] == "subagent"
    assert sent[0][2]["progress_meta"]["queued_behind_active_cap"] is True

    ev_module._handle_schedule_task(
        {
            "type": "schedule_subagent",
            "task_id": "child1000",
            "objective": "Too many again",
            "expected_output": "Nothing",
            "depth": 1,
            "root_task_id": "root123",
            "delegation_role": "subagent",
            "memory_mode": "forked",
            "drive_root": str(tmp_path / "state" / "headless_tasks" / "child1000" / "data"),
            "child_drive_root": str(tmp_path / "state" / "headless_tasks" / "child1000" / "data"),
        },
        FakeCtx(),
    )
    data2 = json.loads((tmp_path / "task_results" / "child1000.json").read_text(encoding="utf-8"))
    assert data2["status"] == STATUS_SCHEDULED
    assert any(task["id"] == "child1000" for task in enqueued)

    child_drive = tmp_path / "state" / "headless_tasks" / "childdone" / "data"
    (child_drive / "memory").mkdir(parents=True)
    (child_drive / "memory" / "identity.md").write_text("child identity", encoding="utf-8")
    child_review_projection = {
        "panels": [{
            "panel_id": "child-panel",
            "aggregate_signal": "DEGRADED",
            "actors": [],
        }],
    }
    child_outcome_axes = {
        "lifecycle": {"status": "completed"},
        "execution": {"status": "ok"},
        "objective": {"status": "best_effort"},
        "review": {"status": "degraded"},
        "artifacts": {"status": "ready"},
    }
    write_task_result(
        child_drive,
        "childdone",
        STATUS_COMPLETED,
        result="summary",
        outcome_axes=child_outcome_axes,
        reason_code="acceptance_degraded",
        review_projection=child_review_projection,
    )

    sent = []
    worker = SimpleNamespace(busy_task_id="childdone")
    ctx = SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={
            "childdone": {
                "task": {
                    "id": "childdone",
                    "chat_id": 1,
                    "drive_root": str(child_drive),
                    "delegation_role": "subagent",
                    "role": "reviewer",
                    "root_task_id": "root123",
                    "parent_task_id": "parent123",
                    "task_constraint": {"mode": "local_readonly_subagent", "allow_enable": False},
                }
            }
        },
        WORKERS={7: worker},
        bridge=SimpleNamespace(push_log=lambda _payload: None),
        send_with_budget=lambda chat_id, text, **kwargs: sent.append((chat_id, text, kwargs)),
        persist_queue_snapshot=lambda reason="": None,
    )

    ev_module._handle_task_done({"task_id": "childdone", "worker_id": 7, "task_type": "task"}, ctx)

    assert load_task_result(tmp_path, "childdone")["result"] == "summary"
    assert not (tmp_path / "task_results" / "artifacts" / "childdone" / "memory_export.json").exists()
    assert sent and sent[-1][2]["progress_meta"]["subagent_role"] == "reviewer"
    terminal_meta = sent[-1][2]["progress_meta"]
    assert terminal_meta["outcome_axes"]["review"]["status"] == "degraded"
    assert terminal_meta["reason_code"] == "acceptance_degraded"
    assert terminal_meta["review_projection"] == child_review_projection

    failed_drive = tmp_path / "state" / "headless_tasks" / "childfail" / "data"
    (failed_drive / "task_results").mkdir(parents=True)
    write_task_result(failed_drive, "childfail", STATUS_FAILED, result="boom")
    sent = []
    worker = SimpleNamespace(busy_task_id="childfail")
    ctx = SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={
            "childfail": {
                "task": {
                    "id": "childfail",
                    "chat_id": 1,
                    "drive_root": str(failed_drive),
                    "delegation_role": "subagent",
                    "role": "reviewer",
                    "root_task_id": "root123",
                    "parent_task_id": "parent123",
                    "task_constraint": {"mode": "local_readonly_subagent", "allow_enable": False},
                }
            }
        },
        WORKERS={8: worker},
        bridge=SimpleNamespace(push_log=lambda _payload: None),
        send_with_budget=lambda chat_id, text, **kwargs: sent.append((chat_id, text, kwargs)),
        persist_queue_snapshot=lambda reason="": None,
    )

    ev_module._handle_task_done({"task_id": "childfail", "worker_id": 8, "task_type": "task"}, ctx)

    assert load_task_result(tmp_path, "childfail")["status"] == STATUS_FAILED
    assert sent and "failed" in sent[-1][1]
    assert sent[-1][2]["progress_meta"]["subagent_event"] == "failed"


def test_handle_schedule_task_fails_fast_when_worker_pool_unavailable(tmp_path, monkeypatch):
    """When the worker pool is empty (e.g. disabled after a crash storm), a
    schedule must NOT be left as a 'scheduled' ghost — it gets a terminal
    workers_unavailable result so the parent can act."""
    from supervisor import events as ev_module
    from supervisor import events_schedule_task as schedule_module
    from ouroboros.task_results import STATUS_FAILED

    monkeypatch.setattr(schedule_module, "_find_duplicate_task", lambda *args, **kwargs: None)
    sent = []

    class FakeCtx:
        DRIVE_ROOT = tmp_path
        PENDING = []
        RUNNING = {}
        WORKERS = {}  # pool disabled / not available

        def load_state(self):
            return {"owner_chat_id": 1}

        def send_with_budget(self, chat_id, text, **kwargs):
            sent.append((chat_id, text, kwargs))

        def enqueue_task(self, task):
            raise AssertionError("must not enqueue when worker pool is unavailable")

        def persist_queue_snapshot(self, reason=""):
            pass

    ev_module._handle_schedule_task(
        {
            "type": "schedule_subagent",
            "task_id": "ghost1",
            "objective": "Work with no workers",
            "expected_output": "Nothing",
            "depth": 1,
            "root_task_id": "rootX",
            "delegation_role": "subagent",
            "memory_mode": "forked",
            "drive_root": str(tmp_path / "state" / "headless_tasks" / "ghost1" / "data"),
            "child_drive_root": str(tmp_path / "state" / "headless_tasks" / "ghost1" / "data"),
        },
        FakeCtx(),
    )

    data = json.loads((tmp_path / "task_results" / "ghost1.json").read_text(encoding="utf-8"))
    assert data["status"] == STATUS_FAILED
    assert data.get("reason_code") == "workers_unavailable"


def test_handle_task_done_skips_workspace_readonly_subagent_artifacts(tmp_path, monkeypatch):
    from supervisor import events as ev_module
    import ouroboros.headless as headless
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result

    calls = []

    def fake_copy(root, task):
        calls.append(("copy", task["id"]))
        return write_task_result(pathlib.Path(root), task["id"], STATUS_COMPLETED, result="child handoff")

    monkeypatch.setattr(headless, "copy_child_task_result", fake_copy)

    def fake_finalize(root, task):
        calls.append(("finalize", task["id"]))
        write_task_result(
            pathlib.Path(root),
            task["id"],
            STATUS_COMPLETED,
            result="done",
            artifact_status="failed",
            artifact_bundle={"status": "failed", "artifacts": []},
        )

    monkeypatch.setattr(headless, "finalize_task_artifacts", fake_finalize)
    pushed = []

    worker = SimpleNamespace(busy_task_id="workspace-child")
    ctx = SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={
            "workspace-child": {
                "task": {
                    "id": "workspace-child",
                    "chat_id": 1,
                    "delegation_role": "subagent",
                    "role": "workspace-reviewer",
                    "root_task_id": "root123",
                    "parent_task_id": "parent123",
                    "workspace_root": str(tmp_path / "workspace"),
                    "task_constraint": {"mode": "local_readonly_subagent"},
                }
            }
        },
        WORKERS={3: worker},
        bridge=SimpleNamespace(push_log=lambda payload: pushed.append(payload)),
        send_with_budget=lambda *args, **kwargs: None,
        persist_queue_snapshot=lambda reason="": None,
    )

    ev_module._handle_task_done({"task_id": "workspace-child", "worker_id": 3, "task_type": "task"}, ctx)

    assert ("copy", "workspace-child") in calls
    assert ("finalize", "workspace-child") not in calls
    assert pushed[-1]["status"] == STATUS_COMPLETED
    assert pushed[-1]["artifact_status"] is None


def test_queue_snapshot_preserves_subagent_contract_fields(tmp_path, monkeypatch):
    from ouroboros.agent_startup_checks import validate_task_authority_sources
    from ouroboros.task_results import write_task_result
    from supervisor import queue as queue_module

    write_task_result(tmp_path, "previous", "completed", result="Previous exact result")
    snapshot_path = tmp_path / "state" / "queue_snapshot.json"
    monkeypatch.setattr(queue_module, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue_module, "QUEUE_SNAPSHOT_PATH", snapshot_path)
    monkeypatch.setattr(queue_module, "PENDING", [])
    monkeypatch.setattr(queue_module, "RUNNING", {})
    monkeypatch.setattr(queue_module, "QUEUE_SEQ_COUNTER_REF", {"value": 0})
    monkeypatch.setattr(queue_module, "append_jsonl", lambda *args, **kwargs: None)

    queue_module.PENDING.append(
        {
            "id": "sub1",
            "type": "task",
            "chat_id": 1,
            "text": "subagent prompt",
            "description": "Review shared surface",
            "objective": "Review shared surface",
            "expected_output": "Distinct handoff table",
            "constraints": "No writes",
            "role": "security reviewer",
            "context": "same context",
            "parent_task_id": "parent1",
            "root_task_id": "root1",
            "session_id": "sess1",
            "actor_id": "subagent:security",
            "delegation_role": "subagent",
            "memory_mode": "forked",
            "allowed_resources": {"web": False, "network": False},
            "deadline_at": "2026-06-04T12:00:00Z",
            "task_contract": {
                "schema_version": 1,
                "objective": "Review shared surface",
                "allowed_resources": {"web": False, "network": False},
                "resource_policy": {
                    "protected_artifacts": [
                        {
                            "id": "reference",
                            "role": "black_box_reference",
                            "paths": ["reference.bin"],
                            "allow": ["execute"],
                        }
                    ]
                },
                "deadline_at": "2026-06-04T12:00:00Z",
            },
            "child_drive_root": str(tmp_path / "state" / "headless_tasks" / "sub1" / "data"),
            "task_constraint": {"mode": "local_readonly_subagent", "allow_enable": False},
            "predecessor_authority_source": {
                "kind": "task_result", "task_id": "previous", "tool": "get_task_result",
                "arguments": {"task_id": "previous", "include_authority": True},
            },
        }
    )

    queue_module.persist_queue_snapshot(reason="test")
    saved = json.loads(snapshot_path.read_text(encoding="utf-8"))["pending"][0]["task"]
    assert saved["objective"] == "Review shared surface"
    assert saved["expected_output"] == "Distinct handoff table"
    assert saved["constraints"] == "No writes"
    assert saved["role"] == "security reviewer"
    assert saved["allowed_resources"] == {"web": False, "network": False}
    assert saved["deadline_at"] == "2026-06-04T12:00:00Z"
    assert saved["task_contract"]["allowed_resources"] == {"web": False, "network": False}
    assert saved["task_contract"]["resource_policy"]["protected_artifacts"][0]["id"] == "reference"
    assert pathlib.Path(saved["child_drive_root"]).parts[-4:] == ("state", "headless_tasks", "sub1", "data")
    assert saved["task_constraint"]["mode"] == "local_readonly_subagent"
    assert saved["predecessor_authority_source"]["task_id"] == "previous"

    queue_module.PENDING.clear()
    assert queue_module.restore_pending_from_snapshot(max_age_sec=900) == 1
    restored = queue_module.PENDING[0]
    assert restored["objective"] == "Review shared surface"
    assert restored["expected_output"] == "Distinct handoff table"
    assert restored["constraints"] == "No writes"
    assert restored["role"] == "security reviewer"
    assert restored["allowed_resources"] == {"web": False, "network": False}
    assert restored["deadline_at"] == "2026-06-04T12:00:00Z"
    assert restored["task_contract"]["allowed_resources"] == {"web": False, "network": False}
    assert restored["task_contract"]["resource_policy"]["protected_artifacts"][0]["paths"] == ["reference.bin"]
    assert pathlib.Path(restored["child_drive_root"]).parts[-4:] == ("state", "headless_tasks", "sub1", "data")
    assert restored["task_constraint"]["mode"] == "local_readonly_subagent"
    assert restored["predecessor_authority_source"]["task_id"] == "previous"
    assert validate_task_authority_sources(
        SimpleNamespace(drive_root=tmp_path, budget_drive_root=tmp_path), restored,
    ) == {}
    assert restored["predecessor_authority"]["source"] == restored["predecessor_authority_source"]
    assert restored["predecessor_authority"]["result"] == "Previous exact result"


def test_assign_tasks_mirrors_running_subagent_status_to_parent_drive(tmp_path, monkeypatch):
    from ouroboros.task_results import STATUS_RUNNING, load_task_result
    from supervisor import queue as queue_module
    from supervisor import state as state_module
    from supervisor import workers as workers_module

    child_drive = tmp_path / "state" / "headless_tasks" / "childrun" / "data"
    child_drive.mkdir(parents=True)
    delivered = []

    class FakeWorkerQueue:
        def put(self, task):
            delivered.append(dict(task))

    task = {
        "id": "childrun",
        "type": "task",
        "chat_id": 1,
        "description": "Inspect handoff",
        "objective": "Inspect handoff",
        "expected_output": "Findings",
        "parent_task_id": "parent123",
        "root_task_id": "root123",
        "session_id": "sess123",
        "actor_id": "subagent:reviewer",
        "delegation_role": "subagent",
        "role": "reviewer",
        "memory_mode": "forked",
        "drive_root": str(child_drive),
        "child_drive_root": str(child_drive),
        "budget_drive_root": str(tmp_path),
        "task_constraint": {"mode": "local_readonly_subagent", "allow_enable": False},
        "metadata": {"root_task_id": "root123"},
    }
    monkeypatch.setattr(workers_module, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers_module, "PENDING", [task])
    monkeypatch.setattr(workers_module, "RUNNING", {})
    monkeypatch.setattr(workers_module, "WORKERS", {1: SimpleNamespace(wid=1, busy_task_id=None, in_q=FakeWorkerQueue())})
    monkeypatch.setattr(workers_module, "load_state", lambda: {})
    monkeypatch.setattr(state_module, "budget_remaining", lambda _state, **_kwargs: 100.0)
    monkeypatch.setattr(queue_module, "persist_queue_snapshot", lambda reason="": None)

    workers_module.assign_tasks()

    parent_result = load_task_result(tmp_path, "childrun")
    assert parent_result["status"] == STATUS_RUNNING
    assert parent_result["child_drive_root"] == str(child_drive)
    assert parent_result["result"] == "Subagent assigned to a worker."
    assert delivered and delivered[0]["id"] == "childrun"


def test_assign_tasks_leaves_subagent_pending_when_running_cap_full(tmp_path, monkeypatch):
    from supervisor import queue as queue_module
    from supervisor import workers as workers_module
    from supervisor import state as state_module

    delivered = []

    class FakeWorkerQueue:
        def put(self, task):
            delivered.append(task)

    pending = [{
        "id": "child2",
        "type": "task",
        "chat_id": 1,
        "description": "Wait",
        "root_task_id": "root123",
        "delegation_role": "subagent",
        "budget_drive_root": str(tmp_path),
    }]
    running = {
        "child1": {
            "task": {
                "id": "child1",
                "root_task_id": "root123",
                "delegation_role": "subagent",
            }
        }
    }
    monkeypatch.setenv("OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT", "1")
    monkeypatch.setattr(workers_module, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers_module, "PENDING", pending)
    monkeypatch.setattr(workers_module, "RUNNING", running)
    monkeypatch.setattr(workers_module, "WORKERS", {1: SimpleNamespace(wid=1, busy_task_id=None, in_q=FakeWorkerQueue())})
    monkeypatch.setattr(workers_module, "load_state", lambda: {})
    monkeypatch.setattr(state_module, "budget_remaining", lambda _state, **_kwargs: 100.0)
    monkeypatch.setattr(queue_module, "persist_queue_snapshot", lambda reason="": None)

    workers_module.assign_tasks()

    assert pending and pending[0]["id"] == "child2"
    assert delivered == []


def test_assign_tasks_honors_depth_reservation_for_first_grandchild(tmp_path, monkeypatch):
    from supervisor import queue as queue_module
    from supervisor import workers as workers_module
    from supervisor import state as state_module

    delivered = []

    class FakeWorkerQueue:
        def put(self, task):
            delivered.append(task)

    pending = [{
        "id": "grandchild1",
        "type": "task",
        "chat_id": 1,
        "description": "Reserved depth child",
        "root_task_id": "root123",
        "parent_task_id": "child1",
        "delegation_role": "subagent",
        "budget_drive_root": str(tmp_path),
    }]
    running = {
        "child1": {
            "task": {
                "id": "child1",
                "root_task_id": "root123",
                "delegation_role": "subagent",
            }
        }
    }
    monkeypatch.setenv("OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT", "1")
    monkeypatch.setattr(workers_module, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers_module, "PENDING", pending)
    monkeypatch.setattr(workers_module, "RUNNING", running)
    monkeypatch.setattr(workers_module, "WORKERS", {1: SimpleNamespace(wid=1, busy_task_id=None, in_q=FakeWorkerQueue())})
    monkeypatch.setattr(workers_module, "load_state", lambda: {})
    monkeypatch.setattr(state_module, "budget_remaining", lambda _state, **_kwargs: 100.0)
    monkeypatch.setattr(queue_module, "persist_queue_snapshot", lambda reason="": None)

    workers_module.assign_tasks()

    assert delivered and delivered[0]["id"] == "grandchild1"
    assert "grandchild1" in workers_module.RUNNING


def test_assignment_depth_fact_reaches_worker_and_survives_child_copyback(tmp_path, monkeypatch):
    from supervisor import queue as queue_module
    from supervisor import workers as workers_module
    from supervisor import state as state_module
    from ouroboros.contracts.task_contract import build_task_contract
    from ouroboros.headless import copy_child_task_result
    from ouroboros.task_results import (
        STATUS_COMPLETED,
        load_task_result,
        write_task_result,
    )

    delivered = []

    class FakeWorkerQueue:
        def put(self, task):
            delivered.append(task)

    child_drive = tmp_path / "state" / "headless_tasks" / "child-depth" / "data"
    child_drive.mkdir(parents=True)
    queued_contract = build_task_contract({
        "delegation_budget": {
            "depth_remaining": 2,
            "depth_provenance": {
                "requested_depth": 3,
                "permitted_depth": 3,
                "attempted_depth": 1,
                "achieved_depth": None,
            },
        },
    })
    pending = [{
        "id": "child-depth",
        "type": "task",
        "chat_id": 1,
        "description": "Assigned depth child",
        "depth": 1,
        "root_task_id": "root-depth",
        "parent_task_id": "root-depth",
        "delegation_role": "subagent",
        "drive_root": str(child_drive),
        "child_drive_root": str(child_drive),
        "budget_drive_root": str(tmp_path),
        "task_contract": queued_contract,
        "metadata": {"task_contract": queued_contract},
    }]
    monkeypatch.setattr(workers_module, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers_module, "PENDING", pending)
    monkeypatch.setattr(workers_module, "RUNNING", {})
    monkeypatch.setattr(
        workers_module,
        "WORKERS",
        {1: SimpleNamespace(wid=1, busy_task_id=None, in_q=FakeWorkerQueue())},
    )
    monkeypatch.setattr(workers_module, "load_state", lambda: {})
    monkeypatch.setattr(state_module, "budget_remaining", lambda _state, **_kwargs: 100.0)
    monkeypatch.setattr(queue_module, "persist_queue_snapshot", lambda reason="": None)

    workers_module.assign_tasks()

    assert delivered and delivered[0]["depth_provenance"]["achieved_depth"] == 1
    worker_contract = delivered[0]["task_contract"]
    assert worker_contract["delegation_budget"]["depth_provenance"]["achieved_depth"] == 1
    assert delivered[0]["metadata"]["task_contract"] == worker_contract

    # Reproduce the real terminal replica: the worker writes the contract it
    # received to its child drive, then the supervisor copies that result back.
    write_task_result(
        child_drive,
        "child-depth",
        STATUS_COMPLETED,
        parent_task_id="root-depth",
        root_task_id="root-depth",
        delegation_role="subagent",
        task_contract=worker_contract,
        depth_provenance=delivered[0]["depth_provenance"],
        result="done",
    )
    copied = copy_child_task_result(tmp_path, delivered[0])
    assert copied is not None
    canonical = load_task_result(tmp_path, "child-depth")
    nested = canonical["task_contract"]["delegation_budget"]["depth_provenance"]
    assert nested == canonical["depth_provenance"]
    assert nested["achieved_depth"] == 1


def test_override_delegation_constraint_requires_parent_lineage(tmp_path, monkeypatch):
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from ouroboros.tools.join_ledger import _override_delegation_constraint
    from ouroboros.tools.registry import ToolContext
    import ouroboros.task_tree_ledger as ledger

    monkeypatch.setattr(ledger, "DATA_DIR", str(tmp_path))
    write_task_result(tmp_path, "child1", STATUS_RUNNING, parent_task_id="parent1", root_task_id="root1", delegation_role="subagent")
    ledger.tree_ledger_append(
        "root1",
        "delegation_constraint",
        "child asks parent to stop fanout",
        task_id="child1",
        role="scout",
        payload={"constraint_id": "c1", "directive": "halt_fanout", "scope": {}, "rationale": "wait for evidence"},
    )
    sibling = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="sibling", task_metadata={"root_task_id": "root1"})
    child = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="child1", task_metadata={"root_task_id": "root1"})
    parent = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="parent1", task_metadata={"root_task_id": "root1"})

    assert "only the parent" in _override_delegation_constraint(child, "c1", "self-clear")
    assert "only the parent" in _override_delegation_constraint(sibling, "c1", "not my constraint")
    assert _override_delegation_constraint(parent, "c1", "I gathered the evidence").startswith("OK:")
    assert ledger.open_delegation_constraints("root1") == []


def test_subagent_hard_timeout_retry_preserves_task_id(tmp_path, monkeypatch):
    from supervisor import queue as queue_module
    from supervisor import workers as workers_module
    from ouroboros.task_results import STATUS_INTERRUPTED, load_task_result

    class FakeProc:
        pid = 12345

        def is_alive(self):
            return False

        def terminate(self):
            raise AssertionError("already dead")

        def join(self, timeout=None):
            return None

    monkeypatch.setattr(queue_module, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue_module, "PENDING", [])
    monkeypatch.setattr(queue_module, "RUNNING", {})
    monkeypatch.setattr(queue_module, "QUEUE_SEQ_COUNTER_REF", {"value": 0})
    monkeypatch.setattr(queue_module, "FINALIZATION_GRACE_SEC", 0)
    monkeypatch.setattr(queue_module, "QUEUE_MAX_RETRIES", 1)
    monkeypatch.setattr(queue_module, "load_state", lambda: {})
    monkeypatch.setattr(queue_module, "append_jsonl", lambda *args, **kwargs: None)
    monkeypatch.setattr(queue_module, "persist_queue_snapshot", lambda reason="": None)
    # Activity model: a "timed out" task is one with no real progress for the idle
    # window AND no progressing subtree (heartbeat alone is not progress). Variant A:
    # run the heavy teardown reaper synchronously (no daemon) for a deterministic test.
    monkeypatch.setattr(queue_module, "_ensure_reaper_started", lambda: None)
    monkeypatch.setattr(queue_module, "_reap_queue", queue_module._stdqueue.Queue())
    monkeypatch.setattr(queue_module, "get_task_idle_timeout_sec", lambda: 1)
    monkeypatch.setattr(queue_module, "get_per_call_timeout_ceiling_sec", lambda: 1)
    worker = SimpleNamespace(busy_task_id="childtimeout", proc=FakeProc(), reaping=False)
    monkeypatch.setattr(workers_module, "WORKERS", {9: worker})
    monkeypatch.setattr(workers_module, "respawn_worker", lambda worker_id: None)
    child_drive = tmp_path / "child-drive"
    service_dir = child_drive / "services" / "childtimeout"
    service_dir.mkdir(parents=True)
    (service_dir / "devserver.log").write_text("READY\n", encoding="utf-8")

    queue_module.RUNNING["childtimeout"] = {
        "task": {
            "id": "childtimeout",
            "type": "task",
            "chat_id": 1,
            "delegation_role": "subagent",
            "drive_root": str(child_drive),
            "child_drive_root": str(child_drive),
            "_attempt": 1,
        },
        # idle for ~1000s, far beyond the monkeypatched idle window max(1, 1+120)=121s,
        # with no progressing subtree -> activity-based stop.
        "started_at": time.time() - 1000,
        "last_heartbeat_at": time.time() - 1000,
        "worker_id": 9,
        "attempt": 1,
    }

    queue_module.enforce_task_timeouts()
    # Drain the off-loop reaper synchronously (kill/archive/respawn).
    while not queue_module._reap_queue.empty():
        queue_module._reap_timed_out_task(queue_module._reap_queue.get_nowait())

    assert queue_module.PENDING
    retried = queue_module.PENDING[0]
    assert retried["id"] == "childtimeout"
    assert retried["_attempt"] == 2
    assert retried["timeout_retry_from"] == "childtimeout"
    assert load_task_result(tmp_path, "childtimeout")["status"] == STATUS_INTERRUPTED
    assert "childtimeout" not in queue_module.RUNNING
    assert not service_dir.exists()


def test_absolute_deadline_does_not_retry_expired_task(tmp_path, monkeypatch):
    from supervisor import queue as queue_module
    from supervisor import workers as workers_module
    from ouroboros.task_results import STATUS_FAILED, load_task_result

    class FakeProc:
        pid = 12345

        def is_alive(self):
            return False

        def terminate(self):
            raise AssertionError("already dead")

        def join(self, timeout=None):
            return None

    monkeypatch.setattr(queue_module, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue_module, "PENDING", [])
    monkeypatch.setattr(queue_module, "RUNNING", {})
    monkeypatch.setattr(queue_module, "QUEUE_SEQ_COUNTER_REF", {"value": 0})
    monkeypatch.setattr(queue_module, "FINALIZATION_GRACE_SEC", 0)
    monkeypatch.setattr(queue_module, "QUEUE_MAX_RETRIES", 3)
    monkeypatch.setattr(queue_module, "load_state", lambda: {})
    monkeypatch.setattr(queue_module, "append_jsonl", lambda *args, **kwargs: None)
    monkeypatch.setattr(queue_module, "persist_queue_snapshot", lambda reason="": None)
    monkeypatch.setattr(queue_module, "_ensure_reaper_started", lambda: None)
    monkeypatch.setattr(queue_module, "_reap_queue", queue_module._stdqueue.Queue())
    monkeypatch.setattr(queue_module, "get_task_idle_timeout_sec", lambda: 1)
    monkeypatch.setattr(queue_module, "get_per_call_timeout_ceiling_sec", lambda: 1)
    worker = SimpleNamespace(busy_task_id="deadline1", proc=FakeProc(), reaping=False)
    monkeypatch.setattr(workers_module, "WORKERS", {9: worker})
    monkeypatch.setattr(workers_module, "respawn_worker", lambda worker_id: None)

    queue_module.RUNNING["deadline1"] = {
        "task": {
            "id": "deadline1",
            "type": "task",
            "chat_id": 1,
            "deadline_at": "2000-01-01T00:00:00Z",
            "_attempt": 1,
        },
        # Past deadline AND idle (no progress for ~1000s): the deadline is gated through
        # idle/subtree-liveness, so an expired-but-idle task is stopped without retry.
        "started_at": time.time() - 1000,
        "last_heartbeat_at": time.time() - 1000,
        "worker_id": 9,
        "attempt": 1,
    }

    queue_module.enforce_task_timeouts()
    # Variant A: the terminal write + retry decision now happen in the off-loop reaper.
    while not queue_module._reap_queue.empty():
        queue_module._reap_timed_out_task(queue_module._reap_queue.get_nowait())

    assert queue_module.PENDING == []
    result = load_task_result(tmp_path, "deadline1")
    assert result["status"] == STATUS_FAILED
    assert result["reason_code"] == "deadline"
    assert result["outcome_axes"]["execution"]["reason_code"] == "deadline"


def test_handle_text_response_keeps_full_reasoning_note():
    from ouroboros.loop import _handle_text_response

    content = "A" * 500
    llm_trace = {"reasoning_notes": [], "tool_calls": []}
    _, _, updated = _handle_text_response(content, llm_trace, {})

    assert updated["reasoning_notes"] == [content]


def test_request_restart_latches_reason_until_task_end(tmp_path, monkeypatch):
    from ouroboros.tools import control as control_module
    from ouroboros.tools import control_runtime

    from supervisor import evolution_lifecycle

    monkeypatch.setattr(control_runtime, "run_cmd", lambda *args, **kwargs: "value")
    written = {}
    # The marker write lives in the shared writer helper (W4-F3: one schema for
    # the tool and the supervisor), so the capture sits at its seam.
    monkeypatch.setattr(
        evolution_lifecycle,
        "atomic_write_json",
        lambda path, payload, **_kwargs: written.setdefault(str(path), payload),
    )

    class _Ctx:
        current_task_type = "task"
        last_push_succeeded = True
        pending_events = []
        pending_restart_reason = None
        repo_dir = tmp_path
        drive_root = tmp_path

        def drive_path(self, rel):
            return tmp_path / rel

    ctx = _Ctx()
    result = control_module._request_restart(ctx, "reload runtime")

    assert "Restart requested" in result
    assert ctx.pending_events == []
    assert ctx.pending_restart_reason == "reload runtime"
    assert written
