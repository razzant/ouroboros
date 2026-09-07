"""Project chat continuity: managed-activity projection, finalizing honesty,
and durable owner-message persistence through the REAL routing chain.

Pins the four server seams of the continuity contract:

1. ``gateway.state._chat_activities_snapshot_safe`` unites direct/ephemeral
   registry turns with ROOT managed queue tasks (``queued``/``working``/
   ``finalizing``), deriving ``finalizing`` from the durable
   ``root_phase_checkpoint``.
2. ``supervisor.events._handle_typing_start`` stamps ``kind="managed_task"``
   on typing from RUNNING queue ROOTS (subagents keep the kind-less legacy).
3. ``agent_task_pipeline.emit_task_results`` marks a root's early final answer
   with ``progress_meta.task_phase="finalizing"`` exactly while post-task
   synthesis is still owed.
4. ``gateway.history`` withholds ``task_terminal_status`` and stamps
   ``task_phase="finalizing"`` on the rows of a completed-but-open-checkpoint
   task; a record without a checkpoint keeps legacy terminal semantics.

Plus the zero-mock persistence chain: ``server._process_bridge_updates`` with
the REAL ``log_chat`` writer must yield a history response that carries the
owner row and its routing annotation.
"""

from __future__ import annotations

import asyncio
import json
import types
from types import SimpleNamespace

import pytest

from supervisor.active_activity import get_direct_activity_registry


@pytest.fixture(autouse=True)
def _clean_registry():
    registry = get_direct_activity_registry()
    registry.clear()
    yield
    registry.clear()


def _root_task(task_id: str, chat_id: int = 5, project_id: str = "proj-a") -> dict:
    return {
        "id": task_id,
        "type": "task",
        "chat_id": chat_id,
        "project_id": project_id,
        "root_task_id": task_id,
        "delegation_role": "root",
        "text": "do the thing",
    }


def _subagent_task(task_id: str, parent_id: str, chat_id: int = 5) -> dict:
    return {
        "id": task_id,
        "type": "task",
        "chat_id": chat_id,
        "root_task_id": parent_id,
        "parent_task_id": parent_id,
        "delegation_role": "subagent",
        "text": "child work",
    }


# ---------------------------------------------------------------------------
# Seam 1: /api/state combined activity snapshot
# ---------------------------------------------------------------------------


def test_chat_activities_snapshot_projects_queue_roots_with_phases(tmp_path, monkeypatch):
    import supervisor.queue as queue_mod
    from ouroboros.gateway import state as gateway_state
    from ouroboros.task_results import STATUS_COMPLETED, STATUS_RUNNING, write_task_result

    monkeypatch.setattr(queue_mod, "PENDING", [
        {**_root_task("queued-root"), "queued_at": "2026-08-17T10:00:00+00:00"},
        _subagent_task("queued-child", "queued-root"),
    ])
    monkeypatch.setattr(queue_mod, "RUNNING", {
        "running-root": {"task": _root_task("running-root"), "started_at": 111.0},
        "finalizing-root": {"task": _root_task("finalizing-root"), "started_at": 222.0},
        "running-child": {"task": _subagent_task("running-child", "running-root"), "started_at": 5.0},
    })
    gateway_state._FINALIZING_MEMO.clear()
    write_task_result(tmp_path, "running-root", STATUS_RUNNING, result="working")
    write_task_result(
        tmp_path, "finalizing-root", STATUS_COMPLETED,
        result="answer stored",
        root_phase_checkpoint={"post_task_synthesis": "running"},
    )
    get_direct_activity_registry().register(
        "direct-1", chat_id=5, kind="direct_chat", client_message_id="cmid-1",
    )

    rows = {row["activity_id"]: row for row in gateway_state._chat_activities_snapshot_safe(tmp_path)}

    assert rows["direct-1"]["kind"] == "direct_chat"
    queued = rows["queued-root"]
    # queued_at is an ISO string on queue rows; the wire carries real epoch.
    started_at = queued.pop("started_at")
    assert started_at > 1_700_000_000
    assert queued == {
        "activity_id": "queued-root",
        "chat_id": 5,
        "project_id": "proj-a",
        "client_message_id": "",
        "kind": "managed_task",
        "phase": "queued",
    }
    assert rows["running-root"]["phase"] == "working"
    assert rows["running-root"]["started_at"] == 111.0
    assert rows["finalizing-root"]["phase"] == "finalizing"
    # Subagents are never snapshot activities: no snapshot source may gain
    # deletion authority over their kind-less client entries.
    assert "queued-child" not in rows
    assert "running-child" not in rows


def test_snapshot_rehomes_converted_task_through_binding(tmp_path, monkeypatch):
    """A mid-run "turn into project" conversion binds the task to the project
    without touching its queue row; the snapshot re-homes chat/project ids
    through the same task-binding projection /api/state already serves."""
    import supervisor.queue as queue_mod
    from ouroboros.gateway import state as gateway_state

    monkeypatch.setattr(queue_mod, "PENDING", [])
    monkeypatch.setattr(queue_mod, "RUNNING", {
        "converted-root": {"task": _root_task("converted-root", chat_id=1, project_id=""), "started_at": 9.0},
    })
    gateway_state._FINALIZING_MEMO.clear()

    rows = {
        row["activity_id"]: row
        for row in gateway_state._chat_activities_snapshot_safe(
            tmp_path,
            {"converted-root": {"project_id": "task-converted", "chat_id": 77}},
        )
    }

    assert rows["converted-root"]["chat_id"] == 77
    assert rows["converted-root"]["project_id"] == "task-converted"


def test_finalizing_probe_follows_checkpoint_lifecycle(tmp_path):
    from ouroboros.gateway import state as gateway_state
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result

    gateway_state._FINALIZING_MEMO.clear()
    assert gateway_state._managed_task_finalizing(tmp_path, "absent") is False

    write_task_result(
        tmp_path, "root-x", STATUS_COMPLETED,
        root_phase_checkpoint={"post_task_synthesis": "pending_once"},
    )
    assert gateway_state._managed_task_finalizing(tmp_path, "root-x") is True

    write_task_result(
        tmp_path, "root-x", STATUS_COMPLETED,
        root_phase_checkpoint={"post_task_synthesis": "completed"},
        result="synthesis landed, checkpoint closed",
    )
    assert gateway_state._managed_task_finalizing(tmp_path, "root-x") is False


def test_api_state_emits_combined_activities_field(tmp_path, monkeypatch):
    import supervisor.queue as queue_mod
    from ouroboros.gateway.state import api_state
    from supervisor import queue, state, workers
    from starlette.requests import Request

    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(state, "TOTAL_BUDGET_LIMIT", 0.0)
    monkeypatch.setattr(state, "load_state", lambda: {"current_branch": "ouroboros"})
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(workers, "PENDING", [])
    monkeypatch.setattr(workers, "RUNNING", {})
    monkeypatch.setattr(queue, "get_evolution_status_snapshot", lambda **_kwargs: {})
    monkeypatch.setattr(queue_mod, "PENDING", [_root_task("q-root", chat_id=3)])
    monkeypatch.setattr(queue_mod, "RUNNING", {})
    request = Request({
        "type": "http", "method": "GET", "path": "/api/state", "headers": [],
        "query_string": b"", "scheme": "http", "server": ("test", 80),
        "client": ("test", 1),
        "app": types.SimpleNamespace(state=types.SimpleNamespace(drive_root=tmp_path, app_start=0.0)),
    })

    payload = json.loads(asyncio.run(api_state(request)).body)

    assert payload["active_direct_turns"] == []
    managed = [row for row in payload["active_chat_activities"] if row["kind"] == "managed_task"]
    assert [row["activity_id"] for row in managed] == ["q-root"]
    assert managed[0]["phase"] == "queued"
    assert managed[0]["chat_id"] == 3


def test_active_chat_activity_contract_mirrors_direct_turn_shape():
    from ouroboros.gateway.contracts import ActiveChatActivity, ActiveDirectTurn, StateResponse
    import pathlib

    assert ActiveChatActivity.__annotations__ == ActiveDirectTurn.__annotations__
    assert "active_chat_activities" in StateResponse.__annotations__
    api_types = (
        pathlib.Path(__file__).resolve().parents[1] / "web" / "modules" / "api_types.js"
    ).read_text(encoding="utf-8")
    for needle in ("ActiveChatActivity", "active_chat_activities", "task_phase"):
        assert needle in api_types, f"api_types.js mirror is missing {needle!r}"


# ---------------------------------------------------------------------------
# Seam 2: typing frames from RUNNING queue roots carry the managed kind
# ---------------------------------------------------------------------------


class _BridgeProbe:
    def __init__(self):
        self.calls = []

    def send_chat_action(self, chat_id, action="typing", **kwargs):
        self.calls.append({"chat_id": chat_id, "action": action, **kwargs})
        return True


def test_typing_start_stamps_managed_kind_for_running_root():
    from supervisor.events import _handle_typing_start

    ctx = SimpleNamespace(
        bridge=_BridgeProbe(),
        RUNNING={"root-1": {"task": _root_task("root-1", chat_id=4)}},
    )
    _handle_typing_start(
        {"type": "typing_start", "chat_id": 4, "task_id": "root-1", "phase": "thinking"},
        ctx,
    )

    assert ctx.bridge.calls[0]["kind"] == "managed_task"
    assert ctx.bridge.calls[0]["activity_id"] == "root-1"


def test_typing_start_keeps_subagent_and_unknown_tasks_kindless():
    from supervisor.events import _handle_typing_start

    ctx = SimpleNamespace(
        bridge=_BridgeProbe(),
        RUNNING={"child-1": {"task": _subagent_task("child-1", "root-1", chat_id=4)}},
    )
    _handle_typing_start(
        {"type": "typing_start", "chat_id": 4, "task_id": "child-1", "phase": "thinking"},
        ctx,
    )
    _handle_typing_start(
        {"type": "typing_start", "chat_id": 4, "task_id": "not-in-queue", "phase": "thinking"},
        ctx,
    )

    assert [call["kind"] for call in ctx.bridge.calls] == ["", ""]


def test_typing_start_registry_kind_outranks_queue_stamp():
    from supervisor.events import _handle_typing_start

    get_direct_activity_registry().register(
        "turn-1", chat_id=4, kind="ephemeral_decision", client_message_id="cmid-9",
    )
    ctx = SimpleNamespace(
        bridge=_BridgeProbe(),
        RUNNING={"turn-1": {"task": _root_task("turn-1", chat_id=4)}},
    )
    _handle_typing_start(
        {"type": "typing_start", "chat_id": 4, "task_id": "turn-1", "phase": "thinking"},
        ctx,
    )

    assert ctx.bridge.calls[0]["kind"] == "ephemeral_decision"


# ---------------------------------------------------------------------------
# Seam 3: the early final answer carries the typed finalizing marker
# ---------------------------------------------------------------------------


def _emit_final(tmp_path, monkeypatch, task):
    import ouroboros.agent_task_pipeline as pipeline

    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **k: None)
    pending_events = []
    drive_logs = tmp_path / "logs"
    drive_logs.mkdir(parents=True, exist_ok=True)
    pipeline.emit_task_results(
        env=SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path),
        memory=object(),
        llm=object(),
        pending_events=pending_events,
        task=task,
        text="the answer",
        usage={"rounds": 1, "cost": 0.0},
        llm_trace={"tool_calls": [], "reasoning_notes": []},
        start_time=0.0,
        drive_logs=drive_logs,
        ctx=SimpleNamespace(pending_restart_reason=""),
    )
    return pending_events


def test_root_final_send_carries_finalizing_phase_marker(tmp_path, monkeypatch):
    events = _emit_final(tmp_path, monkeypatch, {
        "id": "root-final", "root_task_id": "root-final",
        "type": "task", "chat_id": 1, "text": "verify",
    })

    send = next(evt for evt in events if evt["type"] == "send_message")
    assert send["progress_meta"]["task_phase"] == "finalizing"
    done = next(evt for evt in events if evt["type"] == "task_done")
    # The final message precedes task_done; task_done itself is unmarked.
    assert events.index(send) < events.index(done)
    assert "task_phase" not in done


def test_completed_checkpoint_suppresses_finalizing_marker(tmp_path, monkeypatch):
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result

    write_task_result(
        tmp_path, "root-done", STATUS_COMPLETED,
        root_phase_checkpoint={"post_task_synthesis": "completed"},
    )
    events = _emit_final(tmp_path, monkeypatch, {
        "id": "root-done", "root_task_id": "root-done",
        "type": "task", "chat_id": 1, "text": "verify",
    })

    send = next(evt for evt in events if evt["type"] == "send_message")
    assert "task_phase" not in (send.get("progress_meta") or {})


def test_ephemeral_final_keeps_decision_meta_without_phase_marker(tmp_path, monkeypatch):
    events = _emit_final(tmp_path, monkeypatch, {
        "id": "eph-1", "type": "task", "chat_id": 1, "text": "2+2?",
        "_is_direct_chat": True, "_ephemeral_turn": True,
    })

    send = next(evt for evt in events if evt["type"] == "send_message")
    # The ephemeral final carries its own conclusion and outcome, without
    # a post-task finalizing hold or managed-task authority.
    assert send["progress_meta"]["ephemeral_decision"] is True
    assert send["progress_meta"]["task_terminal_status"] == "completed"
    done = next(evt for evt in events if evt["type"] == "task_done")
    assert send["progress_meta"]["outcome_axes"] == done["outcome_axes"]
    assert "task_phase" not in send["progress_meta"]


# ---------------------------------------------------------------------------
# Seam 4: history withholds terminal truth while the checkpoint is open
# ---------------------------------------------------------------------------


def _write_history_rows(tmp_path, task_id):
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "progress.jsonl").write_text(
        json.dumps({
            "ts": "2026-08-17T10:00:00+00:00", "type": "send_message",
            "task_id": task_id, "is_progress": True, "direction": "out",
            "chat_id": 1, "user_id": 1, "text": "step one", "content": "step one",
        }) + "\n",
        encoding="utf-8",
    )
    (logs / "chat.jsonl").write_text(
        json.dumps({
            "ts": "2026-08-17T10:00:05+00:00", "direction": "out",
            "chat_id": 1, "user_id": 1, "text": "final answer",
            "task_id": task_id, "format": "markdown",
        }) + "\n",
        encoding="utf-8",
    )


def _history_messages(tmp_path):
    from ouroboros.gateway.history import make_chat_history_endpoint

    endpoint = make_chat_history_endpoint(tmp_path)
    response = asyncio.run(endpoint(SimpleNamespace(query_params={"limit": "20"})))
    return json.loads(response.body.decode("utf-8"))["messages"]


def test_history_holds_terminal_truth_while_checkpoint_open(tmp_path):
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result

    _write_history_rows(tmp_path, "fin-task")
    write_task_result(
        tmp_path, "fin-task", STATUS_COMPLETED,
        result="stored",
        root_phase_checkpoint={"post_task_synthesis": "running"},
    )

    messages = _history_messages(tmp_path)
    rows = [m for m in messages if m.get("task_id") == "fin-task"]
    assert rows, "expected fin-task rows in the history window"
    assert all(m.get("task_phase") == "finalizing" for m in rows)
    assert all("task_terminal_status" not in m for m in rows)


def test_history_resolves_terminal_truth_once_checkpoint_closes(tmp_path):
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result

    _write_history_rows(tmp_path, "fin-task")
    write_task_result(
        tmp_path, "fin-task", STATUS_COMPLETED,
        result="stored",
        root_phase_checkpoint={"post_task_synthesis": "completed"},
        cost_usd=0.25, cost_final=True,
    )

    messages = _history_messages(tmp_path)
    progress = next(m for m in messages if m.get("is_progress") and m.get("task_id") == "fin-task")
    assert progress["task_terminal_status"] == "completed"
    assert progress.get("task_phase") != "finalizing"


def test_history_without_checkpoint_keeps_legacy_terminal_semantics(tmp_path):
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result

    _write_history_rows(tmp_path, "fin-task")
    write_task_result(tmp_path, "fin-task", STATUS_COMPLETED, result="stored")

    messages = _history_messages(tmp_path)
    progress = next(m for m in messages if m.get("is_progress") and m.get("task_id") == "fin-task")
    assert progress["task_terminal_status"] == "completed"


def test_history_split_drive_running_with_open_checkpoint_is_finalizing(tmp_path):
    """A split-drive project root keeps canonical status scheduled/running
    until copy-back; the open checkpoint alone proves finalization began."""
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    _write_history_rows(tmp_path, "fin-task")
    write_task_result(
        tmp_path, "fin-task", STATUS_RUNNING,
        result="running on child drive",
        root_phase_checkpoint={"post_task_synthesis": "running"},
    )

    messages = _history_messages(tmp_path)
    rows = [m for m in messages if m.get("task_id") == "fin-task"]
    assert rows and all(m.get("task_phase") == "finalizing" for m in rows)
    assert all("task_terminal_status" not in m for m in rows)


def test_history_failed_with_open_checkpoint_stays_terminal(tmp_path):
    """Failure honesty outranks the finalizing hold: a failed record replays
    terminal immediately even while its post-task checkpoint is open."""
    from ouroboros.task_results import STATUS_FAILED, write_task_result

    _write_history_rows(tmp_path, "fin-task")
    write_task_result(
        tmp_path, "fin-task", STATUS_FAILED,
        result="boom",
        root_phase_checkpoint={"post_task_synthesis": "running"},
    )

    messages = _history_messages(tmp_path)
    progress = next(m for m in messages if m.get("is_progress") and m.get("task_id") == "fin-task")
    assert progress["task_terminal_status"] == "failed"
    assert progress.get("task_phase") != "finalizing"


# ---------------------------------------------------------------------------
# Zero-mock persistence chain: bridge intake -> real log_chat -> history
# ---------------------------------------------------------------------------


def test_owner_project_message_survives_into_history_with_annotation(tmp_path, monkeypatch):
    import server
    import supervisor.message_bus as message_bus
    from ouroboros.projects_registry import create_project

    project = create_project(tmp_path, "racer")
    chat_id = int(project["chat_id"])
    pending = [{
        "id": "pending-root",
        "chat_id": chat_id,
        "root_task_id": "pending-root",
        "delegation_role": "root",
        "drive_root": str(tmp_path),
    }]
    acks = []

    class Bridge:
        def get_updates(self, offset=0, timeout=1):
            return [{
                "update_id": 1,
                "message": {
                    "chat": {"id": chat_id},
                    "from": {"id": 1},
                    "text": "please also fix the failing test",
                    "source": "web",
                    "client_message_id": "owner-msg-1",
                },
            }]

        def send_routing_ack(self, *args, **kwargs):
            acks.append((args, kwargs))

        def broadcast(self, _payload):
            return None

    class _Consciousness:
        def inject_observation(self, _text):
            return None

    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        PENDING=pending,
        RUNNING={},
        load_state=lambda: {"owner_id": 1, "owner_chat_id": 1, "session_id": "s-1"},
        update_state=lambda fn: fn({"owner_id": 1, "owner_chat_id": 1}),
        consciousness=_Consciousness(),
        get_chat_agent=lambda: types.SimpleNamespace(_busy=False),
        handle_chat_ephemeral=lambda *a, **k: pytest.fail("mailbox delivery must not run a turn"),
        handle_chat_direct=lambda *a, **k: pytest.fail("mailbox delivery must not run a turn"),
        send_with_budget=lambda *a, **k: pytest.fail("routing receipts must not create bubbles"),
    )
    # REAL persistence: log_chat writes the canonical row under this data root.
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"session_id": "s-1"})

    server._process_bridge_updates(Bridge(), 0, ctx)

    assert acks and acks[-1][1]["status"] == "delivered"
    from ouroboros.gateway.history import make_chat_history_endpoint

    endpoint = make_chat_history_endpoint(tmp_path)
    response = asyncio.run(endpoint(SimpleNamespace(
        query_params={"chat_id": str(chat_id), "limit": "20"},
    )))
    messages = json.loads(response.body.decode("utf-8"))["messages"]
    owner_rows = [m for m in messages if m.get("client_message_id") == "owner-msg-1"]
    assert owner_rows, "the owner row must survive into durable history"
    row = owner_rows[0]
    assert row["role"] == "user"
    assert row["text"] == "please also fix the failing test"
    assert row["chat_annotation"] == {
        "action": "mailbox_delivery",
        "target": "pending-root",
        "target_label": "Task",
        "status": "delivered",
        "attachment_manifest": [],
    }
    assert acks[-1][1]["target_label"] == row["chat_annotation"]["target_label"]
