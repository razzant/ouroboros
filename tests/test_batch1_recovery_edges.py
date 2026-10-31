"""Public consumer regressions for the repaired recovery candidate."""
import copy
import json
import pathlib
import threading
import time

import pytest

from tests.test_schedule_occurrence import _row, _rows, q  # noqa: F401
from tests.test_tool_call_log import _call, _Registry
from tests.test_tool_call_log import _rows as tool_rows


def test_configured_readonly_session_receives_own_scope_and_real_lineage_inputs(tmp_path, monkeypatch):
    from ouroboros import claudexor_daemon, safety
    from ouroboros.tools.registry import ToolRegistry
    from tests.test_delegated_skill_payload import _payload_ctx, _StartStub

    ctx = _payload_ctx(tmp_path, monkeypatch)
    from ouroboros.task_results import write_task_result
    write_task_result(ctx.drive_root, ctx.task_id, "running", root_task_id=ctx.task_id)
    ctx.task_metadata.update(resource_intent={"kind": "explicit_none"}, parent_task_id="parent")
    parent = tmp_path / "data/task_drives/parent"
    parent.mkdir(parents=True)
    (parent / "notes.txt").write_text("the permitted parent input", encoding="utf-8")
    sibling = tmp_path / "data/task_drives/sibling"
    sibling.mkdir()
    (sibling / "secret.txt").write_text("not in this lineage", encoding="utf-8")
    scopes = []

    class Gateway(_StartStub):
        def start_run(self, request, **kwargs):
            scope = pathlib.Path(request["scope"]["root"])
            scopes.append(scope)
            assert scope != ctx.active_repo_dir() and scope != parent
            assert request["access"] == "readonly" and not request.get("execution")
            inputs = json.loads((scope / "inputs.json").read_text())
            [note] = [row for row in inputs if row["source"] == str(parent / "notes.txt")]
            assert (scope / note["local"]).read_text() == "the permitted parent input"
            assert not any("sibling" in row["source"] for row in inputs)
            return {"runId": f"run-readonly-{len(scopes)}", "runDir": str(tmp_path / "engine-run")}

    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda: Gateway({}))
    monkeypatch.setattr(safety, "check_safety", lambda *_a, **_k: (True, ""))
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    result = registry.execute("delegate_start", {
        "subagent_id": "payload-session", "access": "readonly", "prompt": "Read the parent notes.",
    })
    assert json.loads(result).get("run_id") == "run-readonly-1", result
    assert len(scopes) == 1


def test_readonly_input_scan_failure_and_symlink_refuse_before_engine(tmp_path, monkeypatch):
    from ouroboros.delegate_readonly_inputs import prepare_folderless_inputs
    from tests.test_delegated_skill_payload import _payload_ctx

    ctx = _payload_ctx(tmp_path, monkeypatch)
    ctx.task_metadata["resource_intent"] = {"kind": "explicit_none"}
    scratch = ctx.active_repo_dir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (scratch / "delegated_readonly_inputs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        prepare_folderless_inputs(ctx, "invocation")
    assert list(outside.iterdir()) == []


def test_restored_unstarted_claim_recovers_and_delete_preserves_accepted_work(q, monkeypatch):  # noqa: F811
    from supervisor import schedule_occurrence as occurrence

    _row(q, intent={"kind": "system_repo"})
    real_prepare = occurrence.prepare
    monkeypatch.setattr(occurrence, "prepare", lambda _: (_ for _ in ()).throw(RuntimeError("crash")))
    with pytest.raises(RuntimeError):
        q.queue.check_scheduled_tasks()
    claimed = copy.deepcopy(_rows(q)["s1"]["occurrence"])
    monkeypatch.setattr(occurrence, "prepare", real_prepare)
    q.queue.check_scheduled_tasks()
    assert [task["id"] for task in q.pending] == [claimed["task_id"]]
    outcome = q.queue.mutate_scheduled_task("delete", "s1", reason="owner", actor="owner")
    assert outcome["status"] == "withdrawn"
    assert _rows(q)["s1"]["occurrence"]["token"] == claimed["token"]


def test_late_owner_hold_survives_republish_and_cannot_dispatch(q):  # noqa: F811
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor import schedule_occurrence as occurrence

    _row(q, intent={"kind": "system_repo"})
    q.queue.check_scheduled_tasks()
    [task] = q.pending
    token = _rows(q)["s1"]["occurrence"]["token"]
    hold = {"source": "owner", "revision": "test-1"}
    write_task_result(q.root, task["id"], "scheduled", _owner_hold=hold)
    assert occurrence.record_dispatch_possible(task) is False
    assert task["_owner_hold"] == hold
    q.pending.clear()
    q.queue.check_scheduled_tasks()
    [restored] = q.pending
    assert restored["id"] == task["id"] and restored["_owner_hold"] == hold
    assert restored["metadata"]["schedule_occurrence"]["token"] == token
    assert load_task_result(q.root, task["id"])["schedule_admission"]["dispatch"] == "none"


def test_full_dispatch_source_is_canonical_and_available_before_execution(tmp_path):
    from ouroboros.observability import read_call_payload
    from ouroboros.tools.tool_result import ToolResult

    arguments = {"command": "x" * 12000, "nested": {"late": list(range(300))}}

    def handler(_tool, _args):
        [start] = tool_rows(tmp_path / "canonical/logs/tools.jsonl")
        assert start["args_source_status"] == "ready"
        _, payload, _ = read_call_payload(tmp_path / "canonical", task_id="task-1",
                                         call_id=f"tool_dispatch_{start['invocation_id']}")
        assert payload["arguments"] == arguments
        return ToolResult(status="ok", code="OK", text="done")

    _call(_Registry(tmp_path, handler), tmp_path, args=arguments)


def test_wait_end_and_late_accounting_failure_settle_once(tmp_path):
    from ouroboros.usage_accounting import UsageAccountingError

    release = threading.Event()

    def handler(*_):
        release.wait(5)
        raise UsageAccountingError("late accounting failure")

    registry = _Registry(tmp_path, handler)
    _, logs = _call(registry, tmp_path, timeout=0.05)
    release.set()
    deadline = time.monotonic() + 3
    # The settlement callback appends the durable row, THEN publishes its live frame: await both.
    while time.monotonic() < deadline and (len(tool_rows(logs / "tools.jsonl")) < 3
                                           or not any(frame.get("type") == "tool_call" for frame in registry.frames)):
        time.sleep(0.01)
    rows = tool_rows(logs / "tools.jsonl")
    assert [row["type"] for row in rows] == ["tool_call_started", "tool_call_timeout", "tool_call"]
    assert rows[-1]["status"] == "host_error"
    assert len({row["invocation_id"] for row in rows}) == 1
    assert any(frame.get("type") == "tool_call" for frame in registry.frames)


def test_continuation_receipt_snapshot_and_republish_keep_cap_exemption(q, monkeypatch):  # noqa: F811
    from ouroboros import consciousness_allowance
    from ouroboros.task_results import load_task_result
    from supervisor import queue_snapshot

    window = {"status": "available", "limit_usd": 10.0, "accounted_usd": 0.0, "unknown_unmetered": 0, "resets_at": ""}
    monkeypatch.setattr(consciousness_allowance, "allowance_window", lambda _: dict(window))
    _row(q, intent={"kind": "system_repo"}, metadata={"initiator": "consciousness"},
         continuation_of={"task_id": "earlier"})
    q.queue.check_scheduled_tasks()
    [task] = q.pending
    assert task["_consciousness_continuation"] is True
    assert load_task_result(q.root, task["id"])["schedule_admission"]["task"]["_consciousness_continuation"] is True
    assert queue_snapshot.persist_queue_snapshot()
    q.pending.clear()
    queue_snapshot.restore_pending_from_snapshot()
    [restored] = q.pending
    assert restored["id"] == task["id"] and restored["_consciousness_continuation"] is True
    assert q.queue.live_consciousness_root_count() == 0
    q.pending.clear()
    window.update(status="exhausted", accounted_usd=10.0)
    q.queue.check_scheduled_tasks()
    assert not q.pending and _rows(q)["s1"]["hold"]["reason"] == "consciousness_allowance_exhausted"


def test_unknown_destination_holds_but_positive_hidden_destination_is_kept(q):  # noqa: F811
    from supervisor import state

    state.update_state(lambda live: state.mark_unconfirmed(live, "owner_chat_id"))
    _row(q, intent={"kind": "system_repo"}, chat_id=None)
    _row(q, "hidden", intent={"kind": "system_repo"}, chat_id=0)
    q.queue.check_scheduled_tasks()
    assert _rows(q)["s1"]["hold"]["reason"] == "owner_chat_unknown"
    [hidden] = q.pending
    assert hidden["chat_id"] == 0
    state.update_state(lambda live: live.update(owner_chat_id=42), confirm=("owner_chat_id",))
    q.queue.check_scheduled_tasks()
    assert any(task["chat_id"] == 42 for task in q.pending)


def test_registry_rebind_between_prepare_and_admission_is_fenced(q, monkeypatch, tmp_path):  # noqa: F811
    from ouroboros.projects_registry import update_project
    from tests.test_schedule_occurrence import _project

    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    _project(q, folder=old)
    _row(q, intent={"kind": "room_default", "project_id": "proj"}, project_id="proj")
    enqueue = q.queue.enqueue_task
    moved = False

    def rebind(*args, **kwargs):
        nonlocal moved
        if not moved:
            update_project(q.root, "proj", working_dir=str(new))
            moved = True
        return enqueue(*args, **kwargs)

    monkeypatch.setattr(q.queue, "enqueue_task", rebind)
    q.queue.check_scheduled_tasks()
    assert not q.pending and _rows(q)["s1"]["hold"]["reason"] == "project_routing_fence_changed"
    q.queue.check_scheduled_tasks()
    [task] = q.pending
    assert task["workspace_root"] == str(new)


@pytest.mark.parametrize("bound", [False, True], ids=["unbound-logs-only", "bound-authority"])
def test_dispatch_source_and_append_failures_disclose_loss_without_veto(tmp_path, monkeypatch, bound):
    from ouroboros import budget_pause
    from ouroboros.tools.tool_result import ToolResult

    called = []
    registry = _Registry(tmp_path, lambda *_: called.append(True) or ToolResult(status="ok", code="OK", text="done"))
    resolve = pathlib.Path.resolve
    canonical = tmp_path / "canonical"
    def unavailable(path, *args, **kwargs):
        if path == canonical:
            raise OSError(5, "canonical unavailable")
        return resolve(path, *args, **kwargs)
    monkeypatch.setattr(pathlib.Path, "resolve", unavailable)
    if bound:
        registry._ctx.task_id = "task-1"
        with pytest.raises(OSError, match="canonical unavailable"):
            _call(registry, tmp_path)
        assert not called, "unreadable required task identity must still prevent submission"
        return
    result, logs = _call(registry, tmp_path)
    assert called and result["result"] == "done"
    [start, settled] = tool_rows(logs / "tools.jsonl")
    assert start["args_source_status"] == "unavailable:OSError"
    assert settled["start_log"]["canonical"] is False
    assert budget_pause.tool_scope_key(registry._ctx) == ""
    assert "" not in budget_pause._TOOL_FUTURES
    budget_pause.hold_tool_settlement(registry._ctx, start["invocation_id"])()
    budget_pause.forget_tool_scope(registry._ctx)
    assert budget_pause.drain_local_tool_futures(registry._ctx, timeout_sec=0)["drained"]


def test_zero_timeout_file_lock_attempts_once_without_wait(tmp_path, monkeypatch):
    from ouroboros import platform_layer

    path = tmp_path / "lock"
    fd = platform_layer.acquire_exclusive_file_lock(path, timeout_sec=0)
    assert fd is not None
    monkeypatch.setattr(platform_layer.time, "sleep", lambda _: pytest.fail("nonblocking lock waited"))
    try:
        assert platform_layer.acquire_exclusive_file_lock(path, timeout_sec=0) is None
    finally:
        platform_layer.release_exclusive_file_lock(path, fd)
