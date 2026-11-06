from __future__ import annotations

import json
import types



class _ImmediateThread:
    def __init__(self, target, args=(), kwargs=None, daemon=False, **_ignored):
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}

    def start(self):
        self.target(*self.args, **self.kwargs)


class _Consciousness:
    def inject_observation(self, _text):
        return None

    def pause(self):
        return None

    def resume(self):
        return None


def _ctx(tmp_path, *, pending=None, running=None, direct=None):
    return types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        PENDING=list(pending or []),
        RUNNING=dict(running or {}),
        load_state=lambda: {"owner_id": 1, "owner_chat_id": 1},
        update_state=lambda fn: fn({"owner_id": 1, "owner_chat_id": 1}),
        consciousness=_Consciousness(),
        get_chat_agent=lambda: types.SimpleNamespace(_busy=False),
        handle_chat_direct=direct or (lambda *_a, **_k: None),
        send_with_budget=lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("routing receipts must not create assistant bubbles")
        ),
    )


def _read_result_reference(tmp_path, row):
    from ouroboros.tools.control import _get_task_result
    from ouroboros.tools.registry import ToolContext

    source = row["authority_source"]
    assert source["tool"] == "get_task_result" and source["arguments"]["include_authority"]
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    return json.loads(_get_task_result(ctx, **source["arguments"]))["authority"]


def test_project_zero_call_followup_advances_active_fence_then_falls_through_when_sealed(
    tmp_path, monkeypatch,
):
    import server
    from ouroboros.owner_mailbox import drain_owner_messages
    from ouroboros.projects_registry import create_project
    from supervisor import queue as queue_mod

    project = create_project(tmp_path, "sealed-room")
    chat_id = int(project["chat_id"])
    root_id = "sealed-root"
    pending = [{
        "id": root_id,
        "chat_id": chat_id,
        "root_task_id": root_id,
        "delegation_role": "root",
        "drive_root": str(tmp_path),
    }]
    ctx = _ctx(tmp_path, running={root_id: {"task": pending[0]}})
    monkeypatch.setattr(queue_mod, "ACCEPTANCE_FENCES", {
        root_id: {
            "token": "f" * 32,
            "root_task_id": root_id,
            "task_id": root_id,
            "status": "active",
            "owner_message_generation": 0,
        },
    })

    assert server._route_project_chat_to_running_task(
        ctx, chat_id, "during review", "owner-during-review",
    ) == root_id
    assert queue_mod.ACCEPTANCE_FENCES[root_id]["owner_message_generation"] == 1
    queue_mod.ACCEPTANCE_FENCES[root_id]["status"] = "sealed"
    assert server._route_project_chat_to_running_task(
        ctx, chat_id, "late follow-up", "owner-after-seal",
    ) == ""
    assert drain_owner_messages(tmp_path, root_id) == ["during review"]


def test_project_direct_stale_race_releases_admission_lock_once(tmp_path):
    import server
    from ouroboros.projects_registry import create_project

    project = create_project(tmp_path, "stale-direct")
    chat_id = int(project["chat_id"])
    direct_agent = types.SimpleNamespace(
        _accepting_owner_messages=True,
        _busy=True,
        _current_task_id="direct-stale",
        _current_chat_id=chat_id,
        _current_task_text="Work in progress",
        _current_task_metadata={"project_id": str(project["id"])},
        _task_started_ts=2.0,
    )

    class RacingLock:
        def __init__(self):
            self.locked = False
            self.acquire_calls = 0
            self.release_calls = 0

        def acquire(self):
            self.acquire_calls += 1
            self.locked = True
            if self.acquire_calls == 2:
                direct_agent._current_task_id = "replacement-task"

        def release(self):
            self.release_calls += 1
            if not self.locked:
                raise RuntimeError("release unlocked lock")
            self.locked = False

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, *_args):
            self.release()

    lock = RacingLock()
    direct_agent._owner_message_admission_lock = lock
    ctx = _ctx(tmp_path)
    from supervisor.active_activity import get_direct_activity_registry

    get_direct_activity_registry().register(direct_agent._current_task_id, chat_id, actor=direct_agent)

    assert server._route_project_chat_to_running_task(ctx, chat_id, "late follow-up") == ""
    # One release belongs to the manifest snapshot and one to the routing
    # admission section.  The stale path must not perform a third release.
    assert lock.release_calls == 2
    assert lock.locked is False
