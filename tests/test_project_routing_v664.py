from __future__ import annotations

import pathlib
import types

import pytest


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


def _ctx(tmp_path, *, pending=None, running=None, ephemeral=None, direct=None):
    return types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        PENDING=list(pending or []),
        RUNNING=dict(running or {}),
        load_state=lambda: {"owner_id": 1, "owner_chat_id": 1},
        update_state=lambda fn: fn({"owner_id": 1, "owner_chat_id": 1}),
        consciousness=_Consciousness(),
        get_chat_agent=lambda: types.SimpleNamespace(_busy=False),
        handle_chat_ephemeral=ephemeral or (lambda *_a, **_k: None),
        handle_chat_direct=direct or (lambda *_a, **_k: None),
        send_with_budget=lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("routing receipts must not create assistant bubbles")
        ),
    )


def test_project_single_pending_root_gets_zero_call_mailbox_delivery(tmp_path, monkeypatch):
    import server
    from ouroboros.owner_mailbox import drain_owner_messages
    from ouroboros.project_dialogue import latest_chat_annotations
    from ouroboros.projects_registry import create_project

    project = create_project(tmp_path, "racer")
    chat_id = int(project["chat_id"])
    pending = {
        "id": "pending-root",
        "chat_id": chat_id,
        "root_task_id": "pending-root",
        "delegation_role": "root",
        "drive_root": str(tmp_path),
    }
    calls = []
    ctx = _ctx(
        tmp_path,
        pending=[pending],
        ephemeral=lambda *_a, **_k: calls.append("ephemeral"),
        direct=lambda *_a, **_k: calls.append("direct"),
    )

    class Bridge:
        def get_updates(self, offset=0, timeout=1):
            return [{
                "update_id": 1,
                "message": {
                    "chat": {"id": chat_id},
                    "from": {"id": 1},
                    "text": "continue with the failing test",
                    "source": "web",
                    "client_message_id": "owner-1",
                },
            }]

        def send_routing_ack(self, *args, **kwargs):
            calls.append((args, kwargs))

        def broadcast(self, _payload):
            return None

    monkeypatch.setattr("supervisor.message_bus.log_chat", lambda *_a, **_k: None)
    server._process_bridge_updates(Bridge(), 0, ctx)

    assert drain_owner_messages(tmp_path, "pending-root") == ["continue with the failing test"]
    assert "ephemeral" not in calls and "direct" not in calls
    assert calls[-1][1]["status"] == "delivered"
    assert calls[-1][1]["target"] == "pending-root"
    annotation = latest_chat_annotations(tmp_path)["owner-1"]
    assert annotation["action"] == "mailbox_delivery"
    assert annotation["target"] == "pending-root"
    assert annotation["status"] == "delivered"


def test_project_fast_route_stages_attachment_in_actor_child_drive(tmp_path):
    import server
    from ouroboros.projects_registry import create_project
    from ouroboros.tools.core import _read_file
    from ouroboros.tools.registry import ToolContext

    project = create_project(tmp_path, "attachment-room")
    chat_id = int(project["chat_id"])
    child_drive = tmp_path / "task_drives" / "forked-root" / "data"
    child_drive.mkdir(parents=True)
    source = tmp_path / "owner-input.txt"
    source.write_text("child-drive-readable", encoding="utf-8")
    task = {
        "id": "forked-root", "chat_id": chat_id, "root_task_id": "forked-root",
        "delegation_role": "root", "drive_root": str(child_drive),
        "child_drive_root": str(child_drive),
    }
    ctx = _ctx(tmp_path, running={"forked-root": {"task": task}})
    notices = []
    ctx.send_with_budget = lambda _chat, text, **_kwargs: notices.append(text)
    metadata = {
        "chat_attachment_uploads": [{"path": str(source), "label": "owner input"}],
    }

    assert server._route_project_chat_to_running_task(
        ctx, chat_id, "use this", "owner-attachment", task_metadata=metadata,
    ) == "forked-root"

    manifest = metadata["_attachment_manifest"]
    assert pathlib.Path(manifest[0]["abs_path"]).is_relative_to(child_drive)
    tool_ctx = ToolContext(repo_dir=tmp_path, drive_root=child_drive, task_id="forked-root")
    assert "child-drive-readable" in _read_file(
        tool_ctx, manifest[0]["relpath"], root="artifact_store",
    )
    assert notices and "status=staged" in notices[-1]


def test_project_swarm_bypasses_single_root_mailbox_for_new_managed_root(tmp_path, monkeypatch):
    import server
    from ouroboros.owner_mailbox import drain_owner_messages
    from ouroboros.projects_registry import create_project

    project = create_project(tmp_path, "racer")
    chat_id = int(project["chat_id"])
    pending = {
        "id": "pending-root",
        "chat_id": chat_id,
        "root_task_id": "pending-root",
        "delegation_role": "root",
        "drive_root": str(tmp_path),
    }
    calls = []
    ctx = _ctx(
        tmp_path,
        pending=[pending],
        ephemeral=lambda cid, text, image, **kwargs: calls.append((cid, text, kwargs)),
        direct=lambda *_a, **_k: calls.append("direct"),
    )

    class Bridge:
        def get_updates(self, offset=0, timeout=1):
            return [{
                "update_id": 2,
                "message": {
                    "chat": {"id": chat_id}, "from": {"id": 1}, "source": "web",
                    "text": "deeply fix the new issue", "client_message_id": "swarm-project-1",
                    "task_metadata": {"force_plan": True, "force_plan_source": "swarm"},
                },
            }]

        def broadcast(self, _payload):
            return None

    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr("supervisor.message_bus.log_chat", lambda *_a, **_k: None)
    server._process_bridge_updates(Bridge(), 0, ctx)

    assert drain_owner_messages(tmp_path, "pending-root") == []
    assert len(calls) == 1 and calls[0] != "direct"
    metadata = calls[0][2]["task_metadata"]
    assert metadata["force_plan"] is True
    assert metadata["routing_contract"]["valid_actions"] == ["promote_chat_to_task"]
    assert metadata["routing_contract"]["on_uncertain_or_invalid_target"] == "promote_chat_to_task"
    assert metadata["routing_contract"]["manual_options"] == []


def test_empty_main_swarm_uses_ephemeral_router_not_direct_lane(tmp_path, monkeypatch):
    import server
    from ouroboros.projects_registry import create_project

    calls = []
    create_project(tmp_path, "racer", name="Racer")
    ctx = _ctx(
        tmp_path,
        ephemeral=lambda cid, text, image, **kwargs: calls.append(("ephemeral", kwargs)),
        direct=lambda *_a, **_k: calls.append(("direct", {})),
    )

    class Bridge:
        def get_updates(self, offset=0, timeout=1):
            return [{
                "update_id": 3,
                "message": {
                    "chat": {"id": 1}, "from": {"id": 1}, "source": "web",
                    "text": "audit and fix it", "client_message_id": "swarm-main-1",
                    "task_metadata": {"force_plan": True, "force_plan_source": "swarm"},
                },
            }]

        def broadcast(self, _payload):
            return None

    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr("supervisor.message_bus.log_chat", lambda *_a, **_k: None)
    server._process_bridge_updates(Bridge(), 0, ctx)

    assert [kind for kind, _kwargs in calls] == ["ephemeral"]
    contract = calls[0][1]["task_metadata"]["routing_contract"]
    assert contract["valid_actions"] == ["promote_chat_to_task", "route_to_project"]
    assert contract["manual_options"] == []


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
    ctx = _ctx(tmp_path, pending=pending)
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


def test_project_single_active_direct_root_gets_zero_call_mailbox_delivery(tmp_path, monkeypatch):
    import threading

    import server
    from ouroboros.owner_mailbox import drain_owner_messages
    from ouroboros.projects_registry import create_project

    project = create_project(tmp_path, "racer")
    chat_id = int(project["chat_id"])
    direct_agent = types.SimpleNamespace(
        _owner_message_admission_lock=threading.Lock(),
        _accepting_owner_messages=True,
        _busy=True,
        _current_task_id="direct-racer",
        _current_chat_id=chat_id,
        _current_task_text="Tune the racer",
        _current_task_metadata={"project_id": "racer"},
        _task_started_ts=2.0,
    )
    calls = []
    ctx = _ctx(
        tmp_path,
        ephemeral=lambda *_a, **_k: calls.append("ephemeral"),
        direct=lambda *_a, **_k: calls.append("direct"),
    )
    ctx.get_chat_agent = lambda: direct_agent

    class Bridge:
        def get_updates(self, offset=0, timeout=1):
            return [{
                "update_id": 1,
                "message": {
                    "chat": {"id": chat_id},
                    "from": {"id": 1},
                    "text": "also check the brakes",
                    "source": "web",
                    "client_message_id": "direct-followup",
                },
            }]

        def send_routing_ack(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr("supervisor.message_bus.log_chat", lambda *_a, **_k: None)
    server._process_bridge_updates(Bridge(), 0, ctx)

    assert drain_owner_messages(tmp_path, "direct-racer") == ["also check the brakes"]
    assert "ephemeral" not in calls and "direct" not in calls
    assert calls[-1][1]["action"] == "mailbox_delivery"
    assert calls[-1][1]["target"] == "direct-racer"


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
    ctx.get_chat_agent = lambda: direct_agent

    assert server._route_project_chat_to_running_task(ctx, chat_id, "late follow-up") == ""
    # One release belongs to the manifest snapshot and one to the routing
    # admission section.  The stale path must not perform a third release.
    assert lock.release_calls == 2
    assert lock.locked is False


def test_main_inline_decision_has_no_predecision_annotation(tmp_path, monkeypatch):
    import server
    from ouroboros.project_dialogue import latest_chat_annotations
    from ouroboros.projects_registry import create_project

    create_project(tmp_path, "racer")
    calls = []
    ctx = _ctx(
        tmp_path,
        ephemeral=lambda cid, text, image, **kwargs: calls.append((cid, text, image, kwargs)),
        direct=lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("direct lane bypassed router")),
    )

    broadcasts = []

    class Bridge:
        def get_updates(self, offset=0, timeout=1):
            return [{
                "update_id": 1,
                "message": {
                    "chat": {"id": 1}, "from": {"id": 1}, "source": "web",
                    "text": "what happened to the racer?", "client_message_id": "main-1",
                },
            }]

        def broadcast(self, payload):
            broadcasts.append(payload)

    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr("supervisor.message_bus.log_chat", lambda *_a, **_k: None)
    server._process_bridge_updates(Bridge(), 0, ctx)

    assert len(calls) == 1
    metadata = calls[0][3]["task_metadata"]
    assert metadata["main_routing_manifest"]["projects"][0]["project_id"] == "racer"
    assert metadata["routing_contract"]["on_uncertain_or_invalid_target"] == "needs_manual_target"
    assert latest_chat_annotations(tmp_path) == {}
    # Typing is the only wait affordance. An inline answer emits no typed routing
    # action, so the canonical user row must never gain a transient annotation.
    assert broadcasts == []


def test_main_manifest_manual_options_include_project_roots_and_new_project_task(tmp_path):
    import server
    from ouroboros.projects_registry import create_project

    project = create_project(tmp_path, "racer", name="Racer")
    ctx = _ctx(
        tmp_path,
        running={
            "project-root": {
                "task": {
                    "id": "project-root",
                    "chat_id": int(project["chat_id"]),
                    "project_id": "racer",
                    "title": "Tune engine",
                    "delegation_role": "root",
                },
            },
        },
    )

    metadata = server._decision_turn_metadata(ctx, 987654, "external-1", {})

    assert metadata["routing_contract"]["source_lane"] == "main"
    options = metadata["routing_contract"]["manual_options"]
    assert any(row.get("task_id") == "project-root" for row in options)
    assert any(
        row.get("action") == "new_task_in_project"
        and row.get("project_id") == "racer"
        and row.get("label") == "New task in Racer"
        for row in options
    )


def test_project_room_manual_options_are_room_scoped(tmp_path):
    import server
    from ouroboros.projects_registry import create_project

    project = create_project(tmp_path, "racer", name="Racer")
    chat_id = int(project["chat_id"])
    ctx = _ctx(tmp_path)

    metadata = server._decision_turn_metadata(ctx, chat_id, "project-1", {"project_id": "racer"})

    assert metadata["routing_contract"]["source_lane"] == "project"
    assert "main_routing_manifest" not in metadata
    assert metadata["routing_contract"]["manual_options"] == [{
        "action": "new_task_in_project",
        "project_id": "racer",
        "label": "New task in Project",
    }]


def test_project_room_decision_turn_carries_last_task_result_ground_truth(tmp_path):
    """Q8-A: a project-room decision turn receives the thread's most recent task
    result as a BOUNDED typed projection (identity/outcome/workspace facts/
    artifact refs, no raw result text) — the ground truth a 'continue' promotion
    must read instead of chat memory."""
    import server
    from ouroboros.projects_registry import create_project
    from ouroboros.task_results import write_task_result

    project = create_project(tmp_path, "racer", name="Racer")
    write_task_result(
        tmp_path, "old1", "completed", project_id="racer",
        objective="scaffold the racer", result="RAW TEXT MUST NOT LEAK",
        ts="2026-08-10T00:00:01Z",
    )
    write_task_result(
        tmp_path, "new1", "completed", project_id="racer",
        objective="build the racer", result="RAW TEXT MUST NOT LEAK",
        reason_code="", workspace_root=str(tmp_path / "racer-tree"),
        workspace_mode="external",
        metadata={"workspace_preflight": {"git": {"head": "abc123", "branch": "main", "dirty": False}}},
        artifact_bundle={"artifacts": [{"path": str(tmp_path / "racer-tree" / "index.html")}]},
        ts="2026-08-10T00:00:02Z",
    )
    write_task_result(
        tmp_path, "other1", "completed", project_id="boat",
        objective="unrelated", ts="2026-08-10T00:00:03Z",
    )
    ctx = _ctx(tmp_path)

    metadata = server._decision_turn_metadata(
        ctx, int(project["chat_id"]), "project-1", {"project_id": "racer"},
    )

    last = metadata["project_last_task_result"]
    assert last["task_id"] == "new1"  # most recent for THIS project, not "boat"
    assert last["status"] == "completed"
    assert last["workspace_root"] == str(tmp_path / "racer-tree")
    assert last["workspace_mode"] == "external"
    assert last["workspace_git_at_start"]["head"] == "abc123"
    assert last["artifact_refs"] == [str(tmp_path / "racer-tree" / "index.html")]
    assert "result" not in last and "RAW TEXT" not in str(last)


@pytest.mark.parametrize("shape", ["group_crosses_the_window", "group_in_the_self_heal_tail", "equal_ts", "empty_ts"])
def test_project_last_task_result_tie_order_is_total(tmp_path, monkeypatch, shape):
    """The tie order is TOTAL: an equal-mtime group that crosses the 64-entry
    search window is read to its end, a group met in the self-heal tail obeys
    the same rule, a present `ts` beats an absent one, and equal `ts` falls
    back to the task id deterministically. The production order is (mtime,
    file name), so the file NAMES are chosen to sort adversely — the older or
    key-less row ahead of the newer one — and the listing order is irrelevant."""
    import json
    import os

    import server
    from ouroboros import task_results as task_results_module
    from ouroboros.projects_registry import create_project, update_project
    from ouroboros.task_results import task_results_dir, write_task_result

    create_project(tmp_path, "racer", name="Racer")
    if shape == "group_crosses_the_window":
        # Sorted by name: a00..a62 (foreign), y_old at index 63, z_new at 64 —
        # the newer row is the FIRST entry past the window.
        foreign = [f"a{i:02d}" for i in range(63)]
        for name in foreign:
            write_task_result(tmp_path, name, "completed", project_id="boat", objective="c", ts="2026-08-10T00:00:03Z")
        write_task_result(tmp_path, "y_old", "completed", project_id="racer", objective="a", ts="2026-08-10T00:00:01Z")
        write_task_result(tmp_path, "z_new", "completed", project_id="racer", objective="b", ts="2026-08-10T00:00:02Z")
        listing = foreign + ["y_old", "z_new"]
        expected = "z_new"
    elif shape == "group_in_the_self_heal_tail":
        # Sorted by name: a00..a63 (foreign) fill the window; both project rows
        # are met only by the self-heal tail, the older one first.
        foreign = [f"a{i:02d}" for i in range(64)]
        for name in foreign:
            write_task_result(tmp_path, name, "completed", project_id="boat", objective="c", ts="2026-08-10T00:00:03Z")
        write_task_result(tmp_path, "y_old", "completed", project_id="racer", objective="a", ts="2026-08-10T00:00:01Z")
        write_task_result(tmp_path, "z_new", "completed", project_id="racer", objective="b", ts="2026-08-10T00:00:02Z")
        listing = foreign + ["y_old", "z_new"]
        expected = "z_new"
    elif shape == "equal_ts":
        write_task_result(tmp_path, "a", "completed", project_id="racer", objective="a", ts="2026-08-10T00:00:01Z")
        write_task_result(tmp_path, "b", "completed", project_id="racer", objective="b", ts="2026-08-10T00:00:01Z")
        listing = ["a", "b"]
        expected = "b"  # equal ts: the greater task id, whatever the listing order
    else:
        # Sorted by name the key-less row comes FIRST; the stamped row must win.
        write_task_result(tmp_path, "b_stamped", "completed", project_id="racer", objective="a", ts="2026-08-10T00:00:01Z")
        write_task_result(tmp_path, "a_keyless", "completed", project_id="racer", objective="b", ts="2026-08-10T00:00:02Z")
        keyless = task_results_dir(tmp_path, create=False) / "a_keyless.json"
        stripped = json.loads(keyless.read_text(encoding="utf-8"))
        for key in ("ts", "updated_at"):  # the store always stamps both; a foreign/legacy row may lack them
            stripped.pop(key, None)
        keyless.write_text(json.dumps(stripped), encoding="utf-8")
        listing = ["a_keyless", "b_stamped"]
        expected = "b_stamped"  # a present ts beats an absent one
    real_dir = task_results_dir(tmp_path, create=False)
    tick = 1_700_000_000 * 10**9
    for path in real_dir.glob("*.json"):
        os.utime(path, ns=(tick, tick))  # every result in ONE tie group

    class _Listing:  # the real directory with a controlled listing order
        def __init__(self, order):
            self.paths = [real_dir / f"{name}.json" for name in order]

        def glob(self, pattern):
            return iter(self.paths)

        def __truediv__(self, name):
            return real_dir / name

    from ouroboros.projects_registry import get_project

    for order in (listing, list(reversed(listing))):
        update_project(tmp_path, "racer", last_task_result_id="")  # the scan, not the pointer, is under test
        monkeypatch.setattr(task_results_module, "task_results_dir", lambda root, create=True, order=order: _Listing(order))
        assert server._latest_project_task_result(_ctx(tmp_path), "racer")["task_id"] == expected
        # The selected row — never an arbitrary tied one — became the durable pointer.
        assert get_project(tmp_path, "racer")["last_task_result_id"] == expected


def test_project_last_task_result_uncertainty_never_persists_the_pointer(tmp_path, monkeypatch):
    """A result file whose stat (or JSON) cannot be read might be the project's
    NEWER result: the scan still answers best-effort for this call, but the
    durable pointer is not written — a later lookup must see the file once it
    is readable again instead of the frozen older row."""
    import server
    from ouroboros import task_results as task_results_module
    from ouroboros.projects_registry import create_project, get_project
    from ouroboros.task_results import task_results_dir, write_task_result

    create_project(tmp_path, "racer", name="Racer")
    write_task_result(tmp_path, "old1", "completed", project_id="racer", objective="a", ts="2026-08-10T00:00:01Z")
    write_task_result(tmp_path, "new1", "completed", project_id="racer", objective="b", ts="2026-08-10T00:00:02Z")
    real_dir = task_results_dir(tmp_path, create=False)

    class _Unstattable:  # readable JSON whose stat fails (a transient filesystem error)
        def __init__(self, real):
            self._real = real
            self.name = real.name

        def stat(self):
            raise OSError("transient stat failure")

        def __fspath__(self):
            return str(self._real)

        def __str__(self):
            return str(self._real)

    listing = [real_dir / "old1.json", _Unstattable(real_dir / "new1.json")]
    monkeypatch.setattr(task_results_module, "task_results_dir",
                        lambda root, create=True: types.SimpleNamespace(glob=lambda pattern: iter(listing),
                                                                         __truediv__=lambda self, name: real_dir / name))
    ctx = _ctx(tmp_path)
    assert server._latest_project_task_result(ctx, "racer")["task_id"] == "old1"  # best effort for THIS call
    assert not get_project(tmp_path, "racer").get("last_task_result_id")  # nothing frozen
    monkeypatch.undo()
    assert server._latest_project_task_result(ctx, "racer")["task_id"] == "new1"  # readable again: the truth wins
    assert get_project(tmp_path, "racer")["last_task_result_id"] == "new1"


def test_project_last_task_result_lookup_is_bounded(tmp_path, monkeypatch):
    """Finding 1: the one-row query parses newest-first by mtime and STOPS at
    the first project match — never a full task_results replay per interaction
    (projection over replay)."""
    import os

    import server
    import ouroboros.utils as utils
    from ouroboros.projects_registry import create_project
    from ouroboros.task_results import task_result_path, write_task_result

    create_project(tmp_path, "racer", name="Racer")
    write_task_result(tmp_path, "oldracer1", "completed", project_id="racer", objective="v1")
    write_task_result(tmp_path, "newracer1", "completed", project_id="racer", objective="v2")
    write_task_result(tmp_path, "boat1", "completed", project_id="boat", objective="x")
    # Deterministic mtime order regardless of filesystem timestamp granularity.
    for name, mtime in (("oldracer1", 100), ("newracer1", 200), ("boat1", 300)):
        os.utime(task_result_path(tmp_path, name, create=False), (mtime, mtime))

    opened: list = []
    real_read = utils.read_json_dict

    def _counting_read(path):
        opened.append(path.stem)
        return real_read(path)

    monkeypatch.setattr(utils, "read_json_dict", _counting_read)

    row = server._latest_project_task_result(_ctx(tmp_path), "racer")

    assert str(row.get("task_id") or row.get("id")) == "newracer1"
    # Newest-first, stop at the match: boat1 examined, oldracer1 NEVER opened.
    assert opened == ["boat1", "newracer1"]


def test_main_manifest_carries_working_dir_and_workspace_facts(tmp_path):
    """Q8-A: Main-lane router ground truth — registry working_dir on project rows
    and workspace facts on the recent-result projections."""
    import server
    from ouroboros.projects_registry import create_project, update_project
    from ouroboros.task_results import write_task_result

    create_project(tmp_path, "racer", name="Racer")
    update_project(tmp_path, "racer", working_dir=str(tmp_path / "racer-tree"))
    write_task_result(
        tmp_path, "done1", "completed", project_id="racer",
        objective="build the racer", workspace_root=str(tmp_path / "racer-tree"),
        workspace_mode="external", ts="2026-08-10T00:00:01Z",
    )
    ctx = _ctx(tmp_path)

    manifest = server._main_routing_manifest(ctx)

    assert manifest["projects"][0]["working_dir"] == str(tmp_path / "racer-tree")
    final = next(row for row in manifest["final_results"] if row["task_id"] == "done1")
    assert final["workspace_root"] == str(tmp_path / "racer-tree")
    assert final["workspace_mode"] == "external"


def test_project_swarm_keeps_host_scope_when_registry_recheck_is_unavailable(
    tmp_path, monkeypatch,
):
    import server
    from ouroboros import server_routing_context

    ctx = _ctx(tmp_path)
    monkeypatch.setattr(server_routing_context, "_project_id_for_registered_chat", lambda *_args: "")

    metadata = server._decision_turn_metadata(
        ctx,
        987654,
        "project-swarm-1",
        {"project_id": "racer", "force_plan": True, "force_plan_source": "swarm"},
    )

    assert metadata["routing_contract"]["source_lane"] == "project"
    assert metadata["routing_contract"]["valid_actions"] == ["promote_chat_to_task"]
    assert "main_routing_manifest" not in metadata


def test_transport_without_client_id_gets_stable_host_owned_routing_id(tmp_path, monkeypatch):
    import server
    from ouroboros.projects_registry import create_project

    create_project(tmp_path, "racer")
    calls = []
    ctx = _ctx(
        tmp_path,
        ephemeral=lambda cid, text, image, **kwargs: calls.append((cid, kwargs)),
    )
    logged = []
    broadcasts = []

    class Bridge:
        def get_updates(self, offset=0, timeout=1):
            return [{
                "update_id": 77,
                "message": {
                    "chat": {"id": 7001},
                    "from": {"id": 7001},
                    "source": "cli",
                    "text": "continue racer",
                },
            }]

        def broadcast(self, payload):
            broadcasts.append(payload)

    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(
        "supervisor.message_bus.log_chat",
        lambda *args, **kwargs: logged.append(kwargs.get("client_message_id")),
    )

    server._process_bridge_updates(Bridge(), 0, ctx)

    generated = logged[0]
    assert generated.startswith("host-")
    assert calls[0][1]["task_metadata"]["client_message_id"] == generated
    # The stable id is retained for any later actual route/manual receipt; merely
    # entering the decision lane is not itself a message_annotation.
    assert broadcasts[0]["type"] == "chat"  # canonical non-Web owner row is preserved
    assert not any(row.get("type") == "message_annotation" for row in broadcasts)


def test_unread_revision_advances_only_for_visible_result_or_incident(tmp_path, monkeypatch):
    from ouroboros.projects_registry import create_project, get_project
    from supervisor import message_bus

    project = create_project(tmp_path, "racer")
    chat_id = int(project["chat_id"])
    bridge = message_bus.LocalChatBridge()
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 1, "session_id": "s"})

    message_bus.send_with_budget(chat_id, "ordinary progress", is_progress=True, task_id="t")
    assert get_project(tmp_path, "racer")["visible_revision"] == 0

    message_bus.send_with_budget(
        chat_id,
        "worker lost",
        is_progress=True,
        task_id="t",
        progress_meta={"task_incident": "worker_lost", "toast_once": "t:worker_lost"},
    )
    assert get_project(tmp_path, "racer")["visible_revision"] == 1

    message_bus.send_with_budget(chat_id, "final answer", task_id="t")
    assert get_project(tmp_path, "racer")["visible_revision"] == 2


def test_routing_ack_is_typed_and_never_broadcast_as_chat_bubble(monkeypatch):
    from supervisor import message_bus

    ws_payloads = []
    bus_payloads = []
    bridge = message_bus.LocalChatBridge()
    bridge._broadcast_fn = ws_payloads.append
    monkeypatch.setattr(message_bus, "publish_event", lambda topic, payload: bus_payloads.append((topic, payload)))

    bridge.send_routing_ack(
        7,
        client_message_id="m-1",
        action="mailbox_delivery",
        target="task-1",
        status="delivered",
    )

    assert ws_payloads == [{
        "type": "message_annotation",
        "annotation_type": "routing_ack",
        "chat_id": 7,
        "client_message_id": "m-1",
        "action": "mailbox_delivery",
        "target": "task-1",
        "status": "delivered",
        "suppress_bubble": True,
        "ts": ws_payloads[0]["ts"],
    }]
    assert bus_payloads[0][1]["text"] == ""
    assert bus_payloads[0][1]["suppress_bubble"] is True
    assert all(payload.get("type") != "chat" for payload in ws_payloads)


def test_routing_ack_carries_event_time_human_label_without_replacing_target(monkeypatch):
    from supervisor import message_bus
    ws_payloads = []
    bridge = message_bus.LocalChatBridge()
    bridge._broadcast_fn = ws_payloads.append
    monkeypatch.setattr(message_bus, "publish_event", lambda *_a, **_k: None)
    bridge.send_routing_ack(
        7,
        client_message_id="m-labelled",
        action="steer_task",
        target="opaque-task-id",
        target_label="Проект 🚀 › Исправить тесты",
        status="delivered",
    )
    assert ws_payloads[-1]["target"] == "opaque-task-id"
    assert ws_payloads[-1]["target_label"] == "Проект 🚀 › Исправить тесты"


def test_task_presentation_snapshot_prefers_human_names_and_keeps_machine_ids(tmp_path):
    from ouroboros.projects_registry import (
        bind_task_to_project,
        create_project,
        task_presentation_snapshot,
    )
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    project = create_project(tmp_path, "opaque-project-id", name="Космос 🌌")
    bind_task_to_project(
        tmp_path,
        "opaque-task-id",
        project["id"],
        project["chat_id"],
        origin={"absent": "system"},
    )
    write_task_result(
        tmp_path,
        "opaque-task-id",
        STATUS_RUNNING,
        title="Явный заголовок",
        suggested_name="Позднее имя",
        objective="Очень длинная цель",
    )
    snapshot = task_presentation_snapshot(tmp_path, "opaque-task-id")
    assert snapshot == {
        "project_id": "opaque-project-id",
        "project_name": "Космос 🌌",
        "task_id": "opaque-task-id", "project_routable": True,
        "task_name": "Явный заголовок",
        "target_label": "Космос 🌌 › Явный заголовок",
    }


def test_task_presentation_snapshot_rejects_id_only_project_name_and_is_neutral(tmp_path):
    from ouroboros.projects_registry import (
        bind_task_to_project,
        create_project,
        task_presentation_snapshot,
    )

    project = create_project(tmp_path, "opaque-project-id", name="opaque-project-id")
    bind_task_to_project(
        tmp_path,
        "opaque-task-id",
        project["id"],
        project["chat_id"],
        origin={"absent": "system"},
    )

    snapshot = task_presentation_snapshot(tmp_path, "opaque-task-id")

    assert snapshot["project_name"] == "Project"
    assert snapshot["task_name"] == "Task"
    assert snapshot["target_label"] == "Project › Task"


def test_task_presentation_snapshot_reads_existing_title_from_durable_live_queue(tmp_path):
    import json
    from ouroboros.projects_registry import bind_task_to_project, create_project, task_presentation_snapshot

    project = create_project(tmp_path, "human-project", name="Human Project")
    bind_task_to_project(tmp_path, "live-task", project["id"], project["chat_id"],
                         origin={"absent": "system"})
    (tmp_path / "state" / "queue_snapshot.json").write_text(json.dumps({
        "running": [{"id": "live-task", "task": {
            "id": "live-task", "title": "Human Task",
        }}],
        "pending": [],
    }), encoding="utf-8")
    assert task_presentation_snapshot(tmp_path, "live-task")["target_label"] == (
        "Human Project › Human Task")


def test_task_presentation_snapshot_bounds_existing_objective_and_does_not_make_ids_unique(
    tmp_path,
):
    from ouroboros.projects_registry import task_presentation_snapshot

    objective = "Describe the existing work " + ("carefully " * 20)
    first = task_presentation_snapshot(
        tmp_path, "machine-key-one", task={"objective": objective},
    )
    second = task_presentation_snapshot(
        tmp_path, "machine-key-two", task={"objective": objective},
    )

    assert first["task_name"] == second["task_name"]
    assert first["target_label"] == second["target_label"]
    assert first["task_id"] != second["task_id"]
    assert len(first["task_name"]) <= 80
    assert first["task_name"].endswith("…")


def test_project_completion_enqueues_once_for_root_and_never_for_child_or_ephemeral(
    tmp_path, monkeypatch,
):
    from ouroboros.projects_registry import bind_task_to_project, create_project, update_project
    from ouroboros.project_dialogue import enqueue_project_completion_summary

    project = create_project(tmp_path, "launch", name="Launch 🚀")
    bind_task_to_project(
        tmp_path,
        "root-project",
        project["id"],
        project["chat_id"],
        origin={"absent": "system"},
    )
    queued = []
    seen = set()

    def _enqueue(_root, event, **_kwargs):
        delivery_id = event["delivery_id"]
        if delivery_id in seen:
            return False
        seen.add(delivery_id)
        queued.append(dict(event))
        return True

    monkeypatch.setattr(
        "supervisor.terminal_delivery.enqueue_terminal_delivery",
        _enqueue,
    )
    ctx = types.SimpleNamespace(DRIVE_ROOT=tmp_path)
    root = {
        "id": "root-project",
        "project_id": "launch",
        "title": "Ship release",
        "chat_id": project["chat_id"],
    }
    result = {
        "task_id": "root-project",
        "status": "completed",
        "project_id": "launch",
        "title": "Ship release",
        "result": "Release shipped.",
    }
    event = {"status": "completed"}
    done = {"status": "completed", "outcome_axes": {"execution": {"status": "ok"}}}

    assert enqueue_project_completion_summary(
        ctx.DRIVE_ROOT, event, "root-project", root, result, done
    ) is True
    update_project(tmp_path, project["id"], name="Renamed after completion")
    assert enqueue_project_completion_summary(
        ctx.DRIVE_ROOT, event, "root-project", root, result, done
    ) is False
    assert queued == [{
        "type": "send_message",
        "chat_id": 1,
        "task_id": "root-project",
        "text": "Launch 🚀 › Ship release · Done\nRelease shipped.",
        "role": "system",
        "system_type": "project_completion_summary",
        "delivery_id": "project-completion:root-project",
        "progress_meta": {
            "project_id": "launch",
            "project_name": "Launch 🚀",
            "target_label": "Launch 🚀 › Ship release",
            "status": "completed",
        },
    }]

    child = {**root, "id": "child-project", "parent_task_id": "root-project",
             "root_task_id": "root-project", "delegation_role": "subagent"}
    assert enqueue_project_completion_summary(
        ctx.DRIVE_ROOT, event, "child-project", child,
        {**result, "task_id": "child-project"}, done
    ) is False
    assert enqueue_project_completion_summary(
        ctx.DRIVE_ROOT, {"_ephemeral": True}, "root-project", root, result, done
    ) is False
    assert enqueue_project_completion_summary(
        ctx.DRIVE_ROOT, event, "root-project", root, result,
        {**done, "ephemeral_decision": True},
    ) is False
    assert len(queued) == 1
    assert queued[0]["progress_meta"]["target_label"] == "Launch 🚀 › Ship release"


@pytest.mark.parametrize(
    ("result", "event", "expected"),
    [
        ({"status": "failed"}, {}, "Failed"),
        ({"status": "cancelled"}, {}, "Cancelled"),
        (
            {"status": "completed"},
            {"outcome_axes": {"execution": {"status": "degraded"}}},
            "Done with warnings",
        ),
        (
            {"status": "completed", "outcome_axes": {"execution": {"status": "degraded"}}},
            {},
            "Done with warnings",
        ),
        (
            {"status": "completed", "outcome_axes": {"objective": {"status": "fail", "source": "task_acceptance_review"}}},
            {},
            "Failed",
        ),
        (
            {"status": "completed", "outcome_axes": {"execution": {"status": "failed"}}},
            {},
            "Failed",
        ),
        (
            {"status": "completed", "outcome_axes": {"review": {"status": "fail"}}},
            {},
            "Failed",
        ),
    ],
)
def test_project_completion_summary_labels_every_terminal_outcome(result, event, expected):
    from ouroboros.project_dialogue import completion_status_label

    assert completion_status_label(result, event) == expected


def test_project_completion_duplicate_outbox_events_deliver_one_main_row(tmp_path, monkeypatch):
    from ouroboros.project_dialogue import enqueue_project_completion_summary
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from supervisor import events, workers

    project = create_project(tmp_path, "stable-project", name="Stable Project")
    bind_task_to_project(
        tmp_path, "stable-root", project["id"], project["chat_id"],
        origin={"absent": "system"},
    )
    queued = []
    monkeypatch.setattr(
        workers, "get_event_q", lambda: types.SimpleNamespace(put=queued.append),
    )
    events._DELIVERED_MESSAGE_IDS.clear()
    task = {"id": "stable-root", "project_id": project["id"], "title": "Finish"}
    result = {**task, "task_id": "stable-root", "status": "completed"}
    done = {"status": "completed", "outcome_axes": {"execution": {"status": "ok"}}}

    assert enqueue_project_completion_summary(
        tmp_path, {}, "stable-root", task, result, done,
    ) is True
    assert enqueue_project_completion_summary(
        tmp_path, {}, "stable-root", task, result, done,
    ) is True
    assert len(queued) == 2  # duplicate live copies share one durable delivery id

    sent = []
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path, RUNNING={}, send_with_budget=lambda *a, **k: sent.append((a, k)),
        append_jsonl=lambda *_a, **_k: None,
    )
    events._handle_send_message(queued[0], ctx)
    events._handle_send_message(queued[1], ctx)

    assert len(sent) == 1
    assert sent[0][0][0] == 1
    assert sent[0][1]["system_type"] == "project_completion_summary"
    assert sent[0][1]["progress_meta"]["target_label"] == "Stable Project › Finish"


def test_manual_target_event_persists_concrete_options_in_latest_annotation(
    tmp_path, monkeypatch,
):
    from ouroboros.project_dialogue import latest_chat_annotations
    from supervisor import message_bus
    from supervisor.events import _handle_routing_manual_target

    ws_payloads = []
    bridge = message_bus.LocalChatBridge()
    bridge._broadcast_fn = ws_payloads.append
    monkeypatch.setattr(message_bus, "publish_event", lambda *_a, **_k: None)
    ctx = types.SimpleNamespace(DRIVE_ROOT=tmp_path, bridge=bridge)
    event = {
        "chat_id": 9,
        "client_message_id": "owner-choice-1",
        "requested_target": "ghost",
        "options": [
            {"action": "steer_task", "task_id": "task-1", "title": "Fix tests"},
            {"action": "new_task_in_project", "project_id": "racer", "label": "New task in Project"},
        ],
    }

    _handle_routing_manual_target(event, ctx)

    payload = ws_payloads[-1]
    assert payload["type"] == "message_annotation"
    assert payload["status"] == "needs_manual_target"
    assert payload["options"][0]["label"] == "Fix tests"
    assert payload["options"][1] == event["options"][1]
    assert payload["suppress_bubble"] is True
    sidecar = latest_chat_annotations(tmp_path)["owner-choice-1"]
    assert set(sidecar) >= {"client_message_id", "action", "target", "status"}
    assert sidecar["options"] == payload["options"]
