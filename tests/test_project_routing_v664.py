from __future__ import annotations

import json
import pathlib
import types

import pytest
from tests._store_shared import unreadable_result


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


def _read_dialogue_reference(tmp_path, row):
    from ouroboros.tools.chronicle import _memory_read
    from ouroboros.tools.registry import ToolContext

    assert "text" not in row and row["reader"]["tool"] == "memory_read"
    arguments = row["reader"]["arguments"]
    assert arguments["from"] == arguments["to"]
    return _memory_read(ToolContext(repo_dir=tmp_path, drive_root=tmp_path), **arguments)


def test_project_single_pending_root_starts_a_new_head(tmp_path, monkeypatch):
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

    assert drain_owner_messages(tmp_path, "pending-root") == []
    assert "direct" in calls
    assert "owner-1" not in latest_chat_annotations(tmp_path)


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
    # A landed delivery publishes no row: the receipt the caller records carries the stashed report.
    assert notices == [] and "status=staged" in metadata["_attachment_report"]


def test_project_legacy_swarm_flag_uses_ordinary_new_head_for_pending_root(tmp_path, monkeypatch):
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

    monkeypatch.setattr(
        "supervisor.events._handle_promote_chat_to_task",
        lambda event, _ctx: calls.append(("promote", event)) or {"status": "scheduled"},
    )
    monkeypatch.setattr("supervisor.message_bus.log_chat", lambda *_a, **_k: None)
    server._process_bridge_updates(Bridge(), 0, ctx)

    assert drain_owner_messages(tmp_path, "pending-root") == []
    assert calls == ["direct"]


def test_main_legacy_swarm_flag_uses_ordinary_owner_decision(tmp_path, monkeypatch):
    import server
    from ouroboros.projects_registry import create_project

    calls = []
    create_project(tmp_path, "racer", name="Racer")
    ctx = _ctx(
        tmp_path,
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

    monkeypatch.setattr(
        "supervisor.events._handle_promote_chat_to_task",
        lambda event, _ctx: calls.append(("promote", event)) or {"status": "scheduled"},
    )
    monkeypatch.setattr("supervisor.message_bus.log_chat", lambda *_a, **_k: None)
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    server._process_bridge_updates(Bridge(), 0, ctx)

    assert [kind for kind, _kwargs in calls] == ["direct"]


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
        direct=lambda *_a, **_k: calls.append("direct"),
    )
    from supervisor.active_activity import get_direct_activity_registry

    get_direct_activity_registry().register(direct_agent._current_task_id, chat_id, actor=direct_agent)

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
    assert "direct" not in calls
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
    from supervisor.active_activity import get_direct_activity_registry

    get_direct_activity_registry().register(direct_agent._current_task_id, chat_id, actor=direct_agent)

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
        direct=lambda cid, text, image, **kwargs: calls.append((cid, text, image, kwargs)),
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
    assert "on_uncertain_or_invalid_target" not in metadata["routing_contract"]
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
    assert "manual_options" not in metadata["routing_contract"]
    assert {"start_task", "move_to_project", "tell_task"} <= set(metadata["routing_contract"]["valid_actions"])
    assert "route_to_project" not in metadata["routing_contract"]["valid_actions"]


def test_project_room_manual_options_are_room_scoped(tmp_path):
    import server
    from ouroboros.projects_registry import create_project

    project = create_project(tmp_path, "racer", name="Racer")
    chat_id = int(project["chat_id"])
    ctx = _ctx(tmp_path)

    metadata = server._decision_turn_metadata(ctx, chat_id, "project-1", {"project_id": "racer"})

    assert metadata["routing_contract"]["source_lane"] == "project"
    assert "main_routing_manifest" not in metadata
    assert "manual_options" not in metadata["routing_contract"]


def test_project_room_decision_turn_carries_last_task_result_ground_truth(tmp_path):
    """A project room receives its latest result's identity and reader; choosing
    that source returns the original workspace facts, artifacts and result."""
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
    assert "workspace_root" not in last and "artifact_refs" not in last
    assert "result" not in last and "RAW TEXT" not in str(last)
    saved = _read_result_reference(tmp_path, last)
    assert saved["workspace_root"] == str(tmp_path / "racer-tree") and saved["workspace_mode"] == "external"
    from ouroboros.tools.query_data import _query_data
    from ouroboros.tools.registry import ToolContext
    observation = _query_data(ToolContext(repo_dir=tmp_path, drive_root=tmp_path), op="json", kind="task_result",
                              ids=[last["task_id"]], select="metadata.workspace_preflight")
    assert json.loads(observation)["data"]["git"]["head"] == "abc123"
    assert saved["artifact_bundle"]["artifacts"] == [{"path": str(tmp_path / "racer-tree" / "index.html")}]
    assert saved["result"] == "RAW TEXT MUST NOT LEAK"


@pytest.mark.parametrize("shape", ["many_foreign_rows", "equal_ts", "empty_ts"])
def test_project_last_task_result_tie_order_is_total(tmp_path, shape):
    """The order is TOTAL and the store's own: the project's root rows newest first by the
    durable ``ts``, a present ``ts`` beats an absent one, and equal ``ts`` falls back to the
    task id deterministically — however many newer foreign rows exist, and whatever order the
    ids sort in (the older or key-less row is named to sort ahead of the newer one)."""
    import server
    from ouroboros.projects_registry import create_project, get_project, update_project
    from ouroboros.task_results import write_task_result
    from ouroboros.task_store import patch_document

    create_project(tmp_path, "racer", name="Racer")
    if shape == "many_foreign_rows":
        for name in (f"a{i:02d}" for i in range(64)):
            write_task_result(tmp_path, name, "completed", project_id="boat", objective="c", ts="2026-08-10T00:00:03Z")
        write_task_result(tmp_path, "y_old", "completed", project_id="racer", objective="a", ts="2026-08-10T00:00:01Z")
        write_task_result(tmp_path, "z_new", "completed", project_id="racer", objective="b", ts="2026-08-10T00:00:02Z")
        expected = "z_new"
    elif shape == "equal_ts":
        write_task_result(tmp_path, "a", "completed", project_id="racer", objective="a", ts="2026-08-10T00:00:01Z")
        write_task_result(tmp_path, "b", "completed", project_id="racer", objective="b", ts="2026-08-10T00:00:01Z")
        expected = "b"  # equal ts: the greater task id
    else:
        write_task_result(tmp_path, "b_stamped", "completed", project_id="racer", objective="a", ts="2026-08-10T00:00:01Z")
        write_task_result(tmp_path, "a_keyless", "completed", project_id="racer", objective="b", ts="2026-08-10T00:00:02Z")
        # The writer always stamps both; a foreign/legacy row may lack them.
        patch_document(tmp_path, "a_keyless", lambda row: {k: v for k, v in row.items() if k not in ("ts", "updated_at")})
        expected = "b_stamped"  # a present ts beats an absent one

    update_project(tmp_path, "racer", last_task_result_id="")  # the scan, not the pointer, is under test
    assert server._latest_project_task_result(_ctx(tmp_path), "racer")["task_id"] == expected
    # The selected row — never an arbitrary tied one — became the durable pointer.
    assert get_project(tmp_path, "racer")["last_task_result_id"] == expected


def test_project_last_task_result_uncertainty_never_persists_the_pointer(tmp_path, monkeypatch):
    """A row the store cannot answer for might be the project's NEWER result: the lookup is
    refused as unknown (the caller logs and offers nothing) and the durable pointer is not
    written — a later lookup must see the row once it is readable again instead of a frozen
    older row."""
    import server
    from ouroboros.projects_registry import create_project, get_project
    from ouroboros.store import StoreError
    from ouroboros.task_results import write_task_result

    create_project(tmp_path, "racer", name="Racer")
    write_task_result(tmp_path, "old1", "completed", project_id="racer", objective="a", ts="2026-08-10T00:00:01Z")
    write_task_result(tmp_path, "new1", "completed", project_id="racer", objective="b", ts="2026-08-10T00:00:02Z")
    unreadable = unreadable_result(monkeypatch, "new1")
    ctx = _ctx(tmp_path)
    with pytest.raises(StoreError):
        server._latest_project_task_result(ctx, "racer")
    assert not get_project(tmp_path, "racer").get("last_task_result_id")  # nothing frozen
    unreadable["on"] = False
    assert server._latest_project_task_result(ctx, "racer")["task_id"] == "new1"  # readable again: the truth wins
    assert get_project(tmp_path, "racer")["last_task_result_id"] == "new1"


def test_project_last_task_result_lookup_is_bounded(tmp_path, monkeypatch):
    """Finding 1: the one-row query selects the project's root ids newest-first in the store
    and STOPS at the first admissible row — never a full task-result replay per interaction
    (projection over replay)."""
    import server
    from ouroboros import task_results as task_results_module
    from ouroboros.projects_registry import create_project
    from ouroboros.task_results import write_task_result

    create_project(tmp_path, "racer", name="Racer")
    write_task_result(tmp_path, "oldracer1", "completed", project_id="racer", objective="v1", ts="2026-08-10T00:00:01Z")
    write_task_result(tmp_path, "newracer1", "completed", project_id="racer", objective="v2", ts="2026-08-10T00:00:02Z")
    write_task_result(tmp_path, "boat1", "completed", project_id="boat", objective="x", ts="2026-08-10T00:00:03Z")

    opened: list = []
    real_read = task_results_module.read_document

    def _counting_read(root, task_id):
        opened.append(str(task_id))
        return real_read(root, task_id)

    monkeypatch.setattr(task_results_module, "read_document", _counting_read)  # load_task_result's body read

    row = server._latest_project_task_result(_ctx(tmp_path), "racer")

    assert str(row.get("task_id") or row.get("id")) == "newracer1"
    # The project's newest root only: boat1 is filtered in the store, oldracer1 NEVER opened.
    assert opened == ["newracer1"]


def test_main_manifest_carries_working_dir_and_result_readers(tmp_path):
    """Registry working_dir is current; prior workspace facts stay in the named result."""
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
    assert "workspace_root" not in final and "workspace_mode" not in final
    saved = _read_result_reference(tmp_path, final)
    assert saved["workspace_root"] == str(tmp_path / "racer-tree") and saved["workspace_mode"] == "external"


def test_main_manifest_offers_owner_roots_not_swarm_children(tmp_path):
    """I29 (owner decision batch 3, answer 6b=A): the predecessor window held the 16
    newest results of ANY kind, so one swarm wave's children evicted the owner's own
    roots and a result the owner had just read could not be continued from. Children
    are skipped and counted as their OWN omission; the final_results count keeps its
    cap-only meaning."""
    import server
    from ouroboros.task_results import write_task_result

    for index in range(16):
        write_task_result(tmp_path, f"root{index:02d}", "completed", objective=f"owner work {index}",
                          ts=f"2026-08-10T00:00:{index:02d}Z")
    for index in range(2):
        write_task_result(tmp_path, f"child{index}", "completed", objective="helper work",
                          parent_task_id="root15", root_task_id="root15",
                          delegation_role="subagent", ts=f"2026-08-11T00:00:{index:02d}Z")

    manifest = server._main_routing_manifest(_ctx(tmp_path))
    offered = [row["task_id"] for row in manifest["final_results"]]

    assert len(offered) == 16
    assert not [tid for tid in offered if tid.startswith("child")]
    # The OLDEST owner root survives the cap the two children used to consume.
    assert "root00" in offered
    assert manifest["omissions"]["children"] is None  # Main never counts the whole store
    assert manifest["omissions"]["final_results"] == 0  # the window proves no root was cut


def test_omissions_count_children_ordered_after_the_cap(tmp_path):
    """The child count must not be a by-product of the capped loop: children older
    than the 16th root were skipped but counted nowhere, and the cap number silently
    absorbed them (A26 requires the two omissions to stay separate). A room counts
    over its own project's rows; Main's scope is the whole store, which the intake
    on the supervisor loop never walks, so both of its omissions read "not counted"."""
    import server
    from ouroboros.projects_registry import create_project
    from ouroboros.server_routing_context import _project_routing_manifest
    from ouroboros.task_results import write_task_result

    create_project(tmp_path, "racer", name="Racer")
    for index in range(17):
        write_task_result(tmp_path, f"root{index:02d}", "completed", objective=f"owner work {index}",
                          project_id="racer", ts=f"2026-08-20T00:00:{index:02d}Z")
    for index in range(2):
        write_task_result(tmp_path, f"child{index}", "completed", objective="helper work",
                          parent_task_id="root16", root_task_id="root16", project_id="racer",
                          delegation_role="subagent", ts=f"2026-08-01T00:00:{index:02d}Z")

    room = _project_routing_manifest(_ctx(tmp_path), "racer")
    assert len(room["final_results"]) == 16
    assert room["omissions"]["children"] == 2      # counted even though they sort last
    assert room["omissions"]["final_results"] == 1  # exactly the root cut by the cap

    main = server._main_routing_manifest(_ctx(tmp_path))
    assert [row["task_id"] for row in main["final_results"]] == [row["task_id"] for row in room["final_results"]]
    assert main["omissions"]["children"] is None and main["omissions"]["final_results"] is None


def _recent_dialogue(tmp_path):
    import server

    return server._main_routing_manifest(_ctx(tmp_path))["recent_canonical_dialogue"]


def test_main_window_addresses_the_owners_quiz_choice_past_a_long_question(tmp_path):
    """The routing window names a reader; it returns the whole choice and comment
    even when the original quiz frame puts them past a long question."""
    from ouroboros.owner_quiz import quiz_answer_frame
    from supervisor.message_bus import log_chat

    block = {
        "quiz_id": "q-tower", "question": "Which way for the tower? " + "context " * 80,
        "options": ["Keep the tower and repaint it", "Rebuild from scratch"],
        "asked_at": "2026-10-05T10:00:00Z", "answered_at": "2026-10-05T10:05:00Z",
        "answered_index": 0, "comment": "keep it, the base is solid; " * 24, "state": "answered",
    }
    frame = quiz_answer_frame(block, 0, block["comment"])
    assert frame.index("The owner chose option 1") > 500  # past the old clip
    # The row exactly as the answer door writes it (gateway.task_decision).
    log_chat("system", 1, 0, frame, ts=block["answered_at"], source="owner_quiz_answer",
             task_id="root-q", client_message_id="quiz_answer:root-q:q-tower",
             record_type="quiz_answer", quiz=dict(block), drive_root=tmp_path)

    [row] = _recent_dialogue(tmp_path)

    assert len(block["comment"]) > 500
    assert row["chars"] == len(frame.strip())
    shown = _read_dialogue_reference(tmp_path, row)
    assert f'[answer q-tower] chose (1) Keep the tower and repaint it — "{block["comment"]}"' in shown
    assert (row["direction"], row["task_id"]) == ("system", "root-q")


def test_main_window_addresses_people_and_reports_without_copying_their_words(tmp_path):
    """Both spoken rows stay whole at their exact readers; empty rows stay out."""
    from supervisor.message_bus import log_chat

    owner = "w" * 595 + " end."
    lineage = {"subagent_task_id": "child-1", "delegation_role": "subagent",
               "parent_task_id": "root-1", "root_task_id": "root-1"}
    log_chat("in", 1, 1, owner, source="web", client_message_id="owner-600", drive_root=tmp_path)
    log_chat("out", 1, 0, "R" * 2000, task_id="child-1", message_meta=lineage, drive_root=tmp_path)
    log_chat("out", 1, 0, "", task_id="root-1", drive_root=tmp_path)
    log_chat("system", 1, 0, "", task_id="root-1", record_type="task_summary", drive_root=tmp_path)

    rows = _recent_dialogue(tmp_path)

    assert [(row["direction"], row["task_id"]) for row in rows] == [("in", ""), ("out", "child-1")]
    assert [row["chars"] for row in rows] == [600, 2000]
    assert owner in _read_dialogue_reference(tmp_path, rows[0])
    report = _read_dialogue_reference(tmp_path, rows[1])
    assert "R" * 2000 in report and "truncated at 500" not in report


def test_project_routing_facts_keep_host_scope_when_registry_recheck_is_unavailable(
    tmp_path, monkeypatch,
):
    import server
    from ouroboros import server_routing_context

    ctx = _ctx(tmp_path)
    monkeypatch.setattr(server_routing_context, "_project_id_for_registered_chat", lambda *_args: "")

    metadata = server._decision_turn_metadata(
        ctx,
        987654,
        "project-routing-1",
        {"project_id": "racer"},
    )

    assert metadata["routing_contract"]["source_lane"] == "project"
    assert metadata["routing_contract"]["valid_actions"] == [
        "answer_inline", "tell_task", "start_task", "move_to_project",
    ]
    assert "main_routing_manifest" not in metadata


def test_transport_without_client_id_gets_stable_host_owned_routing_id(tmp_path, monkeypatch):
    import server
    from ouroboros.projects_registry import create_project

    create_project(tmp_path, "racer")
    calls = []
    ctx = _ctx(
        tmp_path,
        direct=lambda cid, text, image, **kwargs: calls.append((cid, kwargs)),
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


def test_unread_revision_advances_only_for_conversation_messages(tmp_path, monkeypatch):
    """A progress incident is a row of the task's card: it toasts but leaves unread alone."""
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
    assert get_project(tmp_path, "racer")["visible_revision"] == 0

    message_bus.send_with_budget(chat_id, "final answer", task_id="t")
    assert get_project(tmp_path, "racer")["visible_revision"] == 1


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
        action="tell_task",
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


def test_task_presentation_snapshot_reads_project_for_a_never_named_row_and_keeps_readable_ids(tmp_path):
    from ouroboros.projects_registry import (
        bind_task_to_project,
        create_project,
        task_presentation_snapshot,
    )

    # A row that was never named carries its minted id as the name: that shape reads Project.
    project = create_project(tmp_path, "proj_0a1b2c3d4e5f", name="proj_0a1b2c3d4e5f")
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
    # An owner's own readable id is the name, as the sidebar and the Project reference show it.
    readable = create_project(tmp_path, "community-watch", name="community-watch")
    bind_task_to_project(tmp_path, "watch-task", readable["id"], readable["chat_id"],
                         origin={"absent": "system"})
    assert task_presentation_snapshot(tmp_path, "watch-task")["target_label"] == "community-watch › Task"


def test_project_display_name_shares_the_never_named_rule_with_the_reference():
    import json
    from pathlib import Path

    from ouroboros.project_facts import project_display_name

    fixture = Path(__file__).resolve().parents[1] / "web" / "tests" / "fixtures" / "project_display_name_parity.json"
    for case in json.loads(fixture.read_text(encoding="utf-8"))["cases"]:
        assert project_display_name(case["name"], case["id"]) == case["expected"], case


def test_a_request_without_a_title_is_named_by_its_first_line_not_a_cut_of_the_whole_request():
    from ouroboros.projects_registry import task_presentation_name

    request = ("# Community watch — one six-hourly GitHub pass\n\nThe owner renewed this workflow in Main: "
               "resume it now and every six hours.")
    assert task_presentation_name({"description": request}) == "Community watch — one six-hourly GitHub pass"
    # An authored title still wins over the request, whole.
    assert task_presentation_name({"title": "Watch", "description": request}) == "Watch"


def test_a_scheduled_run_takes_the_schedule_name_as_its_authored_name():
    from supervisor import queue as squeue

    named = squeue._task_from_schedule({
        "id": "community-watch-6h", "name": "Community watch — GitHub every six hours",
        "task": {"text": "# Community watch — one six-hourly GitHub pass\n\nRead the knowledge first."},
    })
    assert (named["title"], named["suggested_name"]) == ("Community watch — GitHub every six hours",) * 2
    unnamed = squeue._task_from_schedule({"id": "plain", "task": {"text": "# Nightly check\n\nRun the battery."}})
    assert "title" not in unnamed  # a derived name never outranks one coined later (admission_names)
    assert unnamed["suggested_name"] == "Nightly check"
    # The host's placeholder for an unnamed follow-up names no work: the run reads its request.
    followup = squeue._task_from_schedule({"id": "followup-0dac19ae-1", "name": "Follow-up of task 0dac19ae",
                                           "task": {"text": "Check whether CI on the PR passed\nIf green, report it."}})
    assert "title" not in followup and followup["suggested_name"] == "Check whether CI on the PR passed"


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


def test_project_completion_enqueues_once_for_root_and_never_for_child_or_direct(
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
        "text": "Launch 🚀 › Ship release · Done",
        "role": "system",
        "system_type": "project_completion_summary",
        "delivery_id": "project-completion:root-project",
        "progress_meta": {
            "outcome_axes": {"schema_version": 1, "lifecycle": {"status": "completed"},
                "execution": {"status": "ok", "reason_code": ""}, "artifacts": {"status": "not_applicable"},
                "objective": {"status": "not_evaluated", "source": "none"},
                "review": {"status": "skipped", "trigger": "not_evaluated"}},
            "project_id": "launch",
            "project_name": "Launch 🚀",
            "target_label": "Launch 🚀 › Ship release",
            "task_name": "Ship release",
            "status": "completed",
            "completion_notice": {"neutral": False, "detail": "", "phase": "done", "salvage": "",
                                  "cause": "", "undescribed": []},
            "terminal_time": {"v": 1, "occurred_at": None, "source": "unknown",
                              "attempt": dict.fromkeys(("task_attempt", "_attempt", "started_at",
                                                        "metadata_attempt", "metadata_task_attempt"))},
        },
    }]

    child = {**root, "id": "child-project", "parent_task_id": "root-project",
             "root_task_id": "root-project", "delegation_role": "subagent"}
    assert enqueue_project_completion_summary(
        ctx.DRIVE_ROOT, event, "child-project", child,
        {**result, "task_id": "child-project"}, done
    ) is False
    assert enqueue_project_completion_summary(
        ctx.DRIVE_ROOT, {"_is_direct_chat": True}, "direct-event",
        {**root, "id": "direct-event"}, {**result, "task_id": "direct-event"}, done
    ) is False
    assert enqueue_project_completion_summary(
        ctx.DRIVE_ROOT, event, "direct-task", {**root, "id": "direct-task", "_is_direct_chat": True},
        {**result, "task_id": "direct-task"}, done,
    ) is False
    assert len(queued) == 1
    assert queued[0]["progress_meta"]["target_label"] == "Launch 🚀 › Ship release"


@pytest.mark.parametrize(
    ("result", "event", "expected"),
    [
        ({"status": "failed"}, {}, "Failed"),
        ({"status": "cancelled"}, {}, "Stopped"),
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
        workers, "get_event_q", lambda **_kw: types.SimpleNamespace(put=queued.append),
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
