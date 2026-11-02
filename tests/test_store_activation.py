"""The cutover (S3 activation): ``store.migrating`` as the write doors' predicate, the four
gateway doors answering 503 ``migrating`` inside the boot import window and proceeding outside
it, the boot door (``store_switch``: a pending switch first, the import owner off the ready
path, readiness after publication), the seam at the restart exit on the kill census, and the
rollback doors that record an authority switch for a store-less target instead of resetting
(UI Rollback, the agent's ``vcs_rollback`` through the button's procedure, the boot-smoke
rollback, Update replace)."""
from __future__ import annotations

import json
import pathlib
import threading
import time
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import store, store_switch
from ouroboros.store import StoreMigrating
from ouroboros.store_import import ensure_current_store
from ouroboros.task_results import write_task_result
from tests.test_store_import import fixture_root
from tests.test_store_switch import Owner, git
from tests.test_store_switch import target_repo as _switch_target_repo

REPO = pathlib.Path(__file__).resolve().parents[1]
target_repo = _switch_target_repo  # ``old``: a tree without the store modules; ``new``: with them


def _wire_git_ops(monkeypatch, repo: pathlib.Path, root: pathlib.Path) -> None:
    import supervisor.git_ops as git_ops

    monkeypatch.setattr(git_ops, "REPO_DIR", repo, raising=False)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", root, raising=False)
    monkeypatch.setattr(git_ops, "git_capture", lambda cmd: (0, git(repo, *cmd[1:]), ""))


# ---- the predicate ---------------------------------------------------------------------------

def test_migrating_is_true_until_publication_and_inside_the_import_window(tmp_path):
    """Readiness is publication: a root holding facts no import published is migrating and the
    predicate imports nothing (never opens for write); a root holding none is not (its first
    write publishes an empty store); the published store is ready outside the window only."""
    root = tmp_path / "data"
    (root / "task_results").mkdir(parents=True)
    (root / "task_results" / "t1.json").write_text(json.dumps(
        {"task_id": "t1", "status": "completed", "ts": "2026-10-10T00:00:00+00:00", "_schema_version": 1}))
    assert store.migrating(root) is True and store.store_ready(root) is False
    assert store.probe(root)[0] == "absent"  # the predicate is a pure read
    idle = tmp_path / "idle"
    idle.mkdir()
    assert store.migrating(idle) is False and store.store_ready(idle) is False and store.probe(idle)[0] == "absent"
    ensure_current_store(root)
    assert store.migrating(root) is False and store.store_ready(root) is True
    with store.importing(root):
        assert store.migrating(root) is True
        assert store.store_ready(root) is False
    assert store.migrating(root) is False and store.store_ready(root) is True


def test_migrating_is_true_for_a_published_store_while_another_thread_imports(tmp_path):
    """Another thread's window over a published store answers migrating to every door."""
    root = tmp_path / "data"
    ensure_current_store(root)
    seen = {}

    def other_thread():
        with store.importing(root):
            seen["inside"] = store.migrating(root)
            gate.set()
            release.wait(5)

    gate, release = threading.Event(), threading.Event()
    thread = threading.Thread(target=other_thread)
    thread.start()
    assert gate.wait(5)
    try:
        assert store.migrating(root) is True
    finally:
        release.set()
        thread.join(5)
    assert seen["inside"] is True and store.migrating(root) is False


# ---- the gateway doors -----------------------------------------------------------------------

def _migrating_body(response) -> dict:
    assert response.status_code == 503
    body = json.loads(response.body) if hasattr(response, "body") else response.json()
    assert body["reason_code"] == "migrating" if "reason_code" in body else body["code"] == "migrating"
    return body


def test_create_door_answers_migrating_inside_the_window_and_proceeds_outside(tmp_path):
    from ouroboros.gateway import tasks

    root = tmp_path / "data"
    write_task_result(root, "taken", "completed")
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(drive_root=root, repo_dir=REPO)))
    body = {"description": "hello", "task_id": "taken"}
    with store.importing(root):
        _migrating_body(tasks._create_task_from_body(request, body))
    outside = tasks._create_task_from_body(request, body)
    assert outside.status_code == 409 and b"already exists" in outside.body  # past the gate, the door's own answer


def test_cancel_door_answers_migrating_inside_the_window_and_proceeds_outside(tmp_path, monkeypatch):
    from supervisor import queue
    from ouroboros.gateway import tasks

    root = tmp_path / "data"
    monkeypatch.setattr(queue, "DRIVE_ROOT", root)  # the intent writer's root is the door's
    write_task_result(root, "done", "completed")
    app = Starlette(routes=[Route("/api/tasks/{task_id}/cancel", tasks.api_task_cancel, methods=["POST"])])
    app.state.drive_root = root
    with TestClient(app) as client:
        with store.importing(root):
            _migrating_body(client.post("/api/tasks/done/cancel", json={}))
        outside = client.post("/api/tasks/done/cancel", json={})
    assert outside.status_code != 503


def test_pause_door_answers_migrating_inside_the_window_and_proceeds_outside(tmp_path, monkeypatch):
    from ouroboros.gateway import task_pause
    from supervisor import owner_pause_control

    root = tmp_path / "data"
    ensure_current_store(root)
    calls = []
    monkeypatch.setattr(owner_pause_control, "request_owner_pause",
                        lambda task_id, *, request_id, by=None: calls.append((task_id, request_id, by)) or {"ok": True})
    app = Starlette(routes=[Route("/api/tasks/{task_id}/pause", task_pause.api_task_pause, methods=["POST"])])
    app.state.drive_root = root
    with TestClient(app) as client:
        with store.importing(root):
            _migrating_body(client.post("/api/tasks/t1/pause", json={"request_id": "r1"}))
        assert calls == []
        outside = client.post("/api/tasks/t1/pause", json={"request_id": "r1"})
    assert outside.status_code == 200 and calls == [("t1", "r1", None)], "a bare body declares nobody"


def test_presence_door_answers_migrating_inside_the_window_and_proceeds_outside(tmp_path, monkeypatch):
    from ouroboros.gateway.host_service import create_host_service_app
    from tests.test_host_service_api import _seed_presence_behavior, _seed_token

    monkeypatch.setattr("ouroboros.presence_admission.focus_enabled", lambda _root: True)
    monkeypatch.setattr("ouroboros.presence_admission.focus_allowance", lambda _root: {"status": "available", "remaining_usd": 10})
    _seed_token(tmp_path, skill="telegram-bot", token="presence-token", permissions=["presence"],
                manifest_permissions=["presence"])
    binding_id = _seed_presence_behavior(tmp_path)
    ensure_current_store(tmp_path)
    runs = []

    def run_presence(**kwargs):
        runs.append(kwargs["event"].source_event_id)
        return SimpleNamespace(outcome="message", text="Hello back", task_id="turn-1", work_ref="")

    app = create_host_service_app(tmp_path, presence_runner=run_presence)
    event = {"source_event_id": "telegram:bot-1:42", "provider": "telegram", "account_id": "bot-1",
             "conversation_id": "room-1", "thread_id": "topic-1", "conversation_key": "ignored",
             "actor": {"platform_actor_id": "user-7"}, "conversation": {}, "message": {}, "text": "Hello"}
    client = TestClient(app)
    headers = {"X-Skill-Token": "presence-token"}
    with store.importing(tmp_path):
        refused = client.post("/presence/turn", headers=headers, json={"binding_id": binding_id, "event": event})
    assert refused.status_code == 503
    assert refused.json()["code"] == "migrating" and refused.json()["disposition"] == "retry" and runs == []
    answered = client.post("/presence/turn", headers=headers, json={"binding_id": binding_id, "event": event})
    assert answered.status_code == 200, answered.json()
    assert runs == ["telegram:bot-1:42"]


def test_the_task_reading_gets_answer_503_migrating_instead_of_importing(tmp_path):
    """GET /api/tasks, GET /api/tasks/{id} and GET /api/chat/history on a root whose store is
    not published (never imported: the boot door owns the import) answer 503 ``migrating`` and
    import nothing; after publication they answer (a published store reads inside the window)."""
    from ouroboros.gateway import tasks
    from ouroboros.gateway.history import make_chat_history_endpoint

    root = tmp_path / "data"
    (root / "task_results").mkdir(parents=True)
    (root / "task_results" / "t1.json").write_text(json.dumps(
        {"task_id": "t1", "status": "completed", "ts": "2026-10-10T00:00:00+00:00", "_schema_version": 1}))
    (root / "logs").mkdir()
    (root / "logs" / "chat.jsonl").write_text(json.dumps(  # a quiz card: the history reads its task's projection
        {"ts": "2026-10-10T00:00:00+00:00", "chat_id": 1, "direction": "out", "type": "quiz", "task_id": "t1",
         "quiz": {"quiz_id": "q1", "question": "Choose?", "state": "open", "options": ["a", "b"]}}) + "\n")
    app = Starlette(routes=[Route("/api/tasks", tasks.api_tasks_list, methods=["GET"]),
                            Route("/api/tasks/{task_id}", tasks.api_task_get, methods=["GET"]),
                            Route("/api/chat/history", make_chat_history_endpoint(root), methods=["GET"])])
    app.state.drive_root = root
    with TestClient(app) as client:
        for path in ("/api/tasks", "/api/tasks/t1", "/api/chat/history"):
            _migrating_body(client.get(path))
        assert store.probe(root)[0] == "absent"  # no GET imported; the file stays where it was
        ensure_current_store(root)
        assert [row["task_id"] for row in client.get("/api/tasks").json()["tasks"]] == ["t1"]
        assert client.get("/api/tasks/t1").json()["status"] == "completed"
        assert client.get("/api/chat/history").status_code == 200


# ---- the boot door ---------------------------------------------------------------------------

def test_boot_imports_off_the_ready_path_and_runs_the_waiting_steps_after_publication(tmp_path, monkeypatch):
    import ouroboros.startup_migrations as startup_migrations

    root = fixture_root(tmp_path)
    entered, release = threading.Event(), threading.Event()
    real = startup_migrations.prepare_startup_state

    def slow_prepare(root_arg, **kwargs):
        entered.set()
        assert release.wait(10)
        return real(root_arg, **kwargs)

    monkeypatch.setattr(startup_migrations, "prepare_startup_state", slow_prepare)
    restarts, steps = [], []
    outcome = store_switch.start(root, repo_dir=REPO, request_restart=lambda: restarts.append(1))
    assert outcome == {"status": "importing"}
    assert store.migrating(root) is True  # the window opened before the thread: no idle gap for a door
    assert entered.wait(10)
    assert store.migrating(root) is True  # every write door answers typed while the owner runs
    store_switch.when_ready(root, lambda: steps.append("recovery"), lambda: steps.append("ready"))
    assert steps == []  # not before publication
    release.set()
    ready = store_switch.wait_ready(root, timeout=30)
    assert ready["status"] == "published", ready
    deadline = time.monotonic() + 10
    while steps != ["recovery", "ready"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert steps == ["recovery", "ready"]  # in order, on the import thread, after publication
    assert store.migrating(root) is False and store.store_ready(root) is True and restarts == []
    assert store_switch.require_published(root)["status"] == "published"
    store_switch.when_ready(root, lambda: steps.append("late"))
    deadline = time.monotonic() + 10
    while steps[-1] != "late" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert steps[-1] == "late"  # after publication a step runs at once


@pytest.mark.parametrize("import_here", [True, False])
def test_steps_registered_after_the_publication_never_run_on_the_registering_thread(tmp_path, import_here):
    """The lifespan registers its steps and only then starts the boot coordinator that one of them
    waits on (the extensions): a fast import that already published, or a door without an import,
    must not run them on the registering thread, which would then never start that coordinator."""
    root = fixture_root(tmp_path)
    if import_here:
        store_switch.start(root, repo_dir=REPO, request_restart=lambda: None)
        assert store_switch.wait_ready(root, timeout=30)["status"] == "published"  # the import finished first
    coordinator, ran = threading.Event(), []
    store_switch.when_ready(root, lambda: ran.append(coordinator.wait(10)))
    coordinator.set()  # this thread's next work, reached only because the registration returned
    deadline = time.monotonic() + 10
    while not ran and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ran == [True]


def test_a_failed_boot_import_leaves_the_root_not_ready_and_no_request_re_imports(tmp_path, monkeypatch):
    """Done is not published: a failed import keeps the window open for the life of the process —
    the doors answer migrating, the reads 503, the waiting steps never run, the supervisor's
    startup phase raises, and no later GET or write imports the facts the root holds."""
    import ouroboros.startup_migrations as startup_migrations
    from ouroboros.store_import import ensure_current_store

    root = fixture_root(tmp_path)
    monkeypatch.setattr(startup_migrations, "prepare_startup_state",
                        lambda *a, **kw: (_ for _ in ()).throw(OSError(24, "EMFILE while folding")))
    restarts, steps = [], []
    assert store_switch.start(root, repo_dir=REPO, request_restart=lambda: restarts.append(1)) == {"status": "importing"}
    store_switch.when_ready(root, lambda: steps.append("recovery"), lambda: steps.append("ready"))
    outcome = store_switch.wait_ready(root, timeout=30)
    assert outcome["status"] == "failed" and "EMFILE" in outcome["error"] and restarts == []
    assert store.migrating(root) is True and store.store_ready(root) is False and store.probe(root)[0] == "absent"
    with pytest.raises(StoreMigrating):
        store_switch.require_published(root)
    store_switch.when_ready(root, lambda: steps.append("late"))
    assert steps == []  # dropped before and after done: readiness follows publication only
    from ouroboros.gateway import tasks

    app = Starlette(routes=[Route("/api/tasks", tasks.api_tasks_list, methods=["GET"]),
                            Route("/api/tasks/{task_id}", tasks.api_task_get, methods=["GET"])])
    app.state.drive_root, app.state.repo_dir = root, REPO
    with TestClient(app) as client:
        _migrating_body(client.get("/api/tasks"))
        _migrating_body(client.get("/api/tasks/t1"))
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(drive_root=root, repo_dir=REPO)))
    _migrating_body(tasks._create_task_from_body(request, {"description": "hello"}))
    with pytest.raises(StoreMigrating):
        write_task_result(root, "t2", "running")
    assert store.probe(root)[0] == "absent"  # nothing re-imported: the next boot retries
    store.end_import(root)  # the next boot (here: the window released) imports what the root holds
    assert ensure_current_store(root)["status"] == "completed" and store.store_ready(root) is True


def test_boot_wait_ready_without_a_boot_answers_at_once(tmp_path):
    assert store_switch.wait_ready(tmp_path / "never", timeout=0.01) == {"status": "not_started"}


def test_startup_recovery_waits_for_the_publication_before_its_first_write(tmp_path, monkeypatch):
    from ouroboros import server_maintenance

    order = []
    monkeypatch.setattr(store_switch, "wait_ready", lambda root, timeout=None: order.append(("wait", root)) or {"status": "published"})
    monkeypatch.setattr(server_maintenance, "_startup_live_task_ids",
                        lambda root, **kw: order.append(("read", root)) or (_ for _ in ()).throw(RuntimeError("stop here")))
    report = server_maintenance._run_startup_task_recovery(tmp_path, REPO, skip_live_data=False, prior_worker_pids=set())
    assert order == [("wait", tmp_path), ("read", tmp_path)]
    assert report["errors"] == ["startup_ownership_unconfirmed: RuntimeError"]
    assert server_maintenance._run_startup_task_recovery(tmp_path, REPO, skip_live_data=True)["errors"] == []
    assert order == [("wait", tmp_path), ("read", tmp_path)]  # the skipped live-data path waits on nothing
    monkeypatch.setattr(store_switch, "wait_ready", lambda root, timeout=None: {"status": "failed", "error": "x"})
    report = server_maintenance._run_startup_task_recovery(tmp_path, REPO, skip_live_data=False, prior_worker_pids=set())
    assert report["errors"] == ["store_not_published"] and order[-1] == ("read", tmp_path)  # deferred: no read, no write


def test_boot_with_a_pending_intent_switches_on_the_import_thread_and_keeps_the_window(tmp_path, monkeypatch, target_repo):
    """The resume runs off the HTTP start path, under the switching gate: ``start`` returns at
    once, the switch requests the restart, nothing is imported into the retired store and the
    window stays open (the doors answer migrating until the switched tree's own boot)."""
    repo, old, new = target_repo
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "data"))
    root = fixture_root(tmp_path)
    ensure_current_store(root)
    store_switch.record_switch_intent(root, target_sha=old, target_ref="v-old", reason="test", door="ui_rollback")
    owner, restarts, steps = Owner(repo), [], []
    monkeypatch.setattr(store_switch, "switching_owner", lambda *_: owner)
    assert store_switch.start(root, repo_dir=repo, request_restart=lambda: restarts.append(1)) == {"status": "importing"}
    store_switch.when_ready(root, lambda: steps.append("ready"))
    outcome = store_switch.wait_ready(root, timeout=60)
    assert outcome["status"] == "switched" and restarts == [1] and len(owner.calls) == 1 and steps == []
    assert git(repo, "rev-parse", "HEAD") == old
    assert store_switch.read_switch_intent(root) is None
    assert store.probe(root)[0] == "absent"  # retired aside; the old body reads the exported files
    assert store.migrating(root) is True  # the window is kept: no door writes into the retired store's root
    with pytest.raises(StoreMigrating):
        store_switch.require_published(root)
    store.end_import(root)


# ---- the seam at the restart exit -------------------------------------------------------------

def _census(**over):
    census = {"worker_exits": {"unconfirmed": []}, "live_children": [], "owned_stop": {"state": "completed"}}
    census.update(over)
    return census


@pytest.mark.parametrize("alive", [True, False])
def test_restart_census_refuses_a_maintenance_exit_until_confirmed(tmp_path, monkeypatch, alive):
    import server
    from ouroboros import maintenance

    root = tmp_path / "data"
    root.mkdir()
    signals, switches = [], []
    child = SimpleNamespace(poll=lambda: None if alive else 0, kill=lambda: signals.append("kill"))
    packer = maintenance.MaintenanceChild()
    packer._process = child
    monkeypatch.setattr(maintenance, "child", packer)
    monkeypatch.setattr(server, "DATA_DIR", root)
    monkeypatch.setattr(server, "begin_owned_stop", lambda *_, **__: None)
    monkeypatch.setattr(server, "stop_owned_work", lambda *_: {"state": "completed"})
    monkeypatch.setattr("supervisor.workers.kill_workers", lambda **_: None)
    monkeypatch.setattr("supervisor.workers.last_worker_exit_census", lambda: {"unconfirmed": []})
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    monkeypatch.setattr("ouroboros.extension_companion.panic_kill_all", lambda: None)
    server._restart_requested.clear()
    cleanup = server._emergency_process_cleanup()
    assert bool(cleanup["live_children"]) is alive
    assert packer._stopped.is_set() and signals == ["kill"]
    monkeypatch.setattr(store_switch, "switch_needs", lambda *a: {"handles": False, "money": False, "facts": False})
    monkeypatch.setattr(store_switch, "rev_parse", lambda *a: "example-target")
    monkeypatch.setattr(store_switch, "switching_owner", lambda *_: lambda intent: (switches.append(intent) or True, "switched"))
    store_switch.record_switch_intent(root, target_sha="example-target", target_ref="example", reason="test", door="ui_rollback")
    outcome = store_switch.switch_at_restart(root, root, cleanup)
    if alive:
        assert outcome["status"] == "refused" and outcome["reason"] == "quiescence_unconfirmed"
        assert switches == []
    else:
        assert outcome["status"] == "switched" and len(switches) == 1


def test_maintenance_census_covers_a_spawn_before_the_child_handle_is_published():
    from ouroboros.maintenance import MaintenanceChild

    packer = MaintenanceChild()
    alive = [True]
    packer._thread = SimpleNamespace(is_alive=lambda: alive[0])
    packer.stop()
    assert packer.live_children() == [packer._thread]
    alive[0] = False
    assert packer.live_children() == []


@pytest.mark.parametrize("door", ["boot", "restart"])
def test_switch_doors_pass_restore_callback_and_restore_before_switch(tmp_path, monkeypatch, door):
    from ouroboros import evidence_packs, evidence_packer
    from tests.test_evidence_packs import _task, _handle_path

    root = tmp_path / "data"
    root.mkdir()
    raw = b"checkpoint restored before switching"
    _, handle = _task(root, "restore", blobs=1, handle=raw)
    assert evidence_packer.run_pass(root)["packed"] == 1
    target = _handle_path(root, "restore", handle)
    assert not target.exists()
    monkeypatch.setattr(store_switch, "switch_needs", lambda *a: {"handles": True, "money": False, "facts": False})
    monkeypatch.setattr(store_switch, "rev_parse", lambda *a: "example-target")
    store_switch.record_switch_intent(root, target_sha="example-target", target_ref="example", reason="test", door="ui_rollback")
    calls = []

    def owner(intent):
        assert target.read_bytes() == raw
        calls.append(intent)
        return True, "switched"

    name = "resume_pending_switch" if door == "boot" else "run_switch_seam"
    original = getattr(store_switch, name)

    def seam(*args, **kwargs):
        assert kwargs["restore_handles"] is evidence_packs.restore_checkpoint_handles
        return original(*args, **kwargs)

    monkeypatch.setattr(store_switch, name, seam)
    monkeypatch.setattr(store_switch, "switching_owner", lambda *_: owner)
    if door == "boot":
        assert store_switch.start(root, repo_dir=root, request_restart=lambda: None) == {"status": "importing"}
        outcome = store_switch.wait_ready(root, timeout=60)
        store.end_import(root)
    else:
        outcome = store_switch.switch_at_restart(root, root, _census())
    assert outcome["status"] == "switched" and len(calls) == 1


def test_switch_at_restart_without_an_intent_is_a_no_op(tmp_path):
    assert store_switch.switch_at_restart(tmp_path / "data", REPO, _census()) == {"status": "no_intent"}


def test_a_tree_that_already_lost_the_store_exports_at_the_restart_exit_without_a_second_reset(tmp_path, monkeypatch, target_repo):
    """T2 (S3 fix 15, Astra M13): the agent's ``vcs_revert`` or a manual checkout moved HEAD to a
    tree without the store and no door recorded an intent; the restart exit exports the facts for
    that tree and moves nothing."""
    repo, old, new = target_repo
    git(repo, "checkout", "-q", "--detach", old)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "data"))
    root = fixture_root(tmp_path)
    ensure_current_store(root)
    monkeypatch.setattr(store_switch, "switching_owner", lambda door: pytest.fail("nothing may reset a second time"))
    outcome = store_switch.switch_at_restart(root, repo, _census())
    assert outcome["status"] == "switched" and outcome["exit"] == 42, outcome
    assert git(repo, "rev-parse", "HEAD") == old and store.probe(root)[0] == "absent"
    assert (root / "task_results" / "root1.json").exists() and store_switch.read_switch_intent(root) is None
    receipt = json.loads((root / store_switch.RECEIPT_REL).read_text(encoding="utf-8"))
    assert receipt["intent"]["target_sha"] == old and receipt["intent"]["door"] == "vcs_rollback"


def test_a_tree_without_the_dialogue_store_gets_its_rows_back_at_the_restart_exit(tmp_path, monkeypatch):
    """A manual checkout of a body that reads the facts store but not the dialogue store (every
    release before it): the restart exit exports the rows only the store holds into the chain
    files that body reads; the facts store stays where it is and nothing resets a second time."""
    from ouroboros import dialogue_store

    root = fixture_root(tmp_path)
    ensure_current_store(root)
    dialogue_store.append(root, "chat", {"direction": "out", "chat_id": 1, "text": "a reply only the store holds"})
    monkeypatch.setattr(store_switch, "switch_needs",
                        lambda *_: {"facts": False, "money": False, "handles": False, "dialogue": True})
    monkeypatch.setattr(store_switch, "rev_parse", lambda *_: "older-head")
    monkeypatch.setattr(store_switch, "switching_owner", lambda door: pytest.fail("nothing may reset a second time"))
    outcome = store_switch.switch_at_restart(root, root, _census())
    assert outcome["status"] == "switched", outcome
    assert "a reply only the store holds" in (root / "logs" / "chat.jsonl").read_text(encoding="utf-8")
    assert store.probe(root)[0] == "present" and store_switch.read_switch_intent(root) is None
    monkeypatch.setattr(store_switch, "switch_needs", lambda *_: dict.fromkeys(("facts", "money", "handles", "dialogue"), False))
    assert store_switch.switch_at_restart(root, root, _census()) == {"status": "no_intent"}, "a tree that reads both"


@pytest.mark.parametrize("cleanup, blocker", [
    (None, "worker_exits_unconfirmed"),
    (_census(worker_exits=None), "worker_exits_unconfirmed"),
    (_census(worker_exits={"unconfirmed": [4242]}), "worker_exits_unconfirmed"),
    (_census(live_children=[object()]), "child_processes_alive"),
    (_census(owned_stop={"state": "unconfirmed", "unconfirmed": ["x"]}), "owned_work_unconfirmed"),
])
def test_switch_at_restart_refuses_on_an_unconfirmed_stop_and_moves_nothing(tmp_path, monkeypatch, target_repo, cleanup, blocker):
    repo, old, new = target_repo
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "data"))
    root = fixture_root(tmp_path)
    ensure_current_store(root)
    store_switch.record_switch_intent(root, target_sha=old, target_ref="v-old", reason="test", door="ui_rollback")
    monkeypatch.setattr(store_switch, "switching_owner", lambda door: pytest.fail("nothing may switch"))
    outcome = store_switch.switch_at_restart(root, repo, cleanup)
    assert outcome["status"] == "refused" and outcome["reason"] == "quiescence_unconfirmed"
    assert blocker in outcome["blockers"]
    assert git(repo, "rev-parse", "HEAD") == new and store.probe(root)[0] == "present"
    assert store_switch.read_switch_intent(root) is None  # set aside with its reason, not pending forever


def test_switch_at_restart_on_a_clean_census_exports_and_switches_through_the_door_owner(tmp_path, monkeypatch, target_repo):
    repo, old, new = target_repo
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "data"))
    root = fixture_root(tmp_path)
    ensure_current_store(root)
    store_switch.record_switch_intent(root, target_sha=old, target_ref="v-old", reason="test", door="ui_rollback")
    owner = Owner(repo)
    monkeypatch.setattr(store_switch, "switching_owner", lambda door: owner if door == "ui_rollback" else pytest.fail(door))
    outcome = store_switch.switch_at_restart(root, repo, _census())
    assert outcome["status"] == "switched" and outcome["exit"] == 42, outcome
    assert git(repo, "rev-parse", "HEAD") == old and owner.calls[0]["target_sha"] == old
    assert store.probe(root)[0] == "absent"
    assert (root / "task_results" / "root1.json").exists()  # the exported facts the old body reads


def test_switch_at_restart_never_raises(tmp_path, monkeypatch, target_repo):
    repo, old, new = target_repo
    root = tmp_path / "data"
    store_switch.record_switch_intent(root, target_sha=old, target_ref="v-old", reason="test", door="ui_rollback")
    monkeypatch.setattr(store_switch, "run_switch_seam", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    outcome = store_switch.switch_at_restart(root, repo, _census())
    assert outcome["status"] == "error" and "boom" in outcome["error"]


# ---- the doors' one decision -----------------------------------------------------------------

def test_switch_or_reset_records_the_intent_for_a_store_less_target_and_resets_otherwise(tmp_path, target_repo):
    repo, old, new = target_repo
    root = tmp_path / "data"
    resets = []

    def reset():
        resets.append(1)
        return True, "reset done"

    ok, message, deferred = store_switch.switch_or_reset(root, repo, target_sha=old, target_ref="v-old",
                                                         reason="ui_rollback", door="ui_rollback", reset=reset)
    assert (ok, deferred) == (True, True) and "recorded" in message and resets == []
    intent = store_switch.read_switch_intent(root)
    assert intent["target_sha"] == old and intent["door"] == "ui_rollback" and intent["reason"] == "ui_rollback"
    with pytest.raises(RuntimeError):  # a second door press before the restart: a refusal, never a second intent
        store_switch.switch_or_reset(root, repo, target_sha=old, target_ref="v-old", reason="x", door="ui_rollback", reset=reset)
    root2 = tmp_path / "data2"
    ok, message, deferred = store_switch.switch_or_reset(root2, repo, target_sha=new, target_ref="v-new",
                                                         reason="ui_rollback", door="ui_rollback", reset=reset)
    assert (ok, message, deferred) == (True, "reset done", False) and resets == [1]
    assert store_switch.read_switch_intent(root2) is None


@pytest.mark.parametrize("door", ["ui_rollback", "vcs_rollback"])
def test_rollback_door_records_the_switch_for_a_store_less_target(tmp_path, monkeypatch, target_repo, door):
    import supervisor.git_ops as git_ops
    from ouroboros.gateway import control

    repo, old, new = target_repo
    root = tmp_path / "data"
    _wire_git_ops(monkeypatch, repo, root)
    monkeypatch.setattr(git_ops, "rollback_to_version", lambda *a, **k: pytest.fail("the checkout must not move now"))
    monkeypatch.setattr(control, "_acquire_repo_mutation_lock", lambda: (None, None))
    monkeypatch.setattr(control, "_release_repo_mutation_lock", lambda fh: None)
    quiesced, restarts = [], []
    monkeypatch.setattr(control, "_quiesce_repo_writers", lambda reason: quiesced.append(reason) or [])
    status, payload = control.rollback_through_door(old, door=door, request_restart=lambda: restarts.append(1) or True)
    assert status == 200 and payload["status"] == "ok" and payload["switch_deferred"] is True and payload["restarting"]
    assert quiesced == ["manual_rollback"] and restarts == [1]  # writers stopped first, the restart requested after
    intent = store_switch.read_switch_intent(root)
    assert intent["target_sha"] == old and intent["door"] == door and intent["target_ref"] == old
    assert intent["reason"] == {"ui_rollback": "ui_rollback", "vcs_rollback": "agent_rollback_tool"}[door]
    assert git(repo, "rev-parse", "HEAD") == new


def test_rollback_door_resets_at_once_for_a_target_with_the_store(tmp_path, monkeypatch, target_repo):
    import supervisor.git_ops as git_ops
    from ouroboros.gateway import control

    repo, old, new = target_repo
    root = tmp_path / "data"
    _wire_git_ops(monkeypatch, repo, root)
    resets = []
    monkeypatch.setattr(git_ops, "rollback_to_version", lambda sha, reason: resets.append((sha, reason)) or (True, "rolled"))
    monkeypatch.setattr(control, "_acquire_repo_mutation_lock", lambda: (None, None))
    monkeypatch.setattr(control, "_release_repo_mutation_lock", lambda fh: None)
    monkeypatch.setattr(control, "_quiesce_repo_writers", lambda reason: [])
    status, payload = control.rollback_through_door(new, door="ui_rollback", request_restart=lambda: False)
    assert status == 200 and payload == {"status": "restart_required", "message": "rolled", "restarting": False,
                                         "switch_deferred": False}
    assert resets == [(new, "ui_rollback")] and store_switch.read_switch_intent(root) is None


def test_rollback_door_fence_refusal_moves_nothing(tmp_path, monkeypatch, target_repo):
    import supervisor.git_ops as git_ops
    from ouroboros.gateway import control

    repo, old, new = target_repo
    _wire_git_ops(monkeypatch, repo, tmp_path / "data")
    monkeypatch.setattr(git_ops, "rollback_to_version", lambda *a, **k: pytest.fail("fenced"))
    monkeypatch.setattr(control, "_acquire_repo_mutation_lock", lambda: (None, None))
    monkeypatch.setattr(control, "_release_repo_mutation_lock", lambda fh: None)
    monkeypatch.setattr(control, "_quiesce_repo_writers", lambda reason: ["active:direct_chat"])
    status, payload = control.rollback_through_door(old, door="vcs_rollback", request_restart=lambda: pytest.fail("no restart"))
    assert status == 409 and payload["blockers"] == ["active:direct_chat"]
    assert store_switch.read_switch_intent(tmp_path / "data") is None


def test_ui_rollback_endpoint_wraps_the_door(monkeypatch):
    from ouroboros.gateway import control

    seen = []
    monkeypatch.setattr(control, "rollback_through_door",
                        lambda target, *, door, request_restart: seen.append((target, door, request_restart())) or (200, {"status": "ok"}))
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(request_restart=lambda owner=False: seen.append(("owner", owner)))))
    response = control._git_rollback_fenced(request, "v1")
    assert response.status_code == 200 and json.loads(response.body) == {"status": "ok"}
    assert seen == [("owner", True), ("v1", "ui_rollback", True)]


# ---- the agent's vcs_rollback through the button's procedure (P4-S) ------------------------------

def test_rollback_request_handler_runs_the_door_and_tells_the_owner(monkeypatch):
    from ouroboros.gateway import control
    from supervisor.events_runtime_controls import _handle_rollback_request

    doors, sent, restarts, done = [], [], [], threading.Event()

    def door(target, *, door, request_restart):
        doors.append((target, door))
        assert request_restart() is True
        return 200, {"status": "ok", "message": "authority switch recorded", "switch_deferred": True}

    monkeypatch.setattr(control, "rollback_through_door", door)
    ctx = SimpleNamespace(request_restart=lambda: restarts.append(1), load_state=lambda: {"owner_chat_id": 7},
                          send_with_budget=lambda chat_id, text, **kw: sent.append((chat_id, text, kw)) or done.set())
    _handle_rollback_request({"type": "rollback_request", "target": "v7.6.0", "target_sha": "abc"}, ctx)
    assert done.wait(10)
    assert doors == [("v7.6.0", "vcs_rollback")] and restarts == [1]
    assert sent[0][0] == 7 and "authority switch" in sent[0][1] and sent[0][2]["system_type"] == "rollback_notice"


def test_rollback_request_handler_reports_a_fence_refusal_with_its_blockers(monkeypatch):
    from ouroboros.gateway import control
    from supervisor.events_runtime_controls import _handle_rollback_request

    sent, done = [], threading.Event()
    monkeypatch.setattr(control, "rollback_through_door",
                        lambda target, *, door, request_restart: (409, {"error": "writers active", "blockers": ["active:x"]}))
    ctx = SimpleNamespace(request_restart=lambda: pytest.fail("no restart on a refusal"),
                          load_state=lambda: {"owner_chat_id": 7},
                          send_with_budget=lambda chat_id, text, **kw: sent.append(text) or done.set())
    _handle_rollback_request({"type": "rollback_request", "target": "v7.6.0"}, ctx)
    assert done.wait(10)
    assert "refused" in sent[0] and "active:x" in sent[0]


def test_rollback_request_is_registered_as_a_supervisor_event():
    from supervisor.event_taxonomy import EVENT_DISPOSITIONS
    from supervisor.events import EVENT_HANDLERS

    assert EVENT_HANDLERS["rollback_request"].__module__ == "supervisor.events_runtime_controls"
    assert EVENT_DISPOSITIONS["rollback_request"].answered_by == "supervisor.events_runtime_controls"


# ---- the boot-smoke rollback and Update replace --------------------------------------------------

def test_boot_smoke_rollback_records_the_switch_for_a_store_less_pre_update_tree(tmp_path, monkeypatch, target_repo):
    import supervisor.update_merge as update_merge

    repo, old, new = target_repo
    root = tmp_path / "data"
    _wire_git_ops(monkeypatch, repo, root)
    resets = []
    monkeypatch.setattr(update_merge, "rollback_managed_update",
                        lambda reason, *, reopen_writer_admission: resets.append((reason, reopen_writer_admission)) or (True, "reset"))
    tx = {"phase": "pending_boot_smoke", "pre_update_sha": old, "pre_update_branch": "ouroboros"}
    assert update_merge._boot_smoke_rollback(tx, "post_boot_smoke_failed")[0] is True
    intent = store_switch.read_switch_intent(root)
    assert intent["door"] == "boot_smoke_rollback" and intent["target_sha"] == old and intent["target_ref"] == "ouroboros"
    assert intent["reason"] == "post_boot_smoke_failed" and resets == []
    # The second failure before the restart is a refusal, not a second intent.
    ok, message = update_merge._boot_smoke_rollback(tx, "post_boot_smoke_failed")
    assert ok is False and "intent" in message
    # A store-bearing pre-update tree resets now, as before the store; a missing SHA is the owner's own refusal.
    assert update_merge._boot_smoke_rollback({"pre_update_sha": new}, "x") == (True, "reset")
    assert update_merge._boot_smoke_rollback({}, "y") == (True, "reset")
    assert resets == [("x", False), ("y", False)]


def test_update_replace_to_a_store_less_tree_defers_the_checkout_and_skips_the_pre_restart_smoke(tmp_path, monkeypatch, target_repo):
    import supervisor.git_ops as git_ops
    import supervisor.update_merge as update_merge
    from ouroboros.gateway import control
    from tests.test_update_apply_routing import BASE, _Request, _plan

    repo, old, new = target_repo
    root = tmp_path / "data"
    _wire_git_ops(monkeypatch, repo, root)
    writes = []
    plans = iter([_plan(target_sha=old), _plan(target_sha=old)])
    monkeypatch.setattr(git_ops, "BRANCH_DEV", "ouroboros", raising=False)
    monkeypatch.setattr(git_ops, "_write_update_intent", lambda _intent: None)
    monkeypatch.setattr(git_ops, "checkout_and_reset", lambda *_a, **_k: pytest.fail("the checkout moves at the restart exit"))
    monkeypatch.setattr(git_ops, "prepare_managed_update", lambda *_a, **_k: (True, {"update_intent": {"target_sha": old}}))
    monkeypatch.setattr(update_merge, "plan_managed_update_merge", lambda **_k: next(plans))
    monkeypatch.setattr(update_merge, "acquire_update_lock", lambda: object())
    monkeypatch.setattr(update_merge, "release_update_lock", lambda _lock: None)
    monkeypatch.setattr(update_merge, "active_update_tx", lambda: {})
    monkeypatch.setattr(update_merge, "write_update_tx", lambda tx: writes.append(dict(tx)))
    monkeypatch.setattr(update_merge, "update_restart_smoke", lambda: pytest.fail("no pre-restart smoke of a tree that is not here"))
    monkeypatch.setattr(control, "_quiesce_repo_writers", lambda _reason: [])
    response = control._apply_replace_recovery_fenced(_Request(), expected_base_sha=BASE, expected_target_sha=old)
    assert response.status_code == 200, response.body
    assert [tx["phase"] for tx in writes][-1] == "pending_boot_smoke" and "passed" not in {tx.get("pre_restart_smoke") for tx in writes}
    intent = store_switch.read_switch_intent(root)
    assert intent["door"] == "update_replace" and intent["target_sha"] == old and intent["reason"] == "ui_update_apply"
    assert git(repo, "rev-parse", "HEAD") == new
