"""Serve first, admit the dialogue before repairs, phases instead of timers.

The lifespan yields before any data-, extension- or child-proportional work; those run as
named steps of the boot coordinator while ``/api/health`` already answers with the ``boot``
document. The chat bridge is published at the store's publication, so an owner message is
accepted durably while the supervisor is still recovering, and a pooled task start still
waits for ``tasks_ready``. The supervisor's first tick waits for the extensions' subscribers
through an explicit event, not by blocking HTTP.
"""

from __future__ import annotations

import threading
import time

import pytest


CONTRACT_KEYS = {"phase", "since", "dialogue_ready", "tasks_ready", "steps", "jobs", "error", "supervisor_load"}


def _wait(predicate, timeout=20.0, *, what="condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


# --- the document ---------------------------------------------------------------------------

def test_the_boot_document_is_the_shared_contract_and_its_phases_follow_the_facts():
    from ouroboros.server_liveness import BootDocument

    doc = BootDocument()
    first = doc.snapshot()
    assert set(first) == CONTRACT_KEYS and first["phase"] == "starting" and first["steps"] == [] and first["jobs"] == {}
    doc.serving()
    assert doc.snapshot()["phase"] == "serving"
    with doc.step("one"):
        running = doc.snapshot()["steps"]
        assert running == [{"name": "one", "state": "running", "started_at": running[0]["started_at"], "ended_at": None, "error": None}]
    with pytest.raises(RuntimeError):
        with doc.step("two"):
            raise RuntimeError("boom")
    steps = {row["name"]: row for row in doc.snapshot()["steps"]}
    assert steps["one"]["state"] == "done" and steps["one"]["ended_at"]
    assert steps["two"] == {**steps["two"], "state": "failed", "error": "RuntimeError: boom"}, "a failed step keeps its error"
    assert doc.snapshot()["phase"] == "serving", "a fail-soft step does not fail the phase"
    doc.steps_done()
    assert doc.snapshot()["phase"] == "serving", "steps alone are not readiness"
    assert doc.snapshot(supervisor_done=True, tasks_ready=True)["phase"] == "ready"
    assert doc.snapshot(supervisor_done=True, supervisor_error="Supervisor init failed: x")["phase"] == "failed"
    doc.dialogue_ready()
    doc.fail("current-facts store not published: failed")
    final = doc.snapshot(supervisor_done=True)
    assert final["phase"] == "failed" and final["error"].startswith("current-facts store") and final["dialogue_ready"] is True
    doc.reset()
    assert doc.snapshot()["phase"] == "starting" and doc.snapshot()["error"] is None


def test_job_facts_read_nothing_from_an_unpublished_or_absent_store(tmp_path):
    from ouroboros import maintenance_jobs, store
    from ouroboros.server_liveness import job_facts

    assert job_facts(tmp_path / "nowhere") == {}
    root = tmp_path / "root"
    root.mkdir()
    maintenance_jobs.record(root, "legacy_reconcile", state="running", done=3, total=10)
    recorded = {"state": "running", "done": 3, "total": 10}
    assert job_facts(root)["legacy_reconcile"] == recorded  # beside the store's own import job
    store.begin_import(root)  # the import window: a read must neither wait nor import
    try:
        assert job_facts(root).get("legacy_reconcile", recorded) == recorded
    finally:
        store.end_import(root)


def test_the_coordinator_records_every_step_and_stops_between_steps_only():
    from ouroboros.server_liveness import BootDocument, run_boot_steps

    doc, stop, order = BootDocument(), threading.Event(), []
    doc.serving()

    def failing():
        order.append("b")
        raise ValueError("bad")

    run_boot_steps([("a", lambda: order.append("a")), ("b", failing), ("c", lambda: order.append("c"))], stop=stop, document=doc)
    assert order == ["a", "b", "c"], "a failed step never hides the next one"
    states = {row["name"]: (row["state"], row["error"]) for row in doc.snapshot()["steps"]}
    assert states == {"a": ("done", None), "b": ("failed", "ValueError: bad"), "c": ("done", None)}
    assert doc.snapshot(supervisor_done=True)["phase"] == "ready"

    doc, order = BootDocument(), []
    doc.serving()
    run_boot_steps([("a", lambda: (order.append("a"), stop.set())), ("b", lambda: order.append("b"))], stop=stop, document=doc)
    assert order == ["a"], "a stop set during a step ends the walk before the next step"
    assert doc.snapshot(supervisor_done=True)["phase"] == "serving", "a stopped boot never claims its steps are done"


# --- the lifespan ---------------------------------------------------------------------------


def test_real_ui_assets_survive_same_process_child_pressure(boot_app, monkeypatch):
    """Actual routes and StaticFiles worker-thread path, not a replacement HTML app."""
    import os
    import pathlib
    import subprocess
    import sys
    import time
    from starlette.testclient import TestClient
    from ouroboros import platform_layer
    from ouroboros.tools import shell_process as sp
    from tests._shared import wait_test_child_stop

    if os.name == "nt":
        pytest.skip("real POSIX retained leaders; Windows Job tests cover that platform")
    srv, _root = boot_app
    owned = []
    stylesheet = pathlib.Path(__file__).resolve().parents[1] / "web" / "style.css"

    def check(client):
        page = client.get("/")
        assert page.status_code == 200 and "style.css" in page.text
        css = client.get("/static/style.css")
        assert css.status_code == 200 and css.content == stylesheet.read_bytes() and css.content

    with TestClient(srv.app) as client:
        check(client)
        try:
            for _ in range(32):
                with sp.tracked_start() as publish:
                    proc = publish(sp.contained_popen(subprocess.Popen,
                        ["sh", "-c", "sleep 60 >/dev/null 2>&1 & echo $!; exit 7"],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True))
                child = int(proc.stdout.readline())
                owned.append((proc, child))
                assert proc.wait(timeout=5) == 7
                assert not sp.retire_tracked(proc)  # each root leaves a live descendant
            check(client)
            for proc, child in owned:
                platform_layer.request_process_tree_kill(proc)
                assert wait_test_child_stop(child, timeout_sec=5)["child_stopped"]
            # The new launch is in this application process, not a remote test controller.
            deadline = time.monotonic() + 5
            while True:
                assert sp._tracked_subprocess_run([sys.executable, "-c", "pass"], timeout=5).returncode == 0
                if not sys.platform.startswith("linux") or all(proc not in sp._active_subprocesses for proc, _ in owned):
                    break
                assert time.monotonic() < deadline, "spawn retries did not retire quiet trees"
            if sys.platform.startswith("linux"):
                for proc, _ in owned:
                    assert proc not in sp._active_subprocesses and proc.wait() == 7
                    with pytest.raises(ChildProcessError):
                        platform_layer.observe_child_exit(proc.pid)
            check(client)
        finally:
            for proc, child in owned:
                platform_layer.request_process_tree_kill(proc)
                assert wait_test_child_stop(child, timeout_sec=5)["child_stopped"]
                assert sp.retire_tracked(proc)
                proc.stdout.close()
                proc.stderr.close()


@pytest.fixture
def boot_app(tmp_path, monkeypatch, request):
    """The real ASGI app on a temp root, the lifespan's patch helper of the extension tests."""
    import server as srv
    from tests.test_extensions_api import _patch_lifespan_for_drive_root_test

    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    srv.app.app.state.drive_root = drive_root
    srv.app.app.state.repo_dir = tmp_path / "repo"
    _patch_lifespan_for_drive_root_test(monkeypatch, srv, {}, request)
    monkeypatch.setattr(srv, "_boot_managed_update_tasks", lambda: None)
    monkeypatch.setattr("ouroboros.config.get_skills_repo_path", lambda: "")
    return srv, drive_root


def _health(client):
    response = client.get("/api/health")
    assert response.status_code == 200, response.text
    return response.json()


def _ready_supervisor(srv) -> bool:
    """A supervisor whose init ended at once: the readiness latches the real one sets."""
    srv._supervisor_ready.set()
    srv._supervisor_init_done.set()
    return True


def test_health_answers_with_the_phase_while_slow_boot_steps_still_run(boot_app, monkeypatch):
    from starlette.testclient import TestClient

    from ouroboros import extension_loader

    srv, drive_root = boot_app
    stops_release, stops_entered = threading.Event(), threading.Event()
    reload_release, reload_entered = threading.Event(), threading.Event()

    def slow_stops(_root):
        stops_entered.set()
        assert stops_release.wait(30)
        return {"retried": 0}

    def slow_reload(_root, _reader, *, repo_path=None):
        reload_entered.set()
        assert reload_release.wait(30)
        return {}

    monkeypatch.setattr(srv, "finish_unconfirmed_stops", slow_stops)
    monkeypatch.setattr(extension_loader, "reload_all", slow_reload)
    monkeypatch.setattr(srv, "has_startup_ready_provider", lambda _settings: True)
    monkeypatch.setattr(srv, "_start_supervisor_if_needed", lambda _settings: _ready_supervisor(srv))
    try:
        with TestClient(srv.app) as client:
            assert stops_entered.wait(10), "the boot coordinator runs the first step"
            body = _health(client)
            boot = body["boot"]
            assert set(boot) == CONTRACT_KEYS and body["status"] == "ok" and body["runtime_version"]
            assert boot["phase"] == "serving" and boot["error"] is None
            assert [row["name"] for row in boot["steps"]] == ["owned_stops"]
            assert boot["steps"][0]["state"] == "running" and boot["steps"][0]["ended_at"] is None
            assert boot["tasks_ready"] is False
            stops_release.set()
            assert reload_entered.wait(10), "the steps run in order up to the extension reload"
            # The store publishes on its own thread: the dialogue opens while the reload still runs.
            boot = _wait(lambda: (b := _health(client)["boot"])["dialogue_ready"] and b, what="dialogue_ready during the reload")
            states = {row["name"]: row["state"] for row in boot["steps"]}
            assert states["owned_stops"] == "done" and states["extensions"] == "running"
            assert boot["phase"] == "serving", "the dialogue opened at the store's publication, before the slow steps ended"
            assert boot["tasks_ready"] is False, "readiness waits for the extensions (the first tick's subscribers)"
            reload_release.set()
            final = _wait(lambda: (b := _health(client)["boot"])["phase"] == "ready" and b, what="phase ready")
            states = {row["name"]: row["state"] for row in final["steps"]}
            assert states["extensions"] == "done" and states["store"] == "done" and states["supervisor_outcome"] == "done"
            assert final["tasks_ready"] is True and final["dialogue_ready"] is True
            state = client.get("/api/state").json()
            assert state["supervisor_ready"] is True and state["boot"]["phase"] == "ready"
    finally:
        stops_release.set()
        reload_release.set()


def test_an_owner_message_is_accepted_durably_while_recovery_still_runs_and_a_task_start_still_waits(boot_app, monkeypatch):
    """The supervisor generation (started at publication, after the extensions) is still in its
    startup recovery: the dialogue the boot opened accepts the owner's message durably, while a
    pooled task start answers 503 until the generation's readiness latch."""
    from starlette.testclient import TestClient

    from supervisor import message_bus

    srv, drive_root = boot_app
    recovery_release, recovery_entered = threading.Event(), threading.Event()

    def supervisor_in_recovery(_settings):
        def generation():
            assert srv._dialogue_bridge is not None, "the supervisor attaches to the dialogue the boot opened"
            recovery_entered.set()
            if recovery_release.wait(30):
                _ready_supervisor(srv)

        threading.Thread(target=generation, name="test-supervisor-recovery", daemon=True).start()
        return True

    monkeypatch.setattr(srv, "has_startup_ready_provider", lambda _settings: True)
    monkeypatch.setattr(srv, "_start_supervisor_if_needed", supervisor_in_recovery)
    from tests._dialogue import chat_rows

    def stored():  # the canonical rows live in state/dialog.sqlite
        try:
            return [row for row in chat_rows(srv.DATA_DIR) if row.get("client_message_id") == "boot-readiness-1"]
        except Exception:
            return []

    before = len(stored())
    try:
        with TestClient(srv.app) as client:
            assert recovery_entered.wait(10), "the supervisor started at publication, after the extensions"
            boot = _health(client)["boot"]
            assert boot["dialogue_ready"] is True and boot["tasks_ready"] is False and boot["phase"] == "serving"
            assert message_bus.try_get_bridge() is not None
            with client.websocket_connect("/ws") as socket:
                socket.send_json({"type": "chat", "content": "hello before repairs", "chat_id": 1,
                                  "client_message_id": "boot-readiness-1", "sender_session_id": "s1"})
                echo = socket.receive_json()
            assert echo["type"] == "chat" and echo["role"] == "user" and echo["ingress_accepted"] is True
            assert echo["client_message_id"] == "boot-readiness-1"
            rows = stored()
            assert len(rows) - before == 1 and rows[-1]["text"] == "hello before repairs", "the canonical row is written"
            [update] = message_bus.try_get_bridge().get_updates(0, timeout=1)
            assert update["message"]["text"] == "hello before repairs", "the turn waits for its executor in the bridge"
            refused = client.post("/api/tasks", json={"description": "pooled work"})
            assert refused.status_code == 503 and "still starting" in refused.text
            recovery_release.set()
            _wait(lambda: _health(client)["boot"]["tasks_ready"], what="tasks_ready after recovery")
    finally:
        recovery_release.set()


def test_the_supervisor_start_waits_for_the_extensions_event_and_the_coordinator_sets_it(boot_app, monkeypatch):
    """Both directions of the ordering guard, at the lifespan: the supervisor start is not called
    while the extension reload still runs, and it is called once the reload ended."""
    from starlette.testclient import TestClient

    from ouroboros import extension_loader

    srv, drive_root = boot_app
    reload_release, reload_entered, started = threading.Event(), threading.Event(), threading.Event()

    def slow_reload(_root, _reader, *, repo_path=None):
        reload_entered.set()
        assert reload_release.wait(30)
        return {}

    monkeypatch.setattr(extension_loader, "reload_all", slow_reload)
    monkeypatch.setattr(srv, "has_startup_ready_provider", lambda _settings: True)
    monkeypatch.setattr(srv, "_start_supervisor_if_needed", lambda _settings: started.set() or True)
    try:
        with TestClient(srv.app) as client:
            assert reload_entered.wait(10)
            _wait(lambda: _health(client)["boot"]["dialogue_ready"], what="dialogue_ready during the reload")
            assert not started.wait(0.5), "the supervisor must not start before the extensions registered"
            assert srv._extensions_ready.is_set() is False
            reload_release.set()
            assert started.wait(10), "the supervisor starts once the extensions ended"
            assert srv._extensions_ready.is_set() is True
    finally:
        reload_release.set()


def test_the_generation_thread_itself_waits_for_the_extensions_event(monkeypatch):
    """The explicit guard: a generation started by any caller (a settings save during boot)
    enters the supervisor only after ``_extensions_ready``; removing the wait fires this."""
    import server as srv

    entered = threading.Event()
    monkeypatch.setattr(srv, "_run_supervisor", lambda _settings: entered.set())
    monkeypatch.setattr(srv, "has_startup_ready_provider", lambda _settings: True)
    monkeypatch.setattr(srv, "_supervisor_thread", None)
    srv._extensions_ready.clear()
    try:
        assert srv._start_supervisor_if_needed({}) is True
        assert not entered.wait(0.5), "the generation waited for the extensions"
        srv._extensions_ready.set()
        assert entered.wait(5), "the generation runs once the extensions registered"
    finally:
        srv._extensions_ready.set()
        thread = srv._supervisor_thread
        if thread is not None:
            thread.join(timeout=5)


# --- the notice a frame gets before the dialogue opened -------------------------------------

def test_the_initialization_notice_says_why_the_message_was_not_saved(monkeypatch):
    import types

    from starlette.applications import Starlette
    from starlette.routing import WebSocketRoute
    from starlette.testclient import TestClient

    from ouroboros.gateway.ws import ws_endpoint
    from ouroboros.server_liveness import BOOT, boot_notice
    from supervisor import message_bus

    monkeypatch.setattr(message_bus, "try_get_bridge", lambda: None)
    BOOT.reset()
    BOOT.serving()
    state = types.SimpleNamespace(supervisor_ready_event=threading.Event(), supervisor_init_done_event=threading.Event(),
                                  get_supervisor_error=lambda: None)
    assert "still being prepared" in boot_notice(state) and "not saved" in boot_notice(state)
    app = Starlette(routes=[WebSocketRoute("/ws", ws_endpoint)])
    app.state.supervisor_ready_event = state.supervisor_ready_event
    app.state.supervisor_init_done_event = state.supervisor_init_done_event
    app.state.get_supervisor_error = state.get_supervisor_error
    try:
        with TestClient(app) as client, client.websocket_connect("/ws") as socket:
            socket.send_json({"type": "chat", "content": "hello", "chat_id": 3, "client_message_id": "cm-early"})
            reply = socket.receive_json()
        assert reply["system_type"] == "initialization_notice" and reply["client_message_id"] == "cm-early"
        assert reply["chat_id"] == 3 and "still being prepared" in reply["content"]
        state.get_supervisor_error = lambda: "Supervisor init failed: boot dependency refused"
        notice = boot_notice(state)
        assert notice.startswith("⚠️ Startup failed") and "boot dependency refused" in notice
    finally:
        BOOT.reset()


def test_the_dialogue_cutover_runs_before_the_dialogue_opens_and_a_failure_stays_a_step(tmp_path, monkeypatch):
    import inspect

    import server
    from ouroboros import dialogue_import
    from ouroboros.server_liveness import BOOT

    source = inspect.getsource(server.lifespan)
    assert source.index("_dialogue_cutover(lifespan_drive_root)") < source.index("_open_dialogue(settings")
    # The lifespan's own steps are the boot owners that choose an existing store's journal (store_journal).
    assert "usage_store.migrate_from_journal(lifespan_drive_root, initialize_journal=True)" in source
    BOOT.reset()
    calls = []
    monkeypatch.setattr(dialogue_import, "cutover", lambda root, **kwargs: calls.append((root, kwargs)))
    server._dialogue_cutover(tmp_path)
    steps = {row["name"]: row for row in BOOT.snapshot()["steps"]}
    assert calls == [(tmp_path, {"initialize_journal": True})] and steps["dialogue_cutover"]["state"] == "done"

    def held(_root, **_kwargs):
        raise RuntimeError("append lock held")

    monkeypatch.setattr(dialogue_import, "cutover", held)
    server._dialogue_cutover(tmp_path)  # never raises: the dialogue still opens
    steps = {row["name"]: row for row in BOOT.snapshot()["steps"]}
    assert steps["dialogue_cutover"]["state"] == "failed" and "append lock held" in steps["dialogue_cutover"]["error"]
    BOOT.reset()


@pytest.mark.parametrize("busy", [True, False])
def test_a_busy_dialogue_store_is_named_instead_of_initialization(busy):
    """A refused acceptance caused by a busy dialogue store says so; any other failure keeps the generic notice."""
    import asyncio
    import json as _json

    from ouroboros.gateway.ws import _accept_chat_after
    from ouroboros.store import StoreBusy

    sent = []

    class Socket:
        async def send_text(self, text):
            sent.append(_json.loads(text))

    def accept():
        cause = StoreBusy("dialogue store busy", reason="contention") if busy else OSError("disk")
        raise RuntimeError("canonical message acceptance could not be persisted") from cause

    asyncio.run(_accept_chat_after(Socket(), None, accept, client_message_id="cm-1", chat_id=3))
    assert sent and sent[0]["client_message_id"] == "cm-1" and sent[0]["system_type"] == "initialization_notice"
    assert ("dialogue store is busy" in sent[0]["content"]) is busy
    assert ("still initializing" in sent[0]["content"]) is (not busy)


# --- transport preparation overlaps registration, never recovery ----------------------------

@pytest.mark.parametrize("prewarm", [False, True])
def test_native_transport_overlap_keeps_extension_delivery_and_shutdown(boot_app, monkeypatch, prewarm):
    """Same native manager and three-second registration in both schedules; no provider calls.

    False is the old sequential schedule, True the actual lifespan's new schedule. Only the
    supervisor's costly side effects are substituted: its real admission callback stays behind
    extension registration. The manager/queue RPC, bus subscriber and shutdown are real.
    """
    import json
    from starlette.testclient import TestClient
    from ouroboros import extension_loader, server_boot
    from ouroboros.event_bus import EventBus, CHAT_OUTBOUND
    from supervisor import workers

    srv, root = boot_app
    monkeypatch.setattr(workers, "_EVENT_Q", None)
    monkeypatch.setattr(workers, "_EVENT_Q_MANAGER", None)
    monkeypatch.setattr(workers, "_EVENT_Q_SHUTDOWN", False)
    bus, received, stamps, managers = EventBus(), [], {}, []
    get_queue = workers.get_event_q
    def measured_queue(**kwargs):
        first = workers._EVENT_Q_MANAGER is None
        if first:
            stamps["queue_enter"] = time.monotonic()
        result = get_queue(**kwargs)
        if first:
            stamps["queue_exit"] = time.monotonic()
        return result
    monkeypatch.setattr(workers, "get_event_q", measured_queue)
    entered = threading.Event()
    prepare = server_boot.prepare_event_queue
    if not prewarm:
        monkeypatch.setattr(server_boot, "prepare_event_queue", lambda *_a, **_kw: None)

    def reload(_root, _reader, **_kw):
        stamps["extensions_enter"] = time.monotonic()
        entered.set()
        time.sleep(3)
        bus.subscribe("saved-work-consumer", CHAT_OUTBOUND, received.append)
        stamps["extensions_exit"] = time.monotonic()
        return {}

    def supervisor(_settings):
        stamps["supervisor_enter"] = time.monotonic()
        assert srv._extensions_ready.is_set()
        assert server_boot.wait_event_queue(srv._event_queue_preparation, lambda: False)
        queue = workers.get_event_q(drive_root=root)
        managers.append(workers._EVENT_Q_MANAGER)
        stamps["queue_observed"] = time.monotonic()
        queue.put({"type": "send_message", "text": "saved terminal"})
        bus.publish(CHAT_OUTBOUND, queue.get(timeout=5))
        _ready_supervisor(srv)
        return True

    monkeypatch.setattr(extension_loader, "reload_all", reload)
    monkeypatch.setattr(srv, "has_startup_ready_provider", lambda _: True)
    monkeypatch.setattr(srv, "_start_supervisor_if_needed", supervisor)
    with TestClient(srv.app) as client:
        assert entered.wait(15)
        if prewarm:
            _wait(lambda: srv._event_queue_preparation is not None, what="preparation published")
            srv._event_queue_preparation.result(timeout=15)
            stamps["preparation_observed"] = time.monotonic()
            assert stamps["queue_enter"] < stamps.get("extensions_exit", float("inf")), "native creation starts during registration"
            assert stamps["queue_exit"] > stamps["extensions_enter"], "the actual creation interval overlaps registration"
            manager = workers._EVENT_Q_MANAGER
            assert workers.get_event_q() is srv._event_queue_preparation.result(), "warm read keeps the endpoint"
            assert manager._process.is_alive()
        _wait(lambda: _health(client)["boot"]["phase"] == "ready", what="ready after transport and registration")
        assert len(received) == 1 and received[0]["text"] == "saved terminal"
        assert stamps["supervisor_enter"] >= stamps["extensions_exit"]
        rows = [json.loads(line) for line in (root / "logs" / "supervisor.jsonl").read_text(encoding="utf-8").splitlines()]
        creation = next(row for row in rows if row.get("type") == "event_queue_generation_started")
        elapsed = stamps["queue_observed"] - stamps["extensions_enter"]
        print("BOOT_NATIVE_MEASUREMENT " + json.dumps({"prewarm": prewarm, "registration_seconds": 3,
            "ready_after_registration_enter_seconds": elapsed, "manager_pid": managers[0]._process.pid,
            "manager_start_wall_seconds": creation["manager_start_wall_seconds"],
            "manager_start_cpu_seconds": creation["manager_start_cpu_seconds"],
            "queue_create_wall_seconds": creation["queue_create_wall_seconds"],
            "queue_create_cpu_seconds": creation["queue_create_cpu_seconds"], "stamps": stamps}))
    assert managers and not managers[0]._process.is_alive(), "actual lifespan shutdown joined the original manager"
    monkeypatch.setattr(server_boot, "prepare_event_queue", prepare)


def test_preparation_stop_before_publication_or_while_waiting_for_lock(tmp_path, monkeypatch):
    from concurrent.futures import wait
    from ouroboros import server_boot
    from ouroboros.server_liveness import BootDocument
    from supervisor import workers, event_observation

    stop, waiting = threading.Event(), threading.Event()
    def importing(*_a, **_kw):
        waiting.set()
        stop.wait(0.01)
        return {"status": "importing"}
    monkeypatch.setattr(server_boot.store_switch, "wait_ready", importing)
    monkeypatch.setattr(event_observation, "EventQueueManager", lambda **_: pytest.fail("no create after Stop"))
    future = server_boot.prepare_event_queue(tmp_path, stop.is_set, BootDocument())
    assert waiting.wait(3)
    stop.set()
    _wait(future.done, timeout=3, what="cancelled preparation thread")
    assert future.cancelled()
    assert server_boot.wait_event_queue(future, stop.is_set) is False

    stop.clear()
    waiting.clear()
    monkeypatch.setattr(server_boot.store_switch, "wait_ready", lambda *_a, **_kw: {"status": "published"})
    monkeypatch.setattr(workers, "_EVENT_Q", None)
    monkeypatch.setattr(workers, "_EVENT_Q_MANAGER", None)
    monkeypatch.setattr(workers, "_EVENT_Q_SHUTDOWN", False)
    with workers._EVENT_Q_LOCK:
        future = server_boot.prepare_event_queue(tmp_path, stop.is_set, BootDocument())
        stop.set()
    wait([future], timeout=3)
    assert future.done()
    assert server_boot.wait_event_queue(future, stop.is_set) is False
    assert workers._EVENT_Q_MANAGER is None


@pytest.mark.parametrize("fail_at", ["start", "queue", "stop_during_start"])
def test_partial_transport_attempt_stays_owned_and_never_retries(tmp_path, monkeypatch, fail_at):
    from ouroboros import server_boot
    from ouroboros.server_liveness import BootDocument
    from supervisor import workers, event_observation

    stop, calls = threading.Event(), []
    class Process:
        pid = 0
        exitcode = None
        def kill(self):
            calls.append("kill")
            self.exitcode = -1
        def is_alive(self):
            return self.exitcode is None
        def join(self, *_a):
            calls.append("join")
    class Manager:
        _process = Process()
        def __init__(self, **_kw):
            calls.append("construct")
        def start(self, **_kw):
            calls.append("start")
            if fail_at == "start":
                self._process.exitcode = 1
                raise TimeoutError("manager start failed")
            if fail_at == "stop_during_start":
                stop.set()  # models the Panic latch independently of the shutdown lock
        def Queue(self):
            calls.append("queue")
            raise OSError("queue RPC failed")
        def shutdown(self):
            calls.append("shutdown")
            self._process.exitcode = 0
    monkeypatch.setattr(workers, "_EVENT_Q", None)
    monkeypatch.setattr(workers, "_EVENT_Q_MANAGER", None)
    monkeypatch.setattr(workers, "_EVENT_Q_SHUTDOWN", False)
    monkeypatch.setattr(event_observation, "EventQueueManager", Manager)
    monkeypatch.setattr(server_boot.store_switch, "wait_ready", lambda *_a, **_kw: {"status": "published"})
    doc = BootDocument()
    future = server_boot.prepare_event_queue(tmp_path, stop.is_set, doc)
    from concurrent.futures import wait
    assert wait([future], timeout=3).done
    manager = workers._EVENT_Q_MANAGER
    if fail_at == "stop_during_start":
        assert server_boot.wait_event_queue(future, stop.is_set) is False
        assert "kill" in calls and "queue" not in calls
    else:
        for _ in range(2):
            with pytest.raises((OSError, TimeoutError), match="failed"):
                server_boot.wait_event_queue(future, lambda: False)
        with pytest.raises(RuntimeError, match="initialization did not finish"):
            workers.get_event_q(drive_root=tmp_path)
    assert calls.count("construct") == calls.count("start") == 1
    assert manager is not None and doc.snapshot()["steps"][0]["state"] == "failed"
    workers.shutdown_event_q()
    assert "shutdown" in calls and not manager._process.is_alive()


def test_preparation_thread_start_failure_is_retained(tmp_path, monkeypatch):
    from ouroboros import server_boot
    def refused(**_kw):
        raise RuntimeError("thread unavailable")
    monkeypatch.setattr(server_boot.threading, "Thread", refused)
    future = server_boot.prepare_event_queue(tmp_path, lambda: False)
    with pytest.raises(RuntimeError, match="thread unavailable"):
        server_boot.wait_event_queue(future, lambda: False)


def test_providerless_boot_does_not_prepare_a_transport(boot_app, monkeypatch):
    from starlette.testclient import TestClient
    srv, _root = boot_app
    monkeypatch.setattr(srv.server_boot, "prepare_event_queue", lambda *_a, **_kw: pytest.fail("providerless transport"))
    with TestClient(srv.app) as client:
        _wait(lambda: _health(client)["boot"]["phase"] == "ready")
        assert srv._event_queue_preparation is None


def test_real_supervisor_retains_preparation_failure_and_runs_recovery_owner(tmp_path, monkeypatch):
    from concurrent.futures import Future
    import server
    from supervisor import workers
    from tests.test_server_shutdown import _supervisor_harness

    _supervisor_harness(monkeypatch, tmp_path, ["stop"])
    future = Future()
    future.set_exception(OSError("native queue failed"))
    monkeypatch.setattr(server, "_event_queue_preparation", future)
    recovered = []
    monkeypatch.setattr(server, "_run_startup_task_recovery", lambda *_a, **kw: recovered.append(kw))
    monkeypatch.setattr(workers, "init", lambda **_: pytest.fail("failed preparation cannot start workers"))
    for _ in range(2):  # a settings revival observes the same failure, never a new queue attempt
        server._run_supervisor({})
        assert server._supervisor_error == "Supervisor init failed: native queue failed"
        assert not server._supervisor_ready.is_set() and server._supervisor_init_done.is_set()
    assert len(recovered) == 2 and future.exception().args == ("native queue failed",)


def test_stop_during_queue_rpc_and_concurrent_first_callers_share_the_attempt(tmp_path, monkeypatch):
    from concurrent.futures import wait
    from types import SimpleNamespace
    from ouroboros import server_boot
    from ouroboros.server_liveness import BootDocument
    from supervisor import workers, event_observation

    stop, entered, release = threading.Event(), threading.Event(), threading.Event()
    calls, returned, endpoint = [], [], object()
    class Manager:
        _process = SimpleNamespace(pid=0, exitcode=None, is_alive=lambda: False)
        def __init__(self, **_kw):
            calls.append("construct")
        def start(self, **_kw):
            calls.append("start")
        def Queue(self):
            calls.append("queue")
            entered.set()
            assert release.wait(5)
            return endpoint
        def shutdown(self):
            calls.append("shutdown")
    for name, value in (("_EVENT_Q", None), ("_EVENT_Q_MANAGER", None), ("_EVENT_Q_SHUTDOWN", False)):
        monkeypatch.setattr(workers, name, value)
    monkeypatch.setattr(event_observation, "EventQueueManager", Manager)
    monkeypatch.setattr(server_boot.store_switch, "wait_ready", lambda *_a, **_kw: {"status": "published"})
    future = server_boot.prepare_event_queue(tmp_path, stop.is_set, BootDocument())
    assert entered.wait(3)
    legacy = threading.Thread(target=lambda: returned.append(workers.get_event_q()), daemon=True)
    legacy.start()
    stop.set()
    try:
        assert server_boot.wait_event_queue(future, stop.is_set) is False
        assert not future.done(), "Stop returned while the queue RPC is still held"
    finally:
        release.set()
        legacy.join(5)
        assert wait([future], timeout=5).done
        workers.shutdown_event_q()
    assert future.result() is endpoint and returned == [endpoint]
    assert calls == ["construct", "start", "queue", "shutdown"]
