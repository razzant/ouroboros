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
def native_head_start():
    """The real ``server_boot.prepare_event_queue``, read before ``boot_app`` replaces it."""
    from ouroboros import server_boot

    return server_boot.prepare_event_queue


@pytest.fixture
def boot_app(tmp_path, monkeypatch, request, native_head_start):
    """The real ASGI app on a temp root, the lifespan's patch helper of the extension tests.

    Its supervisor is each test's stand-in, so a provider-ready lifespan here starts no native event-bus
    manager the stand-in would never use: a test of that transport restores ``native_head_start``. Each
    lifespan begins with no earlier head start to wait for, and leaves none behind."""
    import server as srv
    from tests.test_extensions_api import _patch_lifespan_for_drive_root_test

    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    srv.app.app.state.drive_root = drive_root
    srv.app.app.state.repo_dir = tmp_path / "repo"
    _patch_lifespan_for_drive_root_test(monkeypatch, srv, {}, request)
    monkeypatch.setattr(srv, "_boot_managed_update_tasks", lambda: None)
    monkeypatch.setattr("ouroboros.config.get_skills_repo_path", lambda: "")
    monkeypatch.setattr(srv.server_boot, "prepare_event_queue", lambda *_a, **_kw: None)
    monkeypatch.setattr(srv.server_boot, "_head_start", None)
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


# --- the event bus's head start overlaps registration, never recovery ----------------------------

@pytest.fixture
def fresh_bus(monkeypatch):
    """This process's event bus and the boot's head start as a fresh server process has them."""
    from ouroboros import server_boot
    from supervisor import workers

    for name, value in (("_EVENT_Q", None), ("_EVENT_Q_MANAGER", None), ("_EVENT_Q_SHUTDOWN", False)):
        monkeypatch.setattr(workers, name, value)
    monkeypatch.setattr(server_boot, "_head_start", None)
    return workers


@pytest.mark.parametrize("prewarm", [False, True])
def test_native_transport_overlap_keeps_extension_delivery_and_shutdown(boot_app, monkeypatch, fresh_bus, native_head_start,
                                                                        prewarm):
    """The real manager and its queue RPC, bus subscriber and shutdown, in both schedules; no provider calls.

    False is the old sequential schedule, True the lifespan's: the head start's creation waits
    until registration began (the lifespan starts it a few steps earlier), and registration holds
    until creation began, so their overlap is an ordering fact, never a timing. Only the
    supervisor's costly side effects are substituted: its real admission callback stays behind
    extension registration.
    """
    from starlette.testclient import TestClient
    from ouroboros import extension_loader, server_boot
    from ouroboros.event_bus import EventBus, CHAT_OUTBOUND

    workers = fresh_bus
    srv, root = boot_app
    bus, received, order, managers = EventBus(), [], [], []
    creating, entered = threading.Event(), threading.Event()
    get_queue = workers.get_event_q

    def observed_queue(**kwargs):
        first = workers._EVENT_Q_MANAGER is None
        if first:
            if prewarm:  # the test's hold, never the product's: creation is recorded inside the registration
                assert entered.wait(15), "the extensions began to load while the head start waited"
            order.append("creation began")
            creating.set()
        result = get_queue(**kwargs)
        if first:
            order.append("creation ended")
        return result
    monkeypatch.setattr(workers, "get_event_q", observed_queue)
    if prewarm:  # the lifespan's own schedule; False keeps boot_app's sequential one
        monkeypatch.setattr(server_boot, "prepare_event_queue", native_head_start)

    def reload(_root, _reader, **_kw):
        order.append("registration began")
        entered.set()
        if prewarm:
            assert creating.wait(15), "the head start began while the extensions load"
        bus.subscribe("saved-work-consumer", CHAT_OUTBOUND, received.append)
        order.append("registration ended")
        return {}

    def supervisor(_settings):
        order.append("supervisor began")
        assert srv._extensions_ready.is_set()
        assert server_boot.wait_event_queue(lambda: False)
        queue = workers.get_event_q(drive_root=root)
        managers.append(workers._EVENT_Q_MANAGER)
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
            _wait(lambda: server_boot._head_start is not None, what="head start published")
            endpoint = server_boot._head_start.result(timeout=60)
            assert workers.get_event_q() is endpoint, "the supervisor's ordinary call returns the head start's queue"
        _wait(lambda: _health(client)["boot"]["phase"] == "ready", what="ready after transport and registration")
        assert len(received) == 1 and received[0]["text"] == "saved terminal"
        assert order.index("supervisor began") > order.index("registration ended")
        if prewarm:
            assert order.index("registration began") < order.index("creation began") < order.index("registration ended")
        else:
            assert order.index("creation began") > order.index("supervisor began")
        assert managers[0]._process.is_alive() and len(set(map(id, managers))) == 1
    assert not managers[0]._process.is_alive(), "the lifespan's shutdown joined the one manager"


def test_preparation_stop_before_publication_or_while_waiting_for_lock(tmp_path, monkeypatch, fresh_bus):
    from concurrent.futures import wait
    from ouroboros import server_boot
    from ouroboros.server_liveness import BootDocument
    from supervisor import event_observation

    workers = fresh_bus
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
    assert server_boot.wait_event_queue(stop.is_set) is False

    stop.clear()
    waiting.clear()
    monkeypatch.setattr(server_boot.store_switch, "wait_ready", lambda *_a, **_kw: {"status": "published"})
    with workers._EVENT_Q_LOCK:
        future = server_boot.prepare_event_queue(tmp_path, stop.is_set, BootDocument())
        stop.set()
    wait([future], timeout=3)
    assert future.done()
    assert server_boot.wait_event_queue(stop.is_set) is False
    assert workers._EVENT_Q_MANAGER is None


class _FakeManagers:
    """Manager doubles in creation order: the first one fails as ``fail_at`` says, later ones work.
    ``stuck`` keeps the first one's process alive through kill and join."""

    def __init__(self, fail_at, stop, *, stuck=False):
        from types import SimpleNamespace

        self.calls, self.made, self.endpoint = [], [], object()
        fakes = self

        class Process:
            def __init__(self, index):
                self.index, self.pid, self.exitcode = index, 0, None
            def kill(self):
                fakes.calls.append(f"kill:{self.index}")
                if not (stuck and self.index == 0):
                    self.exitcode = -9
            def is_alive(self):
                return self.exitcode is None
            def join(self, *_a):
                fakes.calls.append(f"join:{self.index}")

        class Manager:
            def __init__(self, **_kw):
                self.index = len(fakes.made)
                self._process = Process(self.index)
                fakes.made.append(self)
                fakes.calls.append(f"construct:{self.index}")
            def start(self, **_kw):
                fakes.calls.append(f"start:{self.index}")
                if self.index == 0 and fail_at == "start":
                    self._process.exitcode = 1
                    raise TimeoutError("manager start failed")
                if self.index == 0 and fail_at == "stop_during_start":
                    stop.set()  # models the Panic latch independently of the shutdown lock
            def Queue(self):  # noqa: N802 - the registered typeid
                fakes.calls.append(f"queue:{self.index}")
                if self.index == 0 and fail_at in ("queue", "stuck"):
                    raise OSError("queue RPC failed")
                return fakes.endpoint
            def shutdown(self):
                fakes.calls.append(f"shutdown:{self.index}")
                if not (stuck and self.index == 0):
                    self._process.exitcode = 0

        self.Manager = Manager
        self.namespace = SimpleNamespace


@pytest.mark.parametrize("custody_fails", [False, True])
def test_manager_custody_precedes_a_refused_queue_rpc(tmp_path, monkeypatch, fresh_bus, caplog, custody_fails):
    """A started manager has custody before Queue can fail; a refused ledger write remains explicit and fail-soft."""
    from ouroboros import process_custody
    from supervisor import event_observation

    fakes = _FakeManagers("queue", threading.Event())
    start = fakes.Manager.start
    records = []

    def started(manager, **kwargs):
        start(manager, **kwargs)
        manager._process.pid = 4242  # synthetic identity, only the recorder below sees it

    def record(root, **kwargs):
        fakes.calls.append("custody:0")
        records.append((root, kwargs))
        if custody_fails:
            raise OSError("ledger refused")

    monkeypatch.setattr(fakes.Manager, "start", started)
    monkeypatch.setattr(event_observation, "EventQueueManager", fakes.Manager)
    monkeypatch.setattr(process_custody, "record_process", record)
    with pytest.raises(OSError, match="queue RPC failed"):
        fresh_bus.get_event_q(drive_root=tmp_path)
    assert fakes.calls == ["construct:0", "start:0", "custody:0", "queue:0"]
    assert records == [(tmp_path, {"pid": 4242, "cmd": "multiprocessing SyncManager",
                                  "purpose": "supervisor_event_queue_manager", "scope": "session",
                                  "reap_process_group": False})]
    assert ("Failed to custody-track event queue manager" in caplog.text) is custody_fails
    assert fresh_bus._EVENT_Q is None and fresh_bus._EVENT_Q_MANAGER is fakes.made[0]
    fresh_bus.shutdown_event_q()
    assert not fakes.made[0]._process.is_alive(), "the unpublished attempt is still owned by normal teardown"


@pytest.mark.parametrize("fail_at", ["start", "queue", "stop_during_start"])
def test_a_failed_head_start_is_ended_before_the_supervisor_creates_the_bus(tmp_path, monkeypatch, fresh_bus, fail_at):
    """A failed attempt never published its queue: the supervisor's ordinary call ends its manager
    (journaled), and only then starts the one bus; Stop leaves the killed one to the teardown."""
    import json
    from concurrent.futures import wait
    from ouroboros import server_boot
    from ouroboros.server_liveness import BootDocument
    from supervisor import event_observation

    workers = fresh_bus
    stop = threading.Event()
    fakes = _FakeManagers(fail_at, stop)
    monkeypatch.setattr(event_observation, "EventQueueManager", fakes.Manager)
    monkeypatch.setattr(server_boot.store_switch, "wait_ready", lambda *_a, **_kw: {"status": "published"})
    doc = BootDocument()
    future = server_boot.prepare_event_queue(tmp_path, stop.is_set, doc)
    assert wait([future], timeout=3).done
    assert doc.snapshot()["steps"][0] == {**doc.snapshot()["steps"][0], "name": "startup:event_queue", "state": "failed"}
    first = workers._EVENT_Q_MANAGER
    assert first is fakes.made[0], "the failed attempt's manager stays owned until it is ended"
    if fail_at == "stop_during_start":
        assert server_boot.wait_event_queue(stop.is_set) is False, "a stopping supervisor creates nothing"
        assert "kill:0" in fakes.calls and "queue:0" not in fakes.calls and len(fakes.made) == 1
    else:
        assert server_boot.wait_event_queue(lambda: False) is True, "the head start's failure is not the generation's"
        assert workers.get_event_q(drive_root=tmp_path) is fakes.endpoint
        assert fakes.calls.index("shutdown:0") < fakes.calls.index("construct:1"), "ended before the next one starts"
        assert not first._process.is_alive() and workers._EVENT_Q_MANAGER is fakes.made[1] and len(fakes.made) == 2
        rows = [json.loads(line) for line in (tmp_path / "logs" / "supervisor.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [row["generation"] for row in rows if row["type"] == "event_queue_generation_stopped"] == [""]
    workers.shutdown_event_q()
    assert fakes.calls.count("shutdown:0") == 1 and not first._process.is_alive()


def test_a_failed_attempt_whose_manager_will_not_exit_stays_owned_and_refuses(tmp_path, monkeypatch, fresh_bus):
    from concurrent.futures import wait
    from ouroboros import server_boot
    from ouroboros.server_liveness import BootDocument
    from supervisor import event_observation

    workers = fresh_bus
    fakes = _FakeManagers("stuck", threading.Event(), stuck=True)
    monkeypatch.setattr(event_observation, "EventQueueManager", fakes.Manager)
    monkeypatch.setattr(server_boot.store_switch, "wait_ready", lambda *_a, **_kw: {"status": "published"})
    assert wait([server_boot.prepare_event_queue(tmp_path, lambda: False, BootDocument())], timeout=3).done
    for _ in range(2):  # a revival's call meets the same named refusal, never a second manager
        with pytest.raises(RuntimeError, match="did not exit; Restart ends it"):
            workers.get_event_q(drive_root=tmp_path)
    assert len(fakes.made) == 1 and workers._EVENT_Q_MANAGER is fakes.made[0] and fakes.made[0]._process.is_alive()
    workers.shutdown_event_q()  # the teardown still owns it: asked, killed and joined once more
    assert fakes.calls.count("shutdown:0") == 3 and fakes.calls.count("kill:0") == 3


def test_a_head_start_that_cannot_start_leaves_creation_to_the_supervisor(tmp_path, monkeypatch, fresh_bus):
    from ouroboros import server_boot
    def refused(**_kw):
        raise RuntimeError("thread unavailable")
    monkeypatch.setattr(server_boot.threading, "Thread", refused)
    future = server_boot.prepare_event_queue(tmp_path, lambda: False)
    with pytest.raises(RuntimeError, match="thread unavailable"):
        future.result(timeout=1)
    assert server_boot.wait_event_queue(lambda: False) is True and fresh_bus._EVENT_Q_MANAGER is None


def test_providerless_boot_does_not_prepare_a_transport(boot_app, monkeypatch, fresh_bus):
    from starlette.testclient import TestClient
    srv, _root = boot_app
    monkeypatch.setattr(srv.server_boot, "prepare_event_queue", lambda *_a, **_kw: pytest.fail("providerless transport"))
    with TestClient(srv.app) as client:
        _wait(lambda: _health(client)["boot"]["phase"] == "ready")
        assert srv.server_boot._head_start is None


def test_the_real_supervisor_creates_the_bus_itself_after_a_failed_head_start(tmp_path, monkeypatch, fresh_bus):
    """A generation, and a settings revival after it, are not failed by a failed head start: each
    reaches its ordinary worker start (whose bus call ends that attempt's manager first)."""
    from concurrent.futures import Future
    import server
    from ouroboros import server_boot
    from tests.test_server_shutdown import _supervisor_harness

    workers = fresh_bus
    future = Future()
    future.set_exception(OSError("native queue failed"))
    monkeypatch.setattr(server_boot, "_head_start", future)
    started = []
    for _ in range(2):
        rec = _supervisor_harness(monkeypatch, tmp_path, ["stop"])
        monkeypatch.setattr(workers, "init", lambda **kw: started.append(kw["drive_root"]))
        server._run_supervisor({})
        assert server._supervisor_error is None and server._supervisor_init_done.is_set() and rec.ticks == 1
    assert started == [tmp_path, tmp_path]


def test_stop_before_supervisor_init_closes_the_step_without_releasing_update_finalization(tmp_path, monkeypatch):
    from concurrent.futures import Future
    import server
    from ouroboros import server_boot, server_liveness
    from ouroboros.server_liveness import BootDocument
    from supervisor import workers
    from tests.test_server_shutdown import _supervisor_harness

    rec = _supervisor_harness(monkeypatch, tmp_path, [])
    doc, init_done = BootDocument(), threading.Event()
    monkeypatch.setattr(server, "BOOT", doc)
    monkeypatch.setattr(server_liveness, "BOOT", doc)
    monkeypatch.setattr(server, "_supervisor_init_done", init_done)
    monkeypatch.setattr(server, "_open_dialogue", lambda *_a, **_kw: None)
    monkeypatch.setattr(server.store_switch, "require_published", lambda *_a: None)
    monkeypatch.setattr("ouroboros.store_import.note_current_stamp", lambda *_a: None)
    monkeypatch.setattr(server_boot, "_head_start", Future())  # unfinished, but Stop must not wait for it
    monkeypatch.setattr(workers, "init", lambda **_kw: pytest.fail("no worker init after Stop"))
    rec.stop.set()

    server._run_supervisor({})

    step = next(row for row in doc.snapshot()["steps"] if row["name"] == "supervisor")
    assert step["state"] == "failed" and step["error"] == "stopped before init" and step["ended_at"]
    assert not init_done.is_set() and not rec.ready.is_set() and server._supervisor_error is None
    assert rec.ticks == 0 and not rec.stop.waits and not rec.alerts
    assert rec.watchdog_stops and all(stop.is_set() for stop in rec.watchdog_stops)


def test_stop_during_queue_rpc_and_concurrent_first_callers_share_the_attempt(tmp_path, monkeypatch, fresh_bus):
    from concurrent.futures import wait
    from types import SimpleNamespace
    from ouroboros import server_boot
    from ouroboros.server_liveness import BootDocument
    from supervisor import event_observation

    workers = fresh_bus
    stop, entered, release = threading.Event(), threading.Event(), threading.Event()
    calls, returned, endpoint = [], [], object()
    class Manager:
        _process = SimpleNamespace(pid=0, exitcode=None, is_alive=lambda: False)
        def __init__(self, **_kw):
            calls.append("construct")
        def start(self, **_kw):
            calls.append("start")
        def Queue(self):  # noqa: N802 - the registered typeid
            calls.append("queue")
            entered.set()
            assert release.wait(5)
            return endpoint
        def shutdown(self):
            calls.append("shutdown")
    monkeypatch.setattr(event_observation, "EventQueueManager", Manager)
    monkeypatch.setattr(server_boot.store_switch, "wait_ready", lambda *_a, **_kw: {"status": "published"})
    future = server_boot.prepare_event_queue(tmp_path, stop.is_set, BootDocument())
    assert entered.wait(3)
    legacy = threading.Thread(target=lambda: returned.append(workers.get_event_q()), daemon=True)
    legacy.start()
    stop.set()
    try:
        assert server_boot.wait_event_queue(stop.is_set) is False
        assert not future.done(), "Stop returned while the queue RPC is still held"
    finally:
        release.set()
        legacy.join(5)
        assert wait([future], timeout=5).done
        workers.shutdown_event_q()
    assert future.result() is endpoint and returned == [endpoint]
    assert calls == ["construct", "start", "queue", "shutdown"]


def test_every_start_step_the_ui_names_is_one_the_server_produces():
    """The start's words (web/modules/host_starting_status.js START_STEPS) name only steps a producer records: the
    supervisor's ``run_startup_phase`` sub-steps, the lifespan's coordinator steps and the head start's own step.
    A renamed producer step would otherwise leave the bar saying "finishing startup" over real work."""
    import ast
    import pathlib
    import re

    repo = pathlib.Path(__file__).resolve().parents[1]
    named = set(re.findall(r"\['([^']+)', '[^']+'\]", re.search(
        r"const START_STEPS = \[(.*?)\];", (repo / "web/modules/host_starting_status.js").read_text(encoding="utf-8"),
        re.S).group(1)))
    produced = set()
    for path in ("server.py", "ouroboros/server_boot.py"):
        for node in ast.walk(ast.parse((repo / path).read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", "")) in {"run_startup_phase", "step"}:
                produced.update(arg.value for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str))
            pair = isinstance(node, ast.Tuple) and len(node.elts) == 2 and isinstance(node.elts[0], ast.Constant)
            if pair and isinstance(node.elts[0].value, str) and isinstance(node.elts[1], (ast.Lambda, ast.Name)):
                produced.add(node.elts[0].value)  # a coordinator step: (name, callable)
    assert {"skills", "extensions", "startup:event_queue", "startup:worker_spawn"} <= produced, sorted(produced)
    assert named and named <= produced, sorted(named - produced)
