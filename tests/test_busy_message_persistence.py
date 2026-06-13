"""Owner messages during a busy task: delivery, restart-survival, routing.

Covers:
  - server.py busy-branch persists plain owner prose to the live task's mailbox
    (disk) instead of an in-memory-only inject, so a mid-round restart (exit
    42/99) does not drop it; and routes plain messages to the running task
    rather than spawning a new task (force_plan/constraint still spawn new).
  - supervisor.workers.auto_resume_after_restart drains owner messages orphaned
    by the restart's new task_id and delivers them in the resume prompt —
    panic-safe (purges on stop), stale-bounded, control entries excluded.
"""
import threading


# --- helpers -------------------------------------------------------------------

def _busy_agent(drive_root):
    class _Env:
        pass

    class _Agent:
        _busy = True
        _current_task_id = "livetask"

        def __init__(self):
            self.env = _Env()
            self.env.drive_root = drive_root
            self.injected = []

        def inject_message(self, text, image_data=None):
            self.injected.append((text, image_data))

    return _Agent()


def _busy_ctx(agent):
    class _Consc:
        def inject_observation(self, *_a, **_k):
            return None

    class _Ctx:
        consciousness = _Consc()

        def load_state(self):
            return {"owner_id": 1}

        def update_state(self, mutator):
            live = {"owner_id": 1}
            mutator(live)
            return live

        def get_chat_agent(self):
            return agent

    return _Ctx()


def _busy_ctx_capture(agent, direct_calls):
    """Like _busy_ctx but records ctx.handle_chat_direct calls (the new-task path)."""
    class _Consc:
        def inject_observation(self, *_a, **_k): return None
        def pause(self): return None
        def resume(self): return None

    class _Ctx:
        consciousness = _Consc()

        def load_state(self):
            return {"owner_id": 1}

        def update_state(self, mutator):
            live = {"owner_id": 1}
            mutator(live)
            return live

        def get_chat_agent(self):
            return agent

        def handle_chat_direct(self, cid, txt, img, task_constraint=None, task_metadata=None):
            direct_calls.append({"text": txt, "task_constraint": task_constraint, "task_metadata": task_metadata})

    return _Ctx()


def _bridge(message):
    class _Bridge:
        def get_updates(self, offset=0, timeout=1):
            return [{"update_id": 5, "message": message}]

        def broadcast(self, *_a, **_k):
            return None

        def send_message(self, *_a, **_k):
            return None

    return _Bridge()


# --- busy-branch persistence + routing -----------------------------------------

def test_busy_owner_text_persisted_to_mailbox(tmp_path, monkeypatch):
    import server
    import supervisor.message_bus as message_bus
    from ouroboros.owner_mailbox import _mailbox_path

    monkeypatch.setattr(message_bus, "log_chat", lambda *a, **k: None)
    agent = _busy_agent(tmp_path)
    ctx = _busy_ctx(agent)
    bridge = _bridge({"chat": {"id": 1}, "from": {"id": 1}, "text": "ты тут?", "source": "web"})

    server._process_bridge_updates(bridge, 0, ctx)

    mbox = _mailbox_path(tmp_path, "livetask")
    assert mbox.exists()
    assert "ты тут?" in mbox.read_text(encoding="utf-8")
    assert agent.injected == []  # disk-only, not in-memory


def test_busy_persist_failure_falls_back_to_inject(tmp_path, monkeypatch):
    # write_owner_message swallows I/O errors and returns False; a failed write
    # must fall back to the in-memory path, never silently drop the message.
    import server
    import supervisor.message_bus as message_bus
    import ouroboros.owner_mailbox as omb

    monkeypatch.setattr(message_bus, "log_chat", lambda *a, **k: None)
    monkeypatch.setattr(omb, "write_owner_message", lambda *a, **k: False)
    agent = _busy_agent(tmp_path)
    ctx = _busy_ctx(agent)
    bridge = _bridge({"chat": {"id": 1}, "from": {"id": 1}, "text": "ты тут?", "source": "web"})

    server._process_bridge_updates(bridge, 0, ctx)

    assert len(agent.injected) == 1
    assert agent.injected[0][0] == "ты тут?"


def test_busy_image_message_falls_back_to_inject(tmp_path, monkeypatch):
    import server
    import supervisor.message_bus as message_bus
    from ouroboros.owner_mailbox import _mailbox_path

    monkeypatch.setattr(message_bus, "log_chat", lambda *a, **k: None)
    agent = _busy_agent(tmp_path)
    ctx = _busy_ctx(agent)
    bridge = _bridge({
        "chat": {"id": 1}, "from": {"id": 1}, "text": "",
        "source": "web", "image_base64": "aGk=", "image_mime": "image/png",
        "image_caption": "pic",
    })

    server._process_bridge_updates(bridge, 0, ctx)

    # The text-only mailbox can't carry the image → in-memory path is used.
    assert not _mailbox_path(tmp_path, "livetask").exists()
    assert len(agent.injected) == 1
    assert agent.injected[0][1] is not None  # image_data forwarded


def test_web_force_plan_false_injects_into_running_task(tmp_path, monkeypatch):
    # Regression for the 2026-06-05 break (commit 00ce410): a REAL web envelope
    # task_metadata={"force_plan": False} arriving while busy must be delivered
    # INTO the running task (owner_mailbox), NOT spawned as a separate task.
    import server
    import supervisor.message_bus as message_bus
    from ouroboros.owner_mailbox import _mailbox_path

    monkeypatch.setattr(message_bus, "log_chat", lambda *a, **k: None)
    agent = _busy_agent(tmp_path)
    direct = []
    ctx = _busy_ctx_capture(agent, direct)
    bridge = _bridge({"chat": {"id": 1}, "from": {"id": 1}, "text": "остановись",
                      "source": "web",
                      "task_metadata": {"force_plan": False, "force_plan_source": ""}})

    server._process_bridge_updates(bridge, 0, ctx)

    mbox = _mailbox_path(tmp_path, "livetask")
    assert mbox.exists() and "остановись" in mbox.read_text(encoding="utf-8")
    assert direct == []          # NO new task spawned
    assert agent.injected == []  # disk-only persist (drained at the round boundary)


def test_web_force_plan_true_still_spawns_new_task(tmp_path, monkeypatch):
    # Preserved behavior: a genuine force_plan=True directive while busy still
    # routes to a new constrained task (handle_chat_direct), never an inject.
    import server
    import supervisor.message_bus as message_bus
    from ouroboros.owner_mailbox import _mailbox_path

    monkeypatch.setattr(message_bus, "log_chat", lambda *a, **k: None)

    class _ImmediateThread:
        def __init__(self, target, args=(), daemon=False):
            self.target = target
            self.args = args

        def start(self):
            self.target(*self.args)

    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)

    agent = _busy_agent(tmp_path)
    direct = []
    ctx = _busy_ctx_capture(agent, direct)
    bridge = _bridge({"chat": {"id": 1}, "from": {"id": 1}, "text": "сделай план",
                      "source": "web",
                      "task_metadata": {"force_plan": True, "force_plan_source": "consilium"}})

    server._process_bridge_updates(bridge, 0, ctx)

    assert len(direct) == 1 and direct[0]["task_metadata"]["force_plan"] is True
    assert not _mailbox_path(tmp_path, "livetask").exists()  # not injected into running task
    assert agent.injected == []


# --- orphaned-mailbox readthrough at resume ------------------------------------

def _seed_recent_restart(tmp_path):
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state" / "pending_restart_verify.json").write_text("{}", encoding="utf-8")
    (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
    # Empty scratchpad: resume must still fire, driven solely by the message.
    (tmp_path / "memory" / "scratchpad.md").write_text("# Scratchpad\n- (empty)\n", encoding="utf-8")


def test_auto_resume_delivers_orphaned_owner_messages(tmp_path, monkeypatch):
    import supervisor.workers as workers
    from ouroboros.owner_mailbox import write_owner_message, KIND_FINALIZE_NOW, _mailbox_path

    original = workers.DRIVE_ROOT
    workers.DRIVE_ROOT = tmp_path
    try:
        _seed_recent_restart(tmp_path)
        write_owner_message(tmp_path, "ты тут?", task_id="oldtask")
        # A stale control entry from before the restart must never be injected.
        write_owner_message(tmp_path, "deadline-control", task_id="oldtask", kind=KIND_FINALIZE_NOW)

        monkeypatch.setattr(workers, "load_state", lambda: {"owner_chat_id": 42})

        class _FakeAgent:
            _busy = False

        monkeypatch.setattr(workers, "_get_chat_agent", lambda: _FakeAgent())
        monkeypatch.setattr(workers.time, "sleep", lambda *_a, **_k: None)

        done = threading.Event()
        cap = {}

        def fake_hcd(chat_id, prompt, image=None):
            cap["chat_id"] = chat_id
            cap["prompt"] = prompt
            done.set()

        monkeypatch.setattr(workers, "handle_chat_direct", fake_hcd)

        workers.auto_resume_after_restart()
        assert done.wait(5), "resume did not fire on orphaned owner messages"
        mbox = _mailbox_path(tmp_path, "oldtask")
    finally:
        workers.DRIVE_ROOT = original

    assert cap["chat_id"] == 42
    assert "ты тут?" in cap["prompt"]
    assert "deadline-control" not in cap["prompt"]  # control entry excluded
    assert not mbox.exists()  # consumed on delivery


def test_auto_resume_drops_stale_orphan_messages(tmp_path, monkeypatch):
    import json
    import supervisor.workers as workers
    from ouroboros.owner_mailbox import _mailbox_path

    original = workers.DRIVE_ROOT
    workers.DRIVE_ROOT = tmp_path
    try:
        _seed_recent_restart(tmp_path)
        # Hand-write an orphan entry dated well beyond the 24h recovery horizon.
        mbox = _mailbox_path(tmp_path, "oldtask")
        mbox.parent.mkdir(parents=True, exist_ok=True)
        mbox.write_text(json.dumps({
            "msg_id": "m1", "ts": "2020-01-01T00:00:00+00:00",
            "text": "ancient question", "kind": "owner_text",
        }) + "\n", encoding="utf-8")

        monkeypatch.setattr(workers, "load_state", lambda: {"owner_chat_id": 42})
        monkeypatch.setattr(workers.time, "sleep", lambda *_a, **_k: None)

        class _FakeAgent:
            _busy = False

        monkeypatch.setattr(workers, "_get_chat_agent", lambda: _FakeAgent())
        called = []
        monkeypatch.setattr(workers, "handle_chat_direct", lambda *a, **k: called.append(a))

        workers.auto_resume_after_restart()
    finally:
        workers.DRIVE_ROOT = original

    # Empty scratchpad + only a stale message → no resurrection, no resume.
    assert called == []


def test_auto_resume_aborts_on_panic_during_init(tmp_path, monkeypatch):
    import supervisor.workers as workers
    from ouroboros.owner_mailbox import write_owner_message

    original = workers.DRIVE_ROOT
    workers.DRIVE_ROOT = tmp_path
    try:
        _seed_recent_restart(tmp_path)
        write_owner_message(tmp_path, "ты тут?", task_id="oldtask")
        monkeypatch.setattr(workers, "load_state", lambda: {"owner_chat_id": 42})

        # Emergency Stop raised during the 2s init window must abort the resume.
        def fake_sleep(*_a, **_k):
            (tmp_path / "state" / "panic_stop.flag").write_text("panic", encoding="utf-8")

        monkeypatch.setattr(workers.time, "sleep", fake_sleep)

        class _FakeAgent:
            _busy = False

        monkeypatch.setattr(workers, "_get_chat_agent", lambda: _FakeAgent())
        called = []
        monkeypatch.setattr(workers, "handle_chat_direct", lambda *a, **k: called.append(a))

        workers.auto_resume_after_restart()
    finally:
        workers.DRIVE_ROOT = original

    assert called == []


def test_auto_resume_panic_purges_mailbox(tmp_path, monkeypatch):
    # Emergency Stop must both skip resume AND purge pending owner messages, so
    # a stop the owner issued can never be resurrected by a later restart (P0).
    import supervisor.workers as workers
    from ouroboros.owner_mailbox import write_owner_message, _mailbox_path

    original = workers.DRIVE_ROOT
    workers.DRIVE_ROOT = tmp_path
    try:
        (tmp_path / "state").mkdir(parents=True, exist_ok=True)
        (tmp_path / "state" / "panic_stop.flag").write_text("panic", encoding="utf-8")
        write_owner_message(tmp_path, "остановись", task_id="oldtask")

        called = []
        monkeypatch.setattr(workers, "handle_chat_direct", lambda *a, **k: called.append(a))
        monkeypatch.setattr(workers.time, "sleep", lambda *_a, **_k: None)

        workers.auto_resume_after_restart()
        mbox = _mailbox_path(tmp_path, "oldtask")
    finally:
        workers.DRIVE_ROOT = original

    assert called == []  # never resume after panic
    assert not mbox.exists()  # pending message purged — cannot resurface later
    assert not (tmp_path / "state" / "panic_stop.flag").exists()  # flag consumed


def test_panic_suppressed_message_not_resurrected_on_later_restart(tmp_path, monkeypatch):
    # A message suppressed by /panic must not come back on a subsequent
    # non-suppressed exit-42 restart within the 24h window.
    import supervisor.workers as workers
    from ouroboros.owner_mailbox import write_owner_message, _mailbox_path

    original = workers.DRIVE_ROOT
    workers.DRIVE_ROOT = tmp_path
    try:
        (tmp_path / "state").mkdir(parents=True, exist_ok=True)
        (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
        write_owner_message(tmp_path, "остановись прямо сейчас", task_id="oldtask")

        monkeypatch.setattr(workers, "load_state", lambda: {"owner_chat_id": 42})
        monkeypatch.setattr(workers.time, "sleep", lambda *_a, **_k: None)

        class _FakeAgent:
            _busy = False

        monkeypatch.setattr(workers, "_get_chat_agent", lambda: _FakeAgent())
        calls = []
        monkeypatch.setattr(workers, "handle_chat_direct", lambda *a, **k: calls.append(a))

        # BOOT #1 — panic: skip resume, purge the pending message.
        (tmp_path / "state" / "panic_stop.flag").write_text("panic", encoding="utf-8")
        workers.auto_resume_after_restart()
        assert calls == []
        assert not _mailbox_path(tmp_path, "oldtask").exists()  # purged

        # BOOT #2 — ordinary exit-42 self-restart (no suppression flag), recent.
        (tmp_path / "state" / "pending_restart_verify.json").write_text("{}", encoding="utf-8")
        (tmp_path / "memory" / "scratchpad.md").write_text("# Scratchpad\n- (empty)\n", encoding="utf-8")
        workers.auto_resume_after_restart()
    finally:
        workers.DRIVE_ROOT = original

    # Nothing was delivered: the suppressed message did not resurface.
    joined = " ".join(str(a) for a in calls)
    assert "остановись" not in joined
