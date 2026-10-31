"""#1307: an unreadable, corrupt or lost state.json is never a fresh install.

Isolated roots only (``state.init(tmp_path)``); every test restores the module paths.
"""
from __future__ import annotations

import errno
import json
import os
import pathlib
import threading
import time
from types import SimpleNamespace

import pytest

from supervisor import state, state_initialization
from tests._store_shared import remove_store_tree


@pytest.fixture
def root(tmp_path, monkeypatch):
    from supervisor import queue
    from supervisor import evolution_lifecycle as lifecycle

    prior_state, prior_queue = state.DRIVE_ROOT, queue.DRIVE_ROOT
    state.init(tmp_path)
    queue.init(tmp_path)
    monkeypatch.setattr(state, "check_openrouter_ground_truth", lambda: None)
    monkeypatch.setitem(lifecycle._STOP_LATCH, "stopped", False)
    yield tmp_path
    state.init(prior_state)
    queue.init(prior_queue)


def _write(path: pathlib.Path, payload) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    path.write_bytes(raw)
    return raw


def _prior(**extra):
    return {"created_at": "2026-07-30T00:00:00+00:00", "session_id": "old-session", "owner_id": 7,
            "owner_chat_id": 7, "owner_external_id": 99, "owner_external_chat_id": 99,
            "evolution_mode_enabled": True, "evolution_owner_stopped": False, "evolution_cycle": 6,
            "bg_consciousness_enabled": True, "usage_ledger_high_water_seq": [1, 50], **extra}


def _files(root):
    out = {}
    for p in (root / "state").glob("*"):
        if p.is_file():
            with open(p, "rb") as handle:  # not Path.read_bytes: tests patch it
                out[p.name] = handle.read()
    return out


@pytest.mark.parametrize("failure", ["enfile", "eacces"])
def test_an_unreadable_primary_is_never_written_and_its_controls_are_unknown(root, monkeypatch, failure):
    _write(state.STATE_PATH, _prior())
    _write(state.STATE_LAST_GOOD_PATH, _prior(evolution_owner_stopped=False))
    before = _files(root)
    real = pathlib.Path.read_bytes

    def flaky(self):
        if self == state.STATE_PATH:
            raise OSError(errno.ENFILE if failure == "enfile" else errno.EACCES, "denied")
        return real(self)

    monkeypatch.setattr(pathlib.Path, "read_bytes", flaky)
    read = state.read_state()
    assert (read.quality, read.source) == ("recovered_transient", "backup")
    shown = state.load_state()
    assert shown["evolution_cycle"] == 6  # display values survive
    assert state.control_value(shown, "evolution_owner_stopped") == (False, None)
    assert not state.control_is(shown, "evolution_mode_enabled", True)
    with pytest.raises(state.StateUnavailable) as refused:
        state.update_state(lambda live: live.__setitem__("session_id", "new"))
    assert refused.value.reason == "primary_unreadable"
    assert state.init_state().quality == "recovered_transient"
    assert _files(root) == before  # nothing minted, nothing overwritten, no witness


def test_a_corrupt_primary_recovers_from_backup_without_laundering_authority(root):
    corrupt = _write(state.STATE_PATH, b'{"evolution_owner_stopped": tru')
    _write(state.STATE_LAST_GOOD_PATH, _prior())
    # Routine bookkeeping triggers the durable recovery write…
    state.update_state(lambda live: live.__setitem__("last_owner_message_at", "now"))
    [copy] = list((root / "state").glob("state.corrupt-*.json"))
    assert copy.read_bytes() == corrupt
    saved = json.loads(state.STATE_PATH.read_bytes())
    assert saved["created_at"] == "2026-07-30T00:00:00+00:00" and saved["session_id"] == "old-session"
    assert "usage_ledger_high_water_seq" not in saved  # #1144: money re-derives from the ledger
    assert set(saved[state.RECOVERY_KEY]["unconfirmed"]) == set(state.CONTROL_KEYS)
    # …and more bookkeeping, then a cold boot, still cannot clear it.
    state.update_state(lambda live: live.pop(state.RECOVERY_KEY, None))
    state.update_state(lambda live: live.__setitem__("evolution_owner_stopped", False))
    assert state.init_state().quality == "recovered"
    cold = state.load_state()
    assert state.control_value(cold, "evolution_owner_stopped") == (False, None)
    assert state.control_value(cold, "owner_external_id") == (False, None)
    # A real decision confirms exactly the control it sets.
    state.update_state(lambda live: live.__setitem__("bg_consciousness_enabled", False),
                       confirm=("bg_consciousness_enabled",))
    after = state.load_state()
    assert state.control_value(after, "bg_consciousness_enabled") == (True, False)
    assert state.control_value(after, "evolution_mode_enabled") == (False, None)


def test_evolution_after_a_recovery_waits_for_the_owner_then_resumes(root, monkeypatch):
    from supervisor import evolution_lifecycle as lifecycle
    from supervisor import queue
    from supervisor.events_runtime_controls import owner_evolution_start

    state.save_state(_prior())
    _write(state.STATE_PATH, b'{"evolution_mode_enabled": tru')
    assert lifecycle.start_evolution_campaign("improve", source="owner")
    state.update_state(lambda live: live.__setitem__("last_owner_message_at", "now"))  # the recovery write
    pending = []
    queue.init_queue_refs(pending, {}, {"value": 0})
    monkeypatch.setattr(state, "TOTAL_BUDGET_LIMIT", 0.0)
    monkeypatch.setattr(queue, "send_with_budget", lambda *a, **k: None)
    monkeypatch.setattr(lifecycle, "evolution_block_reason", lambda: "")
    queue.enqueue_evolution_task_if_needed()
    assert pending == []  # a recovered "enabled" is not the owner's current decision
    assert owner_evolution_start("resume", source="owner_chat") == ""
    queue.enqueue_evolution_task_if_needed()
    # The witnessed write-once binding survives; Start confirms evolution only.
    assert [task["chat_id"] for task in pending] == [7]
    assert state.control_value(state.load_state(), "owner_chat_id") == (True, 7)


def test_writers_never_proceed_unlocked(root, monkeypatch):
    _write(state.STATE_PATH, _prior())
    before = state.STATE_PATH.read_bytes()
    monkeypatch.setattr(state, "acquire_file_lock", lambda *a, **k: None)
    with pytest.raises(state.StateUnavailable) as refused:
        state.update_state(lambda live: live.__setitem__("session_id", "x"))
    assert refused.value.reason == "lock_timeout"
    assert state.update_budget_from_usage({}) is False
    assert state.STATE_PATH.read_bytes() == before


def test_reads_never_create_state(root):
    assert state.load_state()[state.STATE_READ_KEY]["quality"] == "uninitialized"
    with pytest.raises(state.StateUnavailable):
        state.update_state(lambda live: None)
    assert not state.STATE_PATH.exists() and not state.STATE_LAST_GOOD_PATH.exists()
    assert not state_initialization.witness_path(root).exists()


@pytest.mark.parametrize("scaffold", ["empty_dirs", "settings_first", "benchmark_seed"])
def test_a_positive_fresh_bootstrap_creates_one_identity(root, scaffold):
    for rel in ("state", "logs", "memory", "task_results", "locks"):
        (root / rel).mkdir(parents=True, exist_ok=True)
    if scaffold in {"settings_first", "benchmark_seed"}:
        (root / "settings.json").write_text("{}", encoding="utf-8")
    if scaffold == "benchmark_seed":
        (root / state.ISOLATED_BENCHMARK_SENTINEL).write_text("1", encoding="utf-8")
        (root / "logs" / "chat.jsonl").write_text("", encoding="utf-8")  # empty is not history
    read = state.init_state()
    assert read.quality == "current"
    status, witness = state_initialization.read_witness(root)
    assert status == "ok" and witness["phase"] == "complete" and witness["origin"] == "first_boot"
    assert read.values["initialization_id"] == witness["initialization_id"]
    assert state.control_value(state.load_state(), "owner_id") == (True, None)


def test_an_interrupted_initialization_finishes_the_same_identity(root):
    (root / "state").mkdir(exist_ok=True)
    (root / "state" / "state.initialized.json").write_text(json.dumps(
        {"initialization_id": "first-id", "phase": "pending", "origin": "first_boot"}), encoding="utf-8")
    assert state.init_state().values["initialization_id"] == "first-id"
    assert state_initialization.read_witness(root)[1]["phase"] == "complete"


@pytest.mark.parametrize("history", ["witness_complete", "queue_snapshot", "task_result", "chat",
                                     "campaign", "bindings", "evidence_unknown", "witness_corrupt"])
def test_lost_state_on_an_initialized_or_historical_root_is_never_reminted(root, monkeypatch, history):
    (root / "state").mkdir(exist_ok=True)
    if history == "witness_complete":
        (root / "state" / "state.initialized.json").write_text(json.dumps(
            {"initialization_id": "i", "phase": "complete", "completed_at": "t"}), encoding="utf-8")
    elif history == "witness_corrupt":
        (root / "state" / "state.initialized.json").write_text("{", encoding="utf-8")
    elif history == "queue_snapshot":
        (root / "state" / "queue_snapshot.json").write_text("{}", encoding="utf-8")
    elif history == "task_result":
        _write(root / "task_results" / "t1.json", {"task_id": "t1"})
    elif history == "chat":
        _write(root / "logs" / "chat.jsonl", b'{"text": "hi"}\n')
    elif history == "campaign":
        _write(root / "state" / "evolution_campaign.json", {"id": "c"})
    elif history == "bindings":
        _write(root / "state" / "project_task_bindings.json", {})
    else:
        real = os.scandir

        def blind(path):
            raise PermissionError(errno.EACCES, "denied") if str(path).endswith("task_results") else real(path)

        (root / "task_results").mkdir()
        monkeypatch.setattr(os, "scandir", blind)
    read = state.init_state()
    assert read.quality in {"uninitialized", "unavailable"} and read.values == {}
    assert not state.STATE_PATH.exists() and not state.STATE_LAST_GOOD_PATH.exists()
    assert not state.control_is(state.load_state(), "evolution_owner_stopped", False)


def test_a_legacy_readable_state_is_adopted_with_its_values(root):
    raw = _write(state.STATE_PATH, _prior())
    _write(state.STATE_LAST_GOOD_PATH, raw)
    (root / "state" / "queue_snapshot.json").write_text("{}", encoding="utf-8")
    read = state.init_state()
    assert read.quality == "current" and read.values["session_id"] == "old-session"
    assert read.values["created_at"] == "2026-07-30T00:00:00+00:00" and read.values["evolution_cycle"] == 6
    status, witness = state_initialization.read_witness(root)
    assert (status, witness["phase"], witness["origin"]) == ("ok", "complete", "legacy_adopted")


def test_an_owner_reset_is_an_explicit_fresh_start(root):
    _write(root / "task_results" / "leftover.json", {"task_id": "leftover"})
    state_initialization.mark_pending(root, origin="owner_reset")
    read = state.init_state()
    assert read.quality == "current"
    assert state_initialization.read_witness(root)[1]["origin"] == "owner_reset"


def test_stop_during_an_outage_outlives_the_primary_coming_back(root, monkeypatch):
    from supervisor import evolution_lifecycle as lifecycle
    from supervisor.events_runtime_controls import owner_evolution_start, owner_evolution_stop_controls

    state.save_state(_prior())
    assert lifecycle.start_evolution_campaign("improve", source="owner")
    real = pathlib.Path.read_bytes
    outage = {"on": True}
    monkeypatch.setattr(pathlib.Path, "read_bytes", lambda self: (_ for _ in ()).throw(
        OSError(errno.EIO, "io")) if outage["on"] and self == state.STATE_PATH else real(self))
    disclosure = owner_evolution_stop_controls("owner /evolve off")
    assert "runtime state" in disclosure and "campaign" not in disclosure
    outage["on"] = False
    assert state.control_is(state.load_state(), "evolution_mode_enabled", True)  # the stale primary returns
    assert lifecycle.evolution_stop_reason() == "stop_latched"
    lifecycle._STOP_LATCH["stopped"] = False  # a restart loses the latch; the durable intent stays
    assert lifecycle.evolution_stop_reason() == "stop_intent"
    assert not lifecycle.start_evolution_campaign("agent", source="agent_tool")
    assert lifecycle._read_evolution_campaign()["status"] == "active"  # no invented terminal status
    assert owner_evolution_start("resume", source="owner_chat") == ""
    assert lifecycle.evolution_stop_reason() == "" and "stop_intent" not in lifecycle._read_evolution_campaign()
    assert state.control_is(state.load_state(), "evolution_mode_enabled", True)


def test_panic_requests_every_physical_stop_before_any_persistence_and_never_waits(root, monkeypatch):
    import ouroboros.server_control as control
    from ouroboros import workspace_executor
    from ouroboros.tools import services, shell_process
    from supervisor import evolution_lifecycle as lifecycle

    # Production exits permanently; this in-process exit stub must not leave
    # its terminal admission latches set for subsequent consumer tests, nor kill
    # this worker's own multiprocessing children (the shared supervisor event bus).
    for owner in (workspace_executor, services, shell_process):
        monkeypatch.setattr(owner, "_panic_requested", False)
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    _write(state.STATE_PATH, _prior())
    order: list = []
    release = threading.Event()
    monkeypatch.setattr(state, "update_state", lambda *a, **k: order.append("state") or release.wait(30))
    monkeypatch.setattr(lifecycle, "record_evolution_stop_intent", lambda *a, **k: order.append("intent"))
    monkeypatch.setattr(control, "_write_panic_flag", lambda data_dir: (
        order.append("flag"), control_write(data_dir)))
    control_write = lambda data_dir: (data_dir / "state" / "panic_stop.flag").write_text("panic")  # noqa: E731
    monkeypatch.setattr(control.os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
    import ouroboros.platform_layer as platform_layer

    monkeypatch.setattr(platform_layer, "kill_process_on_port", lambda port: order.append(("port", port)))
    # The real panic kills every multiprocessing child of THIS process; in a test that
    # would include another test's process-global event bus manager, leaving a dead
    # proxy that breaks every later emission. Like the other in-process panic tests.
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    started = time.monotonic()
    with pytest.raises(SystemExit):
        control.execute_panic_stop(  # a stalled log handler and a held clock lock block nothing either
            SimpleNamespace(stop=lambda: release.wait(30)),
            lambda **kw: order.append("workers"), data_dir=root, panic_exit_code=99,
            log=SimpleNamespace(critical=lambda *a, **k: release.wait(30)))
    release.set()
    assert time.monotonic() - started < 15  # every write bounded, never awaited
    assert "workers" not in order  # lifelines own pooled child requests
    assert not [step for step in order if isinstance(step, tuple) and step[0] == "port"]  # no port sweep
    assert order.index("flag") < order.index("state")
    assert (root / "state" / "panic_stop.flag").read_text() == "panic"


def test_an_owner_restart_never_replaces_a_panic_flag_still_owed_its_controls(root, monkeypatch):
    import ouroboros.server_restart as restart
    from supervisor import worker_chat_lane

    flag = root / "state" / "panic_stop.flag"
    _write(flag, b"panic")  # boot kept it: the disabled controls were not durable yet
    monkeypatch.setattr(restart, "DATA_DIR", root)
    monkeypatch.setattr(restart, "_safe_restart_serialized", lambda *a, **k: (True, ""))
    monkeypatch.setattr(restart, "_stop_owned_work", lambda ctx: [])
    monkeypatch.setattr(restart, "_request_restart_exit", lambda owner=False: None)
    assert restart._perform_owner_restart(SimpleNamespace(safe_restart=None)) == (True, "")
    from ouroboros.delegate_recovery import active_restart_transaction
    assert flag.read_text() == "panic" and active_restart_transaction(root)["door"] == "owner_restart"
    monkeypatch.setattr(worker_chat_lane, "_pool", lambda: SimpleNamespace(DRIVE_ROOT=root, load_state=state.load_state))
    state.save_state(_prior())
    worker_chat_lane.auto_resume_after_restart()  # the owner-restart boot still settles the Panic debt
    assert not flag.exists()
    assert state.control_value(state.load_state(), "bg_consciousness_enabled") == (True, False)


def test_boot_consumes_the_panic_flag_only_after_its_controls_are_durable(root, monkeypatch):
    from supervisor import worker_chat_lane

    monkeypatch.setattr(worker_chat_lane, "_pool", lambda: SimpleNamespace(DRIVE_ROOT=root, load_state=state.load_state))
    state.save_state(_prior())
    primary, backup = state.STATE_PATH.read_bytes(), state.STATE_LAST_GOOD_PATH.read_bytes()
    state.STATE_PATH.unlink()
    state.STATE_LAST_GOOD_PATH.unlink()
    flag = root / "state" / "panic_stop.flag"
    _write(flag, b"panic")
    worker_chat_lane.auto_resume_after_restart()  # initialized state unavailable: controls not durable
    assert flag.exists()
    # The retained queue is now prior history: repair known state, never mint defaults.
    state.STATE_PATH.write_bytes(primary)
    state.STATE_LAST_GOOD_PATH.write_bytes(backup)
    worker_chat_lane.auto_resume_after_restart()
    assert not flag.exists()
    live = state.load_state()
    for key, value in (("evolution_mode_enabled", False), ("bg_consciousness_enabled", False),
                       ("evolution_owner_stopped", True)):
        assert state.control_value(live, key) == (True, value)


def _bridge(text: str, *, source: str = "telegram", user: int = 99, chat: int = 99):
    class Bridge:
        def get_updates(self, offset=0, timeout=1):
            return [{"update_id": offset, "message": {"chat": {"id": chat}, "from": {"id": user}, "text": text,
                                                       "source": source}}]

    return Bridge()


def _ingress_ctx(root, replies, panics):
    return SimpleNamespace(
        RUNNING={}, consciousness=None, DRIVE_ROOT=root,
        load_state=state.load_state, update_state=state.update_state, save_state=state.save_state,
        send_with_budget=lambda _chat, text, **_kw: replies.append(text),
        kill_workers=lambda **kw: panics.append("kill"),
    )


@pytest.mark.parametrize("readable", [True, False])
def test_external_commands_need_a_positively_bound_owner(root, monkeypatch, readable):
    import server
    import supervisor.message_bus as message_bus

    monkeypatch.setattr(message_bus, "log_chat", lambda *a, **k: None)
    monkeypatch.setattr(message_bus, "record_inbound_message", lambda *a, **k: {}, raising=False)
    panics: list = []
    monkeypatch.setattr(server, "_execute_panic_stop", lambda *a: panics.append("panic"))
    backup = _prior(owner_external_id=None, owner_external_chat_id=None)
    _write(state.STATE_LAST_GOOD_PATH, backup)
    if readable:
        _write(state.STATE_PATH, _prior())
        state.init_state()
    replies: list = []
    ctx = _ingress_ctx(root, replies, panics)
    server._process_bridge_updates(_bridge("/panic", user=5, chat=5), 0, ctx)  # a stranger
    assert panics == []
    if readable:
        server._process_bridge_updates(_bridge("/panic"), 0, ctx)  # the bound owner
        assert panics == ["panic"]
    else:
        # Unknown binding (only a backup is readable): refused and NEVER registered —
        # neither as the external owner nor as the global owner.
        assert any("unknown" in reply for reply in replies)
        live = state.load_state()
        assert state.control_value(live, "owner_external_id") == (False, None)
        assert state.control_value(live, "owner_id") == (False, None)


@pytest.mark.parametrize("primary", ["corrupt", "unreadable", "other_identity"])
def test_a_set_owner_binding_of_the_same_identity_survives_a_recovery(root, monkeypatch, primary):
    import server
    import supervisor.message_bus as message_bus

    monkeypatch.setattr(message_bus, "log_chat", lambda *a, **k: None)
    monkeypatch.setattr(message_bus, "record_inbound_message", lambda *a, **k: {}, raising=False)
    panics: list = []
    monkeypatch.setattr(server, "_execute_panic_stop", lambda *a: panics.append("panic"))
    assert state.init_state().quality == "current"  # a first boot: the witness identity
    identity = json.loads(state.STATE_PATH.read_bytes())["initialization_id"]
    backup = _prior(initialization_id="someone-else" if primary == "other_identity" else identity,
                    owner_id=None)
    _write(state.STATE_LAST_GOOD_PATH, backup)
    _write(state.STATE_PATH, b'{"owner_external_id": 5')
    if primary == "unreadable":
        real = pathlib.Path.read_bytes
        monkeypatch.setattr(pathlib.Path, "read_bytes", lambda self: (_ for _ in ()).throw(
            OSError(errno.EIO, "io")) if self == state.STATE_PATH else real(self))
    replies: list = []
    server._process_bridge_updates(_bridge("/panic", user=5, chat=5), 0, _ingress_ctx(root, replies, panics))
    server._process_bridge_updates(_bridge("/panic"), 0, _ingress_ctx(root, replies, panics))
    shown = state.load_state()
    if primary == "other_identity":  # a copy of another install proves nothing
        assert panics == [] and state.control_value(shown, "owner_external_id") == (False, None)
        return
    assert panics == ["panic"]  # the bound owner only, never the stranger
    assert state.control_value(shown, "owner_external_id") == (True, 99)
    assert state.control_value(shown, "owner_id") == (False, None)  # an empty slot may have been filled
    assert state.control_value(shown, "evolution_owner_stopped") == (False, None)


def test_the_local_panic_door_reads_no_state(root, monkeypatch):
    import server

    panics: list = []
    monkeypatch.setattr(server, "_execute_panic_stop", lambda *a: panics.append("panic"))
    ctx = _ingress_ctx(root, [], panics)
    ctx.load_state = lambda: pytest.fail("the local /panic must not read state first")
    server._process_bridge_updates(_bridge("/panic", source="web", user=1, chat=1), 0, ctx)
    assert panics == ["panic"]


def test_a_file_where_the_state_directory_belongs_is_unknown_not_absent(root):
    import shutil

    shutil.rmtree(root / "state", ignore_errors=True)
    (root / "state").write_text("not a directory", encoding="utf-8")
    read = state.init_state()  # typed unavailable, never a crash and never a mint
    assert read.quality == "unavailable" and read.values == {}
    assert state_initialization.supervisor_evidence(root)[0] in {"none", "unknown"}
    (root / "state").unlink()
    (root / "task_results").write_text("a file", encoding="utf-8")
    assert state_initialization.supervisor_evidence(root)[0] == "unknown"


def _windows_shaped_not_found(monkeypatch):
    """Where POSIX raises ENOTDIR because a PARENT component is a file, Windows raises
    ERROR_PATH_NOT_FOUND: the same FileNotFoundError as a missing parent. Only the readers'
    own operation errors are reshaped; the absence proof's ``os.stat`` stays real."""
    def reshaped(real):
        def call(path, *args, **kwargs):
            try:
                return real(path, *args, **kwargs)
            except NotADirectoryError as exc:
                if os.path.isdir(os.path.dirname(os.fspath(path))):
                    raise  # the path itself is the file: Windows says ENOTDIR too (WinError 267)
                raise FileNotFoundError(errno.ENOENT, "[WinError 3] path not found", os.fspath(path)) from exc
        return call

    monkeypatch.setattr(pathlib.Path, "read_bytes", reshaped(pathlib.Path.read_bytes))
    monkeypatch.setattr(pathlib.Path, "lstat", reshaped(pathlib.Path.lstat))
    monkeypatch.setattr(os, "lstat", reshaped(os.lstat))
    monkeypatch.setattr(os, "scandir", reshaped(os.scandir))


@pytest.mark.parametrize("errors", ["native", "windows"])
def test_only_a_directory_ancestor_proves_the_state_absent(root, monkeypatch, errors):
    import shutil

    if errors == "windows":
        _windows_shaped_not_found(monkeypatch)
    shutil.rmtree(root / "state", ignore_errors=True)
    (root / "state").write_text("not a directory", encoding="utf-8")
    assert state.read_state_copy(state.STATE_PATH)[0] == "unreadable"
    assert state_initialization.read_witness(root)[0] == "unreadable"
    assert state_initialization.supervisor_evidence(root)[0] == "unknown"
    assert state.init_state().quality == "unavailable"
    assert (root / "state").read_bytes() == b"not a directory"  # nothing minted over it
    (root / "state").unlink()  # truly absent below a real directory: a first boot
    assert state.read_state_copy(state.STATE_PATH)[0] == "missing"
    assert state_initialization.supervisor_evidence(root) == ("none", "")
    assert state.init_state().quality == "current" == state.read_state().quality
    assert state_initialization.read_witness(root)[1]["phase"] == "complete"
    remove_store_tree(root / "state")  # the boot above opened its stores
    real_stat = os.stat
    monkeypatch.setattr(os, "stat", lambda path, *a, **k: (_ for _ in ()).throw(PermissionError(
        errno.EACCES, "denied")) if pathlib.Path(path) == root / "state" else real_stat(path, *a, **k))
    assert state.read_state_copy(state.STATE_PATH)[0] == "unreadable"  # an unexaminable ancestor is unknown


def test_a_first_state_whose_witness_cannot_complete_admits_nothing_this_boot(root, monkeypatch):
    real_complete = state_initialization.complete
    monkeypatch.setattr(state_initialization, "complete", lambda *a, **k: False)
    read = state.init_state()
    assert read.quality == "unavailable" and "witness" in read.reason
    with pytest.raises(state.StateUnavailable):
        state.update_state(lambda live: live.update(owner_id=5))  # no registration on that identity
    assert not state.control_value(state.load_state(), "owner_id")[0]
    pending = state_initialization.read_witness(root)[1]
    assert pending["phase"] == "pending"
    monkeypatch.setattr(state_initialization, "complete", real_complete)
    state.init(root)  # the next boot adopts and completes the SAME identity
    assert state.init_state().values["initialization_id"] == pending["initialization_id"]
    assert state_initialization.read_witness(root)[1]["phase"] == "complete"


def test_a_whole_state_write_mints_only_where_initialization_would(root):
    state.save_state({"owner_chat_id": 1})  # a fresh root: a first state with a completed witness
    witness = state_initialization.read_witness(root)[1]
    assert witness["phase"] == "complete" and state.load_state()["initialization_id"] == witness["initialization_id"]
    state.save_state({"owner_chat_id": 2})
    assert state.load_state()["initialization_id"] == witness["initialization_id"]  # identity kept
    state.STATE_PATH.unlink()
    with pytest.raises(state.StateUnavailable):  # a recoverable backup is not overwritten
        state.save_state({})
    state.STATE_LAST_GOOD_PATH.unlink()
    with pytest.raises(state.StateUnavailable):  # a lost initialized state is never re-minted
        state.save_state({})
    assert not state.STATE_PATH.exists()


def test_a_writer_racing_an_owner_reset_never_resurrects_the_old_identity(root):
    raw = _write(state.STATE_PATH, _prior(initialization_id="old-id"))
    _write(state.STATE_LAST_GOOD_PATH, raw)
    state_initialization.mark_pending(root, origin="owner_reset")
    read = state.init_state()
    assert read.quality == "current" and read.values["initialization_id"] != "old-id"
    assert state.control_value(state.load_state(), "owner_external_id") == (True, None)
    kept = sorted(p.name for p in (root / "state").glob("*.pre-reset-*.json"))
    assert len(kept) == 2  # set aside, never deleted


def test_owner_bindings_have_only_known_empty_slot_writers():
    """The proof a backup's SET binding relies on (``_backup_unconfirmed``): every
    production write of an owner binding key sits behind a known-empty-slot check."""
    import ast

    keys = {"owner_id", "owner_chat_id", "owner_external_id", "owner_external_chat_id"}
    repo = pathlib.Path(__file__).resolve().parents[1]
    writers = []
    for path in [repo / "server.py", *repo.joinpath("ouroboros").rglob("*.py"), *repo.joinpath("supervisor").rglob("*.py")]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(
                node, ast.AugAssign) else []
            for target in targets:
                if (isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant)
                        and target.slice.value in keys):
                    writers.append((path.name, target.slice.value))
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"update", "pop"} and any(
                        (isinstance(arg, ast.Constant) and arg.value in keys) for arg in node.args)
                    or isinstance(node, ast.Call) and any(kw.arg in keys for kw in node.keywords)
                    and isinstance(node.func, ast.Attribute) and node.func.attr == "update"):
                writers.append((path.name, "update/pop"))
    assert sorted(writers) == sorted([("server.py", "owner_id"), ("server.py", "owner_chat_id"),
                                      ("server.py", "owner_external_id"), ("server.py", "owner_external_chat_id")])
    source = (repo / "server.py").read_text(encoding="utf-8")
    assert source.count('control_value(live, "owner_id") == (True, None)') == 1
    assert source.count('control_value(live, "owner_external_id") == (True, None)') == 1


def test_an_initialization_write_failure_is_typed_never_fatal(root, monkeypatch):
    def refuse(*_a, **_k):
        raise PermissionError(errno.EACCES, "denied")

    monkeypatch.setattr(state_initialization, "_write", refuse)
    read = state.init_state()  # the pending witness cannot be written: unavailable, supervisor keeps serving
    assert read.quality in {"uninitialized", "unavailable"} and not state.STATE_PATH.exists()


@pytest.mark.parametrize('failure', ['complete', 'backup'])
def test_interrupted_initialization_never_grants_on_repeated_boot(root, monkeypatch, failure):
    real_complete, real_write = state_initialization.complete, state.atomic_write_text
    if failure == 'complete':
        monkeypatch.setattr(state_initialization, 'complete', lambda *a, **k: False)
    else:
        def write(path, body):
            if path == state.STATE_LAST_GOOD_PATH:
                raise OSError(errno.ENOSPC, 'backup full')
            return real_write(path, body)
        monkeypatch.setattr(state, 'atomic_write_text', write)
    for _ in range(2):
        state.init(root)  # no process-local latch is relied on
        assert state.init_state().quality == 'unavailable'
        assert state.control_value(state.load_state(), 'owner_id') == (False, None)
        assert state.control_in_copy(state.STATE_PATH, 'owner_id') == (False, None)
    identity = json.loads(state.STATE_PATH.read_text())['initialization_id']
    monkeypatch.setattr(state_initialization, 'complete', real_complete)
    monkeypatch.setattr(state, 'atomic_write_text', real_write)
    assert state.init_state().quality == 'current'
    assert state.read_state().values['initialization_id'] == identity


def test_complete_witness_does_not_adopt_mismatched_primary(root):
    state.init_state()
    prior = json.loads(state.STATE_PATH.read_text())
    _write(state.STATE_PATH, {**prior, 'initialization_id': 'foreign'})
    assert state.read_state().reason == 'initialization_identity_mismatch'
    assert state.init_state().quality == 'unavailable'
    assert state_initialization.read_witness(root)[1]['initialization_id'] == prior['initialization_id']
    with pytest.raises(state.StateUnavailable):
        state.update_state(lambda st: st.update(owner_id=99))


def test_partial_state_write_is_typed_and_stop_without_campaign_is_durable(root, monkeypatch):
    from supervisor import evolution_lifecycle as lifecycle
    from supervisor.events_runtime_controls import owner_evolution_stop_controls

    state.save_state(_prior())
    real = state.atomic_write_text
    def write(path, body):
        if path == state.STATE_LAST_GOOD_PATH:
            raise OSError(errno.EIO, 'backup unavailable')
        return real(path, body)
    monkeypatch.setattr(state, 'atomic_write_text', write)
    with pytest.raises(state.StateUnavailable) as err:
        state.update_state(lambda st: st.update(last_owner_message_at='now'))
    assert err.value.reason == 'backup_write_failed' and err.value.primary_written
    assert 'runtime state' in owner_evolution_stop_controls('owner off')
    campaign = lifecycle._read_evolution_campaign()
    assert campaign['stop_intent']['reason'] == 'owner off'
    assert 'status' not in campaign and 'id' not in campaign
    lifecycle._STOP_LATCH['stopped'] = False
    assert lifecycle.evolution_stop_reason() == 'stop_intent'


def test_panic_flag_blocks_constructor_tick_and_public_wake_after_failed_disable(root, monkeypatch):
    from ouroboros.consciousness import BackgroundConsciousness
    from supervisor import worker_chat_lane

    state.save_state(_prior())
    _write(root / 'state/panic_stop.flag', b'panic')
    monkeypatch.setattr(worker_chat_lane, '_pool', lambda: SimpleNamespace(DRIVE_ROOT=root))
    monkeypatch.setattr(state, 'update_state', lambda *a, **k: (_ for _ in ()).throw(
        state.StateUnavailable('backup_write_failed', primary_written=True)))
    worker_chat_lane.auto_resume_after_restart()
    clock = BackgroundConsciousness(root, root, lambda: 7, now=0)
    assert not clock.enabled
    assert clock.tick(now=99999) == 'panic_stop'
    monkeypatch.setattr(worker_chat_lane, 'wake_gate_open', lambda: True)
    assert worker_chat_lane.handle_wake_direct(7, 'wake', {})['reason'] == 'consciousness_disabled_or_unknown'
    assert (root / 'state/panic_stop.flag').exists()


def test_recovery_uses_existing_owner_stop_evidence_without_reset(root):
    from supervisor import evolution_lifecycle as lifecycle

    state.save_state(_prior())
    assert lifecycle.record_evolution_stop_intent("owner", "stop during outage")
    _write(state.STATE_PATH, b"broken")
    state.init_state()
    restored = state.load_state()
    assert state.control_value(restored, "evolution_owner_stopped") == (True, True)
    assert state.control_value(restored, "evolution_mode_enabled") == (True, False)
    assert state.control_value(restored, "owner_chat_id") == (True, _prior()["owner_chat_id"])


def test_raw_campaign_and_state_errors_do_not_skip_actual_stop_cancellation(root, monkeypatch):
    import ouroboros.server_owner_routing as routing
    from supervisor import evolution_lifecycle, queue

    attempts = []
    def fail_campaign(*args, **kwargs):
        attempts.append("campaign")
        raise OSError(28, "full")
    def fail_state(*args, **kwargs):
        attempts.append("state")
        raise OSError(5, "io")
    monkeypatch.setattr(evolution_lifecycle, "record_evolution_stop_intent", fail_campaign)
    monkeypatch.setattr(state, "update_state", fail_state)
    monkeypatch.setattr(queue, "stop_evolution_tasks", lambda *_: attempts.append("cancel") or {})
    monkeypatch.setattr(queue, "evolution_stop_report", lambda _: ([], True))
    ctx = SimpleNamespace(DRIVE_ROOT=root, sort_pending=lambda: None, persist_queue_snapshot=lambda **_: None)
    routing._owner_evolution_stop(ctx, 1)
    assert attempts[:3] == ["campaign", "state", "cancel"]
    assert evolution_lifecycle._STOP_LATCH["stopped"] is True
