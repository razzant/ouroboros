"""Typed startup-failure classification, the durable row and the spawn latch (#844).

Offline: the manager tests drive ``OwnedClaudexorDaemon`` with a fake runtime,
a fake spawn and a synthetic clock; one POSIX test spawns a real child that
kills itself with a signal so the raw ``Popen`` return code path is proven.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from ouroboros import claudexor_daemon as owned
from ouroboros.claudexor_startup_failure import (
    ExitFact,
    StartupFailureClass,
    classify_startup_failure,
    read_startup_log_interval,
)

# Sanitized excerpts of the real OOM output in the live daemon.log (2026-09-13,
# thirty crashes at the 4095 MB V8 default old space); paths and pids stripped.
_OOM_REACHED = b"""
<--- Last few GCs --->

[<pid>:0x<addr>]    32613 ms: Mark-Compact (reduce) 4095.0 (4122.0) -> 4094.2 (4120.1) MB, pooled: 0 MB, 219.38 / 0.00 ms  (average mu = 0.373, current mu = 0.004) allocation failure; scavenge might not succeed

FATAL ERROR: Reached heap limit Allocation failed - JavaScript heap out of memory
----- Native stack trace -----

 1: 0x10445e0fc node::OOMErrorHandler(char const*, v8::OOMDetails const&) [<node>]
 2: 0x1046ba4b8 v8::internal::V8::FatalProcessOutOfMemory(v8::internal::Isolate*, char const*, v8::OOMDetails const&) [<node>]
 7: 0x104a352bc v8::internal::JsonParser<unsigned short>::MakeString(v8::internal::JsonString const&, v8::internal::Handle<v8::internal::String>) [<node>]
12: 0x104736c84 v8::internal::Builtin_JsonParse(int, unsigned long*, v8::internal::Isolate*) [<node>]
"""
_OOM_INEFFECTIVE = b"""
<--- Last few GCs --->

[<pid>:0x<addr>]    21236 ms: Mark-Compact (reduce) 4095.0 (4122.2) -> 4094.2 (4120.3) MB, pooled: 0 MB, 225.96 / 0.00 ms  (average mu = 0.299, current mu = 0.003) allocation failure; scavenge might not succeed

FATAL ERROR: Ineffective mark-compacts near heap limit Allocation failed - JavaScript heap out of memory
----- Native stack trace -----
"""
_LEASE_BUSY = b"claudexord: another claudexor daemon owns /data/claudexor/daemon/claudexord.sock.writer/active.writer\n"
_LEASE_STALE = b"claudexord: could not replace stale daemon writer lease /data/claudexor/daemon/claudexord.sock.writer\n"
_FLOOR_BELOW = b"claudexord: candidate version 3.8.2 is below the proven serving floor 3.8.3\n"
_FLOOR_UNORDERED = b"claudexord: candidate version 'dev' cannot be ordered against the proven serving floor 3.8.3\n"

_SIGNAL_EXIT = ExitFact(returncode=-6, descriptor_written=False)


# --- classifier (pure) ---------------------------------------------------------


@pytest.mark.parametrize("text, expected", [
    (_OOM_REACHED, StartupFailureClass.HEAP_EXHAUSTED),
    (_OOM_INEFFECTIVE, StartupFailureClass.HEAP_EXHAUSTED),
    (_LEASE_BUSY, StartupFailureClass.WRITER_LEASE_CONTENDED),
    (_LEASE_STALE, StartupFailureClass.WRITER_LEASE_CONTENDED),
    (_FLOOR_BELOW, StartupFailureClass.ENGINE_FLOOR),
    (_FLOOR_UNORDERED, StartupFailureClass.ENGINE_FLOOR),
    # The general V8 form (H-04 item 2): one line carrying FATAL ERROR + Allocation failed.
    (b"FATAL ERROR: CALL_AND_RETRY_LAST Allocation failed - JavaScript heap out of memory\n",
     StartupFailureClass.HEAP_EXHAUSTED),
    (b"FATAL ERROR: NewSpace::Rebalance Allocation failed - process out of memory\n",
     StartupFailureClass.HEAP_EXHAUSTED),
    (b"FATAL ERROR: v8::ToLocalChecked Empty MaybeLocal\n", StartupFailureClass.UNCLASSIFIED),
    (b"Allocation failed\nFATAL ERROR: something unrelated\n", StartupFailureClass.UNCLASSIFIED),
    (b"", StartupFailureClass.UNCLASSIFIED),
    (b"writer lease lost\n", StartupFailureClass.UNCLASSIFIED),
    (b"current fixture startup exited\nTraceback (most recent call last):\n", StartupFailureClass.UNCLASSIFIED),
])
def test_classifier_names_each_class_and_nothing_else(text, expected):
    fact = (ExitFact(1, False) if expected in {
        StartupFailureClass.WRITER_LEASE_CONTENDED, StartupFailureClass.ENGINE_FLOOR,
    } else _SIGNAL_EXIT)
    assert classify_startup_failure(text, fact) is expected


def test_classifier_reads_the_last_refusal_and_needs_an_exit():
    """Only markers matching the exit class participate in the last-marker rule."""
    assert classify_startup_failure(_FLOOR_BELOW + _OOM_REACHED, _SIGNAL_EXIT) is StartupFailureClass.HEAP_EXHAUSTED
    assert classify_startup_failure(_FLOOR_BELOW + _LEASE_BUSY, ExitFact(1, False)) is StartupFailureClass.WRITER_LEASE_CONTENDED
    assert classify_startup_failure(_LEASE_BUSY + _FLOOR_BELOW, ExitFact(1, False)) is StartupFailureClass.ENGINE_FLOOR
    running = ExitFact(returncode=None, descriptor_written=False)
    assert classify_startup_failure(_OOM_REACHED, running) is StartupFailureClass.UNCLASSIFIED


@pytest.mark.parametrize("returncode, expected", [
    (-6, StartupFailureClass.HEAP_EXHAUSTED),
    (134, StartupFailureClass.HEAP_EXHAUSTED),
    (1, StartupFailureClass.WRITER_LEASE_CONTENDED),
    (0, StartupFailureClass.UNCLASSIFIED),
    (None, StartupFailureClass.UNCLASSIFIED),
])
def test_shared_log_refusals_do_not_relabel_heap_exit(returncode, expected):
    interval = _LEASE_BUSY * 2 + _OOM_REACHED + _LEASE_BUSY * 3
    assert classify_startup_failure(interval, ExitFact(returncode, True)) is expected
    assert classify_startup_failure(_LEASE_BUSY, ExitFact(-9, True)) is StartupFailureClass.UNCLASSIFIED
    assert classify_startup_failure(_OOM_REACHED, ExitFact(1, False)) is StartupFailureClass.UNCLASSIFIED


@pytest.mark.parametrize("returncode, descriptor_written, latches, signal, code", [
    (-6, False, True, 6, None),      # V8 abort, no descriptor: the #844 shape
    (1, False, True, None, 1),       # engine refusal exit
    (0, False, False, None, 0),      # clean exit publishes nothing to latch on
    (-6, True, False, 6, None),      # crashed AFTER publishing control: not the class
    (None, False, False, None, None),  # still running: no exit fact at all
])
def test_exit_fact_latches_only_a_failed_start_without_control(returncode, descriptor_written, latches, signal, code):
    fact = ExitFact(returncode=returncode, descriptor_written=descriptor_written)
    assert fact.failed_without_control is latches
    assert fact.signal == signal and fact.exit_code == code
    assert fact.exited is (returncode is not None)


def test_log_interval_reader_returns_only_this_spawns_bytes(tmp_path):
    log = tmp_path / "daemon.log"
    old = b"old runtime 3.8.2 failed for an unrelated reason\n"
    log.write_bytes(old)
    stat = log.stat()
    identity = (stat.st_dev, stat.st_ino)
    with open(log, "ab") as sink:
        sink.write(_OOM_REACHED)
    interval, data = read_startup_log_interval(log, start=len(old), identity=identity)
    assert interval == (len(old), len(old) + len(_OOM_REACHED)) and data == _OOM_REACHED
    # Bounded: only the tail of a long interval is read, and it still classifies.
    interval, data = read_startup_log_interval(log, start=len(old), identity=identity, limit=64)
    assert interval == (len(old), len(old) + len(_OOM_REACHED)) and data == _OOM_REACHED[-64:]
    # A replaced file is never read as this spawn's interval. The replacement
    # is written beside the live file and renamed over it (the shape every log
    # rotation takes), so its inode is allocated while the old one still exists
    # and differs on every filesystem; unlink-then-create would let ext4 hand
    # the recycled inode number back and make the two files indistinguishable.
    replacement = tmp_path / "daemon.log.rotated"
    replacement.write_bytes(old + _OOM_REACHED)
    os.replace(replacement, log)
    assert log.stat().st_ino != identity[1]
    assert read_startup_log_interval(log, start=len(old), identity=identity) == (None, b"")
    assert read_startup_log_interval(tmp_path / "missing.log", start=0, identity=identity) == (None, b"")


# --- manager: latch, refusal, release, rows -------------------------------------

def _point_owned_home(monkeypatch, config_dir: pathlib.Path, data_dir: pathlib.Path) -> None:
    monkeypatch.setattr(owned, "owned_config_dir", lambda: config_dir)
    monkeypatch.setattr(owned, "owned_descriptor_path",
                        lambda: config_dir / "daemon" / "control-api.json")
    monkeypatch.setattr(owned, "owned_daemon_provisioned",
                        lambda: (config_dir / "daemon" / "control-api.json").is_file())
    import ouroboros.config as config_mod
    monkeypatch.setattr(config_mod, "DATA_DIR", data_dir)


def _rows(data_dir: pathlib.Path) -> list:
    path = data_dir / "logs" / "supervisor.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class _Stand:
    """One synthetic failing-start stand: fake runtime, fake spawn, synthetic clock.

    ``exit_code`` is what the spawned child's ``poll()`` answers and may be
    changed between calls (``None`` = still running).
    """

    def __init__(self, monkeypatch, tmp_path, *, returncode, banner, write_descriptor=False):
        import ouroboros.process_custody as custody_mod
        from ouroboros import claudexor_runtime as runtime
        from ouroboros.gateways import claudexor as gateway_mod
        from ouroboros.gateways.claudexor import ClaudexorUnavailable, DaemonEndpoint

        self.data_dir = tmp_path / "data"
        self.config_dir = self.data_dir / "claudexor"
        _point_owned_home(monkeypatch, self.config_dir, self.data_dir)
        monkeypatch.setattr(owned, "verify_owned_home", lambda **_kw: "")
        monkeypatch.setattr(owned, "_SPAWN_WAIT_SEC", 0.03)
        monkeypatch.setattr(owned, "_SPAWN_POLL_SEC", 0.01)

        class Clock:
            now = 0.0

            def monotonic(self):
                return self.now

            def sleep(self, seconds):
                self.now += seconds

        monkeypatch.setattr(owned, "time", Clock())
        self.ensures, self.spawned = [], []
        self.exit_code = returncode
        stand = self

        class ReadyRuntime:
            pin = SimpleNamespace(version="3.11.0", build_sha="b" * 40)

            def ensure(self):
                stand.ensures.append(1)
                return ["/fixture/node", "/fixture/claudexord.bundle.cjs"]

            def status(self, **_kwargs):
                return {"source": "download", "version": "3.11.0", "build_sha": "b" * 40}

        self.runtime = ReadyRuntime()
        monkeypatch.setattr(runtime, "get_runtime_manager", lambda: self.runtime)
        self.endpoint = DaemonEndpoint(host="127.0.0.1", port=45690, token="fixture-token")
        monkeypatch.setattr(gateway_mod, "discover_daemon_at", lambda _home: self.endpoint)

        class UnreachableGateway:
            def __init__(self, _endpoint):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                pass

            def handshake(self, *, timeout_sec=None):
                raise ClaudexorUnavailable("daemon_unreachable", "not ready")

        monkeypatch.setattr(gateway_mod, "ClaudexorGateway", UnreachableGateway)

        class ExitedChild:
            def __init__(self, pid):
                self.pid = pid

            def poll(self):
                return stand.exit_code

            def terminate(self):
                raise AssertionError("an exited child must not be terminated")

        def spawn(*_args, **kwargs):
            kwargs["stdout"].write(banner)
            kwargs["stdout"].flush()
            if write_descriptor:
                descriptor = self.config_dir / "daemon" / "control-api.json"
                descriptor.parent.mkdir(parents=True, exist_ok=True)
                (descriptor.parent / "token").write_text("fixture-token", encoding="utf-8")
                descriptor.write_text(json.dumps({"host": "127.0.0.1", "port": 45690,
                                                  "tokenPath": str(descriptor.parent / "token")}),
                                      encoding="utf-8")
            child = ExitedChild(424250 + len(self.spawned))
            self.spawned.append(child)
            return child

        monkeypatch.setattr(custody_mod, "spawn_supervised", spawn)
        self.manager = owned.OwnedClaudexorDaemon()

    def fail_once(self):
        from ouroboros.gateways.claudexor import ClaudexorUnavailable

        with pytest.raises(ClaudexorUnavailable) as err:
            self.manager.ensure_running()
        return err.value


def test_signal_exit_without_descriptor_latches_and_refuses_without_a_second_spawn(monkeypatch, tmp_path):
    stand = _Stand(monkeypatch, tmp_path, returncode=-6, banner=_OOM_REACHED)
    first = stand.fail_once()
    assert first.code == "daemon_spawn_failed"
    text = str(first)
    assert "startup_failure=heap_exhausted" in text and "exit_signal=6" in text
    assert "descriptor_written=False" in text
    assert f"startup log interval=0..{len(_OOM_REACHED)} bytes" in text
    assert "Mark-Compact" not in text and "FATAL ERROR" not in text, "log bytes are never quoted"
    assert len(stand.spawned) == 1 and len(stand.ensures) == 1
    latch = stand.manager._last_start_failure
    assert latch is not None and latch["classification"] == "heap_exhausted"
    assert latch["exit_signal"] == 6 and latch["exit_code"] is None
    assert latch["pin_version"] == "3.11.0" and latch["pin_build"] == "b" * 40
    assert latch["log_interval"] == [0, len(_OOM_REACHED)]
    assert latch["log_path"] == str(stand.config_dir / "daemon.log") and latch["at"]
    assert stand.manager._proc is None and stand.manager._startup_attempt == {}

    # Every ordinary caller is refused typed, immediately: no preparation, no spawn.
    second = stand.fail_once()
    assert second.code == "daemon_spawn_failed" and second.status_code == 503
    assert "latched" in str(second) and "startup_failure=heap_exhausted" in str(second)
    assert "selected_version=3.11.0" in str(second) and "exit_signal=6" in str(second)
    assert len(stand.spawned) == 1 and len(stand.ensures) == 1
    assert stand.manager.status_dict()["last_error"].startswith("daemon_spawn_failed: startup_failure=heap_exhausted")

    rows = _rows(stand.data_dir)
    assert [row["type"] for row in rows] == ["claudexor_daemon_start_failed"]
    row = rows[0]
    assert row["purpose"] == owned.CUSTODY_PURPOSE and row["latched"] is True
    assert row["classification"] == "heap_exhausted" and row["exit_signal"] == 6 and row["exit_code"] is None
    assert row["pin_version"] == "3.11.0" and row["pin_build"] == "b" * 40
    assert row["log_path"] == str(stand.config_dir / "daemon.log")
    assert row["log_interval"] == [0, len(_OOM_REACHED)] and row["ts"] == latch["at"]


def test_the_supervisor_sweep_release_permits_exactly_one_retry(monkeypatch, tmp_path):
    stand = _Stand(monkeypatch, tmp_path, returncode=1, banner=_LEASE_BUSY)
    stand.fail_once()
    stand.fail_once()
    assert len(stand.spawned) == 1
    assert stand.manager.clear_start_failure_latch(cleared_by="supervisor_sweep") is True
    assert stand.manager._last_start_failure is None
    assert stand.manager.clear_start_failure_latch(cleared_by="supervisor_sweep") is False
    retried = stand.fail_once()
    assert len(stand.spawned) == 2 and len(stand.ensures) == 2
    assert "startup_failure=writer_lease_contended" in str(retried) and "exit_code=1" in str(retried)
    assert stand.manager._last_start_failure is not None
    types = [(row["type"], row.get("cleared_by")) for row in _rows(stand.data_dir)]
    assert types == [
        ("claudexor_daemon_start_failed", None),
        ("claudexor_daemon_start_latch_cleared", "supervisor_sweep"),
        ("claudexor_daemon_start_failed", None),
    ]
    cleared = _rows(stand.data_dir)[1]
    assert cleared["purpose"] == owned.CUSTODY_PURPOSE
    assert cleared["classification"] == "writer_lease_contended" and cleared["pin_version"] == "3.11.0"
    assert cleared["failed_at"] == _rows(stand.data_dir)[0]["ts"]


def test_a_live_daemon_started_by_anyone_clears_the_latch(monkeypatch, tmp_path):
    stand = _Stand(monkeypatch, tmp_path, returncode=-6, banner=_OOM_INEFFECTIVE)
    stand.fail_once()
    assert stand.manager._last_start_failure is not None
    manager = stand.manager
    monkeypatch.setattr(manager, "_classify_liveness", lambda **_kw: (stand.endpoint, "running", ""))
    manager._engine_version, manager._engine_build_sha = "3.11.0", "b" * 40
    assert manager.ensure_running() is stand.endpoint
    assert manager._last_start_failure is None and len(stand.spawned) == 1
    rows = _rows(stand.data_dir)
    assert rows[-1]["type"] == "claudexor_daemon_start_latch_cleared"
    assert rows[-1]["cleared_by"] == "live_daemon_attached" and rows[-1]["classification"] == "heap_exhausted"
    # Nothing latched: an ordinary later attach writes no release row.
    assert manager.ensure_running() is stand.endpoint and len(_rows(stand.data_dir)) == 2


def test_a_clean_exit_is_classified_and_rowed_but_never_latched(monkeypatch, tmp_path):
    stand = _Stand(monkeypatch, tmp_path, returncode=0, banner=b"claudexord: exited cleanly\n")
    first = stand.fail_once()
    assert "startup_failure=unclassified" in str(first) and "exit_code=0" in str(first)
    assert stand.manager._last_start_failure is None
    stand.fail_once()
    assert len(stand.spawned) == 2, "no latch: the next caller spawns again"
    rows = _rows(stand.data_dir)
    assert [row["latched"] for row in rows] == [False, False]
    assert {row["classification"] for row in rows} == {"unclassified"}


def test_a_crash_after_publishing_control_is_not_the_latched_class(monkeypatch, tmp_path):
    stand = _Stand(monkeypatch, tmp_path, returncode=-11, banner=_OOM_REACHED, write_descriptor=True)
    first = stand.fail_once()
    assert "exit_signal=11" in str(first) and "descriptor_written=True" in str(first)
    assert stand.manager._last_start_failure is None
    stand.fail_once()
    assert len(stand.spawned) == 2
    assert [row["latched"] for row in _rows(stand.data_dir)] == [False, False]
    assert _rows(stand.data_dir)[0]["descriptor_written"] is True


def test_a_child_that_dies_after_the_callers_wait_is_settled_by_the_next_caller(monkeypatch, tmp_path):
    """The real cadence: V8 dies after the startup window, with nobody waiting."""
    stand = _Stand(monkeypatch, tmp_path, returncode=None, banner=_OOM_REACHED)  # alive through the wait
    first = stand.fail_once()
    assert first.code == "daemon_starting" and "retry joins the same startup" in str(first)
    assert len(stand.spawned) == 1 and stand.manager._proc is stand.spawned[0]
    assert not _rows(stand.data_dir) and stand.manager._last_start_failure is None
    stand.exit_code = -6  # the OOM abort lands between callers
    second = stand.fail_once()
    assert second.code == "daemon_spawn_failed" and "latched" in str(second)
    assert "startup_failure=heap_exhausted" in str(second) and "exit_signal=6" in str(second)
    assert len(stand.spawned) == 1 and len(stand.ensures) == 1, "no silent respawn, no preparation"
    assert stand.manager._proc is None and stand.manager._startup_attempt == {}
    latch = stand.manager._last_start_failure
    assert latch is not None and latch["classification"] == "heap_exhausted" and latch["exit_signal"] == 6
    rows = _rows(stand.data_dir)
    assert [row["type"] for row in rows] == ["claudexor_daemon_start_failed"] and rows[0]["latched"] is True
    third = stand.fail_once()
    assert "latched" in str(third) and len(stand.spawned) == 1 and len(_rows(stand.data_dir)) == 1


def test_an_own_child_reaped_on_the_attach_path_is_rowed_but_never_latched(monkeypatch, tmp_path):
    """A crash after publishing control, found by the next attach: one row, no latch."""
    stand = _Stand(monkeypatch, tmp_path, returncode=None, banner=_OOM_REACHED)
    assert stand.fail_once().code == "daemon_starting"
    descriptor = stand.config_dir / "daemon" / "control-api.json"
    descriptor.parent.mkdir(parents=True, exist_ok=True)
    (descriptor.parent / "token").write_text("fixture-token", encoding="utf-8")
    descriptor.write_text(json.dumps({"host": "127.0.0.1", "port": 45690,
                                      "tokenPath": str(descriptor.parent / "token")}), encoding="utf-8")
    stand.exit_code = -11
    manager = stand.manager
    monkeypatch.setattr(manager, "_classify_liveness", lambda **_kw: (stand.endpoint, "running", ""))
    manager._engine_version, manager._engine_build_sha = "3.11.0", "b" * 40
    assert manager.ensure_running() is stand.endpoint
    assert manager._proc is None and manager._startup_attempt == {} and manager._last_start_failure is None
    rows = _rows(stand.data_dir)
    assert [row["type"] for row in rows] == ["claudexor_daemon_start_failed"]
    assert rows[0]["latched"] is False and rows[0]["descriptor_written"] is True
    assert rows[0]["exit_signal"] == 11 and rows[0]["classification"] == "heap_exhausted"
    assert len(stand.spawned) == 1


def _gate_log_read(monkeypatch):
    """Hold a harvest inside its log read so something else can happen meanwhile."""
    import threading

    real = owned.read_startup_log_interval
    entered, release = threading.Event(), threading.Event()

    def gated(*args, **kwargs):
        if kwargs.get("limit") == 0:  # the diagnostic's bounds-only read is not the window
            return real(*args, **kwargs)
        entered.set()
        assert release.wait(5), "the test must release the harvest"
        return real(*args, **kwargs)

    monkeypatch.setattr(owned, "read_startup_log_interval", gated)
    return entered, release


def _call_in_thread(manager, outcomes: dict, name: str):
    import threading

    def call():
        try:
            manager.ensure_running()
            outcomes[name] = "endpoint"
        except Exception as exc:  # noqa: BLE001 - the typed refusal is the outcome under test
            outcomes[name] = exc

    thread = threading.Thread(target=call, name=name)
    thread.start()
    return thread


def test_a_concurrent_caller_in_the_harvest_window_meets_the_latch_not_a_spawn_slot(monkeypatch, tmp_path):
    """The latch is taken under the pop's lock, before the log is read (H-03 item 1)."""
    stand = _Stand(monkeypatch, tmp_path, returncode=None, banner=_OOM_REACHED)
    assert stand.fail_once().code == "daemon_starting"
    stand.exit_code = -6  # unwatched death
    entered, release = _gate_log_read(monkeypatch)
    outcomes: dict = {}
    first = _call_in_thread(stand.manager, outcomes, "first")
    assert entered.wait(5), "the first caller is harvesting"
    second = _call_in_thread(stand.manager, outcomes, "second")
    second.join(5)
    assert not second.is_alive(), "the second caller must not wait on the harvest"
    assert getattr(outcomes["second"], "code", None) == "daemon_spawn_failed"
    assert "latched" in str(outcomes["second"]) and "startup_failure=unclassified" in str(outcomes["second"])
    assert "startup log interval pending" in str(outcomes["second"]), "provisional: the log is not read yet"
    release.set()
    first.join(5)
    assert getattr(outcomes["first"], "code", None) == "daemon_spawn_failed"
    assert "startup_failure=heap_exhausted" in str(outcomes["first"])
    assert len(stand.spawned) == 1 and len(stand.ensures) == 1, "exactly one spawn"
    rows = _rows(stand.data_dir)
    assert [row["type"] for row in rows] == ["claudexor_daemon_start_failed"], "exactly one row"
    assert rows[0]["classification"] == "heap_exhausted" and rows[0]["latched"] is True
    latch = stand.manager._last_start_failure
    assert latch["classification"] == "heap_exhausted" and latch["log_interval"] == [0, len(_OOM_REACHED)]


def test_a_waiting_caller_whose_child_was_settled_by_another_still_names_the_fact(monkeypatch, tmp_path):
    """H-04 item 5: the child dies inside the waiting caller's window; a concurrent caller
    settles it first; the waiting caller's wait-expiry refusal still carries the fact."""
    import threading

    stand = _Stand(monkeypatch, tmp_path, returncode=None, banner=_OOM_REACHED)
    manager = stand.manager
    entered, release = threading.Event(), threading.Event()

    def held_probe(*, timeout_sec=None):
        if stand.spawned:  # only the startup-wait probes, not the pre-spawn re-probe
            entered.set()
            assert release.wait(5), "the test must release the waiting caller"
        return None

    monkeypatch.setattr(manager, "_alive_endpoint", held_probe)
    outcomes: dict = {}
    threads = [_call_in_thread(manager, outcomes, "waiting")]
    try:
        assert entered.wait(5), "the waiting caller spawned and is inside its startup wait"
        assert len(stand.spawned) == 1
        stand.exit_code = -6  # the child dies inside the window
        monkeypatch.setattr(manager, "_alive_endpoint", lambda *, timeout_sec=None: None)
        threads.append(_call_in_thread(manager, outcomes, "settler"))
        threads[-1].join(5)
        assert getattr(outcomes.get("settler"), "code", None) == "daemon_spawn_failed"
        assert "startup_failure=heap_exhausted" in str(outcomes["settler"])
    finally:
        release.set()
        for thread in threads:
            thread.join(5)
    assert getattr(outcomes.get("waiting"), "code", None) == "daemon_spawn_failed"
    text = str(outcomes["waiting"])
    assert "latched: startup_failure=heap_exhausted" in text and "exit_signal=6" in text, \
        "the fallback label reads as manager state, not as this startup's own fact"
    assert "joining another manager" not in text and "settled by a concurrent caller" in text
    assert len(stand.spawned) == 1 and len(_rows(stand.data_dir)) == 1


def test_a_sweep_release_during_the_harvest_window_is_never_re_latched(monkeypatch, tmp_path):
    stand = _Stand(monkeypatch, tmp_path, returncode=None, banner=_LEASE_BUSY)
    assert stand.fail_once().code == "daemon_starting"
    stand.exit_code = 1
    entered, release = _gate_log_read(monkeypatch)
    outcomes: dict = {}
    harvesting = _call_in_thread(stand.manager, outcomes, "harvesting")
    assert entered.wait(5)
    assert stand.manager._last_start_failure is not None, "provisionally latched before the read"
    assert stand.manager.clear_start_failure_latch(cleared_by="supervisor_sweep") is True
    assert stand.manager._last_start_failure is None
    stand.exit_code = None  # the caller's own retry spawns a child that stays alive
    release.set()
    harvesting.join(5)
    assert getattr(outcomes["harvesting"], "code", None) == "daemon_starting"
    assert stand.manager._last_start_failure is None, "the classified record never re-takes a released latch"
    assert len(stand.spawned) == 2
    rows = _rows(stand.data_dir)
    assert [(row["type"], row.get("cleared_by")) for row in rows] == [
        ("claudexor_daemon_start_latch_cleared", "supervisor_sweep"),
        ("claudexor_daemon_start_failed", None),
    ]
    assert rows[0]["classification"] == "unclassified", "released while still provisional"
    assert rows[1]["classification"] == "writer_lease_contended" and rows[1]["latched"] is True


def test_an_owner_stop_after_an_unwatched_death_still_records_the_failure(monkeypatch, tmp_path):
    """H-03 item 3: Restart/Panic go through _terminate_child; the row must not be lost."""
    stand = _Stand(monkeypatch, tmp_path, returncode=None, banner=_OOM_REACHED)
    assert stand.fail_once().code == "daemon_starting"
    stand.exit_code = -6
    assert stand.manager.stop_outcome() == "nothing_to_stop"
    assert stand.manager._proc is None and stand.manager._startup_attempt == {}
    rows = _rows(stand.data_dir)
    assert [row["type"] for row in rows] == ["claudexor_daemon_start_failed"]
    assert rows[0]["latched"] is True and rows[0]["classification"] == "heap_exhausted"
    assert rows[0]["exit_signal"] == 6 and stand.manager._last_start_failure is not None


def test_owner_refresh_releases_the_latch_and_makes_one_ordinary_ensure(monkeypatch, tmp_path):
    """Owner decision D8: the explicit Refresh clears the latch and spawns once; a second press joins."""
    import asyncio

    from ouroboros import claudexor_daemon as daemon_mod
    from ouroboros.gateway import claudexor_accounts as accounts

    stand = _Stand(monkeypatch, tmp_path, returncode=-6, banner=_OOM_REACHED)
    stand.fail_once()
    assert stand.manager._last_start_failure is not None and len(stand.spawned) == 1
    monkeypatch.setattr(daemon_mod, "get_owned_daemon", lambda: stand.manager)
    monkeypatch.setattr(accounts, "_status_payload", lambda include_models: {"daemon": {"state": "stale"}})
    stand.exit_code = None  # the child the Refresh spawns stays alive (starting)
    first = asyncio.run(accounts.api_claudexor_wake(object()))
    assert first.status_code == 503 and "daemon_starting" in json.loads(first.body)["error"]
    assert len(stand.spawned) == 2 and len(stand.ensures) == 2, "exactly one spawn attempt by the Refresh"
    assert stand.manager._last_start_failure is None
    rows = _rows(stand.data_dir)
    assert [(row["type"], row.get("cleared_by")) for row in rows] == [
        ("claudexor_daemon_start_failed", None),
        ("claudexor_daemon_start_latch_cleared", "owner_wake"),
    ]
    second = asyncio.run(accounts.api_claudexor_wake(object()))
    assert second.status_code == 503 and "daemon_starting" in json.loads(second.body)["error"]
    assert len(stand.spawned) == 2, "a second press joins the live startup: no second spawn"
    assert len(_rows(stand.data_dir)) == 2, "nothing latched, nothing released"


def test_owner_refresh_on_a_still_crashing_engine_costs_one_spawn_and_re_latches(monkeypatch, tmp_path):
    import asyncio

    from ouroboros import claudexor_daemon as daemon_mod
    from ouroboros.gateway import claudexor_accounts as accounts

    stand = _Stand(monkeypatch, tmp_path, returncode=-6, banner=_OOM_REACHED)
    stand.fail_once()
    monkeypatch.setattr(daemon_mod, "get_owned_daemon", lambda: stand.manager)
    monkeypatch.setattr(accounts, "_status_payload", lambda include_models: {})
    response = asyncio.run(accounts.api_claudexor_wake(object()))
    assert response.status_code == 503
    error = json.loads(response.body)["error"]
    assert error.startswith("daemon_spawn_failed") and "startup_failure=heap_exhausted" in error
    assert len(stand.spawned) == 2, "an explicit owner action costs one spawn"
    assert stand.manager._last_start_failure is not None, "a still-crashing engine re-latches"
    assert [(row["type"], row.get("cleared_by")) for row in _rows(stand.data_dir)] == [
        ("claudexor_daemon_start_failed", None),
        ("claudexor_daemon_start_latch_cleared", "owner_wake"),
        ("claudexor_daemon_start_failed", None),
    ]


def test_a_child_the_stop_itself_kills_is_not_a_startup_failure(monkeypatch, tmp_path):
    """H-05 item 1 (pins the H-04 correction): the settle runs at the head of ``stop_outcome``,
    BEFORE the stop's own signalling — a live starting child that the ledger pass SIGTERMs
    must not be rowed or latched. Under the H-03 placement (settle inside ``_terminate_child``)
    this test fails with one false ``latched=true`` row and a set latch."""
    import ouroboros.process_custody as custody_mod

    stand = _Stand(monkeypatch, tmp_path, returncode=None, banner=_OOM_REACHED)
    assert stand.fail_once().code == "daemon_starting"
    child = stand.spawned[0]
    monkeypatch.setattr(custody_mod, "process_stop_snapshot", lambda root, purposes: [])
    monkeypatch.setattr(custody_mod, "pending_process_stops", lambda root, purposes: [])

    def stop_by_signal(root, purposes, *, unconfirmed, expected_entries):
        stand.exit_code = -15  # the stop's own SIGTERM lands before _terminate_child looks
        return [child.pid]

    monkeypatch.setattr(custody_mod, "stop_ledgered_processes", stop_by_signal)
    assert stand.manager.stop_outcome() == "stopped"
    assert _rows(stand.data_dir) == [], "a deliberate stop is not a startup failure"
    assert stand.manager._last_start_failure is None and stand.manager._proc is None


def test_a_latch_taken_during_a_callers_runtime_preparation_still_refuses_its_spawn(monkeypatch, tmp_path):
    """Pins the second refusal, right before ``_spawn``: caller A is held inside runtime
    preparation while B's child dies and C settles it (latch set); A must not spawn."""
    import threading

    stand = _Stand(monkeypatch, tmp_path, returncode=None, banner=_OOM_REACHED)
    manager = stand.manager
    real_ensure = stand.runtime.ensure
    entered, release = threading.Event(), threading.Event()

    def gated_ensure():
        command = real_ensure()
        if len(stand.ensures) == 1:  # caller A: the first preparation is held open
            entered.set()
            assert release.wait(5), "the test must release caller A"
        return command

    monkeypatch.setattr(stand.runtime, "ensure", gated_ensure)
    outcomes: dict = {}
    caller_a = _call_in_thread(manager, outcomes, "A")
    try:
        assert entered.wait(5), "caller A is inside runtime preparation"
        assert stand.fail_once().code == "daemon_starting", "caller B spawned a live child"
        assert len(stand.spawned) == 1 and len(stand.ensures) == 2
        stand.exit_code = -6
        third = stand.fail_once()  # caller C settles B's dead child: the latch is set
        assert third.code == "daemon_spawn_failed" and "latched" in str(third)
        assert manager._last_start_failure is not None
    finally:
        release.set()
        caller_a.join(5)
    assert getattr(outcomes.get("A"), "code", None) == "daemon_spawn_failed"
    assert "latched" in str(outcomes["A"]) and "startup_failure=heap_exhausted" in str(outcomes["A"])
    assert len(stand.spawned) == 1, "A never spawned a second child"
    assert len(stand.ensures) == 2, "A's own preparation plus B's; C was refused before preparing"
    assert len(_rows(stand.data_dir)) == 1


def _hold_first_spawn_before_its_lock(monkeypatch, stand):
    """Gate the FIRST caller that enters ``_spawn`` at the marker write, i.e. after its pre-spawn
    refusal and before the spawn lock; later callers pass through untouched."""
    import threading

    entered, release = threading.Event(), threading.Event()
    real_marker = owned._write_ownership_marker
    held = []

    def gated_marker():
        real_marker()
        if not held:
            held.append(True)
            entered.set()
            assert release.wait(5), "the test must release the held caller"

    monkeypatch.setattr(owned, "_write_ownership_marker", gated_marker)
    return entered, release


def test_a_latch_taken_between_the_pre_spawn_check_and_the_spawn_lock_still_refuses(monkeypatch, tmp_path):
    """Pins the re-check under ``_spawn``'s own lock (R-01 scope finding): caller A passed its
    pre-spawn refusal with no own child, B then spawned, B's child died and C settled it (latch set)
    before A took the spawn lock; A must meet the latch, not a free spawn slot."""
    stand = _Stand(monkeypatch, tmp_path, returncode=None, banner=_OOM_REACHED)
    manager = stand.manager
    entered, release = _hold_first_spawn_before_its_lock(monkeypatch, stand)
    outcomes: dict = {}
    caller_a = _call_in_thread(manager, outcomes, "A")
    try:
        assert entered.wait(5), "caller A is inside _spawn, before its lock"
        assert stand.fail_once().code == "daemon_starting", "caller B spawned a live child"
        assert len(stand.spawned) == 1
        stand.exit_code = -6
        third = stand.fail_once()  # caller C settles B's dead child: the latch is set
        assert third.code == "daemon_spawn_failed" and "latched" in str(third)
        assert manager._last_start_failure is not None
    finally:
        release.set()
        caller_a.join(5)
    assert getattr(outcomes.get("A"), "code", None) == "daemon_spawn_failed"
    assert "latched" in str(outcomes["A"]) and "startup_failure=heap_exhausted" in str(outcomes["A"])
    assert len(stand.spawned) == 1, "A never spawned a second child under the latch"
    assert len(_rows(stand.data_dir)) == 1


def test_an_exited_but_unsettled_child_is_never_overwritten_by_a_spawn(monkeypatch, tmp_path):
    """Same window, nobody settled yet: A reaches the spawn lock while B's child is dead but
    unreaped. A must not replace it (its exit fact would be lost); A's own wait settles it."""
    stand = _Stand(monkeypatch, tmp_path, returncode=None, banner=_OOM_REACHED)
    manager = stand.manager
    entered, release = _hold_first_spawn_before_its_lock(monkeypatch, stand)
    outcomes: dict = {}
    caller_a = _call_in_thread(manager, outcomes, "A")
    try:
        assert entered.wait(5), "caller A is inside _spawn, before its lock"
        assert stand.fail_once().code == "daemon_starting", "caller B spawned a live child"
        stand.exit_code = -6  # B's child dies; no caller settles it before A proceeds
    finally:
        release.set()
        caller_a.join(5)
    assert getattr(outcomes.get("A"), "code", None) == "daemon_spawn_failed"
    assert "startup_failure=heap_exhausted" in str(outcomes["A"]) and "latched: " not in str(outcomes["A"])
    assert len(stand.spawned) == 1, "A joined the dead child's fact instead of overwriting it"
    assert manager._last_start_failure is not None and manager._proc is None
    assert len(_rows(stand.data_dir)) == 1


def test_a_joined_peer_startup_that_vanished_has_no_exit_fact(monkeypatch, tmp_path):
    """Only this manager's own child carries an exit fact; joining never latches."""
    stand = _Stand(monkeypatch, tmp_path, returncode=-6, banner=_OOM_REACHED)
    manager = stand.manager
    calls = iter([{424299}, set()])  # joinable at entry, gone by expiry
    monkeypatch.setattr(manager, "_startup_pids", lambda: next(calls, set()))
    failed = stand.fail_once()
    assert failed.code == "daemon_spawn_failed" and "joining another manager" in str(failed)
    assert "startup_failure=" not in str(failed)
    assert manager._last_start_failure is None and not stand.spawned and not _rows(stand.data_dir)


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal exit")
def test_a_real_child_killed_by_a_signal_carries_the_raw_return_code(monkeypatch, tmp_path):
    """Real ``Popen``: a child that dies by SIGTERM after writing the banner into the sink."""
    import ouroboros.process_custody as custody_mod
    from ouroboros import claudexor_runtime as runtime
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    data_dir = tmp_path / "data"
    config_dir = data_dir / "claudexor"
    _point_owned_home(monkeypatch, config_dir, data_dir)
    monkeypatch.setattr(owned, "verify_owned_home", lambda **_kw: "")
    monkeypatch.setattr(owned, "_SPAWN_POLL_SEC", 0.02)
    child_src = (
        "import os, signal, sys, time\n"
        f"sys.stderr.write({_OOM_REACHED.decode('ascii')!r}); sys.stderr.flush()\n"
        "os.kill(os.getpid(), signal.SIGTERM); time.sleep(5)\n"
    )
    monkeypatch.setattr(runtime, "get_runtime_manager", lambda: SimpleNamespace(
        ensure=lambda: [sys.executable, "-c", child_src],
        status=lambda: {"source": "fixture", "version": "9.9.9", "build_sha": "c" * 40},
    ))
    procs = []

    def plain_spawn(command, **kwargs):
        proc = subprocess.Popen(command, stdin=kwargs["stdin"], stdout=kwargs["stdout"],
                                stderr=kwargs["stderr"], env=kwargs["env"])
        procs.append(proc)
        return proc

    monkeypatch.setattr(custody_mod, "spawn_supervised", plain_spawn)
    manager = owned.OwnedClaudexorDaemon()
    try:
        with pytest.raises(ClaudexorUnavailable) as err:
            manager.ensure_running(startup_wait_sec=2.0)
    finally:
        for proc in procs:
            proc.wait(timeout=5)
    text = str(err.value)
    assert err.value.code == "daemon_spawn_failed"
    assert "startup_failure=heap_exhausted" in text and "exit_signal=15" in text
    assert "FATAL ERROR" not in text
    latch = manager._last_start_failure
    assert latch is not None and latch["exit_signal"] == 15 and latch["pin_version"] == "9.9.9"
    assert latch["log_interval"] == [0, len(_OOM_REACHED)]
    assert (config_dir / "daemon.log").read_bytes() == _OOM_REACHED


# --- the supervisor sweep and the owner's Refresh are the retriers --------------

def _track_sweep_threads(monkeypatch) -> list:
    """Capture the threads the sweep starts (the retry runs on one) so tests can join them."""
    import threading

    from ouroboros import server_maintenance as sm

    threads: list = []

    def tracked(**kwargs):
        thread = threading.Thread(**kwargs)
        threads.append(thread)
        return thread

    monkeypatch.setattr(sm, "threading", SimpleNamespace(Thread=tracked))
    return threads


def _run_real_sweep(monkeypatch, manager, order: list) -> list:
    """Drive the real 600 s tick against ``manager`` with NO delegated-run work.

    The reconcile step is a recorder: the real one ensures a gateway only when
    it has orphan work, so with none it is exactly a no-op here — which is why
    the retry must be the sweep's own. The block itself now runs on the
    ``custody-maintenance`` daemon thread (INV-B), so the returned list starts
    with it; both threads are joined before returning, so callers assert on a
    settled state.
    """
    from ouroboros import claudexor_daemon as daemon_mod
    from ouroboros import process_custody as pc
    from ouroboros import server_maintenance as sm
    from supervisor import queue

    monkeypatch.setattr(sm, "_CUSTODY_SWEEP_LOCK", __import__("threading").Lock())  # own latch: a pass may outlive a test
    threads = _track_sweep_threads(monkeypatch)
    monkeypatch.setattr(sm, "_LAST_CANCEL_INTENT_SWEEP", [time.time()])  # 20 s cadence idle
    monkeypatch.setattr(sm, "_installed_skill_names", lambda: None)
    monkeypatch.setattr(pc, "reap_orphaned_processes", lambda root, **kw: order.append("reap") or [])
    monkeypatch.setattr(sm, "_reconcile_delegated_runs",
                        lambda live, **kwargs: order.append("reconcile"))
    monkeypatch.setattr(sm, "_cursor_refresh_settled_terminals", lambda *a, **kwargs: None)
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(daemon_mod, "get_owned_daemon", lambda: manager)
    sm._periodic_supervisor_maintenance([0.0], [time.time()])
    for thread in threads:
        thread.join(5)
    assert all(not thread.is_alive() for thread in threads)
    return threads


def test_the_sweep_itself_makes_the_one_retry_after_releasing_the_latch(monkeypatch, tmp_path, caplog):
    """D5: the sweep (with the owner's Refresh) retries — ordinary callers never pay the retry."""
    import logging

    stand = _Stand(monkeypatch, tmp_path, returncode=-6, banner=_OOM_REACHED)
    stand.fail_once()
    assert len(stand.spawned) == 1 and stand.manager._last_start_failure is not None
    order: list = []
    with caplog.at_level(logging.WARNING):
        threads = _run_real_sweep(monkeypatch, stand.manager, order)
    assert order == ["reap", "reconcile"], "the swallowed refusal never skips the reconcile"
    assert [thread.name for thread in threads] == ["custody-maintenance", "owned-daemon-latch-retry"]
    assert len(stand.spawned) == 2 and len(stand.ensures) == 2, "exactly one retry, made by the sweep"
    assert stand.manager._last_start_failure is not None, "the failed retry re-latched"
    assert any("retry after latch release refused (daemon_spawn_failed)" in rec.getMessage()
               for rec in caplog.records)
    assert [(row["type"], row.get("cleared_by")) for row in _rows(stand.data_dir)] == [
        ("claudexor_daemon_start_failed", None),
        ("claudexor_daemon_start_latch_cleared", "supervisor_sweep"),
        ("claudexor_daemon_start_failed", None),
    ]
    refused = stand.fail_once()
    assert "latched" in str(refused) and "startup_failure=heap_exhausted" in str(refused)
    assert len(stand.spawned) == 2 and len(stand.ensures) == 2, "an ordinary caller still never spawns"


def test_a_healthy_sweep_never_ensures_or_spawns(monkeypatch, tmp_path):
    stand = _Stand(monkeypatch, tmp_path, returncode=-6, banner=_OOM_REACHED)
    order: list = []
    threads = _run_real_sweep(monkeypatch, stand.manager, order)
    assert order == ["reap", "reconcile"], "a healthy sweep still does its custody work"
    assert [thread.name for thread in threads] == ["custody-maintenance"], "no retry thread"
    assert stand.spawned == [] and stand.ensures == [] and not _rows(stand.data_dir)
    assert stand.manager._last_start_failure is None


@pytest.mark.parametrize("released", [True, False])
def test_periodic_sweep_retries_only_after_it_released_a_latch(monkeypatch, released):
    """EXECUTED wiring: release first; the retry (on its thread) only if released; reap then reconcile."""
    from ouroboros import claudexor_daemon as daemon_mod
    from ouroboros import process_custody as pc
    from ouroboros import server_maintenance as sm
    from supervisor import queue

    order: list = []
    threads = _track_sweep_threads(monkeypatch)
    monkeypatch.setattr(sm, "_LAST_CANCEL_INTENT_SWEEP", [time.time()])  # 20 s cadence idle
    monkeypatch.setattr(sm, "_installed_skill_names", lambda: None)
    monkeypatch.setattr(pc, "reap_orphaned_processes", lambda root, **kw: order.append("reap") or [])
    monkeypatch.setattr(sm, "_retry_latched_daemon_start", lambda: order.append("retry"))
    monkeypatch.setattr(sm, "_reconcile_delegated_runs",
                        lambda live, **kwargs: order.append("reconcile"))
    monkeypatch.setattr(sm, "_cursor_refresh_settled_terminals", lambda *a, **kwargs: None)
    monkeypatch.setattr(queue, "RUNNING", {})
    stub = SimpleNamespace(
        clear_start_failure_latch=lambda *, cleared_by: order.append(f"clear:{cleared_by}") or released)
    monkeypatch.setattr(daemon_mod, "get_owned_daemon", lambda: stub)
    sm._periodic_supervisor_maintenance([0.0], [time.time()])
    for thread in threads:
        thread.join(5)
    assert order[0] == "clear:supervisor_sweep" and order.index("reap") < order.index("reconcile")
    # The block's own daemon thread always exists; the retry rides a second one.
    assert ("retry" in order) is released and len(threads) == (2 if released else 1)


def test_a_raising_reap_cannot_pin_the_latch(monkeypatch):
    """The release and retry live in their own try, ahead of the custody reap."""
    from ouroboros import claudexor_daemon as daemon_mod
    from ouroboros import process_custody as pc
    from ouroboros import server_maintenance as sm

    order: list = []
    threads = _track_sweep_threads(monkeypatch)
    monkeypatch.setattr(sm, "_LAST_CANCEL_INTENT_SWEEP", [time.time()])
    monkeypatch.setattr(sm, "_installed_skill_names", lambda: None)

    def raising_reap(root, **kw):
        order.append("reap")
        raise OSError("ledger unreadable")

    monkeypatch.setattr(pc, "reap_orphaned_processes", raising_reap)
    monkeypatch.setattr(sm, "_retry_latched_daemon_start", lambda: order.append("retry"))
    monkeypatch.setattr(sm, "_reconcile_delegated_runs",
                        lambda live, **kwargs: order.append("reconcile"))
    stub = SimpleNamespace(
        clear_start_failure_latch=lambda *, cleared_by: order.append(f"clear:{cleared_by}") or True)
    monkeypatch.setattr(daemon_mod, "get_owned_daemon", lambda: stub)
    sm._periodic_supervisor_maintenance([0.0], [time.time()])
    for thread in threads:
        thread.join(5)
    assert order[0] == "clear:supervisor_sweep" and "retry" in order and "reconcile" not in order
    assert "reap" in order, "released and retried regardless of the raising reap"


def test_the_sweep_retry_runs_on_its_own_thread_and_never_blocks_the_tick(monkeypatch):
    """H-04 item 4: runtime preparation inside the ensure is unbounded; the tick returns at once."""
    import threading

    from ouroboros import claudexor_daemon as daemon_mod
    from ouroboros import process_custody as pc
    from ouroboros import server_maintenance as sm
    from supervisor import queue

    entered, release = threading.Event(), threading.Event()

    def slow_ensure(**_kwargs):
        entered.set()
        assert release.wait(5), "the test must release the ensure"
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(daemon_mod, "ensure_owned_gateway", slow_ensure)
    threads = _track_sweep_threads(monkeypatch)
    monkeypatch.setattr(sm, "_LAST_CANCEL_INTENT_SWEEP", [time.time()])
    monkeypatch.setattr(sm, "_installed_skill_names", lambda: None)
    monkeypatch.setattr(pc, "reap_orphaned_processes", lambda root, **kw: [])
    monkeypatch.setattr(sm, "_reconcile_delegated_runs", lambda live, **kwargs: None)
    monkeypatch.setattr(sm, "_cursor_refresh_settled_terminals", lambda *a, **kwargs: None)
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(daemon_mod, "get_owned_daemon",
                        lambda: SimpleNamespace(clear_start_failure_latch=lambda *, cleared_by: True))
    try:
        sm._periodic_supervisor_maintenance([0.0], [time.time()])  # returns while the ensure is held
        assert entered.wait(5), "the retry thread reached the ensure"
        assert [thread.name for thread in threads] == ["custody-maintenance", "owned-daemon-latch-retry"]
        assert threads[1].daemon and threads[1].is_alive(), "exactly one retry thread, still in the ensure"
    finally:
        release.set()
        for thread in threads:
            thread.join(5)
    assert all(not thread.is_alive() for thread in threads)


def test_the_sweep_retry_swallows_refusals_and_surprises_and_closes_its_gateway(monkeypatch):
    from ouroboros import claudexor_daemon as daemon_mod
    from ouroboros import server_maintenance as sm
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    calls: list = []

    def refused(**kwargs):
        calls.append(kwargs)
        raise ClaudexorUnavailable("daemon_spawn_failed", "latched again")

    monkeypatch.setattr(daemon_mod, "ensure_owned_gateway", refused)
    assert sm._retry_latched_daemon_start() is None
    assert calls == [{"admission_wait_sec": 0, "startup_wait_sec": 0}], "zero admission AND zero startup wait"

    def surprise(**_kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(daemon_mod, "ensure_owned_gateway", surprise)
    assert sm._retry_latched_daemon_start() is None

    closed: list = []
    monkeypatch.setattr(daemon_mod, "ensure_owned_gateway",
                        lambda **_kwargs: SimpleNamespace(close=lambda: closed.append(1)))
    sm._retry_latched_daemon_start()
    assert closed == [1], "an opened gateway is closed at once"


def test_the_sweep_retry_never_waits_and_treats_daemon_starting_as_expected(monkeypatch, caplog):
    """H-03 item 4: the spawn happens under custody; the sweep returns at once, no warning."""
    import logging

    from ouroboros import claudexor_daemon as daemon_mod
    from ouroboros import server_maintenance as sm
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    seen: list = []

    def starting(**kwargs):
        seen.append(kwargs)
        raise ClaudexorUnavailable("daemon_starting", "still starting; retry joins the same startup")

    monkeypatch.setattr(daemon_mod, "ensure_owned_gateway", starting)
    with caplog.at_level(logging.INFO):
        assert sm._retry_latched_daemon_start() is None
    assert seen == [{"admission_wait_sec": 0, "startup_wait_sec": 0}]
    ours = [rec for rec in caplog.records if "retry after latch release" in rec.getMessage()]
    assert [rec.levelno for rec in ours] == [logging.INFO], "expected outcome, not a warning"
    assert "starting under custody" in ours[0].getMessage()
