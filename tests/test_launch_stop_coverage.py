"""Judged is not persisted: nothing of a launch is signalled until what could outlive that signal is durable.

A stop's input (what an earlier pass of this process retained in memory) is what it may discharge, not proof that any
store names it. So every retained member and every member only a parent chain reached needs a successful write before
any signal that could break the chain, on the first attempt and on each retry, native fallback included. Real
consumers over a real runner L and its server C in another session (``_launch_stop_shared``); faults are injected at the
write the stop publishes through. Immediate Panic stays what it was: no table, no custody read, one native kill.
"""

from __future__ import annotations

import os
import threading
import time

import pytest

from tests._launch_stop_shared import (
    _PATIENCE,
    _answers,
    _count_census,
    _deny_signals_to,
    _end,
    _executor_ctx,
    _owed,
    _payload,
    _registry,
    _runner_cmd,
    _same_process_alive,
    _server,
    _settle,
    _start,
    write_scripts,
)
from tests.test_launch_stop_contract import BACKENDS, HIDDEN, _block_registration, _leader_gone, _roots_unknown

pytestmark = [
              pytest.mark.skipif(os.name == "nt", reason="the measured stop is POSIX; Windows keeps its native kill")]


@pytest.fixture
def scripts(tmp_path):
    return write_scripts(tmp_path)


def _writes_fail(monkeypatch, data) -> dict:
    """Every durable write a stop publishes through fails at its own call (the ledger append, the ownership document
    and pending channel) until ``state["failing"]`` clears; a stop can then name nothing."""
    from ouroboros import owned_shutdown, process_custody

    state, ledger = {"failing": True}, process_custody.ledger_path(data)
    real_append, real_write = process_custody.append_jsonl, owned_shutdown.atomic_write_json

    def append(path, *args, **kwargs):
        return False if state["failing"] and str(path) == str(ledger) else real_append(path, *args, **kwargs)

    def write(path, *args, **kwargs):
        if state["failing"]:
            raise OSError("no space left on device")
        return real_write(path, *args, **kwargs)

    monkeypatch.setattr(process_custody, "append_jsonl", append)
    monkeypatch.setattr(owned_shutdown, "atomic_write_json", write)
    return state


def _proc(record):
    return record.proc if hasattr(record, "proc") else record.local_proc


@pytest.mark.parametrize("survivor", ["held by unknown roots", "signal denied"])
@pytest.mark.parametrize("backend", list(BACKENDS))
def test_a_retry_cannot_break_a_chain_no_publication_has_named(tmp_path, monkeypatch, scripts, backend, survivor):
    """The first stop observes L -> C and every write fails, so L and C stay alive and only this process's memory names
    C. A retry starts with C in its input, measured but not persisted, while the writes still fail: it must not signal
    L either, or C (held, or refusing its signal) outlives its only chain with nothing durable naming it. Once the
    writes come back the retry names C first, then ends both, and a confirmed stop retires everything."""
    from ouroboros.tools import services

    data = tmp_path / "data"
    ctx, record, registry, log, leader, (port, pid, birth) = BACKENDS[backend](tmp_path, monkeypatch, scripts)
    proc = _proc(record)
    roots = _roots_unknown(monkeypatch)
    if survivor == "signal denied":
        roots["unknown"] = False
        undeny = _deny_signals_to(monkeypatch, {pid})
    writes = _writes_fail(monkeypatch, data)
    tables = _count_census(monkeypatch)
    try:
        for done, attempt in enumerate(("first", "retry", "another retry"), start=1):
            stopped = _payload(services._stop_service(ctx, name="a"))
            assert stopped["stop_failed"] is True, (attempt, stopped)
            assert len(tables) == done, "the coverage of what it retained costs no table of its own"
            assert proc.poll() is None, f"{attempt}: L was signalled while nothing durable named C"
            assert _answers(port) and _same_process_alive(pid, birth), attempt
            assert stopped["process_cleanup"]["custody"] == "unavailable", (attempt, stopped)
            assert _owed(data, leader) == ([], []), "no write landed: only memory names C"
            assert record.service_id in registry and log.exists()
        writes["failing"] = False
        roots["unknown"] = False
        if survivor == "signal denied":
            undeny()

        finished = _payload(services._stop_service(ctx, name="a"))

        assert finished["process_cleanup"]["exit_confirmed"] is True and "stop_failed" not in finished, finished
        assert not _answers(port) and not _same_process_alive(pid, birth) and proc.poll() is not None
        assert record.service_id not in registry and not log.exists()
    finally:
        writes["failing"] = False
        _settle(proc, identities=[(pid, birth)])


def test_what_a_failed_retry_kept_stays_recoverable_after_its_owner_is_gone(tmp_path, monkeypatch, scripts):
    """The owner exits right after two failed stops. Its memory goes, but the chain it refused to break is still there,
    so the next generation's sweep, reading the launch's ledger row, finds C through L and ends both."""
    from ouroboros import process_custody
    from ouroboros.tools import services

    data = tmp_path / "data"
    ctx, record, registry, _log, leader, (port, pid, birth) = BACKENDS["host service"](tmp_path, monkeypatch, scripts)
    roots = _roots_unknown(monkeypatch)
    writes = _writes_fail(monkeypatch, data)
    try:
        for _attempt in range(2):
            assert _payload(services._stop_service(ctx, name="a"))["stop_failed"] is True
        assert record.proc.poll() is None
        writes["failing"] = False
        roots["unknown"] = False
        registry.clear()  # that process is gone, and every memory of it
        monkeypatch.setattr(process_custody, "_SESSION_ID", "next-generation")

        process_custody.reap_orphaned_processes(data, running_task_ids=set())

        assert not _answers(port) and not _same_process_alive(pid, birth)
    finally:
        writes["failing"] = False
        _settle(record.proc, identities=[(pid, birth)])


@pytest.mark.parametrize("backend", list(BACKENDS))
def test_repeated_stops_before_a_delayed_registration_appends_leave_the_chain_alone(
        tmp_path, monkeypatch, scripts, backend):
    """No ledger row exists yet, so nothing can name what a stop observes. Every stop before the append withholds, not
    only the first: the second one's input already names C, and that does not make it durable. Once registration lands
    the next stop writes C ahead and ends both."""
    from ouroboros import workspace_executor as executor
    from ouroboros.tools import services

    data = tmp_path / "data"
    port_file, started = tmp_path / "a.port", {}
    cmd = _runner_cmd(scripts, port_file, "separate", *HIDDEN)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    if backend == "host service":
        ctx, memory = _registry(tmp_path, monkeypatch)._ctx, services._SERVICES
        kwargs = {}
    else:
        ctx, workspace = _executor_ctx(tmp_path)
        memory, kwargs = executor._SERVICES, {"cwd": str(workspace)}
    roots = _roots_unknown(monkeypatch)
    reached, release = _block_registration(monkeypatch, "before the ledger append", data)
    starter = threading.Thread(target=lambda: started.update(text=services._start_service(
        ctx, name="a", cmd=cmd, readiness={}, **kwargs)), name="starter")
    starter.start()
    assert reached.wait(_PATIENCE), "the start never reached registration"
    port, pid, birth = _server(port_file)
    [record] = memory.values()
    proc = _proc(record)
    try:
        for attempt in ("first", "second"):
            stopped = _payload(services._stop_service(ctx, name="a"))
            assert stopped["stop_failed"] is True, (attempt, stopped)
            assert proc.poll() is None, f"{attempt} stop before the append signalled L"
            assert _answers(port) and _same_process_alive(pid, birth)
        release.set()
        starter.join(_PATIENCE)
        assert not starter.is_alive() and hasattr(proc, "_ouroboros_custody")
        roots["unknown"] = False

        finished = _payload(services._stop_service(ctx, name="a"))

        assert finished["process_cleanup"]["exit_confirmed"] is True, finished
        assert not _answers(port) and not _same_process_alive(pid, birth) and proc.poll() is not None
    finally:
        release.set()
        starter.join(_PATIENCE)
        _settle(proc, identities=[(pid, birth)])


def _unreadable_first_read_of(monkeypatch, pid: int) -> None:
    """The waited stop's first read of the leader's birth fails: it enters the pass with no measured leader."""
    from ouroboros import process_census, process_scope

    real, spent = process_scope.read_identities, []

    def read(pids, **kwargs):
        pids = list(pids)
        if int(pid) in pids and not spent:
            spent.append(1)
            return {**real([item for item in pids if item != pid], **kwargs),
                    int(pid): process_census._Identity(process_census.IDENTITY_UNREADABLE)}
        return real(pids, **kwargs)

    monkeypatch.setattr(process_scope, "read_identities", read)


def _native_kills(monkeypatch) -> list:
    from ouroboros import platform_layer

    real, calls = platform_layer.request_process_tree_kill, []
    monkeypatch.setattr(platform_layer, "request_process_tree_kill", lambda proc: calls.append(proc.pid) or real(proc))
    return calls


@pytest.mark.parametrize("backend", list(BACKENDS))
@pytest.mark.parametrize("case", ["unmeasured leader, chain observed", "failed table, member retained"])
def test_a_waited_stops_native_fallback_does_not_override_withholding(tmp_path, monkeypatch, scripts, backend, case):
    """A shutdown keeps its native group kill for what it could not measure. That kill breaks a chain as well as any
    signal, so when the pass could not make what it knows durable it stays out: L and C stay alive for the next start.
    First case: the leader's birth read failed, the table shows its marker and C below it. Second: an earlier failed
    stop left C in memory, and now the table itself fails. With the writes back the same fallback runs (positive)."""
    from ouroboros import process_scope
    from ouroboros.tools import services

    data = tmp_path / "data"
    ctx, record, _registry_map, _log, leader, (port, pid, birth) = BACKENDS[backend](tmp_path, monkeypatch, scripts)
    proc = _proc(record)
    _roots_unknown(monkeypatch)
    writes = _writes_fail(monkeypatch, data)
    kills = _native_kills(monkeypatch)
    real_census = process_scope.census
    try:
        if case == "unmeasured leader, chain observed":
            record.leader_birth = ""
            _unreadable_first_read_of(monkeypatch, leader)
        else:
            assert _payload(services._stop_service(ctx, name="a"))["stop_failed"] is True
            assert proc.poll() is None
            monkeypatch.setattr(process_scope, "census", lambda *a, **k: None)

        [payload] = services.kill_all_services(data, wait=True, durable=False)

        assert payload["stop_failed"] is True, payload
        assert kills == [], "the native kill overrode the withholding"
        assert proc.poll() is None and _answers(port) and _same_process_alive(pid, birth)
        assert _owed(data, leader) == ([], [])

        writes["failing"] = False
        if case == "unmeasured leader, chain observed":
            record.leader_birth = ""
            _unreadable_first_read_of(monkeypatch, leader)
        [retried] = services.kill_all_services(data, wait=True, durable=False)

        assert retried.get("stop_failed") is True, "nothing measured or a failed table confirms nothing"
        _leader_gone(leader)
        assert kills == [leader], "named now, so the fallback runs"
        assert pid in _owed(data, leader)[0], "the native kill came after C was durable"
    finally:
        writes["failing"] = False
        monkeypatch.setattr(process_scope, "census", real_census)
        _settle(proc, identities=[(pid, birth)])


def test_emergency_stop_stays_immediate_after_a_withheld_stop(tmp_path, monkeypatch, scripts):
    """The owner's immediate Panic is another path: whatever a waited stop withheld, one native kill now, no table,
    no custody read, no publication first."""
    from ouroboros import owned_shutdown, process_custody, process_scope
    from ouroboros.tools import services

    data = tmp_path / "data"
    ctx, record, _registry_map, _log, leader, (port, pid, birth) = BACKENDS["host service"](tmp_path, monkeypatch, scripts)
    _roots_unknown(monkeypatch)
    writes = _writes_fail(monkeypatch, data)
    monkeypatch.setattr(services, "_panic_requested", False)
    monkeypatch.setattr("ouroboros.workspace_executor._panic_requested", False)
    try:
        assert _payload(services._stop_service(ctx, name="a"))["stop_failed"] is True
        assert record.proc.poll() is None and record.known_members
        monkeypatch.setattr(process_scope, "census", lambda *a, **k: pytest.fail("Panic reads no process table"))
        monkeypatch.setattr(process_custody, "_read_ledger_records", lambda *a, **k: pytest.fail("Panic reads no custody"))
        monkeypatch.setattr(owned_shutdown, "_read_document", lambda *a, **k: pytest.fail("Panic reads no custody"))
        monkeypatch.setattr(process_custody, "write_ahead", lambda *a, **k: pytest.fail("Panic publishes nothing first"))

        [requested] = services.kill_all_services(data, request_only=True)

        assert requested["service_id"] == record.service_id and requested.get("requested") is True, requested
        assert record.proc.wait(timeout=_PATIENCE) is not None
    finally:
        monkeypatch.undo()
        writes["failing"] = False
        _settle(record.proc, identities=[(pid, birth)])


def test_a_retry_with_nothing_new_to_write_appends_nothing(tmp_path, monkeypatch):
    """The coverage check of a healthy retry is a read: what the launch's row already names costs no new row, however
    the retained list is ordered, and nothing waits on a write that is not needed."""
    from ouroboros import process_custody
    from tests._launch_stop_shared import _CARRIER, _CHILD, _OLD_BIRTH, _controlled, _launch_row

    world = _controlled(tmp_path, monkeypatch, {})
    row = _launch_row("t-a", purpose="service:a")
    world.spawned(row)
    two = [{"pid": _CHILD, "birth": _OLD_BIRTH}, {"pid": _CARRIER, "birth": _OLD_BIRTH, "held": True}]
    assert process_custody.write_ahead((tmp_path, row), two) is True
    ledger = process_custody.ledger_path(tmp_path)
    named = ledger.read_bytes()

    assert process_custody.write_ahead((tmp_path, row), list(reversed(two))) is True
    assert process_custody.write_ahead((tmp_path, row), two[:1]) is True

    assert ledger.read_bytes() == named, "the row already names them: nothing to append"


def test_a_contended_retry_the_ledger_already_covers_writes_no_copy(tmp_path, monkeypatch):
    """The ledger's lock is held elsewhere, and a read of the ledger proves it names all of it: durable, with no pending
    copy to fold later and nothing that could fail. An identity it does not name still lands in a pending copy."""
    from ouroboros import owned_shutdown, process_custody
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
    from ouroboros.utils import jsonl_append_lock_path
    from tests._launch_stop_shared import _CARRIER, _CHILD, _OLD_BIRTH, _controlled, _launch_row

    world = _controlled(tmp_path, monkeypatch, {})
    row = _launch_row("t-a", purpose="service:a")
    world.spawned(row)
    child = [{"pid": _CHILD, "birth": _OLD_BIRTH}]
    assert process_custody.write_ahead((tmp_path, row), child) is True
    pending = tmp_path / "state" / owned_shutdown.PENDING_DIRNAME
    lock_path = jsonl_append_lock_path(process_custody.ledger_path(tmp_path))
    holder = acquire_exclusive_file_lock(lock_path, timeout_sec=5.0)
    assert holder is not None
    try:
        assert process_custody.write_ahead((tmp_path, row), child, 0.0) is True
        assert not list(pending.glob("*.json")), "already named: nothing to add"

        assert process_custody.write_ahead((tmp_path, row), [{"pid": _CARRIER, "birth": _OLD_BIRTH}], 0.0) is True
        assert len(list(pending.glob("*.json"))) == 1
    finally:
        release_exclusive_file_lock(lock_path, holder)


def test_a_retained_leader_alone_needs_no_write(tmp_path, monkeypatch, scripts):
    """Its row names it, and nothing hangs from a chain here (everything is in its group): a stop whose writes fail
    still ends a leader an earlier stop retained, and owes nothing afterwards."""
    from ouroboros.tools import services

    data = tmp_path / "data"
    registry = _registry(tmp_path, monkeypatch)
    leader = _start(registry, scripts, tmp_path / "a.port", "a", mode="same")["pid"]
    port, pid, birth = _server(tmp_path / "a.port")
    record = services._SERVICES[services._service_key(registry._ctx, "a")]
    undeny = _deny_signals_to(monkeypatch, {leader})
    writes = _writes_fail(monkeypatch, data)
    try:
        first = _payload(services._stop_service(registry._ctx, name="a"))
        assert first["stop_failed"] is True and record.proc.poll() is None, first
        assert leader in [item["pid"] for item in record.known_members], record.known_members
        undeny()

        retried = _payload(services._stop_service(registry._ctx, name="a"))

        assert retried["process_cleanup"]["exit_confirmed"] is True, retried
        assert record.proc.poll() is not None and not _answers(port) and not _same_process_alive(pid, birth)
    finally:
        writes["failing"] = False
        _settle(record.proc, identities=[(pid, birth)])


def test_a_row_the_ledger_no_longer_holds_is_no_durable_coverage(tmp_path, monkeypatch):
    """A stop keeps the row it read; another stop's compaction then removed it from both stores. Publishing the same
    identities that stale row names is a real write, not an unchanged no-op that claims coverage nothing holds."""
    from ouroboros import process_custody
    from tests._launch_stop_shared import _CHILD, _OLD_BIRTH, _PID, _controlled, _launch_row

    world = _controlled(tmp_path, monkeypatch, {})
    child = [{"pid": _CHILD, "birth": _OLD_BIRTH}]
    stale = {**_launch_row("t-a", purpose="service:a"), "known_members": child}
    world.spawned(stale)
    process_custody._rewrite_ledger(tmp_path, [], previous=process_custody.ledger_path(tmp_path).read_bytes())
    assert _owed(tmp_path, _PID) == ([], []), "that compaction retired both copies"

    assert process_custody.write_ahead((tmp_path, stale), child) is True

    names = _owed(tmp_path, _PID)
    assert _CHILD in names[0] and _CHILD in names[1], "the write landed; coverage is real"
    assert world.signals == []


def test_a_retained_member_provably_gone_needs_no_write_and_holds_nothing_back(tmp_path, monkeypatch, scripts):
    """Coverage is owed for what could outlive the signal. A retained identity the table proves gone cannot, so failing
    writes do not hold its launch back: L ends, with nothing left to name."""
    from ouroboros.tools import services

    data = tmp_path / "data"
    ctx, record, registry, _log, leader, _server_identity = BACKENDS["host service"](tmp_path, monkeypatch, scripts, name="a")
    port, pid, birth = _server_identity
    try:
        _end(pid, birth)
        deadline = time.monotonic() + _PATIENCE
        while _same_process_alive(pid, birth):
            assert time.monotonic() < deadline
            time.sleep(0.02)
        record.known_members = [{"pid": pid, "birth": birth, "held": True}]
        writes = _writes_fail(monkeypatch, data)

        stopped = _payload(services._stop_service(ctx, name="a"))

        writes["failing"] = False
        assert record.proc.poll() is not None, stopped
        assert stopped["process_cleanup"]["exit_confirmed"] is True and record.service_id not in registry, stopped
    finally:
        _settle(record.proc, identities=[(pid, birth)])
