"""Real disposable pooled children; never invoke the installation's Panic."""
from __future__ import annotations

import functools
import itertools
import json
import multiprocessing
import os
import pathlib
import signal
import socket
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import platform_layer as platform
from ouroboros.process_containment import pid_is_zombie
from supervisor import worker_process
from tests._shared import stop_socket_sharer, wait_test_child_stop



def _faulted_entry(mode, incoming, outgoing, repo_dir, drive_root, session_id, stop_socket, transport_no_proxy=None):
    from ouroboros import process_custody, utils

    platform.create_new_session()
    if mode == "startup_log":
        original = utils.append_jsonl

        def blocked_entry_log(path, value, *args, **kwargs):
            if value.get("type") == "worker_starting":
                outgoing.put("before_lifeline")
                threading.Event().wait()
            return original(path, value, *args, **kwargs)

        utils.append_jsonl = blocked_entry_log
        worker_process.worker_main(mode, incoming, outgoing, repo_dir, drive_root, session_id, stop_socket,
                                   transport_no_proxy)
        return

    def callback():
        if mode == "callback_raises":
            raise RuntimeError("failed optional cleanup")
        if mode == "callback_hangs":
            threading.Event().wait()

    if mode != "no_lifeline":
        process_custody.start_parent_lifeline(stop_socket=stop_socket, before_exit=callback, poll_sec=.05)
    outgoing.put("ready")
    threading.Event().wait()


def _owner_fault_entry(mode, *args):
    from ouroboros import claudexor_daemon
    from tests.test_batch1_exact_consumers import _pooled_test_entry

    def unavailable_owner(**_kwargs):
        if mode == "owner_raises":
            raise RuntimeError("owner unavailable")
        threading.Event().wait()

    claudexor_daemon.get_owned_daemon = unavailable_owner
    # Actual worker_main -> actual registry -> supported run_command Popen.
    _pooled_test_entry(mode, *args)


def _observed_owner_fault_entry(mode, *args, evidence):
    """Trace the actual request path without holding it for logging or a Queue.

    The socket is fixture-owned; a missing datagram is an evidence gap, not proof
    that the corresponding call never ran. All wrapped operations still execute.
    """
    from ouroboros import process_custody
    from ouroboros.tools import shell_process

    sequence = itertools.count(1)
    evidence.setblocking(False)

    def emit(event, **facts):
        try:
            evidence.send(json.dumps({"seq": next(sequence), "event": event,
                                      "t_ns": time.monotonic_ns(), **facts}).encode())
        except OSError:
            pass  # observer failure cannot veto an emergency request

    start_lifeline = process_custody.start_parent_lifeline
    request_child = shell_process.request_process_tree_kill
    kill_group = os.killpg

    def traced_lifeline(*, before_exit=None, **kwargs):
        def callback():
            emit("callback_enter")
            try:
                return before_exit()
            finally:
                emit("callback_exit")
        return start_lifeline(before_exit=callback if before_exit else None, **kwargs)

    def traced_child(proc, **kwargs):
        emit("child_request_enter", pid=proc.pid,
             registered=proc in shell_process._active_subprocesses)
        try:
            result = request_child(proc, **kwargs)
        except BaseException as exc:
            emit("child_request_error", error=type(exc).__name__)
            raise
        emit("child_request_return", result=result)
        return result

    def traced_kill_group(pgid, sig):
        emit("killpg_enter", pgid=pgid, signal=int(sig))
        try:
            result = kill_group(pgid, sig)
        except BaseException as exc:
            emit("killpg_error", pgid=pgid, error=type(exc).__name__, errno=getattr(exc, "errno", None))
            raise
        emit("killpg_return", pgid=pgid)
        return result

    process_custody.start_parent_lifeline = traced_lifeline
    shell_process.request_process_tree_kill = traced_child
    os.killpg = traced_kill_group
    emit("observer_ready", pid=os.getpid())
    _owner_fault_entry(mode, *args)


def _received_events(events):
    """Report received prefix, not completion: senders may exit before a datagram lands."""
    sequences = sorted(row["seq"] for row in events)
    contiguous = bool(sequences) and sequences == list(range(1, len(sequences) + 1))
    return {"events": events, "received_seq_prefix_contiguous": contiguous,
            "last_received_seq": sequences[-1] if sequences else 0, "events_after_last_received": "unknown"}


def _stop_trace(sock, child_pid):
    """Drain worker events after the frozen verdict, before fixture cleanup."""
    events = []
    while True:
        try:
            events.append(json.loads(sock.recv(16384)))
        except (BlockingIOError, ConnectionResetError):
            break
    child = {"pid": child_pid, "t_ns": time.monotonic_ns(), "alive": platform.pid_is_alive(child_pid),
             "zombie": pid_is_zombie(child_pid), "birth": platform.process_start_time(child_pid),
             "pgid": platform.process_group_id(child_pid)}
    if sys.platform.startswith("linux"):
        try:
            child["wchan"] = pathlib.Path(f"/proc/{child_pid}/wchan").read_text()
        except OSError as exc:
            child["wchan_error"] = type(exc).__name__
    return _received_events(events) | {"child_after_verdict": child}


def _assert_frozen_child_stop(verdict, diagnose, **context):
    """Use the wait's terminal observation, frozen before slower diagnostics."""
    stopped = verdict["child_stopped"]
    trace = dict(context)
    try:
        trace |= diagnose()
    except Exception as exc:
        trace["diagnostic_error"] = f"{type(exc).__name__}: {exc}"
    trace["verdict"] = verdict
    try:
        print("REGISTERED_STOP_TRACE " + json.dumps(trace, sort_keys=True, default=repr), flush=True)
    except Exception as exc:
        trace["print_error"] = f"{type(exc).__name__}: {exc}"
    assert stopped, trace


def _ordinary_close_entry(wid, incoming, outgoing, repo_dir, drive_root, session_id, stop_socket,
                          transport_no_proxy=None):
    import subprocess
    import sys

    from ouroboros import claudexor_daemon, process_custody

    platform.create_new_session()
    daemon = claudexor_daemon.OwnedClaudexorDaemon()
    daemon._proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                    **platform.subprocess_new_group_kwargs())
    process_custody.start_parent_lifeline(stop_socket=stop_socket,
                                        before_exit=lambda: daemon.panic_stop(request_only=True), poll_sec=.05)
    outgoing.put(daemon._proc.pid)
    threading.Event().wait()


@pytest.mark.parametrize("mode", ["startup_log", "no_lifeline", "callback_hangs"])
def test_native_worker_stop_does_not_require_lifeline_or_callback(tmp_path, monkeypatch, mode):
    if mode == "suspended" and os.name == "nt":
        pytest.skip("SIGSTOP is a POSIX failure injection")
    monkeypatch.setattr(worker_process, "worker_main", _faulted_entry)
    ctx = multiprocessing.get_context("spawn")
    incoming, outgoing = ctx.Queue(), ctx.Queue()
    proc = worker_process.spawn_worker_process(ctx, mode, incoming, outgoing, tmp_path, tmp_path)
    try:
        assert outgoing.get(timeout=20) == ("before_lifeline" if mode == "startup_log" else "ready")
        if mode == "suspended":
            os.kill(proc.pid, signal.SIGSTOP)  # our still-custodied native child
        start = time.monotonic()
        receipt = platform.request_process_tree_kill(proc)
        assert time.monotonic() - start < .5
        assert receipt["requested"] and receipt["root_backstop"] == "armed_native_owner"
        assert receipt["confirmation"] == "unconfirmed"
        proc.join(timeout=3)
        assert not proc.is_alive(), receipt
        assert time.monotonic() - start < 3
        if mode in {"startup_log", "no_lifeline", "suspended"}:
            assert receipt["native_request"]["requested"]
            if os.name != "nt":
                assert proc.exitcode == -signal.SIGKILL
    finally:
        try:
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)
            worker_process.close_worker_stop_channel(proc)
            assert proc._ouroboros_stop_socket.fileno() == -1
        finally:  # a failed cleanup assertion must not leak the queues or the socket sharer
            for queue in (incoming, outgoing):
                queue.close()
                queue.cancel_join_thread()
            stop_socket_sharer()


@pytest.mark.parametrize("mode", ["owner_raises", "owner_hangs"])
def test_supported_command_is_stopped_even_when_another_owner_fails(tmp_path, monkeypatch, mode):
    from tests.test_batch1_exact_consumers import _await_published_command_child

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    trace_parent = trace_child = None
    target = _owner_fault_entry
    if os.name != "nt":
        trace_parent, trace_child = socket.socketpair(type=socket.SOCK_DGRAM)
        trace_parent.setblocking(False)
        target = functools.partial(_observed_owner_fault_entry, evidence=trace_child)
    monkeypatch.setattr(worker_process, "worker_main", target)
    ctx = multiprocessing.get_context("spawn")
    incoming, outgoing = ctx.Queue(), ctx.Queue()
    try:
        proc = worker_process.spawn_worker_process(ctx, mode, incoming, outgoing, tmp_path, tmp_path)
    except BaseException:
        if trace_parent is not None:
            trace_parent.close()
            trace_child.close()
        for queue in (incoming, outgoing):
            queue.close()
            queue.cancel_join_thread()
        raise
    if trace_child is not None:
        trace_child.close()
    parent_events = []
    if trace_parent is not None:
        native_kill = proc._popen.kill

        def observed_native_kill():
            parent_events.append({"event": "parent_native_kill", "t_ns": time.monotonic_ns()})
            return native_kill()

        monkeypatch.setattr(proc._popen, "kill", observed_native_kill)
    seen = {}
    try:
        incoming.put({"id": "pooled", "type": "task"})
        # The published-child guarantee: the child's own marker can precede its registry entry.
        child_pid = _await_published_command_child(proc, outgoing, tmp_path / "workspace", seen)
        if os.name != "nt":
            assert os.getpgid(child_pid) == child_pid != os.getpgid(proc.pid)
        parent_events.append({"event": "parent_request", "t_ns": time.monotonic_ns()})
        receipt = platform.request_process_tree_kill(proc)
        proc.join(timeout=3)
        assert not proc.is_alive(), receipt
        verdict = wait_test_child_stop(child_pid)
        _assert_frozen_child_stop(
            verdict, lambda: (_stop_trace(trace_parent, child_pid) | {"parent_events": parent_events}
                                if trace_parent is not None else {"platform": "windows"}),
            mode=mode, receipt=receipt, worker_exit=proc.exitcode, published=seen)
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)
        child_pid = seen.get("pid", 0)  # the marker's own pid, even when its ACK never came
        if child_pid and platform.pid_is_alive(child_pid) and not pid_is_zombie(child_pid):
            platform.force_kill_pid(child_pid)  # only the continuously observed fixture child
        worker_process.close_worker_stop_channel(proc)
        if trace_parent is not None:
            trace_parent.close()
        for queue in (incoming, outgoing):
            queue.close()
            queue.cancel_join_thread()
        stop_socket_sharer()


def test_child_death_during_diagnostics_cannot_pass_the_frozen_verdict(capsys):
    import subprocess

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    late = {}

    def broken_observer():
        raise RuntimeError("observer defect")

    def child_dies_while_tracing():
        child.kill()
        child.wait(timeout=10)
        late["stopped"] = not platform.pid_is_alive(child.pid) or pid_is_zombie(child.pid)
        return {}

    try:
        with pytest.raises(AssertionError, match="RuntimeError: observer defect"):
            _assert_frozen_child_stop(wait_test_child_stop(child.pid, timeout_sec=0), broken_observer, mode="observer_defect")
        with pytest.raises(AssertionError, match="'child_stopped': False"):
            _assert_frozen_child_stop(wait_test_child_stop(child.pid, timeout_sec=0), child_dies_while_tracing, mode="late_death")
        assert late == {"stopped": True}
        _assert_frozen_child_stop(wait_test_child_stop(child.pid, timeout_sec=0), broken_observer, mode="stopped_before_trace")
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)
    printed = [json.loads(line.partition(" ")[2]) for line in capsys.readouterr().out.splitlines()
               if line.startswith("REGISTERED_STOP_TRACE ")]
    assert [(row["mode"], row["verdict"]["child_stopped"], "diagnostic_error" in row) for row in printed] == [
        ("observer_defect", False, True), ("late_death", False, False), ("stopped_before_trace", True, True)]


@pytest.mark.parametrize("observed_zombie", [False, True])
def test_wait_keeps_terminal_observation_across_reaping(monkeypatch, observed_zombie):
    from ouroboros import process_containment

    probes = []

    def zombie(_pid):
        probes.append("zombie")
        return observed_zombie

    def alive(_pid):
        probes.append("alive")
        return False  # already reaped, including between the two status probes

    monkeypatch.setattr(process_containment, "pid_is_zombie", zombie)
    monkeypatch.setattr(platform, "pid_is_alive", alive)
    verdict = wait_test_child_stop(123, timeout_sec=0)
    assert verdict["child_stopped"] is True
    assert probes == (["zombie"] if observed_zombie else ["zombie", "alive"])
    # Subsequent independent probes may cross zombie -> reaped. No re-read may
    # turn the wait's positive terminal witness into the old false failure.
    monkeypatch.setattr(platform, "pid_is_alive", lambda _pid: pytest.fail("terminal witness was discarded"))
    monkeypatch.setattr(process_containment, "pid_is_zombie", lambda _pid: pytest.fail("terminal witness was discarded"))
    _assert_frozen_child_stop(verdict, lambda: {"after_verdict": "reaped"})


def test_wait_rejects_live_child_at_original_bound(monkeypatch):
    from ouroboros import process_containment

    clock, pauses = iter((10.0, 10.0, 13.0)), []
    # Windows closes the call-phase proactor before fixture teardown; it needs
    # the real clock, so the finite fake must end within the test body.
    with monkeypatch.context() as probe:
        probe.setattr(time, "monotonic", lambda: next(clock))
        probe.setattr(time, "sleep", pauses.append)
        probe.setattr(platform, "pid_is_alive", lambda _pid: True)
        probe.setattr(process_containment, "pid_is_zombie", lambda _pid: False)
        verdict = wait_test_child_stop(123)
    assert verdict["child_stopped"] is False and pauses == [.01]
    with pytest.raises(AssertionError, match="'child_stopped': False"):
        _assert_frozen_child_stop(verdict, lambda: {"after_verdict": "stopped too late"})


def test_trace_discloses_received_prefix_never_absent_execution():
    racing = [{"seq": 2, "event": "callback_enter"}, {"seq": 1, "event": "observer_ready"}]
    assert _received_events(racing) == {"events": racing, "received_seq_prefix_contiguous": True,
                                        "last_received_seq": 2, "events_after_last_received": "unknown"}
    gapped = _received_events([{"seq": 1, "event": "observer_ready"}, {"seq": 3, "event": "killpg_enter"}])
    assert not gapped["received_seq_prefix_contiguous"] and gapped["events_after_last_received"] == "unknown"
    assert not _received_events([])["received_seq_prefix_contiguous"]


@pytest.mark.parametrize("broken_channel", [False, True])
def test_server_completes_native_requests_before_settlement_or_exit(tmp_path, monkeypatch, broken_channel):
    from tests.test_server_control_panic_daemon import _run_panic

    parent, child = socket.socketpair()
    parent.setblocking(False)
    native_done = threading.Event()
    proc = SimpleNamespace(pid=123, exitcode=None, _ouroboros_stop_socket=parent,
                           _popen=SimpleNamespace(kill=native_done.set))
    proc._ouroboros_stop_request = lambda: worker_process.request_worker_stop(proc)
    diagnostics = []

    def settle():
        assert native_done.is_set(), "settlement could destroy the parent before its native request"
        return True

    try:
        if broken_channel:
            # The SENDING end: its send fails on every platform. A closed peer is not enough on
            # Windows, where socketpair is loopback TCP and the next send may still be buffered.
            parent.close()
        _run_panic(monkeypatch, tmp_path, daemon_stop=settle, children=(proc,), diagnostics=diagnostics)
        assert native_done.is_set()
        receipt = diagnostics[0]["requests"]["child-123"]
        assert receipt["requested"] is not broken_channel
        assert receipt["native_request"]["requested"]
        assert receipt["confirmation"] == "unconfirmed"  # an adapter signal is NOT observed death
    finally:
        proc.exitcode = 1
        worker_process.close_worker_stop_channel(proc)
        child.close()


def test_retirement_closes_channel_but_keeps_owed_native_owner(tmp_path, monkeypatch):
    """An ordinary cleanup failure cannot discard an already armed native request."""
    parent, child = socket.socketpair()
    parent.setblocking(False)
    native_done = threading.Event()
    proc = SimpleNamespace(pid=123, exitcode=None, _ouroboros_stop_socket=parent,
                           _popen=SimpleNamespace(kill=native_done.set))
    try:
        worker_process.request_worker_stop(proc)
        worker_process.close_worker_stop_channel(proc)
        assert parent.fileno() == -1
        assert native_done.wait(2)
    finally:
        proc.exitcode = 1
        worker_process.close_worker_stop_channel(proc)
        child.close()


def test_hung_server_owner_callback_cannot_skip_native_worker_request(tmp_path, monkeypatch):
    from tests.test_server_control_panic_daemon import _run_panic

    entered, release, native_done = threading.Event(), threading.Event(), threading.Event()
    parent, child = socket.socketpair()
    parent.setblocking(False)
    proc = SimpleNamespace(pid=123, exitcode=None, _ouroboros_stop_socket=parent,
                           _popen=SimpleNamespace(kill=native_done.set))
    proc._ouroboros_stop_request = lambda: worker_process.request_worker_stop(proc)

    def hang(**_kwargs):
        entered.set()
        release.wait(3)  # fixture backstop; assert production proceeds well before it
        return []

    def settle():
        assert entered.is_set() and native_done.is_set()
        release.set()
        return True

    try:
        started = time.monotonic()
        _run_panic(monkeypatch, tmp_path, panic_request=hang, daemon_stop=settle, children=(proc,))
        assert time.monotonic() - started < 2
        assert release.is_set() and native_done.is_set()
    finally:
        release.set()
        proc.exitcode = 1
        worker_process.close_worker_stop_channel(proc)
        child.close()


def test_ordinary_stop_channel_retirement_does_not_panic_owned_daemon(tmp_path, monkeypatch):
    from supervisor import worker_pool_lifecycle, workers

    monkeypatch.setattr(worker_process, "worker_main", _ordinary_close_entry)
    monkeypatch.setattr(worker_pool_lifecycle, "_record_worker_processes", lambda: None)
    ctx = multiprocessing.get_context("spawn")
    incoming, outgoing = ctx.Queue(), ctx.Queue()
    proc = worker_process.spawn_worker_process(ctx, 0, incoming, outgoing, tmp_path, tmp_path)
    slot = SimpleNamespace(proc=proc, in_q=incoming)
    monkeypatch.setattr(workers, "WORKERS", {0: slot})
    daemon_pid, daemon_birth = 0, ""
    try:
        daemon_pid = outgoing.get(timeout=20)
        daemon_birth = platform.process_start_time(daemon_pid)  # pinned while its owner still runs
        assert daemon_birth
        worker_process.close_worker_stop_channel(proc)  # EOF is not the explicit Panic byte
        proc.join(timeout=3)
        assert not proc.is_alive()
        assert platform.pid_is_alive(daemon_pid) and not pid_is_zombie(daemon_pid)
        assert worker_pool_lifecycle.retire_worker(0, slot)
        assert proc._ouroboros_stop_socket.fileno() == -1 and not workers.WORKERS
    finally:
        try:
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)
            if (daemon_birth and platform.pid_is_alive(daemon_pid) and not pid_is_zombie(daemon_pid)
                    and platform.process_start_time(daemon_pid) == daemon_birth):
                platform.force_kill_pid(daemon_pid)  # the fixture's own daemon: its PID AND its birth
                deadline = time.monotonic() + 3
                while (platform.pid_is_alive(daemon_pid) and not pid_is_zombie(daemon_pid)
                       and time.monotonic() < deadline):
                    time.sleep(.01)
                assert not platform.pid_is_alive(daemon_pid) or pid_is_zombie(daemon_pid)
        finally:  # a failed death proof must not leak the stop channel, queues or socket sharer
            worker_process.close_worker_stop_channel(proc)
            for queue in (incoming, outgoing):
                queue.close()
                queue.cancel_join_thread()
            stop_socket_sharer()
