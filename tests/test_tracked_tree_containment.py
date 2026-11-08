"""A tracked child's whole tree, not its root alone: what Panic and Stop now end and what a stop may call settled.

Every tracked child is contained by measured identity. POSIX records, at publication, the group its new session
leads (``shell_process._record_group``). Windows bears it into its own Job (``shell_process.contained_popen``):
created suspended, assigned to a Job that refuses breakaway, then resumed. MSYS2 ``spawn.cc``
(``child_info_spawn::worker``) adds CREATE_BREAKAWAY_FROM_JOB whenever the immediate Job permits breakaway, and
its exec ends the Windows process that started the next program, so a parent-chain walk (``taskkill /T``) cannot
reach it. A stop then calls a tree settled only on a measurement of what that containment holds, never on the
root's exit alone; a child born outside a Job stays unmeasured.

The POSIX cases run real ``sh`` trees and emulate Windows' former root-only request by killing the root PID
alone; every process they start is ended by its own group here. The Windows cases are fakes of the native calls:
they prove the branch logic only, never Win32, Job or MSYS behaviour, which native Windows CI must confirm.
"""
from __future__ import annotations

import ctypes
import gc
import os
import signal
import subprocess
import sys
import time
import types

import pytest

from ouroboros import platform_layer
from ouroboros.platform_layer import subprocess_new_group_kwargs
from ouroboros.tools import shell_process
from tests._shared import wait_test_child_stop

POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="a real sh tree and process groups; Windows has fakes below")
# A retained POSIX leader handle (``_ouroboros_group_held``) exists only where contained_popen builds _RetainedPopen;
# its group calls need os.killpg and os.WNOHANG, which Windows' os lacks. Windows children are Jobs: the cases below.
POSIX_HANDLE = pytest.mark.skipif(os.name == "nt", reason="a retained POSIX leader handle; Windows tracked children are Jobs")


def fake_tree_read(monkeypatch, proc, *, remains):
    """A unit double's explicit containment measurement, independent of poll()."""
    read = shell_process._tree_remains
    monkeypatch.setattr(shell_process, "_tree_remains", lambda child: remains if child is proc else read(child))


@POSIX_ONLY
@pytest.mark.parametrize("request_only", [False, True])
def test_normal_root_exit_keeps_redirected_child_and_waitable_identity(registry, request_only):
    result = shell_process._tracked_subprocess_run(
        ["sh", "-c", "sleep 60 >/dev/null 2>&1 & echo $!; exit 7"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
    assert result.returncode == 7 and result.stderr == ""
    child = int(result.stdout)
    [proc] = shell_process._active_subprocesses
    registry.append((proc, child))
    assert proc.returncode == 7 and shell_process._tree_remains(proc) is True
    # The root result has returned, but waitid can still observe its own status.
    assert platform_layer.observe_child_exit(proc.pid)[0] == proc.pid
    assert not wait_test_child_stop(child, timeout_sec=0.1)["child_stopped"]
    if request_only:
        [receipt] = shell_process.kill_all_tracked_subprocesses(request_only=True)
        assert receipt == {"pid": proc.pid, "requested": True, "scope": "group"}
    else:
        shell_process.kill_all_tracked_subprocesses(deadline=time.monotonic() + 5)
    assert wait_test_child_stop(child, timeout_sec=5)["child_stopped"]
    assert shell_process.retire_tracked(proc)
    with pytest.raises(ChildProcessError):
        platform_layer.observe_child_exit(proc.pid)


@POSIX_ONLY
def test_fast_roots_keep_exact_results_and_reap_zombie_only_groups(registry):
    for code in (0, 23):
        with shell_process.tracked_start() as publish:
            proc = shell_process.contained_popen(subprocess.Popen, ["sh", "-c", f"exit {code}"],
                                                 **subprocess_new_group_kwargs())
            assert proc.wait(timeout=5) == code  # exit BEFORE publication
            publish(proc)
        assert proc.poll() == code and proc.wait() == code
        assert shell_process._tree_remains(proc) is False
        assert shell_process.retire_tracked(proc)
        with pytest.raises(ChildProcessError):
            platform_layer.observe_child_exit(proc.pid)


@POSIX_ONLY
@pytest.mark.parametrize("retired", [False, True])
def test_retained_popen_is_quiet_at_interpreter_shutdown(retired):
    result = subprocess.run([sys.executable, "-c", f'''
import subprocess, sys
from ouroboros.tools.shell_process import contained_popen, retire_tracked
child = contained_popen(subprocess.Popen, [sys.executable, "-c", "pass"], start_new_session=True)
child.reference_cycle = [child]  # a caller can retain the Popen in callback state until final GC
assert child.wait(timeout=5) == 0
if {retired!r}:
    assert retire_tracked(child)
'''], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stderr == "", "destruction must not import after Python has dismantled its import machinery"


@POSIX_HANDLE
@pytest.mark.parametrize("identity", ["foreign", None])
@pytest.mark.parametrize("leader_exited", [False, True])
def test_recycled_or_unreadable_group_is_never_signalled(registry, monkeypatch, identity, leader_exited):
    """Both reused-number falsifiers: a live replacement leader AND one already gone."""
    from ouroboros import process_identity
    import threading

    proc = types.SimpleNamespace(pid=777777, poll=lambda: 0, _ouroboros_pgid=777777,
        _ouroboros_group_held=True, _ouroboros_group_lock=threading.Lock(), _ouroboros_group_birth="ours")
    # Namespace is unhashable; the actual signalling/retirement primitives take a handle.
    monkeypatch.setattr(process_identity, "process_kernel_birth", lambda pid: identity)
    monkeypatch.setattr(platform_layer, "observe_child_exit", lambda *a: (777777 if leader_exited else 0, 0))
    signals = []
    monkeypatch.setattr(platform_layer.os, "killpg", lambda *a: signals.append(a))
    assert shell_process._tree_remains(proc) is None
    assert not platform_layer.request_process_tree_kill(proc)["requested"]
    shell_process._end_tree(proc)
    assert signals == []


@POSIX_HANDLE
@pytest.mark.parametrize("state", ["live", "exited", "reaped", "unreadable", "contended"])
def test_linux_group_requests_require_both_native_birth_and_waitable_child(monkeypatch, state):
    """Exercise Linux's no-ps request path independently of this host's native ABI."""
    import threading

    proc = types.SimpleNamespace(pid=777777, _ouroboros_group_held=True,
        _ouroboros_group_lock=threading.Lock(), _ouroboros_group_birth="123.abcd")
    monkeypatch.setattr(platform_layer, "IS_MACOS", False)
    monkeypatch.setattr(platform_layer, "_proc_start_ticks", lambda pid: 0 if state == "unreadable" else 123)
    monkeypatch.setattr(platform_layer, "process_start_time", lambda pid: pytest.fail("Panic must not run ps"))
    requests = []
    monkeypatch.setattr(platform_layer, "kill_process_group_id", lambda pid: requests.append(pid) or True)

    def observe(pid, flags):
        if state == "reaped":
            raise ChildProcessError()
        return (pid if state == "exited" else 0), 0

    monkeypatch.setattr(platform_layer, "observe_child_exit", observe)
    if state == "contended":
        proc._ouroboros_group_lock.acquire()
    try:
        started = time.monotonic()
        receipt = platform_layer.request_process_tree_kill(proc)
        assert time.monotonic() - started < 0.5
        owned = state in {"live", "exited"}
        assert receipt["requested"] is owned
        assert requests == ([proc.pid] if owned else [])
    finally:
        if state == "contended":
            proc._ouroboros_group_lock.release()


def test_unknown_job_is_retained_and_terminated_through_its_owned_handle(registry, windows, monkeypatch):
    class Child:
        pid = 9500
        def poll(self):
            return 0
    child = Child()
    child._ouroboros_job = "held"
    windows.running["held"] = None
    shell_process._active_subprocesses.add(child)
    assert not shell_process.retire_tracked(child)
    assert child in shell_process._active_subprocesses
    assert shell_process.kill_all_tracked_subprocesses()["state"] == "unconfirmed"
    assert windows.terminated == ["held"]
    windows.running["held"] = False
    assert shell_process.unsettled_tracked_children() is None
    assert child not in shell_process._active_subprocesses


@pytest.fixture
def registry(monkeypatch):
    """This test's view of the one tracked registry; every tree a case started is ended afterwards."""
    saved = set(shell_process._active_subprocesses)
    shell_process._active_subprocesses.clear()
    monkeypatch.setattr(shell_process, "_panic_requested", False)
    monkeypatch.setattr(shell_process, "_spawning_subprocesses", 0)
    trees: list = []
    yield trees
    for proc, child in trees:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
        _end_own_group(proc, child)
        assert shell_process.retire_tracked(proc)
        for pipe in (proc.stdout, proc.stderr):
            if pipe is not None:
                pipe.close()
    shell_process._active_subprocesses.clear()
    shell_process._active_subprocesses.update(saved)


def _end_own_group(proc, child) -> None:
    """Cleanup through the held leader; a recycled child/group number grants no signal."""
    platform_layer.request_process_tree_kill(proc)
    assert wait_test_child_stop(child, timeout_sec=5)["child_stopped"]


def _published_tree(trees) -> tuple:
    """A tracked ``sh`` whose background child holds the same output pipe; the child's PID comes on that pipe."""
    with shell_process.tracked_start() as publish:
        proc = publish(shell_process.contained_popen(subprocess.Popen, ["sh", "-c", "sleep 60 & echo $!; wait"], stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, **subprocess_new_group_kwargs()))
    child = int(proc.stdout.readline())
    trees.append((proc, child))
    return proc, child


def _root_only_kill(proc) -> None:
    """Windows' former request (``TerminateProcess`` on the root without a Job), emulated: the root PID alone."""
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait(timeout=10)


@POSIX_ONLY
def test_publication_records_the_group_its_session_leads(registry):
    proc, _child = _published_tree(registry)
    assert proc.__dict__.get("_ouroboros_pgid") == proc.pid


@POSIX_ONLY
def test_panics_settle_pass_never_calls_a_dead_root_with_a_live_pipe_holder_settled(registry):
    proc, child = _published_tree(registry)
    _root_only_kill(proc)
    assert not wait_test_child_stop(child, timeout_sec=0.2)["child_stopped"], "the pipe holder outlived its root"
    report = shell_process.kill_all_tracked_subprocesses()  # Panic's settle pass: no deadline, no wait
    stopped = wait_test_child_stop(child, timeout_sec=5)["child_stopped"]
    assert stopped, "the settle pass ended the recorded group the dead root left behind"
    assert report is None or report == {"state": "unconfirmed", "pids": [proc.pid], "unpublished": 0}, report
    if report is None:
        assert proc not in shell_process._active_subprocesses


@POSIX_ONLY
def test_the_ordinary_stop_ends_a_dead_roots_group_and_confirms_only_its_measured_end(registry):
    proc, child = _published_tree(registry)
    _root_only_kill(proc)
    report = shell_process.kill_all_tracked_subprocesses(deadline=time.monotonic() + 5)
    assert wait_test_child_stop(child, timeout_sec=0.5)["child_stopped"], "a settled report means the tree ended"
    assert report is None and proc not in shell_process._active_subprocesses, report


@POSIX_ONLY
def test_a_joined_stop_discloses_a_dead_root_whose_group_still_runs(registry):
    proc, child = _published_tree(registry)
    _root_only_kill(proc)
    assert shell_process.unsettled_tracked_children() == {"state": "unconfirmed", "pids": [proc.pid],
                                                          "unpublished": 0}
    _end_own_group(proc, child)
    deadline = time.monotonic() + 5
    while shell_process.unsettled_tracked_children() is not None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert shell_process.unsettled_tracked_children() is None, "a measured empty group is settled"


def test_an_unmeasured_dead_root_stays_unconfirmed(registry, monkeypatch):
    """No containment record (nothing measured holds its tree): its root's exit is never the tree's."""
    exited = _Child(424242, returncode=0)  # published, but neither a group nor a Job was recorded for it
    shell_process._active_subprocesses.add(exited)
    assert shell_process._tree_remains(exited) is None
    assert shell_process.kill_all_tracked_subprocesses() == {"state": "unconfirmed", "pids": [424242],
                                                             "unpublished": 0}
    assert exited in shell_process._active_subprocesses
    assert shell_process.unsettled_tracked_children()["pids"] == [424242]


@pytest.mark.parametrize(("birth", "current", "signals"), [
    ("born", "born", [424242]), ("born", "reused", []), ("", "born", []), ("born", "", []),
])
def test_browser_smoke_cleanup_signals_only_a_positive_matching_birth(monkeypatch, birth, current, signals):
    from tests import test_tracked_chromium_smoke as smoke

    sent = []
    monkeypatch.setattr(smoke, "_gone", lambda *_args: False)
    monkeypatch.setattr(platform_layer, "process_start_time", lambda _pid: current)
    monkeypatch.setattr(platform_layer, "force_kill_pid", sent.append)
    if not birth or not current:
        with pytest.warns(RuntimeWarning, match="birth identity unavailable"):
            smoke._end_leftovers({424242: ("synthetic owned process", birth)})
    else:
        smoke._end_leftovers({424242: ("synthetic owned process", birth)})
    assert sent == signals


# Windows branches, faked. ``_Child`` stands in for a Popen that created its child: weak-referenceable and
# carrying ``_child_created``, as ``subprocess.Popen`` sets it once its child exists.
class _Child:
    def __init__(self, pid, returncode=None):
        self.pid, self.returncode, self._child_created = pid, returncode, True

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


@pytest.fixture
def windows(monkeypatch):
    """The Windows branches of the tracked registry and the platform layer, over recorded fakes."""
    calls = types.SimpleNamespace(events=[], refuse=set(), stuck=set(), no_job=False, terminated=[], closed=[],
                                  requests=[], refused_request=None, walks=[], running={})
    monkeypatch.setattr(shell_process, "IS_WINDOWS", True)
    monkeypatch.setattr(platform_layer, "IS_WINDOWS", True)

    def create(**kwargs):
        calls.events.append(("create", kwargs))
        return None if calls.no_job else "job"

    def assign(job, pid):
        calls.events.append(("assign", job, pid))
        return pid not in calls.refuse

    def resume(pid):
        calls.events.append(("resume", pid))
        return pid not in calls.stuck

    def terminate(job, exit_code=1):
        if job is not None:
            calls.terminated.append(job)
        return ""

    def request(proc, **_kw):
        calls.requests.append(proc.pid)
        if calls.refused_request:
            return {"pid": proc.pid, "requested": False, "scope": "job", "error": calls.refused_request}
        return {"pid": proc.pid, "requested": True, "scope": "job"}

    monkeypatch.setattr(shell_process, "create_kill_on_close_job", create)
    monkeypatch.setattr(shell_process, "assign_pid_to_job", assign)
    monkeypatch.setattr(shell_process, "resume_process", resume)
    monkeypatch.setattr(shell_process, "close_job", lambda job: calls.closed.append(job) or "")
    monkeypatch.setattr(shell_process, "terminate_job", terminate)
    monkeypatch.setattr(platform_layer, "terminate_job", terminate)
    monkeypatch.setattr(platform_layer, "kill_pid_tree", lambda pid, **kw: calls.walks.append((pid, kw)))
    monkeypatch.setattr(shell_process, "request_process_tree_kill", request)
    monkeypatch.setattr(shell_process, "job_has_running_member", lambda job: calls.running.get(job, False))
    return calls


def _born(monkeypatch, windows, pid):
    """``_ContainedPopen``'s own birth over a faked CreateProcess, which here only names the new child."""
    def create_process(self, *args, **kwargs):
        windows.events.append(("spawn", args, kwargs))
        self.pid = pid

    monkeypatch.setattr(subprocess.Popen, "_execute_child", create_process)
    child = object.__new__(shell_process._ContainedPopen)  # Popen.__init__ refuses creationflags off Windows
    child._execute_child("argv", creationflags=0x4)
    return child


def test_windows_a_tracked_child_is_born_suspended_into_its_own_job_then_resumed(windows, monkeypatch):
    child = _born(monkeypatch, windows, 9100)
    assert [event[0] for event in windows.events] == ["spawn", "create", "assign", "resume"]
    assert windows.events[1] == ("create", {"kill_on_close": False}), "no breakaway; leftovers outlive the command"
    assert windows.events[2:] == [("assign", "job", 9100), ("resume", 9100)], "its Job holds it before it runs"
    assert child._ouroboros_job == "job" and shell_process._tree_remains(child) is False
    del child
    gc.collect()
    assert windows.closed == ["job"], "its Job handle closes with its Popen"


@pytest.mark.parametrize("failure", ["no Job", "assignment refused"])
def test_windows_a_child_without_its_own_job_still_runs_and_its_tree_stays_unknown(windows, monkeypatch, failure):
    windows.no_job = failure == "no Job"
    windows.refuse.add(9200)
    child = _born(monkeypatch, windows, 9200)
    assert windows.events[-1] == ("resume", 9200), "it runs"
    assert ("assign", "job", 9200) in windows.events or windows.no_job
    assert windows.closed == [None if windows.no_job else "job"], "a Job that holds nothing is not kept"
    assert "_ouroboros_job" not in child.__dict__
    assert shell_process._tree_remains(child) is None, "a child born outside a Job is never measured clean"


def test_a_held_popen_init_still_bears_the_child_through_the_subclass(windows, monkeypatch):
    """``_held_popen`` (``test_tracked_command_publication``) replaces ``Popen.__init__`` and holds after the
    original returns. The original calls ``self._execute_child``, so the birth override has already run: a held
    child is contained and resumed before the hold. A real spawn on this interpreter; only the Job calls are faked."""
    original = subprocess.Popen.__init__

    def held(popen, *args, **kwargs):
        original(popen, *args, **kwargs)
        windows.events.append(("held", popen.pid))

    monkeypatch.setattr(subprocess.Popen, "__init__", held)
    child = shell_process._ContainedPopen([sys.executable, "-c", "pass"], stdin=subprocess.DEVNULL)
    try:
        assert windows.events == [("create", {"kill_on_close": False}), ("assign", "job", child.pid),
                                  ("resume", child.pid), ("held", child.pid)]
        assert child._ouroboros_job == "job"
    finally:
        child.wait(timeout=30)


@pytest.mark.parametrize(("refused", "says"), [
    (None, "could not be resumed; its end was requested before it ran"),
    ("TerminateJobObject returned false", r"could not be resumed; its end is unconfirmed \(TerminateJobObject"),
])
def test_windows_a_child_that_cannot_be_resumed_never_runs_and_its_start_is_refused(registry, windows, monkeypatch,
                                                                                refused, says):
    """The refusal says what the end request proved: a request at best, never an observed end."""
    windows.stuck.add(9300)
    windows.refused_request = refused
    with pytest.raises(OSError, match=says) as caught:
        _born(monkeypatch, windows, 9300)
    assert [event[0] for event in windows.events] == ["spawn", "create", "assign", "resume"]
    assert windows.requests == [9300], "its end goes through its own Job; it is never resumed"
    [child] = shell_process._active_subprocesses
    assert child.pid == 9300 and child._ouroboros_job == "job", "failed construction keeps the created child"
    assert caught.value.process_not_started is False


def test_a_failed_resume_preserves_created_child_provenance_through_the_command(monkeypatch):
    error = OSError("created child could not be resumed")
    error.process_not_started = False

    def failed(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(shell_process, "contained_popen", failed)
    with pytest.raises(OSError) as caught:
        shell_process._tracked_subprocess_run(["never-spawned"])
    assert caught.value is error and error.process_not_started is False


def test_the_door_bears_only_the_stdlib_popens_windows_children_suspended(windows, monkeypatch):
    born, direct = [], []

    class Genuine:  # the stdlib class's identity, without a CreateProcess on this host
        def __init__(self, *args, **kwargs):
            direct.append((args, kwargs))

    class Contained:
        def __init__(self, *args, **kwargs):
            born.append((args, kwargs))

    class Wrapper(Genuine):  # a test's subclass: not the stdlib class itself
        pass

    monkeypatch.setattr(shell_process, "_GENUINE_POPEN", Genuine)
    monkeypatch.setattr(shell_process, "_ContainedPopen", Contained)
    shell_process.contained_popen(Genuine, ["cmd"], stdout=-1, creationflags=0x08000200)
    shell_process.contained_popen(Genuine, ["cmd"], creationflags=0x08000000)  # hidden, in the caller's own group
    shell_process.contained_popen(Genuine, ["cmd"])
    assert born == [((["cmd"],), {"stdout": -1, "creationflags": 0x08000200 | 0x4}),
                    ((["cmd"],), {"creationflags": 0x08000000 | 0x4}),
                    ((["cmd"],), {"creationflags": 0x4})] and direct == [], (
        "CREATE_SUSPENDED alone joins the producer's own flags: no new process group (0x200) appears")
    assert shell_process.contained_popen(lambda *a, **k: direct.append((a, k)) or "patched", ["cmd"],
                                         creationflags=0x200) == "patched", "a patched name keeps intercepting"
    shell_process.contained_popen(Wrapper, ["cmd"], creationflags=0x200)
    monkeypatch.setattr(shell_process, "IS_WINDOWS", False)
    monkeypatch.setattr(shell_process, "_RetainedPopen", Genuine)
    shell_process.contained_popen(Genuine, ["cmd"], start_new_session=True)
    assert direct == [((["cmd"],), {"creationflags": 0x200})] * 2 + [((["cmd"],), {"start_new_session": True})]
    assert len(born) == 3, "only the stdlib Popen's Windows children are born suspended"


def test_windows_a_born_contained_child_published_into_panic_is_requested_through_its_job(registry, windows,
                                                                                          monkeypatch):
    """Panic lands while the start is inside its Popen: the starter's request ends the child's whole Job."""
    monkeypatch.setattr(shell_process, "request_process_tree_kill", platform_layer.request_process_tree_kill)
    child = _Child(9400)
    child._ouroboros_job = "job"
    with shell_process.tracked_start() as publish:
        shell_process._panic_requested = True  # Panic, while this start was admitted and inside its Popen
        publish(child)
        started = time.monotonic()
    assert time.monotonic() - started < 0.5, "the request never waits"
    assert windows.terminated == ["job"], "the starter's request ended its whole Job, not its root alone"
    assert shell_process._spawning_subprocesses == 0


def test_windows_panic_request_ends_a_tracked_childs_job(windows):
    child = _Child(4000)
    child._ouroboros_job = "job-4"
    assert platform_layer.request_process_tree_kill(child) == {"pid": 4000, "requested": True, "scope": "job"}
    assert windows.terminated == ["job-4"]


def test_windows_tree_kill_ends_the_childs_own_job_before_any_parent_chain_walk(windows, monkeypatch):
    child = _Child(5000)
    child._ouroboros_job = "job-5"
    platform_layer.kill_process_tree(child)
    assert windows.terminated == ["job-5"] and windows.walks == [], "no taskkill /T walk past a live Job stop"
    platform_layer.kill_process_tree(child, exclude_pids={5001})
    assert windows.walks == [(5000, {"exclude_pids": {5001}, "include_process_group": True})]
    monkeypatch.setattr(platform_layer, "terminate_job", lambda job, exit_code=1: "TerminateJobObject failed")
    platform_layer.kill_process_tree(child)
    assert windows.walks == [(5000, {"exclude_pids": {5001}, "include_process_group": True})], \
        "a failed Job may request the root HANDLE, never walk a possibly recycled PID"


@pytest.mark.parametrize(("job", "job_runs", "expected"), [
    ("job-6", False, False),
    ("job-6", True, True),
    ("job-6", None, None),   # an unreadable Job
    (None, False, None),     # born outside a Job
])
def test_windows_settlement_reads_only_the_childs_own_job(windows, job, job_runs, expected):
    windows.running["job-6"] = job_runs
    child = _Child(6000, returncode=1)
    if job is not None:
        child._ouroboros_job = job
    assert shell_process._tree_remains(child) is expected


def test_windows_stop_ends_a_dead_roots_job_and_confirms_only_its_measured_end(registry, windows):
    child = _Child(7000, returncode=1)
    child._ouroboros_job = "job-7"
    windows.running["job-7"] = True
    shell_process._active_subprocesses.add(child)
    report = shell_process.kill_all_tracked_subprocesses()
    assert windows.terminated == ["job-7"], "a dead root's still-running Job is ended"
    assert report == {"state": "unconfirmed", "pids": [7000], "unpublished": 0}, report
    windows.running["job-7"] = False
    assert shell_process.kill_all_tracked_subprocesses(deadline=time.monotonic() + 1) is None
    assert child not in shell_process._active_subprocesses


def test_job_membership_reads_each_listed_member_by_exit_code(monkeypatch):
    """The native read behind ``_tree_remains`` on Windows: listed members by exit code, never the list size."""
    class ProcessIdList(ctypes.Structure):  # the ABI layout (``platform_win32_abi``) on any host
        _fields_ = [("NumberOfAssignedProcesses", ctypes.c_uint32), ("NumberOfProcessIdsInList", ctypes.c_uint32),
                    ("ProcessIdList", ctypes.c_size_t * 1024)]

    listing = {"ok": True, "pids": []}
    seen = []

    def query(job, info_class, buffer, size, returned):
        seen.append((job, info_class, size))
        listed = buffer._obj
        listed.NumberOfAssignedProcesses = listed.NumberOfProcessIdsInList = len(listing["pids"])
        for index, pid in enumerate(listing["pids"]):
            listed.ProcessIdList[index] = pid
        return listing["ok"]

    states = {1: "exited", 2: "running", 3: "no_such_pid", 4: "denied"}
    monkeypatch.setattr(platform_layer, "ctypes", ctypes, raising=False)
    monkeypatch.setattr(platform_layer, "_JOBOBJECT_BASIC_PROCESS_ID_LIST", ProcessIdList, raising=False)
    monkeypatch.setattr(platform_layer, "_JOBOBJECTINFOCLASS_PROCESS_ID_LIST", 3, raising=False)
    monkeypatch.setattr(platform_layer, "_kernel32", types.SimpleNamespace(QueryInformationJobObject=query),
                        raising=False)
    monkeypatch.setattr(platform_layer, "_windows_process_state", states.__getitem__)
    for pids, expected in (([1, 2], True), ([1, 3], False), ([], False), ([1, 4], None)):
        listing["pids"] = pids
        assert platform_layer.job_has_running_member("job") is expected, pids
    listing["ok"] = False
    assert platform_layer.job_has_running_member("job") is None, "an unreadable Job is unknown, never empty"
    assert {entry[:2] for entry in seen} == {("job", 3)} and {entry[2] for entry in seen} == {
        ctypes.sizeof(ProcessIdList)}


def test_a_tracked_childs_job_refuses_breakaway_which_msys2_reads_before_its_spawns(monkeypatch):
    """MSYS2 adds CREATE_BREAKAWAY_FROM_JOB when the immediate Job has BREAKAWAY_OK or SILENT_BREAKAWAY_OK."""
    class Basic(ctypes.Structure):
        _fields_ = [("LimitFlags", ctypes.c_uint32)]

    class Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", Basic)]

    flags = []

    def set_information(handle, info_class, info, size):
        flags.append(info._obj.BasicLimitInformation.LimitFlags)
        return 1

    monkeypatch.setattr(platform_layer, "IS_WINDOWS", True)
    for name, value in (("ctypes", ctypes), ("_ExtendedLimitInfo", Extended), ("_INVALID_HANDLE_VALUE", -1),
                        ("_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE", 0x2000), ("_JOB_OBJECT_LIMIT_BREAKAWAY_OK", 0x800),
                        ("_JOBOBJECTINFOCLASS_EXTENDED", 9),
                        ("_kernel32", types.SimpleNamespace(CreateJobObjectW=lambda *a: 77,
                                                            SetInformationJobObject=set_information))):
        monkeypatch.setattr(platform_layer, name, value, raising=False)
    assert platform_layer.create_kill_on_close_job(kill_on_close=False) == 77
    assert platform_layer.create_kill_on_close_job() == 77
    assert platform_layer.create_kill_on_close_job(allow_breakaway=True) == 77
    assert flags == [0, 0x2000, 0x2000 | 0x800], "a tracked child's Job sets neither breakaway flag"
