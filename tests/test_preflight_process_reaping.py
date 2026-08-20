"""Reaping the container: what counts as a clean sweep and what counts as a leak.

Split verbatim out of ``tests/test_preflight_runner.py`` by theme. This module owns the
member that stays alive across scans, the corpse that is not a live member, the reads that
become unreadable and are leaks rather than absences, the deadline report naming the last
scan that saw something, and the job teardown that must confirm itself.
"""

from __future__ import annotations

import errno
import os
import subprocess

import pytest



@pytest.mark.skipif(os.name == "nt", reason="POSIX marker-membership reaping")
def test_reap_fails_when_a_member_stays_alive_across_scans(monkeypatch):
    """Quiet has to mean EMPTY, not "no pid I had not already seen".

    The rescan loop used to count a scan as quiet whenever it produced no
    PREVIOUSLY UNSEEN pid, on the theory that a pid still listed after its SIGKILL
    is only a corpse awaiting `wait()`. It is not only that: `force_kill_pid`
    swallows EPERM and every other signalling error, so a member the container
    CANNOT kill is added to `seen` on the first scan, contributes nothing new on
    the second, and the loop returns success — the container reports a reaped tree
    while a token-bearing process is still running, which is the exact fail-open
    the containment work exists to close.

    The kill seam here leaves the same marker-bearing pid visible on every scan,
    which is what a failed signal looks like from inside the loop.
    """
    from ouroboros import platform_layer, process_containment

    survivor = os.getpid() + 1_000_000  # never a live pid; every probe is stubbed
    killed: list[int] = []

    monkeypatch.setattr(process_containment, "_REAP_DEADLINE_SEC", 0.3)
    monkeypatch.setattr(process_containment, "pids_with_env_marker",
                        lambda marker, pgid=0, since_ticks=0: [survivor])
    monkeypatch.setattr(
        process_containment, "pid_marker_state", lambda pid, marker: process_containment.MARKER_MEMBER
    )
    monkeypatch.setattr(process_containment, "pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(platform_layer, "force_kill_pid", lambda pid: killed.append(pid))

    container = process_containment.ProcessContainer()
    error = container.reap()

    assert error, "reap reported success while a marker-bearing member was still alive"
    assert "could not be proven gone" in error, error
    assert killed, "the survivor was never even signalled"

    # The control: once the seam actually clears the member, the SAME loop returns
    # success — the failure above is about liveness, not about the loop refusing
    # to terminate.
    monkeypatch.setattr(process_containment, "pids_with_env_marker",
                        lambda marker, pgid=0, since_ticks=0: [])
    assert process_containment.ProcessContainer().reap() == ""

@pytest.mark.skipif(os.name == "nt", reason="POSIX marker-membership reaping")
def test_reap_does_not_mistake_an_unwaited_corpse_for_a_live_member(monkeypatch):
    """...and the other direction, which is why the liveness test is not just
    `still listed`. `_execute_pytest_pass` reaps the container on the timeout path
    BEFORE it waits pytest, so the SIGKILLed root is a zombie: still holding its
    pid and its pgid in `ps`, executing nothing. Counting it live would spin the
    whole cleanup deadline and then hard-block the run on containment for what was
    really a timeout."""
    from ouroboros import process_containment

    corpse = os.getpid() + 1_000_000

    monkeypatch.setattr(process_containment, "_REAP_DEADLINE_SEC", 0.3)
    monkeypatch.setattr(process_containment, "pids_with_env_marker",
                        lambda marker, pgid=0, since_ticks=0: [corpse])
    monkeypatch.setattr(
        process_containment, "pid_marker_state", lambda pid, marker: process_containment.MARKER_MEMBER
    )
    monkeypatch.setattr(process_containment, "pid_is_zombie", lambda pid: True)

    assert process_containment.ProcessContainer().reap() == "", (
        "an already-exited member was counted as live, so containment blocked a timeout"
    )

@pytest.mark.skipif(os.name == "nt", reason="POSIX marker-membership reaping")
def test_a_member_that_becomes_unreadable_is_a_leak_not_a_clean_reap(monkeypatch):
    """A member the container CANNOT read is not a member it has proven gone.

    Membership is read from the process ENVIRONMENT, which stops being readable the
    moment a member `exec`s something setuid or otherwise nondumpable, or changes
    user. Enumeration claims members positively — deliberately, so a stranger this
    user cannot inspect is never swept into the container — which means such a
    member also DISAPPEARS from the scan. Answering "" there would be the exact
    fail-open the container exists to close: an honest-looking clean teardown for a
    process still running.

    So `reap` keeps its own set of pids it has already seen as members, and a
    member whose recheck comes back UNREADABLE is reported as a leak by pid.
    """
    from ouroboros import platform_layer, process_containment

    ghost = os.getpid() + 1_000_000  # never a live pid; every probe below is stubbed
    scans = []

    def _enumerate(marker, pgid=0, since_ticks=0):
        scans.append(marker)
        # Seen once, then unreadable — so no longer enumerable as a member.
        return [ghost] if len(scans) == 1 else []

    monkeypatch.setattr(process_containment, "_REAP_DEADLINE_SEC", 0.3)
    monkeypatch.setattr(process_containment, "pids_with_env_marker", _enumerate)
    monkeypatch.setattr(
        process_containment, "pid_marker_state",
        lambda pid, marker: process_containment.MARKER_UNREADABLE,
    )
    monkeypatch.setattr(process_containment, "pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(platform_layer, "force_kill_pid",
                        lambda pid: pytest.fail(f"pid {pid} was signalled without revalidation"))

    error = process_containment.ProcessContainer().reap()

    assert error, "a member that vanished into unreadability was reported as reaped"
    assert "could not be determined" in error, error
    assert str(ghost) in error, f"the leaked pid is not named for the operator: {error}"

@pytest.mark.skipif(os.name == "nt", reason="POSIX marker-membership reaping")
def test_a_root_unreadable_from_the_very_first_scan_is_still_a_leak(monkeypatch):
    """The pid `spawn` started is a member the container KNOWS, not one it re-reads.

    Every other member joins the container by being enumerated, which means being
    positively READ. The root is different: it is a member by construction. An
    earlier revision still learned about it only through enumeration, so a root
    that turned nondumpable — `exec`ing something setuid, dropping privileges — or
    changed credentials before the FIRST scan appeared in no list at all, and the
    two empty scans that followed were reported as a clean teardown of a process
    that was still running. There was no later scan to catch it either: the "once
    seen, always watched" set only holds pids it managed to see once.

    So `spawn` records the root and `reap` seeds itself with it. Here enumeration
    NEVER returns it and its membership probe is never answerable, which is the
    exact shape of that escape; the container must still block, by pid.
    """
    from ouroboros import platform_layer, process_containment

    class _FakeProc:
        pid = os.getpid() + 1_000_000  # never a live pid; every probe below is stubbed

    monkeypatch.setattr(subprocess, "Popen", lambda argv, **kwargs: _FakeProc())
    monkeypatch.setattr(process_containment, "_REAP_DEADLINE_SEC", 0.3)
    # The root is invisible to enumeration for the whole reap, from the first scan on.
    monkeypatch.setattr(process_containment, "pids_with_env_marker",
                        lambda marker, pgid=0, since_ticks=0: [])
    monkeypatch.setattr(
        process_containment, "pid_marker_state",
        lambda pid, marker: process_containment.MARKER_UNREADABLE,
    )
    monkeypatch.setattr(process_containment, "pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(platform_layer, "force_kill_pid",
                        lambda pid: pytest.fail(f"pid {pid} was signalled without revalidation"))

    container = process_containment.ProcessContainer()
    container.spawn(["pytest"])
    error = container.reap()

    assert error, "a root that was never readable was reported as a clean teardown"
    assert str(_FakeProc.pid) in error, f"the leaked root is not named for the operator: {error}"
    assert "could not be determined" in error, error

    # The control: the same unenumerable root, ANSWERED as gone, is not a leak —
    # otherwise every ordinary pass would block on its own exited pytest.
    monkeypatch.setattr(
        process_containment, "pid_marker_state", lambda pid, marker: process_containment.MARKER_ABSENT
    )
    replacement = process_containment.ProcessContainer()
    replacement.spawn(["pytest"])
    assert replacement.reap() == "", "an exited root was mistaken for an unreadable one"

@pytest.mark.skipif(os.name == "nt", reason="POSIX marker-membership reaping")
def test_a_descendant_unreadable_before_it_was_ever_seen_is_still_a_leak(monkeypatch):
    """The hole the root seed does NOT plug: a descendant nobody ever managed to read.

    The root is a member by construction, and a member seen once stays in `known`
    forever. Between those two sits the case with no cover at all — a grandchild that
    `exec`s something setuid, or drops privileges, BEFORE the first scan. It was never
    enumerated, so it never entered `known`; it is not the root, so the seed does not
    name it; and its environment is unreadable, so the token can never claim it. Every
    scan came back empty and the container certified a clean teardown of a live tree.

    The process GROUP is what closes it, and only because it is kernel-held: the
    grandchild's pgid is readable from outside no matter what the process did to its
    own environment or credentials. Enumeration therefore takes the group as a second
    input, and once the pid is in `known` the unreadable probe makes it undetermined —
    which fails closed."""
    from ouroboros import platform_layer, process_containment

    class _FakeProc:
        pid = os.getpid() + 1_000_000  # never a live pid; every probe below is stubbed

    hidden = _FakeProc.pid + 7  # a grandchild, not the root

    def _enumerate(marker, pgid=0, since_ticks=0):
        # The token claims nothing: this member has been unreadable since before the
        # first scan. Only the kernel-held group still names it.
        return [hidden] if pgid == _FakeProc.pid else []

    monkeypatch.setattr(subprocess, "Popen", lambda argv, **kwargs: _FakeProc())
    monkeypatch.setattr(platform_layer, "process_group_id",
                        lambda pid: _FakeProc.pid if pid in (_FakeProc.pid, hidden) else 0)
    monkeypatch.setattr(process_containment, "_REAP_DEADLINE_SEC", 0.3)
    monkeypatch.setattr(process_containment, "pids_with_env_marker", _enumerate)
    monkeypatch.setattr(
        process_containment, "pid_marker_state",
        lambda pid, marker: (process_containment.MARKER_ABSENT if pid == _FakeProc.pid
                             else process_containment.MARKER_UNREADABLE),
    )
    monkeypatch.setattr(process_containment, "pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(platform_layer, "force_kill_pid",
                        lambda pid: pytest.fail(f"pid {pid} was signalled without revalidation"))

    container = process_containment.ProcessContainer()
    container.spawn(["pytest"])
    assert container._pgid == _FakeProc.pid, (
        "spawn did not record the root's own group, so enumeration has only the token "
        "and a never-readable descendant is invisible for the whole reap"
    )
    error = container.reap()

    assert error, "a descendant that was never readable was reported as a clean teardown"
    assert str(hidden) in error, f"the leaked descendant is not named: {error}"

@pytest.mark.skipif(os.name == "nt", reason="POSIX marker-membership reaping")
def test_the_deadline_report_names_the_last_scan_that_actually_saw_something(monkeypatch):
    """A block that names no pid is a block the operator cannot act on.

    The remediation tells the operator to go and kill the pids listed, so the report
    has to be built from the last scan that SAW one — not from whichever scan the
    deadline happens to land on. Reporting the current scan means a member that
    flickers out of readability on the final probe produces "nothing is proven gone"
    with no pid at all, from a run that named it moments earlier.

    The member below is alive on the first scan and gone from the second, and the
    deadline is set to expire in the 50ms settle between them — so the scan the
    deadline lands on is empty while the run has already named a pid. One quiet scan
    is not two, so this is a BLOCK either way; the question is whether it is an
    actionable one."""
    from ouroboros import platform_layer, process_containment

    flicker = os.getpid() + 1_000_000  # never a live pid; every probe is stubbed
    scans: list[str] = []

    def _enumerate(marker, pgid=0, since_ticks=0):
        scans.append(marker)
        return [flicker] if len(scans) == 1 else []

    # Shorter than one settle interval, so it expires during the sleep after scan 1
    # and the loop exits on scan 2 — before quiet could ever reach two.
    monkeypatch.setattr(process_containment, "_REAP_DEADLINE_SEC", 0.03)
    monkeypatch.setattr(process_containment, "pids_with_env_marker", _enumerate)
    monkeypatch.setattr(
        process_containment, "pid_marker_state",
        lambda pid, marker: (process_containment.MARKER_MEMBER if len(scans) == 1
                             else process_containment.MARKER_ABSENT),
    )
    monkeypatch.setattr(process_containment, "pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(platform_layer, "force_kill_pid", lambda pid: None)

    error = process_containment.ProcessContainer().reap()

    assert len(scans) >= 2, (
        f"the reap exited on its first scan ({len(scans)}), so 'the LAST non-empty "
        "scan' is not being exercised at all"
    )
    assert error, "the flickering member was reported as a clean teardown"
    assert str(flicker) in error, (
        "the deadline report was built from the empty final scan, so it names no pid "
        f"for the operator to act on: {error}"
    )

@pytest.mark.skipif(os.name == "nt", reason="POSIX marker-membership reaping")
def test_a_member_is_signalled_at_most_once_however_long_the_scans_run(monkeypatch):
    """Killing is ONE bounded sweep; everything after it is scan-only.

    The signal is the one operation here that can hit the wrong process: between
    revalidating a pid as a member and sending SIGKILL, that pid can exit and be
    handed to a stranger. The window cannot be closed — it is the reason the
    contract is detection rather than guaranteed teardown — so the fix is to enter
    it as FEW times as possible. An earlier revision had `force_kill_pid` inside
    the rescan loop, re-signalling every still-visible member roughly every 50ms
    for up to ten seconds: two hundred throws of the same dice, buying nothing,
    since a member that survived the first SIGKILL is one we cannot kill (EPERM)
    and the block is already earned.

    The member below stays visible on every scan, which is what an unkillable one
    looks like from inside the loop; it must be signalled exactly once, and the
    verdict must still be a block.
    """
    from ouroboros import platform_layer, process_containment

    survivor = os.getpid() + 1_000_000  # never a live pid; every probe is stubbed
    killed: list[int] = []
    scans: list[str] = []

    def _enumerate(marker, pgid=0, since_ticks=0):
        scans.append(marker)
        return [survivor]

    monkeypatch.setattr(process_containment, "_REAP_DEADLINE_SEC", 0.5)
    monkeypatch.setattr(process_containment, "pids_with_env_marker", _enumerate)
    monkeypatch.setattr(
        process_containment, "pid_marker_state", lambda pid, marker: process_containment.MARKER_MEMBER
    )
    monkeypatch.setattr(process_containment, "pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(platform_layer, "force_kill_pid", lambda pid: killed.append(pid))

    error = process_containment.ProcessContainer().reap()

    assert len(scans) > 2, (
        f"the reap only scanned {len(scans)} time(s), so 'at most once' is vacuous here"
    )
    assert killed == [survivor], (
        f"the sweep is not bounded: {survivor} was signalled {len(killed)} times across "
        f"{len(scans)} scans, re-entering the exit/pid-reuse race on every one"
    )
    assert error and str(survivor) in error, (
        f"signalling once must not weaken the verdict; the leak went unreported: {error}"
    )

def test_the_ps_membership_branch_answers_unreadable_for_a_live_pid(monkeypatch):
    """The same tri-state, on the POSIX systems that have no `/proc`.

    macOS and the BSDs answer membership with `ps -E`, and `ps -E` reports a
    process whose environment it may not print by simply OMITTING the environment
    — byte-identical to a process that never carried the token. Collapsing that
    into "absent" reopened, on exactly those platforms, the escape the tri-state
    was introduced to close: a member turns uninspectable, drops out of
    enumeration, and two quiet scans call it reaped. Only `ps` failing to find the
    pid at all (non-zero exit) may answer ABSENT.

    The `/proc` sibling pin forces the `/proc` branch on every POSIX host, so this
    branch is otherwise unpinned in either direction.
    """
    import types

    from ouroboros import platform_layer, process_containment

    if platform_layer.IS_WINDOWS:
        pytest.skip("POSIX environment-token membership")

    # Both shims lie about `ps`/`/proc` ONLY and delegate everything else, so the
    # branch is pinned on a Linux host too and nothing else running inside this
    # test (pytest's own reporting included) is affected.
    real_isdir = os.path.isdir
    monkeypatch.setattr(platform_layer.os.path, "isdir",
                        lambda path: False if path == "/proc" else real_isdir(path))

    result = {"rc": 0, "out": "/usr/bin/python3 -c pass\n"}
    real_run = subprocess.run

    def _run(argv, **kwargs):
        if not (isinstance(argv, (list, tuple)) and argv and argv[0] == "ps"):
            return real_run(argv, **kwargs)
        return types.SimpleNamespace(returncode=result["rc"], stdout=result["out"], stderr="")

    monkeypatch.setattr(platform_layer.subprocess, "run", _run)

    assert process_containment.pid_marker_state(1234, "TOKEN") == process_containment.MARKER_UNREADABLE, (
        "a live pid whose environment `ps` declined to print was reported as proof "
        "of non-membership, so an uninspectable member leaves containment silently"
    )

    # The two answers that ARE answers: the token is there, or the pid is not.
    result["out"] = "/usr/bin/python3 -c pass TOKEN=1\n"
    assert process_containment.pid_marker_state(1234, "TOKEN") == process_containment.MARKER_MEMBER
    result["rc"], result["out"] = 1, ""
    assert process_containment.pid_marker_state(1234, "TOKEN") == process_containment.MARKER_ABSENT, (
        "a pid `ps` cannot find must be ABSENT, or every ordinary exit blocks the gate"
    )

def test_an_unanswerable_membership_probe_is_unreadable_not_absent(monkeypatch):
    """The unit beneath that pin: `pid_marker_state` distinguishes three answers.

    Its predecessor returned a BOOLEAN, which folded `PermissionError` into "not a
    member" — the read failed, so the pid looked innocent. Only ESRCH/ENOENT (the
    pid is genuinely gone) may answer ABSENT; every other `OSError` is UNREADABLE.
    """
    import builtins

    from ouroboros import platform_layer, process_containment

    if platform_layer.IS_WINDOWS:
        pytest.skip("POSIX environment-token membership")

    # Pinned against the /proc branch on every POSIX host, so the distinction is
    # not silently unpinned on a machine that happens to lack /proc. Both shims
    # lie about `/proc` ONLY and delegate everything else, so nothing else running
    # inside this test (pytest's own reporting included) is affected.
    real_isdir = os.path.isdir
    monkeypatch.setattr(platform_layer.os.path, "isdir",
                        lambda path: True if path == "/proc" else real_isdir(path))
    real_open = builtins.open

    def _open_raising(errno_value):
        def _open(path, *args, **kwargs):
            if str(path).startswith("/proc/"):
                raise OSError(errno_value, os.strerror(errno_value))
            return real_open(path, *args, **kwargs)
        return _open

    monkeypatch.setattr(builtins, "open", _open_raising(errno.EACCES))
    assert process_containment.pid_marker_state(1234, "TOKEN") == process_containment.MARKER_UNREADABLE, (
        "an unreadable environment was reported as proof of non-membership"
    )

    # The control: a pid that is genuinely gone is ANSWERED, not undetermined, or
    # every ordinary exit would block the run.
    monkeypatch.setattr(builtins, "open", _open_raising(errno.ESRCH))
    assert process_containment.pid_marker_state(1234, "TOKEN") == process_containment.MARKER_ABSENT
    monkeypatch.setattr(builtins, "open", _open_raising(errno.ENOENT))
    assert process_containment.pid_marker_state(1234, "TOKEN") == process_containment.MARKER_ABSENT

def test_a_windows_job_teardown_that_does_not_confirm_itself_is_a_containment_failure(monkeypatch):
    """Win32 reports failure by RETURN VALUE, and a false BOOL was being discarded.

    The Job Object is the one place teardown really is kernel-enforced, which is
    why its result is the whole Windows verdict: if `TerminateJobObject` returns
    false the job's processes are still running, and if `CloseHandle` returns false
    kill-on-close — the backstop for a termination that did not take — never fires
    AND the handle leaks. Both used to be called for effect and ignored, so `reap`
    returned "" for a job it had not torn down.

    The code must be read with `ctypes.get_last_error()`, not `ctypes.GetLastError()`:
    the handle is opened with `use_last_error=True`, which makes ctypes SNAPSHOT the
    thread's last error immediately after each call into its own private slot. The
    raw `GetLastError` reads the live thread value, which ctypes' own bookkeeping
    between the failing call and the read has by then overwritten — so the operator
    is handed an unrelated code for a containment failure.
    """
    import inspect
    import types

    from ouroboros import platform_layer, process_containment

    win_src = inspect.getsource(platform_layer.terminate_job) + inspect.getsource(
        platform_layer.close_job)
    assert "get_last_error" in win_src and "ctypes.GetLastError" not in win_src, (
        "the Win32 failure code is read with the raw GetLastError again; with "
        "use_last_error=True that is not the code the failing call set"
    )

    monkeypatch.setattr(platform_layer, "IS_WINDOWS", True)
    monkeypatch.setattr(platform_layer, "ctypes", types.SimpleNamespace(get_last_error=lambda: 5),
                        raising=False)

    results = {"terminate": 0, "close": 1}
    monkeypatch.setattr(
        platform_layer, "_kernel32",
        types.SimpleNamespace(
            TerminateJobObject=lambda job, code: results["terminate"],
            CloseHandle=lambda job: results["close"],
        ),
        raising=False,
    )

    container = process_containment.ProcessContainer()
    container._job = object()
    error = container.reap()
    assert "TerminateJobObject" in error and "5" in error, error

    # A close that fails is equally a leak, and a raised call is not different from
    # a false return — both leave the job unaccounted for.
    results["terminate"], results["close"] = 1, 0
    container = process_containment.ProcessContainer()
    container._job = object()
    assert "kill-on-close never fired" in container.reap()

    def _raise(*args):
        raise OSError("the handle is invalid")

    monkeypatch.setattr(
        platform_layer, "_kernel32",
        types.SimpleNamespace(TerminateJobObject=_raise, CloseHandle=_raise),
        raising=False,
    )
    container = process_containment.ProcessContainer()
    container._job = object()
    error = container.reap()
    assert "the handle is invalid" in error, error

    # The control: a job that confirms both halves is a clean reap.
    monkeypatch.setattr(
        platform_layer, "_kernel32",
        types.SimpleNamespace(TerminateJobObject=lambda job, code: 1, CloseHandle=lambda job: 1),
        raising=False,
    )
    container = process_containment.ProcessContainer()
    container._job = object()
    assert container.reap() == ""
