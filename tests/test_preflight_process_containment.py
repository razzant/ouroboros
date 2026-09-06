"""Holding the processes a pass spawns, and finding the ones that tried to leave.

Split verbatim out of ``tests/test_preflight_runner.py`` by theme. This module owns the
Windows job seam and its spawn race, the process group used as a detection input and never
signalled, the detached child still found after its root exits, the membership token planted
in the environment, and the stranger on a recycled pid that is never signalled.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from ouroboros.platform_layer import force_kill_pid, pid_is_alive


def test_windows_containment_uses_the_shared_job_seam_and_closes_the_spawn_race(monkeypatch):
    """The Windows branch must reuse platform_layer's OWN Job Object seam, and it
    must assign the process to the job BEFORE the process can run.

    Two properties, both invisible on POSIX and therefore covered by nothing
    until now — every other containment test here is `skipif(os.name == "nt")`,
    and the gate/CI runners that execute this file are POSIX:

    * The four helpers are called, rather than a second private ctypes binding of
      the same API being maintained alongside them.
    * Ordering. A Job Object only holds what is assigned to it, so anything the
      process starts between `Popen` returning and the assignment is NOT a member
      and survives terminate/close. `spawn` therefore creates it suspended
      (CREATE_SUSPENDED) and resumes it only after assignment — the same sequence
      launcher.py uses for the agent server.

    Executable everywhere because `IS_WINDOWS`, the group kwargs and the four
    helpers are all stubbed; the assertion is on the call sequence, not on the
    Win32 API.
    """
    from ouroboros import platform_layer, process_containment

    calls: list = []

    class _FakeProc:
        pid = 4321

    def _fake_popen(argv, **kwargs):
        calls.append(("popen", int(kwargs.get("creationflags", 0))))
        return _FakeProc()

    monkeypatch.setattr(platform_layer, "IS_WINDOWS", True)
    # CREATE_NEW_PROCESS_GROUP does not exist in `subprocess` on POSIX, so the
    # real kwargs helper cannot run here; its Windows return value is stubbed.
    monkeypatch.setattr(platform_layer, "subprocess_new_group_kwargs", lambda: {"creationflags": 0x200})
    monkeypatch.setattr(subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(platform_layer, "create_kill_on_close_job", lambda: calls.append(("create_job",)) or "job")
    monkeypatch.setattr(platform_layer, "assign_pid_to_job", lambda job, pid: calls.append(("assign", job, pid)) or True)
    monkeypatch.setattr(platform_layer, "resume_process", lambda pid: calls.append(("resume", pid)) or True)
    monkeypatch.setattr(platform_layer, "terminate_job", lambda job, *rest: calls.append(("terminate", job)) or "")
    monkeypatch.setattr(platform_layer, "close_job", lambda job: calls.append(("close", job)) or "")

    container = process_containment.ProcessContainer()
    proc = container.spawn(["pytest"], cwd=".")
    # `reap` performs BOTH halves of the teardown, because only its return value can
    # reach the pass verdict; `close` afterwards is inert (the handle is consumed).
    container.reap()
    container.close()

    assert proc.pid == 4321
    assert [call[0] for call in calls] == [
        "popen", "create_job", "assign", "resume", "terminate", "close",
    ], f"wrong containment sequence: {calls}"
    flags = calls[0][1]
    assert flags & 0x4, (
        "the process was not created suspended, so a descendant spawned before "
        "job assignment escapes containment"
    )
    assert flags & 0x200, "the new-process-group creation flag was dropped"
    assert calls[2] == ("assign", "job", 4321)
    assert calls[3] == ("resume", 4321)

def test_a_windows_root_that_cannot_be_job_held_is_killed_and_reported(monkeypatch):
    """A root no Job Object can hold is terminated, never resumed uncontained.

    `spawn` creates the process suspended so nothing escapes before assignment.
    An earlier revision resumed it anyway when the job could not be created,
    treating containment as best effort — but the unconditional post-pass reap
    then reported a clean teardown for a tree NOTHING was holding, and on Windows
    neither `CREATE_NEW_PROCESS_GROUP` nor `taskkill /T` can find a descendant
    whose parent has exited. A reap that cannot fail is not a reap.

    So the failure is loud in both directions: the still-suspended root dies
    (it is never left suspended either, which would deadlock a caller waiting on
    it), and `reap()` returns a non-empty reason the pass loop hard-blocks on.
    """
    from ouroboros import platform_layer, process_containment

    calls: list = []

    class _FakeProc:
        pid = 99

    monkeypatch.setattr(platform_layer, "IS_WINDOWS", True)
    monkeypatch.setattr(platform_layer, "subprocess_new_group_kwargs", lambda: {"creationflags": 0x200})
    monkeypatch.setattr(subprocess, "Popen", lambda argv, **kwargs: _FakeProc())
    monkeypatch.setattr(platform_layer, "create_kill_on_close_job", lambda: None)
    monkeypatch.setattr(platform_layer, "assign_pid_to_job", lambda job, pid: calls.append(("assign", pid)) or True)
    monkeypatch.setattr(platform_layer, "resume_process", lambda pid: calls.append(("resume", pid)) or True)
    monkeypatch.setattr(platform_layer, "force_kill_pid", lambda pid: calls.append(("kill", pid)))
    monkeypatch.setattr(platform_layer, "terminate_job", lambda job, *rest: calls.append(("terminate", job)) or "")
    monkeypatch.setattr(platform_layer, "close_job", lambda job: "")

    container = process_containment.ProcessContainer()
    container.spawn(["pytest"])

    assert ("kill", 99) in calls, "the unheld root was left running outside any container"
    assert ("resume", 99) not in calls, (
        "the root was resumed with no Job Object holding it, so anything it spawns "
        "survives terminate/close while the reap still reports success"
    )
    assert container._suspended is False, (
        "the container still believes the root is suspended, so a caller waiting "
        "on it would deadlock"
    )
    reason = container.reap()
    assert reason, "a container that never held the tree reported a clean reap"
    assert "Job Object" in reason, reason
    container.close()

def test_the_process_group_is_a_detection_input_and_is_never_signalled():
    """The process group is read to DETECT members and is never a kill target.

    Every PID-reuse edge this container has had came from signalling the group: an
    emptied pgid is free for reuse, and `killpg` from a snapshot cannot prove the id
    is still ours (a `lstart` fingerprint is second-resolution, so a stranger born in
    the same second passes). Under the detection contract that asymmetry decides it.
    Reading the group can only ever ADD a pid to the leak report, and a stale pgid
    then costs a false BLOCK — the safe direction; signalling it kills a bystander,
    which no rescan can undo. So the group stays as an enumeration and classification
    input (it is the only membership signal that survives a child replacing its whole
    environment) and this pin keeps a future "we already know the pgid, just killpg
    it" from turning a detection input back into a weapon."""
    import inspect

    from ouroboros import platform_layer
    from ouroboros.process_containment import ProcessContainer

    container = ProcessContainer()
    assert container._pgid == 0, (
        "a container that has spawned nothing claims a process group; enumerating "
        "one would report the CALLER's own group members as leaks"
    )
    for method in (ProcessContainer.reap, ProcessContainer._scan, ProcessContainer.adopt,
                   ProcessContainer.spawn):
        source = inspect.getsource(method)
        assert "killpg" not in source and "kill_process_group_id" not in source, (
            f"{method.__name__} signals a process group again: {source}"
        )
    scan_src = inspect.getsource(ProcessContainer._scan)
    group_branch = scan_src.split("elif pgid and _pl.process_group_id(pid) == pgid:", 1)
    assert len(group_branch) == 2, (
        "the group-membership branch is gone from `_scan`; a contained child that "
        "replaces its whole environment is then invisible to every membership signal"
    )
    assert "force_kill_pid" not in group_branch[1].split("elif", 1)[0], (
        "the group branch signals the pid it detected; a pgid is a borrowed name, so "
        "that is a SIGKILL aimed at whoever inherited it"
    )
    assert not hasattr(platform_layer, "snapshot_processes"), (
        "the stale process-table snapshot is back; membership is decided from live "
        "kernel state per scan, and deciding it from a snapshot is the bug"
    )
    container.reap()  # must be inert, not suicide
    container.close()

@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_a_member_that_replaced_its_environment_is_still_detected_by_its_group():
    """The blind spot the group closes: a child spawned with a REPLACED environment.

    `Popen(env={...})` — an ordinary thing for a test to do — drops the container's
    token, so the environment signal reports the child as a non-member and the reap
    comes back clean while it is still running. The kernel still places it in the
    root's process group, and that fact needs no cooperation from the child."""
    from ouroboros.process_containment import (MARKER_MEMBER, ProcessContainer,
                                               pid_marker_state, pids_with_env_marker)

    container = ProcessContainer()
    token = container._token
    child = ("import subprocess, sys, time\n"
             "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],"
             " env={'PATH': '/usr/bin:/bin'})\n"
             "print(p.pid, flush=True)\n"
             "time.sleep(30)\n")
    root = container.spawn(
        [sys.executable, "-c", child],
        env={"PATH": os.environ.get("PATH", "")},
        stdout=subprocess.PIPE,
        text=True,
    )
    scrubbed = 0
    try:
        assert container._pgid == root.pid, (
            "spawn did not make the root its own group LEADER; enumerating any other "
            "group would sweep in processes this container never created"
        )
        scrubbed = int((root.stdout.readline() or "0").strip())
        assert scrubbed, "the env-scrubbed grandchild never started"
        assert pid_marker_state(scrubbed, token) != MARKER_MEMBER, (
            "the grandchild kept the token, so this pin is no longer exercising the "
            "environment signal's blind spot"
        )
        assert scrubbed not in (pids_with_env_marker(token, 0) or []), (
            "the token alone found an env-scrubbed process; the fixture is wrong"
        )
        assert scrubbed in (pids_with_env_marker(token, container._pgid) or []), (
            "the env-scrubbed grandchild was in no membership list; it would outlive "
            "the pass with the reap reporting a clean container"
        )
        alive, undetermined, error = container._scan(token, container._pgid, set(), kill=False)
        assert not error, error
        assert scrubbed in alive, (
            "`_scan` enumerated the group-only member but did not classify it as "
            f"alive: alive={alive} undetermined={undetermined} scrubbed={scrubbed}"
        )
    finally:
        # The group-only member is DETECTED and deliberately never signalled, so it has
        # to go before `reap`, or the reap spends its whole deadline proving that.
        for pid in (scrubbed, root.pid):
            if pid and pid_is_alive(pid):
                force_kill_pid(pid)
        container.reap()
        container.close()
        if root.stdout is not None:
            root.stdout.close()

@pytest.mark.skipif(os.name == "nt", reason="POSIX environment membership")
def test_a_detached_child_is_still_found_after_its_root_exits():
    """The property the preflight depends on: membership survives the ROOT exiting,
    which is exactly the moment the parent->child walk stops working.

    The token the kernel copied into the child's environment at `fork` is what still
    names it — `adopt` on a bare `Popen` cannot plant one, which is why the gate
    always uses `spawn`. The container both kills it (best effort) and, if it were
    still there, would say so; here it is genuinely gone, so `reap` returns clean."""
    from ouroboros.platform_layer import pid_is_alive
    from ouroboros.process_containment import ProcessContainer

    container = ProcessContainer()
    root = container.spawn(
        [sys.executable, "-c",
         "import subprocess,sys;"
         "c=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']);"
         "print(c.pid, flush=True);"
         "sys.exit(0)"],
        stdout=subprocess.PIPE,
        text=True,
    )
    child_pid = 0
    try:
        child_pid = int((root.stdout.readline() or "0").strip())
        root.wait(timeout=30)
        assert child_pid, "the fixture root never reported its child"
        assert pid_is_alive(child_pid), "fixture precondition: the child outlives its root"

        reason = container.reap()

        deadline = time.time() + 10
        while time.time() < deadline and pid_is_alive(child_pid):
            time.sleep(0.1)
        assert not pid_is_alive(child_pid), (
            f"{child_pid} survived its container after the root process exited"
        )
        assert reason == "", f"a tree that really was cleared still reported a leak: {reason}"
    finally:
        container.close()
        if root.stdout is not None:
            root.stdout.close()
        if root.poll() is None:
            force_kill_pid(root.pid)
        if child_pid and pid_is_alive(child_pid):
            force_kill_pid(child_pid)

@pytest.mark.skipif(os.name == "nt", reason="POSIX marker-membership containment")
def test_process_container_kills_a_descendant_that_left_the_group():
    """`setsid()` moves a descendant into its OWN session and process group, so a
    pgid stops naming it, and it keeps running after the root exits with no ppid
    link left to find it by. A daemonising child takes exactly this shape, so this
    is the escape the environment membership token exists for — a group-only
    container reports a clean reap here while the escapee runs on.

    The root spawns the escapee and exits IMMEDIATELY, with no sleep. That is
    the point: the previous containment fingerprinted descendants from a 0.5s
    background poll, so it could only find an escapee that happened to be alive
    and still parented across a sample, and the regression had to sleep for two
    seconds to give it one. A child born, detached and orphaned inside a single
    poll interval — the fastest and most ordinary shape — escaped entirely.
    Membership is now read from the kernel AT REAP TIME, so there is no window
    to be born inside."""
    from ouroboros.platform_layer import pid_is_alive, process_group_id
    from ouroboros.process_containment import ProcessContainer

    container = ProcessContainer()
    root = container.spawn(
        [sys.executable, "-c",
         "import subprocess,sys;"
         "c=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'],"
         "start_new_session=True,"
         "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);"
         "print(c.pid, flush=True);"
         "sys.exit(0)"],
        stdout=subprocess.PIPE,
        text=True,
    )
    escapee = 0
    try:
        escapee = int((root.stdout.readline() or "0").strip())
        assert escapee, "the fixture root never reported its child"
        root_pgid = process_group_id(root.pid)
        assert process_group_id(escapee) != root_pgid, (
            "fixture precondition: setsid() must have moved the child out of the group"
        )
        root.wait(timeout=30)
        assert pid_is_alive(escapee), "fixture precondition: the escapee outlives its root"

        reason = container.reap()

        deadline = time.time() + 10
        while time.time() < deadline and pid_is_alive(escapee):
            time.sleep(0.1)
        assert not pid_is_alive(escapee), (
            f"{escapee} escaped the container by leaving its process group"
        )
        # Detection is the contract, so the two answers must agree: a scan that
        # returned clean while the escapee ran would be the fail-open itself.
        assert reason == "", f"the escapee was cleared but reap reported a leak: {reason}"
    finally:
        container.close()
        if root.stdout is not None:
            root.stdout.close()
        if root.poll() is None:
            force_kill_pid(root.pid)
        if escapee and pid_is_alive(escapee):
            force_kill_pid(escapee)

@pytest.mark.skipif(os.name == "nt", reason="POSIX environment membership")
def test_spawn_plants_the_membership_token_in_a_caller_supplied_env():
    """`_execute_pytest_pass` hands `spawn` its own fully-scrubbed env dict, so the
    token has to be MERGED into it. Dropped, the container has no POSIX membership
    at all: every scan comes back empty and `reap` honestly — and uselessly —
    reports a clean teardown for a tree it was never able to see."""
    from ouroboros.process_containment import ProcessContainer, pids_with_env_marker

    container = ProcessContainer()
    token = container._token
    proc = container.spawn(
        [sys.executable, "-c", "import time; print('up', flush=True); time.sleep(30)"],
        env={"PATH": os.environ.get("PATH", "")},
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert (proc.stdout.readline() or "").strip() == "up", "the fixture never started"
        assert proc.pid in pids_with_env_marker(token), (
            "the container's own root does not carry its membership token, so no "
            "descendant can inherit it either"
        )
    finally:
        container.reap()
        container.close()
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.poll() is None:
            force_kill_pid(proc.pid)
        proc.wait(timeout=10)

@pytest.mark.skipif(os.name == "nt", reason="POSIX environment membership")
def test_the_membership_token_survives_the_preflight_env_scrub(tmp_path, monkeypatch):
    """`_preflight_env` drops the whole `OUROBOROS_*` namespace, so the token
    deliberately lives OUTSIDE it. This suite itself runs nested preflights, and
    a nested `_preflight_env` that stripped the outer container's token would
    hide the entire inner tree from the outer reap — each container matches only
    its own uuid, so the tokens are meant to compose, not to overwrite."""
    from ouroboros.process_containment import CONTAINMENT_ENV_PREFIX
    from ouroboros.preflight_runner import _preflight_env

    assert not CONTAINMENT_ENV_PREFIX.startswith("OUROBOROS_"), (
        "the token sits inside the namespace _preflight_env sweeps"
    )
    outer = CONTAINMENT_ENV_PREFIX + "outer0123456789"
    monkeypatch.setenv(outer, "1")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")

    env = _preflight_env(tmp_path / "root", tmp_path / "root" / "repo")

    assert env.get(outer) == "1", "the outer container's membership token was scrubbed"
    assert "OUROBOROS_SAFETY_MODE" not in env, "the ordinary runtime scrub regressed"

@pytest.mark.skipif(os.name == "nt", reason="POSIX marker-membership containment")
def test_a_stranger_that_took_a_recycled_pid_or_pgid_is_never_signalled(monkeypatch):
    """A pid is a borrowed name: once a member exits the kernel is free to hand its
    number to a build, an editor, or the operator's shell. The container therefore
    signals nothing it has not just re-read as a token-bearing member.

    Three recycled names are exercised, because they resolve DIFFERENTLY and only one
    of the three answers may be a signal. A recycled ROOT PID that the live
    environment DISPROVES is dropped outright. One it cannot read is not disproved,
    so the seeded root fails CLOSED and is reported. A recycled PGID cannot be
    disproved either — the kernel really does place the stranger in it — so it is
    reported too. That is the whole asymmetry the fail-closed branches are allowed to
    exist under: they may cost a false BLOCK the operator can clear, never a SIGKILL
    on a bystander that no rescan can undo. The `pid_is_alive` assertion is the one
    that carries the safety property, and it holds on every platform.

    The membership answer itself is stubbed rather than read from the real stranger,
    because the two POSIX backends answer a live foreign pid DIFFERENTLY by design:
    `/proc` raises ENOENT-or-nothing, so a readable stranger answers `absent`, while
    `ps -E` omits an environment it may not print and is indistinguishable from one
    that never carried the token, which the non-`/proc` branch deliberately calls
    `unreadable` and blocks on. Reading the host would therefore pin whichever
    contract the gate runner happens to have."""
    from ouroboros import process_containment
    from ouroboros.platform_layer import pid_is_alive
    from ouroboros.process_containment import ProcessContainer

    stranger = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # its own leader, so pgid == pid, as a root's is
    )
    answer = [process_containment.MARKER_ABSENT]
    monkeypatch.setattr(
        process_containment, "pid_marker_state",
        lambda pid, marker: (answer[0] if pid == stranger.pid
                             else process_containment.MARKER_ABSENT),
    )
    container = ProcessContainer()
    try:
        # A container whose own tree is gone and whose ROOT PID was recycled onto
        # `stranger`, whose environment POSITIVELY disproves membership.
        container._root = stranger.pid
        assert container.reap() == "", (
            "a stranger holding the recycled root pid was reported as a leak; "
            "membership is claimed positively from the live environment"
        )
        assert pid_is_alive(stranger.pid), (
            f"the container SIGKILLed pid {stranger.pid}, which it never contained"
        )

        # The same recycled root pid, now UNREADABLE. Nothing was disproved, so the
        # seeded root is a leak — and still nothing is signalled, which is what keeps
        # fail-closed from becoming a licence to kill whatever it cannot identify.
        monkeypatch.setattr(process_containment, "_REAP_DEADLINE_SEC", 0.5)
        answer[0] = process_containment.MARKER_UNREADABLE
        unreadable = ProcessContainer()
        unreadable._root = stranger.pid
        reason = unreadable.reap()
        assert str(stranger.pid) in reason, (
            f"an unreadable root was reported as a clean reap: {reason!r}"
        )
        assert pid_is_alive(stranger.pid), (
            f"the container SIGKILLed pid {stranger.pid} on an unreadable probe; "
            "'cannot tell' is a block, never a signal"
        )
        unreadable.close()

        # Same stranger, now holding the recycled process GROUP id.
        group = ProcessContainer()
        group._root, group._pgid = stranger.pid, stranger.pid
        reason = group.reap()
        assert str(stranger.pid) in reason, (
            "the container could not disprove membership and still reported a clean "
            f"reap: {reason!r}"
        )
        assert pid_is_alive(stranger.pid), (
            f"the container SIGKILLed pid {stranger.pid} off a recycled pgid; a group "
            "is a borrowed name and must only ever DETECT"
        )
        group.close()
    finally:
        container.close()
        if stranger.poll() is None:
            force_kill_pid(stranger.pid)
        stranger.wait(timeout=10)
