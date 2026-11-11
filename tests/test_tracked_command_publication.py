"""Tracked children against the generation's stop: admission, publication and the honest outcome.

Every producer starts its child through ``shell_process.tracked_start``. Real sleepers stand in for
the children; a hold on Popen's RETURN puts a start in flight exactly where the stop meets it (the
OB1656 witness schedules). Only children these tests started are signalled, and each is reaped. The
restart exit's own finalization (``exit_finalization``) is the one start a finished stop admits: the
real deferred Restore over isolated Git and store roots, and Panic still ending it.
"""
from __future__ import annotations

import contextlib
import os
import pathlib
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import extension_process_runner, owned_shutdown
from ouroboros.marketplace import isolated_deps
from ouroboros.platform_layer import subprocess_new_group_kwargs
from ouroboros.tools import shell_process, skill_exec, skill_preflight
from supervisor import git_ops, git_ops_reset, update_merge
from tests._shared import calls_function, wait_test_child_stop
from tests.test_store_switch import git
from tests.test_store_switch import switch_root as _switch_root
from tests.test_store_switch import target_repo as _switch_target_repo

switch_root, target_repo = _switch_root, _switch_target_repo  # real store roots; ``old`` lacks the stores

SLEEP = [sys.executable, "-c", "import time; time.sleep(60)"]


def test_tracked_registry_publishers_start_through_tracked_start():
    """A bounded helper the tracked registry names (``_active_subprocesses``) counts its start before
    Popen, is refused once the generation's stop door began and publishes the returned handle, all
    through ``shell_process.tracked_start``; a raw ``add`` can hide a child from that stop."""
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    offenders = []
    for base in ("ouroboros", "supervisor"):
        for path in (repo_root / base).rglob("*.py"):
            rel = path.relative_to(repo_root).as_posix()
            if rel in ("ouroboros/tools/shell_process.py", "ouroboros/tools/shell.py"):
                continue  # the registry's owner and its facade
            text = path.read_text(encoding="utf-8", errors="replace")
            if "_active_subprocesses.add(" in text or (
                    "_active_subprocesses.discard(" in text and "tracked_start()" not in text):
                offenders.append(rel)
    assert not offenders, f"tracked children published without shell_process.tracked_start: {offenders}"


@pytest.fixture
def registry(monkeypatch, tmp_path):
    """This test's view of the one tracked registry; every child it started is reaped afterwards."""
    saved = set(shell_process._active_subprocesses)
    shell_process._active_subprocesses.clear()
    monkeypatch.setattr(shell_process, "_panic_requested", False)
    monkeypatch.setattr(shell_process, "_spawning_subprocesses", 0)
    monkeypatch.setattr(owned_shutdown, "_stop_budget_sec", lambda: 5.0)
    started, data = [], tmp_path / "data"
    data.mkdir()
    yield SimpleNamespace(started=started, data=data, errors=[])
    for proc in started:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
        # Settled only once its containment measures empty, never at the root's exit: a Windows venv python.exe is a
        # redirector whose interpreter child ends a moment after it (``_tree_remains``).
        _wait(lambda proc=proc: shell_process.retire_tracked(proc), why=lambda proc=proc: f"tree of {proc.pid} still runs")
    shell_process._active_subprocesses.clear()
    shell_process._active_subprocesses.update(saved)


def _wait(predicate, timeout=10.0, why=lambda: ""):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, f"condition not reached {why()}"
        time.sleep(0.01)


def _producer(reg, hold=None, cmd=SLEEP, scope=contextlib.nullcontext):
    """A minimal tracked producer: count, Popen (born contained, as every producer's child is), optionally pause
    before publishing, publish."""
    def run():
        try:
            with scope(), shell_process.tracked_start() as publish:
                proc = shell_process.contained_popen(subprocess.Popen, cmd, stdin=subprocess.DEVNULL,
                                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                                     **subprocess_new_group_kwargs())
                reg.started.append(proc)
                if hold is not None:
                    assert hold.wait(30), "the test never released the publication"
                publish(proc)
        except BaseException as exc:  # recorded for the assertion, never swallowed silently
            reg.errors.append(exc)

    thread = threading.Thread(target=run, name="tracked-producer", daemon=True)
    thread.start()
    return thread


def _stopped(proc):
    return wait_test_child_stop(proc.pid, timeout_sec=5)["child_stopped"]


def test_a_begun_stop_or_emergency_latch_refuses_a_start_before_its_popen(registry, monkeypatch):
    popen = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: popen.append(a))
    owned_shutdown.begin_owned_stop(registry.data)
    with pytest.raises(shell_process.CommandAdmissionClosed, match="stop has begun") as refused:
        with shell_process.tracked_start():
            subprocess.Popen(SLEEP)
    assert refused.value.process_not_started is True and shell_process._spawning_subprocesses == 0
    monkeypatch.setattr(owned_shutdown, "_GENERATION_STOP", owned_shutdown._Stop())
    monkeypatch.setattr(shell_process, "_panic_requested", True)
    with pytest.raises(shell_process.CommandAdmissionClosed, match="Emergency Stop"):
        with shell_process.tracked_start():
            subprocess.Popen(SLEEP)
    assert popen == [] and shell_process._spawning_subprocesses == 0


def test_the_stop_waits_for_a_start_in_flight_and_its_starter_ends_the_late_child(registry):
    hold = threading.Event()
    producer = _producer(registry, hold)
    _wait(lambda: registry.started and shell_process._spawning_subprocesses == 1)
    child = registry.started[0]
    owned_shutdown.begin_owned_stop(registry.data)
    outcome = {}
    stop = threading.Thread(target=lambda: outcome.update(owned_shutdown.stop_owned_work(registry.data)))
    stop.start()
    time.sleep(0.3)
    assert stop.is_alive(), "the stop waits for the publication its count still owes"
    assert child.poll() is None and child not in shell_process._active_subprocesses
    hold.set()
    stop.join(10)
    producer.join(10)
    assert not stop.is_alive() and not registry.errors
    assert _stopped(child), "published into a begun stop: its own starter ended it"
    assert outcome["state"] == "completed" and outcome["memory_unconfirmed"] == {}
    assert child not in shell_process._active_subprocesses and shell_process._spawning_subprocesses == 0


def test_a_start_still_inside_popen_at_the_deadline_keeps_the_outcome_unconfirmed(registry, monkeypatch):
    monkeypatch.setattr(owned_shutdown, "_stop_budget_sec", lambda: 0.4)
    hold = threading.Event()
    producer = _producer(registry, hold)
    _wait(lambda: registry.started and shell_process._spawning_subprocesses == 1)
    child = registry.started[0]
    report = shell_process.kill_all_tracked_subprocesses(deadline=time.monotonic() + 0.2)
    assert report == {"state": "unconfirmed", "pids": [], "unpublished": 1}
    outcome = owned_shutdown.stop_owned_work(registry.data)
    assert outcome["state"] == "unconfirmed" and outcome["unconfirmed"] == [], "no durable record names it"
    # Its owner waits until the same deadline, so at it the owner is unfinished or has its own report.
    assert outcome["memory_unconfirmed"]["commands"] in ({"state": "unfinished"}, report)
    assert child.poll() is None, "the finite deadline cannot reach an unpublished child"
    hold.set()
    producer.join(10)
    assert _stopped(child), "the late publication meets the stop and its starter ends the child"
    _wait(lambda: shell_process.unsettled_tracked_children() is None,
          why=lambda: f"its tree is still unsettled: {shell_process.unsettled_tracked_children()}")
    joined = owned_shutdown.stop_owned_work(registry.data)  # a later door joins; it never claims completion
    assert joined["state"] == "unconfirmed"


def test_a_publication_during_the_kill_loop_keeps_its_name_until_its_exit(registry, monkeypatch):
    """The witnessed erasure: a child published after the snapshot was dropped by a registry-wide clear."""
    early = shell_process.contained_popen(subprocess.Popen, SLEEP, **subprocess_new_group_kwargs())
    registry.started.append(early)
    shell_process._active_subprocesses.add(early)
    hold = threading.Event()
    producer = _producer(registry, hold)
    _wait(lambda: len(registry.started) == 2 and shell_process._spawning_subprocesses == 1)
    late = registry.started[1]
    end, entered = shell_process._end_tree, []

    def end_then_let_the_late_start_publish(proc):
        end(proc)  # the stop's per-child end: a POSIX group kill, or a Windows child's own Job
        if proc is early and not entered:
            entered.append(proc)
            hold.set()  # the late child is published while this loop still runs
            _wait(lambda: late in shell_process._active_subprocesses)

    monkeypatch.setattr(shell_process, "_end_tree", end_then_let_the_late_start_publish)
    outcome = owned_shutdown.stop_owned_work(registry.data)
    producer.join(10)
    assert entered == [early] and _stopped(early) and _stopped(late)
    assert outcome["state"] == "completed" and outcome["memory_unconfirmed"] == {}
    assert not shell_process._active_subprocesses, "only handles whose exit was observed left the registry"


def test_an_unconfirmed_kill_keeps_its_handle_and_says_so(registry, monkeypatch):
    monkeypatch.setattr(owned_shutdown, "_stop_budget_sec", lambda: 0.4)
    child = shell_process.contained_popen(subprocess.Popen, SLEEP, **subprocess_new_group_kwargs())
    registry.started.append(child)
    shell_process._active_subprocesses.add(child)
    monkeypatch.setattr(shell_process, "_end_tree", lambda proc: None)  # a request that ends nothing, POSIX group or Job
    report = shell_process.kill_all_tracked_subprocesses(deadline=time.monotonic() + 0.3)
    assert report == {"state": "unconfirmed", "pids": [child.pid], "unpublished": 0}
    assert child in shell_process._active_subprocesses and child.poll() is None, "a sent request is no exit proof"
    outcome = owned_shutdown.stop_owned_work(registry.data)
    # Its owner confirms until the same deadline, so at it the owner is unfinished or has its own report.
    assert outcome["state"] == "unconfirmed" and outcome["unconfirmed"] == []
    assert outcome["memory_unconfirmed"]["commands"] in ({"state": "unfinished"}, report)
    assert child in shell_process._active_subprocesses


def test_a_failed_spawn_settles_its_count_and_keeps_its_own_error(registry):
    with pytest.raises(FileNotFoundError):
        with shell_process.tracked_start() as publish:
            publish(shell_process.contained_popen(subprocess.Popen, [str(registry.data / "missing-binary")], **subprocess_new_group_kwargs()))
    assert shell_process._spawning_subprocesses == 0 and not shell_process._active_subprocesses


def test_panic_requests_never_wait_for_a_start_or_any_registry_lock(registry):
    child = shell_process.contained_popen(subprocess.Popen, SLEEP, **subprocess_new_group_kwargs())
    registry.started.append(child)
    shell_process._active_subprocesses.add(child)
    hold = threading.Event()
    _producer(registry, hold)  # a start held inside its Popen, never published here
    _wait(lambda: shell_process._spawning_subprocesses == 1)
    with shell_process._subprocess_lock, shell_process._start_settled:
        started = time.monotonic()
        receipts = shell_process.kill_all_tracked_subprocesses(request_only=True)
        assert time.monotonic() - started < 0.5
    assert [row["pid"] for row in receipts if row["requested"]] == [child.pid] and _stopped(child)
    hold.set()
    _wait(lambda: shell_process._spawning_subprocesses == 0)
    assert _stopped(registry.started[1]), "published after the latch: its starter requested the kill"


# Each producer of the tracked registry, called as its own consumers call it, with the door already
# begun: each refuses before Popen and answers in its own contract.
def _refused_run_command(tmp_path):
    with pytest.raises(shell_process.CommandAdmissionClosed) as refused:
        shell_process._tracked_subprocess_run(SLEEP, timeout=5)
    assert refused.value.process_not_started is True


def _refused_skill(tmp_path):
    with pytest.raises(shell_process.CommandAdmissionClosed):
        skill_exec._run_skill_subprocess(SLEEP, cwd=str(tmp_path), env=dict(os.environ), timeout_sec=5,
                                         stdout_cap=1000, stderr_cap=1000)


def _refused_extension(tmp_path):
    runner = extension_process_runner
    with pytest.raises(shell_process.CommandAdmissionClosed) as refused:
        runner._run_child({"skill_name": "admission"}, skill_dir=tmp_path, drive_root=tmp_path,
                          repo_dir=tmp_path, env={}, timeout_sec=2)
    assert runner.extension_child_was_spawned(refused.value) is False, "no child: no physical dispatch"


def _refused_preflight(tmp_path):
    result = skill_preflight._run_check(SLEEP, tmp_path)
    assert (result["returncode"], result["timeout"], result["pre_exec_failure"]) == (None, False, "CommandAdmissionClosed")


def _refused_isolated_deps(tmp_path):
    with pytest.raises(shell_process.CommandAdmissionClosed):
        isolated_deps._run(SLEEP, cwd=tmp_path, env=dict(os.environ), timeout_sec=5)


def _refused_update_smoke(tmp_path):
    with pytest.raises(shell_process.CommandAdmissionClosed):
        update_merge._run_update_smoke(SLEEP)


def _refused_git(tmp_path):
    code, _out, err = git_ops._run_git_process_bounded(["git", "status"], timeout=5, cwd=tmp_path)
    assert code == 127 and "admission is closed" in err


def _refused_dependency_sync(tmp_path, monkeypatch):
    req = tmp_path / "requirements-runtime.lock"
    req.write_text("example==1\n", encoding="utf-8")
    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    monkeypatch.delenv("OUROBOROS_PYTEST_ACTIVE")
    monkeypatch.setattr("ouroboros.platform_layer.pip_install_target_args", lambda _exe: ["--target", str(tmp_path / "env")])
    ok, detail = git_ops_reset.sync_runtime_dependencies("test")
    assert not ok and "CommandAdmissionClosed" in detail


PRODUCERS = {
    "run_command": _refused_run_command, "skill_exec": _refused_skill, "extension": _refused_extension,
    "skill_preflight": _refused_preflight, "isolated_deps": _refused_isolated_deps,
    "update_smoke": _refused_update_smoke, "git": _refused_git,
    "dependency_sync": _refused_dependency_sync,
}


@pytest.mark.parametrize("producer", sorted(PRODUCERS))
def test_every_producer_refuses_after_the_door_in_its_own_contract(registry, monkeypatch, tmp_path, producer):
    popen = []

    def no_popen(*args, **kwargs):
        popen.append(args)
        raise AssertionError("no Popen after the door")

    for module in (subprocess, skill_exec, skill_preflight):  # the two module-level ``from subprocess import Popen``
        monkeypatch.setattr(module, "Popen", no_popen)
    owned_shutdown.begin_owned_stop(registry.data)
    run = PRODUCERS[producer]
    run(tmp_path, monkeypatch) if run.__code__.co_argcount == 2 else run(tmp_path)
    assert popen == [] and shell_process._spawning_subprocesses == 0


def test_the_remaining_producers_publish_only_through_the_shared_start():
    """Source shape: each producer function enters ``tracked_start`` and none adds to the set itself."""
    import ast
    import inspect
    import textwrap

    producers = (shell_process._tracked_subprocess_run, skill_exec._run_skill_subprocess,
                 extension_process_runner._child_process.__wrapped__, skill_preflight._run_check, isolated_deps._run,
                 update_merge._run_update_smoke,
                 git_ops._run_git_process_bounded, git_ops_reset.sync_runtime_dependencies)
    for function in producers:
        source = textwrap.dedent(inspect.getsource(function))
        for seam in ("tracked_start", "contained_popen", "retire_tracked"):
            assert calls_function(ast.parse(source), seam), (function.__qualname__, seam)
        assert "_active_subprocesses.discard(" not in source
        assert "_active_subprocesses.add(" not in source, function.__qualname__


@pytest.mark.parametrize("remains", [False, True, None], ids=["empty", "live", "unknown"])
@pytest.mark.parametrize("producer", [*sorted(PRODUCERS), "skill_callback_failure"])
def test_every_producer_retires_only_an_empty_tree(registry, monkeypatch, tmp_path, producer, remains):
    """Run each real owner/consumer around an explicit Popen+containment double."""
    import io
    from ouroboros import tool_custody

    confirmed = []
    monkeypatch.setattr(tool_custody, "external_handoff", lambda kind: lambda: confirmed.append(kind))

    class Child:
        pid = 765432
        returncode = 0
        args = ["test-owned-command"]
        stdin = None
        stdout = io.BytesIO(b"output")
        stderr = io.BytesIO()
        def poll(self):
            return self.returncode
        def wait(self, timeout=None):
            return self.returncode
        def communicate(self, **kwargs):
            return ("output", "") if producer in ("run_command", "git", "update_smoke") else (b"output", b"")

    child = Child()
    state = {"remains": remains}
    read = shell_process._tree_remains
    monkeypatch.setattr(shell_process, "_tree_remains", lambda p, **kw: state["remains"] if p is child else read(p, **kw))
    # Patch the birth seam only. No global Popen replacement can intercept a census or unrelated subprocess.
    for module in (shell_process, skill_exec, skill_preflight, extension_process_runner):
        monkeypatch.setattr(module, "contained_popen", lambda *a, **kw: child)
    monkeypatch.setattr(skill_exec, "_kill_process_group", lambda p: None)
    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", registry.data)
    try:
        if producer == "run_command":
            result = shell_process._tracked_subprocess_run(["test"], text=True)
            assert (result.returncode, result.stdout) == (0, "output")
        elif producer in ("skill_exec", "skill_callback_failure"):
            def failed():
                raise RuntimeError("callback failed")
            kwargs = dict(cwd=str(tmp_path), env={}, timeout_sec=5, stdout_cap=4096, stderr_cap=4096)
            if producer == "skill_callback_failure":
                with pytest.raises(RuntimeError, match="callback failed"):
                    skill_exec._run_skill_subprocess(["test"], on_spawn=failed, **kwargs)
            else:
                assert skill_exec._run_skill_subprocess(["test"], **kwargs)[0] == 0
        elif producer == "extension":
            with extension_process_runner._child_process({"skill_name": "retirement"}, skill_dir=tmp_path,
                    drive_root=registry.data, repo_dir=tmp_path, env={}) as handle:
                assert handle.proc is child
        elif producer == "skill_preflight":
            assert skill_preflight._run_check(["test"], tmp_path)["returncode"] == 0
        elif producer == "isolated_deps":
            assert isolated_deps._run(["test"], cwd=tmp_path, env={}, timeout_sec=5)["returncode"] == 0
        elif producer == "update_smoke":
            assert update_merge._run_update_smoke(["test"])["ok"]
        elif producer == "git":
            assert git_ops._run_git_process_bounded(["test"], timeout=5, cwd=tmp_path) == (0, "output", "")
        else:
            req = tmp_path / "requirements-runtime.lock"
            req.write_text("test==1\n")
            monkeypatch.delenv("OUROBOROS_PYTEST_ACTIVE")
            assert git_ops_reset.sync_runtime_dependencies("retirement")[0]
        assert (child in shell_process._active_subprocesses) is (remains is not False)
        if producer in ("run_command", "skill_exec", "skill_callback_failure", "extension"):
            assert bool(confirmed), "root receipts settle independently of retained tree custody"
        if remains is not False:
            from ouroboros.tool_custody import local_producers_settled
            assert not local_producers_settled(), "worker retirement must keep its process/Job handles"
        state["remains"] = False
        assert shell_process.retire_tracked(child)
        if producer in ("run_command", "skill_exec", "skill_callback_failure", "extension"):
            assert len(confirmed) == 1, "settlement discharges the original receipt once"
    finally:
        state["remains"] = False
        assert shell_process.retire_tracked(child)


def _bound(data, name, target):
    """Run ``target`` inside a real tool-operation binding (as a task's tool call carries it)."""
    from ouroboros import owner_pause

    source = SimpleNamespace(task_id="ob1656-task", drive_root=str(data), task_metadata={}, task_lifecycle_bound=False)
    with owner_pause.tool_handoff(source, name):
        return target()


@pytest.mark.skipif(os.name == "nt", reason="native POSIX witness; Windows Job branches are simulated separately")
@pytest.mark.parametrize("producer", sorted(PRODUCERS))
def test_every_real_producer_preserves_a_redirected_background_child(registry, monkeypatch, tmp_path, producer):
    from ouroboros.tools import shell as shell_facade

    pid_file = tmp_path / "child.pid"
    script = ("import pathlib,subprocess,sys; "
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],"
        "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid)); "
        "print('output',flush=True)")
    born = []
    original = shell_process.contained_popen

    def spawn(popen_cls, _argv, **kwargs):
        proc = original(popen_cls, [sys.executable, "-c", script], **kwargs)
        born.append(proc)
        return proc

    for module in (shell_process, skill_exec, skill_preflight, extension_process_runner):
        monkeypatch.setattr(module, "contained_popen", spawn)
    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", registry.data)
    try:
        if producer == "run_command":
            assert shell_process._tracked_subprocess_run(["test"], timeout=5).returncode == 0
        elif producer == "skill_exec":
            assert skill_exec._run_skill_subprocess(["test"], cwd=str(tmp_path), env=dict(os.environ),
                timeout_sec=5, stdout_cap=4096, stderr_cap=4096)[0] == 0
        elif producer == "extension":
            with extension_process_runner._child_process({"skill_name": "retirement"}, skill_dir=tmp_path,
                    drive_root=registry.data, repo_dir=tmp_path, env=dict(os.environ)) as handle:
                assert handle.proc.wait(timeout=5) == 0
        elif producer == "skill_preflight":
            assert skill_preflight._run_check(["test"], tmp_path)["returncode"] == 0
        elif producer == "isolated_deps":
            assert isolated_deps._run(["test"], cwd=tmp_path, env=dict(os.environ), timeout_sec=5)["returncode"] == 0
        elif producer == "update_smoke":
            assert update_merge._run_update_smoke(["test"])["ok"]
        elif producer == "git":
            assert git_ops._run_git_process_bounded(["test"], timeout=5, cwd=tmp_path)[0] == 0
        else:
            (tmp_path / "requirements-runtime.lock").write_text("test==1\n")
            monkeypatch.delenv("OUROBOROS_PYTEST_ACTIVE")
            assert git_ops_reset.sync_runtime_dependencies("retirement")[0]
        [proc] = born
        child = int(pid_file.read_text())
        assert proc.returncode == 0 and proc in shell_process._active_subprocesses
        assert shell_process._tree_remains(proc) is True
        assert not wait_test_child_stop(child, timeout_sec=0.1)["child_stopped"]
        # The public shell facade and ordinary stop consume the same retained owner.
        assert proc in shell_facade._active_subprocesses
        assert shell_process.kill_all_tracked_subprocesses(deadline=time.monotonic() + 5) is None
        assert wait_test_child_stop(child, timeout_sec=5)["child_stopped"]
    finally:
        for proc in born:
            shell_process._end_tree(proc)  # only the waitable, identity-matched child/group
            proc.wait(timeout=5)
            assert shell_process.retire_tracked(proc)


def _held_popen(monkeypatch, marker, hold, seen):
    """Hold Popen's return for the one marked command: the child runs, unpublished, until released."""
    init = subprocess.Popen.__init__

    def held(popen, *args, **kwargs):
        init(popen, *args, **kwargs)
        argv = args[0] if args else kwargs.get("args")
        if isinstance(argv, (list, tuple)) and marker in argv:
            seen.append(popen)
            assert hold.wait(30), "the test never released the publication"

    monkeypatch.setattr(subprocess.Popen, "__init__", held)


@pytest.mark.parametrize("producer", ["run_command", "skill_exec", "extension"])
def test_a_real_producer_published_into_the_stop_is_ended_by_its_starter(registry, monkeypatch, producer):
    runner = extension_process_runner
    marker = f"ob1656-{producer}"
    cmd = [sys.executable, "-c", "import time; time.sleep(60)", marker]
    hold, seen, results = threading.Event(), [], []
    _held_popen(monkeypatch, marker, hold, seen)
    if producer == "extension":  # only the runner's own child argv becomes the sleeper; its owner code is real
        popen = subprocess.Popen

        def runner_child(argv, **kwargs):  # the stdlib class through the producers' own door, born contained
            return (shell_process.contained_popen(popen, cmd, **kwargs)
                    if "ouroboros.extension_process_runner" in argv else popen(argv, **kwargs))

        monkeypatch.setattr(runner.subprocess, "Popen", runner_child)
    data = registry.data

    def produce():
        try:
            if producer == "run_command":
                results.append(_bound(data, "run_command", lambda: shell_process._tracked_subprocess_run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)))
            elif producer == "skill_exec":
                results.append(_bound(data, "run_skill", lambda: skill_exec._run_skill_subprocess(
                    cmd, cwd=str(data.parent), env=dict(os.environ), timeout_sec=30, stdout_cap=1000, stderr_cap=1000)))
            else:
                results.append(runner._run_child({"skill_name": "ob1656"}, skill_dir=data.parent, drive_root=data,
                                                 repo_dir=data.parent, env=dict(os.environ), timeout_sec=30))
        except BaseException as exc:
            results.append(exc)

    thread = threading.Thread(target=produce, name=f"real-{producer}", daemon=True)
    thread.start()
    _wait(lambda: seen and shell_process._spawning_subprocesses == 1, why=lambda: (results, seen))
    child = seen[0]
    registry.started.append(child)
    owned_shutdown.begin_owned_stop(data)
    outcome = {}
    stop = threading.Thread(target=lambda: outcome.update(owned_shutdown.stop_owned_work(data)))
    stop.start()
    time.sleep(0.3)
    assert stop.is_alive() and child.poll() is None
    hold.set()
    stop.join(15)
    thread.join(15)
    assert not stop.is_alive() and not thread.is_alive()
    assert _stopped(child), "published into the begun stop: the producer's own start ended it"
    assert outcome["memory_unconfirmed"].get("commands") is None, "the stop confirmed the late child's exit"
    if producer == "extension":
        assert isinstance(results[0], BaseException) and runner.extension_child_was_spawned(results[0]) is True
    joined = owned_shutdown.stop_owned_work(data)  # after the producer settled: nothing in memory is owed
    assert joined.get("memory_unconfirmed", {}).get("commands") is None


# The restart exit (``server._check_restart``) takes the stop's census, then ``store_switch.switch_at_restart``
# exports, retires the stores and consumes the intent before the door owner's rescue and reset Git start.
def _restart_exit_census(data):
    """What the restart watcher hands the seam: the generation's one stop, begun and finished first."""
    owned_shutdown.begin_owned_stop(data, cause="restart_requested")
    return {"worker_exits": {"unconfirmed": []}, "live_children": [], "owned_stop": owned_shutdown.stop_owned_work(data)}


def _deferred_restore(monkeypatch, root, repo, old):
    """The real Restore door over isolated roots defers a store-less target; the owner's edit awaits rescue."""
    from ouroboros import store_switch

    for name, value in {"REPO_DIR": repo, "DRIVE_ROOT": root}.items():
        monkeypatch.setattr(git_ops, name, value)
    monkeypatch.setattr(git_ops, "update_state", lambda mutate, **_kw: mutate({}))  # the checkout projection
    (repo / "ouroboros" / "task_results.py").write_text("# owner edit\n", encoding="utf-8")
    ok, _message, deferred = store_switch.switch_or_reset(
        root, repo, target_sha=old, target_ref="v-old", reason="ui_rollback", door="ui_rollback",
        reset=lambda: pytest.fail("a store-less target moves at the restart exit"))
    assert ok and deferred
    census = _restart_exit_census(root)
    assert census["owned_stop"]["state"] == "completed" and owned_shutdown.stop_begun()
    return census


def test_a_deferred_restore_moves_the_checkout_at_the_restart_exit_after_its_stop(registry, switch_root, target_repo,
                                                                                 monkeypatch):
    from ouroboros import store, store_switch

    root, (repo, old, new) = switch_root, target_repo
    outcome = store_switch.switch_at_restart(root, repo, _deferred_restore(monkeypatch, root, repo, old))
    assert outcome["status"] == "switched", outcome
    assert git(repo, "rev-parse", "HEAD") == old and store.probe(root)[0] == "absent"
    [keep] = git(repo, "branch", "--list", "--format=%(refname:short)", "rollback-keep-*").split()
    assert git(repo, "rev-parse", keep) == new, "the previous HEAD is kept"
    [rescue] = (root / "archive" / "rescue").iterdir()
    assert "+# owner edit" in (rescue / "changes.diff").read_text(encoding="utf-8"), "the bounded rescue Git ran"
    assert shell_process.unsettled_tracked_children() is None and not shell_process._active_subprocesses
    with pytest.raises(shell_process.CommandAdmissionClosed, match="stop has begun"):
        with shell_process.tracked_start():  # this thread's admission is closed again after the seam
            pytest.fail("no start after the restart exit's finalization")


def test_panic_during_the_restart_exit_restore_never_waits_and_moves_nothing(registry, switch_root, target_repo,
                                                                            monkeypatch):
    from ouroboros import store_switch

    root, (repo, old, new) = switch_root, target_repo
    census = _deferred_restore(monkeypatch, root, repo, old)
    hold, seen, outcome = threading.Event(), [], {}
    _held_popen(monkeypatch, "--binary", hold, seen)  # the rescue diff: admitted, running, not yet published
    exit_thread = threading.Thread(target=lambda: outcome.update(store_switch.switch_at_restart(root, repo, census)))
    exit_thread.start()
    _wait(lambda: seen and shell_process._spawning_subprocesses == 1, timeout=60)
    with shell_process._subprocess_lock, shell_process._start_settled:
        started = time.monotonic()
        shell_process.kill_all_tracked_subprocesses(request_only=True)
        assert time.monotonic() - started < 0.5, "Panic never waits for the exit's finalization"
    hold.set()
    exit_thread.join(60)
    assert not exit_thread.is_alive() and outcome["status"] == "switch_failed", outcome
    assert git(repo, "rev-parse", "HEAD") == new, "Panic is never bypassed: the reset did not run"
    assert shell_process._spawning_subprocesses == 0 and shell_process.unsettled_tracked_children() is None


def test_panic_after_the_restart_exit_rescue_refuses_the_final_reset(registry, switch_root, target_repo, monkeypatch):
    """Panic lands after Restore's rescue completed and before its final reset, whose ``git_capture`` waits
    without a bound: that reset is refused before its Popen and the checkout stays (the R4 change-coupling
    reviewer's reproduction, rebuilt from its description)."""
    from ouroboros import store_switch

    root, (repo, old, new) = switch_root, target_repo
    census = _deferred_restore(monkeypatch, root, repo, old)
    rescue, argvs = git_ops._create_rescue_snapshot, []

    def rescue_then_panic(*args, **kwargs):
        rescued = rescue(*args, **kwargs)
        shell_process.kill_all_tracked_subprocesses(request_only=True)
        return rescued

    init = subprocess.Popen.__init__

    def recorded(popen, *args, **kwargs):
        argvs.append(list(args[0] if args else kwargs.get("args")))
        init(popen, *args, **kwargs)

    monkeypatch.setattr(git_ops, "_create_rescue_snapshot", rescue_then_panic)
    monkeypatch.setattr(subprocess.Popen, "__init__", recorded)
    outcome = store_switch.switch_at_restart(root, repo, census)
    [snapshot] = (root / "archive" / "rescue").iterdir()
    assert "+# owner edit" in (snapshot / "changes.diff").read_text(encoding="utf-8"), "Panic came after the rescue"
    assert outcome["status"] == "switch_failed" and "Emergency Stop" in outcome["message"], outcome
    assert not [argv for argv in argvs if argv[:3] == ["git", "reset", "--hard"]], "no reset started after Panic"
    assert git(repo, "rev-parse", "HEAD") == new, "Panic is never bypassed: the checkout stays"
    assert shell_process._spawning_subprocesses == 0 and shell_process.unsettled_tracked_children() is None


def test_panic_never_waits_for_an_unbounded_git_start_and_its_starter_ends_it(registry, monkeypatch, tmp_path):
    """``git_capture`` without a timeout still starts through ``tracked_start``: Panic while that Git is
    admitted but inside its Popen takes no lock or wait, and the late publication hands the tree to the kill. The
    alias's shell already runs when Panic comes, so only a kill of the whole tree releases the capture's pipes.

    UNRESOLVED on Windows, a product failure rather than a fixture's (CI 41634598146, job 125912568384): without a Job
    the request ends the published root alone, its settle pass addresses a root already gone, and the capture waits."""
    marker = "ob-late-panic-git"
    hold, seen, results = threading.Event(), [], []
    _held_popen(monkeypatch, marker, hold, seen)
    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)
    capture = threading.Thread(target=lambda: results.append(git_ops.git_capture(
        ["git", "-c", f"alias.{marker}=!touch tree.ready; sleep 60", marker])), name="unbounded-git", daemon=True)
    capture.start()
    try:
        _wait(lambda: seen)
        _wait(lambda: (tmp_path / "tree.ready").exists(), timeout=120.0)  # Git's own start, not the bound under test
        child = seen[0]
        registry.started.append(child)
        assert shell_process._spawning_subprocesses == 1, "the unbounded Git start is admitted and counted"
        with shell_process._subprocess_lock, shell_process._start_settled:
            started = time.monotonic()
            receipts = shell_process.kill_all_tracked_subprocesses(request_only=True)
            assert time.monotonic() - started < 0.5, "Panic never waits for an unbounded Git start"
        assert receipts == [] and child.poll() is None, "not yet published: Panic had nothing of it to name"
    finally:
        hold.set()
    capture.join(30)
    assert not capture.is_alive() and results[0][0] != 0, "the published Git tree was ended, not waited out"
    assert _stopped(child) and child not in shell_process._active_subprocesses
    assert shell_process._spawning_subprocesses == 0


def _git_version(cwd):
    return git_ops._run_git_process_bounded(["git", "--version"], timeout=30, cwd=cwd)


def test_only_the_finalizing_thread_starts_after_the_stop_and_its_child_stays_named(registry, tmp_path):
    _restart_exit_census(registry.data)
    refused = []
    with shell_process.exit_finalization():
        with shell_process.tracked_start() as publish:  # admitted and published, never handed to the stop
            child = publish(shell_process.contained_popen(subprocess.Popen, SLEEP, **subprocess_new_group_kwargs()))
            registry.started.append(child)
        assert child.poll() is None and child in shell_process._active_subprocesses
        code, out, _err = _git_version(tmp_path)  # the rescue's own bounded runner settles and forgets its child
        assert code == 0 and out.startswith("git version")
        other = threading.Thread(target=lambda: refused.append(_git_version(tmp_path)))
        other.start()
        other.join(10)
    assert refused[0][0] == 127 and "admission is closed" in refused[0][2], "another thread's start stays refused"
    with pytest.raises(shell_process.CommandAdmissionClosed, match="stop has begun"):
        with shell_process.tracked_start():
            pytest.fail("this thread is refused again after the block")
    assert shell_process._spawning_subprocesses == 0 and shell_process._active_subprocesses == {child}


def test_a_failed_finalizing_start_settles_and_names_nothing(registry, tmp_path):
    _restart_exit_census(registry.data)
    with shell_process.exit_finalization():
        with pytest.raises(FileNotFoundError):
            with shell_process.tracked_start() as publish:
                publish(shell_process.contained_popen(subprocess.Popen, [str(tmp_path / "missing-binary")], **subprocess_new_group_kwargs()))
        code, _out, err = git_ops._run_git_process_bounded(["git", "status"], timeout=30, cwd=tmp_path / "gone")
    assert code == 127 and err, "no Git process started"
    assert shell_process._spawning_subprocesses == 0 and not shell_process._active_subprocesses
    assert shell_process.unsettled_tracked_children() is None


def test_panic_refuses_a_finalizing_start_before_its_popen(registry, monkeypatch, tmp_path):
    _restart_exit_census(registry.data)
    shell_process.kill_all_tracked_subprocesses(request_only=True)
    popen = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: popen.append(a))
    with shell_process.exit_finalization():
        with pytest.raises(shell_process.CommandAdmissionClosed, match="Emergency Stop"):
            with shell_process.tracked_start():
                subprocess.Popen(SLEEP)
        code, _out, err = git_ops._run_git_process_bounded(["git", "status"], timeout=30, cwd=tmp_path)
    assert code == 127 and "Emergency Stop" in err
    assert popen == [] and shell_process._spawning_subprocesses == 0


def test_panic_never_waits_for_a_finalizing_start_and_ends_what_it_published(registry):
    _restart_exit_census(registry.data)
    with shell_process.exit_finalization(), shell_process.tracked_start() as publish:
        running = publish(shell_process.contained_popen(subprocess.Popen, SLEEP, **subprocess_new_group_kwargs()))
        registry.started.append(running)
    hold = threading.Event()
    _producer(registry, hold, scope=shell_process.exit_finalization)  # admitted, held inside its Popen
    _wait(lambda: len(registry.started) == 2 and shell_process._spawning_subprocesses == 1)
    with shell_process._subprocess_lock, shell_process._start_settled:
        started = time.monotonic()
        receipts = shell_process.kill_all_tracked_subprocesses(request_only=True)
        assert time.monotonic() - started < 0.5
    assert [row["pid"] for row in receipts if row["requested"]] == [running.pid] and _stopped(running)
    hold.set()
    _wait(lambda: shell_process._spawning_subprocesses == 0)
    assert _stopped(registry.started[1]), "published after the latch: its finalizing starter requested the kill"
    assert not registry.errors
