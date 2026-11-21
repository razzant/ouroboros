import contextlib
import hmac
import itertools
import json
import logging
import pathlib
import shutil
import subprocess
import sys
import threading
import time
import types

import pytest

import ouroboros.gateway.host_service as host_service

from ouroboros.contracts.plugin_api import ExtensionRegistrationError
from ouroboros.extension_companion import (
    CompanionDescriptor,
    companion_spawn_env,
    CompanionSupervisor,
    init_server_process_pid,
)
from ouroboros.extension_loader import PluginAPIImpl, _PluginAPIConfig
from ouroboros.extension_plugin_api import ExtensionStaleRecoveryError
import ouroboros.extension_loader as extension_loader
import ouroboros.extension_plugin_api as extension_plugin_api
from ouroboros import extension_health
from ouroboros.skill_loader import SkillReviewState, find_skill, save_enabled, save_review_state


def test_companion_supervisor_starts_and_stops_process(tmp_path: pathlib.Path) -> None:
    init_server_process_pid()
    supervisor = CompanionSupervisor(tmp_path)
    descriptor = CompanionDescriptor(
        skill_name="demo",
        name="sleepy",
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        env={},
    )

    assert supervisor.start(descriptor)
    snapshot = supervisor.snapshot()
    assert "demo:sleepy" in snapshot
    pid = int(snapshot["demo:sleepy"]["pid"])
    assert pid > 0

    supervisor.stop("demo", "sleepy", timeout_sec=1)
    assert supervisor.snapshot() == {}


def test_panic_kill_all_clears_runtime_table(tmp_path: pathlib.Path) -> None:
    init_server_process_pid()
    supervisor = CompanionSupervisor(tmp_path)
    descriptor = CompanionDescriptor(
        skill_name="demo",
        name="panic",
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        env={},
    )

    assert supervisor.start(descriptor)
    assert supervisor.snapshot()
    supervisor.panic_kill_all()
    assert supervisor.snapshot() == {}


def test_exhausted_companion_remains_in_durable_runtime_health(tmp_path: pathlib.Path, monkeypatch) -> None:
    init_server_process_pid()
    drive_root = tmp_path / "drive"
    repo_root = tmp_path / "skills"
    skill_dir = repo_root / "connector"
    drive_root.mkdir()
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: connector\ndescription: connector\nversion: 0.1.0\n"
        "type: extension\nentry: plugin.py\npermissions: []\n---\nbody\n",
        encoding="utf-8",
    )
    (skill_dir / "plugin.py").write_text("def register(api):\n    pass\n", encoding="utf-8")
    loaded = find_skill(drive_root, "connector", repo_path=str(repo_root))
    assert loaded is not None
    save_enabled(drive_root, loaded.name, True)
    save_review_state(
        drive_root,
        loaded.name,
        SkillReviewState(status="pass", content_hash=loaded.content_hash),
    )
    loaded = find_skill(drive_root, loaded.name, repo_path=str(repo_root))
    assert loaded is not None
    # Hold the monitor's terminal health writer at entry: until it writes, the exited
    # companion must stay visible. Then join that monitor, whose real writer may wait
    # seconds for its lock (#985); 60 s is only a hang bound.
    real_writer = extension_health.record_companion_restart_exhausted
    writer_entered, release_writer, monitors = threading.Event(), threading.Event(), []

    def held_writer(*args, **kwargs):
        monitors.append(threading.current_thread())
        writer_entered.set()
        release_writer.wait()
        return real_writer(*args, **kwargs)

    monkeypatch.setattr(extension_health, "record_companion_restart_exhausted", held_writer)
    try:
        assert extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root) is None
        with extension_loader._lock:  # the loaded publication registers the companion this test starts by hand
            extension_loader._record_companion_name(extension_loader._extensions[loaded.name], "bridge")
        supervisor = CompanionSupervisor(drive_root)
        descriptor = CompanionDescriptor(
            skill_name=loaded.name,
            name="bridge",
            command=[sys.executable, "-c", "raise SystemExit(7)"],
            cwd=tmp_path,
            env={},
            max_restarts=0,
        )

        assert supervisor.start(descriptor)
        assert writer_entered.wait(60), "the monitor never reached the terminal health writer"
        assert supervisor.snapshot()["connector:bridge"]["returncode"] == 7
        release_writer.set()
        monitors[0].join(60)
        assert not monitors[0].is_alive(), "the companion monitor did not finish its terminal lifecycle"

        assert supervisor.snapshot() == {}
        health = extension_health.read_extension_health(drive_root, loaded.name) or {}
        observed = health.get("last_observed") or {}
        assert observed["status"] == extension_health.BROKEN
        assert observed["reason"] == "companion_restart_exhausted:bridge"
        state = extension_loader.runtime_state_for_loaded_skill(
            loaded, drive_root, skills=[loaded],
        )
        assert state["live_loaded"] is True
        assert state["companion_failed"] is True
        assert state["reason"] == "companion_restart_exhausted:bridge"
        assert "exhausting its restart budget" in state["load_error"]
        assert extension_health.status_for_runtime_state(state) == extension_health.BROKEN
    finally:
        release_writer.set()
        extension_loader.unload_extension(loaded.name)


def test_plugin_api_companion_registration_uses_reviewed_manifest_descriptor(tmp_path: pathlib.Path) -> None:
    init_server_process_pid(999999)
    state_dir = tmp_path / "state"
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    api = PluginAPIImpl(_PluginAPIConfig(
        skill_name="demo",
        permissions=["companion_process"],
        env_allowlist=[],
        state_dir=state_dir,
        settings_reader=lambda: {},
        companion_processes=[{
            "name": "daemon",
            "command": ["python3", "scripts/daemon.py"],
            "runtime": "python3",
        }],
        skill_dir=skill_dir,
    ))

    api.register_companion_process("daemon")

    api._close_registration()
    init_server_process_pid()


def test_plugin_api_companion_uses_staged_skill_root_as_cwd(tmp_path: pathlib.Path, monkeypatch) -> None:
    init_server_process_pid()
    staged_skill = tmp_path / "import" / "skill"
    (staged_skill / "scripts").mkdir(parents=True)
    (staged_skill / "scripts" / "daemon.py").write_text("print('ok')\n", encoding="utf-8")
    captured = {}

    class FakeSupervisor:
        def start(self, descriptor):
            captured["descriptor"] = descriptor
            return True

        def stop(self, skill_name, name):
            return True

    # The companion supervisor is read by PluginAPIImpl (its owner) at register
    # time and by the loader's spawn/ensure paths; patch both readers.
    monkeypatch.setattr(extension_plugin_api, "get_global_supervisor", lambda: FakeSupervisor())
    monkeypatch.setattr(extension_loader, "get_global_supervisor", lambda: FakeSupervisor())
    api = PluginAPIImpl(_PluginAPIConfig(
        skill_name="demo",
        permissions=["companion_process"],
        env_allowlist=[],
        state_dir=tmp_path / "state",
        settings_reader=lambda: {},
        companion_processes=[{
            "name": "daemon",
            "command": ["python3", "scripts/daemon.py"],
            "runtime": "python3",
            "env": {"HOST_SERVICE_URL": "https://evil.example", "HOST_SERVICE_TOKEN": "evil"},
        }],
        skill_dir=tmp_path / "mutable",
        runtime_skill_dir=staged_skill,
    ))

    api.register_companion_process("daemon")
    # ABI-9: the spawn is a deferred side effect; publication starts it.
    api._publish_registrations()

    descriptor = captured["descriptor"]
    assert descriptor.cwd == staged_skill
    assert (descriptor.cwd / "scripts" / "daemon.py").is_file()
    assert descriptor.env["HOST_SERVICE_URL"].startswith("http://127.0.0.1:")
    assert descriptor.env["HOST_SERVICE_TOKEN"] != "evil"


def _companion_frontmatter(names=("daemon",)) -> str:
    return "companion_processes:\n" + "".join(
        f"  - name: {name}\n    runtime: python3\n    command: [\"python3\", \"scripts/{name}.py\"]\n"
        for name in names)


class _RecordingSupervisor:
    """Fake supervisor: records spawns/stops; reports no running companions."""

    def __init__(self) -> None:
        self.started: list = []
        self.stopped: list = []

    def start(self, descriptor):
        self.started.append(descriptor)
        return True

    def companion_ownership(self, skill_name, names):
        return list(names), {}

    def stop(self, *args, **kwargs):
        self.stopped.append(args)

    def stop_skill(self, skill_name):
        self.stopped.append((skill_name,))


def _reviewed_companion_skill(
    tmp_path: pathlib.Path, name: str = "compskill", extra_frontmatter: str = "", companions=("daemon",),
):
    """A reviewed+enabled companion extension the recovery tests can load."""
    repo_root = tmp_path / "skills"
    drive_root = tmp_path / "drive"
    drive_root.mkdir(exist_ok=True)
    skill_dir = repo_root / name
    (skill_dir / "scripts").mkdir(parents=True)
    for companion in companions:
        (skill_dir / "scripts" / f"{companion}.py").write_text("print('ok')\n", encoding="utf-8")
    (skill_dir / "plugin.py").write_text(
        "def register(api):\n" + "".join(f"    api.register_companion_process({companion!r})\n"
                                         for companion in companions), encoding="utf-8"
    )
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: companion skill\n"
        "version: 0.1.0\n"
        "type: extension\n"
        "entry: plugin.py\n"
        "permissions: [companion_process]\n"
        f"{_companion_frontmatter(companions)}"
        f"{extra_frontmatter}"
        "---\n"
        "body\n",
        encoding="utf-8",
    )
    loaded = find_skill(drive_root, name, repo_path=str(repo_root))
    assert loaded is not None
    save_enabled(drive_root, loaded.name, True)
    save_review_state(
        drive_root, loaded.name,
        SkillReviewState(status="pass", content_hash=loaded.content_hash),
    )
    loaded = find_skill(drive_root, loaded.name, repo_path=str(repo_root))
    assert loaded is not None
    return loaded, repo_root, drive_root


def _patch_supervisor(monkeypatch) -> _RecordingSupervisor:
    # The companion supervisor is read by PluginAPIImpl (its owner) at register
    # time and by the loader's spawn/ensure/unload paths; patch both readers.
    fake = _RecordingSupervisor()
    monkeypatch.setattr(extension_plugin_api, "get_global_supervisor", lambda: fake)
    monkeypatch.setattr(extension_loader, "get_global_supervisor", lambda: fake)
    return fake


_FAKE_COMPANION_PIDS = itertools.count(2_000_000_001)  # beyond any real pid: nothing can be signalled by one


class _FakeCompanionProcess:
    """A ``Popen`` stand-in under the real supervisor: it ends only when the test ends it."""

    def __init__(self, argv):
        self.args, self.pid, self.stdout, self.stderr = list(argv), next(_FAKE_COMPANION_PIDS), None, None
        self.returncode, self._ended, self.job = None, threading.Event(), None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self._ended.wait(timeout)
        return self.returncode

    def end(self, code: int) -> None:
        self.returncode = code
        self._ended.set()


class _FakeSpawns(list):
    """The fakes the harness started, and the monitor threads its supervisor started for them."""

    def __init__(self):
        super().__init__()
        self.monitors: list = []


class _FakeJob:
    """A Windows Job Object stand-in: the one fake companion assigned to it, and how often it was closed."""

    def __init__(self):
        self.member, self.closed = None, 0


def _supervise_fake_companions(monkeypatch, drive_root: pathlib.Path):
    """The real supervisor (monitor, restart budget, health writes, stop) over fake processes: a stop ends the fake
    itself, no OS signal is ever sent, and no synthetic pid reaches the custody ledger or a platform Job: the Windows
    branch (a native Windows run, whose launcher variable ``tests/conftest.py`` strips) gets a fake Job instead."""
    from ouroboros import extension_companion, process_custody

    spawned = _FakeSpawns()

    def popen(argv, **_kwargs):
        spawned.append(_FakeCompanionProcess(argv))
        return spawned[-1]

    def stopped(proc, *_args, **_kwargs):
        assert isinstance(proc, _FakeCompanionProcess), "a stop reached something other than a fake companion"
        proc.end(-15)

    def signalled(*_args, **_kwargs):
        raise AssertionError("a fake companion supervisor reached an OS signal")

    def assigned(job, pid):
        member = next((proc for proc in spawned if proc.pid == pid), None)
        assert isinstance(job, _FakeJob) and job.member is None and member is not None, (
            "a Job assignment reached something other than a new fake Job and a fake companion")
        job.member, member.job = member, job
        return True

    def job_stopped(job, *_args):
        assert isinstance(job, _FakeJob), "a Job stop reached something other than a fake Job"
        if job.member.poll() is None:
            job.member.end(-15)
        return ""

    def job_closed(job):
        assert isinstance(job, _FakeJob), "a Job close reached something other than a fake Job"
        job.closed += 1
        return ""

    monkeypatch.setattr(extension_companion, "subprocess", types.SimpleNamespace(
        Popen=popen, DEVNULL=subprocess.DEVNULL, PIPE=subprocess.PIPE))
    for name in ("terminate_process_tree", "kill_process_tree"):
        monkeypatch.setattr(extension_companion, name, stopped)
    for name in ("request_process_tree_kill", "kill_process_on_port"):
        monkeypatch.setattr(extension_companion, name, signalled)
    for name, fake in (("create_kill_on_close_job", _FakeJob), ("assign_pid_to_job", assigned),
                       ("terminate_job", job_stopped), ("close_job", job_closed)):
        monkeypatch.setattr(extension_companion, name, fake)
    monkeypatch.setattr(process_custody, "record_process", lambda *_args, **_kwargs: None)
    for stamp in ("code_stamp", "fresh_code_stamp"):
        monkeypatch.setattr(extension_health, stamp, lambda: ("test", "test-sha"))
    supervisor = CompanionSupervisor(drive_root)

    def start_monitor(key, runtime):  # the supervisor's own monitor, its thread kept for teardown to join
        monitor = threading.Thread(target=supervisor._monitor_runtime, args=(key, runtime), daemon=True,
                                   name=f"fake-companion-monitor-{key}")
        spawned.monitors.append(monitor)
        monitor.start()

    supervisor._start_monitor = start_monitor
    monkeypatch.setattr(extension_companion, "_GLOBAL_SUPERVISOR", supervisor)
    return supervisor, spawned


def _until(predicate, what: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, what
        time.sleep(0.02)


def _exhaust(supervisor: CompanionSupervisor, key: str, code: int = 7) -> None:
    """End ``key``'s running process with its restart budget spent; return once its monitor settled it."""
    with supervisor._lock:
        runtime = supervisor._runtimes[key]
        supervisor._restart_history[key] = [time.monotonic()] * runtime.descriptor.max_restarts
    runtime.process.end(code)
    _until(lambda: key not in supervisor.snapshot(), f"{key}'s monitor never settled its exhaustion")


def _running(supervisor: CompanionSupervisor, key: str) -> bool:
    with supervisor._lock:
        runtime = supervisor._runtimes.get(key)
    return runtime is not None and not runtime.retiring and runtime.process.poll() is None


def _end_fake_companions(supervisor: CompanionSupervisor, spawned: list, *skills: str) -> None:
    for process in spawned:
        if process.poll() is None:
            process.end(0)  # a clean exit: nothing restarts it
    _until(lambda: not supervisor.snapshot(), "a fake companion's exit was never settled")
    # An empty snapshot can precede a monitor's last health write: its own threads finish before the next test.
    for monitor in getattr(spawned, "monitors", ()):
        monitor.join(30)
    assert not any(monitor.is_alive() for monitor in getattr(spawned, "monitors", ())), "a monitor never finished"
    for skill in skills:
        extension_loader.unload_extension(skill)
    init_server_process_pid()


def _listed_skill(drive_root, repo_root, name):
    from ouroboros.gateway.extensions import _build_extensions_index

    return next(row for row in _build_extensions_index(drive_root, str(repo_root))["skills"] if row["name"] == name)


def test_a_start_that_failed_never_closes_its_own_failure_when_the_clock_steps_back(tmp_path, monkeypatch) -> None:
    """The wall clock steps back between a companion's start and its death, so the failure's time precedes that
    start's. The failure names the start it ended: the listing still shows it, and only a later start closes it."""
    from ouroboros import extension_companion

    init_server_process_pid()
    loaded, repo_root, drive_root = _reviewed_companion_skill(tmp_path)
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive_root)
    clock = ["2026-11-20T10:00:05.000000+00:00"]
    monkeypatch.setattr(extension_companion, "utc_now_iso", lambda *_args, **_kwargs: clock[0])
    key = f"{loaded.name}:daemon"
    try:
        assert extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root, skills=[loaded]) is None
        clock[0] = "2026-11-20T09:59:58.000000+00:00"
        _exhaust(supervisor, key)
        failure = extension_health.read_extension_health(drive_root, loaded.name)["companion_failures"]["daemon"]
        row = _listed_skill(drive_root, repo_root, loaded.name)
        assert (failure["ts"], failure.get("started_at"), row["live_reason"]) == (
            clock[0], "2026-11-20T10:00:05.000000+00:00", "companion_restart_exhausted:daemon")
        clock[0] = "2026-11-20T09:59:59.000000+00:00"  # still behind the first start, yet a later one
        recovered = extension_loader.reconcile_extension(loaded.name, drive_root, lambda: {}, repo_path=str(repo_root))
        assert (recovered["companions"]["action"], recovered["reason"], _running(supervisor, key)) == (
            "started_missing", "ready", True)
        assert _listed_skill(drive_root, repo_root, loaded.name)["live_reason"] == "ready"
    finally:
        _end_fake_companions(supervisor, spawned, loaded.name)


def test_reconcile_answers_and_records_the_companion_it_just_recovered(tmp_path: pathlib.Path, monkeypatch) -> None:
    """A live extension's companion exhausted its restarts; the next reconcile starts it again. Its receipt and the
    durable health show the recovered extension, and a later reconcile that starts nothing keeps them so."""
    init_server_process_pid()
    loaded, repo_root, drive_root = _reviewed_companion_skill(tmp_path)
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive_root)
    key = f"{loaded.name}:daemon"
    try:
        assert extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root, skills=[loaded]) is None
        _exhaust(supervisor, key)
        assert extension_health.read_extension_health(drive_root, loaded.name)["status"] == extension_health.BROKEN

        def reconcile() -> dict:
            state = extension_loader.reconcile_extension(loaded.name, drive_root, lambda: {}, repo_path=str(repo_root))
            health = extension_health.read_extension_health(drive_root, loaded.name)
            return {"action": state["action"], "companions": state["companions"]["action"],
                    "running": _running(supervisor, key), "companion_failed": state.get("companion_failed"),
                    "reason": state["reason"], "load_error": state.get("load_error"),
                    "health": health["status"], "observed": health["last_observed"]["status"],
                    "regressed": health["regressed"]}

        recovered = {"action": "extension_already_live", "running": True, "companion_failed": None,
                     "reason": "ready", "load_error": None, "health": "live", "observed": "live", "regressed": False}
        assert reconcile() == {**recovered, "companions": "started_missing"}
        assert reconcile() == {**recovered, "companions": "already_running"}
    finally:
        _end_fake_companions(supervisor, spawned, loaded.name)


def test_restarting_one_exhausted_companion_keeps_the_other_s_failure(tmp_path: pathlib.Path, monkeypatch) -> None:
    """Both companions of one extension exhausted their restarts, alpha first; a refresh starts beta alone. Alpha
    still does not run, and its failure stays in the health and the runtime state until it starts too."""
    init_server_process_pid()
    loaded, repo_root, drive_root = _reviewed_companion_skill(tmp_path, companions=("alpha", "beta"))
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive_root)
    alpha, beta = f"{loaded.name}:alpha", f"{loaded.name}:beta"

    def observed() -> dict:
        state = extension_loader.runtime_state_for_loaded_skill(loaded, drive_root, skills=[loaded])
        return {"running": [_running(supervisor, alpha), _running(supervisor, beta)],
                "companion_failed": state.get("companion_failed"), "reason": state["reason"],
                "health": extension_health.read_extension_health(drive_root, loaded.name)["status"]}

    try:
        assert extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root, skills=[loaded]) is None
        _exhaust(supervisor, alpha)
        _exhaust(supervisor, beta)
        assert supervisor.refresh(loaded.name, "beta", lambda: {"DEMO": "changed"}, compare_keys=["DEMO"],
                                  still_valid=lambda: True) == "started"
        assert observed() == {"running": [False, True], "companion_failed": True,
                              "reason": "companion_restart_exhausted:alpha", "health": "broken"}
        recovered = extension_loader.reconcile_extension(loaded.name, drive_root, lambda: {}, repo_path=str(repo_root))
        assert recovered["companions"]["started"] == ["alpha"]
        assert observed() == {"running": [True, True], "companion_failed": None, "reason": "ready", "health": "live"}
    finally:
        _end_fake_companions(supervisor, spawned, loaded.name)


def _hold_terminal_writer(monkeypatch, *, after_write: bool = False):
    """Hold the first monitor's terminal health writer at its entry (or right after its write); later ones write."""
    real_writer = extension_health.record_companion_restart_exhausted
    entered, release, monitors = threading.Event(), threading.Event(), []

    def held_writer(*args, **kwargs):
        monitors.append(threading.current_thread())
        if len(monitors) > 1:
            return real_writer(*args, **kwargs)
        result = real_writer(*args, **kwargs) if after_write else None
        entered.set()
        assert release.wait(30), "the held monitor was never released"
        return result if after_write else real_writer(*args, **kwargs)

    monkeypatch.setattr(extension_health, "record_companion_restart_exhausted", held_writer)
    return entered, release, monitors


@pytest.mark.parametrize("after_write", [False, True], ids=["replaced_before_its_write", "replaced_after_its_write"])
def test_a_replaced_runtime_s_late_monitor_records_no_terminal_failure(
    tmp_path: pathlib.Path, monkeypatch, after_write: bool,
) -> None:
    """An exhausted runtime's monitor is held before (or right after) its terminal health write while a start
    replaces that runtime. In either order, the running replacement is not reported failed."""
    init_server_process_pid()
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive_root)
    descriptor = CompanionDescriptor(skill_name="demo", name="bridge", command=[sys.executable, "-c", "pass"],
                                     cwd=tmp_path, env={}, max_restarts=0)
    entered, release, monitors = _hold_terminal_writer(monkeypatch, after_write=after_write)
    try:
        assert supervisor.start(descriptor)
        exhausted = spawned[-1]
        exhausted.end(7)
        assert entered.wait(30), "the exhausted runtime's monitor never reached its terminal health write"
        assert supervisor.start(descriptor)  # the replacement a reload or a refresh starts meanwhile
        replacement = spawned[-1]
        release.set()
        monitors[0].join(30)
        assert not monitors[0].is_alive(), "the released monitor never finished"
        state = extension_health.apply_companion_failure_to_runtime_state(
            {"desired_live": True, "reason": "ready"}, drive_root, "demo")
        health = extension_health.read_extension_health(drive_root, "demo") or {}
        assert {"replacement_running": replacement is not exhausted and replacement.poll() is None,
                "companion_failed": state.get("companion_failed"), "broken": health.get("status") == "broken"} == {
            "replacement_running": True, "companion_failed": None, "broken": False}
        events = [("extension_companion_exhausted", 7, None)]
        if after_write:  # the replacement saw and closed the open failure
            events.append(("extension_companion_recovered", 7, "start"))
        assert _companion_events(drive_root) == events
    finally:
        release.set()
        _end_fake_companions(supervisor, spawned)


@contextlib.contextmanager
def _a_held_exhaustion_and_a_failed_replacement(tmp_path: pathlib.Path, monkeypatch, failure: str):
    """``demo:bridge`` (no restart budget) exits with code 7 and its monitor is held at its terminal health write;
    then a start meant to replace it fails: its spawn refused, or its cost disclosure after the spawn. Later starts
    spawn again. Yields the supervisor, the spawned fakes, the descriptor, the held monitor's release and threads."""
    from ouroboros import extension_companion

    init_server_process_pid()
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive_root)
    refuse = threading.Event()

    def disclose(*_args, **_kwargs):
        if refuse.is_set():
            raise RuntimeError("cost disclosure failed")

    def refused(*_args, **_kwargs):
        raise OSError("spawn refused")

    monkeypatch.setattr(extension_companion, "record_unmetered_external_dispatch", disclose)
    descriptor = CompanionDescriptor(skill_name="demo", name="bridge", command=[sys.executable, "-c", "pass"],
                                     cwd=tmp_path, env={"OPENROUTER_API_KEY": "synthetic"}, max_restarts=0)
    entered, release, monitors = _hold_terminal_writer(monkeypatch)
    try:
        assert supervisor.start(descriptor)
        spawned[-1].end(7)
        assert entered.wait(30), "the exhausted runtime's monitor never reached its terminal health write"
        spawn = extension_companion.subprocess.Popen
        if failure == "spawn":
            monkeypatch.setattr(extension_companion.subprocess, "Popen", refused)
        else:
            refuse.set()
        with pytest.raises((OSError, RuntimeError), match="spawn refused|cost disclosure failed"):
            supervisor.start(descriptor)
        monkeypatch.setattr(extension_companion.subprocess, "Popen", spawn)
        refuse.clear()
        yield supervisor, spawned, descriptor, release, monitors
    finally:
        release.set()
        for monitor in monitors:
            monitor.join(30)
        _end_fake_companions(supervisor, spawned)


def _companion_events(drive_root: pathlib.Path) -> list:
    journal = drive_root / "logs" / "events.jsonl"
    rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()
            if '"extension_companion_' in line] if journal.exists() else []
    return [(row["type"], row.get("returncode"), row.get("by")) for row in rows]


@pytest.mark.parametrize("phase", ["before_monitor", "during_backoff"])
def test_failed_settings_refresh_revokes_the_old_restart_without_losing_the_exit(tmp_path, monkeypatch, phase):
    from ouroboros import extension_companion

    init_server_process_pid()
    supervisor, spawned = _supervise_fake_companions(monkeypatch, tmp_path)
    monkeypatch.setattr(supervisor, "_start_monitor", lambda *_: None)
    descriptor = CompanionDescriptor("demo", "bridge", ["fake"], tmp_path, {"DEMO": "old"}, max_restarts=1)
    key = "demo:bridge"
    refused = []

    def refresh(*_args):
        def refuse_spawn(_argv, **kwargs):
            refused.append(kwargs["env"]["DEMO"])
            raise OSError("refresh spawn refused")

        with monkeypatch.context() as patch:
            patch.setattr(extension_companion.subprocess, "Popen", refuse_spawn)
            with pytest.raises(OSError, match="refresh spawn refused"):
                supervisor.refresh("demo", "bridge", lambda: {"DEMO": "new"}, compare_keys=["DEMO"],
                                   still_valid=lambda: True)

    try:
        assert supervisor.start(descriptor)
        runtime = supervisor._runtimes[key]
        spawned[0].end(7)
        if phase == "before_monitor":
            refresh()
            monkeypatch.setattr(extension_companion, "time", types.SimpleNamespace(
                monotonic=time.monotonic, sleep=lambda _: pytest.fail("the withdrawn restart reached backoff")))
        else:
            monkeypatch.setattr(extension_companion, "time", types.SimpleNamespace(monotonic=time.monotonic, sleep=refresh))
        supervisor._monitor_runtime(key, runtime)

        assert refused == ["new"] and len(spawned) == 1 and not supervisor.snapshot()
        assert key not in supervisor._restart_history
        health = extension_health.read_extension_health(tmp_path, "demo")
        assert health["status"] == "broken" and health["last_observed"]["reason"] == "companion_failed:bridge"
        assert health["companion_failures"]["bridge"]["returncode"] == 7
        assert _companion_events(tmp_path) == [("extension_companion_exited", 7, None)]

        # An ordinary refresh, not the old monitor, can now finish replacement even after settings revert.
        assert supervisor.refresh_blocker("demo", "bridge", {"DEMO": "old"}) == ""
        assert supervisor.refresh("demo", "bridge", lambda: {"DEMO": "old"}, compare_keys=["DEMO"],
                                  still_valid=lambda: True) == "started"
        assert _running(supervisor, key) and len(spawned) == 2
        assert _companion_events(tmp_path)[-1] == ("extension_companion_recovered", 7, "start")
    finally:
        supervisor.stop_all()


@pytest.mark.parametrize(("policy", "code"), [("on_failure", 7), ("never", 7), ("never", 0)],
                         ids=["automatic_restart", "restart_disabled", "clean_exit"])
def test_companion_exit_history_and_current_failure_are_separate(tmp_path, monkeypatch, policy, code):
    from ouroboros import extension_companion

    init_server_process_pid()
    supervisor, spawned = _supervise_fake_companions(monkeypatch, tmp_path)
    monkeypatch.setattr(supervisor, "_start_monitor", lambda *_: None)
    monkeypatch.setattr(extension_companion, "time", types.SimpleNamespace(monotonic=time.monotonic, sleep=lambda _: None))
    observed_at = "2026-01-01T12:00:00+00:00"
    monkeypatch.setattr(extension_companion, "utc_now_iso", lambda: observed_at)
    descriptor = CompanionDescriptor("demo", "bridge", ["fake"], tmp_path, {}, restart_policy=policy)
    key = "demo:bridge"
    try:
        assert supervisor.start(descriptor)
        runtime = supervisor._runtimes[key]
        spawned[0].end(code)
        supervisor._monitor_runtime(key, runtime)
        health = extension_health.read_extension_health(tmp_path, "demo")
        assert len(spawned) == (2 if policy == "on_failure" else 1)
        assert _running(supervisor, key) == (policy == "on_failure")
        assert (key in supervisor._restart_history) == (policy == "on_failure")
        if not code:
            assert health is None and not _companion_events(tmp_path)
            return
        journal = tmp_path / "logs" / "events.jsonl"
        events = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
        assert events == [{"ts": observed_at, "type": "extension_companion_exited", "skill": "demo",
                           "companion": "bridge", "returncode": 7, "version": "test", "git_sha": "test-sha"}]
        if policy == "on_failure":
            assert health is None  # successful restart opens no terminal health failure
        else:
            assert (health["status"], health["last_observed"]["reason"]) == ("broken", "companion_failed:bridge")
            assert health["companion_failures"]["bridge"]["load_error"] == "companion 'bridge' exited with code 7"
            assert supervisor.start(descriptor)
            recovered = extension_health.read_extension_health(tmp_path, "demo")
            assert recovered["status"] == "live" and not recovered["companion_failures"]
            assert recovered["companion_recovered"]["bridge"]["failed_at"] == observed_at
            assert _companion_events(tmp_path) == [
                ("extension_companion_exited", 7, None), ("extension_companion_recovered", 7, "start")]
    finally:
        supervisor.stop_all()


@pytest.mark.parametrize("failure", ["spawn", "after_spawn"])
def test_a_failed_replacement_start_keeps_the_exhaustion_its_monitor_observed(
    tmp_path: pathlib.Path, monkeypatch, failure: str,
) -> None:
    """The exhausted runtime's monitor is held at its terminal write while a start tries to replace that runtime and
    fails: its spawn refused, or its cost disclosure after the spawn. Nothing replaced it: released, the monitor still
    records the failure."""
    with _a_held_exhaustion_and_a_failed_replacement(tmp_path, monkeypatch, failure) as (
            supervisor, _spawned, _descriptor, release, monitors):
        release.set()
        monitors[0].join(30)
        assert not monitors[0].is_alive(), "the released monitor never finished"
        state = extension_health.apply_companion_failure_to_runtime_state(
            {"desired_live": True, "reason": "ready"}, supervisor.data_dir, "demo")
        assert (state.get("companion_failed"), state.get("reason")) == (True, "companion_restart_exhausted:bridge")
        assert _companion_events(supervisor.data_dir) == [("extension_companion_exhausted", 7, None)]


def test_failed_refresh_before_monitor_classifies_exit_keeps_exhaustion(tmp_path, monkeypatch):
    from ouroboros import extension_companion

    init_server_process_pid()
    loaded, repo, drive = _reviewed_companion_skill(tmp_path)
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive)
    monkeypatch.setattr(supervisor, "_start_monitor", lambda *_: None)

    def refuse_spawn(*args, **kwargs):
        raise OSError("replacement refused")

    try:
        assert extension_loader.reconcile_extension(loaded.name, drive, lambda: {}, repo_path=str(repo))["live_loaded"]
        key = f"{loaded.name}:daemon"
        runtime = supervisor._runtimes[key]
        supervisor._restart_history[key] = [time.monotonic()] * runtime.descriptor.max_restarts
        spawned[0].end(7)
        monkeypatch.setattr(extension_companion.subprocess, "Popen", refuse_spawn)
        with pytest.raises(OSError, match="replacement refused"):
            supervisor.refresh(loaded.name, "daemon", lambda: {"DEMO": "changed"},
                               compare_keys=["DEMO"], still_valid=lambda: True)
        assert supervisor.snapshot() == {}
        supervisor._monitor_runtime(key, runtime)  # classify only after refresh failed to replace it
        row = _listed_skill(drive, repo, loaded.name)
        assert row["live_loaded"] and row["live_reason"] == "companion_restart_exhausted:daemon"
        assert extension_health.read_extension_health(drive, loaded.name)["companion_failures"]["daemon"]["returncode"] == 7
        assert _companion_events(drive) == [("extension_companion_exhausted", 7, None)]
        assert len(spawned) == 1
    finally:
        extension_loader.unload_extension(loaded.name)


@pytest.mark.parametrize("later", ["running", "exhausted"])
@pytest.mark.parametrize("failure", ["spawn", "after_spawn"])
def test_a_late_monitor_journals_its_exit_once_a_later_start_succeeded(
    tmp_path: pathlib.Path, monkeypatch, failure: str, later: str,
) -> None:
    """After a failed replacement, another start succeeds while the first monitor is held.
    Its later event identifies the old exit. The replacement keeps its own running or exhausted state."""
    with _a_held_exhaustion_and_a_failed_replacement(tmp_path, monkeypatch, failure) as (
            supervisor, spawned, descriptor, release, monitors):
        drive_root = supervisor.data_dir
        assert supervisor.start(descriptor)  # this one succeeds while the old monitor is still held
        newer = spawned[-1]
        if later == "exhausted":
            _exhaust(supervisor, "demo:bridge", code=9)
        before = extension_health.read_extension_health(drive_root, "demo")
        release.set()
        monitors[0].join(30)
        assert not monitors[0].is_alive(), "the released monitor never finished"
        state = extension_health.apply_companion_failure_to_runtime_state(
            {"desired_live": True, "reason": "ready"}, drive_root, "demo")
        assert {"running": newer.poll() is None and _running(supervisor, "demo:bridge"),
                "health": extension_health.read_extension_health(drive_root, "demo"),
                "load_error": state.get("load_error"), "journal": _companion_events(drive_root)} == {
            "running": later == "running", "health": before,
            "load_error": None if later == "running"
            else "companion 'bridge' exited with code 9 after exhausting its restart budget",
            "journal": ([("extension_companion_exhausted", 9, None)] if later == "exhausted" else [])
                       + [("extension_companion_exhausted", 7, None)]}


class _ReapedUnstoredProcess(_FakeCompanionProcess):
    """``Popen`` while its own ``wait`` holds the status it reaped: that waiter has the exit code, but ``poll`` from
    another thread answers None until the waiter stores it (``Popen._waitpid_lock``)."""

    def __init__(self, argv):
        super().__init__(argv)
        self.reaped, self.store, self._status = threading.Event(), threading.Event(), None

    def end(self, code: int) -> None:
        self._status = code
        self._ended.set()

    def wait(self, timeout=None):
        self._ended.wait(timeout)
        self.reaped.set()
        assert self.store.wait(30), "the reaped status was never stored"
        self.returncode = self._status
        return self.returncode


class _PolledProcess(_FakeCompanionProcess):
    """``Popen`` that no waiter holds yet: as in Popen, only a ``poll`` or a ``wait`` caches the exit code in
    ``returncode``; every ``poll`` is counted."""

    def __init__(self, argv):
        super().__init__(argv)
        self.polls, self._status = 0, None

    def end(self, code: int) -> None:
        self._status = code
        self._ended.set()

    def poll(self):
        self.polls += 1
        if self._ended.is_set():
            self.returncode = self._status
        return self.returncode

    def wait(self, timeout=None):
        self._ended.wait(timeout)
        self.returncode = self._status
        return self.returncode


def test_a_late_monitor_writes_no_recovery_while_a_later_runtime_s_exit_is_unstored(
    tmp_path: pathlib.Path, monkeypatch,
) -> None:
    """A replacement start fails, a later one succeeds, and its runtime exits with code 9: its monitor has reaped the
    exit but not stored it, so ``poll`` still answers None. The old monitor journals only its exit;
    the later runtime's monitor then records code 9."""
    from ouroboros import extension_companion

    with _a_held_exhaustion_and_a_failed_replacement(tmp_path, monkeypatch, "spawn") as (
            supervisor, spawned, descriptor, release, monitors):
        drive_root = supervisor.data_dir

        def popen(argv, **_kwargs):
            spawned.append(_ReapedUnstoredProcess(argv))
            return spawned[-1]

        monkeypatch.setattr(extension_companion.subprocess, "Popen", popen)
        assert supervisor.start(descriptor)
        newer = spawned[-1]
        try:
            newer.end(9)
            assert newer.reaped.wait(30) and newer.poll() is None, "the later runtime's monitor never reaped its exit"
            release.set()
            monitors[0].join(30)
            assert not monitors[0].is_alive(), "the released monitor never finished"
            meanwhile = (extension_health.read_extension_health(drive_root, "demo"), _companion_events(drive_root))
        finally:
            newer.store.set()
        _until(lambda: "demo:bridge" not in supervisor.snapshot(), "the later runtime's exit was never settled")
        health = extension_health.read_extension_health(drive_root, "demo")
        assert {"meanwhile": meanwhile, "status": health["status"],
                "failure": health["companion_failures"]["bridge"]["returncode"],
                "journal": _companion_events(drive_root)} == {
            "meanwhile": (None, [("extension_companion_exhausted", 7, None)]), "status": "broken", "failure": 9,
            "journal": [("extension_companion_exhausted", 7, None), ("extension_companion_exhausted", 9, None)]}


@pytest.mark.parametrize("later", ["recovery", "newer_start", "stop", "panic"])
def test_a_failed_automatic_restart_shows_the_absent_companion_until_a_later_start(
    tmp_path: pathlib.Path, monkeypatch, later: str,
) -> None:
    """A live extension's companion exits with code 3 inside its restart budget, and the restart its monitor attempts
    is refused at the spawn: no later monitor reports that absence. Unless something supersedes it first, the health
    and the runtime state show a failed restart with that exit code (no exhausted budget, none of the spawn error's
    words) until a reconcile starts the companion again and closes it into history. A start, a stop or Panic landing
    before the failure is written leaves it in the journal only."""
    from ouroboros import extension_companion

    init_server_process_pid()
    loaded, repo_root, drive_root = _reviewed_companion_skill(tmp_path)
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive_root)
    key = f"{loaded.name}:daemon"
    monkeypatch.setattr(extension_companion, "time", types.SimpleNamespace(monotonic=time.monotonic,
                                                                           sleep=lambda _seconds: None))
    entered, release, monitors = _hold_terminal_writer(monkeypatch)
    spawn = extension_companion.subprocess.Popen

    def refused(*_args, **_kwargs):
        raise OSError("spawn refused: SYNTHETIC-SECRET")

    try:
        assert extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root, skills=[loaded]) is None
        monkeypatch.setattr(extension_companion.subprocess, "Popen", refused)
        spawned[-1].end(3)
        assert entered.wait(30), "the failed restart never reached its health write"
        assert len(supervisor._restart_history[key]) == 1
        monkeypatch.setattr(extension_companion.subprocess, "Popen", spawn)
        if later == "newer_start":
            assert supervisor.refresh(loaded.name, "daemon", lambda: {"DEMO": "changed"}, compare_keys=["DEMO"],
                                      still_valid=lambda: True) == "started"
        elif later == "stop":
            supervisor.stop(loaded.name, "daemon")
        elif later == "panic":
            supervisor.panic_kill_all()
        release.set()
        monitors[0].join(30)
        assert not monitors[0].is_alive(), "the released monitor never finished"
        health = extension_health.read_extension_health(drive_root, loaded.name) or {}
        state = extension_loader.runtime_state_for_loaded_skill(loaded, drive_root, skills=[loaded])
        assert "SYNTHETIC" not in json.dumps(health)
        assert _companion_events(drive_root) == [("extension_companion_restart_failed", 3, None)]
        if later != "recovery":
            assert not health.get("companion_failures") and state.get("companion_failed") is None
            assert _running(supervisor, key) == (later == "newer_start")
            return
        words = "companion 'daemon' exited with code 3 and its automatic restart failed"
        assert (health["status"], health["companion_failures"]["daemon"]["returncode"]) == ("broken", 3)
        assert (state["companion_failed"], state["reason"], state["load_error"]) == (
            True, "companion_failed:daemon", words)
        recovered = extension_loader.reconcile_extension(loaded.name, drive_root, lambda: {}, repo_path=str(repo_root))
        health = extension_health.read_extension_health(drive_root, loaded.name)
        closed = health["companion_recovered"]["daemon"]
        assert (recovered["companions"]["action"], recovered["reason"], health["status"]) == (
            "started_missing", "ready", "live")
        assert (closed["returncode"], closed["load_error"], closed["restart_failed"], closed["by"]) == (
            3, words, True, "start")
        assert _companion_events(drive_root)[1:] == [("extension_companion_recovered", 3, "start")]
    finally:
        release.set()
        _end_fake_companions(supervisor, spawned, loaded.name)


def test_a_companion_whose_death_is_known_is_missing_while_its_failure_is_written(
    tmp_path: pathlib.Path, monkeypatch,
) -> None:
    """A live extension's companion exhausts its restarts and its monitor is held at the terminal health write, the
    exited runtime still in the supervisor's table. A reconcile meanwhile neither answers it running nor records the
    extension known good on it: the companion is missing, so the reconcile starts it again, and the held exhaustion
    lands as history."""
    init_server_process_pid()
    loaded, repo_root, drive_root = _reviewed_companion_skill(tmp_path)
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive_root)
    key = f"{loaded.name}:daemon"

    def reconcile(sha: str) -> dict:
        return extension_loader.reconcile_extension(loaded.name, drive_root, lambda: {}, repo_path=str(repo_root),
                                                    health_stamp=("test", sha))

    entered, release, monitors = _hold_terminal_writer(monkeypatch)
    try:
        assert reconcile("first")["action"] == "extension_loaded"
        with supervisor._lock:
            exited = supervisor._runtimes[key]
            supervisor._restart_history[key] = [time.monotonic()] * exited.descriptor.max_restarts
        exited.process.end(7)
        assert entered.wait(30), "the exhausted runtime's monitor never reached its health write"
        assert supervisor.snapshot()[key]["returncode"] == 7
        assert supervisor.companion_ownership(loaded.name, ["daemon"]) == (["daemon"], {})
        state = reconcile("second")
        known_good = extension_health.read_extension_health(drive_root, loaded.name)["last_known_good"]
        assert (state["companions"]["action"], len(spawned), _running(supervisor, key)) == ("started_missing", 2, True)
        assert known_good["sha"] == "second" and spawned[-1] is not exited.process
        release.set()
        monitors[0].join(30)
        assert not monitors[0].is_alive(), "the released monitor never finished"
        health = extension_health.read_extension_health(drive_root, loaded.name)
        assert (health["status"], health["companion_failures"], _running(supervisor, key)) == ("live", {}, True)
        assert _companion_events(drive_root) == [("extension_companion_exhausted", 7, None)]
    finally:
        release.set()
        _end_fake_companions(supervisor, spawned, loaded.name)


@pytest.mark.parametrize("ending", ["cached_failure", "restart_backoff", "clean_exit"])
def test_reconcile_known_good_uses_the_already_observed_companion_outcome(tmp_path, monkeypatch, ending):
    from ouroboros import extension_companion

    init_server_process_pid()
    loaded, repo, drive = _reviewed_companion_skill(tmp_path)
    supervisor, _ = _supervise_fake_companions(monkeypatch, drive)
    monkeypatch.setattr(supervisor, "_start_monitor", lambda *_: None)
    key, health_key = f"{loaded.name}:daemon", (drive, loaded.name)
    announce = extension_loader._announce_extension_state_change

    def reconcile(sha):
        return extension_loader.reconcile_extension(loaded.name, drive, lambda: {}, repo_path=str(repo),
                                                    health_stamp=("test", sha))

    def at_announcement(*args, **kwargs):
        runtime = supervisor._runtimes[key]
        runtime.process.end(0 if ending == "clean_exit" else 7)
        if ending != "cached_failure":
            supervisor._monitor_runtime(key, runtime)
        return announce(*args, **kwargs)

    def in_backoff(_seconds):
        # The monitor has removed the failed runtime but has not yet started its replacement.
        assert key not in supervisor._runtimes
        state = {"desired_live": True, "live_loaded": True, "reason": "ready", "action": "extension_loaded"}
        health = extension_health.record_health_for_runtime_state(*health_key, state, stamp=("test", "backoff"),
                                                                 companions=["daemon"])
        assert health["last_known_good"] == good

    try:
        extension_health.record_extension_health(*health_key, status="live", reason="ready", sha="good")
        good = extension_health.read_extension_health(*health_key)["last_known_good"]
        with monkeypatch.context() as patch:
            if ending == "restart_backoff":
                patch.setattr(extension_companion, "time", types.SimpleNamespace(monotonic=time.monotonic,
                                                                                sleep=in_backoff))
            patch.setattr(extension_loader, "_announce_extension_state_change", at_announcement)
            assert reconcile("attempt")["action"] == "extension_loaded"
        expected = "good" if ending == "cached_failure" else "attempt"
        assert extension_health.read_extension_health(*health_key)["last_known_good"]["sha"] == expected
        assert reconcile("recovered")["action"] == "extension_already_live"
        assert _running(supervisor, key)
        assert extension_health.read_extension_health(*health_key)["last_known_good"]["sha"] == "recovered"
    finally:
        extension_loader.unload_extension(loaded.name)
        supervisor.stop_all()


@pytest.mark.parametrize("later", ["running", "clean_exit", "reaped_exit", "panic"])
def test_a_later_start_closes_an_older_write_landing_after_its_check_as_of_that_start(
    tmp_path: pathlib.Path, monkeypatch, later: str,
) -> None:
    """An exhausted runtime's monitor passes its check and holds the record's lock, its write not yet landed, while a
    later start succeeds. That start's own close waits for the record, outside the supervisor's lock, and closes the
    landed failure once, as of its own start, never the old monitor by a later look: the same whether the later
    process still runs, exited cleanly before its monitor even ran (meanwhile the code the start's own snapshot poll
    cached makes it missing, without another poll; nothing stays open after), or exited with code 9 that its monitor
    reaped but has not stored (that failure is its own, recorded after and never cleared by its older start). Panic
    meanwhile returns at once, and a close after it records no recovery."""
    from ouroboros import extension_companion

    init_server_process_pid()
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive_root)
    key = "demo:bridge"
    descriptor = CompanionDescriptor(skill_name="demo", name="bridge", command=[sys.executable, "-c", "pass"],
                                     cwd=tmp_path, env={}, max_restarts=0)
    real_update, entered, release, held, writers = (
        extension_health.update_json_locked, threading.Event(), threading.Event(), [], [])

    def update_json_locked(path, mutator, **kwargs):
        writers.append(threading.current_thread())

        def checked(prior):
            record = mutator(prior)
            if record and record.get("companion_failures") and not held:
                held.append(threading.current_thread())  # checked, under the record's lock, not yet written
                entered.set()
                assert release.wait(30), "the held write was never released"
            return record

        return real_update(path, checked, **kwargs)

    def reaped(argv, **_kwargs):
        spawned.append(_ReapedUnstoredProcess(argv))
        return spawned[-1]

    def dead_at_spawn(argv, **_kwargs):  # exits cleanly before anything waits for it
        spawned.append(_PolledProcess(argv))
        spawned[-1].end(0)
        return spawned[-1]

    monkeypatch.setattr(extension_health, "update_json_locked", update_json_locked)
    later_start = threading.Thread(target=supervisor.start, args=(descriptor,), daemon=True)
    spawn = extension_companion.subprocess.Popen
    try:
        assert supervisor.start(descriptor)
        spawned[-1].end(7)
        assert entered.wait(30), "the exhausted runtime's monitor never reached its write"
        monkeypatch.setattr(extension_companion.subprocess, "Popen", {"reaped_exit": reaped,
                                                                      "clean_exit": dead_at_spawn}.get(later, spawn))
        later_start.start()
        _until(lambda: later_start in writers, "the later start never reached its own close")
        newer = spawned[-1]
        with supervisor._lock:
            started = supervisor._runtimes[key].started_utc
        if later == "clean_exit":  # no monitor yet: the code start's snapshot poll cached is the one fact read
            assert (newer.polls, newer.returncode, supervisor.companion_ownership("demo", ["bridge"]),
                    supervisor.retained_companions("demo", None), newer.polls) == (1, 0, (["bridge"], {}), {}, 1)
        if later == "panic":
            supervisor.panic_kill_all()  # never waits for the record that close waits for
            assert later_start.is_alive(), "Panic waited for the later start's close"
        elif later == "reaped_exit":
            newer.end(9)
        release.set()
        for thread in (later_start, *held):
            thread.join(30)
            assert not thread.is_alive(), "a released writer never finished"
        if later == "reaped_exit":
            assert newer.reaped.wait(30) and newer.poll() is None, "the later runtime's monitor never reaped its exit"
            newer.store.set()
        if later != "running":
            _until(lambda: key not in supervisor.snapshot(), "the later runtime's end was never settled")
        health = extension_health.read_extension_health(drive_root, "demo")
        closed = (health.get("companion_recovered") or {}).get("bridge") or {}
        assert {"journal": _companion_events(drive_root), "closed_at": closed.get("closed_at"),
                "open": {name: fact["returncode"] for name, fact in health["companion_failures"].items()},
                "running": _running(supervisor, key)} == {
            "journal": [("extension_companion_exhausted", 7, None)]
                       + ([] if later == "panic" else [("extension_companion_recovered", 7, "start")])
                       + ([("extension_companion_exhausted", 9, None)] if later == "reaped_exit" else []),
            "closed_at": None if later == "panic" else started,
            "open": {"reaped_exit": {"bridge": 9}, "panic": {"bridge": 7}}.get(later, {}),
            "running": later == "running"}
    finally:
        release.set()
        for process in spawned:
            getattr(process, "store", threading.Event()).set()
        for thread in (later_start, *held):
            if thread.ident is not None:
                thread.join(30)
        _end_fake_companions(supervisor, spawned)


@pytest.mark.parametrize("successor", ["refresh_after_crash", "stop_after_crash", "stop", "panic"])
def test_a_death_a_successor_overtook_stays_history_unless_that_successor_ended_it(
    tmp_path: pathlib.Path, monkeypatch, successor: str,
) -> None:
    """The companion ends while the monitor that observed it waits for the supervisor's lock, and a successor takes
    its entry first: a refresh or a stop finding it already dead (it crashed with code 3), or a stop or Panic ending
    it. A crash that the successor only found stays in the journal as that exit alone, with no failure opened and no
    restart charged, whatever runs now; an end that the stop or Panic caused leaves nothing."""
    init_server_process_pid()
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive_root)
    descriptor = CompanionDescriptor(skill_name="demo", name="bridge", command=[sys.executable, "-c", "pass"],
                                     cwd=tmp_path, env={})
    monitors = []

    def observe_monitor(key, runtime):
        monitors.append(threading.Thread(target=supervisor._monitor_runtime, args=(key, runtime), daemon=True))
        monitors[-1].start()

    monkeypatch.setattr(supervisor, "_start_monitor", observe_monitor)
    try:
        assert supervisor.start(descriptor)
        with supervisor._lock:  # held across the end, so the successor below takes the entry first
            if successor.endswith("_after_crash"):
                spawned[0].end(3)
            if successor == "refresh_after_crash":
                assert supervisor.refresh("demo", "bridge", lambda: {"DEMO": "changed"}, compare_keys=["DEMO"],
                                          still_valid=lambda: True) == "started"
            elif successor == "panic":
                supervisor.panic_kill_all()
            else:
                supervisor.stop("demo", "bridge")
        monitors[0].join(30)
        assert not monitors[0].is_alive(), "the first runtime's monitor never finished"
        health = extension_health.read_extension_health(drive_root, "demo") or {}
        assert (_companion_events(drive_root), health.get("companion_failures"), supervisor._restart_history) == (
            [("extension_companion_exited", 3, None)] if successor.endswith("_after_crash") else [], None, {})
        assert _running(supervisor, "demo:bridge") == (successor == "refresh_after_crash")
    finally:
        _end_fake_companions(supervisor, spawned)


@pytest.mark.parametrize(("when", "budget"), [
    ("before_settle", 1), ("before_settle", 0), ("during_backoff", 1), ("panic_during_backoff", 1),
], ids=["stop_before_settle", "stop_before_settle_spent_budget", "stop_during_backoff", "panic_during_backoff"])
def test_a_withdrawn_restart_keeps_only_the_unexpected_exit(tmp_path, monkeypatch, when, budget):
    from ouroboros import extension_companion

    init_server_process_pid()
    supervisor, spawned = _supervise_fake_companions(monkeypatch, tmp_path)
    monkeypatch.setattr(supervisor, "_start_monitor", lambda *_: None)  # run this monitor at the exact interleaving
    descriptor = CompanionDescriptor("demo", "bridge", ["fake"], tmp_path, {}, max_restarts=budget)
    assert supervisor.start(descriptor)
    key = "demo:bridge"
    runtime = supervisor._runtimes[key]
    spawned[0].end(3)
    if when == "before_settle":
        # stop has forgotten this start and released its lock, but the dead runtime still has its entry.
        monkeypatch.setattr(supervisor, "_terminate_runtime",
                            lambda *_, **_kwargs: supervisor._monitor_runtime(key, runtime))
        supervisor.stop("demo", "bridge")
    else:
        control = supervisor.panic_kill_all if when == "panic_during_backoff" else lambda: supervisor.stop("demo", "bridge")
        monkeypatch.setattr(extension_companion, "time", types.SimpleNamespace(
            monotonic=time.monotonic, sleep=lambda _: control()))
        supervisor._monitor_runtime(key, runtime)
    assert len(spawned) == 1 and not supervisor.snapshot() and not supervisor._restart_history
    assert not (extension_health.read_extension_health(tmp_path, "demo") or {}).get("companion_failures")
    assert _companion_events(tmp_path) == [("extension_companion_exited", 3, None)]


def test_a_reconcile_never_hides_an_exhaustion_recorded_after_its_read(tmp_path: pathlib.Path, monkeypatch) -> None:
    """A reconcile reads a live extension whose companion runs. Before it records health, the companion exhausts its
    restarts and its monitor records that: the reconcile's own record keeps the failure."""
    init_server_process_pid()
    loaded, repo_root, drive_root = _reviewed_companion_skill(tmp_path)
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive_root)
    key = f"{loaded.name}:daemon"
    real_announce = extension_loader._announce_extension_state_change

    def exhaust_before_the_record(*args, **kwargs):
        _exhaust(supervisor, key)
        return real_announce(*args, **kwargs)

    try:
        assert extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root, skills=[loaded]) is None
        monkeypatch.setattr(extension_loader, "_announce_extension_state_change", exhaust_before_the_record)
        state = extension_loader.reconcile_extension(loaded.name, drive_root, lambda: {}, repo_path=str(repo_root))
        monkeypatch.setattr(extension_loader, "_announce_extension_state_change", real_announce)
        health = extension_health.read_extension_health(drive_root, loaded.name)
        later = extension_loader.runtime_state_for_loaded_skill(loaded, drive_root, skills=[loaded])
        assert {"read": state["companions"]["action"], "health": health["status"],
                "observed_reason": health["last_observed"]["reason"],
                "companion_failed": later.get("companion_failed"),
                "receipt": (state.get("companion_failed"), state["reason"])} == {
            "read": "already_running", "health": "broken", "observed_reason": "companion_restart_exhausted:daemon",
            "companion_failed": True, "receipt": (True, "companion_restart_exhausted:daemon")}
    finally:
        _end_fake_companions(supervisor, spawned, loaded.name)


def test_a_load_error_keeps_its_own_words_beside_an_open_companion_failure(tmp_path: pathlib.Path, monkeypatch) -> None:
    """A companion failure stays open while the payload's next load fails: the runtime state and the reconcile's
    receipt name that load error, not the older companion failure, and the reconcile stops at its load-error gate."""
    init_server_process_pid()
    loaded, repo_root, drive_root = _reviewed_companion_skill(tmp_path)
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive_root)
    try:
        extension_health.record_companion_restart_exhausted(drive_root, loaded.name, "daemon", returncode=7)
        extension_loader.record_load_failure(loaded.name, loaded, "skill 'compskill' load failure: boom")
        read = extension_loader.runtime_state_for_loaded_skill(loaded, drive_root, skills=[loaded])
        state = extension_loader.reconcile_extension(loaded.name, drive_root, lambda: {}, repo_path=str(repo_root))
        assert {"read": (read["reason"], read["load_error"], read.get("companion_failed")),
                "receipt": (state["action"], state["reason"], state["load_error"]), "spawned": len(spawned)} == {
            "read": ("load_error", "skill 'compskill' load failure: boom", None),
            "receipt": ("extension_load_error", "load_error", "skill 'compskill' load failure: boom"), "spawned": 0}
    finally:
        with extension_loader._lock:
            extension_loader._load_failures.pop(loaded.name, None)
        _end_fake_companions(supervisor, spawned, loaded.name)


@pytest.mark.parametrize("loaded_before", [False, True], ids=["fresh_load", "already_live"])
def test_an_old_form_companion_failure_ends_at_that_companion_s_next_start(
    tmp_path: pathlib.Path, monkeypatch, loaded_before: bool,
) -> None:
    """A fresh load or an already-live reconcile closes a legacy failure and keeps its only diagnostic."""
    from ouroboros import extension_companion

    init_server_process_pid()
    loaded, repo_root, drive_root = _reviewed_companion_skill(tmp_path)
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive_root)
    key = f"{loaded.name}:daemon"
    try:
        if loaded_before:
            with monkeypatch.context() as earlier:  # its last start precedes the failure the old record names
                earlier.setattr(extension_companion, "utc_now_iso", lambda: "2026-11-17T01:00:00+00:00")
                assert extension_loader.load_extension(
                    loaded, lambda: {}, drive_root=drive_root, skills=[loaded]) is None
            spawned[-1].end(0)
            _until(lambda: key not in supervisor.snapshot(), "the companion's exit was never settled")
        failure = {"version": "7.9.0", "sha": "a" * 40, "status": "broken", "reason": "companion_restart_exhausted:daemon",
                   "load_error": "companion 'daemon' exited with code 1 after exhausting its restart budget",
                   "ts": "2026-11-17T01:50:00Z", "process": "server"}
        if loaded_before:
            failure["returncode"] = 1
        extension_health.health_path(drive_root, loaded.name).write_text(json.dumps({
            "_schema_version": 1, "skill": loaded.name, "status": "broken", "regressed": False,
            "last_known_good": {"version": "7.9.0", "sha": "a" * 40, "ts": "2026-11-16T20:00:00Z"},
            "last_observed": failure, "observations": {"server": failure}}), encoding="utf-8")
        before = extension_loader.runtime_state_for_loaded_skill(loaded, drive_root, skills=[loaded])
        assert before["companion_failed"] is True and before["reason"] == "companion_restart_exhausted:daemon"
        assert before["load_error"] == failure["load_error"]
        state = extension_loader.reconcile_extension(loaded.name, drive_root, lambda: {}, repo_path=str(repo_root))
        health = extension_health.read_extension_health(drive_root, loaded.name)
        later = extension_loader.runtime_state_for_loaded_skill(loaded, drive_root, skills=[loaded])
        assert {"running": _running(supervisor, key), "receipt": state.get("companion_failed"),
                "reason": state["reason"], "health": health["status"], "later": later.get("companion_failed")} == {
            "running": True, "receipt": None, "reason": "ready", "health": "live", "later": None}
        closed = health["companion_recovered"]["daemon"]
        rows = (drive_root / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
        events = [json.loads(line) for line in rows if '"extension_companion_recovered"' in line]
        assert len(events) == 1
        for record in (closed, events[0]):
            assert record["load_error"] == failure["load_error"]
            assert record.get("returncode") == failure.get("returncode") and record["failed_at"] == failure["ts"]
    finally:
        _end_fake_companions(supervisor, spawned, loaded.name)


@pytest.mark.parametrize("converted", [False, True], ids=["old_record", "already_converted"])
def test_an_old_joint_failure_names_only_the_companions_still_failed(
    tmp_path: pathlib.Path, monkeypatch, converted: bool,
) -> None:
    """The original observation keeps both exit codes. Only the companion that still needs recovery is shown."""
    from ouroboros import extension_companion

    init_server_process_pid()
    loaded, repo_root, drive_root = _reviewed_companion_skill(tmp_path, companions=("alpha", "beta"))
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive_root)
    raw = ("companion 'alpha' exited with code 1 after exhausting its restart budget; "
           "companion 'beta' exited with code 2 after exhausting its restart budget")
    failure = {"version": "7.9.0", "sha": "a" * 40, "status": "broken", "reason": "companion_restart_exhausted:alpha,beta",
               "load_error": raw, "ts": "2026-11-17T01:50:00Z", "process": "server"}
    try:
        with monkeypatch.context() as earlier:  # both last started before the failure the old record names
            earlier.setattr(extension_companion, "utc_now_iso", lambda: "2026-11-17T01:00:00+00:00")
            assert extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root, skills=[loaded]) is None
        for process in spawned:
            process.end(0)  # a clean exit: no restart, no new failure
        _until(lambda: not supervisor.snapshot(), "the companions' exits were never settled")
        record = {
            "_schema_version": 1, "skill": loaded.name, "status": "broken", "regressed": False, "last_known_good": None,
            "last_observed": failure, "observations": {"server": failure}}
        if converted:
            record.update(legacy_companion_failure=failure, companion_failures={
                name: {"ts": failure["ts"], "load_error": raw} for name in ("alpha", "beta")})
        extension_health.health_path(drive_root, loaded.name).write_text(json.dumps(record), encoding="utf-8")
        alpha = "companion 'alpha' exhausted its restart budget"
        beta = "companion 'beta' exhausted its restart budget"
        before = extension_loader.runtime_state_for_loaded_skill(loaded, drive_root, skills=[loaded])
        assert (before["reason"], before["load_error"]) == ("companion_restart_exhausted:alpha,beta", f"{alpha}; {beta}")
        assert supervisor.refresh(loaded.name, "alpha", lambda: {"DEMO": "changed"}, compare_keys=["DEMO"],
                                  still_valid=lambda: True) == "started"
        health = extension_health.read_extension_health(drive_root, loaded.name)
        later = extension_loader.runtime_state_for_loaded_skill(loaded, drive_root, skills=[loaded])
        closed = health["companion_recovered"]["alpha"]
        assert (sorted(health["companion_failures"]), later["reason"], later["load_error"]) == (
            ["beta"], "companion_restart_exhausted:beta", beta)
        assert health["last_observed"]["load_error"] == beta and "returncode" not in health["companion_failures"]["beta"]
        assert (closed["load_error"], closed["returncode"], closed["failed_at"]) == (alpha, None, failure["ts"])
        assert health["legacy_companion_failure"] == failure
        state = extension_loader.reconcile_extension(loaded.name, drive_root, lambda: {}, repo_path=str(repo_root))
        health = extension_health.read_extension_health(drive_root, loaded.name)
        assert (state["companions"]["started"], state["reason"], sorted(health["companion_recovered"])) == (
            ["beta"], "ready", ["alpha", "beta"])
    finally:
        _end_fake_companions(supervisor, spawned, loaded.name)


@pytest.mark.parametrize("journal", ["appended", "unwritable"])
def test_a_companion_s_exhaustion_and_recovery_stay_in_the_events_journal(
    tmp_path: pathlib.Path, monkeypatch, caplog, journal: str,
) -> None:
    """A failure's facts outlive its health record: the exhaustion and the recovery each append one event (skill,
    companion, return code, time, code stamp), and a later reconcile neither removes nor repeats them. The record
    keeps the last closed failure too, so an event the journal could not take is logged and the closing it names
    stays readable there."""
    init_server_process_pid()
    loaded, repo_root, drive_root = _reviewed_companion_skill(tmp_path)
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive_root)
    if journal == "unwritable":
        monkeypatch.setattr(extension_health, "append_jsonl", lambda *_args, **_kwargs: False)
    try:
        assert extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root, skills=[loaded]) is None
        with caplog.at_level(logging.WARNING):
            _exhaust(supervisor, f"{loaded.name}:daemon", code=7)
            failed_at = extension_health.read_extension_health(drive_root, loaded.name)["companion_failures"][
                "daemon"]["ts"]
            for _ in range(2):  # the recovery, then a reconcile with nothing to start
                extension_loader.reconcile_extension(loaded.name, drive_root, lambda: {}, repo_path=str(repo_root))
        journal_path = drive_root / "logs" / "events.jsonl"
        rows = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()
                if line.strip()] if journal_path.exists() else []
        events = [row for row in rows if str(row.get("type", "")).startswith("extension_companion_")]
        kept = extension_health.read_extension_health(drive_root, loaded.name)["companion_recovered"]["daemon"]
        unjournaled = [record.getMessage() for record in caplog.records if "not journaled" in record.getMessage()]
        assert (kept["returncode"], kept["failed_at"], kept["by"]) == (7, failed_at, "start")
        assert failed_at <= kept["closed_at"]
        if journal == "unwritable":
            assert events == [] and len(unjournaled) == 2  # the exhaustion and the recovery, each once
            assert "extension_companion_recovered" in unjournaled[1] and kept["closed_at"] in unjournaled[1]
            return
        assert unjournaled == []
        assert [(row["type"], row["skill"], row["companion"], row["returncode"]) for row in events] == [
            ("extension_companion_exhausted", loaded.name, "daemon", 7),
            ("extension_companion_recovered", loaded.name, "daemon", 7)]
        exhausted, recovered = events
        assert recovered["failed_at"] == exhausted["ts"] == failed_at <= recovered["ts"] == kept["closed_at"]
        assert all({"version", "git_sha"} <= set(row) for row in events)
    finally:
        _end_fake_companions(supervisor, spawned, loaded.name)


def test_companion_notifications_follow_saved_server_failure_changes(tmp_path, monkeypatch):
    from ouroboros import utils
    from ouroboros.gateway import ws

    skill = (tmp_path, "demo")
    lock = extension_health.health_path(*skill).with_suffix(".json.lock")
    notices = []

    def broadcast(frame):
        failures = extension_health.read_extension_health(*skill)["companion_failures"]
        notices.append((failures.get("alpha", {}).get("returncode"), lock.exists(), frame))
        raise RuntimeError("UI unavailable")  # delivery cannot undo the health transition

    def lost(*_args, **_kwargs):
        raise OSError("health write lost")

    monkeypatch.setattr(ws, "broadcast_ws_sync", broadcast)
    monkeypatch.setattr(extension_health, "code_stamp", lambda: ("test", "test-sha"))
    record = extension_health.record_companion_restart_exhausted
    record(*skill, "alpha", returncode=7)
    record(*skill, "alpha", returncode=7, observed_at="later")
    record(*skill, "alpha", returncode=9, still_current=lambda: False)
    record(*skill, "alpha", returncode=9)
    extension_health.record_extension_health(*skill, status="live", companions=["alpha"])
    extension_health.record_extension_health(*skill, status="live", process="worker", companions=[],
                                            retained_companions=lambda: pytest.fail("worker read process ownership"))
    with monkeypatch.context() as patch:
        patch.setattr(utils, "atomic_write_json", lost)
        extension_health.clear_companion_restart_exhausted(*skill, "alpha")
    extension_health.clear_companion_restart_exhausted(*skill, "alpha")
    extension_health.clear_companion_restart_exhausted(*skill, "alpha")
    assert notices == [(code, False, {"type": "extension_health_changed"}) for code in (7, 9, None)]


@pytest.mark.parametrize("attempt", ["refresh", "stop", "overlapping"])
def test_unconfirmed_stop_notifies_an_open_ready_card_once(tmp_path, monkeypatch, attempt):
    from ouroboros.gateway import ws

    init_server_process_pid()
    loaded, repo, drive = _reviewed_companion_skill(tmp_path)
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive)

    try:
        assert extension_loader.reconcile_extension(loaded.name, drive, lambda: {}, repo_path=str(repo))["live_loaded"]
        reads = [_listed_skill(drive, repo, loaded.name)]
        assert reads[0]["live_reason"] == "ready" and "retained_companions" not in reads[0]

        def receive(frame):
            if frame == {"type": "extension_health_changed"}:
                reads.append(_listed_skill(drive, repo, loaded.name))

        monkeypatch.setattr(ws, "broadcast_ws_sync", receive)
        terminate = supervisor._terminate_runtime
        monkeypatch.setattr(supervisor, "_terminate_runtime", lambda *_args, **_kwargs: None)
        if attempt == "stop":
            supervisor.stop(loaded.name, "daemon")
            reason = "stop requested"
        else:
            def stop_during_refresh(*_args, **_kwargs):
                with monkeypatch.context() as inner:
                    inner.setattr(supervisor, "_terminate_runtime", lambda *_a, **_kw: None)
                    supervisor.stop(loaded.name, "daemon")
                assert len(reads) == 2, "the first completed attempt already found the process awaiting exit"

            with monkeypatch.context() as patch:
                if attempt == "overlapping":
                    patch.setattr(supervisor, "_terminate_runtime", stop_during_refresh)
                result = supervisor.refresh(loaded.name, "daemon", lambda: {"DEMO": "changed"},
                                            compare_keys=["DEMO"], still_valid=lambda: True)
            assert result == "stop_unconfirmed"
            reason = "settings refresh"
        assert [row.get("retained_companions") for row in reads] == [None, {"daemon": reason}]
        if attempt == "refresh":
            spawned[0].end(-15)
            _until(lambda: len(reads) >= 3, "late exit did not invalidate the pending-stop card")
            assert reads[-1]["live_reason"] == "companion_refresh_pending"
            assert "retained_companions" not in reads[-1] and len(spawned) == 1
            monkeypatch.setattr(supervisor, "_terminate_runtime", terminate)
            assert supervisor.refresh(loaded.name, "daemon", lambda: {"DEMO": "changed"},
                                      compare_keys=["DEMO"], still_valid=lambda: True) == "started"
            assert [row["live_reason"] for row in reads] == [
                "ready", "retained_unresolved", "companion_refresh_pending", "ready"]
            assert len(spawned) == 2
            return
        supervisor.stop(loaded.name, "daemon")  # still awaiting exit of the same process
        assert len(reads) == 2 and len(spawned) == 1
    finally:
        _end_fake_companions(supervisor, spawned, loaded.name)


@pytest.mark.parametrize("health_write", ["ok", "malformed", "busy"])
def test_unparseable_installed_manifest_still_shows_its_retained_companion(tmp_path, monkeypatch, health_write):
    init_server_process_pid()
    loaded, repo, drive = _reviewed_companion_skill(tmp_path)
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive)

    def reconcile():
        return extension_loader.reconcile_extension(loaded.name, drive, lambda: {}, repo_path=str(repo))

    try:
        assert reconcile()["live_loaded"]
        path = extension_health.health_path(drive, loaded.name)
        saved = path.read_bytes()
        monkeypatch.setattr(supervisor, "_terminate_runtime", lambda *_args, **_kwargs: None)
        save_enabled(drive, loaded.name, False)
        waiting = {"daemon": "stop requested"}
        with monkeypatch.context() as patch:
            if health_write == "malformed":
                path.write_bytes(b'{"last_observed":')
            elif health_write == "busy":
                def busy(*_args, **_kwargs):
                    raise TimeoutError("health writer busy before mutation")

                patch.setattr(extension_health, "update_json_locked", busy)
            before = path.read_bytes()
            off = reconcile()
            assert (off["reason"], off["desired_live"], off["retained_companions"]) == ("disabled", False, waiting)
            listed = _listed_skill(drive, repo, loaded.name)
            assert (listed["live_reason"], listed["retained_companions"]) == ("disabled", waiting)
            if health_write != "ok":
                assert path.read_bytes() == before
        if health_write == "malformed":
            path.write_bytes(saved)  # simulate storage repair; the lifecycle itself never rebuilds bad bytes
        (loaded.skill_dir / "SKILL.md").write_text("---\nname: [\n---\n", encoding="utf-8")
        row = _listed_skill(drive, repo, loaded.name)
        assert row["type"] == "instruction" and "manifest parse error" in row["load_error"]
        assert row.get("retained_companions") == waiting
        assert reconcile().get("retained_companions") == waiting
        spawned[0].end(0)
        assert "retained_companions" not in _listed_skill(drive, repo, loaded.name) and len(spawned) == 1
    finally:
        _end_fake_companions(supervisor, spawned, loaded.name)


@pytest.mark.parametrize("loss", ["transient", "persistent"])
def test_an_exhaustion_whose_health_write_does_not_land_is_written_again(
    tmp_path: pathlib.Path, monkeypatch, caplog, loss: str,
) -> None:
    """The exhaustion's health write times out (a held lock, a slow disk). Its monitor writes it again, so the
    extension shows the stopped companion; a store that never takes it is tried a bounded number of times, and the
    health write loss is logged. In either case the observed exhaustion reaches the journal once."""
    from ouroboros.gateway import ws

    init_server_process_pid()
    loaded, repo_root, drive_root = _reviewed_companion_skill(tmp_path, companions=("daemon", "peer"))
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive_root)
    real_update, lost, armed = extension_health.update_json_locked, [], threading.Event()
    observed_during_retry = []

    def update_json_locked(path, mutator, **kwargs):
        if armed.is_set() and (loss == "persistent" or not lost):
            lost.append(path)
            raise TimeoutError("the health lock stayed held")
        if armed.is_set():  # the retry reached the writer, but has yet to acquire the health lock
            armed.clear()
            with supervisor._lock:
                supervisor._runtimes[f"{loaded.name}:peer"].retiring = "stop requested"
            extension_loader.reconcile_extension(loaded.name, drive_root, lambda: {}, repo_path=str(repo_root))
            latest = extension_health.read_extension_health(drive_root, loaded.name)["last_observed"]
            observed_during_retry.append(latest.get("retained_companions"))
        return real_update(path, mutator, **kwargs)

    monkeypatch.setattr(extension_health, "update_json_locked", update_json_locked)
    monkeypatch.setattr(extension_health, "_EXHAUSTED_WRITE_RETRY_SEC", (0.0, 0.0), raising=False)
    try:
        assert extension_loader.reconcile_extension(loaded.name, drive_root, lambda: {}, repo_path=str(repo_root))[
            "action"] == "extension_loaded"  # recorded live
        notices = []
        monkeypatch.setattr(ws, "broadcast_ws_sync", notices.append)
        armed.set()
        with caplog.at_level(logging.WARNING):
            _exhaust(supervisor, f"{loaded.name}:daemon", code=7)
        armed.clear()
        health = extension_health.read_extension_health(drive_root, loaded.name)
        if loss == "transient":
            assert observed_during_retry == [{"peer": "stop requested"}]
            assert health["last_observed"].get("retained_companions") == observed_during_retry[0]
        assert {"attempts": len(lost),
                "health": health["status"],
                "events": _companion_events(drive_root),
                "lost": any("could not be written" in record.getMessage() for record in caplog.records)} == {
            "transient": {"attempts": 1, "health": "broken", "events": [("extension_companion_exhausted", 7, None)],
                          "lost": False},
            "persistent": {"attempts": 3, "health": "live", "events": [("extension_companion_exhausted", 7, None)],
                           "lost": True},
        }[loss]
        if loss == "persistent":
            assert notices == [{"type": "extension_health_changed"}]
            before = extension_health.health_path(drive_root, loaded.name).read_bytes()
            with monkeypatch.context() as patch:
                for proc in spawned:
                    patch.setattr(proc, "poll", lambda: pytest.fail("listing polled a companion"))
                    patch.setattr(proc, "wait", lambda: pytest.fail("listing waited for a companion"))
                row = _listed_skill(drive_root, repo_root, loaded.name)
            assert (row["live_loaded"], row["live_reason"], row["load_error"]) == (True, "health_unavailable", None)
            assert extension_health.health_path(drive_root, loaded.name).read_bytes() == before and len(spawned) == 2
            refreshed = supervisor.refresh(loaded.name, "daemon", lambda: {"DEMO": "changed"},
                                           compare_keys=["DEMO"], still_valid=lambda: True)
            assert refreshed == "started" and notices == [{"type": "extension_health_changed"}] * 2
            assert _listed_skill(drive_root, repo_root, loaded.name)["live_reason"] == "ready"
    finally:
        armed.clear()
        _end_fake_companions(supervisor, spawned, loaded.name)


@pytest.mark.parametrize("next_step", ["reconcile", "failed_refresh", "clean_exit", "later_crash", "unwritable_reconcile"])
def test_a_reconcile_closes_the_failure_a_running_companion_s_start_could_not_record(
    tmp_path: pathlib.Path, monkeypatch, next_step: str,
) -> None:
    """The successful start's health write is lost. A reconcile or a failed subsequent replacement closes the
    old failure as of that start; a reader never needs to start anything to stop showing the stale fault. That
    start stays a fact once its process ends: a clean exit brings the old fault back to no reader, and a later
    crash records the recovery it interrupted before its own failure. While nothing can be written at all, a
    reconcile and a reload answer as the listing reads."""
    init_server_process_pid()
    loaded, repo_root, drive_root = _reviewed_companion_skill(tmp_path)
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive_root)
    key = f"{loaded.name}:daemon"
    real_update, lose, unwritable = extension_health.update_json_locked, threading.Event(), threading.Event()

    def update_json_locked(path, mutator, **kwargs):
        if lose.is_set() or unwritable.is_set():
            lose.clear()
            raise TimeoutError("the health lock stayed held")
        return real_update(path, mutator, **kwargs)

    monkeypatch.setattr(extension_health, "update_json_locked", update_json_locked)
    try:
        assert extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root, skills=[loaded]) is None
        _exhaust(supervisor, key)
        lose.set()
        assert supervisor.refresh(loaded.name, "daemon", lambda: {"DEMO": "changed"}, compare_keys=["DEMO"],
                                  still_valid=lambda: True) == "started"
        assert not lose.is_set(), "the start never tried to record its recovery"
        assert extension_health.read_extension_health(drive_root, loaded.name)["status"] == extension_health.BROKEN
        assert _listed_skill(drive_root, repo_root, loaded.name)["live_reason"] == "ready"
        started = supervisor._runtimes[key].started_utc
        if next_step == "clean_exit":
            spawned[-1].end(0)
            _until(lambda: key not in supervisor.snapshot(), "the clean exit was never settled")
            row = _listed_skill(drive_root, repo_root, loaded.name)
            stale = extension_health.read_extension_health(drive_root, loaded.name)
            assert (row["live_reason"], row["load_error"], stale["status"], len(spawned)) == (
                "ready", None, extension_health.BROKEN, 2)  # read from that start, with nothing started or written
            return
        if next_step == "later_crash":
            _exhaust(supervisor, key, code=9)
            health = extension_health.read_extension_health(drive_root, loaded.name)
            closed, row = health["companion_recovered"]["daemon"], _listed_skill(drive_root, repo_root, loaded.name)
            assert (health["companion_failures"]["daemon"]["returncode"], closed["returncode"], closed["closed_at"],
                    closed["by"]) == (9, 7, started, "exit")
            assert _companion_events(drive_root) == [
                ("extension_companion_exhausted", 7, None), ("extension_companion_exhausted", 9, None),
                ("extension_companion_recovered", 7, "exit")]
            assert row["live_reason"] == "companion_restart_exhausted:daemon" and "code 9" in row["load_error"]
            return
        if next_step == "unwritable_reconcile":
            unwritable.set()
            state = extension_loader.reconcile_extension(loaded.name, drive_root, lambda: {}, repo_path=str(repo_root))
            reloaded = extension_loader.reload_all(drive_root, lambda: {}, repo_path=str(repo_root))[loaded.name]
            row = _listed_skill(drive_root, repo_root, loaded.name)
            unwritable.clear()
            kept = extension_health.read_extension_health(drive_root, loaded.name)
            assert {"receipt": (state["companions"]["action"], state.get("companion_failed"), state["reason"],
                                state.get("load_error")), "reload": reloaded,
                    "listed": (row["live_reason"], row["load_error"]),
                    "record": (kept["status"], sorted(kept["companion_failures"])), "spawned": len(spawned)} == {
                "receipt": ("already_running", None, "ready", None), "reload": None, "listed": ("ready", None),
                "record": (extension_health.BROKEN, ["daemon"]), "spawned": 2}
            return
        if next_step == "failed_refresh":
            reads = iter(({"DEMO": "new"},))

            def unreadable_on_replacement():
                try:
                    return next(reads)
                except StopIteration:
                    raise OSError("synthetic settings read failed after the stop") from None

            with pytest.raises(OSError):
                supervisor.refresh(loaded.name, "daemon", unreadable_on_replacement, compare_keys=["DEMO"],
                                   still_valid=lambda: True)
            health = extension_health.read_extension_health(drive_root, loaded.name)
            failed = health["companion_failures"]["daemon"]
            assert failed["refresh_failed"] and "returncode" not in failed
            assert health["companion_recovered"]["daemon"]["closed_at"] == started
            assert _companion_events(drive_root) == [
                ("extension_companion_exhausted", 7, None), ("extension_companion_refresh_failed", None, None),
                ("extension_companion_recovered", 7, "refresh"),
            ]
            row = _listed_skill(drive_root, repo_root, loaded.name)
            assert (row["live_reason"], row["load_error"]) == ("companion_failed:daemon", failed["load_error"])
            return
        before = len(spawned)
        state = extension_loader.reconcile_extension(loaded.name, drive_root, lambda: {}, repo_path=str(repo_root))
        health = extension_health.read_extension_health(drive_root, loaded.name)
        assert {"companions": state["companions"]["action"], "spawned": len(spawned) - before,
                "running": _running(supervisor, key), "receipt": state.get("companion_failed"),
                "health": health["status"]} == {
            "companions": "already_running", "spawned": 0, "running": True, "receipt": None, "health": "live"}
        journal = drive_root / "logs" / "events.jsonl"
        recovered = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()
                     if '"extension_companion_recovered"' in line]
        assert [(row["by"], row["returncode"]) for row in recovered] == [("reconcile", 7)]
        _exhaust(supervisor, key, code=9)
        supervisor.close_recovered_failures(loaded.name, ["daemon"])  # no running runtime: nothing to close
        assert extension_health.read_extension_health(drive_root, loaded.name)["status"] == extension_health.BROKEN
    finally:
        _end_fake_companions(supervisor, spawned, loaded.name)


def test_a_retained_sibling_does_not_keep_a_running_companion_s_lost_close(tmp_path: pathlib.Path, monkeypatch) -> None:
    """Alpha's recovery, beta's exit, and a successful replacement are three separate observations. Beta's stop
    forgot its start, so after its exit nothing confirms it until the replacement starts it."""
    from ouroboros.gateway.extensions import _build_extensions_index

    init_server_process_pid()
    loaded, repo_root, drive_root = _reviewed_companion_skill(tmp_path, companions=("alpha", "beta"))
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive_root)
    alpha, beta = f"{loaded.name}:alpha", f"{loaded.name}:beta"
    real_update, lose = extension_health.update_json_locked, threading.Event()

    def update_json_locked(path, mutator, **kwargs):
        if lose.is_set():
            lose.clear()
            raise TimeoutError("the health lock stayed held")
        return real_update(path, mutator, **kwargs)

    monkeypatch.setattr(extension_health, "update_json_locked", update_json_locked)
    try:
        def reconcile(sha):
            return extension_loader.reconcile_extension(loaded.name, drive_root, lambda: {}, repo_path=str(repo_root),
                                                        health_stamp=("test", sha))

        def listed():
            return next(row for row in _build_extensions_index(drive_root, str(repo_root))["skills"]
                        if row["name"] == loaded.name)

        assert reconcile("original")["action"] == "extension_loaded"
        known_good = extension_health.read_extension_health(drive_root, loaded.name)["last_known_good"]
        assert known_good["sha"] == "original"
        _exhaust(supervisor, alpha)
        lose.set()
        assert supervisor.refresh(loaded.name, "alpha", lambda: {"DEMO": "changed"}, compare_keys=["DEMO"],
                                  still_valid=lambda: True) == "started"
        assert not lose.is_set(), "the start never tried to record its recovery"
        with supervisor._lock:  # as stop() leaves beta while its death is not observed yet: no current start
            supervisor._last_descriptors.pop(beta, None)
            supervisor._unreplaced.pop(beta, None)
            supervisor._runtimes[beta].retiring = "stop requested"
        warning = {"beta": "stop requested"}
        row = listed()
        assert row["retained_companions"] == warning and row["live_reason"] == "retained_unresolved"
        state = reconcile("held")
        health = extension_health.read_extension_health(drive_root, loaded.name)
        assert state["companions"]["action"] == "retained_unresolved" and _running(supervisor, alpha)
        assert state.get("companion_failed") is None and health["status"] == "live"
        assert _companion_events(drive_root) == [
            ("extension_companion_exhausted", 7, None), ("extension_companion_recovered", 7, "reconcile")]
        row = listed()
        assert state["reason"] == row["live_reason"] == "retained_unresolved"
        assert state["retained_companions"] == row["retained_companions"] == warning
        assert health["last_observed"]["retained_companions"] == warning
        assert row["live_loaded"] and row["load_error"] is None and not health["companion_failures"]
        assert row["last_known_good"] == known_good and len(spawned) == 3

        def exit_before_write(path, mutator, **kwargs):
            supervisor._runtimes[beta].process.end(0)
            _until(lambda: beta not in supervisor.snapshot(), "beta's stop was never settled")
            return real_update(path, mutator, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(extension_health, "update_json_locked", exit_before_write)
            settled = reconcile("settled")
            assert settled["companions"]["retained"] == warning
            assert settled["reason"] == "health_unavailable" and "retained_companions" not in settled
        row = listed()
        assert "retained_companions" not in row and row["live_reason"] == "health_unavailable" and len(spawned) == 3
        observed = row["health_observations"]["server"]
        assert observed["sha"] == "settled" and observed["status"] == "live" and "retained_companions" not in observed
        assert row["last_known_good"] == known_good
        assert reconcile("replaced")["companions"]["started"] == ["beta"]
        row = listed()
        assert row["live_reason"] == "ready" and row["last_known_good"]["sha"] == "replaced" and len(spawned) == 4
        assert "retained_companions" not in row["health_observations"]["server"]
    finally:
        _end_fake_companions(supervisor, spawned, loaded.name)


@pytest.mark.parametrize("beta", ["exited", "awaiting_exit"])
def test_a_companion_s_own_health_write_names_the_processes_awaiting_exit_as_seen_then(
    tmp_path: pathlib.Path, monkeypatch, beta: str,
) -> None:
    """Beta starts or finishes waiting for exit while alpha's recovery is waiting for the health writer."""
    init_server_process_pid()
    loaded, repo_root, drive_root = _reviewed_companion_skill(tmp_path, companions=("alpha", "beta"))
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive_root)
    alpha, beta_key = f"{loaded.name}:alpha", f"{loaded.name}:beta"
    try:
        assert extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root, skills=[loaded]) is None
        _exhaust(supervisor, alpha)
        with supervisor._lock:
            waiting = supervisor._runtimes[beta_key]
            if beta == "exited":
                waiting.retiring = "stop requested"
        real_update = extension_health.update_json_locked

        def change_before_write(path, mutator, **kwargs):
            if beta == "exited":
                waiting.process.end(0)
            else:
                with supervisor._lock:
                    waiting.retiring = "stop requested"
            return real_update(path, mutator, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(extension_health, "update_json_locked", change_before_write)
            assert supervisor.refresh(loaded.name, "alpha", lambda: {"DEMO": "changed"}, compare_keys=["DEMO"],
                                      still_valid=lambda: True) == "started"
        health = extension_health.read_extension_health(drive_root, loaded.name)
        observed = health["last_observed"]
        assert (health["status"], health["companion_failures"]) == ("live", {})
        assert (observed.get("retained_companions"), observed["reason"]) == {
            "exited": (None, "companion_restarted"),
            "awaiting_exit": ({"beta": "stop requested"}, "retained_unresolved")}[beta]
    finally:
        _end_fake_companions(supervisor, spawned, loaded.name)


@pytest.mark.parametrize(("had_good", "caller"), [(False, "reconcile"), (True, "reconcile"), (True, "report_lost"),
                                                  (True, "reload_all"), (True, "direct"),
                                                  pytest.param(True, "queue", id="queued-recovery")])
def test_a_recovery_failing_after_it_restarted_one_companion_records_the_disposed_publication(
    tmp_path: pathlib.Path, monkeypatch, had_good, caller,
) -> None:
    """Beta's refusal disposes the bundle after alpha recovered. The reconcile answers with the load error that
    recovery recorded once (the POST reconcile receipt, and reload_all with no record of its own); a direct caller of
    the recovery still gets the refusal, as does a reconcile when nothing could record it. Only a complete retry
    advances known good."""
    from ouroboros import extension_companion
    from ouroboros.extension_reconcile_queue import process_extension_reconcile_requests, request_extension_reconcile
    from ouroboros.gateway import ws
    from ouroboros.gateway.extension_receipts import extension_reconcile_receipt
    from ouroboros.gateway.extensions import _build_extensions_index

    init_server_process_pid()
    loaded, repo_root, drive_root = _reviewed_companion_skill(tmp_path, companions=("alpha", "beta"))
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive_root)
    alpha, beta = f"{loaded.name}:alpha", f"{loaded.name}:beta"
    spawn = extension_companion.subprocess.Popen

    def refuse_beta(argv, **kwargs):
        if any("beta" in str(part) for part in argv):
            raise OSError("spawn refused")
        return spawn(argv, **kwargs)

    def reconcile() -> dict:
        return extension_loader.reconcile_extension(loaded.name, drive_root, lambda: {}, repo_path=str(repo_root))

    try:
        assert extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root, skills=[loaded]) is None
        good = None
        if had_good:
            assert reconcile()["companions"]["action"] == "already_running"
            good = extension_health.read_extension_health(drive_root, loaded.name)["last_known_good"]
            assert good
        _exhaust(supervisor, alpha)
        with supervisor._lock:
            beta_process = supervisor._runtimes[beta].process
        beta_process.end(0)  # a clean exit: no restart, no failure
        _until(lambda: beta not in supervisor.snapshot(), "beta's clean exit was never settled")
        monkeypatch.setattr(extension_companion.subprocess, "Popen", refuse_beta)
        def unavailable(*args, **kwargs):
            raise TimeoutError("health observation unavailable")

        words = f"skill {loaded.name!r} companion recovery failure: OSError: spawn refused"
        if caller in ("report_lost", "direct"):
            with monkeypatch.context() as patch, pytest.raises(OSError, match="spawn refused"):
                if caller == "report_lost":
                    patch.setattr(extension_loader, "_finalize_extension_reconcile", unavailable)
                    reconcile()
                else:
                    extension_loader.ensure_companions_running(loaded.name, drive_root, lambda: {},
                                                               repo_path=str(repo_root))
            if caller == "report_lost":
                assert reconcile()["reason"] == "load_error"
        elif caller == "reload_all":
            assert extension_loader.reload_all(drive_root, lambda: {}, repo_path=str(repo_root))[loaded.name] == words
        elif caller == "queue":
            marker = request_extension_reconcile(drive_root, loaded.name)
            assert process_extension_reconcile_requests(drive_root, lambda: {}, repo_path=str(repo_root)) == []
            assert json.loads(marker.read_text(encoding="utf-8"))["attempts"] == 1
        else:
            failed = reconcile()
            assert (extension_reconcile_receipt(loaded.name, failed), failed["companions"]) == ({
                "skill": loaded.name, "extension_action": "extension_load_error", "extension_reason": "load_error",
                "live_loaded": False, "load_error": words, "process": "server", "server_reconcile": ""},
                {"action": "recovery_failed"})
        _until(lambda: not supervisor.snapshot(), "the disposed publication's alpha was never settled")
        health = extension_health.read_extension_health(drive_root, loaded.name)
        assert {"health": health["status"], "reason": health["last_observed"]["reason"],
                "error": health["last_observed"]["load_error"],
                "published": loaded.name in extension_loader._extensions,
                "events": _companion_events(drive_root)} == {
            "health": "broken", "reason": "load_error", "error": words, "published": False,
            "events": [("extension_companion_exhausted", 7, None), ("extension_companion_recovered", 7, "start")]}

        def listed():
            return next(row for row in _build_extensions_index(drive_root, str(repo_root))["skills"]
                        if row["name"] == loaded.name)

        row = listed()
        assert not row["live_loaded"] and row["live_reason"] == "load_error"
        assert row["last_known_good"] == health["last_known_good"] == good
        assert "spawn refused" in row["load_error"]
        monkeypatch.setattr(extension_companion.subprocess, "Popen", spawn)
        if caller == "queue":
            assert not health["companion_failures"]
            reads = [row]  # the open card only rereads on the server's hint

            def receive(frame):
                if frame == {"type": "extension_health_changed"}:
                    reads.append(listed())

            monkeypatch.setattr(ws, "broadcast_ws_sync", receive)
        assert reconcile()["action"] == "extension_load_error"  # reading a known load error is not a retry
        if caller == "queue":
            assert len(reads) == 1
            retry, = process_extension_reconcile_requests(drive_root, lambda: {}, repo_path=str(repo_root))
            assert not marker.exists()
            assert [(item["live_reason"], item["load_error"]) for item in reads] == [("load_error", words), ("ready", None)]
        else:
            retry = extension_loader.reconcile_extension(loaded.name, drive_root, lambda: {}, repo_path=str(repo_root),
                                                         retry_load_error=True)
        assert retry["action"] == "extension_loaded"
        row = listed()
        assert row["live_loaded"] and not row["load_error"]
        assert row["last_known_good"] and row["last_known_good"] != good
        assert extension_health.read_extension_health(drive_root, loaded.name)["status"] == "live"
        assert _running(supervisor, alpha) and _running(supervisor, beta)
    finally:
        _end_fake_companions(supervisor, spawned, loaded.name)


@pytest.mark.parametrize(("saved_on", "refusal"), [(True, "spawn"), (False, "spawn"), (True, "panic")],
                         ids=["repair_of_an_on_skill", "first_enable", "repair_under_panic"])
def test_a_failed_enable_keeps_an_on_switch_and_rolls_back_only_a_first_enable(
    tmp_path, monkeypatch, saved_on, refusal,
) -> None:
    """The publication is live and its companion gone in every case; only the switch saved before the attempt
    differs. Enable on a skill already On is a repair: its failure is answered and the switch stays On, so the next
    ordinary load starts it. The same failure of a first Enable saves Off again. Neither starts anything under
    Panic, and a later owner Off still wins."""
    from ouroboros import config, extension_companion
    from ouroboros.skill_lifecycle_actions import run_skill_action
    from ouroboros.skill_loader import load_enabled

    init_server_process_pid()
    monkeypatch.setattr(config, "load_settings", lambda *_args, **_kwargs: {})
    loaded, repo, drive = _reviewed_companion_skill(tmp_path)
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive)
    enabled_file, key = drive / "state" / "skills" / loaded.name / "enabled.json", f"{loaded.name}:daemon"
    spawn = extension_companion.subprocess.Popen

    def refuse_spawn(*_args, **_kwargs):
        if refusal == "panic":
            pytest.fail("a start reached its spawn after Panic")
        raise OSError("repair spawn refused")

    def act(action: str) -> dict:
        return run_skill_action(types.SimpleNamespace(drive_root=drive), loaded.name, action,
                                repo_path=str(repo), _owner_actor="owner_ui")

    def reload() -> dict:
        return extension_loader.reconcile_extension(loaded.name, drive, lambda: {}, repo_path=str(repo),
                                                    retry_load_error=True)

    try:
        assert extension_loader.reconcile_extension(loaded.name, drive, lambda: {}, repo_path=str(repo))["live_loaded"]
        spawned[0].end(0)
        _until(lambda: not supervisor.snapshot(), "the missing companion was not settled")
        if not saved_on:  # saved Off, its publication still loaded: live, yet a first Enable
            save_enabled(drive, loaded.name, False, actor="owner_ui")
        if refusal == "panic":
            supervisor.panic_kill_all()
        monkeypatch.setattr(extension_companion.subprocess, "Popen", refuse_spawn)
        answer = act("enable")
        saved = json.loads(enabled_file.read_text(encoding="utf-8"))
        assert (answer["ok"], answer["status_code"], answer["code"], answer["extension_action"]) == (
            False, 409, "load_error", "extension_load_error")
        assert ("start was declined" if refusal == "panic" else "repair spawn refused") in answer["load_error"]
        assert (answer["enabled"], saved["enabled"], saved["actor"]) == (
            (True, True, "owner_ui") if saved_on else (False, False, "load_error_revert"))
        retried = reload()
        assert (retried["live_loaded"], retried["reason"], len(spawned)) == (
            False, "load_error" if saved_on else "disabled", 1)
        if refusal == "spawn":
            monkeypatch.setattr(extension_companion.subprocess, "Popen", spawn)  # what refused the start is gone
            retried = reload()  # no second Enable
            assert (retried["live_loaded"], _running(supervisor, key), len(spawned)) == (
                (True, True, 2) if saved_on else (False, False, 1))
        off = act("disable")
        assert (off["ok"], off["enabled"], load_enabled(drive, loaded.name)) == (True, False, False)
        assert (reload()["reason"], supervisor.snapshot(), len(spawned)) == (
            "disabled", {}, 2 if saved_on and refusal == "spawn" else 1)
    finally:
        _end_fake_companions(supervisor, spawned, loaded.name)


def test_an_unfinished_repair_of_an_on_skill_answers_over_http_what_still_blocks_it(tmp_path, monkeypatch) -> None:
    """Alpha exhausted its restarts and beta awaits an unconfirmed exit, so Enable on the On skill starts nothing: the
    HTTP answer keeps it On and names both, as the listing does. Once beta's exit is observed the next Enable starts
    both, and its answer carries no word of either."""
    import asyncio

    import httpx
    from starlette.applications import Starlette
    from starlette.routing import Route

    from ouroboros import config
    from ouroboros.gateway import extensions
    from ouroboros.skill_loader import load_enabled

    init_server_process_pid()
    monkeypatch.setattr(config, "load_settings", lambda *_args, **_kwargs: {})
    loaded, repo, drive = _reviewed_companion_skill(tmp_path, companions=("alpha", "beta"))
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(repo))
    monkeypatch.setattr(extensions, "_broadcast_extension_lifecycle", lambda *_args: None)
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive)
    alpha, beta = f"{loaded.name}:alpha", f"{loaded.name}:beta"
    app = Starlette(routes=[Route("/api/skills/{skill}/toggle", extensions.api_skill_toggle, methods=["POST"])])
    app.state.drive_root, app.state.repo_dir = drive, tmp_path / "repo"

    async def enable() -> tuple[int, dict]:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            response = await client.post(f"/api/skills/{loaded.name}/toggle", json={"enabled": True})
        return response.status_code, response.json()

    try:
        assert extension_loader.reconcile_extension(loaded.name, drive, lambda: {}, repo_path=str(repo))["live_loaded"]
        _exhaust(supervisor, alpha)
        with monkeypatch.context() as held:
            held.setattr(supervisor, "_terminate_runtime", lambda *_args, **_kwargs: None)  # beta outlives its stop
            supervisor.stop(loaded.name, "beta", timeout_sec=0.1)
        exhausted = "companion 'alpha' exited with code 7 after exhausting its restart budget"
        status, answer = asyncio.run(enable())
        row = _listed_skill(drive, repo, loaded.name)
        assert (status, answer["enabled"], load_enabled(drive, loaded.name), answer["extension_action"],
                answer["extension_reason"], answer["load_error"], answer["retained_companions"], len(spawned)) == (
            200, True, True, "extension_already_live", "companion_restart_exhausted:alpha", exhausted,
            {"beta": "stop requested"}, 2)
        assert (row["live_reason"], row["load_error"], row["retained_companions"]) == (
            answer["extension_reason"], answer["load_error"], answer["retained_companions"])
        with supervisor._lock:
            supervisor._runtimes[beta].process.end(-15)
        _until(lambda: beta not in supervisor.snapshot(), "beta's exit was never settled")
        status, answer = asyncio.run(enable())
        assert (status, answer["enabled"], answer["extension_reason"], answer["load_error"],
                "retained_companions" in answer) == (200, True, "ready", None, False)
        assert _running(supervisor, alpha) and _running(supervisor, beta) and len(spawned) == 4
    finally:
        _end_fake_companions(supervisor, spawned, loaded.name)


@pytest.mark.parametrize("control", ["disabled", "companion_removed"])
@pytest.mark.parametrize("source", ["monitor", "legacy"])
def test_an_ended_publication_takes_its_companion_failures_with_it(
    tmp_path: pathlib.Path, monkeypatch, control: str, source: str,
) -> None:
    """Disabled, or updated to a payload without that companion, the extension no longer runs the companion that
    exhausted its restarts: the failure leaves with the publication that ran it instead of outliving it."""
    init_server_process_pid()
    loaded, repo_root, drive_root = _reviewed_companion_skill(tmp_path, companions=("alpha", "beta"))
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive_root)
    try:
        assert extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root, skills=[loaded]) is None
        legacy = {"ts": "2026-11-17T01:50:00Z", "version": "7.9.0", "sha": "a" * 40,
                  "status": "broken", "reason": "companion_restart_exhausted:alpha",
                  "load_error": "companion 'alpha' stopped after exhausting its restart budget", "process": "server"}
        if source == "legacy":
            # Only the old record witnessed alpha's exhaustion. Beta still runs.
            with supervisor._lock:
                supervisor._runtimes[f"{loaded.name}:alpha"].process.end(0)
            _until(lambda: f"{loaded.name}:alpha" not in supervisor.snapshot(), "alpha's exit was not settled")
            extension_health.health_path(drive_root, loaded.name).write_text(json.dumps({
                "skill": loaded.name, "status": "broken", "last_observed": legacy}), encoding="utf-8")
        else:
            _exhaust(supervisor, f"{loaded.name}:alpha")
        if control == "disabled":
            save_enabled(drive_root, loaded.name, False)
        else:  # a reviewed update whose manifest declares beta alone
            skill_dir = repo_root / loaded.name
            (skill_dir / "plugin.py").write_text("def register(api):\n    api.register_companion_process('beta')\n",
                                                 encoding="utf-8")
            manifest = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
            (skill_dir / "SKILL.md").write_text(manifest.replace(
                _companion_frontmatter(("alpha", "beta")), _companion_frontmatter(("beta",))), encoding="utf-8")
            updated = find_skill(drive_root, loaded.name, repo_path=str(repo_root))
            save_review_state(drive_root, loaded.name, SkillReviewState(status="pass", content_hash=updated.content_hash))
        state = extension_loader.reconcile_extension(loaded.name, drive_root, lambda: {}, repo_path=str(repo_root))
        health = extension_health.read_extension_health(drive_root, loaded.name)
        open_failure = extension_health.apply_companion_failure_to_runtime_state(
            {"desired_live": True, "reason": "ready"}, drive_root, loaded.name).get("companion_failed")
        assert {"action": state["action"], "running": sorted(row["name"] for row in supervisor.snapshot().values()),
                "open_failure": open_failure, "health": health["status"],
                "receipt": (state.get("companion_failed"), state["reason"], state.get("load_error"))} == {
            "disabled": {"action": "extension_unloaded", "running": [], "open_failure": None, "health": "inactive",
                         "receipt": (None, "disabled", None)},
            "companion_removed": {"action": "extension_loaded", "running": ["beta"], "open_failure": None,
                                  "health": "live", "receipt": (None, "ready", None)},
        }[control]
        extension_loader.reconcile_extension(loaded.name, drive_root, lambda: {}, repo_path=str(repo_root))
        history = extension_health.read_extension_health(drive_root, loaded.name)
        assert not history.get("companion_recovered") and not history["companion_failures"]
        if source == "legacy":
            assert history["skill"] == loaded.name and history.get("legacy_companion_failure") == legacy
            assert _companion_events(drive_root) == []
        else:
            assert _companion_events(drive_root) == [("extension_companion_exhausted", 7, None)]
    finally:
        _end_fake_companions(supervisor, spawned, loaded.name)


@pytest.mark.parametrize("registered", [("beta",), ()], ids=["alpha_removed", "none_registered"])
def test_a_failure_of_a_companion_the_loaded_publication_does_not_register_is_history(
    tmp_path: pathlib.Path, monkeypatch, registered: tuple,
) -> None:
    """Alpha exhausted its restarts; a reviewed update registers beta alone, or no companion, and the health write
    that would drop alpha's failure does not land. The reconcile's answer and the listing verified what the loaded
    publication registers (nothing included) and no longer name alpha; a read that verified nothing still shows
    what the record keeps, and the record and the journal stay as they were. A failure of a companion that is
    registered shows without the kept one."""
    init_server_process_pid()
    loaded, repo_root, drive_root = _reviewed_companion_skill(tmp_path, companions=("alpha", "beta"))
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive_root)
    real_update, unwritable = extension_health.update_json_locked, threading.Event()

    def update_json_locked(path, mutator, **kwargs):
        if unwritable.is_set():
            raise TimeoutError("the health lock stayed held")
        return real_update(path, mutator, **kwargs)

    monkeypatch.setattr(extension_health, "update_json_locked", update_json_locked)
    try:
        assert extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root, skills=[loaded]) is None
        _exhaust(supervisor, f"{loaded.name}:alpha")
        skill_dir = repo_root / loaded.name
        (skill_dir / "plugin.py").write_text("def register(api):\n" + "".join(
            f"    api.register_companion_process({name!r})\n" for name in registered) + "    pass\n", encoding="utf-8")
        manifest = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        (skill_dir / "SKILL.md").write_text(manifest.replace(
            _companion_frontmatter(("alpha", "beta")), _companion_frontmatter(registered) if registered else ""),
            encoding="utf-8")
        updated = find_skill(drive_root, loaded.name, repo_path=str(repo_root))
        save_review_state(drive_root, loaded.name, SkillReviewState(status="pass", content_hash=updated.content_hash))
        unwritable.set()
        state = extension_loader.reconcile_extension(loaded.name, drive_root, lambda: {}, repo_path=str(repo_root))
        row = _listed_skill(drive_root, repo_root, loaded.name)
        unverified = extension_health.apply_companion_failure_to_runtime_state(
            {"desired_live": True, "reason": "ready"}, drive_root, loaded.name)
        unwritable.clear()
        kept = extension_health.read_extension_health(drive_root, loaded.name)
        assert {"receipt": (state["action"], state.get("companion_failed"), state["reason"], state.get("load_error")),
                "listed": (row["live_loaded"], row["live_reason"], row["load_error"]),
                "unverified": (unverified.get("companion_failed"), unverified["reason"]),
                "record": sorted(kept["companion_failures"]), "journal": _companion_events(drive_root)} == {
            "receipt": ("extension_loaded", None, "ready", None), "listed": (True, "ready", None),
            "unverified": (True, "companion_restart_exhausted:alpha"),
            "record": ["alpha"], "journal": [("extension_companion_exhausted", 7, None)]}
        if registered:
            _exhaust(supervisor, f"{loaded.name}:beta", code=9)
            row = _listed_skill(drive_root, repo_root, loaded.name)
            kept = extension_health.read_extension_health(drive_root, loaded.name)
            assert (row["live_reason"], "code 9" in row["load_error"], sorted(kept["companion_failures"])) == (
                "companion_restart_exhausted:beta", True, ["alpha", "beta"])
    finally:
        _end_fake_companions(supervisor, spawned, loaded.name)


@pytest.mark.parametrize("companion", ["exhausted", "running"])
def test_a_replacement_failing_after_a_catalog_setting_change_leaves_nothing_live(
    tmp_path: pathlib.Path, monkeypatch, companion: str,
) -> None:
    """An isolated extension's registration reads a setting. The owner changes it and the replacement's catalog run
    fails: the reconcile unloaded the old publication and published none. Its receipt, its health and the Skills list
    name that load error with nothing loaded, and ``last_known_good`` stays where it was, whether the old
    publication's companion had exhausted its restarts or still ran."""
    from ouroboros import extension_process_runner as runner
    from ouroboros.gateway.extension_receipts import extension_reconcile_receipt
    from ouroboros.gateway.extensions import _build_extensions_index
    from tests._extension_loader_shared import _mark_isolated_deps_installed

    init_server_process_pid()
    loaded, repo_root, drive_root = _reviewed_companion_skill(tmp_path, extra_frontmatter=(
        "dependencies: [dummy_pkg]\ncatalog_inputs:\n  settings: [OUROBOROS_UI_LANGUAGE]\n  environment: []\n"))
    _mark_isolated_deps_installed(drive_root, loaded)
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive_root)
    settings, catalog_fails = {"OUROBOROS_UI_LANGUAGE": "one"}, threading.Event()

    def catalog_child(_payload, **_kwargs):  # the catalog child's protocol answer, without a process
        if catalog_fails.is_set():
            raise runner.ExtensionProcessError("register() raised")
        return {"ok": True, "content_hash": loaded.content_hash, "companions": ["daemon"]}

    monkeypatch.setattr(runner, "_run_child", catalog_child)

    def reconcile() -> dict:
        return extension_loader.reconcile_extension(loaded.name, drive_root, lambda: dict(settings),
                                                    repo_path=str(repo_root))

    try:
        assert reconcile()["action"] == "extension_loaded" and _running(supervisor, f"{loaded.name}:daemon")
        if companion == "exhausted":
            _exhaust(supervisor, f"{loaded.name}:daemon")
        known_good = extension_health.read_extension_health(drive_root, loaded.name)["last_known_good"]
        settings["OUROBOROS_UI_LANGUAGE"] = "two"
        catalog_fails.set()
        state = reconcile()
        health = extension_health.read_extension_health(drive_root, loaded.name)
        row = next(row for row in _build_extensions_index(drive_root, str(repo_root))["skills"]
                   if row["name"] == loaded.name)
        error = f"skill {loaded.name!r} out-of-process catalog failure: ExtensionProcessError: register() raised"
        assert {"receipt": extension_reconcile_receipt(loaded.name, state),
                "matches_current": state["loaded_matches_current"],
                "companion_failed": state.get("companion_failed"),
                "health": (health["status"], health["last_observed"]["reason"], health["last_observed"]["load_error"],
                           health["companion_failures"], health["regressed"]),
                "last_known_good": health["last_known_good"],
                "listed": (row["desired_live"], row["live_loaded"], row["live_reason"], row["load_error"],
                           row["last_known_good"], row["health_observations"]["server"]["status"]),
                "running": supervisor.snapshot()} == {
            "receipt": {"skill": loaded.name, "extension_action": "extension_load_error",
                        "extension_reason": "load_error", "live_loaded": False, "load_error": error,
                        "process": "server", "server_reconcile": ""},
            "matches_current": False,
            "companion_failed": None,
            "health": ("broken", "load_error", error, {}, False),
            "last_known_good": known_good,
            "listed": (True, False, "load_error", error, known_good, "broken"),
            "running": {}}
    finally:
        with extension_loader._lock:
            extension_loader._load_failures.pop(loaded.name, None)
        _end_fake_companions(supervisor, spawned, loaded.name)


_IMPORT_ERROR = "skill 'compskill' load failure: ImportError: boom"


@pytest.mark.parametrize("case", ["recovered_meanwhile", "still_exhausted", "import_error"])
def test_a_reload_deferred_by_a_live_call_keeps_its_own_words(tmp_path: pathlib.Path, monkeypatch, case: str) -> None:
    """A live call holds an isolated extension when the owner changes a setting its catalog reads, so the reconcile
    keeps the old publication and defers the reload (``live_call_active``). Its read showed the companion's exhausted
    restarts: when a settings refresh starts the companion again before the reconcile records, no word of that closed
    failure stays in the receipt, the health or the Skills list; while it stays open, the receipt keeps its deferral
    and shows the same words as the health and the list. A recorded import error is the state's own and keeps its
    words in all three."""
    from ouroboros import extension_process_runner as runner
    from ouroboros.extension_registry_state import extension_work_scope
    from ouroboros.gateway.extension_receipts import extension_reconcile_receipt
    from ouroboros.gateway.extensions import _build_extensions_index
    from tests._extension_loader_shared import _mark_isolated_deps_installed

    init_server_process_pid()
    loaded, repo_root, drive_root = _reviewed_companion_skill(tmp_path, extra_frontmatter=(
        "dependencies: [dummy_pkg]\ncatalog_inputs:\n  settings: [OUROBOROS_UI_LANGUAGE]\n  environment: []\n"))
    _mark_isolated_deps_installed(drive_root, loaded)
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive_root)
    key, settings, catalog_runs = f"{loaded.name}:daemon", {"OUROBOROS_UI_LANGUAGE": "one"}, []

    def catalog_child(_payload, **_kwargs):  # the catalog child's protocol answer, without a process
        catalog_runs.append(dict(settings))
        return {"ok": True, "content_hash": loaded.content_hash, "companions": ["daemon"]}

    monkeypatch.setattr(runner, "_run_child", catalog_child)
    real_announce = extension_loader._announce_extension_state_change

    def recovered_before_the_record(*args, **kwargs):  # after the reconcile's read, before its health write
        assert supervisor.refresh(loaded.name, "daemon", lambda: {"DEMO": "changed"}, compare_keys=["DEMO"],
                                  still_valid=lambda: True) == "started"
        return real_announce(*args, **kwargs)

    try:
        assert extension_loader.reconcile_extension(
            loaded.name, drive_root, lambda: dict(settings), repo_path=str(repo_root))["action"] == "extension_loaded"
        if case == "import_error":
            extension_loader.record_load_failure(loaded.name, loaded, _IMPORT_ERROR)
        else:
            _exhaust(supervisor, key)
        known_good = extension_health.read_extension_health(drive_root, loaded.name)["last_known_good"]
        with extension_loader._lock:
            bundle = extension_loader._extensions[loaded.name]
            bundle.conversation_features.append(f"{loaded.name}:call")
            generation = bundle.generation_digest
        settings["OUROBOROS_UI_LANGUAGE"] = "two"
        if case == "recovered_meanwhile":
            monkeypatch.setattr(extension_loader, "_announce_extension_state_change", recovered_before_the_record)
        with extension_work_scope({"skill": loaded.name, "extension_generation": generation},
                                  types.SimpleNamespace(cancel=lambda: None)):
            state = extension_loader.reconcile_extension(
                loaded.name, drive_root, lambda: dict(settings), repo_path=str(repo_root))
        monkeypatch.setattr(extension_loader, "_announce_extension_state_change", real_announce)
        health = extension_health.read_extension_health(drive_root, loaded.name)
        row = next(row for row in _build_extensions_index(drive_root, str(repo_root))["skills"]
                   if row["name"] == loaded.name)
        with extension_loader._lock:
            kept = extension_loader._extensions[loaded.name].generation_digest == generation
        failure = "companion 'daemon' exited with code 7 after exhausting its restart budget"
        words = {"recovered_meanwhile": None, "still_exhausted": failure, "import_error": _IMPORT_ERROR}[case]
        assert {"receipt": extension_reconcile_receipt(loaded.name, state),
                "companion_failed": state.get("companion_failed"),
                "old_publication": (kept, len(catalog_runs)), "running": _running(supervisor, key),
                "health": (health["status"], health["last_observed"]["reason"], health["last_observed"]["load_error"],
                           sorted(health["companion_failures"]), health["regressed"]),
                "last_known_good": health["last_known_good"],
                "listed": (row["live_reason"], row["load_error"], row["health_observations"]["server"]["load_error"])} == {
            "receipt": {"skill": loaded.name, "extension_action": "extension_reload_pending",
                        "extension_reason": "live_call_active", "live_loaded": True, "load_error": words,
                        "process": "server", "server_reconcile": ""},
            "companion_failed": True if case == "still_exhausted" else None,
            "old_publication": (True, 1), "running": case != "still_exhausted",
            "health": {"recovered_meanwhile": ("live", "live_call_active", "", [], False),
                       "still_exhausted": ("broken", "companion_restart_exhausted:daemon", failure, ["daemon"], False),
                       "import_error": ("live", "live_call_active", _IMPORT_ERROR, [], False)}[case],
            "last_known_good": known_good,
            "listed": {"recovered_meanwhile": ("ready", None, ""),
                       "still_exhausted": ("companion_restart_exhausted:daemon", failure, failure),
                       "import_error": ("load_error", _IMPORT_ERROR, _IMPORT_ERROR)}[case]}
    finally:
        with extension_loader._lock:
            extension_loader._load_failures.pop(loaded.name, None)
        _end_fake_companions(supervisor, spawned, loaded.name)


def test_a_publication_read_before_its_companions_spawn_confirms_nothing(tmp_path, monkeypatch) -> None:
    """The real load publishes alpha and beta, then starts them outside the registry lock. A listing read before
    each spawn finds a registered companion without a start of these files: background status is unconfirmed, not
    ready. Once both started the receipt and the listing read ready, and alpha's observed clean exit keeps it so."""
    init_server_process_pid()
    loaded, repo, drive = _reviewed_companion_skill(tmp_path, companions=("alpha", "beta"))
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive)
    start, samples = supervisor.start, []

    def before_spawn(descriptor, *args, **kwargs):
        row = _listed_skill(drive, repo, loaded.name)
        samples.append((descriptor.name, row["enabled"], row["live_loaded"], row["live_reason"], len(spawned)))
        return start(descriptor, *args, **kwargs)

    monkeypatch.setattr(supervisor, "start", before_spawn)
    try:
        receipt = extension_loader.reconcile_extension(loaded.name, drive, lambda: {}, repo_path=str(repo))
        assert samples == [("alpha", True, True, "health_unavailable", 0),
                           ("beta", True, True, "health_unavailable", 1)]
        assert (receipt["action"], receipt["reason"], len(spawned)) == ("extension_loaded", "ready", 2)
        assert _listed_skill(drive, repo, loaded.name)["live_reason"] == "ready"
        spawned[0].end(0)
        _until(lambda: f"{loaded.name}:alpha" not in supervisor.snapshot(), "alpha's clean exit was never settled")
        row = _listed_skill(drive, repo, loaded.name)
        assert (row["live_reason"], row["load_error"], len(spawned)) == ("ready", None, 2)
    finally:
        _end_fake_companions(supervisor, spawned, loaded.name)


def test_a_start_published_before_it_is_named_last_confirms_nothing(tmp_path, monkeypatch) -> None:
    """``start`` publishes its runtime, writes custody and only then names it the companion's last start. A listing
    read from that custody write finds a process no start claims yet: the first start reads unconfirmed, and a
    settings refresh's replacement, whose predecessor's start and claim are still current, stays pending, not ready.
    Named, the replacement reads ready. (The custody write swallows errors: the hook only records.)"""
    from ouroboros import process_custody

    init_server_process_pid()
    loaded, repo, drive = _reviewed_companion_skill(tmp_path)
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive)
    key, samples, phase = f"{loaded.name}:daemon", [], ["load"]

    def custody_write(*_args, **_kwargs):  # inside start(): its runtime published, its start not yet named last
        runtime = supervisor._runtimes.get(key)
        named = runtime is not None and runtime.descriptor is supervisor._last_descriptors.get(key)
        samples.append((phase[-1], named, key in supervisor._unreplaced,
                        _listed_skill(drive, repo, loaded.name)["live_reason"]))

    monkeypatch.setattr(process_custody, "record_process", custody_write)
    try:
        receipt = extension_loader.reconcile_extension(loaded.name, drive, lambda: {}, repo_path=str(repo))
        assert (receipt["action"], receipt["reason"]) == ("extension_loaded", "ready")
        phase.append("refresh")
        assert supervisor.refresh(loaded.name, "daemon", lambda: {"DEMO": "changed"}, compare_keys=["DEMO"],
                                  still_valid=lambda: True) == "restarted"
        assert samples == [("load", False, False, "health_unavailable"),
                           ("refresh", False, True, "companion_refresh_pending")]
        assert (_listed_skill(drive, repo, loaded.name)["live_reason"], len(spawned)) == ("ready", 2)
    finally:
        _end_fake_companions(supervisor, spawned, loaded.name)


@pytest.mark.parametrize(("case", "route"), [("held_failure", "reconcile"), ("held_failure", "http"),
                                             ("own_load_error", "reconcile")],
                         ids=["held_failure", "held_failure_http_retry", "own_load_error_without_retry"])
def test_a_reload_of_new_bytes_deferred_by_a_live_call_keeps_the_held_companion_s_words(
    tmp_path, monkeypatch, case: str, route: str,
) -> None:
    """A live call holds the publication whose companion exhausted its restarts when a reviewed payload with other
    bytes arrives: the reconcile defers (``live_call_active``) and the new revision is not the loaded one. The
    receipt keeps its deferral and shows the held companion's words as the listing does, also from the public
    ``POST /api/skills/{skill}/reconcile``, which retries load errors. Without that retry, a load error recorded for
    the new revision stays its own words (the projection; an explicit retry's own policy is not judged here)."""
    import asyncio

    import httpx
    from starlette.applications import Starlette
    from starlette.routing import Route

    from ouroboros import config
    from ouroboros.extension_registry_state import extension_work_scope
    from ouroboros.gateway import extensions
    from ouroboros.gateway.extension_receipts import extension_reconcile_receipt

    init_server_process_pid()
    loaded, repo, drive = _reviewed_companion_skill(tmp_path)
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive)
    monkeypatch.setattr(config, "load_settings", lambda *_args, **_kwargs: {})
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(repo))
    monkeypatch.setattr(extensions, "_broadcast_extension_lifecycle", lambda *_args: None)
    app = Starlette(routes=[Route("/api/skills/{skill}/reconcile", extensions.api_skill_reconcile, methods=["POST"])])
    app.state.drive_root = drive

    async def post() -> tuple[int, dict]:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            response = await client.post(f"/api/skills/{loaded.name}/reconcile")
        return response.status_code, response.json()

    try:
        assert extension_loader.reconcile_extension(loaded.name, drive, lambda: {}, repo_path=str(repo))["live_loaded"]
        _exhaust(supervisor, f"{loaded.name}:daemon")
        with extension_loader._lock:
            bundle = extension_loader._extensions[loaded.name]
            bundle.conversation_features.append(f"{loaded.name}:call")
            generation = bundle.generation_digest
        plugin = loaded.skill_dir / "plugin.py"
        plugin.write_text(plugin.read_text(encoding="utf-8") + "\n# new revision\n", encoding="utf-8")
        revised = find_skill(drive, loaded.name, repo_path=str(repo))
        assert revised.content_hash != loaded.content_hash
        save_review_state(drive, loaded.name, SkillReviewState(status="pass", content_hash=revised.content_hash))
        if case == "own_load_error":
            extension_loader.record_load_failure(loaded.name, revised, _IMPORT_ERROR)
        with extension_work_scope({"skill": loaded.name, "extension_generation": generation},
                                  types.SimpleNamespace(cancel=lambda: None)):
            if route == "http":  # the public route: retry_load_error=True
                state, (status, receipt) = {}, asyncio.run(post())
            else:
                state = extension_loader.reconcile_extension(loaded.name, drive, lambda: {}, repo_path=str(repo))
                status, receipt = 200, extension_reconcile_receipt(loaded.name, state)
        row = _listed_skill(drive, repo, loaded.name)
        with extension_loader._lock:
            kept = extension_loader._extensions[loaded.name].generation_digest == generation
        failure = "companion 'daemon' exited with code 7 after exhausting its restart budget"
        words = {"held_failure": failure, "own_load_error": _IMPORT_ERROR}[case]
        assert {"status": status, "receipt": receipt, "kept": kept, "companion_failed": state.get("companion_failed"),
                "listed": (row["live_reason"], row["load_error"])} == {
            "status": 200,
            "receipt": {"skill": loaded.name, "extension_action": "extension_reload_pending",
                        "extension_reason": "live_call_active", "live_loaded": False, "load_error": words,
                        "process": "server", "server_reconcile": ""},
            "kept": True, "companion_failed": {("held_failure", "reconcile"): True}.get((case, route)),
            "listed": {"held_failure": ("companion_restart_exhausted:daemon", failure),
                       "own_load_error": ("load_error", _IMPORT_ERROR)}[case]}
    finally:
        with extension_loader._lock:
            extension_loader._load_failures.pop(loaded.name, None)
        _end_fake_companions(supervisor, spawned, loaded.name)


def _js_typedef(path: pathlib.Path, name: str) -> str:
    text = path.read_text(encoding="utf-8")
    start = text.index(f"@typedef {{Object}} {name}\n")
    return text[start:text.index("*/", start)]


def test_grant_with_enable_and_the_named_catalogue_keep_the_observed_pending_exit(tmp_path, monkeypatch) -> None:
    """Alpha exhausted its restarts and beta awaits an unconfirmed exit. A grant carrying the owner's Enable, the named
    ``list_skills`` row and the listing name both facts alike; the catalogue only reads what the server observed,
    starting and polling nothing. The grant and refusal contracts declare the optional map in Python and JS."""
    from ouroboros import config
    from ouroboros.gateway import contracts, management_contracts
    from ouroboros.skill_catalogue import list_skills_payload
    from ouroboros.skill_lifecycle_actions import _grant
    from ouroboros.skill_loader import save_skill_grants, summarize_skills

    init_server_process_pid()
    loaded, repo, drive = _reviewed_companion_skill(tmp_path, companions=("alpha", "beta"),
                                                    extra_frontmatter="env_from_settings: [TELEGRAM_BOT_TOKEN]\n")
    save_skill_grants(drive, loaded.name, ["TELEGRAM_BOT_TOKEN"], content_hash=loaded.content_hash,
                      requested_keys=["TELEGRAM_BOT_TOKEN"])
    monkeypatch.setattr(config, "load_settings", lambda *_args, **_kwargs: {})
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(repo))
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive)
    waiting, exhausted = {"beta": "stop requested"}, "companion_restart_exhausted:alpha"
    try:
        assert extension_loader.reconcile_extension(loaded.name, drive, lambda: {}, repo_path=str(repo))["live_loaded"]
        _exhaust(supervisor, f"{loaded.name}:alpha")
        with monkeypatch.context() as held:
            held.setattr(supervisor, "_terminate_runtime", lambda *_args, **_kwargs: None)  # beta outlives its stop
            supervisor.stop(loaded.name, "beta", timeout_sec=0.1)
        row = _listed_skill(drive, repo, loaded.name)
        grant = _grant(loaded, drive, str(repo), ["TELEGRAM_BOT_TOKEN"], enable_after_grant=True)
        with monkeypatch.context() as passive:
            for proc in spawned:
                passive.setattr(proc, "poll", lambda: pytest.fail("the catalogue polled a companion"))
            passive.setattr(supervisor, "start", lambda *_args, **_kwargs: pytest.fail("the catalogue started one"))
            named = list_skills_payload(summarize_skills(drive, skills=[loaded]), name=loaded.name)["skills"][0]
        assert (grant["ok"], grant["enabled"], grant["extension_reason"], named["live_reason"], row["live_reason"],
                named["process"], len(spawned)) == (True, True, exhausted, exhausted, exhausted, "server", 2)
        assert (grant.get("retained_companions"), named.get("retained_companions"), row["retained_companions"]) == (
            waiting, waiting, waiting)
    finally:
        _end_fake_companions(supervisor, spawned, loaded.name)
    for typed in (contracts.SkillGrantResponse, management_contracts.SkillActionRefusal):
        assert "retained_companions" in typed.__annotations__, typed.__name__
    web = pathlib.Path(__file__).resolve().parents[1] / "web" / "modules"
    for path, name in ((web / "api_types.js", "SkillGrantResponse"),
                       (web / "management_contract_types.js", "SkillActionRefusal")):
        assert "@property {Object<string,string>=} retained_companions" in _js_typedef(path, name), name


def _pending_exit(skill_name: str, name: str, skill_dir) -> types.SimpleNamespace:
    """A registration whose stop is unconfirmed, as the supervisor holds it; no process stands behind it."""
    return types.SimpleNamespace(retiring="stop requested", death_known=False, descriptor=types.SimpleNamespace(
        skill_name=skill_name, name=name, skill_dir=str(skill_dir)))


@pytest.mark.parametrize("process", ["server", "worker"])
def test_the_named_catalogue_names_only_its_own_folder_s_pending_exit_without_runtime_state(
    tmp_path, monkeypatch, process: str,
) -> None:
    """An unreadable manifest and a name two folders claim have no runtime state. The server's named ``list_skills``
    gives each such row its own folder's pending exits, as the Skills listing does, beside the row's own error and
    with its observer; a namesake's folder and another skill in the same folder lend theirs to no row. Two folders of
    one origin read alike but for the listing's own ``collision_row_id``, which both surfaces carry. A worker holds
    no pending exits and makes none from the server's last record. Nothing is started and no runtime state is read."""
    from ouroboros import extension_companion
    from ouroboros.gateway.extensions import _build_extensions_index
    from ouroboros.skill_loader import collision_row_id
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.skill_exec import _handle_list_skills

    drive = tmp_path / "drive"
    broken = drive / "skills" / "external" / "broken"
    broken.mkdir(parents=True)
    (broken / "SKILL.md").write_text("---\nname: [\n---\n", encoding="utf-8")
    twins = [drive / "skills" / "twin", drive / "skills" / "external" / "twin"]  # one origin: both read ``external``
    for folder in twins:
        folder.mkdir(parents=True)
        (folder / "SKILL.md").write_text("---\nname: twin\ndescription: twin\nversion: 0.1.0\ntype: extension\n"
                                         "entry: plugin.py\npermissions: []\n---\nbody\n", encoding="utf-8")
        (folder / "plugin.py").write_text("def register(api):\n    pass\n", encoding="utf-8")
    supervisor = CompanionSupervisor(drive)
    supervisor._runtimes.update({
        "broken:bridge": _pending_exit("broken", "bridge", broken.resolve()),
        "broken:ghost": _pending_exit("broken", "ghost", tmp_path / "elsewhere" / "broken"),  # a namesake's folder
        "other:bridge": _pending_exit("other", "bridge", broken.resolve()),  # another name in this folder
        "twin:beta": _pending_exit("twin", "beta", twins[0].resolve()),
    })
    extension_health.record_extension_health(drive, "broken", status=extension_health.LIVE,
                                             retained_companions={"bridge": "stop requested"})  # the last record
    monkeypatch.setattr(extension_companion, "_GLOBAL_SUPERVISOR", supervisor)
    monkeypatch.setattr(extension_companion, "is_server_process", lambda: process == "server")
    monkeypatch.setattr(supervisor, "start", lambda *_args, **_kwargs: pytest.fail("the catalogue started one"))
    monkeypatch.setattr(extension_loader, "runtime_state_for_loaded_skill",
                        lambda *_args, **_kwargs: pytest.fail("a row without runtime state read one"))
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", "")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=drive)
    named = {name: json.loads(_handle_list_skills(ctx, name=name)) for name in ("broken", "twin")}
    listed = {(row["name"], row["source"], row.get("collision_row_id")): row.get("retained_companions")
              for row in _build_extensions_index(drive, "")["skills"]}
    rows = [*named["broken"]["skills"], *named["twin"]["skills"]]
    held, other = (collision_row_id(folder.resolve()) for folder in twins)
    broken_exit, twin_exit = ({"bridge": "stop requested"}, {"beta": "stop requested"}) if process == "server" else (
        None, None)
    assert {"codes": (named["broken"]["error"]["code"], named["twin"]["error"]["code"]),
            "errors": [row["load_error"].split(":")[0] for row in rows],
            "named": {(row["location"], row.get("collision_row_id")): (row.get("retained_companions"), row["process"])
                      for row in rows},
            "listed": listed} == {
        "codes": ("SKILL_MANIFEST_UNREADABLE", "SKILL_IDENTITY_COLLISION"),
        "errors": ["manifest parse error", "Skill name collision", "Skill name collision"],
        "named": {("external", None): (broken_exit, "server" if broken_exit else ""),
                  ("external", held): (twin_exit, "server" if twin_exit else ""), ("external", other): (None, "")},
        "listed": {("broken", "external", None): broken_exit, ("twin", "external", held): twin_exit,
                   ("twin", "external", other): None}}


@pytest.mark.parametrize("case", ["same_folder", "own_load_error", "other_folder", "other_folder_load_error",
                                  "other_folder_pending_exit"])
def test_a_held_publication_s_survived_start_reads_alike_on_get_and_on_its_deferred_reconcile(
    tmp_path, monkeypatch, case: str,
) -> None:
    """The companion's restart closed its exhaustion, but that close never reached the record (one lost write); then
    reviewed bytes of a new revision arrive while a live call holds the old publication. The listing reads that held
    publication's names and observed start as its deferred ``POST .../reconcile`` does: neither shows the closed
    failure or calls the new bytes live, and the record keeps the failure open (nothing closed it or started a
    companion). A load error of the new revision keeps its own words. When the name now comes from another folder,
    already On but not loaded, the held publication and the record it left speak for neither read: neither shows its
    failure or (a regression of the held folder included) its recorded health, the receipt keeps its deferred reason
    and, nothing vouching for the new folder's companions, the listing reads ``health_unavailable``, never ``ready``;
    that folder's own load error and its own pending exit stay. Its deferred reconcile writes nothing: the record and
    its history stay byte for byte through both reads and the reconcile, and a read that verifies nothing still shows
    the kept failure. A held publication of the same folder writes its own observation and lists its own record."""
    import asyncio

    import httpx
    from starlette.applications import Starlette
    from starlette.routing import Route

    from ouroboros import config
    from ouroboros.extension_registry_state import extension_work_scope
    from ouroboros.gateway import extensions
    from ouroboros.gateway.extension_receipts import extension_reconcile_receipt

    init_server_process_pid()
    loaded, repo, drive = _reviewed_companion_skill(tmp_path)
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive)
    key, failure = f"{loaded.name}:daemon", "companion 'daemon' exited with code 7 after exhausting its restart budget"
    monkeypatch.setattr(config, "load_settings", lambda *_args, **_kwargs: {})
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(repo))
    monkeypatch.setattr(extensions, "_broadcast_extension_lifecycle", lambda *_args: None)
    real_update, lose = extension_health.update_json_locked, threading.Event()

    def update_json_locked(path, mutator, **kwargs):
        if lose.is_set():
            lose.clear()
            raise TimeoutError("the health lock stayed held")
        return real_update(path, mutator, **kwargs)

    monkeypatch.setattr(extension_health, "update_json_locked", update_json_locked)
    app = Starlette(routes=[Route("/api/skills/{skill}/reconcile", extensions.api_skill_reconcile, methods=["POST"])])
    app.state.drive_root = drive

    async def post() -> tuple[int, dict]:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            response = await client.post(f"/api/skills/{loaded.name}/reconcile")
        return response.status_code, response.json()

    try:
        assert extension_loader.reconcile_extension(loaded.name, drive, lambda: {}, repo_path=str(repo))["live_loaded"]
        _exhaust(supervisor, key)
        lose.set()
        assert supervisor.refresh(loaded.name, "daemon", lambda: {"DEMO": "changed"}, compare_keys=["DEMO"],
                                  still_valid=lambda: True) == "started"
        assert not lose.is_set(), "the restart never tried to record its recovery"
        with extension_loader._lock:
            bundle = extension_loader._extensions[loaded.name]
            bundle.conversation_features.append(f"{loaded.name}:call")
            generation = bundle.generation_digest
        checkout = repo
        foreign = case.startswith("other_folder")
        if foreign:  # the name now comes from another checkout; the old folder's stays loaded
            # The held folder also broke after a code update: its record says so; the new folder's row must not.
            extension_health.record_extension_health(drive, loaded.name, status=extension_health.BROKEN,
                                                     version="test", sha="regressed-sha", reason="load_error",
                                                     load_error="the held folder broke after a code update")
            checkout = tmp_path / "other-checkout"
            shutil.copytree(loaded.skill_dir, checkout / loaded.name)
            monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(checkout))
        plugin = checkout / loaded.name / "plugin.py"
        plugin.write_text(plugin.read_text(encoding="utf-8") + "\n# new revision\n", encoding="utf-8")
        revised = find_skill(drive, loaded.name, repo_path=str(checkout))
        assert revised.content_hash != loaded.content_hash and revised.enabled
        save_review_state(drive, loaded.name, SkillReviewState(status="pass", content_hash=revised.content_hash))
        own_error = case.endswith("load_error")
        if own_error:
            extension_loader.record_load_failure(loaded.name, revised, _IMPORT_ERROR)
        if case == "other_folder_pending_exit":  # an earlier start of the new folder still awaits its exit
            supervisor._runtimes[f"{loaded.name}:ghost"] = _pending_exit(loaded.name, "ghost",
                                                                         revised.skill_dir.resolve())
        health_file, history = extension_health.health_path(drive, loaded.name), drive / "logs" / "events.jsonl"

        def stored() -> tuple[bytes, bytes]:  # the name's record and its history, byte for byte
            return health_file.read_bytes(), history.read_bytes()

        try:
            before = stored()
            first = _listed_skill(drive, checkout, loaded.name)
            read = stored()
            with extension_work_scope({"skill": loaded.name, "extension_generation": generation},
                                      types.SimpleNamespace(cancel=lambda: None)):
                if own_error:  # without the public route's retry the recorded load error stays
                    status, receipt = 200, extension_reconcile_receipt(
                        loaded.name, extension_loader.reconcile_extension(loaded.name, drive, lambda: {},
                                                                          repo_path=str(checkout)))
                else:  # the public route an owner's Reconcile posts
                    status, receipt = asyncio.run(post())
            reconciled = stored()
            row = _listed_skill(drive, checkout, loaded.name)
            reread = stored()
        finally:
            supervisor._runtimes.pop(f"{loaded.name}:ghost", None)
        record, kept = json.loads(reconciled[0]), json.loads(before[0])
        unverified = extension_health.apply_companion_failure_to_runtime_state(
            {"desired_live": True, "reason": "ready", "process": "server"}, drive, loaded.name)
        words, pending = (_IMPORT_ERROR if own_error else None), (
            {"ghost": "stop requested"} if case == "other_folder_pending_exit" else None)
        reason = {"same_folder": "ready", "other_folder": "health_unavailable",
                  "other_folder_pending_exit": "retained_unresolved"}.get(case, "load_error")

        def listed(view: dict) -> tuple:
            return (view["live_loaded"], view["live_reason"], view["load_error"], view.get("retained_companions"),
                    view["health_regressed"], (view["last_known_good"] or {}).get("sha"),
                    sorted(view["health_observations"]))

        shown = (False, reason, words, pending, False, *((None, []) if foreign else ("test-sha", ["server"])))
        assert {"listed": (listed(first), listed(row)),
                "receipt": (status, receipt["extension_action"], receipt["extension_reason"], receipt["live_loaded"],
                            receipt["load_error"], receipt.get("retained_companions"), receipt["process"]),
                "stored": (read == before, reconciled == before, reread == reconciled, reconciled[1] == before[1],
                           (record["companion_failures"], record.get("companion_recovered"))
                           == (kept["companion_failures"], kept.get("companion_recovered")), record["regressed"]),
                "record": sorted(record["companion_failures"]),
                "unverified": (unverified["reason"], unverified["load_error"]),
                "running": _running(supervisor, key), "spawned": len(spawned)} == {
            "listed": (shown, shown),
            "receipt": (200, "extension_reload_pending", "live_call_active", False, words, pending, "server"),
            "stored": (True, foreign, True, True, True, foreign),
            "record": ["daemon"], "unverified": ("companion_restart_exhausted:daemon", failure),
            "running": True, "spawned": 2}
    finally:
        with extension_loader._lock:
            extension_loader._load_failures.pop(loaded.name, None)
        _end_fake_companions(supervisor, spawned, loaded.name)


@pytest.mark.parametrize("seed", ["unregistered_failure", "pending_exit"])
@pytest.mark.parametrize("folder", ["own", "shared_name", "removed"])
def test_a_deferred_reconcile_writes_the_name_s_record_only_from_the_held_folder(
    tmp_path, monkeypatch, folder: str, seed: str,
) -> None:
    """A live call holds the publication, so its ``POST .../reconcile`` defers the reload. In the held
    publication's own folder it writes its observation: a recorded failure of a companion that publication does not
    register is the folder's history and leaves the record, as before, and the folder's observed pending exit stays
    in the receipt and the new observation. Once a second folder claims the name, or none does any longer, the
    reconcile can place the held publication in no folder: the receipt keeps its collision or missing-skill error
    and nothing is written, so the record (that failure, or the healthy observation the held folder's reconcile left
    with its pending exit) stays byte for byte."""
    import asyncio

    import httpx
    from starlette.applications import Starlette
    from starlette.routing import Route

    from ouroboros import config
    from ouroboros.extension_registry_state import extension_work_scope
    from ouroboros.gateway import extensions
    from ouroboros.utils import utc_now_iso

    init_server_process_pid()
    loaded, repo, drive = _reviewed_companion_skill(tmp_path)
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive)
    monkeypatch.setattr(config, "load_settings", lambda *_args, **_kwargs: {})
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(repo))
    monkeypatch.setattr(extensions, "_broadcast_extension_lifecycle", lambda *_args: None)
    app = Starlette(routes=[Route("/api/skills/{skill}/reconcile", extensions.api_skill_reconcile, methods=["POST"])])
    app.state.drive_root = drive
    lingering, waiting = f"{loaded.name}:lingering", {"lingering": "stop requested"}

    async def post() -> tuple[int, dict]:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            response = await client.post(f"/api/skills/{loaded.name}/reconcile")
        return response.status_code, response.json()

    def observed(record: dict) -> tuple:  # the server observation the name's record keeps
        server = record["last_observed"]
        return server["status"], server["reason"], server.get("retained_companions")

    try:
        assert extension_loader.reconcile_extension(loaded.name, drive, lambda: {}, repo_path=str(repo))["live_loaded"]
        if seed == "unregistered_failure":  # a name the loaded publication lacks
            extension_health.record_companion_exit(drive, loaded.name, "ghost", returncode=3,
                                                   observed_at=utc_now_iso(), still_current=lambda: True)
        else:  # an earlier companion of the held folder still awaits its exit: that folder's reconcile records it
            supervisor._runtimes[lingering] = _pending_exit(loaded.name, "lingering", loaded.skill_dir.resolve())
            assert extension_loader.reconcile_extension(loaded.name, drive, lambda: {}, repo_path=str(repo))[
                "retained_companions"] == waiting
        with extension_loader._lock:
            bundle = extension_loader._extensions[loaded.name]
            bundle.conversation_features.append(f"{loaded.name}:call")
            generation = bundle.generation_digest
        if folder == "own":  # reviewed new bytes in its folder: their reload waits for the call
            plugin = loaded.skill_dir / "plugin.py"
            plugin.write_text(plugin.read_text(encoding="utf-8") + "\n# new revision\n", encoding="utf-8")
            revised = find_skill(drive, loaded.name, repo_path=str(repo))
            save_review_state(drive, loaded.name, SkillReviewState(status="pass", content_hash=revised.content_hash))
        elif folder == "shared_name":  # a second folder now claims the name
            shutil.copytree(loaded.skill_dir, drive / "skills" / "external" / loaded.name)
        else:  # the held publication's folder is gone: no folder names the skill
            shutil.rmtree(loaded.skill_dir)
        health_file = extension_health.health_path(drive, loaded.name)
        before = health_file.read_bytes()
        with extension_work_scope({"skill": loaded.name, "extension_generation": generation},
                                  types.SimpleNamespace(cancel=lambda: None)):
            status, receipt = asyncio.run(post())
        after = health_file.read_bytes()
        record, own = json.loads(after), folder == "own"
        pending = waiting if seed == "pending_exit" else None
        seeded = {"unregistered_failure": ("broken", "companion_failed:ghost", None),
                  "pending_exit": ("live", "retained_unresolved", waiting)}[seed]
        assert {"receipt": (status, receipt["extension_action"], receipt["extension_reason"], receipt["live_loaded"],
                            str(receipt["load_error"] or "").split(":")[0], receipt.get("retained_companions"),
                            receipt["process"]),
                "seeded": observed(json.loads(before)), "stored": after == before, "observed": observed(record),
                "record": sorted(record["companion_failures"]),
                "running": _running(supervisor, f"{loaded.name}:daemon"), "spawned": len(spawned)} == {
            "receipt": (200, "extension_reload_pending", "live_call_active", folder == "removed",
                        {"own": "", "shared_name": "Skill name collision", "removed": "skill not found"}[folder],
                        pending if own else None, "server"),
            "seeded": seeded, "stored": not own,
            "observed": ("inactive", "live_call_active", pending) if own else seeded,
            "record": ["ghost"] if seed == "unregistered_failure" and not own else [],
            "running": True, "spawned": 1}
    finally:
        supervisor._runtimes.pop(lingering, None)
        _end_fake_companions(supervisor, spawned, loaded.name)


@pytest.mark.parametrize("case", ["on", "off", "absent_switch", "unreadable_switch", "non_boolean_switch",
                                  "required_off", "conflict", "unknown_error"])
def test_a_grant_still_short_of_its_requests_keeps_the_switch_and_this_folder_s_pending_exit(
    tmp_path, monkeypatch, case: str,
) -> None:
    """Saved Settings added a key the skill requests, so its companion was stopped and still awaits exit. The owner's
    Grant with Enable grants only the other key: the grant answers 200, and its Enable keeps the Toggle's typed
    refusal and words, before any write or reconcile. Whatever ended that Enable before an answer of its own (the
    ``grants`` precheck, another one, an unknown failure), the grant answers the saved switch as it reads, never the
    effective one: On or Off, no saved file at all is Off, a package a deployment requires is Off while saved Off,
    and it is left out while ``enabled.json`` cannot be read or holds no boolean. It keeps this folder's observed
    pending exit, never a namesake folder's."""
    import asyncio

    import httpx
    from starlette.applications import Starlette
    from starlette.routing import Route

    from ouroboros import config, deployment_extensions, skill_lifecycle_actions
    from ouroboros.gateway import extensions
    from ouroboros.skill_loader import save_skill_grants, skill_state_dir_path

    init_server_process_pid()
    keys, settings = ["TELEGRAM_BOT_TOKEN", "ACME_TOKEN"], {}
    loaded, repo, drive = _reviewed_companion_skill(tmp_path, companions=("alpha",),
                                                    extra_frontmatter=f"env_from_settings: {json.dumps(keys)}\n")
    monkeypatch.setattr(config, "load_settings", lambda *_args, **_kwargs: dict(settings))
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(repo))
    monkeypatch.setattr(extensions, "_broadcast_extension_lifecycle", lambda *_args: None)
    save_skill_grants(drive, loaded.name, keys[:1], content_hash=loaded.content_hash, requested_keys=keys)
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive)
    app = Starlette(routes=[Route("/api/skills/{skill}/grants", extensions.api_skill_grants, methods=["POST"]),
                            Route("/api/skills/{skill}/toggle", extensions.api_skill_toggle, methods=["POST"])])
    app.state.drive_root, app.state.repo_dir = drive, tmp_path / "repo-dir"

    async def post(route: str, body: dict) -> tuple[int, dict]:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            response = await client.post(f"/api/skills/{loaded.name}/{route}", json=body)
        return response.status_code, response.json()

    def forbidden(*_args, **_kwargs):
        pytest.fail("a refused Enable wrote, reconciled, started or stopped something")

    words = {"on": "Grant it the keys and permissions it asks for first.",
             "conflict": "Another skill that is on uses the same tool names; turn it off first.",
             "unknown_error": "Setup failed: RuntimeError: synthetic discovery failure"}
    short = ("off", "absent_switch", "unreadable_switch", "non_boolean_switch", "required_off")
    precheck = case == "on" or case in short  # the ``grants`` precheck refuses this Enable
    words.update(dict.fromkeys(short, words["on"]))
    switch, waiting = skill_state_dir_path(drive, loaded.name) / "enabled.json", {"alpha": "stop requested"}
    try:
        assert extension_loader.reconcile_extension(loaded.name, drive, lambda: {}, repo_path=str(repo))["live_loaded"]
        settings["ACME_TOKEN"] = "synthetic"  # a custom key saved in Settings joins the request, missing
        if case == "required_off":  # a deployment requires exactly these bytes: they read On over a saved Off
            required = types.SimpleNamespace(name=loaded.name, path=str(loaded.skill_dir),
                                             content_hash=loaded.content_hash, settings=())
            monkeypatch.setattr(deployment_extensions, "required_extensions", lambda: [required])
        if case in ("off", "required_off"):
            save_enabled(drive, loaded.name, False, actor="owner_ui")
        if case == "absent_switch":
            switch.unlink()  # never saved: Off by default
        with monkeypatch.context() as held:
            held.setattr(supervisor, "_terminate_runtime", lambda *_args, **_kwargs: None)  # alpha outlives its stop
            assert not extension_loader.reconcile_extension(loaded.name, drive, lambda: {}, repo_path=str(repo))[
                "desired_live"]
        assert supervisor.retained_companions(loaded.name, str(loaded.skill_dir)) == waiting
        if case == "unreadable_switch":
            switch.write_bytes(b'{"enabled": tru')  # the saved On is no longer readable; load_enabled reads Off
        if case == "non_boolean_switch":
            switch.write_bytes(b'{"enabled": "yes"}')  # a readable document whose switch is no boolean: also Off there
        saved = switch.read_bytes() if switch.exists() else None
        effective = find_skill(drive, loaded.name, repo_path=str(repo)).enabled  # what every switch reader sees
        with monkeypatch.context() as guarded:
            for name in ("start", "stop", "stop_skill"):
                guarded.setattr(supervisor, name, forbidden)
            guarded.setattr(extension_loader, "reconcile_extension", forbidden)
            guarded.setattr(skill_lifecycle_actions, "save_enabled", forbidden)
            if case == "conflict":
                guarded.setattr(skill_lifecycle_actions, "skill_conflict_status",
                                lambda *_args, **_kwargs: {"skill": "rival", "tools": ["shared"]})
            if case == "unknown_error":
                def unreadable(*_args, **_kwargs):
                    raise RuntimeError("synthetic discovery failure")

                guarded.setattr(skill_lifecycle_actions, "discover_skills", unreadable)
            supervisor._runtimes["compskill:ghost"] = _pending_exit(loaded.name, "ghost", tmp_path / "elsewhere")
            try:
                grant_status, grant = asyncio.run(post("grants", {
                    "items": keys[:1], "enable_after_grant": True, "expected_content_hash": loaded.content_hash}))
                toggled = asyncio.run(post("toggle", {"enabled": True})) if precheck else None
            finally:
                supervisor._runtimes.pop("compskill:ghost")
        assert {"grant": (grant_status, grant["ok"], grant["turn_on_when_ready"], grant["setup_error"],
                          grant["granted_keys"], grant["grants"]["missing_keys"], grant["extension_action"],
                          grant["live_loaded"], grant.get("retained_companions"), grant["process"]),
                "enabled": grant.get("enabled", "absent"), "effective": effective,
                "toggle": toggled and (toggled[0], toggled[1]["code"], toggled[1]["error"],
                                       sorted({"enabled", "retained_companions"} & set(toggled[1]))),
                "switch": (switch.read_bytes() if switch.exists() else None) == saved,
                "waiting": supervisor.retained_companions(loaded.name, str(loaded.skill_dir))} == {
            "grant": (200, True, "not_ready", words[case], keys[:1], keys[1:], None, None, waiting, "server"),
            "enabled": {"on": True, "off": False, "absent_switch": False, "unreadable_switch": "absent",
                        "non_boolean_switch": "absent", "required_off": False, "conflict": True,
                        "unknown_error": True}[case],
            "effective": case in ("on", "required_off", "conflict", "unknown_error"),
            "toggle": (409, "grants", words[case], []) if precheck else None,
            "switch": True, "waiting": waiting}
    finally:
        _end_fake_companions(supervisor, spawned, loaded.name)


def test_a_windows_start_under_the_fake_companion_harness_reaches_only_its_fake_jobs(
    tmp_path: pathlib.Path, monkeypatch,
) -> None:
    """The Windows shape, simulated: ``IS_WINDOWS`` on and no launcher, as ``tests/conftest.py`` leaves a native
    Windows run. Each start the harness serves gets its own fake Job holding exactly that start's synthetic pid, a stop
    ends the companion through that Job, and each death closes it once: no platform Job call sees a synthetic pid."""
    from ouroboros import extension_companion

    init_server_process_pid()
    monkeypatch.delenv("OUROBOROS_MANAGED_BY_LAUNCHER", raising=False)
    monkeypatch.setattr(extension_companion, "IS_WINDOWS", True)
    loaded, repo_root, drive_root = _reviewed_companion_skill(tmp_path)
    supervisor, spawned = _supervise_fake_companions(monkeypatch, drive_root)
    key = f"{loaded.name}:daemon"
    try:
        assert extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root, skills=[loaded]) is None
        _exhaust(supervisor, key)
        recovered = extension_loader.reconcile_extension(loaded.name, drive_root, lambda: {}, repo_path=str(repo_root))
        assert recovered["companions"]["action"] == "started_missing" and _running(supervisor, key)
        extension_loader.unload_extension(loaded.name)  # the running companion's stop goes through its Job
        _until(lambda: all(process.job.closed for process in spawned), "a dead companion's Job was never closed")
        assert [(process.job.member is process, process.returncode, process.job.closed) for process in spawned] == [
            (True, 7, 1), (True, -15, 1)]
    finally:
        _end_fake_companions(supervisor, spawned, loaded.name)


def test_publish_out_of_process_registration_host_spawns_declared_name(tmp_path: pathlib.Path, monkeypatch) -> None:
    """Out-of-process catalog -> host spawns the manifest-declared companion; an
    undeclared cataloged name is rejected at the host trust boundary, and an
    ambiguous publication form (neither initial load nor generation-bound
    recovery) is a typed refusal (Ф3.1 fix-round-4)."""
    init_server_process_pid()
    loaded, _repo_root, drive_root = _reviewed_companion_skill(tmp_path)
    fake = _patch_supervisor(monkeypatch)
    # Production-shape per-skill state dir (fix-round-5): the same directory
    # extension_loader resolves via skill_state_dir, where auth_token.json lives.
    state_dir = extension_loader.skill_state_dir(drive_root, loaded.name)
    try:
        with pytest.raises(ExtensionRegistrationError, match="exactly one of"):
            extension_loader._publish_out_of_process_registration(
                loaded,
                catalog={"companions": ["daemon"]},
                state_dir=state_dir,
                settings_reader=lambda: {},
                granted_keys=[],
                dependency_site_dirs_enabled=False,
            )
        with pytest.raises(ExtensionRegistrationError, match="escaped manifest"):
            extension_loader._publish_out_of_process_registration(
                loaded,
                catalog={"companions": ["evil"]},
                state_dir=state_dir,
                settings_reader=lambda: {},
                granted_keys=[],
                dependency_site_dirs_enabled=False,
                current_hash=loaded.content_hash,
            )
        assert fake.started == []
        assert not (state_dir / host_service.AUTH_TOKEN_FILENAME).exists(), (
            "a refused publication must not have minted a companion token"
        )

        extension_loader._publish_out_of_process_registration(
            loaded,
            catalog={"companions": ["daemon"]},
            state_dir=state_dir,
            settings_reader=lambda: {},
            granted_keys=[],
            dependency_site_dirs_enabled=False,
            current_hash=loaded.content_hash,
        )
        assert len(fake.started) == 1
        assert fake.started[0].name == "daemon"
        # Initial load legitimately mints the token — at publication, into the
        # spawned descriptor's env.
        token_payload = json.loads(
            (state_dir / host_service.AUTH_TOKEN_FILENAME).read_text(encoding="utf-8")
        )
        assert fake.started[0].env["HOST_SERVICE_TOKEN"] == token_payload["token"]
    finally:
        extension_loader.unload_extension("compskill")
        init_server_process_pid()


def test_recovery_publication_requires_a_pre_existing_live_bundle(tmp_path: pathlib.Path, monkeypatch) -> None:
    """Ф3.1 fix-round-4 REPLACEMENT (disclosed): the clause this test replaces
    pinned the OPPOSITE contract — the recovery-form publication helper
    accepted no pre-existing live bundle and spawned the companion anyway,
    which pinned the resurrection bug itself (a stale recovery re-created an
    empty companion-only bundle after disable/unload). The recovery form now
    REQUIRES a still-live bundle at the observed generation: with no live
    bundle it refuses with zero effects — filesystem included (fix-round-5:
    no auth_token.json is minted) — and creates nothing."""
    init_server_process_pid()
    loaded, _repo_root, drive_root = _reviewed_companion_skill(tmp_path)
    fake = _patch_supervisor(monkeypatch)
    state_dir = extension_loader.skill_state_dir(drive_root, loaded.name)
    try:
        with pytest.raises(ExtensionStaleRecoveryError):
            extension_loader._publish_out_of_process_registration(
                loaded,
                catalog={"companions": ["daemon"]},
                state_dir=state_dir,
                settings_reader=lambda: {},
                granted_keys=[],
                dependency_site_dirs_enabled=False,
                expected_generation="0" * 32,
            )
        assert fake.started == []
        with extension_loader._lock:
            assert "compskill" not in extension_loader._extensions
        assert "compskill" not in extension_loader.snapshot()["extensions"]
        assert not (state_dir / host_service.AUTH_TOKEN_FILENAME).exists(), (
            "stale recovery minted auth_token.json before its refusal"
        )
    finally:
        extension_loader.unload_extension("compskill")
        init_server_process_pid()


def test_unload_completing_between_snapshot_and_publication_refuses_recovery(
    tmp_path: pathlib.Path, monkeypatch,
) -> None:
    """Ф3.1 fix-round-4 pin (ABI-9а, TOCTOU): an unload that COMPLETES between
    recovery's liveness/bundle snapshot and its publication turns the
    publication into a typed zero-effect refusal — registries and supervisor
    stay empty and NO new bundle is created. (Before the fix the stale
    recovery re-created an empty companion-only bundle and started its
    companion, resurrecting the extension after disable/unload.)"""
    init_server_process_pid()
    loaded, repo_root, drive_root = _reviewed_companion_skill(tmp_path)
    fake = _patch_supervisor(monkeypatch)
    try:
        assert extension_loader.load_extension(
            loaded, lambda: {}, drive_root=drive_root, skills=[loaded],
        ) is None
        assert len(fake.started) == 1  # the initial publication's spawn
        # Fix-round-5: the token file the initial publication minted must
        # survive the stale refusal byte-for-byte (unload keeps it; only a
        # post-fence publication may touch it).
        token_path = (
            extension_loader.skill_state_dir(drive_root, loaded.name)
            / host_service.AUTH_TOKEN_FILENAME
        )
        token_bytes = token_path.read_bytes()

        real_state_path = extension_loader.skill_state_dir_path

        def _unload_wins_the_race(drive_root_arg, skill_name_arg):
            # Deterministic interleave: the concurrent unload COMPLETES after
            # recovery snapshotted liveness/bundle and before it publishes.
            extension_loader.unload_extension(skill_name_arg)
            return real_state_path(drive_root_arg, skill_name_arg)

        monkeypatch.setattr(extension_loader, "skill_state_dir_path", _unload_wins_the_race)
        result = extension_loader.ensure_companions_running(
            loaded.name, drive_root, lambda: {}, repo_path=str(repo_root),
        )

        assert result["action"] == "stale_recovery_refused"
        assert result["started"] == []
        assert "recovery publication refused" in result["reason"]
        assert len(fake.started) == 1, "stale recovery must not start a companion"
        with extension_loader._lock:
            assert loaded.name not in extension_loader._extensions
            assert not any(
                entry.get("skill") == loaded.name
                for entry in extension_loader._tools.values()
            )
        assert loaded.name not in extension_loader.snapshot()["extensions"]
        assert token_path.read_bytes() == token_bytes, (
            "stale recovery touched auth_token.json"
        )
    finally:
        extension_loader.unload_extension(loaded.name)
        init_server_process_pid()


def test_recovery_publication_refuses_on_generation_mismatch_without_effects(
    tmp_path: pathlib.Path, monkeypatch,
) -> None:
    """Ф3.1 fix-round-4 pin (ABI-9б), rebuilt through the REAL recovery entry
    in fix-round-5: an unload/reload that COMPLETES between recovery's
    generation snapshot and its publication leaves a live bundle under a NEW
    generation — the recovery publication naming the old generation is a
    typed zero-effect refusal: the reloaded bundle keeps its generation, no
    companion is started, and auth_token.json is byte-untouched."""
    init_server_process_pid()
    loaded, repo_root, drive_root = _reviewed_companion_skill(tmp_path)
    fake = _patch_supervisor(monkeypatch)
    try:
        assert extension_loader.load_extension(
            loaded, lambda: {}, drive_root=drive_root, skills=[loaded],
        ) is None
        first_generation = extension_loader.extension_generation_digest(loaded.name)
        assert first_generation
        token_path = (
            extension_loader.skill_state_dir(drive_root, loaded.name)
            / host_service.AUTH_TOKEN_FILENAME
        )
        token_bytes = token_path.read_bytes()

        real_state_path = extension_loader.skill_state_dir_path

        def _reload_wins_the_race(drive_root_arg, skill_name_arg):
            # Deterministic interleave: the unload/reload COMPLETES after
            # recovery snapshotted the generation and before it publishes;
            # the bundle EXISTS again, under a fresh generation. Restore the
            # real resolver first — the reload itself resolves state dirs.
            monkeypatch.setattr(extension_loader, "skill_state_dir_path", real_state_path)
            extension_loader.unload_extension(skill_name_arg)
            assert extension_loader.load_extension(
                loaded, lambda: {}, drive_root=drive_root_arg, skills=[loaded],
            ) is None
            return real_state_path(drive_root_arg, skill_name_arg)

        monkeypatch.setattr(extension_loader, "skill_state_dir_path", _reload_wins_the_race)
        result = extension_loader.ensure_companions_running(
            loaded.name, drive_root, lambda: {}, repo_path=str(repo_root),
        )

        assert result["action"] == "stale_recovery_refused"
        assert result["started"] == []
        second_generation = extension_loader.extension_generation_digest(loaded.name)
        assert second_generation and second_generation != first_generation
        with extension_loader._lock:
            assert loaded.name in extension_loader._extensions
        assert len(fake.started) == 2, (
            "only the initial load and the in-window reload may spawn"
        )
        assert token_path.read_bytes() == token_bytes
    finally:
        extension_loader.unload_extension(loaded.name)
        init_server_process_pid()


def test_stale_recovery_does_not_break_live_publication_authorization(
    tmp_path: pathlib.Path, monkeypatch,
) -> None:
    """Ф3.1 fix-round-5 pin (HIGH): a recovery holding a STALE payload snapshot
    (an old skill root that still exists with the pre-update content) loses
    the race to an unload/reload with CHANGED content; its typed refusal must
    leave auth_token.json byte-untouched. Pre-fix, the recovery minted the
    token DURING descriptor build — the stale root's content hash mismatched
    the G2-bound token file, so the mint ROTATED it before the generation
    fence refused the publication — and the live G2 companion, spawned with
    the current token in its env while the Host Service rereads the file on
    every request, was left permanently unauthorized."""
    init_server_process_pid()
    loaded_v1, repo_root, drive_root = _reviewed_companion_skill(tmp_path)
    fake = _patch_supervisor(monkeypatch)
    try:
        assert extension_loader.load_extension(
            loaded_v1, lambda: {}, drive_root=drive_root, skills=[loaded_v1],
        ) is None
        token_path = (
            extension_loader.skill_state_dir(drive_root, loaded_v1.name)
            / host_service.AUTH_TOKEN_FILENAME
        )
        # The update installs v2 under a NEW payload root; the G1 root
        # (repo_root) survives on disk with the v1 content — the recovery's
        # stale snapshot keeps pointing there.
        repo_root_v2 = tmp_path / "skills_v2"
        shutil.copytree(repo_root, repo_root_v2)
        (repo_root_v2 / loaded_v1.name / "scripts" / "daemon.py").write_text(
            "print('v2')\n", encoding="utf-8"
        )
        real_state_path = extension_loader.skill_state_dir_path
        g2_token_bytes: dict = {}

        def _reload_v2_wins_the_race(drive_root_arg, skill_name_arg):
            # Deterministic interleave: the unload + v2 reload COMPLETES after
            # recovery snapshotted generation/payload and before it publishes;
            # the G2 token is bound to the NEW root's content hash while the
            # stale recovery still holds the v1 snapshot at the old root.
            monkeypatch.setattr(extension_loader, "skill_state_dir_path", real_state_path)
            extension_loader.unload_extension(skill_name_arg)
            loaded_v2 = find_skill(
                drive_root, skill_name_arg, repo_path=str(repo_root_v2),
            )
            assert loaded_v2 is not None
            assert loaded_v2.content_hash != loaded_v1.content_hash
            save_review_state(
                drive_root, skill_name_arg,
                SkillReviewState(status="pass", content_hash=loaded_v2.content_hash),
            )
            loaded_v2 = find_skill(
                drive_root, skill_name_arg, repo_path=str(repo_root_v2),
            )
            assert extension_loader.load_extension(
                loaded_v2, lambda: {}, drive_root=drive_root_arg, skills=[loaded_v2],
            ) is None
            g2_token_bytes["value"] = token_path.read_bytes()
            return real_state_path(drive_root_arg, skill_name_arg)

        monkeypatch.setattr(extension_loader, "skill_state_dir_path", _reload_v2_wins_the_race)
        result = extension_loader.ensure_companions_running(
            loaded_v1.name, drive_root, lambda: {}, repo_path=str(repo_root),
            selected_skill=loaded_v1,  # the stale v1 snapshot the recovery holds
        )

        assert result["action"] == "stale_recovery_refused"
        assert token_path.read_bytes() == g2_token_bytes["value"], (
            "stale recovery rotated auth_token.json; the live publication's "
            "companion is now permanently unauthorized"
        )
        # The live G2 companion's spawn-env token still matches the file the
        # Host Service rereads per request...
        g2_env_token = fake.started[-1].env["HOST_SERVICE_TOKEN"]
        file_token = json.loads(token_path.read_text(encoding="utf-8"))["token"]
        assert hmac.compare_digest(file_token, g2_env_token)
        # ...and end-to-end host authorization for G2 still succeeds.
        # (find_skill is patched only because HostServiceContext resolves the
        # default skills repo path, not this test's temporary v2 root.)
        monkeypatch.setattr(
            host_service, "find_skill",
            lambda data_dir, name: find_skill(
                data_dir, name, repo_path=str(repo_root_v2),
            ),
        )
        ctx = host_service.HostServiceContext(data_dir=drive_root)
        authed_name, payload = ctx.authenticate_token_payload(g2_env_token)
        assert authed_name == loaded_v1.name
        assert payload["token"] == g2_env_token
    finally:
        extension_loader.unload_extension(loaded_v1.name)
        init_server_process_pid()


def _data_root_tree(root: pathlib.Path) -> list:
    # Names AND content identity (size + mtime): a rewrite-in-place of an
    # existing file is a mutation this snapshot must catch (fix-round-7).
    out = []
    for path in sorted(root.rglob("*")):
        stat = path.stat()
        out.append((str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns))
    return out


def test_stale_recovery_with_env_from_settings_has_zero_filesystem_effects(
    tmp_path: pathlib.Path, monkeypatch,
) -> None:
    """Ф3.1 fix-round-6 pin (MEDIUM), hardened in fix-round-7: after the
    recovery's state-dir resolution and before the generation fence there is
    no ``load_settings`` read at EITHER seam (``skill_exec.load_settings``
    used by ``_scrub_env`` and the direct ``config.load_settings`` used by
    the grant projection — the legitimate pre-fence grant/liveness projection
    calls land before the tripwire arms), no settings-lock file appears, and
    the data-root tree is identical by name, size and mtime. The
    settings-derived value, the manifest companion env overlay, the isolated
    PYTHONPATH, the state dir and the auth token materialize only in the
    post-fence attach — proved on the initial publication's spawn env."""
    init_server_process_pid()
    import ouroboros.config as config_mod
    import ouroboros.tools.skill_exec as skill_exec

    # Repoint settings beside this test so the lock-file assertion cannot race
    # a concurrent test process sharing the run-wide settings path.
    monkeypatch.setattr(config_mod, "SETTINGS_PATH", tmp_path / "settings.json")
    settings_calls: list = []
    real_load_settings = config_mod.load_settings

    def _counted_load_settings():
        settings_calls.append(1)
        settings = dict(real_load_settings())
        settings["EXT_DEMO_VALUE"] = "demo-from-settings"
        return settings

    # Both seams: skill_exec's bound reference (_scrub_env) and the direct
    # config attribute (requested_core_setting_keys and any other caller).
    monkeypatch.setattr(config_mod, "load_settings", _counted_load_settings)
    monkeypatch.setattr(skill_exec, "load_settings", _counted_load_settings)

    loaded, repo_root, drive_root = _reviewed_companion_skill(
        tmp_path, extra_frontmatter=(
            "    env:\n"
            "      EXT_OVERLAY: from-manifest\n"
            "env_from_settings: [EXT_DEMO_VALUE]\n"
        ),
    )
    # An isolated-dep site dir so the spawn env must carry PYTHONPATH
    # (.ouroboros_env is outside the content hash by construction).
    site_dir = loaded.skill_dir / ".ouroboros_env" / "python" / "lib" / "python{}.{}".format(*sys.version_info[:2]) / "site-packages"
    site_dir.mkdir(parents=True)
    # EXT_DEMO_VALUE present in settings = custom secret; grant it.
    from ouroboros.skill_loader import save_skill_grants

    save_skill_grants(
        drive_root, loaded.name, ["EXT_DEMO_VALUE"],
        content_hash=loaded.content_hash, requested_keys=["EXT_DEMO_VALUE"],
    )
    fake = _patch_supervisor(monkeypatch)
    lock_path = pathlib.Path(str(config_mod.SETTINGS_PATH) + ".lock")
    try:
        assert extension_loader.load_extension(
            loaded, lambda: {}, drive_root=drive_root, skills=[loaded],
        ) is None
        # The initial publication DID materialize the env — post-fence — and
        # actually DELIVERED every plane: the settings-derived value, the
        # manifest overlay, the isolated PYTHONPATH and the bridge keys.
        assert settings_calls, "publication must read settings for env_from_settings"
        env = fake.started[0].env
        assert env["OUROBOROS_SKILL_NAME"] == loaded.name
        assert env["HOST_SERVICE_TOKEN"]
        assert env["HOST_SERVICE_URL"].startswith("http://")
        assert env["EXT_DEMO_VALUE"] == "demo-from-settings"
        assert env["EXT_OVERLAY"] == "from-manifest"
        assert str(site_dir.resolve()) in (env.get("PYTHONPATH") or "")

        real_state_path = extension_loader.skill_state_dir_path
        snapshot: dict = {}

        def _unload_wins_and_snapshots(drive_root_arg, skill_name_arg):
            # Deterministic interleave (as in the round-4 pin), plus the
            # tripwires armed exactly at the recovery's state-dir resolution:
            # the measured window is "after this snapshot, up to the fence".
            extension_loader.unload_extension(skill_name_arg)
            settings_calls.clear()
            snapshot["tree"] = _data_root_tree(drive_root_arg)
            assert not lock_path.exists()
            return real_state_path(drive_root_arg, skill_name_arg)

        monkeypatch.setattr(extension_loader, "skill_state_dir_path", _unload_wins_and_snapshots)
        result = extension_loader.ensure_companions_running(
            loaded.name, drive_root, lambda: {}, repo_path=str(repo_root),
        )

        assert result["action"] == "stale_recovery_refused"
        assert settings_calls == [], (
            "stale recovery read settings pre-fence (lock file / migration hazard)"
        )
        assert not lock_path.exists(), "settings lock file left behind in the window"
        assert _data_root_tree(drive_root) == snapshot["tree"], (
            "stale recovery mutated the data root at/before its refusal"
        )
    finally:
        extension_loader.unload_extension(loaded.name)
        init_server_process_pid()


def test_transient_hash_error_never_rotates_a_valid_token(
    tmp_path: pathlib.Path, monkeypatch,
) -> None:
    """Ф3.1 fix-round-7 pin (LOW): a transient hash/read failure inside
    ``mint_skill_token`` reuses the stored valid token BYTE-FOR-BYTE instead
    of rotating it — the running companion whose spawn env holds that token
    stays authorized end-to-end. Pre-fix, any hash error collapsed to
    ``content_hash=\"\"``, mismatched the stored hash and rotated the live
    token, de-authorizing the running companion instantly."""
    init_server_process_pid()
    loaded, repo_root, drive_root = _reviewed_companion_skill(tmp_path)
    fake = _patch_supervisor(monkeypatch)
    try:
        assert extension_loader.load_extension(
            loaded, lambda: {}, drive_root=drive_root, skills=[loaded],
        ) is None
        env_token = fake.started[0].env["HOST_SERVICE_TOKEN"]
        state_dir = extension_loader.skill_state_dir_path(drive_root, loaded.name)
        token_path = state_dir / host_service.AUTH_TOKEN_FILENAME
        before = token_path.read_bytes()

        def _transient_failure(*_args, **_kwargs):
            raise OSError("transient read failure")

        monkeypatch.setattr(
            extension_plugin_api, "compute_content_hash", _transient_failure,
        )
        token = extension_plugin_api.mint_skill_token(
            state_dir, loaded.name, loaded.skill_dir,
        )
        assert token == env_token
        assert token_path.read_bytes() == before, (
            "transient hash error rotated a live valid token"
        )
        # End-to-end: the running companion's spawn-env token still authorizes.
        monkeypatch.setattr(
            host_service, "find_skill",
            lambda data_dir, name: find_skill(
                data_dir, name, repo_path=str(repo_root),
            ),
        )
        ctx = host_service.HostServiceContext(data_dir=drive_root)
        authed_name, payload = ctx.authenticate_token_payload(env_token)
        assert authed_name == loaded.name
        assert payload["token"] == env_token
    finally:
        extension_loader.unload_extension(loaded.name)
        init_server_process_pid()


def test_transient_hash_error_without_reusable_token_fails_closed(
    tmp_path: pathlib.Path, monkeypatch,
) -> None:
    """Ф3.1 fix-round-7 pin (LOW): with no reusable stored token, a transient
    hash failure is a typed fail-closed refusal — never a token minted
    against an empty content hash."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    def _transient_failure(*_args, **_kwargs):
        raise OSError("transient read failure")

    monkeypatch.setattr(
        extension_plugin_api, "compute_content_hash", _transient_failure,
    )
    with pytest.raises(extension_plugin_api.SkillTokenHashUnavailableError):
        extension_plugin_api.mint_skill_token(state_dir, "demo", tmp_path / "skill")
    assert not (state_dir / host_service.AUTH_TOKEN_FILENAME).exists()


def test_generation_bound_disposal_skips_a_newer_publication(
    tmp_path: pathlib.Path, monkeypatch, caplog,
) -> None:
    """Ф3.1 fix-round-4 pin (ABI-9в): disposal bound to an OLD generation is a
    disclosed no-op when the live publication is newer — recovery cleanup can
    never unload a publication it did not make. The same call naming the live
    generation unloads it."""
    init_server_process_pid()
    loaded, _repo_root, drive_root = _reviewed_companion_skill(tmp_path)
    _patch_supervisor(monkeypatch)
    try:
        assert extension_loader.load_extension(
            loaded, lambda: {}, drive_root=drive_root, skills=[loaded],
        ) is None
        live_generation = extension_loader.extension_generation_digest(loaded.name)
        assert live_generation

        with caplog.at_level(logging.WARNING, logger="ouroboros.extension_loader"):
            assert extension_loader.unload_extension(
                loaded.name, expected_generation="f" * 32,
            ) is False
        assert "generation-bound disposal skipped" in caplog.text
        with extension_loader._lock:
            assert loaded.name in extension_loader._extensions

        assert extension_loader.unload_extension(
            loaded.name, expected_generation=live_generation,
        ) is True
        with extension_loader._lock:
            assert loaded.name not in extension_loader._extensions
    finally:
        extension_loader.unload_extension(loaded.name)
        init_server_process_pid()

def _node_companion_api(tmp_path: pathlib.Path, monkeypatch, *, runtime: str, command: list[str],
                        env: dict | None = None) -> PluginAPIImpl:
    init_server_process_pid()
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir(exist_ok=True)

    class FakeSupervisor:
        def start(self, descriptor):
            return True

    monkeypatch.setattr(extension_plugin_api, "get_global_supervisor", lambda: FakeSupervisor())
    return PluginAPIImpl(_PluginAPIConfig(
        skill_name="demo",
        permissions=["companion_process"],
        env_allowlist=[],
        state_dir=tmp_path / "state",
        settings_reader=lambda: {},
        companion_processes=[{
            "name": "daemon",
            "command": command,
            "runtime": runtime,
            **({"env": env} if env else {}),
        }],
        skill_dir=skill_dir,
    ))


def _staged_companion_env(api: PluginAPIImpl) -> tuple:
    """The staged descriptor plus its env AFTER the post-fence materialization
    (the campaign structure defers env to `extension_companion.companion_spawn_env`)."""

    spawn = api._staged.companion_spawns[0]
    spawn.descriptor.env.update(companion_spawn_env(spawn.spec, "test-token", env_allow=api._env_allow, granted_upper=api._granted_upper, skill=api._skill, skill_dir=api._skill_dir, state_dir=api._state_dir))
    return spawn.descriptor, spawn.descriptor.env


def test_companion_node_command_rewritten_via_policy_helper(tmp_path: pathlib.Path, monkeypatch) -> None:
    """Node symmetry with the python->sys.executable rewrite: the policy helper
    (bundled-first + health rollback) owns the choice; a healthy PATH means no
    emergency, so the child env stays byte-identical."""
    import os

    from ouroboros import node_runtime

    selected = str(tmp_path / "bundle" / "bin" / "node")
    monkeypatch.setattr(
        node_runtime, "select_skill_node_runtime", lambda timeout_sec=10: (selected, "bundled")
    )
    monkeypatch.setattr(node_runtime, "skill_node_emergency_path_dir", lambda timeout_sec=10: "")
    api = _node_companion_api(tmp_path, monkeypatch, runtime="node", command=["node", "server.js"])

    api.register_companion_process("daemon")

    descriptor, env = _staged_companion_env(api)
    assert descriptor.command == [selected, "server.js"]
    assert env.get("PATH", os.environ["PATH"]) == os.environ["PATH"]


def test_companion_npm_not_rewritten_but_emergency_path_prepended(tmp_path: pathlib.Path, monkeypatch) -> None:
    """Without a pin npm is NOT rewritten: argv stays npm; in the emergency state (PATH node
    dead, bundled selected) the child PATH gains the bundled-node dir so npm's
    `#!/usr/bin/env node` shebang finds the working runtime. (Under the pin: ``reviewed_npm_command``.)"""
    import os

    from ouroboros import node_runtime

    monkeypatch.setattr(node_runtime, "pinned_node_wait", lambda: (False, ""))
    bundle_bin = str(tmp_path / "bundle" / "bin")
    monkeypatch.setattr(
        node_runtime,
        "select_skill_node_runtime",
        lambda timeout_sec=10: (str(pathlib.Path(bundle_bin) / "node"), "bundled"),
    )
    monkeypatch.setattr(
        node_runtime, "skill_node_emergency_path_dir", lambda timeout_sec=10: bundle_bin
    )
    api = _node_companion_api(tmp_path, monkeypatch, runtime="npm", command=["npm", "run", "start"])

    api.register_companion_process("daemon")

    descriptor, env = _staged_companion_env(api)
    assert descriptor.command == ["npm", "run", "start"]
    assert env["PATH"].split(os.pathsep)[0] == bundle_bin
    assert env["PATH"].endswith(os.environ["PATH"])


@pytest.mark.parametrize("command", [["node", "server.js"], ["npm", "run", "start"]])
@pytest.mark.parametrize("pinned", [False, True])
def test_companion_node_unusable_leaves_command_and_env_untouched_or_waits_for_the_pin(
        tmp_path: pathlib.Path, monkeypatch, pinned, command) -> None:
    """Nothing usable on a host the pin does not govern -> honest failure at spawn time, no rewrite, no env edit.
    Under the pin (owner 14 Nov 2026: functions require the new Node) the companion is not started on a bare PATH
    `node`, nor through `npm` (R7, Opus: its shebang found PATH node): registration fails naming why, and asks for the
    pin's Node once (the prepared Node re-queues it)."""
    import os

    from ouroboros import node_runtime, server_boot
    from ouroboros.contracts.plugin_api import ExtensionRegistrationError

    monkeypatch.setattr(
        node_runtime,
        "select_skill_node_runtime",
        lambda timeout_sec=10: ("", "pinned:24.22.3:not_prepared" if pinned
                                else "bundled:absent; path:broken:signal:SIGKILL"),
    )
    monkeypatch.setattr(node_runtime, "pinned_node_wait", lambda: (True, "/planned/node") if pinned else (False, ""))
    requests = []
    monkeypatch.setattr(server_boot, "request_pinned_node", lambda root=None: requests.append(root))
    monkeypatch.setattr(node_runtime, "skill_node_emergency_path_dir", lambda timeout_sec=10: "")
    api = _node_companion_api(tmp_path, monkeypatch, runtime=command[0], command=command)

    if pinned:
        with pytest.raises(ExtensionRegistrationError, match="pinned:24.22.3:not_prepared"):
            api.register_companion_process("daemon")
        assert requests == [None]
        return
    api.register_companion_process("daemon")

    descriptor, env = _staged_companion_env(api)
    assert descriptor.command == command and requests == []
    assert env.get("PATH", os.environ["PATH"]) == os.environ["PATH"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shebang resolution")
@pytest.mark.parametrize("case", ["pinned", "no_pin", "manifest_path"])
def test_an_npm_companions_node_shebang_runs_the_reviewed_node_under_the_pin(
        tmp_path: pathlib.Path, monkeypatch, case) -> None:
    """R7 (Opus): under the pin an `npm` companion still ran a healthy PATH `node` (only a broken PATH prepended the
    reviewed one). Real processes, npm's POSIX layout: `npm` links its `#!/usr/bin/env node` CLI. Under the pin the
    reviewed Node runs that CLI; without a pin, on a healthy PATH, the shebang finds PATH's; a manifest PATH stays the
    author's. A launcher whose CLI is not found is refused under the pin: its Node could be any."""
    import os

    from ouroboros import node_runtime

    def fake_node(where: pathlib.Path, label: str) -> pathlib.Path:
        where.mkdir(parents=True)
        node = where / "node"
        node.write_text('#!/bin/sh\nif [ "$1" = --version ]; then echo v24.22.3; else echo "ran:%s:$1"; fi\n' % label,
                        encoding="utf-8")
        node.chmod(0o755)
        return node

    pinned_node = fake_node(tmp_path / "pinned" / "bin", "pinned")
    path_bin = tmp_path / "path-bin"
    fake_node(path_bin, "path")
    cli = tmp_path / "lib" / "node_modules" / "npm" / "bin" / "npm-cli.js"
    cli.parent.mkdir(parents=True)
    cli.write_text("#!/usr/bin/env node\n", encoding="utf-8")  # npm's CLI: node runs this file
    cli.chmod(0o755)
    npm = path_bin / "npm"
    npm.symlink_to(cli)
    system = os.pathsep.join(["/usr/bin", "/bin"])
    monkeypatch.setenv("PATH", os.pathsep.join([str(path_bin), system]))
    # This test owns the host PATH: the real bootstrap prepends Homebrew's dirs, whose node may be SIGKILLed (an
    # emergency of its own, which would prepend the reviewed Node without any pin).
    monkeypatch.setattr("ouroboros.platform_layer.bootstrap_process_path", lambda *a, **k: None)
    monkeypatch.setattr("ouroboros.platform_layer.resolve_bundled_node", lambda: str(pinned_node))
    monkeypatch.setattr(node_runtime, "pinned_node_wait",
                        lambda: (False, "") if case == "no_pin" else (True, str(pinned_node)))
    api = _node_companion_api(tmp_path, monkeypatch, runtime="npm", command=["npm", "start"], env={
        "PATH": os.pathsep.join([str(path_bin), system])} if case == "manifest_path" else None)

    api.register_companion_process("daemon")
    descriptor, env = _staged_companion_env(api)
    pinned = case == "pinned"
    assert descriptor.command == ([str(pinned_node), str(cli.resolve()), "start"] if pinned else ["npm", "start"])
    ran = subprocess.run(descriptor.command, env=env, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.strip() == (f"ran:pinned:{cli.resolve()}" if pinned else f"ran:path:{npm}")
    if pinned:
        from ouroboros.contracts.plugin_api import ExtensionRegistrationError

        npm.unlink()
        npm.write_text("#!/bin/sh\nexec /an/older/node \"$0\" \"$@\"\n", encoding="utf-8")  # opaque: no CLI found
        npm.chmod(0o755)
        (tmp_path / "opaque").mkdir()
        again = _node_companion_api(tmp_path / "opaque", monkeypatch, runtime="npm", command=["npm", "start"])
        with pytest.raises(ExtensionRegistrationError, match="would choose its own Node"):
            again.register_companion_process("daemon")




@pytest.mark.parametrize("layout", ["official", "shim"])
def test_an_npm_companion_runs_its_cli_on_the_reviewed_node_or_is_refused_without_asking_again(
        tmp_path: pathlib.Path, monkeypatch, layout) -> None:
    """Under the pin a Windows ``npm.cmd`` ran the ``node.exe`` beside it, not the reviewed Node the PATH prepend put
    first. The companion now starts on ``[reviewed node, npm-cli.js]``; a launcher whose CLI is not found is refused,
    and that refusal asks nothing: the prepared Node re-queuing this extension would only meet it again."""
    from ouroboros import node_runtime, server_boot
    from ouroboros.contracts.plugin_api import ExtensionRegistrationError

    pinned = str(tmp_path / "pinned" / "bin" / "node")
    host = tmp_path / "nodejs"
    host.mkdir()
    launcher = host / "npm.cmd"
    launcher.write_text("@ECHO OFF\n", encoding="utf-8")
    cli = host / "node_modules" / "npm" / "bin" / "npm-cli.js"
    if layout == "official":
        cli.parent.mkdir(parents=True)
        cli.write_text("// npm\n", encoding="utf-8")
    monkeypatch.setattr(node_runtime, "select_skill_node_runtime", lambda timeout_sec=10: (pinned, "bundled"))
    monkeypatch.setattr(node_runtime, "pinned_node_wait", lambda: (True, pinned))
    monkeypatch.setattr(node_runtime, "skill_node_emergency_path_dir", lambda timeout_sec=10: "")
    monkeypatch.setattr(node_runtime._platform, "IS_WINDOWS", True)
    monkeypatch.setattr(node_runtime._platform, "bootstrap_process_path", lambda *a, **k: None)
    monkeypatch.setattr(node_runtime.shutil, "which", lambda name, *a, **k: str(launcher) if name == "npm" else None)
    requests = []
    monkeypatch.setattr(server_boot, "request_pinned_node", lambda root=None: requests.append(root))
    api = _node_companion_api(tmp_path, monkeypatch, runtime="npm", command=["npm", "start"])

    if layout == "shim":
        with pytest.raises(ExtensionRegistrationError, match="would choose its own Node"):
            api.register_companion_process("daemon")
        assert requests == []
        return
    api.register_companion_process("daemon")
    descriptor, _env = _staged_companion_env(api)
    assert descriptor.command == [pinned, str(cli), "start"] and requests == []


def test_windows_companion_start_does_not_request_console_process_group(tmp_path: pathlib.Path, monkeypatch) -> None:
    init_server_process_pid()
    captured = {}
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-should-not-leak")
    monkeypatch.setenv("windir", "C:\\Windows")
    monkeypatch.setenv("ComSpec", "C:\\Windows\\System32\\cmd.exe")

    class FakeProcess:
        pid = 12345
        stdout = None
        stderr = None

        def poll(self):
            return None

        def wait(self):
            return 0

    monkeypatch.setattr("ouroboros.extension_companion.IS_WINDOWS", True)
    monkeypatch.setattr("ouroboros.extension_companion.create_kill_on_close_job", lambda: object())
    monkeypatch.setattr("ouroboros.extension_companion.assign_pid_to_job", lambda _job, _pid: True)
    monkeypatch.setattr("ouroboros.extension_companion.close_job", lambda _job: None)

    def fake_popen(command, **kwargs):
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr("ouroboros.extension_companion.subprocess.Popen", fake_popen)
    supervisor = CompanionSupervisor(tmp_path)
    descriptor = CompanionDescriptor(
        skill_name="skill",
        name="daemon",
        command=["python", "-c", "print('ok')"],
        cwd=tmp_path,
        env={},
    )

    assert supervisor.start(descriptor) is True
    assert captured.get("creationflags", 0) & getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) == 0
    captured_upper = {key.upper(): value for key, value in captured["env"].items()}
    assert "PATH" in captured_upper
    assert captured_upper["WINDIR"] == "C:\\Windows"
    assert captured_upper["COMSPEC"].endswith("cmd.exe")
    assert "OPENROUTER_API_KEY" not in captured["env"]


def test_companion_processes_start_outside_the_registry_lock_after_publication(tmp_path: pathlib.Path, monkeypatch) -> None:
    """A companion spawn forks a child and probes its ports — seconds, under
    load. Every other extension call (dispatch, event delivery, a sibling
    skill's load) serializes on the registry lock, so a spawn held UNDER it
    stalled the whole extension surface (the Telegram companion wedge). The
    process starts after the lock is released, against the already-published
    bundle; a start failure still routes to the standard dispose+unload path."""
    init_server_process_pid()
    staged_skill = tmp_path / "import" / "skill"
    (staged_skill / "scripts").mkdir(parents=True)
    (staged_skill / "scripts" / "daemon.py").write_text("print('ok')\n", encoding="utf-8")
    observed = {}

    class LockProbeSupervisor:
        def __init__(self, fail=False):
            self.fail = fail

        def start(self, descriptor):
            free = extension_plugin_api._lock.acquire(blocking=False)
            if free:
                extension_plugin_api._lock.release()
            observed["lock_free_at_start"] = free
            observed["published_at_start"] = extension_plugin_api._extensions.get("demo") is not None
            if self.fail:
                raise RuntimeError("spawn failed")
            return True

        def stop(self, skill_name, name):
            return True

    def _api(supervisor):
        monkeypatch.setattr(extension_plugin_api, "get_global_supervisor", lambda: supervisor)
        monkeypatch.setattr(extension_loader, "get_global_supervisor", lambda: supervisor)
        api = PluginAPIImpl(_PluginAPIConfig(
            skill_name="demo",
            permissions=["companion_process"],
            env_allowlist=[],
            state_dir=tmp_path / "state",
            settings_reader=lambda: {},
            companion_processes=[{
                "name": "daemon",
                "command": ["python3", "scripts/daemon.py"],
                "runtime": "python3",
            }],
            skill_dir=tmp_path / "mutable",
            runtime_skill_dir=staged_skill,
        ))
        api.register_companion_process("daemon")
        return api

    api = _api(LockProbeSupervisor())
    api._publish_registrations()
    assert observed["lock_free_at_start"] is True
    assert observed["published_at_start"] is True

    with extension_plugin_api._lock:
        extension_plugin_api._extensions.pop("demo", None)
    failing = _api(LockProbeSupervisor(fail=True))
    with pytest.raises(RuntimeError, match="spawn failed"):
        failing._publish_registrations()


def test_companion_base_env_forwards_login_identity_without_secrets(monkeypatch) -> None:
    """The login name reaches a companion (its CLIs key keychain/credential lookups on it); a
    secret-shaped sibling that merely starts the same way does not."""
    from ouroboros.extension_companion import _companion_base_env

    for key in ("USER", "LOGNAME", "USERNAME"):
        monkeypatch.setenv(key, "synthetic-login")
    monkeypatch.setenv("USER_API_TOKEN", "synthetic-host-only")
    env = _companion_base_env()
    assert env["USER"] == env["LOGNAME"] == env["USERNAME"] == "synthetic-login"
    assert "USER_API_TOKEN" not in env


# --- settings refresh of a declared companion (PluginAPI 2.4) -----------------


def _sleeper(tmp_path: pathlib.Path, env: dict) -> CompanionDescriptor:
    return CompanionDescriptor(skill_name="demo", name="daemon", cwd=tmp_path, env=dict(env),
                               command=[sys.executable, "-c", "import time; time.sleep(60)"])


@pytest.mark.parametrize("intervening", ["refresh", "stop", "stop_skill", "stop_all", "panic", "unchanged"])
def test_delayed_crash_restart_obeys_the_current_descriptor(tmp_path, monkeypatch, intervening):
    """An old monitor resumes only its still-current registration, including when both generations crash.

    Hold the real monitor at its backoff (before its final admission), not inside start's lock. Real
    children record their token and exit on a private marker; neither timing nor a fake PID decides it.
    The unchanged case proves ordinary on-failure restart still works.
    """
    import ouroboros.extension_companion as companion

    init_server_process_pid()
    entered = {key: threading.Event() for key in ("A", "B")}
    release = {key: threading.Event() for key in ("A", "B")}
    monitors, errors = [], []
    prefix = "descriptor-restart-"
    real_time = companion.time

    def backoff(seconds):
        thread = threading.current_thread()
        if seconds == 0.5 and thread.name.startswith(prefix):
            token = thread.name.removeprefix(prefix)
            entered[token].set()
            assert release[token].wait(30), "the test did not release the monitor"
        else:
            real_time.sleep(seconds)

    monkeypatch.setattr(companion, "time", types.SimpleNamespace(monotonic=real_time.monotonic, sleep=backoff))
    supervisor = CompanionSupervisor(tmp_path)

    def observe_monitor(key, runtime):
        def run():
            try:
                supervisor._monitor_runtime(key, runtime)
            except BaseException as exc:
                errors.append(exc)
        thread = threading.Thread(target=run, name=prefix + runtime.descriptor.env["TOKEN"], daemon=True)
        monitors.append(thread)
        thread.start()

    monkeypatch.setattr(supervisor, "_start_monitor", observe_monitor)
    child = tmp_path / "companion.py"
    child.write_text(
        "import os, sys, time\nfrom pathlib import Path\n"
        "root = Path(sys.argv[1]); token = os.environ['TOKEN']\n"
        "(root / ('started-' + str(os.getpid()))).write_text(token)\n"
        "while not (root / ('crash-' + token)).exists(): time.sleep(.01)\n"
        "raise SystemExit(7)\n", encoding="utf-8")
    descriptor = CompanionDescriptor(skill_name="demo", name="daemon", cwd=tmp_path, env={"TOKEN": "A"},
                                     command=[sys.executable, str(child), str(tmp_path)])
    try:
        assert supervisor.start(descriptor)
        (tmp_path / "crash-A").touch()
        assert entered["A"].wait(30), "A never reached the delayed restart"
        (tmp_path / "crash-A").unlink()
        assert not supervisor.snapshot()
        if intervening == "refresh":
            assert supervisor.refresh("demo", "daemon", lambda: {"TOKEN": "B"}, compare_keys=["TOKEN"],
                                      still_valid=lambda: True) == "started"
            (tmp_path / "crash-B").touch()
            assert entered["B"].wait(30), "B never reached the delayed restart"
            (tmp_path / "crash-B").unlink()
        elif intervening == "stop":
            supervisor.stop("demo", "daemon")
        elif intervening == "stop_skill":
            supervisor.stop_skill("demo")
        elif intervening == "stop_all":
            supervisor.stop_all()
        elif intervening == "panic":
            supervisor.panic_kill_all()
        release["A"].set()
        monitors[0].join(30)
        assert not monitors[0].is_alive() and not errors
        if intervening != "unchanged":
            assert not supervisor.snapshot(), "an obsolete monitor restarted A"
        if intervening == "refresh":
            release["B"].set()
            monitors[1].join(30)
            assert not monitors[1].is_alive() and not errors
        if intervening in ("refresh", "unchanged"):
            current = supervisor._runtimes["demo:daemon"]
            expected = "B" if intervening == "refresh" else "A"
            assert current.descriptor.env["TOKEN"] == expected and current.process.poll() is None
            marker = tmp_path / f"started-{current.process.pid}"
            deadline = real_time.monotonic() + 30
            while not marker.exists():
                assert real_time.monotonic() < deadline, "the restarted child did not announce itself"
                real_time.sleep(.01)
            assert marker.read_text() == expected
    finally:
        for event in release.values():
            event.set()
        supervisor.panic_kill_all()
        for thread in monitors:
            thread.join(30)
        assert all(not thread.is_alive() for thread in monitors), "a test monitor was left running"


@pytest.mark.skipif(sys.platform == "win32", reason="on Windows the real venv redirector meets the tests above")
def test_the_started_companion_is_the_interpreter_even_behind_a_venv_redirector(tmp_path, monkeypatch):
    """A Windows venv's ``python.exe`` runs the base interpreter as its child, so the supervisor's PID (its Job,
    kill and the published-PID checks above) would name the redirector. A POSIX stand-in for that redirector:
    the supervisor starts the interpreter itself and names the venv; off Windows the command is left alone."""
    import functools
    import shlex

    import ouroboros.extension_companion as companion

    init_server_process_pid()
    redirector = tmp_path / "venv" / "Scripts" / "python"
    redirector.parent.mkdir(parents=True)
    redirector.write_text(f'#!/bin/sh\n__PYVENV_LAUNCHER__="$0" {shlex.quote(sys.executable)} "$@"\nexit $?\n',
                          encoding="utf-8")
    redirector.chmod(0o755)
    child = tmp_path / "child.py"
    child.write_text("import os, sys, time\nfrom pathlib import Path\n"
                     "Path(sys.argv[1]).write_text(str(os.getpid()))\n"
                     "time.sleep(60)\n", encoding="utf-8")
    real = companion._interpreter_command
    assert real([str(redirector), "x"], windows=True, executable=str(redirector), base=sys.executable) == (
        [sys.executable, "x"], {"__PYVENV_LAUNCHER__": str(redirector)})
    assert real([str(redirector), "x"], windows=False, executable=str(redirector), base=sys.executable) == (
        [str(redirector), "x"], {})
    assert real(["node", "x"], windows=True, executable=str(redirector), base=sys.executable) == (["node", "x"], {})
    for windows in (True, False):
        monkeypatch.setattr(companion, "_interpreter_command", functools.partial(
            real, windows=windows, executable=str(redirector), base=sys.executable))
        supervisor, marker = CompanionSupervisor(tmp_path), tmp_path / f"announced-{windows}"
        try:
            assert supervisor.start(CompanionDescriptor(skill_name="demo", name="daemon", cwd=tmp_path, env={},
                                                        command=[str(redirector), str(child), str(marker)]))
            process = supervisor._runtimes["demo:daemon"].process
            deadline = time.monotonic() + 30
            while not marker.exists() or not marker.read_text():
                assert time.monotonic() < deadline, "the companion never announced itself"
                time.sleep(.01)
            assert (int(marker.read_text()) == process.pid) is windows, "the supervisor must hold the companion itself"
        finally:
            supervisor.stop_all(timeout_sec=3)


def test_supervisor_refresh_compares_settings_keys_and_respawns_only_after_observed_death(tmp_path, monkeypatch):
    """Only the settings-derived keys decide: a changed bridge token, PATH or PYTHONPATH is no change. A
    change stops the old process, observes its death, builds the environment again (the save made while it
    stopped wins: A→B→C) and starts once. Neither the restart history nor a second refresh resets anything."""
    init_server_process_pid()
    supervisor = CompanionSupervisor(tmp_path)
    try:
        assert supervisor.start(_sleeper(tmp_path, {"TOKEN": "A", "HOST_SERVICE_TOKEN": "h1", "PATH": "/a"}))
        unrelated = CompanionDescriptor(skill_name="other", name="daemon", cwd=tmp_path, env={"TOKEN": "A"},
                                        command=[sys.executable, "-c", "import time; time.sleep(60)"])
        assert supervisor.start(unrelated)
        other = supervisor._runtimes["other:daemon"].process
        first = supervisor._runtimes["demo:daemon"].process
        same = {"TOKEN": "A", "HOST_SERVICE_TOKEN": "h2", "PATH": "/b", "PYTHONPATH": "/x"}
        assert supervisor.refresh("demo", "daemon", lambda: same, compare_keys=["TOKEN"],
                                  still_valid=lambda: True) == "unchanged"
        assert supervisor._runtimes["demo:daemon"].process is first and first.poll() is None

        reads = iter([{"TOKEN": "B"}, {"TOKEN": "C"}])
        dead_at_start = []
        real_start = supervisor.start
        monkeypatch.setattr(supervisor, "start", lambda d, **kw: dead_at_start.append(first.poll() is not None)
                            or real_start(d, **kw))
        supervisor._restart_history["demo:daemon"] = [1.0]
        assert supervisor.refresh("demo", "daemon", lambda: next(reads), compare_keys=["TOKEN"],
                                  still_valid=lambda: True) == "restarted"
        current = supervisor._runtimes["demo:daemon"]
        assert dead_at_start == [True] and current.process is not first
        assert current.descriptor.env == {"TOKEN": "C"}  # the latest save, read after the stop
        assert supervisor._restart_history["demo:daemon"] == [1.0]
        assert supervisor.refresh("demo", "daemon", lambda: {"TOKEN": "C"}, compare_keys=["TOKEN"],
                                  still_valid=lambda: True) == "unchanged"
        # A cleared value is a change too; nothing is started when the caller's admission ended meanwhile.
        assert supervisor.refresh("demo", "daemon", lambda: {}, compare_keys=["TOKEN"],
                                  still_valid=lambda: False) == "refused"
        assert "demo:daemon" not in supervisor._runtimes
        assert supervisor._runtimes["other:daemon"].process is other and other.poll() is None  # untouched
    finally:
        supervisor.stop_all(timeout_sec=1)


def test_supervisor_refresh_settles_only_the_values_its_caller_read(tmp_path):
    """``expected`` binds the four settled answers to the caller's read. A process kept on A while the caller read
    B, and one started on C (saved during the stop) while it read B, both answer ``settings_changed``; the second
    still applied C. Blanks around a value aside, empty, blank and absent are one missing value; without
    ``expected`` the answer is as before, and a refused admission stays a refusal."""
    init_server_process_pid()
    supervisor = CompanionSupervisor(tmp_path)

    def refresh(*reads, expected, still_valid=lambda: True):
        values = iter(reads)
        return supervisor.refresh("demo", "daemon", lambda: dict(next(values)), compare_keys=["TOKEN"],
                                  still_valid=still_valid, expected=expected)

    try:
        assert supervisor.start(_sleeper(tmp_path, {"TOKEN": "A", "PATH": "/a"}))
        first = supervisor._runtimes["demo:daemon"].process
        assert refresh({"TOKEN": "A"}, expected={"TOKEN": "B"}) == "settings_changed"
        assert supervisor._runtimes["demo:daemon"].process is first and first.poll() is None
        assert refresh({"TOKEN": "A"}, expected={"TOKEN": " A "}) == "unchanged"
        assert refresh({"TOKEN": "B"}, {"TOKEN": "C"}, expected={"TOKEN": "B"}) == "settings_changed"
        assert supervisor._runtimes["demo:daemon"].descriptor.env == {"TOKEN": "C"} and first.poll() is not None
        assert refresh({"TOKEN": "C"}, expected={"TOKEN": "C"}) == "unchanged"
        assert refresh({}, {}, expected={"TOKEN": ""}) == "restarted"
        assert "TOKEN" not in supervisor._runtimes["demo:daemon"].descriptor.env
        assert refresh({}, expected={"TOKEN": None}) == "unchanged"
        assert refresh({"TOKEN": "   "}, {"TOKEN": "   "}, expected={"TOKEN": ""}) == "restarted"
        assert refresh({"TOKEN": "   "}, expected={"TOKEN": "C"}) == "settings_changed"
        assert refresh({"TOKEN": "   "}, expected=None) == "unchanged"
        assert refresh({"TOKEN": "D"}, expected={"TOKEN": "Z"}, still_valid=lambda: False) == "refused"
    finally:
        supervisor.stop_all(timeout_sec=1)


def test_supervisor_answers_from_memory_only_while_the_companion_carries_the_read(tmp_path):
    """``refresh_blocker`` with ``expected``: ``unchanged`` while the running companion carries it, nothing to say
    while it carries another value. A crash with the budget spent leaves the monitor's ``not_running``. A refresh
    that stopped the companion and could not read Settings again owes a start: no answer from memory, and its
    next refresh starts one on the value the stopped one carried. An explicit stop forgets it all."""
    from ouroboros.settings_integrity import SettingsDocumentUnknown

    init_server_process_pid()
    supervisor = CompanionSupervisor(tmp_path)

    def gone():
        deadline = time.monotonic() + 15
        while "demo:daemon" in supervisor._runtimes:
            assert time.monotonic() < deadline, "the companion was never retired"
            time.sleep(0.05)

    try:
        assert supervisor.start(_sleeper(tmp_path, {"TOKEN": "A"}))
        assert supervisor.refresh_blocker("demo", "daemon", {"TOKEN": " A "}) == "unchanged"
        assert supervisor.refresh_blocker("demo", "daemon", {"TOKEN": "B"}) == ""
        assert supervisor.refresh_blocker("demo", "daemon") == ""  # without expected the read decides
        crashed = supervisor._runtimes["demo:daemon"]
        supervisor._restart_history["demo:daemon"] = [time.monotonic()] * crashed.descriptor.max_restarts
        crashed.process.kill()
        gone()
        assert supervisor.refresh_blocker("demo", "daemon", {"TOKEN": "A"}) == "not_running"
        assert supervisor.refresh_blocker("demo", "daemon", {"TOKEN": "B"}) == ""
        assert supervisor.refresh("demo", "daemon", lambda: {"TOKEN": "A"}, compare_keys=["TOKEN"],
                                  still_valid=lambda: True, expected={"TOKEN": "A"}) == "not_running"
        assert "demo:daemon" not in supervisor._runtimes, "a refresh restarted what the crash budget stopped"

        assert supervisor.start(_sleeper(tmp_path, {"TOKEN": "A"}))
        reads = iter([{"TOKEN": "B"}])

        def read_then_unknown():
            for values in reads:
                return values
            raise SettingsDocumentUnknown("unreadable after the stop")

        with pytest.raises(SettingsDocumentUnknown):
            supervisor.refresh("demo", "daemon", read_then_unknown, compare_keys=["TOKEN"], still_valid=lambda: True)
        assert "demo:daemon" not in supervisor._runtimes
        assert supervisor.refresh_blocker("demo", "daemon", {"TOKEN": "A"}) == ""
        assert supervisor.refresh("demo", "daemon", lambda: {"TOKEN": "A"}, compare_keys=["TOKEN"],
                                  still_valid=lambda: True, expected={"TOKEN": "A"}) == "started"
        assert supervisor._runtimes["demo:daemon"].process.poll() is None
        assert supervisor.refresh_blocker("demo", "daemon", {"TOKEN": "A"}) == "unchanged"
        supervisor.stop("demo", "daemon")
        assert supervisor.refresh_blocker("demo", "daemon", {"TOKEN": "A"}) == "not_started"
    finally:
        supervisor.stop_all(timeout_sec=1)


def test_supervisor_refresh_never_replaces_an_unconfirmed_stop_and_lifecycle_controls_win(tmp_path, monkeypatch):
    init_server_process_pid()
    supervisor = CompanionSupervisor(tmp_path)
    try:
        assert supervisor.start(_sleeper(tmp_path, {"TOKEN": "A"}))
        held = supervisor._runtimes["demo:daemon"].process
        monkeypatch.setattr(supervisor, "_terminate_runtime", lambda runtime, timeout_sec: None)  # it outlives the stop
        assert supervisor.refresh("demo", "daemon", lambda: {"TOKEN": "B"}, compare_keys=["TOKEN"],
                                  still_valid=lambda: True) == "stop_unconfirmed"
        assert supervisor._runtimes["demo:daemon"].process is held and supervisor.snapshot()["demo:daemon"]["retiring"]
        assert supervisor.refresh_blocker("demo", "daemon") == "retained"
        assert supervisor.refresh("demo", "daemon", lambda: {"TOKEN": "B"}, compare_keys=["TOKEN"],
                                  still_valid=lambda: True) == "retained"
        monkeypatch.undo()
        supervisor.stop("demo", "daemon", timeout_sec=1)  # an explicit stop forgets the descriptor: no revival
        assert supervisor.refresh_blocker("demo", "daemon") == "not_started"
        assert supervisor.refresh("demo", "daemon", lambda: {"TOKEN": "C"}, compare_keys=["TOKEN"],
                                  still_valid=lambda: True) == "not_started"
        assert supervisor.start(_sleeper(tmp_path, {"TOKEN": "A"}))
        supervisor.panic_kill_all()
        assert supervisor.refresh("demo", "daemon", lambda: {"TOKEN": "C"}, compare_keys=["TOKEN"],
                                  still_valid=lambda: True) == "panic"
        assert supervisor.snapshot() == {}
    finally:
        supervisor.stop_all(timeout_sec=1)


def test_supervisor_refresh_admits_and_spawns_inside_its_admission_and_stops_outside_it(tmp_path, monkeypatch):
    """The caller's admission (its lifecycle lock) covers the final check, the environment and the start as one
    step, never the comparison or the stop: a refused admission starts nothing."""
    init_server_process_pid()
    supervisor = CompanionSupervisor(tmp_path)
    events: list[str] = []

    class Admission:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, *_exc):
            events.append("exit")

    try:
        assert supervisor.start(_sleeper(tmp_path, {"TOKEN": "A"}))
        terminate, start = supervisor._terminate_runtime, supervisor.start
        monkeypatch.setattr(supervisor, "_terminate_runtime",
                            lambda runtime, timeout_sec: events.append("stop") or terminate(runtime, timeout_sec=timeout_sec))
        monkeypatch.setattr(supervisor, "start", lambda descriptor, **kw: events.append("start") or start(descriptor, **kw))

        def env(token):
            return lambda: events.append("env") or {"TOKEN": token}

        assert supervisor.refresh("demo", "daemon", env("B"), compare_keys=["TOKEN"], admission=Admission(),
                                  still_valid=lambda: events.append("admit") or True) == "restarted"
        assert events == ["env", "stop", "enter", "admit", "env", "start", "exit"]
        events.clear()
        assert supervisor.refresh("demo", "daemon", env("C"), compare_keys=["TOKEN"], admission=Admission(),
                                  still_valid=lambda: events.append("admit") or False) == "refused"
        assert events == ["env", "stop", "enter", "admit", "exit"] and "demo:daemon" not in supervisor._runtimes
    finally:
        supervisor.stop_all(timeout_sec=1)


_PLUGIN_API_24 = "plugin_api: \"2.4\"\n"
_REFRESH_FRONTMATTER = (
    "env_from_settings: [TELEGRAM_BOT_TOKEN]\n"
    "companion_processes:\n"
    "  - name: daemon\n"
    "    runtime: python3\n"
    "    command: [\"python3\", \"scripts/daemon.py\"]\n"
)


def _refreshable_skill(tmp_path, monkeypatch, token: str, plugin_api: str = _PLUGIN_API_24, payload=None,
                       *, fake_companion: bool = False):
    """A granted extension (declaring ``plugin_api``, 2.4 by default) whose companion runs in a real supervisor,
    settings in a real document; ``payload`` adds or replaces skill files (relative path -> text).
    ``fake_companion`` runs that supervisor over the existing signal-free companion fixture."""
    import ouroboros.config as config_mod
    from ouroboros.skill_loader import save_skill_grants

    init_server_process_pid()
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(config_mod, "SETTINGS_PATH", settings_path)
    settings_path.write_text(json.dumps({"TELEGRAM_BOT_TOKEN": token}), encoding="utf-8")
    repo_root, drive_root = tmp_path / "skills", tmp_path / "drive"
    skill_dir = repo_root / "tgdemo"
    (skill_dir / "scripts").mkdir(parents=True)
    drive_root.mkdir()
    (skill_dir / "scripts" / "daemon.py").write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    for relative, text in (payload or {}).items():
        (skill_dir / relative).write_text(text, encoding="utf-8")
    (skill_dir / "plugin.py").write_text("def register(api):\n    api.register_companion_process('daemon')\n",
                                         encoding="utf-8")
    (skill_dir / "SKILL.md").write_text(
        "---\nname: tgdemo\ndescription: d\nversion: 0.1.0\ntype: extension\nentry: plugin.py\n"
        "permissions: [companion_process, read_settings]\n" + plugin_api + _REFRESH_FRONTMATTER + "---\nbody\n",
        encoding="utf-8")
    loaded = find_skill(drive_root, "tgdemo", repo_path=str(repo_root))
    save_enabled(drive_root, loaded.name, True)
    save_review_state(drive_root, loaded.name, SkillReviewState(status="pass", content_hash=loaded.content_hash))
    save_skill_grants(drive_root, loaded.name, ["TELEGRAM_BOT_TOKEN"], content_hash=loaded.content_hash,
                      requested_keys=["TELEGRAM_BOT_TOKEN"])
    loaded = find_skill(drive_root, "tgdemo", repo_path=str(repo_root))
    supervisor = (_supervise_fake_companions(monkeypatch, drive_root)[0] if fake_companion
                  else CompanionSupervisor(tmp_path))
    monkeypatch.setattr(extension_plugin_api, "get_global_supervisor", lambda: supervisor)
    monkeypatch.setattr(extension_loader, "get_global_supervisor", lambda: supervisor)
    monkeypatch.setattr(extension_loader, "get_skills_repo_path", lambda: str(repo_root), raising=False)
    import ouroboros.extension_liveness as liveness
    real_state = liveness.runtime_state_for_skill_name
    monkeypatch.setattr(liveness, "runtime_state_for_skill_name",
                        lambda name, root, **kw: real_state(name, root, **{**kw, "repo_path": str(repo_root)}))
    assert extension_loader.load_extension(loaded, config_mod.load_settings, drive_root=drive_root,
                                           skills=[loaded], repo_path=str(repo_root)) is None
    api = extension_plugin_api._extensions["tgdemo"].api_instances[0]
    return api, supervisor, settings_path, drive_root, loaded


_MAX_PATH = 260  # Win32: UTF-16 code units, the terminating NUL included
_DEEP_DAEMON = """\
import json, os, sys, time
sys.dont_write_bytecode = True  # no __pycache__ path to fit, nor one under an inherited PYTHONPYCACHEPREFIX
import near
with open("ready.txt", encoding="utf-8") as resource:
    print("READY " + json.dumps({"cwd": os.getcwd(), "near": [near.VALUE, near.__file__],
                                 "resource": resource.read()}), flush=True)
print("\\n" * 4096, end="", flush=True)  # the supervisor's drainer hands over whole 4096-byte reads (or EOF)
time.sleep(60)
"""


def _utf16(path) -> int:
    return len(str(path).encode("utf-16-le")) // 2


def test_companion_starts_from_a_deep_staged_import_directory(tmp_path_factory, monkeypatch):
    """Where the two-digest leaf would put the working directory past MAX_PATH while every path the launch reads
    fits, the real companion imports its neighbour and reads a cwd-relative file, then says READY through that
    runtime's own pipe without a restart (the printer's pid is not compared: a Windows venv launcher runs the script
    in a child); unload stops it and releases the tree."""
    import os
    import uuid

    from ouroboros.skill_loader import skill_state_dir_path

    leaf = "sha256-" + "0" * 64  # one full SHA-256, whatever its digits
    earlier = len(f"{leaf}-{'0' * 64}") - len(leaf)  # what the two-digest leaf added

    def staged_cwd(base, name=leaf):
        return skill_state_dir_path(base / "drive", "tgdemo") / "__extension_imports" / name / "skill"

    reads = (("scripts", "daemon.py"), ("scripts", "near.py"), ("ready.txt",))
    longest = max(_utf16(pathlib.Path("x", *parts)) - 1 for parts in reads)  # with its separator
    root = tmp_path_factory.mktemp("deep").resolve()
    below = _utf16(staged_cwd(root / "x")) - _utf16(root / "x")
    low, high = _MAX_PATH + 1 - earlier, _MAX_PATH - 1 - longest  # the earlier leaf overflows / every read fits
    depth = max((low + high) // 2, _utf16(root) + 2 + below)  # the middle, or as shallow as this root allows
    assert depth <= high, f"no depth below {root} where the earlier leaf overflows and every read fits"
    base = root / ("d" * (depth - below - _utf16(root) - 1))
    base.mkdir()
    nonce = uuid.uuid4().hex
    supervisor = None
    try:
        _, supervisor, _, _, loaded = _refreshable_skill(base, monkeypatch, "synthetic", payload={
            "scripts/daemon.py": _DEEP_DAEMON, "scripts/near.py": f"VALUE = 'near:{nonce}'\n",
            "ready.txt": f"resource:{nonce}"})
        runtime = supervisor._runtimes["tgdemo:daemon"]
        pid, cwd = runtime.process.pid, runtime.descriptor.cwd
        assert cwd == staged_cwd(base, extension_loader.extension_import_identity(loaded)) and _utf16(cwd) == depth
        assert max(_utf16(cwd.joinpath(*parts)) for parts in reads) < _MAX_PATH < _utf16(cwd) + earlier
        deadline = time.monotonic() + 60
        while b"\n" not in runtime.stdout and runtime.process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        out = bytes(runtime.stdout)  # the supervisor's drainer owns the pipe
        line = out.split(b"\n", 1)[0]
        assert b"\n" in out and line.startswith(b"READY "), (out[:512], bytes(runtime.stderr))
        ready = json.loads(line[len(b"READY "):])
        assert (ready["near"][0], ready["resource"]) == (f"near:{nonce}", f"resource:{nonce}")
        assert os.path.samefile(ready["cwd"], cwd) and _utf16(ready["cwd"]) == depth
        assert os.path.samefile(ready["near"][1], cwd / "scripts" / "near.py")
        assert supervisor._runtimes["tgdemo:daemon"] is runtime and supervisor.snapshot()["tgdemo:daemon"]["pid"] == pid
        assert runtime.process.poll() is None and not supervisor._restart_history
        extension_loader.unload_extension("tgdemo")
        assert runtime.process.poll() is not None and supervisor.snapshot() == {}
        assert not (cwd.parent / "owners" / str(os.getpid())).exists()
    finally:
        extension_loader.unload_extension("tgdemo")
        if supervisor is not None:
            supervisor.stop_all(timeout_sec=1)


def test_plugin_api_refresh_applies_a_saved_token_through_the_same_api_and_controls_win(tmp_path, monkeypatch):
    """Save, replace and clear reach the companion through ``refresh_companion_process``; a second refresh
    through the same API still works; an unreadable document keeps the process; disable and unload win."""
    api, supervisor, settings_path, drive_root, loaded = _refreshable_skill(tmp_path, monkeypatch, "1:A")
    try:
        runtime = supervisor._runtimes["tgdemo:daemon"]
        assert runtime.descriptor.env["TELEGRAM_BOT_TOKEN"] == "1:A"
        assert api.get_settings_state(["TELEGRAM_BOT_TOKEN", "OPENAI_API_KEY"]) == {
            "status": "ok", "values": {"TELEGRAM_BOT_TOKEN": "1:A"}, "missing": []}
        assert api.refresh_companion_process("daemon") == "unchanged"
        settings_path.write_text(json.dumps({"TELEGRAM_BOT_TOKEN": "1:B"}), encoding="utf-8")
        assert api.refresh_companion_process("daemon") == "restarted"
        replaced = supervisor._runtimes["tgdemo:daemon"]
        assert runtime.process.poll() is not None and replaced.descriptor.env["TELEGRAM_BOT_TOKEN"] == "1:B"
        assert api.refresh_companion_process("daemon") == "unchanged"  # the same API stays valid
        settings_path.write_text(json.dumps({}), encoding="utf-8")
        assert api.get_settings_state(["TELEGRAM_BOT_TOKEN"]) == {
            "status": "ok", "values": {}, "missing": ["TELEGRAM_BOT_TOKEN"]}
        assert api.refresh_companion_process("daemon") == "restarted"
        assert "TELEGRAM_BOT_TOKEN" not in supervisor._runtimes["tgdemo:daemon"].descriptor.env
        cleared = supervisor._runtimes["tgdemo:daemon"].process
        settings_path.write_text("{\"TELEGRAM_BOT_TOKEN\": ", encoding="utf-8")
        assert api.get_settings_state(["TELEGRAM_BOT_TOKEN"]) == {"status": "unknown", "values": {}, "missing": []}
        assert api.refresh_companion_process("daemon") == "settings_unknown"
        assert supervisor._runtimes["tgdemo:daemon"].process is cleared and cleared.poll() is None
        settings_path.write_text(json.dumps({"TELEGRAM_BOT_TOKEN": "1:C"}), encoding="utf-8")
        save_enabled(drive_root, loaded.name, False)  # disabled before its reconcile ran: still no restart
        assert api.refresh_companion_process("daemon") == "disabled"
        assert supervisor._runtimes["tgdemo:daemon"].process is cleared
        save_enabled(drive_root, loaded.name, True)
        from ouroboros.skill_loader import GRANTS_FILENAME, skill_state_dir

        grants = skill_state_dir(drive_root, loaded.name) / GRANTS_FILENAME
        granted = grants.read_bytes()
        grants.unlink()  # the grant is gone before its reconcile ran
        assert api.refresh_companion_process("daemon") == "missing_grants"
        assert supervisor._runtimes["tgdemo:daemon"].process is cleared
        grants.write_bytes(granted)
        extension_loader.unload_extension("tgdemo")
        assert api.refresh_companion_process("daemon") == "access_closed"
        assert api.get_settings_state(["TELEGRAM_BOT_TOKEN"])["status"] == "access_closed"
        assert supervisor.snapshot() == {}
    finally:
        extension_loader.unload_extension("tgdemo")
        supervisor.stop_all(timeout_sec=1)
        init_server_process_pid()


def test_plugin_api_refresh_settles_only_the_token_its_caller_read(tmp_path, monkeypatch):
    """Through the real API: a caller that read A while B is saved gets ``settings_changed`` (B applied), then
    ``unchanged`` for B; a clear read as missing settles. ``only_if_differs`` answers from memory, reading no
    Settings and minting no token, while the companion carries the read, and refreshes when it does not.
    ``expected`` names only ``env_from_settings`` keys, and a refusal names the key, never a value. Controls answer
    as before: a companion that would need a start meets the admission first."""
    api, supervisor, settings_path, drive_root, loaded = _refreshable_skill(tmp_path, monkeypatch, "1:A")
    key = "TELEGRAM_BOT_TOKEN"
    try:
        assert api.refresh_companion_process("daemon", expected={key: "1:A"}) == "unchanged"
        settings_path.write_text(json.dumps({key: "1:B"}), encoding="utf-8")
        assert api.refresh_companion_process("daemon", expected={key: "1:A"}) == "settings_changed"
        assert supervisor._runtimes["tgdemo:daemon"].descriptor.env[key] == "1:B"
        assert api.refresh_companion_process("daemon", expected={key: "1:B"}) == "unchanged"
        settings_path.write_text(json.dumps({}), encoding="utf-8")
        assert api.refresh_companion_process("daemon", expected={key: ""}) == "restarted"
        reads = []
        for name in ("skill_settings", "mint_skill_token"):
            real = getattr(extension_plugin_api, name)
            monkeypatch.setattr(extension_plugin_api, name,
                                lambda *a, _real=real, _name=name, **k: reads.append(_name) or _real(*a, **k))
        assert api.refresh_companion_process("daemon", expected={key: ""}, only_if_differs=True) == "unchanged"
        assert reads == [], "a confirmation read Settings or minted a token"
        settings_path.write_text(json.dumps({key: "1:C"}), encoding="utf-8")
        assert api.refresh_companion_process("daemon", expected={key: "1:C"}, only_if_differs=True) == "restarted"
        assert "skill_settings" in reads and supervisor._runtimes["tgdemo:daemon"].descriptor.env[key] == "1:C"
        with pytest.raises(ExtensionRegistrationError, match="naming every env_from_settings key"):
            api.refresh_companion_process("daemon", only_if_differs=True)
        for stray in ({"OPENAI_API_KEY": "sk-not-for-you"}, {key.lower(): "1:C"}):
            with pytest.raises(ExtensionRegistrationError, match="not env_from_settings") as refused:
                api.refresh_companion_process("daemon", expected=stray)
            assert not any(value in str(refused.value) for value in stray.values())
        with pytest.raises(ExtensionRegistrationError, match="maps settings keys"):
            api.refresh_companion_process("daemon", expected=[(key, "1:C")])
        save_enabled(drive_root, loaded.name, False)
        assert api.refresh_companion_process("daemon", expected={key: "1:C"}) == "disabled"
        assert api.refresh_companion_process("daemon", expected={key: "1:C"}, only_if_differs=True) == "unchanged"
        settings_path.write_text(json.dumps({key: "1:D"}), encoding="utf-8")
        assert api.refresh_companion_process("daemon", expected={key: "1:D"}, only_if_differs=True) == "disabled"
        assert supervisor._runtimes["tgdemo:daemon"].descriptor.env[key] == "1:C"
    finally:
        extension_loader.unload_extension("tgdemo")
        supervisor.stop_all(timeout_sec=1)
        init_server_process_pid()


@pytest.mark.parametrize("failure", ["unreadable", "refused_spawn", "refused_spawn_after_stop_unconfirmed",
                                     "refused_spawn_after_clean_exit"])
def test_a_companion_a_refresh_stopped_and_could_not_replace_is_started_by_the_next_one(tmp_path, monkeypatch,
                                                                                          failure):
    """A refresh stops the companion on A to apply B; after that stop Settings cannot be read, or the system refuses
    the spawn, so nothing runs. If exit was not confirmed at first, failed retries after exit still show the failure.
    A companion that had already exited cleanly has no monitor to tell of it: the refresh's failed replacement shows
    the same failure, and a replacement its admission refused shows only the pending refresh, no failure.
    A is saved back: the next refresh starts a companion on A, although A is what the stopped one carried. An absence
    a crash left (here exhausted restarts) stays the monitor's: ``not_running``."""
    from ouroboros import extension_companion

    api, supervisor, settings_path, drive, loaded = _refreshable_skill(tmp_path, monkeypatch, "1:A", fake_companion=True)
    key, terminate = "TELEGRAM_BOT_TOKEN", supervisor._terminate_runtime
    first = supervisor._runtimes["tgdemo:daemon"].process
    spawn = extension_companion.subprocess.Popen
    health_key = (drive, loaded.name)
    delayed_stop = failure == "refused_spawn_after_stop_unconfirmed"
    try:
        extension_health.record_extension_health(*health_key, status="live", reason="ready", sha="good")
        good = extension_health.read_extension_health(*health_key)["last_known_good"]
        settings_path.write_text("{", encoding="utf-8")
        assert api.refresh_companion_process("daemon") == "settings_unknown"
        assert first.poll() is None and not extension_health.read_extension_health(*health_key)["companion_failures"]
        settings_path.write_text(json.dumps({key: "1:B"}), encoding="utf-8")
        if delayed_stop:
            with monkeypatch.context() as patch:
                patch.setattr(supervisor, "_terminate_runtime", lambda *_args, **_kwargs: None)
                assert api.refresh_companion_process("daemon", expected={key: "1:B"}) == "stop_unconfirmed"
                assert api.refresh_companion_process("daemon", expected={key: "1:B"}) == "retained"
            first.end(-15)
            _until(lambda: "tgdemo:daemon" not in supervisor.snapshot(), "the intentional stop was not settled")
        if failure == "refused_spawn_after_clean_exit":
            first.end(0)
            _until(lambda: "tgdemo:daemon" not in supervisor.snapshot(), "the clean exit was not settled")
            assert supervisor.refresh("tgdemo", "daemon", lambda: {key: "1:C"}, compare_keys=[key],
                                      still_valid=lambda: False) == "refused"
            row = _listed_skill(drive, loaded.skill_dir.parent, loaded.name)
            assert (row["live_reason"], row["load_error"], _companion_events(drive)) == (
                "companion_refresh_pending", None, [])
            assert not extension_health.read_extension_health(*health_key)["companion_failures"]
        if failure == "unreadable":
            def unreadable_after_stop(*args, **kwargs):
                terminate(*args, **kwargs)
                settings_path.write_text("{", encoding="utf-8")

            monkeypatch.setattr(supervisor, "_terminate_runtime", unreadable_after_stop)
            assert api.refresh_companion_process("daemon", expected={key: "1:B"}) == "settings_unknown"
        else:
            def refused(_argv, **_kwargs):
                raise OSError("diagnostic-must-not-escape")

            monkeypatch.setattr(extension_companion.subprocess, "Popen", refused)
            with pytest.raises(OSError):
                api.refresh_companion_process("daemon", expected={key: "1:B"})
        assert first.poll() is not None and "tgdemo:daemon" not in supervisor._runtimes
        row = _listed_skill(drive, loaded.skill_dir.parent, loaded.name)
        health = extension_health.read_extension_health(*health_key)
        assert (row["live_loaded"], row["live_reason"], health["status"], health["last_known_good"]) == (
            True, "companion_failed:daemon", "broken", good)
        failed = health["companion_failures"]["daemon"]
        assert row["load_error"] == failed["load_error"] and "diagnostic-must-not-escape" not in row["load_error"]
        failure_event = ("extension_companion_refresh_failed", None, None)
        assert "returncode" not in failed and _companion_events(drive) == [failure_event]
        if delayed_stop:  # the already stopped companion needs no preliminary comparison read
            settings_path.write_text("{", encoding="utf-8")
            assert api.refresh_companion_process("daemon", expected={key: "1:B"}) == "settings_unknown"
            assert _companion_events(drive) == [failure_event, failure_event]
        monkeypatch.setattr(supervisor, "_terminate_runtime", terminate)
        monkeypatch.setattr(extension_companion.subprocess, "Popen", spawn)
        settings_path.write_text(json.dumps({key: "1:A"}), encoding="utf-8")
        assert api.refresh_companion_process("daemon", expected={key: "1:A"}) == "started"
        current = supervisor._runtimes["tgdemo:daemon"]
        assert current.process.poll() is None and current.descriptor.env[key] == "1:A"
        assert api.refresh_companion_process("daemon", expected={key: "1:A"}) == "unchanged"
        recovered = extension_health.read_extension_health(*health_key)
        assert recovered["last_known_good"] == good and recovered["companion_failures"] == {}
        assert recovered["companion_recovered"]["daemon"]["failed_at"] == failed["ts"]
        assert _listed_skill(drive, loaded.skill_dir.parent, loaded.name)["live_reason"] == "ready"
        assert _companion_events(drive)[-1] == ("extension_companion_recovered", None, "start")
        _exhaust(supervisor, "tgdemo:daemon", code=-9)
        assert api.refresh_companion_process("daemon", expected={key: "1:A"}) == "not_running"
        assert "tgdemo:daemon" not in supervisor._runtimes, "a refresh restarted what the crash budget stopped"
    finally:
        extension_loader.unload_extension("tgdemo")
        supervisor.stop_all(timeout_sec=1)
        init_server_process_pid()


def test_plugin_api_refresh_is_off_loop_and_refused_to_a_reloaded_generation(tmp_path, monkeypatch):
    import asyncio

    api, supervisor, settings_path, drive_root, loaded = _refreshable_skill(tmp_path, monkeypatch, "1:A")
    try:
        async def on_loop():
            return api.refresh_companion_process("daemon")

        with pytest.raises(ExtensionRegistrationError, match="off the event loop"):
            asyncio.run(on_loop())
        with pytest.raises(ExtensionRegistrationError, match="not declared"):
            api.refresh_companion_process("other")
        extension_loader.unload_extension("tgdemo")
        import ouroboros.config as config_mod

        assert extension_loader.load_extension(loaded, config_mod.load_settings, drive_root=drive_root,
                                               skills=[loaded]) is None
        settings_path.write_text(json.dumps({"TELEGRAM_BOT_TOKEN": "1:B"}), encoding="utf-8")
        assert api.refresh_companion_process("daemon") == "access_closed"  # the old generation's API
        fresh = extension_plugin_api._extensions["tgdemo"].api_instances[0]
        assert fresh.refresh_companion_process("daemon") == "restarted"
    finally:
        extension_loader.unload_extension("tgdemo")
        supervisor.stop_all(timeout_sec=1)
        init_server_process_pid()


def _owner_off(drive_root, loaded):
    """The owner's real Off: the lifecycle action Skills' toggle runs."""
    from ouroboros.skill_lifecycle_actions import run_skill_action

    return run_skill_action(types.SimpleNamespace(drive_root=drive_root), loaded.name, "disable",
                            repo_path=str(loaded.skill_dir.parent), _owner_actor="owner_ui")


def _held(monkeypatch, supervisor, name: str, hold):
    """Replace one supervisor step with ``hold`` running before it; returns (entered, release)."""
    entered, release, real = threading.Event(), threading.Event(), getattr(supervisor, name)

    def step(*args, **kwargs):
        if hold(*args):
            entered.set()
            assert release.wait(15)
        return real(*args, **kwargs)

    monkeypatch.setattr(supervisor, name, step)
    return entered, release


@pytest.mark.parametrize("change", ["none", "off", "revoke"])
def test_an_off_or_revocation_saved_during_the_refresh_stop_wins_at_its_spawn(tmp_path, monkeypatch, change):
    """While the old companion stops, no lock is held: the owner's real Off persists at once. The final admission
    holds the lifecycle lock through the spawn and reads that Off, or the grant removed meanwhile, so nothing
    carrying the new token starts. With nothing saved, the same held stop restarts on the new token."""
    from ouroboros.skill_loader import GRANTS_FILENAME, load_enabled, skill_state_dir

    api, supervisor, settings_path, drive_root, loaded = _refreshable_skill(tmp_path, monkeypatch, "1:A")
    first, outcome, starts = supervisor._runtimes["tgdemo:daemon"], {}, []
    stopping, release = _held(monkeypatch, supervisor, "_terminate_runtime", lambda runtime: runtime is first)
    start = supervisor.start
    monkeypatch.setattr(supervisor, "start", lambda descriptor, **kw: starts.append(
        (load_enabled(drive_root, loaded.name), descriptor.env.get("TELEGRAM_BOT_TOKEN"))) or start(descriptor, **kw))
    grants = skill_state_dir(drive_root, loaded.name) / GRANTS_FILENAME
    granted = grants.read_bytes()
    settings_path.write_text(json.dumps({"TELEGRAM_BOT_TOKEN": "1:B"}), encoding="utf-8")
    refresh = threading.Thread(target=lambda: outcome.update(refresh=api.refresh_companion_process("daemon")))
    owner = threading.Thread(target=lambda: outcome.update(off=_owner_off(drive_root, loaded)))
    try:
        refresh.start()
        assert stopping.wait(15)
        if change == "off":
            owner.start()
            deadline = time.monotonic() + 15
            while load_enabled(drive_root, loaded.name) and time.monotonic() < deadline:
                time.sleep(0.01)
            assert not load_enabled(drive_root, loaded.name), "the owner's Off waited for the refresh's stop"
        elif change == "revoke":
            grants.unlink()
        release.set()
        refresh.join(15)
        if owner.ident:
            owner.join(15)
        assert not refresh.is_alive() and not owner.is_alive()
        if change == "none":
            assert outcome["refresh"] == "restarted" and starts == [(True, "1:B")]
            return
        assert starts == [], "a process with the new token started after the owner's choice was saved"
        assert first.process.poll() is not None and "tgdemo:daemon" not in supervisor._runtimes
        if change == "off":
            # Read at admission (disabled), or the unload that followed the Off closed this API first.
            assert outcome["refresh"] in {"disabled", "access_closed"}
            assert outcome["off"]["ok"] and outcome["off"]["extension_action"] == "extension_unloaded"
        else:
            assert outcome["refresh"] == "missing_grants"
            grants.write_bytes(granted)  # the refused admission wedged nothing: the next refresh starts it
            assert api.refresh_companion_process("daemon") == "started" and starts == [(True, "1:B")]
    finally:
        release.set()
        refresh.join(15)
        if owner.ident:
            owner.join(15)
        extension_loader.unload_extension("tgdemo")
        supervisor.stop_all(timeout_sec=1)
        init_server_process_pid()


def test_an_off_saved_while_a_refresh_spawns_waits_for_that_spawn_and_then_stops_it(tmp_path, monkeypatch):
    """The other order: the Off is persisted under the same lifecycle lock, so it never lands between the final
    admission and the spawn; it waits for the spawn, and its own reconcile then stops that new process."""
    from ouroboros.extension_registry_state import _lifecycle_lock_for
    from ouroboros.skill_loader import load_enabled

    api, supervisor, settings_path, drive_root, loaded = _refreshable_skill(tmp_path, monkeypatch, "1:A")
    outcome, starts = {}, []
    spawning, release = _held(monkeypatch, supervisor, "start",
                              lambda _descriptor: starts.append(load_enabled(drive_root, loaded.name)) or True)
    settings_path.write_text(json.dumps({"TELEGRAM_BOT_TOKEN": "1:B"}), encoding="utf-8")
    refresh = threading.Thread(target=lambda: outcome.update(refresh=api.refresh_companion_process("daemon")))
    owner = threading.Thread(target=lambda: outcome.update(off=_owner_off(drive_root, loaded)))
    try:
        refresh.start()
        assert spawning.wait(15)
        lock = _lifecycle_lock_for("tgdemo")
        held_elsewhere = not lock.acquire(blocking=False)
        if not held_elsewhere:
            lock.release()
        assert held_elsewhere, "the final admission released the lifecycle lock before its spawn"
        owner.start()
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            assert load_enabled(drive_root, loaded.name), "the Off landed between the final admission and the spawn"
            time.sleep(0.01)
        release.set()
        refresh.join(15)
        owner.join(15)
        assert not refresh.is_alive() and not owner.is_alive()
        assert outcome["refresh"] == "restarted" and starts == [True]
        assert not load_enabled(drive_root, loaded.name) and outcome["off"]["extension_action"] == "extension_unloaded"
        assert supervisor.snapshot() == {}  # the process admitted before the Off is stopped by the Off's reconcile
    finally:
        release.set()
        refresh.join(15)
        if owner.ident:
            owner.join(15)
        extension_loader.unload_extension("tgdemo")
        supervisor.stop_all(timeout_sec=1)
        init_server_process_pid()


def test_panic_during_a_refresh_admission_is_not_delayed_and_wins_its_spawn(tmp_path, monkeypatch):
    api, supervisor, settings_path, _drive_root, _loaded = _refreshable_skill(tmp_path, monkeypatch, "1:A")
    outcome = {}
    spawning, release = _held(monkeypatch, supervisor, "start", lambda _descriptor: True)
    settings_path.write_text(json.dumps({"TELEGRAM_BOT_TOKEN": "1:B"}), encoding="utf-8")
    refresh = threading.Thread(target=lambda: outcome.update(refresh=api.refresh_companion_process("daemon")))
    try:
        refresh.start()
        assert spawning.wait(15)
        panic = threading.Thread(target=supervisor.panic_kill_all)
        panic.start()
        panic.join(5)
        assert not panic.is_alive(), "Panic waited for a held refresh admission"
        release.set()
        refresh.join(15)
        assert outcome["refresh"] == "panic" and supervisor.snapshot() == {}
    finally:
        release.set()
        refresh.join(15)
        extension_loader.unload_extension("tgdemo")
        supervisor.stop_all(timeout_sec=1)
        init_server_process_pid()


def test_every_other_reason_not_to_run_reads_not_desired(tmp_path, monkeypatch):
    """The closed outcome set: a reason beside ``disabled`` and ``missing_grants`` (here an edited payload whose
    review is stale before any reload) answers ``not_desired`` and starts nothing."""
    api, supervisor, settings_path, _drive_root, loaded = _refreshable_skill(tmp_path, monkeypatch, "1:A")
    try:
        running = supervisor._runtimes["tgdemo:daemon"].process
        (loaded.skill_dir / "scripts" / "edited.py").write_text("# an owner edit\n", encoding="utf-8")
        settings_path.write_text(json.dumps({"TELEGRAM_BOT_TOKEN": "1:B"}), encoding="utf-8")
        assert api.refresh_companion_process("daemon") == "not_desired"
        assert supervisor._runtimes["tgdemo:daemon"].process is running and running.poll() is None
    finally:
        extension_loader.unload_extension("tgdemo")
        supervisor.stop_all(timeout_sec=1)
        init_server_process_pid()


@pytest.mark.parametrize("declared", ['plugin_api: "2.0"\n', 'plugin_api: "2.3"\n', ""])
def test_plugin_api_2_4_methods_follow_the_loaders_real_negotiation(tmp_path, monkeypatch, declared):
    """Through the real loader: every declared 2.x minimum negotiates the host's 2.4 and is served its settings
    state and companion refresh; a payload without the field binds legacy 1.3 and is refused both."""
    api, supervisor, settings_path, _drive_root, _loaded = _refreshable_skill(tmp_path, monkeypatch, "1:A", declared)
    try:
        if declared:
            assert api.get_settings_state(["TELEGRAM_BOT_TOKEN"]) == {
                "status": "ok", "values": {"TELEGRAM_BOT_TOKEN": "1:A"}, "missing": []}
            assert api.get_settings_state([]) == {"status": "ok", "values": {}, "missing": []}
            settings_path.write_text(json.dumps({"TELEGRAM_BOT_TOKEN": "1:B"}), encoding="utf-8")
            assert api.refresh_companion_process("daemon") == "restarted"
        else:
            for call in (lambda: api.get_settings_state(["TELEGRAM_BOT_TOKEN"]),
                         lambda: api.refresh_companion_process("daemon")):
                with pytest.raises(ExtensionRegistrationError, match='needs plugin_api "2.4".*negotiated 1.3'):
                    call()
    finally:
        extension_loader.unload_extension("tgdemo")
        supervisor.stop_all(timeout_sec=1)
        init_server_process_pid()


@contextlib.contextmanager
def _native_telegram(tmp_path, monkeypatch, *, token: str = "", miniapp: str = "off", hold=None, payload_edit=None,
                     core_port: int = 0, refuse=None):
    """The shipped Telegram payload through the real loader, supervisor and extension loop, beside a loopback Bot
    API stand-in that records every request; ``hold(token, method)`` may block an answer and ``refuse(token,
    method)`` answers it 401. No tunnel, Telegram or cloud request is made. Yields the world the native witnesses
    drive."""
    import asyncio
    import http.server
    import time
    import uuid

    import supervisor.message_bus as mb
    import supervisor.queue as queue_mod
    import ouroboros.config as config_mod
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient
    from ouroboros import owner_quiz
    from ouroboros import secret_requests as sr
    from ouroboros.event_bus import EventBus
    from ouroboros.gateway import secret_decision as sd
    from ouroboros.gateway import settings as settings_mod
    from ouroboros.gateway import task_decision as td
    from ouroboros.skill_loader import (requested_core_setting_keys, requested_skill_permissions,
                                        save_skill_grants)
    from ouroboros.task_results import write_task_result

    asked: list[tuple[str, str]] = []
    menus: dict[str, dict] = {}  # each bot's own menu button for the owner chat
    menu_sets: list[tuple[str, dict]] = []

    class BotApi(http.server.BaseHTTPRequestHandler):
        timeout = 5  # an idle connection (a call cancelled after connecting) ends, so closing can join its thread

        def do_POST(self):  # noqa: N802 — the stdlib handler's name
            token_, method = self.path[len("/bot"):].split("/", 1)
            asked.append((token_, method))
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            form = json.loads(raw or b"{}") if "json" in (self.headers.get("Content-Type") or "") else {}
            if hold is not None:
                hold(token_, method)
            refused = refuse is not None and refuse(token_, method)
            bot = token_.split(":", 1)[0]
            if method == "setChatMenuButton" and not refused:
                menus[bot] = form.get("menu_button") or {"type": "default"}
                menu_sets.append((token_, menus[bot]))
            result = ([] if method == "getUpdates"
                      else {"id": int(bot) if bot.isdigit() else 1, "is_bot": True, "username": "stand_in_bot"}
                      if method == "getMe"
                      else {"id": 42, "type": "private"} if method == "getChat"
                      else menus.get(bot, {"type": "default"}) if method == "getChatMenuButton"
                      else {"message_id": len(asked)} if method == "sendMessage" else True)
            body = json.dumps({"ok": False, "error_code": 401, "description": "Unauthorized"} if refused
                              else {"ok": True, "result": result}).encode()
            self.send_response(401 if refused else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    class JoinedServer(http.server.ThreadingHTTPServer):
        daemon_threads = False  # server_close() joins every request thread it started (block_on_close)

    bot_server = JoinedServer(("127.0.0.1", 0), BotApi)
    serving = threading.Thread(target=bot_server.serve_forever, daemon=True)
    serving.start()
    init_server_process_pid()
    drive = tmp_path / "drive"
    drive.mkdir()
    monkeypatch.setattr(config_mod, "DATA_DIR", drive)
    monkeypatch.setattr(config_mod, "SETTINGS_PATH", drive / "settings.json")
    monkeypatch.setattr(config_mod, "get_skills_repo_path", lambda: "")
    if token:
        (drive / "settings.json").write_text(json.dumps({"TELEGRAM_BOT_TOKEN": token}), encoding="utf-8")
    payload = drive / "skills" / "external" / "telegram"
    shutil.copytree(pathlib.Path(__file__).parents[1] / "skills" / "telegram", payload)
    if payload_edit is not None:
        payload_edit(payload)
    state = drive / "state" / "skills" / "telegram"
    state.mkdir(parents=True)
    if core_port:  # the server port the Mini App's runtime configuration names (get_runtime_info)
        (drive / "state" / "server_port").write_text(str(core_port), encoding="utf-8")
    (state / "settings.json").write_text(json.dumps({
        "TELEGRAM_CHAT_ID": "42", "TELEGRAM_MINIAPP_ENABLED": miniapp,
        "TELEGRAM_API_BASE": f"http://127.0.0.1:{bot_server.server_address[1]}"}), encoding="utf-8")

    def admit():
        """Review, enable and grant the payload as it is now; returns the loaded skill."""
        skill = find_skill(drive, "telegram", repo_path="")
        save_enabled(drive, "telegram", True)
        save_review_state(drive, "telegram", SkillReviewState(status="pass", content_hash=skill.content_hash))
        permissions = requested_skill_permissions(list(skill.manifest.permissions or []),
                                                  list(skill.manifest.subscribe_events or []))
        keys = requested_core_setting_keys(list(skill.manifest.env_from_settings or []))
        save_skill_grants(drive, "telegram", keys, content_hash=skill.content_hash, requested_keys=keys,
                          granted_permissions=permissions, requested_permissions=permissions)
        return find_skill(drive, "telegram", repo_path="")

    skill = admit()
    supervisor = CompanionSupervisor(drive)
    monkeypatch.setattr(extension_plugin_api, "get_global_supervisor", lambda: supervisor)
    monkeypatch.setattr(extension_loader, "get_global_supervisor", lambda: supervisor)
    loop = asyncio.new_event_loop()
    looping = threading.Thread(target=loop.run_forever, daemon=True)
    looping.start()
    bus = EventBus()
    bus.set_loop(loop)
    monkeypatch.setattr(extension_plugin_api, "get_global_event_bus", lambda: bus)
    monkeypatch.setattr(extension_loader, "get_global_event_bus", lambda: bus)
    import ouroboros.extension_liveness as liveness
    real_state = liveness.runtime_state_for_skill_name
    monkeypatch.setattr(liveness, "runtime_state_for_skill_name",
                        lambda name, root, **kw: real_state(name, root, **{**kw, "repo_path": ""}))
    # The protected form's owner: a running setup task whose secret question the form answers.
    monkeypatch.setattr(mb, "get_bridge", lambda: SimpleNamespaceBridge())
    monkeypatch.setattr(mb, "DATA_DIR", drive)
    monkeypatch.setattr(mb, "publish_event", lambda *args: None)
    monkeypatch.setattr(td, "_live_root_task", lambda task_id, *, holds_mail=False: ({"id": task_id}, ""))
    monkeypatch.setattr(queue_mod, "_task_drive_for_task", lambda task, task_id: drive, raising=False)
    write_task_result(drive, "setup", "running", chat_id=1)
    form = Starlette(routes=[Route("/api/decisions/secret", endpoint=sd.api_secret_decision, methods=["POST"])])
    form.state.drive_root = drive
    monkeypatch.setattr(settings_mod, "_start_supervisor_if_needed_for_request", lambda *_a, **_k: False)
    monkeypatch.setattr(settings_mod, "_apply_settings_save_side_effects", lambda *_a, **_k: None)
    owner_settings = Starlette(routes=[Route("/api/settings", endpoint=settings_mod.api_settings_post,
                                             methods=["POST"])])
    owner_settings.state.drive_root = owner_settings.state.repo_dir = drive

    def form_save(value):
        request = sr.resolve_secret_request(drive, "telegram", ["botToken"])  # the skill's own secret_fields
        quiz_id = uuid.uuid4().hex
        owner_quiz.record_asked(drive, "setup", quiz_id=quiz_id, question=sr.question_text(request), options=[],
                                wait_for_answer=True, chat_id=1, secret_request=request)
        owner_quiz.record_publication(drive, "setup", quiz_id, 1)
        decision = f"quiz:setup:{quiz_id}"
        client = TestClient(form)
        status = client.post("/api/decisions/secret", json={"decision_id": decision, "action": "status"}).json()
        response = client.post("/api/decisions/secret", json={
            "request_id": uuid.uuid4().hex, "decision_id": decision, "values": {"botToken": value},
            "observed": {row["id"]: row["configured"] for row in status["keys"]}})
        assert response.status_code == 200, response.text

    def settings_save(value):
        response = TestClient(owner_settings).post("/api/settings", json={"TELEGRAM_BOT_TOKEN": value})
        assert response.status_code == 200, response.text

    def until(condition, what, timeout=30.0):
        deadline = time.monotonic() + timeout
        while not condition():
            assert time.monotonic() < deadline, what
            time.sleep(0.05)

    def published():
        try:
            return json.loads((state / "status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    try:
        yield types.SimpleNamespace(
            drive=drive, state=state, payload=payload, skill=skill, admit=admit, supervisor=supervisor, bus=bus,
            asked=asked, menu_sets=menu_sets, form_save=form_save, settings_save=settings_save, until=until, published=published,
            load_settings=config_mod.load_settings,
            runtime=lambda: supervisor._runtimes.get("telegram:miniapp_gateway"),
            api=lambda: extension_plugin_api._extensions["telegram"].api_instances[0])
    finally:
        extension_loader.unload_extension("telegram")
        supervisor.stop_all(timeout_sec=3)

        async def drain_own_tasks():
            # Unload only requests cancellation of the supervised runners: let them finish on this loop first.
            pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        asyncio.run_coroutine_threadsafe(drain_own_tasks(), loop).result(30)
        loop.call_soon_threadsafe(loop.stop)
        looping.join(10)
        loop.run_until_complete(loop.shutdown_default_executor())  # the observer's to_thread workers
        loop.close()
        bot_server.shutdown()
        bot_server.server_close()
        serving.join(10)
        init_server_process_pid()


def test_the_native_telegram_skill_applies_form_and_settings_saves_to_its_real_companion(tmp_path, monkeypatch):
    """The shipped Telegram payload through the real loader, supervisor and extension loop: its own observer
    applies a token the protected secret form saved, a replacement the Settings route saved, and a clear, to
    the real `companion.py` process (old death observed before each new one) and to its poller. The Mini App
    is Off and the Bot API is a loopback stand-in: no tunnel, Telegram or cloud request is made."""
    import time

    token_a, token_b = "111:SYNTHETIC-A", "222:SYNTHETIC-B"
    with _native_telegram(tmp_path, monkeypatch) as world:
        asked, runtime, published, until = world.asked, world.runtime, world.published, world.until
        assert extension_loader.load_extension(world.skill, world.load_settings, drive_root=world.drive,
                                               skills=[world.skill], repo_path="") is None
        until(lambda: runtime() is not None and published().get("pid") == runtime().process.pid
              and published().get("state") == "disabled", "the companion never published its Off wait")
        first = runtime().process
        assert "TELEGRAM_BOT_TOKEN" not in runtime().descriptor.env
        time.sleep(0.5)
        assert first.poll() is None and not asked, "no token: the companion waits and the poller asks nothing"

        world.form_save(token_a)
        until(lambda: runtime() is not None and runtime().process is not first
              and runtime().descriptor.env.get("TELEGRAM_BOT_TOKEN") == token_a, "the form's token never applied")
        assert first.poll() is not None, "the old companion outlived its replacement"
        second = runtime().process
        until(lambda: published().get("pid") == second.pid, "the new companion never published")
        until(lambda: (token_a, "getUpdates") in asked, "the poller never polled with the form's token")

        world.settings_save(token_b)
        until(lambda: runtime() is not None and runtime().process is not second
              and runtime().descriptor.env.get("TELEGRAM_BOT_TOKEN") == token_b, "the replacement never applied")
        assert second.poll() is not None
        until(lambda: (token_b, "getUpdates") in asked, "the poller never followed the replacement")
        switched = len(asked)
        time.sleep(1.0)
        assert all(token == token_b for token, _method in asked[switched:]), "the old token was asked again"

        third = runtime().process
        world.settings_save("")
        until(lambda: runtime() is not None and runtime().process is not third
              and "TELEGRAM_BOT_TOKEN" not in runtime().descriptor.env, "the clear never applied")
        assert third.poll() is not None
        cleared = len(asked)
        time.sleep(1.5)
        assert len(asked) <= cleared + 1, "the poller kept asking without a token"  # one in-flight poll may finish
        restarts = world.supervisor._restart_history.get("telegram:miniapp_gateway", [])
        assert restarts == [], "a settings refresh spent the crash budget"


@pytest.mark.parametrize("before,saved", [("", "111:SYNTHETIC-A"), ("111:SYNTHETIC-A", "")])
def test_a_token_saved_back_during_a_mini_app_refresh_leaves_no_companion_on_the_other_one(
    tmp_path, monkeypatch, before, saved,
):
    """A → B → A, and clear → B → clear, around one refresh of the shipped payload's real companion. The observer
    reads ``saved`` and asks for a refresh; the owner saves B before that refresh reads Settings, and ``saved`` again
    after its final read, while B's companion spawns. The observer never reads B. What the refresh applied is not
    what the observer asked for, so it does not settle: it asks once more and the real companion converges to the
    saved value, then nothing more is asked. The Mini App is Off and the Bot API a loopback stand-in."""
    other, key = "222:SYNTHETIC-B", "TELEGRAM_BOT_TOKEN"
    with _native_telegram(tmp_path, monkeypatch, token=before) as world:
        supervisor, runtime, until = world.supervisor, world.runtime, world.until
        armed, outcomes, spawned, real_refresh = {}, [], [], supervisor.refresh
        entered, release = threading.Event(), threading.Event()

        def refresh(*args, **kwargs):
            if armed.pop("refresh", False):
                entered.set()
                assert release.wait(15)
            outcomes.append(real_refresh(*args, **kwargs))
            return outcomes[-1]

        monkeypatch.setattr(supervisor, "refresh", refresh)
        spawning, spawn = _held(monkeypatch, supervisor, "start",
                                lambda descriptor: descriptor.env.get(key) == other and armed.pop("start", False))
        held_start = supervisor.start

        def start(descriptor, **kwargs):
            started = held_start(descriptor, **kwargs)
            spawned.append((descriptor.env.get(key, ""), runtime().process if started else None))
            return started

        monkeypatch.setattr(supervisor, "start", start)
        assert extension_loader.load_extension(world.skill, world.load_settings, drive_root=world.drive,
                                               skills=[world.skill], repo_path="") is None
        until(lambda: outcomes == ["unchanged"], "the observer's first refresh never concluded")
        armed["refresh"] = True
        world.settings_save(saved)
        assert entered.wait(15), "the observer never asked to apply the saved value"
        world.settings_save(other)  # before the refresh reads Settings
        armed["start"] = True
        release.set()
        assert spawning.wait(15), "the refresh never spawned the value saved meanwhile"
        world.settings_save(saved)  # after its final read, before the observer's next one
        spawn.set()
        seen = {}

        def converged():
            current = runtime()
            seen["token"] = None if current is None else current.descriptor.env.get(key, "")
            return seen["token"] == saved and current.process.poll() is None and outcomes[-1:] == ["restarted"]

        deadline = time.monotonic() + 20
        while not converged() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert converged(), f"the Mini App stays on {seen['token']!r}, Settings hold {saved!r}; refreshes {outcomes}"
        time.sleep(4.5)  # two more reads of the unchanged value
        assert outcomes == ["unchanged", "settings_changed", "restarted"], "a refresh was asked again when settled"
        assert [token for token, _process in spawned] == [before, other, saved]
        assert all(process.poll() is not None for _token, process in spawned[:-1]) and spawned[-1][1].poll() is None
        assert runtime().process is spawned[-1][1] and world.load_settings().get(key, "") == saved
        assert supervisor._restart_history.get("telegram:miniapp_gateway", []) == []


@pytest.mark.parametrize("saved", ["111:SYNTHETIC-A", ""])
def test_a_companion_recovered_on_other_settings_returns_to_the_token_the_bridge_holds(tmp_path, monkeypatch, saved):
    """The real companion's restarts are exhausted while its settled observer waits on its next read. Generic
    recovery starts it again from Settings that hold B at that moment; ``saved`` is back before the observer reads.
    The observer reads no change, yet the companion returns to ``saved`` within a few reads, and is then left
    alone. The Mini App is Off and the Bot API a loopback stand-in."""
    other, key = "222:SYNTHETIC-B", "TELEGRAM_BOT_TOKEN"
    with _native_telegram(tmp_path, monkeypatch, token=saved) as world:
        supervisor, runtime, until = world.supervisor, world.runtime, world.until
        outcomes, real_refresh = [], supervisor.refresh
        held, release = threading.Event(), threading.Event()

        def refresh(*args, **kwargs):
            outcomes.append(real_refresh(*args, **kwargs))
            return outcomes[-1]

        monkeypatch.setattr(supervisor, "refresh", refresh)
        assert extension_loader.load_extension(world.skill, world.load_settings, drive_root=world.drive,
                                               skills=[world.skill], repo_path="") is None
        until(lambda: outcomes == ["unchanged"], "the observer's first refresh never concluded")
        api = world.api()
        read = api.get_settings_state

        def next_read_waits(keys):
            if keys and not release.is_set():
                held.set()
                assert release.wait(15), "the observer was never released"
            return read(keys)

        monkeypatch.setattr(api, "get_settings_state", next_read_waits)
        try:
            assert held.wait(10), "the observer never reached its next read"
            exhausted = runtime()
            with supervisor._lock:
                supervisor._restart_history["telegram:miniapp_gateway"] = (
                    [time.monotonic()] * exhausted.descriptor.max_restarts)
            exhausted.process.kill()
            until(lambda: runtime() is None, "the exhausted companion was never retired")
            world.settings_save(other)
            recovery = extension_loader.ensure_companions_running("telegram", world.drive, world.load_settings,
                                                                  repo_path="", selected_skill=world.skill)
            assert recovery["action"] == "started_missing", recovery
            recovered = runtime()
            assert recovered.descriptor.env.get(key) == other
            world.settings_save(saved)
        finally:
            release.set()
        seen = {}

        def returned():
            current = runtime()
            seen["token"] = None if current is None else current.descriptor.env.get(key, "")
            return seen["token"] == saved and current.process.poll() is None and outcomes[-1:] == ["restarted"]

        deadline = time.monotonic() + 20
        while not returned() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert returned(), f"the Mini App stays on {seen['token']!r}, Settings hold {saved!r}; refreshes {outcomes}"
        settled = runtime().process
        time.sleep(4.5)  # two more reads of the unchanged value
        assert outcomes == ["unchanged", "restarted"] and runtime().process is settled
        assert recovered.process.poll() is not None and world.load_settings().get(key, "") == saved


def test_a_saved_replacement_brings_back_native_typing_the_refused_token_had_muted(tmp_path, monkeypatch, caplog):
    """The shipped payload through the real loader: the host prepares a reply for the owner's chat and Telegram
    refuses the old token (401), so the bridge's typing cue mutes that chat, once and loudly. The owner saves a
    working token on the Settings route: the skill's own observer applies it and wakes the cue, which types through
    the new token with no message delivered first, and never again through the old one. Only the host's reply
    projection is scripted; the Bot API is the loopback stand-in."""
    from ouroboros.gateway import reply_preparation

    token_a, token_b = "111:SYNTHETIC-A", "222:SYNTHETIC-B"
    monkeypatch.setattr(reply_preparation, "reply_preparation",
                        lambda _data_dir: {"complete": True, "chats": [{"chat_id": 1, "transport": {}}]})
    with _native_telegram(tmp_path, monkeypatch, token=token_a,
                          refuse=lambda token, _method: token == token_a) as world:
        asked, until = world.asked, world.until

        def asked_by(token, method):
            return [1 for asked_token, asked_method in list(asked) if (asked_token, asked_method) == (token, method)]

        def muted():
            return [record for record in caplog.records if "typing indicator stopped" in record.getMessage()]

        assert extension_loader.load_extension(world.skill, world.load_settings, drive_root=world.drive,
                                               skills=[world.skill], repo_path="") is None
        until(lambda: asked_by(token_a, "sendChatAction") and muted(), "the old token's refusal never muted typing")
        world.settings_save(token_b)
        until(lambda: asked_by(token_b, "sendChatAction"), "typing never came back through the replacement")
        assert not asked_by(token_b, "sendMessage"), "it came back before any message reached the chat"
        until(lambda: len(asked_by(token_b, "sendChatAction")) >= 2, "the replacement's typing was not renewed")
        assert len(asked_by(token_a, "sendChatAction")) == 1 and len(muted()) == 1


@pytest.mark.parametrize("control", ["off", "revoke", "reload"])
def test_closing_runtime_access_stops_a_send_in_flight_before_the_unload_callbacks(tmp_path, monkeypatch, control):
    """Off, a revoked grant and a reload all unload through the real lifecycle, which closes the API's runtime
    access before it stops the companion. With that stop held for seconds and the first part of a long reply
    still unanswered, the reply's next request is refused at once: after access closed nothing reaches the Bot
    API, although the skill's unload callbacks (which close the token binding too) have not run yet."""
    import time

    from ouroboros.skill_loader import GRANTS_FILENAME, skill_state_dir

    first_part, stop_held = threading.Event(), threading.Event()
    answered_first = threading.Event()

    def hold(_token, method):
        if method == "sendMessage" and not answered_first.is_set():
            answered_first.set()
            first_part.wait(30)  # the first part stays unanswered until access has closed

    with _native_telegram(tmp_path, monkeypatch, token="111:SYNTHETIC-A", hold=hold) as world:
        assert extension_loader.load_extension(world.skill, world.load_settings, drive_root=world.drive,
                                               skills=[world.skill], repo_path="") is None
        world.until(lambda: ("111:SYNTHETIC-A", "getUpdates") in world.asked, "the poller never started")
        api = world.api()
        real_stop, callbacks_ran = world.supervisor.stop, []

        def slow_stop(skill_name, name, timeout_sec=5.0):
            stop_held.wait(30)  # a companion whose stop takes long
            return real_stop(skill_name, name, timeout_sec=timeout_sec)

        monkeypatch.setattr(world.supervisor, "stop", slow_stop)
        real_callback = extension_loader._run_unload_callback
        monkeypatch.setattr(extension_loader, "_run_unload_callback",
                            lambda *args, **kwargs: (callbacks_ran.append(time.monotonic()),
                                                     real_callback(*args, **kwargs))[-1])
        world.bus.publish("chat.outbound", {"chat_id": 1, "role": "assistant", "markdown": False,
                                            "text": "Every word of this answer matters. " * 700})
        world.until(answered_first.is_set, "the reply's first part never reached the Bot API")

        def lifecycle():
            if control == "off":
                save_enabled(world.drive, "telegram", False)
            elif control == "revoke":
                (skill_state_dir(world.drive, "telegram") / GRANTS_FILENAME).unlink()
            else:  # a new revision of the payload, reviewed: the live load is replaced
                (world.payload / "SKILL.md").write_text(
                    (world.payload / "SKILL.md").read_text(encoding="utf-8") + "\n", encoding="utf-8")
                world.admit()
            return extension_loader.reconcile_extension("telegram", world.drive, world.load_settings, repo_path="")

        outcome: dict = {}
        cycling = threading.Thread(target=lambda: outcome.setdefault("state", lifecycle()), daemon=True)
        cycling.start()
        world.until(lambda: api._runtime_closing, "runtime access never closed")
        closed_at = len(world.asked)
        first_part.set()  # the in-flight part completes now, with its observed outcome
        time.sleep(1.5)
        assert not callbacks_ran, "the unload callbacks ran before the companion stop was released"
        assert world.asked[closed_at:] == [], f"requests after access closed: {world.asked[closed_at:]}"
        stop_held.set()
        cycling.join(30)
        assert outcome["state"]["action"] == ("extension_loaded" if control == "reload" else "extension_unloaded")
        late = [entry for entry in world.asked[closed_at:] if entry[1] == "sendMessage"]
        assert late == [], "the old load's reply went on after it was replaced"


class SimpleNamespaceBridge:
    def send_quiz_state(self, *_args, **_kwargs):
        pass


# The companion entry the Mini App witness runs in its private copy of the payload: the shipped companion.py
# unchanged, with only the two endpoints that would leave the machine replaced: the pinned cloudflared (here a
# stand-in that prints a fresh Quick Tunnel URL per generation and records where it would forward) and the
# public HTTPS probe of that URL (here: the stand-in tunnel is alive).
_SYNTHETIC_COMPANION = """
import json, os, secrets, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cloudflare_tunnel, companion
from public_gateway import PublicProbeOutcome


async def ensure_cloudflared(state_dir):
    root = cloudflare_tunnel._safe_state_root(Path(state_dir))
    binary = cloudflare_tunnel._installed_path(root, cloudflare_tunnel._current_asset())
    binary.parent.mkdir(parents=True, exist_ok=True)
    slug = "gen-" + secrets.token_hex(6)
    record = Path(state_dir) / f"synthetic-tunnel-{slug}.json"
    binary.write_text(
        "#!" + sys.executable + "\\n"
        "import json, sys, time\\n"
        f"open({str(record)!r}, 'w').write(json.dumps({{'host': {slug + '.trycloudflare.com'!r}, 'target': sys.argv[-1]}}))\\n"
        f"print('INF |  https://{slug}.trycloudflare.com', file=sys.stderr, flush=True)\\n"
        "while True: time.sleep(1)\\n", encoding="utf-8")
    binary.chmod(0o700)
    return binary


async def probe_public_gateway_once(public_url, tunnel, **_kwargs):
    return PublicProbeOutcome.READY if tunnel.returncode is None else PublicProbeOutcome.TUNNEL_EXITED


companion.ensure_cloudflared = ensure_cloudflared
companion.probe_public_gateway_once = probe_public_gateway_once
cloudflare_tunnel._verify_binary = lambda *_args, **_kwargs: None
companion.main()
"""


def _use_synthetic_tunnel(payload: pathlib.Path) -> None:
    (payload / "scripts" / "synthetic_companion.py").write_text(_SYNTHETIC_COMPANION, encoding="utf-8")
    manifest = payload / "SKILL.md"
    text = manifest.read_text(encoding="utf-8")
    assert "command: [python3, scripts/companion.py]" in text
    manifest.write_text(text.replace("command: [python3, scripts/companion.py]",
                                     "command: [python3, scripts/synthetic_companion.py]"), encoding="utf-8")


def _init_data(token: str, owner: int) -> str:
    """Telegram's signed WebApp launch data for ``owner``, as a client of the bot ``token`` would send it."""
    import hashlib
    import time
    import urllib.parse

    fields = {"auth_date": str(int(time.time())), "user": json.dumps({"id": owner, "first_name": "Owner"})}
    check = "\n".join(f"{key}={value}" for key, value in sorted(fields.items()))
    secret = hmac.new(b"WebAppData", token.encode(), hashlib.sha256).digest()
    return urllib.parse.urlencode({**fields, "hash": hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()})


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX executable stand-in and SIGTERM shutdown")
def test_a_rotated_token_ends_the_old_mini_app_generation_and_the_new_one_serves(tmp_path, monkeypatch):
    """The real companion, sidecar, sessions and menu manager, respawned by the host's settings refresh: after
    the token rotates, the old generation's cookie, WebSocket, port and public URL are gone, the old bot's
    launch data no longer opens a session, and the new generation serves under a new URL its own bot's menu
    points to. Only cloudflared and the public probe are stand-ins; nothing leaves the machine."""
    import asyncio
    import socket
    import time

    import httpx
    import uvicorn
    import websockets
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, PlainTextResponse
    from starlette.routing import Route, WebSocketRoute

    async def echo(websocket):
        await websocket.accept()
        try:
            while True:
                await websocket.send_text("core:" + await websocket.receive_text())
        except Exception:
            return

    core = Starlette(routes=[Route("/api/health", lambda _r: JSONResponse({"status": "ok"})),
                             Route("/", lambda _r: PlainTextResponse("the core")),
                             WebSocketRoute("/ws", echo)])
    core_server = uvicorn.Server(uvicorn.Config(core, host="127.0.0.1", port=0, log_level="warning", lifespan="off"))
    core_thread = threading.Thread(target=core_server.run, daemon=True)
    core_thread.start()
    deadline = time.monotonic() + 30
    while not core_server.started:
        assert time.monotonic() < deadline, "the stand-in core never started"
        time.sleep(0.02)
    core_port = core_server.servers[0].sockets[0].getsockname()[1]
    token_a, token_b = "111:SYNTHETIC-A", "222:SYNTHETIC-B"

    def generation(world, after=()):
        """The public URL the companion published once ready, and the local port its tunnel forwards to."""
        world.until(lambda: world.published().get("state") == "ready"
                    and world.published().get("public_url") not in after, "the Mini App never became ready",
                    timeout=60.0)
        url = world.published()["public_url"]
        host = url[len("https://"):].rstrip("/")
        record = json.loads((world.state / f"synthetic-tunnel-{host.split('.')[0]}.json").read_text())
        return url, host, int(record["target"].rsplit(":", 1)[1])

    def request(port, host, method, path, *, cookie="", json_body=None):
        headers = {"Host": host, "Origin": f"https://{host}"}
        if cookie:
            headers["Cookie"] = cookie
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", trust_env=False) as client:
            return client.request(method, path, headers=headers, json=json_body)

    def session(port, host, token):
        response = request(port, host, "POST", "/__telegram/auth", json_body={"init_data": _init_data(token, 42)})
        assert response.status_code == 204, (response.status_code, response.text)
        value = response.headers["set-cookie"].split(";", 1)[0]
        assert value.startswith("__Host-ouroboros-telegram=")
        return value

    async def open_socket(port, host, cookie):
        sock = socket.create_connection(("127.0.0.1", port))
        connection = await websockets.connect(f"ws://{host}/ws", sock=sock, origin=f"https://{host}",
                                              additional_headers={"Cookie": cookie}, open_timeout=10)
        await connection.send("hello")
        assert await asyncio.wait_for(connection.recv(), 10) == "core:hello"
        return connection

    try:
        with _native_telegram(tmp_path, monkeypatch, token=token_a, miniapp="on",
                              payload_edit=_use_synthetic_tunnel, core_port=core_port) as world:
            assert extension_loader.load_extension(world.skill, world.load_settings, drive_root=world.drive,
                                                   skills=[world.skill], repo_path="") is None
            old_url, old_host, old_port = generation(world)
            first = world.runtime().process
            old_cookie = session(old_port, old_host, token_a)
            assert request(old_port, old_host, "GET", "/__telegram/session", cookie=old_cookie).status_code == 204
            assert request(old_port, old_host, "GET", "/", cookie=old_cookie).text == "the core"
            assert world.menu_sets[-1][0] == token_a and world.menu_sets[-1][1]["web_app"]["url"] == old_url
            loop = asyncio.new_event_loop()
            try:
                old_socket = loop.run_until_complete(open_socket(old_port, old_host, old_cookie))

                world.settings_save(token_b)  # the owner's Save: applied by the host refresh, not by Disable/Enable
                new_url, new_host, new_port = generation(world, after=(old_url,))
                assert first.poll() is not None, "the old generation's process outlived the rotation"
                with pytest.raises(websockets.ConnectionClosed):  # the old socket ended with its process
                    loop.run_until_complete(asyncio.wait_for(old_socket.recv(), 10))
                new_cookie = session(new_port, new_host, token_b)
                new_socket = loop.run_until_complete(open_socket(new_port, new_host, new_cookie))
                loop.run_until_complete(new_socket.close())
            finally:
                loop.close()
            assert new_url != old_url and new_port != old_port
            with pytest.raises(httpx.ConnectError):  # nothing listens where the old generation served
                request(old_port, old_host, "GET", "/__telegram/session", cookie=old_cookie)
            assert request(new_port, old_host, "GET", "/", cookie=new_cookie).status_code == 421, "the old URL"
            assert request(new_port, new_host, "GET", "/__telegram/session", cookie=old_cookie).status_code == 401
            refused = request(new_port, new_host, "POST", "/__telegram/auth",
                              json_body={"init_data": _init_data(token_a, 42)})
            assert refused.status_code == 401, "the old bot's launch data opened a session"
            assert request(new_port, new_host, "GET", "/", cookie=new_cookie).text == "the core"
            menu = [(token, button.get("web_app", {}).get("url")) for token, button in world.menu_sets]
            assert menu[-1] == (token_b, new_url), menu
            assert (token_a, None) in menu, "the old bot's menu was not given back on its way out"
    finally:
        core_server.should_exit = True
        core_thread.join(10)


@pytest.mark.parametrize("manifest", ["", "/operator/my node", "/generation/old"],
                         ids=["automatic", "manifest-choice", "manifest-equal-to-selected"])
def test_companion_respawn_refreshes_automatic_driver_and_preserves_manifest(tmp_path, monkeypatch, manifest):
    """Astra R2: a manifest naming the very path selected at build time kept the automatic marker, so a respawn
    after another generation was preferred rewrote the manifest's literal choice."""
    import os
    from ouroboros import claudexor_runtime, extension_companion, node_runtime

    monkeypatch.setenv("PLAYWRIGHT_NODEJS_PATH", "")
    monkeypatch.setenv("OUROBOROS_PLAYWRIGHT_NODE_SELECTED", "")
    generation = ["/generation/old"]
    manager = types.SimpleNamespace(reviewed_node=lambda **kw: generation[0])
    monkeypatch.setattr(claudexor_runtime, "get_runtime_manager", lambda: manager)
    env = companion_spawn_env(skill="demo", skill_dir=None, state_dir=tmp_path, spec={"env":
        {"PLAYWRIGHT_NODEJS_PATH": manifest} if manifest else {}},
        env_allow=[], granted_upper=[], token="fixture", settings={})
    output = tmp_path / "child.json"
    descriptor = CompanionDescriptor("demo", "node", [sys.executable, "-c",
        # Published whole (a temporary name, then os.replace): the poll below never reads a half-written file.
        "import json,os,pathlib,sys,time; t=pathlib.Path(sys.argv[1] + '.tmp'); "
        "t.write_text(json.dumps(dict(os.environ))); os.replace(t, sys.argv[1]); time.sleep(60)",
        str(output)], tmp_path, env)
    supervisor = CompanionSupervisor(tmp_path)
    monkeypatch.setattr(extension_companion, "_SERVER_PROCESS_PID", os.getpid())
    try:
        for current in ("/generation/old", "/generation/new"):
            generation[0] = current
            assert supervisor.start(descriptor)
            deadline = time.monotonic() + 20
            while not output.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            actual = json.loads(output.read_text())
            assert actual[node_runtime.PLAYWRIGHT_NODE_ENV] == (manifest or current)
            assert (node_runtime.SELECTED_DRIVER_NODE_ENV in actual) == (not manifest)
            supervisor.stop("demo", "node", timeout_sec=1)
            output.unlink()
    finally:
        supervisor.stop_all(timeout_sec=1)
