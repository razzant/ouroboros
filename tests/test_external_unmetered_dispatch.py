"""Focused disclosure tests for model-capable external skill processes."""

from __future__ import annotations

import io
import itertools
import json
import pathlib
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from ouroboros import extension_companion as companion_mod
from ouroboros import extension_process_runner as extension_runner
from ouroboros.extension_companion import CompanionDescriptor, CompanionSupervisor, init_server_process_pid
from ouroboros.extension_loader import PluginAPIImpl, _PluginAPIConfig
from ouroboros.tools.extension_dispatch import dispatch_extension_tool
from ouroboros.skill_loader import compute_content_hash, save_skill_grants
from ouroboros.tools import skill_exec
from ouroboros.tools.registry import ToolContext
from ouroboros.usage_accounting import UsageScope, usage_scope
from tests.test_skill_exec import _build_skill, _make_ctx, _mark_reviewed_and_enabled
from tests._usage_store_testing import ledger_rows


def _external_rows(drive_root: pathlib.Path) -> list[dict]:
    return [row for row in ledger_rows(drive_root) if row.get("kind") == "external_unmetered"]


def _prepare_script_skill(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_capable: bool = True,
    state_root: pathlib.Path | None = None,
):
    skills_root = tmp_path / "skills"
    manifest = None
    if model_capable:
        manifest = (
            "---\n"
            "name: alpha\n"
            "description: Model-capable script.\n"
            "version: 0.1.0\n"
            "type: script\n"
            "runtime: python3\n"
            "env_from_settings: [OPENROUTER_API_KEY]\n"
            "scripts:\n"
            "  - name: hello.py\n"
            "    description: Print hello.\n"
            "---\n"
            "# body\n"
        )
    skill_dir = _build_skill(
        skills_root, "alpha", script_body="print('ok')\n", manifest=manifest,
    )
    ctx = _make_ctx(tmp_path)
    lifecycle_root = state_root or ctx.drive_root
    _mark_reviewed_and_enabled(lifecycle_root, skill_dir, "alpha")
    if model_capable:
        save_skill_grants(
            lifecycle_root,
            "alpha",
            ["OPENROUTER_API_KEY"],
            content_hash=compute_content_hash(skill_dir),
            requested_keys=["OPENROUTER_API_KEY"],
        )
        monkeypatch.setattr(
            skill_exec,
            "load_settings",
            lambda: {"OPENROUTER_API_KEY": "test-provider-key"},
        )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    return ctx, skill_dir


def test_skill_exec_discloses_once_with_canonical_lineage(tmp_path, monkeypatch):
    budget_root = tmp_path / "canonical-budget"
    budget_root.mkdir()
    ctx, _skill_dir = _prepare_script_skill(
        tmp_path, monkeypatch, state_root=budget_root,
    )
    ctx.task_id = "child-task"
    ctx.budget_drive_root = str(budget_root)
    ctx.task_metadata = {
        "budget_drive_root": str(budget_root),
        "root_task_id": "root-task",
        "parent_task_id": "parent-task",
    }

    def fake_run(*_args, on_spawn=None, **_kwargs):
        assert on_spawn is not None
        on_spawn()
        on_spawn()  # The stable invocation id makes replay idempotent.
        return 0, b"ok\n", b"", False

    monkeypatch.setattr(skill_exec, "_run_skill_subprocess", fake_run)

    result = json.loads(skill_exec._handle_skill_exec(ctx, skill="alpha", script="scripts/hello.py"))

    assert result["exit_code"] == 0
    rows = _external_rows(budget_root)
    assert len(rows) == 1
    expected = {
        "task_id": "child-task",
        "root_task_id": "root-task",
        "parent_task_id": "parent-task",
        "provider": "external-skill",
        "category": "external_skill",
        "source": "skill_exec:alpha:scripts/hello.py",
        "cost_usd": None,
        "cost_final": False,
    }
    assert {key: rows[0].get(key) for key in expected} == expected
    assert _external_rows(ctx.drive_root) == []


def test_ordinary_script_process_is_not_false_unmetered(tmp_path, monkeypatch):
    ctx, _skill_dir = _prepare_script_skill(tmp_path, monkeypatch, model_capable=False)

    def fake_run(*_args, on_spawn=None, **_kwargs):
        assert on_spawn is None
        return 0, b"ok\n", b"", False

    monkeypatch.setattr(skill_exec, "_run_skill_subprocess", fake_run)
    result = json.loads(skill_exec._handle_skill_exec(ctx, skill="alpha", script="scripts/hello.py"))

    assert result["exit_code"] == 0
    assert _external_rows(ctx.drive_root) == []


def test_skill_exec_preflight_and_spawn_failures_do_not_disclose(tmp_path, monkeypatch):
    ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=tmp_path / "drive")
    monkeypatch.setattr(skill_exec, "_skill_tool_preflight", lambda _ctx: "blocked")
    monkeypatch.setattr(
        skill_exec,
        "record_unmetered_external_dispatch",
        lambda *_args, **_kwargs: pytest.fail("preflight must not disclose a dispatch"),
    )
    assert skill_exec._handle_skill_exec(ctx, skill="alpha", script="run.py") == "blocked"

    calls = []
    monkeypatch.setattr(skill_exec, "Popen", lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError()))
    with pytest.raises(FileNotFoundError):
        skill_exec._run_skill_subprocess(
            ["missing-runtime"],
            cwd=str(tmp_path),
            env={},
            timeout_sec=1,
            stdout_cap=128,
            stderr_cap=128,
            on_spawn=lambda: calls.append("spawned"),
        )
    assert calls == []


def test_skill_exec_timeout_keeps_one_post_spawn_disclosure(tmp_path, monkeypatch):
    class HangingProcess:
        def __init__(self):
            self.stdout = io.BytesIO()
            self.stderr = io.BytesIO()
            self.returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = -9
            return self.returncode

    spawned = {"value": False}
    process = HangingProcess()

    def fake_popen(*_args, **_kwargs):
        spawned["value"] = True
        return process

    monkeypatch.setattr(skill_exec, "Popen", fake_popen)
    # Force the first loop iteration past the deadline without sleeping. Three
    # reads: the child's start stamp (typed process facts), the deadline, and
    # the first loop check.
    ticks = iter((0.0, 0.0, 2.0))
    # Replace the module reference, not attributes on the process-global
    # ``time`` module (pytest itself calls monotonic on Windows).
    monkeypatch.setattr(
        skill_exec,
        "time",
        SimpleNamespace(monotonic=lambda: next(ticks), sleep=lambda _seconds: None),
    )
    monkeypatch.setattr(skill_exec, "_kill_process_group", lambda proc: setattr(proc, "returncode", -9))
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="task-1")

    def disclose():
        assert spawned["value"] is True
        skill_exec._record_skill_exec_dispatch(
            ctx,
            dispatch_id="stable-timeout-id",
            skill_name="alpha",
            script_rel="scripts/run.py",
        )

    with pytest.raises(subprocess.TimeoutExpired):
        skill_exec._run_skill_subprocess(
            ["runtime", "script"],
            cwd=str(tmp_path),
            env={},
            timeout_sec=1,
            stdout_cap=128,
            stderr_cap=128,
            on_spawn=disclose,
        )

    rows = _external_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["task_id"] == "task-1"


def test_skill_exec_uses_bound_lineage_when_tool_context_is_sparse(tmp_path):
    child_drive = tmp_path / "child"
    child_drive.mkdir()
    budget_root = tmp_path / "budget"
    budget_root.mkdir()
    ctx = ToolContext(repo_dir=tmp_path, drive_root=child_drive, task_id="bound-child")
    scope = UsageScope(
        drive_root=budget_root,
        task_id="bound-child",
        root_task_id="bound-root",
        parent_task_id="bound-parent",
    )

    with usage_scope(scope):
        skill_exec._record_skill_exec_dispatch(
            ctx,
            dispatch_id="bound-skill-dispatch",
            skill_name="alpha",
            script_rel="run.py",
        )

    row = _external_rows(budget_root)[0]
    assert (row["task_id"], row["root_task_id"], row["parent_task_id"]) == (
        "bound-child",
        "bound-root",
        "bound-parent",
    )
    assert _external_rows(child_drive) == []


@pytest.mark.parametrize("kind", ["tool", "route", "ws"])
def test_extension_dispatch_surfaces_disclose_once(kind, tmp_path, monkeypatch):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    skill = SimpleNamespace(name="alpha", skill_dir=skill_dir)
    monkeypatch.setattr(extension_runner, "_skill_for_dispatch", lambda *_args, **_kwargs: skill)
    monkeypatch.setattr(extension_runner, "_base_env_for_skill", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(extension_runner, "_extension_has_model_credentials", lambda *_args: True)

    def fake_run(_payload, **kwargs):
        callback = kwargs["on_spawn"]
        callback()
        callback()  # Same dispatch id must not append a second row.
        if kind == "route":
            return {"route": {"kind": "json", "data": {}}}
        return {"result": "ok"}

    monkeypatch.setattr(extension_runner, "_run_child", fake_run)
    if kind == "route":
        from contextlib import contextmanager
        @contextmanager
        def fake_child(payload, **kwargs):
            fake_run(payload, **kwargs)
            yield None
        monkeypatch.setattr(extension_runner, "_child_process", fake_child)
    expected_task = "extension:alpha"
    expected_root = "extension:alpha"
    expected_parent = ""

    if kind == "tool":
        budget_root = tmp_path / "budget"
        budget_root.mkdir()
        ctx = ToolContext(
            repo_dir=repo_dir,
            drive_root=drive_root,
            budget_drive_root=str(budget_root),
            task_id="child-task",
            task_metadata={
                "budget_drive_root": str(budget_root),
                "root_task_id": "root-task",
                "parent_task_id": "parent-task",
            },
        )
        extension_runner.dispatch_extension_tool_subprocess(
            {"skill": "alpha", "name": "echo", "skills_repo_path": str(tmp_path)},
            ctx,
            {},
        )
        ledger_root = budget_root
        expected_task, expected_root, expected_parent = "child-task", "root-task", "parent-task"
        expected_source = "extension_tool:alpha:echo"
    elif kind == "route":
        response = extension_runner.dispatch_extension_route_subprocess(
            {"skill": "alpha", "path": "/hello", "skills_repo_path": str(tmp_path)},
            {},
            drive_root=drive_root,
            repo_dir=repo_dir,
        )
        assert _external_rows(drive_root) == [], "preparing a response is not a physical dispatch"
        with response.child_factory():
            pass
        ledger_root = drive_root
        expected_source = "extension_route:alpha:/hello"
    else:
        extension_runner.dispatch_extension_ws_subprocess(
            {"skill": "alpha", "type": "alpha.ping", "skills_repo_path": str(tmp_path)},
            {},
            drive_root=drive_root,
            repo_dir=repo_dir,
        )
        ledger_root = drive_root
        expected_source = "extension_ws:alpha:alpha.ping"

    rows = _external_rows(ledger_root)
    assert len(rows) == 1
    assert rows[0]["task_id"] == expected_task
    assert rows[0]["root_task_id"] == expected_root
    assert rows[0]["parent_task_id"] == expected_parent
    assert rows[0]["provider"] == "external-extension"
    assert rows[0]["category"] == "external_skill"
    assert rows[0]["source"] == expected_source
    assert rows[0]["cost_usd"] is None
    assert rows[0]["cost_final"] is False


def test_ordinary_extension_process_is_not_false_unmetered(tmp_path, monkeypatch):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    skill = SimpleNamespace(name="alpha", skill_dir=skill_dir)
    monkeypatch.setattr(extension_runner, "_skill_for_dispatch", lambda *_args, **_kwargs: skill)
    monkeypatch.setattr(extension_runner, "_base_env_for_skill", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(extension_runner, "_extension_has_model_credentials", lambda *_args: False)

    def fake_run(_payload, **kwargs):
        assert kwargs["on_spawn"] is None
        return {"result": "ok"}

    monkeypatch.setattr(extension_runner, "_run_child", fake_run)
    ctx = ToolContext(repo_dir=repo_dir, drive_root=drive_root, task_id="ordinary")
    assert extension_runner.dispatch_extension_tool_subprocess(
        {"skill": "alpha", "name": "echo", "skills_repo_path": str(tmp_path)},
        ctx,
        {},
    ) == "ok"
    assert _external_rows(drive_root) == []


def test_extension_model_capability_requires_access_grant_and_value(tmp_path, monkeypatch):
    skill = SimpleNamespace(
        name="alpha",
        manifest=SimpleNamespace(
            permissions=["read_settings"],
            env_from_settings=["OPENROUTER_API_KEY"],
        ),
    )
    monkeypatch.setattr(
        "ouroboros.config.load_settings",
        lambda: {"OPENROUTER_API_KEY": "test-provider-key"},
    )
    monkeypatch.setattr(
        extension_runner,
        "grant_status_for_skill",
        lambda *_args: {"granted_keys": ["GITHUB_TOKEN"]},
    )
    assert extension_runner._extension_has_model_credentials(skill, tmp_path) is False
    monkeypatch.setattr(
        extension_runner,
        "grant_status_for_skill",
        lambda *_args: {"granted_keys": ["OPENROUTER_API_KEY"]},
    )
    assert extension_runner._extension_has_model_credentials(skill, tmp_path) is True
    monkeypatch.setattr("ouroboros.config.load_settings", lambda: {"OPENROUTER_API_KEY": ""})
    assert extension_runner._extension_has_model_credentials(skill, tmp_path) is False


@pytest.mark.parametrize(
    "permissions,allowed,granted,value,expected",
    [
        (["read_settings"], ["OPENROUTER_API_KEY"], ["OPENROUTER_API_KEY"], "key", True),
        ([], ["OPENROUTER_API_KEY"], ["OPENROUTER_API_KEY"], "key", False),
        (["read_settings"], [], ["OPENROUTER_API_KEY"], "key", False),
        (["read_settings"], ["OPENROUTER_API_KEY"], [], "key", False),
        (["read_settings"], ["OPENROUTER_API_KEY"], ["OPENROUTER_API_KEY"], "", False),
        (["read_settings"], ["GITHUB_TOKEN"], ["GITHUB_TOKEN"], "token", False),
    ],
)
def test_inprocess_model_probe_requires_actual_granted_provider_setting(
    permissions, allowed, granted, value, expected, tmp_path,
):
    settings_key = allowed[0] if allowed else "OPENROUTER_API_KEY"
    api = PluginAPIImpl(_PluginAPIConfig(
        skill_name="alpha",
        permissions=permissions,
        env_allowlist=allowed,
        state_dir=tmp_path,
        settings_reader=lambda: {settings_key: value},
        granted_keys=granted,
    ))

    assert api._model_credential_available() is expected


def test_inprocess_lifecycle_callbacks_are_opaque_dispatches(tmp_path):
    api = PluginAPIImpl(_PluginAPIConfig(
        skill_name="alpha",
        permissions=["read_settings"],
        env_allowlist=["OPENROUTER_API_KEY"],
        state_dir=tmp_path / "state" / "skills" / "alpha",
        drive_root=tmp_path,
        settings_reader=lambda: {"OPENROUTER_API_KEY": "test-provider-key"},
        granted_keys=["OPENROUTER_API_KEY"],
    ))
    calls = []
    wrapped = api._wrap_runtime_handler(
        lambda event: calls.append(event),
        opaque_surface=("event", "task.completed"),
    )

    api._disclose_model_capable_dispatch("register", "register")
    wrapped({"task_id": "t1"})

    assert calls == [{"task_id": "t1"}]
    rows = _external_rows(tmp_path)
    assert len(rows) == 2
    assert {row["source"] for row in rows} == {
        "extension_register:alpha:register",
        "extension_event:alpha:task.completed",
    }


def test_inprocess_extension_tool_discloses_before_handler(tmp_path, monkeypatch):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="inproc-task")
    calls = []
    ext_tool = {
        "name": "ext_5_alpha_echo",
        "skill": "alpha",
        "handler": lambda: calls.append("handler") or "ok",
        "_model_credential_probe": lambda: True,
    }
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_args, **_kwargs: (True, ""))
    monkeypatch.setattr("ouroboros.extension_loader.is_extension_live", lambda *_args, **_kwargs: True)

    assert dispatch_extension_tool(ctx, ext_tool["name"], ext_tool, {}) == "ok"
    assert calls == ["handler"]
    rows = _external_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["source"] == "extension_tool:alpha:ext_5_alpha_echo"


def test_inprocess_extension_disclosure_failure_blocks_handler(tmp_path, monkeypatch):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="inproc-task")
    calls = []
    ext_tool = {
        "name": "ext_5_alpha_echo",
        "skill": "alpha",
        "handler": lambda: calls.append("handler") or "ok",
        "_model_credential_probe": lambda: True,
    }
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_args, **_kwargs: (True, ""))
    monkeypatch.setattr("ouroboros.extension_loader.is_extension_live", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        extension_runner,
        "record_unmetered_external_dispatch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("ledger down")),
    )

    result = dispatch_extension_tool(ctx, ext_tool["name"], ext_tool, {})

    assert "model-cost disclosure failed" in result
    assert calls == []


@pytest.mark.parametrize("surface_kind,surface", [
    ("tool", "ext_5_alpha_echo"),
    ("route", "/api/extensions/alpha/run"),
    ("ws", "ext_5_alpha_ping"),
])
def test_inprocess_dispatch_helper_records_each_opaque_invocation(
    surface_kind, surface, tmp_path,
):
    spec = {"skill": "alpha", "_model_credential_probe": lambda: True}

    extension_runner.disclose_inprocess_extension_dispatch(
        spec, drive_root=tmp_path, surface_kind=surface_kind, surface=surface,
    )
    extension_runner.disclose_inprocess_extension_dispatch(
        spec, drive_root=tmp_path, surface_kind=surface_kind, surface=surface,
    )

    rows = _external_rows(tmp_path)
    assert len(rows) == 2
    assert len({row["attempt_id"] for row in rows}) == 2
    assert {row["source"] for row in rows} == {
        f"extension_{surface_kind}:alpha:{surface}",
    }


def test_model_capable_companion_spawn_and_restart_each_disclose(tmp_path):
    init_server_process_pid()
    supervisor = CompanionSupervisor(tmp_path)
    descriptor = CompanionDescriptor(
        skill_name="alpha",
        name="model-daemon",
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        env={"OPENROUTER_API_KEY": "test-provider-key"},
    )

    assert supervisor.start(descriptor) is True
    assert supervisor.start(descriptor) is True  # already running: no new dispatch
    assert len(_external_rows(tmp_path)) == 1
    supervisor.stop("alpha", "model-daemon", timeout_sec=1)
    assert supervisor.start(descriptor) is True
    assert len(_external_rows(tmp_path)) == 2
    supervisor.stop("alpha", "model-daemon", timeout_sec=1)


class _SyntheticCompanion:
    """A spawned companion that dies only when it honours a kill request."""

    stdout = stderr = None

    def __init__(self, pid: int, honours_kill: bool):
        self.pid, self.honours_kill = pid, honours_kill
        self.kill_requests, self.exited = 0, threading.Event()

    def request_kill(self, *_args):
        self.kill_requests += 1
        if self.honours_kill:
            self.exited.set()

    def poll(self):
        return -9 if self.exited.is_set() else None

    def wait(self, timeout=None):
        self.exited.wait(timeout)
        return self.poll()


def _isolate_companion_platform(monkeypatch, *, windows: bool, honours_kill: list, ledger: dict,
                                assign_ok: bool = True):
    """Replace every physical seam: spawn, kill, Panic request, custody, clock and a tracked Job."""
    spawned, jobs, members = [], {"created": [], "terminated": [], "closed": []}, {}

    def spawn(*_args, **_kwargs):
        spawned.append(_SyntheticCompanion(20001 + len(spawned), honours_kill[len(spawned)]))
        return spawned[-1]

    def create_job():
        jobs["created"].append(f"job-{len(jobs['created'])}")
        return jobs["created"][-1]

    def assign(job, pid):
        members[job] = next(proc for proc in spawned if proc.pid == pid)
        return assign_ok

    def disclose(*_args, **_kwargs):
        if ledger["down"]:
            raise RuntimeError("ledger down")

    monkeypatch.delenv("OUROBOROS_MANAGED_BY_LAUNCHER", raising=False)
    monkeypatch.setattr(companion_mod, "IS_WINDOWS", windows)
    monkeypatch.setattr(companion_mod.subprocess, "Popen", spawn)
    monkeypatch.setattr(companion_mod, "create_kill_on_close_job", create_job)
    monkeypatch.setattr(companion_mod, "assign_pid_to_job", assign)
    monkeypatch.setattr(companion_mod, "terminate_job", lambda job: jobs["terminated"].append(job)
                        or members[job].request_kill() or "")
    monkeypatch.setattr(companion_mod, "close_job", lambda job: jobs["closed"].append(job) or "")
    monkeypatch.setattr(companion_mod, "kill_process_tree", _SyntheticCompanion.request_kill)
    monkeypatch.setattr(companion_mod, "terminate_process_tree", _SyntheticCompanion.request_kill)
    monkeypatch.setattr(companion_mod, "request_process_tree_kill", lambda proc, job_handle=None: {
        "pid": proc.pid, "job": job_handle, "requested": True})
    monkeypatch.setattr(companion_mod, "record_unmetered_external_dispatch", disclose)
    monkeypatch.setattr("ouroboros.process_custody.record_process", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(companion_mod, "time", SimpleNamespace(
        monotonic=itertools.count(0, 10).__next__, sleep=lambda _sec: None))
    return spawned, jobs


def _join_companion_monitors():
    for thread in threading.enumerate():
        if thread.name.startswith("companion-monitor-"):
            thread.join(timeout=5)
            assert not thread.is_alive(), thread.name


@pytest.mark.parametrize("windows", [False, True])
def test_companion_disclosure_failure_settles_confirmed_death(tmp_path, monkeypatch, windows):
    init_server_process_pid()
    ledger = {"down": True}
    spawned, jobs = _isolate_companion_platform(monkeypatch, windows=windows, honours_kill=[True, True],
                                                ledger=ledger)
    supervisor = CompanionSupervisor(tmp_path)
    descriptor = CompanionDescriptor("alpha", "model-daemon", ["runtime"], tmp_path,
                                     {"OPENROUTER_API_KEY": "test-provider-key"})

    with pytest.raises(RuntimeError, match="ledger down"):
        supervisor.start(descriptor)
    assert spawned[0].kill_requests == 1 and supervisor.snapshot() == {}
    assert jobs["closed"] == jobs["created"] == (["job-0"] if windows else [])

    ledger["down"] = False  # the ordinary healthy start/already-running/stop path is unaffected
    assert supervisor.start(descriptor) is True
    assert supervisor.start(descriptor) is True
    assert [proc.pid for proc in spawned] == [20001, 20002]
    supervisor.stop("alpha", "model-daemon")
    _join_companion_monitors()
    assert supervisor.snapshot() == {}
    assert jobs["terminated"] == (["job-1"] if windows else [])
    # Stop and the monitor both observe the death; each Job still closes exactly once.
    assert jobs["closed"] == jobs["created"] == (["job-0", "job-1"] if windows else [])


@pytest.mark.parametrize("windows", [False, True])
def test_companion_disclosure_failure_retains_unconfirmed_owner_until_death(tmp_path, monkeypatch, windows):
    init_server_process_pid()
    spawned, jobs = _isolate_companion_platform(monkeypatch, windows=windows, honours_kill=[False],
                                                ledger={"down": True})
    supervisor = CompanionSupervisor(tmp_path)
    descriptor = CompanionDescriptor("alpha", "model-daemon", ["runtime"], tmp_path,
                                     {"OPENROUTER_API_KEY": "test-provider-key"})
    try:
        with pytest.raises(RuntimeError, match="ledger down"):
            supervisor.start(descriptor)
        with pytest.raises(RuntimeError, match="replacement refused"):
            supervisor.start(descriptor)  # no second spawn, no false already-running success
        supervisor.stop("alpha", "model-daemon")  # unconfirmed stop keeps the exact owner
        assert len(spawned) == 1 and supervisor.snapshot()["alpha:model-daemon"]["pid"] == 20001
        assert jobs["closed"] == []
        assert supervisor.panic_kill_all(request_only=True) == [
            {"pid": 20001, "job": "job-0" if windows else None, "requested": True}]
    finally:
        spawned[0].exited.set()
    _join_companion_monitors()
    assert supervisor.snapshot() == {}
    assert jobs["closed"] == jobs["created"] == (["job-0"] if windows else [])


@pytest.mark.parametrize("honours_kill", [True, False])
def test_windows_job_assignment_failure_retires_through_the_same_owner(tmp_path, monkeypatch, honours_kill):
    init_server_process_pid()
    spawned, jobs = _isolate_companion_platform(monkeypatch, windows=True, honours_kill=[honours_kill],
                                                ledger={"down": False}, assign_ok=False)
    supervisor = CompanionSupervisor(tmp_path)
    descriptor = CompanionDescriptor("alpha", "daemon", ["runtime"], tmp_path, {})
    try:
        with pytest.raises(RuntimeError, match="Windows Job Object"):
            supervisor.start(descriptor)
        assert jobs["closed"] == ["job-0"]  # the unattached Job never stands in for a kill
        assert spawned[0].kill_requests == 1
        if not honours_kill:
            with pytest.raises(RuntimeError, match="Job assignment failed.*replacement refused"):
                supervisor.start(descriptor)
            assert supervisor.snapshot()["alpha:daemon"]["pid"] == 20001
    finally:
        spawned[0].exited.set()
    _join_companion_monitors()
    assert supervisor.snapshot() == {} and len(spawned) == 1 and jobs["closed"] == ["job-0"]


@pytest.mark.parametrize("windows", [False, True])
@pytest.mark.parametrize("finale", ["panic", "retry"])
def test_loader_rollback_keeps_unconfirmed_companion_until_death(tmp_path, monkeypatch, windows, finale):
    """register -> publish -> start fails -> load_extension's unload -> stop keeps the owner."""
    from ouroboros import extension_loader, extension_plugin_api
    from tests._extension_loader_shared import _prepare_extension
    from tests._shared import clean_extension_runtime_state

    init_server_process_pid()
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    clean_extension_runtime_state()
    ledger = {"down": True}
    spawned, jobs = _isolate_companion_platform(monkeypatch, windows=windows, honours_kill=[False, True],
                                                ledger=ledger)
    supervisor = CompanionSupervisor(tmp_path / "companions")
    for module in (extension_plugin_api, extension_loader):
        monkeypatch.setattr(module, "get_global_supervisor", lambda: supervisor)
    # Only the spawn env is synthetic (model-capable); the registration and rollback are real.
    monkeypatch.setattr(extension_plugin_api, "companion_spawn_env",
                        lambda *_args, **_kwargs: {"OPENROUTER_API_KEY": "test-provider-key"})
    loaded, _repo, drive_root = _prepare_extension(
        tmp_path, "unmetered", "def register(api):\n    api.register_companion_process('daemon')\n",
        permissions=["companion_process"],
        extra_frontmatter="companion_processes:\n  - name: daemon\n    runtime: python3\n"
                          "    command: [\"python3\", \"scripts/daemon.py\"]\n",
    )

    def load():
        return extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root, _force_in_process=True)

    try:
        assert "ledger down" in load()
        assert "replacement refused" in load()  # the rollback kept the owner: no duplicate spawn
        assert len(spawned) == 1 and supervisor.snapshot()["unmetered:daemon"]["pid"] == 20001
        assert spawned[0].kill_requests >= 2 and jobs["closed"] == []
        if finale == "panic":
            assert supervisor.panic_kill_all(request_only=True) == [
                {"pid": 20001, "job": "job-0" if windows else None, "requested": True}]
        spawned[0].exited.set()
        _join_companion_monitors()
        assert supervisor.snapshot() == {}
        if finale == "retry":
            ledger["down"] = False
            assert load() is None
            assert supervisor.snapshot()["unmetered:daemon"]["pid"] == 20002
            extension_loader.unload_extension("unmetered")
            _join_companion_monitors()
            assert supervisor.snapshot() == {}
        assert jobs["closed"] == jobs["created"] == (
            [] if not windows else ["job-0"] if finale == "panic" else ["job-0", "job-1"])
    finally:
        for proc in spawned:
            proc.exited.set()
        _join_companion_monitors()
        clean_extension_runtime_state()


@pytest.mark.parametrize("windows", [False, True])
def test_reconcile_reports_retained_auto_restart_not_already_running(tmp_path, monkeypatch, windows):
    """live load -> crash -> auto-restart disclosure fails, kill unconfirmed -> real reconcile consumers."""
    from ouroboros import extension_health, extension_loader, extension_plugin_api
    from ouroboros.extension_reconcile_queue import process_extension_reconcile_requests, request_extension_reconcile
    from tests._extension_loader_shared import _prepare_extension
    from tests._shared import clean_extension_runtime_state

    init_server_process_pid()
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    clean_extension_runtime_state()
    ledger = {"down": False}
    spawned, jobs = _isolate_companion_platform(monkeypatch, windows=windows, honours_kill=[True, False, True],
                                                ledger=ledger)
    supervisor = CompanionSupervisor(tmp_path / "companions")
    for module in (extension_plugin_api, extension_loader):
        monkeypatch.setattr(module, "get_global_supervisor", lambda: supervisor)
    monkeypatch.setattr(extension_plugin_api, "companion_spawn_env",
                        lambda *_args, **_kwargs: {"OPENROUTER_API_KEY": "test-provider-key"})
    # Reconcile's health stamp would run git through the patched spawn seam.
    monkeypatch.setattr(extension_health, "fresh_code_stamp", lambda: ("test", "test-sha"))
    loaded, repo_root, drive_root = _prepare_extension(
        tmp_path, "restarted", "def register(api):\n    api.register_companion_process('daemon')\n",
        permissions=["companion_process"],
        extra_frontmatter="companion_processes:\n  - name: daemon\n    runtime: python3\n"
                          "    command: [\"python3\", \"scripts/daemon.py\"]\n",
    )

    def reconcile():
        return extension_loader.reconcile_extension(loaded.name, drive_root, lambda: {},
                                                    repo_path=str(repo_root))["companions"]["action"]

    try:
        assert extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root,
                                               _force_in_process=True) is None
        assert reconcile() == "already_running"  # healthy counterpart
        first_monitors = [t for t in threading.enumerate() if t.name.startswith("companion-monitor-")]
        ledger["down"] = True
        spawned[0].exited.set()  # unsuccessful exit within budget: the monitor restarts it itself
        for thread in first_monitors:
            thread.join(timeout=5)
            assert not thread.is_alive(), thread.name
        retained = supervisor.snapshot()["restarted:daemon"]
        assert retained["pid"] == 20002 and retained["retiring"] == "cost disclosure failed"

        assert reconcile() == "retained_unresolved"
        assert extension_loader.ensure_companions_running(
            loaded.name, drive_root, lambda: {}, repo_path=str(repo_root),
        ) == {"action": "retained_unresolved", "started": [], "missing": [],
              "retained": {"daemon": "cost disclosure failed"}}
        request_extension_reconcile(drive_root, loaded.name, reason="retained")
        processed = process_extension_reconcile_requests(drive_root, lambda: {}, repo_path=str(repo_root))
        assert processed[-1]["companions"]["action"] == "retained_unresolved"
        assert len(spawned) == 2 and jobs["closed"] == (["job-0"] if windows else [])  # no duplicate spawn

        spawned[1].exited.set()  # observed death settles the retained owner; no automatic restart
        _join_companion_monitors()
        assert supervisor.snapshot() == {} and len(spawned) == 2
        ledger["down"] = False
        assert reconcile() == "started_missing"  # replacement after observed death
        assert supervisor.snapshot()["restarted:daemon"]["pid"] == 20003
        assert supervisor.snapshot()["restarted:daemon"]["retiring"] == ""
        assert reconcile() == "already_running"
        extension_loader.unload_extension(loaded.name)
        _join_companion_monitors()
        assert supervisor.snapshot() == {} and len(spawned) == 3
        assert jobs["closed"] == jobs["created"] == (["job-0", "job-1", "job-2"] if windows else [])
    finally:
        for proc in spawned:
            proc.exited.set()
        _join_companion_monitors()
        clean_extension_runtime_state()


def test_extension_dispatch_inherits_bound_lineage_without_tool_context(tmp_path):
    scope = UsageScope(
        drive_root=tmp_path,
        task_id="bound-child",
        root_task_id="bound-root",
        parent_task_id="bound-parent",
        category="task",
        source="task",
    )
    with usage_scope(scope):
        extension_runner._record_extension_dispatch(
            dispatch_id="bound-extension-dispatch",
            drive_root=tmp_path,
            skill_name="alpha",
            surface_kind="route",
            surface="/bound",
        )

    row = _external_rows(tmp_path)[0]
    assert (row["task_id"], row["root_task_id"], row["parent_task_id"]) == (
        "bound-child",
        "bound-root",
        "bound-parent",
    )


def test_extension_child_spawn_failure_records_nothing(tmp_path, monkeypatch):
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    callbacks = []
    monkeypatch.setattr(
        extension_runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("spawn failed")),
    )

    with pytest.raises(OSError, match="spawn failed"):
        extension_runner._run_child(
            {"mode": "tool", "skill_name": "alpha"},
            skill_dir=skill_dir,
            drive_root=drive_root,
            repo_dir=repo_dir,
            env={},
            timeout_sec=1,
            on_spawn=lambda: callbacks.append("spawned"),
        )

    assert callbacks == []
    assert _external_rows(drive_root) == []


def test_extension_child_timeout_keeps_one_post_spawn_disclosure(tmp_path, monkeypatch):
    class HangingProcess:
        def __init__(self):
            self.stdin = None
            self.stdout = io.BytesIO()
            self.stderr = io.BytesIO()
            self.returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = -9
            return self.returncode

    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    process = HangingProcess()
    spawned = {"value": False}

    def fake_popen(*_args, **_kwargs):
        spawned["value"] = True
        return process

    # Three reads: the child's start stamp (typed process facts), the deadline,
    # and the first loop check.
    ticks = iter((0.0, 0.0, 2.0))
    monkeypatch.setattr(extension_runner.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        extension_runner,
        "time",
        SimpleNamespace(monotonic=lambda: next(ticks), sleep=lambda _seconds: None),
    )
    monkeypatch.setattr(extension_runner, "_kill_process_group", lambda proc: setattr(proc, "returncode", -9))

    def disclose():
        assert spawned["value"] is True
        extension_runner._record_extension_dispatch(
            dispatch_id="stable-extension-timeout",
            drive_root=drive_root,
            skill_name="alpha",
            surface_kind="tool",
            surface="slow",
        )

    with pytest.raises(extension_runner.ExtensionProcessError, match="timed out"):
        extension_runner._run_child(
            {"mode": "tool", "skill_name": "alpha"},
            skill_dir=skill_dir,
            drive_root=drive_root,
            repo_dir=repo_dir,
            env={},
            timeout_sec=1,
            on_spawn=disclose,
        )

    assert len(_external_rows(drive_root)) == 1
