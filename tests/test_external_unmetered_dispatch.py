"""Focused disclosure tests for model-capable external skill processes."""

from __future__ import annotations

import io
import json
import pathlib
import subprocess
import sys
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


def _external_rows(drive_root: pathlib.Path) -> list[dict]:
    path = drive_root / "state" / "usage_attempts.jsonl"
    if not path.exists():
        return []
    return [
        row
        for raw in path.read_text(encoding="utf-8").splitlines()
        if raw.strip()
        for row in [json.loads(raw)]
        if row.get("kind") == "external_unmetered"
    ]


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
        extension_runner.dispatch_extension_route_subprocess(
            {"skill": "alpha", "path": "/hello", "skills_repo_path": str(tmp_path)},
            {},
            drive_root=drive_root,
            repo_dir=repo_dir,
        )
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


def test_companion_disclosure_failure_kills_unregistered_process(tmp_path, monkeypatch):
    class SpawnedProcess:
        pid = 12345
        stdout = None
        stderr = None

        def poll(self):
            return None

    process = SpawnedProcess()
    killed = []
    init_server_process_pid()
    supervisor = CompanionSupervisor(tmp_path)
    descriptor = CompanionDescriptor(
        skill_name="alpha",
        name="model-daemon",
        command=["runtime"],
        cwd=tmp_path,
        env={"OPENROUTER_API_KEY": "test-provider-key"},
    )
    monkeypatch.setattr(companion_mod.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(companion_mod, "terminate_process_tree", lambda proc: killed.append(proc))
    monkeypatch.setattr(
        companion_mod,
        "record_unmetered_external_dispatch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("ledger down")),
    )

    with pytest.raises(RuntimeError, match="ledger down"):
        supervisor.start(descriptor)

    assert killed == [process]
    assert supervisor.snapshot() == {}


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
