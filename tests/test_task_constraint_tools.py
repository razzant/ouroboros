
from ouroboros.contracts.task_constraint import TaskConstraint, resolve_payload_path
from ouroboros.tools.core import _data_write
from ouroboros.tools.git import _str_replace_editor
from ouroboros.tools.registry import ToolContext


def _ctx(tmp_path):
    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    skill = drive / "skills" / "external" / "alpha"
    skill.mkdir(parents=True)
    return ToolContext(repo_dir=repo, drive_root=drive, task_constraint=TaskConstraint(mode="skill_repair", skill_name="alpha", payload_root="skills/external/alpha", allow_enable=False)), skill


def test_payload_relative_resolver_accepts_short_paths(tmp_path):
    ctx, skill = _ctx(tmp_path)
    assert resolve_payload_path(ctx.drive_root, ctx.task_constraint, "plugin.py") == skill / "plugin.py"
    assert resolve_payload_path(ctx.drive_root, ctx.task_constraint, "skills/external/alpha/plugin.py") == skill / "plugin.py"


def test_str_replace_editor_uses_payload_relative_path(tmp_path):
    ctx, skill = _ctx(tmp_path)
    target = skill / "plugin.py"
    target.write_text("hello = 1\n", encoding="utf-8")
    result = _str_replace_editor(ctx, "plugin.py", "hello = 1", "hello = 2")
    assert "Replaced" in result
    assert target.read_text(encoding="utf-8") == "hello = 2\n"
    assert not (ctx.repo_dir / "plugin.py").exists()


def test_data_write_uses_payload_relative_path(tmp_path):
    ctx, skill = _ctx(tmp_path)
    result = _data_write(ctx, "new_file.py", "VALUE = 1\n")
    assert "OK:" in result
    assert (skill / "new_file.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_data_read_and_list_use_payload_relative_paths(tmp_path):
    from ouroboros.tools.core import _data_list, _data_read
    ctx, skill = _ctx(tmp_path)
    (skill / "plugin.py").write_text("VALUE = 1\n", encoding="utf-8")
    (ctx.drive_root / "memory").mkdir()
    (ctx.drive_root / "memory" / "identity.md").write_text("secret\n", encoding="utf-8")

    assert "VALUE = 1" in _data_read(ctx, "plugin.py")
    listing = _data_list(ctx, ".")
    assert "plugin.py" in listing
    assert "secret" not in _data_read(ctx, "memory/identity.md")


def test_registry_repair_mode_reads_lists_skill_payload_root_without_bucket(tmp_path):
    from ouroboros.tools.registry import ToolRegistry

    ctx, skill = _ctx(tmp_path)
    (skill / "plugin.py").write_text("VALUE = 1\n", encoding="utf-8")
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry._ctx = ctx

    read_result = registry.execute("read_file", {"root": "skill_payload", "path": "plugin.py"})
    list_result = registry.execute("list_files", {"root": "skill_payload", "path": "."})

    assert "VALUE = 1" in read_result
    assert "READ_FILE_ERROR" not in read_result
    assert "plugin.py" in list_result
    assert "LIST_FILES_ERROR" not in list_result


def test_payload_absolute_other_skill_path_is_blocked(tmp_path):
    from ouroboros.tools.core import _data_read
    ctx, _skill = _ctx(tmp_path)
    assert "DATA_READ_BLOCKED" in _data_read(ctx, "skills/external/beta/plugin.py")


def test_repair_mode_blocks_code_search(tmp_path):
    from ouroboros.tools.registry import ToolRegistry
    ctx, _skill = _ctx(tmp_path)
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry._ctx = ctx
    result = registry.execute("search_code", {"query": "ToolRegistry"})
    assert "HEAL_MODE_BLOCKED" in result


def test_claude_code_edit_reverts_repair_sidecars(tmp_path, monkeypatch):
    from types import ModuleType, SimpleNamespace
    import sys
    from ouroboros.tools.shell import _claude_code_edit

    gateway = ModuleType("ouroboros.gateways.claude_code")
    gateway.resolve_claude_code_model = lambda: "test-model"
    gateway.DEFAULT_CLAUDE_CODE_MAX_TURNS = 1
    monkeypatch.setitem(sys.modules, "ouroboros.gateways.claude_code", gateway)

    ctx, skill = _ctx(tmp_path)
    sidecar = skill / ".self_authored.json"
    sidecar.write_text("original", encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    def fake_run_edit(**kwargs):
        sidecar.write_text("modified", encoding="utf-8")
        return SimpleNamespace(
            success=True,
            error="",
            result_text="ok",
            cost_usd=0.0,
            usage={},
            changed_files=[],
            diff_stat="",
            validation_summary="",
            to_tool_output=lambda: "OK",
        )

    gateway.run_edit = fake_run_edit

    result = _claude_code_edit(ctx, "edit", cwd=".")

    assert "SKILL_PAYLOAD_CONTROL_BLOCKED" in result
    assert sidecar.read_text(encoding="utf-8") == "original"


def test_claude_code_edit_reverts_normal_skill_sidecars(tmp_path, monkeypatch):
    from types import ModuleType, SimpleNamespace
    import sys
    from ouroboros.tools.shell import _claude_code_edit

    gateway = ModuleType("ouroboros.gateways.claude_code")
    gateway.resolve_claude_code_model = lambda: "test-model"
    gateway.DEFAULT_CLAUDE_CODE_MAX_TURNS = 1
    monkeypatch.setitem(sys.modules, "ouroboros.gateways.claude_code", gateway)

    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    skill = drive / "skills" / "external" / "alpha"
    skill.mkdir(parents=True)
    sidecar = skill / ".self_authored.json"
    sidecar.write_text("original", encoding="utf-8")
    ctx = ToolContext(repo_dir=repo, drive_root=drive)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def fake_run_edit(**kwargs):
        sidecar.write_text("modified", encoding="utf-8")
        return SimpleNamespace(
            success=True,
            error="",
            result_text="ok",
            cost_usd=0.0,
            usage={},
            changed_files=[],
            diff_stat="",
            validation_summary="",
            to_tool_output=lambda: "OK",
        )

    gateway.run_edit = fake_run_edit

    result = _claude_code_edit(ctx, "edit", cwd="skills/external/alpha")

    assert "SKILL_PAYLOAD_CONTROL_BLOCKED" in result
    assert sidecar.read_text(encoding="utf-8") == "original"


def test_claude_code_edit_omitted_cwd_ignores_stale_short_form(tmp_path, monkeypatch):
    from types import ModuleType, SimpleNamespace
    import sys
    from ouroboros.tools.shell import _claude_code_edit

    gateway = ModuleType("ouroboros.gateways.claude_code")
    gateway.resolve_claude_code_model = lambda: "test-model"
    gateway.DEFAULT_CLAUDE_CODE_MAX_TURNS = 1
    monkeypatch.setitem(sys.modules, "ouroboros.gateways.claude_code", gateway)

    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    (drive / "skills" / "external" / "alpha").mkdir(parents=True)
    ctx = ToolContext(repo_dir=repo, drive_root=drive)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    captured = {}

    def fake_run_edit(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            success=True,
            error="",
            result_text="ok",
            cost_usd=0.0,
            usage={},
            changed_files=[],
            diff_stat="",
            validation_summary="",
            to_tool_output=lambda: "OK",
        )

    gateway.run_edit = fake_run_edit

    result = _claude_code_edit(
        ctx,
        "edit repo",
        bucket="external",
        skill_name="alpha",
    )

    assert "SKILL_SHORT_FORM_IGNORED" in result
    assert captured["cwd"] == str(repo)


def test_registry_light_mode_blocks_omitted_cwd_short_form_claude_edit(tmp_path, monkeypatch):
    from types import ModuleType, SimpleNamespace
    import sys
    from ouroboros import config as cfg
    from ouroboros.tools.registry import ToolRegistry

    cfg.reset_runtime_mode_baseline_for_tests()
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    monkeypatch.delenv(cfg.BOOT_RUNTIME_MODE_ENV_KEY, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    gateway = ModuleType("ouroboros.gateways.claude_code")
    gateway.resolve_claude_code_model = lambda: "test-model"
    gateway.DEFAULT_CLAUDE_CODE_MAX_TURNS = 1
    called = {"value": False}

    def fake_run_edit(**kwargs):
        called["value"] = True
        return SimpleNamespace(
            success=True,
            error="",
            result_text="ok",
            cost_usd=0.0,
            usage={},
            changed_files=[],
            diff_stat="",
            validation_summary="",
            to_tool_output=lambda: "OK",
        )

    gateway.run_edit = fake_run_edit
    monkeypatch.setitem(sys.modules, "ouroboros.gateways.claude_code", gateway)

    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    (drive / "skills" / "external" / "alpha").mkdir(parents=True)
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)

    result = registry.execute(
        "claude_code_edit",
        {"prompt": "edit repo", "bucket": "external", "skill_name": "alpha"},
    )

    assert "LIGHT_MODE_BLOCKED" in result
    assert called["value"] is False


def test_claude_code_edit_rejects_cwd_outside_active_workspace(tmp_path, monkeypatch):
    from ouroboros.tools.shell import _claude_code_edit

    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    outside = tmp_path / "outside"
    repo.mkdir()
    drive.mkdir()
    outside.mkdir()
    ctx = ToolContext(repo_dir=repo, drive_root=drive)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    result = _claude_code_edit(ctx, "edit outside", cwd="../outside")
    absolute_result = _claude_code_edit(ctx, "edit outside", cwd=str(outside))

    assert "cwd escapes allowed edit roots" in result
    assert "cwd escapes allowed edit roots" in absolute_result


def test_claude_code_edit_workspace_prefers_workspace_cwd_over_data_skill(tmp_path, monkeypatch):
    from types import ModuleType, SimpleNamespace
    import sys
    from ouroboros.tools.shell import _claude_code_edit

    gateway = ModuleType("ouroboros.gateways.claude_code")
    gateway.resolve_claude_code_model = lambda: "test-model"
    gateway.DEFAULT_CLAUDE_CODE_MAX_TURNS = 1
    captured = {}

    def fake_run_edit(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            success=True,
            error="",
            result_text="ok",
            cost_usd=0.0,
            usage={},
            changed_files=[],
            diff_stat="",
            validation_summary="",
            to_tool_output=lambda: "OK",
        )

    gateway.run_edit = fake_run_edit
    monkeypatch.setitem(sys.modules, "ouroboros.gateways.claude_code", gateway)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    drive = tmp_path / "data"
    system_repo.mkdir()
    workspace.mkdir()
    (system_repo / "BIBLE.md").write_text("SYSTEM_BIBLE\n", encoding="utf-8")
    (workspace / "BIBLE.md").write_text("WORKSPACE_BIBLE\n", encoding="utf-8")
    (workspace / "skills" / "external" / "alpha").mkdir(parents=True)
    (drive / "skills" / "external" / "alpha").mkdir(parents=True)
    ctx = ToolContext(
        repo_dir=system_repo,
        drive_root=drive,
        workspace_root=workspace,
        workspace_mode="task",
    )

    result = _claude_code_edit(ctx, "edit", cwd="skills/external/alpha")

    assert result == "OK"
    assert captured["cwd"] == str((workspace / "skills" / "external" / "alpha").resolve())
    assert "SYSTEM_BIBLE" in captured["system_prompt"]
    assert "WORKSPACE_BIBLE" not in captured["system_prompt"]


def test_claude_code_edit_reverts_protected_runtime_file_changes(tmp_path, monkeypatch):
    from types import ModuleType, SimpleNamespace
    import sys
    from ouroboros import config as cfg
    from ouroboros.tools.shell import _claude_code_edit

    cfg.reset_runtime_mode_baseline_for_tests()
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.delenv(cfg.BOOT_RUNTIME_MODE_ENV_KEY, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    drive.mkdir()
    (repo / "BIBLE.md").write_text("original\n", encoding="utf-8")
    import subprocess

    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "add", "BIBLE.md"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=a@example.com", "-c", "user.name=A", "commit", "-m", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
    )

    gateway = ModuleType("ouroboros.gateways.claude_code")
    gateway.resolve_claude_code_model = lambda: "test-model"
    gateway.DEFAULT_CLAUDE_CODE_MAX_TURNS = 1

    def fake_run_edit(**kwargs):
        (repo / "BIBLE.md").write_text("mutated\n", encoding="utf-8")
        return SimpleNamespace(
            success=True,
            error="",
            result_text="ok",
            cost_usd=0.0,
            usage={},
            changed_files=["BIBLE.md"],
            diff_stat="",
            validation_summary="",
            to_tool_output=lambda: "OK",
        )

    gateway.run_edit = fake_run_edit
    monkeypatch.setitem(sys.modules, "ouroboros.gateways.claude_code", gateway)
    ctx = ToolContext(repo_dir=repo, drive_root=drive, branch_dev="main")

    result = _claude_code_edit(ctx, "edit protected")

    assert "CORE_PROTECTION_BLOCKED" in result
    assert (repo / "BIBLE.md").read_text(encoding="utf-8") == "original\n"

    cfg.reset_runtime_mode_baseline_for_tests()
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    monkeypatch.delenv(cfg.BOOT_RUNTIME_MODE_ENV_KEY, raising=False)

    result = _claude_code_edit(ctx, "edit protected")

    assert "CORE_PATCH_NOTICE" in result
    assert (repo / "BIBLE.md").read_text(encoding="utf-8") == "mutated\n"

    invalidations = []
    monkeypatch.setattr(
        "ouroboros.tools.shell._invalidate_advisory",
        lambda *args, **kwargs: invalidations.append(kwargs),
    )
    gateway.run_edit = lambda **kwargs: (
        (repo / "notes.txt").write_text("partial\n", encoding="utf-8"),
        SimpleNamespace(
            success=False,
            error="failed after partial edit",
            result_text="partial output",
            cost_usd=0.0,
            usage={},
            changed_files=[],
            diff_stat="",
            validation_summary="",
            to_tool_output=lambda: "FAILED",
        ),
    )[1]
    result = _claude_code_edit(ctx, "partial failed edit")

    assert "CLAUDE_CODE_ERROR" in result
    assert invalidations
    assert invalidations[-1]["source_tool"] == "claude_code_edit"


def test_claude_code_edit_reverts_created_skill_control_dirs(tmp_path, monkeypatch):
    from types import ModuleType, SimpleNamespace
    import sys
    from ouroboros.tools.shell import _claude_code_edit

    gateway = ModuleType("ouroboros.gateways.claude_code")
    gateway.resolve_claude_code_model = lambda: "test-model"
    gateway.DEFAULT_CLAUDE_CODE_MAX_TURNS = 1
    monkeypatch.setitem(sys.modules, "ouroboros.gateways.claude_code", gateway)

    ctx, skill = _ctx(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def fake_run_edit(**kwargs):
        marker = skill / ".ouroboros_env" / "marker.txt"
        marker.parent.mkdir(parents=True)
        marker.write_text("modified", encoding="utf-8")
        return SimpleNamespace(
            success=True,
            error="",
            result_text="ok",
            cost_usd=0.0,
            usage={},
            changed_files=[],
            diff_stat="",
            validation_summary="",
            to_tool_output=lambda: "OK",
        )

    gateway.run_edit = fake_run_edit

    result = _claude_code_edit(ctx, "edit", cwd=".")

    assert "SKILL_PAYLOAD_CONTROL_BLOCKED" in result
    assert not (skill / ".ouroboros_env").exists()


def test_claude_code_edit_reverts_existing_skill_control_dirs(tmp_path, monkeypatch):
    from types import ModuleType, SimpleNamespace
    import sys
    from ouroboros.tools.shell import _claude_code_edit

    gateway = ModuleType("ouroboros.gateways.claude_code")
    gateway.resolve_claude_code_model = lambda: "test-model"
    gateway.DEFAULT_CLAUDE_CODE_MAX_TURNS = 1
    monkeypatch.setitem(sys.modules, "ouroboros.gateways.claude_code", gateway)

    ctx, skill = _ctx(tmp_path)
    env_marker = skill / ".ouroboros_env" / "marker.txt"
    env_marker.parent.mkdir(parents=True)
    env_marker.write_text("before", encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def fake_run_edit(**kwargs):
        marker = skill / ".ouroboros_env" / "marker.txt"
        assert marker.read_text(encoding="utf-8") == "before"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("after", encoding="utf-8")
        return SimpleNamespace(
            success=True,
            error="",
            result_text="ok",
            cost_usd=0.0,
            usage={},
            changed_files=[],
            diff_stat="",
            validation_summary="",
            to_tool_output=lambda: "OK",
        )

    gateway.run_edit = fake_run_edit

    result = _claude_code_edit(ctx, "edit", cwd=".")

    assert "SKILL_PAYLOAD_CONTROL_BLOCKED" in result
    assert env_marker.read_text(encoding="utf-8") == "before"


def test_claude_code_edit_restores_control_dirs_on_sdk_failure(tmp_path, monkeypatch):
    from types import ModuleType, SimpleNamespace
    import sys
    from ouroboros.tools.shell import _claude_code_edit

    gateway = ModuleType("ouroboros.gateways.claude_code")
    gateway.resolve_claude_code_model = lambda: "test-model"
    gateway.DEFAULT_CLAUDE_CODE_MAX_TURNS = 1
    monkeypatch.setitem(sys.modules, "ouroboros.gateways.claude_code", gateway)

    ctx, skill = _ctx(tmp_path)
    env_marker = skill / ".ouroboros_env" / "marker.txt"
    env_marker.parent.mkdir(parents=True)
    env_marker.write_text("before", encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def fake_run_edit(**kwargs):
        marker = skill / ".ouroboros_env" / "marker.txt"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("after", encoding="utf-8")
        return SimpleNamespace(
            success=False,
            error="sdk failed",
            result_text="partial output",
            cost_usd=0.0,
            usage={},
            changed_files=[],
            diff_stat="",
            validation_summary="",
            to_tool_output=lambda: "FAILED",
        )

    gateway.run_edit = fake_run_edit

    result = _claude_code_edit(ctx, "edit", cwd=".")

    assert "CLAUDE_CODE_ERROR" in result
    assert "SKILL_PAYLOAD_CONTROL_RESTORED" in result
    assert env_marker.read_text(encoding="utf-8") == "before"
    assert not list(skill.parent.glob(".ouroboros-control-backup-*"))


def test_claude_code_edit_restores_control_dirs_on_gateway_import_failure(tmp_path, monkeypatch):
    from types import ModuleType
    import sys
    from ouroboros.tools.shell import _claude_code_edit

    gateway = ModuleType("ouroboros.gateways.claude_code")
    gateway.resolve_claude_code_model = lambda: "test-model"
    gateway.DEFAULT_CLAUDE_CODE_MAX_TURNS = 1
    monkeypatch.setitem(sys.modules, "ouroboros.gateways.claude_code", gateway)

    ctx, skill = _ctx(tmp_path)
    env_marker = skill / ".ouroboros_env" / "marker.txt"
    env_marker.parent.mkdir(parents=True)
    env_marker.write_text("before", encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    result = _claude_code_edit(ctx, "edit", cwd=".")

    assert "CLAUDE_CODE_UNAVAILABLE" in result
    assert env_marker.read_text(encoding="utf-8") == "before"
    assert not list(skill.parent.glob(".ouroboros-control-backup-*"))


def test_claude_code_edit_reverts_uppercase_openclaw_sidecar(tmp_path, monkeypatch):
    from types import ModuleType, SimpleNamespace
    import sys
    from ouroboros.tools.shell import _claude_code_edit

    gateway = ModuleType("ouroboros.gateways.claude_code")
    gateway.resolve_claude_code_model = lambda: "test-model"
    gateway.DEFAULT_CLAUDE_CODE_MAX_TURNS = 1
    monkeypatch.setitem(sys.modules, "ouroboros.gateways.claude_code", gateway)

    ctx, skill = _ctx(tmp_path)
    sidecar = skill / "SKILL.openclaw.md"
    sidecar.write_text("before", encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def fake_run_edit(**kwargs):
        sidecar.write_text("after", encoding="utf-8")
        return SimpleNamespace(
            success=True,
            error="",
            result_text="ok",
            cost_usd=0.0,
            usage={},
            changed_files=[],
            diff_stat="",
            validation_summary="",
            to_tool_output=lambda: "OK",
        )

    gateway.run_edit = fake_run_edit

    result = _claude_code_edit(ctx, "edit", cwd=".")

    assert "SKILL_PAYLOAD_CONTROL_BLOCKED" in result
    assert sidecar.read_text(encoding="utf-8") == "before"


def test_claude_code_edit_rejects_non_skill_data_cwd(tmp_path, monkeypatch):
    from ouroboros.tools.shell import _claude_code_edit

    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    drive.mkdir()
    (drive / "settings.json").write_text('{"TOTAL_BUDGET": 10}\n', encoding="utf-8")
    ctx = ToolContext(repo_dir=repo, drive_root=drive)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    result = _claude_code_edit(ctx, "edit settings", cwd=str(drive))

    assert "CLAUDE_CODE_ERROR" in result
    assert "cwd escapes allowed edit roots" in result


def test_repair_data_write_manifest_does_not_create_self_authored_markers(tmp_path, monkeypatch):
    from ouroboros import config as cfg
    ctx, skill = _ctx(tmp_path)
    monkeypatch.setattr(cfg, "DATA_DIR", ctx.drive_root)
    result = _data_write(ctx, "SKILL.md", "---\nname: alpha\ndescription: x\nversion: 0.1\ntype: instruction\n---\n")
    assert "OK:" in result
    assert not (skill / ".self_authored.json").exists()
    assert not (ctx.drive_root / "state" / "skills" / "alpha" / "self_authored.json").exists()


def test_payload_root_must_match_skill_name(tmp_path):
    bad = TaskConstraint(mode="skill_repair", skill_name="alpha", payload_root="skills/external/beta")
    try:
        resolve_payload_path(tmp_path / "data", bad, "plugin.py")
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("mismatched skill_name/payload_root was accepted")


def test_registry_rejects_mismatched_repair_payload_root(tmp_path):
    from ouroboros.tools.registry import ToolRegistry

    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    (drive / "skills" / "external" / "beta").mkdir(parents=True)
    bad_ctx = ToolContext(
        repo_dir=repo,
        drive_root=drive,
        task_constraint=TaskConstraint(mode="skill_repair", skill_name="alpha", payload_root="skills/external/beta"),
    )
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry._ctx = bad_ctx

    result = registry.execute(
        "write_file",
        {
            "root": "skill_payload",
            "bucket": "external",
            "skill_name": "alpha",
            "path": "plugin.py",
            "content": "x",
        },
    )

    assert "HEAL_MODE_BLOCKED" in result or "SKILL_REDIRECT_BLOCKED" in result


def test_light_mode_allows_constrained_str_replace_editor_payload_edit(tmp_path, monkeypatch):
    from ouroboros import config as cfg
    from ouroboros.tools.registry import ToolRegistry

    ctx, skill = _ctx(tmp_path)
    target = skill / "plugin.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry._ctx = ctx
    monkeypatch.setattr(cfg, "get_runtime_mode", lambda: "light")

    result = registry.execute(
        "edit_text",
        {
            "root": "skill_payload",
            "bucket": "external",
            "skill_name": "alpha",
            "path": "plugin.py",
            "old_str": "VALUE = 1",
            "new_str": "VALUE = 2",
        },
    )

    assert "LIGHT_MODE_BLOCKED" not in result
    assert "Replaced" in result
    assert target.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_light_mode_allows_normal_skill_str_replace_without_repair_constraint(tmp_path, monkeypatch):
    from ouroboros import config as cfg
    from ouroboros.tools.registry import ToolRegistry

    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    skill = drive / "skills" / "clawhub" / "alpha"
    skill.mkdir(parents=True)
    target = skill / "plugin.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    monkeypatch.setattr(cfg, "get_runtime_mode", lambda: "light")

    result = registry.execute(
        "edit_text",
        {"path": "skills/clawhub/alpha/plugin.py", "old_str": "VALUE = 1", "new_str": "VALUE = 2"},
    )

    assert "LIGHT_MODE_BLOCKED" not in result
    assert "Replaced" in result
    assert target.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_light_mode_blocks_normal_skill_sidecar_str_replace(tmp_path, monkeypatch):
    from ouroboros import config as cfg
    from ouroboros.tools.registry import ToolRegistry

    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    skill = drive / "skills" / "ouroboroshub" / "alpha"
    skill.mkdir(parents=True)
    sidecar = skill / ".ouroboroshub.json"
    sidecar.write_text('{"version":"1"}\n', encoding="utf-8")
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    monkeypatch.setattr(cfg, "get_runtime_mode", lambda: "light")

    result = registry.execute(
        "edit_text",
        {"path": "skills/ouroboroshub/alpha/.ouroboroshub.json", "old_str": "1", "new_str": "2"},
    )

    assert "Replaced" not in result
    assert "BLOCKED" in result
    assert sidecar.read_text(encoding="utf-8") == '{"version":"1"}\n'


def test_light_mode_blocks_review_excluded_skill_dirs(tmp_path, monkeypatch):
    from ouroboros import config as cfg
    from ouroboros.tools.registry import ToolRegistry

    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    target_dir = drive / "skills" / "external" / "alpha" / "node_modules"
    target_dir.mkdir(parents=True)
    target = target_dir / "dep.js"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    monkeypatch.setattr(cfg, "get_runtime_mode", lambda: "light")

    result = registry.execute(
        "edit_text",
        {"path": "skills/external/alpha/node_modules/dep.js", "old_str": "VALUE = 1", "new_str": "VALUE = 2"},
    )

    assert "LIGHT_MODE_BLOCKED" in result
    assert target.read_text(encoding="utf-8") == "VALUE = 1\n"


def test_data_write_blocks_review_excluded_skill_dirs(tmp_path, monkeypatch):
    from ouroboros import config as cfg
    from ouroboros.tools.core import _data_write

    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    drive.mkdir()
    monkeypatch.setattr(cfg, "DATA_DIR", drive)
    ctx = ToolContext(repo_dir=repo, drive_root=drive)

    result = _data_write(ctx, "skills/external/alpha/__pycache__/evil.py", "VALUE = 2\n")

    assert "DATA_WRITE_BLOCKED" in result


def test_light_mode_allows_skill_payload_write_file(tmp_path, monkeypatch):
    from ouroboros import config as cfg
    from ouroboros.tools.registry import ToolRegistry

    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    skill = drive / "skills" / "external" / "alpha"
    skill.mkdir(parents=True)
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    monkeypatch.setattr(cfg, "get_runtime_mode", lambda: "light")

    result = registry.execute(
        "write_file",
        {
            "root": "skill_payload",
            "bucket": "external",
            "skill_name": "alpha",
            "path": "generated.py",
            "content": "VALUE = 1\n",
        },
    )

    assert "LIGHT_MODE_BLOCKED" not in result
    assert (skill / "generated.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_light_mode_allows_repair_edit_text_with_skill_payload_root(tmp_path, monkeypatch):
    from ouroboros import config as cfg
    from ouroboros.tools.registry import ToolRegistry

    ctx, skill = _ctx(tmp_path)
    target = skill / "plugin.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry._ctx = ctx
    monkeypatch.setattr(cfg, "get_runtime_mode", lambda: "light")

    result = registry.execute(
        "edit_text",
        {
            "root": "skill_payload",
            "bucket": "external",
            "skill_name": "alpha",
            "path": "plugin.py",
            "old_str": "VALUE = 1",
            "new_str": "VALUE = 2",
        },
    )

    assert "LIGHT_MODE_BLOCKED" not in result
    assert "Replaced" in result
    assert target.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_light_mode_still_blocks_repo_str_replace_without_repair_constraint(tmp_path, monkeypatch):
    from ouroboros import config as cfg
    from ouroboros.tools.registry import ToolRegistry

    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    drive.mkdir()
    (repo / "README.md").write_text("VALUE = 1\n", encoding="utf-8")
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    monkeypatch.setattr(cfg, "get_runtime_mode", lambda: "light")

    result = registry.execute(
        "edit_text",
        {"path": "README.md", "old_str": "VALUE = 1", "new_str": "VALUE = 2"},
    )

    assert "LIGHT_MODE_BLOCKED" in result
