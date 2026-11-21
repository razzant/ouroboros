
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
    (skill / "SKILL.md").write_text("# alpha\n", encoding="utf-8")
    return ToolContext(repo_dir=repo, drive_root=drive, task_constraint=TaskConstraint(mode="skill_repair", skill_name="alpha", payload_root="skills/external/alpha", allow_enable=False)), skill


def _admit_repair(ctx, skill):
    """Mint the X3 repair admission these direct-constraint tests bypass.

    Production repair tasks are admitted through the promote seam
    (supervisor/workers.py), which records the binding fail-closed before the
    task exists; tests that build the ``skill_repair`` constraint directly must
    mint the same binding or every payload write is a typed
    SKILL_REPAIR_STALE refusal. Call AFTER all direct payload setup writes —
    the admission pins the payload state the repair starts from.
    """
    from ouroboros.skill_loader import compute_content_hash
    from ouroboros.skill_repair_admission import record_repair_admission

    ctx.task_id = getattr(ctx, "task_id", "") or "repair-constraint-test"
    record_repair_admission(
        ctx.drive_root, "alpha", task_id=ctx.task_id,
        base_content_hash=compute_content_hash(skill),
    )


def test_payload_relative_resolver_accepts_short_paths(tmp_path):
    ctx, skill = _ctx(tmp_path)
    assert resolve_payload_path(ctx.drive_root, ctx.task_constraint, "plugin.py") == skill / "plugin.py"
    assert resolve_payload_path(ctx.drive_root, ctx.task_constraint, "skills/external/alpha/plugin.py") == skill / "plugin.py"


def test_str_replace_editor_uses_payload_relative_path(tmp_path):
    ctx, skill = _ctx(tmp_path)
    target = skill / "plugin.py"
    target.write_text("hello = 1\n", encoding="utf-8")
    _admit_repair(ctx, skill)
    result = _str_replace_editor(ctx, "plugin.py", "hello = 1", "hello = 2")
    assert "Replaced" in result
    assert target.read_text(encoding="utf-8") == "hello = 2\n"
    assert not (ctx.repo_dir / "plugin.py").exists()


def test_data_write_uses_payload_relative_path(tmp_path):
    ctx, skill = _ctx(tmp_path)
    _admit_repair(ctx, skill)
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


def test_skill_development_preserves_code_search(tmp_path):
    from ouroboros.tools.registry import ToolRegistry
    ctx, _skill = _ctx(tmp_path)
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry._ctx = ctx
    (ctx.repo_dir / "probe.py").write_text("class RepairSearchProbe:\n    pass\n")
    result = registry.execute("search_code", {"query": "RepairSearchProbe"})
    assert "probe.py" in result and "RepairSearchProbe" in result, result


def test_repair_data_write_manifest_does_not_create_self_authored_markers(tmp_path, monkeypatch):
    from ouroboros import config as cfg
    ctx, skill = _ctx(tmp_path)
    monkeypatch.setattr(cfg, "DATA_DIR", ctx.drive_root)
    _admit_repair(ctx, skill)
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

    assert "SKILL_REDIRECT_BLOCKED" in result


def test_cross_skill_writer_uses_payload_refusal_and_keeps_selected_write(tmp_path):
    from ouroboros.tools.registry import ToolRegistry

    ctx, skill = _ctx(tmp_path)
    other = ctx.drive_root / "skills" / "external" / "beta"
    other.mkdir()
    (other / "SKILL.md").write_text("# beta\n", encoding="utf-8")
    _admit_repair(ctx, skill)
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry._ctx = ctx
    blocked = registry.execute_result("write_file", {
        "bucket": "external", "skill_name": "beta", "path": "notes.txt", "content": "wrong target",
    })
    assert blocked.status == "blocked" and blocked.code == "SKILL_PAYLOAD_BLOCKED"
    assert "SKILL_REDIRECT_BLOCKED" in blocked.text
    assert not (other / "notes.txt").exists()
    allowed = registry.execute("write_file", {"root": "skill_payload", "path": "notes.txt", "content": "selected target"})
    assert (skill / "notes.txt").read_text() == "selected target", allowed


def test_light_mode_allows_constrained_str_replace_editor_payload_edit(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolRegistry

    ctx, skill = _ctx(tmp_path)
    target = skill / "plugin.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    _admit_repair(ctx, skill)
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry._ctx = ctx
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")

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
    from ouroboros.tools.registry import ToolRegistry

    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    skill = drive / "skills" / "clawhub" / "alpha"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# alpha\n", encoding="utf-8")
    target = skill / "plugin.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")

    result = registry.execute(
        "edit_text",
        {"path": "skills/clawhub/alpha/plugin.py", "old_str": "VALUE = 1", "new_str": "VALUE = 2"},
    )

    assert "LIGHT_MODE_BLOCKED" not in result
    assert "Replaced" in result
    assert target.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_light_mode_blocks_normal_skill_sidecar_str_replace(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolRegistry

    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    skill = drive / "skills" / "ouroboroshub" / "alpha"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# alpha\n", encoding="utf-8")
    sidecar = skill / ".ouroboroshub.json"
    sidecar.write_text('{"version":"1"}\n', encoding="utf-8")
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")

    result = registry.execute(
        "edit_text",
        {"path": "skills/ouroboroshub/alpha/.ouroboroshub.json", "old_str": "1", "new_str": "2"},
    )

    assert "Replaced" not in result
    assert "BLOCKED" in result
    assert sidecar.read_text(encoding="utf-8") == '{"version":"1"}\n'


def test_review_excluded_skill_dirs_stay_blocked_in_light_mode(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolRegistry

    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    target_dir = drive / "skills" / "external" / "alpha" / "node_modules"
    target_dir.mkdir(parents=True)
    (target_dir.parent / "SKILL.md").write_text("# alpha\n", encoding="utf-8")
    target = target_dir / "dep.js"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")

    result = registry.execute(
        "edit_text",
        {"path": "skills/external/alpha/node_modules/dep.js", "old_str": "VALUE = 1", "new_str": "VALUE = 2"},
    )

    assert "STR_REPLACE_BLOCKED" in result
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
    from ouroboros.tools.registry import ToolRegistry

    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    skill = drive / "skills" / "external" / "alpha"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# alpha\n", encoding="utf-8")
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")

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
    from ouroboros.tools.registry import ToolRegistry

    ctx, skill = _ctx(tmp_path)
    target = skill / "plugin.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    _admit_repair(ctx, skill)
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry._ctx = ctx
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")

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
    from ouroboros.tools.registry import ToolRegistry

    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    drive.mkdir()
    (repo / "README.md").write_text("VALUE = 1\n", encoding="utf-8")
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")

    result = registry.execute(
        "edit_text",
        {"path": "README.md", "old_str": "VALUE = 1", "new_str": "VALUE = 2"},
    )

    assert "LIGHT_MODE_BLOCKED" in result
