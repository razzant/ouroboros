
import pytest

from ouroboros.contracts.task_constraint import TaskConstraint, resolve_payload_path
from ouroboros.tools.core import _data_write
from ouroboros.tools.git import _str_replace_editor
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.tool_result import ToolResult


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
    from ouroboros.tools.core_file_tools import _data_list, _data_read
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
    from ouroboros.tools.core_file_tools import _data_read
    ctx, _skill = _ctx(tmp_path)
    assert "DATA_READ_BLOCKED" in _data_read(ctx, "skills/external/beta/plugin.py")


def test_repair_mode_blocks_code_search(tmp_path):
    from ouroboros.tools.registry import ToolRegistry
    ctx, _skill = _ctx(tmp_path)
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry._ctx = ctx
    result = registry.execute("search_code", {"query": "ToolRegistry"})
    assert "HEAL_MODE_BLOCKED" in result


def test_registry_heal_guard_owner_facades_preserve_identity():
    import inspect

    from ouroboros.tools import registry, registry_guards

    assert registry._HEAL_MODE_ALLOWED_TOOLS is registry_guards._HEAL_MODE_ALLOWED_TOOLS
    assert registry._task_constraint_path_allowed is registry_guards._task_constraint_path_allowed
    assert registry._heal_protected_payload_sidecar is registry_guards._heal_protected_payload_sidecar
    assert (
        inspect.signature(registry._task_constraint_path_allowed)
        .parameters["constraint"]
        .annotation
        == "Optional[TaskConstraint]"
    )
    assert not hasattr(registry.ToolRegistry, "_heal_mode_block")


def test_heal_guard_native_denials_preserve_exact_text(tmp_path):
    from ouroboros.tools.registry_guards import _heal_mode_guard_result

    ctx, _skill = _ctx(tmp_path)
    redirect = (
        "⚠️ SKILL_REDIRECT_BLOCKED: active skill_repair "
        "task is scoped to the selected skill payload."
    )
    payload_access = (
        "⚠️ HEAL_MODE_BLOCKED: Repair payload access is limited "
        "to the selected skill payload."
    )
    data_access = (
        "⚠️ HEAL_MODE_BLOCKED: Repair data access is limited "
        "to the selected skill payload under data/skills/external "
        "data/skills/clawhub, or data/skills/ouroboroshub."
    )
    sidecar = (
        "⚠️ HEAL_MODE_BLOCKED: Repair may not edit marketplace "
        "or official provenance sidecars (.clawhub.json, "
        ".ouroboroshub.json, SKILL.openclaw.md, .seed-origin). "
        "Edit the user-authored payload files instead."
    )
    listing = (
        "⚠️ HEAL_MODE_BLOCKED: Repair data listing is limited "
        "to the selected skill payload under data/skills/external "
        "data/skills/clawhub, or data/skills/ouroboroshub."
    )
    general = (
        "⚠️ HEAL_MODE_BLOCKED: Repair tasks may inspect/edit skill "
        "payloads and run skill_review only. Shell, browser automation, "
        "repo mutation, skill execution, extension tools, MCP tools, "
        "delegation, and enable/disable flows are unavailable. Use "
        "the Skills UI after a fresh executable review."
    )
    cases = [
        (
            "write_file",
            {"root": "skill_payload", "bucket": "clawhub", "skill_name": "alpha"},
            None,
            False,
            redirect,
        ),
        (
            "write_file",
            {
                "root": "skill_payload",
                "bucket": "clawhub",
                "skill_name": "alpha",
                "path": ".clawhub.json",
            },
            None,
            False,
            redirect,
        ),
        (
            "read_file",
            {"root": "skill_payload", "bucket": "clawhub", "skill_name": "alpha"},
            None,
            False,
            payload_access,
        ),
        (
            "read_file",
            {
                "root": "skill_payload",
                "bucket": "external",
                "skill_name": "alpha",
                "path": "skills/external/beta/plugin.py",
            },
            None,
            False,
            data_access,
        ),
        (
            "write_file",
            {
                "root": "skill_payload",
                "bucket": "external",
                "skill_name": "alpha",
                "files": [
                    {"path": "plugin.py"},
                    {"path": "skills/external/beta/plugin.py"},
                ],
            },
            None,
            False,
            data_access,
        ),
        (
            "write_file",
            {
                "root": "skill_payload",
                "bucket": "external",
                "skill_name": "alpha",
                "path": "skills/external/beta/.clawhub.json",
            },
            None,
            False,
            data_access,
        ),
        (
            "write_file",
            {
                "root": "skill_payload",
                "bucket": "external",
                "skill_name": "alpha",
                "path": ".clawhub.json",
            },
            None,
            False,
            sidecar,
        ),
        (
            "write_file",
            {
                "root": "skill_payload",
                "bucket": "external",
                "skill_name": "alpha",
                "files": [
                    {"path": ".clawhub.json"},
                    {"path": "skills/external/beta/plugin.py"},
                ],
            },
            None,
            False,
            sidecar,
        ),
        (
            "list_files",
            {
                "root": "skill_payload",
                "bucket": "external",
                "skill_name": "alpha",
                "path": "skills/external/beta",
            },
            None,
            False,
            listing,
        ),
        (
            "edit_text",
            {"path": "skills/external/beta/plugin.py"},
            None,
            False,
            "⚠️ HEAL_MODE_BLOCKED: Repair edit_text is limited to the selected skill payload.",
        ),
        (
            "edit_text",
            {"path": ".ouroboroshub.json"},
            None,
            False,
            sidecar,
        ),
        (
            "skill_review",
            {"skill": "beta"},
            None,
            False,
            "⚠️ HEAL_MODE_BLOCKED: Repair may only review the selected skill.",
        ),
        (
            "skill_preflight",
            {"skill": "beta"},
            None,
            False,
            "⚠️ HEAL_MODE_BLOCKED: Repair may only preflight the selected skill.",
        ),
        ("search_code", {}, None, False, general),
        ("ext_demo__ping", {}, object(), False, general),
        ("mcp_demo__ping", {}, None, True, general),
    ]

    for name, args, ext_tool, is_mcp, text in cases:
        assert _heal_mode_guard_result(
            ctx,
            name,
            args,
            ctx.task_constraint,
            ext_tool,
            is_mcp,
        ) == ToolResult(status="blocked", code="HEAL_MODE_BLOCKED", text=text)


def test_heal_guard_allow_paths_return_none(tmp_path):
    from ouroboros.tools.registry_guards import _heal_mode_guard_result

    ctx, _skill = _ctx(tmp_path)
    allowed = [
        ("read_file", {"root": "skill_payload", "path": "plugin.py"}),
        ("read_file", {"root": "skill_payload"}),
        ("list_files", {"root": "skill_payload", "path": "."}),
        (
            "write_file",
            {
                "root": "skill_payload",
                "bucket": "external",
                "skill_name": "alpha",
                "files": [{"path": "plugin.py"}, {"path": "nested/new.py"}],
            },
        ),
        ("edit_text", {"path": "plugin.py"}),
        ("list_skills", {}),
        ("skill_review", {"skill": "alpha"}),
        ("skill_preflight", {"skill": "alpha"}),
    ]

    for name, args in allowed:
        assert _heal_mode_guard_result(
            ctx,
            name,
            args,
            ctx.task_constraint,
            None,
            False,
        ) is None


def test_registry_native_heal_guard_preserves_order_and_zero_dispatch(tmp_path, monkeypatch):
    from ouroboros import mcp_client, safety
    from ouroboros.tools import extension_dispatch
    from ouroboros.tools import tool_resolution as resolution_module
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.tool_result import LegacyTextResultAdapter

    ctx, _skill = _ctx(tmp_path)
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry._ctx = ctx
    calls = []

    def forbidden(label):
        def fail(*_args, **_kwargs):
            calls.append(label)
            pytest.fail(f"heal denial reached {label}")
        return fail

    registry.override_handler("search_code", forbidden("builtin handler"))
    registry.override_handler("write_file", forbidden("write handler"))
    monkeypatch.setattr(safety, "check_safety", forbidden("safety"))
    monkeypatch.setattr(
        LegacyTextResultAdapter,
        "from_text",
        classmethod(forbidden("legacy adapter")),
    )
    monkeypatch.setattr(
        extension_dispatch,
        "_dispatch_extension_tool_result",
        forbidden("extension dispatch"),
    )
    monkeypatch.setattr(
        extension_dispatch,
        "_dispatch_mcp_tool_result",
        forbidden("MCP dispatch"),
    )
    monkeypatch.setattr(
        resolution_module,
        "build_resolved_resource_binding",
        forbidden("target binding"),
    )

    builtin = registry.execute_result("search_code", {"query": "ToolRegistry"})
    assert builtin.status == "blocked"
    assert builtin.code == "HEAL_MODE_BLOCKED"
    assert registry.execute("search_code", {"query": "ToolRegistry"}) == builtin.text

    monkeypatch.setattr(
        extension_dispatch,
        "_extension_dispatch_candidate",
        lambda _ctx, _name: (object(), False),
    )
    extension = registry.execute_result("ext_demo__ping", {})
    assert extension == ToolResult(
        status="blocked",
        code="HEAL_MODE_BLOCKED",
        text=(
            "⚠️ HEAL_MODE_BLOCKED: Repair tasks may inspect/edit skill payloads and run "
            "skill_review only. Shell, browser automation, repo mutation, skill execution, "
            "extension tools, MCP tools, delegation, and enable/disable flows are unavailable. "
            "Use the Skills UI after a fresh executable review."
        ),
    )

    monkeypatch.setattr(
        extension_dispatch,
        "_extension_dispatch_candidate",
        lambda _ctx, _name: (None, False),
    )
    discovery = []
    monkeypatch.setattr(
        mcp_client,
        "ensure_configured_from_settings",
        lambda *, refresh=False: discovery.append(refresh),
    )
    monkeypatch.setattr(mcp_client, "is_mcp_tool_name", lambda _name: True)
    mcp = registry.execute_result("mcp_demo__ping", {})
    assert mcp.status == "blocked"
    assert mcp.code == "HEAL_MODE_BLOCKED"
    assert discovery == [False]
    assert calls == []

    ctx.task_metadata = {"task_contract": {"disabled_tools": ["search_code"]}}
    earlier = registry.execute_result("search_code", {"query": "ToolRegistry"})
    assert earlier.code == "RESOURCE_CONSTRAINT_BLOCKED"
    ctx.task_metadata = {}
    ctx.is_ephemeral_turn = True
    earlier = registry.execute_result("run_command", {"command": "true"})
    assert earlier.code == "ACCESS_BLOCKED"
    assert earlier.text.startswith("⚠️ EPHEMERAL_TURN_RESTRICTED")
    ctx.is_ephemeral_turn = False

    public_arg_error = registry._execute_legacy_text("search_code", {"not_an_argument": True})
    assert isinstance(public_arg_error, str)
    assert public_arg_error.startswith("⚠️ TOOL_ARG_ERROR (search_code)")

    root_redirect_args = {
        "root": "user_files",
        "path": str(ctx.repo_dir / "module.py"),
        "old_str": "before",
        "new_str": "after",
    }
    root_redirect = registry._execute_legacy_text("edit_text", dict(root_redirect_args))
    root_redirect_text = (
        "⚠️ ROOT_REQUIRED_ACTIVE_WORKSPACE: absolute path "
        f"{root_redirect_args['path']!r} is under the active workspace, but root='user_files' does not "
        "write there. Retry the same call with root='active_workspace' (the same path is accepted)."
    )
    assert root_redirect == ToolResult(
        status="blocked",
        code="ROOT_REQUIRED_ACTIVE_WORKSPACE",
        text=root_redirect_text,
        meta={"required_root": "active_workspace"},
    )
    assert registry.execute("edit_text", dict(root_redirect_args)) == root_redirect_text

    payload_arg_error = registry.execute_result(
        "write_file",
        {"bucket": "external", "path": "SKILL.md"},
    )
    # T1 (owner batch #4 answer 1): the skill-payload selector refusal is a policy
    # denial, which its own first line always said; the generic argument-error code
    # contradicted it and would promote the refusal to an execution failure.
    assert payload_arg_error == ToolResult(
        status="blocked",
        code="SKILL_PAYLOAD_BLOCKED",
        text=(
            "⚠️ SKILL_PAYLOAD_ARG_ERROR: bucket and skill_name must be supplied together; "
            "bucket must be one of external/clawhub/ouroboroshub (native excluded); "
            "skill_name must sanitize to a non-empty slug."
        ),
    )
    assert registry.execute(
        "write_file",
        {"bucket": "external", "path": "SKILL.md"},
    ) == payload_arg_error.text

    short_redirect = registry.execute_result(
        "write_file",
        {"bucket": "external", "skill_name": "beta", "path": "SKILL.md"},
    )
    assert short_redirect == ToolResult(
        status="blocked",
        code="HEAL_MODE_BLOCKED",
        text=(
            "⚠️ SKILL_REDIRECT_BLOCKED: a skill_repair task is active for 'alpha'; "
            "cannot use bucket+skill_name args to redirect this call to 'beta'. "
            "Drop the bucket/skill_name args, or finish/cancel the active repair task first."
        ),
    )
    assert registry.execute(
        "write_file",
        {"bucket": "external", "skill_name": "beta", "path": "SKILL.md"},
    ) == short_redirect.text
    assert calls == []


def test_loop_preserves_legacy_heal_projection_with_native_code(tmp_path):
    import json

    from ouroboros.loop_tool_execution import _execute_single_tool
    from ouroboros.tools.registry import ToolRegistry

    logs = tmp_path / "logs"
    logs.mkdir()
    ctx, _skill = _ctx(tmp_path)
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry._ctx = ctx
    short_redirect = (
        "⚠️ SKILL_REDIRECT_BLOCKED: a skill_repair task is active for 'alpha'; "
        "cannot use bucket+skill_name args to redirect this call to 'beta'. "
        "Drop the bucket/skill_name args, or finish/cancel the active repair task first."
    )
    payload_arg_error = (
        "⚠️ SKILL_PAYLOAD_ARG_ERROR: bucket and skill_name must be supplied together; "
        "bucket must be one of external/clawhub/ouroboroshub (native excluded); "
        "skill_name must sanitize to a non-empty slug."
    )
    cases = [
        (
            "skill_review",
            {"skill": "beta"},
            "⚠️ HEAL_MODE_BLOCKED: Repair may only review the selected skill.",
            "heal_mode_blocked",
            "blocked",
            "HEAL_MODE_BLOCKED",
        ),
        (
            "write_file",
            {
                "root": "skill_payload",
                "bucket": "clawhub",
                "skill_name": "alpha",
            },
            "⚠️ SKILL_REDIRECT_BLOCKED: active skill_repair task is scoped to the selected skill payload.",
            # T1 §A.18: the publisher's code wins over its own first line; both
            # statuses sit in the policy-denial partition, so the report is unchanged.
            "heal_mode_blocked",
            "blocked",
            "HEAL_MODE_BLOCKED",
        ),
        (
            "write_file",
            {
                "bucket": "external",
                "skill_name": "beta",
                "path": "SKILL.md",
            },
            short_redirect,
            "heal_mode_blocked",
            "blocked",
            "HEAL_MODE_BLOCKED",
        ),
        (
            "write_file",
            {"bucket": "external", "path": "SKILL.md"},
            payload_arg_error,
            "skill_payload_blocked",
            "blocked",
            "SKILL_PAYLOAD_BLOCKED",
        ),
    ]
    for index, (name, args, text, legacy_status, typed_status, typed_code) in enumerate(cases):
        row = _execute_single_tool(
            registry,
            {
                "id": f"call-{index}",
                "function": {"name": name, "arguments": json.dumps(args)},
            },
            logs,
        )
        assert row["result"] == text
        assert row["is_error"] is True
        assert row["result_meta"]["status"] == legacy_status
        assert row["result_meta"]["tool_result_status"] == typed_status
        assert row["result_meta"]["tool_result_code"] == typed_code
        assert row["result_meta"]["tool_result_meta"] == {}


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
    (skill / "SKILL.md").write_text("# alpha\n", encoding="utf-8")
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
    (skill / "SKILL.md").write_text("# alpha\n", encoding="utf-8")
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


def test_review_excluded_skill_dirs_stay_blocked_in_light_mode(tmp_path, monkeypatch):
    from ouroboros import config as cfg
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
    monkeypatch.setattr(cfg, "get_runtime_mode", lambda: "light")

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
    from ouroboros import config as cfg
    from ouroboros.tools.registry import ToolRegistry

    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    skill = drive / "skills" / "external" / "alpha"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# alpha\n", encoding="utf-8")
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
