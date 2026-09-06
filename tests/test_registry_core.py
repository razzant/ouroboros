"""Focused owner/facade contracts for the ToolRegistry extraction."""

from __future__ import annotations

import inspect

import pytest


def test_registry_core_extraction_preserves_only_proven_facades():
    """Execution moves once; the compatibility module exports only proven ABI."""
    import ouroboros.tools as tools_package
    from ouroboros.tools import (
        registry,
        registry_core,
        registry_guards,
        extension_dispatch,
        tool_catalog,
        tool_context,
        tool_resolution,
        tool_result,
    )

    resolution_names = {
        "_GENERIC_VCS_TARGET_TOOLS",
        "_PATH_NORMALIZED_TOOLS",
        "_PROCESS_TARGET_TOOLS",
        "_SKILL_LIFECYCLE_TARGET_TOOLS",
        "_TARGET_BINDING_OPERATIONS",
        "_VERIFY_RUN_KINDS",
        "_binding_items",
        "_binding_set_is_light_restricted",
        "_binding_set_targets_system_repo",
        "_build_builtin_target_binding",
        "_coerce_real_path",
        "_normalize_dispatch_path_args",
        "_target_binding_operation",
        "active_repo_dir_for",
        "system_repo_dir_for",
    }
    guard_names = {
        "_EPHEMERAL_ALLOWED_TOOLS",
        "_GITHUB_TOKEN_TOOLS",
        "_HEAL_MODE_ALLOWED_TOOLS",
        "_WEB_TOOLS",
        "_authorized_managed_update_resolver",
        "_builtin_tool_availability",
        "_disabled_tools",
        "_heal_protected_payload_sidecar",
        "_managed_update_code_tool_block",
        "_resource_allowed",
        "_task_constraint_path_allowed",
    }
    private_resolution_names = {
        "_DispatchPathNormalization",
        "_ROOT_ARG_REPO_WRITE_TOOLS",
        "_TOP_LEVEL_PATH_WRITE_TOOLS",
        "_TOOL_ARG_ALIASES",
        "_IGNORE_ROOT_ARG_TOOLS",
        "_binding_error_text",
        "_entry_has_public_param_schema",
        "_entry_public_params",
        "_format_tool_arg_error",
        "_handler_public_params",
        "_light_binding_failure_redirect",
        "_light_binding_failure_result",
        "_normalize_dispatch_path_args_result",
        "_normalize_tool_call_args",
        "_payload_write_paths",
        "_prepare_public_builtin_args",
        "_resolve_python_predispatch",
    }
    private_guard_names = {
        "_payload_dispatch_constraint",
        "_stray_skill_payload_failsoft",
    }
    private_extension_names = {
        "_dispatch_extension_tool_result",
        "_dispatch_mcp_tool_result",
        "_extension_dispatch_candidate",
    }
    proven = {
        "BrowserState",
        "ToolContext",
        "ToolEntry",
        "ToolRegistry",
        "_compose_execute_result",
        *resolution_names,
        *guard_names,
    }

    # 31 since #447: `_binding_state_drive_root` went with the deleted post-hoc
    # owner-state restore organ, so the facade no longer proves it.
    assert len(proven) == 31
    # Tip adaptation of the reference pin: this tree's facade deliberately keeps
    # the broad HISTORICAL import surface (importers and monkeypatch targets are
    # not migrated in this window), so the reference's exact-32-name equality
    # over vars(registry) does not transfer. The durable fact it protected —
    # execution moved ONCE and the facade authors nothing — is pinned
    # structurally instead: the module may DEFINE only the disclosed read-carve
    # helper; everything else must be a re-export.
    import ast
    import pathlib

    facade_ast = ast.parse(
        pathlib.Path(inspect.getsourcefile(registry)).read_text(encoding="utf-8")
    )
    facade_defined = {
        node.name
        for node in facade_ast.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    assert facade_defined == {"_owner_control_mention_blocks"}
    assert all(hasattr(registry, name) for name in proven)
    assert registry.ToolRegistry is registry_core.ToolRegistry is tools_package.ToolRegistry
    assert registry.ToolContext is tool_context.ToolContext is tools_package.ToolContext
    assert registry.BrowserState is tool_context.BrowserState
    assert registry.ToolEntry is tool_catalog.ToolEntry is tools_package.ToolEntry
    assert registry._compose_execute_result is tool_result._compose_execute_result
    assert all(getattr(registry, name) is getattr(tool_resolution, name) for name in resolution_names)
    assert all(getattr(registry, name) is getattr(registry_guards, name) for name in guard_names)
    assert all(hasattr(tool_resolution, name) for name in private_resolution_names)
    assert all(hasattr(registry_guards, name) for name in private_guard_names)
    assert all(hasattr(extension_dispatch, name) for name in private_extension_names)
    retired_private = private_resolution_names | private_guard_names | private_extension_names
    assert all(not hasattr(registry_core, name) for name in retired_private)
    # (The reference also asserted the privates' absence from the facade; on
    # this tree the facade re-exports them for historical importers — see the
    # adaptation note above — so only the owner-module homing is pinned.)
    assert not hasattr(registry.ToolRegistry, "_dispatch_extension_tool")
    assert not hasattr(registry.ToolRegistry, "_dispatch_mcp_tool")
    assert not hasattr(registry.ToolRegistry, "_resolve_python_predispatch")
    assert registry.ToolRegistry.__module__ == "ouroboros.tools.registry_core"
    assert str(inspect.signature(registry.ToolRegistry.execute)) == (
        "(self, name: 'str', args: 'Dict[str, Any]') -> 'str'"
    )
    assert str(inspect.signature(registry.ToolRegistry.execute_result)) == (
        "(self, name: 'str', args: 'Dict[str, Any]') -> 'ToolResult'"
    )
    assert not hasattr(registry_core, "get_tools")
    assert not hasattr(registry_core, "_HEAL_PROTECTED_PAYLOAD_FILENAMES")


def test_registry_core_uses_canonical_managed_update_resolver(tmp_path, monkeypatch):
    import ouroboros.config as config
    import ouroboros.safety as safety
    from ouroboros.tools import registry, registry_guards

    repo = tmp_path / "repo"
    drive = tmp_path / "drive"
    repo.mkdir()
    drive.mkdir()
    tools = registry.ToolRegistry(repo_dir=repo, drive_root=drive)

    def handler(_ctx, *, _resolved_binding=None, **_kwargs):
        assert _resolved_binding is not None
        return "OK"

    tools.override_handler("write_file", handler)

    monkeypatch.setattr(config, "get_runtime_mode", lambda: "light")
    monkeypatch.setattr(safety, "check_safety", lambda *_args, **_kwargs: (True, ""))
    monkeypatch.setattr(
        registry_guards,
        "_authorized_managed_update_resolver",
        lambda _ctx: True,
    )

    def facade_call_is_not_execution_authority(_ctx):
        raise AssertionError("registry facade was consulted as execution authority")

    monkeypatch.setattr(
        registry,
        "_authorized_managed_update_resolver",
        facade_call_is_not_execution_authority,
    )

    result = tools.execute_result(
        "write_file",
        {"root": "system_repo", "path": "BIBLE.md", "content": "unchanged"},
    )

    assert result.status == "ok"
    assert result.text == "OK"


def test_active_workspace_root_redirect_is_native_with_legacy_loop_projection(tmp_path, monkeypatch):
    import json

    import ouroboros.safety as safety
    from ouroboros.loop_tool_execution import _execute_single_tool
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.tool_result import LegacyTextResultAdapter, ToolResult

    repo = tmp_path / "repo"
    drive = tmp_path / "drive"
    logs = drive / "logs"
    repo.mkdir()
    logs.mkdir(parents=True)
    tools = ToolRegistry(repo_dir=repo, drive_root=drive)
    calls = []

    def forbidden(label):
        def fail(*_args, **_kwargs):
            calls.append(label)
            raise AssertionError(f"root redirect reached {label}")

        return fail

    tools.override_handler("write_file", forbidden("handler"))
    monkeypatch.setattr(safety, "check_safety", forbidden("safety"))
    # Tip adaptation: tool_resolution reads the binding builder through the
    # call-time facade handle _registry(), so the facade is the patch point.
    from ouroboros.tools import registry as registry_facade

    monkeypatch.setattr(
        registry_facade,
        "build_resolved_resource_binding",
        forbidden("binding"),
    )
    monkeypatch.setattr(
        LegacyTextResultAdapter,
        "from_text",
        forbidden("legacy adapter"),
    )

    target = str(repo / "result.txt")
    args = {"root": "user_files", "path": target, "content": "result\n"}
    expected_text = (
        "⚠️ ROOT_REQUIRED_ACTIVE_WORKSPACE: absolute path "
        f"{target!r} is under the active workspace, but root='user_files' does not "
        "write there. Retry the same call with root='active_workspace' (the same path is accepted)."
    )
    expected = ToolResult(
        status="blocked",
        code="ROOT_REQUIRED_ACTIVE_WORKSPACE",
        text=expected_text,
        meta={"required_root": "active_workspace"},
    )

    result = tools.execute_result("write_file", dict(args))
    assert result == expected
    assert tools.execute("write_file", dict(args)) == expected_text
    assert not (repo / "result.txt").exists()

    row = _execute_single_tool(
        tools,
        {
            "id": "root-redirect",
            "function": {"name": "write_file", "arguments": json.dumps(args)},
        },
        logs,
    )
    assert row["tool_result"] == expected
    assert row["result"] == expected_text
    assert row["is_error"] is True
    assert row["result_meta"]["status"] == "root_required_active_workspace"
    assert row["result_meta"]["tool_result_status"] == "blocked"
    assert row["result_meta"]["tool_result_code"] == "ROOT_REQUIRED_ACTIVE_WORKSPACE"
    assert row["result_meta"]["tool_result_meta"] == {"required_root": "active_workspace"}
    assert calls == []


def test_registry_uses_typed_required_root_not_note_or_tool_name(tmp_path, monkeypatch):
    import ouroboros.safety as safety
    from ouroboros.tools import tool_resolution
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.tool_result import ToolResult

    repo = tmp_path / "repo"
    drive = tmp_path / "drive"
    repo.mkdir()
    drive.mkdir()
    tools = ToolRegistry(repo_dir=repo, drive_root=drive)
    calls = []

    def handler(_ctx, *, _resolved_binding=None, **_kwargs):
        calls.append(_resolved_binding)
        return "OK"

    tools.override_handler("write_file", handler)
    tools.override_handler("read_file", handler)
    monkeypatch.setattr(safety, "check_safety", lambda *_args, **_kwargs: (True, ""))

    normalizations = []

    def normalize(_ctx, name, _args):
        normalizations.append(name)
        if name == "write_file":
            return tool_resolution._DispatchPathNormalization(
                text="⚠️ AUTO_ROUTED_TO_ACTIVE_WORKSPACE: benign additive note",
            )
        return tool_resolution._DispatchPathNormalization(
            text="typed required-root fact",
            required_root="active_workspace",
        )

    monkeypatch.setattr(
        tool_resolution,
        "_normalize_dispatch_path_args_result",
        normalize,
    )

    benign = tools.execute_result(
        "write_file",
        {"root": "active_workspace", "path": "note.txt", "content": "unchanged"},
    )
    assert benign == ToolResult(
        status="ok",
        code="OK",
        text="OK\n\n⚠️ AUTO_ROUTED_TO_ACTIVE_WORKSPACE: benign additive note",
        meta={"route_note": True},
    )
    assert len(calls) == 1

    required = tools.execute_result(
        "read_file",
        {"root": "active_workspace", "path": "note.txt"},
    )
    assert required == ToolResult(
        status="blocked",
        code="ROOT_REQUIRED_ACTIVE_WORKSPACE",
        text="typed required-root fact",
        meta={"required_root": "active_workspace"},
    )
    assert len(calls) == 1
    assert normalizations == ["write_file", "read_file"]


@pytest.mark.parametrize(
    ("scenario", "tool_name", "args", "expected_code", "expected_text", "legacy_status"),
    (
        (
            "workspace_metadata",
            "read_file",
            {"path": "README.md"},
            "WORKSPACE_BLOCKED",
            "⚠️ WORKSPACE_MODE_BLOCKED: invalid external workspace metadata: "
            "fixture workspace overlap. Workspace tasks must not overlap the "
            "Ouroboros repo, runtime data, or control plane.",
            "workspace_blocked",
        ),
        (
            "acting_repo_write",
            "write_file",
            {"path": "result.txt", "content": "result\n"},
            "ACCESS_BLOCKED",
            "⚠️ ACTING_NO_WORKSPACE_BLOCKED: this acting subagent has no resolved isolated "
            "workspace; write only to root=task_drive, root=artifact_store, or root=user_files. "
            "active_workspace/system_repo map to the live Ouroboros repo and are blocked.",
            "blocked",
        ),
        (
            "acting_process",
            "run_command",
            {"cmd": ["true"]},
            "ACCESS_BLOCKED",
            "⚠️ ACTING_NO_WORKSPACE_BLOCKED: shell/coding/service/integration tools need an "
            "isolated workspace (their default target is the live repo). Schedule a self_worktree "
            "/ external_workspace child for that work.",
            "blocked",
        ),
        (
            "light_repo_mutation",
            "write_file",
            {"path": "README.md", "content": "changed\n"},
            "LIGHT_MODE_BLOCKED",
            "⚠️ LIGHT_MODE_BLOCKED: runtime_mode=light blocks Ouroboros "
            "self-repo/control-plane mutation via 'write_file'. For user-visible "
            "deliverables use root=user_files (for example Desktop/file.html), "
            "root=artifact_store for the canonical task artifact, or root=task_drive "
            "for scratch. Skill payload edits remain allowed only through "
            "root=skill_payload with bucket and skill_name "
            "(data/skills/<bucket>/<skill>/) or skill_repair constraints. "
            "Switch to advanced/pro only for reviewed Ouroboros self-modification.",
            "light_mode_blocked",
        ),
        (
            "light_service",
            "start_service",
            {"name": "fixture", "cmd": ["sleep", "1"]},
            "LIGHT_MODE_BLOCKED",
            "⚠️ LIGHT_MODE_BLOCKED: runtime_mode=light refuses start_service against the "
            "Ouroboros repository because long-running services can mutate after initial tool "
            "checks. For external services, set cwd under user_files, task_drive, or "
            "artifact_store; switch to advanced/pro only for reviewed Ouroboros self-modification.",
            "light_mode_blocked",
        ),
        (
            "protected_write",
            "write_file",
            {"path": "BIBLE.md", "content": "changed\n"},
            "CORE_PROTECTION_BLOCKED",
            "⚠️ CORE_PROTECTION_BLOCKED: runtime_mode='advanced' refuses to run tool "
            "'write_file' against protected safety-critical path: BIBLE.md. Switch to "
            "runtime_mode='pro' and let the normal triad + scope review cover the protected "
            "core/contract/release change before commit.",
            "protected_blocked",
        ),
    ),
)
def test_stable_host_predispatch_denials_are_native_and_keep_legacy_projection(
    scenario,
    tool_name,
    args,
    expected_code,
    expected_text,
    legacy_status,
    tmp_path,
    monkeypatch,
):
    import json

    import ouroboros.config as config
    import ouroboros.safety as safety
    import ouroboros.tools.registry_core as registry_core
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.loop_tool_execution import _execute_single_tool
    from ouroboros.tools.registry import ToolContext, ToolRegistry
    from ouroboros.tools.tool_result import LegacyTextResultAdapter, ToolResult

    repo = tmp_path / "repo"
    drive = tmp_path / "drive"
    logs = drive / "logs"
    repo.mkdir()
    logs.mkdir(parents=True)
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    if scenario.startswith("acting_"):
        registry.set_context(
            ToolContext(
                repo_dir=repo,
                drive_root=drive,
                task_constraint=TaskConstraint(
                    mode="acting_subagent",
                    surface="self_worktree",
                ),
            )
        )

    downstream_calls: list[str] = []

    def forbidden(label):
        def fail(*_args, **_kwargs):
            downstream_calls.append(label)
            raise AssertionError(f"predispatch denial reached {label}")

        return fail

    if tool_name in registry._entries:
        registry.override_handler(tool_name, forbidden("handler"))
    monkeypatch.setattr(safety, "check_safety", forbidden("safety"))
    monkeypatch.setattr(
        LegacyTextResultAdapter,
        "from_text",
        forbidden("legacy adapter"),
    )
    monkeypatch.setattr(
        registry_core,
        "workspace_mode_block_reason",
        (
            (lambda _ctx: "fixture workspace overlap")
            if scenario == "workspace_metadata"
            else (lambda _ctx: "")
        ),
    )
    monkeypatch.setattr(
        config,
        "get_runtime_mode",
        lambda: "light" if scenario.startswith("light_") else "advanced",
    )
    if scenario == "light_repo_mutation":
        monkeypatch.setattr(
            registry_core,
            "light_cognitive_or_root_redirect",
            lambda _name, _args: None,
        )

    expected = ToolResult(
        status="blocked",
        code=expected_code,
        text=expected_text,
    )
    assert registry.execute_result(tool_name, dict(args)) == expected
    assert registry.execute(tool_name, dict(args)) == expected_text

    row = _execute_single_tool(
        registry,
        {
            "id": f"predispatch-{scenario}",
            "function": {"name": tool_name, "arguments": json.dumps(args)},
        },
        logs,
        f"task-{scenario}",
    )
    assert row["tool_result"] == expected
    assert row["result"] == expected_text
    assert row["is_error"] is True
    assert row["result_meta"] == {
        "status": legacy_status,
        "tool_result_status": "blocked",
        "tool_result_code": expected_code,
        "tool_result_meta": {},
    }
    assert downstream_calls == []


@pytest.mark.parametrize(
    ("tool_name", "args", "detail", "status", "code", "text", "legacy_status", "is_error", "adapter_calls"),
    (
        ("read_file", {"path": "x"}, "profile=acting cannot read active_workspace.", "blocked", "ACCESS_BLOCKED", "⚠️ TOOL_ACCESS_BLOCKED: profile=acting cannot read active_workspace.", "blocked", True, 0),
        ("query_code", {"op": "digest"}, "binding failed", "error", "TOOL_ARG_ERROR", "⚠️ TOOL_ARG_ERROR (query_code): RuntimeError: binding failed", "argument_error", True, 0),
        ("apply_patch", {"patch": "*** Begin Patch\n*** End Patch"}, "binding failed", "error", "TOOL_ERROR", "⚠️ TOOL_ERROR: RuntimeError: binding failed", "error", True, 0),
        ("edit_batch", {"edits": [{"path": "x", "old_str": "a", "new_str": "b", "count": 1}]}, "binding failed", "error", "TOOL_ERROR", "⚠️ TOOL_ERROR: RuntimeError: binding failed", "error", True, 0),
        ("vcs_status", {}, "binding failed", "ok", "GIT_ERROR", "⚠️ GIT_ERROR: RuntimeError: binding failed", "git_error", False, 0),
        ("vcs_diff", {}, "binding failed", "ok", "GIT_ERROR", "⚠️ GIT_ERROR: RuntimeError: binding failed", "git_error", False, 0),
        ("read_file", {"path": "x"}, "binding failed", "error", "LEGACY_TOOL_ERROR", "⚠️ READ_FILE_ERROR: RuntimeError: binding failed", "error", True, 1),
    ),
)
def test_binding_failures_cut_over_only_the_exact_native_families(
    tool_name,
    args,
    detail,
    status,
    code,
    text,
    legacy_status,
    is_error,
    adapter_calls,
    tmp_path,
    monkeypatch,
):
    import json

    import ouroboros.config as config
    import ouroboros.tools.registry_core as registry_core
    import ouroboros.tools.tool_resolution as tool_resolution
    from ouroboros.loop_tool_execution import _execute_single_tool
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.tool_result import LegacyTextResultAdapter, ToolResult

    repo = tmp_path / "repo"
    drive = tmp_path / "drive"
    logs = drive / "logs"
    repo.mkdir()
    logs.mkdir(parents=True)
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    downstream = []
    registry.override_handler(
        tool_name,
        lambda *_args, **_kwargs: downstream.append("handler") or "unreachable",
    )
    monkeypatch.setattr(config, "get_runtime_mode", lambda: "advanced")
    monkeypatch.setattr(
        "ouroboros.safety.check_safety",
        lambda *_args, **_kwargs: downstream.append("safety") or (True, ""),
    )
    monkeypatch.setattr(
        registry_core,
        "_build_builtin_target_binding",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(detail)),
    )
    adapted = []
    original_adapter = LegacyTextResultAdapter.from_text
    monkeypatch.setattr(
        LegacyTextResultAdapter,
        "from_text",
        classmethod(
            lambda _cls, name, value: (
                adapted.append((name, value))
                or original_adapter(name, value)
            )
        ),
    )

    row = _execute_single_tool(
        registry,
        {
            "id": f"binding-{tool_name}",
            "function": {"name": tool_name, "arguments": json.dumps(args)},
        },
        logs,
        "task-binding",
    )

    assert row["tool_result"] == ToolResult(status=status, code=code, text=text)
    assert row["result"] == text
    assert row["is_error"] is is_error
    assert row["result_meta"] == {
        "status": legacy_status,
        "tool_result_status": status,
        "tool_result_code": code,
        "tool_result_meta": {},
    }
    assert len(adapted) == adapter_calls
    assert downstream == []
    if tool_name == "apply_patch":
        assert tool_resolution._binding_error_text(
            "future_binding_tool",
            "active_workspace",
            RuntimeError("binding failed"),
        ) == ToolResult(
            status="error",
            code="TOOL_ERROR",
            text="⚠️ TOOL_ERROR: RuntimeError: binding failed",
        )


def test_light_binding_root_redirect_is_native_without_invented_metadata(
    tmp_path,
    monkeypatch,
):
    import json

    import ouroboros.config as config
    import ouroboros.tools.registry_core as registry_core
    import ouroboros.tools.tool_resolution as tool_resolution
    from ouroboros.loop_tool_execution import _execute_single_tool
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.tool_result import LegacyTextResultAdapter, ToolResult

    repo = tmp_path / "repo"
    drive = tmp_path / "drive"
    home = tmp_path / "home"
    logs = drive / "logs"
    repo.mkdir()
    home.mkdir()
    logs.mkdir(parents=True)
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    calls = []
    registry.override_handler(
        "write_file",
        lambda *_args, **_kwargs: calls.append("handler") or "unreachable",
    )
    monkeypatch.setattr(config, "get_runtime_mode", lambda: "light")
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    monkeypatch.setattr(
        registry_core,
        "_build_builtin_target_binding",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("outside root")),
    )
    monkeypatch.setattr(
        "ouroboros.safety.check_safety",
        lambda *_args, **_kwargs: calls.append("safety") or (True, ""),
    )
    monkeypatch.setattr(
        LegacyTextResultAdapter,
        "from_text",
        classmethod(
            lambda _cls, *_args, **_kwargs: pytest.fail("legacy adapter used")
        ),
    )
    # as_posix(): the dispatch layer normalizes path args to forward slashes, so
    # the refusal echoes the posix spelling; feed it in that spelling so the
    # expected repr matches on Windows too (POSIX: identical to str()).
    target = (home / "Desktop" / "report.html").as_posix()
    args = {"path": target, "content": "<html></html>"}
    text = (
        "⚠️ ROOT_REQUIRED_USER_FILES: an absolute home path "
        f"({target!r}) was given but root defaulted to 'active_workspace'. "
        "Pass root='user_files' to write under the owner's home, e.g. "
        "write_file(root='user_files', path='Desktop/file.html', content=...)."
    )
    expected = ToolResult(status="blocked", code="ROOT_REQUIRED_USER_FILES", text=text)

    row = _execute_single_tool(
        registry,
        {
            "id": "binding-light-root",
            "function": {"name": "write_file", "arguments": json.dumps(args)},
        },
        logs,
        "task-binding-light",
    )

    assert row["tool_result"] == expected
    assert row["is_error"] is True
    assert row["result_meta"] == {
        "status": "root_required_user_files",
        "tool_result_status": "blocked",
        "tool_result_code": "ROOT_REQUIRED_USER_FILES",
        "tool_result_meta": {},
    }
    cognitive = tool_resolution._light_binding_failure_result(
        "write_file",
        {"root": "runtime_data", "path": "memory/identity.md"},
    )
    assert isinstance(cognitive, str) and cognitive.startswith("⚠️ COGNITIVE_TOOL_REQUIRED:")
    assert calls == []


@pytest.mark.parametrize(
    ("scenario", "expected_status", "expected_code", "legacy_status"),
    (
        # T1: the cognitive redirect is its own ok-status code (owner batch #4),
        # and the root redirect names the root it demands (§A.15).
        ("cognitive", "ok", "COGNITIVE_TOOL_REQUIRED", "ok"),
        ("user_files", "blocked", "ROOT_REQUIRED_USER_FILES", "root_required_user_files"),
    ),
)
def test_light_actionable_redirects_keep_legacy_mapping_without_light_remap(
    scenario,
    expected_status,
    expected_code,
    legacy_status,
    tmp_path,
    monkeypatch,
):
    import json

    from ouroboros.loop_tool_execution import _execute_single_tool
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.tool_resolution import (
        _build_builtin_target_binding,
        _light_binding_failure_result,
    )
    from ouroboros.tools.tool_result import LegacyTextResultAdapter, ToolResult

    repo = tmp_path / "repo"
    drive = tmp_path / "drive"
    home = tmp_path / "home"
    logs = drive / "logs"
    repo.mkdir()
    home.mkdir()
    logs.mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    downstream_calls: list[str] = []

    def forbidden(label):
        def fail(*_args, **_kwargs):
            downstream_calls.append(label)
            raise AssertionError(f"light redirect reached {label}")

        return fail

    registry.override_handler("write_file", forbidden("handler"))
    monkeypatch.setattr("ouroboros.safety.check_safety", forbidden("safety"))
    adapter_calls = []
    original_adapter = LegacyTextResultAdapter.from_text
    monkeypatch.setattr(
        LegacyTextResultAdapter,
        "from_text",
        classmethod(
            lambda _cls, name, text: (
                adapter_calls.append((name, text))
                or original_adapter(name, text)
            )
        ),
    )

    if scenario == "cognitive":
        args = {
            "root": "runtime_data",
            "path": "memory/identity.md",
            "content": "x" * 60,
        }
        expected_text = (
            "⚠️ COGNITIVE_TOOL_REQUIRED: cognitive memory is not written via 'write_file'. "
            "Use the dedicated first-class tools (always available in light mode): "
            "update_identity for memory/identity.md, update_scratchpad for "
            "memory/scratchpad.md, knowledge_write for memory/knowledge/<topic>.md. They "
            "apply the correct structure (journaling, timestamped blocks, index maintenance). "
            "Read the current state before writing (Bible P12)."
        )
    else:
        # as_posix() for the same reason as in
        # test_light_binding_root_redirect_is_native_without_invented_metadata.
        target = (home / "Desktop" / "report.html").as_posix()
        args = {"path": target, "content": "<html></html>"}
        expected_text = (
            "⚠️ ROOT_REQUIRED_USER_FILES: an absolute home path "
            f"({target!r}) was given but root defaulted to 'active_workspace'. "
            "Pass root='user_files' to write under the owner's home, e.g. "
            "write_file(root='user_files', path='Desktop/file.html', content=...)."
        )

    expected = ToolResult(
        status=expected_status,
        code=expected_code,
        text=expected_text,
    )
    assert registry.execute_result("write_file", dict(args)) == expected
    assert registry.execute("write_file", dict(args)) == expected_text
    assert "LIGHT_MODE_BLOCKED" not in expected_text

    row = _execute_single_tool(
        registry,
        {
            "id": f"light-redirect-{scenario}",
            "function": {"name": "write_file", "arguments": json.dumps(args)},
        },
        logs,
        f"task-light-redirect-{scenario}",
    )
    assert row["tool_result"] == expected
    assert row["result"] == expected_text
    # T1 §A.11 (owner batch #4): the cognitive redirect names a better tool, so it
    # is no longer an error row; the root redirect is still a real refusal.
    assert row["is_error"] is (expected_status != "ok")
    assert row["result_meta"] == {
        "status": legacy_status,
        "tool_result_status": expected_status,
        "tool_result_code": expected_code,
        "tool_result_meta": {},
    }
    # The refusal CONTRACT (status/code/text/meta, asserted above) is identical on
    # every OS; only the ROUTE can differ, and the code's own predicate for it is
    # never which platform we are on (R-WINWAVE class 5, expression follow-up).
    # Two arms reach the same product outcome: registry_core's light repo-mutation
    # branch (when _build_builtin_target_binding RETURNED a binding) answers with
    # the legacy TEXT, and the except arm (when it RAISES) answers from the
    # resolution layer. The adapter count follows the SHAPE that arm returns, NOT
    # whether the binding resolved: `_light_binding_failure_result` types the root
    # redirect as a NATIVE ToolResult (0 adapter calls) and hands the cognitive
    # redirect back as legacy TEXT (3 calls, exactly like the binding-resolves
    # branch). So derive the expected route from the code's own except arm.
    # `os.name == "nt"` was a stand-in for one outcome of the binding predicate
    # (on Windows pytest's tmp_path arrives in 8.3 short form, resolve() expands
    # it and relative_to() misses inside the binding) — and keying on the binding
    # predicate ALONE inherits that stand-in's second, unstated assumption, that
    # the cognitive scenario can never take the except arm: forced onto it, the
    # cognitive redirect still costs 3 adapter calls while `binding_resolves`
    # would demand 0.
    try:
        _build_builtin_target_binding(registry._ctx, "write_file", dict(args))
    except Exception:
        failure_route = _light_binding_failure_result("write_file", dict(args))
    else:
        failure_route = None
    expected_adapter_calls = 0 if isinstance(failure_route, ToolResult) else 3
    assert len(adapter_calls) == expected_adapter_calls
    assert downstream_calls == []
