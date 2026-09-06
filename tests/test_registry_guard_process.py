from __future__ import annotations

import inspect
import pathlib
import subprocess
from types import SimpleNamespace

import ouroboros.tools.registry as registry_module
import ouroboros.tools.registry_guard_process as process_guard
import ouroboros.tools.registry_guards as registry_guards
from ouroboros.artifacts import task_artifact_dir_path
from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.tools.tool_result import LegacyTextResultAdapter, ToolResult


# Tip adaptations against the reference tables, each disclosed:
# - the whole owner-control detector family carries the ``writeish`` read-carve
#   keyword on THIS tree (the v6.80.0 contract extended to every member; the
#   reference had it on the floor detector only);
# - ``_detect_scope_review_floor_self_lowering`` is ABSENT here: the setting and
#   its guard were retired outright in the 7.0 ABI window (owner Q10=A);
# - three constant cardinalities moved by upstream drift after the fork
#   (_SUBAGENT_SHELL_SECRET_MARKERS 17->18, _DENIED_READ_OPTIONS 11->12,
#   _SKILL_OWNER_STATE_STEMS 12->14; 14->15 with the CPL4-C11 uninstall
#   tombstone joining the owner-state allowlist);
# - ``_workspace_shell_write_block`` takes the upstream per-segment ``target_rows``
#   (which replaced ``write_target_argvs``).
_FUNCTION_SIGNATURES = {
    "_detect_runtime_mode_elevation": "(text_lower: 'str', *, writeish: 'bool' = True) -> 'bool'",
    "_subagent_shell_targets_secret": "(cmd_path_lower: 'str', *, ctx: 'Any' = None, cwd: 'Any' = None) -> 'bool'",
    "_detect_mutative_toggle_self_change": "(text_lower: 'str', *, writeish: 'bool' = True) -> 'bool'",
    "_detect_evolution_owner_control_self_change": "(text_lower: 'str', *, writeish: 'bool' = True) -> 'bool'",
    "_detect_context_mode_self_lowering": "(text_lower: 'str', *, writeish: 'bool' = True) -> 'bool'",
    "_trusted_read_head": "(token: 'str') -> 'str'",
    "_denied_read_option": "(token: 'str', denied: 'frozenset') -> 'bool'",
    "_is_pure_read_inspection": "(text_lower: 'str') -> 'bool'",
    "_detect_safety_mode_self_lowering": "(text_lower: 'str', *, writeish: 'bool' = True) -> 'bool'",
    "_detect_owner_skill_attest_self_call": "(text_lower: 'str', *, writeish: 'bool' = True) -> 'bool'",
    "_mentions_skill_owner_state": "(text_lower: 'str', *, writeish: 'bool' = True) -> 'bool'",
    "_mentions_detached_process": "(text_lower: 'str') -> 'bool'",
    "_run_shell_safety_check": "(self, args: 'Dict[str, Any]', runtime_mode: 'str', binding: 'Any' = None) -> 'ToolResult | None'",
    "_light_repo_snapshot": "(repo_dir: 'pathlib.Path') -> 'Optional[Dict[str, Any]]'",
    "_format_light_repo_write_note": "(before: 'Dict[str, Any]', after: 'Dict[str, Any]', tool_name: 'str' = 'run_command') -> 'str'",
    "_git_ref_snapshot": "(repo_dir: 'pathlib.Path') -> 'Optional[Dict[str, str]]'",
    "_owner_settings_snapshot": "() -> 'Optional[str]'",
    "_run_shell_post_checks": "(self, result: 'str | ToolResult', *, light_repo_before: 'Optional[Dict[str, Any]]', workspace_refs_before: 'Optional[Dict[str, str]]', settings_before: 'Optional[str]' = None, tool_name: 'str' = 'run_command') -> 'str | ToolResult'",
}

_REGISTRY_GUARD_SIGNATURES = {
    "_executor_backend_candidate_allowed": "(ctx: 'Any', candidate: 'str', allowed_roots: 'List[pathlib.Path]') -> 'bool'",
    "_command_mentions_protected_root": "(cmd_path_lower: 'str', root_text: 'str') -> 'bool'",
    "_authorized_managed_update_resolver": "(ctx: 'Any') -> 'bool'",
    "_light_mode_payload_mutation_allowed": "(*, ctx: 'Any', tool_name: 'str', args: 'Dict[str, Any]', runtime_mode: 'str', effective_constraint: 'Optional[TaskConstraint]', implicit_skill_cwd_allowed: 'bool', allow_short_relative: 'bool') -> 'bool'",
    "_protected_shell_block": "(self, raw_cmd, cmd_path_lower, binding, acting_self_worktree, writeish) -> 'ToolResult | None'",
    "_git_protected_roots": "(self) -> 'list'",
    "_resolved_shell_cwd": "(self, args: 'Dict[str, Any]', binding: 'Any' = None) -> 'pathlib.Path | ToolResult'",
    "_external_workspace_git_block": "(self, raw_cmd: 'Any', work_dir: 'pathlib.Path') -> 'ToolResult | None'",
    "_external_runtime_protected_paths": "(self, binding: 'Any' = None) -> 'tuple[list, list, list, list]'",
    "_external_shell_runtime_or_secret_block": "(self, raw_cmd: 'Any', cmd_path_lower: 'str', args: 'Dict[str, Any]', work_dir: 'Optional[pathlib.Path]' = None, binding: 'Any' = None) -> 'ToolResult | None'",
    "_workspace_shell_write_block": "(self, args: 'Dict[str, Any]', raw_cmd: 'Any', cmd_path_lower: 'str', explicit_write_targets: 'list[str]', target_rows: 'list', executable_path_tokens: 'set[str]', runtime_mode: 'str', acting_subagent: 'bool', binding: 'Any') -> 'ToolResult | None'",
    "_shell_git_and_runtime_block": "(self, raw_cmd: 'Any', args: 'Dict[str, Any]', cmd_path_lower: 'str', workspace_mode: 'bool', acting_self_worktree: 'bool', binding: 'Any') -> 'ToolResult | None'",
}

_CONSTANT_CARDINALITIES = {
    "_READ_ONLY_INSPECTION_COMMANDS": 41,
    "_COMMAND_HEAD_WRAPPERS": 11,
    "_SEARCH_TOOL_EXEC_OPTIONS": 4,
    "_DENIED_READ_OPTIONS": 12,
    "_TRUSTED_EXECUTABLE_DIRS": 6,
    "_NESTED_EXECUTION_MARKERS": 4,
    "_NESTED_EXECUTION_TOKENS": 6,
    "_SKILL_OWNER_STATE_STEMS": 15,
    "_DETACHED_PROCESS_MARKERS": 5,
}

_RETIRED_REGISTRY_DEPENDENCIES = frozenset({
    "LIGHT_SHELL_WRITER_COMMANDS",
    "SKILL_OWNER_STATE_FILENAMES",
    "SKILL_OWNER_STATE_STEMS",
    "build_resolved_resource_binding",
    "interpreter_family",
    "light_shell_repo_mutation",
    "parse_porcelain_paths",
    "protected_artifact_shell_block_reason",
    "runtime_data_guard_targets",
    "safe_relpath",
    "shell_command_string",
    "strip_leading_env_assignments",
    "sudo_noninteractive_violation",
    "unwrap_env_argv",
    "workspace_executor_state_write_block",
    "writer_target_tokens",
})


def test_process_guard_owner_surface_is_exact_and_retired_from_registry():
    for name, signature in _FUNCTION_SIGNATURES.items():
        assert str(inspect.signature(getattr(process_guard, name))) == signature
    for name, cardinality in _CONSTANT_CARDINALITIES.items():
        assert len(getattr(process_guard, name)) == cardinality

    # Tip adaptation: the reference also asserted every moved name's absence
    # from the registry facade; on this tree the facade deliberately re-exports
    # the historical surface for its importers and monkeypatch targets, so only
    # the OWNER homing (above) and the ToolRegistry method retirement are
    # pinned. _RETIRED_REGISTRY_DEPENDENCIES stays as the moved-name inventory.
    assert _RETIRED_REGISTRY_DEPENDENCIES
    assert not hasattr(ToolRegistry, "_run_shell_safety_check")
    assert not hasattr(ToolRegistry, "_run_shell_post_checks")
    # The post-hoc owner-state snapshot/restore organ was DELETED (#447); no
    # module owns it any more, on the registry or on this leaf.
    assert not hasattr(process_guard, "_snapshot_owner_files")
    assert not hasattr(process_guard, "_restore_owner_files")


def test_registry_shell_guard_owner_surface_is_exact_and_retired_from_registry():
    for name, signature in _REGISTRY_GUARD_SIGNATURES.items():
        assert str(inspect.signature(getattr(registry_guards, name))) == signature

    assert (
        registry_module._authorized_managed_update_resolver
        is registry_guards._authorized_managed_update_resolver
    )
    retired_module_names = {
        "PROTECTED_RUNTIME_PATHS",
        "PROTECTED_RUNTIME_PATHS_LOWER",
        "SKILL_PAYLOAD_CONTROL_DIRNAMES",
        "_executor_backend_candidate_allowed",
        "_command_mentions_protected_root",
        "_light_mode_payload_mutation_allowed",
        "is_absolute_path_text",
        "is_external_workspace",
        "is_skill_payload_path",
        "normalize_root",
        "path_text_is_inside",
        "resolve_shell_cwd",
        "resolve_skill_payload_target",
        "run_shell_git_block_reason",
        "shell_argv",
        "shell_argv_with_path_tokens",
        "shell_has_write_indicator",
        "shell_writer_targets_protected",
        "task_artifact_dir_path",
        "task_id_for_artifacts",
        "workspace_git_safety_violation",
    }
    # Tip adaptation: facade re-exports stay (see the note in the sibling test);
    # the ToolRegistry method retirement below is the durable half of the pin.
    assert retired_module_names
    retired_methods = set(_REGISTRY_GUARD_SIGNATURES) - {
        "_executor_backend_candidate_allowed",
        "_command_mentions_protected_root",
        "_authorized_managed_update_resolver",
        "_light_mode_payload_mutation_allowed",
    }
    assert all(not hasattr(ToolRegistry, name) for name in retired_methods)


class _RegistryStub:
    def __init__(self, root: pathlib.Path):
        self.acting = False
        self._ctx = SimpleNamespace(
            drive_root=root,
            repo_dir=root,
            task_id="task-process-guard",
            is_workspace_mode=lambda: False,
            task_drive_root=lambda: root / "task_drive",
        )
        self.work_dir = root

    def _acting_self_worktree(self):
        return False

    def _is_acting_subagent(self):
        return self.acting

    def _is_local_readonly_subagent(self):
        return False


def test_process_guard_uses_explicit_registry_guard_owners_once_in_order(
    tmp_path, monkeypatch,
):
    stub = _RegistryStub(tmp_path)
    stub._ctx.is_workspace_mode = lambda: True
    monkeypatch.setattr(
        registry_module,
        "protected_artifact_shell_block_reason",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        registry_module,
        "workspace_executor_state_write_block",
        lambda *_args, **_kwargs: None,
    )
    stages = (
        "_resolved_shell_cwd",
        "_workspace_shell_write_block",
        "_protected_shell_block",
        "_shell_git_and_runtime_block",
    )
    denial = ToolResult(status="blocked", code="WORKSPACE_BLOCKED", text="blocked")

    for stop_index in range(len(stages) + 1):
        calls: list[str] = []

        def resolved(owner, args, binding=None):
            assert owner is stub
            assert args["cmd"] == ["touch", "out.txt"]
            assert binding == ()
            calls.append("_resolved_shell_cwd")
            return denial if stop_index == 0 else tmp_path

        def workspace(owner, *args):
            assert owner is stub
            calls.append("_workspace_shell_write_block")
            return denial if stop_index == 1 else None

        def protected(owner, *args):
            assert owner is stub
            calls.append("_protected_shell_block")
            return denial if stop_index == 2 else None

        def git_runtime(owner, *args):
            assert owner is stub
            calls.append("_shell_git_and_runtime_block")
            return denial if stop_index == 3 else None

        monkeypatch.setattr(registry_guards, "_resolved_shell_cwd", resolved)
        monkeypatch.setattr(registry_guards, "_workspace_shell_write_block", workspace)
        monkeypatch.setattr(registry_guards, "_protected_shell_block", protected)
        monkeypatch.setattr(registry_guards, "_shell_git_and_runtime_block", git_runtime)

        result = process_guard._run_shell_safety_check(
            stub,
            {"cmd": ["touch", "out.txt"]},
            "advanced",
            (),
        )
        expected_calls = list(stages[: min(stop_index + 1, len(stages))])
        assert calls == expected_calls
        assert result is (None if stop_index == len(stages) else denial)


def test_authorized_managed_update_resolver_preserves_allow_and_fail_closed(
    monkeypatch,
):
    from supervisor import update_merge

    ctx = SimpleNamespace(task_id="task-resolver", task_metadata={"tx": "fixture"})
    calls = []

    # Tip adaptation: the guard reads the TYPED strict variant
    # (authorized_assisted_task_strict -> (marker_status, tx)) so the loud
    # A4 corruption channel exists; the reference patched the plain reader.
    def authorized(task_id, metadata):
        calls.append((task_id, metadata))
        return "valid", {"tx": "truthy"}

    monkeypatch.setattr(update_merge, "authorized_assisted_task_strict", authorized)
    assert registry_guards._authorized_managed_update_resolver(ctx) is True
    assert calls == [("task-resolver", {"tx": "fixture"})]
    assert ctx._managed_authority_read_error == ""

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("fixture")

    monkeypatch.setattr(update_merge, "authorized_assisted_task_strict", unavailable)
    assert registry_guards._authorized_managed_update_resolver(ctx) is False
    assert "fixture" in ctx._managed_authority_read_error

    def corrupt(*_args, **_kwargs):
        return "corrupt", {}

    monkeypatch.setattr(update_merge, "authorized_assisted_task_strict", corrupt)
    assert registry_guards._authorized_managed_update_resolver(ctx) is False
    assert "update_tx_corrupt" in ctx._managed_authority_read_error


def test_git_protected_roots_preserve_order_and_duplicate_semantics(tmp_path):
    roots = {
        name: tmp_path / name
        for name in (
            "system", "drive", "meta-drive", "child-drive",
            "headless-drive", "budget-drive",
        )
    }
    stub = _RegistryStub(tmp_path)
    stub._ctx = SimpleNamespace(
        system_repo_dir=roots["system"],
        repo_dir=roots["system"],
        drive_root=roots["drive"],
        task_metadata={
            "drive_root": str(roots["meta-drive"]),
            "child_drive_root": str(roots["child-drive"]),
            "headless_child_drive_root": str(roots["headless-drive"]),
            "budget_drive_root": str(roots["budget-drive"]),
        },
    )

    assert registry_guards._git_protected_roots(stub) == [
        roots["system"],
        roots["system"],
        roots["drive"],
        roots["meta-drive"],
        roots["child-drive"],
        roots["headless-drive"],
        roots["budget-drive"],
    ]


def test_owner_settings_snapshot_distinguishes_absent_from_unreadable(tmp_path, monkeypatch):
    """The tripwire BASELINE is honest about not being able to read (#447).

    The deleted restore recorded an OSError as "file absent" and could then
    unlink the live settings.json; the replacement returns None, which disarms
    the tripwire instead of arming it against a phantom baseline."""
    from ouroboros import config

    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(config, "SETTINGS_PATH", settings_path)
    assert process_guard._owner_settings_snapshot() == ""

    settings_path.write_text('{"owner":"before"}', encoding="utf-8")
    assert process_guard._owner_settings_snapshot() == '{"owner":"before"}'

    def boom(*_args, **_kwargs):
        raise OSError("transient")

    monkeypatch.setattr(pathlib.Path, "read_text", boom)
    assert process_guard._owner_settings_snapshot() is None


def test_light_repo_formatter_preserves_sorted_bounded_path_disclosure():
    note = process_guard._format_light_repo_write_note(
        {"paths": ["z.py", "b.py"]},
        {"paths": ["a.py", "b.py"]},
        tool_name="run_script",
    )
    assert note == (
        "⚠️ LIGHT_MODE_REPO_WRITE_BLOCKED: runtime_mode=light detected a mutation of the "
        "Ouroboros repository after run_script. The command result is blocked and no "
        "automatic rollback was attempted to avoid overwriting concurrent human edits. "
        "Affected/dirty paths: a.py, b.py, z.py. Switch to advanced/pro for repo writes."
    )

    crowded = process_guard._format_light_repo_write_note(
        {"paths": [f"p{index:02d}" for index in range(31)]},
        {"paths": []},
    )
    assert "p29, ... (+1 more)" in crowded
    assert "p30" not in crowded
    assert "Affected/dirty paths: (status changed; no paths parsed)." in (
        process_guard._format_light_repo_write_note({}, {})
    )


def test_git_ref_snapshot_detects_a_ref_only_change(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "tracked.txt").write_text("content\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(
        [
            "git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
            "commit", "-qm", "fixture",
        ],
        cwd=repo,
        check=True,
    )

    before = process_guard._git_ref_snapshot(repo)
    subprocess.run(["git", "tag", "fixture-tag"], cwd=repo, check=True)
    after = process_guard._git_ref_snapshot(repo)

    assert before is not None and after is not None
    assert before["head"] == after["head"]
    assert before["digest"] != after["digest"]


def test_process_post_checks_append_every_tripwire_note_after_the_payload(
    tmp_path, monkeypatch,
):
    """Every tripwire ANNOTATES and none rolls back (#447 В12/H1).

    Line 1 stays with the command payload, so the failure classifier still reads
    the producer's own first line; the settings tripwire is a typed fact, not a
    status change, because the host never proved the command caused the write."""
    stub = _RegistryStub(tmp_path)
    events: list[str] = []

    def light_snapshot(repo_dir):
        assert repo_dir == tmp_path
        events.append("light_snapshot")
        return {"digest": "light-after", "paths": ["changed.py"]}

    def ref_snapshot(repo_dir):
        assert repo_dir == tmp_path
        events.append("ref_snapshot")
        return {"head": "same", "digest": "refs-after"}

    def settings_snapshot():
        events.append("settings_snapshot")
        return '{"owner":"after"}'

    monkeypatch.setattr(process_guard, "_owner_settings_snapshot", settings_snapshot)
    monkeypatch.setattr(process_guard, "_light_repo_snapshot", light_snapshot)
    monkeypatch.setattr(process_guard, "_git_ref_snapshot", ref_snapshot)
    monkeypatch.setattr(registry_module, "system_repo_dir_for", lambda _ctx: tmp_path)
    monkeypatch.setattr(registry_module, "active_repo_dir_for", lambda _ctx: tmp_path)

    result = process_guard._run_shell_post_checks(
        stub,
        ToolResult(
            status="ok",
            code="SHELL_NO_MATCH",
            text="handler output",
            meta={"exit_code": 1},
        ),
        light_repo_before={"digest": "light-before", "paths": []},
        workspace_refs_before={"head": "same", "digest": "refs-before"},
        settings_before='{"owner":"before"}',
        tool_name="run_script",
    )

    assert events == ["settings_snapshot", "light_snapshot", "ref_snapshot"]
    assert isinstance(result, ToolResult)
    assert result.text.splitlines()[0] == "handler output"
    assert result.text == (
        "handler output\n\n"
        "⚠️ OWNER_SETTINGS_CHANGED: data/settings.json changed while this command ran. "
        "Owner settings change only through save_settings / the Settings UI; this write "
        "was NOT auto-reverted (a post-hoc rollback can clobber a concurrent legitimate "
        "owner edit) — the owner surface is the place to verify and restore.\n\n"
        "⚠️ LIGHT_MODE_REPO_WRITE_BLOCKED: runtime_mode=light detected a mutation of the "
        "Ouroboros repository after run_script. The command result is blocked and no "
        "automatic rollback was attempted to avoid overwriting concurrent human edits. "
        "Affected/dirty paths: changed.py. Switch to advanced/pro for repo writes.\n\n"
        "⚠️ WORKSPACE_GIT_REF_CHANGED: run_command changed git HEAD or refs inside the "
        "external workspace. External workspace runs must leave changes as files/patch "
        "artifacts, not commits/tags/resets."
    )
    assert (result.status, result.code) == ("blocked", "WORKSPACE_GIT_REF_CHANGED")
    assert dict(result.meta) == {
        "exit_code": 1,
        "tripwire": "owner_settings_changed",
        "light_repo_changed": True,
        "workspace_git_refs_changed": True,
    }


def test_settings_tripwire_annotates_without_changing_the_command_outcome(
    tmp_path, monkeypatch,
):
    stub = _RegistryStub(tmp_path)
    monkeypatch.setattr(
        process_guard, "_owner_settings_snapshot", lambda: '{"owner":"after"}',
    )

    success = process_guard._run_shell_post_checks(
        stub,
        ToolResult(
            status="ok",
            code="OK",
            text="exit_code=0\nSTDOUT:\nok",
            meta={"exit_code": 0},
        ),
        light_repo_before=None,
        workspace_refs_before=None,
        settings_before='{"owner":"before"}',
    )
    failure = process_guard._run_shell_post_checks(
        stub,
        ToolResult(
            status="error",
            code="SHELL_EXIT_ERROR",
            text="⚠️ SHELL_EXIT_ERROR: failed",
            meta={"exit_code": 2},
        ),
        light_repo_before=None,
        workspace_refs_before=None,
        settings_before='{"owner":"before"}',
    )

    assert isinstance(success, ToolResult)
    assert (success.status, success.code) == ("ok", "OK")
    assert dict(success.meta) == {
        "exit_code": 0,
        "tripwire": "owner_settings_changed",
    }
    assert isinstance(failure, ToolResult)
    assert (failure.status, failure.code) == ("error", "SHELL_EXIT_ERROR")
    assert dict(failure.meta) == {
        "exit_code": 2,
        "tripwire": "owner_settings_changed",
    }
    assert failure.text.splitlines()[0] == "⚠️ SHELL_EXIT_ERROR: failed"


def test_unreadable_settings_baseline_disarms_the_tripwire(tmp_path, monkeypatch):
    stub = _RegistryStub(tmp_path)
    monkeypatch.setattr(process_guard, "_owner_settings_snapshot", lambda: None)

    result = process_guard._run_shell_post_checks(
        stub,
        ToolResult(status="ok", code="OK", text="out", meta={}),
        light_repo_before=None,
        workspace_refs_before=None,
        settings_before='{"owner":"before"}',
    )
    assert isinstance(result, ToolResult)
    assert result.text == "out"
    assert dict(result.meta) == {}


def test_registry_dispatch_calls_process_post_owner_once_after_handler(
    tmp_path, monkeypatch,
):
    from ouroboros import safety

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry.set_context(ToolContext(repo_dir=repo, drive_root=data, task_id="task-post-owner"))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    calls: list[str] = []

    def pre_guard(*_args, **_kwargs):
        calls.append("pre_guard")
        return None

    def check_safety(*_args, **_kwargs):
        calls.append("safety")
        return True, ""

    def snapshot():
        calls.append("snapshot")
        return '{"owner":"before"}'

    def handler(_ctx, cmd, _resolved_binding=None, **_kwargs):
        assert cmd == ["echo", "ok"]
        assert _resolved_binding is not None
        calls.append("handler")
        return "handler output"

    def post(owner, result, **kwargs):
        assert owner is registry
        assert result == "handler output"
        assert kwargs == {
            "light_repo_before": None,
            "workspace_refs_before": None,
            "settings_before": '{"owner":"before"}',
            "tool_name": "run_command",
        }
        calls.append("post")
        return result

    original_adapter = LegacyTextResultAdapter.from_text

    def adapt(tool_name, text):
        calls.append("adapter")
        return original_adapter(tool_name, text)

    monkeypatch.setattr(process_guard, "_run_shell_safety_check", pre_guard)
    monkeypatch.setattr(process_guard, "_owner_settings_snapshot", snapshot)
    monkeypatch.setattr(process_guard, "_run_shell_post_checks", post)
    monkeypatch.setattr(safety, "check_safety", check_safety)
    monkeypatch.setattr(LegacyTextResultAdapter, "from_text", adapt)
    registry.override_handler("run_command", handler)

    assert registry.execute("run_command", {"cmd": ["echo", "ok"]}) == "handler output"
    assert calls == ["pre_guard", "safety", "snapshot", "handler", "post", "adapter"]
    assert not hasattr(registry._ctx, "_active_builtin_tool_result")


def test_custom_handler_cannot_reuse_stale_builtin_result_sidecar(
    tmp_path, monkeypatch,
):
    from ouroboros import safety

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    stale = ToolResult(
        status="error",
        code="SHELL_EXIT_ERROR",
        text="custom output",
        meta={"exit_code": 93},
    )
    registry._ctx._active_builtin_tool_result = stale
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr(safety, "check_safety", lambda *_args, **_kwargs: (True, ""))
    monkeypatch.setattr(process_guard, "_run_shell_safety_check", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(process_guard, "_owner_settings_snapshot", lambda: "")
    monkeypatch.setattr(
        process_guard,
        "_run_shell_post_checks",
        lambda _owner, result, **_kwargs: result,
    )
    registry.override_handler(
        "run_command",
        lambda _ctx, cmd, _resolved_binding=None, **_kwargs: "custom output",
    )

    result = registry.execute_result("run_command", {"cmd": ["echo", "ok"]})

    assert (result.status, result.code, result.text, dict(result.meta)) == (
        "ok",
        "OK",
        "custom output",
        {},
    )
    assert registry._ctx._active_builtin_tool_result is stale

    def mismatched(ctx, cmd, _resolved_binding=None, **_kwargs):
        ctx._active_builtin_tool_result = ToolResult(
            status="error",
            code="SHELL_EXIT_ERROR",
            text="different output",
            meta={"exit_code": 93},
        )
        return "custom output"

    registry.override_handler("run_command", mismatched)
    mismatch = registry.execute_result(
        "run_command", {"cmd": ["echo", "ok"]},
    )
    assert (mismatch.status, mismatch.code, dict(mismatch.meta)) == (
        "ok",
        "OK",
        {},
    )
    assert registry._ctx._active_builtin_tool_result is stale

    delattr(registry._ctx, "_active_builtin_tool_result")
    mismatch_without_prior = registry.execute_result(
        "run_command", {"cmd": ["echo", "ok"]},
    )
    assert (
        mismatch_without_prior.status,
        mismatch_without_prior.code,
        dict(mismatch_without_prior.meta),
    ) == ("ok", "OK", {})
    assert not hasattr(registry._ctx, "_active_builtin_tool_result")


def test_process_guard_denials_preserve_exact_text(tmp_path, monkeypatch):
    stub = _RegistryStub(tmp_path)
    monkeypatch.setattr(registry_module, "protected_artifact_shell_block_reason", lambda *args, **kwargs: None)
    monkeypatch.setattr(registry_module, "workspace_executor_state_write_block", lambda *args, **kwargs: None)

    def check(command, mode="advanced"):
        result = process_guard._run_shell_safety_check(stub, {"cmd": command}, mode, ())
        assert isinstance(result, ToolResult)
        assert result.status == "blocked"
        assert dict(result.meta) == {}
        legacy = LegacyTextResultAdapter.from_text("run_command", result.text)
        assert (result.status, result.code, dict(result.meta)) == (
            legacy.status,
            legacy.code,
            dict(legacy.meta),
        )
        return result

    sudo = check(["sudo", "true"])
    assert sudo.code == "SUDO_INTERACTIVE_BLOCKED"
    assert sudo.text == (
        "⚠️ SUDO_INTERACTIVE_BLOCKED: sudo must be noninteractive. Use sudo -n for commands "
        "that can run without a password; if sudo -n fails, report validation/install blocked "
        "by environment."
    )

    stub.acting = True
    secret = check("cat .env")
    assert secret.code == "SUBAGENT_SECRET_READ_BLOCKED"
    assert secret.text == (
        "⚠️ SUBAGENT_SECRET_READ_BLOCKED: subagents may not read Ouroboros secrets, "
        "credentials, or owner-control state via shell. Use the gated read_file tool "
        "(which denies secrets) for any inspection you actually need."
    )
    stub.acting = False

    cases = (
        (
            'save_settings({"ouroboros_runtime_mode":"pro"})',
            "ELEVATION_BLOCKED",
            "⚠️ ELEVATION_BLOCKED: shell command pattern looks like an OUROBOROS_RUNTIME_MODE "
            "elevation attempt (mentions ``save_settings`` together with ``OUROBOROS_RUNTIME_MODE``, "
            "or invokes ``ouroboros.config.save_settings`` directly). Runtime mode is "
            "owner-controlled — change it by stopping the agent and editing settings.json "
            "directly, then restart.",
        ),
        (
            'save_settings({"ouroboros_context_mode":"low"})',
            "CONTEXT_MODE_SELF_LOWERING_BLOCKED",
            "⚠️ CONTEXT_MODE_SELF_LOWERING_BLOCKED: shell command pattern looks like an attempt "
            "to lower OUROBOROS_CONTEXT_MODE to low through settings.json or /api/owner/context-mode. "
            "Context mode is owner-controlled — ask the owner to change the Low/Max toggle or edit "
            "settings while the agent is stopped.",
        ),
        # Tip adaptation: the SCOPE_REVIEW_FLOOR self-lowering denial is gone —
        # the setting and its guard were retired in the 7.0 ABI window (Q10=A).
        (
            'save_settings({"ouroboros_safety_mode":"off"})',
            "SAFETY_MODE_SELF_LOWERING_BLOCKED",
            "⚠️ SAFETY_MODE_SELF_LOWERING_BLOCKED: shell command pattern looks like an attempt to "
            "change OUROBOROS_SAFETY_MODE (e.g. to ``light``/``off``) through settings.json, "
            "/api/settings, or /api/owner/safety-mode. LLM-safety coverage is owner-controlled "
            "(BIBLE P3) — the agent must not reduce its own supervision. Ask the owner to change "
            "it via the dedicated /api/owner/safety-mode endpoint, or stop the agent and edit "
            "settings.json directly.",
        ),
        (
            "curl -x post /api/owner/skills/alpha/attest-review",
            "OWNER_SKILL_ATTESTATION_SELF_CALL_BLOCKED",
            "⚠️ OWNER_SKILL_ATTESTATION_SELF_CALL_BLOCKED: shell command pattern looks like an "
            "attempt to loopback-POST /api/owner/skills/<skill>/attest-review. Owner-attestation "
            "skips the expensive LLM skill review and is OWNER-ONLY — the agent must not "
            "self-attest its own skill to bypass the immune system's review. Ask the owner to "
            "attest it from the Skills UI.",
        ),
        (
            'save_settings({"ouroboros_allow_mutative_subagents":"true"})',
            "ELEVATION_BLOCKED",
            "⚠️ ELEVATION_BLOCKED: OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS is owner-controlled (it "
            "grants subagents write power against the live body). Change it by stopping the agent "
            "and editing settings.json directly, then restart — the agent must not self-enable "
            "mutative subagents.",
        ),
        (
            'save_settings({"ouroboros_post_task_evolution":"true"})',
            "ELEVATION_BLOCKED",
            "⚠️ ELEVATION_BLOCKED: the self-evolution controls (OUROBOROS_POST_TASK_EVOLUTION and "
            "OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE) are owner-controlled — they enable or "
            "steer self-modification cycles. Change them via the owner Settings UI, or stop the "
            "agent and edit settings.json directly — the agent must not self-set evolution controls.",
        ),
        (
            # A WRITE shape: since #447 A2 the family read-carve lets a pure
            # inspection (`echo`/`grep`/`rg`) name these files, so the denial is
            # pinned on a spelling that actually writes.
            "cp payload.json state/skills/alpha/review.json",
            "SKILL_STATE_WRITE_BLOCKED",
            "⚠️ SKILL_STATE_WRITE_BLOCKED: skill review, enablement, grants, and marketplace "
            "provenance are owner/review controlled state. Use skill_review, toggle_skill/the "
            "Skills UI, or the desktop launcher confirmation flow. Pure read-only inspection "
            "(grep/rg/cat/jq) of these names is allowed.",
        ),
        (
            "nohup echo state/skills/alpha/unknown.json",
            "SKILL_STATE_WRITE_BLOCKED",
            "⚠️ SKILL_STATE_WRITE_BLOCKED: detached shell processes must not target skill state "
            "directories. Use the reviewed skill lifecycle tools instead.",
        ),
        (
            "gh repo create example",
            "SAFETY_VIOLATION",
            "⚠️ SAFETY_VIOLATION: Creating/deleting GitHub repositories requires admin approval.",
        ),
        (
            "gh auth login",
            "SAFETY_VIOLATION",
            # #447 A7: the argv-positional gh resolver names the read-only
            # subcommands that stay available instead of a bare refusal.
            "⚠️ SAFETY_VIOLATION: Modifying GitHub authentication is not permitted. "
            "Read-only `gh auth status` / `gh auth token` are allowed.",
        ),
    )
    for command, code, expected in cases:
        result = check(command)
        assert (result.code, result.text) == (code, expected)

    monkeypatch.setattr(registry_module, "light_shell_repo_mutation", lambda *args, **kwargs: True)
    light_repo = check("echo ok", "light")
    assert light_repo.code == "LIGHT_MODE_BLOCKED"
    assert light_repo.text == (
        "⚠️ LIGHT_MODE_BLOCKED: runtime_mode=light refuses shell commands that mutate the "
        "Ouroboros repository. For external deliverables, run with cwd under user_files "
        "(for example /Users/<you>/Desktop), root=artifact_store, or root=task_drive. Switch "
        "to advanced/pro only for reviewed Ouroboros self-modification."
    )

    monkeypatch.setattr(registry_module, "light_shell_repo_mutation", lambda *args, **kwargs: False)
    # Tip adaptation: the write-shape seam here is non_interpreter_write_shape
    # (the reference's shell_has_write_indicator no longer exists).
    monkeypatch.setattr(registry_module, "non_interpreter_write_shape", lambda *args, **kwargs: True)
    monkeypatch.setattr(registry_module, "runtime_data_guard_targets", lambda *args, **kwargs: ["/blocked"])
    task_drive = tmp_path / "task_drive"
    artifact_dir = task_artifact_dir_path(tmp_path, "task-process-guard", create=False)
    light_data = check("echo ok", "light")
    assert light_data.code == "LIGHT_MODE_BLOCKED"
    assert light_data.text == (
        "⚠️ LIGHT_MODE_BLOCKED: runtime_mode=light blocks this command's "
        "access to runtime_data outside the permitted task roots. This task's real roots are: "
        f"artifact_store={artifact_dir}, task_drive={task_drive} — staged attachments live "
        f"under {artifact_dir / 'attachments'}. Use those absolute paths in scripts, or "
        "root=artifact_store / root=task_drive / root=user_files in file tools. Blocked paths: "
        "/blocked"
    )

    monkeypatch.setattr(
        registry_module,
        "protected_artifact_shell_block_reason",
        lambda *args, **kwargs: "⚠️ RESOURCE_POLICY_BLOCKED: protected fixture.",
    )
    resource = check(["cat", "fixture"])
    assert (resource.code, resource.text) == (
        "RESOURCE_POLICY_BLOCKED",
        "⚠️ RESOURCE_POLICY_BLOCKED: protected fixture.",
    )

    monkeypatch.setattr(registry_module, "protected_artifact_shell_block_reason", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        registry_module,
        "workspace_executor_state_write_block",
        lambda *args, **kwargs: "⚠️ WORKSPACE_EXECUTOR_STATE_WRITE_BLOCKED: fixture.",
    )
    workspace = check(["touch", "fixture"])
    assert (workspace.code, workspace.text) == (
        "WORKSPACE_BLOCKED",
        "⚠️ WORKSPACE_EXECUTOR_STATE_WRITE_BLOCKED: fixture.",
    )


def test_registry_dispatch_calls_process_owner_once_before_safety_and_handler(
    tmp_path, monkeypatch,
):
    from ouroboros import safety

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    ctx = ToolContext(repo_dir=repo, drive_root=data, task_id="task-process-dispatch")
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry.set_context(ctx)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")

    calls: list[str] = []
    denied = {"value": True}

    def guard(owner, args, runtime_mode, binding=None):
        assert owner is registry
        assert args["cmd"] == ["echo", "ok"]
        assert runtime_mode == "advanced"
        assert binding is not None
        calls.append("guard")
        if not denied["value"]:
            return None
        return ToolResult(
            status="blocked",
            code="LEGACY_BLOCKED",
            text="⚠️ TEST_PROCESS_BLOCKED",
        )

    def check_safety(*_args, **_kwargs):
        calls.append("safety")
        return True, ""

    def handler(_ctx, contract_kind, check, _resolved_binding=None, **_kwargs):
        assert contract_kind == "explicit_command"
        assert check == ["echo", "ok"]
        assert _resolved_binding is not None
        calls.append("handler")
        return "OK"

    adapter_calls: list[str] = []
    original_adapter = LegacyTextResultAdapter.from_text

    def adapt(tool_name, text):
        adapter_calls.append(text)
        return original_adapter(tool_name, text)

    monkeypatch.setattr(process_guard, "_run_shell_safety_check", guard)
    monkeypatch.setattr(LegacyTextResultAdapter, "from_text", adapt)
    monkeypatch.setattr(safety, "check_safety", check_safety)
    registry.override_handler("verify_and_record", handler)
    args = {"contract_kind": "explicit_command", "check": ["echo", "ok"]}

    assert registry.execute("verify_and_record", dict(args)) == "⚠️ TEST_PROCESS_BLOCKED"
    assert calls == ["guard"]
    assert adapter_calls == []

    calls.clear()
    denied["value"] = False
    assert registry.execute("verify_and_record", dict(args)) == "OK"
    assert calls == ["guard", "safety", "handler"]
    assert adapter_calls == ["OK"]


def test_loop_preserves_legacy_process_fields_while_consuming_native_denial(
    tmp_path, monkeypatch,
):
    import ouroboros.loop_tool_execution as execution

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    logs = tmp_path / "logs"
    repo.mkdir()
    data.mkdir()
    logs.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry.set_context(ToolContext(repo_dir=repo, drive_root=data, task_id="task-process-loop"))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr(execution, "persist_call", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        LegacyTextResultAdapter,
        "from_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("legacy adapter used")),
    )

    row = execution._execute_single_tool(
        registry,
        {
            "id": "call-process-denial",
            "function": {"name": "run_command", "arguments": '{"cmd":["sudo","true"]}'},
        },
        logs,
        "task-process-loop",
    )

    assert row["result"].startswith("⚠️ SUDO_INTERACTIVE_BLOCKED:")
    assert row["is_error"] is True
    assert row["result_meta"] == {
        "status": "blocked",
        "tool_result_status": "blocked",
        "tool_result_code": "SUDO_INTERACTIVE_BLOCKED",
        "tool_result_meta": {},
    }
