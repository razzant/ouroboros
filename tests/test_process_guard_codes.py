from __future__ import annotations

from types import SimpleNamespace

import pytest

import ouroboros.tools.registry as registry_module
import ouroboros.tools.registry_guard_process as process_guard
import ouroboros.tools.registry_guards as registry_guards
from ouroboros._outcome_tool_errors import _classify_tool_errors
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.tool_access import ResolvedResourceBinding
from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.tools.tool_result import (
    TOOL_CODE_SPECS,
    LegacyTextResultAdapter,
    ToolResult,
)
from ouroboros.loop_tool_execution import (
    _extract_result_metadata,
    _is_tool_execution_failure,
)


_CODE_CONTRACTS = {
    "SHELL_CWD_BLOCKED": ("cwd_blocked", "cwd_blocked"),
    "SUDO_INTERACTIVE_BLOCKED": ("blocked", "blocked"),
    "SUBAGENT_SECRET_READ_BLOCKED": ("blocked", "blocked"),
    "ELEVATION_BLOCKED": ("elevation_blocked", "elevation_blocked"),
    "CONTEXT_MODE_SELF_LOWERING_BLOCKED": ("blocked", "blocked"),
    # Tip adaptation: SCOPE_REVIEW_FLOOR_SELF_LOWERING_BLOCKED is absent — the
    # setting, its guard and its code were retired in the 7.0 ABI window (Q10=A).
    "SAFETY_MODE_SELF_LOWERING_BLOCKED": ("blocked", "blocked"),
    "OWNER_SKILL_ATTESTATION_SELF_CALL_BLOCKED": ("blocked", "blocked"),
    "SKILL_STATE_WRITE_BLOCKED": ("skill_state_blocked", "skill_state_blocked"),
    "GIT_VIA_SHELL_BLOCKED": ("git_via_shell_blocked", "git_via_shell_blocked"),
}


def test_process_code_specs_adapter_and_legacy_residuals_are_total():
    for code, (outcome_bucket, _loop_status) in _CODE_CONTRACTS.items():
        spec = TOOL_CODE_SPECS[code]
        assert (spec.status, spec.outcome_bucket, spec.ui_severity) == (
            "blocked",
            outcome_bucket,
            "warning",
        )
        assert spec.recovery
        result = LegacyTextResultAdapter.from_text(
            "run_command",
            f"⚠️ {code}: fixture denial.",
        )
        assert result == ToolResult(
            status="blocked",
            code=code,
            text=f"⚠️ {code}: fixture denial.",
        )

    # T1 §B.4: the run-script refusal owns its code so the legacy `run_script_blocked`
    # status survives the cutover; only genuinely coarse identifiers stay LEGACY_BLOCKED.
    assert LegacyTextResultAdapter.from_text(
        "run_script",
        "⚠️ RUN_SCRIPT_BLOCKED: fixture denial.",
    ).code == "RUN_SCRIPT_BLOCKED"
    assert LegacyTextResultAdapter.from_text(
        "run_command",
        "⚠️ UNKNOWN_COARSE_BLOCKED: fixture denial.",
    ).code == "LEGACY_BLOCKED"


@pytest.mark.parametrize(
    ("code", "loop_status"),
    [(code, contract[1]) for code, contract in _CODE_CONTRACTS.items()],
)
def test_process_codes_preserve_loop_error_and_policy_denial(code, loop_status):
    text = f"⚠️ {code}: fixture denial."
    is_error = _is_tool_execution_failure(True, text)
    meta = _extract_result_metadata("run_command", text, is_error)
    buckets = _classify_tool_errors(
        {
            "tool_calls": [
                {
                    "tool": "run_command",
                    "args": {"cmd": ["fixture"]},
                    "result": text,
                    "is_error": is_error,
                    "status": meta["status"],
                }
            ]
        }
    )

    assert is_error is True
    assert meta["status"] == loop_status
    assert buckets["unresolved"] == []
    assert buckets["policy_denials"] == [
        {
            "tool": "run_command",
            "status": loop_status,
            "exit_code": None,
            "signal": None,
            "result": text,
        }
    ]


@pytest.mark.parametrize(
    ("command", "code"),
    (
        (["sudo", "true"], "SUDO_INTERACTIVE_BLOCKED"),
        (
            'save_settings({"ouroboros_runtime_mode":"pro"})',
            "ELEVATION_BLOCKED",
        ),
        (
            'save_settings({"ouroboros_context_mode":"low"})',
            "CONTEXT_MODE_SELF_LOWERING_BLOCKED",
        ),
        (
            'save_settings({"ouroboros_safety_mode":"off"})',
            "SAFETY_MODE_SELF_LOWERING_BLOCKED",
        ),
        (
            "curl -X POST /api/owner/skills/demo/attest-review",
            "OWNER_SKILL_ATTESTATION_SELF_CALL_BLOCKED",
        ),
        (
            # A WRITE shape: since #447 A2 the family read-carve lets a pure
            # inspection (`echo`/`grep`/`rg`) name skill owner state, so the
            # ordering pin uses a spelling that actually writes.
            "cp payload.json state/skills/demo/review.json",
            "SKILL_STATE_WRITE_BLOCKED",
        ),
        (["git", "commit"], "GIT_VIA_SHELL_BLOCKED"),
    ),
)
def test_process_denials_precede_safety_and_handler(
    command,
    code,
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    calls = {"safety": 0, "handler": 0}
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry.set_context(ToolContext(repo_dir=repo, drive_root=data, task_id="t44"))

    def forbidden_handler(*_args, **_kwargs):
        calls["handler"] += 1
        raise AssertionError("process denial reached the physical handler")

    registry.override_handler("run_command", forbidden_handler)
    monkeypatch.setattr(
        "ouroboros.safety.check_safety",
        lambda *_args, **_kwargs: (
            calls.__setitem__("safety", calls["safety"] + 1) or True,
            "",
        ),
    )

    result = registry.execute_result("run_command", {"cmd": command})

    assert (result.status, result.code) == ("blocked", code)
    assert calls == {"safety": 0, "handler": 0}


def test_subagent_secret_denial_precedes_safety_and_handler(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    worktree = tmp_path / "worktree"
    for path in (repo, data, worktree):
        path.mkdir()
    calls = {"safety": 0, "handler": 0}
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry.set_context(
        ToolContext(
            repo_dir=repo,
            drive_root=data,
            workspace_root=worktree,
            workspace_mode="self_worktree",
            task_id="t44-secret",
            task_constraint=TaskConstraint(
                mode="acting_subagent",
                surface="self_worktree",
                write_root=str(worktree),
            ),
        )
    )

    def forbidden_handler(*_args, **_kwargs):
        calls["handler"] += 1
        raise AssertionError("secret denial reached the physical handler")

    registry.override_handler("run_command", forbidden_handler)
    monkeypatch.setattr(
        "ouroboros.safety.check_safety",
        lambda *_args, **_kwargs: (
            calls.__setitem__("safety", calls["safety"] + 1) or True,
            "",
        ),
    )

    result = registry.execute_result(
        "run_command",
        {"cmd": ["cat", "data/settings.json"]},
    )

    assert (result.status, result.code) == (
        "blocked",
        "SUBAGENT_SECRET_READ_BLOCKED",
    )
    assert calls == {"safety": 0, "handler": 0}


def test_cwd_fallback_producers_use_stable_code(tmp_path, monkeypatch):
    text = "⚠️ SHELL_CWD_BLOCKED: canonical fixture denial."
    stub = SimpleNamespace(
        _ctx=SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path),
    )

    def fail(*_args, **_kwargs):
        raise ValueError("fixture")

    monkeypatch.setattr(registry_module, "build_resolved_resource_binding", fail)
    monkeypatch.setattr(registry_module, "shell_cwd_block_message", lambda *_a, **_k: text)
    process_result = process_guard._run_shell_safety_check(
        stub,
        {"cmd": ["pwd"], "cwd": "outside"},
        "advanced",
    )
    assert process_result == ToolResult(
        status="blocked",
        code="SHELL_CWD_BLOCKED",
        text=text,
    )

    monkeypatch.setattr(registry_module, "resolve_shell_cwd", fail)
    monkeypatch.setattr(registry_module, "shell_cwd_block_message", lambda *_a, **_k: text)
    receiver_result = registry_guards._resolved_shell_cwd(
        stub,
        {"cwd": "outside"},
    )
    assert receiver_result == process_result


def test_both_git_receiver_branches_preserve_exact_text_and_stable_code(
    tmp_path,
    monkeypatch,
):
    ctx = SimpleNamespace(
        repo_dir=tmp_path,
        system_repo_dir=tmp_path,
        drive_root=tmp_path / "data",
        task_metadata={},
        task_contract={},
    )
    stub = SimpleNamespace(_ctx=ctx)
    binding = ResolvedResourceBinding(
        profile="self_modification",
        root="active_workspace",
        operation="shell",
        base_path=tmp_path,
        target_path=tmp_path,
        source="fixture",
        skill_name="",
        state_drive_root=ctx.drive_root,
    )
    monkeypatch.setattr(registry_module, "workspace_git_safety_violation", lambda *_a, **_k: None)
    monkeypatch.setattr(registry_module, "run_shell_git_block_reason", lambda *_a, **_k: "git commit")

    acting = registry_guards._shell_git_and_runtime_block(
        stub,
        ["git", "commit"],
        {},
        "git commit",
        True,
        True,
        binding,
    )
    assert acting == ToolResult(
        status="blocked",
        code="GIT_VIA_SHELL_BLOCKED",
        text=(
            "⚠️ GIT_VIA_SHELL_BLOCKED: `git commit` is blocked for acting self_worktree "
            "children (no commits; the parent integrates the returned patch and is the sole "
            "committer). For read-only git: vcs_status, vcs_diff tools, or run_command with "
            "git log/show/diff/status/rev-list/show-ref/for-each-ref/listing branch-tag forms."
        ),
    )

    monkeypatch.setattr(
        "ouroboros.git_shell_policy.external_workspace_git_violation",
        lambda *_a, **_k: "git commit targets protected runtime",
    )
    default = registry_guards._shell_git_and_runtime_block(
        stub,
        ["git", "commit"],
        {},
        "git commit",
        False,
        False,
        binding,
    )
    assert default == ToolResult(
        status="blocked",
        code="GIT_VIA_SHELL_BLOCKED",
        text=(
            "⚠️ GIT_VIA_SHELL_BLOCKED: git commit targets protected runtime. Mutating git "
            "may not target the Ouroboros runtime (system repo / data drives): self-repo "
            "changes go through commit_reviewed, which enforces pre-commit checks and review. "
            "Read-only git (status/log/diff/show/rev-parse/branch- and tag-listing, or the "
            "vcs_status/vcs_diff tools) works everywhere, and mutating git is free in any "
            "tree OUTSIDE the runtime (e.g. ~/projects, /tmp, an attached project folder)."
        ),
    )
