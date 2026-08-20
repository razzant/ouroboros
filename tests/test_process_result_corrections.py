from __future__ import annotations

from types import SimpleNamespace

import ouroboros.tools.registry_guard_process as process_guard
from ouroboros.tools.registry import ToolRegistry
from ouroboros.tools.tool_result import LegacyTextResultAdapter, ToolResult


def test_legacy_string_postchecks_stay_legacy_until_central_adapter(
    tmp_path, monkeypatch,
):
    import time

    stub = SimpleNamespace(_ctx=SimpleNamespace())
    original_adapter = LegacyTextResultAdapter.from_text
    state: dict[str, object] = {}
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        LegacyTextResultAdapter,
        "from_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("postchecks must not adapt legacy text"),
        ),
    )
    monkeypatch.setattr(
        process_guard,
        "_restore_owner_files",
        lambda *_args, **_kwargs: bool(state["owner"]),
    )
    monkeypatch.setattr(
        process_guard,
        "_light_repo_snapshot",
        lambda _repo_dir: state["light_after"],
    )
    monkeypatch.setattr(
        process_guard,
        "_git_ref_snapshot",
        lambda _repo_dir: state["refs_after"],
    )
    monkeypatch.setattr(process_guard, "system_repo_dir_for", lambda _ctx: tmp_path)
    monkeypatch.setattr(process_guard, "active_repo_dir_for", lambda _ctx: tmp_path)

    cases = (
        (
            {"owner": True, "light_after": None, "refs_after": None},
            None,
            None,
            "handler output\n\n⚠️ OWNER_STATE_RESTORED: run_command attempted to "
            "change owner-only settings or skill trust state; protected files were restored.",
            ("ok", "OK"),
        ),
        (
            {
                "owner": False,
                "light_after": {"digest": "after", "paths": ["changed.py"]},
                "refs_after": None,
            },
            {"digest": "before", "paths": []},
            None,
            "⚠️ LIGHT_MODE_REPO_WRITE_BLOCKED: runtime_mode=light detected a mutation "
            "of the Ouroboros repository after run_command. The command result is blocked "
            "and no automatic rollback was attempted to avoid overwriting concurrent human "
            "edits. Affected/dirty paths: changed.py. Switch to advanced/pro for repo writes."
            "\n\nOriginal command output:\nhandler output",
            ("blocked", "LIGHT_MODE_REPO_WRITE_BLOCKED"),
        ),
        (
            {
                "owner": False,
                "light_after": None,
                "refs_after": {"head": "same", "digest": "after"},
            },
            None,
            {"head": "same", "digest": "before"},
            "⚠️ WORKSPACE_GIT_REF_CHANGED: run_command changed git HEAD or refs inside "
            "the external workspace. External workspace runs must leave changes as files/"
            "patch artifacts, not commits/tags/resets.\n\nOriginal command output:\n"
            "handler output",
            ("blocked", "WORKSPACE_GIT_REF_CHANGED"),
        ),
    )
    for case_state, light_before, refs_before, expected_text, expected_mapping in cases:
        state.update(case_state)
        result = process_guard._run_shell_post_checks(
            stub,
            "handler output",
            owner_snapshot={},
            state_drive_root=tmp_path,
            light_repo_before=light_before,
            workspace_refs_before=refs_before,
        )

        assert type(result) is str
        assert result == expected_text
        central = original_adapter("run_command", result)
        assert (central.status, central.code, central.text, dict(central.meta)) == (
            *expected_mapping,
            expected_text,
            {},
        )


def test_direct_typed_override_survives_postchecks_and_composer(
    tmp_path, monkeypatch,
):
    import time

    from ouroboros import safety

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        safety,
        "check_safety",
        lambda *_args, **_kwargs: (True, "⚠️ SAFETY_WARNING: reviewed"),
    )
    monkeypatch.setattr(
        process_guard,
        "_run_shell_safety_check",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        process_guard,
        "_snapshot_owner_files",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        process_guard,
        "_restore_owner_files",
        lambda *_args, **_kwargs: False,
    )
    registry.override_handler(
        "run_command",
        lambda _ctx, cmd, _resolved_binding=None, **_kwargs: ToolResult(
            status="error",
            code="SHELL_EXIT_ERROR",
            text="custom output",
            meta={"exit_code": 93},
        ),
    )

    direct_typed = registry.execute_result(
        "run_command", {"cmd": ["echo", "ok"]},
    )

    assert (
        direct_typed.status,
        direct_typed.code,
        direct_typed.text,
        dict(direct_typed.meta),
    ) == (
        "error",
        "SHELL_EXIT_ERROR",
        "⚠️ SAFETY_WARNING: reviewed\n\n---\ncustom output",
        {"exit_code": 93, "safety_warning": True},
    )
    assert not hasattr(registry._ctx, "_active_builtin_tool_result")
