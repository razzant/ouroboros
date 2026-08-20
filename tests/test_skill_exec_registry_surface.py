"""What the skill tools declare to the registry, and the guards that come with them.

Split out of ``tests/test_skill_exec.py`` by theme: the review timeout kept separate from
the execution one, the policy entries every skill tool needs, the frozen-module list, the
self-authored marker writes run_shell blocks, the data-plane listing that needs no external
repo, the phase-3 runtime allowlist with its Windows fallback, and the bounded hard-timeout
ceiling.
"""

from __future__ import annotations

from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_enabled, save_review_state
from ouroboros.tools import skill_exec as skill_exec_mod
from ouroboros.tools.registry import ToolRegistry

from tests._skill_exec_shared import (
    _build_skill,
    _make_ctx,
)
from tests._skill_exec_shared import (  # noqa: F401  (autouse fixture applies on import)
    _clean_extension_runtime,
)


def test_review_skill_uses_long_timeout_separate_from_skill_exec():
    entries = {entry.name: entry for entry in skill_exec_mod.get_tools()}

    assert entries["skill_exec"].timeout_sec == skill_exec_mod._HARD_TIMEOUT_CEILING_SEC
    assert entries["skill_review"].timeout_sec >= 1800
    assert entries["skill_review"].timeout_sec > entries["skill_exec"].timeout_sec


def test_run_shell_blocks_self_authored_marker_writes(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    ctx = _make_ctx(tmp_path)
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry._ctx = ctx

    result = registry.execute(
        "run_command",
        {"cmd": ["sh", "-c", "printf '{}' > /tmp/x/.self_authored.json"]},
    )

    assert "SAFETY_VIOLATION" in result
    assert ".self_authored.json" in result


def test_skill_exec_tools_have_policy_entries():
    """Every new tool must carry an explicit TOOL_POLICY entry."""
    from ouroboros.safety import TOOL_POLICY, POLICY_CHECK, POLICY_SKIP

    assert TOOL_POLICY["list_skills"] == POLICY_SKIP
    assert TOOL_POLICY["skill_review"] == POLICY_SKIP
    assert TOOL_POLICY["toggle_skill"] == POLICY_SKIP
    assert TOOL_POLICY["skill_preflight"] == POLICY_SKIP
    assert TOOL_POLICY["skill_exec"] == POLICY_CHECK


def test_skill_exec_in_frozen_modules(tmp_path):
    from ouroboros.tools.registry import ToolRegistry

    assert "skill_exec" in ToolRegistry(tmp_path, tmp_path)._FROZEN_TOOL_MODULES


def test_list_skills_uses_data_plane_without_external_repo(tmp_path, monkeypatch):
    monkeypatch.delenv("OUROBOROS_SKILLS_REPO_PATH", raising=False)
    ctx = _make_ctx(tmp_path)
    skill_dir = _build_skill(ctx.drive_root / "skills" / "external", "alpha")
    save_enabled(ctx.drive_root, "alpha", True)
    save_review_state(ctx.drive_root, "alpha", SkillReviewState(
        status="clean",
        content_hash=compute_content_hash(skill_dir),
        findings=[],
    ))
    result = skill_exec_mod._handle_list_skills(ctx)
    assert "alpha" in result
    assert "SKILLS_UNAVAILABLE" not in result


def test_runtime_allowlist_covers_phase3_runtimes():
    allowed = set(skill_exec_mod._ALLOWED_RUNTIMES)
    assert {"python", "python3", "bash", "node", "deno", "ruby", "go"} <= allowed


def test_python3_runtime_falls_back_to_python_for_windows():
    """Phase 3 round 6 regression: Windows installs often only ship
    ``python.exe`` (no ``python3.exe``). ``_ALLOWED_RUNTIMES["python3"]``
    must include ``python`` as a fallback so reviewed skills declaring
    ``runtime: python3`` still resolve to a real binary there."""
    assert skill_exec_mod._ALLOWED_RUNTIMES["python3"] == ("python3", "python")


def test_hard_timeout_ceiling_is_bounded():
    assert 60 <= skill_exec_mod._HARD_TIMEOUT_CEILING_SEC <= 900
