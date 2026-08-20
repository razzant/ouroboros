"""The skill-repair heal context: the payload it may touch, and everything it may not.

Split out of ``tests/test_skill_exec.py`` by theme: the enable paths blocked directly and
indirectly, the payload tools and review it does allow for the selected skill, and every
escape it refuses — marketplace sidecars, out-of-scope data, symlink escapes, the wrong
source root, payload-root and traversal markers, and self-authored marker writes.
"""

from __future__ import annotations

import pathlib

import pytest

from ouroboros.tools import skill_exec as skill_exec_mod
from ouroboros.tools.registry import ToolRegistry

from tests._skill_exec_shared import (
    _admit_repair,
    _build_skill,
    _make_ctx,
    _mark_reviewed,
    _set_skill_repair,
)
from tests._skill_exec_shared import (  # noqa: F401  (autouse fixture applies on import)
    _clean_extension_runtime,
)


def test_toggle_skill_blocked_in_heal_context(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    skill_dir = _build_skill(skills_root, "alpha")
    ctx = _make_ctx(tmp_path)
    _set_skill_repair(ctx, "alpha", "skills/external/alpha")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    _mark_reviewed(ctx.drive_root, skill_dir, "alpha")
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry._ctx = ctx

    result = registry.execute("toggle_skill", {"skill": "alpha", "enabled": True})

    assert "HEAL_MODE_BLOCKED" in result or "SKILL_REDIRECT_BLOCKED" in result


@pytest.mark.parametrize("tool_name,args", [
    ("run_command", {"cmd": ["python", "-c", "print('x')"]}),
    ("browse_page", {"url": "http://127.0.0.1"}),
    ("browser_action", {"action": "evaluate", "value": "fetch('/api/skills/x/toggle')"}),
    ("schedule_subagent", {
        "objective": "enable skill",
        "expected_output": "skill enabled",
    }),
    ("skill_exec", {"skill": "alpha", "script": "hello.py"}),
    ("write_file", {"root": "skill_payload", "bucket": "external", "skill_name": "alpha", "path": ".self_authored.json", "content": "{}"}),
])
def test_heal_context_blocks_indirect_enable_paths(tool_name, args, tmp_path):
    ctx = _make_ctx(tmp_path)
    _set_skill_repair(ctx, "alpha", "skills/external/alpha")
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry._ctx = ctx

    result = registry.execute(tool_name, args)

    assert "HEAL_MODE_BLOCKED" in result or "SKILL_REDIRECT_BLOCKED" in result


def test_heal_context_allows_payload_tools_and_review(tmp_path):
    ctx = _make_ctx(tmp_path)
    _set_skill_repair(ctx, "alpha", "skills/external/alpha")
    _build_skill(ctx.drive_root / "skills" / "external", "alpha")
    _admit_repair(ctx, "alpha", "skills/external/alpha")
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry._ctx = ctx

    result = registry.execute(
        "write_file",
        {
            "root": "skill_payload",
            "bucket": "external",
            "skill_name": "alpha",
            "path": "notes.txt",
            "content": "x",
        },
    )

    assert "HEAL_MODE_BLOCKED" not in result
    assert "OK" in result


def test_heal_context_allows_ouroboroshub_payload_tools(tmp_path):
    ctx = _make_ctx(tmp_path)
    _set_skill_repair(ctx, "nanobanana", "skills/ouroboroshub/nanobanana")
    _build_skill(ctx.drive_root / "skills" / "ouroboroshub", "nanobanana")
    _admit_repair(ctx, "nanobanana", "skills/ouroboroshub/nanobanana")
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry._ctx = ctx

    result = registry.execute(
        "write_file",
        {
            "root": "skill_payload",
            "bucket": "ouroboroshub",
            "skill_name": "nanobanana",
            "path": "plugin.py",
            "content": "# fixed",
        },
    )

    assert "HEAL_MODE_BLOCKED" not in result
    assert "OK" in result


@pytest.mark.parametrize("sidecar", [".ouroboroshub.json", ".clawhub.json"])
def test_heal_context_blocks_marketplace_sidecar_writes(sidecar, tmp_path):
    ctx = _make_ctx(tmp_path)
    _set_skill_repair(ctx, "nanobanana", "skills/ouroboroshub/nanobanana")
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry._ctx = ctx

    result = registry.execute(
        "write_file",
        {
            "root": "skill_payload",
            "bucket": "ouroboroshub",
            "skill_name": "nanobanana",
            "path": sidecar,
            "content": "{}",
        },
    )

    assert "HEAL_MODE_BLOCKED" in result or "SKILL_REDIRECT_BLOCKED" in result
    assert "provenance sidecars" in result


@pytest.mark.parametrize("tool_name,args", [
    ("write_file", {"root": "skill_payload", "bucket": "external", "skill_name": "beta", "path": "notes.txt", "content": "x"}),
    ("read_file", {"root": "skill_payload", "bucket": "external", "skill_name": "beta", "path": "SKILL.md"}),
    ("list_files", {"root": "skill_payload", "bucket": "external", "skill_name": "beta", "path": "."}),
    ("skill_review", {"skill": "beta"}),
    ("skill_preflight", {"skill": "beta"}),
])
def test_heal_context_blocks_out_of_scope_data_access(tool_name, args, tmp_path):
    ctx = _make_ctx(tmp_path)
    _set_skill_repair(ctx, "alpha", "skills/external/alpha")
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry._ctx = ctx

    result = registry.execute(tool_name, args)

    assert "HEAL_MODE_BLOCKED" in result or "SKILL_REDIRECT_BLOCKED" in result


def test_heal_context_blocks_symlink_escape_from_selected_skill(tmp_path):
    ctx = _make_ctx(tmp_path)
    _set_skill_repair(ctx, "alpha", "skills/external/alpha")
    skill_root = pathlib.Path(ctx.drive_root) / "skills" / "external" / "alpha"
    memory_root = pathlib.Path(ctx.drive_root) / "memory"
    skill_root.mkdir(parents=True)
    memory_root.mkdir()
    (memory_root / "identity.md").write_text("secret-ish", encoding="utf-8")
    try:
        (skill_root / "escape").symlink_to(memory_root / "identity.md")
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unavailable on this filesystem")
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry._ctx = ctx

    result = registry.execute(
        "read_file",
        {"root": "skill_payload", "bucket": "external", "skill_name": "alpha", "path": "escape"},
    )

    assert "HEAL_MODE_BLOCKED" in result or "SKILL_REDIRECT_BLOCKED" in result


def test_heal_context_blocks_wrong_source_root(tmp_path):
    ctx = _make_ctx(tmp_path)
    _set_skill_repair(ctx, "alpha", "skills/clawhub/alpha")
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry._ctx = ctx

    result = registry.execute(
        "write_file",
        {
            "root": "skill_payload",
            "bucket": "external",
            "skill_name": "alpha",
            "path": "notes.txt",
            "content": "x",
        },
    )

    assert "HEAL_MODE_BLOCKED" in result or "SKILL_REDIRECT_BLOCKED" in result


def test_heal_context_blocks_native_payload_root_marker(tmp_path):
    ctx = _make_ctx(tmp_path)
    _set_skill_repair(ctx, "alpha", "skills/native/alpha")
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry._ctx = ctx

    result = registry.execute(
        "read_file",
        {"root": "skill_payload", "bucket": "native", "skill_name": "alpha", "path": "SKILL.md"},
    )

    assert "HEAL_MODE_BLOCKED" in result


def test_heal_context_rejects_traversal_skill_marker(tmp_path):
    ctx = _make_ctx(tmp_path)
    _set_skill_repair(ctx, "../..", "../../")
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry._ctx = ctx

    result = registry.execute(
        "read_file",
        {"root": "skill_payload", "bucket": "external", "skill_name": "alpha", "path": "settings.json"},
    )

    assert "HEAL_MODE_BLOCKED" in result


def test_heal_context_rejects_traversal_payload_root_marker(tmp_path):
    ctx = _make_ctx(tmp_path)
    _set_skill_repair(ctx, "alpha", "skills/external/alpha/../../memory")
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry._ctx = ctx

    result = registry.execute(
        "read_file",
        {"root": "skill_payload", "bucket": "external", "skill_name": "alpha", "path": "memory/identity.md"},
    )

    assert "HEAL_MODE_BLOCKED" in result


def test_heal_context_blocks_self_authored_marker_write(tmp_path):
    ctx = _make_ctx(tmp_path)
    payload = ctx.drive_root / "skills" / "external" / "alpha"
    payload.mkdir(parents=True)
    _set_skill_repair(ctx, "alpha", "skills/external/alpha")
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry._ctx = ctx

    result = registry.execute(
        "write_file",
        {
            "root": "skill_payload",
            "bucket": "external",
            "skill_name": "alpha",
            "path": ".self_authored.json",
            "content": '{"origin":"self_authored"}',
        },
    )

    assert "HEAL_MODE_BLOCKED" in result


def test_heal_review_does_not_reconcile_live_extension(tmp_path, monkeypatch):
    import types

    ctx = _make_ctx(tmp_path)
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    _build_skill(ctx.drive_root / "skills" / "external", "alpha")
    _set_skill_repair(ctx, "alpha", "skills/external/alpha")
    calls = []

    monkeypatch.setattr(
        skill_exec_mod,
        "_review_skill_impl",
        lambda _ctx, skill_name: types.SimpleNamespace(
            skill_name=skill_name,
            status="pass",
            content_hash="hash",
            reviewer_models=[],
            findings=[],
            error="",
        ),
    )

    from ouroboros import extension_loader
    monkeypatch.setattr(extension_loader, "reconcile_extension", lambda *a, **kw: calls.append(a) or {"action": "extension_loaded"})

    # The review_skill tool result is now rendered-markdown only (the raw JSON
    # payload duplicate was removed in C4); assert on the lifecycle payload the
    # tool renders from instead.
    from ouroboros.skill_review_runner import run_skill_review_lifecycle_blocking
    result = run_skill_review_lifecycle_blocking(
        ctx, "alpha", source="tool",
        review_impl=lambda rc, rn: skill_exec_mod._review_skill_impl(rc, rn),
    )

    assert calls == []
    assert result["extension_reason"] == "heal_review_only"
