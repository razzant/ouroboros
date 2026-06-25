"""Phase 3 regression tests for ``ouroboros.tools.skill_exec``.

Covers tool registration, runtime-mode gating, review-status gating,
path-confinement guards, and actual subprocess execution against a
trivial python3 script. No network, no real LLM calls — the
``review_skill`` tool is exercised indirectly via state fixtures.
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import shutil
import sys
import threading
from unittest.mock import patch

import pytest

from ouroboros.skill_loader import (
    SkillPayloadUnreadable,
    SkillReviewState,
    compute_content_hash,
    save_enabled,
    save_review_state,
)
from ouroboros.tools import skill_exec as skill_exec_mod
from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.contracts.task_constraint import TaskConstraint


from tests._shared import clean_extension_runtime_state


@pytest.fixture(autouse=True)
def _clean_extension_runtime():
    clean_extension_runtime_state()
    yield
    clean_extension_runtime_state()


def _valid_script_manifest(
    name: str = "weather",
    *,
    runtime: str = "python3",
    timeout_sec: int = 30,
    scripts_only: bool = True,
) -> str:
    return (
        "---\n"
        f"name: {name}\n"
        "description: Simple greeter.\n"
        "version: 0.1.0\n"
        f"type: {'script' if scripts_only else 'extension'}\n"
        f"runtime: {runtime}\n"
        f"timeout_sec: {timeout_sec}\n"
        "scripts:\n"
        "  - name: hello.py\n"
        "    description: Print hello.\n"
        "---\n"
        "# body\n"
    )


def _build_skill(
    skills_root: pathlib.Path,
    name: str,
    *,
    script_body: str = "print('hello from skill')\n",
    manifest: str | None = None,
) -> pathlib.Path:
    skill_dir = skills_root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(manifest or _valid_script_manifest(name), encoding="utf-8")
    scripts = skill_dir / "scripts"
    scripts.mkdir(exist_ok=True)
    (scripts / "hello.py").write_text(script_body, encoding="utf-8")
    return skill_dir


def _make_ctx(tmp_path: pathlib.Path) -> ToolContext:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    return ToolContext(repo_dir=repo_dir, drive_root=drive_root)


def _set_skill_repair(ctx: ToolContext, name: str = "alpha", payload_root: str = "skills/external/alpha") -> None:
    ctx.task_constraint = TaskConstraint(mode="skill_repair", skill_name=name, payload_root=payload_root, allow_enable=False, allow_review=True)


def _mark_reviewed_and_enabled(drive_root: pathlib.Path, skill_dir: pathlib.Path, name: str):
    content_hash = compute_content_hash(skill_dir)
    save_enabled(drive_root, name, True)
    save_review_state(
        drive_root,
        name,
        SkillReviewState(status="pass", content_hash=content_hash),
    )


def _mark_reviewed(drive_root: pathlib.Path, skill_dir: pathlib.Path, name: str):
    save_review_state(
        drive_root,
        name,
        SkillReviewState(status="pass", content_hash=compute_content_hash(skill_dir)),
    )


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def test_review_skill_uses_long_timeout_separate_from_skill_exec():
    entries = {entry.name: entry for entry in skill_exec_mod.get_tools()}

    assert entries["skill_exec"].timeout_sec == skill_exec_mod._HARD_TIMEOUT_CEILING_SEC
    assert entries["skill_review"].timeout_sec >= 1800
    assert entries["skill_review"].timeout_sec > entries["skill_exec"].timeout_sec


def test_skill_preflight_success_and_no_pycache(tmp_path, monkeypatch):
    ctx = _make_ctx(tmp_path)
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    skill_dir = _build_skill(skills_root, "alpha", script_body="print('ok')\n")

    from ouroboros.tools.skill_preflight import _handle_skill_preflight

    result = json.loads(_handle_skill_preflight(ctx, skill="alpha"))

    assert result["ok"] is True
    assert result["files_checked"] >= 1
    assert result["files_failed"] == 0
    assert not (skill_dir / "scripts" / "__pycache__").exists()


def test_skill_preflight_reports_python_syntax_error(tmp_path, monkeypatch):
    ctx = _make_ctx(tmp_path)
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    _build_skill(skills_root, "alpha", script_body="def broken(:\n")

    from ouroboros.tools.skill_preflight import _handle_skill_preflight

    result = json.loads(_handle_skill_preflight(ctx, skill="alpha"))

    assert result["ok"] is False
    assert result["files_failed"] == 1
    assert "SyntaxError" in result["files"][0]["stderr"]


def test_skill_preflight_file_limit_omission_is_degraded_not_blocked(tmp_path, monkeypatch):
    # A file count beyond the syntax-check headroom is a DEGRADED note, NOT a hard block:
    # the skill-review pass now reads every file under a pack-level token budget (chunked
    # when oversized), so preflight must not re-introduce an arbitrary file-count gate.
    ctx = _make_ctx(tmp_path)
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    skill_dir = _build_skill(skills_root, "alpha")
    scripts = skill_dir / "scripts"
    from ouroboros.tools import skill_preflight as sp

    for idx in range(sp._PREFLIGHT_HARD_FILE_LIMIT + 2):
        (scripts / f"extra_{idx}.py").write_text("print('ok')\n", encoding="utf-8")

    result = json.loads(sp._handle_skill_preflight(ctx, skill="alpha"))

    assert result["ok"] is True  # proceeds to the authoritative token-budgeted review
    assert result["omitted_count"] > 0
    assert result.get("degraded") is True
    assert "token budget" in result.get("degraded_note", "")


def test_skill_preflight_missing_validator_runtime_is_tolerated(tmp_path, monkeypatch):
    # A missing external runtime (e.g. node not installed, or a Homebrew node
    # code-signing-killed by macOS) is an environment gap, not a syntax verdict.
    # Preflight must skip it rather than block; tri-model review stays authoritative.
    ctx = _make_ctx(tmp_path)
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    skill_dir = _build_skill(skills_root, "alpha")
    (skill_dir / "scripts" / "check.js").write_text("console.log('ok')\n", encoding="utf-8")

    from ouroboros.tools import skill_preflight as sp
    monkeypatch.setattr(sp, "_resolve_runtime", lambda runtime: None if runtime == "node" else "/bin/echo")

    result = json.loads(sp._handle_skill_preflight(ctx, skill="alpha", paths=["scripts/check.js"]))

    assert result["ok"] is True
    assert result.get("degraded") is True
    js = next(f for f in result["files"] if f["path"].endswith("check.js"))
    assert js.get("skipped") is True
    assert js.get("skip_reason") == "runtime_unavailable"


def test_skill_preflight_validates_literal_widget_schema(tmp_path, monkeypatch):
    ctx = _make_ctx(tmp_path)
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    manifest = (
        "---\n"
        "name: alpha\n"
        "description: widget test\n"
        "version: 0.1.0\n"
        "type: extension\n"
        "entry: plugin.py\n"
        "permissions: [widget, route]\n"
        "---\n"
        "body\n"
    )
    skill_dir = skills_root / "alpha"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(manifest, encoding="utf-8")
    (skill_dir / "plugin.py").write_text(
        "_UI_RENDER = {\n"
        "    'kind': 'declarative',\n"
        "    'schema_version': 1,\n"
        "    'components': [\n"
        "        {'type': 'form', 'action_route': 'generate', 'fields': [{'name': 'prompt'}]},\n"
        "    ],\n"
        "}\n"
        "def register(api):\n"
        "    api.register_ui_tab('main', 'Main', render=_UI_RENDER)\n",
        encoding="utf-8",
    )

    from ouroboros.tools import skill_preflight as sp

    result = json.loads(sp._handle_skill_preflight(ctx, skill="alpha"))

    assert result["ok"] is False
    assert any("requires route or api_route" in item["detail"] for item in result["widgets"])


def test_skill_preflight_reports_missing_pluginapi_permissions(tmp_path, monkeypatch):
    ctx = _make_ctx(tmp_path)
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    manifest = (
        "---\n"
        "name: alpha\n"
        "description: permissions test\n"
        "version: 0.1.0\n"
        "type: extension\n"
        "entry: plugin.py\n"
        "permissions: [net]\n"
        "env_from_settings: [OPENROUTER_API_KEY]\n"
        "---\n"
        "body\n"
    )
    skill_dir = skills_root / "alpha"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(manifest, encoding="utf-8")
    (skill_dir / "plugin.py").write_text(
        "def register(api):\n"
        "    api.register_route('status', lambda request: {})\n"
        "    api.register_ui_tab('main', 'Main', render={'kind':'declarative','schema_version':1,'components': []})\n"
        "    api.get_settings(['OPENROUTER_API_KEY'])\n",
        encoding="utf-8",
    )

    from ouroboros.tools import skill_preflight as sp

    result = json.loads(sp._handle_skill_preflight(ctx, skill="alpha"))

    assert result["ok"] is False
    missing = {item["permission"] for item in result["permissions"] if not item["ok"]}
    assert {"route", "widget", "read_settings"} <= missing


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


def test_run_shell_restores_obfuscated_self_authored_state_marker(tmp_path):
    ctx = _make_ctx(tmp_path)
    marker = ctx.drive_root / "state" / "skills" / "alpha" / "self_authored.json"
    marker.parent.mkdir(parents=True)
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry._ctx = ctx

    before = registry._snapshot_owner_files()
    marker.write_text('{"origin":"self_authored"}', encoding="utf-8")

    restored = registry._restore_owner_files(before)

    assert restored is True
    assert not marker.exists()


def test_claude_code_edit_resolves_skill_cwd_under_drive_root(tmp_path, monkeypatch):
    from ouroboros.tools import shell as shell_mod
    import types

    ctx = _make_ctx(tmp_path)
    skill_dir = ctx.drive_root / "skills" / "external" / "alpha"
    skill_dir.mkdir(parents=True)
    seen = {}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr("ouroboros.tools.git._acquire_git_lock", lambda _ctx: object())
    monkeypatch.setattr("ouroboros.tools.git._release_git_lock", lambda _lock: None)
    fake_module = types.ModuleType("ouroboros.gateways.claude_code")
    fake_module.DEFAULT_CLAUDE_CODE_MAX_TURNS = 3
    fake_module.resolve_claude_code_model = lambda: "claude-test"

    class _Result:
        success = True
        cost_usd = 0.0
        usage = {}
        changed_files = []
        diff_stat = ""
        validation_summary = ""
        error = ""
        result_text = ""

        def to_tool_output(self):
            return "OK"

    def fake_run_edit(**kwargs):
        seen.update(kwargs)
        return _Result()

    fake_module.run_edit = fake_run_edit
    monkeypatch.setitem(sys.modules, "ouroboros.gateways.claude_code", fake_module)

    result = shell_mod._claude_code_edit(ctx, prompt="edit", cwd="skills/external/alpha")

    assert result == "OK"
    assert seen["cwd"] == str(skill_dir.resolve())


def test_claude_code_edit_rejects_escaping_skill_cwd(tmp_path, monkeypatch):
    from ouroboros.tools import shell as shell_mod

    ctx = _make_ctx(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    result = shell_mod._claude_code_edit(ctx, prompt="edit", cwd="skills/../../repo")

    assert "skill cwd is invalid" in result


def test_claude_code_edit_blocks_native_skill_cwd(tmp_path, monkeypatch):
    from ouroboros.tools import shell as shell_mod

    ctx = _make_ctx(tmp_path)
    (ctx.drive_root / "skills" / "native" / "alpha").mkdir(parents=True)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    result = shell_mod._claude_code_edit(ctx, prompt="edit", cwd="skills/native/alpha")

    assert "skill cwd is invalid" in result


def test_skill_exec_tools_have_policy_entries():
    """Every new tool must carry an explicit TOOL_POLICY entry."""
    from ouroboros.safety import TOOL_POLICY, POLICY_CHECK, POLICY_SKIP

    assert TOOL_POLICY["list_skills"] == POLICY_SKIP
    assert TOOL_POLICY["skill_review"] == POLICY_SKIP
    assert TOOL_POLICY["toggle_skill"] == POLICY_SKIP
    assert TOOL_POLICY["skill_preflight"] == POLICY_SKIP
    assert TOOL_POLICY["skill_exec"] == POLICY_CHECK


def test_skill_exec_in_frozen_modules():
    from ouroboros.tools.registry import ToolRegistry

    assert "skill_exec" in ToolRegistry._FROZEN_TOOL_MODULES


# ---------------------------------------------------------------------------
# Preflight: data-plane skills are sufficient without external repo path
# ---------------------------------------------------------------------------


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


def test_skill_exec_refuses_when_unconfigured(tmp_path, monkeypatch):
    monkeypatch.delenv("OUROBOROS_SKILLS_REPO_PATH", raising=False)
    ctx = _make_ctx(tmp_path)
    result = skill_exec_mod._handle_skill_exec(ctx, skill="x", script="y")
    assert "SKILLS_UNAVAILABLE" in result


# ---------------------------------------------------------------------------
# Runtime-mode semantics in v5.1.2 (Frame A):
# ``light`` blocks repo self-modification but ALLOWS reviewed + enabled
# skills to execute. The previous Frame-B regression (light blocking
# skill_exec) is replaced by ``test_skill_exec_runs_in_light_mode`` in
# tests/test_runtime_mode_core.py — covering the positive path.
# Light still blocks every escalation channel of the runtime_mode axis
# itself; that is enforced by the chokepoint in
# ``ouroboros.config.save_settings`` and ``_data_write`` settings.json
# block, exercised in tests/test_runtime_mode_elevation.py.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Review-status + enable gating
# ---------------------------------------------------------------------------


def test_skill_exec_refuses_disabled_skill(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    skill_dir = _build_skill(skills_root, "hello")
    ctx = _make_ctx(tmp_path)
    # Only mark review PASS; leave enabled=False.
    content_hash = compute_content_hash(skill_dir)
    save_review_state(
        ctx.drive_root,
        "hello",
        SkillReviewState(status="pass", content_hash=content_hash),
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    result = skill_exec_mod._handle_skill_exec(
        ctx, skill="hello", script="scripts/hello.py"
    )
    assert "SKILL_EXEC_BLOCKED" in result
    assert "disabled" in result


def test_skill_exec_refuses_non_pass_review(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    skill_dir = _build_skill(skills_root, "hello")
    ctx = _make_ctx(tmp_path)
    content_hash = compute_content_hash(skill_dir)
    save_enabled(ctx.drive_root, "hello", True)
    save_review_state(
        ctx.drive_root,
        "hello",
        SkillReviewState(status="blockers", content_hash=content_hash),
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    result = skill_exec_mod._handle_skill_exec(
        ctx, skill="hello", script="scripts/hello.py"
    )
    assert "SKILL_EXEC_BLOCKED" in result
    assert "'blockers'" in result


def test_skill_exec_refuses_stale_review(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    skill_dir = _build_skill(skills_root, "hello")
    ctx = _make_ctx(tmp_path)
    save_enabled(ctx.drive_root, "hello", True)
    # Save review keyed to an old hash, then edit the script.
    save_review_state(
        ctx.drive_root,
        "hello",
        SkillReviewState(status="pass", content_hash="OLD_HASH"),
    )
    (skill_dir / "scripts" / "hello.py").write_text("print('edited')\n", encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    result = skill_exec_mod._handle_skill_exec(
        ctx, skill="hello", script="scripts/hello.py"
    )
    assert "SKILL_EXEC_BLOCKED" in result
    assert "edited since the last review" in result


def test_skill_exec_refuses_extension_skill_in_phase3(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    manifest = (
        "---\n"
        "name: ext1\n"
        "version: 0.1.0\n"
        "type: extension\n"
        "entry: plugin.py\n"
        "permissions: [widget]\n"
        "---\n"
        "body\n"
    )
    skill_dir = _build_skill(skills_root, "ext1", manifest=manifest)
    (skill_dir / "plugin.py").write_text("def register(api): pass\n", encoding="utf-8")
    ctx = _make_ctx(tmp_path)
    _mark_reviewed_and_enabled(ctx.drive_root, skill_dir, "ext1")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    result = skill_exec_mod._handle_skill_exec(
        ctx, skill="ext1", script="plugin.py"
    )
    # Phase 4: extension skills no longer return SKILL_EXEC_DEFERRED —
    # they return SKILL_EXEC_EXTENSION pointing the caller at the
    # in-process PluginAPI surface (Phase 5 wires the dispatchers).
    assert "SKILL_EXEC_EXTENSION" in result
    assert "extension_loader" in result


# ---------------------------------------------------------------------------
# Path confinement
# ---------------------------------------------------------------------------


def test_skill_exec_rejects_absolute_and_parent_paths(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    skill_dir = _build_skill(skills_root, "hello")
    ctx = _make_ctx(tmp_path)
    _mark_reviewed_and_enabled(ctx.drive_root, skill_dir, "hello")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")

    for bad in ("/etc/passwd", "~/.ssh/id_rsa", "../../etc/passwd", ""):
        result = skill_exec_mod._handle_skill_exec(
            ctx, skill="hello", script=bad
        )
        if bad == "":
            assert "SKILL_EXEC_ERROR" in result
        else:
            assert "SKILL_EXEC_ERROR" in result


def test_skill_exec_rejects_file_outside_declared_scripts(tmp_path, monkeypatch):
    """Regression (Phase 3 round 4): skill_exec's executable surface must
    equal the manifest-declared ``scripts:`` list, not the broader
    reviewed-content set. Assets, SKILL.md, or stray in-repo files must
    not be runnable even if they live in the reviewed skill directory."""
    skills_root = tmp_path / "skills"
    skill_dir = _build_skill(skills_root, "hello")
    # Drop a stray file directly in skill_dir (not declared in manifest).
    (skill_dir / "unreviewed.py").write_text("print('unreviewed')\n", encoding="utf-8")
    (skill_dir / "assets").mkdir()
    (skill_dir / "assets" / "data.py").write_text("print('asset-code')\n", encoding="utf-8")
    ctx = _make_ctx(tmp_path)
    _mark_reviewed_and_enabled(ctx.drive_root, skill_dir, "hello")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")

    for bad in ("unreviewed.py", "assets/data.py", "SKILL.md"):
        result = skill_exec_mod._handle_skill_exec(
            ctx, skill="hello", script=bad
        )
        assert "SKILL_EXEC_ERROR" in result, f"bad={bad!r}: {result}"
        assert "not a declared script" in result, f"bad={bad!r}: {result}"


def test_skill_exec_refuses_instruction_type_skill(tmp_path, monkeypatch):
    """Phase 3 only executes ``type: script`` skills. An ``instruction``
    skill that went through review PASS must still be blocked at
    execution (its manifest declares no scripts anyway, but we want
    belt-and-braces type gating)."""
    skills_root = tmp_path / "skills"
    skill_dir = skills_root / "guide"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: guide\n"
        "description: Pure markdown guide.\n"
        "version: 0.1.0\n"
        "type: instruction\n"
        "---\n"
        "# body\nread me.\n",
        encoding="utf-8",
    )
    # Drop a file just to see if skill_exec tries to run it.
    (skill_dir / "scripts").mkdir()
    (skill_dir / "scripts" / "boom.py").write_text("print('boom')\n", encoding="utf-8")
    ctx = _make_ctx(tmp_path)
    _mark_reviewed_and_enabled(ctx.drive_root, skill_dir, "guide")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")

    result = skill_exec_mod._handle_skill_exec(
        ctx, skill="guide", script="scripts/boom.py"
    )
    assert "SKILL_EXEC_ERROR" in result
    assert "'instruction'" in result, result


def test_skill_exec_rejects_runtime_outside_allowlist(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    skill_dir = _build_skill(
        skills_root,
        "hello",
        manifest=_valid_script_manifest("hello", runtime="perl"),
    )
    ctx = _make_ctx(tmp_path)
    _mark_reviewed_and_enabled(ctx.drive_root, skill_dir, "hello")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    result = skill_exec_mod._handle_skill_exec(
        ctx, skill="hello", script="scripts/hello.py"
    )
    assert "SKILL_EXEC_ERROR" in result
    assert "allowlist" in result


# ---------------------------------------------------------------------------
# Happy path: actual subprocess execution
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("python3") is None, reason="python3 not on PATH")
def test_skill_exec_runs_reviewed_skill_successfully(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    # Use a script that prints env + cwd so we can verify environment scrubbing.
    skill_dir = _build_skill(
        skills_root,
        "hello",
        script_body=(
            "import json, os, sys\n"
            # ``has_home`` must be True when either the Unix ``HOME`` or the
            # Windows ``USERPROFILE`` is forwarded — the scrub layer copies
            # both (see ``_ALWAYS_FORWARDED_ENV``); checking only ``HOME``
            # would spuriously fail on Windows CI where the parent process
            # exports ``USERPROFILE`` instead.
            "print(json.dumps({'cwd': os.getcwd(), 'skill': os.environ.get('OUROBOROS_SKILL_NAME'), "
            "'argv': sys.argv[1:], "
            "'has_home': ('HOME' in os.environ) or ('USERPROFILE' in os.environ), "
            "'openrouter_leaked': 'OPENROUTER_API_KEY' in os.environ}))\n"
        ),
    )
    ctx = _make_ctx(tmp_path)
    _mark_reviewed_and_enabled(ctx.drive_root, skill_dir, "hello")

    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    # Deliberately set a secret that the scrubbed env must NOT forward.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-must-not-leak")

    raw = skill_exec_mod._handle_skill_exec(
        ctx, skill="hello", script="scripts/hello.py", args=["alpha", "beta"]
    )
    payload = json.loads(raw)
    assert payload["skill"] == "hello"
    assert payload["script"] == "scripts/hello.py"
    assert payload["exit_code"] == 0
    stdout_line = payload["stdout"].strip().splitlines()[-1]
    stdout = json.loads(stdout_line)
    # cwd must be inside the skill directory, not the main repo.
    assert stdout["cwd"].startswith(str(skill_dir))
    assert stdout["skill"] == "hello"
    assert stdout["argv"] == ["alpha", "beta"]
    assert stdout["has_home"] is True
    # Secret key must not leak into the subprocess environment.
    assert stdout["openrouter_leaked"] is False
    events = [
        json.loads(line)
        for line in (ctx.drive_root / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert events[-1]["type"] == "skill_exec_finished"
    assert events[-1]["skill"] == "hello"
    assert events[-1]["exit_code"] == 0


def test_skill_exec_queues_lifecycle_event_for_supervisor(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    skill_dir = _build_skill(
        skills_root,
        "hello",
        script_body="print('ok')\n",
    )
    ctx = _make_ctx(tmp_path)
    queued = []

    class _Queue:
        def put_nowait(self, item):
            queued.append(item)

    ctx.event_queue = _Queue()
    _mark_reviewed_and_enabled(ctx.drive_root, skill_dir, "hello")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")

    raw = skill_exec_mod._handle_skill_exec(ctx, skill="hello", script="scripts/hello.py")

    assert json.loads(raw)["exit_code"] == 0
    assert queued[-1]["type"] == "skill_exec_finished"
    assert queued[-1]["skill"] == "hello"
    assert queued[-1]["exit_code"] == 0


def test_skill_exec_runs_in_light_mode(tmp_path, monkeypatch):
    """v5.1.2 Frame A: ``light`` allows reviewed + enabled skills to
    execute. The privilege scope ``light`` controls is repo
    self-modification and the runtime_mode elevation ratchet, NOT
    owner-approved skills (skills already pass tri-model review +
    enabled.json toggle + content-hash freshness + sandboxed
    subprocess). This is the positive replacement for the deleted
    Frame-B regression ``test_skill_exec_blocked_in_light_mode``.
    """
    skills_root = tmp_path / "skills"
    skill_dir = _build_skill(
        skills_root,
        "hello",
        script_body="import json; print(json.dumps({'ok': True}))\n",
    )
    ctx = _make_ctx(tmp_path)
    _mark_reviewed_and_enabled(ctx.drive_root, skill_dir, "hello")

    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")

    raw = skill_exec_mod._handle_skill_exec(
        ctx, skill="hello", script="scripts/hello.py"
    )
    # Must NOT be the v5.0.0 Frame-B sentinel.
    assert "SKILL_EXEC_BLOCKED" not in raw
    payload = json.loads(raw)
    assert payload["skill"] == "hello"
    assert payload["exit_code"] == 0
    stdout_line = payload["stdout"].strip().splitlines()[-1]
    assert json.loads(stdout_line) == {"ok": True}


# ---------------------------------------------------------------------------
# toggle_skill
# ---------------------------------------------------------------------------


def test_toggle_skill_persists_enable_state(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    skill_dir = _build_skill(skills_root, "alpha")
    ctx = _make_ctx(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    _mark_reviewed(ctx.drive_root, skill_dir, "alpha")

    # Enable, then disable.
    enabled_resp = json.loads(skill_exec_mod._handle_toggle_skill(ctx, skill="alpha", enabled=True))
    assert enabled_resp["enabled"] is True
    assert "alpha" in enabled_resp["message"]

    disabled_resp = json.loads(skill_exec_mod._handle_toggle_skill(ctx, skill="alpha", enabled=False))
    assert disabled_resp["enabled"] is False


def test_toggle_skill_allows_warnings_review(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    skill_dir = _build_skill(skills_root, "alpha")
    ctx = _make_ctx(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    save_review_state(
        ctx.drive_root,
        "alpha",
        SkillReviewState(status="warnings", content_hash=compute_content_hash(skill_dir)),
    )

    enabled_resp = json.loads(skill_exec_mod._handle_toggle_skill(ctx, skill="alpha", enabled=True))

    assert enabled_resp["enabled"] is True
    assert enabled_resp["review_status"] == "warnings"
    assert enabled_resp["executable_review"] is True


def test_toggle_skill_allows_warnings_under_blocking(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    skill_dir = _build_skill(skills_root, "alpha")
    ctx = _make_ctx(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    save_review_state(
        ctx.drive_root,
        "alpha",
        SkillReviewState(status="warnings", content_hash=compute_content_hash(skill_dir)),
    )

    resp = skill_exec_mod._handle_toggle_skill(ctx, skill="alpha", enabled=True)

    enabled_resp = json.loads(resp)
    assert enabled_resp["enabled"] is True
    assert enabled_resp["review_status"] == "warnings"
    assert enabled_resp["executable_review"] is True


def test_skill_exec_allows_warnings_under_blocking(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    skill_dir = _build_skill(skills_root, "alpha")
    ctx = _make_ctx(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    save_enabled(ctx.drive_root, "alpha", True)
    save_review_state(
        ctx.drive_root,
        "alpha",
        SkillReviewState(status="warnings", content_hash=compute_content_hash(skill_dir)),
    )

    resp = skill_exec_mod._handle_skill_exec(ctx, skill="alpha", script="hello.py")

    payload = json.loads(resp)
    assert payload["exit_code"] == 0
    assert "hello from skill" in payload["stdout"]


def test_toggle_skill_blocks_stale_dependency_fingerprint(tmp_path, monkeypatch):
    from ouroboros.marketplace.isolated_deps import (
        DEPS_STATE_FILENAME,
        FINGERPRINT_FILENAME,
        isolated_env_dir,
    )
    from ouroboros.skill_loader import skill_state_dir

    skills_root = tmp_path / "skills"
    manifest = _valid_script_manifest("alpha").replace(
        "scripts:\n",
        "install_specs:\n"
        "  - kind: pip\n"
        "    package: wheel\n"
        "scripts:\n",
    )
    skill_dir = _build_skill(skills_root, "alpha", manifest=manifest)
    ctx = _make_ctx(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    _mark_reviewed(ctx.drive_root, skill_dir, "alpha")
    stale_state = {"status": "installed", "specs_hash": "old"}
    state_dir = skill_state_dir(ctx.drive_root, "alpha")
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / DEPS_STATE_FILENAME).write_text(json.dumps(stale_state), encoding="utf-8")
    env_dir = isolated_env_dir(skill_dir)
    env_dir.mkdir(parents=True)
    (env_dir / FINGERPRINT_FILENAME).write_text(json.dumps(stale_state), encoding="utf-8")

    resp = skill_exec_mod._handle_toggle_skill(ctx, skill="alpha", enabled=True)

    assert "dependency fingerprint is stale" in resp
    assert not (state_dir / "enabled.json").exists()


def test_skill_exec_refuses_missing_manifest_permission_grant(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    manifest = (
        "---\n"
        "name: alpha\n"
        "description: Permission grant test.\n"
        "version: 0.1.0\n"
        "type: script\n"
        "runtime: python3\n"
        "permissions: [inject_chat]\n"
        "scripts:\n"
        "  - name: hello.py\n"
        "    description: Print hello.\n"
        "---\n"
        "# body\n"
    )
    skill_dir = _build_skill(skills_root, "alpha", manifest=manifest)
    ctx = _make_ctx(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    _mark_reviewed_and_enabled(ctx.drive_root, skill_dir, "alpha")

    resp = skill_exec_mod._handle_skill_exec(ctx, skill="alpha", script="hello.py")

    assert "SKILL_EXEC_GRANT_REQUIRED" in resp
    assert "inject_chat" in resp


def test_toggle_skill_reports_missing_manifest_permission_grant(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    manifest = (
        "---\n"
        "name: alpha\n"
        "description: Permission grant test.\n"
        "version: 0.1.0\n"
        "type: script\n"
        "runtime: python3\n"
        "permissions: [inject_chat]\n"
        "scripts:\n"
        "  - name: hello.py\n"
        "    description: Print hello.\n"
        "---\n"
        "# body\n"
    )
    skill_dir = _build_skill(skills_root, "alpha", manifest=manifest)
    ctx = _make_ctx(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    _mark_reviewed(ctx.drive_root, skill_dir, "alpha")

    resp = skill_exec_mod._handle_toggle_skill(ctx, skill="alpha", enabled=True)

    assert "SKILL_TOGGLE_ERROR" in resp
    assert "inject_chat" in resp


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
    ("schedule_subagent", {"text": "enable skill"}),
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
    (ctx.drive_root / "skills" / "external" / "alpha").mkdir(parents=True)
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
    (ctx.drive_root / "skills" / "ouroboroshub" / "nanobanana").mkdir(parents=True)
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


def test_review_skill_tool_records_lifecycle_job_state_and_events(tmp_path, monkeypatch):
    from ouroboros.skill_review import SkillReviewOutcome
    import ouroboros.skill_lifecycle_queue as lifecycle_queue

    lifecycle_queue._events.clear()
    lifecycle_queue._active = None
    lifecycle_queue._lock = None
    lifecycle_queue._dedupe_jobs.clear()

    ctx = _make_ctx(tmp_path)
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    skill_dir = _build_skill(skills_root, "alpha")
    content_hash = compute_content_hash(skill_dir)

    monkeypatch.setattr(
        skill_exec_mod,
        "_review_skill_impl",
        lambda _ctx, skill_name: SkillReviewOutcome(
            skill_name=skill_name,
            status="pass",
            content_hash=content_hash,
            reviewer_models=["fake/reviewer"],
            findings=[],
            error="",
        ),
    )

    from ouroboros.skill_review_runner import run_skill_review_lifecycle_blocking
    result = run_skill_review_lifecycle_blocking(
        ctx, "alpha", source="tool",
        review_impl=lambda rc, rn: skill_exec_mod._review_skill_impl(rc, rn),
    )

    assert result["status"] == "clean"
    assert result["deps_status"] == "not_required"
    review_job = json.loads(
        (ctx.drive_root / "state" / "skills" / "alpha" / "review_job.json").read_text(encoding="utf-8")
    )
    assert review_job["status"] == "completed"
    assert review_job["review_status"] == "clean"
    assert review_job["job_id"].startswith("skill-job-")
    lifecycle_event = lifecycle_queue.queue_snapshot()["events"][-1]
    assert lifecycle_event["kind"] == "review"
    assert lifecycle_event["target"] == "alpha"
    events_text = (ctx.drive_root / "logs" / "events.jsonl").read_text(encoding="utf-8")
    assert "skill_review_started" in events_text
    assert "skill_review_completed" in events_text


def test_stale_review_job_is_marked_interrupted(tmp_path, monkeypatch):
    from ouroboros.skill_review_runner import (
        mark_stale_review_job_interrupted,
        review_job_state_path,
    )

    ctx = _make_ctx(tmp_path)
    job_path = review_job_state_path(ctx.drive_root, "alpha")
    job_path.write_text(
        json.dumps(
            {
                "status": "running",
                "skill": "alpha",
                "content_hash": "abc",
                "job_id": "skill-job-old",
                "started_at": "2026-01-01T00:00:00+00:00",
                "last_heartbeat_at": "2026-01-01T00:00:00+00:00",
                "pid": 123456,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("ouroboros.skill_review_runner._pid_alive", lambda _pid: False)

    mark_stale_review_job_interrupted(ctx.drive_root, "alpha", current_content_hash="abc")

    data = json.loads(job_path.read_text(encoding="utf-8"))
    assert data["status"] == "interrupted"
    assert data["interrupt_reason"] == "owner_process_exited"
    events_text = (ctx.drive_root / "logs" / "events.jsonl").read_text(encoding="utf-8")
    assert "skill_review_interrupted" in events_text
    progress = [
        json.loads(line)
        for line in (ctx.drive_root / "logs" / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert progress[-1]["task_id"] == "skill_lifecycle_review_alpha_skill-job-old"
    assert progress[-1]["lifecycle"]["status"] == "interrupted"
    assert progress[-1]["lifecycle"]["phase"] == "interrupted"


def test_reconcile_stale_review_jobs_heals_dead_running_job(tmp_path, monkeypatch):
    # The periodic supervisor reconcile (server.py) calls this to heal a worker
    # that died mid-review and left review_job.json at status=running in a
    # headless/no-UI run where boot/extensions-API reconciles never fire.
    from ouroboros.skill_review_runner import (
        reconcile_stale_review_jobs,
        review_job_state_path,
    )

    ctx = _make_ctx(tmp_path)
    job_path = review_job_state_path(ctx.drive_root, "beta")
    job_path.parent.mkdir(parents=True, exist_ok=True)
    job_path.write_text(
        json.dumps(
            {
                "status": "running",
                "skill": "beta",
                "content_hash": "h1",
                "job_id": "skill-job-dead",
                "started_at": "2026-01-01T00:00:00+00:00",
                "last_heartbeat_at": "2026-01-01T00:00:00+00:00",
                "pid": 999999,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("ouroboros.skill_review_runner._pid_alive", lambda _pid: False)

    healed = reconcile_stale_review_jobs(ctx.drive_root)

    assert healed == 1
    data = json.loads(job_path.read_text(encoding="utf-8"))
    assert data["status"] == "interrupted"
    assert data["interrupt_reason"] == "owner_process_exited"


def test_async_review_cancellation_waits_for_review_thread(tmp_path, monkeypatch):
    from ouroboros.skill_review import SkillReviewOutcome
    from ouroboros.skill_review_runner import run_skill_review_lifecycle
    import ouroboros.skill_lifecycle_queue as lifecycle_queue

    lifecycle_queue._events.clear()
    lifecycle_queue._active = None
    lifecycle_queue._lock = None
    lifecycle_queue._dedupe_jobs.clear()

    ctx = _make_ctx(tmp_path)
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    skill_dir = _build_skill(skills_root, "alpha")
    content_hash = compute_content_hash(skill_dir)
    started = threading.Event()
    release = threading.Event()

    def fake_review(_ctx, skill_name):
        started.set()
        release.wait(2)
        return SkillReviewOutcome(
            skill_name=skill_name,
            status="pass",
            content_hash=content_hash,
            reviewer_models=["fake/reviewer"],
            findings=[],
            error="",
        )

    async def main():
        task = asyncio.create_task(
            run_skill_review_lifecycle(ctx, "alpha", source="test", review_impl=fake_review)
        )
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.sleep(0.05)
        active = lifecycle_queue.queue_snapshot()["active"]
        assert active is not None
        assert active["target"] == "alpha"
        quick = asyncio.create_task(
            lifecycle_queue.run_lifecycle_job(
                kind="review",
                target="beta",
                dedupe_key="review:beta:hash",
                runner=lambda: asyncio.sleep(0, result={"quick": True}),
                options=lifecycle_queue.LifecycleJobOptions(drive_root=ctx.drive_root),
            )
        )
        await asyncio.sleep(0.05)
        assert not quick.done()
        release.set()
        result = await asyncio.wait_for(task, timeout=2)
        assert result["status"] == "clean"
        assert await asyncio.wait_for(quick, timeout=2) == {"quick": True}
        assert lifecycle_queue.queue_snapshot()["active"] is None

    asyncio.run(main())


def test_toggle_skill_requires_both_args(tmp_path, monkeypatch):
    ctx = _make_ctx(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(tmp_path / "skills"))
    (tmp_path / "skills").mkdir()
    assert "SKILL_TOGGLE_ERROR" in skill_exec_mod._handle_toggle_skill(ctx, skill="", enabled=True)
    assert "SKILL_TOGGLE_ERROR" in skill_exec_mod._handle_toggle_skill(ctx, skill="x", enabled=None)


def test_toggle_skill_rejects_ambiguous_non_boolean(tmp_path, monkeypatch):
    """Phase 3 round 13 regression: ``bool('false') == True``. The
    toggle must reject non-boolean / non-canonical string inputs
    rather than silently enabling when the caller meant to disable."""
    import json as _json
    skills_root = tmp_path / "skills"
    skill_dir = _build_skill(skills_root, "alpha")
    ctx = _make_ctx(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    _mark_reviewed(ctx.drive_root, skill_dir, "alpha")

    # These look booleans-ish but could flip enabled incorrectly under
    # naive ``bool()`` coercion. The handler must accept them ONLY
    # when the string matches a canonical true/false literal.
    # Narrow allowlist is OK: "True", "false", "1", "0".
    assert _json.loads(skill_exec_mod._handle_toggle_skill(ctx, skill="alpha", enabled="True"))["enabled"] is True
    assert _json.loads(skill_exec_mod._handle_toggle_skill(ctx, skill="alpha", enabled="false"))["enabled"] is False
    assert _json.loads(skill_exec_mod._handle_toggle_skill(ctx, skill="alpha", enabled=1))["enabled"] is True
    assert _json.loads(skill_exec_mod._handle_toggle_skill(ctx, skill="alpha", enabled=0))["enabled"] is False

    # Non-boolean / non-canonical → rejected with SKILL_TOGGLE_ERROR.
    for bogus in ("maybe", "probably", 42, 2.5, [], {}):
        resp = skill_exec_mod._handle_toggle_skill(ctx, skill="alpha", enabled=bogus)
        assert "SKILL_TOGGLE_ERROR" in resp, f"bogus={bogus!r} was accepted: {resp}"


def test_toggle_skill_rejects_stale_pass_review(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    _build_skill(skills_root, "alpha")
    ctx = _make_ctx(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    save_review_state(
        ctx.drive_root,
        "alpha",
        SkillReviewState(status="pass", content_hash="OLD_HASH"),
    )

    resp = skill_exec_mod._handle_toggle_skill(ctx, skill="alpha", enabled=True)
    assert "SKILL_TOGGLE_ERROR" in resp
    assert "fresh executable review" in resp


def test_skill_exec_rejects_misserialized_args(tmp_path, monkeypatch):
    """Phase 3 round 16 regression: args as a scalar/string must be
    rejected explicitly, not exploded per-character into argv."""
    skills_root = tmp_path / "skills"
    skill_dir = _build_skill(skills_root, "hello")
    ctx = _make_ctx(tmp_path)
    _mark_reviewed_and_enabled(ctx.drive_root, skill_dir, "hello")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")

    for bogus in ("alpha", 1, 2.5, True, False, {"k": "v"}):
        result = skill_exec_mod._handle_skill_exec(
            ctx, skill="hello", script="scripts/hello.py", args=bogus
        )
        assert "SKILL_EXEC_ERROR" in result, f"args={bogus!r}: {result}"


def test_skill_exec_kills_runaway_stdout_output(tmp_path, monkeypatch):
    """Phase 3 round 17 regression: stdout/stderr byte caps must be
    enforced at STREAMING time, not post-hoc. A malicious skill that
    writes >>cap bytes must be killed and surface SKILL_EXEC_OVERFLOW
    instead of buffering into Ouroboros memory."""
    skills_root = tmp_path / "skills"
    # Write far more than _MAX_STDOUT_BYTES (64 KB) — 4 MiB forces a
    # streamer that only post-hoc caps to buffer the whole thing.
    body = (
        "import sys\n"
        "chunk = 'x' * 4096\n"
        "for _ in range(1024):\n"
        "    sys.stdout.write(chunk)\n"
        "    sys.stdout.flush()\n"
    )
    skill_dir = _build_skill(skills_root, "flood", script_body=body)
    ctx = _make_ctx(tmp_path)
    _mark_reviewed_and_enabled(ctx.drive_root, skill_dir, "flood")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    result = skill_exec_mod._handle_skill_exec(
        ctx, skill="flood", script="scripts/hello.py"
    )
    assert "SKILL_EXEC_OVERFLOW" in result, result[:500]
    # Output in the returned payload must be bounded by the cap.
    import json as _json
    json_start = result.find("{")
    payload = _json.loads(result[json_start:])
    # Streamed stdout buffer must be close to the cap, not megabytes.
    assert len(payload["stdout"]) <= skill_exec_mod._MAX_STDOUT_BYTES + 1024
    assert payload["output_overflow"] is True


def test_skill_exec_surfaces_wall_clock_timeout(tmp_path, monkeypatch):
    """Phase 3 round 23 regression: wall-clock timeout surfaces as
    ``SKILL_EXEC_TIMEOUT`` with captured partial output instead of
    silently hanging."""
    skills_root = tmp_path / "skills"
    # Manifest declares 1-second timeout; script sleeps 10s.
    manifest = (
        "---\n"
        "name: sleepy\n"
        "description: Sleeps too long.\n"
        "version: 0.1.0\n"
        "type: script\n"
        "runtime: python3\n"
        "timeout_sec: 1\n"
        "scripts:\n"
        "  - name: hello.py\n"
        "---\n"
        "body\n"
    )
    skill_dir = _build_skill(
        skills_root,
        "sleepy",
        manifest=manifest,
        script_body=(
            "import sys, time\n"
            "sys.stdout.write('hi\\n'); sys.stdout.flush()\n"
            "time.sleep(10)\n"
        ),
    )
    ctx = _make_ctx(tmp_path)
    _mark_reviewed_and_enabled(ctx.drive_root, skill_dir, "sleepy")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    result = skill_exec_mod._handle_skill_exec(
        ctx, skill="sleepy", script="scripts/hello.py"
    )
    assert "SKILL_EXEC_TIMEOUT" in result, result[:400]
    assert "1s limit" in result
    # Partial stdout captured before the kill.
    assert "hi" in result


def test_skill_exec_surfaces_nonzero_exit_as_failure(tmp_path, monkeypatch):
    """Phase 3 round 16 regression: a crashing skill script must be
    reported as a failed tool outcome (with SKILL_EXEC_FAILED sentinel),
    not a normal structured response the model might skim past."""
    skills_root = tmp_path / "skills"
    skill_dir = _build_skill(
        skills_root,
        "crashy",
        script_body="import sys\nprint('before crash')\nsys.exit(7)\n",
    )
    ctx = _make_ctx(tmp_path)
    _mark_reviewed_and_enabled(ctx.drive_root, skill_dir, "crashy")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")

    result = skill_exec_mod._handle_skill_exec(
        ctx, skill="crashy", script="scripts/hello.py"
    )
    assert "SKILL_EXEC_FAILED" in result
    assert "exit_code" in result
    assert "7" in result
    events = [
        json.loads(line)
        for line in (ctx.drive_root / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert events[-1]["type"] == "skill_exec_failed"
    assert events[-1]["skill"] == "crashy"
    assert events[-1]["exit_code"] == 7


def test_toggle_skill_loads_and_unloads_extension_plugin(tmp_path, monkeypatch):
    """Phase 4 regression: enabling a type=extension skill via
    toggle_skill must actually call extension_loader.load_extension,
    and disabling must call unload_extension — otherwise the extension
    surface is mystery state relative to what the Skills UI says."""
    from ouroboros import extension_loader
    skills_root = tmp_path / "skills"
    skill_dir = skills_root / "ext_live"
    skill_dir.mkdir(parents=True)
    import json as _json
    (skill_dir / "SKILL.md").write_text(
        (
            "---\n"
            "name: ext_live\n"
            "description: Runtime ext.\n"
            "version: 0.1.0\n"
            "type: extension\n"
            "entry: plugin.py\n"
            f"permissions: {_json.dumps(['tool'])}\n"
            "---\n"
            "body\n"
        ),
        encoding="utf-8",
    )
    (skill_dir / "plugin.py").write_text(
        (
            "def _t(ctx): return 'ok'\n"
            "def register(api):\n"
            "    api.register_tool('t', _t, description='', schema={})\n"
        ),
        encoding="utf-8",
    )
    ctx = _make_ctx(tmp_path)
    content_hash = compute_content_hash(
        skill_dir, manifest_entry="plugin.py", manifest_scripts=None
    )
    save_review_state(
        ctx.drive_root,
        "ext_live",
        SkillReviewState(status="pass", content_hash=content_hash),
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))

    # Clean slate.
    extension_loader.unload_extension("ext_live")
    assert "ext_live" not in extension_loader.snapshot()["extensions"]

    # Enable → plugin gets loaded into the runtime registry.
    enable_resp = _json.loads(
        skill_exec_mod._handle_toggle_skill(ctx, skill="ext_live", enabled=True)
    )
    assert enable_resp["extension_action"] == "extension_loaded"
    snap = extension_loader.snapshot()
    assert "ext_live" in snap["extensions"]
    assert extension_loader.extension_surface_name("ext_live", "t") in snap["tools"]

    # Disable → the plugin is torn down.
    disable_resp = _json.loads(
        skill_exec_mod._handle_toggle_skill(ctx, skill="ext_live", enabled=False)
    )
    assert disable_resp["extension_action"] == "extension_unloaded"
    snap = extension_loader.snapshot()
    assert "ext_live" not in snap["extensions"]


def test_review_skill_reconciles_live_extension_after_review(tmp_path, monkeypatch):
    from ouroboros import extension_loader
    from ouroboros.skill_loader import find_skill
    from ouroboros.skill_review import SkillReviewOutcome

    skills_root = tmp_path / "skills"
    skill_dir = skills_root / "ext_reviewed"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        (
            "---\n"
            "name: ext_reviewed\n"
            "description: Runtime ext.\n"
            "version: 0.1.0\n"
            "type: extension\n"
            "entry: plugin.py\n"
            "permissions: [\"tool\"]\n"
            "---\n"
            "body\n"
        ),
        encoding="utf-8",
    )
    (skill_dir / "plugin.py").write_text(
        (
            "def _t(ctx): return 'v1'\n"
            "def register(api):\n"
            "    api.register_tool('t', _t, description='', schema={})\n"
        ),
        encoding="utf-8",
    )
    ctx = _make_ctx(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py", manifest_scripts=None)
    save_enabled(ctx.drive_root, "ext_reviewed", True)
    save_review_state(
        ctx.drive_root,
        "ext_reviewed",
        SkillReviewState(status="pass", content_hash=content_hash),
    )
    loaded = find_skill(ctx.drive_root, "ext_reviewed")
    assert loaded is not None
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=ctx.drive_root)
    assert err is None, err
    tool = extension_loader.get_tool(extension_loader.extension_surface_name("ext_reviewed", "t"))
    assert tool is not None
    assert tool["handler"](None) == "v1"

    (skill_dir / "plugin.py").write_text(
        (
            "def _t(ctx): return 'v2'\n"
            "def register(api):\n"
            "    api.register_tool('t', _t, description='', schema={})\n"
        ),
        encoding="utf-8",
    )

    def _fake_review(ctx_arg, skill_name):
        refreshed = find_skill(pathlib.Path(ctx_arg.drive_root), skill_name)
        assert refreshed is not None
        save_review_state(
            pathlib.Path(ctx_arg.drive_root),
            skill_name,
            SkillReviewState(status="pass", content_hash=refreshed.content_hash),
        )
        return SkillReviewOutcome(
            skill_name=skill_name,
            status="pass",
            findings=[],
            reviewer_models=["fake/reviewer"],
            content_hash=refreshed.content_hash,
            error="",
        )

    from ouroboros.skill_review_runner import run_skill_review_lifecycle_blocking
    with patch.object(skill_exec_mod, "_review_skill_impl", side_effect=_fake_review):
        result = run_skill_review_lifecycle_blocking(
            ctx, "ext_reviewed", source="tool",
            review_impl=lambda rc, rn: skill_exec_mod._review_skill_impl(rc, rn),
        )
    assert result["extension_action"] == "extension_loaded"
    tool = extension_loader.get_tool(extension_loader.extension_surface_name("ext_reviewed", "t"))
    assert tool is not None
    assert tool["handler"](None) == "v2"


def test_toggle_skill_refuses_when_load_error_set(tmp_path, monkeypatch):
    """Phase 3 round 13 regression: a sanitised-name collision marks
    both skills with load_error. ``toggle_skill`` must not mutate state
    for such skills — otherwise the two directories would still end up
    sharing ``enabled.json``."""
    skills_root = tmp_path / "skills"
    _build_skill(skills_root, "hello world")
    _build_skill(skills_root, "hello_world")
    ctx = _make_ctx(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    result = skill_exec_mod._handle_toggle_skill(ctx, skill="hello_world", enabled=True)
    assert "SKILL_TOGGLE_ERROR" in result
    assert "loader rejected" in result
    # enabled.json must NOT have been written under the collision key.
    state_file = ctx.drive_root / "state" / "skills" / "hello_world" / "enabled.json"
    assert not state_file.exists()


def test_toggle_skill_disable_collision_does_not_write_shared_state(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    _build_skill(skills_root, "hello world")
    _build_skill(skills_root, "hello_world")
    ctx = _make_ctx(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))

    result = json.loads(
        skill_exec_mod._handle_toggle_skill(ctx, skill="hello_world", enabled=False)
    )
    assert result["enabled"] is False
    assert result["extension_reason"] == "name_collision"
    assert "not persisted as disabled" in result["message"]
    state_file = ctx.drive_root / "state" / "skills" / "hello_world" / "enabled.json"
    assert not state_file.exists()


def test_skill_exec_returns_controlled_error_when_payload_becomes_unreadable(
    tmp_path, monkeypatch
):
    skills_root = tmp_path / "skills"
    skill_dir = _build_skill(skills_root, "alpha")
    ctx = _make_ctx(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    save_enabled(ctx.drive_root, "alpha", True)
    save_review_state(
        ctx.drive_root,
        "alpha",
        SkillReviewState(
            status="pass",
            content_hash=compute_content_hash(skill_dir, manifest_entry="", manifest_scripts=[{"name": "run.py"}]),
        ),
    )
    with patch.object(
        skill_exec_mod,
        "compute_content_hash",
        side_effect=SkillPayloadUnreadable(
            "blocked.txt",
            PermissionError("permission denied"),
        ),
    ):
        result = skill_exec_mod._handle_skill_exec(ctx, skill="alpha", script="run.py")
    assert "SKILL_EXEC_ERROR" in result
    assert "payload became unreadable" in result


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_runtime_allowlist_covers_phase3_runtimes():
    allowed = set(skill_exec_mod._ALLOWED_RUNTIMES)
    assert {"python", "python3", "bash", "node", "deno", "ruby", "go"} <= allowed


def test_python3_runtime_falls_back_to_python_for_windows():
    """Phase 3 round 6 regression: Windows installs often only ship
    ``python.exe`` (no ``python3.exe``). ``_ALLOWED_RUNTIMES["python3"]``
    must include ``python`` as a fallback so reviewed skills declaring
    ``runtime: python3`` still resolve to a real binary there."""
    assert skill_exec_mod._ALLOWED_RUNTIMES["python3"] == ("python3", "python")


def test_skill_exec_bare_name_resolves_only_to_scripts_dir(tmp_path, monkeypatch):
    """Phase 3 round 8 regression: a bare manifest name (``hello.py``)
    must resolve ONLY to ``scripts/hello.py`` — never to a top-level
    shadow file of the same name. Otherwise a skill author could drop a
    hostile ``hello.py`` next to the real ``scripts/hello.py`` and
    skill_exec would pick the top-level one."""
    skills_root = tmp_path / "skills"
    skill_dir = _build_skill(
        skills_root,
        "hello",
        script_body="print('FROM_SCRIPTS_DIR')\n",
    )
    # Drop a shadow file at the top level — this must NOT run.
    (skill_dir / "hello.py").write_text("print('FROM_SHADOW_TOPLEVEL')\n", encoding="utf-8")
    ctx = _make_ctx(tmp_path)
    _mark_reviewed_and_enabled(ctx.drive_root, skill_dir, "hello")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")

    raw = skill_exec_mod._handle_skill_exec(
        ctx, skill="hello", script="hello.py"
    )
    # Must succeed and run the scripts/hello.py file, not the shadow.
    import json as _json
    payload = _json.loads(raw)
    assert "FROM_SCRIPTS_DIR" in payload["stdout"]
    assert "FROM_SHADOW_TOPLEVEL" not in payload["stdout"]


def test_hard_timeout_ceiling_is_bounded():
    assert 60 <= skill_exec_mod._HARD_TIMEOUT_CEILING_SEC <= 900


def test_env_denylist_blocks_secret_forwarding(tmp_path, monkeypatch):
    """Core settings keys are withheld unless a content-bound grant exists.

    Patch target note (round 21 fix): ``skill_exec.py`` imports
    ``load_settings`` from ``ouroboros.config`` as a bound alias via
    ``from ouroboros.config import … load_settings``. Monkeypatching
    the original in ``ouroboros.config`` leaves the alias unaffected —
    we patch the alias on ``ouroboros.tools.skill_exec`` directly so
    the code under test actually sees the mocked payload."""
    from ouroboros.tools import skill_exec as se

    skill_state_dir_path = tmp_path / "state" / "skills" / "ok"
    skill_state_dir_path.mkdir(parents=True, exist_ok=True)

    with patch.object(
        se,
        "load_settings",
        return_value={
            "OPENROUTER_API_KEY": "sk-or-v1-LEAK-ME",
            "OUROBOROS_NETWORK_PASSWORD": "deadbeef",
            "GITHUB_TOKEN": "ghp_leak",
            "TIMEZONE": "UTC",
            "SOME_OK_KEY": "visible-value",
        },
    ):
        env = se._scrub_env(
            manifest_env_keys=[
                "OPENROUTER_API_KEY",
                "GITHUB_TOKEN",
                "OUROBOROS_NETWORK_PASSWORD",
                "SOME_OK_KEY",
            ],
            skill_state_dir_path=skill_state_dir_path,
            skill_name="ok",
        )
    # Core keys are dropped when no explicit owner grant exists.
    assert "OPENROUTER_API_KEY" not in env, (
        "Runtime must refuse to forward the OpenRouter key without a grant."
    )
    assert "GITHUB_TOKEN" not in env
    assert "OUROBOROS_NETWORK_PASSWORD" not in env
    # Non-protected manifest-requested keys still flow without grants; custom
    # secrets become grant-bound after the owner stores them in Settings.
    assert env["SOME_OK_KEY"] == "visible-value"

    with patch.object(se, "load_settings", return_value={"OPENROUTER_API_KEY": "sk-or-v1-GRANTED"}):
        granted_env = se._scrub_env(
            manifest_env_keys=["OPENROUTER_API_KEY"],
            skill_state_dir_path=skill_state_dir_path,
            skill_name="ok",
            granted_keys=["OPENROUTER_API_KEY"],
        )
    assert granted_env["OPENROUTER_API_KEY"] == "sk-or-v1-GRANTED"

    with patch.object(se, "load_settings", return_value={"SOME_OK_KEY": "visible-value"}):
        custom_env = se._scrub_env(
            manifest_env_keys=["SOME_OK_KEY"],
            skill_state_dir_path=skill_state_dir_path,
            skill_name="ok",
            granted_keys=["SOME_OK_KEY"],
        )
    assert custom_env["SOME_OK_KEY"] == "visible-value"


def test_skill_exec_uses_shared_settings_denylist():
    from ouroboros.contracts.plugin_api import FORBIDDEN_SKILL_SETTINGS

    assert skill_exec_mod._FORBIDDEN_ENV_FORWARD_KEYS == FORBIDDEN_SKILL_SETTINGS
