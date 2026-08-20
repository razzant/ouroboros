"""``toggle_skill``: which enable is persisted, and which one is refused.

Split out of ``tests/test_skill_exec.py`` by theme: the persisted enable state, the enabled
peer conflict, warnings verdicts under both enforcement modes, the stale dependency
fingerprint, the missing manifest permission grant, the argument validation, the stale pass
review, a load error, and the disable collision that must not write shared state.
"""

from __future__ import annotations

import json

from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_enabled, save_review_state
from ouroboros.tools import skill_exec as skill_exec_mod

from tests._skill_exec_shared import (
    _build_skill,
    _make_ctx,
    _mark_reviewed,
    _valid_script_manifest,
)
from tests._skill_exec_shared import (  # noqa: F401  (autouse fixture applies on import)
    _clean_extension_runtime,
)


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


def test_toggle_and_exec_refuse_enabled_peer_conflict(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    alpha_manifest = _valid_script_manifest("alpha").replace(
        "scripts:\n",
        "conflicts: [beta]\nscripts:\n",
    )
    alpha_dir = _build_skill(skills_root, "alpha", manifest=alpha_manifest)
    _build_skill(skills_root, "beta")
    ctx = _make_ctx(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    _mark_reviewed(ctx.drive_root, alpha_dir, "alpha")
    save_enabled(ctx.drive_root, "beta", True)

    toggle = skill_exec_mod._handle_toggle_skill(
        ctx,
        skill="alpha",
        enabled=True,
    )
    assert "SKILL_TOGGLE_ERROR" in toggle
    assert "beta" in toggle

    save_enabled(ctx.drive_root, "alpha", True)
    execution = skill_exec_mod._handle_skill_exec(
        ctx,
        skill="alpha",
        script="scripts/hello.py",
    )
    assert "SKILL_EXEC_BLOCKED" in execution
    assert "beta" in execution


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
