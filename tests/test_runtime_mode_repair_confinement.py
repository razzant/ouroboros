"""Repair-mode confinement against the bucket+skill_name short form.

Split verbatim out of ``tests/test_runtime_mode_core.py`` by theme. This module owns
the precedence rule that a real skill_repair task_constraint wins, the cross-skill
redirect refused before any resolution, the redundant matching pair, and the explicit
repo/data paths that win over a stale bucket+skill_name.
"""

from __future__ import annotations


import pytest

from ouroboros.tools.registry import ToolRegistry

from tests._runtime_mode_core_shared import _git_repo, _make_skill_payload, _registry


# ===========================================================================
# Repair-mode confinement vs bucket+skill_name short-form (v5.16.0-rc.1
# adversarial-review round 1 finding: three independent critics flagged a
# cross-skill escape where an agent in heal mode for skill A could pass
# bucket+skill_name args pointing at skill B and have the synthesized
# constraint override the real heal task_constraint. These tests pin the
# precedence rule: real skill_repair task_constraint wins, mismatched
# bucket+skill_name args return ⚠️ SKILL_REDIRECT_BLOCKED before any
# resolution happens.)
# ===========================================================================


def _ctx_with_skill_repair(tmp_path, skill_name: str, bucket: str = "external"):
    """Build a minimal ToolRegistry whose ctx already carries a skill_repair
    task_constraint for ``skill_name``. Returns the registry."""
    from ouroboros.contracts.task_constraint import TaskConstraint

    reg = _registry(tmp_path)
    reg._ctx.task_constraint = TaskConstraint(
        mode="skill_repair",
        skill_name=skill_name,
        payload_root=f"skills/{bucket}/{skill_name}",
    )
    # X3/F8: a repair TASK writes only under its admission binding (the promote
    # seam records one for every real repair, and a repair without one is typed
    # STALE rather than silently unverified). Mint the same binding here so these
    # tests keep exercising runtime-mode routing rather than the CAS gate.
    payload_dir = tmp_path / "skills" / bucket / skill_name
    if payload_dir.is_dir():
        from ouroboros.skill_loader import compute_content_hash
        from ouroboros.skill_repair_admission import record_repair_admission

        reg._ctx.task_id = str(getattr(reg._ctx, "task_id", "") or "repair-runtime-mode-test")
        record_repair_admission(
            tmp_path, skill_name, task_id=reg._ctx.task_id,
            base_content_hash=compute_content_hash(payload_dir),
        )
    return reg


@pytest.mark.parametrize("tool_name,extra_args", [
    ("write_file", {"path": "plugin.py", "content": "evil-payload\n"}),
    ("edit_text", {"path": "plugin.py", "old_str": "x", "new_str": "y"}),
    ("write_file", {"root": "skill_payload", "path": "plugin.py", "content": "evil-payload\n"}),
])
def test_repair_mode_blocks_cross_skill_redirect_via_bucket_skill_name(
    tool_name, extra_args, tmp_path, monkeypatch
):
    """If a heal task is active for alpha and the agent passes
    bucket+skill_name args naming a different skill bravo, the call must NOT
    silently write into bravo's payload. SKILL_REDIRECT_BLOCKED is the
    intended failure mode (registry-level + handler-level defense-in-depth)."""
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    _make_skill_payload(tmp_path, "external", "alpha")
    _make_skill_payload(tmp_path, "external", "bravo")
    reg = _ctx_with_skill_repair(tmp_path, "alpha")

    args = dict(extra_args)
    args["bucket"] = "external"
    args["skill_name"] = "bravo"
    result = reg.execute(tool_name, args)

    assert "SKILL_REDIRECT_BLOCKED" in result, (
        f"expected SKILL_REDIRECT_BLOCKED for {tool_name} with cross-skill "
        f"bucket+skill_name args under active skill_repair; got: {result[:200]}"
    )
    # Bravo's payload must remain untouched.
    bravo_plugin = tmp_path / "skills" / "external" / "bravo" / "plugin.py"
    assert not bravo_plugin.exists() or bravo_plugin.read_text(encoding="utf-8") == "def register(api):\n    pass\n", (
        f"unexpected write to bravo's payload: {bravo_plugin.read_text(encoding='utf-8')[:200]}"
    )


def test_repair_mode_matching_bucket_skill_name_is_silently_redundant(tmp_path, monkeypatch):
    """When bucket+skill_name match the active skill_repair task_constraint
    they are redundant but not erroneous — the call proceeds via the real TC,
    no SKILL_REDIRECT_BLOCKED. Real TC stays authoritative."""
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    _make_skill_payload(tmp_path, "external", "alpha")
    reg = _ctx_with_skill_repair(tmp_path, "alpha")

    result = reg.execute(
        "write_file",
        {
            "root": "skill_payload",
            "path": "extra.py",
            "content": "x\n",
            "bucket": "external",
            "skill_name": "alpha",
        },
    )

    assert "SKILL_REDIRECT_BLOCKED" not in result, result[:200]
    assert "DATA_WRITE_ERROR" not in result, result[:200]
    landed = tmp_path / "skills" / "external" / "alpha" / "extra.py"
    assert landed.is_file(), f"expected file at {landed}; got result={result[:200]}"


def test_synthesize_payload_constraint_unit():
    """Direct contract on the synthesis helper. Covers every branch so callers
    can rely on None == 'no short-form payload context'."""
    from ouroboros.contracts.skill_payload_policy import (
        SKILL_PAYLOAD_BUCKETS,
        synthesize_payload_constraint,
    )

    # Happy path — every allowed bucket.
    for bucket in SKILL_PAYLOAD_BUCKETS:
        tc = synthesize_payload_constraint(bucket, "weather")
        assert tc is not None
        assert tc.mode == "skill_repair"
        assert tc.skill_name == "weather"
        assert tc.payload_root == f"skills/{bucket}/weather"

    # Native is excluded — launcher seed update lane stays authoritative.
    assert synthesize_payload_constraint("native", "anything") is None

    # Unknown bucket.
    assert synthesize_payload_constraint("notabucket", "weather") is None

    # Empty / whitespace inputs.
    assert synthesize_payload_constraint("", "weather") is None
    assert synthesize_payload_constraint("external", "") is None
    assert synthesize_payload_constraint("   ", "weather") is None

    # Name that sanitizes away to nothing.
    assert synthesize_payload_constraint("external", "....") is None
    assert synthesize_payload_constraint("external", "/") is None
    assert synthesize_payload_constraint("external", "__omit__") is None

    # Sanitizer normalises odd input but still returns a usable constraint.
    tc = synthesize_payload_constraint("external", "weather/v2")
    assert tc is not None and tc.skill_name == "weather_v2"


def test_repo_path_wins_over_stale_bucket_skill_name(tmp_path, monkeypatch):
    repo = _git_repo(tmp_path)
    drive = tmp_path / "drive"
    (drive / "skills" / "external" / "alpha").mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    reg = ToolRegistry(repo_dir=repo, drive_root=drive)

    result = reg.execute(
        "edit_text",
        {
            "path": "README.md",
            "old_str": "ok",
            "new_str": "repo-ok",
            "bucket": "external",
            "skill_name": "alpha",
        },
    )

    assert "Replaced" in result, result[:300]
    assert "SKILL_SHORT_FORM_IGNORED" in result
    assert (repo / "README.md").read_text(encoding="utf-8") == "repo-ok\n"
    assert not (drive / "skills" / "external" / "alpha" / "README.md").exists()


def test_data_settings_path_wins_over_stale_bucket_skill_name(tmp_path, monkeypatch):
    from ouroboros import config as cfg

    drive = tmp_path / "drive"
    repo = tmp_path / "repo"
    repo.mkdir()
    (drive / "skills" / "external" / "alpha").mkdir(parents=True)
    (drive / "settings.json").write_text('{"TOTAL_BUDGET": 10}\n', encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr(cfg, "DATA_DIR", drive)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", drive / "settings.json")
    reg = ToolRegistry(repo_dir=repo, drive_root=drive)

    result = reg.execute(
        "write_file",
        {
            "root": "runtime_data",
            "path": "settings.json",
            "content": "{}\n",
            "bucket": "external",
            "skill_name": "alpha",
        },
    )

    assert "DATA_WRITE_BLOCKED" in result, result[:300]
    assert not (drive / "skills" / "external" / "alpha" / "settings.json").exists()
    assert (drive / "settings.json").read_text(encoding="utf-8") == '{"TOTAL_BUDGET": 10}\n'


def test_data_settings_case_variant_wins_over_stale_bucket_skill_name(tmp_path, monkeypatch):
    from ouroboros import config as cfg

    drive = tmp_path / "drive"
    repo = tmp_path / "repo"
    repo.mkdir()
    (drive / "skills" / "external" / "alpha").mkdir(parents=True)
    (drive / "settings.json").write_text('{"TOTAL_BUDGET": 10}\n', encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr(cfg, "DATA_DIR", drive)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", drive / "settings.json")
    reg = ToolRegistry(repo_dir=repo, drive_root=drive)

    result = reg.execute(
        "write_file",
        {
            "root": "runtime_data",
            "path": "Settings.json",
            "content": "{}\n",
            "bucket": "external",
            "skill_name": "alpha",
        },
    )

    assert "DATA_WRITE_BLOCKED" in result, result[:300]
    assert not (drive / "skills" / "external" / "alpha" / "Settings.json").exists()


def test_explicit_data_skills_path_wins_over_stale_bucket_skill_name(tmp_path, monkeypatch):
    drive = tmp_path / "drive"
    repo = tmp_path / "repo"
    repo.mkdir()
    skill = drive / "skills" / "external" / "alpha"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# alpha\n", encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    reg = ToolRegistry(repo_dir=repo, drive_root=drive)

    result = reg.execute(
        "write_file",
        {
            "root": "runtime_data",
            "path": "data/skills/external/alpha/plugin.py",
            "content": "VALUE = 1\n",
            "bucket": "external",
            "skill_name": "alpha",
        },
    )

    assert "DATA_WRITE_ERROR" not in result, result[:300]
    assert "SKILL_SHORT_FORM_IGNORED" not in result
    assert (skill / "plugin.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert not (drive / "data" / "skills" / "external" / "alpha" / "plugin.py").exists()


def test_short_form_requires_existing_payload_root(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    reg = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "drive")
    (tmp_path / "repo").mkdir()

    result = reg.execute(
        "edit_text",
        {
            "path": "plugin.py",
            "old_str": "x",
            "new_str": "y",
            "bucket": "external",
            "skill_name": "ghost",
        },
    )

    assert "skill payload not found" in result, result[:300]


def test_cross_skill_redirect_error_unit():
    """The helper that produces SKILL_REDIRECT_BLOCKED text. Empty string means
    'no conflict, proceed'; non-empty means 'reject the call'."""
    from ouroboros.contracts.skill_payload_policy import (
        cross_skill_redirect_error,
        synthesize_payload_constraint,
    )
    from ouroboros.contracts.task_constraint import TaskConstraint

    alpha_tc = TaskConstraint(
        mode="skill_repair", skill_name="alpha", payload_root="skills/external/alpha"
    )
    bravo_synth = synthesize_payload_constraint("external", "bravo")
    alpha_synth = synthesize_payload_constraint("external", "alpha")

    # Mismatched names → non-empty redirect message.
    err = cross_skill_redirect_error(alpha_tc, bravo_synth)
    assert err and "alpha" in err and "bravo" in err

    # Matching names → empty (redundant, not erroneous).
    assert cross_skill_redirect_error(alpha_tc, alpha_synth) == ""

    # No active TC → no redirect possible.
    assert cross_skill_redirect_error(None, bravo_synth) == ""

    # No synth → nothing to redirect.
    assert cross_skill_redirect_error(alpha_tc, None) == ""

    # Existing TC of a different mode (hypothetical future) → not skill_repair,
    # so no confinement to enforce here.
    other_mode = TaskConstraint(mode="other", skill_name="alpha", payload_root="skills/external/alpha")
    assert cross_skill_redirect_error(other_mode, bravo_synth) == ""
