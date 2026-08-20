"""Phase 3 regression tests for discovery and loading in ``ouroboros.skill_loader``.

Covers the data-plane buckets discovery reads, the location inventory and the selector that
checks identity before location, the enabled-skill conflicts, the manifests that parse and
the broken or unreadable ones that surface as load errors, the identity a loaded skill takes
from its directory, and the sanitized-name collisions that stay discoverable for repair. No
network, no real review calls — these tests stay hermetic against ``tmp_path``.

Content hashing, state persistence, availability and grants were split verbatim into
``tests/test_skill_content_hash.py``, ``tests/test_skill_state_persistence.py``,
``tests/test_skill_availability.py`` and ``tests/test_skill_grants.py``; the skill writer and
the valid manifest they share live in ``tests/_skill_loader_shared.py``.
"""

from __future__ import annotations

import json
import os

import pytest

from ouroboros.skill_loader import (
    LoadedSkill,
    _select_skill_location,
    _skill_location_inventory,
    discover_skills,
    enabled_skill_conflicts,
    find_skill,
    load_skill,
    save_enabled,
    skill_conflict_status,
)

from tests._skill_loader_shared import (
    _valid_script_manifest,
    _write_skill,
)


# ---------------------------------------------------------------------------
# Discovery + loading
# ---------------------------------------------------------------------------
def test_discover_skills_returns_empty_when_data_plane_missing(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    assert discover_skills(drive_root, repo_path="") == []
    # A missing path is also silently tolerated — same "no skills" signal.
    assert discover_skills(drive_root, repo_path=str(tmp_path / "does-not-exist")) == []


def test_discover_skills_uses_data_plane_native_bucket(tmp_path):
    drive_root = tmp_path / "drive"
    native_root = drive_root / "skills" / "native"
    _write_skill(
        native_root,
        "weather",
        manifest=_valid_script_manifest("weather"),
        scripts={"fetch.py": "print('ok')\n"},
    )
    (native_root / "weather" / ".seed-origin").write_text("seeded_from=test\n", encoding="utf-8")
    skills = discover_skills(drive_root, repo_path="")
    names = {s.name for s in skills}
    assert "weather" in names
    assert skills[0].source == "native"


@pytest.mark.parametrize("layout", ["direct", "flat", "grouped"])
def test_skill_location_inventory_supports_configured_repo_layouts(
    tmp_path, layout
):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    checkout = tmp_path / "checkout"
    if layout == "grouped":
        skill_dir = _write_skill(
            checkout / "group",
            "alpha",
            manifest=_valid_script_manifest("alpha"),
        )
        repo_path = checkout
    else:
        skill_dir = _write_skill(
            checkout,
            "alpha",
            manifest=_valid_script_manifest("alpha"),
        )
        repo_path = skill_dir if layout == "direct" else checkout

    candidates = _skill_location_inventory(drive_root, repo_path=str(repo_path))

    assert len(candidates) == 1
    assert candidates[0].name == "alpha"
    assert candidates[0].location == "user_repo"
    assert candidates[0].skill_dir == skill_dir.resolve()
    assert not (drive_root / "state").exists()


def test_skill_location_inventory_dedupes_and_prefers_data_location(tmp_path):
    drive_root = tmp_path / "drive"
    skill_dir = _write_skill(
        drive_root / "skills" / "external",
        "alpha",
        manifest=_valid_script_manifest("alpha"),
    )

    # The configured checkout overlaps the canonical data tree. The same
    # resolved package is inventoried once and keeps its data location.
    candidates = _skill_location_inventory(
        drive_root,
        repo_path=str(skill_dir.parent),
    )

    assert len(candidates) == 1
    assert candidates[0].skill_dir == skill_dir.resolve()
    assert candidates[0].location == "external"


def test_skill_location_selector_checks_identity_before_location(tmp_path):
    drive_root = tmp_path / "drive"
    data_skill = _write_skill(
        drive_root / "skills" / "external",
        "alpha",
        manifest=_valid_script_manifest("alpha"),
    )
    checkout = tmp_path / "checkout"
    repo_skill = _write_skill(
        checkout,
        "alpha",
        manifest=_valid_script_manifest("alpha"),
    )
    candidates = _skill_location_inventory(drive_root, repo_path=str(checkout))

    with pytest.raises(ValueError, match="Skill name collision"):
        _select_skill_location(
            candidates,
            name="alpha",
            location="external",
        )

    # Read/list/search callers may opt into exact-location selection without
    # erasing the collision evidence needed by mutation/lifecycle callers.
    assert _select_skill_location(
        candidates,
        name="alpha",
        location="external",
        require_unique_identity=False,
    ).skill_dir == data_skill.resolve()
    assert _select_skill_location(
        candidates,
        name="alpha",
        location="user_repo",
        require_unique_identity=False,
    ).skill_dir == repo_skill.resolve()

    with pytest.raises(ValueError, match="not in requested location"):
        _select_skill_location(
            candidates,
            name="alpha",
            location="native",
            require_unique_identity=False,
        )
    assert _select_skill_location(
        candidates,
        name="missing",
        location="external",
    ) is None


def test_skill_location_selector_rejects_same_location_ambiguity_for_reads(
    tmp_path,
):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    checkout = tmp_path / "checkout"
    _write_skill(
        checkout / "one",
        "alpha",
        manifest=_valid_script_manifest("alpha"),
    )
    _write_skill(
        checkout / "two",
        "alpha",
        manifest=_valid_script_manifest("alpha"),
    )
    candidates = _skill_location_inventory(drive_root, repo_path=str(checkout))

    with pytest.raises(ValueError, match="Skill location collision"):
        _select_skill_location(
            candidates,
            name="alpha",
            location="user_repo",
            require_unique_identity=False,
        )


@pytest.mark.parametrize("declaration_owner", ["telegram", "telegram-bridge"])
def test_enabled_skill_conflicts_are_symmetric_for_one_sided_declarations(
    tmp_path, declaration_owner
):
    drive_root = tmp_path / "drive"
    native_root = drive_root / "skills" / "native"
    for name, other in (("telegram", "telegram-bridge"), ("telegram-bridge", "telegram")):
        conflicts = f"conflicts: [{other}]\n" if name == declaration_owner else ""
        _write_skill(
            native_root,
            name,
            manifest=(
                "---\n"
                f"name: {name}\n"
                "description: Telegram fixture.\n"
                "version: 1.0.0\n"
                "type: instruction\n"
                f"{conflicts}"
                "---\n"
            ),
        )
    save_enabled(drive_root, "telegram-bridge", True)

    skills = discover_skills(drive_root, repo_path="")
    telegram = next(skill for skill in skills if skill.name == "telegram")
    assert enabled_skill_conflicts(telegram, skills) == ["telegram-bridge"]
    assert skill_conflict_status(telegram, skills) == {
        "code": "skill_conflict",
        "skills": ["telegram-bridge"],
        "omitted": 0,
    }
    from ouroboros.skill_readiness import skill_readiness_for_execution

    readiness = skill_readiness_for_execution(
        drive_root,
        telegram,
        require_enabled=False,
        require_grants=False,
        skills=skills,
    )
    assert readiness.conflict["code"] == "skill_conflict"
    assert any(item.startswith("skill_conflict:") for item in readiness.owner_action_blockers)


def test_missing_and_disabled_conflict_targets_are_inert(tmp_path):
    drive_root = tmp_path / "drive"
    native_root = drive_root / "skills" / "native"
    _write_skill(
        native_root,
        "telegram",
        manifest=(
            "---\nname: telegram\ndescription: Telegram\nversion: 1.0.0\n"
            "type: instruction\nconflicts: [telegram-bridge]\n---\n"
        ),
    )
    skills = discover_skills(drive_root, repo_path="")
    telegram = skills[0]
    assert enabled_skill_conflicts(telegram, skills) == []

    _write_skill(
        native_root,
        "telegram-bridge",
        manifest=(
            "---\nname: telegram-bridge\ndescription: Old\nversion: 1.0.0\n"
            "type: instruction\n---\n"
        ),
    )
    skills = discover_skills(drive_root, repo_path="")
    telegram = next(skill for skill in skills if skill.name == "telegram")
    assert enabled_skill_conflicts(telegram, skills) == []


def test_load_skill_parses_manifest_and_computes_hash(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    repo_root = tmp_path / "skills"
    _write_skill(
        repo_root,
        "weather",
        manifest=_valid_script_manifest(),
        scripts={"fetch.py": "print('hi')\n"},
    )
    loaded = load_skill(repo_root / "weather", drive_root)
    assert isinstance(loaded, LoadedSkill)
    assert loaded.name == "weather"
    assert loaded.manifest.type == "script"
    assert loaded.manifest.runtime == "python3"
    assert loaded.content_hash  # non-empty
    assert loaded.enabled is False  # default
    assert loaded.review.status == "pending"
    assert loaded.available_for_execution is False


def test_load_skill_returns_none_for_non_skill_dir(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    other = tmp_path / "random"
    other.mkdir()
    (other / "README.txt").write_text("hi", encoding="utf-8")
    assert load_skill(other, drive_root) is None


def test_load_skill_surfaces_broken_manifest(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    repo_root = tmp_path / "skills"
    _write_skill(
        repo_root,
        "broken",
        manifest='{"name": ',  # truncated JSON
        manifest_name="skill.json",
    )
    loaded = load_skill(repo_root / "broken", drive_root)
    assert loaded is not None
    assert loaded.load_error
    assert loaded.available_for_execution is False


def test_load_skill_surfaces_unreadable_manifest(tmp_path):
    """Phase 3 round 16 regression: an existing-but-unreadable manifest
    must surface as ``load_error`` instead of silently looking like
    "not a skill dir at all"."""
    import platform
    if platform.system() == "Windows":
        pytest.skip("chmod-based permission test not portable to Windows")
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    repo_root = tmp_path / "skills"
    skill_dir = _write_skill(
        repo_root,
        "unread",
        manifest=_valid_script_manifest("unread"),
    )
    manifest_path = skill_dir / "SKILL.md"
    original_mode = manifest_path.stat().st_mode
    os.chmod(manifest_path, 0o000)
    try:
        loaded = load_skill(skill_dir, drive_root)
    finally:
        os.chmod(manifest_path, original_mode)
    # Root users can read anything regardless of perms — skip the
    # assertion in that case (rare, but CI runners vary).
    if os.geteuid() == 0:  # pragma: no cover — only hit in root CI
        pytest.skip("root user bypasses 0o000 chmod, cannot trigger OSError")
    assert loaded is not None, "Unreadable manifest must still appear in discovery."
    assert loaded.load_error, "load_error should be populated for unreadable manifests."
    assert "unreadable" in loaded.load_error.lower()


def test_discover_skills_picks_up_multiple(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    repo_root = tmp_path / "skills"
    _write_skill(repo_root, "alpha", manifest=_valid_script_manifest("alpha"))
    _write_skill(repo_root, "beta", manifest=_valid_script_manifest("beta"))
    skills = discover_skills(drive_root, repo_path=str(repo_root))
    names = {s.name for s in skills}
    assert names == {"alpha", "beta"}
    assert {s.source for s in skills} == {"user_repo"}


def test_unique_candidate_keeps_rich_self_authored_provenance(tmp_path):
    drive_root = tmp_path / "drive"
    skill_dir = _write_skill(
        drive_root / "skills" / "external",
        "alpha",
        manifest=_valid_script_manifest("alpha"),
    )
    marker = {
        "schema_version": 1,
        "origin": "self_authored",
        "task_id": "task-1",
        "created_at": "2026-08-11T00:00:00Z",
    }
    (skill_dir / ".self_authored.json").write_text(
        json.dumps(marker),
        encoding="utf-8",
    )
    state_dir = drive_root / "state" / "skills" / "alpha"
    state_dir.mkdir(parents=True)
    (state_dir / "self_authored.json").write_text(
        json.dumps(marker),
        encoding="utf-8",
    )

    skills = discover_skills(drive_root, repo_path="")

    assert len(skills) == 1
    assert skills[0].source == "self_authored"
    assert skills[0].is_self_authored is True


def test_find_skill_returns_match_and_missing(tmp_path, monkeypatch):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    repo_root = tmp_path / "skills"
    _write_skill(repo_root, "alpha", manifest=_valid_script_manifest("alpha"))
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(repo_root))
    assert find_skill(drive_root, "alpha") is not None
    assert find_skill(drive_root, "does-not-exist") is None


def test_loaded_skill_identity_is_directory_basename_not_manifest_name(tmp_path):
    """Phase 3 round 9 regression: tool schemas advertise ``skill`` as
    the directory name in ``OUROBOROS_SKILLS_REPO_PATH``. ``LoadedSkill.name``
    + the durable state dir key MUST match that so ``skill_exec("weather")``
    resolves ``skills/weather/`` regardless of ``manifest.name`` free-form
    content (``Weather Skill``, localised label, etc.).
    """
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    repo_root = tmp_path / "skills"
    weird_manifest = (
        "---\n"
        "name: Weather Skill Display\n"
        "description: Check the weather.\n"
        "version: 0.1.0\n"
        "type: script\n"
        "runtime: python3\n"
        "timeout_sec: 30\n"
        "scripts:\n"
        "  - name: fetch.py\n"
        "---\n"
        "body\n"
    )
    _write_skill(
        repo_root,
        "weather",
        manifest=weird_manifest,
        scripts={"fetch.py": "print('ok')\n"},
    )
    loaded = find_skill(drive_root, "weather", repo_path=str(repo_root))
    assert loaded is not None
    assert loaded.name == "weather"
    # Manifest display name preserved as metadata.
    assert loaded.manifest.name == "Weather Skill Display"
    # Addressable by directory name, NOT by the sanitised manifest name.
    from ouroboros.skill_loader import _sanitize_skill_name as _sn
    assert _sn("Weather Skill Display") != loaded.name


def test_sanitized_name_collision_surfaces_as_load_error(tmp_path):
    """Phase 3 round 12 regression: ``skills/hello world/`` and
    ``skills/hello_world/`` both sanitise to the same identity. The
    loader must refuse to merge their state and surface a load_error
    on each collision member."""
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    repo_root = tmp_path / "skills"
    _write_skill(
        repo_root,
        "hello world",
        manifest=_valid_script_manifest("hello world"),
    )
    _write_skill(
        repo_root,
        "hello_world",
        manifest=_valid_script_manifest("hello_world"),
    )
    skills = discover_skills(drive_root, repo_path=str(repo_root))
    assert len(skills) == 2
    for s in skills:
        assert s.load_error
        assert "name collision" in s.load_error.lower()
        assert s.identity_collision is True
        assert s.available_for_execution is False
    assert not (drive_root / "state").exists()


def test_collision_discovery_and_summary_do_not_touch_payload_or_state(
    tmp_path, monkeypatch
):
    import ouroboros.skill_loader as loader

    drive_root = tmp_path / "drive"
    _write_skill(
        drive_root / "skills" / "external",
        "alpha",
        manifest=_valid_script_manifest("alpha"),
    )
    checkout = tmp_path / "checkout"
    _write_skill(
        checkout,
        "alpha",
        manifest=_valid_script_manifest("alpha"),
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(checkout))

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("collision path touched rich payload/provenance state")

    monkeypatch.setattr(loader, "load_skill", _unexpected)
    monkeypatch.setattr(loader, "_classify_skill_source", _unexpected)
    monkeypatch.setattr(loader, "is_self_authored_skill_dir", _unexpected)
    monkeypatch.setattr(loader, "load_enabled", _unexpected)
    monkeypatch.setattr(loader, "load_review_state", _unexpected)
    skills = loader.discover_skills(drive_root)
    summary = loader.summarize_skills(drive_root)

    assert len(skills) == 2
    assert all(skill.identity_collision for skill in skills)
    assert summary["count"] == 2
    assert summary["broken"] == 2
    assert summary["blocked_by_grants"] == 0
    assert all(not row["blocked_by_grants"] for row in summary["skills"])
    assert not (drive_root / "state").exists()


def test_unique_broken_manifest_remains_discoverable_for_repair(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    checkout = tmp_path / "checkout"
    skill_dir = _write_skill(
        checkout,
        "broken",
        manifest='{"name": ',
        manifest_name="skill.json",
    )

    skills = discover_skills(drive_root, repo_path=str(checkout))

    assert len(skills) == 1
    assert skills[0].skill_dir == skill_dir.resolve()
    assert skills[0].identity_collision is False
    assert "manifest parse error" in skills[0].load_error.lower()
