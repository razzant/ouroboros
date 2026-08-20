"""Skill grants: what a grant is bound to, and when it is issued automatically.

Split out of ``tests/test_skill_loader.py`` by theme: the content and request a grant is bound
to, the extension and privileged permissions the status supports, the auto-grant outcome that
carries the request even when the toggle is off, the executable review gate it uses, the
partial approvals a save merges, and the instruction skills grants do not apply to.
"""

from __future__ import annotations


def test_skill_grants_are_content_and_request_bound(tmp_path):
    from ouroboros.contracts.skill_manifest import SkillManifest
    from ouroboros.skill_loader import (
        LoadedSkill,
        SkillReviewState,
        grant_status_for_skill,
        save_skill_grants,
    )

    drive_root = tmp_path / "drive"
    skill_dir = tmp_path / "skill"
    drive_root.mkdir()
    skill_dir.mkdir()
    manifest = SkillManifest(
        name="granty",
        description="grant test",
        version="0.1",
        type="script",
        env_from_settings=["OPENROUTER_API_KEY"],
    )
    skill = LoadedSkill(
        name="granty",
        skill_dir=skill_dir,
        manifest=manifest,
        content_hash="hash-a",
        review=SkillReviewState(status="pass", content_hash="hash-a"),
    )
    save_skill_grants(
        drive_root,
        "granty",
        ["OPENROUTER_API_KEY", "GITHUB_TOKEN"],
        content_hash="hash-a",
        requested_keys=["OPENROUTER_API_KEY"],
    )
    status = grant_status_for_skill(drive_root, skill)
    assert status["granted_keys"] == ["OPENROUTER_API_KEY"]
    assert status["all_granted"] is True
    skill.content_hash = "hash-b"
    stale = grant_status_for_skill(drive_root, skill)
    assert stale["granted_keys"] == []
    assert stale["missing_keys"] == ["OPENROUTER_API_KEY"]

    skill.content_hash = "hash-a"
    skill.source = "clawhub"
    unsupported = grant_status_for_skill(drive_root, skill)
    assert unsupported["unsupported_for_skill_type"] is False
    assert unsupported["usable"] is True
    assert unsupported["granted_keys"] == ["OPENROUTER_API_KEY"]


def test_grant_status_supports_extension_skills(tmp_path):
    """v5.2.2 dual-track grants: ``type: extension`` skills are now
    eligible for owner core-key grants alongside ``type: script``."""
    from ouroboros.contracts.skill_manifest import SkillManifest
    from ouroboros.skill_loader import (
        LoadedSkill,
        SkillReviewState,
        grant_status_for_skill,
        save_skill_grants,
    )

    drive_root = tmp_path / "drive"
    skill_dir = tmp_path / "ext"
    drive_root.mkdir()
    skill_dir.mkdir()
    manifest = SkillManifest(
        name="ext_grant",
        description="extension grant test",
        version="0.1",
        type="extension",
        env_from_settings=["OPENROUTER_API_KEY"],
        permissions=["read_settings"],
    )
    skill = LoadedSkill(
        name="ext_grant",
        skill_dir=skill_dir,
        manifest=manifest,
        content_hash="ext-hash",
        review=SkillReviewState(status="pass", content_hash="ext-hash"),
    )
    no_grant = grant_status_for_skill(drive_root, skill)
    assert no_grant["unsupported_for_skill_type"] is False
    assert no_grant["all_granted"] is False
    assert no_grant["missing_keys"] == ["OPENROUTER_API_KEY"]

    save_skill_grants(
        drive_root,
        "ext_grant",
        ["OPENROUTER_API_KEY"],
        content_hash="ext-hash",
        requested_keys=["OPENROUTER_API_KEY"],
    )
    granted = grant_status_for_skill(drive_root, skill)
    assert granted["unsupported_for_skill_type"] is False
    assert granted["all_granted"] is True
    assert granted["usable"] is True
    assert granted["granted_keys"] == ["OPENROUTER_API_KEY"]


def test_grant_status_supports_privileged_permissions(tmp_path):
    from ouroboros.contracts.skill_manifest import SkillManifest
    from ouroboros.skill_loader import (
        LoadedSkill,
        SkillReviewState,
        grant_status_for_skill,
        save_skill_grants,
    )

    drive_root = tmp_path / "drive"
    skill_dir = tmp_path / "ext"
    drive_root.mkdir()
    skill_dir.mkdir()
    manifest = SkillManifest(
        name="injector",
        description="inject grant test",
        version="0.1",
        type="extension",
        permissions=["inject_chat", "subscribe_event"],
        subscribe_events=["chat.outbound"],
    )
    skill = LoadedSkill(
        name="injector",
        skill_dir=skill_dir,
        manifest=manifest,
        content_hash="inject-hash",
        review=SkillReviewState(status="pass", content_hash="inject-hash"),
    )

    missing = grant_status_for_skill(drive_root, skill)
    assert missing["missing_permissions"] == ["inject_chat", "subscribe_event:chat.outbound"]
    assert missing["usable"] is False

    save_skill_grants(
        drive_root,
        "injector",
        [],
        content_hash="inject-hash",
        requested_keys=[],
        granted_permissions=["inject_chat", "subscribe_event:chat.outbound"],
        requested_permissions=["inject_chat", "subscribe_event:chat.outbound"],
    )
    granted = grant_status_for_skill(drive_root, skill)
    assert granted["all_granted"] is True
    assert granted["usable"] is True
    assert granted["granted_permissions"] == ["inject_chat", "subscribe_event:chat.outbound"]


def test_auto_grant_if_enabled_returns_outcome_with_requested_even_when_off(tmp_path, monkeypatch):
    import ouroboros.config as config
    from ouroboros.contracts.skill_manifest import SkillManifest
    from ouroboros.skill_loader import (
        LoadedSkill,
        SkillReviewState,
        auto_grant_if_enabled,
        load_skill_grants,
    )

    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "missing-settings.json")
    monkeypatch.setenv("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", "false")
    drive_root = tmp_path / "drive"
    skill_dir = tmp_path / "skill"
    drive_root.mkdir()
    skill_dir.mkdir()
    skill = LoadedSkill(
        name="auto",
        skill_dir=skill_dir,
        manifest=SkillManifest(
            name="auto",
            description="auto grant test",
            version="0.1",
            type="extension",
            env_from_settings=["OPENROUTER_API_KEY"],
            permissions=["inject_chat"],
        ),
        content_hash="hash-a",
        review=SkillReviewState(status="pass", content_hash="hash-a"),
    )

    outcome = auto_grant_if_enabled(drive_root, skill)

    assert outcome.granted is False
    assert outcome.requested_keys == ["OPENROUTER_API_KEY"]
    assert outcome.requested_permissions == ["inject_chat"]
    assert outcome.granted_keys == []
    assert outcome.granted_permissions == []
    assert load_skill_grants(drive_root, "auto")["granted_keys"] == []


def test_auto_grant_if_enabled_marks_granted_when_toggle_on(tmp_path, monkeypatch):
    import ouroboros.config as config
    from ouroboros.contracts.skill_manifest import SkillManifest
    from ouroboros.skill_loader import (
        LoadedSkill,
        SkillReviewState,
        auto_grant_if_enabled,
        load_skill_grants,
    )

    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "missing-settings.json")
    monkeypatch.setenv("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", "true")
    drive_root = tmp_path / "drive"
    skill_dir = tmp_path / "skill"
    drive_root.mkdir()
    skill_dir.mkdir()
    skill = LoadedSkill(
        name="auto",
        skill_dir=skill_dir,
        manifest=SkillManifest(
            name="auto",
            description="auto grant test",
            version="0.1",
            type="extension",
            env_from_settings=["OPENROUTER_API_KEY"],
            permissions=["inject_chat"],
        ),
        content_hash="hash-a",
        review=SkillReviewState(status="pass", content_hash="hash-a"),
    )

    outcome = auto_grant_if_enabled(drive_root, skill)

    assert outcome.granted is True
    assert outcome.requested_keys == ["OPENROUTER_API_KEY"]
    assert outcome.granted_keys == ["OPENROUTER_API_KEY"]
    assert outcome.requested_permissions == ["inject_chat"]
    assert outcome.granted_permissions == ["inject_chat"]
    grants = load_skill_grants(drive_root, "auto")
    assert grants["granted_keys"] == ["OPENROUTER_API_KEY"]
    assert grants["granted_permissions"] == ["inject_chat"]


def test_auto_grant_if_enabled_uses_executable_review_gate(tmp_path, monkeypatch):
    import ouroboros.config as config
    from ouroboros.contracts.skill_manifest import SkillManifest
    from ouroboros.skill_loader import (
        LoadedSkill,
        SkillReviewState,
        auto_grant_if_enabled,
        load_skill_grants,
    )

    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "missing-settings.json")
    monkeypatch.setenv("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", "true")
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    drive_root = tmp_path / "drive"
    skill_dir = tmp_path / "skill"
    drive_root.mkdir()
    skill_dir.mkdir()
    skill = LoadedSkill(
        name="auto_blocked",
        skill_dir=skill_dir,
        manifest=SkillManifest(
            name="auto_blocked",
            description="auto grant blocker test",
            version="0.1",
            type="extension",
            env_from_settings=["OPENROUTER_API_KEY"],
        ),
        content_hash="hash-a",
        review=SkillReviewState(status="blockers", content_hash="hash-a"),
    )

    outcome = auto_grant_if_enabled(drive_root, skill)

    assert outcome.granted is False
    assert outcome.requested_keys == ["OPENROUTER_API_KEY"]
    assert outcome.granted_keys == []
    assert load_skill_grants(drive_root, "auto_blocked")["granted_keys"] == []


def test_save_skill_grants_merges_partial_approvals(tmp_path):
    """A subsequent partial-key grant must not silently revoke
    previously-approved keys. The merge is bound to the same
    content_hash + requested_keys; any change to either resets the
    persisted state because the owner has not consented to the new
    shape yet."""
    from ouroboros.skill_loader import (
        load_skill_grants,
        save_skill_grants,
    )

    drive_root = tmp_path / "drive"
    drive_root.mkdir()

    save_skill_grants(
        drive_root,
        "merge_demo",
        ["OPENROUTER_API_KEY"],
        content_hash="hash-x",
        requested_keys=["OPENROUTER_API_KEY", "GITHUB_TOKEN"],
    )
    save_skill_grants(
        drive_root,
        "merge_demo",
        ["GITHUB_TOKEN"],
        content_hash="hash-x",
        requested_keys=["OPENROUTER_API_KEY", "GITHUB_TOKEN"],
    )
    after_merge = load_skill_grants(drive_root, "merge_demo")
    assert sorted(after_merge["granted_keys"]) == ["GITHUB_TOKEN", "OPENROUTER_API_KEY"]

    # New content hash invalidates the previous persisted state.
    save_skill_grants(
        drive_root,
        "merge_demo",
        ["OPENROUTER_API_KEY"],
        content_hash="hash-y",
        requested_keys=["OPENROUTER_API_KEY", "GITHUB_TOKEN"],
    )
    after_rotate = load_skill_grants(drive_root, "merge_demo")
    assert after_rotate["content_hash"] == "hash-y"
    assert after_rotate["granted_keys"] == ["OPENROUTER_API_KEY"]


def test_grant_status_unsupported_for_instruction_skills(tmp_path):
    """Instruction-type skills cannot receive core grants — they have
    no executable surface, so a grant would be meaningless."""
    from ouroboros.contracts.skill_manifest import SkillManifest
    from ouroboros.skill_loader import (
        LoadedSkill,
        SkillReviewState,
        grant_status_for_skill,
    )

    drive_root = tmp_path / "drive"
    skill_dir = tmp_path / "instr"
    drive_root.mkdir()
    skill_dir.mkdir()
    manifest = SkillManifest(
        name="instr_grant",
        description="instruction grant test",
        version="0.1",
        type="instruction",
        env_from_settings=["OPENROUTER_API_KEY"],
    )
    skill = LoadedSkill(
        name="instr_grant",
        skill_dir=skill_dir,
        manifest=manifest,
        content_hash="instr-hash",
        review=SkillReviewState(status="pass", content_hash="instr-hash"),
    )
    status = grant_status_for_skill(drive_root, skill)
    assert status["unsupported_for_skill_type"] is True
    assert status["all_granted"] is False
    assert status["usable"] is False
