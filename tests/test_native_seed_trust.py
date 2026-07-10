"""Block 4a/4b: launcher-seeded native skills get a hash-pinned trust verdict.

The payload bytes shipped through the repo commit gate, so the LAUNCHER (and
only the launcher, at the moments it writes the payload) stamps review.json
with status=clean bound to the post-seed content hash. Zero-grant skills also
auto-enable. CHECKLISTS.md documents this as a named, hash-pinned, audited
exception; these tests pin the mechanics.
"""

from __future__ import annotations

import logging
import pathlib

import pytest

from ouroboros.launcher_bootstrap import (
    _migrate_control_file_hashes,
    _seed_skills_into,
    _stamp_native_seed_trust,
)
from ouroboros.skill_loader import (
    HASH_EXEMPT_CONTROL_FILENAMES,
    compute_content_hash,
    load_enabled,
    load_review_state,
    load_skill,
    save_review_state,
)

log = logging.getLogger(__name__)


def _write_skill(root: pathlib.Path, name: str, *, permissions: str) -> pathlib.Path:
    skill = root / name
    skill.mkdir(parents=True, exist_ok=True)
    (skill / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: test skill\n"
        "version: 0.1.0\n"
        "type: extension\n"
        "entry: plugin.py\n"
        f"permissions: {permissions}\n"
        "env_from_settings: []\n"
        "when_to_use: testing\n"
        "---\n\n# Test\n",
        encoding="utf-8",
    )
    (skill / "plugin.py").write_text(
        "def register(api):\n    pass\n", encoding="utf-8"
    )
    return skill


@pytest.fixture()
def seed_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS", "true")
    seed_dir = tmp_path / "repo_skills"
    data_root = tmp_path / "data"
    target_root = data_root / "skills"
    target_root.mkdir(parents=True)
    # get_trust_native_seeded_skills() reads the on-disk SETTINGS_PATH BEFORE the
    # env var, so a settings.json written by another test into the shared run-level
    # data dir would override this test's env opt-in/out. Pin SETTINGS_PATH to a
    # per-test (non-existent) path so the env flag deterministically governs trust.
    monkeypatch.setattr("ouroboros.config.SETTINGS_PATH", data_root / "settings.json")
    return seed_dir, target_root, data_root


def test_fresh_bootstrap_stamps_trust_and_auto_enables_zero_grant(seed_env):
    seed_dir, target_root, data_root = seed_env
    _write_skill(seed_dir, "zero_grant_ext", permissions="[tool, subprocess]")

    copied = _seed_skills_into(seed_dir, target_root, log)

    assert copied == 1
    dest = target_root / "native" / "zero_grant_ext"
    assert (dest / ".seed-origin").is_file()
    review = load_review_state(data_root, "zero_grant_ext", skill_dir=dest)
    assert review.status == "clean"
    assert review.review_profile == "native_seed"
    assert review.reviewer_models == ["repo_commit_gate"]
    # Hash pinned to the POST-seed payload (control files excluded).
    skill = load_skill(dest, data_root)
    assert review.content_hash == skill.content_hash
    assert review.is_stale_for(skill.content_hash) is False
    # Zero-grant (tool/subprocess only) => auto-enabled.
    assert load_enabled(data_root, "zero_grant_ext") is True


def test_externally_facing_permissions_stay_disabled(seed_env):
    """net/route/widget surface skills get the trust verdict but NOT auto-enable."""
    seed_dir, target_root, data_root = seed_env
    _write_skill(seed_dir, "weather_like", permissions="[net, tool, route, widget]")

    assert _seed_skills_into(seed_dir, target_root, log) == 1
    review = load_review_state(
        data_root, "weather_like", skill_dir=target_root / "native" / "weather_like"
    )
    assert review.status == "clean"
    assert load_enabled(data_root, "weather_like") is False


def test_flag_off_skips_stamp_entirely(seed_env, monkeypatch):
    seed_dir, target_root, data_root = seed_env
    monkeypatch.setenv("OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS", "false")
    _write_skill(seed_dir, "zero_grant_ext", permissions="[tool, subprocess]")

    assert _seed_skills_into(seed_dir, target_root, log) == 1
    review = load_review_state(data_root, "zero_grant_ext")
    assert review.status != "clean"
    assert load_enabled(data_root, "zero_grant_ext") is False


def test_edit_after_stamp_goes_stale(seed_env):
    """The trust verdict is hash-pinned: editing the payload flips it stale."""
    seed_dir, target_root, data_root = seed_env
    _write_skill(seed_dir, "zero_grant_ext", permissions="[tool, subprocess]")
    _seed_skills_into(seed_dir, target_root, log)
    dest = target_root / "native" / "zero_grant_ext"

    (dest / "plugin.py").write_text(
        "def register(api):\n    api.evil()\n", encoding="utf-8"
    )

    review = load_review_state(data_root, "zero_grant_ext")
    skill = load_skill(dest, data_root)
    assert review.is_stale_for(skill.content_hash) is True


def test_seed_origin_removal_invalidates_native_seed_verdict(seed_env):
    """Provenance binding: a native_seed trust verdict must NOT survive
    .seed-origin removal (the skill is reclassified user-managed and the
    launcher-trust verdict reads back as pending/non-executable)."""
    seed_dir, target_root, data_root = seed_env
    _write_skill(seed_dir, "zero_grant_ext", permissions="[tool, subprocess]")
    _seed_skills_into(seed_dir, target_root, log)
    dest = target_root / "native" / "zero_grant_ext"

    skill_before = load_skill(dest, data_root)
    assert skill_before.review.status == "clean"

    (dest / ".seed-origin").unlink()

    skill_after = load_skill(dest, data_root)
    assert skill_after.review.status != "clean"
    assert skill_after.review.content_hash == ""  # reads back as no review at all
    assert skill_after.review.is_stale_for(skill_after.content_hash) is True


def test_control_files_excluded_from_hash():
    """Writing .seed-origin must not change the payload hash (4a hardening)."""
    assert ".seed-origin" in HASH_EXEMPT_CONTROL_FILENAMES


def test_control_file_hash_values(tmp_path):
    # The exemption applies only to NATIVE-bucket payloads (launcher-owned).
    skill = _write_skill(tmp_path / "native", "hash_probe", permissions="[tool]")
    before = compute_content_hash(skill, manifest_entry="plugin.py")
    (skill / ".seed-origin").write_text("seeded_from=test\n", encoding="utf-8")
    after = compute_content_hash(skill, manifest_entry="plugin.py")
    assert before == after
    legacy = compute_content_hash(
        skill, manifest_entry="plugin.py", include_control_files=True
    )
    assert legacy != after


def test_nested_seed_origin_stays_in_reviewed_surface(tmp_path):
    """Only the TOP-LEVEL lifecycle marker is hash-exempt: a nested file that
    merely shares the name is runtime payload and must affect the hash."""
    skill = _write_skill(tmp_path / "native", "nested_probe", permissions="[tool]")
    before = compute_content_hash(skill, manifest_entry="plugin.py")
    nested = skill / "data"
    nested.mkdir()
    (nested / ".seed-origin").write_text("sneaky runtime-readable bytes\n", encoding="utf-8")
    after = compute_content_hash(skill, manifest_entry="plugin.py")
    assert before != after


def test_non_native_seed_origin_stays_in_reviewed_surface(tmp_path):
    """P3: outside the native bucket a top-level .seed-origin is ordinary
    runtime-reachable payload — it must stay in the review hash."""
    skill = _write_skill(tmp_path / "external", "ext_probe", permissions="[tool]")
    before = compute_content_hash(skill, manifest_entry="plugin.py")
    (skill / ".seed-origin").write_text("runtime-readable bytes\n", encoding="utf-8")
    after = compute_content_hash(skill, manifest_entry="plugin.py")
    assert before != after


def test_legacy_hash_migration_repins_unedited_payload(tmp_path, monkeypatch):
    """A review pinned to the legacy (control-file-included) hash re-pins to
    the new hash when the payload is byte-identical; edited payloads stay stale."""
    monkeypatch.setenv("OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS", "true")
    data_root = tmp_path / "data"
    target_root = data_root / "skills"
    native = target_root / "native"
    skill = _write_skill(native, "legacy_skill", permissions="[tool]")
    (skill / ".seed-origin").write_text("seeded_from=test\n", encoding="utf-8")

    legacy_hash = compute_content_hash(
        skill, manifest_entry="plugin.py", include_control_files=True
    )
    loaded = load_skill(skill, data_root)
    review = loaded.review
    review.status = "clean"
    review.content_hash = legacy_hash
    save_review_state(data_root, "legacy_skill", review)

    _migrate_control_file_hashes(target_root, log)

    migrated = load_review_state(data_root, "legacy_skill")
    fresh = load_skill(skill, data_root)
    assert migrated.content_hash == fresh.content_hash
    assert migrated.is_stale_for(fresh.content_hash) is False

    # An EDITED payload must not be repinned by the migration.
    (skill / "plugin.py").write_text("def register(api):\n    api.x()\n", encoding="utf-8")
    _migrate_control_file_hashes(target_root, log)
    edited = load_review_state(data_root, "legacy_skill")
    fresh_edited = load_skill(skill, data_root)
    assert edited.is_stale_for(fresh_edited.content_hash) is True


def test_resync_restamp_preserves_explicit_owner_disable(seed_env):
    """Owner sovereignty (B-fix): a version-resync re-stamp must refresh the
    trust verdict but NEVER override an explicit owner disable."""
    from ouroboros.skill_loader import save_enabled

    seed_dir, target_root, data_root = seed_env
    _write_skill(seed_dir, "zero_grant_ext", permissions="[tool, subprocess]")
    _seed_skills_into(seed_dir, target_root, log)
    assert load_enabled(data_root, "zero_grant_ext") is True

    # Owner explicitly disables; a later launcher re-stamp must respect it.
    save_enabled(data_root, "zero_grant_ext", False)
    dest = target_root / "native" / "zero_grant_ext"
    _stamp_native_seed_trust(data_root, dest, log)

    assert load_enabled(data_root, "zero_grant_ext") is False
    # The trust verdict itself is still refreshed.
    review = load_review_state(data_root, "zero_grant_ext", skill_dir=dest)
    assert review.status == "clean"


def test_stamp_helper_ignores_broken_skill(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS", "true")
    data_root = tmp_path / "data"
    broken = tmp_path / "data" / "skills" / "native" / "broken"
    broken.mkdir(parents=True)
    (broken / "SKILL.md").write_text("---\nname: [\n---\n", encoding="utf-8")

    _stamp_native_seed_trust(data_root, broken, log)  # must not raise

    review = load_review_state(data_root, "broken")
    assert review.status != "clean"


def test_bundled_unix_computer_use_net_permission_needs_no_grants_but_owner_enable():
    """The shipped skill declares `net` for its remote backends. `net` needs no
    owner grant (grants only gate inject_chat/subscribe_event), but it DOES remove
    the skill from the launcher's native auto-enable class — the owner (or a bench
    runner via save_enabled) must enable it explicitly. This is deliberate."""
    from ouroboros.launcher_bootstrap import _NATIVE_AUTO_ENABLE_EXEMPT_PERMISSIONS
    from ouroboros.skill_loader import requested_skill_permissions

    manifest = (
        pathlib.Path(__file__).resolve().parents[1]
        / "skills" / "unix_computer_use" / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert "permissions: [tool, subprocess, net]" in manifest
    assert "env_from_settings: []" in manifest
    # net requires no owner grant...
    assert requested_skill_permissions(["tool", "subprocess", "net"], []) == []
    # ...but is NOT in the auto-enable-exempt set (so the skill needs owner enable).
    assert "net" not in _NATIVE_AUTO_ENABLE_EXEMPT_PERMISSIONS
