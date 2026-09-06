"""v5 Cycle 1 GPT critic Finding 4 — direct coverage for the
``_per_skill_version_resync`` and ``_read_skill_manifest_version``
helpers.

The previous v4.50 RC chain landed the version-aware resync to make
the v4-script-weather → v5-extension-weather upgrade deterministic,
but every test exercised the resync only INDIRECTLY via the
``ensure_data_skills_seeded`` entry point. The critic identified six
specific invariants the resync code must hold:

    a. skip when target absent (the resurrection-after-deletion path);
    b. skip when no ``.seed-origin`` marker (user-managed skill protection);
    c. reseed on drift + state-dir survival + user-mod files inside skill dir wiped;
    d. noop on identical version;
    e. accept downgrade (launcher-owned-by-design — pin this);
    f. ``_read_skill_manifest_version`` handles inline comments,
       single-line JSON, and pre-frontmatter ``version:`` lines correctly
       (delegated to the shared parser via the v5 fix).

This module pins all six.
"""

from __future__ import annotations

import logging
import pathlib
import shutil
import textwrap
from typing import Tuple

import pytest


SKILL_TEMPLATE = textwrap.dedent(
    """
    ---
    name: NAME
    description: Test fixture skill.
    version: VERSION
    type: instruction
    ---

    # NAME
    """
).strip() + "\n"


def _write_skill(parent: pathlib.Path, slug: str, version: str = "1.0.0") -> pathlib.Path:
    skill_dir = parent / slug
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        SKILL_TEMPLATE.replace("NAME", slug).replace("VERSION", version),
        encoding="utf-8",
    )
    return skill_dir


@pytest.fixture
def fake_log():
    return logging.getLogger("ouroboros.tests.resync")


@pytest.fixture
def staging(tmp_path) -> Tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """Build (seed_dir, native_root, drive_root) — drive_root points at
    a fake data plane the resync can write migration records into."""
    drive_root = tmp_path / "data"
    native_root = drive_root / "skills" / "native"
    seed_dir = tmp_path / "repo_skills"
    seed_dir.mkdir()
    native_root.mkdir(parents=True)
    return seed_dir, native_root, drive_root


# ---------------------------------------------------------------------------
# (a) skip when target absent
# ---------------------------------------------------------------------------


def test_resync_skips_when_target_absent_after_user_deletion(staging, fake_log):
    """Cycle 1 GPT-4(a) — the resync pass must NOT resurrect a seed skill
    the user deleted from native/. Only the first-time bootstrap may
    write a missing seed skill; resync exclusively upgrades existing.
    """
    from ouroboros.launcher_bootstrap import _per_skill_version_resync

    seed_dir, native_root, drive_root = staging
    _write_skill(seed_dir, "weather", version="0.2.0")
    # User deleted weather from native/ — nothing to upgrade.
    upgraded = _per_skill_version_resync(seed_dir, native_root, fake_log, drive_root=drive_root)
    assert upgraded == 0
    assert not (native_root / "weather").exists()


def test_seed_marker_allows_only_new_post_bootstrap_native_skill(tmp_path, fake_log):
    from ouroboros.launcher_bootstrap import _seed_skills_into

    seed_dir = tmp_path / "repo_skills"
    target_root = tmp_path / "data" / "skills"
    native_root = target_root / "native"
    _write_skill(seed_dir, "weather", version="1.0.0")
    _write_skill(seed_dir, "telegram", version="1.0.0")
    _write_skill(seed_dir, "unix_computer_use", version="0.1.0")
    native_root.mkdir(parents=True)
    (native_root / ".bootstrap-seed-complete").write_text("already seeded\n", encoding="utf-8")

    copied = _seed_skills_into(seed_dir, target_root, fake_log)

    assert copied == 2
    assert not (native_root / "weather").exists()
    assert (native_root / "telegram" / "SKILL.md").is_file()
    assert (native_root / "unix_computer_use" / "SKILL.md").is_file()
    marker = (native_root / "unix_computer_use" / ".seed-origin").read_text(encoding="utf-8")
    assert "post_bootstrap_new_seed=true" in marker
    assert (native_root / ".post-bootstrap-seed-unix_computer_use").is_file()
    assert (native_root / ".post-bootstrap-seed-telegram").is_file()

    shutil.rmtree(native_root / "unix_computer_use")
    assert _seed_skills_into(seed_dir, target_root, fake_log) == 0
    assert not (native_root / "unix_computer_use").exists()
    assert (native_root / "telegram").is_dir()


def test_post_bootstrap_seed_preserves_existing_same_name_payload(tmp_path, fake_log):
    from ouroboros.launcher_bootstrap import _seed_skills_into

    seed_dir = tmp_path / "repo_skills"
    target_root = tmp_path / "data" / "skills"
    native_root = target_root / "native"
    external_root = target_root / "external"
    _write_skill(seed_dir, "telegram", version="1.0.0")
    existing = _write_skill(external_root, "telegram", version="0.9.0")
    original = (existing / "SKILL.md").read_bytes()
    native_root.mkdir(parents=True)
    (native_root / ".bootstrap-seed-complete").write_text(
        "already seeded\n", encoding="utf-8"
    )

    assert _seed_skills_into(seed_dir, target_root, fake_log) == 0
    assert (existing / "SKILL.md").read_bytes() == original
    assert not (native_root / "telegram").exists()
    assert not (native_root / ".post-bootstrap-seed-telegram").exists()

    shutil.rmtree(existing)
    assert _seed_skills_into(seed_dir, target_root, fake_log) == 1
    assert (native_root / "telegram" / "SKILL.md").is_file()


# ---------------------------------------------------------------------------
# (b) skip when no .seed-origin marker
# ---------------------------------------------------------------------------


def test_resync_skips_user_managed_skills_without_seed_origin(staging, fake_log):
    """Cycle 1 GPT-4(b) — a user dropped a skill folder under native/
    that happens to share a name with a seed skill but has no
    ``.seed-origin`` marker. The resync must not touch it.
    """
    from ouroboros.launcher_bootstrap import _per_skill_version_resync

    seed_dir, native_root, drive_root = staging
    _write_skill(seed_dir, "weather", version="0.2.0")
    user_skill = _write_skill(native_root, "weather", version="0.1.0")
    # No .seed-origin — user-managed.
    user_extra = user_skill / "user_notes.md"
    user_extra.write_text("user data", encoding="utf-8")

    upgraded = _per_skill_version_resync(seed_dir, native_root, fake_log, drive_root=drive_root)
    assert upgraded == 0
    assert user_extra.is_file()
    # Manifest still at user version.
    text = (user_skill / "SKILL.md").read_text(encoding="utf-8")
    assert "version: 0.1.0" in text


# ---------------------------------------------------------------------------
# (c) reseed on drift + state-dir survival + user-mod files inside skill wiped
# ---------------------------------------------------------------------------


def test_resync_reseeds_on_drift_and_wipes_in_skill_user_files(staging, fake_log):
    """Cycle 1 GPT-4(c) — when the seed version differs from the installed
    version AND the skill carries a ``.seed-origin`` marker, the resync
    replaces the tree wholesale. User files INSIDE the skill dir are
    wiped (native skills are launcher-owned). Files OUTSIDE under
    ``data/state/skills/<name>/`` are not touched (different plane).
    """
    from ouroboros.launcher_bootstrap import _per_skill_version_resync

    seed_dir, native_root, drive_root = staging
    _write_skill(seed_dir, "weather", version="0.2.0")
    # Installed seeded copy at older version.
    installed = _write_skill(native_root, "weather", version="0.1.0")
    (installed / ".seed-origin").write_text("seeded_from=test\n", encoding="utf-8")
    user_extra = installed / "user_extra.txt"
    user_extra.write_text("user mod", encoding="utf-8")

    # Build a parallel state-dir to confirm it survives.
    state_dir = drive_root / "state" / "skills" / "weather"
    state_dir.mkdir(parents=True)
    (state_dir / "enabled.json").write_text('{"enabled": true}', encoding="utf-8")
    (state_dir / "review.json").write_text('{"status": "pass"}', encoding="utf-8")

    upgraded = _per_skill_version_resync(seed_dir, native_root, fake_log, drive_root=drive_root)
    assert upgraded == 1
    # New version landed.
    assert "version: 0.2.0" in (installed / "SKILL.md").read_text(encoding="utf-8")
    # User mod inside skill dir is gone — launcher owns native/.
    assert not user_extra.exists()
    # .seed-origin rewritten with upgrade=true marker.
    so = (installed / ".seed-origin").read_text(encoding="utf-8")
    assert "upgrade=true" in so
    # State-dir fully preserved.
    assert (state_dir / "enabled.json").is_file()
    assert (state_dir / "review.json").is_file()


def test_resync_no_longer_writes_migration_record_on_drift(staging, fake_log):
    """Native upgrade banners are retired; resync only replaces launcher-owned payloads."""
    from ouroboros.launcher_bootstrap import _per_skill_version_resync

    seed_dir, native_root, drive_root = staging
    _write_skill(seed_dir, "weather", version="0.2.0")
    installed = _write_skill(native_root, "weather", version="0.1.0")
    (installed / ".seed-origin").write_text("seeded_from=test\n", encoding="utf-8")

    _per_skill_version_resync(seed_dir, native_root, fake_log, drive_root=drive_root)
    assert not (drive_root / "state" / "migrations.json").exists()


# ---------------------------------------------------------------------------
# (d) noop on identical version
# ---------------------------------------------------------------------------


def test_resync_noop_on_identical_version(staging, fake_log):
    """Cycle 1 GPT-4(d) — same version on both sides means no upgrade
    fires and no migration record is written.
    """
    from ouroboros.launcher_bootstrap import _per_skill_version_resync

    seed_dir, native_root, drive_root = staging
    _write_skill(seed_dir, "weather", version="0.2.0")
    installed = _write_skill(native_root, "weather", version="0.2.0")
    (installed / ".seed-origin").write_text("seeded_from=test\n", encoding="utf-8")

    upgraded = _per_skill_version_resync(seed_dir, native_root, fake_log, drive_root=drive_root)
    assert upgraded == 0
    assert not (drive_root / "state" / "migrations.json").exists()


# ---------------------------------------------------------------------------
# (e) downgrade is accepted (launcher-owned)
# ---------------------------------------------------------------------------


def test_resync_accepts_downgrade(staging, fake_log):
    """Cycle 1 GPT-4(e) — a downgrade scenario where the seed ships an
    OLDER version than the installed copy still triggers a reseed.
    Native skills are launcher-owned by design; the launcher's
    bundled seed is the source of truth, regardless of direction.
    Pin this to catch a future commit that adds a "seed_version >=
    target_version" guard.
    """
    from ouroboros.launcher_bootstrap import _per_skill_version_resync

    seed_dir, native_root, drive_root = staging
    _write_skill(seed_dir, "weather", version="0.1.0")
    installed = _write_skill(native_root, "weather", version="0.5.0")
    (installed / ".seed-origin").write_text("seeded_from=test\n", encoding="utf-8")

    upgraded = _per_skill_version_resync(seed_dir, native_root, fake_log, drive_root=drive_root)
    assert upgraded == 1
    assert "version: 0.1.0" in (installed / "SKILL.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# (f) _read_skill_manifest_version edge cases
# ---------------------------------------------------------------------------


def test_read_version_strips_yaml_inline_comment(tmp_path):
    """Cycle 1 GPT-1 — `version: 0.2.0 # comment` must NOT include the comment."""
    from ouroboros.launcher_bootstrap import _read_skill_manifest_version

    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: x\ndescription: y\nversion: 0.2.0 # bumped manually\ntype: instruction\n---\n",
        encoding="utf-8",
    )
    assert _read_skill_manifest_version(skill) == "0.2.0"


def test_read_version_handles_single_line_skill_json(tmp_path):
    """Cycle 1 GPT-2 — compact JSON `{"name":"foo","version":"0.2.0"}` must parse."""
    from ouroboros.launcher_bootstrap import _read_skill_manifest_version

    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "skill.json").write_text(
        '{"name":"foo","description":"y","version":"0.2.0","type":"instruction"}',
        encoding="utf-8",
    )
    assert _read_skill_manifest_version(skill) == "0.2.0"


def test_read_version_ignores_pre_frontmatter_version_lines(tmp_path):
    """Cycle 1 GPT-3 — a body line ``version: ignore-me`` BEFORE the
    ``---`` frontmatter must not be returned."""
    from ouroboros.launcher_bootstrap import _read_skill_manifest_version

    skill = tmp_path / "skill"
    skill.mkdir()
    # Note: parse_skill_manifest_text uses ``\A---`` anchored regex so
    # any content before the first ``---`` is treated as outside the
    # frontmatter and the parser raises. The new helper now defers to
    # the parser, so a pre-frontmatter ``version:`` line cannot leak.
    (skill / "SKILL.md").write_text(
        "version: not-a-real-version\n\n---\nname: x\ndescription: y\nversion: 0.2.0\ntype: instruction\n---\n",
        encoding="utf-8",
    )
    # Either we get the real frontmatter version, or we get empty
    # because the parser rejected the malformed content. EITHER way,
    # we must NOT return the body line's stale version string.
    out = _read_skill_manifest_version(skill)
    assert out != "not-a-real-version"


def test_read_version_returns_empty_for_malformed_manifest(tmp_path):
    """The helper must swallow parser exceptions so the resync pass
    just skips the upgrade for a malformed seed without taking down
    server startup."""
    from ouroboros.launcher_bootstrap import _read_skill_manifest_version

    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "this is not a manifest at all",
        encoding="utf-8",
    )
    assert _read_skill_manifest_version(skill) == ""


def test_read_version_returns_empty_for_missing_files(tmp_path):
    from ouroboros.launcher_bootstrap import _read_skill_manifest_version

    skill = tmp_path / "skill"
    skill.mkdir()
    assert _read_skill_manifest_version(skill) == ""


def test_telegram_format_upgrade_reseeds_version_1_1_1(tmp_path, fake_log):
    """The launcher-owned Telegram payload must resync for format parity.

    The pinned string is the RESYNC KEY, not decoration: `_per_skill_version_resync`
    re-seeds a marker-owned native skill only when the seed and installed `SKILL.md`
    `version` differ, so a payload change shipped without a bump never reaches an
    existing install. Re-pin this whenever skills/telegram payload changes.
    """
    from ouroboros.launcher_bootstrap import _per_skill_version_resync

    seed_dir = pathlib.Path(__file__).resolve().parents[1] / "skills"
    drive_root = tmp_path / "data"
    native_root = drive_root / "skills" / "native"
    installed = _write_skill(native_root, "telegram", version="1.0.1")
    (installed / ".seed-origin").write_text("seeded_from=test\n", encoding="utf-8")

    upgraded = _per_skill_version_resync(
        seed_dir,
        native_root,
        fake_log,
        drive_root=drive_root,
    )

    assert upgraded == 1
    assert "version: 1.1.1" in (installed / "SKILL.md").read_text(encoding="utf-8")
    assert (installed / "plugin.py").is_file()
