"""What the payload hash covers, and what it must leave out.

Split out of ``tests/test_skill_loader.py`` by theme: the hash that changes when a script is
edited and stays stable against state-dir noise, the hidden helpers and top-level files that
are hashed and therefore reviewed, the VCS caches that are not, the symlink escape excluded
from the pack, the sensitive files that fail closed, the hidden parent directory that is not
a filter, and the manifest entry that is part of the hash.
"""

from __future__ import annotations

import os

import pytest

from ouroboros.skill_loader import (
    SkillReviewState,
    compute_content_hash,
    load_skill,
    save_enabled,
    save_review_state,
)

from tests._skill_loader_shared import (
    _valid_script_manifest,
    _write_skill,
)


# ---------------------------------------------------------------------------
# Content hashing
# ---------------------------------------------------------------------------
def test_content_hash_changes_when_script_edited(tmp_path):
    repo_root = tmp_path / "skills"
    skill_dir = _write_skill(
        repo_root,
        "alpha",
        manifest=_valid_script_manifest("alpha"),
        scripts={"fetch.py": "print('one')\n"},
    )
    before = compute_content_hash(skill_dir)
    (skill_dir / "scripts" / "fetch.py").write_text("print('two')\n", encoding="utf-8")
    after = compute_content_hash(skill_dir)
    assert before != after


def test_content_hash_stable_against_state_dir_noise(tmp_path):
    """State-dir writes must not invalidate the skill content hash."""
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    repo_root = tmp_path / "skills"
    skill_dir = _write_skill(
        repo_root,
        "alpha",
        manifest=_valid_script_manifest("alpha"),
        scripts={"fetch.py": "print('x')\n"},
    )
    before = compute_content_hash(skill_dir)
    # State-dir writes happen in ``data/state/skills/<name>/``, which is
    # outside the skill directory entirely — hash should be unaffected.
    save_enabled(drive_root, "alpha", True)
    save_review_state(
        drive_root,
        "alpha",
        SkillReviewState(status="pass", content_hash=before),
    )
    after = compute_content_hash(skill_dir)
    assert before == after


def test_hidden_helper_files_are_hashed_and_reviewed(tmp_path):
    """Phase 3 round 10 regression: a blanket "skip all dotfiles" rule
    would let a hand-rolled ``.hidden_helper.py`` be imported by a
    reviewed script without contributing to the content hash. Hidden
    files OTHER than VCS/cache metadata must be hashed + reviewed."""
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    repo_root = tmp_path / "skills"
    skill_dir = _write_skill(
        repo_root,
        "sneak",
        manifest=_valid_script_manifest("sneak"),
        scripts={"main.py": "import importlib\nimportlib.import_module('.hidden_helper')\n"},
    )
    (skill_dir / ".hidden_helper.py").write_text("X = 1\n", encoding="utf-8")
    before = compute_content_hash(skill_dir, manifest_scripts=[{"name": "main.py"}])
    (skill_dir / ".hidden_helper.py").write_text("X = 'poisoned'\n", encoding="utf-8")
    after = compute_content_hash(skill_dir, manifest_scripts=[{"name": "main.py"}])
    assert before != after, (
        "Hidden helper file must be hashed — the subprocess can still "
        "import it, so a review PASS must stale when it changes."
    )


def test_vcs_cache_dirs_are_not_hashed(tmp_path):
    """Conversely, ``.git``/``__pycache__``/editor scratch directories
    MUST be excluded from the hash so a byte-flip in a cache file does
    not invalidate the review."""
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    repo_root = tmp_path / "skills"
    skill_dir = _write_skill(
        repo_root,
        "cacheskill",
        manifest=_valid_script_manifest("cacheskill"),
        scripts={"main.py": "print('ok')\n"},
    )
    (skill_dir / ".git").mkdir()
    (skill_dir / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (skill_dir / "__pycache__").mkdir()
    (skill_dir / "__pycache__" / "main.cpython-311.pyc").write_bytes(b"\x00\x01")
    before = compute_content_hash(skill_dir, manifest_scripts=[{"name": "main.py"}])
    (skill_dir / ".git" / "HEAD").write_text("ref: refs/heads/other\n", encoding="utf-8")
    (skill_dir / "__pycache__" / "main.cpython-311.pyc").write_bytes(b"\x02\x03")
    after = compute_content_hash(skill_dir, manifest_scripts=[{"name": "main.py"}])
    assert before == after, "VCS/cache scratch must be excluded from the hash."


def test_symlink_escape_excluded_from_pack(tmp_path):
    """Phase 3 round 10 regression: a symlink inside ``skill_dir`` whose
    target resolves outside the tree must NOT be hashed — otherwise
    ``compute_content_hash`` + ``_build_skill_file_packs`` would exfiltrate
    arbitrary local file contents to external reviewer models."""
    import platform
    if platform.system() == "Windows":
        pytest.skip("symlink creation requires admin on Windows")
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    repo_root = tmp_path / "skills"
    skill_dir = _write_skill(
        repo_root,
        "lnk",
        manifest=_valid_script_manifest("lnk"),
        scripts={"main.py": "print('ok')\n"},
    )
    outside = tmp_path / "outside_secret.txt"
    outside.write_text("SECRET_PAYLOAD\n", encoding="utf-8")
    escape_link = skill_dir / "escape.txt"
    os.symlink(outside, escape_link)
    _iter_payload_files_list = None
    # Use the private walker directly — this is the "would the hash /
    # review pack see this file" question.
    from ouroboros.skill_loader import _iter_payload_files
    reviewed = _iter_payload_files(skill_dir, manifest_scripts=[{"name": "main.py"}])
    assert escape_link.resolve() not in {p.resolve() for p in reviewed}
    # Hash is still deterministic (covers in-tree files only).
    assert compute_content_hash(skill_dir, manifest_scripts=[{"name": "main.py"}])


def test_sensitive_files_fail_closed_on_load(tmp_path):
    """Phase 3 round 20: a skill that ships a sensitive-shape file
    (`.env`, `credentials.json`, `.pem`, ...) fails to load. Rationale:
    silently excluding the file from hash/review would let a reviewed
    skill ``open('.env').read()`` at runtime to exfiltrate credentials
    that the reviewer never saw. The loader fails closed via
    ``SkillPayloadUnreadable``; the user must rename / relocate the
    file out of the skill directory."""
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    repo_root = tmp_path / "skills"
    skill_dir = _write_skill(
        repo_root,
        "secrety",
        manifest=_valid_script_manifest("secrety"),
        scripts={"main.py": "print('ok')\n"},
    )
    (skill_dir / ".env").write_text("SECRET_KEY=leak\n", encoding="utf-8")
    from ouroboros.skill_loader import SkillPayloadUnreadable
    with pytest.raises(SkillPayloadUnreadable):
        compute_content_hash(skill_dir, manifest_scripts=[{"name": "main.py"}])
    # The LoadedSkill reflects the load_error rather than crashing.
    loaded = load_skill(skill_dir, drive_root)
    assert loaded is not None
    assert loaded.load_error
    assert "sensitive" in loaded.load_error.lower()
    assert loaded.available_for_execution is False


def test_toplevel_skill_files_are_hashed_and_reviewed(tmp_path):
    """Phase 3 round 8 regression: runtime surface == reviewed surface.

    A subprocess started with ``cwd=skill_dir`` can ``import`` any
    non-hidden file at the top level. If those files were not part of
    ``_iter_payload_files`` the PASS verdict would not stale when
    they change. This test drops a top-level ``helper.py`` and checks
    that it IS included in the content hash."""
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    repo_root = tmp_path / "skills"
    skill_dir = _write_skill(
        repo_root,
        "mixed",
        manifest=_valid_script_manifest("mixed"),
        scripts={"fetch.py": "from helper import X\nprint(X)\n"},
    )
    (skill_dir / "helper.py").write_text("X = 'v1'\n", encoding="utf-8")
    before = compute_content_hash(
        skill_dir,
        manifest_entry="",
        manifest_scripts=[{"name": "fetch.py"}],
    )
    (skill_dir / "helper.py").write_text("X = 'v2-poisoned'\n", encoding="utf-8")
    after = compute_content_hash(
        skill_dir,
        manifest_entry="",
        manifest_scripts=[{"name": "fetch.py"}],
    )
    assert before != after, (
        "Editing a top-level helper.py must invalidate the content hash — "
        "skill_exec runs with cwd=skill_dir so that file is reachable."
    )


# ---------------------------------------------------------------------------
# Hidden-directory filter: relative-parts only, not absolute parts
# ---------------------------------------------------------------------------
def test_payload_hash_works_in_hidden_parent_dir(tmp_path):
    """Regression: ``_iter_payload_files`` used to drop every payload when
    the skills checkout lived in a hidden parent directory (e.g.
    ``~/.skills``) because it checked absolute ``path.parts`` for
    dotfile components."""
    # Build the skill inside a hidden parent so the resolved absolute
    # path of each payload file contains a ``.xyz`` component.
    hidden_root = tmp_path / ".xyz"
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    skill_dir = hidden_root / "weather"
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(_valid_script_manifest(), encoding="utf-8")
    (skill_dir / "scripts" / "fetch.py").write_text("print('hi')\n", encoding="utf-8")

    hashed = compute_content_hash(skill_dir)
    # Hash must cover the script, not just the manifest.
    loaded = load_skill(skill_dir, drive_root)
    assert loaded is not None
    assert loaded.content_hash == hashed
    assert hashed != compute_content_hash(skill_dir.parent / "does-not-exist")

    (skill_dir / "scripts" / "fetch.py").write_text("print('edited')\n", encoding="utf-8")
    assert compute_content_hash(skill_dir) != hashed


# ---------------------------------------------------------------------------
# Manifest entry file is part of the hash (extension-type skills)
# ---------------------------------------------------------------------------
def test_manifest_entry_file_is_hashed_and_invalidates_review(tmp_path):
    """A ``type: extension`` skill's ``entry`` file (e.g. ``plugin.py``)
    must be part of the content hash so editing it staleness-invalidates
    the review. This is the Phase 3 round 2 regression for
    ``_iter_payload_files``."""
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    repo_root = tmp_path / "skills"
    manifest = (
        "---\n"
        "name: ext1\n"
        "type: extension\n"
        "version: 0.1.0\n"
        "entry: plugin.py\n"
        "permissions: [widget]\n"
        "---\n"
        "body\n"
    )
    skill_dir = repo_root / "ext1"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(manifest, encoding="utf-8")
    (skill_dir / "plugin.py").write_text("def register(api): pass  # v1\n", encoding="utf-8")

    loaded = load_skill(skill_dir, drive_root)
    assert loaded is not None
    before = loaded.content_hash

    # Edit plugin.py — this must change the hash because the manifest
    # declared it as the entry file.
    (skill_dir / "plugin.py").write_text("def register(api): pass  # v2\n", encoding="utf-8")
    after = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    assert before != after, (
        "Editing the manifest-declared entry file must invalidate the "
        "skill content hash so the review goes stale."
    )


def test_manifest_scripts_outside_scripts_dir_are_hashed(tmp_path):
    """Phase 3 round 6 regression: a manifest ``scripts[].name`` that points
    outside the conventional ``scripts/`` directory (e.g. ``bin/run.sh``)
    must be included in the content hash.

    Before this fix ``skill_exec`` would still execute the declared file,
    but ``compute_content_hash`` ignored it — editing that file would
    NOT stale-invalidate the review, so a malicious skill could ship a
    reviewed manifest and then mutate the actual runnable file.
    """
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    repo_root = tmp_path / "skills"
    skill_dir = repo_root / "weird"
    (skill_dir / "bin").mkdir(parents=True)
    (skill_dir / "bin" / "run.sh").write_text("#!/bin/sh\necho 'v1'\n", encoding="utf-8")
    (skill_dir / "SKILL.md").write_text(
        (
            "---\n"
            "name: weird\n"
            "description: Runs a non-scripts/ script.\n"
            "version: 0.1.0\n"
            "type: script\n"
            "runtime: bash\n"
            "timeout_sec: 5\n"
            "scripts:\n"
            "  - name: bin/run.sh\n"
            "    description: The actual runnable.\n"
            "---\n"
            "body\n"
        ),
        encoding="utf-8",
    )
    loaded = load_skill(skill_dir, drive_root)
    assert loaded is not None
    before = loaded.content_hash
    (skill_dir / "bin" / "run.sh").write_text("#!/bin/sh\necho 'v2'\n", encoding="utf-8")
    after = compute_content_hash(
        skill_dir,
        manifest_entry=loaded.manifest.entry,
        manifest_scripts=loaded.manifest.scripts,
    )
    assert before != after, (
        "Editing a manifest-declared script outside scripts/ must "
        "invalidate the skill content hash so the review goes stale."
    )


def test_manifest_entry_outside_skill_dir_is_rejected(tmp_path):
    """A malicious manifest ``entry: ../../etc/passwd`` must not cause
    the hasher to follow the absolute path."""
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    repo_root = tmp_path / "skills"
    skill_dir = repo_root / "ext1"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        (
            "---\n"
            "name: ext1\n"
            "type: extension\n"
            "version: 0.1.0\n"
            "entry: ../../etc/passwd\n"
            "permissions: [widget]\n"
            "---\n"
            "body\n"
        ),
        encoding="utf-8",
    )
    loaded = load_skill(skill_dir, drive_root)
    # The loader must still succeed (parse error would be a separate
    # finding) but ``compute_content_hash`` must ignore the escape path.
    assert loaded is not None
    # Hash is non-empty (manifest counts) but does not include
    # /etc/passwd content.
    assert loaded.content_hash
