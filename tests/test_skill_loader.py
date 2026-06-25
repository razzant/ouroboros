"""Phase 3 regression tests for ``ouroboros.skill_loader``.

Covers discovery, content-hashing, enabled-state persistence, and review
state round-trip. No network, no real review calls — these tests stay
hermetic against ``tmp_path``.
"""
from __future__ import annotations

import json
import os
import pathlib

import pytest

from ouroboros.skill_loader import (
    LoadedSkill,
    SkillReviewState,
    VALID_REVIEW_STATUSES,
    compute_content_hash,
    discover_skills,
    find_skill,
    list_available_for_execution,
    load_enabled,
    load_review_state,
    load_skill,
    save_enabled,
    save_review_state,
    skill_review_gate,
    skill_state_dir,
    summarize_skills,
)


def _write_skill(
    repo_root: pathlib.Path,
    name: str,
    *,
    manifest: str,
    scripts: dict[str, str] | None = None,
    manifest_name: str = "SKILL.md",
) -> pathlib.Path:
    skill_dir = repo_root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / manifest_name).write_text(manifest, encoding="utf-8")
    if scripts:
        (skill_dir / "scripts").mkdir(exist_ok=True)
        for filename, body in scripts.items():
            (skill_dir / "scripts" / filename).write_text(body, encoding="utf-8")
    return skill_dir


def _valid_script_manifest(name: str = "weather") -> str:
    return (
        "---\n"
        f"name: {name}\n"
        "description: Check the weather.\n"
        "version: 0.1.0\n"
        "type: script\n"
        "runtime: python3\n"
        "timeout_sec: 30\n"
        "permissions: [net]\n"
        "scripts:\n"
        "  - name: fetch.py\n"
        "    description: Fetch current weather.\n"
        "---\n"
        "# Weather skill\n\nCall fetch.py with a city.\n"
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
    import os, platform
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


def test_find_skill_returns_match_and_missing(tmp_path, monkeypatch):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    repo_root = tmp_path / "skills"
    _write_skill(repo_root, "alpha", manifest=_valid_script_manifest("alpha"))
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(repo_root))
    assert find_skill(drive_root, "alpha") is not None
    assert find_skill(drive_root, "does-not-exist") is None


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


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


def test_enabled_round_trip(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    assert load_enabled(drive_root, "x") is False
    save_enabled(drive_root, "x", True)
    assert load_enabled(drive_root, "x") is True
    save_enabled(drive_root, "x", False)
    assert load_enabled(drive_root, "x") is False


@pytest.mark.parametrize("payload,write_bytes", [
    (json.dumps({"enabled": "false"}).encode("utf-8"), None),  # non-boolean value
    (b"{\"enabled\": \xff}", None),                            # non-UTF-8 bytes
])
def test_load_enabled_fails_closed_on_corrupt_state(payload, write_bytes, tmp_path):
    """load_enabled must default to False on any corrupt state file.

    Parametrized in v5.15.x from test_load_enabled_fails_closed_on_non_boolean_payload
    + test_load_enabled_fails_closed_on_non_utf8_state_file."""
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    raw_path = skill_state_dir(drive_root, "x") / "enabled.json"
    raw_path.write_bytes(payload)
    assert load_enabled(drive_root, "x") is False


def test_review_state_round_trip(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    # Default when no file on disk.
    assert load_review_state(drive_root, "x").status == "pending"
    state = SkillReviewState(
        status="pass",
        content_hash="abcd",
        findings=[{"item": "manifest_schema", "verdict": "PASS", "severity": "critical", "reason": "ok"}],
        reviewer_models=["openai/gpt-5.5"],
        timestamp="2026-04-21T00:00:00+00:00",
        prompt_chars=1234,
        cost_usd=0.5,
        raw_actor_records=[{"model_id": "openai/gpt-5.5", "raw_text": "full"}],
    )
    save_review_state(drive_root, "x", state)
    reloaded = load_review_state(drive_root, "x")
    assert reloaded.status == "clean"
    assert reloaded.content_hash == "abcd"
    assert reloaded.reviewer_models == ["openai/gpt-5.5"]
    assert reloaded.prompt_chars == 1234
    assert reloaded.raw_actor_records == [{"model_id": "openai/gpt-5.5", "raw_text": "full"}]

    raw = json.loads((skill_state_dir(drive_root, "x") / "review.json").read_text(encoding="utf-8"))
    assert "status" not in raw


def test_load_review_state_live_aggregates_soft_findings(tmp_path, monkeypatch):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    state = SkillReviewState(
        status="advisory",
        content_hash="abcd",
        findings=[{
            "item": "timeout_and_output_discipline",
            "verdict": "FAIL",
            "severity": "advisory",
            "reason": "soft",
        }],
    )
    save_review_state(drive_root, "x", state)

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    assert load_review_state(drive_root, "x", skill_type="script").status == "warnings"

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    assert load_review_state(drive_root, "x", skill_type="script").status == "warnings"


def test_load_review_state_fails_closed_on_invalid_numeric_fields(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    raw_path = skill_state_dir(drive_root, "x") / "review.json"
    raw_path.write_text(
        json.dumps(
            {
                "status": "pass",
                "content_hash": "abcd",
                "prompt_chars": "not-an-int",
                "cost_usd": "not-a-float",
            }
        ),
        encoding="utf-8",
    )
    reloaded = load_review_state(drive_root, "x")
    assert reloaded.status == "clean"
    assert reloaded.prompt_chars == 0
    assert reloaded.cost_usd == 0.0


def test_load_review_state_fails_closed_on_non_utf8_state_file(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    raw_path = skill_state_dir(drive_root, "x") / "review.json"
    raw_path.write_bytes(b"{\"status\": \"pass\", \"content_hash\": \xff}")
    reloaded = load_review_state(drive_root, "x")
    assert reloaded.status == "pending"
    assert reloaded.content_hash == ""


def test_review_state_unknown_status_clamped_to_pending(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    raw_path = skill_state_dir(drive_root, "x") / "review.json"
    raw_path.write_text(
        json.dumps({"status": "TURBO", "content_hash": "abcd"}),
        encoding="utf-8",
    )
    reloaded = load_review_state(drive_root, "x")
    assert reloaded.status == "pending"
    assert reloaded.content_hash == "abcd"


# ---------------------------------------------------------------------------
# available_for_execution gating
# ---------------------------------------------------------------------------


def test_available_for_execution_requires_pass_review_and_enabled(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    repo_root = tmp_path / "skills"
    _write_skill(
        repo_root,
        "alpha",
        manifest=_valid_script_manifest("alpha"),
        scripts={"fetch.py": "print('x')\n"},
    )
    # Step 1: pending + disabled → not available.
    assert list_available_for_execution(drive_root, repo_path=str(repo_root)) == []

    # Step 2: enabled but still pending → not available.
    save_enabled(drive_root, "alpha", True)
    assert list_available_for_execution(drive_root, repo_path=str(repo_root)) == []

    # Step 3: pass review with the current hash → available.
    loaded = find_skill(drive_root, "alpha", repo_path=str(repo_root))
    assert loaded is not None
    save_review_state(
        drive_root,
        "alpha",
        SkillReviewState(status="pass", content_hash=loaded.content_hash),
    )
    available = list_available_for_execution(drive_root, repo_path=str(repo_root))
    assert [s.name for s in available] == ["alpha"]

    # Step 4: edit the script → review goes stale → not available again.
    (loaded.skill_dir / "scripts" / "fetch.py").write_text("print('edited')\n", encoding="utf-8")
    available = list_available_for_execution(drive_root, repo_path=str(repo_root))
    assert available == []


def test_available_for_execution_rejects_unsupported_runtime(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    repo_root = tmp_path / "skills"
    _write_skill(
        repo_root,
        "alpha",
        manifest=_valid_script_manifest("alpha").replace("runtime: python3", "runtime: perl"),
        scripts={"fetch.py": "print('x')\n"},
    )
    save_enabled(drive_root, "alpha", True)
    loaded = find_skill(drive_root, "alpha", repo_path=str(repo_root))
    assert loaded is not None
    save_review_state(
        drive_root,
        "alpha",
        SkillReviewState(status="pass", content_hash=loaded.content_hash),
    )
    refreshed = find_skill(drive_root, "alpha", repo_path=str(repo_root))
    assert refreshed is not None
    assert refreshed.available_for_execution is False


def test_extension_skill_never_executable_in_phase3(tmp_path):
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
    skill_dir = _write_skill(repo_root, "ext1", manifest=manifest)
    (skill_dir / "plugin.py").write_text("def register(api): pass\n", encoding="utf-8")
    save_enabled(drive_root, "ext1", True)
    loaded = find_skill(drive_root, "ext1", repo_path=str(repo_root))
    assert loaded is not None
    save_review_state(
        drive_root,
        "ext1",
        SkillReviewState(status="pass", content_hash=loaded.content_hash),
    )
    loaded = find_skill(drive_root, "ext1", repo_path=str(repo_root))
    assert loaded.manifest.is_extension()
    assert loaded.available_for_execution is False, (
        "Phase 3 must defer type=extension execution until Phase 4."
    )


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
    import os, platform
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
        assert s.available_for_execution is False


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


def test_extension_status_reflects_persisted_verdict_in_phase4(tmp_path, monkeypatch):
    """Phase 4 lifted the old Phase 3 ``pending_phase4`` overlay — now
    that the extension loader exists, a persisted review verdict for a
    ``type: extension`` skill must surface verbatim so operators and
    the Skills UI see the real state."""
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    repo_root = tmp_path / "skills"
    manifest = (
        "---\n"
        "name: ext2\n"
        "type: extension\n"
        "version: 0.1.0\n"
        "entry: plugin.py\n"
        "permissions: [widget]\n"
        "---\n"
        "body\n"
    )
    skill_dir = _write_skill(repo_root, "ext2", manifest=manifest)
    (skill_dir / "plugin.py").write_text("def register(api): pass\n", encoding="utf-8")

    loaded_initial = find_skill(drive_root, "ext2", repo_path=str(repo_root))
    assert loaded_initial is not None
    save_review_state(
        drive_root,
        "ext2",
        SkillReviewState(status="pass", content_hash=loaded_initial.content_hash),
    )

    reloaded = find_skill(drive_root, "ext2", repo_path=str(repo_root))
    assert reloaded is not None
    # Real verdict surfaces — Phase 4 retired the ``pending_phase4`` overlay.
    assert reloaded.review.status == "clean"

    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(repo_root))
    summary = summarize_skills(drive_root)
    statuses = {s["name"]: s["review_status"] for s in summary["skills"]}
    assert statuses["ext2"] == "clean"


# ---------------------------------------------------------------------------
# summarize_skills shape
# ---------------------------------------------------------------------------


def test_summarize_skills_shape_contains_counts_and_flat_list(tmp_path, monkeypatch):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    repo_root = tmp_path / "skills"
    _write_skill(repo_root, "alpha", manifest=_valid_script_manifest("alpha"))
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(repo_root))
    summary = summarize_skills(drive_root)
    assert summary["count"] == 1
    assert summary["available"] == 0
    assert summary["pending_review"] == 1
    assert summary["blocker_review"] == 0
    assert summary["warning_review"] == 0
    assert summary["broken"] == 0
    assert [s["name"] for s in summary["skills"]] == ["alpha"]


def test_summarize_skills_reflects_runtime_mode_light(tmp_path, monkeypatch):
    """v5.1.2 Frame A: a reviewed + enabled skill stays ``available``
    in light mode, because ``skill_exec`` no longer refuses light.
    The static-readiness signal and the available-for-execution flag
    converge in this release."""
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    repo_root = tmp_path / "skills"
    _write_skill(
        repo_root,
        "alpha",
        manifest=_valid_script_manifest("alpha"),
        scripts={"fetch.py": "print('ok')\n"},
    )
    # Mark reviewed + enabled so the skill would be statically available.
    loaded = find_skill(drive_root, "alpha", repo_path=str(repo_root))
    assert loaded is not None
    save_enabled(drive_root, "alpha", True)
    save_review_state(
        drive_root,
        "alpha",
        SkillReviewState(status="pass", content_hash=loaded.content_hash),
    )

    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(repo_root))

    # advanced → available
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    adv = summarize_skills(drive_root)
    assert adv["available"] == 1
    assert adv["skills"][0]["available_for_execution"] is True

    # v5.1.2 Frame A: light is also ``available`` — skills run regardless
    # of runtime_mode (light still blocks repo self-modification +
    # elevation ratchet, just not skill execution).
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    light = summarize_skills(drive_root)
    assert light["available"] == 1
    assert light["skills"][0]["available_for_execution"] is True
    assert light["skills"][0]["review_gate"]["executable_review"] is True
    assert light["skills"][0]["executable_review"] is True
    assert light["skills"][0]["static_ready"] is True


def test_summarize_skills_blocks_missing_isolated_deps(tmp_path, monkeypatch):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    repo_root = tmp_path / "skills"
    manifest = _valid_script_manifest("alpha").replace(
        "scripts:\n",
        "install_specs:\n"
        "  - kind: pip\n"
        "    package: wheel\n"
        "scripts:\n",
    )
    skill_dir = _write_skill(
        repo_root,
        "alpha",
        manifest=manifest,
        scripts={"fetch.py": "print('ok')\n"},
    )
    loaded = find_skill(drive_root, "alpha", repo_path=str(repo_root))
    assert loaded is not None
    save_enabled(drive_root, "alpha", True)
    save_review_state(
        drive_root,
        "alpha",
        SkillReviewState(status="pass", content_hash=compute_content_hash(skill_dir)),
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(repo_root))

    summary = summarize_skills(drive_root)

    assert list_available_for_execution(drive_root, repo_path=str(repo_root)) == []
    assert summary["available"] == 0
    assert summary["skills"][0]["available_for_execution"] is False
    assert summary["skills"][0]["static_ready"] is False


def test_available_summary_keeps_runtime_and_script_substrate_gate(tmp_path, monkeypatch):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    repo_root = tmp_path / "skills"
    unsupported_runtime = _valid_script_manifest("bad_runtime").replace(
        "runtime: python3\n",
        "runtime: perl\n",
    )
    missing_script = _valid_script_manifest("missing_script")
    skill_dirs = {
        "bad_runtime": _write_skill(
            repo_root,
            "bad_runtime",
            manifest=unsupported_runtime,
            scripts={"fetch.py": "print('ok')\n"},
        ),
        "missing_script": _write_skill(
            repo_root,
            "missing_script",
            manifest=missing_script,
            scripts={},
        ),
    }
    for name, skill_dir in skill_dirs.items():
        save_enabled(drive_root, name, True)
        save_review_state(
            drive_root,
            name,
            SkillReviewState(status="pass", content_hash=compute_content_hash(skill_dir)),
        )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(repo_root))

    summary = summarize_skills(drive_root)

    assert list_available_for_execution(drive_root, repo_path=str(repo_root)) == []
    assert summary["available"] == 0
    by_name = {row["name"]: row for row in summary["skills"]}
    assert by_name["bad_runtime"]["available_for_execution"] is False
    assert by_name["bad_runtime"]["static_ready"] is False
    assert by_name["missing_script"]["available_for_execution"] is False
    assert by_name["missing_script"]["static_ready"] is False


def test_valid_review_statuses_exported():
    assert "clean" in VALID_REVIEW_STATUSES
    assert "warnings" in VALID_REVIEW_STATUSES
    assert "blockers" in VALID_REVIEW_STATUSES
    # Legacy persisted names remain accepted for migration.
    assert "pass" in VALID_REVIEW_STATUSES
    assert "pending" in VALID_REVIEW_STATUSES
    assert "pending_phase4" in VALID_REVIEW_STATUSES


def test_skill_review_gate_allows_warnings_under_blocking(monkeypatch):
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")

    gate = skill_review_gate("warnings", stale=False)

    assert gate["executable_review"] is True
    assert gate["blocking_reason"] == "warnings_do_not_block_execution"
    assert gate["review_enforcement"] == "blocking"


def test_skill_review_gate_allows_legacy_advisory_pass(monkeypatch):
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")

    gate = skill_review_gate("advisory_pass", stale=False)

    assert gate["executable_review"] is True


def test_skill_review_gate_revalidates_advisory_pass_under_blocking(monkeypatch):
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")

    gate = skill_review_gate("advisory_pass", stale=False)

    assert gate["executable_review"] is True
    assert gate["blocking_reason"] == "warnings_do_not_block_execution"


def test_warnings_available_under_blocking(tmp_path, monkeypatch):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    repo_root = tmp_path / "skills"
    skill_dir = _write_skill(
        repo_root,
        "alpha",
        manifest=_valid_script_manifest("alpha"),
        scripts={"fetch.py": "print('ok')\n"},
    )
    loaded = find_skill(drive_root, "alpha", repo_path=str(repo_root))
    assert loaded is not None
    save_enabled(drive_root, "alpha", True)
    save_review_state(
        drive_root,
        "alpha",
        SkillReviewState(status="advisory_pass", content_hash=compute_content_hash(skill_dir)),
    )
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(repo_root))

    summary = summarize_skills(drive_root)

    assert len(list_available_for_execution(drive_root, repo_path=str(repo_root))) == 1
    assert summary["available"] == 1
    assert summary["skills"][0]["available_for_execution"] is True
    assert summary["skills"][0]["static_ready"] is True
    assert summary["skills"][0]["review_gate"]["blocking_reason"] == "warnings_do_not_block_execution"


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


# ---------------------------------------------------------------------------
# Safety: skill name sanitization
# ---------------------------------------------------------------------------


def test_skill_state_dir_resists_path_escape(tmp_path):
    """A malicious manifest ``name: ../../etc`` cannot escape the state root."""
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    malicious = "../../etc/passwd"
    state_path = skill_state_dir(drive_root, malicious)
    resolved = state_path.resolve()
    state_root_resolved = (drive_root / "state" / "skills").resolve()
    # The returned path must stay under data/state/skills/.
    assert resolved.is_relative_to(state_root_resolved)


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
