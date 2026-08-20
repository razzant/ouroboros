"""Which skills are executable, what the summary says about them, and the review gate behind both.

Split out of ``tests/test_skill_loader.py`` by theme: the pass review plus enabled flag
execution requires, the unsupported runtime and phase-3 extension it rejects, the persisted
phase-4 verdict, the counts and flat list the summary carries with its runtime-mode and
dependency gates, and every ``skill_review_gate`` rule over warnings and legacy advisory
passes.
"""

from __future__ import annotations

from ouroboros.skill_loader import (
    SkillReviewState,
    VALID_REVIEW_STATUSES,
    compute_content_hash,
    find_skill,
    list_available_for_execution,
    save_enabled,
    save_review_state,
    skill_review_gate,
    summarize_skills,
)

from tests._skill_loader_shared import (
    _valid_script_manifest,
    _write_skill,
)


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
