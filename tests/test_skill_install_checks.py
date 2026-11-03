"""Recorded self-check (Cyber Pro only) and owner update skip share exact-release acceptance and its floor."""
from __future__ import annotations

import json
import pathlib

import pytest

from ouroboros import config
from ouroboros import skill_source_install as installer
from ouroboros.skill_install_record import read_install_record, update_install_record
from ouroboros.skill_loader import auto_grant_if_enabled, find_skill
from ouroboros.skill_review_status import acceptance_in_force
from ouroboros.skill_source_catalog import tree_release_digest
from tests._skill_source_fixtures import write_skill


def test_owner_skip_summary_describes_exact_bytes_without_install_timing():
    from ouroboros.skill_review_status import skill_review_gate

    gate = skill_review_gate("pending", current_hash="release", acceptance={
        "basis": "owner_skip", "release_digest": "release"})
    assert gate["summary"] == "The owner skipped the review for these exact bytes."


def test_check_guidance_keeps_the_self_check_to_cyber_pro():
    from ouroboros.tools.skill_source_tools import _SCHEMAS

    for tool in ("skill_install", "skill_update"):
        description = _SCHEMAS[tool]["properties"]["check"]["description"]
        assert "outside Cyber Pro this is the only check" in description
        assert "only the owner skips a review" in description
        assert "self: Cyber Pro only" in description
    doc = (pathlib.Path(__file__).resolve().parents[1] / "docs/CREATING_SKILLS.md").read_text(encoding="utf-8")
    doc = " ".join(doc.split())
    assert "outside Cyber Pro only the owner skips a review" in doc
    assert "outside Cyber Pro that basis admits nothing" in doc


@pytest.fixture
def set_mode():
    def _set(mode: str) -> None:
        config.reset_runtime_mode_baseline_for_tests()
        config.initialize_runtime_mode_baseline(mode)
    yield _set
    config.reset_runtime_mode_baseline_for_tests()


@pytest.fixture
def release(tmp_path, monkeypatch):
    drive = tmp_path / "data"
    folder = write_skill(tmp_path / "source", "demo", extra="env_from_settings: [GITHUB_TOKEN]\n")
    row = {"id": "team", "name": "Team", "kind": "hub", "url": "https://example.invalid/team",
           "trust": "review", "confirmed": True}
    item = {"id": "team/demo", "name": "demo", "version": "1.0.0", "digest": tree_release_digest(folder)}
    monkeypatch.setenv("OUROBOROS_SKILL_SOURCES", json.dumps([row]))
    monkeypatch.setattr(installer, "find_item", lambda *a, **kw: (row, item, {}))
    monkeypatch.setattr(installer, "entry_payload_dir", lambda *a, **kw: folder)
    monkeypatch.setattr("ouroboros.config.SETTINGS_PATH", drive / "settings.json")
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.delenv("OUROBOROS_SKILLS_REPO_PATH", raising=False)
    return drive, folder, row, item


@pytest.mark.parametrize("grants", [True, False])
def test_cyber_pro_self_check_records_reason_accepts_exact_release_and_obeys_auto_grants(release, monkeypatch,
                                                                                        set_mode, grants):
    drive, _folder, _row, item = release
    set_mode("cyber_pro")
    monkeypatch.setenv("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", str(grants).lower())
    result = installer.install_item(drive, item["id"], check="self", check_notes="Read all files; only formats text.",
                                    actor="ouroboros", task_id="task-1")
    record = read_install_record(drive, "demo")
    assert record["basis"] == result["basis"] == "self_check"
    assert record["check"]["by"] == "ouroboros" and record["check"]["task_id"] == "task-1"
    assert record["check"]["notes"] == "Read all files; only formats text." and record["check"]["at"]
    assert result["acceptance"]["accepted"] and not result.get("needs_review")
    assert result["acceptance"]["granted_keys"] == (["GITHUB_TOKEN"] if grants else [])
    loaded = find_skill(drive, "demo")
    assert loaded.review.gate_for(loaded.content_hash)["blocking_reason"] == "self_check"
    (loaded.skill_dir / "added.txt").write_text("edit", encoding="utf-8")
    edited = find_skill(drive, "demo")
    assert edited.review.gate_for(edited.content_hash)["blocking_reason"] == "cyber_authority"
    assert acceptance_in_force(edited.review.acceptance, edited.content_hash) == ""
    # Inverted pin: the edited bytes are no acceptance, yet Cyber Pro admits them, and whatever admits
    # the bytes admits their packages (test_skill_enable_dependencies covers the install at Enable).
    assert edited.review.gate_for(edited.content_hash)["executable_review"]


@pytest.mark.parametrize("access", ["light", "advanced", "pro"])
def test_self_check_is_refused_outside_cyber_pro_and_nothing_lands(release, set_mode, access):
    drive, _folder, _row, item = release
    set_mode(access)
    with pytest.raises(installer.SkillInstallError) as exc:
        installer.install_item(drive, item["id"], check="self", check_notes="Read every file.", actor="ouroboros")
    assert exc.value.code == "bad_check" and "only in Cyber Pro" in str(exc.value)
    assert read_install_record(drive, "demo") is None
    assert not (drive / "skills" / "ouroboroshub" / "demo").exists()
    result = installer.install_item(drive, item["id"], actor="ouroboros")
    assert result["basis"] == "review" and result["needs_review"]


def test_a_rejected_update_cannot_be_self_checked_outside_cyber_pro(release, set_mode):
    drive, folder, row, item = release
    set_mode("pro")
    installer.install_item(drive, item["id"])
    old = find_skill(drive, "demo").content_hash
    manifest = folder / "SKILL.md"
    manifest.write_text(manifest.read_text(encoding="utf-8").replace("1.0.0", "2.0.0"), encoding="utf-8")
    item.update(version="2.0.0", digest=tree_release_digest(folder))
    rejected = {"version": "2.0.0", "digest": item["digest"], "reason": "review_blockers"}
    update_install_record(drive, "demo", update_rejected=rejected)
    with pytest.raises(installer.SkillInstallError) as exc:
        installer.update_installed(drive, "demo", mode="replace", local_digest=old, rows=[row],
                                   check="self", check_notes="Read the release the review rejected.")
    assert exc.value.code == "bad_check"
    record = read_install_record(drive, "demo")
    assert record["basis"] == "review" and record["update_rejected"]["digest"] == item["digest"]
    assert find_skill(drive, "demo").content_hash == old


def test_a_cyber_pro_self_check_admits_nothing_after_cyber_pro_ends(release, monkeypatch, set_mode):
    drive, _folder, _row, item = release
    monkeypatch.setenv("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", "true")
    set_mode("cyber_pro")
    installer.install_item(drive, item["id"], check="self", check_notes="Read every file.", actor="ouroboros")
    set_mode("pro")
    loaded = find_skill(drive, "demo")
    gate = loaded.review.gate_for(loaded.content_hash)
    assert not gate["executable_review"] and gate["blocking_reason"] == "review_pending"
    assert acceptance_in_force(loaded.review.acceptance, loaded.content_hash) == ""
    assert not auto_grant_if_enabled(drive, loaded).granted


@pytest.mark.parametrize("grants", [True, False])
def test_cyber_pro_keeps_a_trusted_install_basis_for_keys_and_dependencies(release, monkeypatch, set_mode, grants):
    drive, _folder, row, item = release
    row["trust"] = "trusted"
    monkeypatch.setenv("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", str(grants).lower())
    set_mode("cyber_pro")
    result = installer.install_item(drive, item["id"])
    assert result["basis"] == "trusted"
    assert result["acceptance"]["granted_keys"] == (["GITHUB_TOKEN"] if grants else [])
    loaded = find_skill(drive, "demo")
    assert loaded.review.gate_for(loaded.content_hash)["blocking_reason"] == "trusted_source"


def test_cyber_pro_alone_grants_nothing(release, monkeypatch, set_mode):
    drive, _folder, _row, item = release
    monkeypatch.setenv("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", "true")
    set_mode("cyber_pro")
    installer.install_item(drive, item["id"])
    loaded = find_skill(drive, "demo")
    gate = loaded.review.gate_for(loaded.content_hash)
    assert gate["executable_review"] and gate["blocking_reason"] == "cyber_authority"
    assert not auto_grant_if_enabled(drive, loaded).granted  # keys stay basis-bound; packages do not


@pytest.mark.parametrize("notes", ["", "   ", "x" * 2001], ids=["empty", "whitespace", "too_long"])
def test_self_check_requires_bounded_nonempty_notes_before_landing(release, set_mode, notes):
    drive, _folder, _row, item = release
    set_mode("cyber_pro")
    with pytest.raises(installer.SkillInstallError) as exc:
        installer.install_item(drive, item["id"], check="self", check_notes=notes)
    assert exc.value.code == "bad_check"
    assert read_install_record(drive, "demo") is None
    assert not (drive / "skills" / "ouroboroshub" / "demo").exists()


def test_trusted_source_ignores_check_and_reports_it(release):
    drive, _folder, row, item = release
    row["trust"] = "trusted"
    result = installer.install_item(drive, item["id"], check="self", check_notes="")
    assert result["basis"] == "trusted" and result["check_ignored"]
    assert "check" not in read_install_record(drive, "demo")


def test_default_check_still_requests_the_configured_review(release):
    drive, _folder, _row, item = release
    result = installer.install_item(drive, item["id"])
    assert result["basis"] == "review" and result["needs_review"]
    assert "check" not in read_install_record(drive, "demo")


def test_self_check_keeps_the_deterministic_preflight(release, monkeypatch, set_mode):
    drive, _folder, _row, item = release
    set_mode("cyber_pro")
    monkeypatch.setattr("ouroboros.skill_review.run_acceptance_preflight", lambda *a, **k: {"ok": False, "error": "bad"})
    result = installer.install_item(drive, item["id"], check="self", check_notes="read it")
    assert not result["acceptance"]["accepted"] and result["acceptance"]["granted_keys"] == []
    assert result["acceptance"]["turn_on"] == "not_ready"


@pytest.mark.parametrize("choice,basis", [("self", "self_check"), ("owner", "owner_skip")])
def test_update_records_self_check_or_owner_skip_and_clears_rejection(release, set_mode, choice, basis):
    drive, folder, row, item = release
    set_mode("cyber_pro" if choice == "self" else "pro")
    installer.install_item(drive, item["id"])
    old = find_skill(drive, "demo").content_hash
    update_install_record(drive, "demo", update_rejected={"version": "2", "digest": "x", "reason": "review_blockers"})
    manifest = folder / "SKILL.md"
    manifest.write_text(manifest.read_text(encoding="utf-8").replace("1.0.0", "2.0.0"), encoding="utf-8")
    item.update(version="2.0.0", digest=tree_release_digest(folder))
    kwargs = {"skip_review": True} if choice == "owner" else {"check": "self", "check_notes": "Read the new release."}
    result = installer.update_installed(drive, "demo", mode="replace", local_digest=old, rows=[row], **kwargs)
    record = read_install_record(drive, "demo")
    assert record["basis"] == result["basis"] == basis
    assert record["release"]["digest"] == item["digest"]
    assert "update_rejected" not in record and "pending_update" not in record
    assert result["acceptance"]["accepted"] and not result.get("needs_review")
    loaded = find_skill(drive, "demo")
    assert loaded.review.gate_for(loaded.content_hash)["blocking_reason"] == basis
    (loaded.skill_dir / "local.txt").write_text("owner edit", encoding="utf-8")
    edited = find_skill(drive, "demo")
    assert acceptance_in_force(edited.review.acceptance, edited.content_hash) == ""
    assert edited.review.gate_for(edited.content_hash)["executable_review"] == (choice == "self")  # Cyber Pro judgment


@pytest.mark.parametrize("mode", ["keep", "merge"])
def test_owner_update_skip_is_replace_only(release, mode):
    drive, _folder, row, item = release
    installer.install_item(drive, item["id"])
    with pytest.raises(installer.SkillInstallError) as exc:
        installer.update_installed(drive, "demo", mode=mode, skip_review=True, rows=[row])
    assert exc.value.code == "bad_mode"
    assert read_install_record(drive, "demo")["basis"] == "review"


def test_public_install_fields_omit_task_context_and_rollback_details():
    from ouroboros.skill_install_record import install_listing_fields

    assert install_listing_fields({}) == {}
    record = {"base_digest": "", "modified": True,
              "check": {"by": "ouroboros", "notes": "read it", "task_id": "private", "at": "now"},
              "pending_update": {"version": "2", "digest": "b", "from_digest": "a", "requested_at": "then"},
              "update_rejected": {"version": "3", "digest": "c", "reason": "review_blockers", "at": "now"}}
    assert install_listing_fields(record) == {
        "modified": True, "check": {"by": "ouroboros", "notes": "read it", "at": "now"},
        "pending_update": {"version": "2", "digest": "b"},
        "update_rejected": {"version": "3", "digest": "c", "reason": "review_blockers", "at": "now"}}


def test_builtin_from_review_row_keeps_recorded_self_check(tmp_path, monkeypatch, set_mode):
    from tests._skill_source_fixtures import builtin_repo

    set_mode("cyber_pro")
    drive = tmp_path / "data"
    repo = builtin_repo(tmp_path)
    row = {"id": "builtin", "kind": "builtin", "name": "Built in", "trust": "review", "confirmed": True}
    monkeypatch.setenv("OUROBOROS_SKILL_SOURCES", json.dumps([row]))
    monkeypatch.setattr("ouroboros.config.SETTINGS_PATH", drive / "settings.json")
    result = installer.install_item(drive, "builtin/telegram", rows=[row], repo_dir=repo,
                                    check="self", check_notes="Read the bundled instruction.")
    record = read_install_record(drive, "telegram")
    assert result["basis"] == record["basis"] == "self_check" and result["acceptance"]["accepted"]
    assert record["check"]["notes"] == "Read the bundled instruction."


@pytest.fixture
def registry_release(tmp_path, monkeypatch):
    from ouroboros.marketplace.clawhub import ClawHubSkillSummary
    from tests.test_marketplace_api import _patch_clawhub_install_pipeline

    drive = tmp_path / "data"
    row = {"id": "clawhub", "name": "ClawHub", "kind": "registry", "trust": "review", "confirmed": True}
    monkeypatch.setenv("OUROBOROS_SKILL_SOURCES", json.dumps([row]))
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setattr("ouroboros.config.SETTINGS_PATH", drive / "settings.json")
    monkeypatch.delenv("OUROBOROS_SKILLS_REPO_PATH", raising=False)
    storage = _patch_clawhub_install_pipeline(monkeypatch, tmp_path, auto_specs=[])
    registry = {"latest": "1.0.0", "changed": False, "staged_versions": []}
    monkeypatch.setattr(storage, "_registry_info", lambda slug: ClawHubSkillSummary(
        slug=slug, latest_version=registry["latest"]))
    original_stage = storage._stage_archive
    def stage(*args, **kwargs):
        staged = original_stage(*args, **kwargs)
        version = kwargs["version"]
        registry["staged_versions"].append(version)
        manifest = staged.staging_dir / "SKILL.md"
        text = manifest.read_text(encoding="utf-8").replace("1.0.0", version)
        if version != "1.0.0":
            text = text.replace("type: instruction", "env_from_settings: [GITHUB_TOKEN]\ntype: instruction")
        manifest.write_text(text + ("changed release" if registry["changed"] else ""), encoding="utf-8")
        return staged
    monkeypatch.setattr(storage, "_stage_archive", stage)
    assert storage.install_skill(drive, tmp_path, slug="demo", auto_review=False).ok
    return drive, storage, registry


@pytest.mark.parametrize("choice", ["confirmed", "changed", "newest"])
def test_registry_owner_skip_checks_staged_bytes_and_pins_only_confirmed_release(registry_release, choice):
    from ouroboros.skill_loader import SkillReviewState, save_review_state, skill_state_dir

    drive, _storage, registry = registry_release
    registry["latest"] = "2.0.0"
    assert installer.update_installed(drive, "demo", mode="replace")["needs_review"]
    rejected_digest = find_skill(drive, "demo").content_hash
    save_review_state(drive, "demo", SkillReviewState(status="blockers", content_hash=rejected_digest))
    assert installer.settle_pending_update(drive, "demo")["status"] == "restored"
    before = read_install_record(drive, "demo")
    assert before["update_rejected"]["version"] == "2.0.0"
    saved = {p.name: p.read_bytes() for p in skill_state_dir(drive, "demo").iterdir() if p.is_file()}
    registry.update(latest="3.0.0", changed=choice == "changed")
    digest = "" if choice == "newest" else rejected_digest
    if choice == "changed":
        with pytest.raises(installer.SkillInstallError) as exc:
            installer.update_installed(drive, "demo", mode="replace", digest=digest, skip_review=True)
        assert exc.value.code == "candidate_changed" and exc.value.status == 409
        assert exc.value.facts["candidate"]["version"] == "2.0.0"
        assert exc.value.facts["candidate"]["digest"] != digest
        assert read_install_record(drive, "demo") == before
        assert find_skill(drive, "demo").content_hash == before["release"]["digest"]
        assert {p.name: p.read_bytes() for p in skill_state_dir(drive, "demo").iterdir() if p.is_file()} == saved
    else:
        result = installer.update_installed(drive, "demo", mode="replace", digest=digest, skip_review=True)
        record = read_install_record(drive, "demo")
        assert record["basis"] == result["basis"] == "owner_skip"
        assert record["release"]["version"] == ("3.0.0" if choice == "newest" else "2.0.0")
        assert (record["release"]["digest"] == rejected_digest) is (choice == "confirmed")
        assert result["acceptance"]["accepted"] and result["acceptance"]["granted_keys"] == ["GITHUB_TOKEN"]
        assert "update_rejected" not in record
    assert registry["staged_versions"][-1] == ("3.0.0" if choice == "newest" else "2.0.0")


@pytest.mark.parametrize("via_source", [False, True])
def test_registry_update_without_digest_still_selects_latest(registry_release, via_source):
    drive, storage, registry = registry_release
    registry["latest"] = "3.0.0"
    if via_source:
        assert installer.update_installed(drive, "demo", mode="replace")["needs_review"]
    else:
        assert storage.update_skill(drive, drive.parent, sanitized_name="demo", auto_review=False).ok
    assert registry["staged_versions"][-1] == "3.0.0"
    assert read_install_record(drive, "demo")["release"]["version"] == "3.0.0"


def test_registry_staged_hash_failure_returns_install_result(registry_release, monkeypatch):
    drive, storage, _registry = registry_release
    before = read_install_record(drive, "demo")
    def fail(*args, **kwargs):
        raise OSError("staged tree unreadable")
    monkeypatch.setattr(storage, "_registry_install_record", fail)
    result = storage.install_skill(drive, drive.parent, slug="demo", version="2.0.0", overwrite=True,
                                   auto_review=False)
    assert isinstance(result, storage.InstallResult) and not result.ok
    assert "staged tree unreadable" in result.error
    assert not result.staged.staging_dir.exists()
    assert read_install_record(drive, "demo") == before
    assert find_skill(drive, "demo").content_hash == before["release"]["digest"]


@pytest.mark.parametrize("access", ["light", "advanced", "pro"])
def test_a_saved_self_check_keeps_review_each_strictness_outside_cyber_pro(release, monkeypatch, set_mode, access):
    from ouroboros.review_records import build_author_disposition
    from ouroboros.skill_acceptance import load_acceptance_facts
    from ouroboros.skill_review_status import skill_review_gate

    drive, _folder, _row, item = release
    set_mode("cyber_pro")
    installer.install_item(drive, item["id"], check="self", check_notes="Read every file.", actor="ouroboros")
    set_mode(access)
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    current = find_skill(drive, "demo").content_hash
    facts = load_acceptance_facts(drive, "demo")
    assert facts["strict_review"] is True
    author = build_author_disposition(disposition="accepted", rationale="checked", subject_hash=current,
                                      enforcement="advisory", action="finish")
    for disposition in (None, author):  # blockers stay off, even after an Advisory author finish
        gate = skill_review_gate("blockers", enforcement="advisory", current_hash=current, acceptance=facts,
                                 author_disposition=disposition)
        assert (gate["executable_review"], gate["blocking_reason"]) == (False, "blocker_findings_need_owner")
    assert skill_review_gate("clean", enforcement="advisory", current_hash=current, acceptance=facts)["executable_review"]
    skipped = skill_review_gate("blockers", enforcement="advisory", current_hash=current,
                                acceptance={**facts, "basis": "owner_skip"})
    assert (skipped["executable_review"], skipped["blocking_reason"]) == (True, "owner_skip")


@pytest.mark.parametrize("owner_digest_matches", [True, False])
def test_an_expired_self_check_does_not_mask_a_matching_owner_edit(owner_digest_matches):
    current = "d" * 64
    acceptance = {"basis": "self_check", "release_digest": current,
                  "owner_edit_digest": current if owner_digest_matches else "e" * 64}
    assert acceptance_in_force(acceptance, current, cyber=False) == ("owner_edit" if owner_digest_matches else "")
    assert acceptance_in_force(acceptance, current, cyber=True) == "self_check"
