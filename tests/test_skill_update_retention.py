"""Reviewed updates retain a durable return path through the ordinary lifecycle."""
from __future__ import annotations

import json
import pathlib
import threading
from types import SimpleNamespace

import pytest

from ouroboros import skill_source_install as installer
from ouroboros.skill_install_record import read_install_record
from ouroboros.skill_loader import SkillReviewState, find_skill, load_enabled, save_enabled, save_review_state, skill_state_dir
from ouroboros.skill_review import SkillReviewOutcome
from ouroboros.skill_review_runner import run_skill_review_lifecycle_blocking
from ouroboros.skill_source_catalog import tree_release_digest
from tests.test_skill_install_checks import registry_release, release  # noqa: F401  (fixtures)


def _pending(release):  # noqa: F811
    drive, folder, row, item = release
    row["trust"] = "trusted"
    installer.install_item(drive, item["id"])
    save_enabled(drive, "demo", True, actor="owner_ui")
    old = find_skill(drive, "demo")
    save_review_state(drive, "demo", SkillReviewState(status="clean", content_hash=old.content_hash))
    saved = {name: (skill_state_dir(drive, "demo") / name).read_bytes()
             for name in ("review.json", "grants.json", "install.json")}
    row["trust"] = "review"
    manifest = folder / "SKILL.md"
    manifest.write_text(manifest.read_text(encoding="utf-8").replace("1.0.0", "2.0.0"), encoding="utf-8")
    item.update(version="2.0.0", digest=tree_release_digest(folder))
    result = installer.run_in_lane(drive, "update", "demo", lambda: installer.update_installed(
        drive, "demo", mode="replace", rows=[row]))
    assert result["needs_review"]
    return old.content_hash, saved


def _review(drive, status):
    def impl(ctx, name):
        skill = find_skill(drive, name)
        findings = ([{"item": "skill_preflight", "verdict": "FAIL", "model": "deterministic_preflight"}]
                    if status == "pending" else [])
        save_review_state(drive, name, SkillReviewState(status=status, content_hash=skill.content_hash, findings=findings))
        return SkillReviewOutcome(skill_name=name, status=status, content_hash=skill.content_hash, findings=findings)
    return run_skill_review_lifecycle_blocking(SimpleNamespace(drive_root=drive, repo_dir=drive.parent, messages=[]),
                                              "demo", review_impl=impl, repo_path="")


@pytest.mark.parametrize("status,reason", [("blockers", "review_blockers"), ("pending", "preflight_failed")])
def test_failed_review_restores_previous_payload_state_and_execution(release, status, reason):  # noqa: F811
    drive, _folder, _row, item = release
    previous, saved = _pending(release)
    record = read_install_record(drive, "demo")
    assert record["pending_update"]["from_digest"] == previous
    assert record["pending_update"]["digest"] == item["digest"]
    assert not find_skill(drive, "demo").review.gate_for(item["digest"])["executable_review"]
    _review(drive, status)
    restored = find_skill(drive, "demo")
    assert restored.content_hash == previous and load_enabled(drive, "demo")
    assert restored.review.gate_for(previous)["executable_review"]
    record = read_install_record(drive, "demo")
    assert record["update_rejected"]["reason"] == reason
    assert record["update_rejected"]["digest"] == item["digest"] and record["update_rejected"]["version"] == "2.0.0"
    assert "pending_update" not in record
    for filename in ("review.json", "grants.json"):
        assert (skill_state_dir(drive, "demo") / filename).read_bytes() == saved[filename]
    assert record["release"] == json.loads(saved["install.json"])["release"]


def test_clean_review_keeps_new_release_and_drops_durable_state(release):  # noqa: F811
    drive, _folder, _row, item = release
    _pending(release)
    assert (skill_state_dir(drive, "demo") / "update_backup.json").is_file()
    _review(drive, "clean")
    assert find_skill(drive, "demo").content_hash == item["digest"]
    assert "pending_update" not in read_install_record(drive, "demo")
    assert not (skill_state_dir(drive, "demo") / "update_backup.json").exists()


def test_startup_restores_interrupted_update_and_leaves_running_review(release, monkeypatch):  # noqa: F811
    import os
    from ouroboros.utils import utc_now_iso

    drive, _folder, _row, item = release
    previous, _ = _pending(release)
    job = skill_state_dir(drive, "demo") / "review_job.json"
    job.write_text(json.dumps({"status": "running", "pid": os.getpid(), "last_heartbeat_at": utc_now_iso()}), encoding="utf-8")
    installer.settle_pending_updates_at_startup(drive)
    assert find_skill(drive, "demo").content_hash == item["digest"]
    job.write_text(json.dumps({"status": "interrupted"}), encoding="utf-8")
    installer.settle_pending_updates_at_startup(drive)
    assert find_skill(drive, "demo").content_hash == previous
    assert read_install_record(drive, "demo")["update_rejected"]["reason"] == "review_unfinished"


@pytest.mark.parametrize("kind", ["hub", "git", "builtin", "registry"])
@pytest.mark.parametrize("cut", ["before_swap", "between_moves", "after_swap"])
@pytest.mark.parametrize("same_release", [False, True], ids=["new-release", "same-release"])
def test_startup_restores_landing_before_new_install_record(request, tmp_path, monkeypatch, kind, cut, same_release):
    from ouroboros.marketplace import fetcher, install as storage

    if kind == "registry":
        drive, _storage, registry = request.getfixturevalue("registry_release")
        if not same_release:
            registry["latest"] = "2.0.0"
        rows = None
    else:
        drive, folder, row, item = request.getfixturevalue("release")
        row.update(kind=kind, trust="trusted")
        if kind == "builtin":
            folder.parent.rename(tmp_path / "skills")
            folder = tmp_path / "skills" / "demo"
            monkeypatch.setattr("ouroboros.config.REPO_DIR", tmp_path)
        installer.install_item(drive, item["id"])
        row["trust"] = "review"
        if not same_release:
            manifest = folder / "SKILL.md"
            manifest.write_text(manifest.read_text(encoding="utf-8").replace("1.0.0", "2.0.0"), encoding="utf-8")
            item.update(version="2.0.0", digest=tree_release_digest(folder))
        rows = [row]

    previous = read_install_record(drive, "demo")
    old = find_skill(drive, "demo")
    save_review_state(drive, "demo", SkillReviewState(status="clean", content_hash=old.content_hash))
    save_enabled(drive, "demo", True, actor="owner_ui")
    env_marker = old.skill_dir / ".ouroboros_env" / "ready.txt"
    env_marker.parent.mkdir()
    env_marker.write_text("previous dependencies", encoding="utf-8")
    state = skill_state_dir(drive, "demo")
    saved_review = (state / "review.json").read_bytes()
    land = fetcher.land_staged_tree

    class InterruptedLanding(BaseException):
        """A crash bypasses ordinary Exception rollback handlers."""

    def interrupt(staging, target, *, replacement_suffix):
        assert (state / "update_backup.json").is_file()
        if cut == "between_moves":
            target.rename(target.with_name(f"{target.name}.{replacement_suffix}"))
        elif cut == "after_swap":
            land(staging, target, replacement_suffix=replacement_suffix)
        raise InterruptedLanding()

    with monkeypatch.context() as patch:
        patch.setattr(fetcher, "land_staged_tree", interrupt)
        patch.setattr(storage, "land_staged_tree", interrupt)
        with pytest.raises(InterruptedLanding):
            installer.update_installed(drive, "demo", mode="replace", rows=rows)
    assert read_install_record(drive, "demo") == previous
    pending = json.loads((state / "update_backup.json").read_text(encoding="utf-8"))["pending_update"]
    assert (old.skill_dir / "SKILL.md").exists() == (cut != "between_moves")
    if cut == "after_swap":
        assert find_skill(drive, "demo").content_hash == pending["digest"]
        assert not env_marker.exists()

    installer.settle_pending_updates_at_startup(drive)
    assert find_skill(drive, "demo").content_hash == old.content_hash
    assert env_marker.read_text(encoding="utf-8") == "previous dependencies"
    assert (state / "review.json").read_bytes() == saved_review
    assert load_enabled(drive, "demo")
    restored = read_install_record(drive, "demo")
    assert restored["source"] == previous["source"] and restored["release"] == previous["release"]
    assert "pending_update" not in restored
    assert restored["update_rejected"]["reason"] == "review_unfinished"
    assert restored["update_rejected"]["digest"] == pending["digest"]
    assert not (state / "update_backup.json").exists()


def test_review_exception_restores_previous_without_waiting_for_restart(release):  # noqa: F811
    drive, _folder, _row, _item = release
    previous, _ = _pending(release)
    def fail(*a):
        raise RuntimeError("review interrupted")
    with pytest.raises(RuntimeError, match="review interrupted"):
        run_skill_review_lifecycle_blocking(SimpleNamespace(drive_root=drive, repo_dir=drive.parent, messages=[]),
                                           "demo", review_impl=fail, repo_path="")
    assert find_skill(drive, "demo").content_hash == previous
    assert read_install_record(drive, "demo")["update_rejected"]["reason"] == "review_unfinished"


@pytest.mark.parametrize("edited", [False, True])
def test_owner_skip_during_review_keeps_new_bytes(release, edited):  # noqa: F811
    from ouroboros.skill_lifecycle_actions import run_skill_action
    from ouroboros.skill_lifecycle_queue import lend_lifecycle_lane
    from ouroboros.skill_review import _persist_projection

    drive, _folder, _row, item = release
    _pending(release)
    waiting, finish = threading.Event(), threading.Event()
    errors, outcomes = [], []
    def slow_review(ctx, name):
        reviewed = find_skill(drive, name)
        with lend_lifecycle_lane():
            waiting.set()
            assert finish.wait(60)
        outcome = SkillReviewOutcome(skill_name=name, status="blockers", content_hash=item["digest"])
        _persist_projection(ctx, reviewed, drive, outcome)
        return outcome
    ctx = SimpleNamespace(drive_root=drive, repo_dir=drive.parent, messages=[])
    def run():
        try:
            outcomes.append(run_skill_review_lifecycle_blocking(ctx, "demo", review_impl=slow_review, repo_path=""))
        except Exception as exc:
            errors.append(exc)
    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert waiting.wait(60)
        if edited:
            (find_skill(drive, "demo").skill_dir / "owner.txt").write_text("owner edit", encoding="utf-8")
        accepted = find_skill(drive, "demo").content_hash
        result = run_skill_action(ctx, "demo", "attest", expected_content_hash=accepted,
                                  _owner_actor="owner_ui", repo_path="")
        assert result["ok"], result
        assert "pending_update" not in read_install_record(drive, "demo")
        accepted_review = (skill_state_dir(drive, "demo") / "review.json").read_bytes()
    finally:
        finish.set()
        thread.join(60)
    assert not thread.is_alive() and not errors
    assert outcomes[0]["status"] == "pending" and "not persisted" in outcomes[0]["error"]
    assert (skill_state_dir(drive, "demo") / "review.json").read_bytes() == accepted_review
    assert find_skill(drive, "demo").content_hash == accepted
    assert find_skill(drive, "demo").review.gate_for(accepted)["executable_review"]


def test_partial_restore_retains_evidence_and_never_replaces_payload_twice(release, monkeypatch):  # noqa: F811
    from ouroboros.marketplace import install as storage

    drive, _folder, _row, _item = release
    previous, _ = _pending(release)
    real_restore = storage.restore_lifecycle_state
    with monkeypatch.context() as patch:
        patch.setattr(storage, "restore_lifecycle_state", lambda *a: ["injected state write failure"])
        installer.settle_pending_updates_at_startup(drive)
    payload = find_skill(drive, "demo").skill_dir
    assert find_skill(drive, "demo").content_hash == previous
    (payload / "owner.txt").write_text("edit after partial restore", encoding="utf-8")
    assert (skill_state_dir(drive, "demo") / "update_backup.json").exists()
    assert storage.restore_lifecycle_state is real_restore
    installer.settle_pending_updates_at_startup(drive)
    assert (payload / "owner.txt").read_text(encoding="utf-8") == "edit after partial restore"
    assert read_install_record(drive, "demo")["update_rejected"]["reason"] == "review_unfinished"


def test_uninstall_pending_update_cannot_resurrect_on_startup(release):  # noqa: F811

    drive, _folder, _row, _item = release
    _pending(release)
    assert installer.uninstall_installed(drive, "demo")["ok"]
    installer.settle_pending_updates_at_startup(drive)
    assert find_skill(drive, "demo") is None
    assert not (skill_state_dir(drive, "demo") / "update_backup.json").exists()


@pytest.mark.parametrize("same_bytes", [False, True])
def test_replacing_pending_update_with_other_source_survives_old_settlement(release, same_bytes):  # noqa: F811
    drive, folder, row, item = release
    _pending(release)
    pending_digest = item["digest"]
    row.update(id="other", name="Other", url="https://example.invalid/other")
    item["id"] = "other/demo"
    if not same_bytes:
        (folder / "other.txt").write_text("replacement from another source", encoding="utf-8")
        item["digest"] = tree_release_digest(folder)
    replacement = installer.run_in_lane(drive, "install", "demo", lambda: installer.install_item(
        drive, item["id"], replace=True))
    assert replacement["ok"] and replacement["replaced"]
    # The replacing transaction owns supersession, even when the bytes coincide.
    assert not (skill_state_dir(drive, "demo") / "update_backup.json").exists()
    assert installer.settle_pending_update(drive, "demo", expected_digest=pending_digest)["status"] == "none"
    installer.settle_pending_updates_at_startup(drive)
    assert find_skill(drive, "demo").content_hash == item["digest"]
    record = read_install_record(drive, "demo")
    assert record["source"]["id"] == "other"
    assert "update_rejected" not in record


@pytest.mark.parametrize("replacement", ["different_release", "same_release", "uninstalled"])
def test_stale_snapshot_is_discarded_without_restoring_newer_state(release, replacement):  # noqa: F811
    from ouroboros.skill_install_record import clear_install_record, write_install_record

    drive, _folder, _row, item = release
    _pending(release)
    payload = find_skill(drive, "demo").skill_dir
    record = read_install_record(drive, "demo")
    record.pop("pending_update")
    record["source"] = {"id": "other", "kind": "hub", "name": "Other"}
    if replacement == "different_release":
        (payload / "other.txt").write_text("newer install", encoding="utf-8")
        record["release"]["digest"] = tree_release_digest(payload)
    write_install_record(drive, "demo", record)
    if replacement == "uninstalled":
        import shutil
        shutil.rmtree(payload)
        clear_install_record(drive, "demo")
    expected = read_install_record(drive, "demo")
    assert installer.settle_pending_update(drive, "demo", expected_digest=item["digest"]) == {"status": "superseded"}
    installer.settle_pending_updates_at_startup(drive)
    assert read_install_record(drive, "demo") == expected
    if replacement == "uninstalled":
        assert find_skill(drive, "demo") is None
    else:
        assert find_skill(drive, "demo").content_hash == record["release"]["digest"]
    assert not (skill_state_dir(drive, "demo") / "update_backup.json").exists()


def test_old_completion_leaves_newer_pending_snapshot_intact(release):  # noqa: F811
    drive, _folder, _row, item = release
    old_digest, _ = _pending(release)
    backup = skill_state_dir(drive, "demo") / "update_backup.json"
    saved = backup.read_bytes()
    assert installer.settle_pending_update(drive, "demo", expected_digest=old_digest) == {"status": "superseded"}
    assert backup.read_bytes() == saved
    assert read_install_record(drive, "demo")["pending_update"]["digest"] == item["digest"]


def test_registry_adapter_does_not_import_source_orchestration():
    import ast
    from ouroboros.marketplace import install as storage

    tree = ast.parse(pathlib.Path(storage.__file__).read_text(encoding="utf-8"))
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                and node.module == "ouroboros.skill_source_install"]


@pytest.mark.parametrize("field", ["target_dir", "backup_dir"])
def test_update_restore_ignores_mismatched_paths_and_keeps_evidence(release, tmp_path, caplog, field):  # noqa: F811
    from ouroboros.skill_install_record import discard_update_snapshot, restore_update_snapshot

    drive, _folder, _row, item = release
    _pending(release)
    backup_file = skill_state_dir(drive, "demo") / "update_backup.json"
    data = json.loads(backup_file.read_text(encoding="utf-8"))
    kept_payload = pathlib.Path(data["backup_dir"])
    unrelated = tmp_path / "elsewhere" / pathlib.Path(data[field]).name
    unrelated.mkdir(parents=True)
    marker = unrelated / "keep.txt"
    marker.write_text("unrelated files", encoding="utf-8")
    data[field] = str(unrelated)
    backup_file.write_text(json.dumps(data), encoding="utf-8")
    evidence = backup_file.read_bytes()
    assert not restore_update_snapshot(drive, "demo", "review_unfinished")
    discard_update_snapshot(drive, "demo")
    assert marker.read_text(encoding="utf-8") == "unrelated files"
    assert kept_payload.is_dir() and backup_file.read_bytes() == evidence
    assert find_skill(drive, "demo").content_hash == item["digest"]
    assert "paths do not match" in caplog.text


@pytest.mark.parametrize("failure", ["source_land", "source_record", "builtin_copy", "builtin_digest"])
def test_update_landing_failure_restores_with_landing_reason(release, tmp_path, monkeypatch, failure):  # noqa: F811
    drive, folder, row, item = release
    if failure.startswith("builtin"):
        folder.parent.rename(tmp_path / "skills")
        folder = tmp_path / "skills" / "demo"
        row["kind"] = "builtin"
        monkeypatch.setattr("ouroboros.config.REPO_DIR", tmp_path)
    previous, _ = _pending((drive, folder, row, item))
    installer.settle_pending_update(drive, "demo", force=True)
    def fail(*args, **kwargs):
        raise OSError("injected landing failure")
    if failure.startswith("builtin"):
        monkeypatch.setattr("ouroboros.skill_builtin.install_builtin_skill", lambda *a, **kw: {
            "ok": False, "code": "copy_failed" if failure == "builtin_copy" else "digest_mismatch",
            "error": "injected landing failure"})
    elif failure == "source_record":
        monkeypatch.setattr(installer, "write_install_record", fail)
    else:
        monkeypatch.setattr("ouroboros.marketplace.fetcher.land_staged_tree", fail)
    with pytest.raises((OSError, installer.SkillInstallError), match="injected landing failure"):
        installer.update_installed(drive, "demo", mode="replace", rows=[row])
    assert find_skill(drive, "demo").content_hash == previous
    assert read_install_record(drive, "demo")["update_rejected"]["reason"] == "landing_failed"
    assert not (skill_state_dir(drive, "demo") / "update_backup.json").exists()


@pytest.mark.parametrize("failure", ["land", "record", "dependencies"])
def test_registry_landing_failure_restores_with_landing_reason(registry_release, monkeypatch, failure):  # noqa: F811
    drive, storage, registry = registry_release
    previous = read_install_record(drive, "demo")
    registry["latest"] = "2.0.0"
    def fail(*args, **kwargs):
        raise OSError("injected registry failure")
    if failure == "dependencies":
        adapt = storage.adapt_openclaw_skill
        def with_deps(*args, **kwargs):
            result = adapt(*args, **kwargs)
            result.provenance["install_specs"]["auto"] = [{"kind": "pip", "package": "demo"}]
            return result
        monkeypatch.setattr(storage, "adapt_openclaw_skill", with_deps)
        monkeypatch.setattr("ouroboros.marketplace.isolated_deps.install_isolated_dependencies", fail)
    elif failure == "record":
        from ouroboros.skill_install_record import write_install_record
        def write(drive, name, record):
            if record["release"]["version"] == "2.0.0":
                fail()
            return write_install_record(drive, name, record)
        monkeypatch.setattr("ouroboros.skill_install_record.write_install_record", write)
    else:
        monkeypatch.setattr(storage, "_land_staged_into_data_plane", fail)
    result = storage.update_skill(drive, drive.parent, sanitized_name="demo")
    assert not result.ok and "injected registry failure" in result.error
    assert find_skill(drive, "demo").content_hash == previous["release"]["digest"]
    assert read_install_record(drive, "demo")["update_rejected"]["reason"] == "landing_failed"


def test_dependency_failure_after_review_restores_with_landing_reason(release, monkeypatch):  # noqa: F811
    drive, _folder, _row, _item = release
    previous, _ = _pending(release)
    monkeypatch.setattr("ouroboros.skill_dependencies.ensure_skill_dependencies",
                        lambda *a, **kw: ("failed", "injected dependency failure"))
    result = _review(drive, "clean")
    assert result["deps_status"] == "failed"
    assert find_skill(drive, "demo").content_hash == previous
    assert read_install_record(drive, "demo")["update_rejected"]["reason"] == "landing_failed"


def test_startup_returns_before_slow_update_settlement_finishes(tmp_path, monkeypatch):
    from ouroboros.skill_starter import at_startup

    entered, release_settle, finished, returned = (threading.Event() for _ in range(4))
    errors = []
    def slow_settle(drive):
        entered.set()
        try:
            assert release_settle.wait(60)
        finally:
            finished.set()
    monkeypatch.setattr(installer, "settle_pending_updates_at_startup", slow_settle)
    def startup():
        try:
            assert at_startup(tmp_path, {}, provider_ready=False) == ""
            returned.set()
        except Exception as exc:
            errors.append(exc)
    caller = threading.Thread(target=startup)
    caller.start()
    try:
        assert entered.wait(30)
        assert returned.wait(5), "startup waited for the restore instead of scheduling it"
        assert not finished.is_set()
    finally:
        release_settle.set()
        caller.join(60)
        assert finished.wait(30)
    assert not caller.is_alive() and not errors


@pytest.mark.parametrize("trust,auto_review", [("review", False), ("trusted", False), ("review", True)])
def test_registry_update_uses_new_trust_and_retains_review_rollback(tmp_path, monkeypatch, trust, auto_review):
    from ouroboros.marketplace.clawhub import ClawHubSkillSummary
    from tests.test_marketplace_api import _patch_clawhub_install_pipeline

    drive = tmp_path / "data"
    row = {"id": "registry", "name": "Registry", "kind": "registry", "url": "https://example.invalid",
           "trust": "trusted" if trust == "review" else "review", "confirmed": True}
    monkeypatch.setenv("OUROBOROS_SKILL_SOURCES", json.dumps([row]))
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setattr("ouroboros.config.SETTINGS_PATH", drive / "settings.json")
    monkeypatch.delenv("OUROBOROS_SKILLS_REPO_PATH", raising=False)
    storage = _patch_clawhub_install_pipeline(monkeypatch, tmp_path, auto_specs=[])
    first = storage.install_skill(drive, tmp_path, slug="demo", auto_review=False)
    assert first.ok
    old = find_skill(drive, "demo")
    save_review_state(drive, "demo", SkillReviewState(status="clean", content_hash=old.content_hash))
    previous = read_install_record(drive, "demo")
    monkeypatch.setattr(storage, "_registry_info", lambda slug: ClawHubSkillSummary(slug=slug, latest_version="2.0.0"))
    original_stage = storage._stage_archive
    def stage(*args, **kwargs):
        staged = original_stage(*args, **kwargs)
        manifest = staged.staging_dir / "SKILL.md"
        manifest.write_text(manifest.read_text(encoding="utf-8").replace("1.0.0", "2.0.0"), encoding="utf-8")
        return staged
    monkeypatch.setattr(storage, "_stage_archive", stage)
    row["trust"] = trust
    monkeypatch.setenv("OUROBOROS_SKILL_SOURCES", json.dumps([row]))
    if auto_review:
        def blockers(drive, repo, name):
            skill = find_skill(drive, name)
            save_review_state(drive, name, SkillReviewState(status="blockers", content_hash=skill.content_hash))
            return "blockers", [], ""
        monkeypatch.setattr(storage, "_run_skill_review", blockers)
        result = installer.run_in_lane(drive, "update", "demo", lambda: storage.update_skill(
            drive, tmp_path, sanitized_name="demo"))
        assert result.ok and result.review_status == "blockers"
        assert find_skill(drive, "demo").content_hash == old.content_hash
        assert read_install_record(drive, "demo")["update_rejected"]["reason"] == "review_blockers"
        return
    result = installer.update_installed(drive, "demo", mode="replace", local_digest=old.content_hash)
    record = read_install_record(drive, "demo")
    assert result["basis"] == record["basis"] == trust
    if trust == "trusted":
        assert result["acceptance"]["accepted"] and not result.get("needs_review")
        assert "pending_update" not in record
    else:
        assert result["needs_review"] and "acceptance" not in result
        assert record["pending_update"]["from_digest"] == old.content_hash
        _review(drive, "blockers")
        assert find_skill(drive, "demo").content_hash == old.content_hash
        restored = read_install_record(drive, "demo")
        assert restored["release"] == previous["release"]
        assert restored["update_rejected"]["reason"] == "review_blockers"
