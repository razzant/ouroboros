"""Direct ClawHub consumers hand current bytes to the shared dependency owner after review.

Registry transport and reviewer results are controlled. The real landing, loader, gate,
dependency records, update settlement and HTTP projection run; only package execution
and extension processes are replaced, so these checks install no external software.
"""

from __future__ import annotations

import asyncio
import json
import logging
import pathlib
from types import SimpleNamespace

import pytest

from ouroboros.gateway import marketplace as api
from ouroboros.marketplace import install
from ouroboros.marketplace.clawhub import ClawHubClientError
from ouroboros.marketplace.isolated_deps import read_deps_state
from ouroboros.skill_dependencies import skill_deps_not_ready
from ouroboros.skill_install_record import read_install_record
from ouroboros.skill_loader import SkillReviewState, find_skill, save_enabled, save_review_state
from ouroboros.skill_review import SkillReviewOutcome
from tests._shared import current_python_env
from tests.test_marketplace_api import (
    _BodyRequest,
    _patch_clawhub_install_pipeline,
    _run_lifecycle_inline,
    _stub_marketplace_roots,
)


@pytest.fixture
def registry(tmp_path, monkeypatch):
    # Load the asynchronous consumer before replacing the public review function;
    # otherwise its import-time default would retain this fixture's stub after teardown.
    from ouroboros import skill_review_runner  # noqa: F401

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", "")
    monkeypatch.setenv("OUROBOROS_SKILL_SOURCES", json.dumps([
        {"id": "clawhub", "name": "ClawHub", "kind": "registry", "trust": "review", "confirmed": True}]))
    original_review = install._run_skill_review
    specs = [{"kind": "pip", "package": "ddgs", "bins": []}]
    _patch_clawhub_install_pipeline(monkeypatch, tmp_path, auto_specs=specs)
    monkeypatch.setattr(install, "_run_skill_review", original_review)
    original_stage = install._stage_archive

    def stage(*args, **kwargs):
        staged = original_stage(*args, **kwargs)
        manifest = staged.staging_dir / "SKILL.md"
        manifest.write_text(manifest.read_text().replace("1.0.0", kwargs["version"]), encoding="utf-8")
        sidecar = staged.staging_dir / ".clawhub.json"
        sidecar.write_text(json.dumps({**json.loads(sidecar.read_text()), "install_specs": {"auto": specs}}))
        return staged

    monkeypatch.setattr(install, "_stage_archive", stage)
    state = SimpleNamespace(drive=tmp_path, repo=tmp_path / "repo", events=[], feedback="clean", fail_packages=False)

    def review(_ctx, name):
        state.events.append("review")
        loaded = find_skill(tmp_path, name, repo_path="")
        if state.feedback == "unavailable":
            raise RuntimeError("reviewer unavailable")
        status = state.feedback
        findings, author = [], {}
        if status.startswith("author_"):
            from ouroboros.review_records import build_author_disposition

            digest = "old-hash" if status == "author_stale" else loaded.content_hash
            author = build_author_disposition(disposition="accepted", rationale="Current files checked locally.",
                                              subject_hash=digest, enforcement="advisory", action="finish")
            author["review_reference"] = {
                "surface": "skill", "basis": "unavailable", "job_id": "finished-review",
                "content_hash": digest, "finished_at": "2026-10-25T12:00:00Z",
                "status": "running" if status == "author_running" else "failed", "review_status": "pending",
                "preflight": {"content_hash": digest, "status": "pass"}}
            status = "pending"
        elif status == "stale":
            status = "clean"
        elif status in {"blockers", "warnings"}:
            findings = [{"item": "bug_hunting", "verdict": "FAIL", "reason": "Controlled finding.",
                         "severity": "critical" if status == "blockers" else "advisory"}]
        save_review_state(tmp_path, name, SkillReviewState(
            status=status, content_hash=loaded.content_hash, findings=findings, author_disposition=author))
        if state.feedback == "stale":
            (loaded.skill_dir / "changed.txt").write_text("changed during review", encoding="utf-8")
        return SkillReviewOutcome(skill_name=name, status=status, content_hash=loaded.content_hash,
                                  findings=findings, error="reviewer unavailable" if status == "pending" else "")

    def packages(_packages, env_root, *args, **kwargs):
        state.events.append("packages")
        if state.fail_packages:
            raise RuntimeError("package download failed")
        current_python_env(env_root)
        return []

    monkeypatch.setattr("ouroboros.skill_review.review_skill", review)
    monkeypatch.setattr("ouroboros.marketplace.isolated_deps._install_python_packages", packages)
    _stub_marketplace_roots(monkeypatch, tmp_path)
    _run_lifecycle_inline(monkeypatch)
    return state


@pytest.mark.parametrize("mode,enforcement,feedback,converted,admitted", [
    ("advanced", "blocking", "clean", False, True),
    ("advanced", "blocking", "warnings", False, True),
    ("advanced", "blocking", "pending", False, False),
    ("advanced", "blocking", "blockers", False, False),
    ("advanced", "blocking", "stale", False, False),
    ("advanced", "advisory", "pending", False, False),
    ("advanced", "advisory", "blockers", False, False),
    ("advanced", "advisory", "author_accepted", True, True),
    ("advanced", "advisory", "author_accepted", False, False),
    ("advanced", "advisory", "author_stale", True, False),
    ("advanced", "advisory", "author_running", True, False),
    ("cyber_pro", "blocking", "unavailable", False, True),
])
def test_direct_install_uses_current_acceptance(registry, monkeypatch, mode, enforcement, feedback, converted, admitted):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", mode)
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", enforcement)
    registry.feedback = feedback
    record = {"source": {"id": "clawhub", "kind": "registry"}, "basis": "review"}
    if converted:
        record["converted_from"] = ".clawhub.json"
    result = install.install_skill(registry.drive, registry.repo, slug="demo", install_record=record)
    loaded = find_skill(registry.drive, "demo", repo_path="")

    assert result.ok
    assert loaded.review.gate_for(loaded.content_hash)["executable_review"] is admitted
    assert registry.events == (["review", "packages"] if admitted else ["review"])
    assert result.deps_status == ("installed" if admitted else "pending_review")
    if admitted:
        assert skill_deps_not_ready(registry.drive, loaded, strict=True) == ("", "")
        assert result.deps_fingerprint["status"] == "installed"
    else:
        assert not read_deps_state(registry.drive, "demo")
    if feedback.startswith("author_") or feedback in {"pending", "unavailable"}:
        assert result.review_status == "pending" and "reviewer unavailable" in result.review_error
        assert loaded.review.status == "pending"


@pytest.mark.parametrize("operation", ["install", "update"])
def test_http_cyber_review_error_keeps_dependencies_ready_before_reload(registry, monkeypatch, operation):
    if operation == "update":
        assert install.install_skill(registry.drive, registry.repo, slug="demo").ok
        save_enabled(registry.drive, "demo", True, actor="owner_ui")
        registry.events.clear()
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    registry.feedback = "unavailable"
    monkeypatch.setattr("ouroboros.extension_loader.is_extension_live", lambda *a: True)
    monkeypatch.setattr("ouroboros.extension_loader.unload_extension", lambda *a: registry.events.append("unload"))

    def reload(name, drive, _settings):
        loaded = find_skill(drive, name, repo_path="")
        assert loaded.manifest.version == "2.0.0"
        assert loaded.review.gate_for(loaded.content_hash)["executable_review"]
        assert skill_deps_not_ready(drive, loaded, strict=True) == ("", "")
        registry.events.append("reload")

    monkeypatch.setattr("ouroboros.extension_loader.reconcile_extension", reload)
    request = _BodyRequest({"slug": "demo", "version": "2.0.0"}, path_params={"name": "demo"})
    response = asyncio.run((api.api_marketplace_update if operation == "update" else api.api_marketplace_install)(request))
    result = json.loads(response.body)
    assert response.status_code == 200 and result["ok"]
    assert result["review_status"] == "pending" and "reviewer unavailable" in result["review_error"]
    assert result["deps_status"] == "installed" and result["deps_fingerprint"]["status"] == "installed"
    assert registry.events == (["unload", "review", "packages", "reload"] if operation == "update"
                               else ["review", "packages"])
    loaded = find_skill(registry.drive, "demo", repo_path="")
    assert skill_deps_not_ready(registry.drive, loaded, strict=True) == ("", "")
    if operation == "update":
        assert loaded.enabled and not read_install_record(registry.drive, "demo").get("pending_update")


@pytest.mark.parametrize("feedback", ["pending", "blockers", "stale"])
def test_blocking_update_restores_previous_payload_and_dependencies(registry, feedback):
    first = install.install_skill(registry.drive, registry.repo, slug="demo")
    before = (first.target_dir / "SKILL.md").read_bytes()
    previous_deps = read_deps_state(registry.drive, "demo", first.target_dir)
    registry.events.clear()
    registry.feedback = feedback

    result = install.update_skill(registry.drive, registry.repo, sanitized_name="demo", version="2.0.0")

    assert result.ok and result.deps_status == "pending_review"
    assert registry.events == ["review"]
    assert (first.target_dir / "SKILL.md").read_bytes() == before
    assert read_deps_state(registry.drive, "demo", first.target_dir) == previous_deps
    assert read_install_record(registry.drive, "demo")["update_rejected"]["version"] == "2.0.0"


@pytest.mark.parametrize("failure", ["download", "packages"])
def test_cyber_update_failure_restores_previous_payload_and_dependencies(registry, monkeypatch, failure):
    first = install.install_skill(registry.drive, registry.repo, slug="demo")
    before = (first.target_dir / "SKILL.md").read_bytes()
    previous_deps = read_deps_state(registry.drive, "demo", first.target_dir)
    registry.events.clear()
    registry.feedback, registry.fail_packages = "unavailable", True
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    if failure == "download":
        def download(*args, **kwargs):
            raise ClawHubClientError("registry download failed")
        monkeypatch.setattr(install, "_registry_download", download)

    result = install.update_skill(registry.drive, registry.repo, sanitized_name="demo", version="2.0.0")

    assert not result.ok and "download failed" in result.error
    if failure == "packages":
        assert result.deps_status == "failed" and "reviewer unavailable" in result.review_error
        assert registry.events == ["review", "packages"]
    else:
        assert registry.events == []
    assert (first.target_dir / "SKILL.md").read_bytes() == before
    assert read_deps_state(registry.drive, "demo", first.target_dir) == previous_deps


def test_unreadable_payload_does_not_reach_packages_in_cyber(registry, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    original_open = pathlib.Path.open

    def review(_ctx, _name):
        target = registry.drive / "skills/clawhub/demo/payload.txt"
        target.write_text("unreadable", encoding="utf-8")

        def open_file(path, *args, **kwargs):
            if path == target:
                raise PermissionError("payload read denied")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(pathlib.Path, "open", open_file)
        return SkillReviewOutcome(skill_name="demo", status="pending", error="payload read denied")

    monkeypatch.setattr("ouroboros.skill_review.review_skill", review)
    result = install.install_skill(registry.drive, registry.repo, slug="demo")
    assert result.review_error == "payload read denied" and result.deps_status == "pending_review"
    assert "payload unreadable" in find_skill(registry.drive, "demo", repo_path="").load_error
    assert registry.events == [] and not read_deps_state(registry.drive, "demo")


def test_reload_exception_remains_separate_from_review_and_dependencies(registry, monkeypatch, caplog):
    assert install.install_skill(registry.drive, registry.repo, slug="demo").ok
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    registry.feedback = "unavailable"
    registry.events.clear()
    monkeypatch.setattr("ouroboros.extension_loader.is_extension_live", lambda *a: True)
    monkeypatch.setattr("ouroboros.extension_loader.unload_extension", lambda *a: None)

    def reload(*args):
        registry.events.append("reload")
        raise RuntimeError("extension reload failed")

    monkeypatch.setattr("ouroboros.extension_loader.reconcile_extension", reload)
    with caplog.at_level(logging.DEBUG, logger="ouroboros.marketplace.install"):
        result = install.update_skill(registry.drive, registry.repo, sanitized_name="demo", version="2.0.0")

    # The legacy API logs reload exceptions; it does not turn them into review verdicts
    # or dependency failures, nor restore an already accepted update for that reason.
    assert registry.events == ["review", "packages", "reload"]
    assert "extension reload failed" in caplog.text
    assert result.ok and result.deps_status == "installed"
    assert result.review_status == "pending" and "reviewer unavailable" in result.review_error
    assert find_skill(registry.drive, "demo", repo_path="").manifest.version == "2.0.0"
