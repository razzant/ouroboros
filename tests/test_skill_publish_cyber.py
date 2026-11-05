"""Cyber publication keeps criticism independent from actual delivery facts."""

from __future__ import annotations

import json

import pytest

from ouroboros.gateway import skill_publish as preflight
from ouroboros.skill_publish_eligibility import submit_hub_eligibility
from ouroboros.skill_publish_scanner import scan_named_bytes
from ouroboros.skill_publish_snapshot import SkillPublishSnapshotError, capture_skill_publish_snapshot
from ouroboros.tools import skill_publish
from tests.test_skill_publish import _loaded, _patch_validate
from tests.test_skill_publish_preflight import _build, _contract, _patch_domain, _scan
from tests.test_skill_publish_preflight import _loaded as preflight_loaded, _snapshot as preflight_snapshot
from tests.test_skill_publish_snapshot import _reviewed_skill, _write_skill
from tests.test_skill_publish_transaction import (
    _finding, _install_transaction_fakes, _scan_result, _snapshot, _submit,
)


@pytest.fixture(autouse=True)
def cyber(monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    preflight._PREFLIGHT_SCAN_CACHE.clear()
    yield
    preflight._PREFLIGHT_SCAN_CACHE.clear()


@pytest.mark.parametrize("path", [
    "SKILL.md", "catalog.json", "pull-request-body.md",
])
def test_high_finding_preserves_original_facts_and_publishes_once(monkeypatch, tmp_path, path):
    finding = _finding(path, confidence="high", disposition="blocker")

    def scanner(named):
        return _scan_result(finding) if path in named else _scan_result()

    ctx, events, captured = _install_transaction_fakes(
        monkeypatch, tmp_path, snapshot=_snapshot(), scanner=scanner,
    )
    result = _submit(ctx, note="Public note.")

    assert result["ok"] is True
    assert result["safety_advisory"] is True
    assert result["blocker_count"] == 1
    assert result["findings"][0]["disposition"] == "blocker"
    assert result["findings"][0]["confidence"] == "high"
    assert result["scanner_status"] == "findings"
    assert captured["llm_calls"] == 1
    assert "Model summary" in captured["pr_body"]
    assert "blockers=1" in captured["pr_body"]
    assert "RAW_REVIEW_REASON" in captured["pr_body"]
    assert "Fresh clean review" not in captured["pr_body"]
    assert [row[1] for row in events if row[0] == "mutation"] == ["fork", "branch", "commit", "pr"]


@pytest.mark.parametrize("reason", [
    "scanner_missing", "scanner_corrupt", "scanner_input_invalid",
])
def test_scanner_failure_is_disclosed_without_erasing_real_pr(monkeypatch, tmp_path, reason):
    ctx, events, captured = _install_transaction_fakes(
        monkeypatch, tmp_path, snapshot=_snapshot(),
        scanner=lambda _named: _scan_result(reason_code=reason),
    )
    create = skill_publish.create_pr_receipt

    def actual_scanner_receipt(ctx, attempt, **kwargs):
        receipt = create(ctx, attempt, **kwargs)
        receipt["ruleset_sha256"] = attempt.scanner.get("ruleset_sha256", "")
        return receipt

    monkeypatch.setattr(skill_publish, "create_pr_receipt", actual_scanner_receipt)
    result = _submit(ctx)
    assert result["ok"] is True
    assert result["status"] == "pr_opened"
    assert result["scanner_status"] == "scanner_error"
    assert reason in result["scanner_errors"]
    assert result["receipt"]["ruleset_sha256"] == ""
    assert reason in captured["pr_body"]
    assert captured["llm_calls"] == 1
    assert [row[1] for row in events if row[0] == "mutation"] == ["fork", "branch", "commit", "pr"]


def test_cyber_permission_flag_does_not_require_owner_confirmation(monkeypatch, tmp_path):
    ctx, _events, _captured = _install_transaction_fakes(monkeypatch, tmp_path, snapshot=_snapshot())
    result = json.loads(skill_publish._submit_skill_to_hub(ctx, "demo"))
    assert result["ok"] is True


def test_publication_keeps_old_negative_review_separate_from_current_bytes(monkeypatch, tmp_path):
    ctx, _events, captured = _install_transaction_fakes(monkeypatch, tmp_path, snapshot=_snapshot())
    _skill, loaded = skill_publish._validate_local_skill(ctx, "demo")
    loaded.review.status = "blockers"
    loaded.review.content_hash = "e" * 64
    result = _submit(ctx)
    assert result["ok"] is True
    assert result["review_status"] == "blockers" and result["review_stale"] is True
    assert result["reviewed_content_hash"] == "e" * 64
    assert result["snapshot_hash"] == _snapshot().content_hash
    assert "Local review status: blockers" in captured["pr_body"]
    assert "RAW_REVIEW_REASON" in captured["pr_body"]
    assert loaded.review.findings[0]["verdict"] == "FAIL"


@pytest.mark.parametrize("status,profile,source", [
    ("blockers", "", "external"), ("pending", "", "external"),
    ("clean", "owner_attested", "external"),
])
def test_local_review_and_source_policy_are_advisory(monkeypatch, tmp_path, status, profile, source):
    loaded = _loaded(status=status, source=source)
    loaded.skill_dir = tmp_path / "demo"
    loaded.skill_dir.mkdir()
    if source == "native":
        (loaded.skill_dir / ".seed-origin").write_text("builtin")
    loaded.review.review_profile = profile
    _patch_validate(monkeypatch, loaded)
    _skill, returned = skill_publish._validate_local_skill(None, "demo")
    assert returned is loaded
    assert returned.review.status == status
    eligibility = submit_hub_eligibility(
        source=source, review_status=status, review_profile=profile,
        review_stale=True, github_token_configured=True,
    )
    assert eligibility["visible"] and eligibility["task_start_allowed"]


def test_capture_uses_current_bytes_without_rewriting_critic_hash(tmp_path):
    skill_dir = _write_skill(tmp_path / "skills" / "demo")
    loaded = _reviewed_skill(skill_dir, tmp_path)
    critic_hash = loaded.review.content_hash
    (skill_dir / "payload.txt").write_bytes(b"current exact bytes")
    snapshot = capture_skill_publish_snapshot(loaded)
    assert snapshot.content_hash != critic_hash
    assert snapshot.file("payload.txt").content == b"current exact bytes"
    assert loaded.review.content_hash == critic_hash
    assert loaded.review.is_stale_for(snapshot.content_hash)


def test_missing_payload_is_still_a_capture_failure(tmp_path):
    skill_dir = _write_skill(tmp_path / "skills" / "demo")
    loaded = _reviewed_skill(skill_dir, tmp_path)
    (skill_dir / "SKILL.md").unlink()
    with pytest.raises(SkillPublishSnapshotError, match="snapshot_manifest_missing"):
        capture_skill_publish_snapshot(loaded)


@pytest.mark.parametrize("scanner_error", [False, True])
def test_selected_preflight_keeps_criticism_but_allows_cyber_publication(monkeypatch, tmp_path, scanner_error):
    loaded = preflight_loaded(status="blockers", review_hash="e" * 64, profile="owner_attested")
    scan = (_scan(status="scanner_error", contract="", reason_code="scanner_report_invalid", repair_hint="Original failure hint")
            if scanner_error else _scan((_finding("SKILL.md", confidence="high", disposition="blocker"),)))
    calls = []
    _patch_domain(monkeypatch, loaded, preflight_snapshot(), scan, calls)
    if scanner_error:
        monkeypatch.setattr(preflight, "probe_scanner_contract", lambda **_kw: _contract(
            status="scanner_error", reason_code=scan.reason_code, repair_hint=scan.repair_hint,
        ))
    payload = _build(tmp_path).payload
    assert payload["publication_ready"] is True
    assert payload["state"] == "warnings"
    assert payload["review"]["status"] == "blockers"
    assert payload["review"]["stale"] is True
    assert payload["review"]["profile"] == "owner_attested"
    assert payload["scanner"]["status"] == scan.status
    assert payload["blocker_count"] == scan.blocker_count
    if scanner_error:
        assert payload["scanner"]["reason_code"] == "scanner_report_invalid"
        assert payload["scanner"]["repair_hint"] == "Original failure hint"


def test_actual_github_failure_remains_failure_after_advice(monkeypatch, tmp_path):
    ctx, _events, _captured = _install_transaction_fakes(
        monkeypatch, tmp_path, snapshot=_snapshot(),
        scanner=lambda _named: _scan_result(reason_code="scanner_missing"),
    )
    calls = []

    def fail(*_args, **_kwargs):
        calls.append("prepare")
        raise skill_publish.SkillPublishGitHubError(
            "fork_sync_failed", "Check GitHub access.", detail="HTTP 403", http_status=403,
        )

    monkeypatch.setattr(skill_publish, "prepare_publish_repository", fail)
    result = _submit(ctx)
    assert result["ok"] is False
    assert result["reason_code"] == "fork_sync_failed"
    assert result["github_status"] == 403
    assert result["scanner_status"] == "scanner_error"
    assert "scanner_missing" in result["scanner_errors"]
    assert "receipt" not in result
    assert calls == ["prepare"]


def test_real_cached_scanner_high_finding_reaches_one_simulated_pr(monkeypatch, tmp_path):
    from tests.test_skill_publish_scanner_real import _binary, _executable

    binary = _binary()  # Existing cache only; never downloads or installs.
    candidate = "".join(("gh", "p_", "aB3dE5fG", "7hJ9kL2m", "N4pQ6rS8", "tV0wX2yZ", "5cD7"))
    ctx, events, captured = _install_transaction_fakes(
        monkeypatch, tmp_path, snapshot=_snapshot(body=f"token = {candidate!r}\n".encode()),
    )
    monkeypatch.setattr(skill_publish, "_scanner_executable", lambda _ctx: _executable(binary))
    monkeypatch.setattr(skill_publish, "scan_named_bytes", scan_named_bytes)
    create = skill_publish.create_pr_receipt

    def actual_scanner_receipt(ctx, attempt, **kwargs):
        receipt = create(ctx, attempt, **kwargs)
        receipt["ruleset_sha256"] = attempt.scanner["ruleset_sha256"]
        return receipt

    monkeypatch.setattr(skill_publish, "create_pr_receipt", actual_scanner_receipt)
    result = _submit(ctx)
    assert result["ok"] is True and result["blocker_count"] == 1
    assert result["findings"][0]["detector"] == "github-pat"
    assert result["findings"][0]["disposition"] == "blocker"
    assert candidate not in json.dumps(result)
    assert "blockers=1" in captured["pr_body"]
    assert [row[1] for row in events if row[0] == "mutation"] == ["fork", "branch", "commit", "pr"]
