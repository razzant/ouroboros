"""End-to-end, network-free tests for the single publication transaction."""

from __future__ import annotations

import base64
import json
import pathlib
import types
from typing import Callable

import pytest

from ouroboros.config import SKILL_SOURCE_EXTERNAL
from ouroboros.skill_loader import SkillReviewState
from ouroboros.skill_publish_scanner import (
    ScannerExecutable,
    SecretFinding,
    SecretScanResult,
)
from ouroboros.skill_publish_snapshot import (
    CapturedPublishManifest,
    CapturedSkillFile,
    SkillPublishSnapshot,
)
from ouroboros.tools import skill_publish
from ouroboros.tools.registry import ToolContext

SNAPSHOT_SHA = "a" * 64
RULESET_SHA = "b" * 64
BASE_SHA = "1" * 40
COMMIT_SHA = "2" * 40


def _snapshot(
    *,
    body: bytes = b"# exact captured body\n",
    description: str = "A safe test skill.",
) -> SkillPublishSnapshot:
    manifest_file = CapturedSkillFile.from_bytes("skill.json", b'{"name":"demo","version":"1.0.0"}')
    skill_file = CapturedSkillFile.from_bytes("SKILL.md", body)
    return SkillPublishSnapshot(
        skill="demo",
        source=SKILL_SOURCE_EXTERNAL,
        manifest_file=manifest_file,
        manifest=CapturedPublishManifest(
            path="skill.json",
            name="demo",
            description=description,
            version="1.0.0",
            skill_type="instruction",
            when_to_use="Use for a test.",
        ),
        content_hash=SNAPSHOT_SHA,
        full_files=(manifest_file, skill_file),
        public_files=(manifest_file, skill_file),
        control_files=(),
    )


def _finding(
    path: str,
    *,
    confidence: str = "medium",
    disposition: str = "warning",
) -> SecretFinding:
    return SecretFinding(
        path=path,
        line=1,
        detector="test-detector",
        confidence=confidence,
        reason="Scanner finding requires review before publication.",
        verification="not_attempted",
        disposition=disposition,
    )


def _scan_result(*findings: SecretFinding, reason_code: str = "") -> SecretScanResult:
    if reason_code:
        return SecretScanResult(
            status="scanner_error",
            engine="betterleaks",
            version="",
            ruleset_sha256="",
            scan_contract_sha256="",
            findings=(),
            blocker_count=0,
            warning_count=0,
            audited_false_positive_count=0,
            reason_code=reason_code,
            repair_hint=("Run `python -m ouroboros.betterleaks_runtime install`, then retry."),
        )
    rows = tuple(findings)
    return SecretScanResult(
        status="findings" if rows else "clean",
        engine="betterleaks",
        version="1.8.1",
        ruleset_sha256=RULESET_SHA,
        scan_contract_sha256="c" * 64,
        findings=rows,
        blocker_count=sum(row.disposition == "blocker" for row in rows),
        warning_count=sum(row.disposition == "warning" for row in rows),
        audited_false_positive_count=sum(row.disposition == "audited_false_positive" for row in rows),
    )


def _install_transaction_fakes(
    monkeypatch,
    tmp_path: pathlib.Path,
    *,
    snapshot: SkillPublishSnapshot,
    scanner: Callable[[dict[str, bytes]], SecretScanResult] | None = None,
    model_body: str = "## Summary\nModel summary.\n\n## What This Skill Does\nSafe.",
):
    events = []
    captured = {
        "prompts": [],
        "additions": [],
        "deletions": [],
        "pr_body": "",
        "llm_calls": 0,
    }
    review = SkillReviewState(
        status="warnings",
        content_hash=SNAPSHOT_SHA,
        findings=[
            {
                "item": "bug_hunting",
                "verdict": "FAIL",
                "severity": "advisory",
                "reason": "RAW_REVIEW_REASON",
            }
        ],
    )
    loaded = types.SimpleNamespace(review=review, source=SKILL_SOURCE_EXTERNAL)
    monkeypatch.setattr(skill_publish, "_validate_local_skill", lambda *_args: ("demo", loaded))
    monkeypatch.setattr(
        skill_publish,
        "get_ouroboroshub_catalog_url",
        lambda: "https://raw.githubusercontent.com/hub/project/main/catalog.json",
    )
    monkeypatch.setattr(skill_publish, "capture_skill_publish_snapshot", lambda _loaded: snapshot)
    monkeypatch.setattr(
        skill_publish,
        "_scanner_executable",
        lambda _ctx: ScannerExecutable(path=tmp_path / "betterleaks", identity="d" * 64, status="ready"),
    )

    def fake_scan(named_bytes, **_kwargs):
        copied = {str(key): bytes(value) for key, value in named_bytes.items()}
        events.append(("scan", tuple(copied)))
        return scanner(copied) if scanner else _scan_result()

    monkeypatch.setattr(skill_publish, "scan_named_bytes", fake_scan)
    monkeypatch.setattr(skill_publish, "github_login", lambda _ctx: "alice")
    monkeypatch.setattr(
        skill_publish,
        "fetch_upstream_catalog",
        lambda *_args: ({"skills": []}, BASE_SHA),
    )

    def prepare(_ctx, attempt, **_kwargs):
        events.append(("mutation", "fork"))
        attempt.mark("fork_ready", repository="alice/project", actor="alice")
        attempt.mark("fork_synced", repository="alice/project", actor="alice")

    def branch(*_args):
        events.append(("mutation", "branch"))
        return BASE_SHA

    def commit(_ctx, _login, _repo, _branch, _sha, _title, additions, deletions):
        events.append(("mutation", "commit"))
        captured["additions"] = additions
        captured["deletions"] = deletions
        return COMMIT_SHA, "https://github.com/alice/project/commit/" + COMMIT_SHA

    def create(_ctx, attempt, **kwargs):
        events.append(("mutation", "pr"))
        captured["pr_body"] = kwargs["body"]
        attempt.mark(
            "pr_create_attempted",
            repository="hub/project",
            actor="alice",
            branch=kwargs["branch"],
            commit_sha=kwargs["commit_sha"],
        )
        return {
            "kind": "github_pull_request",
            "repository": "hub/project",
            "url": "https://github.com/hub/project/pull/7",
            "number": 7,
            "skill": "demo",
            "snapshot_hash": SNAPSHOT_SHA,
            "ruleset_sha256": RULESET_SHA,
        }

    monkeypatch.setattr(skill_publish, "prepare_publish_repository", prepare)
    monkeypatch.setattr(skill_publish, "ensure_branch", branch)
    monkeypatch.setattr(skill_publish, "commit_payload", commit)
    monkeypatch.setattr(skill_publish, "create_pr_receipt", create)

    class FakeLLM:
        def chat(self, **kwargs):
            captured["llm_calls"] += 1
            captured["prompts"].append(kwargs["messages"][0]["content"])
            return {"content": model_body}, {}

    monkeypatch.setattr(skill_publish, "LLMClient", FakeLLM)
    return ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="task-1"), events, captured


def _submit(ctx: ToolContext, **kwargs):
    return json.loads(
        skill_publish._submit_skill_to_hub(
            ctx,
            "demo",
            confirm_public_submission=True,
            **kwargs,
        )
    )


def test_success_scans_every_outbound_artifact_before_first_mutation(monkeypatch, tmp_path):
    snapshot = _snapshot(body=b"RAW_SKILL_BODY")
    ctx, events, captured = _install_transaction_fakes(monkeypatch, tmp_path, snapshot=snapshot)
    result = _submit(ctx, note="public note")
    assert result["ok"] is True
    assert result["receipt"]["url"].endswith("/pull/7")
    first_mutation = next(index for index, row in enumerate(events) if row[0] == "mutation")
    assert all(row[0] == "scan" for row in events[:first_mutation])
    assert captured["llm_calls"] == 1
    prompt = captured["prompts"][0]
    assert "RAW_SKILL_BODY" not in prompt
    assert "RAW_REVIEW_REASON" not in prompt
    committed = {row["path"]: base64.b64decode(row["contents"]) for row in captured["additions"]}
    assert committed["skills/demo/SKILL.md"] == b"RAW_SKILL_BODY"
    assert captured["pr_body"].count("## Author Checklist") == 1
    assert captured["pr_body"].count("## Known advisory findings") == 1
    assert captured["pr_body"].count("## Secret scan attestation") == 1


def test_high_payload_finding_blocks_before_any_github_or_mutation(monkeypatch, tmp_path):
    def scanner(named):
        if "SKILL.md" in named:
            return _scan_result(_finding("SKILL.md", confidence="high", disposition="blocker"))
        return _scan_result()

    ctx, events, captured = _install_transaction_fakes(monkeypatch, tmp_path, snapshot=_snapshot(), scanner=scanner)
    monkeypatch.setattr(
        skill_publish,
        "github_login",
        lambda _ctx: pytest.fail("GitHub must not be called after a local blocker"),
    )
    result = _submit(ctx)
    assert result["ok"] is False
    assert result["reason_code"] == "secret_blocked"
    assert result["blocker_count"] == 1
    assert not any(row[0] == "mutation" for row in events)
    assert captured["llm_calls"] == 0
    assert set(result["findings"][0]) == {
        "path",
        "line",
        "detector",
        "confidence",
        "reason",
        "verification",
        "disposition",
    }


def test_medium_payload_warning_and_audited_high_do_not_block(monkeypatch, tmp_path):
    findings = (
        _finding("SKILL.md", confidence="medium", disposition="warning"),
        _finding(
            "SKILL.md",
            confidence="high",
            disposition="audited_false_positive",
        ),
    )

    def scanner(named):
        return _scan_result(*findings) if "SKILL.md" in named else _scan_result()

    ctx, _events, _captured = _install_transaction_fakes(monkeypatch, tmp_path, snapshot=_snapshot(), scanner=scanner)
    result = _submit(ctx)
    assert result["ok"] is True
    assert result["blocker_count"] == 0
    assert result["warning_count"] == 1
    assert result["audited_false_positive_count"] == 1


def test_medium_prompt_finding_keeps_optional_model_and_high_only_policy(monkeypatch, tmp_path):
    def scanner(named):
        if "pr-body-model-prompt.txt" in named:
            return _scan_result(_finding("pr-body-model-prompt.txt"))
        if "pull-request-body.md" in named:
            return _scan_result(_finding("pull-request-body.md"))
        return _scan_result()

    ctx, _events, captured = _install_transaction_fakes(
        monkeypatch,
        tmp_path,
        snapshot=_snapshot(description="Ambiguous but publishable text."),
        scanner=scanner,
    )
    result = _submit(ctx)
    assert result["ok"] is True
    assert captured["llm_calls"] == 1
    assert "Model summary" in captured["pr_body"]
    assert result["warning_count"] == 1


def test_high_prompt_finding_skips_optional_model_and_uses_fallback(monkeypatch, tmp_path):
    def scanner(named):
        if "pr-body-model-prompt.txt" in named:
            return _scan_result(
                _finding(
                    "pr-body-model-prompt.txt",
                    confidence="high",
                    disposition="blocker",
                )
            )
        return _scan_result()

    ctx, _events, captured = _install_transaction_fakes(
        monkeypatch,
        tmp_path,
        snapshot=_snapshot(),
        scanner=scanner,
    )
    result = _submit(ctx)
    assert result["ok"] is True
    assert captured["llm_calls"] == 0
    assert "A safe test skill" in captured["pr_body"]


def test_model_finding_discards_optional_prose_without_retry(monkeypatch, tmp_path):
    def scanner(named):
        if "optional-pr-body.md" in named:
            return _scan_result(
                _finding(
                    "optional-pr-body.md",
                    confidence="high",
                    disposition="blocker",
                )
            )
        return _scan_result()

    ctx, _events, captured = _install_transaction_fakes(
        monkeypatch,
        tmp_path,
        snapshot=_snapshot(),
        scanner=scanner,
        model_body="MODEL_CANDIDATE",
    )
    result = _submit(ctx)
    assert result["ok"] is True
    assert captured["llm_calls"] == 1
    assert "MODEL_CANDIDATE" not in captured["pr_body"]
    assert "A safe test skill" in captured["pr_body"]


def test_medium_model_finding_keeps_optional_prose(monkeypatch, tmp_path):
    def scanner(named):
        if "optional-pr-body.md" in named or "pull-request-body.md" in named:
            return _scan_result(_finding(next(iter(named))))
        return _scan_result()

    ctx, _events, captured = _install_transaction_fakes(
        monkeypatch,
        tmp_path,
        snapshot=_snapshot(),
        scanner=scanner,
        model_body="## Summary\nMODEL_WARNING\n\n## What This Skill Does\nSafe.",
    )
    result = _submit(ctx)
    assert result["ok"] is True
    assert captured["llm_calls"] == 1
    assert "MODEL_WARNING" in captured["pr_body"]
    assert result["warning_count"] == 1


@pytest.mark.parametrize(
    ("note", "model_body"),
    [
        ("```python\nnote example", "## Summary\nSafe summary."),
        ("", "## Summary\n~~~~ text\nmodel example"),
    ],
)
def test_unterminated_component_fence_cannot_capture_host_sections(
    monkeypatch,
    tmp_path,
    note,
    model_body,
):
    ctx, _events, captured = _install_transaction_fakes(
        monkeypatch,
        tmp_path,
        snapshot=_snapshot(),
        model_body=model_body,
    )
    result = _submit(ctx, note=note)
    assert result["ok"] is True
    for heading in (
        "## Author Checklist",
        "## Known advisory findings",
        "## Secret scan attestation",
    ):
        prefix = captured["pr_body"].split(heading, 1)[0]
        assert skill_publish._close_unterminated_fence(prefix) == prefix


def test_update_deletes_files_absent_from_exact_snapshot(monkeypatch, tmp_path):
    ctx, _events, captured = _install_transaction_fakes(
        monkeypatch,
        tmp_path,
        snapshot=_snapshot(),
    )
    old_catalog = {
        "skills": [
            {
                "slug": "demo",
                "version": "0.9.0",
                "files": [
                    {"path": "skill.json"},
                    {"path": "SKILL.md"},
                    {"path": "removed.py"},
                ],
            }
        ]
    }
    monkeypatch.setattr(
        skill_publish,
        "fetch_upstream_catalog",
        lambda *_args: (old_catalog, BASE_SHA),
    )
    result = _submit(ctx)
    assert result["ok"] is True
    assert captured["deletions"] == [{"path": "skills/demo/removed.py"}]


def test_duplicate_target_slug_is_typed_before_mutation(monkeypatch, tmp_path):
    ctx, events, _captured = _install_transaction_fakes(
        monkeypatch,
        tmp_path,
        snapshot=_snapshot(),
    )
    duplicate_catalog = {
        "skills": [
            {"slug": "demo", "version": "0.8.0", "files": []},
            {"slug": "demo", "version": "0.9.0", "files": []},
        ]
    }
    monkeypatch.setattr(
        skill_publish,
        "fetch_upstream_catalog",
        lambda *_args: (duplicate_catalog, BASE_SHA),
    )

    result = _submit(ctx)

    assert result["reason_code"] == "upstream_catalog_invalid"
    assert not any(row[0] == "mutation" for row in events)


def test_high_author_note_blocks_before_github(monkeypatch, tmp_path):
    def scanner(named):
        if "author-note.md" in named:
            return _scan_result(_finding("author-note.md", confidence="high", disposition="blocker"))
        return _scan_result()

    ctx, events, _captured = _install_transaction_fakes(monkeypatch, tmp_path, snapshot=_snapshot(), scanner=scanner)
    monkeypatch.setattr(
        skill_publish,
        "github_login",
        lambda _ctx: pytest.fail("GitHub must not be called after a note blocker"),
    )
    result = _submit(ctx, note="candidate assembled by the test")
    assert result["reason_code"] == "secret_blocked"
    assert result["completed_stage"] == "snapshot_captured"
    assert not any(row[0] == "mutation" for row in events)


def test_scanner_unavailable_is_typed_repair_evidence(monkeypatch, tmp_path):
    ctx, events, _captured = _install_transaction_fakes(
        monkeypatch,
        tmp_path,
        snapshot=_snapshot(),
        scanner=lambda _named: _scan_result(reason_code="scanner_missing"),
    )
    result = _submit(ctx)
    assert result["ok"] is False
    assert result["status"] == "repair_needed"
    assert result["reason_code"] == "scanner_missing"
    assert "betterleaks_runtime install" in result["repair_hint"]
    assert not any(row[0] == "mutation" for row in events)


def test_confirmation_failure_is_parseable_and_calls_nothing(tmp_path):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    result = json.loads(skill_publish._submit_skill_to_hub(ctx, "demo"))
    assert result["ok"] is False
    assert result["reason_code"] == "confirmation_required"
    assert result["completed_effects"] == []


def test_later_scanner_error_does_not_erase_known_scanner_identity():
    attempt = skill_publish._PublishAttempt(skill="demo")
    attempt.observe_scan(_scan_result(), include_findings=False)
    attempt.observe_scan(_scan_result(reason_code="scanner_timeout"), include_findings=False)
    assert attempt.scanner == {
        "engine": "betterleaks",
        "version": "1.8.1",
        "ruleset_sha256": RULESET_SHA,
    }


def test_publisher_has_no_legacy_regex_secret_gate():
    source = pathlib.Path(skill_publish.__file__).read_text(encoding="utf-8")
    assert "contains_real_secret_value" not in source
    assert "permission_statement" not in source
