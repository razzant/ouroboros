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
    from ouroboros.skill_publish_destination import PublishDestination

    monkeypatch.setattr(skill_publish, "resolve_publish_destination", lambda *_a, **_k: PublishDestination(
        "hub", "hub", "project", "main", "skills", "ouroboros", "Ouroboros"))
    monkeypatch.setattr(skill_publish, "fetch_upstream_file", lambda *_a, **_k: None)  # an older hub: no hub.json
    monkeypatch.setattr(skill_publish, "fetch_branch_head", lambda _ctx, _owner, _repo, branch: (BASE_SHA, branch or "main"))
    monkeypatch.setattr(skill_publish, "fetch_repository", lambda *_a: {"default_branch": "main", "can_push": False})
    monkeypatch.setattr(skill_publish, "fetch_tree_paths", lambda *_a: [])

    def no_merge(*_args):
        raise AssertionError("only a git destination this account may push to merges its request")

    monkeypatch.setattr(skill_publish, "merge_pull_request", no_merge)
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

    def branch(*_args, **_kwargs):
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
    assert captured["pr_body"].count("## Recorded reviewer findings") == 1
    assert captured["pr_body"].count("## Secret scan attestation") == 1


def test_success_writes_state_plane_publication_receipt(monkeypatch, tmp_path):
    from ouroboros.marketplace.provenance import read_publication_record

    ctx, _events, _captured = _install_transaction_fakes(monkeypatch, tmp_path, snapshot=_snapshot())
    monkeypatch.setattr(skill_publish, "utc_now_iso", lambda: "2026-08-23T00:00:00+00:00")
    result = _submit(ctx)
    assert result["ok"] is True
    assert result["publication_recorded"] is True
    assert "publication_record_error" not in result
    record_path = tmp_path / "state" / "skills" / "demo" / "ouroboroshub.json"
    assert json.loads(record_path.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "published": {
            "slug": "demo",
            "version": "1.0.0",
            "content_hash": SNAPSHOT_SHA,
            "repository": "hub/project",
            "pr_number": 7,
            "pr_url": "https://github.com/hub/project/pull/7",
            "published_at": "2026-08-23T00:00:00+00:00",
        },
    }
    published, diagnostic = read_publication_record(tmp_path, "demo")
    assert diagnostic is None
    assert published["pr_number"] == 7


def test_receipt_write_failure_keeps_pr_success_and_is_typed(monkeypatch, tmp_path):
    ctx, _events, _captured = _install_transaction_fakes(monkeypatch, tmp_path, snapshot=_snapshot())

    def _explode(*_args, **_kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(skill_publish, "write_publication_record", _explode)
    result = _submit(ctx)
    # The PR success is never converted into a failure by the local write.
    assert result["ok"] is True
    assert result["status"] == "pr_opened"
    assert result["receipt"]["number"] == 7
    assert result["publication_recorded"] is False
    assert result["publication_record_error"] == "RuntimeError: disk full"
    assert not (tmp_path / "state" / "skills" / "demo" / "ouroboroshub.json").exists()


def test_failed_publication_writes_no_receipt_and_no_flag(monkeypatch, tmp_path):
    def scanner(named):
        if "SKILL.md" in named:
            return _scan_result(_finding("SKILL.md", confidence="high", disposition="blocker"))
        return _scan_result()

    ctx, _events, _captured = _install_transaction_fakes(monkeypatch, tmp_path, snapshot=_snapshot(), scanner=scanner)
    result = _submit(ctx)
    assert result["ok"] is False
    assert "publication_recorded" not in result
    assert not (tmp_path / "state" / "skills" / "demo" / "ouroboroshub.json").exists()


@pytest.mark.parametrize("failure", ["fork_sync", "branch_create"])
def test_publish_failure_projection_uses_confirmed_transport_progress(monkeypatch, tmp_path, failure):
    from ouroboros import skill_publish_github as github
    from ouroboros.skill_publish_result import apply_skill_publish_receipt_veto, extract_skill_publish_result_metadata
    from ouroboros.skill_publish_client import GitHubError

    ctx, _events, _captured = _install_transaction_fakes(monkeypatch, tmp_path, snapshot=_snapshot())
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_SYNTHETIC1234567890")
    monkeypatch.setattr(skill_publish, "prepare_publish_repository", github.prepare_publish_repository)
    monkeypatch.setattr(skill_publish, "ensure_branch", github.ensure_branch)
    requests = []

    def transport(method, path, **_kwargs):
        requests.append((method, path))
        if path == "/repos/alice/project":
            return {"name": "project"}
        if path.endswith("/merge-upstream"):
            raise GitHubError("sync rejected", http_status=403)
        if "/git/ref/heads/" in path:
            raise GitHubError("Not Found", http_status=404)
        assert method == "POST" and path == "/repos/alice/project/git/refs"
        if failure == "fork_sync":
            raise GitHubError("Object missing", http_status=422)
        raise GitHubError("exceeded 30s", failure="timeout")

    monkeypatch.setattr(github, "request", transport)
    result = _submit(ctx)
    assert result["ok"] is False
    assert result["completed_stage"] == "fork_ready"  # the fork is never synchronized up front
    assert result["reason_code"] == f"{failure}_failed"
    outcome = {"outcome_axes": {"objective": {"status": "degraded"}, "review": {"status": "degraded"}}}
    apply_skill_publish_receipt_veto(outcome, {
        "type": "skill_publish", "metadata": {"skill_publish_target": {"skill": "demo", "repository": "hub/project"}},
    }, {"tool_calls": [{"tool": "submit_skill_to_hub", **extract_skill_publish_result_metadata(json.dumps(result))}]})
    assert outcome["outcome_axes"]["objective"]["status"] == ("fail" if failure == "fork_sync" else "degraded")
    assert len(requests) == 4 if failure == "fork_sync" else len(requests) == 3


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
        "## Recorded reviewer findings",
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
    assert not {"error_detail", "github_status", "github_operation"} & result.keys()


@pytest.mark.parametrize("long_message", [False, True])
def test_github_failure_envelope_is_reached_from_the_https_boundary(monkeypatch, tmp_path, long_message):
    """Real repository preparation and branch creation + real client; only HTTPS is fake.

    The fork never saw the destination commit, GitHub refuses the branch (422), and the one
    synchronization that follows fails."""
    from ouroboros import skill_publish_github
    from ouroboros.skill_publish_result import _bounded_text, extract_skill_publish_result_metadata

    ctx, events, _captured = _install_transaction_fakes(monkeypatch, tmp_path, snapshot=_snapshot())
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_SYNTHETIC1234567890")
    monkeypatch.setattr(
        skill_publish, "prepare_publish_repository", skill_publish_github.prepare_publish_repository,
    )
    monkeypatch.setattr(skill_publish, "ensure_branch", skill_publish_github.ensure_branch)
    commands = []

    import httpx
    from ouroboros import skill_publish_client

    def request(method, url, **_kwargs):
        commands.append((method, url))
        if url.endswith("/repos/alice/project"):
            return httpx.Response(200, json={"name": "project"})
        if "/git/ref/heads/" in url:
            return httpx.Response(404, json={"message": "Not Found"})
        if url.endswith("/git/refs"):
            return httpx.Response(422, json={"message": "Object does not exist"})
        assert method == "POST" and url.endswith("/merge-upstream")
        message = "token ghp_SYNTHETIC1234567890 was rejected"
        if long_message:
            message = "x" * 700 + "\n" + message
        return httpx.Response(403, json={"message": message})

    monkeypatch.setattr(skill_publish_client.httpx, "request", request)
    result = _submit(ctx)
    assert result["ok"] is False
    assert result["reason_code"] == "fork_sync_failed"
    assert result["completed_stage"] == "fork_ready"
    assert result["github_status"] == 403
    assert result["github_operation"] == "merge-upstream"
    assert result["error_detail"].startswith("GitHub HTTP 403: ")
    # Status comes from the response, independently of the bounded message.
    assert "ghp_SYNTHETIC1234567890" not in json.dumps(result)
    assert "receipt" not in result
    assert not any(row[0] == "mutation" for row in events)
    assert sum(url.endswith("/merge-upstream") for method, url in commands) == 1
    projected = extract_skill_publish_result_metadata(json.dumps(result))["skill_publish_attempt"]
    assert projected["github_status"] == 403
    assert projected["github_operation"] == "merge-upstream"
    # The projection is a single bounded line; the envelope keeps the transport's
    # own multi-line omission note when the head was cut. Pinned, not accidental.
    assert projected["error_detail"] == _bounded_text(result["error_detail"], 640)
    if long_message:
        assert "OMISSION NOTE" in result["error_detail"] and "\n" in result["error_detail"]
        assert "\n" not in projected["error_detail"] and len(projected["error_detail"]) <= 640
    else:
        assert projected["error_detail"] == result["error_detail"]


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


@pytest.mark.parametrize("http_status", [403, None])
def test_github_failure_envelope_keeps_cause_and_last_completed_stage(monkeypatch, tmp_path, http_status):
    ctx, events, _captured = _install_transaction_fakes(monkeypatch, tmp_path, snapshot=_snapshot())
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_SYNTHETIC1234567890")
    detail = "⚠️ GH_ERROR: gh: Resource not accessible by personal access token (HTTP 403)"

    def prepare(_ctx, attempt, **kwargs):
        attempt.mark("fork_ready", repository="alice/project", actor="alice")
        raise skill_publish.SkillPublishGitHubError(
            "fork_sync_failed", "Update GITHUB_TOKEN in Settings → Secrets, then retry.",
            detail=detail, http_status=http_status, operation="merge-upstream",
        )

    monkeypatch.setattr(skill_publish, "prepare_publish_repository", prepare)
    result = _submit(ctx)
    assert result["ok"] is False
    assert result["reason_code"] == "fork_sync_failed"
    assert result["completed_stage"] == "fork_ready"
    assert result["completed_effects"][-1]["stage"] == "fork_ready"
    assert result["error_detail"] == detail
    if http_status is None:
        assert "github_status" not in result
    else:
        assert result["github_status"] == http_status
    assert result["github_operation"] == "merge-upstream"
    from ouroboros.skill_publish_result import extract_skill_publish_result_metadata

    projected = extract_skill_publish_result_metadata(json.dumps(result))["skill_publish_attempt"]
    assert projected["error_detail"] == detail
    assert projected["github_operation"] == "merge-upstream"
    assert projected.get("github_status") == http_status
    assert "receipt" not in result
    assert not any(row[0] == "mutation" for row in events)
    assert not (tmp_path / "state" / "skills" / "demo" / "ouroboroshub.json").exists()
