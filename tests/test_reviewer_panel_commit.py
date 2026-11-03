"""Advisory commit composition reaches the real gate before staging or review spend."""
from pathlib import Path

import pytest

from ouroboros import review_substrate as substrate
from ouroboros.tools import git as git_mod
from ouroboros.tools.registry import ToolContext
from tests import test_review_change_end_to_end as e2e

staged_body = e2e.staged_body


def test_advisory_commit_seats_exactly_the_named_reviewers(staged_body, tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    record, sent = e2e._gate_with_critic(staged_body, tmp_path, monkeypatch, mode="pro",
                                     reviewers=["s1", "scout"], reason=e2e.REASON, expected_status="reviewed")
    assert sent == ["s1", "scout"]
    assert record["panel"]["assigned"] == ["s1"] and record["panel"]["additional"] == ["scout"]
    assert record["panel"]["reason"] == e2e.REASON and record["panel"]["widened_from"] == []
    assert record["panel"]["strictness"] == "advisory"


def test_reasonless_advisory_commit_refuses_before_staging(staged_body, tmp_path, monkeypatch):
    repo = Path(staged_body["repo"])
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    monkeypatch.setattr(substrate, "run_review_request", lambda *a, **k: pytest.fail("no wave may be paid"))
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "gate-drive")
    e2e.shared.git(repo, "reset", "-q", "HEAD")
    result = git_mod._commit_reviewed(ctx, e2e.COMMIT_MESSAGE, reviewers=["s1"])
    assert "TOOL_ARG_ERROR" in result and "REVIEW_PANEL_REASON_REQUIRED" in result
    assert e2e.shared.git(repo, "diff", "--cached", "--name-only") == ""
