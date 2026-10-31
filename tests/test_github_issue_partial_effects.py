"""Consolidated issue writes keep each effect and identity in its own receipt.

Creation includes labels in one request. A closing comment and a close are two explicit methods;
a failed or unanswered first call never implicitly sends the second or repeats the first.
"""
import json
import subprocess

import pytest

from tests.test_github_read import _comment, _issue
from tests.test_github_write import _receipt, gh as _gh

gh = _gh


@pytest.mark.parametrize("record", [
    _issue(7, title="Title", body="Body", labels=[]),
    {"html_url": "https://github.com/owner/selected/issues/7"},
])
def test_create_label_or_identity_mismatch_keeps_the_answer_and_never_sends_a_second_write(gh, record):
    gh.answer = lambda *_: json.dumps(record)
    line, receipt = _receipt(gh.write(method="create_issue", title="Title", body="Body", labels=["bug"]))
    assert "GH_WRITE_READBACK_MISMATCH" in line and receipt["readback"]["verified"] is False
    assert "labels" in receipt["readback"]["mismatch"] and receipt["url"] == record["html_url"]
    assert len(gh.sent()) == 1 and gh.sent()[0] == (
        "POST", "repos/{owner}/{repo}/issues", {"title": "Title", "body": "Body", "labels": ["bug"]})
    assert receipt["id"] == record.get("id")


def test_create_timeout_names_an_unknown_outcome_and_never_recreates_or_adds_labels(gh):
    gh.answer = lambda p, *_: subprocess.TimeoutExpired(["gh"], 60) if p.method == "POST" else "[]"
    line, receipt = _receipt(gh.write(method="create_issue", title="Title", labels=["bug"]))
    assert "do not resend" in line and receipt["outcome"] == "unknown"
    assert [method for method, _, _ in gh.sent()].count("POST") == 1
    assert not any(method == "PATCH" for method, _, _ in gh.sent())


@pytest.mark.parametrize("labels", [[], ["bug"]])
def test_create_success_includes_labels_in_one_receipt_and_request(gh, labels):
    gh.answer = lambda *_: json.dumps(_issue(7, title="Title", body="Body", labels=[{"name": label} for label in labels]))
    _, receipt = _receipt(gh.write(method="create_issue", title="Title", body="Body", labels=labels))
    assert receipt["outcome"] == "confirmed" and receipt["readback"]["verified"]
    assert receipt["url"].endswith("/issues/7") and len(gh.sent()) == 1


@pytest.mark.parametrize("timeout", [False, True])
def test_failed_comment_does_not_implicitly_close_the_issue(gh, timeout):
    gh.answer = lambda p, *_: (subprocess.TimeoutExpired(["gh"], 60) if timeout else (1, "gh: Validation Failed (HTTP 422)")) if p.method == "POST" else "[]"
    _, receipt = _receipt(gh.write(method="comment", kind="issue", target=7, body="Fixed"))
    assert receipt["outcome"] == ("unknown" if timeout else "not_applied")
    assert [method for method, _, _ in gh.sent()].count("POST") == 1
    assert not any(method == "PATCH" for method, _, _ in gh.sent())


@pytest.mark.parametrize("refused", [False, True])
def test_explicit_close_has_its_own_receipt_and_never_repeats_the_confirmed_comment(gh, refused):
    gh.answer = lambda *_: json.dumps(_comment(3, body="Fixed"))
    _, comment = _receipt(gh.write(method="comment", target=7, body="Fixed"))
    assert comment["outcome"] == "confirmed" and comment["id"] == 3
    gh.answer = lambda *_: (1, "gh: Validation Failed (HTTP 422)") if refused else json.dumps(_issue(7, state="closed"))
    _, closed = _receipt(gh.write(method="update", target=7, state="closed"))
    assert closed["outcome"] == ("not_applied" if refused else "confirmed")
    assert [method for method, _, _ in gh.sent()] == ["POST", "PATCH"]
    assert gh.sent()[-1][2] == {"state": "closed"}


def test_close_target_refusal_before_any_launch_passes_through_typed(gh):
    gh.ctx.task_metadata = {"_project_room_note": "registry unavailable"}
    gh.ctx.workspace_root = None
    gh.ctx.is_direct_chat = True
    text = gh.write(method="update", target=7, state="closed", repo="")
    assert "GH_TARGET_UNAVAILABLE" in text and gh.calls == []


def test_github_workflow_inventory_and_authority_memberships():
    from ouroboros.tool_capabilities import OBSERVE_WORLD_MUTATION_TOOLS
    from ouroboros.tools.registry_guards import _GITHUB_TOKEN_TOOLS
    from tests.test_smoke import EXPECTED_TOOLS

    assert {"github_read", "github_write"} <= _GITHUB_TOKEN_TOOLS
    assert "github_write" in OBSERVE_WORLD_MUTATION_TOOLS and "github_read" not in OBSERVE_WORLD_MUTATION_TOOLS
    assert {"github_read", "github_write"} <= set(EXPECTED_TOOLS)
