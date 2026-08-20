"""Hermetic tests for the one-shot GitHub publication transport."""

from __future__ import annotations

import base64
import json
import types

import pytest

from ouroboros import skill_publish_github as github

BASE_SHA = "1" * 40
COMMIT_SHA = "2" * 40
SNAPSHOT_SHA = "a" * 64
RULESET_SHA = "b" * 64


def _attempt():
    facts = types.SimpleNamespace(
        skill="demo",
        snapshot_hash=SNAPSHOT_SHA,
        scanner={"ruleset_sha256": RULESET_SHA},
        marks=[],
    )

    def mark(stage, **values):
        facts.marks.append((stage, values))

    facts.mark = mark
    return facts


def _pull_row(*, sha: str = COMMIT_SHA, owner: str = "alice", base: str = "main"):
    return {
        "number": 7,
        "html_url": "https://github.com/hub/project/pull/7",
        "head": {
            "sha": sha,
            "ref": "submit/demo-v1.0.0",
            "repo": {"owner": {"login": owner}},
        },
        "base": {"ref": base},
    }


def test_commit_payload_hardcodes_ouroboros_coauthor(monkeypatch):
    captured = {}

    def fake_json(_ctx, _args, **kwargs):
        captured.update(json.loads(kwargs["input_data"]))
        return {
            "data": {
                "createCommitOnBranch": {
                    "commit": {
                        "oid": COMMIT_SHA,
                        "url": "https://github.com/alice/project/commit/" + COMMIT_SHA,
                    }
                }
            }
        }

    monkeypatch.setattr(github, "_json_object", fake_json)
    sha, url = github.commit_payload(
        types.SimpleNamespace(),
        "alice",
        "project",
        "submit/demo-v1.0.0",
        BASE_SHA,
        "Add skill: demo v1.0.0",
        [{"path": "skills/demo/SKILL.md", "contents": "YQ=="}],
        [{"path": "skills/demo/removed.py"}],
    )
    message = captured["variables"]["input"]["message"]
    assert sha == COMMIT_SHA
    assert url.endswith(COMMIT_SHA)
    assert message == {
        "headline": "Add skill: demo v1.0.0",
        "body": ("Co-authored-by: Ouroboros <311266734+ouroboros-agent@users.noreply.github.com>"),
    }
    assert captured["variables"]["input"]["fileChanges"] == {
        "additions": [{"path": "skills/demo/SKILL.md", "contents": "YQ=="}],
        "deletions": [{"path": "skills/demo/removed.py"}],
    }


def test_upstream_catalog_is_read_from_the_exact_resolved_base_sha(monkeypatch):
    calls = []

    def fake_json(_ctx, args, **_kwargs):
        calls.append(args)
        if "/git/refs/heads/" in args[-1]:
            return {"object": {"sha": BASE_SHA}}
        return {
            "content": base64.b64encode(b'{"skills":[]}').decode("ascii"),
        }

    monkeypatch.setattr(github, "_json_object", fake_json)
    catalog, base_sha = github.fetch_upstream_catalog(
        types.SimpleNamespace(),
        "hub",
        "project",
        "main",
    )
    assert catalog == {"skills": []}
    assert base_sha == BASE_SHA
    assert calls[-1][-1] == f"/repos/hub/project/contents/catalog.json?ref={BASE_SHA}"


def test_owner_actor_skips_fork_and_sync(monkeypatch):
    monkeypatch.setattr(
        github,
        "_gh_cmd",
        lambda *_args, **_kwargs: pytest.fail("owner path must issue no fork command"),
    )
    attempt = _attempt()
    github.prepare_publish_repository(
        types.SimpleNamespace(),
        attempt,
        owner="HubOwner",
        repo="project",
        base_branch="main",
        login="hubowner",
    )
    assert attempt.marks == [("fork_ready", {"repository": "hubowner/project", "actor": "hubowner"})]


def test_non_owner_sync_failure_is_typed(monkeypatch):
    calls = []

    def fake_gh(args, _ctx, **_kwargs):
        calls.append(args)
        if args[:2] == ["repo", "view"]:
            return '{"name":"project"}'
        return "⚠️ GH_ERROR: synthetic"

    monkeypatch.setattr(github, "_gh_cmd", fake_gh)
    with pytest.raises(github.SkillPublishGitHubError) as caught:
        github.prepare_publish_repository(
            types.SimpleNamespace(),
            _attempt(),
            owner="hub",
            repo="project",
            base_branch="main",
            login="alice",
        )
    assert caught.value.reason_code == "fork_sync_failed"
    assert [call[0] for call in calls] == ["repo", "api"]


def test_direct_pr_url_yields_validated_receipt_without_lookup(monkeypatch):
    calls = []

    def fake_gh(args, _ctx, **_kwargs):
        calls.append(args)
        return "https://github.com/hub/project/pull/7"

    monkeypatch.setattr(github, "_gh_cmd", fake_gh)
    monkeypatch.setattr(
        github,
        "_json_value",
        lambda *_args, **_kwargs: pytest.fail("valid direct output must not settle"),
    )
    attempt = _attempt()
    receipt = github.create_pr_receipt(
        types.SimpleNamespace(),
        attempt,
        owner="hub",
        repo="project",
        base_branch="main",
        login="alice",
        branch="submit/demo-v1.0.0",
        title="Add demo",
        body="body",
        commit_sha=COMMIT_SHA,
    )
    assert receipt and receipt["number"] == 7
    assert len(calls) == 1
    assert [stage for stage, _facts in attempt.marks] == ["pr_create_attempted"]


def test_ambiguous_create_uses_one_exact_read_only_settlement(monkeypatch):
    create_calls = []
    lookup_calls = []

    def fake_gh(args, _ctx, **_kwargs):
        create_calls.append(args)
        return "⚠️ GH_TIMEOUT: synthetic"

    def fake_json(_ctx, args, **_kwargs):
        lookup_calls.append(args)
        return [_pull_row()]

    monkeypatch.setattr(github, "_gh_cmd", fake_gh)
    monkeypatch.setattr(github, "_json_value", fake_json)
    receipt = github.create_pr_receipt(
        types.SimpleNamespace(),
        _attempt(),
        owner="hub",
        repo="project",
        base_branch="main",
        login="alice",
        branch="submit/demo-v1.0.0",
        title="Add demo",
        body="body",
        commit_sha=COMMIT_SHA,
    )
    assert receipt and receipt["url"].endswith("/pull/7")
    assert len(create_calls) == 1
    assert len(lookup_calls) == 1
    assert "--method" in lookup_calls[0]
    assert "GET" in lookup_calls[0]


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [_pull_row(), _pull_row()],
        [_pull_row(sha="3" * 40)],
        [_pull_row(owner="other")],
        [_pull_row(base="wrong")],
        [_pull_row() | {"number": 8}],
    ],
)
def test_ambiguous_settlement_never_claims_wrong_or_nonunique_pr(monkeypatch, rows):
    monkeypatch.setattr(github, "_gh_cmd", lambda *_args, **_kwargs: "garbage")
    monkeypatch.setattr(github, "_json_value", lambda *_args, **_kwargs: rows)
    receipt = github.create_pr_receipt(
        types.SimpleNamespace(),
        _attempt(),
        owner="hub",
        repo="project",
        base_branch="main",
        login="alice",
        branch="submit/demo-v1.0.0",
        title="Add demo",
        body="body",
        commit_sha=COMMIT_SHA,
    )
    assert receipt is None


def test_existing_branch_is_never_overwritten(monkeypatch):
    monkeypatch.setattr(github, "_gh_cmd", lambda *_args, **_kwargs: '{"ref":"exists"}')
    with pytest.raises(github.SkillPublishGitHubError) as caught:
        github.ensure_branch(
            types.SimpleNamespace(),
            "alice",
            "project",
            "submit/demo-v1.0.0",
            BASE_SHA,
        )
    assert caught.value.reason_code == "submission_branch_exists"
