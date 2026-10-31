"""Hermetic publication operations, exact receipts and typed transport failures."""

from __future__ import annotations

import base64
import types
from urllib.parse import parse_qs, urlsplit

import pytest

from ouroboros import skill_publish_github as github
from ouroboros.skill_publish_client import GitHubError

BASE_SHA = "1" * 40
COMMIT_SHA = "2" * 40
SNAPSHOT_SHA = "a" * 64
RULESET_SHA = "b" * 64
BRANCH = "submit/demo-v1.0.0"


@pytest.fixture(autouse=True)
def synthetic_github(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_SYNTHETIC1234567890")
    monkeypatch.setattr(github, "request", lambda *_a, **_k: pytest.fail("unexpected GitHub request"), raising=False)
    monkeypatch.setattr(github, "_gh_run", lambda *_a, **_k: pytest.fail("publication used gh"), raising=False)
    monkeypatch.setattr("ouroboros.tools.github._gh_run", lambda *_a, **_k: pytest.fail("publication used gh"))


@pytest.fixture
def responses(monkeypatch):
    """A single ordered client fake, including method/path assertions for every operation."""
    calls = []

    def install(*steps):
        remaining = list(steps)

        def request(method, path, **kwargs):
            calls.append((method, path, kwargs))
            assert remaining, (method, path)
            expected_method, expected_path, answer = remaining.pop(0)
            assert (method, path) == (expected_method, expected_path)
            if isinstance(answer, Exception):
                raise answer
            return answer

        monkeypatch.setattr(github, "request", request)
        return calls, remaining

    return install


def _attempt():
    facts = types.SimpleNamespace(skill="demo", snapshot_hash=SNAPSHOT_SHA,
                                  scanner={"ruleset_sha256": RULESET_SHA}, marks=[])
    facts.mark = lambda stage, **values: facts.marks.append((stage, values))
    return facts


def _pull_row(*, sha=COMMIT_SHA, owner="alice", base="main"):
    return {"number": 7, "html_url": "https://github.com/hub/project/pull/7",
            "head": {"sha": sha, "ref": BRANCH, "repo": {"owner": {"login": owner}}},
            "base": {"ref": base}}


def _create(attempt=None):
    return github.create_pr_receipt(types.SimpleNamespace(), attempt or _attempt(), owner="hub", repo="project",
                                    base_branch="main", login="alice", branch=BRANCH, title="Add demo",
                                    body="body\nwith newlines", commit_sha=COMMIT_SHA)


def test_account_repository_and_exact_files_use_https(responses):
    calls, pending = responses(
        ("GET", "/user", {"login": "alice"}),
        ("GET", "/repos/hub/project", {"default_branch": "trunk", "permissions": {"push": True}}),
        ("GET", "/repos/hub/project/git/refs/heads/trunk", {"object": {"sha": BASE_SHA}}),
        ("GET", f"/repos/hub/project/contents/a%20file.md?ref={BASE_SHA}", {"content": "aGVsbG8="}),
        ("GET", f"/repos/hub/project/contents/missing?ref={BASE_SHA}", GitHubError("Not Found", http_status=404)),
        ("GET", f"/repos/hub/project/git/trees/{BASE_SHA}?recursive=1",
         {"tree": [{"path": "skills/demo/a.md", "type": "blob"}, {"path": "skills/other/a.md", "type": "blob"}]}),
    )
    ctx = types.SimpleNamespace()
    assert github.github_login(ctx) == "alice"
    assert github.fetch_repository(ctx, "hub", "project") == {"default_branch": "trunk", "can_push": True}
    assert github.fetch_branch_head(ctx, "hub", "project", "trunk") == (BASE_SHA, "trunk")
    assert github.fetch_upstream_file(ctx, "hub", "project", BASE_SHA, "a file.md") == b"hello"
    assert github.fetch_upstream_file(ctx, "hub", "project", BASE_SHA, "missing") is None
    assert github.fetch_tree_paths(ctx, "hub", "project", BASE_SHA, "skills/demo") == ["a.md"]
    assert not pending and all(not row[2].get("body") for row in calls)


def test_upstream_catalog_is_read_from_the_exact_resolved_base_sha(responses):
    _, pending = responses(
        ("GET", "/repos/hub/project", {"default_branch": "trunk"}),
        ("GET", "/repos/hub/project/git/refs/heads/trunk", {"object": {"sha": BASE_SHA}}),
        ("GET", f"/repos/hub/project/contents/catalog.json?ref={BASE_SHA}",
         {"content": base64.b64encode(b'{"skills":[]}').decode("ascii")}),
    )
    assert github.fetch_upstream_catalog(types.SimpleNamespace(), "hub", "project", "") == ({"skills": []}, BASE_SHA)
    assert not pending


def test_commit_payload_preserves_graphql_cas_and_coauthor(responses):
    calls, _ = responses(("POST", "/graphql", {"data": {"createCommitOnBranch": {
        "commit": {"oid": COMMIT_SHA, "url": "https://github.com/alice/project/commit/" + COMMIT_SHA}}}}))
    additions = [{"path": "skills/demo/SKILL.md", "contents": "YQ=="}]
    deletions = [{"path": "skills/demo/removed.py"}]
    sha, url = github.commit_payload(types.SimpleNamespace(), "alice", "project", BRANCH, BASE_SHA,
                                     "Add demo", additions, deletions)
    assert sha == COMMIT_SHA and url.endswith(COMMIT_SHA)
    payload = calls[0][2]["body"]
    assert "createCommitOnBranch" in payload["query"] and calls[0][2]["timeout"] == 60
    assert payload["variables"]["input"] == {
        "branch": {"repositoryNameWithOwner": "alice/project", "branchName": BRANCH},
        "message": {"headline": "Add demo",
                    "body": "Co-authored-by: Ouroboros <311266734+ouroboros-agent@users.noreply.github.com>"},
        "fileChanges": {"additions": additions, "deletions": deletions}, "expectedHeadOid": BASE_SHA,
    }


def test_owner_actor_skips_fork_and_sync():
    attempt = _attempt()
    github.prepare_publish_repository(types.SimpleNamespace(), attempt, owner="HubOwner", repo="project",
                                       base_branch="main", login="hubowner")
    assert attempt.marks == [("fork_ready", {"repository": "hubowner/project", "actor": "hubowner"})]


@pytest.mark.parametrize("exists", [False, True])
def test_non_owner_creates_or_connects_fork_without_sync(responses, exists):
    steps = [("GET", "/repos/alice/project", {"name": "project"} if exists else GitHubError("missing", http_status=404))]
    if not exists:
        steps.append(("POST", "/repos/hub/project/forks", {"name": "project"}))
    calls, pending = responses(*steps)
    attempt = _attempt()
    github.prepare_publish_repository(types.SimpleNamespace(), attempt, owner="hub", repo="project",
                                       base_branch="main", login="alice")
    assert not pending and [stage for stage, _ in attempt.marks] == ["fork_ready"]
    if not exists:
        assert calls[1][2]["body"] == {} and calls[1][2]["timeout"] == 60


def test_fork_denial_keeps_github_status_and_message(responses):
    responses(("GET", "/repos/alice/project", GitHubError("Not Found", http_status=404)),
              ("POST", "/repos/hub/project/forks", GitHubError("Resource not accessible", http_status=403)))
    attempt = _attempt()
    with pytest.raises(github.SkillPublishGitHubError) as caught:
        github.prepare_publish_repository(types.SimpleNamespace(), attempt, owner="hub", repo="project",
                                           base_branch="main", login="alice")
    assert caught.value.reason_code == "fork_prepare_failed" and caught.value.http_status == 403
    assert "Resource not accessible" in caught.value.detail and "Settings → Secrets" in caught.value.repair_hint
    assert not attempt.marks


def test_existing_branch_is_never_overwritten(responses):
    calls, _ = responses(("GET", f"/repos/alice/project/git/ref/heads/{BRANCH}", {"ref": "exists"}))
    with pytest.raises(github.SkillPublishGitHubError, match="submission_branch_exists"):
        github.ensure_branch(types.SimpleNamespace(), "alice", "project", BRANCH, BASE_SHA)
    assert len(calls) == 1


@pytest.mark.parametrize("refused", [False, True])
def test_a_fork_is_synchronized_only_when_github_refuses_the_commit(responses, refused):
    steps = [("GET", f"/repos/alice/project/git/ref/heads/{BRANCH}", GitHubError("missing", http_status=404))]
    if refused:
        steps += [("POST", "/repos/alice/project/git/refs", GitHubError("Object missing", http_status=422)),
                  ("POST", "/repos/alice/project/merge-upstream", {})]
    steps.append(("POST", "/repos/alice/project/git/refs", {"object": {"sha": BASE_SHA}}))
    calls, pending = responses(*steps)
    attempt = _attempt()
    assert github.ensure_branch(types.SimpleNamespace(), "alice", "project", BRANCH, BASE_SHA,
                                 base_branch="main", attempt=attempt) == BASE_SHA
    assert not pending and len(attempt.marks) == int(refused)
    assert calls[-1][2]["body"] == {"ref": f"refs/heads/{BRANCH}", "sha": BASE_SHA}
    if refused:
        assert calls[2][2]["body"] == {"branch": "main"} and calls[2][2]["timeout"] == 45


@pytest.mark.parametrize("status", [403, 409, None])
def test_other_branch_failures_never_synchronize(responses, status):
    calls, _ = responses(
        ("GET", f"/repos/alice/project/git/ref/heads/{BRANCH}", GitHubError("missing", http_status=404)),
        ("POST", "/repos/alice/project/git/refs", GitHubError("refused", http_status=status)),
    )
    with pytest.raises(github.SkillPublishGitHubError, match="branch_create_failed"):
        github.ensure_branch(types.SimpleNamespace(), "alice", "project", BRANCH, BASE_SHA, base_branch="main")
    assert len(calls) == 2


@pytest.mark.parametrize("status,failure,hint", [(403, "http", "Settings → Secrets"), (409, "http", "conflict"),
                                               (None, "timeout", "outcome may be unknown")])
def test_sync_failure_preserves_evidence_and_stops(responses, status, failure, hint):
    calls, _ = responses(
        ("GET", f"/repos/alice/project/git/ref/heads/{BRANCH}", GitHubError("missing", http_status=404)),
        ("POST", "/repos/alice/project/git/refs", GitHubError("Object missing", http_status=422)),
        ("POST", "/repos/alice/project/merge-upstream",
         GitHubError("ghp_SYNTHETIC1234567890 refused", http_status=status, failure=failure)),
    )
    attempt = _attempt()
    with pytest.raises(github.SkillPublishGitHubError) as caught:
        github.ensure_branch(types.SimpleNamespace(), "alice", "project", BRANCH, BASE_SHA,
                             base_branch="main", attempt=attempt)
    error = caught.value
    assert error.reason_code == "fork_sync_failed" and error.http_status == status
    assert error.operation == "merge-upstream" and hint in error.repair_hint
    assert "ghp_SYNTHETIC1234567890" not in error.detail and len(calls) == 3 and not attempt.marks


def test_direct_pr_url_yields_validated_receipt_without_lookup(responses):
    calls, pending = responses(("POST", "/repos/hub/project/pulls", {"html_url": _pull_row()["html_url"]}))
    attempt = _attempt()
    assert _create(attempt)["number"] == 7
    assert not pending and len(calls) == 1
    assert calls[0][2] == {"timeout": 60, "body": {"base": "main", "head": "alice:" + BRANCH,
                         "title": "Add demo", "body": "body\nwith newlines"}}
    assert [stage for stage, _ in attempt.marks] == ["pr_create_attempted"]


@pytest.mark.parametrize("rows,valid", [([_pull_row()], True), ([], False), ([_pull_row(), _pull_row()], False),
    ([_pull_row(sha="3" * 40)], False), ([_pull_row(owner="other")], False), ([_pull_row(base="wrong")], False),
    ([_pull_row() | {"number": 8}], False), ([_pull_row() | {"html_url": "https://github.com/other/repo/pull/7"}], False)])
def test_ambiguous_settlement_requires_one_exact_pr(monkeypatch, rows, valid):
    calls = []

    def request(method, path, **kwargs):
        calls.append((method, path))
        if method == "POST":
            return {"html_url": "malformed URL"}
        assert method == "GET"
        assert parse_qs(urlsplit(path).query) == {"state": ["open"], "head": ["alice:" + BRANCH],
                                                 "base": ["main"], "per_page": ["100"]}
        return rows

    monkeypatch.setattr(github, "request", request)
    receipt = _create()
    assert (receipt is not None) == valid and len(calls) == 2


@pytest.mark.parametrize("settles", [False, True])
def test_timeout_creates_once_and_settles_read_only(monkeypatch, settles):
    calls = []

    def request(method, path, **kwargs):
        calls.append(method)
        if method == "POST":
            raise GitHubError("timed out", failure="timeout")
        if settles:
            return [_pull_row()]
        raise GitHubError("read refused", http_status=403)

    monkeypatch.setattr(github, "request", request)
    if settles:
        assert _create()["number"] == 7
    else:
        with pytest.raises(github.SkillPublishGitHubError) as caught:
            _create()
        assert caught.value.http_status is None and caught.value.operation == "pr create"
        assert "timed out" in caught.value.detail and "before retrying" in caught.value.repair_hint
    assert calls == ["POST", "GET"]


def test_successful_create_with_failed_settlement_warns_before_retry(monkeypatch):
    def request(method, path, **kwargs):
        if method == "POST":
            return {"html_url": "malformed URL"}
        raise GitHubError("read refused", http_status=403)

    monkeypatch.setattr(github, "request", request)
    with pytest.raises(github.SkillPublishGitHubError) as caught:
        _create()
    assert caught.value.repair_hint.startswith("GitHub PR creation reported success")
    assert "before retrying" in caught.value.repair_hint and caught.value.http_status == 403


@pytest.mark.parametrize("ref,content,reason,fragment", [
    ({"object": {"sha": BASE_SHA}}, b"{", "upstream_catalog_invalid", "Expecting"),
    ({"object": {"sha": BASE_SHA}}, b"[]", "upstream_catalog_invalid", "list, not an object"),
    ({"object": "not a mapping"}, b"{}", "upstream_read_failed", "not a mapping"),
    (["wrong shape"], b"{}", "upstream_read_failed", "Expected a JSON object"),
])
def test_malformed_upstream_answers_keep_their_cause(monkeypatch, ref, content, reason, fragment):
    monkeypatch.setattr(github, "request", lambda method, path, **kwargs: ref if "/git/refs/" in path else {
        "content": base64.b64encode(content).decode("ascii")})
    with pytest.raises(github.SkillPublishGitHubError) as caught:
        github.fetch_upstream_catalog(types.SimpleNamespace(), "hub", "project", "main")
    assert caught.value.reason_code == reason and fragment in caught.value.detail


@pytest.mark.parametrize("status,failure", [(403, "http"), (200, "graphql")])
def test_commit_failure_keeps_graphql_cause(responses, status, failure):
    responses(("POST", "/graphql", GitHubError("synthetic rejection", http_status=status, failure=failure)))
    with pytest.raises(github.SkillPublishGitHubError) as caught:
        github.commit_payload(types.SimpleNamespace(), "alice", "project", BRANCH, BASE_SHA, "Add demo", [])
    assert caught.value.reason_code == "commit_create_failed" and caught.value.operation == "graphql"
    assert caught.value.http_status == status and "synthetic rejection" in caught.value.detail
