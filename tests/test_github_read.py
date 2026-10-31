"""``github_read``: whoami, get, list, search, checks and the GET/query-only api, over a fake ``gh``.

Every answer is one line plus JSON; a cut is stated, a page says whether it is complete and how to
go on, and a result the tool limit cannot hold is saved whole as a task artifact behind a manifest.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest

from ouroboros.tools import github, github_checks, github_read
from tests.test_github_project_target import _context

BASE = "https://github.example/owner/selected"
API = "https://api.github.example"
SHA = "c" * 40


def _rest(payload, *, link_next="", status=200):
    """A ``gh api --include`` answer: status line, headers, blank line, JSON body."""
    head = f"HTTP/2.0 {status} OK\r\nContent-Type: application/json; charset=utf-8\r\nX-Oauth-Scopes: repo, read:org\r\n"
    if link_next:
        head += f'Link: <{link_next}>; rel="next", <{API}/x?page=9>; rel="last"\r\n'
    return head + "\r\n" + json.dumps(payload)


def _call(argv, kwargs=None):
    """The parts of one ``gh api`` argv: route, method, hostname, fields and whether headers were asked."""
    parts = SimpleNamespace(route="", method="GET", hostname="", include=False, fields={}, graphql="", input=False)
    rest = list(argv[2:])
    while rest:
        item = rest.pop(0)
        if item == "--include":
            parts.include = True
        elif item == "--method":
            parts.method = rest.pop(0)
        elif item == "--hostname":
            parts.hostname = rest.pop(0)
        elif item == "--input":
            parts.input, _ = True, rest.pop(0)
        elif item in "-F":
            key, _sep, value = rest.pop(0).partition("=")
            if key == "query":
                parts.graphql = value
            else:
                parts.fields[key] = (item, value)
        elif item == "graphql":
            parts.route = "graphql"
        else:
            parts.route = item
    if parts.route == "graphql" and kwargs and kwargs.get("input"):
        envelope = json.loads(kwargs["input"])
        parts.graphql, parts.fields = envelope["query"], envelope["variables"]
    return parts


def _issue(number, **extra):
    return {"number": number, "title": f"Issue {number}", "state": "open", "user": {"login": "dev", "type": "User"},
            "labels": [{"name": "bug"}], "assignees": [], "created_at": "2026-10-01T00:00:00Z",
            "updated_at": f"2026-10-{number:02d}T00:00:00Z", "closed_at": None, "comments": 2,
            "html_url": f"{BASE}/issues/{number}", "body": "b" * 10, **extra}


def _pr(number, **extra):
    return {"number": number, "title": f"PR {number}", "state": "open", "draft": False, "user": {"login": "dev"},
            "head": {"ref": f"feat/{number}", "label": f"dev:feat/{number}", "sha": SHA}, "base": {"ref": "main"},
            "created_at": "2026-10-01T00:00:00Z", "updated_at": f"2026-10-{number:02d}T00:00:00Z", "merged_at": None,
            "html_url": f"{BASE}/pull/{number}", "node_id": f"PR_node{number}", "labels": [], "body": "", **extra}


def _comment(cid, **extra):
    return {"id": cid, "user": {"login": "alice", "type": "User"}, "created_at": "2026-10-03T00:00:00Z",
            "updated_at": "2026-10-03T00:00:00Z", "html_url": f"{BASE}/issues/7#issuecomment-{cid}",
            "issue_url": f"{API}/repos/owner/selected/issues/7", "body": f"comment {cid}", **extra}


@pytest.mark.parametrize("retired,kind", [("get_github_issue", "comment"), ("get_github_pr", "comment"),
                                         ("get_github_pr", "review_comment"), ("get_github_pr", "review")])
def test_retired_detail_names_withhold_target_lists_but_allow_repo_sweeps(gh, monkeypatch, retired, kind):
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")
    gh.ctx.task_metadata["task_contract"] = {"disabled_tools": [retired]}
    registry = ToolRegistry(repo_dir=gh.ctx.repo_dir, drive_root=gh.ctx.drive_root)
    registry.set_context(gh.ctx)
    gh.answer = lambda p, argv, kw: _rest([])
    refused = registry.execute("github_read", {"method": " list ", "kind": f" {kind} ", "target": " 7 "})
    assert refused.startswith("⚠️ RESOURCE_CONSTRAINT_BLOCKED") and retired in refused and not gh.calls
    # Reviews require a PR; comments and review comments also offer repository-wide sweeps.
    sweep = kind if kind != "review" else "comment"
    for target in (None, "", 0):
        args = {"method": "list", "kind": sweep, **({"target": target} if target is not None else {})}
        text = registry.execute("github_read", args)
        assert text.startswith("github_read list "), text
        assert _payload(text)[1]["complete"] is True
    assert len(gh.calls) == 3


@pytest.fixture
def gh(monkeypatch, tmp_path):
    """A recording fake gh: ``world.answer(parts, argv, kwargs)`` returns stdout, ``(code, stderr)`` or raises."""
    world = SimpleNamespace(calls=[], answer=lambda parts, argv, kwargs: "[]")
    monkeypatch.setattr(github, "github_token_from_env_or_settings", lambda: "ghp_fixture")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_fixture")

    def run(argv, **kwargs):
        world.calls.append((argv, kwargs))
        if argv[1:5] == ["repo", "view", "--json", "nameWithOwner"]:
            return SimpleNamespace(returncode=0, stdout=json.dumps({"nameWithOwner": "owner/selected"}), stderr="")
        answer = world.answer(_call(argv, kwargs) if argv[1] == "api" else SimpleNamespace(route=" ".join(argv[1:3])), argv, kwargs)
        if isinstance(answer, tuple):
            return SimpleNamespace(returncode=answer[0], stdout="", stderr=answer[1])
        return SimpleNamespace(returncode=0, stdout=answer, stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    world.ctx, world.cwd = _context(tmp_path, "queued")
    entry = next(item for item in github.get_tools() if item.name == "github_read")  # the registered, invocation-scoped handler
    world.read = lambda **args: entry.handler(world.ctx, **args)
    world.routes = lambda: [_call(argv).route for argv, _ in world.calls if argv[1] == "api"]
    return world


def _payload(text):
    line, _nl, body = text.partition("\n")
    return line, json.loads(body)


# ---- whoami ------------------------------------------------------------------------------------------
def test_whoami_names_the_acting_login_token_kind_source_and_scopes(gh, monkeypatch):
    gh.answer = lambda p, argv, kw: _rest({"login": "ouroboros-agent", "id": 1, "type": "User", "name": None, "html_url": f"{BASE}"})
    line, payload = _payload(gh.read(method="whoami"))
    assert line == "github_read whoami: @ouroboros-agent (classic personal access token from env GITHUB_TOKEN)"
    assert payload["item"]["scopes"] == ["repo", "read:org"] and payload["item"]["token_source"] == "env GITHUB_TOKEN"
    assert "ghp_fixture" not in json.dumps(payload)
    argv, kw = gh.calls[0]
    assert argv == ["gh", "api", "--include", "user"] and kw["cwd"] == str(gh.ctx.repo_dir)  # the generic transport
    monkeypatch.setattr(github, "github_token_from_env_or_settings", lambda: "")
    monkeypatch.delenv("GITHUB_TOKEN")
    _line, payload = _payload(gh.read(method="whoami", repo="github.example/owner/selected"))
    assert payload["item"]["token_kind"] == "gh CLI login" and payload["item"]["host"] == "github.example"
    assert gh.calls[-1][0][-2:] == ["--hostname", "github.example"]


# ---- get ---------------------------------------------------------------------------------------------
def test_get_issue_states_the_body_cut_and_pages_its_comments_facet(gh):
    def answer(p, argv, kw):
        if p.route == "repos/{owner}/{repo}/issues/7":
            return json.dumps(_issue(7, body="x" * 2500))
        assert p.route.startswith("repos/{owner}/{repo}/issues/7/comments?") and p.include
        return _rest([_comment(1), _comment(2)], link_next=f"{API}/repos/owner/selected/issues/7/comments?per_page=2&page=2")
    gh.answer = answer

    line, payload = _payload(gh.read(method="get", kind="issue", target=7, include=["comments"], limit=2))

    assert line.startswith("github_read get issue #7 in owner/selected: 'Issue 7' (open, 1 facet(s))")
    item = payload["item"]
    assert item["body_chars"] == 2500 and item["body_truncated"] and len(item["body"]) == 2000 and item["author"] == "dev"
    assert "body cut at 2000 chars (of 2500): detail='full' returns it whole" in payload["omitted"]
    facet = payload["facets"]["comments"]
    assert [c["id"] for c in facet["items"]] == [1, 2] and facet["items"][0]["issue_number"] == 7
    assert (facet["complete"], facet["next_cursor"], facet["more"]) == (False, "2", "list(kind='comment', target=7, cursor='2')")
    assert "per_page=2&page=1" in gh.routes()[1]


def test_get_issue_that_is_a_pull_request_says_so(gh):
    gh.answer = lambda p, argv, kw: json.dumps(_issue(9, pull_request={"url": "x"}))
    _line, payload = _payload(gh.read(method="get", kind="issue", target=9))
    assert payload["item"]["is_pr"] is True and any("get(kind='pr')" in note for note in payload["omitted"])


def test_get_pr_facets_use_rest_pages_graphql_threads_and_the_diff_subcommand(gh):
    def answer(p, argv, kw):
        if p.route == "repos/{owner}/{repo}/pulls/7":
            return json.dumps(_pr(7, additions=3, deletions=1, changed_files=1, mergeable=True, body="desc"))
        if p.route == "graphql":
            if "nodes(ids:" in p.graphql:
                return json.dumps({"data": {"nodes": [{"id": "PR_node7", "commits": {"totalCount": 1}, "reviewDecision": "APPROVED"}]}})
            assert p.fields["owner"] == "owner" and p.fields["name"] == "selected"
            assert p.fields["number"] == 7 and p.fields["first"] == 30 and p.fields["after"] is None
            return json.dumps({"data": {"repository": {"pullRequest": {"reviewThreads": {
                "totalCount": 1, "pageInfo": {"hasNextPage": False, "endCursor": "Y3Vyc29y"},
                "nodes": [{"id": "PRRT_1", "isResolved": False, "isOutdated": False, "path": "a.py", "line": 3, "startLine": None,
                           "diffSide": "RIGHT", "comments": {"totalCount": 1, "nodes": [
                               {"id": "PRRC_1", "databaseId": 11, "author": {"login": "bob"}, "body": "nit", "createdAt": "t", "url": "u"}]}}]}}}}})
        if p.route == "pr diff":
            return "diff --git a/a.py b/a.py\n" + "+x\n" * 5000
        if "/pulls/7/files?" in p.route:
            return _rest([{"filename": "a.py", "status": "modified", "additions": 3, "deletions": 1, "changes": 4, "patch": "@@"}])
        if "/pulls/7/commits?" in p.route:
            return _rest([{"sha": SHA, "html_url": f"{BASE}/commit/{SHA}", "commit": {"message": "head\n\nbody", "author": {
                "name": "Dev", "email": "dev@example", "date": "2026-10-01T00:00:00Z"}}}])
        if "/pulls/7/reviews?" in p.route:
            return _rest([{"id": 5, "user": {"login": "rev"}, "state": "APPROVED", "submitted_at": "t", "commit_id": SHA, "html_url": "u", "body": "lgtm"}])
        assert "/pulls/7/comments?" in p.route and "since=2026-10-01" in p.route
        return _rest([_comment(3, path="a.py", line=3, in_reply_to_id=None)])
    gh.answer = answer

    line, payload = _payload(gh.read(method="get", kind="pr", target=7, since="2026-10-01T00:00:00Z",
                                     include=["files", "commits", "reviews", "review_comments", "threads", "diff"]))

    assert "6 facet(s)" in line and payload["item"]["head_sha"] == SHA and payload["item"]["mergeable"] is True
    facets = payload["facets"]
    assert facets["files"]["items"] == [{"filename": "a.py", "status": "modified", "additions": 3, "deletions": 1, "changes": 4}]
    assert facets["commits"]["items"][0]["author_email"] == "dev@example" and facets["commits"]["items"][0]["message_headline"] == "head"
    assert facets["reviews"]["items"][0]["state"] == "APPROVED" and facets["review_comments"]["items"][0]["path"] == "a.py"
    thread = facets["threads"]["items"][0]
    assert (thread["id"], thread["resolved"], thread["comments"][0]["id"]) == ("PRRT_1", False, 11) and facets["threads"]["total_count"] == 1
    assert facets["diff"]["truncated"] and facets["diff"]["chars"] > 8000 and len(facets["diff"]["text"]) == 8000
    assert any("diff cut at 8000" in note for note in payload["omitted"])
    diff_argv = next(argv for argv, _ in gh.calls if argv[1:3] == ["pr", "diff"])
    assert diff_argv == ["gh", "pr", "diff", "7"] and "--repo" not in diff_argv


def test_get_with_a_cursor_pages_exactly_one_facet(gh):
    text = gh.read(method="get", kind="pr", target=7, include=["files", "commits"], cursor="2")
    assert text.startswith("⚠️ TOOL_ARG_ERROR: a cursor on get pages exactly one include facet") and not gh.calls
    assert gh.read(method="get", kind="pr", target=7, include=["weird"]).startswith("⚠️ TOOL_ARG_ERROR: unknown include facet(s) weird")


@pytest.mark.parametrize("detail", ["concise", "full"])
@pytest.mark.parametrize("fields", [[], ["review_decision"]])
def test_get_pr_keeps_review_decision_in_both_detail_modes(gh, detail, fields):
    def answer(p, argv, kw):
        if p.route != "graphql":
            return json.dumps(_pr(7, body="x" * 3000))
        assert p.fields == {"ids": ["PR_node7"]} and "reviewDecision" in p.graphql
        return json.dumps({"data": {"nodes": [{"id": "PR_node7", "commits": {"totalCount": 1}, "reviewDecision": "APPROVED"}]}})
    gh.answer = answer
    _line, payload = _payload(gh.read(method="get", kind="pr", target=7, detail=detail, fields=fields))
    assert payload["item"]["review_decision"] == "APPROVED" and len(gh.calls) == 2
    if fields:
        assert payload["item"] == {"review_decision": "APPROVED"}
    else:
        assert payload["item"]["body"] == "x" * (3000 if detail == "full" else 2000)


def test_full_pr_list_enrichment_preserves_raw_head_and_base(gh):
    def answer(p, argv, kw):
        if p.route != "graphql":
            return _rest([_pr(7, head={"ref": "feature", "repo": {"id": 1}}, base={"ref": "main", "repo": {"id": 2}})])
        return json.dumps({"data": {"nodes": [{"id": "PR_node7", "commits": {"totalCount": 1}, "reviewDecision": "APPROVED",
            "headRefName": "feature", "headRefOid": SHA, "headRepositoryOwner": {"login": "dev"}, "baseRefName": "main"}]}})
    gh.answer = answer
    _line, payload = _payload(gh.read(method="list", kind="pr", detail="full"))
    assert payload["items"][0]["head"]["repo"] == {"id": 1}
    assert payload["items"][0]["base"]["repo"] == {"id": 2}


@pytest.mark.parametrize("detail", ["concise", "full"])
@pytest.mark.parametrize("null_field", [None, "headRefOid", "headRefName", "headRepositoryOwner", "baseRefName", "all"])
def test_pr_list_overlays_only_nonnull_graphql_head_and_base_facts(gh, detail, null_field):
    node = {"id": "PR_node7", "commits": {"totalCount": 4}, "reviewDecision": "APPROVED",
            "headRefName": "new-feature", "headRefOid": "d" * 40,
            "headRepositoryOwner": {"login": "new-dev"}, "baseRefName": "new-base"}
    nullable = ("headRefOid", "headRefName", "headRepositoryOwner", "baseRefName")
    for key in nullable if null_field == "all" else (null_field,):
        if key is not None:
            node[key] = None
    gh.answer = lambda p, argv, kw: json.dumps({"data": {"nodes": [node]}}) if p.route == "graphql" else _rest([_pr(7, state="closed")])
    _line, payload = _payload(gh.read(method="list", kind="pr", state="closed", detail=detail))
    assert gh.routes()[0].startswith("repos/{owner}/{repo}/pulls?state=closed&") and len(gh.calls) == 2
    item = payload["items"][0]
    sha = SHA if null_field in ("headRefOid", "all") else "d" * 40
    ref = "feat/7" if null_field in ("headRefName", "all") else "new-feature"
    owner = "dev" if null_field in ("headRepositoryOwner", "all") else "new-dev"
    base = "main" if null_field in ("baseRefName", "all") else "new-base"
    if detail == "full":
        assert item["head"] == {"sha": sha, "ref": ref, "label": f"{owner}:{ref}"}
        assert item["base"]["ref"] == base
    else:
        assert (item["head_sha"], item["head"], item["base"]) == (sha, f"{owner}:{ref}", base)
    assert item["commits"] == 4 and item["review_decision"] == "APPROVED"


@pytest.mark.parametrize("detail", ["concise", "full"])
@pytest.mark.parametrize("nested", ["comments", "replies"])
def test_nested_cuts_are_disclosed_in_both_detail_modes(gh, detail, nested):
    def answer(p, argv, kw):
        if p.route != "graphql":
            return json.dumps(_pr(7))
        if "nodes(ids:" in p.graphql:
            return json.dumps({"data": {"nodes": [{"id": "PR_node7", "commits": {"totalCount": 1}, "reviewDecision": "APPROVED"}]}})
        node = {"id": "PRRT_1" if nested == "comments" else "DC_1",
                nested: {"totalCount": 51, "nodes": [{"id": f"C_{i}", "body": "x"} for i in range(50)]}}
        connection = {"totalCount": 1, "pageInfo": {"hasNextPage": False}, "nodes": [node]}
        repository = {"pullRequest": {"reviewThreads": connection}} if nested == "comments" else {"discussion": {"comments": connection}}
        return json.dumps({"data": {"repository": repository}})
    gh.answer = answer
    args = {"method": "get", "kind": "pr", "include": ["threads"]} if nested == "comments" else {"method": "list", "kind": "discussion"}
    _line, payload = _payload(gh.read(**args, target=7, detail=detail))
    notice = " ".join(payload["omitted"])
    assert f"1 {nested} beyond the 50 fetched" in notice
    assert ("review_comment" in notice and "target=7" in notice) if nested == "comments" else ("DC_1" in notice and "replies(first:100" in notice)
    if detail == "full":
        page = payload["facets"]["threads"] if nested == "comments" else payload
        assert len(page["items"][0][nested]["nodes"]) == 50


@pytest.mark.parametrize("kind,target,route,label", [
    ("comment", 44, "repos/{owner}/{repo}/issues/comments/44", "comment 44"),
    ("review_comment", 45, "repos/{owner}/{repo}/pulls/comments/45", "review_comment 45"),
    ("release", "v1.2", "repos/{owner}/{repo}/releases/tags/v1.2", "release v1.2"),
    ("label", "bug", "repos/{owner}/{repo}/labels/bug", "label bug"),
])
def test_get_reads_one_object_by_kind(gh, kind, target, route, label):
    gh.answer = lambda p, argv, kw: json.dumps({**_comment(44), "html_url": f"{BASE}/x", "tag_name": "v1.2", "name": "bug", "assets": []})
    line, payload = _payload(gh.read(method="get", kind=kind, target=target))
    assert line == f"github_read get {label} in owner/selected" and gh.routes() == [route]
    assert payload["target"] == target and payload["item"]


def test_get_discussion_reads_graphql_and_reviews_are_listed_in_their_pr(gh):
    gh.answer = lambda p, argv, kw: json.dumps({"data": {"repository": {"discussion": {
        "id": "D_1", "number": 5, "title": "Q", "url": "u", "createdAt": "t", "updatedAt": "t", "author": {"login": "ann"},
        "category": {"name": "Q&A"}, "isAnswered": True, "comments": {"totalCount": 3}, "locked": False, "body": "why?"}}}})
    line, payload = _payload(gh.read(method="get", kind="discussion", target=5))
    assert line == "github_read get discussion #5: 'Q'" and payload["item"]["category"] == "Q&A" and payload["item"]["body"] == "why?"
    assert gh.read(method="get", kind="review", target=5).startswith("⚠️ TOOL_ARG_ERROR: a review is read in its pull request")
    assert gh.read(method="get", kind="issue", target=0).startswith("⚠️ TOOL_ARG_ERROR: issue number must be positive.")


def test_explicit_repo_is_substituted_into_routes_with_its_host(gh):
    gh.answer = lambda p, argv, kw: json.dumps(_issue(7))
    gh.read(method="get", kind="issue", target=7, repo="github.example/owner/selected")
    argv, kw = gh.calls[0]
    assert argv == ["gh", "api", "repos/owner/selected/issues/7", "--hostname", "github.example"]
    assert kw["cwd"] == str(gh.cwd) and "GH_REPO" not in kw["env"]
    gh.answer = lambda p, argv, kw: json.dumps({"data": {"repository": {"discussion": None}}})
    text = gh.read(method="get", kind="discussion", target=5, repo="owner/selected")
    assert "discussion #5 was not found" in text
    fields = _call(*gh.calls[-1]).fields
    assert fields["owner"] == "owner" and fields["name"] == "selected"


# ---- list --------------------------------------------------------------------------------------------
def test_list_issues_filters_pull_requests_and_reads_completeness_from_the_link_header(gh):
    gh.answer = lambda p, argv, kw: _rest([_issue(1), _issue(2, pull_request={"url": "x"}), _issue(3)],
                                          link_next=f"{API}/repos/owner/selected/issues?page=2&per_page=3")
    line, payload = _payload(gh.read(method="list", kind="issue", since="2026-09-30T00:00:00Z", limit=3, labels="bug", author="dev", state="all"))

    assert line == "github_read list issue (all) in owner/selected: 2 item(s) since 2026-09-30T00:00:00Z, more: cursor='2'"
    route = gh.routes()[0]
    assert route.startswith("repos/{owner}/{repo}/issues?") and "state=all" in route and "labels=bug" in route and "creator=dev" in route
    assert "since=2026-09-30T00%3A00%3A00Z" in route and "sort=updated" in route and "per_page=3&page=1" in route
    assert [i["number"] for i in payload["items"]] == [1, 3] and payload["items"][0]["body_preview"] == "b" * 10
    assert (payload["complete"], payload["next_cursor"]) == (False, "2")
    assert "1 pull request(s) on this page not shown; list(kind='pr') reads them" in payload["omitted"]
    gh.answer = lambda p, argv, kw: _rest([])
    _line, payload = _payload(gh.read(method="list", kind="issue", cursor="2"))
    assert "page=2" in gh.routes()[-1] and payload["complete"] is True and payload["items"] == []
    assert gh.read(method="list", kind="issue", cursor="abc").startswith("⚠️ TOOL_ARG_ERROR: cursor 'abc' is not a page cursor")


def test_list_prs_keeps_exact_commit_counts_and_review_decisions_from_one_graphql_read(gh):
    rows = [_pr(100), _pr(101, draft=True)]

    def answer(p, argv, kw):
        if p.route == "graphql":
            request = json.loads(kw["input"])
            assert request == {"query": github_read._PR_COMMIT_COUNTS_QUERY, "variables": {"ids": ["PR_node100", "PR_node101"]}}
            return json.dumps({"data": {"nodes": [{"id": "PR_node101", "commits": {"totalCount": 7}, "reviewDecision": None},
                                                  {"id": "PR_node100", "commits": {"totalCount": 250}, "reviewDecision": "APPROVED"}]}})
        assert p.route.startswith("repos/{owner}/{repo}/pulls?state=open&sort=updated&direction=desc")
        return _rest(rows)
    gh.answer = answer

    line, payload = _payload(gh.read(method="list", kind="pr"))

    assert line == "github_read list pr (open) in owner/selected: 2 item(s), complete"
    assert [(i["number"], i["commits"], i["review_decision"], i["draft"]) for i in payload["items"]] == [(100, 250, "APPROVED", False), (101, 7, None, True)]
    count_argv, count_kw = gh.calls[1]
    assert count_argv == ["gh", "api", "graphql", "--hostname", "github.example", "--input", "-"] and count_kw["cwd"] == str(gh.ctx.repo_dir)


@pytest.mark.parametrize("case", ["bool-count", "missing-node", "no-id", "foreign-repo", "graphql-error"])
def test_an_unreadable_pr_count_withholds_the_list(gh, case):
    rows = [_pr(1), _pr(2)]
    if case == "no-id":
        rows[1]["node_id"] = None
    if case == "foreign-repo":
        rows[1]["html_url"] = "https://github.example/owner/other/pull/2"
    answers = {"bool-count": json.dumps({"data": {"nodes": [{"id": "PR_node1", "commits": {"totalCount": 1}}, {"id": "PR_node2", "commits": {"totalCount": True}}]}}),
               "missing-node": json.dumps({"data": {"nodes": [{"id": "PR_node1", "commits": {"totalCount": 1}}]}}),
               "graphql-error": (1, "gh: Could not resolve to a node (HTTP 404)")}
    gh.answer = lambda p, argv, kw: answers.get(case, "{}") if p.route == "graphql" else _rest(rows)
    text = gh.read(method="list", kind="pr")
    assert text.startswith("⚠️ ") and "PR 1" not in text and "commits" not in text.split("\n")[0]
    assert len(gh.calls) == (1 if case in ("no-id", "foreign-repo") else 2)


def test_list_pr_since_stops_at_the_first_older_row(gh):
    gh.answer = lambda p, argv, kw: json.dumps({"data": {"nodes": [{"id": "PR_node3", "commits": {"totalCount": 1}, "reviewDecision": None}]}}) \
        if p.route == "graphql" else _rest([_pr(3), _pr(1)], link_next=f"{API}/x?page=2")
    _line, payload = _payload(gh.read(method="list", kind="pr", since="2026-10-02T00:00:00Z"))
    assert [i["number"] for i in payload["items"]] == [3] and payload["complete"] is True and payload["next_cursor"] == ""


@pytest.mark.parametrize("filters", [{"state": "merged"}, {"labels": "bug"}, {"author": "dev"}])
@pytest.mark.parametrize("detail", ["concise", "full"])
def test_search_backed_pr_lists_keep_facts_and_withhold_unreadable_counts(gh, filters, detail):
    def answer(p, argv, kw):
        if p.route == "graphql":
            return json.dumps({"data": {"nodes": [{"id": "PR_node4", "commits": {"totalCount": 101},
                "reviewDecision": "APPROVED", "headRefName": "feat", "headRefOid": SHA,
                "headRepositoryOwner": {"login": "dev"}, "baseRefName": "main", "isDraft": True, "mergedAt": "t"}]}})
        return _rest({"total_count": 1, "items": [_issue(4, node_id="PR_node4", html_url=f"{BASE}/pull/4", pull_request={})]})
    gh.answer = answer
    _line, payload = _payload(gh.read(method="list", kind="pr", detail=detail, **filters))
    item = payload["items"][0]
    assert item["commits"] == 101 and item["review_decision"] == "APPROVED" and item["draft"] is True
    assert (item["head"]["ref"], item["base"]["ref"]) == ("feat", "main") if detail == "full" else (item["head"], item["base"]) == ("dev:feat", "main")
    assert gh.calls[0][0] == ["gh", "repo", "view", "--json", "nameWithOwner"]
    assert "advanced_search=true" in gh.routes()[0] and len(gh.routes()) == 2
    gh.answer = lambda p, argv, kw: '{"data":{"nodes":[]}}' if p.route == "graphql" else answer(p, argv, kw)
    text = gh.read(method="list", kind="pr", **filters)
    assert "commit counts" in text and '"items"' not in text


@pytest.mark.parametrize("kind,target,route,label", [
    ("comment", 0, "repos/{owner}/{repo}/issues/comments?since=S&sort=updated&direction=asc", "comment (repo-wide, issues and PR conversations)"),
    ("comment", 7, "repos/{owner}/{repo}/issues/7/comments?since=S", "comment on #7"),
    ("review_comment", 0, "repos/{owner}/{repo}/pulls/comments?since=S&sort=updated&direction=asc", "review_comment (repo-wide inline review comments)"),
    ("review_comment", 7, "repos/{owner}/{repo}/pulls/7/comments?since=S", "review_comment on PR #7"),
    ("review", 7, "repos/{owner}/{repo}/pulls/7/reviews?", "review on PR #7"),
    ("notification", 0, "repos/{owner}/{repo}/notifications?since=S&all=true", "notification (all)"),
    ("release", 0, "repos/{owner}/{repo}/releases?", "release"),
    ("label", 0, "repos/{owner}/{repo}/labels?", "label"),
])
def test_list_kinds_take_since_and_target_on_the_matching_route(gh, kind, target, route, label):
    gh.answer = lambda p, argv, kw: _rest([])
    args = {"method": "list", "kind": kind, "since": "S", "state": "all"}
    if target:
        args["target"] = target
    line, payload = _payload(gh.read(**args))
    assert gh.routes()[0] == route.replace("?S", "?since=S") + ("&" if not route.endswith("?") else "") + "per_page=30&page=1"
    since = " since S" if kind != "label" else ""
    assert line.startswith(f"github_read list {label} in selected Project: 0 item(s){since}, complete") and payload["complete"]
    if kind == "notification":
        assert any("classic PAT" in note for note in payload["omitted"])


def test_list_reviews_need_a_pull_request(gh):
    assert gh.read(method="list", kind="review").startswith("⚠️ TOOL_ARG_ERROR: list(kind='review') needs target") and not gh.calls


@pytest.mark.parametrize("detail", ["concise", "full"])
@pytest.mark.parametrize("since", ["", "2026-10-01T00:00:00Z"])
def test_label_list_discloses_unsupported_since_without_claiming_a_filter(gh, detail, since):
    # A synthetic old timestamp makes this nonempty fixture distinguish an applied filter from an ignored one.
    label = {"id": 1, "name": "bug", "color": "ff0000", "updated_at": "2026-09-01T00:00:00Z"}
    gh.answer = lambda p, argv, kw: _rest([label], link_next=f"{API}/repos/owner/selected/labels?page=2")
    line, payload = _payload(gh.read(method="list", kind="label", since=since, detail=detail))
    assert [row["name"] for row in payload["items"]] == ["bug"]
    assert "item(s), more:" in line and "since=" not in gh.routes()[0]
    assert "since" not in payload and (payload["complete"], payload["next_cursor"]) == (False, "2")
    if since:
        assert since not in line and "labels endpoint has no since filter" in payload["note"]
        assert "unfiltered" in payload["note"]
    else:
        assert "note" not in payload


def test_list_discussions_and_their_comments_with_replies_are_graphql_pages(gh):
    def answer(p, argv, kw):
        assert p.route == "graphql" and p.fields["owner"] == "owner"
        if "discussions(" in p.graphql:
            assert p.fields["first"] == 2 and p.fields["after"] == "abc"
            return json.dumps({"data": {"repository": {"discussions": {"totalCount": 9, "pageInfo": {"hasNextPage": True, "endCursor": "def"}, "nodes": [
                {"id": "D_1", "number": 1, "title": "A", "url": "u", "createdAt": "t", "updatedAt": "2026-10-05", "author": {"login": "x"},
                 "category": {"name": "General"}, "isAnswered": False, "comments": {"totalCount": 1}, "body": "hello"}]}}}})
        assert p.fields["number"] == 1
        return json.dumps({"data": {"repository": {"discussion": {"comments": {"totalCount": 1, "pageInfo": {"hasNextPage": False, "endCursor": None}, "nodes": [
            {"id": "DC_1", "databaseId": 10, "body": "c", "author": {"login": "y"}, "createdAt": "t", "updatedAt": "t", "url": "u", "isAnswer": False,
             "replies": {"totalCount": 51, "nodes": [{"id": "DC_2", "databaseId": 11, "body": "r", "author": {"login": "z"}, "createdAt": "t", "url": "u"}]}}]}}}}})
    gh.answer = answer
    line, payload = _payload(gh.read(method="list", kind="discussion", limit=2, cursor="abc"))
    assert line == "github_read list discussion in selected Project: 1 item(s), more: cursor='def'"
    assert payload["total_count"] == 9 and payload["items"][0]["category"] == "General" and payload["items"][0]["body_preview"] == "hello"
    line, payload = _payload(gh.read(method="list", kind="discussion", target=1))
    assert line.startswith("github_read list discussion #1 comments (with replies)") and payload["items"][0]["replies"][0]["id"] == "DC_2"
    assert any("50 replies beyond the 1 fetched" in note for note in payload["omitted"])


# ---- search ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind,key,old", [("review", "submitted_at", "2026-01-01"),
                                        ("release", "published_at", "2026-01-01"), ("release", "published_at", None)])
def test_since_preserves_unordered_rest_page_continuations(gh, kind, key, old):
    gh.answer = lambda p, argv, kw: _rest([{"id": 1, key: old}], link_next=f"{API}/x?page=2")
    _line, payload = _payload(gh.read(method="list", kind=kind, target=7, since="2026-10-01"))
    assert payload["items"] == [] and (payload["complete"], payload["next_cursor"]) == (False, "2")


@pytest.mark.parametrize("target,newest_first", [(7, False), (None, True)])
def test_since_respects_discussion_connection_order(gh, target, newest_first):
    def answer(p, argv, kw):
        connection = {"totalCount": 2, "pageInfo": {"hasNextPage": True, "endCursor": "next"},
                      "nodes": [{"id": "D_1", "updatedAt": "2026-01-01"}]}
        node = {"discussion": {"comments": connection}} if target else {"discussions": connection}
        assert ("orderBy:{field:UPDATED_AT,direction:DESC}" in p.graphql) is newest_first
        return json.dumps({"data": {"repository": node}})
    gh.answer = answer
    _line, payload = _payload(gh.read(method="list", kind="discussion", target=target, since="2026-10-01"))
    assert payload["items"] == []
    assert (payload["complete"], payload["next_cursor"]) == ((True, "") if newest_first else (False, "next"))


@pytest.mark.parametrize("kind", ["comment", "review_comment"])
def test_since_on_ascending_comment_sweeps_keeps_the_server_cursor(gh, kind):
    def answer(p, argv, kw):
        assert "since=2026-10-01" in p.route and "sort=updated&direction=asc" in p.route
        return _rest([_comment(1, updated_at="2026-10-02")], link_next=f"{API}/x?page=2")
    gh.answer = answer
    _line, payload = _payload(gh.read(method="list", kind=kind, since="2026-10-01"))
    assert len(payload["items"]) == 1 and (payload["complete"], payload["next_cursor"]) == (False, "2")


def test_search_scopes_to_the_repository_and_reports_total_count(gh):
    gh.answer = lambda p, argv, kw: _rest({"total_count": 120, "incomplete_results": True, "items": [_issue(5)]}, link_next=f"{API}/search/issues?page=2")
    line, payload = _payload(gh.read(method="search", kind="issue", query="is:open label:bug", repo="owner/selected", since="2026-10-01T00:00:00Z"))
    assert line == "github_read search issue in owner/selected: 1 item(s) of 120 for 'is:open label:bug repo:owner/selected updated:>=2026-10-01 is:issue', more: cursor='2'"
    assert payload["total_count"] == 120 and any("incomplete_results" in note for note in payload["omitted"])
    assert gh.routes()[0].startswith("search/issues?q=is%3Aopen+label%3Abug+repo%3Aowner%2Fselected+updated%3A%3E%3D2026-10-01+is%3Aissue&advanced_search=true")
    assert gh.read(method="search", kind="issue").startswith("⚠️ TOOL_ARG_ERROR: search needs a query")
    assert gh.read(method="search", kind="label", query="x").startswith("⚠️ TOOL_ARG_ERROR: search needs kind ∈ issue, pr, discussion")


def test_search_discussions_is_a_graphql_search_without_repository_variables(gh):
    gh.answer = lambda p, argv, kw: json.dumps({"data": {"search": {"discussionCount": 1, "pageInfo": {"hasNextPage": False, "endCursor": None}, "nodes": [
        {"id": "D_9", "number": 9, "title": "T", "url": "u", "createdAt": "t", "updatedAt": "t", "author": {"login": "a"}, "category": {"name": "Ideas"},
         "isAnswered": False, "comments": {"totalCount": 0}, "repository": {"nameWithOwner": "owner/selected"}}]}}})
    line, payload = _payload(gh.read(method="search", kind="discussion", query="roadmap", repo="owner/selected"))
    assert line == "github_read search discussion in owner/selected: 1 item(s) of 1 for 'roadmap repo:owner/selected'"
    fields = _call(*gh.calls[0]).fields
    assert "owner" not in fields and fields["q"] == "roadmap repo:owner/selected" and payload["items"][0]["repo"] == "owner/selected"


# ---- checks ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("target,expected", [(7, {"number": 7, "sha": ""}), ("7", {"number": "7", "sha": ""}), (SHA, {"number": 0, "sha": SHA}),
                                             (SHA.upper(), {"number": 0, "sha": SHA}), ("main", {"number": 0, "sha": "main"})])
def test_checks_is_the_existing_reader_registered_as_a_method(gh, monkeypatch, target, expected):
    seen = {}

    def get_checks(ctx, number=0, sha="", wait_seconds=0, repo=""):
        seen.update(number=number, sha=sha, wait_seconds=wait_seconds, repo=repo)
        return "GitHub checks for commit"
    monkeypatch.setattr(github_checks, "get_checks", get_checks)
    assert gh.read(method="checks", target=target, wait_seconds=30, repo="owner/selected") == "GitHub checks for commit"
    assert seen == {**expected, "wait_seconds": 30, "repo": "owner/selected"}


# ---- api ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("next_route", [f"{API}/repos/owner/selected/issues/7/timeline?per_page=30&page=2", ""])
def test_rest_api_preserves_pagination_and_a_usable_next_route(gh, next_route):
    def answer(p, argv, kw):
        assert p.include and p.method == "GET"
        return _rest([{"event": "labeled"}], link_next=next_route)
    gh.answer = answer
    _line, payload = _payload(gh.read(method="api", route="repos/{owner}/{repo}/issues/7/timeline"))
    assert payload["data"] == [{"event": "labeled"}]
    assert (payload["complete"], payload["next_cursor"]) == (not bool(next_route), next_route)
    if next_route:
        assert next_route in payload["more"]
        gh.read(method="api", route=payload["next_cursor"])
        assert gh.routes()[-1] == next_route


@pytest.mark.parametrize("document", [
    'query{search(query:"fix mutation bug",type:ISSUE,first:1){issueCount}}',
    'query{search(query:"""fix mutation bug\nwith a block string""",type:ISSUE,first:1){issueCount}}',
    '# mutation Fake { ignored }\nquery { viewer { login } }',
    'query mutation { viewer { mutation: login } }',
])
def test_graphql_query_classification_ignores_strings_comments_and_names(gh, document):
    gh.answer = lambda p, argv, kw: '{"data":{}}'
    assert _payload(gh.read(method="api", graphql=document))[1]["data"] == {"data": {}}
    assert len(gh.calls) == 1


@pytest.mark.parametrize("document", [
    'mutation { addReaction(input:{}) { clientMutationId } }',
    '# comment before the operation\nmutation M { addReaction(input:{}) { clientMutationId } }',
    'query Q { viewer { login } } mutation M { addReaction(input:{}) { clientMutationId } }',
])
def test_graphql_real_mutations_are_refused_before_any_read_send(gh, document):
    text = gh.read(method="api", graphql=document)
    assert "GH_API_READ_ONLY" in text and "github_write(method='api'" in text and not gh.calls


@pytest.mark.parametrize("query_value", ["repo:owner/selected is:open", "mutation { deleteIssue(input:{id:1}) { clientMutationId } }"])
def test_graphql_envelope_preserves_document_and_variable_types(gh, query_value):
    document = "query($query:String!){search(query:$query,type:ISSUE,first:1){issueCount}}"
    variables = {"query": query_value, "input": {"body": "hello", "enabled": True}, "ids": ["I_1", "I_2"],
                 "after": None, "weight": 1.25}
    gh.answer = lambda p, argv, kw: json.dumps({"data": {"search": {"issueCount": 1}}})
    _line, payload = _payload(gh.read(method="api", graphql=document, variables=variables, repo="github.example/owner/selected"))
    argv, kw = gh.calls[0]
    assert argv == ["gh", "api", "graphql", "--input", "-", "--hostname", "github.example"]
    assert json.loads(kw["input"]) == {"query": document, "variables": variables}
    assert payload["data"]["data"]["search"]["issueCount"] == 1


def test_graphql_bound_repository_variables_use_the_selected_project_slug(gh):
    gh.answer = lambda p, argv, kw: json.dumps({"data": {"repository": {"id": "R_1"}}})
    gh.read(method="api", graphql="query($owner:String!,$name:String!){repository(owner:$owner,name:$name){id}}")
    assert gh.calls[0][0] == ["gh", "repo", "view", "--json", "nameWithOwner"]
    assert json.loads(gh.calls[1][1]["input"])["variables"] == {"owner": "owner", "name": "selected"}
    assert all(kw["cwd"] == str(gh.cwd) for _argv, kw in gh.calls)


def test_api_sends_get_routes_and_graphql_queries_only(gh):
    gh.answer = lambda p, argv, kw: json.dumps([{"event": "labeled"}])
    line, payload = _payload(gh.read(method="api", route="repos/{owner}/{repo}/issues/7/timeline?per_page=100", repo="owner/selected"))
    assert line == "github_read api GET repos/{owner}/{repo}/issues/7/timeline?per_page=100 in owner/selected: 1 item(s)"
    assert gh.calls[0][0] == ["gh", "api", "--include", "repos/owner/selected/issues/7/timeline?per_page=100"] and payload["data"] == [{"event": "labeled"}]
    gh.answer = lambda p, argv, kw: json.dumps({"data": {"viewer": {"login": "me"}}})
    _line, payload = _payload(gh.read(method="api", graphql="query { viewer { login } }"))
    assert payload["data"] == {"data": {"viewer": {"login": "me"}}} and "owner" not in _call(*gh.calls[-1]).fields
    _line, payload = _payload(gh.read(method="api", graphql="query($owner:String!,$name:String!){repository(owner:$owner,name:$name){id}}", variables={"x": 1}))
    assert _call(*gh.calls[-1]).fields == {"owner": "owner", "name": "selected", "x": 1}
    for document in ("mutation { addReaction(input:{}) { clientMutationId } }", "mutation Add($id:ID!) { x }", "query Q { a }\nmutation M { b }"):
        text = gh.read(method="api", graphql=document)
        assert text.startswith("⚠️ GH_API_READ_ONLY") and "github_write(method='api'" in text
    assert gh.read(method="api", route="POST repos/x/y/issues").startswith("⚠️ GH_API_READ_ONLY")
    assert gh.read(method="api").startswith("⚠️ TOOL_ARG_ERROR: api takes exactly one of route")
    assert gh.read(method="api", route="user", graphql="query{x}").startswith("⚠️ TOOL_ARG_ERROR: api takes exactly one of route")
    assert len(gh.routes()) == 3  # three API sends; bound slug resolution is a separate read


# ---- shape, detail, fields, artifacts ------------------------------------------------------------------
def test_detail_full_returns_github_objects_whole_and_fields_project_them(gh):
    gh.answer = lambda p, argv, kw: json.dumps(_issue(7, body="y" * 3000, extra_key="kept"))
    _line, payload = _payload(gh.read(method="get", kind="issue", target=7, detail="full"))
    assert payload["item"]["body"] == "y" * 3000 and payload["item"]["extra_key"] == "kept" and "omitted" not in payload
    gh.answer = lambda p, argv, kw: json.dumps(_issue(7))
    _line, payload = _payload(gh.read(method="get", kind="issue", target=7, fields=["number", "title", "nope"]))
    assert payload["item"] == {"number": 7, "title": "Issue 7"} and payload["omitted"] == ["fields not present: nope"]


def test_a_result_beyond_the_tool_limit_is_saved_whole_as_a_task_artifact(gh, monkeypatch):
    from ouroboros.artifacts import task_artifact_dir_path

    monkeypatch.setattr(github_read, "tool_result_limit", lambda name: 400)
    gh.answer = lambda p, argv, kw: _rest([_issue(n, body="z" * 300) for n in range(1, 6)], link_next=f"{API}/x?page=2")
    text = gh.read(method="list", kind="issue", limit=5)
    line, manifest = _payload(text)
    assert line.startswith("github_read list issue (open) in owner/selected: 5 item(s), more: cursor='2' — saved whole as task artifact github_read-list-issue-")
    saved = manifest["saved"]
    path = task_artifact_dir_path(gh.ctx.drive_root, "task-fixture") / saved["path"]
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == saved["sha256"] and len(data) == saved["bytes"] and saved["absolute_path"] == str(path)
    whole = json.loads(data)
    assert len(whole["items"]) == 5 and whole["items"][0]["body_preview"].startswith("zzz")  # nothing was cut
    assert manifest["summary"] == {"complete": False, "next_cursor": "2", "items": 5} and "nothing cut" in manifest["note"]
    assert len(text) < 1200


@pytest.mark.parametrize("args,fragment", [
    ({"method": "nope"}, "method must be one of"),
    ({"method": "get"}, "get needs kind ∈"),
    ({"method": "list"}, "list needs kind ∈"),
    ({"method": "list", "kind": "widget"}, "kind must be one of"),
    ({"method": "list", "kind": "issue", "limit": 0}, "limit is a whole number from 1 to 100"),
    ({"method": "list", "kind": "issue", "limit": "ten"}, "limit is a whole number from 1 to 100"),
    ({"method": "get", "kind": "issue", "target": 7, "detail": "huge"}, "detail is 'concise' or 'full'"),
    ({"method": "whoami", "repo": None}, "GH_TARGET_INVALID"),
])
def test_argument_refusals_are_typed_and_send_nothing(gh, args, fragment):
    gh.ctx._active_builtin_tool_result = None
    text = gh.read(**args)
    assert fragment in text and not gh.calls
    result = gh.ctx._active_builtin_tool_result
    assert result.status == "error" and result.meta.get("operation_outcome") == "completed_no_effect"


def test_a_limit_over_the_page_size_is_capped_with_a_note(gh):
    gh.answer = lambda p, argv, kw: _rest([])
    _line, payload = _payload(gh.read(method="list", kind="label", limit=500))
    assert "per_page=100" in gh.routes()[0] and payload["omitted"] == ["limit 500 capped at 100 (GitHub's page size); page on with cursor"]


def test_github_read_is_registered_read_only_with_one_required_method(gh):
    from ouroboros import tool_capabilities
    from ouroboros.consciousness_authority import OBSERVE_DISABLED
    from ouroboros.safety import POLICY_SKIP, TOOL_POLICY
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.registry_guards import _GITHUB_TOKEN_TOOLS

    entry = next(item for item in github.get_tools() if item.name == "github_read")
    assert TOOL_POLICY["github_read"] == POLICY_SKIP and "github_read" in _GITHUB_TOKEN_TOOLS and not entry.mutates_worktree
    assert entry.schema["parameters"]["required"] == ["method"] and entry.schema["parameters"]["properties"]["method"]["enum"] == list(github_read.METHODS)
    assert "repo" in entry.schema["parameters"]["properties"]
    assert github_checks._CHECKS_WAIT_CAP_SEC + github_checks._CHECKS_READ_BUDGET_SEC < entry.timeout_sec
    for listing in (OBSERVE_DISABLED, tool_capabilities.OBSERVE_WORLD_MUTATION_TOOLS, tool_capabilities.READ_ONLY_PARALLEL_TOOLS):
        assert "github_read" not in listing
    assert "github_read" in ToolRegistry._FROZEN_TOOL_MODULES or "github" in ToolRegistry._FROZEN_TOOL_MODULES
    gh.answer = lambda p, argv, kw: json.dumps(_issue(7))
    registry = ToolRegistry(repo_dir=gh.ctx.repo_dir, drive_root=gh.ctx.drive_root)
    registry.set_context(gh.ctx)
    result = registry.execute_result("github_read", {"method": "get", "kind": "issue", "target": 7})
    assert result.status == "ok" and result.text.startswith("github_read get issue #7"), result
    rejected = registry.execute_result("github_read", {"method": "get", "kind": "issue", "number": 7})
    assert rejected.status != "ok" and "number" in rejected.text
