"""#1607: ``list_github_prs`` keeps exact commit counts without fetching commit objects.

The bound ``gh pr list`` call keeps its target, filters, limit and order and asks for
``id,url`` instead of ``commits``. A non-empty page adds ONE fixed GraphQL read of
``commits.totalCount`` by node id, sent to the host of the URLs GitHub returned.
A count that cannot be read exactly withholds the list; it never becomes 0.
"""
from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from ouroboros.tools import github
from tests.test_github_project_target import _context

BASE = "https://github.example/owner/selected"
FIELDS = "number,title,author,headRefName,baseRefName,createdAt,isDraft,reviewDecision,id,url"
COUNT_ARGV = ["gh", "api", "graphql", "--hostname", "github.example", "--input", "-"]
_REAL_RUN = subprocess.run


def _rows(*counts, base=BASE):
    return [{"number": 100 + i, "title": f"PR {i}", "author": {"login": "dev"}, "headRefName": f"feat/{i}",
             "baseRefName": "main", "createdAt": "2026-10-08T10:00:00Z", "isDraft": i == 0,
             "reviewDecision": "APPROVED" if i == 1 else None, "id": f"PR_node{i}", "url": f"{base}/pull/{100 + i}"}
            for i in range(len(counts))]


def _nodes(rows, counts):
    return [{"id": row["id"], "commits": {"totalCount": count}} for row, count in zip(rows, counts)]


@pytest.fixture
def gh(monkeypatch):
    world = SimpleNamespace(calls=[], rows=[], count_answer=None)
    monkeypatch.setattr(github, "github_token_from_env_or_settings", lambda: "fixture-token")

    def run(argv, **kwargs):
        world.calls.append((argv, kwargs))
        if argv[1:3] == ["pr", "list"]:
            return SimpleNamespace(returncode=0, stdout=json.dumps(world.rows), stderr="")
        assert argv == COUNT_ARGV, argv
        answer = world.count_answer
        if isinstance(answer, tuple):
            return SimpleNamespace(returncode=answer[0], stdout="", stderr=answer[1])
        return SimpleNamespace(returncode=0, stdout=answer if isinstance(answer, str) else json.dumps(answer),
                               stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    return world


def _answer(nodes):
    return {"data": {"nodes": nodes}}


@pytest.mark.parametrize("state", ["open", "closed", "merged", "all"])
def test_fifty_prs_keep_metadata_order_and_exact_counts_in_two_reads(tmp_path, gh, state):
    ctx, _cwd = _context(tmp_path, "queued")
    counts = [0, 1, 100, 101, 250] + [3] * 45
    gh.rows = _rows(*counts)
    gh.count_answer = _answer(list(reversed(_nodes(gh.rows, counts))))  # association by id, not order

    text = github._list_prs(ctx, state=state, limit=99)

    (list_argv, _), (count_argv, count_kw) = gh.calls
    assert list_argv == ["gh", "pr", "list", "--state", state, "--limit", "50", "--json", FIELDS]
    request = json.loads(count_kw["input"])
    assert request == {"query": github._PR_COMMIT_COUNTS_QUERY, "variables": {"ids": [r["id"] for r in gh.rows]}}
    assert "commits{totalCount}" in request["query"] and "nodes(ids:$ids)" in request["query"]
    lines = text.splitlines()
    assert lines[0] == f"**50 {state} PR(s):**"
    assert lines[2] == ("- **PR #100** [DRAFT] PR 0 (by @dev, feat/0→main, 0 commits, created 2026-10-08)")
    assert lines[3] == ("- **PR #101** [APPROVED] PR 1 (by @dev, feat/1→main, 1 commits, created 2026-10-08)")
    assert [line.split(", ")[2] for line in lines[4:7]] == ["100 commits", "101 commits", "250 commits"]
    assert [line.split("**")[1] for line in lines[2:]] == [f"PR #{100 + i}" for i in range(50)]


def test_an_empty_page_needs_no_count_read(tmp_path, gh):
    ctx, _cwd = _context(tmp_path, "system")
    assert github._list_prs(ctx, state="open") == "No open pull requests found."
    assert len(gh.calls) == 1


@pytest.mark.parametrize("kind", ["system", "queued", "room"])
@pytest.mark.parametrize("repo", ["", "github.example/owner/selected"])
def test_the_count_read_follows_the_returned_url_not_ambient_targets(tmp_path, gh, monkeypatch, kind, repo):
    ctx, expected_cwd = _context(tmp_path, kind)
    monkeypatch.setenv("GH_REPO", "unrelated/wrong-repo")
    monkeypatch.setenv("GH_HOST", "configured.example")
    gh.rows = _rows(2, 5)
    gh.count_answer = _answer(_nodes(gh.rows, [2, 5]))
    entry = next(item for item in github.get_tools() if item.name == "list_github_prs")

    text = entry.handler(ctx, repo=repo)

    assert "2 commits" in text and "5 commits" in text, text
    (list_argv, list_kw), (count_argv, count_kw) = gh.calls
    assert list_kw["cwd"] == str(expected_cwd)
    assert (list_argv[-2:] == ["--repo", repo]) if repo else "--repo" not in list_argv
    assert ("GH_REPO" in list_kw["env"]) is (kind == "system")
    # The generic transport: host from GitHub's own URL, never --repo, GH_REPO or GH_HOST;
    # it runs in the host's own repository directory (ctx.repo_dir).
    assert count_argv == COUNT_ARGV and count_kw["cwd"] == str(ctx.repo_dir)
    assert count_kw["env"]["GH_TOKEN"] == list_kw["env"]["GH_TOKEN"] == "fixture-token"


def test_an_unusable_host_repository_directory_is_a_visible_count_error(tmp_path, gh, monkeypatch):
    ctx, _cwd = _context(tmp_path, "queued")
    ctx.repo_dir = tmp_path / "missing-system"
    gh.rows = _rows(1)

    def run(argv, **kwargs):  # the real launcher refuses the missing cwd before gh could start
        if argv[1] == "api":
            assert kwargs["cwd"] == str(ctx.repo_dir)
            return _REAL_RUN(argv, **kwargs)
        return SimpleNamespace(returncode=0, stdout=json.dumps(gh.rows), stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    text = github._list_prs(ctx)

    assert text.startswith("⚠️ GH_ERROR") and "PR #" not in text, text


def _malformed(rows):
    good = _nodes(rows, [4, 7])
    return {
        "graphql-error": (1, "gh: Could not resolve to a node with the global id of 'PR_node1'\n"),
        "timeout": "timeout",
        "not-json": "not json",
        "null-data": {"data": None},
        "null-node": _answer([good[0], None]),
        "missing-node": _answer(good[:1]),
        "unknown-node": _answer([good[0], {"id": "PR_other", "commits": {"totalCount": 7}}]),
        "duplicate-node": _answer([good[0], good[0], good[1]]),
        "bool-count": _answer([good[0], {"id": "PR_node1", "commits": {"totalCount": True}}]),
        "negative-count": _answer([good[0], {"id": "PR_node1", "commits": {"totalCount": -1}}]),
        "float-count": _answer([good[0], {"id": "PR_node1", "commits": {"totalCount": 7.0}}]),
        "null-count": _answer([good[0], {"id": "PR_node1", "commits": {"totalCount": None}}]),
        "missing-commits": _answer([good[0], {"id": "PR_node1"}]),
    }


@pytest.mark.parametrize("case", list(_malformed(_rows(4, 7))))
def test_an_unreadable_count_withholds_the_list(tmp_path, gh, monkeypatch, case):
    ctx, _cwd = _context(tmp_path, "queued")
    gh.rows = _rows(4, 7)
    gh.count_answer = _malformed(gh.rows)[case]
    if case == "timeout":
        def run(argv, **kwargs):
            if argv[1] == "api":
                raise subprocess.TimeoutExpired(argv, 30)
            return SimpleNamespace(returncode=0, stdout=json.dumps(gh.rows), stderr="")
        monkeypatch.setattr(subprocess, "run", run)

    text = github._list_prs(ctx)

    assert text.startswith("⚠️ ") and "PR #" not in text and "0 commits" not in text, text


@pytest.mark.parametrize("case", ["no-url", "no-id", "foreign-repo", "foreign-host", "number-mismatch",
                                  "repeated-id", "not-a-row"])
def test_rows_without_one_consistent_identity_never_reach_the_count_read(tmp_path, gh, case):
    ctx, _cwd = _context(tmp_path, "queued")
    rows = _rows(1, 2)
    if case == "no-url":
        rows[1].pop("url")
    elif case == "no-id":
        rows[1]["id"] = None
    elif case == "foreign-repo":
        rows[1]["url"] = "https://github.example/owner/other/pull/101"
    elif case == "foreign-host":
        rows[1]["url"] = "https://github.com/owner/selected/pull/101"
    elif case == "number-mismatch":
        rows[1]["url"] = f"{BASE}/pull/999"
    elif case == "repeated-id":
        rows[1]["id"] = rows[0]["id"]
    else:
        rows[1] = "PR 101"
    gh.rows = rows

    text = github._list_prs(ctx)

    assert text.startswith("⚠️ TOOL_ERROR: commit counts of the listed PRs are unavailable"), text
    assert len(gh.calls) == 1


def test_registry_types_the_list_and_its_count_refusal(tmp_path, gh, monkeypatch):
    import ouroboros.safety as safety
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setattr(safety, "check_safety", lambda *_a, **_k: (True, ""))
    monkeypatch.setattr(github, "github_cli_configured", lambda: True)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    gh.rows = _rows(4, 7)
    gh.count_answer = _answer(_nodes(gh.rows, [4, 7]))
    listed = registry.execute_result("list_github_prs", {"state": "all", "limit": 50})
    assert listed.status == "ok" and "4 commits" in listed.text and "7 commits" in listed.text, listed

    gh.count_answer = _answer(_nodes(gh.rows, [4, True]))
    refused = registry.execute_result("list_github_prs", {"state": "all", "limit": 50})
    assert (refused.status, refused.code) == ("error", "TOOL_ERROR") and "PR #" not in refused.text, refused
