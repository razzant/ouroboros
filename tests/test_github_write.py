"""``github_write``: one receipt per write, read back from GitHub's answer, unknowns never resent.

Driven over a fake ``gh``; every method's receipt, the read-back mismatch, the 4xx not_applied answer,
the unknown outcome with its one lookup, and the expect preconditions.
"""
from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from ouroboros.tools import github, github_write
from tests.test_github_project_target import _context
from tests.test_github_read import API, BASE, SHA, _call, _comment, _issue, _pr

NOW = "2026-10-22T10:00:00+00:00"


@pytest.fixture
def gh(monkeypatch, tmp_path):
    """A recording fake gh; ``world.answer(parts, argv, kwargs)`` returns stdout, ``(code, stderr)`` or raises."""
    world = SimpleNamespace(calls=[], identity="ouroboros-agent", answer=lambda parts, argv, kwargs: "{}")
    monkeypatch.setattr(github, "github_token_from_env_or_settings", lambda: "ghp_fixture")
    monkeypatch.setattr(github_write, "utc_now_iso", lambda: NOW)

    def run(argv, **kwargs):
        world.calls.append((argv, kwargs))
        if argv[1:5] == ["repo", "view", "--json", "nameWithOwner"]:
            return SimpleNamespace(returncode=0, stdout=json.dumps({"nameWithOwner": "owner/selected"}), stderr="")
        if argv[1] == "api" and _call(argv).route == "user":
            return SimpleNamespace(returncode=0, stdout=json.dumps({"login": world.identity}), stderr="")
        answer = world.answer(_call(argv, kwargs) if argv[1] == "api" else SimpleNamespace(route=" ".join(argv[1:3])), argv, kwargs)
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, tuple):
            return SimpleNamespace(returncode=answer[0], stdout="", stderr=answer[1])
        return SimpleNamespace(returncode=0, stdout=answer, stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    world.ctx, world.cwd = _context(tmp_path, "queued")
    world.ctx._active_builtin_tool_result = None

    entry = next(item for item in github.get_tools() if item.name == "github_write")  # the registered, invocation-scoped handler

    def write(**args):
        world.ctx._active_builtin_tool_result = None
        text = entry.handler(world.ctx, **args)
        world.result = world.ctx._active_builtin_tool_result
        return text

    world.write = write
    world.sent = lambda: [(_call(argv).method, _call(argv).route, json.loads(kw["input"]) if kw.get("input") else None)
                          for argv, kw in world.calls if argv[1] == "api" and _call(argv).route != "graphql"]
    return world


def _receipt(text):
    line, _nl, body = text.partition("\n")
    return line, json.loads(body)


@pytest.mark.parametrize("retired,tool,blocked,allowed", [
    ("comment_on_issue", "github_write", {"method": " comment ", "target": 7, "body": "x"},
     {"method": " update ", "target": 7, "title": "New"}),
    ("close_github_issue", "github_write", {"method": " update ", "target": 7, "state": " closed "},
     {"method": " update ", "target": 7, "state": " open "}),
    ("list_github_issues", "github_read", {"method": " list ", "kind": " issue "},
     {"method": " list ", "kind": " label "}),
])
def test_registry_admission_and_dispatch_share_canonical_arguments(gh, monkeypatch, retired, tool, blocked, allowed):
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")
    gh.ctx.task_metadata["task_contract"] = {"disabled_tools": [retired]}
    registry = ToolRegistry(repo_dir=gh.ctx.repo_dir, drive_root=gh.ctx.drive_root)
    registry.set_context(gh.ctx)
    gh.answer = lambda p, argv, kw: "[]" if "/labels?" in p.route else json.dumps(_issue(7, title="New", state="open"))

    refused = registry.execute(tool, blocked)
    assert refused.startswith("⚠️ RESOURCE_CONSTRAINT_BLOCKED") and retired in refused
    assert not gh.calls
    admitted = registry.execute(tool, allowed)
    assert admitted.startswith(tool + " "), admitted
    assert len(gh.calls) == 1
    if tool == "github_write":
        assert gh.sent()[0][2] == ({"state": "open"} if "state" in allowed else {"title": "New"})


# ---- comment -----------------------------------------------------------------------------------------
def test_comment_on_an_issue_returns_the_shared_receipt_from_githubs_answer(gh):
    gh.answer = lambda p, argv, kw: json.dumps(_comment(501, body="hello", user={"login": "ouroboros-agent", "type": "User"}))
    line, receipt = _receipt(gh.write(method="comment", target=7, body="hello"))

    assert gh.sent() == [("POST", "repos/{owner}/{repo}/issues/7/comments", {"body": "hello"})]
    assert receipt == {"effect": "issue_comment.create", "target": "owner/selected#7", "url": f"{BASE}/issues/7#issuecomment-501",
                       "id": 501, "actor": "ouroboros-agent", "at": "2026-10-03T00:00:00Z", "outcome": "confirmed",
                       "readback": {"verified": True, "mismatch": [], "via": "response"}}
    assert line == f"github_write comment → owner/selected#7: confirmed {BASE}/issues/7#issuecomment-501 by @ouroboros-agent at 2026-10-03T00:00:00Z"
    assert (gh.result.status, gh.result.code, gh.result.meta["operation_outcome"]) == ("ok", "OK", "completed")


def test_comment_reply_to_answers_inline_in_the_review_thread(gh):
    gh.answer = lambda p, argv, kw: json.dumps({**_comment(9, body="fixed"), "pull_request_url": f"{API}/repos/owner/selected/pulls/7", "in_reply_to_id": 4})
    _line, receipt = _receipt(gh.write(method="comment", target=7, body="fixed", reply_to=4, repo="owner/selected"))
    assert gh.calls[0][0][:4] == ["gh", "api", "--method", "POST"] and gh.sent()[0][1] == "repos/owner/selected/pulls/7/comments/4/replies"
    assert receipt["effect"] == "review_comment.reply" and receipt["target"] == "owner/selected#7 reply to 4" and receipt["outcome"] == "confirmed"


def test_comment_on_a_discussion_resolves_its_node_then_sends_the_mutation(gh):
    def answer(p, argv, kw):
        if "discussion(number" in p.graphql:
            assert p.fields["number"] == 5 and p.fields["owner"] == "owner"
            return json.dumps({"data": {"repository": {"discussion": {"id": "D_5"}}}})
        assert "addDiscussionComment" in p.graphql and p.fields["discussionId"] == "D_5" and p.fields["replyToId"] == "DC_1"
        return json.dumps({"data": {"addDiscussionComment": {"comment": {"id": "DC_9", "databaseId": 9, "url": "u", "body": "thanks",
                                                                          "createdAt": "2026-10-22T09:00:00Z", "author": {"login": "ouroboros-agent"}}}}})
    gh.answer = answer
    _line, receipt = _receipt(gh.write(method="comment", kind="discussion", target=5, body="thanks", reply_to="DC_1"))
    assert receipt["effect"] == "discussion_comment.create" and receipt["target"] == "selected Project discussion #5 reply to DC_1"
    assert (receipt["id"], receipt["actor"], receipt["at"], receipt["outcome"]) == ("DC_9", "ouroboros-agent", "2026-10-22T09:00:00Z", "confirmed")
    gh.answer = lambda p, argv, kw: json.dumps({"data": {"repository": {"discussion": None}}})
    text = gh.write(method="comment", kind="discussion", target=6, body="x")
    assert "discussion #6 was not found" in text and gh.result.meta["operation_outcome"] == "completed_no_effect"


def test_readback_mismatch_is_loud_but_the_effect_stays_confirmed(gh):
    gh.answer = lambda p, argv, kw: json.dumps(_comment(3, body="edited by a bot"))
    line, receipt = _receipt(gh.write(method="comment", target=7, body="hello"))
    assert line.startswith("⚠️ GH_WRITE_READBACK_MISMATCH: GitHub's record differs from what was sent in body; github_write comment → owner/selected#7: confirmed")
    assert receipt["readback"] == {"verified": False, "mismatch": ["body"], "via": "response"} and gh.result.status == "ok"
    gh.answer = lambda p, argv, kw: json.dumps(_comment(3, body="hello", issue_url=f"{API}/repos/owner/selected/issues/8"))
    _line, receipt = _receipt(gh.write(method="comment", target=7, body="hello"))
    assert receipt["readback"]["mismatch"] == ["target"]


# ---- create_issue ------------------------------------------------------------------------------------
def test_create_issue_sends_title_body_and_labels_in_one_request(gh):
    gh.answer = lambda p, argv, kw: json.dumps(_issue(12, title="T", body="B", labels=[{"name": "bug"}, {"name": "help"}], user={"login": "me"}))
    line, receipt = _receipt(gh.write(method="create_issue", title="T", body="B", labels=["bug", "help"]))
    assert gh.sent() == [("POST", "repos/{owner}/{repo}/issues", {"title": "T", "body": "B", "labels": ["bug", "help"]})]
    assert receipt["effect"] == "issue.create" and receipt["target"] == "owner/selected#12" and receipt["url"] == f"{BASE}/issues/12"
    assert receipt["readback"]["verified"] and line.startswith("github_write create_issue → owner/selected#12: confirmed")
    gh.answer = lambda p, argv, kw: json.dumps(_issue(13, title="T", body="B", labels=[{"name": "bug"}]))
    _line, receipt = _receipt(gh.write(method="create_issue", title="T", body="B", labels="bug, help"))
    assert receipt["readback"] == {"verified": False, "mismatch": ["labels"], "via": "response"}


# ---- update ------------------------------------------------------------------------------------------
def test_omitted_labels_preserve_them_and_explicit_empty_labels_clear_them(gh):
    assert "default" not in github_write.SCHEMA["parameters"]["properties"]["labels"]
    gh.answer = lambda p, argv, kw: json.dumps(_issue(7, title="New"))
    gh.write(method="update", target=7, title="New")
    assert gh.sent()[-1][2] == {"title": "New"}
    gh.write(method="update", target=7, title="New", labels=[])
    assert gh.sent()[-1][2] == {"title": "New", "labels": []}


@pytest.mark.parametrize("fields,actor", [({"state": "closed"}, "ouroboros-agent"), ({"title": "New"}, None)])
def test_update_actor_is_the_closer_or_unknown_never_the_issue_author(gh, fields, actor):
    gh.answer = lambda p, argv, kw: json.dumps(_issue(7, **fields, user={"login": "alice"}, closed_by={"login": "ouroboros-agent"}))
    _line, receipt = _receipt(gh.write(method="update", target=7, **fields))
    assert receipt["actor"] == actor and len(gh.calls) == 1


def test_update_closes_with_a_reason_and_replaces_labels_in_one_patch(gh):
    gh.answer = lambda p, argv, kw: json.dumps(_issue(7, state="closed", state_reason="not_planned", labels=[{"name": "wontfix"}],
                                                       updated_at="2026-10-22T09:30:00Z", user={"login": "dev"}))
    line, receipt = _receipt(gh.write(method="update", target=7, state="closed", state_reason="not_planned", labels=["wontfix"]))
    assert gh.sent() == [("PATCH", "repos/{owner}/{repo}/issues/7", {"state": "closed", "state_reason": "not_planned", "labels": ["wontfix"]})]
    assert receipt["effect"] == "issue.update" and receipt["at"] == "2026-10-22T09:30:00Z" and receipt["readback"]["verified"]
    assert line.startswith("github_write update → owner/selected#7: confirmed")
    gh.answer = lambda p, argv, kw: json.dumps(_issue(7, state="open", title="New"))
    _line, receipt = _receipt(gh.write(method="update", target=7, title="New", state="closed"))
    assert receipt["readback"] == {"verified": False, "mismatch": ["state"], "via": "response"}
    assert gh.write(method="update", target=7).startswith("⚠️ TOOL_ARG_ERROR: update changes at least one of")
    assert gh.write(method="update", target=7, state="resolved").startswith("⚠️ TOOL_ARG_ERROR: update state is 'open' or 'closed'")
    assert gh.write(method="update", target=7, state_reason="tired").startswith("⚠️ TOOL_ARG_ERROR: state_reason must be one of")


# ---- review ------------------------------------------------------------------------------------------
def test_review_submits_the_event_body_and_inline_comments(gh):
    gh.answer = lambda p, argv, kw: json.dumps({"id": 77, "user": {"login": "me"}, "state": "CHANGES_REQUESTED", "body": "please",
                                                "submitted_at": "2026-10-22T09:40:00Z", "html_url": f"{BASE}/pull/7#pullrequestreview-77"})
    _line, receipt = _receipt(gh.write(method="review", target=7, event="REQUEST_CHANGES", body="please",
                                       comments=[{"path": "a.py", "line": 3, "body": "nit", "side": "RIGHT", "extra": "dropped"}]))
    assert gh.sent() == [("POST", "repos/{owner}/{repo}/pulls/7/reviews", {"event": "REQUEST_CHANGES", "body": "please",
                                                                         "comments": [{"path": "a.py", "body": "nit", "line": 3, "side": "RIGHT"}]})]
    assert receipt["effect"] == "pr_review.submit" and receipt["id"] == 77 and receipt["at"] == "2026-10-22T09:40:00Z" and receipt["readback"]["verified"]
    assert gh.write(method="review", target=7, event="COMMENT").startswith("⚠️ TOOL_ARG_ERROR: a REQUEST_CHANGES or COMMENT review needs a body")
    assert gh.write(method="review", target=7, event="LGTM").startswith("⚠️ TOOL_ARG_ERROR: event must be")
    assert gh.write(method="review", target=7, event="APPROVE", comments=[{"path": "a.py"}]).startswith("⚠️ TOOL_ARG_ERROR: comments is a list")


# ---- resolve_thread ------------------------------------------------------------------------------------
@pytest.mark.parametrize("state,mutation,resolved", [("resolved", "resolveReviewThread", True), ("unresolved", "unresolveReviewThread", False)])
def test_resolve_thread_is_a_graphql_mutation_read_back_by_its_state(gh, state, mutation, resolved):
    gh.answer = lambda p, argv, kw: json.dumps({"data": {mutation: {"thread": {"id": "PRRT_1", "isResolved": resolved}}}})
    line, receipt = _receipt(gh.write(method="resolve_thread", target="PRRT_1", state=state, repo="owner/selected"))
    parts = _call(*gh.calls[0])
    assert mutation in parts.graphql and parts.fields == {"id": "PRRT_1"}
    assert receipt["effect"] == f"review_thread.{mutation.removesuffix('ReviewThread')}" and receipt["id"] == "PRRT_1"
    assert receipt["readback"]["verified"] and receipt["at"] == NOW and receipt["at_source"] == "local"
    assert line == f"github_write resolve_thread → owner/selected thread PRRT_1: confirmed at {NOW}"
    assert gh.write(method="resolve_thread", target=7).startswith("⚠️ TOOL_ARG_ERROR: target must be the review thread node id")


# ---- api ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("verb,route,actor", [("PATCH", "issues/7", None), ("PATCH", "issues/comments/3", None),
                                           ("POST", "issues/7/comments", "alice"), ("POST", "issues/7/assignees", None)])
def test_api_attributes_created_objects_but_not_modified_targets(gh, verb, route, actor):
    gh.answer = lambda p, argv, kw: json.dumps(_comment(3))
    _line, receipt = _receipt(gh.write(method="api", http_method=verb, route="repos/{owner}/{repo}/" + route))
    assert receipt["actor"] == actor and "at_source" not in receipt


def test_api_marks_a_locally_stamped_time(gh):
    gh.answer = lambda p, argv, kw: "{}"
    _line, receipt = _receipt(gh.write(method="api", http_method="DELETE", route="repos/{owner}/{repo}/issues/comments/3"))
    assert receipt["at"] == NOW and receipt["at_source"] == "local" and receipt["actor"] is None


def test_graphql_api_alias_cannot_attribute_an_edit_to_the_object_author(gh):
    gh.answer = lambda p, argv, kw: json.dumps({"data": {"createIssue": {"issue": {
        "id": "I_7", "author": {"login": "issue-creator"}, "updatedAt": NOW}}}})
    _line, receipt = _receipt(gh.write(method="api", graphql='mutation { createIssue: updateIssue(input:{id:"I_7",title:"New"}) '
                                      '{ issue { id author { login } updatedAt } } }'))
    assert receipt["actor"] is None and receipt["at"] == NOW


def test_resolve_receipt_uses_githubs_resolved_by(gh):
    def answer(p, argv, kw):
        assert "resolvedBy{login}" in p.graphql
        return json.dumps({"data": {"resolveReviewThread": {"thread": {"id": "PRRT_1", "isResolved": True,
                                                                     "resolvedBy": {"login": "ouroboros-agent"}}}}})
    gh.answer = answer
    _line, receipt = _receipt(gh.write(method="resolve_thread", target="PRRT_1"))
    assert receipt["actor"] == "ouroboros-agent" and receipt["at_source"] == "local"


@pytest.mark.parametrize("document", [
    'query{search(query:"fix mutation bug",type:ISSUE,first:1){issueCount}}',
    '# mutation Fake { ignored }\nquery { viewer { login } }',
    'query{search(query:"""fix mutation bug""",type:ISSUE,first:1){issueCount}}',
])
def test_graphql_queries_with_mutation_text_are_refused_by_write(gh, document):
    text = gh.write(method="api", graphql=document)
    assert "github_read(method='api'" in text and not gh.calls
    assert gh.result.code == "TOOL_ARG_ERROR"


def test_graphql_commented_mutation_is_admitted_by_write(gh):
    gh.answer = lambda p, argv, kw: '{"data":{}}'
    _line, receipt = _receipt(gh.write(method="api", graphql='# comment\nmutation M { addReaction(input:{}) { clientMutationId } }'))
    assert receipt["outcome"] == "confirmed" and len(gh.calls) == 1


def test_api_sends_rest_mutations_and_graphql_mutations_and_names_the_effect(gh):
    gh.answer = lambda p, argv, kw: json.dumps([{"name": "bug", "id": 1}])
    line, receipt = _receipt(gh.write(method="api", http_method="PUT", route="repos/owner/selected/issues/7/labels", input={"labels": ["bug"]}, repo="owner/selected"))
    assert gh.sent() == [("PUT", "repos/owner/selected/issues/7/labels", {"labels": ["bug"]})]
    assert receipt["effect"] == "PUT repos/{owner}/{repo}/issues/{n}/labels" and receipt["outcome"] == "confirmed" and "readback" not in receipt
    assert line == "github_write api → owner/selected repos/owner/selected/issues/7/labels: confirmed at " + NOW
    gh.answer = lambda p, argv, kw: json.dumps({"data": {"addReaction": {"reaction": {"id": "R_1", "content": "THUMBS_UP"}}}})
    _line, receipt = _receipt(gh.write(method="api", graphql="mutation($id:ID!){addReaction(input:{subjectId:$id,content:THUMBS_UP}){reaction{id content}}}",
                                       variables={"id": "IC_1"}))
    assert receipt["effect"] == "mutation addReaction" and receipt["id"] == "R_1"
    assert _call(*gh.calls[-1]).fields == {"id": "IC_1"}
    text = gh.write(method="api", graphql="query { viewer { login } }")
    assert "sends mutations; a query goes through github_read(method='api'" in text and gh.result.meta["operation_outcome"] == "completed_no_effect"
    assert gh.write(method="api", http_method="GET", route="user").startswith("⚠️ TOOL_ARG_ERROR: http_method must be one of")
    assert gh.write(method="api").startswith("⚠️ TOOL_ARG_ERROR: api takes exactly one of")
    assert len(gh.calls) == 2


# ---- expect --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("args", [
    {"method": "comment", "kind": "discussion", "target": 7, "body": "x"},
    {"method": "resolve_thread", "target": "PRRT_1"},
    {"method": "api", "route": "repos/{owner}/{repo}/issues/7", "http_method": "PATCH"},
    {"method": "create_issue", "title": "T"},
])
@pytest.mark.parametrize("value", ["open", "", None])
def test_unsupported_expect_is_typed_and_never_sent(gh, args, value):
    text = gh.write(**args, expect={"state": value})
    assert "cannot check it" in text and not gh.calls
    assert gh.result.code == "TOOL_ARG_ERROR" and gh.result.meta["operation_outcome"] == "completed_no_effect"


@pytest.mark.parametrize("method", github_write.METHODS)
@pytest.mark.parametrize("value", [SHA, "", None])
def test_unknown_expect_key_is_never_ignored(gh, method, value):
    gh.write(method=method, target=7, body="x", title="T", expect={"expected_head_sha": value})
    assert gh.result.code == "TOOL_ARG_ERROR" and "unknown expect key" in gh.result.text and not gh.calls


@pytest.mark.parametrize("blank", ["", None])
def test_blank_head_expect_checks_issue_state_before_commenting(gh, blank):
    gh.answer = lambda p, argv, kw: json.dumps(_issue(7)) if p.method == "GET" else json.dumps(_comment(1, body="x"))
    _line, receipt = _receipt(gh.write(method="comment", target=7, body="x", expect={"head_sha": blank, "state": "open"}))
    assert receipt["outcome"] == "confirmed"
    assert gh.sent() == [("GET", "repos/{owner}/{repo}/issues/7", None),
                         ("POST", "repos/{owner}/{repo}/issues/7/comments", {"body": "x"})]


@pytest.mark.parametrize("state", ["open", "closed"])
def test_comment_state_expectation_still_checks_before_posting(gh, state):
    gh.answer = lambda p, argv, kw: json.dumps(_issue(7)) if p.method == "GET" else json.dumps(_comment(1, body="x"))
    text = gh.write(method="comment", target=7, body="x", expect={"state": state})
    assert gh.sent()[0] == ("GET", "repos/{owner}/{repo}/issues/7", None)
    if state == "open":
        assert _receipt(text)[1]["outcome"] == "confirmed" and [s[0] for s in gh.sent()] == ["GET", "POST"]
    else:
        assert "GH_EXPECTATION_FAILED" in text and "state is 'open', expected 'closed'" in text
        assert gh.result.meta["operation_outcome"] == "completed_no_effect" and len(gh.sent()) == 1


@pytest.mark.parametrize("blank", ["", None])
def test_all_blank_expect_values_need_no_precondition_read(gh, blank):
    gh.answer = lambda p, argv, kw: json.dumps(_comment(1, body="x"))
    _line, receipt = _receipt(gh.write(method="comment", target=7, body="x",
                                     expect={"head_sha": blank, "state": blank, "updated_at": blank}))
    assert receipt["outcome"] == "confirmed"
    assert gh.sent() == [("POST", "repos/{owner}/{repo}/issues/7/comments", {"body": "x"})]


@pytest.mark.parametrize("blank", ["", None])
def test_update_ignores_blank_timestamp_beside_real_state_expectation(gh, blank):
    gh.answer = lambda p, argv, kw: json.dumps(_issue(7, title="New"))
    _line, receipt = _receipt(gh.write(method="update", target=7, title="New", expect={"state": "open", "updated_at": blank}))
    assert receipt["outcome"] == "confirmed" and receipt["readback"]["verified"] is True
    assert gh.sent() == [("GET", "repos/{owner}/{repo}/issues/7", None),
                         ("PATCH", "repos/{owner}/{repo}/issues/7", {"title": "New"})]


@pytest.mark.parametrize("current", ["open", "closed"])
def test_api_checks_an_explicit_issue_target_before_mutating(gh, current):
    gh.answer = lambda p, argv, kw: json.dumps(_issue(7, state=current))
    text = gh.write(method="api", target=7, kind="issue", http_method="PATCH", route="repos/{owner}/{repo}/issues/7",
                    input={"title": "T"}, expect={"state": "open"})
    assert gh.sent()[0][:2] == ("GET", "repos/{owner}/{repo}/issues/7")
    if current == "open":
        assert _receipt(text)[1]["outcome"] == "confirmed" and [s[0] for s in gh.sent()] == ["GET", "PATCH"]
    else:
        assert "GH_EXPECTATION_FAILED" in text and len(gh.sent()) == 1


def test_other_misplaced_optional_arguments_still_do_not_refuse(gh):
    gh.answer = lambda p, argv, kw: json.dumps(_comment(1, body="x"))
    _line, receipt = _receipt(gh.write(method="comment", target=7, body="x", state="closed", title="unused", expect={}))
    assert receipt["outcome"] == "confirmed" and gh.sent()[0][2] == {"body": "x"}


def test_expect_refuses_before_sending_when_the_target_moved(gh):
    gh.answer = lambda p, argv, kw: json.dumps(_pr(7, updated_at="2026-10-22T08:00:00Z"))
    text = gh.write(method="comment", target=7, body="x", expect={"head_sha": "d" * 40, "state": "open"})
    assert text.startswith("⚠️ GH_EXPECTATION_FAILED: #7: head_sha is " + repr(SHA) + ", expected " + repr("d" * 40) + ". Nothing was sent")
    assert gh.result.meta["operation_outcome"] == "completed_no_effect" and gh.sent() == [("GET", "repos/{owner}/{repo}/pulls/7", None)]
    gh.calls.clear()
    gh.answer = lambda p, argv, kw: json.dumps(_pr(7, updated_at="2026-10-22T08:00:00Z")) if p.method == "GET" else json.dumps(_comment(1, body="x"))
    _line, receipt = _receipt(gh.write(method="comment", target=7, body="x", expect={"head_sha": SHA, "state": "OPEN", "updated_at": "2026-10-22T08:00:00Z"}))
    assert receipt["outcome"] == "confirmed" and [s[0] for s in gh.sent()] == ["GET", "POST"]
    gh.calls.clear()
    gh.answer = lambda p, argv, kw: json.dumps(_issue(7, state="closed"))
    text = gh.write(method="update", target=7, title="t", expect={"state": "open"})
    assert "state is 'closed', expected 'open'" in text and gh.sent() == [("GET", "repos/{owner}/{repo}/issues/7", None)]


@pytest.mark.parametrize("identifier,args,reply,code,outcome", [
    ("GH_WRITE_UNKNOWN", {"method": "comment", "target": 7, "body": "x"},
     subprocess.TimeoutExpired(["gh"], 60), "TOOL_ERROR", "unknown"),
    ("GH_WRITE_NOT_APPLIED", {"method": "create_issue", "title": "T"},
     (1, "gh: Validation Failed (HTTP 422)"), "TOOL_ERROR", "completed_no_effect"),
    ("GH_EXPECTATION_FAILED", {"method": "comment", "target": 7, "body": "x", "expect": {"state": "closed"}},
     json.dumps(_issue(7)), "TOOL_ERROR", "completed_no_effect"),
    ("GH_API_READ_ONLY", {"method": "api", "route": "POST repos/{owner}/{repo}/issues"},
     "{}", "TOOL_ARG_ERROR", "completed_no_effect"),
    ("GH_WRITE_READBACK_MISMATCH", {"method": "comment", "target": 7, "body": "x"},
     json.dumps(_comment(3, body="edited by a bot")), "OK", "completed"),
])
def test_github_producers_keep_their_typed_loop_outcomes(gh, monkeypatch, identifier, args, reply, code, outcome):
    from ouroboros.loop_tool_execution import _typed_execution_failure, _typed_result_metadata
    from ouroboros.tools.registry import ToolRegistry
    from tests.test_tool_classification_differential import CURRENT_PRODUCER_CONTRACTS

    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")
    monkeypatch.setattr(github, "github_cli_configured", lambda: True)
    gh.answer = lambda p, argv, kw: "[]" if identifier == "GH_WRITE_UNKNOWN" and p.method == "GET" else reply
    registry = ToolRegistry(repo_dir=gh.ctx.repo_dir, drive_root=gh.ctx.drive_root)
    registry.set_context(gh.ctx)
    tool = "github_read" if identifier == "GH_API_READ_ONLY" else "github_write"
    result = registry.execute_result(tool, args)
    assert result.text.startswith(f"⚠️ {identifier}:") and result.code == code
    is_error = _typed_execution_failure(True, result)
    meta = _typed_result_metadata(tool, result.text, is_error, result)
    assert (is_error, meta["status"]) == CURRENT_PRODUCER_CONTRACTS[f"native:{code}:{identifier}"]
    assert meta["tool_result_meta"]["operation_outcome"] == outcome
    sends = [s for s in gh.sent() if s[0] == "POST"]
    assert len(sends) == (0 if identifier in ("GH_API_READ_ONLY", "GH_EXPECTATION_FAILED") else 1)
    if identifier == "GH_WRITE_UNKNOWN":
        assert is_error and "do not resend" in result.text
    if identifier == "GH_WRITE_READBACK_MISMATCH":
        receipt = json.loads(result.text.split("\n", 1)[1])
        assert receipt["outcome"] == "confirmed" and receipt["readback"]["verified"] is False


# ---- outcomes: not_applied and unknown -------------------------------------------------------------------
@pytest.mark.parametrize("method,args,row", [
    ("comment", {"target": 7, "body": "Thanks!"}, _comment(3, body="Thanks!")),
    ("create_issue", {"title": "Same", "body": "Same"}, _issue(4, title="Same", body="Same")),
    ("review", {"target": 7, "event": "APPROVE"}, {"id": 5, "body": "", "state": "APPROVED", "submitted_at": "2026-10-22T09:59:30Z"}),
])
@pytest.mark.parametrize("author", ["someone-else", "ouroboros-agent"])
def test_unanswered_create_confirms_only_the_acting_authors_row(gh, method, args, row, author):
    gh.answer = lambda p, argv, kw: subprocess.TimeoutExpired(["gh"], 60) if p.method == "POST" else json.dumps([
        {**row, "user": {"login": author}}])
    _line, receipt = _receipt(gh.write(method=method, repo="github.example/owner/selected", **args))
    assert receipt["outcome"] == ("confirmed" if author == gh.identity else "unknown")
    if author != gh.identity:
        assert receipt["observed_match"]["actor"] == author
        assert gh.result.meta["operation_outcome"] == "unknown"
    assert [s[0] for s in gh.sent()] == ["POST", "GET", "GET"]
    assert gh.sent()[-1][1] == "user" and all(_call(argv).hostname == "github.example" for argv, _kw in gh.calls)
    assert len({kw["env"]["GH_TOKEN"] for _argv, kw in gh.calls}) == 1


def test_matching_update_without_actor_evidence_stays_unknown(gh):
    gh.answer = lambda p, argv, kw: subprocess.TimeoutExpired(["gh"], 60) if p.method == "PATCH" else json.dumps(
        _issue(7, title="New", user={"login": "ouroboros-agent"}))
    _line, receipt = _receipt(gh.write(method="update", target=7, title="New"))
    assert receipt["outcome"] == "unknown" and receipt["observed_match"]["actor"] is None


def test_matching_comment_with_unavailable_identity_stays_unknown(gh):
    gh.identity = None
    gh.answer = lambda p, argv, kw: subprocess.TimeoutExpired(["gh"], 60) if p.method == "POST" else json.dumps([_comment(3, body="Thanks!")])
    _line, receipt = _receipt(gh.write(method="comment", target=7, body="Thanks!"))
    assert receipt["outcome"] == "unknown" and receipt["observed_match"]["acting_login"] is None


def test_a_github_4xx_is_not_applied_and_proves_no_effect(gh):
    gh.answer = lambda p, argv, kw: (1, "gh: Validation Failed (HTTP 422)")
    line, receipt = _receipt(gh.write(method="create_issue", title="T"))
    assert line.startswith("⚠️ GH_WRITE_NOT_APPLIED: GitHub refused (HTTP 422); nothing changed. github_write create_issue → selected Project new issue 'T': not_applied")
    assert receipt["outcome"] == "not_applied" and receipt["http_status"] == 422 and receipt["error"].startswith("⚠️ GH_ERROR")
    assert (gh.result.status, gh.result.code, gh.result.meta["operation_outcome"]) == ("error", "TOOL_ERROR", "completed_no_effect")
    assert len(gh.calls) == 1  # no lookup for a definite refusal


@pytest.mark.parametrize("failure", ["timeout", "502", "reset"])
def test_an_unanswered_send_is_looked_up_once_and_confirmed_when_found(gh, failure):
    answers = {"timeout": subprocess.TimeoutExpired(["gh"], 60), "502": (1, "HTTP 502: Bad Gateway (https://api.github.com)"),
               "reset": (1, "gh: unexpected EOF")}
    gh.answer = lambda p, argv, kw: answers[failure] if p.method == "POST" else json.dumps([_comment(2, body="other"), _comment(3, body="hello", user={"login": "ouroboros-agent"})])
    line, receipt = _receipt(gh.write(method="comment", target=7, body="hello"))
    sent = gh.sent()
    assert sent[0][:2] == ("POST", "repos/{owner}/{repo}/issues/7/comments")
    assert sent[1] == ("GET", "repos/{owner}/{repo}/issues/7/comments?since=2026-10-22T09%3A58%3A00Z&per_page=100", None)
    assert receipt["outcome"] == "confirmed" and receipt["id"] == 3 and receipt["actor"] == "ouroboros-agent"
    assert receipt["readback"] == {"verified": True, "mismatch": [], "via": "lookup"} and gh.result.meta["operation_outcome"] == "completed"
    assert line.startswith("github_write comment → owner/selected#7: confirmed")


@pytest.mark.parametrize("lookup", ["absent", "failed"])
def test_an_unanswered_send_not_found_stays_unknown_and_is_never_resent(gh, lookup):
    gh.answer = lambda p, argv, kw: subprocess.TimeoutExpired(["gh"], 60) if p.method == "POST" else (json.dumps([_comment(2, body="other")]) if lookup == "absent" else (1, "HTTP 503"))
    line, receipt = _receipt(gh.write(method="comment", target=7, body="hello"))
    assert line.startswith("⚠️ GH_WRITE_UNKNOWN: the request was sent and no answer proved its outcome; do not resend. Reconcile with "
                           "github_read(method='list', kind='comment', target=7, since='2026-10-22T09:58:00Z') before any further write. "
                           "github_write comment → selected Project#7: unknown")
    assert receipt["outcome"] == "unknown" and receipt["failure"] == "timeout" and receipt["error"].startswith("⚠️ GH_TIMEOUT")
    assert receipt["lookup"].startswith("github_read(method='list', kind='comment', target=7")
    assert (gh.result.status, gh.result.code, gh.result.meta["operation_outcome"]) == ("error", "TOOL_ERROR", "unknown")
    assert [s[0] for s in gh.sent()] == ["POST", "GET"]  # exactly one lookup, no second POST


def test_each_method_has_its_own_lookup_after_an_unanswered_send(gh):
    cases = {
        "create_issue": (dict(title="T", body="B"), "repos/{owner}/{repo}/issues?state=all&since=", [_issue(4, title="T", body="B", user={"login": "ouroboros-agent"})], "owner/selected#4"),
        "update": (dict(target=7, state="closed"), "repos/{owner}/{repo}/issues/7", _issue(7, state="closed", closed_by={"login": "ouroboros-agent"}), "selected Project#7"),
        "review": (dict(target=7, event="APPROVE", body="ok"), "repos/{owner}/{repo}/pulls/7/reviews?per_page=100",
                   [{"id": 1, "body": "ok", "state": "APPROVED", "submitted_at": "2026-10-22T09:59:30Z", "user": {"login": "ouroboros-agent"}}], "selected Project#7"),
    }
    for method, (args, route, found, target) in cases.items():
        gh.calls.clear()
        gh.answer = lambda p, argv, kw, found=found: subprocess.TimeoutExpired(["gh"], 60) if p.method != "GET" else json.dumps(found)
        _line, receipt = _receipt(gh.write(method=method, **args))
        assert receipt["outcome"] == "confirmed" and receipt["readback"]["via"] == "lookup" and receipt["target"] == target, (method, receipt)
        assert gh.sent()[1][1].startswith(route), (method, gh.sent())
    gh.calls.clear()
    gh.answer = lambda p, argv, kw: subprocess.TimeoutExpired(["gh"], 60) if "mutation" in p.graphql else json.dumps({"data": {"node": {"id": "PRRT_1", "isResolved": True, "resolvedBy": {"login": "ouroboros-agent"}}}})
    _line, receipt = _receipt(gh.write(method="resolve_thread", target="PRRT_1"))
    assert receipt["outcome"] == "confirmed" and receipt["readback"]["via"] == "lookup" and len(gh.calls) == 3
    gh.calls.clear()
    gh.answer = lambda p, argv, kw: subprocess.TimeoutExpired(["gh"], 60)
    _line, receipt = _receipt(gh.write(method="api", http_method="DELETE", route="repos/{owner}/{repo}/issues/comments/5"))
    assert receipt["outcome"] == "unknown" and receipt["effect"] == "DELETE repos/{owner}/{repo}/issues/comments/{n}" and len(gh.calls) == 1


# ---- argument refusals and registration ------------------------------------------------------------------
@pytest.mark.parametrize("args,fragment", [
    ({"method": "comment", "target": 0, "body": "x"}, "target must be the positive issue, pull request or discussion number"),
    ({"method": "comment", "target": 7, "body": "  "}, "comment body cannot be empty"),
    ({"method": "create_issue", "title": " "}, "issue title cannot be empty"),
    ({"method": "update", "target": -1, "title": "t"}, "target must be the positive issue or pull request number"),
    ({"method": "review", "target": 0, "event": "APPROVE"}, "target must be the positive pull request number"),
    ({"method": "merge"}, "method must be one of"),
    ({"method": "comment", "target": 7, "body": "x", "repo": None}, "GH_TARGET_INVALID"),
    ({"method": "comment", "target": 7, "body": "x", "expect": "soon"}, "expect must be an object"),
])
def test_argument_refusals_are_typed_and_send_nothing(gh, args, fragment):
    text = gh.write(**args)
    assert fragment in text and not gh.calls
    assert (gh.result.status, gh.result.code) == ("error", "TOOL_ARG_ERROR") and gh.result.meta["operation_outcome"] == "completed_no_effect"


def test_github_write_is_registered_as_a_checked_world_mutation(gh, monkeypatch):
    from ouroboros import tool_capabilities
    from ouroboros.consciousness_authority import OBSERVE_DISABLED
    from ouroboros.safety import POLICY_CHECK, TOOL_POLICY
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.registry_guards import _GITHUB_TOKEN_TOOLS

    entry = next(item for item in github.get_tools() if item.name == "github_write")
    assert TOOL_POLICY["github_write"] == POLICY_CHECK and "github_write" in _GITHUB_TOKEN_TOOLS and not entry.mutates_worktree
    assert "github_write" in tool_capabilities.OBSERVE_WORLD_MUTATION_TOOLS and "github_write" in OBSERVE_DISABLED
    assert entry.schema["parameters"]["required"] == ["method"] and entry.schema["parameters"]["properties"]["method"]["enum"] == list(github_write.METHODS)
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")
    monkeypatch.setattr(github, "github_cli_configured", lambda: True)
    gh.answer = lambda p, argv, kw: json.dumps(_comment(1, body="hi"))
    registry = ToolRegistry(repo_dir=gh.ctx.repo_dir, drive_root=gh.ctx.drive_root)
    registry.set_context(gh.ctx)
    result = registry.execute_result("github_write", {"method": "comment", "target": 7, "body": "hi"})
    assert result.status == "ok" and result.text.startswith("github_write comment → owner/selected#7: confirmed"), result
    rejected = registry.execute_result("github_write", {"method": "comment", "number": 7, "body": "hi"})
    assert rejected.status != "ok" and "number" in rejected.text and len(gh.calls) == 1
