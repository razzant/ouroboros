"""The methods of ``github_write``: comment, create_issue, create_pr, update, review, resolve_thread and api.

Not a tool module: ``github.py`` registers the tool and owns the transport this module calls.
Every write returns ONE receipt ``{effect, target, url, id, actor, at, outcome}`` as JSON behind one
human line; ``actor`` is GitHub's acting identity or null, and local times are labelled. The response is read back against what was
sent where that is cheap. A send whose answer never came (timeout, 5xx, reset) is looked up once by
its exact body, target and acting identity; found → confirmed, otherwise the outcome stays ``unknown`` and the text
says so — nothing is resent. A direct API 4xx is ``not_applied``; CLI PR writes need separate readback even after a 4xx.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from typing import Any, Dict, List, Optional

from ouroboros.tools.github import GhResult, _canonical_github_args, _gh_api, _gh_graphql, _graphql_operations, _no_launch_yet, _refuse, _split_repo
from ouroboros.tools.github_read import _repo_of
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.tool_result import ToolResult, _publish_tool_result
from ouroboros.utils import utc_now_iso

METHODS = ("comment", "create_issue", "create_pr", "update", "review", "resolve_thread", "api")
REVIEW_EVENTS = {"APPROVE": "APPROVED", "REQUEST_CHANGES": "CHANGES_REQUESTED", "COMMENT": "COMMENTED"}
STATE_REASONS = ("completed", "not_planned", "reopened", "duplicate")
HTTP_METHODS = ("POST", "PATCH", "PUT", "DELETE")
_LOOKUP_SLACK_SEC = 120  # The lookup after an unknown send reads what changed shortly before the send.
_RESPONSE_ACTOR = object()
_NUMBER_SEGMENT_RE = re.compile(r"/\d+(?=/|$)")
_MUTATION_NAME_RE = re.compile(r"mutation\s+([A-Za-z_]\w*)|mutation\s*(?:\([^)]*\))?\s*\{\s*([A-Za-z_]\w*)")
_GQL = {
    "discussion_id": "query($owner:String!,$name:String!,$number:Int!){repository(owner:$owner,name:$name){discussion(number:$number){id}}}",
    "add_discussion_comment": "mutation($discussionId:ID!,$body:String!,$replyToId:ID){addDiscussionComment(input:{discussionId:$discussionId,"
                              "body:$body,replyToId:$replyToId}){comment{id databaseId url body createdAt author{login} discussion{number}}}}",
    "resolve": "mutation($id:ID!){resolveReviewThread(input:{threadId:$id}){thread{id isResolved resolvedBy{login}}}}",
    "unresolve": "mutation($id:ID!){unresolveReviewThread(input:{threadId:$id}){thread{id isResolved resolvedBy{login}}}}",
    "thread": "query($id:ID!){node(id:$id){... on PullRequestReviewThread{id isResolved resolvedBy{login}}}}",
}


class _Write:
    def __init__(self, ctx: ToolContext, method: str, repo: str, **args: Any):
        self.ctx, self.method, self.repo, self.args = ctx, method, repo, args
        self.sent_at = ""
        self.observed_match = None

    def refuse(self, text: str, code: str = "TOOL_ARG_ERROR", *, no_effect: Optional[bool] = None) -> str:
        """A typed refusal. Unless stated, an argument refusal attests no effect only while this public invocation
        has launched no gh; a TOOL_ERROR comes after a transport answer and attests nothing."""
        if no_effect is None:
            no_effect = code == "TOOL_ARG_ERROR" and _no_launch_yet()
        return _refuse(self.ctx, text, code, no_effect=no_effect)

    def number(self) -> Optional[int]:
        target = self.args.get("target")
        if isinstance(target, bool):
            return None
        if isinstance(target, int):
            return target if target > 0 else None
        text = str(target or "").strip()
        return int(text) if text.isdigit() and int(text) > 0 else None

    def slug(self, url: object = "") -> str:
        parts = _split_repo(self.repo) if self.repo else None
        return _repo_of(url) or (f"{parts[1]}/{parts[2]}" if parts else "") or "selected Project"

    def labels(self) -> List[str]:
        raw = self.args.get("labels")
        if isinstance(raw, str):
            return [l.strip() for l in raw.split(",") if l.strip()]
        return [str(l).strip() for l in (raw or []) if str(l).strip()]

    def rest(self, route: str, method: str, body: Optional[dict]) -> GhResult:
        self.sent_at = utc_now_iso()
        return _gh_api(self.ctx, route, repo=self.repo, method=method, body=body, timeout=60)

    def graphql(self, document: str, variables: dict, *, repo_vars: bool) -> GhResult:
        self.sent_at = utc_now_iso()
        return _gh_graphql(self.ctx, document, variables, repo=self.repo, repo_vars=repo_vars, timeout=60)

    def lookup_since(self) -> str:
        when = _dt.datetime.fromisoformat(self.sent_at) - _dt.timedelta(seconds=_LOOKUP_SLACK_SEC)
        return when.strftime("%Y-%m-%dT%H:%M:%SZ")

    # ---- the receipt --------------------------------------------------------------------------
    def receipt(self, effect: str, target: str, *, response: Optional[dict] = None, res: Optional[GhResult] = None,
                readback: Optional[dict] = None, url: object = None, oid: object = None, actor: object = _RESPONSE_ACTOR,
                at: object = None, lookup: str = "", extra: Optional[dict] = None,
                settlement: Optional[str] = None, result_status: Optional[str] = None) -> str:
        """Publish the one receipt shape; ``res`` is the failed transport answer when there is one."""
        if res is not None and res.failure in ("target", "cli_missing"):
            return res.text  # the transport already published its typed refusal; nothing was sent
        response = response if isinstance(response, dict) else {}
        receipt: Dict[str, Any] = {**(extra or {}),
            "effect": effect, "target": target,
            "url": url if url is not None else response.get("html_url") or response.get("url"),
            "id": oid if oid is not None else response.get("id"),
            "actor": _actor(response) if actor is _RESPONSE_ACTOR else actor,
            "at": at if at is not None else response.get("created_at") or response.get("submitted_at") or response.get("updated_at"),
        }
        # CLI PR commands may fail after an effect, or succeed before a differing readback.
        # Their separate readback owns settlement; direct API methods retain response semantics.
        if settlement is not None:
            receipt["outcome"] = settlement
            status = result_status or ("error" if res is not None and not res.ok else "ok")
            code = "OK" if status == "ok" else "TOOL_ERROR"
            outcome = {"confirmed": "completed", "not_applied": "completed_no_effect", "unknown": "unknown"}[settlement]
            prefix = ""
            if res is not None and not res.ok:
                receipt.update(error=res.text, failure=res.failure, exit_code=res.exit_code, http_status=res.http_status)
            if settlement == "unknown":
                prefix = "⚠️ GH_WRITE_UNKNOWN: the effect is unconfirmed; do not resend. "
                receipt["lookup"] = lookup
            if readback is not None:
                receipt["readback"] = readback
        elif res is None or res.ok:
            receipt["outcome"] = "confirmed"
            if readback is not None:
                receipt["readback"] = readback
            status, code, outcome, prefix = "ok", "OK", "completed", ""
            if readback and readback.get("verified") is False:
                prefix = "⚠️ GH_WRITE_READBACK_MISMATCH: GitHub's record differs from what was sent in "
                prefix += ", ".join(readback.get("mismatch") or []) + "; "
        elif res.failure == "exit" and res.http_status is not None and 400 <= res.http_status < 500:
            receipt.update(outcome="not_applied", http_status=res.http_status, error=res.text)
            status, code, outcome = "error", "TOOL_ERROR", "completed_no_effect"
            prefix = f"⚠️ GH_WRITE_NOT_APPLIED: GitHub refused (HTTP {res.http_status}); nothing changed. "
        else:
            receipt.update(outcome="unknown", error=res.text, failure=res.failure)
            if self.observed_match:
                receipt["observed_match"] = self.observed_match
            if res.http_status is not None:
                receipt["http_status"] = res.http_status
            if lookup:
                receipt["lookup"] = lookup
            status, code, outcome = "error", "TOOL_ERROR", "unknown"
            prefix = ("⚠️ GH_WRITE_UNKNOWN: the request was sent and no answer proved its outcome; do not resend. "
                      "Reconcile with " + (lookup or "github_read") + " before any further write. ")
        line = f"{prefix}github_write {self.method} → {target}: {receipt['outcome']}"
        if receipt.get("url"):
            line += f" {receipt['url']}"
        if receipt.get("actor"):
            line += f" by @{receipt['actor']}"
        if receipt.get("at"):
            line += f" at {receipt['at']}"
        return _publish_tool_result(self.ctx, ToolResult(status=status, code=code, text=line + "\n" + json.dumps(
            receipt, ensure_ascii=False, indent=1), meta={"operation_outcome": outcome}))


def _actor(obj: dict, *keys: str) -> Optional[str]:
    for key in keys or ("user", "author", "actor"):
        inner = obj.get(key) if isinstance(obj, dict) else None
        if isinstance(inner, dict) and inner.get("login"):
            return str(inner["login"])
    return None


def _parse(res: GhResult) -> Optional[dict]:
    try:
        data = json.loads(res.text) if res.text.strip() else {}
    except ValueError:
        return None
    return data if isinstance(data, dict) else {"data": data}


def _readback(sent: Dict[str, Any], got: Dict[str, Any]) -> dict:
    mismatch = [key for key, value in sent.items() if got.get(key) != value]
    return {"verified": not mismatch, "mismatch": mismatch, "via": "response"}


def _number_of(url: object) -> Optional[int]:
    tail = str(url or "").rsplit("/", 1)[-1]
    return int(tail) if tail.isdigit() else None


def _expectation(write: _Write, number: int) -> str:
    """Refuse before sending when the target moved past what the caller read; '' when it holds."""
    expect = write.args.get("expect") or {}
    if not isinstance(expect, dict):
        return write.refuse("⚠️ TOOL_ARG_ERROR: expect must be an object with head_sha, state and/or updated_at.")
    wanted = {k: str(v) for k, v in expect.items() if k in ("head_sha", "state", "updated_at") and v not in (None, "")}
    if not wanted:
        return ""
    res = _gh_api(write.ctx, f"repos/{{owner}}/{{repo}}/{'pulls' if 'head_sha' in wanted else 'issues'}/{number}", repo=write.repo)
    if not res.ok:
        return res.text
    current = _parse(res) or {}
    facts = {"head_sha": str((current.get("head") or {}).get("sha") or ""), "state": str(current.get("state") or "").lower(),
             "updated_at": str(current.get("updated_at") or "")}
    moved = [f"{k} is {facts[k] or 'unset'!r}, expected {v!r}" for k, v in wanted.items() if facts[k] != (v.lower() if k == "state" else v)]
    if moved:
        return _refuse(write.ctx, "⚠️ GH_EXPECTATION_FAILED: #" + str(number) + ": " + "; ".join(moved)
                       + ". Nothing was sent; read it again before writing.", "TOOL_ERROR", no_effect=True)
    return ""


def _validate_expect(write: _Write) -> str:
    """Every supplied precondition must have an interpretation before any request is sent."""
    expect = write.args.get("expect")
    if expect is None:
        return ""
    if not isinstance(expect, dict):
        return write.refuse("⚠️ TOOL_ARG_ERROR: expect must be an object with head_sha, state and/or updated_at.")
    pr_method = write.method == "create_pr" or (write.method == "update" and write.args.get("kind") == "pr"
                and any(key in write.args for key in ("title", "body", "base", "draft")))
    allowed = {"head_sha", "base_sha"} if pr_method else {"head_sha", "state", "updated_at"}
    unknown = set(expect) - allowed
    if unknown:
        return write.refuse("⚠️ TOOL_ARG_ERROR: unknown expect key(s): " + ", ".join(sorted(unknown)) + ".")
    if "base_sha" in expect and write.method != "create_pr" and write.args.get("base") is None:
        return write.refuse("⚠️ TOOL_ARG_ERROR: expect.base_sha requires the PR base to check.")
    supported = write.method in ("create_pr", "update", "review") or (write.method == "comment" and write.args.get("kind") != "discussion")
    if write.method == "api":
        supported = write.number() is not None and write.args.get("kind") in (None, "", "issue", "pr")
    if expect and not supported:
        return write.refuse("⚠️ TOOL_ARG_ERROR: expect requires an issue/PR target on comment, update, review or api; "
                            f"{write.method} with this target cannot check it.")
    return ""


# ---- methods ---------------------------------------------------------------------------------------
def _comment(write: _Write) -> str:
    body, number, kind = str(write.args.get("body") or ""), write.number(), str(write.args.get("kind") or "")
    reply_to = write.args.get("reply_to")
    if number is None:
        return write.refuse("⚠️ TOOL_ARG_ERROR: target must be the positive issue, pull request or discussion number.")
    if not body.strip():
        return write.refuse("⚠️ TOOL_ARG_ERROR: comment body cannot be empty.")
    if kind == "discussion":
        res = _gh_graphql(write.ctx, _GQL["discussion_id"], {"number": number}, repo=write.repo)
        node = ((_parse(res) or {}).get("data") or {}).get("repository", {}).get("discussion") if res.ok else None
        if not res.ok:
            return res.text
        if not isinstance(node, dict) or not node.get("id"):
            return write.refuse(f"⚠️ GH_ERROR: discussion #{number} was not found (HTTP 404); nothing was sent.", "TOOL_ERROR",
                                no_effect=True)  # the id read is a read; the comment was never sent
        variables = {"discussionId": node["id"], "body": body}
        if reply_to:
            variables["replyToId"] = str(reply_to)
        res = write.graphql(_GQL["add_discussion_comment"], variables, repo_vars=False)
        target = f"{write.slug()} discussion #{number}" + (f" reply to {reply_to}" if reply_to else "")
        if not res.ok:
            return write.receipt("discussion_comment.create", target, res=res, lookup=f"github_read(method='list', kind='discussion', target={number})")
        comment = (((_parse(res) or {}).get("data") or {}).get("addDiscussionComment") or {}).get("comment") or {}
        readback = _readback({"body": body}, comment)
        return write.receipt("discussion_comment.create", target, url=comment.get("url"), oid=comment.get("id"),
                             actor=_actor(comment, "author"), at=comment.get("createdAt"), readback=readback)
    if error := _expectation(write, number):
        return error
    if reply_to:
        route, effect = f"repos/{{owner}}/{{repo}}/pulls/{number}/comments/{reply_to}/replies", "review_comment.reply"
        lookup = f"github_read(method='list', kind='review_comment', target={number}, since='{{since}}')"
    else:
        route, effect = f"repos/{{owner}}/{{repo}}/issues/{number}/comments", "issue_comment.create"
        lookup = f"github_read(method='list', kind='comment', target={number}, since='{{since}}')"
    res = write.rest(route, "POST", {"body": body})
    target = f"{write.slug()}#{number}" + (f" reply to {reply_to}" if reply_to else "")
    if res.ok:
        got = _parse(res) or {}
        readback = _readback({"body": body}, got)
        if _number_of(got.get("pull_request_url") or got.get("issue_url")) != number:
            readback.update(verified=False, mismatch=readback["mismatch"] + ["target"])
        return write.receipt(effect, f"{write.slug(got.get('html_url'))}#{number}" + (f" reply to {reply_to}" if reply_to else ""),
                             response=got, readback=readback)
    if _definite(res):
        return write.receipt(effect, target, res=res)
    since = write.lookup_since()
    found = _find(write, route.rsplit("/replies", 1)[0].replace(f"/comments/{reply_to}", "/comments") if reply_to else route,
                  {"since": since, "per_page": 100},
                  lambda row: row.get("body") == body and (not reply_to or str(row.get("in_reply_to_id")) == str(reply_to)))
    if found:
        return write.receipt(effect, f"{write.slug(found.get('html_url'))}#{number}" + (f" reply to {reply_to}" if reply_to else ""),
                             response=found, readback={"verified": True, "mismatch": [], "via": "lookup"})
    return write.receipt(effect, target, res=res, lookup=lookup.replace("{since}", since))


def _create_issue(write: _Write) -> str:
    title, body = str(write.args.get("title") or ""), str(write.args.get("body") or "")
    if not title.strip():
        return write.refuse("⚠️ TOOL_ARG_ERROR: issue title cannot be empty.")
    payload: Dict[str, Any] = {"title": title, "body": body}
    labels = write.labels()
    if labels:
        payload["labels"] = labels
    res = write.rest("repos/{owner}/{repo}/issues", "POST", payload)
    if res.ok:
        got = _parse(res) or {}
        readback = _readback({"title": title, "body": body or None}, {**got, "body": got.get("body") or None})
        if labels and {str(l.get("name")) for l in got.get("labels") or [] if isinstance(l, dict)} != set(labels):
            readback.update(verified=False, mismatch=readback["mismatch"] + ["labels"])
        return write.receipt("issue.create", f"{write.slug(got.get('html_url'))}#{got.get('number')}", response=got, readback=readback)
    if _definite(res):
        return write.receipt("issue.create", f"{write.slug()} new issue {title[:60]!r}", res=res)
    since = write.lookup_since()
    found = _find(write, "repos/{owner}/{repo}/issues", {"state": "all", "since": since, "sort": "created", "direction": "desc", "per_page": 50},
                  lambda row: row.get("title") == title and (row.get("body") or "") == body and "pull_request" not in row)
    if found:
        return write.receipt("issue.create", f"{write.slug(found.get('html_url'))}#{found.get('number')}", response=found,
                             readback={"verified": True, "mismatch": [], "via": "lookup"})
    return write.receipt("issue.create", f"{write.slug()} new issue {title[:60]!r}", res=res,
                         lookup=f"github_read(method='list', kind='issue', state='all', since='{since}')")


def _update(write: _Write) -> str:
    if write.args.get("kind") == "pr" and (any(key in write.args for key in ("title", "body", "base", "draft"))
                                           or not any(key in write.args for key in ("state", "state_reason", "labels"))):
        from ouroboros.tools.github_pr_write import update

        return update(write)
    if any(key in write.args for key in ("base", "draft")):
        return write.refuse("⚠️ TOOL_ARG_ERROR: base and draft updates require kind='pr'.")
    number = write.number()
    if number is None:
        return write.refuse("⚠️ TOOL_ARG_ERROR: target must be the positive issue or pull request number.")
    fields: Dict[str, Any] = {}
    for key in ("title", "body", "state", "state_reason"):
        value = write.args.get(key)
        if value not in (None, ""):
            fields[key] = str(value)
    if "labels" in write.args and write.args.get("labels") is not None:
        fields["labels"] = write.labels()
    if fields.get("state") not in (None, "open", "closed"):
        return write.refuse("⚠️ TOOL_ARG_ERROR: update state is 'open' or 'closed' (resolved/unresolved belong to resolve_thread).")
    if fields.get("state_reason") not in (None, *STATE_REASONS):
        return write.refuse(f"⚠️ TOOL_ARG_ERROR: state_reason must be one of {', '.join(STATE_REASONS)}.")
    if not fields:
        return write.refuse("⚠️ TOOL_ARG_ERROR: update changes at least one of title, body, state, state_reason, labels.")
    if error := _expectation(write, number):
        return error
    res = write.rest(f"repos/{{owner}}/{{repo}}/issues/{number}", "PATCH", fields)
    target = f"{write.slug()}#{number}"

    def compare(got: dict) -> dict:
        shown = {**got, "labels": sorted(str(l.get("name")) for l in got.get("labels") or [] if isinstance(l, dict))}
        wanted = {**fields, **({"labels": sorted(fields["labels"])} if "labels" in fields else {})}
        return _readback(wanted, shown)

    if res.ok:
        got = _parse(res) or {}
        return write.receipt("issue.update", f"{write.slug(got.get('html_url'))}#{number}", response=got, readback=compare(got),
                             at=got.get("updated_at"), actor=_actor(got, "closed_by") if fields.get("state") == got.get("state") == "closed" else None)
    if _definite(res):
        return write.receipt("issue.update", target, res=res)
    current = _find(write, f"repos/{{owner}}/{{repo}}/issues/{number}", {}, lambda row: compare(row)["verified"],
                    actor=lambda row: _actor(row, "closed_by") if fields.get("state") == "closed" else None)
    if current:
        return write.receipt("issue.update", target, response=current, readback={"verified": True, "mismatch": [], "via": "lookup"},
                             at=current.get("updated_at"), actor=_actor(current, "closed_by") if fields.get("state") == "closed" else None)
    return write.receipt("issue.update", target, res=res, lookup=f"github_read(method='get', kind='issue', target={number})")


def _review(write: _Write) -> str:
    number, event, body = write.number(), str(write.args.get("event") or "COMMENT").upper(), str(write.args.get("body") or "")
    comments = write.args.get("comments") or []
    if number is None:
        return write.refuse("⚠️ TOOL_ARG_ERROR: target must be the positive pull request number.")
    if event not in REVIEW_EVENTS:
        return write.refuse("⚠️ TOOL_ARG_ERROR: event must be APPROVE, REQUEST_CHANGES or COMMENT.")
    if not isinstance(comments, list) or any(not isinstance(c, dict) or not str(c.get("path") or "") or not str(c.get("body") or "").strip()
                                              for c in comments):
        return write.refuse("⚠️ TOOL_ARG_ERROR: comments is a list of {path, body, line?, side?, start_line?, start_side?} objects.")
    if event != "APPROVE" and not body.strip() and not comments:
        return write.refuse("⚠️ TOOL_ARG_ERROR: a REQUEST_CHANGES or COMMENT review needs a body or inline comments.")
    if error := _expectation(write, number):
        return error
    payload: Dict[str, Any] = {"event": event, "body": body}
    if comments:
        payload["comments"] = [{k: c[k] for k in ("path", "body", "line", "side", "start_line", "start_side", "position") if k in c} for c in comments]
    if (write.args.get("expect") or {}).get("head_sha"):
        payload["commit_id"] = str(write.args["expect"]["head_sha"])
    res = write.rest(f"repos/{{owner}}/{{repo}}/pulls/{number}/reviews", "POST", payload)
    target = f"{write.slug()}#{number}"
    wanted = {"state": REVIEW_EVENTS[event], "body": body}
    if res.ok:
        got = _parse(res) or {}
        return write.receipt("pr_review.submit", f"{write.slug(got.get('html_url'))}#{number}", response=got, readback=_readback(wanted, got))
    if _definite(res):
        return write.receipt("pr_review.submit", target, res=res)
    found = _find(write, f"repos/{{owner}}/{{repo}}/pulls/{number}/reviews", {"per_page": 100},
                  lambda row: row.get("body") == body and row.get("state") == wanted["state"] and str(row.get("submitted_at") or "") >= write.lookup_since())
    if found:
        return write.receipt("pr_review.submit", target, response=found, readback={"verified": True, "mismatch": [], "via": "lookup"})
    return write.receipt("pr_review.submit", target, res=res, lookup=f"github_read(method='list', kind='review', target={number})")


def _resolve_thread(write: _Write) -> str:
    thread, state = str(write.args.get("target") or "").strip(), str(write.args.get("state") or "resolved")
    if not thread or thread.isdigit():
        return write.refuse("⚠️ TOOL_ARG_ERROR: target must be the review thread node id (read it with github_read get(kind='pr', include=['threads'])).")
    if state not in ("resolved", "unresolved"):
        return write.refuse("⚠️ TOOL_ARG_ERROR: resolve_thread state is 'resolved' (default) or 'unresolved'.")
    name = "resolve" if state == "resolved" else "unresolve"
    res = write.graphql(_GQL[name], {"id": thread}, repo_vars=False)
    effect, target = f"review_thread.{name}", f"{write.slug()} thread {thread}"
    expected = state == "resolved"
    if res.ok:
        node = (((_parse(res) or {}).get("data") or {}).get(f"{name}ReviewThread") or {}).get("thread") or {}
        readback = {"verified": node.get("isResolved") is expected, "mismatch": [] if node.get("isResolved") is expected else ["state"], "via": "response"}
        return write.receipt(effect, target, oid=node.get("id") or thread, url="", actor=_actor(node, "resolvedBy") if expected else None,
                             at=write.sent_at, readback=readback, extra={"at_source": "local"})
    if _definite(res):
        return write.receipt(effect, target, res=res, oid=thread, url="", actor="", at="")
    check = _gh_graphql(write.ctx, _GQL["thread"], {"id": thread}, repo=write.repo, repo_vars=False)
    node = ((_parse(check) or {}).get("data") or {}).get("node") if check.ok else None
    if isinstance(node, dict) and node.get("isResolved") is expected and _confirm_own(
            write, [node], actor=lambda row: _actor(row, "resolvedBy") if expected else None):
        return write.receipt(effect, target, oid=thread, url="", actor=_actor(node, "resolvedBy") if expected else None, at=write.sent_at,
                             readback={"verified": True, "mismatch": [], "via": "lookup"}, extra={"at_source": "local"})
    return write.receipt(effect, target, res=res, oid=thread, url="", actor="", at="",
                         lookup=f"github_read(method='api', graphql=<node(id:{thread!r}) isResolved>)")


def _api(write: _Write) -> str:
    route, document = str(write.args.get("route") or "").strip(), str(write.args.get("graphql") or "").strip()
    if bool(route) == bool(document):
        return write.refuse("⚠️ TOOL_ARG_ERROR: api takes exactly one of route (with http_method and input) or graphql (a mutation document).")
    if error := _expectation(write, write.number()):
        return error
    if document:
        if _graphql_operations(document) != {"mutation"}:
            return write.refuse("⚠️ TOOL_ARG_ERROR: github_write(method='api', graphql=...) sends mutations; a query goes through "
                                "github_read(method='api', graphql=...).")
        variables = write.args.get("variables") or {}
        if not isinstance(variables, dict):
            return write.refuse("⚠️ TOOL_ARG_ERROR: variables must be an object.")
        match = _MUTATION_NAME_RE.search(document)
        effect = "mutation " + ((match.group(1) or match.group(2)) if match else "(unnamed)")
        res = write.graphql(document, variables, repo_vars="$owner" in document and "$name" in document)
        target = f"{write.slug()} graphql"
    else:
        http_method = str(write.args.get("http_method") or "POST").upper()
        if http_method not in HTTP_METHODS:
            return write.refuse(f"⚠️ TOOL_ARG_ERROR: http_method must be one of {', '.join(HTTP_METHODS)} (a GET is github_read).")
        body = write.args.get("input")
        if body is not None and not isinstance(body, dict):
            return write.refuse("⚠️ TOOL_ARG_ERROR: input must be a JSON object (the request body).")
        effect = f"{http_method} {_template(route, write.repo)}"
        res = write.rest(route, http_method, body)
        target = f"{write.slug()} {route.split('?', 1)[0]}"
    if res.ok:
        got = _parse(res) or {}
        first = _first_object(got)
        # Arbitrary GraphQL selections/aliases do not prove creation from the response key alone.
        created = not document and http_method == "POST" and route.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1] in {
            "comments", "reviews", "reactions", "replies", "issues", "pulls", "releases"}
        at = (first.get("created_at") or first.get("createdAt") or first.get("submitted_at")) if created else (
            first.get("updated_at") or first.get("updatedAt"))
        return write.receipt(effect, target, url=first.get("html_url") or first.get("url") or "", oid=first.get("id") or first.get("node_id") or "",
                             actor=_actor(first) if created else None, at=at or write.sent_at,
                             readback=None, extra=None if at else {"at_source": "local"})
    return write.receipt(effect, target, res=res, url="", oid="", actor="", at="",
                         lookup="" if _definite(res) else "github_read(method='api', ...) on the same object")


def _first_object(data: dict) -> dict:
    """The REST object itself, or the first object inside a GraphQL ``data`` payload."""
    if "data" in data and isinstance(data["data"], dict):
        for value in data["data"].values():
            if isinstance(value, dict):
                inner = next((v for v in value.values() if isinstance(v, dict)), None)
                return inner if inner is not None and not any(k in value for k in ("html_url", "url", "id")) else value
    return data


def _template(route: str, repo: str) -> str:
    path = route.split("?", 1)[0]
    parts = _split_repo(repo) if repo else None
    if parts:
        path = path.replace(f"repos/{parts[1]}/{parts[2]}", "repos/{owner}/{repo}")
    return _NUMBER_SEGMENT_RE.sub("/{n}", path)


def _definite(res: GhResult) -> bool:
    """Whether a failed send proves nothing was applied (a GitHub 4xx, a local refusal, no CLI)."""
    return res.failure in ("target", "cli_missing") or (res.failure == "exit" and res.http_status is not None and 400 <= res.http_status < 500)


def _confirm_own(write: _Write, rows: list, *, actor=_actor) -> Optional[dict]:
    """A content/state match alone is evidence, not proof that our unanswered send applied."""
    if not rows:
        return None
    identity = _gh_api(write.ctx, "user", repo=write.repo)
    login = (_parse(identity) or {}).get("login") if identity.ok else None
    found = next((row for row in rows if login and str(actor(row) or "").casefold() == str(login).casefold()), None)
    if found is None:
        row = rows[0]
        write.observed_match = {"id": row.get("id"), "url": row.get("html_url") or row.get("url"),
                                "actor": actor(row), "acting_login": login,
                                "note": "Matching content/state does not establish this acting identity's effect."}
    return found


def _find(write: _Write, route: str, params: dict, match, *, actor=_actor) -> Optional[dict]:
    """One lookup read after an unknown send: the matching row (or object), else None."""
    from urllib.parse import urlencode

    query = urlencode({k: v for k, v in params.items() if v not in (None, "")})
    res = _gh_api(write.ctx, route + (("&" if "?" in route else "?") + query if query else ""), repo=write.repo)
    if not res.ok:
        return None
    try:
        data = json.loads(res.text)
    except ValueError:
        return None
    rows = data if isinstance(data, list) else [data] if isinstance(data, dict) else []
    return _confirm_own(write, [row for row in rows if isinstance(row, dict) and match(row)], actor=actor)


def run_write(ctx: ToolContext, method: str = "", repo: str = "", **args: Any) -> str:
    """Dispatch one ``github_write`` call; the registry has already refused unknown arguments."""
    if not isinstance(repo, str):
        return _refuse(ctx, "⚠️ GH_TARGET_INVALID: repo must be a string; omit it to use the selected Project.", no_effect=_no_launch_yet())
    args = _canonical_github_args({"method": method, **args})
    method = args.pop("method")
    if method not in METHODS:
        return _refuse(ctx, f"⚠️ TOOL_ARG_ERROR: method must be one of {', '.join(METHODS)}.", no_effect=_no_launch_yet())
    write = _Write(ctx, method, repo, **args)
    if error := _validate_expect(write):
        return error
    if method == "create_pr":
        from ouroboros.tools.github_pr_write import create

        return create(write)
    return {"comment": _comment, "create_issue": _create_issue, "update": _update, "review": _review,
            "resolve_thread": _resolve_thread, "api": _api}[method](write)


SCHEMA: Dict[str, Any] = {
    "name": "github_write",
    "description": (
        "Write to GitHub as the configured identity (see github_read whoami) on the active Project or an explicit repo. "
        "Every call returns one receipt {effect, target, url, id, actor, at, outcome}, actor known or null, local time "
        "labelled, and a read-back of what was sent; an answer that never came is looked up once and otherwise reported "
        "outcome='unknown' (never resend; reconcile with github_read first); a direct API 4xx is 'not_applied'. "
        "Methods: comment = a conversation comment on an issue or pull request (target = number; kind='discussion' "
        "comments on a discussion, reply_to = its comment node id; reply_to = a review comment id answers inline in a PR "
        "code thread); create_issue = a new issue (title, body, labels); create_pr = open a PR from an existing head "
        "into base with title, body, draft and required expect={head_sha,base_sha}. update kind='pr' edits title/body/base "
        "with expect.head_sha (base also needs expect.base_sha), or toggles draft with expect.head_sha in a separate call. "
        "These PR writes precheck and read back each field: not a server-side compare-and-swap; no push/fork/merge, "
        "and a failed CLI send or unverified readback stays unknown. Other updates change title, body, state open|closed, "
        "state_reason (completed, not_planned, reopened, duplicate) or labels (replaces) of an issue or PR; review = submit "
        "a pull request review (event APPROVE|REQUEST_CHANGES|COMMENT, body, inline comments [{path, line, body}]); "
        "resolve_thread = resolve or unresolve a review thread by its node id (from github_read get include=['threads']); "
        "api = any REST mutation (http_method, route, input) or GraphQL mutation (graphql, variables) GitHub offers and "
        "no method covers yet — the receipt names it as 'METHOD route' or 'mutation Name'. expect={head_sha, state, "
        "updated_at} checks issue/PR targets and refuses before sending if changed or unsupported. Merging is pr_merge."
    ),
    "parameters": {"type": "object", "properties": {
        "method": {"type": "string", "enum": list(METHODS),
                   "description": "comment | create_issue | create_pr | update | review | resolve_thread | api"},
        "kind": {"type": "string", "enum": ["issue", "pr", "discussion"],
                 "description": "comment: 'discussion' targets a discussion number; issue and pr share one conversation route"},
        "target": {"type": ["integer", "string"], "default": "",
                   "description": "issue/PR/discussion number; for resolve_thread the review thread node id"},
        "body": {"type": "string", "description": "Markdown body; PR update: omitted leaves it, empty clears it"},
        "title": {"type": "string", "description": "create_issue / create_pr / update: the title"},
        "head": {"type": "string", "description": "create_pr: existing branch or OWNER:BRANCH; a fork uses the same repository name"},
        "base": {"type": "string", "description": "create_pr / update kind=pr: base branch; requires expect.base_sha"},
        "draft": {"type": "boolean", "description": "create_pr: default false; update kind=pr: true makes draft, false makes ready; separate from field edits"},
        "state": {"type": "string", "enum": ["open", "closed", "resolved", "unresolved"],
                  "description": "update: open|closed; resolve_thread: resolved (default) | unresolved"},
        "state_reason": {"type": "string", "enum": list(STATE_REASONS), "description": "update: why it is closed or reopened"},
        "labels": {"type": "array", "items": {"type": "string"},
                   "description": "create_issue: labels to add; update: the full label list (replaces)"},
        "event": {"type": "string", "enum": list(REVIEW_EVENTS), "description": "review: APPROVE, REQUEST_CHANGES or COMMENT"},
        "comments": {"type": "array", "default": [], "items": {"type": "object", "properties": {
            "path": {"type": "string"}, "body": {"type": "string"}, "line": {"type": "integer"},
            "side": {"type": "string", "enum": ["LEFT", "RIGHT"]}, "start_line": {"type": "integer"},
            "start_side": {"type": "string", "enum": ["LEFT", "RIGHT"]}}, "required": ["path", "body"]},
                     "description": "review: inline comments on the diff"},
        "reply_to": {"type": ["integer", "string"], "default": "",
                     "description": "comment: review comment id to answer inline (PR), or discussion comment node id"},
        "expect": {"type": "object", "default": {}, "properties": {
            "head_sha": {"type": "string"}, "base_sha": {"type": "string"}, "state": {"type": "string"}, "updated_at": {"type": "string"}},
                   "description": "Preconditions read from github_read; a changed target refuses before sending"},
        "http_method": {"type": "string", "enum": list(HTTP_METHODS), "default": "POST", "description": "api: the REST verb"},
        "route": {"type": "string", "default": "", "description": "api: REST route, e.g. repos/{owner}/{repo}/issues/7/labels"},
        "input": {"type": "object", "default": {}, "description": "api: the JSON request body"},
        "graphql": {"type": "string", "default": "", "description": "api: GraphQL mutation document ($owner/$name filled from repo when declared)"},
        "variables": {"type": "object", "default": {}, "description": "api: GraphQL variables"},
    }, "required": ["method"]},
}
