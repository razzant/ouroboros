"""The methods of ``github_read``: whoami, get, list, search, checks and a GET/query-only ``api``.

Not a tool module: ``github.py`` registers the tool and owns the transport this module calls.
Every answer is one human line plus JSON. A cut is never silent: a bounded body carries
``body_chars`` and an ``omitted`` note, a page carries ``complete`` and ``next_cursor``, and a
result the tool limit cannot hold is saved whole as a task artifact behind a short manifest.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

from ouroboros.tool_capabilities import tool_result_limit
from ouroboros.tools.github import _canonical_github_args, _gh_api, _gh_graphql, _gh_run, _graphql_operations, _no_launch_yet, _refuse, _split_repo, github_token_from_env_or_settings
from ouroboros.tools.registry import ToolContext

METHODS = ("whoami", "get", "list", "search", "checks", "download", "api")
KINDS = ("issue", "pr", "discussion", "comment", "review_comment", "review", "notification", "release", "label", "run", "artifact")
FACETS = ("comments", "review_comments", "reviews", "threads", "files", "diff", "commits", "log")
_LIMIT_DEFAULT, _LIMIT_MAX = 30, 100
_BODY_CHARS, _PREVIEW_CHARS, _DIFF_CHARS, _REPLIES = 2000, 200, 8000, 50
_LINK_NEXT_RE = re.compile(r'<([^>]*)>;\s*rel="next"')
_PAGE_RE = re.compile(r"[?&]page=(\d+)")
_URL_REPO_RE = re.compile(r"^https://([^/]+)/([^/]+)/([^/]+)/")
# Exact commit counts and the review decision of listed PRs by node id: one fixed read, no commit objects.
_PR_COMMIT_COUNTS_QUERY = ("query($ids:[ID!]!){nodes(ids:$ids){...on PullRequest{id commits{totalCount} reviewDecision "
                           "headRefName headRefOid headRepositoryOwner{login} baseRefName isDraft mergedAt}}}")
_PAGE_INFO = "pageInfo{hasNextPage endCursor}"
_DISCUSSION_FIELDS = "id number title url createdAt updatedAt author{login} category{name} isAnswered comments{totalCount}"
_GQL = {
    "discussions": "query($owner:String!,$name:String!,$first:Int!,$after:String){repository(owner:$owner,name:$name)"
                   "{discussions(first:$first,after:$after,orderBy:{field:UPDATED_AT,direction:DESC}){totalCount "
                   + _PAGE_INFO + " nodes{" + _DISCUSSION_FIELDS + " body}}}}",
    "discussion": "query($owner:String!,$name:String!,$number:Int!){repository(owner:$owner,name:$name)"
                  "{discussion(number:$number){" + _DISCUSSION_FIELDS + " locked body}}}",
    "discussion_comments": "query($owner:String!,$name:String!,$number:Int!,$first:Int!,$after:String){repository(owner:$owner,"
                           "name:$name){discussion(number:$number){comments(first:$first,after:$after){totalCount " + _PAGE_INFO
                           + " nodes{id databaseId body author{login} createdAt updatedAt url isAnswer replies(first:"
                           + str(_REPLIES) + "){totalCount nodes{id databaseId body author{login} createdAt url}}}}}}}",
    "threads": "query($owner:String!,$name:String!,$number:Int!,$first:Int!,$after:String){repository(owner:$owner,name:$name)"
               "{pullRequest(number:$number){reviewThreads(first:$first,after:$after){totalCount " + _PAGE_INFO
               + " nodes{id isResolved isOutdated path line startLine diffSide comments(first:" + str(_REPLIES)
               + "){totalCount nodes{id databaseId author{login} body createdAt url}}}}}}}",
    "search_discussions": "query($q:String!,$first:Int!,$after:String){search(query:$q,type:DISCUSSION,first:$first,after:$after)"
                          "{discussionCount " + _PAGE_INFO + " nodes{... on Discussion{" + _DISCUSSION_FIELDS
                          + " repository{nameWithOwner}}}}}",
}


class _Read:
    """One invocation: arguments, the notes it accumulates and the delivery."""

    def __init__(self, ctx: ToolContext, method: str, repo: str, **args: Any):
        self.ctx, self.method, self.repo, self.args = ctx, method, repo, args
        self.kind = str(args.get("kind") or "")
        self.detail = str(args.get("detail") or "concise")
        self.since = str(args.get("since") or "").strip()
        self.cursor = str(args.get("cursor") or "").strip()
        self.fields = [str(f) for f in (args.get("fields") or []) if str(f)]
        self.omitted: List[str] = []
        limit = args.get("limit")
        self.limit = _LIMIT_DEFAULT if limit in (None, "") else (int(limit) if isinstance(limit, int) and not isinstance(limit, bool) else None)
        if self.limit is not None and self.limit > _LIMIT_MAX:
            self.omitted.append(f"limit {self.limit} capped at {_LIMIT_MAX} (GitHub's page size); page on with cursor")
            self.limit = _LIMIT_MAX

    # ---- helpers -------------------------------------------------------------------------------
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

    def slug(self) -> Tuple[str, str]:
        """``(slug, error)``: the explicit repository, or the bound Project's as GitHub names it."""
        parts = _split_repo(self.repo) if self.repo else None
        if parts:
            return f"{parts[1]}/{parts[2]}", ""
        res = _gh_run(["repo", "view", "--json", "nameWithOwner"], self.ctx, repo="")
        if not res.ok:
            return "", res.text
        try:
            return str(json.loads(res.text)["nameWithOwner"]), ""
        except (ValueError, TypeError, KeyError):
            return "", self.refuse("⚠️ TOOL_ERROR: the bound repository name could not be read from gh.", "TOOL_ERROR")

    def page(self, route: str, params: Optional[dict] = None) -> Tuple[Optional[list], Optional[dict], str]:
        """One REST page: ``(rows, page_facts, error)``; ``complete`` is read from the Link header."""
        if self.cursor and not self.cursor.isdigit():
            return None, None, self.refuse(f"⚠️ TOOL_ARG_ERROR: cursor {self.cursor!r} is not a page cursor of this list.")
        number = int(self.cursor or 1)
        query = {k: v for k, v in (params or {}).items() if v not in (None, "")}
        query.update(per_page=self.limit, page=number)
        res = _gh_api(self.ctx, route + ("&" if "?" in route else "?") + urlencode(query), repo=self.repo, include=True)
        if not res.ok:
            return None, None, res.text
        _status, headers, body = _split_http(res.text)
        try:
            rows = json.loads(body)
        except ValueError:
            return None, None, self.refuse(f"⚠️ TOOL_ERROR: failed to parse GitHub's JSON: {body[:300]}", "TOOL_ERROR")
        link = _LINK_NEXT_RE.search(headers.get("link", ""))
        page = _PAGE_RE.search(link.group(1)) if link else None
        facts = {"complete": page is None, "next_cursor": page.group(1) if page else ""}
        return rows, facts, ""

    def graphql(self, name: str, variables: dict, *, repo_vars: bool = True) -> Tuple[Optional[dict], str]:
        res = _gh_graphql(self.ctx, _GQL[name], variables, repo=self.repo, repo_vars=repo_vars)
        if not res.ok:
            return None, res.text
        try:
            answer = json.loads(res.text)
        except ValueError:
            return None, self.refuse(f"⚠️ TOOL_ERROR: failed to parse GitHub's GraphQL answer: {res.text[:300]}", "TOOL_ERROR")
        if answer.get("errors"):
            messages = "; ".join(str(e.get("message") or e) for e in answer["errors"] if isinstance(e, (dict, str)))[:600]
            return None, self.refuse(f"⚠️ GH_ERROR: GraphQL: {messages}", "TOOL_ERROR")
        return answer.get("data") or {}, ""

    def connection(self, node: dict) -> Tuple[list, dict]:
        info = node.get("pageInfo") or {}
        facts = {"complete": not info.get("hasNextPage"), "next_cursor": str(info.get("endCursor") or "") if info.get("hasNextPage") else ""}
        if isinstance(node.get("totalCount"), int):
            facts["total_count"] = node["totalCount"]
        return [n for n in node.get("nodes") or [] if isinstance(n, dict)], facts

    def body(self, text: object, item: dict, key: str = "body") -> None:
        """Concise: the body up to ``_BODY_CHARS`` with the cut stated; full: whole."""
        text = str(text or "")
        item[key + "_chars"] = len(text)
        if self.detail == "full" or len(text) <= _BODY_CHARS:
            item[key] = text
        else:
            item[key], item[key + "_truncated"] = text[:_BODY_CHARS], True
            self.omitted.append(f"{key} cut at {_BODY_CHARS} chars (of {len(text)}): detail='full' returns it whole")

    def project(self, item: dict) -> dict:
        if not self.fields:
            return item
        missing = [f for f in self.fields if f not in item]
        if missing:
            self.omitted.append("fields not present: " + ", ".join(missing))
        return {f: item[f] for f in self.fields if f in item}

    def rows(self, raw: list, shape) -> list:
        for row in raw:
            if isinstance(row, dict):
                self.nested_cuts(row)
        return [self.project(dict(r) if self.detail == "full" else shape(r)) for r in raw if isinstance(r, dict)]

    def nested_cuts(self, row: dict) -> None:
        """Nested pagination belongs to source completeness, independently of the display projection."""
        for key in ("comments", "replies"):
            connection = row.get(key)
            if not isinstance(connection, dict) or "nodes" not in connection:
                continue
            total, fetched = connection.get("totalCount"), len(connection.get("nodes") or [])
            if isinstance(total, int) and total > fetched:
                recovery = (f"github_read(method='list', kind='review_comment', target={self.number()})" if key == "comments" else
                            f"github_read(method='api', graphql=...) on DiscussionComment node {row.get('id')!r}, paging replies(first:100, after:...) with pageInfo")
                self.omitted.append(f"node {row.get('id')}: {total - fetched} {key} beyond the {fetched} fetched not shown; "
                                    f"{recovery} reads the rest")

    def deliver(self, line: str, payload: dict) -> str:
        payload = {"method": self.method, "kind": self.kind or None, "repo": self.repo or "selected Project", **payload}
        if self.omitted:
            payload["omitted"] = list(dict.fromkeys(self.omitted))
        text = line + "\n" + _dumps(payload)
        if len(text) <= tool_result_limit("github_read"):
            return text
        return _save(self.ctx, line, payload, self.method, self.kind)

    def since_filter(self, rows: list, key: str, facts: dict, *, newest_first: bool = False) -> list:
        """Filter a page; only a source descending by this timestamp permits stopping the walk early."""
        if not self.since:
            return rows
        kept = [r for r in rows if isinstance(r, dict) and str(r.get(key) or "") >= self.since]
        if newest_first and any(isinstance(r, dict) and r.get(key) and str(r[key]) < self.since for r in rows):
            facts.update(complete=True, next_cursor="")
        return kept


def _dumps(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=1)


def _split_http(text: str) -> Tuple[Optional[int], Dict[str, str], str]:
    """Status, lower-cased headers and body of a ``gh api --include`` answer."""
    head, sep, body = text.replace("\r\n", "\n").partition("\n\n")
    if not sep and not head.startswith("HTTP/"):
        return None, {}, text
    lines = head.splitlines()
    status = None
    if lines and lines[0].startswith("HTTP/"):
        parts = lines[0].split()
        status = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        lines = lines[1:]
    headers = {k.strip().lower(): v.strip() for k, _s, v in (line.partition(":") for line in lines) if _s}
    return status, headers, body


def _save(ctx: ToolContext, line: str, payload: dict, method: str, kind: str) -> str:
    """Save the whole result as a task artifact; the answer is its manifest."""
    from ouroboros.artifacts import store_task_artifact_bytes, task_artifact_dir_path, task_id_for_artifacts

    data = _dumps(payload).encode("utf-8")
    task_id = task_id_for_artifacts(ctx)
    name = f"github_read-{method}-{kind or 'any'}-{hashlib.sha256(data).hexdigest()[:12]}.json"
    receipt = store_task_artifact_bytes(ctx.drive_root, task_id, name, data)
    summary: Dict[str, Any] = {k: payload[k] for k in ("target", "total_count", "complete", "next_cursor") if k in payload}
    if isinstance(payload.get("items"), list):
        summary["items"] = len(payload["items"])
    if isinstance(payload.get("item"), dict):
        summary["item"] = {k: payload["item"][k] for k in ("number", "title", "state", "url", "login") if k in payload["item"]}
    for facet, page in (payload.get("facets") or {}).items():
        if isinstance(page, dict):
            summary[facet] = {k: page[k] for k in ("total_count", "complete", "next_cursor") if k in page}
            summary[facet]["items"] = len(page.get("items") or []) if "items" in page else len(str(page.get("text") or ""))
    manifest = {"method": method, "kind": kind or None, "saved": {
        "path": receipt["path"], "absolute_path": str(task_artifact_dir_path(ctx.drive_root, task_id) / receipt["path"]),
        "sha256": receipt["sha256"], "bytes": receipt["bytes"]}, "summary": summary,
        "note": "The complete result exceeds the tool result limit and is saved whole (nothing cut); read the file with "
                "read_file, or narrow the call with fields, limit or a single include facet."}
    return line + f" — saved whole as task artifact {receipt['path']}\n" + _dumps(manifest)


def _repo_of(url: object) -> str:
    match = _URL_REPO_RE.match(str(url or ""))
    return f"{match.group(2)}/{match.group(3)}" if match else ""


def _login(obj: object, key: str = "user") -> str:
    inner = (obj or {}).get(key) if isinstance(obj, dict) else None
    return str((inner or {}).get("login") or "") if isinstance(inner, dict) else ""


def _preview(read: _Read, item: dict, text: object) -> None:
    text = str(text or "")
    item["body_chars"] = len(text)
    item["body_preview"] = text[:_PREVIEW_CHARS] + ("…" if len(text) > _PREVIEW_CHARS else "")


# ---- shapes (concise) ------------------------------------------------------------------------------
def _issue_shape(read: _Read, it: dict, *, full_body: bool) -> dict:
    item = {"number": it.get("number"), "title": it.get("title"), "state": it.get("state"),
            "author": _login(it), "labels": [str(l.get("name")) for l in it.get("labels") or [] if isinstance(l, dict)],
            "assignees": [_login({"a": a}, "a") for a in it.get("assignees") or []],
            "created_at": it.get("created_at"), "updated_at": it.get("updated_at"), "closed_at": it.get("closed_at"),
            "comments": it.get("comments"), "url": it.get("html_url"), "is_pr": "pull_request" in it}
    if it.get("state_reason"):
        item["state_reason"] = it["state_reason"]
    if full_body:
        read.body(it.get("body"), item)
    else:
        _preview(read, item, it.get("body"))
    return item


def _pr_shape(read: _Read, it: dict, *, full_body: bool) -> dict:
    head, base = it.get("head") or {}, it.get("base") or {}
    item = {"number": it.get("number"), "title": it.get("title"), "state": it.get("state"), "draft": bool(it.get("draft")),
            "author": _login(it), "head": head.get("label") or head.get("ref"), "head_sha": head.get("sha"),
            "base": base.get("ref"), "created_at": it.get("created_at"), "updated_at": it.get("updated_at"),
            "merged_at": it.get("merged_at"), "url": it.get("html_url"),
            "labels": [str(l.get("name")) for l in it.get("labels") or [] if isinstance(l, dict)]}
    for key in ("mergeable", "mergeable_state", "additions", "deletions", "changed_files", "commits", "review_comments", "comments"):
        if key in it:
            item[key] = it[key]
    if full_body:
        read.body(it.get("body"), item)
    else:
        _preview(read, item, it.get("body"))
    return item


def _comment_shape(read: _Read, c: dict) -> dict:
    item = {"id": c.get("id"), "author": _login(c), "author_type": (c.get("user") or {}).get("type"),
            "created_at": c.get("created_at"), "updated_at": c.get("updated_at"), "url": c.get("html_url")}
    for key in ("path", "line", "side", "in_reply_to_id", "pull_request_review_id"):
        if key in c:
            item[key] = c[key]
    if c.get("issue_url"):
        item["issue_number"] = int(str(c["issue_url"]).rsplit("/", 1)[-1]) if str(c["issue_url"]).rsplit("/", 1)[-1].isdigit() else None
    if c.get("pull_request_url"):
        item["pr_number"] = int(str(c["pull_request_url"]).rsplit("/", 1)[-1]) if str(c["pull_request_url"]).rsplit("/", 1)[-1].isdigit() else None
    read.body(c.get("body"), item)
    return item


def _review_shape(read: _Read, r: dict) -> dict:
    item = {"id": r.get("id"), "author": _login(r), "state": r.get("state"), "submitted_at": r.get("submitted_at"),
            "commit_id": r.get("commit_id"), "url": r.get("html_url")}
    read.body(r.get("body"), item)
    return item


def _file_shape(read: _Read, f: dict) -> dict:
    return {k: f.get(k) for k in ("filename", "status", "additions", "deletions", "changes", "previous_filename") if k in f}


def _commit_shape(read: _Read, c: dict) -> dict:
    commit = c.get("commit") or {}
    author = commit.get("author") or {}
    return {"sha": c.get("sha"), "author_name": author.get("name"), "author_email": author.get("email"),
            "authored_at": author.get("date"), "message_headline": str(commit.get("message") or "").partition("\n")[0],
            "url": c.get("html_url")}


def _notification_shape(read: _Read, n: dict) -> dict:
    subject = n.get("subject") or {}
    return {"id": n.get("id"), "reason": n.get("reason"), "unread": n.get("unread"), "updated_at": n.get("updated_at"),
            "subject": subject.get("title"), "subject_type": subject.get("type"), "subject_url": subject.get("url"),
            "repo": (n.get("repository") or {}).get("full_name")}


def _release_shape(read: _Read, r: dict) -> dict:
    item = {"id": r.get("id"), "tag": r.get("tag_name"), "name": r.get("name"), "draft": r.get("draft"),
            "prerelease": r.get("prerelease"), "author": _login(r, "author"), "created_at": r.get("created_at"),
            "published_at": r.get("published_at"), "url": r.get("html_url"),
            "assets": [str(a.get("name")) for a in r.get("assets") or [] if isinstance(a, dict)]}
    read.body(r.get("body"), item)
    return item


def _label_shape(read: _Read, l: dict) -> dict:
    return {"name": l.get("name"), "color": l.get("color"), "description": l.get("description"), "default": l.get("default")}


def _discussion_shape(read: _Read, d: dict, *, full_body: bool = False) -> dict:
    item = {"id": d.get("id"), "number": d.get("number"), "title": d.get("title"), "author": _login(d, "author"),
            "category": (d.get("category") or {}).get("name"), "answered": d.get("isAnswered"),
            "created_at": d.get("createdAt"), "updated_at": d.get("updatedAt"),
            "comments": (d.get("comments") or {}).get("totalCount"), "url": d.get("url")}
    if d.get("repository"):
        item["repo"] = d["repository"].get("nameWithOwner")
    if "locked" in d:
        item["locked"] = d["locked"]
    if "body" in d:
        if full_body:
            read.body(d.get("body"), item)
        else:
            _preview(read, item, d.get("body"))
    return item


def _discussion_comment_shape(read: _Read, c: dict) -> dict:
    replies = c.get("replies") or {}
    item = {"id": c.get("id"), "database_id": c.get("databaseId"), "author": _login(c, "author"), "created_at": c.get("createdAt"),
            "updated_at": c.get("updatedAt"), "url": c.get("url"), "is_answer": c.get("isAnswer"),
            "replies_total": replies.get("totalCount")}
    read.body(c.get("body"), item)
    item["replies"] = [{"id": r.get("id"), "database_id": r.get("databaseId"), "author": _login(r, "author"),
                        "created_at": r.get("createdAt"), "url": r.get("url"), "body": r.get("body")}
                       for r in replies.get("nodes") or []]
    return item


def _thread_shape(read: _Read, t: dict) -> dict:
    comments = t.get("comments") or {}
    item = {"id": t.get("id"), "resolved": t.get("isResolved"), "outdated": t.get("isOutdated"), "path": t.get("path"),
            "line": t.get("line"), "start_line": t.get("startLine"), "side": t.get("diffSide"),
            "comments_total": comments.get("totalCount"),
            "comments": [{"id": c.get("databaseId"), "node_id": c.get("id"), "author": _login(c, "author"),
                          "created_at": c.get("createdAt"), "url": c.get("url"), "body": c.get("body")}
                         for c in comments.get("nodes") or []]}
    return item


# ---- methods ---------------------------------------------------------------------------------------
def _whoami(read: _Read) -> str:
    repo = read.repo
    host = (_split_repo(repo) or ("", "", ""))[0] if repo else ""
    args = ["api", "--include", "user"] + (["--hostname", host] if host else [])
    res = _gh_run(args, read.ctx)
    if not res.ok:
        return res.text
    status, headers, body = _split_http(res.text)
    try:
        user = json.loads(body)
    except ValueError:
        return read.refuse(f"⚠️ TOOL_ERROR: failed to parse GitHub's user JSON: {body[:300]}", "TOOL_ERROR")
    token = github_token_from_env_or_settings()
    kinds = (("ghp_", "classic personal access token"), ("github_pat_", "fine-grained personal access token"),
             ("gho_", "OAuth token (gh login)"), ("ghs_", "GitHub App installation token"), ("ghu_", "GitHub App user token"))
    token_kind = next((label for prefix, label in kinds if token.startswith(prefix)), "token of unknown form" if token else "gh CLI login")
    import os

    source = ("env GITHUB_TOKEN" if os.environ.get("GITHUB_TOKEN") else "env GH_TOKEN" if os.environ.get("GH_TOKEN")
              else "settings GITHUB_TOKEN" if token else "gh CLI host configuration")
    scopes = headers.get("x-oauth-scopes", "")
    item = {"login": user.get("login"), "id": user.get("id"), "type": user.get("type"), "name": user.get("name"),
            "url": user.get("html_url"), "token_kind": token_kind, "token_source": source,
            "scopes": [s.strip() for s in scopes.split(",") if s.strip()] if scopes else None,
            "token_expires": headers.get("github-authentication-token-expiration") or None, "host": host or "github.com"}
    return read.deliver(f"github_read whoami: @{item['login']} ({token_kind} from {source})", {"item": read.project(item)})


def _get(read: _Read) -> str:
    kind, number = read.kind, read.number()
    if kind == "run":
        from ouroboros.tools.github_workflows import get_run

        return get_run(read)
    include = [str(f) for f in (read.args.get("include") or [])]
    if "log" in include:
        return read.refuse("⚠️ TOOL_ARG_ERROR: include=['log'] requires kind='run'.")
    unknown = [f for f in include if f not in FACETS]
    if unknown:
        return read.refuse(f"⚠️ TOOL_ARG_ERROR: unknown include facet(s) {', '.join(unknown)}; facets: {', '.join(FACETS)}.")
    if read.cursor and len(include) != 1:
        return read.refuse("⚠️ TOOL_ARG_ERROR: a cursor on get pages exactly one include facet; pass one facet, or list(kind=…, target=…).")
    if kind in ("issue", "pr"):
        if number is None:
            return read.refuse(f"⚠️ TOOL_ARG_ERROR: {kind} number must be positive.")
        route = f"repos/{{owner}}/{{repo}}/{'issues' if kind == 'issue' else 'pulls'}/{number}"
        res = _gh_api(read.ctx, route, repo=read.repo)
        if not res.ok:
            return res.text
        try:
            raw = json.loads(res.text)
            assert isinstance(raw, dict)
        except (ValueError, AssertionError):
            return read.refuse(f"⚠️ TOOL_ERROR: failed to parse {kind} JSON: {res.text[:300]}", "TOOL_ERROR")
        item = dict(raw) if read.detail == "full" else (_issue_shape if kind == "issue" else _pr_shape)(read, raw, full_body=True)
        if kind == "issue" and "pull_request" in raw:
            read.omitted.append(f"#{number} is a pull request; get(kind='pr') adds its PR facets (files, diff, commits, reviews, threads)")
        facets = {}
        for facet in include:
            if kind == "issue" and facet != "comments":
                read.omitted.append(f"facet {facet} applies to kind='pr'")
                continue
            page, error = _facet(read, number, facet)
            if error:
                return error
            facets[facet] = page
        if kind == "pr":
            facts, error = _pr_facts(read, [raw])
            if error:
                return error
            item["review_decision"] = facts[raw["node_id"]].get("reviewDecision")
        payload: Dict[str, Any] = {"target": number, "item": read.project(item)}
        if facets:
            payload["facets"] = facets
        label = f"{kind} #{number} in {_repo_of(raw.get('html_url')) or read.repo or 'selected Project'}"
        return read.deliver(f"github_read get {label}: {str(raw.get('title') or '')[:80]!r} ({raw.get('state')}"
                            + (f", {len(facets)} facet(s)" if facets else "") + ")", payload)
    if kind in ("comment", "review_comment"):
        if number is None:
            return read.refuse(f"⚠️ TOOL_ARG_ERROR: target must be the {kind} id.")
        route = f"repos/{{owner}}/{{repo}}/{'issues' if kind == 'comment' else 'pulls'}/comments/{number}"
        return _one(read, route, _comment_shape, f"{kind} {number}")
    if kind == "release":
        tag = str(read.args.get("target") or "").strip()
        if not tag:
            return read.refuse("⚠️ TOOL_ARG_ERROR: target must be the release tag.")
        return _one(read, f"repos/{{owner}}/{{repo}}/releases/tags/{tag}", _release_shape, f"release {tag}")
    if kind == "label":
        name = str(read.args.get("target") or "").strip()
        if not name:
            return read.refuse("⚠️ TOOL_ARG_ERROR: target must be the label name.")
        return _one(read, f"repos/{{owner}}/{{repo}}/labels/{name}", _label_shape, f"label {name}")
    if kind == "discussion":
        if number is None:
            return read.refuse("⚠️ TOOL_ARG_ERROR: discussion number must be positive.")
        data, error = read.graphql("discussion", {"number": number})
        if error:
            return error
        node = (data.get("repository") or {}).get("discussion")
        if not isinstance(node, dict):
            return read.refuse(f"⚠️ GH_ERROR: discussion #{number} was not found (HTTP 404).", "TOOL_ERROR")
        item = dict(node) if read.detail == "full" else _discussion_shape(read, node, full_body=True)
        return read.deliver(f"github_read get discussion #{number}: {str(node.get('title') or '')[:80]!r}",
                            {"target": number, "item": read.project(item)})
    if kind == "review":
        return read.refuse("⚠️ TOOL_ARG_ERROR: a review is read in its pull request: list(kind='review', target=<pr number>).")
    if kind == "notification":
        return read.refuse("⚠️ TOOL_ARG_ERROR: notifications are listed: list(kind='notification').")
    return read.refuse(f"⚠️ TOOL_ARG_ERROR: get needs kind ∈ {', '.join(KINDS)}.")


def _one(read: _Read, route: str, shape, label: str) -> str:
    res = _gh_api(read.ctx, route, repo=read.repo)
    if not res.ok:
        return res.text
    try:
        raw = json.loads(res.text)
        assert isinstance(raw, dict)
    except (ValueError, AssertionError):
        return read.refuse(f"⚠️ TOOL_ERROR: failed to parse {label} JSON: {res.text[:300]}", "TOOL_ERROR")
    item = dict(raw) if read.detail == "full" else shape(read, raw)
    return read.deliver(f"github_read get {label} in {_repo_of(raw.get('html_url')) or read.repo or 'selected Project'}",
                        {"target": read.args.get("target"), "item": read.project(item)})


def _facet(read: _Read, number: int, facet: str) -> Tuple[Optional[dict], str]:
    """The first page (or the cursor's page) of one facet of a pull request or issue."""
    if facet == "diff":
        res = _gh_run(["pr", "diff", str(number)], read.ctx, timeout=60, repo=read.repo)
        if not res.ok:
            return None, res.text
        text = res.text
        page: Dict[str, Any] = {"chars": len(text), "complete": True}
        if read.detail != "full" and len(text) > _DIFF_CHARS:
            page.update(text=text[:_DIFF_CHARS], truncated=True)
            read.omitted.append(f"diff cut at {_DIFF_CHARS} chars (of {len(text)}): detail='full' returns it whole")
        else:
            page["text"] = text
        return page, ""
    if facet == "threads":
        data, error = read.graphql("threads", {"number": number, "first": read.limit, "after": read.cursor or None})
        if error:
            return None, error
        node = ((data.get("repository") or {}).get("pullRequest") or {}).get("reviewThreads") or {}
        nodes, facts = read.connection(node)
        return {"items": read.rows(nodes, lambda t: _thread_shape(read, t)), **facts}, ""
    route, shape = {
        "comments": (f"repos/{{owner}}/{{repo}}/issues/{number}/comments", _comment_shape),
        "review_comments": (f"repos/{{owner}}/{{repo}}/pulls/{number}/comments", _comment_shape),
        "reviews": (f"repos/{{owner}}/{{repo}}/pulls/{number}/reviews", _review_shape),
        "files": (f"repos/{{owner}}/{{repo}}/pulls/{number}/files", _file_shape),
        "commits": (f"repos/{{owner}}/{{repo}}/pulls/{number}/commits", _commit_shape),
    }[facet]
    rows, facts, error = read.page(route, {"since": read.since} if facet in ("comments", "review_comments") else None)
    if error:
        return None, error
    if not isinstance(rows, list):
        return None, read.refuse(f"⚠️ TOOL_ERROR: GitHub answered the {facet} facet with no list.", "TOOL_ERROR")
    page = {"items": read.rows(rows, lambda r: shape(read, r)), **facts}
    if not facts["complete"]:
        page["more"] = f"list(kind={'review_comment' if facet == 'review_comments' else facet.rstrip('s')!r}, target={number}, cursor={facts['next_cursor']!r})" \
            if facet in ("comments", "review_comments", "reviews") else f"get(include=[{facet!r}], cursor={facts['next_cursor']!r})"
    return page, ""


def _pr_facts(read: _Read, rows: list) -> Tuple[Optional[dict], str]:
    """Exact commit counts and review decisions of listed PRs by node id, or the error that withholds the list.

    The PR URLs GitHub returned name the one host and repository the single fixed GraphQL read then asks.
    A count is never guessed; a list without counts is not shown."""
    from ouroboros.merge_receipts import _PR_URL_RE

    def unavailable(reason: str) -> Tuple[None, str]:
        return None, read.refuse(f"⚠️ TOOL_ERROR: commit counts of the listed PRs are unavailable ({reason}); "
                                 "no list is shown without them.", "TOOL_ERROR")

    pairs = [(pr, _PR_URL_RE.fullmatch(str(pr.get("html_url") or "")) if isinstance(pr, dict) else None) for pr in rows]
    if any(where is None or where[4] != str(pr.get("number")) or not isinstance(pr.get("node_id"), str) for pr, where in pairs) \
            or len({where.groups()[:3] for _pr, where in pairs}) != 1:
        return unavailable("GitHub returned rows without one consistent repository URL and id")
    ids, host = [pr["node_id"] for pr, _where in pairs], pairs[0][1].group(1)
    if len(set(ids)) != len(ids):
        return unavailable("GitHub returned a repeated PR id")
    res = _gh_run(["api", "graphql", "--hostname", host, "--input", "-"], read.ctx,
                  input_data=json.dumps({"query": _PR_COMMIT_COUNTS_QUERY, "variables": {"ids": ids}}))
    if not res.ok:
        return None, res.text
    try:
        nodes = json.loads(res.text)["data"]["nodes"]
        facts = {node["id"]: node for node in nodes}
        counts = [node["commits"]["totalCount"] for node in nodes]
    except (ValueError, TypeError, KeyError):
        return unavailable("the count response is malformed")
    if len(facts) != len(nodes) or set(facts) != set(ids) or any(type(c) is not int or c < 0 for c in counts):
        return unavailable("the count response does not hold one count for every listed PR")
    return facts, ""


def _pr_items(read: _Read, rows: list) -> Tuple[Optional[list], str]:
    """The same exact PR facts for REST and search-backed list pages."""
    if not rows:
        return [], ""
    facts, error = _pr_facts(read, rows)
    if error:
        return None, error
    items = []
    for row in rows:
        node = facts[row["node_id"]]
        enriched = {**row, "commits": node["commits"]["totalCount"]}
        head = dict(row.get("head") or {})
        for field, key in (("headRefName", "ref"), ("headRefOid", "sha")):
            if node.get(field) is not None:
                head[key] = node[field]
        owner = (node.get("headRepositoryOwner") or {}).get("login")
        if head.get("ref") is not None and (owner is not None or node.get("headRefName") is not None):
            if owner is None:
                owner = str(head.get("label") or "").rpartition(":")[0]
            head["label"] = f"{owner}:{head['ref']}" if owner else head["ref"]
        if head:
            enriched["head"] = head
        if node.get("baseRefName") is not None:
            enriched["base"] = {**(row.get("base") or {}), "ref": node["baseRefName"]}
        if "headRefName" in node:
            enriched.update(draft=node.get("isDraft"), merged_at=node.get("mergedAt"))
        item = enriched if read.detail == "full" else _pr_shape(read, enriched, full_body=False)
        item["review_decision"] = node.get("reviewDecision")
        items.append(read.project(item))
    return items, ""


def _list(read: _Read) -> str:
    kind, number, state = read.kind, read.number(), str(read.args.get("state") or "open")
    if kind == "artifact":
        from ouroboros.tools.github_workflows import list_artifacts

        return list_artifacts(read)
    if kind == "run":
        return read.refuse("⚠️ TOOL_ARG_ERROR: get(kind='run', target=run_id) reads a run; checks lists a commit's runs.")
    if kind not in KINDS:
        return read.refuse(f"⚠️ TOOL_ARG_ERROR: list needs kind ∈ {', '.join(KINDS)}.")
    if kind == "discussion":
        return _list_discussions(read, number)
    if kind == "pr" and (state == "merged" or read.args.get("labels") or read.args.get("author")):
        parts = [f"is:pr is:{state}" if state != "all" else "is:pr"] + [f"label:\"{l.strip()}\"" for l in str(read.args.get("labels") or "").split(",") if l.strip()]
        if read.args.get("author"):
            parts.append(f"author:{read.args['author']}")
        read.args["query"] = " ".join(parts)
        return _search(read)
    params: Dict[str, Any] = {}
    label, shape, key = kind, None, "updated_at"
    if kind == "issue":
        route, shape = "repos/{owner}/{repo}/issues", lambda r: _issue_shape(read, r, full_body=False)
        params = {"state": state, "labels": read.args.get("labels") or "", "creator": read.args.get("author") or "",
                  "since": read.since, "sort": "updated", "direction": "desc"}
        label = f"issue ({state})"
    elif kind == "pr":
        route, shape = "repos/{owner}/{repo}/pulls", lambda r: _pr_shape(read, r, full_body=False)
        params = {"state": state, "sort": "updated", "direction": "desc"}
        label = f"pr ({state})"
    elif kind == "comment":
        route = f"repos/{{owner}}/{{repo}}/issues/{number}/comments" if number else "repos/{owner}/{repo}/issues/comments"
        shape, params = lambda r: _comment_shape(read, r), {"since": read.since}
        if not number:
            params.update(sort="updated", direction="asc")
        label = f"comment on #{number}" if number else "comment (repo-wide, issues and PR conversations)"
    elif kind == "review_comment":
        route = f"repos/{{owner}}/{{repo}}/pulls/{number}/comments" if number else "repos/{owner}/{repo}/pulls/comments"
        shape, params = lambda r: _comment_shape(read, r), {"since": read.since}
        if not number:
            params.update(sort="updated", direction="asc")
        label = f"review_comment on PR #{number}" if number else "review_comment (repo-wide inline review comments)"
    elif kind == "review":
        if number is None:
            return read.refuse("⚠️ TOOL_ARG_ERROR: list(kind='review') needs target = the pull request number.")
        route, shape, label = f"repos/{{owner}}/{{repo}}/pulls/{number}/reviews", lambda r: _review_shape(read, r), f"review on PR #{number}"
        key = "submitted_at"
    elif kind == "notification":
        route, shape = "repos/{owner}/{repo}/notifications", lambda r: _notification_shape(read, r)
        params = {"since": read.since, "all": "true" if state == "all" else ""}
        label = "notification" + (" (all)" if state == "all" else " (unread)")
    elif kind == "release":
        route, shape, key = "repos/{owner}/{repo}/releases", lambda r: _release_shape(read, r), "published_at"
    else:
        route, shape, key = "repos/{owner}/{repo}/labels", lambda r: _label_shape(read, r), ""
    rows, facts, error = read.page(route, params)
    if error:
        return error
    if not isinstance(rows, list):
        return read.refuse(f"⚠️ TOOL_ERROR: GitHub answered {kind} with no list: {str(rows)[:300]}", "TOOL_ERROR")
    if kind != "pr":  # a PR page keeps every row for the identity check of the count read
        rows = [r for r in rows if isinstance(r, dict)]
    if kind == "issue":
        prs = sum("pull_request" in r for r in rows)
        if prs:
            read.omitted.append(f"{prs} pull request(s) on this page not shown; list(kind='pr') reads them")
        rows = [r for r in rows if "pull_request" not in r]
    since_filtered = bool(params.get("since"))  # The successful request applied a server-side filter.
    if kind in ("pr", "review", "release") and read.since and key:
        rows = read.since_filter(rows, key, facts, newest_first=kind == "pr")
        since_filtered = True
    payload: Dict[str, Any] = {"target": number} if number else {}
    if kind == "pr" and rows:
        payload["items"], error = _pr_items(read, rows)
        if error:
            return error
    else:
        payload["items"] = read.rows(rows, shape)
    payload.update(facts)
    if read.since and not since_filtered:
        payload["note"] = "The labels endpoint has no since filter; this page is unfiltered."
    if kind == "notification" and not payload["items"] and facts["complete"]:
        read.omitted.append("an empty notification list under a fine-grained token or GitHub App is a permission gap: notifications need a classic PAT with the notifications scope")
    repo = _repo_of((rows[0].get("html_url") if rows else "") or (rows[0].get("url") if rows else "")) or read.repo or "selected Project"
    more = f", more: cursor={facts['next_cursor']!r}" if not facts["complete"] else ", complete"
    return read.deliver(f"github_read list {label} in {repo}: {len(payload['items'])} item(s)"
                        + (f" since {read.since}" if since_filtered else "") + more, payload)


def _list_discussions(read: _Read, number: Optional[int]) -> str:
    variables = {"first": read.limit, "after": read.cursor or None}
    if number:
        data, error = read.graphql("discussion_comments", {**variables, "number": number})
        if error:
            return error
        node = (data.get("repository") or {}).get("discussion")
        if not isinstance(node, dict):
            return read.refuse(f"⚠️ GH_ERROR: discussion #{number} was not found (HTTP 404).", "TOOL_ERROR")
        nodes, facts = read.connection(node.get("comments") or {})
        nodes = read.since_filter(nodes, "updatedAt", facts) if read.since else nodes
        payload = {"target": number, "items": read.rows(nodes, lambda c: _discussion_comment_shape(read, c)), **facts}
        label = f"discussion #{number} comments (with replies)"
    else:
        data, error = read.graphql("discussions", variables)
        if error:
            return error
        nodes, facts = read.connection((data.get("repository") or {}).get("discussions") or {})
        nodes = read.since_filter(nodes, "updatedAt", facts, newest_first=True)
        payload = {"items": read.rows(nodes, lambda d: _discussion_shape(read, d)), **facts}
        label = "discussion"
    more = f", more: cursor={facts['next_cursor']!r}" if not facts["complete"] else ", complete"
    return read.deliver(f"github_read list {label} in {read.repo or 'selected Project'}: {len(payload['items'])} item(s)"
                        + (f" since {read.since}" if read.since else "") + more, payload)


def _search(read: _Read) -> str:
    kind, query = read.kind, str(read.args.get("query") or "").strip()
    if kind not in ("issue", "pr", "discussion"):
        return read.refuse("⚠️ TOOL_ARG_ERROR: search needs kind ∈ issue, pr, discussion and a query.")
    if not query:
        return read.refuse("⚠️ TOOL_ARG_ERROR: search needs a query (GitHub search syntax, e.g. 'is:open label:bug').")
    slug, error = read.slug()
    if error:
        return error
    q = f"{query} repo:{slug}"
    if read.since:
        q += f" updated:>={read.since[:10]}"
    if kind == "discussion":
        data, error = read.graphql("search_discussions", {"q": q, "first": read.limit, "after": read.cursor or None}, repo_vars=False)
        if error:
            return error
        node = data.get("search") or {}
        nodes, facts = read.connection(node)
        if isinstance(node.get("discussionCount"), int):
            facts["total_count"] = node["discussionCount"]
        payload = {"query": q, "items": read.rows(nodes, lambda d: _discussion_shape(read, d)), **facts}
    else:
        q += " is:pr" if kind == "pr" else " is:issue"
        rows, facts, error = read.page("search/issues", {"q": q, "advanced_search": "true"})
        if error:
            return error
        if not isinstance(rows, dict) or not isinstance(rows.get("items"), list):
            return read.refuse(f"⚠️ TOOL_ERROR: GitHub's search answer holds no items: {str(rows)[:300]}", "TOOL_ERROR")
        if read.method == "list" and kind == "pr":
            items, error = _pr_items(read, rows["items"])
            if error:
                return error
        else:
            items = read.rows(rows["items"], lambda r: _issue_shape(read, r, full_body=False))
        payload = {"query": q, "items": items, **facts}
        if isinstance(rows.get("total_count"), int):
            payload["total_count"] = rows["total_count"]
        if rows.get("incomplete_results"):
            read.omitted.append("GitHub reports incomplete_results: the search timed out before covering everything; narrow the query")
    total = f" of {payload['total_count']}" if "total_count" in payload else ""
    more = f", more: cursor={facts['next_cursor']!r}" if not facts["complete"] else ""
    return read.deliver(f"github_read {read.method} {kind} in {slug}: {len(payload['items'])} item(s){total} for {q!r}{more}", payload)


def _checks(read: _Read) -> str:
    from ouroboros.tools.github_checks import get_checks

    # A string that is not a number is a commit SHA for the reader to validate; anything else is a PR number.
    target = read.args.get("target")
    sha = target.strip().lower() if isinstance(target, str) and target.strip() and not target.strip().isdigit() else ""
    return get_checks(read.ctx, number=0 if sha else target, sha=sha, wait_seconds=read.args.get("wait_seconds") or 0, repo=read.repo)


def _api(read: _Read) -> str:
    route, document = str(read.args.get("route") or "").strip(), str(read.args.get("graphql") or "").strip()
    if bool(route) == bool(document):
        return read.refuse("⚠️ TOOL_ARG_ERROR: api takes exactly one of route (a REST GET path) or graphql (a query document).")
    if document:
        if "mutation" in _graphql_operations(document):
            return read.refuse("⚠️ GH_API_READ_ONLY: github_read(method='api') sends GET routes and GraphQL queries only; "
                               "a mutation goes through github_write(method='api', graphql=...).")
        variables = read.args.get("variables") or {}
        if not isinstance(variables, dict):
            return read.refuse("⚠️ TOOL_ARG_ERROR: variables must be an object.")
        res = _gh_graphql(read.ctx, document, variables, repo=read.repo, repo_vars="$owner" in document and "$name" in document)
        label = "graphql query"
    else:
        if route.split("?", 1)[0].split()[0].upper() in ("POST", "PATCH", "PUT", "DELETE") or " " in route.split("?", 1)[0]:
            return read.refuse("⚠️ GH_API_READ_ONLY: github_read(method='api') sends GET routes only; a REST mutation goes "
                               "through github_write(method='api', http_method=..., route=...).")
        res = _gh_api(read.ctx, route, repo=read.repo, include=True, timeout=60)
        label = f"GET {route}"
    if not res.ok:
        return res.text
    _status, headers, body = _split_http(res.text) if route else (None, {}, res.text)
    try:
        data = json.loads(body)
    except ValueError:
        data = body
    payload: Dict[str, Any] = {"request": label, "data": data if not read.fields or not isinstance(data, dict) else read.project(data)}
    if route:
        link = _LINK_NEXT_RE.search(headers.get("link", ""))
        payload.update(complete=link is None, next_cursor=link.group(1) if link else "")
        if link:
            payload["more"] = f"github_read(method='api', route={link.group(1)!r})"
    if isinstance(data, dict) and data.get("errors"):
        payload["errors"] = data["errors"]
    count = f"{len(data)} item(s)" if isinstance(data, list) else f"{len(body)} chars"
    return read.deliver(f"github_read api {label} in {read.repo or 'selected Project'}: {count}", payload)


def run_read(ctx: ToolContext, method: str = "", repo: str = "", **args: Any) -> str:
    """Dispatch one ``github_read`` call; the registry has already refused unknown arguments."""
    if not isinstance(repo, str):
        return _refuse(ctx, "⚠️ GH_TARGET_INVALID: repo must be a string; omit it to use the selected Project.", no_effect=_no_launch_yet())
    args = _canonical_github_args({"method": method, **args})
    method = args.pop("method")
    if method not in METHODS:
        return _refuse(ctx, f"⚠️ TOOL_ARG_ERROR: method must be one of {', '.join(METHODS)}.", no_effect=_no_launch_yet())
    read = _Read(ctx, method, repo, **args)
    if read.limit is None or read.limit < 1:
        return read.refuse("⚠️ TOOL_ARG_ERROR: limit is a whole number from 1 to 100.")
    if read.detail not in ("concise", "full"):
        return read.refuse("⚠️ TOOL_ARG_ERROR: detail is 'concise' or 'full'.")
    if read.kind and read.kind not in KINDS:
        return read.refuse(f"⚠️ TOOL_ARG_ERROR: kind must be one of {', '.join(KINDS)}.")
    if method in ("get", "list", "search", "download") and not read.kind:
        return read.refuse(f"⚠️ TOOL_ARG_ERROR: {method} needs kind ∈ {', '.join(KINDS)}.")
    if method == "download":
        from ouroboros.tools.github_workflows import download

        return download(read)
    return {"whoami": _whoami, "get": _get, "list": _list, "search": _search, "checks": _checks, "api": _api}[method](read)


SCHEMA: Dict[str, Any] = {
    "name": "github_read",
    "description": (
        "Read GitHub (issues, pull requests, discussions, comments, reviews, checks, notifications, releases, labels) for "
        "the active Project or an explicit repo. Read-only; concise by default, detail='full' for whole objects, fields=[…] "
        "to keep only some keys; every cut is stated (body_chars, omitted) and a result too large for one answer is saved "
        "whole as a task artifact. Methods: whoami = the acting login, token kind and scopes; get = one object by kind and "
        "target (issue/pr number, comment id, release tag, label name, discussion number) with optional include facets "
        "(pr: comments, review_comments, reviews, threads, files, diff, commits; issue: comments); list = a page of a kind "
        "with since + cursor + complete (kind='comment' is the conversation thread of issues AND pull requests, repo-wide "
        "when target is omitted; kind='review_comment' is inline code review comments, repo-wide when target is omitted; "
        "kind='review' is review submissions of one PR; kind='discussion' lists discussions, or one discussion's comments "
        "with replies when target is set); search = GitHub search syntax in query, kind issue|pr|discussion, with "
        "total_count; checks = the recorded checks of one commit (target = PR number or full SHA, optional wait_seconds); "
        "api = any REST GET route (query string allowed, {owner}/{repo} placeholders follow repo) or a GraphQL query, "
        "never a mutation (that is github_write(method='api')). get kind='run', target=run_id, include=['log'] reads "
        "a completed run's log (optional job_id, attempt, failed_only, max_bytes, tail_lines); exact bytes, size and sha256 "
        "are retained as task artifacts before a bounded preview. The stdout cap does not bound gh's internal cache. "
        "list kind='artifact', target=run_id pages artifacts using limit/cursor; download kind='artifact', target=artifact_id "
        "requires max_bytes and retains the exact zip without extraction. Writes are github_write; merges are pr_merge."
    ),
    "parameters": {"type": "object", "properties": {
        "method": {"type": "string", "enum": list(METHODS),
                   "description": "whoami | get | list | search | checks | download | api"},
        "kind": {"type": "string", "enum": list(KINDS),
                 "description": "Object kind for get/list/search: issue, pr, discussion, comment (conversation), review_comment "
                                "(inline code review), review, notification, release, label, run, artifact"},
        "target": {"type": ["integer", "string"], "default": "",
                   "description": "issue/PR/discussion number; comment id (get kind=comment|review_comment); release tag; label name; "
                                  "PR number or 40-hex commit SHA for checks; run id for get run/list artifact, artifact id for download"},
        "include": {"type": "array", "items": {"type": "string", "enum": list(FACETS)}, "default": [],
                    "description": "get facets, first page each: comments, review_comments, reviews, threads (review threads with ids "
                                   "for github_write resolve_thread), files, diff, commits; run: log"},
        "job_id": {"type": "integer", "default": 0, "description": "Run log: job id belonging to the selected run/attempt; omitted reads all jobs"},
        "attempt": {"type": "integer", "default": 0, "description": "Run: 0 selects latest; logs pin the observed attempt before capture"},
        "failed_only": {"type": "boolean", "default": False, "description": "Run log: failed steps only; empty when none failed"},
        "max_bytes": {"type": "integer", "description": "Run log: default 4 MiB, ceiling 32 MiB; artifact download: required explicit cap, ceiling 64 MiB"},
        "tail_lines": {"type": "integer", "default": 120, "description": "Run log preview: last 1–2000 lines, clipped to 400 chars each; read_file pages retained text"},
        "since": {"type": "string", "default": "", "description": "ISO-8601 UTC instant for supported list/facet timestamps; labels have no since filter"},
        "cursor": {"type": "string", "default": "", "description": "next_cursor of the previous page (list, search, or get with one facet)"},
        "limit": {"type": "integer", "default": _LIMIT_DEFAULT, "minimum": 1, "maximum": _LIMIT_MAX, "description": "Page size, 1–100"},
        "detail": {"type": "string", "enum": ["concise", "full"], "default": "concise",
                   "description": "concise = short rows and bounded bodies with the cut stated; full = GitHub's whole objects"},
        "query": {"type": "string", "default": "", "description": "search: GitHub search syntax (is:open, label:bug, author:x, updated:>=…)"},
        "fields": {"type": "array", "items": {"type": "string"}, "default": [], "description": "Keep only these keys of each item"},
        "state": {"type": "string", "enum": ["open", "closed", "merged", "all"], "default": "open",
                  "description": "list issue|pr state (merged → search); notification: all = read ones too"},
        "labels": {"type": "string", "default": "", "description": "list filter: comma-separated label names"},
        "author": {"type": "string", "default": "", "description": "list filter: author login"},
        "wait_seconds": {"type": "integer", "default": 0,
                         "description": "checks: poll until every registered run completes or this many seconds pass (max 240)"},
        "route": {"type": "string", "default": "", "description": "api: REST GET route, e.g. repos/{owner}/{repo}/issues/7/timeline?per_page=100"},
        "graphql": {"type": "string", "default": "", "description": "api: GraphQL query document ($owner/$name are filled from repo when declared)"},
        "variables": {"type": "object", "default": {}, "description": "api: GraphQL variables"},
    }, "required": ["method"]},
}
