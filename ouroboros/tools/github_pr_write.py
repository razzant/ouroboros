"""PR methods of ``github_write``: creation, field edits and draft state.

Internal method owner: ``github_write`` dispatches these methods and publishes their shared receipt. Each call reads the
pull request, its head commit and (when it matters) its base commit first and refuses before any effect when they
differ from what the caller expected; then sends exactly ONE mutating ``gh`` command (a field edit is one REST PATCH
at the read pull request's own URL: ``gh pr edit`` also queries organization metadata a repo-scoped token cannot
read); then reads the pull request back and states, field by field, what GitHub holds. Edits and draft changes use
the first read's canonical repository and confirm only the same PR node. These checks are not atomic with the write.
Once a mutation was sent its effect is unknown until the read-back shows it: a failed or lost answer is reported with gh's exit facts beside that
read-back and never resent (a nonzero exit alone does not prove GitHub rejected the request, and a request GitHub is
still processing can land after the read). A 4xx from the direct create POST settles that request as not applied.
GitHub documents no all-or-nothing edit, so a field PATCH's 4xx settles only while the read-back holds every sent field as the precheck
read it. Nothing here pushes, forks or merges. ``skill_publish_github`` keeps its
token-only HTTP flow; the two share only the pure identity predicate in ``github_pr_identity``.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from typing import List, Optional, Tuple

from ouroboros.github_pr_identity import pull_request_fields_from_cli, same_pull_request
from ouroboros.tools.github import (
    GhResult, GhTarget, _ambient_repo, _gh_run, _http_refused, _refuse, _repo_hint, _repo_identity, _send_facts, _split_repo,
)
from ouroboros.tools.github_checks import _gh_json, _one_line, _whole
from ouroboros.tools.registry import ToolContext
from ouroboros.utils import utc_now_iso

_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
# A Git branch name this tool passes to gh: no whitespace or ref-forbidden characters, never flag-shaped.
_BRANCH_RE = re.compile(r"^(?!-)(?!.*\.\.)(?!.*@\{)[^\s~^:?*\[\\\x00-\x1f\x7f]{1,255}(?<!/)(?<!\.lock)$")
_LOGIN_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}$")
_PR_FIELDS = "number,url,state,title,body,isDraft,headRefName,baseRefName,headRefOid,headRepositoryOwner"
_READ_TIMEOUT, _WRITE_TIMEOUT, _LIST_LIMIT = 30, 60, 50


def _sha(value: object) -> str:
    text = str(value or "").strip().lower()
    return text if _FULL_SHA_RE.fullmatch(text) else ""


def _branch(value: object) -> str:
    text = str(value or "").strip() if isinstance(value, str) else ""
    return text if _BRANCH_RE.fullmatch(text) else ""


def _one_line_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and "\n" not in value and "\r" not in value


def _receipt(write, text: str, outcome: str, status: str, *, pr: Optional[dict] = None,
             readback: Optional[dict] = None, res: Optional[GhResult] = None) -> str:
    created = getattr(write, "pr_created", None)
    observed = pr or getattr(write, "pr_observed", None)
    pr = observed or created or {}
    res = res or getattr(write, "pr_send", None)
    number = pr.get("number") or (created or {}).get("number") or write.args.get("target")
    bound = getattr(write, "pr_repo", "") or write.repo
    parts = _split_repo(bound) if bound else None
    canonical = "/".join(parts[1:3]) if parts else ""
    target = f"{canonical or write.slug(pr.get('url') or (created or {}).get('html_url'))}#{number}" if number else (
        f"{canonical or write.slug()} {write.args.get('head')} → {write.args.get('base')}")
    effect = "pr.create" if write.method == "create_pr" else "pr.draft" if "draft" in write.args else "pr.update"
    if readback is None:
        readback = {"verified": False, "via": "lookup", "item": observed}
    # A lost current read keeps the attempt's confirmed identity (canonical repository, PR node) apart from the read it lacks.
    attempt = readback.get("attempt") or {}
    full = ", detail='full'" if attempt else ""
    lookup = (f"github_read(method='get', kind='pr', target={number}"
              f"{_repo_hint(bound)}{full})" if number else
              f"github_read(method='list', kind='pr'{_repo_hint(bound)})")
    return write.receipt(effect, target, response=created if created is not None else pr,
                         oid=(created or {}).get("node_id") or attempt.get("id") or number, actor=None, at=utc_now_iso(),
                         res=res, readback=readback, lookup=lookup, extra={"at_source": "local", "detail": text},
                         settlement={"completed": "confirmed", "completed_no_effect": "not_applied", "unknown": "unknown"}[outcome],
                         result_status=status)


def _ok(write, text: str, *, outcome: str = "completed", pr: Optional[dict] = None,
        readback: Optional[dict] = None, res: Optional[GhResult] = None) -> str:
    return _receipt(write, text, outcome, "ok", pr=pr, readback=readback, res=res)


def _failed(write, text: str, *, no_effect: bool = False, pr: Optional[dict] = None,
            readback: Optional[dict] = None, res: Optional[GhResult] = None) -> str:
    return _receipt(write, text, "completed_no_effect" if no_effect else "unknown", "error",
                    pr=pr, readback=readback, res=res)


def _precheck_failed(write, res: GhResult, what: str, nothing: str) -> str:
    """A read before the mutation failed, so nothing was sent; the transport's own refusals pass through."""
    if res.not_submitted:
        return write.unsent(res)
    return _failed(write, f"⚠️ PR_PRECHECK_FAILED: {what} could not be read ({res.text}); {nothing}.", no_effect=True)


def _where(repo: str) -> str:
    return repo or "the active Project repository"


def _reread(number: int, hint: str, pr: dict) -> str:
    """The one read of the current PR that settles a lost readback: the whole object, then its node_id against the attempt's."""
    return (f"github_read(method='get', kind='pr', target={number}{hint}, detail='full'), compare its node_id with "
            f"{pr['id']}, and resend nothing")


def _lost_readback(write, pr: dict, number: int, res: GhResult) -> dict:
    """The attempt's identity the precheck confirmed (canonical repository, PR node), kept apart from the current read
    that was lost: a missing read never turns into an applied outcome or drops which pull request was written."""
    return {"verified": False, "via": "lookup", "item": None, "current_read": f"missing: {res.text[:200]}",
            "attempt": {"repo": write.pr_repo, "id": pr["id"], "number": number, "url": pr["url"]}}


def _malformed(what: str, res: GhResult) -> GhResult:
    return GhResult(False, f"⚠️ GH_ERROR: {what} is malformed: {res.text[:300]}", None, None, "malformed")


def _pr_record(data: object, number: int) -> Tuple[Optional[dict], str]:
    """The pull request fields this writer relies on, each checked for type, or why GitHub's answer lacks one."""
    if not isinstance(data, dict):
        return None, "not a JSON object"
    checks = {
        "number": lambda value: _whole(value) == number,
        "url": lambda value: isinstance(value, str) and bool(value),
        "state": lambda value: isinstance(value, str) and bool(value),
        "title": lambda value: isinstance(value, str),
        "isDraft": lambda value: isinstance(value, bool),
        "headRefName": lambda value: isinstance(value, str) and bool(value),
        "baseRefName": lambda value: isinstance(value, str) and bool(value),
        "headRefOid": lambda value: bool(_sha(value)),
    }
    for key, ok in checks.items():
        if key not in data or not ok(data[key]):
            return None, f"field {key!r} is {data.get(key)!r}"
    if "body" not in data:
        return None, "field 'body' is missing"
    if not isinstance(data["body"], str):
        return None, "field 'body' is not text"
    owner = data.get("headRepositoryOwner")
    if not isinstance(owner, dict) or not isinstance(owner.get("login"), str) or not owner["login"]:
        return None, f"field 'headRepositoryOwner' is {owner!r}"
    return data, ""


def _resolve_repo(write) -> Tuple[Optional[GhTarget], str]:
    """Confirm the selected repository before deriving write routes from GitHub's canonical name."""
    ctx = write.ctx
    selector = write.repo or _ambient_repo(ctx)
    selected = _split_repo(selector)
    if selector and selected is None:
        return None, _refuse(ctx, "⚠️ GH_TARGET_INVALID: repo must be [HOST/]OWNER/REPO.", no_effect=True)
    res = _gh_run(["repo", "view", "--json", "nameWithOwner,url"], ctx, timeout=_READ_TIMEOUT, repo=selector)
    if not res.ok:
        return None, _precheck_failed(write, res, "the repository", "nothing was sent")
    info = _gh_json(res, dict) or {}
    actual = _repo_identity(info.get("nameWithOwner"), info.get("url"))
    if actual is None:
        return None, _failed(write, "⚠️ PR_PRECHECK_FAILED: GitHub's repository name and URL do not identify "
                                   "the same repository; nothing was sent.", no_effect=True)
    if selected is None:
        return actual, ""
    host, owner, name = selected
    actual_key = (actual.host.casefold(), actual.owner.casefold(), actual.name.casefold())
    if host and host.casefold() != actual_key[0]:
        return None, _failed(write, f"⚠️ PR_PRECHECK_FAILED: GitHub named host {actual.host}, not {host}; "
                                   "nothing was sent.", no_effect=True)
    if (owner.casefold(), name.casefold()) != actual_key[1:]:
        # A rename is accepted only if a GET addressed to the old name confirms the destination.
        route = "repos/" + "/".join(urllib.parse.quote(part, safe="") for part in (owner, name))
        args = ["api", route] + (["--hostname", host] if host else [])
        res = _gh_run(args, ctx, timeout=_READ_TIMEOUT, repo=selector)
        if not res.ok:
            return None, _precheck_failed(write, res, f"the repository alias {selector}", "nothing was sent")
        info = _gh_json(res, dict) or {}
        alias = _repo_identity(info.get("full_name"), info.get("html_url"))
        if alias is None or (alias.host.casefold(), alias.owner.casefold(), alias.name.casefold()) != actual_key:
            return None, _failed(write, f"⚠️ PR_PRECHECK_FAILED: GitHub did not confirm that {selector} resolves to "
                                       f"{actual.host}/{actual.owner}/{actual.name}; nothing was sent.", no_effect=True)
    return actual, ""


def _ref_sha(ctx: ToolContext, repo: GhTarget, owner: str, branch: str) -> Tuple[str, Optional[GhResult]]:
    """The commit a branch points at now, read from GitHub by the repository the run resolved; typed when malformed."""
    path = f"repos/{owner}/{repo.name}/git/ref/heads/{urllib.parse.quote(branch, safe='/')}"
    res = _gh_run(["api", path, "--hostname", repo.host], ctx, timeout=_READ_TIMEOUT)
    if not res.ok:
        return "", res
    data = _gh_json(res, dict)
    target = data.get("object") if isinstance(data, dict) else None
    sha = _sha(target.get("sha")) if isinstance(target, dict) else ""
    if not sha:
        return "", _malformed(f"the ref answer for {owner}/{repo.name} {branch}", res)
    return sha, None


def _view(ctx: ToolContext, number: int, repo: str, *, require_id: bool = False) -> Tuple[Optional[dict], Optional[GhResult]]:
    """The pull request as GitHub answers it, with every relied-on field validated; typed when malformed."""
    fields = _PR_FIELDS + ",id" if require_id else _PR_FIELDS
    res = _gh_run(["pr", "view", str(number), "--json", fields], ctx, timeout=_READ_TIMEOUT, repo=repo)
    if not res.ok:
        return None, res
    pr, why = _pr_record(_gh_json(res, dict), number)
    if pr is None:
        return None, _malformed(f"the pull request answer ({why})", res)
    if require_id and (not isinstance(pr.get("id"), str) or not pr["id"].strip()):
        return None, _malformed("the pull request answer (no pull request ID)", res)
    return pr, None


def _edit_view(write, number: int, *, same_id: str = "") -> Tuple[Optional[dict], Optional[GhTarget], Optional[GhResult]]:
    """Resolve once from the caller's selection; read back at the observed address and check identity."""
    from ouroboros.merge_receipts import _PR_URL_RE

    repo = getattr(write, "pr_repo", write.repo)
    pr, error = _view(write.ctx, number, repo, require_id=True)
    if pr is None:
        return None, None, error
    where = _PR_URL_RE.fullmatch(pr["url"])
    parts = _split_repo("/".join(where.groups()[:3])) if where else None
    selected = _split_repo(repo or _ambient_repo(write.ctx))
    if parts is None or int(where[4]) != number:
        why = "named no canonical HTTPS URL for that pull request"
    elif selected and selected[0] and parts[0].casefold() != selected[0].casefold():
        why = "named a different GitHub host"
    elif same_id and pr["id"] != same_id:
        why = "named a different pull request ID"
    else:
        target = GhTarget(*parts)
        write.pr_repo = f"{target.host}/{target.slug}"
        return pr, target, None
    return None, None, GhResult(False, f"⚠️ GH_ERROR: the read of PR #{number} {why}.", None, None, "malformed")


def _open_prs(ctx: ToolContext, repo: str, branch: str, base: str) -> Tuple[Optional[list], Optional[GhResult]]:
    """The open pull requests GitHub lists from a head branch name into a base; owners are told apart afterwards."""
    res = _gh_run(["pr", "list", "--state", "open", f"--head={branch}", f"--base={base}", "--limit", str(_LIST_LIMIT),
                   "--json", _PR_FIELDS], ctx, timeout=_READ_TIMEOUT, repo=repo)
    if not res.ok:
        return None, res
    rows = _gh_json(res, list)
    if rows is None:
        return None, _malformed("the pull request list answer", res)
    return [row for row in rows if isinstance(row, dict)], None


def _matching(rows: list, *, head_owner: str, head_branch: str, base: str, head_sha: Optional[str]) -> list:
    return [row for row in rows if same_pull_request(pull_request_fields_from_cli(row), head_owner=head_owner,
                                                     head_branch=head_branch, base_branch=base, head_sha=head_sha)]


def _pr_line(pr: dict) -> str:
    owner = pr.get("headRepositoryOwner") if isinstance(pr.get("headRepositoryOwner"), dict) else {}
    return (f"PR #{pr.get('number')} {pr.get('url') or ''}: {str(pr.get('state') or 'state unknown').lower()}, "
            f"{owner.get('login') or '?'}:{pr.get('headRefName') or '?'} → {pr.get('baseRefName') or '?'}, "
            f"head {pr.get('headRefOid') or 'unknown'}, draft={bool(pr.get('isDraft'))}, title {_one_line(pr.get('title'), 100)!r}")


def _normalized(text: object) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n").rstrip()


def _reconcile(pr: dict, *, title: Optional[str] = None, body: Optional[str] = None, base: Optional[str] = None,
               draft: Optional[bool] = None, head_owner: Optional[str] = None, head_branch: Optional[str] = None,
               head_sha: Optional[str] = None) -> Tuple[List[str], List[str], List[str]]:
    """What the read-back confirms, where it differs on a requested field, and notes on the head commit.

    The head commit is what the caller read, not what the call changes: its movement is reported, never counted
    as a differing field."""
    confirmed: List[str] = []
    differs: List[str] = []
    notes: List[str] = []
    if title is not None:
        held = pr.get("title")
        (confirmed if held == title else differs).append(
            "title confirmed" if held == title else f"title is {_one_line(held, 120)!r}, you sent {_one_line(title, 120)!r}")
    if body is not None:
        held = pr["body"]
        if held == body:
            confirmed.append(f"body confirmed ({len(body)} chars)")
        elif _normalized(held) == _normalized(body):
            confirmed.append(f"body confirmed after line-ending and trailing-whitespace normalization ({len(held)} chars held)")
        else:
            differs.append(f"body differs ({len(held)} chars held, {len(body)} sent)")
    if base is not None:
        held = pr.get("baseRefName")
        (confirmed if held == base else differs).append(
            f"base {base} confirmed" if held == base else f"base is {held!r}, you asked for {base!r}")
    if draft is not None:
        held = bool(pr.get("isDraft"))
        (confirmed if held == draft else differs).append(
            f"draft={held} confirmed" if held == draft else f"draft is {held}, you asked for {draft}")
    if head_branch is not None:
        owner = pr.get("headRepositoryOwner") if isinstance(pr.get("headRepositoryOwner"), dict) else {}
        held = f"{owner.get('login') or '?'}:{pr.get('headRefName') or '?'}"
        wanted = f"{head_owner}:{head_branch}"
        same = pr.get("headRefName") == head_branch and str(owner.get("login") or "").casefold() == str(head_owner or "").casefold()
        (confirmed if same else differs).append(f"head {wanted} confirmed" if same else f"head is {held}, you asked for {wanted}")
    if head_sha is not None:
        held = _sha(pr.get("headRefOid"))
        (confirmed if held == head_sha else notes).append(
            f"head at {head_sha} confirmed" if held == head_sha else f"head moved to {held} since you read it (you expected {head_sha})")
    return confirmed, differs, notes


def _changed_since(before: dict, after: dict, sent: dict) -> List[str]:
    """Sent fields whose read-back differs from the precheck read (a null body is empty), saying nothing of who changed them."""
    changed = []
    for key, field in (("title", "title"), ("body", "body"), ("base", "baseRefName")):
        old, new = before.get(field) or "", after.get(field) or ""
        if key in sent and old != new:
            changed.append(f"body changed ({len(old)} chars before, {len(new)} now)" if key == "body" else
                           f"{key} changed from {_one_line(old, 120)!r} to {_one_line(new, 120)!r}")
    return changed


def _readback_lines(confirmed: List[str], differs: List[str], notes: List[str]) -> List[str]:
    lines = ["Read-back: " + "; ".join(confirmed)] if confirmed else []
    if differs:
        lines.append("⚠️ Read-back differs: " + "; ".join(differs) + ". Read the pull request again before relying on it.")
    if notes:
        lines.append("Note: " + "; ".join(notes))
    return lines


def _settled(write, headline: str, pr: dict, **expected: object) -> str:
    """A CLI success is completed only when the separate readback holds every requested field."""
    confirmed, differs, notes = _reconcile(pr, **expected)
    return _ok(write, "\n".join([headline] + _readback_lines(confirmed, differs, notes)), pr=pr,
               readback={"verified": not differs, "mismatch": differs, "confirmed": confirmed, "notes": notes,
                         "via": "lookup", "item": pr}, outcome="unknown" if differs else "completed")


def create(write) -> str:
    """Open one pull request from an existing head branch, after the head and base read as the caller expected.
    The affirmative "created" claim rests on the provider's own response (its number, its node id and the canonical
    address of the resolved repository); a head that moved after the send is an observation, not a mismatch."""
    ctx, repo, args = write.ctx, write.repo, write.args
    title, head, base, body, draft = (args.get(k, default) for k, default in
                                     (("title", ""), ("head", ""), ("base", ""), ("body", ""), ("draft", False)))
    expected_head_sha, expected_base_sha = ((args.get("expect") or {}).get(k, "") for k in ("head_sha", "base_sha"))
    if not _one_line_text(title):
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: title is one non-empty line.", no_effect=True)
    head_arg = head.strip() if isinstance(head, str) else ""
    owner_part, _sep, branch_part = head_arg.rpartition(":")
    head_branch, base_arg = _branch(branch_part), _branch(base)
    if not head_branch or (owner_part and not _LOGIN_RE.fullmatch(owner_part)):
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: head is an existing branch name, or OWNER:BRANCH for a head held by another "
                            "account; this tool pushes and forks nothing.", no_effect=True)
    if not base_arg:
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: base is the branch name the change should merge into.", no_effect=True)
    head_sha, base_sha = _sha(expected_head_sha), _sha(expected_base_sha)
    if not head_sha or not base_sha:
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: expect.head_sha and expect.base_sha are the full 40-hex commits you read "
                            "for the head and base branches; the call refuses before any effect when GitHub holds others.",
                       no_effect=True)
    if head_sha == base_sha:
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: expect.head_sha equals expect.base_sha; there is nothing to request.",
                       no_effect=True)
    if not isinstance(body, str):
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: body is a string; an omitted body is an empty body.", no_effect=True)
    if not isinstance(draft, bool):
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: draft is a boolean.", no_effect=True)

    repo_id, error = _resolve_repo(write)
    if repo_id is None:
        return error
    bound_repo = f"{repo_id.host}/{repo_id.slug}"
    write.pr_repo = bound_repo  # Receipts and follow-ups bind the resolved repository, never the caller's selector.
    head_owner = owner_part or repo_id.owner
    head_label = f"{head_owner}:{head_branch}"
    if head_owner.casefold() == repo_id.owner.casefold() and head_branch == base_arg:
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: head and base are the same branch.", no_effect=True)
    current, res = _ref_sha(ctx, repo_id, head_owner, head_branch)
    if res is not None:
        return _precheck_failed(write, res, f"head {head_label}", "nothing was created")
    if current != head_sha:
        return _failed(write, f"⚠️ PR_PRECHECK_FAILED: head {head_label} is at {current}, you expected {head_sha}; nothing was "
                            "created. Re-read the head and decide again.", no_effect=True)
    current, res = _ref_sha(ctx, repo_id, repo_id.owner, base_arg)
    if res is not None:
        return _precheck_failed(write, res, f"base {base_arg}", "nothing was created")
    if current != base_sha:
        return _failed(write, f"⚠️ PR_PRECHECK_FAILED: base {base_arg} is at {current}, you expected {base_sha}; nothing was "
                            "created. Re-read the base and decide again.", no_effect=True)
    rows, res = _open_prs(ctx, bound_repo, head_branch, base_arg)
    if res is not None:
        return _precheck_failed(write, res, f"the open pull requests from {head_branch} into {base_arg}", "nothing was created")
    existing = _matching(rows, head_owner=head_owner, head_branch=head_branch, base=base_arg, head_sha=None)
    if existing:
        return _failed(write, f"⚠️ PR_EXISTS: {_pr_line(existing[0])} is already open for {head_label} into {base_arg} in "
                            f"{_where(repo)}; nothing was created. Change it with github_write(method='update', kind='pr', target="
                            f"{existing[0].get('number')}{_repo_hint(bound_repo)}).", no_effect=True)

    # Both refs already exist on GitHub. Submit them directly, without gh's checkout/push discovery.
    sent = _gh_run(["api", f"repos/{repo_id.owner}/{repo_id.name}/pulls", "--method", "POST",
                    "--input", "-", "--hostname", repo_id.host], ctx, timeout=_WRITE_TIMEOUT,
                   input_data=json.dumps({"title": title, "body": body, "head": head_arg, "base": base_arg, "draft": draft}),
                   repo=bound_repo)
    if sent.not_submitted:
        return write.unsent(sent)
    write.pr_send = sent
    expected = dict(title=title, body=body, base=base_arg, draft=draft, head_owner=head_owner, head_branch=head_branch,
                    head_sha=head_sha)
    if sent.ok:
        from ouroboros.merge_receipts import _PR_URL_RE

        answer = _gh_json(sent, dict) or {}
        created_url = str(answer.get("html_url") or "")
        where = _PR_URL_RE.fullmatch(created_url)
        number = _whole(answer.get("number")) or (int(where[4]) if where else 0)
        node_id = answer.get("node_id")
        # The create response's own identity, as its provider states it: an explicit number, a non-empty node id
        # and a canonical URL for that same pull request at the resolved repository. A display URL alone is no
        # identity, and no other answer speaks for this response: only a validated response may carry the
        # affirmative "created" claim, with or without a read-back.
        response_identity = bool(
            where and _whole(answer.get("number")) == number and int(where[4]) == number
            and where[1].casefold() == repo_id.host.casefold()
            and f"{where[2]}/{where[3]}".casefold() == repo_id.slug.casefold()
            and isinstance(node_id, str) and bool(node_id.strip()))
        if response_identity:
            write.pr_created = answer
        found = []
        if not number:
            rows, res = _open_prs(ctx, bound_repo, head_branch, base_arg)
            found = _matching(rows or [], head_owner=head_owner, head_branch=head_branch, base=base_arg, head_sha=head_sha)
            if res is not None or len(found) != 1:
                return _failed(write, "⚠️ PR_CREATE_UNKNOWN: gh reported success but named no pull request, "
                                    "and the read-back " + (f"failed ({res.text})" if res is not None else
                                                            f"found {len(found)} open pull requests for {head_label} into {base_arg}")
                               + f". The tool made one create attempt. Inspect {_where(repo)} (github_read(method='list', kind='pr'{_repo_hint(bound_repo)})) "
                                 "before creating again.")
            number = _whole(found[0].get("number")) or 0
        if number <= 0:
            return _failed(write, "⚠️ PR_CREATE_UNKNOWN: gh reported success but read-back named no valid PR number. "
                           "The tool made one create attempt. Inspect the repository before creating again.")
        pr, res = _view(ctx, number, bound_repo, require_id=True)
        write.pr_observed = pr or (found[0] if found else None)
        if pr is None:
            if not response_identity:
                return _failed(write, "⚠️ PR_CREATE_UNKNOWN: gh reported success but its response carries no validated "
                                      "pull request identity (a number, a node id and the canonical "
                                      f"{repo_id.host}/{repo_id.slug} address) and the read-back failed ({res.text}): "
                                      "the create response and GitHub's records do not establish the same PR node. "
                                      "The tool made one create attempt. Inspect "
                                      f"{_where(repo)} (github_read(method='list', kind='pr'{_repo_hint(bound_repo)})) "
                                      "before creating again.")
            return _ok(write, f"✅ PR #{number} created: {created_url}\n⚠️ Read-back failed "
                            f"({res.text}): the pull request's fields are unconfirmed; read it with "
                            f"github_read(method='get', kind='pr', target={number}{_repo_hint(bound_repo)}).", outcome="unknown")
        read_where = _PR_URL_RE.fullmatch(pr["url"])
        # Identity is the node, never the head commit: a push that moved the head after this create is a field
        # observation (_reconcile's "head moved" note), not an identity mismatch. The read-back, not the response,
        # validates the number on this path.
        same_identity = (
            where and int(where[4]) == number and where[1].casefold() == repo_id.host.casefold()
            and f"{where[2]}/{where[3]}".casefold() == repo_id.slug.casefold()
            and isinstance(node_id, str) and bool(node_id.strip()) and pr["id"] == node_id
            and read_where and int(read_where[4]) == number and read_where[1].casefold() == repo_id.host.casefold()
            and same_pull_request(pull_request_fields_from_cli(pr), head_owner=head_owner, head_branch=head_branch,
                                  base_branch=base_arg, head_sha=None))
        if not same_identity:
            return _failed(write, "⚠️ PR_CREATE_UNKNOWN: the create response and read-back do not establish the same PR node "
                                  "at the resolved repository; do not create again.")
        if not response_identity:
            return _failed(write, "⚠️ PR_CREATE_UNKNOWN: gh reported success and the read-back holds the pull request at "
                                  f"#{number}, but its create response names no validated identity for it (a number, a node "
                                  f"id and the canonical {repo_id.host}/{repo_id.slug} address), so this call cannot claim it "
                                  "created that pull request. The tool made one create attempt. Inspect "
                                  f"{_where(repo)} (github_read(method='get', kind='pr', target={number}{_repo_hint(bound_repo)})) "
                                  "before creating again.")
        return _settled(write, f"✅ PR #{number} created: {pr['url']}", pr, **expected)

    if _http_refused(sent):
        return _failed(write, f"⚠️ PR_CREATE_NOT_APPLIED: GitHub refused the create request (HTTP {sent.http_status}).",
                       no_effect=True, res=sent, readback={"verified": False, "via": "not_read"})
    # A lost answer or CLI failure without a definite server refusal needs one read-back, never a resend.
    prefix = (f"⚠️ PR_CREATE_UNKNOWN: the create request for {head_label} into {base_arg} in {_where(repo)} "
              f"{_send_facts(sent)}. The tool made one attempt. Its effect is unknown until GitHub shows it.")
    rows, res = _open_prs(ctx, bound_repo, head_branch, base_arg)
    if res is not None:
        return _failed(write, prefix + f" Read-back failed too ({res.text}); list the open pull requests "
                                     f"(github_read(method='list', kind='pr'{_repo_hint(bound_repo)})) before creating again.")
    found = _matching(rows, head_owner=head_owner, head_branch=head_branch, base=base_arg, head_sha=None)
    if not found:
        return _failed(write, prefix + f" Read-back: GitHub listed no open pull request from {head_label} into {base_arg} at that "
                                     "moment; a request it is still processing would appear later, so read again "
                                     f"(github_read(method='list', kind='pr'{_repo_hint(bound_repo)})) before creating.")
    write.pr_observed = found[0]
    exact = _matching(found, head_owner=head_owner, head_branch=head_branch, base=base_arg, head_sha=head_sha)
    return _failed(write, prefix + f" Read-back: {_pr_line((exact or found)[0])} is open for this head and base"
                   + (" at the expected commit" if exact else " at another commit")
                   + " — this call's effect or an earlier one; do not create again. Compare its fields and use "
                     f"github_write(method='update', kind='pr', target={(exact or found)[0].get('number')}{_repo_hint(bound_repo)}) if they differ.")


def update(write) -> str:
    """Change the title, body and/or base of one open pull request in one edit, after it reads as expected."""
    ctx, repo, args = write.ctx, write.repo, write.args
    if any(args.get(k) is not None for k in ("state", "state_reason", "labels")):
        return write.refuse("⚠️ TOOL_ARG_ERROR: PR title/body/base or draft changes are one operation; change state/labels in a separate update.")
    if "draft" in args:
        if any(args.get(k) is not None for k in ("title", "body", "base")):
            return write.refuse("⚠️ TOOL_ARG_ERROR: change draft separately from title/body/base to keep one mutation per receipt.")
        return _draft(write)
    number, title, body, base = (args.get(k) for k in ("target", "title", "body", "base"))
    expected_head_sha, expected_base_sha = ((args.get("expect") or {}).get(k, "") for k in ("head_sha", "base_sha"))
    number = _whole(number)
    if number is None or number <= 0:
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: target is the positive pull request number.", no_effect=True)
    head_sha = _sha(expected_head_sha)
    if not head_sha:
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: expect.head_sha is the full 40-hex head commit you read for this pull "
                            "request; the call refuses before any effect when the head moved.", no_effect=True)
    if title is None and body is None and base is None:
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: nothing to change: pass title, body (\"\" clears it; omitted leaves it) "
                            "and/or base.", no_effect=True)
    if title is not None and not _one_line_text(title):
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: title is one non-empty line.", no_effect=True)
    if body is not None and not isinstance(body, str):
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: body is a string: \"\" clears it, omit it to leave it unchanged.", no_effect=True)
    base_arg, base_sha = None, ""
    if base is not None:
        base_arg, base_sha = _branch(base), _sha(expected_base_sha)
        if not base_arg:
            return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: base is a branch name.", no_effect=True)
        if not base_sha:
            return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: expect.base_sha is required with base: the full 40-hex commit you read "
                                "for the new base branch.", no_effect=True)

    pr, repo_id, res = _edit_view(write, number)
    if pr is None:
        return _precheck_failed(write, res, f"pull request #{number}", "nothing was changed")
    state = str(pr.get("state") or "")
    if state != "OPEN":
        return _failed(write, f"⚠️ PR_PRECHECK_FAILED: {_pr_line(pr)} is {state.lower() or 'in an unknown state'}, not open; "
                            "nothing was changed.", no_effect=True)
    current = _sha(pr["headRefOid"])
    if current != head_sha:
        return _failed(write, f"⚠️ PR_PRECHECK_FAILED: the head of PR #{number} is at {current}, you expected {head_sha}; nothing "
                            "was changed. Re-read the pull request and decide again.", no_effect=True)
    requested_base = base_arg
    if base_arg is not None and pr.get("baseRefName") == base_arg:
        # Supplying the current base does not cancel requested title/body edits.
        base_arg = None
    if (base_arg is None and (title is None or title == pr["title"])
            and (body is None or body == pr["body"])):
        return _ok(write, f"PR #{number} already holds the requested values; nothing was changed. {pr['url']}",
                   outcome="completed_no_effect", pr=pr)
    # The edit and any new base ref use the canonical address of the first read, not the original selector.
    if base_arg is not None:
        current, res = _ref_sha(ctx, repo_id, repo_id.owner, base_arg)
        if res is not None:
            return _precheck_failed(write, res, f"base {base_arg}", "nothing was changed")
        if current != base_sha:
            return _failed(write, f"⚠️ PR_PRECHECK_FAILED: base {base_arg} is at {current}, you expected {base_sha}; nothing was "
                                "changed.", no_effect=True)

    # The one mutation: one REST PATCH of exactly the fields to change (the held base is not resent).
    fields = {key: value for key, value in (("title", title), ("body", body), ("base", base_arg)) if value is not None}
    sent = _gh_run(["api", f"repos/{repo_id.owner}/{repo_id.name}/pulls/{number}", "--method", "PATCH", "--input", "-",
                    "--hostname", repo_id.host], ctx, timeout=_WRITE_TIMEOUT, input_data=json.dumps(fields))
    if sent.not_submitted:
        return write.unsent(sent)
    requested = dict(title=title, body=body, base=requested_base)
    after, _, res = _edit_view(write, number, same_id=pr["id"])
    write.pr_observed = after
    hint = _repo_hint(write.pr_repo)
    if sent.ok:
        if after is None:
            return _ok(write, f"✅ gh accepted the edit of PR #{number}.\n⚠️ Read-back failed ({res.text}): the fields are "
                            f"unconfirmed; read them with {_reread(number, hint, pr)}.", outcome="unknown",
                       pr=pr, readback=_lost_readback(write, pr, number, res))
        confirmed, differs, notes = _reconcile(after, **requested, head_sha=head_sha)
        headline = (f"✅ PR #{number} updated: {after['url']}" if not differs else
                    f"⚠️ gh accepted the edit of PR #{number} ({after['url']}), but the read-back does not hold every requested value.")
        return _ok(write, "\n".join([headline] + _readback_lines(confirmed, differs, notes)),
                   outcome="unknown" if differs else "completed", pr=after,
                   readback={"verified": not differs, "mismatch": differs, "confirmed": confirmed,
                             "notes": notes, "via": "lookup", "item": after})

    write.pr_send = sent
    prefix = (f"⚠️ PR_UPDATE_UNKNOWN: the edit request for PR #{number} in {_where(repo)} {_send_facts(sent)}. "
              "The tool made one attempt. Its effect is unknown until GitHub shows it.")
    if after is None:
        return _failed(write, prefix + f" Read-back failed too ({res.text}); read the pull request "
                                     f"with {_reread(number, hint, pr)}.",
                       pr=pr, readback=_lost_readback(write, pr, number, res))
    if _http_refused(sent):  # a refusal answer; the unchanged read-back, not the status, proves no field changed
        changed = _changed_since(pr, after, fields)
        if not changed:
            notes = "".join(f" Note: {note}." for note in _reconcile(after, head_sha=head_sha)[2])
            return _failed(write, f"⚠️ GH_WRITE_NOT_APPLIED: GitHub refused the edit of PR #{number} in {_where(repo)}: the "
                                f"request {_send_facts(sent)}. One request was sent and not retried; the read-back holds every "
                                f"sent field as the precheck read it, so nothing was changed.{notes}", no_effect=True)
        return _failed(write, prefix + " GitHub answered with a refusal, yet the read-back differs from the precheck in a sent "
                       "field: " + "; ".join(changed) + ". GitHub documents no all-or-nothing edit, so the read-back cannot "
                       "tell which request made that change; read the pull request again before any further edit.")
    confirmed, differs, _notes = _reconcile(after, **requested)
    if not confirmed:
        return _failed(write, prefix + " Read-back: none of the requested values was held at that moment (" + "; ".join(differs)
                       + "); a request GitHub is still processing could still land, so read again before resending.")
    if not differs:
        return _failed(write, prefix + " Read-back: GitHub holds every requested value (" + "; ".join(confirmed) + "); do not resend.")
    return _failed(write, prefix + " Read-back at that moment: held: " + "; ".join(confirmed) + "; not held: " + "; ".join(differs)
                   + ". Read again before resending only the fields not held.")


def _draft(write) -> str:
    """Mark one open pull request ready for review (or back to draft), after it reads as expected."""
    ctx, repo = write.ctx, write.repo
    number, draft = write.args.get("target"), write.args.get("draft")
    ready = not draft if isinstance(draft, bool) else draft
    expected_head_sha = (write.args.get("expect") or {}).get("head_sha", "")
    number = _whole(number)
    if number is None or number <= 0:
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: target is the positive pull request number.", no_effect=True)
    if not isinstance(ready, bool):
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: draft is a boolean: false marks the draft ready for review, true converts "
                            "the pull request back to a draft.", no_effect=True)
    head_sha = _sha(expected_head_sha)
    if not head_sha:
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: expect.head_sha is the full 40-hex head commit you read for this pull "
                            "request; the call refuses before any effect when the head moved.", no_effect=True)
    want = "ready for review" if ready else "a draft"

    pr, _, res = _edit_view(write, number)
    if pr is None:
        return _precheck_failed(write, res, f"pull request #{number}", "nothing was changed")
    state = str(pr.get("state") or "")
    if state != "OPEN":
        return _failed(write, f"⚠️ PR_PRECHECK_FAILED: {_pr_line(pr)} is {state.lower() or 'in an unknown state'}, not open; "
                            "nothing was changed.", no_effect=True)
    current = _sha(pr["headRefOid"])
    if current != head_sha:
        return _failed(write, f"⚠️ PR_PRECHECK_FAILED: the head of PR #{number} is at {current}, you expected {head_sha}; nothing "
                            "was changed. Re-read the pull request and decide again.", no_effect=True)
    if pr["isDraft"] == (not ready):
        return _ok(write, f"PR #{number} is already {want}; nothing was changed. {pr['url']}", outcome="completed_no_effect", pr=pr)

    sent = _gh_run(["pr", "ready", str(number)] + ([] if ready else ["--undo"]), ctx,
                   timeout=_WRITE_TIMEOUT, repo=write.pr_repo)
    if sent.not_submitted:
        return write.unsent(sent)
    after, _, res = _edit_view(write, number, same_id=pr["id"])
    write.pr_observed = after
    hint = _repo_hint(write.pr_repo)
    if sent.ok:
        if after is None:
            return _ok(write, f"✅ gh accepted marking PR #{number} {want}.\n⚠️ Read-back failed ({res.text}): the state is "
                            f"unconfirmed; read it with {_reread(number, hint, pr)}.", outcome="unknown",
                       pr=pr, readback=_lost_readback(write, pr, number, res))
        confirmed, differs, notes = _reconcile(after, draft=not ready, head_sha=head_sha)
        headline = (f"✅ PR #{number} is now {want}: {after['url']}" if not differs else
                    f"⚠️ gh accepted marking PR #{number} {want} ({after['url']}), but the read-back does not show that state.")
        return _ok(write, "\n".join([headline] + _readback_lines(confirmed, differs, notes)),
                   outcome="unknown" if differs else "completed", pr=after,
                   readback={"verified": not differs, "mismatch": differs, "confirmed": confirmed,
                             "notes": notes, "via": "lookup", "item": after})

    write.pr_send = sent
    prefix = (f"⚠️ PR_READY_UNKNOWN: the request to mark PR #{number} in {_where(repo)} {want} {_send_facts(sent)}. "
              "The tool made one attempt. Its effect is unknown until GitHub shows it.")
    if after is None:
        return _failed(write, prefix + f" Read-back failed too ({res.text}); read the pull request "
                                     f"with {_reread(number, hint, pr)}.",
                       pr=pr, readback=_lost_readback(write, pr, number, res))
    held = after["isDraft"]
    if held == (not ready):
        return _failed(write, prefix + f" Read-back: GitHub reports draft={held}, the requested state; do not resend.")
    return _failed(write, prefix + f" Read-back: GitHub still reported draft={held} at that moment; a request it is still "
                                 "processing could still land, so read again before resending.")
