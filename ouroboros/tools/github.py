"""GitHub tools: issues, pull requests, comments, checks."""

from __future__ import annotations

import json
import logging
import os
import pathlib
import re
import subprocess
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import wraps
from typing import List, Optional

from ouroboros.secret_masking import redact_known_values
from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.tools.tool_result import LegacyTextResultAdapter, ToolResult, _publish_tool_result, _replace_tool_result
from ouroboros.utils import truncate_within_limit
from ouroboros.utils import truncate_review_artifact as _truncate_with_notice

log = logging.getLogger(__name__)
_GENERIC_TRANSPORT = object()
# One public tool invocation's "a gh subprocess was already launched" fact.
_SUBMITTED: ContextVar[Optional[List[bool]]] = ContextVar("github_invocation_submitted", default=None)


def _one_invocation(handler):
    """Scope ``_gh_run``'s first-submission fact to one public tool invocation."""
    @wraps(handler)
    def invoke(ctx, *args, **kwargs):
        token = _SUBMITTED.set([False])
        try:
            return handler(ctx, *args, **kwargs)
        finally:
            _SUBMITTED.reset(token)
    return invoke


# gh's own HTTP status shapes (see ``_gh_run``); the first match in stderr order wins.
_GH_STATUS_RE = re.compile(
    r"\(HTTP (\d{3})\)[ \t\r]*$"
    r"|^(?:[a-z][a-z ]*: )*HTTP (\d{3})(?::| \(|[ \t\r]*$)",
    re.MULTILINE,
)


@dataclass(frozen=True)
class GhResult:
    ok: bool
    text: str
    exit_code: int | None
    http_status: int | None
    # "target" is a local refusal and "deadline" a request the checks reader never sent;
    # neither is a subprocess exit or exception.
    failure: str


def _refuse(ctx: ToolContext, text: str, code: str = "TOOL_ARG_ERROR", *, no_effect: bool = False) -> str:
    """Publish a refusal this module AUTHORS as a typed result; text unchanged.

    The registry types a string result by its first-line typed marker (the
    warning sign plus an UPPER_SNAKE code), so prose such as ``⚠️ issue number must be positive`` was recorded ``status=ok``
    although the producer already knew it had refused. Both codes carry
    ``status="error"``."""
    return _publish_tool_result(ctx, ToolResult(status="error", code=code, text=text,
        meta={"operation_outcome": "completed_no_effect"} if no_effect else {}))


def github_token_from_env_or_settings() -> str:
    from ouroboros.config import load_settings
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
    if not token:
        try:
            token = load_settings().get("GITHUB_TOKEN", "")
        except Exception:
            token = ""
    return str(token or "").strip()


def _gh_env(ctx: ToolContext) -> dict:
    env = os.environ.copy()
    token = github_token_from_env_or_settings()
    if token:
        env["GH_TOKEN"] = token
        env["GITHUB_TOKEN"] = token
    return env


def github_cli_configured() -> bool:
    """Local credential configuration, not a live authentication assertion."""
    if github_token_from_env_or_settings():
        return True
    config_dir = os.environ.get("GH_CONFIG_DIR", "")
    if not config_dir:
        base = os.environ.get("XDG_CONFIG_HOME", "")
        config_dir = str(pathlib.Path(base) / "gh") if base else ""
    if not config_dir:
        from ouroboros.platform_layer import IS_WINDOWS

        app_data = os.environ.get("APPDATA", "") if IS_WINDOWS else ""
        config_dir = str(pathlib.Path(app_data) / "GitHub CLI") if app_data else str(pathlib.Path.home() / ".config" / "gh")
    try:
        import yaml

        hosts = yaml.safe_load((pathlib.Path(config_dir) / "hosts.yml").read_text(encoding="utf-8"))
        return isinstance(hosts, dict) and any(
            isinstance(host, dict) and bool(host.get("user") or host.get("users") or host.get("oauth_token"))
            for host in hosts.values()
        )
    except (OSError, ValueError, yaml.YAMLError):
        return False


def _gh_run(args: List[str], ctx: ToolContext, timeout: int = 30, input_data: Optional[str] = None,
            *, repo: object = _GENERIC_TRANSPORT) -> GhResult:
    # Only omitted internal API/Hub calls keep the generic transport contract.
    # Public repository tools always pass repo, including '' for Project focus.
    # The target refusals below publish a typed argument error into the calling
    # tool's sidecar; the publication transport omits `repo`, so it can never
    # reach them and its own final result is never shadowed from here.
    # A refusal before this invocation's FIRST gh launch also attests that the
    # invocation had no effect; after any launch it attests nothing.
    submitted = _SUBMITTED.get()
    first = submitted is not None and not submitted[0]

    def refused(text: str, failure: str, typed: Optional[ToolResult] = None) -> GhResult:
        if first:
            typed = _replace_tool_result(typed or LegacyTextResultAdapter.from_text("", text),
                                         meta_updates={"operation_outcome": "completed_no_effect"})
        return GhResult(False, _publish_tool_result(ctx, typed) if typed else text, None, None, failure)

    if repo is not _GENERIC_TRANSPORT and not isinstance(repo, str):
        return refused("", "target", ToolResult(
            status="error", code="TOOL_ARG_ERROR",
            text="⚠️ GH_TARGET_INVALID: repo must be a string; omit it to use the selected Project.",
        ))
    try:
        cwd, env = pathlib.Path(ctx.repo_dir), _gh_env(ctx)
        cmd = ["gh", *args]
        if repo is not _GENERIC_TRANSPORT:
            from ouroboros.tool_access import build_resolved_resource_binding

            metadata = getattr(ctx, "task_metadata", {})
            metadata = metadata if isinstance(metadata, dict) else {}
            workspace = getattr(ctx, "workspace_root", None)
            room_dir = str(metadata.get("_project_room_dir") or "")
            project = str(getattr(ctx, "project_id", "") or "")
            if not repo:
                note = str(metadata.get("_project_room_note") or "")
                selected = workspace or room_dir
                if note or (selected and not pathlib.Path(selected).is_dir()):
                    return refused(
                        f"⚠️ GH_TARGET_UNAVAILABLE: {note or 'The selected Project directory is unavailable.'}",
                        "target")
                if project and not selected:
                    return refused("", "target", ToolResult(
                        status="error", code="TOOL_ARG_ERROR",
                        text="⚠️ GH_TARGET_REQUIRED: this Project has no repository directory; pass repo='[HOST/]OWNER/REPO'.",
                    ))
            binding = build_resolved_resource_binding(ctx, operation="shell", process_cwd="")
            cwd = binding.target_path
            if workspace and cwd != pathlib.Path(workspace).resolve(strict=False):
                return refused("⚠️ GH_TARGET_UNAVAILABLE: the task's Project binding could not be resolved.",
                               "target")
            if workspace or room_dir or project:
                env.pop("GH_REPO", None)  # Ambient defaults cannot replace the selected Project.
            if repo:
                cmd.extend(["--repo", repo])
        if submitted is not None:
            submitted[0] = True
        res = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            input=input_data,
            env=env,
        )
        if res.returncode != 0:
            # Redact the WHOLE stderr first (a cut could split a token), read gh's own
            # status marker before any bounding, then keep a bounded head. gh writes the
            # status in three deterministic shapes and nowhere else: ``gh: <msg> (HTTP NNN)``
            # at the end of a line (``gh api`` with a message), ``gh: HTTP NNN`` (``gh api``
            # without one) and ``HTTP NNN: <msg> (<url>)`` — optionally wrapped as
            # ``failed to fork: HTTP NNN: …`` — from every other command. A marker quoted
            # mid-sentence is prose, not a status.
            err = redact_known_values(res.stderr or "", [github_token_from_env_or_settings()])
            status = _GH_STATUS_RE.search(err)
            head = " | ".join([line.strip() for line in err.splitlines() if line.strip()][:3])
            head = truncate_within_limit(head, 600)
            # gh's canMerge refusal precedes its mutation (pkg/cmd/pr/merge).
            # HTTP status alone proves nothing about which CLI step failed.
            pre_effect = (args[:2] == ["pr", "merge"] and not res.stdout.strip() and re.fullmatch(
                r"X Pull request [\w.-]+/[\w.-]+#\d+ is not mergeable: "
                r"(?:the base branch policy prohibits the merge|the head branch is not up to date with the base branch)\.\n"
                r"To have the pull request merged after all the requirements have been met, add the `--auto` flag\.\n"
                r"To use administrator privileges to immediately merge the pull request, add the `--admin` flag\.",
                err.strip()) is not None)
            return GhResult(False, "⚠️ GH_ERROR: " + head, res.returncode,
                            int(status.group(1) or status.group(2)) if status else None,
                            "pre_effect" if pre_effect else "exit")
        return GhResult(True, res.stdout.strip(), res.returncode, None, "")
    except FileNotFoundError as e:
        missing = str(getattr(e, "filename", "") or "")
        if not missing or pathlib.Path(missing).name == "gh":
            return refused(
                "⚠️ GH_ERROR: `gh` CLI not found. Install GitHub CLI and ensure it is on PATH (https://cli.github.com/)",
                "cli_missing")
        detail = truncate_within_limit(redact_known_values(str(e), [github_token_from_env_or_settings()]), 600)
        return GhResult(False, f"⚠️ GH_ERROR: {detail}", None, None, "exception")
    except subprocess.TimeoutExpired:
        return GhResult(False, f"⚠️ GH_TIMEOUT: exceeded {timeout}s.", None, None, "timeout")
    except Exception as e:
        detail = truncate_within_limit(redact_known_values(str(e), [github_token_from_env_or_settings()]), 600)
        return GhResult(False, f"⚠️ GH_ERROR: {detail}", None, None, "exception")


def _gh_cmd(args: List[str], ctx: ToolContext, timeout: int = 30, input_data: Optional[str] = None,
            *, repo: object = _GENERIC_TRANSPORT) -> str:
    return _gh_run(args, ctx, timeout=timeout, input_data=input_data, repo=repo).text


def _list_issues(ctx: ToolContext, state: str = "open", labels: str = "", limit: int = 20, repo: str = "") -> str:
    args = [
        "issue", "list",
        "--state", state,
        "--limit", str(min(limit, 50)),
        "--json", "number,title,body,labels,createdAt,author,assignees,state",
    ]
    if labels:
        args.extend(["--label", labels])

    raw = _gh_cmd(args, ctx, repo=repo)
    if raw.startswith("⚠️"):
        return raw

    try:
        issues = json.loads(raw)
    except json.JSONDecodeError:
        return _refuse(ctx, f"⚠️ TOOL_ERROR: failed to parse issues JSON: {raw[:500]}", "TOOL_ERROR")

    if not issues:
        return f"No {state} issues found."

    lines = [f"**{len(issues)} {state} issue(s):**\n"]
    for issue in issues:
        labels_str = ", ".join(l.get("name", "") for l in issue.get("labels", []))
        author = issue.get("author", {}).get("login", "unknown")
        lines.append(
            f"- **#{issue['number']}** {issue['title']}"
            f" (by @{author}{', labels: ' + labels_str if labels_str else ''})"
        )
        body = (issue.get("body") or "").strip()
        if body:
            preview = body[:200] + ("..." if len(body) > 200 else "")
            lines.append(f"  > {preview}")

    return "\n".join(lines)


def _get_issue(ctx: ToolContext, number: int, repo: str = "") -> str:
    if number <= 0:
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: issue number must be positive", no_effect=True)

    args = [
        "issue", "view", str(number),
        "--json", "number,title,body,labels,createdAt,author,assignees,state,comments",
    ]

    raw = _gh_cmd(args, ctx, repo=repo)
    if raw.startswith("⚠️"):
        return raw

    try:
        issue = json.loads(raw)
    except json.JSONDecodeError:
        return _refuse(ctx, f"⚠️ TOOL_ERROR: failed to parse issue JSON: {raw[:500]}", "TOOL_ERROR")

    labels_str = ", ".join(l.get("name", "") for l in issue.get("labels", []))
    author = issue.get("author", {}).get("login", "unknown")

    lines = [
        f"## Issue #{issue['number']}: {issue['title']}",
        f"**State:** {issue['state']}  |  **Author:** @{author}",
    ]
    if labels_str:
        lines.append(f"**Labels:** {labels_str}")

    body = (issue.get("body") or "").strip()
    if body:
        lines.append(f"\n**Body:**\n{_truncate_with_notice(body, 3000)}")

    comments = issue.get("comments", [])
    if comments:
        shown_comments = comments[:10]
        lines.append(f"\n**Comments (showing {len(shown_comments)} of {len(comments)}):**")
        for c in shown_comments:
            c_author = c.get("author", {}).get("login", "unknown")
            c_body = _truncate_with_notice((c.get("body") or "").strip(), 500)
            lines.append(f"\n@{c_author}:\n{c_body}")

    return "\n".join(lines)


def _comment_on_issue(ctx: ToolContext, number: int, body: str, repo: str = "") -> str:
    if number <= 0:
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: issue number must be positive", no_effect=True)

    if not body or not body.strip():
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: comment body cannot be empty.", no_effect=True)

    args = ["issue", "comment", str(number), "--body-file", "-"]
    raw = _gh_cmd(args, ctx, input_data=body, repo=repo)
    if raw.startswith("⚠️"):
        return raw
    return f"✅ Comment added to issue #{number}."


def _close_issue(ctx: ToolContext, number: int, comment: str = "", repo: str = "") -> str:
    if number <= 0:
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: issue number must be positive", no_effect=True)

    if comment and comment.strip():
        result = _comment_on_issue(ctx, number, comment, repo=repo)
        if result.startswith("⚠️"):
            return result

    args = ["issue", "close", str(number)]
    raw = _gh_cmd(args, ctx, repo=repo)
    if raw.startswith("⚠️"):
        return raw
    return f"✅ Issue #{number} closed."

def _list_prs(ctx: ToolContext, state: str = "open", limit: int = 20, repo: str = "") -> str:
    args = [
        "pr", "list",
        "--state", state,
        "--limit", str(min(limit, 50)),
        "--json", "number,title,author,headRefName,baseRefName,createdAt,isDraft,reviewDecision,commits",
    ]
    raw = _gh_cmd(args, ctx, repo=repo)
    if raw.startswith("⚠️"):
        return raw

    try:
        prs = json.loads(raw)
    except json.JSONDecodeError:
        return _refuse(ctx, f"⚠️ TOOL_ERROR: failed to parse PRs JSON: {raw[:500]}", "TOOL_ERROR")

    if not prs:
        return f"No {state} pull requests found."

    lines = [f"**{len(prs)} {state} PR(s):**\n"]
    for pr in prs:
        author = pr.get("author", {}).get("login", "unknown")
        head = pr.get("headRefName", "?")
        base = pr.get("baseRefName", "?")
        draft = " [DRAFT]" if pr.get("isDraft") else ""
        review = pr.get("reviewDecision") or ""
        review_str = f" [{review}]" if review else ""
        n_commits = len(pr.get("commits", []))
        lines.append(
            f"- **PR #{pr['number']}**{draft}{review_str} {pr['title']}"
            f" (by @{author}, {head}→{base}, {n_commits} commits, created {pr['createdAt'][:10]})"
        )

    return "\n".join(lines)


def _get_pr(ctx: ToolContext, number: int, repo: str = "") -> str:
    if number <= 0:
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: PR number must be positive.", no_effect=True)

    meta_args = [
        "pr", "view", str(number),
        "--json", "number,title,body,author,headRefName,baseRefName,headRepository,"
                  "createdAt,updatedAt,state,isDraft,reviewDecision,mergeable,"
                  "additions,deletions,changedFiles,commits,reviews,comments",
    ]
    raw = _gh_cmd(meta_args, ctx, timeout=30, repo=repo)
    if raw.startswith("⚠️"):
        return raw

    try:
        pr = json.loads(raw)
    except json.JSONDecodeError:
        return _refuse(ctx, f"⚠️ TOOL_ERROR: failed to parse PR JSON: {raw[:500]}", "TOOL_ERROR")

    author = pr.get("author", {}).get("login", "unknown")
    head_repo = (pr.get("headRepository") or {}).get("nameWithOwner", "?")

    lines = [
        f"## PR #{pr['number']}: {pr['title']}",
        f"**State:** {pr['state']}  |  **Author:** @{author}",
        f"**Branch:** {head_repo}@{pr.get('headRefName','?')} → {pr.get('baseRefName','?')}",
        f"**Changes:** +{pr.get('additions',0)} / -{pr.get('deletions',0)}"
        f" across {pr.get('changedFiles',0)} file(s)",
        f"**Mergeable:** {pr.get('mergeable', 'unknown')}",
    ]
    if pr.get("isDraft"):
        lines.append("**⚠️ Draft PR**")
    if pr.get("reviewDecision"):
        lines.append(f"**Review decision:** {pr['reviewDecision']}")

    body = (pr.get("body") or "").strip()
    if body:
        lines.append(f"\n**Description:**\n{_truncate_with_notice(body, 2000)}")

    commits = pr.get("commits", [])
    if commits:
        lines.append(
            f"\n**Commits ({len(commits)}) — original author preserved on cherry-pick:**"
        )
        shas_for_pick = []
        for c in commits:
            node = c.get("commit", c)
            sha = c.get("oid", "?")[:12]
            full_sha = c.get("oid", "?")
            msg = (node.get("messageHeadline") or node.get("message") or "?")[:70]
            authored_by = node.get("authors", {})
            if isinstance(authored_by, dict):
                authored_by = authored_by.get("nodes", [])
            if authored_by:
                a = authored_by[0]
                author_str = f"{a.get('name','?')} <{a.get('email','?')}>"
            else:
                author_str = "unknown"
            lines.append(f"  {sha} | {author_str} | {msg}")
            shas_for_pick.append(full_sha)
        lines.append(f"\nCommit SHAs:\n  {shas_for_pick}")

    diff_names_raw = _gh_cmd(["pr", "diff", str(number), "--name-only"], ctx, timeout=30, repo=repo)
    if not diff_names_raw.startswith("⚠️") and diff_names_raw.strip():
        file_list = diff_names_raw.strip().splitlines()
        lines.append(f"\n**Changed files ({len(file_list)}):**")
        for f in file_list[:50]:
            lines.append(f"  {f}")
        if len(file_list) > 50:
            lines.append(f"  ... and {len(file_list) - 50} more")

    diff_raw = _gh_cmd(["pr", "diff", str(number)], ctx, timeout=60, repo=repo)
    if not diff_raw.startswith("⚠️") and diff_raw.strip():
        lines.append("\n**Diff (truncated to 8000 chars):**\n```diff")
        lines.append(_truncate_with_notice(diff_raw, 8000))
        lines.append("```")

    reviews = pr.get("reviews", [])
    comments = pr.get("comments", [])
    if reviews or comments:
        lines.append(f"\n**Reviews ({len(reviews)}) + PR comments ({len(comments)}):**")
        for rv in reviews[:5]:
            rv_author = (rv.get("author") or {}).get("login", "?")
            rv_state = rv.get("state", "?")
            rv_body = _truncate_with_notice((rv.get("body") or "").strip(), 300)
            lines.append(f"  [{rv_state}] @{rv_author}: {rv_body}")
        for cm in comments[:5]:
            cm_author = (cm.get("author") or {}).get("login", "?")
            cm_body = _truncate_with_notice((cm.get("body") or "").strip(), 300)
            lines.append(f"  @{cm_author}: {cm_body}")

    if not (repo or getattr(ctx, "workspace_root", None) or getattr(ctx, "project_id", "")):
        lines.append(
            f"\n**Integration steps:**\n"
            f"  1. fetch_pr_ref(pr_number={number})\n"
            f"  2. create_integration_branch(pr_number={number})\n"
            f"  3. cherry_pick_pr_commits(shas=[...])  # SHAs above; use override_author only for placeholder identities\n"
            f"  4. stage_adaptations()                 # optional; do NOT commit_reviewed on the integration branch\n"
            f"  5. stage_pr_merge(branch='integrate/pr-{number}') → commit_reviewed\n"
            f"  6. comment_on_pr(number={number}, body='Integrated as ...')"
        )

    return "\n".join(lines)


def _comment_on_pr(ctx: ToolContext, number: int, body: str, repo: str = "") -> str:
    if number <= 0:
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: PR number must be positive.", no_effect=True)
    if not (body or "").strip():
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: comment body cannot be empty.", no_effect=True)

    args = ["pr", "comment", str(number), "--body-file", "-"]
    raw = _gh_cmd(args, ctx, input_data=body, repo=repo)
    if raw.startswith("⚠️"):
        return raw
    return f"✅ Comment added to PR #{number}."


def _pr_merge(ctx: ToolContext, number: int, expected_head_sha: str, method: str,
              review_task_ids: Optional[List[str]] = None, reviewed_head_sha: str = "",
              reviewed_base_sha: str = "", review_scope: str = "full", review_verdict: str = "",
              review_record_id: str = "", repo: str = "") -> str:
    """Thin transport binding; the receipt contract lives in ``merge_receipts``."""
    from ouroboros.merge_receipts import REVIEW_SCOPES, _VERDICT_RE, _sha, run_pr_merge
    from ouroboros.tool_access import canonical_data_root

    if review_scope not in REVIEW_SCOPES or (review_verdict and not _VERDICT_RE.fullmatch(review_verdict)):
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: review_scope is full|delta; review_verdict is a short word such as PASS.", no_effect=True)
    record_id = str(review_record_id or "").strip()
    declared = ({"reviewed_head_sha": _sha(reviewed_head_sha), "reviewed_base_sha": _sha(reviewed_base_sha),
                 "scope": review_scope, "verdict": review_verdict}
                if (reviewed_head_sha or review_verdict or (review_task_ids and not record_id)) else None)
    receipt = run_pr_merge(
        ctx, lambda args, **kw: _gh_run(args, ctx, repo=repo, **kw), lambda args, **kw: _gh_run(args, ctx, **kw),
        drive_root=canonical_data_root(ctx), task_id=str(ctx.task_id or ""), number=int(number or 0),
        expected_head_sha=expected_head_sha, method=method,
        review={"declared": declared, "task_ids": list(review_task_ids or []), "record_id": record_id})
    if receipt.get("refused"):
        code = "TOOL_ARG_ERROR" if receipt["refused"] == "arguments" else "TOOL_ERROR"
        return _refuse(ctx, f"⚠️ PR_MERGE_REFUSED: {receipt['refused']} — {receipt.get('detail', '')}", code, no_effect=True)
    from ouroboros.merge_receipts import card_row_text

    status = (receipt.get("outcome") or {}).get("status", "unknown")
    lines = [card_row_text(receipt), f"receipt_id={receipt['receipt_id']} (task result: merge_receipts)"]
    if receipt.get("readback_only"):
        lines.append("An earlier request for this PR had no confirmed outcome, so this call only read GitHub back "
                     "and sent no new merge request. Unknown and queued requests remain observation-only.")
    if receipt.get("republished"):
        lines.append("Receipt publication retried; the merge was not repeated.")
    publication = receipt.get("publication") or {}
    if receipt.get("receipt_write_gap"):
        lines.append("⚠️ Merge receipt persistence is unknown after the external effect: "
                     + receipt["receipt_write_gap"])
    if status in ("merged", "queued") and (publication.get("body") or {}).get("status") != "published":
        lines.append("⚠️ The PR-body receipt block was not confirmed; call pr_merge again to retry publication only.")
    card = publication.get("card") or {}
    if status in ("merged", "queued") and card.get("status") not in ("owed", "delivered"):
        lines.append("⚠️ The task-card receipt is not confirmed: " + str(card.get("reason") or "publication unknown")
                     + "; call pr_merge again for observation/publication only.")
    text = "\n".join(lines)
    outcome = "completed" if status == "merged" else "unknown"
    if status == "refused" and (receipt.get("effect") or {}).get("failure") in {"cli_missing", "target", "pre_effect"}:
        outcome = "completed_no_effect"
    return _publish_tool_result(ctx, ToolResult(
        status="ok" if status in ("merged", "queued") else "error",
        code="OK" if status in ("merged", "queued") else "TOOL_ERROR",
        text=text if status in ("merged", "queued") else f"⚠️ PR_MERGE_{status.upper()}: " + text,
        meta={"operation_outcome": outcome}))


def _create_issue(ctx: ToolContext, title: str, body: str = "", labels: str = "", repo: str = "") -> str:
    if not title or not title.strip():
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: issue title cannot be empty.", no_effect=True)

    args = ["issue", "create", f"--title={title}"]
    if body:
        args.append("--body-file=-")
        raw = _gh_cmd(args, ctx, input_data=body, repo=repo)
    else:
        raw = _gh_cmd(args, ctx, repo=repo)

    if labels:
        if not raw.startswith("⚠️"):
            import re
            match = re.search(r'/issues/(\d+)', raw)
            if match:
                issue_num = int(match.group(1))
                label_args = ["issue", "edit", str(issue_num), f"--add-label={labels}"]
                _gh_cmd(label_args, ctx, repo=repo)

    if raw.startswith("⚠️"):
        return raw
    return f"✅ Issue created: {raw}"


def _get_checks(ctx: ToolContext, number: int = 0, sha: str = "", wait_seconds: int = 0, repo: str = "") -> str:
    """``get_github_checks``: the reader lives in ``github_checks`` and calls this module's transport."""
    from ouroboros.tools.github_checks import get_checks

    return get_checks(ctx, number=number, sha=sha, wait_seconds=wait_seconds, repo=repo)


def get_tools() -> List[ToolEntry]:
    tools = [
        ToolEntry("list_github_prs", {
            "name": "list_github_prs",
            "description": (
                "List GitHub pull requests for the current repository. "
                "Shows PR number, title, author, branch, commit count, and state. "
                "Use before get_github_pr to identify which PR to inspect."
            ),
            "parameters": {"type": "object", "properties": {
                "state": {"type": "string", "default": "open",
                          "enum": ["open", "closed", "merged", "all"],
                          "description": "Filter by PR state"},
                "limit": {"type": "integer", "default": 20,
                          "description": "Max PRs to return (max 50)"},
            }, "required": []},
        }, _list_prs),

        ToolEntry("get_github_pr", {
            "name": "get_github_pr",
            "description": (
                "Get full details of a GitHub PR: metadata, description, commit list "
                "with original author names/emails, changed files list, diff/patch "
                "(truncated to 8000 chars), review comments, and mergeable state. "
                "Includes exact commit SHAs for the selected repository."
            ),
            "parameters": {"type": "object", "properties": {
                "number": {"type": "integer", "description": "PR number"},
            }, "required": ["number"]},
        }, _get_pr),

        ToolEntry("get_github_checks", {
            "name": "get_github_checks",
            "description": (
                "Read what GitHub records about the checks of one commit: every workflow run (its latest attempt) with its state, the state "
                "counts of its jobs (for a pull request, of the rollup's jobs), the failed, unfinished and cancelled jobs and their steps, "
                "failure annotations (test names when the workflow publishes them) and, for a pull request, third-party "
                "checks and commit statuses. Read-only: pushes and dispatches nothing. Reports facts and names each source "
                "it could not read; it gives no verdict, and a workflow that did not start has no record to report."
            ),
            "parameters": {"type": "object", "properties": {
                "number": {"type": "integer", "default": 0,
                           "description": "Pull request number; its head commit is read. Pass exactly one of number / sha."},
                "sha": {"type": "string", "default": "",
                        "description": "Full 40-hex commit SHA (resolve a branch or tag with `git rev-parse <ref>`)."},
                "wait_seconds": {"type": "integer", "default": 0,
                                 "description": "Poll until a workflow run is registered and every registered run is completed, "
                                                "or this many seconds pass (max 240); the report states what is unfinished. "
                                                "A run-list read that fails, also during the wait, ends the call with that error."},
            }, "required": []},
        }, _get_checks),

        ToolEntry("comment_on_pr", {
            "name": "comment_on_pr",
            "description": (
                "Add a comment to a GitHub pull request. "
                "Use to acknowledge receipt, report integration status, request changes, "
                "or leave an audit trail after integration."
            ),
            "parameters": {"type": "object", "properties": {
                "number": {"type": "integer", "description": "PR number"},
                "body": {"type": "string", "description": "Comment text (markdown)"},
            }, "required": ["number", "body"]},
        }, _comment_on_pr),

        ToolEntry("pr_merge", {
            "name": "pr_merge",
            "description": (
                "Merge a GitHub pull request so a receipt exists: states the exact head you expect "
                "and the method (never auto-merge or admin), records the host review record you name or "
                "the review you declare beside what the host observes, reads GitHub back, and writes the "
                "receipt to this task's record, its card and the PR body. A missing review is recorded loudly, never a lock. "
                "An unknown or queued merge stays observation/publication-only on repeat calls; no resend. "
                "Distinct from stage_pr_merge, which stages a local merge for a reviewed commit."
            ),
            "parameters": {"type": "object", "properties": {
                "number": {"type": "integer", "description": "PR number"},
                "expected_head_sha": {"type": "string", "description": "The PR head you intend to merge; GitHub refuses if it moved"},
                "method": {"type": "string", "enum": ["merge", "squash", "rebase"]},
                "review_task_ids": {"type": "array", "items": {"type": "string"}, "default": [],
                                    "description": "Task ids of the reviews you rely on; the host records what it can observe of each"},
                "reviewed_head_sha": {"type": "string", "default": "", "description": "The head those reviews covered (declared)"},
                "reviewed_base_sha": {"type": "string", "default": "", "description": "The base those reviews covered (declared)"},
                "review_scope": {"type": "string", "enum": ["full", "delta"], "default": "full",
                                 "description": "delta = only the change since an earlier review; never counted as whole-PR coverage"},
                "review_verdict": {"type": "string", "default": "", "description": "The declared verdict word, e.g. PASS"},
                "review_record_id": {"type": "string", "default": "",
                                     "description": "Id of the host review record to bind (e.g. the review_record_id commit_reviewed "
                                                    "returns); the host then reads the reviewed subject and verdict from it instead "
                                                    "of your declaration. An id with no record is refused before any merge"},
            }, "required": ["number", "expected_head_sha", "method"]},
        }, _pr_merge),

        ToolEntry("list_github_issues", {
            "name": "list_github_issues",
            "description": "List GitHub issues. Use to check for new tasks, bug reports, or feature requests from the user or contributors.",
            "parameters": {"type": "object", "properties": {
                "state": {"type": "string", "default": "open", "enum": ["open", "closed", "all"], "description": "Filter by state"},
                "labels": {"type": "string", "default": "", "description": "Filter by label (comma-separated)"},
                "limit": {"type": "integer", "default": 20, "description": "Max issues to return (max 50)"},
            }, "required": []},
        }, _list_issues),

        ToolEntry("get_github_issue", {
            "name": "get_github_issue",
            "description": "Get full details of a GitHub issue including body and comments.",
            "parameters": {"type": "object", "properties": {
                "number": {"type": "integer", "description": "Issue number"},
            }, "required": ["number"]},
        }, _get_issue),

        ToolEntry("comment_on_issue", {
            "name": "comment_on_issue",
            "description": "Add a comment to a GitHub issue. Use to respond to issues, share progress, or ask clarifying questions.",
            "parameters": {"type": "object", "properties": {
                "number": {"type": "integer", "description": "Issue number"},
                "body": {"type": "string", "description": "Comment text (markdown)"},
            }, "required": ["number", "body"]},
        }, _comment_on_issue),

        ToolEntry("close_github_issue", {
            "name": "close_github_issue",
            "description": "Close a GitHub issue with optional closing comment.",
            "parameters": {"type": "object", "properties": {
                "number": {"type": "integer", "description": "Issue number"},
                "comment": {"type": "string", "default": "", "description": "Optional closing comment"},
            }, "required": ["number"]},
        }, _close_issue),

        ToolEntry("create_github_issue", {
            "name": "create_github_issue",
            "description": "Create a new GitHub issue. Use for tracking tasks, documenting bugs, or planning features.",
            "parameters": {"type": "object", "properties": {
                "title": {"type": "string", "description": "Issue title"},
                "body": {"type": "string", "default": "", "description": "Issue body (markdown)"},
                "labels": {"type": "string", "default": "", "description": "Labels (comma-separated)"},
            }, "required": ["title"]},
        }, _create_issue),
    ]
    # Wrapped here, not at definition: nested handler calls share one invocation.
    tools = [replace(entry, handler=_one_invocation(entry.handler)) for entry in tools]
    for entry in tools:
        entry.schema["parameters"]["properties"]["repo"] = {
            "type": "string", "default": "",
            "description": "Explicit [HOST/]OWNER/REPO. Omit for the active Project repository; required for a Project without a repository folder. An omitted HOST follows GitHub CLI host configuration.",
        }
    return tools
