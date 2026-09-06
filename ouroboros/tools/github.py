"""GitHub tools: issues, comments, reactions."""

from __future__ import annotations

import json
import logging
import os
import pathlib
import subprocess
from typing import List, Optional

from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.utils import truncate_review_artifact as _truncate_with_notice

log = logging.getLogger(__name__)

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


def _gh_cmd(args: List[str], ctx: ToolContext, timeout: int = 30, input_data: Optional[str] = None,
            *, repo: Optional[str] = None) -> str:
    # None keeps explicit API/Hub callers on their existing transport contract.
    # The eight repository tools pass a string, including '' for Project focus.
    try:
        cwd, env = pathlib.Path(ctx.repo_dir), _gh_env(ctx)
        cmd = ["gh", *args]
        if repo is not None:
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
                    return f"⚠️ GH_TARGET_UNAVAILABLE: {note or 'The selected Project directory is unavailable.'}"
                if project and not selected:
                    return "⚠️ GH_TARGET_REQUIRED: this Project has no repository directory; pass repo='[HOST/]OWNER/REPO'."
            binding = build_resolved_resource_binding(ctx, operation="shell", process_cwd="")
            cwd = binding.target_path
            if workspace and cwd != pathlib.Path(workspace).resolve(strict=False):
                return "⚠️ GH_TARGET_UNAVAILABLE: the task's Project binding could not be resolved."
            if workspace or room_dir or project:
                env.pop("GH_REPO", None)  # Ambient defaults cannot replace the selected Project.
            if repo:
                cmd.extend(["--repo", repo])
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
            err = (res.stderr or "").strip()
            return f"⚠️ GH_ERROR: {err.split(chr(10))[0][:200]}"
        return res.stdout.strip()
    except FileNotFoundError:
        return "⚠️ GH_ERROR: `gh` CLI not found. Install GitHub CLI and ensure it is on PATH (https://cli.github.com/)"
    except subprocess.TimeoutExpired:
        return f"⚠️ GH_TIMEOUT: exceeded {timeout}s."
    except Exception as e:
        return f"⚠️ GH_ERROR: {e}"

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
        return f"⚠️ Failed to parse issues JSON: {raw[:500]}"

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
        return "⚠️ issue number must be positive"

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
        return f"⚠️ Failed to parse issue JSON: {raw[:500]}"

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
        return "⚠️ issue number must be positive"

    if not body or not body.strip():
        return "⚠️ Comment body cannot be empty."

    args = ["issue", "comment", str(number), "--body-file", "-"]
    raw = _gh_cmd(args, ctx, input_data=body, repo=repo)
    if raw.startswith("⚠️"):
        return raw
    return f"✅ Comment added to issue #{number}."


def _close_issue(ctx: ToolContext, number: int, comment: str = "", repo: str = "") -> str:
    if number <= 0:
        return "⚠️ issue number must be positive"

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
        return f"⚠️ Failed to parse PRs JSON: {raw[:500]}"

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
        return "⚠️ PR number must be positive."

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
        return f"⚠️ Failed to parse PR JSON: {raw[:500]}"

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
            f"  5. stage_pr_merge(branch='integrate/pr-{number}') → preflight_review → commit_reviewed\n"
            f"  6. comment_on_pr(number={number}, body='Integrated as ...')"
        )

    return "\n".join(lines)


def _comment_on_pr(ctx: ToolContext, number: int, body: str, repo: str = "") -> str:
    if number <= 0:
        return "⚠️ PR number must be positive."
    if not (body or "").strip():
        return "⚠️ Comment body cannot be empty."

    args = ["pr", "comment", str(number), "--body-file", "-"]
    raw = _gh_cmd(args, ctx, input_data=body, repo=repo)
    if raw.startswith("⚠️"):
        return raw
    return f"✅ Comment added to PR #{number}."


def _create_issue(ctx: ToolContext, title: str, body: str = "", labels: str = "", repo: str = "") -> str:
    if not title or not title.strip():
        return "⚠️ Issue title cannot be empty."

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
    for entry in tools:
        entry.schema["parameters"]["properties"]["repo"] = {
            "type": "string", "default": "",
            "description": "Explicit [HOST/]OWNER/REPO. Omit for the active Project repository; required for a Project without a repository folder. An omitted HOST follows GitHub CLI host configuration.",
        }
    return tools
