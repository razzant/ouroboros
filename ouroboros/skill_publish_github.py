"""GitHub transport and exact pull-request settlement for skill publication."""

from __future__ import annotations

import base64
import json
import re
import urllib.parse
from typing import Any, Dict, List, Tuple

from ouroboros.skill_publish_result import validate_skill_publish_receipt
from ouroboros.tools.github import _gh_cmd
from ouroboros.tools.registry import ToolContext

_HEX_OID_RE = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")


class SkillPublishGitHubError(RuntimeError):
    """Closed, candidate-free GitHub transport failure."""

    def __init__(self, reason_code: str, repair_hint: str, *, status: str = "partial") -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.repair_hint = repair_hint
        self.status = status


def _json_value(
    ctx: ToolContext,
    args: List[str],
    *,
    reason_code: str,
    timeout: int = 30,
    input_data: str | None = None,
) -> Any:
    raw = _gh_cmd(args, ctx, timeout=timeout, input_data=input_data)
    if raw.startswith("⚠️"):
        raise SkillPublishGitHubError(
            reason_code,
            "Inspect GitHub connectivity and repository access, then retry.",
        )
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise SkillPublishGitHubError(
            reason_code,
            "Inspect GitHub connectivity and repository access, then retry.",
        ) from exc


def _json_object(
    ctx: ToolContext,
    args: List[str],
    *,
    reason_code: str,
    timeout: int = 30,
    input_data: str | None = None,
) -> Dict[str, Any]:
    data = _json_value(
        ctx,
        args,
        reason_code=reason_code,
        timeout=timeout,
        input_data=input_data,
    )
    if not isinstance(data, dict):
        raise SkillPublishGitHubError(
            reason_code,
            "Inspect GitHub connectivity and repository access, then retry.",
        )
    return data


def github_login(ctx: ToolContext) -> str:
    raw = _gh_cmd(["api", "/user", "--jq", ".login"], ctx).strip()
    if raw.startswith("⚠️") or not raw or len(raw) > 80:
        raise SkillPublishGitHubError(
            "github_actor_unavailable",
            "Repair GitHub authentication, then retry.",
            status="blocked",
        )
    return raw


def fetch_upstream_catalog(ctx: ToolContext, owner: str, repo: str, base_branch: str) -> Tuple[Dict[str, Any], str]:
    ref = _json_object(
        ctx,
        ["api", f"/repos/{owner}/{repo}/git/refs/heads/{base_branch}"],
        reason_code="upstream_read_failed",
    )
    base_sha = str((ref.get("object") or {}).get("sha") or "")
    if not _HEX_OID_RE.fullmatch(base_sha):
        raise SkillPublishGitHubError(
            "upstream_read_failed",
            "Inspect the configured Hub base branch, then retry.",
        )
    content = _json_object(
        ctx,
        ["api", f"/repos/{owner}/{repo}/contents/catalog.json?ref={base_branch}"],
        reason_code="upstream_read_failed",
    )
    try:
        catalog_bytes = base64.b64decode(str(content.get("content") or ""))
        catalog = json.loads(catalog_bytes.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise SkillPublishGitHubError(
            "upstream_catalog_invalid",
            "Repair the upstream Hub catalog, then retry.",
        ) from exc
    if not isinstance(catalog, dict):
        raise SkillPublishGitHubError(
            "upstream_catalog_invalid",
            "Repair the upstream Hub catalog, then retry.",
        )
    return catalog, base_sha.lower()


def prepare_publish_repository(
    ctx: ToolContext,
    attempt: Any,
    *,
    owner: str,
    repo: str,
    base_branch: str,
    login: str,
) -> None:
    repository = f"{login}/{repo}"
    if login.casefold() == owner.casefold():
        attempt.mark("fork_ready", repository=repository, actor=login)
        return
    existing = _gh_cmd(["repo", "view", repository, "--json", "name"], ctx)
    if existing.startswith("⚠️"):
        created = _gh_cmd(
            ["repo", "fork", f"{owner}/{repo}", "--clone=false"],
            ctx,
            timeout=60,
        )
        if created.startswith("⚠️"):
            raise SkillPublishGitHubError(
                "fork_prepare_failed",
                "Repair the GitHub fork, then retry.",
            )
    attempt.mark("fork_ready", repository=repository, actor=login)
    merged = _gh_cmd(
        [
            "api",
            "-X",
            "POST",
            f"/repos/{login}/{repo}/merge-upstream",
            "-f",
            f"branch={base_branch}",
        ],
        ctx,
        timeout=45,
    )
    if merged.startswith("⚠️"):
        raise SkillPublishGitHubError(
            "fork_sync_failed",
            "Repair or synchronize the GitHub fork, then retry.",
        )
    attempt.mark("fork_synced", repository=repository, actor=login)


def ensure_branch(ctx: ToolContext, login: str, repo: str, branch: str, base_sha: str) -> str:
    existing = _gh_cmd(
        ["api", f"/repos/{login}/{repo}/git/ref/heads/{branch}"],
        ctx,
    )
    if not existing.startswith("⚠️"):
        raise SkillPublishGitHubError(
            "submission_branch_exists",
            "Remove the old submission branch or bump the skill version, then retry.",
        )
    created = _json_object(
        ctx,
        [
            "api",
            "-X",
            "POST",
            f"/repos/{login}/{repo}/git/refs",
            "-f",
            f"ref=refs/heads/{branch}",
            "-f",
            f"sha={base_sha}",
        ],
        reason_code="branch_create_failed",
    )
    branch_sha = str((created.get("object") or {}).get("sha") or "")
    if not _HEX_OID_RE.fullmatch(branch_sha):
        raise SkillPublishGitHubError(
            "branch_create_failed",
            "Inspect the GitHub submission branch, then retry.",
        )
    return branch_sha.lower()


def commit_payload(
    ctx: ToolContext,
    login: str,
    repo: str,
    branch: str,
    base_sha: str,
    headline: str,
    additions: List[Dict[str, str]],
) -> Tuple[str, str]:
    query = """
mutation($input: CreateCommitOnBranchInput!) {
  createCommitOnBranch(input: $input) {
    commit { oid url }
  }
}
""".strip()
    payload = {
        "query": query,
        "variables": {
            "input": {
                "branch": {
                    "repositoryNameWithOwner": f"{login}/{repo}",
                    "branchName": branch,
                },
                "message": {
                    "headline": headline,
                    "body": ("Co-authored-by: Ouroboros <311266734+ouroboros-agent@users.noreply.github.com>"),
                },
                "fileChanges": {"additions": additions},
                "expectedHeadOid": base_sha,
            }
        },
    }
    result = _json_object(
        ctx,
        ["api", "graphql", "--input", "-"],
        timeout=60,
        input_data=json.dumps(payload),
        reason_code="commit_create_failed",
    )
    if result.get("errors"):
        raise SkillPublishGitHubError(
            "commit_create_failed",
            "Inspect the GitHub submission branch, then retry.",
        )
    commit = ((result.get("data") or {}).get("createCommitOnBranch") or {}).get("commit") or {}
    commit_sha = str(commit.get("oid") or "")
    commit_url = str(commit.get("url") or "")
    if not _HEX_OID_RE.fullmatch(commit_sha):
        raise SkillPublishGitHubError(
            "commit_create_failed",
            "Inspect the GitHub submission branch, then retry.",
        )
    return commit_sha.lower(), commit_url[:360]


def _receipt_from_url(
    url: str,
    *,
    repository: str,
    skill: str,
    snapshot_hash: str,
    ruleset_sha256: str,
) -> Dict[str, Any] | None:
    candidate = str(url or "").strip()
    if "\n" in candidate or "\r" in candidate:
        return None
    try:
        segments = urllib.parse.urlsplit(candidate).path.split("/")
        number = int(segments[-1]) if segments[-1].isdigit() else 0
    except (TypeError, ValueError):
        return None
    receipt = {
        "kind": "github_pull_request",
        "repository": repository,
        "url": candidate,
        "number": number,
        "skill": skill,
        "snapshot_hash": snapshot_hash,
        "ruleset_sha256": ruleset_sha256,
    }
    return validate_skill_publish_receipt(
        receipt,
        expected_repository=repository,
        expected_skill=skill,
        expected_snapshot_hash=snapshot_hash,
        expected_ruleset_sha256=ruleset_sha256,
    )


def _lookup_open_pr_receipt(
    ctx: ToolContext,
    *,
    owner: str,
    repo: str,
    base_branch: str,
    login: str,
    branch: str,
    commit_sha: str,
    skill: str,
    snapshot_hash: str,
    ruleset_sha256: str,
) -> Dict[str, Any] | None:
    try:
        rows = _json_value(
            ctx,
            [
                "api",
                "--method",
                "GET",
                f"/repos/{owner}/{repo}/pulls",
                "-f",
                "state=open",
                "-f",
                f"head={login}:{branch}",
                "-f",
                f"base={base_branch}",
                "-f",
                "per_page=100",
            ],
            reason_code="pr_open_indeterminate",
            timeout=30,
        )
    except SkillPublishGitHubError:
        return None
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        return None
    row = rows[0]
    head = row.get("head") if isinstance(row.get("head"), dict) else {}
    base = row.get("base") if isinstance(row.get("base"), dict) else {}
    head_repo = head.get("repo") if isinstance(head.get("repo"), dict) else {}
    head_owner = head_repo.get("owner") if isinstance(head_repo.get("owner"), dict) else {}
    if (
        str(head.get("sha") or "").casefold() != commit_sha.casefold()
        or str(head.get("ref") or "") != branch
        or str(head_owner.get("login") or "").casefold() != login.casefold()
        or str(base.get("ref") or "") != base_branch
    ):
        return None
    receipt = _receipt_from_url(
        str(row.get("html_url") or ""),
        repository=f"{owner}/{repo}",
        skill=skill,
        snapshot_hash=snapshot_hash,
        ruleset_sha256=ruleset_sha256,
    )
    return receipt if receipt is not None and receipt["number"] == row.get("number") else None


def create_pr_receipt(
    ctx: ToolContext,
    attempt: Any,
    *,
    owner: str,
    repo: str,
    base_branch: str,
    login: str,
    branch: str,
    title: str,
    body: str,
    commit_sha: str,
) -> Dict[str, Any] | None:
    repository = f"{owner}/{repo}"
    attempt.mark(
        "pr_create_attempted",
        repository=repository,
        actor=login,
        branch=branch,
        commit_sha=commit_sha,
    )
    raw = _gh_cmd(
        [
            "pr",
            "create",
            "--repo",
            repository,
            "--base",
            base_branch,
            "--head",
            f"{login}:{branch}",
            "--title",
            title,
            "--body-file",
            "-",
        ],
        ctx,
        timeout=60,
        input_data=body,
    )
    direct = None
    if not raw.startswith("⚠️"):
        direct = _receipt_from_url(
            raw,
            repository=repository,
            skill=attempt.skill,
            snapshot_hash=attempt.snapshot_hash,
            ruleset_sha256=str(attempt.scanner.get("ruleset_sha256") or ""),
        )
    if direct is not None:
        return direct
    return _lookup_open_pr_receipt(
        ctx,
        owner=owner,
        repo=repo,
        base_branch=base_branch,
        login=login,
        branch=branch,
        commit_sha=commit_sha,
        skill=attempt.skill,
        snapshot_hash=attempt.snapshot_hash,
        ruleset_sha256=str(attempt.scanner.get("ruleset_sha256") or ""),
    )


__all__ = [
    "SkillPublishGitHubError",
    "commit_payload",
    "create_pr_receipt",
    "ensure_branch",
    "fetch_upstream_catalog",
    "github_login",
    "prepare_publish_repository",
]
