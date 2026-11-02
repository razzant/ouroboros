"""GitHub transport and exact pull-request settlement for skill publication."""

from __future__ import annotations

import base64
import json
import re
import urllib.parse
from typing import Any, Dict, List, Tuple

from ouroboros.github_pr_identity import pull_request_fields_from_rest, same_pull_request
from ouroboros.secret_masking import redact_known_values
from ouroboros.skill_publish_result import validate_skill_publish_receipt
from ouroboros.skill_publish_client import GitHubError, MISSING_TOKEN_MESSAGE, request
from ouroboros.tools.github import github_token_from_env_or_settings
from ouroboros.tools.registry import ToolContext
from ouroboros.utils import truncate_within_limit

_HEX_OID_RE = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")


class SkillPublishGitHubError(RuntimeError):
    """Closed, candidate-free GitHub transport failure."""

    def __init__(self, reason_code: str, repair_hint: str, *, status: str = "partial",
                 detail: str = "", http_status: int | None = None, operation: str = "") -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.repair_hint = repair_hint
        self.status = status
        # Transport errors already fit (600-char head plus prefix). Also bound
        # malformed success responses before they become diagnostic detail.
        self.detail = truncate_within_limit(
            redact_known_values(detail, [github_token_from_env_or_settings()]), 640,
        ) if detail else ""
        self.http_status = http_status
        self.operation = operation


def github_repair_hint(error: GitHubError, *, operation: str, repository: str,
                       branch: str = "", default: str) -> str:
    """An actionable hint from the producer's status, without guessing a 403's cause."""
    if error.failure == "no_token":
        return MISSING_TOKEN_MESSAGE
    where = f"{operation} on {repository}" + (f" (branch {branch})" if branch else "")
    if error.http_status in (401, 403):
        return (f"GitHub answered HTTP {error.http_status} for {where}; read error_detail. If the "
                "token lacks access, update GITHUB_TOKEN in Settings → Secrets, then retry.")
    if error.http_status == 409:
        return f"GitHub reported a conflict (HTTP 409) for {where}; resolve it on GitHub, then retry."
    if error.failure == "timeout":
        return (f"GitHub did not finish {where} within its time limit; the outcome may be unknown — "
                "inspect it on GitHub before retrying.")
    if error.failure == "invalid_response":
        return f"GitHub answered {where}, but not with the expected JSON; read error_detail."
    return default


def _json_value(
    ctx: ToolContext,
    method: str,
    path: str,
    *,
    reason_code: str,
    operation: str,
    repository: str,
    branch: str = "",
    timeout: int = 30,
    body: Dict[str, Any] | None = None,
    object_required: bool = False,
) -> Any:
    try:
        data = request(method, path, timeout=timeout, body=body)
        if object_required and not isinstance(data, dict):
            raise GitHubError("Expected a JSON object: " + json.dumps(data), failure="invalid_response")
        return data
    except GitHubError as exc:
        raise SkillPublishGitHubError(
            reason_code,
            github_repair_hint(exc, operation=operation, repository=repository, branch=branch,
                               default="Inspect GitHub connectivity and repository access, then retry."),
            detail=str(exc), http_status=exc.http_status, operation=operation,
        ) from exc


def _json_object(ctx: ToolContext, method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
    return _json_value(ctx, method, path, object_required=True, **kwargs)


def github_login(ctx: ToolContext) -> str:
    try:
        info = _json_object(ctx, "GET", "/user", reason_code="github_actor_unavailable",
                            operation="user", repository="<account>")
    except SkillPublishGitHubError as exc:
        exc.status = "blocked"
        raise
    login = info.get("login")
    if not isinstance(login, str) or not login.strip() or len(login) > 80:
        raise SkillPublishGitHubError(
            "github_actor_unavailable", "Repair GitHub authentication, then retry.", status="blocked",
            detail=json.dumps(info), operation="user",
        )
    return login.strip()


def fetch_repository(ctx: ToolContext, owner: str, repo: str) -> Dict[str, Any]:
    """``{default_branch, can_push}`` of the destination for the authenticated account."""
    info = _json_object(ctx, "GET", f"/repos/{owner}/{repo}", reason_code="upstream_read_failed",
                        operation="repos", repository=f"{owner}/{repo}")
    branch = str(info.get("default_branch") or "")
    if not branch:
        raise SkillPublishGitHubError("upstream_read_failed", "Inspect the destination repository, then retry.",
                                      detail=json.dumps(info)[:600], operation="repos")
    permissions = info.get("permissions") if isinstance(info.get("permissions"), dict) else {}
    return {"default_branch": branch, "can_push": permissions.get("push") is True}


def fetch_branch_head(ctx: ToolContext, owner: str, repo: str, branch: str) -> Tuple[str, str]:
    """``(head commit, branch)`` of the destination; an empty branch is the repository's default."""
    if not branch:
        branch = fetch_repository(ctx, owner, repo)["default_branch"]
    ref = _json_object(
        ctx,
        "GET", f"/repos/{owner}/{repo}/git/refs/heads/{urllib.parse.quote(branch)}",
        reason_code="upstream_read_failed",
        operation="git/refs", repository=f"{owner}/{repo}", branch=branch,
    )
    ref_object = ref.get("object")
    base_sha = str((ref_object.get("sha") if isinstance(ref_object, dict) else "") or "")
    if not _HEX_OID_RE.fullmatch(base_sha):
        raise SkillPublishGitHubError(
            "upstream_read_failed",
            "Inspect the configured Hub base branch, then retry.",
            detail=json.dumps(ref), operation="git/refs",
        )
    return base_sha.lower(), branch


def fetch_upstream_file(ctx: ToolContext, owner: str, repo: str, sha: str, path: str) -> bytes | None:
    """One file of the destination at an exact commit; ``None`` when the commit has no such file."""
    repository = f"{owner}/{repo}"
    try:
        content = request("GET", f"/repos/{repository}/contents/{urllib.parse.quote(path)}?ref={sha}")
    except GitHubError as exc:
        if exc.http_status == 404:
            return None
        raise SkillPublishGitHubError(
            "upstream_read_failed",
            github_repair_hint(exc, operation="contents", repository=repository,
                               default="Inspect the destination repository, then retry."),
            detail=str(exc), http_status=exc.http_status, operation="contents",
        ) from exc
    try:
        return base64.b64decode(str(content.get("content") or ""))
    except (ValueError, AttributeError) as exc:
        raise SkillPublishGitHubError(
            "upstream_read_failed", "Inspect the destination repository, then retry.",
            detail=f"{path} at {sha}: {content!r}: {exc}", operation="contents",
        ) from exc


def fetch_tree_paths(ctx: ToolContext, owner: str, repo: str, sha: str, folder: str) -> List[str]:
    """File paths under ``folder`` (relative to it) at an exact commit of the destination."""
    tree = _json_object(ctx, "GET", f"/repos/{owner}/{repo}/git/trees/{sha}?recursive=1",
                        reason_code="upstream_read_failed", operation="git/trees", repository=f"{owner}/{repo}")
    prefix = folder.strip("/") + "/" if folder.strip("/") else ""
    return sorted(str(item.get("path"))[len(prefix):] for item in tree.get("tree") or []
                  if isinstance(item, dict) and item.get("type") == "blob" and str(item.get("path") or "").startswith(prefix))


def fetch_upstream_catalog(ctx: ToolContext, owner: str, repo: str, base_branch: str) -> Tuple[Dict[str, Any], str]:
    base_sha, base_branch = fetch_branch_head(ctx, owner, repo, base_branch)
    content = _json_object(
        ctx,
        "GET", f"/repos/{owner}/{repo}/contents/catalog.json?ref={base_sha}",
        reason_code="upstream_read_failed",
        operation="contents", repository=f"{owner}/{repo}", branch=base_branch,
    )
    try:
        catalog_bytes = base64.b64decode(str(content.get("content") or ""))
        catalog = json.loads(catalog_bytes.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise SkillPublishGitHubError(
            "upstream_catalog_invalid",
            "Repair the upstream Hub catalog, then retry.",
            detail=f"catalog.json at {base_sha}: {exc}", operation="contents",
        ) from exc
    if not isinstance(catalog, dict):
        raise SkillPublishGitHubError(
            "upstream_catalog_invalid",
            "Repair the upstream Hub catalog, then retry.",
            detail=f"catalog.json at {base_sha}: top-level JSON is {type(catalog).__name__}, not an object",
            operation="contents",
        )
    return catalog, base_sha


def prepare_publish_repository(
    ctx: ToolContext,
    attempt: Any,
    *,
    owner: str,
    repo: str,
    base_branch: str,
    login: str,
) -> None:
    """Make sure the account has its fork (its own repository needs none). The submission branch is
    created at the destination's exact commit, which a fork shares, so the fork is not synchronized."""
    repository = f"{login}/{repo}"
    if login.casefold() == owner.casefold():
        attempt.mark("fork_ready", repository=repository, actor=login)
        return
    try:
        request("GET", f"/repos/{repository}")
    except GitHubError:
        _json_object(ctx, "POST", f"/repos/{owner}/{repo}/forks", body={}, timeout=60,
                     reason_code="fork_prepare_failed", operation="repo fork", repository=f"{owner}/{repo}")
    attempt.mark("fork_ready", repository=repository, actor=login)


def _sync_fork(ctx: ToolContext, attempt: Any, login: str, repo: str, base_branch: str) -> None:
    """Bring a fork up to the destination branch; only when GitHub refused a branch at its commit."""
    _json_object(ctx, "POST", f"/repos/{login}/{repo}/merge-upstream", body={"branch": base_branch},
                 timeout=45, reason_code="fork_sync_failed", operation="merge-upstream",
                 repository=f"{login}/{repo}", branch=base_branch)
    if attempt is not None:
        attempt.mark("fork_synced", repository=f"{login}/{repo}", actor=login)


def ensure_branch(ctx: ToolContext, login: str, repo: str, branch: str, base_sha: str, *,
                  base_branch: str = "", attempt: Any = None) -> str:
    try:
        existing = request("GET", f"/repos/{login}/{repo}/git/ref/heads/{urllib.parse.quote(branch)}")
    except GitHubError:
        pass
    else:
        raise SkillPublishGitHubError(
            "submission_branch_exists",
            "Remove the old submission branch or bump the skill version, then retry.",
            detail=json.dumps(existing), operation="git/ref",
        )
    path = f"/repos/{login}/{repo}/git/refs"
    body = {"ref": f"refs/heads/{branch}", "sha": base_sha}
    try:
        created = _json_object(ctx, "POST", path, body=body, reason_code="branch_create_failed",
                               operation="git/refs", repository=f"{login}/{repo}", branch=branch)
    except SkillPublishGitHubError as exc:
        # A fork that never saw this commit refuses a ref to it (422): synchronize once, then retry.
        if exc.http_status != 422 or not base_branch:
            raise
        _sync_fork(ctx, attempt, login, repo, base_branch)
        created = _json_object(ctx, "POST", path, body=body, reason_code="branch_create_failed",
                               operation="git/refs", repository=f"{login}/{repo}", branch=branch)
    branch_sha = str((created.get("object") or {}).get("sha") or "")
    if not _HEX_OID_RE.fullmatch(branch_sha):
        raise SkillPublishGitHubError(
            "branch_create_failed",
            "Inspect the GitHub submission branch, then retry.",
            detail=json.dumps(created), operation="git/refs",
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
    deletions: List[Dict[str, str]] | None = None,
) -> Tuple[str, str]:
    query = """
mutation($input: CreateCommitOnBranchInput!) {
  createCommitOnBranch(input: $input) {
    commit { oid url }
  }
}
""".strip()
    file_changes: Dict[str, Any] = {"additions": additions}
    if deletions:
        file_changes["deletions"] = deletions
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
                "fileChanges": file_changes,
                "expectedHeadOid": base_sha,
            }
        },
    }
    result = _json_object(
        ctx,
        "POST", "/graphql",
        timeout=60,
        body=payload,
        reason_code="commit_create_failed",
        operation="graphql", repository=f"{login}/{repo}", branch=branch,
    )
    if result.get("errors"):
        raise SkillPublishGitHubError(
            "commit_create_failed",
            "Inspect the GitHub submission branch, then retry.",
            detail=json.dumps(result), operation="graphql",
        )
    commit = ((result.get("data") or {}).get("createCommitOnBranch") or {}).get("commit") or {}
    commit_sha = str(commit.get("oid") or "")
    commit_url = str(commit.get("url") or "")
    if not _HEX_OID_RE.fullmatch(commit_sha):
        raise SkillPublishGitHubError(
            "commit_create_failed",
            "Inspect the GitHub submission branch, then retry.",
            detail=json.dumps(result), operation="graphql",
        )
    return commit_sha.lower(), commit_url[:360]


def merge_pull_request(ctx: ToolContext, owner: str, repo: str, number: int, head_sha: str) -> Dict[str, Any]:
    """Merge the publication's own pull request at its exact head (a destination this account may push
    to publishes at once). Never raises: a refused merge leaves the open request as the result."""
    try:
        data = request("PUT", f"/repos/{owner}/{repo}/pulls/{int(number)}/merge",
                       body={"sha": head_sha, "merge_method": "squash"}, timeout=60)
    except GitHubError as exc:
        return {"merged": False, "error": truncate_within_limit(str(exc), 400), "http_status": exc.http_status}
    merged_sha = str(data.get("sha") or "") if isinstance(data, dict) else ""
    if merged_sha and data.get("merged") is True and _HEX_OID_RE.fullmatch(merged_sha):
        return {"merged": True, "commit": merged_sha.lower()}
    detail = truncate_within_limit(redact_known_values(json.dumps(data), [github_token_from_env_or_settings()]), 400)
    return {"merged": False, "error": detail or "GitHub did not merge the request", "http_status": None}


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
    query = urllib.parse.urlencode({"state": "open", "head": f"{login}:{branch}",
                                    "base": base_branch, "per_page": 100})
    rows = _json_value(
        ctx, "GET", f"/repos/{owner}/{repo}/pulls?{query}",
        reason_code="pr_open_indeterminate",
        operation="pulls", repository=f"{owner}/{repo}", branch=branch,
        timeout=30,
    )
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        return None
    row = rows[0]
    if not same_pull_request(pull_request_fields_from_rest(row), head_owner=login, head_branch=branch,
                             base_branch=base_branch, head_sha=commit_sha):
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
    create_error = None
    direct = None
    try:
        result = request("POST", f"/repos/{repository}/pulls", timeout=60,
                         body={"base": base_branch, "head": f"{login}:{branch}", "title": title, "body": body})
    except GitHubError as exc:
        create_error = exc
    else:
        direct = _receipt_from_url(
            str(result.get("html_url") or "") if isinstance(result, dict) else "",
            repository=repository,
            skill=attempt.skill,
            snapshot_hash=attempt.snapshot_hash,
            ruleset_sha256=str(attempt.scanner.get("ruleset_sha256") or ""),
        )
    if direct is not None:
        return direct
    try:
        settled = _lookup_open_pr_receipt(
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
    except SkillPublishGitHubError as exc:
        if create_error is None:
            # The mutator succeeded and only the read-only settlement failed: say so
            # before the settlement's own hint, or "retry" would risk a duplicate PR.
            raise SkillPublishGitHubError(
                exc.reason_code,
                "GitHub PR creation reported success but the pull request could not be settled; "
                "inspect the recorded branch and commit on GitHub before retrying. " + exc.repair_hint,
                status=exc.status, detail=exc.detail, http_status=exc.http_status, operation=exc.operation,
            ) from exc
        # Retain the mutator's evidence if the read-only settlement also failed.
        settled = None
    if settled is None and create_error is not None:
        raise SkillPublishGitHubError(
            "pr_open_indeterminate",
            github_repair_hint(create_error, operation="pr create", repository=repository, branch=branch,
                               default="Inspect the recorded branch and commit before deciding whether to retry."),
            detail=str(create_error), http_status=create_error.http_status, operation="pr create",
        )
    return settled


__all__ = [
    "SkillPublishGitHubError",
    "commit_payload",
    "create_pr_receipt",
    "ensure_branch",
    "fetch_branch_head",
    "fetch_repository",
    "fetch_tree_paths",
    "fetch_upstream_catalog",
    "fetch_upstream_file",
    "github_login",
    "github_repair_hint",
    "merge_pull_request",
    "prepare_publish_repository",
]
