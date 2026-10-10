"""Git PR integration tools with author-preserving cherry-picks and reviewed merges."""

from __future__ import annotations

import logging
import os
import pathlib
import re
import subprocess
from typing import List, Optional

from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.tools.git import _acquire_git_lock, _release_git_lock, _sanitize_git_error

log = logging.getLogger(__name__)

_PR_BRANCH_PREFIX = "integrate/pr-"
# The verbs that check out, commit or stage in the body's own worktree: an ordinary
# author runs them in its body candidate (``body_candidate.authoring_seam``).
BODY_WORKTREE_TOOLS = frozenset({
    "create_integration_branch", "cherry_pick_pr_commits", "stage_adaptations", "stage_pr_merge",
})


def _g(args: List[str], cwd: pathlib.Path,
       env: Optional[dict] = None, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + args, cwd=str(cwd),
        capture_output=True, text=True, timeout=timeout,
        **({"env": env} if env is not None else {}),
    )


def _ouroboros_committer_env(repo_dir: pathlib.Path) -> dict:
    env = os.environ.copy()
    name_r = subprocess.run(
        ["git", "config", "--local", "user.name"], cwd=str(repo_dir),
        capture_output=True, text=True, timeout=5,
    )
    email_r = subprocess.run(
        ["git", "config", "--local", "user.email"], cwd=str(repo_dir),
        capture_output=True, text=True, timeout=5,
    )
    local_name = name_r.stdout.strip() if name_r.returncode == 0 else ""
    local_email = email_r.stdout.strip() if email_r.returncode == 0 else ""
    if local_name and local_email:
        env["GIT_COMMITTER_NAME"] = local_name
        env["GIT_COMMITTER_EMAIL"] = local_email
    else:
        env["GIT_COMMITTER_NAME"] = "Ouroboros"
        env["GIT_COMMITTER_EMAIL"] = "ouroboros@local.mac"
    return env


def _validate_git_ref_arg(value: str, param_name: str) -> Optional[str]:
    if value.startswith("-"):
        return (
            f"⚠️ INVALID_ARG: {param_name!r} must not start with '-' "
            f"(got {value!r}). Option-like values are rejected for safety."
        )
    return None


_AUTHOR_FORBIDDEN_CHARS = ("\r", "\n", "\t", "<", ">", "\x00")
_SHA_PATTERN = re.compile(r"^[0-9a-f]{7,40}$", re.IGNORECASE)


def _validate_override_author(override: Optional[dict]) -> Optional[str]:
    if override is None:
        return None
    if not isinstance(override, dict):
        return (
            "⚠️ CHERRY_PICK_ERROR: override_author must be a dict with "
            "'name' and 'email' keys (got "
            f"{type(override).__name__})."
        )
    name = override.get("name")
    email = override.get("email")
    if not isinstance(name, str) or not name.strip():
        return (
            "⚠️ CHERRY_PICK_ERROR: override_author['name'] must be a "
            "non-empty string."
        )
    if not isinstance(email, str) or not email.strip():
        return (
            "⚠️ CHERRY_PICK_ERROR: override_author['email'] must be a "
            "non-empty string."
        )
    if "@" not in email:
        return (
            "⚠️ CHERRY_PICK_ERROR: override_author['email'] must contain '@' "
            f"(got {email!r})."
        )
    for ch in _AUTHOR_FORBIDDEN_CHARS:
        if ch in name or ch in email:
            return (
                "⚠️ CHERRY_PICK_ERROR: override_author contains a forbidden "
                "character (newline, CR, tab, NUL, '<', or '>'). These "
                "would corrupt git commit metadata."
            )
    return None


def _fetch_pr_ref(ctx: ToolContext, pr_number: int, remote: str = "origin") -> str:
    if pr_number <= 0:
        return "⚠️ PR_FETCH_ERROR: pr_number must be a positive integer."
    err = _validate_git_ref_arg(remote, "remote")
    if err:
        return f"⚠️ PR_FETCH_ERROR: {err}"

    repo_dir = pathlib.Path(ctx.repo_dir)
    local_ref = f"pr/{pr_number}"
    refspec = f"+refs/pull/{pr_number}/head:{local_ref}"

    lock = _acquire_git_lock(ctx)
    try:
        result = _g(["fetch", remote, refspec], repo_dir, timeout=120)
        if result.returncode != 0:
            err = (result.stderr or "").strip()
            return f"⚠️ PR_FETCH_ERROR: {_sanitize_git_error(err)}"

        fetched_sha = _g(["rev-parse", local_ref], repo_dir).stdout.strip()

        base_r = _g(["merge-base", "ouroboros", local_ref], repo_dir)
        base_sha = base_r.stdout.strip() if base_r.returncode == 0 else ""
        if base_sha:
            count_r = _g(["rev-list", "--count", f"{base_sha}..{local_ref}"], repo_dir)
            commit_count = count_r.stdout.strip() if count_r.returncode == 0 else "?"
            log_r = _g(["log", "--format=%h | %an <%ae> | %ai | %s",
                        f"{base_sha}..{local_ref}"], repo_dir)
        else:
            commit_count = "?"
            log_r = _g(["log", "--format=%h | %an <%ae> | %ai | %s", local_ref], repo_dir)
    finally:
        _release_git_lock(lock)

    commit_log = (log_r.stdout.strip()
                  if log_r.returncode == 0 else "(could not list commits)")

    return (
        f"✅ Fetched PR #{pr_number} → local ref '{local_ref}'\n"
        f"  HEAD SHA: {fetched_sha[:12]}\n"
        f"  Commits vs ouroboros: {commit_count}\n\n"
        f"Commits (author | date | subject):\n{commit_log}\n\n"
        f"Next step: create_integration_branch(pr_number={pr_number})"
    )


def _create_integration_branch(
    ctx: ToolContext,
    pr_number: int,
    base_branch: str = "",
) -> str:
    if pr_number <= 0:
        return "⚠️ PR_BRANCH_ERROR: pr_number must be a positive integer."
    # The body's working branch: ouroboros, or the bound candidate's branch.
    base_branch = str(base_branch or getattr(ctx, "branch_dev", "") or "ouroboros").strip()
    err = _validate_git_ref_arg(base_branch, "base_branch")
    if err:
        return f"⚠️ PR_BRANCH_ERROR: {err}"

    repo_dir = pathlib.Path(ctx.repo_dir)
    branch_name = f"{_PR_BRANCH_PREFIX}{pr_number}"

    if _g(["branch", "--list", branch_name], repo_dir).stdout.strip():
        return (
            f"⚠️ PR_BRANCH_ERROR: Branch '{branch_name}' already exists.\n"
            f"To start fresh: git branch -D {branch_name}"
        )

    status_r = _g(["status", "--porcelain"], repo_dir)
    if status_r.returncode == 0 and status_r.stdout.strip():
        return (
            "⚠️ PR_BRANCH_ERROR: Working tree has uncommitted or untracked changes.\n"
            "Commit, stash, or clean before creating an integration branch.\n"
            f"Unclean files:\n{status_r.stdout.strip()[:300]}"
        )

    lock = _acquire_git_lock(ctx)
    try:
        head_r = _g(["rev-parse", "--abbrev-ref", "HEAD"], repo_dir)
        current_branch = head_r.stdout.strip() if head_r.returncode == 0 else "?"

        co = _g(["checkout", base_branch], repo_dir)
        if co.returncode != 0:
            return (
                f"⚠️ PR_BRANCH_ERROR: Cannot checkout '{base_branch}': "
                f"{_sanitize_git_error((co.stderr or '').strip())}"
            )

        br = _g(["checkout", "-b", branch_name], repo_dir)
        if br.returncode != 0:
            return (
                f"⚠️ PR_BRANCH_ERROR: Cannot create branch '{branch_name}': "
                f"{_sanitize_git_error((br.stderr or '').strip())}"
            )

        base_sha = _g(["rev-parse", "HEAD"], repo_dir).stdout.strip()[:12]
    finally:
        _release_git_lock(lock)

    return (
        f"✅ Created integration branch '{branch_name}' from '{base_branch}' ({base_sha})\n"
        f"  (was on: {current_branch})\n\n"
        f"Next steps:\n"
        f"  1. cherry_pick_pr_commits(shas=[...])    ← replays external commits with\n"
        f"                                              original author attribution\n"
        f"  2. stage_adaptations()                   ← optional: stage Ouroboros\n"
        f"                                              adaptation changes (no commit)\n"
        f"  3. stage_pr_merge(branch='{branch_name}') → commit_reviewed\n"
        f"     (staged adaptations from step 2 land in the final merge commit)"
    )


def _rollback_failed_amend(
    ctx: ToolContext,
    repo_dir: pathlib.Path,
    sha: str,
    amend_r: subprocess.CompletedProcess,
    applied: List[str],
) -> str:
    amend_err = (amend_r.stderr or amend_r.stdout or "").strip()
    _g(["reset", "--hard", "HEAD~1"], repo_dir)
    applied.pop()
    return (
        f"⚠️ CHERRY_PICK_ERROR: author amend failed on {sha[:12]} "
        f"(rolled back to pre-commit state):\n{amend_err[:500]}\n\n"
        f"Applied and kept before amend failure: "
        f"{[s for s in applied] or 'none'}\n"
        f"Amend failures indicate a git config or author-string problem, "
        f"not a PR content problem — fail-fast is intentional."
    )


def _validate_sha_list(
    shas: List[str],
    repo_dir: pathlib.Path,
) -> object:
    resolved: List[str] = []
    for sha in shas:
        sha = sha.strip()
        if not _SHA_PATTERN.match(sha):
            return (
                f"⚠️ CHERRY_PICK_ERROR: '{sha}' is not a commit SHA "
                f"(expected 7-40 hex characters). Symbolic refs like branch "
                f"names, HEAD, or tag names are not accepted — resolve them "
                f"to a commit SHA first via `git rev-parse` or fetch_pr_ref."
            )
        r = _g(["rev-parse", "--verify", f"{sha}^{{commit}}"], repo_dir)
        if r.returncode != 0:
            return (
                f"⚠️ CHERRY_PICK_ERROR: Cannot resolve SHA '{sha}' to a commit. "
                f"Verify it was fetched with fetch_pr_ref and is a commit object."
            )
        resolved.append(r.stdout.strip())
        type_r = _g(["cat-file", "-t", sha], repo_dir)
        if type_r.returncode != 0 or type_r.stdout.strip() != "commit":
            obj_type = type_r.stdout.strip() or "unknown"
            return (
                f"⚠️ CHERRY_PICK_ERROR: '{sha}' is a {obj_type!r} object, "
                f"not a commit. Only commit SHAs are accepted — tags and other "
                f"refs must be dereferenced first (e.g. by looking up the target "
                f"commit SHA via fetch_pr_ref or git log)."
            )
    return resolved


def _amend_author_on_head(
    repo_dir: pathlib.Path,
    override_author: dict,
    orig_date: str,
    committer_env: dict,
) -> subprocess.CompletedProcess:
    author_str = f'{override_author["name"]} <{override_author["email"]}>'
    return _g(
        ["commit", "--amend", "--no-edit",
         f"--author={author_str}",
         f"--date={orig_date}"],
        repo_dir, env=committer_env,
    )


def _cherry_pick_pr_commits(
    ctx: ToolContext,
    shas: List[str],
    stop_on_conflict: bool = True,
    override_author: Optional[dict] = None,
) -> str:
    override_error = _validate_override_author(override_author)
    if override_error:
        return override_error

    if not shas:
        return "⚠️ CHERRY_PICK_ERROR: shas list cannot be empty."

    repo_dir = pathlib.Path(ctx.repo_dir)

    head_r = _g(["rev-parse", "--abbrev-ref", "HEAD"], repo_dir)
    current_branch = head_r.stdout.strip() if head_r.returncode == 0 else ""
    if not current_branch.startswith(_PR_BRANCH_PREFIX):
        return (
            f"⚠️ CHERRY_PICK_ERROR: Current branch is '{current_branch}', "
            f"not an integration branch (expected prefix '{_PR_BRANCH_PREFIX}').\n"
            f"Run create_integration_branch first."
        )

    resolved_or_error = _validate_sha_list(shas, repo_dir)
    if isinstance(resolved_or_error, str):
        return resolved_or_error
    resolved = resolved_or_error

    if (_g(["diff", "--cached", "--name-only"], repo_dir).stdout.strip()
            or _g(["diff", "--name-only"], repo_dir).stdout.strip()):
        return (
            "⚠️ CHERRY_PICK_ERROR: Working tree has staged or unstaged changes.\n"
            "Commit or restore to HEAD before cherry-picking."
        )

    committer_env = _ouroboros_committer_env(repo_dir)

    lock = _acquire_git_lock(ctx)
    applied: List[str] = []
    skipped: List[str] = []
    attribution_lines: List[str] = []
    try:
        for sha in resolved:
            orig_date = ""
            if override_author is not None:
                date_r = _g(["log", "-1", "--format=%aI", sha], repo_dir)
                if date_r.returncode != 0 or not date_r.stdout.strip():
                    return (
                        f"⚠️ CHERRY_PICK_ERROR: Cannot read author date for {sha[:12]} "
                        f"(git log returned {date_r.returncode}). Aborting before "
                        f"cherry-pick to avoid losing the original timestamp.\n"
                        f"Applied before date-read failure: "
                        f"{[s for s in applied] or 'none'}"
                    )
                orig_date = date_r.stdout.strip()

            result = _g(["cherry-pick", "--no-edit", sha], repo_dir, env=committer_env)
            if result.returncode != 0:
                err = (result.stderr or result.stdout or "").strip()
                _g(["cherry-pick", "--abort"], repo_dir)
                if stop_on_conflict:
                    return (
                        f"⚠️ CHERRY_PICK_CONFLICT on {sha[:12]}:\n{err[:500]}\n\n"
                        f"Applied before conflict: {[s[:12] for s in applied] or 'none'}\n"
                        f"Run `git cherry-pick --abort` if needed, then resolve manually."
                    )
                skipped.append(sha[:12])
                continue

            applied.append(sha[:12])

            if override_author is not None:
                amend_r = _amend_author_on_head(
                    repo_dir, override_author, orig_date, committer_env,
                )
                if amend_r.returncode != 0:
                    return _rollback_failed_amend(
                        ctx, repo_dir, sha, amend_r, applied,
                    )
            author_r = _g(["log", "-1", "--format=%an <%ae>", sha], repo_dir)
            if author_r.returncode != 0:
                continue
            attr = author_r.stdout.strip()
            if attr and attr not in attribution_lines:
                attribution_lines.append(attr)
    finally:
        _release_git_lock(lock)

    if not applied:
        return (
            f"⚠️ CHERRY_PICK_ERROR: No commits were successfully applied.\n"
            f"Skipped (conflict): {skipped}"
        )

    override_note = ""
    if override_author is not None:
        override_note = (
            f"\n\nAuthor override applied: "
            f"{override_author['name']} <{override_author['email']}> "
            f"— original author dates preserved, repo-local committer "
            f"identity (Ouroboros fallback) unchanged."
        )
        attribution_description = (
            "author identity rewritten via override; "
            "original author dates and repo-local committer identity preserved"
        )
        hint_lead = (
            "Attribution: override author now appears on all cherry-picked "
            "commits in the integration branch."
        )
    else:
        attribution_description = "real commits, original authorship preserved"
        hint_lead = (
            "Attribution: original author commits preserved in integration branch."
        )

    partial = ""
    if skipped:
        partial = (
            f"\n\n⚠️ PARTIAL INGESTION — skipped (conflict): {skipped}\n"
            f"Resolve manually or re-run with those SHAs omitted."
        )

    if attribution_lines:
        co_lines = "\n".join(f"Co-authored-by: {a}" for a in attribution_lines)
        author_hint = (
            f"\n\n{hint_lead}\n"
            f"Include in your final commit_reviewed (merge) message:\n{co_lines}"
        )
    else:
        author_hint = ""

    return (
        f"✅ Cherry-picked {len(applied)} of {len(resolved)} commit(s) onto "
        f"'{current_branch}' ({attribution_description}):\n"
        + "\n".join(f"  {sha}" for sha in applied)
        + f"\n\nNext:\n"
          f"  stage_adaptations()                      ← optional: stage Ouroboros\n"
          f"                                              adaptation changes (no commit)\n"
          f"  stage_pr_merge(branch='{current_branch}') → commit_reviewed\n"
          f"  (staged adaptations land in the merge commit — no intermediate commit needed)"
        + override_note
        + author_hint
        + partial
    )


def _stage_adaptations(ctx: ToolContext) -> str:
    repo_dir = pathlib.Path(ctx.repo_dir)

    head_r = _g(["rev-parse", "--abbrev-ref", "HEAD"], repo_dir)
    current_branch = head_r.stdout.strip() if head_r.returncode == 0 else ""
    if not current_branch.startswith(_PR_BRANCH_PREFIX):
        return (
            f"⚠️ STAGE_ADAPTATIONS_ERROR: Current branch is '{current_branch}', "
            f"not an integration branch. Only use stage_adaptations on integrate/pr-* branches."
        )

    lock = _acquire_git_lock(ctx)
    try:
        _g(["add", "-A"], repo_dir)
        staged = _g(["diff", "--cached", "--name-only"], repo_dir).stdout.strip()
        if not staged:
            return "⚠️ STAGE_ADAPTATIONS_ERROR: Nothing to stage (working tree is clean)."
    finally:
        _release_git_lock(lock)

    files = staged.splitlines()
    return (
        f"✅ Staged {len(files)} file(s) on '{current_branch}' (NOT committed):\n"
        + "\n".join(f"  {f}" for f in files[:20])
        + (f"\n  ... and {len(files)-20} more" if len(files) > 20 else "")
        + f"\n\nNext: stage_pr_merge(branch='{current_branch}') — do NOT commit here;\n"
          f"  adaptation changes land in the merge commit on {ctx.branch_dev}."
    )


def _stage_pr_merge(
    ctx: ToolContext,
    branch: str,
) -> str:
    branch = (branch or "").strip()
    if not branch:
        return "⚠️ PR_MERGE_ERROR: branch parameter is required."
    err = _validate_git_ref_arg(branch, "branch")
    if err:
        return f"⚠️ PR_MERGE_ERROR: {err}"
    target_branch = ctx.branch_dev
    if branch == target_branch:
        return (
            f"⚠️ PR_MERGE_ERROR: branch and target_branch are the same ('{branch}'). "
            f"Specify an integration branch (integrate/pr-N)."
        )

    repo_dir = pathlib.Path(ctx.repo_dir)

    if not _g(["branch", "--list", branch], repo_dir).stdout.strip():
        return f"⚠️ PR_MERGE_ERROR: Branch '{branch}' does not exist."

    head_r = _g(["rev-parse", "--abbrev-ref", "HEAD"], pathlib.Path(ctx.repo_dir))
    current = head_r.stdout.strip() if head_r.returncode == 0 else ""
    if current != branch:
        return (
            f"⚠️ PR_MERGE_ERROR: Must be on '{branch}' before calling stage_pr_merge "
            f"(currently on '{current}'). Checkout the integration branch first."
        )

    lock = _acquire_git_lock(ctx)
    try:
        porcelain = _g(["status", "--porcelain"], repo_dir).stdout or ""
        dirty_lines = [
            ln for ln in porcelain.splitlines()
            if len(ln) >= 2 and (ln[1] != " " or ln[:2] == "??")
        ]
        if dirty_lines:
            sample = "\n".join(dirty_lines[:10])
            return (
                f"⚠️ PR_MERGE_ERROR: Integration branch has unstaged or untracked changes.\n"
                f"Stage all intentional changes with stage_adaptations() first.\n"
                f"Unclean files:\n{sample}"
            )

        adaptation_patch: bytes = b""
        staged_before = _g(["diff", "--cached", "--name-only"], repo_dir).stdout.strip()
        if staged_before:
            diff_r = subprocess.run(
                ["git", "diff", "--cached", "--binary"],
                cwd=repo_dir, capture_output=True,
            )
            if diff_r.returncode != 0:
                return (
                    f"⚠️ PR_MERGE_ERROR: Failed to capture staged adaptation patch "
                    f"(git diff --cached --binary returned {diff_r.returncode}). "
                    f"Staged changes are preserved. Fix the repo state and retry."
                )
            adaptation_patch = diff_r.stdout
            staged_before.splitlines()
            _g(["reset", "--hard", "HEAD"], repo_dir)

        def _restore_on_error() -> None:
            _g(["checkout", branch], repo_dir)
            if adaptation_patch:
                subprocess.run(
                    ["git", "apply", "--index", "-"],
                    cwd=repo_dir, input=adaptation_patch, capture_output=True,
                )

        co = _g(["checkout", target_branch], repo_dir)
        if co.returncode != 0:
            if adaptation_patch:
                subprocess.run(
                    ["git", "apply", "--index", "-"],
                    cwd=repo_dir, input=adaptation_patch, capture_output=True,
                )
            return (
                f"⚠️ PR_MERGE_ERROR: Cannot checkout '{target_branch}': "
                f"{_sanitize_git_error((co.stderr or '').strip())}"
            )

        if _g(["diff", "--name-only"], repo_dir).stdout.strip():
            _restore_on_error()
            return (
                f"⚠️ PR_MERGE_ERROR: Working tree on '{target_branch}' has unstaged "
                f"tracked changes.\nCommit or restore to HEAD before merging."
            )

        merge = _g(["merge", "--no-ff", "--no-commit", branch], repo_dir)
        if merge.returncode != 0:
            err = (merge.stderr or merge.stdout or "").strip()
            _g(["reset", "--hard", "HEAD"], repo_dir)
            _restore_on_error()
            return (
                f"⚠️ PR_MERGE_ERROR: Merge failed (restored to integration branch):\n"
                f"{_sanitize_git_error(err[:500])}"
            )

        merge_head = _g(["rev-parse", "-q", "--verify", "MERGE_HEAD"], repo_dir)
        if merge_head.returncode != 0:
            _restore_on_error()
            return (
                f"⚠️ PR_MERGE_ERROR: Branch '{branch}' is already fully merged into "
                f"'{target_branch}' (nothing to merge / already up to date)."
            )

        if adaptation_patch:
            apply_r = subprocess.run(
                ["git", "apply", "--index", "-"],
                cwd=repo_dir, input=adaptation_patch, capture_output=True,
            )
            if apply_r.returncode != 0:
                apply_err = (apply_r.stderr or apply_r.stdout or b"").decode(
                    "utf-8", errors="replace"
                ).strip()
                _g(["reset", "--hard", "HEAD"], repo_dir)
                _restore_on_error()
                return (
                    f"⚠️ PR_MERGE_ERROR: Merge staged but adaptation re-apply failed "
                    f"(patch conflicts with merged tree). Restored to integration branch.\n"
                    f"Resolve manually: edit files, stage with git add, then retry.\n"
                    f"{_sanitize_git_error(apply_err[:300])}"
                )
    finally:
        _release_git_lock(lock)

    base_r = _g(["merge-base", target_branch, branch], repo_dir)
    authors = []
    if base_r.returncode == 0:
        log_r = _g(["log", "--format=%an <%ae>",
                    f"{base_r.stdout.strip()}..{branch}"], repo_dir)
        if log_r.returncode == 0:
            authors = sorted(set(log_r.stdout.strip().splitlines()))

    co_authored = "\n".join(f"Co-authored-by: {a}" for a in authors) if authors else ""
    author_hint = (
        f"\n\nAttribution — include in commit_reviewed message:\n{co_authored}"
        if co_authored else ""
    )

    return (
        f"✅ Staged merge '{branch}' → '{target_branch}' (NOT committed)\n"
        f"  MERGE_HEAD is set — commit_reviewed will create a proper merge commit\n"
        f"  with both parents, preserving integration branch history.\n"
        f"  Branch '{branch}' left intact.\n\n"
        f"Next:\n"
        f"  commit_reviewed(commit_message='...')"
        + author_hint
    )


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("fetch_pr_ref", {
            "name": "fetch_pr_ref",
            "description": (
                "Fetch a GitHub PR's commits locally via pull/{n}/head ref. "
                "Uses force-update prefix so rebased/force-pushed PRs refetch correctly. "
                "Preserves original author metadata exactly as pushed. "
                "After fetching, list commit SHAs for cherry_pick_pr_commits."
            ),
            "parameters": {"type": "object", "properties": {
                "pr_number": {"type": "integer", "description": "GitHub PR number"},
                "remote": {"type": "string", "default": "origin",
                           "description": "Git remote name (default: origin)"},
            }, "required": ["pr_number"]},
        }, _fetch_pr_ref, is_code_tool=True, mutates_worktree=True),

        ToolEntry("create_integration_branch", {
            "name": "create_integration_branch",
            "description": (
                "Create a fresh integration branch (integrate/pr-N) from the body's working branch. "
                "External cherry-picked commits and Ouroboros adaptation changes are "
                "kept separate here before merging."
            ),
            "parameters": {"type": "object", "properties": {
                "pr_number": {"type": "integer", "description": "GitHub PR number"},
                "base_branch": {"type": "string", "description": (
                    "Branch to create from (default: ouroboros, or this task's candidate branch)")},
            }, "required": ["pr_number"]},
        }, _create_integration_branch, is_code_tool=True, mutates_worktree=True),

        ToolEntry("cherry_pick_pr_commits", {
            "name": "cherry_pick_pr_commits",
            "description": (
                "Replay PR commits onto the current integration branch using "
                "git cherry-pick --no-edit. By default each commit is created with "
                "the original author name/email/date preserved — GitHub attribution "
                "is real, not just a Co-authored-by annotation. Committer identity "
                "is set explicitly from the repo-local git config (with Ouroboros "
                "fallback when local identity is missing) for deterministic attribution. "
                "Optional override_author={'name': 'X', 'email': 'Y'} rewrites "
                "author name+email on every cherry-picked commit via git commit "
                "--amend --author --date while preserving the original author DATE "
                "and the repo-local committer (with Ouroboros fallback). Use when external contributor's "
                "commits use placeholder identity (e.g. ran Ouroboros locally "
                "without configuring git user.email). The override applies to "
                "the entire batch uniformly. "
                "Must be on an integrate/pr-N branch. "
                "When stop_on_conflict=False, skipped SHAs are explicitly reported."
            ),
            "parameters": {"type": "object", "properties": {
                "shas": {"type": "array", "items": {"type": "string"},
                         "description": "Ordered list of commit SHAs (oldest first)"},
                "stop_on_conflict": {"type": "boolean", "default": True,
                                     "description": "Abort on first conflict (default: true)"},
                "override_author": {
                    "type": "object",
                    "description": (
                        "Optional: rewrite author name+email on all cherry-picked "
                        "commits. Original author date and repo-local committer "
                        "identity with Ouroboros fallback when local identity is "
                        "missing are preserved. Applied to the entire batch uniformly."
                    ),
                    "properties": {
                        "name": {"type": "string",
                                 "description": "Author display name (no newlines, '<', or '>')"},
                        "email": {"type": "string",
                                  "description": "Author email (must contain '@', no newlines or angle brackets)"},
                    },
                    "required": ["name", "email"],
                    "additionalProperties": False,
                },
            }, "required": ["shas"]},
        }, _cherry_pick_pr_commits, is_code_tool=True, mutates_worktree=True),

        ToolEntry("stage_adaptations", {
            "name": "stage_adaptations",
            "description": (
                "Stage all current working-tree changes on the integration branch WITHOUT "
                "committing (git add -A only). Use after cherry_pick_pr_commits to prepare "
                "Ouroboros adaptation/fixup changes. Finalize via commit_reviewed "
                "to comply with BIBLE.md P3 (all commits must pass review). "
                "Must be on an integrate/pr-N branch."
            ),
            "parameters": {"type": "object", "properties": {}},
        }, _stage_adaptations, is_code_tool=True, mutates_worktree=True),

        ToolEntry("stage_pr_merge", {
            "name": "stage_pr_merge",
            "description": (
                "Stage a no-fast-forward merge of an integration branch into the body's working branch "
                "WITHOUT committing (git merge --no-ff --no-commit). Sets MERGE_HEAD so "
                "commit_reviewed creates a proper merge commit with both parents. Target is "
                "branch_dev (the bound candidate branch during self-development; commit_reviewed checks it out before "
                "committing — any other target would lose MERGE_HEAD). The "
                "integration-branch history (with original author commits) is permanently "
                "linked. Finalize via commit_reviewed."
            ),
            "parameters": {"type": "object", "properties": {
                "branch": {"type": "string",
                           "description": "Integration branch to merge (e.g. integrate/pr-17)"},
            }, "required": ["branch"]},
        }, _stage_pr_merge, is_code_tool=True, mutates_worktree=True),
    ]
