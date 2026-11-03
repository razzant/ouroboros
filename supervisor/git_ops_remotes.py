"""Personal persistence remote (`origin`) configuration and push, split out of
``supervisor/git_ops.py`` (module-size discipline, v7 G1 split).

Owns the personal-remote surface: pointing `origin` at the owner's repository,
storing the token in the repo-local credential helper, and pushing the current
branch and tags. The parent keeps the rebindable module state (``init`` REBINDS
REPO_DIR/BRANCH_DEV and friends), the capture plumbing and the managed-remote
probes, and re-exports every name here, so ``supervisor.git_ops`` stays the one
public surface. Parent members and rebindable globals are read through the
call-time handle ``_go()`` — never a from-import, which would freeze the
binding this module saw at import time.
"""

from __future__ import annotations

import logging
import subprocess
from typing import List, Optional, Tuple


def _go():
    """The parent module, read at call time.

    ``supervisor.git_ops`` owns the rebindable module state (``init`` REBINDS
    REPO_DIR and BRANCH_DEV) and the helpers tests monkeypatch on the parent
    (``git_capture``, ``_has_remote``, the sibling re-exports). Reading them
    through the module keeps one binding: a from-import here would freeze the
    value this module saw at import time.
    """
    from supervisor import git_ops

    return git_ops


# The parent's logger name is pinned so moved log records keep their `%(name)s`
# in server.log/stdout — the same logger object the parent binds.
log = logging.getLogger("supervisor.git_ops")


def configure_remote(repo_slug: str, token: str) -> Tuple[bool, str]:
    """Configure origin while storing the token in git credential helper."""
    if not repo_slug or not token:
        return False, "Missing repo slug or token"

    clean_url = f"https://github.com/{repo_slug}.git"

    if _go()._has_remote("origin"):
        rc, _, err = _go().git_capture(["git", "remote", "set-url", "origin", clean_url])
    else:
        rc, _, err = _go().git_capture(["git", "remote", "add", "origin", clean_url])
    if rc != 0:
        return False, f"Failed to configure remote: {err}"

    _go()._configure_credential_helper(repo_slug, token)
    return True, "ok"


def configure_personal_remote(
    repo_slug: str,
    token: str,
    *,
    auto_fork: bool = True,
    confirm_replace_origin: bool = False,
) -> Tuple[bool, str, str]:
    """Configure the personal persistence remote (`origin`), ensuring `managed` exists."""
    if not token:
        return False, "Missing GitHub token", ""
    # Ensure the official update path lives on `managed` BEFORE (re)pointing
    # `origin` at the personal repo, so replacing a clone-default `origin` that
    # still points at the official upstream never orphans the official update
    # remote. Shared by every caller (startup + Settings save). Best-effort:
    # personal-origin configuration proceeds even if this step fails.
    try:
        _go().ensure_official_update_remote()
    except Exception:
        log.warning("Official update remote setup failed during personal remote config", exc_info=True)
    resolved_slug = str(repo_slug or "").strip()
    warnings: List[str] = []
    # Always validate a configured slug (rejects the official repo and origin
    # conflicts); only empty-slug fork resolution is gated on auto_fork.
    if resolved_slug or auto_fork:
        try:
            from ouroboros.repo_remotes import ensure_personal_origin_target

            result = ensure_personal_origin_target(
                _go().REPO_DIR,
                token,
                configured_repo=resolved_slug,
                confirm_replace_origin=confirm_replace_origin,
            )
        except Exception as exc:
            return False, f"Personal remote provisioning failed: {exc}", ""
        if not result.ok:
            return False, result.message or result.action or "personal remote provisioning failed", ""
        resolved_slug = result.repo_slug
        warnings = list(result.warnings or [])
    if not resolved_slug:
        return False, "Missing repo slug", ""
    ok, msg = _go().configure_remote(resolved_slug, token)
    if not ok:
        return ok, msg, resolved_slug
    if warnings:
        msg = msg + " (" + "; ".join(warnings[:5]) + ")"
    return True, msg, resolved_slug


def _configure_credential_helper(repo_slug: str, token: str) -> None:
    """Store credentials in repo-local .git/credentials, not global state."""
    cred_path = _go().REPO_DIR / ".git" / "credentials"
    _go().git_capture([
        "git", "config", "--local", "credential.helper",
        f"store --file={cred_path}",
    ])
    cred_line = f"https://x-access-token:{token}@github.com"
    try:
        cred_path.write_text(cred_line + "\n", encoding="utf-8")
        cred_path.chmod(0o600)
    except Exception as e:
        log.warning("Failed to write repo credentials file: %s", e)


def push_to_remote(branch: Optional[str] = None, push_tags: bool = True, *,
                   created_tag: str = "", expected_commit_sha: str = "") -> Tuple[bool, str]:
    """Push the branch and at most its explicitly operation-created annotated tag.

    Missing creation evidence means branch-only, even with push_tags=True. The
    compatibility boolean reports branch success, not complete tag publication;
    the message names partial outcomes. Both pushes are bounded, never forced or
    automatically retried. Existing tags are neither discovered nor backfilled.
    """
    if not _go()._has_remote("origin"):
        return False, "No remote configured"

    target = branch or _go().BRANCH_DEV
    branch_ref = target if target.startswith("refs/heads/") else f"refs/heads/{target}"
    rc, out, err = _go()._git_network_bounded(
        ["push", "--no-follow-tags", "-u", "origin", f"{branch_ref}:{branch_ref}"])
    if rc != 0:
        log.warning("Personal-origin branch push failed or unconfirmed: %s", err)
        return False, "Branch publication failed or is unconfirmed; see Logs. No automatic retry"

    result = f"Pushed {target} to origin"
    if push_tags and created_tag:
        tag_ref = f"refs/tags/{created_tag}"
        try:
            # Validate in the same repository the network runner publishes.
            rc_t, out_t, err_t = _go().git_capture(
                ["git", "-C", str(_go().REPO_DIR), "for-each-ref",
                 "--format=%(refname) %(objecttype) %(objectname) %(*objecttype) %(*objectname)", tag_ref],
                repair_index=False)
            tag = next((line.split() for line in out_t.splitlines()
                        if line.split()[:1] == [tag_ref]), [])
            if (rc_t != 0 or len(tag) != 5 or tag[1] != "tag"
                    or tag[3:] != ["commit", expected_commit_sha]):
                raise ValueError(err_t or "missing or mismatched annotated operation tag")
            rc_t, _, err_t = _go().git_capture(
                ["git", "-C", str(_go().REPO_DIR), "merge-base", "--is-ancestor",
                 expected_commit_sha, branch_ref], repair_index=False)
            if rc_t != 0:
                raise ValueError(err_t or "release commit is not on the selected branch")
            # Pin the validated object: later ref movement cannot substitute it.
            rc_t, _, err_t = _go()._git_network_bounded(
                ["push", "--no-follow-tags", "origin", f"{tag[2]}:{tag_ref}"])
            if rc_t != 0:
                raise RuntimeError(err_t or "git tag push failed")
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            log.warning("Personal-origin tag %s failed or unconfirmed: %s", created_tag, exc)
            result += (f" (partial publication: tag {created_tag} failed or unconfirmed; "
                       "see Logs; no automatic retry)")
        else:
            result += f" + tag {created_tag}"
    return True, result
