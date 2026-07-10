"""Project working-folder sources (v6.59.0, Phase 3): attach an existing folder or
clone a git URL as a project's working_dir.

Both entry points return the ATTACHED/CLONED path plus a typed error, never raise,
and stamp NO registry state themselves — the gateway/tool caller registers the
project and records provenance (attached | cloned | genesis | none) + `clone_url`
as HISTORICAL facts. Operational git data (branch, remotes, dirtiness) is always
read from the live ``.git``, never cached in the registry.

Attach doctrine (quiz 13 "notification" model): attaching is the OWNER'S explicit
act in the UI/tool, so `trusted_at` is stamped automatically and the dialog carries
the honest "the agent gets write+shell in this folder" text — no second
confirmation gate. `init_git` is OPT-IN ONLY: an attach NEVER auto-runs `git init`
on the owner's folder without the flag (the folder belongs to the owner; mutating
it is a decision, not a default).

Clone doctrine: server-side, atomic (clone into a ``.tmp.<pid>`` sibling, rename
into place on success), never interactive (``GIT_TERMINAL_PROMPT=0`` + null
askpass), with a TYPED ``auth_required`` classification so the UI can say
"this repo needs credentials" instead of dumping a git stderr blob.
"""
from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
from typing import Any, Optional

from ouroboros.platform_layer import bootstrap_process_path

# https://host/path(.git) | ssh://user@host/path | user@host:path(.git)
_HTTPS_URL_RE = re.compile(r"^https?://[\w.\-]+(:\d+)?/\S+$")
_SSH_URL_RE = re.compile(r"^ssh://[\w.\-@]+(:\d+)?/\S+$")
_SCP_LIKE_RE = re.compile(r"^[\w.\-]+@[\w.\-]+:\S+$")

_AUTH_MARKERS = (
    "authentication failed",
    "could not read username",
    "could not read password",
    "permission denied (publickey",
    "terminal prompts disabled",
    "invalid username or password",
    "authentication required",
    "access denied",
)

CLONE_TIMEOUT_SEC = 900


def valid_git_url(url: str) -> bool:
    text = str(url or "").strip()
    return bool(
        _HTTPS_URL_RE.match(text) or _SSH_URL_RE.match(text) or _SCP_LIKE_RE.match(text)
    )


def derive_repo_dir_name(url: str) -> str:
    """Directory name from a git URL's last path segment (sans .git)."""
    tail = str(url or "").rstrip("/").rsplit("/", 1)[-1].rsplit(":", 1)[-1]
    if tail.endswith(".git"):
        tail = tail[: -len(".git")]
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]", "-", tail).strip("-.")
    return cleaned or "cloned-project"


def validate_attach_path(
    raw_path: Any, *, system_repo_dir: Any, drive_root: Any
) -> tuple[Optional[pathlib.Path], str]:
    """Validate an owner folder for attach. Checks run on the RESOLVED realpath
    (symlinks followed) so a symlink cannot smuggle the home root or repo/data in:
    must exist, be a directory, not be the home root itself, and not overlap the
    Ouroboros system repo or data drive. Being a git repo is NOT required at attach
    time (``init_git`` is the opt-in; task admission separately requires a git
    worktree root and loud-fails otherwise). Returns (resolved, error)."""
    text = str(raw_path or "").strip()
    if not text:
        return None, "path is required"
    try:
        resolved = pathlib.Path(text).expanduser().resolve(strict=True)
    except FileNotFoundError:
        return None, f"path does not exist: {text}"
    except (OSError, ValueError) as exc:
        return None, f"path is not usable: {type(exc).__name__}: {exc}"
    if not resolved.is_dir():
        return None, f"path is not a directory: {text}"
    home = pathlib.Path.home().resolve(strict=False)
    if resolved == home:
        return None, "refusing to attach the home directory itself; pick a project folder"
    from ouroboros.tool_access import path_is_relative_to

    for protected, label in (
        (pathlib.Path(system_repo_dir).resolve(strict=False), "Ouroboros system repo"),
        (pathlib.Path(drive_root).resolve(strict=False), "Ouroboros data drive"),
    ):
        if resolved == protected or path_is_relative_to(resolved, protected) or path_is_relative_to(protected, resolved):
            return None, f"path must not overlap the {label}"
    return resolved, ""


def is_git_worktree_root(path: pathlib.Path) -> bool:
    """True when ``path`` IS a git worktree root (the same fact task admission's
    validate_workspace_root later requires — checked at attach time so a non-git
    attach cannot register a project whose room tasks are born dead, triad r5)."""
    bootstrap_process_path()
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(path), capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return False
    top = (res.stdout or "").strip() if res.returncode == 0 else ""
    if not top:
        return False
    try:
        return pathlib.Path(top).resolve(strict=False) == pathlib.Path(path).resolve(strict=False)
    except OSError:
        return False


def _unstage_sensitive_paths(path: pathlib.Path) -> list[str]:
    """Unstage credential-shaped files after ``git add -A`` and keep them untracked
    via `.git/info/exclude` (local-only — the owner's folder files are never edited).
    Same `_sensitive_untracked_reason` SSOT the workspace patch and coop checkpoint
    use (triad r4: an attach snapshot must not bake `.env`/keys into history).
    Returns the skipped relative paths for disclosure."""
    from ouroboros.headless import _sensitive_untracked_reason

    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "-z"],
        cwd=str(path), capture_output=True, text=True, timeout=60,
    )
    skipped = [
        rel for rel in (staged.stdout or "").split("\0")
        if rel and _sensitive_untracked_reason(rel)
    ]
    if not skipped:
        return []
    subprocess.run(
        ["git", "rm", "-q", "--cached", "--"] + skipped,
        cwd=str(path), capture_output=True, text=True, timeout=60,
    )
    exclude = path / ".git" / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    with exclude.open("a", encoding="utf-8") as fh:
        fh.write("\n# ouroboros attach-snapshot: credential-shaped files stay untracked\n")
        for rel in skipped:
            fh.write(f"/{rel}\n")
    return skipped


def attach_snapshot_init(path: pathlib.Path) -> tuple[str, list[str]]:
    """OPT-IN ``init_git``: initialize git in an attached non-git folder and commit an
    attach-snapshot of the CURRENT state with a local identity (no global config
    touched). Credential-shaped files are EXCLUDED from the snapshot (disclosed via
    the returned list) — secrets must never be baked into git history (BIBLE
    prohibition; triad r4). Idempotent for an existing repo. Returns
    ``(error, skipped_sensitive)``: error "" on success."""
    bootstrap_process_path()
    try:
        if (path / ".git").exists():
            return "", []
        init = subprocess.run(["git", "init", "-q"], cwd=str(path), capture_output=True, text=True, timeout=30)
        if init.returncode != 0:
            return (init.stderr or init.stdout or "git init failed").strip()[:300], []
        add = subprocess.run(["git", "add", "-A"], cwd=str(path), capture_output=True, text=True, timeout=120)
        if add.returncode != 0:
            return (add.stderr or add.stdout or "git add failed").strip()[:300], []
        skipped = _unstage_sensitive_paths(path)
        commit = subprocess.run(
            [
                "git", "-c", "user.name=Ouroboros", "-c", "user.email=ouroboros@local",
                "commit", "-q", "--allow-empty", "-m", "ouroboros: attach snapshot",
            ],
            cwd=str(path), capture_output=True, text=True, timeout=120,
        )
        if commit.returncode != 0:
            return (commit.stderr or commit.stdout or "git commit failed").strip()[:300], skipped
        return "", skipped
    except Exception as exc:  # noqa: BLE001 — attach must fail typed, not raise
        return f"{type(exc).__name__}: {exc}", []


def clone_project_repo(git_url: str, dest_name: str = "") -> tuple[str, str, str]:
    """Clone ``git_url`` into the durable projects root. Returns
    ``(path, error_code, error_detail)`` — error_code is "" on success,
    ``invalid_url`` / ``exists`` / ``auth_required`` / ``clone_failed`` otherwise.

    Atomicity: clones into ``<dest>.tmp.<pid>`` then renames into place, so an
    interrupted clone never leaves a half-usable project folder. Non-interactive:
    ``GIT_TERMINAL_PROMPT=0`` + null askpass — a private repo fails FAST with the
    typed ``auth_required`` instead of hanging on a hidden prompt."""
    url = str(git_url or "").strip()
    if not valid_git_url(url):
        return "", "invalid_url", "git_url must be an https://, ssh:// or user@host:path git URL"
    from ouroboros.config import get_subagent_projects_root

    projects_root = pathlib.Path(get_subagent_projects_root()).expanduser()
    projects_root.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^a-zA-Z0-9_.-]", "-", str(dest_name or "").strip()).strip("-.") or derive_repo_dir_name(url)
    dest = projects_root / name
    if dest.exists():
        return "", "exists", f"destination already exists: {dest}"
    tmp = projects_root / f"{name}.tmp.{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    bootstrap_process_path()
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_ASKPASS"] = ""  # no GUI credential prompt; with TERMINAL_PROMPT=0 → fail fast
    env.setdefault("GIT_SSH_COMMAND", "ssh -oBatchMode=yes")
    try:
        proc = subprocess.run(
            ["git", "clone", "--", url, str(tmp)],
            capture_output=True, text=True, timeout=CLONE_TIMEOUT_SEC, env=env,
        )
    except subprocess.TimeoutExpired:
        shutil.rmtree(tmp, ignore_errors=True)
        return "", "clone_failed", f"clone timed out after {CLONE_TIMEOUT_SEC}s"
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(tmp, ignore_errors=True)
        return "", "clone_failed", f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        shutil.rmtree(tmp, ignore_errors=True)
        lowered = detail.lower()
        if any(marker in lowered for marker in _AUTH_MARKERS):
            return "", "auth_required", detail[:600]
        return "", "clone_failed", detail[:600] or "git clone failed"
    try:
        tmp.rename(dest)
    except OSError as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        return "", "clone_failed", f"rename into place failed: {exc}"
    return str(dest), "", ""
