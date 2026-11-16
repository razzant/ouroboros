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

Durable-folder doctrine: a Project's ``working_dir`` outlives every task of its room,
so it never names a copy whose life the host manages (``temporary_copy_refusal``).
A task may still run in such a copy; the registry refuses only the lasting binding,
on every write of a new folder (``require_durable_folder``).

No-folder doctrine: ``no_folder`` ({at, by}) is the recorded choice that a Project keeps
no folder, apart from ``provenance`` (how a folder once came). It stands until a folder
is chosen for the Project itself; a folder a task merely runs in never replaces it
(``apply_folder``). A row without it and without a folder made no choice: its first
promoted task still provisions one.
"""
from __future__ import annotations

import logging
import os
import pathlib
import re
import shutil
import subprocess
from typing import Any, Optional

from ouroboros.platform_layer import bootstrap_process_path

log = logging.getLogger(__name__)

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


class TemporaryCopyRefused(ValueError):
    """A host-managed temporary copy offered as a Project's lasting folder."""


def _one_folder(left: pathlib.Path, right: pathlib.Path) -> bool:
    """Whether two paths name one existing folder, for a refusal's wording only (an error reads as two)."""
    try:
        return os.path.samefile(left, right)
    except OSError:
        return False


def temporary_copy_refusal(path: Any, *, drive_root: Any) -> str:
    """Why ``path`` cannot be a Project's lasting ``working_dir``, else "" (empty names no folder).

    The host manages the life of everything inside its working-copies root
    (``config.get_subagent_worktree_root``: acting checkouts, delegated snapshots, own-body
    candidates and their scratch) and of every copy a private-copy record names wherever it
    lives (an earlier root, a managed deployment's staging copies), its subfolders included.
    Identity is physical, never case-folded: the folder and the root compare as their volume keeps
    them (``workspace_admission.kept_path``), and each spelling of the folder and its ancestors is
    looked up by its own record file; when all miss, one passive read proves a record another spelling wrote
    (``private_copy_read.records_naming``). An unreadable record of one of these spellings, or an
    unlistable records folder, refuses as unknown, never as a copy; an unreadable file that merely
    shares a folder's name ties nothing to it and is logged. These are the host's own root and
    records, never a folder's name. A folder that merely contains that root or a copy is not a copy.
    Clones and genesis projects (the durable projects root) and owner folders pass."""
    text = str(path or "").strip()
    if not text:
        return ""
    from ouroboros.config import get_subagent_worktree_root
    from ouroboros.launch_roots import WorkRootUnavailable
    from ouroboros.private_copy_read import checkout_registration, records_naming
    from ouroboros.workspace_admission import kept_path

    folder = pathlib.Path(text).expanduser().resolve(strict=False)
    kept = kept_path(folder)
    try:
        copies: Optional[pathlib.Path] = pathlib.Path(get_subagent_worktree_root()).expanduser().resolve(strict=False)
    except WorkRootUnavailable:
        copies = None  # an undecided default root holds no copy; the records still answer
    if copies is not None and kept.is_relative_to(kept_path(copies)):  # the root itself or inside it
        return f"{folder} is inside the host's working-copies root {copies}, whose copies are removed when their work ends"
    # One record file per spelling of each folder up the tree, the given spelling first, never the filesystem root.
    spellings = list(dict.fromkeys((folder, *folder.parents[:-1], kept, *kept.parents[:-1])))
    found = {copy: checkout_registration(copy, drive_root) for copy in spellings}
    copy = next((copy for copy, registration in found.items() if registration == "registered"), None)
    read = records_naming(spellings, drive_root) if copy is None else None
    if read is not None and read.rows:
        copy = pathlib.Path(str(read.rows[0]["path"]))
    if copy is not None:
        if copy in (folder, kept) or (read is not None and _one_folder(copy, folder)):
            what = "a host-managed private copy" + ("" if copy == folder else f" (recorded as {copy})")
        else:
            what = f"inside the host-managed private copy {copy}"
        return f"{folder} is {what}, removed when its work ends"
    unread = next((copy for copy, registration in found.items() if registration == "unknown"), None)
    if unread is not None or read.unlistable:
        cause = f"the private-copy record of {unread}" if unread is not None else "the private-copy records folder"
        return f"whether {folder} is a host-managed temporary copy is unknown: {cause} cannot be read"
    gaps = ([f"unreadable records sharing a folder's name, tied to none: {', '.join(sorted(read.unreadable))}"]
            if read.unreadable else []) + (["the legacy registry awaits its import"] if read.import_pending else [])
    if gaps:  # passing, never as proof the folder has no record
        log.warning("%s passes as a lasting Project folder on an incomplete read of the copy records: %s",
                    folder, "; ".join(gaps))
    return ""


def require_durable_folder(path: Any, *, drive_root: Any) -> None:
    """The registry's one check before it writes a new ``working_dir``: raise
    ``TemporaryCopyRefused`` with ``temporary_copy_refusal``'s reason."""
    reason = temporary_copy_refusal(path, drive_root=drive_root)
    if reason:
        raise TemporaryCopyRefused(reason)


class NoFolderChosen(ValueError):
    """A folder offered to a Project whose recorded choice is to keep none, by a writer that is not choosing its folder."""


def apply_folder(entry: dict, key: str, value: Any, *, chosen: bool, drive_root: Any) -> None:
    """The registry's one transition of a room's default resource, on its row and before the save.

    ``working_dir``: a new address is never a host-managed temporary copy and ends ``no_folder``. Only a
    writer choosing the Project's folder may end it (``chosen``: its update names ``no_folder`` too);
    any other raises ``NoFolderChosen`` and writes nothing. ``no_folder``: ``{at, by}`` records the
    choice to keep none and empties the folder, a repeat keeping the first record; ``None`` withdraws it.
    Each change starts a new routing generation: preparation that read the old default is refused."""
    before = (str(entry.get("working_dir") or ""), entry.get("no_folder"))
    if key == "no_folder":
        if value is None:
            entry.pop("no_folder", None)
        elif (not isinstance(value, dict) or set(value) != {"at", "by"}
              or any(not isinstance(item, str) or not item.strip() for item in value.values())):
            raise ValueError("no_folder must be None or {at, by} of nonempty strings")
        elif "no_folder" not in entry:
            entry.update(no_folder=dict(value), working_dir="")
    elif value != entry.get("working_dir"):
        if value and "no_folder" in entry and not chosen:
            raise NoFolderChosen("the Project keeps no folder by a recorded choice; choose its folder to change that")
        require_durable_folder(value, drive_root=drive_root)
        entry["working_dir"] = value
        if value:
            entry.pop("no_folder", None)
    if (str(entry.get("working_dir") or ""), entry.get("no_folder")) != before:
        entry["routing_generation"] = int(entry.get("routing_generation") or 0) + 1


def validate_attach_path(
    raw_path: Any, *, system_repo_dir: Any, drive_root: Any
) -> tuple[Optional[pathlib.Path], str]:
    """Validate an owner folder for attach. Checks run on the RESOLVED realpath
    (symlinks followed) so a symlink cannot smuggle the home root or repo/data in:
    must exist, be a directory, not be the home root itself, not overlap the
    Ouroboros system repo or data drive, and not be a host-managed temporary copy
    (refused here, before ``init_git`` or a registry row). Ordinary directories need no Git;
    ``init_git`` remains an explicit optional operation. Returns (resolved, error)."""
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
    from ouroboros.workspace_admission import protected_overlap

    overlap = protected_overlap(resolved, system_repo_dir=system_repo_dir, drive_root=drive_root)
    if overlap:  # the rule admission applies after init_git, here before it writes anything
        return None, f"path must not overlap the {overlap}"
    temporary = temporary_copy_refusal(resolved, drive_root=drive_root)
    if temporary:
        return None, f"{temporary}; a Project folder outlives its tasks: attach your own folder or clone the repository"
    return resolved, ""


def _unstage_sensitive_paths(path: pathlib.Path, *, warnings=None) -> list[str]:
    """Unstage credential files after ``git add -A`` and keep them untracked
    via `.git/info/exclude` (local-only — the owner's folder files are never edited).
    Same two checks the workspace patch and the coop checkpoint apply: the exact
    credential leaves of `_sensitive_untracked_reason` and the private-key content
    evidence of `pem_private_key_reason`. Effective Cyber keeps PEM findings in
    ``warnings`` without unstaging those bytes. Returns genuinely skipped paths."""
    from ouroboros.headless import _sensitive_untracked_reason
    from ouroboros.workspace_patch_capture import pem_capture_refusal

    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "-z"],
        cwd=str(path), capture_output=True, text=True, timeout=60,
        encoding="utf-8", errors="surrogateescape",  # Git's UTF-8 path bytes, whatever the locale's code page
    )
    skipped = [
        rel for rel in (staged.stdout or "").split("\0")
        if rel and (_sensitive_untracked_reason(rel) or pem_capture_refusal(path, rel, warnings=warnings))
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


def attach_snapshot_init(path: pathlib.Path, *, warnings=None) -> tuple[str, list[str]]:
    """OPT-IN ``init_git``: initialize git in an attached non-git folder and commit an
    attach-snapshot of the CURRENT state with a local identity (no global config
    touched). The existing credential policy governs exclusion; PEM findings
    remain advisory in effective Cyber mode and are returned through ``warnings``.
    Idempotent for an existing repo. Returns
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
        skipped = _unstage_sensitive_paths(path, warnings=warnings)
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
    ``invalid_url`` / ``exists`` / ``auth_required`` / ``clone_failed`` /
    ``work_root_unavailable`` otherwise.

    Atomicity: clones into ``<dest>.tmp.<pid>`` then renames into place, so an
    interrupted clone never leaves a half-usable project folder. Non-interactive:
    ``GIT_TERMINAL_PROMPT=0`` + null askpass — a private repo fails FAST with the
    typed ``auth_required`` instead of hanging on a hidden prompt."""
    url = str(git_url or "").strip()
    if not valid_git_url(url):
        return "", "invalid_url", "git_url must be an https://, ssh:// or user@host:path git URL"
    from ouroboros.config import get_subagent_projects_root
    from ouroboros.launch_roots import WorkRootUnavailable

    try:
        projects_root = pathlib.Path(get_subagent_projects_root()).expanduser()
    except WorkRootUnavailable as exc:
        return "", "work_root_unavailable", str(exc)
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
