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
it is a decision, not a default). Attach does NOT require the folder to be a git
worktree either (A11/A12): a plain folder is a legitimate PLACE for a project, and
the git question is asked separately — as the typed `git_init_required` offer
`workspace_admission` raises before the first FILE task — with `attach_snapshot_init`
as the one thing the owner's "yes" runs, whether it comes from the create dialog's
`init_git` or from `POST /api/projects/{id}/init-git` afterwards.

What replaced the git REQUIREMENT is a CONTAINMENT guard, and the two are not the
same rule. "Not a git repo" is fine; "a subdirectory of somebody else's git repo"
is not, because saying yes there would `git init` a second repository nested inside
the owner's, after which every diff, rollback and commit Ouroboros makes happens in
a shadow repo the owner's VCS cannot see. `enclosing_git_worktree` answers that one
question, so plain folders and worktree ROOTS both still attach.

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

# Environment that REDIRECTS git's idea of which repository it is standing in.
# Every containment/durability probe must run with these stripped: inherited from
# whatever launched Ouroboros they answer about a repository the owner never named
# — `GIT_DIR`/`GIT_WORK_TREE` make a PLAIN folder report someone else's toplevel
# (an A11/A12 violation: refusing a folder that encloses nothing), and
# `GIT_CEILING_DIRECTORIES` stops the upward search so a real repo SUBDIRECTORY
# reports nothing and is admitted. The clone path already scrubs its env; the
# probes now follow that precedent.
_GIT_LOCATION_ENV = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_COMMON_DIR",
    "GIT_CEILING_DIRECTORIES",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
)


def _git_probe(cwd: pathlib.Path, *args: str) -> str:
    """Run a read-only `git` probe in ``cwd`` and return trimmed stdout, or "".

    "" means NOTHING WAS LEARNED — never "the answer is no". Callers must already
    hold a filesystem-derived answer before calling; a probe may only WIDEN what is
    refused, because git fails in ways that have nothing to do with the question:
    `safe.directory` refuses a foreign-owned repo (rc=128), an older git refuses a
    repo with unknown `extensions.*` (sha256/reftable), and a hostile
    `[include] path=<fifo>` stalls until the timeout."""
    env = {k: v for k, v in os.environ.items() if k not in _GIT_LOCATION_ENV}
    try:
        res = subprocess.run(
            ["git", *args],
            cwd=str(cwd), capture_output=True, text=True, timeout=5, env=env,
        )
    except Exception:
        return ""
    return (res.stdout or "").strip() if res.returncode == 0 else ""


def _is_git_storage_dir(path: pathlib.Path) -> bool:
    """Does ``path`` look like git's own storage (a bare repo, or a ``.git`` dir)?

    Detected structurally — `HEAD` + `objects/` + `refs/` — rather than by asking
    git, because this is used to decide CONTAINMENT and containment must have an
    answer even when git will not give one."""
    try:
        return (
            (path / "HEAD").is_file()
            and (path / "objects").is_dir()
            and (path / "refs").is_dir()
        )
    except OSError:
        return False


def _linked_worktree_host(path: pathlib.Path) -> str:
    """The repository a LINKED worktree at ``path`` belongs to, or "".

    A linked worktree's ``.git`` is a FILE holding ``gitdir: <repo>/.git/worktrees/<name>``
    — read straight off the disk, so the answer survives every way git can refuse to
    speak. A submodule's ``.git`` file also exists but points at ``.git/modules/…``,
    which is a DURABLE checkout, not a removable view, so only the ``worktrees``
    component counts."""
    marker = path / ".git"
    try:
        if not marker.is_file():
            return ""
        text = marker.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    if not text.startswith("gitdir:"):
        return ""
    raw = text.split("gitdir:", 1)[1].strip().splitlines()[0].strip()
    if not raw:
        return ""
    try:
        target = pathlib.Path(raw)
        target = (target if target.is_absolute() else path / target).resolve(strict=False)
    except OSError:
        return ""
    parts = target.parts
    if "worktrees" not in parts:
        return ""
    git_dir = pathlib.Path(*parts[: parts.index("worktrees")])
    return str(git_dir.parent) if git_dir.name == ".git" else str(git_dir)


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
    Ouroboros system repo or data drive. Being a git repo is NOT required — not at
    attach time and not for the project to keep the folder; ``init_git`` is the
    opt-in, and task admission raises the typed ``git_init_required`` offer for an
    untracked folder rather than refusing it.

    What IS required is that the folder not sit INSIDE another git repository
    (``enclosing_git_worktree``, answered from the filesystem so a git that refuses
    to speak cannot turn UNKNOWN into ADMIT). That is containment, not a git
    requirement: a plain folder and a worktree root both pass, and only a
    subdirectory of somebody's repo is refused — by name, so the owner can attach
    the root the error points at, EXCEPT when that root is itself one of Ouroboros's
    removable checkouts, where naming it would recommend a place that gets deleted.
    Git's own storage (a bare repo, a ``.git`` directory) is refused too: it holds a
    repository, it is not a folder to work in. Returns (resolved, error)."""
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
    if _is_git_storage_dir(resolved):
        return None, (
            f"{resolved} is a git repository's internal storage (a bare repo or a .git "
            "directory), not a working folder — attach the checkout instead"
        )
    enclosing = enclosing_git_worktree(resolved)
    if enclosing:
        detail = (
            "A project folder nested in someone else's repository cannot be put under git "
            "of its own without hiding a second repository inside theirs"
        )
        # Only point at the enclosing root when it is somewhere a project could
        # actually live. If the thing containing this folder is one of Ouroboros's
        # own removable checkouts, "attach that root instead" recommends a place
        # that a `git worktree remove` or a retention sweep deletes — advice worse
        # than none.
        if ephemeral_checkout_reason(pathlib.Path(enclosing)):
            return None, (
                f"this folder is inside the git repository at {enclosing}, which is itself a "
                f"temporary checkout and cannot hold a project either. {detail}"
            )
        return None, (
            f"this folder is inside the git repository at {enclosing} — attach that root "
            f"instead. {detail}"
        )
    return resolved, ""


def enclosing_git_worktree(path: pathlib.Path) -> str:
    """The git repository root that CONTAINS ``path`` without BEING it, or "".

    Deliberately a CONTAINMENT question, not a git one. A plain folder answers ""
    (nothing encloses it) and so does a worktree ROOT (the repository is the folder
    itself), so both remain attachable under A11/A12. Only a SUBDIRECTORY of a
    repository answers with that repository's root — the one shape where making
    the folder a project's place is wrong in a way the owner cannot see later:
    ``git init`` there nests a second repository inside theirs, the nested folder
    then passes task admission as a worktree root, and every diff, rollback and
    commit afterwards lands in the shadow repo while the owner's real VCS reports
    only an untracked directory.

    The answer comes from the FILESYSTEM, not from git. Asking git made the guard
    fail OPEN: `git rev-parse` exits non-zero for reasons that say nothing about
    containment — `safe.directory` refuses a foreign-owned repo, an older git
    refuses unknown `extensions.*` (sha256/reftable), a hostile `[include]` stalls
    past the timeout — and every one of those was read as "nothing encloses this",
    admitting the exact shape the guard exists to refuse. Walking ``parents`` for a
    ``.git`` (directory OR file) or a bare-repo layout always has an answer, so
    UNKNOWN never means ADMIT. A submodule working directory is caught the same
    way, by the superproject's ``.git`` above it, where git's own `--show-toplevel`
    reports the submodule itself and admits.

    Never raises."""
    try:
        resolved = pathlib.Path(path).resolve(strict=False)
    except OSError:
        return ""
    for ancestor in resolved.parents:
        try:
            if (ancestor / ".git").exists() or _is_git_storage_dir(ancestor):
                return str(ancestor)
        except OSError:
            continue
    # Supplement, never a source of ADMIT: the walk has already answered, and this
    # can only ADD a refusal. Worth one probe for a checkout whose enclosing
    # repository is not on its parent chain at all (`core.worktree`, a submodule
    # relocated out of its superproject); skipped entirely for a folder with no
    # `.git` of its own, which no such shape can have.
    if not (resolved / ".git").exists():
        return ""
    bootstrap_process_path()
    superproject = _git_probe(resolved, "rev-parse", "--show-superproject-working-tree")
    if not superproject:
        return ""
    try:
        top = pathlib.Path(superproject).resolve(strict=False)
    except OSError:
        return ""
    return "" if top == resolved else str(top)


def ephemeral_checkout_reason(path: pathlib.Path) -> str:
    """Why ``path`` must not become a project's DURABLE place, or "".

    A project's folder outlives every task that ever runs in it, so the checkouts
    Ouroboros makes FOR ITSELF are disqualified even though each is a perfectly
    good workspace for the task holding it:

    - a LINKED git worktree — ``--git-common-dir`` differs from its own
      ``--git-dir`` — is a temporary view of another repository's history that one
      ``git worktree remove`` deletes, taking the project's place with it;
    - anything under the acting-subagent worktree root is a checkout of the
      Ouroboros body itself AND is age-swept by the orphan GC, so a project
      pointed at one would lose its folder on a retention pass;
    - anything under the thread worktree root is a branch-off checkout owned by a
      thread's lifecycle, not by a project.

    This is the DURABLE-place rule, which is why it lives beside the attach guards
    rather than inside them: attach paths are typed by the owner, but an adopted
    folder arrives from a task record, and a task's workspace is exactly where
    these checkouts show up. Never raises."""
    try:
        resolved = pathlib.Path(path).resolve(strict=False)
    except OSError:
        return ""
    from ouroboros.tool_access import path_is_relative_to

    roots: list[tuple[str, pathlib.Path, str]] = []
    try:
        from ouroboros.config import get_subagent_worktree_root

        roots.append((
            "acting-subagent worktree root",
            pathlib.Path(get_subagent_worktree_root()).expanduser().resolve(strict=False),
            "those checkouts are copies of Ouroboros itself and the orphan sweep deletes them by age",
        ))
    except Exception:
        pass
    try:
        from ouroboros.thread_worktrees import thread_worktree_root

        roots.append((
            "thread worktree root",
            thread_worktree_root(),
            "a thread's branch-off checkout belongs to that thread's lifecycle, not to a project",
        ))
    except Exception:
        pass
    for label, root, why in roots:
        if resolved == root or path_is_relative_to(resolved, root):
            return (
                f"{resolved} sits under the Ouroboros {label} ({root}) — {why}, so it cannot "
                "be a project's permanent folder"
            )

    def _linked(host: str) -> str:
        return (
            f"{resolved} is a linked git worktree of the repository at {host} — a "
            "worktree is a temporary checkout that can be removed at any time, so it cannot be "
            "a project's permanent folder; use the repository itself"
        )

    # Read off the DISK first. A linked worktree's `.git` is a file naming the host
    # repository's `.git/worktrees/<name>`; that fact is there whether or not git
    # will talk to us. The probe below used to be the only source, and its failure
    # mode was "" — i.e. DURABLE — so a foreign-owned, extension-newer or stalled
    # repository handed a project a place a `git worktree remove` can delete.
    host = _linked_worktree_host(resolved)
    if host:
        return _linked(host)

    # Supplement only, and only where a linked worktree could possibly BE: its
    # `.git` is always a file. A `.git` DIRECTORY is a main worktree (own and
    # common git-dir are the same by construction) and no `.git` at all means the
    # probe would describe some ANCESTOR repository, which is the containment
    # question, not this one. Skipping those two cases is also what keeps a
    # stalled or foreign-owned repository from costing 10 s of subprocess timeout
    # on the ordinary path.
    if not (resolved / ".git").is_file():
        return ""
    bootstrap_process_path()
    own_dir = _git_probe(resolved, "rev-parse", "--git-dir")
    common_dir = _git_probe(resolved, "rev-parse", "--git-common-dir")
    if not own_dir or not common_dir:
        # Nothing LEARNED, which is not the same as nothing WRONG — but the
        # filesystem has already given the durable-place answer above, so the
        # remaining probe can only have widened it.
        return ""
    try:
        own = pathlib.Path(own_dir)
        shared = pathlib.Path(common_dir)
        own = (own if own.is_absolute() else resolved / own).resolve(strict=False)
        shared = (shared if shared.is_absolute() else resolved / shared).resolve(strict=False)
    except OSError:
        return ""
    if own == shared:
        return ""
    return _linked(str(shared.parent))


def _staged_sensitive_partition(path: pathlib.Path) -> tuple[list[str], list[str], str]:
    """Split the STAGED credential-shaped paths by whether HEAD already has them.

    Returns ``(absent_from_head, present_in_head, error)``. Only the first list may
    ever be unstaged, and that distinction is the whole point of this function.

    ``attach_snapshot_init`` runs on a repository it just created, where every
    staged path is untracked BY CONSTRUCTION, so "unstage everything that looks
    like a credential" was correct there and only there. On an EXISTING repository
    — which is what branching off snapshots — ``git diff --cached --name-only``
    also lists TRACKED modifications, and ``git rm --cached`` on one of those does
    not "leave it out of the commit": it stages a DELETION. The file leaves
    ``git ls-files``, the owner's branch gets a commit removing it, and nothing in
    history is protected by the removal, because a tracked file's contents are
    already in history. Keeping a secret OUT of history and TAKING a tracked file
    out of the owner's branch are opposite acts that share one git command, so the
    two cases must be told apart before it runs.

    ``error`` is non-empty when membership in HEAD could not be determined. That is
    never treated as "absent": failing that direction is exactly the data loss this
    function exists to prevent, so the caller refuses the whole snapshot instead.
    An UNBORN HEAD is not an error — it is the definitive answer "nothing is
    tracked yet", which is the attach case.

    ``--no-renames`` is load-bearing, not tidiness. Rename detection is git's
    default, and it prints a staged rename as its DESTINATION alone: a tracked
    ``secrets.env`` renamed to ``secrets2.env`` arrives here as one path that HEAD
    has never heard of, so it is classified absent, unstaged, and the SOURCE's
    staged deletion — invisible to this function — is committed. The owner's
    tracked file leaves their branch while ``present_in_head`` reports an empty
    list, which is the one assertion this function exists to make truthfully.
    Turning detection off makes both halves of a rename visible as what the index
    actually holds: a deletion of a path HEAD has, and an addition of one it does
    not.
    """
    from ouroboros.headless import _sensitive_untracked_reason

    def _run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-c", "core.quotepath=off", *args],
            cwd=str(path), capture_output=True, text=True, timeout=60,
            env={**os.environ, "LC_ALL": "C", "GIT_LITERAL_PATHSPECS": "1"},
        )

    staged = _run("diff", "--cached", "--name-only", "--no-renames", "-z")
    if staged.returncode != 0:
        return [], [], (staged.stderr or "git diff --cached failed").strip()[:300]
    flagged = [
        rel for rel in (staged.stdout or "").split("\0")
        if rel and _sensitive_untracked_reason(rel)
    ]
    if not flagged:
        return [], [], ""
    if _run("rev-parse", "--verify", "-q", "HEAD").returncode != 0:
        # No commit yet: nothing can be tracked, so every flagged path is new.
        return flagged, [], ""
    listed = _run("ls-tree", "-r", "-z", "--name-only", "--full-tree", "HEAD")
    if listed.returncode != 0:
        return [], [], (listed.stderr or "git ls-tree HEAD failed").strip()[:300]
    in_head = {rel for rel in (listed.stdout or "").split("\0") if rel}
    return (
        [rel for rel in flagged if rel not in in_head],
        [rel for rel in flagged if rel in in_head],
        "",
    )


def _unstage_staged_paths(path: pathlib.Path, rels: list[str]) -> str:
    """``git rm --cached`` the given paths. Returns "" or the failure detail.

    ``GIT_LITERAL_PATHSPECS`` because these are FILENAMES, not patterns: a file
    literally named ``*.env`` or ``:(glob)token.json`` would otherwise be handed to
    git as pathspec magic and unstage something nobody named."""
    if not rels:
        return ""
    proc = subprocess.run(
        ["git", "-c", "core.quotepath=off", "rm", "-q", "--cached", "--", *rels],
        cwd=str(path), capture_output=True, text=True, timeout=60,
        env={**os.environ, "LC_ALL": "C", "GIT_LITERAL_PATHSPECS": "1"},
    )
    if proc.returncode != 0:
        return (proc.stderr or proc.stdout or "git rm --cached failed").strip()[:300]
    return ""


def _unstage_sensitive_paths(path: pathlib.Path) -> list[str]:
    """Unstage credential-shaped files after ``git add -A`` and keep them untracked
    via `.git/info/exclude` (local-only — the owner's folder files are never edited).
    Same `_sensitive_untracked_reason` SSOT the workspace patch and coop checkpoint
    use (triad r4: an attach snapshot must not bake `.env`/keys into history).
    Returns the skipped relative paths for disclosure.

    Only for a repository this process just created (``attach_snapshot_init``),
    which is why writing ``.git/info/exclude`` is unconditional here: the file is
    this function's own, one line old. A snapshot of a PRE-EXISTING repository must
    use :func:`_staged_sensitive_partition` directly and leave the owner's exclude
    file alone."""
    absent, _present, error = _staged_sensitive_partition(path)
    if error or not absent:
        return []
    if _unstage_staged_paths(path, absent):
        return []
    exclude = path / ".git" / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    with exclude.open("a", encoding="utf-8") as fh:
        fh.write("\n# ouroboros attach-snapshot: credential-shaped files stay untracked\n")
        for rel in absent:
            fh.write(f"/{rel}\n")
    return absent


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
