"""BRANCH OFF and MERGE BACK — the two explicit operations behind A7.

A thread's LOCATION is never a stored toggle. It is DERIVED from whether a
durable worktree exists for it (:func:`thread_location`), so there is no flag
that can disagree with the filesystem and no state machine to keep in sync. The
owner performs two operations instead:

**BRANCH OFF** provisions a worktree for the thread from a base the OWNER picks
(:func:`branch_off_bases` lists the current branch, every other branch, every
tag, and the "exactly as it is now" option) and binds it through the durable
registry in :mod:`ouroboros.thread_worktrees`. Any commit-ish the owner types is
accepted too — the list is an offer, not a restriction (A8).

"Exactly as it is now" is the only base that does not already exist as a commit,
so it is made into one: a SNAPSHOT commit on whatever the project folder has
checked out — its current branch, or a detached HEAD, in which case the commit is
held by the thread's new branch and nothing else — using the same shape as the
attach snapshot and the coop checkpoint (local identity, no global config
touched, credential-shaped files deliberately left out and disclosed). It refuses
outright on a folder git has stopped part-way through a merge, cherry-pick,
revert or rebase, because committing there would bake conflict markers into
tracked files and clear the sequencer state the owner was resolving. Reused
rather than reinvented, and never silent — the resulting sha and the skipped
paths come back in the receipt.

**MERGE BACK** merges the thread's branch into the project's own checkout under
A9's preconditions: nothing running anywhere in the project OR in the folder (the
project-WIDE activity query, NOT the writer lane — and the folder half too, since
the lane is folder-keyed and a second project's task in the same folder reduces
to a different id), a project folder that is ON a branch (a detached HEAD has
nothing for the merge commit to land on), a clean local tree, and a checkout
whose work is actually ON the branch being merged — uncommitted edits and commits
made on some other branch inside that checkout are refusals, because a merge moves
COMMITS ON A BRANCH and reporting success while the owner's work sits outside it
is the one failure they cannot see. A conflict is SHOWN and STOPS the operation:
the merge is aborted and the abort is VERIFIED, so "the folder was left exactly
as it was" is a checked fact rather than an assumption; when the abort does not
take, the answer says the folder is stopped mid-merge and what to do about it.
Merging never removes the worktree; removal is the separate, inspected act in
:mod:`ouroboros.thread_worktrees` (A10).

Every refusal here is a TYPED reason, never a raised string: these answer owner
gestures, and a UI cannot branch on a stack trace.
"""

from __future__ import annotations

import logging
import os
import pathlib
import subprocess
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

#: The owner-chosen base meaning "exactly as it is now, including uncommitted
#: edits". Not a git ref — it is the request for a snapshot commit (A8).
BASE_SNAPSHOT = "@snapshot"

#: Typed refusals. Every one of these is an answer to an owner gesture, so it
#: names what to do next rather than what went wrong internally.
REASON_NO_FOLDER = "no_project_folder"
REASON_FOLDER_MISSING = "folder_missing"
REASON_GIT_INIT_REQUIRED = "git_init_required"
REASON_FOLDER_UNUSABLE = "folder_unusable"
REASON_UNKNOWN_PROJECT = "unknown_project"
REASON_UNKNOWN_THREAD = "unknown_thread"
REASON_ALREADY_BRANCHED = "already_branched"
REASON_NOT_BRANCHED = "not_branched"
REASON_UNKNOWN_BASE = "unknown_base"
REASON_SNAPSHOT_FAILED = "snapshot_failed"
REASON_BRANCH_FAILED = "branch_failed"
REASON_PROJECT_BUSY = "project_busy"
REASON_LOCAL_TREE_DIRTY = "local_tree_dirty"
REASON_MERGE_CONFLICT = "merge_conflict"
REASON_MERGE_FAILED = "merge_failed"
REASON_MERGE_ABORT_FAILED = "merge_abort_failed"
REASON_CHECKOUT_DIRTY = "checkout_dirty"
REASON_CHECKOUT_HEAD_OFF_BRANCH = "checkout_head_off_branch"
REASON_CHECKOUT_MISSING = "checkout_missing"
REASON_PROJECT_HEAD_DETACHED = "project_head_detached"
REASON_THREAD_NOT_LIVE = "thread_not_live"


def _git_timeout_sec() -> float:
    """Bounded like every other owner-facing git call on a request path.

    Read from the ONE settings SSOT rather than pinned as a module-local number:
    DEVELOPMENT.md's gate says a new numeric timeout is a `config.py` setting with
    a getter and env registration, so this path has its OWN key,
    ``OUROBOROS_THREAD_GIT_TIMEOUT_SEC``, clamped like every sibling.

    It is deliberately not the task diff's key. Reusing that one satisfied the
    same gate but narrowed this ceiling from 120s to 30s, and the two paths are
    not the same kind of work: the diff endpoint runs one bounded READ against a
    commit, while :func:`_snapshot_commit` runs ``git add -A`` and ``git commit``
    over a working tree of unknown size. A snapshot that times out is exactly the
    failure the branch-off refusals exist to contain, so the WRITE does not
    inherit the READ's ceiling — it gets a knob the owner can turn on its own.
    """
    from ouroboros.config import get_thread_git_timeout_sec

    return get_thread_git_timeout_sec()


def _git(root: Any, *args: str) -> subprocess.CompletedProcess:
    """One bounded git call, invoked the same hardened way every time.

    A spawn failure or timeout comes back as rc=124 so every caller can treat "did
    not succeed" uniformly instead of splitting between an exit code and an
    exception.

    Three settings are pinned rather than inherited, because all three decide what
    this module READS and DOES to the owner's folder:

    * ``core.quotepath=off`` — a non-ASCII path must arrive as itself, not as the
      C-quoted spelling of a file that does not exist;
    * ``GIT_LITERAL_PATHSPECS=1`` — every path this module passes git is a
      FILENAME, so a file literally named ``*.env`` must never be read as a glob;
    * ``LC_ALL=C`` — git's messages are diagnostics here, and a localized build
      would make them unreadable in a log without changing any decision, because
      no decision is taken by matching git's prose (see :func:`_snapshot_commit`).
    """
    from ouroboros.platform_layer import bootstrap_process_path

    bootstrap_process_path()
    argv = ["git", "-c", "core.quotepath=off", *args]
    try:
        return subprocess.run(
            argv,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=_git_timeout_sec(),
            env={**os.environ, "LC_ALL": "C", "GIT_LITERAL_PATHSPECS": "1"},
        )
    except Exception as exc:  # noqa: BLE001 — includes TimeoutExpired
        return subprocess.CompletedProcess(
            argv, 124, stdout="", stderr=f"{type(exc).__name__}: {exc}"
        )


def _detail(proc: subprocess.CompletedProcess) -> str:
    return (proc.stderr or proc.stdout or "").strip()[:500]


def _refused(reason: str, message: str, **extra: Any) -> Dict[str, Any]:
    return {"ok": False, "reason": reason, "message": message, **extra}


#: How many dirty entries an envelope CARRIES. A display bound, never the
#: answer to "how many are there" — `dirty_files_total` rides beside every
#: bounded listing this module emits so no client counts the slice.
_DIRTY_FILES_SENT = 200


def _dirty_total(inspection: Dict[str, Any]) -> int:
    """The true number of dirty entries an inspection found.

    Falls back to the listing's own length for an inspection that predates the
    field, which is the old behaviour and never an under-count of what is
    actually in hand.
    """
    listed = len(inspection.get("dirty_files") or [])
    return max(listed, int(inspection.get("dirty_files_total") or 0))


def _live_thread_refusal(thread: Dict[str, Any], project_id: str) -> Optional[Dict[str, Any]]:
    """Refuse a git operation on a thread that is fenced or already gone.

    A thread being deleted is closed to routing and having its tasks cancelled;
    provisioning a checkout for it, or merging its branch, would be work on a
    room the owner has written off. An ARCHIVED thread is fine — archiving hides
    a thread, it does not close it.
    """
    from ouroboros.project_threads_registry import THREAD_ACTIVE, THREAD_ARCHIVED

    lifecycle = str(thread.get("lifecycle") or THREAD_ACTIVE)
    if lifecycle in {THREAD_ACTIVE, THREAD_ARCHIVED}:
        return None
    return _refused(
        REASON_THREAD_NOT_LIVE,
        f"This thread is {lifecycle}; it cannot branch off or merge back.",
        project_id=str(project_id), thread_id=int(thread.get("id") or 0),
    )


# --------------------------------------------------------------------------- #
# The project's PLACE
# --------------------------------------------------------------------------- #

def resolve_project_repo(drive_root: Any, project_id: str) -> Dict[str, Any]:
    """The git worktree root a project's threads branch off from, or a refusal.

    A project with no designated place cannot branch (there is nothing to branch
    FROM), and a place that is not tracked by git gets T2's typed
    ``git_init_required`` OFFER rather than an error — the same decision object
    the task-admission path returns, built by the same function, so the owner
    sees one consistent answer no matter which surface asked (A12).
    """
    from ouroboros.config import DATA_DIR, REPO_DIR
    from ouroboros.projects_registry import get_project
    from ouroboros.workspace_admission import (
        GitInitRequiredError,
        WorkspaceRootError,
        validate_workspace_root,
    )

    project = get_project(drive_root, project_id)
    if project is None:
        return _refused(REASON_UNKNOWN_PROJECT, f"unknown project: {project_id}")
    pid = str(project.get("id") or "")
    working_dir = str(project.get("working_dir") or "").strip()
    if not working_dir:
        return _refused(
            REASON_NO_FOLDER,
            "This project has no working folder yet, so there is nothing to branch "
            "off from. Give it a place first — attach a folder, clone a repo, or "
            "create one.",
            project_id=pid,
        )
    if not pathlib.Path(working_dir).expanduser().is_dir():
        return _refused(
            REASON_FOLDER_MISSING,
            f"The project's folder is gone: {working_dir}",
            project_id=pid,
            working_dir=working_dir,
        )
    try:
        root = validate_workspace_root(
            working_dir, system_repo_dir=REPO_DIR, drive_root=DATA_DIR,
        )
    except GitInitRequiredError as exc:
        decision = dict(exc.decision)
        decision["project_id"] = pid
        # Branching is one of the three things the offer already names.
        return _refused(
            REASON_GIT_INIT_REQUIRED,
            str(decision.get("message") or ""),
            project_id=pid,
            working_dir=working_dir,
            decision=decision,
        )
    except WorkspaceRootError as exc:
        return _refused(
            REASON_FOLDER_UNUSABLE, str(exc), project_id=pid, working_dir=working_dir,
        )
    return {"ok": True, "project_id": pid, "repo_dir": str(root), "project": project}


def thread_location(data_dir: Any, project_id: str, thread_id: Any) -> Dict[str, Any]:
    """WHERE a thread works — derived, never stored (A7).

    ``{"where": "project_folder"}`` or ``{"where": "worktree", ...}``. The single
    question "does a durable worktree exist for this thread" is the whole state
    machine; there is no toggle that can drift out of agreement with it.
    """
    from ouroboros.thread_worktrees import get_thread_worktree

    row = get_thread_worktree(data_dir, project_id, thread_id)
    if not row:
        return {"where": "project_folder"}
    return {
        "where": "worktree",
        "path": str(row.get("path") or ""),
        "branch": str(row.get("branch") or ""),
        "base_sha": str(row.get("base_sha") or ""),
        "created_at": str(row.get("created_at_iso") or ""),
    }


# --------------------------------------------------------------------------- #
# BRANCH OFF
# --------------------------------------------------------------------------- #

def _current_branch(repo_dir: pathlib.Path) -> str:
    """The branch a working tree is ON, or ``""`` when it is on none.

    ``rev-parse --abbrev-ref HEAD`` answers the literal string ``"HEAD"`` for a
    DETACHED head. That is not a branch name, and every caller that treated it as
    one was wrong in a different way: the bases list quoted it back to the owner
    as ``current_branch: "HEAD"`` while its own loop knew better, and merge-back
    would have merged onto nothing. One helper, one answer — a detached head has
    no branch, and says so by returning nothing.
    """
    head = _git(repo_dir, "rev-parse", "--abbrev-ref", "HEAD")
    name = (head.stdout or "").strip() if head.returncode == 0 else ""
    return "" if name == "HEAD" else name


def branch_off_bases(repo_dir: Any) -> Dict[str, Any]:
    """Every base the owner may branch off from, in offer order (A8).

    The current branch first because it is the common answer, then the other
    branches, then tags. The "exactly as it is now" entry is ALWAYS present — it
    is one option in the list, not a restriction — and it discloses whether the
    tree actually has uncommitted work, so the owner can tell whether choosing it
    would create a snapshot commit or simply reuse HEAD.

    A commit-ish the owner types instead is accepted by :func:`branch_off_thread`
    and deliberately not enumerated here: listing every commit is not an offer.

    ``current_branch`` is ``""`` for a folder standing on a DETACHED head. It used
    to be the literal string ``"HEAD"`` — git's spelling for "no branch" — which
    the bases loop already knew to skip while the field handed it to the client as
    a branch name to display and to branch off from.
    """
    root = pathlib.Path(str(repo_dir))
    current = _current_branch(root)
    bases: List[Dict[str, Any]] = []
    seen: set = set()
    if current:
        bases.append({"ref": current, "kind": "branch", "label": f"{current} (current)"})
        seen.add(current)
    for kind, pattern in (("branch", "refs/heads"), ("tag", "refs/tags")):
        listed = _git(root, "for-each-ref", "--format=%(refname:short)", pattern)
        if listed.returncode != 0:
            continue
        for name in (listed.stdout or "").splitlines():
            name = name.strip()
            if not name or name in seen:
                continue
            seen.add(name)
            bases.append({"ref": name, "kind": kind, "label": name})
    status = _git(root, "status", "--porcelain")
    dirty = status.returncode == 0 and bool((status.stdout or "").strip())
    return {
        "current_branch": current,
        "bases": bases,
        "snapshot": {
            "ref": BASE_SNAPSHOT,
            "kind": "snapshot",
            "label": "Exactly as it is now (including uncommitted edits)",
            "dirty": dirty,
            "creates_commit": dirty,
        },
    }


#: The refs git leaves behind while it is holding a working tree open part-way
#: through a multi-step operation, and the owner-facing name of each. Asked of
#: git rather than of ``.git/`` on disk, for the same reason :func:`_in_merge` is:
#: a linked worktree keeps its own per-worktree git dir and the obvious path is
#: not where the file lives.
_SEQUENCER_REFS = (
    ("MERGE_HEAD", "merge"),
    ("CHERRY_PICK_HEAD", "cherry-pick"),
    ("REVERT_HEAD", "revert"),
    ("REBASE_HEAD", "rebase"),
)


def _sequencer_operation(repo_dir: pathlib.Path) -> str:
    """The multi-step git operation this tree is stopped inside, or ``""``."""
    for ref, name in _SEQUENCER_REFS:
        if _git(repo_dir, "rev-parse", "--verify", "-q", ref).returncode == 0:
            return name
    return ""


def _snapshot_aborted(repo_dir: pathlib.Path, detail: str) -> Dict[str, Any]:
    """Give the owner's INDEX back, then report why the snapshot did not happen.

    ``git add -A`` has staged the entire folder by the time anything downstream can
    fail. Returning the failure without undoing that leaves the owner in a
    repository they did not ask for and would not recognise — every file staged,
    ``git status`` unrecognisable, and no commit to explain it. A MIXED reset puts
    the index back to HEAD and touches NO file in the working tree, so every file
    the owner has is exactly as they left it.

    Their INDEX is not: a mixed reset also discards a curated one (an owner
    mid-``git add -p``). ``git add -A`` had already destroyed that staging before
    this point, so the reset cannot restore it and does not pretend to — this is
    the least-wrong end state, not a lossless one.
    """
    reset = _git(repo_dir, "reset", "-q")
    if reset.returncode != 0:
        log.warning("snapshot: the index could not be restored after a failure")
        return {
            "ok": False,
            "detail": f"{detail} — and the staged index could not be restored: {_detail(reset)}",
        }
    return {"ok": False, "detail": detail}


def _snapshot_commit(repo_dir: pathlib.Path, label: str) -> Dict[str, Any]:
    """Commit the project folder EXACTLY as it stands, so it can be branched from.

    Same shape as ``project_sources.attach_snapshot_init`` and the coop
    checkpoint: a local identity (the owner's global git config is never touched),
    and credential-shaped files that git does not yet track left OUT of the commit
    through the ONE ``_sensitive_untracked_reason`` authority, so a snapshot never
    bakes a NEW secret into history. The skipped paths are returned, never
    swallowed.

    What this deliberately does NOT do is touch a file HEAD already tracks. Here
    the repository is the OWNER'S and pre-existing, so ``git rm --cached`` on a
    tracked path stages a DELETION: the file leaves ``git ls-files``, the owner's
    branch gains a commit removing it, and nothing is protected in exchange —
    a tracked file's contents are in history already and unstaging cannot retract
    them. So a credential-shaped file that is already tracked is snapshotted like
    any other tracked file and DISCLOSED as ``tracked_sensitive``; the owner is
    told, and nothing of theirs is deleted to make a point.

    The owner's ``.git/info/exclude`` is left alone for the same reason: it is
    their file, and this operation has no business rewriting it.

    A clean tree needs no commit at all and simply reports the current HEAD:
    "as it is now" is already a commit in that case. So does a tree whose only
    changes were untracked secrets — with those left out there is nothing to
    commit, and that is read from the INDEX (``git diff --cached --quiet``) rather
    than from git's English prose, which a localized or reworded git would break.
    """
    from ouroboros.project_sources import (
        _staged_sensitive_partition,
        _unstage_staged_paths,
    )

    def _head_snapshot(**extra: Any) -> Dict[str, Any]:
        head = _git(repo_dir, "rev-parse", "HEAD")
        if head.returncode != 0:
            return {"ok": False, "detail": _detail(head)}
        return {"ok": True, "sha": (head.stdout or "").strip(), "created": False, **extra}

    stopped = _sequencer_operation(repo_dir)
    if stopped:
        # Checked FIRST, before `git add -A` has touched anything. A folder git has
        # stopped part-way through holds the owner's half-resolved conflict: the
        # overlapping files contain conflict markers, and the sequencer ref is the
        # only record of what was being merged. Committing there stages those
        # markers into tracked files AND clears MERGE_HEAD, so `git merge --abort`
        # stops working and the resolution the owner was in the middle of is gone
        # with no way back. "Exactly as it is now" is not a thing that can be
        # committed while git itself is holding the folder open.
        return {
            "ok": False,
            "detail": (
                f"git has this folder stopped part-way through a {stopped}. Finish it "
                "or undo it in that folder first — snapshotting now would commit the "
                "conflict markers as if they were your files and throw away the "
                f"{stopped} git is still holding open."
            ),
        }

    status = _git(repo_dir, "status", "--porcelain")
    if status.returncode != 0:
        return {"ok": False, "detail": _detail(status)}
    if not (status.stdout or "").strip():
        return _head_snapshot(skipped_sensitive=[], tracked_sensitive=[])
    add = _git(repo_dir, "add", "-A")
    if add.returncode != 0:
        return _snapshot_aborted(repo_dir, _detail(add))
    try:
        skipped, tracked_sensitive, error = _staged_sensitive_partition(repo_dir)
        if error:
            # "Which of these does HEAD already have" has no safe default: guessing
            # ABSENT deletes the owner's tracked file, guessing PRESENT commits a new
            # secret. Refuse the snapshot and say so.
            return _snapshot_aborted(
                repo_dir,
                f"could not tell which credential-shaped files are already tracked: {error}",
            )
        unstage_error = _unstage_staged_paths(repo_dir, skipped)
        if unstage_error:
            return _snapshot_aborted(repo_dir, unstage_error)
    except Exception as exc:  # noqa: BLE001 — includes TimeoutExpired and OSError
        # These two RAISE as well as return: their subprocess calls are unguarded,
        # so a timeout or a spawn failure travelled straight out of here — past a
        # `branch_off_thread` whose try covers only provisioning — with `git add -A`
        # already done. The owner was left with their entire folder staged, no
        # commit to explain it, and a 500 instead of a sentence. The returned-error
        # path always restored the index; the raised one has to do the same.
        return _snapshot_aborted(repo_dir, f"{type(exc).__name__}: {exc}")
    staged = _git(repo_dir, "diff", "--cached", "--quiet")
    if staged.returncode == 0:
        # Nothing left to commit: every change was a credential-shaped new file
        # and it stayed out. The base is HEAD, and the index goes back untouched.
        _git(repo_dir, "reset", "-q")
        return _head_snapshot(
            skipped_sensitive=skipped, tracked_sensitive=tracked_sensitive,
        )
    commit = _git(
        repo_dir,
        "-c", "user.name=Ouroboros", "-c", "user.email=ouroboros@local",
        "commit", "-q", "-m", f"ouroboros: snapshot before branching off {label}".strip(),
    )
    if commit.returncode != 0:
        return _snapshot_aborted(repo_dir, _detail(commit))
    head = _git(repo_dir, "rev-parse", "HEAD")
    if head.returncode != 0:
        return {"ok": False, "detail": _detail(head)}
    return {
        "ok": True,
        "sha": (head.stdout or "").strip(),
        "created": True,
        "skipped_sensitive": skipped,
        "tracked_sensitive": tracked_sensitive,
    }


def branch_off_thread(
    drive_root: Any,
    project_id: str,
    thread_id: Any,
    *,
    base_ref: str = "",
    data_dir: Optional[Any] = None,
    worktree_root: Optional[Any] = None,
) -> Dict[str, Any]:
    """Provision this thread's own checkout from an owner-chosen base (A7/A8).

    ``base_ref`` is a branch, a tag, any commit-ish, or :data:`BASE_SNAPSHOT`;
    empty means the project's current HEAD. The worktree is bound to the thread
    through the durable registry, which refuses to clobber an existing checkout
    or branch — so a second branch-off is an error the owner sees, never a silent
    reset of work they already did.
    """
    from ouroboros.projects_registry import get_thread
    from ouroboros.thread_worktrees import provision_thread_worktree

    resolved = resolve_project_repo(drive_root, project_id)
    if not resolved.get("ok"):
        return resolved
    pid = str(resolved["project_id"])
    repo_dir = pathlib.Path(str(resolved["repo_dir"]))
    data_root = data_dir if data_dir is not None else drive_root

    thread = get_thread(drive_root, pid, thread_id)
    if thread is None:
        return _refused(
            REASON_UNKNOWN_THREAD, f"unknown thread {thread_id!r} in project {pid!r}",
            project_id=pid,
        )
    not_live = _live_thread_refusal(thread, pid)
    if not_live is not None:
        return not_live
    tid = int(thread["id"])
    location = thread_location(data_root, pid, tid)
    if location["where"] == "worktree":
        return _refused(
            REASON_ALREADY_BRANCHED,
            "This thread is already working in its own branch. Merge it back or "
            "remove the checkout before branching off again.",
            project_id=pid, thread_id=tid, location=location,
        )

    wanted = str(base_ref or "").strip()
    snapshot: Dict[str, Any] = {}
    if wanted == BASE_SNAPSHOT:
        # The ONE arm of branch-off that WRITES the owner's folder: `git add -A`
        # plus a commit, in the project folder itself. Merge-back guards its write
        # twice — it holds the folder's lane and then asks `project_is_busy` — and
        # this one asked neither, so a live task's half-written scratch file became
        # a commit on the owner's branch while `project_is_busy` was answering True
        # for that exact folder one line earlier (I6). Guarded the same way, and
        # only here: every other base reads a commit-ish and writes nothing to the
        # project folder, so branching off a branch or a tag must keep working
        # while a task runs.
        from ouroboros.project_lease import reserved_folder_lane
        with reserved_folder_lane(repo_dir):
            if project_is_busy(pid, repo_dir):
                return _refused(
                    REASON_PROJECT_BUSY,
                    "A task is running or queued in this project right now. "
                    "\"Exactly as it is now\" commits everything in the project "
                    "folder, so it would bake that task's half-written files into "
                    "your history — it waits until that task finishes. Branching "
                    "off a branch, a tag or a commit does not touch the folder and "
                    "works now.",
                    project_id=pid, thread_id=tid,
                )
            snapshot = _snapshot_commit(repo_dir, str(thread.get("name") or f"thread {tid}"))
        if not snapshot.get("ok"):
            return _refused(
                REASON_SNAPSHOT_FAILED,
                "Could not snapshot the folder as it is now: "
                f"{snapshot.get('detail') or 'git refused'}",
                project_id=pid, thread_id=tid,
            )
        base = str(snapshot["sha"])
    elif wanted:
        verified = _git(repo_dir, "rev-parse", "--verify", f"{wanted}^{{commit}}")
        if verified.returncode != 0:
            return _refused(
                REASON_UNKNOWN_BASE,
                f"{wanted!r} is not a branch, tag or commit in this repository.",
                project_id=pid, thread_id=tid, base_ref=wanted,
            )
        base = wanted
    else:
        base = ""

    try:
        handle = provision_thread_worktree(
            repo_dir=repo_dir,
            project_id=pid,
            thread_id=tid,
            base_ref=base,
            data_dir=data_root,
            worktree_root=worktree_root,
        )
    except Exception as exc:  # noqa: BLE001 — provisioning refusals are typed answers
        # The snapshot commit is already MADE by the time provisioning can refuse,
        # and it is on the owner's own branch. Reporting only "branching failed"
        # told them nothing happened while their uncommitted work had silently
        # become a commit they did not author and could not find. The commit is
        # NOT undone here — a reset is a second mutation on a folder an operation
        # has just failed in, and this refusal has no way to know the owner has not
        # already looked. It is NAMED instead, which is lossless and checkable.
        return _refused(
            REASON_BRANCH_FAILED,
            _branch_failed_message(str(exc)[:500], snapshot),
            project_id=pid, thread_id=tid,
            **({"snapshot_commit": _snapshot_receipt(snapshot)} if snapshot.get("created") else {}),
        )
    out: Dict[str, Any] = {
        "ok": True,
        "project_id": pid,
        "thread_id": tid,
        "base_ref": wanted or "HEAD",
        "location": thread_location(data_root, pid, tid),
        "branch": handle.branch,
        "path": handle.path,
        "base_sha": handle.base_sha,
    }
    if snapshot:
        out["snapshot_commit"] = _snapshot_receipt(snapshot)
    return out


def _snapshot_receipt(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """What the snapshot did, in the shape both the success and the refusal use."""
    return {
        "sha": str(snapshot.get("sha") or ""),
        "created": bool(snapshot.get("created")),
        # Kept OUT of the commit and still untracked in the owner's folder.
        "skipped_sensitive": list(snapshot.get("skipped_sensitive") or []),
        # Credential-shaped but ALREADY tracked, so snapshotted like anything else
        # git tracks. Disclosed rather than deleted (T3R-1).
        "tracked_sensitive": list(snapshot.get("tracked_sensitive") or []),
    }


def _branch_failed_message(detail: str, snapshot: Dict[str, Any]) -> str:
    """The refusal copy, plus the snapshot commit when one was already made."""
    if not snapshot.get("created"):
        return detail
    sha = str(snapshot.get("sha") or "")
    return (
        f"{detail} — and your folder is NOT as you left it: the uncommitted changes "
        f"were committed first, as {sha[:12] or 'a snapshot commit'}, on the branch "
        "that folder had checked out. Nothing is lost; if you did not want a commit, "
        "undo it with a reset in that folder."
    )


# --------------------------------------------------------------------------- #
# MERGE BACK
# --------------------------------------------------------------------------- #

def project_is_busy(project_id: str, repo_dir: Any = None) -> bool:
    """Is ANYTHING alive in this project, or in this FOLDER? (A9's precondition.)

    Reads the project-WIDE ACTIVITY query, deliberately NOT the writer lane: a
    merge touches the project as a whole, so a task in a DIFFERENT folder of the
    same project still blocks it. Which means it counts the two things the lane
    deliberately ignores, and both are stated here because "any task running
    anywhere in this project" was not what the code actually asked (T3R-14):

    * SUBAGENTS count. The lane exempts them so a swarm cannot deadlock against
      its own parent; that is a scheduling rule about who may be ASSIGNED. A
      subagent still writes files in the project, and exempting it here let a
      merge rewrite the folder while a swarm member was mid-write.
    * PENDING counts, except work that cannot start without the owner. A queued
      task for this project can be assigned at any instant, including the one
      right after this returns. A BUDGET-PAUSED one cannot: it waits for the
      owner, possibly forever and across a queue-snapshot restore, so counting it
      made "this project is busy" permanently true and locked the owner out of
      merging their own work back with nothing on screen to explain why. Work only
      the owner can release is their decision, already taken — not activity to
      wait behind.

    ``repo_dir`` adds the FOLDER half, and it is not optional in practice: the
    lane is keyed on the folder alone now, so "is this folder busy" no longer
    reduces to a project id. Project *alpha* merging into a folder while project
    *beta*'s task writes in it read as idle, and so did a task carrying a
    ``workspace_root`` with no ``project_id`` at all — which holds no lane and is
    still in the folder.

    A non-TASK holder counts too. ``reserved_folder_lane`` is how a merge-back
    holds the folder it is rewriting, and it was unioned into
    ``running_project_lanes`` so the SCHEDULER sees it — but this function, the
    SSOT behind every owner gesture's precondition, read only the two task
    queries. So during a merge-back the scheduler correctly saw the folder held
    while a second merge-back, a checkout removal and a thread delete were all
    told IDLE, and a second holder was admitted into the folder mid-merge. There
    must be ONE answer to "is this folder occupied", and it is this one (I5).

    Fail-CLOSED — if the queue cannot be read, the project counts as busy,
    because "cannot tell" must never license a merge into a folder something
    might be writing in.
    """
    try:
        from ouroboros.project_lease import (
            normalize_workspace_root,
            reserved_folder_lanes,
            running_project_ids,
            running_workspace_roots,
        )
        from supervisor.queue import _queue_lock
        from supervisor.workers import PENDING, RUNNING

        with _queue_lock:
            running = list(RUNNING.values())
            pending = list(PENDING)
        if str(project_id) in running_project_ids(running, pending):
            return True
        folder = normalize_workspace_root(repo_dir)
        if not folder:
            return False
        # The reservation set keys on the SAME normalization this line already
        # applied, so the comparison is a real one rather than theatre.
        # `include_own=False`: the caller may BE a holder (merge-back asks this
        # from inside its own reservation), and an operation refused by its own
        # claim would never run at all.
        if ("", folder) in reserved_folder_lanes(include_own=False):
            return True
        return folder in running_workspace_roots(running, pending)
    except Exception:
        log.debug("project_is_busy could not read the queue for %s", project_id, exc_info=True)
        return True


#: A14's copy, in ONE place. Every surface that tells the owner their work will
#: wait says exactly this, and says the true thing: the task is QUEUED behind the
#: running one and will run when it finishes. It is not rejected, not dropped,
#: and not silently reordered. The remedy is offered in the same breath, because
#: "you have to wait" without "here is how not to" is a dead end.
#:
#: It names the FOLDER and not the project, because after T0R2-5 the lane is
#: keyed on the folder alone: whatever is holding it may belong to another
#: thread, another project, or no project at all. "Another thread in this
#: project" was a guess about the occupant, and a wrong guess sends the owner
#: looking for a room that is not the one making them wait (T3R-15).
QUEUE_NOTICE = (
    "Another task is working in this folder right now. "
    "A task you start here will be QUEUED behind it and will run as soon as that "
    "one finishes — it is not rejected. Branching this thread off gives it its "
    "own copy of the folder, so both can run at the same time."
)
#: The same fact for a thread already in its own checkout, where waiting means
#: something is running in THAT checkout — branching again would not help.
QUEUE_NOTICE_OWN_CHECKOUT = (
    "This thread already has a task running in its own checkout. A new task here "
    "will be QUEUED behind it and will run as soon as that one finishes."
)


def queue_notice(
    drive_root: Any,
    project_id: str,
    thread_id: Any,
    *,
    data_dir: Optional[Any] = None,
    running: Optional[Any] = None,
) -> Dict[str, Any]:
    """Would a task started in THIS thread wait, and what should the owner hear?

    Returns ``{queued, reason, message, remedy}``. ``remedy`` is ``branch_off``
    only when branching would actually help — a thread already working in its own
    checkout is waiting on ITSELF, and offering to branch again there would be
    advice that does not work.

    A14 exists because the earlier copy said a second thread's task was rejected.
    It never was: the writer lane SERIALIZES, it does not refuse, and telling an
    owner their work was thrown away when it is sitting in the queue is the kind
    of wrong that makes people stop trusting the queue entirely.

    Fail-OPEN, unlike the merge precondition: if the queue cannot be read this
    says nothing rather than warning about a wait that may not exist. A false
    warning here costs trust; a missing one costs a few seconds of surprise.

    ALL of the work is inside that guard, which it was not (T3R-13): the lease
    import and the two resolver calls sat outside it, so a `project_lease` that
    would not import, a registry read that raised, or a folder probe that threw
    turned a decorative advisory into a 500 for the whole branch-bases route —
    taking the owner's list of bases down with it. This notice is the least
    important thing on that answer; nothing about it may be able to remove the
    rest.
    """
    quiet = {"queued": False, "reason": "", "message": "", "remedy": ""}
    try:
        from ouroboros.project_lease import candidate_is_leasable, running_project_lanes

        data_root = data_dir if data_dir is not None else drive_root
        resolved = resolve_project_repo(drive_root, project_id)
        location = thread_location(data_root, project_id, thread_id)
        if location["where"] == "worktree":
            workspace = str(location.get("path") or "")
        elif resolved.get("ok"):
            workspace = str(resolved.get("repo_dir") or "")
        else:
            # No usable folder means no folder lane to contend for; whatever else
            # is wrong with this project, waiting is not it.
            return quiet
        if running is None:
            from supervisor.queue import _queue_lock
            from supervisor.workers import RUNNING

            with _queue_lock:
                running = list(RUNNING.values())
        # The SAME project->folder map the scheduler uses. A RUNNING task that named
        # no folder of its own keys on ("", registered_folder) in assign_tasks; read
        # without the map it would key on (project_id, "") here, and this notice
        # would promise "no wait" for the very folder the queue is about to make the
        # owner wait on. It can be None — an unreadable registry — which the lease
        # reads as "the folder is unknown" rather than "there is none" (I3), so this
        # notice stays in agreement with the scheduler in that case too.
        from ouroboros.projects_registry import project_working_dirs

        folders = project_working_dirs(data_root)
        lanes = running_project_lanes(running, folders)
        candidate = {"id": "", "project_id": str(project_id), "workspace_root": workspace}
        if candidate_is_leasable(candidate, lanes, folders):
            return quiet
        own = location["where"] == "worktree"
    except Exception:
        log.debug("queue_notice could not be computed for %s", project_id, exc_info=True)
        return quiet
    return {
        "queued": True,
        "reason": "folder_busy",
        "message": QUEUE_NOTICE_OWN_CHECKOUT if own else QUEUE_NOTICE,
        "remedy": "" if own else "branch_off",
    }


def _in_merge(repo_dir: pathlib.Path) -> bool:
    """Does this working tree have a merge IN PROGRESS right now?

    Asked of git rather than of ``.git/MERGE_HEAD`` on disk, because a linked
    worktree keeps its own per-worktree git dir and the file is not where the
    obvious path says. Answering from the ref is also what makes the post-abort
    check meaningful: it is the same question git itself asks.
    """
    return _git(repo_dir, "rev-parse", "--verify", "-q", "MERGE_HEAD").returncode == 0


def _checkout_ahead_refusal(
    row: Dict[str, Any], pid: str, tid: int, branch: str, inspection: Dict[str, Any],
    *, acknowledged: bool = False,
) -> Optional[Dict[str, Any]]:
    """Refuse a merge that would silently leave the thread's work behind.

    Merging is a BRANCH operation, and a branch knows nothing about work that
    never reached it. Two shapes reached the owner as ``ok: true``:

    * the checkout has uncommitted work — the merge brings the branch's commits
      home and reports success, while the edits the owner was looking at in that
      folder stay behind. ``merged: true`` makes it read like everything came;
    * the checkout's HEAD is on a DIFFERENT branch than the one bound to the
      thread — every commit went to that branch, the bound branch never moved, and
      the merge answers ``ok: true, merged: false``, which the UI renders as
      "nothing new to merge — the folder already has this work". It does not.

    Both are refusals, not warnings: an owner who is told "merged" stops looking.
    The evidence is the SAME ``inspect_thread_worktree`` the removal prompt uses,
    so the two surfaces cannot disagree about what a checkout is holding.

    The DIRTY one is acknowledgeable, on A10's existing consent shape. A checkout
    an agent has actually worked in almost always holds something untracked — a
    log, a build artifact, a scratch file — and a refusal with no way past it
    would make merge-back unreachable for exactly the threads that did work. The
    branch being WRONG is not acknowledgeable: that is not work left behind, it
    is a merge that would do nothing while reporting success.

    ``inspection`` is passed IN — the caller needs it again to name what stayed
    behind on a successful merge, and inspecting a checkout twice could answer
    twice.
    """
    checkout = pathlib.Path(str(row.get("path") or ""))
    head_ref = _git(checkout, "rev-parse", "--abbrev-ref", "HEAD")
    on = (head_ref.stdout or "").strip() if head_ref.returncode == 0 else ""
    if on and branch and on != branch:
        # `--abbrev-ref` answers the literal string "HEAD" for a DETACHED head,
        # which is not a branch name and must not be quoted back as one.
        where = "not on any branch (a detached HEAD)" if on == "HEAD" else f"on {on!r}"
        remedy = (
            f"Check {branch!r} back out in that folder first"
            if on == "HEAD"
            else f"Switch the checkout back to {branch!r}, or merge {on!r} into it there, first"
        )
        return _refused(
            REASON_CHECKOUT_HEAD_OFF_BRANCH,
            f"This thread's checkout is {where}, not on {branch!r} — the branch "
            "this thread merges back. Anything committed there is NOT on "
            f"{branch!r}, so merging now would report success and bring none of "
            f"it. {remedy}.",
            project_id=pid, thread_id=tid, branch=branch,
            checkout_branch=on, path=str(checkout), inspection=inspection,
        )
    if acknowledged:
        # The owner has SEEN this and said merge anyway. Nothing is hidden: the
        # files that will stay behind ride the successful answer too.
        return None
    if inspection.get("dirty") or inspection.get("error"):
        detail = (
            "could not be read, so what it is holding is unknown"
            if inspection.get("error")
            else "has changes that were never committed"
        )
        return _refused(
            REASON_CHECKOUT_DIRTY,
            f"This thread's checkout {detail}. Merging brings its COMMITS home and "
            "nothing else, so that work would stay behind while the answer said "
            "the merge was done. Commit it in the checkout first, discard it, or "
            "merge anyway knowing it stays in the checkout.",
            project_id=pid, thread_id=tid, branch=branch,
            path=str(checkout), inspection=inspection,
            dirty_files=list(inspection.get("dirty_files") or [])[:_DIRTY_FILES_SENT],
            # The bounded list never travels without the true size of the set it
            # was cut from, or a client counts the slice.
            dirty_files_total=_dirty_total(inspection),
            # A10's consent shape, reused: the refusal names the flag that
            # answers it, so the owner is never stuck with only "no".
            acknowledgeable=True,
        )
    return None


def merge_back_thread(
    drive_root: Any,
    project_id: str,
    thread_id: Any,
    *,
    data_dir: Optional[Any] = None,
    busy: Optional[bool] = None,
    acknowledge_checkout_dirty: bool = False,
) -> Dict[str, Any]:
    """Merge a branched thread's work back into the project's own checkout (A9).

    FOUR preconditions, each refused with a typed reason and honest copy:
    nothing alive anywhere in the project, a project folder standing ON a branch,
    a clean local tree, and a checkout whose work is actually ON the branch being
    merged.

    The branch one is about where the merge LANDS: ``git merge --no-ff`` onto a
    detached HEAD succeeds and leaves the merge commit on no branch, after which
    both of this phase's safety judges — the unmerged-commit count and ``git
    branch -d`` — read that dangling commit as the project's HEAD and agree there
    is nothing left to lose. The work ends up reflog-only. Like the checkout being
    off its branch, it is deliberately NOT acknowledgeable.

    The last exists because a
    merge moves COMMITS ON A BRANCH: uncommitted edits in the checkout, and
    commits made there on some other branch, do not travel with it, and answering
    ``ok: true`` while they stay behind is the one failure the owner cannot see.
    ``acknowledge_checkout_dirty`` is the owner's answer to that first case, in
    A10's existing consent shape — and the files that stay behind are named on
    the successful answer too, so acknowledging it is not the same as forgetting.

    "Clean local tree" means TRACKED changes. Untracked files in the owner's
    folder are not part of a merge and do not blur which work came from where,
    which is what that precondition is actually protecting; counting them meant a
    project holding one stray `.env` or build artifact could never merge anything
    back, forever, with copy telling the owner to commit or stash a file they
    deliberately keep out of git.

    A conflict is SHOWN with its paths and STOPS the operation — the merge is
    aborted, so the owner's folder is left byte-for-byte as it was and the thread
    keeps every commit in its own branch. The abort is CHECKED, because that last
    sentence is a claim about a git command that can fail; when it did not take,
    :data:`REASON_MERGE_ABORT_FAILED` says the folder is stopped mid-merge and
    what the owner has to do about it.

    The worktree SURVIVES a successful merge. Removing it is a separate, inspected
    act (A10) so the owner is always the one who decides that the checkout has
    served its purpose.

    ``busy`` overrides the live activity query (tests, and callers that already
    hold the answer).
    """
    from ouroboros.projects_registry import get_thread
    from ouroboros.thread_worktrees import get_thread_worktree

    resolved = resolve_project_repo(drive_root, project_id)
    if not resolved.get("ok"):
        return resolved
    pid = str(resolved["project_id"])
    repo_dir = pathlib.Path(str(resolved["repo_dir"]))
    data_root = data_dir if data_dir is not None else drive_root

    thread = get_thread(drive_root, pid, thread_id)
    if thread is None:
        return _refused(
            REASON_UNKNOWN_THREAD, f"unknown thread {thread_id!r} in project {pid!r}",
            project_id=pid,
        )
    not_live = _live_thread_refusal(thread, pid)
    if not_live is not None:
        return not_live
    tid = int(thread["id"])
    row = get_thread_worktree(data_root, pid, tid)
    if not row:
        return _refused(
            REASON_NOT_BRANCHED,
            "This thread works in the project folder, so there is nothing to merge back.",
            project_id=pid, thread_id=tid,
        )
    branch = str(row.get("branch") or "")
    checkout = pathlib.Path(str(row.get("path") or ""))
    if not checkout.is_dir():
        return _refused(
            REASON_CHECKOUT_MISSING,
            f"The thread's checkout is gone from disk: {checkout}. Its branch "
            f"{branch!r} still holds the commits.",
            project_id=pid, thread_id=tid, branch=branch,
        )

    # M5: HOLD the folder for the merge's duration. `project_is_busy` is a bare
    # READ — it answers about the instant it ran, and the instant after that a
    # task the scheduler was already holding could be admitted straight into the
    # folder this is rewriting. That is the two-writer state the lane exists to
    # prevent, reached through the gap between a check and the work it checked
    # for. The reservation is released in a `finally` inside the context manager,
    # so a failed merge can never strand a folder nobody can schedule into.
    from ouroboros.project_lease import reserved_folder_lane

    with reserved_folder_lane(repo_dir):
        if project_is_busy(pid, repo_dir) if busy is None else bool(busy):
            return _refused(
                REASON_PROJECT_BUSY,
                "A task is running or queued in this project right now. Merging while "
                "something is writing could mix half-finished work into the folder, so "
                "it waits until that task finishes.",
                project_id=pid, thread_id=tid, branch=branch,
            )
        # A folder already stopped part-way through a merge cannot be merged into,
        # and it must not be told to "commit or stash" — that is advice for a folder
        # with edits in it, not one with MERGE_HEAD set and conflict markers in the
        # files. Without this, the retry after `merge_abort_failed` sent the owner
        # somewhere that could only make it worse.
        if _in_merge(repo_dir):
            return _refused(
                REASON_MERGE_ABORT_FAILED,
                "The project folder is stopped part-way through an earlier merge: git "
                "still has that merge in progress. Nothing here is lost and the thread "
                "keeps its branch, but the folder has to come out of it first — in that "
                "folder, `git merge --abort` goes back, or resolve the files and commit "
                "to go forward.",
                project_id=pid, thread_id=tid, branch=branch,
                folder_left_mid_merge=True, working_dir=str(repo_dir),
            )
        # A merge needs somewhere to LAND. `git merge --no-ff` onto a detached head
        # succeeds — it writes the merge commit and moves HEAD to it — and the commit
        # belongs to no branch at all: the moment the folder checks anything else out
        # it is reachable only through the reflog. Both of this phase's safety judges
        # are fooled by the same wrong reference: `inspect_thread_worktree` counts
        # unmerged commits against the project's HEAD, which now IS that dangling
        # merge, so it answers zero; and `git branch -d` agrees the thread branch is
        # merged, because against that HEAD it is. So a one-click removal deletes the
        # checkout AND the branch, and the only remaining copy of the work is a reflog
        # entry. Deliberately NOT acknowledgeable, exactly like a checkout standing off
        # its branch: this is not work left behind, it is a merge with no destination.
        if not _current_branch(repo_dir):
            return _refused(
                REASON_PROJECT_HEAD_DETACHED,
                "The project folder is not on any branch (a detached HEAD), so there is "
                "nothing for this merge to land on: git would make the merge commit and "
                "no branch would point at it, leaving the work unreachable the moment "
                "the folder checks something else out. Check a branch out in that folder "
                f"first, then merge {branch!r} back.",
                project_id=pid, thread_id=tid, branch=branch, working_dir=str(repo_dir),
            )
        # TRACKED changes only. An untracked file is not part of a merge and cannot
        # blur which work came from where, which is the whole point of this check.
        status = _git(repo_dir, "status", "--porcelain", "--untracked-files=no")
        if status.returncode != 0:
            return _refused(
                REASON_MERGE_FAILED, _detail(status), project_id=pid, thread_id=tid, branch=branch,
            )
        dirty = [line for line in (status.stdout or "").splitlines() if line.strip()]
        if dirty:
            return _refused(
                REASON_LOCAL_TREE_DIRTY,
                "The project folder has uncommitted changes. Commit or stash them "
                "first — merging on top of them would blur which work came from where.",
                project_id=pid, thread_id=tid, branch=branch,
                dirty_files=dirty[:_DIRTY_FILES_SENT], dirty_files_total=len(dirty),
            )

        # A9's LAST precondition, and the one only the checkout can answer: is the
        # branch about to be merged actually where the thread's work IS? Checked here,
        # after the cheap preconditions and BEFORE anything is merged, so a refusal
        # leaves nothing half-done.
        from ouroboros.thread_worktrees import inspect_thread_worktree

        inspection = inspect_thread_worktree(row)
        ahead = _checkout_ahead_refusal(
            row, pid, tid, branch, inspection,
            acknowledged=bool(acknowledge_checkout_dirty),
        )
        if ahead is not None:
            return ahead
        # Named on the SUCCESS too: acknowledging work stays behind is not the same
        # as forgetting it did — and how MUCH stayed behind is the same honest
        # count the refusals state, not the length of the listing.
        left_behind = list(inspection.get("dirty_files") or [])[:_DIRTY_FILES_SENT]
        left_behind_total = _dirty_total(inspection)

        before = _git(repo_dir, "rev-parse", "HEAD")
        head_before = (before.stdout or "").strip() if before.returncode == 0 else ""
        merge = _git(
            repo_dir,
            "-c", "user.name=Ouroboros", "-c", "user.email=ouroboros@local",
            "merge", "--no-ff", "--no-edit", branch,
        )
        if merge.returncode != 0:
            conflicted = _git(repo_dir, "diff", "--name-only", "--diff-filter=U")
            paths = [p for p in (conflicted.stdout or "").splitlines() if p.strip()]
            # STOP, and leave the folder exactly as it was. The thread keeps its
            # branch and every commit in it; nothing is discarded by aborting — but
            # "the folder was left as it was" is a CLAIM about a git command that can
            # fail, so it is only made once that command has been checked and the
            # mid-merge state is CONFIRMED gone. An unchecked abort turned a conflict
            # into an assertion the owner had no way to test, while their folder sat
            # with MERGE_HEAD, `UU` entries and conflict markers in the files.
            #
            # Only aborted when a merge actually started: a merge that git refused
            # outright ("not something we can merge", unrelated histories) leaves
            # nothing in progress, and `merge --abort` failing there means the folder
            # is FINE, not stopped part-way.
            if _in_merge(repo_dir):
                abort = _git(repo_dir, "merge", "--abort")
                if abort.returncode != 0 or _in_merge(repo_dir):
                    return _refused(
                        REASON_MERGE_ABORT_FAILED,
                        "The merge hit a conflict AND could not be undone, so the "
                        "project folder is stopped part-way through it: git still has "
                        "the merge in progress and the overlapping files hold conflict "
                        "markers. Nothing is lost — the thread keeps its branch and "
                        "every commit — but the folder needs you before anything else "
                        "can run in it: in that folder, `git merge --abort` goes back, "
                        "or resolve the files and commit to go forward.",
                        project_id=pid, thread_id=tid, branch=branch,
                        conflicts=paths[:200],
                        abort_detail=_detail(abort),
                        folder_left_mid_merge=True,
                        working_dir=str(repo_dir),
                    )
            if paths:
                return _refused(
                    REASON_MERGE_CONFLICT,
                    "These files changed on both sides, so the merge was stopped and "
                    "the folder left as it was. The thread keeps its branch and all "
                    "its commits — resolve the overlap and merge again.",
                    project_id=pid, thread_id=tid, branch=branch, conflicts=paths[:200],
                )
            return _refused(
                REASON_MERGE_FAILED, _detail(merge),
                project_id=pid, thread_id=tid, branch=branch,
            )
        after = _git(repo_dir, "rev-parse", "HEAD")
        head_after = (after.stdout or "").strip() if after.returncode == 0 else ""
        return {
            "ok": True,
            "project_id": pid,
            "thread_id": tid,
            "branch": branch,
            "merged": head_after != head_before,
            "head_before": head_before,
            "head_after": head_after,
            # A10: merging never removes the checkout. The owner removes it, or not.
            "worktree_kept": True,
            # What the merge did NOT bring, named on the success. Only ever non-empty
            # when the owner acknowledged it, and said out loud there rather than left
            # for them to rediscover in a folder they have stopped looking at.
            "checkout_left_behind": left_behind,
            #: The true size of that set — `checkout_left_behind` is the listing.
            "dirty_files_total": left_behind_total,
            "location": thread_location(data_root, pid, tid),
        }


__all__ = [
    "BASE_SNAPSHOT",
    "QUEUE_NOTICE",
    "QUEUE_NOTICE_OWN_CHECKOUT",
    "REASON_ALREADY_BRANCHED",
    "REASON_BRANCH_FAILED",
    "REASON_CHECKOUT_MISSING",
    "REASON_FOLDER_MISSING",
    "REASON_FOLDER_UNUSABLE",
    "REASON_GIT_INIT_REQUIRED",
    "REASON_CHECKOUT_DIRTY",
    "REASON_CHECKOUT_HEAD_OFF_BRANCH",
    "REASON_LOCAL_TREE_DIRTY",
    "REASON_MERGE_ABORT_FAILED",
    "REASON_MERGE_CONFLICT",
    "REASON_MERGE_FAILED",
    "REASON_NOT_BRANCHED",
    "REASON_NO_FOLDER",
    "REASON_PROJECT_BUSY",
    "REASON_PROJECT_HEAD_DETACHED",
    "REASON_SNAPSHOT_FAILED",
    "REASON_THREAD_NOT_LIVE",
    "REASON_UNKNOWN_BASE",
    "REASON_UNKNOWN_PROJECT",
    "REASON_UNKNOWN_THREAD",
    "branch_off_bases",
    "branch_off_thread",
    "merge_back_thread",
    "project_is_busy",
    "queue_notice",
    "resolve_project_repo",
    "thread_location",
]
