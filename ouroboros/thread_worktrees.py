"""Durable git worktrees owned by project THREADS.

Deliberately SEPARATE from ``subagent_worktrees`` even though both wrap the
same git primitive, because every lifecycle rule is inverted:

===================  ==============================  ==========================
                     subagent worktree               thread worktree
===================  ==============================  ==========================
create over a stale  force-removes checkout+branch   REFUSES (an owner's work
checkout                                             is never clobbered)
removal              ``--force``, unconditional      requires INSPECTION; a
                                                     dirty tree or unmerged
                                                     commits must be
                                                     acknowledged explicitly.
                                                     A permitted removal also
                                                     deletes the thread branch,
                                                     which would otherwise block
                                                     re-provisioning forever
startup GC           age sweep past the retention    NONE. A thread's worktree
                     window deletes the checkout     is durable and is only
                                                     removed by an explicit act
===================  ==============================  ==========================

Only the git-op lock and the path-containment guards are reused (imported as
public names from ``subagent_worktrees``); none of its provisioning, removal or
prune behaviour is. The registry lives in its own durable file, so the subagent
orphan sweep — which iterates ITS registry, not the filesystem — can never see
a thread worktree at all.

State: ``data/state/thread_worktrees.json`` via the canonical durable-JSON
pattern, keyed by ``(project_id, thread_id)``.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ouroboros.contracts.schema_versions import with_schema_version
from ouroboros.subagent_worktrees import (
    assert_worktree_root_isolated,
    force_rmtree,
    path_is_within,
    run_git,
    safe_path_component,
    worktree_ops_lock,
)
from ouroboros.utils import (
    atomic_write_json, read_json_dict, truncate_review_artifact, utc_now_iso,
)

log = logging.getLogger(__name__)

_REGISTRY_NAME = "thread_worktrees.json"
_SCHEMA_VERSION = 1
_BRANCH_PREFIX = "thread/"
_LOCK = threading.RLock()

# A PATH, not a numeric knob, so the SSOT gate that governs timeouts does not
# apply — it is env-overridable exactly so a relocated Ouroboros home still works,
# and the removal guard now validates the root a row was provisioned under against
# this one. T0 justified it with "no owner-facing surface reaches it yet
# (branch-off is a later phase)"; T3 IS that phase, and branch-off, merge-back,
# the checkout diff and the inspected removal all reach it.
_ROOT_ENV = "OUROBOROS_THREAD_WORKTREE_ROOT"
#: Character bound for a surviving checkout's `reason` and for the git text a
#: failed inspection reports. Both go through the `truncate_review_artifact`
#: SSOT, so an overflow carries its omission marker instead of vanishing.
_KEPT_REASON_LIMIT = 200
_GIT_TEXT_LIMIT = 500


def thread_worktree_root() -> Path:
    """Durable root for thread checkouts — outside ``repo/`` and ``data/``."""
    raw = str(os.environ.get(_ROOT_ENV, "") or "").strip()
    root = raw or os.path.expanduser(os.path.join("~", "Ouroboros", "thread_worktrees"))
    return Path(root).expanduser().resolve()


def _git_timeout_sec() -> float:
    """The ceiling for every git call in this module, from the ONE settings SSOT.

    Read through ``config`` rather than pinned as a module-local number, and it is
    the SAME knob ``thread_branching`` uses (``OUROBOROS_THREAD_GIT_TIMEOUT_SEC``,
    120s, clamped 5-300): branch-off, merge-back, the inspection and the removal
    are arms of one owner gesture on one repository, so a wedged git must expire at
    the same point on all of them. Deliberately not the task-diff READ key, whose
    30s ceiling silently narrowed this path once already (H-ter).
    """
    from ouroboros.config import get_thread_git_timeout_sec

    return get_thread_git_timeout_sec()


def _git(root: Any, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    """One BOUNDED git call — the only way this module invokes git.

    ``subagent_worktrees.run_git`` passes no timeout, and that is correct for ITS
    callers: background provisioning and the startup orphan sweep, which nothing
    waits on. This module's calls are reached from six OWNER-FACING routes that did
    not exist before T3 — ``GET``/``POST`` on a thread's worktree, branch-off,
    merge-back, thread delete and ``DELETE /api/projects/{id}`` — where a git that
    never returns holds the owner's request AND a thread-pool thread forever, with
    the registry ``_LOCK`` released but the git-op lock still held, so every other
    worktree gesture on that repo wedges behind it too.

    An expiry is a typed, owner-legible OUTCOME, never a traceback:

    * ``check=False`` (every read, and both deletions) comes back as an ordinary
      ``CompletedProcess`` with ``returncode=124`` and a sentence naming the
      ceiling, so :func:`inspect_thread_worktree` reports it as ``error`` — which
      already counts as "cannot tell", i.e. UNSAFE — and
      :func:`remove_thread_worktree` reports ``removal_failed`` and keeps its row.
    * ``check=True`` (provisioning, which must refuse rather than continue) raises
      ``ValueError`` with the same sentence, the exact channel
      ``thread_branching.branch_off_thread`` already turns into a typed
      ``branch_failed`` refusal carrying the message.

    ``TimeoutExpired`` is the only exception converted. A ``CalledProcessError``
    from ``check=True`` still propagates unchanged, because provisioning's refusals
    are built on it.
    """
    limit = _git_timeout_sec()
    try:
        return run_git(root, *args, check=check, timeout=limit)
    except subprocess.TimeoutExpired:
        detail = (
            f"git {' '.join(str(a) for a in args[:4])} in {root} did not finish within "
            f"{limit:g}s (OUROBOROS_THREAD_GIT_TIMEOUT_SEC)"
        )
        log.warning("Thread worktree git call timed out: %s", detail)
        if check:
            raise ValueError(detail) from None
        return subprocess.CompletedProcess(["git", *args], 124, stdout="", stderr=detail)


@dataclass(frozen=True)
class ThreadWorktree:
    project_id: str
    thread_id: int
    path: str
    branch: str
    base_sha: str
    repo_dir: str
    created_at: float
    created_at_iso: str = ""
    #: The root this checkout was PROVISIONED under, recorded so removal can
    #: validate containment against the same boundary that admitted it (T0R2-9).
    #: Validating against the root resolved at removal time made every existing
    #: row `path_outside_root` — permanently unremovable through the API — the
    #: moment `OUROBOROS_THREAD_WORKTREE_ROOT` changed.
    worktree_root: str = ""


def _registry_path(data_dir: Any) -> Path:
    return Path(data_dir) / "state" / _REGISTRY_NAME


def _load(data_dir: Any) -> List[Dict[str, Any]]:
    data = read_json_dict(_registry_path(data_dir))
    rows = data.get("worktrees") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict) and row.get("path")]


def _save(data_dir: Any, rows: List[Dict[str, Any]]) -> None:
    path = _registry_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, with_schema_version({"worktrees": rows}, _SCHEMA_VERSION))


def _key(project_id: Any, thread_id: Any) -> tuple:
    try:
        return str(project_id or "").strip(), int(thread_id)
    except (TypeError, ValueError):
        return str(project_id or "").strip(), -1


def _matches(row: Dict[str, Any], key: tuple) -> bool:
    try:
        return (str(row.get("project_id") or ""), int(row.get("thread_id"))) == key
    except (TypeError, ValueError):
        return False


def list_thread_worktrees(data_dir: Any) -> List[Dict[str, Any]]:
    """Every registered thread worktree (never age-filtered)."""
    with _LOCK:
        return [dict(row) for row in _load(data_dir)]


def get_thread_worktree(data_dir: Any, project_id: Any, thread_id: Any) -> Optional[Dict[str, Any]]:
    key = _key(project_id, thread_id)
    with _LOCK:
        for row in _load(data_dir):
            if _matches(row, key):
                return dict(row)
    return None


def provision_thread_worktree(
    *,
    repo_dir: Any,
    project_id: str,
    thread_id: Any,
    base_ref: str = "",
    data_dir: Any,
    worktree_root: Optional[Any] = None,
) -> ThreadWorktree:
    """Create a durable worktree for one thread, or REFUSE.

    Unlike the subagent path this never force-removes a stale checkout or
    branch: an existing registration, an existing directory or an existing
    branch is the owner's work, and clobbering it silently is the exact failure
    this registry exists to prevent. Re-provisioning is therefore an error, not
    a reset — remove the worktree explicitly first.
    """
    key = _key(project_id, thread_id)
    if not key[0] or key[1] < 0:
        raise ValueError(f"unusable thread key: {project_id!r}#{thread_id!r}")
    repo = Path(repo_dir).resolve()
    root = Path(worktree_root).expanduser().resolve() if worktree_root else thread_worktree_root()
    assert_worktree_root_isolated(root, repo, Path(data_dir))
    name = safe_path_component(f"{key[0]}__{key[1]}")
    wt_path = (root / name).resolve()
    branch = f"{_BRANCH_PREFIX}{name}"
    # The ops lock is keyed on the REPO, not on this registry's worktree root:
    # `git worktree add` rewrites <repo>/.git/worktrees, which the SUBAGENT
    # registry mutates too. Two roots meant two lockfiles over one .git.
    with _LOCK, worktree_ops_lock(repo, mkdir_root=root):
        rows = _load(data_dir)
        if any(_matches(row, key) for row in rows):
            raise ValueError(
                f"thread {key[0]}#{key[1]} already has a worktree — remove it explicitly first"
            )
        if wt_path.exists():
            raise ValueError(f"refusing to reuse an existing path: {wt_path}")
        existing_branch = _git(repo, "rev-parse", "--verify", branch, check=False)
        if existing_branch.returncode == 0:
            raise ValueError(
                f"branch {branch!r} already exists — delete it deliberately before branching off again"
            )
        if base_ref:
            _git(repo, "rev-parse", "--verify", f"{base_ref}^{{commit}}")
            base_sha = _git(repo, "rev-parse", base_ref).stdout.strip()
        else:
            base_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
        wt_path.parent.mkdir(parents=True, exist_ok=True)
        # No --force: git must refuse rather than take over a foreign checkout.
        _git(repo, "worktree", "add", "-b", branch, str(wt_path), base_sha)
        handle = ThreadWorktree(
            project_id=key[0],
            thread_id=key[1],
            path=str(wt_path),
            branch=branch,
            base_sha=base_sha,
            repo_dir=str(repo),
            created_at=time.time(),
            created_at_iso=utc_now_iso(),
            worktree_root=str(root),
        )
        _save(data_dir, [*rows, asdict(handle)])
        log.info("Thread worktree provisioned: %s#%s at %s", key[0], key[1], wt_path)
        return handle


def _project_head(row: Dict[str, Any]) -> str:
    """The PROJECT folder's current HEAD sha, or "" when it cannot be read."""
    repo = Path(str(row.get("repo_dir") or ""))
    try:
        if not repo.is_dir():
            return ""
        head = _git(repo, "rev-parse", "HEAD", check=False)
    except Exception:
        return ""
    return (head.stdout or "").strip() if head.returncode == 0 else ""


#: How many dirty entries an inspection LISTS. The bound stays — an unbounded
#: list on an owner-facing envelope is its own problem — but it is a display
#: bound only: ``dirty_files_total`` carries the honest size of the set beside
#: it, and every sentence that states a number states that one.
_DIRTY_FILES_SHOWN = 200


def inspect_thread_worktree(row: Dict[str, Any]) -> Dict[str, Any]:
    """What removing this worktree would DESTROY — the evidence a removal needs.

    Returns ``{exists, dirty, dirty_files, dirty_files_total, unmerged_commits,
    unmerged_against, error}``. Never raises: an unreadable checkout reports
    ``error`` and is treated as unsafe (``dirty``), because "cannot tell" must
    never read as "nothing to lose".

    ``dirty_files`` is BOUNDED at :data:`_DIRTY_FILES_SHOWN` entries and
    ``dirty_files_total`` is the true count, always present beside it. The list
    is for showing; the number is what every owner-facing sentence must state.
    A bare ``files[:200]`` with no total is exactly the silent truncation
    DEVELOPMENT.md forbids, and it did not stay harmless: a long-running agent
    leaves ordinary modified TRACKED files, and 800 of them made the removal
    refusal — the sentence immediately before an irreversible delete — tell the
    owner "200 uncommitted file changes". The safety gate held (the checkout is
    still refused, the acknowledgement is still required) but the magnitude the
    owner decides on was wrong by a factor of four. A wholly-ignored DIRECTORY
    does not do it, because git collapses ``node_modules/`` to one entry; it
    takes plain modified files, which is the ordinary case.

    IGNORED files are counted as dirty (``--ignored=matching``, the ``!!``
    entries). ``git status --porcelain`` alone hides exactly the files a thread's
    checkout is most likely to be the only copy of: a ``.env`` written into that
    folder, a ``local.db``, a ``build/`` an agent produced. The checkout read
    ``dirty: false`` and one-click removal force-deleted them with no prompt —
    the same ``.env`` the snapshot works hard to keep OUT of history is the one
    this deleted from disk. They are not "changes", but the question this function
    answers is what removal would DESTROY, and they are destroyed. Counted here,
    they ride the acknowledgement path that already exists rather than needing a
    second one.

    ``unmerged_commits`` is counted against the PROJECT's current HEAD, not
    against the frozen ``base_sha`` this checkout branched from. The question A10
    asks is "what would the project folder never receive", and the answer moves
    every time the project's HEAD does: counting against the branch point meant a
    worktree whose work had ALREADY been merged back still reported every one of
    those commits as unmerged, so the owner was asked to acknowledge destroying
    work that was already safe in their folder. Evidence that cries wolf is worse
    than no evidence, because the owner learns to click through it.

    The base is the FALLBACK, and deliberately the conservative direction: when
    the project's HEAD cannot be read, counting from the branch point can only
    over-report, which refuses a removal rather than permitting one.

    Counted from BOTH tips — the checkout's HEAD and the thread's own branch —
    because those come apart. A checkout standing on a detached HEAD, or moved
    onto some other branch, has a ``thread/<name>`` branch that still holds every
    commit made in it; asking only where HEAD points reported ZERO and the owner
    was told the removal "deletes only the folder". Nothing was actually lost —
    ``git branch -d`` refuses an unmerged branch, so the commits survived — but
    A10's evidence has to be true when it is READ, not merely harmless.
    """
    out: Dict[str, Any] = {
        "exists": False, "dirty": False, "dirty_files": [], "dirty_files_total": 0,
        "unmerged_commits": 0, "unmerged_against": "", "error": "",
    }
    wt_path = Path(str(row.get("path") or ""))
    if not wt_path.is_dir():
        # A registered checkout that is not on disk is "cannot tell", not
        # "nothing to lose" — an unmounted volume, a folder moved out from under
        # the registry, a `git worktree remove` run by hand. Answering
        # `{exists: False, dirty: False, unmerged_commits: 0}` was the exact shape
        # this docstring says must never happen: the removal prompt read it as a
        # clean checkout and offered one-click removal of something whose contents
        # nobody could see. It rides the acknowledgement path instead.
        out["error"] = f"the checkout is not on disk: {wt_path}"
        out["dirty"] = True
        return out
    out["exists"] = True
    try:
        # `core.quotepath=off` so a non-ASCII path arrives as itself. Without it
        # this listed the C-quoted spelling while `merge_back_thread`'s own status
        # call — which pins it — listed the real one, and the two surfaces
        # disagreed about the same file.
        status = _git(
            wt_path, "-c", "core.quotepath=off", "status", "--porcelain",
            "--ignored=matching", check=False,
        )
        if status.returncode != 0:
            # Not a checkout any more (or git refused): unsafe by construction.
            out["error"] = truncate_review_artifact(
                (status.stderr or "git status failed").strip(), limit=_GIT_TEXT_LIMIT,
            )
            out["dirty"] = True
            return out
        files = [line for line in status.stdout.splitlines() if line.strip()]
        out["dirty"] = bool(files)
        # The TRUE count first, then the bounded listing — never the length of
        # the slice standing in for the size of the set.
        out["dirty_files_total"] = len(files)
        out["dirty_files"] = files[:_DIRTY_FILES_SHOWN]
        reference = _project_head(row) or str(row.get("base_sha") or "")
        if reference:
            out["unmerged_against"] = reference
            tips = ["HEAD"]
            branch = str(row.get("branch") or "").strip()
            if branch and _git(
                wt_path, "rev-parse", "--verify", "-q", branch, check=False,
            ).returncode == 0:
                tips.append(branch)
            ahead = _git(
                wt_path, "rev-list", "--count", *tips, "--not", reference, check=False,
            )
            if ahead.returncode != 0:
                out["error"] = truncate_review_artifact(
                    (ahead.stderr or "git rev-list failed").strip(), limit=_GIT_TEXT_LIMIT,
                )
                out["dirty"] = True
                return out
            out["unmerged_commits"] = int((ahead.stdout or "0").strip() or 0)
    except Exception as exc:
        out["error"] = truncate_review_artifact(str(exc), limit=_GIT_TEXT_LIMIT)
        out["dirty"] = True
    return out


#: ``git status --porcelain`` codes for content git is not tracking. ``!!`` only
#: appears because :func:`inspect_thread_worktree` asks for ``--ignored=matching``.
_UNTRACKED_CODE = "??"
_IGNORED_CODE = "!!"


def checkout_work_at_risk(inspection: Dict[str, Any]) -> Dict[str, Any]:
    """Split what a checkout holds into work that CANNOT be rebuilt, and the rest.

    Returns ``{at_risk, unmerged_commits, tracked_files, untracked_files,
    ignored_files, omitted_files, unreadable}``. A pure read over an existing
    inspection — it asks nothing of git and never touches the disk.

    The three file lists are split out of the inspection's BOUNDED listing, so
    their lengths are counts of what was shown, not of what is there.
    ``omitted_files`` is the difference (``dirty_files_total`` minus the entries
    actually listed) and exists so the deletion copy — which states each of those
    lengths — can disclose that it is not stating the whole set. Without it the
    delete refusal would have gone on saying "changes to 200 files git is
    tracking" about 800 of them, in the same release that taught the removal
    refusal to say 800.

    ``inspect_thread_worktree`` deliberately counts an ignored ``node_modules/``
    as dirt, because REMOVING the checkout destroys it and A10's prompt must say
    so. But "would be destroyed" and "cannot be rebuilt" are different questions,
    and only the second one may BLOCK an owner gesture aimed at something else.
    Thread DELETION asks the second: a checkout whose only dirt is a build
    directory or a stray log is not work at risk, and refusing the delete over it
    made "delete the thread and its folder" a three-step detour through merge-back
    and an acknowledged removal — friction the owner explicitly did not ask for.

    At risk means, exactly:

    * ``unmerged_commits`` — commits the project folder never received. The
      checkout's branch is their last copy.
    * ``tracked_files`` — modifications to files git is TRACKING. Their previous
      contents are in history; these edits are nowhere else.
    * ``unreadable`` — the inspection could not be taken. "Cannot tell" must never
      read as "nothing to lose", so it counts as at risk on its own.

    Untracked and ignored files are neither destroyed silently nor treated as
    unlosable: they are NAMED and go through the same acknowledgement the removal
    route uses. Acknowledging is a step; being refused is a wall.
    """
    files = [str(line) for line in (inspection.get("dirty_files") or []) if str(line).strip()]
    tracked: List[str] = []
    untracked: List[str] = []
    ignored: List[str] = []
    for line in files:
        code = line[:2]
        if code == _IGNORED_CODE:
            ignored.append(line)
        elif code == _UNTRACKED_CODE:
            untracked.append(line)
        else:
            tracked.append(line)
    unreadable = str(inspection.get("error") or "").strip()
    commits = int(inspection.get("unmerged_commits") or 0)
    # An inspection that predates `dirty_files_total` (or one hand-built by a
    # caller) reads as "the listing IS the set" — the old behaviour, never a
    # negative omission.
    total = int(inspection.get("dirty_files_total") or 0)
    return {
        "at_risk": bool(commits > 0 or tracked or unreadable),
        "unmerged_commits": commits,
        "tracked_files": tracked,
        "untracked_files": untracked,
        "ignored_files": ignored,
        "omitted_files": max(0, total - len(files)),
        "unreadable": unreadable,
    }


def remove_thread_worktree(
    *,
    data_dir: Any,
    project_id: str,
    thread_id: Any,
    acknowledge_unmerged: bool = False,
    worktree_root: Optional[Any] = None,
    busy: Optional[bool] = None,
) -> Dict[str, Any]:
    """Remove a thread worktree AFTER inspecting what that would destroy.

    Returns ``{removed, reason, inspection}``. A dirty tree or commits the base
    never received refuse the removal unless ``acknowledge_unmerged`` is passed
    — the caller must have SHOWN the owner the inspection first. There is no
    silent path and no timer that reaches this function.

    An ACTIVE project refuses with ``project_busy``, the same reason and the same
    409 shape merge-back uses, because this deletes a folder something may be
    writing in. ``project_lease.running_project_ids``,
    ``thread_branching.project_is_busy`` and ARCHITECTURE all described that
    precondition, and merge-back was its only caller: a task running in the
    checkout made merge-back refuse correctly and removal answer ``removed: True``
    while deleting the folder under the live worker. ``busy`` overrides the live
    query for tests and for callers that already hold the answer.

    That query is a bare READ, so the inspection and both deletions run inside
    ``project_lease.reserved_folder_lane`` on the CHECKOUT — the same reservation
    merge-back holds over the project folder. Without it the scheduler could admit
    a task into the checkout in the gap between "nothing is running here" and the
    ``rmtree``, which is exactly the two-writer state the lane exists to prevent.
    The reservation reorders NOTHING: every refusal still happens before anything
    is destroyed, so a refused removal leaves the checkout exactly as it was.

    Containment is checked against the root the row was PROVISIONED under
    (T0R2-9), not against whatever this process resolves today. Resolving it at
    removal time meant relocating ``OUROBOROS_THREAD_WORKTREE_ROOT`` — or simply
    passing a different ``worktree_root`` — turned every existing row into
    ``path_outside_root``: unremovable through the API forever, with the mirror
    hazard that a moved root would ADMIT a path it should never have admitted.
    A pre-T3 row carries no provisioning root; it falls back to the resolved one,
    which is exactly the behaviour it was written under.

    A checkout that SURVIVES the removal attempt reports ``reason="removal_failed"``
    and KEEPS its registry row. Both deletions run best-effort (``check=False`` and
    a swallowing rmtree), so a checkout held by a git lock, a read-only parent or a
    busy file outlives them; reporting that as removed and dropping the row would
    leave an orphan holding the branch that the registry can no longer see,
    re-provision or remove.

    A CLEAN removal also deletes the ``thread/<name>`` branch, and that is a
    decision rather than a tidy-up. ``provision_thread_worktree`` refuses to reuse
    an existing branch — deliberately, so an owner's work is never clobbered — so
    leaving the branch behind made branch → merge → remove a ONE-SHOT round trip:
    the second branch-off of the same thread failed with "branch already exists"
    and the owner had no surface that could delete it. The alternative considered
    was suffixing the branch name with a timestamp, which was rejected: it makes
    every thread's branch name unpredictable to the owner reading `git branch`,
    and it accumulates dead branches in their repository forever.

    Deleting is safe here precisely because "clean" is checked twice by two
    independent judges: this module's inspection (no dirty tree, no commits the
    project's HEAD lacks) AND ``git branch -d``, which refuses on its own account
    if the branch holds anything unmerged. A removal the owner had to
    ACKNOWLEDGE keeps its branch — those commits are the last copy of that work,
    and the acknowledgement was about the checkout, not about the history.
    """
    key = _key(project_id, thread_id)
    root = Path(worktree_root).expanduser().resolve() if worktree_root else thread_worktree_root()
    with _LOCK:
        seen = next((row for row in _load(data_dir) if _matches(row, key)), None)
    if seen is None:
        return {"removed": False, "reason": "unknown", "inspection": {}}
    # The git-op lock is keyed on the REPO, exactly as provisioning keys it, and not
    # on the worktree root: `git worktree remove`/`prune` rewrite the SAME
    # `<repo>/.git/worktrees` metadata that provisioning and the SUBAGENT registry
    # mutate. Keying it on the root gave those owners two different lockfiles over
    # one `.git`, so removal could rewrite that metadata concurrently with a
    # provisioning it was supposed to serialize against (`_ops_lock_path`).
    with worktree_ops_lock(str(seen.get("repo_dir") or root), mkdir_root=root):
        # `_LOCK` covers the registry READ and the final SAVE, nothing between.
        # Holding it across two git calls, `force_rmtree`, a prune and a
        # `git branch -d` blocked every `thread_location`/`get_thread_worktree`
        # read for the duration — and back when this module called the unbounded
        # `run_git` directly that duration had no ceiling at all, so one wedged git
        # call froze the sidebar. Every call here now goes through the bounded
        # module-local `_git`, and the git-op lock still serializes the whole
        # operation per root, which is what actually needs to be exclusive.
        with _LOCK:
            match = next((row for row in _load(data_dir) if _matches(row, key)), None)
        if match is None:
            return {"removed": False, "reason": "unknown", "inspection": {}}
        # M5's reservation, applied to the OTHER destructive gesture. Merge-back
        # HOLDS the folder it rewrites for the duration; this DELETES one and held
        # nothing at all — `_project_is_busy` is a bare READ, so a task the
        # scheduler was already considering could be admitted into the checkout in
        # the gap between the check and the `rmtree`, and a message arriving after
        # the check could queue work into the folder being deleted. Reserving the
        # CHECKOUT's lane makes the scheduler refuse admission for the WHOLE
        # window, exactly as merge-back does for the project folder, and
        # `reserved_folder_lanes(include_own=False)` is why this holder is never
        # refused by its own claim.
        #
        # Deliberately NOT a routing fence, and deliberately not reordered: the
        # inspection and every refusal stay BEFORE anything is fenced or removed
        # (86aaf2b1), so a REFUSED delete never destroys the checkout first. A
        # reservation closes the check-then-act gap without touching that order.
        from ouroboros.project_lease import reserved_folder_lane

        with reserved_folder_lane(str(match.get("path") or "")):
            # Removal deletes a folder a task may be WRITING in. `project_is_busy`,
            # `running_project_ids` and ARCHITECTURE all said this was guarded; only
            # merge-back actually asked. Reproduced: a running task in the checkout,
            # merge-back correctly refuses `project_busy`, and removal answered
            # `removed: True` and deleted the folder under the live worker.
            if _project_is_busy(project_id, match) if busy is None else bool(busy):
                return {"removed": False, "reason": "project_busy", "inspection": {}}
            inspection = inspect_thread_worktree(match)
            unsafe = bool(inspection["dirty"]) or int(inspection["unmerged_commits"]) > 0
            if unsafe and not acknowledge_unmerged:
                return {"removed": False, "reason": "unmerged_work", "inspection": inspection}
            wt_path = Path(str(match.get("path") or ""))
            guard_root = _trusted_guard_root(match, root, data_dir)
            if (
                not str(wt_path).strip()
                or not path_is_within(wt_path, guard_root)
                or wt_path.name != safe_path_component(f"{key[0]}__{key[1]}")
            ):
                # A malformed registry row must never delete an arbitrary path — and
                # that was exactly what it could do, because the boundary it was
                # checked against came from the SAME untrusted row (T0R2-9 moved the
                # root onto the row and the guard moved with it). Two independent
                # facts are required now: a root this process would itself accept, and
                # the path this thread's checkout is DERIVED to have.
                return {"removed": False, "reason": "path_outside_root", "inspection": inspection}
            repo = Path(str(match.get("repo_dir") or "."))
            branch = str(match.get("branch") or "").strip()
            _git(repo, "worktree", "remove", "--force", str(wt_path), check=False)
            if wt_path.exists():
                force_rmtree(wt_path)
            _git(repo, "worktree", "prune", check=False)
            if wt_path.exists():
                # Both deletions above run best-effort (`check=False` / a swallowing
                # rmtree), so a checkout held by a git lock, a read-only parent or a
                # busy file SURVIVES them. Reporting that as removed and dropping the
                # row would leave an orphan holding the branch that the registry can no
                # longer see, re-provision or remove. Say so and KEEP the row — and
                # leave the branch alone, the surviving checkout is still on it.
                log.warning(
                    "Thread worktree %s#%s could not be removed — %s still exists; "
                    "registry row retained",
                    key[0], key[1], wt_path,
                )
                return {
                    "removed": False,
                    "reason": "removal_failed",
                    "inspection": inspection,
                    "branch": branch,
                    "branch_removed": False,
                    "branch_kept_reason": "the checkout survived removal, so its branch still points at it",
                }
            # The branch is kept for COMMITS, never merely for dirt. An acknowledged
            # removal whose only dirt was an ignored `node_modules/` has nothing on
            # that branch the repository would not still have, and keeping it left a
            # `thread/<name>` behind that the next branch-off refuses on — and that
            # nothing can reach at all once the thread it belonged to is tombstoned.
            # `git branch -d` is still the second judge and still refuses on its own
            # account, so this only ever ASKS; it never forces.
            branch_removed, branch_kept_reason = _drop_clean_branch(
                repo, branch, bool(inspection["error"]) or int(inspection["unmerged_commits"]) > 0,
            )
            with _LOCK:
                _save(data_dir, [row for row in _load(data_dir) if not _matches(row, key)])
            log.info("Thread worktree removed: %s#%s (%s)", key[0], key[1], wt_path)
            return {
                "removed": True,
                "reason": "",
                "inspection": inspection,
                "branch": branch,
                "branch_removed": branch_removed,
                "branch_kept_reason": branch_kept_reason,
            }


def project_thread_worktrees(data_dir: Any, project_id: Any) -> List[Dict[str, Any]]:
    """Every registered checkout belonging to ONE project.

    The project-delete precondition needs this and had no way to ask: the registry
    is keyed by ``(project_id, thread_id)`` but only ``list_thread_worktrees``
    (whole file) and ``get_thread_worktree`` (one thread) were exposed, so
    ``api_project_delete`` never looked and tombstoned the project with N
    checkouts and N ``thread/*`` branches still on disk (I1).
    """
    pid = str(project_id or "").strip()
    if not pid:
        return []
    return [row for row in list_thread_worktrees(data_dir) if str(row.get("project_id") or "") == pid]


def project_checkouts_at_risk(data_dir: Any, project_id: Any) -> List[Dict[str, Any]]:
    """Which of a project's checkouts hold work that cannot be rebuilt.

    Returns one entry per checkout that WOULD be destroyed by deleting the
    project, each ``{thread_id, path, branch, inspection, risk}``. The judge is
    ``checkout_work_at_risk`` — the SAME narrower fact thread deletion asks, not
    removal's broader "would be destroyed" — so deleting a project is refused for
    exactly the reasons deleting one of its threads is refused, and a stray
    ``node_modules/`` never blocks it.
    """
    out: List[Dict[str, Any]] = []
    for row in project_thread_worktrees(data_dir, project_id):
        inspection = inspect_thread_worktree(row)
        risk = checkout_work_at_risk(inspection)
        if risk["at_risk"]:
            out.append({
                "thread_id": int(row.get("thread_id") or 0),
                "path": str(row.get("path") or ""),
                "branch": str(row.get("branch") or ""),
                "inspection": inspection,
                "risk": risk,
            })
    return out


def remove_project_thread_worktrees(data_dir: Any, project_id: Any) -> Dict[str, Any]:
    """Remove every checkout of a project, reporting each outcome.

    Returns ``{removed: [...], kept: [{thread_id, path, branch, reason}],
    branches: [...]}``. Never raises: a checkout that survives is REPORTED,
    because the project row is on its way to a tombstone and a swallowed failure
    would be the orphan this function exists to prevent — ``path`` and ``branch``
    ride each ``kept`` entry so the disclosure can say WHERE it was left.

    ``acknowledge_unmerged`` is decided PER ROW, by asking the same judge the
    route asked, and that is the fix for a real destruction path. It used to be a
    hardcoded ``True``, justified by "the CALLER has already refused on anything at
    risk" — which is true of ``api_project_delete``'s pre-fence inspection and NOT
    true of the moment this runs. Reproduced: a clean checkout passes the
    pre-check; the route's own removal correctly refuses ``project_busy`` because a
    task is still writing there; the task then COMMITS work and edits a tracked
    file; the post-quiescence sweep (``supervisor.task_lifecycle
    ._sweep_project_checkouts``) calls this, and the hardcoded acknowledgement
    destroyed both with no re-inspection and no fresh consent — exactly the
    "acknowledge on the owner's behalf" that A10/D4 forbid.

    So each row is re-inspected here and ``acknowledge_unmerged`` is
    ``not risk["at_risk"]``: rebuildable dirt the owner already confirmed still
    goes, while a checkout that became at-risk AFTER the owner looked comes back
    ``unmerged_work`` and lands in ``kept``. The consequence is disclosed rather
    than hidden — see the caller, which records the survivors on the tombstoned row
    and tells the owner where they are.

    DISCLOSED residual, not a silent one: the judge still runs OUTSIDE
    :func:`remove_thread_worktree`'s git-op lock, so work appearing between this
    inspection and the one taken inside it would still ride the acknowledgement.
    That window is now the same one ``api_thread_delete`` and ``api_project_delete``
    already have — an ops-lock acquisition and one ``git status`` — instead of the
    entire fence-and-quiesce span this used to reason across, and closing it
    completely means moving the risk judge inside the lock, which changes the
    removal's own signature and is deliberately not done here.
    """
    removed: List[int] = []
    kept: List[Dict[str, Any]] = []
    branches: List[str] = []
    for row in project_thread_worktrees(data_dir, project_id):
        tid = int(row.get("thread_id") or 0)
        survivor = {
            "thread_id": tid,
            "path": str(row.get("path") or ""),
            "branch": str(row.get("branch") or ""),
        }
        try:
            # The SAME judge `api_project_delete` used before fencing, asked again
            # NOW: consent given for a clean checkout is not consent for whatever
            # a still-running task put in it afterwards.
            risk = checkout_work_at_risk(inspect_thread_worktree(row))
            outcome = remove_thread_worktree(
                data_dir=data_dir, project_id=str(project_id or ""), thread_id=tid,
                acknowledge_unmerged=not risk["at_risk"],
            )
        except Exception as exc:  # noqa: BLE001 — an orphan must never be silent
            log.warning("Project deletion could not remove checkout %s#%s", project_id, tid, exc_info=True)
            # Bounded, not silently cut: this reason is the only account of WHY a
            # folder the tombstone can no longer reach survived, so an overflow
            # arrives with the SSOT omission marker rather than losing its tail in
            # silence (BIBLE P1, DEVELOPMENT.md "No silent truncation").
            kept.append({**survivor, "reason": truncate_review_artifact(
                f"{type(exc).__name__}: {exc}", limit=_KEPT_REASON_LIMIT,
            )})
            continue
        if outcome.get("removed"):
            removed.append(tid)
            if outcome.get("branch_removed") and outcome.get("branch"):
                branches.append(str(outcome["branch"]))
        else:
            kept.append({**survivor, "reason": str(outcome.get("reason") or "removal_failed")})
    return {"removed": removed, "kept": kept, "branches": branches}


def _project_is_busy(project_id: Any, row: Dict[str, Any]) -> bool:
    """Is anything alive in this project, or in the CHECKOUT? — merge-back's judge.

    Imported lazily because ``thread_branching`` imports THIS module; the answer
    has to be the same one merge-back gets, or the two owner gestures would
    disagree about whether the folder is safe to touch. The folder asked about is
    the CHECKOUT — that is the one this deletes — and the project-wide half of the
    query already covers a TASK running anywhere else in the project.

    The PROJECT folder is asked about as well, because a non-task holder is not a
    task and the project-wide half cannot see it (I5). A merge-back reserves the
    project folder while it rewrites it, and this removal rewrites the same
    repository's ``.git/worktrees`` and deletes the very ``thread/<name>`` branch
    that merge is reading — so "nothing is happening in the project" has to
    include the gesture that is happening in it.
    """
    from ouroboros.thread_branching import project_is_busy

    pid = str(project_id or "")
    if project_is_busy(pid, str(row.get("path") or "")):
        return True
    repo_dir = str(row.get("repo_dir") or "")
    return bool(repo_dir) and project_is_busy(pid, repo_dir)


def _trusted_guard_root(row: Dict[str, Any], fallback: Path, data_dir: Any) -> Path:
    """The boundary a removal is allowed to delete inside.

    The root recorded at provisioning is preferred (T0R2-9: validating against
    whatever the process resolves TODAY stranded every existing row as
    ``path_outside_root`` the moment the configured root moved), but it is a value
    on an untrusted row, so it is re-checked with the SAME guard provisioning used
    before it is believed. A stored root that would never have been admitted falls
    back to this process's own root, where a bogus path simply fails containment.
    """
    from ouroboros.config import REPO_DIR

    stored = str(row.get("worktree_root") or "").strip()
    if not stored:
        return fallback
    try:
        candidate = Path(stored).expanduser().resolve()
        assert_worktree_root_isolated(candidate, Path(REPO_DIR), Path(data_dir))
    except Exception:
        log.warning("Thread worktree row carries an unusable provisioning root: %r", stored)
        return fallback
    return candidate


def _drop_clean_branch(repo: Path, branch: str, holds_commits: bool) -> tuple:
    """Delete a thread branch that has nothing left to lose. ``(removed, why_kept)``.

    ``git branch -d`` — never ``-D``. The safe form is the point: it is git's own
    second opinion on whether the branch holds anything the repository would not
    still have afterwards, and it refusing means the branch stays. A branch that
    stays is disclosed with the reason, never silently.

    ``holds_commits`` is about the HISTORY, not about the working tree: commits
    the project folder never received, or an inspection that could not be taken
    at all. Uncommitted or ignored files are not on the branch, so keeping it
    would protect nothing while leaving a `thread/<name>` the next branch-off
    refuses on.
    """
    if not branch or not branch.startswith(_BRANCH_PREFIX):
        return False, "not a thread branch"
    if holds_commits:
        # The owner acknowledged losing the CHECKOUT. Its commits are a separate
        # thing and this is the last copy of them.
        return False, "the checkout held unmerged work, so its branch keeps the commits"
    try:
        dropped = _git(repo, "branch", "-d", branch, check=False)
    except Exception as exc:
        return False, f"the branch could not be deleted: {type(exc).__name__}: {exc}"
    if dropped.returncode != 0:
        return False, truncate_review_artifact(
            (dropped.stderr or dropped.stdout or "git refused to delete the branch").strip(),
            limit=_GIT_TEXT_LIMIT,
        )
    return True, ""


__all__ = [
    "ThreadWorktree",
    "checkout_work_at_risk",
    "get_thread_worktree",
    "inspect_thread_worktree",
    "list_thread_worktrees",
    "project_checkouts_at_risk",
    "project_thread_worktrees",
    "provision_thread_worktree",
    "remove_project_thread_worktrees",
    "remove_thread_worktree",
    "thread_worktree_root",
]
