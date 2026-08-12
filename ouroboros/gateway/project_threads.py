"""Thread branching, checkout diff and lifecycle routes (Gateway Boundary).

Thin transport only: every decision about branching lives in
:mod:`ouroboros.thread_branching`, every decision about the checkout registry in
:mod:`ouroboros.thread_worktrees`, every diff decision in
:mod:`ouroboros.gateway.task_diff`, and every registry mutation in
:mod:`ouroboros.project_threads_registry`. Nothing here decides anything.

These are OWNER surfaces reached through the gateway and deliberately NOT
LLM-callable tools (R10). Branching off, merging back, removing a checkout,
archiving and deleting are gestures with consequences for the owner's own folder
and history; putting them in the tool ABI would let the agent perform them on its
own reading of a conversation, and SYSTEM.md would grow a paragraph explaining
when not to.

They live in their own module rather than in ``gateway/projects.py`` because that
file is already near the module-size gate and because the thread lifecycle is a
coherent surface of its own — the split is the same one the registry took.

Refusal shape: the branching module answers with a TYPED reason and owner-facing
copy, so these routes pass it through with HTTP 409 (a precondition the owner can
resolve) or 404/400 where the request itself is wrong. A UI cannot branch on a
stack trace, and a 500 would tell the owner nothing about what to do next.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.gateway._helpers import json_exception, request_drive_root

log = logging.getLogger(__name__)

#: Reasons that mean "the request named something that does not exist".
_NOT_FOUND_REASONS = frozenset({"unknown_project", "unknown_thread"})
#: Reasons that mean "the request itself is malformed", as opposed to a state the
#: owner can resolve. Everything else is a PRECONDITION (409).
_BAD_REQUEST_REASONS = frozenset({"unknown_base"})


def _refusal_status(reason: str) -> int:
    if reason in _NOT_FOUND_REASONS:
        return 404
    if reason in _BAD_REQUEST_REASONS:
        return 400
    return 409


def _answer(outcome: Dict[str, Any]) -> JSONResponse:
    """Pass a typed outcome through, choosing the status from its reason."""
    if outcome.get("ok"):
        return JSONResponse(outcome)
    return JSONResponse(outcome, status_code=_refusal_status(str(outcome.get("reason") or "")))


async def _json_body(request: Request) -> Any:
    try:
        body = await request.json()
    except Exception:
        body = {}
    return body if isinstance(body, dict) else None


def _route_ids(request: Request) -> tuple:
    return (
        str(request.path_params.get("project_id") or "").strip(),
        str(request.path_params.get("thread_id") or "").strip(),
    )


# --------------------------------------------------------------------------- #
# BRANCH OFF / MERGE BACK / remove
# --------------------------------------------------------------------------- #

async def api_thread_branch_bases(request: Request) -> JSONResponse:
    """GET /api/projects/{project_id}/threads/{thread_id}/branch-bases.

    The OWNER's list of bases (A8): the current branch, every other branch, every
    tag, and the always-present "exactly as it is now" entry, which discloses
    whether choosing it would create a snapshot commit. Any commit-ish typed
    instead is accepted by the branch-off route — this list is an offer.
    """
    from ouroboros.projects_registry import get_thread
    from ouroboros.thread_branching import (
        branch_off_bases,
        queue_notice,
        resolve_project_repo,
        thread_location,
    )

    try:
        project_id, thread_id = _route_ids(request)
        drive_root = request_drive_root(request)
        resolved = await asyncio.to_thread(resolve_project_repo, drive_root, project_id)
        if not resolved.get("ok"):
            return _answer(resolved)
        pid = str(resolved["project_id"])
        thread = get_thread(drive_root, pid, thread_id)
        if thread is None:
            return JSONResponse(
                {"ok": False, "reason": "unknown_thread", "message": f"unknown thread: {thread_id}"},
                status_code=404,
            )
        listed = await asyncio.to_thread(branch_off_bases, resolved["repo_dir"])
        return JSONResponse({
            # The response contract declares `ok` and the refusal path sets it, so
            # a client reading `body.ok` first — which is what the shared envelope
            # asks of it — read every SUCCESSFUL bases list as a refusal.
            "ok": True,
            "project_id": pid,
            "thread_id": int(thread["id"]),
            "location": thread_location(drive_root, pid, thread["id"]),
            # A14: the honest queue sentence rides the answer the owner is
            # already reading when they decide whether to branch — that is the
            # ONE moment where "your task would wait, and here is how not to" is
            # both true and actionable.
            "queue_notice": await asyncio.to_thread(queue_notice, drive_root, pid, thread["id"]),
            **listed,
        })
    except Exception as exc:
        return json_exception(exc)


async def api_thread_branch_off(request: Request) -> JSONResponse:
    """POST /api/projects/{project_id}/threads/{thread_id}/branch-off.

    One of A7's two explicit operations. The thread's location is not stored: it
    becomes "worktree" because a worktree now exists.
    """
    from ouroboros.thread_branching import branch_off_thread

    try:
        project_id, thread_id = _route_ids(request)
        body = await _json_body(request)
        if body is None:
            return JSONResponse({"error": "body must be a JSON object"}, status_code=400)
        drive_root = request_drive_root(request)
        outcome = await asyncio.to_thread(
            branch_off_thread,
            drive_root, project_id, thread_id,
            base_ref=str(body.get("base_ref") or "").strip(),
        )
        if outcome.get("ok"):
            _broadcast_thread_change(drive_root, str(outcome["project_id"]), outcome["thread_id"])
        return _answer(outcome)
    except Exception as exc:
        return json_exception(exc)


async def api_thread_merge_back(request: Request) -> JSONResponse:
    """POST /api/projects/{project_id}/threads/{thread_id}/merge-back.

    A9's preconditions are enforced in the branching module and reported as typed
    409s. A conflict comes back as ``merge_conflict`` WITH its paths: the merge is
    already aborted by then, so the owner's folder is untouched and the thread
    still holds every commit in its branch.

    ``acknowledge_checkout_dirty`` in the body is the owner's answer to
    ``checkout_dirty`` — A10's consent shape, reused: the refusal names the flag,
    the owner re-sends with it, and the files that stayed behind are named again
    on the success. A body is optional here so the plain merge-back call is
    unchanged.
    """
    from ouroboros.thread_branching import merge_back_thread

    try:
        project_id, thread_id = _route_ids(request)
        drive_root = request_drive_root(request)
        body = await _json_body(request) or {}
        outcome = await asyncio.to_thread(
            merge_back_thread, drive_root, project_id, thread_id,
            acknowledge_checkout_dirty=bool(body.get("acknowledge_checkout_dirty")),
        )
        if outcome.get("ok"):
            _broadcast_thread_change(drive_root, str(outcome["project_id"]), outcome["thread_id"])
        return _answer(outcome)
    except Exception as exc:
        return json_exception(exc)


async def api_thread_worktree_inspect(request: Request) -> JSONResponse:
    """GET /api/projects/{project_id}/threads/{thread_id}/worktree.

    WHAT removing this checkout would destroy, so the prompt A10 requires can be
    shown BEFORE anything is removed: dirty files, commits the base never
    received, and an honest ``error`` when the checkout cannot be read at all
    (which counts as unsafe — "cannot tell" must never read as "nothing to lose").
    """
    from ouroboros.projects_registry import get_thread
    from ouroboros.thread_branching import thread_location
    from ouroboros.thread_worktrees import get_thread_worktree, inspect_thread_worktree

    try:
        project_id, thread_id = _route_ids(request)
        drive_root = request_drive_root(request)
        thread = get_thread(drive_root, project_id, thread_id)
        if thread is None:
            return JSONResponse(
                {"ok": False, "reason": "unknown_thread", "message": f"unknown thread: {thread_id}"},
                status_code=404,
            )
        tid = int(thread["id"])
        row = get_thread_worktree(drive_root, project_id, tid)
        if not row:
            return JSONResponse({
                "ok": True, "project_id": project_id, "thread_id": tid,
                "location": {"where": "project_folder"}, "inspection": {},
            })
        inspection = await asyncio.to_thread(inspect_thread_worktree, row)
        return JSONResponse({
            "ok": True,
            "project_id": project_id,
            "thread_id": tid,
            "location": thread_location(drive_root, project_id, tid),
            "inspection": inspection,
        })
    except Exception as exc:
        return json_exception(exc)


async def api_thread_worktree_remove(request: Request) -> JSONResponse:
    """POST /api/projects/{project_id}/threads/{thread_id}/worktree/remove.

    A10: never silent, never automatic, never on a timer. Unmerged work refuses
    with ``unmerged_work`` and the inspection attached; the owner acknowledges by
    re-sending with ``acknowledge_unmerged: true``, which is the whole consent
    mechanism — there is no other path into this function.
    """
    from ouroboros.projects_registry import get_thread
    from ouroboros.thread_branching import thread_location
    from ouroboros.thread_worktrees import remove_thread_worktree

    try:
        project_id, thread_id = _route_ids(request)
        body = await _json_body(request)
        if body is None:
            return JSONResponse({"error": "body must be a JSON object"}, status_code=400)
        drive_root = request_drive_root(request)
        thread = get_thread(drive_root, project_id, thread_id)
        if thread is None:
            return JSONResponse(
                {"ok": False, "reason": "unknown_thread", "message": f"unknown thread: {thread_id}"},
                status_code=404,
            )
        tid = int(thread["id"])
        outcome = await asyncio.to_thread(
            remove_thread_worktree,
            data_dir=drive_root,
            project_id=project_id,
            thread_id=tid,
            acknowledge_unmerged=bool(body.get("acknowledge_unmerged")),
        )
        payload: Dict[str, Any] = {
            "ok": bool(outcome.get("removed")),
            "project_id": project_id,
            "thread_id": tid,
            "removed": bool(outcome.get("removed")),
            "reason": str(outcome.get("reason") or ""),
            "inspection": outcome.get("inspection") or {},
            "location": thread_location(drive_root, project_id, tid),
            # A clean removal deletes the thread branch too, so the round trip
            # branch → merge → remove → branch again is repeatable (T3R-5). A
            # branch that SURVIVED says why, because it is what stops the next
            # branch-off and the owner would otherwise meet that as a bare
            # "branch already exists".
            "branch": str(outcome.get("branch") or ""),
            "branch_removed": bool(outcome.get("branch_removed")),
            "branch_kept_reason": str(outcome.get("branch_kept_reason") or ""),
            # `unmerged_work` HAS an owner-answerable flag — `acknowledge_unmerged`
            # in this route's own body — and never declared it, so a client that
            # reads `acknowledgeable` (the shared envelope's field, which every
            # other answerable refusal sets) bailed on the one refusal whose
            # message ends "or confirm you want it gone" (I9). The sentence
            # offered a confirmation the gesture could not give.
            "acknowledgeable": str(outcome.get("reason") or "") == "unmerged_work",
        }
        if payload["ok"]:
            _broadcast_thread_change(drive_root, project_id, tid)
            return JSONResponse(payload)
        payload["message"] = _removal_message(payload["reason"], payload["inspection"], payload)
        return JSONResponse(payload, status_code=_refusal_status(payload["reason"]))
    except Exception as exc:
        return json_exception(exc)


def _removal_message(
    reason: str, inspection: Dict[str, Any], outcome: Dict[str, Any] | None = None,
) -> str:
    """Owner-facing copy for a refused removal — what is at stake, in plain words."""
    if reason == "removal_failed":
        # The one refusal that fell through to "something went wrong", which this
        # module's docstring forbids (I17). The cause is in the log and nowhere the
        # owner can see it, while `branch_kept_reason` already carries a usable
        # sentence that `describeOutcome` never reads on a refusal. Both go here.
        detail = str((inspection or {}).get("error") or "").strip()
        kept = str((outcome or {}).get("branch_kept_reason") or "").strip()
        parts = [
            "The checkout is still on disk, so nothing was removed and its registry "
            "entry was kept — a folder the registry can no longer see could not be "
            "re-provisioned or removed at all."
        ]
        if detail:
            parts.append(f"Reading it reported: {detail}.")
        parts.append(
            "Something is usually holding it: a shell or editor open in that folder, "
            "a git lock, or a read-only parent. Close whatever is using it and try "
            "again — the removal is safe to retry."
        )
        if kept:
            parts.append(f"Its branch was left alone — {kept}.")
        return " ".join(parts)
    if reason == "unmerged_work":
        parts = []
        commits = int(inspection.get("unmerged_commits") or 0)
        if commits:
            parts.append(f"{commits} commit{'s' if commits != 1 else ''} the project folder never received")
        # The TRUE number of dirty entries, never the length of the bounded
        # listing the inspection carries. This sentence is the last thing an
        # owner reads before acknowledging an irreversible removal, and it used
        # to state the cap: 800 modified files were announced as 200.
        listed = len(inspection.get("dirty_files") or [])
        dirty_total = max(listed, int(inspection.get("dirty_files_total") or 0))
        if dirty_total:
            parts.append(_plural(dirty_total, "uncommitted file change"))
        if inspection.get("error"):
            parts.append("a checkout that could not be read, so its contents are unknown")
        detail = " and ".join(parts) if parts else "work the project folder never received"
        sentences = [f"This checkout still holds {detail}. Removing it deletes that work."]
        if dirty_total > listed:
            sentences.append(
                f"Only the first {listed} of those files are listed here." if listed
                else "None of those files are listed here."
            )
        sentences.append("Merge it back first, or confirm you want it gone.")
        return " ".join(sentences)
    if reason == "unknown":
        return "This thread has no checkout to remove."
    if reason == "project_busy":
        # The SAME sentence merge-back gives for the same fact, because it is the
        # same judge: removing a checkout deletes a folder a task may be writing in.
        return (
            "A task is running or queued in this project right now. Removing a "
            "checkout deletes the folder it is working in, so it waits until that "
            "task finishes."
        )
    if reason == "path_outside_root":
        return (
            "The registry entry points outside the folder these checkouts live in, "
            "so nothing was deleted. This needs a look before it can be removed."
        )
    return "The checkout could not be removed."


# --------------------------------------------------------------------------- #
# The checkout's own diff (A13 / X9)
# --------------------------------------------------------------------------- #

async def api_thread_diff(request: Request) -> JSONResponse:
    """GET /api/projects/{project_id}/threads/{thread_id}/diff.

    A13: the Changes screen for a branched thread shows THAT checkout's diff.
    Thin route over ``task_diff.thread_checkout_diff_payload`` — same hardened git
    invocation, same patch envelope and the same process-wide admission gate as
    the task diff, so the two surfaces cannot drift apart or fan out git processes
    independently against the owner's repo.
    """
    from ouroboros.gateway.task_diff import diff_gate, thread_checkout_diff_payload
    from ouroboros.projects_registry import get_thread

    try:
        project_id, thread_id = _route_ids(request)
        drive_root = request_drive_root(request)
        thread = get_thread(drive_root, project_id, thread_id)
        if thread is None:
            return JSONResponse({"error": f"unknown thread: {thread_id}"}, status_code=404)
        tid = int(thread["id"])
    except Exception as exc:
        return json_exception(exc)
    try:
        async with diff_gate():
            payload = await asyncio.to_thread(
                thread_checkout_diff_payload, drive_root, project_id, tid,
            )
    except Exception as exc:
        return json_exception(exc, 503)
    return JSONResponse({"project_id": project_id, "thread_id": tid, **payload})


# --------------------------------------------------------------------------- #
# Lifecycle: archive / restore / delete (D4 with X10's admission fencing)
# --------------------------------------------------------------------------- #

def _lifecycle_answer(
    drive_root: Any, project_id: str, thread_id: Any, thread: Dict[str, Any], **extra: Any,
) -> JSONResponse:
    _broadcast_thread_change(drive_root, project_id, thread_id)
    return JSONResponse({
        "ok": True,
        "project_id": str(project_id),
        "thread_id": int(thread.get("id") or 0),
        "chat_id": int(thread.get("chat_id") or 0),
        "lifecycle": str(thread.get("lifecycle") or ""),
        **extra,
    })


async def _lifecycle_route(request: Request, operation, **kwargs: Any) -> JSONResponse:
    """Shared shell for archive / restore / delete: resolve, run, answer typed."""
    from ouroboros.project_threads_registry import ThreadLifecycleError
    from ouroboros.projects_registry import get_project, get_thread

    project_id, thread_id = _route_ids(request)
    drive_root = request_drive_root(request)
    project = get_project(drive_root, project_id)
    if project is None:
        return JSONResponse(
            {"ok": False, "reason": "unknown_project", "message": f"unknown project: {project_id}"},
            status_code=404,
        )
    pid = str(project["id"])
    if get_thread(drive_root, pid, thread_id) is None:
        return JSONResponse(
            {"ok": False, "reason": "unknown_thread", "message": f"unknown thread: {thread_id}"},
            status_code=404,
        )
    try:
        return operation(drive_root, pid, thread_id, **kwargs)
    except ThreadLifecycleError as exc:
        return JSONResponse(
            {"ok": False, "reason": exc.reason, "message": str(exc), "project_id": pid},
            status_code=409,
        )


async def api_thread_archive(request: Request) -> JSONResponse:
    """POST /api/projects/{project_id}/threads/{thread_id}/archive.

    HIDE, and nothing else (D4). History rows, task bindings, the fork cursors of
    this thread's children and its git worktree are all left exactly as they are,
    and `restore` puts it back. Archiving thread #0 is refused: that thread IS the
    project, and the project has its own operations.

    X10, decided explicitly: archiving does NOT refuse while a task is running,
    and the thread stays VISIBLE until that task is terminal. Refusing would make
    the owner babysit a run they have already finished caring about; hiding it
    immediately would leave live output arriving in a room they cannot open. The
    answer reports `visible_until_terminal` so the surface can say which of the
    two just happened.
    """
    from ouroboros.gateway.state import live_thread_chat_ids
    from ouroboros.projects_registry import archive_thread

    def _run(drive_root, pid, thread_id):
        thread = archive_thread(drive_root, pid, thread_id)
        if thread is None:
            return JSONResponse(
                {"ok": False, "reason": "unknown_thread", "message": "unknown thread"},
                status_code=404,
            )
        live = int(thread.get("chat_id") or 0) in live_thread_chat_ids()
        return _lifecycle_answer(
            drive_root, pid, thread_id, thread,
            archived_at=str(thread.get("archived_at") or ""),
            visible_until_terminal=live,
        )

    try:
        return await _lifecycle_route(request, _run)
    except Exception as exc:
        return json_exception(exc)


async def api_thread_restore(request: Request) -> JSONResponse:
    """POST /api/projects/{project_id}/threads/{thread_id}/restore — un-archive."""
    from ouroboros.projects_registry import restore_thread

    def _run(drive_root, pid, thread_id):
        thread = restore_thread(drive_root, pid, thread_id)
        if thread is None:
            return JSONResponse(
                {"ok": False, "reason": "unknown_thread", "message": "unknown thread"},
                status_code=404,
            )
        return _lifecycle_answer(drive_root, pid, thread_id, thread)

    try:
        return await _lifecycle_route(request, _run)
    except Exception as exc:
        return json_exception(exc)


async def api_thread_delete(request: Request) -> JSONResponse:
    """POST /api/projects/{project_id}/threads/{thread_id}/delete.

    X10's three steps, in this order and no other: FENCE routing, then cancel and
    quiesce the tasks selected by EXACT thread chat id, then tombstone. Fencing
    first is what stops a message landing in a room already on its way out;
    selecting by exact chat id is what stops a sibling thread's work being
    cancelled along with it.

    What a tombstone does and does not do, stated plainly because the answer says
    so too: the id and chat id are reserved forever (a reused 28-bit chat id would
    merge a dead thread's history into a live conversation) and the journal rows
    physically REMAIN — the journal is shared by every chat and nothing here
    rewrites it.

    The thread's CHECKOUT goes with it. A tombstoned thread is invisible on every
    surface, `list_thread_worktrees` has no route, and branch/merge refuse
    `thread_not_live` — so leaving the checkout behind orphaned a folder and a
    branch that A10's "explicit removal" could no longer reach, on durable state
    that is exempt from every GC.

    What may BLOCK that is only work genuinely at risk (`checkout_work_at_risk`):
    commits the project folder never received, modifications to files git is
    TRACKING, or an inspection that could not be taken at all. Those refuse with
    `checkout_holds_work` and name the explicit removal route, because a gesture
    aimed at a thread must never destroy the last copy of something.

    A checkout whose only dirt is ignored or untracked content — a
    `node_modules/`, a `build.log` — does NOT refuse. It rides the same
    acknowledgement shape the removal route already uses
    (`acknowledge_unmerged` in the body): the first call answers
    `checkout_holds_rebuildable_files` naming exactly what is there, and the
    owner's yes deletes both. Refusing over a build directory made "delete this
    thread and its folder" a three-step detour through merge-back and an
    acknowledged removal, which is friction the owner did not ask for; naming it
    and asking once is two steps and destroys nothing silently.
    """
    from ouroboros.contracts.chat_id_policy import MAIN_THREAD_ID
    from ouroboros.project_threads_registry import (
        THREAD_ACTIVE,
        THREAD_ARCHIVED,
        THREAD_DELETING,
        ThreadLifecycleError,
    )
    from ouroboros.projects_registry import begin_thread_deletion, get_thread
    from ouroboros.thread_branching import thread_location
    from ouroboros.thread_worktrees import (
        checkout_work_at_risk,
        get_thread_worktree,
        inspect_thread_worktree,
        remove_thread_worktree,
    )
    from supervisor.task_lifecycle import start_thread_deletion

    #: The states `begin_thread_deletion` will accept. Asked HERE, before the
    #: checkout is touched, because the checkout removal now happens first: a
    #: transition the registry would refuse must not delete a folder on its way to
    #: a 409. `_set_thread_lifecycle` remains the SSOT for the transition itself —
    #: this only decides whether to start.
    _DELETABLE_FROM = frozenset({THREAD_ACTIVE, THREAD_ARCHIVED, THREAD_DELETING})

    def _refuse(drive_root, pid, tid, reason, message, inspection):
        return JSONResponse({
            "ok": False,
            "reason": reason,
            "message": message,
            "project_id": pid,
            "thread_id": tid,
            "inspection": inspection,
            # `checkout_holds_rebuildable_files` is a QUESTION, not a wall, and a
            # menu can only render the second call if the answer says so — the
            # same field `checkout_dirty` uses on the merge-back envelope.
            "acknowledgeable": reason == "checkout_holds_rebuildable_files",
            "location": thread_location(drive_root, pid, tid),
        }, status_code=_refusal_status(reason))

    def _run(drive_root, pid, thread_id, acknowledged=False):
        thread = get_thread(drive_root, pid, thread_id) or {}
        tid = int(thread.get("id") or 0)
        lifecycle = str(thread.get("lifecycle") or THREAD_ACTIVE)
        if tid == MAIN_THREAD_ID:
            raise ThreadLifecycleError(
                "thread_zero_is_the_project",
                "This thread IS the project. Archive or delete the project itself.",
            )
        if lifecycle not in _DELETABLE_FROM:
            raise ThreadLifecycleError(
                "lifecycle_conflict",
                f"this thread is {lifecycle}; it cannot become {THREAD_DELETING}",
            )
        removed: Dict[str, Any] = {}
        row = get_thread_worktree(drive_root, pid, tid)
        if row:
            # BEFORE the fence: a refusal must leave the thread exactly as it was,
            # not fenced-but-undeleted with no way forward. The inspection is read
            # here rather than left to `remove_thread_worktree` because DELETION
            # blocks on a narrower fact than removal does — `checkout_work_at_risk`.
            inspection = inspect_thread_worktree(row)
            risk = checkout_work_at_risk(inspection)
            if risk["at_risk"]:
                return _refuse(
                    drive_root, pid, tid, "checkout_holds_work",
                    _delete_refusal_message(risk), inspection,
                )
            if (risk["ignored_files"] or risk["untracked_files"]) and not acknowledged:
                return _refuse(
                    drive_root, pid, tid, "checkout_holds_rebuildable_files",
                    _delete_confirm_message(risk), inspection,
                )
            # Nothing at risk: the acknowledgement (when one was needed) has been
            # given, so the checkout goes. `acknowledge_unmerged` is passed because
            # ignored files make the removal's own judge read "dirty"; the branch
            # still faces `git branch -d` on its own account.
            removed = remove_thread_worktree(
                data_dir=drive_root, project_id=pid, thread_id=tid,
                acknowledge_unmerged=True,
            )
            if not removed.get("removed"):
                # `project_busy`, `path_outside_root`, or work that appeared
                # between the inspection and the removal. Not the owner's cue to
                # confirm anything — the reason says what to do.
                reason = str(removed.get("reason") or "")
                late = removed.get("inspection") or inspection
                return _refuse(
                    drive_root, pid, tid,
                    "checkout_holds_work" if reason == "unmerged_work" else reason,
                    f"{_removal_message(reason, late, removed)} This thread cannot be deleted "
                    "while its checkout is in that state, because deleting it would "
                    "leave the folder and its branch with no surface that can reach them.",
                    late,
                )
        fenced = begin_thread_deletion(drive_root, pid, thread_id)
        if fenced is None:
            return JSONResponse(
                {"ok": False, "reason": "unknown_thread", "message": "unknown thread"},
                status_code=404,
            )
        tid = int(fenced.get("id") or 0)
        chat_id = int(fenced.get("chat_id") or 0)
        location = thread_location(drive_root, pid, tid)
        start_thread_deletion(drive_root, pid, tid, chat_id)
        return _lifecycle_answer(
            drive_root, pid, tid, fenced,
            # Every honest disclosure this operation owes the owner, in the answer
            # rather than in a docstring nobody reading the UI will see.
            journal_rows_retained=True,
            worktree_kept=location["where"] == "worktree",
            worktree_removed=bool(removed.get("removed")),
            branch=str(removed.get("branch") or ""),
            branch_removed=bool(removed.get("branch_removed")),
            location=location,
        )

    try:
        body = await _json_body(request)
        if body is None:
            return JSONResponse({"error": "body must be a JSON object"}, status_code=400)
        return await _lifecycle_route(
            request, _run, acknowledged=bool(body.get("acknowledge_unmerged")),
        )
    except Exception as exc:
        return json_exception(exc)


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _omission_clause(risk: Dict[str, Any]) -> str:
    """The sentence a per-category count owes when the listing was bounded.

    ``checkout_work_at_risk`` splits a BOUNDED listing, so its category lengths
    are counts of what was shown. Which category the unlisted entries fall into
    is not knowable from the listing, so this states the one thing that is true:
    how many were left out. Empty — not "0 more" — when nothing was.
    """
    omitted = int(risk.get("omitted_files") or 0)
    if not omitted:
        return ""
    return f" {_plural(omitted, 'further changed file')} in that checkout {'is' if omitted == 1 else 'are'} not listed here."


def _delete_refusal_message(risk: Dict[str, Any]) -> str:
    """Why a deletion will NOT go ahead — naming the work that is actually at stake.

    "Holds work" is not a category the owner can act on. Commits that exist
    nowhere else and edits to tracked files are different problems with different
    answers, so they are named separately, and the sentence ends on the route out
    rather than on the wall.
    """
    parts = []
    if risk["unmerged_commits"]:
        parts.append(
            f"{_plural(int(risk['unmerged_commits']), 'commit')} that exist nowhere else — "
            "the project folder never received them"
        )
    if risk["tracked_files"]:
        parts.append(f"changes to {_plural(len(risk['tracked_files']), 'file')} git is tracking")
    if risk["unreadable"]:
        parts.append("a checkout that could not be read, so its contents are unknown")
    detail = " and ".join(parts) if parts else "work the project folder never received"
    return (
        f"This thread's checkout holds {detail}.{_omission_clause(risk)} Deleting the thread "
        "would delete that folder, and a deleted thread has no surface left that could reach "
        "it — so the delete stops here. Merge the work back, or remove the checkout explicitly "
        "(Remove checkout…, which asks you to confirm first), then delete the thread."
    )


def _delete_confirm_message(risk: Dict[str, Any]) -> str:
    """The one question a deletion asks: this folder holds rebuildable files.

    Named, never merely counted — "some files" teaches nothing about whether the
    owner cares. Ignored and untracked are kept apart because they mean different
    things: one is content git was TOLD to ignore, the other is content git has
    simply never been told about.
    """
    parts = []
    if risk["ignored_files"]:
        parts.append(f"{_plural(len(risk['ignored_files']), 'file')} git was told to ignore")
    if risk["untracked_files"]:
        parts.append(f"{_plural(len(risk['untracked_files']), 'file')} git is not tracking")
    return (
        f"This thread's checkout holds {' and '.join(parts)} — nothing committed, and "
        f"nothing changed in a file git is tracking.{_omission_clause(risk)} Deleting the "
        "thread deletes that folder with them. Confirm to delete both."
    )


# --------------------------------------------------------------------------- #
# Broadcast
# --------------------------------------------------------------------------- #

def _broadcast_thread_change(drive_root: Any, project_id: str, thread_id: Any) -> None:
    """Ride ``projects_changed`` (X11): no new ABI event for a registry mutation
    the client already refreshes on. The affected thread's chat id travels with it
    so an open client's known-chat set is current before any live frame arrives.
    """
    try:
        from ouroboros.projects_registry import get_thread
        from supervisor.message_bus import get_bridge

        thread = get_thread(drive_root, project_id, thread_id) or {}
        get_bridge().broadcast({
            "type": "projects_changed",
            "project_id": str(project_id),
            "chat_id": thread.get("chat_id"),
        })
    except Exception:
        log.debug("projects_changed broadcast failed for %s#%s", project_id, thread_id, exc_info=True)


__all__ = [
    "api_thread_archive",
    "api_thread_branch_bases",
    "api_thread_branch_off",
    "api_thread_delete",
    "api_thread_diff",
    "api_thread_merge_back",
    "api_thread_restore",
    "api_thread_worktree_inspect",
    "api_thread_worktree_remove",
]
