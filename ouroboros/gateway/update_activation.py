"""The unified Update's HTTP actions on ``POST /api/update/apply``: prepare, activate, discard.

``prepare`` and ``apply_without_local_changes`` build the installation's candidate in its
own workspace (``supervisor/update_prepare.py``) on a background thread: the serving
checkout, its writers and its imports keep running, and the panel follows the
candidate's state through ``/api/update/status``. Like activate and discard, leaving the
local work out names the candidate it was asked for: a different live candidate refuses the
request (409 ``candidate_changed``, nothing started). The 202 answer is final: the preparation
it started checks again, before it makes a copy and at its admission, and a change by then
refuses it, discards any copy it made and leaves the live candidate as it was. ``activate`` is the owner's
"Restart to update": under the update lock it stops the repository writers with the
existing fence, holds the local model runtime's install (the server's own package writer) when the
candidate replaces the serving packages, reads the base and the environment again (a moved one makes the
candidate stale and reopens the writers), authorizes the pre-import switch (``body_adoption.authorize_update``),
starts the app installer when the Update brings a new official app (``update_native.hand_over``)
and requests the restart that alone may arm it. ``discard_candidate`` settles the record;
the worktree GC then retires its checkout and scratch. Nothing here edits the serving
tree; the legacy ``auto_merge``/``assisted``/``replace`` lanes stay in ``control.py``.
``carried_list_download`` serves the live candidate's complete list of carried work on
``GET /api/files/download?carried_list=<candidate id>``, the link ``candidate_view`` names.
"""

from __future__ import annotations

import asyncio
import logging
import os
import stat
import threading
from urllib.parse import urlencode

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from ouroboros.gateway import update_progress

log = logging.getLogger(__name__)

STRATEGIES = frozenset({"prepare", "apply_without_local_changes", "activate", "discard_candidate"})


def _refused(exc, status: int = 409) -> JSONResponse:
    return JSONResponse({"error": exc.text, "reason": exc.code, "outcome": "not_applied",
                         **({"details": exc.facts} if getattr(exc, "facts", None) else {})}, status_code=status)


def _start_preparation(without_local_changes: bool, candidate_id: str = "") -> JSONResponse:
    """Admit one preparation and run it off this request. Leaving the local work out names its candidate: the
    one the request carries, else the one live now; the admission refuses any other, before and while it starts."""
    from supervisor import update_prepare
    from supervisor.update_checks import workspace_checks

    expected = None
    if without_local_changes:
        expected = candidate_id or str((update_prepare.current() or {}).get("candidate_id") or "")
    try:
        update_prepare.admit(expected)
    except update_prepare.PrepareRefused as exc:
        if exc.code == "UPDATE_CANDIDATE_CHANGED":
            return _changed(exc.text, exc.facts)  # the candidate asked for and the live one, for clients and logs
        return JSONResponse({"error": exc.text, "reason": exc.code, "outcome": "not_applied"}, status_code=409)

    def run() -> None:
        try:
            update_prepare.prepare(fetch=True, without_local_changes=without_local_changes, checks=workspace_checks,
                                   expected=expected, admitted=True)
        except update_prepare.PrepareRefused as exc:
            update_prepare.note_refusal(exc)
        except Exception:
            log.warning("update preparation crashed", exc_info=True)
        finally:
            update_progress.notify_changed()

    update_progress.notify_changed()  # every client's panel and indicator: the admitted slot reads as preparing
    try:
        threading.Thread(target=run, daemon=True, name="update-prepare").start()
    except BaseException:
        update_prepare._running.release()  # the admitted slot was never handed to a preparation
        update_progress.notify_changed()
        raise
    return JSONResponse({"status": "preparing", "strategy": "apply_without_local_changes" if without_local_changes
                         else "prepare"}, status_code=202)


def _changed(why: str, details: dict | None = None) -> JSONResponse:
    return JSONResponse({"error": why, "reason": "candidate_changed", "outcome": "not_applied",
                         **({"details": details} if details else {})}, status_code=409)


def candidate_view(latest_sha: str = "") -> dict:
    """The candidate as clients read it (``update_prepare.projection``), with the complete list of the work it
    carries as ``list_url``: the download of that list by this candidate's id (``carried_list_download``), never
    a path, absent while the list does not exist."""
    from supervisor import update_prepare

    candidate = update_prepare.projection(latest_sha=latest_sha)
    listing = str((candidate.get("carried") or {}).get("list") or "")
    try:
        if listing and candidate.get("candidate_id") and os.path.isfile(listing):
            candidate["carried"] = {**candidate["carried"], "list_url": "/api/files/download?" + urlencode(
                {"carried_list": candidate["candidate_id"]})}
    except Exception:
        log.debug("the carried list's download link was not resolved", exc_info=True)
    return candidate


async def carried_list_download(request: Request) -> Response:
    """``GET /api/files/download?carried_list=<candidate id>``: the complete list of the work the live candidate
    carries (``update_prepare.carried_list_path``), chosen by that exact id. Like a chat attachment's ``upload``,
    it never depends on the Files root, so a phone signed in over the network downloads it too; it reads no path
    and no other file. Another candidate than the live one answers 409 ``candidate_changed``, a live one without
    a list 404 ``list_unavailable``."""
    from ouroboros.gateway._helpers import json_error
    from ouroboros.gateway.files import _file_response
    from supervisor import update_prepare

    params = request.query_params
    if len(params.getlist("carried_list")) != 1 or "path" in params or "upload" in params:
        return json_error("carried_list must be the only file selector", status=400)
    candidate_id = params.get("carried_list", "")
    if not candidate_id:
        return json_error("Invalid candidate id.", status=400)
    try:
        row = await asyncio.to_thread(update_prepare.current)
    except Exception as exc:
        return json_error(f"the update record could not be read: {exc}", status=500)
    if row is None or row.get("candidate_id") != candidate_id:
        return json_error("the prepared update changed; check again", status=409, reason_code="candidate_changed")
    listing = update_prepare.carried_list_path(row)
    try:
        # Only the list this record says it wrote, as a regular file (never a link to another one).
        if str((row.get("carried") or {}).get("list") or "") != str(listing) or not stat.S_ISREG(
                os.lstat(listing).st_mode):
            raise FileNotFoundError(listing)
    except OSError:
        return json_error("this update keeps no list of carried work", status=404, reason_code="list_unavailable")
    return _file_response(listing, filename=listing.name)


def _activate_fenced(request: Request, candidate_id: str) -> JSONResponse:
    from ouroboros import body_adoption
    from ouroboros.gateway import control
    from supervisor import update_prepare
    from supervisor.git_ops import DRIVE_ROOT
    from supervisor.update_merge import acquire_update_lock, active_update_tx, release_update_lock

    seen = update_prepare.current()
    if seen is None or (candidate_id and seen.get("candidate_id") != candidate_id):
        return _changed("the prepared update changed; check again")
    candidate_id = str(seen.get("candidate_id") or "")  # the exact candidate this request may activate
    if seen.get("state") != "ready":
        return JSONResponse({"error": f"the update candidate is {seen.get('state')}, not ready",
                             "reason": "candidate_not_ready", "outcome": "not_applied"}, status_code=409)
    refusal = control._index_lock_precondition()
    if refusal is not None:
        return refusal
    try:
        lock = acquire_update_lock()
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc), "outcome": "not_applied"}, status_code=409)
    installs, restarting = None, False
    try:
        if active_update_tx():
            return JSONResponse({"error": "a legacy managed update is in progress", "outcome": "not_applied"},
                                status_code=409)
        blockers = control._quiesce_repo_writers("activation")
        if blockers:
            return control._fence_failure(blockers)
        # The fence may have waited: a discard or a new preparation meanwhile changed what is current.
        # Everything from this re-read until the record says activating holds the lifecycle lock.
        with update_prepare.lifecycle():
            row = update_prepare.current()
            if row is None or row.get("candidate_id") != candidate_id or row.get("state") != "ready":
                control._respawn_workers_after_failed_update()
                return _changed("the prepared update changed while the writers stopped; nothing was switched")
            if (row.get("environment") or {}).get("generation"):
                # The fence stopped the task writers; the server's own package writer is the local model runtime's
                # install: held from the package check below until the restart switches them, or refused.
                from ouroboros.local_model import get_manager

                installs = get_manager()
                busy = installs.hold_runtime_install("an update is being activated; install it after the restart")
                if busy:
                    installs = None
                    control._respawn_workers_after_failed_update()
                    return JSONResponse({"error": f"{busy}; restart to update once it finished",
                                         "reason": "packages_busy", "outcome": "not_applied"}, status_code=409)
            why = update_prepare.activation_check(row)
            if why:
                control._respawn_workers_after_failed_update()
                return JSONResponse({"error": why, "reason": why.split(":", 1)[0], "outcome": "not_applied",
                                     "candidate": candidate_view()}, status_code=409)
            try:
                handoff = body_adoption.authorize_update(DRIVE_ROOT, row)
            except body_adoption.AdoptionRefused as exc:
                control._respawn_workers_after_failed_update()
                return JSONResponse({"error": exc.text, "reason": exc.code, "outcome": "not_applied"},
                                    status_code=409)
            body_adoption.bind_update_restart(DRIVE_ROOT)
            native = row.get("native") or {}
            if native.get("status") == "staged":
                # The installer runs, detached and in custody, BEFORE the restart is requested: it acts only
                # once that restart's exit tail armed the switch, and the server then holds its exit for the
                # old app's Quit (``body_adoption.native_hold``). One that cannot start leaves nothing armed.
                from supervisor import update_native

                started = update_native.installer_command(native, data_dir=DRIVE_ROOT, adoption_id=handoff["id"])
                if started["ok"]:
                    started = update_native.hand_over(started, data_dir=DRIVE_ROOT, adoption_id=handoff["id"])
                if not started["ok"]:
                    body_adoption.abandon(DRIVE_ROOT, "native_hand_over_failed", candidate_id=candidate_id)
                    control._respawn_workers_after_failed_update()
                    return JSONResponse({"error": f"the new app could not be installed: {started['reason']}",
                                         "reason": "native_hand_over_failed", "outcome": "not_applied"},
                                        status_code=409)
            update_prepare.mark(row, state="activating", activation={"adoption_id": handoff["id"]})
        if not control._request_restart(request):
            with update_prepare.lifecycle():
                # An installer already started finds its record gone and ends with nothing changed.
                body_adoption.abandon(DRIVE_ROOT, "restart_unavailable", candidate_id=candidate_id)
                live = update_prepare.find(candidate_id)
                if live is not None and live.get("state") == "activating":
                    update_prepare.mark(live, state="ready", activation={})
            control._respawn_workers_after_failed_update()
            return JSONResponse({"error": "the restart could not be requested; nothing was switched",
                                 "status": "restart_required", "outcome": "not_applied"}, status_code=500)
        restarting = True  # the hold stays: the restart switches the packages
        update_progress.notify_changed()  # other clients and the indicator: this candidate is activating now
        return JSONResponse({"status": "ok", "restarting": True, "strategy": "activate",
                             "candidate": candidate_view()})
    finally:
        if installs is not None and not restarting:
            installs.release_runtime_install()
        release_update_lock(lock)


def _discard(candidate_id: str) -> JSONResponse:
    from ouroboros import subagent_worktrees
    from supervisor import update_prepare

    row = update_prepare.current()
    if row is None or (candidate_id and row.get("candidate_id") != candidate_id):
        return JSONResponse({"error": "no such prepared update", "reason": "candidate_changed",
                             "outcome": "not_applied"}, status_code=409)
    try:
        update_prepare.discard(row)
    except update_prepare.PrepareRefused as exc:
        return _refused(exc)

    def retire() -> None:
        from supervisor.queue_transitions import task_settlement_liveness

        # Retired only once its resolver's exit is proven; otherwise a later GC pass decides.
        subagent_worktrees.prune_orphans(only={str(row["candidate_id"])}, live=task_settlement_liveness)

    threading.Thread(target=retire, daemon=True, name="update-candidate-retire").start()
    update_progress.notify_changed()
    return JSONResponse({"status": "discarded", "strategy": "discard_candidate"})


async def apply(request: Request, strategy: str, body: dict) -> JSONResponse:
    """One of ``STRATEGIES``; the caller validated the body against ``UpdateApplyRequest``."""
    candidate_id = str(body.get("candidate_id") or "").strip()
    if strategy in ("prepare", "apply_without_local_changes"):
        return _start_preparation(strategy == "apply_without_local_changes", candidate_id)
    if strategy == "activate":
        return await asyncio.to_thread(_activate_fenced, request, candidate_id)
    return await asyncio.to_thread(_discard, candidate_id)
