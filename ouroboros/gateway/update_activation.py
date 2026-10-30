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
and requests the restart that alone may arm it. Whatever ends it short of that restart settles what it may have left
before the writers return (``_abort_activation``). ``discard_candidate`` settles the record;
the worktree GC then retires its checkout and scratch. Nothing here edits the serving
tree; the legacy ``auto_merge``/``assisted``/``replace`` lanes stay in ``control.py``.
``carried_list_download`` serves the live candidate's complete list of carried work on
``GET /api/files/download?carried_list=<candidate id>``, the link ``candidate_view`` names, and
``candidate_preflight`` answers ``POST /api/update/preflight`` naming a candidate: the effects of its
Restart to update, which the panel reads before the owner confirms.
"""

from __future__ import annotations

import asyncio
import json
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


def candidate_preflight(candidate_id: str) -> JSONResponse:
    """``POST /api/update/preflight`` naming ``candidate_id``: what that candidate's Restart to update does, read now.

    Nothing is fetched or planned: ``merge_plan`` is the range it was prepared for and ``effects``
    (``update_impact.candidate_effects``) its composition's engine transition, the engine's active work and
    the runtime its preparation selected. Another live candidate, or none, answers 409 ``candidate_changed``
    before anything is observed."""
    from ouroboros.gateway._helpers import json_error
    from ouroboros.update_impact import candidate_effects
    from supervisor import update_prepare

    row = update_prepare.current()
    if row is None or row.get("candidate_id") != candidate_id:
        return json_error("the prepared update changed; check again", status=409, reason_code="candidate_changed")
    target = row.get("target") or {}
    plan = {"base_sha": str(row.get("base_sha") or ""), "target_sha": str(target.get("target_sha") or ""),
            "target_ref": str(target.get("target_ref") or ""), "update_channel": str(target.get("channel") or "")}
    return JSONResponse({"merge_plan": plan, "effects": candidate_effects(row)})


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


class _ActivationRefused(Exception):
    """A step after authorization cannot go on: an expected end of ``_abort_activation``, in its own words."""

    def __init__(self, text: str, http_status: int = 409, **fields) -> None:
        super().__init__(text)
        self.text, self.http_status, self.fields = text, http_status, fields


def _writers_back(body: dict, status: int = 409) -> JSONResponse:
    """``body`` as the answer of a Restart to update that stopped the writers and leaves nothing armed: they return
    through the saved-work owner first. Where they cannot, the answer says so and every client keeps it until a
    restart (``update_progress.retain_stopped_activation``), as chat keeps refusing writers until then."""
    from ouroboros.gateway import control

    if control._respawn_workers_after_failed_update():
        return JSONResponse(body, status_code=status)
    response = JSONResponse({**body, "error": f"{body['error']}; running work stays stopped until Ouroboros restarts",
                             "restart_required": True}, status_code=status)
    update_progress.retain_stopped_activation(response)
    return response


def _abort_activation(candidate_id: str, failure: Exception, *,
                      restart_uncertain: bool = False) -> tuple[JSONResponse, bool]:
    """Any other end of a Restart to update after its fence and before its restart: a failed step (an activation
    snapshot at its deadline, a helper copy the disk refused) or an installer or restart that cannot go on.

    Writers return only once this activation can no longer act. Under ``lifecycle()``, a record it already marked goes
    back to ``ready``, then its handoff closes and is read back: its record is all an installer already started acts
    on (``app_install``: only ``armed``), and all an exit tail arms; that close and ``arm`` hold the helper's lock, so
    the first is the one outcome. Whatever cannot be put back stays as it is for the next boot, which settles a handoff
    and its record together, and so does everything once a restart may be accepted (``restart_uncertain``: its exit
    tail owns this handoff): writers stay stopped until that restart, which either switches to this update or leaves it
    not applied. Returns the answer and whether a transition is still open.
    """
    from ouroboros import body_adoption
    from supervisor import update_prepare
    from supervisor.git_ops import DRIVE_ROOT

    if isinstance(failure, _ActivationRefused):
        what, status, fields = failure.text, failure.http_status, failure.fields
    elif isinstance(failure, update_prepare.PrepareRefused):  # this server's stop refused its next Git step, say
        what, status, fields = failure.text, 409, {"reason": failure.code}
    else:
        log.error("update activation failed after its writer fence", exc_info=failure)
        what, status = f"the activation failed: {type(failure).__name__}: {failure}", 500
        fields = {"reason": "activation_failed"}
    if restart_uncertain:
        still_open = "its restart may already have been accepted"
    else:
        with update_prepare.lifecycle():
            try:
                live = update_prepare.find(candidate_id)
                if live is not None and live.get("state") == "activating":
                    # Ready again, still naming this attempt: an earlier handoff's late boot result is not its own.
                    named = (live.get("activation") or {}).get("adoption_id")
                    update_prepare.mark(live, state="ready", activation={"adoption_id": named} if named else {})
                phase = body_adoption.close_update(DRIVE_ROOT, fields["reason"], candidate_id)
            except Exception as again:
                log.warning("update activation record could not be settled", exc_info=True)
                phase = f"{type(again).__name__}: {again}"
        still_open = f"it could not be closed ({phase})" if phase else ""
    if not still_open:
        body = {"error": f"{what}; nothing was switched", **fields, "outcome": "not_applied"}
        return _writers_back(body, status), False
    response = JSONResponse({"error": f"{what}; {still_open}, so running work stays stopped and the next restart "
                                      "either switches to this update or leaves it not applied",
                             "reason": "activation_unresolved", "restart_required": True}, status_code=500)
    update_progress.retain_stopped_activation(response, concluded=False)
    return response, True


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
    installs, fenced, signalling, restarting = None, False, False, False
    try:
        if active_update_tx():
            return JSONResponse({"error": "a legacy managed update is in progress", "outcome": "not_applied"},
                                status_code=409)
        fenced = True
        blockers = control._quiesce_repo_writers("activation")
        if blockers:
            refused = control._fence_failure(blockers)
            if json.loads(refused.body)["restart_required"]:
                # The fence keeps the writers refused: never respawned into what it could not prove, and every client
                # keeps the restart. Running work stopped only if the pool went down for an update; a refusal that
                # reopened the writers before any teardown stays this request's answer.
                from supervisor.workers import worker_pool_admission_state

                stopped = worker_pool_admission_state().get("disabled_reason") == "managed_update"
                update_progress.retain_stopped_activation(refused, stopped=stopped)
            return refused
        # The fence may have waited: a discard or a new preparation meanwhile changed what is current.
        # Everything from this re-read until the record says activating holds the lifecycle lock.
        with update_prepare.lifecycle():
            row = update_prepare.current()
            if row is None or row.get("candidate_id") != candidate_id or row.get("state") != "ready":
                return _writers_back({"error": "the prepared update changed while the writers stopped; nothing was "
                                               "switched", "reason": "candidate_changed", "outcome": "not_applied"})
            if (row.get("environment") or {}).get("generation"):
                # The fence stopped the task writers; the server's own package writer is the local model runtime's
                # install: held from the package check below until the restart switches them, or refused.
                from ouroboros.local_model import get_manager

                installs = get_manager()
                busy = installs.hold_runtime_install("an update is being activated; install it after the restart")
                if busy:
                    installs = None
                    return _writers_back({"error": f"{busy}; restart to update once it finished",
                                          "reason": "packages_busy", "outcome": "not_applied"})
            why = update_prepare.activation_check(row)
            if why:
                return _writers_back({"error": why, "reason": why.split(":", 1)[0], "outcome": "not_applied",
                                      "candidate": candidate_view()})
            try:
                handoff = body_adoption.authorize_update(DRIVE_ROOT, row)
            except body_adoption.AdoptionRefused as exc:
                return _writers_back({"error": exc.text, "reason": exc.code, "outcome": "not_applied"})
            if not body_adoption.bind_update_restart(DRIVE_ROOT):
                raise _ActivationRefused("the activation could not be bound to its restart",
                                         reason="restart_binding_failed")
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
                    raise _ActivationRefused(f"the new app could not be installed: {started['reason']}",
                                             reason="native_hand_over_failed")
            update_prepare.mark(row, state="activating", activation={"adoption_id": handoff["id"]})
        signalling = True  # a callback that raises may have signalled first
        requested = control._request_restart(request)
        signalling = False
        if not requested:
            raise _ActivationRefused("the restart could not be requested", 500, reason="restart_unavailable",
                                     status="restart_required")
        restarting = True  # the hold stays: the restart switches the packages
        update_progress.notify_changed()  # other clients and the indicator: this candidate is activating now
        return JSONResponse({"status": "ok", "restarting": True, "strategy": "activate",
                             "candidate": candidate_view()})
    except Exception as exc:
        if restarting or not fenced:  # that restart owns what follows; before the fence nothing stopped
            raise
        from ouroboros.server_process import _restart_requested

        # Still open: its restart may switch the packages, so the hold stays.
        response, restarting = _abort_activation(candidate_id, exc,
                                                 restart_uncertain=signalling or _restart_requested.is_set())
        return response
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
