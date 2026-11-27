"""The task API's id: memory and the pool asked, then claimed in the store, then held in memory (#4317).

``POST /api/tasks`` takes a caller's ``task_id`` or generates one. The task admission is asked first without a
hold or a store read (``task_admission.reserve_task_admission``), the canonical store then decides outside Q
(``task_id_claims.claim_provisional_id``: a record no reader of tasks sees), and only the winner of that claim
reserves the id in memory, before a drive, an attachment or an artifact may exist under it. A reservation never
stands over another record's id, where a helper's own event meeting it would be dropped as a replay. A failure
before anything is queued is rolled back here too (``roll_back_admission``).
"""

from __future__ import annotations

import logging
import pathlib
import uuid
from typing import Any

log = logging.getLogger(__name__)


def claim_task_admission(drive_root: Any, task_id: str, admission_token: str) -> tuple:
    """``(id, refusal)`` of the task API's one id: TASK_ID (the caller's) or a fresh 16-hex one, claimed in the
    order above. A request that loses leaves the other owner's record, paths and reservation alone. ``refusal`` is
    None or the ``(HTTP status, message, fields)`` the caller answers with. A generated id found held, or one that
    cannot be checked or was never claimed, is not named (it may be another record's); an unconfirmed claim names
    its attempted id with ``status: "unconfirmed"``, and whether this request holds it stays unknown."""
    from ouroboros.task_id_claims import (
        ClaimRefused, TaskIdTaken, TaskIdUnconfirmed, TaskIdUninspectable, TaskIdUnrepresentable, claim_failure_words,
        claim_id, claim_provisional_id, release_provisional_claim, retained_task_paths,
    )
    from supervisor import queue  # the API's admission seam, read at call time

    root, refused = pathlib.Path(drive_root), {}

    def ask(tid: str, *, hold: bool) -> None:
        answer = queue.reserve_task_admission(tid, admission_token, require_worker_pool=True, drive_root=root,
                                              stored_check=False, hold=hold)
        if answer.get("status") != "blocked":
            return
        refused.update(answer, task_id=tid)
        raise TaskIdTaken(tid, "reserved") if answer.get("reason") == "duplicate_task_id" else ClaimRefused()

    def write(tid: str) -> None:
        ask(tid, hold=False)
        claim_provisional_id(root, tid, admission_token)
        try:
            ask(tid, hold=True)
        except (TaskIdTaken, ClaimRefused):
            if not release_provisional_claim(root, tid, admission_token):
                raise RuntimeError("its claim could not be released after the reservation was refused")
            raise

    def release(tid: str) -> bool:
        if not release_provisional_claim(root, tid, admission_token):
            return False
        queue.release_task_admission(tid, admission_token)
        return True

    def duplicate(tid: str) -> tuple:  # a caller's id another admission or record holds: what its replay is answered
        return "", (409, "task admission refused: duplicate_task_id", {
            "task_id": tid, "reason_code": "duplicate_task_id", "worker_pool_disabled_reason": ""})

    named = {"task_id": task_id} if task_id else {}  # only the caller's own id is echoed for an id never claimed
    try:
        return claim_id(task_id, write, release, width=16, folders=lambda tid: retained_task_paths(root, tid))[0], None
    except TaskIdTaken as exc:
        if not task_id:
            return "", (503, "no free task id could be claimed", {"reason_code": "task_id_exhausted"})
        if exc.reason == "existing_files":
            return "", (409, f"task_id already has headless state: {task_id}", {})
        if exc.reason == "stop_pending":
            return "", (409, "task admission refused: a Stop of this task_id is pending",
                        {**named, "reason_code": "task_id_stop_pending"})
        if exc.reason in {"reserved", "claimed", "another_request"}:
            return duplicate(exc.task_id)
        return "", (409, f"task_id already exists: {task_id}", {})
    except TaskIdUnrepresentable as exc:
        return "", (400, str(exc), {"task_id": exc.task_id, "reason_code": "task_id_unrepresentable"})
    except TaskIdUninspectable as exc:  # nothing held or written: a caller's id is refused as one not checked
        if task_id:
            return "", (409, f"task_id already exists or could not be checked; nothing was changed: {task_id}",
                        {**named, "reason_code": "task_id_inspection_failed", "detail": exc.detail})
        return "", (503, f"task admission refused: {claim_failure_words(exc)}",
                    {"reason_code": "task_id_inspection_failed"})
    except ClaimRefused:
        reason = str(refused.get("reason") or "admission_reservation_failed")
        return "", (503 if reason.startswith("worker_pool_") else 409, f"task admission refused: {reason}", {
            **named, "reason_code": reason,
            "worker_pool_disabled_reason": str(refused.get("worker_pool_disabled_reason") or "")})
    except TaskIdUnconfirmed as exc:
        return "", (503, "Task id claim is unconfirmed: whether this request holds the task_id is unknown. Nothing was "
                    "queued, and a record read under it is not proof it is this request's.",
                    {"task_id": exc.task_id, "status": "unconfirmed"})


def receipt_lands_on(current: Any, admission_token: str) -> bool:
    """Whether this admission's receipt may land on CURRENT, the row under its id as its own write transaction finds
    it: none yet (``land_api_receipt`` creates it only while this admission's claim holds the id), its own receipt (a
    replay), or the row its task's own first launch or Stop wrote, which carries the token the queue kept for it
    (``_admission_owner_token``: the launch's dispatch marks and mirror and a queued task's cancel write it; Q is
    released before the receipt, so any of them may write first). A row without that token proves nothing of this
    admission and is never claimed by it."""
    if not current:
        return True
    held = current.get("api_admission")
    return bool(admission_token) and (current.get("_admission_owner_token") == admission_token or (
        isinstance(held, dict) and held.get("token") == admission_token))


def land_api_receipt(drive_root: Any, task_id: str, admission_token: str, fields: Any, write: Any = None) -> None:
    """Write the API admission's receipt (``fields``, its ``api_admission`` token among them) outside Q, in one store
    write that decides on the record it finds (``receipt_lands_on``): a new row only while this admission's own
    provisional claim holds the id, else only its own record, whose status and words stand (a Stop's ``cancelled``,
    a launch's ``assigned``), adding the token and the facts it lacks; a row its dispatch marks made before any
    status (``requested``) reads ``scheduled``. Raises unless that exact receipt is the record under the id: unknown
    is never success, and an unproven record is left as it is. WRITE is the caller's ``write_task_result``."""
    from ouroboros.task_id_claims import ID_CLAIMS
    from ouroboros.task_results import STATUS_REQUESTED, STATUS_SCHEDULED, load_task_result
    from ouroboros.task_results import write_task_result as write_row

    write_task_result = write or write_row  # one writer by its name: the terminal-writer inventory reads this call

    def mine(now: Any, new: Any) -> Any:  # read inside the write transaction
        if not receipt_lands_on(now, admission_token):
            return None
        status = str(now.get("status") or "")
        return {**{key: value for key, value in new.items() if key == "api_admission" or key not in now},
                "status": status if status not in {"", STATUS_REQUESTED} else STATUS_SCHEDULED}

    def claimed(txn: Any, current: Any, document: Any) -> Any:  # a new row only under this admission's own claim
        return document if current or (txn.get_row(ID_CLAIMS, task_id) or {}).get("token") == admission_token else None

    write_task_result(drive_root, task_id, STATUS_SCHEDULED, **fields, _field_projector=mine, _last_word=claimed)
    if (load_task_result(drive_root, task_id, strict=True) or {}).get("api_admission") != fields["api_admission"]:
        raise RuntimeError("the record under the task id is not proven this admission's own; nothing was added")


def admission_claimed(drive_root: Any, task_id: str) -> Any:
    """Whether a ``POST /api/tasks`` admission holds TASK_ID, which a Stop found neither in a row nor live: True by
    its provisional claim alone, before any row or queue entry (a single Stop records its intent there and fences
    that admission, never 404); ``"queued"`` when the live queue, read again after no claim stood, holds the task (a
    claim goes only once its task is queued and its receipt landed: the Stop runs as asked); else False. None when the
    store did not answer: unknown is never absence."""
    from ouroboros.task_id_claims import provisional_claim
    from supervisor import queue

    try:
        if provisional_claim(drive_root, task_id) is not None:
            return True
    except Exception:
        log.warning("The provisional claim of %s could not be read for a Stop", task_id, exc_info=True)
        return None
    return "queued" if queue.task_subtree_is_live(task_id) else False


def make_staging_folders(drive_root: Any, task_id: str, *, drive: bool, artifacts: bool) -> tuple:
    """The folders this admission's staging writes into, made by it alone before anything lands there
    (``task_id_claims.make_own_folders``): its task state when it prepares a drive of its own, the canonical
    artifact folder when anything is written there (the workspace preflight, attachments without a drive).
    ``roll_back_admission`` removes only these, and only while they hold nothing. ``OSError`` when one already
    stands (another writer's) or cannot be made: nothing is then made or touched."""
    from ouroboros.task_id_claims import make_own_folders, retained_task_paths

    state, artifacts_dir = retained_task_paths(drive_root, task_id)[:2]
    return make_own_folders(((state,) if drive else ()) + ((artifacts_dir,) if artifacts else ()))


STAGING_RETAINED = ("Nothing was queued. What it staged stays under this task_id with its failed record; sending "
                    "the task again takes another task_id.")


_ADMITTED_HANDOFF_FIELDS = (
    "chat_id", "title", "suggested_name", "description", "project_id",
    "workspace_root", "workspace_mode", "memory_mode", "child_drive_root",
    "budget_drive_root", "delegation_role", "parent_task_id", "root_task_id",
    "session_id", "actor_id", "metadata", "attachment_manifest", "attachment_manifest_ref",
)


def _handoff_custody_free(task_id: str, admission_token: str) -> bool:
    """Memory custody check under Q (always released before any store or cancel lock)."""
    from supervisor import queue

    with queue._queue_lock:
        reservation = queue.ADMISSION_RESERVATIONS.get(task_id)
        return not ((reservation and reservation != admission_token) or task_id in queue.RUNNING
                    or any(row.get("id") == task_id for row in queue.PENDING if isinstance(row, dict)))


def record_admitted_handoff_for_stop(drive_root: Any, task_id: str, admission_token: str, *,
                                     saved_task: Any = None) -> bool:
    """Materialize a dead-producer hand-off covered by an active Stop as ``STATUS_SCHEDULED`` so the existing
    Stop path finishes it with its usual ``cancelled`` outcome (never ``never_admitted``). Idempotent on
    contended post-commit readback; returns False if custody or the Stop/claim preconditions no longer hold."""
    from ouroboros._usage_wait import ACQUISITION_SLICE_SEC
    from ouroboros.cancel_intents import cancellation_projection_lock, has_active_intent
    from ouroboros.store import acquisition_budget
    from ouroboros.task_id_claims import ID_CLAIMS
    from ouroboros.task_results import STATUS_SCHEDULED, load_task_result, write_task_result
    from supervisor import queue
    from supervisor.workspace_preparation import store_contention

    root = pathlib.Path(drive_root)
    if not _handoff_custody_free(task_id, admission_token):
        return False
    fields: dict[str, Any] = {
        "_admission_owner_token": admission_token,
        "api_admission": {"token": admission_token, "status": "accepted"},
        "strict_existing_dict": True,
    }
    if isinstance(saved_task, dict) and saved_task.get("_admission_owner_token") in (None, "", admission_token):
        for key in _ADMITTED_HANDOFF_FIELDS:
            if key in saved_task and saved_task[key] is not None:
                fields[key] = saved_task[key]

    def absent_only(current: Any, incoming: Any) -> Any:
        return incoming if not current else None

    def claimed(txn: Any, current: Any, document: Any) -> Any:
        return document if not current and (txn.get_row(ID_CLAIMS, task_id) or {}).get("token") == admission_token else None

    with cancellation_projection_lock(root), acquisition_budget(ACQUISITION_SLICE_SEC):
        if not has_active_intent(root, task_id, strict=True):
            return False
        current = load_task_result(root, task_id, strict=True) or {}
        if current:
            return receipt_lands_on(current, admission_token)
        try:
            write_task_result(root, task_id, STATUS_SCHEDULED, _field_projector=absent_only, _last_word=claimed, **fields)
        except Exception as exc:
            if store_contention(exc):
                raise
            log.warning("Admitted hand-off write raised for %s; checking readback", task_id, exc_info=True)
        stored = load_task_result(root, task_id, strict=True) or {}
        if not (stored and receipt_lands_on(stored, admission_token)):
            return False
    queue.release_task_admission(task_id, admission_token)
    return True


def record_unknown_handoff(drive_root: Any, task_id: str, admission_token: str, *,
                           refusal_token: str = "", staging_retained: bool = False) -> bool:
    """Record an honest unknown-admission failure when the original saved queue snapshot was unreadable at startup:
    keeps any staged files and keeps the task_id owned without writing ``never_admitted``. Idempotent on contended
    post-commit readback; yields to an active Stop."""
    from ouroboros._usage_wait import ACQUISITION_SLICE_SEC
    from ouroboros.cancel_intents import cancellation_projection_lock, has_active_intent
    from ouroboros.store import acquisition_budget
    from ouroboros.task_id_claims import ID_CLAIMS
    from ouroboros.task_results import STATUS_FAILED, load_task_result, write_task_result
    from supervisor import queue
    from supervisor.workspace_preparation import store_contention

    root = pathlib.Path(drive_root)
    receipt = str(refusal_token or "") or uuid.uuid4().hex
    if not _handoff_custody_free(task_id, admission_token):
        return False

    def absent_only(current: Any, incoming: Any) -> Any:
        return incoming if not current else None

    def claimed(txn: Any, current: Any, document: Any) -> Any:
        return document if not current and (txn.get_row(ID_CLAIMS, task_id) or {}).get("token") == admission_token else None

    fields: dict[str, Any] = {
        "_admission_owner_token": admission_token,
        "_admission_refusal_token": receipt,
        "reason_code": "saved_queue_unreadable",
        "admission_outcome": "unknown",
        "admission": {"reason_code": "saved_queue_unreadable", "outcome": "unknown"},
        "result": (
            "Task admission outcome is unknown: the saved queue snapshot was unreadable at startup, so whether "
            "this task had been queued could not be established. Any staged files are kept and this task_id "
            "remains owned."
        ),
        "accounted_upper_bound_usd": 0.0,
        "strict_existing_dict": True,
    }
    if staging_retained:
        fields["admission_cleanup"] = {"staging_retained": True}

    with cancellation_projection_lock(root), acquisition_budget(ACQUISITION_SLICE_SEC):
        if has_active_intent(root, task_id, strict=True):
            return False
        current = load_task_result(root, task_id, strict=True) or {}
        if current:
            return bool(admission_token and current.get("status") == STATUS_FAILED
                        and current.get("_admission_owner_token") == admission_token
                        and current.get("_admission_refusal_token") == receipt)
        try:
            write_task_result(root, task_id, STATUS_FAILED, _field_projector=absent_only, _last_word=claimed, **fields)
        except Exception as exc:
            if store_contention(exc):
                raise
            log.warning("Unknown hand-off write raised for %s; checking readback", task_id, exc_info=True)
        stored = load_task_result(root, task_id, strict=True) or {}
        if stored.get("status") != STATUS_FAILED or stored.get("_admission_refusal_token") != receipt:
            return False
    queue.release_task_admission(task_id, admission_token)
    return True


def record_unsettled_handoff(drive_root: Any, task_id: str, admission_token: str, *,
                             refusal_token: str = "") -> Any:
    """At startup, a hand-off whose server ended, which the recovered queue (read whole before startup publication)
    does not hold, which has no active Stop, and which left staging (``task_id_claims.retire_settled_claims``):
    nothing of it can run. Its never-admitted refusal is written onto no row but its own
    (``task_admission.persist_never_admitted_refusal``, which raises unless that exact refusal is the row under the
    id then), interlocked against Stop under the cancel-intent lock and guarded by the standing claim; the staging
    stays retained under it for that row's retention, nothing is removed."""
    from ouroboros.cancel_intents import cancellation_projection_lock, has_active_intent
    from ouroboros.task_id_claims import ID_CLAIMS
    from supervisor.task_admission import persist_never_admitted_refusal
    from supervisor.workspace_preparation import INTERRUPTED_BY_RESTART

    root = pathlib.Path(drive_root)
    receipt = str(refusal_token or "") or uuid.uuid4().hex

    def claimed(txn: Any, current: Any, document: Any) -> Any:
        return document if current or (txn.get_row(ID_CLAIMS, task_id) or {}).get("token") == admission_token else None

    return persist_never_admitted_refusal(
        root, task_id, admission_token=admission_token, reason_code=INTERRUPTED_BY_RESTART,
        admission={"reason_code": INTERRUPTED_BY_RESTART}, admission_cleanup={"staging_retained": True},
        result=f"Task was not scheduled: its server ended before its admission settled. {STAGING_RETAINED}",
        accounted_upper_bound_usd=0.0, _admission_refusal_token=receipt,
        _guard=lambda: cancellation_projection_lock(root),
        _only_if=lambda current: not current and not has_active_intent(root, task_id, strict=True),
        _last_word=claimed)


def roll_back_admission(drive_root: Any, task_id: str, admission_token: str, *, own: Any = (),
                        refused: bool = False, caller_named: bool = True, cause: Any = "",
                        reason_code: str = "admission_preparation_failed") -> Any:
    """Roll back a known pre-enqueue failure while its reservation excludes rivals: None once nothing of the id is
    queued, no record holds it, the folders this attempt made for its staging (``own``: made by exclusive creation,
    ``task_id_claims.make_own_folders``, never inferred from the id) are gone and its claim, then its reservation,
    are released, so the same id may be sent again. Of those folders only what holds nothing goes
    (``task_id_claims.remove_own_paths``): making one proves who made it, never whose its files are. One that holds
    any file (or that the owner's Keep holds) stays with all of it, and the proven refusal is recorded instead:
    while this attempt's own claim still stands, its token-checked never-admitted row (``reason_code``,
    ``task_admission.persist_never_admitted_refusal``, written again while only a busy store refused it:
    ``task_id_claims.while_contended``) takes the id and leaves the files to that row's retention; then its claim
    goes and, once that row is confirmed, its reservation. The answer is then the facts the caller adds to its own
    refusal (``staging_retained``). A folder under the id this attempt did not make is another writer's and stays as
    it is, and so does a record found under it (this attempt wrote none): once this attempt's own part is undone its
    claim and reservation alone are released too, and the answer is that the id is occupied (409
    ``task_id_occupied``; a generated id is not named). A pending Stop does not keep them: what it would stop is this
    attempt, which is undone, and the Stop still takes the id from the next claim until it settles. Anything unproven
    (a claim not standing, a refusal not confirmed) keeps claim and reservation held. ``refused``: the queue's typed
    refusal already released the reservation, which the rollback takes back while it runs. ``cause`` is the failure
    being answered. Otherwise ``(HTTP status, body)``."""
    from ouroboros.owner_keep import gate
    from ouroboros.task_id_claims import (
        present, release_provisional_claim, remove_own_paths, retained_task_paths, while_contended,
    )
    from ouroboros.task_results import load_task_result
    from supervisor import queue

    root, own = pathlib.Path(drive_root), tuple(own)
    words = f"{type(cause).__name__}: {cause}" if isinstance(cause, BaseException) else str(cause)
    detail = {"detail": words} if words else {}
    named = {"task_id": task_id} if caller_named else {}

    def kept(why: str) -> tuple:
        return 503, {"task_id": task_id, "status": "unconfirmed", "error": (
            f"Task admission was not completed and its rollback is unconfirmed ({why}). This request queued nothing; "
            "whether it still holds the task_id is unknown, so a record read under it is not proof it is this "
            "request's."), **detail}

    with queue._queue_lock:  # memory only; the held reservation and claim exclude rivals after it
        queued = task_id in queue.RUNNING or any(row.get("id") == task_id for row in queue.PENDING)
        if refused and not queued and queue.ADMISSION_RESERVATIONS.get(task_id) is None:
            queue.ADMISSION_RESERVATIONS[task_id] = admission_token
        if queued or queue.ADMISSION_RESERVATIONS.get(task_id) != admission_token:
            return kept("another admission or a queued task holds the id")
    # Keep the reservation through cleanup, but not Q through drive preparation.
    try:
        state, artifacts, direct = retained_task_paths(root, task_id)
        if load_task_result(root, task_id, strict=True) is not None:  # the id is that record's: this attempt's
            release_provisional_claim(root, task_id, admission_token)  # own claim and reservation alone go
            queue.release_task_admission(task_id, admission_token)
            return 409, {"error": "Task admission was not completed: a record under the task id is not proven this "
                         "attempt's; nothing was removed.", "status": "rejected", "reason_code": "task_id_occupied",
                         **named, **detail}
        foreign = [path for path in (state, artifacts, direct) if path not in own and present(path)]
        with gate(root, task_id, timeout_sec=0) as held:  # the owner's keep holds what it staged too
            gone = not (held and own) and remove_own_paths(own)  # a partial drive too: only what holds nothing goes
        if not gone:  # what it staged stays: its proven refusal takes the id, for that row's ordinary retention
            from ouroboros.task_id_claims import ID_CLAIMS, provisional_claim, retire_provisional_claim
            from supervisor.task_admission import persist_never_admitted_refusal

            if (provisional_claim(root, task_id) or {}).get("token") != admission_token:
                return kept("its claim is not the one standing")
            refusal_token = uuid.uuid4().hex

            def claimed(txn: Any, current: Any, document: Any) -> Any:
                return document if current or (txn.get_row(ID_CLAIMS, task_id) or {}).get("token") == admission_token else None

            while_contended(lambda: persist_never_admitted_refusal(  # raises unless this exact refusal is the row
                root, task_id, admission_token=admission_token, reason_code=reason_code,
                admission={"reason_code": reason_code}, admission_cleanup={"staging_retained": True},
                result=f"Task was not scheduled ({words or reason_code}). {STAGING_RETAINED}",
                accounted_upper_bound_usd=0.0, _admission_refusal_token=refusal_token, _last_word=claimed))
            retire_provisional_claim(root, task_id, admission_token)
            queue.release_task_admission(task_id, admission_token)
            return {"task_id": task_id, "staging_retained": True}
        if not release_provisional_claim(root, task_id, admission_token):
            return kept("its claim could not be released")
    except Exception as exc:
        log.warning("Failed to clean the admission attempt of task %s", task_id, exc_info=True)
        return kept(type(exc).__name__)
    queue.release_task_admission(task_id, admission_token)
    if foreign:  # its own part is undone; what holds the id now is not this request's to remove
        return 409, {"error": "Task admission was not completed: a folder this request did not make stands under the "
                     "task id; nothing of it was removed and nothing was queued.", "status": "rejected",
                     "reason_code": "task_id_occupied", **named, **detail}
    return None
