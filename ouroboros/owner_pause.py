"""The owner's Pause of a whole task tree: its durable fence and launch gate.

Owner Batch4 (5A/7A), amended 2026-10-08 (#1562, full variant): Pause fences
NEW effects of every member of one root's tree; operations already handed to an
executor finish; the member's OWN delegated runs get a concrete stop
(``external_runs`` policy ``stop_task_owned``) while reviewers that already
launched finish their same check (``review_episode``, armed only at the
attempt's actual handoff under this module's launch lock); each member parks
warm (owner wait) or as the exact same-ID pause the budget rail already writes
(``budget_pause``, reason ``owner``). Nothing here is a scheduler, a ledger or
a second pause store:

- The FENCE is one projection on the ROOT's task result (``owner_pause``),
  written by the supervisor's accept step through the task result's own
  locked read-modify-write (atomic replace), and released by the root's
  explicit Resume. Every member reads the same file: the queue's admission
  fence keeps new descendants from being admitted or assigned, and this
  durable projection is what a member that is ALREADY running consults at
  each launch handoff.
- Preparation and durable claims precede confirmed executor submission,
  serialized with Pause by a short per-root lock. No lock spans body/network
  completion. Claims and started operations are custody, not wire receipts;
  a crash never proves success. Nested/queued transports take their final gate
  when ready, and returned invocations cannot acquire a late background start.
- A member's SAFE BOUNDARY (the loop's round drain, a pre-dispatch model wait,
  a warm owner wait woken by the ``owner_pause`` mailbox control) reads the
  same fence and enters the exact pause without buying a model round.

The fence being closed is NOT the tree being Paused: the root's ``state``
turns ``paused`` only when no member of the tree still runs and every parked
member's own sent work (delegated runs included) has settled; reviewers the
Pause lets finish never keep it pausing (the fence lists them as
``finishing_reviews``). Until then the truthful state is ``requested``.
"""

from __future__ import annotations

import logging
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar, copy_context
import pathlib
import threading
import time
import uuid
from typing import Any, Dict, Iterable, Optional, Tuple

from ouroboros.store import StoreBusy, StoreError, acquisition_budget, admission_acquisition, physical_acquisition

log = logging.getLogger(__name__)

_TOOL_OPERATION = ContextVar("owner_pause_tool_operation", default=None)
_HANDED_TOOL = ContextVar("owner_pause_handed_tool", default=None)
_HANDED_MODEL = ContextVar("owner_pause_handed_model", default="")
# One PHYSICAL review attempt's completion allowance (``review_episode``):
# armed only by that attempt's first launch the fence admitted.
_REVIEW_EPISODE = ContextVar("owner_pause_review_episode", default=None)
_REVIEW_ADMISSION = ContextVar("owner_pause_review_admission", default=None)

RAIL_OWNER_PAUSE = "owner_pause"
REASON_OWNER = "owner"

FENCE_REQUESTED = "requested"
FENCE_PAUSED = "paused"
FENCE_RELEASED = "released"
CLOSED_FENCE_STATES = frozenset({FENCE_REQUESTED, FENCE_PAUSED})

# The saved pause of a member whose sent delegated work had not settled yet: a
# checkpoint exists, but the member is not cleanly Paused (never a false Paused).
SETTLEMENT_EXTERNAL_RUNNING = "external_writers_running"
SETTLEMENT_SETTLED = "settled"

NOT_STARTED_TEXT = ("⚠️ OWNER_PAUSE_NOT_STARTED: NOT STARTED — the owner paused this task tree before this "
                    "operation was launched. Nothing ran; it may be issued again after Resume.")


class OwnerPauseRefused(Exception):
    """Pause authority refused fence installation or a new launch."""


class LaunchBusy(OwnerPauseRefused):
    """A root submission lock could not be acquired; the physical reason is retained."""

    def __init__(self, reason="unknown"):
        super().__init__("owner_launch_authority_unavailable")
        self.reason = reason


# --- the durable fence --------------------------------------------------------------

_CACHE_LOCK = threading.Lock()
_CACHE: Dict[str, bool] = {}  # observed authority only, never cached control values


def _lock_path(root_drive: Any, root_task_id: str) -> pathlib.Path:
    from ouroboros.task_results import task_results_dir

    return task_results_dir(pathlib.Path(root_drive), create=True) / f"{root_task_id}.launch.lock"


def _control_projection(root_drive: Any, root_task_id: str) -> Dict[str, Any]:
    """One addressed control read; an unknown fence is never an open one."""
    from ouroboros.store_hot_facts import controls

    root_task_id = str(root_task_id or "").strip()
    if not root_task_id or root_drive in (None, ""):
        return {}
    key = f"{root_drive}:{root_task_id}"
    try:
        projection = controls(root_drive, root_task_id)
    except StoreBusy:
        raise
    except StoreError as exc:
        raise ValueError("owner_pause_authority_unreadable") from exc  # the store could not answer: unknown, never open
    if not projection:
        from ouroboros.task_result_schema import TASK_RESULT_QUARANTINE_DIR
        from ouroboros.task_results import task_results_dir

        quarantine = task_results_dir(pathlib.Path(root_drive), create=False) / TASK_RESULT_QUARANTINE_DIR
        with _CACHE_LOCK:
            prior = _CACHE.get(key)
        if prior or (quarantine / f"{root_task_id}.json").exists() or any(quarantine.glob(f"{root_task_id}.*.json")):
            raise ValueError("owner_pause_authority_missing")  # a row that was, or was refused: never an open fence
        return {}
    # Remember only that this process saw authority, preserving the missing-row
    # refusal. Current controls always come from the store or the request snapshot.
    with _CACHE_LOCK:
        _CACHE[key] = True
    return projection


def read_fence(root_drive: Any, root_task_id: str, *, _observations=None) -> Dict[str, Any]:
    """Current owner Pause; the same memo also carries admission's small sleep projection."""
    if _observations is None:
        projection = _control_projection(root_drive, root_task_id)
    else:
        if root_task_id not in _observations:
            _observations[root_task_id] = _control_projection(root_drive, root_task_id)
        projection = _observations[root_task_id]
    fence = projection.get("owner_pause") or {}
    if not isinstance(fence, dict) or (fence and fence.get("state") not in
                                       {FENCE_REQUESTED, FENCE_PAUSED, FENCE_RELEASED}):
        raise ValueError("owner_pause_authority_unreadable")
    return dict(fence)


def fence_closed(fence: Dict[str, Any]) -> bool:
    return str((fence or {}).get("state") or "") in CLOSED_FENCE_STATES


def paused_gap(root_drive: Any, task_id: str) -> Dict[str, Any]:
    """``{"membership_gaps": ...}`` the task's still-closed Pause last named, as its Resume reads it,
    else ``{}``: the fence's current fact, never the episode's older notice. Never raises."""
    try:
        fence = read_fence(root_drive, task_id)
    except Exception:
        return {}
    gaps = fence.get("membership_gaps") if fence_closed(fence) else None
    return {"membership_gaps": dict(gaps)} if isinstance(gaps, dict) and gaps.get("count") else {}


def root_actor(root_drive: Any, root_task_id: str, task_id: str, *, strict: bool = False) -> bool:
    """Whether ``task_id`` acts for ``root_task_id``'s whole tree (#1730).

    The root itself, or the CURRENT physical attempt its reciprocal timeout-retry
    chain ends at (``cancel_intents._validated_single_cancel_target``: typed root
    attempts, host-written forward links). A descendant, a superseded attempt or
    an unreadable chain is not; lineage ids alone grant nothing. The tree's
    fence and latch stay on the logical root; grants stay on physical rows.
    A grant issuer uses ``strict``: unreadable identity must refuse the action,
    never select member authority instead of root authority.
    """
    task_id, root_task_id = str(task_id or ""), str(root_task_id or "")
    if not task_id or not root_task_id or task_id == root_task_id:
        return bool(task_id) and task_id == root_task_id
    from ouroboros.cancel_intents import _validated_single_cancel_target
    from ouroboros.store_hot_facts import read

    try:
        row = read(root_drive, root_task_id) or {}
        if not row.get("superseded_by") and not row.get("retry_task_id"):
            return False
        return _validated_single_cancel_target(root_drive, root_task_id) == task_id
    except Exception:
        if strict:
            raise
        log.warning("Retry lineage of %s is unreadable; %s is not its root actor", root_task_id, task_id,
                    exc_info=True)
        return False


def _resume_outstanding(current: Dict[str, Any], fence: Dict[str, Any]) -> bool:
    """The physical root's unrevoked Resume grant names this logical fence.

    The fence stays closed from the grant until the resumed worker consumes it
    (``reopen_for_resume``); that grant would release whatever shares its fence.
    """
    from ouroboros.budget_pause import STATE_RESUME_GRANTED, STATE_RESUMED

    row = current.get("budget_pause") if isinstance(current.get("budget_pause"), dict) else {}
    grant = row.get("grant") if isinstance(row.get("grant"), dict) else {}
    return bool(fence.get("fence_id") and grant.get("owner_pause_fence_id") == fence["fence_id"]
                and not grant.get("revoked_at") and row.get("state") in {STATE_RESUME_GRANTED, STATE_RESUMED})


def install_fence(root_drive: Any, root_task_id: str, *, request_id: str,
                  late_work: bool = False, requested_by: str = "owner",
                  actor_id: str = "", by: Optional[dict] = None) -> Tuple[Dict[str, Any], bool]:
    """Close the root's fence durably; ``(fence, created)``.

    Idempotent by ``request_id`` across Resume and later Pause generations;
    every acknowledged action stays on this same root projection. A
    second press while closed names that fence, never an additional Pause,
    unless outstanding Resume authority names it: then this fresh Pause gets
    its own fence identity, which that grant cannot release. A
    retry's ``actor_id`` is revalidated under the logical launch lock; its
    physical row carries the grant, while the fence stays on the logical root. A
    terminal root refuses: a finished tree has nothing to pause — unless its
    answered task still owns late work: an open post-task checkpoint read
    under this lock (the Pause ends that upkeep at once), or a live review
    operation or live descendants the caller attests (``late_work``). ``by`` is the author a press
    declared (``control_actor.pause_author``): the fence keeps its first one,
    and a press that declared nobody claims none. Raises
    ``OwnerPauseRefused`` without any write on refusal; any write failure
    propagates, so the caller never acknowledges an undurable Pause.
    """
    from ouroboros.post_task_checkpoint import post_task_synthesis_is_open
    from ouroboros.task_results import (
        _TRULY_TERMINAL_STATUSES, load_task_result, require_writable_task_result_schema, stamp_task_result_schema,
    )
    from ouroboros.task_store import patch_document
    from ouroboros.utils import utc_now_iso

    from ouroboros.control_actor import control_actor, pause_author
    author = {**control_actor(), **(by or {})}
    attribution = {"by": author} if author else {}
    outcome: Dict[str, Any] = {}

    def update(current: dict) -> Optional[dict]:
        if not current:
            raise OwnerPauseRefused("root_result_missing")
        require_writable_task_result_schema(current)
        old = current.get("owner_pause") if isinstance(current.get("owner_pause"), dict) else {}
        requests = dict(old.get("requests") or {})
        if old.get("request_id"):
            requests.setdefault(old["request_id"], {
                "fence_id": old.get("fence_id"), "generation": old.get("generation")})
        prior = requests.get(request_id)
        if prior:
            replay = dict(old) if prior.get("fence_id") == old.get("fence_id") else {
                **prior, "request_id": request_id, "root_task_id": str(root_task_id),
                "state": FENCE_RELEASED}
            outcome.update(fence=replay, created=False)
            return None
        checkpoint = current.get("root_phase_checkpoint")
        late_open = isinstance(checkpoint, dict) and post_task_synthesis_is_open(checkpoint.get("post_task_synthesis"))
        if current.get("status") in _TRULY_TERMINAL_STATUSES and not (late_work or late_open):
            raise OwnerPauseRefused("task_terminal")
        if fence_closed(old) and not _resume_outstanding(actor_current if actor_current is not None else current, old):
            requests[request_id] = {**attribution, "fence_id": old["fence_id"], "generation": old.get("generation")}
            fence = {**old, **({} if pause_author(old)["actor"] else attribution), "requests": requests}
            outcome.update(fence=fence, created=False)
            return stamp_task_result_schema({**current, "owner_pause": fence})
        fence = {
            **attribution, "fence_id": uuid.uuid4().hex, "request_id": str(request_id or ""),
            # ``owner``: this owner Pause control, whoever pressed it (``by`` is who declared the press);
            # a stop door names itself (``server_shutdown``, ``owner_restart``): the next start reads it.
            "state": FENCE_REQUESTED, "requested_by": str(requested_by or "owner"), "requested_at": utc_now_iso(),
            "requested_at_ts": time.time(), "root_task_id": str(root_task_id),
            "generation": int(old.get("generation") or 0) + 1,
            # Still closed here only under outstanding Resume authority: name it.
            **({"supersedes_fence_id": old["fence_id"]} if fence_closed(old) else {}),
        }
        requests[request_id] = {**attribution, "fence_id": fence["fence_id"], "generation": fence["generation"]}
        fence["requests"] = requests
        outcome.update(fence=fence, created=True)
        return stamp_task_result_schema({**current, "owner_pause": fence})

    from ouroboros.store import STORE_WAIT_SEC
    from ouroboros._usage_wait import ACQUISITION_SLICE_SEC

    started = time.monotonic()
    while True:
        try:
            with launch_lock(root_drive, root_task_id, wait=False), acquisition_budget(0):
                actor_current = None
                if actor_id and actor_id != root_task_id:
                    if not root_actor(root_drive, root_task_id, actor_id):
                        raise OwnerPauseRefused("retry_lineage_unverified")
                    actor_current = load_task_result(pathlib.Path(root_drive), actor_id, strict=True)
                    if not actor_current:
                        raise OwnerPauseRefused("retry_lineage_unverified")
                patch_document(pathlib.Path(root_drive), str(root_task_id), update)
            break
        except (StoreBusy, LaunchBusy) as exc:
            remaining = STORE_WAIT_SEC - (time.monotonic() - started)
            if (exc.reason != "contention" or remaining <= 0
                    or isinstance(exc, StoreBusy) and not getattr(exc, "rollback_confirmed", False)):
                raise
            # This is the owner's request, not a model turn. Its idempotent write
            # retries within the existing physical store budget, BEFORE acknowledgement.
            time.sleep(min(ACQUISITION_SLICE_SEC, remaining))
    return dict(outcome["fence"]), bool(outcome["created"])


def set_fence_state(root_drive: Any, root_task_id: str, *, fence_id: str, state: str,
                    expected_state: str = "", **fields: Any) -> Dict[str, Any]:
    """Compare-and-set the fence's state on its own ``fence_id`` (never another Pause's).

    ``expected_state`` also binds the state its caller read: a Resume that
    released this same fence after that read refuses a stale write here.
    """
    from ouroboros.task_results import (
        require_writable_task_result_schema, stamp_task_result_schema,
    )
    from ouroboros.utils import utc_now_iso

    written: Dict[str, Any] = {}

    def update(current: dict) -> Optional[dict]:
        require_writable_task_result_schema(current)
        old = current.get("owner_pause") if isinstance(current.get("owner_pause"), dict) else {}
        if str(old.get("fence_id") or "") != str(fence_id or ""):
            raise ValueError("owner pause fence identity changed")
        if expected_state and str(old.get("state") or "") != expected_state:
            raise ValueError("owner pause fence state changed")
        if str(old.get("state") or "") == state and not fields:
            written.update(old)
            return None
        fence = {**old, **fields, "state": state, f"{state}_at": utc_now_iso()}
        written.update(fence)
        from ouroboros.pause_notices import notice_fields
        notice = (notice_fields(current, root_drive, root_task_id, fence_id, "owner", fence=fence)
                  if state == FENCE_PAUSED and old.get("state") != FENCE_PAUSED else {})
        return stamp_task_result_schema({**current, **notice, "owner_pause": fence})

    from ouroboros.task_store import patch_document
    patch_document(pathlib.Path(root_drive), str(root_task_id), update)
    return dict(written)


def release_fence(root_drive: Any, root_task_id: str, *, reason: str) -> Dict[str, Any]:
    """Reopen the root's fence for an explicit Resume (a no-op when already open).

    Called where the resumed ROOT actually starts: the worker consuming its
    exact grant, or the queue selecting a never-started root. A grant revoked
    before it ran (a restart, a Stop) therefore leaves the tree fenced.
    """
    fence = read_fence(root_drive, root_task_id)
    if not fence_closed(fence):
        return fence
    return set_fence_state(root_drive, root_task_id, fence_id=str(fence.get("fence_id") or ""),
                           state=FENCE_RELEASED, release_reason=str(reason or "owner_resume"))


def reopen_for_resume(root_drive: Any, root_task_id: str, task_id: str, *,
                      fence_id: str, grant_id: str) -> None:
    """Consume one explicit Resume's fence authority where its work starts.

    A root (or the current attempt of its retry, ``root_actor``) reopens its
    tree; an owner-selected child of a terminal root opens only its own member.
    A closed fence that ``install_fence`` minted over this
    outstanding Resume (``supersedes_fence_id``) is a newer owner Pause: the
    Resume is superseded, releases nothing, and the member parks at its next
    safe boundary. Any other identity change refuses.
    """
    publication_gap = None
    with launch_lock(root_drive, root_task_id), acquisition_budget(0):
        current = read_fence(root_drive, root_task_id) if fence_id else {}
        if fence_id and current.get("fence_id") != fence_id:
            if not fence_closed(current) or current.get("supersedes_fence_id") != fence_id:
                raise ValueError("owner_pause_fence_changed")
        elif root_actor(root_drive, root_task_id, task_id):
            try:
                release_fence(root_drive, root_task_id, reason="owner_resume_consumed")
            except StoreBusy as exc:
                publication_gap = exc
        else:
            select_member_resume(root_drive, root_task_id, task_id, fence_id=fence_id, grant_id=grant_id)
    if publication_gap is not None:
        _finish_resume_publication(root_drive, root_task_id, task_id, fence_id, grant_id, publication_gap)


def _finish_resume_publication(root_drive: Any, root_task_id: str, task_id: str, fence_id: str, grant_id: str,
                               error: StoreBusy | LaunchBusy) -> None:
    """Retry only the root fence write, outside its lock.

    Nothing the Resume already started restarts here. Each try
    rechecks the physical actor's grant under its logical root lock; unknown
    rollback or the existing STORE_WAIT_SEC physical acquisition bound leaves
    the caller's hold intact; no task/review lifetime is extended.
    """
    from ouroboros.store import STORE_WAIT_SEC
    from ouroboros._usage_wait import ACQUISITION_SLICE_SEC
    from ouroboros.cancel_intents import _validated_single_cancel_target, cancel_pending
    from ouroboros.task_results import load_task_result

    started = time.monotonic()
    needs_rollback = True  # The first error came from the publication writer.
    while True:
        remaining = STORE_WAIT_SEC - (time.monotonic() - started)
        if (error.reason != "contention" or remaining <= 0
                or needs_rollback and isinstance(error, StoreBusy) and not getattr(error, "rollback_confirmed", False)):
            raise error
        time.sleep(min(ACQUISITION_SLICE_SEC, remaining))
        needs_rollback = False  # Lock/authority probes below have not attempted a write.
        try:
            with launch_lock(root_drive, root_task_id, wait=False), acquisition_budget(0):
                current = read_fence(root_drive, root_task_id)
                if current.get("fence_id") != fence_id:
                    if fence_closed(current) and current.get("supersedes_fence_id") == fence_id:
                        return  # The newer Pause keeps the original resumed work held.
                    raise OwnerPauseRefused("owner_pause_fence_changed")
                if task_id != root_task_id and _validated_single_cancel_target(root_drive, root_task_id) != task_id:
                    raise OwnerPauseRefused("root_resume_authority_changed")
                row = load_task_result(root_drive, task_id, strict=True) or {}
                grant = (row.get("budget_pause") or {}).get("grant") or {}
                if (not grant_id or grant.get("grant_id") != grant_id
                        or grant.get("owner_pause_fence_id") != fence_id or grant.get("revoked_at")):
                    raise OwnerPauseRefused("root_resume_authority_changed")
                if (any(cancel_pending(root_drive, member, strict=True) for member in {root_task_id, task_id})
                        or row.get("status") == "cancelled"):
                    raise OwnerPauseRefused("cancelled")
                needs_rollback = True
                release_fence(root_drive, root_task_id, reason="owner_resume_consumed")
                return
        except (StoreBusy, LaunchBusy) as exc:
            error = exc


@acquisition_budget(0)
def select_member_resume(root_drive: Any, root_task_id: str, task_id: str, *,
                         fence_id: str, grant_id: str) -> None:
    """Consume one child Resume under a terminal root; keep its siblings fenced.

    Caller holds ``launch_lock``. The consumed explicit grant, root terminality
    and exact fence are checked again here; neither lineage nor a terminal
    transition grants this exception. A new Pause has no selected members.
    """
    from ouroboros.budget_pause import budget_pause_row, STATE_RESUMED
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, load_task_result
    from ouroboros.utils import utc_now_iso

    root = load_task_result(root_drive, root_task_id, strict=True) or {}
    member = load_task_result(root_drive, task_id, strict=True) or {}
    row = budget_pause_row(pathlib.Path(root_drive), task_id)
    grant = row.get("grant") or {}
    if (task_id == root_task_id or root.get("status") not in _TRULY_TERMINAL_STATUSES
            or str(member.get("root_task_id") or (member.get("metadata") or {}).get("root_task_id") or "") != root_task_id
            or row.get("state") != STATE_RESUMED or not grant_id
            or grant.get("grant_id") != grant_id or not grant.get("consumed_at")
            or grant.get("revoked_at") or grant.get("authority") != "explicit_resume"
            or grant.get("selected_by") != "owner"
            or grant.get("owner_pause_fence_id") != fence_id):
        raise OwnerPauseRefused("member_resume_authority_changed")
    fence = read_fence(root_drive, root_task_id)
    if fence.get("fence_id") != fence_id or not fence_closed(fence):
        raise OwnerPauseRefused("owner_pause_fence_changed")
    selected = dict(fence.get("selected_members") or {})
    selected[task_id] = {"grant_id": grant_id, "selected_at": utc_now_iso()}
    set_fence_state(root_drive, root_task_id, fence_id=fence_id, state=fence["state"],
                    selected_members=selected)


def consumed_member_resume(root_drive: Any, root_task_id: str, task_id: str, *, attempt: int) -> Dict[str, str]:
    """A child's consumed Resume whose worker stopped before ``reopen_for_resume``; ``{}`` otherwise.

    Read only (#1716): the member's own consumed, unrevoked grant naming this
    still-closed fence, from the ``attempt`` whose saved work is resumed. A root
    (whose own Resume door lifts its fence), another attempt, a newer Pause or an
    unreadable row answers ``{}``. Only the owner's held-row selection completes
    it, after every refusal check (``complete_consumed_member_resume``).
    """
    from ouroboros.budget_pause import STATE_RESUMED, budget_pause_row

    try:
        row = budget_pause_row(pathlib.Path(root_drive), task_id)
        grant, fence = row.get("grant") or {}, read_fence(root_drive, root_task_id)
    except Exception:
        return {}
    fence_id = str(grant.get("owner_pause_fence_id") or "")
    if (root_actor(root_drive, root_task_id, task_id) or row.get("state") != STATE_RESUMED
            or not grant.get("consumed_at")
            or grant.get("revoked_at") or int(row.get("task_attempt") or 0) != int(attempt)
            or not fence_id or fence.get("fence_id") != fence_id or not fence_closed(fence)
            or task_id in (fence.get("selected_members") or {})):
        return {}
    return {"fence_id": fence_id, "grant_id": str(grant.get("grant_id") or "")}


def interrupted_member_resume(root_drive: Any, root_task_id: str, task_id: str, *, attempt: int) -> Optional[bool]:
    """Whether the owner's own Resume of this member was consumed, then interrupted (read only).

    Not the owner's deliberate Pause, so Main offers the member's existing Resume
    (#1660): its consumed, unrevoked, owner-selected explicit grant from ``attempt``
    names the still-closed fence of its terminal root, which has not selected it yet
    (``consumed_member_resume``) or selected it on that same grant. A newer Pause
    (another fence), another attempt, a revoked or model grant or a live root answers
    False; ``None`` when a fact it needs cannot be read, which the census keeps as
    unknown. A current root retry is a root actor, never this member exception.
    The Resume itself rechecks all of it.
    """
    from ouroboros.budget_pause import STATE_RESUMED, budget_pause_row
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES

    try:
        if root_actor(root_drive, root_task_id, task_id, strict=True):
            return False
        row = budget_pause_row(pathlib.Path(root_drive), task_id)
        grant = row.get("grant") or {}
        own = bool(row.get("state") == STATE_RESUMED and int(row.get("task_attempt") or 0) == int(attempt)
                   and grant.get("grant_id") and grant.get("consumed_at") and not grant.get("revoked_at")
                   and grant.get("authority") == "explicit_resume" and grant.get("selected_by") == "owner")
    except Exception:
        return None
    if not own:  # the member's own facts already say no: its root is never read
        return False
    try:
        fence = read_fence(root_drive, root_task_id)
        root = _control_projection(root_drive, root_task_id)
    except Exception:
        return None
    selected = (fence.get("selected_members") or {}).get(task_id)
    return bool(root.get("status") in _TRULY_TERMINAL_STATUSES
                and fence_closed(fence) and grant.get("owner_pause_fence_id") == fence.get("fence_id")
                and (selected is None or isinstance(selected, dict) and selected.get("grant_id") == grant["grant_id"]))


def complete_consumed_member_resume(root_drive: Any, root_task_id: str, task_id: str, consumed: Dict[str, str],
                                    publish: Any, withdraw: Any) -> str:
    """Release one owner-selected member on its consumed Resume's fence; ``""`` or why not.

    The caller holds the queue lock, so the launch lock is never awaited: a busy
    tree refuses. Under it the queue selection is published first (``publish``),
    while this member's launch is still fenced, so a failed snapshot leaves the
    fence untouched; then ``select_member_resume`` re-checks the grant and fence and
    releases the member. A failed release withdraws the selection; a process lost
    in between leaves the member fenced, so nothing starts, and restore treats the
    row like any other selected, undispatched one.
    """
    try:
        with launch_lock(root_drive, root_task_id, wait=False):
            if not publish():
                return "snapshot_not_persisted"
            try:
                select_member_resume(root_drive, root_task_id, task_id, fence_id=consumed["fence_id"],
                                     grant_id=consumed["grant_id"])
            except Exception:
                withdraw()
                raise
    except Exception as exc:
        log.warning("Consumed Resume of %s was not completed; its fence stays closed", task_id, exc_info=True)
        return f"{type(exc).__name__}: {str(exc)[:200]}"
    return ""


# --- member side: the gate every launch family shares ----------------------------------

def admit_delegated_start(drive_root: Any, payload: Dict[str, Any]) -> bool:
    """Bind billing and register START_REQUESTED before releasing launch admission.

    This is custody, not an observation of transport bytes. A crash after the
    claim must be reconciled using its existing idempotency key.
    """
    from types import SimpleNamespace
    from ouroboros._usage_wait import retry_presend
    from ouroboros.delegate_custody import emit, START_REQUESTED
    from ouroboros.model_wait import current_model_wait
    from ouroboros.usage_admission import task_billing_fields
    from ouroboros.utils import utc_now_iso

    source = SimpleNamespace(drive_root=drive_root, task_id=str(payload.get("task_id") or ""),
                             root_task_id=str(payload.get("root_task_id") or payload.get("task_id") or ""))
    # START custody is this session's paid admission and its late receipt's bucket: an
    # unestablished authority raises typed here, before any row or POST (owner 2026-10-14).
    # A host operation its caller bound explicitly as non-task (no task lineage) is its own
    # uncapped group on the global axis, exactly as its reservations are; never a prefix guess.
    from ouroboros.usage_accounting import current_usage_scope
    bound = current_usage_scope()
    if not source.root_task_id:  # no task identity at all: an unattributed host start, no group to guess
        binding = {}
    elif bound is not None and bound.non_task_operation is True and str(bound.root_task_id or "") == source.root_task_id:
        binding = {"billing_group_id": source.root_task_id, "billing_group_limit_usd": None}
    else:
        binding = task_billing_fields({"id": source.task_id}, source.root_task_id, None, drive_root)
    payload = {"ts": utc_now_iso(), "event_id": uuid.uuid4().hex, **payload}
    payload["billing_group"] = {k: v for k, v in binding.items() if k.startswith("billing_group_")}
    retrying = False

    def claim_once():
        nonlocal retrying
        owner = current_model_wait()
        if retrying and owner is not None:
            reason = owner.control_reason()
            if reason:
                from ouroboros.llm_attempt import PhysicalDispatchInterrupted
                raise PhysicalDispatchInterrupted(reason)
        retrying = True
        with launch_admission(source, wait=False):
            return emit(drive_root, START_REQUESTED, payload, propagate_contention=True)

    return retry_presend(claim_once, check=False)  # The POST remains outside this unsent claim.


def _member_coordinates(source: Any) -> Tuple[str, str, str]:
    """``(root_drive, root_task_id, task_id)`` of a tool context or usage scope."""
    if getattr(source, "non_task_operation", False) is True:
        return "", "", ""
    meta = getattr(source, "task_metadata", None)
    meta = meta if isinstance(meta, dict) else {}
    task_id = str(getattr(source, "task_id", "") or "")
    root_task_id = str(meta.get("root_task_id") or getattr(source, "root_task_id", "") or task_id)
    root_drive = (meta.get("budget_drive_root") or getattr(source, "budget_drive_root", None)
                  or getattr(source, "drive_root", None) or "")
    return str(root_drive or ""), root_task_id, task_id


def awaits_tree_selection(fence: dict, task_id: str, root_task_id: str) -> bool:
    """An explicitly resumed terminal tree selects each child, including retained warm stacks."""
    grant = fence.get("resume_grant") or {}
    return bool(fence.get("state") == FENCE_RELEASED and grant.get("tree_only")
                and not grant.get("revoked_at") and task_id != root_task_id
                and task_id in (fence.get("warm_members") or ()) and task_id not in (fence.get("selected_members") or {}))


def member_fence(source: Any, *, _observations=None) -> Dict[str, Any]:
    """Read the shared fence strictly: unreadable authority refuses new effects."""
    root_drive, root_task_id, _task_id = _member_coordinates(source)
    if not root_drive or not root_task_id:
        return {}
    marker = _REVIEW_EPISODE.get()
    if (isinstance(marker, dict) and marker.get("root_task_id") == root_task_id
            and marker.get("pause_before_launch")):
        # The author detached before this slot's first handoff. Its preparation
        # may still unwind after Resume, but that does not authorize a late send.
        return {"state": FENCE_REQUESTED, "reason": "owner_pause_before_review_launch"}
    try:
        from ouroboros.model_wait import current_model_wait
        owner = current_model_wait()
        required = getattr(source, "task_lifecycle_bound", False) or (
            owner is not None and owner.task_id == _task_id)
        observations = _observations if _observations is not None else {}
        fence = read_fence(root_drive, root_task_id, _observations=observations)
        if required and not observations[root_task_id].get("present"):
            raise ValueError("owner_pause_authority_missing")
    except StoreBusy:
        raise
    except Exception:
        log.warning("Owner pause authority unreadable for %s", root_task_id, exc_info=True)
        return {"state": "unknown", "reason": "owner_pause_authority_unreadable"}
    if fence_closed(fence) and _task_id in (fence.get("selected_members") or {}):
        return {}  # One consumed owner Resume, bound to this fence generation.
    if fence_closed(fence) and _episode_dispatched_before(fence, root_task_id):
        return {}  # The already-started review episode finishes its own check.
    if awaits_tree_selection(fence, _task_id, root_task_id):
        return {**fence, "state": FENCE_PAUSED, "awaiting_selection": True}
    return fence if fence_closed(fence) else {}


def wait_member_fence(source: Any) -> Dict[str, Any]:
    """Read at the member's safe boundary, retaining its stack through contention.

    Only this pure observation repeats, outside queue, launch and money locks.
    A busy store proves neither an open fence nor an accepted Pause. The usual
    reader still owns unknown authority; supervisor/launch callers do not wait.
    """
    from ouroboros._usage_wait import ACQUISITION_SLICE_SEC
    from ouroboros.budget_pause import _HOLD_POLL_SEC, _hold_control_reason, _publish_hold
    from ouroboros.model_wait import ModelWaitInterrupted

    started, episode_id, entered = time.monotonic(), uuid.uuid4().hex, False

    def publish(phase):
        detail = "Waiting for control storage" if phase == "entered" else "Control storage wait ended"
        _publish_hold(source, {
            "checkpoint_kind": "control_store_wait", "state": "waiting" if phase == "entered" else "ended",
            "hold_reason": "control_store_contention", "phase": phase, "episode_id": episode_id,
            "elapsed_sec": time.monotonic() - started, "detail": detail, "content": detail,
            "text": detail, "role": "system", "system_type": "task_checkpoint",
        })

    try:
        while True:
            try:
                with acquisition_budget(ACQUISITION_SLICE_SEC):
                    return member_fence(source)
            except StoreBusy as exc:
                if exc.reason != "contention" or getattr(exc, "rollback_confirmed", None) is False:
                    raise
                control = _hold_control_reason(source)
                if control:
                    raise ModelWaitInterrupted(control) from exc
                if not entered:
                    entered = True
                    publish("entered")
                time.sleep(_HOLD_POLL_SEC)
    finally:
        if entered:
            publish("ended")


@contextmanager
def review_episode(root_task_id: str, operation: Any = None):
    """Bind ONE physical review attempt's Pause exception to this context (owner 2026-10-08).

    The owner's full variant: a reviewer that already started finishes the
    SAME check while its author is paused — its further model rounds,
    inspections and run starts — but nothing new starts. The marker yielded
    here is armed only by this attempt's first ACTUAL physical handoff: the
    exact model sender handed to its executor (``submit_model``), or an
    operation submitted at its final start point (``operation_start``: the
    delegated run's POST, a process start, a tool executor) — each still under
    the root's launch lock that admitted it with the fence OPEN. Registering
    an operation, a ``START_REQUESTED`` row (the POST still meets the fence), a
    noted dispatch, a preparation, a capacity probe or a ledger transition is
    no authority. A Pause accepted later has
    a newer generation, so the armed attempt alone passes it; an attempt that
    had not launched when the Pause landed is refused at $0 like any other
    launch. The author's own context never carries a marker: its tools, sends,
    commit, apply and publication stay refused.
    """
    marker = {"root_task_id": str(root_task_id or ""), "operation": operation, "armed_generation": None}
    token = _REVIEW_EPISODE.set(marker)
    try:
        yield marker
    finally:
        _REVIEW_EPISODE.reset(token)


def episode_armed(marker: Any) -> bool:
    """Whether this physical review attempt launched before any later Pause."""
    return isinstance(marker, dict) and marker.get("armed_generation") is not None


def _episode_dispatched_before(fence: Dict[str, Any], root_task_id: str) -> bool:
    marker = _REVIEW_EPISODE.get()
    if (not episode_armed(marker) or marker.get("root_task_id") != str(root_task_id or "")
            or getattr(marker.get("operation"), "closed", False)):
        return False
    try:
        return int(marker["armed_generation"]) < int(fence.get("generation") or 0)
    except (TypeError, ValueError):
        return False


def arm_review_episode(source: Any) -> None:
    """Call INSIDE ``launch_admission`` right after the actual handoff succeeded:
    the fence admitted it under the same launch lock, so this attempt's episode
    began before any Pause that lock has not yet let in. Arms once."""
    root_drive, root_task_id, _task_id = _member_coordinates(source)
    marker = _REVIEW_EPISODE.get()
    if not isinstance(marker, dict) or episode_armed(marker) or marker.get("root_task_id") != str(root_task_id or ""):
        return
    admitted = _REVIEW_ADMISSION.get()
    if (not isinstance(admitted, dict) or admitted.get("root_drive") != root_drive
            or admitted.get("root_task_id") != root_task_id):
        return
    # The store was read before submission under the same launch lock. A busy
    # post-send read must neither erase this started episode nor invite a retry.
    marker["armed_generation"] = admitted["generation"]


def scope_fence() -> Dict[str, Any]:
    """The closed fence over the model send bound to this execution context."""
    try:
        from ouroboros.usage_accounting import current_usage_scope

        scope = current_usage_scope()
    except Exception:
        return {}
    if scope is None or not str(getattr(scope, "task_id", "") or ""):
        return {}
    return member_fence(scope)


@contextmanager
def launch_lock(root_drive: Any, root_task_id: str, *, wait: bool = True):
    """Serialize local admission; ``wait=False`` never waits under a queue lock."""
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

    if not root_drive or not root_task_id:
        yield
        return
    path = _lock_path(root_drive, root_task_id)
    outcome = {}
    fd = acquire_exclusive_file_lock(path, owner_aware_stale=True, outcome=outcome,
                                     **({} if wait else {"timeout_sec": 0.0}))
    if fd is None:
        raise LaunchBusy(outcome.get("reason", "unknown"))
    try:
        yield
    finally:
        release_exclusive_file_lock(path, fd)


def _validate_launch(source: Any, root_resume=None):
    root_drive, root_id, task_id = _member_coordinates(source)
    observations = {}
    fence = member_fence(source, _observations=observations)
    def controls(member_id):
        if member_id not in observations:
            observations[member_id] = _control_projection(root_drive, member_id)
        return observations[member_id]
    selected_resume_start = actor = False
    if fence:
        from ouroboros.budget_pause import STATE_RESUME_GRANTED

        allowed = False
        if root_resume and fence_closed(fence):
            row = controls(task_id).get("budget_pause") or {}
            grant = row.get("grant") or {}
            actor = root_allowed = root_actor(root_drive, root_id, task_id)
            if not root_allowed and grant.get("selected_by") == "owner":
                from ouroboros.task_results import _TRULY_TERMINAL_STATUSES
                root_allowed = controls(root_id).get("status") in _TRULY_TERMINAL_STATUSES
            allowed = bool(row.get("state") == STATE_RESUME_GRANTED
                and root_allowed
                and (grant.get("owner_pause_fence_id") or row.get("owner_fence_id")) == fence.get("fence_id")
                and row.get("pause_id") == root_resume.get("pause_id")
                and grant.get("grant_id") == root_resume.get("grant_id")
                and grant.get("authority") == "explicit_resume"
                and not grant.get("revoked_at") and not grant.get("consumed_at"))
        if not allowed:
            raise OwnerPauseRefused(str(fence.get("reason") or "owner_pause"))
        selected_resume_start = not actor
    from ouroboros.budget_pause import STATE_PAUSING, STATE_PAUSED, STATE_RESUME_GRANTED

    if root_drive and task_id:
        for member_id in {root_id, task_id}:
            try:
                row = controls(member_id).get("budget_pause") or {}
            except StoreBusy:
                raise
            except Exception as exc:
                raise OwnerPauseRefused("model_sleep_authority_unreadable") from exc
            if row.get("reason") != "sleep" or row.get("state") not in {
                    STATE_PAUSING, STATE_PAUSED, STATE_RESUME_GRANTED}:
                continue
            if member_id == root_id and task_id != root_id:
                selected = read_fence(root_drive, root_id, _observations=observations).get("selected_members") or {}
                if selected_resume_start or task_id in selected:
                    # Explicit member Resume outlives its terminal root's
                    # saved sleep. It does not release any sibling or root.
                    continue
            grant = row.get("grant") or {}
            resume_start = bool(root_resume and member_id == task_id
                and row.get("state") == STATE_RESUME_GRANTED
                and row.get("pause_id") == root_resume.get("pause_id")
                and grant.get("grant_id") == root_resume.get("grant_id")
                and not grant.get("revoked_at") and not grant.get("consumed_at"))
            if not resume_start:
                raise OwnerPauseRefused("model_sleep")

    return observations.get(root_id, {})


@contextmanager
def launch_admission(source: Any, *, root_resume: Optional[Dict[str, Any]] = None, wait: bool = True):
    """Validate and submit under Pause exclusion; contention waits only BEFORE yielding.

    Money/queue holders pass wait=False and unwind to their existing retry owner.
    The yielded operation is never replayed by this context manager.
    """
    from ouroboros.model_wait import current_model_wait
    from ouroboros._usage_wait import retry_presend

    root_drive, root_id, task_id = _member_coordinates(source)
    def enter():
        owner = current_model_wait()
        if owner is not None and owner.task_id == task_id and owner.closed:
            raise OwnerPauseRefused("operation_already_returned")
        stack = ExitStack()
        review_admission = None
        try:
            stack.enter_context(launch_lock(root_drive, root_id, wait=False))
            with admission_acquisition():
                controls = _validate_launch(source, root_resume)
                marker = _REVIEW_EPISODE.get()
                if (isinstance(marker, dict) and not episode_armed(marker)
                        and marker.get("root_task_id") == root_id):
                    review_admission = {"root_drive": root_drive, "root_task_id": root_id,
                        "generation": int((controls.get("owner_pause") or {}).get("generation") or 0)}
        except BaseException:
            stack.close()
            raise
        return stack, review_admission
    stack, admitted = retry_presend(enter, check=False) if wait else enter()
    with stack, admission_acquisition():
        token = _REVIEW_ADMISSION.set(admitted)
        try:
            yield
        finally:
            _REVIEW_ADMISSION.reset(token)


def tree_member_results(root_drive: Any, root_task_id: str, *, linked: Iterable[str] = ()) -> Any:
    """Select complete retained membership, then strictly read its authority.

    ``task_store.tree_facts`` selects the tree (``linked``: ids the caller's own queue,
    custody or model-request sources tie to it) and names the rows it could not read. A
    member whose strict read refuses or no longer finds its own record joins them as linked:
    it holds this tree while the others still answer. An unavailable store refuses the census.
    Selection never supplies live status or custody. Returns a ``TreeCensus``.
    """
    from ouroboros.store_hot_facts import lineage_root
    from ouroboros.task_results import load_task_result
    from ouroboros.task_store import TreeCensus, tree_facts

    try:
        census = tree_facts(root_drive, root_task_id, linked=linked)
    except StoreBusy:
        raise
    except StoreError as exc:
        raise ValueError("tree_membership_unreadable") from exc
    members, unresolved = {}, dict(census.unresolved)
    for task_id in census.members:
        try:
            row = load_task_result(root_drive, task_id, strict=True)
        except StoreError:
            raise
        except (ValueError, TypeError, AttributeError, KeyError, RecursionError):
            unresolved[task_id] = {"reason": "record_unreadable", "linked": True}
            continue
        if row is None:  # its row without the body that answers for it (or gone since): it alone is unknown
            unresolved[task_id] = {"reason": "body_missing", "linked": True}
            continue
        if lineage_root(row) == root_task_id:
            members[task_id] = row
    return TreeCensus(members, dict(sorted(unresolved.items())))


def tree_model_requests(root_drive: Any, root_task_id: str) -> list:
    """The root's open model attempts (the usage store's open-set index): a census reads them
    first so their task ids tie damaged rows to this tree. Raises when unreadable: unknown, never none."""
    from ouroboros import usage_store

    with usage_store.read(root_drive) as txn:  # captured in one step, decoded after the read closed
        captured = txn.open_attempt_rows(root_task_id=str(root_task_id))
    return usage_store.decode_attempts(captured)


@contextmanager
def tool_handoff(source: Any, name: str):
    """Bind one actual invocation; its existing future owns local completion.

    A directly called registry without that owner keeps the durable fallback.
    Merely entering registry preparation is never an ordinary handler handoff.
    """
    from ouroboros.tool_custody import invocation_outcome, retain_invocation, settle_invocation
    from ouroboros._usage_wait import retry_presend

    inherited = _TOOL_OPERATION.get()
    outcome = (inherited[2] if inherited and inherited[0] is source and inherited[1] == name
               and not inherited[2].get("registry_entered") else invocation_outcome(source, name))
    outcome["registry_entered"] = True
    token = _TOOL_OPERATION.set((source, name, outcome))
    root_drive, root_id, task_id = _member_coordinates(source)
    refusal = None
    try:
        handed = _HANDED_TOOL.get()
        with _CACHE_LOCK:
            outcome["handed"] = bool(handed and handed["invocation"] == (id(source), name) and not handed["claimed"])
            if outcome["handed"]:
                handed["claimed"] = True
        if not outcome.get("memory_owned") and root_drive and task_id:
            def claim_once():
                from ouroboros.model_wait import current_model_wait
                owner = current_model_wait()
                if owner is not None and owner.task_id == task_id and owner.closed:
                    raise OwnerPauseRefused("operation_already_returned")
                with (launch_lock(root_drive, root_id, wait=False) if outcome["handed"]
                      else launch_admission(source, wait=False)):
                    retain_invocation(outcome)
            retry_presend(claim_once, check=False)
        yield outcome
    except (OwnerPauseRefused, StoreError) as exc:
        refusal = exc
        raise
    finally:
        with outcome["phase_lock"]:
            outcome["closed"] = True
            if refusal is not None:
                # Snapshot with closure, not before a competing admitted start.
                refusal.invocation_not_started = outcome.get("not_started") is True
        _TOOL_OPERATION.reset(token)
        if outcome.get("future_row") is None:
            settle_invocation(outcome)


def current_tool_operation(source: Any, name: str) -> str:
    """Exact synchronous invocation identity, never an exemption by tool name."""
    active = _TOOL_OPERATION.get()
    return str(active[2].get("operation_id") or "") if active and active[0] is source and active[1] == name else ""


def record_mcp_call_returned(name: str) -> None:
    """This invocation's MCP runner ended: its ``asyncio.run`` returned or raised after loop
    shutdown. Only that runner records it, also after a timeout closed the call (a nested
    runner holds the invocation until then), never result text or provider meta. It settles
    this local claim alone, not its remote duty, other processes, runs or money."""
    active = _TOOL_OPERATION.get()
    if active and active[1] == name:
        active[2]["mcp_call_returned"] = True


@contextmanager
def operation_start(source: Any = None):
    """Final local operation-start point after preparation, serialized with Pause.

    Hold only across the local submission (for example Popen), never its wait.
    Preparation must finish first. A nested start refused here proves no effect
    for that submission, not for an earlier start in the same invocation.
    """
    active = _TOOL_OPERATION.get()
    source = source if source is not None else (active[0] if active else None)
    with launch_admission(source):
        if active and active[0] is source:
            _mark_tool_start(active[2])
        yield
        arm_review_episode(source)  # the submission inside succeeded under this same lock


def _mark_tool_start(outcome: dict) -> None:
    """Order admission against closure; never hold this lock across I/O.

    Called after root admission. The durable RLock remains independent: taking
    it here could make Pause wait for another thread's custody write.
    """
    with outcome["phase_lock"]:
        if outcome.get("closed"):
            raise OwnerPauseRefused("operation_already_returned")
        outcome["not_started"] = False


def submit_tool(source: Any, name: str, submit: Any, function: Any, *args: Any):
    """Hand one invocation to its existing sticky executor under Pause exclusion."""
    with launch_admission(source):
        with physical_acquisition():
            context = copy_context()
            context.run(_HANDED_TOOL.set, {"invocation": (id(source), name), "claimed": False})
            handed = submit(context.run, function, *args)
        arm_review_episode(source)
        return handed


def run_operation(source: Any, function: Any, /, *args: Any, **kwargs: Any):
    """Submit an opaque synchronous call; join outside the short launch lock.

    The forwarders' own parameters are positional-only: every keyword, such as
    a tool argument named ``source`` or ``function``, belongs to the callee.

    The worker owns only this call. Its copied context does not authorize any
    nested process, model or delegated submission after a subsequent Pause.
    """
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="operation") as executor:
        with operation_start(source):
            with physical_acquisition():
                context = copy_context()
                future = executor.submit(context.run, function, *args, **kwargs)
        return future.result()


def run_tool_handler(source: Any, function: Any, /, *args: Any, **kwargs: Any):
    """The prepared callable passes directly to its already-running worker.

    Taking this exact invocation under the launch lock is the handoff, not a
    queued preparation or a row write. The next action outside is the body;
    every nested operation still has its own fresh operation_start gate.
    """
    from ouroboros.model_wait import current_model_wait, dispatch_deadline_remaining_sec
    from ouroboros.llm_attempt import PhysicalDispatchInterrupted
    from ouroboros.tools.process_facts import process_facts_handoff
    from ouroboros._usage_wait import retry_presend

    active = _TOOL_OPERATION.get()
    outcome = active[2] if active and active[0] is source else None
    try:
        with process_facts_handoff(function) as invoke:
            def take():
                root, tree, _task = _member_coordinates(source)
                with (launch_lock(root, tree, wait=False) if outcome and outcome.get("handed")
                      else launch_admission(source, wait=False)):
                    owner = current_model_wait()
                    reason = owner.control_reason(strict_current=True) if owner is not None else None
                    remaining = dispatch_deadline_remaining_sec()
                    if reason or remaining is not None and remaining <= 0:
                        raise PhysicalDispatchInterrupted(reason or "deadline")
                    if outcome is not None:
                        _mark_tool_start(outcome)
                    arm_review_episode(source)
            # Validate the inherited caller controls, not this tool's soft wait
            # envelope. Its retry owner must use that same context. The body
            # keeps the worker envelope for inner waits after admission returns.
            admission = outcome.get("admission_context") if outcome is not None else None
            if admission is None:
                retry_presend(take, check=False)
            else:
                admission.run(retry_presend, take, check=False)
            return invoke(*args, **kwargs)
    finally:
        if outcome is not None:
            outcome["settled"] = True


def submit_async_operation(source: Any, function: Any, /, *args: Any, **kwargs: Any):
    """Submit to the current event loop without executing the body under a lock."""
    import asyncio

    async def invoke():
        return await function(*args, **kwargs)

    with operation_start(source):
        with physical_acquisition():
            return asyncio.create_task(invoke())


def submit_async_preparation(function: Any, *args: Any, source: Any = None, **kwargs: Any):
    """Admit a nested transport's local preparation before it opens anything.

    Entering an MCP stdio transport starts the server process: that handoff
    serializes with Pause here, so an accepted Pause refuses it. It is not the
    operation start: ``submit_async_operation`` stays the final gate for the
    call itself (and for a returned invocation). The body runs after the short
    lock, never under it. ``source`` names the task outside a tool invocation
    (a task's MCP discovery); otherwise the active invocation's task decides.
    """
    import asyncio

    active = _TOOL_OPERATION.get()

    async def invoke():
        return await function(*args, **kwargs)

    source = source if source is not None else active[0] if active else None
    with launch_admission(source):
        if active and active[0] is source:
            _mark_tool_start(active[2])  # preparation can open a process/socket
        with physical_acquisition():
            task = asyncio.create_task(invoke())
        return task


def model_handed_off(attempt_id: str = "") -> bool:
    """Only this physical attempt may finish despite a later owner Pause."""
    if not attempt_id:
        from ouroboros.usage_accounting import last_physical_attempt_capture
        attempt_id = getattr(last_physical_attempt_capture(), "attempt_id", "")
    return bool(attempt_id and _HANDED_MODEL.get() == attempt_id)


def submit_model(reservation: Any, submit: Any, function: Any):
    """Transfer the exact sender to an executor, mutually exclusive with Pause."""
    from ouroboros.llm_attempt import _PhysicalSendNotStarted, require_physical_dispatch_window

    try:
        with launch_admission(reservation.scope, wait=False):
            require_physical_dispatch_window()
            with physical_acquisition():
                context = copy_context()
                context.run(_HANDED_MODEL.set, reservation.attempt_id)
                handed = submit(context.run, function)
            arm_review_episode(reservation.scope)  # the exact sender now owns these bytes
            return handed
    except OwnerPauseRefused as exc:
        if isinstance(exc, LaunchBusy) and exc.reason == "contention":
            raise
        raise _PhysicalSendNotStarted(str(exc)) from exc
