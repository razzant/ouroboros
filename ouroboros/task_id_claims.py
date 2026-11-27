"""A new task id is claimed in the canonical store before anything is staged under it (#4317).

Four producers mint ids: a helper (``schedule_subagent``, with its depth refusal), ``start_task`` and the task
API. Each claims its id in ONE store transaction that finds no record under it, nor under a name the file system
resolves to it (``taken_in``: a row, a body the reader cannot bind, an identity the import left occupied, another
request's provisional claim), and stages its first file only afterwards, into folders it makes by exclusive
creation (``make_own_folders``): its rollback removes only those, never a folder its id merely names, and of them
only what holds nothing (``remove_own_paths``); one that holds anything stays with all of it and keeps the id. That
transaction commits under the lock every Stop records its intent under, after a strict read finds no Stop of the
id pending (``stop_fenced``): a Stop lands before the claim, and takes the id, or after it, on the claimed record.
Only a proven occupancy (``TaskIdTaken``) lets a generated id be minted again. A name that cannot be inspected, or
a store that was busy before anything was written, refuses the claim unheld and is never read as free
(``TaskIdUninspectable``). A claim or a release of unknown outcome (``TaskIdUnconfirmed``) names the id and never
moves the request to another one: whether the id is held stays unknown, a record under it is not proof it is this
request's, and nothing is deleted on it.

A claim is a row where the producer's first record is one (a helper's REQUESTED row, ``start_task``'s intent,
a depth refusal), or a provisional record of the ``ID_CLAIMS`` set where the producer's first row comes after
its staging (the task API's receipt): that one is no task row, so no reader of tasks sees it. A Stop that finds
no row meets it there (``stop_provisional_claim``): before the hand-off it fences the claim (phase ``stopped``),
so the admission queues nothing and rolls back what it staged; a handed-off one may be queued and the Stop stays
pending for the queue's custody. The claim goes only on proof: a rollback that removed what it staged, a task row
holding the id (its own refusal row records what it staged), or, after the queue is recovered, a producer that
ended before its hand-off and left nothing staged, or a hand-off the ended producer left that the queue, read whole,
does not hold. Any other later phase without a row may be queued: it stays held. A helper row not yet completed
into a request is internal state that public task reads leave out (``provisional_request``).
"""

from __future__ import annotations

import contextlib
import logging
import os
import pathlib
import stat
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from ouroboros.platform_layer import IS_WINDOWS
from ouroboros.task_results import STATUS_REQUESTED, validate_task_id
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)

REQUEST_CLAIM = "request_claim"  # a helper row's {"token", "phase"}: the one request it belongs to (``same_request``)
ID_CLAIMS = "task_id_claims"  # store set: the provisional claims of ids that have no row yet (the task API's)
ID_CLAIM_ATTEMPTS = 8  # fresh ids tried before a producer gives up: each retry needs a proven occupancy
_AFTER = "\U0010ffff"  # sorts after every character an id holds: [name, name + _AFTER) is every id it starts


class TaskIdTaken(ValueError):
    """The id is held by another record (``reason``) and nothing was written under it: a row the claim does not
    recognize as its own, a body its reader cannot bind, an identity the import left occupied, another request's
    claim, a name the file system resolves to one of these, or folders an earlier owner left."""

    def __init__(self, task_id: str, reason: str) -> None:
        super().__init__(f"task id is taken by another record: task:{task_id} ({reason})")
        self.task_id, self.reason = str(task_id), reason


class ClaimRefused(Exception):
    """Refused before anything was held or written (a reservation another owner holds, a worker pool that is
    down, a name that cannot be checked or held): ``claim_id`` passes it on, never reading it as unknown."""


class TaskIdUninspectable(ClaimRefused):
    """What the id keys could not be inspected before any claim: nothing was held or written, and the id is neither
    read as free nor replaced by another."""

    def __init__(self, task_id: str, detail: str) -> None:
        super().__init__(f"task:{task_id} could not be checked: {detail}")
        self.task_id, self.detail = str(task_id), detail


class TaskIdUnrepresentable(ClaimRefused):
    """A new id this file system cannot hold as written (Windows drops a name's trailing dots)."""

    def __init__(self, task_id: str) -> None:
        super().__init__(f"task id cannot name files on this system (Windows drops trailing dots): task:{task_id}")
        self.task_id = str(task_id)


class TaskIdUnconfirmed(RuntimeError):
    """The claim of ``task_id`` (or its release) has an unknown outcome: whether this request holds the id is
    unknown, so it is reported, never replaced by another, and grants nothing to delete. A record read under it is
    not proof that it is this request's."""

    def __init__(self, task_id: str, detail: str) -> None:
        super().__init__(f"the claim of task:{task_id} is unconfirmed: {detail}")
        self.task_id, self.detail = str(task_id), detail


def claim_failure_words(exc: BaseException) -> str:
    """Why no generated id was claimed, naming none: an id found held is another record's, and one that could not
    be checked may be."""
    if isinstance(exc, TaskIdTaken):
        return ("a Stop of its task id is pending" if exc.reason == "stop_pending"
                else "every generated task id is held by another record")
    if isinstance(exc, TaskIdUninspectable):
        return f"the generated task id could not be checked ({exc.detail.split(':', 1)[0]})"
    return f"{type(exc).__name__}: {exc}"


def request_token_of(row: Any) -> str:
    """The helper request token ROW records (``REQUEST_CLAIM``), "" for a row written before tokens."""
    claim = row.get(REQUEST_CLAIM) if isinstance(row, dict) else None
    return str(claim.get("token") or "") if isinstance(claim, dict) else ""


def same_request(row: Any, token: Any) -> bool:
    """Whether ROW, a stored result, is the record of the helper request TOKEN names: equal tokens. A request and
    a row written before tokens (none on either side) match only as a helper's row (``delegation_role``
    ``subagent``), never another protocol's record. An empty ROW is no request's record."""
    if not isinstance(row, dict) or not row:
        return False
    held, token = request_token_of(row), str(token or "")
    return held == token if held or token else row.get("delegation_role") == "subagent"


def unsent_request(row: Any, token: str) -> bool:
    """Whether ROW is still TOKEN's own REQUESTED record, which its producer may complete or release before sending
    it. An absent row, one that ended (a Stop, recovery's refusal) or another request's is none."""
    return same_request(row, token) and row.get("status") == STATUS_REQUESTED


def provisional_request(row: Any) -> bool:
    """Whether ROW is a helper's claim its producer has neither completed into a request nor closed: internal
    state, no task for a public reader. ``PROVISIONAL_ROWS`` selects the same rows in SQL."""
    claim = row.get(REQUEST_CLAIM) if isinstance(row, dict) else None
    return isinstance(claim, dict) and claim.get("phase") == "claimed" and row.get("status") == STATUS_REQUESTED


def public_row(row: Any) -> Any:
    """ROW as a public reader (get, peek, wait, forward) sees it: ``{}``, no task, for a helper's claim still being
    prepared (``provisional_request``); any other ROW unchanged. Custody and recovery read the stored row itself."""
    return {} if provisional_request(row) else row


# ``provisional_request`` over the ``tasks`` table: only a requested row's body is read. ``PUBLIC_TASKS`` is its
# complement for a reader whose statement names the table under another alias (a lineage, an inventory).
PROVISIONAL_ROWS = (
    f"tasks.status IS '{STATUS_REQUESTED}' AND EXISTS (SELECT 1 FROM task_bodies AS claimed WHERE "
    "claimed.task_id=tasks.task_id AND CASE WHEN json_valid(claimed.body) THEN "
    "json_extract(claimed.body, '$.request_claim.phase') END='claimed')")
PUBLIC_TASKS = f"NOT IN (SELECT tasks.task_id FROM tasks WHERE {PROVISIONAL_ROWS})"


def finish_request(token: str) -> Callable[[Dict[str, Any], Dict[str, Any]], Optional[Dict[str, Any]]]:
    """The write projector of a producer's final update: it lands only on TOKEN's own claimed, unsent record."""
    def project(current: Dict[str, Any], incoming: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return incoming if unsent_request(current, token) and provisional_request(current) else None

    return project


def mint_task_id(width: int) -> str:
    """A random task id of WIDTH hex digits (8 for helpers, 16 for new roots: the widths ids always had)."""
    return uuid.uuid4().hex[:width]


# ---- what the id keys outside the store ------------------------------------------------------------------------

def retained_task_paths(root: Any, task_id: str, *extra: pathlib.Path) -> Tuple[pathlib.Path, ...]:
    """Every folder an id keys before it has a record, one list for all four producers: its task state (drive,
    inputs), its artifacts and a direct task drive, with the ``extra`` ones a producer stages (a helper's
    retained readonly inputs)."""
    from ouroboros.headless import ARTIFACTS_DIR, HEADLESS_TASKS_DIR, TASK_DRIVES_DIR

    tid, base = validate_task_id(task_id), pathlib.Path(root)
    return (base / HEADLESS_TASKS_DIR / tid, base / ARTIFACTS_DIR / tid, base / TASK_DRIVES_DIR / tid, *extra)


def present(path: pathlib.Path) -> bool:
    """Whether anything, a dangling link included, stands at PATH. Only the system's "no such file" is absence: any
    other inspection error raises, never read as missing."""
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    return True


def occupied(paths: Iterable[pathlib.Path]) -> bool:
    """Whether any of PATHS exists (``present``). A volume that resolves names without regard to case, or without
    trailing dots, answers for an alias too: another owner's folder spelled otherwise is this one."""
    return any(present(path) for path in paths)


def _spelled(path: pathlib.Path) -> bool:
    """Whether PATH's folder stands on disk under exactly that name: one traversal of its parent, stopped at the
    match."""
    with os.scandir(path.parent) as entries:
        return any(entry.name == path.name for entry in entries)


def make_own_folders(paths: Iterable[pathlib.Path]) -> Tuple[pathlib.Path, ...]:
    """Make each of PATHS by exclusive creation (``os.mkdir``; a missing parent is shared and made as needed)
    before anything is staged into it, and return them: a request's own folders under its claimed id are the
    ones this call made, never inferred from the id, and its rollback removes only those. A folder that already
    stands (``FileExistsError``) or cannot be made is not this request's: what this call made is taken back
    while still empty, so nothing another writer put there is touched, and the error is raised."""
    made: List[pathlib.Path] = []
    try:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            os.mkdir(path)
            made.append(path)
    except OSError:
        for path in reversed(made):
            with contextlib.suppress(OSError):
                os.rmdir(path)  # empty only: a folder another writer filled meanwhile stays, as theirs
        raise
    return tuple(made)


def remove_own_paths(paths: Iterable[pathlib.Path]) -> bool:
    """Take back the folders a request made under its claimed id (``make_own_folders``): True once none of them
    stands. Only a folder that holds nothing but empty folders goes, deepest first, by the kernel's empty-folder
    removal (``os.rmdir``): making a folder proves who made it, never whose its files are when it is removed (a
    writer that draws the same id without a claim may have written there since), so one that holds any file or
    link stays exactly as it is, and its request keeps the id beside it (a typed hold; a record of it leaves the
    files to that record's retention). A folder not provably this request's (a link, a name whose exact spelling on
    disk is not the claimed one) stays too, as does one that cannot be inspected."""
    gone = True
    for path in paths:
        try:
            if not present(path):
                continue
            if not stat.S_ISDIR(os.lstat(path).st_mode) or not _spelled(path) or not _holds_nothing(path):
                gone = False
                continue
            for folder, _folders, _files in os.walk(path, topdown=False):
                with contextlib.suppress(OSError):
                    os.rmdir(folder)  # empty only: whatever landed meanwhile keeps its folder
            gone = gone and not present(path)
        except OSError:
            log.warning("A folder under a claimed task id could not be inspected; it stays", exc_info=True)
            gone = False
    return gone


def _holds_nothing(path: pathlib.Path) -> bool:
    """Whether PATH's tree is empty folders alone: no file and no link (a linked folder is listed, never walked).
    A folder that cannot be listed raises: unknown is never read as empty."""
    def unreadable(exc: OSError) -> None:
        raise exc

    for folder, folders, files in os.walk(path, onerror=unreadable):
        if files or any(os.path.islink(os.path.join(folder, name)) for name in folders):
            return False
    return True


# ---- the claim itself ------------------------------------------------------------------------------------------

def claim_id(task_id: str, write: Callable[[str], Any], release: Callable[[str], bool], *, width: int,
             folders: Callable[[str], Iterable[pathlib.Path]] = lambda _tid: ()) -> Tuple[str, Any]:
    """``(id, WRITE's result)``: one request's claim of TASK_ID, or, when it is empty, of a fresh id of WIDTH hex
    digits. ``write(id)`` makes the strict claim in the store under the Stop interlock (``stop_fenced``;
    ``TaskIdTaken`` when another record or a pending Stop holds the id) and ``release(id)`` undoes it (True once
    none of it remains). Folders of ``folders(id)`` that already exist are an earlier owner's and take the id as a
    row does, before the claim and after it (a writer that claims nothing may have made one meanwhile; the claim is
    released first). A fixed id is refused with ``TaskIdTaken``; a generated one is minted again on that proof
    only. Folders that cannot be inspected and a store busy before anything was written refuse the claim unheld
    (``TaskIdUninspectable``), and a new id Windows cannot hold refuses too (``TaskIdUnrepresentable``). Any other
    failure of the write, the check after it or the release is ``TaskIdUnconfirmed``."""
    def take(tid: str) -> Tuple[str, Any]:
        if IS_WINDOWS and tid.endswith("."):
            raise TaskIdUnrepresentable(tid)
        try:
            paths = tuple(folders(tid))
            held = occupied(paths)
        except OSError as exc:
            raise TaskIdUninspectable(tid, f"{type(exc).__name__}: {exc}") from exc
        if held:
            raise TaskIdTaken(tid, "existing_files")
        try:
            stored = write(tid)
        except (TaskIdTaken, ClaimRefused):
            raise
        except Exception as exc:
            if _unwritten(exc):
                raise TaskIdUninspectable(tid, f"the store was busy before the claim: {exc}") from exc
            raise TaskIdUnconfirmed(tid, f"{type(exc).__name__}: {exc}") from exc
        try:
            held = occupied(paths)
        except OSError as exc:
            raise TaskIdUnconfirmed(tid, f"its folders could not be checked after the claim: {exc}") from exc
        if held:
            if release(tid):
                raise TaskIdTaken(tid, "existing_files")
            raise TaskIdUnconfirmed(tid, "its claim could not be released beside folders another writer made")
        return tid, stored

    if task_id:
        return take(validate_task_id(task_id))
    for attempt in range(ID_CLAIM_ATTEMPTS):
        tid = mint_task_id(width)
        try:
            return take(tid)
        except TaskIdTaken:
            if attempt + 1 >= ID_CLAIM_ATTEMPTS:
                raise
            log.warning("Generated task id %s is taken by another record; minting another", tid)
    raise AssertionError("unreachable")


def _unwritten(exc: BaseException) -> bool:
    """A store acquisition gap that rolled nothing in (``StoreBusy`` contention): the claim was never written."""
    from supervisor.workspace_preparation import store_contention

    return store_contention(exc)


def while_contended(step: Callable[[], Any], *, stop: Optional[Callable[[], bool]] = None) -> Any:
    """STEP's result, STEP run again while only contention refused it (``_unwritten``: nothing landed), one short
    slice apart (``_usage_wait.ACQUISITION_SLICE_SEC``) within the store's usual wait (``store.STORE_WAIT_SEC``),
    nothing held between tries; any other error, an active ``stop()``, or contention past that wait, raises."""
    import time

    from ouroboros._usage_wait import ACQUISITION_SLICE_SEC
    from ouroboros.store import STORE_WAIT_SEC

    deadline = time.monotonic() + STORE_WAIT_SEC
    while True:
        try:
            return step()
        except Exception as exc:
            if not _unwritten(exc) or (stop and stop()) or time.monotonic() + ACQUISITION_SLICE_SEC >= deadline:
                raise
        time.sleep(ACQUISITION_SLICE_SEC)


def stop_interlocked(root: Any, step: Callable[[], Any], *, stop: Optional[Callable[[], bool]] = None) -> Any:
    """STEP's result, run under the lock every Stop records its intent under
    (``cancel_intents.cancellation_projection_lock``) with each store acquisition in it one short slice: a Stop,
    whose own acquisition of that lock gives up after 4 s, never waits behind STEP for the store's usual 10 s. A busy
    store lets the lock go and STEP runs again, reading its state anew (``while_contended``)."""
    from ouroboros._usage_wait import ACQUISITION_SLICE_SEC
    from ouroboros.cancel_intents import cancellation_projection_lock
    from ouroboros.store import acquisition_budget

    def locked() -> Any:
        with cancellation_projection_lock(root), acquisition_budget(ACQUISITION_SLICE_SEC):
            return step()

    return while_contended(locked, stop=stop)


def stop_fenced(root: Any, task_id: str, write: Callable[[], Any]) -> Any:
    """WRITE (a claim's one store publication) under the lock every Stop records its intent under
    (``cancel_intents.cancellation_projection_lock``), after a strict read there finds no Stop of TASK_ID pending:
    a Stop lands either before the claim, which then takes nothing (``TaskIdTaken`` ``stop_pending``), or after it,
    on the record its custody settles. A lock or a Stop projection that cannot be read refuses before anything is
    written (``TaskIdUninspectable``); WRITE's own errors pass unchanged. Nothing else runs inside: no Q (whose
    holders may wait for this lock), no queue reservation, no filesystem staging, and no long store wait: WRITE's
    acquisitions take one short slice (``_usage_wait.ACQUISITION_SLICE_SEC``), so a busy store refuses the claim
    unwritten (``StoreBusy``, which ``claim_id`` reads as ``TaskIdUninspectable``) while every Stop waits behind
    this lock no longer than that."""
    from ouroboros._usage_wait import ACQUISITION_SLICE_SEC
    from ouroboros.cancel_intents import cancellation_projection_lock, has_active_intent
    from ouroboros.store import acquisition_budget

    tid = validate_task_id(task_id)
    with contextlib.ExitStack() as stack:
        try:
            stack.enter_context(cancellation_projection_lock(root))
            pending = has_active_intent(root, tid, strict=True)
        except Exception as exc:
            raise TaskIdUninspectable(tid, f"its Stops could not be read: {type(exc).__name__}: {exc}") from exc
        if pending:
            raise TaskIdTaken(tid, "stop_pending")
        with acquisition_budget(ACQUISITION_SLICE_SEC):
            return write()


def case_insensitive(directory: Any) -> bool:
    """Whether DIRECTORY's volume resolves a name without regard to case, probed on one real entry of it (the same
    object under its other spelling) and kept per volume. True, and not kept, when no entry can be probed: unknown
    is never read as distinct names, so an id spelled otherwise is refused."""
    try:
        device = os.stat(directory).st_dev
        if device in _FOLDING:
            return _FOLDING[device]
        with os.scandir(directory) as entries:
            for entry in entries:
                swapped = entry.name.swapcase()
                if swapped == entry.name:
                    continue
                try:
                    held = os.lstat(entry.path)
                except FileNotFoundError:
                    continue
                try:
                    same = os.path.samestat(held, os.lstat(os.path.join(directory, swapped)))
                except FileNotFoundError:
                    same = False
                _FOLDING[device] = same
                return same
    except OSError:
        log.debug("Case folding of %s could not be probed", directory, exc_info=True)
    return True


_FOLDING: Dict[int, bool] = {}


def _name_key(name: str, folds: bool) -> str:
    """What the file system resolves NAME to, as far as ids go: Windows drops its trailing dots, and a volume
    that ``folds`` ignores its case."""
    name = name.rstrip(".") if IS_WINDOWS else name
    return name.lower() if folds else name


def _alias_in(first: Callable[[str, str], Optional[str]], task_id: str, folds: bool) -> bool:
    """Whether an ordered id index holds another name the file system resolves to TASK_ID's; ``first(low, high)``
    is its first id in [low, high). The walk goes one character at a time through the spellings that exist, so its
    seeks follow the id's length and those spellings, never the size of the history."""
    spellings = [""]
    for char in (task_id.rstrip(".") if IS_WINDOWS else task_id):
        forms = sorted({char.lower(), char.upper()} if folds else {char})
        spellings = [name for name in (prefix + form for prefix in spellings for form in forms)
                     if first(name, name + _AFTER) is not None]
    for name in spellings:
        while True:
            if name != task_id and first(name, name + "\0") == name:
                return True
            name += "."  # Windows: the same name with any number of trailing dots
            if not IS_WINDOWS or first(name, name + _AFTER) is None:
                break
    return False


def abandoned(claim: Dict[str, Any]) -> bool:
    """A provisional claim whose producer provably ended before the hand-off (phase ``claimed``, or ``stopped`` by
    a Stop's fence): nothing was queued under it. A handed-off claim, or one of a producer that may live, is an
    admission that may be in flight: it stays."""
    from ouroboros.local_custody_repair import owner_ended

    producer = claim.get("producer") if isinstance(claim.get("producer"), dict) else {}
    return claim.get("phase") in _UNSENT_PHASES and owner_ended(producer.get("pid"), producer.get("birth"))


_UNSENT_PHASES = frozenset({"claimed", "stopped"})  # a provisional claim's phases before any hand-off


def taken_in(txn: Any, task_id: str, *, token: str = "", row_reason: str = "row_without_readable_body",
             folds: bool = False, stale: Optional[Dict[str, Any]] = None) -> None:
    """Raise ``TaskIdTaken`` unless this transaction finds TASK_ID free for TOKEN: no task row or body (a row its
    reader could not bind included), no identity the import refused or quarantined (``store_import.OCCUPIED``), no
    other request's provisional claim (TOKEN's own stands; only ``stale``, the exact record judged ``abandoned``
    before this transaction, is taken over), and none of these under another name the file system resolves to it
    (``folds`` case, Windows trailing dots; their file-keyed state would alias). Rows and claims are found by index
    seeks (``_alias_in``), the import's bounded list by name. Reads only: every probe of a process or a volume
    happens before the transaction (``case_insensitive``, ``abandoned``)."""
    from ouroboros.store_import import OCCUPIED

    if txn.row(task_id, "task_id") is not None or txn.body_text(task_id) is not None:
        raise TaskIdTaken(task_id, row_reason)
    key = _name_key(task_id, folds)
    if any(reason and _name_key(str(name), folds) == key for name, reason in (txn.meta(OCCUPIED) or {}).items()):
        raise TaskIdTaken(task_id, "occupied_by_import")
    held = txn.get_row(ID_CLAIMS, task_id)
    if held is not None and held.get("token") != token and (stale is None or held != stale):
        raise TaskIdTaken(task_id, "claimed")
    if not (folds or IS_WINDOWS):
        return

    def first(sql: str, *params: str) -> Callable[[str, str], Optional[str]]:
        def seek(low: str, high: str) -> Optional[str]:
            found = txn.conn.execute(sql, (*params, low, high)).fetchone()
            return None if found is None else found[0]

        return seek

    rows = first("SELECT task_id FROM tasks WHERE task_id>=? AND task_id<? ORDER BY task_id LIMIT 1")
    claims = first("SELECT id FROM obligations WHERE set_name=? AND id>=? AND id<? ORDER BY id LIMIT 1", ID_CLAIMS)
    if _alias_in(rows, task_id, folds) or _alias_in(claims, task_id, folds):
        raise TaskIdTaken(task_id, "name_alias")


def release_task_claim(drive_root: Any, task_id: str, claim: Callable[[Dict[str, Any]], bool]) -> bool:
    """Undo a row claim before anything else used its id: delete TASK_ID's row in one write transaction only
    while ``claim`` still recognizes it, no Stop of it is pending and the owner kept nothing of it. The Stop is read
    there under the lock every Stop records its intent under (``stop_interlocked``), so a Stop either lands first and
    keeps the row its custody settles, or comes after and finds no record. True once no row of this claim remains
    (deleted now, or absent); False when another record stands there (left as it is), a Stop or the owner's Keep
    holds it, or the store did not answer: an unknown outcome keeps the id occupied, so no later request takes it
    silently."""
    from ouroboros import store
    from ouroboros.cancel_intents import has_active_intent

    tid = validate_task_id(task_id)

    def release() -> bool:
        with store.open_store(drive_root, write=True) as txn:
            current = txn.load(tid)
            if current is None:
                return txn.row(tid, "task_id") is None
            if (not claim(current) or isinstance(current.get("owner_keep"), dict)
                    or has_active_intent(drive_root, tid, strict=True)):
                return False
            txn.delete_task(tid)
            return True

    try:
        return stop_interlocked(drive_root, release)
    except Exception:
        log.warning("The claim of task:%s was not released; its row keeps the id", tid, exc_info=True)
        return False


def close_unsent_claim(drive_root: Any, task_id: str, token: str, **facts: Any) -> bool:
    """The ending of a helper claim whose record must keep its id (its retained selection is keyed by it): TOKEN's
    claimed row closes as never admitted instead of being released, the ending recovery would give it. Never over
    a pending Stop, under the lock every Stop records its intent under (``stop_interlocked``). True once closed."""
    from ouroboros.cancel_intents import has_active_intent
    from ouroboros.task_results import STATUS_FAILED, write_task_result

    tid = validate_task_id(task_id)
    try:
        stored = stop_interlocked(drive_root, lambda: None if has_active_intent(drive_root, tid, strict=True) else (
            write_task_result(drive_root, tid, STATUS_FAILED, _field_projector=finish_request(token),
                              reason_code="request_unsent", admission_outcome="never_admitted",
                              accounted_upper_bound_usd=0.0, **facts)))
        return stored is not None and stored.get("status") == STATUS_FAILED and request_token_of(stored) == token
    except Exception:
        log.warning("The unsent claim of task:%s was not closed; its row keeps the id", tid, exc_info=True)
        return False


# ---- the provisional claim of the task API ----------------------------------------------------------------------

def claim_provisional_id(drive_root: Any, task_id: str, token: str) -> None:
    """Hold TASK_ID for TOKEN, stamped with this process as its producer, in one store transaction that finds it
    free (``taken_in``) and commits under the Stop interlock (``stop_fenced``). The record is no task row:
    ``load_task_result``, the task API's reads and every reader of rows see nothing until the producer's own first
    row exists. A replay of TOKEN is the same claim. A first read that fails wrote nothing
    (``TaskIdUninspectable``)."""
    from ouroboros import store
    from ouroboros.platform_layer import process_start_time

    tid, me = validate_task_id(task_id), os.getpid()
    stamp = {"token": token, "phase": "claimed", "at": utc_now_iso(),
             "producer": {"pid": me, "birth": process_start_time(me)}}
    folds = case_insensitive(drive_root)
    try:
        with store.open_store(drive_root, write=False) as txn:  # a holder's liveness is judged outside any write
            held = txn.get_row(ID_CLAIMS, tid)
    except Exception as exc:
        raise TaskIdUninspectable(tid, f"its claims could not be read: {type(exc).__name__}: {exc}") from exc
    stale = held if held is not None and held.get("token") != token and abandoned(held) else None

    def write() -> None:
        with store.open_store(drive_root, write=True) as txn:
            taken_in(txn, tid, token=token, row_reason="another_request", folds=folds, stale=stale)
            if (txn.get_row(ID_CLAIMS, tid) or {}).get("token") != token:
                txn.put_row(ID_CLAIMS, tid, stamp)

    stop_fenced(drive_root, tid, write)


def advance_provisional_claim(drive_root: Any, task_id: str, token: str, phase: str) -> str:
    """Record that TOKEN's admission passes into PHASE (``handoff``: from here on it may be queued, so the claim
    is no longer abandoned with its producer): "" once it has. Otherwise why it has not, and nothing is queued:
    ``stopped`` (a Stop fenced the claim, or one of the id is pending: read under the Stop interlock,
    ``stop_interlocked``, so a Stop recorded later finds the hand-off), ``lost`` (TOKEN's claim is not the one
    standing) or ``unknown`` (the store or the Stop projection did not answer)."""
    from ouroboros import store
    from ouroboros.cancel_intents import has_active_intent

    tid = validate_task_id(task_id)

    def advance() -> str:
        with store.open_store(drive_root, write=True) as txn:
            held = txn.get_row(ID_CLAIMS, tid)
            if not held or held.get("token") != token:
                return "lost"
            if held.get("phase") == "stopped" or has_active_intent(drive_root, tid, strict=True):
                return "stopped"
            txn.put_row(ID_CLAIMS, tid, {**held, "phase": phase})
            return ""

    try:
        return stop_interlocked(drive_root, advance)
    except Exception:
        log.warning("The claim of task:%s could not enter phase %s", tid, phase, exc_info=True)
        return "unknown"


def release_provisional_claim(drive_root: Any, task_id: str, token: str) -> bool:
    """A proved rollback's release: True once no claim of TOKEN stands under TASK_ID (deleted now, or absent);
    False when another request's stands or the store did not answer: the id then stays held. A pending Stop does
    not keep it: the rollback removed what this admission staged, a Stop of the id still takes it from every next
    claim (``stop_fenced``), and that Stop's custody finds no claim and no row to settle."""
    from ouroboros import store

    tid = validate_task_id(task_id)
    try:
        with store.open_store(drive_root, write=True) as txn:
            held = txn.get_row(ID_CLAIMS, tid)
            if held is None:
                return True
            if held.get("token") != token:
                return False
            txn.delete_row(ID_CLAIMS, tid)
            return True
    except Exception:
        log.warning("The provisional claim of task:%s was not released; it keeps the id", tid, exc_info=True)
        return False


def provisional_claim(drive_root: Any, task_id: str) -> Optional[Dict[str, Any]]:
    """TASK_ID's provisional claim as the store holds it now, None when there is none. Raises when the store does
    not answer: unknown is never absence."""
    from ouroboros import store

    with store.open_store(drive_root, write=False) as txn:
        return txn.get_row(ID_CLAIMS, validate_task_id(task_id))


def stop_provisional_claim(drive_root: Any, task_id: str) -> str:
    """A Stop's custody found no row under TASK_ID: what holds the id without one, decided in one store
    transaction against the claim's hand-off (``advance_provisional_claim``). ``absent``: nothing, the id names no
    task. ``stopped``: a provisional claim before its hand-off, fenced now (or already), so its admission can never
    queue the task and rolls back. ``handoff``: an admission that may be queuing it right now, which the Stop's
    pending intent meets in the queue or its receipt. Raises when the store does not answer: unknown is never
    absence."""
    from ouroboros import store

    tid = validate_task_id(task_id)
    with store.open_store(drive_root, write=False) as txn:  # the common answer, for any id, needs no write
        if txn.get_row(ID_CLAIMS, tid) is None:
            return "absent"  # a claim made after this read meets the Stop already recorded (``stop_fenced``)
    with store.open_store(drive_root, write=True) as txn:
        held = txn.get_row(ID_CLAIMS, tid)
        if held is None:
            return "absent"
        if held.get("phase") not in _UNSENT_PHASES:
            return "handoff"
        if held.get("phase") != "stopped":
            txn.put_row(ID_CLAIMS, tid, {**held, "phase": "stopped", "stopped_at": utc_now_iso()})
        return "stopped"


def retire_provisional_claim(drive_root: Any, task_id: str, token: str = "",
                             stale: Optional[Dict[str, Any]] = None) -> bool:
    """Retire TASK_ID's provisional claim (TOKEN's, when named) on proof read in the same transaction: a task row
    holds the id now (the admission's receipt or refusal), or the record is still ``stale``, one judged
    ``abandoned`` before it. True once no such claim stands; otherwise, and when the store does not answer, it stays
    and the id stays held."""
    from ouroboros import store

    tid = validate_task_id(task_id)
    try:
        with store.open_store(drive_root, write=True) as txn:
            held = txn.get_row(ID_CLAIMS, tid)
            if held is None or (token and held.get("token") != token):
                return held is None
            if txn.row(tid, "task_id") is None and (stale is None or held != stale):
                return False
            txn.delete_row(ID_CLAIMS, tid)
            return True
    except Exception:
        log.warning("The provisional claim of task:%s was not retired; it keeps the id", tid, exc_info=True)
        return False


def has_provisional_claims(drive_root: Any) -> bool:
    """Whether ``ID_CLAIMS`` holds any provisional claim right now: one indexed ``LIMIT 1`` probe, False when the
    store does not yet exist."""
    from ouroboros import store

    if not store.store_path(drive_root).exists():
        return False
    with store.open_store(drive_root, write=False) as txn:
        return txn.conn.execute("SELECT 1 FROM obligations WHERE set_name=? LIMIT 1", (ID_CLAIMS,)).fetchone() is not None


def retire_settled_claims(
    drive_root: Any, *, queued: Optional[Iterable[str]] = None,
    original_queued: Optional[Iterable[str]] = None,
    stopped_tasks: Optional[Dict[str, Dict[str, Any]]] = None,
    queue_unreadable: bool = False,
    stop: Optional[Callable[[], bool]] = None,
) -> List[str]:
    """After the queue is recovered: retire each provisional claim a task row holds now (a receipt whose own
    retirement never ran) or whose producer ended before any hand-off and left nothing staged. A dead-producer
    hand-off without a row is settled only when the queue's boot state is known (``queued`` or ``queue_unreadable``):
    - under an active Stop/cancel intent: materialized as admitted (``STATUS_SCHEDULED``) so the existing Stop path
      finishes it as ``cancelled``, never ``never_admitted``;
    - when the original saved queue was unreadable before startup publication: recorded as an honest unknown failure
      (``saved_queue_unreadable``), keeping staged files and ID ownership;
    - when present in ``original_queued``: kept held (never proved ``never_admitted``);
    - only when positively absent from the original saved queue, not live, and unstopped: if it left staging, its
      never-admitted refusal takes the id first (Stop-interlocked); otherwise the claim retires Stop-interlocked.
    Without a queue state (``queued is None and not queue_unreadable``) hand-offs stay held. Returns the retired ids."""
    from ouroboros import store
    from ouroboros.cancel_intents import has_active_intent
    from ouroboros.local_custody_repair import owner_ended

    held_ids = set(queued or ())
    original_ids = set(original_queued or ())
    saved_stopped = stopped_tasks or {}
    with store.open_store(drive_root, write=False) as txn:
        claims = txn.set_rows(ID_CLAIMS)
        receipted = {tid for tid in claims if txn.row(tid, "task_id") is not None}
    retired = []
    for tid, claim in claims.items():  # a holder's liveness and files are judged outside any write
        if stop and stop():
            break
        token = str(claim.get("token") or "")
        if tid in receipted:
            if retire_provisional_claim(drive_root, tid, token):
                retired.append(tid)
            continue
        if abandoned(claim):
            if not _staged_remains(drive_root, tid) and retire_provisional_claim(drive_root, tid, token, claim):
                retired.append(tid)
            continue
        producer = claim.get("producer") if isinstance(claim.get("producer"), dict) else {}
        dead_handoff = (claim.get("phase") == "handoff" and tid not in held_ids
                        and owner_ended(producer.get("pid"), producer.get("birth")))
        if not dead_handoff or (queued is None and not queue_unreadable):
            continue
        try:
            stopped = has_active_intent(drive_root, tid, strict=True)
        except Exception:
            log.warning("Active cancel intents for task:%s could not be read; its claim stays", tid, exc_info=True)
            continue
        if stopped:
            if (_record_admitted_handoff_for_stop(drive_root, tid, token, saved_task=saved_stopped.get(tid), stop=stop)
                    and retire_provisional_claim(drive_root, tid, token)):
                retired.append(tid)
        elif queue_unreadable:
            if (_record_unknown_handoff(drive_root, tid, token, staging_retained=_staged_remains(drive_root, tid),
                                        stop=stop)
                    and retire_provisional_claim(drive_root, tid, token)):
                retired.append(tid)
        elif tid in original_ids:
            continue
        elif _staged_remains(drive_root, tid):
            if (_record_unsettled_handoff(drive_root, tid, token, stop=stop)
                    and retire_provisional_claim(drive_root, tid, token)):
                retired.append(tid)
        elif _retire_unstopped_handoff(drive_root, tid, token, claim, stop=stop):
            retired.append(tid)
    return retired


def _record_admitted_handoff_for_stop(drive_root: Any, task_id: str, token: str, *, saved_task: Any = None,
                                      stop: Optional[Callable[[], bool]] = None) -> bool:
    """Materialize a stopped dead-producer hand-off as ``STATUS_SCHEDULED`` under ``while_contended``."""
    from supervisor.task_api_claim import record_admitted_handoff_for_stop

    try:
        return bool(while_contended(
            lambda: record_admitted_handoff_for_stop(drive_root, task_id, token, saved_task=saved_task), stop=stop))
    except Exception:
        log.warning("The stopped hand-off of task:%s was not materialized; its claim keeps the id", task_id,
                    exc_info=True)
        return False


def _record_unknown_handoff(drive_root: Any, task_id: str, token: str, *, staging_retained: bool = False,
                            stop: Optional[Callable[[], bool]] = None) -> bool:
    """Record an unknown-admission failure for a hand-off whose original saved queue was unreadable."""
    from supervisor.task_api_claim import record_unknown_handoff

    refusal_token = uuid.uuid4().hex
    try:
        return bool(while_contended(
            lambda: record_unknown_handoff(drive_root, task_id, token, refusal_token=refusal_token,
                                           staging_retained=staging_retained), stop=stop))
    except Exception:
        log.warning("The unknown hand-off of task:%s was not recorded; its claim keeps the id", task_id,
                    exc_info=True)
        return False


def _record_unsettled_handoff(drive_root: Any, task_id: str, token: str, *,
                              stop: Optional[Callable[[], bool]] = None) -> bool:
    """Whether TASK_ID's never-admitted refusal (``task_api_claim.record_unsettled_handoff``) is the row under it now:
    a busy store is tried again with the same refusal token (``while_contended``); anything else leaves the claim
    holding the id."""
    from supervisor.task_api_claim import record_unsettled_handoff

    refusal_token = uuid.uuid4().hex
    try:
        while_contended(lambda: record_unsettled_handoff(drive_root, task_id, token, refusal_token=refusal_token),
                        stop=stop)
        return True
    except Exception:
        log.warning("The unsettled hand-off of task:%s was not recorded; its claim keeps the id", task_id,
                    exc_info=True)
        return False


def _retire_unstopped_handoff(drive_root: Any, task_id: str, token: str, stale: Dict[str, Any], *,
                              stop: Optional[Callable[[], bool]] = None) -> bool:
    """Retire a proven-unadmitted, un-staged hand-off claim under the Stop interlock so a concurrent Stop never
    loses it to an unlocked delete."""
    from ouroboros import store
    from ouroboros.cancel_intents import has_active_intent

    tid = validate_task_id(task_id)

    def step() -> bool:
        if has_active_intent(drive_root, tid, strict=True):
            return False
        with store.open_store(drive_root, write=True) as txn:
            held = txn.get_row(ID_CLAIMS, tid)
            if held is None:
                return True
            if (token and held.get("token") != token) or (txn.row(tid, "task_id") is None and held != stale):
                return False
            txn.delete_row(ID_CLAIMS, tid)
            return True

    try:
        return bool(stop_interlocked(drive_root, step, stop=stop))
    except Exception:
        log.warning("The provisional claim of task:%s was not retired; it keeps the id", tid, exc_info=True)
        return False


def _staged_remains(drive_root: Any, task_id: str) -> bool:
    """Whether a folder TASK_ID keys stands, or cannot be inspected."""
    try:
        return occupied(retained_task_paths(drive_root, task_id))
    except (OSError, ValueError):
        log.warning("Folders of task:%s could not be inspected; its claim stays", task_id, exc_info=True)
        return True
