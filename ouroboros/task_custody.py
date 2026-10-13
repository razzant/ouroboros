"""Custody of a task's own execution drives and unread mail (TZ-1 A, V10).

ONE coordinator decides whether a child execution drive may be deleted:
``settle_child_drive``. The drive prune (off the loop thread) and admission
rollback call it; nothing else removes a task's drive. A drive goes only when,
for EVERY task occupying it (its owner and, say, a timeout retry that kept the
original drive), the caller's supervisor probe proves no live owner (unknown
retains), the DURABLE canonical row is settled with post-work closed, child
copy-back was adopted or is blocked by cancellation, no ref promotion is
pending, and every obligation the drive still holds is in canonical custody:

* every deliverable a canonical row, the child's result or the store's own
  registration records inside the child store (derived FIRST, then checked
  against the store: an absent store or file is a missing source, never
  permission), plus every unrecorded file the store lists;
* the input closure of the mail addressed to the task (attachments and their
  manifest sources) and of its task contract;
* the child's verification receipts, unioned into the canonical stream;
* every exact unread mailbox line (``unread_mailbox``), held by the canonical row.

Missing, unreadable or mismatched material RETAINS the drive; a later pass
converges after a restore. A recorded immutable capture is published only from
bytes that verify against it; a mutable file is compared and published by its
CURRENT bytes (agreeing recorded digests prove nothing about them; with its
source gone it is held only by a canonical row whose digest its canonical copy
verifies against and no child-side claim contradicts); a canonical file is never
replaced (a differing copy publishes beside it under a content-named collision);
unrecorded files are carried without inventing rows. Every placement is a
create-only link: a name that appeared after the identity check retains the pass.

Ownership interlock. Preparation (verification, staging into private unserved
staging) runs without locks and writes nothing served. The probe is re-asked
BEFORE the custody locks (it takes the supervisor's queue lock; a custody holder
never takes that lock). Phase A takes each occupant's custody lock - the lock
every publisher of the canonical store shares (copy-back, ref retry, host
artifact finalization, mailbox cleanup, this settlement) - re-reads the row
against its attempt basis and custody revision, unions receipts, derives the
input closure from the CURRENT mailbox and the durable unread rows (a held
projection is re-verified, an absent one carried), re-checks each destination's
identity, places staged files, and writes ONE same-status row projected from the
row-locked CURRENT, then re-reads it. The maintenance generation is re-asked
after the locks are taken, before receipts, inputs and files move and at the row
commit, and every write owner asks it right before each copy, placement or unlink
(``publication_fence``): a close observed there starts no further effect. Phase B
runs under the caller's ``guard`` - the supervisor's queue lock, which admission and
assignment hold - and, under fresh custody and mail locks, re-takes the occupant
census, re-asks the probe, re-checks every row and mailbox, and only then moves the
drive out of its place. Nothing under the queue lock copies or hashes, so Phase A is
not atomic with queue admission: an owner admitted after its probe can meet the
settled attempt's create-only canonical files, while the attempt basis refuses the
row write and Phase B's probe keeps the drive. Lock order: supervisor queue lock ->
task custody lock -> task mail lock -> task-result row lock -> jsonl append lock.
Row projectors and effective reads never take a lock.

Unread mail (V10): an accepted terminal transition captures the exact mailbox
lines no attempt acknowledged (read before the row lock, unioned under it), and
custody only grows (``merge_unread_mail``): exact bytes are the identity, never a
``msg_id``. Reads here are pure: ``store_artifact_view`` lists the task's own
stores without hashing, copying or registering anything, ranks identity claims
(immutable capture > measured record > registration > listing) apart from the
file's location, and refuses a conflicting identity instead of downgrading it.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import hashlib
import json
import logging
import os
import pathlib
import shutil
import uuid
from typing import Any, Callable, ContextManager, Dict, Iterator, List, Optional, Tuple

from ouroboros.owner_mailbox import _MAILBOX_DIR as _OWNER_MAILBOX_DIR
from ouroboros.owner_mailbox import _mailbox_path
from ouroboros.store import StoreError
from ouroboros.task_results import load_task_result, validate_task_id, write_task_result
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)

HEADLESS_TASKS_DIR = pathlib.Path("state") / "headless_tasks"
TASK_DRIVES_DIR = pathlib.Path("task_drives")
_ARTIFACTS_DIR = pathlib.Path("task_results") / "artifacts"
_MAILBOX_DIR = pathlib.Path(_OWNER_MAILBOX_DIR)
_STAGING_DIR = pathlib.Path("state") / "custody_staging"
_TRASH_DIR = pathlib.Path("state") / "custody_trash"
# The addressed words a sender is owed; controls (hurry, finalize_now, ...) are not mail.
UNREAD_MAIL_KINDS = frozenset({"owner_text", "task_message", "quiz_answer"})
OWNER_MAIL_KINDS = frozenset({"owner_text", "quiz_answer"})  # what the owner authored
# A store's own bookkeeping and non-deliverable subtrees (inputs, media, sources).
_STORE_BOOKKEEPING = frozenset({".artifact_manifest.json", ".artifact_manifest.json.lock",
                                ".scratch_manifest.json", "verification_receipts.jsonl"})
_STORE_INPUT_SUBTREES = frozenset({"attachments", "chat_media", "source_handles"})
_SETTLED = frozenset({"completed", "failed", "cancelled", "rejected_duplicate"})
_INPUT_FIELDS = ("task_contract", "attachment_manifest", "attachment_manifest_ref", "metadata")
# Under the supervisor's queue lock a contended per-task lock means a publisher or sender
# is mid-flight: the drive waits for the next pass instead of the loop waiting for a file.
_INTERLOCK_LOCK_TIMEOUT_SEC = 0.5

LiveProbe = Callable[[str], Optional[bool]]
Guard = Callable[[], ContextManager[bool]]


class _Retained(Exception):
    """A typed reason this drive stays: custody is not proven yet."""


class CustodyBusy(OSError):
    """A publisher could not take the task's custody lock: retry later, nothing failed."""


class PublicationClosed(OSError):
    """The maintenance generation closed: a fenced publication starts no further write or unlink."""


_PUBLICATION_FENCE: contextvars.ContextVar = contextvars.ContextVar("custody_publication_fence", default=None)


@contextlib.contextmanager
def publication_fence(closed: Optional[Callable[[], bool]]) -> Iterator[None]:
    """Scope a maintenance generation's ``closed()`` over one publication: each write owner
    under it (``artifacts.copy_artifact_file``, ``store_actor_source_bytes``, the observability
    CAS publication, the receipt union, a placement, a mailbox unlink) asks it right before its
    write or unlink, after any lock wait (``fence_publication``). A write already past its
    check finishes; a materialization that fails removes only files it created itself. Nothing
    else is rolled back: the next generation re-derives from what landed. None keeps the scope."""
    token = _PUBLICATION_FENCE.set(closed) if closed is not None else None
    try:
        yield
    finally:
        if token is not None:
            _PUBLICATION_FENCE.reset(token)


def fence_publication() -> None:
    """Raise ``PublicationClosed`` once the enclosing ``publication_fence`` closed; a no-op outside one."""
    closed = _PUBLICATION_FENCE.get()
    if closed is not None and closed():
        raise PublicationClosed("the maintenance generation closed before this write")


def custody_anchor(drive_root: Any, task_id: str) -> pathlib.Path:
    """The canonical data root owning a drive by host layout, else the drive itself.

    Any ``state/headless_tasks/<id>/data`` or ``task_drives/<id>`` drive anchors on its
    canonical root whichever task writes into it (a timeout retry keeps the original
    drive), so every writer of one mailbox takes the same lock.
    """
    validate_task_id(task_id)
    return canonical_owner_root(drive_root)


def canonical_owner_root(drive_root: Any) -> pathlib.Path:
    """``custody_anchor``'s layout rule alone, for writers without a task id."""
    root = pathlib.Path(drive_root).resolve(strict=False)
    if root.name == "data" and len(root.parents) > 3 and root.parents[1].name == "headless_tasks" \
            and root.parents[2].name == "state":
        return root.parents[3]
    if len(root.parents) > 1 and root.parent.name == "task_drives":
        return root.parents[1]
    return root


@contextlib.contextmanager
def _task_lock(drive_root: Any, task_id: str, suffix: str, timeout_sec: float) -> Iterator[bool]:
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

    tid = validate_task_id(task_id)
    path = custody_anchor(drive_root, tid) / "task_results" / f"{tid}.{suffix}.lock"
    fd = acquire_exclusive_file_lock(path, timeout_sec=timeout_sec, stale_sec=120.0, poll_sec=0.01,
                                     owner_aware_stale=True)
    try:
        yield fd is not None
    finally:
        if fd is not None:
            release_exclusive_file_lock(path, fd)


def task_custody_lock(drive_root: Any, task_id: str, *, timeout_sec: float = 5.0) -> ContextManager[bool]:
    """The one per-task serialization of canonical-store PUBLICATION: copy-back and drive
    settlement take it around every file they place and the row write that lists it. It
    lives on the canonical side (``task_results/<id>.custody.lock``), never inside a drive
    that settlement deletes, for the canonical root and every own child drive alike.
    Yields whether it was acquired."""
    return _task_lock(drive_root, task_id, "custody", timeout_sec)


def task_mail_lock(drive_root: Any, task_id: str, *, timeout_sec: float = 5.0) -> ContextManager[bool]:
    """The per-task serialization of MAILBOX effects: appends, acknowledgements, mailbox
    cleanup and the final settlement check, so no row lands between a custody read and the
    unlink or move it permits. Short holds only; never held across a copy."""
    return _task_lock(drive_root, task_id, "mail", timeout_sec)


def own_child_drives(canonical_root: Any, task_id: str) -> List[pathlib.Path]:
    """TASK's own execution drives from the host layout alone, never created.

    No result field names them, so a forged row cannot widen what a read or a
    settlement touches; a drive reached through a symlink is not own.
    """
    canonical, tid = pathlib.Path(canonical_root).resolve(strict=False), validate_task_id(task_id)
    drives = (canonical / HEADLESS_TASKS_DIR / tid / "data", canonical / TASK_DRIVES_DIR / tid)
    return [drive for drive in drives if drive.is_dir() and drive.resolve(strict=False) == drive]


def task_artifact_stores(canonical_root: Any, task_id: str) -> List[pathlib.Path]:
    """Stores that may serve TASK's files: the canonical one (never resolved further, so
    a path through a symlinked component belongs to no store), then each own drive's."""
    canonical, tid = pathlib.Path(canonical_root).resolve(strict=False), validate_task_id(task_id)
    stores = [canonical / _ARTIFACTS_DIR / tid]
    for drive in own_child_drives(canonical, tid):
        store = drive / _ARTIFACTS_DIR / tid
        if store.resolve(strict=False) == store:
            stores.append(store)
    return stores


def store_relpath(store: pathlib.Path, raw_path: Any) -> str:
    """POSIX path of a recorded file inside ``store`` (symlinks resolved), else ''."""
    text = str(raw_path or "").strip()
    if not text:
        return ""
    path = pathlib.Path(text).resolve(strict=False)
    return path.relative_to(store).as_posix() if path != store and path.is_relative_to(store) else ""


# ----------------------------------------------------------------- unread mail


def _row_key(row: str) -> str:
    """A captured row's identity: its exact bytes (a ``msg_id`` proves nothing about them)."""
    return "sha:" + hashlib.sha256(row.encode("utf-8")).hexdigest()


def unread_mail_rows(drive_root: Any, task_id: str) -> Tuple[List[str], bool]:
    """Exact mailbox lines of addressed mail no attempt acknowledged, and whether the read
    was complete (a torn or failed read is never proof of emptiness). Reads only.

    The drain delivers the FIRST line of a ``msg_id`` and acknowledges that id; a later
    line reusing the id was never delivered, so it stays unread whatever the ack says."""
    from ouroboros.owner_mailbox import acknowledged_task_message_ids, mailbox_lines

    path = _mailbox_path(drive_root, task_id)
    if not path.exists():
        return [], True
    status: Dict[str, bool] = {}
    acknowledged = acknowledged_task_message_ids(pathlib.Path(drive_root), task_id, _read_status=status)
    complete = bool(status.get("complete"))
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return [], False
    complete = complete and (not content or content.endswith("\n"))
    rows: List[str] = []
    delivered: set = set()
    for line in mailbox_lines(content):
        try:
            entry = json.loads(line)
        except ValueError:
            complete = False
            continue
        if not isinstance(entry, dict):
            complete = False
            continue
        if str(entry.get("kind") or "owner_text") not in UNREAD_MAIL_KINDS:
            continue
        msg_id = str(entry.get("msg_id") or "")
        first = msg_id not in delivered
        if msg_id:
            delivered.add(msg_id)
        if msg_id and first and msg_id in acknowledged:
            continue  # the one delivered line of this id: read
        rows.append(line)
    return rows, complete


def merge_unread_mail(*values: Any) -> Optional[Dict[str, Any]]:
    """Union of ``unread_mailbox`` custody values: exact rows never shrink (the first copy
    of a line wins), ``inputs`` (canonical input closures per row) accumulate, and
    ``read_complete`` is the latest capture's (``captured_at``; a later complete read
    covers every row still unread). None when no value is a capture."""
    rows: List[str] = []
    keys: set = set()
    inputs: Dict[str, Any] = {}
    complete: Optional[bool] = None
    latest = ""
    for value in values:
        if not isinstance(value, dict):
            continue
        for row in value.get("rows") or []:
            if isinstance(row, str) and (key := _row_key(row)) not in keys:
                keys.add(key)
                rows.append(row)
        if isinstance(value.get("inputs"), dict):
            inputs.update({str(key): item for key, item in value["inputs"].items() if isinstance(item, dict)})
        stamp = str(value.get("captured_at") or "")
        if complete is None or stamp >= latest:
            latest, complete = stamp, value.get("read_complete") is True
    if complete is None:
        return None
    custody = {"schema": 1, "total": len(rows), "rows": rows, "read_complete": complete, "captured_at": latest}
    return {**custody, "inputs": {key: inputs[key] for key in inputs if key in keys}} if inputs else custody


def owner_mail_rows(drive_root: Any, task_id: str) -> Tuple[List[str], bool]:
    """Exact OWNER-authored mailbox lines, read AND unread, and whether the read was
    complete. Reads only. ``unread_mail_rows`` keeps what no attempt read; this keeps
    what the owner said, since an acknowledged correction's last addressable copy is
    the mailbox the settled cleanup unlinks (owner Batch4, Continue)."""
    from ouroboros.owner_mailbox import mailbox_lines

    path = _mailbox_path(drive_root, task_id)
    if not path.exists():
        return [], True
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return [], False
    complete = not content or content.endswith("\n")
    rows: List[str] = []
    for line in mailbox_lines(content):
        try:
            entry = json.loads(line)
        except ValueError:
            complete = False
            continue
        if not isinstance(entry, dict):
            complete = False
        elif str(entry.get("kind") or "owner_text") in OWNER_MAIL_KINDS:
            rows.append(line)
    return rows, complete


def capture_owner_mail(root: Any, task_id: str) -> Optional[Dict[str, Any]]:
    """The terminal capture of a ROOT's owner rows (``owner_mailbox``): its own mailbox and
    each own child drive's, in the ``unread_mailbox`` shape (exact rows, grow-only). Always a
    capture, an empty one included — it proves the mailbox was read at the terminal, so a
    later reader can tell "nothing was said" from "not captured". Never ACKs; never raises
    (a failure is ``read_complete=False``, which a Continue refuses as a gap)."""
    rows: List[str] = []
    complete = True
    try:
        for drive in [pathlib.Path(root), *own_child_drives(root, task_id)]:
            found, whole = owner_mail_rows(drive, task_id)
            rows.extend(found)
            complete = complete and whole
    except Exception:
        log.warning("Owner mail capture failed for %s", task_id, exc_info=True)
        complete = False
    return merge_unread_mail({"rows": rows, "read_complete": complete, "captured_at": utc_now_iso()})


def unread_mail_held(custody: Any, rows: List[str]) -> bool:
    """Whether the canonical custody keeps every one of these exact rows."""
    held = merge_unread_mail(custody) or {"rows": []}
    return not ({_row_key(row) for row in rows} - {_row_key(row) for row in held["rows"]})


def capture_unread_mail(root: Any, task_id: str) -> Optional[Dict[str, Any]]:
    """The terminal capture of TASK's unread mail under ROOT: its own mailbox and each own
    child drive's. Never ACKs; never raises (a failure is an incomplete capture); None
    when every mailbox was read completely and held no unread row."""
    rows: List[str] = []
    complete = True
    try:
        for drive in [pathlib.Path(root), *own_child_drives(root, task_id)]:
            found, whole = unread_mail_rows(drive, task_id)
            rows.extend(found)
            complete = complete and whole
    except Exception:
        log.warning("Unread mail capture failed for %s", task_id, exc_info=True)
        complete = False
    if not rows and complete:
        return None
    return merge_unread_mail({"rows": rows, "read_complete": complete, "captured_at": utc_now_iso()})


def unread_mail_notice(custody: Any, *, preview_chars: int = 300, limit: int = 10) -> str:
    """One bounded host line naming the unread mail a result holds and where its exact rows are."""
    held = merge_unread_mail(custody)
    if not held or not held["rows"]:
        return ""
    shown = []
    for row in held["rows"][:limit]:
        try:
            entry = json.loads(row)
        except (TypeError, ValueError):
            entry = {}
        entry = entry if isinstance(entry, dict) else {}
        text = str(entry.get("text") or "")
        source = str(entry.get("source_task_id") or "owner")
        shown.append(f"- {entry.get('ts') or '?'} {entry.get('kind') or 'owner_text'} from {source}: "
                     + text[:preview_chars] + (f" [... {len(text) - preview_chars} more chars]"
                                               if len(text) > preview_chars else ""))
    more = held["total"] - len(shown)
    return (f"[UNREAD_MAILBOX] {held['total']} message(s) written to this task's mailbox were never read by "
            "its model" + ("" if held["read_complete"] else " (the last mailbox read was incomplete)") + ":\n"
            + "\n".join(shown) + (f"\n- ... {more} more" if more > 0 else "")
            + "\nExact rows: get_task_result(include_authority=True) -> authority.unread_mailbox.rows.")


def _attachment_bearing(entry: Any) -> bool:
    return isinstance(entry, dict) and bool(entry.get("attachment_manifest") or "attachment_manifest_ref" in entry)


def _input_keys(rows: List[str]) -> set:
    """Keys of the exact rows that carry inputs (attachments inline or by manifest ref)."""
    keys = set()
    for line in rows:
        try:
            if _attachment_bearing(json.loads(line)):
                keys.add(_row_key(line))
        except ValueError:
            continue
    return keys


def _mail_fields(current: Dict[str, Any], rows: List[str], inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The ``unread_mailbox`` refresh CURRENT still owes for these exact ROWS and their canonical
    input closures (every input-bearing row needs one), {} when it already holds them. A refresh
    that still lacks a closure leaves the owed state visible to the caller's re-read."""
    held = merge_unread_mail(current.get("unread_mailbox"))
    have = (held or {}).get("inputs") or {}
    if unread_mail_held(held, rows) and all(key in have for key in {*(inputs or {}), *_input_keys(rows)}):
        return {}
    capture = {"rows": rows, "read_complete": True, "captured_at": utc_now_iso()}
    return {"unread_mailbox": merge_unread_mail(held, {**capture, "inputs": inputs or {}})}


def _owner_mail_fields(current: Dict[str, Any], captured: Dict[str, Any]) -> Dict[str, Any]:
    """Deletion requires complete current capture and exact canonical readback, ACKs included."""
    if captured.get("read_complete") is not True:
        raise _Retained("owner_mailbox_unreadable")
    held = current.get("owner_mailbox") or {}
    if held.get("read_complete") is True and unread_mail_held(held, captured["rows"]):
        return {}
    return {"owner_mailbox": merge_unread_mail(held, captured)}


def settle_task_mailbox(canonical_root: Any, task_id: str, mailbox_root: Any, *, carry_inputs: bool = True,
                        stop: Optional[Callable[[], bool]] = None) -> bool:
    """Unlink one mailbox and its acks only once the task is settled with post-work and input
    copy closed (``owner_mailbox.settled_mailbox_cleanup_allowed``) and the canonical row
    durably holds every owner row (acknowledged ones included), every unread row
    AND a verified canonical projection of every input-bearing
    row's attachments (a drive's acknowledged inputs are carried through
    ``promote_owner_attachments`` first). Carrying or re-verifying inputs copies and hashes
    under the custody lock, so ``carry_inputs=False`` (the loop thread's task-done seam) keeps
    such a mailbox for the off-loop sweep or the drive settlement. ``stop()`` is the off-loop
    generation's close, fenced over every copy and asked again after each lock wait, at the
    row commit and before the unlinks: a close keeps the mailbox. A no-op retry rewrites
    nothing. Returns whether the mailbox is gone."""
    from ouroboros.owner_mailbox import promote_owner_attachments, settled_mailbox_cleanup_allowed

    canonical, source = pathlib.Path(canonical_root).resolve(strict=False), pathlib.Path(mailbox_root)
    mailbox = _mailbox_path(mailbox_root, task_id)
    acks = mailbox.with_name(f"{validate_task_id(task_id)}.acks.jsonl")
    with publication_fence(stop):
        try:
            current = load_task_result(canonical, task_id, strict=True) or {}
            if not settled_mailbox_cleanup_allowed(current):
                return False  # the row (or its absence) still owns this mailbox
            inputs: Dict[str, Any] = {}
            if mailbox.exists():
                rows, complete = unread_mail_rows(mailbox_root, task_id)
                if not complete:
                    raise _Retained("unread_mailbox_unreadable")
                if _input_keys(rows) or source.resolve(strict=False) != canonical:
                    if not carry_inputs:
                        return False  # inputs to carry or verify: not on this thread
                    with task_custody_lock(canonical, task_id, timeout_sec=_INTERLOCK_LOCK_TIMEOUT_SEC) as locked:
                        if not locked:
                            return False
                        fence_publication()
                        if source.resolve(strict=False) != canonical:
                            state: Dict[str, Any] = {"pending_refs": [], "unavailable_refs": [], "promoted_source_handle_count": 0}
                            promote_owner_attachments(canonical, source, task_id, state)
                            if state["pending_refs"]:
                                raise _Retained("inputs_uncustodied")
                        inputs = _mail_closure(canonical, source, task_id, current, rows)
            with task_mail_lock(mailbox_root, task_id) as locked:
                if not locked:
                    return False
                fence_publication()
                current = load_task_result(canonical, task_id, strict=True) or {}
                if not settled_mailbox_cleanup_allowed(current):
                    return False
                if mailbox.exists():
                    rows, complete = unread_mail_rows(mailbox_root, task_id)
                    if not complete:
                        raise _Retained("unread_mailbox_unreadable")
                    owner_capture = capture_owner_mail(canonical, task_id)
                    def mail_fields(row):
                        return {**_mail_fields(row, rows, inputs),
                                **_owner_mail_fields(row, owner_capture)}
                    if mail_fields(current):
                        _write_custody_fields(canonical, task_id, current, mail_fields,
                                              closed=stop)
                        readback = load_task_result(canonical, task_id, strict=True) or {}
                        if mail_fields(readback):
                            return False  # rows landed since the closure, or inputs are owed: the next call carries them
                    fence_publication()
                    mailbox.unlink()
                acks.unlink(missing_ok=True)
                return True
        except (_Retained, OSError, ValueError, TimeoutError, StoreError):
            if stop is None or not stop():
                log.warning("Mailbox of %s kept: its unread rows are not in canonical custody", task_id, exc_info=True)
            return not mailbox.exists()


def _write_custody_fields(canonical_root: Any, task_id: str, observed: Dict[str, Any],
                          fields_for: Callable[[Dict[str, Any]], Dict[str, Any]],
                          basis: Optional[tuple] = None, applied: Optional[List[bool]] = None,
                          closed: Optional[Callable[[], bool]] = None) -> None:
    """Same-status write of the custody fields ``fields_for(CURRENT)`` yields for the
    row-locked CURRENT (never a row read earlier): a changed attempt basis aborts, a covered
    projection writes nothing (no timestamp churn), a generation ``closed()`` at the commit
    retains. ``applied`` gets True on a real write."""
    def project(current: Dict[str, Any], _incoming: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if closed is not None and closed():
            raise _Retained("generation_closed")  # re-asked at the commit itself, not only before the walk
        if not current or (basis is not None and attempt_basis(current) != basis):
            return None
        patch = {key: value for key, value in fields_for(current).items() if current.get(key) != value}
        if not patch:
            return None
        if applied is not None:
            applied.append(True)
        return {**patch, "status": current["status"]}

    write_task_result(canonical_root, task_id, str(observed.get("status") or ""), _field_projector=project,
                      strict_existing_dict=True)


# ------------------------------------------------------------ store view (pure)


def _registrations(store: pathlib.Path) -> Dict[str, Dict[str, Any]]:
    """A store's host registrations (recorded identities of top-level files); an existing but
    unreadable manifest is ``_Retained``: its identities cannot be proven absent."""
    from ouroboros.utils import read_json_dict

    path = store / ".artifact_manifest.json"
    document = read_json_dict(path)
    if document is None and (path.exists() or path.is_symlink()):
        raise _Retained("artifact_registration_unreadable")
    manifest = (document or {}).get("artifacts")
    return {str(key): dict(row) for key, row in manifest.items() if isinstance(row, dict)} \
        if isinstance(manifest, dict) else {}


def _identity_rank(row: Dict[str, Any]) -> int:
    """Recorded immutable capture > measured record > unmeasured listing/record."""
    if row.get("immutable") and row.get("sha256"):
        return 3
    return 2 if row.get("sha256") and row.get("measured") is not False else 1


def _conflicting(first: Dict[str, Any], second: Dict[str, Any]) -> bool:
    """Two identity claims (rank >= 2) that name different bytes for one relpath."""
    return _identity_rank(first) >= 2 and _identity_rank(second) >= 2 and (
        first.get("sha256") != second.get("sha256")
        or (isinstance(first.get("size"), int) and isinstance(second.get("size"), int) and first["size"] != second["size"]))


def store_artifact_view(canonical_root: Any, task_id: str, recorded: List[Dict[str, Any]],
                        child_rows: Optional[Dict[pathlib.Path, List[Dict[str, Any]]]] = None) -> List[Dict[str, Any]]:
    """The artifact rows a read shows: ``recorded`` canonical rows, own child stores'
    recorded rows, every own store's registrations (a registration whose file is gone still
    shows its identity) and unmeasured listings (size from stat, ``measured: False``). One
    relpath is one row. Its IDENTITY is the highest-ranked claim wherever it was recorded
    (immutable capture > measured record > listing); two claims that name different bytes
    are a refusal (``status: failed``, the conflict in ``errors``), never a downgrade to the
    weaker one. Its LOCATION is the canonical store's copy when one is listed there, else
    the child's. Pure: nothing is hashed, copied or registered."""
    from ouroboros.artifacts import collect_task_artifact_records, merge_artifact_records

    stores = task_artifact_stores(canonical_root, task_id)
    claims: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}  # rel -> [(store index, row)]
    order: List[str] = []
    free: List[Dict[str, Any]] = []

    def claim(row: Dict[str, Any], *, free_ok: bool) -> None:
        if row.get("kind") == "task_source":
            return  # retention authority is not deliverable identity
        located = next(((index, rel) for index, store in enumerate(stores)
                        if (rel := store_relpath(store, row.get("path")))), None)
        if located is None:
            if free_ok:
                free.append(row)
            return
        index, rel = located
        if rel not in claims:
            order.append(rel)
        claims.setdefault(rel, []).append((index, row))

    for row in recorded:
        if isinstance(row, dict):
            claim(dict(row), free_ok=True)
    for store, rows in (child_rows or {}).items():
        for row in rows:
            if isinstance(row, dict) and store_relpath(store, row.get("path")):
                claim(dict(row), free_ok=False)
    for index, store in enumerate(stores):
        try:
            registered = _registrations(store)
        except _Retained:
            registered = {}
        for name, row in registered.items():
            claim({**row, "path": str(store / name)}, free_ok=False)
        for row in collect_task_artifact_records(store.parents[2], task_id, measure=False):
            claim(row, free_ok=False)

    rows: List[Dict[str, Any]] = []
    for rel in order:
        placed = claims[rel]
        identity = max((row for _index, row in placed), key=_identity_rank)
        listed = [(index, row) for index, row in placed if row.get("measured") is False]
        location = min(listed, default=(0, identity), key=lambda item: item[0])[1]
        row = {**identity, "path": location["path"], "name": identity.get("name") or location.get("name")}
        row.update({"relpath": rel} if "/" in rel else {})
        conflicts = sorted({str(other.get("sha256")) for _index, other in placed if _conflicting(identity, other)})
        if conflicts:
            row.update(status="failed", errors=[*(row.get("errors") or []), "recorded identities conflict for "
                       f"this file: {identity.get('sha256')} against {', '.join(conflicts)}"])
        rows.append(row)
    return merge_artifact_records(free, rows)


# ----------------------------------------------------------------- settlement


def attempt_basis(row: Dict[str, Any]) -> tuple:
    """Status plus the real attempt identity of a row (never status and ts alone)."""
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return (row.get("status"), row.get("task_attempt"), row.get("_attempt"), row.get("started_at"),
            metadata.get("attempt"), metadata.get("task_attempt"), row.get("child_drive_root"))


def custody_revision(row: Dict[str, Any]) -> str:
    """A digest of the fields custody publishes, so a concurrent publisher's write between
    two looks at the row is seen even under the same status and attempt."""
    fields = {key: row.get(key) for key in ("artifacts", "artifact_bundle", "child_ref_promotion", "unread_mailbox",
                                             *_INPUT_FIELDS)}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _identity(path: pathlib.Path) -> Optional[tuple]:
    """A destination's on-disk identity without following links; None when absent."""
    try:
        observed = path.lstat()
    except FileNotFoundError:
        return None
    return (observed.st_dev, observed.st_ino, observed.st_size, observed.st_mtime_ns, observed.st_ctime_ns)


def _occupants(drive: pathlib.Path, task_id: str) -> List[str]:
    """Every task whose result, store or mailbox lives in DRIVE (a timeout retry keeps the
    original drive); a name that is not a task id retains the drive."""
    from ouroboros.store import StoreError
    from ouroboros.task_store import task_ids

    ids = {task_id}
    try:
        ids.update(task_ids(drive))
    except StoreError as exc:
        raise _Retained("drive_occupant_unknown") from exc
    for directory, suffix in ((drive / _ARTIFACTS_DIR, ""), (drive / _MAILBOX_DIR, ".jsonl")):
        if not directory.is_dir():
            continue
        for entry in directory.iterdir():
            name = entry.name
            if suffix and (not name.endswith(suffix) or name.endswith(".acks.jsonl")):
                continue
            if not suffix and not entry.is_dir():
                continue
            stem = name[: -len(suffix)] if suffix else name
            try:
                ids.add(validate_task_id(stem))
            except ValueError as exc:
                raise _Retained("drive_occupant_unknown") from exc
    return sorted(ids)


def _verified(path: pathlib.Path, expected: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The measured identity of a regular canonical file matching EXPECTED (sha/size when
    it names them), else None."""
    from ouroboros.artifacts import stream_artifact_file

    if not path.is_file() or path.is_symlink():
        return None
    try:
        return stream_artifact_file(path, expected=expected)
    except (OSError, ValueError):
        return None


def _held_copy(target: pathlib.Path, canonical_rows: List[Dict[str, Any]], row: Dict[str, Any],
               preferred: pathlib.Path) -> str:
    """The canonical relpath already holding ROW's recorded immutable bytes (its own relpath
    first, then any canonical row with the same identity, e.g. a collision name), verified
    now; '' when none does."""
    rels = [store_relpath(target, preferred)] + [store_relpath(target, item.get("path")) for item in canonical_rows
                                                 if item.get("sha256") == row.get("sha256")
                                                 and item.get("size") == row.get("size")]
    return next((rel for rel in rels if rel and _verified(target.joinpath(*rel.split("/")), row)), "")


def _collision(dest: pathlib.Path, digest: str) -> pathlib.Path:
    """Where different bytes publish beside a canonical file that keeps its name."""
    stem, dot, suffix = dest.name.rpartition(".")
    return dest.with_name(f"{stem}.{digest[:8]}.{suffix}" if dot and stem else f"{dest.name}.{digest[:8]}")


def _obligations(source: pathlib.Path, current: Dict[str, Any], child: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Every recorded deliverable inside the child store SOURCE, by relpath, derived from the
    canonical row, the child's result and the store's registrations BEFORE the store is
    looked at; the highest-ranked identity wins, two immutable captures that disagree are a
    conflict."""
    recorded: Dict[str, Dict[str, Any]] = {}
    registrations = [{**row, "path": str(source / name)} for name, row in _registrations(source).items()] \
        if source.is_dir() else []
    rows = [*registrations, *(child.get("artifacts") or []), *(current.get("artifacts") or [])]
    for row in rows:
        rel = store_relpath(source, row.get("path")) if isinstance(row, dict) else ""
        if not rel:
            continue
        if rel.rsplit("/", 1)[-1] in _STORE_BOOKKEEPING:
            raise _Retained("artifact_obligation_reserved")
        seen = recorded.get(rel)
        if seen is not None and _conflicting(seen, row) and (seen.get("immutable") or row.get("immutable")):
            raise _Retained("artifact_identity_conflict")
        if seen is None or _identity_rank(row) >= _identity_rank(seen):
            recorded[rel] = dict(row)
    return recorded


def _listed(source: pathlib.Path) -> Dict[str, pathlib.Path]:
    """The regular files SOURCE holds outside its bookkeeping and input subtrees."""
    from ouroboros.artifacts import iter_artifact_tree

    if not (source.exists() or source.is_symlink()):
        return {}
    if source.resolve(strict=False) != source:
        raise _Retained("child_artifact_store_unsafe")
    listed: Dict[str, pathlib.Path] = {}
    try:
        for path in iter_artifact_tree(source):
            rel = path.relative_to(source).as_posix()
            if path.is_symlink() or not path.is_file() or path.name in _STORE_BOOKKEEPING \
                    or rel.split("/", 1)[0] in _STORE_INPUT_SUBTREES:
                continue
            listed[rel] = path
    except OSError as exc:
        raise _Retained("child_artifact_store_unreadable") from exc
    return listed


def _child_store_plan(canonical: pathlib.Path, drive: pathlib.Path, task_id: str, current: Dict[str, Any],
                      child: Dict[str, Any], staging: pathlib.Path) -> List[Dict[str, Any]]:
    """Stage every obligation of TASK's store in DRIVE the canonical store does not hold yet.
    Returns ``{rel, row?, staged?, dest_identity}`` publication items (``rel`` canonical;
    ``row`` None for an unrecorded file, carried without a row). Missing, unreadable or
    mismatched material raises ``_Retained``; nothing touches the served store."""
    from ouroboros.artifacts import copy_artifact_file, stream_artifact_file

    source, target = drive / _ARTIFACTS_DIR / task_id, canonical / _ARTIFACTS_DIR / task_id
    canonical_rows = [row for row in current.get("artifacts") or [] if isinstance(row, dict)]
    listed = _listed(source)
    recorded = _obligations(source, current, child)
    if target.resolve(strict=False) != target:
        raise _Retained("canonical_store_unsafe")
    canonical_by_rel = {rel: row for row in canonical_rows if (rel := store_relpath(target, row.get("path")))}
    plan: List[Dict[str, Any]] = []
    for rel in sorted(set(recorded) | set(listed)):
        row = recorded.get(rel)
        dest = target.joinpath(*rel.split("/"))
        if dest.parent.resolve(strict=False) != target.joinpath(*rel.split("/")[:-1]) or dest.is_symlink():
            raise _Retained("canonical_store_unsafe")  # a linked component would redirect the write
        if row is not None and row.get("immutable"):
            if not (isinstance(row.get("size"), int) and row.get("sha256")):
                raise _Retained("artifact_identity_incomplete")
            if held := _held_copy(target, canonical_rows, row, dest):
                plan.append({"rel": held, "row": row, "dest_identity": _identity(target.joinpath(*held.split("/")))})
                continue
            if rel not in listed:
                raise _Retained("artifact_source_missing")
            expected: Optional[Dict[str, Any]] = row
        else:
            expected = None
            if rel not in listed:
                # A recorded mutable file whose source is gone is held only by a canonical row
                # recording a digest its canonical copy verifies against AND that no child-side
                # claim contradicts: anything less is convenient metadata, not preserved proof.
                canonical_row = canonical_by_rel.get(rel)
                if canonical_row is None or not canonical_row.get("sha256") or _verified(dest, canonical_row) is None \
                        or (row is not None and row.get("sha256") and row["sha256"] != canonical_row["sha256"]):
                    raise _Retained("artifact_source_missing")
                continue  # copy-back published it; the stale child-path row is superseded
            try:
                observed = stream_artifact_file(listed[rel])  # the child's bytes NOW, whatever any record says
            except OSError as exc:
                raise _Retained("child_artifact_store_unreadable") from exc
            if dest.is_file() and _verified(dest, observed) is not None:  # already published (the usual case)
                if row is not None:
                    plan.append({"rel": rel, "row": {**row, **observed}, "dest_identity": _identity(dest)})
                continue
        staged = staging / str(len(plan))
        staged.parent.mkdir(parents=True, exist_ok=True)
        try:
            measured = copy_artifact_file(listed[rel], staged, expected=expected)
        except OSError as exc:
            raise _Retained("artifact_source_mismatch" if expected is not None else "child_artifact_store_unreadable") from exc
        if dest.exists() or dest.is_symlink():  # different bytes: publish beside, never replace
            beside = _collision(dest, measured["sha256"])
            if beside.exists() or beside.is_symlink():
                if _verified(beside, measured) is None:
                    raise _Retained("artifact_identity_conflict")
                staged.unlink(missing_ok=True)
                if row is not None:
                    plan.append({"rel": store_relpath(target, beside), "row": {**row, **measured},
                                 "dest_identity": _identity(beside)})
                continue
            dest = beside
        plan.append({"rel": store_relpath(target, dest), "row": {**row, **measured} if row is not None else None,
                     "staged": staged, "dest_identity": None})
    return plan


def _publish(canonical: pathlib.Path, task_id: str, plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Under the custody lock: re-check every destination's identity against preparation,
    place staged files create-only (a hard link fails on an existing name where a rename
    would replace it; the name that appeared meanwhile retains the pass), each placement
    fenced by the generation, and return the canonical rows they now back."""
    from ouroboros.artifacts import _register_task_artifact_records

    target = canonical / _ARTIFACTS_DIR / task_id
    rows = []
    for item in plan:
        dest = target.joinpath(*item["rel"].split("/"))
        if _identity(dest) != item["dest_identity"] or dest.parent.resolve(strict=False) != dest.parent:
            raise _Retained("destination_changed")
        if item.get("staged") is not None:
            fence_publication()  # per placement: a close stops the rest; placed files stay create-only
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(item["staged"], dest)
            except FileExistsError as exc:
                raise _Retained("destination_changed") from exc
            os.unlink(item["staged"])
            item["dest_identity"] = _identity(dest)
        if item.get("row") is None:
            continue
        row = {key: value for key, value in item["row"].items() if key not in {"copy_status", "copy_error", "measured"}}
        row.update(path=str(dest), name=dest.name)
        row.update({"relpath": item["rel"]} if "/" in item["rel"] else {})
        if item.get("staged") is not None:
            row.update(status="ready", errors=[])
        rows.append(row)
    # Source registrations remain custody obligations without entering the result
    # bundle. Publish them before its row lock; failed registration retains the drive.
    sources = [row for row in rows if row.get("kind") == "task_source"]
    if sources:
        fence_publication()
        _register_task_artifact_records(target, sources)
    return rows


def _custody_fields(current: Dict[str, Any], source: pathlib.Path, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """CURRENT's artifact rows with every row inside the child store replaced by its canonical
    publication (lifecycle fields untouched; the bundle re-derives from the rows)."""
    from ouroboros.artifacts import merge_artifact_records
    from ouroboros.outcomes import artifact_bundle_from_result

    kept = [row for row in current.get("artifacts") or [] if isinstance(row, dict)
            and row.get("kind") != "task_source" and not store_relpath(source, row.get("path"))]
    artifacts = merge_artifact_records(kept, (row for row in rows if row.get("kind") != "task_source"))
    if artifacts == current.get("artifacts") or (not artifacts and not current.get("artifacts")):
        return {}
    # Re-derived from the lifecycle status and the rows, never from the superseded bundle's own status.
    bundle = artifact_bundle_from_result({**current, "artifacts": artifacts, "artifact_bundle": None})
    return {"artifacts": artifacts, "artifact_bundle": bundle}


def _verify_readback(row: Dict[str, Any], source: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
    listed = {(str(item.get("path")), item.get("sha256")) for item in row.get("artifacts") or [] if isinstance(item, dict)}
    for item in rows:
        if item.get("kind") == "task_source":
            registered = _registrations(pathlib.Path(item["path"]).parent).get(item["name"], {})
            if any(registered.get(key) != item.get(key) for key in ("kind", "path", "size", "sha256", "immutable")):
                raise _Retained("canonical_source_registration_missing")
        elif (item["path"], item.get("sha256")) not in listed:
            raise _Retained("canonical_listing_missing")
    if any(store_relpath(source, item.get("path")) for item in row.get("artifacts") or [] if isinstance(item, dict)):
        raise _Retained("canonical_listing_references_child")


def _copied_back(drive: pathlib.Path, row: Dict[str, Any]) -> bool:
    """Whether copy-back published this drive's child result onto ROW (its promotion mark)."""
    promotion, source = row.get("child_ref_promotion"), str(row.get("headless_child_drive_root") or "")
    return bool(isinstance(promotion, dict) and promotion.get("schema_version") == 1
                and promotion.get("status") in {"complete", "incomplete"}
                and source and pathlib.Path(source).resolve(strict=False) == drive)


def _pending_refs_under(row: Dict[str, Any], drive: pathlib.Path) -> bool:
    promotion = row.get("child_ref_promotion")
    if not isinstance(promotion, dict) or promotion.get("status") == "complete":
        return False
    return any(isinstance(ref, dict) and str(ref.get("path") or "")
               and pathlib.Path(str(ref["path"])).resolve(strict=False).is_relative_to(drive)
               for ref in promotion.get("pending_refs") or [])


def _projection_verified(canonical: pathlib.Path, task_id: str, projection: Any) -> bool:
    """Whether a recorded input projection still resolves in the canonical store to files that
    verify against their captured identity (its key in the row proves nothing)."""
    from ouroboros.artifacts import resolve_attachment_manifest, stream_artifact_file

    try:
        for item in resolve_attachment_manifest(canonical, task_id, projection):
            if item.get("status") != "rejected":
                stream_artifact_file(pathlib.Path(item["abs_path"]), expected=item)
    except (OSError, ValueError, TypeError, KeyError):
        return False
    return True


def _carry_row_inputs(canonical: pathlib.Path, source: pathlib.Path, task_id: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    """One row's attachments (inline, or its >25-row manifest source) into the canonical store
    through the attachment owners, as ONE closure from one store: the canonical store when it
    resolves and verifies whole, else SOURCE (a canonical manifest with a member missing is no
    closure). Each member verifies against the captured identity; a canonical file with other
    bytes is never replaced and fails the store. Returns the canonical, resolvable projection;
    a closure neither store supplies whole raises."""
    from ouroboros.artifacts import (
        attachment_manifest_projection,
        materialize_inherited_attachment_manifest,
        resolve_attachment_manifest,
    )

    failure: Optional[Exception] = None
    for owner in (canonical, source):
        fence_publication()  # a closed generation tries no further store
        try:
            copied, error = materialize_inherited_attachment_manifest(
                resolve_attachment_manifest(owner, task_id, entry), canonical, task_id)
            if error:
                raise OSError(error)
            return attachment_manifest_projection(canonical, task_id, copied)
        except (OSError, ValueError, TypeError) as exc:
            failure = exc
    raise OSError(f"no store supplies the whole input closure of {task_id}: {failure}") from failure


def _mail_closure(canonical: pathlib.Path, source: pathlib.Path, task_id: str, current: Dict[str, Any],
                  rows: List[str]) -> Dict[str, Any]:
    """The verified canonical input projection of every input-bearing row among ROWS and the
    durable unread rows CURRENT holds, keyed by exact row: a held projection is re-verified,
    an absent or failing one is carried now (``_carry_row_inputs``). Raises ``_Retained``."""
    held = merge_unread_mail(current.get("unread_mailbox")) or {}
    inputs: Dict[str, Any] = {}
    try:
        for line in [*(held.get("rows") or []), *rows]:
            key = _row_key(line)
            if key in inputs or not _attachment_bearing(entry := json.loads(line)):
                continue
            projection = (held.get("inputs") or {}).get(key)
            if not (isinstance(projection, dict) and _projection_verified(canonical, task_id, projection)):
                projection = _carry_row_inputs(canonical, source, task_id, entry)
            inputs[key] = projection
    except (OSError, ValueError, TypeError) as exc:
        raise _Retained("inputs_uncustodied") from exc
    return inputs


def _input_closure(canonical: pathlib.Path, drive: pathlib.Path, task_id: str, current: Dict[str, Any],
                   rows: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Under the custody lock: the mail closure (``_mail_closure``) plus the task contract's own
    closure and the drive's acknowledged inputs through the existing attachment owners.
    Returns ``(row fields to write, unread_mailbox inputs)``; a copy that fails retains."""
    from ouroboros.artifacts import promote_task_attachment_refs

    inputs = _mail_closure(canonical, drive, task_id, current, rows)
    state = {"schema_version": 1, "status": "complete", "promoted_ref_count": 0, "promoted_source_handle_count": 0,
             "pending_refs": [], "unavailable_refs": []}
    snapshot = copy.deepcopy(current)
    promote_task_attachment_refs(canonical, drive, task_id, snapshot, state)
    if state["pending_refs"]:
        raise _Retained("inputs_uncustodied")
    return {key: snapshot[key] for key in _INPUT_FIELDS if snapshot.get(key) != current.get(key)}, inputs


def _prepare_occupant(canonical: pathlib.Path, drive: pathlib.Path, task_id: str, *, live: LiveProbe,
                      staging: pathlib.Path, admission_rollback: bool, stop=None) -> Dict[str, Any]:
    """Everything outside the locks: liveness, durable settledness, copy-back, pending refs and
    private staging; nothing served is written. Raises ``_Retained``."""
    from ouroboros.owner_mailbox import settled_mailbox_cleanup_allowed
    from ouroboros.store import StoreError
    from ouroboros.task_results import cancellation_blocks_child_result

    alive = live(task_id)
    if alive is not False:
        raise _Retained("liveness_unknown" if alive is None else "task_live")
    try:
        current = load_task_result(canonical, task_id, strict=True) or {}
        child = load_task_result(drive, task_id, strict=True) or {}
    except (OSError, ValueError, StoreError) as exc:
        raise _Retained("task_result_unreadable") from exc
    status = str(current.get("status") or "")
    if admission_rollback and not child and not current.get("started_at") \
            and status not in {"running", "interrupted"}:
        return {"task_id": task_id, "current": current, "plan": [], "rollback": True, "live": live,
                "basis": attempt_basis(current), "revision": custody_revision(current)}
    if status not in _SETTLED:
        raise _Retained("canonical_not_settled")  # only the DURABLE row settles; a projection is not custody
    if not settled_mailbox_cleanup_allowed(current):
        raise _Retained("post_work_open")
    cancelled = cancellation_blocks_child_result(current)
    if str(child.get("status") or "") in _SETTLED and not cancelled and not _copied_back(drive, current):
        raise _Retained("child_result_unadopted")
    if _pending_refs_under(current, drive):
        raise _Retained("child_refs_pending")  # retry_pending_child_ref_promotions owns the retry
    from ouroboros.history_retention import call_inventory_custodied
    if not call_inventory_custodied(canonical, drive, task_id, current, stop=stop):
        raise _Retained("call_inventory_pending")
    if not unread_mail_rows(drive, task_id)[1]:
        raise _Retained("unread_mailbox_unreadable")  # a torn mailbox stages nothing (the closure is Phase A's)
    plan = _child_store_plan(canonical, drive, task_id, current, child, staging / task_id)
    return {"task_id": task_id, "current": current, "plan": plan, "basis": attempt_basis(current), "live": live,
            "revision": custody_revision(current),
            "call_inventory_mtime_ns": (current.get("child_ref_promotion") or {}).get("call_inventory_mtime_ns")}


def _receipts_identity(drive: pathlib.Path, task_id: str) -> Optional[tuple]:
    from ouroboros.outcome_receipt_store import verification_receipts_path

    return _identity(verification_receipts_path(drive, task_id))


def _secure_occupant(canonical: pathlib.Path, drive: pathlib.Path, prepared: Dict[str, Any],
                     closed: Callable[[], bool]) -> int:
    """Phase A, under the occupant's custody lock: re-check the row, union receipts, carry the
    input closure, publish, write once from the row-locked CURRENT and verify the re-read
    row; ``closed()`` is re-asked before each of those effects. Records the revision Phase B
    must still see. Returns files published."""
    from ouroboros.outcome_receipt_store import publish_verification_receipt_union

    def fence() -> None:
        if closed():
            raise _Retained("generation_closed")  # observed under the lock: no further effect

    task_id = prepared["task_id"]
    fence()
    current = load_task_result(canonical, task_id, strict=True) or {}
    if attempt_basis(current) != prepared["basis"] or custody_revision(current) != prepared["revision"]:
        raise _Retained("canonical_changed")
    if prepared.get("rollback"):
        prepared["revision_after"] = prepared["revision"]
        return 0
    rows, complete = unread_mail_rows(drive, task_id)
    if not complete:
        raise _Retained("unread_mailbox_unreadable")  # a torn mailbox stops before any file moves
    owner_capture = capture_owner_mail(canonical, task_id)
    _owner_mail_fields(current, owner_capture)  # refuse incomplete reads before publication
    prepared["receipts"] = _receipts_identity(drive, task_id)
    if prepared["receipts"] is not None and not publish_verification_receipt_union(canonical, task_id, drive):
        raise _Retained("verification_receipts_uncustodied")
    fence()
    prepared["input_fields"], prepared["inputs"] = _input_closure(canonical, drive, task_id, current, rows)
    fence()
    source = drive / _ARTIFACTS_DIR / task_id
    published = _publish(canonical, task_id, prepared["plan"])

    def fields_for(row: Dict[str, Any]) -> Dict[str, Any]:
        return {**_custody_fields(row, source, published), **_mail_fields(row, rows, prepared["inputs"]),
                **_owner_mail_fields(row, owner_capture),
                **prepared["input_fields"]}

    _write_custody_fields(canonical, task_id, current, fields_for, basis=prepared["basis"], closed=closed)
    readback = load_task_result(canonical, task_id, strict=True) or {}
    if attempt_basis(readback) != prepared["basis"]:
        raise _Retained("canonical_changed")
    _verify_readback(readback, source, published)
    if _owner_mail_fields(readback, owner_capture) or _mail_fields(readback, rows, prepared["inputs"]):
        raise _Retained("unread_mailbox_uncustodied")
    if any(readback.get(key) != value for key, value in prepared["input_fields"].items()):
        raise _Retained("inputs_uncustodied")
    if _pending_refs_under(readback, drive):
        raise _Retained("child_refs_pending")
    prepared["revision_after"] = custody_revision(readback)
    return sum(1 for item in prepared["plan"] if item.get("staged") is not None)


def _recheck_occupant(canonical: pathlib.Path, drive: pathlib.Path, prepared: Dict[str, Any]) -> None:
    """Phase B, under the interlock and fresh custody and mail locks: the row, the mailbox
    and the receipts are exactly what Phase A left (bounded reads, no copy or hash)."""
    task_id = prepared["task_id"]
    current = load_task_result(canonical, task_id, strict=True) or {}
    if attempt_basis(current) != prepared["basis"] or custody_revision(current) != prepared["revision_after"]:
        raise _Retained("canonical_changed")
    rows, complete = unread_mail_rows(drive, task_id)
    if not complete:
        raise _Retained("unread_mailbox_unreadable")
    if prepared.get("rollback"):
        if rows or current.get("started_at") or load_task_result(drive, task_id, strict=True):
            raise _Retained("admission_rollback_evidence")
        return
    if (_owner_mail_fields(current, capture_owner_mail(canonical, task_id))
            or _mail_fields(current, rows, prepared["inputs"])):
        raise _Retained("unread_mailbox_uncustodied")  # late mail: the next pass unions and carries it
    if _receipts_identity(drive, task_id) != prepared["receipts"]:
        raise _Retained("verification_receipts_uncustodied")
    if _pending_refs_under(current, drive):
        raise _Retained("child_refs_pending")
    try:
        call_revision = (drive / "observability" / "calls" / task_id).stat().st_mtime_ns
    except FileNotFoundError:
        call_revision = None
    if call_revision != prepared["call_inventory_mtime_ns"]:
        raise _Retained("call_inventory_changed")  # The next pass re-arms retention outside the queue lock.


def settle_child_drive(canonical_root: Any, task_id: str, drive: Any, *, live: Optional[LiveProbe],
                       guard: Optional[Guard] = None, stop: Optional[Callable[[], bool]] = None,
                       admission_rollback: bool = False, report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The one decision to delete one of TASK's own execution drives; see the module docstring.

    ``live(task_id)`` is the caller's supervisor probe (True live, False proven not live,
    None unknown); without one nothing is deleted. ``guard()`` is the supervisor's
    ownership interlock (its queue lock, yielding False once the generation closed): the
    final census, probe and move happen inside it. ``stop()`` closes a maintenance
    generation between phases. ``admission_rollback`` additionally lets a never-started
    task's drive go without a settled row. Returns ``{"status": "removed" | "retained",
    "reason", "published"}`` and, given a prune ``report``, records a kept drive.
    """
    canonical = pathlib.Path(canonical_root).resolve(strict=False)
    tid = validate_task_id(task_id)
    drive = pathlib.Path(drive).resolve(strict=False)
    base = drive.parent if drive.parent.name == tid and drive.name == "data" else drive
    outcome: Dict[str, Any] = {"status": "retained", "reason": "", "published": 0}
    staging = canonical / _STAGING_DIR / f"{tid}-{uuid.uuid4().hex[:12]}"
    closed = stop or (lambda: False)
    try:
        if drive not in own_child_drives(canonical, tid):
            raise _Retained("not_own_drive")
        if live is None:
            raise _Retained("liveness_unknown")
        occupants = _occupants(drive, tid)
        prepared = [_prepare_occupant(canonical, drive, occupant, live=live, staging=staging,
                                      admission_rollback=admission_rollback and occupant == tid, stop=closed)
                    for occupant in occupants]
        if closed():
            raise _Retained("generation_closed")
        if any(live(item["task_id"]) is not False for item in prepared):
            raise _Retained("task_live")  # ownership back after preparation; asked BEFORE any custody lock
        with contextlib.ExitStack() as stack:  # Phase A: publication, per occupant, every write fenced
            stack.enter_context(publication_fence(closed))
            for item in prepared:
                if not stack.enter_context(task_custody_lock(canonical, item["task_id"])):
                    raise _Retained("custody_busy")
            for item in prepared:
                outcome["published"] += _secure_occupant(canonical, drive, item, closed)
        if closed():
            raise _Retained("generation_closed")
        with (guard() if guard is not None else contextlib.nullcontext(True)) as allowed:  # Phase B: the move
            if not allowed:
                raise _Retained("generation_closed")
            if _occupants(drive, tid) != occupants:
                raise _Retained("drive_occupant_changed")
            if any(live(item["task_id"]) is not False for item in prepared):
                raise _Retained("task_live")
            with contextlib.ExitStack() as stack:
                for item in prepared:
                    if not stack.enter_context(task_custody_lock(canonical, item["task_id"],
                                                                 timeout_sec=_INTERLOCK_LOCK_TIMEOUT_SEC)):
                        raise _Retained("custody_busy")
                    if not stack.enter_context(task_mail_lock(canonical, item["task_id"],
                                                              timeout_sec=_INTERLOCK_LOCK_TIMEOUT_SEC)):
                        raise _Retained("mailbox_busy")
                if closed():
                    raise _Retained("generation_closed")  # closed while the locks were awaited: no move
                for item in prepared:
                    _recheck_occupant(canonical, drive, item)
                trash = canonical / _TRASH_DIR / f"{tid}-{uuid.uuid4().hex[:12]}"
                trash.parent.mkdir(parents=True, exist_ok=True)
                os.replace(base, trash)
        outcome.update(status="removed")
        shutil.rmtree(trash, ignore_errors=True)
    except _Retained as retained:
        # A fenced write surfaces as its caller's own failure reason: a closed generation names the close.
        outcome["reason"] = "generation_closed" if closed() else str(retained)
    except (OSError, ValueError, TimeoutError) as exc:
        outcome["reason"] = "generation_closed" if closed() else f"custody_error: {type(exc).__name__}"
        if not closed():
            log.warning("Child drive %s kept: custody could not be proven", drive, exc_info=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    if report is not None:
        if outcome["published"]:
            report.setdefault("custodied", []).append({"task_id": tid, "published": outcome["published"]})
        if outcome["status"] != "removed":
            kept = {"task_id": tid, "reason": outcome["reason"]}
            report.setdefault("skipped", []).append(kept)
            report.setdefault("custody_pending", []).append(kept)
    return outcome


def sweep_custody_leftovers(canonical_root: Any, *, min_age_sec: float = 3600.0) -> None:
    """Remove trash and staging a crashed settlement left behind (older than ``min_age_sec``,
    so a running settlement keeps its own); both are unserved and unreferenced by
    construction: trash is a fully custodied drive, staging a private copy."""
    import time

    for directory in (_TRASH_DIR, _STAGING_DIR):
        root = pathlib.Path(canonical_root) / directory
        if not root.is_dir() or root.is_symlink():
            continue
        for entry in root.iterdir():
            try:
                if time.time() - entry.lstat().st_mtime < min_age_sec:
                    continue
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    entry.unlink(missing_ok=True)
            except OSError:
                continue
