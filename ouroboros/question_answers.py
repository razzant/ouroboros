"""Address accepted answers to their living asker, an admitted successor, or a fresh head.

A living asker holds its answers while paused, held or queued and reads them
when it continues. Mailbox write is owed delivery, never a read receipt. A
final drain/terminal race is repaired from the durable recipient and ACK
identity; no room's sole unrelated reader and no claimed-but-unadmitted
Continue may take the answer. Once the owner's ingress accepted it, every
retry converges on that message. A file answer's write with no receipt is
repeated only at its own reader, letter id and store; once that reader ended,
only a read settles it. Within the serving process, one short-scoped
per-question in-flight owner guards ordinary delivery so concurrent callers do
not double-stage or double-deliver; across restarts the canonical disk record
remains authoritative. An owner's closure of such a card is pinned by its winning close to the asker's one admitted
immediate Continue reading then (``closure_route``): that task and store only, never a second address.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

_INFLIGHT_LOCK = threading.Lock()
_INFLIGHT_ORDINARY: set[tuple[str, str, str]] = set()

# The host's own act, said only by this letter: the shared frame also builds the room's history row and a relayed
# quote's frame, where it would be false. The letter may reach the asker's admitted Continue, so it names its route.
MAILBOX_ROUTE_FACT = ("Host fact: this mailbox delivery is addressed to this work; this route does not broadcast "
                      "the answer to unrelated work.")


def answer_message_id(asker: str, quiz_id: str) -> str:
    """The mailbox identity ``deliver_answer`` gives a card's live answer, whichever task reads it."""
    return f"quiz_answer:{asker}:{quiz_id}"


def admitted_successor(root, predecessor: str) -> str:
    """A durable admission, independently verified on both ends of Continue."""
    from ouroboros.owner_continue import binding_sha
    from ouroboros.task_results import load_task_result
    try:
        result = load_task_result(root, predecessor, strict=True) or {}
    except (ValueError, OSError):
        return ""
    claim = result.get("continued_by") or {}
    binding = claim.get("binding") or {}
    successor = str(claim.get("successor_task_id") or "")
    if (claim.get("state") != "admitted" or not successor or binding.get("successor_task_id") != successor
            or binding.get("predecessor_task_id") != predecessor
            or claim.get("binding_sha256") != binding_sha(binding)):
        return ""
    try:
        target = load_task_result(root, successor, strict=True) or {}
    except (ValueError, OSError):
        return ""
    admission = target.get("continuation_admission") or {}
    if admission.get("binding") != binding or admission.get("binding_sha256") != binding_sha(binding):
        return ""
    return successor


def reading_recipient(root, asker: str, live_reader) -> tuple[str, dict | None]:
    current, seen = asker, set()
    while current and current not in seen:
        seen.add(current)
        task, _reason = live_reader(current)
        if task is not None:
            return current, task
        current = admitted_successor(root, current)
    return "", None


def closure_route(root, asker: str, live_reader) -> dict:
    """What an owner's winning close pins as the reader of its closure letter: the asker's ONE admitted immediate
    Continue (never the asker, never a later one: ``reading_recipient`` walks on, a closure does not) when
    ``live_reader`` finds it reading now, and the mail store it reads from; ``{}`` for anything else."""
    from supervisor import queue as q

    successor = admitted_successor(root, asker)
    task, _reason = live_reader(successor) if successor else (None, "")
    if task is None:
        return {}
    return {"recipient_task_id": successor, "recipient_drive": str(q._task_drive_for_task(task, successor))}


def _update_delivery(root, asker, quiz_id, changes):
    from ouroboros.owner_quiz import _mutate_projection, _KEEP, question_delivery_settled
    def update(quizzes):
        block = quizzes.get(quiz_id)
        if not block or question_delivery_settled(block):
            return _KEEP
        block["delivery"] = {**(block.get("delivery") or {}), **changes}
        return True
    return (_mutate_projection(root, asker, update).get(quiz_id) or {})


def _staged_answer_files(root, drive, recipient: str, block: dict) -> tuple[str, list]:
    """The answer's recorded files staged into its reader's own store now, through the owner-mail staging path:
    the note naming where that work reads each one (a file changed or gone since it was recorded stays refused with
    its reason) and the manifest the letter records. A file never received is named by the shared frame instead."""
    if not block.get("attachments"):
        return "", []
    from ouroboros.chat_uploads import staging_specs
    from ouroboros.server_owner_routing import _stage_mailbox_attachments

    note, manifest, _ = _stage_mailbox_attachments(
        None, Path(drive), recipient, {"chat_attachment_uploads": staging_specs(block["attachments"], root)})
    return note, manifest


def _read_settled(root, asker: str, quiz_id: str, delivery: dict) -> tuple[bool, str]:
    """A read of the answer's letter settles it: its receipt now, its discharge once its history row is written."""
    from ouroboros.owner_quiz import mark_discharged
    from ouroboros.utils import utc_now_iso

    _update_delivery(root, asker, quiz_id, {"read_at": utc_now_iso()})
    if delivery.get("history_at"):
        mark_discharged(root, asker, quiz_id)
        return False, ""
    return False, "quiz_history_write_pending"


def _captured_msg_ids(custody) -> set[str] | None:
    """Exact ``msg_id``s of a terminal mailbox capture dict, or None when malformed."""
    if not isinstance(custody, dict):
        return None
    rows = custody.get("rows")
    if not isinstance(rows, list):
        return None
    ids: set[str] = set()
    for line in rows:
        if not isinstance(line, str):
            return None
        try:
            entry = json.loads(line)
        except ValueError:
            return None
        if not isinstance(entry, dict):
            return None
        mid = entry.get("msg_id")
        if isinstance(mid, str) and mid:
            ids.add(mid)
    return ids


def _settled_unread_stamp(recipient: str, msg_id: str, drive) -> dict[str, str]:
    """Exact ordinary-letter non-consumption proof bound to one settled ``(reader, letter, store)``."""
    return {
        "recipient_task_id": str(recipient),
        "msg_id": str(msg_id),
        "recipient_drive": str(Path(drive).resolve(strict=False)),
    }


def mail_store_key(root, task_id: str, drive) -> str:
    """The durable identity of one of ``task_id``'s mailbox stores, as its owner capture records it
    (``task_custody.capture_owner_mail``) for the non-write proof below: ``canonical`` for the owner root ``root``
    anchors on, ``headless`` or ``task_drive`` for that task's own child layout there, compared lexically so it
    outlives the drive's retirement; '' for any other place, which no capture covers. Never a path."""
    from ouroboros.task_custody import canonical_owner_root, child_layout_owner

    try:
        parts = Path(drive).resolve(strict=False).relative_to(canonical_owner_root(root)).parts
    except (OSError, ValueError, RuntimeError):
        return ""
    if not parts:
        return "canonical"
    return ("headless" if parts[0] == "state" else "task_drive") if child_layout_owner(parts) == str(task_id) else ""


def mail_stores(values) -> dict:
    """``{"stores": [...]}``: every store the owner captures among ``values`` read whole, united (custody only grows,
    so a retired drive keeps the coverage its capture earned); {} when none records stores, as a capture from before
    they were recorded."""
    stores = None
    for value in values:
        if isinstance(value, dict) and isinstance(value.get("stores"), list):
            stores = {*(stores or ()), *(key for key in value["stores"] if isinstance(key, str) and key)}
    return {} if stores is None else {"stores": sorted(stores)}


def settle_quiz_mailbox_receipts(canonical_root, task_id: str, mailbox_root, *, lock: bool = False) -> None:
    """Persist exact ordinary quiz-answer read/unread receipts before ``mailbox_root``'s mailbox
    and ACK ledger are removed for settled ``task_id``.

    With ``lock=False``, the caller holds ``task_mail_lock(mailbox_root, task_id)``. Reads the
    mailbox and ACK ledger completely under that lock:
    - An ordinary ``kind="quiz_answer"`` letter positively present in the ACK ledger persists
      its existing read receipt (``read_at`` / discharge) before unlink.
    - An ordinary ``kind="quiz_answer"`` letter absent from the complete ACK ledger persists
      ``settled_unread`` bound to ``(task_id, msg_id, mailbox_root)`` on the asker's active
      ``owner_quiz`` delivery block (only when that block still names this exact reader, letter
      id, and store, has no ``read_at``, and is neither settled nor a secret outcome), and makes
      that debt due again (``next_retry_at``) for the existing recovery to move it on.
    Any unreadable mailbox/ACK ledger or failed store write raises so the caller retains the files.
    """
    import contextlib

    from ouroboros.owner_mailbox import (
        KIND_QUIZ_ANSWER,
        _mailbox_path,
        acknowledged_task_message_ids,
        mailbox_lines,
    )
    from ouroboros.owner_quiz import (
        _KEEP,
        _mutate_projection,
        mark_discharged,
        question_delivery_settled,
        secret_outcome_block,
    )
    from ouroboros.task_custody import fence_publication, task_mail_lock
    from ouroboros.utils import utc_now_iso

    path = _mailbox_path(Path(mailbox_root), task_id)
    if not path.exists():
        return
    with (task_mail_lock(mailbox_root, task_id) if lock else contextlib.nullcontext(True)) as locked:
        if not locked:
            raise TimeoutError("mailbox_busy")
        fence_publication()
        if not path.exists():
            return
        status: dict[str, bool] = {}
        acknowledged = acknowledged_task_message_ids(Path(mailbox_root), task_id, _read_status=status)
        content = path.read_text(encoding="utf-8")
        if not status.get("complete") or (content and not content.endswith("\n")):
            raise ValueError("quiz_mailbox_unreadable")
        seen: set[str] = set()
        store_norm = str(Path(mailbox_root).resolve(strict=False))
        for line in mailbox_lines(content):
            entry = json.loads(line)
            if not isinstance(entry, dict):
                raise ValueError("quiz_mailbox_row_malformed")
            if entry.get("kind") != KIND_QUIZ_ANSWER or secret_outcome_block(entry) is not None:
                continue
            provenance = entry.get("quiz_delivery")
            if not isinstance(provenance, dict):
                continue
            asker = str(provenance.get("task_id") or "")
            quiz_id = str(provenance.get("quiz_id") or "")
            msg_id = str(entry.get("msg_id") or "")
            if not (asker and quiz_id and msg_id) or msg_id in seen:
                continue
            seen.add(msg_id)
            canonical = Path(provenance.get("canonical_root") or canonical_root)
            is_read = msg_id in acknowledged
            discharge = [False]

            def update(quizzes, *, _qid=quiz_id, _mid=msg_id, _read=is_read, _discharge=discharge):
                fence_publication()
                block = quizzes.get(_qid)
                if (not isinstance(block, dict) or block.get("state") != "answered"
                        or secret_outcome_block(block) is not None or question_delivery_settled(block)):
                    return _KEEP
                delivery = block.get("delivery")
                if not isinstance(delivery, dict) or delivery.get("read_at"):
                    return _KEEP
                target_recipient = str(delivery.get("recipient_task_id") or "")
                target_msg = str(delivery.get("msg_id") or "")
                target_drive = str(Path(delivery.get("recipient_drive") or canonical).resolve(strict=False))
                legacy = (not target_recipient and delivery.get("mailbox_at") and not delivery.get("route")
                          and task_id == asker and _mid == f"quiz_answer:{_qid}")
                if not (legacy or (target_recipient == task_id and target_msg == _mid and target_drive == store_norm)):
                    return _KEEP
                if _read:
                    updated = {**delivery, "read_at": utc_now_iso()}
                    if legacy:
                        updated.update(recipient_task_id=asker, msg_id=_mid)
                    updated.pop("settled_unread", None)
                    block["delivery"] = updated
                    if updated.get("history_at"):
                        _discharge[0] = True
                    return True
                stamp = _settled_unread_stamp(task_id, _mid, mailbox_root)
                if delivery.get("settled_unread") == stamp and (not legacy or delivery.get("recipient_task_id")):
                    return _KEEP
                # Proven unread: the owner's existing retry clock (``owner_quiz.quiz_delivery_due``) may move it on
                # its next recovery pass instead of waiting out a backoff its reader's life ran up.
                updated = {**delivery, "settled_unread": stamp, "next_retry_at": ""}
                if legacy:
                    updated.update(recipient_task_id=asker, msg_id=_mid, recipient_drive=store_norm)
                block["delivery"] = updated
                return True

            fence_publication()
            _mutate_projection(canonical, asker, update)
            if discharge[0]:
                fence_publication()
                mark_discharged(canonical, asker, quiz_id)


def _settled_non_write(root, task_id: str, msg_id: str, recipient_drive) -> bool:
    """True only when settled ``task_id``'s complete, consistent terminal captures prove ``msg_id`` was never written
    to ``recipient_drive`` (read or unread): a store its owner capture recorded reading whole (``mail_store_key``),
    never one found on disk now, so a retired own drive keeps its coverage and a store no capture read gains none.
    A capture that records no ``stores`` names no store it read: the old same-attempt replica copy could replace
    this field with the child drive's own capture before the copy-side guard landed, and early canonical terminal
    writes left the field off alike, so no placement survives to tell them apart: its silence answers for none and
    the letter stays owed, never re-delivered on that guess."""
    from ouroboros.task_results import load_task_result
    from ouroboros.task_status import SETTLED_STATUSES

    if not task_id or not msg_id:
        return False
    try:
        result = load_task_result(root, task_id, strict=True) or {}
    except Exception:
        return False
    if not isinstance(result, dict) or str(result.get("status") or "") not in SETTLED_STATUSES:
        return False
    owner = result.get("owner_mailbox")
    if not (isinstance(owner, dict) and owner.get("read_complete") is True):
        return False
    owner_ids = _captured_msg_ids(owner)
    if owner_ids is None or msg_id in owner_ids:
        return False
    unread = result.get("unread_mailbox")
    if unread is not None:
        if not (isinstance(unread, dict) and unread.get("read_complete") is True):
            return False
        unread_ids = _captured_msg_ids(unread)
        if unread_ids is None or msg_id in unread_ids:
            return False
    key, stores = mail_store_key(root, task_id, recipient_drive), owner.get("stores")
    return bool(key) and isinstance(stores, list) and key in stores


def custody_proves_read(root, task_id: str, row: str, drives) -> bool:
    """Whether ``task_id``'s own custody proves it read the mailbox letter whose exact line is ``row``: no live
    mailbox among ``drives`` holds that line any more (each read whole), and its record, read after them, keeps it
    as owner mail and never as unread, each capture complete. Every cleanup captures a store's unread lines whole
    before it unlinks them, so only an acknowledged letter leaves that way; a live copy or a partial read proves
    nothing (``owner_continue`` asks the live acknowledgement first)."""
    from ouroboros.task_custody import owner_mail_rows
    from ouroboros.task_results import load_task_result

    def lines(capture):  # a complete capture's exact lines, else None
        held = capture.get("rows") if isinstance(capture, dict) and capture.get("read_complete") is True else None
        return held if isinstance(held, list) else None

    try:
        if any(not whole or row in found for found, whole in (owner_mail_rows(drive, task_id) for drive in drives)):
            return False  # still live, or unreadable: only its acknowledgement could speak
        record = load_task_result(root, task_id, strict=True) or {}
    except Exception:
        return False
    owned, unread = lines(record.get("owner_mailbox")), record.get("unread_mailbox")
    kept = None if unread is None else lines(unread)  # absent: no capture ever held an unread row
    return owned is not None and row in owned and (unread is None or kept is not None and row not in kept)


def _settled_terminal_uncertain(root, task_id: str) -> bool:
    """Whether ``task_id``'s stored result is unreadable or actually settled without proving
    non-write for this letter (distinct from a schema-less/no-terminal legacy fixture)."""
    from ouroboros.task_results import load_task_result
    from ouroboros.task_status import SETTLED_STATUSES

    try:
        result = load_task_result(root, task_id, strict=True) or {}
    except Exception:
        return True
    return isinstance(result, dict) and str(result.get("status") or "") in SETTLED_STATUSES


def _capture_proves_unread(root, task_id: str, msg_id: str, mailbox_at: str) -> bool:
    """True only when settled ``task_id``'s own retained unread capture is complete, names ``msg_id``, and is not
    older than the accepted write: the letter sat in that reader's mailbox, unacknowledged, in the terminal state
    itself, so a later unlink leaves it consumable by nobody and recovery may move the answer on. The unread
    capture is a grow-only union, so a positive row is its own producer's placement evidence; its absence, an
    empty, partial, or foreign capture, or one predating the accepted write, proves nothing. Confirmed reads
    (``read_at``, a live ack, a discharge) settle through their own receipts before this is ever consulted."""
    from ouroboros.task_results import load_task_result
    from ouroboros.task_status import SETTLED_STATUSES

    if not task_id or not msg_id or not mailbox_at:
        return False
    try:
        result = load_task_result(root, task_id, strict=True) or {}
    except Exception:
        return False
    if not isinstance(result, dict) or str(result.get("status") or "") not in SETTLED_STATUSES:
        return False
    unread = result.get("unread_mailbox")
    if not (isinstance(unread, dict) and unread.get("read_complete") is True):
        return False
    captured_at = unread.get("captured_at")
    if not (isinstance(captured_at, str) and captured_at >= str(mailbox_at)):
        return False
    ids = _captured_msg_ids(unread)
    return ids is not None and msg_id in ids


def _earlier_effect(
    root,
    asker: str,
    quiz_id: str,
    delivery: dict,
    live_reader,
    files: bool = False,
    *,
    ordinary: bool = True,
) -> tuple:
    """The letter an earlier attempt saved for this answer, resolved from its recipient's own durable facts:
    ``(outcome, recipient, delivery, live_pin)``. ``outcome`` ends this attempt when that letter was read (only a
    read settles it), still waits with its living recipient, or went to a recipient that ended without a provable
    non-read or non-write (an unreadable ledger, or missing/incomplete proof: it stays owed, no other reader gets
    it); ``None`` leaves the answer for this attempt to route. ``live_pin`` is True only while an unconfirmed file
    write's earlier recipient is still alive and its write may be repeated only at that address. Protected-question
    outcomes (``ordinary=False``) keep their exact pre-change recovery semantics."""
    from ouroboros.owner_mailbox import mail_read_state, mailbox_holds
    from ouroboros.utils import utc_now_iso

    previous = str(delivery.get("recipient_task_id") or "")
    if not previous and delivery.get("mailbox_at") and not delivery.get("route"):
        previous = asker
        delivery = {**delivery, "recipient_task_id": asker, "msg_id": f"quiz_answer:{quiz_id}"}
        _update_delivery(root, asker, quiz_id, delivery)
    if not previous or not delivery.get("msg_id"):
        return None, previous, delivery, False
    msg_id = str(delivery["msg_id"])
    recipient_drive = delivery.get("recipient_drive") or root
    roots = {Path(root), Path(recipient_drive)}
    read = [True] if delivery.get("read_at") else [mail_read_state(r, previous, msg_id) for r in roots]
    if True in read:
        return _read_settled(root, asker, quiz_id, delivery), previous, delivery, False
    if not delivery.get("mailbox_at") and any(mailbox_holds(r, previous, msg_id) for r in roots):
        # The append can land before its receipt write fails: the copy still in the mailbox is the receipt.
        repaired = _update_delivery(root, asker, quiz_id, {"mailbox_at": utc_now_iso()}).get("delivery")
        delivery = repaired or {**delivery, "mailbox_at": utc_now_iso()}
    if not ordinary:
        # A protected outcome keeps its earlier route: held while its reader lives, else on to that reader's admitted
        # Continue or the caller's history fallback. Its letter names the outcome's facts, never a value.
        current, reason = live_reader(previous) if delivery.get("mailbox_at") else (None, "")
        held = current is not None or reason == "task_not_reading"
        return ((False, "awaiting_recipient_read") if held else None), previous, delivery, False
    settled_unread = delivery.get("settled_unread") == _settled_unread_stamp(previous, msg_id, recipient_drive)
    if delivery.get("mailbox_at") or files or None in read:
        # An unread copy stays owed to its recipient while that work exists:
        # Pause, sleep or a hold moves it nowhere; only its end does.
        current, reason = live_reader(previous)
        if current is not None or reason == "task_not_reading":
            if delivery.get("mailbox_at"):
                return (False, "awaiting_recipient_read"), previous, delivery, False
            return None, previous, delivery, bool(files)
        # Its end is observed before this very letter is read back. Check live mailbox and acks first,
        # then exact ordinary settlement proof (bound to reader, letter id and store before unlink).
        held_before = any(mailbox_holds(r, previous, msg_id) for r in roots)
        read = [mail_read_state(r, previous, msg_id) for r in roots]
        if True in read:
            return _read_settled(root, asker, quiz_id, delivery), previous, delivery, False
        held_after = any(mailbox_holds(r, previous, msg_id) for r in roots)
        if delivery.get("mailbox_at"):
            if held_before and None not in read and held_after:
                return None, previous, delivery, False
            if settled_unread and None not in read and not held_before and not held_after:
                return None, previous, delivery, False
            if None not in read and not held_before and not held_after and _capture_proves_unread(
                    root, previous, msg_id, str(delivery.get("mailbox_at") or "")):
                # A legacy cleanup left this accepted letter without a stamp, yet the reader's own complete
                # terminal unread state names it unread: the unlink consumed nothing, so the debt moves on to its
                # Continue or the late path instead of staying owed forever.
                return None, previous, delivery, False
            return (False, "mailbox_write_failed"), previous, delivery, False
        if files:
            if None not in read and not held_before and not held_after and (
                settled_unread or _settled_non_write(root, previous, msg_id, recipient_drive)
            ):
                return None, previous, delivery, False
            return (False, "mailbox_write_failed"), previous, delivery, False
        return (False, "mailbox_write_failed"), previous, delivery, False
    # Unconfirmed non-file ordinary write with readable live acks and no live copy:
    # honor settled exact receipts and refuse uncertain terminal captures.
    current, reason = live_reader(previous)
    if current is None and reason != "task_not_reading":
        if (not settled_unread
                and not _settled_non_write(root, previous, msg_id, recipient_drive)
                and _settled_terminal_uncertain(root, previous)):
            return (False, "mailbox_write_failed"), previous, delivery, False
    return None, previous, delivery, False


def _repeat_address(root, files: bool, previous: str, delivery: dict):
    """``(reader, letter id, store)``: the only place a file answer's unconfirmed write may be repeated while that
    reader lives. ``None`` when nothing binds it: no earlier attempt, a receipt, or no recorded attachments."""
    if not (files and previous and delivery.get("msg_id")) or delivery.get("mailbox_at"):
        return None
    return previous, str(delivery["msg_id"]), Path(delivery.get("recipient_drive") or root)


def _earlier_unsettled(root, asker: str, quiz_id: str, delivery: dict, live_reader, files: bool, *, ordinary: bool = True):
    """``_earlier_effect`` read afresh, outside the admission lock, where an attempt found its address changed or gone:
    its outcome, or owed while a file answer's unconfirmed write still has a living reader (nothing moves blindly);
    ``None`` leaves the answer to route."""
    outcome, earlier, earlier_delivery, live_pin = _earlier_effect(
        root, asker, quiz_id, delivery, live_reader, files, ordinary=ordinary)
    if outcome is None and live_pin and _repeat_address(root, files, earlier, earlier_delivery) is not None:
        return False, "mailbox_write_failed"
    return outcome


def _accepted_late_row(root, asker: str, quiz_id: str, block: dict):
    """The accepted owner-ingress row for this answer's late path, or None (a store that cannot say raises)."""
    from supervisor import message_bus

    chat_id = block.get("publication_chat_id")
    if type(chat_id) is not int:
        return None
    return message_bus.accepted_chat_message(root, chat_id, f"quiz_late_answer:{asker}:{quiz_id}")


def _late_accepted(root, asker: str, quiz_id: str, block: dict) -> bool:
    """Whether the owner's ingress already accepted this answer as the owner's message (a store that cannot say raises)."""
    return _accepted_late_row(root, asker, quiz_id, block) is not None


def _delivery_key(root, asker: str, quiz_id: str) -> tuple[str, str, str]:
    """Process-local in-flight ownership key for one ordinary question's delivery."""
    return (str(Path(root).resolve(strict=False)), str(asker), str(quiz_id))


def _strict_quiz_block(root, asker: str, quiz_id: str) -> dict | None:
    """Fresh canonical ``owner_quiz`` block, or None when storage is unreadable or the block is absent."""
    from ouroboros.task_store import read_document

    try:
        document = read_document(root, str(asker))
    except Exception:
        return None
    quizzes = document.get("owner_quiz") if isinstance(document, dict) else None
    block = quizzes.get(str(quiz_id)) if isinstance(quizzes, dict) else None
    return dict(block) if isinstance(block, dict) else None


def _inflight_follower_outcome(root, asker: str, quiz_id: str) -> tuple[bool, str]:
    """Read-only outcome when another call in this process already owns ordinary delivery for this
    question: report settled or positively completed prior effects and actual delivery-unknown state
    without writing, staging, or returning an in-flight marker that gateway :684-686 would persist."""
    from ouroboros.owner_quiz import question_delivery_settled

    current = _strict_quiz_block(root, asker, quiz_id)
    if current is None:
        return False, ""
    delivery = current.get("delivery") if isinstance(current.get("delivery"), dict) else {}
    if question_delivery_settled(current):
        if (delivery.get("kind") == "answer" and delivery.get("state") == "delivered"
                and delivery.get("route") == "fresh_head" and not delivery.get("read_at")):
            return True, ""
        return False, ""
    if delivery.get("read_at"):
        return (False, "") if delivery.get("history_at") else (False, "quiz_history_write_pending")
    try:
        accepted = _accepted_late_row(root, asker, quiz_id, current)
    except Exception:
        accepted = None
    if isinstance(accepted, dict):
        from supervisor.message_ingress import dispatch_completed, dispatch_lost
        try:
            if dispatch_completed(accepted, root):
                return True, ""
            if dispatch_lost(accepted, root):
                return False, "late_answer_delivery_unknown"
        except Exception:
            pass
    if delivery.get("reason_code") == "late_answer_delivery_unknown":
        return False, "late_answer_delivery_unknown"
    return False, ""


def acknowledge_answer(root, recipient: str, entry: dict) -> None:
    from ouroboros.owner_quiz import mark_discharged, quiz_states
    from ouroboros.utils import utc_now_iso
    provenance = entry.get("quiz_delivery") or {}
    asker, quiz_id = provenance.get("task_id"), provenance.get("quiz_id")
    if not asker or not quiz_id:
        return
    canonical = Path(provenance.get("canonical_root") or root)
    block = quiz_states(canonical, asker).get(quiz_id) or {}
    delivery = block.get("delivery") or {}
    if delivery.get("recipient_task_id") == recipient and delivery.get("msg_id") == entry.get("msg_id"):
        _update_delivery(canonical, asker, quiz_id, {"read_at": utc_now_iso()})
        if delivery.get("history_at"):
            mark_discharged(canonical, asker, quiz_id)


def _deliver_answer_unlocked(
    root, asker: str, quiz_id: str, block: dict, live_reader, fallback, *, ordinary: bool | None = None,
) -> tuple[bool, str]:
    """Core delivery transition once per-question ownership is held (or for protected outcomes)."""
    from ouroboros.owner_mailbox import KIND_QUIZ_ANSWER, write_owner_message
    from ouroboros.owner_quiz import (
        mark_discharged, question_delivery_settled, quiz_states, recorded_answer_frame, secret_outcome_block,
    )
    from ouroboros.utils import utc_now_iso
    from supervisor import queue as q
    from supervisor.workers import get_direct_chat_agent

    if ordinary is None:
        block = quiz_states(root, asker).get(quiz_id) or block
        ordinary = secret_outcome_block(block) is None
    if ordinary and question_delivery_settled(block):
        settled_delivery = block.get("delivery") if isinstance(block.get("delivery"), dict) else {}
        if (settled_delivery.get("kind") == "answer" and settled_delivery.get("state") == "delivered"
                and settled_delivery.get("route") == "fresh_head" and not settled_delivery.get("read_at")):
            return True, ""
        return False, ""
    files = bool(block.get("attachments"))  # a file, or the host's fact of one never received: the recorded evidence
    outcome, previous, delivery, live_pin = _earlier_effect(
        root, asker, quiz_id, block.get("delivery") or {}, live_reader, files, ordinary=ordinary)
    if outcome is not None:
        return outcome
    # A file answer whose earlier write is unconfirmed while its reader lives: repeated only at that reader, letter id
    # and store (a words-only, button or secret outcome keeps its earlier route).
    pin = _repeat_address(root, files, previous, delivery) if live_pin else None
    # An answer the owner's ingress already accepted keeps that one effect: its retry converges
    # there, never on a Continue admitted since (a press does the same, card_decision._deliver).
    late_id = f"quiz_late_answer:{asker}:{quiz_id}"
    accepted = _late_accepted(root, asker, quiz_id, block)
    recipient, task = ("", None) if accepted else reading_recipient(root, asker, live_reader)
    attempted, moved = set(), set()
    while task is not None and recipient not in attempted:
        attempted.add(recipient)
        agent = get_direct_chat_agent(recipient)
        if agent is None:
            from ouroboros.conversation_lease import presence_author_agent
            agent = presence_author_agent(root, recipient)
        lock = getattr(agent, "_owner_message_admission_lock", None) or q._queue_lock
        # A Reply's files are copied into this reader's own store BEFORE its admission lock, as owner mail is: file
        # I/O never holds Stop, Pause or another work's mail. The lock re-reads the same reader and store first.
        drive = Path(root) if agent is not None else q._task_drive_for_task(task, recipient)
        held = previous == recipient and delivery.get("mailbox_at")
        elsewhere = pin is not None and (recipient, Path(drive)) != (pin[0], pin[2])  # no copy, no letter there
        note, staged = ("", []) if held or elsewhere else _staged_answer_files(root, drive, recipient, block)
        with lock:
            # Same admission owner as room intake: no final-drain admission after closure.
            current, _reason = live_reader(recipient)
            store = drive if current is None or agent is not None else q._task_drive_for_task(current, recipient)
            if current is not None and store == drive and not elsewhere:
                msg_id = pin[1] if pin else answer_message_id(asker, quiz_id)
                if held:
                    return False, "awaiting_recipient_read"
                # Name the owed effect BEFORE append. A crash retries that same identity.
                _update_delivery(root, asker, quiz_id, {"recipient_task_id": recipient, "msg_id": msg_id,
                    "recipient_drive": str(drive), "state": "owed", "mailbox_at": "",
                    "reason_code": "awaiting_recipient_read",
                    **({"route": ""} if ordinary and delivery.get("route") else {}),
                    **({"settled_unread": None} if ordinary and delivery.get("settled_unread") else {})})
                # A secret outcome is host metadata: no owner words to route. A pressed label is the asker's text.
                letter = recorded_answer_frame(block, asker=asker, reader=recipient)
                letter += "" if not ordinary else "\n" + MAILBOX_ROUTE_FACT
                if not write_owner_message(drive, letter + note, recipient, msg_id=msg_id,
                        kind=KIND_QUIZ_ANSWER, quiz_delivery={"task_id": asker, "quiz_id": quiz_id,
                                                          "canonical_root": str(root), "answer_snapshot": dict(block)},
                        attachment_manifest=staged or None):
                    return False, "mailbox_write_failed"  # the append's outcome is unknown: its copies stay
                _update_delivery(root, asker, quiz_id, {"mailbox_at": utc_now_iso()})
                return False, "awaiting_recipient_read"
        # Not admitted: the reader ended, or lives on in another store, while this attempt copied the files (they stay
        # in that store: another attempt's letter may name the same copies, one name per bytes) or waited for the lock.
        # Another attempt may have delivered meanwhile: its saved recipient and letter are resolved as at the start,
        # from the record read strictly now (unreadable, or the answer gone from it: it stays owed), before this one
        # copies again, follows a Continue or takes the late path.
        from ouroboros.task_store import read_document
        try:
            now = ((read_document(root, asker) or {}).get("owner_quiz") or {}).get(quiz_id)
        except Exception:
            now = None
        if not isinstance(now, dict):
            return False, "mailbox_write_failed"
        if question_delivery_settled(now):
            return False, ""
        outcome = _earlier_unsettled(root, asker, quiz_id, now.get("delivery") or {}, live_reader, files,
                                     ordinary=ordinary)
        if outcome is not None:
            return outcome
        if current is not None:  # the reader lives on in another store (a queued work started meanwhile): copy again
            if recipient in moved:
                return False, "mailbox_write_failed"  # it keeps moving: the answer stays owed for recovery
            moved.add(recipient)
            attempted.discard(recipient)
            task = current
            continue
        if _late_accepted(root, asker, quiz_id, now):
            break  # the owner's ingress accepted it meanwhile: that stays its one effect
        # The selected reader ended before admission. Follow its existing
        # Continue under that recipient's own lock before using a fresh head.
        recipient, task = reading_recipient(root, admitted_successor(root, recipient), live_reader)
    if pin is not None and not accepted and not attempted:
        # Its reader ended after the answer's earlier write and before any reader was reached: resolved afresh.
        outcome = _earlier_unsettled(root, asker, quiz_id, delivery, live_reader, files, ordinary=ordinary)
        if outcome is not None:
            return outcome
    # No living holder. Named ingress carries the typed question-answer fact: owner
    # routing gives it only to the asker's live late-answer head, else a fresh head;
    # it releases no hold.
    late_reason = ("late_answer_delivery_unknown"
                   if ordinary and delivery.get("reason_code") == "late_answer_delivery_unknown"
                   else "asker_not_reading")
    block = _update_delivery(root, asker, quiz_id, {"recipient_task_id": "", "msg_id": late_id,
        "route": "fresh_head", "state": "owed", "reason_code": late_reason,
        **({"settled_unread": None} if ordinary and delivery.get("settled_unread") else {})}) or block
    forwarded, reason = fallback(root, asker, quiz_id, block, source=str(block.get("answer_source") or "web"))
    if forwarded:
        mark_discharged(root, asker, quiz_id)
    return forwarded, reason


def deliver_answer(root, asker: str, quiz_id: str, block: dict, live_reader, fallback) -> tuple[bool, str]:
    """Return (forwarded_to_owner_ingress, reason); a living recipient's mailbox remains owed until ACK."""
    from ouroboros.owner_quiz import quiz_states, secret_outcome_block

    if secret_outcome_block(block) is not None:
        current = quiz_states(root, asker).get(quiz_id) or block
        return _deliver_answer_unlocked(root, asker, quiz_id, current, live_reader, fallback, ordinary=False)
    key = _delivery_key(root, asker, quiz_id)
    with _INFLIGHT_LOCK:
        if key in _INFLIGHT_ORDINARY:
            entered = False
        else:
            _INFLIGHT_ORDINARY.add(key)
            entered = True
    if not entered:
        return _inflight_follower_outcome(root, asker, quiz_id)
    try:
        latest = _strict_quiz_block(root, asker, quiz_id)
        if latest is None:
            return False, ""
        if secret_outcome_block(latest) is not None:
            return _deliver_answer_unlocked(root, asker, quiz_id, latest, live_reader, fallback, ordinary=False)
        if latest.get("state") != "answered":
            return False, ""
        return _deliver_answer_unlocked(root, asker, quiz_id, latest, live_reader, fallback, ordinary=True)
    finally:
        with _INFLIGHT_LOCK:
            _INFLIGHT_ORDINARY.discard(key)
