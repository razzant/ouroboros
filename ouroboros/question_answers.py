"""Address accepted answers to their living asker, an admitted successor, or a fresh head.

A living asker holds its answers while paused, held or queued and reads them
when it continues. Mailbox write is owed delivery, never a read receipt. A
final drain/terminal race is repaired from the durable recipient and ACK
identity; no room's sole unrelated reader and no claimed-but-unadmitted
Continue may take the answer. Once the owner's ingress accepted it, every
retry converges on that message.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

# The host's own act, said only by this letter: the shared frame also builds the room's history row and a relayed
# quote's frame, where it would be false. The letter may reach the asker's admitted Continue, so it names its route.
MAILBOX_ROUTE_FACT = ("Host fact: this mailbox delivery is addressed to this work; this route does not broadcast "
                      "the answer to unrelated work.")


def answer_message_id(asker: str, quiz_id: str) -> str:
    """The mailbox identity ``deliver_answer`` gives a card's live answer, whichever task reads it."""
    return f"quiz_answer:{asker}:{quiz_id}"


def admitted_successor(root, predecessor: str, *, read=None) -> str:
    """A durable admission, independently verified on both ends of Continue."""
    from ouroboros.owner_continue import binding_sha
    from ouroboros.task_results import load_task_result
    if read is None:
        def read(tid):
            return load_task_result(root, tid, strict=True)
    try:
        result = read(predecessor) or {}
    except (ValueError, OSError):
        return ""
    claim = result.get("continued_by") or {}
    if not isinstance(claim, dict):
        return ""
    binding = claim.get("binding") or {}
    successor = str(claim.get("successor_task_id") or "")
    if (not isinstance(binding, dict) or claim.get("state") != "admitted" or not successor or binding.get("successor_task_id") != successor
            or binding.get("predecessor_task_id") != predecessor
            or claim.get("binding_sha256") != binding_sha(binding)):
        return ""
    try:
        target = read(successor) or {}
    except (ValueError, OSError):
        return ""
    admission = target.get("continuation_admission") or {}
    if not isinstance(admission, dict) or admission.get("binding") != binding or admission.get("binding_sha256") != binding_sha(binding):
        return ""
    return successor


def reading_recipient(root, asker: str, live_reader) -> tuple[str, dict | None]:
    current, seen = asker, set()
    recipient = ("", None)
    while current:
        if current in seen:
            return "", None
        seen.add(current)
        task, _reason = live_reader(current)
        if task is not None:
            if current == asker:
                return current, task
            recipient = (current, task)
        current = admitted_successor(root, current)
    return recipient


@contextmanager
def reader_mailbox(root, asker, live_reader):
    """Hold the addressed reader's admission lock through its mailbox append, following a last-drain race."""
    from ouroboros.conversation_lease import presence_author_agent
    from supervisor import queue as q
    from supervisor.workers import get_direct_chat_agent

    recipient, task = reading_recipient(root, asker, live_reader)
    attempted = set()
    while task is not None and recipient not in attempted:
        attempted.add(recipient)
        agent = get_direct_chat_agent(recipient) or presence_author_agent(root, recipient)
        with getattr(agent, "_owner_message_admission_lock", None) or q._queue_lock:
            current, _reason = live_reader(recipient)
            if current is not None:
                yield recipient, Path(root) if agent is not None else q._task_drive_for_task(current, recipient)
                return
        recipient, task = reading_recipient(root, asker, live_reader)
    yield "", None


def reader_line_ended(root, asker, *, stamps=None):
    """Positive end-of-line evidence even before the supervisor has restored its queue."""
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, load_task_result

    records = {}

    def read(tid):
        if tid not in records:
            records[tid], stamp = load_task_result(root, tid, strict=True, signed=True)
            if stamps is not None:
                stamps[tid] = stamp
        return records[tid]

    current, seen = asker, set()
    while current and current not in seen:
        seen.add(current)
        result = read(current) or {}
        if result.get("status") not in _TRULY_TERMINAL_STATUSES:
            return False
        if not result.get("continued_by"):
            return True
        current = admitted_successor(root, current, read=read)
    return False  # An incomplete admission or a cycle does not prove the letter is owed nowhere.


def _update_delivery(root, asker, quiz_id, changes, *, expected=None):
    from ouroboros.owner_quiz import _mutate_projection, _KEEP, question_delivery_settled
    def update(quizzes):
        block = quizzes.get(quiz_id)
        if not block or question_delivery_settled(block):
            return _KEEP
        delivery = block.get("delivery") or {}
        if expected and any(delivery.get(k) != v for k, v in expected.items()):
            return _KEEP
        block["delivery"] = {**delivery, **changes}
        if changes.get("read_at") and delivery.get("kind") == "closure" and delivery.get("reason_code") == "publication_address_unknown":
            # The new read can retire a previously imported no-target verdict on the next retry.
            block["delivery"].pop("reason_code", None)
        return True
    return (_mutate_projection(root, asker, update).get(quiz_id) or {})


def acknowledge_answer(root, recipient: str, entry: dict) -> None:
    """Record this recipient's saved answer or closure letter against its current route."""
    from ouroboros.owner_quiz import mark_discharged
    from ouroboros.utils import utc_now_iso
    provenance = entry.get("quiz_delivery")
    if not isinstance(provenance, dict):
        return
    asker, quiz_id = provenance.get("task_id"), provenance.get("quiz_id")
    if not asker or not quiz_id:
        return
    canonical = Path(provenance.get("canonical_root") or root)
    expected = {"recipient_task_id": recipient, "msg_id": entry.get("msg_id"),
                "kind": "closure" if entry.get("kind") == "quiz_closed" else "answer"}
    block = _update_delivery(canonical, asker, quiz_id, {"read_at": utc_now_iso()}, expected=expected)
    delivery = block.get("delivery") or {}
    if delivery.get("read_at") and delivery.get("history_at"):
        mark_discharged(canonical, asker, quiz_id)


def closure_source(entry: dict) -> dict:
    """A closure letter names the original ask, not a card of the reader."""
    source = entry.get("quiz_delivery")
    if entry.get("kind") != "quiz_closed" or not isinstance(source, dict):
        return {}
    asker, quiz_id = source.get("task_id"), source.get("quiz_id")
    return source if (all(isinstance(v, str) and v for v in (asker, quiz_id))
                      and entry.get("msg_id") == f"quiz_closed:{quiz_id}") else {}


def closure_directive(entry: dict) -> dict:
    """Only an attested owner reason is speech; the frame is the host's."""
    from ouroboros.owner_quiz import secret_outcome_block

    source = closure_source(entry)
    words = source.get("owner_close_words")
    if source.get("closed_actor") != "owner" or not isinstance(words, str) or not words.strip() or secret_outcome_block(entry):
        return {}
    return {"source": "owner_question_close", "content": words, "msg_id": entry["msg_id"],
            "source_task_id": source["task_id"], "quiz_id": source["quiz_id"], "ts": source.get("closed_at") or ""}


def legacy_closure_directive(row: dict) -> bool:
    return row.get("source") == "owner_mailbox" and str(row.get("msg_id") or "").startswith("quiz_closed:")


def restore_closure_directives(root, task_id, rows):
    """Project a legacy checkpoint's owner corpus without changing its saved transcript."""
    from ouroboros.owner_mailbox import KIND_QUIZ_CLOSED, drain_owner_entries

    if not any(isinstance(row, dict) and legacy_closure_directive(row) for row in rows):
        return rows
    letters = {entry["msg_id"]: closure_directive(entry) for entry in drain_owner_entries(
        Path(root), task_id, include_acknowledged=True, kinds={KIND_QUIZ_CLOSED})} if root else {}
    result = []
    for row in rows:
        if not isinstance(row, dict) or not legacy_closure_directive(row):
            result.append(row)
        elif not row.get("carried") and (words := letters.get(row["msg_id"])) and all(
                row.get(k, words[k]) == words[k] for k in ("source_task_id", "quiz_id")):
            result.append(words)
    return result


def closure_frame(quiz_id, closed_at, reason):
    """Ordinary closure text shared by the mailbox writer and the reader of historical deliveries."""
    return (f"[Owner question closed without answer] quiz {quiz_id} — closed {closed_at}. Reason: {reason}\n"
            "This closes the question; no option was chosen and no approval was given.")


def _closure_letter(root, asker, quiz_id, block):
    from ouroboros.owner_quiz import secret_outcome_block
    from ouroboros.secret_requests import cancelled_frame

    source = {"task_id": asker, "quiz_id": quiz_id, "closed_actor": block["closed_actor"],
              "closed_at": block["closed_at"], "canonical_root": str(root)}
    if secret_outcome_block(block) is not None:
        return cancelled_frame(block), {**source, "answer_snapshot": dict(block)}
    if block.get("owner_wrote_reason") is True:
        source["owner_close_words"] = block["closed_reason"]
    return closure_frame(quiz_id, block["closed_at"], block["closed_reason"]), source


def _matches_closure(root, recipient, asker, quiz_id, block, entry):
    """Recognize the producer's letter. Historical frame bytes corroborate a card, never supply authority."""
    if (entry.get("kind") != "quiz_closed" or entry.get("msg_id") != f"quiz_closed:{quiz_id}"
            or block.get("quiz_id") != quiz_id or block.get("state") != "closed" or block.get("closed_actor") != "owner"
            or not all(isinstance(block.get(k), str) and block[k] for k in ("closed_at", "closed_reason"))
            or (block.get("delivery") or {}).get("kind") != "closure"
            or (block.get("delivery") or {}).get("logical_id") != f"quiz_closure:{asker}:{quiz_id}"):
        return False
    frame, source = _closure_letter(root, asker, quiz_id, block)
    if entry.get("text") != frame:
        return False
    if "quiz_delivery" not in entry:
        return (recipient == asker and "first_route_pending" not in block["delivery"] and "owner_wrote_reason" not in block
                and "answer_snapshot" not in source and set(entry) == {"kind", "ts", "text", "msg_id"}
                and isinstance(entry["ts"], str) and bool(entry["ts"]))
    written = closure_source(entry)
    if (written.get("task_id"), written.get("quiz_id")) != (asker, quiz_id):
        return False
    address = written.get("canonical_root")
    if not isinstance(address, str) or not address or Path(address).resolve() != Path(root).resolve():
        return False
    if ("answer_snapshot" in source) != ("answer_snapshot" in written):
        return False
    if "answer_snapshot" in source:
        snapshot = written.get("answer_snapshot")
        if not isinstance(snapshot, dict) or any(snapshot.get(k) != block.get(k) for k in (
                "quiz_id", "closed_at", "closed_actor", "closed_reason", "secret_request", "secret_outcome")):
            return False
        if "closed_actor" not in written:  # the old secret cancel carried only the card
            return (recipient == asker and "first_route_pending" not in block["delivery"]
                    and set(written) == {"task_id", "quiz_id", "canonical_root", "answer_snapshot"})
    return all(written.get(k) == source.get(k) for k in ("closed_actor", "closed_at", "owner_close_words"))


def _closure_mail(root, recipient, drive, result, *, retiring=False):
    """Join the recipient's exact letter to its ACK. Callers changing the route hold its mail lock.
    A capture supplies bytes. Only a surviving transcript ACK supplies a read; None denotes ambiguous evidence."""
    import json
    from ouroboros.owner_mailbox import acknowledged_task_message_ids
    from ouroboros.task_custody import mailbox_records, own_child_drives

    drive = Path(drive).resolve()
    records, complete = mailbox_records(drive, recipient)
    def read_receipts(location):
        status, ordinary = {}, set()
        ids = acknowledged_task_message_ids(location, recipient, _read_status=status, _require_terminator=True,
                                            _loop_receipts=ordinary, _require_owner=retiring and location == drive)
        return ids, ordinary, status

    drive_receipts = read_receipts(drive)
    if retiring and not any(entry.get("kind") == "quiz_closed" for _, entry in records):
        receipts, ordinary, status = drive_receipts
        if not complete or not status["complete"]:
            return {}, False
        if not records and receipts and "owner_mailbox" not in result:
            return {}, False  # no verified capture for a receipt-only file
        if not receipts or (records and not any(mid.startswith("quiz_closed:") for mid in ordinary)):
            return {}, True
    locations = {Path(root).resolve(), drive, *own_child_drives(root, recipient, strict=True)}
    letters, reads, held, duplicates = {}, {}, {}, set()
    receipts_complete = True
    for location in locations:
        rows, whole = (records, complete) if location == drive else mailbox_records(location, recipient)
        if not whole:
            return {}, False  # a missing letter could have the same ID
        receipts, loop, status = drive_receipts if location == drive else read_receipts(location)
        if status.get("unavailable"):
            return {}, False
        receipts_complete = receipts_complete and status["complete"]
        for mid in receipts:
            if mid.startswith("quiz_closed:"):
                reads.setdefault(mid, {})[location] = mid in loop
        for line, entry in rows:
            mid = entry.get("msg_id")
            if isinstance(mid, str) and mid.startswith("quiz_closed:"):
                letters.setdefault(mid, []).append((location, line, entry))
            elif entry.get("kind") == "quiz_closed":
                return {}, False
    owner_lines = set()
    for key in ("owner_mailbox", "unread_mailbox"):
        if key not in result:
            continue  # a complete empty unread capture is normally omitted
        capture = result[key]
        if (not isinstance(capture, dict) or not isinstance(capture.get("rows"), list)
                or (key == "owner_mailbox" and capture.get("read_complete") is not True)
                or capture.get("total", len(capture["rows"])) != len(capture["rows"])):
            return {}, False
        seen = set()
        for line in capture["rows"]:
            try:
                entry = json.loads(line)
            except (TypeError, ValueError):
                return {}, False
            if not isinstance(entry, dict):
                return {}, False
            mid = entry.get("msg_id")
            if isinstance(mid, str) and mid.startswith("quiz_closed:"):
                if mid in seen:
                    duplicates.add(mid)
                seen.add(mid)
                held.setdefault(mid, set()).add(line)
                if key == "owner_mailbox":
                    owner_lines.add(line)
            elif entry.get("kind") == "quiz_closed":
                return {}, False
    observations = dict.fromkeys(letters.keys() | reads.keys() | held.keys())
    for mid, copies in letters.items():
        if len(copies) == 1 and mid not in duplicates:
            location, line, entry = copies[0]
            if not (held.get(mid, set()) - {line} or reads.get(mid, {}).keys() - {location}):
                observations[mid] = location, entry, reads.get(mid, {}).get(location, False)
    # Old cleanup unlinked the mailbox first. Its surviving ordinary ACK can still identify the captured letter.
    for mid in reads.keys() - letters.keys() - duplicates:
        lines = held.get(mid, set())
        if len(reads[mid]) == len(lines) == 1 and lines <= owner_lines:
            location, read = next(iter(reads[mid].items()))
            if read:
                observations[mid] = location, json.loads(next(iter(lines))), True
    if retiring and any(by_drive.get(drive) and observations[mid] is None for mid, by_drive in reads.items()):
        return observations, False  # retain an ordinary receipt whose original letter is ambiguous
    return observations, receipts_complete


_ROUTE_FIELDS = ("recipient_task_id", "recipient_drive", "msg_id", "mailbox_at", "read_at", "first_route_pending")


def _closure_route(delivery):
    return {key: delivery.get(key) for key in _ROUTE_FIELDS}


def _on_closure_route(route, recipient, asker, mid, location):
    return ((route["recipient_task_id"] or asker) == recipient and route["msg_id"] in (None, "", mid)
            and (not route["recipient_drive"] or Path(route["recipient_drive"]).resolve() == location))


def reconcile_closure_mail(root, recipient, drive, result, *, check_only=False):
    """Settle ordinary reads and release unread routes on their cards before retiring the locked mailbox.
    A failed projection keeps the source. Historical ambiguity stays on the original route."""
    from ouroboros.task_custody import fence_publication
    from ouroboros.task_results import load_task_result, validate_task_id
    from ouroboros.utils import utc_now_iso

    observed, complete = _closure_mail(root, recipient, drive, result, retiring=True)
    if not complete:
        return False
    cards = {recipient: result}
    for mid, copy in observed.items():
        if not copy or copy[0] != Path(drive).resolve():
            continue
        location, entry, read = copy
        source = closure_source(entry)
        asker = source.get("task_id", recipient)
        if asker not in cards:
            try:
                valid = validate_task_id(asker) == asker
            except ValueError:
                valid = False
            cards[asker] = (load_task_result(root, asker, strict=True) or {}) if valid else {}
        qid = mid.removeprefix("quiz_closed:")
        block = (cards[asker].get("owner_quiz") or {}).get(qid) or {}
        delivery = block.get("delivery") or {}
        route = _closure_route(delivery)
        if (not _matches_closure(root, recipient, asker, qid, block, entry)
                or route["first_route_pending"] is True or not _on_closure_route(route, recipient, asker, mid, location)):
            if read:
                return False
            continue
        if delivery.get("state") != "owed" or route["read_at"]:
            continue
        if check_only:
            return False
        changes = ({"recipient_task_id": recipient, "recipient_drive": str(location), "msg_id": mid, "read_at": utc_now_iso()}
                   if read else {"recipient_task_id": "", "recipient_drive": "", "msg_id": "",
                                 "mailbox_at": "", "first_route_pending": True})
        fence_publication()
        updated = _update_delivery(root, asker, qid, changes, expected=route).get("delivery") or {}
        if updated.get("state") == "owed" and _closure_route(updated) == route:
            return False
    return True


def closure_was_read(root, recipient, entry, block, result):
    """Read-only Continue evidence. Retirement may adopt the ordinary ACK and remove it during this observation."""
    from ouroboros.owner_quiz import quiz_states

    source = closure_source(entry)
    if not source:
        return False
    asker, qid, mid = source["task_id"], source["quiz_id"], entry["msg_id"]
    delivery = block.get("delivery") or {}
    if ((delivery.get("recipient_task_id"), delivery.get("msg_id")) != (recipient, mid)
            or not _matches_closure(root, recipient, asker, qid, block, entry)):
        return False
    if delivery.get("read_at"):
        return True
    try:
        observed, _ = _closure_mail(root, recipient, delivery.get("recipient_drive") or root, result)
    except (OSError, ValueError, TypeError):
        observed = {}
    latest = (quiz_states(root, asker).get(qid) or {}).get("delivery") or {}
    if (latest.get("recipient_task_id"), latest.get("msg_id")) != (recipient, mid):
        return False
    copy = observed.get(mid)
    return bool(latest.get("read_at") or (copy and copy[2]
                and _on_closure_route(_closure_route(latest), recipient, asker, mid, copy[0])
                and _matches_closure(root, recipient, asker, qid, block, copy[1])))


def deliver_closure(root, asker, quiz_id, block, live_reader) -> str:
    """Place the owner's closure on its admitted reader's line. A closure starts no new turn."""
    from ouroboros.owner_mailbox import KIND_QUIZ_CLOSED, mailbox_drain_ended, write_owner_message
    from ouroboros.task_custody import task_mail_lock
    from ouroboros.task_results import load_task_result
    from ouroboros.utils import utc_now_iso

    route = _closure_route(block.get("delivery") or {})
    if route["read_at"]:
        return ""
    msg_id = f"quiz_closed:{quiz_id}"
    if route["msg_id"] not in (None, "", msg_id):
        return "mailbox_read_unknown"
    previous = route["recipient_task_id"] or asker
    previous_drive = Path(route["recipient_drive"] or root)
    fresh = route["first_route_pending"] is True and not any(
        route[k] for k in ("recipient_task_id", "recipient_drive", "msg_id", "mailbox_at"))
    with reader_mailbox(root, asker, live_reader) as (recipient, drive):
        if not fresh:
            rerouting = drive is not None and recipient != previous
            with task_mail_lock(previous_drive, previous) as locked:
                if not locked:
                    return "mailbox_read_unknown"
                previous_row = load_task_result(root, previous, strict=True) or {}
                ended = rerouting and any(mailbox_drain_ended(location, previous) for location in {Path(root), previous_drive})
                try:
                    observed, complete = _closure_mail(root, previous, previous_drive, previous_row)
                except (OSError, ValueError, TypeError):
                    return "mailbox_read_unknown"
                copy = observed.get(msg_id)
                if not complete and not (copy and copy[2]):
                    return "mailbox_read_unknown"
                if msg_id in observed:
                    if (not copy or not _on_closure_route(route, previous, asker, msg_id, copy[0])
                            or not _matches_closure(root, previous, asker, quiz_id, block, copy[1])):
                        return "mailbox_read_unknown"
                    if copy[2]:
                        target = {"recipient_task_id": previous, "recipient_drive": str(copy[0]), "msg_id": msg_id}
                        updated = _update_delivery(root, asker, quiz_id, {**target, "read_at": utc_now_iso()}, expected=route)
                        return "" if (updated.get("delivery") or {}).get("read_at") else "awaiting_recipient_read"
                    if previous == recipient:
                        if not route["mailbox_at"]:
                            _update_delivery(root, asker, quiz_id, {"mailbox_at": utc_now_iso()}, expected=route)
                        return "awaiting_recipient_read"
                else:
                    unread = previous_row.get("unread_mailbox", {"read_complete": True})
                    if route["mailbox_at"] or (rerouting and ("owner_mailbox" not in previous_row or not unread.get("read_complete"))):
                        return "mailbox_read_unknown"
                if rerouting and not ended:
                    return "awaiting_recipient_read"
        if drive is None:
            return "no_recipient"  # only a signed terminal line may retire the unread letter
        frame, source = _closure_letter(root, asker, quiz_id, block)
        target = {"recipient_task_id": recipient, "recipient_drive": str(drive), "msg_id": msg_id}
        named = _update_delivery(root, asker, quiz_id, {**target, "mailbox_at": "", "first_route_pending": False}, expected=route)
        current = named.get("delivery") or {}
        if current.get("read_at") or current.get("state") != "owed":
            return ""
        if current.get("mailbox_at") or any(current.get(k) != v for k, v in target.items()):
            return "awaiting_recipient_read"
        if not write_owner_message(drive, frame, recipient, msg_id=msg_id, kind=KIND_QUIZ_CLOSED, quiz_delivery=source):
            return "mailbox_write_failed"
        _update_delivery(root, asker, quiz_id, {"mailbox_at": utc_now_iso()}, expected=target)
        return "awaiting_recipient_read"


def deliver_answer(root, asker: str, quiz_id: str, block: dict, live_reader, fallback) -> tuple[bool, str]:
    """Return (forwarded_to_owner_ingress, reason); a living recipient's mailbox remains owed until ACK."""
    from ouroboros.owner_mailbox import KIND_QUIZ_ANSWER, mail_read_state, mailbox_holds, write_owner_message
    from ouroboros.owner_quiz import mark_discharged, quiz_states, recorded_answer_frame, secret_outcome_block
    from ouroboros.utils import utc_now_iso
    from supervisor import message_bus

    block = quiz_states(root, asker).get(quiz_id) or block
    delivery = block.get("delivery") or {}
    previous = str(delivery.get("recipient_task_id") or "")
    if not previous and delivery.get("mailbox_at") and not delivery.get("route"):
        previous = asker
        delivery = {**delivery, "recipient_task_id": asker, "msg_id": f"quiz_answer:{quiz_id}"}
        _update_delivery(root, asker, quiz_id, delivery)
    if previous and delivery.get("msg_id"):
        roots = {Path(root), Path(delivery.get("recipient_drive") or root)}
        if delivery.get("read_at") or any(mail_read_state(r, previous, delivery["msg_id"], transcript_only=True) is True for r in roots):
            _update_delivery(root, asker, quiz_id, {"read_at": utc_now_iso()})
            if delivery.get("history_at"):
                mark_discharged(root, asker, quiz_id)
                return False, ""
            return False, "quiz_history_write_pending"
    if previous and delivery.get("msg_id") and not delivery.get("mailbox_at"):
        # The append can land before its receipt write fails: the copy still in the mailbox is the receipt.
        if any(mailbox_holds(r, previous, delivery["msg_id"]) for r in {Path(root), Path(delivery.get("recipient_drive") or root)}):
            repaired = _update_delivery(root, asker, quiz_id, {"mailbox_at": utc_now_iso()}).get("delivery")
            delivery = repaired or {**delivery, "mailbox_at": utc_now_iso()}
    if previous and delivery.get("mailbox_at"):
        # An unread copy stays owed to its recipient while that work exists:
        # Pause, sleep or a hold moves it nowhere; only its end does (below).
        current, reason = live_reader(previous)
        if current is not None or reason == "task_not_reading":
            return False, "awaiting_recipient_read"
    # An answer the owner's ingress already accepted keeps that one effect: its retry converges
    # there, never on a Continue admitted since (a press does the same, card_decision._deliver).
    late_id, chat_id = f"quiz_late_answer:{asker}:{quiz_id}", block.get("publication_chat_id")
    accepted = type(chat_id) is int and message_bus.accepted_chat_message(root, chat_id, late_id) is not None
    if not accepted:
        with reader_mailbox(root, asker, live_reader) as (recipient, drive):
            if drive is not None:
                msg_id = answer_message_id(asker, quiz_id)
                if previous == recipient and delivery.get("mailbox_at"):
                    return False, "awaiting_recipient_read"
                # Name the owed effect BEFORE append. A crash retries that same identity.
                _update_delivery(root, asker, quiz_id, {"recipient_task_id": recipient, "msg_id": msg_id,
                    "recipient_drive": str(drive), "state": "owed", "mailbox_at": "", "reason_code": "awaiting_recipient_read"})
                letter = recorded_answer_frame(block)  # a secret outcome is host metadata: no owner words to route
                letter += "" if secret_outcome_block(block) is not None else "\n" + MAILBOX_ROUTE_FACT
                if not write_owner_message(drive, letter, recipient, msg_id=msg_id,
                        kind=KIND_QUIZ_ANSWER, quiz_delivery={"task_id": asker, "quiz_id": quiz_id,
                                                          "canonical_root": str(root), "answer_snapshot": dict(block)}):
                    return False, "mailbox_write_failed"
                _update_delivery(root, asker, quiz_id, {"mailbox_at": utc_now_iso()})
                return False, "awaiting_recipient_read"
    # No living holder. Named ingress carries the typed question-answer fact: owner
    # routing gives it only to the asker's live late-answer head, else a fresh head;
    # it releases no hold.
    block = _update_delivery(root, asker, quiz_id, {"recipient_task_id": "", "msg_id": late_id,
        "route": "fresh_head", "state": "owed", "reason_code": "asker_not_reading"}) or block
    forwarded, reason = fallback(root, asker, quiz_id, block, source=str(block.get("answer_source") or "web"))
    if forwarded:
        mark_discharged(root, asker, quiz_id)
    return forwarded, reason
