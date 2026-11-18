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


def _retained_unread_closure(root, previous, asker, quiz_id, msg_id):
    """Use the old reader's captured unread letter, not the fact its drain ended."""
    import json
    from ouroboros.task_results import load_task_result
    from ouroboros.task_status import SETTLED_STATUSES

    try:
        row = load_task_result(root, previous, strict=True) or {}
        owner, unread = row.get("owner_mailbox"), row.get("unread_mailbox")
        if row.get("status") not in SETTLED_STATUSES or not all(isinstance(v, dict) and v.get("read_complete") is True
                                                               for v in (owner, unread)):
            return False
        matching = []
        for line in owner.get("rows") or []:
            entry = json.loads(line)
            if isinstance(entry, dict) and entry.get("msg_id") == msg_id:
                matching.append((line, closure_source(entry)))
        if len(matching) != 1:
            return False  # an unread duplicate does not prove the original went unread
        line, source = matching[0]
        return (source.get("task_id"), source.get("quiz_id")) == (asker, quiz_id) and line in (unread.get("rows") or [])
    except (OSError, ValueError, TypeError, RuntimeError):
        return False


def deliver_closure(root, asker, quiz_id, block, live_reader) -> str:
    """Place the owner's closure in the same work an answer would reach. A closure starts no new turn."""
    from ouroboros.owner_mailbox import KIND_QUIZ_CLOSED, mail_read_state, mailbox_drain_ended, mailbox_holds, write_owner_message
    from ouroboros.owner_quiz import secret_outcome_block
    from ouroboros.utils import utc_now_iso

    delivery = block.get("delivery") or {}
    if delivery.get("read_at"):
        return ""
    msg_id = f"quiz_closed:{quiz_id}"
    previous = delivery.get("recipient_task_id") or asker  # older closures addressed only their asker
    previous_drive = Path(delivery.get("recipient_drive") or root)
    route = {key: delivery.get(key) for key in ("recipient_task_id", "recipient_drive", "msg_id", "mailbox_at",
                                               "read_at", "first_route_pending")}
    fresh = route["first_route_pending"] is True and not any(
        route[k] for k in ("recipient_task_id", "recipient_drive", "msg_id", "mailbox_at"))
    with reader_mailbox(root, asker, live_reader) as (recipient, drive):
        if not fresh:
            roots = {Path(root), previous_drive}
            rerouting = drive is not None and previous != recipient
            # Read terminal custody before ACKs: a live predecessor could still acknowledge.
            ended = rerouting and any(mailbox_drain_ended(r, previous) for r in roots)
            reads = [mail_read_state(r, previous, msg_id, transcript_only=True) for r in roots]
            if True in reads:
                updated = _update_delivery(root, asker, quiz_id, {"read_at": utc_now_iso()}, expected=route)
                return "" if (updated.get("delivery") or {}).get("read_at") else "awaiting_recipient_read"
            if rerouting:
                if not ended:
                    return "mailbox_read_unknown" if None in reads else "awaiting_recipient_read"
                # Retirement removes mail before ACKs. Check for the letter after reading ACKs.
                if (None in reads or not any(mailbox_holds(r, previous, msg_id) for r in roots)) and not (
                        _retained_unread_closure(root, previous, asker, quiz_id, msg_id)):
                    return "mailbox_read_unknown"
            elif None in reads:
                return "mailbox_read_unknown"
        if drive is None:
            return "no_recipient"  # only a signed terminal line may retire the unread letter
        if previous == recipient and delivery.get("mailbox_at"):
            return "awaiting_recipient_read"
        frame = (f"[Owner question closed without answer] quiz {quiz_id} — "
                 f"closed {block['closed_at']}. Reason: {block['closed_reason']}\n"
                 "This closes the question; no option was chosen and no approval was given.")
        source = {"task_id": asker, "quiz_id": quiz_id, "closed_actor": block["closed_actor"],
                  "closed_at": block["closed_at"], "canonical_root": str(root)}
        if block.get("owner_wrote_reason"):
            source["owner_close_words"] = block["closed_reason"]
        if secret_outcome_block(block) is not None:
            from ouroboros.secret_requests import cancelled_frame

            frame = cancelled_frame(block)
            source["answer_snapshot"] = dict(block)
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
