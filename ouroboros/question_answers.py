"""Address accepted answers to their living asker, an admitted successor, or a fresh head.

A living asker holds its answers while paused, held or queued and reads them
when it continues. Mailbox write is owed delivery, never a read receipt. A
final drain/terminal race is repaired from the durable recipient and ACK
identity; no room's sole unrelated reader and no claimed-but-unadmitted
Continue may take the answer. Once the owner's ingress accepted it, every
retry converges on that message.
"""
from __future__ import annotations

from pathlib import Path


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


def _update_delivery(root, asker, quiz_id, changes):
    from ouroboros.owner_quiz import _mutate_projection, _KEEP, question_delivery_settled
    def update(quizzes):
        block = quizzes.get(quiz_id)
        if not block or question_delivery_settled(block):
            return _KEEP
        block["delivery"] = {**(block.get("delivery") or {}), **changes}
        return True
    return (_mutate_projection(root, asker, update).get(quiz_id) or {})


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


def deliver_answer(root, asker: str, quiz_id: str, block: dict, live_reader, fallback) -> tuple[bool, str]:
    """Return (forwarded_to_owner_ingress, reason); a living recipient's mailbox remains owed until ACK."""
    from ouroboros.owner_mailbox import KIND_QUIZ_ANSWER, mail_read_state, mailbox_holds, write_owner_message
    from ouroboros.owner_quiz import mark_discharged, recorded_answer_frame, quiz_states
    from ouroboros.utils import utc_now_iso
    from supervisor import message_bus, queue as q
    from supervisor.workers import get_direct_chat_agent

    block = quiz_states(root, asker).get(quiz_id) or block
    delivery = block.get("delivery") or {}
    previous = str(delivery.get("recipient_task_id") or "")
    if not previous and delivery.get("mailbox_at") and not delivery.get("route"):
        previous = asker
        delivery = {**delivery, "recipient_task_id": asker, "msg_id": f"quiz_answer:{quiz_id}"}
        _update_delivery(root, asker, quiz_id, delivery)
    if previous and delivery.get("msg_id"):
        roots = {Path(root), Path(delivery.get("recipient_drive") or root)}
        if delivery.get("read_at") or any(mail_read_state(r, previous, delivery["msg_id"]) is True for r in roots):
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
    recipient, task = ("", None) if accepted else reading_recipient(root, asker, live_reader)
    attempted = set()
    while task is not None and recipient not in attempted:
        attempted.add(recipient)
        agent = get_direct_chat_agent(recipient)
        if agent is None:
            from ouroboros.conversation_lease import presence_author_agent
            agent = presence_author_agent(root, recipient)
        lock = getattr(agent, "_owner_message_admission_lock", None) or q._queue_lock
        with lock:
            # Same admission owner as room intake: no final-drain admission after closure.
            current, _reason = live_reader(recipient)
            if current is not None:
                drive = Path(root) if agent is not None else q._task_drive_for_task(current, recipient)
                msg_id = answer_message_id(asker, quiz_id)
                if previous == recipient and delivery.get("mailbox_at"):
                    return False, "awaiting_recipient_read"
                # Name the owed effect BEFORE append. A crash retries that same identity.
                _update_delivery(root, asker, quiz_id, {"recipient_task_id": recipient, "msg_id": msg_id,
                    "recipient_drive": str(drive), "state": "owed", "mailbox_at": "", "reason_code": "awaiting_recipient_read"})
                if not write_owner_message(drive, recorded_answer_frame(block), recipient, msg_id=msg_id,
                        kind=KIND_QUIZ_ANSWER, quiz_delivery={"task_id": asker, "quiz_id": quiz_id,
                                                          "canonical_root": str(root), "answer_snapshot": dict(block)}):
                    return False, "mailbox_write_failed"
                _update_delivery(root, asker, quiz_id, {"mailbox_at": utc_now_iso()})
                return False, "awaiting_recipient_read"
        # The selected reader ended before admission. Follow its existing
        # Continue under that recipient's own lock before using a fresh head.
        recipient, task = reading_recipient(root, admitted_successor(root, recipient), live_reader)
    # No living holder. Named ingress carries the typed question-answer fact: owner
    # routing gives it only to the asker's live late-answer head, else a fresh head;
    # it releases no hold.
    block = _update_delivery(root, asker, quiz_id, {"recipient_task_id": "", "msg_id": late_id,
        "route": "fresh_head", "state": "owed", "reason_code": "asker_not_reading"}) or block
    forwarded, reason = fallback(root, asker, quiz_id, block, source=str(block.get("answer_source") or "web"))
    if forwarded:
        mark_discharged(root, asker, quiz_id)
    return forwarded, reason
