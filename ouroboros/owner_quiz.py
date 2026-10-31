"""Durable owner-quiz lifecycle projection (#Q-2b, the answer half of the
escalation channel).

This module owns everything quiz-lifecycle-specific — the asked/answered
projection in the task result, the request-id idempotent answer write, and
the terminal reconciliation — so the pinned surfaces (the escalate tool, the
decision ingress, ``supervisor/queue_transitions``) keep thin dispatch only,
mirroring ``ouroboros/owner_hurry.py``.

Shape inside the current task-store record (canonical drive root, exactly
like the hurry projection):

    "owner_quiz": {
        "<quiz_id>": {
            "quiz_id", "question", "options": [label, ...], "stake",
            "option_details"?: [detail, ...], "recommended_index"?: int,
            "assumption", "state": open|answered|expired_terminal|closed,
            "asked_at", "answered_at"?, "answered_index"?, "request_id"?,
            "comment"?, "reconciled_at"?, "chat_id"?, "max_wait_minutes"?,
            "answered_after_terminal"?, "host_facts"?, "materials"?,
            "publication_chat_id"?, "publication_bound_at"?,
            "closed_reason"?, "closed_at"?, "closed_by_task_id"?, "discharged_at"?,
        }, ...
    }

Expiry stays STRUCTURAL: ``reconcile_terminal`` runs on the task-done seam and
flips every still-open block to ``expired_terminal``; there is no host TTL. What
changed (owner decision В17a=A, which explicitly retired 30=A's "a quiz dies
with its author") is the fate of a LATE answer: the ingress may still record it
on an expired block (``record_answered(allow_expired=True)``), first answer
still wins, and the block keeps ``answered_after_terminal`` for audit. The
stored ``publication_chat_id`` is bound before exposure, so the answer goes to
the published card's chat even if the ask-time ``chat_id`` became stale, as an
ordinary owner message once its author is gone. That accepted message carries
the host's exact answer snapshot before the canonical block can be discharged.
The writers mutate ``owner_quiz`` and its paired terminal ``owner_wait`` via ``patch_document``
(never ``write_task_result`` — its status-regression guard can drop the
write), so concurrent terminal writers merge around it.
"""

from __future__ import annotations

import datetime
from typing import Any, Callable, Dict, List, Optional

from ouroboros.utils import utc_now_iso
from ouroboros.task_store import patch_document

STATE_OPEN = "open"
STATE_ANSWERED = "answered"
STATE_EXPIRED_TERMINAL = "expired_terminal"
STATE_CLOSED = "closed"

# Admission bounds OPEN questions only. Accepted unanswered questions and
# unsettled delivery debt never compete with the bounded discharged history.
_QUIZ_CAP = 16

# The explicit closure facts every card view copies beside the answer facts.
CLOSURE_FIELDS = ("closed_reason", "closed_at", "closed_by_task_id", "closed_actor", "closed_request_id")
# The owner closes in one tap; a closure without the owner's words records this sentence.
OWNER_CLOSE_REASON = "Closed by the owner without an answer."

def current_question(block: Any) -> bool:
    """Unanswered remains current after its author finishes, until explicit closure."""
    return isinstance(block, dict) and str(block.get("state") or STATE_OPEN) in {
        STATE_OPEN, STATE_EXPIRED_TERMINAL,
    }


def quiz_needs_retention(block: Any) -> bool:
    """Canonical question or answer/closure still owing durable delivery evidence; a debt the
    owner removed from current questions (``delivery_dismissed``) is no longer current."""
    return isinstance(block, dict) and (current_question(block) or
        (isinstance(block.get("delivery"), dict) and not question_delivery_settled(block)
         and not delivery_dismissed(block)))


def delivery_dismissed(block: Any) -> bool:
    """The owner removed an accepted answer of unknown delivery from current questions: nothing
    was delivered or resent, and the answer, its unknown outcome and its history stay."""
    return isinstance(block, dict) and bool((block.get("delivery") or {}).get("dismissed_at"))


def question_delivery_settled(block: Any) -> bool:
    """Typed delivery is authority; the old discharge stamp is a legacy fallback."""
    if not isinstance(block, dict):
        return False
    delivery = question_delivery_projection(block)
    return delivery["state"] == "delivered" if delivery else bool(block.get("discharged_at"))


def _delivery_debt(task_id: str, quiz_id: str, kind: str) -> Dict[str, Any]:
    return {"kind": kind, "state": "owed", "logical_id": f"quiz_{kind}:{task_id}:{quiz_id}",
            "attempts": 0, "next_retry_at": ""}


def question_delivery_projection(block: Any) -> Dict[str, str]:
    """Small factual UI view, without retry mechanics or internal error details."""
    delivery = block.get("delivery") if isinstance(block, dict) else None
    if not isinstance(delivery, dict) or delivery.get("kind") not in {"publication", "answer", "closure"}:
        return {}
    if delivery.get("state") not in {"owed", "delivered"}:
        return {}
    view = {"kind": delivery["kind"], "state": delivery["state"]}
    if delivery.get("reason_code"):
        view["reason_code"] = str(delivery["reason_code"])
    if delivery.get("dismissed_at"):
        view["dismissed"] = True
    return view


def record_answer_delivery_after_terminal(drive_root: Any, task_id: str, quiz_id: str) -> Dict[str, Any]:
    """The author settled before its recorded answer could be delivered; preserve that distinction."""
    def mutate(quizzes: Dict[str, Dict[str, Any]]) -> Any:
        block = quizzes.get(str(quiz_id))
        if (not block or block.get("state") != STATE_ANSWERED or question_delivery_settled(block)
                or block.get("answered_after_terminal") or block.get("delivery_after_terminal")):
            return _KEEP
        block["delivery_after_terminal"] = True
        return True

    return dict(_mutate_projection(drive_root, task_id, mutate).get(str(quiz_id)) or {})


def quiz_delivery_due(block: Any, *, now: Optional[str] = None) -> bool:
    """Current addressed debt only; the retained chat history is never polled. A debt waiting
    on a fact no retry supplies stays owed and current without automatic attempts: an answer or
    closure address only an import can bind (due again once bound), or an accepted late answer
    whose dispatch witness ended with its process (``late_answer_delivery_unknown``)."""
    if not isinstance(block, dict) or question_delivery_settled(block) or delivery_dismissed(block):
        return False
    delivery = block.get("delivery") if isinstance(block.get("delivery"), dict) else {}
    if delivery.get("state") == "delivered":
        return False
    if not delivery:
        return False  # Legacy schema absence authorizes neither publication nor owner-input replay.
    reason = delivery.get("reason_code")
    if reason == "late_answer_delivery_unknown" or (
            reason == "publication_address_unknown" and delivery.get("kind") != "publication"
            and "publication_chat_id" not in block):
        return False
    after = str(delivery.get("next_retry_at") or "")
    try:
        return not after or datetime.datetime.fromisoformat(after) <= datetime.datetime.fromisoformat(now or utc_now_iso())
    except (TypeError, ValueError):
        return True  # a corrupt retry stamp cannot erase the underlying obligation


def question_waits(document: Any, *, now: str) -> tuple:
    """What a task's retained questions wait on when none is due at NOW: ``(waits, due_at)``.

    ``owner_answer`` for a current question (open or ``expired_terminal``: only the owner's
    answer or closure moves it, В17a=A), an owed delivery's own typed reason (``delivery_retry``
    without one), and the earliest persisted retry instant that alone keeps a debt from being due.
    """
    waits, due = set(), []
    quizzes = document.get("owner_quiz") if isinstance(document, dict) else None
    for block in quizzes.values() if isinstance(quizzes, dict) else ():
        if not quiz_needs_retention(block):
            continue
        if current_question(block):
            waits.add("owner_answer")
        delivery = block.get("delivery") if isinstance(block.get("delivery"), dict) else {}
        if delivery and not question_delivery_settled(block) and not delivery_dismissed(block):
            waits.add(str(delivery.get("reason_code") or "delivery_retry"))
            when = str(delivery.get("next_retry_at") or "")
            if when and quiz_delivery_due(block, now=when):
                due.append(when)
    return sorted(waits) or ["nothing_owed"], min(due, key=_utc) if due else None


def _utc(value: str) -> datetime.datetime:
    stamp = datetime.datetime.fromisoformat(value)
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=datetime.timezone.utc)


def claim_quiz_delivery(drive_root: Any, task_id: str, quiz_id: str, *, force: bool = False) -> Dict[str, Any]:
    """Persist a bounded retry window BEFORE effects, also surviving a worker crash.

    No retry-count ceiling erases debt. Automatic attempts back off from 30 seconds
    to one hour; a deliberate owner retry may request the same canonical effect now.
    """
    claimed: Dict[str, Any] = {}
    now = utc_now_iso()

    def mutate(quizzes: Dict[str, Dict[str, Any]]) -> Any:
        block = quizzes.get(str(quiz_id))
        if not block or question_delivery_settled(block) or delivery_dismissed(block):
            return _KEEP  # a removed unknown is never attempted again, not even by an owner retry
        delivery = dict(block.get("delivery") or {})
        if not delivery or delivery.get("state") == "delivered" or (not force and not quiz_delivery_due(block, now=now)):
            return _KEEP
        attempts = max(0, int(delivery.get("attempts") or 0)) + 1
        delay = min(3600, 30 * 2 ** min(attempts - 1, 7))
        delivery.update(state="owed", attempts=attempts, last_attempt_at=now,
                        next_retry_at=(datetime.datetime.fromisoformat(now) + datetime.timedelta(seconds=delay)).isoformat())
        block["delivery"] = delivery
        claimed.update(block)
        return True

    _mutate_projection(drive_root, task_id, mutate)
    return claimed


def record_quiz_delivery_failure(drive_root: Any, task_id: str, quiz_id: str, reason: str) -> None:
    """Keep owed work with a typed cause, never a fabricated success or reset timer."""
    def mutate(quizzes: Dict[str, Dict[str, Any]]) -> Any:
        block = quizzes.get(str(quiz_id))
        if not block or question_delivery_settled(block) or delivery_dismissed(block):
            return _KEEP
        delivery = dict(block.get("delivery") or {})
        if delivery.get("state") == "delivered":
            return _KEEP
        delivery.update(state="owed", reason_code=str(reason), failed_at=utc_now_iso())
        block["delivery"] = delivery
        return True

    _mutate_projection(drive_root, task_id, mutate)


def mark_quiz_history_written(drive_root: Any, task_id: str, quiz_id: str, receipt: str = "history_at",
                              source_row: Optional[dict] = None) -> None:
    """Strict history append receipt; retries no longer fold history to find this fact.
    ``mailbox_at``: the live author's mailbox holds the answer or closure, which neither a
    retry nor the late path repeats while the history row alone stays owed."""
    def mutate(quizzes: Dict[str, Dict[str, Any]]) -> Any:
        block = quizzes.get(str(quiz_id))
        if not block:
            return _KEEP
        delivery = dict(block.get("delivery") or {})
        if delivery.get(receipt):
            return _KEEP
        delivery[receipt] = utc_now_iso()
        if source_row is not None:
            delivery["answer_row"] = dict(source_row)
        block["delivery"] = delivery
        return True

    _mutate_projection(drive_root, task_id, mutate)


def mark_published(drive_root: Any, task_id: str, quiz_id: str) -> None:
    """Only the strict publication writer records that the ask reached durable chat."""
    def mutate(quizzes: Dict[str, Dict[str, Any]]) -> Any:
        block = quizzes.get(str(quiz_id))
        if not block or block.get("published_at"):
            return _KEEP
        stamp = utc_now_iso()
        block["published_at"] = stamp
        delivery = dict(block.get("delivery") or {})
        if delivery.get("kind") == "publication":
            delivery.update(state="delivered", delivered_at=stamp)
            delivery.pop("reason_code", None)
            block["delivery"] = delivery
        return True

    _mutate_projection(drive_root, task_id, mutate)


class _Keep:
    pass


_KEEP = _Keep()


def _mutate_projection(
    drive_root: Any, task_id: str,
    mutator: Callable[[Dict[str, Dict[str, Any]]], Any],
) -> Dict[str, Dict[str, Any]]:
    """Locked writer for the ``owner_quiz`` key only (hurry idiom).

    ``mutator(quizzes)`` mutates the dict in place and returns ``_KEEP`` to
    abort (file untouched) or anything else to commit. Returns the post-write
    (or pre-write, on abort) quizzes view.
    """
    from ouroboros.task_results import (
        require_writable_task_result_schema,
        stamp_task_result_schema,
    )

    view: Dict[str, Dict[str, Any]] = {}

    def _mutate(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        view.clear()
        # ABI 7.0 (hurry idiom): never write over a row another schema owns;
        # every write stamps the row (stamp-on-write, not a converter).
        require_writable_task_result_schema(current)
        raw = current.get("owner_quiz")
        quizzes = {
            str(k): dict(v) for k, v in raw.items() if isinstance(v, dict)
        } if isinstance(raw, dict) else {}
        outcome = mutator(quizzes)
        if outcome is _KEEP:
            view.update(quizzes)
            return None
        if len(quizzes) > _QUIZ_CAP:
            wait = current.get("owner_wait") or {}
            paired_wait = str(wait.get("quiz_id") or "") if wait.get("state") == "waiting" else ""
            discharged = sorted(
                (key for key, block in quizzes.items()
                 if key != paired_wait and not quiz_needs_retention(block)),
                key=lambda key: str(quizzes[key].get("asked_at") or ""),
            )
            for key in discharged[:max(0, len(quizzes) - _QUIZ_CAP)]:
                quizzes.pop(key, None)
        updated = dict(current)
        updated["owner_quiz"] = quizzes
        view.update(quizzes)
        return stamp_task_result_schema(updated)

    patch_document(drive_root, str(task_id), _mutate)
    return view


def record_asked(
    drive_root: Any, task_id: str, *,
    quiz_id: str, question: str, options: List[str],
    stake: str = "", assumption: str = "",
    wait_for_answer: bool = False,
    option_details: Optional[List[str]] = None,
    recommended_index: Optional[int] = None,
    chat_id: Optional[int] = None,
    max_wait_minutes: Optional[int] = None,
    host_facts: str = "",
    materials: Optional[List[Dict[str, Any]]] = None,
    reply_to: Optional[Dict[str, Any]] = None,
    presence_contact: Optional[Dict[str, Any]] = None,
    request_fingerprint: str = "",
) -> Dict[str, Any]:
    """Worker-side projection write at ask time.

    The stored option labels are the ingress's validation authority: an
    ``option_index`` outside this list is refused, and the answer echoes the
    verbatim label back to the asking task.

    ``chat_id`` is the asker's proposed chat. Before exposure the publication
    writer binds ``publication_chat_id`` and its evidence; only that proven
    address can receive a late answer, including after Project conversion.
    ``max_wait_minutes`` records the bound a
    waiting asker chose, so replay can say what the task waited for.
    ``host_facts`` is the host-written sentence the card shows under the
    question (asking task, how its run started, the owner's last message in
    the chat); stored only when non-empty."""
    if option_details is not None and (
        not isinstance(option_details, list) or len(option_details) != len(options)
        or not all(isinstance(value, str) for value in option_details)
    ):
        raise ValueError("option_details must preserve the labels' length and order")
    from ouroboros.delegate_reply import normalize_reply_to

    reply_to = normalize_reply_to(reply_to)
    stamp = utc_now_iso()
    block = {
        "quiz_id": str(quiz_id), "question": str(question or ""),
        "options": [str(label) for label in options],
        **({"option_details": list(option_details)} if option_details is not None else {}),
        **({"recommended_index": int(recommended_index)} if isinstance(recommended_index, int) else {}),
        "stake": str(stake or ""), "assumption": str(assumption or ""),
        "state": STATE_OPEN, "asked_at": stamp,
        "delivery": _delivery_debt(task_id, quiz_id, "publication"),
        **({"wait_for_answer": True} if wait_for_answer else {}),
        **({"chat_id": int(chat_id)} if isinstance(chat_id, int) and not isinstance(chat_id, bool) else {}),
        **({"max_wait_minutes": int(max_wait_minutes)}
           if isinstance(max_wait_minutes, int) and not isinstance(max_wait_minutes, bool) else {}),
        **({"host_facts": str(host_facts)} if str(host_facts or "") else {}),
        **({"materials": [dict(ref) for ref in materials]} if materials else {}),
        **({"reply_to": reply_to} if reply_to else {}),
        **({"presence_contact": dict(presence_contact)} if presence_contact else {}),
        **({"request_fingerprint": request_fingerprint} if request_fingerprint else {}),
    }

    refused: Dict[str, str] = {}

    def _mutator(quizzes: Dict[str, Dict[str, Any]]) -> Any:
        if prior := quizzes.get(str(quiz_id)):
            if request_fingerprint and prior.get("request_fingerprint") != request_fingerprint:
                refused["reason"] = "request_id_conflict"
            elif wait_for_answer and prior.get("wait_ended_at") and prior.get("state") == STATE_OPEN:
                refused["reason"] = "wait_ended"
            return _KEEP  # asked once; a redelivery never resets an answer
        open_count = sum(
            1 for row in quizzes.values()
            if str(row.get("state") or STATE_OPEN) == STATE_OPEN
        )
        if open_count >= _QUIZ_CAP:
            # Refuse the ask instead of letting the cap evict an OPEN block:
            # an evicted open card would stay clickable in chat but 404 on
            # answer. The asker already carries an assumption to proceed on.
            refused["reason"] = "open_quiz_cap"
            return _KEEP
        quizzes[str(quiz_id)] = block
        return True

    from ouroboros.store import StoreError
    try:
        # The ask and its ``owner_questions`` flag commit in one store transaction.
        view = _mutate_projection(drive_root, task_id, _mutator)
    except StoreError:
        return {"refused": "owner_questions_unavailable"}
    if refused:
        return {"refused": refused["reason"]}
    return dict(view.get(str(quiz_id)) or block)


def record_publication(drive_root: Any, task_id: str, quiz_id: str, chat_id: int, *, evidence: str = "pre_exposure") -> Dict[str, Any]:
    """Bind the final publication destination before exposing any card.

    The first durable address wins across stale queued events and task rebinding.
    The original ``chat_id`` remains ask-time provenance, never delivery authority.
    """
    if not isinstance(chat_id, int) or isinstance(chat_id, bool):
        raise ValueError("quiz publication requires an integer chat_id")
    if evidence not in {"pre_exposure", "history"}:
        raise ValueError("quiz publication requires a strict pre-exposure or exact history witness")

    def mutate(quizzes: Dict[str, Dict[str, Any]]) -> Any:
        block = quizzes.get(str(quiz_id))
        if block is None or "publication_chat_id" in block:
            return _KEEP
        block.update(publication_chat_id=chat_id, publication_bound_at=utc_now_iso(),
                     publication_evidence=evidence)
        return True

    return dict(_mutate_projection(drive_root, task_id, mutate).get(str(quiz_id)) or {})


def dismiss_unknown_delivery(drive_root: Any, task_id: str, *, quiz_id: str, request_id: str) -> Dict[str, Any]:
    """The owner removes an accepted late answer whose dispatch witness was lost
    (``late_answer_delivery_unknown``) from current questions. The record keeps the answer, the
    unknown outcome and the removal (time, request); it is never marked delivered, never an
    approval, and nothing is resent. Idempotent: an already removed debt answers as a duplicate."""
    outcome: Dict[str, Any] = {}

    def mutate(quizzes: Dict[str, Dict[str, Any]]) -> Any:
        block = quizzes.get(str(quiz_id))
        if not block:
            outcome.update(ok=False, error="quiz_not_found")
            return _KEEP
        delivery = dict(block.get("delivery") or {})
        if delivery.get("dismissed_at"):
            outcome.update(ok=True, duplicate=True, state=block.get("state"), block=dict(block))
            return _KEEP
        if (block.get("state") != STATE_ANSWERED or question_delivery_settled(block)
                or delivery.get("state") != "owed" or delivery.get("reason_code") != "late_answer_delivery_unknown"):
            outcome.update(ok=False, error="delivery_not_unknown", state=block.get("state"), block=dict(block))
            return _KEEP
        delivery.update(dismissed_at=utc_now_iso(), dismissed_request_id=str(request_id))
        block["delivery"] = delivery
        outcome.update(ok=True, duplicate=False, state=STATE_ANSWERED, block=dict(block))
        return True

    _mutate_projection(drive_root, task_id, mutate)
    return outcome


def mark_discharged(drive_root: Any, task_id: str, quiz_id: str) -> Dict[str, Any]:
    """After durable answer delivery or closure evidence, permit bounded history GC."""
    discharged: Dict[str, Any] = {}
    def mutate(quizzes: Dict[str, Dict[str, Any]]) -> Any:
        block = quizzes.get(str(quiz_id))
        if not block or current_question(block) or question_delivery_settled(block):
            return _KEEP
        block["discharged_at"] = utc_now_iso()
        delivery = dict(block.get("delivery") or {})
        delivery.update(state="delivered", delivered_at=block["discharged_at"])
        delivery.pop("reason_code", None)
        block["delivery"] = delivery
        discharged.update(block)
        return True

    _mutate_projection(drive_root, task_id, mutate)
    return discharged


def record_closed(
    drive_root: Any, task_id: str, *, quiz_id: str, reason: str, actor_task_id: str = "",
    actor: str = "model", request_id: str = "",
) -> Dict[str, Any]:
    """The owner closes any unanswered question; a model closes its own, a recorded Continue/retry
    predecessor's, or a question whose task already finished (``expired_terminal``), always with
    its own explanation. A question a live task still owns stays its own line's or the owner's.

    Same Project membership, shared folders and prose never grant this authority.
    Closure is not an owner answer and never carries an answer index or approval.
    """
    reason = str(reason or "")
    if actor == "owner" and not reason.strip():
        reason = OWNER_CLOSE_REASON
    if not reason.strip():
        return {"ok": False, "error": "close_reason_required"}
    if actor not in {"owner", "model"}:
        return {"ok": False, "error": "close_actor_invalid"}
    # Authority beyond the author's line covers only a finished task's question (checked under the lock).
    own_line = actor != "model" or question_task_owned_by(drive_root, task_id, actor_task_id)
    request_id = str(request_id or (f"model-close:{actor_task_id}:{quiz_id}" if actor == "model" else ""))
    if not request_id:
        return {"ok": False, "error": "request_id_required"}
    outcome: Dict[str, Any] = {}

    def mutate(quizzes: Dict[str, Dict[str, Any]]) -> Any:
        block = quizzes.get(str(quiz_id))
        if not block:
            outcome.update(ok=False, error="quiz_not_found")
            return _KEEP
        if block.get("state") == STATE_CLOSED:
            duplicate = block.get("closed_request_id") == request_id and block.get("closed_actor", "model") == actor
            outcome.update(ok=duplicate, duplicate=duplicate, state=STATE_CLOSED, block=dict(block),
                           **({} if duplicate else {"error": "quiz_closed"}))
            return _KEEP
        if not current_question(block):
            outcome.update(ok=False, error="quiz_already_answered", state=block.get("state"), block=dict(block))
            return _KEEP
        if not own_line and block.get("state") != STATE_EXPIRED_TERMINAL:
            outcome.update(ok=False, error="question_not_owned", state=block.get("state"), block=dict(block))
            return _KEEP
        delivery = block.get("delivery") or {}
        pending_publication = delivery.get("kind") == "publication" and delivery.get("state") == "owed"
        block.update(state=STATE_CLOSED, closed_at=utc_now_iso(), closed_reason=reason,
                     closed_actor=actor, closed_request_id=request_id,
                     delivery=_delivery_debt(task_id, quiz_id, "closure"))
        if pending_publication and "publication_chat_id" not in block:
            block["delivery"]["publication_pending"] = True
        if actor == "model":
            block["closed_by_task_id"] = str(actor_task_id)
        outcome.update(ok=True, state=STATE_CLOSED, duplicate=False, block=dict(block))
        return True

    _mutate_projection(drive_root, task_id, mutate)
    return outcome


def question_task_owned_by(drive_root: Any, task_id: str, actor_task_id: str) -> bool:
    """Author authority through recorded Continue claims and host retry attempts."""
    from ouroboros.owner_continue import recorded_continuation
    from ouroboros.task_results import load_task_result, resolve_task_lineage

    current, seen = str(actor_task_id or ""), set()
    while current and current not in seen:
        seen.add(current)
        if current == task_id or recorded_continuation(drive_root, task_id, current):
            return True
        try:
            row = load_task_result(drive_root, current, strict=True) or {}
        except (OSError, ValueError):
            return False
        lineage = resolve_task_lineage(
            current, metadata=row.get("metadata"), root_task_id=row.get("root_task_id"),
            parent_task_id=row.get("parent_task_id"), delegation_role=row.get("delegation_role"),
            original_task_id=row.get("original_task_id"), timeout_retry_from=row.get("timeout_retry_from"),
        )
        if not lineage["is_retry_root_attempt"]:
            return False
        current = lineage["original_task_id"]
    return False


def record_answered(
    drive_root: Any, task_id: str, *,
    quiz_id: str, option_index: Optional[int], request_id: str, comment: str = "",
    allow_expired: bool = False,
    answer_source: str = "web",
) -> Dict[str, Any]:
    """Ingress-side answer write — request-id idempotent, first answer wins.

    ``option_index=None`` is the owner's OWN answer: none of the offered
    options was taken, so no ``answered_index`` is written at all (a stored 0
    would read as "chose the first option" on every later replay) and the
    verbatim ``comment`` carries the answer.

    ``allow_expired`` admits the LATE answer (В17a=A): a structurally expired
    card of a finished task is still answerable, and the accepted block carries
    ``answered_after_terminal`` so replay can tell it from an answer the asking
    task itself received. First-wins is untouched — an already ``answered``
    block stays a refusal on any other ``request_id``.

    The answer frame keeps a required wait distinct from proceeding under an
    assumption. Terminal reconciliation retires the live flag, so its matching
    wait-ended/reconciled stamps preserve that distinction for late answers.
    Every surface settles the card on ``answered`` regardless of wait facts.

    Returns ``{"ok", "state", "duplicate", "error", "block"}``:
    - unknown quiz_id → ``error="quiz_not_found"``;
    - open (or expired with ``allow_expired``) + valid index (or no index +
      comment) → answered (``ok=True``);
    - same ``request_id`` replay → the recorded confirmation, ``duplicate``;
    - already answered, or expired without ``allow_expired``, under a different
      ``request_id`` → refusal with the truthful current ``state``;
    - out-of-range index → ``error="option_out_of_range"``;
    - no index and no comment → ``error="answer_empty"`` (an answer that says
      nothing is not an answer).
    """
    stamp = utc_now_iso()
    outcome: Dict[str, Any] = {}

    def _mutator(quizzes: Dict[str, Dict[str, Any]]) -> Any:
        block = quizzes.get(str(quiz_id))
        if not isinstance(block, dict):
            outcome.update({"ok": False, "error": "quiz_not_found", "state": ""})
            return _KEEP
        state = str(block.get("state") or STATE_OPEN)
        if str(block.get("request_id") or "") and str(block.get("request_id")) == str(request_id or ""):
            outcome.update({"ok": True, "state": state, "duplicate": True, "block": dict(block)})
            return _KEEP
        late = allow_expired and state == STATE_EXPIRED_TERMINAL
        if state != STATE_OPEN and not late:
            outcome.update({"ok": False, "error": "quiz_closed", "state": state, "block": dict(block)})
            return _KEEP
        options = block.get("options") if isinstance(block.get("options"), list) else []
        if option_index is None:
            if not str(comment or "").strip():
                outcome.update({"ok": False, "error": "answer_empty", "state": state})
                return _KEEP
        elif not isinstance(option_index, int) or not (0 <= option_index < len(options)):
            outcome.update({"ok": False, "error": "option_out_of_range", "state": state})
            return _KEEP
        block.update({
            "state": STATE_ANSWERED, "answered_at": stamp,
            "request_id": str(request_id or ""),
            "answer_source": str(answer_source or "web"),
            "delivery": _delivery_debt(task_id, quiz_id, "answer"),
            # Audit only: the card was answered after its author finished.
            **({"answered_after_terminal": True} if late else {}),
            # No index key at all for an own answer — see the docstring.
            **({"answered_index": int(option_index)} if option_index is not None else {}),
            **({"comment": str(comment)} if str(comment or "").strip() else {}),
        })
        quizzes[str(quiz_id)] = block
        outcome.update({"ok": True, "state": STATE_ANSWERED, "duplicate": False, "block": dict(block)})
        return True

    _mutate_projection(drive_root, task_id, _mutator)
    return outcome


def mark_wait_ended(drive_root: Any, task_id: str, quiz_id: str) -> bool:
    """The wait behind an OPEN card ended and the task resumed: the block stops
    saying ``wait_for_answer`` (replay renders the truth) and keeps the instant for audit.
    The card stays open and answerable. Returns whether a block changed — and it
    changes ONLY an open, still-waiting block (F10): a card answered a second before
    the wake is never rolled back, and the callers announce nothing unless this is True."""
    changed: List[bool] = []

    def _mutator(quizzes: Dict[str, Dict[str, Any]]) -> Any:
        block = quizzes.get(str(quiz_id))
        if (not isinstance(block, dict) or block.get("state") != STATE_OPEN
                or not block.get("wait_for_answer")):
            return _KEEP
        block.pop("wait_for_answer", None)
        block["wait_ended_at"] = utc_now_iso()
        changed.append(True)
        return block

    _mutate_projection(drive_root, task_id, _mutator)
    return bool(changed)


def reconcile_terminal(drive_root: Any, task_id: str) -> List[str]:
    """Task-done reconciliation: every still-open quiz expires structurally.

    Returns the quiz ids that flipped to ``expired_terminal`` so the caller
    can emit their ``quiz_state`` frames. Never resurrects or rewrites an
    answered block."""
    stamp = utc_now_iso()
    expired: List[str] = []
    terminal_quizzes: List[str] = []

    def _mutator(quizzes: Dict[str, Dict[str, Any]]) -> Any:
        for key, block in quizzes.items():
            state = str(block.get("state") or STATE_OPEN)
            if state == STATE_OPEN:
                block.update({"state": STATE_EXPIRED_TERMINAL, "reconciled_at": stamp})
                if block.pop("wait_for_answer", None):
                    block.setdefault("wait_ended_at", stamp)
                expired.append(str(key))
                terminal_quizzes.append(str(key))
            elif state in (STATE_EXPIRED_TERMINAL, STATE_ANSWERED, STATE_CLOSED):
                # A previous call may have committed quiz expiry before the
                # paired task-result repair failed. Keep the second pass
                # idempotent; an accepted answer can also await worker capacity
                # when the task ends. Neither case rewrites the quiz's answer.
                terminal_quizzes.append(str(key))
        return True if expired else _KEEP

    _mutate_projection(drive_root, task_id, _mutator)
    if terminal_quizzes:
        from ouroboros.task_results import (
            require_writable_task_result_schema,
            stamp_task_result_schema,
        )

        def _close_owner_wait(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            require_writable_task_result_schema(current)
            wait = current.get("owner_wait")
            if not isinstance(wait, dict) or str(wait.get("state") or "") != "waiting":
                return None
            quiz_id = str(wait.get("quiz_id") or "")
            if quiz_id not in terminal_quizzes:
                return None
            updated = dict(current)
            updated["owner_wait"] = {
                **wait,
                "state": STATE_EXPIRED_TERMINAL,
                "reconciled_at": stamp,
            }
            return stamp_task_result_schema(updated)

        # The quiz and its waiting continuation share the same task-result
        # authority. Close the paired wait after the quiz projection so a
        # terminal task cannot replay as both expired and still waiting.
        patch_document(drive_root, str(task_id), _close_owner_wait)
    return expired


def quiz_states(drive_root: Any, task_id: str) -> Dict[str, Dict[str, Any]]:
    """Read-only view of the projection for history replay merge."""
    try:
        from ouroboros.task_store import read_document

        raw = read_document(drive_root, str(task_id))
    except Exception:
        return {}
    quizzes = raw.get("owner_quiz") if isinstance(raw, dict) else None
    if not isinstance(quizzes, dict):
        return {}
    return {str(k): dict(v) for k, v in quizzes.items() if isinstance(v, dict)}


def quiz_answer_frame(
    block: Dict[str, Any], option_index: Optional[int], comment: str,
) -> str:
    """Host-authored structural frame around the owner's VERBATIM choice.

    The ONE frame builder shared by the live mailbox control (the decision
    ingress) and the late-answer delivery (the drain / direct turn), so the
    model reads the same words whichever way the answer arrives.

    The asked/answered timestamps ride inside so the MODEL judges freshness
    itself (owner decision 30=A — no host staleness verdict). With no
    ``option_index`` the owner took none of the offered options and wrote
    their own answer — say exactly that, without calling it a rejection the
    owner never stated, so the model never reads the free answer as a gloss on
    a chosen option."""
    options = block.get("options") if isinstance(block.get("options"), list) else []
    lines = [
        f"[Owner quiz answer] quiz {block.get('quiz_id')} — asked {block.get('asked_at')}, "
        f"answered {block.get('answered_at')}.",
        f"Question was: {block.get('question')}",
    ]
    from ouroboros.delegate_reply import reply_instruction

    if block.get("reply_to"):
        lines.append(reply_instruction(block["reply_to"]))
    if option_index is None:
        lines.append(
            "The owner answered in their own words without choosing an offered "
            f"option. Verbatim: {comment}"
        )
    else:
        label = str(options[option_index]) if 0 <= option_index < len(options) else ""
        lines.append(f"The owner chose option {option_index + 1}: {label}")
        if comment:
            lines.append(f"Owner comment (verbatim): {comment}")
    ended_at_terminal = bool(block.get("wait_ended_at")) and (
        block.get("wait_ended_at") == block.get("reconciled_at")
    )
    if (str(block.get("assumption") or "") and not block.get("wait_for_answer")
            and not ended_at_terminal):
        lines.append(
            f"You continued under the assumption: {block.get('assumption')} — "
            "judge yourself whether work has moved past the answered fork."
        )
    from ouroboros.presence_owner_contact import answer_source_context
    return "\n".join(lines) + answer_source_context(block)


def recorded_answer_frame(block: Dict[str, Any]) -> str:
    """The frame of an ANSWERED block, read from what the projection recorded."""
    index = block.get("answered_index")
    return quiz_answer_frame(
        block, index if isinstance(index, int) and not isinstance(index, bool) else None,
        str(block.get("comment") or ""),
    )


def late_answer_owner_text(block: Dict[str, Any]) -> str:
    """What the HUMAN said on the card, for the owner's own chat row and bubble.

    The verbatim comment when the owner wrote one; otherwise the pressed option
    exactly as the button showed it, ``{n}. {label}``. Never the host frame:
    the bubble is the owner's words, the frame is for the model."""
    comment = str(block.get("comment") or "")
    if comment.strip():
        return comment
    options = block.get("options") if isinstance(block.get("options"), list) else []
    index = block.get("answered_index")
    if isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(options):
        return f"{index + 1}. {options[index]}"
    return ""


def late_answer_ref(value: Any) -> Optional[Dict[str, Any]]:
    """Typed late-answer provenance, preserving the host's accepted answer snapshot.

    The decision ingress supplies this snapshot from the canonical winning block;
    client answer fields never supply it. Mailbox and direct-turn continuations
    carry the same accepted-message authority after discharged question GC.
    """
    if not isinstance(value, dict):
        return None
    task_id = str(value.get("task_id") or "").strip()
    quiz_id = str(value.get("quiz_id") or "").strip()
    if not task_id or not quiz_id:
        return None
    ref: Dict[str, Any] = {"task_id": task_id, "quiz_id": quiz_id}
    snapshot = value.get("answer_snapshot")
    block = snapshot.get("quiz") if isinstance(snapshot, dict) else None
    if (isinstance(block, dict) and snapshot.get("task_id") == task_id
            and block.get("quiz_id") == quiz_id and block.get("state") == STATE_ANSWERED
            and (block.get("answered_after_terminal") is True or block.get("delivery_after_terminal") is True
                 or (block.get("delivery") or {}).get("route") == "fresh_head")):
        ref["answer_snapshot"] = {"task_id": task_id, "quiz": dict(block)}
    return ref


def late_answer_model_text(drive_root: Any, late_answer: Any, owner_text: str) -> str:
    """The model-facing delivery of a LATE quiz answer (owner message ``owner_text``).

    The owner's chat row carries only their own words; the receiving model
    needs the card they answered. Its first line names the asking task, which
    had finished (a late answer is only accepted after it did; TZ-2 B3); the
    frame uses the accepted message's matching host-written snapshot, including
    its delivery-time source-work facts. Legacy messages without that snapshot
    read the canonical block on ``drive_root``.
    When neither authority can be read, the owner's words are delivered with
    one host line saying so — disclosed, never a silent loss of the card's
    context. A message with no valid ``late_answer`` provenance is returned
    unchanged."""
    ref = late_answer_ref(late_answer)
    if ref is None:
        return owner_text
    snapshot = ref.get("answer_snapshot")
    # The accepted message freezes the answer AND delivery-time source-work
    # facts. The older ask block cannot replace its later promoted-work link.
    block = dict(snapshot["quiz"]) if isinstance(snapshot, dict) else {}
    if not block:
        try:
            block = quiz_states(drive_root, ref["task_id"]).get(ref["quiz_id"]) or {}
        except Exception:
            # Legacy intake still delivers the owner's words with the gap below.
            block = {}
    state = "which had finished" if block.get("answered_after_terminal") or block.get("delivery_after_terminal") else "which is not reading"
    head = f"[Late answer to a question asked by task {ref['task_id']}, {state}]\n"
    if isinstance(block, dict) and str(block.get("state") or "") == STATE_ANSWERED:
        return head + recorded_answer_frame(block)
    return (
        f"{head}{owner_text}\n"
        f"[Host note] This owner message answers quiz {ref['quiz_id']}; that card "
        "could not be read, so only the owner's own words are shown."
    )
