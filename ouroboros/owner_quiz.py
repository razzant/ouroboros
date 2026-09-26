"""Durable owner-quiz lifecycle projection (#Q-2b, the answer half of the
escalation channel).

This module owns everything quiz-lifecycle-specific — the asked/answered
projection in the task result, the request-id idempotent answer write, and
the terminal reconciliation — so the pinned surfaces (the escalate tool, the
decision ingress, ``supervisor/queue_transitions``) keep thin dispatch only,
mirroring ``ouroboros/owner_hurry.py``.

Shape inside ``task_results/<task_id>.json`` (canonical drive root, exactly
like the hurry projection):

    "owner_quiz": {
        "<quiz_id>": {
            "quiz_id", "question", "options": [label, ...], "stake",
            "option_details"?: [detail, ...], "recommended_index"?: int,
            "assumption", "state": open|answered|expired_terminal,
            "asked_at", "answered_at"?, "answered_index"?, "request_id"?,
            "comment"?, "reconciled_at"?, "chat_id"?, "max_wait_minutes"?,
            "answered_after_terminal"?, "host_facts"?,
        }, ...
    }

Expiry stays STRUCTURAL: ``reconcile_terminal`` runs on the task-done seam and
flips every still-open block to ``expired_terminal``; there is no host TTL. What
changed (owner decision В17a=A, which explicitly retired 30=A's "a quiz dies
with its author") is the fate of a LATE answer: the ingress may still record it
on an expired block (``record_answered(allow_expired=True)``), first answer
still wins, and the block keeps ``answered_after_terminal`` for audit. The
stored ``chat_id`` is the card's own chat, so that answer can be delivered as an
ordinary owner message once its author is gone.
The writers mutate ``owner_quiz`` and its paired terminal ``owner_wait`` via ``update_json_locked``
(never ``write_task_result`` — its status-regression guard can drop the
write), so concurrent terminal writers merge around it.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from ouroboros.utils import update_json_locked, utc_now_iso

STATE_OPEN = "open"
STATE_ANSWERED = "answered"
STATE_EXPIRED_TERMINAL = "expired_terminal"

# At most this many quiz blocks are retained per task (oldest evicted first);
# a task asking more than this many questions keeps the recent ones live.
_QUIZ_CAP = 16


class _Keep:
    pass


_KEEP = _Keep()


def _quiz_result_path(drive_root: Any, task_id: str, *, create: bool = True):
    from ouroboros.task_results import task_result_path

    return task_result_path(drive_root, str(task_id), create=create)


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
            # Evict CLOSED blocks first (oldest asked_at): an evicted OPEN
            # block would resurrect as an unanswered card on replay (the chat
            # row froze state=open) whose click then 404s.
            def _eviction_key(key: str):
                block = quizzes[key]
                closed = str(block.get("state") or STATE_OPEN) != STATE_OPEN
                return (0 if closed else 1, str(block.get("asked_at") or ""))

            for key in sorted(quizzes, key=_eviction_key)[:-_QUIZ_CAP]:
                quizzes.pop(key, None)
        updated = dict(current)
        updated["owner_quiz"] = quizzes
        view.update(quizzes)
        return stamp_task_result_schema(updated)

    update_json_locked(_quiz_result_path(drive_root, task_id), _mutate)
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
) -> Dict[str, Any]:
    """Worker-side projection write at ask time.

    The stored option labels are the ingress's validation authority: an
    ``option_index`` outside this list is refused, and the answer echoes the
    verbatim label back to the asking task.

    ``chat_id`` is the card's own chat, recorded by the asker (chat 0 is the
    real hidden partition, never "no chat"): a late answer arriving after the
    task is gone is delivered there as an ordinary owner message instead of
    into a mailbox nobody drains. ``max_wait_minutes`` records the bound a
    waiting asker chose, so replay can say what the task waited for.
    ``host_facts`` is the host-written sentence the card shows under the
    question (asking task, how its run started, the owner's last message in
    the chat); stored only when non-empty."""
    if option_details is not None and (
        not isinstance(option_details, list) or len(option_details) != len(options)
        or not all(isinstance(value, str) for value in option_details)
    ):
        raise ValueError("option_details must preserve the labels' length and order")
    stamp = utc_now_iso()
    block = {
        "quiz_id": str(quiz_id), "question": str(question or ""),
        "options": [str(label) for label in options],
        **({"option_details": list(option_details)} if option_details is not None else {}),
        **({"recommended_index": int(recommended_index)} if isinstance(recommended_index, int) else {}),
        "stake": str(stake or ""), "assumption": str(assumption or ""),
        "state": STATE_OPEN, "asked_at": stamp,
        **({"wait_for_answer": True} if wait_for_answer else {}),
        **({"chat_id": int(chat_id)} if isinstance(chat_id, int) and not isinstance(chat_id, bool) else {}),
        **({"max_wait_minutes": int(max_wait_minutes)}
           if isinstance(max_wait_minutes, int) and not isinstance(max_wait_minutes, bool) else {}),
        **({"host_facts": str(host_facts)} if str(host_facts or "") else {}),
    }

    refused: Dict[str, str] = {}

    def _mutator(quizzes: Dict[str, Dict[str, Any]]) -> Any:
        if str(quiz_id) in quizzes:
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

    view = _mutate_projection(drive_root, task_id, _mutator)
    if refused:
        return {"refused": refused["reason"]}
    return dict(view.get(str(quiz_id)) or block)


def record_answered(
    drive_root: Any, task_id: str, *,
    quiz_id: str, option_index: Optional[int], request_id: str, comment: str = "",
    allow_expired: bool = False,
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

    ``wait_for_answer`` stays on the answered block on purpose: the answer
    frame (``gateway.task_decision._quiz_answer_frame``) reads it to say the
    task was waiting rather than proceeding under its assumption, and every
    surface settles the card on ``answered`` regardless of wait facts.

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
                expired.append(str(key))
                terminal_quizzes.append(str(key))
            elif state in (STATE_EXPIRED_TERMINAL, STATE_ANSWERED):
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
        update_json_locked(
            _quiz_result_path(drive_root, task_id),
            _close_owner_wait,
        )
    return expired


def quiz_states(drive_root: Any, task_id: str) -> Dict[str, Dict[str, Any]]:
    """Read-only view of the projection for history replay merge."""
    try:
        import json

        # create=False: a read-only replay view must not mkdir as a side effect.
        raw = json.loads(_quiz_result_path(drive_root, task_id, create=False).read_text(encoding="utf-8"))
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
    if str(block.get("assumption") or "") and not block.get("wait_for_answer"):
        lines.append(
            f"You continued under the assumption: {block.get('assumption')} — "
            "judge yourself whether work has moved past the answered fork."
        )
    return "\n".join(lines)


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


def late_answer_ref(value: Any) -> Optional[Dict[str, str]]:
    """The typed ``late_answer`` provenance ``{task_id, quiz_id}``, or None."""
    if not isinstance(value, dict):
        return None
    task_id = str(value.get("task_id") or "").strip()
    quiz_id = str(value.get("quiz_id") or "").strip()
    if not task_id or not quiz_id:
        return None
    return {"task_id": task_id, "quiz_id": quiz_id}


def late_answer_model_text(drive_root: Any, late_answer: Any, owner_text: str) -> str:
    """The model-facing delivery of a LATE quiz answer (owner message ``owner_text``).

    The owner's chat row carries only their own words; the receiving model
    needs the card they answered. Its first line names the asking task, which
    had finished (a late answer is only accepted after it did; TZ-2 B3); the
    frame is rebuilt from the stored block on the canonical ``drive_root`` (the
    same projection the ingress recorded). When that block cannot be read
    (evicted, unanswered, or unreadable), the owner's words are delivered with
    one host line saying so — disclosed, never a silent loss of the card's
    context. A message with no valid ``late_answer`` provenance is returned
    unchanged."""
    ref = late_answer_ref(late_answer)
    if ref is None:
        return owner_text
    block: Dict[str, Any] = {}
    try:
        block = quiz_states(drive_root, ref["task_id"]).get(ref["quiz_id"]) or {}
    except Exception:
        block = {}
    head = f"[Late answer to a question asked by task {ref['task_id']}, which had finished]\n"
    if isinstance(block, dict) and str(block.get("state") or "") == STATE_ANSWERED:
        return head + recorded_answer_frame(block)
    return (
        f"{head}{owner_text}\n"
        f"[Host note] This owner message answers quiz {ref['quiz_id']}; that card "
        "could not be read, so only the owner's own words are shown."
    )
