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
            "assumption", "state": open|answered|expired_terminal,
            "asked_at", "answered_at"?, "answered_index"?, "request_id"?,
            "comment"?, "reconciled_at"?,
        }, ...
    }

Structural expiry only (owner decision 30=A): a quiz dies with its author —
``reconcile_terminal`` runs on the task-done seam; there is no host TTL.
The writer mutates ONLY the ``owner_quiz`` key via ``update_json_locked``
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
            # block would resurrect as an "Awaiting answer" card on replay
            # (the chat row froze state=open) whose click then 404s.
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
) -> Dict[str, Any]:
    """Worker-side projection write at ask time.

    The stored option labels are the ingress's validation authority: an
    ``option_index`` outside this list is refused, and the answer echoes the
    verbatim label back to the asking task."""
    stamp = utc_now_iso()
    block = {
        "quiz_id": str(quiz_id), "question": str(question or ""),
        "options": [str(label) for label in options],
        "stake": str(stake or ""), "assumption": str(assumption or ""),
        "state": STATE_OPEN, "asked_at": stamp,
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
) -> Dict[str, Any]:
    """Ingress-side answer write — request-id idempotent, first answer wins.

    ``option_index=None`` is the owner's OWN answer: none of the offered
    options was taken, so no ``answered_index`` is written at all (a stored 0
    would read as "chose the first option" on every later replay) and the
    verbatim ``comment`` carries the answer.

    Returns ``{"ok", "state", "duplicate", "error", "block"}``:
    - unknown quiz_id → ``error="quiz_not_found"``;
    - open + valid index (or no index + comment) → answered (``ok=True``);
    - same ``request_id`` replay → the recorded confirmation, ``duplicate``;
    - already answered/expired with a different ``request_id`` → refusal with
      the truthful current ``state`` (the card settles, never re-invites);
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
        if state != STATE_OPEN:
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
            # No index key at all for an own answer — see the docstring.
            **({"answered_index": int(option_index)} if option_index is not None else {}),
            **({"comment": str(comment)} if str(comment or "").strip() else {}),
        })
        quizzes[str(quiz_id)] = block
        outcome.update({"ok": True, "state": STATE_ANSWERED, "duplicate": False, "block": dict(block)})
        return True

    _mutate_projection(drive_root, task_id, _mutator)
    return outcome


def reconcile_terminal(drive_root: Any, task_id: str) -> List[str]:
    """Task-done reconciliation: every still-open quiz expires structurally.

    Returns the quiz ids that flipped to ``expired_terminal`` so the caller
    can emit their ``quiz_state`` frames. Never resurrects or rewrites an
    answered block."""
    stamp = utc_now_iso()
    expired: List[str] = []

    def _mutator(quizzes: Dict[str, Dict[str, Any]]) -> Any:
        for key, block in quizzes.items():
            if str(block.get("state") or STATE_OPEN) == STATE_OPEN:
                block.update({"state": STATE_EXPIRED_TERMINAL, "reconciled_at": stamp})
                expired.append(str(key))
        return True if expired else _KEEP

    _mutate_projection(drive_root, task_id, _mutator)
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
