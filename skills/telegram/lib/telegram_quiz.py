"""Answering an Ouroboros quiz card from Telegram (#472).

The host's ``chat.quiz`` event carries the card identity (``task_id``,
``quiz_id``). The owner's button tap or reply is relayed to the SAME decision
ingress the web card uses — Host Service ``POST /chat/decision`` →
``task_decision.answer_decision`` — so the answer is idempotent per
``request_id`` (``tg:<update_id>``), first answer wins, and a late answer is
accepted exactly as it is for the browser card. The only state kept here maps a
short callback token and the sent message to that identity: Telegram caps
``callback_data`` at 64 bytes, too short for the ids themselves. A card also
remembers its settled lifecycle state and the host's recorded answer, so the
host's ``chat.quiz_state`` facts edit it forward only. Nothing here parses the
owner's words; a reply is delivered verbatim as their own answer.

The card is the answer's receipt. It shows what the host recorded, never what this
bridge sent: the winning option and/or the owner's words, and under them how the
host's delivery stands, in the web card's words (``question_presentation.js``). A
saved, pending or unknown delivery is one status line, a confirmed one none; no line
claims the task read the answer. A tap is also told by a toast. A tap or a text answer gets
a separate message only for a problem: a conflict, a closed question, an error, an
unconfirmed answer or a card that could not be updated (owner's choice, 27 Oct: A).

The card carries the whole authored text: the project it belongs to, the host's
facts about the asking task, the question, the stake, and every option with its
detail. A card longer than one Telegram message is never cut: the explanation
goes out first as ordered plain messages through the client's existing chunker,
and a compact message with the project line, the numbered option labels, the
hint and the keyboard follows last. Only that last message is remembered, so a
tap, a reply to it and the answered-edit keep working exactly as for a short
card; a reply to an earlier explanation part is an ordinary owner message
carrying the bridge's host-row reference. An open question (no
options) is the same card without a keyboard; its hint asks for a reply.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import weakref
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Tuple

from .telegram_api import _TELEGRAM_TEXT_LIMIT, _u16len
from .telegram_i18n import Index
from .telegram_state import _read_json_file, _state_file

_QUIZ_STATE_FILE = "quiz_state.json"
_MAX_REMEMBERED = 50
_CALLBACK_PREFIX = "qz:"
_BUTTON_LABEL_MAX = 40
_ANSWER_ECHO_MAX = 200
# The answered edit appends "Answered: <echo>" to the remembered single message; a card is sent
# as ONE message only when that edit still fits Telegram's limit (echo + label + prefix + newline).
_ANSWERED_EDIT_RESERVE = _ANSWER_ECHO_MAX + 32

HostPost = Callable[[Any, str, Dict[str, Any]], Awaitable[Tuple[int, Dict[str, Any]]]]

_TEXTS_EN = {
    "hint": "Tap an option, or reply to this message with your own answer.",
    "hint_open": "Reply to this message with your answer.",
    # The host's delivery of an accepted answer, as the web card says it; never a read receipt.
    "status_saved": "Answer saved · the task reads it when it continues",
    "status_pending": "Answer recorded · delivery pending",
    "status_unknown": "Answer recorded · delivery unknown",
    "receipt_delivered": "Answer delivered",
    "receipt_recorded": "Answer recorded",
    "late": "The task had already finished.",
    "lost": "Not accepted: this question was already answered. Recorded answer: {answer}",
    "not_updated": "The question card above could not be updated.",
    "unconfirmed": "The answer was not confirmed (HTTP {status}), so it may not be recorded. "
                   "Send the same answer again later.",
    "already": "This question was already answered.",
    "expired": "This question has expired — the task moved on.",
    "gone": "This question is no longer known to Ouroboros.",
    "failed": "Could not deliver the answer (HTTP {status}). Try again.",
    "answered_line": "Answered: {answer}",
    "answered_plain": "Answered.",
    "resumed": "The task continued; an answer is still accepted.",
    "expired_terminal": "The task finished; a late answer is accepted as your message.",
    "superseded": "Replaced by a newer question.",
    "closed": "Question closed.",
    "project": "Project",
    "question": "Question",
    "stake": "At stake",
    "meanwhile": "Continuing meanwhile",
    "waiting": "Waiting for your answer; Stop and the task deadline still apply.",
}
_TEXTS = Index("quiz", _TEXTS_EN, "the question cards sent to the owner's phone: hints, outcomes, lifecycle lines")

# A remembered card's lifecycle only moves forward, as on the web card: a closed
# wait or an expiry never reopens a settled card, and nothing downgrades an answer.
# An answer and a closure are both final, and neither replaces the other: closure is never an answer.
_LIFECYCLE_RANK = {"open": 0, "expired_terminal": 1, "superseded": 2, "answered": 3, "closed": 3}
_FINAL_STATES = ("answered", "closed")
# The host's facts about an answer that a card remembers beside its state.
_ANSWER_FACTS = ("answered_index", "comment", "question_delivery")
_TOAST_MAX = 200  # Telegram's callback answer limit

# The host publishes ``chat.quiz`` and ``chat.quiz_state`` as independent
# coroutines on the extension loop, so a fact could outrun the card's own send
# (nothing remembered yet) or two edits could land out of order. One lock per
# card on the running loop orders creation and every edit, so each edit re-reads
# the stored state before it is sent; a fact that finds no card yet is retained
# and applied right after the card is remembered. Locks live per loop (weakly),
# so a closed loop takes its locks with it.
_CARD_LOCKS: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, Dict[str, asyncio.Lock]]" = (
    weakref.WeakKeyDictionary())
_RETAINED_FACTS: Dict[str, Dict[str, Any]] = {}


def _texts(lang: str) -> Mapping[str, str]:
    return _TEXTS[lang]


def hint(lang: str) -> str:
    return _texts(lang)["hint"]


def hint_open(lang: str) -> str:
    return _texts(lang)["hint_open"]


def mint_token(task_id: str, quiz_id: str) -> str:
    """Short stable token for ``callback_data`` (Telegram's 64-byte cap)."""
    return hashlib.sha256(f"{task_id}:{quiz_id}".encode("utf-8")).hexdigest()[:12]


def card_options(raw_options: Any, *, limit: int) -> Tuple[List[str], List[str], Optional[int]]:
    """Labels, details and the recommended index of the event's options."""
    labels: List[str] = []
    details: List[str] = []
    recommended: Optional[int] = None
    for option in raw_options if isinstance(raw_options, list) else []:
        label = str(option.get("label") or "").strip() if isinstance(option, dict) else ""
        if not label or len(labels) >= limit:
            continue
        if option.get("recommended") is True and recommended is None:
            recommended = len(labels)
        labels.append(label)
        details.append(str(option.get("detail") or "").strip())
    return labels, details, recommended


def button_labels(labels: List[str], recommended_index: Optional[int]) -> List[str]:
    """Button captions: the recommended option carries a leading star."""
    return [f"★ {label}" if index == recommended_index else label
            for index, label in enumerate(labels)]


def quiz_keyboard(token: str, labels: List[str]) -> List[List[dict]]:
    """One button row per option; ``callback_data`` = ``qz:<token>:<index>``."""
    return [
        [{"text": f"{index}. {label}"[:_BUTTON_LABEL_MAX],
          "callback_data": f"{_CALLBACK_PREFIX}{token}:{index - 1}"}]
        for index, label in enumerate(labels, 1)
    ]


def _project_line(project_name: str, lang: str) -> List[str]:
    name = str(project_name or "").strip()
    return [f"{_texts(lang)['project']}: {name}"] if name else []


def _option_line(index: int, label: str, *, detail: str = "", recommended: bool = False) -> str:
    line = f"{index}. {'★ ' if recommended else ''}{label}"
    return f"{line} — {detail}" if detail else line


def render_quiz_text(question: str, labels: List[str], stake: str, assumption: str,
                     *, wait_for_answer: bool = False, project_name: str = "",
                     option_details: Optional[List[str]] = None,
                     recommended_index: Optional[int] = None,
                     host_facts: str = "", lang: str = "en") -> str:
    """The whole card body, never shortened: every authored field is kept."""
    texts = _texts(lang)
    details = list(option_details or [])
    lines = _project_line(project_name, lang)
    if host_facts:
        lines.append(host_facts)  # ids, times and counts of one question: a sentence, not a translation key
    lines.append(f"{texts['question']}: {question}")
    if stake:
        lines.append(f"{texts['stake']}: {stake}")
    lines.extend(
        _option_line(index, label,
                     detail=str(details[index - 1] or "") if index - 1 < len(details) else "",
                     recommended=recommended_index == index - 1)
        for index, label in enumerate(labels, 1)
    )
    if wait_for_answer:
        lines.append(texts["waiting"])
    elif assumption:
        lines.append(f"{texts['meanwhile']}: {assumption}")
    return "\n".join(lines)


def render_compact_text(labels: List[str], *, project_name: str = "",
                        recommended_index: Optional[int] = None, lang: str = "en") -> str:
    """The keyboard message of an overflowing card: project and numbered labels."""
    lines = _project_line(project_name, lang)
    lines.extend(_option_line(index, label, recommended=recommended_index == index - 1)
                 for index, label in enumerate(labels, 1))
    return "\n".join(lines)


def answered_edit_reserve(lang: str) -> int:
    """The room a one-message card keeps for its answered edit, in UTF-16 units: the English
    reserve plus however much longer this language makes the edit's own lines (the answered
    label and the longest delivery line), so a translation never pushes the edit past the limit."""
    def chrome(texts: Mapping[str, str]) -> int:
        statuses = (texts[key] for key in ("status_saved", "status_pending", "status_unknown"))
        return _u16len("\n" + texts.format("answered_line", answer="")) + 1 + max(map(_u16len, statuses))
    return _ANSWERED_EDIT_RESERVE + max(0, chrome(_texts(lang)) - chrome(_texts("")))


async def send_quiz_card(client, chat_id: int, *, body: str, compact: str, hint_text: str,
                         keyboard: List[List[dict]], reserve: int = _ANSWERED_EDIT_RESERVE) -> Tuple[int, bool]:
    """Send the card; return the keyboard message id and whether it overflowed.

    A card that fits Telegram's per-message limit (UTF-16 units) WITH room for
    its later answered edit (``answered_edit_reserve``) is one message with the
    keyboard. A longer one is sent as ordered plain parts through the client's
    chunker, then the compact keyboard message; nothing authored is truncated.
    """
    async def send(text: str) -> int:
        # An open question (no options) carries no keyboard: a plain message
        # the owner replies to, remembered exactly like a keyboard message.
        if keyboard:
            return int(await client.send_message_with_inline_keyboard(
                chat_id, text, keyboard, parse_mode="") or 0)
        return int(await client.send_message(chat_id, text, parse_mode="") or 0)

    full = f"{body}\n{hint_text}"
    if _u16len(full) + reserve <= _TELEGRAM_TEXT_LIMIT:
        return await send(full), False
    await client.send_message(chat_id, body, parse_mode="")
    return await send(f"{compact}\n{hint_text}"), True


def _load(api) -> Dict[str, Any]:
    data = _read_json_file(_state_file(api, _QUIZ_STATE_FILE))
    quizzes = data.get("quizzes") if isinstance(data, dict) else None
    return {"quizzes": dict(quizzes) if isinstance(quizzes, dict) else {}}


def remember_quiz(api, token: str, record: Dict[str, Any]) -> None:
    """Bounded token → card mapping (the newest ``_MAX_REMEMBERED`` cards)."""
    data = _load(api)
    quizzes = data["quizzes"]
    quizzes.pop(token, None)
    quizzes[token] = dict(record)
    for stale in list(quizzes)[:-_MAX_REMEMBERED]:
        quizzes.pop(stale, None)
    path = _state_file(api, _QUIZ_STATE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)


def quiz_for_token(api, token: str) -> Optional[Dict[str, Any]]:
    record = _load(api)["quizzes"].get(str(token or ""))
    return dict(record) if isinstance(record, dict) else None


def remember_state(api, token: str, record: Dict[str, Any], state: str,
                   facts: Optional[Dict[str, Any]] = None) -> None:
    """Persist a settled lifecycle state on the card: the no-rollback evidence. An answer
    also keeps the host's recorded facts (``merge_answer_facts``) for later partial facts."""
    if not _LIFECYCLE_RANK.get(state, 0):
        return
    updated = {key: value for key, value in record.items() if facts is None or key not in _ANSWER_FACTS}
    updated.update(state=state, **(facts or {}))
    if updated != record:
        remember_quiz(api, token, updated)


def quiz_for_message(api, chat_id: int, message_id: int) -> Optional[Dict[str, Any]]:
    """The card sent as ``message_id`` in ``chat_id`` (for reply-to answers)."""
    if not message_id:
        return None
    for record in _load(api)["quizzes"].values():
        if (isinstance(record, dict)
                and int(record.get("chat_id") or 0) == int(chat_id)
                and int(record.get("message_id") or 0) == int(message_id)):
            return dict(record)
    return None


async def _deliver(
    api, post: HostPost, record: Dict[str, Any], *,
    option_index: Optional[int], comment: str, update_id: int,
) -> Tuple[int, Dict[str, Any]]:
    body: Dict[str, Any] = {
        "request_id": f"tg:{int(update_id)}",
        "decision_id": f"quiz:{record.get('task_id')}:{record.get('quiz_id')}",
    }
    if option_index is not None:
        body["option_index"] = int(option_index)
    if comment:
        body["comment"] = comment
    return await post(api, "/chat/decision", body)


def _winner(fact: Mapping[str, Any]) -> Dict[str, Any]:
    """The recorded answer a host fact names: its option index and/or the owner's own words."""
    index, comment = fact.get("answered_index"), fact.get("comment")
    return {**({"answered_index": index} if isinstance(index, int) and not isinstance(index, bool) else {}),
            **({"comment": comment} if isinstance(comment, str) and comment else {})}


def _delivery(fact: Mapping[str, Any]) -> Dict[str, Any]:
    """The host's projected answer delivery (``owner_quiz.question_delivery_projection``)."""
    view = fact.get("question_delivery")
    if not isinstance(view, dict) or view.get("kind") != "answer" or view.get("state") not in ("owed", "delivered"):
        return {}
    return {key: view[key] for key in ("kind", "state", "reason_code", "dismissed") if key in view}


def merge_answer_facts(held: Mapping[str, Any], fact: Mapping[str, Any]) -> Dict[str, Any]:
    """One answered fact over what is already known of the card: its winner and delivery.

    Host facts arrive in any order and some name only a part: a read acknowledgement carries
    the delivery alone, and absence never clears a winner. A fact naming any winner field
    names the whole winner (first answer wins, so it never changes) and is never mixed with
    another fact's fields or with this bridge's own attempt. Delivery moves forward only: an
    older ``owed`` snapshot never takes back a ``delivered`` one.
    """
    facts = _winner(fact) or _winner(held)
    old, new = _delivery(held), _delivery(fact)
    delivery = old if not new or (old.get("state") == "delivered" and new.get("state") != "delivered") else new
    return {**facts, **({"question_delivery": delivery} if delivery else {})}


def _status_line(facts: Mapping[str, Any], lang: str) -> str:
    """The card's delivery line: saved, pending or unknown while owed (dismissed included:
    removal from current questions delivers nothing), none once delivered or unknown to the host."""
    delivery = _delivery(facts)
    if delivery.get("state") != "owed":
        return ""
    reason = delivery.get("reason_code")
    key = ("status_saved" if reason == "awaiting_recipient_read"
           else "status_unknown" if reason == "late_answer_delivery_unknown" else "status_pending")
    return _texts(lang)[key]


def _receipt(facts: Mapping[str, Any], late: bool, lang: str) -> str:
    """One sentence on an accepted answer from the host's facts alone: the owed status, a
    confirmed delivery, or only the record. ``forwarded`` proves neither a reader nor a place."""
    texts = _texts(lang)
    state = _delivery(facts).get("state")
    line = _status_line(facts, lang) or texts["receipt_delivered" if state == "delivered" else "receipt_recorded"]
    return f"{line}." + (f" {texts['late']}" if late else "")


def _echo(answer: str) -> str:
    return answer if len(answer) <= _ANSWER_ECHO_MAX else answer[:_ANSWER_ECHO_MAX] + "…"


def _clip_u16(text: str, units: int) -> str:
    """``text`` within ``units`` UTF-16 units, an ellipsis marking a cut."""
    if _u16len(text) <= units:
        return text
    budget, kept = units - 1, []
    for char in text:
        budget -= _u16len(char)
        if budget < 0:
            break
        kept.append(char)
    return "".join(kept) + "…"


def _answer_echo(record: Mapping[str, Any], facts: Mapping[str, Any]) -> str:
    options = list(record.get("options") or [])
    index, parts = facts.get("answered_index"), []
    if index is not None:
        parts.append(f"{index + 1}. {options[index]}" if 0 <= index < len(options) else f"{index + 1}.")
    if facts.get("comment"):
        parts.append(_echo(str(facts["comment"])))
    return " — ".join(parts)


def _answered_text(record: Mapping[str, Any], facts: Mapping[str, Any], lang: str) -> str:
    """The settled card: its text, the recorded answer and the delivery line. The answer's
    echo gives way first, so the edit always fits one Telegram message."""
    texts = _texts(lang)
    base = str(record.get("text") or "")
    status = _status_line(facts, lang)
    tail = f"\n{status}" if status else ""
    answer = _answer_echo(record, facts)
    if not answer:
        return f"{base}\n{texts['answered_plain']}{tail}"
    room = _TELEGRAM_TEXT_LIMIT - _u16len(f"{base}\n{texts.format('answered_line', answer='')}{tail}")
    return f"{base}\n{texts.format('answered_line', answer=_clip_u16(answer, max(1, room)))}{tail}"


def lifecycle_edit(
    record: Dict[str, Any], event: Dict[str, Any], lang: str,
) -> Optional[Tuple[str, List[List[dict]]]]:
    """The (text, keyboard) edit a host ``chat.quiz_state`` fact asks of a sent card.

    ``None`` when the fact changes nothing here: a state this card cannot show, an
    ``open`` that does not close a wait, a fact older than the card's own state, or
    the other final state.
    The card mirrors the web one (``web/modules/question_presentation.js``): an
    answer settles it on the recorded option and/or the owner's own words and the
    host's delivery line, keeping what an earlier fact recorded (``merge_answer_facts``);
    a closed wait drops the waiting line; an expired card stays answerable, because a late
    answer is still accepted as the owner's message (В17a=A); a superseded card is
    a read-only record, and so is a closed question, which states its closure and the
    stated reason, never an answer. A final state is always re-applied — the edit is idempotent.
    """
    state = str(event.get("state") or "")
    held = str(record.get("state") or "open")
    if state not in _LIFECYCLE_RANK or (state == "open" and event.get("wait_for_answer") is not False):
        return None
    if _LIFECYCLE_RANK[state] < _LIFECYCLE_RANK.get(held, 0) or (held in _FINAL_STATES and state != held):
        return None
    texts = _texts(lang)
    base = str(record.get("text") or "")
    if state == "closed":
        reason = str(event.get("closed_reason") or "").strip()
        return f"{base}\n{texts['closed']}" + (f"\n{_echo(reason)}" if reason else ""), []
    if state == "answered":
        return _answered_text(record, merge_answer_facts(record, event), lang), []
    if state == "superseded":
        return f"{base}\n{texts['superseded']}", []
    status = texts["resumed" if state == "open" else "expired_terminal"]
    labels = [str(label) for label in record.get("options") or []]
    if not labels:
        return f"{base}\n{status}\n{texts['hint_open']}", []
    token = mint_token(str(record.get("task_id") or ""), str(record.get("quiz_id") or ""))
    return f"{base}\n{status}\n{texts['hint']}", quiz_keyboard(token, labels)


def lifecycle_target(
    api, event: Dict[str, Any], lang: str,
) -> Optional[Tuple[int, int, str, List[List[dict]]]]:
    """``(chat_id, message_id, text, keyboard)`` for a card sent here, else ``None``.

    A card never sent to Telegram has nothing to edit. A settled state is
    remembered before the edit: it is the host's fact, whatever the edit does.
    """
    task_id = str(event.get("task_id") or "").strip()
    quiz_id = str(event.get("quiz_id") or "").strip()
    token = mint_token(task_id, quiz_id)
    record = quiz_for_token(api, token) if task_id and quiz_id else None
    message_id = int((record or {}).get("message_id") or 0)
    edit = lifecycle_edit(record, event, lang) if record and message_id else None
    if record is None or edit is None:
        return None
    state = str(event.get("state") or "")
    remember_state(api, token, record, state, merge_answer_facts(record, event) if state == "answered" else None)
    return int(record.get("chat_id") or 0), message_id, edit[0], edit[1]


def card_lock(token: str) -> asyncio.Lock:
    """The running loop's lock for one card (bounded like the remembered cards)."""
    locks = _CARD_LOCKS.setdefault(asyncio.get_running_loop(), {})
    lock = locks.get(token)
    if lock is None:
        idle = [key for key, held in locks.items() if not held.locked()]
        for stale in idle[:max(0, len(locks) - _MAX_REMEMBERED)]:
            locks.pop(stale, None)
        lock = locks[token] = asyncio.Lock()
    return lock


def _retain_fact(token: str, event: Dict[str, Any]) -> None:
    """Keep the highest-ranked fact that arrived before its card was remembered; two answered
    facts merge, so a later delivery-only fact never drops the winner the first one named."""
    held = _RETAINED_FACTS.get(token) or {}
    if _LIFECYCLE_RANK.get(str(event.get("state") or ""), 0) >= _LIFECYCLE_RANK.get(str(held.get("state") or ""), 0):
        fact = dict(event)
        if fact.get("state") == "answered" and held.get("state") == "answered":
            fact = {key: value for key, value in fact.items() if key not in _ANSWER_FACTS}
            fact.update(merge_answer_facts(held, event))
        _RETAINED_FACTS.pop(token, None)
        _RETAINED_FACTS[token] = fact
    for stale in list(_RETAINED_FACTS)[:-_MAX_REMEMBERED]:
        _RETAINED_FACTS.pop(stale, None)


async def _apply_fact(api, token: str, event: Dict[str, Any], lang: str, client_factory) -> None:
    if quiz_for_token(api, token) is None:
        _retain_fact(token, event)  # the card's send may still be in flight
        return
    target = lifecycle_target(api, event, lang)
    if target is None:
        return
    if not await client_factory().edit_message_text_with_inline_keyboard(*target, parse_mode=""):
        api.log("warning", f"Telegram quiz card edit failed ({event.get('state')}).")  # never retried


async def follow_lifecycle(api, event: Dict[str, Any], lang: str, *, client_factory) -> None:
    """Apply one ``chat.quiz_state`` fact under its card's lock (see ``_CARD_LOCKS``)."""
    task_id = str(event.get("task_id") or "").strip()
    quiz_id = str(event.get("quiz_id") or "").strip()
    if not task_id or not quiz_id:
        return
    token = mint_token(task_id, quiz_id)
    async with card_lock(token):
        await _apply_fact(api, token, event, lang, client_factory)


async def apply_retained_fact(api, token: str, lang: str, *, client_factory, published=None) -> None:
    """Right after ``remember_quiz``, under the same lock: the fact that outran creation,
    or the settled state the ``chat.quiz`` event itself carries (a queued ask can land
    after its question was answered or closed); the higher-ranked of the two applies,
    and the facts that outran creation count as the newer ones."""
    outran = _RETAINED_FACTS.pop(token, None)
    if published and str(published.get("state") or "open") != "open":
        _retain_fact(token, published)
    if outran is not None:
        _retain_fact(token, outran)
    event = _RETAINED_FACTS.pop(token, None)
    if event is not None:
        await _apply_fact(api, token, event, lang, client_factory)


def _outcome(status: int, payload: Mapping[str, Any], attempted: Mapping[str, Any]) -> str:
    """``accepted``, ``conflict``, ``closed``, ``unconfirmed`` or ``refused``: the host's state and
    recorded winner outrank the HTTP code. A 409 or a 5xx naming the recorded winner is accepted
    when that winner is this very answer (sent again after an unconfirmed one, or recorded with its
    delivery still owed) and a conflict otherwise: before refusing a later answer the host retries
    the owed delivery of the earlier winner, and that retry can fail with a 503 carrying the earlier
    winner. A 5xx naming no winner leaves the outcome unknown."""
    state, winner = str(payload.get("state") or ""), _winner(payload)
    if status < 400:
        return "accepted"
    if status == 409 and state == "closed":
        return "closed"
    if (status == 409 and (state == "answered" or winner)) or (status >= 500 and state == "answered" and winner):
        return "accepted" if winner == dict(attempted) else "conflict"
    return "unconfirmed" if status >= 500 else "refused"


def _refusal_text(status: int, lang: str) -> str:
    """No answer recorded, or (any 5xx) none confirmed: its outcome stays unknown."""
    texts = _texts(lang)
    if status >= 500:
        return texts.format("unconfirmed", status=status)
    return texts["gone"] if status == 404 else texts["expired"] if status == 409 else texts.format("failed", status=status)


def _lost(record: Mapping[str, Any], facts: Mapping[str, Any], lang: str) -> str:
    answer = _answer_echo(record, facts)
    return _texts(lang).format("lost", answer=answer) if answer else _texts(lang)["already"]


async def _settle_answer(api, client, record: Dict[str, Any], payload: Dict[str, Any],
                         lang: str) -> Tuple[bool, Dict[str, Any]]:
    """Settle the card on the host's record; ``(edit landed, merged facts)``. A failed edit
    is returned to the caller to say once, never retried."""
    token = mint_token(str(record.get("task_id") or ""), str(record.get("quiz_id") or ""))
    fact = {**payload, "state": "answered"}
    # The card's lifecycle lock (``_CARD_LOCKS``): an expiry edit already on the wire
    # lands first, then this answer re-reads the stored card and settles it last.
    async with card_lock(token):
        record = quiz_for_token(api, token) or record
        facts = merge_answer_facts(record, fact)
        edit = lifecycle_edit(record, fact, lang)
        message_id = int(record.get("message_id") or 0)
        if edit is None:
            return True, facts
        remember_state(api, token, record, "answered", facts)
        if not message_id:
            return True, facts
        settled = await client.edit_message_text_with_inline_keyboard(
            int(record.get("chat_id") or 0), message_id, edit[0], edit[1], parse_mode="")
    if not settled:
        api.log("warning", "Telegram quiz card edit failed (answered).")
    return bool(settled), facts


async def _settle_closed(api, client, record: Dict[str, Any], payload: Dict[str, Any], lang: str) -> None:
    """A refusal naming the question closed settles the card on that host fact, never on an answer."""
    if str(payload.get("state") or "") == "closed":
        fact = {**payload, "task_id": record.get("task_id"), "quiz_id": record.get("quiz_id")}
        await follow_lifecycle(api, fact, lang, client_factory=lambda: client)


async def _tell_outcome(api, client, record: Dict[str, Any], status: int, payload: Dict[str, Any],
                        outcome: str, chat_id: int, lang: str) -> None:
    """Settle the card on the host's record and say a problem once, as one message: a conflict naming
    the recorded answer, a closed question, a refusal, an unconfirmed answer, or a card that could not
    be updated (that line joins a conflict's own rather than adding a second message)."""
    texts = _texts(lang)
    if outcome in ("accepted", "conflict"):
        settled, facts = await _settle_answer(api, client, record, payload, lang)
        lines = [_lost(record, facts, lang)] if outcome == "conflict" else []
        if not settled:
            lines = (lines or [_receipt(facts, payload.get("answered_after_terminal") is True, lang)]) + [texts["not_updated"]]
    else:
        lines = [texts["closed"] if outcome == "closed" else _refusal_text(status, lang)]
    if lines:
        await client.send_message(chat_id, "\n".join(lines), parse_mode="")
    if outcome == "closed":
        await _settle_closed(api, client, record, payload, lang)


async def answer_from_callback(
    api, client, cb_data: str, *, cb_id: str, update_id: int, lang: str, post: HostPost,
) -> None:
    """A tapped option → the decision ingress; the toast and the card tell the host's record, and a
    problem also gets its one message, as a text answer does (``_tell_outcome``)."""
    parts = str(cb_data or "").split(":")
    record = quiz_for_token(api, parts[1]) if len(parts) == 3 else None
    try:
        index = int(parts[2]) if len(parts) == 3 else -1
    except ValueError:
        index = -1
    options = list((record or {}).get("options") or [])
    if record is None or not 0 <= index < len(options):
        await client.answer_callback_query(cb_id, text=_texts(lang)["gone"])
        return
    status, payload = await _deliver(api, post, record, option_index=index, comment="", update_id=update_id)
    outcome = _outcome(status, payload, {"answered_index": index})
    late = payload.get("answered_after_terminal") is True
    if outcome == "accepted":
        token = mint_token(str(record.get("task_id") or ""), str(record.get("quiz_id") or ""))
        toast = "✅ " + _receipt(merge_answer_facts(quiz_for_token(api, token) or record, payload), late, lang)
    else:
        texts = _texts(lang)
        toast = (texts["already"] if outcome == "conflict" else texts["closed"] if outcome == "closed"
                 else _refusal_text(status, lang))
    await client.answer_callback_query(cb_id, text=_clip_u16(toast, _TOAST_MAX))
    # The card settles on the RECORDED answer: a first-wins loser learns the whole winner.
    await _tell_outcome(api, client, record, status, payload, outcome, int(record.get("chat_id") or 0), lang)


async def answer_from_reply(
    api, client, record: Dict[str, Any], answer_text: str, *,
    chat_id: int, update_id: int, lang: str, post: HostPost,
) -> None:
    """A reply to the card → the owner's own verbatim answer (comment-only). The settled card is
    its receipt; one message comes only for a problem (``_tell_outcome``)."""
    status, payload = await _deliver(api, post, record, option_index=None, comment=answer_text, update_id=update_id)
    outcome = _outcome(status, payload, {"comment": answer_text} if answer_text else {})
    await _tell_outcome(api, client, record, status, payload, outcome, chat_id, lang)
