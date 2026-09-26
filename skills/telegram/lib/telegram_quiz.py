"""Answering an Ouroboros quiz card from Telegram (#472).

The host's ``chat.quiz`` event carries the card identity (``task_id``,
``quiz_id``). The owner's button tap or reply is relayed to the SAME decision
ingress the web card uses — Host Service ``POST /chat/decision`` →
``task_decision.answer_decision`` — so the answer is idempotent per
``request_id`` (``tg:<update_id>``), first answer wins, and a late answer is
accepted exactly as it is for the browser card: the host records it and delivers
it into the card's chat as an ordinary owner message, and the toast says which
of those happened. The only state kept here maps a
short callback token and the sent message to that identity: Telegram caps
``callback_data`` at 64 bytes, too short for the ids themselves. A card also
remembers its settled lifecycle state, so the host's ``chat.quiz_state`` facts
edit it forward only. Nothing here parses the owner's words; a reply is
delivered verbatim as their own answer.

The card carries the whole authored text: the project it belongs to, the host's
facts about the asking task, the question, the stake, and every option with its
detail. A card longer than one Telegram message is never cut: the explanation
goes out first as ordered plain messages through the client's existing chunker,
and a compact message with the project line, the numbered option labels, the
hint and the keyboard follows last. Only that last message is remembered, so a
tap, a reply to it and the answered-edit keep working exactly as for a short
card; a reply to one of the earlier explanation parts is an ordinary owner
message, deliberately without per-part bookkeeping. An open question (no
options) is the same card without a keyboard; its hint asks for a reply.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import weakref
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from .telegram_api import _TELEGRAM_TEXT_LIMIT, _u16len
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

_TEXTS = {
    "en": {
        "hint": "Tap an option, or reply to this message with your own answer.",
        "hint_open": "Reply to this message with your answer.",
        "recorded": "✅ Answer delivered to the task.",
        "late_delivered": "✅ The task had already finished — your answer was delivered to the chat.",
        "late_recorded": "✅ Answer recorded. The task had already finished and this card has no chat to deliver it to.",
        "already": "This question was already answered.",
        "expired": "This question has expired — the task moved on.",
        "gone": "This question is no longer known to Ouroboros.",
        "failed": "Could not deliver the answer (HTTP {status}). Try again.",
        "answered_line": "Answered: {answer}",
        "answered_plain": "Answered.",
        "resumed": "The task continued; an answer is still accepted.",
        "expired_terminal": "The task finished; a late answer is accepted as your message.",
        "superseded": "Replaced by a newer question.",
        "project": "Project",
        "question": "Question",
        "stake": "At stake",
        "meanwhile": "Continuing meanwhile",
        "waiting": "Waiting for your answer; Stop and the task deadline still apply.",
    },
    "ru": {
        "hint": "Нажмите вариант или ответьте на это сообщение своим текстом.",
        "hint_open": "Ответьте на это сообщение своим текстом.",
        "recorded": "✅ Ответ передан задаче.",
        "late_delivered": "✅ Задача уже завершилась — ответ доставлен в чат.",
        "late_recorded": "✅ Ответ записан. Задача уже завершилась, а доставлять его в чат некуда.",
        "already": "На этот вопрос уже отвечали.",
        "expired": "Вопрос устарел — задача уже двинулась дальше.",
        "gone": "Этот вопрос Ouroboros больше не знает.",
        "failed": "Не удалось передать ответ (HTTP {status}). Попробуйте ещё раз.",
        "answered_line": "Ответ: {answer}",
        "answered_plain": "Ответ получен.",
        "resumed": "Задача продолжила работу; ответ всё ещё принимается.",
        "expired_terminal": "Задача завершилась; поздний ответ придёт как ваше сообщение.",
        "superseded": "Вопрос заменён более новым.",
        "project": "Проект",
        "question": "Вопрос",
        "stake": "Что на кону",
        "meanwhile": "Пока продолжаю так",
        "waiting": "Жду вашего ответа; Stop и срок задачи по-прежнему действуют.",
    },
}

# A remembered card's lifecycle only moves forward, as on the web card: a closed
# wait or an expiry never reopens a settled card, and nothing downgrades an answer.
_LIFECYCLE_RANK = {"open": 0, "expired_terminal": 1, "superseded": 2, "answered": 3}

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


def _texts(lang: str) -> Dict[str, str]:
    return _TEXTS["ru" if lang == "ru" else "en"]


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
        lines.append(host_facts)
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


async def send_quiz_card(client, chat_id: int, *, body: str, compact: str, hint_text: str,
                         keyboard: List[List[dict]]) -> Tuple[int, bool]:
    """Send the card; return the keyboard message id and whether it overflowed.

    A card that fits Telegram's per-message limit (UTF-16 units) WITH room for
    its later answered edit is one message with the keyboard. A longer one is
    sent as ordered plain parts through the client's chunker, then the compact
    keyboard message; nothing authored is truncated.
    """
    async def send(text: str) -> int:
        # An open question (no options) carries no keyboard: a plain message
        # the owner replies to, remembered exactly like a keyboard message.
        if keyboard:
            return int(await client.send_message_with_inline_keyboard(
                chat_id, text, keyboard, parse_mode="") or 0)
        return int(await client.send_message(chat_id, text, parse_mode="") or 0)

    full = f"{body}\n{hint_text}"
    if _u16len(full) + _ANSWERED_EDIT_RESERVE <= _TELEGRAM_TEXT_LIMIT:
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


def remember_state(api, token: str, record: Dict[str, Any], state: str) -> None:
    """Persist a settled lifecycle state on the card: the no-rollback evidence."""
    if _LIFECYCLE_RANK.get(state, 0) and str(record.get("state") or "") != state:
        remember_quiz(api, token, {**record, "state": state})


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


def _outcome_text(status: int, payload: Dict[str, Any], lang: str) -> str:
    texts = _texts(lang)
    if status < 400:
        if payload.get("answered_after_terminal") is True:
            # The card outlived its task: the answer became an owner message in
            # the card's chat, unless that chat has no owner turn to start.
            return texts["late_delivered" if payload.get("forwarded") else "late_recorded"]
        return texts["recorded"]
    if status == 404:
        return texts["gone"]
    if status == 409:
        answered = payload.get("answered_index") is not None or str(payload.get("state") or "") == "answered"
        return texts["already"] if answered else texts["expired"]
    return texts["failed"].format(status=status)


def _echo(answer: str) -> str:
    return answer if len(answer) <= _ANSWER_ECHO_MAX else answer[:_ANSWER_ECHO_MAX] + "…"


def _answered_text(record: Dict[str, Any], answer: str, lang: str) -> str:
    texts = _texts(lang)
    line = texts["answered_line"].format(answer=answer) if answer else texts["answered_plain"]
    return f"{record.get('text') or ''}\n{line}"


def lifecycle_edit(
    record: Dict[str, Any], event: Dict[str, Any], lang: str,
) -> Optional[Tuple[str, List[List[dict]]]]:
    """The (text, keyboard) edit a host ``chat.quiz_state`` fact asks of a sent card.

    ``None`` when the fact changes nothing here: a state this card cannot show, an
    ``open`` that does not close a wait, or a fact older than the card's own state.
    The card mirrors the web one (``web/modules/question_presentation.js``): an
    answer settles it on the recorded option and/or the owner's own words; a closed
    wait drops the waiting line; an expired card stays answerable, because a late
    answer is still accepted as the owner's message (В17a=A); a superseded card is
    a read-only record. An answer is always re-applied — the edit is idempotent.
    """
    state = str(event.get("state") or "")
    if state not in _LIFECYCLE_RANK or (state == "open" and event.get("wait_for_answer") is not False):
        return None
    if _LIFECYCLE_RANK[state] < _LIFECYCLE_RANK.get(str(record.get("state") or "open"), 0):
        return None
    texts = _texts(lang)
    base = str(record.get("text") or "")
    if state == "answered":
        options = list(record.get("options") or [])
        index = event.get("answered_index")
        parts = []
        if isinstance(index, int) and not isinstance(index, bool):
            parts.append(f"{index + 1}. {options[index]}" if 0 <= index < len(options) else f"{index + 1}.")
        if str(event.get("comment") or ""):
            parts.append(_echo(str(event["comment"])))
        return _answered_text(record, " — ".join(parts), lang), []
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
    remember_state(api, token, record, str(event.get("state") or ""))
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
    """Keep the highest-ranked fact that arrived before its card was remembered."""
    held = _RETAINED_FACTS.get(token) or {}
    if _LIFECYCLE_RANK.get(str(event.get("state") or ""), 0) >= _LIFECYCLE_RANK.get(str(held.get("state") or ""), 0):
        _RETAINED_FACTS.pop(token, None)
        _RETAINED_FACTS[token] = dict(event)
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


async def apply_retained_fact(api, token: str, lang: str, *, client_factory) -> None:
    """Right after ``remember_quiz``, under the same lock: the fact that outran creation."""
    event = _RETAINED_FACTS.pop(token, None)
    if event is not None:
        await _apply_fact(api, token, event, lang, client_factory)


async def _mark_answered(api, client, record: Dict[str, Any], answer: str, lang: str) -> None:
    token = mint_token(str(record.get("task_id") or ""), str(record.get("quiz_id") or ""))
    # The card's lifecycle lock (``_CARD_LOCKS``): an expiry edit already on the wire
    # lands first, then this answer re-reads the stored card and settles it last.
    async with card_lock(token):
        record = quiz_for_token(api, token) or record
        remember_state(api, token, record, "answered")
        message_id = int(record.get("message_id") or 0)
        if not message_id:
            return
        await client.edit_message_text_with_inline_keyboard(
            int(record.get("chat_id") or 0), message_id, _answered_text(record, answer, lang), [], parse_mode="",
        )


async def answer_from_callback(
    api, client, cb_data: str, *, cb_id: str, update_id: int, lang: str, post: HostPost,
) -> None:
    """A tapped option → the decision ingress; toast the honest outcome."""
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
    await client.answer_callback_query(cb_id, text=_outcome_text(status, payload, lang))
    recorded = payload.get("answered_index")
    if status < 400 or (status == 409 and isinstance(recorded, int)):
        # Settle the card on the RECORDED option (a first-wins loser learns the winner).
        chosen = recorded if isinstance(recorded, int) and 0 <= recorded < len(options) else index
        await _mark_answered(api, client, record, f"{chosen + 1}. {options[chosen]}", lang)


async def answer_from_reply(
    api, client, record: Dict[str, Any], answer_text: str, *,
    chat_id: int, update_id: int, lang: str, post: HostPost,
) -> None:
    """A reply to the card → the owner's own verbatim answer (comment-only)."""
    status, payload = await _deliver(api, post, record, option_index=None, comment=answer_text, update_id=update_id)
    await client.send_message(chat_id, _outcome_text(status, payload, lang))
    if status < 400:
        await _mark_answered(api, client, record, _echo(answer_text), lang)
