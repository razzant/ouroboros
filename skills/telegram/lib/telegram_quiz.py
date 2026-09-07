"""Answering an Ouroboros quiz card from Telegram (#472).

The host's ``chat.quiz`` event carries the card identity (``task_id``,
``quiz_id``). The owner's button tap or reply is relayed to the SAME decision
ingress the web card uses — Host Service ``POST /chat/decision`` →
``task_decision.answer_decision`` — so the answer is idempotent per
``request_id`` (``tg:<update_id>``), first answer wins, and a late answer gets
the same honest 404/409 the browser gets. The only state kept here maps a
short callback token and the sent message to that identity: Telegram caps
``callback_data`` at 64 bytes, too short for the ids themselves. Nothing here
parses the owner's words; a reply is delivered verbatim as their own answer.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from .telegram_state import _read_json_file, _state_file

_QUIZ_STATE_FILE = "quiz_state.json"
_MAX_REMEMBERED = 50
_CALLBACK_PREFIX = "qz:"
_BUTTON_LABEL_MAX = 40
_ANSWER_ECHO_MAX = 200

HostPost = Callable[[Any, str, Dict[str, Any]], Awaitable[Tuple[int, Dict[str, Any]]]]

_TEXTS = {
    "en": {
        "hint": "Tap an option, or reply to this message with your own answer.",
        "recorded": "✅ Answer delivered to the task.",
        "already": "This question was already answered.",
        "expired": "This question has expired — the task moved on.",
        "gone": "This question is no longer known to Ouroboros.",
        "failed": "Could not deliver the answer (HTTP {status}). Try again.",
        "answered_line": "Answered: {answer}",
    },
    "ru": {
        "hint": "Нажмите вариант или ответьте на это сообщение своим текстом.",
        "recorded": "✅ Ответ передан задаче.",
        "already": "На этот вопрос уже отвечали.",
        "expired": "Вопрос устарел — задача уже двинулась дальше.",
        "gone": "Этот вопрос Ouroboros больше не знает.",
        "failed": "Не удалось передать ответ (HTTP {status}). Попробуйте ещё раз.",
        "answered_line": "Ответ: {answer}",
    },
}


def _texts(lang: str) -> Dict[str, str]:
    return _TEXTS["ru" if lang == "ru" else "en"]


def hint(lang: str) -> str:
    return _texts(lang)["hint"]


def mint_token(task_id: str, quiz_id: str) -> str:
    """Short stable token for ``callback_data`` (Telegram's 64-byte cap)."""
    return hashlib.sha256(f"{task_id}:{quiz_id}".encode("utf-8")).hexdigest()[:12]


def quiz_keyboard(token: str, labels: List[str]) -> List[List[dict]]:
    """One button row per option; ``callback_data`` = ``qz:<token>:<index>``."""
    return [
        [{"text": f"{index}. {label}"[:_BUTTON_LABEL_MAX],
          "callback_data": f"{_CALLBACK_PREFIX}{token}:{index - 1}"}]
        for index, label in enumerate(labels, 1)
    ]


def render_quiz_text(question: str, labels: List[str], stake: str, assumption: str) -> str:
    lines = [f"Question: {question}"]
    if stake:
        lines.append(f"At stake: {stake}")
    lines.extend(f"{index}. {label}" for index, label in enumerate(labels, 1))
    if assumption:
        lines.append(f"Continuing meanwhile: {assumption}")
    return "\n".join(lines)


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
        return texts["recorded"]
    if status == 404:
        return texts["gone"]
    if status == 409:
        answered = payload.get("answered_index") is not None or str(payload.get("state") or "") == "answered"
        return texts["already"] if answered else texts["expired"]
    return texts["failed"].format(status=status)


async def _mark_answered(client, record: Dict[str, Any], answer: str, lang: str) -> None:
    message_id = int(record.get("message_id") or 0)
    if not message_id:
        return
    text = f"{record.get('text') or ''}\n{_texts(lang)['answered_line'].format(answer=answer)}"
    await client.edit_message_text_with_inline_keyboard(
        int(record.get("chat_id") or 0), message_id, text, [], parse_mode="",
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
        await _mark_answered(client, record, f"{chosen + 1}. {options[chosen]}", lang)


async def answer_from_reply(
    api, client, record: Dict[str, Any], answer_text: str, *,
    chat_id: int, update_id: int, lang: str, post: HostPost,
) -> None:
    """A reply to the card → the owner's own verbatim answer (comment-only)."""
    status, payload = await _deliver(api, post, record, option_index=None, comment=answer_text, update_id=update_id)
    await client.send_message(chat_id, _outcome_text(status, payload, lang))
    if status < 400:
        echo = answer_text if len(answer_text) <= _ANSWER_ECHO_MAX else answer_text[:_ANSWER_ECHO_MAX] + "…"
        await _mark_answered(client, record, echo, lang)
