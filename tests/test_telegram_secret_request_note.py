"""A secret request reaches Telegram as a note on where to enter it, never as an answerable card.

The host's ``chat.quiz`` event for a ``showSecretHitl`` request carries ``secret_request`` and a
host-authored question that names the protected form in the app. The bridge sends it without a
keyboard or reply prompt, answers a reply to it locally without posting anything to the host,
and follows the request's lifecycle without the late-answer or answer-echo lines.
"""
from __future__ import annotations

import asyncio

from tests.test_telegram_quiz_answers import _LAST_CLIENT, Api, Client, _load_plugin, _run_poller, _settings
from tests._cases import case_folders

CANARY = "zq7-CANARY-Ωlj3~p0+kx/2"
_HINT = ("Secret needed for telegram: botToken. Enter it in the protected form in the Ouroboros app on the web or "
         "desktop; do not send it in this chat.")
_EVENT = {
    "chat_id": 1, "quiz_id": "q-secret", "task_id": "task-1", "question": _HINT, "options": [], "stake": "",
    "assumption": "", "transport": {}, "wait_for_answer": True,
    "secret_request": {"plugin_name": "telegram", "fields": [{"id": "botToken", "setting_key": "TELEGRAM_BOT_TOKEN"}]},
}


def _send_note(plugin, tmp_path, monkeypatch):
    _settings(tmp_path)
    monkeypatch.setattr(plugin, "TelegramClient", Client)
    api = Api(tmp_path)
    asyncio.run(plugin._make_quiz(api)(dict(_EVENT)))
    return api


def test_a_secret_request_is_a_note_without_keyboard_wait_line_or_reply_prompt(tmp_path, monkeypatch):
    plugin = _load_plugin()
    api = _send_note(plugin, tmp_path, monkeypatch)
    client = _LAST_CLIENT[-1]
    assert client.panels == []
    assert client.sent == [(42, f"Question: {_HINT}\nA reply here is not sent to Ouroboros.")]
    record = plugin.telegram_quiz.quiz_for_message(api, "", 42, 1)
    assert record["secret"] is True and record["options"] == []
    assert "Reply to this message" not in client.sent[0][1] and "Waiting for your answer" not in client.sent[0][1]


def test_a_reply_with_a_value_is_answered_locally_and_posted_nowhere(tmp_path, monkeypatch):
    plugin = _load_plugin()
    api = _send_note(plugin, tmp_path, monkeypatch)
    Client.updates = [{"update_id": 300, "message": {
        "message_id": 1001, "chat": {"id": 42, "type": "private"}, "from": {"id": 42},
        "text": CANARY, "reply_to_message": {"message_id": 1},
    }}]
    posts = []
    injected = _run_poller(plugin, api, monkeypatch, posts)
    assert posts == [] and injected == [], "nothing reaches the host: not the decision ingress, not the chat"
    replies = [text for _chat, text in _LAST_CLIENT[-1].sent]
    assert replies == ["Not sent: a secret goes only into the protected form in the Ouroboros app. "
                       "You can delete your message here."]
    assert all(CANARY not in message for _level, message in api.logs)


def test_the_note_follows_the_lifecycle_without_answer_or_late_answer_lines(tmp_path, monkeypatch):
    plugin = _load_plugin()
    quiz = plugin.telegram_quiz
    record = {"task_id": "task-1", "quiz_id": "q-secret", "chat_id": 42, "message_id": 1, "options": [],
              "text": f"Question: {_HINT}", "secret": True}
    assert quiz.lifecycle_edit(dict(record), {"state": "answered", "comment": CANARY}, "en") == (
        f"Question: {_HINT}\nSaved in Ouroboros.", [])
    late = quiz.lifecycle_edit(dict(record), {"state": "expired_terminal"}, "en")
    assert late == (f"Question: {_HINT}\nA reply here is not sent to Ouroboros.", [])
    assert "late answer" not in late[0]
    closed = quiz.lifecycle_edit(dict(record), {"state": "closed", "closed_reason": "The owner cancelled the secret "
                                                "request; this request saved nothing."}, "en")
    assert closed[0].endswith("Question closed.\nThe owner cancelled the secret request; this request saved nothing.")
    ordinary = {**record, "secret": False, "text": "Question: Which db?"}
    assert "late answer" in quiz.lifecycle_edit(ordinary, {"state": "expired_terminal"}, "en")[0]


# --- a reply to the note is local whatever its shape (R3) -------------------------------------------
import pytest  # noqa: E402

_REFUSAL = ("Not sent: a secret goes only into the protected form in the Ouroboros app. "
            "You can delete your message here.")
_MODES = ["strict", "safe_commands", "full_access"]
_SHAPES = {
    "text": {"text": CANARY},
    "slash_token_text": {"text": "/" + CANARY},
    "slash_token_caption_photo": {"photo": [{"file_id": "p1"}], "caption": "/" + CANARY},
    "captionless_photo": {"photo": [{"file_id": "p1"}]},
    "caption_photo": {"photo": [{"file_id": "p1"}], "caption": CANARY},
    "document_caption": {"document": {"file_id": "d1", "file_name": "t.txt"}, "caption": CANARY},
    "captionless_document": {"document": {"file_id": "d1", "file_name": "t.txt"}},
    "voice": {"voice": {"file_id": "v1"}},
    "video_caption": {"video": {"file_id": "m1"}, "caption": "/" + CANARY},
}


class _MediaClient(Client):
    downloads: list = []

    async def download_photo(self, file_id):
        _MediaClient.downloads.append(file_id)
        return "aGk=", "image/jpeg"

    async def download_file(self, file_id):
        _MediaClient.downloads.append(file_id)
        return b"hi"


def _owner_reply(plugin, api, monkeypatch, mode, fields, *, to=1, date=None):
    """One owner message through the real handler; fake host post/inject/download, one fake client."""
    posts, injected, lookups = [], [], []
    _MediaClient.downloads = []

    async def fake_post(_api, path, body):
        posts.append((path, body))
        return 200, {"ok": True, "state": "answered"}

    async def fake_inject(_api, payload):
        injected.append(payload)

    real = plugin.telegram_quiz.quiz_for_message
    monkeypatch.setattr(plugin.telegram_quiz, "quiz_for_message",
                        lambda *args: (lookups.append(args), real(*args))[1])
    monkeypatch.setattr(plugin, "_host_post", fake_post)
    monkeypatch.setattr(plugin, "_inject", fake_inject)
    monkeypatch.setattr(plugin, "_POLL_GENERATION_STARTED", 1)
    client = _MediaClient("token")
    message = {"message_id": 1001, "chat": {"id": 42, "type": "private"}, "from": {"id": 42},
               "date": date or 2_000_000_000, **fields}
    if to is not None:
        message["reply_to_message"] = {"message_id": to}
    asyncio.run(plugin._handle_owner_message(api, client, {"update_id": 301, "message": message}, 301, mode, "en"))
    return client, posts, injected, lookups


def test_any_shape_of_reply_to_a_secret_note_is_refused_locally(subtests, tmp_path, monkeypatch):
    case_paths = case_folders(tmp_path)
    for mode in _MODES:
        for shape in sorted(_SHAPES):
            tmp_path = next(case_paths)
            with subtests.test(mode=mode, shape=shape), monkeypatch.context() as monkeypatch:
                plugin = _load_plugin()
                api = _send_note(plugin, tmp_path, monkeypatch)
                client, posts, injected, lookups = _owner_reply(plugin, api, monkeypatch, mode, _SHAPES[shape])
                assert posts == [] and injected == [], "neither the decision ingress nor ordinary chat"
                assert _MediaClient.downloads == [], "no attachment is downloaded for a refused reply"
                assert client.sent == [(42, _REFUSAL)], "exactly one local refusal"
                assert len(lookups) == 1, "the card is looked up once"
                assert all(CANARY not in text for _chat, text in client.sent)
                assert all(CANARY not in message for _level, message in api.logs)


@pytest.mark.parametrize("command", ["/menu", "/help", "/language@ouro_bot"])
def test_explicit_local_commands_still_run_as_a_reply_to_the_note(tmp_path, monkeypatch, command):
    plugin = _load_plugin()
    api = _send_note(plugin, tmp_path, monkeypatch)
    client, posts, injected, _lookups = _owner_reply(plugin, api, monkeypatch, "full_access", {"text": command})
    assert posts == [] and injected == []
    assert _REFUSAL not in [text for _chat, text in client.sent] and (client.panels or client.sent)


def test_panic_still_reaches_the_host_as_a_reply_to_the_note(tmp_path, monkeypatch):
    plugin = _load_plugin()
    api = _send_note(plugin, tmp_path, monkeypatch)
    for mode in _MODES:
        client, posts, injected, _lookups = _owner_reply(plugin, api, monkeypatch, mode, {"text": "/panic"})
        assert posts == [] and [item["text"] for item in injected] == ["/panic"]
        assert _REFUSAL not in [text for _chat, text in client.sent]


def test_language_with_an_argument_is_not_a_way_around_the_refusal(tmp_path, monkeypatch):
    plugin = _load_plugin()
    api = _send_note(plugin, tmp_path, monkeypatch)
    chosen = []

    async def fake_choose(_api, requested, _lang, _reply):
        chosen.append(requested)

    monkeypatch.setattr(plugin, "_choose_language", fake_choose)
    client, posts, injected, _lookups = _owner_reply(plugin, api, monkeypatch, "full_access", {"text": "/language " + CANARY})
    assert chosen == [] and posts == [] and injected == []
    assert client.sent[-1][1] == _REFUSAL
    # The same words without a reply are the owner's ordinary language request.
    _owner_reply(plugin, api, monkeypatch, "full_access", {"text": "/language Quenya"}, to=None)
    assert chosen == ["Quenya"]


def test_ordinary_replies_and_media_keep_their_paths(subtests, tmp_path, monkeypatch):
    case_paths = case_folders(tmp_path)
    for mode in _MODES:
        tmp_path = next(case_paths)
        with subtests.test(mode=mode), monkeypatch.context() as monkeypatch:
            plugin = _load_plugin()
            api = _send_note(plugin, tmp_path, monkeypatch)
            asyncio.run(plugin._make_quiz(api)({**_EVENT, "quiz_id": "q-plain", "task_id": "task-2", "question": "Which db?",
                                                 "options": [{"label": "a"}], "secret_request": None}))
            record = plugin.telegram_quiz.quiz_for_token(api, "", plugin.telegram_quiz.mint_token("task-2", "q-plain"))
            assert record and not record.get("secret")
            plain = record["message_id"]
            # A text reply to an ordinary question is its answer, as before.
            _client, posts, injected, lookups = _owner_reply(plugin, api, monkeypatch, mode, {"text": "postgres"}, to=plain)
            assert len(posts) == 1 and injected == [] and len(lookups) == 1
            # A captionless photo replying to it, and a photo with no reply at all, reach ordinary chat with the image.
            for to in (plain, None):
                _client, posts, injected, _lookups = _owner_reply(plugin, api, monkeypatch, mode, {"photo": [{"file_id": "p1"}]}, to=to)
                assert posts == [] and len(injected) == 1 and injected[0]["image_base64"] == "aGk="
                assert _MediaClient.downloads == ["p1"]
            # A document with a caption and a voice note ride the shared attachment path.
            _client, posts, injected, _lookups = _owner_reply(plugin, api, monkeypatch, mode, {"voice": {"file_id": "v1"}}, to=None)
            assert len(injected) == 1 and injected[0].get("attachments")
            # Without any reply no lookup happens at all.
            assert _owner_reply(plugin, api, monkeypatch, mode, {"text": "hello"}, to=None)[3] == []
