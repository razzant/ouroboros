"""Security & trust-boundary regression tests for the unified Telegram skill (bridge, Presence, Mini App)."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import pathlib
import sqlite3
import sys
import time
import types
from typing import Any, Dict, List

import httpx
import pytest
from starlette.testclient import TestClient

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SKILL_ROOT = _REPO_ROOT / "skills" / "telegram"


def _load_plugin(pkg_name: str):
    if pkg_name in sys.modules:
        del sys.modules[pkg_name]
    for key in [k for k in sys.modules if k.startswith(pkg_name + ".")]:
        del sys.modules[key]
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(_SKILL_ROOT)]
    sys.modules[pkg_name] = pkg
    spec = importlib.util.spec_from_file_location(f"{pkg_name}.plugin", _SKILL_ROOT / "plugin.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _StubApi:
    def __init__(self, state_dir: pathlib.Path, *, token: str = "123456:TOK_A", data_dir: pathlib.Path | None = None):
        self._state_dir = pathlib.Path(state_dir)
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir = pathlib.Path(data_dir or state_dir.parent)
        self._token = token
        self.logs: List[tuple[str, str]] = []
        self.settings = {"TELEGRAM_BOT_TOKEN": token}

    def get_state_dir(self) -> str:
        return str(self._state_dir)

    def get_runtime_info(self) -> Dict[str, Any]:
        return {"data_dir": str(self._data_dir)}

    def get_settings(self) -> Dict[str, Any]:
        return dict(self.settings)

    def get_settings_state(self, keys=None) -> Dict[str, Any]:
        vals = {k: self.settings.get(k) for k in keys} if keys else dict(self.settings)
        return {"status": "ok", "values": vals}

    def get_secret(self, key: str, default: str = "") -> str:
        return str(self.settings.get(key, default) or "")

    def get_env(self, key: str, default: str = "") -> str:
        return str(self.settings.get(key, default) or "")

    def get_host_service_token(self):
        return types.SimpleNamespace(use_in_request=lambda: "host-tok")

    def log(self, level: str, message: str) -> None:
        self.logs.append((level, message))


# --- Group 1: Owner confirmation, revocation, bot/API-base binding, remote HTTPS ---


def test_group1_owner_pairing_revocation_identity_and_https(tmp_path):
    plugin = _load_plugin("tg_sec_g1")
    from tg_sec_g1.lib import telegram_outbox, telegram_owner_auth, telegram_quiz, telegram_state
    from tg_sec_g1.lib.telegram_api import TelegramBindingChanged, TelegramClient, TelegramRequestRejected
    from tg_sec_g1.scripts.telegram_settings import merge_settings, validate_telegram_api_base

    # Remote plaintext HTTP rejected; loopback HTTP and remote HTTPS accepted
    for bad in ("http://192.168.1.48:8081", "http://bot.example.com:8081", "http://198.51.100.9"):
        with pytest.raises(ValueError):
            validate_telegram_api_base(bad)
    for ok in ("http://127.0.0.1:8081", "http://localhost:8081", "http://[::1]:8081", "https://192.168.1.48:8081"):
        assert validate_telegram_api_base(ok) == ok

    # Unbound install rejects stranger DMs (no first-DM TOFU); pairing code is single-use + lockout
    api = _StubApi(tmp_path / "state")
    client_stub = types.SimpleNamespace(bot_id="123456", token="123456:TOK_A", api_base_origin="https://api.telegram.org")

    async def _run_auth():
        # Stranger DM rejected
        dm = {"message": {"chat": {"id": 42, "type": "private"}, "from": {"id": 42}, "text": "hello"}}
        assert await plugin._authorized_owner(api, client_stub, dm, "", "en") is None
        # Issue pairing code -> wrong attempts lock out after 5 tries
        issued = telegram_owner_auth.issue_owner_pairing_code(api)
        for _ in range(5):
            assert telegram_owner_auth.redeem_owner_pairing_code(api, 42, "/pair WRONG") is False
        assert telegram_owner_auth.redeem_owner_pairing_code(api, 42, f"/pair {issued['code']}") is False
        # Fresh code -> group chat rejected; private DM binds once; replay from stranger rejected
        fresh = telegram_owner_auth.issue_owner_pairing_code(api)["code"]
        grp = {"message": {"chat": {"id": -1001, "type": "supergroup"}, "from": {"id": 42}, "text": f"/pair {fresh}"}}
        assert await plugin._authorized_owner(api, client_stub, grp, "", "en") is None
        pair_dm = {"message": {"chat": {"id": 42, "type": "private"}, "from": {"id": 42}, "text": f"/pair {fresh}"}}
        assert await plugin._authorized_owner(api, client_stub, pair_dm, "", "en") == "42"
        replay_dm = {"message": {"chat": {"id": 99, "type": "private"}, "from": {"id": 99}, "text": f"/pair {fresh}"}}
        assert await plugin._authorized_owner(api, client_stub, replay_dm, "42", "en") is None

    asyncio.run(_run_auth())
    assert telegram_owner_auth.live_owner_chat_id(api) == 42

    # Revocation stops outbox parts/retries, quiz card edits, subagent card edits, and client binding_guard
    merge_settings(tmp_path / "state", {"TELEGRAM_CHAT_ID": "99"})
    assert telegram_owner_auth.owner_matches_live(api, 42) is False
    assert telegram_owner_auth.owner_matches_live(api, 99) is True

    # TelegramClient binding_guard aborts immediately
    guarded = TelegramClient("123456:TOK_A", binding_guard=lambda: telegram_owner_auth.owner_matches_live(api, 42))
    with pytest.raises(TelegramBindingChanged):
        guarded.check_binding()

    # Outbox record addressed to revoked chat 42 stops immediately before calling client.request
    called: list[str] = []
    outbox = telegram_outbox.outbox_for(api)
    rec_op = {"id": "op1", "chat_id": 42, "topic": "chat.outbound", "state": "pending", "parts": [{"n": 0, "state": "pending"}]}
    op = telegram_outbox._Operation(outbox, "chat.outbound", {"text": "secret"}, rec_op)
    fake_cl = types.SimpleNamespace(bot_id="123456", request=lambda *a, **k: called.append("sent"))
    with pytest.raises(telegram_outbox.OutboxStopped):
        asyncio.run(op._send(fake_cl, rec_op["parts"][0], "sendMessage", {"chat_id": 42, "text": "secret"}, None, 10))
    assert called == []

    # Quiz lifecycle target & subagent card refuse revoked chat 42
    tok = telegram_quiz.mint_token("t1", "q1")
    rec = {"task_id": "t1", "quiz_id": "q1", "chat_id": 42, "message_id": 10, "options": ["a"], "text": "Q?"}
    telegram_quiz.remember_quiz(api, "123456", tok, rec)
    assert telegram_quiz.lifecycle_target(api, "123456", {"task_id": "t1", "quiz_id": "q1", "state": "closed"}, "en") is None
    card_calls: list[str] = []
    fake_card_cl = types.SimpleNamespace(
        bot_id="123456",
        send_message=lambda *a, **k: card_calls.append("send"),
        edit_message_text=lambda *a, **k: card_calls.append("edit"),
    )
    asyncio.run(telegram_state._render_subagent_card(api, fake_card_cl, 42, {"subagent_task_id": "c1", "text": "x"}, "running", "en"))
    assert card_calls == []

    # Loopback-only owner/pair route refuses remote/MiniApp and issues one-time code on loopback
    from tg_sec_g1.scripts.telegram_settings import MINIAPP_MARKER_HEADER
    pair_handler = plugin._make_owner_pair(api)
    remote_req = types.SimpleNamespace(client=types.SimpleNamespace(host="198.51.100.7"), headers={})
    miniapp_req = types.SimpleNamespace(client=types.SimpleNamespace(host="127.0.0.1"), headers={MINIAPP_MARKER_HEADER: "1"})
    local_req = types.SimpleNamespace(client=types.SimpleNamespace(host="127.0.0.1"), headers={})
    assert asyncio.run(pair_handler(remote_req)).status_code == 403
    assert asyncio.run(pair_handler(miniapp_req)).status_code == 403
    pair_resp = asyncio.run(pair_handler(local_req))
    assert pair_resp.status_code == 200
    assert pair_resp.headers.get("cache-control") == "no-store"
    pair_body = json.loads(pair_resp.body.decode("utf-8"))
    assert pair_body["ok"] is True and pair_body["command"].startswith("/pair ")

    # getMe bot_id mismatch raises 401 TelegramRequestRejected; matching getMe binds (token, api_base)
    bad_me = types.SimpleNamespace(
        bot_id="123456", token="123456:TOK_A", api_base_origin="https://api.telegram.org",
        call=lambda m, **k: asyncio.sleep(0, result={"ok": True, "result": {"id": 999999, "username": "impostor"}}),
    )
    with pytest.raises(TelegramRequestRejected):
        asyncio.run(plugin._validate_bot(api, bad_me, "full_access", "en"))

    rt = plugin._presence_runtime(api)
    good_me = types.SimpleNamespace(
        bot_id="123456", token="123456:TOK_A", api_base_origin="https://api.telegram.org",
        call=lambda m, **k: asyncio.sleep(0, result={"ok": True, "result": {"id": 123456, "username": "real_bot"}}),
    )
    asyncio.run(plugin._validate_bot(api, good_me, "full_access", "en"))
    assert rt.recorded_bot() == "123456"
    # Switching TELEGRAM_API_BASE invalidates the recorded bot identity for the previous server
    merge_settings(tmp_path / "state", {"TELEGRAM_API_BASE": "https://bot-alt.example:8081"})
    assert rt.recorded_bot() == ""


# --- Group 2: Secret exposure ledger, quotes, edits, uncertain sends, fail-closed ---


def test_group2_secret_ledger_quotes_edits_uncertain_and_fail_closed(tmp_path):
    _load_plugin("tg_sec_g2")
    from tg_sec_g2.lib import telegram_quiz, telegram_secret_guard as sg

    api = _StubApi(tmp_path / "state")
    # Durable non-evicting ledger survives 5500+ unrelated entries
    sg.record_secret_message(api, "123456", 42, 10, kind="card", task_id="t1", quiz_id="q1")
    with sqlite3.connect(tmp_path / "state" / "secret_guard.sqlite3") as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO secret_messages(bot_id, api_base, chat_id, message_id, kind, task_id, quiz_id, created_at) "
            "VALUES ('123456', '', 42, ?, 'card', '', '', 1.0)",
            [(10000 + idx,) for idx in range(5100)],
        )
    assert sg.lookup_secret_message(api, "123456", 42, 10) is not None

    # Direct reply quarantines message 501; later quote/external_reply/forward/edit of 501 ALL blocked
    direct = {"message_id": 501, "chat": {"id": 42, "type": "private"}, "text": "sk-secret", "reply_to_message": {"message_id": 10}}
    assert sg.classify_secret_exposure(api, "123456", direct) is not None
    assert sg.lookup_secret_message(api, "123456", 42, 501)["kind"] == "refused_reply"

    quote_of_refused = {
        "message_id": 502, "chat": {"id": 42, "type": "private"}, "text": "re-sending",
        "external_reply": {"chat": {"id": 42}, "message_id": 501}, "quote": {"text": "sk-secret"},
    }
    assert sg.classify_secret_exposure(api, "123456", quote_of_refused) is not None
    assert sg.lookup_secret_message(api, "123456", 42, 502)["kind"] == "refused_reply"

    fwd_of_refused = {
        "message_id": 503, "chat": {"id": 42, "type": "private"}, "text": "sk-secret",
        "forward_origin": {"chat": {"id": 42}, "message_id": 501},
    }
    assert sg.classify_secret_exposure(api, "123456", fwd_of_refused) is not None

    edit_of_refused = {"message_id": 501, "chat": {"id": 42, "type": "private"}, "text": "edited sk-secret", "edit_date": 1700000001}
    assert sg.classify_secret_exposure(api, "123456", edit_of_refused, is_edit=True) is not None

    # Uncertain secret card send fails closed on replies/quotes to unproven messages, allows known non-secret ref
    api2 = _StubApi(tmp_path / "state2")
    sg.mark_secret_uncertain(api2, "123456", 42, min_epoch=1700000000)
    unproven = {
        "message_id": 601, "chat": {"id": 42, "type": "private"}, "text": "maybe-secret",
        "reply_to_message": {"message_id": 77, "date": 1700000010, "from": {"id": 123456, "is_bot": True}},
    }
    assert sg.classify_secret_exposure(api2, "123456", unproven, known_non_secret_fn=lambda _c, _m: False) is not None
    assert sg.classify_secret_exposure(api2, "123456", unproven, known_non_secret_fn=lambda _c, m: m == 77) is None

    # Corrupt quiz_state.json -> fail closed (quiz_for_message & classify_secret_exposure both return secret+unreadable)
    (tmp_path / "state2" / "quiz_state.json").write_text("{not-valid-json", encoding="utf-8")
    q_hit = telegram_quiz.quiz_for_message(api2, "123456", 42, 88)
    assert q_hit is not None and q_hit.get("secret") is True and q_hit.get("unreadable") is True
    reply_corrupt = {"message_id": 602, "chat": {"id": 42, "type": "private"}, "text": "x", "reply_to_message": {"message_id": 88}}
    hit = sg.classify_secret_exposure(api2, "123456", reply_corrupt)
    assert hit is not None and hit.get("unreadable") is True


# --- Group 3: Panic isolation, overflow, literal commands, Host parity, ordinary message retry ---


def test_group3_panic_isolation_overflow_literal_commands_and_retry(tmp_path, monkeypatch):
    plugin = _load_plugin("tg_sec_g3")
    from ouroboros.gateway import host_service
    from tg_sec_g3.lib.presence.custody import CustodyStore
    from tg_sec_g3.lib.presence.runtime import PresenceRuntime
    from tg_sec_g3.lib.telegram_owner_auth import is_literal_slash_message, is_panic_command
    from tg_sec_g3.lib.telegram_poll_binding import InjectNotAccepted

    # Entity-aware command vs literal detection
    assert is_panic_command({"text": "/panic"}) is True
    assert is_panic_command({"text": "/panic@mybot"}, bot_username="mybot") is True
    assert is_panic_command({"text": "/panic@otherbot"}, bot_username="mybot") is False
    assert is_panic_command({"text": "/panicking"}) is False
    assert is_panic_command({"text": "/panic!"}) is False
    code_msg = {"text": "/panic", "entities": [{"type": "code", "offset": 0, "length": 6}]}
    assert is_panic_command(code_msg) is False
    assert is_literal_slash_message(code_msg) is True

    # Host /chat/inject distinguishes literal_text & /panicking from real /panic
    from ouroboros import server_control

    panics: list[str] = []
    queued: list[str] = []

    async def _auth(*_):
        return "telegram", {"inject_chat": True}

    monkeypatch.setattr(host_service, "_authenticated", _auth)
    monkeypatch.setattr(server_control, "request_panic", lambda _b, text, **_k: (panics.append(text), True)[1])
    monkeypatch.setattr(host_service, "dispatch_accepted_restart", lambda _b, text, **_k: queued.append(text))
    app = host_service.create_host_service_app(tmp_path / "host", bridge_getter=lambda: object())
    with TestClient(app) as tc:
        assert tc.post("/chat/inject", json={"text": "/panic", "literal_text": True}).status_code == 202
        assert tc.post("/chat/inject", json={"text": "/panicking"}).status_code == 202
        assert panics == [] and queued == ["/panic", "/panicking"]
        assert tc.post("/chat/inject", json={"text": "/panic"}).status_code == 202
        assert panics == ["/panic"]

    # Presence inbox overflow advances telegram_offset + records durable overflow without raising InjectNotAccepted
    api = _StubApi(tmp_path / "state")
    submitter = types.SimpleNamespace(
        delivery_reporting_status="supported", places_version=1, place_binding_version=1, connector_identity={},
    )
    rt = PresenceRuntime(api=api, state_dir=tmp_path / "state", submitter=submitter)
    rt.on_bot_validated("123456:TOK_A", "123456", "test_bot")
    # Fill inbox artificially up to limit and verify overflow records gap & advances offset cleanly
    store = CustodyStore.shared(tmp_path / "state" / "custody.sqlite3")
    with store._connect() as conn:
        conn.executemany(
            "INSERT INTO inbox(event_id, update_id, conversation_key, payload_json, state, attempts, available_at, received_at) "
            "VALUES (?, ?, 'tg:-1001', '{}', 'pending', 0, 0, 0)",
            [(f"ev-{i}", i) for i in range(1, 2005)],
        )
    flood_upd = {
        "update_id": 9001,
        "message": {"message_id": 9001, "date": 1700000000, "chat": {"id": -1001, "type": "supergroup", "title": "Spam"},
                    "from": {"id": 777, "first_name": "Spam"}, "text": "flood"},
    }
    assert asyncio.run(rt.ingest_update(flood_upd, account_id="123456")) is False
    assert store.telegram_offset() == 9002
    with store._connect() as conn:
        assert int(conn.execute("SELECT value FROM metadata WHERE key='inbox_overflow_count'").fetchone()[0]) == 1

    # Ordinary owner message provably unaccepted by Host (ConnectError, 429, 503) raises InjectNotAccepted
    class _FailAsyncClient:
        def __init__(self, outcome):
            self._outcome = outcome

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            return False

        async def post(self, *_a, **_k):
            if isinstance(self._outcome, Exception):
                raise self._outcome
            return types.SimpleNamespace(status_code=self._outcome, json=lambda: {"ok": False})

    (tmp_path / "state" / "settings.json").write_text(json.dumps({"TELEGRAM_CHAT_ID": "42"}), encoding="utf-8")
    api.get_skill_token = lambda: types.SimpleNamespace(use_in_request=lambda: "host-tok")
    for outcome in (httpx.ConnectError("down"), httpx.ConnectTimeout("to"), 429):
        monkeypatch.setattr(plugin.httpx, "AsyncClient", lambda *a, _o=outcome, **k: _FailAsyncClient(_o))
        with pytest.raises(InjectNotAccepted):
            asyncio.run(plugin._inject(api, {"text": "hello owner", "chat_id": 42}))


# --- Group 4: Chats, topics, receipts, moderation, albums, typing, HTML fallback, stale reaction ---


def test_group4_chats_topics_receipts_moderation_albums_and_rechecks(tmp_path):
    _load_plugin("tg_sec_g4")
    from tg_sec_g4.lib.presence.custody import CustodyStore
    from tg_sec_g4.lib.presence.file_access import receipt_visible_to
    from tg_sec_g4.lib.presence.host import PlaceBindingRevoked
    from tg_sec_g4.lib.presence.outbox_worker import ReactionDeliveryHelper
    from tg_sec_g4.lib.presence.runtime import PresenceRuntime
    from tg_sec_g4.lib.presence.tools import (
        make_moderation_tool, make_operation_tool, make_send_tool, make_typing_tool,
    )
    from tg_sec_g4.lib.presence.transport import PresenceBotClient, TelegramApiError

    # 1) receipt_visible_to: same place_id across different topics/chats NEVER leaks receipts
    ctx_topic7 = types.SimpleNamespace(
        task_id="t-7",
        task_metadata={"presence": {"place_id": "place-shared", "event": {
            "provider": "telegram", "account_id": "123456", "conversation_id": "-1001", "thread_id": "7",
        }}},
    )
    row_topic7 = {"account_id": "123456", "chat_id": "-1001", "topic_id": "7", "_place_binding": {"place_id": "place-shared"}, "_caller_scope": {"place_id": "place-shared"}}
    row_topic9 = {"account_id": "123456", "chat_id": "-1001", "topic_id": "9", "_place_binding": {"place_id": "place-shared"}, "_caller_scope": {"place_id": "place-shared"}}
    row_other_chat = {"account_id": "123456", "chat_id": "-1002", "topic_id": "7", "_place_binding": {"place_id": "place-shared"}, "_caller_scope": {"place_id": "place-shared"}}
    assert receipt_visible_to(ctx_topic7, row_topic7) is True
    assert receipt_visible_to(ctx_topic7, row_topic9) is False
    assert receipt_visible_to(ctx_topic7, row_other_chat) is False

    # 2) Tools: membership moderation rejects topic_id; unretained edit/delete/react/reply refused; incomplete album refused
    api = _StubApi(tmp_path / "state")
    store = CustodyStore.shared(tmp_path / "state" / "custody.sqlite3")
    with store._connect() as conn:
        # Seed known forum -1001: self msg 10 in topic 7, incomplete album msg 20 (album_origin_known=0), confirmed album msg 30 (album_origin_known=1)
        conn.execute(
            "INSERT INTO telegram_messages VALUES "
            "('123456', '-1001', 10, '7', 1, '123456', 'bot', 1, 'mine', '', 0),"
            "('123456', '-1001', 20, '7', 1, '555', 'bob', 0, 'alb', 'mg-inc', 0),"
            "('123456', '-1001', 30, '7', 1, '555', 'bob', 0, 'alb0', 'mg-ok', 1)"
        )
        conn.execute(
            "INSERT INTO inbox(event_id, update_id, conversation_key, payload_json, state, attempts, available_at, received_at) "
            "VALUES ('ev-typ', 1, 'tg:-1001:7', '{}', 'leased', 1, 0, 0)"
        )
    binder_calls: list[tuple[str, str, str]] = []
    binder = lambda a, c, t="": (binder_calls.append((a, c, t)), {"version": 1, "place_id": "p1", "connected_revision": 1})[1]
    send_tool = make_send_tool(api, lambda: "123456", binder)
    mod_tool = make_moderation_tool(api, lambda: "123456", binder)
    op_tool = make_operation_tool(api, lambda: "123456", binder)
    typ_tool = make_typing_tool(api, binder)

    # Group-wide member moderation refuses topic_id and binds at group level
    assert mod_tool(action="ban_member", chat_id="-1001", user_id=555, topic_id=7, request_id="b1")["ok"] is False
    assert mod_tool(action="restrict_member", chat_id="-1001", user_id=555, permissions={"can_send_messages": False}, topic_id=7, request_id="r1")["ok"] is False
    assert mod_tool(action="ban_member", chat_id="-1001", user_id=555, request_id="b2")["ok"] is True
    assert binder_calls[-1] == ("123456", "-1001", "")

    # reply_to_message_id topic mismatch and unretained forum target refused; matching topic allowed
    assert send_tool(chat_id="-1001", kind="message", text="hi", topic_id="9", reply_to_message_id="10", request_id="s1")["ok"] is False
    assert send_tool(chat_id="-1001", kind="message", text="hi", topic_id="7", reply_to_message_id="999", request_id="s2")["ok"] is False
    assert send_tool(chat_id="-1001", kind="message", text="hi", topic_id="7", reply_to_message_id="10", request_id="s3")["ok"] is True

    # editMessageText refuses unretained & non-self; accepts retained self in matching topic
    assert op_tool(method="editMessageText", chat_id="-1001", message_id=999, topic_id=7, text="x", request_id="e1")["ok"] is False
    assert op_tool(method="editMessageText", chat_id="-1001", message_id=20, topic_id=7, text="x", request_id="e2")["ok"] is False
    assert op_tool(method="editMessageText", chat_id="-1001", message_id=10, topic_id=7, text="x", request_id="e3")["ok"] is True

    # setMessageReaction refuses incomplete album (album_origin_known=0) and unretained forum msg; allows confirmed origin
    assert "album_target_unconfirmed" in op_tool(method="setMessageReaction", chat_id="-1001", message_id=20, topic_id=7, reaction=[{"type": "emoji", "emoji": "👍"}], request_id="rx1")["error"]
    assert op_tool(method="setMessageReaction", chat_id="-1001", message_id=999, topic_id=7, reaction=[{"type": "emoji", "emoji": "👍"}], request_id="rx2")["ok"] is False
    assert op_tool(method="setMessageReaction", chat_id="-1001", message_id=30, topic_id=7, reaction=[{"type": "emoji", "emoji": "👍"}], request_id="rx3")["ok"] is True

    # telegram_typing enforces exact topic match and captures _place_binding checked in process_typing
    typ_ctx = types.SimpleNamespace(
        task_id="t-typ",
        task_metadata={"presence": {"event": {
            "provider": "telegram", "account_id": "123456", "conversation_id": "-1001", "thread_id": "7", "source_event_id": "ev-typ",
        }}},
    )
    assert typ_tool(typ_ctx, chat_id="-1001")["ok"] is False  # omitted topic_id in topic turn!
    assert typ_tool(typ_ctx, chat_id="-1001", topic_id=7)["ok"] is True
    rows = store.typing_requests()
    assert len(rows) == 1 and json.loads(rows[0]["place_binding_json"])["place_id"] == "p1"

    # process_typing checks the saved binding and stops on PlaceBindingRevoked
    checked_bindings: list[Any] = []

    async def _revoked_place_hold(_acct, _chat, thread_id="", binding=None):
        checked_bindings.append(binding)
        raise PlaceBindingRevoked("disconnected")

    rt = PresenceRuntime(api=api, state_dir=tmp_path / "state", submitter=types.SimpleNamespace(place_hold=_revoked_place_hold, connector_identity={}), client_factory=lambda: types.SimpleNamespace(is_current=lambda: True))
    rt.on_bot_validated("123456:TOK_A", "123456", "bot")
    assert asyncio.run(rt.process_typing()) == 0
    assert checked_bindings == [{"version": 1, "place_id": "p1", "connected_revision": 1}]
    assert store.typing_requests() == []

    # 3) HTML 400 fallback re-invokes before_send before plain-text retry
    bot = PresenceBotClient("123456:TOK_A")
    calls: list[dict] = []

    async def _fake_call(method, params=None, **_k):
        calls.append(dict(params or {}))
        if len(calls) == 1:
            raise TelegramApiError(method, "can't parse entities", error_code=400)
        return {"message_id": 55}

    bot._post_json = _fake_call
    before_count = 0

    async def _before_revoke():
        nonlocal before_count
        before_count += 1
        if before_count == 1:
            raise PlaceBindingRevoked("revoked before plain fallback")

    with pytest.raises(PlaceBindingRevoked):
        asyncio.run(bot.send_text_chunk("-1001", "<b>hi</b>", parse_mode="HTML", before_send=_before_revoke))
    assert before_count == 1 and len(calls) == 1  # plain-text fallback NEVER sent!

    # 4) Stale ReactionDeliveryHelper aborts before Telegram call and never mutates store
    store_rx = CustodyStore.shared(tmp_path / "rx_state" / "custody.sqlite3")
    store_rx.enqueue_outbox("rx-stale", {
        "kind": "operation", "method": "setMessageReaction", "chat_id": "-1001", "account_id": "123456",
        "parameters": {"chat_id": "-1001", "message_id": 30, "reaction": [{"type": "emoji", "emoji": "👍"}]},
        "_reporting": {"version": 0},
    })
    stale_lease = store_rx.claim_outbox(lease_sec=1.0)
    assert stale_lease is not None and stale_lease.delivery_id == "rx-stale"
    # Another worker supersedes the lease
    with store_rx._connect() as conn:
        conn.execute("UPDATE outbox SET lease_token='newer-worker', attempts=attempts+1 WHERE delivery_id='rx-stale'")
    tg_called: list[str] = []

    async def _ready():
        return types.SimpleNamespace(
            is_current=lambda: True,
            operation=lambda *a, **k: (tg_called.append("called"), asyncio.sleep(0, result=True))[1],
        )

    helper = ReactionDeliveryHelper(
        store_rx, ready_client=_ready, resolved_reporting=lambda p: asyncio.sleep(0, result=p.get("_reporting") or {}),
        check_place=lambda *_a: asyncio.sleep(0), note_error=lambda _m: None, check_account=lambda _p: None,
        current_gen=lambda: 1,
    )
    asyncio.run(helper.process(stale_lease))
    assert tg_called == []
    with store_rx._connect() as conn:
        row = conn.execute("SELECT state, lease_token FROM outbox WHERE delivery_id='rx-stale'").fetchone()
        assert row["state"] == "leased" and row["lease_token"] == "newer-worker"


# --- Group 5: Zero-TOCTOU fd-pinned uploads, legacy media hold, streaming multipart, reclaim ---


def test_group5_zero_toctou_legacy_hold_streaming_and_reclaim(tmp_path):
    _load_plugin("tg_sec_g5")
    from tg_sec_g5.lib.presence import outbox_files
    from tg_sec_g5.lib.presence.custody import CustodyStore
    from tg_sec_g5.lib.presence.file_access import _open_confined_regular_fd
    from tg_sec_g5.lib.presence.runtime import PresenceRuntime
    from tg_sec_g5.lib.presence.transport import _AsyncMultipartStream, _UPLOAD_STREAM_CHUNK

    # 1) _open_confined_regular_fd refuses symlinks inside authorized root (zero-TOCTOU O_NOFOLLOW)
    root = tmp_path / "allowed"
    root.mkdir()
    secret = tmp_path / "outside.txt"
    secret.write_text("top-secret", encoding="utf-8")
    link = root / "escape.txt"
    link.symlink_to(secret)
    with pytest.raises(OSError):
        _open_confined_regular_fd(root, link)
    good = root / "ok.txt"
    good.write_bytes(b"hello-fd")
    fd, sz = _open_confined_regular_fd(root, good)
    try:
        kept = outbox_files.keep_fd(fd, good, tmp_path / "state", 1024 * 1024)
    finally:
        import os
        os.close(fd)
    assert sz == 8 and pathlib.Path(kept["path"]).read_bytes() == b"hello-fd"

    # 2) Legacy media row without _frozen is HELD (never reads raw file_path)
    api = _StubApi(tmp_path / "state")
    store = CustodyStore.shared(tmp_path / "state" / "custody.sqlite3")
    raw_file = tmp_path / "raw_unfrozen.bin"
    raw_file.write_bytes(b"do-not-send")
    store.enqueue_outbox("legacy-media", {
        "kind": "document", "chat_id": "42", "account_id": "123456",
        "file_path": str(raw_file), "caption": "", "_reporting": {"version": 0},
    })
    sent_media: list[str] = []
    rt = PresenceRuntime(
        api=api, state_dir=tmp_path / "state",
        submitter=types.SimpleNamespace(place_hold=lambda *_a, **_k: asyncio.sleep(0, result=""), connector_identity={}),
        client_factory=lambda: types.SimpleNamespace(
            is_current=lambda: True,
            send_media=lambda *a, **k: (sent_media.append("sent"), asyncio.sleep(0, result={"message_id": 1}))[1],
        ),
    )
    rt.on_bot_validated("123456:TOK_A", "123456", "bot")
    assert asyncio.run(rt.process_one_outbox()) is True
    assert sent_media == []
    assert "Held:" in store.delivery_receipt("legacy-media")["error"]

    # 3) Streaming multipart yields bounded chunks matching exact Content-Length
    big = tmp_path / "big.bin"
    payload_bytes = b"A" * (_UPLOAD_STREAM_CHUNK * 2 + 1234)
    big.write_bytes(payload_bytes)
    stream = _AsyncMultipartStream.open({"chat_id": "42"}, [("document", big)])
    chunks: list[bytes] = []

    async def _drain():
        async for ch in stream:
            chunks.append(ch)

    asyncio.run(_drain())
    joined = b"".join(chunks)
    assert len(joined) == stream.content_length
    assert max(len(c) for c in chunks) <= _UPLOAD_STREAM_CHUNK + 1024
    assert payload_bytes in joined

    # 4) Confirmed delivered copy is reclaimed; uncertain copy is PRESERVED
    frozen_deliv = outbox_files.keep(good, tmp_path / "state", 1024 * 1024)
    frozen_unc = outbox_files.keep(good, tmp_path / "state", 1024 * 1024)
    assert pathlib.Path(frozen_deliv["path"]).is_file() and pathlib.Path(frozen_unc["path"]).is_file()
    store.enqueue_outbox("ok-media", {
        "kind": "document", "chat_id": "43", "account_id": "123456",
        "file_path": frozen_deliv["path"], "_frozen": [frozen_deliv], "caption": "", "_reporting": {"version": 0},
    })
    with store._connect() as conn:
        conn.execute("UPDATE outbox SET available_at=9999999999 WHERE delivery_id='legacy-media'")
    assert asyncio.run(rt.process_one_outbox()) is True
    assert store.delivery_receipt("ok-media")["state"] == "delivered"
    assert not pathlib.Path(frozen_deliv["path"]).exists()

    # Uncertain send preserves its frozen copy
    store.enqueue_outbox("unc-media", {
        "kind": "document", "chat_id": "44", "account_id": "123456",
        "file_path": frozen_unc["path"], "_frozen": [frozen_unc], "caption": "", "_reporting": {"version": 0},
    })
    rt._custom_client_factory = lambda: types.SimpleNamespace(
        is_current=lambda: True,
        send_media=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("network drop mid-upload")),
    )
    assert asyncio.run(rt.process_one_outbox()) is True
    assert store.delivery_receipt("unc-media")["state"] == "uncertain"
    assert pathlib.Path(frozen_unc["path"]).is_file()


# --- Group 6: Separate Host protocol detection, downgrade hold, monotonic lease fencing ---


def test_group6_protocol_detection_downgrade_hold_and_lease_fencing(tmp_path):
    _load_plugin("tg_sec_g6")
    from tg_sec_g6.lib.presence.custody import CustodyStore
    from tg_sec_g6.lib.presence.events import TelegramEvent
    from tg_sec_g6.lib.presence.host import PresenceHostClient

    # 1) places_version=1 WITHOUT presence_place_binding_version -> detected separately; bind_place returns None (no loop!)
    class _Transport:
        def __init__(self, identity_body: Dict[str, Any]):
            self.identity_body = identity_body
            self.calls: list[str] = []

        async def request_json(self, _m, url, **_k):
            self.calls.append(url)
            if url.endswith("/identity"):
                return dict(self.identity_body)
            return {"state": "active", "place_id": "p1", "label": "G"}

    t_old = _Transport({"places_version": 1, "presence_delivery_version": 1})
    hc = PresenceHostClient(state_dir=tmp_path / "s1", token_provider=lambda: "tok", host_base="http://127.0.0.1:8767", transport=t_old)
    assert asyncio.run(hc.bind_place("123456", "-1001")) is None
    assert hc.places_version == 1 and hc.place_binding_version == 0

    # 2) Queued row carrying _place_binding is HELD (never released!) when Host lacks binding check or downgrades to places_version=0
    saved_binding = {"version": 1, "place_id": "p1", "connected_revision": 3}
    hold_old = asyncio.run(hc.place_hold("123456", "-1001", binding=saved_binding))
    assert hold_old.startswith("Held:")
    t_legacy = _Transport({"places_version": 0, "presence_delivery_version": 0})
    hc_legacy = PresenceHostClient(state_dir=tmp_path / "s2", token_provider=lambda: "tok", host_base="http://127.0.0.1:8767", transport=t_legacy)
    hold_downgrade = asyncio.run(hc_legacy.place_hold("123456", "-1001", binding=saved_binding))
    assert hold_downgrade.startswith("Held:")

    # 3) Monotonic lease_token fencing on inbox, presence_work, and isolated per-attempt staging dirs
    store = CustodyStore.shared(tmp_path / "s3" / "custody.sqlite3")
    ev = TelegramEvent(
        source_event_id="tg:123456:-1001:1", provider="telegram", account_id="123456",
        conversation_id="-1001", thread_id="", conversation_key="tg:-1001",
        actor={"id": "5"}, conversation={"id": "-1001", "type": "supergroup"},
        message={"message_id": 1, "text": "hi"}, text="hi",
    )
    store.commit_update(1, ev)
    lease_a = store.claim_inbox(lease_sec=1.0)
    assert lease_a and lease_a.lease_token
    with store._connect() as conn:
        conn.execute("UPDATE inbox SET lease_until=0 WHERE event_id=?", (ev.source_event_id,))
    lease_b = store.claim_inbox(lease_sec=60.0)
    assert lease_b and lease_b.lease_token != lease_a.lease_token
    # Stale worker A cannot checkpoint, release, fail, or record_submission
    assert store.inbox_lease_valid(ev.source_event_id, lease_a.lease_token) is False
    assert store.checkpoint_inbox(ev.source_event_id, {"stale": True}, lease_token=lease_a.lease_token) is False
    assert store.release_inbox(ev.source_event_id, reason="stale", retry_after_sec=0, lease_token=lease_a.lease_token) is False
    assert store.record_submission(
        ev.source_event_id, binding_id="", turn_ref="tA", outcome="deferred", text="", work_ref="wA", lease_token=lease_a.lease_token
    ) is False
    # Active worker B succeeds and creates work row
    assert store.record_submission(
        ev.source_event_id, binding_id="", turn_ref="tB", outcome="deferred", text="", work_ref="wB", lease_token=lease_b.lease_token
    ) is True

    # Monotonic lease_token fencing on presence_work
    w_a = store.claim_work(lease_sec=1.0)
    assert w_a and w_a.lease_token
    with store._connect() as conn:
        conn.execute("UPDATE presence_work SET lease_until=0 WHERE work_ref='wB'")
    w_b = store.claim_work(lease_sec=60.0)
    assert w_b and w_b.lease_token != w_a.lease_token
    assert store.record_work_outputs("wB", [{"output_ref": "o1", "text": "stale"}], lease_token=w_a.lease_token) is False
    assert store.complete_work("wB", status="completed", text="stale", lease_token=w_a.lease_token) is False
    assert store.complete_work("wB", status="completed", text="fresh", lease_token=w_b.lease_token) is True


# --- Group 7: Mini App sidecar session revocation aborts queued & streaming HTTP ---


def test_group7_sidecar_revocation_aborts_queued_and_streaming():
    scripts = str(_SKILL_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import sidecar

    async def _scenario():
        # 1) Queued request waiting on held sockets is immediately aborted with 401 on clear_public_url()
        #    and NEVER reaches the upstream core!
        started = asyncio.Event()
        release = asyncio.Event()
        seen_paths: list[str] = []

        async def _core(reader, writer):
            try:
                head = await reader.readuntil(b"\r\n\r\n")
                path = head.decode().split(" ", 2)[1]
                seen_paths.append(path)
                if len(seen_paths) >= 32:
                    started.set()
                await release.wait()
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
                await writer.drain()
            except Exception:
                pass
            finally:
                writer.close()

        server = await asyncio.start_server(_core, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        gw = sidecar.TelegramProxySidecar("123456:BOT_SECRET", 42, port)
        gw.set_public_url("https://quiet-river-42.trycloudflare.com/")
        token, _sess = gw._issue_session()
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=gw),
            base_url="https://quiet-river-42.trycloudflare.com",
            cookies={sidecar.SESSION_COOKIE: token},
        )
        holders = [asyncio.create_task(client.get(f"/hold/{i}")) for i in range(32)]
        try:
            await asyncio.wait_for(started.wait(), 5)
            queued_req = asyncio.create_task(client.get("/queued-after-revoke"))
            await asyncio.sleep(0.1)
            gw.clear_public_url()
            resp = await asyncio.wait_for(queued_req, 5)
            assert resp.status_code == 401
            release.set()
            await asyncio.wait_for(asyncio.gather(*holders, return_exceptions=True), 5)
            assert "/queued-after-revoke" not in seen_paths
        finally:
            release.set()
            await client.aclose()
            await gw.aclose()
            server.close()
            await server.wait_closed()

        # 2) Active streaming response stops and closes upstream immediately on revocation
        closed_flag = False
        revoked_ev = asyncio.Event()
        sent_events: list[dict] = []

        async def _chunks():
            yield b"first"
            await asyncio.sleep(30)
            yield b"never"

        async def _close():
            nonlocal closed_flag
            closed_flag = True

        stream_resp = sidecar._ProxyStreamResponse(
            200, [(b"content-type", b"text/event-stream")], _chunks(), _close,
            is_valid=lambda: not revoked_ev.is_set(), wait_invalid=revoked_ev.wait,
        )

        async def _send(msg):
            sent_events.append(msg)
            if msg.get("body") == b"first":
                revoked_ev.set()

        async def _recv():
            await asyncio.sleep(30)
            return {"type": "http.disconnect"}

        await asyncio.wait_for(stream_resp({}, _recv, _send), 5)
        assert closed_flag is True
        assert [e.get("body") for e in sent_events if e["type"] == "http.response.body"] == [b"first"]

    asyncio.run(_scenario())


# --- Group 8: Strictly passive telegram_i18n (zero shared install writes) ---


def test_group8_i18n_passive_read_only_no_shared_writes(tmp_path, monkeypatch):
    plugin = _load_plugin("tg_sec_g8")
    i18n = plugin.telegram_i18n
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    i18n.configure(data_dir)
    monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", "fr")
    texts = plugin._LOCALIZED_TEXTS["fr"]
    assert texts["btn_metrics"] == "📉 Status & Metrics"
    assert i18n.host_sentence("Fallback sentence.", "fr") == "Fallback sentence."
    assert list(data_dir.rglob("*")) == []  # ZERO writes to shared installation state!


def test_round14_adversarial_regressions(tmp_path, monkeypatch):
    import concurrent.futures
    import stat

    plugin = _load_plugin("tg_sec_r14")
    from tg_sec_r14.lib import telegram_outbox, telegram_owner_auth, telegram_presence_bridge, telegram_quiz, telegram_secret_guard as sg
    from tg_sec_r14.lib.presence import custody, file_access, host, outbox_files, outbox_worker, tools, transport
    from tg_sec_r14.scripts.telegram_settings import merge_settings
    from ouroboros.server_control import PanicIngress, dispatch_accepted_restart, is_exact_panic_command
    from supervisor.message_bus import LocalChatBridge

    # 1A) Outbox admission binds owner & api_base; changing owner stops resumed delivery (never delivers to successor)
    api = _StubApi(tmp_path / "r14_state")
    merge_settings(tmp_path / "r14_state", {"TELEGRAM_CHAT_ID": "18"})
    ob = telegram_outbox.Outbox(api)
    rec = ob._new_record("outbound", {"text": "private to 18"}, 1, "pending")
    assert rec["chat_id"] == 18 and rec["api_base"] == "https://api.telegram.org"
    quiz_rec = asyncio.run(ob._custody("quiz", {"task_id": "t1", "quiz_id": "q1", "secret_request": {"field": "tok"}}, 2))
    assert quiz_rec is not None and quiz_rec.get("secret") is True

    # 1B) Concurrent pairing redemption: exactly 1 of 20 threads wins; corrupted expires_at fails closed;
    #     /pair and /start <code> are consumed locally on both owner and non-owner lanes (never enter Presence or Main).
    pair_api = _StubApi(tmp_path / "r14_pair")
    code = telegram_owner_auth.issue_owner_pairing_code(pair_api)["code"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(lambda i: telegram_owner_auth.redeem_owner_pairing_code(pair_api, 100 + i, f"/pair {code}"), range(20)))
    assert sum(1 for ok in results if ok) == 1
    assert telegram_owner_auth.is_pairing_command(pair_api, {"chat": {"id": -1001, "type": "supergroup"}, "text": f"/pair {code}"}) is True
    assert telegram_owner_auth.is_pairing_command(pair_api, {"chat": {"id": 100, "type": "private"}, "text": f"/start {code}"}) is True
    # Tampered expires_at fails closed without raising ValueError
    pair_api2 = _StubApi(tmp_path / "r14_pair2")
    c2 = telegram_owner_auth.issue_owner_pairing_code(pair_api2)["code"]
    (tmp_path / "r14_pair2" / "owner_pairing.json").write_text(json.dumps({"code": c2, "expires_at": "not-a-float", "failed_attempts": 0}))
    assert telegram_owner_auth.redeem_owner_pairing_code(pair_api2, 42, f"/pair {c2}") is False

    # 2) Secret guard: custom api_base lookup still blocks; channel_post & user forward blocked; multi-part mid=0 marks uncertain; quiz_state 0o600
    sg.record_secret_message(api, "123456", 18, 501, api_base="https://api.telegram.org", kind="card", task_id="t1", quiz_id="q1")
    sg.record_secret_message(api, "123456", 18, 502, api_base="https://api.telegram.org", kind="refused_reply", task_id="t1", quiz_id="q1")
    assert sg.classify_secret_exposure(api, "123456", {"chat": {"id": 18}, "message_id": 503, "reply_to_message": {"message_id": 501}}, api_base="http://127.0.0.1:8081") is not None
    # User-origin forward (no message_id) from chat with refused_reply fails closed
    assert sg.classify_secret_exposure(api, "123456", {"chat": {"id": -10099}, "message_id": 9, "forward_origin": {"type": "user", "sender_user": {"id": 18}, "date": 1700000000}}) is not None
    # channel_post referencing secret is blocked BEFORE Presence ingest
    ingested: list = []
    fake_rt = types.SimpleNamespace(ingest_update=lambda u, **k: asyncio.sleep(0, result=ingested.append(u)))
    fake_cl = types.SimpleNamespace(bot_id="123456", api_base_origin="http://127.0.0.1:8081")
    asyncio.run(telegram_presence_bridge.handle_non_owner_update(
        api, fake_cl, {"update_id": 1, "channel_post": {"chat": {"id": -10077, "type": "channel"}, "message_id": 10, "forward_origin": {"type": "channel", "chat": {"id": 18}, "message_id": 501}}},
        "18", runtime_getter=lambda _a: fake_rt,
    ))
    assert ingested == []
    # Multi-part secret send where chunk 1 returns mid=0 keeps 0 in sent_ids so secret_confirmed is False
    part_results = iter([0, 200, 201])
    fake_quiz_cl = types.SimpleNamespace(
        send_message=lambda *a, **k: asyncio.sleep(0, result=next(part_results)),
        send_message_with_inline_keyboard=lambda *a, **k: asyncio.sleep(0, result=next(part_results)),
    )
    sids: list[int] = []
    asyncio.run(telegram_quiz.send_quiz_card(fake_quiz_cl, 18, body="x" * 5000, compact="c", hint_text="h", keyboard=[], sent_ids=sids))
    assert 0 in sids
    telegram_quiz.remember_quiz(api, "123456", "tok1", {"task_id": "t1", "quiz_id": "q1", "chat_id": 18, "message_id": 200, "options": [], "text": "Q"})
    assert stat.S_IMODE((tmp_path / "r14_state" / "quiz_state.json").stat().st_mode) == 0o600

    # 3) Exact /panic & literal_text end-to-end across PanicIngress, dispatch_accepted_restart, message_bus, quiz replies, and _validate_bot
    assert is_exact_panic_command("/panic") is True and is_exact_panic_command("/panicking") is False
    stopped: list[str] = []
    bus = LocalChatBridge()
    bus.panic = PanicIngress(lambda: stopped.append("PANIC"))
    bus.panic._owner = [True, (18, 18)]
    bus.enqueue_local_message("/panicking right now", source="skill:telegram", user_id=18, chat_id=18)
    bus.enqueue_local_message("/panic", source="skill:telegram", user_id=18, chat_id=18, task_metadata={"literal_text": True})
    assert stopped == []
    restarted: list[str] = []
    dispatch_accepted_restart(bus, "/restart", callback=lambda *a, **k: (lambda: restarted.append("R")), source="skill:telegram", user_id=18, chat_id=18, task_metadata={"literal_text": True})
    assert restarted == []
    # /main + code `/panic` marks inner span as literal
    main_code_msg = {"text": "/main /panic", "entities": [{"type": "bot_command", "offset": 0, "length": 5}, {"type": "code", "offset": 6, "length": 6}]}
    assert telegram_owner_auth.is_panic_command(main_code_msg) is False
    assert telegram_owner_auth.is_literal_span(main_code_msg, "/panic") is True
    # _validate_bot makes ONLY getMe (zero blocking setMyCommands ahead of getUpdates)
    val_calls: list[str] = []
    val_cl = types.SimpleNamespace(bot_id="123456", token="123456:TOK_A", api_base_origin="https://api.telegram.org",
                                   call=lambda m, **k: (val_calls.append(m), asyncio.sleep(0, result={"ok": True, "result": {"id": 123456, "username": "bot"}}))[1])
    asyncio.run(plugin._validate_bot(api, val_cl, "full_access", "en"))
    assert val_calls == ["getMe"]

    # 4) HTML-400 fallback PlaceHeld clears CALL_STARTED; eventless focus receipt isolation; reaction key whitelist
    assert outbox_worker.not_performed(outbox_worker.PlaceHeld("paused")) is True
    assert outbox_worker.not_performed(host.PlaceBindingRevoked("revoked")) is True
    focus_ctx = types.SimpleNamespace(task_id="task-other", task_metadata={"focus": {"place_id": "place-1"}})
    row_payload = {"account_id": "123456", "chat_id": "-1001", "topic_id": "99", "_caller_scope": {"task_id": "task-owner", "place_id": "place-1"}}
    assert file_access.receipt_visible_to(focus_ctx, row_payload) is False
    assert tools._reaction_values_or_refuse([{"type": "emoji", "emoji": "👍", "evil_extra": 1}]) == [{"type": "emoji", "emoji": "👍"}]

    # 5) Zero-TOCTOU verified fd continuity: unlinking/swapping file on disk after open_verified still streams original bytes!
    src = tmp_path / "orig.txt"
    src.write_bytes(b"ORIGINAL_VERIFIED_BYTES")
    frozen = outbox_files.keep(src, tmp_path / "r14_state", 1024)
    p_payload = {"kind": "document", "file_path": frozen["path"], "_frozen": [frozen]}
    vfds = outbox_files.open_verified(p_payload, state_dir=tmp_path / "r14_state", require_for_media=True)
    try:
        # Swap on-disk frozen file AFTER verification — stream MUST still transmit ORIGINAL_VERIFIED_BYTES from verified fd!
        pathlib.Path(frozen["path"]).unlink()
        pathlib.Path(frozen["path"]).write_bytes(b"ATTACKER_SWAPPED_BYTES!")
        stream = transport._AsyncMultipartStream.open({"chat_id": "1"}, [("document", pathlib.Path(frozen["path"]))], vfds)
        async def _drain():
            chunks = [c async for c in stream]
            return b"".join(chunks)
        body = asyncio.run(_drain())
        assert b"ORIGINAL_VERIFIED_BYTES" in body and b"ATTACKER_SWAPPED_BYTES" not in body
    finally:
        outbox_files.close_verified(vfds)

    # 6) Scoped lease recovery preserves live registered owner leases & releases retired owner leases
    st_store = custody.CustodyStore(tmp_path / "r14_leases.sqlite3")
    st_store.register_owner()
    st_store.enqueue_outbox("d-live", {"kind": "message", "chat_id": "1", "text": "hi"})
    live_lease = st_store.claim_outbox(lease_sec=120.0)
    assert live_lease is not None
    assert st_store.release_own_leases()["outbox"] == 0  # Live registered owner lease is NOT stolen!
    st_store.unregister_owner()
    assert st_store.release_own_leases()["outbox"] == 1  # Retired owner lease IS released!
    st_store.close()


def test_round15_adversarial_regressions(tmp_path, monkeypatch):
    import hashlib, inspect, os

    plugin = _load_plugin("tg_sec_r15")
    from tg_sec_r15.lib import telegram_api, telegram_outbox, telegram_owner_auth, telegram_presence_bridge, telegram_secret_guard as sg
    from tg_sec_r15.lib.presence import custody, custody_schema, events, file_access, transport
    from tg_sec_r15.scripts import platform_support, telegram_menu
    from tg_sec_r15.scripts.telegram_settings import merge_settings

    # 1) Full canonical (scheme, host, port) origin check + _ACTIVE parts failed on "changed"
    st_dir = tmp_path / "r15_origin"
    api = _StubApi(st_dir)
    merge_settings(st_dir, {"TELEGRAM_CHAT_ID": "18", "TELEGRAM_API_BASE": "http://localhost:8082"})
    assert not telegram_outbox._same_api_authority(st_dir, "http://localhost:8081")
    assert telegram_outbox._same_api_authority(st_dir, "http://localhost:8082") and not telegram_outbox._same_api_authority(st_dir, "http://localhost:80")
    ob = telegram_outbox.Outbox(api)
    ob._bodies["outbound"] = lambda _ev: asyncio.sleep(0)
    rec = {"schema": 1, "id": "000000000001-deadbeef", "seq": 1, "topic": "outbound", "created_at": time.time(),
           "state": "retry_wait", "parts": [{"n": 0, "state": "retry_wait"}], "bot": "123456",
           "chat_id": 18, "api_base": "http://localhost:8081", "next_at": 0.0, "retry_delay": 5.0, "reason": ""}
    ob._ops[rec["id"]] = rec
    ob._written.add(rec["id"])
    asyncio.run(ob._resume(rec, ahead=True))
    assert (rec["state"], rec["reason"], rec["parts"][0]["state"]) == ("failed", "changed", "failed")

    # 2) Secret quiz _Probe (_WouldSend) & OutboxDeferred never poison secret_uncertain; clear_uncertain_secret clears
    q_api = _StubApi(tmp_path / "r15_quiz")
    merge_settings(tmp_path / "r15_quiz", {"TELEGRAM_CHAT_ID": "18"})
    q_handler = plugin._make_quiz(q_api, plugin._typing_cue(q_api))
    q_ev = {"task_id": "t1", "quiz_id": "q1", "question": "Enter token", "options": [], "transport": {},
            "secret_request": {"plugin_name": "p", "fields": [{"key": "tok"}]}}
    tok_ctx = telegram_api.OUTBOUND_JOURNAL.set(telegram_outbox._Probe())
    try:
        with pytest.raises(telegram_outbox._WouldSend): asyncio.run(q_handler(q_ev))
    finally:
        telegram_api.OUTBOUND_JOURNAL.reset(tok_ctx)
    assert sg.has_uncertain_secret(q_api, "123456", 18) is None
    monkeypatch.setattr(plugin.telegram_quiz, "send_quiz_card", lambda *a, **k: (_ for _ in ()).throw(telegram_outbox.OutboxDeferred()))
    with pytest.raises(telegram_outbox.OutboxDeferred): asyncio.run(q_handler(q_ev))
    assert sg.has_uncertain_secret(q_api, "123456", 18) is None
    sg.mark_secret_uncertain(q_api, "123456", 18, reason="test")
    assert sg.has_uncertain_secret(q_api, "123456", 18) is not None
    sg.clear_uncertain_secret(q_api, "123456", 18)
    assert sg.has_uncertain_secret(q_api, "123456", 18) is None

    # 3) Code-formatted /mode & /menu do not run locally; /main /main start offset; /m@x/restart not stripped
    assert plugin._MAIN_CMD_RE.match("/m@x/restart") is None
    mm_msg = {"text": "/main /main", "entities": [{"type": "code", "offset": 6, "length": 5}]}
    assert telegram_owner_auth.is_literal_span(mm_msg, "/main", start=0) is False and telegram_owner_auth.is_literal_span(mm_msg, "/main", start=6) is True
    mode_api = _StubApi(tmp_path / "r15_mode")
    merge_settings(tmp_path / "r15_mode", {"TELEGRAM_CHAT_ID": "18", "TELEGRAM_OWNER_CHAT_MODE": "direct"})
    sent_local, injected_local = [], []
    fake_cl = types.SimpleNamespace(bot_id="123456", api_base_origin="https://api.telegram.org",
                                    send_message=lambda cid, txt, **k: asyncio.sleep(0, result=sent_local.append(txt) or 1),
                                    send_message_with_inline_keyboard=lambda cid, txt, kb, **k: asyncio.sleep(0, result=sent_local.append(txt) or 1))
    monkeypatch.setattr(plugin, "_inject", lambda _a, p: asyncio.sleep(0, result=injected_local.append(p)))
    for raw_cmd in ("/mode presence", "/menu"):
        upd = {"update_id": 1, "message": {"chat": {"id": 18, "type": "private"}, "from": {"id": 18}, "message_id": 10,
                                           "text": raw_cmd, "entities": [{"type": "code", "offset": 0, "length": len(raw_cmd)}]}}
        asyncio.run(plugin._handle_owner_message_inner(mode_api, fake_cl, upd, 1, "strict", "en"))
    assert sent_local == [] and len(injected_local) == 2 and plugin._load_settings(mode_api)["TELEGRAM_OWNER_CHAT_MODE"] == "direct"
    assert all(p.get("literal_text") is True for p in injected_local)

    # 4) external_reply user/hidden_user referencing refused_reply fails closed before Presence ingest
    sg.record_secret_message(api, "123456", 18, 701, kind="refused_reply")
    for origin in ({"type": "user", "sender_user": {"id": 18}}, {"type": "hidden_user", "sender_user_name": "Anon"}):
        ext_msg = {"chat": {"id": -10055}, "message_id": 44, "external_reply": {"origin": origin}, "quote": {"text": "sk-123"}}
        assert sg.classify_secret_exposure(api, "123456", ext_msg) is not None
        got = []
        rt = types.SimpleNamespace(ingest_update=lambda u, **k: asyncio.sleep(0, result=got.append(u)))
        asyncio.run(telegram_presence_bridge.handle_non_owner_update(api, fake_cl, {"update_id": 2, "message": ext_msg}, "18", runtime_getter=lambda _a: rt))
        assert got == []

    # 5) Distinct colon-free staging directories per lease + fail-closed verified_fds + Windows-safe _pid_alive
    cstore = custody.CustodyStore(tmp_path / "r15_c.sqlite3")
    ev = events.parse_telegram_update({"update_id": 1, "message": {"chat": {"id": -100, "type": "supergroup", "title": "G"}, "from": {"id": 9, "first_name": "U"}, "message_id": 1, "date": 1700000000, "text": "hi"}}, bot_account_id="123456", places=True)
    assert ev is not None and cstore.commit_update(1, ev)
    l1 = cstore.claim_inbox(lease_sec=10.0)
    assert l1 and cstore.release_inbox(ev.source_event_id, reason="retry", retry_after_sec=0)
    l2 = cstore.claim_inbox(lease_sec=10.0)
    assert l2 and l1.lease_token != l2.lease_token and l1.lease_token[:16] == l2.lease_token[:16]
    d1, d2 = hashlib.sha256(l1.lease_token.encode()).hexdigest()[:16], hashlib.sha256(l2.lease_token.encode()).hexdigest()[:16]
    assert d1 != d2 and ":" not in d1 and ":" not in d2
    cstore.close()
    assert "process_alive" in inspect.getsource(custody_schema._pid_alive) and 'os.name == "nt"' in inspect.getsource(custody_schema._pid_alive)
    assert custody_schema._pid_alive(os.getpid()) is True and platform_support.process_alive(os.getpid()) is True
    tmp_up = tmp_path / "f.bin"
    tmp_up.write_bytes(b"abc")
    with pytest.raises(transport.TelegramRequestNotSent, match="verified upload descriptor missing"):
        transport._AsyncMultipartStream.open({}, [("document", tmp_up)], verified_fds={})

    # 6) Positive binding_id in receipt_visible_to + non-ASCII/NaN/group-burn pairing + menu getMe + Host /panic
    monkeypatch.setattr("ouroboros.presence_authority.presence_caller_binding", lambda c: getattr(c, "_bid", None))
    c_b1 = types.SimpleNamespace(task_id="t1", _bid="bind-1", task_metadata={"presence": {"place_id": "p1", "event": {"provider": "telegram", "account_id": "123", "conversation_id": "-100", "thread_id": ""}}})
    c_b2 = types.SimpleNamespace(task_id="t1", _bid="bind-2", task_metadata=c_b1.task_metadata)
    p_row = {"account_id": "123", "chat_id": "-100", "topic_id": "", "_caller_scope": file_access.caller_scope(c_b1)}
    assert p_row["_caller_scope"]["binding_id"] == "bind-1" and file_access.receipt_visible_to(c_b1, p_row) is True
    assert file_access.receipt_visible_to(c_b2, p_row) is False and file_access.receipt_visible_to(c_b1, {**p_row, "_caller_scope": {}}) is False
    p_api = _StubApi(tmp_path / "r15_pair")
    pcode = telegram_owner_auth.issue_owner_pairing_code(p_api)["code"]
    assert telegram_owner_auth.redeem_owner_pairing_code(p_api, 18, "/pair ñ") is False
    assert telegram_owner_auth.is_pairing_command(p_api, {"chat": {"id": -100, "type": "supergroup"}, "text": f"/pair {pcode}"}) is True
    assert not (tmp_path / "r15_pair" / "owner_pairing.json").exists()
    (tmp_path / "r15_pair" / "owner_pairing.json").write_text(json.dumps({"code": pcode, "expires_at": float("inf"), "failed_attempts": 0}))
    assert telegram_owner_auth.redeem_owner_pairing_code(p_api, 18, f"/pair {pcode}") is False
    mclient = telegram_menu.TelegramMenuManager("123456:TOK", 18, tmp_path / "r15_menu")
    with pytest.raises(telegram_menu.TelegramMenuRejectedError, match="does not match"):
        mclient._verify_me_identity({"id": 999999})
    assert 'startswith("/panic")' not in (_REPO_ROOT / "ouroboros" / "gateway" / "control.py").read_text(encoding="utf-8")
    assert 'startswith("/panic")' not in (_REPO_ROOT / "ouroboros" / "gateway" / "ws.py").read_text(encoding="utf-8")
