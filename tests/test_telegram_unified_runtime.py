"""End-to-end runtime & loader regressions for the unified built-in Telegram skill (v1.5.0)."""

from __future__ import annotations

import asyncio, contextlib, json, sqlite3, types
from pathlib import Path
from typing import Any, Dict, List
import pytest

from ouroboros import extension_loader
from ouroboros.extension_process_runner import _handler_wants_ctx
from ouroboros.skill_builtin import install_builtin_skill
from ouroboros.skill_loader import find_skill, save_enabled
from ouroboros.tools.extension_dispatch import _dispatch_extension_tool_result
from skills.telegram.lib.presence.custody import CustodyStore
from skills.telegram.lib.presence.outbox_worker import MAX_OUTBOX_ATTEMPTS
from skills.telegram.lib.presence.runtime import PresenceRuntime
from skills.telegram.lib.presence.tools import register_presence_tools
from skills.telegram.lib.presence.transport import TelegramApiError
from skills.telegram.lib.telegram_mirror import (
    MIRROR_MODE_ALL, MIRROR_MODE_CONTROL_ONLY, MIRROR_MODE_MAIN, MIRROR_MODE_TELEGRAM_ONLY, target_chat,
)
from skills.telegram.lib.telegram_poll_binding import PollCursor, update_key

_REPO_ROOT = Path(__file__).resolve().parents[1]


class _StubApi:
    def __init__(self, state_dir: Path, token: str = "123456:TOK_A"):
        self.state_dir, self.token, self.tools, self.logs = state_dir, token, {}, []
    def get_state_dir(self) -> str: return str(self.state_dir)
    def get_settings(self, keys): return {k: self.token for k in keys if k == "TELEGRAM_BOT_TOKEN" and self.token}
    def register_tool(self, name: str, fn, *, description: str = "", schema: dict | None = None) -> None: self.tools[name] = fn
    def log(self, level: str, message: str) -> None: self.logs.append((level, message))


class _StubSubmitter:
    def __init__(self):
        self.places_version = self.delivery_reporting_version = self.presence_wake_signals = 1
        self.presence_identity = self.presence_reaction_context = self.presence_explicit_completion = 1
        self.delivery_reporting_status, self.continuation_version = "supported", 2
        self.connector_identity: Dict[str, Any] = {}
        self.reports: List[dict] = []
    async def discover_delivery_support(self) -> int: return 1
    async def place_hold(self, account_id: str, conversation_id: str, thread_id: str = "") -> str: return ""
    async def report_delivery(self, payload: Dict[str, Any]) -> None: self.reports.append(dict(payload))


class _StubBotClient:
    download_limit = 20 * 1024 * 1024
    def __init__(self):
        self.sent: List[tuple[str, str]] = []
        self.typing: List[tuple[str, str, int | None]] = []
        self.fail_429_remaining = 0
    async def send_text_chunk(self, chat_id: str, text: str, **kwargs) -> Dict[str, Any]:
        if self.fail_429_remaining > 0:
            self.fail_429_remaining -= 1
            raise TelegramApiError("sendMessage", "Too Many Requests", 429, retry_after=40)
        self.sent.append((chat_id, text))
        return {"message_id": len(self.sent), "chat": {"id": int(chat_id), "type": "private"}, "text": text}
    async def send_chat_action(self, chat_id: str, action: str = "typing", *, topic_id: int | None = None) -> bool:
        self.typing.append((chat_id, action, topic_id))
        return True


def test_real_extension_loader_registers_tools_routes_settings_and_unloads_cleanly(tmp_path, monkeypatch):
    drive_root = tmp_path / "data"
    drive_root.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(drive_root))
    monkeypatch.setenv("OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS", "true")
    monkeypatch.setenv("OUROBOROS_AUTO_GRANT_TRUSTED_SKILLS", "true")
    monkeypatch.setattr("ouroboros.config.SETTINGS_PATH", drive_root / "settings.json")
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_kw: (True, ""))
    monkeypatch.setattr("ouroboros.extension_loader.is_extension_live", lambda *_a, **_kw: True)
    # Keep loader test free of spawning any real companion process or network calls
    monkeypatch.setattr("ouroboros.extension_plugin_api.is_server_process", lambda: False)

    res_install = install_builtin_skill("telegram", drive_root=drive_root, seed_dir=_REPO_ROOT / "skills")
    assert res_install["ok"] is True
    save_enabled(drive_root, "telegram", True)
    skill = find_skill(drive_root, "telegram")
    assert skill is not None
    err = extension_loader.load_extension(skill, lambda: {}, drive_root=drive_root, _force_in_process=True)
    assert err is None
    try:
        tools = {name: desc for name, desc in extension_loader._tools.items() if desc.get("skill") == "telegram"}
        short_names = {name.rsplit("telegram_", 1)[-1]: desc for name, desc in tools.items()}
        assert set(short_names) == {"send", "moderate", "operation", "react", "typing", "receipt"}
        assert all(short_names[k].get("wants_ctx") is True for k in short_names)
        routes = {r.get("path") if isinstance(r, dict) else str(r) for r in extension_loader.list_routes() if (r.get("skill") == "telegram" if isinstance(r, dict) else "/telegram/" in str(r))}
        assert {"/api/extensions/telegram/status", "/api/extensions/telegram/settings/save", "/api/extensions/telegram/miniapp/status", "/api/extensions/telegram/miniapp/prepare-share", "/api/extensions/telegram/connection", "/api/extensions/telegram/connection/copies"} <= routes
        assert any((sec.get("skill") == "telegram" if isinstance(sec, dict) else "telegram" in str(key)) for key, sec in extension_loader._settings_sections.items())
        receipt_name = next(name for name in tools if name.endswith("telegram_receipt"))
        receipt_desc = extension_loader.get_tool(receipt_name)
        assert receipt_desc is not None
        res = _dispatch_extension_tool_result(None, receipt_name, receipt_desc, {"request_id": "missing-req", "operation": "send"})
        assert res.status == "ok" and json.loads(res.text)["state"] == "not_found"
    finally:
        extension_loader.unload_extension("telegram")

    assert not [d for d in extension_loader._tools.values() if d.get("skill") == "telegram"]
    assert not [r for r in extension_loader.list_routes() if isinstance(r, dict) and r.get("skill") == "telegram"]


def test_bot_swap_binds_turn_account_refuses_unconfirmed_gap_and_fences_typing(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    api = _StubApi(state_dir, token="")
    submitter = _StubSubmitter()
    bot_client = _StubBotClient()
    rt = PresenceRuntime(api=api, state_dir=state_dir, submitter=submitter, client_factory=lambda: bot_client)
    register_presence_tools(api, rt)

    # 1) During credential gap (no confirmed bot, no turn account), send/moderate/operation/group-react refuse at admission
    send_gap = json.loads(api.tools["telegram_send"](chat_id="7001", kind="message", text="gap", request_id="gap-1"))
    assert send_gap["ok"] is False and send_gap["code"] == "bot_unconfirmed"
    mod_gap = json.loads(api.tools["telegram_moderate"](action="ban_member", chat_id="-1001", user_id=9, request_id="m-gap"))
    assert mod_gap["ok"] is False and mod_gap["code"] == "bot_unconfirmed"
    grp_react_gap = json.loads(api.tools["telegram_react"](action="clear", chat_id="-100555", message_id=9, request_id="r-gap"))
    assert grp_react_gap["ok"] is False and grp_react_gap["code"] == "bot_unconfirmed"
    assert rt.store.outbox_payload("telegram-send:gap-1") == {}

    # 2) Validate Bot A (123456), start live Presence turn, queue typing for Bot A
    api.token = "123456:TOK_A"
    rt.on_bot_validated("123456:TOK_A", "123456", "bot_a")
    upd_a = {"update_id": 1, "message": {"message_id": 1, "date": 1_900_000_000, "chat": {"id": 7001, "type": "private", "first_name": "Mira"}, "from": {"id": 7001, "is_bot": False, "first_name": "Mira"}, "text": "hello"}}
    assert asyncio.run(rt.ingest_update(upd_a, account_id="123456")) is True
    lease = rt.store.claim_inbox()
    assert lease is not None
    turn_ctx = types.SimpleNamespace(task_id="turn-a", task_metadata={"presence": {"event": {"provider": "telegram", "account_id": "123456", "conversation_id": "7001", "thread_id": "", "source_event_id": lease.event_id}}})
    typ_res = json.loads(api.tools["telegram_typing"](turn_ctx, chat_id="7001"))
    assert typ_res["ok"] is True

    # 3) Bot swaps to Bot B (987654) while Bot A's turn is still live!
    api.token = "987654:TOK_B"
    rt.on_token_changed()
    rt.on_bot_validated("987654:TOK_B", "987654", "bot_b")

    # Pre-swap typing MUST NOT sendChatAction under Bot B (even if re-inserted directly with Bot A's account_id)
    rt.store.request_typing(chat_id="7001", topic_id="", turn_ref="turn-a", source_event_id=lease.event_id, account_id="123456")
    assert asyncio.run(rt.process_typing()) == 0
    assert bot_client.typing == []

    # Bot A's live turn calling telegram_send MUST stamp account_id="123456" and be held, NEVER sent under Bot B!
    res = json.loads(api.tools["telegram_send"](turn_ctx, chat_id="7001", kind="message", text="from A turn", request_id="swap-1"))
    assert res["ok"] is True
    assert rt.store.outbox_payload("telegram-send:swap-1")["account_id"] == "123456"
    assert asyncio.run(rt.process_one_outbox()) is True
    assert bot_client.sent == []
    assert rt.store.delivery_receipt("telegram-send:swap-1")["state"] == "held"


def test_outbox_holds_on_unvalidated_token_and_defers_429_without_spending_attempts(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    api = _StubApi(state_dir, token="123456:TOK_A")
    submitter = _StubSubmitter()
    bot_client = _StubBotClient()
    bot_client.fail_429_remaining = MAX_OUTBOX_ATTEMPTS + 2  # more than MAX_OUTBOX_ATTEMPTS!
    rt = PresenceRuntime(api=api, state_dir=state_dir, submitter=submitter)
    rt.store.enqueue_outbox(
        "telegram-send:boot-wait",
        {"kind": "message", "chat_id": "7001", "text": "queued at boot", "account_id": "123456", "markdown": False},
    )

    # 1) Unvalidated token at boot holds without consuming MAX_OUTBOX_ATTEMPTS
    for _ in range(MAX_OUTBOX_ATTEMPTS + 2):
        with rt.store._connect() as conn:
            conn.execute("UPDATE outbox SET available_at=0 WHERE delivery_id='telegram-send:boot-wait'")
        assert asyncio.run(rt.process_one_outbox()) is True
    rec_boot = rt.store.delivery_receipt("telegram-send:boot-wait")
    assert rec_boot["state"] == "held" and rec_boot["attempts"] == 0

    # 2) Once validated, repeated 429 flood responses defer by retry_after without burning attempts, then deliver
    rt.on_bot_validated("123456:TOK_A", "123456", "bot_a")
    rt._custom_client_factory = lambda: bot_client
    for _ in range(MAX_OUTBOX_ATTEMPTS + 2):
        with rt.store._connect() as conn:
            conn.execute("UPDATE outbox SET available_at=0 WHERE delivery_id='telegram-send:boot-wait'")
        assert asyncio.run(rt.process_one_outbox()) is True
    rec_429 = rt.store.delivery_receipt("telegram-send:boot-wait")
    assert rec_429["state"] == "pending" and rec_429["attempts"] == 0
    with rt.store._connect() as conn:
        row = conn.execute("SELECT available_at FROM outbox WHERE delivery_id='telegram-send:boot-wait'").fetchone()
        assert float(row["available_at"]) > 0
        conn.execute("UPDATE outbox SET available_at=0 WHERE delivery_id='telegram-send:boot-wait'")
    assert asyncio.run(rt.process_one_outbox()) is True
    assert rt.store.delivery_receipt("telegram-send:boot-wait")["state"] == "delivered"
    assert bot_client.sent == [("7001", "queued at boot")]


def test_worker_survives_sqlite_lock_and_run_preserves_provider_error_during_discovery(tmp_path, monkeypatch):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    api = _StubApi(state_dir, token="123456:TOK_A")
    bot_client = _StubBotClient()

    class _SlowDiscoverSubmitter(_StubSubmitter):
        def __init__(self):
            super().__init__(); self.in_discover, self.release_discover = asyncio.Event(), asyncio.Event()
        async def discover_delivery_support(self) -> int:
            self.in_discover.set(); await self.release_discover.wait(); return 1
    submitter = _SlowDiscoverSubmitter()
    rt = PresenceRuntime(api=api, state_dir=state_dir, submitter=submitter, client_factory=lambda: bot_client)
    async def _run_race():
        task = asyncio.create_task(rt.run())
        await asyncio.wait_for(submitter.in_discover.wait(), timeout=5.0)
        rt.on_token_rejected("123456:TOK_A", "Unauthorized"); submitter.release_discover.set(); await asyncio.sleep(0.05)
        observed_state = rt._runtime_state; rt.stop()
        with contextlib.suppress(asyncio.CancelledError): await asyncio.wait_for(task, timeout=5.0)
        return observed_state
    assert asyncio.run(_run_race()) == "provider_error"
    rt2 = PresenceRuntime(api=api, state_dir=state_dir, submitter=_StubSubmitter(), client_factory=lambda: bot_client)
    rt2.on_bot_validated("123456:TOK_A", "123456", "bot_a")
    rt2.store.enqueue_outbox("telegram-send:after-lock", {"kind": "message", "chat_id": "7001", "text": "survived lock", "account_id": "123456", "markdown": False})
    orig_claim, fail_once = rt2.store.claim_outbox, {"flag": True}
    def _flaky_claim(*a, **kw):
        if fail_once["flag"]: fail_once["flag"] = False; raise sqlite3.OperationalError("database is locked")
        return orig_claim(*a, **kw)
    monkeypatch.setattr(rt2.store, "claim_outbox", _flaky_claim)
    async def _run_lock_recovery():
        task = asyncio.create_task(rt2.run())
        for _ in range(40):
            if rt2.store.delivery_receipt("telegram-send:after-lock")["state"] == "delivered": break
            await asyncio.sleep(0.1)
        rt2.stop()
        with contextlib.suppress(asyncio.CancelledError): await asyncio.wait_for(task, timeout=5.0)
    asyncio.run(_run_lock_recovery())
    assert rt2.store.delivery_receipt("telegram-send:after-lock")["state"] == "delivered"


def test_mirror_excludes_presence_provenance_and_poll_cursor_distinguishes_media_edits(tmp_path):
    # 1) Assistant reply with presence_provenance (and no numeric chat_id/task_type) is excluded in ALL 4 mirror modes
    presence_reply = {
        "role": "assistant",
        "text": "Presence reply",
        "transport": {"kind": "telegram", "conversation_id": "42"},
        "presence_provenance": {"binding_id": "b1", "place_id": "p1"},
    }
    for mode in (MIRROR_MODE_ALL, MIRROR_MODE_MAIN, MIRROR_MODE_TELEGRAM_ONLY, MIRROR_MODE_CONTROL_ONLY):
        assert target_chat({"TELEGRAM_CHAT_ID": "42", "TELEGRAM_MIRROR_MODE": mode}, presence_reply) == 0

    # Explicit Presence-to-Main contact notice still mirrors even without chat_id
    contact_notice = {
        "role": "system",
        "system_type": "main_notice",
        "task_type": "presence",
        "presence_provenance": {"binding_id": "b1"},
        "presence_contact": {"place_id": "p1"},
        "text": "Need owner decision",
    }
    assert target_chat({"TELEGRAM_CHAT_ID": "42", "TELEGRAM_MIRROR_MODE": MIRROR_MODE_CONTROL_ONLY}, contact_notice) == 42

    # 2) Same-second edited_message with changed photo file_id and empty caption MUST NOT be treated as replay
    cursor = PollCursor(tmp_path / "cursor.json", "https://api.telegram.org", "123456")
    edit_a = {"update_id": 11, "edited_message": {"message_id": 77, "date": 1_900_000_000, "edit_date": 1_900_000_005, "chat": {"id": -100555, "type": "supergroup"}, "photo": [{"file_id": "photo_v1", "file_unique_id": "u1", "width": 100, "height": 100}]}}
    edit_b = {"update_id": 12, "edited_message": {"message_id": 77, "date": 1_900_000_000, "edit_date": 1_900_000_005, "chat": {"id": -100555, "type": "supergroup"}, "photo": [{"file_id": "photo_v2", "file_unique_id": "u2", "width": 100, "height": 100}]}}
    assert update_key(edit_a) != update_key(edit_b)
    cursor.save(12, [update_key(edit_a)])
    assert cursor.is_replay(edit_a) is True and cursor.is_replay(edit_b) is False


def test_server_renumbering_preserves_distinct_messages_and_accountless_row_does_not_block_confirmed_sends(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    api = _StubApi(state_dir, token="123456:TOK_A")
    bot_client = _StubBotClient()
    rt = PresenceRuntime(api=api, state_dir=state_dir, submitter=_StubSubmitter(), client_factory=lambda: bot_client)
    rt.on_bot_validated("123456:TOK_A", "123456", "bot_a")
    register_presence_tools(api, rt)

    # 1) Server renumbering: same update_id=17 for distinct messages (101 vs 202) MUST BOTH commit;
    # replaying message 101 under a different update_id=99 MUST deduplicate cleanly.
    msg101 = {
        "update_id": 17,
        "message": {"message_id": 101, "date": 1_900_000_000, "chat": {"id": 7001, "type": "private"}, "from": {"id": 7001, "is_bot": False}, "text": "from server 1"},
    }
    msg202 = {
        "update_id": 17,
        "message": {"message_id": 202, "date": 1_900_000_010, "chat": {"id": 7001, "type": "private"}, "from": {"id": 7001, "is_bot": False}, "text": "from server 2"},
    }
    msg101_replay = {**msg101, "update_id": 99}
    assert asyncio.run(rt.ingest_update(msg101, account_id="123456")) is True
    assert asyncio.run(rt.ingest_update(msg202, account_id="123456")) is True
    assert asyncio.run(rt.ingest_update(msg101_replay, account_id="123456")) is False
    first = rt.store.claim_inbox()
    assert first is not None and first.payload["message"]["message_id"] == 101
    rt.store.mark_inbox_failed(first.event_id, reason="done")
    second = rt.store.claim_inbox()
    assert second is not None and second.payload["message"]["message_id"] == 202

    # 2) Pre-existing accountless outbox row for chat 42 stays held and MUST NOT deadlock confirmed sends to chat 42
    rt.store.enqueue_outbox("telegram-send:legacy-no-acct", {"kind": "message", "chat_id": "42", "text": "ambiguous", "markdown": False})
    assert json.loads(api.tools["telegram_send"](chat_id="42", kind="message", text="confirmed 42", request_id="ok-42"))["ok"] is True
    assert asyncio.run(rt.process_one_outbox()) is True  # holds legacy-no-acct
    assert asyncio.run(rt.process_one_outbox()) is True  # delivers ok-42!
    assert rt.store.delivery_receipt("telegram-send:legacy-no-acct")["state"] == "held"
    assert rt.store.delivery_receipt("telegram-send:ok-42")["state"] == "delivered"
    assert bot_client.sent == [("42", "confirmed 42")]

    # 3) Restarting run() on an already-validated runtime restores _runtime_state="polling" (not stuck "stopped")
    rt._runtime_state = "stopped"

    async def _restart_check():
        task = asyncio.create_task(rt.run())
        await asyncio.sleep(0.05)
        state = rt._runtime_state
        rt.stop()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5.0)
        return state

    assert asyncio.run(_restart_check()) == "polling"


def test_r3_midflight_place_reset_command_cooldown_and_discovery_server_renumbering(tmp_path):
    import importlib.util, sys
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    api = _StubApi(state_dir, token="123456:TOK_A")
    bot_client = _StubBotClient()

    # 1) Clearing validated identity DURING await place_hold MUST hold both normal sends and reactions
    class _ResettingSubmitter(_StubSubmitter):
        def __init__(self):
            super().__init__()
            self.rt = None

        async def place_hold(self, account_id: str, conversation_id: str, thread_id: str = "") -> str:
            if self.rt is not None:
                self.rt.on_token_changed()  # simulates same-token API server swap clearing _bot_id mid-flight
            return ""

    reset_sub = _ResettingSubmitter()
    rt = PresenceRuntime(api=api, state_dir=state_dir, submitter=reset_sub, client_factory=lambda: bot_client)
    reset_sub.rt = rt
    rt.on_bot_validated("123456:TOK_A", "123456", "bot_a")
    rt.store.enqueue_outbox("telegram-send:mid-reset", {"kind": "message", "chat_id": "7001", "text": "no send", "account_id": "123456", "markdown": False})
    assert asyncio.run(rt.process_one_outbox()) is True
    assert bot_client.sent == [] and rt.store.delivery_receipt("telegram-send:mid-reset")["state"] == "held"

    rt.on_bot_validated("123456:TOK_A", "123456", "bot_a")
    rt.store.enqueue_outbox("telegram-operation:mid-react", {
        "kind": "operation", "method": "setMessageReaction", "chat_id": "7002", "account_id": "123456",
        "parameters": {"chat_id": "7002", "message_id": 1, "reaction": [{"type": "emoji", "emoji": "👍"}]},
    })
    assert asyncio.run(rt.process_one_outbox()) is True
    assert rt.store.delivery_receipt("telegram-operation:mid-react")["state"] == "held"

    # 2) Circles discovery survives Bot API server renumbering (member@900 -> kicked@7 -> replay member@1 -> rejoin@8)
    rt.on_bot_validated("123456:TOK_A", "123456", "bot_a")
    grp = {"id": -42, "type": "supergroup", "title": "Group"}
    u_join = {"update_id": 900, "my_chat_member": {"chat": grp, "date": 1_900_000_010, "old_chat_member": {"user": {"id": 123456, "is_bot": True}, "status": "left"}, "new_chat_member": {"user": {"id": 123456, "is_bot": True}, "status": "member"}}}
    u_kick = {"update_id": 7, "my_chat_member": {"chat": grp, "date": 1_900_000_020, "old_chat_member": {"user": {"id": 123456, "is_bot": True}, "status": "member"}, "new_chat_member": {"user": {"id": 123456, "is_bot": True}, "status": "kicked"}}}
    u_join_replay = {**u_join, "update_id": 1}
    u_rejoin = {"update_id": 8, "my_chat_member": {"chat": grp, "date": 1_900_000_020, "old_chat_member": {"user": {"id": 123456, "is_bot": True}, "status": "kicked"}, "new_chat_member": {"user": {"id": 123456, "is_bot": True}, "status": "member"}}}
    asyncio.run(rt.ingest_update(u_join, account_id="123456"))
    assert [g["conversation_id"] for g in rt.store.offered_groups("123456")] == ["-42"]
    asyncio.run(rt.ingest_update(u_kick, account_id="123456"))
    assert rt.store.offered_groups("123456") == []
    asyncio.run(rt.ingest_update(u_join_replay, account_id="123456"))
    assert rt.store.offered_groups("123456") == []  # replay MUST NOT resurrect kicked group!
    asyncio.run(rt.ingest_update(u_rejoin, account_id="123456"))
    assert [g["conversation_id"] for g in rt.store.offered_groups("123456")] == ["-42"]

    # 3) _sync_bot_commands cooldown applies per failed target (bot, chat) and allows immediate retry for a new chat
    pkg = types.ModuleType("tg_r3_cmd_test")
    pkg.__path__ = [str(_REPO_ROOT / "skills" / "telegram")]
    sys.modules["tg_r3_cmd_test"] = pkg
    spec = importlib.util.spec_from_file_location("tg_r3_cmd_test.plugin", _REPO_ROOT / "skills" / "telegram" / "plugin.py")
    assert spec and spec.loader
    plugin = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = plugin
    spec.loader.exec_module(plugin)

    class _FailCmdClient:
        bot_id = "123456"
        def __init__(self): self.calls = 0
        async def call(self, method, data=None, **_kw):
            self.calls += 1
            raise RuntimeError("chat not found")

    c = _FailCmdClient()
    key = str(state_dir.resolve())
    assert plugin._should_sync_commands(key, "123456", "42") is True
    asyncio.run(plugin._sync_bot_commands(api, c, "full_access", "en", "42"))
    assert c.calls == 1
    assert plugin._should_sync_commands(key, "123456", "42") is False  # cooldown active for failed target 42!
    assert plugin._should_sync_commands(key, "123456", "43") is True   # newly changed chat 43 tries immediately!


def test_r4_binding_fence_partial_rebind_replay_retention_and_mutation_guards(tmp_path):
    import http.server, importlib.util, socketserver, sys, threading
    from skills.telegram.scripts.telegram_settings import merge_settings
    from skills.telegram.lib.telegram_token import binding_for

    # Local mock HTTP Bot API servers (8101/8102 ephemeral ports) for live transport probes
    class _BotHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            method = self.path.rsplit("/", 1)[-1]
            self.server.hits.append((method, body))
            cb = getattr(self.server, "on_hit", None)
            if callable(cb):
                status, payload = cb(method, body)
            else:
                res = True if method in {"setMessageReaction", "sendChatAction"} else {"message_id": 77, "chat": {"id": int(body["chat_id"]), "type": "private"}}
                status, payload = 200, {"ok": True, "result": res}
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *_args):
            pass

    def _start_srv():
        srv = socketserver.TCPServer(("127.0.0.1", 0), _BotHandler)
        srv.hits = []
        srv.on_hit = None
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        return srv, f"http://127.0.0.1:{srv.server_address[1]}"

    srv1, base1 = _start_srv()
    srv2, base2 = _start_srv()
    try:
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        merge_settings(state_dir, {"TELEGRAM_API_BASE": base1})
        api = _StubApi(state_dir, token="123456:TOK_A")
        binding = binding_for(api)
        binding.read_now(api)
        binding.mark_validated(binding.snapshot()[1])

        # 1) Server swap (srv1 -> srv2) + same-bot revalidation DURING await place_hold must NOT hit srv1!
        class _SwapSubmitter(_StubSubmitter):
            def __init__(self): super().__init__(); self.rt, self.mode = None, ""
            async def place_hold(self, account_id: str, conversation_id: str, thread_id: str = "") -> str:
                if self.mode == "swap_server":
                    merge_settings(state_dir, {"TELEGRAM_API_BASE": base2})
                    self.rt.on_token_changed(); self.rt.on_bot_validated("123456:TOK_A", "123456", "bot_a"); self.mode = ""
                elif self.mode == "clear_identity":
                    self.rt.on_token_changed(); self.mode = ""
                return ""

        sub = _SwapSubmitter()
        rt = PresenceRuntime(api=api, state_dir=state_dir, submitter=sub)
        sub.rt = rt
        rt.on_bot_validated("123456:TOK_A", "123456", "bot_a")

        # 1a) Normal send paused during place_hold while server swaps srv1 -> srv2 and same bot revalidates
        sub.mode = "swap_server"
        rt.store.enqueue_outbox("telegram-send:srv-swap", {"kind": "message", "chat_id": "8001", "text": "hello", "account_id": "123456", "markdown": False})
        assert asyncio.run(rt.process_one_outbox()) is True
        assert sub.mode == "" and srv1.hits == [] and srv2.hits == []
        assert rt.store.delivery_receipt("telegram-send:srv-swap")["state"] == "held"

        # 1b) Reaction paused during place_hold while server swaps srv1 -> srv2 and same bot revalidates
        merge_settings(state_dir, {"TELEGRAM_API_BASE": base1})
        rt.on_bot_validated("123456:TOK_A", "123456", "bot_a")
        sub.mode = "swap_server"
        rt.store.enqueue_outbox("telegram-operation:srv-swap-rx", {
            "kind": "operation", "method": "setMessageReaction", "chat_id": "8002", "account_id": "123456",
            "parameters": {"chat_id": "8002", "message_id": 1, "reaction": [{"type": "emoji", "emoji": "👍"}]},
        })
        assert asyncio.run(rt.process_one_outbox()) is True
        assert sub.mode == "" and srv1.hits == [] and srv2.hits == []
        assert rt.store.delivery_receipt("telegram-operation:srv-swap-rx")["state"] == "held"

        # 2) HTML 400 fallback retry after identity cleared mid-request must NOT issue plain-text sendMessage!
        def _html_400_and_clear(method, body):
            if body.get("parse_mode") == "HTML":
                rt.on_token_changed()
                return 400, {"ok": False, "error_code": 400, "description": "Bad Request: can't parse entities"}
            return 200, {"ok": True, "result": {"message_id": 88, "chat": {"id": int(body["chat_id"]), "type": "private"}}}

        srv2.on_hit = _html_400_and_clear
        rt.store.enqueue_outbox("telegram-send:html-retry", {"kind": "message", "chat_id": "8003", "text": "*bold*", "account_id": "123456", "markdown": True})
        assert asyncio.run(rt.process_one_outbox()) is True
        assert len(srv2.hits) == 1 and srv2.hits[0][1].get("parse_mode") == "HTML"
        assert rt.store.delivery_receipt("telegram-send:html-retry")["state"] == "held"
        srv2.on_hit = None
        srv2.hits.clear()

        # 3) process_typing() clearing identity during await place_hold must NOT dispatch sendChatAction!
        rt.on_bot_validated("123456:TOK_A", "123456", "bot_a")
        u_type = {"update_id": 90, "message": {"message_id": 1, "date": 1_900_000_005, "chat": {"id": 8004, "type": "private", "first_name": "T"}, "from": {"id": 8004, "is_bot": False, "first_name": "T"}, "text": "ping"}}
        assert asyncio.run(rt.ingest_update(u_type, account_id="123456")) is True
        type_lease = rt.store.claim_inbox()
        assert type_lease is not None
        rt.store.request_typing(chat_id="8004", topic_id="", turn_ref="turn-t1", source_event_id=type_lease.event_id, account_id="123456")
        sub.mode = "clear_identity"
        assert asyncio.run(rt.process_typing()) == 0
        assert sub.mode == "" and srv2.hits == []

        # 4) Rejected inbox replay must NOT revert retained message text used for reaction excerpt!
        rt.on_bot_validated("123456:TOK_A", "123456", "bot_a")
        chat = {"id": -99, "type": "supergroup", "title": "Retention"}
        user = {"id": 501, "is_bot": False, "first_name": "Alice"}
        u_orig = {"update_id": 100, "message": {"message_id": 10, "date": 1_900_000_010, "chat": chat, "from": user, "text": "original"}}
        u_edit = {"update_id": 101, "edited_message": {"message_id": 10, "date": 1_900_000_010, "edit_date": 1_900_000_020, "chat": chat, "from": user, "text": "revised"}}
        u_replay = {**u_orig, "update_id": 2}
        u_react = {"update_id": 3, "message_reaction": {"chat": chat, "message_id": 10, "user": user, "date": 1_900_000_030, "old_reaction": [], "new_reaction": [{"type": "emoji", "emoji": "🔥"}]}}
        assert asyncio.run(rt.ingest_update(u_orig, account_id="123456")) is True
        assert asyncio.run(rt.ingest_update(u_edit, account_id="123456")) is True
        assert asyncio.run(rt.ingest_update(u_replay, account_id="123456")) is False
        assert asyncio.run(rt.ingest_update(u_react, account_id="123456")) is True
        with rt.store._connect() as conn:
            row = conn.execute("SELECT payload_json FROM inbox WHERE update_id=3").fetchone()
            assert json.loads(row["payload_json"])["message"]["reaction"]["target"]["excerpt"]["text"] == "revised"

        # 5) Same-second my_chat_member replay dedup (record_only), out-of-range date clamp, and stopped on_poll_ok guard!
        g2 = {"id": -100, "type": "supergroup", "title": "SameSec"}
        t0 = 1_900_000_050
        m_join = {"update_id": 50, "my_chat_member": {"chat": g2, "date": t0, "old_chat_member": {"user": {"id": 123456, "is_bot": True}, "status": "left"}, "new_chat_member": {"user": {"id": 123456, "is_bot": True}, "status": "member"}}}
        m_kick = {"update_id": 51, "my_chat_member": {"chat": g2, "date": t0, "old_chat_member": {"user": {"id": 123456, "is_bot": True}, "status": "member"}, "new_chat_member": {"user": {"id": 123456, "is_bot": True}, "status": "kicked"}}}
        m_join_replay = {**m_join, "update_id": 1}
        m_huge_date = {"update_id": 52, "my_chat_member": {"chat": {"id": -101, "type": "supergroup", "title": "HugeDate"}, "date": 10**13, "old_chat_member": {"user": {"id": 123456, "is_bot": True}, "status": "left"}, "new_chat_member": {"user": {"id": 123456, "is_bot": True}, "status": "member"}}}
        asyncio.run(rt.ingest_update(m_join, account_id="123456", record_only=True))
        asyncio.run(rt.ingest_update(m_kick, account_id="123456", record_only=True))
        asyncio.run(rt.ingest_update(m_join_replay, account_id="123456", record_only=True))
        assert "-100" not in [g["conversation_id"] for g in rt.store.offered_groups("123456")]
        asyncio.run(rt.ingest_update(m_huge_date, account_id="123456", record_only=True))
        assert "-101" in [g["conversation_id"] for g in rt.store.offered_groups("123456")]
        rt._runtime_state = "stopped"
        rt.on_poll_ok()
        assert rt._runtime_state == "stopped"

        # 6) Partial command rebind: clearing chat 42 then failing on 43 MUST allow immediate re-sync back to 42!
        pkg = types.ModuleType("tg_r4_cmd_test")
        pkg.__path__ = [str(_REPO_ROOT / "skills" / "telegram")]
        sys.modules["tg_r4_cmd_test"] = pkg
        spec = importlib.util.spec_from_file_location("tg_r4_cmd_test.plugin", _REPO_ROOT / "skills" / "telegram" / "plugin.py")
        assert spec and spec.loader
        plugin = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = plugin
        spec.loader.exec_module(plugin)

        class _PartialFailClient:
            bot_id = "123456"
            def __init__(self): self.calls = []
            async def call(self, method, data=None, **_kw):
                scope = json.loads((data or {}).get("scope") or "{}")
                self.calls.append((method, scope, (data or {}).get("commands")))
                if scope.get("chat_id") == 43 and (data or {}).get("commands") != "[]":
                    raise RuntimeError("chat 43 unavailable")
                return True

        c = _PartialFailClient()
        key = str(state_dir.resolve())
        asyncio.run(plugin._sync_bot_commands(api, c, "full_access", "en", "42"))
        assert plugin._should_sync_commands(key, "123456", "42") is False
        asyncio.run(plugin._sync_bot_commands(api, c, "full_access", "en", "43"))  # clears 42, fails on 43
        assert plugin._should_sync_commands(key, "123456", "42") is True  # MUST re-sync 42 immediately!
    finally:
        srv1.shutdown()
        srv2.shutdown()
        srv1.server_close()
        srv2.server_close()


def test_r6_security_and_live_readiness_hardening(tmp_path):
    import importlib.util, sys
    from skills.telegram.lib.presence.events import parse_telegram_update
    from skills.telegram.lib.presence.host import PlaceBindingRevoked
    from skills.telegram.lib.presence.tools import register_presence_tools
    from skills.telegram.lib.telegram_mirror import target_chat
    from skills.telegram.lib.telegram_notifier import _pinned_chat_id
    from skills.telegram.lib.telegram_poll_binding import InjectNotAccepted, PollCursor, handle_poll_batch
    from skills.telegram.lib.telegram_token import binding_for
    from skills.telegram.scripts.telegram_settings import load_settings, merge_settings
    from ouroboros.skill_loader import parse_skill_manifest_text

    # Load isolated plugin module
    pkg = types.ModuleType("tg_r6_sec_test")
    pkg.__path__ = [str(_REPO_ROOT / "skills" / "telegram")]
    sys.modules["tg_r6_sec_test"] = pkg
    spec = importlib.util.spec_from_file_location("tg_r6_sec_test.plugin", _REPO_ROOT / "skills" / "telegram" / "plugin.py")
    assert spec and spec.loader
    plugin = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = plugin
    spec.loader.exec_module(plugin)

    # 0) SKILL.md declares conflict with Hub telegram-bot
    manifest = parse_skill_manifest_text((_REPO_ROOT / "skills" / "telegram" / "SKILL.md").read_text(encoding="utf-8"))
    assert "telegram-bot" in manifest.conflicts

    # 1) Panic normalization (/main /panic, strict command boundary rejecting /panicking & literal code) + concurrency!
    assert plugin._panic_update({"message": {"text": "/main /panic"}}) is True
    assert plugin._panic_update({"message": {"text": "/m@my_bot /panic now"}}) is True
    assert plugin._panic_update({"message": {"text": "/panic!"}}) is False
    assert plugin._panic_update({"message": {"text": "/main /panicking"}}) is False
    assert plugin._panic_update({"message": {"caption": "/panic!"}}) is False
    assert plugin._panic_update({"message": {"text": "/panic", "entities": [{"type": "code", "offset": 0, "length": 6}]}}) is False
    assert plugin._panic_update({"message": {"text": "/main hello"}}) is False
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    merge_settings(state_dir, {"TELEGRAM_CHAT_ID": "42"})
    api = _StubApi(state_dir, token="123456:TOK_A")
    binding = binding_for(api)
    binding.read_now(api)
    _, gen = binding.snapshot()
    cursor = PollCursor(state_dir / "poll_cursor.json", "https://api.telegram.org", "123456")

    async def _panic_race():
        order, release_slow = [], asyncio.Event()
        async def _on_msg(_api, _client, upd, _uid, _mode, _lang, **_kw):
            txt = (upd.get("message") or {}).get("text") or ""
            if txt == "slow-media":
                await release_slow.wait(); order.append("slow-done")
            else:
                order.append(txt); release_slow.set()
        batch = [
            {"update_id": 10, "message": {"message_id": 1, "date": 2_000_000_000, "chat": {"id": 42, "type": "private"}, "from": {"id": 42}, "text": "slow-media"}},
            {"update_id": 11, "message": {"message_id": 2, "date": 2_000_000_001, "chat": {"id": 42, "type": "private"}, "from": {"id": 42}, "text": "/main /panic"}},
        ]
        await asyncio.wait_for(handle_poll_batch(api, types.SimpleNamespace(bot_id="123456"), batch, cursor, binding, gen, "42", 20, "full_access", "en", authorize=plugin._authorized_owner, is_panic=plugin._panic_update, on_message=_on_msg, on_callback=lambda *a: None), timeout=5.0)
        return order
    assert asyncio.run(_panic_race()) == ["/main /panic", "slow-done"]

    # 1b) Cancellation after early panic rolls back in-flight owner & group updates; cancel during finally wait propagates!
    async def _panic_cancel_checks():
        from skills.telegram.lib.telegram_api import TelegramBindingChanged
        cl = types.SimpleNamespace(bot_id="123456")
        # B1a+B1b: cancel in-flight owner (4001) & non-owner group (4010) after early /panic finished -> rollback u0, keep panic replay
        c_b1, p_done, u_in = PollCursor(state_dir / "c_b1.json", "https://api.telegram.org", "123456"), asyncio.Event(), asyncio.Event()
        async def _on_b1(_a, _c, upd, *_r, **_k):
            if (upd.get("message") or {}).get("text") == "/panic": p_done.set()
            else: u_in.set(); await asyncio.Event().wait()
        b_b1 = [{"update_id": 4001, "message": {"message_id": 41, "date": 2_000_000_000, "chat": {"id": 42, "type": "private"}, "from": {"id": 42}, "text": "slow"}},
                {"update_id": 4002, "message": {"message_id": 42, "date": 2_000_000_001, "chat": {"id": 42, "type": "private"}, "from": {"id": 42}, "text": "/panic"}}]
        t1 = asyncio.create_task(handle_poll_batch(api, cl, b_b1, c_b1, binding, gen, "42", 20, "full_access", "en", authorize=plugin._authorized_owner, is_panic=plugin._panic_update, on_message=_on_b1, on_callback=lambda *a: None))
        await asyncio.wait_for(asyncio.gather(p_done.wait(), u_in.wait()), timeout=2.0); t1.cancel()
        with pytest.raises(asyncio.CancelledError): await t1
        c_b1r = PollCursor(state_dir / "c_b1.json", "https://api.telegram.org", "123456")
        assert t1.cancelled() and c_b1r.offset == 0 and not c_b1r.is_replay(b_b1[0]) and c_b1r.is_replay(b_b1[1])
        c_b1g, p2_done, g_in = PollCursor(state_dir / "c_b1g.json", "https://api.telegram.org", "123456"), asyncio.Event(), asyncio.Event()
        async def _on_g_msg(_a, _c, *_r, **_k): p2_done.set()
        async def _on_g_non(*_a): g_in.set(); await asyncio.Event().wait()
        b_b1g = [{"update_id": 4010, "message": {"message_id": 51, "date": 2_000_000_000, "chat": {"id": -1001, "type": "supergroup"}, "from": {"id": 77}, "text": "grp"}},
                 {"update_id": 4011, "message": {"message_id": 52, "date": 2_000_000_001, "chat": {"id": 42, "type": "private"}, "from": {"id": 42}, "text": "/panic"}}]
        t1g = asyncio.create_task(handle_poll_batch(api, cl, b_b1g, c_b1g, binding, gen, "42", 20, "full_access", "en", authorize=plugin._authorized_owner, is_panic=plugin._panic_update, on_message=_on_g_msg, on_callback=lambda *a: None, on_non_owner=_on_g_non))
        await asyncio.wait_for(asyncio.gather(p2_done.wait(), g_in.wait()), timeout=2.0); t1g.cancel()
        with pytest.raises(asyncio.CancelledError): await t1g
        c_b1gr = PollCursor(state_dir / "c_b1g.json", "https://api.telegram.org", "123456")
        assert t1g.cancelled() and c_b1gr.offset == 0 and not c_b1gr.is_replay(b_b1g[0]) and c_b1gr.is_replay(b_b1g[1])
        # B1c: 3-poll confirmed early callback behind deferred Presence MUST NOT re-run after intervening partial poll!
        c3_path, cb_runs, p3_runs = state_dir / "c_3poll.json", [], []
        cb_in, p3_blk, g4000_in = asyncio.Event(), asyncio.Event(), asyncio.Event()
        b3 = [{"update_id": 3999, "message": {"message_id": 60, "date": 2_000_000_000, "chat": {"id": -1001, "type": "supergroup"}, "from": {"id": 77}, "text": "g1"}},
              {"update_id": 4000, "message": {"message_id": 61, "date": 2_000_000_001, "chat": {"id": -1001, "type": "supergroup"}, "from": {"id": 77}, "text": "g2"}},
              {"update_id": 4001, "callback_query": {"id": "cb-4001", "from": {"id": 42}, "message": {"message_id": 62, "chat": {"id": 42, "type": "private"}}, "data": "ok"}},
              {"update_id": 4002, "message": {"message_id": 63, "date": 2_000_000_002, "chat": {"id": 42, "type": "private"}, "from": {"id": 42}, "text": "/panic"}}]
        phase = [1]
        async def _non3(_a, _c, upd, _pc):
            uid = upd["update_id"]
            if phase[0] == 1 and uid == 3999: raise InjectNotAccepted("full")
            if phase[0] == 2 and uid == 4000: g4000_in.set(); await asyncio.Event().wait()
        async def _cb3(_a, _c, _u, _pc, _m, _l, confirm):
            cb_runs.append(phase[0]); confirm()
            if phase[0] == 1: cb_in.set(); await asyncio.Event().wait()
            return "en"
        async def _msg3(_a, _c, *_r, **_k):
            if phase[0] == 1: p3_blk.set(); await asyncio.Event().wait()
            if phase[0] == 2: await asyncio.Event().wait()
            p3_runs.append(phase[0])
        # Poll 1: defer 3999, confirm early 4001 + pending panic 4002, cancel -> offset 0, 4001 replay
        t_p1 = asyncio.create_task(handle_poll_batch(api, cl, b3, PollCursor(c3_path, "https://api.telegram.org", "123456"), binding, gen, "42", 20, "full_access", "en", authorize=plugin._authorized_owner, is_panic=plugin._panic_update, on_message=_msg3, on_callback=_cb3, on_non_owner=_non3))
        await asyncio.wait_for(asyncio.gather(cb_in.wait(), p3_blk.wait()), timeout=2.0); t_p1.cancel()
        with pytest.raises(asyncio.CancelledError): await asyncio.wait_for(t_p1, timeout=1.0)
        c3_1 = PollCursor(c3_path, "https://api.telegram.org", "123456")
        assert t_p1.cancelled() and c3_1.offset == 0 and c3_1.is_replay(b3[2]) and not c3_1.is_replay(b3[3])
        # Poll 2: accept 3999, block on 4000, cancel -> offset 4000, 4001 MUST STAY replay via early map!
        phase[0] = 2
        t_p2 = asyncio.create_task(handle_poll_batch(api, cl, b3, c3_1, binding, gen, "42", 20, "full_access", "en", authorize=plugin._authorized_owner, is_panic=plugin._panic_update, on_message=_msg3, on_callback=_cb3, on_non_owner=_non3))
        await asyncio.wait_for(g4000_in.wait(), timeout=2.0); t_p2.cancel()
        with pytest.raises(asyncio.CancelledError): await asyncio.wait_for(t_p2, timeout=1.0)
        c3_2 = PollCursor(c3_path, "https://api.telegram.org", "123456")
        assert t_p2.cancelled() and c3_2.offset == 4000 and c3_2.is_replay(b3[2])
        # Poll 3: finish batch -> 4001 NOT re-applied (cb_runs == [1]), panic runs once, offset 4003!
        phase[0] = 3
        await handle_poll_batch(api, cl, b3, c3_2, binding, gen, "42", 20, "full_access", "en", authorize=plugin._authorized_owner, is_panic=plugin._panic_update, on_message=_msg3, on_callback=_cb3, on_non_owner=_non3)
        c3_3 = PollCursor(c3_path, "https://api.telegram.org", "123456")
        assert c3_3.offset == 4003 and cb_runs == [1] and p3_runs == [3]
        # B2 + save-error cancel propagation + failing-owner budget
        c_b2, p4_in, log_mark = PollCursor(state_dir / "c_b2.json", "https://api.telegram.org", "123456"), asyncio.Event(), len(api.logs)
        async def _on_b2(_a, _c, upd, *_r, **_k):
            if (upd.get("message") or {}).get("text") == "/panic": p4_in.set(); await asyncio.Event().wait()
            raise TelegramBindingChanged()
        t2 = asyncio.create_task(handle_poll_batch(api, cl, b_b1, c_b2, binding, gen, "42", 20, "full_access", "en", authorize=plugin._authorized_owner, is_panic=plugin._panic_update, on_message=_on_b2, on_callback=lambda *a: None))
        await asyncio.wait_for(p4_in.wait(), timeout=2.0); await asyncio.sleep(0.02); t2.cancel()
        with pytest.raises(asyncio.CancelledError): await asyncio.wait_for(t2, timeout=1.0)
        c_b2r = PollCursor(state_dir / "c_b2.json", "https://api.telegram.org", "123456")
        assert t2.cancelled() and c_b2r.offset == 0 and not c_b2r.is_replay(b_b1[0]) and any("stays unconfirmed" in m for _, m in api.logs[log_mark:])
        bad_parent = state_dir / "not_a_dir"; bad_parent.write_text("x", encoding="utf-8")
        c_err, e_in = PollCursor(bad_parent / "cur.json", "https://api.telegram.org", "123456"), asyncio.Event()
        async def _cb_err(_a, _c, _u, _pc, _m, _l, confirm): confirm(); e_in.set(); await asyncio.Event().wait()
        te = asyncio.create_task(handle_poll_batch(api, cl, [b3[2]], c_err, binding, gen, "42", 20, "full_access", "en", authorize=plugin._authorized_owner, is_panic=plugin._panic_update, on_message=_msg3, on_callback=_cb_err))
        await asyncio.wait_for(e_in.wait(), timeout=2.0); te.cancel()
        with pytest.raises(asyncio.CancelledError): await asyncio.wait_for(te, timeout=1.0)
        assert te.cancelled()
        # A normally completing batch awaited INSIDE a caller's except CancelledError MUST still raise OSError from cursor.save!
        child = asyncio.create_task(asyncio.Event().wait()); child.cancel()
        try: await child
        except asyncio.CancelledError:
            async def _cb_ok(_a, _c, _u, _pc, _m, _l, confirm): confirm(); return "en"
            with pytest.raises(OSError):
                await handle_poll_batch(api, cl, [b3[2]], PollCursor(bad_parent / "cur2.json", "https://api.telegram.org", "123456"), binding, gen, "42", 20, "full_access", "en", authorize=plugin._authorized_owner, is_panic=plugin._panic_update, on_message=_msg3, on_callback=_cb_ok)
        # Non-early confirmed callback cancelled while panic is pending advances offset to 4002 & keeps callback as replay!
        c_ne, ne_in, ne_p = PollCursor(state_dir / "c_ne.json", "https://api.telegram.org", "123456"), asyncio.Event(), asyncio.Event()
        async def _cb_ne(_a, _c, _u, _pc, _m, _l, confirm): confirm(); ne_in.set(); await asyncio.Event().wait()
        async def _p_ne(_a, _c, *_r, **_k): ne_p.set(); await asyncio.Event().wait()
        t_ne = asyncio.create_task(handle_poll_batch(api, cl, b3[2:], c_ne, binding, gen, "42", 20, "full_access", "en", authorize=plugin._authorized_owner, is_panic=plugin._panic_update, on_message=_p_ne, on_callback=_cb_ne))
        await asyncio.wait_for(asyncio.gather(ne_in.wait(), ne_p.wait()), timeout=2.0); t_ne.cancel()
        with pytest.raises(asyncio.CancelledError): await asyncio.wait_for(t_ne, timeout=1.0)
        c_ner = PollCursor(state_dir / "c_ne.json", "https://api.telegram.org", "123456")
        assert t_ne.cancelled() and c_ner.offset == 4002 and c_ner.is_replay(b3[2]) and not c_ner.is_replay(b3[3])
        c_b3, seen_b3 = PollCursor(state_dir / "c_b3.json", "https://api.telegram.org", "123456"), []
        async def _on_b3(_a, _c, upd, *_r, **_k):
            t = (upd.get("message") or {}).get("text") or ""; seen_b3.append(t)
            if t == "boom": raise RuntimeError("boom")
        async def _defer_non(*_a): raise InjectNotAccepted("full")
        b_b3 = [{"update_id": 4030, "message": {"message_id": 70, "date": 2_000_000_000, "chat": {"id": 777, "type": "private"}, "from": {"id": 777}, "text": "str"}},
                {"update_id": 4031, "message": {"message_id": 71, "date": 2_000_000_001, "chat": {"id": 42, "type": "private"}, "from": {"id": 42}, "text": "boom"}},
                {"update_id": 4032, "message": {"message_id": 72, "date": 2_000_000_002, "chat": {"id": 42, "type": "private"}, "from": {"id": 42}, "text": "ok2"}}]
        cl_b3 = types.SimpleNamespace(bot_id="123456", send_message=lambda *_a: asyncio.sleep(0))
        await handle_poll_batch(api, cl_b3, b_b3, c_b3, binding, gen, "42", 1, "full_access", "en", authorize=plugin._authorized_owner, is_panic=plugin._panic_update, on_message=_on_b3, on_callback=lambda *a: None, on_non_owner=_defer_non)
        assert seen_b3 == ["boom"]
    asyncio.run(_panic_cancel_checks())

    # 2) Revoking/rebinding TELEGRAM_CHAT_ID mid-batch immediately revokes old owner & blocks re-TOFU; sender_chat unauthenticated
    merge_settings(state_dir, {"TELEGRAM_CHAT_ID": ""})
    u42 = {"update_id": 12, "message": {"message_id": 3, "chat": {"id": 42, "type": "private"}, "from": {"id": 42}, "text": "hi"}}
    assert asyncio.run(plugin._authorized_owner(api, types.SimpleNamespace(bot_id="123456"), u42, "42", "en")) is None
    assert load_settings(state_dir).get("TELEGRAM_CHAT_ID") == ""  # MUST NOT re-TOFU!
    merge_settings(state_dir, {"TELEGRAM_CHAT_ID": "99"})
    assert asyncio.run(plugin._authorized_owner(api, types.SimpleNamespace(bot_id="123456"), u42, "42", "en")) is None
    assert target_chat({"TELEGRAM_CHAT_ID": ""}, {"transport": {"kind": "telegram", "conversation_id": "42"}}) == 0
    assert _pinned_chat_id({"TELEGRAM_CHAT_ID": "-100999"}) == 0
    anon_ev = parse_telegram_update({"update_id": 13, "message": {"message_id": 4, "date": 1_900_000_000, "chat": {"id": -1001, "type": "supergroup"}, "sender_chat": {"id": -1001, "type": "supergroup", "title": "Anon"}, "text": "hi"}}, bot_account_id="123456", places=True)
    assert anon_ev is not None and anon_ev.actor.get("authenticated") is False

    # 3) Split & old secret cards refuse replies on every chunk & past 50-card eviction; HTTP proxy guarded by owner check
    sent_ids: list[int] = []
    class _SplitClient:
        bot_id, text_limit, seq = "123456", 50, 100
        async def send_message(self, _cid, _txt, parse_mode=""): self.seq += 1; return self.seq
        async def send_message_with_inline_keyboard(self, _cid, _txt, _kb, parse_mode=""): self.seq += 1; return self.seq
    last_id, overflowed = asyncio.run(plugin.telegram_quiz.send_quiz_card(_SplitClient(), 42, body="A" * 120, compact="Secret", hint_text="Do not reply", keyboard=[], reserve=4000, sent_ids=sent_ids))
    assert overflowed is True and len(sent_ids) >= 3 and sent_ids[-1] == last_id
    plugin.telegram_quiz.remember_quiz(api, "123456", "tok-sec", {"task_id": "t-sec", "quiz_id": "q-sec", "chat_id": 42, "message_id": last_id, "message_ids": sent_ids, "options": [], "secret": True})
    for idx in range(60):
        plugin.telegram_quiz.remember_quiz(api, "123456", f"tok-ord-{idx}", {"task_id": f"t-{idx}", "quiz_id": f"q-{idx}", "chat_id": 42, "message_id": 500 + idx, "options": ["ok"]})
    for mid in sent_ids:
        found = plugin.telegram_quiz.quiz_for_message(api, "123456", 42, mid)
        assert found is not None and found.get("secret") is True
    merge_settings(state_dir, {"TELEGRAM_API_BASE": "http://127.0.0.1:8081", "TELEGRAM_PROXY": ""})
    save_handler = plugin._make_settings_save(api)
    class _RemoteReq:
        method, client, headers = "POST", types.SimpleNamespace(host="203.0.113.55"), {}
        async def json(self): return {"TELEGRAM_PROXY": "socks5h://127.0.0.1:1080"}
    resp = asyncio.run(save_handler(_RemoteReq()))
    assert resp.status_code == 200 and load_settings(state_dir).get("TELEGRAM_PROXY") == "" and "left unchanged" in json.loads(resp.body)["message"]

    # 4) File access check BEFORE copy, receipt scoping + path redaction, Safety text/caption rule, idempotency conflict, album reaction, & PlaceBindingRevoked
    bot_client = _StubBotClient()
    class _RevokingSubmitter(_StubSubmitter):
        def __init__(self): super().__init__(); self.revoked = False
        async def place_hold(self, account_id: str, conversation_id: str, thread_id: str = "", *, binding=None) -> str:
            if self.revoked and isinstance(binding, dict): raise PlaceBindingRevoked("Cancelled: Place disconnected")
            return ""
    sub = _RevokingSubmitter()
    rt = PresenceRuntime(api=api, state_dir=state_dir, submitter=sub, client_factory=lambda: bot_client)
    rt.on_bot_validated("123456:TOK_A", "123456", "bot_a")
    rt.bind_place_sync = lambda acct, chat, topic="": {"version": 1, "connector": "telegram", "provider": "telegram", "account_id": acct, "conversation_id": chat, "thread_id": topic, "place_id": f"place:{chat}", "connected_revision": 1}
    register_presence_tools(api, rt)
    send_tool, receipt_tool, op_tool, mod_tool = api.tools["telegram_send"], api.tools["telegram_receipt"], api.tools["telegram_operation"], api.tools["telegram_moderate"]
    outside, uploads = tmp_path / "secret_outside.txt", tmp_path / "uploads"
    outside.write_text("top-secret", encoding="utf-8"); uploads.mkdir()
    ok_file = uploads / "ok.png"; ok_file.write_bytes(b"\x89PNG\r\n\x1a\n12345678")
    focus_a = types.SimpleNamespace(drive_root=tmp_path, task_id="task-a", task_metadata={"presence": {"place_id": "place:-1001", "event": {"provider": "telegram", "account_id": "123456", "conversation_id": "-1001", "thread_id": "", "source_event_id": "telegram:123456:m:-1001:1"}}})
    focus_b = types.SimpleNamespace(drive_root=tmp_path, task_id="task-b", task_metadata={"presence": {"place_id": "place:-2002", "event": {"provider": "telegram", "account_id": "123456", "conversation_id": "-2002", "thread_id": "", "source_event_id": "telegram:123456:m:-2002:1"}}})
    assert json.loads(send_tool(None, chat_id="-1001", kind="document", file_path=str(ok_file)))["code"] == "file_access_denied"
    assert json.loads(send_tool(focus_a, chat_id="-1001", kind="document", file_path=str(state_dir / "settings.json")))["code"] == "file_access_denied"
    assert json.loads(send_tool(focus_a, chat_id="-1001", kind="document", file_path=str(outside)))["code"] == "file_access_denied"
    assert json.loads(send_tool(focus_a, chat_id="-1001", kind="album", items=[{"kind": "photo", "file_path": str(ok_file)}, {"kind": "photo", "file_path": str(outside)}]))["code"] == "file_access_denied"
    assert not list((state_dir / "outbox_files").glob("**/*")) if (state_dir / "outbox_files").exists() else True

    # Safety fidelity: media cannot smuggle text beside caption; message cannot smuggle caption
    assert json.loads(send_tool(focus_a, chat_id="-1001", kind="photo", file_path=str(ok_file), text="safe", caption="evil"))["code"] == "use_caption"
    assert json.loads(send_tool(focus_a, chat_id="-1001", kind="message", text="hi", caption="extra"))["code"] == "unexpected_media_args"

    # Authorized upload succeeds; receipt visible to focus_a (with local paths stripped), hidden from focus_b!
    q_ok = json.loads(send_tool(focus_a, chat_id="-1001", kind="photo", file_path=str(ok_file), caption="nice", request_id="shared-id-1"))
    assert q_ok["ok"] is True and q_ok["state"] == "queued"
    rec_a = json.loads(receipt_tool(focus_a, request_id="shared-id-1", operation="send"))
    assert rec_a["state"] == "pending" and "source_path" not in rec_a["files"][0] and "copy_path" not in rec_a["files"][0]
    assert json.loads(receipt_tool(focus_b, request_id="shared-id-1", operation="send"))["state"] == "not_found"

    # Idempotency conflict: reusing shared-id-1 for different target/content returns request_id_conflict (never already_queued!)
    assert json.loads(send_tool(focus_b, chat_id="-2002", kind="message", text="other", request_id="shared-id-1"))["code"] == "request_id_conflict"
    assert json.loads(send_tool(focus_a, chat_id="-1001", kind="photo", file_path=str(ok_file), caption="nice", request_id="shared-id-1"))["state"] == "already_queued"

    # Place disconnect/reconnect permanently fails the queued row!
    sub.revoked = True
    assert asyncio.run(rt.process_one_outbox()) is True
    assert rt.store.delivery_receipt("telegram-send:shared-id-1")["state"] == "failed"
    sub.revoked = False

    # Album non-first item reaction refused; non-self editMessageText refused; delete_message forum topic mismatch refused
    chat_f = {"id": -3003, "type": "supergroup", "is_forum": True, "title": "Forum"}
    usr = {"id": 501, "is_bot": False, "first_name": "U"}
    asyncio.run(rt.ingest_update({"update_id": 60, "message": {"message_id": 20, "date": 1_900_000_000, "chat": chat_f, "from": usr, "is_topic_message": True, "message_thread_id": 7, "media_group_id": "mg1", "caption": "p1", "photo": [{"file_id": "f1", "file_unique_id": "u1"}]}}, account_id="123456"))
    asyncio.run(rt.ingest_update({"update_id": 61, "message": {"message_id": 21, "date": 1_900_000_000, "chat": chat_f, "from": usr, "is_topic_message": True, "message_thread_id": 7, "media_group_id": "mg1", "photo": [{"file_id": "f2", "file_unique_id": "u2"}]}}, account_id="123456"))
    assert "album_target_unconfirmed" in json.loads(op_tool(focus_a, method="setMessageReaction", chat_id="-3003", message_id=21, topic_id=7, reaction=[{"type": "emoji", "emoji": "👍"}], request_id="alb-rx"))["error"]
    assert "own messages" in json.loads(op_tool(focus_a, method="editMessageText", chat_id="-3003", message_id=20, topic_id=7, text="hack", request_id="ed-other"))["error"]
    assert "forum_topic_mismatch" in json.loads(mod_tool(focus_a, action="delete_message", chat_id="-3003", message_id=20, topic_id=9, request_id="del-wrong-topic"))["error"]

    # Moderation pre-call CALL_STARTED settles crashed attempt as uncertain (NEVER auto-retried!)
    rt.store.enqueue_outbox("telegram-moderate:crash-ban", {
        "kind": "moderation", "chat_id": "-3003", "action": "ban_member", "account_id": "123456",
        "parameters": {"chat_id": "-3003", "user_id": 501}, "_call_started": {"part": 0, "method": "moderate:ban_member"},
    })
    assert asyncio.run(rt.process_one_outbox()) is True
    assert rt.store.delivery_receipt("telegram-moderate:crash-ban")["state"] == "uncertain"

    # R7+R8 regression checks:
    from skills.telegram.lib.presence.custody import InboxFullError
    from skills.telegram.lib.presence.events import parse_telegram_update as _parse_ev
    from skills.telegram.lib.presence.file_access import authorize_upload_path
    from skills.telegram.lib.presence.host import PlaceBindingUnavailable, PresenceHostError, PresenceHostHTTPError, _parse_places_version
    from skills.telegram.lib.presence import outbox_files as _obf
    import concurrent.futures

    # A) Full inbox raises InboxFullError without advancing telegram_offset; duplicates still ack; owner lane stays live
    cap_store = CustodyStore(tmp_path / "cap.sqlite3")
    ev1 = _parse_ev({"update_id": 100, "message": {"message_id": 1, "date": 1_900_000_000, "chat": {"id": -9, "type": "group", "title": "G"}, "from": usr, "text": "a"}}, bot_account_id="123456", places=True)
    ev2 = _parse_ev({"update_id": 101, "message": {"message_id": 2, "date": 1_900_000_000, "chat": {"id": -9, "type": "group", "title": "G"}, "from": usr, "text": "b"}}, bot_account_id="123456", places=True)
    assert cap_store.commit_update(100, ev1, max_pending_inbox=1) is True
    assert cap_store.commit_update(100, ev1, max_pending_inbox=1) is False
    with pytest.raises(InboxFullError):
        cap_store.commit_update(101, ev2, max_pending_inbox=1)
    assert cap_store.telegram_offset() == 101
    cap_store.close()
    merge_settings(state_dir, {"TELEGRAM_CHAT_ID": "42"})
    owner_seen, target_val = [], [{"id": "row-A"}]
    async def _owner_live_on_inbox_full():
        cur2 = PollCursor(state_dir / "poll_cursor_live.json", "https://api.telegram.org", "123456")
        async def _full_non_owner(*_a): raise InjectNotAccepted("Presence inbox full")
        async def _rec_owner(_api, _cl, upd, *_a, reply_target=None, **_kw):
            txt = (upd.get("message") or {}).get("text") or ""
            if txt == "reply-retry":
                resolved = reply_target(lambda: target_val[0])
                owner_seen.append(f"{txt}:{resolved['id']}")
                if len(owner_seen) == 2:
                    raise InjectNotAccepted("retry reply")
                return
            owner_seen.append(txt)
        b2 = [
            {"update_id": 2001, "message": {"message_id": 1, "date": 2_000_000_000, "chat": {"id": 777, "type": "private"}, "from": {"id": 777}, "text": "stranger"}},
            {"update_id": 2002, "message": {"message_id": 2, "date": 2_000_000_001, "chat": {"id": 42, "type": "private"}, "from": {"id": 42}, "text": "/menu"}},
            {"update_id": 2003, "message": {"message_id": 3, "date": 2_000_000_002, "chat": {"id": 42, "type": "private"}, "from": {"id": 42}, "text": "reply-retry"}},
        ]
        # max_updates=1 still runs owner #1 behind blocked Presence update!
        d1, _ = await handle_poll_batch(api, types.SimpleNamespace(bot_id="123456"), b2, cur2, binding, gen, "42", 1, "full_access", "en",
            authorize=plugin._authorized_owner, is_panic=plugin._panic_update, on_message=_rec_owner, on_callback=lambda *a: None, on_non_owner=_full_non_owner)
        # Next poll runs reply-retry (resolves row-A, defers & holds target); reload cursor + change lookup -> MUST keep row-A!
        await handle_poll_batch(api, types.SimpleNamespace(bot_id="123456"), b2, cur2, binding, gen, "42", 2, "full_access", "en",
            authorize=plugin._authorized_owner, is_panic=plugin._panic_update, on_message=_rec_owner, on_callback=lambda *a: None, on_non_owner=_full_non_owner)
        target_val[0] = {"id": "row-B"}
        cur2b = PollCursor(state_dir / "poll_cursor_live.json", "https://api.telegram.org", "123456")
        await handle_poll_batch(api, types.SimpleNamespace(bot_id="123456"), b2, cur2b, binding, gen, "42", 2, "full_access", "en",
            authorize=plugin._authorized_owner, is_panic=plugin._panic_update, on_message=_rec_owner, on_callback=lambda *a: None, on_non_owner=_full_non_owner)
        return d1, cur2b.offset, cur2b.is_replay(b2[1]), cur2b.is_replay(b2[2])
    assert asyncio.run(_owner_live_on_inbox_full()) == (True, 0, True, True) and owner_seen == ["/menu", "reply-retry:row-A", "reply-retry:row-A"]

    # B) Historical outbox row without _idempotency_fingerprint fails closed on mismatched file hash
    legacy_payload = {"kind": "document", "chat_id": "-1001", "account_id": "123456", "_frozen": [{"path": "/x/a.bin", "sha256": "aaa", "size": 3}]}
    rt.store.enqueue_outbox("telegram-send:legacy-1", legacy_payload)
    assert rt.store.enqueue_outbox_checked("telegram-send:legacy-1", {**legacy_payload, "_frozen": [{"path": "/x/a.bin", "sha256": "bbb", "size": 3}]}, fingerprint="fp-different") == "conflict"

    # C) Atomic outbox_files quota (concurrent writers) + actual-bytes staging quota (no declared file_size)
    cap_dir, src4 = tmp_path / "qstate", tmp_path / "four.bin"
    src4.write_bytes(b"1234")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futs = [pool.submit(_obf.keep, src4, cap_dir, 100, storage_cap=6) for _ in range(2)]
        assert sum(not f.exception() for f in futs) == 1 and sum(isinstance(f.exception(), _obf.OutboxStorageFull) for f in futs) == 1
    assert _obf._dir_bytes(cap_dir / "outbox_files") == 4
    (cap_dir / "staging" / "prev").mkdir(parents=True, exist_ok=True); (cap_dir / "staging" / "prev" / "a.bin").write_bytes(b"1234")
    tmp_dl = cap_dir / "staging_tmp" / "dl.bin"; tmp_dl.parent.mkdir(parents=True, exist_ok=True); tmp_dl.write_bytes(b"5678")
    with pytest.raises(_obf.StagingStorageFull):
        _obf.commit_staged_file(cap_dir, tmp_dl, cap_dir / "staging" / "cur" / "b.bin", 6)
    assert _obf._dir_bytes(cap_dir / "staging") == 4 and not tmp_dl.exists()

    # D) Case-sensitive positive containment + strict /identity places_version validation
    ws_low, ws_hi = tmp_path / "case_ws", tmp_path / "CASE_WS"; ws_low.mkdir(); ws_hi.mkdir()
    if ws_low.resolve() != ws_hi.resolve():
        secret_hi = ws_hi / "secret.txt"; secret_hi.write_text("nope", encoding="utf-8")
        case_ctx = types.SimpleNamespace(repo_dir=ws_low, workspace_root=ws_low, workspace_mode="external", drive_root=tmp_path / "d", task_id="t-case", task_metadata={}, is_workspace_mode=lambda: True)
        with pytest.raises(PermissionError):
            authorize_upload_path(case_ctx, str(secret_hi), state_dir)
    assert _parse_places_version({"places_version": 0}) == 0 and _parse_places_version({"places_version": 1}) == 1
    for bad in ({}, {"places_version": "1"}, {"places_version": -1}, {"places_version": True}, {"places_version": 1.0}):
        with pytest.raises(PresenceHostError):
            _parse_places_version(bad)

    # E) PlaceBindingUnavailable refuses tool enqueue; 409 retained preserves binding; place_unattributable settles terminally; presence_auth_blocked holds (never wipes!)
    from ouroboros import places as _places
    from ouroboros.place_guard import outbox_binding_facts as _ob_facts
    from skills.telegram.lib.presence.host import PresenceHostClient as _PHC

    real_host_dir = tmp_path / "real_host_places"; real_host_dir.mkdir()
    _places.connect_place(real_host_dir, connector="telegram", provider="telegram", account_id="123456", conversation_id="-1001")
    class _RealPlaceTransport:
        async def request_json(self, method, url, *, headers, payload, timeout_sec):
            if url.endswith("/identity"): return {"ok": True, "places_version": 1, "presence_place_binding_version": 1, "presence_delivery_version": 1}
            import urllib.parse as _up
            qs = dict(_up.parse_qsl(_up.urlsplit(url).query))
            return {"ok": True, **_ob_facts(real_host_dir, "telegram", {k: qs.get(k, "") for k in ("provider", "account_id", "conversation_id", "thread_id")}, require_place=qs.get("require_place") == "true")}
    phc = _PHC(state_dir=state_dir, token_provider=lambda: "tok", host_base="http://127.0.0.1:9", transport=_RealPlaceTransport())
    with pytest.raises(PlaceBindingUnavailable) as exc_unattr:
        asyncio.run(phc.bind_place("123456", "999999"))
    assert exc_unattr.value.code == "place_not_connected"  # real Host returned place_unattributable -> terminal!

    rt.bind_place_sync = lambda *_a, **_k: (_ for _ in ()).throw(PlaceBindingUnavailable("Host 503", code="place_binding_unavailable"))
    register_presence_tools(api, rt)
    assert json.loads(api.tools["telegram_send"](focus_a, chat_id="-1001", kind="photo", file_path=str(ok_file), caption="c", request_id="bind-fail-1"))["code"] == "place_binding_unavailable"

    class _RetainedSub(_StubSubmitter):
        places_version = place_binding_version = 1
        async def bind_place(self, acct, chat, topic=""):
            if chat == "-99": raise PlaceBindingUnavailable("Add in Circles", code="place_not_connected")
            if chat == "-97": raise PlaceBindingUnavailable("Check access", code="presence_auth_blocked")
            if chat == "-98": return None
            return {"version": 1, "connector": "telegram", "provider": "telegram", "account_id": acct, "conversation_id": chat, "thread_id": topic, "place_id": f"plc:{chat}", "connected_revision": 1}
        async def submit(self, event, staged):
            if event["conversation_id"] == "-96":
                raise PresenceHostHTTPError(403, "blocked", {"ok": False, "disposition": "blocked", "code": "presence_auth_blocked"})
            raise PresenceHostHTTPError(409, "retained", {"ok": False, "disposition": "retained", "turn_ref": f"turn-{event['conversation_id']}", "delivery_reporting_version": 1})

    ret_dir, pauses = tmp_path / "ret_state", []
    ret_rt = PresenceRuntime(api=_StubApi(ret_dir), state_dir=ret_dir, submitter=_RetainedSub(), client_factory=lambda: bot_client)
    async def _rec_pause(sec): pauses.append(sec)
    ret_rt._pause = _rec_pause
    ret_rt.on_bot_validated("123456:TOK_A", "123456", "bot_a")
    for uid, cid in ((70, -55), (71, -99), (72, -98), (73, -97), (74, -96)):
        asyncio.run(ret_rt.ingest_update({"update_id": uid, "message": {"message_id": uid, "date": 1_900_000_000, "chat": {"id": cid, "type": "group", "title": "G"}, "from": usr, "text": "x"}}, account_id="123456"))
    assert all(asyncio.run(ret_rt.process_one_inbox()) for _ in range(5))
    wl = ret_rt.store.claim_work()
    assert wl is not None and wl.work_ref == "turn--55" and (wl.event.get("_place_binding") or {}).get("connected_revision") == 1 and ret_rt.store.claim_work() is None
    snap = ret_rt.store.status_snapshot()
    assert snap["inbox_failed"] == 1 and snap["inbox_waiting"] == 3 and pauses == [2.0, 2.0]  # both bind & turn auth_blocked pause & preserve!
    ret_rt.store.close()

    # F) bind_place_sync enforces ONE hard wall-clock deadline + encoding token check + WriteTimeout!
    import httpcore as _hc, socket as _sock, threading as _th, time as _tm
    from skills.telegram.lib.presence.host import _DeadlineStream, _check_identity_encoding, bind_place_sync as _bind_sync
    _check_identity_encoding([(b"content-encoding", b"identity, identity")])
    with pytest.raises(PresenceHostError): _check_identity_encoding([(b"content-encoding", b"identity, gzip")])
    with pytest.raises(_hc.WriteTimeout): _DeadlineStream(types.SimpleNamespace(write=lambda *a, **k: None), 0.0).write(b"x")  # type: ignore[arg-type]
    def _run_slow_probe(mode: str) -> float:
        srv = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM); srv.setsockopt(_sock.SOL_SOCKET, _sock.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0)); port = srv.getsockname()[1]; srv.listen(4); stop = _th.Event()
        def _serve():
            while not stop.is_set():
                srv.settimeout(0.1)
                try: conn, _ = srv.accept()
                except OSError: continue
                try:
                    raw = conn.recv(4096).decode("latin-1", "ignore")
                    if mode == "two_req":
                        _tm.sleep(0.38)
                        body = b'{"places_version":1,"presence_place_binding_version":1}' if "/identity" in raw else b'{"binding":{"version":1,"place_id":"p1"}}'
                        conn.sendall(f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
                    elif mode == "hdr_drip":
                        for b in b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}":
                            if stop.is_set(): break
                            conn.sendall(bytes([b])); _tm.sleep(0.08)
                    elif mode == "body_drip":
                        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\n")
                        for b in b'{"x"':
                            if stop.is_set(): break
                            _tm.sleep(0.42); conn.sendall(bytes([b]))
                except OSError: pass
                finally: conn.close()
        th = _th.Thread(target=_serve, daemon=True); th.start(); t0 = _tm.monotonic()
        try:
            with pytest.raises(PlaceBindingUnavailable):
                _bind_sync(host_base=f"http://127.0.0.1:{port}", token_provider=lambda: "tok", account_id="123456", conversation_id="-1001", delivery_reporting_status="unknown" if mode == "two_req" else "supported", timeout_sec=0.55)
            return _tm.monotonic() - t0
        finally:
            stop.set(); srv.close(); th.join(timeout=1.5)
    for m in ("two_req", "hdr_drip", "body_drip"):
        assert _run_slow_probe(m) < 0.95
