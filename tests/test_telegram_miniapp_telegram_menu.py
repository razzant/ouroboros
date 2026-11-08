from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

SCRIPTS = Path(__file__).parents[1] / "skills" / "telegram" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from telegram_menu import (
    TelegramMenuError,
    TelegramMenuManager,
    TelegramMenuRejectedError,
    normalize_public_url,
    snapshot_owner_chat_id,
)


class FakeTelegram:
    def __init__(self) -> None:
        self.button: dict[str, Any] = {"type": "default"}
        self.set_calls: list[dict[str, Any]] = []
        self.request_timeouts: list[dict[str, float]] = []
        self.fail_after_set = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.request_timeouts.append(dict(request.extensions.get("timeout") or {}))
        method = request.url.path.rsplit("/", 1)[-1]
        payload = json.loads(request.content.decode("utf-8"))
        if method == "getMe":
            result: Any = {"id": 99, "username": "owner_bot", "is_bot": True}
        elif method == "getChat":
            result = {"id": 12345, "type": "private"}
        elif method == "getChatMenuButton":
            result = self.button
        elif method == "setChatMenuButton":
            self.button = payload["menu_button"]
            self.set_calls.append(self.button)
            if self.fail_after_set:
                self.fail_after_set = False
                raise httpx.ReadError("lost reply", request=request)
            result = True
        else:
            raise AssertionError(method)
        return httpx.Response(200, json={"ok": True, "result": result})


def manager(tmp_path: Path, remote: FakeTelegram) -> TelegramMenuManager:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(remote.handler),
        base_url="https://api.telegram.test",
    )
    return TelegramMenuManager(
        "secret-token",
        12345,
        tmp_path,
        api_root="https://api.telegram.test",
        client=client,
    )


def test_public_url_is_exact() -> None:
    assert normalize_public_url("https://abc-123.trycloudflare.com") == (
        "https://abc-123.trycloudflare.com/"
    )
    for bad in (
        "http://abc.trycloudflare.com/",
        "https://abc.trycloudflare.com/path",
        "https://abc.trycloudflare.com/?x=1",
        "https://user@abc.trycloudflare.com/",
        "https://trycloudflare.com/",
        "https://abc.trycloudflare.com.attacker.test/",
    ):
        with pytest.raises(TelegramMenuError):
            normalize_public_url(bad)


def test_install_rotate_and_restore_exact_original(tmp_path: Path) -> None:
    remote = FakeTelegram()
    menu = manager(tmp_path, remote)

    async def run() -> None:
        assert await menu.verify_private_owner() == "@owner_bot"
        assert await menu.install("https://first.trycloudflare.com/") is True
        assert await menu.install("https://first.trycloudflare.com/") is False
        assert await menu.install("https://second.trycloudflare.com/") is True
        assert remote.button["web_app"]["url"] == "https://second.trycloudflare.com/"
        assert await menu.restore() is True
        assert remote.button == {"type": "default"}
        assert not menu.snapshot_path.exists()
        await menu._client.aclose()  # type: ignore[union-attr]

    asyncio.run(run())
    assert len(remote.set_calls) == 3


def test_shutdown_restore_bounds_every_telegram_request(tmp_path: Path) -> None:
    remote = FakeTelegram()
    menu = manager(tmp_path, remote)

    async def run() -> None:
        assert await menu.install("https://first.trycloudflare.com/") is True
        remote.request_timeouts.clear()
        assert await menu.restore(request_timeout_sec=0.7) is True
        assert len(remote.request_timeouts) == 3
        assert all(
            values == {"connect": 0.7, "read": 0.7, "write": 0.7, "pool": 0.7}
            for values in remote.request_timeouts
        )
        await menu._client.aclose()  # type: ignore[union-attr]

    asyncio.run(run())


def test_restart_reconciles_lost_reply_after_remote_set(tmp_path: Path) -> None:
    remote = FakeTelegram()
    menu = manager(tmp_path, remote)
    remote.fail_after_set = True

    async def run() -> None:
        with pytest.raises(TelegramMenuError):
            await menu.install("https://first.trycloudflare.com/")
        saved = json.loads(menu.snapshot_path.read_text(encoding="utf-8"))
        assert saved["phase"] == "mutating"
        assert remote.button["web_app"]["url"] == "https://first.trycloudflare.com/"

        # The next start observes the target, finalizes the transaction, and
        # performs no duplicate Telegram mutation.
        assert await menu.install("https://first.trycloudflare.com/") is False
        assert len(remote.set_calls) == 1
        assert await menu.restore() is True
        await menu._client.aclose()  # type: ignore[union-attr]

    asyncio.run(run())


def test_external_change_is_never_overwritten(tmp_path: Path) -> None:
    remote = FakeTelegram()
    menu = manager(tmp_path, remote)

    async def run() -> None:
        await menu.install("https://first.trycloudflare.com/")
        remote.button = {"type": "commands"}
        with pytest.raises(TelegramMenuError, match="outside this skill"):
            await menu.install("https://second.trycloudflare.com/")
        with pytest.raises(TelegramMenuError, match="outside this skill"):
            await menu.restore()
        assert remote.button == {"type": "commands"}
        assert len(remote.set_calls) == 1
        await menu._client.aclose()  # type: ignore[union-attr]

    asyncio.run(run())


def test_exact_original_menu_recovers_stale_installed_snapshot(tmp_path: Path) -> None:
    remote = FakeTelegram()
    menu = manager(tmp_path, remote)

    async def run() -> None:
        await menu.install("https://first.trycloudflare.com/")
        remote.button = {"type": "default"}

        assert await menu.restore() is False
        assert not menu.snapshot_path.exists()
        assert len(remote.set_calls) == 1

        assert await menu.install("https://second.trycloudflare.com/") is True
        saved = json.loads(menu.snapshot_path.read_text(encoding="utf-8"))
        assert saved["original"] == {"type": "default"}
        assert saved["owned"]["web_app"]["url"] == "https://second.trycloudflare.com/"
        assert remote.button == saved["owned"]
        assert len(remote.set_calls) == 2
        await menu._client.aclose()  # type: ignore[union-attr]

    asyncio.run(run())


def test_snapshot_exposes_durable_prior_owner_for_cold_reconciliation(tmp_path: Path) -> None:
    remote = FakeTelegram()
    menu = manager(tmp_path, remote)

    async def run() -> None:
        await menu.install("https://first.trycloudflare.com/")
        assert snapshot_owner_chat_id(tmp_path, "99") == 12345  # this bot's ledger (getMe id 99)
        assert snapshot_owner_chat_id(tmp_path, "98") is None  # another bot has none
        await menu._client.aclose()  # type: ignore[union-attr]

    asyncio.run(run())


def test_api_errors_never_expose_token(tmp_path: Path) -> None:
    async def run() -> None:
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    400,
                    json={"ok": False, "description": "bad /botsecret-token/path"},
                )
            )
        )
        menu = TelegramMenuManager(
            "secret-token",
            12345,
            tmp_path,
            api_root="https://api.telegram.test",
            client=client,
        )
        with pytest.raises(TelegramMenuError) as captured:
            await menu.current()
        assert "secret-token" not in str(captured.value)
        await client.aclose()

    asyncio.run(run())


ROOT = "https://api.telegram.test"
OLD_URL, NEW_URL = "https://old-menu.trycloudflare.com/", "https://new-menu.trycloudflare.com/"


class Bots:
    """Telegram for several bots: getMe reports each token's bot id, buttons are per bot and chat, and a revoked
    token is refused (401) like any other request it makes."""

    def __init__(self, tokens: dict[str, Any], buttons: dict[tuple[int, int], dict[str, Any]]) -> None:
        self.tokens, self.buttons, self.revoked = dict(tokens), dict(buttons), set()
        self.fail_after_set = False  # the next set reaches Telegram, its reply is lost

    def handler(self, request: httpx.Request) -> httpx.Response:
        token, method = request.url.path.removeprefix("/bot").split("/", 1)
        if token in self.revoked or token not in self.tokens:
            return httpx.Response(401, json={"ok": False, "error_code": 401, "description": "Unauthorized"})
        bot, payload = self.tokens[token], json.loads(request.content or b"{}")
        if method == "getMe":
            result: Any = {"id": bot, "is_bot": True, "username": f"bot{bot}"}
        elif method == "getChat":
            result = {"id": payload["chat_id"], "type": "private"}
        elif method == "getChatMenuButton":
            result = self.buttons.get((bot, payload["chat_id"]), {"type": "default"})
        elif method == "setChatMenuButton":
            self.buttons[(bot, payload["chat_id"])], result = payload["menu_button"], True
            if self.fail_after_set:
                self.fail_after_set = False
                raise httpx.ReadError("lost reply", request=request)
        else:
            raise AssertionError(method)
        return httpx.Response(200, json={"ok": True, "result": result})


def bot_menu(state: Path, http: httpx.AsyncClient, token: str, chat: int = 12345, **kwargs: Any) -> TelegramMenuManager:
    return TelegramMenuManager(token, chat, state, api_root=ROOT, client=http, **kwargs)


def web_app(url: str) -> dict[str, Any]:
    return {"type": "web_app", "text": "Ouroboros", "web_app": {"url": url}}


@pytest.mark.parametrize("old_original,new_original", [("default", "commands"), ("commands", "default")])
def test_a_replacement_bot_keeps_its_own_ledger_beside_one_its_predecessor_could_not_restore(
    tmp_path: Path, old_original: str, new_original: str,
) -> None:
    """Bot A's token is revoked before A could restore its button. Bot B neither reads nor clears A's record: it
    records its own original, installs and restores it. A, back later with a new token, restores its own."""
    bots = Bots({"111:old": 111, "222:new": 222},
                {(111, 12345): {"type": old_original}, (222, 12345): {"type": new_original}})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(bots.handler)) as http:
            a = bot_menu(tmp_path, http, "111:old")
            assert await a.install(OLD_URL) is True
            bots.revoked.add("111:old")
            with pytest.raises(TelegramMenuRejectedError):
                await a.restore()
            kept = a.snapshot_path.read_bytes()
            b = bot_menu(tmp_path, http, "222:new")
            assert await b.install(NEW_URL) is True  # no drift: B's ledger starts from B's own menu
            assert json.loads(b.snapshot_path.read_text(encoding="utf-8"))["original"] == {"type": new_original}
            await b.check_owned(NEW_URL)
            assert await b.restore() is True and bots.buttons[(222, 12345)] == {"type": new_original}
            assert not b.snapshot_path.exists() and a.snapshot_path.read_bytes() == kept
            bots.tokens["111:fresh"] = 111  # bot A again, with a new token of the same bot
            assert await bot_menu(tmp_path, http, "111:fresh").restore() is True
            assert bots.buttons[(111, 12345)] == {"type": old_original} and not a.snapshot_path.exists()

    asyncio.run(run())


def test_a_rotated_token_of_the_same_bot_shares_its_ledger(tmp_path: Path) -> None:
    bots = Bots({"111:one": 111, "111:two": 111}, {(111, 12345): {"type": "commands"}})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(bots.handler)) as http:
            first = bot_menu(tmp_path, http, "111:one")
            assert await first.install(OLD_URL) is True
            bots.revoked.add("111:one")  # the old token cannot restore any more
            second = bot_menu(tmp_path, http, "111:two")
            assert await second.install(NEW_URL) is True  # it rotates the button its bot owns
            saved = json.loads(second.snapshot_path.read_text(encoding="utf-8"))
            assert second.snapshot_path == first.snapshot_path
            assert saved["original"] == {"type": "commands"} and saved["owned"] == web_app(NEW_URL)
            assert await second.restore() is True and bots.buttons[(111, 12345)] == {"type": "commands"}

    asyncio.run(run())


def test_an_owner_change_reads_only_this_bots_ledger(tmp_path: Path) -> None:
    """Bot B's previous owner is restored from B's own record by a manager carrying B's verified identity (as
    the companion builds it); bot A's record for the same chat is neither the prior owner nor touched."""
    bots = Bots({"111:a": 111, "222:b": 222}, {})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(bots.handler)) as http:
            a = bot_menu(tmp_path, http, "111:a", chat=42)
            b = bot_menu(tmp_path, http, "222:b", chat=42)
            assert await a.install(OLD_URL) and await b.install(NEW_URL)
            kept = a.snapshot_path.read_bytes()
            new_owner = bot_menu(tmp_path, http, "222:b", chat=43)
            assert await new_owner.verify_private_owner() == "@bot222"
            assert snapshot_owner_chat_id(tmp_path, new_owner.bot) == 42
            assert snapshot_owner_chat_id(tmp_path, "333") is None
            prior = bot_menu(tmp_path, http, "222:b", chat=42, bot=new_owner.bot)
            assert await prior.restore() is True and bots.buttons[(222, 42)] == {"type": "default"}
            assert snapshot_owner_chat_id(tmp_path, "222") is None and a.snapshot_path.read_bytes() == kept
            assert await new_owner.install(NEW_URL) is True
            assert snapshot_owner_chat_id(tmp_path, "222") == 43 and snapshot_owner_chat_id(tmp_path, "111") == 42

    asyncio.run(run())


def test_a_bot_showing_a_chat_only_records_button_derives_its_own_ledger_and_the_file_stays(tmp_path: Path) -> None:
    """A ledger written before ledgers named their bot is never changed or removed. The bot whose current menu shows
    a Mini App button it records derives its own ledger, naming that basis (an inference from the menu now, not a
    proven past writer) and the exact source bytes; every later state keeps it. Another bot, or the same bot for
    another owner, derives nothing from it."""
    legacy_bytes = json.dumps({"schema": 2, "chat_id": 12345, "phase": "installed", "original": {"type": "commands"},
                               "owned": web_app(OLD_URL)}).encode("utf-8")
    adopted = {"ledger": "menu_button_snapshot.json", "schema": 2, "basis": "current_menu_match",
               "sha256": hashlib.sha256(legacy_bytes).hexdigest()}
    for state in (tmp_path / "bots", tmp_path / "owner"):
        state.mkdir()
        (state / "menu_button_snapshot.json").write_bytes(legacy_bytes)
    bots = Bots({"222:SECRET-other": 222, "111:SECRET-same": 111},
                {(222, 12345): {"type": "default"}, (111, 12345): web_app(OLD_URL), (111, 777): web_app(OLD_URL)})
    third_url, fourth_url = "https://third-menu.trycloudflare.com/", "https://fourth-menu.trycloudflare.com/"

    def unchanged(state: Path) -> bool:
        return (state / "menu_button_snapshot.json").read_bytes() == legacy_bytes

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(bots.handler)) as http:
            other = bot_menu(tmp_path / "bots", http, "222:SECRET-other")
            assert await other.install(NEW_URL) is True
            saved = json.loads(other.snapshot_path.read_text(encoding="utf-8"))
            assert saved["original"] == {"type": "default"} and "adopted_from" not in saved
            assert await other.restore() is True and unchanged(tmp_path / "bots")

            same = bot_menu(tmp_path / "bots", http, "111:SECRET-same")
            written: list[dict[str, Any]] = []
            write = same._write
            same._write = lambda payload: written.append(payload) or write(payload)  # type: ignore[method-assign]
            assert await same.install(NEW_URL) is True  # derived, then rotated from the recorded button
            assert await same.install(third_url) is True
            bots.fail_after_set = True  # the next rotation reaches Telegram but its reply is lost
            with pytest.raises(TelegramMenuError):
                await same.install(fourth_url)
            assert json.loads(same.snapshot_path.read_text(encoding="utf-8"))["phase"] == "mutating"
            await same.check_snapshot_owned()  # the interrupted rotation settles to installed, provenance kept
            saved = json.loads(same.snapshot_path.read_text(encoding="utf-8"))
            assert saved["phase"] == "installed" and saved["original"] == {"type": "commands"}
            assert await same.restore() is True and bots.buttons[(111, 12345)] == {"type": "commands"}
            assert not same.snapshot_path.exists() and unchanged(tmp_path / "bots")
            # Derived, then rotate, rotate, an interrupted rotation settled, and the restore: every state keeps it.
            assert [payload["phase"] for payload in written] == ["installed", "mutating"] * 4
            assert all(payload["adopted_from"] == adopted and payload["bot"] == "111" for payload in written)
            assert "SECRET" not in json.dumps(written)

            # The derived ledger has ended; the file it came from is still there and proves nothing new.
            again = bot_menu(tmp_path / "bots", http, "111:SECRET-same")
            assert await again.install(NEW_URL) is True
            saved = json.loads(again.snapshot_path.read_text(encoding="utf-8"))
            assert saved["original"] == {"type": "commands"} and "adopted_from" not in saved
            assert await again.restore() is True
            assert await bot_menu(tmp_path / "bots", http, "222:SECRET-other").install(NEW_URL) is True
            assert unchanged(tmp_path / "bots")
            # The same bot for another owner (chat 777) derives nothing from chat 12345's record.
            elsewhere = bot_menu(tmp_path / "owner", http, "111:SECRET-same", chat=777)
            assert await elsewhere.restore() is False
            assert unchanged(tmp_path / "owner") and not elsewhere.snapshot_path.exists()

    asyncio.run(run())


def test_a_ledger_is_named_by_the_bot_getme_reports_never_by_the_token_text(tmp_path: Path) -> None:
    bots = Bots({"999:prefix-of-another-bot": 111, "5:bad": "not-a-number"}, {})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(bots.handler)) as http:
            menu = bot_menu(tmp_path, http, "999:prefix-of-another-bot")
            assert await menu.install(OLD_URL) is True
            assert menu.bot == "111" and menu.snapshot_path.name == "menu_button_snapshot.bot-111.json"
            assert snapshot_owner_chat_id(tmp_path, "999") is None
            with pytest.raises(TelegramMenuError, match="invalid bot identity"):
                await bot_menu(tmp_path, http, "5:bad").install(NEW_URL)
            assert sorted(path.name for path in tmp_path.iterdir()) == ["menu_button_snapshot.bot-111.json"]

    asyncio.run(run())
