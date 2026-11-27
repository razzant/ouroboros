"""In-memory Telegram Bot API transports shared by menu and tunnel lifecycle tests."""
from __future__ import annotations

import json
from typing import Any

import httpx

OLD_URL, NEW_URL = "https://old-menu.trycloudflare.com/", "https://new-menu.trycloudflare.com/"


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


def web_app(url: str) -> dict[str, Any]:
    return {"type": "web_app", "text": "Ouroboros", "web_app": {"url": url}}
