from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest


def test_server_subcommand_sanitizes_argv(monkeypatch):
    from ouroboros import cli

    seen = {}

    class FakeServer:
        @staticmethod
        def executable_main():
            seen["argv"] = list(sys.argv)
            return 0

    monkeypatch.setitem(sys.modules, "server", FakeServer)
    monkeypatch.setattr(sys, "argv", ["ouroboros", "server", "--host", "127.0.0.1", "--port", "9000"])

    result = cli._server_command(SimpleNamespace(host="127.0.0.1", port=9000, no_ui=True))

    assert result == 0
    assert seen["argv"] == ["ouroboros"]
    assert json.loads(__import__("os").environ["OUROBOROS_SERVER_REEXEC_ARGV_JSON"]) == [
        "-m",
        "ouroboros.cli",
        "server",
        "--host",
        "127.0.0.1",
        "--port",
        "9000",
    ]
    assert sys.argv == ["ouroboros", "server", "--host", "127.0.0.1", "--port", "9000"]


def test_settings_context_mode_posts_owner_endpoint(monkeypatch):
    from ouroboros import cli

    seen = {}

    class FakeClient:
        def request(self, method, path, body=None):
            seen["request"] = (method, path, body)
            return {"ok": True, "context_mode": body["mode"]}

    monkeypatch.setattr(cli, "_client", lambda _args, **_kwargs: FakeClient())

    result = cli._owner_context_mode_command(SimpleNamespace(mode="low"))

    assert result == 0
    assert seen["request"] == ("POST", "/api/owner/context-mode", {"mode": "low"})


def test_chat_history_limit_maps_to_n_human(monkeypatch, capsys):
    """`chat history --limit N` requests the quota the server actually honors.

    The CLI sends the explicit `n_human` quota (the server separately honors
    legacy `limit` as a fallback default for already-shipped clients), so the
    flag is no longer a placebo (issue #172).
    """
    from ouroboros import cli

    seen = {}

    class FakeClient:
        def request(self, method, path, body=None):
            seen["request"] = (method, path)
            return {"messages": []}

    monkeypatch.setattr(cli, "_client", lambda _args, **_kwargs: FakeClient())

    result = cli._chat_history_command(SimpleNamespace(limit=25))

    assert result == 0
    assert seen["request"] == ("GET", "/api/chat/history?n_human=25")
    capsys.readouterr()


def test_chat_history_non_positive_limit_sends_no_quota(monkeypatch, capsys):
    """`--limit 0` (or a negative) asks for the server's window, not an empty chat.

    Before the flag was wired to `n_human` it was a placebo: every value returned the
    default window. An explicit `n_human=0` would now return zero conversation rows,
    so a script that passed 0 to mean "whatever the server gives" would silently lose
    its history. The server is already lenient the same way for the legacy `limit`.
    """
    from ouroboros import cli

    seen = []

    class FakeClient:
        def request(self, method, path, body=None):
            seen.append((method, path))
            return {"messages": []}

    monkeypatch.setattr(cli, "_client", lambda _args, **_kwargs: FakeClient())

    for limit in (0, -5):
        assert cli._chat_history_command(SimpleNamespace(limit=limit)) == 0

    assert seen == [("GET", "/api/chat/history"), ("GET", "/api/chat/history")]
    capsys.readouterr()

    # The parser is the command surface, so the help a user reads says it too.
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["chat", "history", "--help"])
    rendered = " ".join(capsys.readouterr().out.split())  # argparse wraps the help
    assert "omitted, zero or negative = the server's default window" in rendered


def test_chat_history_without_limit_sends_no_quota(monkeypatch, capsys):
    """Through the real parser: an omitted `--limit` requests the bare endpoint.

    No quota parameter is sent, so the default window has exactly one owner
    (the server's `_DEFAULT_N_HUMAN`) and the CLI cannot drift from it.
    """
    from ouroboros import cli

    seen = {}

    class FakeClient:
        def request(self, method, path, body=None):
            seen["request"] = (method, path)
            return {"messages": []}

    monkeypatch.setattr(cli, "_client", lambda _args, **_kwargs: FakeClient())

    args = cli.build_parser().parse_args(["chat", "history"])

    assert args.func(args) == 0
    assert seen["request"] == ("GET", "/api/chat/history")
    capsys.readouterr()


@pytest.mark.parametrize("hub", ["sources", "ouroboroshub"])
@pytest.mark.parametrize("command,method,path,body", [
    (["installed"], "GET", "/api/extensions", None),
    (["search", "weather"], "GET", "/api/skill-catalog/search?q=weather", None),
    (["preview", "ouroboros/weather"], "GET", "/api/skill-catalog/item/ouroboros/weather", None),
    (["install", "ouroboros/weather"], "POST", "/api/skill-catalog/install", {"id": "ouroboros/weather"}),
    (["update", "weather"], "POST", "/api/skills/weather/update", {"mode": "replace"}),
    (["uninstall", "weather"], "POST", "/api/skills/weather/uninstall", {}),
])
def test_marketplace_hub_alias_uses_source_routes(monkeypatch, hub, command, method, path, body):
    from ouroboros import cli

    calls = []
    client = SimpleNamespace(request=lambda *args: calls.append(args) or {"ok": True})
    monkeypatch.setattr(cli, "_client", lambda _args: client)
    args = cli.build_parser().parse_args(["marketplace", hub, *command])
    assert args.func(args) == 0
    if command == ["installed"] and hub == "ouroboroshub":
        path = "/api/marketplace/ouroboroshub/installed"
    assert calls == [(method, path, body)]


@pytest.mark.parametrize("hub", ["sources", "ouroboroshub"])
@pytest.mark.parametrize("source", ["ouroboros", "ouroboros-community"])
@pytest.mark.parametrize("command", ["install", "preview"])
def test_marketplace_bare_name_resolves_the_public_hub_offer(monkeypatch, hub, source, command):
    from ouroboros import cli

    calls = []

    def request(method, path, body=None):
        calls.append((method, path, body))
        return {"items": [{"id": "mine/weather", "name": "weather", "source_id": "mine"},
                          {"id": f"{source}/weather", "name": "weather", "source_id": source}]}

    monkeypatch.setattr(cli, "_client", lambda _args: SimpleNamespace(request=request))
    args = cli.build_parser().parse_args(["marketplace", hub, command, "weather"])
    assert args.func(args) == 0
    action = (("POST", "/api/skill-catalog/install", {"id": f"{source}/weather"}) if command == "install"
              else ("GET", f"/api/skill-catalog/item/{source}/weather", None))
    assert calls == [("GET", "/api/skill-catalog/search?q=weather", None), action]


@pytest.mark.parametrize("command", ["install", "preview"])
def test_marketplace_bare_name_does_not_select_a_different_source(monkeypatch, command):
    from ouroboros import cli

    calls = []
    def request(method, path, body=None):
        calls.append(method)
        return {"items": [{"id": "mine/weather", "name": "weather", "source_id": "mine"}]}

    monkeypatch.setattr(cli, "_client", lambda _args: SimpleNamespace(request=request))
    args = cli.build_parser().parse_args(["marketplace", "sources", command, "weather"])
    with pytest.raises(cli.CLIError, match="catalog id"):
        args.func(args)
    assert calls == ["GET"]
