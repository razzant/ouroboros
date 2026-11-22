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


def test_recall_sends_repeated_alternatives_or_one_row_encoded_and_never_starts_a_server(monkeypatch, capsys):
    """``ouroboros recall`` is the command face of ``GET /api/recall``: ``--words`` repeats (alternatives) and
    excludes ``--row``; every value is percent-encoded so ``#``, ``+``, quotes and Unicode arrive as typed."""
    import urllib.parse

    from ouroboros import cli

    seen = []

    class FakeClient:
        def request(self, method, path, body=None, *, timeout=None):
            seen.append((method, path))
            assert timeout == cli.RECALL_WAIT_SEC
            return {"mode": "search", "rows": []}

    def client(_args, **kwargs):
        seen.append(("client", kwargs))
        return FakeClient()

    monkeypatch.setattr(cli, "_client", client)
    assert cli.main(["recall", "--words", 'архив "точно так" #3', "--words", "a+b", "--order", "oldest",
                     "--room-id", "7", "--count", "5"]) == 0
    assert seen[0] == ("client", {})  # no start=True: the command never launches a server
    path = seen[1][1]
    assert path.startswith("/api/recall?") and " " not in path and "#" not in path
    assert urllib.parse.parse_qsl(path.split("?", 1)[1]) == [
        ("words", 'архив "точно так" #3'), ("words", "a+b"), ("room_id", "7"), ("order", "oldest"), ("count", "5")]
    seen.clear()
    address = "row:1@2026-11-01T10:00:00Z#" + "a" * 64
    assert cli.main(["recall", "--row", address]) == 0
    assert urllib.parse.parse_qsl(seen[1][1].split("?", 1)[1]) == [("row", address)]
    for wrong in (["recall"], ["recall", "--words", "a", "--row", address]):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(wrong)
    capsys.readouterr()


def test_recall_finds_the_server_as_chat_send_does_and_runs_on_the_standard_library(tmp_path, monkeypatch):
    """Explicit ``--url`` wins, then ``OUROBOROS_URL``, then the recorded port, then 127.0.0.1:8765; with no
    site-packages at all the command still runs (an unreachable server is its exit-2 error, never an import one)."""
    import os
    import pathlib
    import subprocess

    from ouroboros import cli

    seen = []
    monkeypatch.setattr(cli.OuroborosHTTPClient, "request", lambda self, method, path, body=None, **_k: seen.append(
        self.base_url + path.split("?", 1)[0]) or {})
    port_file = tmp_path / "server_port"
    port_file.write_text("41999", encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_PORT_FILE", str(port_file))
    monkeypatch.delenv("OUROBOROS_URL", raising=False)
    cli.main(["recall", "--words", "x"])
    monkeypatch.setenv("OUROBOROS_URL", "http://127.0.0.1:42001")
    cli.main(["recall", "--words", "x"])
    cli.main(["--url", "http://127.0.0.1:42002", "recall", "--words", "x"])
    port_file.unlink()
    monkeypatch.delenv("OUROBOROS_URL")
    cli.main(["recall", "--words", "x"])
    assert [url for url in seen if url.endswith("/api/recall")] == [
        "http://127.0.0.1:41999/api/recall", "http://127.0.0.1:42001/api/recall",
        "http://127.0.0.1:42002/api/recall", "http://127.0.0.1:8765/api/recall"]
    repo = pathlib.Path(__file__).resolve().parents[1]
    done = subprocess.run([sys.executable, "-S", "-m", "ouroboros.cli", "--url", "http://127.0.0.1:9", "recall",
                           "--words", "x"], cwd=repo, capture_output=True, text=True, timeout=60,
                          env={**os.environ, "PYTHONPATH": str(repo), "PYTHONDONTWRITEBYTECODE": "1"})
    assert done.returncode == 2 and "cannot reach Ouroboros server at http://127.0.0.1:9" in done.stderr, done.stderr


def test_recall_waits_for_a_slow_whole_conversation_search_past_the_clients_default_wait(monkeypatch, capsys):
    """A search of every room reads the whole conversation; the command waits for it (``RECALL_WAIT_SEC``) where the
    client's ordinary wait would give up. The default client is shortened here so the server's delay exceeds it."""
    import http.server
    import threading

    from ouroboros import cli

    class Slow(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - the stdlib's handler name
            if self.path.startswith("/api/recall"):
                threading.Event().wait(1.5)
            body = json.dumps({"mode": "search", "rows": []} if self.path.startswith("/api/recall") else {}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Slow)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        monkeypatch.setattr(cli, "_client", lambda _args, **_kwargs: cli.OuroborosHTTPClient(url, timeout=0.5))
        with pytest.raises(cli.ConnectionCLIError):  # the ordinary wait ends before the answer
            cli.OuroborosHTTPClient(url, timeout=0.5).request("GET", "/api/recall?words=x")
        assert cli.main(["recall", "--words", "x"]) == 0
        assert json.loads(capsys.readouterr().out) == {"mode": "search", "rows": []}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["recall", "--help"])
    rendered = " ".join(capsys.readouterr().out.split())
    assert ("Without a usable index it scans the conversation and can take a minute or more. Give --room-id to "
            "search one room.") in rendered
    assert "only the owner's rows (a card they answered included)" in rendered and "my human" not in rendered
