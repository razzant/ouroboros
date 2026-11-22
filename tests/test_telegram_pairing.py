"""The Telegram skill's pairing module as the core loads it: one file by path, standard library only.

``bot_identity`` talks to a real HTTP server on loopback that plays the Bot API; nothing leaves the machine.
The bridge's own use of the module (binding, ``/start``, the first ``/panic``) lives in
tests/test_telegram_bridge_safety.py.
"""
from __future__ import annotations

import ast
import contextlib
import http.server
import importlib.util
import json
import logging
import sys
import threading
import time
from pathlib import Path

import pytest

SKILL = Path(__file__).resolve().parents[1] / "skills" / "telegram"
TOKEN = "123456:SECRET-part_of-the-token"


def _standalone():
    """Load lib/telegram_pairing.py alone, as the core's onboarding does: no package, no bridge."""
    spec = importlib.util.spec_from_file_location("telegram_pairing_standalone", SKILL / "lib" / "telegram_pairing.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


pairing = _standalone()


def test_the_module_needs_nothing_beyond_the_standard_library():
    tree = ast.parse((SKILL / "lib" / "telegram_pairing.py").read_text(encoding="utf-8"))
    top = {node.module.split(".")[0] if isinstance(node, ast.ImportFrom) else alias.name.split(".")[0]
           for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
           for alias in node.names}
    assert not any(isinstance(node, ast.ImportFrom) and node.level for node in ast.walk(tree))  # no relative imports
    assert top <= set(sys.stdlib_module_names) | {"__future__"}


class _BotApi(http.server.ThreadingHTTPServer):
    """A Bot API stand-in on loopback: ``answers`` maps a path to (status, body, headers, delay)."""

    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.answers: dict = {}
        self.paths: list = []

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 — http.server's name
        self.server.paths.append(self.path)
        status, body, headers, delay = self.server.answers.get(self.path, (404, b'{"ok":false,"error_code":404}', {}, 0))
        time.sleep(delay)
        self.send_response(status)
        for key, value in {"Content-Type": "application/json", **headers}.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        with contextlib.suppress(OSError):
            self.wfile.write(body)

    def log_message(self, *_args):  # the stand-in itself prints nothing
        pass


@pytest.fixture
def bot_api():
    server = _BotApi()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


ME = f"/bot{TOKEN}/getMe"


@pytest.mark.parametrize("answer,expected", [
    ((200, json.dumps({"ok": True, "result": {"id": 123456, "is_bot": True, "username": "pair_bot"}}).encode(), {}, 0),
     {"ok": True, "username": "pair_bot", "id": 123456}),
    ((401, b'{"ok":false,"error_code":401,"description":"Unauthorized"}', {}, 0), {"ok": False, "cause": "refused"}),
    ((404, b'{"ok":false,"error_code":404,"description":"Not Found"}', {}, 0), {"ok": False, "cause": "refused"}),
    ((200, b'{"ok":false,"error_code":401,"description":"Unauthorized"}', {}, 0), {"ok": False, "cause": "refused"}),
    ((500, b'{"ok":false,"error_code":500}', {}, 0), {"ok": False, "cause": "unreachable"}),
    ((502, b"<html>bad gateway</html>", {"Content-Type": "text/html"}, 0), {"ok": False, "cause": "unreachable"}),
    ((200, b"not json", {}, 0), {"ok": False, "cause": "unreachable"}),
    ((200, json.dumps({"ok": True, "result": {"id": 999, "username": "other_bot"}}).encode(), {}, 0),
     {"ok": False, "cause": "unreachable"}),  # an answer naming another bot names no bot of this token
    ((200, json.dumps({"ok": True, "result": {"id": 123456, "username": "bad name/?"}}).encode(), {}, 0),
     {"ok": False, "cause": "unreachable"}),  # a name a t.me link cannot carry as is
    ((200, b"{}", {}, 1.5), {"ok": False, "cause": "unreachable"}),  # past the timeout
], ids=["ok", "401", "404", "ok-false-401", "500", "html-502", "malformed", "other-bot", "bad-username", "timeout"])
def test_bot_identity_says_which_bot_or_why_not_and_asks_one_bare_getme(bot_api, caplog, answer, expected):
    caplog.set_level(logging.DEBUG)
    bot_api.answers[ME] = answer
    result = pairing.bot_identity(f"  {TOKEN}\n", bot_api.origin, timeout=0.5)
    assert result == expected
    assert bot_api.paths == [ME]  # one getMe, nothing else (no setMyCommands, no menu, no polling)
    text = json.dumps(result) + "".join(record.getMessage() for record in caplog.records)
    assert "SECRET" not in text and bot_api.origin not in text and "127.0.0.1" not in text


def test_a_redirect_is_never_followed(bot_api):
    """The token travels in the path, so a redirect is refused unfollowed: the second address is never asked."""
    elsewhere = f"/bot{TOKEN}/elsewhere"
    bot_api.answers[ME] = (302, b"", {"Location": f"{bot_api.origin}{elsewhere}"}, 0)
    bot_api.answers[elsewhere] = (200, b'{"ok":true,"result":{"id":123456,"username":"pair_bot"}}', {}, 0)
    assert pairing.bot_identity(TOKEN, bot_api.origin) == {"ok": False, "cause": "unreachable"}
    assert bot_api.paths == [ME]


@pytest.mark.parametrize("token", ["", "abc", "123", "123:", ":abc", "123:abc/../x", "123:a b", None, 123])
def test_a_value_that_cannot_be_a_token_is_refused_without_a_request(bot_api, token):
    assert pairing.bot_identity(token, bot_api.origin) == {"ok": False, "cause": "refused"}
    assert bot_api.paths == []


@pytest.mark.parametrize("base", ["ftp://127.0.0.1", "http://127.0.0.1:1/path", "http://u:p@127.0.0.1:1",
                                  "http://127.0.0.1:0", "http://127.0.0.1:1"])
def test_an_invalid_or_unreachable_base_reads_unreachable(base):
    result = pairing.bot_identity(TOKEN, base, timeout=2)
    assert result == {"ok": False, "cause": "unreachable"} and "SECRET" not in json.dumps(result)


def test_the_declared_base_is_read_as_the_bridge_reads_it():
    """One rule: whatever the bridge's settings accept, the core's check calls the same origin; what they refuse,
    it refuses too (the bridge's validator is the reference; httpx and urllib spell the same origin alike here)."""
    scripts = str(SKILL / "scripts")
    sys.path.insert(0, scripts)
    try:
        from telegram_settings import validate_telegram_api_base
    finally:
        sys.path.remove(scripts)
    for value in ("", "https://api.telegram.org", "https://api.telegram.org/", "http://127.0.0.1:8081",
                  "http://127.0.0.1:8081/", "http://localhost:80", "https://example.com:8443", "http://[::1]:8081",
                  "http://LOCALHOST:8081", " http://127.0.0.1:8081 "):
        assert pairing.canonical_api_base(value) == validate_telegram_api_base(value), value
    for value in ("ftp://x", "http://x/path", "http://u:p@x", "http://x?q", "http://x#f", "http://x:0",
                  "http://x:70000", "http://"):
        with pytest.raises(ValueError):
            validate_telegram_api_base(value)
        with pytest.raises(ValueError):
            pairing.canonical_api_base(value)


def test_saved_api_base_reads_the_skills_own_settings_and_never_falls_back_to_the_cloud(tmp_path, bot_api):
    """No settings, or none naming a server: the cloud (None). A saved server: that origin. Settings that cannot
    be read: ``UNAVAILABLE``, which ``bot_identity`` answers as unreachable without sending the token anywhere."""
    assert pairing.saved_api_base(tmp_path) is None
    (tmp_path / "settings.json").write_text(json.dumps({"TELEGRAM_CHAT_ID": "42"}), encoding="utf-8")
    assert pairing.saved_api_base(tmp_path) is None
    (tmp_path / "settings.json").write_text(json.dumps({"TELEGRAM_API_BASE": f" {bot_api.origin} "}), encoding="utf-8")
    assert pairing.saved_api_base(tmp_path) == bot_api.origin
    bot_api.answers[ME] = (200, b'{"ok":true,"result":{"id":123456,"username":"pair_bot"}}', {}, 0)
    assert pairing.bot_identity(TOKEN, pairing.saved_api_base(tmp_path))["ok"] is True and bot_api.paths == [ME]
    for broken in ("{broken", "[1, 2]"):
        (tmp_path / "settings.json").write_text(broken, encoding="utf-8")
        assert pairing.saved_api_base(tmp_path) == pairing.UNAVAILABLE
        assert pairing.bot_identity(TOKEN, pairing.saved_api_base(tmp_path)) == {"ok": False, "cause": "unreachable"}
    assert bot_api.paths == [ME]  # nothing was sent while the server could not be known


@pytest.mark.parametrize("text,payload", [
    ("/start", ""), ("/start@pair_bot", ""), ("  /START  Abc-1_2 ", "Abc-1_2"), ("/start@pair_bot Abc", "Abc"),
    ("/start two words", "two words"), ("/started", None), ("/start@", None), ("/start@a@b x", None),
    ("hello /start x", None), ("", None), (None, None),
])
def test_start_and_its_deep_link_payload(text, payload):
    assert pairing.start_command(text) == payload


def test_a_code_is_what_new_code_mints_never_a_short_typed_value():
    fresh = pairing.new_code()
    assert pairing.normalize_code(fresh) == fresh and len(fresh) == 32 and fresh != pairing.new_code()
    assert pairing.normalize_code("A" * 22) == "A" * 22 and pairing.normalize_code("A" * 64) == "A" * 64
    for bad in ("", "1234", "A" * 21, "A" * 65, "two words " * 3, " " + "A" * 30, "ü" * 30, "a/" * 15, None, 123):
        assert pairing.normalize_code(bad) == ""


def test_a_code_waits_until_it_binds_once_and_its_one_write_keeps_only_its_digest():
    code, other = pairing.new_code(), pairing.new_code()
    waiting = {pairing.CODE_KEY: code}
    assert pairing.pending_code(waiting) == code and pairing.pending_code({}) == ""
    assert pairing.pending_code({pairing.CODE_KEY: "1234"}) == "" and pairing.pending_code(None) == ""
    assert pairing.matching_digest(waiting, code) == pairing.code_digest(code)
    assert pairing.matching_digest(waiting, other) == "" and pairing.matching_digest(waiting, None) == ""
    assert pairing.matching_digest({}, code) == ""
    # The pairing's write: the owner, the code cleared and its digest kept, so the same code never waits again.
    patch = pairing.paired(waiting, chat_id=42, digest=pairing.code_digest(code))
    after = {**waiting, **patch}
    assert patch == {"TELEGRAM_CHAT_ID": "42", pairing.CONSUMED_KEY: pairing.code_digest(code), pairing.CODE_KEY: ""}
    assert code not in json.dumps(after) and pairing.pending_code({**after, pairing.CODE_KEY: code}) == ""
    # A new code saved meanwhile stays waiting: the write clears only the code it consumed.
    kept = pairing.paired({pairing.CODE_KEY: other}, chat_id=42, digest=pairing.code_digest(code))
    assert pairing.CODE_KEY not in kept and pairing.pending_code({pairing.CODE_KEY: other, **kept}) == other


def test_the_first_writer_binds_only_while_nothing_waits_and_its_start_payload_spends_that_code():
    code = pairing.new_code()
    assert not pairing.first_writer_may_bind(True) and pairing.first_writer_may_bind(False)
    assert pairing.first_written({}, chat_id=42) == {"TELEGRAM_CHAT_ID": "42"}
    assert pairing.first_written({pairing.CODE_KEY: code}, chat_id=42, payload=code) == {}  # a code waits
    assert pairing.first_written({"TELEGRAM_CHAT_ID": "7"}, chat_id=42) == {}  # an owner is bound meanwhile
    bound = pairing.first_written({}, chat_id=42, payload=code)  # the backlog's /start <code>, before the code
    assert bound == {"TELEGRAM_CHAT_ID": "42", pairing.CONSUMED_KEY: pairing.code_digest(code)}
    assert pairing.pending_code({**bound, pairing.CODE_KEY: code}) == ""  # the code delivered after it: spent
    assert pairing.pending_code({**bound, pairing.CODE_KEY: pairing.new_code()})  # any other code waits
    assert pairing.first_written({}, chat_id=42, payload="not-a-code") == {"TELEGRAM_CHAT_ID": "42"}
    assert pairing.owner_chat({"TELEGRAM_CHAT_ID": " 42 "}) == 42
    assert pairing.owner_chat({"TELEGRAM_CHAT_ID": "x"}) == pairing.owner_chat(None) == 0


def _registered(tmp_path):
    """What the shipped entry point registers, collected from ``plugin.register(api)`` itself: its routes (with
    their methods) and its supervised tasks, up to the settings form, which comes after every route."""
    from tests.test_telegram_bridge_safety import FakeApi, _load_plugin

    plugin = _load_plugin(tmp_path)

    class Registered(Exception):
        pass

    class RegistrationApi(FakeApi):
        def __init__(self, state_dir):
            super().__init__(state_dir)
            self.routes, self.tasks = {}, []

        def get_runtime_info(self):
            return {"data_dir": str(tmp_path / "data")}

        def register_supervised_task(self, name, *_args, **_kwargs):
            self.tasks.append(name)

        def subscribe_event(self, *_args, **_kwargs):
            pass

        def register_route(self, path, handler, methods=("GET",)):
            self.routes[path] = tuple(methods)

        def register_settings_section(self, *_args, **_kwargs):
            raise Registered

    (tmp_path / "state").mkdir()
    api = RegistrationApi(tmp_path / "state")
    try:
        plugin.register(api)
    except Registered:
        pass
    finally:
        plugin.telegram_i18n.configure(None)
    return api


def test_the_connection_card_declaration_names_what_exists(tmp_path):
    """The card the core reads: the token is the one secret field; the pairing code is this skill's own setting,
    saved through a route the entry point registers, never a core Settings key; the functions it names are this
    module's; the status route and the option's route are registered by ``plugin.register``; the option's key is
    one the skill's settings form owns. Field types and links, not wording, are the contract."""
    from ouroboros.contracts.skill_manifest import parse_skill_manifest_text

    manifest = parse_skill_manifest_text((SKILL / "SKILL.md").read_text(encoding="utf-8"))
    card = manifest.raw_extra["connection_card"]
    assert manifest.secret_fields == {card["token"]: "TELEGRAM_BOT_TOKEN"}
    assert manifest.env_from_settings == ["TELEGRAM_BOT_TOKEN"]  # the code never reaches core Settings
    assert card["pairing"]["code"] == {"key": pairing.CODE_KEY, "store": "skill_settings", "route": "settings/save"}
    for name in ("identity", "new_code"):
        path, function = card["pairing"][name].split(":")
        assert path == "lib/telegram_pairing.py" and callable(getattr(pairing, function))
    assert card["pairing"]["link"].format(username="pair_bot", code="c0de") == "https://t.me/pair_bot?start=c0de"
    registered = _registered(tmp_path)
    assert registered.routes[card["pairing"]["status_route"]] == ("GET",)
    assert "POST" in registered.routes[card["pairing"]["code"]["route"]]
    assert "owner_registration" in registered.tasks
    scripts = str(SKILL / "scripts")
    sys.path.insert(0, scripts)
    try:
        from telegram_settings import _PAIRING_CODE_KEY, _SETTINGS_FORM_KEYS
    finally:
        sys.path.remove(scripts)
    assert _PAIRING_CODE_KEY == pairing.CODE_KEY and pairing.CODE_KEY not in _SETTINGS_FORM_KEYS  # never served
    [option] = card["options"]
    assert option["key"] in _SETTINGS_FORM_KEYS and option["store"] == "skill_settings"
    assert "POST" in registered.routes[option["route"]]
    assert (option["on"], option["off"], option["new_connection"]) == ("on", "off", True)
    for field in ("title", "pitch"):
        assert isinstance(card[field], str) and card[field].strip()
    assert isinstance(card["pairing"]["hint"], str) and card["pairing"]["hint"].strip()
    assert card["steps"] and all(isinstance(step["text"], str) and step["text"].strip() for step in card["steps"])
    assert all(set(step) <= {"text", "link"} for step in card["steps"])
    assert [step["link"] for step in card["steps"] if "link" in step] == ["https://t.me/BotFather"]
