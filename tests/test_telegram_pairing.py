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


def test_saved_api_base_reads_the_skills_own_settings(tmp_path):
    assert pairing.saved_api_base(tmp_path) is None
    (tmp_path / "settings.json").write_text(json.dumps({"TELEGRAM_API_BASE": " http://127.0.0.1:8081 "}),
                                            encoding="utf-8")
    assert pairing.saved_api_base(tmp_path) == "http://127.0.0.1:8081"


@pytest.mark.parametrize("text,payload", [
    ("/start", ""), ("/start@pair_bot", ""), ("  /START  Abc-1_2 ", "Abc-1_2"), ("/start@pair_bot Abc", "Abc"),
    ("/start two words", "two words"), ("/started", None), ("/start@", None), ("/start@a@b x", None),
    ("hello /start x", None), ("", None), (None, None),
])
def test_start_and_its_deep_link_payload(text, payload):
    assert pairing.start_command(text) == payload


def test_a_code_is_what_a_start_parameter_carries_exactly():
    fresh = pairing.new_code()
    assert pairing.normalize_code(fresh) == fresh and len(fresh) == 32 and fresh != pairing.new_code()
    assert pairing.normalize_code("A" * 64) == "A" * 64
    for bad in ("", "A" * 65, "two words", " padded", "ünicode", "a/b", None, 123):
        assert pairing.normalize_code(bad) == ""


def test_a_code_waits_until_it_binds_once_and_never_matches_another_payload(tmp_path):
    code = pairing.new_code()
    assert pairing.pending(tmp_path, code) and not pairing.pending(tmp_path, "") and not pairing.pending(tmp_path, "no!")
    assert pairing.matching_digest(tmp_path, code, code) == pairing.code_digest(code)
    assert pairing.matching_digest(tmp_path, code, code + "x") == ""
    assert pairing.matching_digest(tmp_path, "", "") == "" and pairing.matching_digest(tmp_path, code, None) == ""
    assert not pairing.first_writer_may_bind(True) and pairing.first_writer_may_bind(False)
    pairing.record_consumed(tmp_path, pairing.code_digest(code))
    stored = (tmp_path / pairing.CONSUMED_FILE).read_text(encoding="utf-8")
    assert code not in stored  # only its digest is kept
    assert not pairing.pending(tmp_path, code) and pairing.matching_digest(tmp_path, code, code) == ""
    fresh = pairing.new_code()
    assert pairing.pending(tmp_path, fresh) and pairing.matching_digest(tmp_path, fresh, fresh)  # a new code binds
    (tmp_path / pairing.CONSUMED_FILE).write_text("{broken", encoding="utf-8")
    assert pairing.pending(tmp_path, code)  # an unreadable record never leaves the owner unable to pair
    assert pairing.local_status(tmp_path, code) == {"owner_bound": False, "pending": True}
    (tmp_path / "settings.json").write_text(json.dumps({"TELEGRAM_CHAT_ID": "42"}), encoding="utf-8")
    assert pairing.local_status(tmp_path, "") == {"owner_bound": True, "pending": False}


def test_the_connection_card_declaration_names_what_exists():
    """The card the core reads: the token and code are declared secret fields; the functions it names are this
    module's; its status route and its option's route are the ones the bridge registers; an option's key is one
    the skill's own settings form owns, and its "on"/"off" keys stay text in YAML."""
    from ouroboros.contracts.skill_manifest import parse_skill_manifest_text

    manifest = parse_skill_manifest_text((SKILL / "SKILL.md").read_text(encoding="utf-8"))
    card = manifest.raw_extra["connection_card"]
    assert manifest.secret_fields[card["token"]] == "TELEGRAM_BOT_TOKEN"
    assert manifest.secret_fields[card["pairing"]["code"]] == pairing.CODE_KEY
    assert pairing.CODE_KEY in manifest.env_from_settings
    for name in ("identity", "new_code"):
        path, function = card["pairing"][name].split(":")
        assert path == "lib/telegram_pairing.py" and callable(getattr(pairing, function))
    assert card["pairing"]["link"].format(username="pair_bot", code="c0de") == "https://t.me/pair_bot?start=c0de"
    assert card["pairing"]["status_route"] == "pairing/status"
    plugin_source = (SKILL / "plugin.py").read_text(encoding="utf-8")
    owner_source = (SKILL / "lib" / "telegram_owner.py").read_text(encoding="utf-8")
    assert 'api.register_route("pairing/status"' in owner_source and "register_pairing(api)" in plugin_source
    scripts = str(SKILL / "scripts")
    sys.path.insert(0, scripts)
    try:
        from telegram_settings import _SETTINGS_FORM_KEYS
    finally:
        sys.path.remove(scripts)
    [option] = card["options"]
    assert option["key"] in _SETTINGS_FORM_KEYS and option["store"] == "skill_settings"
    assert option["route"] == "settings/save" and 'api.register_route("settings/save"' in plugin_source
    assert (option["on"], option["off"], option["new_connection"]) == ("on", "off", True)
