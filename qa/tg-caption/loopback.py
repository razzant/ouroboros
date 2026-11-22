"""Wire observations and the candidate server for the Telegram conversation."""

import asyncio
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import sys
import threading
from time import monotonic_ns

import httpx

from ouroboros.server_process import read_service_bindings
from ouroboros.skill_install_record import read_install_record
from ouroboros.skill_loader import (
    find_skill, load_skill, requested_core_setting_keys, requested_skill_permissions, save_enabled, save_skill_grants,
)
from ouroboros.skill_starter import read_marker
from tests.candidate_checkout import require_candidate_interpreter, require_running_supervisor
from tests.system_e2e.harness import KeylessIsolatedServer, keyless_settings, wait_until, write_settings_file
from tests.test_telegram_api_base_loopback import _BotServer
from tests.test_telegram_proxy import _TOKEN


BOT_ID = _TOKEN.split(":", 1)[0]  # repository fixture, accepted by _BotServer only
CREATE_OR_EDIT = {"sendMessage", "sendPhoto", "sendVideo", "sendDocument", "sendAudio", "editMessageText"}


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


class Bot(_BotServer):
    """The existing HTTP fixture, with one update per poll and a record of its responses."""

    def __init__(self, first_id, files, state, note):
        super().__init__(first_update_id=first_id, files=files)
        self.state = state
        self.note = note
        self.lock = threading.Lock()
        self.events = []

    def offer(self, *updates):
        with self.lock:
            for update in updates:
                actual = self.push({"message": update["message"]})
                assert actual == update["update_id"]

    def _get_updates(self, offset):
        # The client still acknowledges every update. A/B/Reply can never all
        # enter the same cached batch and trivially suppress the later replay.
        return super()._get_updates(offset)[:1]

    def respond(self, verb, path, form):
        method = "file" if path.startswith("/file/") else path.rsplit("/", 1)[-1]
        with self.lock:
            event = {"method": method, "verb": verb, "at": monotonic_ns()}
            self.events.append(event)
            try:
                code, body, content_type = super().respond(verb, path, form)
            except KeyError:
                if method != "getFile":
                    raise
                code, body, content_type = 400, b'{"ok":false,"error_code":400}', "application/json"
            event["status"] = code
            if method == "getUpdates" and code == 200:
                updates = json.loads(body)["result"]
                event.update(offset=int(form.get("offset") or 0), updates=[u["update_id"] for u in updates],
                             messages=[u["message"]["message_id"] for u in updates])
                if updates:
                    self.note(f"POLL  offset={event['offset']}  update={event['updates'][0]}  message={event['messages'][0]}")
            elif method == "getFile":
                event["file_id"] = form.get("file_id")
            elif method == "file":
                event.update(path=path.removeprefix(f"/file/bot{_TOKEN}/"), sha256=digest(body), size=len(body))
            elif method == "getMe" and code == 200:
                response = json.loads(body)
                response["result"]["id"] = int(BOT_ID)
                body = json.dumps(response).encode()
            elif method == "sendMessage" and code == 200:
                # Distinct provider IDs make two physical sends distinguishable.
                mid = 8000 + sum(e["method"] == method for e in self.events)
                body = json.dumps({"ok": True, "result": {"message_id": mid}}).encode()
                event.update(chat=str(form["chat_id"]), text=form["text"], provider_id=mid)
                self.note(f"REQUEST  Bot API sendMessage  peer message_id={mid}")
            return code, body, content_type

    def seen(self, *methods):
        with self.lock:
            return [e.copy() for e in self.events if not methods or e["method"] in methods]

    def acknowledged(self, update):
        polls = self.seen("getUpdates")
        return any(update in e.get("updates", []) and any(
            ack.get("offset", 0) > update for ack in polls[index + 1:]) for index, e in enumerate(polls))

    @contextmanager
    def listening(self):
        # Run the fixture's real TCP listener while the synchronous browser drives the SPA.
        with asyncio.Runner() as runner:
            runner.run(self.__aenter__())
            loop = runner.get_loop()
            thread = threading.Thread(target=loop.run_forever, daemon=True)
            thread.start()
            try:
                yield
            finally:
                loop.call_soon_threadsafe(loop.stop)
                thread.join(5)
                assert not thread.is_alive()
                runner.run(self.__aexit__(None, None, None))


def observe_posts(model, note):
    """Capture bytes at the listener before the stub classifies, holds or answers them."""
    handler = model._server.RequestHandlerClass
    posts = []

    class Observed(handler):
        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            body = json.loads(raw)
            posts.append({"at": monotonic_ns(), "path": self.path, "sha256": digest(raw),
                          "local": self.client_address[0] == self.server.server_address[0] == "127.0.0.1",
                          "body": body})
            if body.get("tools"):
                note(f"HTTP  model POST  local={self.client_address[0] == '127.0.0.1'}  bytes={len(raw)}  sha256={digest(raw)}")
            original = self.rfile
            try:
                self.rfile = io.BytesIO(raw)
                super().do_POST()
            finally:
                self.rfile = original

    model._server.RequestHandlerClass = Observed
    return posts  # in memory only


@contextmanager
def telegram_server(data, bot, model, chat, candidate, note):
    root = Path.cwd()
    state = bot.state
    # A child of state/skills before boot classifies this guest as an upgrade,
    # skipping the first-start starter selection entirely.
    data.mkdir()
    settings = keyless_settings(model, OUROBOROS_MAX_WORKERS=1, OUROBOROS_UI_LANGUAGE="en",
                                OUROBOROS_STARTER_SKILLS='["builtin/telegram"]', OUROBOROS_STARTER_SKILLS_TURN_ON=False)
    settings["TELEGRAM_BOT_TOKEN"] = _TOKEN
    write_settings_file(data / "settings.json", settings)
    server = KeylessIsolatedServer(root, data, data / "settings.json")
    assert server.port != server.host_service_port
    require_candidate_interpreter(sys.executable, server._env(), root)
    try:
        note("BOOT  installing the bundled Telegram skill")
        server.start()
        ready = wait_until(lambda: read_marker(data).get("state") in {"upgrade", "pending", "done"}, 65)
        marker = read_marker(data)
        result = marker.get("results", {}).get("builtin/telegram", {})
        phase, status = marker.get("state"), result.get("status")
        phase = phase if phase in {"fresh", "upgrade", "running", "pending", "done"} else "unknown"
        status = status if status in {"installed", "pending", "skipped", "failed"} else "missing"
        note(f"STARTER  phase={phase} selected={'builtin/telegram' in (marker.get('requested') or [])} result={status}")
        assert ready and phase == "done" and status == "installed"
        skill = find_skill(data, "telegram")
        bundled = load_skill(root / "skills" / "telegram", data)
        record = read_install_record(data, "telegram") or {}
        release = record.get("release", {})
        native = bool(skill and skill.skill_dir == data.resolve() / "skills" / "native" / "telegram"
                      and record.get("source", {}).get("kind") == "builtin")
        exact = bool(skill and bundled and skill.content_hash
                     and skill.content_hash == bundled.content_hash == release.get("digest")
                     and str(skill.manifest.version or "") == str(bundled.manifest.version or "") == release.get("version"))
        reviewed = bool(skill and skill.review.gate_for(skill.content_hash)["executable_review"])
        disabled = bool(skill and not skill.enabled)
        note(f"INSTALL  present={skill is not None} builtin={native} exact={exact} review_gate={reviewed} disabled={disabled}")
        assert native and exact and reviewed and disabled
        note(f"INSTALL  candidate Telegram sha256={skill.content_hash}")
        state.mkdir(parents=True, exist_ok=True)
        write_settings_file(state / "settings.json", {
            "TELEGRAM_CHAT_ID": str(chat), "TELEGRAM_API_BASE": bot.origin, "TELEGRAM_MINIAPP_ENABLED": "off",
            "TELEGRAM_MIRROR_MODE": "telegram_only", "TELEGRAM_MIRROR_PROGRESS": "off",
        })
        keys = requested_core_setting_keys(list(skill.manifest.env_from_settings or []))
        perms = requested_skill_permissions(list(skill.manifest.permissions or []), list(skill.manifest.subscribe_events or []))
        save_skill_grants(data, "telegram", keys, content_hash=skill.content_hash, requested_keys=keys,
                          granted_permissions=perms, requested_permissions=perms)
        save_enabled(data, "telegram", True)
        server.stop()
        assert not server.stop_unconfirmed
        assert not bot.seen("getUpdates")
        note("BOOT  starting the enabled bridge and the Main page")
        server.start()
        assert server.attestation["repo_head"] == candidate
        binding = read_service_bindings(data)["main"]
        assert (binding["pid"], binding["port"]) == (server.proc.pid, server.port)
        with httpx.Client(base_url=server.base_url, trust_env=False) as http:
            live = http.get("/api/state").raise_for_status().json()
            require_running_supervisor(live)
            served = live.get("served_ui")
            assert served is None or served.get("source") == "builtin"
            asset = "modules/chat_attachments.js"
            js = http.get("/static/" + asset).raise_for_status().content
            assert js == (root / "web" / asset).read_bytes()
            language = http.get("/api/ui/i18n").raise_for_status().json()
            assert language["english"] and language["language"] == "en" and not language["memory_error"]
        polled = wait_until(lambda: any(e.get("status") == 200 for e in bot.seen("getUpdates")), 65)
        polls = bot.seen("getUpdates")
        note(f"BRIDGE  getMe_ok={any(e.get('status') == 200 for e in bot.seen('getMe'))} "
             f"poll_seen={bool(polls)} poll_ok={bool(polled)}")
        assert polled and all(e.get("updates") == [] for e in polls)
        note(f"READY  bundled Telegram polling; English SPA asset sha256={digest(js)}")
        yield server
    finally:
        server.stop()
        assert not server.stop_unconfirmed
