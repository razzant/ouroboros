"""Joint qualification of the OuroborosHub Telegram history import against this checkout's real Host.

The Hub skill's own ``HistoryImporter``, live intake (``Participant``), ``Journal`` and loopback HTTP
``Host`` client run unchanged. Only Telegram is synthetic: the Hub's ``history_fakes`` provider, with
the sender entities a real ``getHistory`` answer carries. The Host is this checkout's
``create_host_service_app`` on a loopback socket, served as ``server_boot.serve_host_service`` serves
it, with a seeded skill token, an owner-connected place and consciousness switched on by the real state
writer. Its focus runner is the one stand-in: no model runs here, so a recorder takes its place, and
reaching it is the observable that attention admitted an event. Everything read back is real: the
dialogue store, the history endpoint, the places API, the wake observation and the memory facts.

CI never collects this module: it needs a Hub checkout, and its name is not ``test_*.py``. safe_test keeps
only its own environment, so the Hub path goes through ``env`` inside the command:

    python -I -S scripts/safe_test.py --temp-parent /tmp -- /usr/bin/env \\
        OUROBOROS_TELEGRAM_SKILL_ROOT=<hub>/skills/telegram-user-session \\
        .venv/bin/python -m pytest -p no:cacheprovider tests/telegram_history_joint.py

``OUROBOROS_JOINT_MUTATION`` names one deliberate fault from ``MUTATIONS``: a red control run.
``.review-evidence/joint-b520/run.sh`` runs every step and checks each one's exit and outcome counts.

The core repair (task b520a762) adapted QA 2f335d42's original. That original, its strict-xfail REDs and
run are retained in ``.review-evidence/joint-original``. Host and companion restarts are rebuilt in this
interpreter, never as new processes. The page around a row is judged by the endpoint's own coverage keys.
The Hub (r3, staged index b954733a) sends Telegram's message id as ``message.source_order`` and its date as
``message.sent_at`` on live and history events, so rows of one Telegram second read in send order.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import threading
import time
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import pytest

from ouroboros import dialogue_store as ds
from ouroboros import places as p
from ouroboros.presence_bindings import conversation_key
from ouroboros.presence_runner import _stable_numeric_id

SKILL = Path(os.environ.get("OUROBOROS_TELEGRAM_SKILL_ROOT") or "/nonexistent")
pytestmark = pytest.mark.skipif(not (SKILL / "userbot" / "history_import.py").is_file(),
                                reason="set OUROBOROS_TELEGRAM_SKILL_ROOT to a telegram-user-session checkout")
CONNECTOR = "telegram-user-session"
TOKEN = "joint-qualification-token"
ACCOUNT, GROUP, MIRA, TOM = 8667525238, -1000000000123, 4242, 4243
DAY, MINUTE = timedelta(days=1), timedelta(minutes=1)


def hub():
    """The Hub skill's own modules, imported from its checkout (bytecode goes to safe_test's prefix)."""
    pytest.importorskip("telethon")
    for path in (SKILL / "tests", SKILL):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import history_fakes
    from telethon import types
    from userbot import config, history_import, normalize, participant, participant_state, presence, telegram
    return SimpleNamespace(fakes=history_fakes, T=types, H=history_import, config=config, normalize=normalize,
                           participant=participant, state=participant_state, presence=presence, telegram=telegram)


def room(peer):
    """The Host's chat id of one Telegram conversation's General thread."""
    return _stable_numeric_id("presence-conversation", conversation_key("telegram", str(ACCOUNT), str(peer), ""))


def files(root):
    return {str(path.relative_to(root)): path.stat().st_mtime_ns for path in Path(root).rglob("*") if path.is_file()}


class HostServer:
    """This checkout's Host Service app on one loopback socket, as ``server_boot.serve_host_service`` runs it,
    on a server thread of this interpreter."""

    def __init__(self, root, runner, port=0):
        import uvicorn

        from ouroboros.gateway.host_service import create_host_service_app
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", port))
        self.port = self.sock.getsockname()[1]
        self.server = uvicorn.Server(uvicorn.Config(create_host_service_app(root, presence_runner=runner),
                                                    host="127.0.0.1", port=self.port, log_level="warning",
                                                    log_config=None))
        self.server.capture_signals = nullcontext  # an embedded server never owns process signals
        self.thread = threading.Thread(target=self.server.run, kwargs={"sockets": [self.sock]},
                                       name="joint-host-service", daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 20
        while not self.server.started:
            assert self.thread.is_alive() and time.monotonic() < deadline, "the Host Service did not start"
            time.sleep(0.01)

    def stop(self):
        self.server.should_exit = True
        self.thread.join(20)
        self.sock.close()
        assert not self.thread.is_alive(), "the Host Service thread did not stop"


class Recorder:
    """The focus runner's stand-in. No model runs; a call means attention admitted that event."""

    def __init__(self):
        self.events = []

    def __call__(self, *, admission, event, staged_files, admitted):
        from ouroboros.presence_runner import PresenceTurnResult, presence_turn_task_id
        self.events.append(event.source_event_id)
        return PresenceTurnResult(outcome="silent", text="",
                                  task_id=presence_turn_task_id(admission.binding_id, event.source_event_id))


class Wire:
    """The Hub client's real loopback transport, recording each history request and its answer. A queued
    ``after`` loss lets the request land and then loses its answer."""

    def __init__(self, send):
        self.send, self.losses, self.calls, self.history, self.answers = send, [], [], [], []

    def __call__(self, method, path, payload, headers):
        self.calls.append((method, path.split("?")[0]))
        if path != "/presence/history":
            return self.send(method, path, payload, headers)
        loss = self.losses.pop(0) if self.losses else None
        self.history.append(json.dumps(payload, sort_keys=True))
        status, body = self.send(method, path, payload, headers)
        self.answers.append((status, body))
        if loss == "after":
            raise ConnectionResetError("the answer was lost after the request landed")
        return status, body


class Companion:
    """One telegram-user-session companion over its state directory (journal, intake, importer), built in this
    interpreter. A ``fresh`` one is configured here; otherwise its directory already holds its configuration."""

    def __init__(self, world, state, fresh=True):
        h = world.hub
        if fresh:
            state.mkdir(exist_ok=True)
            h.config.save_config(state, h.config.Config(123, "a" * 32, "+1234567890", account_id=ACCOUNT))
            h.state.save_policy(state, {"mode": "participant", "outgoing": False})
        self.world, self.state, self.journal = world, state, h.state.Journal(state)
        for peer, title, kind in world.chats if fresh else ():
            self.journal.remember_chat(ACCOUNT, peer, title, kind, "dialogs")
        host = h.participant.Host(world.url, h.presence.PresenceToken(TOKEN))
        self.wire = Wire(host.transport)
        host.transport = self.wire
        world.wires.append(self.wire)
        adapter = h.telegram.TelegramAdapter(state / "unused.session", 123, "a" * 32, client=world.telegram)
        self.engine = h.participant.Participant(state, adapter, host, self.journal, ACCOUNT)
        self.importer = h.H.HistoryImporter(self.engine, clock=lambda: world.clock[0])

    def deliver_live(self, message, peer, sender, kind="new"):
        """Live intake of one Telegram update: the Hub's capture, then its POST /presence/turn."""
        world = self.world
        event = SimpleNamespace(message=message, chat_id=peer, sender_id=sender, raw_text=message.message,
                                is_private=peer > 0)
        event.get_sender = lambda: asyncio.sleep(0, result=world.people[sender])
        event.get_chat = lambda: world.telegram.get_entity(peer)

        async def deliver():
            await self.engine.capture(event, kind)
            await self.engine.incoming_once()
        asyncio.run(deliver())

    def report_delivered(self, peer, message):
        """This companion sent ``message`` and its receipt goes through POST /presence/delivery."""
        key = f"auto:{message.id}"
        self.journal.report(key, {"schema_version": 1, "delivery_id": key, "part_id": "0", "state": "delivered",
                                  "provider": "telegram", "account_id": str(ACCOUNT), "conversation_id": str(peer),
                                  "thread_id": "", "text": message.message, "format": "text",
                                  "message": {"message_id": str(message.id)}, "origin": {"kind": "automatic"}})
        asyncio.run(self.engine.reports_once())
        assert self.journal.db.execute("SELECT state FROM reports WHERE id=?", (key,)).fetchone()[0] == "recorded"

    def close(self):
        self.journal.close()


class World:
    """One Ouroboros data root with its Host Service, and the companions and synthetic Telegram beside it."""

    def __init__(self, base, *, place_state="active", connect=True, consciousness=True, entities=None,
                 companion_state=None):
        self.hub = h = hub()
        self.data = base / "d"
        self.data.mkdir()
        if consciousness:  # through the real state writer, as the owner's switch does; no admission stand-in
            from supervisor import state
            state.init(self.data)
            state.init_state()
            state.update_state(lambda st: st.update(bg_consciousness_enabled=True), confirm=("bg_consciousness_enabled",))
        from tests.test_host_service_api import _seed_token
        _seed_token(self.data, skill=CONNECTOR, token=TOKEN, permissions=["presence"], manifest_permissions=["presence"])
        self.place = self.connect(place_state) if connect else None
        self.recorder = Recorder()
        self.host = HostServer(self.data, self.recorder)
        self.url = f"http://127.0.0.1:{self.host.port}"
        self.now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        self.clock = [time.time()]
        self.people = {ACCOUNT: h.fakes.user(ACCOUNT, "Ouroboros", "ouro_demo"), MIRA: h.fakes.user(MIRA, "Mira", "mira"),
                       TOM: h.fakes.user(TOM, "Tom", "tom")}
        entities = entities or {GROUP: h.fakes.group(GROUP, "Board games on Fridays"), MIRA: self.people[MIRA]}
        self.histories = {peer: [] for peer in entities}
        people = self.people

        class Telegram(h.fakes.History):
            """``getHistory`` over fixed messages, each with the sender entity Telegram sends alongside it."""

            async def get_messages(self, entity, *args, **kwargs):
                found = await super().get_messages(entity, *args, **kwargs)
                for message in found if isinstance(found, list) else [found] if found is not None else []:
                    message._sender = people.get(ACCOUNT if message.out else message.sender_id)
                return found
        self.telegram = Telegram(entities, self.histories)
        self.chats = ((GROUP, "Board games on Fridays", "supergroup"), (MIRA, "Mira", "private"))
        self.companions, self.wires = [], []
        self.companion = self.open_companion(companion_state or base / "h", fresh=companion_state is None)

    # ---- the owner and the Host ----------------------------------------------------------------
    def connect(self, state="active"):
        place = p.connect_place(self.data, connector=CONNECTOR, provider="telegram", account_id=str(ACCOUNT),
                                label="Telegram", wakes=["any_message"], state=state)
        self.place = place
        return place

    def restart_host(self):
        """A Host Service restart rebuilt in this interpreter: the server stops, and a new app and server start on
        its port over the same data root with fresh in-memory state. No process ends or starts."""
        self.host.stop()
        self.host = HostServer(self.data, self.recorder, port=int(self.url.rsplit(":", 1)[1]))

    # ---- the companion and Telegram ------------------------------------------------------------
    def open_companion(self, state, fresh=True):
        companion = Companion(self, state, fresh)
        self.companions.append(companion)
        return companion

    def restart_companion(self):
        """A companion restart rebuilt in this interpreter: its journal closes, and a new companion opens the same
        state directory and marks its running import interrupted, as a starting process does
        (``interrupt_running``). The import then waits for the owner's Resume."""
        state = self.companion.state
        self.companion.close()
        self.companions.remove(self.companion)
        self.companion = self.open_companion(state, fresh=False)
        self.hub.H.interrupt_running(self.companion.journal)

    def say(self, peer, mid, text, when, sender=None, *, own=False, **kw):
        """One message in Telegram's history of ``peer``; ``own`` ones were sent from the account itself."""
        fakes = self.hub.fakes
        message = (fakes.message(mid, peer, text, when, out=True, **({"account": ACCOUNT} if peer < 0 else {}), **kw)
                   if own else fakes.message(mid, peer, text, when, sender=sender, **kw))
        self.histories[peer].append(message)
        return message

    # ---- the owner's import ---------------------------------------------------------------------
    def plan(self, chats, start, end=None, cap=500):
        h = self.hub
        spec = h.H.validate_plan({"chats": [str(chat) for chat in chats], "from": start.isoformat(),
                                  "to": (end or datetime.now(timezone.utc)).isoformat(), "max_per_chat": cap},
                                 h.state.policy(self.companion.state), ACCOUNT)
        return h.H.save_plan(self.companion.journal, ACCOUNT, spec)

    def act(self, plan, action):
        return self.hub.H.transition(self.companion.journal, ACCOUNT, plan, action, self.hub.state.policy(self.companion.state))

    def view(self, plan):
        return next(view for view in self.hub.H.plans(self.companion.journal, ACCOUNT) if view["id"] == plan)

    def step(self, count=1):
        for _ in range(count):
            self.clock[0] += 700  # past every pacing gap and backoff
            asyncio.run(self.companion.importer.once())

    def drive(self, plan, limit=80):
        for _ in range(limit):
            if self.view(plan)["state"] != "running":
                break
            self.step()
        return self.view(plan)["state"]

    def sent(self):
        """The message ids of each history request every companion process sent, in order."""
        return [[record["event"]["message"]["message_id"] for record in json.loads(body)["records"]]
                for wire in self.wires for body in wire.history]

    # ---- what the Host holds ----------------------------------------------------------------------
    def rows(self, peer):
        return [entry.row for entry in ds.iter_rows(self.data, "chat") if entry.row.get("chat_id") == room(peer)]

    def by_message(self, peer):
        """Each stored row of ``peer``'s room by its Telegram message id; every id appears once."""
        found = {}
        for row in self.rows(peer):
            mid = str((row.get("transport") or {}).get("message", {}).get("message_id"))
            assert mid not in found, ("one Telegram message stored twice", mid, found[mid], row)
            found[mid] = row
        return found

    def reading_order(self, peer):
        """The room's rows by their reading key (``place_arrivals.reading_key``: send time, order, position)."""
        from ouroboros.place_arrivals import reading_key
        return [entry.history_id for entry in sorted((entry for entry in ds.iter_rows(self.data, "chat")
                                                      if entry.row.get("chat_id") == room(peer)),
                                                     key=lambda entry: reading_key(entry.row, entry.seq))]

    def pages(self, peer, quota):
        """The room read newest first through the real history endpoint, page by page."""
        from starlette.applications import Starlette
        from starlette.routing import Route
        from starlette.testclient import TestClient

        from ouroboros.gateway.history import make_chat_history_endpoint
        found, cursor = [], None
        with TestClient(Starlette(routes=[Route("/api/chat/history", make_chat_history_endpoint(self.data))])) as client:
            while True:
                page = client.get("/api/chat/history", params={"chat_id": room(peer), "n_human": quota,
                                                               **({"cursor": cursor} if cursor else {})}).json()
                assert "error" not in page, page
                found.append(page)
                if not page["has_more"]:
                    return found
                cursor = page["next_cursor"]

    def close(self):
        try:
            for companion in self.companions:
                companion.close()
        finally:
            self.host.stop()


def _turn_fallback(monkeypatch, h):
    """A connector that sends earlier messages as new arrivals through POST /presence/turn."""
    real = h.participant.Host.call

    def call(self, method, path, body=None):
        if path != "/presence/history":
            return real(self, method, path, body)
        results = []
        for record in body["records"]:
            status, answer = real(self, "POST", "/presence/turn", {"event": {**record["event"], "facts": ["any_message"]},
                                                                   "staged_files": [], "continuation_version": 1})
            source = record["event"]["source_event_id"]
            results.append({"source_event_id": source, "ok": True, "duplicate": False, "custody": "history"}
                           if status == 200 else {"source_event_id": source, "ok": False, "code": answer.get("code")})
        conflicts = any(not result["ok"] for result in results)
        return (409 if conflicts else 200), {"ok": not conflicts, "import_id": body["import_id"], "results": results}
    monkeypatch.setattr(h.participant.Host, "call", call)


def _replay_not_heard(monkeypatch, _h):
    """The Host before this change: a live replay of an imported revision is not answered silent first."""
    from ouroboros import place_arrivals
    real = place_arrivals.heard_replay

    def heard_replay(root, task_id, identity="", continuation_version=0):
        with ds.transaction(root, write=False) as conn:
            row = place_arrivals._record(conn, task_id)
        if row is not None and json.loads(row["dispatch_input"] or "{}").get("place_attention", {}).get("decision") != "heard":
            return None
        return real(root, task_id, identity, continuation_version)
    monkeypatch.setattr(place_arrivals, "heard_replay", heard_replay)


def _stored_order(monkeypatch, _h):
    """A reader that ignores imported send times: the room pages in stored order."""
    from ouroboros import place_arrivals
    monkeypatch.setattr(place_arrivals, "holds_history_import", lambda _root, _chat_id: False)


def _participant_authorship(monkeypatch, h):
    """A connector that files the account's own past messages as someone else's."""
    real = h.H.history_record
    monkeypatch.setattr(h.H, "history_record", lambda *args, **kw: {**real(*args, **kw), "authorship": "participant"})


MUTATIONS = {"turn_fallback": _turn_fallback, "replay_not_heard": _replay_not_heard, "stored_order": _stored_order,
             "participant_authorship": _participant_authorship}


@pytest.fixture(autouse=True)
def _mutation(monkeypatch):
    name = os.environ.get("OUROBOROS_JOINT_MUTATION", "")
    if name:
        MUTATIONS[name](monkeypatch, hub())
    yield


@pytest.fixture
def world(tmp_path):
    made = World(tmp_path)
    try:
        yield made
    finally:
        made.close()


def _store_or_lock(name):
    """The dialogue store and the per-conversation intake locks it is written under."""
    return name.startswith("state/dialog") or (name.startswith("state/presence_turn_gate/") and name.endswith(".lock"))


def _base_history(w):
    """Board games on Fridays over a year, and Mira's chat: what Telegram holds, oldest first."""
    T, now = w.hub.T, w.now
    photo = T.MessageMediaPhoto(photo=T.Photo(id=78, access_hash=1, file_reference=b"r", date=None, sizes=[], dc_id=1))
    say = w.say
    w.histories[GROUP].append(w.hub.fakes.service(11, GROUP, now - 40 * DAY, sender=TOM))  # Tom joined by a link
    messages = SimpleNamespace(
        year_ago=say(GROUP, 10, "Last year on this day we played Catan.", now - 365 * DAY, MIRA),
        photo=say(GROUP, 12, "", now - 20 * DAY, TOM, media=photo),
        reply=say(GROUP, 13, "Nice table!", now - 20 * DAY + 5 * MINUTE, MIRA,
                  reply_to=T.MessageReplyHeader(reply_to_msg_id=12)),
        edited=say(GROUP, 14, "Friday at eight?", now - 10 * DAY, MIRA, edit_date=now - 9 * DAY),
        phone=say(GROUP, 15, "I can bring two decks.", now - 8 * DAY, own=True),
        first=say(GROUP, 16, "Sent first within one second", now - 3 * DAY, MIRA),
        second=say(GROUP, 17, "Sent second within that second", now - 3 * DAY, TOM),
        lately=say(GROUP, 18, "Tomorrow then.", now - DAY, MIRA),
        # Minutes ago: what this companion sent and what live intake met while it ran.
        sent_here=say(GROUP, 19, "I will bring the snacks.", now - 9 * MINUTE, own=True),
        captured=say(GROUP, 20, "See you all on Friday.", now - 8 * MINUTE, TOM),
        older=say(GROUP, 21, "Who has the rules?", now - 7 * MINUTE, MIRA),
        differs=say(GROUP, 22, "Slides are in the pinned post.", now - 6 * MINUTE, TOM),
        asked=say(MIRA, 30, "Are we still on for Friday?", now - 12 * DAY, MIRA),
        answered=say(MIRA, 31, "Yes, see you there.", now - 12 * DAY + 2 * MINUTE, own=True))
    return messages


def test_selected_history_lands_once_with_its_authors_and_times_and_starts_nothing(world, monkeypatch):
    """The Hub importer over the real Host: each earlier message lands once as history, with its Telegram author
    and send time, never as the owner or Ouroboros; a live copy keeps its custody, other content conflicts and
    stays held; nothing wakes, nothing is unread, mailed or answered; the room then reads in time order; a
    Telegram redelivery of an imported revision is heard silent before attention while a real edit wakes."""
    from ouroboros.chat_chain import row_address
    from ouroboros.consciousness_wake import observe_wake
    from ouroboros.dialogue_provenance import dialogue_author
    from ouroboros.gateway import places_api
    from ouroboros.memory_inventory import place_activity, row_meta
    from ouroboros.owner_mailbox import _mailbox_path
    from ouroboros.place_arrivals import account_candidates
    w, h = world, world.hub
    iso = h.normalize.iso
    m = _base_history(w)
    # The account is even confirmed as the owner's own: its past messages still stay platform history.
    p.edit_place(w.data, w.place.place_id, "confirm_owner_account", author="owner", platform_actor_id=str(ACCOUNT),
                 account_id=str(ACCOUNT))
    # Before the import: this companion's own send and its receipt, a message live intake took, and two an
    # earlier companion install handed over, one of them under text that differs from Telegram's history.
    w.companion.report_delivered(GROUP, m.sent_here)
    w.companion.deliver_live(m.captured, GROUP, TOM)
    earlier = w.open_companion(w.data.parent / "earlier")
    earlier.deliver_live(m.older, GROUP, MIRA)
    earlier.deliver_live(h.fakes.message(22, GROUP, "Slides are in the pinned message.", m.differs.date, sender=TOM),
                         GROUP, TOM)
    live_ids = [f"telegram:{ACCOUNT}:{GROUP}:0:{mid}" for mid in (20, 21, 22)]
    assert w.recorder.events == live_ids  # consciousness is on and every message wakes here
    before, dispatches, calls = files(w.data), ds.pending_dispatches(w.data), len(w.companion.wire.calls)
    revisions = {row["chat_id"]: (row["revision"], row["read_revision"]) for row in ds.rooms(w.data)}
    companion_jobs = w.companion.journal.counts()["jobs"]

    plan = w.plan([GROUP, MIRA], w.now - 400 * DAY)
    w.act(plan, "start")
    assert w.drive(plan) == "done"

    # Nothing else happened: no turn, wake, unread mark, dispatch, outbox or reply ...
    assert w.recorder.events == live_ids
    changed = sorted(name for name, stamp in files(w.data).items() if before.get(name) != stamp)
    assert changed and all(_store_or_lock(name) for name in changed), changed
    assert ds.pending_dispatches(w.data) == dispatches
    assert {row["chat_id"]: (row["revision"], row["read_revision"]) for row in ds.rooms(w.data)
            if row["chat_id"] in revisions} == revisions and room(MIRA) not in revisions
    assert all(row["revision"] == 0 for row in ds.rooms(w.data) if row["chat_id"] == room(MIRA))
    assert w.companion.journal.counts()["jobs"] == companion_jobs == {}

    # What the Host holds: each Telegram message once, imported ones with their author, send time and marker.
    stored, private = w.by_message(GROUP), w.by_message(MIRA)
    assert sorted(stored, key=int) == [str(mid) for mid in (10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)]
    assert sorted(private, key=int) == ["30", "31"]
    imported = {mid: row for mid, row in {**stored, **private}.items() if "historical_import" in row}
    assert sorted(imported, key=int) == ["10", "12", "13", "14", "15", "16", "17", "18", "30", "31"]
    for mid, row in imported.items():
        message = next(msg for msgs in w.histories.values() for msg in msgs if str(msg.id) == mid)
        own = mid in {"15", "31"}
        marker = row["historical_import"]
        assert (marker["import_id"], marker["authorship"], marker["transport_skill"]) == (
            plan, "platform_account" if own else "participant", CONNECTOR), row
        assert row["ts"] == iso(message.date) < marker["received_at"], row  # the Telegram send time, never a receipt
        assert row["direction"] == "in" and row["type"] == "place_arrival" and row["source"] == "presence:telegram"
        assert row["sender_session_id"] == str(ACCOUNT if own else message.sender_id)
        assert row["sender_label"] == ("Ouroboros" if own else w.people[message.sender_id].first_name)
        assert "attachments" not in row and "attachment_manifest" not in row
    assert "sent from the connected account itself" in dialogue_author(imported["15"])
    assert imported["14"]["client_message_id"] == f"telegram:{ACCOUNT}:{GROUP}:0:14:edit:" + quote(
        h.normalize.content_revision(m.edited), safe="")
    assert imported["14"]["transport"]["message"]["edited_at"] == iso(m.edited.edit_date)
    assert imported["14"]["text"] == "Friday at eight?"  # the current revision; its original text is unknown
    assert imported["12"]["transport"]["message"]["media"] == {"kind": "photo", "media_id": "78", "status": "not_downloaded"}
    assert imported["13"]["transport"]["message"]["reply_to_message_id"] == 12
    assert stored["19"]["type"] == "presence_delivery" and stored["19"]["direction"] == "out"  # its receipt row only
    assert stored["21"].get("historical_import") is None and stored["22"]["text"] == "Slides are in the pinned message."

    # ... no mailbox, and the wake and memory read the rows as history, not arrivals.
    assert not any(_mailbox_path(w.data, row["task_id"]).exists() for row in imported.values())
    observation = observe_wake(w.data, boundary=None, since=0, now=time.time())
    kinds = [kind for kind, _offset, _line in observation.events]
    assert kinds.count("history_import") == 10 and kinds.count("correspondent_message") == 3, kinds
    assert not any(row["text"] and row["text"] in observation.full_text() for row in imported.values())
    entries = [(row_address(row), row_meta(row), index) for index, row in enumerate(w.rows(GROUP))]
    assert place_activity(entries)["arrivals"] == 3
    assert {person["platform_actor_id"] for person in account_candidates(w.data, w.place.place_id)} == {str(MIRA), str(TOM)}

    # What went over the wire: one request per conversation, newest first, only what is not already there; the
    # Hub checks both capabilities before each conversation and again right before its POST (Hub r3).
    assert [path for _method, path in w.companion.wire.calls[calls:]] == [
        "/identity", "/presence/place", "/identity", "/presence/history"] * 2
    assert {path for _method, path in w.companion.wire.calls[calls:]} == {"/identity", "/presence/place", "/presence/history"}
    assert w.sent() == [["22", "21", "18", "17", "16", "15", "14", "13", "12", "10"], ["31", "30"]]
    statuses = [status for status, _body in w.companion.wire.answers]
    assert statuses == [409, 200]
    view = w.view(plan)
    assert view["counts"] == {"recorded": 10, "duplicate": 1, "conflict": 1}, view
    assert [(held["chat"], held["state"], held["message_id"]) for held in view["held"]] == [(str(GROUP), "conflict", "22")]
    group, mira = view["chats"]
    assert group["skipped"] == {"service": 1, "delivered": 1, "live_captured": 1} and group["read"] == 13
    assert mira["read"] == 2 and not mira["skipped"]
    results = {row[0]: json.loads(row[1]) for row in w.companion.journal.db.execute(
        "SELECT source_event_id, result FROM import_records WHERE plan=?", (plan,))}
    assert results[live_ids[1]]["custody"] == "live"  # the earlier install's live copy stands

    # The rooms read and the history pages: time order, each row once, the account never the owner.
    monkeypatch.setattr(places_api, "request_drive_root", lambda _request: w.data)
    monkeypatch.setattr("ouroboros.place_projection.place_declarations", lambda _root: [])
    from starlette.applications import Starlette
    from starlette.testclient import TestClient
    with TestClient(Starlette(routes=places_api.places_routes())) as client:
        rooms = {entry["room_chat_id"]: entry for entry in client.get(f"/api/places/{w.place.place_id}/rooms").json()["rooms"]}
    assert set(rooms) == {room(GROUP), room(MIRA)}
    assert (rooms[room(GROUP)]["label"], rooms[room(GROUP)]["last_at"]) == ("Board games on Fridays", stored["22"]["ts"])
    pages = w.pages(GROUP, 3)
    delivered = [row for page in reversed(pages) for row in page["messages"] if row.get("history_id")]
    assert [row["history_id"] for row in delivered] == w.reading_order(GROUP) and len(delivered) == 12
    by_time = [row["text"] for row in delivered if row["text"] not in {m.first.message, m.second.message}]
    # Imports and live captures read at their Telegram send times. The receipt row of this companion's own
    # message 19 records no send time (``/presence/delivery`` carries none, and none is invented), so it reads
    # when its receipt was recorded: here after the live captures this synthetic world dated minutes earlier.
    assert by_time == [h.normalize.canonical_text(msg) for msg in (m.year_ago, m.photo, m.reply, m.edited, m.phone,
                                                                    m.lately, m.captured, m.older)] + [
        "Slides are in the pinned message.", h.normalize.canonical_text(m.sent_here)]
    spans = [page["coverage"]["spans"]["chat"] for page in pages]
    assert all(span["chain"].startswith("dialog-chrono-") and not span["gaps"] for span in spans)
    assert all(older["to"] == newer["from"] for newer, older in zip(spans, spans[1:])) and spans[-1]["from"] == 0
    shown = {row["text"]: row for row in delivered}
    for text, authorship in ((m.phone.message, "platform_account"), (m.lately.message, "participant")):
        facts = shown[text]["place_history"]
        assert shown[text]["role"] == "user" and facts["historical_import"]["authorship"] == authorship
        assert "owner_confirmed" not in facts and "attention" not in facts

    # Later live intake: Telegram redelivers an imported revision (heard silent before attention, although
    # every message wakes here), then Mira really edits it (a new revision, a new live event that wakes).
    count, before = len(w.rows(GROUP)), files(w.data)
    w.companion.deliver_live(m.lately, GROUP, MIRA)
    assert w.recorder.events == live_ids and len(w.rows(GROUP)) == count
    changed = sorted(name for name, stamp in files(w.data).items() if before.get(name) != stamp)
    assert all(_store_or_lock(name) for name in changed), changed  # no task, mailbox, outbox or notice either
    inbox = w.companion.journal.db.execute("SELECT state, response FROM inbox WHERE id=?",
                                           (f"telegram:{ACCOUNT}:{GROUP}:0:18",)).fetchone()
    assert inbox["state"] == "done" and json.loads(inbox["response"])["body"]["outcome"] == "silent"
    revised = h.fakes.message(18, GROUP, "Tomorrow at eight then.", m.lately.date, sender=MIRA, edit_date=datetime.now(timezone.utc))
    w.companion.deliver_live(revised, GROUP, MIRA, kind="edit")
    edit_id = f"telegram:{ACCOUNT}:{GROUP}:0:18:edit:" + quote(h.normalize.content_revision(revised), safe="")
    assert w.recorder.events == [*live_ids, edit_id]
    edit_row = next(row for row in w.rows(GROUP) if row.get("client_message_id") == edit_id)
    assert "historical_import" not in edit_row and edit_row["transport"]["message"]["sent_at"] == iso(m.lately.date)
    assert edit_row["ts"] > edit_row["transport"]["message"]["edited_at"] > imported["18"]["ts"]
    w.histories[GROUP][w.histories[GROUP].index(m.lately)] = revised  # Telegram's history now holds the edit

    # The owner imports the same dates again: every record answers what the Host already holds.
    again = w.plan([GROUP, MIRA], w.now - 400 * DAY)
    w.act(again, "start")
    assert w.drive(again) == "done"
    assert w.view(again)["counts"] == {"duplicate": 10, "conflict": 1}
    assert w.view(again)["chats"][0]["skipped"] == {"service": 1, "delivered": 1, "live_captured": 2}
    assert len(w.rows(GROUP)) == count + 1 and w.recorder.events == [*live_ids, edit_id]


def test_unknown_outcomes_and_restarts_resend_the_same_frozen_records(world, monkeypatch):
    """A lost answer, a Host Service and a companion each rebuilt in this interpreter as a restart leaves them, a
    write the Host could not confirm (503) and a Host that is down: every retry resends the same frozen records
    byte for byte, each message lands once, and no Telegram page is read twice."""
    w = world
    for n in range(250):
        w.say(GROUP, 1000 + n, f"backlog {n:03d}", w.now - (250 - n) * DAY, (MIRA, TOM)[n % 2])
    plan = w.plan([GROUP], w.now - 300 * DAY)
    w.act(plan, "start")
    w.companion.wire.losses.append("after")
    w.step(2)  # read and freeze the newest page, then send it: it lands, and its answer is lost
    assert w.view(plan)["counts"] == {"pending": 100} and len(w.rows(GROUP)) == 100
    w.restart_host()
    w.step()  # the same records again: each answers duplicate, custody history
    first, resent = w.companion.wire.history
    assert resent == first and [status for status, _ in w.companion.wire.answers] == [200, 200]
    assert {result["duplicate"] for result in w.companion.wire.answers[-1][1]["results"]} == {True}
    w.restart_companion()
    assert (w.view(plan)["state"], w.view(plan)["reason"]) == ("paused", "interrupted")
    w.act(plan, "resume")
    real, appended = ds.append, []

    def flaky(root, stream, row, **kwargs):
        if row.get("historical_import"):
            appended.append(row["client_message_id"])
            if len(appended) == 3:
                raise ds.StoreBusy("dialogue store busy", reason="contention")
        return real(root, stream, row, **kwargs)
    monkeypatch.setattr(ds, "append", flaky)
    w.step(2)  # the next page: two records confirmed, the third not, the rest not attempted
    monkeypatch.setattr(ds, "append", real)
    status, answer = w.companion.wire.answers[-1]
    assert status == 503 and answer["code"] == "presence_history_unwritable"
    assert [result.get("disposition") for result in answer["results"]][:4] == [None, None, "retry", "retry"]
    assert w.view(plan)["counts"] == {"duplicate": 100, "recorded": 2, "pending": 98}
    w.step()  # only the unconfirmed records, unchanged
    partial, rest = (json.loads(body)["records"] for body in w.companion.wire.history[-2:])
    assert rest == partial[2:] and w.companion.wire.answers[-1][0] == 200
    w.step()  # the last page is read and frozen
    w.host.stop()
    w.step()  # the Host is down: nothing lands, the records wait
    assert w.view(plan)["counts"]["pending"] == 50 and len(w.rows(GROUP)) == 200
    w.host = HostServer(w.data, w.recorder, port=int(w.url.rsplit(":", 1)[1]))
    assert w.drive(plan) == "done"
    assert w.view(plan)["counts"] == {"duplicate": 100, "recorded": 150}
    stored = w.by_message(GROUP)
    assert len(stored) == 250 and all(row["historical_import"]["import_id"] == plan for row in stored.values())
    attempts = {}
    for body in (body for wire in w.wires for body in wire.history):
        for record in json.loads(body)["records"]:
            attempts.setdefault(record["event"]["source_event_id"], set()).add(json.dumps(record, sort_keys=True))
    assert len(attempts) == 250 and all(len(variants) == 1 for variants in attempts.values())
    assert len(w.telegram.reads()) == 3 and w.recorder.events == []
    assert [row["history_id"] for page in reversed(w.pages(GROUP, 150)) for row in page["messages"]] == w.reading_order(GROUP)


def test_the_owner_connected_place_gates_every_read_and_write(tmp_path):
    """No place: the import pauses before reading Telegram. A Paused place records history and wakes nothing. A
    disconnect while frozen records wait refuses them unwritten; after reconnecting, Resume sends the same ones."""
    w = World(tmp_path, connect=False)
    try:
        for n in range(3):
            w.say(GROUP, 50 + n, f"before the place {n}", w.now - (5 - n) * DAY, MIRA)
        for n in range(2):
            w.say(MIRA, 60 + n, f"in Mira's chat {n}", w.now - (4 - n) * DAY, MIRA)
        plan = w.plan([GROUP], w.now - 30 * DAY)
        w.act(plan, "start")
        assert w.drive(plan) == "paused"
        view = w.view(plan)
        assert (view["reason"], view["detail"]) == ("place_not_connected", {"chat": str(GROUP)})
        assert w.telegram.calls == [] and not w.rows(GROUP) and w.sent() == []
        w.connect("paused")
        w.act(plan, "resume")
        assert w.drive(plan) == "done"
        assert [row["text"] for row in w.rows(GROUP)] == [f"before the place {n}" for n in (2, 1, 0)]
        assert w.recorder.events == []
        second = w.plan([MIRA], w.now - 30 * DAY)
        w.act(second, "start")
        w.step()  # its page is read and frozen
        p.edit_place(w.data, w.place.place_id, "disconnect", author="owner")
        w.step()  # its records are refused unwritten
        view = w.view(second)
        assert (view["state"], view["reason"], view["counts"]) == ("paused", "place_not_connected", {"pending": 2})
        assert w.companion.wire.answers[-1][0] == 409 and not w.rows(MIRA)
        w.connect("active")
        w.act(second, "resume")
        assert w.drive(second) == "done"
        assert w.sent()[-1] == w.sent()[-2] == ["61", "60"] and len(w.rows(MIRA)) == 2
        assert w.recorder.events == [] and len(w.telegram.reads()) == 2
    finally:
        w.close()


def test_pages_around_an_imported_row_join_the_newest_page_or_leave_a_real_gap(world):
    """The history endpoint also answers a page around one imported row (``around=task:<its turn>``), although
    no browser reader asks for one in a place room. Judged by the endpoint's own coverage keys (CLIENT_CONTRACT
    §5, ``[time, order, position]``): the newest page and its older pages meet from the beginning (null) to the
    present (null); a page around a row near the present reaches the newest page's lower cut, one far below
    stops beneath it (rows neither page delivered); each around page begins a chain of its own."""
    w = world
    for n in range(260):  # ten of them share one Telegram second, as a forwarded batch does
        when = w.now - (300 - min(n, 130)) * DAY if 130 <= n < 140 else w.now - (300 - n) * DAY
        w.say(GROUP, 3000 + n, f"thread {n:03d}", when, (MIRA, TOM)[n % 2])
    plan = w.plan([GROUP], w.now - 400 * DAY)
    w.act(plan, "start")
    assert w.drive(plan) == "done" and w.view(plan)["counts"] == {"recorded": 260}
    order = w.reading_order(GROUP)
    task = {entry.history_id: entry.row["task_id"] for entry in ds.iter_rows(w.data, "chat")}
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway.history import make_chat_history_endpoint
    with TestClient(Starlette(routes=[Route("/api/chat/history", make_chat_history_endpoint(w.data))])) as client:
        def read(quota, **params):
            page = client.get("/api/chat/history", params={"chat_id": room(GROUP), "n_human": quota, **params}).json()
            assert "error" not in page, page
            return page
        recent, chain = read(20), [read(100)]
        while chain[-1]["has_more"]:
            chain.append(read(100, cursor=chain[-1]["next_cursor"]))
        near, far = (read(20, around="task:" + task[order[-rank]]) for rank in (10, 200))
    assert near["anchor"]["state"] == far["anchor"]["state"] == "found"
    assert [row["history_id"] for page in reversed(chain) for row in page["messages"]] == order

    def keys(page):
        return page["coverage"]["spans"]["chat"]["keys"]

    def compared(key):  # a public reading key as the reader compares it: no order (null) before every order
        return (key[0], -1 if key[1] is None else key[1], key[2])
    spans = [keys(page) for page in chain]
    assert spans[0]["to"] is None and spans[-1]["from"] is None
    assert all(older["to"] == newer["from"] for newer, older in zip(spans, spans[1:]))
    cut = keys(recent)["from"]
    assert keys(near)["to"] is None or compared(keys(near)["to"]) >= compared(cut)  # null: it reached the present
    assert compared(keys(far)["to"]) < compared(cut)
    chains = {page["coverage"]["spans"]["chat"]["chain"] for page in (recent, near, far)}
    assert len(chains) == 3 and all("-at-" in page["coverage"]["spans"]["chat"]["chain"] for page in (near, far))


def _same_second(w):
    """Telegram dates a message to the second; a forwarded batch or an album shares one. Six such messages within
    one page, and 130 more that cross the importer's 100-record page; the room should read them in the order
    Telegram sent them (its message ids)."""
    batch = [w.say(GROUP, 100 + n, f"forwarded {n}", w.now - 2 * DAY, MIRA) for n in range(6)]
    album = [w.say(GROUP, 300 + n, f"album {n:03d}", w.now - DAY, TOM) for n in range(130)]
    plan = w.plan([GROUP], w.now - 30 * DAY)
    w.act(plan, "start")
    assert w.drive(plan) == "done" and w.view(plan)["counts"] == {"recorded": 136}
    read = [row["text"] for page in reversed(w.pages(GROUP, 150)) for row in page["messages"]]
    assert read == [message.message for message in batch + album], read[:8] + ["..."] + read[-8:]


def test_messages_sent_within_one_telegram_second_read_in_their_send_order(world):
    """QA 2f335d42 held this as a strict-xfail RED (kept in .review-evidence/joint-original): the real Hub now
    sends ``message.source_order`` and the core reads by it (``presence_source_order_version: 1``)."""
    _same_second(world)
