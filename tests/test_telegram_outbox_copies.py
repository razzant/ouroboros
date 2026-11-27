"""Kept copies of what was not confirmed (skills/telegram/lib/telegram_copies.py, telegram_outbox.py).

Content whose only copy is the outbox's saved event keeps it after a delivery not confirmed, through expiry,
reloads and the finished-record bound, never sent again, until the owner deletes exactly the copies the form showed;
telemetry never keeps one; content a verified home in the app holds lets its copy go once that home is read again.
The host's own producer (supervisor.message_bus) writes the history rows, chat media and task files on a disposable
data root and publishes the events; the shipped bridge, handlers and journal run them against the loopback Bot API
origin of tests/test_telegram_outbox.py. Nothing leaves the machine; no owner token or setting is read.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

import httpx
import pytest
from starlette.applications import Starlette
from starlette.routing import Route

from tests.test_telegram_outbox import _BOT, _Bridge, _closed_base, _not_delivered, _Origin, _run, _until

_REPO = Path(__file__).resolve().parents[1]
_WAITING = {"pending", "retry_wait", "sending"}
# Line breaks a stray file's own name can hold: a POSIX file system stores a line feed, Windows refuses such a name
# (Errno 22), so there the name breaks only at U+2028, where ``str.splitlines`` ends a line just the same.
_NAME_BREAKS = ("\n", "\u2028") if os.name != "nt" else ("\u2028",)


def _bridge(data: Path, base: str, **settings) -> _Bridge:
    """One load of the skill with its state where an installation keeps it: ``<data>/state/skills/telegram``."""
    (data / "state" / "skills").mkdir(parents=True, exist_ok=True)
    bridge = _Bridge(data / "state" / "skills" / "telegram", base, **settings)
    bridge.api.get_runtime_info = lambda: {"data_dir": str(data)}
    return bridge


class _Producer:
    """``supervisor.message_bus`` on the data root: it stores the row, chat media or task file, then publishes."""

    def __init__(self, monkeypatch, data: Path):
        from supervisor import message_bus

        self.module, self.data, self.events = message_bus, data, []
        self.bus = message_bus.LocalChatBridge({})
        self.bus._broadcast_fn = lambda _frame: None
        monkeypatch.setattr(message_bus, "DATA_DIR", data)
        monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 7})
        monkeypatch.setattr(message_bus, "get_bridge", lambda: self.bus)
        monkeypatch.setattr(message_bus, "publish_event", lambda topic, event: self.events.append((topic, event)))

    def _one(self, topic: str) -> tuple:
        [found] = [item for item in self.events if item[0] == topic]
        self.events.clear()
        return found

    def reply(self, text: str, **kwargs) -> tuple:
        self.module.send_with_budget(1, text, task_id="work-1", **kwargs)
        return self._one("chat.outbound")

    def photo(self, data: bytes, mime: str = "image/png") -> tuple:
        assert self.bus.send_photo(1, data, caption="chart", mime=mime, task_id="work-1")[0]
        return self._one("chat.photo")

    def video(self, data: bytes) -> tuple:
        assert self.bus.send_video(1, data, caption="clip", mime="video/mp4", task_id="work-1")[0]
        return self._one("chat.video")

    def document(self, data: bytes, *, captured: bool, filename: str = "report.pdf", caption: str = "") -> tuple:
        ref = None
        if captured:  # a task file captured by name, size and hash, as send_file captures it
            from ouroboros.artifacts import task_artifact_dir_path

            folder = task_artifact_dir_path(self.data, "work-1", create=True)
            (folder / filename).write_bytes(data)
            ref = {"kind": "task_artifact", "root": "artifact_store", "task_id": "work-1", "path": filename,
                   "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        assert self.bus.send_document(1, data, filename=filename, caption=caption, mime="application/pdf",
                                      task_id="work-1", file_ref=ref)[0]
        return self._one("chat.document")


def _blob(size: int, seed: int) -> bytes:
    return random.Random(seed).randbytes(size)


def _saved(bridge: _Bridge, record: dict) -> dict:
    return json.loads((bridge.state / "outbox" / f"{record['id']}.event.json").read_bytes())


def _handle(name: str) -> str:
    """The handle the journal gives a stray file (``telegram_outbox._stray``)."""
    return "stray-" + hashlib.sha256(name.encode("utf-8", "surrogateescape")).hexdigest()[:16]


def _on_disk(bridge: _Bridge) -> dict:
    """By topic, the state each operation's record held when the journal last finished writing it to disk. Observed
    at the journal's own atomic write, never by opening the file its sender replaces meanwhile (on Windows such an
    open is refused while the replace lands, and makes that replace fail); a write that fails records nothing."""
    module, written = bridge.module("telegram_outbox"), {}
    real = module._write_private

    def write(path, content):
        real(path, content)  # replaced and flushed, or it raises here as it would anyway
        if not path.name.endswith(".event.json"):
            record = json.loads(content)
            written[record.get("topic")] = str(record.get("state"))

    module._write_private = write
    return written


def _records(bridge: _Bridge) -> list:
    return [record for record in bridge.records() if record["state"] not in _WAITING]


def test_only_copies_outlive_refusal_unknown_expiry_reload_and_the_finished_bound(tmp_path, monkeypatch):
    """Content no verified home holds keeps its saved event exactly as published, never sent again."""
    data = tmp_path / "data"

    async def scenario():
        bridge = _bridge(data, _closed_base())
        produce = _Producer(monkeypatch, data)
        inline = produce.document(_blob(40_000, 1), captured=False, filename="notes.bin")  # no captured file
        bitmap = produce.photo(b"BM" + _blob(30_000, 2), mime="image/bmp")  # a type chat media never stores
        monkeypatch.setattr(produce.module, "store_chat_media_bytes",
                            lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")))
        unsaved = produce.photo(b"\x89PNG" + _blob(30_000, 3))  # its chat media could not be saved
        logged = produce.reply("The full answer.", log_text="A shorter line for history.")  # history holds other words
        saved_work = ("chat.outbound", {"text": "Saved draft of the plan.", "card_row": "timeline", "role": "system",
                                        "terminal_receipt": "retained_work", "chat_id": 1, "transport": {}})
        salvage = ("chat.outbound", {"text": "Work stopped.", "role": "system", "system_type": "project_completion_summary",
                                     "completion_notice": {"salvage": "Intermediate result: 3 of 5 checks done."},
                                     "chat_id": 1, "transport": {}})
        question = ("chat.quiz", {"task_id": "work-9", "quiz_id": "q9", "question": "Keep the old cache?",
                                  "options": [{"label": "Yes"}, {"label": "No"}], "chat_id": 1, "transport": {}})
        events = [inline, bitmap, unsaved, logged, saved_work, salvage, question]
        for topic, event in events:
            await bridge.publish(topic, event)
        assert [record["copy"] for record in bridge.records()] == ["only"] * 7
        # Expiry: still waiting after a day, none of them is sent then either.
        bridge.now += 24 * 3600 + 1
        async with _Origin() as late:
            bridge.serve(late.base)
            await bridge.drain()
        assert late.requests == []
        assert [(r["state"], r["reason"], r["held"]) for r in bridge.records()] == [("failed", "expired", True)] * 7
        for (topic, event), record in zip(events, bridge.records()):
            assert _saved(bridge, record) == event  # bytes and words exactly as published
        await bridge.stop()
        # A refusal and an unknown answer keep theirs too; a reload and a flood of finished telemetry keep all.
        async with _Origin(["403", "close"]) as origin:
            again = _bridge(data, origin.base, TELEGRAM_MIRROR_PROGRESS="on")
            await again.publish("chat.photo", bitmap[1])
            await again.publish("chat.document", inline[1])
            telemetry = {"text": "step", "is_progress": True, "transport": {}}
            async with _Origin(["403"] * 120) as refusing:
                again.serve(refusing.base)
                for _ in range(105):
                    await again.publish("chat.outbound", telemetry)
            assert len(refusing.requests) == 105
            await again.stop()
            before = len(origin.requests)
            reloaded = _bridge(data, origin.base, TELEGRAM_MIRROR_PROGRESS="on")
            await reloaded.drain()
            assert len(origin.requests) == before  # nothing kept is ever sent again
        held = [record for record in reloaded.records() if record.get("held")]
        assert [(r["state"], r["topic"]) for r in held[-2:]] == [("refused", "photo"), ("unknown", "document")]
        assert len(held) == 9 and all((reloaded.state / "outbox" / f"{r['id']}.event.json").exists() for r in held)
        finished = [record for record in reloaded.records() if not record.get("held")]
        assert len(finished) == 100 and {r["copy"] for r in finished} == {"replaceable"}  # the bound evicts telemetry
        assert not any((reloaded.state / "outbox" / f"{r['id']}.event.json").exists() for r in finished)
        await reloaded.stop()
    _run(scenario)


def test_telemetry_is_never_kept_but_typed_saved_work_on_a_card_row_is(tmp_path):
    """Negative control: the class follows typed fields, never the words."""
    async def scenario():
        async with _Origin(["403"] * 6) as origin:
            bridge = _bridge(tmp_path / "data", origin.base, TELEGRAM_MIRROR_PROGRESS="on", TELEGRAM_SUBAGENT_CARDS="on")
            same = {"text": "Saved work: the plan.", "transport": {}, "chat_id": 1}
            await bridge.publish("chat.outbound", {**same, "is_progress": True})
            await bridge.publish("chat.outbound", {**same, "card_row": "timeline", "is_progress": True})
            await bridge.publish("chat.outbound", {**same, "card_row": "timeline", "is_progress": True,
                                                   "terminal_receipt": "retained_work"})
            await bridge.publish("chat.outbound", {**same, "role": "system", "system_type": "project_completion_summary",
                                                   "is_progress": True, "completion_notice": {"salvage": "Draft."}})
        records = bridge.records()
        assert [(r["subject"]["kind"], r["copy"], r.get("held")) for r in records] == [
            ("progress", "replaceable", None), ("progress", "replaceable", None),
            ("saved work", "only", True), ("salvaged result", "only", True)]
        assert sorted(name for name in bridge.files() if name.endswith(".event.json")) == sorted(
            f"{r['id']}.event.json" for r in records[2:])
    _run(scenario)


def test_content_with_a_verified_home_goes_once_the_home_is_read_again(tmp_path, monkeypatch):
    """Text, a captured document, a photo and a video the app holds keep no copy; one whose home vanished before
    its end keeps it; an address or quiz id whose content differs proves no home."""
    data = tmp_path / "data"

    async def scenario():
        produce = _Producer(monkeypatch, data)
        text = produce.reply("The answer, as history holds it.")
        document = produce.document(_blob(50_000, 4), captured=True)
        photo = produce.photo(b"\x89PNG" + _blob(60_000, 5))
        video = produce.video(_blob(200_000, 6))
        vanishing = produce.photo(b"\x89PNG" + _blob(60_000, 7))
        other = produce.reply("Another reply.")
        borrowed = ("chat.outbound", {**text[1], "text": "Words that row never held."})  # a real address, other words
        from ouroboros.owner_quiz import record_asked

        record_asked(data, "work-1", quiz_id="q1", question="Which db?", options=["sqlite", "postgres"], chat_id=1)
        asked = ("chat.quiz", {"task_id": "work-1", "quiz_id": "q1", "question": "Which db?", "chat_id": 1,
                               "options": [{"label": "sqlite"}, {"label": "postgres"}], "transport": {}})
        reworded = ("chat.quiz", {**asked[1], "question": "Which database, again?"})
        events = [text, document, photo, video, vanishing, other, borrowed, asked, reworded]
        bridge = _bridge(data, _closed_base())
        for topic, event in events:
            await bridge.publish(topic, event)
        records = bridge.records()
        assert [r["copy"] for r in records] == ["home"] * 6 + ["only", "home", "only"]
        assert "file_base64" not in _saved(bridge, records[1])  # its handler reads the captured file
        assert _saved(bridge, records[2])["image_base64"] == photo[1]["image_base64"]  # media keeps its bytes
        media = records[4]["home"]["media"]
        from ouroboros.artifacts import resolve_chat_media_path

        resolve_chat_media_path(data, media["task_id"], media["name"]).unlink()  # gone before its end
        async with _Origin(["403"] * 9) as origin:
            bridge.serve(origin.base)
            await bridge.drain()
        assert [(r["state"], bool(r.get("held"))) for r in bridge.records()] == [
            ("refused", False)] * 4 + [("refused", True), ("refused", False), ("refused", True),
                                       ("refused", False), ("refused", True)]
        kept = {f"{r['id']}.event.json" for r in bridge.records() if r.get("held")}
        assert {name for name in bridge.files() if name.endswith(".event.json")} == kept
        assert bridge.plugin._connection_fields(bridge.api)["delivery_status"] == (
            "9 recent delivery issues (see Delivery issues)")
        words = _not_delivered(bridge)
        assert words.count("Its content is in the app.") == 6 and words.count("Only copy kept on this computer") == 3
        assert "Words that row never held" not in words and "The answer" not in words  # never the words
        assert "File report.pdf, 49 KB · Main" in words and "Photo, 59 KB · Main" in words
        await bridge.stop()
    _run(scenario)


def test_canonical_media_keeps_its_bytes_through_a_busy_queue_connect_error_429_and_restart(tmp_path, monkeypatch):
    data = tmp_path / "data"
    picture, clip = b"\x89PNG" + _blob(250_000, 8), _blob(1_200_000, 9)

    async def scenario():
        produce = _Producer(monkeypatch, data)
        photo, video = produce.photo(picture), produce.video(clip)
        async with _Origin(["hang"]) as busy:
            bridge = _bridge(data, busy.base)
            first = asyncio.ensure_future(bridge.publish("chat.outbound", {"text": "first, hangs", "transport": {}}))
            await _until(busy.hanging.is_set)
            await bridge.publish(*photo)  # the queue is busy: saved behind the hung request
            await bridge.publish(*video)
            saved = [_saved(bridge, record) for record in bridge.records()[1:]]
            assert [s.get("image_base64") or s.get("video_base64") for s in saved] == [
                photo[1]["image_base64"], video[1]["video_base64"]]
            first.cancel()  # the process stops mid-request
            await asyncio.gather(first, return_exceptions=True)
        cold = _bridge(data, _closed_base())  # restart: Telegram cannot be reached
        on_disk = _on_disk(cold)
        cold.start()
        # Stop once the photo's wait is on disk: a stop during that write leaves its saved "sending", which the
        # next load must read as unknown (it may have arrived), by design.
        await _until(lambda: on_disk.get("photo") == "retry_wait")
        await cold.stop()
        states = {r["topic"]: (r["state"], [part["state"] for part in r["parts"]]) for r in cold.records()}
        assert states == {"outbound": ("unknown", ["unknown"]), "photo": ("retry_wait", ["retry_wait"]),
                          "video": ("pending", [])}  # the interrupted part is unknown; the media wait
        async with _Origin(["429:2"]) as origin:
            warm = _bridge(data, origin.base)  # restart again: flood control, then delivery
            await warm.drain()
        assert origin.methods() == ["sendPhoto", "sendPhoto", "sendVideo"]
        photo_bodies = [body for name, _p, _f, body in origin.requests if name == "sendPhoto"]
        assert all(picture in body for body in photo_bodies)
        assert clip in next(body for name, _p, _f, body in origin.requests if name == "sendVideo")
        assert [r["topic"] for r in warm.records()] == ["outbound"]  # media delivered: nothing of them is kept
        await warm.stop()
    _run(scenario)


def test_content_that_waits_while_its_filter_changes_ends_superseded_and_keeps_its_only_copy(tmp_path, monkeypatch):
    """The owner may stop mirroring what waits, never lose it: content saved to be sent whose handler sends nothing
    once the mirror mode changed (the bridge restarted meanwhile) ends superseded, nothing is sent, and an only copy
    stays exactly as published; content the app holds lets its copy go; telemetry leaves no trace."""
    data = tmp_path / "data"

    async def scenario():
        produce = _Producer(monkeypatch, data)
        reply = ("chat.outbound", {"text": "An answer no history row holds.", "chat_id": 1, "transport": {}})
        bitmap = produce.photo(b"BM" + _blob(30_000, 80), mime="image/bmp")  # a type chat media never stores
        inline = produce.document(_blob(40_000, 81), captured=False, filename="notes.bin")
        picture = produce.photo(b"\x89PNG" + _blob(30_000, 82))  # the app's chat media holds it
        progress = ("chat.outbound", {"text": "Checking logs.", "is_progress": True, "chat_id": 1, "transport": {}})
        waiting = [reply, bitmap, inline, picture, progress]
        bridge = _bridge(data, _closed_base(), TELEGRAM_MIRROR_PROGRESS="on")
        await bridge.publish("chat.outbound", {"text": "First, Telegram unreachable.", "chat_id": 1, "transport": {}})
        for topic, event in waiting:
            await bridge.publish(topic, event)  # each would be sent as published: saved behind the first
        assert [(r["state"], r["copy"]) for r in bridge.records()] == [("retry_wait", "only")] + [
            ("pending", "only")] * 3 + [("pending", "home"), ("pending", "replaceable")]
        async with _Origin() as origin:  # mirroring only Telegram's own chats, without progress, after a restart
            again = _bridge(data, origin.base, TELEGRAM_MIRROR_MODE="telegram_only", TELEGRAM_MIRROR_PROGRESS="off")
            await again.drain()
        assert origin.requests == []
        records = again.records()
        assert [(r["topic"], r["state"], r.get("held")) for r in records] == [
            ("outbound", "superseded", True), ("outbound", "superseded", True), ("photo", "superseded", True),
            ("document", "superseded", True), ("photo", "superseded", None)]  # the progress note left no trace
        assert [_saved(again, record) for record in records[1:4]] == [event for _topic, event in waiting[:3]]
        assert not (again.state / "outbox" / f"{records[4]['id']}.event.json").exists()
        words = _not_delivered(again)
        assert words.count("Not sent: no longer mirrored. Only copy kept on this computer (Kept copies).") == 4
        assert words.count("Not sent: no longer mirrored. Its content is in the app.") == 1
        await again.stop()
    _run(scenario)


def test_what_a_filter_holds_back_as_published_is_never_saved_nor_waits_for_a_token(tmp_path, monkeypatch):
    """Negative control: a photo the mirror mode filters as it is published leaves nothing, whether the queue is free
    or waits (its handler's first filter drops it before anything is saved), and never waits for a bot token; one
    the bridge would send waits for the token."""
    data = tmp_path / "data"

    async def scenario():
        web = _Producer(monkeypatch, data).photo(b"BM" + _blob(30_000, 84), mime="image/bmp")
        phone = ("chat.photo", {**web[1], "transport": {"kind": "telegram", "conversation_id": 42}})
        bridge = _bridge(data, _closed_base(), TELEGRAM_MIRROR_MODE="telegram_only")
        await bridge.publish(*web)  # the queue is free
        assert bridge.records() == [] and bridge.files() == []
        await bridge.publish("chat.outbound", {"text": "From the phone.", "transport": phone[1]["transport"]})
        await bridge.publish(*web)  # the queue waits
        [head] = bridge.records()
        assert head["topic"] == "outbound" and bridge.files() == [f"{head['id']}.event.json", f"{head['id']}.json"]
        blind = _Bridge(tmp_path / "blind", _closed_base(), token="", TELEGRAM_MIRROR_MODE="telegram_only")
        blind.api.get_runtime_info = lambda: {"data_dir": str(data)}
        await blind.publish(*web)
        assert blind.records() == []  # filtered: no token is needed to send nothing
        await blind.publish(*phone)
        assert [(r["topic"], r["state"]) for r in blind.records()] == [("photo", "pending")]  # it waits for one
    _run(scenario)


def test_deciding_a_waiting_photo_ends_no_silent_progress_run(tmp_path, monkeypatch):
    """The decision of content that waits sends nothing, so it changes nothing: the progress note Silent mode would
    replace stays tracked while a photo behind a waiting answer is decided (its own run, later, ends that run)."""
    data = tmp_path / "data"

    async def scenario():
        photo = _Producer(monkeypatch, data).photo(b"\x89PNG" + _blob(9_000, 85))
        async with _Origin() as origin:
            bridge = _bridge(data, origin.base, TELEGRAM_SILENT_MODE="on", TELEGRAM_MIRROR_PROGRESS="on")
            await bridge.publish("chat.outbound", {"text": "Step one.", "is_progress": True, "chat_id": 1, "transport": {}})
        state = bridge.module("telegram_state")
        tracked = state._get_silent_msg(bridge.api, _BOT, 42)
        assert tracked == 501
        bridge.serve(_closed_base())
        await bridge.publish("chat.outbound", {"text": "An answer that waits.", "chat_id": 1, "transport": {}})
        await bridge.publish(*photo)  # decided behind it: it would send, and is kept waiting
        assert [r["topic"] for r in bridge.waiting()] == ["outbound", "photo"]
        assert state._get_silent_msg(bridge.api, _BOT, 42) == tracked
    _run(scenario)


def test_kept_copies_have_no_bound_of_their_own_and_never_hold_back_new_work(tmp_path, monkeypatch):
    """Only copies stay until deleted, without a bound of their own (one would drop an only copy or refuse new
    messages), and Kept copies says what they take. Kept bytes far above any waiting bound (one kept record reads
    300 MiB, as a long outage could leave) hold back no new message: the waiting bounds alone still refuse what does
    not fit, and new only copies and content the app holds are admitted and delivered while every kept copy stays."""
    data = tmp_path / "data"

    async def scenario():
        produce = _Producer(monkeypatch, data)
        bitmaps = [produce.photo(b"BM" + _blob(20_000, seed), mime="image/bmp") for seed in range(10, 23)]
        async with _Origin(["403"] * 6) as origin:
            bridge = _bridge(data, origin.base)
            for topic, event in bitmaps[:6]:
                await bridge.publish(topic, event)
            kept = list(bridge.records())
            size = kept[1]["payload_bytes"]
            assert [r.get("held") for r in kept] == [True] * 6
            kept[0]["payload_bytes"] = 300 * 1024 * 1024  # what this one record says it keeps
            await bridge.outbox._write(kept[0])
            bridge.outbox._refresh()
            module = bridge.module("telegram_outbox")
            monkeypatch.setattr(module, "MAX_ACTIVE_BYTES", 3 * size + size // 2)  # room for a head and three photos
            asked = len(origin.requests)
            # Concurrent admissions behind a waiting head: the account includes reservations before any write.
            bridge.serve(_closed_base())
            await bridge.publish("chat.outbound", produce.reply("A head that waits.")[1])
            await asyncio.gather(*(bridge.publish(topic, event) for topic, event in bitmaps[6:12]))
            admitted = [r for r in bridge.records() if r["state"] in _WAITING and r["topic"] == "photo"]
            refused = [r for r in bridge.records() if r.get("reason") == "full"]
            assert len(admitted) == 3 and len(refused) == 3 and bridge.outbox._capacity == {}
            assert len(origin.requests) == asked  # refused before any request
            bridge.serve(origin.base)
            await bridge.drain()
            # Then a new only copy and content the app holds, each admitted and delivered at once.
            canonical = [produce.reply("A canonical reply."), produce.photo(b"\x89PNG" + _blob(9_000, 30)),
                         produce.document(_blob(9_000, 31), captured=True), produce.video(_blob(6_000, 32))]
            for topic, event in [bitmaps[12], *canonical]:
                await bridge.publish(topic, event)
            assert origin.methods()[asked:] == ["sendMessage", "sendPhoto", "sendPhoto", "sendPhoto", "sendPhoto",
                                                "sendMessage", "sendPhoto", "sendDocument", "sendVideo"]
            assert base64.b64decode(canonical[3][1]["video_base64"]) in \
                next(body for name, _path, _form, body in origin.requests if name == "sendVideo")
            summary = bridge.outbox.summary()
            assert summary["kept_total"] == 6 and summary["kept_bytes"] == sum(r["payload_bytes"] for r in kept)
            assert all((bridge.state / "outbox" / f"{r['id']}.event.json").exists() and r.get("held") for r in kept)
            connection = bridge.module("telegram_connection")
            assert connection._kept_words(summary, time.time()).splitlines()[0] == (
                f"6 kept copies take {connection._size(summary['kept_bytes'])} on this computer.")
            assert connection._size(summary["kept_bytes"]).startswith("300.")
            await bridge.stop()
            again = _bridge(data, origin.base)  # kept and refused alike are durable
            again.outbox._load()
            assert [r["reason"] for r in again.records() if r.get("reason") == "full"] == ["full"] * 3
            assert again.outbox.summary()["kept_total"] == 6
            assert "Not sent: the outbox was full. No copy was kept." in _not_delivered(again)
    _run(scenario)


def test_an_orphan_and_an_unreadable_record_stay_shown_and_unsent_and_a_failed_end_write_keeps_the_bytes(tmp_path):
    data = tmp_path / "data"

    async def scenario():
        bridge = _bridge(data, _closed_base())
        outbox = bridge.state / "outbox"
        outbox.mkdir(parents=True, exist_ok=True)
        orphan = {"image_base64": base64.b64encode(b"orphan bytes").decode(), "mime": "image/png", "transport": {}}
        (outbox / "000000000900-deadbeef.event.json").write_text(json.dumps(orphan))
        (outbox / "000000000901-cafebabe.json").write_text("{not json")
        async with _Origin() as origin:
            bridge.serve(origin.base)
            await bridge.drain()
            assert origin.requests == []  # a missing record proves nothing: never sent
            kept = bridge.module("telegram_connection")._kept_words(bridge.outbox.summary(), time.time())
            lines = kept.splitlines()  # what they take; then each stray: what it is with its exact name, then its handle
            assert re.fullmatch(r"2 kept copies take \d+ KB on this computer\.", lines[0]) and lines[1] == ""
            assert lines[2].endswith(" · file 000000000901-cafebabe.json") and lines[3] == _handle("000000000901-cafebabe.json")
            assert lines[5].endswith(" · file 000000000900-deadbeef.event.json")
            assert lines[6] == _handle("000000000900-deadbeef.event.json")
            assert "Saved message without its delivery record; it may have been sent" in kept
            # A held copy whose end record cannot be written keeps its bytes; the next load ends it again.
            module = bridge.module("telegram_outbox")
            real = module._write_private
            def failing(path, content):
                if b'"held": true' in content:
                    raise PermissionError("read-only state")
                real(path, content)
            module._write_private = failing
            async with _Origin(["403"]) as refusing:
                bridge.serve(refusing.base)
                await bridge.publish("chat.outbound", {"text": "Only here.", "transport": {}})
            [record] = [r for r in bridge.records() if r["topic"] == "outbound"]
            assert (outbox / f"{record['id']}.event.json").exists()
            module._write_private = real
            await bridge.stop()
            after = _bridge(data, origin.base)
            await after.drain()
        assert origin.requests == []
        assert sorted(path.name for path in outbox.iterdir() if "-dead" in path.name or "-cafe" in path.name) == [
            "000000000900-deadbeef.event.json", "000000000901-cafebabe.json"]
        [ended] = [r for r in after.records() if r["topic"] == "outbound"]
        assert ended.get("held") and (outbox / f"{ended['id']}.event.json").exists()
        await after.stop()
    _run(scenario)


async def _copies_route(bridge: _Bridge):
    """The kept-copies form's own route, as ``register_connection`` registers it, behind an ASGI client."""
    routes = {}
    bridge.api.register_route = lambda path, handler, methods: routes.__setitem__(path, (handler, methods))
    bridge.module("telegram_connection").register_connection(bridge.api)
    handler, methods = routes["connection/copies"]
    app = Starlette(routes=[Route("/copies", endpoint=handler, methods=list(methods))])
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def test_deleting_the_shown_copies_keeps_one_kept_after_the_form_was_drawn(tmp_path):
    async def scenario():
        async with _Origin(["403"] * 4) as origin:
            bridge = _bridge(tmp_path / "data", origin.base)
            for text in ("first", "second"):
                await bridge.publish("chat.outbound", {"text": text, "transport": {}, "chat_id": 1})
            outbox = bridge.state / "outbox"
            (outbox / "000000000950-0bad0bad.event.json").write_text("{}")
            again = _bridge(tmp_path / "data", origin.base)
            await again.drain()
            client = await _copies_route(again)
            shown = (await client.get("/copies")).json()
            ids = shown["delete_ids"].splitlines()
            assert shown["confirm_delete"] is False and len(ids) == 3
            await again.publish("chat.outbound", {"text": "third, kept later", "transport": {}, "chat_id": 1})
            newer = again.records()[-1]
            assert newer.get("held")
            refused = await client.post("/copies", json={**shown, "confirm_delete": False})
            assert refused.status_code == 400 and len(again.outbox.summary()["kept"]) == 4  # nothing without the tick
            done = (await client.post("/copies", json={**shown, "confirm_delete": True})).json()
            assert done["message"] == ("Deleted 3 kept copies. 1 other kept copy stays; open this page again to "
                                       "delete it.")
            assert newer["id"] in done["kept_copies"].splitlines() and done["delete_ids"] == ""  # disarmed
            # The renderer keeps the tick; a second press sends the disarmed field and deletes nothing.
            again_pressed = (await client.post("/copies", json={**done, "confirm_delete": True})).json()
            assert again_pressed["message"].startswith("Nothing to delete.") and newer.get("held")
            assert (await client.get("/copies")).json()["delete_ids"] == newer["id"]  # opened again: armed anew
            # Exactly the shown copies' bytes and the stray are gone; their outcomes stay, naming the deletion.
            assert sorted(path.name for path in outbox.glob("*.event.json")) == [f"{newer['id']}.event.json"]
            gone = [r for r in again.records() if r["id"] in ids]
            assert len(gone) == 2 and all(not r["held"] and r["deleted_at"] for r in gone)
            assert _not_delivered(again).count("Its kept copy was deleted.") == 2
            await client.aclose()
            await again.stop()
            assert len(origin.requests) == 3  # deleting sends nothing
    _run(scenario)


def test_names_never_create_or_hide_what_a_press_deletes(tmp_path, monkeypatch):
    """A file name with line breaks spelling another copy's id and a stray handle, a stray named in Cyrillic with
    spaces and one whose name holds a line break before that id (``_NAME_BREAKS``): the press deletes exactly the
    handles its own field listed. The oldest kept copy (out of the listed window, named inside the file names) and one
    kept after the form was drawn stay; every name shows on one line."""
    async def scenario():
        async with _Origin(["403"] * 8) as origin:
            bridge = _bridge(tmp_path / "data", origin.base)
            monkeypatch.setattr(bridge.module("telegram_outbox"), "SHOWN", 3)
            await bridge.publish("chat.outbound", {"text": "oldest", "transport": {}, "chat_id": 1})
            oldest = bridge.records()[0]["id"]
            for text in ("second", "third"):
                await bridge.publish("chat.outbound", {"text": text, "transport": {}, "chat_id": 1})
            tricky = f"report\n{oldest}\nstray-0123456789abcdef\n\u202enot.json.pdf"
            await bridge.publish("chat.document", {"file_base64": base64.b64encode(b"doc").decode(), "chat_id": 1,
                                                   "filename": tricky, "transport": {}})
            outbox = bridge.state / "outbox"
            strays = sorted(["черновик отчёта 1.event.json", *(f"line{br}{oldest}.event.json" for br in _NAME_BREAKS)])
            for name in strays:  # listed by exact name
                (outbox / name).write_text(json.dumps({"text": "stray", "transport": {}}), encoding="utf-8")
            again = _bridge(tmp_path / "data", origin.base)
            await again.drain()
            monkeypatch.setattr(again.module("telegram_outbox"), "SHOWN", 3)
            again.outbox._refresh()
            client = await _copies_route(again)
            shown = (await client.get("/copies")).json()
            listed = shown["delete_ids"].splitlines()
            held = [r["id"] for r in again.records() if r.get("held")]
            assert listed == [held[3], held[2], held[1], *map(_handle, strays)] and oldest == held[0]
            display = shown["kept_copies"]
            assert "report\\u000a" + oldest + "\\u000astray-0123456789abcdef\\u000a\\u202enot.json.pdf" in display
            assert oldest not in display.splitlines() and "file черновик отчёта 1.event.json" in display
            assert all(f"file line\\u{ord(br):04x}{oldest}.event.json" in display for br in _NAME_BREAKS)
            took = re.compile(r"\d+ kept copies take [\d.]+ [KM]B on this computer\.")  # what they take, first
            assert took.fullmatch(display.splitlines()[0])
            assert all(line == "" or "·" in line or line in listed or line.endswith("deleted.") or took.fullmatch(line)
                       for line in display.splitlines())  # no line a name made
            await again.publish("chat.outbound", {"text": "kept after the form", "transport": {}, "chat_id": 1})
            newer = again.records()[-1]["id"]
            # The display is never read back: even text there spelling the oldest id on its own line deletes nothing.
            posted = {**shown, "kept_copies": f"{shown['kept_copies']}\n\n{oldest}", "confirm_delete": True}
            done = (await client.post("/copies", json=posted)).json()
            assert done["message"].startswith(f"Deleted {3 + len(strays)} kept copies. 2 other kept copies stay")
            still = {r["id"] for r in again.records() if r.get("held")}
            assert still == {oldest, newer}
            assert sorted(path.name for path in outbox.glob("*.event.json")) == sorted(
                [f"{oldest}.event.json", f"{newer}.event.json"])
            forged = await client.post("/copies", json={"delete_ids": f"{oldest}\nnot-an-id", "confirm_delete": True})
            assert forged.status_code == 400 and oldest in {r["id"] for r in again.records() if r.get("held")}
            await client.aclose()
            await again.stop()
    _run(scenario)


def test_orphan_and_unreadable_sizes_are_shown_and_never_hold_back_new_work(tmp_path, monkeypatch):
    """An orphan event file and an unreadable record found at load are listed, and count toward what kept copies
    take by their real sizes, once each; a stray whose size cannot be read is said so, never counted as zero.
    Neither holds back a new only copy nor content the app holds; the orphan stays."""
    data = tmp_path / "data"

    async def scenario():
        produce = _Producer(monkeypatch, data)
        bitmap = produce.photo(b"BM" + _blob(20_000, 60), mime="image/bmp")
        canonical = [produce.reply("A reply the app holds."), produce.photo(b"\x89PNG" + _blob(5_000, 61)),
                     produce.video(_blob(5_000, 62)), produce.document(_blob(5_000, 63), captured=True)]
        state = data / "state" / "skills" / "telegram"
        (state / "outbox").mkdir(parents=True)
        orphan = state / "outbox" / "000000000700-0dd0dd00.event.json"
        orphan.write_bytes(json.dumps({"image_base64": base64.b64encode(_blob(40_000, 64)).decode(),
                                       "mime": "image/bmp", "transport": {}}).encode())
        torn = state / "outbox" / "000000000700-0dd0dd00.json"
        torn.write_text("{torn", encoding="utf-8")
        async with _Origin() as origin:
            bridge = _bridge(data, origin.base)
            bridge.outbox._load()
            summary = bridge.outbox.summary()
            assert (summary["kept_total"], summary["kept_bytes"], summary["kept_unknown"]) == (
                2, orphan.stat().st_size + torn.stat().st_size, 0)  # each file once, by its real size
            await bridge.publish(*bitmap)
            for topic, event in canonical:
                await bridge.publish(topic, event)
            assert origin.methods() == ["sendPhoto", "sendMessage", "sendPhoto", "sendVideo", "sendDocument"]
            await bridge.stop()
            (state / "outbox" / "000000000701-0dd0dd01.event.json").symlink_to(state / "outbox" / "gone")
            unknown = _bridge(data, origin.base)  # a reload: the dangling link's size cannot be read
            unknown.outbox._load()
            summary = unknown.outbox.summary()
            assert (summary["kept_total"], summary["kept_unknown"]) == (3, 1)
            kept = unknown.module("telegram_connection")._kept_words(summary, time.time())
            assert kept.splitlines()[0].endswith(" on this computer; the size of 1 can't be read.")
            assert "size unknown · file 000000000701-0dd0dd01.event.json" in kept
            await unknown.publish(*bitmap)
            await unknown.publish(*canonical[0])
            assert origin.methods()[-2:] == ["sendPhoto", "sendMessage"] and unknown.outbox.summary()["not_sent"] == 0
            await unknown.stop()
        assert orphan.exists()  # accepted content is never dropped
    _run(scenario)


def test_a_home_holds_every_part_the_bridge_sends_or_the_copy_stays(tmp_path, monkeypatch):
    """Same words, bytes or question are not enough: a completion's salvaged output, a photo's or document's caption
    and a question's host facts or option details are sent too. The host's producers write the homes; where the home
    lacks or differs in such a part (a row an older body wrote, a question whose facts arrived with its send), the
    copy is kept."""
    from ouroboros.owner_quiz import record_asked

    data = tmp_path / "data"

    async def scenario():
        produce = _Producer(monkeypatch, data)
        notice = {"neutral": False, "detail": "Stopped by the owner.", "salvage": "Draft: 3 of 5 checks done."}
        complete = produce.reply("Task stopped.", role="system", system_type="project_completion_summary",
                                 progress_meta={"completion_notice": notice})
        legacy = produce.reply("Task stopped.", role="system", system_type="project_completion_summary")
        replayed = ("chat.outbound", {**legacy[1], "completion_notice": notice})  # a row written before notices
        picture = b"\x89PNG" + _blob(4_000, 70)
        photo = produce.photo(picture)
        recaptioned = ("chat.photo", {**photo[1], "caption": "chart, revised"})
        report = produce.document(_blob(4_000, 71), captured=True, filename="report.pdf")
        renamed = ("chat.document", {**report[1], "caption": "the final one"})
        meanwhile = "I keep sqlite meanwhile."
        record_asked(data, "work-1", quiz_id="q1", question="Which db?", options=["sqlite", "postgres"],
                     option_details=["simple", "scales"], assumption=meanwhile, chat_id=1)
        record_asked(data, "work-1", quiz_id="q2", question="Ship it?", options=["Yes", "No"], assumption=meanwhile,
                     chat_id=1)
        assert produce.bus.send_quiz(1, "q1", "Which db?", [{"label": "sqlite"}, {"label": "postgres"}],
                                     assumption=meanwhile, task_id="work-1")[0]
        asked = produce._one("chat.quiz")
        assert produce.bus.send_quiz(1, "q2", "Ship it?", [{"label": "Yes"}, {"label": "No"}], assumption=meanwhile,
                                     task_id="work-1", host_facts="Asked by task work-1 while it waits.")[0]
        with_facts = produce._one("chat.quiz")
        assert with_facts[1]["host_facts"] and asked[1]["options"][0]["detail"] == "simple"
        copies = _bridge(data, _closed_base()).module("telegram_copies")
        kinds = [copies.classify(data, topic.split(".")[1], event)[0]
                 for topic, event in (complete, replayed, photo, recaptioned, report, renamed, asked, with_facts)]
        assert kinds == ["home", "only", "home", "only", "home", "only", "home", "only"]
        # Through the journal: the copies a home does not hold whole survive a refused delivery.
        async with _Origin(["403"] * 8) as origin:
            bridge = _bridge(data, origin.base)
            for topic, event in (replayed, recaptioned, renamed, with_facts, complete, photo, report, asked):
                await bridge.publish(topic, event)
        assert [bool(r.get("held")) for r in bridge.records()] == [True] * 4 + [False] * 4
        assert _saved(bridge, bridge.records()[2])["caption"] == "the final one"
    _run(scenario)


_RECIPE = re.compile(r"<!-- telegram-kept-copy-recipe -->\n```python\n(.*?)```\n<!-- /telegram-kept-copy-recipe -->", re.S)


@pytest.mark.parametrize("answer", ["close", "403"], ids=["unknown", "refused"])
def test_a_kept_binary_copy_reaches_a_task_artifact_through_run_script_with_no_telegram_request(tmp_path, monkeypatch,
                                                                                                 answer):
    """The SKILL.md recipe as written, through the real run_script tool and the real artifact route, for a copy whose
    delivery is unknown (the answer was lost after the request) or refused; neither outcome changes."""
    from ouroboros import event_bus
    from tests._dialogue import chat_rows

    data, repo = tmp_path / "data", tmp_path / "repo"
    repo.mkdir()
    picture = b"BM" + _blob(3 * 1024 * 1024, 40)  # a 3 MiB bitmap: a type chat media never stores
    recipe = _RECIPE.search((_REPO / "skills" / "telegram" / "SKILL.md").read_text(encoding="utf-8")).group(1)
    published = []

    async def scenario():
        produce = _Producer(monkeypatch, data)
        topic, event = produce.photo(picture, mime="image/bmp")
        salvage = {"text": "Work stopped.", "role": "system", "system_type": "project_completion_summary",
                   "completion_notice": {"neutral": False, "detail": "Stopped by its budget.",
                                         "review_gate": "Review: 1 of 2 reviewers answered.",
                                         "salvage": "Intermediate result: 3 of 5 checks done."},
                   "chat_id": 1, "transport": {}}
        quote = {"text": "Please check the March numbers.", "attachments": [], "source": {"chat_id": 1, "label": "Main"}}
        relay = {"role": "assistant", "text": "Passed on to the project.", "system_type": "relay_entry",
                 "kind": "relay_entry", "relay": {"relay_id": "r", "quotes": [quote]}, "chat_id": 1, "transport": {}}
        document = produce.document(b"REAL FILE CONTENT\nline 2\n", captured=False,
                                    filename="notes.txt", caption="see attached")
        state = "unknown" if answer == "close" else "refused"
        sent = ["sendPhoto", "sendMessage", "sendMessage", "sendDocument"]
        async with _Origin([answer] * 4) as origin:
            bridge = _bridge(data, origin.base)
            await bridge.publish(topic, event)
            await bridge.publish("chat.outbound", salvage)
            await bridge.publish("chat.outbound", relay)
            await bridge.publish(*document)
            record, words, quoted, attached = bridge.records()
            assert all(r["state"] == state and r.get("held") for r in bridge.records())
            assert origin.methods() == sent
            bridge.start()  # the sender runs throughout: a kept copy is never its work
            journal = {path.name: path.read_bytes() for path in (bridge.state / "outbox").iterdir()}
            rows = len(chat_rows(data))
            monkeypatch.setattr(event_bus, "publish_event", lambda *args, **_k: published.append(args))
            name = f"telegram-copy-{record['id']}.bmp"
            started = time.monotonic()
            result, response = await asyncio.to_thread(_retrieve_kept_copy, data, repo, recipe, record["id"], name, monkeypatch)
            elapsed = time.monotonic() - started
            await asyncio.sleep(0.2)
            assert "ARTIFACT_OUTPUTS" in result, result
            assert response.status_code == 200 and hashlib.sha256(response.content).hexdigest() == \
                hashlib.sha256(picture).hexdigest()
            print(next(line for line in result.splitlines() if "ARTIFACT_OUTPUTS" in line))
            for mode, name, expected in ((None, "notes.txt", b"REAL FILE CONTENT\nline 2\n"),
                                         (None, "notes.bin", b"REAL FILE CONTENT\nline 2\n"),
                                         ("bytes", "explicit.txt", b"REAL FILE CONTENT\nline 2\n"),
                                         ("words", "caption.txt", b"see attached")):
                result, response = await asyncio.to_thread(
                    _retrieve_kept_copy, data, repo, recipe, attached["id"], name, monkeypatch, mode)
                print(f"TXT_RETRIEVAL {state} {mode or 'default'}: {result}")
                assert response.status_code == 200 and response.content == expected, result
                assert f'"words": {str(mode == "words").lower()}' in result
                assert f'"bytes": {len(expected)}' in result
                assert hashlib.sha256(expected).hexdigest() in result
            result, caption = await asyncio.to_thread(
                _retrieve_kept_copy, data, repo, recipe, record["id"], "photo-caption.txt", monkeypatch, "words")
            assert caption.status_code == 200 and caption.content == b"chart", result
            text_name = f"telegram-copy-{words['id']}.txt"  # a kept message's words, the same way
            for mode, name in (("words", text_name), (None, "message.txt")):
                result, text = await asyncio.to_thread(
                    _retrieve_kept_copy, data, repo, recipe, words["id"], name, monkeypatch, mode)
                assert text.status_code == 200 and text.content == (  # every word the bridge would have sent
                    "Work stopped.\nStopped by its budget.\nReview: 1 of 2 reviewers answered.\n"
                    "Intermediate result: 3 of 5 checks done.".encode()), result
                assert '"words": true' in result
            quote_name = f"telegram-copy-{quoted['id']}.txt"  # the owner's quoted words above the note
            result, text = await asyncio.to_thread(
                _retrieve_kept_copy, data, repo, recipe, quoted["id"], quote_name, monkeypatch, "words")
            assert text.status_code == 200 and text.content == (
                "Please check the March numbers.\nPassed on to the project.".encode()), result
            await asyncio.sleep(0.2)
            assert origin.methods() == sent and published == [] and produce.events == []
            assert len(chat_rows(data)) == rows  # no chat row, no event: the copy only became this task's file
            assert {path.name: path.read_bytes() for path in (bridge.state / "outbox").iterdir()} == journal
            assert [(r["state"], r["held"]) for r in bridge.records()] == [(state, True)] * 4
            await bridge.stop()
        print(f"RETRIEVAL {state} {len(picture)} bytes, recipe start to GET end {elapsed:.3f} s, sha256 "
              f"{hashlib.sha256(picture).hexdigest()}")

    for key, value in (("OUROBOROS_RUNTIME_MODE", "advanced"), ("OUROBOROS_SAFETY_MODE", "off"),
                       ("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")):
        monkeypatch.setenv(key, value)
    monkeypatch.setattr("ouroboros.safety._run_llm_check",
                        lambda *_a, **_k: pytest.fail("Safety Off must not call a model"))
    _run(scenario)


@pytest.mark.parametrize("event,mode,expected", [
    ({"file_base64": "", "caption": "empty file"}, None, b""),
    ({"file_base64": "", "caption": "empty file"}, "bytes", b""),
    ({"text": "message"}, "bytes", "This copy holds no file bytes"),
    ({"file_base64": "", "caption": "empty file"}, "typo", "Choose bytes or words"),
])
def test_kept_copy_recipe_empty_file_and_invalid_modes(tmp_path, monkeypatch, capsys, event, mode, expected):
    """Empty bytes remain a file; invalid choices fail before writing an output."""
    import sys

    drive = tmp_path / "task_drives" / "retrieve-1"
    drive.mkdir(parents=True)
    outbox = tmp_path / "state" / "skills" / "telegram" / "outbox"
    outbox.mkdir(parents=True)
    (outbox / "kept.event.json").write_text(json.dumps(event))
    monkeypatch.chdir(drive)
    monkeypatch.setattr(sys, "argv", ["recipe.py", "kept", "notes.txt"] + ([mode] if mode else []))
    recipe = _RECIPE.search((_REPO / "skills" / "telegram" / "SKILL.md").read_text(encoding="utf-8")).group(1)
    if isinstance(expected, str):
        with pytest.raises(SystemExit, match=expected):
            exec(compile(recipe, "SKILL.md recipe", "exec"), {})
        assert not (drive / "notes.txt").exists()
    else:
        exec(compile(recipe, "SKILL.md recipe", "exec"), {})
        assert (drive / "notes.txt").read_bytes() == expected
        assert json.loads(capsys.readouterr().out) == {
            "file": "notes.txt", "bytes": 0, "sha256": hashlib.sha256(expected).hexdigest(), "words": False}


def _retrieve_kept_copy(data, repo, recipe, copy_id, name, monkeypatch, mode=None):
    from starlette.testclient import TestClient
    from ouroboros.gateway.tasks import api_task_artifact
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry.set_context(ToolContext(repo_dir=repo, drive_root=data, task_id="retrieve-1"))
    result = registry.execute("run_script", {"script": recipe, "interpreter": "python3", "cwd": "task_drive",
                                             "args": [copy_id, name] + ([mode] if mode is not None else []),
                                             "outputs": [name]})
    app = Starlette(routes=[Route("/api/tasks/{task_id}/artifacts/{name}", endpoint=api_task_artifact,
                                  methods=["GET"])])
    app.state.drive_root = data
    return result, TestClient(app).get(f"/api/tasks/retrieve-1/artifacts/{name}")
