"""How a delivery's saved copy ends and how a kept copy is deleted (skills/telegram/lib/telegram_outbox.py).

Accepted content keeps its saved event until its end is decided, whatever runs meanwhile: a refusal whose finished
bound would take the record while the end reads the content's home again, or while a stop is being saved, and a
process that ended between a stop and its end. A kept copy's deletion removes its bytes before its record says so,
so a removal or a record write that fails, and a reload after it, neither hide nor invent bytes; presses that
overlap delete each copy once, the later one saying what the earlier deleted; a refusal or an end whose cleanup
fails keeps its file, shown and counted. A captured document whose task file goes keeps its message and reference,
said as such. The shipped bridge, handlers and journal run against the loopback Bot API origin of
tests/test_telegram_outbox.py; faults are injected at the journal's own disk seams and the copy classification.
No Telegram, owner token or network.
"""
from __future__ import annotations

import asyncio
import base64
import json
import threading
import time

import pytest

from tests.test_telegram_outbox import _LONG, _closed_base, _not_delivered, _Origin, _run, _until
from tests.test_telegram_outbox_copies import _RECIPE, _REPO, _blob, _bridge, _copies_route, _handle, _Producer, _retrieve_kept_copy

_ONLY = {"chat_id": 1, "transport": {}}  # no history row holds it: its saved event is its only copy


def _kept(bridge) -> str:
    return bridge.module("telegram_connection")._kept_words(bridge.outbox.summary(), time.time())


def _took(bridge, count: int, size: int) -> str:
    """The first line of Kept copies: what they take, in the form's own sizes."""
    copies = "1 kept copy takes" if count == 1 else f"{count} kept copies take"
    return f"{copies} {bridge.module('telegram_connection')._size(size)} on this computer."


def _change_before_retry(bridge, record) -> None:
    """The saved event no longer matches the part already sent: its retry stops (``changed``)."""
    bridge.outbox._payload_path(record).write_text(json.dumps({**_ONLY, "text": "other words " + _LONG}),
                                                   encoding="utf-8")


def test_the_bound_never_takes_content_while_its_end_reads_its_home_again(tmp_path, monkeypatch):
    """An operation not confirmed reads its content's home again in a thread before it keeps the copy; a refusal
    meanwhile runs the finished bound. That end is not decided yet, so the bound passes it: the only copy is kept, on
    disk and after a reload."""
    data = tmp_path / "data"

    async def scenario():
        async with _Origin(["close"]) as origin:
            bridge = _bridge(data, origin.base)
            module = bridge.module("telegram_outbox")
            monkeypatch.setattr(module, "MAX_FINISHED", 1)
            monkeypatch.setattr(module, "MAX_ACTIVE", 1)
            reading, release, reads = threading.Event(), threading.Event(), []
            classify = module.copies.classify

            def paused(data_dir, topic, event):
                reads.append(event.get("filename"))
                if event.get("filename") == "only.txt" and reads.count("only.txt") == 2:  # its end, after admission
                    reading.set()
                    assert release.wait(10)
                return classify(data_dir, topic, event)

            monkeypatch.setattr(module.copies, "classify", paused)
            only = {**_ONLY, "file_base64": base64.b64encode(b"content only here").decode(), "filename": "only.txt"}
            ending = asyncio.ensure_future(bridge.publish("chat.document", only))
            assert await asyncio.to_thread(reading.wait, 10)
            [record] = bridge.records()
            payload = bridge.outbox._payload_path(record)
            assert record["state"] == "unknown" and not record.get("held") and payload.exists()
            try:
                await bridge.publish("chat.outbound", {**_ONLY, "text": "Admitted behind it."})
                await bridge.publish("chat.outbound", {**_ONLY, "text": "Refused: the queue holds one."})
                assert bridge.records()[-1]["reason"] == "full"  # a refusal: its record ran the bound
                assert payload.exists() and record["id"] in bridge.outbox._ops
            finally:
                release.set()
            await ending
        assert record.get("held") and record.get("finished_at") and payload.exists()
        again = _bridge(data, _closed_base())
        again.outbox._load()
        assert record["id"] in {item["id"] for item in again.outbox.summary()["kept"]}
        assert json.loads(payload.read_bytes()) == only
    _run(scenario)


def test_a_stopped_operations_copy_outlives_the_bound_while_its_stop_is_saved(tmp_path, monkeypatch):
    """Beyond the end itself: a stop makes the record final before its end runs, and saving the stop awaits the
    disk. A refusal's bound meanwhile passes it (its end is not decided), so the end keeps its only copy."""
    data = tmp_path / "data"

    async def scenario():
        async with _Origin(close_after=1) as first, _Origin() as second:
            bridge = _bridge(data, first.base)
            await bridge.publish("chat.outbound", {**_ONLY, "text": _LONG})  # part 1 arrives; part 2 waits
            [record] = bridge.records()
            _change_before_retry(bridge, record)
            module = bridge.module("telegram_outbox")
            monkeypatch.setattr(module, "MAX_FINISHED", 1)
            monkeypatch.setattr(module, "MAX_ACTIVE", 1)
            stopping, release = asyncio.Event(), asyncio.Event()
            write = bridge.outbox._write

            async def paused(item):
                if item["id"] == record["id"] and item.get("reason") == "changed" and not item.get("finished_at"):
                    stopping.set()  # the stop's own save, on the disk's way
                    await release.wait()
                await write(item)

            monkeypatch.setattr(bridge.outbox, "_write", paused)
            bridge.serve(second.base)
            bridge.start()
            bridge.now += 60
            bridge.outbox.wake()
            await asyncio.wait_for(stopping.wait(), 10)
            payload = bridge.outbox._payload_path(record)
            try:
                assert record["state"] == "failed" and not record.get("finished_at")
                await bridge.publish("chat.outbound", {**_ONLY, "text": "Admitted behind it."})
                await bridge.publish("chat.outbound", {**_ONLY, "text": "Refused: the queue holds one."})
                assert bridge.records()[-1]["reason"] == "full"
                assert payload.exists() and record["id"] in bridge.outbox._ops
            finally:
                release.set()
            await _until(lambda: bool(record.get("held")))
            await _until(lambda: not bridge.waiting())
            await bridge.stop()
        assert payload.exists() and record["reason"] == "changed" and record.get("finished_at")
        assert len(first.sent()) == 1 and [form["text"] for form in second.sent()] == ["Admitted behind it."]
        assert json.loads(payload.read_bytes())["text"].startswith("other words")  # never sent, never dropped
    _run(scenario)


def test_an_end_the_process_never_decided_is_decided_at_the_next_start(tmp_path, monkeypatch):
    """A stop was saved, then the process ended before the end kept or removed the copy: the next start decides
    that end (here: the only copy is kept), so its bytes are neither hidden from Kept copies nor left for the bound."""
    data = tmp_path / "data"

    async def scenario():
        async with _Origin(close_after=1) as first, _Origin() as second:
            bridge = _bridge(data, first.base)
            await bridge.publish("chat.outbound", {**_ONLY, "text": _LONG})
            [record] = bridge.records()
            _change_before_retry(bridge, record)
            module = bridge.module("telegram_outbox")
            reading, release = threading.Event(), threading.Event()
            classify = module.copies.classify

            def paused(data_dir, topic, event):
                if str(event.get("text") or "").startswith("other words"):  # the end reads the stopped one's home
                    reading.set()
                    release.wait(10)
                return classify(data_dir, topic, event)

            monkeypatch.setattr(module.copies, "classify", paused)
            bridge.serve(second.base)
            bridge.start()
            bridge.now += 60
            bridge.outbox.wake()
            assert await asyncio.to_thread(reading.wait, 10)
            on_disk = json.loads(bridge.outbox._record_path(record).read_text(encoding="utf-8"))
            assert (on_disk["state"], on_disk["reason"], on_disk.get("finished_at"), on_disk.get("held")) == (
                "failed", "changed", None, None)
            await bridge.stop()  # the process ends here
            release.set()
            again = _bridge(data, second.base)
            again.start()
            await _until(lambda: again.outbox.summary().get("kept_total") == 1)  # its end decided and shown
            [ended] = again.records()
            assert ended.get("held") and ended["copy"] == "only"
            assert (again.state / "outbox" / f"{ended['id']}.event.json").exists()
            assert ended["id"] in _kept(again) and "Only copy kept on this computer" in _not_delivered(again)
            await again.stop()
        assert len(first.sent()) == 1 and second.requests == []
    _run(scenario)


def test_deleting_kept_copies_removes_bytes_before_records_and_no_failure_hides_or_invents_any(tmp_path, monkeypatch):
    """A press whose removal fails leaves that copy kept, listed and counted; one whose bytes went but whose
    record could not say so is deleted in memory and, after a reload, from what the disk shows; pressing again
    deletes what stayed. Nothing is ever sent."""
    data = tmp_path / "data"

    async def scenario():
        async with _Origin(["403"] * 3) as origin:
            bridge = _bridge(data, origin.base)
            for text in ("first", "second", "third"):
                await bridge.publish("chat.outbound", {**_ONLY, "text": f"The {text} answer, kept."})
            a, b, c = bridge.records()
            size = {r["id"]: r["payload_bytes"] for r in (a, b, c)}
            assert all(r.get("held") for r in (a, b, c)) and bridge.outbox.summary()["kept_bytes"] == sum(size.values())
            module = bridge.module("telegram_outbox")
            unlink, write = module._unlink, module._write_private

            def refusing_unlink(path):
                if path.name == f"{a['id']}.event.json":
                    raise PermissionError("read-only file")
                unlink(path)

            def refusing_write(path, content):
                if path.name == f"{b['id']}.json":
                    raise PermissionError("read-only state")
                write(path, content)

            monkeypatch.setattr(module, "_unlink", refusing_unlink)
            monkeypatch.setattr(module, "_write_private", refusing_write)
            client = await _copies_route(bridge)
            shown = (await client.get("/copies")).json()
            done = (await client.post("/copies", json={**shown, "delete_ids": f"{a['id']}\n{b['id']}",
                                                       "confirm_delete": True})).json()
            await client.aclose()
            assert done["message"] == ("Deleted 1 kept copy. 1 could not be deleted (see the skill's log). 2 other "
                                       "kept copies stay; open this page again to delete them.")
            outbox = bridge.state / "outbox"
            assert a.get("held") and (outbox / f"{a['id']}.event.json").exists()  # the failed removal: still kept
            assert not b.get("held") and b.get("deleted_at") and not (outbox / f"{b['id']}.event.json").exists()
            assert json.loads((outbox / f"{b['id']}.json").read_text())["held"] is True  # its record lagged
            summary = bridge.outbox.summary()
            assert (summary["kept_total"], summary["kept_bytes"]) == (2, size[a["id"]] + size[c["id"]])
            assert done["kept_copies"].splitlines()[0] == _took(bridge, 2, size[a["id"]] + size[c["id"]])
            assert "Refused by Telegram. Its kept copy was deleted." in _not_delivered(bridge)
            monkeypatch.setattr(module, "_unlink", unlink)
            monkeypatch.setattr(module, "_write_private", write)
            again = _bridge(data, origin.base)
            again.outbox._load()
            records = {r["id"]: r for r in again.records()}
            assert records[a["id"]].get("held") and records[c["id"]].get("held")
            assert not records[b["id"]].get("held") and records[b["id"]].get("deleted_at")  # read as the disk shows
            summary = again.outbox.summary()
            assert (summary["kept_total"], summary["kept_bytes"]) == (2, size[a["id"]] + size[c["id"]])
            assert await again.outbox.clear([a["id"]]) == {"deleted": 1, "missing": 0, "failed": 0}
            assert sorted(name for name in again.files() if name.endswith(".event.json")) == [f"{c['id']}.event.json"]
            assert len(origin.requests) == 3  # deleting sends nothing
    _run(scenario)


_STRAY = "000000000090-0bad0bad.event.json"  # an event file the next load finds without its record


async def _overlapping_presses(bridge, client, press: dict, monkeypatch, failing: str) -> tuple:
    """The same shown ids pressed on two devices: the second press reaches the outbox's ``clear`` while the first is
    still removing its first copy's bytes. ``failing`` names a file whose removal fails."""
    module = bridge.module("telegram_outbox")
    unlink, clear = module._unlink, bridge.outbox.clear
    reached, release, second_in, presses = threading.Event(), threading.Event(), asyncio.Event(), []

    def paused(path):
        reached.set()
        assert release.wait(10)
        if path.name == failing:
            raise PermissionError("read-only file")
        unlink(path)

    async def entering(names):
        presses.append(names)
        if len(presses) == 2:
            second_in.set()
        return await clear(names)

    monkeypatch.setattr(module, "_unlink", paused)
    monkeypatch.setattr(bridge.outbox, "clear", entering)
    first, second = asyncio.ensure_future(client.post("/copies", json=press)), None
    try:
        assert await asyncio.to_thread(reached.wait, 10)  # the first press is removing bytes
        second = asyncio.ensure_future(client.post("/copies", json=press))
        await asyncio.wait_for(second_in.wait(), 10)  # meanwhile the second press is in clear too
    finally:
        release.set()
    answers = await asyncio.gather(first, second)
    monkeypatch.setattr(module, "_unlink", unlink)
    assert [answer.status_code for answer in answers] == [200, 200]
    return tuple(answer.json() for answer in answers)


@pytest.mark.parametrize("shown", ["kept-copies", "stray"])
def test_presses_that_overlap_delete_each_copy_once_and_the_later_says_what_the_earlier_deleted(tmp_path, monkeypatch,
                                                                                                 shown):
    """Two devices press Delete copies on the same shown ids at once: ordinary kept copies, a stray file and a copy
    whose removal fails, or a stray file alone; a run holds the queue meanwhile, as a send that hangs would. The later
    press waits for the earlier, never for the queue, then counts what that one deleted as no longer kept: no error,
    no second deletion. The copy whose removal failed stays kept and listed; one beyond the listed window and one kept
    after the form was drawn keep their exact bytes. Nothing is sent."""
    data = tmp_path / "data"

    async def scenario():
        async with _Origin(["403"] * 5) as origin:
            bridge = _bridge(data, origin.base)
            for text in ("first", "second", "third", "fourth") if shown == "kept-copies" else ():
                await bridge.publish("chat.outbound", {**_ONLY, "text": f"The {text} answer, kept."})
            outbox = bridge.state / "outbox"
            outbox.mkdir(parents=True, exist_ok=True)
            (outbox / _STRAY).write_text(json.dumps({"text": "saved before its record", "transport": {}}), encoding="utf-8")
            again = _bridge(data, origin.base)  # the next load finds the stray file
            monkeypatch.setattr(again.module("telegram_outbox"), "SHOWN", 3)
            again.outbox._load()
            kept = [r["id"] for r in again.records() if r.get("held")]  # oldest first; the form lists the newest three
            client = await _copies_route(again)
            form = (await client.get("/copies")).json()
            assert form["delete_ids"].splitlines() == [*kept[:0:-1], _handle(_STRAY)]
            await again.publish("chat.outbound", {**_ONLY, "text": "Kept after the form was drawn."})
            newer = again.records()[-1]["id"]
            stays = {newer, *kept[:1], *kept[2:3]}  # kept later, beyond the listed window, and the one that fails
            before = {path.name: path.read_bytes() for path in outbox.glob("*.event.json")}
            asked = len(origin.requests)
            async with again.outbox._fifo():  # a run holds the queue's lock throughout
                one, two = await _overlapping_presses(again, client, {**form, "confirm_delete": True}, monkeypatch,
                                                      f"{kept[2]}.event.json" if kept else "")
            await client.aclose()
            if kept:
                assert one["message"] == ("Deleted 3 kept copies. 1 could not be deleted (see the skill's log). 3 other "
                                          "kept copies stay; open this page again to delete them.")
                assert two["message"] == ("Deleted 0 kept copies. 1 could not be deleted (see the skill's log). 3 no "
                                          "longer kept. 3 other kept copies stay; open this page again to delete them.")
            else:
                assert one["message"] == "Deleted 1 kept copy. 1 other kept copy stays; open this page again to delete it."
                assert two["message"] == ("Deleted 0 kept copies. 1 no longer kept. 1 other kept copy stays; open this "
                                          "page again to delete it.")
            remaining = {f"{item}.event.json" for item in stays}
            assert {path.name: path.read_bytes() for path in outbox.glob("*.event.json")} == {
                name: before[name] for name in remaining}  # exactly what stays, byte for byte; the rest is gone
            assert {r["id"] for r in again.records() if r.get("held")} == stays and one["delete_ids"] == two["delete_ids"] == ""
            assert all(item in two["kept_copies"].splitlines() for item in stays)
            assert all(r.get("deleted_at") for r in again.records() if r["id"] in kept and r["id"] not in stays)
            summary = again.outbox.summary()
            assert (summary["kept_total"], summary["kept_bytes"]) == (len(stays), sum(map(len, map(before.get, remaining))))
            reloaded = _bridge(data, origin.base)
            reloaded.outbox._load()
            assert {r["id"] for r in reloaded.records() if r.get("held")} == stays
            assert reloaded.outbox.summary()["kept_total"] == len(stays)
            if kept:  # the deletions' lock is free again: pressing once more deletes what stayed
                assert await again.outbox.clear([kept[2]]) == {"deleted": 1, "missing": 0, "failed": 0}
            assert len(origin.requests) == asked  # deleting sends nothing
    _run(scenario)


@pytest.mark.parametrize("delete_load", ["same", "reload"])
@pytest.mark.parametrize("record_written", [True, False], ids=["record-written", "no-record"])
def test_a_refusal_whose_cleanup_fails_keeps_its_file_shown_and_counted(tmp_path, monkeypatch, record_written, delete_load):
    """An event saved before its record failed is refused; when its saved event cannot be removed either, that
    file is a kept copy (its refusal's record says so) or, when no record can be written, a stray found as such:
    shown, counted by its real size, and the same after a reload. Nothing is sent."""
    data = tmp_path / "data"

    async def scenario():
        async with _Origin() as origin:
            bridge = _bridge(data, origin.base)
            module = bridge.module("telegram_outbox")
            unlink, write, record_writes = module._unlink, module._write_private, []

            def refusing_unlink(path):
                if path.name.endswith(".event.json"):
                    raise PermissionError("read-only file")
                unlink(path)

            def failing_write(path, content):
                if not path.name.endswith(".event.json"):
                    record_writes.append(path.name)
                    if len(record_writes) == 1 or not record_written:
                        raise PermissionError("read-only state")
                write(path, content)

            monkeypatch.setattr(module, "_unlink", refusing_unlink)
            monkeypatch.setattr(module, "_write_private", failing_write)
            await bridge.publish("chat.outbound", {**_ONLY, "text": "Saved, then its record failed."})
            assert origin.requests == []
            [name] = [name for name in bridge.files() if name.endswith(".event.json")]
            size = (bridge.state / "outbox" / name).stat().st_size
            summary = bridge.outbox.summary()
            assert (summary["not_sent"], summary["kept_total"], summary["kept_bytes"]) == (1, 1, size)
            assert _not_delivered(bridge).endswith(
                " · Reply · Main\nNot sent: the outbox couldn't save it. Only copy kept on this computer (Kept copies).")
            if record_written:
                [record] = bridge.records()
                assert record.get("held") and record["payload_bytes"] == size and f"{record['id']}.event.json" == name
            else:
                assert bridge.records() == [] and [item["name"] for item in summary["kept"]] == [name]
                # Even an unreadable metadata file with the same record id is not this lost payload.
                other = bridge.state / "outbox" / name.replace(".event.json", ".json")
                other.write_text("unreadable metadata", encoding="utf-8")
                bridge.outbox._strays.append(module._stray(other, "unreadable"))
                assert await bridge.outbox.clear([_handle(other.name)]) == {"deleted": 1, "missing": 0, "failed": 0}
                assert bridge.outbox.summary()["kept_total"] == 1 and "Only copy kept" in _not_delivered(bridge)
                assert not bridge.outbox._lost[0].get("deleted_at")
            assert _kept(bridge).splitlines()[0] == _took(bridge, 1, size)
            kept = summary["kept"][0]
            assert await bridge.outbox.clear([kept["id"]]) == {"deleted": 0, "missing": 0, "failed": 1}
            assert bridge.outbox.summary()["kept_total"] == 1 and "Only copy kept" in _not_delivered(bridge)
            monkeypatch.setattr(module, "_unlink", unlink)
            monkeypatch.setattr(module, "_write_private", write)
            again = bridge if delete_load == "same" else _bridge(data, origin.base)
            again.outbox._load()
            summary = again.outbox.summary()
            assert (summary["kept_total"], summary["kept_bytes"]) == (1, size)
            kept = summary["kept"][0]
            assert kept.get("held") if record_written else (kept["stray"], kept["name"]) == ("orphan", name)
            assert await again.outbox.clear([kept["id"]]) == {"deleted": 1, "missing": 0, "failed": 0}
            assert not [name for name in again.files() if name.endswith(".event.json")]
            assert again.outbox.summary()["kept_total"] == 0
            if record_written or delete_load == "same":
                assert "Its kept copy was deleted." in _not_delivered(again)
                assert "Only copy kept" not in _not_delivered(again)
            assert origin.requests == []
    _run(scenario)


def test_a_saved_copy_its_end_cannot_remove_stays_a_kept_copy(tmp_path, monkeypatch):
    """Content the app holds lets its saved copy go at its end; when that removal fails, the file stays a kept copy,
    shown, counted and deletable, instead of bytes no list names."""
    data = tmp_path / "data"

    async def scenario():
        produce = _Producer(monkeypatch, data)
        async with _Origin(["403"]) as origin:
            bridge = _bridge(data, origin.base)
            module = bridge.module("telegram_outbox")
            unlink = module._unlink

            def refusing_unlink(path):
                if path.name.endswith(".event.json"):
                    raise PermissionError("read-only file")
                unlink(path)

            monkeypatch.setattr(module, "_unlink", refusing_unlink)
            await bridge.publish(*produce.reply("An answer the app holds."))
            [record] = bridge.records()
            assert (record["state"], record["copy"], record.get("held")) == ("refused", "home", True)
            assert _not_delivered(bridge).endswith("Refused by Telegram. A copy stays on this computer (Kept copies).")
            size = (bridge.state / "outbox" / f"{record['id']}.event.json").stat().st_size
            assert bridge.outbox.summary()["kept_bytes"] == size
            monkeypatch.setattr(module, "_unlink", unlink)
            assert await bridge.outbox.clear([record["id"]]) == {"deleted": 1, "missing": 0, "failed": 0}
            assert bridge.outbox.summary()["kept_total"] == 0 and len(origin.requests) == 1
    _run(scenario)


def test_a_documents_reference_is_kept_when_its_task_file_goes_and_said_as_such(tmp_path, monkeypatch):
    """A captured document is saved without its inline bytes (its handler reads the task file). When that file
    disappears before its end, the copy kept is its message and the file's reference: no stale home is claimed, the
    form says what is kept and what is not, and the recipe saves no bytes it does not have. Nothing is sent."""
    from ouroboros import event_bus
    from tests._dialogue import chat_rows

    data, repo = tmp_path / "data", tmp_path / "repo"
    repo.mkdir()
    caption = "The only retained caption: final report from Friday."
    for key, value in (("OUROBOROS_RUNTIME_MODE", "advanced"), ("OUROBOROS_SAFETY_MODE", "off"),
                       ("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")):
        monkeypatch.setenv(key, value)
    monkeypatch.setattr("ouroboros.safety._run_llm_check", lambda *_a, **_k: pytest.fail("No model call expected"))

    async def scenario():
        produce = _Producer(monkeypatch, data)
        document = produce.document(_blob(40_000, 90), captured=True, filename="report.pdf", caption=caption)
        bridge = _bridge(data, _closed_base())
        await bridge.publish(*document)
        [record] = bridge.records()
        saved = json.loads(bridge.outbox._payload_path(record).read_bytes())
        assert record["copy"] == "home" and "file_base64" not in saved and saved["file_ref"]["path"] == "report.pdf"
        from ouroboros.artifacts import task_artifact_dir_path

        (task_artifact_dir_path(data, "work-1") / "report.pdf").unlink()  # the canonical file goes while it waits
        bridge.now += 24 * 3600 + 1
        async with _Origin() as late:
            bridge.serve(late.base)
            await bridge.drain()
        assert late.requests == []
        assert (record["state"], record["reason"], record["copy"], record.get("held")) == (
            "failed", "expired", "reference", True)
        assert "home" not in record  # the home found at admission is gone and no longer claimed
        assert _not_delivered(bridge).endswith(" · File report.pdf, 39 KB · Main\nNot sent within a day. Its message "
                                               "and the file's reference are kept on this computer, not the file "
                                               "(Kept copies).")
        kept = _kept(bridge).splitlines()
        size = bridge.outbox._payload_path(record).stat().st_size
        assert kept[0] == _took(bridge, 1, size) and kept[2].endswith(
            " · File report.pdf (reference only) · Main") and kept[3] == record["id"]
        await bridge.stop()
        recipe = _RECIPE.search((_REPO / "skills" / "telegram" / "SKILL.md").read_text(encoding="utf-8")).group(1)
        before = {p.name: p.read_bytes() for p in (bridge.state / "outbox").iterdir()}
        rows, published = len(chat_rows(data)), []
        monkeypatch.setattr(event_bus, "publish_event", lambda *args, **_k: published.append(args))
        for name, mode in (("report.pdf", None), ("report.txt", None), ("explicit.txt", "bytes"),
                           ("caption.txt", "words")):
            result, response = await asyncio.to_thread(
                _retrieve_kept_copy, data, repo, recipe, record["id"], name, monkeypatch, mode)
            print(f"REFERENCE_RETRIEVAL {name}: {result}")
            if mode == "words":
                assert "ARTIFACT_OUTPUTS" in result and response.status_code == 200, result
                assert response.content == caption.encode()
                assert '"words": true' in result
            else:
                assert "This copy holds no bytes, only its task file's reference" in result
                assert response.status_code == 404
        assert {p.name: p.read_bytes() for p in (bridge.state / "outbox").iterdir()} == before
        assert len(chat_rows(data)) == rows and published == [] and produce.events == []
        assert late.requests == [] and record["held"] and record["state"] == "failed"
    _run(scenario)
