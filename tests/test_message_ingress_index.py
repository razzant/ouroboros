"""Which accepted row an id names, answered by the dialogue store (``message_ingress.accepted_chat_message``).

Every web send and named skill delivery asks this under the ingress lock. The answer must be the
one a full scan of the retained chain gave -- the newest row naming the id, a historical id never
treated as fresh -- for the rows the archive import brought in and for the rows appended since.
While the archive is still importing, a miss is checked in the segments not yet imported, so an
old id redelivered during the import rejoins instead of landing a second row; a store that cannot
be read refuses typed instead of answering "absent".
"""
from __future__ import annotations

import json
import threading

import pytest

from ouroboros import dialogue_import, dialogue_store, store
from ouroboros.utils import iter_jsonl_objects, jsonl_chain_handles
from supervisor import message_ingress
from supervisor.message_ingress import accepted_chat_message
from tests.test_chat_attachments import _rows, _send_web, _web_bridge


def _scan(root, chat_id, client_message_id):
    """The full-chain replay the store replaces: the reference answer (chat ids read as the room
    readers read them, ``chat_chain._row_chat_id``)."""
    def same_chat(row):
        try:
            return int(row.get("chat_id", 1)) == chat_id
        except (TypeError, ValueError):
            return False

    with jsonl_chain_handles(root / "logs" / "chat.jsonl", strict=True) as handles:
        for path, handle in reversed(handles):
            found = None
            for row in iter_jsonl_objects(path, _handle=handle):
                if row.get("direction") == "in" and same_chat(row) and row.get("client_message_id") == client_message_id:
                    found = row  # the newest row of the newest generation holding the id
            if found is not None:
                return found
    return None


def _line(cmid, text, *, chat_id=1, direction="in", ascii_only=False):
    return json.dumps({"direction": direction, "chat_id": chat_id, "client_message_id": cmid, "text": text},
                      ensure_ascii=ascii_only) + "\n"


PROBES = [(1, "a"), (2, "a"), (3, "a"), (1, "dup"), (1, 'тест"2'), (1, "torn-archive"), (1, "1"),
          (1, "live-1"), (1, "unfinished"), (1, "absent"), (1, "liv"), (1, "out-only")]


def _agree(root, expected):
    for chat_id, cmid in PROBES:
        assert accepted_chat_message(root, chat_id, cmid) == expected(root, chat_id, cmid), (chat_id, cmid)


def _legacy_chain(root):
    (root / "archive").mkdir()
    (root / "logs").mkdir()
    # An older generation: duplicates, an outbound row mentioning "in", a malformed line, an escaped id,
    # a chat_id of another type and a complete row that lost only its newline when it was rotated.
    (root / "archive" / "chat_20261001T000000.jsonl").write_text(
        _line("a", "first a") + _line("out-only", 'said "in" here', direction="out") + _line("dup", "oldest dup")
        + "{not json\n" + _line("a", "other chat", chat_id=2) + _line('тест"2', "escaped", ascii_only=True)
        + _line("1", "string chat", chat_id="1") + _line("", "an unnamed inbound row")
        + _line("torn-archive", "lost its newline").rstrip("\n"), encoding="utf-8")
    (root / "archive" / "chat_20261002T000000.jsonl").write_text(
        _line("dup", "newer dup, first in its generation") + _line("dup", "second in the same generation")
        + _line("a", "a again, newer generation"), encoding="utf-8")
    live = root / "logs" / "chat.jsonl"
    live.write_text(_line("live-1", "live") + _line("unfinished", "a crashed writer's").rstrip("\n"), encoding="utf-8")


def test_the_store_answers_exactly_as_the_full_scan_of_the_chain_it_imported(tmp_path):
    _legacy_chain(tmp_path)
    expected = {(chat_id, cmid): _scan(tmp_path, chat_id, cmid) for chat_id, cmid in PROBES}
    dialogue_import.cutover(tmp_path)  # the live file: the final segment, imported at once
    assert accepted_chat_message(tmp_path, 1, "live-1") == expected[(1, "live-1")]
    assert accepted_chat_message(tmp_path, 1, "unfinished") == expected[(1, "unfinished")], "a crashed writer's complete row"
    _agree(tmp_path, lambda root, chat_id, cmid: expected[(chat_id, cmid)])  # misses check the pending segments
    assert accepted_chat_message(tmp_path, 1, "") is None, "an empty id names nothing (the old fold answered the unnamed row)"
    dialogue_import.run_dialogue_import(tmp_path)
    _agree(tmp_path, lambda root, chat_id, cmid: expected[(chat_id, cmid)])
    assert accepted_chat_message(tmp_path, 1, "dup")["text"] == "second in the same generation", "the newest row naming the id"
    appended = dialogue_store.append(tmp_path, "chat", {"direction": "in", "chat_id": 1, "client_message_id": "after", "text": "x"})
    assert accepted_chat_message(tmp_path, 1, "after") == appended.row
    again = dialogue_store.append(tmp_path, "chat", {"direction": "in", "chat_id": 1, "client_message_id": "dup", "text": "glued on"})
    assert again.duplicate and accepted_chat_message(tmp_path, 1, "dup")["text"] == "second in the same generation", "a historical id stays"
    assert dialogue_store.counts(tmp_path)["chat"] == 14, "13 chain rows (the malformed line is none) and the one appended"


def _seed(root, rows_per_generation=400, generations=4):
    (root / "archive").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    for index in range(generations):
        lines = "".join(_line(f"g{index}-{n}", "x" * 200) + _line("", "reply " * 40, direction="out")
                        for n in range(rows_per_generation))
        target = root / "logs" / "chat.jsonl" if index == generations - 1 else root / "archive" / f"chat_2026100{index + 1}T000000.jsonl"
        target.write_text(lines, encoding="utf-8")


def test_a_historical_id_from_an_ended_process_only_rejoins_during_and_after_the_import(tmp_path, monkeypatch):
    """Never old as fresh: a redelivered frame naming a row from long ago rejoins it (no second row, no
    dispatch), and a changed frame under that id is refused -- while its segment is still importing
    and after."""
    _seed(tmp_path)
    old = {"ts": "2026-10-01T00:00:00+00:00", "direction": "in", "chat_id": 1, "user_id": 1, "source": "web",
           "client_message_id": "old-id", "text": "давнее сообщение", "ingress_accepted": True, "ingress_process": "an-ended-host"}
    with (tmp_path / "archive" / "chat_20261001T000000.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(old, ensure_ascii=False) + "\n")
    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    dialogue_import.cutover(tmp_path)
    assert dialogue_import.coverage(tmp_path, "chat")["pending"] == 3, "the old row's segment is not imported yet"
    for stage in ("importing", "imported"):
        queued, rows = bridge._inbox.qsize(), dialogue_store.counts(tmp_path)["chat"]
        _send_web(bridge, "давнее сообщение", [], cmid="old-id")
        assert bridge._inbox.qsize() == queued, f"{stage}: unknown is never replayed"
        assert echoes[-1]["client_message_id"] == "old-id" and not {"ingress_dispatched", "ingress_pending"} & set(echoes[-1])
        with pytest.raises(ValueError, match="different message"):
            _send_web(bridge, "другие слова", [], cmid="old-id")
        assert dialogue_store.counts(tmp_path)["chat"] == rows, f"{stage}: no second row for the old id"
        dialogue_import.run_dialogue_import(tmp_path)
    assert accepted_chat_message(tmp_path, 1, "old-id")["text"] == "давнее сообщение"
    assert message_ingress.dispatch_facts(old, tmp_path) is None, "an imported row owes nothing"


def test_a_redelivery_after_a_cutover_that_failed_past_its_rotation_rejoins(tmp_path, monkeypatch):
    """The first cutover rotated the live file, then failed before its plan (the boot opens the
    dialogue anyway): a redelivered frame naming a row of that file finishes the cutover before its
    id is looked up, so the rotated segment is searched as pending history and the frame rejoins."""
    from tests._dialogue import seed

    accepted = {"ts": "2026-10-16T21:00:00+00:00", "direction": "in", "chat_id": 1, "user_id": 1, "source": "web",
                "client_message_id": "recent-id", "text": "недавнее сообщение"}
    seed(tmp_path, live={"chat": [accepted]})
    with monkeypatch.context() as patch:
        patch.setattr(dialogue_import, "_plan",
                      lambda *_a: (_ for _ in ()).throw(dialogue_store.StoreBusy("held after the final rotation")))
        with pytest.raises(dialogue_store.StoreBusy):
            dialogue_import.cutover(tmp_path)
    assert dialogue_store.meta(tmp_path, "cutover") is None and (tmp_path / "logs" / "chat.jsonl").stat().st_size == 0
    bridge, _echoes = _web_bridge(tmp_path, monkeypatch)
    _send_web(bridge, "недавнее сообщение", [], cmid="recent-id")
    assert bridge._inbox.qsize() == 0, "an accepted message rejoins; it is never dispatched again"
    dialogue_import.run_dialogue_import(tmp_path)
    assert [row["client_message_id"] for row in _rows(tmp_path) if row.get("client_message_id")] == ["recent-id"]
    _send_web(bridge, "новое сообщение", [], cmid="fresh-id")
    assert bridge._inbox.qsize() == 1, "a fresh id is still accepted and dispatched"


def test_an_unreadable_store_refuses_instead_of_answering_absent(tmp_path, monkeypatch):
    bridge, _echoes = _web_bridge(tmp_path, monkeypatch)
    _send_web(bridge, "первое", [], cmid="first")
    dialogue_store.path(tmp_path).write_bytes(b"not a database at all" * 10)
    store.close_connections()
    with pytest.raises(store.StoreError):
        accepted_chat_message(tmp_path, 1, "first")
    with pytest.raises(store.StoreError):
        _send_web(bridge, "второе", [], cmid="second")
    assert bridge._inbox.qsize() == 1, "nothing was accepted or dispatched on an unreadable store"


def test_concurrent_fresh_web_sends_are_each_accepted_once(tmp_path, monkeypatch):
    _seed(tmp_path, rows_per_generation=300, generations=4)
    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    start = threading.Barrier(12)
    errors = []

    def send(n):
        try:
            start.wait()
            _send_web(bridge, f"параллельно {n}", [], cmid=f"par-{n}")
            _send_web(bridge, f"параллельно {n}", [], cmid=f"par-{n}")  # its redelivery rejoins
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=send, args=(n,)) for n in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert not errors
    named = [row["client_message_id"] for row in _rows(tmp_path) if row["client_message_id"].startswith("par-")]
    assert sorted(named) == sorted(f"par-{n}" for n in range(12)), "one row per id"
    assert bridge._inbox.qsize() == 12 and len(echoes) == 24


def test_the_store_is_per_root_and_a_missing_one_names_nothing(tmp_path):
    other = tmp_path / "other"
    dialogue_store.append(other, "chat", {"direction": "in", "chat_id": 1, "client_message_id": "x", "text": "there"})
    assert accepted_chat_message(other, 1, "x")["text"] == "there"
    assert accepted_chat_message(tmp_path, 1, "x") is None and not dialogue_store.present(tmp_path), "a lookup creates no store"
    assert accepted_chat_message(other, 2, "x") is None and accepted_chat_message(other, 1, "") is None
