"""Which accepted row an id names, without replaying the chat chain per send (``message_ingress._AcceptedIds``).

Every web send and named skill delivery asks this under the ingress lock. The answer must be the
one a full scan of the retained chain gives — the newest generation holding the id, the first row
naming it there, a historical id never treated as fresh — while a warm lookup reads only the live
file's identity, two short anchors and the bytes appended since the last one (DEVELOPMENT 03
"Projection over replay"). Rotation folds the rotated remainder, a rewrite folds again, and an
unreadable chain refuses instead of answering "absent".
"""
from __future__ import annotations

import json
import os
import pathlib
import threading

import pytest

from ouroboros.utils import append_jsonl, iter_jsonl_objects, jsonl_chain_handles
from supervisor import message_ingress
from supervisor.message_ingress import accepted_chat_message
from tests.test_chat_attachments import _rows, _send_web, _web_bridge


def _scan(root, chat_id, client_message_id):
    """The full-chain replay the index replaces: the reference answer."""
    with jsonl_chain_handles(pathlib.Path(root) / "logs" / "chat.jsonl", strict=True) as handles:
        for path, handle in reversed(handles):
            for row in iter_jsonl_objects(path, _handle=handle):
                if (row.get("direction") == "in" and row.get("chat_id") == chat_id
                        and row.get("client_message_id") == client_message_id):
                    return row
    return None


def _line(cmid, text, *, chat_id=1, direction="in", ascii_only=False):
    return json.dumps({"direction": direction, "chat_id": chat_id, "client_message_id": cmid, "text": text},
                      ensure_ascii=ascii_only) + "\n"


def _rotate(root):
    from supervisor.state import rotate_jsonl_log_if_needed

    before = set((root / "archive").glob("chat_*.jsonl")) if (root / "archive").exists() else set()
    rotate_jsonl_log_if_needed(root, "chat.jsonl", "chat", max_bytes=1)
    assert len(set((root / "archive").glob("chat_*.jsonl")) - before) == 1, "the real rotator renamed the live file"


PROBES = [(1, "a"), (2, "a"), (3, "a"), (1, "dup"), (1, 'тест"2'), (1, "torn-archive"), (1, "1"),
          (1, "live-1"), (1, "unfinished"), (1, "after"), (1, "rotated-tail"), (1, "new-live"), (1, "glued"),
          (1, "absent"), (1, "liv"), (1, ""), (1, "out-only")]


def _agree(root):
    for chat_id, cmid in PROBES:
        assert accepted_chat_message(root, chat_id, cmid) == _scan(root, chat_id, cmid), (chat_id, cmid)


def test_the_index_answers_exactly_as_the_full_scan_through_appends_tails_and_rotations(tmp_path):
    (tmp_path / "archive").mkdir()
    (tmp_path / "logs").mkdir()
    # An older generation: duplicates, an outbound row mentioning "in", a malformed line, an escaped id,
    # a chat_id of another type and a complete row that lost only its newline when it was rotated.
    (tmp_path / "archive" / "chat_20261001T000000.jsonl").write_text(
        _line("a", "first a") + _line("out-only", 'said "in" here', direction="out") + _line("dup", "oldest dup")
        + "{not json\n" + _line("a", "other chat", chat_id=2) + _line('тест"2', "escaped", ascii_only=True)
        + _line("1", "string chat", chat_id="1") + _line("", "an unnamed inbound row")
        + _line("torn-archive", "lost its newline").rstrip("\n"), encoding="utf-8")
    (tmp_path / "archive" / "chat_20261002T000000.jsonl").write_text(
        _line("dup", "newer dup, first in its generation") + _line("dup", "second in the same generation")
        + _line("a", "a again, newer generation"), encoding="utf-8")
    live = tmp_path / "logs" / "chat.jsonl"
    live.write_text(_line("live-1", "live") + _line("unfinished", "a crashed writer's").rstrip("\n"), encoding="utf-8")
    _agree(tmp_path)  # cold: the unfinished complete row is read, as the scan reads it

    append_jsonl(live, {"direction": "in", "chat_id": 1, "client_message_id": "after", "text": "x"},
                 ensure_record_boundary=True)  # completes the unfinished line
    _agree(tmp_path)
    append_jsonl(live, {"direction": "in", "chat_id": 1, "client_message_id": "rotated-tail", "text": "y"})
    _rotate(tmp_path)  # appended after the last fold, then rotated: its remainder is folded from the archive
    _agree(tmp_path)
    append_jsonl(live, {"direction": "in", "chat_id": 1, "client_message_id": "new-live", "text": "z"})
    with live.open("ab") as handle:  # a complete object without its newline...
        handle.write(_line("glued", "first half").rstrip("\n").encode("utf-8"))
    _agree(tmp_path)
    append_jsonl(live, {"direction": "in", "chat_id": 1, "client_message_id": "dup", "text": "glued on"})
    _agree(tmp_path)  # ...then an append without a boundary: one malformed line, both skipped, as the scan does
    assert message_ingress._accepted_ids(tmp_path).cold_folds == 1, "one fold for the whole sequence"


def _seed(root, rows_per_generation=400, generations=4):
    (root / "archive").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    for index in range(generations):
        lines = "".join(_line(f"g{index}-{n}", "x" * 200) + _line("", "reply " * 40, direction="out")
                        for n in range(rows_per_generation))
        target = root / "logs" / "chat.jsonl" if index == generations - 1 else root / "archive" / f"chat_2026100{index + 1}T000000.jsonl"
        target.write_text(lines, encoding="utf-8")


def test_a_warm_fresh_lookup_reads_only_the_live_file_and_the_lines_appended_since(tmp_path, monkeypatch):
    _seed(tmp_path)
    live = tmp_path / "logs" / "chat.jsonl"
    assert accepted_chat_message(tmp_path, 1, "fresh-0") is None  # the one cold fold
    index = message_ingress._accepted_ids(tmp_path)
    assert index.cold_folds == 1
    append_jsonl(live, {"direction": "in", "chat_id": 1, "client_message_id": "appended", "text": "new"})
    append_jsonl(live, {"direction": "out", "chat_id": 1, "client_message_id": "", "text": "an answer"})
    opened, parsed = [], []
    real_open, real_parsed = pathlib.Path.open, message_ingress._parsed
    monkeypatch.setattr(pathlib.Path, "open", lambda self, *a, **k: opened.append(self.name) or real_open(self, *a, **k))
    monkeypatch.setattr(message_ingress, "_parsed", lambda raw: parsed.append(raw) or real_parsed(raw))
    for attempt in range(3):
        assert accepted_chat_message(tmp_path, 1, f"fresh-{attempt + 1}") is None
    assert opened == ["chat.jsonl"] * 3, "no archive is opened on the warm path"
    assert len(parsed) == 1 and b"appended" in parsed[0], "only the appended inbound line was parsed, once"
    assert accepted_chat_message(tmp_path, 1, "g0-7")["client_message_id"] == "g0-7", "a historical id is known"
    assert opened[-1] == "chat_20261001T000000.jsonl", "a hit re-reads its one line from its own generation"
    assert accepted_chat_message(tmp_path, 1, "appended")["text"] == "new"
    assert index.cold_folds == 1


def test_a_historical_id_from_an_ended_process_only_rejoins_after_rotations(tmp_path, monkeypatch):
    """Never old as fresh: a redelivered frame naming a row from long ago rejoins it (no second row, no
    dispatch), and a changed frame under that id is refused — whatever the index folded since."""
    from ouroboros import process_custody

    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    _send_web(bridge, "давнее сообщение", [], cmid="old-id")
    for round_ in range(3):  # later traffic, warm lookups and rotations bury it in the archive
        for n in range(5):
            _send_web(bridge, f"later {round_}-{n}", [], cmid=f"later-{round_}-{n}")
        _rotate(tmp_path)
    monkeypatch.setattr(process_custody, "_SESSION_ID", "the-next-host-process")
    queued, rows = bridge._inbox.qsize(), len(_rows(tmp_path)) + sum(
        1 for path in (tmp_path / "archive").glob("chat_*.jsonl") for _ in path.read_text(encoding="utf-8").splitlines())
    _send_web(bridge, "давнее сообщение", [], cmid="old-id")
    assert bridge._inbox.qsize() == queued, "unknown is never replayed"
    assert echoes[-1]["client_message_id"] == "old-id" and not {"ingress_dispatched", "ingress_pending"} & set(echoes[-1])
    with pytest.raises(ValueError, match="different message"):
        _send_web(bridge, "другие слова", [], cmid="old-id")
    after = len(_rows(tmp_path)) + sum(
        1 for path in (tmp_path / "archive").glob("chat_*.jsonl") for _ in path.read_text(encoding="utf-8").splitlines())
    assert after == rows, "no second row for the old id"
    assert message_ingress._accepted_ids(tmp_path).cold_folds == 1


def test_a_rewritten_live_file_folds_again_and_an_unreadable_one_refuses(tmp_path, monkeypatch):
    _seed(tmp_path, rows_per_generation=50, generations=3)
    live = tmp_path / "logs" / "chat.jsonl"
    assert accepted_chat_message(tmp_path, 1, "g2-3")["client_message_id"] == "g2-3"
    index = message_ingress._accepted_ids(tmp_path)
    # The same inode rewritten in place with other rows (longer than the folded prefix).
    live.write_text(_line("rewritten", "y" * 50000), encoding="utf-8")
    assert accepted_chat_message(tmp_path, 1, "g2-3") is None and _scan(tmp_path, 1, "g2-3") is None
    assert accepted_chat_message(tmp_path, 1, "rewritten")["client_message_id"] == "rewritten"
    assert accepted_chat_message(tmp_path, 1, "g0-3")["client_message_id"] == "g0-3", "archived ids stay known"
    assert index.cold_folds == 2, "the changed prefix was folded again, never advanced over"

    real_open = pathlib.Path.open

    def refuse_live(self, *args, **kwargs):
        if self == live:
            raise PermissionError("the live chat log cannot be read")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "open", refuse_live)
    with pytest.raises(OSError):
        accepted_chat_message(tmp_path, 1, "fresh")
    monkeypatch.setattr(pathlib.Path, "open", real_open)
    assert accepted_chat_message(tmp_path, 1, "fresh") is None
    assert index.cold_folds == 3, "unknown was not cached as absent: the next lookup read the chain again"


def test_concurrent_fresh_web_sends_fold_the_chain_once_and_each_id_is_accepted_once(tmp_path, monkeypatch):
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
    assert message_ingress._accepted_ids(tmp_path).cold_folds == 1


def test_the_index_is_per_chain_and_a_missing_chain_names_nothing(tmp_path):
    assert accepted_chat_message(tmp_path / "fresh-install", 1, "x") is None
    other = tmp_path / "other"
    (other / "logs").mkdir(parents=True)
    (other / "logs" / "chat.jsonl").write_text(_line("x", "elsewhere"), encoding="utf-8")
    assert accepted_chat_message(other, 1, "x")["text"] == "elsewhere"
    assert accepted_chat_message(tmp_path / "fresh-install", 1, "x") is None
    assert os.path.abspath(other / "logs" / "chat.jsonl") in message_ingress._ACCEPTED_IDS
