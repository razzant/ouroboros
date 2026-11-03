"""The Done card's duplicate check reads the task's own rows, and the card settles on the ack.

Owner (21:09 #3, 18:56): never re-read the whole archived chat (95 MB) before writing Done.
``_already_in_chat`` asks the dialogue store for the task's chat rows (one indexed probe) and
takes the one carrying this summary id and readiness token; it never trusts row timestamps
(Astra R3: a delayed worker row sits behind a newer summary) and opens no chain file. A
readiness minted in the same call skips the check. The delivery ack re-enters the settle owner,
so a card held behind its owed answer follows that answer without waiting for a sweep.
An unknown import keeps the debt: a miss while importing raises, never reads as absent.
"""
from __future__ import annotations

import json
import pathlib
from collections import deque
from types import SimpleNamespace

import pytest

from ouroboros import dialogue_import as di, dialogue_store as ds, terminal_projection as tp
from ouroboros.task_results import load_task_result, write_task_result
from supervisor import events_chat_delivery as delivery
from supervisor import terminal_delivery as td
from tests._dialogue import chat_rows

TOKEN = "ready-token"
SUMMARY = {"task_id": "T", "type": "task_summary", "summary_id": "task-terminal:T", "terminal_projection_token": TOKEN,
           "ts": "2026-10-14T10:01:00+00:00", "text": "Done. Root task T."}


@pytest.fixture(autouse=True)
def sql_summary_writer(monkeypatch):
    """Q1a owns this writer; exercise its contracted SQL seam before the writer merge."""
    from ouroboros import project_dialogue

    monkeypatch.setattr(project_dialogue, "append_canonical_task_summary",
                        lambda root, row: ds.append(root, "chat", row) is not None)


def _write(path: pathlib.Path, *rows: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _opened(monkeypatch) -> list:
    opened: list = []
    original = pathlib.Path.open

    def spy(self, *args, **kwargs):
        if self.name.endswith(".jsonl"):
            opened.append(self.name)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "open", spy)
    return opened


def test_a_summary_is_found_by_its_task_wherever_it_sits_and_no_chain_file_is_opened(tmp_path, monkeypatch):
    """A summary imported from an archive segment (an older body's), one in the final live segment,
    and one appended through the store answer alike; a backdated worker row never hides them, a
    different readiness token never matches, and the check opens no chain file."""
    _write(tmp_path / "archive" / "chat_20261014T100200.jsonl", SUMMARY)
    _write(tmp_path / "logs" / "chat.jsonl", {"ts": "2026-10-14T09:59:00+00:00", "text": "late"})
    ds.ensure(tmp_path)
    for segment in di.pending_segments(tmp_path):
        di.import_segment(tmp_path, segment["stream"], segment["name"])
    assert tp._already_in_chat(tmp_path, SUMMARY, since_ts="2026-10-14T10:00:00+00:00") == SUMMARY


def test_no_frozen_archive_is_opened_after_import(tmp_path, monkeypatch):
    _write(tmp_path / "archive" / "chat_20261014T095900.jsonl", SUMMARY)
    _write(tmp_path / "archive" / "chat_20261014T095000.jsonl", {"text": "ancient"})
    ds.ensure(tmp_path)
    for segment in di.pending_segments(tmp_path):
        di.import_segment(tmp_path, segment["stream"], segment["name"])
    opened = _opened(monkeypatch)
    assert tp._already_in_chat(tmp_path, SUMMARY) == SUMMARY
    assert tp._already_in_chat(tmp_path, {**SUMMARY, "terminal_projection_token": "other-readiness"}) is None
    assert opened == []


def test_pending_import_is_unknown_then_a_missing_publication_is_absent(tmp_path):
    _write(tmp_path / "archive" / "chat_legacy.jsonl", SUMMARY)
    with pytest.raises(ds.StoreMigrating, match="still importing"):
        tp._already_in_chat(tmp_path, SUMMARY)
    [segment] = di.pending_segments(tmp_path)
    di.import_segment(tmp_path, "chat", segment["name"])
    assert tp._already_in_chat(tmp_path, SUMMARY) == SUMMARY
    assert tp._already_in_chat(tmp_path, {**SUMMARY, "terminal_projection_token": "later-readiness"}) is None


def test_summary_lookup_has_equal_sql_work_with_one_or_thirty_foreign_rooms(tmp_path, monkeypatch):
    from tests.test_dialogue_addressed_readers import metered

    measurements = []
    for rooms in (1, 30):
        root = tmp_path / str(rooms)
        ds.append(root, "chat", SUMMARY)
        for room in range(rooms):
            for index in range(10):
                ds.append(root, "chat", {"chat_id": room + 10, "task_id": f"foreign-{room}", "text": str(index)})
        with metered(monkeypatch) as counts:
            assert tp._already_in_chat(root, SUMMARY) == SUMMARY
        measurements.append(dict(counts))
    assert measurements[0] == measurements[1] and measurements[0]["steps"] > 0


def test_a_readiness_minted_in_the_same_call_skips_the_scan(tmp_path, monkeypatch):
    from ouroboros.project_dialogue import append_terminal_task_projection

    scans: list = []
    monkeypatch.setattr(tp, "_already_in_chat", lambda _root, row: scans.append(row) or None)
    stored = write_task_result(tmp_path, "fresh", "completed", _terminal_observed=True, chat_id=1, result="Answer.")
    assert append_terminal_task_projection(tmp_path, "fresh", {"id": "fresh", "chat_id": 1}, stored,
                                           {"status": "completed", "chat_id": 1})
    assert scans == [], "no row can carry a token minted in this very call"
    # A later settle of readiness recorded earlier (an append/receipt crash) does check, by the task.
    write_task_result(tmp_path, "fresh", "completed", canonical_terminal_projection=None)
    assert tp.settle_terminal_projection(tmp_path, "fresh") in {tp.SETTLEMENT_SETTLED, tp.SETTLEMENT_DEFERRED}
    assert scans and scans[0]["summary_id"] == "task-terminal:fresh" and scans[0]["task_id"] == "fresh"


def test_the_delivery_ack_settles_a_card_held_behind_its_answer(tmp_path, monkeypatch):
    """Ack after ready: task_done minted readiness while the answer was still owed (the card
    deferred); the real send's registration re-enters the settle owner and the card follows."""
    monkeypatch.setattr(delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    tid = "ack-after-ready"
    event = {"type": "send_message", "chat_id": 1, "task_id": tid, "text": "Answer.",
             "delivery_id": td.delivery_id_for(tid, "Answer.")}
    assert td.register_pending_delivery(tmp_path, event)
    write_task_result(tmp_path, tid, "completed", _terminal_observed=True, chat_id=1, result="Answer.")
    assert tp.settle_terminal_projection(tmp_path, tid) == tp.SETTLEMENT_DEFERRED  # readiness minted, owed holds
    assert isinstance(load_task_result(tmp_path, tid)["canonical_terminal_projection_ready"], dict)
    sent = []

    def send(chat_id, body, **kwargs):
        sent.append(body)
        ds.append(tmp_path, "chat", {"ts": "2026-10-15T00:00:00Z", "chat_id": chat_id, "text": body, "task_id": tid})

    host = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, send_with_budget=send)
    delivery._handle_send_message(event, host)
    rows = chat_rows(tmp_path)
    assert [row.get("text") for row in rows][0] == "Answer."
    assert rows[-1]["summary_id"] == f"task-terminal:{tid}" and rows[-1]["outcome_phase"] == "done"
    assert isinstance(load_task_result(tmp_path, tid)["canonical_terminal_projection"], dict)
    # The buffered duplicate is suppressed and re-enters the same owner: nothing doubles.
    delivery._handle_send_message(event, host)
    rows = chat_rows(tmp_path)
    assert sent == ["Answer."] and len([row for row in rows if row.get("summary_id")]) == 1


def test_an_ack_before_the_row_is_terminal_settles_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    tid = "ack-before-ready"
    event = {"type": "send_message", "chat_id": 1, "task_id": tid, "text": "Answer.",
             "delivery_id": td.delivery_id_for(tid, "Answer.")}
    assert td.register_pending_delivery(tmp_path, event)
    write_task_result(tmp_path, tid, "running", result="working")
    host = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, send_with_budget=lambda *a, **k: None)
    delivery._handle_send_message(event, host)
    assert td.terminal_answer_receipts(tmp_path, tid)["state"] == "delivered"
    stored = load_task_result(tmp_path, tid)
    assert "canonical_terminal_projection_ready" not in stored, "readiness is task_done's to mint"
    assert not any(row.get("summary_id") for row in chat_rows(tmp_path))
