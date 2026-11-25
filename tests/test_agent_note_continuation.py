"""The Continue and Restart of a root a program's note opened (#3873): one typed machine opening.

The predecessor's own record carries the note's ``admission_author`` (its declared author and the exact
address of its canonical row). A Continue of that root reads the machine's words through the shared
reader of the machine note (``dialogue_provenance.agent_note_text``) at that address — never the neutral
receipt fallback, never my human's words — and seeds no owner corpus from them. A saved queued row
returns from a stop door with its conversation carrier (``saved_carrier``): the note's admission is as
positive as the owner door's. Everything here runs on the real local store in ``tmp_path``: no sockets,
no DNS, no models, no letters.
"""
from __future__ import annotations

import pytest

from tests._budget_pause_exact_helpers import _install_queue

NONCE = "press-0002-abcdef"


def _note_row(root, words: str, client_message_id: str = "agent-note-42", *, late: bool = False) -> dict:
    """The canonical row of a program's note through the real ingress writer (``message_ingress``)."""
    from ouroboros.dialogue_provenance import recorded_author
    from supervisor.message_ingress import log_chat

    row = log_chat("in", 1, 0, words, drive_root=root, source="web", client_message_id=client_message_id,
                   message_meta={"declared_author": recorded_author("Build agent"),
                                 **({"late_report_sha256": "c" * 64} if late else {})})
    assert row and row.get("system_type") == "agent_note_received"
    assert row.get("agent_note_text") == words, "the machine's words live in agent_note_text, not text"
    assert row.get("text") != words, "text is the neutral receipt fact old readers show"
    return dict(row)


def _admission_author(row: dict) -> dict:
    """The host-only opening record the routing stamps for the turn a note opens (``_agent_note``)."""
    from ouroboros.chat_chain import format_address, row_address

    from ouroboros.dialogue_provenance import declared_author

    author = declared_author(row)
    assert author, "the canonical note row carries its declared author"
    return {"author": dict(author), "row_address": format_address(row_address(row))}


@pytest.mark.parametrize("late", [False, True])
def test_continue_admits_a_root_a_note_opened_and_keeps_the_machine_words_apart(tmp_path, monkeypatch, late):
    """The button promises a Continue of a note-opened root and the admission delivers it: a typed
    ``note`` opening (author, exact address, the machine's literal words), no owner corpus seeded."""
    from supervisor.continuation_admission import admit_continuation
    from ouroboros.task_results import STATUS_CANCELLED, load_task_result, write_task_result

    _install_queue(tmp_path, monkeypatch)
    words = "  The example comparison is ready.\n  Проверка: всё готово.\n"
    row = _note_row(tmp_path, words, late=late)
    note = _admission_author(row)
    write_task_result(tmp_path, "pred-1", STATUS_CANCELLED, chat_id=1, root_task_id="pred-1",
                      result="interrupted", admission_author=note,
                      cancel_origin={"source": "snapshot_restore", "reason": "server_shutdown"})

    outcome = admit_continuation("pred-1", action_nonce=NONCE)

    assert outcome["ok"], outcome
    successor = outcome["successor_task_id"]
    assert successor.startswith("pred-1-c")
    from supervisor import queue as q
    from supervisor import workers

    pending = next(task for task in workers.PENDING if task["id"] == successor)
    metadata = pending["metadata"]
    assert metadata["continuation"]["owner_sources"]["note"] == {**note, "source": "agent_note",
        "task_id": "pred-1", "words": "" if late else words, **({"late_result": True} if late else {})}
    assert metadata["continuation"]["owner_sources"]["original"] is None
    assert "admission_author" not in metadata, "a system record never becomes the owner's words"
    assert note["row_address"] in pending["text"]
    assert (words in pending["text"]) is (not late)
    assert ('"Build agent"' in pending["text"]) is (not late)
    assert "Program message received." not in pending["text"], "the neutral fallback never stands for the words"
    corpus = [(r["source"], r["content"]) for r in metadata.get("owner_corpus") or []]
    assert corpus == [], "a machine note seeds no owner corpus row"
    claim = load_task_result(tmp_path, "pred-1")["continued_by"]
    assert claim["successor_task_id"] == successor and claim["state"] == "admitted"


def test_continue_refuses_when_the_notes_own_row_cannot_be_read(tmp_path, monkeypatch):
    """A typed opening whose exact row no longer resolves is a named gap (``agent_note_unreadable``),
    never the neutral fallback or a reconstruction: admission refuses with nothing written."""
    from supervisor.continuation_admission import admit_continuation
    from ouroboros.task_results import STATUS_CANCELLED, load_task_result, write_task_result

    _install_queue(tmp_path, monkeypatch)
    words = "The example comparison is ready."
    row = _note_row(tmp_path, words)
    note = _admission_author(row)
    note["row_address"] = "row:1@2020-01-01T00:00:00+00:00#" + "0" * 12  # no row ever resolved here
    write_task_result(tmp_path, "pred-1", STATUS_CANCELLED, chat_id=1, root_task_id="pred-1",
                      result="interrupted", admission_author=note,
                      cancel_origin={"source": "snapshot_restore", "reason": "server_shutdown"})

    outcome = admit_continuation("pred-1", action_nonce=NONCE)

    assert outcome == {"ok": False, "error": "owner_source_missing", "gaps": ["agent_note_unreadable"]}
    assert "continued_by" not in load_task_result(tmp_path, "pred-1")


def test_a_saved_note_opened_row_returns_from_a_restart_with_its_conversation_carrier(tmp_path):
    """``saved_carrier``: a queued row a note opened (no owner-door ingress) keeps its conversation-lane
    carrier across a stop door — the machine note's admission is as positive as the owner door's."""
    from ouroboros.task_carrier import EXECUTOR_WORKER, LANE_CONVERSATION
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from supervisor.conversation_lane import saved_carrier

    words = "The example comparison is ready."
    row = _note_row(tmp_path, words)
    note = _admission_author(row)
    tid = "note-turn"
    owner = {"kind": "direct", "lane": LANE_CONVERSATION, "executor": EXECUTOR_WORKER,
             "task_id": tid, "task_attempt": 1}
    write_task_result(tmp_path, tid, STATUS_RUNNING, execution_owner=owner, task_attempt=1,
                      admitted_dispatch_attempt=1, chat_id=1, root_task_id=tid)
    saved = {"id": tid, "chat_id": 1, "_is_direct_chat": True, "_attempt": 1, "admission_author": note}

    carrier = saved_carrier(saved, tmp_path)

    assert carrier == {"_execution_owner": dict(owner)}