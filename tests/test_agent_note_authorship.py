"""Permanent core checks for an agent note's recorded author (#3873), on harmless synthetic data.

These exercise the author area only: the command door's empty-cmd no-op, the CLI's literal note text
and naming-flag parse, owner-words exclusion, the plan/review author phrase and the mailbox event's
author fields. Every record is a fictional program's ordinary note, with no network or live installation.
The sibling tests/test_agent_notes.py owns durable receipt and completion behavior.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from ouroboros import cli, dialogue_provenance
from ouroboros.dialogue_provenance import (admission_note, agent_note_heading, agent_note_text, owner_speech,
                                           recorded_author, recorded_person, row_author, run_origin)
from ouroboros.gateway import agent_note, control
from ouroboros.owner_mailbox import PROVENANCE_AGENT_NOTE, deliver_task_message
from ouroboros.owner_words import objective_author_line, objective_author_text, owner_row

AUTHOR = recorded_author("Build agent")
NOTE_ROW = {"direction": "in", "source": "web", "user_id": 0, "chat_id": 1,
            "text": "  The example comparison is ready.\n", "declared_author": AUTHOR}


class _Request:
    """The minimal Starlette request the command door reads before any bridge or drive root."""

    def __init__(self, body):
        self._body = body
        self.app = SimpleNamespace(state=SimpleNamespace())

    async def json(self):
        return self._body


def _command(body):
    response = asyncio.run(control.api_command(_Request(body)))
    return response.status_code, json.loads(response.body)


# --- the command door: an explicitly empty cmd stays the old no-op; neither cmd nor note is refused ---

def test_explicitly_empty_cmd_is_a_200_noop():
    status, payload = _command({"cmd": ""})
    assert status == 200 and payload == {"status": "ok"}


def test_body_with_neither_cmd_nor_note_is_refused():
    for body in ({}, {"text": "orphan words"}, {"reply_to": ""}):
        status, payload = _command(body)
        assert status == 400 and payload.get("accepted") is False, body


def test_note_request_preserves_program_text_and_declared_signature():
    words = "  The example comparison is ready.\n  Its summary is in the project.\n"
    text, author = agent_note.note_request({"text": words, "author": {"kind": "agent", "label": "Build agent"}})
    assert text == words
    assert author == {"kind": "agent", "label": "Build agent", "basis": "self_declared"}

    row = {**NOTE_ROW, "role": "system", "system_type": "agent_note_received", "text": "Program message received.",
           "agent_note_text": text, "declared_author": author}
    assert agent_note_text(row) == words
    signature = row_author(row)
    assert signature["kind"] == "agent" and signature["basis"] == "self_declared"
    assert '"Build agent" (declared, not verified)' in signature["label"]
    assert not owner_speech(row, signature) and not owner_row(row)


def test_note_row_is_agent_not_owner():
    author = row_author(NOTE_ROW)
    assert author["kind"] == "agent" and "Build agent" in author["label"]
    assert owner_speech(NOTE_ROW, author) is False
    assert owner_row(NOTE_ROW) is False
    assert recorded_person(author) is False


def test_old_owner_row_without_declared_author_is_unchanged():
    old = {"direction": "in", "source": "web", "user_id": 1, "chat_id": 1, "text": "ordinary owner words"}
    assert row_author(old)["kind"] == "human"
    assert owner_row(old) is True  # a pre-feature row keeps its old reading; nothing re-attributes it


# --- the plan council / result review name the declared agent, not "not recorded" ---

def test_objective_author_names_the_declared_agent():
    task = {"type": "task", "metadata": {}, "admission_author": {"author": AUTHOR, "row_address": "row:1@t#abc"}}
    origin = run_origin(task)
    assert origin["owner_ingress"] is False and origin["actor_kind"] == "agent"
    line = objective_author_line(origin)
    assert "Build agent" in line and "not my human's words" in line and "not recorded" not in line
    ctx = SimpleNamespace(current_task_type="task", task_metadata={}, _admission_author=admission_note(task))
    assert objective_author_text(ctx) == line


def test_objective_author_keeps_owner_and_presence_readings():
    owner_ctx = SimpleNamespace(current_task_type="task", _admission_author=None,
                                task_metadata={"origin_message_ref": {"row_sha256": "abc", "ts": "2026-11-23T00:00:00Z"}})
    assert "my human" in objective_author_text(owner_ctx)
    presence = objective_author_line({"task_type": "presence", "text_author": "Mira", "actor_kind": "user"})
    assert presence == "a message from Mira in a connected place"


# --- the CLI: literal note text, a naming flag parsed wherever it sits before --, never owner words by mistake ---

class _Client:
    def __init__(self):
        self.posts = []

    def request(self, method, path, body=None, **kwargs):
        self.posts.append((method, path, body))
        return {"status": "ok", "accepted": True, "row_address": "row:1@t#abc", "author": {"kind": "agent", "label": "x"}}


def _send(monkeypatch, *, from_agent, text):
    client = _Client()
    monkeypatch.setattr(cli, "_client", lambda args: client)
    monkeypatch.setattr(cli, "_print_json", lambda data: None)
    flags = [] if from_agent is None else ["--from-agent", from_agent]
    args = cli.build_parser().parse_args(["chat", "send", *flags, *text])
    code = args.func(args)
    assert code == 0
    assert len(client.posts) == 1
    return client.posts[0][2]


def test_cli_note_preserves_literal_whitespace(monkeypatch):
    body = _send(monkeypatch, from_agent="Build agent", text=["--", "  hello  world  "])
    assert body == {"text": "  hello  world  ", "author": {"kind": "agent", "label": "Build agent"}}
    body = _send(monkeypatch, from_agent="Build agent", text=["--", "The docs mention", "--from-agent"])
    assert body == {"text": "The docs mention --from-agent", "author": {"kind": "agent", "label": "Build agent"}}


def test_cli_misplaced_naming_flag_names_the_agent_not_the_owner(monkeypatch):
    body = _send(monkeypatch, from_agent=None, text=["hello", "--from-agent", "Bot", "world"])
    assert body == {"text": "hello world", "author": {"kind": "agent", "label": "Bot"}}
    body = _send(monkeypatch, from_agent=None, text=["hello", "--from-agent=Bot"])
    assert body["author"]["label"] == "Bot" and body["text"] == "hello"


def test_cli_words_after_double_dash_are_literal_owner_text(monkeypatch):
    body = _send(monkeypatch, from_agent=None, text=["hello", "--", "--from-agent", "Bot"])
    # The `--` a started ordinary 7.x text already contains stays in it; only the words after it are
    # unscanned, so a naming flag there stays the owner's literal words.
    assert body == {"cmd": "hello -- --from-agent Bot"}
    body = _send(monkeypatch, from_agent=None, text=["hello", "world"])
    assert body == {"cmd": "hello world"}


def test_cli_naming_errors_never_become_owner_words(monkeypatch):
    for from_agent, text in ((None, ["hello", "--from-agent"]),          # missing value
                             (None, ["hello", "--from-agent="]),         # empty value
                             ("A", ["hello", "--from-agent", "B"]),      # named twice
                             (None, ["--from-agent", "A", "--from-agent", "B"])):
        with pytest.raises(cli.CLIError):
            _send(monkeypatch, from_agent=from_agent, text=text)


# --- the mailbox event carries the author; the transcript names the declared agent ---

def test_injected_note_event_carries_author_and_prefix():
    entry = {"provenance": PROVENANCE_AGENT_NOTE, "source_task_id": "", "text": "the comparison is ready",
             "ts": "2026-11-23T00:00:00Z", "msg_id": "m1",
             "agent_note": {"author": AUTHOR, "row_address": "row:1@t#abc"}}
    events, appended = [], []
    queue = SimpleNamespace(put_nowait=events.append)
    deliver_task_message(entry, "receiver-1", queue, appended.append)
    assert events[0]["provenance"] == PROVENANCE_AGENT_NOTE
    assert events[0]["agent_note"]["author"]["label"] == "Build agent"
    assert events[0]["agent_note"]["row_address"] == "row:1@t#abc"
    assert "Build agent" in appended[0] and "not my human's words" in appended[0]
    assert agent_note_heading(AUTHOR, ts="2026-11-23T00:00:00Z", address="row:1@t#abc") in appended[0]


def test_system_message_event_has_no_agent_note():
    entry = {"provenance": "system", "source_task_id": "", "text": "a silence fact", "msg_id": "m2"}
    events = []
    deliver_task_message(entry, "receiver-1", SimpleNamespace(put_nowait=events.append), lambda _text: None)
    assert "agent_note" not in events[0]


# --- the shared projection: literal body as program data, fail-closed, never the receipt fact ---

def test_agent_note_text_projects_literal_body_fail_closed():
    canonical = {"direction": "in", "role": "system", "system_type": "agent_note_received",
                 "text": "Program message received.", "agent_note_text": "  literal body \n",
                 "declared_author": AUTHOR}
    assert agent_note_text(canonical) == "  literal body \n"
    assert agent_note_text({**canonical, "agent_note_text": None}) == ""  # fail-closed, not the receipt fact
    assert agent_note_text({k: v for k, v in canonical.items() if k != "agent_note_text"}) == ""
    assert agent_note_text(NOTE_ROW) == NOTE_ROW["text"]  # pre-release author format: body in text
    assert agent_note_text({"direction": "in", "text": "owner words", "user_id": 1}) == ""  # not a note
    # every reader projects the literal body as program data under the declared agent, never the owner
    assert dialogue_provenance.dialogue_text(canonical).startswith("  literal body \n")
    assert row_author(canonical)["kind"] == "agent"
    assert owner_speech(canonical, row_author(canonical)) is False
    assert owner_row(canonical) is False  # the typed system_type excludes it from the owner corpus


def _program_row(*, late=False):
    row = {**NOTE_ROW, "role": "system", "system_type": "agent_note_received", "text": "Program message received.",
           "agent_note_text": "The example table has six rows.", "ts": "2026-11-23T00:02:00Z"}
    return {**row, "late_report_sha256": "a" * 64} if late else row


def test_unread_mail_keeps_the_program_signature_and_late_message_locator():
    from ouroboros.task_custody import unread_mail_notice

    signature = {"author": AUTHOR, "row_address": "row:1@2026-11-23T00:02:00Z#abc"}
    ordinary = {"kind": "task_message", "provenance": PROVENANCE_AGENT_NOTE, "agent_note": signature,
                "text": "The table is ready."}
    late = {**ordinary, "agent_note": {**signature, "late_report_ref": "example-letter"},
            "text": "The example table has six rows."}
    human = {"kind": "owner_text", "text": "Please check the table."}
    custody = {"rows": [json.dumps(row) for row in (human, ordinary, late)], "read_complete": True}
    notice = unread_mail_notice(custody)
    assert 'from owner: Please check the table.' in notice
    assert 'agent "Build agent" (declared, not verified)' in notice and ordinary['text'] in notice
    assert signature["row_address"] in notice and "Late program message retained" in notice
    assert late['text'] not in notice and json.loads(custody["rows"][2]) == late


def test_automatic_memory_views_show_ordinary_program_words_but_only_locate_a_late_letter(tmp_path):
    from ouroboros.chat_chain import format_address, row_address
    from ouroboros.consciousness_wake import _event_line, _row_kind
    from ouroboros.memory_fallback import _QUOTES, _rows

    rooms = SimpleNamespace(label=lambda _row: "Main")
    assert "allowed: agent," in _QUOTES
    for late in (False, True):
        row = _program_row(late=late)
        address = row_address(row)
        assert _row_kind(row) == "agent_note"
        wake = _event_line("agent_note", row, {}, rooms, {}, 0, when="now")
        [light] = _rows(tmp_path, [(address, row)], {address["row_sha256"]: 0})
        for projection in (wake, light.line):
            assert format_address(address) in projection
            if late:
                assert row["agent_note_text"] not in projection and "Late program message retained" in projection
            else:
                assert row["agent_note_text"] in projection and "Build agent" in projection
        assert light.address_only is late
        explicit = dialogue_provenance.render_memory_row(address, row, author=row_author(row))
        assert row["agent_note_text"] in explicit


def test_first_hour_does_not_count_program_mail_as_owner_speech(tmp_path):
    from ouroboros import dialogue_store

    ready = _program_row()
    (tmp_path / "state").mkdir()
    dialogue_store.append(tmp_path, "chat", ready)
    dialogue_store.append(tmp_path, "chat", {**NOTE_ROW, "ts": "2026-11-23T00:03:00Z"})  # retained author format
    assert dialogue_store.owner_exchange(tmp_path) == (False, False)
    dialogue_store.append(tmp_path, "chat", {"direction": "in", "chat_id": 1, "user_id": 1, "source": "web",
                                            "ts": "2026-11-23T00:04:00Z", "text": "Could you check it?"})
    assert dialogue_store.owner_exchange(tmp_path) == (True, False)