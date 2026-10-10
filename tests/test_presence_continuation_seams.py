"""#1536 repair seams below the full turn: gate stop order, reentry source, projection, context.

Each test drives the real owner (gate files, chat chain, projection lock, task-result
store) in a temporary data root; nothing is mocked except where a failure is injected.
"""

from __future__ import annotations

import contextvars
import json
import os
import pathlib
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import presence_continuation as pc
from ouroboros.presence_runner import (
    PresenceTurnGate, PresenceTurnLease, _build_task, _read_previous_turn, _write_previous_turn,
    presence_turn_replay, record_continuing_turn,
)
from ouroboros.task_results import write_task_result
from tests.test_presence_runner import _admission

KEY = "telegram:bot-1:room-1:topic-1"
OTHER = "telegram:bot-1:room-1:topic-2"


def _row(key, text, **extra):
    return {"ts": "2026-10-06T10:00:00+00:00", "chat_id": 7, "direction": "in", "text": text,
            "client_message_id": f"telegram:bot-1:{abs(hash(text)) % 1000}", "sender_label": "alex",
            "task_id": "turn-x", "transport": {"conversation_key": key}, **extra}


def _append(root, *rows, raw: bytes = b""):
    path = pathlib.Path(root) / "logs" / "chat.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n")
        handle.write(raw)
    return path


def _binding(root, cursor, task_id="turn-me"):
    event = SimpleNamespace(conversation_key=KEY, provider="telegram", account_id="bot-1",
                            conversation_id="room-1", thread_id="topic-1", continuation_version=1,
                            delivery_reporting_version=0)
    return pc.ReviewWaitBinding(lease=None, drive_root=pathlib.Path(root), task_id=task_id, identity="i",
                                event=event, cursor=cursor)


# --- finding 2: no stopped caller is ever handed the conversation ---------------------------------

def test_a_stop_before_an_immediately_free_acquisition_takes_nothing(tmp_path):
    gate = PresenceTurnGate(1, state_root=tmp_path / "state")
    asked = []
    assert gate.acquire_resources(KEY, lambda: asked.append(1) or True) is None
    assert asked == [1]
    gate.acquire(KEY).release()  # nothing was left held: the conversation and slot are free at once


def test_a_stop_landing_as_the_acquisition_succeeds_returns_what_was_taken(tmp_path):
    gate = PresenceTurnGate(1, state_root=tmp_path / "state")
    answers = iter([False, True])  # asked before the first attempt, then right after it succeeded
    assert gate.acquire_resources(KEY, lambda: next(answers)) is None
    taken = threading.Event()
    threading.Thread(target=lambda: (gate.acquire(KEY).release(), taken.set()), daemon=True).start()
    assert taken.wait(5), "the resources taken as the stop landed were not returned"


def test_a_lent_lease_reacquires_only_without_a_stop(tmp_path):
    gate = PresenceTurnGate(1, state_root=tmp_path / "state")
    lease = gate.acquire(KEY)
    assert lease.lendable() and lease.lend() and not lease.held()
    assert lease.reacquire(lambda: True) is False and not lease.held()
    gate.acquire(KEY).release()  # a refused reacquisition holds nothing
    assert lease.reacquire(lambda: False) is True and lease.held()
    lease.release()
    assert not lease.lendable() and lease.reacquire(lambda: False) is False


def test_a_queued_reacquisition_paces_its_stop_reads_and_still_ends(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.presence_runner._GATE_STOP_POLL_SEC", 0.2)
    gate = PresenceTurnGate(1, state_root=tmp_path / "state")
    holder = gate.acquire(KEY)
    reads, stop_at = [], time.monotonic() + 0.7
    assert gate.acquire_resources(KEY, lambda: reads.append(1) or time.monotonic() >= stop_at) is None
    assert 3 <= len(reads) <= 8  # about one control read per pacing interval, not per 50 ms poll
    holder.release()


def test_an_ungated_lease_is_not_lendable():
    from contextlib import ExitStack

    lease = PresenceTurnLease(KEY, ExitStack())
    assert not lease.lendable() and not lease.lend() and lease.held()


# --- finding 3: reentry follows the chain from its cursor and names every gap ----------------------

def test_rows_after_the_cursor_follow_a_rotation_into_the_archive(tmp_path):
    from supervisor.state import rotate_chat_log_if_needed

    _append(tmp_path, _row(KEY, "before the yield"))
    cursor = pc._chat_cursor(tmp_path)
    _append(tmp_path, _row(KEY, "first after"), _row(OTHER, "another thread"))
    rotate_chat_log_if_needed(tmp_path, max_bytes=1)
    # The rotator renames the generation verbatim into archive/ and starts an empty live file.
    assert (tmp_path / "logs" / "chat.jsonl").stat().st_size == 0 and list((tmp_path / "archive").glob("chat_*.jsonl"))
    _append(tmp_path, _row(KEY, "second after"))
    rows, gaps = pc._conversation_rows_since(tmp_path, cursor, KEY)
    assert [row["text"] for row in rows] == ["first after", "second after"] and gaps == []


def test_a_log_absent_at_the_yield_reads_every_later_generation(tmp_path):
    from supervisor.state import rotate_chat_log_if_needed

    _append(tmp_path, _row(KEY, "old archived"))
    rotate_chat_log_if_needed(tmp_path, max_bytes=1)
    time.sleep(1.1)  # archive names carry a one-second UTC stamp
    cursor = pc._chat_cursor(tmp_path)
    assert cursor["offset"] == 0 and cursor["after_archive"]
    _append(tmp_path, _row(KEY, "new one"))
    rotate_chat_log_if_needed(tmp_path, max_bytes=1)
    _append(tmp_path, _row(KEY, "new two"))
    rows, gaps = pc._conversation_rows_since(tmp_path, cursor, KEY)
    assert [row["text"] for row in rows] == ["new one", "new two"] and gaps == []


def test_unreadable_lines_are_gaps_that_say_whether_they_name_this_conversation(tmp_path):
    _append(tmp_path, _row(KEY, "before"))
    cursor = pc._chat_cursor(tmp_path)
    _append(tmp_path, _row(KEY, "kept"), raw=(b'{"text": "cut", "transport": {"conversation_key": "' + KEY.encode()
                                               + b'"\n' + b"\xff\xfe not utf-8\n" + b"[1, 2]\n"))
    _append(tmp_path, _row(KEY, "also kept"), raw=b'{"partial": ')
    rows, gaps = pc._conversation_rows_since(tmp_path, cursor, KEY)
    assert [row["text"] for row in rows] == ["kept", "also kept"]
    assert [(gap["kind"], gap["mentions_conversation"]) for gap in gaps] == [
        ("jsonl_malformed", True), ("jsonl_decode_error", False), ("jsonl_non_object", False),
        ("trailing_row_incomplete", False)]
    assert all(gap["path"] == "logs/chat.jsonl" and gap["offset"] >= cursor["offset"] for gap in gaps)


def test_a_same_inode_truncation_below_the_cursor_is_a_gap_not_silence(tmp_path):
    # Leave ample size difference: event IDs contain process-randomized hash
    # digits, so replacing "second" with "short" could keep the byte size equal.
    path = _append(tmp_path, _row(KEY, "first line stays"), _row(KEY, "a much longer second line to remove"))
    cursor = pc._chat_cursor(tmp_path)
    inode = os.stat(path).st_ino
    with open(path, "r+b") as handle:
        handle.truncate(len(json.dumps(_row(KEY, "first line stays")).encode()) + 1)
    _append(tmp_path, _row(KEY, "short"))
    assert os.stat(path).st_ino == inode and os.stat(path).st_size < cursor["offset"]
    rows, gaps = pc._conversation_rows_since(tmp_path, cursor, KEY)
    assert rows == [] and [gap["kind"] for gap in gaps] == ["generation_truncated_below_cursor"]


def test_a_generation_rewritten_at_the_cursor_or_replaced_is_a_gap(tmp_path):
    path = _append(tmp_path, _row(KEY, "first"), _row(KEY, "second"))
    cursor = pc._chat_cursor(tmp_path)
    data = path.read_bytes()
    path.write_bytes(data[:-1] + b" padding to move every later byte\n" + data[-1:])
    rows, gaps = pc._conversation_rows_since(tmp_path, cursor, KEY)
    assert rows == [] and gaps[0]["kind"] == "generation_rewritten_at_cursor"
    path.write_bytes(json.dumps(_row(KEY, "a different first line")).encode() + b"\n")
    assert pc._conversation_rows_since(tmp_path, cursor, KEY) == ([], [{"kind": "cursor_generation_missing"}])


def test_the_scan_budget_is_a_disclosed_gap(tmp_path, monkeypatch):
    _append(tmp_path, _row(KEY, "before"))
    cursor = pc._chat_cursor(tmp_path)
    _append(tmp_path, *[_row(OTHER, "x" * 200) for _ in range(20)], _row(KEY, "too far"))
    monkeypatch.setattr(pc, "_REENTRY_SCAN_BYTES", 1000)
    rows, gaps = pc._conversation_rows_since(tmp_path, cursor, KEY)
    assert rows == [] and gaps[-1]["kind"] == "scan_budget_exhausted"


def test_the_note_carries_whole_facts_and_names_what_it_cannot_carry(tmp_path, monkeypatch):
    from ouroboros.chat_chain import resolve_row

    _append(tmp_path, _row(KEY, "before"))
    cursor = pc._chat_cursor(tmp_path)
    long_text = "line one\nline two " + "y" * 900  # longer than the old 600-character clip
    words = [f"message {index}" for index in range(30)]  # more than the old 20-row clip
    _append(tmp_path, *[_row(KEY, text, ts=f"2026-10-06T10:00:{index:02d}+00:00") for index, text in enumerate(words)],
            _row(KEY, long_text, ts="2026-10-06T10:01:00+00:00"))
    ctx = SimpleNamespace(task_contract={"capability_ceiling": None})
    note = pc.reentry_note(_binding(tmp_path, cursor), ctx)
    assert all(json.dumps(text) in note for text in words) and json.dumps(long_text) in note
    assert "Coverage: every row of this conversation's canonical log since you yielded is listed." in note
    monkeypatch.setattr(pc, "_REENTRY_NOTE_CHARS", 2000)
    note = pc.reentry_note(_binding(tmp_path, cursor), ctx)
    assert json.dumps(long_text) in note and json.dumps(words[0]) not in note
    first, last = note.split("are not repeated here: ")[1].split(")")[0].split(" through ")
    assert resolve_row(tmp_path, first)[0]["text"] == words[0]  # the canonical reader resolves each omission
    assert resolve_row(tmp_path, last)[1]["status"] == "ok"
    assert 'get_task_result({"task_id": "turn-me", "presence_reentry_sha256":' in note


def test_the_note_never_names_a_reader_the_turn_does_not_hold(tmp_path):
    from ouroboros.presence_authority import presence_ceiling_payload

    _append(tmp_path, _row(KEY, "before"))
    cursor = pc._chat_cursor(tmp_path)
    _append(tmp_path, raw=b"{broken\n")
    with_reader = _admission().capability_ceiling
    without = type(with_reader)(**{**with_reader.__dict__, "tool_grants": ()})
    for ceiling, expected in ((with_reader, "chat_history("), (without, "No history reader is among this turn's tools")):
        ctx = SimpleNamespace(task_contract={"capability_ceiling": presence_ceiling_payload(ceiling)})
        note = pc.reentry_note(_binding(tmp_path, cursor), ctx)
        assert "a malformed line at logs/chat.jsonl byte" in note and expected in note


# --- finding 1: the projection a successor reads is durable, and a lost pointer is rebuilt ---------

def test_later_park_retains_child_without_rewriting_initial_or_crossing_source_identity(tmp_path):
    from dataclasses import replace
    from ouroboros.task_results import load_task_result
    from tests.test_presence_continuation import event as turn_event

    binding = pc.ReviewWaitBinding(None, tmp_path, "author", "source-identity", turn_event())
    write_task_result(tmp_path, "author", "running", metadata={"presence_event_identity": binding.identity})
    first = pc._persist(binding, {}, "", "panel-1", "first-park")
    second = pc._persist(binding, {}, "work-late", "panel-2", "second-park")
    assert second["initial"] == first["initial"] and first["initial"]["work_ref"] == ""
    assert second["child_work_ref"] == "work-late"
    # A later empty observation or host terminal must not withdraw admitted independent work.
    third = pc._persist(binding, {}, "", "panel-3", "third-park")
    assert third["child_work_ref"] == "work-late" and third["initial"] == first["initial"]
    with pytest.raises(ValueError, match="this event's RUNNING row"):
        pc._persist(replace(binding, identity="another-source"), {}, "unrelated-work", "panel-x", "bad-park")
    assert load_task_result(tmp_path, "author")["presence_continuation"] == third


def test_later_child_must_read_back_before_the_park_can_publish(tmp_path, monkeypatch):
    from ouroboros import task_results
    from tests.test_presence_continuation import event as turn_event

    binding = pc.ReviewWaitBinding(None, tmp_path, "author", "source-identity", turn_event())
    write_task_result(tmp_path, "author", "running", metadata={"presence_event_identity": binding.identity})
    pc._persist(binding, {}, "", "panel-1", "first-park")
    load = task_results.load_task_result

    def missing_child(*args, **kwargs):
        stored = load(*args, **kwargs)
        stored["presence_continuation"].pop("child_work_ref", None)
        return stored

    monkeypatch.setattr(task_results, "load_task_result", missing_child)
    with pytest.raises(ValueError, match="did not read back"):
        pc._persist(binding, {}, "work-late", "panel-2", "second-park")


@pytest.mark.parametrize("failure", ["raises", "lost"])
def test_a_yielding_authors_projection_write_is_strict(tmp_path, monkeypatch, failure):
    def broken(path, value):
        if failure == "raises":
            raise OSError("disk full")

    monkeypatch.setattr("ouroboros.presence_runner.atomic_write_json", broken)
    with pytest.raises(OSError):
        record_continuing_turn(tmp_path, KEY, "turn-1", outcome="deferred", message="", work_ref="",
                               lent_at="2026-10-06T10:00:00+00:00")
    # The ordinary terminal write keeps its best-effort contract.
    _write_previous_turn(tmp_path, KEY, "turn-1", outcome="message", message="hi", sends=[], work_ref="",
                         finished_at="2026-10-06T10:00:00+00:00", delivery="unknown")


def test_a_v1_replay_rebuilds_a_lost_pointer_before_returning_the_envelope(tmp_path):
    from ouroboros.presence_runner import presence_event_identity, presence_turn_task_id
    from tests.test_presence_continuation import event as turn_event

    admission, event = _admission(), turn_event()
    task_id = presence_turn_task_id(admission.binding_id, event.source_event_id)
    identity = presence_event_identity(admission.binding_id, event)
    task = _build_task(admission, event, drive_root=tmp_path, staged_files=(), physical_task_id=task_id)
    envelope = {"status": "continuing", "outcome": "message", "text": "Released.", "output_ref": "presence-output-1",
                "work_ref": ""}
    record = {"version": 1, "continuation_ref": task_id, "event_identity": identity, "initial": envelope,
              "outputs": [{"output_ref": "presence-output-1", "outcome": "message", "text": "Released."}],
              "source_event_id": event.source_event_id, "conversation_key": KEY, "lent_at": "2026-10-06T10:00:00+00:00"}
    write_task_result(tmp_path, task_id, "running", metadata=task["metadata"], source="presence",
                      presence_continuation=record)
    record_continuing_turn(tmp_path, KEY, task_id, outcome="message", message="Released.", work_ref="",
                           lent_at=record["lent_at"])
    # The author re-selected its released output at its terminal (silent), and its pointer write was lost.
    metadata = {**task["metadata"], "presence_outcome": "silent", "presence_result_text": ""}
    write_task_result(tmp_path, task_id, "completed", metadata=metadata, terminal_origin="model_final",
                      result="Released.")
    assert _read_previous_turn(tmp_path, KEY)["continuing"] is True  # the yield, not the terminal
    replayed = presence_turn_replay(tmp_path, task_id, KEY, identity, 1)
    assert (replayed.status, replayed.text, replayed.output_ref) == ("continuing", "Released.", "presence-output-1")
    pointer = _read_previous_turn(tmp_path, KEY)
    assert (pointer["task_id"], pointer["message"], pointer.get("continuing"), pointer.get("open_turns")) == (
        task_id, "Released.", None, None)
    (tmp_path / "state" / "presence_turn_gate" / pathlib.Path(
        next(p.name for p in (tmp_path / "state" / "presence_turn_gate").glob("last-*.json")))).unlink()
    assert presence_turn_replay(tmp_path, task_id, KEY, identity, 1) == replayed
    assert _read_previous_turn(tmp_path, KEY)["task_id"] == task_id  # a missing pointer is rebuilt too


@pytest.mark.parametrize("successor", [False, True])
def test_a_second_park_and_late_terminal_keep_the_newest_turn_pointer(tmp_path, monkeypatch, successor):
    """Real runner, cap-one lease, continuation store and context; only the author is scripted."""
    from ouroboros import presence_runner as runner
    from ouroboros.presence_context import build_presence_context_section
    from ouroboros.task_results import load_task_result
    from tests.test_presence_continuation import event as turn_event

    first, second = turn_event(42), turn_event(43)
    admission = _admission()
    first_id = runner.presence_turn_task_id(admission.binding_id, first.source_event_id)
    second_id = runner.presence_turn_task_id(admission.binding_id, second.source_event_id)
    gate = PresenceTurnGate(1, state_root=tmp_path / "state")
    clock = {"now": "2026-10-06T10:00:00+00:00"}
    monkeypatch.setattr(pc, "utc_now_iso", lambda: clock["now"])
    monkeypatch.setattr(runner, "utc_now_iso", lambda: clock["now"])

    def run(event):
        return runner.run_presence_turn(admission=admission, event=event, repo_dir=tmp_path,
                                        drive_root=tmp_path, gate=gate, agent_factory=lambda **_kw: Author())

    class Author:
        def handle_task(self, task):
            if task["id"] == first_id:
                binding = self.review_wait_callback.args[0]
                ctx = SimpleNamespace(task_metadata=task["metadata"])
                assert pc._yield_conversation(binding, ctx, {"review_binding": "panel-1"}) == (True, "")
                assert not binding.lease.held()
                if successor:
                    clock["now"] = "2026-10-06T10:01:00+00:00"
                    assert run(second).text == "Newer reply"
                assert binding.lease.reacquire(lambda: False)
                clock["now"] = "2026-10-06T10:02:00+00:00"
                assert pc._yield_conversation(binding, ctx, {"review_binding": "panel-2"}) == (True, "")
                continued = load_task_result(tmp_path, first_id)["presence_continuation"]
                assert (continued["first_lent_at"], continued["lent_at"]) == (
                    "2026-10-06T10:00:00+00:00", clock["now"])
                pointer = _read_previous_turn(tmp_path, KEY)
                assert pointer["task_id"] == (second_id if successor else first_id)
                assert pointer["open_turns"] == [first_id]
                # This is the actual projection/renderer a later incoming or proactive turn consumes.
                next_task = _build_task(admission, turn_event(44), drive_root=tmp_path, staged_files=(),
                                        physical_task_id="turn-next")
                presence = next_task["metadata"]["presence"]
                assert presence["previous_turn"]["task_id"] == pointer["task_id"]
                assert presence["open_turns"] == [{"task_id": first_id, "status": "running"}]
                section = build_presence_context_section(tmp_path, presence, "turn-next")
                assert first_id in section and (not successor or "Newer reply" in section)
                assert binding.lease.reacquire(lambda: False)
                clock["now"] = "2026-10-06T10:03:00+00:00"
                reply = "Older author finished"
            else:
                assert task["metadata"]["presence"]["open_turns"] == [
                    {"task_id": first_id, "status": "running"}]
                reply = "Newer reply"
            metadata = {**task["metadata"], "presence_outcome": "message", "presence_result_text": reply}
            write_task_result(tmp_path, task["id"], "completed", metadata=metadata, result=reply,
                              terminal_origin="model_final")
            return [{"type": "presence_result", "outcome": "message", "text": reply, "work_ref": ""}]

    assert run(first).text == "Older author finished"
    pointer = _read_previous_turn(tmp_path, KEY)
    assert pointer["task_id"] == (second_id if successor else first_id)
    assert pointer["message"] == ("Newer reply" if successor else "Older author finished")
    assert "open_turns" not in pointer and "continuing" not in pointer
    # A source-event replay also must not move the pointer back to the returning author.
    replayed = runner.presence_turn_replay(
        tmp_path, first_id, KEY, runner.presence_event_identity(admission.binding_id, first), 1)
    assert replayed.status == "continuing" and replayed.text == ""
    assert _read_previous_turn(tmp_path, KEY) == pointer


def test_a_dead_yielded_author_is_not_presented_as_still_responsible(tmp_path):
    from ouroboros.presence_context import build_presence_context_section
    from tests.test_presence_continuation import event as turn_event

    record_continuing_turn(tmp_path, KEY, "turn-gone", outcome="deferred", message="", work_ref="",
                           lent_at="2026-10-06T10:00:00+00:00")
    write_task_result(tmp_path, "turn-gone", "failed", reason_code="orphaned_running_after_worker_restart",
                      status_reconciled_from="running")
    task = _build_task(_admission(), turn_event(43), drive_root=tmp_path, staged_files=(), physical_task_id="turn-next")
    presence = task["metadata"]["presence"]
    assert presence["previous_turn"]["author_status"] == "failed"
    assert presence["open_turns"] == [{"task_id": "turn-gone", "status": "failed"}]
    section = build_presence_context_section(tmp_path, presence, "turn-next")
    assert "no longer live (task status failed)" in section and "still the responsible author" not in section


# --- finding 4: an initiated turn starts from its own context ---------------------------------------

def test_an_initiated_turn_inherits_settings_and_calendar_but_no_call_state():
    from ouroboros import send_clock, usage_accounting
    from ouroboros.model_wait import _LOGICAL, calendar_scope, execution_deadline_scope, independent_turn_context
    from ouroboros.settings_integrity import _TASK_SETTINGS
    from ouroboros.tools import tool_result

    caller = {variable: object() for variable in (usage_accounting._CURRENT_SCOPE, usage_accounting._PHYSICAL_CONTEXT,
                                                  usage_accounting._LAST_PHYSICAL_ATTEMPT, send_clock._SCOPE,
                                                  tool_result._TOOL_RESULT_STATE, _TASK_SETTINGS)}

    def initiate():
        for variable, value in caller.items():
            variable.set(value)
        with calendar_scope("2099-01-01T00:00:00+00:00"), execution_deadline_scope(time.monotonic() + 30):
            return independent_turn_context()

    context = contextvars.copy_context().run(initiate)
    seen = context.run(lambda: {variable: variable.get() for variable in caller})
    assert seen.pop(_TASK_SETTINGS) is caller[_TASK_SETTINGS]  # the caller's settings view carries
    assert all(value is None for value in seen.values()), seen  # its call state does not
    from ouroboros.model_wait import _CALENDAR

    assert context.run(_CALENDAR.get) == ("2099-01-01T00:00:00+00:00",) and context.run(_LOGICAL.get) == ()
