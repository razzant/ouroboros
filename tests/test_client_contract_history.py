"""The published history contract against its real producers (docs/CLIENT_CONTRACT.md §5).

Rows are written by the real chat seams (``LocalChatBridge`` and ``send_with_budget``), read back
through ``GET /api/chat/history`` and checked against ``ChatHistoryMessage``/``ChatHistoryResponse``.
The live/history merge rule the contract publishes is checked frame by frame: every stored live
frame finds exactly its own history row by the published identity, and no two rows share one.
"""

from __future__ import annotations

import asyncio
import json
import typing
from types import SimpleNamespace

import pytest

from ouroboros.gateway.contracts import ChatOutbound
from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.gateway.history_contracts import ChatHistoryMessage, ChatHistoryResponse, HistoricalTerminal
from ouroboros.gateway.schema import validate_ingress
from ouroboros.owner_quiz import record_asked
from supervisor import message_bus

TS = "2026-10-13T09:00:00.000001+00:00"
ROW_KEYS = set(typing.get_type_hints(ChatHistoryMessage)) | set(typing.get_type_hints(ChatOutbound))


def _read(tmp_path, **query):
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params=query)))
    return response.status_code, json.loads(response.body.decode("utf-8"))


@pytest.fixture
def chat(tmp_path, monkeypatch):
    """The real bridge over a disposable data root; frames are what a socket would receive."""
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "progress.jsonl").touch()
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "publish_event", lambda *_a, **_kw: None)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 7})
    bridge = message_bus.LocalChatBridge({})
    frames: list = []
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    return SimpleNamespace(bridge=bridge, frames=frames, root=tmp_path)


def _produce(chat) -> None:
    bridge = chat.bridge
    bridge.handle_web_message("Plan the trip", sender_session_id="s1", client_message_id="cm-1", chat_id=1)
    message_bus.send_with_budget(1, "Working on it", is_progress=True, task_id="t1")
    message_bus.send_with_budget(1, "Here is the plan.", task_id="t1")
    message_bus.send_with_budget(1, "Here is the plan.", task_id="t2")  # same words, another task
    message_bus.send_with_budget(1, "Review settled.", task_id="t1", role="system", system_type="custody_notice")
    for fmt in ("", "markdown"):
        for index, meta in enumerate(({"source": "custom-service"}, {}, {"source": ""})):
            message_bus.send_with_budget(1, f"**Notice {index}** ({fmt or 'plain'})", fmt=fmt,
                                         task_id="t1", progress_meta=meta)
    bridge.send_document(1, b"%PDF-1.4 data", filename="plan.pdf", caption="the plan",
                         mime="application/pdf", task_id="t1")
    bridge.send_photo(1, b"\x89PNG\r\n\x1a\n" + b"0" * 16, caption="map", task_id="t1")
    bridge.send_links(1, [{"label": "Tickets", "url": "https://example.com/tickets"}], title="Book", task_id="t1")
    # A card is published only for a question its task recorded first (the production ask order).
    record_asked(chat.root, "t1", quiz_id="q1", question="Train or plane?", options=["Train", "Plane"],
                 assumption="train meanwhile")
    ok, detail = bridge.send_quiz(1, quiz_id="q1", question="Train or plane?",
                                  options=[{"label": "Train"}, {"label": "Plane"}], assumption="train meanwhile",
                                  task_id="t1")
    assert ok, detail


def _live_key(frame: dict):
    """The identity docs/CLIENT_CONTRACT.md §5 tells a client to merge a live frame by."""
    kind = frame["type"]
    if kind == "chat" and frame.get("role") == "user" and frame.get("client_message_id"):
        return ("owner", frame["client_message_id"])
    if kind == "quiz":
        return ("quiz", frame["task_id"], frame["quiz_id"])
    if kind in {"photo", "video", "document", "links"}:
        return ("media", kind, frame["task_id"], frame["ts"])
    return ("text", bool(frame.get("is_progress")), frame["role"], frame.get("system_type", ""),
            frame["task_id"], frame["ts"], frame["content"])


def _row_key(row: dict):
    """The same identity, read from a history row; the owner row is the `role: "user"` one (a question
    card also stores its publication receipt as `client_message_id`)."""
    if row.get("role") == "user" and row.get("client_message_id"):
        return ("owner", row["client_message_id"])
    if row.get("msg_type") == "quiz":
        return ("quiz", row["task_id"], row["quiz"]["quiz_id"])
    if row.get("msg_type"):
        return ("media", row["msg_type"], row["task_id"], row["ts"])
    return ("text", row["is_progress"], row["role"], row["system_type"], row["task_id"], row["ts"], row["text"])


def test_history_rows_follow_the_published_row_contract(chat):
    _produce(chat)
    status, body = _read(chat.root)
    assert status == 200
    assert not validate_ingress(body, ChatHistoryResponse), validate_ingress(body, ChatHistoryResponse)
    assert body["messages"], "the producers wrote rows"
    for row in body["messages"]:
        assert not validate_ingress(row, ChatHistoryMessage), (row, validate_ingress(row, ChatHistoryMessage))
        undeclared = set(row) - ROW_KEYS
        assert not undeclared, f"history row keys the contract does not publish: {sorted(undeclared)}"
        assert row["history_id"] == f"{row['history_position']['source']}:{row['history_position']['offset']}"
    timestamps = [row["ts"] for row in body["messages"]]
    assert timestamps == sorted(timestamps), "rows are oldest first"


@pytest.mark.parametrize("source", ["unknown", "supervisor_cancel", "executor_terminal", "legacy"])
def test_main_receipt_contract_preserves_the_clock_that_was_published(chat, monkeypatch, source):
    from ouroboros import dialogue_store
    from ouroboros.task_results import write_task_result
    from ouroboros.terminal_projection import _project_row

    tid = "room-clock"
    published = "2026-10-14T12:34:56+00:00"
    write_task_result(chat.root, tid, "running", chat_id=1, task_attempt=0, started_at="2026-10-12T08:00:00Z")
    monkeypatch.setattr("ouroboros.task_results.utc_now_iso", lambda: TS)
    stored = write_task_result(chat.root, tid, "cancelled" if source == "supervisor_cancel" else "completed",
                               _cancel_observed=source == "supervisor_cancel", _terminal_observed=source == "executor_terminal",
                               suggested_name="Inbox check", schedule_name="Daily", wake_summary="Inbox checked.", wake_silent=True)
    monkeypatch.setattr("ouroboros.terminal_projection.utc_now_iso", lambda: published)
    event = _project_row(tid, stored, {}, {"chat_id": 1}, delivery={"state": "exhausted", "preserved_path": "saved/answer"})
    if source == "legacy":
        event.pop("terminal_time")
    dialogue_store.append(chat.root, "chat", event)
    code, body = _read(chat.root)
    assert code == 200
    [receipt] = [r for r in body["messages"] if r.get("task_id") == tid]
    frozen = receipt["historical_terminal"]
    assert receipt["ts"] == frozen["ts"] == published
    if source == "legacy":
        assert "terminal_time" not in receipt and "terminal_time" not in frozen
    else:
        assert receipt["terminal_time"] == frozen["terminal_time"] == event["terminal_time"]
        assert frozen["task_attempt"] == 0 and frozen["terminal_time"]["source"] == source
        assert frozen["terminal_time"]["occurred_at"] == (None if source == "unknown" else TS)
    for key in ("wake_summary", "schedule_name", "suggested_name", "wake_silent", "quiet_effective", "terminal_delivery"):
        assert receipt[key] == frozen[key] == event[key]
    assert set(receipt) <= set(typing.get_type_hints(ChatHistoryMessage))
    assert set(frozen) <= set(typing.get_type_hints(HistoricalTerminal))
    assert not validate_ingress(receipt, ChatHistoryMessage)


def test_live_executor_frame_and_its_history_row_identify_the_recorded_attempt(chat):
    observation = {"task_id": "work", "task_attempt": "0", "run_id": "job", "attempt_id": "run-1",
                   "harness_id": "codex", "phase": "running", "revision": 1}
    message_bus.send_with_budget(1, "Reading", task_id="work", is_progress=True,
                                 progress_meta={"executor_observation": observation})
    code, body = _read(chat.root)
    assert code == 200
    [live] = [frame for frame in chat.frames if frame.get("type") == "chat" and frame.get("task_id") == "work"]
    [recorded] = [row for row in body["messages"] if row.get("task_id") == "work"]
    assert _live_key(live) == _row_key(recorded)
    assert live["executor_observation"] == recorded["executor_observation"] == observation
    assert "task_attempt" not in live and recorded["task_attempt"] == 0
    assert not validate_ingress(live, ChatOutbound) and not validate_ingress(recorded, ChatHistoryMessage)


RETRIED = "inventory"


def _attempt_names(root, attempt):
    """How a task and its stored result name an attempt: a pooled dispatch's ordinal, a first direct turn's
    owner-bound zero, or nothing (a writer older than the stamp)."""
    if attempt is None:
        return {}, {}
    if attempt == 0:
        from ouroboros.task_status import execution_owner_record

        owner = execution_owner_record(root, {"id": RETRIED}, "direct")
        return {"_execution_owner": owner}, {"execution_owner": owner}
    return {"_attempt": attempt}, {"task_attempt": attempt}


def _facts_row(chat, attempt, **usage):
    """An attempt's own facts row, written by its post-task stage over the result that attempt stored."""
    from ouroboros.post_task_synthesis import _record_task_facts

    task = {"id": RETRIED, "chat_id": 1, **_attempt_names(chat.root, attempt)[0]}
    _record_task_facts(SimpleNamespace(drive_root=chat.root), task, usage, {"tool_calls": [{"tool": "read_file"}]},
                       chat.root / "logs")


@pytest.mark.parametrize(("earlier", "current"), [
    (1, 2), (0, 1), (None, 2), (None, 0),  # another attempt's ending; an unnumbered one is no numbered attempt's, zero included
    (None, None),  # no ordinal anywhere: the rows cannot be told apart and all follow the stored result
])
def test_a_finalizing_retry_holds_its_early_answer_apart_from_an_earlier_attempts_ending(chat, earlier, current):
    """An attempt failed and recorded its facts; the same task answered again and still finalizes. Its narration and
    its early answer are stored without an attempt, as their producers write them."""
    from tests._store_shared import seed_result

    message_bus.send_with_budget(1, "Counting the east bed", is_progress=True, task_id=RETRIED)
    seed_result(chat.root, RETRIED, {"chat_id": 1, "status": "failed", "reason_code": "tool_error",
                                     **_attempt_names(chat.root, earlier)[1]})
    _facts_row(chat, earlier, rounds=4, reason_code="tool_error", outcome_axes={"lifecycle": {"status": "failed"}})
    seed_result(chat.root, RETRIED, {"chat_id": 1, "status": "completed", "reason_code": "final_message",
                                     "root_phase_checkpoint": {"post_task_synthesis": "running"},
                                     "loop_outcome": {"usage": {"total_rounds": 17}},
                                     **_attempt_names(chat.root, current)[1]})
    message_bus.send_with_budget(1, "Counting it again", is_progress=True, task_id=RETRIED)
    message_bus.send_with_budget(1, "The east bed holds sage.", task_id=RETRIED,
                                 progress_meta={"task_phase": "finalizing", "task_terminal_status": "completed"})
    _facts_row(chat, current, rounds=17, reason_code="final_message", outcome_axes={"lifecycle": {"status": "completed"}})

    status, body = _read(chat.root)
    assert status == 200
    rows = [row for row in body["messages"] if row.get("task_id") == RETRIED]
    summaries = [row for row in rows if row["system_type"] == "task_summary"]
    speech = [row for row in rows if row["system_type"] != "task_summary"]
    assert sorted(row["text"] for row in speech) == ["Counting it again", "Counting the east bed", "The east bed holds sage."]
    assert [row["is_progress"] for row in speech].count(True) == 2 and len(summaries) == 2
    for row in speech:  # the stored result's early answer, whichever attempt wrote the row: no row names one
        assert "task_attempt" not in row
        assert (row["task_phase"], row["outcome_final"], row["rounds"], row["reason_code"]) == (
            "finalizing", False, 17, "final_message")
    old, own = (next(row for row in summaries if row["status"] == ended) for ended in ("failed", "completed"))
    assert (own["task_phase"], own["outcome_final"], own["rounds"]) == ("finalizing", False, 17)
    if earlier is None and current is None:
        assert (old["task_phase"], old["rounds"], old["reason_code"]) == ("finalizing", 17, "final_message")
    else:  # its own ending stays its own and is no early answer of the attempt that still finalizes
        assert old["task_phase"] == "unfinished" and "current_outcome" not in old
        assert (old.get("task_attempt"), old["rounds"], old["tool_calls"], old["reason_code"]) == (earlier, 4, 1, "tool_error")


@pytest.mark.parametrize(("earlier", "current"), [(1, 2), (None, 0)])
def test_an_earlier_attempts_row_carries_the_stored_end_as_that_results_own_clock(chat, monkeypatch, earlier, current):
    """The result writer dates the attempt it saw end; another attempt's row shows that end beside its own facts."""
    from ouroboros.task_results import write_task_result

    started = "2026-10-12T08:00:00Z"
    write_task_result(chat.root, RETRIED, "running", chat_id=1, task_attempt=current, started_at=started)
    monkeypatch.setattr("ouroboros.task_results.utc_now_iso", lambda: TS)
    stored = write_task_result(chat.root, RETRIED, "completed", _terminal_observed=True, reason_code="final_message")
    assert (stored["terminal_time"]["source"], stored["terminal_time"]["occurred_at"]) == ("executor_terminal", TS)
    # A facts row of the earlier attempt, written from its own snapshot after the store moved on.
    _facts_row(chat, earlier, rounds=4, reason_code="tool_error", outcome_axes={"lifecycle": {"status": "failed"}})

    status, body = _read(chat.root)
    assert status == 200
    [old] = [row for row in body["messages"] if row.get("task_id") == RETRIED]
    assert (old.get("task_attempt"), old["status"], old["rounds"]) == (earlier, "failed", 4)
    ended = old["current_outcome"]
    assert (ended["task_attempt"], ended["status"], ended["terminal_time"]) == (current, "completed", stored["terminal_time"])
    assert ended["terminal_time"]["attempt"]["task_attempt"] == current
    assert ended["terminal_time"]["attempt"]["started_at"] == started


def test_the_reply_author_label_is_a_published_field_of_live_frames_and_history_rows(chat):
    """What a Reply to a row signs (``chat_references.projection_fields``) rides the live frame and the replay of every
    stored row, so the published types, their browser twins and the contract's words declare it, and the schema types
    it: the owner's row says You, an answer Ouroboros, the host's own row System."""
    import pathlib

    from ouroboros.gateway.reference_contracts import ChatReferenceFields
    from tests.test_gateway_parity import _js_typedef_fields

    _produce(chat)
    chat.bridge.handle_web_message("Cue: greet the caller.", client_message_id="call:c1:open",
                                   task_metadata={"host_fact": {"kind": "call_open", "call_id": "c1"}})
    frames = [frame for frame in chat.frames if frame.get("type") == "chat" and "reply_author_label" in frame]
    status, body = _read(chat.root)
    rows = [row for row in body["messages"] if "reply_author_label" in row]
    assert status == 200 and frames and rows
    assert {frame["reply_author_label"] for frame in frames} >= {"You", "Ouroboros", "System"}
    assert {row["reply_author_label"] for row in rows} >= {"You", "Ouroboros", "System"}
    for frame in frames:
        assert not validate_ingress(frame, ChatOutbound), (frame, validate_ingress(frame, ChatOutbound))
    for row in rows:
        assert not validate_ingress(row, ChatHistoryMessage), (row, validate_ingress(row, ChatHistoryMessage))
    assert validate_ingress({**frames[0], "reply_author_label": 3}, ChatOutbound), "typed: a label is a string"
    assert validate_ingress({**rows[0], "reply_author_label": 3}, ChatHistoryMessage)
    assert "reply_author_label" in typing.get_type_hints(ChatReferenceFields) and "reply_author_label" in ROW_KEYS
    web = pathlib.Path(__file__).resolve().parents[1]
    assert "reply_author_label" in _js_typedef_fields((web / "web/modules/api_types.js").read_text(encoding="utf-8"), "ChatOutbound")
    assert "reply_author_label" in _js_typedef_fields(
        (web / "web/modules/client_contract_types.js").read_text(encoding="utf-8"), "ChatHistoryMessage")
    assert "`reply_author_label`" in (web / "docs/CLIENT_CONTRACT.md").read_text(encoding="utf-8")


def test_every_stored_live_frame_finds_exactly_its_history_row(chat):
    _produce(chat)
    stored = [frame for frame in chat.frames if frame["type"] in {"chat", "photo", "document", "links", "quiz"}]
    assert {frame["type"] for frame in stored} == {"chat", "photo", "document", "links", "quiz"}
    owner = [frame for frame in stored if frame.get("client_message_id") == "cm-1"]
    assert len(owner) == 1 and owner[0]["role"] == "user" and owner[0]["ingress_accepted"] is True
    assert any(frame.get("is_progress") for frame in stored), "a progress frame is checked too"
    _status, body = _read(chat.root)
    rows = body["messages"]
    row_keys = [_row_key(row) for row in rows]
    assert len(row_keys) == len(set(row_keys)), "two history rows share one published identity"
    for frame in stored:
        matches = [row for row in rows if _row_key(row) == _live_key(frame)]
        assert len(matches) == 1, (frame, matches)
        # The facts the rule relies on: the stored row keeps the live frame's exact time string.
        assert matches[0]["ts"] == frame["ts"]
        assert matches[0].get("source", "") == frame.get("source", ""), (frame, matches[0])
    live_keys = [_live_key(frame) for frame in stored]
    assert len(live_keys) == len(set(live_keys)), "the same words on another task stay distinct"


def test_the_same_stored_row_keeps_its_history_id_across_reads(chat):
    _produce(chat)
    _status, first = _read(chat.root)
    message_bus.send_with_budget(1, "A later answer.", task_id="t3")
    _status, second = _read(chat.root)
    before = {row["history_id"]: _row_key(row) for row in first["messages"]}
    after = {row["history_id"]: _row_key(row) for row in second["messages"]}
    assert before.items() <= after.items()
    assert len(after) == len(before) + 1


def test_system_submission_correlation_is_not_the_owner_rows_identity(chat):
    from ouroboros.task_finalization import host_operation_reply_kwargs

    chat.bridge.handle_web_message("/status", client_message_id="cm-status", chat_id=1)
    message_bus.send_with_budget(
        1, "Status is available.", role="system", system_type="command_reply",
        **host_operation_reply_kwargs(None, "", "cm-status"),
    )
    frames = [frame for frame in chat.frames if frame["type"] == "chat"]
    assert [frame["role"] for frame in frames] == ["user", "system"]
    assert all(frame["client_message_id"] == "cm-status" for frame in frames)
    assert len({_live_key(frame) for frame in frames}) == 2
    status, body = _read(chat.root)
    assert status == 200
    for frame in frames:
        matches = [row for row in body["messages"] if _row_key(row) == _live_key(frame)]
        assert len(matches) == 1 and matches[0]["role"] == frame["role"]
    system, = [row for row in body["messages"] if row["role"] == "system"]
    assert not system.get("client_message_id"), "live correlation is not stored owner identity"


def test_a_refused_cursor_keeps_the_response_shape_and_echoes_the_cursor(chat):
    _produce(chat)
    status, body = _read(chat.root, cursor="not-a-cursor")
    assert status == 400 and body["reason_code"] == "history_cursor_invalid"
    assert not validate_ingress(body, ChatHistoryResponse)
    assert body["messages"] == [] and body["has_more"] is True and body["next_cursor"] == "not-a-cursor"
    assert body["window"] == {"complete": False, "truncated_by": ["history_cursor_invalid"]}


def test_paging_with_next_cursor_reaches_older_rows_without_repeating_them(chat):
    for index in range(5):
        message_bus.send_with_budget(1, f"Answer {index}", task_id=f"p{index}")
    _status, recent = _read(chat.root, n_human="2")
    assert recent["has_more"] is True and recent["window"]["complete"] is False
    assert [row["text"] for row in recent["messages"]] == ["Answer 3", "Answer 4"]
    _status, older = _read(chat.root, cursor=recent["next_cursor"])
    assert [row["text"] for row in older["messages"]] == ["Answer 1", "Answer 2"]
    seen = {row["history_id"] for row in recent["messages"]}
    assert not seen & {row["history_id"] for row in older["messages"]}
    _status, again = _read(chat.root, cursor=older["page_cursor"])
    assert [row["history_id"] for row in again["messages"]] == [row["history_id"] for row in older["messages"]]


def test_a_shared_row_and_a_reply_to_it_stay_inside_both_published_twins(chat):
    """A Project's notice written to Main and the owner's direct Reply to it, made by the real producers: their live
    frames and their history rows in both rooms stay inside the published contract, with the facts a shared row adds
    (``room_revisions``, ``also_rooms``, the Reply reference's ``target_home``) declared and typed on both twins."""
    import pathlib

    from ouroboros.gateway.reference_contracts import ChatReplyReference
    from ouroboros.projects_registry import create_project
    from ouroboros.task_results import write_task_result
    from tests.test_gateway_parity import _js_typedef_fields

    nova = int(create_project(chat.root, "nova", name="Nova")["chat_id"])
    write_task_result(chat.root, "nova-root", "running", chat_id=nova, title="nova-root")
    message_bus.send_with_budget(1, "Owner action needed in Settings.", task_id="nova-root", system_type="main_notice")
    notice = next(frame for frame in chat.frames if frame.get("system_type") == "main_notice")
    chat.bridge.handle_web_message("Why only here?", client_message_id="cm-reply", chat_id=1,
                                   task_metadata={"reply_to": notice["row_address"]}, dispatch=lambda *a, **k: None)
    echo = next(frame for frame in chat.frames if frame.get("client_message_id") == "cm-reply" and frame["role"] == "user")
    frame_keys, reference_keys = set(typing.get_type_hints(ChatOutbound)), set(typing.get_type_hints(ChatReplyReference))
    for frame in (notice, echo):
        assert frame.get("also_rooms") == [nova], frame
        assert not validate_ingress(frame, ChatOutbound), validate_ingress(frame, ChatOutbound)
    assert {"also_rooms", "room_revisions", "reply_to"} <= frame_keys
    assert set(notice["room_revisions"]) == {"1", str(nova)}
    assert echo["reply_to"]["target_home"] == "work" and set(echo["reply_to"]) <= reference_keys
    addresses = {notice["row_address"], echo["row_address"]}
    for room in (1, nova):
        status, body = _read(chat.root, chat_id=str(room))
        assert status == 200 and not validate_ingress(body, ChatHistoryResponse)
        rows = [row for row in body["messages"] if row.get("row_address") in addresses]
        assert len(rows) == 2, (room, body["messages"])
        for row in rows:
            assert not validate_ingress(row, ChatHistoryMessage), (row, validate_ingress(row, ChatHistoryMessage))
            assert not set(row) - ROW_KEYS, f"history row keys the contract does not publish: {sorted(set(row) - ROW_KEYS)}"
        shared = next(row for row in rows if row.get("system_type") == "main_notice")
        reply = next(row for row in rows if row["role"] == "user")
        assert set(shared["room_revisions"]) == {"1", str(nova)}
        assert reply["reply_to"]["target_home"] == "work" and set(reply["reply_to"]) <= reference_keys
    # The reference is typed, not only present: a value outside the published one is refused.
    assert validate_ingress({**reply, "reply_to": {**reply["reply_to"], "target_home": "elsewhere"}}, ChatHistoryMessage)
    twins = (pathlib.Path(__file__).resolve().parents[1] / "web" / "modules" / "client_contract_types.js").read_text(
        encoding="utf-8")
    assert _js_typedef_fields(twins, "ChatReplyReference") == reference_keys
