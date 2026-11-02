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
from ouroboros.gateway.history_contracts import ChatHistoryMessage, ChatHistoryResponse
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
