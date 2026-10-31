"""A Project root's deliberate Main notice.

``send_user_message(destination="main")`` addresses the owner's main chat from an
owner-visible Project room. It is one message: written to Main, shown by its
work's Project too (one row identity, read once). These tests pin both ends: the
tool's frame, the supervisor's routing, the durable row the history replay reads,
and that the notice never stands in for the task's final answer. The default
destination keeps today's current-room behaviour.
"""
import asyncio
import json
import queue
import types
from collections import deque

import pytest

from ouroboros.tools.control_runtime import _send_user_message


def _ctx(tmp_path, chat_id, *, event_queue=None, meta=None, contract=None):
    ctx = types.SimpleNamespace(
        current_chat_id=chat_id, task_id="root-project", task_metadata=dict(meta or {}),
        event_queue=event_queue, pending_events=[], drive_root=tmp_path,
        drive_logs=lambda: tmp_path / "logs",
    )
    if contract is not None:
        ctx.task_contract = contract
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    return ctx


def _project(tmp_path):
    from ouroboros.projects_registry import bind_task_to_project, create_project

    project = create_project(tmp_path, "launch", name="Launch")
    bind_task_to_project(tmp_path, "root-project", project["id"], project["chat_id"],
                         origin={"absent": "system"})
    return int(project["chat_id"])


class TestProducer:
    def test_main_destination_addresses_main_with_its_own_type(self, tmp_path):
        q = queue.Queue()
        ctx = _ctx(tmp_path, _project(tmp_path), event_queue=q)
        assert _send_user_message(ctx, "The Cursor account is out of usage.", destination="main") == (
            "OK: notice sent to the main chat.")
        frame = q.get_nowait()
        assert frame["chat_id"] == 1
        assert frame["system_type"] == "main_notice"
        assert frame["task_id"] == "root-project"  # provenance kept
        logged = [json.loads(line) for line in (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        assert logged[-1]["destination"] == "main"

    def test_external_positive_owner_ingress_is_allowed_but_numeric_id_alone_is_not(self, tmp_path):
        q = queue.Queue()
        unknown = _ctx(tmp_path, 5_000_000_001, event_queue=q)
        assert _send_user_message(unknown, "An owner notice", destination="main").startswith("⚠️ MAIN_NOTICE_BLOCKED:")
        assert q.empty() and not unknown.pending_events
        # The owner door stamps this fact; a large number alone cannot mint it.
        owner = _ctx(tmp_path, 5_000_000_001, event_queue=q, meta={"origin_suppressed": True})
        assert _send_user_message(owner, "An owner notice", destination="main").startswith("OK:")
        assert q.get_nowait()["chat_id"] == 1

    def test_main_room_id_alone_is_not_owner_proof(self, tmp_path):
        q = queue.Queue()
        # A wake or scheduled root runs in Main with no owner-door stamp.
        bare = _ctx(tmp_path, 1, event_queue=q)
        assert _send_user_message(bare, "An owner notice", destination="main").startswith("⚠️ MAIN_NOTICE_BLOCKED:")
        assert q.empty() and not bare.pending_events
        assert not (tmp_path / "logs" / "events.jsonl").exists()
        owner = _ctx(tmp_path, 1, event_queue=q, meta={"origin_message_ref": {"chat_id": 1, "client_message_id": "m1"}})
        assert _send_user_message(owner, "An owner notice", destination="main") == "OK: notice sent to the main chat."
        frame = q.get_nowait()
        assert frame["chat_id"] == 1 and frame["system_type"] == "main_notice"

    @pytest.mark.parametrize("destination", ["current", "", None, " Current "])
    def test_default_and_empty_destination_stay_in_the_current_room(self, tmp_path, destination):
        q = queue.Queue()
        ctx = _ctx(tmp_path, 1_000_123, event_queue=q)
        assert _send_user_message(ctx, "A word while I work.", destination=destination) == "OK: message sent to owner chat."
        frame = q.get_nowait()
        assert frame["chat_id"] == 1_000_123
        assert frame["system_type"] == "proactive_message"

    def test_unknown_destination_is_refused_and_nothing_is_sent(self, tmp_path):
        q = queue.Queue()
        ctx = _ctx(tmp_path, 1_000_123, event_queue=q)
        result = _send_user_message(ctx, "hello", destination="telegram")
        assert result.startswith("⚠️ SEND_USER_MESSAGE_DESTINATION:")
        assert "destination='telegram'" in result and "'main'" in result
        assert q.empty() and not ctx.pending_events

    @pytest.mark.parametrize("case", ["child_metadata", "child_contract", "child_lineage", "presence", "a2a", "hidden", "hidden_string"])
    def test_child_presence_and_a2a_callers_gain_no_main_voice(self, tmp_path, case):
        q = queue.Queue()
        chat_id, meta, contract = (_project(tmp_path) if case in
                                   {"child_metadata", "child_contract", "child_lineage", "presence"}
                                   else 1_000_123), None, None
        if case == "child_metadata":
            meta = {"parent_task_id": "p", "root_task_id": "p"}
        elif case == "child_contract":
            contract = {"delegation_role": "subagent"}
        elif case == "child_lineage":
            contract = {"lineage": {"parent_task_id": "p", "delegation_role": "subagent"}}
        elif case == "presence":
            contract = {"capability_ceiling": {}}
        elif case == "a2a":
            chat_id = -5
        elif case == "hidden":
            chat_id = 0
        else:
            chat_id = "0"
        ctx = _ctx(tmp_path, chat_id, event_queue=q, meta=meta, contract=contract)
        result = _send_user_message(ctx, "The owner should know.", destination="main")
        assert result.startswith("⚠️ MAIN_NOTICE_BLOCKED:")
        assert q.empty() and not ctx.pending_events
        # The current room is unchanged for the same caller.
        assert _send_user_message(ctx, "Still here.").startswith("OK:")


class TestFinalCustody:
    def test_a_deferred_notice_is_never_selected_as_the_final_answer(self, tmp_path):
        from ouroboros.task_finalization import deliver_final_message_live

        ctx = _ctx(tmp_path, _project(tmp_path), event_queue=None)
        assert _send_user_message(ctx, "Owner-held problem.", destination="main") == (
            "OK: notice queued for delivery to the main chat.")
        assert _send_user_message(ctx, "Mid-task reply.") == "OK: message queued for delivery."
        live = queue.Queue()
        # No final buffered yet: neither typed send may ship as the answer.
        assert deliver_final_message_live(live, ctx.pending_events, ctx.task_id) is False
        assert live.empty()
        ctx.pending_events.append({"type": "send_message", "task_id": ctx.task_id, "chat_id": 1_000_123,
                                   "text": "The final answer.", "is_progress": False})
        assert deliver_final_message_live(live, ctx.pending_events, ctx.task_id) is True
        assert [row["text"] for row in live.queue] == ["The final answer."]
        assert live.queue[0]["delivery_id"].startswith("final:root-project:")
        assert "delivery_id" not in ctx.pending_events[0]


@pytest.fixture
def host(tmp_path, monkeypatch):
    """The real supervisor handler, message bus and durable chat log."""
    from ouroboros.utils import append_jsonl
    from supervisor import events_chat_delivery as delivery, message_bus

    frames = []
    bridge = message_bus.LocalChatBridge({})
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 7})
    monkeypatch.setattr(message_bus, "publish_event", lambda *_a, **_k: None)
    monkeypatch.setattr(delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    ctx = types.SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, append_jsonl=append_jsonl,
                                send_with_budget=message_bus.send_with_budget)
    return types.SimpleNamespace(ctx=ctx, frames=frames,
                                 handle=lambda evt: delivery._handle_send_message(evt, ctx))


def _history(tmp_path, chat_id):
    from ouroboros.gateway.history import make_chat_history_endpoint

    endpoint = make_chat_history_endpoint(tmp_path)
    response = asyncio.run(endpoint(types.SimpleNamespace(query_params={"chat_id": str(chat_id)})))
    return json.loads(response.body)["messages"]


class TestDeliveryAndReplay:
    @pytest.mark.parametrize("live", [True, False], ids=["live", "deferred"])
    @pytest.mark.parametrize("born_in", ["project", "main", "main_unstamped"])
    def test_notice_is_one_message_in_main_and_its_project_live_and_on_reload(self, tmp_path, host, live, born_in):
        project_chat = _project(tmp_path)
        # A root bound after admission keeps the chat it was born in on its row;
        # lineage still routes its ordinary replies into the Project thread.
        # That durable binding makes it a Project root whether or not the owner
        # door stamped it (a wake or scheduled root converted or self-scoped).
        origin_chat, meta = {
            "project": (project_chat, None),
            "main": (1, {"origin_message_ref": {"chat_id": 1, "client_message_id": "m1"}}),
            "main_unstamped": (1, None),
        }[born_in]
        worker_q = queue.Queue() if live else None
        ctx = _ctx(tmp_path, origin_chat, event_queue=worker_q, meta=meta)
        assert _send_user_message(ctx, "Cursor account needs on-demand usage enabled.", destination="main").startswith("OK:")
        assert _send_user_message(ctx, "Continuing with two reviewers.").startswith("OK:")
        events = list(worker_q.queue) if live else list(ctx.pending_events)  # live queue or end-of-task drain
        for evt in events:
            host.handle(evt)

        chats = [frame for frame in host.frames if frame.get("type") == "chat"]
        by_type = {frame.get("system_type"): frame for frame in chats}
        notice, reply = by_type["main_notice"], by_type["proactive_message"]
        # One frame, one notification: written to Main, also reaching the Project's panel.
        assert [frame.get("system_type") for frame in chats].count("main_notice") == 1
        assert notice["chat_id"] == 1 and not notice.get("project_thread")
        assert notice["also_rooms"] == [project_chat]
        assert notice["task_id"] == "root-project"
        assert reply["chat_id"] == project_chat and reply.get("project_thread") is True and "also_rooms" not in reply
        from ouroboros import dialogue_store
        # Unread where it is shown: one revision in Main and one in the Project, stamped on the row.
        assert dialogue_store.room(tmp_path, 1)["revision"] == 1
        assert dialogue_store.room(tmp_path, project_chat)["revision"] == 2
        assert notice["room_revisions"] == {"1": 1, str(project_chat): notice["room_revisions"][str(project_chat)]}

        main_rows = _history(tmp_path, 1)
        project_rows = _history(tmp_path, project_chat)
        main_notices = [row for row in main_rows if row.get("system_type") == "main_notice"]
        project_notices = [row for row in project_rows if row.get("system_type") == "main_notice"]
        assert [row["text"] for row in main_notices] == ["Cursor account needs on-demand usage enabled."]
        assert main_notices[0]["role"] == "assistant" and main_notices[0]["task_id"] == "root-project"
        # The same row, not a copy: one address in both rooms, the live frame's too.
        assert [row["row_address"] for row in project_notices] == [main_notices[0]["row_address"]] == [notice["row_address"]]
        # Ordinary work speech still lives in its Project only.
        assert all(row.get("system_type") != "proactive_message" for row in main_rows)
        assert [row["text"] for row in project_rows if row.get("system_type") == "proactive_message"] == [
            "Continuing with two reviewers."]
