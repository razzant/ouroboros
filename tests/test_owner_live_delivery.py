"""Live-first owner delivery: gates, lineage stamping, and final-answer pick."""
import queue
import types

import pytest

from tests.test_one_work_placement import runtime as runtime  # fixture

from ouroboros.tools.owner_delivery import deliver_owner_event


class _Queue:
    def __init__(self, full=False):
        self.full = full
        self.items = []

    def put_nowait(self, item):
        if self.full:
            raise queue.Full()
        self.items.append(item)

    put = put_nowait


def _ctx(chat_id=123, *, event_queue=None, meta=None, drive_root=None):
    return types.SimpleNamespace(
        current_chat_id=chat_id,
        pending_events=[],
        drive_root=drive_root,
        task_id="t-live",
        task_metadata={"parent_task_id": "t-parent", "root_task_id": "t-root", **(meta or {})},
        event_queue=event_queue,
    )


class TestDeliverOwnerEvent:
    def test_live_path_stamps_lineage_and_skips_buffer(self):
        q = _Queue()
        ctx = _ctx(event_queue=q)
        mode = deliver_owner_event(ctx, {"type": "send_message", "chat_id": 123, "text": "hi"})
        assert mode == "live"
        assert ctx._owner_message_sent is True
        assert ctx.pending_events == []
        assert len(q.items) == 1
        frame = q.items[0]
        assert frame["task_id"] == "t-live"
        assert frame["parent_task_id"] == "t-parent"
        assert frame["root_task_id"] == "t-root"

    def test_no_queue_defers_with_lineage(self):
        ctx = _ctx(event_queue=None)
        mode = deliver_owner_event(ctx, {"type": "send_photo", "chat_id": 123})
        assert mode == "deferred"
        assert ctx._owner_message_sent is True
        assert ctx.pending_events[0]["root_task_id"] == "t-root"

    def test_retry_duplicates_are_accepted_policy(self):
        # A live-delivered frame from attempt 1 is not recalled; a retried
        # task re-narrates with a fresh ctx and delivers again. The seam
        # performs NO cross-attempt dedup — this pins the accepted policy.
        q = _Queue()
        for _attempt in (1, 2):
            ctx = _ctx(event_queue=q)  # fresh per-attempt ctx, sticky reset
            assert deliver_owner_event(
                ctx, {"type": "send_message", "chat_id": 1, "text": "same"}
            ) == "live"
        assert len(q.items) == 2

    def test_a2a_chat_always_deferred(self):
        q = _Queue()
        ctx = _ctx(chat_id=-5, event_queue=q)
        mode = deliver_owner_event(ctx, {"type": "send_message", "chat_id": -5, "text": "x"})
        assert mode == "deferred"
        assert q.items == []
        assert not getattr(ctx, "_owner_message_sent", False)

    def test_full_queue_goes_sticky_deferred(self):
        broken = _Queue(full=True)
        ctx = _ctx(event_queue=broken)
        first = deliver_owner_event(ctx, {"type": "send_message", "chat_id": 1, "text": "a"})
        assert first == "deferred"
        assert ctx._owner_message_sent is True
        # The queue recovers, but ordering wins: the task stays deferred so
        # frame B cannot overtake the already-buffered frame A.
        ctx.event_queue = _Queue()
        second = deliver_owner_event(ctx, {"type": "send_message", "chat_id": 1, "text": "b"})
        assert second == "deferred"
        assert [e["text"] for e in ctx.pending_events] == ["a", "b"]
        assert ctx.event_queue.items == []

    def test_origin_is_stamped_before_delivery_and_explicit_event_value_wins(self):
        q = _Queue()
        ctx = _ctx(event_queue=q, meta={"initiator": "consciousness"})
        frame = {"type": "send_message", "chat_id": 123, "text": "wake"}
        assert deliver_owner_event(ctx, frame) == "live"
        assert q.items[0]["initiator"] == "consciousness"
        assert q.items[0]["progress_meta"]["initiator"] == "consciousness"

        explicit = {"type": "send_message", "chat_id": 123, "text": "explicit",
                    "progress_meta": {"initiator": "explicit"}}
        assert deliver_owner_event(ctx, explicit) == "live"
        assert q.items[1]["initiator"] == "explicit"
        assert q.items[1]["progress_meta"]["initiator"] == "explicit"


class TestSendToolsLiveIntegration:
    def test_a2a_message_does_not_count_as_owner_delivery(self, tmp_path):
        from ouroboros.tools.control import _send_user_message

        ctx = _ctx(chat_id=-5, drive_root=tmp_path)
        ctx.drive_logs = lambda: tmp_path
        receipt = _send_user_message(ctx, "A message to the peer.")
        assert receipt == "OK: message queued for delivery to the peer conversation."
        assert ctx.pending_events[0]["chat_id"] == -5
        assert not getattr(ctx, "_owner_message_sent", False)

    def test_send_user_message_live_and_chat_zero(self, tmp_path):
        from ouroboros.tools.control import _send_user_message

        q = _Queue()
        ctx = _ctx(chat_id=0, event_queue=q)
        ctx.drive_logs = lambda: tmp_path
        result = _send_user_message(ctx, "hello owner")
        assert "OK" in result
        assert len(q.items) == 1
        frame = q.items[0]
        assert frame["chat_id"] == 0  # chat 0 is a real hidden session
        assert frame["task_id"] == "t-live"
        assert frame["is_progress"] is False
        assert (tmp_path / "events.jsonl").exists()

    def test_send_user_message_carries_proactive_discriminator(self, tmp_path):
        from ouroboros.tools.control import _send_user_message

        q = _Queue()
        ctx = _ctx(event_queue=q, drive_root=tmp_path)
        ctx.drive_logs = lambda: tmp_path
        assert "accepted for delivery to owner chat" in _send_user_message(ctx, "ping")
        frame = q.items[0]
        # Replay safety: an untyped assistant row with a task_id reads as the
        # task's final on history reload; the discriminator (persisted via
        # log_chat record_type) keeps the live card open.
        assert frame["system_type"] == "proactive_message"

    def test_send_user_message_stamps_consciousness_and_keeps_owner_turn_plain(self, tmp_path):
        from ouroboros.tools.control import _send_user_message

        q = _Queue()
        wake = _ctx(event_queue=q, drive_root=tmp_path, meta={"initiator": "consciousness"})
        wake.drive_logs = lambda: tmp_path
        assert "accepted for delivery" in _send_user_message(wake, "wake note")
        assert q.items[0]["progress_meta"]["initiator"] == "consciousness"

        owner_q = _Queue()
        owner = _ctx(event_queue=owner_q, drive_root=tmp_path)
        owner.drive_logs = lambda: tmp_path
        assert "accepted for delivery" in _send_user_message(owner, "owner note")
        assert "initiator" not in owner_q.items[0].get("progress_meta", {})

    def test_send_photo_accepts_chat_zero(self, tmp_path):
        from ouroboros.tools.core import _send_photo

        img = tmp_path / "shot.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 200)
        q = _Queue()
        ctx = _ctx(chat_id=0, event_queue=q)
        result = _send_photo(ctx, file_path=str(img))
        assert "OK" in result
        assert len(q.items) == 1
        assert q.items[0]["chat_id"] == 0

    def test_media_and_links_go_live_with_honest_receipts(self, tmp_path):
        from ouroboros.tools.core import _send_file, _send_links, _send_video

        vid = tmp_path / "c.mp4"
        vid.write_bytes(b"\x00" * 32)
        doc = tmp_path / "r.csv"
        doc.write_text("a,b\n", encoding="utf-8")

        q = _Queue()
        ctx = _ctx(event_queue=q, drive_root=tmp_path)
        # A live put is acceptance by the bus, not a confirmed send (#1789): the receipt says so.
        assert "accepted for delivery to owner chat" in _send_video(ctx, file_path=str(vid))
        assert "accepted for delivery to owner chat" in _send_file(ctx, file_path=str(doc))
        assert "accepted for delivery to owner chat" in _send_links(
            ctx, links=[{"label": "Docs", "url": "https://example.com/d"}])
        assert [e["type"] for e in q.items] == ["send_video", "send_document", "send_links"]
        assert ctx.pending_events == []  # live XOR deferred: never both

        deferred_ctx = _ctx(event_queue=None)
        assert "queued for delivery" in _send_video(deferred_ctx, file_path=str(vid))
        assert "queued for delivery" in _send_file(deferred_ctx, file_path=str(doc))
        assert "queued for delivery" in _send_links(
            deferred_ctx, links=[{"label": "Docs", "url": "https://example.com/d"}])
        assert len(deferred_ctx.pending_events) == 3


class TestFinalAnswerSelection:
    def test_deferred_mid_task_reply_stays_separate_from_the_live_final(self, tmp_path):
        from ouroboros.task_finalization import deliver_final_message_live
        from ouroboros.tools.control import _send_user_message

        ctx = _ctx(event_queue=None)
        ctx.drive_logs = lambda: tmp_path
        assert "queued for delivery" in _send_user_message(ctx, "A separate reply while I work.")
        ctx.pending_events.append({
            "type": "send_message", "task_id": ctx.task_id, "chat_id": ctx.current_chat_id,
            "text": "The final answer.", "is_progress": False,
        })
        q = _Queue()
        assert deliver_final_message_live(q, ctx.pending_events, ctx.task_id)
        assert [row["text"] for row in q.items] == ["The final answer."]
        assert ctx.pending_events[0]["system_type"] == "proactive_message"
        assert ctx.pending_events[0]["text"] == "A separate reply while I work."

    def test_final_beats_deferred_proactive_with_same_task_id(self):
        from ouroboros.task_finalization import deliver_final_message_live

        q = _Queue()
        buffer = [
            {"type": "send_message", "task_id": "t1", "chat_id": 1,
             "text": "mid-task proactive", "is_progress": False},
            {"type": "send_message", "task_id": "t1", "chat_id": 1,
             "text": "the final answer", "is_progress": False},
        ]
        assert deliver_final_message_live(q, buffer, "t1") is True
        assert len(q.items) == 1
        assert q.items[0]["text"] == "the final answer"


class TestSupervisorPhotoChatZero:
    def test_handle_send_photo_delivers_to_chat_zero(self):
        from supervisor.events_chat_delivery import _handle_send_photo

        sent = []

        class _Bridge:
            def send_photo(self, chat_id, data, caption="", mime="", task_id=""):
                sent.append(chat_id)
                return True, ""

        ctx = types.SimpleNamespace(bridge=_Bridge(), append_jsonl=lambda *a, **k: None,
                                    DRIVE_ROOT=None)
        import base64 as b64
        evt = {"type": "send_photo", "chat_id": 0, "task_id": "", "parent_task_id": "",
               "root_task_id": "", "image_base64": b64.b64encode(b"x" * 120).decode(),
               "caption": "", "mime": "image/png"}
        _handle_send_photo(evt, ctx)
        assert sent == [0]


@pytest.mark.parametrize("move", [False, True, "unread"])
@pytest.mark.parametrize("name, frame_type", [("send_user_message", "chat"), ("send_photo", "photo"),
    ("send_video", "video"), ("send_file", "document"), ("send_links", "links"), ("show_card", "card")])
def test_peer_receipt_follows_public_project_placement_and_real_delivery(runtime, monkeypatch, move, name, frame_type):
    """A raw peer id remains deferred, but the delivery consumer resolves a later Project binding."""
    import base64
    import builtins
    import json
    from collections import deque
    from pathlib import Path
    from ouroboros.projects_registry import project_binding_for_task
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.registry import ToolContext, ToolRegistry
    from ouroboros.utils import append_jsonl
    from supervisor import events_chat_delivery, message_bus
    from supervisor.events_project_routing import _handle_ensure_project_scope

    root, repo, workers = runtime
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")
    task = {"id": "one", "chat_id": -5, "type": "task", "project_id": "", "metadata": {}}
    workers.RUNNING["one"] = {"task": task}
    write_task_result(root, "one", "running", chat_id=-5)
    frames, peer_replies = [], []
    bridge = message_bus.LocalChatBridge({})
    bridge._broadcast_fn = frames.append
    bridge.subscribe_response(-5, peer_replies.append)
    monkeypatch.setattr(message_bus, "DATA_DIR", root)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 7})
    monkeypatch.setattr(message_bus, "publish_event", lambda *_a, **_kw: None)
    monkeypatch.setattr(events_chat_delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    host = types.SimpleNamespace(DRIVE_ROOT=root, RUNNING=workers.RUNNING, PENDING=workers.PENDING,
                                 append_jsonl=append_jsonl, send_with_budget=message_bus.send_with_budget, bridge=bridge)
    ctx = ToolContext(repo_dir=repo, drive_root=root, budget_drive_root=root, task_id="one",
                      current_chat_id=-5, task_metadata={}, task_contract={})
    ctx.event_queue = types.SimpleNamespace(put_nowait=lambda event: _handle_ensure_project_scope(event, host))
    registry = ToolRegistry(repo_dir=repo, drive_root=root)
    registry.set_context(ctx)
    if move:
        placement = registry.execute("move_to_project", {"project_name": "Room"})
        assert placement.startswith("OK:"), placement
    video, document = repo / "clip.mp4", repo / "report.txt"
    video.write_bytes(b"0" * 64)
    document.write_text("report", encoding="utf-8")
    args = {"send_user_message": {"text": "Progress for the current conversation."},
            "send_photo": {"image_base64": base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 4096).decode()},
            "send_video": {"file_path": str(video)}, "send_file": {"file_path": str(document)},
            "send_links": {"links": [{"label": "Docs", "url": "https://example.test/x"}]},
            "show_card": {"title": "Status", "fallback_text": "Status", "components": [{"type": "markdown", "text": "Status"}]}}
    binding_file = root / "state" / "project_task_bindings.json"
    real_open, failures = builtins.open, []

    def unread_binding(file, *args, **kwargs):
        if isinstance(file, (str, bytes, Path)) and Path(file) == binding_file:
            failures.append(str(file))
            raise OSError("transient binding source I/O failure")
        return real_open(file, *args, **kwargs)

    with monkeypatch.context() as fault:
        if move == "unread":
            fault.setattr(builtins, "open", unread_binding)
        receipt = registry.execute(name, args[name])
    assert bool(failures) == (move == "unread")
    words = json.loads(receipt)["delivery"] if name == "show_card" else receipt
    queued = list(ctx.pending_events)
    for event in queued:
        assert event["chat_id"] == -5
        getattr(events_chat_delivery, "_handle_" + event["type"])(event, host)
    delivered = [frame for frame in frames if frame.get("type") == frame_type
                 and frame.get("task_id") == "one" and frame.get("system_type") != "project_handoff"]
    if move:
        binding = project_binding_for_task(root, "one")
        assert len(queued) == 1  # raw peer transport still waits for the existing drain
        assert delivered and delivered[-1]["chat_id"] == binding["project_chat_id"]
        assert peer_replies == []
        expected = "destination unavailable" if move == "unread" else 'queued for delivery to Project room "Room"'
        assert expected in words and "PEER_DELIVERY_UNSUPPORTED" not in words
    elif name == "send_user_message":
        assert not delivered and len(queued) == 1 and len(peer_replies) == 1
        assert words == "OK: message queued for delivery to the peer conversation."
    else:
        assert not delivered and not peer_replies
        assert "PEER_DELIVERY_UNSUPPORTED" in words and "not delivered" in words
        assert not queued and not getattr(ctx, "_owner_message_sent", False)
    if name == "show_card":
        from ouroboros.card_store import load_card
        card = json.loads(receipt)
        assert load_card(root, card["card_id"])["revision"] == card["revision"] == 1
        assert "shown" not in card and ("queued" not in card or move)
