"""Live-first owner delivery: gates, lineage stamping, and final-answer pick."""
import queue
import types

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


def _ctx(chat_id=123, *, event_queue=None, meta=None):
    return types.SimpleNamespace(
        current_chat_id=chat_id,
        pending_events=[],
        drive_root=None,
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
        assert ctx.pending_events[0]["root_task_id"] == "t-root"

    def test_background_consciousness_always_deferred_and_unstamped(self):
        from ouroboros.tool_capabilities import BACKGROUND_DELEGATION_ROLE

        q = _Queue()
        ctx = _ctx(event_queue=q, meta={"delegation_role": BACKGROUND_DELEGATION_ROLE})
        mode = deliver_owner_event(ctx, {"type": "send_message", "chat_id": 1, "text": "x"})
        assert mode == "deferred"
        assert q.items == []
        # BG frames stay exactly as before the seam: buffered, no pseudo-lineage.
        assert "task_id" not in ctx.pending_events[0]

    def test_consciousness_stamps_the_shared_background_role(self):
        # Literal-drift pin: the producer (consciousness) and the gate
        # (owner_delivery) must share ONE constant, not two literals.
        import inspect

        from ouroboros import consciousness

        src = inspect.getsource(consciousness)
        assert "BACKGROUND_DELEGATION_ROLE" in src
        assert '"delegation_role": "background"' not in src

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

    def test_full_queue_goes_sticky_deferred(self):
        broken = _Queue(full=True)
        ctx = _ctx(event_queue=broken)
        first = deliver_owner_event(ctx, {"type": "send_message", "chat_id": 1, "text": "a"})
        assert first == "deferred"
        # The queue recovers, but ordering wins: the task stays deferred so
        # frame B cannot overtake the already-buffered frame A.
        ctx.event_queue = _Queue()
        second = deliver_owner_event(ctx, {"type": "send_message", "chat_id": 1, "text": "b"})
        assert second == "deferred"
        assert [e["text"] for e in ctx.pending_events] == ["a", "b"]
        assert ctx.event_queue.items == []


class TestSendToolsLiveIntegration:
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
        ctx = _ctx(event_queue=q)
        ctx.drive_logs = lambda: tmp_path
        assert "sent to owner chat" in _send_user_message(ctx, "ping")
        frame = q.items[0]
        # Replay safety: an untyped assistant row with a task_id reads as the
        # task's final on history reload; the discriminator (persisted via
        # log_chat record_type) keeps the live card open.
        assert frame["system_type"] == "proactive_message"

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
        ctx = _ctx(event_queue=q)
        assert "sent to owner chat" in _send_video(ctx, file_path=str(vid))
        assert "sent to owner chat" in _send_file(ctx, file_path=str(doc))
        assert "sent to owner chat" in _send_links(
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
