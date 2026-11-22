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


class TestShowRecord:
    """A send of produced material records on its receipt what it put up and which reviews on record name those
    exact bytes: a host fact, never a gate."""

    @staticmethod
    def _ctx(tmp_path, monkeypatch, review="auto"):
        from ouroboros import config

        monkeypatch.setattr(config, "get_task_review_mode", lambda: review)
        root = tmp_path / "data"
        root.mkdir(parents=True, exist_ok=True)
        return types.SimpleNamespace(current_chat_id=123, pending_events=[], drive_root=root, budget_drive_root=root,
                                     task_id="t-show", task_metadata={}, event_queue=_Queue())

    def test_a_shown_file_names_its_bytes_and_each_review_on_record_that_names_them(self, tmp_path, monkeypatch):
        """Both directions: with no record naming the bytes the line says none; a result review whose retained
        subject names them and a plan ask that attached them are then named by id, the review with its state."""
        import hashlib
        from ouroboros import plan_record, review_ledger as ledger
        from ouroboros.result_material import answer_text
        from ouroboros.tools.core import _send_file
        from ouroboros.tools.review_result import subject_source

        ctx = self._ctx(tmp_path, monkeypatch)
        doc = tmp_path / "mockup.html"
        doc.write_bytes(b"<p>dock</p>")
        digest = hashlib.sha256(b"<p>dock</p>").hexdigest()
        receipt, show = _send_file(ctx, file_path=str(doc)).split("\n")
        assert receipt == "OK: file 'mockup.html' accepted for delivery to owner chat."
        assert show == (f"Host fact: accepted for delivery: mockup.html, sha256 {digest[:12]}, 11 B. "
                        "Reviews on record naming these bytes: none in this task's recent records.")

        subject = "## What was asked\n\nA dock mockup.\n\n" + answer_text(reply="Here it is.", produced={"files": [
            {"path": str(doc), "kind": "artifact", "sha256": digest, "size": 11, "retained": "a copy"}]},
            verification="none recorded")
        subject_source(ctx, subject)
        record_id = ledger.new_record_id()
        ledger.write_record(ctx.drive_root, ledger.ReviewLedgerRecord(
            record_id=record_id, state="settled", task_id="t-show", surface="result",
            subject={"diff_sha": hashlib.sha256(subject.encode("utf-8")).hexdigest()}, verdict={"aggregate": "PASS"}))
        plan_record.append_ask(ctx.drive_root, "t-show", {"ask_id": "a1", "revision": None, "participants": [],
                                                          "attached": [{"locator": "mockup.html", "sha256": digest}]})
        assert _send_file(ctx, file_path=str(doc)).split("\n")[1].endswith(
            f"Reviews on record naming these bytes: result review {record_id} (settled, PASS); plan ask a1.")
        other = tmp_path / "other.html"
        other.write_bytes(b"<p>other</p>")
        assert _send_file(ctx, file_path=str(other)).endswith(
            "Reviews on record naming these bytes: none in this task's recent records.")

    @pytest.mark.parametrize("name, label", [("report-雪.txt", "report-雪.txt"),
                                              ("report-\udcff.txt", r'"report-\udcff.txt"')], ids=["utf8", "byte-name"])
    def test_local_document_acceptance_retains_bytes_and_renders_name(self, tmp_path, monkeypatch, name, label):
        import errno
        import hashlib
        import os

        from ouroboros.gateway.files import resolve_task_file_reference
        from ouroboros.tools.core_artifacts import _send_file

        if "\udcff" in name and os.name != "posix":
            pytest.skip("POSIX byte filename")
        ctx = self._ctx(tmp_path, monkeypatch)
        doc, data = tmp_path / name, b"A readable document.\n"
        try:
            doc.write_bytes(data)
        except OSError as exc:
            if "\udcff" in name and exc.errno in {errno.EILSEQ, errno.EINVAL}:
                pytest.skip("fixture filesystem does not create non-UTF-8 names")
            raise
        caption = r"Exact caption: \udcff."
        text = _send_file(ctx, file_path=str(doc), caption=caption)
        text.encode("utf-8")
        receipt, show = text.split("\n")
        assert receipt == f"OK: file '{label}' accepted for delivery to owner chat."
        assert show.startswith(f"Host fact: accepted for delivery: {label}, sha256 {hashlib.sha256(data).hexdigest()[:12]}")
        [event] = ctx.event_queue.items  # fake sink, no real recipient
        assert ctx._owner_message_sent and not ctx.pending_events
        assert event["filename"] == name and event["caption"] == caption
        assert resolve_task_file_reference(ctx.drive_root, ctx.task_id, event["file_ref"]).read_bytes() == data

    def test_material_fact_distinguishes_byte_and_literal_labels_without_filesystem_assumptions(self, tmp_path, monkeypatch):
        from ouroboros.tools.owner_delivery import show_record, shown_material

        ctx = self._ctx(tmp_path, monkeypatch)
        pieces = [shown_material("report-\udcff.txt", b"A"), shown_material(r"report-\udcff.txt", b"B")]
        receipt = show_record(ctx, "live", pieces)
        receipt.encode("utf-8")
        assert r'"report-\udcff.txt", sha256 ' in receipt and r'; report-\udcff.txt, sha256 ' in receipt
        assert pieces[0]["name"] == "report-\udcff.txt"

    def test_a_negative_names_its_scope_and_a_bounded_read_that_came_back_full_is_unknown(self, tmp_path, monkeypatch):
        """Both directions: a helper's review that names the bytes is named on the helper's own send, while its root's
        send says none in its own recent records, never a bare none; a bounded read that came back full claims nothing
        beyond it: unknown."""
        import copy
        import hashlib
        from ouroboros import review_ledger as ledger
        from ouroboros.tools.owner_delivery import show_record, shown_material
        from ouroboros.tools.review_result import subject_source

        ctx = self._ctx(tmp_path, monkeypatch)
        helper = copy.copy(ctx)
        helper.task_id = "t-helper"
        item = shown_material("preview.html", b"<p>Done</p>")
        subject = f"Produced preview.html, sha256 {item['sha256']}"
        subject_source(helper, subject)
        record_id = ledger.new_record_id()
        ledger.write_record(ctx.drive_root, ledger.ReviewLedgerRecord(
            record_id=record_id, state="settled", task_id=helper.task_id, root_task_id=ctx.task_id, surface="result",
            subject={"diff_sha": hashlib.sha256(subject.encode("utf-8")).hexdigest()}, verdict={"aggregate": "PASS"}))
        assert show_record(helper, "live", [item]).endswith(f"result review {record_id} (settled, PASS).")
        assert show_record(ctx, "live", [item]).endswith(
            "Reviews on record naming these bytes: none in this task's recent records.")
        full = [{"task_id": f"t-helper-{index}", "record_id": f"r{index}"} for index in range(50)]
        monkeypatch.setattr(ledger, "recent_records", lambda *_args, **_kwargs: full)
        assert show_record(ctx, "live", [item]).endswith("Reviews on record naming these bytes: unknown.")

    def test_a_torn_hot_index_cannot_prove_that_no_review_named_the_bytes(self, tmp_path, monkeypatch):
        import hashlib

        from ouroboros import review_ledger as ledger
        from ouroboros.tools.owner_delivery import show_record, shown_material
        from ouroboros.tools.review_result import subject_source

        ctx = self._ctx(tmp_path, monkeypatch)
        item = shown_material("drawing.txt", b"A bridge drawing")
        index = ledger.index_path(ctx.drive_root)
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_bytes(b"")
        assert show_record(ctx, "live", [item]).endswith(
            "Reviews on record naming these bytes: none in this task's recent records.")

        subject = f"Produced drawing.txt, sha256 {item['sha256']}"
        subject_source(ctx, subject)
        record_id = ledger.new_record_id()
        ledger.write_record(ctx.drive_root, ledger.ReviewLedgerRecord(
            record_id=record_id, state="settled", task_id=ctx.task_id, surface="result",
            subject={"diff_sha": hashlib.sha256(subject.encode()).hexdigest()}, verdict={"aggregate": "PASS"}))
        with index.open("ab") as tail:
            tail.write(b'{"record_id":"unfinished-review"')
        assert show_record(ctx, "live", [item]).endswith(f"result review {record_id} (settled, PASS).")
        other = shown_material("bridge-notes.txt", b"The bridge notes")
        assert show_record(ctx, "live", [other]).endswith("Reviews on record naming these bytes: unknown.")

    def test_no_line_without_material_or_under_review_off_and_unknown_when_records_cannot_be_read(
            self, tmp_path, monkeypatch):
        """Both directions: links carry no material and an install with task review Off records no show; an app
        card's page is named by its bytes; a record that cannot be read makes the reviews unknown, never none."""
        import hashlib
        import json
        from ouroboros import review_ledger as ledger
        from ouroboros.tools.cards import _show_card
        from ouroboros.tools.core import _send_links, _send_photo

        image = tmp_path / "shot.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 200)
        ctx = self._ctx(tmp_path, monkeypatch)
        assert "Host fact" not in _send_links(ctx, links=[{"label": "Docs", "url": "https://example.com/d"}])
        page = "<!doctype html><button>Roll</button>"
        card = json.loads(_show_card(ctx, title="Dice", fallback_text="A dice roller", app={"html": page}))
        assert card["delivery"].split("\n")[1].startswith(
            f"Host fact: accepted for delivery: app page, sha256 {hashlib.sha256(page.encode()).hexdigest()[:12]}, "
            f"{len(page.encode())} B.")

        def busy(*_args, **_kwargs):
            raise OSError("index busy")

        monkeypatch.setattr(ledger, "recent_records", busy)
        assert _send_photo(ctx, file_path=str(image)).endswith("Reviews on record naming these bytes: unknown.")
        off = self._ctx(tmp_path / "off", monkeypatch, review="off")
        assert _send_photo(off, file_path=str(image)) == "OK: photo accepted for delivery to owner chat."

    def test_a_card_names_the_bytes_its_renderer_draws(self, tmp_path, monkeypatch):
        """Both directions: an image whose path reads the card's data is named by those bytes, never by the src the
        renderer passes over, and so is a gallery drawn from data; a literal src, a path that reads nothing and a src
        aimed at another target (a card holds no other state) are named by their src."""
        import base64
        import hashlib
        import json
        from ouroboros.tools.cards import _show_card

        ctx = self._ctx(tmp_path, monkeypatch)
        first, second = (b"\x89PNG\r\n\x1a\n" + fill * 200 for fill in (b"A", b"B"))
        short = {data: hashlib.sha256(data).hexdigest()[:12] for data in (first, second)}

        def uri(data):
            return "data:image/png;base64," + base64.b64encode(data).decode()

        def shown(*components, data=None):
            delivery = json.loads(_show_card(ctx, title="Preview", fallback_text="Image", components=list(components),
                                             data=data))["delivery"]
            return delivery.partition("\n")[2]

        bound = shown({"type": "image", "src": uri(first), "path": "hero"}, data={"hero": uri(second)})
        assert short[second] in bound and short[first] not in bound
        assert short[second] in shown({"type": "image", "path": "hero"}, data={"hero": uri(second)})
        gallery = shown({"type": "gallery", "path": "shots"}, data={"shots": [{"src": uri(first)}, {"src": uri(second)}]})
        assert short[first] in gallery and short[second] in gallery
        assert short[first] in shown({"type": "image", "src": uri(first)})
        unread = shown({"type": "image", "src": uri(first), "path": "missing"}, data={"hero": uri(second)})
        assert short[first] in unread and short[second] not in unread
        aimed = shown({"type": "image", "src": uri(first), "path": "hero", "target": "elsewhere"},
                      data={"hero": uri(second)})
        assert short[first] in aimed and short[second] not in aimed

    @pytest.mark.parametrize(("path", "source"), [
        (None, "inline"),
        ("photos.url", "inline"),
        ("photos.0.url", "bound"),
        ("photos.map", "unknown"),
        ("photos.toString", "unknown"),
    ], ids=["inline", "missing-array-property", "bound-index", "array-method", "object-method"])
    def test_a_card_receipt_distinguishes_absent_and_inherited_array_properties(self, tmp_path, monkeypatch, path, source):
        import base64
        import hashlib
        import json

        from ouroboros.tools.cards import _show_card

        ctx = self._ctx(tmp_path, monkeypatch)
        images = {"inline": b"Fallback preview", "bound": b"Bound preview"}
        urls = {name: "data:image/png;base64," + base64.b64encode(data).decode("ascii") for name, data in images.items()}
        image = {"type": "image", "src": urls["inline"]}
        if path is not None:
            image["path"] = path
        result = json.loads(_show_card(ctx, title="Preview", fallback_text="Photo", components=[image],
                                       data={"photos": [{"url": urls["bound"]}]}))
        [event] = ctx.event_queue.items
        assert event["type"] == "send_card" and event["card_id"] == result["card_id"]
        accepted, show = result["delivery"].splitlines()
        assert accepted == "OK: card accepted for delivery to owner chat."
        if source == "unknown":
            assert show == "Host fact: card media bytes: unknown."
        else:
            digest = hashlib.sha256(images[source]).hexdigest()[:12]
            assert show.startswith(f"Host fact: accepted for delivery: image components[0], sha256 {digest},")

    @pytest.mark.parametrize(("visible", "review"), [(False, "auto"), (False, "off"), (True, "auto")])
    def test_optional_media_data_cannot_break_an_accepted_card_receipt(self, tmp_path, monkeypatch, visible, review):
        import json

        from ouroboros.card_store import load_card
        from ouroboros.tools.cards import _show_card

        ctx = self._ctx(tmp_path, monkeypatch, review=review)
        image = "data:image/png;base64,aGVyZQ=="
        result = json.loads(_show_card(ctx, title="Optional preview", fallback_text="One preview", components=[
            {"type": "image", "path": "optional", "condition_key": "visible"},
            {"type": "image", "src": image},
        ], data={"visible": visible, "optional": 10**310}))
        [event] = ctx.event_queue.items
        assert event["type"] == "send_card" and event["card_id"] == result["card_id"]
        assert load_card(ctx.drive_root, result["card_id"])
        assert result["delivery"].splitlines()[0] == "OK: card accepted for delivery to owner chat."
        assert ("image components[1], sha256" in result["delivery"]) == (review == "auto")
        assert "image components[0]" not in result["delivery"]
        assert ("Other card media bytes: unknown." in result["delivery"]) == (visible and review == "auto")

    def test_a_show_collector_failure_leaves_the_card_accepted(self, tmp_path, monkeypatch):
        import json

        from ouroboros.card_store import load_card
        from ouroboros.tools import cards

        ctx = self._ctx(tmp_path, monkeypatch)
        card = {"title": "Preview", "fallback_text": "A drawing",
                "components": [{"type": "image", "src": "data:image/png;base64,aGVyZQ=="}]}
        healthy = json.loads(cards._show_card(ctx, **card))
        assert "image components[0], sha256" in healthy["delivery"]

        def unreadable(*_args, **_kwargs):
            raise ValueError("unreadable card media")

        monkeypatch.setattr(cards, "_shown_media", unreadable)
        receipt = json.loads(cards._show_card(ctx, **card))
        assert receipt["delivery"].splitlines() == ["OK: card accepted for delivery to owner chat.",
                                                    "Host fact: card media bytes: unknown."]
        assert load_card(ctx.drive_root, receipt["card_id"])
        assert [event["card_id"] for event in ctx.event_queue.items] == [healthy["card_id"], receipt["card_id"]]

    @pytest.mark.parametrize(("condition", "visible"), [(False, False), ({}, True), ([], True)])
    def test_a_card_receipt_follows_the_visibility_of_its_media(self, tmp_path, monkeypatch, condition, visible):
        import hashlib
        import json

        from ouroboros.tools.cards import _show_card

        ctx = self._ctx(tmp_path, monkeypatch)
        digest = hashlib.sha256(b"here").hexdigest()[:12]
        result = json.loads(_show_card(ctx, title="Preview", fallback_text="One preview", components=[
            {"type": "group", "condition_key": "show", "components": [{"type": "image", "path": "preview"}]},
        ], data={"show": condition, "preview": "data:image/png;base64,aGVyZQ=="}))
        assert (f"sha256 {digest}" in result["delivery"]) is visible
        assert ("Host fact:" in result["delivery"]) is visible

    @pytest.mark.parametrize("routed_item", [False, True])
    def test_a_card_receipt_reads_nested_gallery_media_and_honors_route_precedence(self, tmp_path, monkeypatch, routed_item):
        import hashlib
        import json

        from ouroboros.tools.cards import _show_card

        ctx = self._ctx(tmp_path, monkeypatch)
        route_bytes, nested_bytes = b"elsewhere", b"here"
        route_digest, nested_digest = (hashlib.sha256(raw).hexdigest()[:12] for raw in (route_bytes, nested_bytes))
        items = [{"type": "group", "components": [{"type": "image", "path": "preview"}]}]
        if routed_item:
            items.insert(0, {"route": "another-source", "src": "data:image/png;base64,ZWxzZXdoZXJl"})
        result = json.loads(_show_card(ctx, title="Gallery", fallback_text="Gallery preview",
                                       components=[{"type": "gallery", "path": "items"}],
                                       data={"items": items, "preview": "data:image/png;base64,aGVyZQ=="}))
        assert route_digest not in result["delivery"]
        assert f"sha256 {nested_digest}" in result["delivery"]
        assert ("Other card media bytes: unknown." in result["delivery"]) is routed_item

    def test_a_card_receipt_names_the_first_tab_and_reports_unknown_tabs_on_replace(self, tmp_path, monkeypatch):
        import hashlib
        import json

        from ouroboros.tools.cards import _show_card

        ctx = self._ctx(tmp_path, monkeypatch)
        first, second = (hashlib.sha256(raw).hexdigest()[:12] for raw in (b"here", b"elsewhere"))
        card = {"title": "Previews", "fallback_text": "Two previews", "components": [{"type": "tabs", "tabs": [
            {"label": "Summary", "components": [{"type": "image", "src": "data:image/png;base64,aGVyZQ=="}]},
            {"label": "Detail", "components": [{"type": "image", "src": "data:image/png;base64,ZWxzZXdoZXJl"}]},
        ]}]}
        fresh = json.loads(_show_card(ctx, **card))
        assert first in fresh["delivery"] and second not in fresh["delivery"]
        replacement = json.loads(_show_card(ctx, **card, replace=fresh["card_id"]))
        assert replacement["card_id"] == fresh["card_id"] and replacement["revision"] == 2
        assert replacement["delivery"].endswith("Host fact: card media bytes: unknown.")
        assert "sha256" not in replacement["delivery"]
