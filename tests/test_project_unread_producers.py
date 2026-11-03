"""Project unread counts conversation messages, not task-card content.

The durable ``visible_revision`` advances once per new standalone message in
the Project's conversation feed: an ordinary or proactive root reply, a root
final, a question, a standalone System row (a root's terminal incident
included), and delivered media, files and links. A row the host places inside a
task card (``card_row``, a child's terminal incident included), a child task's
own message (its delegation lineage, host-enriched at delivery) and every
progress row — narration, lifecycle, typed incidents — change a card, not the
conversation, and leave the revision alone. Where a row is shown decides, never
its kind (DESIGN "Unread").

Every case runs the real delivery seam (``events_chat_delivery._handle_send_message``
or the bridge's media sends) against a real registered Project, so the counter
under test is the one the sidebar reads.
"""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

from ouroboros.owner_quiz import quiz_states, record_asked
from ouroboros.projects_registry import create_project, get_project
from ouroboros.utils import append_jsonl
from ouroboros import dialogue_store as ds
from tests._dialogue import chat_rows, progress_rows
from supervisor import events_chat_delivery as delivery
from supervisor import message_bus

CHILD = {"id": "kid-1", "delegation_role": "subagent", "parent_task_id": "root-1",
         "root_task_id": "root-1", "subagent_role": "researcher"}
ROOT = {"id": "root-1"}


@pytest.fixture
def room(tmp_path, monkeypatch):
    project = create_project(tmp_path, "racer")
    chat_id = int(project["chat_id"])
    bridge = message_bus.LocalChatBridge({})
    frames = []
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 7, "session_id": "s"})
    monkeypatch.setattr(message_bus, "publish_event", lambda *_a, **_k: None)
    monkeypatch.setattr(delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    host = SimpleNamespace(
        DRIVE_ROOT=tmp_path, append_jsonl=append_jsonl, bridge=bridge,
        send_with_budget=message_bus.send_with_budget,
        RUNNING={task["id"]: {"task": dict(task)} for task in (CHILD, ROOT)},
    )

    def revision() -> int:
        return int(get_project(tmp_path, "racer")["visible_revision"])

    def deliver(task_id: str, text: str = "words", **fields) -> int:
        before = revision()
        delivery._handle_send_message({"type": "send_message", "chat_id": chat_id,
                                       "task_id": task_id, "text": text, **fields}, host)
        return revision() - before

    return SimpleNamespace(chat_id=chat_id, bridge=bridge, host=host, frames=frames,
                           revision=revision, deliver=deliver, root=tmp_path)


@pytest.mark.parametrize("task_id, fields", [
    pytest.param("root-1", {}, id="root-final"),
    pytest.param("root-1", {"system_type": "proactive_message", "format": "markdown"},
                 id="root-send-user-message"),
    pytest.param("root-1", {"role": "system", "system_type": "cancel_receipt"},
                 id="root-system-row-without-placement"),
    pytest.param("", {"role": "system", "system_type": "task_admission_notice"},
                 id="standalone-system-row"),
])
def test_conversation_messages_advance_unread_once(room, task_id, fields):
    assert room.deliver(task_id, **fields) == 1


@pytest.mark.parametrize("task_id, fields", [
    pytest.param("root-1", {"is_progress": True, "progress_meta": {"narration": True}},
                 id="root-narration"),
    pytest.param("kid-1", {"is_progress": True}, id="child-narration"),
    pytest.param("kid-1", {"is_progress": True, "progress_meta": {
        "subagent_event": "scheduled", "delegation_role": "subagent", "parent_task_id": "root-1"}},
        id="child-lifecycle"),
    pytest.param("kid-1", {}, id="child-final"),
    pytest.param("kid-1", {"system_type": "proactive_message", "format": "markdown"},
                 id="child-send-user-message"),
    pytest.param("root-1", {"is_progress": True, "progress_meta": {
        "task_incident": "worker_lost", "toast_once": "root-1:worker_lost"}},
        id="typed-incident-progress"),
    pytest.param("root-1", {"role": "system", "system_type": "custody_notice", "progress_meta": {
        "card_row": "timeline", "card_row_id": "final:root-1:x:custody_notice"}},
        id="custody-card-row"),
    pytest.param("root-1", {"role": "system", "system_type": "acceptance_late_settlement",
                            "progress_meta": {"card_row": "reviews", "card_row_id": "acceptance-late:k"}},
                 id="late-review-card-row"),
    pytest.param("never-seen", {"role": "system", "system_type": "custody_notice", "progress_meta": {
        "card_row": "timeline", "card_row_id": "final:never-seen:x:custody_notice"}},
        id="card-row-whose-card-is-not-loaded"),
])
def test_task_card_content_leaves_unread_alone(room, task_id, fields):
    assert room.deliver(task_id, **fields) == 0
    assert any(row.get("text") == "words" for row in chat_rows(room.root) + progress_rows(room.root)), \
        "the content itself is still delivered and persisted"


def salvage_receipt(room, task: dict) -> dict:
    """The host-salvage ``terminal_incident`` receipt, as the real producer builds it."""
    from supervisor.terminal_delivery import project_terminal_result_event

    event = project_terminal_result_event(
        room.root, task, task["id"], result_text="raw intermediate output", terminal_origin="host_salvage",
        base_event={"type": "send_message", "chat_id": room.chat_id, "task_id": task["id"],
                    "text": "raw intermediate output"})
    assert event["role"] == "system" and event["system_type"] == "terminal_incident"
    return event


@pytest.mark.parametrize("task, advance", [
    pytest.param(ROOT, 1, id="root-receipt-is-a-standalone-system-message"),
    pytest.param(CHILD, 0, id="child-receipt-is-placed-in-its-card"),
])
def test_a_terminal_incident_counts_where_it_is_shown_not_by_its_kind(room, task, advance):
    event = salvage_receipt(room, task)
    before = room.revision()
    delivery._handle_send_message(event, room.host)
    assert room.revision() - before == advance
    (live,) = [frame for frame in room.frames if frame.get("type") == "chat"]
    assert live["system_type"] == "terminal_incident" and bool(live.get("card_row")) is (advance == 0), \
        "the root's receipt reaches the feed alone; the child's names its card"


def test_a_root_provider_death_notice_is_a_standalone_system_message(room):
    from ouroboros.task_finalization import send_provider_death_notice

    before = room.revision()
    assert send_provider_death_notice(room.host, room.chat_id, "root-1",
                                      {"terminal_provider_notice": "Provider outcome unknown."})
    assert room.revision() - before == 1
    (live,) = [frame for frame in room.frames if frame.get("type") == "chat"]
    assert live["role"] == "system" and live["system_type"] == "terminal_incident" and "card_row" not in live


def test_child_final_after_its_running_row_is_gone_stays_card_content(room):
    """Late delivery recovers the child's lineage from its durable result."""
    from ouroboros.task_results import write_task_result

    room.host.RUNNING.clear()
    write_task_result(room.root, "kid-1", "completed", **{k: v for k, v in CHILD.items() if k != "id"})
    assert room.deliver("kid-1", "late child answer") == 0
    assert room.deliver("root-1", "the root's own answer") == 1


def test_media_links_and_questions_are_conversation_messages(room):
    """A question, a photo, a video, a file and a link card are standalone feed bubbles
    whoever produced them, so each advances unread exactly once."""
    before = room.revision()
    assert room.bridge.send_photo(room.chat_id, b"png", caption="shot", task_id="kid-1")[0]
    assert room.bridge.send_video(room.chat_id, b"mp4", caption="clip", task_id="root-1")[0]
    assert room.bridge.send_document(room.chat_id, b"csv", filename="r.csv", task_id="root-1")[0]
    assert room.bridge.send_links(room.chat_id, [{"label": "Docs", "url": "https://example.com"}],
                                  title="Links", task_id="root-1")[0]
    record_asked(room.root, "root-1", quiz_id="q1", question="Merge now?", options=["Yes", "No"],
                 stake="release timing", assumption="continuing with the merge", chat_id=room.chat_id)
    assert room.bridge.send_quiz(room.chat_id, quiz_id="q1", question="Merge now?",
                                 options=[{"label": "Yes"}, {"label": "No"}], stake="release timing",
                                 assumption="continuing with the merge", task_id="root-1") == (True, "ok")
    assert room.revision() - before == 5


def test_empty_text_and_other_rooms_never_advance(room, tmp_path):
    create_project(tmp_path, "other")
    assert room.deliver("root-1", "   ") == 0
    message_bus.send_with_budget(1, "a Main reply", task_id="root-1")
    assert room.revision() == 0


def test_every_counted_message_is_stored_before_its_revision_advances(room, monkeypatch):
    """A client that sees the new revision reads history begun after it, so the row the revision
    counts must already be stored: otherwise that read could paint without it and be acknowledged."""
    append = ds.append
    stored = []
    registry_before = (room.root / "state/projects.json").read_bytes()

    def observed(root, stream, row, **kwargs):
        receipt = append(root, stream, row, **kwargs)
        if stream == "chat" and kwargs.get("feed"):
            assert ds.room(root, room.chat_id)["revision"] == len(chat_rows(root))
            stored.append(len(chat_rows(root)))
        return receipt

    monkeypatch.setattr(ds, "append", observed)
    assert room.deliver("root-1", "the root's own answer") == 1
    assert room.bridge.send_photo(room.chat_id, b"png", caption="shot", task_id="root-1")[0]
    assert room.bridge.send_video(room.chat_id, b"mp4", caption="clip", task_id="root-1")[0]
    assert room.bridge.send_document(room.chat_id, b"csv", filename="r.csv", task_id="root-1")[0]
    assert room.bridge.send_links(room.chat_id, [{"label": "Docs", "url": "https://example.com"}],
                                  title="Links", task_id="root-1")[0]
    record_asked(room.root, "root-1", quiz_id="q1", question="Merge now?", options=["Yes", "No"],
                 stake="release timing", assumption="continuing with the merge", chat_id=room.chat_id)
    assert room.bridge.send_quiz(room.chat_id, quiz_id="q1", question="Merge now?",
                                 options=[{"label": "Yes"}, {"label": "No"}], stake="release timing",
                                 assumption="continuing with the merge", task_id="root-1") == (True, "ok")
    assert stored == [1, 2, 3, 4, 5, 6], "row and revision commit together"
    assert (room.root / "state/projects.json").read_bytes() == registry_before


@pytest.mark.parametrize("kind", ["text", "photo", "video", "document", "links", "quiz"])
def test_failed_canonical_append_never_mints_a_phantom_revision(room, monkeypatch, kind):
    """Unread describes stored conversation, including strict quiz publication.

    Ordinary unbound sends retain best-effort live delivery; a question remains
    owed until its canonical row is durable, before any live exposure.
    """
    assert room.deliver("root-1", "stored answer") == 1
    initial = chat_rows(room.root)
    append = ds.append
    if kind == "quiz":
        record_asked(room.root, "root-1", quiz_id="write-q", question="Continue?", options=["Yes", "No"],
                     stake="next step", assumption="continue", chat_id=room.chat_id)
    send = {
        "text": lambda: message_bus.send_with_budget(room.chat_id, "new reply", task_id="root-1"),
        "photo": lambda: room.bridge.send_photo(room.chat_id, b"png", caption="new photo"),
        "video": lambda: room.bridge.send_video(room.chat_id, b"mp4", caption="new video"),
        "document": lambda: room.bridge.send_document(room.chat_id, b"csv", filename="new.csv"),
        "links": lambda: room.bridge.send_links(room.chat_id, [{"label": "Docs", "url": "https://example.com"}]),
        "quiz": lambda: room.bridge.send_quiz(room.chat_id, quiz_id="write-q", question="Continue?",
                                               options=[{"label": "Yes"}, {"label": "No"}], stake="next step",
                                               assumption="continue", task_id="root-1"),
    }[kind]
    def refuse(*args, **kwargs):
        raise ds.StoreError("dialogue write failed")

    monkeypatch.setattr(ds, "append", refuse)
    before_frames = len(room.frames)
    if kind == "quiz":
        with pytest.raises(RuntimeError, match="canonical message acceptance could not be persisted"):
            send()
        assert len(room.frames) == before_frames, "no quiz or Main pointer precedes its canonical row"
        block = quiz_states(room.root, "root-1")["write-q"]
        assert block["publication_chat_id"] == room.chat_id
        assert block["delivery"]["state"] == "owed" and not block.get("published_at")
    else:
        result = send()
        # The text sender answers with the room it selected (an additive return, PR #2150); media senders keep (True, "ok").
        assert (result == {"chat_id": room.chat_id} if kind == "text" else result == (True, "ok")), result
        assert len(room.frames) > before_frames, "existing best-effort live delivery stays available"
    assert chat_rows(room.root) == initial
    assert room.revision() == 1, "the missing canonical row cannot be acknowledged as a new revision"
    monkeypatch.setattr(ds, "append", append)
    send()
    assert len(chat_rows(room.root)) == 2
    assert room.revision() == 2, "a later successful write still advances exactly once"
    if kind == "quiz":
        assert quiz_states(room.root, "root-1")["write-q"]["delivery"]["state"] == "delivered"
        assert len(room.frames) > before_frames
