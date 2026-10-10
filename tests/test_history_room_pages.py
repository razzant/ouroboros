"""A room's pages are counted in its own rows (owner decisions 2026-10-05, 1A).

Opening a room returns its newest rows however many other rooms' archives were
written since; every older page returns the next older rows of the same room and
is never empty while history remains; archives a Project's lens rules out are
skipped by their summaries (``history_segments``), never at the cost of a row.
"""

import os

from ouroboros.gateway import history_paging
from ouroboros.gateway.history_segments import SegmentSummary, room_segment_lens, segment_summary
from tests.test_chat_history_paging import isolated_runtime, pages, request, row, write  # noqa: F401


def foreign_archives(root, stamp, count=3):
    for index in range(count):
        write(root / "archive" / f"chat_{stamp}{index}.jsonl",
              [row(item, text="foreign" + "x" * 1500) for item in range(300)])


def test_older_pages_walk_back_in_room_rows_and_are_never_empty(tmp_path):
    from ouroboros.projects_registry import create_project

    project = create_project(tmp_path, "quiet", name="Quiet")
    chat = project["chat_id"]
    write(tmp_path / "archive" / "chat_20260901T000000.jsonl",
          [row(index, chat_id=chat, ts=f"2026-09-01T00:{index // 60:02d}:{index % 60:02d}Z") for index in range(200)])
    foreign_archives(tmp_path, "20260902T00000")
    write(tmp_path / "archive" / "chat_20260903T000000.jsonl",
          [row(200 + index, chat_id=chat, ts=f"2026-09-03T00:{index // 60:02d}:{index % 60:02d}Z")
           for index in range(200)])
    foreign_archives(tmp_path, "20260904T00000")
    write(tmp_path / "logs" / "chat.jsonl", [row(999, text="foreign live")])
    loaded = list(pages(tmp_path, chat_id=str(chat)))
    counts = [len(page["messages"]) for page in loaded]
    assert counts == [150, 150, 100], counts
    seen = [message["text"] for page in loaded for message in page["messages"]]
    assert len(set(seen)) == 400 and "foreign live" not in seen
    for newer, older in zip(loaded, loaded[1:]):
        assert max(m["ts"] for m in older["messages"]) <= min(m["ts"] for m in newer["messages"])
    assert loaded[-1]["has_more"] is False


def test_rows_admitted_by_task_binding_or_origin_are_never_skipped(tmp_path):
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.projects_registry import bind_task_to_project, create_project

    project = create_project(tmp_path, "lens", name="Lens")
    ask = "Please build the lens"
    bind_task_to_project(tmp_path, "root-1", project["id"], origin={
        "ref": build_owner_message_ref(chat_id=1, client_message_id="ask-1", ts="2026-09-01T00:00:00Z", text=ask),
        "text": ask})
    # The owner's row in Main, a bound task's row and its child's row, each in its own
    # archive among foreign ones: none of these archives names the Project's chat id.
    write(tmp_path / "archive" / "chat_20260901T000000.jsonl", [row(
        0, text=ask, client_message_id="ask-1", ts="2026-09-01T00:00:00Z")])
    foreign_archives(tmp_path, "20260901T00001")
    write(tmp_path / "archive" / "chat_20260902T000000.jsonl", [row(
        1, direction="out", text="bound answer", task_id="root-1", ts="2026-09-02T00:00:00Z")])
    foreign_archives(tmp_path, "20260902T00001")
    write(tmp_path / "archive" / "chat_20260903T000000.jsonl", [row(
        2, direction="out", text="child answer", task_id="kid-1", parent_task_id="root-1",
        root_task_id="root-1", ts="2026-09-03T00:00:00Z")])
    foreign_archives(tmp_path, "20260903T00001")
    write(tmp_path / "logs" / "chat.jsonl", [row(9, text="foreign live")])
    status, payload = request(tmp_path, chat_id=str(project["chat_id"]))
    assert status == 200
    texts = [message["text"] for message in payload["messages"]]
    assert {ask, "bound answer", "child answer"} <= set(texts), texts
    origin = next(message for message in payload["messages"] if message["text"] == ask)
    assert not origin.get("origin_projected"), "the canonical row adopts the retained origin"
    assert payload["window"]["complete"] is True


def test_project_progress_beyond_old_archives_opens_with_the_room(tmp_path):
    from ouroboros.projects_registry import create_project

    project = create_project(tmp_path, "narration", name="Narration")
    chat = project["chat_id"]
    write(tmp_path / "archive" / "progress_20260901T000000.jsonl",
          [row(index, "progress", chat_id=chat, task_id="room-task") for index in range(5)])
    for index in range(4):
        write(tmp_path / "archive" / f"progress_20260902T00000{index}.jsonl",
              [row(item, "progress", task_id="foreign", content="busy" + "x" * 1500) for item in range(300)])
    write(tmp_path / "logs" / "progress.jsonl", [row(9, "progress", task_id="foreign")])
    write(tmp_path / "logs" / "chat.jsonl", [row(0, chat_id=chat)])
    status, payload = request(tmp_path, chat_id=str(chat))
    assert status == 200
    narration = [message["text"] for message in payload["messages"] if message.get("is_progress")]
    assert narration == [f"progress-{index}" for index in range(5)]


def test_the_read_ceiling_bounds_each_request_but_loses_nothing(tmp_path, monkeypatch):
    """A request stops at the physical ceiling and its cursor resumes there; the client
    keeps reading until rows land, so a stop is latency, never a gap or a lost row."""
    from ouroboros.projects_registry import create_project

    project = create_project(tmp_path, "ceiling", name="Ceiling")
    chat = project["chat_id"]
    for index in range(4):
        write(tmp_path / "archive" / f"chat_20260901T00000{index}.jsonl",
              [row(index * 10 + item, chat_id=chat, ts=f"2026-09-01T0{index}:00:{item:02d}Z") for item in range(10)])
    write(tmp_path / "logs" / "chat.jsonl", [row(99, text="foreign live")])
    monkeypatch.setattr(history_paging, "_READ_CEILING_BYTES", 1)
    loaded = list(pages(tmp_path, chat_id=str(chat)))
    assert len(loaded) > 2 and loaded[0]["window"]["complete"] is False
    assert sorted(message["text"] for page in loaded for message in page["messages"]) == sorted(
        f"human-{index}" for index in range(40))


def test_main_and_rooms_sharing_its_lens_never_skip_an_archive():
    assert room_segment_lens(1, {7}, [], {"task": 7}) is None
    assert room_segment_lens(42, {7}, [], {"task": 7}) is None
    may_hold = room_segment_lens(7, {7}, [], {"task": 7, "other": 8})
    assert may_hold(SegmentSummary(frozenset({7}), frozenset(), frozenset()))
    assert may_hold(SegmentSummary(frozenset({1}), frozenset({"task"}), frozenset()))
    assert not may_hold(SegmentSummary(frozenset({1, 8}), frozenset({"other"}), frozenset()))


def test_a_rewritten_archive_gets_a_fresh_summary(tmp_path):
    path = tmp_path / "chat_20260901T000000.jsonl"
    write(path, [row(0, chat_id=5)])
    assert segment_summary(path, os.stat(path)).chat_ids == frozenset({5})
    write(path, [row(0, chat_id=6), row(1, chat_id=6)])
    assert segment_summary(path, os.stat(path)).chat_ids == frozenset({6})


def test_a_replayed_page_skips_the_archives_its_first_read_ruled_out(tmp_path, monkeypatch):
    """A page re-read by its frozen handle (a released page coming back) costs what its
    first read did: the same room view rules the same archives out."""
    from ouroboros.projects_registry import create_project

    project = create_project(tmp_path, "replay", name="Replay")
    chat = project["chat_id"]
    own = tmp_path / "archive" / "chat_20260901T000000.jsonl"
    write(own, [row(index, chat_id=chat) for index in range(3)])
    foreign_archives(tmp_path, "20260902T00000", count=4)
    write(tmp_path / "logs" / "chat.jsonl", [row(9, text="foreign live")])
    status, first = request(tmp_path, chat_id=str(chat))
    assert status == 200 and len(first["messages"]) == 3
    parsed, entries = [], history_paging.HistorySource._entries
    monkeypatch.setattr(history_paging.HistorySource, "_entries", lambda self, start, end, gaps: (
        parsed.append((self.source, start, end)), entries(self, start, end, gaps))[1])
    status, again = request(tmp_path, chat_id=str(chat), cursor=first["page_cursor"])
    assert status == 200
    assert [message["history_id"] for message in again["messages"]] == [
        message["history_id"] for message in first["messages"]]
    live_base = sum(path.stat().st_size for path in (tmp_path / "archive").glob("chat_*.jsonl"))
    assert parsed and all(end <= own.stat().st_size or start >= live_base
                          for source, start, end in parsed if source == "chat"), parsed
