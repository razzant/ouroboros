"""A room's pages are counted in its own rows (owner decisions 2026-10-05, 1A).

Opening a room returns its newest rows however many other rooms' archives were
written since; every older page returns the next older rows of the same room and
is never empty while history remains; a room's rows are addressed out of the store
(``RoomLens``), never read out of other rooms' history.
"""

import pytest

from ouroboros.gateway import history, history_paging
from tests.test_chat_history_paging import isolated_runtime, pages, request, row, write  # noqa: F401


@pytest.mark.parametrize("project_room", [False, True])
@pytest.mark.parametrize("source", ["chat", "progress"])
@pytest.mark.parametrize("in_span", [False, True])
def test_import_gap_follows_delivered_span_after_quota_deferral(tmp_path, project_room, source, in_span):
    from ouroboros.projects_registry import create_project

    chat_id = create_project(tmp_path, "gap-room", name="Gap room")["chat_id"] if project_room else 1
    path = tmp_path / "logs" / f"{source}.jsonl"
    write(path, [row(index, source, chat_id=chat_id, ts=f"2026-09-12T00:00:{index:02d}Z")
                 for index in range(20)])
    path.write_bytes(b"{broken\n" + path.read_bytes() + (b"{broken\n" if in_span else b""))
    if source == "progress":
        write(tmp_path / "logs" / "chat.jsonl", [row(0, chat_id=chat_id)])
    params = {"chat_id": str(chat_id), "n_human": "2", "n_progress": "2"}
    status, recent = request(tmp_path, **params)
    assert status == 200
    span = recent["coverage"]["spans"][source]
    assert span["from"] > len(b"{broken\n")
    assert span["gaps"] == (["malformed_jsonl"] if in_span else [])
    assert (f"{source}_malformed_jsonl" in recent["window"]["truncated_by"]) is in_span
    status, replay = request(tmp_path, cursor=recent["page_cursor"], **params)
    assert status == 200 and replay["messages"] == recent["messages"]
    assert replay["coverage"] == recent["coverage"]
    loaded = list(pages(tmp_path, **{**params, "n_human": "0" if source == "progress" else "2"}))
    assert any("malformed_jsonl" in page["coverage"]["spans"][source]["gaps"] for page in loaded)


def test_lineage_deferral_also_recomputes_gap_range(tmp_path):
    path = tmp_path / "logs" / "progress.jsonl"
    write(path, [row(index, "progress", delegation_role="subagent", parent_task_id="parent",
                     task_id="child", subagent_event="running") for index in range(310)])
    path.write_bytes(b"{broken\n" + path.read_bytes())
    status, page = request(tmp_path)
    assert status == 200 and len(page["messages"]) == 300
    assert "lineage_cap" in page["window"]["truncated_by"]
    assert page["coverage"]["spans"]["progress"]["from"] > len(b"{broken\n")
    assert page["coverage"]["spans"]["progress"]["gaps"] == []


@pytest.mark.parametrize("failure", [history_paging.StoreMigrating, OSError])
def test_final_gap_read_failure_remains_retryable_endpoint_error(tmp_path, monkeypatch, failure):
    write(tmp_path / "logs" / "progress.jsonl", [row(i, "progress") for i in range(20)])
    original, calls = history_paging.HistorySource.gaps, []

    def unavailable(self, lower, upper):
        if self.source == "progress" and lower > 0:
            calls.append((lower, upper))
            raise failure("controlled final-range read failure")
        return original(self, lower, upper)

    monkeypatch.setattr(history_paging.HistorySource, "gaps", unavailable)
    status, failed = request(tmp_path, n_progress="2")
    assert calls and status == 503
    assert failed["reason_code"] == ("migrating" if failure is history_paging.StoreMigrating
                                     else "history_source_unavailable")
    assert failed["window"]["complete"] is False and failed["has_more"] is True
    monkeypatch.setattr(history_paging.HistorySource, "gaps", original)
    status, recovered = request(tmp_path, n_progress="2")
    assert status == 200 and len(recovered["messages"]) == 2


def test_narrowed_import_gap_does_not_clear_projection_failure(tmp_path, monkeypatch):
    write(tmp_path / "logs" / "progress.jsonl", [row(i, "progress") for i in range(20)])
    original = history._collect_progress_rows

    def partial(*args, **kwargs):
        rows, count, gaps = original(*args, **kwargs)
        return rows, count, gaps | {history_paging.PROJECTION_FAILED}

    monkeypatch.setattr(history, "_collect_progress_rows", partial)
    status, page = request(tmp_path, n_progress="2")
    assert status == 200 and page["coverage"]["spans"]["progress"]["from"] > 0
    assert page["coverage"]["spans"]["progress"]["gaps"] == [history_paging.PROJECTION_FAILED]
    assert "progress_projection_failed" in page["window"]["truncated_by"]
    assert "after_cursor" not in page


def test_gap_refresh_only_reads_changed_spans_and_preserves_other_evidence(tmp_path, monkeypatch):
    calls = []

    class Reader:
        def __init__(self, root, source, lens, upper):
            assert root == tmp_path and source == "chat" and lens == "room" and upper == 100

        def gaps(self, lower, upper):
            calls.append((lower, upper))
            return {"invalid_jsonl_row"}

    monkeypatch.setattr(history_paging, "HistorySource", Reader)
    page = {"before": {"chat": 40, "progress": 10}, "selections": {"chat": ([], 0, set()),
            "progress": ([], 10, set())}, "lens": {"chat": "room"}, "upper": {"chat": 100},
            "page_ends": {"chat": 80}}
    gaps = {"chat": {"malformed_jsonl", "segment_unimportable", "projection_failed", "other"},
            "progress": {"malformed_jsonl"}}
    result = history_paging.delivered_history_gaps(tmp_path, page, gaps)
    assert calls == [(40, 80)]
    assert result == {"chat": {"invalid_jsonl_row", "projection_failed", "other"},
                      "progress": {"malformed_jsonl"}}


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
    monkeypatch.setattr(history_paging, "_READ_CEILING_ROWS", 1)
    loaded = list(pages(tmp_path, chat_id=str(chat)))
    assert len(loaded) > 2 and loaded[0]["window"]["complete"] is False
    assert sorted(message["text"] for page in loaded for message in page["messages"]) == sorted(
        f"human-{index}" for index in range(40))


def test_a_replayed_page_decodes_only_the_rooms_rows(tmp_path, monkeypatch):
    """A page re-read by its frozen handle (a released page coming back) costs what its
    first read did: the same room lens addresses the same rows, no foreign archive row is decoded."""
    from ouroboros.projects_registry import create_project

    project = create_project(tmp_path, "replay", name="Replay")
    chat = project["chat_id"]
    own = tmp_path / "archive" / "chat_20260901T000000.jsonl"
    write(own, [row(index, chat_id=chat) for index in range(3)])
    foreign_archives(tmp_path, "20260902T00000", count=4)
    write(tmp_path / "logs" / "chat.jsonl", [row(9, text="foreign live")])
    status, first = request(tmp_path, chat_id=str(chat))
    assert status == 200 and len(first["messages"]) == 3
    decoded, entries = [], history_paging.HistorySource._entries
    monkeypatch.setattr(history_paging.HistorySource, "_entries", lambda self, rows: (
        decoded.extend((self.source, entry.seq) for entry in rows), entries(self, rows))[1])
    status, again = request(tmp_path, chat_id=str(chat), cursor=first["page_cursor"])
    assert status == 200
    assert [message["history_id"] for message in again["messages"]] == [
        message["history_id"] for message in first["messages"]]
    assert decoded and all(seq < own.stat().st_size for source, seq in decoded if source == "chat"), decoded
