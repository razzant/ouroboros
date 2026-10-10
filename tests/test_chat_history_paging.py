"""Complete retained history uses physical continuation and shared projection."""

import asyncio
import json
import os
from types import SimpleNamespace

import pytest

from ouroboros.gateway import history
from ouroboros.gateway.history_paging import HistorySource, decode_cursor, encode_cursor


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    from supervisor import queue, state

    for module, names in ((state, ("DRIVE_ROOT", "STATE_PATH", "STATE_LAST_GOOD_PATH", "STATE_LOCK_PATH")),
                          (queue, ("DRIVE_ROOT", "QUEUE_SNAPSHOT_PATH"))):
        for name in names:
            monkeypatch.setattr(module, name, getattr(module, name))
    state.init(tmp_path)
    queue.init(tmp_path)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(history, "_active_lifecycle_row", lambda _filter: None)
    assert state.DRIVE_ROOT == queue.DRIVE_ROOT == tmp_path


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def row(index, source="chat", **extra):
    return {"ts": "2026-09-12T00:00:00Z", "chat_id": 1,
            **({"direction": "in", "text": f"human-{index}"} if source == "chat"
               else {"content": f"progress-{index}", "task_id": "root"}), **extra}


def request(root, **params):
    response = asyncio.run(history.make_chat_history_endpoint(root)(SimpleNamespace(query_params=params)))
    return response.status_code, json.loads(response.body)


def pages(root, **params):
    previous = set()
    for _ in range(100):
        status, payload = request(root, **params)
        assert status == 200, payload
        yield payload
        if not payload["has_more"]:
            assert payload["next_cursor"] is None
            return
        cursor = payload["next_cursor"]
        assert cursor and cursor not in previous
        previous.add(cursor)
        params["cursor"] = cursor
    pytest.fail("cursor did not reach physical exhaustion")


def test_full_history_beyond_both_caps_and_five_archives_with_equal_timestamps(tmp_path):
    for source, count in (("chat", 360), ("progress", 145)):
        for segment in range(5):
            write(tmp_path / "archive" / f"{source}_20260901T00000{segment}.jsonl",
                  [row(segment * count + index, source) for index in range(count)])
        write(tmp_path / "logs" / f"{source}.jsonl", [row(5 * count, source)])
    loaded = list(pages(tmp_path))
    messages = [message for page in loaded for message in page["messages"]]
    assert len(messages) == 1801 + 726
    assert len({message["history_id"] for message in messages}) == len(messages)
    assert {message["text"] for message in messages} >= {"human-0", "human-1800", "progress-0", "progress-725"}
    assert all(message["history_position"]["source"] in {"chat", "progress"} for message in messages)
    assert loaded[0]["window"]["complete"] is False
    assert loaded[-1]["window"]["complete"] is False  # last page is not the whole history


def test_quota_deferred_backdated_system_row_is_not_skipped(tmp_path):
    write(tmp_path / "logs" / "chat.jsonl", [
        row(0), row(1), row(2, direction="system", type="task_summary", task_id="old",
                             ts="2020-01-01T00:00:00Z", text="backdated final"),
    ])
    loaded = list(pages(tmp_path, n_human="1"))
    assert [message["text"] for message in loaded[0]["messages"]] == ["human-1"]
    assert "backdated final" in {message["text"] for page in loaded for message in page["messages"]}


def test_lineage_cap_keeps_the_newest_rows_and_discloses_the_cap(tmp_path):
    """The conversation decides whether older pages exist: a room holding narration
    alone has none, so a swarm larger than the cap keeps its newest lineage on the
    recent page, and the window says the cap cut it."""
    write(tmp_path / "logs" / "progress.jsonl", [
        row(index, "progress", delegation_role="subagent", parent_task_id="parent",
            task_id="child", subagent_event="running") for index in range(310)
    ])
    loaded = list(pages(tmp_path))
    assert len(loaded) == 1 and len(loaded[0]["messages"]) == 300
    assert loaded[0]["messages"][-1]["text"] == "progress-309"
    assert "lineage_cap" in loaded[0]["window"]["truncated_by"]
    assert all(message["parent_task_id"] == "parent" for message in loaded[0]["messages"])


def test_sparse_project_opens_at_its_rows_without_parsing_other_rooms_archives(tmp_path, monkeypatch):
    from ouroboros.projects_registry import create_project

    project = create_project(tmp_path, "sparse", name="Sparse")
    archives = [tmp_path / "archive" / f"chat_20260901T00000{segment}.jsonl" for segment in range(6)]
    write(archives[0], [row(index, chat_id=project["chat_id"]) for index in range(3)])
    for archive in archives[1:]:
        write(archive, [row(index, text="foreign" + "x" * 2000) for index in range(350)])
    write(tmp_path / "logs" / "chat.jsonl", [row(9, text="foreign live")])
    parsed, entries = [], HistorySource._entries
    monkeypatch.setattr(HistorySource, "_entries", lambda self, start, end, gaps: (
        parsed.append((self.source, start, end)), entries(self, start, end, gaps))[1])
    loaded = list(pages(tmp_path, chat_id=str(project["chat_id"])))
    assert len(loaded) == 1 and loaded[0]["window"]["complete"] is True
    assert [message["text"] for message in loaded[0]["messages"]] == ["human-0", "human-1", "human-2"]
    own_end, live_base = archives[0].stat().st_size, sum(archive.stat().st_size for archive in archives)
    assert all(end <= own_end or start >= live_base for source, start, end in parsed if source == "chat"), \
        "the five foreign archives are ruled out by their summaries, never parsed"


def test_frozen_page_and_older_cursor_survive_rotation_and_live_append(tmp_path):
    write(tmp_path / "logs" / "chat.jsonl", [row(index) for index in range(400)])
    write(tmp_path / "logs" / "progress.jsonl", [row(index, "progress") for index in range(100)])
    status, first = request(tmp_path)
    assert status == 200
    original_ids = [message["history_id"] for message in first["messages"]]
    for source in ("chat", "progress"):
        archive = tmp_path / "archive" / f"{source}_20260912T000000.jsonl"
        archive.parent.mkdir(exist_ok=True)
        os.replace(tmp_path / "logs" / f"{source}.jsonl", archive)
        write(tmp_path / "logs" / f"{source}.jsonl", [row("new-live", source)])
    status, replayed = request(tmp_path, cursor=first["page_cursor"])
    assert status == 200
    assert [message["history_id"] for message in replayed["messages"]] == original_ids
    older = list(pages(tmp_path, cursor=first["next_cursor"]))
    all_messages = first["messages"] + [message for page in older for message in page["messages"]]
    assert len(all_messages) == 500
    assert len({message["history_id"] for message in all_messages}) == 500
    assert not any("new-live" in message["text"] for message in all_messages)


def test_page_handle_replays_its_exact_archive_range(tmp_path):
    for segment in range(6):
        write(tmp_path / "archive" / f"chat_20260901T00000{segment}.jsonl", [row(segment)])
    first = request(tmp_path, n_human="2")[1]
    second = request(tmp_path, cursor=first["next_cursor"])[1]
    replayed = request(tmp_path, cursor=second["page_cursor"])[1]
    assert replayed["messages"] == second["messages"]
    assert replayed["next_cursor"] == second["next_cursor"]


def test_accepted_quiz_answer_is_hidden_replay_evidence_across_pages(tmp_path):
    quiz = {"quiz_id": "old", "question": "Choose?", "state": "open",
            "options": [{"label": "First"}, {"label": "Second", "recommended": True}]}
    winning = {**quiz, "state": "answered", "comment": "  My own answer\nnext line  "}
    write(tmp_path / "logs" / "chat.jsonl", [
        row(0, direction="out", type="quiz", task_id="quiz-task", quiz=quiz),
        *[row(index) for index in range(1, 170)],
        row(171, direction="system", type="quiz_answer", task_id="quiz-task", quiz=winning),
    ])
    loaded = list(pages(tmp_path))
    [evidence] = [message for message in loaded[0]["messages"] if message.get("system_type") == "quiz_answer"]
    assert evidence["text"] == "" and evidence["quiz"] == winning
    assert "answered_index" not in evidence["quiz"]
    [ask] = [message for page in loaded[1:] for message in page["messages"] if message.get("msg_type") == "quiz"]
    assert ask["quiz"]["quiz_id"] == evidence["quiz"]["quiz_id"]
    assert ask["quiz"]["options"][1]["recommended"] is True


@pytest.mark.parametrize("current", ["running", "finalizing", "missing", "malformed", "future"])
def test_historical_terminal_never_overrides_current_result_authority(tmp_path, current):
    write(tmp_path / "logs" / "progress.jsonl", [row(0, "progress", task_id="old-task")])
    terminal = row(0, direction="system", type="task_summary", task_id="old-task", status="completed",
                   summary_kind="terminal_result_projection", outcome_authority="canonical_task_result_after_finalization",
                   outcome_final=True, outcome_phase="done")
    write(tmp_path / "logs" / "chat.jsonl", [terminal])
    result = tmp_path / "task_results" / "old-task.json"
    result.parent.mkdir(exist_ok=True)
    if current == "malformed":
        result.write_text("{broken", encoding="utf-8")
    elif current != "missing":
        result.write_text(json.dumps({"_schema_version": 2 if current == "future" else 1,
                                      "task_id": "old-task", "status": "completed" if current == "finalizing" else "running",
                                      **({"root_phase_checkpoint": {"post_task_synthesis": "running"}}
                                         if current == "finalizing" else {})}), encoding="utf-8")
    status, payload = request(tmp_path)
    assert status == 200
    [narration] = [message for message in payload["messages"] if message.get("is_progress")]
    evidence = [message for message in payload["messages"] if message.get("summary_kind")]
    assert "task_terminal_status" not in narration
    if current in {"running", "finalizing"}:
        assert len(evidence) == 1 and evidence[0]["historical_terminal"]["status"] == "completed"
        assert "historical_terminal" not in narration
    else:
        assert evidence == []  # the visible narration already carries this exact fact
        assert narration["historical_terminal"]["status"] == "completed"
    if current == "finalizing":
        assert narration["task_phase"] == "finalizing" and narration["outcome_final"] is False
    if current in {"malformed", "future"}:
        assert (result.parent / "quarantine" / result.name).exists()


def test_required_source_failure_retries_the_same_page(tmp_path, monkeypatch):
    write(tmp_path / "logs" / "chat.jsonl", [row(index) for index in range(200)])
    first = request(tmp_path)[1]
    original = HistorySource._read
    monkeypatch.setattr(HistorySource, "_read", lambda *_a: (_ for _ in ()).throw(OSError("injected read failure")))
    status, failed = request(tmp_path, cursor=first["next_cursor"])
    assert status == 503 and failed["reason_code"] == "history_source_unavailable"
    assert failed["next_cursor"] == first["next_cursor"] and failed["has_more"] is True
    monkeypatch.setattr(HistorySource, "_read", original)
    status, retried = request(tmp_path, cursor=failed["next_cursor"])
    assert status == 200 and any(message["text"] == "human-0" for message in retried["messages"])


def test_membership_drift_and_foreign_room_cursor_are_explicit(tmp_path):
    from ouroboros.projects_registry import bind_task_to_project, create_project

    write(tmp_path / "logs" / "chat.jsonl", [row(index) for index in range(200)])
    project = create_project(tmp_path, "new-room", name="Room")
    first = request(tmp_path)[1]
    status, foreign = request(tmp_path, chat_id=str(project["chat_id"]), cursor=first["next_cursor"])
    assert status == 409 and foreign["reason_code"] == "history_view_changed"
    bind_task_to_project(tmp_path, "some-task", project["id"], origin={"absent": "system"})
    status, drift = request(tmp_path, cursor=first["next_cursor"])
    assert status == 409 and drift["reason_code"] == "history_view_changed"
    assert drift["next_cursor"] == first["next_cursor"]


def test_continuation_reads_only_before_its_position(tmp_path, monkeypatch):
    write(tmp_path / "logs" / "chat.jsonl", [row(index, text=f"human-{index}" + "x" * 2000) for index in range(1000)])
    first = request(tmp_path)[1]
    encoded = first["next_cursor"]
    import base64
    state = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
    cursor = decode_cursor(encoded, 1, state["view"])
    seen = []
    original = HistorySource._read

    def read(source, start, end):
        seen.append((source.source, start, end))
        return original(source, start, end)

    monkeypatch.setattr(HistorySource, "_read", read)
    assert request(tmp_path, cursor=encoded)[0] == 200
    assert seen and all(end <= cursor["before"][source] for source, _start, end in seen)


def test_partial_live_line_is_outside_frozen_history_even_after_rotation(tmp_path):
    path = tmp_path / "logs" / "chat.jsonl"
    write(path, [row(index) for index in range(200)])
    with path.open("ab") as handle:
        handle.write(b'{"direction":"in","text":"later')
    first = request(tmp_path)[1]
    with path.open("ab") as handle:
        handle.write(b'"}\n')
    archive = tmp_path / "archive" / "chat_20260912T000000.jsonl"
    archive.parent.mkdir()
    os.replace(path, archive)
    path.touch()
    replayed = request(tmp_path, cursor=first["page_cursor"])[1]
    assert replayed["messages"] == first["messages"]
    assert not any(message["text"] == "later" for page in pages(tmp_path, cursor=first["next_cursor"]) for message in page["messages"])


def recoded(encoded, **fields):
    """The same cursor with ``fields`` replaced; ``None`` drops a field."""
    import base64

    cursor = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
    for key, value in fields.items():
        if value is None:
            cursor.pop(key, None)
        else:
            cursor[key] = value
    return encode_cursor(cursor)


def test_a_cursor_minted_before_the_unfinished_boundary_field_reads_as_a_clean_boundary(tmp_path):
    write(tmp_path / "logs" / "chat.jsonl", [row(index) for index in range(200)])
    first = request(tmp_path)[1]
    for name in ("next_cursor", "page_cursor"):
        current = request(tmp_path, cursor=first[name])
        legacy = request(tmp_path, cursor=recoded(first[name], unfinished=None))
        assert current[0] == legacy[0] == 200
        assert legacy[1]["messages"] == current[1]["messages"]
        assert legacy[1]["next_cursor"] == current[1]["next_cursor"], "re-minted with the field, unchanged"


@pytest.mark.parametrize("unfinished", [
    "chat", False, {"chat": True}, ["chat", "chat"], ["progress", "chat"], ["archive"], [1], [["chat"]], [{}],
])
def test_a_malformed_unfinished_boundary_is_an_invalid_cursor(tmp_path, unfinished):
    write(tmp_path / "logs" / "chat.jsonl", [row(index) for index in range(200)])
    first = request(tmp_path)[1]
    for name in ("next_cursor", "page_cursor"):
        status, payload = request(tmp_path, cursor=recoded(first[name], unfinished=unfinished))
        assert status == 400 and payload["reason_code"] == "history_cursor_invalid", unfinished


def test_de_roled_recent_child_final_is_enriched_when_older_parent_arrives(tmp_path):
    write(tmp_path / "logs" / "chat.jsonl", [
        row(0, direction="out", task_id="parent", ts="2020-01-01T00:00:00Z", task_terminal_status="completed"),
        row(1, direction="out", task_id="child", ts="2020-01-02T00:00:00Z",
            delegation_role="subagent", parent_task_id="parent", root_task_id="parent", subagent_task_id="child"),
    ])
    write(tmp_path / "logs" / "progress.jsonl", [row(index, "progress", task_id="other") for index in range(70)])
    for task in ("parent", "child"):
        result = tmp_path / "task_results" / f"{task}.json"
        result.parent.mkdir(exist_ok=True)
        result.write_text(json.dumps({"_schema_version": 1, "task_id": task, "status": "completed"}), encoding="utf-8")
    loaded = list(pages(tmp_path, n_human="1"))
    first = next(message for message in loaded[0]["messages"] if message.get("task_id") == "child")
    assert "parent_task_id" not in first
    enriched = next(message for page in loaded[1:] for message in page["messages"]
                    if message.get("task_id") == "child" and message.get("parent_task_id") == "parent")
    assert enriched["history_id"] == first["history_id"]
    assert any(message.get("task_id") == "parent" for page in loaded[1:] for message in page["messages"])


def test_archived_media_and_folded_review_attempts_use_the_same_projection(tmp_path):
    old = [row(0, direction="out", type="document", task_id="old-task", filename="report.pdf",
               download_url="/api/tasks/old-task/artifacts/report.pdf", size_bytes=256),
           row(1, direction="out", type="photo", task_id="old-task", mime="image/png",
               download_url="/api/tasks/old-task/artifacts/image.png")]
    old.extend(row(index, direction="system", type="skill_review", task_id="review-child",
                   presentation_owner_task_id="old-task", root_task_id="old-task",
                   group_id="task:old-task:alpha", skill="alpha", job_id=f"old-job-{index}", status="clean")
               for index in range(4))
    write(tmp_path / "archive" / "chat_20260901T000000.jsonl", old)
    for segment in range(1, 6):
        write(tmp_path / "archive" / f"chat_20260901T00000{segment}.jsonl",
              [row(segment * 50 + index) for index in range(50)])
    messages = [message for page in pages(tmp_path) for message in page["messages"]]
    document = next(message for message in messages if message.get("msg_type") == "document")
    photo = next(message for message in messages if message.get("msg_type") == "photo")
    assert document["filename"] == "report.pdf" and document["size_bytes"] == 256
    assert photo["download_url"].endswith("image.png")
    attempts = [attempt for message in messages if message.get("review_group") for attempt in message["review_group"]["attempts"]]
    assert {attempt["job_id"] for attempt in attempts} == {f"old-job-{index}" for index in range(4)}
    assert all(attempt.get("history_id", "").startswith("chat:") for attempt in attempts)


def test_archive_parse_gap_and_zero_quota_are_not_a_cursor_loop(tmp_path):
    for segment in range(4):
        write(tmp_path / "archive" / f"chat_20260901T00000{segment}.jsonl", [row(segment)])
    oldest = tmp_path / "archive" / "chat_20260901T000000.jsonl"
    oldest.write_bytes(b"{malformed\n" + oldest.read_bytes())
    write(tmp_path / "logs" / "progress.jsonl", [row(index, "progress") for index in range(5)])
    loaded = list(pages(tmp_path, n_human="1", n_progress="0"))
    assert loaded[-1]["has_more"] is False and loaded[-1]["window"]["complete"] is False
    assert "chat_malformed_jsonl" in loaded[-1]["window"]["truncated_by"]
    assert not any(message.get("is_progress") for page in loaded for message in page["messages"])


def test_a_closed_bounded_wait_reaches_the_replayed_card(tmp_path):
    """The durable wait lifecycle rides the replay overlay: once the bound closed (the
    projection dropped `wait_for_answer`, kept `wait_ended_at`), the replayed card no longer
    says the task is waiting (astra scope round 4)."""
    from ouroboros.owner_quiz import mark_wait_ended, record_asked

    quiz = {"quiz_id": "w1", "question": "Proceed?", "state": "open", "wait_for_answer": True,
            "options": [{"label": "Yes"}, {"label": "No"}]}
    write(tmp_path / "logs" / "chat.jsonl", [row(0, direction="out", type="quiz", task_id="wait-task", quiz=quiz)])
    record_asked(tmp_path, "wait-task", quiz_id="w1", question="Proceed?", options=["Yes", "No"],
                 assumption="", wait_for_answer=True, chat_id=1)
    assert mark_wait_ended(tmp_path, "wait-task", "w1") is True
    [ask] = [message for page in pages(tmp_path) for message in page["messages"] if message.get("msg_type") == "quiz"]
    assert ask["quiz"]["state"] == "open" and ask["quiz"]["wait_ended_at"]
    assert "wait_for_answer" not in ask["quiz"]
