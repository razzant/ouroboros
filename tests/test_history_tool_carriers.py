"""Tool evidence survives pages and absent narration without becoming speech."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from ouroboros.gateway import history
from ouroboros.tool_call_log import replay_evidence_for_tasks

pytestmark = pytest.mark.serial
TS = "2026-10-04T09:00:00Z"


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _tool(task, fact="tool_call", **extra):
    return {"task_id": task, "ts": TS, "type": fact, "tool": "read_file",
            "invocation_id": f"{task}-call", "is_error": False, **extra}


def _read(root, **kwargs):
    return json.loads(history._assemble_history_response(root, 1, 1, 1, **kwargs))


def _room_read(root, chat_id):
    response = asyncio.run(history.make_chat_history_endpoint(root)(
        SimpleNamespace(query_params={"chat_id": str(chat_id)})))
    assert response.status_code == 200
    return json.loads(response.body)


@pytest.fixture(autouse=True)
def _no_live_activity(monkeypatch):
    monkeypatch.setattr("ouroboros.gateway.state._chat_activities_snapshot_safe", lambda *_: [])


@pytest.mark.parametrize("role", ["in", "out"])
def test_plain_speech_and_user_rows_keep_authorship_beside_host_evidence(tmp_path, role):
    _write(tmp_path / "logs/chat.jsonl", [{"direction": role, "task_id": "root", "ts": TS,
                                           "text": "Actual words", "chat_id": 1}])
    _write(tmp_path / "logs/tools.jsonl", [_tool("root")])
    _write(tmp_path / "task_results/root.json", [{"_schema_version": 1, "task_id": "root", "status": "completed"}])
    result = _read(tmp_path)
    speech, carrier = result["messages"]
    assert speech["text"] == "Actual words" and speech["role"] == ("user" if role == "in" else "assistant")
    assert speech["history_id"] and "tool_evidence" not in speech
    assert carrier["system_type"] == "task_evidence" and carrier["role"] == "system"
    assert carrier["text"] == "" and carrier["narration"] is False
    assert "history_id" not in carrier and "history_position" not in carrier
    assert carrier["task_terminal_status"] == "completed"
    assert carrier["tool_evidence"]["observations"][0]["key"] == "tool:root:root-call"
    assert result["window"]["complete"] is True, "an evidence carrier consumes no physical-page quota"


def test_child_only_window_recovers_parent_tools_without_parent_speech(tmp_path):
    _write(tmp_path / "logs/progress.jsonl", [{"task_id": "child", "parent_task_id": "root",
        "root_task_id": "root", "delegation_role": "subagent", "subagent_event": "completed",
        "ts": TS, "content": "The child finished", "chat_id": 1}])
    _write(tmp_path / "logs/tools.jsonl", [_tool("root")])
    _write(tmp_path / "task_results/root.json", [{"_schema_version": 1, "task_id": "root", "status": "completed", "chat_id": 1}])
    carriers = [row for row in _read(tmp_path)["messages"] if row.get("system_type") == "task_evidence"]
    assert [row["task_id"] for row in carriers] == ["root"]
    assert carriers[0]["task_terminal_status"] == "completed"


def test_speechless_current_root_is_addressed_without_importing_other_rooms(tmp_path, monkeypatch):
    from ouroboros.projects_registry import create_project

    project = create_project(tmp_path, "other", name="Other room")
    rows = [{"activity_id": task, "chat_id": chat, "kind": "direct_chat", "phase": "thinking"}
            for task, chat in (("root", 1), ("foreign", project["chat_id"]), ("hidden", 0))]
    monkeypatch.setattr("ouroboros.gateway.state._chat_activities_snapshot_safe", lambda *_: rows)
    _write(tmp_path / "logs/tools.jsonl", [_tool(row["activity_id"]) for row in rows])
    assert [row["task_id"] for row in _read(tmp_path)["messages"]] == ["root"]


def test_current_root_with_original_zero_address_follows_its_project_binding(tmp_path, monkeypatch):
    from ouroboros.projects_registry import bind_task_to_project, create_project

    project = create_project(tmp_path, "bound", name="Bound room")
    bind_task_to_project(tmp_path, "root", project["id"], origin={"absent": "system"})
    monkeypatch.setattr("ouroboros.gateway.state._chat_activities_snapshot_safe", lambda *_: [
        {"activity_id": "root", "chat_id": 0, "kind": "direct_chat", "phase": "thinking"}])
    _write(tmp_path / "logs/tools.jsonl", [_tool("root")])
    [carrier] = _room_read(tmp_path, project["chat_id"])["messages"]
    assert carrier["task_id"] == "root" and carrier["system_type"] == "task_evidence"
    assert carrier["tool_evidence"]["observations"][0]["key"] == "tool:root:root-call"
    assert _room_read(tmp_path, 1)["messages"] == []
    assert _room_read(tmp_path, 0)["messages"] == [], "hidden chat is not a new browser room"


@pytest.mark.parametrize("route", ["bound_foreign", "stored_foreign", "unknown", "same_room", "bound_same"])
def test_child_cannot_address_parent_tool_evidence_to_its_own_room(tmp_path, monkeypatch, route):
    from ouroboros import tool_call_log
    from ouroboros.projects_registry import bind_task_to_project, create_project

    current = create_project(tmp_path, "child-room", name="Child room")
    foreign = create_project(tmp_path, "parent-room", name="Parent room")
    bind_task_to_project(tmp_path, "child", current["id"], origin={"absent": "system"})
    if route.startswith("bound_"):
        bind_task_to_project(tmp_path, "parent", (foreign if route == "bound_foreign" else current)["id"],
                             origin={"absent": "system"})
    result = {"_schema_version": 1, "task_id": "parent", "status": "completed"}
    if route != "unknown":
        result["chat_id"] = (foreign if route in {"stored_foreign", "bound_same"} else current)["chat_id"]
    _write(tmp_path / "task_results/parent.json", [result])
    _write(tmp_path / "logs/progress.jsonl", [{"task_id": "child", "parent_task_id": "parent",
        "root_task_id": "parent", "delegation_role": "subagent", "subagent_event": "completed",
        "content": "Child complete", "chat_id": current["chat_id"], "ts": TS}])
    _write(tmp_path / "logs/tools.jsonl", [_tool("parent"), _tool("child")])
    selected = []
    replay = tool_call_log.replay_evidence_for_tasks

    def observe(root, task_ids):
        selected.extend(task_ids)
        return replay(root, task_ids)

    monkeypatch.setattr(tool_call_log, "replay_evidence_for_tasks", observe)
    messages = _room_read(tmp_path, current["chat_id"])["messages"]
    assert messages[0]["task_id"] == "child" and messages[0]["parent_task_id"] == "parent"
    allowed = route in {"same_room", "bound_same"}
    assert ("parent" in selected) is allowed, "lineage alone does not select a parent's tools"
    carriers = [row for row in messages if row.get("system_type") == "task_evidence"]
    assert [row["task_id"] for row in carriers] == (["parent"] if allowed else [])
    if allowed:
        assert carriers[0]["task_terminal_status"] == "completed"


def test_current_root_narration_outside_page_keeps_evidence_without_fake_cursor(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.gateway.state._chat_activities_snapshot_safe", lambda *_: [
        {"activity_id": "root", "chat_id": 1, "phase": "working", "kind": "managed_task"}])
    _write(tmp_path / "logs/chat.jsonl", [
        {"direction": "out", "task_id": "root", "text": "Earlier speech", "ts": TS, "chat_id": 1},
        {"direction": "in", "text": "Latest question", "ts": "2026-10-04T10:00:00Z", "chat_id": 1}])
    _write(tmp_path / "logs/tools.jsonl", [_tool("root")])
    first = _read(tmp_path)
    assert first["has_more"] is True
    assert [row["text"] for row in first["messages"]] == ["Latest question", ""]
    older = _read(tmp_path, cursor=first["next_cursor"])
    assert older["messages"][0]["text"] == "Earlier speech"
    assert older["messages"][0]["history_id"] != first["messages"][0]["history_id"]
    assert all("history_id" not in row for row in older["messages"] if row.get("system_type") == "task_evidence")


@pytest.mark.parametrize("witness", ["summary", "result", "parent_delivery"])
def test_empty_known_missing_replay_is_carried_as_incomplete(tmp_path, witness):
    _write(tmp_path / "logs/chat.jsonl", [{"direction": "out", "task_id": "root", "text": "Saved speech", "ts": TS}])
    if witness == "summary":
        _write(tmp_path / "logs/chat.jsonl", [{"direction": "system", "type": "task_summary", "task_id": "root",
                                               "text": "", "tool_calls": 1, "ts": TS}])
    else:
        fields = {"tool_calls": 1} if witness == "result" else {
            "completion_observations": {"delivery_counts": {"send_message": {"calls": 1}}}}
        _write(tmp_path / "task_results/root.json", [{"_schema_version": 1, "task_id": "root", "status": "completed", "chat_id": 1, **fields}])
        if witness == "parent_delivery":
            _write(tmp_path / "logs/chat.jsonl", [])
            _write(tmp_path / "logs/progress.jsonl", [{"task_id": "child", "parent_task_id": "root",
                "delegation_role": "subagent", "subagent_event": "completed", "content": "Child finished", "ts": TS}])
    for index in range(4):
        _write(tmp_path / "archive" / f"tools_{index}.jsonl", [_tool("root" if index == 0 else "other")])
    carrier = _read(tmp_path)["messages"][-1]
    assert carrier["system_type"] == ("task_summary" if witness == "summary" else "task_evidence")
    assert carrier["tool_calls"] == 1
    assert carrier["tool_evidence"]["observations"] == []
    assert carrier["tool_evidence"]["coverage"]["archives_bounded"] is True


@pytest.mark.parametrize("archive_count", [0, 3, 4, 12])
def test_unrelated_archive_bound_does_not_invent_tool_work(tmp_path, archive_count):
    _write(tmp_path / "logs/chat.jsonl", [{"direction": "out", "task_id": "greeting", "text": "Hello", "ts": TS}])
    for index in range(archive_count):
        _write(tmp_path / "archive" / f"tools_{index:02}.jsonl", [_tool("other")])
    [reply] = _read(tmp_path)["messages"]
    assert reply["text"] == "Hello" and reply["role"] == "assistant"
    assert "tool_calls" not in reply, "absence of a task witness is never a zero count"
    evidence = replay_evidence_for_tasks(tmp_path, ["greeting"])["greeting"]
    assert evidence["coverage"]["archives_bounded"] is (archive_count > 3), "raw read bounds remain honest"


def test_empty_reader_gap_still_has_an_inert_carrier(tmp_path, monkeypatch):
    _write(tmp_path / "logs/chat.jsonl", [{"direction": "out", "task_id": "root", "text": "Saved speech", "ts": TS}])
    monkeypatch.setattr("ouroboros.tool_call_log.replay_evidence_for_tasks", lambda _root, _ids: {
        "root": {"observations": [], "legacy": {"calls": 0}, "coverage": {
            "source": "logs/tools.jsonl", "gaps": ["read_failed"], "shown": 0, "matched": 0}}})
    carrier = _read(tmp_path)["messages"][-1]
    assert carrier["system_type"] == "task_evidence"
    assert carrier["tool_evidence"]["coverage"]["gaps"] == ["read_failed"]
    assert "tool_calls" not in carrier and "history_id" not in carrier


def test_shared_noisy_log_windows_are_decoded_once_per_request(tmp_path, monkeypatch):
    from ouroboros import utils

    rows = [_tool("root"), _tool("second")]
    rows += [_tool("noisy", invocation_id=str(index), result_preview="x" * 1100) for index in range(6000)]
    path = tmp_path / "logs/tools.jsonl"
    _write(path, rows)
    parser = utils.iter_jsonl_objects
    decoded = []

    def observed(source, *, tail_bytes=None, gap_reasons=None):
        decoded.append((source, tail_bytes))
        yield from parser(source, tail_bytes=tail_bytes, gap_reasons=gap_reasons)

    monkeypatch.setattr(utils, "iter_jsonl_objects", observed)
    result = replay_evidence_for_tasks(tmp_path, ["root", "second"])
    assert all(len(evidence["observations"]) == 1 for evidence in result.values())
    assert len(decoded) == len(set(decoded)), "selected tasks share the same physical parse"
    assert sum(min(path.stat().st_size, window or path.stat().st_size) for _, window in decoded) < 3 * path.stat().st_size
    before = len(decoded)
    replay_evidence_for_tasks(tmp_path, ["root"])
    assert len(decoded) > before, "no cross-request memo can hide new settlements"


def test_known_unfinished_speech_does_not_claim_completion(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.task_status.load_effective_task_result", lambda *_args, **_kwargs: {"status": "running"})
    _write(tmp_path / "logs/chat.jsonl", [{"direction": "out", "task_id": "root", "text": "Still investigating", "ts": TS}])
    _write(tmp_path / "logs/tools.jsonl", [_tool("root")])
    rows = _read(tmp_path)["messages"]
    assert all(row.get("task_phase") == "unfinished" for row in rows)
    assert all("task_terminal_status" not in row for row in rows)


def test_shared_replay_rereads_a_source_that_changes_inside_the_request(tmp_path, monkeypatch):
    from ouroboros import utils

    path = tmp_path / "logs/tools.jsonl"
    _write(path, [_tool("first")])
    parser = utils.iter_jsonl_objects

    def changing(source, **kwargs):
        yield from parser(source, **kwargs)
        if source == path:
            _write(path, [_tool("first"), _tool("second")])

    monkeypatch.setattr(utils, "iter_jsonl_objects", changing)
    evidence = replay_evidence_for_tasks(tmp_path, ["first", "second"])
    assert list(evidence) == ["first", "second"]
    assert evidence["second"]["observations"][0]["key"] == "tool:second:second-call"


def _read_wide(root):
    return json.loads(history._assemble_history_response(root, 1, 10, 10, None))


def _archived_tools_for(root, tasks):
    """Speech for every task while its tool rows live only in archives: each task needs backfill."""
    _write(root / "logs/chat.jsonl", [{"direction": "out", "task_id": task, "text": f"{task} spoke", "ts": TS, "chat_id": 1}
                                      for task in tasks])
    _write(root / "logs/tools.jsonl", [_tool("other")])
    _write(root / "archive/tools_01.jsonl", [_tool(task) for task in tasks])
    _write(root / "archive/tools_00.jsonl", [_tool("older")])


def _counting_listing(monkeypatch, redirect=None):
    """Count real ``archive/`` enumerations per prefix; ``redirect`` swaps the directory the tools prefix lists."""
    from ouroboros import jsonl_tail

    listed = []
    real = jsonl_tail.archive_segments

    def counted(archive_dir, archive_prefix, gaps=None):
        listed.append(archive_prefix)
        return real(redirect if redirect is not None and archive_prefix == "tools" else archive_dir, archive_prefix, gaps)

    monkeypatch.setattr(jsonl_tail, "archive_segments", counted)
    return listed


def test_one_history_request_lists_the_archive_directory_once(tmp_path, monkeypatch):
    _archived_tools_for(tmp_path, ["a", "b"])
    listed = _counting_listing(monkeypatch)
    carriers = {row["task_id"]: row for row in _read_wide(tmp_path)["messages"] if row.get("system_type") == "task_evidence"}
    assert set(carriers) == {"a", "b"}
    coverage = {task: row["tool_evidence"]["coverage"] for task, row in carriers.items()}
    assert all(facts["archives"] == 2 and facts["archives_available"] == 2 for facts in coverage.values()), "both backfilled"
    assert {row["tool_evidence"]["observations"][0]["key"] for row in carriers.values()} == {"tool:a:a-call", "tool:b:b-call"}
    assert listed.count("tools") == 1, "one request enumerates archive/ once for the tools prefix, not once per task"


def test_shared_archive_listing_keeps_the_history_body_byte_identical(tmp_path, monkeypatch):
    from ouroboros import tool_call_log
    from ouroboros.tool_call_log import replay_evidence

    _archived_tools_for(tmp_path, ["a", "b", "c"])
    _write(tmp_path / "task_results/a.json", [{"_schema_version": 1, "task_id": "a", "status": "completed", "chat_id": 1}])
    shared = history._assemble_history_response(tmp_path, 1, 10, 10, None)
    # The per-task reader: every task lists and parses on its own (the pre-sharing shape).
    monkeypatch.setattr(tool_call_log, "replay_evidence_for_tasks", lambda root, task_ids: {
        task: replay_evidence(root, task) for task in dict.fromkeys(task_ids)})
    assert shared == history._assemble_history_response(tmp_path, 1, 10, 10, None)
    assert json.loads(shared)["messages"][-1]["tool_evidence"]["coverage"]["archives"] == 2


def test_unreadable_archive_directory_reports_the_same_gap_on_every_carrier(tmp_path, monkeypatch):
    _write(tmp_path / "logs/chat.jsonl", [{"direction": "out", "task_id": task, "text": f"{task} spoke", "ts": TS, "chat_id": 1}
                                          for task in ("a", "b")])
    _write(tmp_path / "logs/tools.jsonl", [_tool("a"), _tool("b")])
    # Listing a regular file fails with an OSError that is not FileNotFoundError: the enumeration gap.
    listed = _counting_listing(monkeypatch, redirect=tmp_path / "logs/tools.jsonl")
    carriers = {row["task_id"]: row for row in _read_wide(tmp_path)["messages"] if row.get("system_type") == "task_evidence"}
    assert set(carriers) == {"a", "b"}
    for task, row in carriers.items():
        coverage = row["tool_evidence"]["coverage"]
        assert coverage["gaps"] == ["unreadable_source"] and coverage["archives_available"] == 0
        assert row["tool_evidence"]["observations"][0]["key"] == f"tool:{task}:{task}-call"
    assert listed.count("tools") == 1, "the one enumeration's gap is replayed to every task"


def test_a_carrier_that_is_the_tasks_only_terminal_row_keeps_the_whole_terminal_truth(tmp_path):
    # web/modules/chat.js settles a reloaded card from the LAST row carrying task_terminal_status; for a speech-plus-tools
    # task with no summary or progress row in the window that row is the carrier, so it must carry what the summary path
    # reads (axes, review projection, reason, cancel origin, cost), or the card replays as a plain Done.
    truth = {"_schema_version": 1, "status": "completed", "chat_id": 1, "reason_code": "finished",
             "accounted_upper_bound_usd": 1.25, "cost_final": True, "metadata": {"initiator": "consciousness"},
             "review_projection": {"panels": []}, "model_execution": {"model": "m"}, "cancel_origin": {"by": "owner"}}
    _write(tmp_path / "logs/chat.jsonl", [
        {"direction": "out", "task_id": "bare", "text": "Speech only", "ts": TS, "chat_id": 1},
        {"direction": "system", "type": "task_summary", "task_id": "summarized", "text": "", "ts": TS, "chat_id": 1}])
    _write(tmp_path / "logs/tools.jsonl", [_tool("bare"), _tool("summarized")])
    for task in ("bare", "summarized"):
        _write(tmp_path / f"task_results/{task}.json", [{**truth, "task_id": task}])
    rows = {row["task_id"]: row for row in _read_wide(tmp_path)["messages"]
            if row.get("system_type") in {"task_evidence", "task_summary"}}
    carrier, summary = rows["bare"], rows["summarized"]
    assert carrier["system_type"] == "task_evidence" and summary["system_type"] == "task_summary"
    assert carrier["task_terminal_status"] == "completed"
    for key in ("review_projection", "outcome_axes", "reason_code", "cancel_origin", "model_execution"):
        assert key in carrier and carrier[key] == summary[key], key


@pytest.mark.parametrize("direction,row_type", [("out", "main_notice"), ("system", "project_handoff")])
def test_a_row_main_pins_for_a_project_task_does_not_mint_its_card_in_main(tmp_path, direction, row_type):
    # Main shows a Project root's notice and lifecycle rows by a pinning exemption; the task
    # lives in its Project room, so its tool evidence (and the card it mints) stays there (#1505).
    from ouroboros.projects_registry import bind_task_to_project, create_project

    project = create_project(tmp_path, "room", name="Project room")
    bind_task_to_project(tmp_path, "root", project["id"], origin={"absent": "system"})
    _write(tmp_path / "logs/chat.jsonl", [
        {"direction": direction, "type": row_type, "task_id": "root", "text": "Pinned", "ts": TS, "chat_id": 1},
        {"direction": "out", "task_id": "root", "text": "Project answer", "ts": TS, "chat_id": project["chat_id"]}])
    _write(tmp_path / "logs/tools.jsonl", [_tool("root")])
    _write(tmp_path / "task_results/root.json", [{"_schema_version": 1, "task_id": "root", "status": "completed",
                                                  "chat_id": project["chat_id"]}])
    main = _room_read(tmp_path, 1)["messages"]
    assert [(row["task_id"], row["system_type"]) for row in main] == [("root", row_type)], main
    room = _room_read(tmp_path, project["chat_id"])["messages"]
    carriers = [row for row in room if row.get("system_type") == "task_evidence"]
    assert [row["task_id"] for row in carriers] == ["root"], "the Project room keeps its task's tools"
    assert carriers[0]["tool_evidence"]["observations"][0]["key"] == "tool:root:root-call"


def test_a_main_rooted_tasks_own_main_notice_still_carries_its_tools(tmp_path):
    _write(tmp_path / "logs/chat.jsonl", [{"direction": "out", "type": "main_notice", "task_id": "root",
                                           "text": "Owner action", "ts": TS, "chat_id": 1}])
    _write(tmp_path / "logs/tools.jsonl", [_tool("root")])
    rows = _room_read(tmp_path, 1)["messages"]
    assert [(row["task_id"], row["system_type"]) for row in rows] == [("root", "main_notice"), ("root", "task_evidence")]
