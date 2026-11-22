"""Complete retained history uses physical continuation and shared projection."""

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import dialogue_import, dialogue_store as ds
from ouroboros.gateway import history
from ouroboros.gateway.history_paging import HistorySource, decode_cursor, encode_cursor
from ouroboros.store import StoreError
from tests._store_shared import seed_result


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


def imported(root):
    """What the boot and the maintenance child do before a reader runs: the cutover takes the
    chain files a fixture wrote (or an older body appended since), the job imports the archive."""
    dialogue_import.cutover(root)
    if dialogue_import.pending_segments(root):
        assert dialogue_import.run_dialogue_import(root)["state"] == "done"


def request(root, **params):
    imported(root)
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
    decoded, entries = [], HistorySource._entries
    monkeypatch.setattr(HistorySource, "_entries", lambda self, rows: (
        decoded.extend((self.source, entry.seq) for entry in rows), entries(self, rows))[1])
    loaded = list(pages(tmp_path, chat_id=str(project["chat_id"])))
    assert len(loaded) == 1 and loaded[0]["window"]["complete"] is True
    assert [message["text"] for message in loaded[0]["messages"]] == ["human-0", "human-1", "human-2"]
    own_end = archives[0].stat().st_size
    assert decoded and all(seq < own_end for source, seq in decoded if source == "chat"), \
        "the five foreign archives' rows are addressed out of the room's page, never decoded"


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


@pytest.mark.parametrize("kind", ["quiz_answer", "quiz_closed"])
def test_quiz_disposition_is_hidden_replay_evidence_across_pages(tmp_path, kind):
    quiz = {"quiz_id": "old", "question": "Choose?", "state": "open",
            "options": [{"label": "First"}, {"label": "Second", "recommended": True}]}
    winning = ({**quiz, "state": "answered", "comment": "  My own answer\nnext line  "} if kind == "quiz_answer"
               else {**quiz, "state": "closed", "closed_reason": "Superseded", "closed_actor": "owner",
                     "closed_request_id": "close-1"})
    write(tmp_path / "logs" / "chat.jsonl", [
        row(0, direction="out", type="quiz", task_id="quiz-task", quiz=quiz),
        *[row(index) for index in range(1, 170)],
        row(171, direction="system", type=kind, task_id="quiz-task", quiz=winning),
    ])
    loaded = list(pages(tmp_path))
    [evidence] = [message for message in loaded[0]["messages"] if message.get("system_type") == kind]
    assert evidence["text"] == "" and evidence["quiz"] == winning
    assert "answered_index" not in evidence["quiz"]
    [ask] = [message for page in loaded[1:] for message in page["messages"] if message.get("msg_type") == "quiz"]
    assert ask["quiz"]["quiz_id"] == evidence["quiz"]["quiz_id"]
    assert ask["quiz"]["options"][1]["recommended"] is True


@pytest.mark.parametrize("current", ["running", "finalizing", "missing", "future"])
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
    elif current == "future":
        result.write_text(json.dumps({"_schema_version": 2, "task_id": "old-task", "status": "running"}), encoding="utf-8")
    elif current != "missing":
        seed_result(tmp_path, "old-task", {"_schema_version": 1,
                                      "task_id": "old-task", "status": "completed" if current == "finalizing" else "running",
                                      **({"root_phase_checkpoint": {"post_task_synthesis": "running"}}
                                         if current == "finalizing" else {})})
    from ouroboros.store_import import ensure_current_store
    ensure_current_store(tmp_path)  # the boot door's import judges the planted file; a GET never imports
    status, payload = request(tmp_path)
    assert status == 200
    [narration] = [message for message in payload["messages"] if message.get("is_progress")]
    evidence = [message for message in payload["messages"] if message.get("summary_kind")]
    assert "task_terminal_status" not in narration
    if current in {"running", "finalizing"}:
        assert len(evidence) == 1 and evidence[0]["historical_terminal"]["status"] == "completed"
        assert evidence[0]["task_phase"] == ("finalizing" if current == "finalizing" else "unfinished")
        assert "historical_terminal" not in narration
    else:
        assert evidence == []  # the visible narration already carries this exact fact
        assert narration["historical_terminal"]["status"] == "completed"
    if current == "finalizing":
        assert narration["task_phase"] == "finalizing" and narration["outcome_final"] is False
    if current in {"malformed", "future"}:
        assert (result.parent / "quarantine" / result.name).exists()


@pytest.mark.parametrize("carrier", ["summary", "progress", "answer"])
def test_finished_attempt_retains_its_delivered_answer_facts_on_this_page(tmp_path, carrier):
    tid = "delivered-before-checkpoint"
    seed_result(tmp_path, tid, {"status": "failed", "chat_id": 1, "task_attempt": 2,
                                "reason_code": "finalization_incomplete", "terminal_origin": "model_final",
                                "finalization_recovery": {"answer": "delivered"},
                                "outcome_axes": {"lifecycle": {"status": "failed"}, "execution": {"status": "infra_failed"}},
                                "result": "private full result"})
    chat = [row(1, direction="out", task_id=tid, task_attempt=2, text="The answer")]
    if carrier == "summary":
        chat.append(row(2, direction="system", type="task_summary", task_id=tid, task_attempt=2, status="failed"))
    if carrier != "answer":
        write(tmp_path / "logs" / "progress.jsonl", [row(i, "progress", task_id=tid, task_attempt=2,
              ts=f"2026-09-12T00:00:0{i}Z") for i in (1, 3)])
    write(tmp_path / "logs" / "chat.jsonl", chat)
    code, response = request(tmp_path)
    assert code == 200
    rows = [r for r in response["messages"] if r.get("task_id") == tid]
    [rich] = [r for r in rows if r.get("finalization_recovery")]
    assert (rich["task_terminal_status"], rich["outcome_phase"], rich["terminal_origin"],
            rich["finalization_recovery"]["answer"], rich["task_attempt"]) == ("failed", "done", "model_final", "delivered", 2)
    if carrier == "summary":
        assert rich["system_type"] == "task_summary"
    elif carrier == "progress":
        assert rich["is_progress"] and rich["ts"].endswith("03Z")
    else:
        assert rich["text"] == "The answer"
    assert "private full result" not in json.dumps(rows)


def test_older_page_keeps_a_pruned_childs_own_terminal_receipt(tmp_path, monkeypatch):
    from ouroboros.gateway import history_paging
    from ouroboros.terminal_projection import _project_row
    from tests._store_shared import delete_result

    tid, parent = "garden-reader", "garden"
    lineage = {"task_id": tid, "root_task_id": parent, "parent_task_id": parent, "delegation_role": "subagent"}
    stored = seed_result(tmp_path, tid, {**lineage, "status": "failed", "chat_id": 1, "task_attempt": 2,
                         "reason_code": "finalization_incomplete", "terminal_origin": "model_final",
                         "finalization_recovery": {"answer": "delivered"}, "result": "Private answer bytes",
                         "outcome_axes": {"execution": {"status": "infra_failed"}}})
    receipt = {**_project_row(tid, stored, {}, {"chat_id": 1}), "ts": row(1)["ts"]}
    write(tmp_path / "logs/chat.jsonl", [row(0), receipt, row(2), row(3)])
    write(tmp_path / "logs/progress.jsonl", [row(4, "progress", task_id=parent), row(5, "progress", **lineage,
          subagent_task_id=tid, subagent_event="progress")])
    assert delete_result(tmp_path, tid)
    monkeypatch.setattr(history_paging, "_RECENT_WINDOW_ROWS", 1)
    latest, *older = pages(tmp_path, n_human="1")
    assert any(r.get("parent_task_id") == parent and r["task_id"] == tid for r in latest["messages"])
    [page] = [p for p in older if any(r.get("summary_kind") == "terminal_result_projection" for r in p["messages"])]
    [hidden] = [r for r in page["messages"] if r.get("task_id") == tid]
    assert hidden["text"] == "" and hidden["system_type"] == "task_summary"
    assert hidden["historical_terminal"]["phase"] == "done" and hidden["task_attempt"] == 2
    assert {key: hidden[key] for key in ("reason_code", "terminal_origin", "finalization_recovery", "outcome_axes")} == {
        key: stored[key] for key in ("reason_code", "terminal_origin", "finalization_recovery", "outcome_axes")}
    assert "current_outcome" not in hidden and "Private answer bytes" not in json.dumps(hidden)


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("stored_clock", ["own", "earlier"])
def test_old_ending_keeps_its_own_attempt_beside_the_stored_outcome(tmp_path, frozen, stored_clock):
    from ouroboros.terminal_time import task_attempt_witness

    def ended(attempt, started, occurred):
        """The host's end fact as the result writer stores it: the whole witness of the attempt it observed."""
        return {"v": 1, "source": "executor_terminal", "occurred_at": occurred,
                "attempt": task_attempt_witness({"task_attempt": attempt, "started_at": started})}

    tid = "two-endings"
    old_time = ended(1, "2026-09-12T08:00:00Z", "2026-09-12T09:00:00Z")
    current_time = ended(2, "2026-09-12T10:00:00Z", "2026-09-12T10:20:00Z")
    seed_result(tmp_path, tid, {"status": "completed", "chat_id": 1, "task_attempt": 2, "started_at": "2026-09-12T10:00:00Z",
                                "reason_code": "final_message", "result": "second private answer",
                                "terminal_time": current_time if stored_clock == "own" else old_time})
    old = row(2, direction="system", type="task_summary", task_id=tid, task_attempt=1, status="failed",
              reason_code="finalization_incomplete", terminal_origin="model_final", finalization_recovery={"answer": "delivered"},
              terminal_time=old_time)
    if frozen:
        old.update(summary_kind="terminal_root_projection", outcome_authority="canonical_task_result_after_finalization",
                   outcome_phase="done", outcome_final=True)
    write(tmp_path / "logs" / "chat.jsonl", [old])
    observation = {"task_id": tid, "task_attempt": "1", "run_id": "first-run", "attempt_id": "run-attempt",
                   "harness_id": "codex", "phase": "running", "revision": 1}
    write(tmp_path / "logs" / "progress.jsonl", [row(1, "progress", task_id=tid, executor_observation=observation)])
    code, response = request(tmp_path)
    assert code == 200
    [ending] = [item for item in response["messages"] if item.get("system_type") == "task_summary"]
    recorded = ending.get("historical_terminal", ending)
    assert (recorded["status"], recorded["task_attempt"], recorded["terminal_time"]) == ("failed", 1, old_time)
    current = ending["current_outcome"]
    assert (current["status"], current["reason_code"], current["task_attempt"]) == ("completed", "final_message", 2)
    if stored_clock == "own":
        assert current["terminal_time"] == current_time
        assert current["terminal_time"]["occurred_at"] == "2026-09-12T10:20:00Z"
    else:  # labelled observed, yet witnessed for another attempt: this attempt's end stays unknown
        assert "terminal_time" not in current
    assert "finalization_recovery" not in current and "second private answer" not in json.dumps(ending)
    [progress] = [item for item in response["messages"] if item.get("is_progress")]
    assert progress["executor_observation"] == observation
    assert progress["task_attempt"] == 1 and "task_terminal_status" not in progress


@pytest.mark.parametrize(("published_attempt", "result_attempt", "publish_current"), [
    (None, None, False), (2, 2, False),  # equal known and wholly legacy result enrichment still work
    (None, 0, False), (None, 1, False), (None, 2, False), (None, 2, True),
    (0, None, False), (1, None, False), (2, None, False),  # a numbered publication never borrows an unknown result
])
def test_project_history_scopes_the_current_result_to_its_own_attempt(
        tmp_path, published_attempt, result_attempt, publish_current):
    from ouroboros.projects_registry import create_project

    tid, room = "retried-report", create_project(tmp_path, "reports", name="Reports")["chat_id"]
    old = row(1, chat_id=room, direction="system", type="task_summary", task_id=tid, text="Old publication",
              ts="2026-09-12T00:00:01Z", status="failed", task_terminal_status="failed",
              outcome_phase="error", outcome_final=True, reason_code="tool_error", terminal_origin="host_notice",
              wake_summary="Old closing sentence.", wake_silent=True, quiet_effective=True, rounds=4, tool_calls=3,
              accounted_upper_bound_usd=1.0)
    if published_attempt is not None:
        old["task_attempt"] = published_attempt
    result = {"status": "completed", "chat_id": room, "reason_code": "final_message", "terminal_origin": "model_final",
              "result": "Current answer bytes", "wake_summary": "New closing sentence.", "schedule_name": "New schedule",
              "loop_outcome": {"usage": {"total_rounds": 17}}, "accounted_upper_bound_usd": 2.0, "tool_calls": 22}
    if result_attempt is not None:
        result["task_attempt"] = result_attempt
    seed_result(tmp_path, tid, result)
    chat = [old, row(2, chat_id=room, direction="out", task_id=tid, text="Ordinary saved speech")]
    if publish_current:
        chat.append(row(3, chat_id=room, direction="system", type="task_summary", task_id=tid, text="New publication",
                        ts="2026-09-12T00:00:03Z", status="completed", task_attempt=result_attempt,
                        wake_silent=False, quiet_effective=False, tool_calls=22))
    write(tmp_path / "logs" / "chat.jsonl", chat)

    code, payload = request(tmp_path, chat_id=str(room))
    assert code == 200
    visible = {message["text"]: message for message in payload["messages"] if message.get("task_id") == tid}
    ending = visible["Old publication"]
    assert visible["Ordinary saved speech"]["role"] == "assistant"
    assert ending["history_id"].startswith("chat:") and ending["ts"] == old["ts"]
    if published_attempt == result_attempt:
        assert ending.get("task_attempt") == published_attempt
        assert (ending["task_terminal_status"], ending["reason_code"], ending["rounds"], ending["wake_summary"],
                ending["accounted_upper_bound_usd"]) == ("completed", "final_message", 17, "New closing sentence.", 2.0)
    else:
        assert ending.get("task_attempt") == published_attempt and "schedule_name" not in ending
        for key in ("status", "task_terminal_status", "outcome_phase", "outcome_final", "reason_code", "terminal_origin",
                    "wake_summary", "wake_silent", "quiet_effective", "rounds", "tool_calls", "accounted_upper_bound_usd"):
            assert ending[key] == old[key], key
        current = ending["current_outcome"]
        assert (current.get("task_attempt"), current["status"], current["reason_code"], current["terminal_origin"],
                current["wake_summary"], current["schedule_name"], current["quiet_effective"]) == (
                    result_attempt, "completed", "final_message", "model_final", "New closing sentence.", "New schedule", False)
        assert current["rounds"] == 17
        assert not {"tool_calls", "accounted_upper_bound_usd", "ts", "text"} & current.keys()
    if publish_current:
        second = visible["New publication"]
        assert second["history_id"] != ending["history_id"]
        assert (second["task_attempt"], second["task_terminal_status"], second["rounds"], second["tool_calls"]) == (2, "completed", 17, 22)
        assert "continuation_offer" not in ending
        assert ending["current_outcome"]["continuation_offer"] == second["continuation_offer"]
    assert "Current answer bytes" not in json.dumps(payload)


@pytest.mark.parametrize(("current_call", "known_nonquiet", "expected_quiet"), [
    ("accepted", {}, True), ("another", {}, False), ("accepted", {"wake_silent": False}, False),
    ("accepted", {"owner_message_sent": True}, False),
])
def test_unnumbered_canonical_publication_can_corroborate_only_its_solves_quiet_flag(
        tmp_path, current_call, known_nonquiet, expected_quiet):
    from ouroboros.projects_registry import create_project
    from ouroboros.terminal_projection import _project_row

    tid, room = "quiet-report", create_project(tmp_path, "quiet", name="Quiet")["chat_id"]
    solve = {"source": "usable_solve_response", "llm_call_id": "accepted"}
    old = _project_row(tid, {"status": "completed", "chat_id": room, "root_task_id": tid,
                             "wake_silent": True, "wake_summary": "Words from the old row.", "model_execution": solve},
                       {}, {"chat_id": room})
    old["ts"] = "2026-09-12T00:00:01Z"
    write(tmp_path / "logs" / "chat.jsonl", [old])
    seed_result(tmp_path, tid, {"status": "completed", "chat_id": room, "task_attempt": 2,
                                "model_execution": {**solve, "llm_call_id": current_call}, **known_nonquiet})
    code, payload = request(tmp_path, chat_id=str(room))
    assert code == 200
    [ending] = [message for message in payload["messages"] if message.get("task_id") == tid]
    assert ending["text"] == old["text"] and ending["quiet_effective"] is True and "task_attempt" not in ending
    assert ending["wake_summary"] == old["wake_summary"]
    current = ending["current_outcome"]
    assert current["task_attempt"] == 2 and current["quiet_effective"] is expected_quiet
    assert "wake_summary" not in current and "ts" not in current


@pytest.mark.parametrize(("observed", "stamped", "expected"), [
    ("0", None, 0), ("", None, None), ("2", None, 2), ("0", 1, 1),
])
def test_progress_keeps_the_executor_attempt_recorded_at_the_event(tmp_path, observed, stamped, expected):
    tid = "executor-retry"
    witness = {"task_id": tid, "task_attempt": observed, "run_id": "session", "attempt_id": "dispatch",
               "harness_id": "codex", "phase": "running", "revision": 0}
    progress = row(1, "progress", task_id=tid, executor_observation=witness)
    if stamped is not None:
        progress["task_attempt"] = stamped
    write(tmp_path / "logs" / "progress.jsonl", [progress])
    seed_result(tmp_path, tid, {"status": "failed", "task_attempt": 2, "chat_id": 1, "reason_code": "tool_error", "tool_calls": 9,
                                "suggested_name": "Second attempt", "model_execution": {"used_model": "second-model"}})
    code, response = request(tmp_path)
    assert code == 200
    [frame] = [r for r in response["messages"] if r["is_progress"]]
    if expected is None:
        assert "task_attempt" not in frame
    else:
        assert frame["task_attempt"] == expected
    if stamped is None:
        assert frame["executor_observation"] == witness
    else:
        assert "executor_observation" not in frame  # contradicts the row's explicit attempt
    if expected == 2:
        assert (frame["task_terminal_status"], frame["model_execution"]["used_model"]) == ("failed", "second-model")
        assert frame["tool_calls"] == 9
    else:
        assert not {"task_terminal_status", "reason_code", "model_execution", "suggested_name", "tool_calls"} & frame.keys()
        assert (frame["current_outcome"]["task_attempt"], frame["current_outcome"]["outcome_phase"]) == (2, "error")


@pytest.mark.parametrize("field", ["task_outcome", "current_outcome"])
@pytest.mark.parametrize("hidden", [False, True])
def test_legacy_summary_exposes_its_recorded_outcome_without_copying_the_answer(tmp_path, field, hidden):
    origin = {"place_id": "retained-project", "place_label": "Project thread"}
    outcome = {"status": "failed", "reason_code": "finalization_incomplete", "terminal_origin": "model_final",
               "finalization_recovery": {"answer": "delivered"}, "task_attempt": 1,
               "artifact_bundle": {"status": "missing"}, "place_origin": origin}
    entry = row(1, direction="system", type="task_summary", task_id="legacy",
                **{field: {**outcome, "artifact_bundle": {"status": "missing", "files": ["private filename"]}, "result": "private full answer"}})
    if hidden:
        entry.update(summary_kind="terminal_root_projection", outcome_authority="canonical_task_result_after_finalization",
                     outcome_final=True, outcome_phase="error", status="failed")
        write(tmp_path / "logs" / "progress.jsonl", [row(2, "progress", task_id="legacy")])
    write(tmp_path / "logs" / "chat.jsonl", [entry])
    code, response = request(tmp_path)
    assert code == 200
    [summary] = [r for r in response["messages"] if r.get("system_type") == "task_summary"]
    assert summary[field] == outcome and "outcome_axes" not in summary
    assert "private full answer" not in json.dumps(summary) and "private filename" not in json.dumps(summary)


def test_pruned_project_summary_remains_a_visible_row_with_its_frozen_ending(tmp_path):
    from ouroboros.projects_registry import create_project

    room = create_project(tmp_path, "card-history", name="Card history")["chat_id"]
    facts = row(0, chat_id=room, task_id="report", direction="system", type="task_summary",
                summary_kind="host_task_facts", text="", rounds=3, task_attempt=1, status="completed")
    ending = {**facts, "summary_kind": "terminal_root_projection", "text": "Published completion",
              "outcome_authority": "canonical_task_result_after_finalization", "outcome_final": True,
              "outcome_phase": "done", "ts": "2026-09-12T00:00:01Z"}
    older = row(0, chat_id=room, task_id="report", direction="system", type="task_summary",
                text="Previous result", status="failed", reason_code="tool_error", rounds=1)
    write(tmp_path / "logs/chat.jsonl", [older, facts, ending])
    code, payload = request(tmp_path, chat_id=str(room))
    assert code == 200
    visible = [r for r in payload["messages"] if r.get("task_id") == "report"]
    [legacy] = [r for r in visible if r["text"] == "Previous result"]
    assert (legacy["status"], legacy["reason_code"], legacy["rounds"]) == ("failed", "tool_error", 1)
    assert not {"task_attempt", "historical_terminal"} & legacy.keys()
    assert [(r["text"], r["historical_terminal"]["phase"]) for r in visible if r is not legacy] == [
        ("", "done"), ("Published completion", "done")]
    assert all("summary_kind" not in r for r in visible)  # otherwise the client skips the entire row


@pytest.mark.parametrize("ending_first", [False, True])
def test_legacy_status_keeps_the_stamped_delivery_warning_after_result_pruning(tmp_path, ending_first):
    from ouroboros.terminal_projection import _project_row

    origin = {"place_id": "exports", "place_label": "Export requests"}
    legacy = row(1, direction="system", type="task_summary", task_id="report", task_attempt=1,
                 status="completed", text="Legacy status", ts="2026-09-12T00:00:03Z")
    ending = _project_row("report", {"status": "completed", "task_attempt": 1, "root_task_id": "report"},
                          {}, {"chat_id": 1}, delivery={"state": "exhausted"})
    ending.update(ts="2026-09-12T00:00:02Z", place_origin=origin)
    assert (ending["outcome_phase"], ending["terminal_delivery"]["state"]) == ("warn", "exhausted")
    write(tmp_path / "logs" / "chat.jsonl", [ending, legacy] if ending_first else [legacy, ending])
    code, response = request(tmp_path)
    assert code == 200
    [summary] = [r for r in response["messages"] if r["text"] == "Legacy status"]
    assert summary["status"] == "completed" and "outcome_axes" not in summary and "current_outcome" not in summary
    frozen = summary["historical_terminal"]
    assert (frozen["phase"], frozen["task_attempt"], frozen["place_origin"]) == ("warn", 1, origin)


@pytest.mark.parametrize("field", ["task_outcome", "current_outcome"])
def test_a_stored_new_attempt_supersedes_a_nested_summary(tmp_path, field):
    tid = "nested-attempt"
    old = {"status": "failed", "reason_code": "tool_error", "task_attempt": 1}
    write(tmp_path / "logs" / "chat.jsonl", [row(1, direction="system", type="task_summary", task_id=tid, **{field: old})])
    seed_result(tmp_path, tid, {"chat_id": 1, "task_attempt": 2, "status": "completed", "reason_code": "final_message"})
    code, response = request(tmp_path)
    assert code == 200
    [summary] = [r for r in response["messages"] if r.get("system_type") == "task_summary"]
    current = summary["current_outcome"]
    assert summary["task_outcome"] == old
    assert summary["task_attempt"] == 1 and "task_terminal_status" not in summary
    assert (current["task_attempt"], current["status"], current["reason_code"]) == (2, "completed", "final_message")


@pytest.mark.parametrize(("field", "attempt", "status", "phase", "same_revision"), [
    pytest.param("current_outcome", 1, "completed", "warn", False, id="legacy-same-attempt-new-text"),
    pytest.param("current_outcome", 1, "completed", "warn", True, id="same-result-revision"),
    ("task_outcome", 1, "failed", "error", False),
    ("current_outcome", 2, "completed", "done", False), ("task_outcome", 2, "failed", "error", False),
])
def test_saved_delivery_warning_follows_its_attempt_across_history_pages(
        tmp_path, monkeypatch, field, attempt, status, phase, same_revision):
    from ouroboros.gateway import history_paging
    from ouroboros.late_review_marks import current_outcome
    from ouroboros.terminal_projection import _project_row

    tid = "paged-report"
    saved_copy = tmp_path / "first-answer.txt"
    saved_copy.write_text("First answer", encoding="utf-8")
    delivery = {"state": "exhausted", "preserved_path": str(saved_copy)}
    saved = seed_result(tmp_path, tid, {
        "status": "completed", "reason_code": "final_message", "task_attempt": 1, "root_task_id": tid,
        "chat_id": 1, "result": "First answer", "updated_at": "2026-11-10T10:00:00Z",
    })
    old = {**current_outcome(saved, delivery), "task_attempt": 1}
    if not same_revision:
        old.pop("revision")  # this legacy receipt identifies only the attempt
    rich = row(1, direction="system", type="task_summary", task_id=tid, text="Published report", **{field: old})
    receipt = _project_row(tid, saved, {}, {"chat_id": 1}, delivery=delivery)
    write(tmp_path / "logs" / "chat.jsonl", [row(0), rich, row(2), receipt, row(4)])
    if not same_revision:
        seed_result(tmp_path, tid, {"status": status, "task_attempt": attempt, "chat_id": 1, "result": "Second answer",
                                    "updated_at": "2026-11-10T11:00:00Z",
                                    "reason_code": "tool_error" if status == "failed" else "final_message"})
    monkeypatch.setattr(history_paging, "_RECENT_WINDOW_ROWS", 1)
    loaded = list(pages(tmp_path, n_human="1"))
    located = [(i, r) for i, page in enumerate(loaded) for r in page["messages"] if r.get("task_id") == tid]
    [(receipt_page, compact), (rich_page, summary)] = located
    assert 0 < receipt_page < rich_page and compact["history_id"] != summary["history_id"]
    assert compact["historical_terminal"]["phase"] == "warn" and compact["historical_terminal"]["task_attempt"] == 1
    assert compact["historical_terminal"]["terminal_delivery"] == delivery
    assert summary["text"] == "Published report"
    current = summary["current_outcome"]
    assert summary["task_outcome"]["terminal_delivery"] == delivery
    assert (current["task_attempt"], current["status"], current["outcome_phase"]) == (attempt, status, phase)
    if attempt == 1:
        warning = {"state": "exhausted", "scope": "task"}
        assert current["terminal_delivery"] == (delivery if same_revision else warning)
        if not same_revision:
            assert compact["current_outcome"]["terminal_delivery"] == warning
        assert (summary["task_terminal_status"], summary["outcome_phase"]) == (status, phase)
        assert "terminal_delivery_exhausted" in summary["outcome_axes"]["objective"]["warnings"]
        assert "terminal_delivery_exhausted" in current["outcome_axes"]["objective"]["warnings"]
    else:
        assert compact["current_outcome"]["status"] == status and compact["current_outcome"]["task_attempt"] == attempt
        assert "terminal_delivery" not in current and "terminal_delivery_exhausted" not in json.dumps(current)


@pytest.mark.parametrize("debt", ["old-answer", "same-answer", "same-answer-canonical-drive", "same-answer-gone-replica",
                                  "old-revision", "other-chat", "other-attempt", "unidentified-replica",
                                  "unidentified-answer", "unidentified-chat", "adopted-answer",
                                  "shared-current-copy", "shared-old-copy"])
def test_retry_on_an_older_page_shows_the_current_answers_outbox_disposition(tmp_path, monkeypatch, debt):
    from ouroboros.gateway import history_paging
    from ouroboros.observability import preserve_salvaged_output
    from ouroboros.terminal_outbox import _registry_document, _registry_path
    from ouroboros.terminal_projection import completion_notice
    from supervisor.terminal_delivery import delivery_id_for

    tid = "delayed-delivery"
    old_id, current_id = (delivery_id_for(tid, words) for words in ("First answer", "Second answer"))
    old_path, current_path = "saved/first.txt", "saved/second.txt"
    if debt.startswith(("shared-", "same-answer-")):
        copied = "First answer" if debt == "shared-old-copy" else "Second answer"
        old_path = current_path = preserve_salvaged_output(tmp_path, f"{tid}.undelivered", copied)
    old_debt = {"state": "exhausted", "delivery_id": old_id, "preserved_path": old_path}
    old = {"status": "failed", "task_attempt": 2 if debt == "old-revision" else 1, "reason_code": "tool_error",
           "terminal_delivery": old_debt}
    summary = row(1, direction="system", type="task_summary", task_id=tid, task_outcome=old)
    if debt == "old-revision":
        summary.update(task_attempt=2, terminal_delivery=old_debt)
    write(tmp_path / "logs/chat.jsonl", [summary, row(2), row(3)])
    current = {"chat_id": 1, "task_attempt": 2, "status": "completed", "reason_code": "final_message"}
    if debt == "same-answer-canonical-drive":
        current["drive_root"] = str(tmp_path)
    if debt in {"same-answer-gone-replica", "unidentified-replica"}:
        replica = tmp_path / "child-drive"
        current["drive_root"] = str(replica)
        if debt == "unidentified-replica":
            seed_result(replica, tid, {**current, "result": "Answer from the replica"})
    if debt != "unidentified-answer":
        current["result"] = "Second answer"
    if debt == "adopted-answer":
        current.update(status="failed", reason_code="finalization_incomplete", terminal_origin="host_salvage",
                       result="Original unabridged output", finalization_recovery={
                           "answer": "exhausted", "delivery_id": current_id, "attempt": 2})
    seed_result(tmp_path, tid, current)
    monkeypatch.setattr(history_paging, "_RECENT_WINDOW_ROWS", 1)
    # The old debt was recorded last. Ordering by exhausted_at would select it even when both answers failed.
    exhausted = {old_id: {"task_id": tid, "chat_id": 1, "text": "First answer", "preserved_path": old_path,
                          "exhausted_at": "2026-11-10T12:00:00Z"}}
    if debt not in {"old-answer", "old-revision", "unidentified-answer"}:
        exhausted[current_id] = {"task_id": tid, "text": "Second answer", "exhausted_at": "2026-11-10T11:00:00Z",
                                 "preserved_path": current_path}
        if debt != "unidentified-chat":
            exhausted[current_id]["chat_id"] = 4 if debt == "other-chat" else 1
        if debt == "other-attempt":
            exhausted[current_id]["progress_meta"] = {"task_attempt": 1}
    path = _registry_path(tmp_path)
    path.write_text(json.dumps(_registry_document({}, [], {}, exhausted=exhausted)), encoding="utf-8")

    def outcomes():
        loaded = list(pages(tmp_path, n_human="1"))
        assert all(r.get("task_id") != tid for r in loaded[0]["messages"])
        [summary] = [r for page in loaded[1:] for r in page["messages"] if r.get("task_id") == tid]
        assert summary["task_outcome"] == old
        return summary["current_outcome"]

    outcome = outcomes()
    assert outcome["task_attempt"] == 2
    warnings = outcome["outcome_axes"].get("objective", {}).get("warnings", [])
    detail = completion_notice(outcome)["detail"]
    if debt.startswith("same-answer") or debt in {"adopted-answer", "shared-current-copy", "shared-old-copy"}:
        pointer = None if debt == "shared-old-copy" else current_path
        assert outcome["terminal_delivery"] == {"state": "exhausted", "delivery_id": current_id, "chat_id": 1,
                                               "preserved_path": pointer}
        assert "terminal_delivery_exhausted" in warnings
        if pointer:
            assert f"a copy of the answer is saved at {pointer}" in detail
        else:
            assert "saved copy for this answer could not be verified" in detail and current_path not in detail
        assert outcome["outcome_phase"] == ("error" if debt == "adopted-answer" else "warn")
    elif debt.startswith("unidentified-"):
        assert outcome["terminal_delivery"] == {"state": "exhausted", "scope": "task"}
        assert outcome["outcome_phase"] == "warn" and "terminal_delivery_exhausted" in warnings
        assert "affected answer is unknown" in detail and "saved/" not in detail
    else:
        assert "terminal_delivery" not in outcome and outcome["outcome_phase"] == "done"
        assert "terminal_delivery_exhausted" not in warnings
    assert "saved/first.txt" not in json.dumps(outcome) and "Second answer" not in json.dumps(outcome)
    if debt == "same-answer":
        path.write_text("{", encoding="utf-8")
        assert outcomes()["terminal_delivery"] == {"state": "unknown"}
        path.write_text(json.dumps(_registry_document({}, [], {})), encoding="utf-8")
        assert outcomes()["outcome_phase"] == "done"


@pytest.mark.parametrize("current", ["delivered", "saved", "overwritten"])
def test_source_delivery_across_project_main_late_card_and_history(tmp_path, monkeypatch, current):
    from collections import deque

    from ouroboros.late_review_marks import announce_current_outcome
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from ouroboros.task_results import write_task_result
    from ouroboros.terminal_projection import SETTLEMENT_DEFERRED, completion_notice, settle_terminal_projection
    from ouroboros.utils import append_jsonl
    from supervisor import events_chat_delivery, message_bus, terminal_delivery as outbox
    from tests._cancel_intents_shared import DeliveryBus
    from tests._dialogue import chat_rows

    tid = "two-answers"
    project = create_project(tmp_path, "launch", name="Launch")
    chat = project["chat_id"]
    bind_task_to_project(tmp_path, tid, project["id"], chat, origin={"absent": "system"})
    task = {"id": tid, "root_task_id": tid, "project_id": project["id"], "chat_id": chat, "_attempt": 3}
    write_task_result(tmp_path, tid, "running", chat_id=chat, root_task_id=tid, project_id=project["id"], task_attempt=3)
    answers = [outbox.build_completed_result_event(tmp_path, task, tid, {"result": words, "terminal_origin": "model_final"})
               for words in ("First answer", "Current answer")]
    old, new = answers
    for event in answers:
        assert event["chat_id"] == chat and outbox.register_pending_delivery(tmp_path, event)
    write_task_result(tmp_path, tid, "completed", chat_id=chat, root_task_id=tid, project_id=project["id"],
                      task_attempt=3, terminal_origin="model_final", result=new["text"])

    def retire(event):
        with monkeypatch.context() as worker:
            worker.setattr("ouroboros.utils.in_worker_process", lambda: True)
            outbox._retire_exhausted(tmp_path, [event])

    bus = DeliveryBus()
    bridge = message_bus.LocalChatBridge({})
    frames = []
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 1})
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(message_bus, "try_get_bridge", lambda: None)
    monkeypatch.setattr(message_bus, "publish_event", lambda *_a, **_kw: None)
    monkeypatch.setattr(events_chat_delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    sender = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, EVENT_Q=bus,
                             send_with_budget=message_bus.send_with_budget, append_jsonl=append_jsonl)
    if current == "delivered":
        assert events_chat_delivery._handle_send_message(new, sender)
        assert any(f.get("delivery_id") == new["delivery_id"] and f["content"] == new["text"] for f in frames)
    else:
        retire(old if current == "saved" else new)
    pending = new if current == "saved" else old
    assert outbox.terminal_answer_receipts(tmp_path, tid)["owed_delivery_ids"] == [pending["delivery_id"]]
    waits = {}
    assert settle_terminal_projection(tmp_path, tid, event_queue=bus, waits=waits) == SETTLEMENT_DEFERRED
    assert waits["on"] == "terminal_delivery" and bus.empty()
    assert not any(r.get("summary_id") == f"task-terminal:{tid}" for r in chat_rows(tmp_path))

    retire(pending)
    assert settle_terminal_projection(tmp_path, tid, event_queue=bus) == SETTLEMENT_DEFERRED  # only Main is owed
    [card] = [r for r in chat_rows(tmp_path) if r.get("summary_id") == f"task-terminal:{tid}"]
    mirror = bus.get_nowait()
    assert bus.empty() and mirror["progress_meta"]["completion_answer"] == new["text"]
    assert events_chat_delivery._handle_send_message(mirror, sender)
    code, main = request(tmp_path, chat_id="1")
    assert code == 200
    [mirror_row] = [r for r in main["messages"] if r.get("system_type") == "project_completion_summary"]
    assert mirror_row["completion_answer"] == new["text"] and mirror_row["text"] == mirror["text"]
    [old_mark] = [r for r in outbox.terminal_answer_receipts(tmp_path, tid)["exhausted"]
                  if r["delivery_id"] == old["delivery_id"]]
    shared = old_mark["preserved_path"]
    assert shared and Path(shared).read_text(encoding="utf-8") == pending["text"]
    expected = None if current == "delivered" else {
        "state": "exhausted", "delivery_id": new["delivery_id"], "chat_id": chat,
        "preserved_path": shared if current == "saved" else None}
    assert card.get("terminal_delivery") == expected

    write_task_result(tmp_path, tid, "completed", outcome_axes={"review": {"late_verdict": {"status": "PASS"}}})

    def revised():
        live = announce_current_outcome(tmp_path, tid, "review-b", "subject-b")["current_outcome"]
        code, page = request(tmp_path, chat_id=str(chat))
        assert code == 200
        [ending] = [r for r in page["messages"] if r.get("task_id") == tid and r.get("system_type") == "task_summary"]
        return live, ending["current_outcome"]

    live, reloaded = revised()
    assert live["task_attempt"] == reloaded["task_attempt"] == 3
    assert live.get("terminal_delivery") == reloaded.get("terminal_delivery") == expected
    details = (card["text"], card["completion_notice"]["detail"], mirror_row["text"], mirror_row["completion_notice"]["detail"],
               completion_notice(live)["detail"], completion_notice(reloaded)["detail"])
    for detail in details:
        assert (shared in detail) is (current == "saved")
        assert ("could not be verified" in detail) is (current == "overwritten")
        assert ("delivery attempts ran out" in detail) is (current != "delivered")
    # The bounded outbox can forget these failures. A published warning outlives it,
    # and its common filename still has to contain this answer at the next reading.
    registry = tmp_path / "state" / "terminal_deliveries.json"
    data = json.loads(registry.read_text(encoding="utf-8"))
    data.pop("exhausted")
    registry.write_text(json.dumps(data), encoding="utf-8")
    assert all(outcome.get("terminal_delivery") == expected for outcome in revised())
    if current == "saved":
        from ouroboros.observability import preserve_salvaged_output

        assert preserve_salvaged_output(tmp_path, f"{tid}.undelivered", old["text"]) == shared
        for outcome in revised():
            assert outcome["terminal_delivery"] == {**expected, "preserved_path": None}
            assert shared not in completion_notice(outcome)["detail"]


def test_required_source_failure_retries_the_same_page(tmp_path, monkeypatch):
    write(tmp_path / "logs" / "chat.jsonl", [row(index) for index in range(200)])
    first = request(tmp_path)[1]
    original = ds.probe_rows
    monkeypatch.setattr(ds, "probe_rows", lambda *_a, **_k: (_ for _ in ()).throw(StoreError("injected store failure")))
    status, failed = request(tmp_path, cursor=first["next_cursor"])
    assert status == 503 and failed["reason_code"] == "history_source_unavailable"
    assert failed["next_cursor"] == first["next_cursor"] and failed["has_more"] is True
    monkeypatch.setattr(ds, "probe_rows", original)
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
    original = ds.probe_rows

    def probe(root, stream, keys, **kwargs):
        seen.append((stream, kwargs["lower"], kwargs["before"]))
        return original(root, stream, keys, **kwargs)

    monkeypatch.setattr(ds, "probe_rows", probe)
    assert request(tmp_path, cursor=encoded)[0] == 200
    assert seen and all(end <= cursor["before"][source] for source, _start, end in seen)


def test_a_torn_line_an_older_body_left_is_a_gap_outside_frozen_history(tmp_path):
    """A store-unaware body appended after the cutover and did not finish its last line: the next
    cutover imports its rows at their place and records the torn line as a skipped interval -- a
    disclosed gap above the frozen page, never a row and never a change to that page."""
    path = tmp_path / "logs" / "chat.jsonl"
    write(path, [row(index) for index in range(200)])
    first = request(tmp_path)[1]
    with path.open("ab") as handle:
        handle.write(json.dumps(row(200, text="whole")).encode() + b"\n" + b'{"direction":"in","text":"later')
    replayed = request(tmp_path, cursor=first["page_cursor"])[1]
    assert replayed["messages"] == first["messages"]
    status, fresh = request(tmp_path)
    assert status == 200 and "chat_malformed_jsonl" in fresh["window"]["truncated_by"]
    assert any(message["text"] == "whole" for message in fresh["messages"])
    assert not any(message["text"] == "later" for page in pages(tmp_path) for message in page["messages"])


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
    "chat", False, [{}],
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
        seed_result(tmp_path, task, {"_schema_version": 1, "task_id": task, "status": "completed"})
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


def test_main_refuses_its_recent_read_while_no_row_of_the_project_registry_can_be_read(tmp_path, monkeypatch, caplog):
    """Main is "every room that is not a Project": decided from an empty set (a registry no row of
    which can be read) it would take the owner's rows typed in Project rooms (#3067's mechanism). It
    answers the typed unavailable refusal instead. An invalid row beside readable ones is no such
    case: the rooms the readable rows name are still Projects, and Main reads on. The refusal names its cause,
    in its body and once in the server log, since nothing else would say why Main stays empty."""
    from ouroboros import projects_registry
    from ouroboros.projects_registry import bind_task_to_project, create_project

    monkeypatch.setattr(projects_registry, "_UNREADABLE_SAID", set())  # this process has named no cause yet
    said = lambda: [r.levelname for r in caplog.records if "vouches for no room" in r.getMessage()]  # noqa: E731

    rooms = [create_project(tmp_path, slug, name=slug) for slug in ("alpha", "beta")]
    bind_task_to_project(tmp_path, "bound", rooms[0]["id"], origin={"absent": "system"})
    write(tmp_path / "logs" / "chat.jsonl", [
        row(0, text="main owner"),
        *(row(1, text=f"room owner {room['chat_id']}", chat_id=room["chat_id"]) for room in rooms),
        row(2, direction="out", text="bound reply", task_id="bound", chat_id=rooms[0]["chat_id"]),
    ])
    room_rows = {f"room owner {room['chat_id']}" for room in rooms} | {"bound reply"}
    texts = lambda payload: {message.get("text") for message in payload["messages"]}  # noqa: E731
    status, healthy = request(tmp_path)
    assert status == 200 and "main owner" in texts(healthy) and not texts(healthy) & room_rows
    assert "detail" not in healthy

    registry = tmp_path / "state" / "projects.json"
    intact = registry.read_bytes()
    registry.write_bytes(intact[: len(intact) // 2])  # no row can be read
    status, refused = request(tmp_path)
    assert not texts(refused) & room_rows, "a Project room's row reached Main's window"
    assert status == 503 and refused["reason_code"] == "history_source_unavailable"
    assert refused["has_more"] is True and refused["window"]["complete"] is False
    assert refused["detail"] == "Project registry unavailable", "the refusal names its cause"
    assert said() == ["WARNING"] and request(tmp_path)[0] == 503 and said() == ["WARNING"], "once a cause"

    data = json.loads(intact)
    registry.write_text(json.dumps({**data, "projects": [*data["projects"], {"chat_id": 7}]}), encoding="utf-8")
    status, beside = request(tmp_path)
    assert status == 200 and "main owner" in texts(beside) and not texts(beside) & room_rows

    for unknown in ("lost beside its commit witness", "only an invalid row"):  # neither is "no Projects"
        registry.unlink(missing_ok=True)
        if unknown == "only an invalid row":
            registry.write_text(json.dumps({**data, "projects": [{"chat_id": 7}]}), encoding="utf-8")
        status, refused = request(tmp_path)
        assert status == 503 and not texts(refused) & room_rows, unknown
        assert refused["detail"] == "Project registry unavailable", unknown

    registry.write_bytes(intact)
    restored = list(pages(tmp_path))
    assert "main owner" in {text for page in restored for text in texts(page)}
    assert not {text for page in restored for text in texts(page)} & room_rows

    # One lenient read decides the request: the rooms it named stay Projects even when a second read would fail
    # (an open that fails on and off), never "no Projects" within the same request.
    from ouroboros.gateway import history as history_module

    named, lenient = projects_registry.readable_rooms(tmp_path), projects_registry.reserved_project_chat_ids
    answers = iter([named])
    monkeypatch.setattr(projects_registry, "reserved_project_chat_ids", lambda root, *, strict=False: (
        (_ for _ in ()).throw(OSError("too many open files")) if strict else lenient(root)))
    monkeypatch.setattr(projects_registry, "readable_rooms", lambda root: next(answers, None))
    context = history_module._project_history_context(tmp_path, 1)
    assert named and context[0] == named and context[-1] is False


def test_main_reads_on_a_fresh_install_without_a_project_registry(tmp_path):
    write(tmp_path / "logs" / "chat.jsonl", [row(0, text="main owner")])
    assert not (tmp_path / "state" / "projects.json").exists()
    status, payload = request(tmp_path)
    assert status == 200 and [message["text"] for message in payload["messages"]] == ["main owner"]
