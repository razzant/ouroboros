"""The wake-up message and envelope (Background Consciousness redesign P2), and the
server's projection of the alarm clock's snapshot into one honest status line."""

from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

import pytest

from ouroboros import consciousness_wake as wake

REPO = pathlib.Path(__file__).resolve().parents[1]
T0 = 1_800_000_000.0


def _iso(ts):
    return wake._iso(ts)


def _result(root, task_id, *, status="completed", ts, cost=1.25, direct=False, quizzes=None, description="",
            project_id=""):
    (root / "task_results").mkdir(exist_ok=True)
    row = {"task_id": task_id, "status": status, "ts": _iso(ts), "updated_at": _iso(ts), "_schema_version": 1,
           "accounted_upper_bound_usd": cost, "description": description or f"do {task_id}",
           "metadata": {}, "_is_direct_chat": direct}
    if project_id:
        row["project_id"] = project_id
    if quizzes:
        row["owner_quiz"] = quizzes
    (root / "task_results" / f"{task_id}.json").write_text(json.dumps(row), encoding="utf-8")


def test_wake_task_metadata_carries_origin_level_and_tree_cap(monkeypatch):
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_AUTONOMY", "act")
    meta = wake.wake_task_metadata("observe", "heartbeat", root_cost_ceiling_usd=3.5)
    assert meta["initiator"] == "consciousness" and meta["usage_category"] == "consciousness"
    assert meta["consciousness_autonomy"] == "observe" and meta["runtime_mode_cap"] == "light"
    assert meta["model_role"] == "consciousness" and meta["wake_reason"] == "heartbeat"
    # P3e: the cap the wake's own root scope binds, never the non-root member ceiling.
    assert "promote_chat_to_task" in meta["disabled_tools"] and meta["root_cost_ceiling_usd"] == 3.5
    assert "root_limit_usd" not in meta
    full = wake.wake_task_metadata("full", "event:x")
    assert full["disabled_tools"] == [] and full["runtime_mode_cap"] == "" and "root_cost_ceiling_usd" not in full
    # A non-positive cap is not stamped: it would read as "no narrowing" downstream and
    # the alarm never launches on an exhausted allowance anyway.
    assert "root_cost_ceiling_usd" not in wake.wake_task_metadata("act", "heartbeat", root_cost_ceiling_usd=0.0)
    assert wake.wake_task_metadata("bogus", "")["consciousness_autonomy"] == "act"  # falls back to the setting


def test_events_list_settled_tasks_open_cards_and_owner_messages_since_the_last_wake(tmp_path):
    since = T0 - 3600
    _result(tmp_path, "old01", ts=since - 10)
    _result(tmp_path, "new01", ts=since + 10, status="failed", cost=0.5, description="build the thing")
    _result(tmp_path, "new02", ts=since + 50, status="completed", cost=1.0, description="later thing")
    _result(tmp_path, "run01", ts=since + 20, status="running")
    _result(tmp_path, "chat1", ts=since + 30, direct=True)  # an owner's own turn: already in Recent chat
    _result(tmp_path, "ask01", ts=since - 100, status="running",
            quizzes={"q1": {"state": "open", "asked_at": _iso(since + 5)}, "q2": {"state": "answered", "answered_at": "x"}})
    # A backlog of five older cards on one task: only the newest CARD_LINES_MAX cards are listed,
    # newest first, so old cards never starve the settled lines below (opus round 3).
    _result(tmp_path, "ask02", ts=since - 200, status="running",
            quizzes={f"c{i}": {"state": "open", "asked_at": _iso(since - 1000 + i)} for i in range(5)})
    # The previous wake's own card, left behind when its turn ended (expired_terminal, В17a:
    # still answerable), is exactly the "I'll come back to it" case — it must be listed even
    # though the wake's row itself is excluded from the settled-task lines.
    _result(tmp_path, "prev1", ts=since + 40, direct=True,
            quizzes={"q3": {"state": "expired_terminal", "asked_at": _iso(since + 40)},
                     "q4": {"state": "expired_terminal", "answered_at": "y"}})
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "chat.jsonl").write_text("\n".join([
        json.dumps({"direction": "in", "ts": _iso(since + 5), "text": "hi"}),
        json.dumps({"direction": "out", "ts": _iso(since + 6), "text": "hello"}),
        json.dumps({"direction": "in", "ts": _iso(since - 5), "text": "earlier"}),
        json.dumps({"direction": "in", "ts": _iso(since + 7), "text": "again"}),
    ]) + "\n", encoding="utf-8")
    lines = wake.wake_events(tmp_path, since=since, now=T0, exclude_task_id="prev1")
    assert any(line.startswith("- owner card q1 on task ask01: Unanswered") for line in lines)
    assert any(line.startswith("- owner card q3 on task prev1: Unanswered · the task finished") for line in lines)
    assert not [line for line in lines if "q2" in line or "q4" in line]
    cards = [line.split()[3] for line in lines if line.startswith("- owner card ")]
    assert cards == ["q3", "q1", "c4", "c3"]  # newest four; c2..c0 wait for their turn
    assert "- task new01 failed, $0.50: build the thing" in lines
    settled = [line for line in lines if line.startswith("- task ")]
    assert [line.split()[2] for line in settled] == ["new02", "new01"]  # newest first
    assert lines.index(settled[0]) < next(i for i, line in enumerate(lines) if line.startswith("- owner card q1 "))
    assert "- 2 message(s) from your human (see Recent chat)" in lines
    assert not [line for line in lines if "old01" in line or "run01" in line or "chat1" in line or "- task prev1 " in line]
    assert wake.wake_events(tmp_path / "missing", since=since, now=T0) == []



def test_answered_cards_of_the_window_are_events_with_their_facts(tmp_path):
    since = T0 - 3600
    long_comment = "Keep the old parser.\n\nReason: " + "the migration cost is too high; " * 120 + "END-OF-COMMENT"
    _result(tmp_path, "early", ts=since + 100, cost=0.25, description="earlier settled work")
    _result(tmp_path, "late", ts=since + 3000, cost=0.75, description="later settled work")
    _result(tmp_path, "asker", ts=since - 500, status="completed", quizzes={
        # Answered by a button inside the window: stamps, the chosen label, a question preview.
        "qbtn": {"state": "answered", "asked_at": _iso(since - 7200), "answered_at": _iso(since + 600),
                 "options": ["Rewrite it", "Keep it"], "answered_index": 1,
                 "question": "Should the parser be rewritten before the release?"},
        # Answered in the owner's own words inside the window: the words arrive whole.
        "qown": {"state": "answered", "asked_at": _iso(since + 60), "answered_at": _iso(since + 2400),
                 "options": ["A", "B"], "comment": long_comment, "question": "Which parser?"},
        # Answered before the window: not news for this wake.
        "qold": {"state": "answered", "asked_at": _iso(since - 9000), "answered_at": _iso(since - 60),
                 "options": ["A", "B"], "answered_index": 0, "question": "Old question"},
        # Still unanswered after its task finished: listed as before, in the card list.
        "qopen": {"state": "expired_terminal", "asked_at": _iso(since - 300), "question": "Still waiting?"},
    })
    lines = wake.wake_events(tmp_path, since=since, now=T0, reason="task_finished:asker:completed")
    btn = next(line for line in lines if line.startswith("- owner card qbtn "))
    assert btn == ("- owner card qbtn on task asker: answered 50 min ago; asked 3 h 0 min ago; "
                   "chose option 2: Keep it; question: Should the parser be rewritten before the release?")
    own = next(line for line in lines if line.startswith("- owner card qown "))
    assert own.startswith("- owner card qown on task asker: answered 20 min ago; asked 59 min ago; "
                          "answered in own words: Keep the old parser.")
    assert long_comment in own and "END-OF-COMMENT; question: Which parser?" in own  # whole, never clipped
    assert not any("qold" in line for line in lines)
    assert any(line.startswith("- owner card qopen on task asker: Unanswered · the task finished") for line in lines)
    # Answers sort with the settled facts by their own stamp (newest first), after the trigger;
    # the trigger's de-duplication of its own task never hides an answer on that task.
    assert lines[0].startswith("- wake cause: task asker finished")
    events = [line.split()[1:4] for line in lines[1:] if not line.startswith("- owner card qopen ")]
    assert events == [["task", "late", "completed,"], ["owner", "card", "qown"],
                      ["owner", "card", "qbtn"], ["task", "early", "completed,"]]
    # No verdict words about what the owner meant.
    assert not any(word in btn + own for word in ("understood", "confused", "misunderstood"))


def test_answered_card_line_renders_mixed_answers_and_missing_facts_honestly():
    now = T0
    both = wake._answered_card_line("t1", "q1", {
        "answered_at": _iso(now - 120), "asked_at": _iso(now - 240), "options": ["Go", "Stop"],
        "answered_index": 0, "comment": "but only on weekdays", "question": "Deploy?"}, now=now)
    assert both == ("- owner card q1 on task t1: answered 2 min ago; asked 4 min ago; "
                    "chose option 1: Go; with the words: but only on weekdays; question: Deploy?")
    bare = wake._answered_card_line("t1", "q2", {"answered_at": "not a stamp", "answered_index": 5}, now=now)
    assert bare == ("- owner card q2 on task t1: answered at an unknown time; asked at an unknown time; "
                    "chose option 6: label unavailable; question: question text unavailable")
    empty = wake._answered_card_line("t1", "q3", {"answered_at": _iso(now - 60)}, now=now)
    assert "answer text unavailable" in empty

@pytest.mark.parametrize("status, origin", [("failed", "host_notice"), ("cancelled", ""),
                                           ("completed", "host_notice"), ("failed", "model_final")])
def test_failed_inline_presence_is_visible_on_regular_wake_without_reviving_owner_turns(tmp_path, status, origin):
    from ouroboros.presence_runner import _build_task
    from ouroboros.task_results import write_task_result
    from tests.test_presence_runner import _admission, _event

    task = _build_task(_admission(), _event(), drive_root=tmp_path, staged_files=())
    assert task["_is_direct_chat"] is True
    metadata = {**task["metadata"], "presence_outcome": "deferred", "presence_result_text": "",
                "presence_work_ref": "still-running-child"}
    write_task_result(tmp_path, task["id"], status, _is_direct_chat=True, metadata=metadata,
                      terminal_origin=origin, result="Host diagnostic remains available in the task.")
    _result(tmp_path, "owner-failed", ts=T0, direct=True, status="failed")
    _result(tmp_path, "successful-presence", ts=T0, direct=True)
    path = tmp_path / "task_results" / "successful-presence.json"
    successful = json.loads(path.read_text(encoding="utf-8"))
    successful.update(metadata={"presence": {}, "presence_outcome": "message"}, terminal_origin="model_final")
    path.write_text(json.dumps(successful), encoding="utf-8")

    lines = wake.wake_events(tmp_path, since=0, now=T0, reason="heartbeat")
    fact = next(line for line in lines if line.startswith(f"- task {task['id']} "))
    assert f" {status}" in fact and "Presence outcome=deferred" in fact
    assert "get_task_result" in fact and "deferred work=still-running-child" in fact
    assert not any("owner-failed" in line or "successful-presence" in line for line in lines)
    assert not any(task["id"] in line for line in wake.wake_events(
        tmp_path, since=0, now=T0, reason="heartbeat", exclude_task_id=task["id"]))
    metadata["initiator"] = "consciousness"
    write_task_result(tmp_path, task["id"], status, _is_direct_chat=True, metadata=metadata, terminal_origin=origin)
    assert not any(task["id"] in line for line in wake.wake_events(tmp_path, since=0, now=T0, reason="heartbeat"))


def test_project_digest_pins_human_project_and_related_task_before_cards(tmp_path):
    since = T0 - 3600
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "projects.json").write_text(json.dumps({
        "projects": [{"id": "p1", "name": "System audit", "chat_id": 42, "lifecycle": "active"}],
    }), encoding="utf-8")
    _result(tmp_path, "finished", ts=since - 10, project_id="p1", description="audit completed")
    _result(tmp_path, "still-running", ts=since + 20, status="running", project_id="p1", description="in progress")
    _result(tmp_path, "direct-finished", ts=since + 30, direct=True, project_id="p1", description="owner chat")
    _result(tmp_path, "card-task", ts=since - 100, status="running",
            quizzes={"q1": {"state": "expired_terminal", "asked_at": _iso(since - 50),
                              "question": "Should the audit continue?"}})
    lines = wake.wake_events(tmp_path, since=since, now=T0, reason="project_digest:p1:finished")
    assert lines[0].startswith("- wake cause: project System audit settled task finished (completed), $1.25")
    assert "audit completed" in lines[0]
    assert "1 h 0 min ago" in lines[0]
    assert "still-running" not in lines[0] and "direct-finished" not in lines[0]
    assert lines[1].startswith("- owner card q1 on task card-task: Unanswered · the task finished")
    assert "Should the audit continue?" in lines[1]


def test_project_digest_task_id_wins_when_multiple_settled_rows_share_a_project(tmp_path):
    since = T0 - 3600
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "projects.json").write_text(json.dumps({
        "projects": [{"id": "p1", "name": "System audit", "chat_id": 42, "lifecycle": "active"}],
    }), encoding="utf-8")
    _result(tmp_path, "trigger", ts=since - 10, project_id="p1", description="triggered result")
    _result(tmp_path, "newer", ts=since + 10, project_id="p1", description="newer result")
    lines = wake.wake_events(tmp_path, since=since, now=T0, reason="project_digest:p1:trigger")
    assert "task trigger" in lines[0] and "triggered result" in lines[0]
    assert "latest settled" not in lines[0]
    assert "task newer" not in lines[0]


def test_trigger_stays_first_when_task_results_are_unreadable(tmp_path, monkeypatch):
    import ouroboros.task_results as task_results

    def broken(_root):
        raise OSError("broken task store")

    monkeypatch.setattr(task_results, "list_task_results", broken)
    lines = wake.wake_events(tmp_path, since=T0 - 1, now=T0, reason="task_finished:t1:completed")
    assert lines[0] == "- wake cause: task t1 finished (completed)"
    assert lines[1] == "- task_results unreadable: OSError"


def test_render_substitutes_every_placeholder_and_truncates_events_honestly(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "prompts").mkdir(parents=True)
    (repo / "prompts" / "CONSCIOUSNESS.md").write_text(
        (REPO / "prompts" / "CONSCIOUSNESS.md").read_text(encoding="utf-8"), encoding="utf-8")
    for index in range(15):
        _result(tmp_path, f"t{index:02d}", ts=T0 - 100 + index)
    text = wake.render_wake_message(
        tmp_path, repo, reason="task_finished:t14:completed", last_wake_at=T0 - 5400, since=T0 - 5400, now=T0,
        level="act", disabled_tools=["toggle_evolution", "request_restart"], spent_usd=4.0, daily_usd=20.0,
        running=1, max_tasks=2, interval=3300)
    for key in wake.PLACEHOLDERS:
        assert "{" + key + "}" not in text, key
    assert text.startswith("You are Ouroboros. No one has asked for a task")
    assert "1 h 30 min ago" in text and "autonomy: act — everything your runtime mode allows except" in text
    assert "toggle_evolution, request_restart" in text
    assert "allowance accounting (last 24 h): 4.00 / 20.00 USD" in text and "tasks running: 1/2" in text
    assert "next interval: 3300 s" in text
    assert "- wake cause: task t14 finished (completed)" in text
    assert text.count("- task t") == wake.EVENT_LINES_MAX - 1 and "(+5 more; see recent_tasks, get_task_result, and chat_history)" in text
    quiet = wake.render_wake_message(
        tmp_path, repo, reason="heartbeat", last_wake_at=0.0, since=T0 + 1, now=T0, level="full",
        disabled_tools=[], spent_usd=None, daily_usd=0, running=0, max_tasks=0, interval=900)
    assert "no wake since this process started" in quiet and "wake cause: scheduled heartbeat" in quiet
    assert "unavailable tools: none" in quiet and "allowance accounting (last 24 h): unknown / 0.00 USD" in quiet
    assert "including evolution" in quiet


def test_render_survives_a_missing_template_and_an_unreadable_task_store(tmp_path):
    (tmp_path / "task_results").mkdir()
    (tmp_path / "task_results" / "broken.json").write_text("{not json", encoding="utf-8")
    text = wake.render_wake_message(
        tmp_path, tmp_path / "no-repo", reason="heartbeat", last_wake_at=0.0, since=T0 - 1, now=T0, level="act",
        disabled_tools=[], spent_usd=0.0, daily_usd=20.0, running=0, max_tasks=2, interval=3300)
    assert text.startswith("[Wake-up · heartbeat]") and "{" not in text


def test_template_names_only_its_placeholders_and_the_wake_hints():
    template = (REPO / "prompts" / "CONSCIOUSNESS.md").read_text(encoding="utf-8")
    import re

    assert set(re.findall(r"\{([a-z_]+)\}", template)) == set(wake.PLACEHOLDERS)
    for hint in ("A pause is a legitimate decision", "Distinguish incremental cash cost",
                 "do not request task acceptance", "wake facts", "recent facts"):
        assert hint in template, hint
    assert "a heartbeat, a task that finished, a project digest" not in template
    assert "up to 10 rounds" not in template and "300 seconds" not in template


# --- the server projection --------------------------------------------------------


@pytest.fixture
def describe(monkeypatch):
    import server

    holder = {}
    monkeypatch.setattr(server, "_consciousness", SimpleNamespace(status_snapshot=lambda: dict(holder)))
    return lambda snapshot, enabled=True: (holder.clear(), holder.update(snapshot), server._describe_bg_consciousness_state(enabled))[2]


BASE = {"enabled": True, "level": "act", "next_wake_at": "2027-01-15T12:30:00+00:00", "pending_reason": "",
        "last_wake_at": "", "last_wake_task_id": "", "last_wake_outcome": "", "last_error": "",
        "spent_24h_usd": 1.0, "daily_usd": 20.0, "allowance_resets_at": "", "tasks_running": 0, "max_tasks": 2,
        "live_wake_task_id": ""}


def test_projection_names_every_honest_status(describe):
    import server

    assert describe(BASE, enabled=False)["status"] == "disabled"
    assert describe(BASE, enabled=False)["enabled"] is False  # the caller's flag, not the snapshot's
    sleeping = describe(BASE)
    assert sleeping["status"] == "sleeping" and sleeping["detail"] == f"Sleeping until {server._clock_of(BASE['next_wake_at'])}."
    assert sleeping["next_wake_at"] == BASE["next_wake_at"] and sleeping["enabled"] is True
    pending = describe({**BASE, "pending_reason": "task_finished:a:completed"})
    assert pending["detail"].endswith(" Early wake pending: task_finished:a:completed.")
    thinking = describe({**BASE, "live_wake_task_id": "wake0001"})
    assert thinking["status"] == "thinking" and "wake0001" in thinking["detail"]
    first = describe({**BASE, "last_wake_outcome": "skipped:waiting_for_first_conversation"})
    assert first["status"] == "waiting_for_first_conversation"
    exhausted = describe({**BASE, "last_wake_outcome": "skipped:allowance_exhausted", "spent_24h_usd": 21.5,
                          "allowance_resets_at": "2027-01-15T18:00:00+00:00"})
    assert exhausted["status"] == "allowance_exhausted"
    assert "$21.50 of $20.00" in exhausted["detail"] and server._clock_of("2027-01-15T18:00:00+00:00") in exhausted["detail"]
    assert "at least" not in exhausted["detail"] and "degraded" not in exhausted["detail"]
    # PLAN 5.5: an unmetered or quarantined ledger makes the number a floor, and the status says so.
    floor = describe({**BASE, "last_wake_outcome": "skipped:allowance_exhausted", "spent_24h_usd": 21.5,
                      "unknown_unmetered": 2, "integrity_degraded": True})
    assert "at least $21.50 of $20.00" in floor["detail"] and "ledger integrity degraded" in floor["detail"]
    unknown = describe({**BASE, "last_wake_outcome": "skipped:allowance_unknown", "last_error": "OSError: ledger"})
    assert unknown["status"] == "allowance_unknown" and "OSError: ledger" in unknown["detail"]
    rejected = describe({**BASE, "last_wake_outcome": "rejected:budget_exhausted"})
    assert rejected["status"] == "wake_rejected" and "budget_exhausted" in rejected["detail"]
    failed = describe({**BASE, "last_wake_outcome": "failed", "last_error": "wake-up w1 failed in its runner"})
    assert failed["status"] == "wake_failed" and "backing off" in failed["detail"]
    done = describe({**BASE, "last_wake_outcome": "done", "last_wake_at": "2027-01-15T11:35:00+00:00"})
    assert done["status"] == "sleeping"


def test_projection_without_a_constructed_clock_is_stopped_not_running(monkeypatch):
    import server

    monkeypatch.setattr(server, "_consciousness", None)
    described = server._describe_bg_consciousness_state(True)
    assert described["status"] == "stopped" and "not constructed" in described["detail"]
    assert server._describe_bg_consciousness_state(False)["status"] == "disabled"


def test_render_substitutes_placeholders_in_one_pass(tmp_path):
    """A task title that happens to contain "{daily_usd}" is a fact, not a placeholder."""
    since = T0 - 3600
    repo = tmp_path / "repo"
    (repo / "prompts").mkdir(parents=True)
    (repo / "prompts" / "CONSCIOUSNESS.md").write_text("spent {spent_usd} / {daily_usd}; events: {events}", encoding="utf-8")
    _result(tmp_path, "odd01", ts=since + 10, status="completed", cost=0.1, description="check {daily_usd} later")
    text = wake.render_wake_message(tmp_path, repo, reason="heartbeat", last_wake_at=since,
                                    since=since, now=T0, level="act", disabled_tools=[], spent_usd=1.0,
                                    daily_usd=20.0, running=0, max_tasks=2, interval=900)
    assert text.startswith("spent 1.00 / 20.00;") and "check {daily_usd} later" in text
    floor = wake.render_wake_message(tmp_path, repo, reason="heartbeat", last_wake_at=since,
                                     since=since, now=T0, level="act", disabled_tools=[], spent_usd=1.0,
                                     daily_usd=20.0, running=0, max_tasks=2, interval=900, spent_is_floor=True)
    assert floor.startswith("spent at least 1.00 / 20.00;")
