"""Cause titles precede wake work; only this attempt's accepted sentence can replace them."""
import queue
import threading
from types import SimpleNamespace

import pytest

from ouroboros.task_finalization import terminal_result_fields, wake_cause_title
from ouroboros.task_results import load_task_result, write_task_result
from supervisor import worker_chat_lane as lane
from tests import test_consciousness as clock_tests, test_place_focus as focus_tests
from tests._wake_summary_shared import seed_wake

clock = clock_tests.clock
focus = focus_tests.focus


@pytest.mark.parametrize("summary", ["Проверил состояние системы.", ""])
def test_wake_admission_names_before_activity_and_finish_replaces_only_with_summary(clock, monkeypatch, summary):
    from ouroboros import project_naming
    from supervisor import message_bus, state, workers

    frames, namers, finished = [], [], threading.Event()
    monkeypatch.setattr(workers, "DRIVE_ROOT", clock.root)
    monkeypatch.setattr(workers, "get_event_q", lambda: queue.Queue())
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 100)
    monkeypatch.setattr(project_naming, "spawn_turn_namer", lambda *a, **kw: namers.append(1))
    monkeypatch.setattr(message_bus, "get_bridge", lambda: SimpleNamespace(
        broadcast=frames.append, send_chat_action=lambda *a, **kw: frames.append({"type": "activity"})))
    workers.open_repo_writer_admission()

    def handle(task):
        task_id = task["id"]
        row = load_task_result(clock.root, task_id)
        assert row["suggested_name"] == "Own time" and not row.get("title")
        actor._event_queue.put({"type": "log_event", "data": {
            "type": "tool_call_started", "task_id": task_id, "tool": "read_file"}})
        seed_wake(clock.root, task_id=task_id, status="running", metadata=task["metadata"],
                  suggested_name=row["suggested_name"], title="")
        write_task_result(clock.root, task_id, "completed", result="Whole note.",
                          **terminal_result_fields({"wake_summary": summary} if summary else {}))
        return []

    actor = SimpleNamespace(handle_task=handle)
    monkeypatch.setattr(workers, "_get_chat_agent", lambda: actor)

    def bind(task):
        clock.clock._own_wake = {"task_id": task["id"], "initiator": "consciousness", "wake_cause": "own_clock"}

    def done(task_id, ok):
        clock.clock._wake_finished(task_id, ok)
        finished.set()

    receipt = lane.handle_wake_direct(7, "Wake", {"initiator": "consciousness", "wake_cause": "own_clock"},
                                      bind_input=bind, on_finished=done)
    assert receipt["admitted"] and finished.wait(10)
    expected = [{"type": "task_named", "task_id": receipt["task_id"], "suggested_name": "Own time"}]
    assert frames[0] == expected[0], "the title precedes the admission activity receipt"
    if summary:
        expected.append({**expected[0], "suggested_name": summary})
    assert [frame for frame in frames if frame["type"] == "task_named"] == expected
    assert namers == []
    row = load_task_result(clock.root, receipt["task_id"])
    assert row["suggested_name"] == (summary or "Own time") and not row.get("title")
    assert "summary" not in row


def test_owner_admission_leaves_title_empty_and_still_names_on_first_work(monkeypatch, tmp_path):
    from ouroboros import project_naming
    from tests.test_consciousness_wake_lane import _lane

    _lane(monkeypatch, tmp_path)
    frames, namers = [], []
    monkeypatch.setattr(lane, "_broadcast_task_named", frames.append)
    monkeypatch.setattr(project_naming, "spawn_turn_namer", lambda *a, **kw: namers.append(a))

    def handle(task):
        actor._event_queue.put({"type": "log_event", "data": {
            "type": "tool_call_started", "task_id": task["id"], "tool": "read_file"}})
        return []

    actor = SimpleNamespace(handle_task=handle)
    admitted = lane._admit_chat_task(actor, 7, "Owner work", task_metadata={"initiator": "owner"})
    assert admitted and not load_task_result(tmp_path, admitted["task"]["id"])
    assert frames == []
    assert lane._execute_chat_task(admitted)
    assert len(namers) == 1 and frames == []


@pytest.mark.parametrize("summary", ["Прочитал обсуждение и ответил.", ""])
def test_focus_runner_names_its_place_then_its_accepted_summary(focus, monkeypatch, summary):
    from ouroboros import presence_runner as runner

    root, place, admission, event = focus
    frames = []
    monkeypatch.setattr(lane, "_broadcast_task_named", frames.append)

    def handle(task):
        task_id = task["id"]
        row = load_task_result(root, task_id)
        assert row["suggested_name"] == place.label and not row.get("title")
        metadata = {**task["metadata"], "presence_outcome": "message", "presence_result_text": "Answer"}
        seed_wake(root, task_id=task_id, status="running", metadata=metadata,
                  suggested_name=row["suggested_name"], title="")
        write_task_result(root, task_id, "completed", result="Answer", terminal_origin="model_final",
                          **terminal_result_fields({"wake_summary": summary} if summary else {}))
        return [{"type": "presence_result", "task_id": task_id, "outcome": "message", "text": "Answer"}]

    result = runner.run_presence_turn(admission=admission, event=event, repo_dir=root, drive_root=root,
                                     agent_factory=lambda **_: SimpleNamespace(handle_task=handle))
    assert result.outcome == "message"
    assert [frame["suggested_name"] for frame in frames] == [place.label] + ([summary] if summary else [])
    assert {frame["type"] for frame in frames} == {"task_named"}


@pytest.mark.parametrize("metadata,expected", [
    ({"wake_cause": "own_clock"}, "Own time"),
    ({"wake_cause": "focus", "focus": {"place_id": "place-1", "label": "Research room"}}, "Research room"),
    ({"wake_cause": "focus", "focus": {"place_id": "place-1"},
      "presence": {"place": {"place_id": "place-1", "label": "Lab"}}}, "Lab"),
    ({"initiator": "owner"}, ""),
    ({"initiator": "consciousness", "consciousness_cause": "started"}, ""),
])
def test_admission_titles_use_only_typed_wake_causes_and_place_labels(metadata, expected):
    assert wake_cause_title(metadata) == expected


@pytest.mark.parametrize("metadata,fields,expected", [({"initiator": "owner"}, {}, "Old coined title"),
    ({"initiator": "consciousness", "wake_cause": "own_clock"}, {}, "Own time"),
    # A schedule-born root's sentence goes with its attempt; the schedule's authored name comes back.
    ({"schedule_occurrence": {"schedule_id": "daily-check", "token": "t1"}},
     {"title": "Daily check", "wake_summary": "Checked the disk: 30 GiB free."}, "Daily check")])
def test_new_attempt_resets_a_wakes_legacy_name_but_preserves_owner_naming(tmp_path, metadata, fields, expected):
    seed_wake(tmp_path, status="running", metadata=metadata, task_attempt=1, suggested_name="Old coined title", **fields)
    row = write_task_result(tmp_path, "214a31b6", "running", task_attempt=2)
    assert row["suggested_name"] == expected and not row.get("wake_summary")


@pytest.mark.parametrize("summary", ["Прочитал обсуждение: всё спокойно.", ""])
@pytest.mark.parametrize("direct", [False, True])
def test_focus_event_line_keeps_its_summary_on_both_terminal_carriers(focus, summary, direct):
    from ouroboros import consciousness_wake

    root, _, _, _ = focus
    task = focus_tests.task_for(focus)
    row = seed_wake(root, task_id=task["id"], metadata=task["metadata"], source="presence",
                    _is_direct_chat=direct, wake_summary=summary)
    now = consciousness_wake._parse_iso(row["terminal_time"]["occurred_at"]) + 60
    observation = consciousness_wake.observe_wake(root, boundary=None, since=now - 3600, now=now)
    line = next(line for _, _, line in observation.events if task["id"] in line)
    assert line.endswith(": " + summary) if summary else not line.endswith(": ")
    assert (line in observation.full_text()) is (bool(summary) or not direct)
    assert row["result"] not in observation.full_text()


def test_ordinary_owner_direct_turn_stays_counted_without_a_summary(tmp_path):
    from ouroboros import consciousness_wake

    row = seed_wake(tmp_path, task_id="owner-turn", metadata={"initiator": "owner"})
    now = consciousness_wake._parse_iso(row["terminal_time"]["occurred_at"]) + 60
    observation = consciousness_wake.observe_wake(tmp_path, boundary=None, since=now - 3600, now=now)
    assert any(kind == "direct_turn" and "owner-turn" in line for kind, _, line in observation.events)
    assert "owner-turn" not in observation.full_text() and row["result"] not in observation.full_text()
