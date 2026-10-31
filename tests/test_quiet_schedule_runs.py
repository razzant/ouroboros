"""Quiet scheduled results stay internal through Project/Main completion delivery; a quiet run
may still say what it did in one sentence, which its card, Activity and the next run read."""

import json
from types import SimpleNamespace

import pytest

from ouroboros.projects_registry import bind_task_to_project, create_project
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.terminal_projection import SETTLEMENT_DEFERRED, SETTLEMENT_NONE, SETTLEMENT_SETTLED, settle_terminal_projection
from supervisor.event_observation import ObservedEventQueue
from tests._dialogue import chat_rows
from tests.test_finish_silent_wake import wake  # noqa: F401 -- the real loop and terminal pipeline

SENTENCE = ("Checked the disk: 30 GiB free, 30 less than ninety minutes ago; the cause is still unknown, "
            "nothing else changed overnight, the backups and the gateway are healthy, and I did not wake you for it.")
assert len(SENTENCE) > 160  # Activity keeps the sentence whole; only an answer's end is cut at 160


@pytest.fixture
def scheduled_project(tmp_path):
    project = create_project(tmp_path, "checks", name="Checks")
    task = {
        "id": "scheduled-check", "type": "task", "title": "Check for changes",
        "text": "Check for changes", "chat_id": project["chat_id"], "project_id": project["id"],
        "root_task_id": "scheduled-check", "delegation_role": "root",
        "metadata": {"schedule_occurrence": {"schedule_id": "daily-check", "token": "occurrence-1"}},
    }
    bind_task_to_project(tmp_path, task["id"], project["id"], project["chat_id"],
                         origin={"absent": "system"})
    return task


@pytest.mark.parametrize("silent,notice", [(True, ""), (False, ""), (True, "A tool failed.")],
                         ids=["quiet", "ordinary", "quiet-with-notice"])
def test_project_completion_mirrors_only_published_results(tmp_path, scheduled_project, silent, notice):
    task = scheduled_project
    write_task_result(tmp_path, task["id"], "completed", **{k: v for k, v in task.items() if k != "id"},
                      result="The complete result.", terminal_origin="model_final", wake_silent=silent,
                      terminal_host_notice=notice, outcome_axes={"execution": {"status": "ok"}})
    bus = ObservedEventQueue()
    quiet = silent and not notice
    assert settle_terminal_projection(tmp_path, task["id"], task=task, event_queue=bus) == (
        SETTLEMENT_SETTLED if quiet else SETTLEMENT_DEFERRED)
    rows = chat_rows(tmp_path)
    summary = next(row for row in rows if row.get("type") == "task_summary")
    assert summary["chat_id"] == task["chat_id"]
    assert summary["wake_silent"] is silent and summary["quiet_effective"] is bool(quiet)
    if quiet:
        assert summary["text"] == "Done · no message"
        assert bus.empty()
        assert not any(row.get("type") == "project_completion_summary" for row in rows)
    else:
        mirror = bus.get_nowait()
        assert mirror["system_type"] == "project_completion_summary"
        assert mirror["chat_id"] == 1
        answer = mirror["progress_meta"].get("completion_answer")
        assert answer == (None if silent else "The complete result.")
        assert summary["text"] != "Done · no message"
    stored = load_task_result(tmp_path, task["id"], strict=True)
    assert stored["result"] == "The complete result."
    if quiet:
        assert stored["canonical_terminal_projection"]["main_disposition"] == "ineligible"
        assert stored["canonical_terminal_projection_ready"] is None
        assert settle_terminal_projection(tmp_path, task["id"], task=task, event_queue=bus) == SETTLEMENT_NONE
        assert len(chat_rows(tmp_path)) == len(rows) and bus.empty()


@pytest.mark.parametrize("fault", [False, True], ids=["normal", "lifecycle-fault"])
@pytest.mark.parametrize("silent,notice", [(True, ""), (False, ""), (True, "A tool failed.")],
                         ids=["quiet", "ordinary", "quiet-with-notice"])
def test_task_done_carries_canonical_quiet_facts(tmp_path, scheduled_project, monkeypatch, fault, silent, notice):
    from ouroboros.headless import prepare_terminal_task_files
    from supervisor import events
    from tests.test_task_done_durable_vs_transport import _mk_ctx

    task = scheduled_project
    write_task_result(tmp_path, task["id"], "completed", **{k: v for k, v in task.items() if k != "id"},
                      result="Internal check result.", terminal_origin="model_final", wake_silent=silent,
                      terminal_host_notice=notice, outcome_axes={"execution": {"status": "ok"}})
    assert not prepare_terminal_task_files(tmp_path, task)["error"]
    sent, pushed = [], []
    ctx = _mk_ctx(tmp_path, sent, pushed)
    ctx.RUNNING[task["id"]] = {"task": task}
    bus = ObservedEventQueue()
    monkeypatch.setattr("supervisor.workers.get_event_q", lambda **_kw: bus)
    events._handle_task_done({
        "task_id": task["id"], "status": "running" if fault else "completed",
        "chat_id": task["chat_id"], "_files_prepared_attempt": 1,
        "wake_silent": not silent, "quiet_effective": not silent,
    }, ctx)
    done = next(row for row in pushed if row.get("type") == "task_done")
    assert done["wake_silent"] is silent
    assert done["quiet_effective"] is bool(silent and not notice)
    records = [json.loads(line) for line in (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    recorded = next(row for row in records if row.get("type") == "task_done")
    assert recorded["wake_silent"] is silent and recorded["quiet_effective"] == done["quiet_effective"]
    assert task["id"] not in ctx.RUNNING


@pytest.fixture
def sched(wake, monkeypatch):  # noqa: F811
    """The real scheduler at the loop's own data root, so one run flows from admission to the next claim."""
    import ouroboros.agent_task_pipeline as pipeline
    from supervisor import queue, state, workers

    root = wake.root
    state.init(root)
    state.save_state({"owner_chat_id": 1})
    queue.init(root)
    pending: list = []
    queue.init_queue_refs(pending, {}, {"value": 0})
    (root / "repo").mkdir(exist_ok=True)
    monkeypatch.setattr(workers, "REPO_DIR", root / "repo")
    monkeypatch.setattr("ouroboros.config.get_bg_wakeup_min_sec", lambda: 0)
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **k: None)
    return SimpleNamespace(queue=queue, pending=pending, root=root)


@pytest.mark.parametrize("summary", [SENTENCE, ""], ids=["sentence", "no-sentence"])
def test_a_quiet_main_born_schedule_run_says_what_it_did(wake, sched, monkeypatch, summary):  # noqa: F811
    """Admission names the run after its schedule; the run's own sentence rides beside that name on the
    terminal row (never as its title), becomes the schedule's last-run line whole, and reaches the next run."""
    from ouroboros.headless import prepare_terminal_task_files
    from supervisor import queue_schedules as store, schedule_occurrence as occurrence
    from supervisor.events import _handle_task_done
    from tests import test_finish_silent_wake as door
    from tests.test_schedule_occurrence import _row, _rows
    from tests.test_task_done_durable_vs_transport import _mk_ctx

    _row(sched, "daily-check", intent={"kind": "system_repo"}, cron=True, chat_id=1, name="Daily check")
    sched.queue.check_scheduled_tasks()
    sched.queue.check_scheduled_tasks()
    [task] = sched.pending
    admitted = load_task_result(sched.root, task["id"], strict=True)
    assert admitted["status"] == "scheduled" and admitted["title"] == admitted["suggested_name"] == "Daily check"
    assert occurrence.record_dispatch_possible(task)
    task["_skip_post_task_synthesis"] = True
    monkeypatch.setattr(door, "TASK_ID", task["id"])  # the loop and the pipeline act as this admitted run
    bus = ObservedEventQueue()
    monkeypatch.setattr("supervisor.workers.get_event_q", lambda **_kw: bus)
    wake.ctx.is_direct_chat = False
    text, usage, trace, _ = wake.run([door.finish("Internal check result.", silent=True,
                                                  **({"summary": summary} if summary else {}))], metadata=task["metadata"], bus=bus)
    assert text == "Internal check result." and usage.get("wake_summary", "") == summary
    pending, stored = door.emit(wake, text, usage, trace, bus=bus, metadata=task["metadata"], task=task)
    assert stored["status"] == "completed" and stored["wake_silent"] is True
    assert stored["title"] == stored["suggested_name"] == "Daily check", "the schedule's name stays the title"
    assert stored.get("wake_summary", "") == summary and stored["result"] == "Internal check result."
    assert door.sends(pending) == [] and door.sends(door.drained(bus)) == [] and door.owed_rows(sched.root) == []
    # The supervisor's terminal handling writes the room's one row and the live frame.
    assert not prepare_terminal_task_files(sched.root, task)["error"]
    sent, pushed = [], []
    ctx = _mk_ctx(sched.root, sent, pushed)
    ctx.RUNNING[task["id"]] = {"task": task}
    _handle_task_done({**next(e for e in pending if e.get("type") == "task_done"), "_files_prepared_attempt": 1}, ctx)
    rows = chat_rows(sched.root)
    row = next(row for row in rows if row.get("summary_id") == f"task-terminal:{task['id']}")
    assert row["chat_id"] == 1 and row["text"] == "Done · no message" and row["quiet_effective"] is True
    assert row["suggested_name"] == row["schedule_name"] == "Daily check"
    assert row.get("wake_summary", "") == summary
    frame = next(f for f in pushed if f.get("type") == "task_done")
    assert frame.get("wake_summary", "") == summary and frame["schedule_name"] == "Daily check"  # the live card's origin
    assert not sent and not any(r.get("type") == "project_completion_summary" for r in rows)
    assert "Internal check result." not in json.dumps(rows, ensure_ascii=False)
    # The scheduler reads the settled run once: Activity's line and the next run's memory of it.
    sched.pending.clear()
    data = store.load_schedule_store(sched.root)
    data["tasks"][0]["next_run_at"] = "2000-01-01T00:00:00+00:00"
    store._write_scheduled_tasks(data, sched.root)
    sched.queue.check_scheduled_tasks()
    outcome = _rows(sched)["daily-check"]["last_outcome"]
    assert outcome["task_id"] == task["id"] and outcome["state"] == "completed"
    assert outcome["quiet"] is True and outcome["notice"] is False and outcome["line"] == summary
    [following] = sched.pending
    assert following["metadata"]["schedule_occurrence"]["previous"] == outcome
    assert following["title"] == following["suggested_name"] == "Daily check"
