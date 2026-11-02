"""#1799 (owner choice A): a due promise whose known folder is gone starts as ordinary work there.

Real producer (``schedule_followup`` stamping its origin's explicit folder), real scheduler
admission, the production agent bootstrap and loop on the admitted task (``handle_task``, the
worker's own entry; only inference is scripted), the real file and shell tools and the terminal
pipeline. Nothing is substituted for the missing folder; a broken or forbidden address, an
unreadable registry and unknown intent still wait, and money, Stop and access are the ordinary ones.
"""
from __future__ import annotations

import json
import shutil

import pytest

from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.tools.registry import ToolContext

pytestmark = pytest.mark.serial

DUE = "2000-01-01T00:00:00+00:00"


@pytest.fixture
def due(tmp_path, monkeypatch):
    from supervisor import queue

    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_kw: (True, ""))
    data = tmp_path / "data"
    data.mkdir()
    queue.init(data)
    pending: list = []
    queue.init_queue_refs(pending, {}, {"value": 0})
    return data, pending, queue


def _promise_from(data, folder, *, origin="origin-task"):
    """The origin task works in ``folder`` and promises to come back to it."""
    from ouroboros.tools.followup import _handle_schedule_followup

    metadata = {"resource_intent": {"kind": "explicit_resource", "root": str(folder)}, "root_task_id": origin}
    write_task_result(data, origin, "running", metadata=metadata, workspace_root=str(folder),
                      workspace_mode="external", delegation_role="root", chat_id=1)
    ctx = ToolContext(repo_dir=data.parent, drive_root=data, task_id=origin, task_metadata=metadata,
                      workspace_root=folder, current_chat_id=1)
    reply = _handle_schedule_followup(ctx, objective="Finish the report in its folder", run_at=DUE,
                                      relation="independent")
    assert reply.startswith("FOLLOWUP_SCHEDULED"), reply
    return reply.split("follow-up ", 1)[1].split(" ", 1)[0]


def _rows(data):
    from supervisor.queue import list_scheduled_tasks

    return {row["id"]: row for row in list_scheduled_tasks(data)["tasks"]}


def _admitted(due, tmp_path):
    """The origin promised to come back to its folder; the owner moved it away before it came due."""
    data, pending, queue = due
    folder = tmp_path / "work" / "report"
    folder.mkdir(parents=True)
    promise = _promise_from(data, folder)
    shutil.rmtree(tmp_path / "work")
    queue.check_scheduled_tasks()
    [task] = pending
    return task, folder, promise


def _run_as_worker(monkeypatch, data, task, model, *, wait=None):
    """``worker_process``'s order: the task's own forked drive and entry, then its terminal files.
    Only the model's replies are scripted; ``wait`` is the pool's half of an owner wait."""
    from functools import partial
    from ouroboros.headless import prepare_terminal_task_files
    from tests.test_presence_continuation_bootstrap import install_bootstrap_harness

    h = install_bootstrap_harness(monkeypatch, data, review_mode="off")
    monkeypatch.setenv("OUROBOROS_IN_WORKER", "1")
    h.script = model
    assert task["drive_root"] != str(data)
    agent = h.factory(repo_dir=str(h.repo), drive_root=task["drive_root"], event_queue=None,
                      budget_drive_root=task["budget_drive_root"])
    if wait is not None:
        from ouroboros.owner_wait import worker_owner_wait

        agent.owner_wait_callback = partial(worker_owner_wait, 0, *wait)
    events = agent.handle_task(task)
    assert any(event.get("type") == "task_done" for event in events)
    prepared = prepare_terminal_task_files(data, task)
    assert not prepared.get("error"), prepared
    return h, agent


def test_the_ordinary_occurrence_starts_with_its_missing_folder_and_meets_it(due, tmp_path, monkeypatch):
    from ouroboros.headless_status import ARTIFACT_TERMINAL_STATUSES
    from tests.test_presence_continuation_bootstrap import tool_call

    data, _pending, _queue = due
    task, folder, promise = _admitted(due, tmp_path)
    row = _rows(data)[promise]
    assert not row.get("hold") and row["occurrence"]["phase"] == "admitted" and row["enabled"] is False
    assert task["workspace_root"] == str(folder.resolve()) and task["workspace_mode"] == "external"
    assert task["metadata"]["resource_intent"] == {"kind": "explicit_resource", "root": str(folder)}
    assert task["metadata"]["workspace_availability"]["state"] == "missing"
    assert "workspace_availability: missing when this work was admitted" in task["text"]
    seen = []

    def model(messages):
        seen.append(messages)
        if len(seen) == 1:
            return {"content": None, "tool_calls": [
                tool_call("run_command", {"cmd": ["pwd"]}, "where"),
                tool_call("list_files", {"path": "."}, "look")]}
        if len(seen) == 2:
            return {"content": None, "tool_calls": [
                tool_call("write_file", {"path": "notes.md", "content": "Report notes, folder restored.\n"}, "write"),
                tool_call("run_command", {"cmd": ["ls"]}, "again")]}
        return {"content": "The report folder was missing; I recreated notes.md there and stopped.", "tool_calls": []}

    h, agent = _run_as_worker(monkeypatch, data, task, model)
    ctx = agent.tools._ctx
    assert str(ctx.workspace_root) == str(folder.resolve()) and ctx.is_workspace_mode()
    first_input = json.dumps(seen[0], ensure_ascii=False)
    assert "workspace_availability: missing when this work was admitted" in first_input
    results = {row.get("tool_call_id"): str(row.get("content")) for messages in seen for row in messages
               if row.get("role") == "tool"}
    # No fallback: the shell refuses the absent folder instead of running in the system repository.
    assert "SHELL_CWD_BLOCKED" in results["where"] and str(folder.resolve()) in results["where"]
    assert str(h.repo) not in results["where"].split("cwd is not a directory:", 1)[1].split(".", 1)[0]
    assert "notes.md" in results["again"] and (folder / "notes.md").read_text() == "Report notes, folder restored.\n"
    assert not (h.repo / "notes.md").exists()
    stored = load_task_result(data, task["id"])
    assert stored["status"] == "completed" and stored["workspace_root"] == str(folder.resolve())
    assert stored["artifact_status"] in ARTIFACT_TERMINAL_STATUSES


def _met_the_missing_folder(seen, last):
    """First the mind looks (the shell refuses the absent cwd, the listing finds nothing), then ``last``."""
    from tests.test_presence_continuation_bootstrap import tool_call

    def model(messages):
        seen.append(messages)
        if len(seen) == 1:
            return {"content": None, "tool_calls": [
                tool_call("run_command", {"cmd": ["pwd"]}, "where"),
                tool_call("list_files", {"path": "."}, "look")]}
        return last(len(seen))
    return model


def _finished_where_nothing_remains(data, task, folder, h, seen, answer):
    """Ended through the worker's terminal files: no folder, no write anywhere, an honest record."""
    import pathlib

    assert not folder.exists() and not folder.parent.exists() and not (h.repo / "notes.md").exists()
    results = {row.get("tool_call_id"): str(row.get("content")) for messages in seen for row in messages
               if row.get("role") == "tool"}
    assert "SHELL_CWD_BLOCKED" in results["where"] and str(folder.resolve()) in results["where"]
    stored = load_task_result(data, task["id"])
    assert stored["status"] == "completed" and stored["result"] == answer
    assert stored["workspace_root"] == str(folder.resolve())
    execution = stored["outcome_axes"]["execution"]
    assert [denial["status"] for denial in execution["policy_denials"]] == ["cwd_blocked"]
    assert execution["mutation_attribution"]["known_path_count"] == 0
    # Neither a failed Git capture nor a retained workspace: nothing remains there to capture.
    assert stored["artifact_status"] == "missing" and not stored.get("artifact_error")
    assert stored["outcome_axes"]["artifacts"]["status"] == "missing"
    assert not [row for row in stored["artifacts"] if row["kind"] == "workspace_patch"]
    [row] = [row for row in stored["artifacts"] if row["kind"] == "workspace_patch_manifest"]
    manifest = json.loads(pathlib.Path(row["path"]).read_text(encoding="utf-8"))
    assert manifest["capture_kind"] == "directory_direct" and manifest["before"] == "absent"
    assert manifest["status"] == "missing" and manifest["registered_outputs"] == []
    assert "still absent, so nothing remains there to capture" in manifest["note"]
    # Its consumers: the terminal files are ready (nothing stays owed), and once the post-task stage this
    # harness stubs out has closed, the status fold the owner reads (``log_events.js`` paints the same)
    # says error, never done.
    from ouroboros.headless import terminal_task_files_ready
    from ouroboros.project_dialogue import outcome_phase

    assert terminal_task_files_ready(data, task, stored) is True
    closed = {**stored, "root_phase_checkpoint": {**(stored.get("root_phase_checkpoint") or {}),
                                                  "post_task_synthesis": "completed"}}
    assert outcome_phase(closed, {"type": "task_done", "task_id": task["id"]}) == "error"


def test_a_mind_that_cannot_go_on_without_its_folder_finishes_without_recreating_it(due, tmp_path, monkeypatch):
    data, _pending, _queue = due
    task, folder, _promise = _admitted(due, tmp_path)
    seen = []
    answer = "The report folder is gone, so I stopped without recreating it or working anywhere else."
    h, _agent = _run_as_worker(monkeypatch, data, task, _met_the_missing_folder(
        seen, lambda _round: {"content": answer, "tool_calls": []}))
    _finished_where_nothing_remains(data, task, folder, h, seen, answer)


def test_a_mind_that_waits_for_the_owner_finishes_without_recreating_it(due, tmp_path, monkeypatch):
    """It asks where the folder went and waits; the pool parks it, the owner's answer reaches the
    task's own drive, the pool grants the resume, and the run ends through the same terminal files."""
    import pathlib
    import queue as queues
    import threading

    from ouroboros.owner_mailbox import KIND_QUIZ_ANSWER, write_owner_message
    from tests.test_presence_continuation_bootstrap import tool_call

    data, _pending, _queue = due
    task, folder, _promise = _admitted(due, tmp_path)
    commands, events, waits, failures = queues.Queue(), queues.Queue(), [], []

    def pool():  # the supervisor's half of the pooled owner wait
        try:
            while (event := events.get(timeout=120)).get("phase") != "park":
                pass
            identity = {key: event[key] for key in ("type", "task_id", "task_attempt", "wait_id")}
            commands.put({**identity, "phase": "parked"})
            quiz = event["checkpoint"]["quiz_id"]
            assert write_owner_message(pathlib.Path(task["drive_root"]), "It is gone for good; do not recreate it.",
                                       task["id"], msg_id=f"quiz_answer:{task['id']}:{quiz}", kind=KIND_QUIZ_ANSWER)
            while (event := events.get(timeout=120)).get("phase") != "resume":
                pass
            waits.append(event["resume_reason"])
            commands.put({**identity, "phase": "resume_granted"})
        except Exception as exc:  # surfaced below, never swallowed
            failures.append(exc)

    def last(round_number):
        if round_number == 2:
            return {"content": None, "tool_calls": [tool_call("escalate", {
                "question": "The report folder is gone. Did it move, or should I stop?",
                "options": ["It moved", "Stop"], "wait_for_answer": True}, "ask")]}
        return {"content": answer, "tool_calls": []}

    answer = "You said the report folder is gone for good, so I stopped without recreating it."
    seen = []
    supervisor = threading.Thread(target=pool, daemon=True)
    supervisor.start()
    h, _agent = _run_as_worker(monkeypatch, data, task, _met_the_missing_folder(seen, last),
                               wait=(commands, events))
    supervisor.join(timeout=30)
    assert not failures and waits == ["answer"], (failures, waits)
    assert len(seen) == 3 and "It is gone for good" in json.dumps(seen[2], ensure_ascii=False)
    _finished_where_nothing_remains(data, task, folder, h, seen, answer)


@pytest.mark.parametrize("case", ["inside_data_drive", "file_in_its_place", "unknown_intent"])
def test_addresses_that_are_not_merely_absent_still_wait(due, tmp_path, case):
    from supervisor.schedule_lifecycle import upsert_scheduled_task

    data, pending, queue = due
    if case == "inside_data_drive":
        intent = {"kind": "explicit_resource", "root": str(data / "state" / "gone")}
    elif case == "file_in_its_place":
        target = tmp_path / "now-a-file"
        target.write_text("not a folder")
        intent = {"kind": "explicit_resource", "root": str(target)}
    else:
        intent = None
    upsert_scheduled_task({"id": "s1", "name": "Promise", "source": "task_followup",
                           "trigger": {"type": "once", "run_at": DUE},
                           "task": {"type": "task", "text": "continue", "chat_id": 1,
                                    "metadata": {**({"resource_intent": intent} if intent else {}),
                                                 "origin_task_id": "nobody"}}},
                          drive_root=data, host_followup={"followup_origin": {"task_id": "nobody"},
                                                          "followup_relation": {"kind": "independent"}})
    queue.check_scheduled_tasks()
    assert pending == []
    assert _rows(data)["s1"]["hold"]["reason"] == ("resource_intent_unknown" if case == "unknown_intent"
                                                   else "workspace_unusable")


def test_a_room_whose_folder_is_gone_starts_its_promise_there(due, tmp_path):
    from ouroboros.projects_registry import create_project
    from supervisor.schedule_lifecycle import upsert_scheduled_task

    data, pending, queue = due
    folder = tmp_path / "room-folder"
    folder.mkdir()
    create_project(data, "proj", name="Room", working_dir=str(folder))
    upsert_scheduled_task({"id": "s1", "name": "Room promise", "source": "task_followup",
                           "trigger": {"type": "once", "run_at": DUE},
                           "task": {"type": "task", "text": "continue", "chat_id": 1, "project_id": "proj",
                                    "metadata": {"resource_intent": {"kind": "room_default", "project_id": "proj"},
                                                 "origin_task_id": "room-turn"}}},
                          drive_root=data, host_followup={"followup_origin": {"task_id": "room-turn"},
                                                          "followup_relation": {"kind": "independent"}})
    folder.rmdir()
    queue.check_scheduled_tasks()
    [task] = pending
    assert task["workspace_root"] == str(folder.resolve()) and task["project_id"] == "proj"
    assert task["metadata"]["workspace_availability"]["state"] == "missing"


_LONG_STOP = r'''
import datetime, json, pathlib, sys, time
from supervisor import queue, state, workers
from ouroboros import config
from ouroboros.task_results import load_task_result
from ouroboros.utils import atomic_write_json, read_json_dict
base, stage, kind, control, task_id = pathlib.Path(sys.argv[1]), *sys.argv[2:]
root, repo, place = base / "data", base / "repo", base / "place"
folder = place / "report"
root.mkdir(exist_ok=True); repo.mkdir(exist_ok=True)
state.init(root)
workers.init(place if stage == "restart" and control == "protected_now" else repo, root, 1)
config.get_bg_wakeup_min_sec = lambda: 0
now = datetime.datetime.now(datetime.timezone.utc)
if stage == "admit":
    from ouroboros.projects_registry import create_project
    from supervisor.schedule_lifecycle import upsert_scheduled_task
    state.save_state({"owner_chat_id": 1})
    folder.mkdir(parents=True)
    intent, extra = {"kind": "explicit_resource", "root": str(folder)}, {}
    if kind == "room":
        create_project(root, "proj", name="Room", working_dir=str(folder))
        intent, extra = {"kind": "room_default", "project_id": "proj"}, {"project_id": "proj"}
    if control == "deadline":
        extra["deadline_at"] = (now + datetime.timedelta(seconds=2)).isoformat()
    upsert_scheduled_task({"id": "s1", "name": "Promise", "source": "task_followup",
                           "trigger": {"type": "once", "run_at": "2000-01-01T00:00:00+00:00"},
                           "task": {"type": "task", "text": "continue", "chat_id": 1, **extra,
                                    "metadata": {"resource_intent": intent, "origin_task_id": "origin"}}},
                          drive_root=root, host_followup={"followup_origin": {"task_id": "origin"},
                                                          "followup_relation": {"kind": "independent"}})
    if control != "admitted_present":
        folder.rmdir()  # gone before the promise came due: admitted with its known address
    queue.check_scheduled_tasks()
    [task] = workers.PENDING
    if control == "admitted_present":
        folder.rmdir()  # gone only after an admission that found it
    time.sleep(3 if control == "deadline" else 0)
    print(json.dumps({"task_id": task["id"], "workspace_root": task["workspace_root"],
                      "availability": task["metadata"].get("workspace_availability")}))
else:
    from tests.test_worker_crash_retry import _make_worker
    from tests.worker_pool_support import affirm_ready
    alive = root / "state" / "queue.alive"  # the application stayed stopped for two hours
    atomic_write_json(alive, {**(read_json_dict(alive) or {}), "ts": (now - datetime.timedelta(hours=2)).isoformat()})
    if control == "now_file":
        folder.write_text("something else sits at the address now")
    elif control == "redirected":
        place.rename(base / "elsewhere")
        place.symlink_to(base / "elsewhere", target_is_directory=True)
    elif control == "room_moved":
        from ouroboros.projects_registry import update_project
        (base / "other").mkdir()
        update_project(root, "proj", working_dir=str(base / "other"))
    queue.restore_pending_from_snapshot(terminalized=[])
    restored = next((dict(t) for t in workers.PENDING if t["id"] == task_id), {})
    worker = affirm_ready(_make_worker(alive=True, busy_task_id=None, exitcode=None))
    workers.WORKERS[0] = worker
    notices = []
    workers.send_with_budget = lambda _chat, text, **_kw: notices.append(text)  # the owner's chat
    state.budget_remaining = lambda *_a, **_k: 10.0
    resume = queue.resume_budget_paused_task(task_id) if restored and control != "no_resume" else {}
    if control == "budget":
        state.budget_remaining = lambda *_a, **_k: 0.0
    workers.assign_tasks()
    workers.assign_tasks()
    row = next((t for t in workers.PENDING if t["id"] == task_id), None)
    sent = worker.in_q.put.call_args.args[0] if worker.in_q.put.call_count else {}
    print(json.dumps({
        "restored_hold": (restored.get("_project_admission_restore_hold") or {}).get("detail", ""),
        "resume": resume.get("selection", ""), "puts": worker.in_q.put.call_count,
        "sent": {"workspace_root": sent.get("workspace_root"),
                 "availability": (sent.get("metadata") or {}).get("workspace_availability")},
        "held": None if row is None else {
            "project": (row.get("_project_admission_restore_hold") or {}).get("detail", ""),
            "saved_work": (row.get("_budget_pause_hold") or {}).get("reason", ""),
            "budget": (row.get("_budget_pause") or {}).get("status", ""),
            "terminal": (row.get("_terminalization_retry") or {}).get("trigger", "")},
        "stored": {key: (load_task_result(root, task_id) or {}).get(key) for key in ("status", "reason_code")},
        "notices": notices, "folder_exists": folder.exists() or folder.is_symlink()}, default=str))
'''


@pytest.mark.parametrize("kind,control", [
    ("explicit", "missing"), ("room", "missing"), ("explicit", "no_resume"), ("explicit", "budget"),
    ("room", "deadline"), ("explicit", "now_file"), ("room", "redirected"), ("explicit", "protected_now"),
    ("room", "admitted_present"), ("room", "room_moved"),
])
def test_a_long_stop_keeps_the_admission_of_a_missing_folder(tmp_path, kind, control):
    """Real queue save, interpreter exit and a restore old enough (the production 900 s) to
    revalidate the Project admission, then the owner's Resume and assignment. The admission that
    kept the missing folder's address survives that revalidation, also after its room moved to
    another folder (admitted work keeps its frozen basis, never the new folder). The stop's Pause,
    money, the deadline, something now at the address, a redirected or forbidden address and a
    folder lost only after an admission that found it still bind."""
    import os
    import pathlib
    import subprocess
    import sys

    env = {**os.environ, "OUROBOROS_DATA_DIR": str(tmp_path / "data"), "OUROBOROS_REPO_DIR": str(tmp_path / "repo")}
    results, task_id = [], ""
    for stage in ("admit", "restart"):
        completed = subprocess.run([sys.executable, "-c", _LONG_STOP, str(tmp_path), stage, kind, control, task_id],
                                   cwd=pathlib.Path(__file__).resolve().parents[1], env=env,
                                   capture_output=True, text=True, encoding="utf-8", timeout=90)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        results.append(json.loads(completed.stdout.strip().splitlines()[-1]))
        task_id = results[0]["task_id"]
    admitted, after = results
    assert after["restored_hold"], after  # the long stop sends the row through Project revalidation
    if control == "admitted_present":
        assert admitted["availability"] is None
    else:
        assert admitted["availability"]["state"] == "missing"
    if control in {"missing", "room_moved"}:
        assert after["resume"] == "budget_hold_released" and after["puts"] == 1 and after["held"] is None, after
        assert after["sent"]["workspace_root"] == admitted["workspace_root"]  # never the room's new folder
        assert after["sent"]["availability"] == admitted["availability"]
        assert after["folder_exists"] is False  # nothing recreated or substituted
        return
    assert after["puts"] == 0, after
    if control == "no_resume":
        assert after["held"] == {"project": "", "saved_work": "saved_work_hold", "budget": "", "terminal": ""}
    elif control == "budget":
        assert after["held"]["project"] == "" and after["held"]["budget"] == "paused_before_dispatch", after
        assert [text for text in after["notices"] if "Model budget reached" in text], after
    elif control == "deadline":
        assert after["held"] is None or after["held"]["terminal"] == "deadline", after
        assert after["held"] is not None or after["stored"]["status"] in {"failed", "cancelled"}, after
    else:
        assert after["held"]["project"] == "The original prepared folder is unavailable or resolves elsewhere.", after


def test_the_assisted_update_resolver_stamps_the_repository_it_works_in(monkeypatch, tmp_path):
    """Its follow-up no longer waits as intent-unknown: the resolver records the resource it uses."""
    from supervisor import update_merge, workers
    from ouroboros.tools.followup import followup_resource_intent

    captured = []
    monkeypatch.setattr(workers, "ensure_worker_pool_started", lambda **_kw: True)
    monkeypatch.setattr(workers, "PENDING", [])
    monkeypatch.setattr(workers, "RUNNING", {})
    monkeypatch.setattr("supervisor.queue.enqueue_with_admission_receipt",
                        lambda task, **_kw: captured.append(task) or task)
    monkeypatch.setattr(update_merge, "write_update_tx", lambda _tx: None)
    monkeypatch.setattr("ouroboros.task_results.load_task_result", lambda *_a, **_kw: {})
    tx = {"task_id": "update_assisted_merge_t1", "target_sha": "a" * 40, "conflict_paths": ["x.py"],
          "owner_chat_id": 1, "resolver_submitted_id": ""}
    assert update_merge.enqueue_assisted_resolution_task(tx) == "update_assisted_merge_t1"
    [task] = captured
    assert task["metadata"]["resource_intent"] == {"kind": "system_repo"}
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=task["id"], task_metadata=task["metadata"])
    assert followup_resource_intent(ctx, task["metadata"], "") == {"kind": "system_repo"}
