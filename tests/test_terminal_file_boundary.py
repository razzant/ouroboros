"""Completed work saves files before its terminal frame, without replaying the model."""

from copy import deepcopy
import queue
import threading
from types import SimpleNamespace

import pytest

from ouroboros import agent_task_pipeline as pipeline, headless, owner_mailbox
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.utils import append_jsonl
from supervisor import worker_process
from supervisor.terminal_delivery import cleanup_settled_owner_mailbox


@pytest.fixture
def worker(tmp_path, monkeypatch):
    """Exercise the real worker loop without creating OS children or provider calls."""
    import ouroboros.agent as agent_module
    import ouroboros.config as config
    import ouroboros.extension_loader as extensions
    import ouroboros.platform_layer as platform
    import ouroboros.process_custody as custody
    import ouroboros.utils as utils
    from supervisor import git_ops, queue as task_queue, state

    root, repo = tmp_path / "data", tmp_path / "repo"
    (root / "logs").mkdir(parents=True)
    repo.mkdir()
    task_queue.init(root)
    state.init(root)
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", root)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setattr(platform, "create_new_session", lambda: None)
    monkeypatch.setattr(custody, "start_parent_lifeline", lambda **_kw: None)
    monkeypatch.setattr(config, "initialize_runtime_mode_baseline", lambda: None)
    monkeypatch.setattr(config, "get_skills_repo_path", lambda: "")
    monkeypatch.setattr(extensions, "reload_all", lambda *_a, **_kw: None)
    monkeypatch.setattr(worker_process, "_adopt_published_extensions", lambda *_a: None)
    monkeypatch.setattr(worker_process, "_prepare_worker_task_runtime", lambda: None)
    monkeypatch.setattr(utils, "set_log_sink", lambda _sink: None)
    monkeypatch.setattr(utils, "get_git_info", lambda _root: ("fixture", "test-sha"))
    for method in ("chat", "chat_async"):
        monkeypatch.setattr("ouroboros.llm.LLMClient." + method,
                            lambda *_a, **_kw: pytest.fail("provider call in boundary fixture"))
    crashes, calls, reads = [], [], []
    task = {"id": "file-root", "type": "task", "_attempt": 3,
            "drive_root": str(root), "chat_id": 1}
    events = [{"type": "send_message", "task_id": task["id"], "text": "The answer"},
              {"type": "task_metrics_event", "task_id": task["id"], "tool_calls": 2},
              {"type": "task_done", "task_id": task["id"], "status": ""}]

    class Input:
        def get(self):
            reads.append(True)
            return task if len(reads) == 1 else {"type": "shutdown"}

    class Agent:
        def handle_task(self, current):
            calls.append(current["id"])
            return events

    monkeypatch.setattr(agent_module, "make_agent", lambda **_kw: Agent())
    monkeypatch.setattr(worker_process, "_log_worker_crash", lambda *a: crashes.append(a))
    output = queue.Queue()
    return SimpleNamespace(root=root, task=task, events=events, output=output,
                           calls=calls, reads=reads, crashes=crashes,
                           run=lambda: worker_process.worker_main(0, Input(), output, str(repo), str(root)))


def test_worker_saves_once_after_prior_frames_and_before_done(worker, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    preparations = []
    original = deepcopy(worker.events)

    def prepare(root, task):
        preparations.append((root, task["id"], task["_attempt"]))
        entered.set()
        assert release.wait(5)
        write_task_result(root, task["id"], "completed", result="The answer", artifact_status="ready")
        return {"task_id": task["id"], "result": {}, "error": ""}

    monkeypatch.setattr(headless, "prepare_terminal_task_files", prepare, raising=False)
    thread = threading.Thread(target=worker.run)
    thread.start()
    try:
        assert entered.wait(5)
        assert [worker.output.get_nowait()["type"] for _ in range(2)] == [
            "send_message", "task_metrics_event"]
        assert worker.output.empty(), "task_done must not release the busy slot before saving"
        assert len(worker.reads) == 1, "this worker cannot take another task while saving"
        from supervisor.events import dispatch_event

        pushed = []
        busy = SimpleNamespace(busy_task_id="file-root")
        ctx = SimpleNamespace(
            DRIVE_ROOT=worker.root, WORKERS={0: busy},
            RUNNING={"other": {"task": {"id": "other", "chat_id": 1}}},
            bridge=SimpleNamespace(push_log=pushed.append),
        )
        dispatch_event({"type": "task_heartbeat", "task_id": "other", "phase": "working"}, ctx)
        assert ctx.RUNNING["other"]["last_heartbeat_at"] > 0
        assert pushed[-1]["task_id"] == "other"
        assert busy.busy_task_id == "file-root"
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and not worker.crashes
    done = worker.output.get_nowait()
    assert done["type"] == "task_done" and done["_files_prepared_attempt"] == 3
    assert worker.output.empty() and worker.calls == ["file-root"]
    assert preparations == [(worker.root, "file-root", 3)]
    assert worker.events == original, "transport stamps must not mutate the producer's frames"
    assert load_task_result(worker.root, "file-root")["result"] == "The answer"


@pytest.mark.parametrize("failure", ["returned", "raised"])
def test_file_failure_does_not_crash_worker_or_replay_task(worker, monkeypatch, failure):
    def prepare(root, task):
        if failure == "raised":
            raise OSError("canonical store refused the write")
        return {"task_id": task["id"], "result": None, "error": "canonical write failed"}

    monkeypatch.setattr(headless, "prepare_terminal_task_files", prepare, raising=False)
    worker.run()
    assert worker.calls == ["file-root"] and len(worker.reads) == 2 and not worker.crashes
    frames = [worker.output.get_nowait() for _ in range(3)]
    assert frames[-1]["_files_prepared_attempt"] == 3
    assert load_task_result(worker.root, "file-root") is None, "a stamp is not a saved result"


@pytest.mark.parametrize("kind", ["direct", "foreign_done", "owner_wait"])
def test_only_own_pooled_terminal_prepares_files(worker, monkeypatch, kind):
    if kind == "direct":
        worker.task["_is_direct_chat"] = True
    elif kind == "foreign_done":
        worker.events[-1]["task_id"] = "another-task"
    else:
        worker.events.pop()
    monkeypatch.setattr(headless, "prepare_terminal_task_files",
                        lambda *_a: pytest.fail("non-pooled/foreign/wait frame prepared"), raising=False)
    worker.run()
    assert not worker.crashes
    frames = [worker.output.get_nowait() for _ in worker.events]
    assert not any("_files_prepared_attempt" in frame for frame in frames)


@pytest.mark.parametrize("pooled", [True, False])
def test_post_task_cleanup_waits_for_pooled_files_only(tmp_path, monkeypatch, pooled):
    root = tmp_path / "data"
    root.mkdir()
    task = {"id": "post-files", "type": "task", "text": "The answer", "drive_root": str(root)}
    write_task_result(root, task["id"], "completed", result="The answer")
    owner_mailbox.write_owner_message(root, "accepted follow-up", task["id"], msg_id="owner-input")
    mailbox = owner_mailbox._mailbox_path(root, task["id"])
    monkeypatch.setattr(pipeline, "in_worker_process", lambda: pooled)
    monkeypatch.setattr(pipeline, "_pre_synthesis_usage_snapshot", lambda *_a: {})
    monkeypatch.setattr("ouroboros.llm.LLMClient", lambda: object())
    monkeypatch.setattr("ouroboros.memory.Memory", lambda **_kw: object())
    for name in ("_run_chat_consolidation", "_run_scratchpad_consolidation", "_record_task_facts",
                 "_run_reflection", "_update_improvement_backlog", "_apply_reflection_memory_actions"):
        monkeypatch.setattr(pipeline, name, lambda *_a, **_kw: None)
    monkeypatch.setattr("ouroboros.post_task_evolution.maybe_promote", lambda *_a: None)
    env = SimpleNamespace(drive_root=root, repo_dir=tmp_path, drive_path=lambda rel: root / rel)
    pipeline._run_post_task_processing_async(env, task, {}, {}, {}, root / "logs", blocking=True)
    assert mailbox.exists() is pooled
    if pooled:
        cleanup_settled_owner_mailbox(root, task["id"], task)
        assert not mailbox.exists()


@pytest.mark.parametrize("post,pending,keep", [
    ("pending_once", [], True), ("running", [], True),
    ("completed", [{"kind": "task_attachment", "path": "accepted-input"}], True),
    ("completed", [], False), ("degraded", [{"kind": "blob_ref"}], False),
])
def test_startup_mailbox_sweep_keeps_the_same_outstanding_obligations(tmp_path, post, pending, keep):
    task_id = "sweep-files"
    owner_mailbox.write_owner_message(tmp_path, "accepted follow-up", task_id, msg_id="owner-input")
    mailbox = owner_mailbox._mailbox_path(tmp_path, task_id)
    write_task_result(tmp_path, task_id, "completed", root_phase_checkpoint={"post_task_synthesis": post},
                      child_ref_promotion={"pending_refs": pending})
    original = mailbox.read_bytes()
    report = owner_mailbox.sweep_settled_owner_mailboxes(tmp_path)
    assert mailbox.exists() is keep
    assert report == {"removed": [] if keep else [task_id], "kept": int(keep)}
    if keep:
        assert mailbox.read_bytes() == original


def test_pooled_post_work_preserves_real_followup_until_copyback(tmp_path, monkeypatch):
    """The actual accepted-input source, not a fabricated pending-ref marker."""
    from ouroboros import artifacts

    parent, task_id = tmp_path / "canonical", "followup-copy"
    child = headless.prepare_task_drive(parent, task_id, "empty")
    task = {"id": task_id, "root_task_id": task_id, "type": "task",
            "drive_root": str(child), "budget_drive_root": str(parent), "text": "The answer"}
    source = tmp_path / "owner-input.txt"
    source.write_text("Input accepted after this task started", encoding="utf-8")
    attachments = artifacts.stage_task_attachments(child, task_id, [{"path": str(source)}])
    assert attachments and attachments[0].get("relpath"), attachments
    assert owner_mailbox.write_owner_message(child, "Use my file", task_id,
                                            msg_id="followup", attachment_manifest=attachments)
    assert owner_mailbox.acknowledge_task_messages(child, task_id, ["followup"], wake_id="fixture")
    write_task_result(child, task_id, "completed", result="The answer", artifact_status="ready")
    write_task_result(parent, task_id, "completed", result="Early canonical answer",
                      root_phase_checkpoint={"post_task_synthesis": "pending_once"})
    monkeypatch.setattr(pipeline, "in_worker_process", lambda: True)
    monkeypatch.setattr(pipeline, "_pre_synthesis_usage_snapshot", lambda *_a: {})
    monkeypatch.setattr("ouroboros.llm.LLMClient", lambda: object())
    monkeypatch.setattr("ouroboros.memory.Memory", lambda **_kw: object())
    for name in ("_run_chat_consolidation", "_run_scratchpad_consolidation", "_record_task_facts",
                 "_run_reflection", "_update_improvement_backlog", "_apply_reflection_memory_actions"):
        monkeypatch.setattr(pipeline, name, lambda *_a, **_kw: None)
    monkeypatch.setattr("ouroboros.post_task_evolution.maybe_promote", lambda *_a: None)
    env = SimpleNamespace(drive_root=child, repo_dir=tmp_path, drive_path=lambda rel: child / rel)
    pipeline._run_post_task_processing_async(env, task, {}, {}, {}, child / "logs", blocking=True)
    mailbox = owner_mailbox._mailbox_path(child, task_id)
    assert mailbox.is_file(), "post-task completion must not erase copyback's accepted input source"
    result = headless.copy_child_task_result(parent, task)
    assert result["child_ref_promotion"]["pending_refs"] == []
    captured = artifacts.task_artifact_dir_path(parent, task_id) / attachments[0]["relpath"]
    assert captured.read_bytes() == source.read_bytes()
    cleanup_settled_owner_mailbox(parent, task_id, task)
    assert not mailbox.exists()


@pytest.fixture
def terminal_context(tmp_path, monkeypatch):
    from supervisor import events, queue as task_queue

    task_queue.init(tmp_path)
    task = {"id": "prepared", "type": "task", "_attempt": 3, "chat_id": 1}
    worker = SimpleNamespace(busy_task_id=task["id"], reaping=False,
                             proc=SimpleNamespace(is_alive=lambda: True))
    pushed = []
    ctx = SimpleNamespace(
        DRIVE_ROOT=tmp_path, WORKERS={7: worker},
        RUNNING={task["id"]: {"task": task, "attempt": 3, "worker_id": 7}},
        bridge=SimpleNamespace(push_log=pushed.append),
        persist_queue_snapshot=lambda **_kw: None,
        send_with_budget=lambda *_a, **_kw: None,
        append_jsonl=append_jsonl,
    )
    monkeypatch.setattr("ouroboros.project_dialogue.append_terminal_task_projection", lambda *_a: None)
    monkeypatch.setattr("ouroboros.project_dialogue.enqueue_project_completion_summary", lambda *_a: None)
    monkeypatch.setattr(events, "_checkpoint_coop_roots_on_root_done", lambda *_a: None)
    monkeypatch.setattr(events, "_bound_project_chat_id", lambda *_a: None)
    monkeypatch.setattr("supervisor.update_merge.abort_orphaned_assisted_tx", lambda *_a: None)
    monkeypatch.setattr("supervisor.update_merge.release_assisted_writer_gate_after_task", lambda *_a: None)
    return ctx, worker, pushed


@pytest.mark.parametrize("status", ["completed", "cancelled"])
def test_prepared_frame_uses_current_disk_without_second_copy(terminal_context, monkeypatch, status):
    from supervisor.events import dispatch_event

    ctx, worker, pushed = terminal_context
    review = {"panels": [{"panel_id": "current", "aggregate_signal": "PASS", "actors": []}]}
    write_task_result(ctx.DRIVE_ROOT, "prepared", status, result="Current answer",
                      review_projection=review, artifact_status="ready")
    for name in ("copy_child_task_result", "finalize_task_artifacts", "prepare_terminal_task_files"):
        monkeypatch.setattr(headless, name, lambda *_a: pytest.fail("prepared frame repeated bulk I/O"))
    dispatch_event({"type": "task_done", "task_id": "prepared", "worker_id": 7,
                    "_files_prepared_attempt": 3, "status": "completed",
                    "review_projection": {"panels": [{"panel_id": "old"}]}}, ctx)
    assert "prepared" not in ctx.RUNNING and worker.busy_task_id is None
    assert pushed[-1]["status"] == status and pushed[-1]["review_projection"] == review
    assert "_files_prepared_attempt" not in pushed[-1]
    stored = load_task_result(ctx.DRIVE_ROOT, "prepared")
    assert stored["result"] == "Current answer" and "_files_prepared_attempt" not in stored


@pytest.mark.parametrize("stamp,wid", [(2, 7), (4, 7), (True, 7), ("3", 7), (3, 8)])
def test_stale_prepared_frame_cannot_finish_current_attempt(terminal_context, monkeypatch, stamp, wid):
    from supervisor.events import dispatch_event

    ctx, worker, pushed = terminal_context
    monkeypatch.setattr(headless, "copy_child_task_result", lambda *_a: pytest.fail("stale copy"))
    dispatch_event({"type": "task_done", "task_id": "prepared", "worker_id": wid,
                    "_files_prepared_attempt": stamp, "status": "completed"}, ctx)
    assert "prepared" in ctx.RUNNING and worker.busy_task_id == "prepared" and pushed == []
    assert load_task_result(ctx.DRIVE_ROOT, "prepared") is None


@pytest.fixture
def file_recovery(terminal_context, monkeypatch):
    """Keep the real queue/job functions, driving finite jobs without a resident thread."""
    from supervisor import queue as task_queue, workers

    ctx, worker, pushed = terminal_context
    jobs, returned = queue.Queue(), queue.Queue()
    monkeypatch.setattr(task_queue, "_ensure_reaper_started", lambda: None)
    monkeypatch.setattr(task_queue, "_reap_queue", jobs)
    monkeypatch.setattr(workers, "RUNNING", ctx.RUNNING)
    monkeypatch.setattr(workers, "WORKERS", ctx.WORKERS)
    monkeypatch.setattr(workers, "DRIVE_ROOT", ctx.DRIVE_ROOT)
    monkeypatch.setattr(workers, "get_event_q", lambda: returned)
    return ctx, worker, pushed, jobs, returned


def test_legacy_completion_uses_one_off_drain_job(file_recovery, monkeypatch):
    from supervisor import events, task_reaper

    ctx, worker, pushed, jobs, returned = file_recovery
    entered, release = threading.Event(), threading.Event()
    preparations = []

    def prepare(root, task):
        preparations.append(task["id"])
        entered.set()
        assert release.wait(5)
        write_task_result(root, task["id"], "completed", result="The saved answer", artifact_status="ready")
        return {"task_id": task["id"], "result": None, "error": ""}

    monkeypatch.setattr(headless, "prepare_terminal_task_files", prepare)
    event = {"type": "task_done", "task_id": "prepared", "worker_id": 7, "status": "completed"}
    events.dispatch_event(event, ctx)
    assert len(preparations) == 0 and jobs.qsize() == 1 and not pushed
    thread = threading.Thread(target=task_reaper._recover_terminal_files, args=(jobs.get_nowait(),))
    thread.start()
    try:
        assert entered.wait(5)
        events.dispatch_event(event, ctx)
        task_reaper.retry_terminal_file_recoveries()
        assert jobs.empty() and not pushed and worker.busy_task_id == "prepared"
        ctx.RUNNING["other"] = {"task": {"id": "other", "chat_id": 1}}
        events.dispatch_event({"type": "task_heartbeat", "task_id": "other"}, ctx)
        assert ctx.RUNNING["other"]["last_heartbeat_at"] > 0
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and preparations == ["prepared"]
    events.dispatch_event(event, ctx)  # Original duplicate is not a failed returned frame.
    task_reaper.retry_terminal_file_recoveries()
    assert jobs.empty()
    events.dispatch_event(returned.get_nowait(), ctx)
    assert worker.busy_task_id is None and "prepared" not in ctx.RUNNING
    assert pushed[-1]["status"] == "completed" and preparations == ["prepared"]


def test_canonical_write_outage_recovers_saved_child_without_model_replay(file_recovery, monkeypatch):
    from supervisor import events, task_reaper

    ctx, worker, pushed, jobs, returned = file_recovery
    task = ctx.RUNNING["prepared"]["task"]
    child = headless.prepare_task_drive(ctx.DRIVE_ROOT, "prepared", "empty")
    task.update(drive_root=str(child), budget_drive_root=str(ctx.DRIVE_ROOT))
    write_task_result(child, "prepared", "completed", result="Saved child answer", artifact_status="ready")
    write_task_result(ctx.DRIVE_ROOT, "prepared", "completed",
                      root_phase_checkpoint={"post_task_synthesis": "completed"})
    source = (child / "task_results" / "prepared.json").read_bytes()
    event = {"type": "task_done", "task_id": "prepared", "worker_id": 7,
             "status": "completed", "_files_prepared_attempt": 3}
    with monkeypatch.context() as fault:
        fault.setattr(headless, "write_task_result", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("disk full")))
        first = headless.prepare_terminal_task_files(ctx.DRIVE_ROOT, task)
        assert first["error"]
        events.dispatch_event(event, ctx)
        assert jobs.qsize() == 1 and pushed == []
        task_reaper._recover_terminal_files(jobs.get_nowait())
        assert returned.empty() and pushed == [] and worker.busy_task_id == "prepared"
        assert (child / "task_results" / "prepared.json").read_bytes() == source
    task_reaper.retry_terminal_file_recoveries()
    assert jobs.qsize() == 1
    task_reaper._recover_terminal_files(jobs.get_nowait())
    events.dispatch_event(returned.get_nowait(), ctx)
    assert worker.busy_task_id is None and "prepared" not in ctx.RUNNING
    assert pushed[-1]["status"] == "completed"
    assert load_task_result(ctx.DRIVE_ROOT, "prepared")["result"] == "Saved child answer"


@pytest.mark.parametrize("early_canonical", [False, True])
def test_missing_terminal_returns_to_existing_lifecycle_fault(file_recovery, early_canonical):
    from supervisor import events, task_reaper

    ctx, worker, pushed, jobs, returned = file_recovery
    if early_canonical:
        child = headless.prepare_task_drive(ctx.DRIVE_ROOT, "prepared", "empty")
        ctx.RUNNING["prepared"]["task"]["drive_root"] = str(child)
        write_task_result(ctx.DRIVE_ROOT, "prepared", "completed", result="Previously authored answer",
                          accounted_upper_bound_usd=12.5,
                          outcome_axes={"review": {"status": "pass"},
                                        "objective": {"status": "pass", "source": "task_acceptance_review"}},
                          root_phase_checkpoint={"post_task_synthesis": "completed"})
    events.dispatch_event({"type": "task_done", "task_id": "prepared", "worker_id": 7,
                           "status": "completed"}, ctx)
    task_reaper._recover_terminal_files(jobs.get_nowait())
    events.dispatch_event(returned.get_nowait(), ctx)
    assert worker.busy_task_id is None and "prepared" not in ctx.RUNNING
    stored = load_task_result(ctx.DRIVE_ROOT, "prepared")
    from ouroboros.project_dialogue import outcome_phase
    from ouroboros.task_status import _terminal_failure_from_outcome

    assert stored["status"] == ("completed" if early_canonical else "failed")
    assert outcome_phase(stored, pushed[-1]) == "error" and _terminal_failure_from_outcome(stored)
    assert stored["reason_code"] == "task_done_lifecycle_fault"
    if early_canonical:
        assert stored["result"] == "Previously authored answer"
        assert stored["accounted_upper_bound_usd"] == 12.5
        assert stored["outcome_axes"]["review"]["status"] == "pass"
        assert stored["outcome_axes"]["objective"]["status"] == "pass"


def test_unknown_legacy_frame_uses_current_durable_result(terminal_context, monkeypatch):
    from supervisor.events import dispatch_event

    ctx, worker, pushed = terminal_context
    ctx.RUNNING.clear()
    write_task_result(ctx.DRIVE_ROOT, "prepared", "cancelled", result="Current cancelled result")
    monkeypatch.setattr(headless, "prepare_terminal_task_files", lambda *_a: pytest.fail("legacy repeat"))
    dispatch_event({"type": "task_done", "task_id": "prepared", "status": "completed"}, ctx)
    assert pushed[-1]["status"] == "cancelled"


@pytest.mark.parametrize("status", ["completed", "cancelled"])
def test_lifecycle_fault_cannot_damage_current_ready_truth(terminal_context, status):
    from supervisor.events import _resolve_lifecycle_fault

    ctx, worker, pushed = terminal_context
    review = {"panels": [{"panel_id": "current-pass", "aggregate_signal": "PASS", "actors": []}]}
    write_task_result(ctx.DRIVE_ROOT, "prepared", status, result="Authored answer",
                      review_projection=review, accounted_upper_bound_usd=12.5,
                      outcome_axes={"review": {"status": "pass"},
                                    "objective": {"status": "pass", "source": "task_acceptance_review"}},
                      artifact_status="ready")
    _resolve_lifecycle_fault({"task_id": "prepared", "worker_id": 7}, ctx, "running")
    stored = load_task_result(ctx.DRIVE_ROOT, "prepared")
    assert stored["status"] == status and stored["result"] == "Authored answer"
    assert stored["accounted_upper_bound_usd"] == 12.5
    assert stored["review_projection"] == pushed[-1]["review_projection"] == review
    assert stored["outcome_axes"]["objective"]["status"] == "pass"
    assert not stored.get("reason_code") and not pushed[-1].get("reason_code")
    assert worker.busy_task_id is None


def test_unreadable_returned_frame_rearms_current_recovery(file_recovery, monkeypatch):
    from supervisor import events, events_task_done, task_reaper

    ctx, worker, pushed, jobs, returned = file_recovery
    write_task_result(ctx.DRIVE_ROOT, "prepared", "completed", result="Saved answer", artifact_status="ready")
    events.dispatch_event({"type": "task_done", "task_id": "prepared", "worker_id": 7}, ctx)
    task_reaper._recover_terminal_files(jobs.get_nowait())
    frame = returned.get_nowait()
    with monkeypatch.context() as fault:
        fault.setattr(events_task_done, "load_task_result", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("read unavailable")))
        events.dispatch_event(frame, ctx)
    assert not pushed and worker.busy_task_id == "prepared"
    task_reaper.retry_terminal_file_recoveries()
    task_reaper._recover_terminal_files(jobs.get_nowait())
    events.dispatch_event(returned.get_nowait(), ctx)
    assert worker.busy_task_id is None and pushed[-1]["status"] == "completed"
