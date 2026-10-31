"""Actual invocation ownership, final admission and lazy custody ordering."""
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
import datetime
import json
import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import budget_pause, owner_pause, task_store, tool_custody
from ouroboros.loop_tool_execution import _execute_with_timeout
from ouroboros.model_wait import task_model_wait_scope
from ouroboros.task_results import load_task_result, write_task_result
from tests.test_batch4_producer_custody import _registry

pytestmark = pytest.mark.serial


@contextmanager
def owner(registry, *, attempt=1, control=None):
    ctx = registry._ctx
    with task_model_wait_scope(task={"id": ctx.task_id, "_attempt": attempt},
            drive_root=ctx.drive_root, event_queue=None, worker_slot_held=False,
            **({"owner_control": lambda: control[0]} if control is not None else {})) as held:
        held.tool_context = ctx
        yield held


def run(registry, *, timeout=3, name="knowledge_read", args=None):
    return _execute_with_timeout(registry, {"id": "call", "function": {
        "name": name, "arguments": json.dumps(args or {"topic": "fixture"})}},
        registry._ctx.drive_root / "logs", timeout, task_id=registry._ctx.task_id)


def claims(root):
    return load_task_result(root, "root", strict=True).get("launch_handoffs", {})


def test_joined_read_has_one_executor_and_no_claim_despite_delayed_passive_callback(tmp_path, monkeypatch):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    (registry._ctx.workspace_root / "source.txt").write_text("source bytes", encoding="utf-8")
    release, entered, registered = threading.Event(), threading.Event(), threading.Event()
    add_done, init, patch = Future.add_done_callback, ThreadPoolExecutor.__init__, task_store.patch_handoff
    writes, executors = [], []
    read = registry._entries["read_file"].handler
    def after_registration(*args, _resolved_binding=None, **kwargs):
        assert registered.wait(3)
        return read(*args, _resolved_binding=_resolved_binding, **kwargs)
    registry.override_handler("read_file", after_registration)

    def delayed(self, callback):
        if getattr(callback, "__name__", "") == "done":
            def paused(value):
                entered.set()
                assert release.wait(5)
                callback(value)
            result = add_done(self, paused)
            registered.set()
            return result
        return add_done(self, callback)

    def record(*args, **kwargs):
        if args[2] == "launch_handoffs":
            writes.append(args[3])
        return patch(*args, **kwargs)

    def count(self, *args, **kwargs):
        executors.append(True)
        return init(self, *args, **kwargs)

    monkeypatch.setattr(Future, "add_done_callback", delayed)
    monkeypatch.setattr(ThreadPoolExecutor, "__init__", count)
    monkeypatch.setattr(task_store, "patch_handoff", record)
    try:
        with owner(registry):
            result = run(registry, name="read_file", args={"path": "source.txt"})
            assert not result["is_error"] and "source bytes" in result["result"], result
            assert entered.wait(2) and writes == [] and executors == [True]
            assert budget_pause.drain_local_tool_futures(registry._ctx, timeout_sec=0)["drained"]
    finally:
        release.set()


@pytest.mark.parametrize("control_kind", ["open", "pause", "stop", "deadline", "inherited_deadline"])
def test_fresh_control_after_registry_preparation_precedes_actual_handler(tmp_path, monkeypatch, control_kind):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    prepared, release = threading.Event(), threading.Event()
    controls, holders, effects = [None], [], []
    from ouroboros import model_wait
    hard_deadline = model_wait.monotonic_now() + 60
    registry.override_handler("knowledge_read", lambda *_a, **_k: effects.append(True) or "done")
    invoke = registry._invoke_builtin_handler

    def preparation(*args, **kwargs):
        prepared.set()
        assert release.wait(5)
        return invoke(*args, **kwargs)

    monkeypatch.setattr(registry, "_invoke_builtin_handler", preparation)

    def call():
        with owner(registry, control=controls) as held, owner_pause.review_episode("root") as episode:
            holders.append((held, episode))
            with (model_wait.execution_deadline_scope(hard_deadline)
                  if control_kind == "inherited_deadline" else nullcontext()):
                return run(registry)

    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(call)
        try:
            assert prepared.wait(3)
            if control_kind == "pause":
                owner_pause.install_fence(tmp_path, "root", request_id="after-preparation")
            elif control_kind == "stop":
                controls[0] = "cancelled"
            elif control_kind == "inherited_deadline":
                monkeypatch.setattr(model_wait, "monotonic_now", lambda _slot=None: hard_deadline + 1)
            elif control_kind == "deadline":
                holders[0][0].task["deadline_at"] = (datetime.datetime.now(datetime.timezone.utc)
                    - datetime.timedelta(seconds=1)).isoformat()
        finally:
            release.set()
        if control_kind in {"stop", "deadline", "inherited_deadline"}:
            from ouroboros.llm_attempt import PhysicalDispatchInterrupted
            with pytest.raises(PhysicalDispatchInterrupted):
                future.result(timeout=5)
        else:
            future.result(timeout=5)
    assert bool(effects) is (control_kind == "open")
    assert owner_pause.episode_armed(holders[0][1]) is (control_kind == "open")
    assert not claims(tmp_path)


@pytest.mark.parametrize("fail_publication", [False, True])
def test_completion_during_publication_settles_the_same_identity(tmp_path, monkeypatch, fail_publication):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    entered, finish = threading.Event(), threading.Event()
    identities = []

    def body(*_a, **_kw):
        entered.set()
        assert finish.wait(5)
        return "joined"

    registry.override_handler("knowledge_read", body)
    from ouroboros import loop_tool_execution
    def timeout_after_entry(*_args):
        assert entered.wait(3)
        raise TimeoutError()
    monkeypatch.setattr(loop_tool_execution, "future_result", timeout_after_entry)
    retain = tool_custody.retain_invocation

    def complete_before_commit(outcome, update=None):
        if outcome.get("memory_owned") and not identities:
            identities.append(outcome["operation_id"])
            callbacks = threading.Event()
            outcome["future_row"]["future"].add_done_callback(lambda _f: callbacks.set())
            finish.set()
            assert callbacks.wait(3)
            retain(outcome, update)
            if fail_publication:
                raise OSError("lost publication reply after COMMIT")
            return
        return retain(outcome, update)

    monkeypatch.setattr(tool_custody, "retain_invocation", complete_before_commit)
    try:
        with owner(registry):
            result = run(registry)
            assert result["is_error"] and entered.is_set() and len(identities) == 1
            assert budget_pause.drain_local_tool_futures(registry._ctx, timeout_sec=0)["drained"]
        assert not claims(tmp_path)
    finally:
        finish.set()


def test_failed_publication_holds_direct_caller_through_real_cleanup(tmp_path, monkeypatch):
    registry, queue, _ = _registry(tmp_path, monkeypatch)
    cleanup, entered, failed, body_done = (threading.Event() for _ in range(4))
    controls, holders = [None], []

    def body(*_a, **_kw):
        release = tool_custody.hold_invocation()
        def tail():
            try:
                entered.set()
                assert cleanup.wait(5)
            finally:
                release()
        thread = threading.Thread(target=tail)
        thread.start()
        holders.append(thread)
        body_done.set()
        return "handler joined; cleanup remains"

    registry.override_handler("knowledge_read", body)
    def refuse(*_a, **_kw):
        failed.set()
        raise OSError("custody drive unavailable")
    monkeypatch.setattr(tool_custody, "retain_invocation", refuse)

    def call():
        with owner(registry, control=controls) as held:
            holders.append(held)
            return run(registry)

    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(call)
        try:
            assert entered.wait(3) and body_done.wait(3) and failed.wait(3)
            controls[0] = "cancelled"  # direct Stop cannot kill a Python thread
            assert not future.done() and not holders[0].closed
            with queue._queue_lock, owner_pause.launch_lock(tmp_path, "root", wait=False):
                pass
            assert not budget_pause.drain_local_tool_futures(registry._ctx, timeout_sec=0)["drained"]
        finally:
            cleanup.set()
        assert not future.result(timeout=5)["is_error"]
    holders[1].join(3)
    assert holders[0].closed and not holders[1].is_alive()


@pytest.mark.parametrize("member", [False, True])
def test_control_cache_never_pairs_old_body_with_new_pause_signature(tmp_path, monkeypatch, member):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    if member:
        write_task_result(tmp_path, "child", "running", root_task_id="root")
        registry._ctx.task_id = "child"
    target = "child" if member else "root"
    from ouroboros import store_hot_facts
    read, captured = store_hot_facts.controls, []

    def interleaved(root, task_id):
        snapshot = read(root, task_id)
        if task_id == target and not captured:
            captured.append(True)
            if member:
                write_task_result(root, task_id, "running", root_task_id="root",
                                  budget_pause={"state": "paused", "reason": "sleep", "pause_id": "new"})
            else:
                owner_pause.install_fence(root, "root", request_id="snapshot-interleave")
        return snapshot

    monkeypatch.setattr(store_hot_facts, "controls", interleaved)
    owner_pause._control_projection(tmp_path, target)  # writer commits after this control snapshot
    with pytest.raises(owner_pause.OwnerPauseRefused):
        with owner_pause.launch_admission(registry._ctx):
            pytest.fail("stale open control was admitted")
    assert captured == [True]


def test_admission_reads_current_controls_by_address_never_a_body(tmp_path, monkeypatch):
    """Nothing is memoized across admissions: each reads the typed control columns by address, so a
    write to the row (here a handoff touch) can never leave a stale projection, and no body is decoded."""
    registry, _, _ = _registry(tmp_path, monkeypatch)
    from ouroboros import store_hot_facts, task_results
    read_body, read_controls, bodies, addressed = task_results.load_task_result, store_hot_facts.controls, [], []
    monkeypatch.setattr(task_results, "load_task_result", lambda *a, **k: bodies.append(a[1]) or read_body(*a, **k))
    monkeypatch.setattr(store_hot_facts, "controls", lambda root, task_id: addressed.append(task_id) or read_controls(root, task_id))
    with owner_pause.launch_admission(registry._ctx):
        pass
    per_admission = list(addressed)
    task_store.patch_handoff(tmp_path, "root", "model_waits", "wait", lambda _old: {"state": "waiting"})
    with owner_pause.launch_admission(registry._ctx):
        pass
    assert per_admission and addressed == per_admission * 2 and bodies == []


def test_external_process_is_custodied_before_both_publication_failures(tmp_path, monkeypatch):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    from ouroboros import workspace_executor
    from ouroboros.tools import shell_process
    patch, popen = task_store.patch_handoff, subprocess.Popen
    processes = []

    def start(*args, **kwargs):
        if args[0] != [sys.executable, "-c", "import time; time.sleep(30)"]:
            return popen(*args, **kwargs)
        assert any(row.get("external_pending") for row in claims(tmp_path).values())
        proc = popen(*args, **kwargs)
        processes.append(proc)
        return proc

    def fail_update(*args, **kwargs):
        if processes and args[2] == "launch_handoffs":
            raise OSError("database failed after spawn")
        return patch(*args, **kwargs)

    def fail_record(*_a, **_kw):
        raise OSError("independent process custody failed")

    registry.override_handler("knowledge_read", lambda *_a, **_kw: shell_process._tracked_subprocess_run(
        [sys.executable, "-c", "import time; time.sleep(30)"], stdout=subprocess.PIPE, stderr=subprocess.PIPE))
    monkeypatch.setattr(subprocess, "Popen", start)
    monkeypatch.setattr(task_store, "patch_handoff", fail_update)
    monkeypatch.setattr(workspace_executor, "_register_process", fail_record)
    try:
        with owner(registry):
            result = run(registry)
        assert result["is_error"] and processes[0].poll() is None
        assert any(row.get("external_pending") for row in claims(tmp_path).values())
    finally:
        for proc in processes:
            proc.kill()
            proc.wait(timeout=3)
            shell_process._active_subprocesses.discard(proc)


@pytest.mark.parametrize("kind", ["shell", "skill"])
def test_unbound_process_entry_remains_usable(tmp_path, kind):
    command = [sys.executable, "-c", "print('unbound works')"]
    if kind == "shell":
        from ouroboros.tools.shell_process import _tracked_subprocess_run
        result = _tracked_subprocess_run(command, cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert result.returncode == 0 and b"unbound works" in result.stdout
    else:
        from ouroboros.tools.skill_exec import _run_skill_subprocess
        code, stdout, _stderr, overflow = _run_skill_subprocess(command, cwd=str(tmp_path), env=dict(os.environ),
            timeout_sec=3, stdout_cap=4096, stderr_cap=4096)
        assert code == 0 and b"unbound works" in stdout and not overflow


def test_native_reviewer_tail_uses_frozen_author_attempt(tmp_path, monkeypatch):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    from ouroboros import loop_tool_execution
    from ouroboros.review_native_episode import NativeToolRoundReviewExecutor, _ToolAbandoned
    entered, finish = threading.Event(), threading.Event()
    def body(*_a, **_kw):
        entered.set()
        assert finish.wait(5)
        return "native done"
    registry.override_handler("knowledge_read", body)
    from ouroboros import model_wait
    monkeypatch.setattr(loop_tool_execution, "_get_tool_timeout", lambda *_a: 5)
    def timeout_after_entry(*_args):
        assert entered.wait(3)
        raise TimeoutError()
    monkeypatch.setattr(model_wait, "future_result", timeout_after_entry)
    author = SimpleNamespace(task_id="root", task_attempt=3, drive_root=tmp_path)
    try:
        with owner(registry, attempt=3):
            assert registry._ctx.task_attempt == 1  # inspection context did not acquire the author's attempt
            with pytest.raises(_ToolAbandoned):
                NativeToolRoundReviewExecutor._execute_bounded(None, registry, "knowledge_read", {"topic": "x"})
            assert entered.is_set()
            assert not budget_pause.drain_local_tool_futures(author, timeout_sec=0)["drained"]
            claim = next(iter(claims(tmp_path).values()))
            assert claim["local_owner"]["task_attempt"] == 3
            budget_pause.forget_tool_scope(author)
            assert not budget_pause.drain_local_tool_futures(author, timeout_sec=0)["drained"]
            finish.set()
            assert budget_pause.drain_local_tool_futures(author, timeout_sec=3)["drained"]
            with budget_pause._TOOL_LOCK:
                assert budget_pause.tool_scope_key(author) not in budget_pause._TOOL_FUTURES
        assert not claims(tmp_path)
    finally:
        finish.set()


def test_cancelled_before_entry_has_one_host_settlement(tmp_path, monkeypatch):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    from ouroboros import loop_tool_execution
    pending, effects = Future(), []
    registry.override_handler("knowledge_read", lambda *_a, **_kw: effects.append(True) or "done")
    @contextmanager
    def queued(*_args, **_kwargs):
        try:
            yield lambda *_a: pending
        finally:
            pending.cancel()
    monkeypatch.setattr(loop_tool_execution, "abandoned_on_timeout", queued)
    with owner(registry):
        result = run(registry, timeout=.01)
    assert result["is_error"] and pending.cancelled() and effects == []
    rows = [json.loads(line) for line in (tmp_path / "logs/tools.jsonl").read_text(encoding="utf-8").splitlines()]
    settled = [row for row in rows if row.get("type") == "tool_call"]
    assert len(settled) == 1 and settled[0]["status"] == "host_error"
    assert len({row["invocation_id"] for row in rows}) == 1
    assert not claims(tmp_path)


def test_real_sql_reader_wait_releases_control_locks_and_keeps_one_tail(tmp_path, monkeypatch):
    registry, queue, _ = _registry(tmp_path, monkeypatch)
    from ouroboros import loop_tool_execution
    from tests.test_admission_contention import database_reader
    entered, finish, publishing = threading.Event(), threading.Event(), threading.Event()
    calls = []
    def body(*_a, **_kw):
        calls.append(True)
        entered.set()
        assert finish.wait(5)
        return "done once"
    registry.override_handler("knowledge_read", body)
    def timeout(*_args):
        assert entered.wait(3)
        raise TimeoutError()
    monkeypatch.setattr(loop_tool_execution, "future_result", timeout)
    retain = tool_custody.retain_invocation
    def publishing_tail(outcome, update=None):
        publishing.set()
        return retain(outcome, update)
    monkeypatch.setattr(tool_custody, "retain_invocation", publishing_tail)
    def call():
        with owner(registry):
            return run(registry)
    with ThreadPoolExecutor(1) as executor:
        try:
            with database_reader(tmp_path) as release_reader:
                future = executor.submit(call)
                assert publishing.wait(3) and not future.done()
                with queue._queue_lock, owner_pause.launch_lock(tmp_path, "root", wait=False):
                    pass
                assert not budget_pause.drain_local_tool_futures(registry._ctx, timeout_sec=0)["drained"]
                release_reader()
                assert future.result(timeout=3)["is_error"]
                assert len(claims(tmp_path)) == 1 and calls == [True]
        finally:
            finish.set()
    assert budget_pause.drain_local_tool_futures(registry._ctx, timeout_sec=3)["drained"]
    assert not claims(tmp_path)


def test_model_wait_close_prunes_its_frozen_scope_under_another_current_attempt(tmp_path):
    from ouroboros.model_wait import TaskModelWait
    old = TaskModelWait(task={"id": "root", "_attempt": 3}, drive_root=tmp_path,
                        event_queue=None, worker_slot_held=False)
    context = SimpleNamespace(task_id="root", task_attempt=3, drive_root=tmp_path)
    done = Future()
    done.set_result("joined")
    budget_pause.register_tool_future(context, "old", "read_file", done)()
    old_key = budget_pause.tool_scope_key(context, attempt=3)
    current = Future()
    try:
        with task_model_wait_scope(task={"id": "root", "_attempt": 4}, drive_root=tmp_path,
                                   event_queue=None, worker_slot_held=False):
            budget_pause.register_tool_future(context, "new", "read_file", current)()
            new_key = budget_pause.tool_scope_key(context, attempt=4)
            old.close()  # no answer consumer, and the ContextVar belongs to attempt four
            with budget_pause._TOOL_LOCK:
                assert old_key not in budget_pause._TOOL_FUTURES
                assert new_key in budget_pause._TOOL_FUTURES
    finally:
        current.set_result("late joined")
    with budget_pause._TOOL_LOCK:
        assert new_key not in budget_pause._TOOL_FUTURES


@pytest.mark.parametrize("control", ["open", "stop", "deadline"])
def test_final_admission_retry_uses_parent_controls_while_body_keeps_soft_envelope(tmp_path, monkeypatch, control):
    from ouroboros import model_wait, queue_observation
    from ouroboros.llm_attempt import PhysicalDispatchInterrupted
    from tests.test_admission_contention import database_reader
    registry, _, _ = _registry(tmp_path, monkeypatch)
    prepared, proceed, waiting = threading.Event(), threading.Event(), threading.Event()
    controls, holders, inner_bounds = [None], [], []
    def body(*_a, _resolved_binding=None, **_kw):
        inner_bounds.append(model_wait.dispatch_deadline_remaining_sec())
        return "applied"
    registry.override_handler("edit_text", body)
    invoke = registry._invoke_builtin_handler
    def preparation(*args, **kwargs):
        prepared.set()
        assert proceed.wait(5)
        return invoke(*args, **kwargs)
    monkeypatch.setattr(registry, "_invoke_builtin_handler", preparation)
    sleep = queue_observation._sleep
    def observed_wait(seconds):
        waiting.set()
        sleep(seconds)
    monkeypatch.setattr(queue_observation, "_sleep", observed_wait)
    def call():
        with owner(registry, control=controls) as held:
            holders.append(held)
            return run(registry, timeout=.05, name="edit_text",
                       args={"path": "a.txt", "old_str": "x", "new_str": "y"})
    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(call)
        try:
            assert prepared.wait(3)
            with database_reader(tmp_path, exclusive=True) as release_reader:
                time.sleep(.08)  # the tool wait expires; its foreground owner still joins
                proceed.set()
                assert waiting.wait(3), "soft envelope vetoed the unsent admission retry"
                if control == "stop":
                    controls[0] = "cancelled"
                elif control == "deadline":
                    holders[0].task["deadline_at"] = "2000-01-01T00:00:00Z"
                if control != "open":
                    with pytest.raises(PhysicalDispatchInterrupted):
                        future.result(timeout=3)  # hard controls work while the database remains held
                    assert inner_bounds == []
                release_reader()
            if control == "open":
                assert not future.result(timeout=3)["is_error"]
                assert inner_bounds == [0.0]  # only final admission ignores the tool's wait envelope
        finally:
            proceed.set()
    assert not claims(tmp_path)
