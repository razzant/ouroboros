"""Actual invocation ownership, final admission and lazy custody ordering."""
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import closing, contextmanager, nullcontext
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
from tests._invocation_harness import _registry



from tests._invocation_harness import owner, run  # noqa: F401 -- shared (tests/_invocation_harness.py)


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


@pytest.mark.parametrize("control_kind", ["open", "pause", "inherited_deadline"])
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


@pytest.mark.parametrize("mode,statement", [("PERSIST", "COMMIT"), ("WAL", "BEGIN IMMEDIATE")])
def test_real_sql_reader_wait_releases_control_locks_and_keeps_one_tail(tmp_path, monkeypatch, mode, statement):
    from ouroboros import loop_tool_execution, store
    from tests._store_journal_shared import write_hold

    monkeypatch.setattr(store, "_journal_mode", lambda *_: mode)
    registry, queue, _ = _registry(tmp_path, monkeypatch)
    entered, finish = threading.Event(), threading.Event()
    held, attempting = threading.Event(), threading.Event()
    calls = []

    def body(*_a, **_kw):
        calls.append(True)
        entered.set()
        finish.wait()
        return "done once"

    registry.override_handler("knowledge_read", body)

    def timeout(*_args):
        held.wait()
        raise TimeoutError()

    monkeypatch.setattr(loop_tool_execution, "future_result", timeout)
    writing = threading.local()
    patch = task_store.patch_handoff
    drive = os.path.realpath(tmp_path)
    database = store.store_path(tmp_path)
    attempted_ids = set()

    def publish(root, task_id, kind, op_id, mutator):
        target = task_id == "root" and kind == "launch_handoffs" and os.path.realpath(root) == drive
        writing.tail = op_id if target else None
        try:
            return patch(root, task_id, kind, op_id, mutator)
        finally:
            writing.tail = None

    monkeypatch.setattr(task_store, "patch_handoff", publish)
    lease = store.connection

    def trace(sql):
        if held.is_set() and sql == statement:
            attempted_ids.add(writing.tail)
            attempting.set()

    @contextmanager
    def observe_connection(path, *args, **kwargs):
        with lease(path, *args, **kwargs) as conn:
            if path != database or not getattr(writing, "tail", None):
                yield conn
                return
            # SQLite reports the real handoff statement while the other thread
            # owns its conflicting transaction. The store keeps its own busy handler.
            conn.set_trace_callback(trace)
            try:
                yield conn
            finally:
                conn.set_trace_callback(None)

    monkeypatch.setattr(store, "connection", observe_connection)

    def call():
        with owner(registry):
            return run(registry)

    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(call)
        try:
            assert entered.wait(3)
            with closing(write_hold(database)) as blocker:
                assert blocker.execute("PRAGMA journal_mode").fetchone()[0].upper() == mode
                assert blocker.in_transaction
                held.set()
                assert attempting.wait(3) and not future.done()
                assert queue._queue_lock.acquire(blocking=False)
                try:
                    with owner_pause.launch_lock(tmp_path, "root", wait=False):
                        pass
                finally:
                    queue._queue_lock.release()
                assert not budget_pause.drain_local_tool_futures(registry._ctx, timeout_sec=0)["drained"]
            timed_out = future.result(timeout=3)
            assert timed_out["is_error"] and timed_out["tool_result"].code == "TOOL_TIMEOUT"
            invocation_id = timed_out["invocation_id"]
            active = claims(tmp_path)
            assert set(active) == attempted_ids == {invocation_id}
            assert (active[invocation_id]["state"], active[invocation_id]["tool"]) == ("claimed", "knowledge_read")
            assert calls == [True]
        finally:
            held.set()
            finish.set()
    assert budget_pause.drain_local_tool_futures(registry._ctx, timeout_sec=3)["drained"]
    assert not claims(tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "logs/tools.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["type"] for row in rows] == ["tool_call_started", "tool_call_timeout", "tool_call"]
    assert {row["invocation_id"] for row in rows} == {invocation_id}
    settled = rows[-1]
    assert settled["is_error"] is False and (settled["status"], settled["tool_result_code"]) == ("ok", "OK")
    assert settled["result_preview"] == "done once"


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
                    with pytest.raises(PhysicalDispatchInterrupted) as refused:
                        future.result(timeout=3)  # hard controls work while the database remains held
                    assert inner_bounds == []
                    assert getattr(refused.value, "invocation_not_started", None) is True, "nothing began"
                release_reader()
            if control == "open":
                assert not future.result(timeout=3)["is_error"]
                assert inner_bounds == [0.0]  # only final admission ignores the tool's wait envelope
        finally:
            proceed.set()
    assert not claims(tmp_path)


@pytest.mark.parametrize(("kind", "control"), [("mcp", "open"), ("isolated", "stop"), ("isolated", "deadline")])
def test_a_dynamic_entry_waits_out_a_contended_control_read_under_parent_controls(tmp_path, monkeypatch, kind, control):
    """MCP and isolated extensions have no run_tool_handler, so their entry gate reads the caller's controls itself;
    with no in-memory owner control that read goes to the store. Under real contention it waits the same way: off every
    lock, under the caller's controls rather than the tool's expired soft envelope, then one effect. A deadline seen
    while it waits, or the owner's durable Stop, whose writer competes for the same store and lands meanwhile, begins
    nothing and says so."""
    from ouroboros import cancel_intents, loop_tool_execution, queue_observation, safety, store
    from ouroboros.llm_attempt import PhysicalDispatchInterrupted
    from tests.test_admission_contention import database_reader
    from tests.test_live_effect_admission import dynamic_tool
    registry, _, _ = _registry(tmp_path, monkeypatch)
    effects, holders = [], []
    prepared, proceed, waiting, stopped = threading.Event(), threading.Event(), threading.Event(), threading.Event()
    name = dynamic_tool(monkeypatch, kind, effects)
    checked = safety.check_safety

    def after_preparation(tool, *args, **kwargs):  # registry preparation is done; the entry gate comes next
        if tool == name:
            prepared.set()
            assert proceed.wait(5)
        return checked(tool, *args, **kwargs)
    monkeypatch.setattr(safety, "check_safety", after_preparation)
    # A joined caller, as a foreground mutator's: the tool's soft wait expires while its admission still waits.
    monkeypatch.setattr(loop_tool_execution, "FOREGROUND_MUTATIVE_TOOLS",
                        loop_tool_execution.FOREGROUND_MUTATIVE_TOOLS | {name})
    sleep = queue_observation._sleep

    def observed_wait(seconds):
        first = not waiting.is_set()
        waiting.set()
        if first and control == "stop":  # the gate's first wait outlasts the owner's Stop
            assert stopped.wait(10), "the owner's Stop was never recorded"
        sleep(seconds)
    monkeypatch.setattr(queue_observation, "_sleep", observed_wait)

    def stop():
        while True:  # an ingress retries its own contention on the same store
            try:
                cancel_intents.request_cancel(tmp_path, "root", requested_stop_policy=cancel_intents.STOP_POLICY_IMMEDIATE)
                return stopped.set()
            except store.StoreBusy:
                time.sleep(.02)

    def call():
        with owner(registry) as held:
            holders.append(held)
            return run(registry, timeout=.05, name=name, args={})
    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(call)
        try:
            assert prepared.wait(3)
            with database_reader(tmp_path, exclusive=True) as release_reader:
                time.sleep(.08)  # the tool's soft wait expires before its entry gate
                proceed.set()
                assert waiting.wait(3), "the contended entry read did not wait"
                assert effects == []
                if control == "stop":
                    stopping = threading.Thread(target=stop)
                    stopping.start()
                elif control == "deadline":
                    holders[0].task["deadline_at"] = "2000-01-01T00:00:00Z"
                    with pytest.raises(PhysicalDispatchInterrupted) as refused:
                        future.result(timeout=3)  # seen while waiting, the database still held
                release_reader()
            if control == "stop":
                stopping.join(10)
                with pytest.raises(PhysicalDispatchInterrupted) as refused:
                    future.result(timeout=5)
            if control == "open":
                result = future.result(timeout=3)
                assert not result["is_error"], result["result"]
                assert effects == [name]
            else:
                assert getattr(refused.value, "invocation_not_started", None) is True, "nothing began"
                assert effects == []
        finally:
            proceed.set()
    assert not claims(tmp_path)


def one_contended_strict_read(monkeypatch, controls, *, stop=False, meanwhile=None):
    """Armed, the next strict control read meets a contended store once (rollback confirmed), as when another
    connection holds it; with ``stop`` the owner's Stop lands meanwhile, and ``meanwhile`` runs there. Only a
    dispatch gate reads strictly: launch validation and the waiting owner's own checks read non-strictly, so they pass
    through."""
    from ouroboros import model_wait, store
    armed, hits = [False], []
    read = model_wait.TaskModelWait.control_reason

    def control_reason(self, *, strict_current=False):
        if armed[0] and strict_current and not hits:
            hits.append(True)
            if stop:
                controls[0] = "cancelled"
            if meanwhile is not None:
                meanwhile()
            busy = store.StoreBusy("contended control read", reason="contention")
            busy.rollback_confirmed = True
            raise busy
        return read(self, strict_current=strict_current)
    monkeypatch.setattr(model_wait.TaskModelWait, "control_reason", control_reason)
    return lambda: armed.__setitem__(0, True), hits


@pytest.mark.parametrize(("gate", "control"), [("operation_start", "open"), ("submit_async_preparation", "stop")])
def test_a_nested_start_waits_out_its_contended_control_read_and_keeps_its_begun_phase(tmp_path, monkeypatch, gate,
                                                                                         control):
    """A started handler's nested start reads fresh controls in its own admission. When that read is contended, only the
    admission repeats, its launch lock released, and the nested effect then begins once. A Stop landing meanwhile starts
    nothing new, and the refusal says the invocation had begun."""
    import asyncio
    from ouroboros.llm_attempt import PhysicalDispatchInterrupted
    registry, _, _ = _registry(tmp_path, monkeypatch)
    controls, events = [None], []
    arm, hits = one_contended_strict_read(monkeypatch, controls, stop=control == "stop")

    def body(*_a, **_kw):
        events.append("handler")
        arm()
        if gate == "operation_start":
            with owner_pause.operation_start():
                events.append("nested")
        else:
            async def nested():
                events.append("nested")

            async def start():
                await owner_pause.submit_async_preparation(nested)
            asyncio.run(start())
        return "nested start done"
    registry.override_handler("knowledge_read", body)
    with owner(registry, control=controls):
        if control == "open":
            result = run(registry)
        else:
            with pytest.raises(PhysicalDispatchInterrupted) as refused:
                run(registry)
    assert hits == [True], "the nested gate's own strict read met the contended store"
    if control == "open":
        assert not result["is_error"], result["result"]
        assert events == ["handler", "nested"]
    else:
        assert events == ["handler"], "nothing new began after the Stop"
        assert getattr(refused.value, "invocation_not_started", None) is False, "the handler had begun"
    assert not claims(tmp_path)


@pytest.mark.parametrize("kind", ["builtin", "mcp"])
def test_a_contended_store_inside_a_body_is_never_retried(tmp_path, monkeypatch, kind):
    """Only the unsent gate waits. The same contention met by a body or its transport is that call's own error: the
    effect is entered once, never again."""
    from ouroboros import mcp_client, store
    from tests.test_live_effect_admission import dynamic_tool
    registry, _, _ = _registry(tmp_path, monkeypatch)
    entered = []

    def contended(*_a, **_kw):
        entered.append(kind)
        if len(entered) == 1:
            busy = store.StoreBusy("contended inside the effect", reason="contention")
            busy.rollback_confirmed = True
            raise busy
        return "entered a second time"
    if kind == "builtin":
        registry.override_handler("knowledge_read", contended)
        name, args = "knowledge_read", {"topic": "fixture"}
    else:
        name, args = dynamic_tool(monkeypatch, "mcp", []), {}
        monkeypatch.setattr(mcp_client, "_call_mcp_tool_result", contended)
    with owner(registry):
        result = run(registry, name=name, args=args)
    assert entered == [kind] and result["is_error"], result["result"]
    assert not claims(tmp_path)


SOFT_WAIT_ARGS = {"edit_text": {"path": "a.txt", "old_str": "x", "new_str": "y"},
                  "knowledge_read": {"topic": "fixture"}}


def nested_start(gate, events, label="nested"):
    """One nested start of a running handler: a process start's gate, or an async transport's preparation gate."""
    import asyncio
    if gate == "operation_start":
        with owner_pause.operation_start():
            events.append(label)
        return

    async def nested():
        events.append(label)

    async def start():
        await owner_pause.submit_async_preparation(nested)
    asyncio.run(start())


def nested_start_after_soft_wait(registry, monkeypatch, tool, gate, *, control, clock, deadline, before_start,
                                 reader=None):
    """Spend the soft wait on a test clock only after handler entry, then make one nested start. For a read the
    caller leaves first; a foreground edit must join. Keep the task owner live for the tail, but bind the End
    reader only around the call, as a live call's batch does."""
    from ouroboros import loop_tool_execution, model_wait
    from ouroboros.llm_attempt import PhysicalDispatchInterrupted
    events, body_bounds, refusals, outcome = [], [], [], {}
    entered, proceed, done = (threading.Event() for _ in range(3))

    def monotonic_now(_slot=None):
        return clock[0]

    def timeout_after_entry(future, timeout):
        assert entered.wait(5), "the handler did not enter before the simulated soft wait"
        clock[0] += timeout
        try:
            return future.result(timeout=0)  # The handler is parked, so this reports a real Future timeout.
        finally:
            if tool == "edit_text":
                proceed.set()  # The loop joins a foreground mutator rather than returning a timeout.

    monkeypatch.setattr(model_wait, "monotonic_now", monotonic_now)
    monkeypatch.setattr(loop_tool_execution, "monotonic_now", monotonic_now)
    monkeypatch.setattr(loop_tool_execution, "future_result", timeout_after_entry)

    def body(*_a, _resolved_binding=None, **_kw):
        try:
            entered.set()
            assert proceed.wait(5), "the test did not release the handler after the soft wait"
            body_bounds.append(model_wait.dispatch_deadline_remaining_sec())
            assert clock[0] < deadline, "the caller's bound is still open after the soft wait"
            before_start()
            try:
                nested_start(gate, events)
            except (PhysicalDispatchInterrupted, owner_pause.OwnerPauseRefused) as refused:
                refusals.append((type(refused).__name__, getattr(refused, "control_reason", str(refused)),
                                 getattr(refused, "invocation_not_started", None)))
                raise
            return "applied"
        finally:
            done.set()
    registry.override_handler(tool, body)
    with owner(registry, control=control):
        try:
            with model_wait.execution_deadline_scope(deadline), (
                    owner_pause.call_end_reader(registry._ctx, reader) if reader is not None else nullcontext()):
                try:
                    result = run(registry, timeout=.05, name=tool, args=SOFT_WAIT_ARGS[tool])
                    outcome["is_error"] = result["is_error"]
                    if tool == "knowledge_read":
                        assert result["tool_result"].code == "TOOL_TIMEOUT" and not done.is_set()
                except (PhysicalDispatchInterrupted, owner_pause.OwnerPauseRefused) as refused:
                    if tool == "knowledge_read":
                        # A read's caller must return TOOL_TIMEOUT; a caller-side control refusal is a defect.
                        # Re-raise so the parked body's own expected refusal cannot mask the wrong caller outcome.
                        raise
                    outcome["raised"] = type(refused).__name__
        finally:
            proceed.set()
            assert budget_pause.drain_local_tool_futures(registry._ctx, timeout_sec=5)["drained"]
        assert done.is_set(), outcome
    return events, body_bounds, refusals, outcome


@pytest.mark.parametrize(("tool", "gate", "control"), [
    ("edit_text", "operation_start", "open"),
    ("edit_text", "operation_start", "end"),
    ("edit_text", "operation_start", "durable_stop"),
    ("knowledge_read", "submit_async_preparation", "caller_deadline"),
    ("knowledge_read", "submit_async_preparation", "stop"),
    ("knowledge_read", "submit_async_preparation", "pause"),
])
def test_a_nested_start_after_the_soft_wait_keeps_the_callers_bounds(tmp_path, monkeypatch, tool, gate, control):
    """A tool's soft wait decides only when the loop stops waiting for it. A later nested start of its body is a new
    start whose gate reads the bounds the invocation inherited, as nested starts did before that gate existed: it
    begins while they allow, and the caller's own expired deadline, the call's End, a Stop or a Pause refuse it,
    saying the invocation had begun. The body keeps its spent envelope, so its own model sends still stop."""
    from ouroboros import cancel_intents
    registry, _, _ = _registry(tmp_path, monkeypatch)
    controls, ended = [None], [None]
    clock = [0.0]
    deadline = .4 if control == "caller_deadline" else 30

    def before_start():
        if control == "caller_deadline":
            clock[0] = deadline + 1
        elif control == "end":
            ended[0] = "owner_end"
        elif control == "stop":
            controls[0] = "cancelled"
        elif control == "durable_stop":  # no in-memory control: the gate reads the owner's recorded Stop
            cancel_intents.request_cancel(tmp_path, "root", requested_stop_policy=cancel_intents.STOP_POLICY_IMMEDIATE)
        elif control == "pause":
            owner_pause.install_fence(tmp_path, "root", request_id="after-the-soft-wait")
    events, body_bounds, refusals, outcome = nested_start_after_soft_wait(
        registry, monkeypatch, tool, gate, control=None if control == "durable_stop" else controls,
        clock=clock, deadline=deadline,
        before_start=before_start, reader=lambda: ended[0])
    assert body_bounds == [0.0], "the body itself is past its soft wait"
    if control == "open":
        assert events == ["nested"] and refusals == []
        assert outcome == {"is_error": tool == "knowledge_read"}, outcome  # joined and applied / left in background
    elif control == "pause":
        assert events == [] and [refusal[0] for refusal in refusals] == ["OwnerPauseRefused"]
    else:
        reason = {"caller_deadline": "execution_deadline", "end": "owner_end"}.get(control, "cancelled")
        assert events == [] and refusals == [("PhysicalDispatchInterrupted", reason, False)], refusals
    assert not claims(tmp_path)


@pytest.mark.parametrize(("tool", "gate", "control"), [
    ("edit_text", "operation_start", "open"),
    ("edit_text", "operation_start", "stop"),
    ("knowledge_read", "submit_async_preparation", "caller_deadline"),
])
def test_a_contended_nested_start_after_the_soft_wait_waits_under_the_callers_bounds(tmp_path, monkeypatch, tool, gate,
                                                                                    control):
    """The same nested start when its gate's read meets a contended store: its wait checks the caller's bounds too, so
    the spent soft envelope never ends it and the start then begins once, its lock released while it waits. The
    caller's deadline or a Stop seen by that check refuse it there, before any sleep, saying the invocation had
    begun."""
    from ouroboros import queue_observation
    from ouroboros._usage_wait import ACQUISITION_SLICE_SEC
    registry, _, _ = _registry(tmp_path, monkeypatch)
    controls = [None]
    clock = [0.0]
    deadline = .4 if control == "caller_deadline" else 30

    def expire():  # the caller's own bound passes while the read is contended
        clock[0] = deadline + 1
    arm, hits = one_contended_strict_read(monkeypatch, controls, stop=control == "stop",
                                          meanwhile=expire if control == "caller_deadline" else None)
    waits, sleep = [], queue_observation._sleep

    def observed_wait(seconds):
        waits.append(seconds)
        sleep(seconds)
    monkeypatch.setattr(queue_observation, "_sleep", observed_wait)
    events, body_bounds, refusals, _outcome = nested_start_after_soft_wait(
        registry, monkeypatch, tool, gate, control=controls, clock=clock, deadline=deadline, before_start=arm)
    assert hits == [True] and body_bounds == [0.0]
    if control == "open":
        assert events == ["nested"] and refusals == [] and waits == [ACQUISITION_SLICE_SEC], (refusals, waits)
    else:
        reason = "execution_deadline" if control == "caller_deadline" else "cancelled"
        assert events == [] and waits == [], "the wait's own check refused it"
        assert refusals == [("PhysicalDispatchInterrupted", reason, False)], refusals
    assert not claims(tmp_path)


def test_concurrent_nested_starts_check_their_own_copies_of_the_caller_context(tmp_path, monkeypatch):
    """Three threads of one body start nested effects together. The first holds the launch lock inside its gate read
    while the other two, refused that lock, check their controls at the same moment in their waits. Each read runs in
    its own copy of the invocation's caller context, which no thread enters itself (a context entered twice at once
    raises); each lock is released by the thread and context that took it, and every start begins once."""
    import contextvars
    from ouroboros import model_wait
    registry, _, _ = _registry(tmp_path, monkeypatch)
    controls, events, errors = [None], [], []
    holding, waited = threading.Event(), threading.Event()
    together = threading.Barrier(2, timeout=3)
    gate, read = owner_pause.require_tool_dispatch, model_wait.TaskModelWait.control_reason

    def held_gate(source):
        if threading.current_thread().name == "nested-first" and not holding.is_set():
            holding.set()
            assert waited.wait(5), "the refused starts never checked their controls together"
        return gate(source)

    def control_reason(self, *, strict_current=False):
        if (not strict_current and threading.current_thread().name in {"nested-second", "nested-third"}
                and not waited.is_set()):
            together.wait()  # the two waits' checks overlap
            waited.set()
        return read(self, strict_current=strict_current)
    monkeypatch.setattr(owner_pause, "require_tool_dispatch", held_gate)
    monkeypatch.setattr(model_wait.TaskModelWait, "control_reason", control_reason)

    def start(label):
        try:
            nested_start("operation_start", events, label)
        except BaseException as exc:  # noqa: BLE001 - reported below with its thread
            errors.append(f"{label}: {type(exc).__name__}: {exc}")

    def body(*_a, **_kw):
        threads = {label: threading.Thread(target=contextvars.copy_context().run, args=(start, label),
                                           name=f"nested-{label}") for label in ("first", "second", "third")}
        threads["first"].start()
        assert holding.wait(3)
        threads["second"].start()
        threads["third"].start()
        for thread in threads.values():
            thread.join(10)
        return "started"
    registry.override_handler("knowledge_read", body)
    with owner(registry, control=controls):
        result = run(registry, timeout=10)
    assert errors == [], errors
    assert events[0] == "first" and sorted(events) == ["first", "second", "third"], events
    assert not result["is_error"], result["result"]
    root_drive, root_id, _task = owner_pause._member_coordinates(registry._ctx)
    with owner_pause.launch_lock(root_drive, root_id, wait=False):  # every start released its own lock
        pass
    assert not claims(tmp_path)


@pytest.mark.parametrize("control", ["open", "stop"])
@pytest.mark.parametrize("form", ["native", "gateway"])
def test_a_gateway_call_has_the_custody_of_its_native_mcp_call(tmp_path, monkeypatch, form, control):
    """A call_mcp_tool envelope resolved at the executor entry runs as its MCP tool's own call: the same tool name on
    the started and settled rows (the envelope recorded as ``via``), no claim left open, and an owner Stop before
    launch begins nothing, exactly as for the native call."""
    from ouroboros.llm_attempt import PhysicalDispatchInterrupted
    from ouroboros.tools.tool_discovery import resolve_gateway_call
    from tests.test_live_effect_admission import dynamic_tool
    registry, _, _ = _registry(tmp_path, monkeypatch)
    effects = []
    name = dynamic_tool(monkeypatch, "mcp", effects)
    call = {"id": "call", "function": {"name": name, "arguments": json.dumps({"query": "fixture"})}}
    if form == "gateway":
        envelope = {"name": name, "arguments": {"query": "fixture"}}
        call = resolve_gateway_call({"id": "call", "function": {"name": "call_mcp_tool",
                                                                "arguments": json.dumps(envelope)}})
        assert (call["function"]["name"], call["via"]) == (name, "call_mcp_tool")
    with owner(registry, control=["cancelled" if control == "stop" else None]):
        if control == "stop":
            with pytest.raises(PhysicalDispatchInterrupted):
                _execute_with_timeout(registry, call, registry._ctx.drive_root / "logs", 3, task_id=registry._ctx.task_id)
        else:
            result = _execute_with_timeout(registry, call, registry._ctx.drive_root / "logs", 3,
                                           task_id=registry._ctx.task_id)
            assert result["fn_name"] == name
    assert bool(effects) is (control == "open")
    log = registry._ctx.drive_root / "logs" / "tools.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    assert (bool(rows) or control == "stop") and {row.get("tool") for row in rows} <= {name}
    assert {row.get("via") for row in rows} <= {"call_mcp_tool" if form == "gateway" else None}
    assert not claims(tmp_path)
