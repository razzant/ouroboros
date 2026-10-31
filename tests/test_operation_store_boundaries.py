"""Positive no-entry, exact completion and closed-context custody consumers."""
import sys
from contextvars import copy_context

import pytest

from ouroboros import owner_pause, store, task_store, tool_custody
from tests.test_batch4_producer_custody import _registry
from tests.test_invocation_ownership import claims, owner, run

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("after_body", [False, True])
def test_extension_store_refusal_preserves_dispatch_provenance(tmp_path, monkeypatch, after_body):
    from ouroboros import extension_process_runner
    from ouroboros.tools import extension_dispatch

    registry, _, _ = _registry(tmp_path, monkeypatch)
    calls = []
    def body(**_kwargs):
        calls.append(True)
        raise store.StoreBusy("extension body entered", reason="uninspectable")
    descriptor = {"name": "ext_probe_store", "handler": body, "wants_ctx": False,
                  "content_hash": "c" * 64, "skill": ""}
    monkeypatch.setattr(extension_dispatch, "_extension_dispatch_candidate", lambda *_a: (descriptor, False))
    monkeypatch.setattr(extension_dispatch, "_generation_digest_for", lambda *_a: "d" * 64)
    monkeypatch.setattr(extension_process_runner, "disclose_inprocess_extension_dispatch", lambda *_a, **_k: None)
    actual = owner_pause._validate_launch
    def refuse_on_entry(source, *args, **kwargs):
        # The actual admission runs in its captured caller context, which need
        # not carry the worker's _TOOL_OPERATION; fail this gate, not that proxy.
        if not after_body:
            raise store.StoreBusy("extension admission inspection", reason="uninspectable")
        return actual(source, *args, **kwargs)
    monkeypatch.setattr(owner_pause, "_validate_launch", refuse_on_entry)
    with owner(registry):
        result = run(registry, name="ext_probe_store", args={})["tool_result"]
    assert calls == ([True] if after_body else [])
    if after_body:
        assert result.code == "EXTENSION_ERROR"
        assert result.meta["physical_dispatch"] is True
        assert result.meta["extension_generation"] == "d" * 64
    else:
        assert result.code == "TOOL_ADMISSION_UNAVAILABLE"
        assert result.meta["owner_pause_not_started"] is True
        assert "physical_dispatch" not in result.meta and "extension_generation" not in result.meta


@pytest.mark.parametrize("bound", [False, True])
@pytest.mark.parametrize("after_body", [False, True])
def test_store_refusal_preserves_actual_entry_stage(tmp_path, monkeypatch, bound, after_body):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    calls = []
    original = owner_pause._validate_launch

    def refused(*args, **kwargs):
        raise store.StoreBusy("controlled admission read", reason="permission_denied")

    def body(*_args, **_kwargs):
        calls.append(True)
        raise store.StoreBusy("body already entered", reason="permission_denied")

    registry.override_handler("knowledge_read", body)
    if not after_body:
        monkeypatch.setattr(owner_pause, "_validate_launch", refused)
    if bound:
        with owner(registry):
            result = run(registry)["tool_result"]
    else:
        result = registry.execute_result("knowledge_read", {"topic": "probe"})
    assert calls == ([True] if after_body else [])
    assert result.meta.get("owner_pause_not_started") is (None if after_body else True)
    if not after_body:
        assert result.code == "TOOL_ADMISSION_UNAVAILABLE"
        assert result.meta["launch_reason"] == "permission_denied"
    else:
        assert result.status == "error" and "NOT STARTED" not in result.text
    monkeypatch.setattr(owner_pause, "_validate_launch", original)


@pytest.mark.parametrize("bound", [False, True])
def test_transient_exact_settlement_retries_no_business_work(tmp_path, monkeypatch, bound):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    patch = task_store.patch_handoff
    attempts, body_calls, confirmed_tokens = [], [], []

    def contended(root, task, kind, identity, change):
        def inspect(claim):
            result = change(claim)
            if kind == "launch_handoffs" and result is task_store.DELETE_ROW:
                attempts.append(identity)
                if len(attempts) == 1:
                    raise store.StoreBusy("one settle contention", reason="contention")
            return result
        return patch(root, task, kind, identity, inspect)
    monkeypatch.setattr(task_store, "patch_handoff", contended)

    def body(*_args, **_kwargs):
        body_calls.append(True)
        confirm = tool_custody.external_handoff("fixture-joined-effect")
        confirm()
        confirmed_tokens.append(True)
        return "preserved result"
    registry.override_handler("knowledge_read", body)
    if bound:
        with owner(registry):
            result = run(registry)["tool_result"]
    else:
        result = registry.execute_result("knowledge_read", {"topic": "probe"})
    assert result.text == "preserved result" and body_calls == confirmed_tokens == [True]
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    assert not claims(tmp_path)


@pytest.mark.parametrize("route", ["shell", "mcp"])
@pytest.mark.parametrize("bound", [False, True])
def test_late_refused_handoff_cannot_recreate_claim(tmp_path, monkeypatch, route, bound):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    contexts, opened = [], []

    def body(*_args, **_kwargs):
        contexts.append(copy_context())
        return "joined"
    registry.override_handler("knowledge_read", body)
    if bound:
        with owner(registry):
            assert run(registry)["result"] == "joined"
    else:
        assert registry.execute_result("knowledge_read", {"topic": "probe"}).text == "joined"
    assert not claims(tmp_path)
    if route == "shell":
        from ouroboros.tools.shell_process import _tracked_subprocess_run
        def late():
            return _tracked_subprocess_run([sys.executable, "-c", "pass"], cwd=str(tmp_path),
                                           stdout=-1, stderr=-1, text=True, timeout=2)
    else:
        from ouroboros import mcp_client
        from tests.test_mcp_call_completion import _wire
        opened = _wire(monkeypatch)
        def late():
            return mcp_client.get_manager()._call_tool_result("mcp_demo__effect", {})
    try:
        with pytest.raises(owner_pause.OwnerPauseRefused, match="operation_already_returned"):
            contexts[0].run(late)
        assert not opened and not claims(tmp_path)
    finally:
        if route == "mcp":
            mcp_client.reset_manager_for_tests()


@pytest.mark.parametrize("reason", ["permission_denied", "contention"])
def test_unavailable_settle_preserves_result_and_owes_the_return(tmp_path, monkeypatch, reason):
    from ouroboros import custody_owed

    registry, _, _ = _registry(tmp_path, monkeypatch)
    patch = task_store.patch_handoff
    writes = []
    # A structural finite retry bound, shortened only in this test, not policy.
    monkeypatch.setattr(store, "STORE_WAIT_SEC", 0.01)
    def failed(root, task, kind, identity, change):
        def inspect(claim):
            result = change(claim)
            if result is task_store.DELETE_ROW:
                writes.append(identity)
                raise store.StoreBusy("settlement unavailable", reason=reason)
            return result
        return patch(root, task, kind, identity, inspect)
    monkeypatch.setattr(task_store, "patch_handoff", failed)
    calls = []
    registry.override_handler("knowledge_read", lambda *_a, **_k: calls.append(True) or "kept")
    assert registry.execute_result("knowledge_read", {"topic": "probe"}).text == "kept"
    assert calls == [True] and len(claims(tmp_path)) == 1
    assert len(set(writes)) == 1
    if reason != "contention":
        assert len(writes) == 1
    # The returned call is owed, never forgotten: the same write lands once the store accepts it.
    assert custody_owed.pending() == 1 and custody_owed.drain() == 1
    monkeypatch.setattr(task_store, "patch_handoff", patch)
    assert custody_owed.drain() == 0 and claims(tmp_path) == {}, "no rerun of the tool, no second claim"
    assert calls == [True]


def test_ambiguous_completion_is_idempotent_and_preserves_other_duties(tmp_path, monkeypatch):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    patch = task_store.patch_handoff
    raised, timestamps = [], []
    from ouroboros.task_results import load_task_result
    with owner_pause.tool_handoff(registry._ctx, "probe") as outcome:
        tool_custody.retain_unconfirmed_host_operation("external publisher still pending")
        outcome["settled"] = True
        def uncertain(root, task, kind, identity, change):
            def inspect(claim):
                result = change(claim)
                if isinstance(result, dict) and result.get("state") == "returned":
                    timestamps.append(result["returned_at"])
                return result
            result = patch(root, task, kind, identity, inspect)
            if timestamps and not raised:
                raised.append(True)
                raise store.StoreBusy("COMMIT completed before response was lost", reason="contention")
            return result
        monkeypatch.setattr(task_store, "patch_handoff", uncertain)
    claim = load_task_result(tmp_path, "root")["launch_handoffs"][outcome["operation_id"]]
    assert claim["state"] == "returned" and claim["host_unconfirmed"]
    assert len(timestamps) == 2 and timestamps[0] == timestamps[1]


def test_delayed_real_mcp_runner_cannot_open_transport_after_callers_return(tmp_path, monkeypatch):
    """A real MCPManager -> _run_async copied thread, delayed before its coroutine.

    This is scheduling delay, not another tool or a fabricated returned flag.
    The caller's finite wait ends first; preparation then encounters closure.
    """
    import asyncio
    import threading

    from ouroboros import mcp_client
    from tests.test_mcp_call_completion import _wire
    registry, _, _ = _registry(tmp_path, monkeypatch)
    calls = _wire(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    threads, transports = [], []
    actual_run, actual_join, actual_transport = asyncio.run, threading.Thread.join, mcp_client._transport_factory
    actual_wrapper = mcp_client._run_async

    def delayed(coro, *args, **kwargs):
        if threading.current_thread().name == "mcp-sync-runner":
            threads.append(threading.current_thread())
            entered.set()
            assert release.wait(5)
        return actual_run(coro, *args, **kwargs)
    def joined(thread, timeout=None):
        if thread.name == "mcp-sync-runner":
            assert entered.wait(5)
        return actual_join(thread, timeout)
    def transport(cfg):
        transports.append(True)
        return actual_transport(cfg)
    monkeypatch.setattr(asyncio, "run", delayed)
    monkeypatch.setattr(threading.Thread, "join", joined)
    monkeypatch.setattr(mcp_client, "_transport_factory", transport)
    monkeypatch.setattr(mcp_client, "_run_async", lambda factory, **kw: actual_wrapper(factory, **{**kw, "join_timeout": 0}))
    try:
        async def invoke():
            return registry.execute_result("mcp_demo__effect", {})
        result = actual_run(invoke())
        assert (result.status, result.code) == ("timeout", "MCP_TIMEOUT")
        assert result.meta["mcp_call_not_sent"] is False, "unknown when the caller stopped waiting"
        assert len(claims(tmp_path)) == 1  # real runner has not yet unwound
    finally:
        release.set()
        for thread in threads:
            actual_join(thread, 5)
            assert not thread.is_alive()
        mcp_client.reset_manager_for_tests()
    assert not calls and not transports
    # Only the runner's own positive unwind (preparation refused by closure, nothing
    # sent) discharges the exact duty and ends the local call it held.
    assert not claims(tmp_path)


@pytest.mark.parametrize("bound", [False, True])
def test_body_store_error_still_runs_shell_post_checks(tmp_path, monkeypatch, bound):
    from ouroboros.tools import registry_guard_process

    registry, _, _ = _registry(tmp_path, monkeypatch)
    marker, checked = tmp_path / "body-effect", []
    def body(*_args, _resolved_binding=None, **_kwargs):
        marker.write_text("already executed")
        raise store.StoreBusy("after execution", reason="permission_denied")
    def post_checks(_registry, result, **kwargs):
        checked.append((result, kwargs["tool_name"]))
        return result
    registry.override_handler("run_command", body)
    monkeypatch.setattr(registry_guard_process, "_run_shell_post_checks", post_checks)
    args = {"cmd": ["sh", "-c", "true"]}
    if bound:
        with owner(registry):
            result = run(registry, name="run_command", args=args)["tool_result"]
    else:
        result = registry.execute_result("run_command", args)
    assert marker.exists(), result.text
    assert marker.read_text() == "already executed"
    assert len(checked) == 1 and checked[0][1] == "run_command"
    assert result.status == "error" and "NOT STARTED" not in result.text
    assert result.meta.get("owner_pause_not_started") is not True


def test_postcheck_store_error_keeps_outer_executor_failure(tmp_path, monkeypatch):
    from ouroboros.tools import registry_guard_process

    registry, _, _ = _registry(tmp_path, monkeypatch)
    calls, checked = [], []

    def body(*_args, _resolved_binding=None, **_kwargs):
        calls.append(True)
        return "exit_code=0"

    def post_checks(_registry, result, **kwargs):
        checked.append(kwargs["tool_name"])
        raise store.StoreBusy("post-check unavailable", reason="permission_denied")

    registry.override_handler("run_command", body)
    monkeypatch.setattr(registry_guard_process, "_run_shell_post_checks", post_checks)
    with owner(registry):
        result = run(registry, name="run_command", args={"cmd": ["sh", "-c", "true"]})
    assert calls == [True] and checked == ["run_command"]
    assert result["tool_result"].code == "EXECUTOR_ERROR"
    assert result["tool_result"].meta.get("owner_pause_not_started") is not True


def test_sticky_store_admission_refusal_never_enters_executor_body(tmp_path, monkeypatch):
    import json

    from ouroboros import loop_tool_execution as execution

    registry, _, _ = _registry(tmp_path, monkeypatch)
    entered = []
    registry.override_handler("browser_action", lambda *_a, **_k: entered.append(True) or "done")
    def refuse(*_args, **_kwargs):
        raise store.StoreBusy("sticky pre-entry read", reason="permission_denied")
    monkeypatch.setattr(owner_pause, "_validate_launch", refuse)
    executor = execution.StatefulToolExecutor()
    call = {"id": "sticky-store", "function": {"name": "browser_action",
            "arguments": json.dumps({"action": "snapshot"})}}
    try:
        result = execution._execute_with_timeout(registry, call, tmp_path / "logs", 3, "root", executor)
    finally:
        executor.shutdown()
    assert entered == []
    typed = result["tool_result"]
    assert typed.code == "TOOL_ADMISSION_UNAVAILABLE"
    assert typed.meta["owner_pause_not_started"] is True
    assert typed.meta["launch_reason"] == "permission_denied"
    assert not claims(tmp_path)


@pytest.mark.parametrize("route", ["shell", "mcp"])
@pytest.mark.parametrize("bound", [False, True])
def test_close_before_external_publication_creates_no_phantom_duty(tmp_path, monkeypatch, route, bound):
    """Split real copied-context publication before its durable lock acquisition.

    The old early phase check passed before this barrier and recreated custody
    after closure. New publication tests phase under durable exclusion instead.
    """
    import threading

    from ouroboros import mcp_client
    from tests.test_mcp_call_completion import _wire

    registry, _, _ = _registry(tmp_path, monkeypatch)
    waiting, release = threading.Event(), threading.Event()
    threads, errors, writes, starts = [], [], [], []
    patch = task_store.patch_handoff
    if route == "mcp":
        starts = _wire(monkeypatch)
    else:
        import subprocess
        actual_popen = subprocess.Popen
        def unexpected_spawn(*args, **kwargs):
            if threading.current_thread().name == "late-publisher":
                starts.append(True)
                raise AssertionError("closed publisher spawned a process")
            return actual_popen(*args, **kwargs)  # owner birth observations are not the effect
        monkeypatch.setattr(subprocess, "Popen", unexpected_spawn)

    def observe(root, task, kind, identity, update):
        def change(current):
            result = update(current)
            if threading.current_thread().name == "late-publisher" and isinstance(result, dict):
                writes.append(result)
            return result
        return patch(root, task, kind, identity, change)
    monkeypatch.setattr(task_store, "patch_handoff", observe)

    def body(*_args, **_kwargs):
        outcome = owner_pause._TOOL_OPERATION.get()[2]
        actual_lock = outcome["lock"]
        class DelayedLock:
            delayed = False
            def __enter__(self):
                if threading.current_thread().name == "late-publisher" and not self.delayed:
                    self.delayed = True
                    waiting.set()
                    assert release.wait(5)
                return actual_lock.__enter__()
            def __exit__(self, *args):
                return actual_lock.__exit__(*args)
        outcome["lock"] = DelayedLock()
        context = copy_context()
        def late():
            try:
                if route == "shell":
                    from ouroboros.tools.shell_process import _tracked_subprocess_run
                    _tracked_subprocess_run([sys.executable, "-c", "pass"], cwd=str(tmp_path), timeout=2)
                else:
                    mcp_client.get_manager()._call_tool_result("mcp_demo__effect", {})
            except BaseException as exc:
                errors.append(exc)
        thread = threading.Thread(target=lambda: context.run(late), name="late-publisher")
        threads.append(thread)
        thread.start()
        assert waiting.wait(3)
        return "joined before late publication"
    registry.override_handler("knowledge_read", body)
    try:
        if bound:
            with owner(registry):
                result = run(registry)["tool_result"]
        else:
            result = registry.execute_result("knowledge_read", {"topic": "probe"})
        assert result.text == "joined before late publication"
        assert not claims(tmp_path)
    finally:
        release.set()
        for thread in threads:
            thread.join(5)
            assert not thread.is_alive()
        mcp_client.reset_manager_for_tests()
    assert len(errors) == 1 and isinstance(errors[0], owner_pause.OwnerPauseRefused)
    assert not starts and not writes and not claims(tmp_path)


def test_preclose_external_duty_survives_return_and_exact_late_confirmation(tmp_path, monkeypatch):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    confirmations, contexts = [], []
    def body(*_args, **_kwargs):
        confirmations.append(tool_custody.external_handoff("already admitted effect"))
        contexts.append(copy_context())
        return "local return"
    registry.override_handler("knowledge_read", body)
    assert registry.execute_result("knowledge_read", {"topic": "probe"}).text == "local return"
    before = claims(tmp_path)
    assert len(before) == 1
    assert next(iter(before.values()))["state"] == "returned"
    assert next(iter(before.values()))["external_pending"]
    # Legitimate late updates to old custody are not a new start.
    contexts[0].run(tool_custody.retain_unconfirmed_host_operation, "pending host receipt")
    confirmations[0]()
    after = claims(tmp_path)
    assert len(after) == 1 and next(iter(after.values()))["host_unconfirmed"]
    assert not next(iter(after.values()))["external_pending"]
    confirmations[0]()
    assert claims(tmp_path) == after


def test_confirmation_after_refused_publication_does_not_manufacture_claim(tmp_path, monkeypatch):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    patch = task_store.patch_handoff
    def refuse(root, task, kind, identity, update):
        def change(current):
            result = update(current)
            if isinstance(result, dict) and result.get("external_pending"):
                raise store.StoreBusy("new duty never published", reason="permission_denied")
            return result
        return patch(root, task, kind, identity, change)
    monkeypatch.setattr(task_store, "patch_handoff", refuse)
    def body(*_args, **_kwargs):
        with pytest.raises(store.StoreBusy):
            tool_custody.external_handoff("unpublished duty")
        return "kept"
    registry.override_handler("knowledge_read", body)
    with owner(registry):
        assert run(registry)["result"] == "kept"
    assert not claims(tmp_path)


def test_settlement_does_not_retire_live_model_receiver(tmp_path, monkeypatch):
    import threading

    from ouroboros import usage_accounting as ua
    from ouroboros.model_wait import TaskModelWait, operation_wait_scope
    from supervisor.continuation_admission import conflicting_writers
    from tests._usage_store_testing import ledger_rows

    registry, queue, _ = _registry(tmp_path, monkeypatch)
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    entered, release = threading.Event(), threading.Event()
    consumer = TaskModelWait(task={"id": "root"}, drive_root=tmp_path,
                             event_queue=None, worker_slot_held=True)
    errors = []
    def send():
        entered.set()
        assert release.wait(10)
        raise RuntimeError("controlled unknown model response")
    def receiver():
        try:
            with operation_wait_scope(consumer):
                ua.execute_physical_attempt(ua.AttemptRequest(
                    model="m", provider="test", drive_root=tmp_path,
                    task_id="root", root_task_id="root", reservation_usd=0.5), send)
        except RuntimeError as exc:
            errors.append(str(exc))
    thread = threading.Thread(target=receiver)
    thread.start()
    try:
        assert entered.wait(5)
        before = ledger_rows(tmp_path)
        assert before[-1]["state"] == "dispatched"
        registry.override_handler("knowledge_read", lambda *_a, **_k: "local call joined")
        assert registry.execute_result("knowledge_read", {"topic": "x"}).text == "local call joined"
        assert not claims(tmp_path) and not consumer.closed and thread.is_alive()
        assert ledger_rows(tmp_path) == before
        assert any(row["kind"] == "model_handoff" for row in conflicting_writers(queue, "root"))
    finally:
        release.set()
        thread.join(10)
        consumer.close()
    assert not thread.is_alive() and errors == ["controlled unknown model response"]


def test_presence_keeps_live_sibling_when_other_call_settles(tmp_path, monkeypatch):
    import threading

    from ouroboros.presence_continuation import _custody_blockers

    registry, _, _ = _registry(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    def sibling(*_args, **_kwargs):
        entered.set()
        assert release.wait(10)
        return "sibling joined"
    registry.override_handler("knowledge_read", sibling)
    thread = threading.Thread(target=registry.execute_result, args=("knowledge_read", {"topic": "x"}))
    thread.start()
    try:
        assert entered.wait(5)
        before = claims(tmp_path)
        assert len(before) == 1
        registry.override_handler("knowledge_list", lambda *_a, **_k: "other joined")
        assert registry.execute_result("knowledge_list", {}).text == "other joined"
        assert claims(tmp_path) == before
        assert any(row["kind"] == "tool_handoff" for row in _custody_blockers(tmp_path, "root"))
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive() and not _custody_blockers(tmp_path, "root")
