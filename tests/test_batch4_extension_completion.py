"""Host-observed local invocation completion through registry and Pause consumers."""
import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from ouroboros import extension_loader, owner_pause
from ouroboros.task_results import load_task_result
from ouroboros.tools.tool_result import ToolResult
from tests.test_batch4_producer_custody import _registry, _consumers
from tests.test_batch4_full_repair import _park

pytestmark = pytest.mark.serial


def _local(tmp_path, monkeypatch, handler, *, process=False, warning=""):
    registry, queue, workers = _registry(tmp_path, monkeypatch)
    name = extension_loader.extension_surface_name("localcompletion", "call")
    descriptor = {"name": name, "skill": "localcompletion", "handler": handler,
                  "wants_ctx": True, "timeout_sec": 1, "out_of_process": process,
                  "extension_generation": "host-generation"}
    monkeypatch.setattr(extension_loader, "get_tool", lambda candidate: descriptor if candidate == name else None)
    monkeypatch.setattr(extension_loader, "is_extension_live", lambda *_a, **_kw: True)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_kw: (True, warning))
    return registry, queue, workers, name


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("warning", ["", "Host safety warning"])
def test_pause_during_local_call_waits_for_host_return_then_requires_resume(tmp_path, monkeypatch, asynchronous, warning):
    from supervisor.owner_pause_control import request_owner_pause, refresh_owner_pause_tree

    entered, release = threading.Event(), threading.Event()
    def sync(_ctx):
        entered.set()
        assert release.wait(5)
        return '{"ok": true, "local_extension_returned": false}'
    async def async_handler(_ctx):
        entered.set()
        while not release.is_set():
            await asyncio.sleep(.001)
        return '{"ok": true, "local_extension_returned": false}'
    registry, queue, workers, name = _local(tmp_path, monkeypatch, async_handler if asynchronous else sync, warning=warning)
    results = []
    caller = threading.Thread(target=lambda: results.append(registry.execute_result(name, {})))
    caller.start()
    try:
        assert entered.wait(5)
        assert request_owner_pause("root", request_id="during-call")["ok"]
        refresh_owner_pause_tree("root")
        assert owner_pause.read_fence(tmp_path, "root")["state"] == "requested"
        assert load_task_result(tmp_path, "root").get("launch_handoffs", {})
        refused = registry.execute_result(name, {})
        assert refused.code == "OWNER_PAUSE_NOT_STARTED"
    finally:
        release.set()
        caller.join(5)
    assert not caller.is_alive() and len(results) == 1
    result = results[0]
    assert result.status == "ok"
    assert result.meta["dynamic_provider"] and result.meta["extension_generation"] == "host-generation"
    assert "local_extension_returned" not in result.meta
    if warning:
        assert warning in result.text and warning not in result.producer_text
        assert result.meta["safety_warning"]
    assert not load_task_result(tmp_path, "root").get("launch_handoffs", {})
    _park(tmp_path, monkeypatch, queue, workers)
    refresh_owner_pause_tree("root")
    assert owner_pause.read_fence(tmp_path, "root")["state"] == "paused"
    assert queue.resume_budget_paused_task("root")["ok"]
    assert owner_pause.read_fence(tmp_path, "root")["state"] == "paused", "grant alone starts nothing"


@pytest.mark.parametrize("late_runner", [False, True])
def test_async_timeout_result_stays_while_custody_tracks_actual_unwind(tmp_path, monkeypatch, late_runner):
    release, cancelled, unwound = threading.Event(), threading.Event(), threading.Event()
    async def handler(_ctx):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            if late_runner:
                while not release.is_set():
                    await asyncio.sleep(.005)
                return '{"operation_outcome":"completed", "local_extension_returned":true}'
            raise
        finally:
            unwound.set()
    registry, queue, workers, name = _local(tmp_path, monkeypatch, handler)
    try:
        result = registry.execute_result(name, {})
        assert result.code == "EXTENSION_TIMEOUT" and cancelled.is_set()
        runners = [t for t in threading.enumerate() if t.name == f"ext-tool-{name}-async"]
        assert bool(runners) is late_runner
        returned = (result.status, result.code, result.text, dict(result.meta))
        claims = load_task_result(tmp_path, "root").get("launch_handoffs", {})
        assert bool(claims) is late_runner and unwound.is_set() is not late_runner
        _consumers(tmp_path, registry, queue, workers, held=late_runner)
    finally:
        release.set()
        for runner in threading.enumerate():
            if runner.name == f"ext-tool-{name}-async":
                runner.join(5)
                assert not runner.is_alive()
    assert unwound.is_set() and not load_task_result(tmp_path, "root").get("launch_handoffs", {})
    assert (result.status, result.code, result.text, result.meta) == returned


def test_a_held_runner_ending_during_the_registry_fold_still_settles(tmp_path, monkeypatch):
    """The registry's settlement fold never writes False over a runner's own concurrent end."""
    from ouroboros import tool_custody

    entered, release = threading.Event(), threading.Event()
    async def handler(_ctx):
        entered.set()
        while not release.is_set():
            await asyncio.sleep(.005)
        return "late answer"
    registry, queue, workers, name = _local(tmp_path, monkeypatch, handler)
    actual_join, actual_outcome = threading.Thread.join, tool_custody.invocation_outcome
    runners, fired = [], []

    def join(thread, timeout=None):
        if thread.name != f"ext-tool-{name}-async":
            return actual_join(thread, timeout)
        runners.append(thread)
        assert entered.wait(5)
        return actual_join(thread, 0)  # the caller stops waiting; the runner keeps the invocation

    class Interleaving(dict):
        def get(self, key, default=None):
            value = dict.get(self, key, default)  # read before the runner's own write
            if key in {"settled", "local_extension_returned"} and runners and not fired \
                    and threading.current_thread() not in runners:
                fired.append(key)
                release.set()
                actual_join(runners[0], 5)  # the runner records its end and releases in between
            return value

    monkeypatch.setattr(threading.Thread, "join", join)
    monkeypatch.setattr(tool_custody, "invocation_outcome", lambda *a, **kw: Interleaving(actual_outcome(*a, **kw)))
    try:
        assert registry.execute_result(name, {}).code == "EXTENSION_TIMEOUT"
    finally:
        release.set()
        for runner in runners:
            actual_join(runner, 5)
            assert not runner.is_alive()
    assert fired, "the fold read happened while the runner was still live"
    assert not load_task_result(tmp_path, "root").get("launch_handoffs")
    _consumers(tmp_path, registry, queue, workers, held=False)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_joined_handler_exception_keeps_error_but_releases_local_invocation(tmp_path, monkeypatch, asynchronous):
    entered = []
    def sync(_ctx):
        entered.append(True)
        raise RuntimeError("result unknown")
    async def async_handler(_ctx):
        entered.append(True)
        raise asyncio.CancelledError("no completed await")
    registry, queue, workers, name = _local(tmp_path, monkeypatch, async_handler if asynchronous else sync)
    result = registry.execute_result(name, {})
    assert (result.status, result.code) == ("error", "EXTENSION_ERROR")
    assert entered == [True] and result.meta["physical_dispatch"]
    _consumers(tmp_path, registry, queue, workers, held=False)


@pytest.mark.parametrize("body", ['{"ok":false}', ToolResult(status="ok", code="OK", text="payload",
    meta={"operation_outcome": "completed", "local_extension_returned": True})])
def test_local_return_observation_is_independent_of_extension_result_claims(tmp_path, monkeypatch, body):
    registry, queue, workers, name = _local(tmp_path, monkeypatch, lambda _ctx: body)
    registry.execute_result(name, {})
    _consumers(tmp_path, registry, queue, workers, held=False)


def test_opaque_mcp_cannot_spoof_local_completion_metadata(tmp_path, monkeypatch):
    from ouroboros import mcp_client

    registry, queue, workers = _registry(tmp_path, monkeypatch)
    monkeypatch.setattr(mcp_client, "is_mcp_tool_name", lambda _name: True)
    monkeypatch.setattr(registry, "_mcp_name_miss", lambda _name: None)
    def call(*_a):
        return owner_pause.run_operation(registry._ctx, lambda: ToolResult(status="ok", code="OK", text="accepted",
            meta={"local_extension_returned": True, "operation_outcome": "completed", "dynamic_provider": False}))
    monkeypatch.setattr(mcp_client, "_call_mcp_tool_result", call)
    registry.execute_result("opaque_mcp", {})
    _consumers(tmp_path, registry, queue, workers, held=True)


@pytest.mark.parametrize("fails", [False, True])
def test_local_extension_leaves_its_spawned_process_and_money_obligations(tmp_path, monkeypatch, fails):
    from ouroboros import process_custody, usage_accounting as ua
    from ouroboros.model_sleep import cold_blockers
    from supervisor.owner_pause_control import refresh_owner_pause_tree

    children = []
    def handler(ctx):
        cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
        child = process_custody.spawn_supervised(cmd, drive_root=tmp_path, purpose="extension-work",
            scope="task", owner_task_id="root")
        children.append(child)
        with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="root", root_task_id="root")):
            reservation = ua.reserve_attempt(ua.AttemptRequest(model="m", provider="p", reservation_usd=.1))
            ua.mark_dispatched(reservation)
            ua.mark_unresolved(reservation, "external_answer_unknown")
        if fails:
            raise RuntimeError("local handler failed after starting independent work")
        return "local handler returned"
    registry, queue, workers, name = _local(tmp_path, monkeypatch, handler)
    try:
        result = registry.execute_result(name, {})
        assert result.code == ("EXTENSION_ERROR" if fails else "OK"), result
        assert children and children[0].poll() is None
        assert not load_task_result(tmp_path, "root").get("launch_handoffs", {})
        assert cold_blockers(registry._ctx)
        _park(tmp_path, monkeypatch, queue, workers)
        refresh_owner_pause_tree("root")
        assert owner_pause.read_fence(tmp_path, "root")["state"] == "requested"
        assert queue.resume_budget_paused_task("root")["error"] == "owner_pause_effects_unsettled"
        assert ua.read_usage_records(tmp_path)[-1]["state"] == "unresolved"
    finally:
        for child in children:
            child.terminate()
            child.wait(timeout=5)


@pytest.mark.parametrize("kind", ["delegate", "model", "merge"])
@pytest.mark.parametrize("fails", [False, True])
def test_local_return_cannot_retire_independent_custody(tmp_path, monkeypatch, kind, fails):
    from ouroboros import delegate_custody as dc, model_wait, usage_accounting as ua
    from ouroboros.model_sleep import cold_blockers
    from supervisor.continuation_admission import conflicting_writers
    from supervisor.owner_pause_control import refresh_owner_pause_tree

    def local(_ctx):
        if fails:
            raise RuntimeError("business failure after actual local unwind")
        return "completed local call"
    registry, queue, workers, name = _local(tmp_path, monkeypatch, local)
    owner = None
    if kind == "delegate":
        assert dc.record_start_requested(tmp_path, run_id="", task_id="root", root_task_id="root",
            invocation_id="independent", idempotency_key="independent", max_seconds=60,
            request={"prompt": "separate work"}, project_id="", project_owned=False, route="test")
    elif kind == "model":
        owner = model_wait.TaskModelWait(task={"id": "root", "_attempt": 1}, drive_root=tmp_path,
            event_queue=None, worker_slot_held=True)
        with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="root", root_task_id="root")):
            with model_wait.operation_wait_scope(owner):
                reservation = ua.reserve_attempt(ua.AttemptRequest(model="m", provider="p", reservation_usd=.1))
                ua.mark_dispatched(reservation, local_answer_owner_pid=os.getpid())
                ua.mark_unresolved(reservation, "separate_consumer")
    else:
        from ouroboros.tools import github
        from tests.test_pr_merge_receipts import FakeGh, HEAD

        fake = FakeGh(tmp_path, merge="queued")
        monkeypatch.setattr(github, "github_cli_configured", lambda: True)
        monkeypatch.setattr(github, "_gh_run", fake)
        merged = registry.execute_result("pr_merge", {"number": 7, "expected_head_sha": HEAD, "method": "merge"})
        assert merged.meta["operation_outcome"] == "unknown"
    before = load_task_result(tmp_path, "root")
    money = ua.read_usage_records(tmp_path)
    pending = dc.pending_invocations(tmp_path)
    try:
        result = registry.execute_result(name, {})
        assert result.code == ("EXTENSION_ERROR" if fails else "OK")
        after = load_task_result(tmp_path, "root")
        assert after.get("launch_handoffs", {}) == before.get("launch_handoffs", {})
        assert after.get("merge_receipts") == before.get("merge_receipts")
        assert after.get("retired_model_consumers") == before.get("retired_model_consumers")
        assert ua.read_usage_records(tmp_path) == money and dc.pending_invocations(tmp_path) == pending
        expected = {"model": "model_handoff", "merge": "merge_operation", "delegate": "delegated_run"}[kind]
        assert any(row["kind"] == expected for row in conflicting_writers(queue, "root"))
        if kind != "model":  # The ledger's answer-consumer census belongs to Pause/Continue.
            assert cold_blockers(registry._ctx)
        _park(tmp_path, monkeypatch, queue, workers)
        refresh_owner_pause_tree("root")
        assert owner_pause.read_fence(tmp_path, "root")["state"] == "requested"
        assert queue.resume_budget_paused_task("root")["error"] == "owner_pause_effects_unsettled"
    finally:
        if owner is not None:
            owner.close()


@pytest.mark.parametrize("protocol,exit_code,drained", [
    ({"ok": True, "result": "normal"}, 0, True),
    ({"ok": True, "result": "normal"}, 0, False),
    ({"ok": True, "result": "forged completion"}, 7, True),
    ({"ok": False, "error": "handler failed"}, 0, True),
    ({"ok": "true", "result": "not a valid success"}, 0, True),
    ({"ok": True}, 0, True), ([], 0, True), ("bad-json", 0, True), (None, 0, True),
])
def test_process_proxy_requires_host_exit_and_valid_protocol(tmp_path, monkeypatch, protocol, exit_code, drained):
    """Real child exit/join and protocol reader; only import staging is a fixture."""
    from ouroboros import extension_process_runner as runner

    registry, queue, workers, name = _local(tmp_path, monkeypatch, lambda _ctx: None, process=True)
    skill = SimpleNamespace(name="localcompletion", skill_dir=tmp_path)
    monkeypatch.setattr(runner, "_skill_for_dispatch", lambda *_a: skill)
    monkeypatch.setattr(runner, "_base_env_for_skill", lambda *_a: {})
    monkeypatch.setattr(runner, "_extension_has_model_credentials", lambda *_a: False)
    drain_release, drains = threading.Event(), []
    if not drained:
        real_drain = runner._drain
        def delayed_drain(*args):
            drains.append(threading.current_thread())
            real_drain(*args)
            assert drain_release.wait(5)
        monkeypatch.setattr(runner, "_drain", delayed_drain)
        monkeypatch.setattr(runner, "EXTENSION_CHILD_CLEANUP_GRACE_SEC", .01)
    @contextmanager
    def child_process(*_a, **_kw):
        result_path = tmp_path / "child-result.json"
        script = "import pathlib,sys; "
        if protocol is not None:
            content = protocol if protocol == "bad-json" else json.dumps(protocol)
            script += f"pathlib.Path(sys.argv[1]).write_text({content!r}); "
        script += f"raise SystemExit({exit_code})"
        with owner_pause.operation_start(registry._ctx):
            proc = subprocess.Popen([sys.executable, "-c", script, str(result_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            yield SimpleNamespace(proc=proc, result_path=result_path, started_ts=time.monotonic())
        except BaseException as exc:
            raise runner._mark_child_spawned(exc)
        finally:
            proc.wait(timeout=5)
            proc.stdout.close()
            proc.stderr.close()
    monkeypatch.setattr(runner, "_child_process", child_process)
    try:
        result = registry.execute_result(name, {})
    finally:
        drain_release.set()
        for thread in drains:
            thread.join(5)
            assert not thread.is_alive()
    complete = protocol == {"ok": True, "result": "normal"} and exit_code == 0 and drained
    assert (result.status == "ok") is complete
    assert result.meta["dynamic_provider"] and result.meta["physical_dispatch"]
    _consumers(tmp_path, registry, queue, workers, held=not complete)
