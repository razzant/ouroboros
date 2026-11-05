"""Host-observed local invocation completion through registry and Pause consumers."""
import asyncio
import json
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from ouroboros import extension_loader, owner_pause
from ouroboros.task_results import load_task_result
from tests.test_batch4_producer_custody import _registry, _consumers
from tests.test_batch4_full_repair import _park



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
