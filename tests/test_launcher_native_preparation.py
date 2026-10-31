"""Native preparation owns its clock and teardown before server readiness starts."""
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.serial


@pytest.fixture
def lifecycle(monkeypatch, tmp_path):
    import launcher

    shutdown = threading.Event()
    monkeypatch.setattr(launcher, '_shutdown_event', shutdown)
    monkeypatch.setattr(launcher, '_agent_proc', None)
    monkeypatch.setattr(launcher, '_agent_job', None)
    monkeypatch.setattr(launcher, '_external_host_update', tmp_path / 'native-hook')
    monkeypatch.setattr(launcher, '_external_host_result', {})
    monkeypatch.setattr(launcher, 'PORT_FILE', tmp_path / 'port')
    monkeypatch.setattr(launcher, '_pre_generation_cleanup', lambda: [])
    monkeypatch.setattr(launcher, '_await_generation_endpoint', lambda *a, **kw: 1)
    monkeypatch.setattr(launcher, '_update_server_process_record_port', lambda *a: None)
    monkeypatch.setattr(launcher, '_cleanup_recorded_server_group_for_pid', lambda *a: None)
    monkeypatch.setattr(launcher, '_generation_port', None)
    yield launcher, shutdown
    shutdown.set()


def test_preparation_finishes_before_any_http_readiness_probe(lifecycle, monkeypatch):
    launcher, shutdown = lifecycle
    entered, release, spawned = threading.Event(), threading.Event(), threading.Event()
    calls, result = [], []

    def prepare(*args):
        entered.set()
        assert release.wait(5)
        return {'status': 'verified'}

    def start(port):
        process = SimpleNamespace(pid=123, returncode=0, wait=lambda: shutdown.wait(5))
        launcher._agent_proc = process
        calls.append('spawn')
        spawned.set()
        return process

    def health(port, timeout=None, abort_event=None, proc=None):
        assert spawned.is_set(), 'HTTP readiness probing started before native preparation ended'
        calls.append(('http', timeout))
        return True

    monkeypatch.setattr(launcher, 'update_external_host', prepare)
    monkeypatch.setattr(launcher, 'start_agent', start)
    monkeypatch.setattr(launcher, '_wait_for_server', health)
    thread = threading.Thread(target=launcher.agent_lifecycle_loop)
    waiter = threading.Thread(target=lambda: result.append(launcher._await_server_ready(1, shutdown, thread)))
    thread.start()
    try:
        assert entered.wait(2)
        waiter.start()
        # This controlled interval exceeds an immediate fake HTTP-budget outcome:
        # old code calls health now and fails before native preparation is released.
        time.sleep(0.1)
        assert waiter.is_alive() and not calls and not result
        release.set()
        waiter.join(3)
        assert result == [(True, 1)]
        assert calls[0] == 'spawn' and ('http', None) in calls, 'the launcher probes its own child without a clock'
    finally:
        release.set(); shutdown.set()
        if waiter.ident is not None:
            waiter.join(3)
        thread.join(3)
    assert not thread.is_alive() and not waiter.is_alive()


def test_preparation_failure_still_starts_core_with_failed_proof(lifecycle, monkeypatch, tmp_path):
    launcher, shutdown = lifecycle
    hook = tmp_path / 'failed.py'
    hook.write_text('raise SystemExit(7)\n')
    monkeypatch.setattr(launcher, '_external_host_update', hook)
    monkeypatch.setattr(launcher, 'EMBEDDED_PYTHON', sys.executable)
    seen = []

    def start(port):
        seen.append(dict(launcher._external_host_result))
        shutdown.set()
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(launcher, 'start_agent', start)
    monkeypatch.setattr(launcher, 'stop_agent', lambda: None)
    launcher.agent_lifecycle_loop()
    assert seen == [{'status': 'failed'}]


def test_preparation_shutdown_reaps_actual_detached_child_before_launcher_exit(tmp_path, monkeypatch):
    from ouroboros.launcher_bootstrap import update_external_host
    from ouroboros.process_containment import pid_is_zombie
    from ouroboros.platform_layer import pid_is_alive
    import logging
    import launcher

    child_file = tmp_path / 'child.pid'
    hook = tmp_path / 'native.py'
    hook.write_text(
        'import pathlib, subprocess, sys, time\n'
        'child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], '
        'start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n'
        f'pending = pathlib.Path({str(child_file.with_suffix(".pending"))!r})\n'
        'pending.write_text(str(child.pid))\n'
        f'pending.replace({str(child_file)!r})\n'
        'time.sleep(60)\n'
    )
    shutdown = threading.Event()
    result = []
    thread = threading.Thread(target=lambda: result.append(update_external_host(
        hook, sys.executable, logging.getLogger('native-preparation-test'), shutdown)))
    thread.start()
    child = None
    try:
        deadline = time.monotonic() + 5
        while not child_file.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert child_file.exists()
        child = int(child_file.read_text(encoding="utf-8"))
        assert pid_is_alive(child)
        shutdown.set()
        monkeypatch.setattr(launcher, '_shutdown_event', shutdown)
        monkeypatch.setattr(launcher, '_external_host_update', hook)
        monkeypatch.setattr(launcher, 'stop_agent', lambda: None)
        def final_cleanup(*args, **kwargs):
            assert not thread.is_alive(), 'Launcher exited before native owned cleanup'
        monkeypatch.setattr(launcher, '_kill_orphaned_children', final_cleanup)
        with pytest.raises(SystemExit) as stopped:
            launcher._run_headless_main('http://127.0.0.1:1', 1, thread)
        assert stopped.value.code == 0
        assert not thread.is_alive()
        assert result == [{'status': 'cancelled'}]
        assert not pid_is_alive(child) or pid_is_zombie(child)
    finally:
        shutdown.set(); thread.join(20)
        if child and pid_is_alive(child) and not pid_is_zombie(child):
            # Fixture cleanup only, never a product path fallback.
            import os, signal
            os.kill(child, signal.SIGKILL)


def test_native_budget_stays_total_across_shutdown_observation_slices(monkeypatch, tmp_path):
    import logging
    from ouroboros import launcher_bootstrap as bootstrap, config

    ticks = iter([0.0, 0.0, 0.2, 0.4, 0.6])
    # Keep the finite test clock out of asyncio's Windows Proactor cleanup.
    monkeypatch.setattr(bootstrap, 'time', SimpleNamespace(monotonic=lambda: next(ticks)))
    monkeypatch.setattr(config, 'EXTERNAL_PLATFORM_UPDATE_TIMEOUT_SEC', 0.5)
    waits, cleanup = [], []
    def communicate(timeout):
        waits.append(timeout)
        raise subprocess.TimeoutExpired(['fixture'], timeout)
    proc = SimpleNamespace(args=['fixture'], stdout=None, stderr=None,
                           communicate=communicate, wait=lambda **kw: None)
    container = SimpleNamespace(spawn=lambda *a, **kw: proc,
                                reap=lambda: cleanup.append('reap') or '',
                                close=lambda: cleanup.append('close'))
    monkeypatch.setattr('ouroboros.process_containment.ProcessContainer', lambda: container)
    result = bootstrap.update_external_host(tmp_path/'hook', sys.executable, logging.getLogger('test'), threading.Event())
    assert result == {'status': 'failed'}
    assert waits == pytest.approx([0.2, 0.2, 0.1]) and cleanup == ['reap', 'close']


def test_no_native_hook_reports_the_port_the_generation_proved(lifecycle, monkeypatch):
    """The UI waits for the lifecycle thread's proof, never a guessed port or a stale file."""
    launcher, shutdown = lifecycle
    monkeypatch.setattr(launcher, '_external_host_update', None)
    monkeypatch.setattr(launcher, '_wait_for_server', lambda *a, **kw: pytest.fail('the UI wait probes no port'))
    monkeypatch.setattr(launcher, '_generation_port', 9876)
    assert launcher._await_server_ready(1, shutdown, SimpleNamespace(is_alive=lambda: True)) == (True, 9876)
    monkeypatch.setattr(launcher, '_generation_port', None)
    assert launcher._await_server_ready(1, shutdown, SimpleNamespace(is_alive=lambda: False)) == (False, 1)
