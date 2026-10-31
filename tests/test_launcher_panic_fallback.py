"""Actual tray/desktop callers: disposable children, no live server or control port."""
import json
import pathlib
import subprocess
import sys
import threading

import pytest


@pytest.mark.parametrize('server_mode', ['hung', 'unavailable'])
@pytest.mark.parametrize('surface', ['tray', 'desktop_bridge'])
def test_real_launcher_stop_and_next_boot_keep_panic_authority(tmp_path, server_mode, surface):
    from ouroboros.test_environment import isolated_environment
    from ouroboros.platform_layer import pid_is_alive
    repo = pathlib.Path(__file__).resolve().parents[1]
    env = isolated_environment(tmp_path, repo)
    root = pathlib.Path(env['OUROBOROS_DATA_DIR'])
    command = [sys.executable, '-m', 'tests.test_launcher_panic_fallback', server_mode, surface]
    output = root / 'launcher-test.log'
    try:
        with output.open('w') as log:
            result = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=log, timeout=35)
        assert result.returncode == 99, output.read_text()
        assert (root / 'state/panic_stop.flag').read_bytes() == b'panic'
        receipts = json.loads((root / 'known-children.json').read_text())
        assert all(not pid_is_alive(row['pid']) for row in receipts), output.read_text() + str(receipts)
        source = json.loads((root / 'state/panic_source.json').read_text())
        assert source['ingress'] == ('launcher_tray' if surface == 'tray' else surface)
        assert source['owner_surface'] == 'owner' and source['server_pid'] is None
    finally:
        from ouroboros.install_identity import acting_as
        from ouroboros.process_custody import stop_ledgered_processes
        with acting_as(root):  # this test launched the isolated installation
            stop_ledgered_processes(root, {'test_hung_server', 'test_work'}, timeout_sec=2)
    # A new isolated interpreter exercises the real boot consumer, never server.main.
    boot = subprocess.run([sys.executable, '-m', 'tests.test_launcher_panic_fallback', 'next-boot'],
                          cwd=repo, env=env, capture_output=True, text=True, timeout=20)
    assert boot.returncode == 0, boot.stdout + boot.stderr
    assert 'BOOT_CONTROLS_OFF_NO_RETURN' in boot.stdout


def _exercise(mode, surface="tray"):
    import os
    from ouroboros.config import DATA_DIR
    from supervisor import state, queue, workers
    root = pathlib.Path(DATA_DIR)
    state.init(root)
    if mode == 'next-boot':
        queue.init(root)
        workers.DRIVE_ROOT = root
        workers.PENDING, workers.RUNNING = [], {}
        queue.init_queue_refs(workers.PENDING, workers.RUNNING, {'value': 0})
        from supervisor.worker_chat_lane import auto_resume_after_restart
        auto_resume_after_restart()
        saved = state.load_state()
        assert saved['bg_consciousness_enabled'] is False
        assert saved['evolution_mode_enabled'] is False
        assert saved['evolution_owner_stopped'] is True
        assert not workers.PENDING
        assert not (root / 'state/panic_stop.flag').exists()
        print('BOOT_CONTROLS_OFF_NO_RETURN')
        return
    state.init_state()
    state.update_state(lambda st: st.update(bg_consciousness_enabled=True, evolution_mode_enabled=True))
    from ouroboros.process_custody import record_process, spawn_supervised
    from ouroboros.launcher_bootstrap import spawn_bootstrap_command
    from ouroboros.platform_layer import process_start_time
    from ouroboros.process_scope import plant_installation_scope
    plant_installation_scope(root)
    # Held listening child and worker are known Popen objects, never a port/PID search.
    code = "import socket,time; s=socket.socket(); s.bind(('127.0.0.1',0)); s.listen(); time.sleep(60)"
    server = spawn_supervised([sys.executable, '-I', '-S', '-c', code],
                              drive_root=root, purpose='test_hung_server', scope='session',
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutdown = threading.Event()
    work_command = [sys.executable, '-I', '-S', '-c', 'import time; time.sleep(60)']
    # Actual launcher-owned work has a captured bootstrap owner. A generic cold
    # ledger row alone belongs to the bounded census and cannot prove exit here.
    work = spawn_bootstrap_command(work_command, shutdown,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)['proc']
    record_process(root, pid=work.pid, cmd=work_command, purpose='test_work',
                   scope='task', owner_task_id='fixture-work')  # parent fixture cleanup on failure
    (root / 'known-children.json').write_text(json.dumps([
        {'pid': child.pid, 'birth': process_start_time(child.pid)} for child in (server, work)]))
    if mode == 'unavailable':
        server.kill()
        server.wait(timeout=5)
    from ouroboros import launcher_background as background
    background.indicator_class = lambda: None
    def unexpected():
        raise AssertionError('no ordinary exit or mutable port lookup')
    bg = background.Background(unexpected, unexpected, shutdown, panic_child=lambda: (server, None))
    assert not hasattr(bg, 'panic_fallback')
    if surface == 'tray':
        bg.request_panic()
    else:
        api = background.DesktopApi()
        api._background = bg
        assert api.panic_stop() == {'status': 'stopping'}
    assert bg.shutdown.is_set() and bg.quit.is_set()
    # The real non-daemon emergency owner must persist and hard-exit this interpreter.
    threading.Event().wait(20)
    os._exit(1)


if __name__ == '__main__':
    _exercise(*sys.argv[1:])
