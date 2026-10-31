"""Early tray Panic in a cold interpreter must reach stop owners and its active bootstrap tree."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import launcher_bootstrap as boot

REPO = Path(__file__).resolve().parents[1]


def test_bootstrap_run_keeps_subprocess_result_and_timeout_contract():
    stopped = threading.Event()
    result = boot.run_bootstrap_command([sys.executable, '-c', 'import sys; print(sys.stdin.read())'],
                                        stopped, input='hello', capture_output=True, text=True, check=True)
    assert result.returncode == 0 and result.stdout == 'hello\n' and result.stderr == ''
    with pytest.raises(subprocess.CalledProcessError) as failed:
        boot.run_bootstrap_command([sys.executable, '-c', 'raise SystemExit(3)'], stopped, check=True)
    assert failed.value.returncode == 3
    with pytest.raises(subprocess.TimeoutExpired):
        boot.run_bootstrap_command([sys.executable, '-c', 'import time; time.sleep(3)'], stopped, timeout=0.05)
    assert boot._BOOTSTRAP_COMMANDS == []
    stopped.set()
    with pytest.raises(InterruptedError):
        boot.run_bootstrap_command([sys.executable, '-c', 'raise SystemExit(0)'], stopped)


def test_cold_launcher_panic_stops_real_bootstrap_root_and_child(tmp_path):
    # No HTTP endpoint, real server, account or installed data is involved. The
    # harmless worker and grandchild can only write delayed files in tmp_path.
    late, grand_late = tmp_path / 'late', tmp_path / 'grand-late'
    started, grand_started = tmp_path / 'started', tmp_path / 'grand-started'
    grand = ('import pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text("started"); '
             'time.sleep(3); pathlib.Path(sys.argv[2]).write_text("survived")')
    worker = ('import pathlib,subprocess,sys,time; '
              f'subprocess.Popen([sys.executable,"-c",{grand!r},sys.argv[2],sys.argv[3]]); '
              'pathlib.Path(sys.argv[1]).write_text("started"); time.sleep(3); '
              'pathlib.Path(sys.argv[4]).write_text("survived")')
    receipt = tmp_path / 'panic.json'
    script = f'''
import json,os,pathlib,sys,threading,time
import launcher
from ouroboros import launcher_background as bg, server_control as sc
bg.indicator_class = lambda: None
# Persistence of controls is tested separately; do not open a runtime database.
sc._persist_panic_controls = lambda root: None
real_exit = os._exit
class Log:
    def critical(self, message, *args):
        pathlib.Path({str(receipt)!r}).write_text(json.dumps(args[0]))
sc.log = Log()
original = sc.execute_panic_stop
def panic(*args, **kwargs):
    kwargs['log'] = Log()
    return original(*args, **kwargs)
sc.execute_panic_stop = panic
launcher._shutdown_event = threading.Event()
def work():
    try:
        launcher._hidden_run([sys.executable,'-c',{worker!r},{str(started)!r},{str(grand_started)!r},
                              {str(grand_late)!r},{str(late)!r}], capture_output=True)
    except InterruptedError:
        pass
threading.Thread(target=work, daemon=True).start()
end = time.monotonic() + 5
while not (pathlib.Path({str(started)!r}).exists() and pathlib.Path({str(grand_started)!r}).exists()):
    if time.monotonic() > end: raise RuntimeError('bootstrap never started')
    time.sleep(.01)
bg.Background(lambda: real_exit(7), lambda: 1, launcher._shutdown_event).request_panic()
threading.Event().wait(10)
raise RuntimeError('Panic never exited')
'''
    env = {**os.environ, 'OUROBOROS_DATA_DIR': str(tmp_path / 'data'),
           'OUROBOROS_SETTINGS_PATH': str(tmp_path / 'data/settings.json'), 'PYTHONPATH': str(REPO)}
    result = subprocess.run([sys.executable, '-c', script], cwd=REPO, env=env,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 99, result.stdout + result.stderr
    requests = json.loads(receipt.read_text())
    assert 'executor' not in requests, requests  # cold import errors must not skip the remaining owners
    assert {'commands', 'services', 'retained-daemon', 'scope-census'} <= requests.keys(), requests
    time.sleep(3.2)
    assert not late.exists() and not grand_late.exists(), 'bootstrap work survived the launcher Panic'
    assert (tmp_path / 'data/state/panic_stop.flag').read_text() == 'panic'


@pytest.mark.parametrize('windows', [False, True])
def test_interactive_installer_keeps_ordinary_terminal_and_only_panic_requests_its_stop(monkeypatch, windows):
    from ouroboros import platform_layer

    calls, stopped = [], threading.Event()
    monkeypatch.setattr(platform_layer, 'IS_WINDOWS', windows)
    proc = SimpleNamespace(pid=41)
    monkeypatch.setattr(boot, '_BOOTSTRAP_COMMANDS', [])
    monkeypatch.setattr(boot.subprocess, 'Popen', lambda command, **kw: calls.append((command, kw)) or proc)
    monkeypatch.setattr('ouroboros.platform_layer.request_process_tree_kill',
                        lambda child, **kw: calls.append(('stop', child, kw)) or {'requested': True})
    owner = boot.spawn_bootstrap_command(['installer'], stopped, interactive=True)
    assert owner['container'] is None and owner['proc'] is proc
    expected = {'creationflags': 0x08000000} if windows else {}
    assert calls == [(['installer'], expected)], 'hide the Windows console without session/group detachment or a Job'
    assert boot.stop_bootstrap_commands(include_interactive=False) == []
    assert boot.stop_bootstrap_commands() == [{'requested': True}]
    assert calls[-1] == ('stop', proc, {'job_handle': None})


@pytest.mark.parametrize('frozen', [False, True])
def test_bootstrap_runs_when_host_executable_is_a_frozen_app(monkeypatch, tmp_path, frozen):
    from ouroboros import process_containment

    python = sys.executable
    actual_container = process_containment.ProcessContainer
    lifelines = []

    def container(**kwargs):
        if 'lifeline' in kwargs:
            lifelines.append(kwargs['lifeline'])
        return actual_container(**kwargs)

    monkeypatch.setattr(process_containment, 'ProcessContainer', container)
    monkeypatch.setattr(sys, 'frozen', frozen, raising=False)
    if frozen:
        # A frozen app is not a Python CLI. Never execute it as the guardian.
        monkeypatch.setattr(sys, 'executable', str(tmp_path / 'not-a-python-cli'))
    result = boot.run_bootstrap_command([python, '-c', 'print("bootstrap completed")'],
                                        threading.Event(), capture_output=True, text=True)
    assert result.returncode == 0 and result.stdout == 'bootstrap completed\n'
    assert lifelines[0] is not frozen
    assert boot._BOOTSTRAP_COMMANDS == []
