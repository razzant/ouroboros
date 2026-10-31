"""Actual Deno network authority and cancellation through existing process custody."""
from __future__ import annotations

import json
import shutil
import threading
import time

import pytest

from ouroboros.tools import skill_exec
from tests.test_skill_install_resources import serve
from tests.test_skill_runtime_commands import execute, installed_script

pytestmark = pytest.mark.serial


@pytest.mark.parametrize('net,resource_allowed', [(False, True), (True, False), (True, True)])
def test_deno_network_requires_effect_and_task_resource(tmp_path, monkeypatch, net, resource_allowed):
    if not shutil.which('deno'):
        pytest.skip('deno runtime unavailable')
    with serve({'/feature': b'network feature'}) as (url, requests):
        ctx, _, script = installed_script(tmp_path, monkeypatch, runtime='deno',
            permissions=['net'] if net else [], body='console.log(await (await fetch(Deno.args[0])).text());')
        ctx.task_metadata = {'task_contract': {'allowed_resources': {'network': resource_allowed}}}
        result = execute(ctx, script, [url + '/feature'])
        if net and resource_allowed:
            assert json.loads(result)['stdout'].strip() == 'network feature'
            assert requests == ['/feature']
        else:
            assert 'SKILL_EXEC_FAILED' in result and 'NotCapable' in result
            assert requests == []


def test_deno_network_false_also_prevents_static_import_fetch(tmp_path, monkeypatch):
    if not shutil.which('deno'):
        pytest.skip('deno runtime unavailable')
    with serve({'/module.js': b'export const value=42;'}) as (url, requests):
        ctx, _, script = installed_script(tmp_path, monkeypatch, runtime='deno', permissions=['net'],
            body=f"import {{value}} from '{url}/module.js';console.log(value);")
        ctx.task_metadata = {'task_contract': {'allowed_resources': {'network': False}}}
        result = execute(ctx, script)
        assert 'SKILL_EXEC_FAILED' in result and requests == []


@pytest.mark.parametrize('runtime,body', [
    ('python3', 'import time\nwhile True: time.sleep(1)\n'),
    ('node', 'setInterval(()=>{},1000);'),
    ('deno', 'setInterval(()=>{},1000);'),
    ('go', 'package main\nfunc main(){for {}}\n'),
])
def test_owned_running_skill_process_can_be_cancelled_without_survivors(tmp_path, monkeypatch, runtime, body):
    if not skill_exec._resolve_runtime_binary(runtime)[0]:
        pytest.skip(f'{runtime} runtime unavailable')
    ctx, _, script = installed_script(tmp_path, monkeypatch, runtime=runtime, body=body)
    results = []
    thread = threading.Thread(target=lambda: results.append(execute(ctx, script)))
    thread.start()
    own = None
    try:
        # Go compiles the skill first: a cold build on a loaded Windows runner can take longer than 20 s.
        until = time.monotonic() + (60 if runtime == 'go' else 20)
        while time.monotonic() < until and thread.is_alive():
            with skill_exec._subprocess_lock:
                for proc in skill_exec._active_subprocesses:
                    args = [str(a) for a in proc.args]
                    if runtime == 'go':
                        matches = len(args) == 1 and args[0].startswith(str(ctx.drive_root / 'state/skills/runtime_fixture/go-exec-'))
                    else:
                        matches = str(script) in args
                    if matches:
                        own = proc
                        break
            if own:
                break
            time.sleep(0.01)
        assert own is not None, results
        skill_exec._kill_process_group(own)
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert results and 'SKILL_EXEC_FAILED' in results[0], results
        assert own.poll() is not None
        with skill_exec._subprocess_lock:
            assert own not in skill_exec._active_subprocesses
        assert not list((ctx.drive_root / 'state/skills/runtime_fixture').glob('go-exec-*'))
    finally:
        if own is not None and own.poll() is None:
            skill_exec._kill_process_group(own)
        thread.join(timeout=65)
