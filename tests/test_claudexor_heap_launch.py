"""The managed host passes only the engine's validated heap recommendation."""
import json
import subprocess

import pytest

from ouroboros import claudexor_runtime as runtime, platform_layer
from tests.test_claudexor_runtime_delivery import _archive, _pin, NODE_VERSION


@pytest.mark.parametrize('launch,options,expected,error', [
    ({'nodeArgs': ['--max-old-space-size=16384']}, '', ['--max-old-space-size=16384'], False),
    ({'nodeArgs': ['--max-old-space-size=8192']}, '--max-old-space-size=6144', [], False),
    ({'nodeArgs': ['--max-old-space-size=8192']}, '--trace-warnings', ['--max-old-space-size=8192'], False),
    (None, '', [], False),
    ({'nodeArgs': []}, '', [], False),
    ({'nodeArgs': ['--inspect']}, '', [], True),
    ({'nodeArgs': ['--max-old-space-size=8192', '--require=elsewhere']}, '', [], True),
    ({'nodeArgs': ['--max-old-space-size=8']}, '', [], True),
    ({'nodeArgs': '--max-old-space-size=8192'}, '', [], True),
    ({'nodeArgs': [8192]}, '', [], True),
    ('invalid', '', [], True),
])
def test_ensure_applies_engine_args_only_to_spawn(tmp_path, monkeypatch, launch, options, expected, error):
    pin = _pin(_archive(tmp_path / 'runtime.tar.gz'))
    manager = runtime.ClaudexorRuntimeManager(pin)
    monkeypatch.setattr(runtime, '_explicit_binary', lambda: ('', ''))
    monkeypatch.setattr(manager, '_managed_metadata', lambda: {'version': pin.version})
    monkeypatch.setattr(manager, '_resolve_node', lambda _: '/fixture/node')
    monkeypatch.setattr(platform_layer, 'probe_node_version', lambda _: NODE_VERSION)
    monkeypatch.setenv('NODE_OPTIONS', options)
    plain = manager.resolve_command()
    assert plain[0] == '/fixture/node' and len(plain) == 2
    payload = {'version': pin.version, 'buildSha': pin.build_sha}
    if launch is not None:
        payload['launch'] = launch
    probes = []

    def probe(command, **kwargs):
        probes.append(command)
        assert kwargs['env']['NODE_OPTIONS'] == options
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), '')

    monkeypatch.setattr(runtime.subprocess, 'run', probe)
    assert manager.ensure() == [plain[0], *expected, plain[1]]
    assert probes == [[*plain, '--probe']]
    assert bool(manager._last_error) is error
    if error:
        assert 'runtime_launch_invalid' in manager._last_error
    assert manager.resolve_command() == plain  # read-only selection stays probe-free
    # A later identity probe must never measure the previously enlarged heap.
    manager._probe_payload([plain[0], '--max-old-space-size=16384', plain[1]],
                           expected_node_version=NODE_VERSION)
    assert probes[-1] == [*plain, '--probe']


def test_explicit_binary_ignores_engine_launch_recommendation(monkeypatch):
    manager = runtime.ClaudexorRuntimeManager(None)
    monkeypatch.setattr(runtime, '_explicit_binary', lambda: ('/fixture/external', ''))
    assert manager.ensure() == ['/fixture/external']
