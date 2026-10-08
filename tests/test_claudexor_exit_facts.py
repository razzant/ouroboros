"""Exit/capacity facts cross the real status and context seams without a wake."""
import json
from types import SimpleNamespace

import pytest

from ouroboros import claudexor_daemon as daemon, claudexor_exit_facts as facts, config
from ouroboros.gateways import claudexor as transport
from ouroboros.utils import append_jsonl


def exit_row(root, *, serving=True, cause='heap_exhausted'):
    append_jsonl(root / 'logs/supervisor.jsonl', {
        'type': 'claudexor_daemon_start_failed', 'descriptor_written': serving,
        'classification': cause, 'exit_signal': 6, 'exit_code': None,
        'pin_version': '3.22.0', 'at': '2026-10-08T01:02:03Z',
    })


def rows(root):
    return [json.loads(line) for line in (root / 'logs/supervisor.jsonl').read_text(
        encoding='utf-8').splitlines()]


def memory():
    return {'heapUsedBytes': 3 * 2**30, 'heapLimitBytes': 16 * 2**30,
            'rssBytes': 4 * 2**30, 'externalBytes': 100,
            'nodeHeapArgs': ['--max-old-space-size=16384'],
            'atAdmission': {'heapUsedBytes': 2 * 2**30, 'rssBytes': 3 * 2**30,
                            'at': '2026-10-08T02:00:00Z'},
            'sampledAt': '2026-10-08T02:01:00Z'}


@pytest.fixture
def status_env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(facts, '_CAPACITY_SEEN', {})
    descriptor = tmp_path / 'descriptor.json'
    descriptor.write_text('{}', encoding='utf-8')
    monkeypatch.setattr(daemon, 'owned_descriptor_path', lambda: descriptor)
    monkeypatch.setattr(daemon, 'verify_owned_home', lambda: '')
    import ouroboros.claudexor_runtime as runtime
    monkeypatch.setattr(runtime, 'get_runtime_manager', lambda: SimpleNamespace(status=lambda **kw: {}))
    manager = daemon.OwnedClaudexorDaemon()
    manager._engine_version, manager._engine_build_sha = '3.22.1', 'fixture-build'
    endpoint = SimpleNamespace(host='127.0.0.1', port=12345)
    monkeypatch.setattr(manager, '_classify_liveness', lambda: (endpoint, 'running', ''))
    calls = []

    class Gateway:
        response = {'memory': memory()}
        error = None

        def __init__(self, selected):
            assert selected is endpoint

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def daemon_status(self, **kwargs):
            calls.append(kwargs)
            if self.error:
                raise self.error
            return self.response

    monkeypatch.setattr(transport, 'ClaudexorGateway', Gateway)
    return manager, Gateway, descriptor, calls


def test_status_exposes_exit_and_records_capacity_once_per_generation(tmp_path, status_env):
    manager, gateway, descriptor, calls = status_env
    exit_row(tmp_path)
    for _ in range(3):
        status = manager.status_dict()
        assert status['last_exit']['phase'] == 'serving'
        assert status['last_exit']['observed_at'] == '2026-10-08T01:02:03Z'
        assert status['memory'] == memory()
    capacity = [row for row in rows(tmp_path) if row['type'] == 'claudexor_engine_capacity']
    assert len(capacity) == 1 and len(calls) == 3
    assert capacity[0]['heap_limit_bytes'] == 16 * 2**30
    assert capacity[0]['admission_heap_used_bytes'] == 2 * 2**30
    assert capacity[0]['engine_build_sha'] == 'fixture-build'
    assert capacity[0]['node_heap_args'] == ['--max-old-space-size=16384']
    facts._CAPACITY_SEEN.clear()  # A second reader rejoins the existing log observation.
    manager.status_dict()
    assert len([r for r in rows(tmp_path) if r['type'] == 'claudexor_engine_capacity']) == 1
    descriptor.write_text('{"new_generation":true}', encoding='utf-8')
    manager.status_dict()
    assert len([r for r in rows(tmp_path) if r['type'] == 'claudexor_engine_capacity']) == 2


def test_old_engine_status_absence_is_unknown_and_offline_never_calls(tmp_path, status_env, monkeypatch):
    manager, gateway, _, calls = status_env
    gateway.error = transport.ClaudexorUnavailable('not_found', 'old engine', status_code=404)
    assert manager.status_dict()['memory'] is None
    row, = rows(tmp_path)
    assert row['heap_limit_bytes'] is None and row['admission_heap_used_bytes'] is None
    assert row['node_heap_args'] is None
    before = len(calls)
    monkeypatch.setattr(manager, '_classify_liveness', lambda: (None, 'stale', ''))
    assert manager.status_dict()['memory'] is None and len(calls) == before


def test_context_has_one_saved_observation_line_without_daemon_access(tmp_path, monkeypatch):
    from ouroboros.context_health import build_health_invariants

    def forbidden(*args, **kwargs):
        raise AssertionError('context must not contact or wake the daemon')

    monkeypatch.setattr(daemon, 'ensure_owned_gateway', forbidden)
    monkeypatch.setattr(daemon, 'read_owned_gateway', forbidden)
    monkeypatch.setattr(daemon, 'get_owned_daemon', forbidden)
    monkeypatch.setattr(transport, 'ClaudexorGateway', forbidden)
    env = SimpleNamespace(drive_path=lambda p: tmp_path / p, repo_path=lambda p: tmp_path / p)
    assert 'ENGINE EXITED' not in build_health_invariants(env)
    exit_row(tmp_path, serving=False, cause='writer_lease_contended')
    exit_row(tmp_path)
    facts._record_capacity(tmp_path, 'generation', '3.22.1', 'fixture-build', memory())
    rendered = build_health_invariants(env)
    assert rendered.count('WARNING: ENGINE EXITED') == 1
    assert 'exited 2 time(s)' in rendered and 'heap_exhausted, signal 6, while serving' in rendered
    assert 'observed at 2026-10-08T01:02:03Z, not the death time' in rendered
    assert 'heap limit 16.00 GiB' in rendered and 'admission heap 2.00 GiB' in rendered


def test_saved_window_is_bounded_and_malformed_rows_are_fail_soft(tmp_path):
    path = tmp_path / 'logs/supervisor.jsonl'
    exit_row(tmp_path)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'padding': '.' * facts._RECENT_BYTES}) + '\n')
        stream.write('broken\n')
    assert facts.health_line(tmp_path) == ''
    exit_row(tmp_path, serving=False)
    assert 'while startup' in facts.health_line(tmp_path)
    assert facts.memory_facts({}) is None
    partial = facts.memory_facts({'memory': {'heapUsedBytes': 0, 'heapLimitBytes': True}})
    assert partial['heapUsedBytes'] == 0 and partial['heapLimitBytes'] is None
    assert partial['atAdmission'] is None and partial['nodeHeapArgs'] is None
