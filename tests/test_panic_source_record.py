import json
from unittest.mock import Mock

from ouroboros import panic_source


class InlineThread:
    def __init__(self, *, target, **kwargs):
        self.target = target
    def start(self):
        self.target()


def test_ingress_record_never_trusts_client_owner(monkeypatch, tmp_path):
    from ouroboros import config, platform_layer
    monkeypatch.setattr(config, 'resolve_data_dir', lambda: tmp_path)
    monkeypatch.setattr(platform_layer, 'local_connection_peer', lambda _: {'pid': 123, 'executable': 'helper'})
    monkeypatch.setattr(panic_source.threading, 'Thread', InlineThread)
    panic_source.ingress_record(Mock(client=('127.0.0.1', 12345)), 'web_ui_http',
                                client_surface={'owner_surface': 'owner'})
    row = json.loads((tmp_path / 'state/panic_source.json').read_text())
    assert row['owner_surface'] == 'unknown'
    assert row['peer']['pid'] == 123
    assert row['ingress'] == 'web_ui_http'


def test_source_write_failure_is_logged_and_swallowed(monkeypatch, tmp_path, caplog):
    from ouroboros import utils
    monkeypatch.setattr(panic_source.threading, 'Thread', InlineThread)
    monkeypatch.setattr(utils, 'atomic_write_json', Mock(side_effect=OSError('disk failed')))
    panic_source.record_source(tmp_path, ingress='launcher_fallback', owner_surface='owner')
    assert 'could not be recorded' in caplog.text


def test_late_peer_enrichment_cannot_replace_newer_causal_source(monkeypatch, tmp_path):
    import threading
    from ouroboros import platform_layer
    entered, release = threading.Event(), threading.Event()
    def held_lookup(_address):
        entered.set()
        assert release.wait(5)
        return {'pid': 123}
    monkeypatch.setattr(platform_layer, 'local_connection_peer', held_lookup)
    before = set(threading.enumerate())
    panic_source.record_source(tmp_path, ingress='web_ui_http', peer={'address': ['127.0.0.1', 12345]})
    assert entered.wait(5)
    # Keep both sources in the event log even when the later launcher stop owns
    # the latest-source projection.
    monkeypatch.setattr(panic_source.threading, 'Thread', InlineThread)
    panic_source.record_source(tmp_path, ingress='launcher_fallback', receiver_is_server=False)
    release.set()
    for thread in set(threading.enumerate()) - before:
        thread.join(5)
    latest = json.loads((tmp_path / 'state/panic_source.json').read_text())
    assert latest['ingress'] == 'launcher_fallback'
    events = [json.loads(line) for line in (tmp_path / 'logs/events.jsonl').read_text().splitlines()]
    assert [row['ingress'] for row in events] == ['web_ui_http', 'launcher_fallback', 'web_ui_http']
    assert events[-1]['record_id'] == events[0]['record_id']
    assert events[-1]['peer']['pid'] == 123 and events[0]['peer']['pid'] is None


def test_launcher_receiver_is_not_reported_as_server(monkeypatch, tmp_path):
    import os
    monkeypatch.setattr(panic_source.threading, 'Thread', InlineThread)
    panic_source.record_source(tmp_path, ingress='launcher_fallback', receiver_is_server=False)
    path = tmp_path / 'state/panic_source.json'
    row = json.loads(path.read_text())
    assert row['receiver_pid'] == os.getpid()
    assert row['server_pid'] is None and row['server_pid_basis'] == 'unknown'
    (tmp_path / 'state/server_process.json').write_text(json.dumps({'pid': 1234}))
    panic_source.record_source(tmp_path, ingress='desktop_bridge', receiver_is_server=False)
    row = json.loads(path.read_text())
    assert row['server_pid'] == 1234 and row['server_pid_basis'] == 'launcher_record'
    assert row['receiver_pid'] == os.getpid()


def test_delayed_initial_writer_retains_both_events_without_replacing_newer_source(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from ouroboros import utils

    pending = []
    times = iter(["2026-10-20T13:00:00+00:00", "2026-10-20T13:00:01+00:00"])
    monkeypatch.setattr(utils, "utc_now_iso", lambda: next(times))
    monkeypatch.setattr(panic_source.threading, "Thread",
                        lambda **kw: SimpleNamespace(start=lambda: pending.append(kw["target"])))
    panic_source.record_source(tmp_path, ingress="web_ui_ws")
    panic_source.record_source(tmp_path, ingress="desktop_bridge", receiver_is_server=False)
    pending[1]()
    pending[0]()
    latest = json.loads((tmp_path / "state/panic_source.json").read_text())
    assert latest["ingress"] == "desktop_bridge"
    events = [json.loads(line) for line in (tmp_path / "logs/events.jsonl").read_text().splitlines()]
    assert [row["ingress"] for row in events] == ["desktop_bridge", "web_ui_ws"]
    assert len({row["record_id"] for row in events}) == 2


def test_remote_peer_never_matches_an_unrelated_local_source_port(monkeypatch):
    from ouroboros import platform_peer
    import subprocess
    lookup = Mock(side_effect=AssertionError('remote peer cannot invoke local lookup'))
    monkeypatch.setattr(subprocess, 'run', lookup)
    for host in ('192.0.2.1', '2001:db8::1', 'remote.example'):
        assert platform_peer.local_connection_peer((host, 12345)) == {}
    lookup.assert_not_called()


def test_frozen_peer_lookup_keeps_unknown_instead_of_launching_app(monkeypatch):
    from ouroboros import platform_peer
    import subprocess
    import sys
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    lookup = Mock(side_effect=AssertionError('frozen executable is not Python -c'))
    monkeypatch.setattr(subprocess, 'run', lookup)
    assert platform_peer.local_connection_peer(('127.0.0.1', 12345)) == {}
    lookup.assert_not_called()
