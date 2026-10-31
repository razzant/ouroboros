"""Fresh file stores commit schema, headers and seed data before atomic publication."""
from __future__ import annotations

import contextlib
import json
import pathlib
import sqlite3
import subprocess
import sys

import pytest

from ouroboros import dialogue_store, store, store_import, usage_store
from ouroboros.utils import atomic_write_json
from tests._usage_store_testing import seed

KINDS = ('current', 'dialog', 'usage')


def _target(root, kind):
    return root / 'state' / f'{kind}.sqlite'


def _seed(root, kind):
    (root / 'state').mkdir(parents=True, exist_ok=True)
    if kind == 'current':
        atomic_write_json(root / 'task_results' / 'fixture.json', {
            '_schema_version': 1, 'task_id': 'fixture', 'status': 'completed',
            'ts': '2026-01-01T00:00:00Z', 'updated_at': '2026-01-01T00:00:00Z',
            'result': 'preserved fixture'})
    elif kind == 'usage':
        seed(root, 2, cost=0.125)


def _create(root, kind):
    if kind == 'current':
        store_import.ensure_current_store(root, with_children=False)
    elif kind == 'usage':
        usage_store.migrate_from_journal(root)
    else:
        dialogue_store.create(root)


def _assert_content(root, kind):
    target = _target(root, kind)
    with contextlib.closing(sqlite3.connect(f'{target.as_uri()}?mode=ro', uri=True)) as conn:
        assert conn.execute('PRAGMA integrity_check').fetchone() == ('ok',)
        assert conn.execute('PRAGMA application_id').fetchone()[0] != 0
        if kind == 'current':
            body = json.loads(conn.execute("SELECT body FROM task_bodies WHERE task_id='fixture'").fetchone()[0])
            assert body['result'] == 'preserved fixture'
            assert json.loads(conn.execute("SELECT value FROM meta WHERE key='phase'").fetchone()[0]) == 'published'
        elif kind == 'usage':
            rows = conn.execute('SELECT attempt_id,state,cost_usd FROM attempts ORDER BY attempt_id').fetchall()
            assert rows == [('seed-0', 'settled', '0.125'), ('seed-1', 'settled', '0.125')]
            assert json.loads(conn.execute("SELECT value FROM meta WHERE key='import'").fetchone()[0])['status'] == 'completed'
        else:
            assert conn.execute('PRAGMA user_version').fetchone()[0] == dialogue_store.SCHEMA_VERSION
            assert conn.execute('SELECT stream,next_seq FROM streams ORDER BY stream').fetchall() == [('chat', 0), ('progress', 0)]
            assert conn.execute('SELECT COUNT(*) FROM import_ledger').fetchone()[0] == 2
    # The real public reader must accept the published header and contents too.
    if kind == 'current':
        with store.open_store(root, write=False) as txn:
            assert txn.load('fixture')['result'] == 'preserved fixture'
    elif kind == 'usage':
        with usage_store.read(root) as txn:
            assert len(txn.attempts()) == 2
    else:
        assert dialogue_store.meta(root, 'schema_version') == dialogue_store.SCHEMA_VERSION


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('tier', (store.TIER_ENFORCED, store.TIER_NAME))
def test_creation_ddl_and_seed_share_one_full_transaction(tmp_path, monkeypatch, kind, tier):
    from ouroboros import platform_layer
    monkeypatch.setattr(platform_layer, 'kernel_file_locks_enforced', lambda *_a, **_kw: tier == store.TIER_ENFORCED)
    _seed(tmp_path, kind)
    connect = store.connect
    traces, settings = [], []

    def observed(path, selected_tier, **kwargs):
        conn = connect(path, selected_tier, **kwargs)
        if kwargs.get('create'):
            settings.append((selected_tier, conn.execute('PRAGMA synchronous').fetchone()[0],
                             conn.execute('PRAGMA journal_mode').fetchone()[0]))
            conn.set_trace_callback(lambda sql: traces.append((sql.strip().rstrip(';').upper(), conn.in_transaction)))
        return conn

    monkeypatch.setattr(store, 'connect', observed)
    _create(tmp_path, kind)
    assert len(settings) == 1 and settings[0][:2] == (tier, 2)
    assert settings[0][2] == store._journal_mode(_target(tmp_path, kind), tier).lower()
    writes = [(sql, active) for sql, active in traces if sql.startswith(('CREATE ', 'ALTER ', 'INSERT ', 'UPDATE ', 'DELETE ', 'DROP '))]
    assert writes and all(active for _, active in writes), writes
    assert sum(sql == 'BEGIN IMMEDIATE' for sql, _ in traces) == 1
    assert sum(sql == 'COMMIT' for sql, _ in traces) == 1
    _assert_content(tmp_path, kind)
    before = _target(tmp_path, kind).read_bytes()
    _create(tmp_path, kind)
    assert _target(tmp_path, kind).read_bytes() == before


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('stage', ('schema', 'seed', 'commit'))
def test_creation_sql_failure_never_publishes_and_retry_succeeds(tmp_path, monkeypatch, kind, stage):
    _seed(tmp_path, kind)
    connect, reached = store.connect, []

    def failing(path, tier, **kwargs):
        conn = connect(path, tier, **kwargs)
        if kwargs.get('create'):
            def deny(action, first, second, _db, _trigger):
                matches = ((stage == 'schema' and action == sqlite3.SQLITE_CREATE_INDEX)
                           or (stage == 'seed' and action == sqlite3.SQLITE_INSERT and first != 'sqlite_master')
                           or (stage == 'commit' and action == sqlite3.SQLITE_TRANSACTION and first == 'COMMIT'))
                if matches:
                    reached.append((action, first, second))
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK
            conn.set_authorizer(deny)
        return conn

    monkeypatch.setattr(store, 'connect', failing)
    with pytest.raises(Exception):
        _create(tmp_path, kind)
    assert reached, 'The intended SQL failure was not exercised'
    assert not _target(tmp_path, kind).exists()
    assert not list((tmp_path / 'state').glob(f'{kind}.sqlite.import-*'))
    monkeypatch.setattr(store, 'connect', connect)
    _create(tmp_path, kind)
    _assert_content(tmp_path, kind)


# The child really exits without unwinding Python's context managers. It touches only
# its explicit tmp_path root and inherited safe_test configuration; no server is run.
_CRASH = r'''
import pathlib, sys, os, sqlite3
from ouroboros import store
from tests.test_store_initialization import _create
root, kind, stage = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
connect, commit, replace = store.connect, store.commit, os.replace
hit = root / 'crash-stage.txt'
def die():
    hit.write_text(stage, encoding='utf-8')
    os._exit(71)
def instrumented(path, tier, **kwargs):
    conn = connect(path, tier, **kwargs)
    if kwargs.get('create') and stage == 'schema':
        def auth(action, first, second, db, trigger):
            if action == sqlite3.SQLITE_CREATE_INDEX:
                die()
            return sqlite3.SQLITE_OK
        conn.set_authorizer(auth)
    return conn
def at_commit(conn, **kwargs):
    if stage == 'before_commit':
        die()
    result = commit(conn, **kwargs)
    if stage == 'after_commit':
        die()
    return result
def at_replace(src, dst):
    result = replace(src, dst)
    if stage == 'after_publish' and pathlib.Path(dst) == root / 'state' / (kind + '.sqlite'):
        die()
    return result
store.connect, store.commit, os.replace = instrumented, at_commit, at_replace
_create(root, kind)
raise AssertionError('crash point not reached')
'''


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('stage', ('schema', 'before_commit', 'after_commit', 'after_publish'))
def test_process_death_then_public_creation_preserves_atomicity(tmp_path, kind, stage):
    _seed(tmp_path, kind)
    result = subprocess.run([sys.executable, '-c', _CRASH, str(tmp_path), kind, stage],
                            cwd=pathlib.Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
    assert result.returncode == 71, result.stdout + result.stderr
    assert (tmp_path / 'crash-stage.txt').read_text(encoding='utf-8') == stage
    assert _target(tmp_path, kind).exists() == (stage == 'after_publish')
    if stage == 'after_publish':
        _assert_content(tmp_path, kind)
    _create(tmp_path, kind)
    _assert_content(tmp_path, kind)
