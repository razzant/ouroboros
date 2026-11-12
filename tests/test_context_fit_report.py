"""Read-only diagnostic over real synthetic stores, with no provider or server."""
import errno
import hashlib
import json
import os
import sqlite3
from contextlib import closing
from urllib.parse import unquote, urlsplit

import pytest

from devtools.context_fit_report import report, measure_core
from devtools._context_fit_readonly import readonly_inputs, server_alive, snapshot_root
from tests import _memory_scale_fixture as scale


def fingerprint(root):
    return {str(path.relative_to(root)): (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
            for path in root.rglob('*') if path.is_file()}


def offline_fixture(root):
    """Finish synthetic writers before testing an immutable, offline reader (also on macOS SQLite)."""
    for path in root.rglob('*.sqlite'):
        with closing(sqlite3.connect(path)) as db:
            assert db.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()[0] == 0


def opens_inside(database, root):
    """Whether a sqlite3.connect name lands in ``root``, by path components under the root's own rules, so ``data2``
    stays outside ``data``. A file URI is read as SQLite reads it: percent-decoded, a Windows drive after the path's
    leading slash (file:///D:/a is D:\\a). Searching for the root's text would miss what as_uri rewrites: a Windows
    root's backslashes, any space or non-ASCII letter."""
    name = os.fsdecode(database)
    if name.startswith('file:'):
        name = unquote(urlsplit(name).path)
        name = name.lstrip('/') if root.drive else name
    return type(root)(name).is_relative_to(root)


def test_measurement_uses_ordinary_scale_fitter_and_composition(tmp_path, monkeypatch):
    from ouroboros import memory_floor as floor
    with monkeypatch.context() as old:
        old.setattr(floor, 'LADDER', tuple(step for step in floor.LADDER if step not in {'F7a', 'F5a', 'F8', 'F8a'}))
        before = measure_core(scale.core(tmp_path), scale.tools(), root=tmp_path, window=92000, reserve=10000)
    assert before['verdict'] == 'MinimumViewUnfit' and before['deficit_tokens'] > 0
    result = measure_core(scale.core(tmp_path), scale.tools(), root=tmp_path, window=92000, reserve=10000)
    assert result['verdict'] == 'fits', result
    assert sum(row['tokens'] for row in result['blocks']) == result['input_tokens']
    assert result['ladder'].get('F7a') and result['ladder'].get('F5a')
    assert result['input_tokens'] + 10000 <= 92000
    print('SCALE_COMPARISON ' + json.dumps({'before': before, 'after': result}, sort_keys=True))


@pytest.mark.parametrize('mode', ['low', 'max'])
def test_a_directly_required_handbook_is_reported_whole_in_every_mode(tmp_path, mode):
    """A task's own requirement puts the whole handbook in the block Max otherwise fills with its chapter map."""
    from dataclasses import replace

    core = replace(scale.core(tmp_path), development_md='# Development\n\n## Review\n\n' + 'practice ' * 400)

    def development(required):
        result = measure_core(replace(core, development_required=required), scale.tools(), root=tmp_path,
                              window=1_000_000, reserve=10000, mode=mode)
        return [row['form'] for row in result['blocks'] if row['block'] == 'DEVELOPMENT']

    assert development(True) == ['whole']
    assert development(False) == (['orientation'] if mode == 'max' else [])


def test_the_start_text_is_written_as_the_mind_reads_it_and_a_declared_density_prices_the_start(tmp_path):
    from devtools.context_fit_report import main

    core, dump = scale.core(tmp_path), tmp_path / 'scratch' / 'start.txt'
    dump.parent.mkdir()
    plain = measure_core(core, scale.tools(), root=tmp_path, window=1_000_000, reserve=10000, dump=dump)
    text = dump.read_text(encoding='utf-8')
    assert text.startswith(core.base_prompt) and '## My story' in text and '## This room (Main)' in text
    assert (plain['density'], plain['view']['work_room_tokens']) == (1.0, plain['margin_tokens'])
    dense = measure_core(core, scale.tools(), root=tmp_path, window=1_000_000, reserve=10000, ratio=1.6)
    assert dense['density'] == dense['fit']['measurement']['measurement_density'] == 1.6
    assert dense['input_tokens'] > 1.5 * plain['input_tokens'] and dense['margin_tokens'] < plain['margin_tokens']
    with pytest.raises(SystemExit):  # the report never writes under the data root it reads
        main(['--data', str(tmp_path), '--room', '1', '--window', '1000', '--reserve', '10', '--dump', str(dump)])


def test_view_counts_name_todays_forms_by_the_law_of_forms():
    """Whole, my story's records and the closing sentences are retellings, task lines without one and a silent room's
    line host facts, the words nobody considered named apart; at the deepest walk before uncovered owner words every
    other reduced form is an emergency form by its step (pages by address, a conversation by its counters)."""
    from devtools.context_fit_report import view_counts
    from ouroboros import memory_floor as mf, memory_view as mv
    from tests import _memory_view_synthetic as syn

    snapshot = syn.snapshot(room_facts=syn.work_room(), story=syn.told_story(), conversations=syn.place_threads())
    whole = view_counts(snapshot, mv.FULL_VIEW)
    assert {k: v for k, v in whole.items() if k != "reduced_forms"} == {
        'retellings': {"legacy, a helper's": 1, 'part': 1, 'page': 6, 'task summary': 3},
        'host_facts': {'task line': 2, 'silent room': 1}, 'unconsidered': {'as text': 2, 'as a line': 1},
        'emergency': {}, 'rows_as_words': 15}
    deepest = mf.fit_memory_view(snapshot, {'margin': 0, 'physical': None})
    result = view_counts(snapshot, deepest)
    from ouroboros.memory_form_facts import form_counts
    assert result['reduced_forms'] == form_counts(snapshot, deepest)
    assert result['rows_as_words'] == 2
    assert result['emergency']['conversation count'] == 1
    assert 'F6w' not in result['emergency']  # those rows are swallowed by the count



@pytest.mark.parametrize('stored_tail', [False, True])
def test_real_scale_root_read_only_report(tmp_path, monkeypatch, capsys, stored_tail):
    import socket
    from ouroboros import agent_startup_checks
    from devtools.context_fit_report import main

    def forbidden(*args, **kwargs):
        pytest.fail('offline measurement attempted a network connection or live process scan')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(agent_startup_checks, 'check_stray_server_processes', forbidden)
    root = scale.write_root(tmp_path / 'data')
    if stored_tail:
        from ouroboros import dialogue_store, memory_inventory
        from ouroboros.chronicle_store import ChronicleStore

        filled = memory_inventory.fill_tail(root, lambda: False)
        assert filled['state'] == 'done' and filled['items'] > 0, filled
        assert ChronicleStore(root).tail_state()['complete']
        # The server opportunistically extends this complete cache on its next
        # read. The diagnostic must read the new row without publishing a cache.
        dialogue_store.append(root, 'chat', {'chat_id': 1, 'direction': 'in',
            'text': 'The owner spoke after the stored tail was filled.',
            'ts': '2026-10-23T01:00:00+00:00', 'client_message_id': 'after-tail'})
    offline_fixture(root)
    before = fingerprint(root)
    connected = []
    original = sqlite3.connect
    source = root.resolve()
    def connect(database, *args, **kwargs):
        if opens_inside(database, source):
            connected.append(str(database))
            assert 'mode=ro&immutable=1' in str(database)
        return original(database, *args, **kwargs)
    monkeypatch.setattr(sqlite3, 'connect', connect)
    result = report(root, 1, 92000, 10000)
    assert connected and fingerprint(root) == before
    assert sum(row['count'] for row in result['blocks'] if row['block'] == 'other conversations') == 228
    assert sum(row['count'] for row in result['blocks'] if row['block'] == 'this room notes') == 6
    assert any(row['count'] and row['form'] in ('line', 'aggregate') for row in result['blocks']
               if row['block'] == 'this room notes')
    assert next(row for row in result['blocks'] if row['block'] == 'story')['count'] == 113
    assert result['verdict'] == 'fits' and result['margin_tokens'] >= 0, result
    with readonly_inputs(root), pytest.raises(PermissionError):
        (root / 'forbidden').write_text('not allowed')
    with readonly_inputs(root), pytest.raises(PermissionError):
        sqlite3.connect(root / 'forbidden.sqlite')
    (root / 'state/server_process.json').write_text(json.dumps({'pid': os.getpid()}))
    before_snapshot = fingerprint(root)
    assert main(['--data', str(root), '--room', '1', '--window', '92000', '--reserve', '10000',
                 '--snapshot', '--json']) == 0
    copied = json.loads(capsys.readouterr().out)
    assert copied['snapshot'] and copied['verdict'] == 'fits'
    assert fingerprint(root) == before_snapshot
    print('SCALE_ROOT_REPORT ' + json.dumps(result, sort_keys=True))
    print('SCALE_ROOT_SNAPSHOT_REPORT ' + json.dumps(copied, sort_keys=True))


def test_readonly_tail_reads_new_rows_without_extending_the_stored_cache(tmp_path):
    from ouroboros import dialogue_store, memory_inventory
    from tests import _memory_inventory_shared as shared

    shared.world(tmp_path)
    assert memory_inventory.fill_tail(tmp_path, lambda: False) == {'state': 'done', 'items': 10}
    dialogue_store.append(tmp_path, 'chat', shared.msg('2026-10-23T01:00:00+00:00', 'After the stored tail'))
    offline_fixture(tmp_path)
    before = fingerprint(tmp_path)
    with readonly_inputs(tmp_path):
        rows = memory_inventory.rows_after_frontier(tmp_path)
        assert len(rows) == 11
    assert fingerprint(tmp_path) == before


def test_live_pid_and_kernel_lock_refuse_without_snapshot(tmp_path):
    from ouroboros import platform_layer
    (tmp_path / 'state').mkdir()
    pid = tmp_path / 'state/server_process.json'
    pid.write_text(json.dumps({'pid': os.getpid()}))
    assert server_alive(tmp_path)
    with pytest.raises(ValueError, match='live server'):
        report(tmp_path, 1, 92000, 10000)
    pid.unlink()
    with (tmp_path / 'state/server.lock').open('w') as lock:
        platform_layer.file_lock_exclusive_nb(lock.fileno())  # held as admit_data_root holds it
        assert server_alive(tmp_path)
        with pytest.raises(ValueError, match='live server'):
            report(tmp_path, 1, 92000, 10000)
        platform_layer.file_unlock(lock.fileno())  # a Windows lock goes before its handle
    assert not server_alive(tmp_path)


@pytest.mark.parametrize('state, live', [('error', True), ('no_such_pid', False)])
def test_windows_liveness_never_signals_and_an_unexplained_state_stays_live(tmp_path, monkeypatch, state, live):
    """Windows rules on any host, not a native run: os.kill(pid, 0) there is CTRL_C_EVENT to the pid's console
    group, so a recorded pid is read through its process state, and only absence proven there clears it."""
    from ouroboros import platform_layer
    monkeypatch.setattr(platform_layer, 'IS_WINDOWS', True)
    monkeypatch.setattr(platform_layer, '_windows_process_state', lambda pid: state)
    monkeypatch.setattr(os, 'kill', lambda *args: pytest.fail(f'a liveness read signalled {args}'))
    (tmp_path / 'state').mkdir()
    (tmp_path / 'state/server_process.json').write_text(json.dumps({'pid': 4242}), encoding='utf-8')
    assert server_alive(tmp_path) is live


def test_the_launchers_utf8_record_is_read_as_utf8_under_a_windows_default_encoding(tmp_path, monkeypatch):
    """Windows Python reads text in the ANSI code page by default (emulated here); the launcher writes its record
    as UTF-8, so a non-ASCII profile path must not make a provably gone server an unknown one."""
    import io

    from ouroboros import platform_layer
    from ouroboros.utils import atomic_write_json
    monkeypatch.setattr(io, 'text_encoding', lambda encoding, stacklevel=2: encoding or 'cp1252')
    monkeypatch.setattr(platform_layer, 'pid_provably_gone', lambda pid: True)
    (tmp_path / 'state').mkdir()
    atomic_write_json(tmp_path / 'state/server_process.json', {'pid': 4242, 'repo_dir': 'C:\\Users\\Антон\\Ouroboros'})
    assert not server_alive(tmp_path)


@pytest.mark.parametrize('answer', ['held on Windows', 'unknown Windows error', 'no POSIX locks', 'unreadable'])
def test_only_a_granted_lock_proves_the_root_offline(tmp_path, monkeypatch, answer):
    """The lock is tried through the server's own descriptor lock on the existing file. A missing lock is no
    evidence and stays missing; held, refused or unreadable all leave the root possibly served."""
    from ouroboros import platform_layer
    lock = tmp_path / 'state/server.lock'
    assert not server_alive(tmp_path) and not lock.exists()
    lock.parent.mkdir()
    original = platform_layer._file_lock
    def shared_probe(fd, *, exclusive, blocking):
        assert not exclusive and not blocking  # a read-only NFS descriptor cannot take an exclusive lock
        return original(fd, exclusive=exclusive, blocking=blocking)
    monkeypatch.setattr(platform_layer, '_file_lock', shared_probe)
    if answer == 'unreadable':
        lock.mkdir()
    else:
        lock.write_bytes(b'pid=4242\n')
        assert not server_alive(tmp_path)  # present, and nobody holds it
        refusal = {'held on Windows': platform_layer._win32_lock_error(33),
                   'unknown Windows error': platform_layer._win32_lock_error(6),
                   'no POSIX locks': OSError(errno.ENOLCK, os.strerror(errno.ENOLCK))}[answer]

        def refuse(fd, *, blocking=True):
            assert blocking is False
            raise refusal
        monkeypatch.setattr(platform_layer, 'file_lock_shared', refuse)
    assert server_alive(tmp_path)


def test_snapshot_measures_a_held_root_and_its_wal_without_probing_liveness(tmp_path, monkeypatch):
    """An owned live server pid refuses a direct read and outlives it. With that server's admission held and
    frames committed only to the WAL, --snapshot asks no liveness, opens nothing in the source, recovers the WAL
    in a private copy under the platform temp parent and leaves the source, its lock and its server as found."""
    import subprocess
    import sys

    from devtools import _context_fit_readonly as readonly
    from ouroboros import dialogue_store, platform_layer, test_environment
    from tests._store_journal_shared import use_journal

    use_journal(monkeypatch, 'WAL')  # this scenario needs committed WAL frames even when the host defaults to PERSIST
    root = scale.write_root(tmp_path / 'data')
    offline_fixture(root)
    platform_temp = tmp_path / 'platform-temp'  # what Windows has instead of /tmp
    platform_temp.mkdir()
    monkeypatch.setattr(test_environment, 'default_temp_parent', lambda: platform_temp)
    server = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(900)'])
    lock = os.open(root / 'state/server.lock', os.O_RDWR | os.O_CREAT | getattr(os, 'O_BINARY', 0), 0o644)
    writer = sqlite3.connect(dialogue_store.path(root))  # an open reader keeps the appended frames in the WAL
    try:
        (root / 'state/server_process.json').write_text(json.dumps({'pid': server.pid}), encoding='utf-8')
        with pytest.raises(ValueError, match='live server'):
            report(root, 1, 92000, 10000)
        platform_layer.file_lock_exclusive_nb(lock)  # held as admit_data_root holds it
        writer.execute('SELECT count(*) FROM dialogue').fetchone()
        dialogue_store.append(root, 'chat', {'chat_id': 1, 'direction': 'in', 'text': 'Committed to the WAL only.',
                                             'ts': '2026-10-23T02:00:00+00:00', 'client_message_id': 'wal-only'},
                              emit_live=False)
        assert (root / 'state/dialog.sqlite-wal').stat().st_size, 'premise: committed frames not yet checkpointed'
        alive, snapshot, copies, opened = readonly.server_alive, readonly.snapshot_root, [], []

        def copy_then_read(source, destination):
            snapshot(source, destination)
            assert not (destination / 'state/dialog.sqlite-wal').exists()
            uri = dialogue_store.path(destination).as_uri() + '?mode=ro&immutable=1'
            with closing(sqlite3.connect(uri, uri=True)) as db:
                copies.append((destination, db.execute(
                    "SELECT count(*) FROM dialogue WHERE client_message_id = 'wal-only'").fetchone()[0]))
        connect = sqlite3.connect

        def recorded(database, *args, **kwargs):
            opened.append(os.fsdecode(database))
            return connect(database, *args, **kwargs)
        monkeypatch.setattr(readonly, 'server_alive', lambda path: pytest.fail('--snapshot probed liveness'))
        monkeypatch.setattr(readonly, 'snapshot_root', copy_then_read)
        monkeypatch.setattr(sqlite3, 'connect', recorded)
        before = fingerprint(root)
        result = report(root, 1, 92000, 10000, snapshot=True)
        assert result['snapshot'] and result['verdict'] == 'fits', result
        assert fingerprint(root) == before
        source = root.resolve()
        stored = dialogue_store.path(source)
        assert opens_inside(stored, source) and opens_inside(stored.as_uri() + '?mode=ro&immutable=1', source), \
            'premise: this observer sees a source database in both spellings, on this platform'
        assert opened and not [database for database in opened if opens_inside(database, source)]
        [(copied, rows)] = copies
        assert rows == 1 and copied.is_relative_to(platform_temp) and not copied.is_relative_to(root)
        assert server.poll() is None
        (root / 'state/server_process.json').unlink()
        assert alive(root)  # the admission alone still holds: the snapshot neither took nor broke it
    finally:
        platform_layer.file_unlock(lock)
        os.close(lock)
        writer.close()
        server.kill()
        server.wait()


def test_source_observers_place_each_spelling_by_path_components():
    """Each platform's rules on any host, not a native run: a plain path or a file URI lands in its root whatever
    as_uri encoded on the way; a sibling root, another drive and SQLite's in-memory name do not."""
    from pathlib import PurePosixPath, PureWindowsPath

    encoded = '%D0%90%D0%BD%D1%82%D0%BE%D0%BD%20%D0%A0'  # as_uri's spelling of 'Антон Р'
    windows, posix = PureWindowsPath('D:/a/Антон Р/data'), PurePosixPath('/a/Антон Р/data')
    for root, name, inside in (
        (windows, f'file:///D:/a/{encoded}/data/state/dialog.sqlite?mode=ro&immutable=1', True),
        (windows, f'file://localhost/d:/a/{encoded}/data/state/dialog.sqlite?mode=rw', True),
        (windows, (windows / 'state/dialog.sqlite').as_uri(), True),
        (windows, 'D:\\a\\Антон Р\\data\\state\\dialog.sqlite', True),
        (windows, f'file:///D:/a/{encoded}/data2/state/dialog.sqlite?mode=ro', False),
        (windows, 'D:\\a\\Антон Р\\data2\\state\\dialog.sqlite', False),
        (windows, f'file:///E:/a/{encoded}/data/state/dialog.sqlite', False),
        (windows, ':memory:', False),
        (posix, f'file:///a/{encoded}/data/state/dialog.sqlite?mode=ro&immutable=1', True),
        (posix, f'file:/a/{encoded}/data?mode=rw', True),
        (posix, (posix / 'state/dialog.sqlite').as_uri(), True),
        (posix, '/a/Антон Р/data/state/dialog.sqlite', True),
        (posix, posix / 'state/dialog.sqlite', True),
        (posix, os.fsencode('/a/Антон Р/data/state/dialog.sqlite'), True),
        (posix, f'file:///a/{encoded}/data2/state/dialog.sqlite?mode=ro', False),
        (posix, '/a/Антон Р/data2/state/dialog.sqlite', False),
        (posix, f'file:///b/{encoded}/data/state/dialog.sqlite', False),
        (posix, ':memory:', False),
    ):
        assert opens_inside(name, root) is inside, (root, name)


def test_store_writes_by_file_uri_are_refused_where_the_platform_places_them(tmp_path, monkeypatch):
    """The product's store writer opens SQLite by file URI (store.connect: path.as_uri() + '?mode=rw[c]'). It is
    refused here; then its Windows URIs, built and mapped by Windows rules on any host (not a native run), must
    land inside the guarded root as SQLite opens them there: file:///C:/... is C:\\..., not \\C:\\..."""
    import nturl2path
    from pathlib import PureWindowsPath

    from devtools import _context_fit_readonly as readonly
    from ouroboros import store

    (tmp_path / 'state').mkdir()
    before = fingerprint(tmp_path)
    with readonly_inputs(tmp_path), pytest.raises(PermissionError):
        store.connect(tmp_path / 'state/forbidden.sqlite', store.TIER_ENFORCED, create=True)
    assert fingerprint(tmp_path) == before and not list((tmp_path / 'state').iterdir())

    class WindowsRules(PureWindowsPath):
        def resolve(self):  # these names are absolute already
            return self
    root = WindowsRules('C:/Users/Антон Р/Ouroboros/data')
    monkeypatch.setattr(readonly, 'Path', WindowsRules)
    def map_windows_uri(path):
        if path == '/C:/no:place/x.sqlite':  # exercise a conversion failure, independent of the Python version
            raise OSError('scripted URI conversion failure')
        return nturl2path.url2pathname(path)
    monkeypatch.setattr(readonly, 'url2pathname', map_windows_uri)
    monkeypatch.setattr(readonly, '_GUARDED', {root})
    stored = root / 'state' / 'dialog.sqlite'
    for refused in (stored.as_uri() + '?mode=rw', stored.as_uri() + '?mode=rwc', stored.as_uri() + '?mode=ro',
                    'file://localhost/' + stored.as_posix() + '?mode=rw', str(stored),
                    'file:///C:/no:place/x.sqlite?mode=rw'):
        with pytest.raises(PermissionError):
            readonly._check_database(refused)
    readonly._check_database(stored.as_uri() + '?mode=ro&immutable=1')
    readonly._check_database(WindowsRules('D:/scratch/x.sqlite').as_uri() + '?mode=rwc')


def test_snapshot_includes_committed_wal_without_changing_original(tmp_path):
    root, copy = tmp_path / 'original', tmp_path / 'copy'
    (root / 'state').mkdir(parents=True)
    copy.mkdir()
    db = sqlite3.connect(root / 'state/dialog.sqlite')
    try:
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('CREATE TABLE sample (value)')
        db.execute('INSERT INTO sample VALUES (123)')
        db.commit()
        before = fingerprint(root)
        from ouroboros import dialogue_store
        with readonly_inputs(root), pytest.raises(ValueError, match='WAL input requires --snapshot'):
            with dialogue_store.transaction(root, write=False):
                pytest.fail('uncheckpointed WAL must not be read as immutable')
        snapshot_root(root, copy)
        assert fingerprint(root) == before
        with sqlite3.connect((copy / 'state/dialog.sqlite').as_uri() + '?mode=ro&immutable=1', uri=True) as frozen:
            assert frozen.execute('SELECT value FROM sample').fetchone() == (123,)
    finally:
        db.close()


def test_foreign_work_retains_server_pointer_in_scratch_not_inline_json(tmp_path):
    from ouroboros.context_health import _foreign_work_line
    from ouroboros import observability
    from unittest.mock import patch
    from dataclasses import replace
    from ouroboros.section_floor import HostSection

    root = scale.write_root(tmp_path / 'data')
    foreign = {f'task-{i:03d}': {'results': [f'run-{i}-{j}-' + 'x' * 68 for j in range(4)], 'patches': []}
               for i in range(66)}
    expected = _foreign_work_line(root, foreign)
    before = fingerprint(root)
    scratch = tmp_path / 'scratch'
    with readonly_inputs(root, scratch_root=scratch):
        actual = _foreign_work_line(root, foreign)
        assert actual == expected and 'retention unavailable' not in actual
        ref = observability.write_blob(root, foreign)
        assert observability.read_blob_ref(scratch, ref) == foreign
        assert str(scratch) in ref['path']
        from ouroboros import context_health
        assert context_health._stray_server_note(None) == ''
        with patch.object(observability, 'write_blob', side_effect=PermissionError('old diagnostic')):
            old = _foreign_work_line(root, foreign)
        def measured(line):
            core = scale.core(tmp_path / 'captures')
            core = replace(core, dynamic_sections=(*core.dynamic_sections, HostSection('health', line)))
            return measure_core(core, scale.tools(), root=root, window=92000, reserve=10000)
        old_fit, new_fit = measured(old), measured(actual)
    assert fingerprint(root) == before
    # The target's geometry/write-rights guidance also belongs in the diagnostic.
    assert len(old) > 24000 and len(actual) == 519
    assert old_fit['verdict'] == 'fits' and old_fit['work_room_deficit_tokens'] > 0, old_fit
    health = lambda fit: next(row['tokens'] for row in fit['blocks'] if row['block'] == 'health')
    assert health(old_fit) > health(new_fit) + 5000  # the walk may spend that saving on more whole memory
    assert new_fit['verdict'] == 'fits', new_fit
    assert new_fit['profile'] == 'task_local_nano'
    assert new_fit['fit']['measurement']['response_reserve_tokens'] == 10000
    print('FOREIGN_WORK_COMPARISON ' + json.dumps(dict(old_chars=len(old), new_chars=len(actual),
        before=old_fit, after=new_fit), sort_keys=True))


def test_session_report_uses_real_host_envelope_on_scale_root(tmp_path, monkeypatch):
    from tests._socket_seal import block_network_except_socketpair

    def forbidden(*args, **kwargs):
        pytest.fail('offline session report attempted a provider connection')
    # The report ends its host on an event loop, which Windows builds through connect(): only that self-pipe passes.
    block_network_except_socketpair(monkeypatch, forbidden)
    root = scale.write_root(tmp_path / 'data')
    offline_fixture(root)
    before = fingerprint(root)
    result = report(root, 1, 92000, 10000, session_provider='gemini_live')
    assert fingerprint(root) == before
    assert result['phase'] == 'session_candidate' and result['verdict'] == 'fits', result
    assert result['profile'] == 'task_local_nano'
    assert result['fit']['measurement']['capacity_total_tokens'] == 92000
    assert result['work_room_levels'] == 1
    assert sum(row['tokens'] for row in result['blocks']) == result['input_tokens']
    assert result['diagnostic_notes'] and result['session_provider'] == 'gemini_live'
    print('SCALE_SESSION_REPORT ' + json.dumps(result, sort_keys=True))
