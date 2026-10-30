"""Read-only adapters for an offline diagnostic, never application store policy."""
from contextlib import ExitStack, contextmanager, closing
import ctypes
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
from unittest.mock import patch


def server_alive(root):
    for relative in ('state/server_process.json', 'server.pid', 'state/server.pid', 'ouroboros.pid'):
        path = root / relative
        if not path.exists():
            continue
        try:
            value = json.loads(path.read_text())
            pid = int(value.get('pid', 0) if isinstance(value, dict) else value)
            if pid > 0:
                os.kill(pid, 0)
                return True
        except ProcessLookupError:
            pass
        except (OSError, ValueError, TypeError):
            return True  # Unreadable admission evidence is not proof of an offline root.
    lock = root / 'state/server.lock'
    if lock.exists():
        import fcntl
        with lock.open('rb') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
                fcntl.flock(handle, fcntl.LOCK_UN)
            except OSError:
                return True
    return False


def _clone(source, destination):
    if sys.platform == 'darwin':
        clone = ctypes.CDLL(None, use_errno=True).clonefile
        if clone(os.fsencode(source), os.fsencode(destination), 0) == 0:
            return
    shutil.copyfile(source, destination)


def snapshot_root(source, destination):
    """Copy only context inputs; stabilize SQLite/WAL on the private copy.

    No source connection, checkpoint, lock file creation or journal change.
    Retry a changing file set; fail rather than present a torn snapshot as fact.
    """
    def inputs():
        paths = [source / 'settings.json', source / 'projects.json']
        for directory in ('memory', 'state', 'knowledge', 'skills', 'projects'):
            base = source / directory
            if base.exists():
                paths.extend(path for path in base.rglob('*') if path.is_file())
        paths.extend((source / 'logs').glob('*.jsonl'))
        return sorted(path for path in paths if path.is_file() and not path.name.endswith(('.lock', '.pid')))
    def stamp(paths):
        return [(str(path), path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns) for path in paths]
    for _attempt in range(3):
        paths = inputs()
        before = stamp(paths)
        for path in paths:
            if path.is_symlink() or not path.resolve().is_relative_to(source):
                raise ValueError('Snapshot refuses symlinked context inputs')
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.unlink(missing_ok=True)
            _clone(path, target)
        if before == stamp(inputs()):
            break
        shutil.rmtree(destination)
        destination.mkdir()
    else:
        raise ValueError('Context inputs kept changing during snapshot; retry on an offline copy')
    for wal in destination.rglob('*-wal'):
        if not wal.stat().st_size:
            continue
        database = Path(str(wal)[:-4])
        frozen = database.with_name(database.name + '.frozen')
        # WAL is read only on our copy; the output is a new private database.
        with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as src:
            with closing(sqlite3.connect(frozen)) as dst:
                src.backup(dst)
        database.unlink()
        frozen.replace(database)
        wal.unlink(missing_ok=True)
        Path(str(database) + '-shm').unlink(missing_ok=True)


_GUARDED = set()


def _check_database(database):
    from urllib.parse import unquote, urlsplit
    database = str(database)
    path = Path(unquote(urlsplit(database).path) if database.startswith('file:') else database).resolve()
    if any(path == root or path.is_relative_to(root) for root in _GUARDED) and 'mode=ro&immutable=1' not in database:
        raise PermissionError('Diagnostic requires immutable read-only SQLite inputs')


def _audit(event, args):
    if not _GUARDED:
        return
    paths = ()
    if event == 'sqlite3.connect':
        _check_database(args[0])
    if event == 'open':
        path, mode, flags = args
        if (mode and any(c in mode for c in 'wax+')) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
            paths = (path,)
    elif event in {'os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod', 'os.utime', 'os.truncate'}:
        paths = args[:1]
    elif event in {'os.rename', 'os.link', 'os.symlink'}:
        paths = args[:2]
    for path in paths:
        if isinstance(path, (str, bytes, os.PathLike)):
            absolute = Path(os.fsdecode(path)).resolve()
            if any(absolute == root or absolute.is_relative_to(root) for root in _GUARDED):
                raise PermissionError('context_fit_report forbids writes under the input root')


sys.addaudithook(_audit)


@contextmanager
def readonly_inputs(root, *, source_root=None, scratch_root=None):
    """Use immutable SQLite readers; skip migration, activation and name-lock writes."""
    from ouroboros import config, dialogue_store, store, capability_evidence, usage_store, platform_layer, context_health
    from ouroboros import settings_integrity, mcp_client, observability
    from ouroboros.chronicle_store import ChronicleStore
    from ouroboros.memory import Memory
    from supervisor import state

    sqlite_connect = sqlite3.connect
    def checked_connect(database, *args, **kwargs):
        # Some packaged SQLite builds omit the native sqlite3.connect audit event.
        _check_database(database)
        return sqlite_connect(database, *args, **kwargs)

    def connect(path):
        path = Path(path).resolve()
        wal = Path(str(path) + '-wal')
        if wal.exists() and wal.stat().st_size:
            raise ValueError('WAL input requires --snapshot; immutable readers must not omit committed frames')
        journal = Path(str(path) + '-journal')
        if journal.exists():
            with journal.open('rb') as source:
                if source.read(8).strip(b'\0'):
                    raise ValueError('A hot rollback journal requires recovery on a private offline copy')
        db = sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA query_only=ON')
        return db

    @contextmanager
    def current(data, *, write=False, **kwargs):
        if write:
            raise PermissionError('Diagnostic cannot write current facts')
        with closing(connect(store.store_path(data))) as db:
            yield store.Txn(db, data)

    @contextmanager
    def dialogue(data, *, write=False, **kwargs):
        if write:
            raise PermissionError('Diagnostic cannot write dialogue')
        with closing(connect(dialogue_store.path(data))) as db:
            yield db

    @contextmanager
    def usage(data, **kwargs):
        with closing(connect(Path(data) / usage_store.STORE_REL)) as db:
            yield usage_store.Txn(db, Path(data), write=False)

    @contextmanager
    def chronicle(self):
        with closing(connect(self.index_path)) as db:
            if self._state(db, 'offset', 0) != self.log_path.stat().st_size:
                raise ValueError('Chronicle index is stale; prepare an offline snapshot with the application first')
            yield db

    acquire_lock = platform_layer.acquire_exclusive_file_lock
    def no_source_lock(path, *args, **kwargs):
        if Path(path).resolve().is_relative_to(root):
            raise PermissionError('Diagnostic cannot create a source lock')
        return acquire_lock(path, *args, **kwargs)

    guarded = {root.resolve(), Path(source_root).resolve() if source_root is not None else root.resolve()}
    added = guarded - _GUARDED
    _GUARDED.update(guarded)
    try:
        with ExitStack() as stack:
            scratch = Path(scratch_root) if scratch_root is not None else Path(stack.enter_context(
                tempfile.TemporaryDirectory(prefix='context-fit-blobs-', dir='/tmp')))
            if any(scratch.resolve().is_relative_to(path) for path in guarded):
                raise ValueError('Diagnostic scratch must be outside the measured root')
            write_blob = observability.write_blob
            for target, name, value in (
                (sqlite3, 'connect', checked_connect), (sqlite3.dbapi2, 'connect', checked_connect),
                (store, 'open_store', current), (dialogue_store, 'transaction', dialogue),
                (usage_store, 'read', usage),
                (ChronicleStore, '_index', chronicle),
                (ChronicleStore, 'ensure_activated', lambda self, **kw: self.activation()),
                (Memory, 'ensure_files', lambda self: None), (config, 'DATA_DIR', root),
                (config, 'SETTINGS_PATH', root / 'settings.json'),
                (config, 'load_settings', lambda: config.load_settings_lock_held(_settings_lock_held=False)),
                (platform_layer, 'acquire_exclusive_file_lock', no_source_lock),
                (state, 'DRIVE_ROOT', root),
                # Retain the same content-addressed pointer the server renders. Only the
                # destination changes; neither fallback JSON nor diagnostic notices enter P.
                (observability, 'write_blob', lambda data, payload, **kw: write_blob(scratch, payload, **kw)),
                (context_health, '_stray_server_note', lambda env: ''),
                (capability_evidence, 'canonical_evidence_root', lambda: root),
                (mcp_client, '_manager', mcp_client.MCPManager()),
            ):
                stack.enter_context(patch.object(target, name, value))
            # Saved settings belong to the inspected root, not the invoking shell.
            with settings_integrity.settings_read_environ({}):
                settings = config.load_settings()
            from ouroboros.skill_sources import load_sources, SOURCES_SETTING
            sources = load_sources(settings)
            for row in sources:
                if row['kind'] == 'local':
                    path = Path(os.path.abspath(os.path.expanduser(row['path'])))
                    base = source_root or root
                    if not path.is_relative_to(base):
                        raise ValueError('Local skill sources outside the data root cannot be measured offline')
                    row['path'] = str(root / path.relative_to(base))
            settings[SOURCES_SETTING] = sources
            local = settings.get('OUROBOROS_SKILLS_REPO_PATH')
            if local:
                settings['OUROBOROS_SKILLS_REPO_PATH'] = str(root / Path(local).relative_to(source_root or root))
            projected = {key: value for key, value in os.environ.items() if key not in config.settings_env_keys()}
            config.apply_settings_to_env(settings, environ=projected)
            stack.enter_context(patch.dict(os.environ, projected, clear=True))
            stack.enter_context(patch.object(config, 'load_settings', lambda: dict(settings)))
            stack.enter_context(config.task_settings_scope(settings_integrity.task_settings_snapshot(settings, projected)))
            yield
    finally:
        _GUARDED.difference_update(added)
