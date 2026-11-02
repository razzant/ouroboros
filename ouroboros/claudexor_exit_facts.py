"""Bounded engine exit observations, capacity facts and route-health facts, without lifecycle policy.

Context reads only the existing supervisor log and the newest route-health row's
file. The status surface may sample an already authenticated, running endpoint; the
custody sweep's route-health refresh attaches to a serving engine; neither starts a
daemon. Observation timestamps name the host's harvest, never the unobserved death
time.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import pathlib
import re
import threading
from typing import Any

from ouroboros.utils import iter_jsonl_objects, utc_now_iso

log = logging.getLogger(__name__)
_RECENT_BYTES = 256_000  # Same bounded supervisor-log window as context health.
_CAPACITY_LOCK = threading.Lock()
_CAPACITY_SEEN: dict[str, str] = {}
ROUTE_HEALTH_TYPE = 'claudexor_route_health'
ROUTE_HEALTH_FILE = 'state/claudexor_route_health.json'  # the newest row, whole: what the vitals read
_ROUTE_HEALTH_MAX_BYTES = 262_144  # a read bound: a row with every route and three groups each is under 32 KB


def _recent(root):
    return iter_jsonl_objects(pathlib.Path(root) / 'logs/supervisor.jsonl', tail_bytes=_RECENT_BYTES)


def saved_facts(root) -> dict[str, Any]:
    """Historical observations in the recent byte window; no daemon access."""
    last, capacity, count = None, None, 0
    try:
        for row in _recent(root):
            if row.get('type') == 'claudexor_daemon_start_failed':
                count += 1
                last = {
                    'phase': 'serving' if row.get('descriptor_written') else 'startup',
                    'classification': row.get('classification') or 'unclassified',
                    'exit_signal': row.get('exit_signal'), 'exit_code': row.get('exit_code'),
                    'engine_version': row.get('pin_version') or None,
                    'observed_at': row.get('at') or row.get('ts') or None,
                }
            elif row.get('type') == 'claudexor_engine_capacity':
                capacity = row
    except Exception:
        log.debug('Engine observation log unavailable', exc_info=True)
    return {'last_exit': last, 'recent_exit_count': count, 'capacity': capacity}


def _bytes(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def memory_facts(payload) -> dict | None:
    """Unknown measurements stay null, including a legacy engine's absent route."""
    memory = payload.get('memory') if isinstance(payload, dict) else None
    if not isinstance(memory, dict):
        return None
    result = {key: _bytes(memory.get(key)) for key in (
        'heapUsedBytes', 'heapLimitBytes', 'rssBytes', 'externalBytes')}
    args = memory.get('nodeHeapArgs')
    result['nodeHeapArgs'] = args if isinstance(args, list) and all(
        isinstance(arg, str) and re.fullmatch(r'--max-old-space-size=\d{3,6}', arg, flags=re.ASCII)
        for arg in args) else None
    result['sampledAt'] = memory.get('sampledAt') if isinstance(memory.get('sampledAt'), str) else None
    admission = memory.get('atAdmission')
    result['atAdmission'] = ({'heapUsedBytes': _bytes(admission.get('heapUsedBytes')),
                              'rssBytes': _bytes(admission.get('rssBytes')),
                              'at': admission.get('at') if isinstance(admission.get('at'), str) else None}
                             if isinstance(admission, dict) else None)
    return result


def _number(value):
    return value if (isinstance(value, (int, float)) and not isinstance(value, bool)
                     and math.isfinite(value) and value >= 0) else None


def _numbers(value, keys, read=_number):
    return {key: read(value.get(key)) for key in keys} if isinstance(value, dict) else None


def _label(value, choices=None):
    return value if isinstance(value, str) and (choices is None or value in choices) else None


def _loop_facts(value):
    result = _numbers(value, ('windowMs', 'utilization'))
    if result is None:
        return None
    if result['utilization'] is not None and result['utilization'] > 1:
        result['utilization'] = None
    result['windowEndedAt'] = _label(value.get('windowEndedAt'))
    result['delay'] = _numbers(value.get('delay'), ('resolutionMs', 'p50Ms', 'p99Ms', 'maxMs'))
    if result['delay'] is not None:
        result['delay']['samples'] = _bytes(value['delay'].get('samples'))
    result['gc'] = _numbers(value.get('gc'), ('totalMs', 'maxMs'))
    if result['gc'] is not None:
        result['gc']['count'] = _bytes(value['gc'].get('count'))
    return result


def _store_facts(value):
    result = _numbers(value, ('flush_lag_ms', 'interval_ms'))
    if result is None:
        return None
    result.update(_numbers(value, ('wal_bytes', 'busy_waits', 'obligations_open'), _bytes))
    result['last_barrier_at'] = _label(value.get('last_barrier_at'))
    result['integrity'] = _label(value.get('integrity'), ('pending', 'ok', 'failed'))
    flusher = value.get('flusher')
    result['flusher'] = _numbers(flusher, ('interval_ms', 'flush_lag_ms'))
    if result['flusher'] is not None:
        result['flusher'].update(
            state=_label(flusher.get('state'), ('starting', 'up', 'down')),
            last_barrier_at=_label(flusher.get('last_barrier_at')),
            **_numbers(flusher, ('generation', 'acknowledged_generation', 'pending_registrations', 'pending_waiters'), _bytes))
    migration = value.get('migration')
    result['migration'] = _numbers(migration,
        ('completedPartitions', 'totalPartitions', 'processedBytes', 'totalBytes'), _bytes)
    if result['migration'] is not None:
        result['migration'].update(
            phase=_label(migration.get('phase'), ('reading', 'importing', 'verifying', 'complete', 'publishing', 'failed')),
            currentPartition=_label(migration.get('currentPartition')))
    return result


def _capacity_facts(value):
    if not isinstance(value, dict):
        return None
    # Only actual positive safe integers or the engine's literal certify a cap.
    # The effective snapshot has no configured/restartRequired authority.
    result = {}
    for key in ('maxConcurrent', 'maxConcurrentNonModelJobs', 'maxConcurrentModelOperations'):
        cap = value.get(key)
        result[key] = cap if (cap == 'unlimited' or type(cap) is int and 0 < cap <= 2**53 - 1) else None
    sources = value.get('sources')
    result['sources'] = ({key: _label(sources.get(key), ('default', 'config', 'environment', 'embedder', 'unknown'))
                          for key in ('max_concurrent', 'max_concurrent_non_model_jobs', 'max_concurrent_model_operations')}
                         if isinstance(sources, dict) else None)
    return result


def runtime_facts(payload):
    """Selected status measurements, never admission or a second settings view.

    Missing/invalid measurements remain null. Old engines need no new route,
    and the store's unopened SQL metrics are not healthy zeroes. The capacity
    log retains one dated snapshot; each status read serves its own fresh facts.
    """
    payload = payload if isinstance(payload, dict) else {}
    admission = payload.get('admission')
    return {
        'servingMode': _label(payload.get('servingMode'), ('normal', 'recovery_only')),
        'loop': _loop_facts(payload.get('loop')),
        'store': _store_facts(payload.get('store')),
        'capacity': _capacity_facts(payload.get('capacity')),
        'admission': ({key: _numbers(admission.get(key), ('model', 'non_model'), _bytes)
                       for key in ('active', 'queued')} if isinstance(admission, dict) else None),
    }


def _record_capacity(root, generation, version, build, memory, runtime=None):
    """One observation per generation, serialized with the log's existing lock.

    The small process memo survives log rotation. Under the append lock, a
    bounded read also deduplicates other readers/processes while the row remains
    in the recent window. After both process restart and log eviction, an old
    generation can be observed again; this is diagnostic history, not authority.
    """
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
    from ouroboros.utils import assert_test_data_path, jsonl_append_lock_path

    key = str(pathlib.Path(root).resolve())
    with _CAPACITY_LOCK:
        if _CAPACITY_SEEN.get(key) == generation:
            return
        path = pathlib.Path(root) / 'logs/supervisor.jsonl'
        assert_test_data_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = jsonl_append_lock_path(path)
        fd = acquire_exclusive_file_lock(lock_path, timeout_sec=2, owner_aware_stale=True)
        if fd is None:
            return
        try:
            if not any(row.get('type') == 'claudexor_engine_capacity'
                       and row.get('generation') == generation for row in _recent(root)):
                memory = memory or {}
                admission = memory.get('atAdmission') or {}
                row = {'ts': utc_now_iso(), 'type': 'claudexor_engine_capacity',
                       'generation': generation, 'engine_version': version or None,
                       'engine_build_sha': build or None,
                       'heap_limit_bytes': memory.get('heapLimitBytes'),
                       'admission_heap_used_bytes': admission.get('heapUsedBytes'),
                       'admission_at': admission.get('at'), 'node_heap_args': memory.get('nodeHeapArgs'),
                       **(runtime or {})}
                # append_jsonl takes this same non-reentrant lock. Write one
                # encoded row here so read/deduplicate/append are one interval.
                with path.open('ab') as stream:
                    stream.write((json.dumps(row) + '\n').encode('utf-8'))
            _CAPACITY_SEEN[key] = generation
        finally:
            release_exclusive_file_lock(lock_path, fd)


def route_health_digest(facts) -> str:
    """What the mind reads: the routes and the scan's completeness. The clocks and the scanned count
    move with every read and are not a change."""
    material = {key: facts.get(key) for key in ('routes', 'complete', 'incomplete_reason', 'window_days')}
    return hashlib.sha256(json.dumps(material, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def latest_route_health(root) -> dict[str, Any] | None:
    """The newest route-health row, kept whole in ``ROUTE_HEALTH_FILE``: it outlives the supervisor
    log's bounded tail and its rotation, and a fresh process reads the same row. None when absent,
    unreadable, over the read bound or not such a row: no fact, never an empty one."""
    path = pathlib.Path(root) / ROUTE_HEALTH_FILE
    try:
        with path.open('rb') as stream:
            data = stream.read(_ROUTE_HEALTH_MAX_BYTES + 1)
        row = json.loads(data.decode('utf-8')) if len(data) <= _ROUTE_HEALTH_MAX_BYTES else None
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError, ValueError):
        log.debug('Route health fact unreadable', exc_info=True)
        return None
    return row if isinstance(row, dict) and row.get('type') == ROUTE_HEALTH_TYPE else None


def record_route_health(root, facts, *, engine_version='') -> bool:
    """Record the engine's facts when they differ from the newest recorded row; True when written. A row
    is the engine's facts (``gateways.claudexor_route_health.route_health_facts``) plus ``ts``,
    ``engine_version`` and the ``digest`` the next refresh compares. A change appends the row to
    ``logs/supervisor.jsonl`` as history, then replaces ``ROUTE_HEALTH_FILE`` with it; that file is the
    comparison, so an unchanged refresh in this or any later process writes nothing, and an evicted or
    rotated history row changes nothing. Both writes happen under the log's append lock, as
    ``_record_capacity``'s do; a failed replace leaves the old digest, so the next refresh writes again."""
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
    from ouroboros.utils import assert_test_data_path, atomic_write_json, jsonl_append_lock_path

    if not isinstance(facts, dict) or not isinstance(facts.get('routes'), list):
        return False
    digest = route_health_digest(facts)
    with _CAPACITY_LOCK:
        if (latest_route_health(root) or {}).get('digest') == digest:
            return False
        path = pathlib.Path(root) / 'logs/supervisor.jsonl'
        assert_test_data_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = jsonl_append_lock_path(path)
        fd = acquire_exclusive_file_lock(lock_path, timeout_sec=2, owner_aware_stale=True)
        if fd is None:
            return False
        try:
            if (latest_route_health(root) or {}).get('digest') == digest:  # another process wrote it
                return False
            row = {'ts': utc_now_iso(), 'type': ROUTE_HEALTH_TYPE, 'engine_version': engine_version or None,
                   'digest': digest, **facts}
            with path.open('ab') as stream:  # one encoded row under the same lock as the read above
                stream.write((json.dumps(row) + '\n').encode('utf-8'))
            state = pathlib.Path(root) / ROUTE_HEALTH_FILE
            state.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(state, row)
            return True
        finally:
            release_exclusive_file_lock(lock_path, fd)


def refresh_route_health(open_gateway=None):
    """Read the engine's per-route failure shares and record them when they changed; the facts, or None
    when no engine answered. ``open_gateway`` returns an authenticated gateway this call closes; without
    one the owned engine is attached read-only (``read_owned_gateway``: never a spawn, so a fact is never
    worth a start). The custody sweep's step (ARCHITECTURE §9). Never raises."""
    from ouroboros.config import DATA_DIR
    from ouroboros.gateways.claudexor_route_health import read_route_health

    try:
        if open_gateway is None:
            from ouroboros.claudexor_daemon import read_owned_gateway

            open_gateway = read_owned_gateway
        with contextlib.closing(open_gateway()) as gateway:
            facts = read_route_health(gateway)
            version = str(getattr(gateway, 'engine_version', '') or '')
    except Exception:
        log.debug('Route health refresh skipped: no engine to attach', exc_info=True)
        return None
    if facts is not None:
        try:
            record_route_health(DATA_DIR, facts, engine_version=version)
        except Exception:
            log.debug('Route health row not written', exc_info=True)
    return facts


def engine_status_facts(endpoint, version, build) -> dict[str, Any]:
    """Read engine measurements once; an absent endpoint stays offline."""
    from ouroboros.config import DATA_DIR

    facts = {'last_exit': saved_facts(DATA_DIR)['last_exit'], 'memory': None, **runtime_facts(None)}
    if endpoint is None:
        return facts
    try:
        from ouroboros.claudexor_daemon import owned_descriptor_path
        from ouroboros.gateways.claudexor import ClaudexorGateway, ClaudexorUnavailable, SHORT_POLL_TIMEOUT_SEC

        before = owned_descriptor_path().stat()
        with ClaudexorGateway(endpoint) as gateway:
            try:
                payload = gateway.daemon_status(timeout_sec=SHORT_POLL_TIMEOUT_SEC)
            except ClaudexorUnavailable as exc:
                if exc.status_code != 404:
                    raise
                payload = {}  # Old engines have no status route.
        after = owned_descriptor_path().stat()
        identity = lambda st: (st.st_dev, st.st_ino, st.st_mtime_ns, st.st_size)
        if identity(before) != identity(after):
            return facts  # The descriptor changed during the read; no mixed generation.
        facts['memory'] = memory_facts(payload)
        observed = runtime_facts(payload)
        facts.update(observed)
        generation = hashlib.sha256(json.dumps([
            endpoint.host, endpoint.port, version, build, identity(after),
        ]).encode('utf-8')).hexdigest()
        _record_capacity(DATA_DIR, generation, version, build, facts['memory'], observed)
    except Exception:
        log.debug('Engine capacity observation unavailable', exc_info=True)
    return facts


def health_line(root) -> str:
    """One historical fact line, with no notification or behavioral instruction."""
    facts = saved_facts(root)
    last = facts['last_exit']
    if not last:
        return ''
    exit_fact = (f"signal {last['exit_signal']}" if last['exit_signal'] is not None
                 else f"exit code {last['exit_code']}" if last['exit_code'] is not None else 'exit unknown')
    text = (f"WARNING: ENGINE EXITED — managed Claudexor {last['engine_version'] or 'unknown version'} "
            f"exited {facts['recent_exit_count']} time(s) in the recent log "
            f"(last: {last['classification']}, {exit_fact}, while {last['phase']}; "
            f"observed at {last['observed_at'] or 'unknown time'}, not the death time)")
    capacity = facts['capacity'] or {}
    amounts = []
    for field, label in (('heap_limit_bytes', 'heap limit'), ('admission_heap_used_bytes', 'admission heap')):
        value = _bytes(capacity.get(field))
        if value is not None:
            amounts.append(f'{label} {value / 2**30:.2f} GiB')
    if amounts:
        text += f"; last observed capacity (engine {capacity.get('engine_version') or 'unknown'}): " + ', '.join(amounts)
    return text + '.'
