"""Bounded engine exit observations and capacity facts, without lifecycle policy.

Context reads only the existing supervisor log. The status surface may sample an
already authenticated, running endpoint; it never discovers or starts a daemon.
Observation timestamps name the host's harvest, never the unobserved death time.
"""
from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import re
import threading
from typing import Any

from ouroboros.utils import iter_jsonl_objects, utc_now_iso

log = logging.getLogger(__name__)
_RECENT_BYTES = 256_000  # Same bounded supervisor-log window as context health.
_CAPACITY_LOCK = threading.Lock()
_CAPACITY_SEEN: dict[str, str] = {}


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


def _record_capacity(root, generation, version, build, memory):
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
                       'admission_at': admission.get('at'), 'node_heap_args': memory.get('nodeHeapArgs')}
                # append_jsonl takes this same non-reentrant lock. Write one
                # encoded row here so read/deduplicate/append are one interval.
                with path.open('ab') as stream:
                    stream.write((json.dumps(row) + '\n').encode('utf-8'))
            _CAPACITY_SEEN[key] = generation
        finally:
            release_exclusive_file_lock(lock_path, fd)


def engine_status_facts(endpoint, version, build) -> dict[str, Any]:
    """Add last exit and live memory to status; an absent endpoint stays offline."""
    from ouroboros.config import DATA_DIR

    facts = {'last_exit': saved_facts(DATA_DIR)['last_exit'], 'memory': None}
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
        generation = hashlib.sha256(json.dumps([
            endpoint.host, endpoint.port, version, build, identity(after),
        ]).encode('utf-8')).hexdigest()
        _record_capacity(DATA_DIR, generation, version, build, facts['memory'])
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
