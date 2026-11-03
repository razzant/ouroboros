#!/usr/bin/env python3
"""OB#1234 outside sampler (v3): root on the disposable runner, outside the observed tree.

It never starts, wraps or signals what it observes. It only reads /proc of processes
that live in the container's PID namespace and whose effective UID (status ``Uid:``) is --uid:

* identity: host PID, starttime, NSpid, PPid, the parent's starttime read at identification
  (None = link unknown), exe, argv, selected loader/bus env; a failed read is kept as an errno;
* re-identification: exe is re-read on every poll; when it changes (fork -> exec,
  e.g. node -> Chromium) the old image is moved to ``images`` and the record is
  re-identified, so a pre-exec copy of node is never labelled Chromium. argv of
  Chromium processes is re-read too, because zygote children rewrite it after fork;
* maps: ONLY for Chromium processes (exe under ``.local-browsers/`` or named chrome /
  chrome-headless-shell / headless_shell), re-read every --maps-every seconds while the
  process lives; union of mapped file paths with first-seen time;
* disappearance: when the PID is gone the time it was first NOT seen is recorded.
  That is not an exit status. No exit status of any process is witnessed by this tool.
* gaps: failed /proc listings, processes whose namespace or status could not be read
  (``unclassified``), failed exe re-reads (the last known identity is kept but not
  re-confirmed) and the largest interval between polls; ``limits`` in sampler-meta.json.

Stops on --stop-file, SIGTERM/SIGINT or --ttl (finite by construction). Output under
--out: sampler.identity.json (own PID/starttime, written first), processes.json and
sampler-meta.json (written at stop). Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path

VALUE_KEYS = {
    'LD_LIBRARY_PATH', 'LD_LIBRARY_PATH_ORIG', 'LD_PRELOAD', 'OUROBOROS_DESKTOP_ENV_SAVED',
    'PYTHONHOME', 'PYTHONPATH', 'PYTHONUSERBASE', 'PYTHONNOUSERSITE', 'PLAYWRIGHT_BROWSERS_PATH',
    'OUROBOROS_BUNDLE_DIR', 'OUROBOROS_PRESENTATION', 'DEBUG', 'XDG_RUNTIME_DIR', 'XDG_DATA_DIRS',
    'DISPLAY', 'WAYLAND_DISPLAY',
}
VALUE_PREFIXES = ('OUROBOROS_DESKTOP_HOST_', 'GI_', 'GIO_', 'DBUS_')
CHROMIUM_NAMES = ('chrome', 'chrome-headless-shell', 'headless_shell')
LIMITS = ['periodic sampling: processes and mappings that come and go between two polls are not observed',
          'only processes whose own PID namespace is the container namespace are read; nested PID namespaces are not observed',
          'exit statuses are not witnessed; gone_seen_at is the first poll that no longer saw the process',
          'PPid and the parent starttime are read once per image; starttime has clock-tick resolution, so a reuse within one tick is not detected']


def chromium_like(exe) -> bool:
    exe = exe or ''  # None: never readable
    return '/.local-browsers/' in exe or exe.rsplit('/', 1)[-1] in CHROMIUM_NAMES


def starttime(pid: int):
    try:
        return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        return None


def read_argv(pid: int):
    # Strip trailing NULs only: a rewritten cmdline may lack the final one and must keep its text.
    text = Path(f'/proc/{pid}/cmdline').read_bytes().rstrip(b'\0').decode('utf-8', 'backslashreplace')
    return text.split('\0') if text else []


def read_status(pid: int) -> dict:
    return dict(line.split(':', 1) for line in Path(f'/proc/{pid}/status').read_text().splitlines() if ':' in line)


class Sampler:
    def __init__(self, ns_pid: int, uid: int, out: Path, interval: float, maps_every: float, ttl: float):
        self.ns_ino = os.stat(f'/proc/{ns_pid}/ns/pid').st_ino
        self.uid, self.out, self.interval, self.maps_every, self.ttl = uid, out, interval, maps_every, ttl
        self.records: dict[str, dict] = {}   # "pid:starttime" -> record
        self.live: dict[int, str] = {}        # pid -> key
        self.other_ns: set[str] = set()       # "pid:starttime" in another PID namespace (a process's own never changes)
        self.unclassified: dict[str, dict] = {}  # "pid:starttime" -> failed namespace/status read: ours or not is unknown
        self.last_poll = None
        self.meta = {'ns_pid': ns_pid, 'pid_ns_inode': self.ns_ino, 'uid': uid, 'started': time.time(),
                     'kernel': os.uname().release, 'interval': interval, 'maps_every': maps_every, 'ttl': ttl,
                     'polls': 0, 'stop_reason': None, 'max_poll_gap_s': 0.0, 'list_errors': [], 'stat_unread': 0, 'limits': LIMITS,
                     'exit_status': 'NOT WITNESSED: this sampler records disappearance only'}
        self.stop = False

    def classify(self, pid: int, start: int, key: str):
        """A new record if the process is ours, else None. A failed read is a recorded gap, never 'not ours'."""
        try:
            if os.stat(f'/proc/{pid}/ns/pid').st_ino != self.ns_ino:
                self.other_ns.add(key)
                return None
            status = read_status(pid)
            euid = int(status['Uid'].split()[1])  # as ps -u and pkill -u; the /proc inode owner can differ
            ident = self.identity(pid, start, status) if euid == self.uid else None
        except (OSError, KeyError, IndexError, ValueError) as exc:
            self.unclassified.setdefault(key, {'at': time.time(), 'error': repr(exc)})
            return None
        self.unclassified.pop(key, None)
        if ident is None:  # not ours now; read again next poll, because a UID can change
            return None
        rec = {'key': key, 'host_pid': pid, 'starttime': start, 'first_seen': time.time(),
               'images': [], 'argv_history': [], 'maps': {}, 'maps_reads': 0, 'maps_attempts': 0,
               'maps_errors': [], 'last_maps': 0.0, 'exe_reread_failures': None, 'gone_seen_at': None, **ident}
        self.records[key] = rec
        self.live[pid] = key
        return rec

    def identity(self, pid: int, start: int, status: dict) -> dict:
        proc = Path(f'/proc/{pid}')
        read_errors, exe, argv = {}, None, None
        try:
            exe = os.readlink(proc / 'exe')
        except OSError as exc:
            read_errors['exe'] = exc.errno
        try:
            argv = read_argv(pid)
        except OSError as exc:
            read_errors['cmdline'] = exc.errno
        ppid = int((status.get('PPid') or '0').strip() or 0)
        parent_start = starttime(ppid) if ppid else None
        env, env_error = {}, None
        try:
            for part in (proc / 'environ').read_bytes().split(b'\0'):
                if b'=' in part:
                    k, v = part.decode('utf-8', 'backslashreplace').split('=', 1)
                    env[k] = v
        except OSError as exc:
            env_error = exc.errno  # unknown is not absent
        return {'ns_pid': (status.get('NSpid', '').split() or [''])[-1],
                'ppid': ppid, 'comm': status.get('Name', '').strip(), 'uid': status.get('Uid', '').split(),
                # A parent is never younger than its child: a younger process on that PID is a reused PID.
                'parent_starttime': parent_start if parent_start is not None and parent_start <= start else None,
                'exe': exe, 'argv': argv, 'read_errors': read_errors, 'identified_at': time.time(),
                'environ_readable': env_error is None, 'environ_errno': env_error,
                'env_names': sorted(env),
                'env_values': {k: v for k, v in env.items() if k in VALUE_KEYS or k.startswith(VALUE_PREFIXES)}}

    def read_maps(self, pid: int, rec: dict):
        rec['maps_attempts'] += 1
        try:
            text = Path(f'/proc/{pid}/maps').read_text(errors='backslashreplace')
        except OSError as exc:
            rec['maps_errors'].append(exc.errno)
            return
        now = time.time()
        for line in text.splitlines():
            parts = line.split(None, 5)
            if len(parts) == 6 and parts[5].startswith('/'):
                rec['maps'].setdefault(parts[5], now)
        rec['maps_reads'] += 1
        rec['last_maps'] = now
        try:
            argv = read_argv(pid)
            if argv and argv != rec['argv']:
                rec['argv_history'].append({'at': now, 'argv': argv})
                rec['argv'] = argv
        except OSError as exc:
            rec['read_errors']['cmdline_reread'] = exc.errno  # argv stays the last value read

    def recheck(self, pid: int, rec: dict):
        """A failed exe re-read keeps the last known identity and records that it was not re-confirmed."""
        try:
            exe = os.readlink(f'/proc/{pid}/exe')
            if exe == rec['exe']:
                return
            fresh = self.identity(pid, rec['starttime'], read_status(pid))
        except (OSError, ValueError) as exc:
            seen = rec['exe_reread_failures'] or {'count': 0, 'first_at': time.time()}
            rec['exe_reread_failures'] = seen | {'count': seen['count'] + 1, 'last_at': time.time(), 'last_error': repr(exc)}
            return
        # exec (or a first readable exe): never carry the previous image's facts forward
        rec['images'].append({k: rec[k] for k in ('exe', 'argv', 'env_values', 'identified_at')}
                             | {'maps_paths': len(rec['maps'])})
        rec['maps'], rec['maps_reads'], rec['maps_attempts'] = {}, 0, 0
        rec['maps_errors'], rec['argv_history'], rec['last_maps'] = [], [], 0.0
        rec.update(fresh)

    def poll(self):
        self.meta['polls'] += 1
        tick = time.monotonic()
        if self.last_poll is not None:
            self.meta['max_poll_gap_s'] = max(self.meta['max_poll_gap_s'], round(tick - self.last_poll, 3))
        self.last_poll = tick
        try:
            current = {int(n) for n in os.listdir('/proc') if n.isdigit()}
        except OSError as exc:
            self.meta['list_errors'].append({'at': time.time(), 'errno': exc.errno})
            return
        for pid in current:
            start = starttime(pid)
            if start is None:
                self.meta['stat_unread'] += 1  # gone before it could even be keyed
                continue
            key = f'{pid}:{start}'
            if key in self.other_ns:
                continue
            rec = self.records.get(key)
            if rec is None:
                rec = self.classify(pid, start, key)
                if rec is None:
                    continue
            else:
                self.recheck(pid, rec)
            if chromium_like(rec['exe']) and time.time() - rec['last_maps'] >= self.maps_every:
                self.read_maps(pid, rec)
        now = time.time()
        for pid, key in list(self.live.items()):
            born = starttime(pid) if pid in current else None
            if pid not in current or (born is not None and born != self.records[key]['starttime']):
                self.records[key]['gone_seen_at'] = now
                del self.live[pid]

    def run(self, stop_file: Path):
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: setattr(self, 'stop', True))
        (self.out / 'sampler.identity.json').write_text(json.dumps(
            {'pid': os.getpid(), 'starttime': starttime(os.getpid()), 'ttl': self.ttl, 'started': self.meta['started']}))
        deadline = time.monotonic() + self.ttl
        while True:
            if self.stop:
                self.meta['stop_reason'] = 'signal'
                break
            if stop_file.exists():
                self.meta['stop_reason'] = 'stop_file'
                break
            if time.monotonic() >= deadline:
                self.meta['stop_reason'] = 'ttl'
                break
            self.poll()
            time.sleep(self.interval)
        self.poll()
        self.meta['stopped'] = time.time()
        records = list(self.records.values())
        for rec in records:
            rec['maps'] = [{'path': p, 'first_seen': t} for p, t in sorted(rec['maps'].items())]
            rec.pop('last_maps', None)
            rec['chromium'] = chromium_like(rec['exe'])
        self.meta['records'] = len(records)
        self.meta['still_live_at_stop'] = sorted(self.live.values())
        self.meta['unclassified'], self.meta['other_ns_processes'] = self.unclassified, len(self.other_ns)
        (self.out / 'processes.json').write_text(json.dumps(records, indent=1))
        (self.out / 'sampler-meta.json').write_text(json.dumps(self.meta, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ns-pid', type=int, required=True, help='host PID of any process in the container PID namespace')
    ap.add_argument('--uid', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--stop-file', type=Path, required=True)
    ap.add_argument('--ttl', type=float, required=True, help='hard upper bound in seconds')
    ap.add_argument('--interval', type=float, default=0.05)
    ap.add_argument('--maps-every', type=float, default=0.5)
    args = ap.parse_args()
    assert os.geteuid() == 0, 'reads other users\' /proc: run as root on the disposable runner'
    args.out.mkdir(parents=True, exist_ok=True)
    Sampler(args.ns_pid, args.uid, args.out, args.interval, args.maps_every, args.ttl).run(args.stop_file)


if __name__ == '__main__':
    main()
