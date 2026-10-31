"""Run one bounded lab command, retaining create-only streams before dispatch.

A used or partially written label is never reused. A timeout ends this local
client, not a proof that a guest operation stopped; inspect before any new act.
The command writes into live spools LABEL.stdout.live/.stderr.live, which a
descendant that kept the descriptors may extend after this client returns. At
that return the length of each spool is fixed first; exactly those bytes are
copied once into the read-only snapshots LABEL.stdout/.stderr, and the receipt
hashes the copied bytes. The spools stay beside them as possibly growing
observations, never as the receipt's evidence. Only the first 6000 bytes of
each snapshot are printed.
"""
import contextlib
import datetime
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
EVIDENCE = pathlib.Path(os.environ.get('LAB_EVIDENCE') or HERE.parent / 'evidence')
TIMEOUT_NOTE = 'CAPTURE_TIMEOUT: guest/external outcome unknown; inspect before retry'


def snapshot(spool, target):
    """Copy the spool's bytes present now (length fixed before reading) into a new read-only file."""
    spool.flush()
    length = os.fstat(spool.fileno()).st_size
    digest = hashlib.sha256()
    copied = 0
    with open(spool.name, 'rb') as source, target.open('xb') as copy:
        while copied < length:
            block = source.read(min(1 << 20, length - copied))
            if not block:
                break
            digest.update(block)
            copy.write(block)
            copied += len(block)
        copy.flush()
        os.fchmod(copy.fileno(), 0o444)
    return {'file': target.name, 'bytes': copied, 'sha256': digest.hexdigest(),
            'spool': pathlib.Path(spool.name).name, 'spool_bytes_at_snapshot': length}


def main(argv):
    if len(argv) < 3:
        raise SystemExit('usage: capture.py LABEL SECONDS ARGV...')
    label, seconds, command = argv[0], float(argv[1]), argv[2:]
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', label) or seconds <= 0:
        raise SystemExit('invalid label or timeout')
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    paths = {suffix: EVIDENCE / (label + suffix)
             for suffix in ('.json', '.stdout', '.stderr', '.stdout.live', '.stderr.live')}
    if any(path.exists() for path in paths.values()):
        raise SystemExit('label already used or partially retained: ' + label)
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    receipt = {'label': label, 'start_utc': start, 'argv': command,
               'cwd': os.getcwd(), 'state': 'not_started', 'exit_code': None}
    then = time.monotonic()
    with contextlib.ExitStack() as stack:
        # Claim before effects; a race on any file refuses without dispatch.
        record = stack.enter_context(paths['.json'].open('x', encoding='utf-8'))
        record.write(json.dumps(receipt) + '\n')
        record.flush()
        out = stack.enter_context(paths['.stdout.live'].open('xb'))
        err = stack.enter_context(paths['.stderr.live'].open('xb'))
        receipt['state'] = 'dispatching'
        record.seek(0)
        record.truncate()
        record.write(json.dumps(receipt) + '\n')
        record.flush()
        try:
            proc = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=out,
                                  stderr=err, timeout=seconds)
            rc = proc.returncode
            receipt['state'] = 'client_returned'
        except subprocess.TimeoutExpired:
            rc = 124
            receipt.update(state='client_timed_out', guest_outcome='unknown', note=TIMEOUT_NOTE)
        except OSError as exc:
            rc = 125
            receipt.update(state='not_started', error=type(exc).__name__ + ': ' + str(exc))
        receipt.update(end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       duration_s=round(time.monotonic() - then, 3), exit_code=rc,
                       stdout=snapshot(out, paths['.stdout']), stderr=snapshot(err, paths['.stderr']),
                       stream_scope='read-only snapshot of the live spool at local client return; the '
                                    'spool may keep growing and is not this receipt\'s evidence')
        record.seek(0)
        record.truncate()
        record.write(json.dumps(receipt, indent=2) + '\n')
        record.flush()
        os.fchmod(record.fileno(), 0o444)
    print(json.dumps({k: receipt[k] for k in ('label', 'state', 'exit_code', 'duration_s')}))
    for stream, path in ((sys.stdout, paths['.stdout']), (sys.stderr, paths['.stderr'])):
        with path.open('rb') as handle:
            excerpt = handle.read(6000)
        stream.write(excerpt.decode('utf-8', 'replace'))
        if path.stat().st_size > len(excerpt):
            stream.write('\n[excerpt only; full snapshot in %s]\n' % path)
        stream.flush()
    if receipt['state'] == 'client_timed_out':
        sys.stderr.write('\n' + TIMEOUT_NOTE + '\n')
    return rc


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
