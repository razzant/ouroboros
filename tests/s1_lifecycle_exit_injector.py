"""Copied as sitecustomize.py ONLY into a disposable test interpreter path.

SIGUSR1 produces a REAL non-signal exit(23) only for the exact explicitly armed
PID. This injects death, not a fake Process/exitcode or a retry decision. The
server's native multiprocessing owner must observe 23 itself. SIGKILL is tested
separately, and must remain ineligible for retry.
"""
import json
import os
from pathlib import Path
import signal


def _exit_if_armed(_signal, _frame):
    root = Path(os.environ["S1_LIFECYCLE_INJECTION"])
    permit = json.loads((root / f"permit-{os.getpid()}.json").read_text())
    if permit.get("pid") == os.getpid() and permit.get("exit_code") == 23:
        os._exit(23)


if os.environ.get("S1_LIFECYCLE_INJECTION") and hasattr(signal, "SIGUSR1"):
    signal.signal(signal.SIGUSR1, _exit_if_armed)
    root = Path(os.environ["S1_LIFECYCLE_INJECTION"])
    (root / f"installed-{os.getpid()}.json").write_text(json.dumps({
        "pid": os.getpid(), "signal": int(signal.SIGUSR1), "exit_code": 23,
    }))
