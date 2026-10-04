"""Opt-in diagnostics for the disposable Windows qualification runner only.

Installed into the runner's base-Python site-packages, never into candidate source.
The production launcher overwrites PYTHONPATH with its candidate, so a sidecar
directory on the driver's PYTHONPATH cannot instrument that child. No behavior,
settings, pipe reads or shutdown callbacks are overridden here.
"""
import os

if os.environ.get("OUROBOROS_LAUNCHER_STOP_STDIN") == "1":
    import faulthandler
    from pathlib import Path
    import sys

    root = os.environ.get("OUROBOROS_DATA_DIR", "")
    if root and sys.platform == "win32":
        path = Path(root) / "logs" / f"qualification-stack-{os.getpid()}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        _qualification_stack_handle = path.open("a", encoding="utf-8")
        faulthandler.dump_traceback_later(60, repeat=True, file=_qualification_stack_handle)
        print(f"QUALIFICATION_STACK_CAPTURE {path}", flush=True)
