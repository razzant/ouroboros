"""Hold the real restart watcher's final cleanup for subprocess regression tests.

Reuse the repair's real-server/capture helper. Only gate restart cleanup, request
its event, and select launcher-managed transfer so no successor is executed.
"""
import json
import os
from pathlib import Path
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tests import _server_capture_exit as helper

scratch = Path(sys.argv[1])
gates = scratch / "held"
original_invoke = helper.invoke


def record(name, **facts):
    (gates / (name + ".json")).write_text(json.dumps({
        "pid": os.getpid(), "monotonic": time.monotonic(), **facts,
    }))


def invoke(entrypoint, args):
    import server

    # Exercise the actual watcher and its final exit, without spawning/re-execing
    # another server. This is the existing launcher-managed restart branch.
    server._LAUNCHER_MANAGED = True
    cleanup = server._emergency_process_cleanup
    switch = server.store_switch.switch_at_restart

    def gated_cleanup():
        record("restart-cleanup-entered")
        deadline = time.monotonic() + 20
        while not (gates / "release-restart-cleanup").exists():
            if time.monotonic() > deadline:
                raise TimeoutError("probe did not release restart cleanup")
            time.sleep(.01)
        result = cleanup()
        record("restart-cleanup-completed")
        return result

    def observed_switch(*args, **kwargs):
        result = switch(*args, **kwargs)
        record("restart-switch-completed", result=result)
        return result

    server._emergency_process_cleanup = gated_cleanup
    server.store_switch.switch_at_restart = observed_switch

    def request_restart():
        deadline = time.monotonic() + 35
        while not (gates / "request-restart").exists():
            if time.monotonic() > deadline:
                return
            time.sleep(.01)
        server._restart_requested.set()
        record("restart-requested")

    threading.Thread(target=request_restart, daemon=True).start()
    return original_invoke(entrypoint, args)


helper.invoke = invoke
helper.main()
