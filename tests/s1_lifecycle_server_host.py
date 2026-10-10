"""Test launcher: attest imports, then enter the unmodified server entry point.

No owner, queue, checkpoint, retry, acknowledgement or lifecycle function is
replaced. The same wrapper runs after a production POSIX direct re-exec.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import time
import uuid


def main():
    candidate = Path.cwd().resolve()
    sys.path.insert(0, str(candidate))
    origins = {name: importlib.util.find_spec(name).origin
               for name in ("ouroboros", "supervisor")}
    assert all(Path(path).resolve().is_relative_to(candidate) for path in origins.values()), origins
    evidence = Path(os.environ["S1_LIFECYCLE_EVIDENCE"])
    boot = {
        "boot_id": uuid.uuid4().hex, "pid": os.getpid(), "time_ns": time.time_ns(),
        "cwd": str(candidate), "origins": origins, "python": sys.executable,
        "python_version": sys.version,
        "roots": {key: os.environ.get(key, "") for key in (
            "HOME", "OUROBOROS_APP_ROOT", "OUROBOROS_REPO_DIR", "OUROBOROS_DATA_DIR",
            "OUROBOROS_SETTINGS_PATH", "PYTHONPATH")},
        "server_sha256": hashlib.sha256((candidate / "server.py").read_bytes()).hexdigest(),
    }
    with (evidence / "boots.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(boot) + "\n")
    # server_control.restart_current_process already supports this exact argv
    # carrier. It changes no exit, ACK or restart decision.
    os.environ["OUROBOROS_SERVER_REEXEC_ARGV_JSON"] = json.dumps([str(Path(__file__).resolve())])
    sys.argv = [str(candidate / "server.py")]
    runpy.run_path(str(candidate / "server.py"), run_name="__main__")


if __name__ == "__main__":
    main()
