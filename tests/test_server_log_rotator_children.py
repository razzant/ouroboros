"""Only the server process writes ``logs/server.log``. Importing ``server.py`` — as a
spawn/forkserver worker does, under ``__mp_main__`` — configures no logging at all; a pool
worker configures a stream handler only, in ``worker_main``
(``ouroboros/process_logging.py``, pinned in ``tests/test_process_logging.py``), so two
processes never rotate ``server.log`` against each other. A module-level
``multiprocessing.parent_process()`` check would be None in such a child, so the proof is a
REAL child re-running the module under that name."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


@pytest.mark.serial
def test_a_spawn_child_importing_server_configures_no_logging(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    code = (
        "import logging, runpy, sys\n"
        "runpy.run_path('server.py', run_name='__mp_main__')\n"
        "print('HANDLERS', sorted(type(h).__name__ for h in logging.getLogger().handlers))\n"
    )
    env = {**os.environ, "OUROBOROS_DATA_DIR": str(data), "PYTHONPATH": str(REPO)}
    env.pop("PYTEST_CURRENT_TEST", None)
    completed = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True,
                               text=True, timeout=180)
    assert completed.returncode == 0, completed.stderr[-2000:]
    handlers = next(line for line in completed.stdout.splitlines() if line.startswith("HANDLERS"))
    assert handlers == "HANDLERS []", handlers
    assert not (data / "logs" / "server.log").exists()
