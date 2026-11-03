"""A timed-out ``node --test`` run leaves no process behind under ``tests.group_run``.

``node --test`` runs each test file in a child process. Bare ``subprocess.run(timeout=)``
kills only the runner, so on POSIX that child survives with ppid 1; the falsifying test
proves the check sees such a survivor, then kills it by its exact PID. The spinning file
only burns CPU and never allocates.
"""
import os
import shutil
import subprocess
import sys
import time

import pytest

from ouroboros.platform_layer import force_kill_pid, pid_provably_gone
from tests.group_run import kill_survivors, run_in_group

NODE = shutil.which("node")
pytestmark = [pytest.mark.serial,  # real processes, killed by PID
              pytest.mark.skipif(not NODE, reason="node is not installed")]

SPIN = """\
const fs = require('node:fs');
const test = require('node:test');
test('spins without allocating', () => {
  fs.writeFileSync(process.env.SPIN_PIDS, `${process.pid} ${process.ppid}`);
  for (;;) {}
});
"""


def _spin(tmp_path, launch):
    """Time out a spinning test file; return the PIDs of its file child and of the runner."""
    spin, pids = tmp_path / "spin.test.cjs", tmp_path / "pids"
    spin.write_text(SPIN, encoding="utf-8")
    with pytest.raises(subprocess.TimeoutExpired):
        launch([NODE, "--test", str(spin)], cwd=tmp_path, env={**os.environ, "SPIN_PIDS": str(pids)}, timeout=3)
    deadline = time.monotonic() + 5  # a surviving child may still be writing
    while not (pids.exists() and len(pids.read_text(encoding="utf-8").split()) == 2):
        assert time.monotonic() < deadline, "node did not start the spinning test file before the timeout"
        time.sleep(0.05)
    child, runner = map(int, pids.read_text(encoding="utf-8").split())
    return child, runner


def test_timeout_stops_the_test_file_child_with_its_runner(tmp_path):
    assert not kill_survivors(_spin(tmp_path, run_in_group))


@pytest.mark.skipif(sys.platform == "win32", reason="libuv's job object ends node's children with their parent")
def test_bare_run_timeout_leaves_the_test_file_child_running(tmp_path):
    child, runner = _spin(tmp_path, lambda args, **kwargs: subprocess.run(args, capture_output=True, **kwargs))
    try:
        assert not kill_survivors([runner])
        time.sleep(0.5)
        assert not pid_provably_gone(child), "the orphan check no longer sees a survivor"
    finally:
        if not pid_provably_gone(child):
            force_kill_pid(child)
    assert not kill_survivors([child])
