"""pytest inside the interpreter under test, with its own packages on a side path only this process sees.

The product reads ``sys.executable`` and ``sys.prefix`` in-process, so the process that prepares must be the bare
interpreter itself, and a uv venv of it (what CI runs) would not be. Its test dependencies come from ``sys.path`` alone,
never ``PYTHONPATH``, so every ``-m venv``, pip and server child sees exactly what the distro installed.

    /usr/bin/python3 -B target_driver.py SUBJECT DRIVER_SITE JUNIT NODEID...
"""
from __future__ import annotations

import os
import sys

# pytest-timeout per test, below the workflow's cell-step bound (34 min) and the job's (40): one hung node is a recorded
# INCOMPLETE row with the census after it and the upload still to come. The largest legitimate node installs the real
# lock (the product allows its pip 1800 s); slower than this, the row reads INCOMPLETE, never PASS.
PER_TEST_TIMEOUT = 900


def main() -> int:
    subject, driver_site, junit, *nodeids = sys.argv[1:]
    if not nodeids:
        print("no nodes selected", file=sys.stderr)
        return 4
    sys.path[:0] = [subject, driver_site]  # ahead of this file's folder, which stays on the path for the plugin
    os.chdir(subject)
    import pytest

    return int(pytest.main(["-c", "pyproject.toml", "--rootdir", subject, "-p", "distro_overlay_plugin",
                            "-p", "no:cacheprovider", f"--timeout={PER_TEST_TIMEOUT}", "--junitxml", junit, "-v",
                            *nodeids]))


if __name__ == "__main__":
    sys.exit(main())
