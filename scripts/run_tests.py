#!/usr/bin/env python3
"""Run the local test battery the fast, correct way.

    python scripts/run_tests.py                  full battery (node lane + every default-lane test)
    python scripts/run_tests.py --sequential     full battery exactly as the commit gate and CI split it
    python scripts/run_tests.py tests/test_x.py  focused run; every argument is forwarded to pytest

A bare call always means the FULL battery — it never skips a test on the
strength of an earlier run. The default mode is one xdist run in which the
`serial` tests are pinned to a few file-sharded groups (`--serial-shards`,
tests/conftest.py; a file's group is a stable hash of its path) and therefore
overlap the parallel tests instead of waiting for them; `--sequential` keeps the marker split and xdist flags of
`ouroboros.preflight_runner` and `.github/workflows/ci.yml` in two passes.
The reviewed-commit gate and CI stay the authority: this script is feedback.

The marker exclusions and xdist flags are imported from the gate's SSOT
(`LANE_EXCLUSION_EXPR`, `PARALLEL_PASS_FLAGS`), never restated here. Data-root
isolation is `tests/conftest.py`'s job and is not repeated either.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shlex
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_MAX_SERIAL_SHARDS = 4
_PYTEST_EXIT_NO_TESTS = 5


def _workers() -> int:
    """What `-n auto` will resolve to: the operator override, else xdist's own count."""
    raw = os.environ.get("OUROBOROS_PREFLIGHT_TEST_WORKERS", "").strip()
    if raw.isdigit():
        return max(2, int(raw))
    try:  # xdist prefers physical cores whenever psutil is importable
        import psutil
        physical = psutil.cpu_count(logical=False)
    except Exception:
        physical = None
    return max(2, physical or os.cpu_count() or 2)


def serial_shards(workers: int) -> int:
    """How many serial groups may run at once: one per four workers, at most four.

    Four cores or fewer keep ONE serial group, i.e. serial tests still never
    overlap each other there — they only stop waiting for the parallel tests.
    """
    return max(1, min(_MAX_SERIAL_SHARDS, workers // 4))


def _with(flags: list[str], option: str, value: str) -> list[str]:
    out = list(flags)
    out[out.index(option) + 1] = value
    return out


def battery_commands(sequential: bool) -> list[tuple[str, list[str]]]:
    from ouroboros.preflight_runner import LANE_EXCLUSION_EXPR, PARALLEL_PASS_FLAGS

    base = [sys.executable, "-m", "pytest", "tests/"]
    if sequential:
        return [
            ("parallel", [*base, "-m", f"not serial and {LANE_EXCLUSION_EXPR}", *PARALLEL_PASS_FLAGS]),
            ("serial", [*base, "-m", f"serial and {LANE_EXCLUSION_EXPR}"]),
        ]
    overlapped = _with(PARALLEL_PASS_FLAGS, "--dist", "loadgroup")
    return [(
        "overlapped",
        [*base, "-m", LANE_EXCLUSION_EXPR, *overlapped, f"--serial-shards={serial_shards(_workers())}"],
    )]


def _run(label: str, argv: list[str], cwd: pathlib.Path) -> int:
    print(f"\n=== {label}: {shlex.join(argv[2:] if argv[0] == sys.executable else argv)}", flush=True)
    started = time.monotonic()
    env = dict(os.environ)
    override = env.get("OUROBOROS_PREFLIGHT_TEST_WORKERS", "").strip()
    if override.isdigit():  # the gate steers `-n auto` the same way (preflight_runner._preflight_env)
        env["PYTEST_XDIST_AUTO_NUM_WORKERS"] = str(max(2, int(override)))
    code = subprocess.call(argv, cwd=str(cwd), env=env)
    print(f"=== {label}: exit {code} in {time.monotonic() - started:.0f}s", flush=True)
    return code


def _node_lane() -> int:
    from ouroboros.preflight_node import (
        NODE_MIN_VERSION, WEB_DIR, _version_tuple, candidate_node_tests, probe_node_version, resolve_node,
    )

    files = candidate_node_tests(REPO)
    if not files:
        return 0
    node = resolve_node()
    version = probe_node_version(node) if node else ""
    floor = ".".join(str(part) for part in NODE_MIN_VERSION)
    if not version or _version_tuple(version) < NODE_MIN_VERSION:
        found = f"{node} is v{version}" if version else (f"{node} did not answer --version" if node else "no node found")
        print(f"\n=== node: NOT_RUN — {found}; the web lane needs node >= {floor}, so this is not the full "
              "battery: install node or pass focused targets", flush=True)
        return 1
    return _run("node", [node, "--test", *files], REPO / WEB_DIR)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequential", action="store_true", help="two passes, exactly as the commit gate and CI")
    args, pytest_args = parser.parse_known_args(argv)
    if not (REPO / "tests" / "conftest.py").is_file():
        print(f"{REPO} is not an Ouroboros checkout (no tests/conftest.py)", file=sys.stderr)
        return 2
    if pytest_args:
        if args.sequential:
            parser.error("--sequential selects the full two-pass battery; drop it or drop the pytest arguments")
        return _run("focused", [sys.executable, "-m", "pytest", *pytest_args], REPO)
    started = time.monotonic()
    code = _node_lane()
    ran = 0
    for label, command in battery_commands(args.sequential) if code == 0 else []:
        code = _run(label, command, REPO)
        if code == _PYTEST_EXIT_NO_TESTS:
            code = 0
            continue
        ran += 1
        if code != 0:
            break
    if code == 0 and ran == 0:
        print("no tests were collected in any lane", file=sys.stderr)
        code = 1
    print(f"\n=== battery: {'GREEN' if code == 0 else 'RED'} in {time.monotonic() - started:.0f}s", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
