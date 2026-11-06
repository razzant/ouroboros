"""The warm bytecode prefix of one test run (test diet, 2026-11-05).

Every process of a run compiled from source: the run's ``PYTHONPYCACHEPREFIX`` (the
owner's root, ``test_environment.isolated_environment``) is the only place imports look
for bytecode, and ``PYTHONDONTWRITEBYTECODE=1``, kept for the run so no child ever writes
into a signed bundle (tests/test_v636_regressions.py), means nothing lands there. The
owner of a battery (scripts/run_tests.py, the hermetic gate) warms the prefix once instead,
through one contained root process (``run``): ``compileall --invalidation-mode checked-hash``
over the checkout's Python (valid whatever the sources' mtimes), then one collect-only walk
of the test tree in as many shards as the run has workers, which writes pytest's rewritten
test modules and the bytecode of every library module collection imports. A walker hands
pytest the ``tests`` directory as its one argument and passes over the other shards' modules
(``pytest_ignore_collect`` below, loaded by path like the root): handed its files one by
one, pytest re-collects their directory once per file (``Session.collect``, duplicates kept
for files named on the command line), and with the collector paused for collection
(tests/conftest.py) each walker held every node it discarded, 4.3 GB for a quarter of the
tree; four of them reached 14 GB on a 16 GB runner and were stopped (run 42126810926). Every later
controller, worker and child of the run reads them with the write ban kept. The prefix is
declared disposable scratch of the run (``OUROBOROS_TEST_SCRATCH.txt``, role
``bytecode-prefix``), so the launcher or the gate releases it with the run's root; nothing
persists. A warm that fails leaves the run cold and says so; it decides nothing.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid

MODULE_PATH = Path(__file__).resolve()
# The owner starts the root by path, like scripts/run_tests.py starts safe_test: no package import in the way.
_BOOTSTRAP = "import runpy, sys; raise SystemExit(runpy.run_path(sys.argv[1])['run'](sys.argv[2:]))"
# Keys a shard must not inherit: the write ban (the warm exists to write), the battery's evidence root and
# durations table (a collect-only shard is no session of the battery), a parent's slot allocation.
_NOT_FOR_SHARDS = ("PYTHONDONTWRITEBYTECODE", "OUROBOROS_TEST_EVIDENCE_ROOT", "OUROBOROS_TEST_DURATIONS",
                   "OURO_TEST_WORKER_BUDGET", "PYTEST_XDIST_AUTO_NUM_WORKERS",
                   "PYTEST_XDIST_WORKER", "PYTEST_XDIST_TESTRUNUID", "PYTEST_XDIST_WORKER_COUNT")
_CHECKOUT_DIRS = ("ouroboros", "supervisor", "tests", "scripts", "devtools")
SHARD_ENV = "OUROBOROS_TEST_WARM_SHARD"  # a walker's list of its test modules, one checkout-relative path per line
# The walkers' plugin, written beside the shard lists under a name of this run: a fixed name could be shadowed by a
# module of the candidate's own, since the walker's cwd comes before PYTHONPATH.
_WALK_PLUGIN = "ouroboros_warm_walk_"
_WALKED: dict[str, set[str]] = {}


def command(python: str, repo, *, workers: int, files_from=None) -> list[str]:
    """The owner's argv for the warm root; ``files_from`` names a file listing the test modules to walk
    (a selected run), else the whole tree."""
    argv = [python, "-c", _BOOTSTRAP, str(MODULE_PATH), "--repo", str(repo), "--workers", str(max(1, int(workers)))]
    return argv + (["--files-from", str(files_from)] if files_from else [])


def writing_env(env) -> dict:
    """The run's environment with the write ban lifted, for the warm's own processes alone."""
    return {key: value for key, value in env.items() if key not in _NOT_FOR_SHARDS}


def test_files(repo, files_from=None) -> list[str]:
    if files_from:
        return [line.strip() for line in Path(files_from).read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(path.relative_to(repo).as_posix() for path in (Path(repo) / "tests").rglob("test_*.py"))


def shards(files, count: int) -> list[list[str]]:
    """``count`` interleaved slices of ``files``, never an empty one."""
    count = max(1, min(int(count), len(files) or 1))
    return [files[index::count] for index in range(count) if files[index::count]]


def walker_command(python: str, shard, plugin: str) -> list[str]:
    """One walker's argv: the ``tests`` directory as a single argument (its modules outside it by name) and the
    ``plugin`` that keeps the walk to the shard listed under ``SHARD_ENV``."""
    return [python, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "--no-header",
            "-p", plugin, "tests", *[name for name in shard if not name.startswith("tests/")]]


def _key(path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def pytest_ignore_collect(collection_path, config):
    """In a walker, pass over the test modules of the other shards; outside a walker, decide nothing."""
    source = os.environ.get(SHARD_ENV)
    if (not source or not collection_path.name.startswith("test_") or collection_path.suffix != ".py"
            or not collection_path.is_file()):
        return None
    if source not in _WALKED:
        base = Path(str(config.invocation_params.dir))
        _WALKED[source] = {_key(base / line.strip())
                           for line in Path(source).read_text(encoding="utf-8").splitlines() if line.strip()}
    return True if _key(collection_path) not in _WALKED[source] else None


def run(argv) -> int:
    """The warm root: compile the checkout, then collect the tree in shards; one line of facts on stdout."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--files-from", default=None)
    args = parser.parse_args(argv)
    repo, env, workers = Path(args.repo).resolve(), writing_env(os.environ), max(1, args.workers)
    if prefix := env.get("PYTHONPYCACHEPREFIX"):  # the run's prefix: declared so its owner releases it with the root
        import runpy

        environment = runpy.run_path(str(MODULE_PATH.with_name("test_environment.py")))  # by path: cwd is the candidate
        Path(prefix).mkdir(parents=True, exist_ok=True)
        if not (Path(prefix) / environment["SCRATCH_MARKER"]).exists():
            environment["declare_scratch"](prefix, "bytecode-prefix",
                                           owner="the warm bytecode prefix of one test run (ouroboros/test_bytecode.py)")
    roots = [str(repo / name) for name in _CHECKOUT_DIRS if (repo / name).is_dir()] + [str(p) for p in repo.glob("*.py")]
    started = time.monotonic()
    compiled = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "-j", str(workers), "--invalidation-mode", "checked-hash", *roots],
        cwd=repo, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    facts = {"compile_seconds": round(time.monotonic() - started, 1), "compile_exit": compiled.returncode}
    files = test_files(repo, args.files_from)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="ouroboros-warm-") as lists:
        plugin = _WALK_PLUGIN + uuid.uuid4().hex
        (Path(lists) / f"{plugin}.py").write_text(
            f"import runpy\npytest_ignore_collect = runpy.run_path({str(MODULE_PATH)!r})['pytest_ignore_collect']\n",
            encoding="utf-8")
        walk_env = {**env, "PYTHONPATH": os.pathsep.join(filter(None, (lists, env.get("PYTHONPATH"))))}
        walkers = []
        for index, shard in enumerate(shards(files, workers)):
            listed = Path(lists) / f"shard-{index}.txt"
            listed.write_text("".join(name + "\n" for name in shard), encoding="utf-8")
            walkers.append(subprocess.Popen(walker_command(sys.executable, shard, plugin), cwd=repo,
                                            env={**walk_env, SHARD_ENV: str(listed)},
                                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        exits = sorted({walker.wait() for walker in walkers})
    facts.update(files=len(files), shards=len(walkers), collect_seconds=round(time.monotonic() - started, 1),
                 collect_exits=exits)
    print("WARM_BYTECODE " + json.dumps(facts), flush=True)
    return 0

