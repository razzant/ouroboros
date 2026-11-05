"""The warm bytecode prefix of one test run (test diet, 2026-11-05).

Every process of a run compiled from source: the run's ``PYTHONPYCACHEPREFIX`` (the
owner's root, ``test_environment.isolated_environment``) is the only place imports look
for bytecode, and ``PYTHONDONTWRITEBYTECODE=1``, kept for the run so no child ever writes
into a signed bundle (tests/test_v636_regressions.py), means nothing lands there. The
owner of a battery (scripts/run_tests.py, the hermetic gate) warms the prefix once instead,
through one contained root process (``run``): ``compileall --invalidation-mode checked-hash``
over the checkout's Python (valid whatever the sources' mtimes), then one collect-only walk
of the test tree in as many shards as the run has workers, which writes pytest's rewritten
test modules and the bytecode of every library module collection imports. Every later
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
import time

MODULE_PATH = Path(__file__).resolve()
# The owner starts the root by path, like scripts/run_tests.py starts safe_test: no package import in the way.
_BOOTSTRAP = "import runpy, sys; raise SystemExit(runpy.run_path(sys.argv[1])['run'](sys.argv[2:]))"
# Keys a shard must not inherit: the write ban (the warm exists to write), the battery's evidence root and
# durations table (a collect-only shard is no session of the battery), a parent's slot allocation.
_NOT_FOR_SHARDS = ("PYTHONDONTWRITEBYTECODE", "OUROBOROS_TEST_EVIDENCE_ROOT", "OUROBOROS_TEST_DURATIONS",
                   "OURO_TEST_WORKER_BUDGET", "PYTEST_XDIST_AUTO_NUM_WORKERS",
                   "PYTEST_XDIST_WORKER", "PYTEST_XDIST_TESTRUNUID", "PYTEST_XDIST_WORKER_COUNT")
_CHECKOUT_DIRS = ("ouroboros", "supervisor", "tests", "scripts", "devtools")


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
    walkers = [subprocess.Popen([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
                                 "--no-header", *shard], cwd=repo, env=env, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL) for shard in shards(files, workers)]
    exits = sorted({walker.wait() for walker in walkers})
    facts.update(files=len(files), shards=len(walkers), collect_seconds=round(time.monotonic() - started, 1),
                 collect_exits=exits)
    print("WARM_BYTECODE " + json.dumps(facts), flush=True)
    return 0


def warm_lane(agent_python: str, worktree, temp_root, evidence: dict, timeout, files=()) -> str:
    """The hermetic gate's warm, before its passes: the root runs contained and behind the owner's work, under
    the gate's env (``_preflight_env``), for as many shards as the gate admitted workers; its facts join the
    run's lanes. Returns a hard-block diagnosis only when the warm's tree outlived it (containment), never for
    a warm that failed or ran out of ``timeout`` (what the run's budget still held, None for none; the gate
    charges the passes nothing for it): those leave the passes cold, recorded as such."""
    from ouroboros import preflight_runner as pr
    from ouroboros.preflight_inventory import note_lane_process
    from ouroboros.process_containment import ProcessContainer
    from ouroboros.test_evidence import FileOutput

    temp_root = Path(temp_root)
    files_from = None
    if files:
        files_from = pr.probe_dir(temp_root) / "warm-files.txt"
        files_from.write_text("\n".join(files) + "\n", encoding="utf-8")
    argv = command(agent_python, worktree, workers=pr.allocated_worker_count(temp_root), files_from=files_from)
    container, output, proc = ProcessContainer(lifeline=True, overload=True), FileOutput(temp_root, "warm"), None
    started = time.monotonic()
    try:
        proc = container.spawn(argv, background=True, cwd=str(worktree), env=pr._preflight_env(temp_root, worktree),
                               stdout=output.stdout, stderr=output.stderr)
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pr._terminate_preflight_tree(proc, temp_root)
            returncode = None
        reap_error = container.reap()
        preview = output.finish(returncode=returncode, containment_error=reap_error, argv=argv)
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            pr._terminate_preflight_tree(proc, temp_root)
        note_lane_process(exc, container, proc)
        raise
    finally:
        if proc is not None and proc.poll() is None:
            pr._terminate_preflight_tree(proc, temp_root)
        container.reap()
        container.close()
        output.close()
    facts = next((line.removeprefix("WARM_BYTECODE ").strip() for line in preview.splitlines()
                  if line.startswith("WARM_BYTECODE ")), "")
    pr._record_lane(evidence, "warm", {"evidence_errors": output.errors}, returncode=returncode,
                    containment_error=reap_error, seconds=round(time.monotonic() - started, 1),
                    facts=json.loads(facts) if facts else None)
    if reap_error:
        return pr._diagnosis(
            "⚠️ PRE_PUSH_TEST_ERROR: PREFLIGHT_CONTAINMENT_FAILED (hard block): the warm ran, but processes it "
            "spawned were still alive afterwards (or could not be determined to be gone)",
            "The bytecode warm collects the candidate's test tree once; a tree it leaves alive survives into the "
            "passes and past teardown, so its verdict is not accepted. Find what the candidate's collection starts "
            "and does not wait for.", reap_error, 8000)
    return ""
