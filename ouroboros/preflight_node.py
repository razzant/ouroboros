"""Node lane of the hermetic commit gate: `node --test` over web/tests/*.test.js.

A sibling of ``preflight_runner.run_hermetic_pytest`` (which sits at its
module-size band ceiling, hence a separate module): once the candidate worktree
is assembled, this lane runs the browser-module suite exactly as both CI jobs
and ``web/package.json`` do, so client-side regressions stop sailing through a
gate that previously executed no JavaScript at all.

The lane keys on CONTENT, not configuration: it is active only when the
candidate tree carries ``web/tests/*.test.js`` files, so the gate's synthetic
fixture repositories (which have none) never require a node runtime. While the
lane IS active, a missing or too-old node is a typed hard block — never a
silent skip, because a gate that quietly drops a suite is indistinguishable
from a green one (the same stance ``PREFLIGHT_PLUGIN_MISSING`` takes for the
pytest plugins).
"""

from __future__ import annotations

import pathlib
import os
import shutil
import subprocess
import time
from typing import Optional

from ouroboros.platform_layer import probe_node_version, resolve_bundled_node

# Version floor for the browser-module suite: the object form of
# `t.mock.timers.enable({apis: [...]})` used by web/tests needs Node >= 20.11.
# CI provisions 22; the bundled runtime is 24.x.
NODE_MIN_VERSION = (20, 11)

# The web package boundary and its test glob. `web/package.json` runs
# `node --test tests/*.test.js` FROM web/; the gate and both CI jobs (through
# `scripts/run_tests.py --node`) run `candidate_node_tests` over these two
# constants, so the gate, CI, and the package script cannot drift apart silently.
# The lane globs in Python (sorted, cross-platform) instead of relying on a shell.
WEB_DIR = "web"
TESTS_GLOB = "tests/*.test.js"

_REMEDIATION = (
    "The hermetic gate runs the browser-module suite (web/tests/*.test.js) with "
    "`node --test`, exactly as both CI jobs do. Provide Node.js >= 20.11: run "
    "`bash scripts/download_node_standalone.sh` from the repository root (it "
    "stages the official signed runtime under node-standalone/, which resolves "
    "first), or install a `node` on PATH. The gate does NOT skip the lane when "
    "node is absent — a gate that silently drops a suite is indistinguishable "
    "from a green one."
)


def candidate_node_tests(worktree: "pathlib.Path | str") -> list:
    """Sorted web-relative test paths in the CANDIDATE tree; empty = lane off.

    Globbed BEFORE any node resolution, so a repository without a browser suite
    never pays for — or hard-fails on — a runtime it does not need.
    """
    web = pathlib.Path(worktree) / WEB_DIR
    return sorted(
        entry.relative_to(web).as_posix()
        for entry in web.glob(TESTS_GLOB)
        if entry.is_file()
    )


def resolve_node() -> Optional[str]:
    """Bundled signed node first, then PATH.

    The packaged app's macOS code-signing enforcement can SIGKILL a Homebrew
    node launched from its process tree; the bundled official runtime is
    re-signed by the build and always safe, so it wins when present.
    """
    try:
        bundled = resolve_bundled_node()
    except Exception:
        bundled = None
    return bundled or shutil.which("node")


def node_environment(env: dict) -> dict:
    """Give Python tests and their children the Node selected for the web lane.
    Preserve other PATH entries; discovery alone adds no runtime requirement or probe."""
    projected = dict(env)
    node = resolve_node()
    if node:
        current = shutil.which("node", path=env.get("PATH", os.defpath))
        if pathlib.Path(node).is_absolute() and os.path.normcase(node) == os.path.normcase(current or ""):
            return projected  # The selected Node is already reachable; keep other tool precedence.
        directory = str(pathlib.Path(node).absolute().parent)
        entries = env.get("PATH", "").split(os.pathsep) if env.get("PATH") else []
        projected["PATH"] = os.pathsep.join([directory, *(entry for entry in entries
            if os.path.normcase(entry) != os.path.normcase(directory))])
    return projected


def _version_tuple(version: str) -> tuple:
    """Leading numeric components of a `--version` string, floor-comparable."""
    parts = []
    for piece in version.strip().removeprefix("v").split(".")[: len(NODE_MIN_VERSION)]:
        digits = ""
        for ch in piece:
            if not ch.isdigit():
                break
            digits += ch
        parts.append(int(digits or 0))
    return tuple(parts)


def run_node_tests(
    worktree: pathlib.Path,
    temp_root: pathlib.Path,
    timeout: float,
    max_output: int,
) -> Optional[dict]:
    """Run the candidate's browser-module suite once, contained, or block typed.

    Returns ``None`` while the lane is inactive (no ``web/tests/*.test.js`` in
    the candidate). Otherwise a small result dict whose ``error`` key carries
    the bounded typed diagnosis — ``None`` on green. The caller runs this as
    the FIRST consumer of the shared preflight budget, before any pytest pass.
    ``reap_error`` is the container's reason when a process the lane spawned
    was not proven gone (empty otherwise): the caller then keeps the tree.
    """
    # Lazy sibling import: preflight_runner imports this module at its top, so
    # the shared helpers must be reached at call time, not import time.
    from ouroboros import preflight_runner as pr
    from ouroboros.preflight_inventory import note_lane_process
    from ouroboros.process_containment import ProcessContainer

    files = candidate_node_tests(worktree)
    if not files:
        return None
    result = {"files": len(files), "node": None, "returncode": None, "error": None, "reap_error": "", "execution": "not_run"}
    node = resolve_node()
    if not node:
        result["error"] = pr._diagnosis(
            "⚠️ PRE_PUSH_TEST_ERROR: PREFLIGHT_NODE_MISSING (hard block): no "
            f"Node.js runtime for the {len(files)} web/tests suite file(s) in the candidate",
            _REMEDIATION, "", max_output,
        )
        return result
    result["node"] = node
    version = probe_node_version(node)
    if not version:
        result["error"] = pr._diagnosis(
            "⚠️ PRE_PUSH_TEST_ERROR: PREFLIGHT_NODE_MISSING (hard block): "
            f"{node} exists but `--version` failed, so it cannot run the suite",
            _REMEDIATION + " (On macOS a Homebrew node launched from the packaged "
            "app is SIGKILLed by code-signing enforcement — the bundled runtime "
            "is the fix.)",
            "", max_output,
        )
        return result
    if _version_tuple(version) < NODE_MIN_VERSION:
        floor = ".".join(str(part) for part in NODE_MIN_VERSION)
        result["error"] = pr._diagnosis(
            "⚠️ PRE_PUSH_TEST_ERROR: PREFLIGHT_NODE_TOO_OLD (hard block): "
            f"{node} is v{version}, below the v{floor} floor the suite needs",
            _REMEDIATION, "", max_output,
        )
        return result

    # Same containment pattern as the pytest passes (`_execute_pytest_pass`):
    # spawned INSIDE the container — `node --test` forks one worker per file —
    # reaped unconditionally, and a live-or-undeterminable leftover is a hard
    # block whatever the exit code said. Run from the candidate's web/ with
    # web-relative paths so imports resolve exactly as under `npm test`.
    container = ProcessContainer(lifeline=True, overload=True)
    started = time.monotonic()
    # NODE_OPTIONS admits --test-only/--test-*-pattern/--require: an inherited
    # operator value can silently green a red suite — same class the pytest
    # lane closes by scrubbing PYTEST_* (see _preflight_env).
    env = pr._preflight_env(temp_root, worktree, cwd=pathlib.Path(worktree) / WEB_DIR)
    env.pop("NODE_OPTIONS", None)
    from ouroboros.test_evidence import FileOutput

    capture = FileOutput(temp_root, "node")
    proc = None
    argv = [node, "--test", f"--test-concurrency={env['PYTEST_XDIST_AUTO_NUM_WORKERS']}", *files]
    try:
        proc = container.spawn(argv, cwd=str(pathlib.Path(worktree) / WEB_DIR),
                               env=env, stdout=capture.stdout, stderr=capture.stderr)
        result["execution"] = "executed"
        try:
            returncode: Optional[int] = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pr._terminate_preflight_tree(proc, temp_root)
            returncode = None
        reap_error = container.reap()
        if proc.poll() is None:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                reap_error = reap_error or "direct node process did not exit after reap"
        output = capture.finish(returncode=returncode, containment_error=reap_error, argv=argv)
        result["evidence_errors"] = list(capture.errors)
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
        capture.close()
    elapsed = time.monotonic() - started
    result["returncode"] = returncode
    result["reap_error"] = reap_error
    if reap_error:
        result["error"] = pr._diagnosis(
            "⚠️ PRE_PUSH_TEST_ERROR: PREFLIGHT_CONTAINMENT_FAILED (hard block): "
            "the node pass ran, but processes it spawned were still alive "
            "afterwards (or could not be determined to be gone)",
            "The container DETECTS leaked processes; the pass verdict is not "
            "accepted, whatever it was. Find the web test that spawns or leaks "
            "a real process and make it clean up.",
            reap_error, max_output,
        )
    elif returncode is None:
        result["error"] = pr._with_timeout_excerpt(
            f"⚠️ PRE_PUSH_TEST_ERROR: node --test timed out after {timeout:.0f} "
            "seconds in the node pass (it shares the preflight total budget)",
            output, max_output,
        )
    elif returncode != 0:
        result["error"] = pr._diagnosis(
            "⚠️ PRE_PUSH_TEST_ERROR: NODE_TESTS_FAILED (hard block): "
            f"node pass, exit {returncode}, {elapsed:.0f}s",
            f"`node --test` over the candidate's {WEB_DIR}/tests suite failed "
            f"under {node} (v{version}). Fix the failing browser-module test "
            "named below; both CI jobs run this exact suite.",
            output, max_output,
        )
    return result
