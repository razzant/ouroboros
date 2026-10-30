#!/usr/bin/env python3
"""Run a verification command with disposable filesystem and configuration roots.

Use `python -I -S scripts/safe_test.py --temp-parent /tmp -- <venv-python> -m pytest ...`.
This stdlib launcher audits its environment helper before loading it, then prints
the boundary before starting the command. It installs nothing and is not a sandbox.

Stage two retains complete raw output and declared diagnostics, reaps the
contained command, then releases declared reproducible test scratch on green
and red. Unknown command data, debug and unconfirmed teardown stay in place.
The last stderr line names the root once: `SAFE_TEST_RETAINED` or
`SAFE_TEST_RELEASED` describes that path, not every process anywhere.

`--timeout SECONDS` is a deadline for everything the command started, which
`subprocess.run(timeout=)` or killing the runner's PID is not: `node --test`, for
one, keeps each test file in a child that outlives its runner. At the deadline
the attributed tree gets SIGTERM, ten seconds of grace, then the forced reap;
the launcher exits 124 like GNU `timeout`. A first SIGINT, SIGTERM or SIGHUP is
passed to stage two, which exits 128+signal; repeated signals never interrupt
teardown. SIGTERM and SIGHUP stop the tree the same way. SIGINT first reaches
only the command's root, which gets the ten seconds to export its results
(pytest's JUnit and KeyboardInterrupt report; `scripts/run_tests.py` passes it on
to its pytest within them); whatever remains, even once the root has exited,
then gets SIGTERM, grace and the forced reap. Windows has no tree-wide SIGTERM:
its kill-on-close Job stops the tree at once, and the console's Ctrl-C already
reaches stage two, so nothing is forwarded there.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import tempfile


def _wait_owning(argv: list[str], repo: Path, env: dict) -> int:
    """Start stage two and wait for it; a first POSIX stop signal is passed on, never acted on here."""
    received: list[int] = []
    supervisor = None

    def forward(signum, _frame):
        # Windows Popen.send_signal(SIGINT) is invalid; that console's Ctrl-C reached stage two itself.
        if not received:
            received.append(signum)
            if supervisor is not None and os.name != "nt":
                try:
                    supervisor.send_signal(signum)
                except OSError:
                    pass  # Already gone: its own exit status says how.

    stops = ([signal.SIGINT] if os.name == "nt"
             else [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
    for signum in stops:
        signal.signal(signum, forward)
    if received:  # A stop before stage two existed: nothing to run or tear down.
        return 128 + received[0]
    supervisor = subprocess.Popen(argv, cwd=repo, env=env)
    if received and os.name != "nt":
        supervisor.send_signal(received[0])  # Arrived while Popen was starting it.
    code = supervisor.wait()  # Handlers return, so PEP 475 resumes this wait.
    return code if code >= 0 else 128 - code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temp-parent", type=Path)
    parser.add_argument("--evidence-dir", type=Path, help="persistent parent for a fresh evidence directory")
    parser.add_argument("--keep", action="store_true", help="retain the complete scratch environment for debugging")
    parser.add_argument("--timeout", type=float, metavar="SECONDS",
                        help="stop the command and every process it started after SECONDS; exit 124")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("supply a verification command after --")
    if args.timeout is not None and not args.timeout > 0:
        parser.error("--timeout must be a positive number of seconds")
    repo = Path(__file__).resolve().parents[1]
    helper = repo / "ouroboros" / "test_environment.py"
    tree = ast.parse(helper.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        names = ([item.name for item in node.names] if isinstance(node, ast.Import)
                 else [node.module] if isinstance(node, ast.ImportFrom) else [])
        if any(name not in {"__future__", "ast", "os", "pathlib", "sys", "tempfile"} for name in names):
            raise RuntimeError("test environment helper must import only its audited stdlib leaves")
    environment = runpy.run_path(str(helper))
    boundary = environment["isolated_environment"]
    parent = Path(args.temp_parent or environment["default_temp_parent"]()).resolve()
    if parent == repo or repo in parent.parents:
        # A disposable root nested in the checkout is not disposable: `git`
        # started in it walks up into THIS working tree, and a run that
        # snapshots or resets its own temp path then operates on the
        # operator's source. Refuse instead of accepting a false boundary.
        parser.error("--temp-parent must be outside the repository working tree")
    ancestor = environment["git_ancestor"](parent)
    if ancestor is not None:
        # The same hazard one level up: any enclosing checkout, not only this one.
        parser.error(f"--temp-parent must not be inside a Git working tree ({ancestor})")
    # Importing ouroboros runs its pending-adoption hook. Verification must not
    # accidentally switch the source checkout before loading stage two.
    git_dir = repo / ".git"
    if git_dir.is_file():
        target = git_dir.read_text(encoding="utf-8").removeprefix("gitdir:").strip()
        git_dir = (repo / target).resolve()
    if (git_dir / "ouroboros-body-adoption").exists():
        parser.error("source checkout has a pending body adoption; finish it before verification")
    # Short on purpose: server sockets live several levels below this root and
    # macOS caps an AF_UNIX path at 104 bytes. Born `.noindex`: Spotlight never
    # registers the fixtures below it (fake `Ouroboros.app` bundles among them).
    root = Path(tempfile.mkdtemp(prefix="ob-", suffix=".noindex", dir=parent)).resolve()
    try:
        env = boundary(root, repo, keep=(
            "OUROBOROS_RUN_UI_SMOKE", "OUROBOROS_EXPECT_BROWSER_ENGINES",
            "OUROBOROS_PREFLIGHT_TEST_WORKERS", "OUROBOROS_PREFLIGHT_SERIAL",
            # CI requires these test dependencies; without them a missing plugin or Node skips.
            "OUROBOROS_PREFLIGHT_REQUIRE_PLUGINS", "OUROBOROS_PREFLIGHT_REQUIRE_NODE",
            "PLAYWRIGHT_BROWSERS_PATH", "OUROBOROS_E2E_DEEP",
            # The run's shard weights (ci.yml `shard-plan`): tests/default_lane.py, tests/browser_lane.py.
            "OUROBOROS_CI_DURATIONS",
            # Opt-in native desktop qualification; ordinary runs remain skipped.
            "OUROBOROS_WINDOWS_DESKTOP_QA", "OUROBOROS_MACOS_DESKTOP_QA",
            "OUROBOROS_TEST_EVIDENCE_ROOT", *environment["DIAGNOSTIC_OUTPUT_KEYS"],
        ))
        # Each pytest controller claims its own fresh basetemp beneath this owned
        # root (tests/conftest.py): pytest deletes an EXISTING --basetemp, so no two
        # invocations — e.g. run_tests.py's two passes — may ever share one.
        env["OUROBOROS_TEST_TEMP_ROOT"] = str(root)
        print("SAFE_TEST_BOUNDARY " + json.dumps({key: value for key, value in env.items()
              if key in {"HOME", "PYTHONUSERBASE", "PYTHONPYCACHEPREFIX"}
              or key.startswith("OUROBOROS_")}), flush=True)
        # Complete environment replacement reaches the interpreter BEFORE site,
        # plugins, conftest or any application module can import.
        bootstrap = (
            "import sys; sys.dont_write_bytecode=True; "
            "sys.path.insert(0, sys.argv[1]); "
            "from ouroboros.safe_test_supervisor import main; "
            "raise SystemExit(main(sys.argv[2], sys.argv[1], sys.argv[6:], "
            "evidence_parent=sys.argv[3] or None, keep=sys.argv[4]=='1', "
            "timeout=float(sys.argv[5]) if sys.argv[5] else None))"
        )
        return _wait_owning(
            [sys.executable, "-I", "-S", "-c", bootstrap, str(repo), str(root),
             str(args.evidence_dir.resolve()) if args.evidence_dir else "", "1" if args.keep else "0",
             repr(args.timeout) if args.timeout is not None else "", *command],
            repo, env,
        )
    finally:
        # The one final line: stage two has ended (or never started), so this
        # describes the root as it is now. A failed stderr cannot change the status.
        try:
            print(f"SAFE_TEST_{'RETAINED' if root.exists() else 'RELEASED'} {root}", file=sys.stderr, flush=True)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
