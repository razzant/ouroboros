#!/usr/bin/env python3
"""Run the local test battery the fast, correct way.

    python scripts/run_tests.py                  full battery (node lane + every default-lane test)
    python scripts/run_tests.py --sequential     full battery exactly as the commit gate and CI split it
    python scripts/run_tests.py --selected --base HEAD  commit/PR floor + affected tests, with both lanes
    python scripts/run_tests.py --selected --keep-going ...  the same, but a red lane does not skip the next
    python scripts/run_tests.py --node            only the complete Node lane (CI runs it as its own step)
    python scripts/run_tests.py --evidence-dir DIR  pytest diagnostics in DIR/<phase>, passed by argv
    python scripts/run_tests.py tests/test_x.py  focused run; every argument is forwarded to pytest

A bare call always means the FULL battery — it never skips a test on the
strength of an earlier run. The default mode is one xdist run in which the
`serial` tests are pinned to a few file-sharded groups (`--serial-shards`,
tests/conftest.py; a file's group is a stable hash of its path) and therefore
overlap the parallel tests instead of waiting for them; `--sequential` keeps the marker split and xdist flags of
`ouroboros.preflight_runner` and `.github/workflows/ci.yml` in two passes.
The reviewed-commit gate and CI stay the authority: this script is feedback.

Re-executes through safe_test: isolated HOME/data, owner/provider env scrubbed,
durable logs/source/diagnostics in <checkout>/../.test-evidence by default.
Ctrl-C reaches each phase's runner, allowing pytest JUnit/results export before escalation.
Marker exclusions/xdist flags come from the gate's SSOT
(`LANE_EXCLUSION_EXPR`, `PARALLEL_PASS_FLAGS`); conftest isolates test roots.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import subprocess
import sys
import tempfile
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_MAX_SERIAL_SHARDS = 4
_PYTEST_EXIT_NO_TESTS = 5
_INTERRUPT_EXPORT_SEC, _INTERRUPT_TERM_SEC = 7, 2  # busy xdist workers already spend ~5 s in teardown


def _workers() -> int:
    """Keep xdist's CPU default within the physical and inherited allocation."""
    from ouroboros.test_parallelism import worker_budget

    try:  # xdist prefers physical cores whenever psutil is importable
        import psutil
        physical = psutil.cpu_count(logical=False)
    except Exception:
        physical = None
    return worker_budget(physical or os.cpu_count() or 1)


def serial_shards(workers: int, table: dict | None = None) -> int:
    """How many serial groups may run at once.

    With recorded durations that split this system's test time (``serial``,
    tests/ci_verdicts.py), the serial tests get the share of the workers their
    share of the time needs: at least one group, at most half the workers. Without
    them, one group per four workers, at most four. Either way three workers or
    fewer keep ONE serial group, i.e. serial tests never overlap each other there —
    they only stop waiting for the parallel tests.
    """
    share = _serial_share(table)
    if share is None:
        return max(1, min(_MAX_SERIAL_SHARDS, workers // 4))
    return max(1, min(workers // 2, round(workers * share)))


def _serial_share(table: dict | None) -> float | None:
    """The serial part of this system's recorded test time, or None when the table does not say."""
    from tests.ci_verdicts import platform_name

    system = platform_name(sys.platform)
    total = sum(((table or {}).get("platforms") or {}).get(system, {}).values())
    serial = ((table or {}).get("serial") or {}).get(system)
    return sum(serial.values()) / total if total and serial is not None else None


def recorded_durations() -> tuple[dict | None, str]:
    """The table serial groups are sized by, and what it is.

    The run's ``OUROBOROS_CI_DURATIONS`` (the table tests/default_lane.py shards CI
    by), else the one the newest earlier one-run battery beside this run's evidence
    makes (``tests/ci_evidence.py durations`` over its pass). A battery that did not
    end (interrupted, stopped at its deadline) is passed over: its reports favour the
    serial groups, which are handed out first.
    """
    from tests.ci_verdicts import collect_durations, durations_table

    table, why = durations_table()
    if table is not None or why:
        return table, why or f"run {table['source']['run_id']}"
    current = os.environ.get("OUROBOROS_TEST_EVIDENCE_ROOT")
    ended = []
    for run in pathlib.Path(current).parent.glob("run-*") if current else ():
        try:
            recorded = (run / "pytest" / "overlapped" / "results.json").stat().st_mtime
            if json.loads((run / "result.json").read_text(encoding="utf-8")).get("returncode") in (0, 1):
                ended.append((recorded, run))
        except (OSError, ValueError, AttributeError):
            continue
    if not ended:
        return None, "no recorded durations"
    run = max(ended)[1]
    return collect_durations(run / "pytest", run_id=run.name, sha=""), f"the battery of {run.name}"


def serial_plan(table: dict | None, source: str, workers: int, shards: int) -> str:
    """The shard count's reason and, from recorded durations, each serial group's predicted seconds."""
    from tests.ci_verdicts import platform_name, serial_group

    if _serial_share(table) is None:
        return f"{shards} serial groups of {workers} workers ({source})"
    system, groups = platform_name(sys.platform), {}
    for path, ms in table["serial"][system].items():
        group = serial_group(path, shards)
        groups[group] = groups.get(group, 0) + ms
    serial, total = sum(groups.values()), sum(table["platforms"][system].values())
    predicted = ", ".join(f"{group} {groups[group] / 1000:.0f}" for group in sorted(groups))
    return (f"{shards} serial groups of {workers} workers, sized by {source}: serial {serial / 1000:.0f} of "
            f"{total / 1000:.0f} test-s; predicted seconds per group: {predicted}; the other tests about "
            f"{(total - serial) / 1000 / max(1, workers - shards):.0f} s per remaining worker")


def _with(flags: list[str], option: str, value: str) -> list[str]:
    out = list(flags)
    out[out.index(option) + 1] = value
    return out


def battery_commands(sequential: bool) -> list[tuple[str, list[str]]]:
    from ouroboros.preflight_runner import LANE_EXCLUSION_EXPR, PARALLEL_PASS_FLAGS, SERIAL_PASS_FLAGS

    base = [sys.executable, "-m", "pytest", "tests/"]
    if sequential:
        return [
            ("parallel", [*base, "-m", f"not serial and {LANE_EXCLUSION_EXPR}", *PARALLEL_PASS_FLAGS]),
            ("serial", [*base, "-m", f"serial and {LANE_EXCLUSION_EXPR}", *SERIAL_PASS_FLAGS]),
        ]
    overlapped = _with(PARALLEL_PASS_FLAGS, "--dist", "loadgroup")
    workers, (table, source) = _workers(), recorded_durations()
    shards = serial_shards(workers, table)
    print(f"=== {serial_plan(table, source, workers, shards)}", flush=True)
    return [("overlapped", [*base, "-m", LANE_EXCLUSION_EXPR, *overlapped, f"--serial-shards={shards}"])]


def _run(label: str, argv: list[str], cwd: pathlib.Path) -> int:
    from ouroboros.preflight_node import node_environment
    from ouroboros.process_containment import ProcessContainer
    from ouroboros.test_parallelism import NESTED_WORKERS_ENV, bounded_test_command

    started = time.monotonic()
    env = node_environment(os.environ)
    env["PYTEST_XDIST_AUTO_NUM_WORKERS"] = str(_workers())
    env[NESTED_WORKERS_ENV] = env["PYTEST_XDIST_AUTO_NUM_WORKERS"]
    argv = bounded_test_command(argv, env)
    print(f"\n=== {label}: {shlex.join(argv[2:] if argv[0] == sys.executable else argv)}", flush=True)
    container = ProcessContainer(lifeline=True, overload=True)
    proc = None
    try:
        proc = container.spawn(argv, cwd=str(cwd), env=env)
        code = proc.wait()
    except KeyboardInterrupt:
        code = 130
        if proc is not None:
            _stop_interrupted(container, proc)
    finally:
        try:
            diagnostic = container.reap()
            if proc is not None:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    diagnostic = "; ".join(filter(None, [diagnostic, "test root did not exit after reap"]))
            if diagnostic:
                print(f"=== {label}: {diagnostic}", file=sys.stderr, flush=True)
        finally:
            container.close()
    if diagnostic and code in (0, _PYTEST_EXIT_NO_TESTS):
        code = 1
    print(f"=== {label}: exit {code} in {time.monotonic() - started:.0f}s", flush=True)
    return code


def _stop_interrupted(container, proc) -> None:
    """Pass the INT safe_test forwarded here on to the runner, whose own session hides it.

    The export wait and the TERM grace are allotted inside safe_test's INT window for
    this process (its `_GRACE_SEC`); the following reap has its own deadline, so that
    window, not this allotment, bounds the whole. On overrun safe_test stops and reaps
    the tree itself: still 130, but this runner's own report may be incomplete.
    """
    if container.interrupt_root():
        try:
            proc.wait(timeout=_INTERRUPT_EXPORT_SEC)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            pass  # A slow export or a repeated INT only ends this wait; the tree is still stopped.
    container.terminate_tree(_INTERRUPT_TERM_SEC)


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
    return _run("node", [node, "--test", f"--test-concurrency={_workers()}", *files], REPO / WEB_DIR)


def _selected(args, evidence) -> int:
    from ouroboros.preflight_inventory import inventory_problem, read_inventory, selection_passes
    from ouroboros.preflight_runner import _probe_dir, _install_worker_probe
    from ouroboros.test_selection import select_tests

    try:
        selection = select_tests(REPO, base=args.base or "HEAD", extra_tests=args.extra_test,
                                 full_tests=args.full_tests, test_reason=args.test_reason)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"selection failed: {exc}", file=sys.stderr)
        return 2
    print(f"=== {selection.tier}: {len(selection.files)}/{selection.available} Python modules; "
          f"{selection.record()['rule']}; base {selection.base}; complete Node lane", flush=True)
    # Retained diagnostics; process ownership belongs to _run, as in the full CLI.
    root = evidence or pathlib.Path(tempfile.mkdtemp(prefix="ouroboros-selected-"))
    root.mkdir(parents=True, exist_ok=True)
    record = selection.record()
    record["inventory"] = {}
    record["outcomes"] = {"parallel": "skipped", "serial": "skipped"}
    output = root / "selection.json"
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"=== selection evidence: {output}", flush=True)
    if args.defer_full and selection.tier == "full":
        selection = _defer_full(selection)
        record = {**selection.record(), "inventory": {}, "outcomes": record["outcomes"], "deferred_full": True}
        output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    code = _node_lane()
    record["node"] = "success" if code == 0 else "failure"
    red = code  # with --keep-going a later lane runs on, but never hides an earlier red
    previous = os.environ.get("PYTHONPATH")
    completed = False
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (str(_probe_dir(root)), previous)))
    try:
        probe = _install_worker_probe(root)
        for spec in selection_passes(selection, probe_module=probe, temp_root=root) if code == 0 or args.keep_going else []:
            command = [sys.executable, "-m", "pytest", *spec.args,
                       "--ci-evidence-dir", str(root / spec.label)]
            code = _run(spec.label, command, REPO)
            inventory, problem = read_inventory(root, spec.label)
            record["inventory"][spec.label] = inventory
            record["outcomes"][spec.label] = "success" if code in (0, _PYTEST_EXIT_NO_TESTS) and not problem else "failure"
            output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            if code not in (0, _PYTEST_EXIT_NO_TESTS):
                red = red or code
                if not args.keep_going:
                    break
                continue
            code = int(bool(problem))
            if problem:
                print(f"inventory failed: {problem}", file=sys.stderr)
                red = red or code
                if not args.keep_going:
                    break
        code = red or code
        if code == 0 and (problem := inventory_problem(selection, record["inventory"])):
            print(problem, file=sys.stderr)
            code = 1
        completed = True
    finally:
        if previous is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = previous
        record["outcome"] = "success" if completed and code == 0 else "failure"
        output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        if destination := os.environ.get("GITHUB_OUTPUT"):
            with open(destination, "a", encoding="utf-8") as stream:
                for label, outcome in record["outcomes"].items():
                    stream.write(f"{label}_outcome={outcome}\n")
    print(f"=== {selection.tier} battery: {'GREEN' if code == 0 else 'RED'}", flush=True)
    return code


def _defer_full(selection):
    """A full escalation this leg cannot finish (Windows runs full in eight landed shards).

    The leg still runs the contract floor, the Node lane and what the changes that did not
    escalate select by themselves (changed test modules, direct importers, widening), and
    says what it left to the landed matrix.
    """
    import dataclasses

    from ouroboros.test_selection import FLOOR_TESTS, select_from_paths

    escalating = ("verification apparatus changed", "body cutover/recovery changed")
    why = "; ".join(f"{path}: {reason}" for path, reason in selection.reasons
                    if reason in escalating or reason.startswith("author requested full"))
    rest = [path for path in selection.changed if (path, escalating[0]) not in selection.reasons
            and (path, escalating[1]) not in selection.reasons]
    try:
        chosen = select_from_paths(REPO, rest, base=selection.base, extra_tests=selection.extra_tests,
                                   test_reason=selection.test_reason) if rest else None
        files = chosen.files if chosen is not None and chosen.tier != "full" else ()
        if chosen is not None and chosen.tier == "full":  # an escalation not named above: the floor alone
            why = f"{why}; the other changes escalate too ({'; '.join(r for _p, r in chosen.reasons)})"
    except (ValueError, OSError, subprocess.SubprocessError) as exc:  # the floor still runs
        files, why = (), f"{why}; the other changes could not be selected: {exc}"
    note = (f"full scope deferred to the landed matrix; this leg runs the contract floor, the Node lane and "
            f"{len(set(files) - set(FLOOR_TESTS))} module(s) the other changes select ({why})")
    print(f"=== {note}", flush=True)
    if destination := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(destination, "a", encoding="utf-8") as stream:
            stream.write(f"**{note}**\n")
    return dataclasses.replace(selection, tier="selected", files=tuple(sorted(set(FLOOR_TESTS) | set(files))))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequential", action="store_true", help="two passes, exactly as the commit gate and CI")
    parser.add_argument("--selected", action="store_true", help="floor + affected tests and complete Node lane")
    parser.add_argument("--base", help="Git base for --selected; defaults to HEAD")
    parser.add_argument("--extra-test", action="append", default=[], help="widen selection with a test module")
    parser.add_argument("--full-tests", action="store_true", help="widen --selected to the full battery")
    parser.add_argument("--test-reason", default="", help="reason for authored widening")
    parser.add_argument("--defer-full", action="store_true",
                        help="with --selected: on a full escalation run only the floor and Node lane, and say so")
    parser.add_argument("--keep-going", action="store_true",
                        help="with --selected: a red lane does not skip the next one (the post-merge check "
                             "wants every failure of the run, not the first)")
    parser.add_argument("--node", action="store_true", help="only the complete Node lane, owned like every pass")
    parser.add_argument("--evidence-dir", type=pathlib.Path,
                        help="write pytest diagnostics under DIR/<phase>; never inherited by nested pytest",
                        metavar="DIR")
    args, pytest_args = parser.parse_known_args(argv)
    evidence = args.evidence_dir.resolve() if args.evidence_dir is not None else None
    explicit_pytest_evidence = any(arg.split("=", 1)[0] == "--ci-evidence-dir" for arg in pytest_args)
    if evidence is None and not explicit_pytest_evidence and os.environ.get("OUROBOROS_TEST_EVIDENCE_ROOT"):
        evidence = pathlib.Path(os.environ["OUROBOROS_TEST_EVIDENCE_ROOT"]) / "pytest"
    if not (REPO / "tests" / "conftest.py").is_file():
        print(f"{REPO} is not an Ouroboros checkout (no tests/conftest.py)", file=sys.stderr)
        return 2
    if args.node:
        if args.selected or args.sequential or pytest_args:
            parser.error("--node runs only the Node lane")
        return _node_lane()
    if args.selected:
        if pytest_args:
            parser.error("--selected cannot be combined with arbitrary pytest arguments")
        return _selected(args, evidence)
    if args.base or args.extra_test or args.full_tests or args.test_reason or args.defer_full or args.keep_going:
        parser.error("--base, --defer-full, --keep-going and widening options require --selected")
    if pytest_args:
        if args.sequential:
            parser.error("--sequential selects the full two-pass battery; drop it or drop the pytest arguments")
        if evidence is not None:
            pytest_args = ["--ci-evidence-dir", str(evidence / "focused"), *pytest_args]
        return _run("focused", [sys.executable, "-m", "pytest", *pytest_args], REPO)
    started = time.monotonic()
    print(f"=== full: {sum(1 for _ in (REPO / 'tests').rglob('test_*.py'))} Python modules; "
          "full default suite; complete Node lane", flush=True)
    code = _node_lane()
    ran = 0
    for label, command in battery_commands(args.sequential) if code == 0 else []:
        if evidence is not None:
            command = [*command, "--ci-evidence-dir", str(evidence / label)]
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
    # CLI execution always acquires a real outer owner. A merely present temp
    # variable cannot prove one exists; in-process main() remains a plain helper.
    bootstrap = "import runpy,sys; raise SystemExit(runpy.run_path(sys.argv[1])['main'](sys.argv[2:]))"
    owner = [sys.executable, "-I", "-S", str(REPO / "scripts/safe_test.py"), "--",
             sys.executable, "-c", bootstrap, str(pathlib.Path(__file__).resolve()), *sys.argv[1:]]
    if os.name == "nt":
        sys.exit(subprocess.call(owner))  # Windows exec is a new process; keep the waiting parent.
    os.execv(sys.executable, owner)  # POSIX: the signalled PID is the owner itself.
