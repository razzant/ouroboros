"""The checks a prepared Update passes in its own workspace before it is ``ready``.

They run over the final composition W' (the merge commit plus the carried staged and
unstaged layers), never over the serving checkout and never with its environment: each
run is pytest under ``test_environment.isolated_environment`` (HOME, data root and caches in a
short disposable root, ``_process_root``; no credentials, pip refused, the whole tree reaped at its deadline), started
through the shell's tracked process owner (``shell_process._tracked_subprocess_run``: its
own process group, registered before it can outlive the step, refused once Emergency Stop
closed command admission); the packages it imports are the generation's own
(``update_prepare.prepare_generation_environment``), and when the Update brings a new app,
W' runs on the interpreter that will run it: that app's staged one, directly over its own user
site, or through the clone of a ``--runtime-venv`` of the installed app's Python, which names that
interpreter while the checks run (``runtime_venv.point_base``); the clone of a venv of the owner's
own Python runs on that Python, which the venv keeps.

What runs is this installation's own part (Update plan revision 6: a council-informed technical
choice within the owner's chosen outcomes, not the owner's wording): the shared selector
(``test_selection.select_from_paths``), unchanged, over what W' adds to the pinned official
target T, every path where W' differs from T, plus the test modules this installation's own
work carried in W (``_contribution``), so an adaptation that renames or drops one names it
instead of losing it. The required floor always runs: T's own, read from T's selector without running
it (never W''s, never this process's), so an official rename of a witness is followed and a local
removal fails the checks, named; a T floor of no shape read here leaves them unverified. A local
cutover, verifier or lock change keeps the selector's full tier, while a change the official update
alone brings never escalates.
It runs as the commit gate runs a battery: the non-serial cases spread over the admitted
workers, then the serial ones (``preflight_runner``'s lanes, ``test_parallelism.worker_budget``),
under one deadline, each lane's cases and seconds recorded. Only the cases that fail there run
again on W, the composition serving now, by their complete pytest node ids (classes and
parameters included): a case that fails on W with the same signature in a complete run is named
as already failing and does not block; any other failure blocks, and one the official target
fails the same way here is attributed to it (``target_failing``), never waived. A run that did
not complete (its deadline, a refused start, a collection error, a crashed test worker, nothing
collected) leaves the candidate unverified: never a test failure and never ready. Changed paths
no selected test reaches by name or direct import are named as unknown coverage, never a verdict.
Each run keeps what it saw beside its JUnit reports, in its own ``junit-*`` folder of the candidate's
scratch (retired with the checkout): the exact selection, commit, interpreter, targets (the argument file
every lane reads) and argv, and every lane's complete output (``run_tests``); the checks record names that
folder as the run's ``evidence``.
"""

from __future__ import annotations

import logging
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ElementTree
from typing import Any, Dict, List, Set, Tuple

log = logging.getLogger(__name__)

_TEST_TIMEOUT_SEC = 3600
# xdist's own report of a worker lost mid-case (it runs the rest elsewhere and exits 1): no test result, an outage.
_WORKER_CRASH = "crashed while running"
_SELECTOR = "ouroboros/test_selection.py"  # where the official target declares its required floor


def _materialize(workspace: pathlib.Path, commit: str) -> None:
    from supervisor.update_prepare import _run  # the tracked Git owner: a begun stop refuses it, typed

    _run(workspace, "reset", "--hard", "--quiet", commit)


def run_tracked(cmd: List[str], cwd: pathlib.Path, timeout: float, env=None) -> Dict[str, Any]:
    """``cmd`` through the shell's tracked process owner: ``returncode`` 124 at the deadline (its whole
    tree reaped), 125 when nothing started: Emergency Stop has retired command admission, or the system
    refused the start (a missing interpreter, a command line it does not take). ``output`` names
    what ``stdout``/``stderr`` hold: ``complete``; at the deadline ``partial``, what the child printed before
    it (POSIX keeps it on the timeout), or ``unknown`` where the platform keeps none; ``none`` if nothing ran."""
    from ouroboros.tools.shell_process import _tracked_subprocess_run

    try:
        done = _tracked_subprocess_run(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        out, err = (part.decode("utf-8", "replace") if isinstance(part, bytes) else part or ""
                    for part in (exc.stdout, exc.stderr))
        return {"returncode": 124, "stdout": out, "output": "unknown" if os.name == "nt" else "partial",
                "stderr": "\n".join(filter(None, (err.rstrip("\n"), f"timed out after {timeout:.0f}s")))}
    except RuntimeError as exc:  # the Panic latch: refused before any process started
        return {"returncode": 125, "stdout": "", "stderr": str(exc), "output": "none"}
    except (OSError, ValueError) as exc:  # Popen refused it; the owner marks that nothing started
        if not getattr(exc, "process_not_started", False):
            raise
        return {"returncode": 125, "stdout": "", "stderr": f"not started: {type(exc).__name__}: {exc}",
                "output": "none"}
    return {"returncode": done.returncode, "stdout": done.stdout or "", "stderr": done.stderr or "",
            "output": "complete"}


def _nodeid(case) -> str:
    """The case's complete pytest node id: ``file::Class::Nested::name[params]``.

    xunit1 reports carry the file and a dotted ``classname`` (module path plus any classes);
    the module part is the file, the rest are the classes. ``name`` keeps its parameters.
    """
    path = str(case.get("file") or "")
    classname = str(case.get("classname") or "")
    module = path[:-3].replace("/", ".") if path.endswith(".py") else ""
    if not path:  # no file attribute: the dotted module path is all there is
        path, classes = classname.replace(".", "/") + ".py", []
    elif module and classname.startswith(module + "."):
        classes = classname[len(module) + 1:].split(".")
    else:
        classes = []
    return "::".join([path, *classes, str(case.get("name") or "")])


def _cases(report: pathlib.Path) -> Dict[str, Any]:
    """Every test case of a JUnit report: ``{nodeid: failure signature or ""}`` (empty: it passed or skipped)."""
    cases: Dict[str, Any] = {}
    for case in ElementTree.parse(report).getroot().iter("testcase"):
        failure = case.find("failure") if case.find("failure") is not None else case.find("error")
        cases[_nodeid(case)] = "" if failure is None else (
            f"{failure.tag}:{failure.get('type') or ''}:{(failure.get('message') or '').strip().splitlines()[:1]}")
    return cases


def _process_root() -> pathlib.Path:
    """A short disposable root for the test processes' HOME, TMPDIR and caches, where the commit gate keeps its
    own (``test_environment.default_temp_parent``): nested in a candidate's scratch, a multiprocessing socket
    outgrows macOS's 104-byte AF_UNIX limit, and a forkserver floor case then fails on W' and W alike, as if
    already failing. ``selected_tests`` removes it when its runs end; each run's reports stay in the scratch."""
    from ouroboros.test_environment import default_temp_parent

    return pathlib.Path(tempfile.mkdtemp(prefix="ouroboros-update-checks-", dir=str(default_temp_parent())))


def _test_environment(row: Dict[str, Any], root: pathlib.Path) -> Dict[str, str]:
    """HOME, data root and caches beneath ``root`` (``_process_root``; ``test_environment.isolated_environment``)
    with the packages of the generation under test: its own when the lock changed, otherwise the serving ones. A venv
    (a private environment too) brings its packages through its interpreter (``_interpreters``); a user site through
    ``PYTHONUSERBASE``."""
    from ouroboros.test_environment import isolated_environment
    from supervisor.update_prepare import serving_user_base

    env = isolated_environment(root, pathlib.Path(row["path"]))
    generation = (row.get("environment") or {}).get("generation") or {}
    if generation.get("kind") not in ("venv", "private"):
        env["PYTHONNOUSERSITE"] = ""  # the user site enabled (an empty value is unset for Python)
        env["PYTHONUSERBASE"] = str(generation.get("user_base") or serving_user_base())
    return env


def _interpreters(row: Dict[str, Any], env: Dict[str, str]) -> Tuple[str, str, Dict[str, str]]:
    """``(W' interpreter, W interpreter, W environment)``: W always runs on the packages serving now, W'
    on the interpreter that will run it (a new app's own, when the Update brings one)."""
    from ouroboros.runtime_venv import interpreter

    generation = (row.get("environment") or {}).get("generation") or {}
    if generation.get("kind") in ("venv", "private"):
        return interpreter(generation["user_base"]), sys.executable, env
    return generation.get("python") or sys.executable, sys.executable, {
        **env, "PYTHONUSERBASE": generation.get("current") or env["PYTHONUSERBASE"]}


def _lanes(workers: int) -> List[Tuple[str, List[str]]]:
    """The commit gate's two lanes (``preflight_runner``): the non-serial cases over ``workers`` xdist workers, then
    the serial ones alone, each with CI's per-test hang bound."""
    from ouroboros.preflight_runner import LANE_EXCLUSION_EXPR, PARALLEL_PASS_FLAGS, SERIAL_PASS_FLAGS

    return [("parallel", ["-m", f"not serial and {LANE_EXCLUSION_EXPR}",
                          *(str(workers) if flag == "auto" else flag for flag in PARALLEL_PASS_FLAGS)]),
            ("serial", ["-m", f"serial and {LANE_EXCLUSION_EXPR}", *SERIAL_PASS_FLAGS])]


def run_tests(workspace: pathlib.Path, commit: str, targets: List[str], scratch: pathlib.Path,
              env: Dict[str, str] = None, python: str = "", record: Dict[str, Any] = None) -> Dict[str, Any]:
    """Run ``targets`` (files or node ids) on ``commit`` materialized in the workspace, in the gate's two lanes.

    ``complete`` only when every lane finished the collection it was given within the one
    deadline (exit 0 or 1, or 5 for a lane none of these cases belongs to), the lanes reported
    cases and no test worker crashed (``crashed``); anything else is unverified, never a pass,
    never "no failures" and never a failure a baseline could excuse.
    ``failed`` maps each failing node id to its failure signature (kind, exception type,
    first message line); ``lanes`` keeps each lane's exit, case count and seconds.
    ``evidence`` is the run's own folder beside its reports: ``run.json`` (commit, interpreter, user
    base, targets, ``record`` (what chose them) and each lane's exact argv, exit, cases, seconds and
    ``output``) and each lane's complete ``<lane>.stdout.txt`` and ``<lane>.stderr.txt``; ``output``
    names a lane cut at the deadline (``partial`` or ``unknown``, ``run_tracked``) or never started (``none``).
    The targets reach every lane through pytest's own argument file (``@targets.txt`` there, one per line, read
    again by each xdist worker): a full tier names every test module, which no command line takes on Windows
    (CreateProcess stops at 32767 characters); ``run.json`` names it (``targets_file``) beside the exact targets.
    """
    from ouroboros.test_parallelism import worker_budget
    from ouroboros.utils import atomic_write_json, write_text_atomic

    if not targets:
        raise ValueError("no test targets: an empty selection would run the whole suite")
    _materialize(workspace, commit)
    scratch.mkdir(parents=True, exist_ok=True)
    folder = pathlib.Path(tempfile.mkdtemp(prefix="junit-", dir=str(scratch)))
    targets_file = folder / "targets.txt"
    write_text_atomic(targets_file, "".join(f"{target}\n" for target in targets))
    started = time.monotonic()
    deadline, cases, lanes, tails = started + _TEST_TIMEOUT_SEC, {}, [], []
    manifest = {"commit": commit, "python": python or sys.executable, "user_base": (env or {}).get("PYTHONUSERBASE"),
                "tmpdir": (env or {}).get("TMPDIR"), "targets": list(targets), "targets_file": str(targets_file),
                "deadline_seconds": _TEST_TIMEOUT_SEC, **(record or {}), "lanes": []}
    for label, flags in _lanes(worker_budget(os.cpu_count() or 1)):
        report, begun = folder / f"{label}.xml", time.monotonic()
        cmd = [python or sys.executable, "-m", "pytest", f"@{targets_file}", "-q", "-p", "no:cacheprovider", "-o",
               "junit_family=xunit1", f"--junitxml={report}", *flags]
        run = (run_tracked(cmd, workspace, deadline - begun, env=env) if deadline > begun else
               {"returncode": 124, "stdout": "", "stderr": "the deadline passed before this lane", "output": "none"})
        for stream in ("stdout", "stderr"):
            write_text_atomic(folder / f"{label}.{stream}.txt", run[stream])
        try:
            found = _cases(report)
        except (OSError, ElementTree.ParseError):
            found = {}
        cases.update(found)
        lanes.append({"lane": label, "returncode": run["returncode"], "cases": len(found),
                      "seconds": round(time.monotonic() - begun, 1)})
        manifest["lanes"].append({**lanes[-1], "argv": cmd, "output": run["output"]})
        if run["returncode"] not in (0, 5):
            tails.append(f"[{label}] " + (run["stdout"] or run["stderr"])[-2000:])
    failed = {nodeid: signature for nodeid, signature in cases.items() if signature}
    crashed = sorted(nodeid for nodeid, signature in failed.items() if _WORKER_CRASH in signature)
    complete = not crashed and bool(cases) and all(lane["returncode"] in (0, 1) and lane["cases"]
                                                   or lane["returncode"] == 5 and not lane["cases"] for lane in lanes)
    returncode = next((lane["returncode"] for lane in lanes if lane["returncode"] not in (0, 5)), 0)
    result = {"complete": complete, "ok": complete and returncode == 0 and not failed, "commit": commit,
              "failed": failed, "crashed": crashed, "cases": len(cases), "returncode": returncode, "lanes": lanes,
              "seconds": round(time.monotonic() - started, 1), "tail": "\n".join(tails)[-2000:],
              "evidence": str(folder)}
    atomic_write_json(folder / "run.json", {**manifest, **{key: result[key] for key in (
        "complete", "returncode", "cases", "failed", "crashed", "seconds")}})
    return result


def _names(workspace: pathlib.Path, old: str, new: str) -> List[str]:
    from supervisor.update_prepare import _paths, _run  # the tracked Git owner

    return _paths(_run(workspace, "diff-tree", "-r", "-z", "--no-renames", "--name-only", old, new).stdout)


def _contribution(workspace: pathlib.Path, row: Dict[str, Any]) -> Tuple[Set[str], Set[str]]:
    """``(footprint, carried tests)``: every path where W' differs from the official target T, and the test modules
    this installation's own work carried in W (W against the official commit H was last based on)."""
    from ouroboros.test_selection import is_test_module
    from supervisor.update_prepare import _out

    layers, target = row["layers"], row["target"]["target_sha"]
    official = _out(workspace, "merge-base", layers["old_commit"], target)
    carried = {path for path in _names(workspace, official, layers["old_files"]) if is_test_module(path)}
    return set(_names(workspace, target, layers["new_files"])), carried


def _uncovered(selection, footprint: Set[str]) -> Dict[str, Any]:
    """The changed paths no selected test reaches by name or direct import: unknown coverage, named, never a verdict
    (a full tier runs every Python test, so then only other files)."""
    from ouroboros.test_selection import _module, is_test_module

    bound = {name.strip() for _path, why in selection.reasons if why.startswith("direct import: ")
             for name in why[len("direct import: "):].split(",")}
    paths = sorted(path for path in footprint if not is_test_module(path) and not (
        path.endswith(".py") and (selection.tier == "full" or _module(path) in bound)))
    return {"uncovered": paths[:50], "uncovered_count": len(paths)}


def _incomplete(run: Dict[str, Any]) -> str:
    lanes = ", ".join(f"{lane['lane']}: exit {lane['returncode']}, {lane['cases']} case(s)" for lane in run["lanes"])
    crashed = run.get("crashed") or []
    return (f"the selected run did not complete ({lanes}"
            + (f"; a test worker crashed while running {', '.join(crashed[:3])}" if crashed else "")
            + "); 124 is its deadline, 125 a refused start, 2 an interrupted collection, 5 nothing collected")


def _target_failing(workspace: pathlib.Path, row: Dict[str, Any], failed: Dict[str, str], scratch: pathlib.Path,
                    env: Dict[str, str], python: str) -> Dict[str, Any]:
    """Which blocking cases the official target fails the same way on this installation: attribution for the owner
    and the mind, never a waiver. Only those cases, only where T has their file, on the packages W' runs on."""
    from supervisor.update_prepare import _run

    target = row["target"]["target_sha"]
    present = sorted(nodeid for nodeid in failed if _run(workspace, "cat-file", "-e",
                                                         f"{target}:{nodeid.split('::', 1)[0]}", check=False)
                     .returncode == 0)
    if not present:
        return {"target_failing": [], "target_run": {"cases": 0}}
    run = run_tests(workspace, target, present, scratch, env, python)
    return {"target_failing": sorted(nodeid for nodeid in present
                                     if run["complete"] and run["failed"].get(nodeid) == failed[nodeid]),
            "target_run": {key: run[key] for key in ("complete", "cases", "returncode", "seconds", "evidence")}}


def _target_floor(workspace: pathlib.Path, target: str) -> Dict[str, Any]:
    """The pinned official target's required floor, read from its own selector (``test_selection.floor_of``), never
    from W' or from this process: ``{"source", "sha256", "tests"}``; ``ValueError`` says why T has none to read."""
    import hashlib
    import json

    from ouroboros.test_selection import floor_of
    from supervisor.update_prepare import _run  # the tracked Git owner

    source = f"{target}:{_SELECTOR}"
    shown = _run(workspace, "show", source, check=False)
    if shown.returncode != 0:
        raise ValueError(f"{source} cannot be read ({shown.stderr.decode('utf-8', 'replace').strip()[:300]})")
    try:
        tests = floor_of(shown.stdout.decode("utf-8"))  # undecodable bytes are a ValueError as well
    except ValueError as exc:
        raise ValueError(f"{source}: {exc}") from None
    return {"source": source, "sha256": hashlib.sha256(json.dumps(tests, sort_keys=True).encode()).hexdigest(),
            "tests": tests}


def selected_tests(row: Dict[str, Any], scratch: pathlib.Path) -> Dict[str, Any]:
    """This installation's part of W' (``_contribution``) through the shared selector, run on W'; a failure blocks
    unless that exact case fails on W, the composition serving now, with the same signature in a complete run;
    a blocking case is attributed on T, never waived; an incomplete run is unverified. The required floor is
    T's own (``_target_floor``): one W' lacks fails the checks, named, and one T does not declare in the shape
    read here leaves them unverified."""
    from ouroboros.test_selection import FloorMissing, select_from_paths

    workspace, composition, previous = pathlib.Path(row["path"]), row["composition_commit"], row["layers_base"]["k2"]
    footprint, carried = _contribution(workspace, row)
    try:
        floor = _target_floor(workspace, row["target"]["target_sha"])
    except ValueError as exc:  # nothing is selected against a floor guessed from another version
        return {"ok": False, "unverified": f"the official target's required floor is unknown: {exc}", "failed": {},
                "blocking": [], "floor": {"source": f"{row['target']['target_sha']}:{_SELECTOR}", "error": str(exc)}}
    _materialize(workspace, composition)
    dropped = sorted(path for path in carried if not (workspace / path).is_file())
    try:
        selection = select_from_paths(workspace, sorted(footprint | (carried - set(dropped))),
                                      base=row["target"]["target_sha"], floor=floor["tests"])
    except FloorMissing as exc:  # W' lacks a witness the official target requires: refused, never skipped
        return {"ok": False, "floor": floor, "floor_missing": exc.missing, "blocking": exc.missing, "failed": {},
                "changed": len(footprint), "local_tests": len(carried), "dropped_local_tests": dropped}
    root = _process_root()  # removed when these runs end, whatever ends them
    try:
        env = _test_environment(row, root)
        python, serving_python, serving_env = _interpreters(row, env)
        new = run_tests(workspace, composition, list(selection.files), scratch, env, python, record={
            "selection": selection.record(), "floor": floor, "footprint": sorted(footprint),
            "local_tests": sorted(carried), "dropped_local_tests": dropped})
        result = {"tier": selection.tier, "file_count": len(selection.files), "changed": len(footprint), "floor": floor,
                  "process_root": str(root),
                  "local_tests": len(carried), "dropped_local_tests": dropped, **_uncovered(selection, footprint),
                  "new": {key: new[key] for key in ("complete", "ok", "cases", "crashed", "returncode", "lanes", "seconds",
                                                    "tail", "evidence")},
                  "failed": new["failed"]}
        if not new["complete"]:
            return {**result, "ok": False, "unverified": _incomplete(new), "blocking": sorted(new["failed"])}
        if new["ok"]:
            return {**result, "ok": True, "blocking": [], "already_failing": []}
        old = run_tests(workspace, previous, sorted(new["failed"]), scratch, serving_env, serving_python)
        already = sorted(nodeid for nodeid, signature in new["failed"].items()
                         if old["complete"] and old["failed"].get(nodeid) == signature)
        blocking = sorted(set(new["failed"]) - set(already))
        if blocking:
            result.update(_target_failing(workspace, row, {nodeid: new["failed"][nodeid] for nodeid in blocking},
                                          scratch, env, python))
        _materialize(workspace, composition)
        return {**result, "ok": not blocking, "blocking": blocking, "already_failing": already,
                "baseline": {key: old[key] for key in ("complete", "failed", "returncode", "evidence")}}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def workspace_checks(row: Dict[str, Any]) -> Dict[str, Any]:
    """The checks for ``update_prepare.prepare``: ``ok`` is the selected tests' verdict."""
    from ouroboros.body_candidate import scratch_root

    # The candidate's declared sibling scratch: retired with its checkout, never kept as unknown data.
    scratch = scratch_root(row["path"]) / "update-checks"
    tests = selected_tests(row, scratch)
    return {"ok": bool(tests.get("ok")), "tests": tests}
