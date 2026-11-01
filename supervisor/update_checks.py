"""The checks a prepared Update passes in its own workspace before it is ``ready``.

They run over the final composition W' (the merge commit plus the carried staged and
unstaged layers), never over the serving checkout and never with its environment: each
run is pytest in the commit gate's own test environment (``preflight_runner._preflight_env``: HOME, data root,
caches and TMPDIR in a short disposable root, ``_process_root``; no credentials, pip refused, the Node the gate
selects first on PATH, the admitted worker budget for xdist and nested runners, the whole tree reaped at its
deadline), its tests' durable evidence in the candidate's scratch (the ``EVIDENCE_POINTER`` the gate reads), started
through the shell's tracked process owner (``shell_process._tracked_subprocess_run``: its
own process group, registered before it can outlive the step, refused once Emergency Stop
closed command admission), each test lane contained as the gate's passes are (``run_tracked``: nothing it started
outlives it); the packages it imports are the generation's own
(``update_prepare.prepare_generation_environment``), and when the Update brings a new app,
W' runs on the interpreter that will run it: that app's staged one, directly over its own user
site, or through the clone of a ``--runtime-venv`` of the installed app's Python, which names that
interpreter while the checks run (``runtime_venv.point_base``); the clone of a venv of the owner's
own Python runs on that Python, which the venv keeps. Packages in a user base reach the run through CPython's own
site processing (``PYTHONUSERBASE``: the stdlib first, then that user site with its ``.pth`` paths and startup lines,
then the interpreter's own site-packages), never through ``PYTHONPATH``, and that base is declared to the tested
tree's harness (``test_environment.INSTALLATION_USER_BASE_ENV``; a venv or private environment declares nothing). A
harness with the opt-in keeps its fresh user base for writes and makes the declared site readable by every process
that keeps its environment (its xdist workers and a test's own children, ``.pth`` startup lines included) and by one
it re-roots (``test_environment.isolated_environment``, a cleared environment); a child that isolates itself (``-I``,
``-s``, its own ``PYTHONNOUSERSITE=1``) keeps its choice. A harness
that predates the opt-in (a target such as the published O0-O1 range) gives its tests' own children none of the
installation's packages: no nested user-site coverage there, recorded per run (``user_site.harness``), and through
such a W a shared failure excuses nothing (``baseline.excuses``): it blocks, measured and attributed through T.

What runs is this installation's own part (Update plan revision 6: a council-informed technical
choice within the owner's chosen outcomes, not the owner's wording): the shared selector
(``test_selection.select_from_paths``), unchanged, over what W' adds to the pinned official
target T, every path where W' differs from T, plus the test modules this installation's own
work carried in W (``_contribution``), so an adaptation that renames or drops one names it
instead of losing it. The required floor always runs: T's own, read from T's selector without running
it (never W''s, never this process's), so an official rename of a witness is followed and a local
removal fails the checks, named; a T floor of no shape read here leaves them unverified. A local
cutover, verifier or lock change keeps the selector's full tier, while a change the official update
alone brings never escalates. The full tier names the suite directory, as the gate and CI do;
a selected tier its exact modules. A full tier over a tree with browser-module tests needs a Node at
their floor; one that is absent, cannot start, does not answer or is unusable leaves the checks
unverified before any test runs, named (``_node``).
It runs as the commit gate runs a battery: the non-serial cases spread over the admitted
workers, then the serial ones (``preflight_runner``'s lanes, ``test_parallelism.worker_budget``),
each lane's cases and seconds recorded. One monotonic budget (``config.UPDATE_CHECKS_TIMEOUT_SEC``,
or the gate's explicit ``OUROBOROS_PREFLIGHT_TIMEOUT_SEC``) begins just before W' is materialized and
bounds every process that follows (each materialization, test lane and T's one presence read), each getting what
remains of it; restoring W' afterwards, also after a check that raised, keeps the Git owner's own bound. A begun stop
that refused or ended one of those Git steps or a prerequisite probe raises its typed refusal, never read as their
answer. Only the cases that fail there run
again on W, the composition serving now, by their complete pytest node ids (classes and
parameters included): a case that fails on W with the same signature in a complete run is named
as already failing and does not block (unless W's harness predates the user-site opt-in on an installation whose
packages live in a user base, above); any other failure blocks, and one the official target
fails the same way here is attributed to it (``target_failing``), never waived. A run that did
not complete (its deadline, a refused start, a collection error, a crashed test worker, nothing
collected) leaves the candidate unverified: never ready. W' failures it measured stay named; a W
comparison that did not complete excuses none of them and shows none failing only with the update,
and an incomplete T attributes nothing. Changed paths
no selected test reaches by name or direct import are named as unknown coverage, never a verdict.
Each run keeps what it saw beside its JUnit reports, in its own ``junit-*`` folder of the candidate's
scratch (retired with the checkout): the exact selection, commit, interpreter, targets (the argument file
every lane reads), what remained of the budget and argv, its materialization's own output, and every lane's complete
output (``run_tests``); the checks record names that folder as the run's ``evidence``. T's presence read keeps its
input and output in a ``target-files-*`` folder beside them (``target_run.presence``).
"""

from __future__ import annotations

import ast
import logging
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ElementTree
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)

# xdist's own report of a worker lost mid-case (it runs the rest elsewhere and exits 1): no test result, an outage.
_WORKER_CRASH = "crashed while running"
_SELECTOR = "ouroboros/test_selection.py"  # where the official target declares its required floor
_PROBE_WORDS = {"unstartable": "could not be started", "timeout": "did not answer in time", "unusable": "failed",
                "absent": "is no existing absolute directory"}
# Both files of a tested tree's harness name it when that harness reads a declared installation user site
# (``test_environment.installation_user_site``); a tree without them predates the opt-in.
_HARNESS = ("tests/conftest.py", "ouroboros/test_environment.py")
# Why a failure W shares is not excused when W's harness does not read the declaration (``selected_tests``).
_UNEXCUSED = ("W's test harness {} the installation user-site opt-in: its tests' own children could not import the "
              "installation's packages, so a failure W shares proves nothing and blocks, attributed through T")
# The interpreter under test asked where it reads the declared base's user site-packages (the harness's own
# ``user_site_packages``, loaded by path): that path as an ASCII literal (``ascii``: a pipe in the locale's encoding,
# a Windows ANSI one among them, may not encode a profile path), then whether it exists, printed; exit 3 when the base
# itself is not an existing absolute directory. An existing base without them holds nothing to add (``_declared``).
_DECLARED_SITE = ("import os, runpy, sys; base = sys.argv[2]; site = runpy.run_path(sys.argv[1])['user_site_packages']"
                  "(base); print(ascii(site)); print(os.path.isdir(site)); "
                  "raise SystemExit(0 if os.path.isabs(base) and os.path.isdir(base) else 3)")


class _Unverified(Exception):
    """A prerequisite of the runs is not there: the checks are unverified, named, before any test runs.
    ``record`` keeps what was found (``prerequisite``, ``status`` and the probe's facts)."""

    def __init__(self, text: str, **record: Any):
        super().__init__(text)
        self.record = record


def _budget() -> int:
    """The checks' one budget in seconds: ``config.UPDATE_CHECKS_TIMEOUT_SEC``, unless the commit gate's explicit
    ``OUROBOROS_PREFLIGHT_TIMEOUT_SEC`` names another, shorter or longer (``preflight_runner``'s own resolver)."""
    from ouroboros.config import UPDATE_CHECKS_TIMEOUT_SEC
    from ouroboros.preflight_runner import _resolve_preflight_timeout

    return _resolve_preflight_timeout(UPDATE_CHECKS_TIMEOUT_SEC)


def _materialize(workspace: pathlib.Path, commit: str, timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    from supervisor.update_prepare import _run  # the tracked Git owner: a begun stop refuses it, typed

    return _run(workspace, "reset", "--hard", "--quiet", commit, timeout=timeout)


def _restore(workspace: pathlib.Path, composition: str) -> None:
    """W' in the workspace again after a W or T check that raised: outside the checks' budget, through the Git owner
    under its own bound. The check's own exception stays the answer; a restore refused or failed as well is logged."""
    try:
        _materialize(workspace, composition)
    except Exception:
        log.warning("the update workspace could not be restored to %s after its checks raised", composition,
                    exc_info=True)


def _kept(folder: pathlib.Path, name: str, argv: List[str], timeout: float, outcome: str,
          returncode: Optional[int], stdout: bytes, stderr: bytes) -> Dict[str, Any]:
    """One Git step's own output kept in ``folder`` as it came (``<name>.stdout.bin``, ``<name>.stderr.bin``; at its
    deadline the Git owner's termination message, git's own last words included), and its facts: ``outcome``
    ``complete`` (any exit), ``timeout`` or ``not_started``."""
    from ouroboros.utils import write_bytes_atomic

    for stream, content in (("stdout", stdout), ("stderr", stderr)):
        write_bytes_atomic(folder / f"{name}.{stream}.bin", content or b"")
    return {"argv": list(argv), "timeout": round(max(timeout, 0), 1), "outcome": outcome, "returncode": returncode,
            "stdout": f"{name}.stdout.bin", "stderr": f"{name}.stderr.bin"}


def run_tracked(cmd: List[str], cwd: pathlib.Path, timeout: float, env=None,
                outputs: Optional[Tuple[pathlib.Path, pathlib.Path]] = None,
                custody: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """``cmd`` through the shell's tracked process owner: ``returncode`` 124 at the deadline (its whole
    tree reaped), 125 when nothing started: Emergency Stop has retired command admission, or the system
    refused the start (a missing interpreter, a command line it does not take). ``output`` names
    what ``stdout``/``stderr`` hold: ``complete``; at the deadline ``partial``, what the child printed before
    it (POSIX keeps it on the timeout), or ``unknown`` where the platform keeps none; ``none`` if nothing ran.

    With ``outputs`` (a stdout and a stderr file) the run is contained as the commit gate contains a pass
    (``preflight_runner._execute_pytest_pass``: ``ProcessContainer(lifeline=True, overload=True)``, its streams those
    regular files, which no detached descendant holds open as it would a pipe), and whatever ended it, its whole
    attributed tree is reaped before this returns, a descendant in its own session or reparented to init included.
    The files keep what the child printed up to its deadline on every platform (``partial``); ``stdout``/``stderr``
    are read back from them. ``containment`` names what that reap could not prove gone ("" when it proved all of it
    gone); the run's test root (``OUROBOROS_TEST_TEMP_ROOT``) is then marked retained, the enclosing layer's retention
    protocol (``test_environment.retain_tree``), and ``custody`` is handed it as soon as the reap names it, before
    anything here can raise: a caller's release keeps its root on it whether or not this returns or a marker is
    written."""
    from ouroboros.tools.shell_process import _tracked_subprocess_run

    if outputs is not None:
        return _contained(cmd, cwd, timeout, env, *outputs, custody=custody)
    return _ended(lambda: _tracked_subprocess_run(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                                                  stderr=subprocess.PIPE, text=True, timeout=timeout), timeout)


def _ended(start, timeout: float) -> Dict[str, Any]:
    """What one tracked start ended as (``run_tracked``)."""
    try:
        done = start()
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


def _contained(cmd: List[str], cwd: pathlib.Path, timeout: float, env, stdout: pathlib.Path,
               stderr: pathlib.Path, custody: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """``run_tracked`` with ``outputs``: started through the tracked owner (its admission, publication and deadline
    kill) by the container's own ``spawn``, reaped before anything it wrote is read or released. As the gate waits its
    direct process after its reap, a root whose exit the owner's kill did not see (it stays tracked) is waited then,
    and once it exited leaves the tracked registry; one still running is named in ``containment``."""
    from ouroboros.process_containment import ProcessContainer
    from ouroboros.test_environment import retain_tree
    from ouroboros.tools.shell_process import _active_subprocesses, _tracked_subprocess_run

    container, containment = ProcessContainer(lifeline=True, overload=True), "the contained run was not reaped"
    spawned: List[subprocess.Popen] = []

    def spawn(argv, **kwargs):
        spawned.append(container.spawn(argv, **kwargs))
        return spawned[-1]

    try:
        with stdout.open("wb") as out, stderr.open("wb") as err:
            run = _ended(lambda: _tracked_subprocess_run(cmd, cwd=str(cwd), env=env, stdout=out, stderr=err,
                                                         timeout=timeout, spawn=spawn), timeout)
    finally:
        try:
            containment = container.reap()  # also after green: a detached member never waits on a file's EOF
            for proc in spawned:
                if proc.poll() is None:
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        containment = containment or f"pid {proc.pid} did not exit after its reap"
                if proc.returncode is not None:
                    _active_subprocesses.discard(proc)
        finally:
            if containment and custody is not None:
                custody(containment)
            container.close()
            root = (env or {}).get("OUROBOROS_TEST_TEMP_ROOT")
            if containment and root:
                retain_tree(root, f"update checks: a lane's tree was not proven gone: {containment}")
    printed = [path.read_bytes().decode("utf-8", "replace") for path in (stdout, stderr)]
    said = "\n".join(filter(None, (printed[1].rstrip("\n"), run["stderr"]))) if run["stderr"] else printed[1]
    return {**run, "stdout": printed[0], "stderr": said, "containment": containment,
            "output": "partial" if run["output"] == "unknown" else run["output"]}


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
    already failing. Named as ``scripts/safe_test.py`` names its root (``ob-``, without the ``.noindex`` that would
    take this margin): the Panic fixture's forkserver listener (``<root>/b0/t/popen-gw1/<test>/tmp/pymp-*/listener-*``)
    is then 99 bytes on Linux, within the 107 CPython binds there (108 with its NUL), where
    ``ouroboros-update-checks-`` made it 120 and stopped the fixture before Panic. On macOS ``/tmp`` resolves 8 bytes
    longer, past 103, and that fixture forks by spawn there. ``selected_tests`` removes it when its runs end unless
    something beneath it is marked retained, then keeps and names it (``_release``); each run's reports stay in the
    scratch."""
    from ouroboros.test_environment import default_temp_parent

    return pathlib.Path(tempfile.mkdtemp(prefix="ob-", dir=str(default_temp_parent())))


def _release(root: pathlib.Path, unproven: str = "") -> Dict[str, Any]:
    """The process root once its runs ended, released as the commit gate releases its own
    (``preflight_runner._release_hermetic_tree``): removed only when nothing beneath it is marked retained
    (``test_environment.retention_markers``: a lane whose reap could not prove its tree gone, ``run_tracked``; a nested
    runner or the tested tree's own harness that could not prove its processes gone; a subtree that cannot be walked)
    and no lane names a tree its reap could not prove gone (``unproven``, handed over by each run as its reap names it,
    also when that run then raises: as in the gate, that decision does not depend on the marker's write), else kept, logged and named with why: ``{"state": "removed" | "retained",
    "path"}``, a kept one with its ``markers`` (empty when none could be written) and ``reason``. A removal that fails
    keeps and marks it too."""
    from ouroboros.test_environment import retain_tree, retention_markers
    from ouroboros.test_evidence import remove_scratch_tree

    markers = retention_markers(root)
    if not markers and not unproven:
        try:
            remove_scratch_tree(root)
            return {"state": "removed", "path": str(root)}
        except OSError as exc:
            retain_tree(root, f"update checks: the process root could not be removed: {exc}")
            markers = retention_markers(root) or [root]
    elif not markers:
        retain_tree(root, f"update checks: a lane's tree was not proven gone: {unproven}")
        markers = retention_markers(root)
    reasons = []
    for marker in markers:
        try:
            reasons.append(marker.read_text(encoding="utf-8").strip())
        except (OSError, UnicodeError):
            reasons.append(f"{marker} could not be read")
    reason = ("; ".join(filter(None, reasons)) or f"update checks: a lane's tree was not proven gone: {unproven}")[:2000]
    log.warning("the update checks kept their process root %s: %s", root, reason)
    return {"state": "retained", "path": str(root), "markers": [str(marker) for marker in markers], "reason": reason}


def _probe(cmd: List[str], root: pathlib.Path, env: Dict[str, str], subject: str, missing: Optional[int] = None,
           **record: Any) -> Dict[str, Any]:
    """One prerequisite probe through the tracked owner, in ``root``, bounded by
    ``config.UPDATE_CHECKS_PROBE_TIMEOUT_SEC`` before the checks' budget begins. A begun stop raises its typed refusal
    (``update_prepare.stopped``), never read as the probe's answer; one that could not start, ran out of time or
    failed is ``_Unverified``, with what it printed (``missing``: the exit that means what it looked for is absent)."""
    from ouroboros.config import UPDATE_CHECKS_PROBE_TIMEOUT_SEC
    from supervisor.update_prepare import stopped

    done = run_tracked(cmd, root, UPDATE_CHECKS_PROBE_TIMEOUT_SEC, env=env)
    if done["returncode"] == 0:
        return done
    stopped()
    status = {125: "unstartable", 124: "timeout", missing: "absent"}.get(done["returncode"], "unusable")
    detail = (done["stderr"] or done["stdout"]).strip()[-600:] or f"exit {done['returncode']}"
    raise _Unverified(f"{subject} {_PROBE_WORDS[status]} ({detail})", **record, status=status, argv=cmd,
                      returncode=done["returncode"], detail=detail)


def _environments(row: Dict[str, Any], root: pathlib.Path,
                  evidence: pathlib.Path) -> Tuple[Tuple[str, Dict[str, str]], Tuple[str, Dict[str, str]]]:
    """``((W' interpreter, its environment), (W interpreter, its environment))``; T runs as W' does.

    Both start from the commit gate's own (``preflight_runner._preflight_env``: beneath ``root``, scrubbed, the Node
    it selects first on PATH, this budget of xdist and nested workers) with the tests' durable evidence in
    ``evidence``, outside ``root`` (its ``EVIDENCE_POINTER``). W' runs on the interpreter that will run it (a new app's
    own, when the Update brings one) and the packages of the generation under test: its own when the lock changed,
    otherwise the serving ones; W always on the packages serving now. A venv (a private environment too, or an
    interpreter that keeps them itself) brings its packages through its interpreter and declares nothing. Packages in
    a user base come through ``PYTHONUSERBASE`` (``_user_base``), so CPython's own site processing orders them (the
    stdlib first, then that user site with its ``.pth`` paths and startup lines, then the interpreter's own
    site-packages), and that base is declared to the tested tree's harness, whose processes keep a fresh user base of
    their own: a harness with the opt-in (``test_environment.installation_user_site``) makes the declared site readable
    by every process that keeps its environment; one that predates it leaves its xdist workers without the ``.pth``
    startup lines and a test's own children without the installation's packages (``_harnesses``)."""
    from ouroboros.preflight_runner import _preflight_env
    from ouroboros.runtime_venv import interpreter
    from ouroboros.test_evidence import EVIDENCE_POINTER
    from ouroboros.test_parallelism import worker_budget
    from supervisor.update_prepare import serving_environment, serving_user_base

    (root / EVIDENCE_POINTER).write_text(str(evidence), encoding="utf-8")
    env = _preflight_env(root, pathlib.Path(row["path"]), workers=worker_budget(os.cpu_count() or 1))
    generation = (row.get("environment") or {}).get("generation") or {}
    if generation.get("kind") in ("venv", "private"):
        return (interpreter(generation["user_base"]), env), (sys.executable, env)
    kind, serving = ("user_site", generation.get("current")) if generation else serving_environment()
    if kind != "user_site":  # no generation, and the serving packages are the interpreter's own
        return (sys.executable, env), (sys.executable, env)
    serving = str(serving or serving_user_base())
    return ((generation.get("python") or sys.executable, _user_base(env, str(generation.get("user_base") or serving))),
            (sys.executable, _user_base(env, serving)))


def _user_base(env: Dict[str, str], base: str) -> Dict[str, str]:
    """``env`` with ``base`` the user base the run reads (an empty PYTHONNOUSERSITE is unset for Python), declared to
    the tested tree's harness as where the installation keeps its packages (``INSTALLATION_USER_BASE_ENV``)."""
    from ouroboros.test_environment import INSTALLATION_USER_BASE_ENV

    return {**env, "PYTHONNOUSERSITE": "", "PYTHONUSERBASE": base, INSTALLATION_USER_BASE_ENV: base}


def _declared(runs: List[Tuple[str, Dict[str, str]]], root: pathlib.Path) -> List[str]:
    """Each run's declared installation user base must be an existing absolute directory, asked of the interpreter
    that reads it through ``_probe`` (the harness's own ``test_environment.user_site_packages``): before any test runs,
    a missing one is a prerequisite named ``absent``, never a silent fallback. Returns, each once, the declared site
    paths that do not exist: an installation whose packages are all the interpreter's own (a packaged boot's
    ``pip --user`` the bundle already satisfied writes none) adds nothing there, its tests run, the base still
    declared. W' and W are asked independently: one may hold packages, the other none."""
    from ouroboros import test_environment

    asked, empty = set(), []
    for python, env in runs:
        base = env[test_environment.INSTALLATION_USER_BASE_ENV]
        if (python, base) not in asked:
            asked.add((python, base))
            done = _probe([python, "-c", _DECLARED_SITE, test_environment.__file__, base], root, env,
                          f"the installation user base {base} declared for {python}", missing=3,
                          prerequisite="user_site", user_base=base, python=python)
            *_, site, present = ["", "", *done["stdout"].strip().splitlines()]
            try:
                site = ast.literal_eval(site)  # the probe's ASCII literal, back to the exact path
            except (ValueError, SyntaxError):
                pass
            if present != "True" and site not in empty:
                empty.append(site)
    return empty


def _harnesses(workspace: pathlib.Path, commits: Dict[str, str]) -> Dict[str, str]:
    """Whether each role's tested tree reads a declared installation user site: ``opt_in`` when both ``_HARNESS``
    files name ``installation_user_site``, else ``predates_opt_in``, read from Git, never run; ``unknown`` when the
    read itself failed. One ``git grep`` through the Git owner before the budget; a begun stop raises typed."""
    from supervisor.update_prepare import _run, stopped

    try:
        done = _run(workspace, "grep", "-l", "-F", "-e", "installation_user_site", *sorted(set(commits.values())),
                    "--", *_HARNESS, check=False)
    except subprocess.TimeoutExpired:
        stopped()
        return {role: "unknown" for role in commits}
    if done.returncode not in (0, 1):  # 1: no tree names it
        stopped()
        return {role: "unknown" for role in commits}
    named = set(done.stdout.decode("utf-8", "surrogateescape").splitlines())
    return {role: "opt_in" if all(f"{commit}:{path}" in named for path in _HARNESS) else "predates_opt_in"
            for role, commit in commits.items()}


def _node(workspace: pathlib.Path, selection, env: Dict[str, str], root: pathlib.Path) -> Dict[str, str]:
    """The Node a full tier needs where the tree has browser-module tests (``web/tests``; Python tests drive them with
    Node): the one the gate selects (``preflight_node.resolve_node``: the bundled one, else PATH's), already first on
    the runs' PATH, must answer ``--version`` at the suite's floor (``NODE_MIN_VERSION``) through ``_probe``. ``{}``
    when none is needed (a selected tier, or no such tests); ``{"node", "version"}`` when it answers; otherwise
    ``_Unverified``: absent, unstartable, a timeout or unusable. Nothing is downloaded and no other Node is tried."""
    from ouroboros.preflight_node import NODE_MIN_VERSION, _version_tuple, candidate_node_tests, resolve_node

    files = candidate_node_tests(workspace) if selection.tier == "full" else []
    if not files:
        return {}
    node, subject = resolve_node(), f"Node.js for the full tier's {len(files)} browser-module test file(s)"
    if not node:
        raise _Unverified(f"{subject} is absent: none is bundled and none is on PATH", prerequisite="node",
                          status="absent")
    version = _probe([node, "--version"], root, env, f"{subject} ({node})", prerequisite="node",
                     node=node)["stdout"].strip().removeprefix("v")
    if _version_tuple(version) < NODE_MIN_VERSION:
        floor = ".".join(str(part) for part in NODE_MIN_VERSION)
        raise _Unverified(f"{subject} ({node}) is unusable: {version or 'no version'} is below the v{floor} floor",
                          prerequisite="node", status="unusable", node=node, version=version)
    return {"node": node, "version": version}


def _lanes(workers: int) -> List[Tuple[str, List[str]]]:
    """The commit gate's two lanes (``preflight_runner``): the non-serial cases over ``workers`` xdist workers, then
    the serial ones alone, each with CI's per-test hang bound."""
    from ouroboros.preflight_runner import LANE_EXCLUSION_EXPR, PARALLEL_PASS_FLAGS, SERIAL_PASS_FLAGS

    return [("parallel", ["-m", f"not serial and {LANE_EXCLUSION_EXPR}",
                          *(str(workers) if flag == "auto" else flag for flag in PARALLEL_PASS_FLAGS)]),
            ("serial", ["-m", f"serial and {LANE_EXCLUSION_EXPR}", *SERIAL_PASS_FLAGS])]


def run_tests(workspace: pathlib.Path, commit: str, targets: List[str], scratch: pathlib.Path,
              env: Dict[str, str] = None, python: str = "", record: Dict[str, Any] = None,
              deadline: Optional[float] = None, unproven: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run ``targets`` (files, node ids or the suite directory) on ``commit`` materialized in the workspace, in the
    gate's two lanes.

    ``deadline`` (``time.monotonic()``) bounds the materialization, under the Git owner's own bound as well, and
    every lane, each getting what remains: the checks' one budget, shared with the runs before and after this
    one; a run alone gets the whole budget (``_budget``). A materialization out of time leaves the run unstarted, its
    own output kept (``materialization`` in ``run.json``); a begun stop that ended or refused it raises its typed
    refusal, as does any other failure of it. ``complete`` only when every lane finished the collection
    it was given by then (exit 0 or 1, or 5 for a lane none of these cases belongs to), the lanes reported
    cases, no test worker crashed (``crashed``) and no lane exited 1 though none of its own cases failed (its
    session failed outside its tests, so no case names that failure: ``no_failing_case`` on that lane, in ``lanes``
    and ``run.json`` only then); anything else is unverified, never a pass,
    never "no failures" and never a failure a baseline could excuse.
    ``failed`` maps each failing node id to its failure signature (kind, exception type,
    first message line); ``lanes`` keeps each lane's exit, case count and seconds.
    ``evidence`` is the run's own folder beside its reports: ``run.json`` (commit, interpreter, user
    base and the one it declared as the installation's, PYTHONPATH, targets, ``record`` (what chose them),
    ``deadline_seconds`` (what remained of the budget
    when this run began) and each lane's exact argv, exit, cases, seconds and
    ``output``) and each lane's complete ``<lane>.stdout.txt`` and ``<lane>.stderr.txt``; ``output``
    names a lane cut at the deadline (``partial``, ``run_tracked``) or never started (``none``). Each lane runs
    contained as the gate's passes are (``run_tracked`` with its two files as ``outputs``): nothing it started outlives
    it, and a lane whose reap could not prove its tree gone names why (``containment``, in ``lanes`` and ``run.json``
    only then) and leaves the run incomplete; ``unproven`` (the caller's list) receives ``<lane>: <why>`` as soon as
    that reap names it, before the lane's output or ``run.json`` is written and before a stop is raised, so the
    caller's release (``_release``) keeps its root on it also when this run raises. A run left incomplete while a stop
    has begun (Emergency Stop or this server's stop, which end and refuse its lanes) raises that stop's typed refusal
    once ``run.json`` and every lane's output are written; a complete one stays its measured result. The lanes leave
    pytest's cache plugin on, as the gate's passes do: the tested harness keeps its cache in its own session boundary.
    The targets reach every lane through pytest's own argument file (``@targets.txt`` there, one per line, read
    again by each xdist worker): one line for the suite directory of a full tier, else every module or node id,
    as many as no command line takes on Windows (CreateProcess stops at 32767 characters); ``run.json`` names it
    (``targets_file``) beside the exact targets.
    """
    from ouroboros.test_environment import INSTALLATION_USER_BASE_ENV
    from ouroboros.test_parallelism import NESTED_WORKERS_ENV, nested_workers, worker_budget
    from ouroboros.utils import atomic_write_json, write_text_atomic
    from supervisor.update_prepare import _GIT_TIMEOUT_SEC, stopped

    if not targets:
        raise ValueError("no test targets: an empty selection would run the whole suite")
    started = time.monotonic()
    deadline = started + _budget() if deadline is None else deadline
    bound, unstarted, attempt = min(_GIT_TIMEOUT_SEC, deadline - started), "", ("not_started", None, b"", b"")
    if bound <= 0:
        unstarted = f"the deadline passed before {commit} was materialized"
    else:
        try:
            done = _materialize(workspace, commit, bound)
            attempt = ("complete", done.returncode, done.stdout, done.stderr)
        except subprocess.TimeoutExpired as exc:
            stopped()  # a stop that ended it is that stop, never this run's deadline
            attempt = ("timeout", None, exc.output or b"", exc.stderr or b"")
            unstarted = (f"materializing {commit} did not finish within {bound:.0f}s: "
                         + (exc.stderr or b"").decode("utf-8", "replace").strip())
        except Exception:
            stopped()  # likewise: a Git step the stop ended is the stop's typed refusal; any other failure as before
            raise
    scratch.mkdir(parents=True, exist_ok=True)
    folder = pathlib.Path(tempfile.mkdtemp(prefix="junit-", dir=str(scratch)))
    targets_file = folder / "targets.txt"
    write_text_atomic(targets_file, "".join(f"{target}\n" for target in targets))
    cases, lanes, tails = {}, [], []
    manifest = {"commit": commit, "python": python or sys.executable, "user_base": (env or {}).get("PYTHONUSERBASE"),
                "installation_user_base": (env or {}).get(INSTALLATION_USER_BASE_ENV),
                "pythonpath": (env or {}).get("PYTHONPATH"), "tmpdir": (env or {}).get("TMPDIR"),
                "targets": list(targets), "targets_file": str(targets_file),
                "deadline_seconds": round(deadline - started, 1),
                "materialization": _kept(folder, "materialize", ["git", "reset", "--hard", "--quiet", commit], bound,
                                         *attempt), **(record or {}), "lanes": []}
    # The worker budget the environment admitted to its xdist and nested runners, else this process's own.
    for label, flags in _lanes(nested_workers(env) if NESTED_WORKERS_ENV in (env or {})
                               else worker_budget(os.cpu_count() or 1)):
        report, begun = folder / f"{label}.xml", time.monotonic()
        cmd = [python or sys.executable, "-m", "pytest", f"@{targets_file}", "-q", "-o",
               "junit_family=xunit1", f"--junitxml={report}", *flags]
        streams = {stream: folder / f"{label}.{stream}.txt" for stream in ("stdout", "stderr")}
        run = (run_tracked(cmd, workspace, deadline - begun, env=env, outputs=(streams["stdout"], streams["stderr"]),
                           custody=None if unproven is None else (
                               lambda why, label=label: unproven.append(f"{label}: {why}")))
               if deadline > begun and not unstarted else
               {"returncode": 124, "stdout": "", "output": "none",
                "stderr": unstarted or "the deadline passed before this lane"})
        for stream, path in streams.items():  # the lane streamed there; kept as its record (the deadline's line too)
            write_text_atomic(path, run[stream])
        try:
            found = _cases(report)
        except (OSError, ElementTree.ParseError):
            found = {}
        cases.update(found)
        lanes.append({"lane": label, "returncode": run["returncode"], "cases": len(found),
                      "seconds": round(time.monotonic() - begun, 1)})
        if run.get("containment"):  # its tree not proven gone: never a complete lane
            lanes[-1]["containment"] = run["containment"]
            tails.append(f"[{label}] its process tree was not proven gone: {run['containment']}")
        if run["returncode"] == 1 and found and not any(found.values()):  # its session failed outside its tests
            lanes[-1]["no_failing_case"] = True
        manifest["lanes"].append({**lanes[-1], "argv": cmd, "output": run["output"]})
        if run["returncode"] not in (0, 5):
            tails.append(f"[{label}] " + (run["stdout"] or run["stderr"])[-2000:])
    failed = {nodeid: signature for nodeid, signature in cases.items() if signature}
    crashed = sorted(nodeid for nodeid, signature in failed.items() if _WORKER_CRASH in signature)
    complete = not crashed and bool(cases) and all((lane["returncode"] in (0, 1) and lane["cases"]
                                                    or lane["returncode"] == 5 and not lane["cases"])
                                                   and not lane.get("containment") and not lane.get("no_failing_case")
                                                   for lane in lanes)
    returncode = next((lane["returncode"] for lane in lanes if lane["returncode"] not in (0, 5)), 0)
    result = {"complete": complete, "ok": complete and returncode == 0 and not failed, "commit": commit,
              "failed": failed, "crashed": crashed, "cases": len(cases), "returncode": returncode, "lanes": lanes,
              "seconds": round(time.monotonic() - started, 1), "tail": "\n".join(tails)[-2000:],
              "evidence": str(folder)}
    atomic_write_json(folder / "run.json", {**manifest, **{key: result[key] for key in (
        "complete", "returncode", "cases", "failed", "crashed", "seconds")}})
    if not complete:
        stopped()  # a begun stop that ended or refused a lane is that stop, raised once the run's evidence is kept
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


def _incomplete(run: Dict[str, Any], what: str = "the selected run") -> str:
    lanes = ", ".join(f"{lane['lane']}: exit {lane['returncode']}, {lane['cases']} case(s)" for lane in run["lanes"])
    crashed = run.get("crashed") or []
    unproven = "; ".join(f"{lane['lane']}: {lane['containment']}" for lane in run["lanes"] if lane.get("containment"))
    outside = ", ".join(lane["lane"] for lane in run["lanes"] if lane.get("no_failing_case"))
    return (f"{what} did not complete ({lanes}"
            + (f"; a test worker crashed while running {', '.join(crashed[:3])}" if crashed else "")
            + (f"; its process tree was not proven gone ({unproven})" if unproven else "")
            + (f"; {outside} exited 1 though none of its cases failed" if outside else "")
            + "); 124 is its deadline, 125 a refused start, 2 an interrupted collection, 5 nothing collected")


def _target_failing(workspace: pathlib.Path, row: Dict[str, Any], failed: Dict[str, str], scratch: pathlib.Path,
                    env: Dict[str, str], python: str, deadline: float,
                    unproven: Optional[List[str]] = None) -> Dict[str, Any]:
    """Which blocking cases the official target fails the same way on this installation: attribution for the owner
    and the mind, never a waiver. Only those cases, only where T has their file, on the packages W' runs on, within
    what remains of the checks' budget. Which of their files T has is one batched read through the Git owner
    (``cat-file --batch-check``), its input and own output kept (``target_run.presence``). A read or a T run that
    did not complete attributes nothing and names why (``target_run.unavailable``, ``complete`` false); the blocking
    cases stand either way. A begun stop that refused or ended the read raises its typed refusal. T's lanes hand
    ``unproven`` what their reaps could not prove gone (``run_tests``), also when its run raises."""
    from ouroboros.utils import atomic_write_json, write_bytes_atomic
    from supervisor.update_prepare import _GIT_TIMEOUT_SEC, _run, stopped

    target, files = row["target"]["target_sha"], sorted({nodeid.split("::", 1)[0] for nodeid in failed})
    folder = pathlib.Path(tempfile.mkdtemp(prefix="target-files-", dir=str(scratch)))
    argv, names = ["git", "cat-file", "--batch-check=%(objecttype)"], "".join(f"{target}:{path}\n" for path in files)
    write_bytes_atomic(folder / "stdin.txt", names.encode("utf-8", "surrogateescape"))
    bound, unavailable, found = min(_GIT_TIMEOUT_SEC, deadline - time.monotonic()), "", set()
    attempt = ("not_started", None, b"", b"")
    if bound <= 0:
        unavailable = "the deadline passed before T's files were read"
    else:
        try:
            done = _run(workspace, *argv[1:], data=names.encode("utf-8", "surrogateescape"), check=False,
                        timeout=bound)
        except subprocess.TimeoutExpired as exc:
            stopped()  # a stop that ended it is that stop, never this read's deadline
            attempt, unavailable = ("timeout", None, exc.output or b"", exc.stderr or b""), (
                f"reading T's files did not finish within {bound:.0f}s")
        else:
            attempt = ("complete", done.returncode, done.stdout, done.stderr)
            answers = done.stdout.decode("utf-8", "surrogateescape").splitlines()
            if done.returncode != 0 or len(answers) != len(files):
                stopped()  # likewise for a read the stop ended
                unavailable = (f"reading T's files failed (exit {done.returncode}, {len(answers)} answer(s) "
                               f"for {len(files)} file(s))")
            else:
                found = {path for path, answer in zip(files, answers) if answer == "blob"}
    presence = {**_kept(folder, "cat-file", argv, bound, *attempt), "stdin": "stdin.txt", "files": files,
                "present": sorted(found), "unavailable": unavailable}
    atomic_write_json(folder / "presence.json", presence)
    if unavailable:
        return {"target_failing": [], "target_run": {"complete": False, "cases": 0, "unavailable": unavailable,
                                                     "presence": str(folder)}}
    present = sorted(nodeid for nodeid in failed if nodeid.split("::", 1)[0] in found)
    if not present:
        return {"target_failing": [], "target_run": {"cases": 0, "presence": str(folder)}}
    run = run_tests(workspace, target, present, scratch, env, python, deadline=deadline, unproven=unproven)
    return {"target_failing": sorted(nodeid for nodeid in present
                                     if run["complete"] and run["failed"].get(nodeid) == failed[nodeid]),
            "target_run": {**{key: run[key] for key in ("complete", "cases", "returncode", "seconds", "evidence")},
                           "presence": str(folder)}}


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
    a blocking case is attributed on T, never waived; an incomplete run is unverified, and so is a W comparison
    that did not complete, W''s measured failures still named. The required floor is T's own (``_target_floor``):
    one W' lacks fails the checks, named, and one T does not declare in the shape read here leaves them unverified.
    A missing prerequisite (``_node``; a declared installation user base that is no existing absolute directory,
    ``_declared``) leaves them unverified before any test runs (``prerequisite``). Where the installation keeps its
    packages in a user base, ``user_site`` records the bases declared, the site-packages of any that holds none
    (``no_site_packages``: the tests run, with nothing added) and whether each role's tested harness reads them
    (``_harnesses``); a W whose harness predates that opt-in excuses no shared failure (``baseline.excuses``,
    ``not_excused``, ``why``), whether or not its base holds packages.
    The checks' one budget (``budget_seconds``) begins just before W' is materialized and bounds every process of W',
    W and T alike; restoring W' after W and T, also when they raised, keeps the Git owner's own bound."""
    from ouroboros.preflight_runner import DEFAULT_PYTEST_ARGS
    from ouroboros.test_environment import INSTALLATION_USER_BASE_ENV
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
    scratch.mkdir(parents=True, exist_ok=True)
    root = _process_root()  # released when these runs end, whatever ends them: removed, or kept and named (_release)
    release: Dict[str, Any] = {}  # filled then; every record returned below holds this one dict
    # Lanes whose reap could not prove their tree gone, handed over by each run as they are seen (also by a run that
    # then raises): the release keeps the root on them.
    unproven: List[str] = []
    try:
        evidence = pathlib.Path(tempfile.mkdtemp(prefix="test-evidence-", dir=str(scratch)))  # durable, outside root
        result = {"tier": selection.tier, "file_count": len(selection.files), "changed": len(footprint), "floor": floor,
                  "process_root": str(root), "process_root_release": release, "test_evidence": str(evidence),
                  "budget_seconds": _budget(),
                  "local_tests": len(carried), "dropped_local_tests": dropped, **_uncovered(selection, footprint)}
        try:
            (python, env), (serving_python, serving_env) = _environments(row, root, evidence)
            result["node"] = _node(workspace, selection, env, root)
            declared = [(run, run_env) for run, run_env in ((python, env), (serving_python, serving_env))
                        if INSTALLATION_USER_BASE_ENV in run_env]
            if declared:  # where each run's tested tree can read the installation's packages, kept beside the verdict
                empty = _declared(declared, root)
                result["user_site"] = {"declared": env.get(INSTALLATION_USER_BASE_ENV),
                                       "serving": serving_env.get(INSTALLATION_USER_BASE_ENV),
                                       "harness": _harnesses(workspace, {"W'": composition, "W": previous,
                                                                         "T": row["target"]["target_sha"]}),
                                       **({"no_site_packages": empty} if empty else {})}
        except _Unverified as exc:  # nothing ran: never a test failure, never ready
            return {**result, "ok": False, "unverified": f"a prerequisite of the checks is missing: {exc}",
                    "prerequisite": exc.record, "failed": {}, "blocking": []}
        deadline = time.monotonic() + result["budget_seconds"]  # one budget from here: W', W and T, every lane
        # The full tier names the suite directory, as the gate's full run and CI do; a selection its exact modules.
        new = run_tests(workspace, composition, DEFAULT_PYTEST_ARGS[:1] if selection.tier == "full"
                        else list(selection.files), scratch, env, python, deadline=deadline, record={
                            "selection": selection.record(), "floor": floor, "footprint": sorted(footprint),
                            "local_tests": sorted(carried), "dropped_local_tests": dropped, "node": result["node"],
                            "user_site": result.get("user_site")}, unproven=unproven)
        result.update(new={key: new[key] for key in ("complete", "ok", "cases", "crashed", "returncode", "lanes",
                                                     "seconds", "tail", "evidence")}, failed=new["failed"])
        if not new["complete"]:
            return {**result, "ok": False, "unverified": _incomplete(new), "blocking": sorted(new["failed"])}
        if new["ok"]:
            return {**result, "ok": True, "blocking": [], "already_failing": []}
        try:
            old = run_tests(workspace, previous, sorted(new["failed"]), scratch, serving_env, serving_python,
                            deadline=deadline, unproven=unproven)
            shared = sorted(nodeid for nodeid, signature in new["failed"].items()
                            if old["complete"] and old["failed"].get(nodeid) == signature)
            # Through a harness that predates the opt-in, W's own children lacked the installation's packages: a
            # failure it shares proves nothing, so it stays blocking (measured, attributed through T, never waived).
            reads = (result.get("user_site") or {}).get("harness", {}).get("W", "opt_in")
            excuses = reads == "opt_in"
            already = shared if excuses else []
            blocking = sorted(set(new["failed"]) - set(already))
            if blocking:
                result.update(_target_failing(workspace, row, {nodeid: new["failed"][nodeid] for nodeid in blocking},
                                              scratch, env, python, deadline, unproven))
        except Exception:
            _restore(workspace, composition)
            raise
        _materialize(workspace, composition)  # outside the budget, under the Git owner's own bound
        if not old["complete"]:  # none excused, none known to fail only with the update
            result["unverified"] = _incomplete(old, "the comparison run on W, the installation serving now,")
        baseline = {**{key: old[key] for key in ("complete", "failed", "returncode", "evidence")}, "excuses": excuses}
        if not excuses:
            baseline.update(not_excused=shared, why=_UNEXCUSED.format(
                "predates" if reads == "predates_opt_in" else "could not be read for"))
        return {**result, "ok": not blocking, "blocking": blocking, "already_failing": already, "baseline": baseline}
    finally:
        release.update(_release(root, "; ".join(unproven)))


def workspace_checks(row: Dict[str, Any]) -> Dict[str, Any]:
    """The checks for ``update_prepare.prepare``: ``ok`` is the selected tests' verdict."""
    from ouroboros.body_candidate import scratch_root

    # The candidate's declared sibling scratch: retired with its checkout, never kept as unknown data.
    scratch = scratch_root(row["path"]) / "update-checks"
    tests = selected_tests(row, scratch)
    return {"ok": bool(tests.get("ok")), "tests": tests}
