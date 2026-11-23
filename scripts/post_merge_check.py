#!/usr/bin/env python3
"""Check what landed on `ouroboros`, find the merge that broke it and hand it over.

    python scripts/post_merge_check.py base [--evidence-dir DIR]
    python scripts/post_merge_check.py growth --evidence-dir DIR --base BASE --head HEAD
    python scripts/post_merge_check.py classify --evidence-dir DIR [--lint] [--size --base BASE]
    python scripts/post_merge_check.py attribute --evidence-dir DIR --base BASE --head HEAD
    python scripts/post_merge_check.py route --evidence-dir DIR --run-url URL [--os linux|windows]
    python scripts/post_merge_check.py summary --evidence-dir DIR [--os linux|windows]

`.github/workflows/post-merge.yml` runs these steps after pushes to `ouroboros` in the official
repository (owner decision 3A, 2026-10-21), on a Linux and a Windows leg (owner decision 3A,
2026-10-28). Nobody waits before a merge; merges that land while a check runs are checked
together by the next one. A test that fails is run once more: passing then makes it FLAKY
(reported, not handed over), anything else makes it FAILED. On the Linux leg two deterministic
gates are FAILED with no retry: each finding of the release's lint gate (`ruff check --select F`,
owner decision 2A, 2026-10-22) and each red test of its size check (`pytest -m size_ratchet`,
measured against the check's base; the window's book-growth mode comes from the `growth` step).
First-parent bisection needs comparable evidence for blame, not for RED. Historical probes omit
the protected path if not comparable and take only fresh, coherent Python exports. Routing comments
on culprit PRs at new or expanded filings, even if filing fails. It narrows or closes per-leg
`post-merge-red` issues as IDs clear and files returning reds anew. No block, revert or schedule; GitHub uses only
the job's GITHUB_TOKEN. The script imports only stdlib-based code (`base` runs on the bare `window`
interpreter); Git, Ruff and test runners run as subprocesses.

DIR is the evidence root of `scripts/run_tests.py --selected --keep-going --evidence-dir DIR`
(selection.json, parallel/, serial/: tests/ci_evidence.py's results.json and events.jsonl). Each
step adds its record beside them: base.json, growth.json, classification.json (sessions in retry/
and size/), attribution.json (probes/), routing.json (recheck/, recheck-size/). The window both legs
measure, and the hand-over records one attempt leaves the next, are scripts/post_merge_window.py's.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from functools import cache
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zlib

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # run as a script, only scripts/ is on sys.path
    sys.path.insert(0, str(REPO_ROOT))

from scripts.post_merge_routing import (  # noqa: E402
    BRANCH, LINT_PREFIX, NODE_LANE, SIZE_PREFIX, SYSTEMS, GitHub, _reason, route,
)
from scripts.post_merge_bisect import (  # noqa: E402
    MAX_PROBES, PROTECTED, bisect, comparable, fresh_pytest_export, grep_names, node_blocker, probe_files, protected_projection,
)
from scripts.post_merge_window import (  # noqa: E402
    carried_records, choose_window, gap_text, listed_runs, merge_records, prior_window, read_handoff, records,
)
from ouroboros.update_qualification import POST_MERGE_RUNS, actions_pages  # noqa: E402
from tests.ci_verdicts import expected_failures, final, platform_name, session_unavailable, valid  # noqa: E402

LINT_ARGS = ("check", "--select", "F", "--output-format", "json", "--exit-zero")  # the release lint gate
# What a module of the size check declares: a decorated test, or a `pytestmark` carrying the marker.
# A line that only mentions it (a fixture string, a comment) declares nothing. POSIX classes: git grep -E.
SIZE_DECLARATIONS = (r"^[[:space:]]*@pytest\.mark\.size_ratchet([^[:alnum:]_]|$)",
                     r"^[[:space:]]*pytestmark[[:space:]]*[:=].*pytest\.mark\.size_ratchet([^[:alnum:]_]|$)")
ZERO_SHA = "0" * 40
# Retry, probe and re-check sessions keep the per-test bound of both selected passes
# (ouroboros/preflight_runner.py SERIAL_PASS_FLAGS). On Linux SIGALRM fails only the hung test;
# on Windows the thread method ends the session, whose journal still names the test that was
# running. Either way a hang ends as a named failure instead of the job's timeout, and a module
# that does not import leaves the other ids running.
ROUTE_STEP_SECONDS, TEST_SECONDS = 600, 300  # post-merge.yml's route step limit; each session's per-test bound
SESSION_FLAGS = ("--continue-on-collection-errors", f"--timeout={TEST_SECONDS}")
# The recheck of the open issues starts no test after this (pytest-timeout checks between tests), so one begun at the
# edge may still run its whole bound; 75 s is this author's reserve from run 42588728248, not a protocol maximum.
RECHECK_SECONDS = ROUTE_STEP_SECONDS - TEST_SECONDS - 75
_USAGE_ERROR = 4


# --- git -------------------------------------------------------------------------------------

def _git(*args: str, cwd: Path | None = None, strip: bool = True) -> str:
    out = subprocess.run(["git", *args], cwd=cwd or REPO_ROOT, capture_output=True, encoding="utf-8",
                         errors="replace", check=True, timeout=300).stdout
    return out.strip() if strip else out


def is_ancestor(older: str, newer: str) -> bool:
    """A strict ancestor on the integration line, including across fast-forwards.

    False is an answer Git gave: `older` is not down the first-parent line that `newer` ends, also where that line
    ends before reaching it, or one of the two is a commit this clone does not hold, which no line of it joins. A
    Git that could not answer raises instead (a count that failed although both commits are held, a process that was
    killed or timed out): unknown is never "no". What unknown means is the caller's to decide:
    scripts/post_merge_window.py keeps what it read and confirms no range."""
    if older == newer or not re.fullmatch(r"[0-9a-f]{40}", older):
        return False
    try:
        count = int(_git("rev-list", "--first-parent", "--count", f"{older}..{newer}"))
    except subprocess.CalledProcessError:
        for sha in (older, newer):
            try:
                _git("cat-file", "-e", sha)
            except subprocess.CalledProcessError as answer:
                if answer.returncode != 1:
                    raise
                return False  # Git's own "no such object": the count failed because this clone does not hold it
        raise
    # The commit `count` steps down the first-parent line, or nothing where the line ends sooner: an answer either way.
    return _git("rev-list", "--first-parent", f"--skip={count}", "--max-count=1", newer) == older


def first_parent(sha: str) -> str:
    try:
        return _git("rev-parse", "--verify", f"{sha}^1")
    except subprocess.CalledProcessError:
        return sha  # a root commit: nothing before it to compare with


def first_parent_chain(base: str, head: str) -> list[str]:
    """[base, c1, ..., head]: the commits `ouroboros` stood at, oldest first."""
    return [base, *_git("rev-list", "--first-parent", "--reverse", f"{base}..{head}").split()]


def merges(base: str, head: str) -> list[tuple[str, str]]:
    """(sha, subject) of each first-parent commit in (base, head], newest first."""
    try:
        text = _git("log", "--first-parent", "--format=%H%x09%s", f"{base}..{head}")
    except (OSError, subprocess.SubprocessError):
        return []
    return [tuple(line.split("\t", 1)) for line in text.splitlines() if "\t" in line]


def tree_source(sha: str):
    """A reader of commit `sha`'s files: the text, or None where that tree has no such file."""
    def read(path: str) -> str | None:
        if not _git("ls-tree", "--name-only", sha, "--", path):
            return None
        return _git("show", f"{sha}:{path}", strip=False)
    return read


def working_source(path: str) -> str | None:
    try:
        return (REPO_ROOT / path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def _blob(sha: str, path: str) -> str:
    try:
        return _git("rev-parse", "--verify", "--quiet", f"{sha}:{path}")
    except subprocess.CalledProcessError:
        return ""


# --- evidence --------------------------------------------------------------------------------

def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _unfinished(began: Counter, ended: Counter) -> int:
    """How many starts a journal never saw end, counted per start: `running` names a public id once, so two raw tests
    that share one leave one entry there and two here."""
    return sum(max(0, count - ended[node]) for node, count in began.items())


def read_session(directory: Path, *, session_id: str | None = None) -> dict:
    """One pytest session as tests/ci_evidence.py recorded it in `directory`.

    `final`: it exported results.json; without that its incremental journal (events.jsonl) gives
    the reports. `running`: tests the journal saw start and never finish. `exit`: pytest's status.
    `github` and `system`: the commit, run and operating system its export names (a journal names none).
    `unmatched`: the `--ci-listed` ids its complete collection found no test for (tests/ci_evidence.py).
    `selected`: the tests it chose to run after every filter, None where it names none (an older producer, a journal).
    `unfinished_starts`: the starts it never saw end, counted per start (`running` names a public id once, and two raw
    tests can share one); a start is no failed report.
    """
    directory = Path(directory)
    running, journal_rows, journal, intact = {}, [], True, True
    began, ended = Counter(), Counter()  # starts and their ends per public id
    try:
        journal = not session_unavailable(directory)
        lines = (directory / "events.jsonl").read_bytes().splitlines() if journal else []
    except OSError:
        lines, journal = [], False
    if session_id and lines:
        try:
            header = json.loads(lines[0])
        except ValueError:
            header = None
        if header != {"event": "session", "session": session_id}:
            lines, journal = [], False
    for line in lines:
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError):
            intact = False
            continue  # a kill can tear the last line
        if isinstance(row, dict) and row.get("event") == "session":
            continue
        if (not isinstance(row, dict) or not isinstance(row.get("nodeid"), str)
                or row.get("event") not in ("start", "finish", "report", "lane")):
            intact = False
            continue
        if row.get("event") == "start":
            running[row["nodeid"]] = None
            began[row["nodeid"]] += 1
        elif row.get("event") == "finish" or (row.get("event") == "report" and row.get("phase") == "crash"):
            running.pop(row["nodeid"], None)  # a crashed worker never sends finish
            ended[row["nodeid"]] += 1
        if row.get("event") == "report":
            if valid(row):
                journal_rows.append(row)
            else:
                intact = False
    data, error = final(directory, session=session_id)
    if not error:
        status, environment = data.get("session_exit_code"), data.get("environment")
        selected = data.get("selected_nodeids")
        if status in (0, 1):  # an export that ended shows each test that finished, whether or not its journal row survived
            torn = Counter(row["nodeid"] for row in data["reports"] if row["phase"] == "teardown" and not row.get("is_subtest"))
            for node in torn:
                running.pop(node, None)
            ended = Counter({node: max(ended[node], torn[node]) for node in {*ended, *torn}})
        return {"final": True, "evidence": True, "rows": data["reports"], "unfinished_starts": _unfinished(began, ended),
                "collection": [row["nodeid"] for row in data.get("collection_failures") or []
                               if isinstance(row, dict) and isinstance(row.get("nodeid"), str)],
                "collection_skips": [node for node in data.get("collection_skips") or [] if isinstance(node, str)],
                "running": list(running), "exit": status if type(status) is int else None,
                "collected": data.get("tests_collected"),
                "selected": [node for node in selected if isinstance(node, str)] if isinstance(selected, list) else None,
                "github": data["github"] if isinstance(data.get("github"), dict) else {},
                "system": platform_name(environment.get("platform", "")) if isinstance(environment, dict) else "",
                "unmatched": [node for node in (data.get("listed") or {}).get("unmatched") or [] if isinstance(node, str)]
                if isinstance(data.get("listed"), dict) else []}
    return {"final": False, "evidence": journal, "rows": journal_rows, "collection": [], "collection_skips": [],
            "unfinished_starts": _unfinished(began, ended),
            "running": list(running), "exit": None, "collected": None, "selected": None, "github": {}, "system": "",
            "unmatched": [], "journal_intact": journal and intact}


def _failure_phases(session: dict) -> dict[str, dict[str, str]]:
    """Red observations by exact public ID and phase, including collectors and incomplete tests."""
    found = {}
    for row in session["rows"]:
        phase = row["phase"]
        if phase == "crash" or row["outcome"] == "failed":
            found.setdefault(row["nodeid"], {})[phase] = (
                "its worker crashed" if phase == "crash" else f"failed in {phase}")
    for nodeid in session["collection"]:
        found.setdefault(nodeid, {})["collection"] = "did not collect"
    for nodeid in session["running"]:
        found.setdefault(nodeid, {})["unfinished"] = "was running when the session ended"
    return found


def candidates(session: dict) -> dict[str, str]:
    """The first red reason for each ID, for readers that do not compare retry phases."""
    return {nodeid: next(iter(phases.values())) for nodeid, phases in _failure_phases(session).items()}


def _inside(node: str, nodeid: str) -> bool:
    """Whether `node` is `nodeid` or lies within it: a test of a module or class, a parametrization of a function."""
    return node == nodeid or node.startswith((nodeid + "::", nodeid + "["))


def _ran_whole(session: dict) -> tuple[set[str], set[str]]:
    """(whole, spoiled): the tests whose setup, call and teardown passed with nothing else, and those that did not."""
    phases, spoiled = {}, set(session["running"])
    for row in session["rows"]:
        if row["outcome"] != "passed" or row["phase"] == "crash":
            spoiled.add(row["nodeid"])
        elif not row.get("is_subtest"):
            phases.setdefault(row["nodeid"], set()).add(row["phase"])
    return {node for node, seen in phases.items() if seen >= {"setup", "call", "teardown"}} - spoiled, spoiled


def _shared(session: dict) -> set[str]:
    """Public IDs that name more than one test of `session`: two raw parameters can redact to the same ID."""
    setups = Counter(row["nodeid"] for row in session["rows"] if row["phase"] == "setup" and not row.get("is_subtest"))
    return {node for counts in (setups, Counter(session["selected"] or ())) for node, count in counts.items()
            if count > 1}


def _tells_tests_apart(session: dict) -> bool:
    """Whether a whole pass of `session` can be taken for one test's, given `_shared`. A pass that names its selection
    counts the tests behind each public ID. An xdist controller collects nothing itself and names none, as do an older
    producer and a journal: such a pass counts only where it reported every test it collected, whole and once
    (`_parent_calls`), so that no sibling of a public ID went unreported. A journal counted nothing and never does."""
    return session["selected"] is not None or _parent_calls(session) is not None


def outcomes(session: dict, ids, *, whole: bool = False) -> dict[str, str]:
    """Classify ids within the selected lane: failures/hangs fail, test/collector skips never pass, executed xfails stay
    xfailed (ci_verdicts). Aggregates need every selected member to run whole; excluded exact ids stay unknown.
    Without selection, exact calls and completed (exit 0/1) aggregate exports retain legacy semantics (the bisection
    reader, `check_ids`) unless `whole`: the recheck that retires a red needs setup, call and teardown of every member.
    Only check_ids/recheck_ids establish absence; a filtered member is not missing."""
    failed = set(session["running"]) | {row["nodeid"] for row in session["rows"]
                                        if row["outcome"] == "failed" or row["phase"] == "crash"}
    expected = set(expected_failures(session["rows"])) - failed
    skipped = {row["nodeid"] for row in session["rows"] if row["outcome"] == "skipped"} - failed - expected
    skipped |= set(session["collection_skips"])
    called = {row["nodeid"] for row in session["rows"] if row["phase"] == "call" and row["outcome"] == "passed"}
    seen, ended = failed | expected | skipped | called, session["final"] and session["exit"] in (0, 1)
    selected, done = session["selected"], (seen if session["selected"] is None and not whole
                                           else _ran_whole(session)[0] | expected)
    result = {}
    for nodeid in ids:
        members = {node for node in seen if _inside(node, nodeid)}
        if members & failed or any(_inside(node, nodeid) or _inside(nodeid, node) for node in session["collection"]):
            result[nodeid] = "failed"
        elif members & skipped:
            result[nodeid] = "skipped"
        else:
            chosen = ({node for node in selected if _inside(node, nodeid)} if selected is not None
                      else members if members == {nodeid} or ended else set())
            result[nodeid] = "unknown" if not chosen or chosen - done else "xfailed" if chosen & expected else "passed"
    return result


def container_args(ids, read_source) -> tuple[list[str], list[str]]:
    """Pytest arguments that run every id a tree still defines, and the ids whose module it lacks.

    pytest refuses a whole session when one argument names nothing ("not found", exit 4), and a
    test added by a later merge does not exist at an earlier commit. So each id becomes the deepest
    node its module still defines by `ast` of that tree (function without parameters, class, or
    the module), and the session's results are read for the exact id afterwards. Names end where
    the parameters begin: a `::` or `@` inside a parameter is part of it.
    """
    args, absent = set(), []
    for nodeid in ids:
        path, _, rest = nodeid.partition("::")
        names = rest.split("[", 1)[0].split("::") if rest else []
        source = read_source(path)
        if source is None:
            absent.append(nodeid)
        else:
            args.add("::".join([path, *_defined_prefix(source, names)]))
    return sorted(arg for arg in args if not any(arg.startswith(other + "::") for other in args)), absent


def _defined_prefix(source: str, names: list[str]) -> list[str]:
    try:
        body = ast.parse(source).body
    except (SyntaxError, ValueError):
        return []  # the module itself: its collection error is the session's answer
    found = []
    for index, raw in enumerate(names):
        name = raw.split("[", 1)[0]
        kinds = (ast.FunctionDef, ast.AsyncFunctionDef) if index == len(names) - 1 else (ast.ClassDef,)
        node = next((node for node in body if isinstance(node, kinds) and node.name == name), None)
        if node is None:
            break
        found.append(name)
        body = node.body
    return found


# --- running tests ---------------------------------------------------------------------------

def run_pytest(args, directory: Path, *, cwd: Path | None = None, env: dict | None = None) -> int:
    """One `scripts/run_tests.py` session of `args` in the checkout `cwd`, evidence in `directory`.

    The arguments travel in an @argsfile (pytest reads one per line); a non-ASCII one stays on the
    command line, because Python 3.10 opens that file in the locale's encoding. `env` is added to
    this process's environment for the session.
    """
    cwd, directory = Path(cwd or REPO_ROOT), Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    plain = [arg for arg in args if arg.isascii() and arg.isprintable()]
    argsfile = directory / "argsfile.txt"
    argsfile.write_text("".join(arg + "\n" for arg in plain), encoding="ascii")
    command = [sys.executable, str(cwd / "scripts" / "run_tests.py"), f"@{argsfile}",
               *[arg for arg in args if arg not in plain], "--ci-evidence-dir", str(directory), *SESSION_FLAGS]
    print(f"=== post-merge: {len(args)} pytest argument(s) in {cwd}", flush=True)
    try:
        return subprocess.call(command, cwd=str(cwd), env={**os.environ, **env} if env else None)
    except OSError as error:
        print(f"=== post-merge: the session did not start: {error}", flush=True)
        return 127


def run_node_lane(cwd: Path | None = None) -> int:
    cwd = Path(cwd or REPO_ROOT)
    try:
        return subprocess.call([sys.executable, str(cwd / "scripts" / "run_tests.py"), "--node"], cwd=str(cwd))
    except OSError:
        return 127


def _confirmed_probe_session(session: dict | None) -> bool:
    """Attribution needs normal, matching exits and the export's red observations must agree."""
    return bool(session and session["final"] and session["exit"] in (0, 1)
                and type(session.get("runner_exit")) is int and session["runner_exit"] == session["exit"]
                and bool(candidates(session)) == (session["exit"] == 1))


def check_ids(ids, directory: Path, *, read_source, run=run_pytest) -> dict[str, str]:
    """Historical outcomes: a missing module is absent; an unreadable one or an unconfirmed session is unknown."""
    unreadable = set()

    @cache
    def source(path):
        try:
            return read_source(path)
        except (OSError, subprocess.SubprocessError, ValueError):
            unreadable.add(path)
            return None  # Keep it out of argv without treating it as a removal.

    args, unrunnable = container_args(ids, source)
    result = {nodeid: "unknown" if nodeid.split("::", 1)[0] in unreadable else "absent" for nodeid in unrunnable}
    if args:
        session = fresh_pytest_export(directory, lambda: run(args, directory), read_session)
        pending = [nodeid for nodeid in ids if nodeid not in result]
        result.update(outcomes(session, pending) if _confirmed_probe_session(session) else dict.fromkeys(pending, "unknown"))
    return result


def recheck_ids(ids, directory: Path, *, head: str = "", read_source=working_source, run=run_pytest,
                budget=RECHECK_SECONDS) -> dict[str, str]:
    """`outcomes` of exactly `ids` at `head` after one session that collects their modules whole, runs only the tests they
    name (tests/ci_evidence.py `--ci-listed`; older trees: ``check_ids``) and starts none after `budget` s, from the module
    `head` picks (sorted, rotated by its CRC-32: a slow red front holds back no fixed tail): the rest stays unknown. An id
    is absent when its file is gone, or its module collected without an error yet held no test it names (a renamed test,
    a parameter that left); a deselected test, a skip at collection or a session without its final export proves none. A
    pass needs a session that names its selection or reported every test it collected (`_tells_tests_apart`): the journal
    of a killed one retires nothing, though its reds and skips stand."""
    present = [nodeid for nodeid in ids if read_source(nodeid.split("::", 1)[0]) is not None]
    result = {nodeid: "absent" for nodeid in ids if nodeid not in present}
    if present:
        (listed := Path(directory) / "listed.txt").parent.mkdir(parents=True, exist_ok=True)
        listed.write_text("".join(nodeid + "\n" for nodeid in present), encoding="utf-8")
        start = zlib.crc32(head.encode()) % len(modules := sorted({nodeid.split("::", 1)[0] for nodeid in present}))
        run([*modules[start:], *modules[:start], f"--ci-listed={listed}", f"--session-timeout={budget}"], directory)
        session = read_session(directory)
        shared = _shared(session)  # N raw tests behind one public ID pool their phases: no pass is known to be theirs
        apart = _tells_tests_apart(session)  # a journal, or an export short of a collected test, may hide a sibling of one ID
        result.update({nodeid: "unknown" if answer == "passed" and (not apart or any(_inside(node, nodeid) for node in shared))
                       else answer for nodeid, answer in outcomes(session, present, whole=True).items()})
        broken = {node.split("::", 1)[0] for node in session["collection"]}  # a module that did not collect whole
        result.update({nodeid: "absent" for nodeid in session["unmatched"]
                       if result.get(nodeid) == "unknown" and nodeid.split("::", 1)[0] not in broken})
    return result


# --- base ------------------------------------------------------------------------------------

# Small receipts fit in a job step; large ones travel by an exact artifact reference.
_MISSING = object()


def _compact_json(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _project_handoff(value, evidence, environ):
    if len(_compact_json(value)) <= 800:
        return value, False
    run, attempt = int(environ["GITHUB_RUN_ID"]), int(environ["GITHUB_RUN_ATTEMPT"])
    path = Path(evidence) / "handoff.json"
    _write_json(path, value)
    return {"artifact": {"run": run, "attempt": attempt, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}, True


def choose_base(runs, before: str, head: str, *, get=None, ancestor=None, parent=None, initial="", replay=None,
                handoff=None, own=0) -> tuple[str, str, bool]:
    """(base, reason, continuous): what scripts/post_merge_window.py's `choose_window` establishes on this checkout's
    first-parent line, else the push alone, which never seeds the next window. `replay` receives what the window
    carries in; `own` is the number of this check's own run."""
    ancestor = ancestor or is_ancestor
    base, why, continuous = choose_window(runs, head, get=get, ancestor=ancestor, initial=initial,
                                          replay={} if replay is None else replay, handoff=handoff, own=own)
    if base:
        return base, why, continuous
    try:
        follows = bool(before) and before != ZERO_SHA and before != head and ancestor(before, head)
    except Exception as error:  # noqa: BLE001 -- a comparison Git could not make names no tip; what the window read stays
        print(f"::warning title=Post-merge base::Git could not place the branch tip before this push ({_reason(error)})")
        follows = False
    if follows:
        return before, "the branch tip before this push (earlier coverage remains pending)", False
    return (parent or first_parent)(head), "the first parent of this push (earlier coverage remains pending)", False


def command_base(args, environ=os.environ) -> int:
    head = args.head or environ.get("GITHUB_SHA") or _git("rev-parse", "HEAD")
    before = environ.get("BEFORE", "")
    initial = environ.get("INITIAL_BASE", "")
    if initial and (environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
                    or environ.get("GITHUB_REPOSITORY") != "razzant/ouroboros"
                    or environ.get("GITHUB_REF") != f"refs/heads/{BRANCH}" or head != environ.get("GITHUB_SHA")
                    or not is_ancestor(initial, head)):
        print("::error::An initial base must be a full first-parent ancestor of the official branch dispatch.")
        return 1
    github = GitHub(environ.get("GITHUB_REPOSITORY", ""), environ.get("GH_TOKEN", ""))
    # Unfiltered workflow history includes dispatches and avoids GitHub's 1,000 filtered-run limit.
    runs = listed_runs(actions_pages(github.get, POST_MERGE_RUNS, "workflow_runs"))
    replay, own = {}, int(environ.get("GITHUB_RUN_ID") or 0)  # this check's own run is no history to it
    base, why, continuous = choose_base(runs, before, head, get=github.get, initial=initial, replay=replay, own=own,
                                       handoff=lambda *args: github.handoff_json(*args))
    print(f"post-merge base: {base} ({why})")
    for gap in replay["history_gaps"]:
        print(f"::warning title=Post-merge history gap::{gap_text(gap)}.")
    window = {"format": 1, "base": base, "head": head, "why": why, "continuous": continuous, "replay": replay}
    if args.evidence_dir:
        _write_json(Path(args.evidence_dir) / "base.json", window)
    if environ.get("GITHUB_OUTPUT"):
        checkpoint, stored = _project_handoff(replay, args.evidence_dir, environ)
        with open(environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
            output.write(f"base={base}\nwindow={_compact_json({**window, 'replay': checkpoint})}\n"
                         f"replay={_compact_json(checkpoint)}\nhandoff_file={'yes' if stored else 'no'}\n")
    return 0


# --- growth ----------------------------------------------------------------------------------

def approved_merges(shas, environ, decide) -> list[str]:
    """The merges whose pull request carries the owner's `book-growth` label.

    `decide(environ)` is scripts/book_growth_mode.py's decision for one push to `ouroboros`, so each
    merge of the window is judged exactly as its own push would have been; unreadable evidence
    approves nothing there either.
    """
    keep = {key: environ[key] for key in ("REPO", "OWNER", "GH_TOKEN") if environ.get(key)}
    return [sha for sha in shas if decide({**keep, "EVENT": "push", "REF": f"refs/heads/{BRANCH}", "SHA": sha}) == "approved"]


def _book_growth_mode(environ) -> str:
    from scripts import book_growth_mode

    return book_growth_mode.main(environ)


def command_growth(args, environ=os.environ, decide=_book_growth_mode) -> int:
    """The book-growth mode the size check measures this window with: approved when the owner approved
    one of its merges (a growth by another merge of the same window then passes too), else block."""
    shas = [sha for sha, _subject in merges(args.base, args.head)]
    approved = approved_merges(shas, environ, decide)
    mode = "approved" if approved else "block"
    _write_json(Path(args.evidence_dir) / "growth.json", {"format": 1, "base": args.base, "head": args.head,
                                                          "mode": mode, "merges": shas, "approved": approved})
    print(f"post-merge book growth: {mode} ({len(approved)} of {len(shas)} merge(s) approved by the owner)")
    return 0


# --- classify --------------------------------------------------------------------------------

def lint_findings(cwd: Path | None = None, paths=()) -> dict[str, str] | None:
    """{id: message} of the lint gate in `cwd` (over `paths`, else the whole tree); None if ruff could not run.

    The id names the code, the file and the message's first backticked name rather than a line,
    which moves with every edit above it.
    """
    root = Path(cwd or REPO_ROOT).resolve()
    try:
        done = subprocess.run([sys.executable, "-m", "ruff", *LINT_ARGS, *(paths or ["."])], cwd=root,
                              capture_output=True, text=True, encoding="utf-8", timeout=600)
        rows = json.loads(done.stdout or "null")
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if done.returncode != 0 or not isinstance(rows, list):
        return None
    found = {}
    for row in rows:
        path = Path(row.get("filename") or "").resolve()
        name = re.search(r"`([^`]+)`", row.get("message") or "")
        found[f"{LINT_PREFIX}{row.get('code')} {path.relative_to(root).as_posix() if path.is_relative_to(root) else path} "
              f"{name.group(1) if name else row.get('message')}"] = row.get("message") or ""
    return found


def size_modules(root: Path, *, rev: str = "", exclude=()) -> list[str] | None:
    """The test modules of the size check in the checkout `root` (or in `rev`'s tree when it is named).

    The same tests as the release's `pytest tests/ -m size_ratchet`, without collecting the suite, and
    no module that merely mentions the marker: its collection is no business of the size check.
    """
    args = ["grep", "-l", "-z", "-E", *(arg for pattern in SIZE_DECLARATIONS for arg in ("-e", pattern))]
    args += [rev, "--", "tests"] if rev else ["--", "tests"]
    args.extend(f":(exclude,literal){path}" for path in sorted(exclude))
    try:
        listed = _git(*args, cwd=REPO_ROOT if rev else Path(root))
    except (OSError, subprocess.SubprocessError) as error:
        return [] if getattr(error, "returncode", None) == 1 else None  # no match is an answer; another failure is not
    return None if (names := grep_names(rev, listed)) is None else sorted(
        p for p in names if p not in exclude and Path(p).name.startswith("test_") and p.endswith(".py"))


def size_session(directory: Path, *, base: str, mode: str = "block", cwd: Path | None = None,
                 run=run_pytest, modules=None) -> dict | None:
    """The size check's fresh pytest export and separate wrapper exit, or None without that export.

    `modules` names the lane to measure: None (the default) reads it from the checkout, an empty
    list measures nothing.
    """
    root = Path(cwd or REPO_ROOT)
    modules = size_modules(root) if modules is None else list(modules)
    if not modules:
        return None
    return fresh_pytest_export(directory, lambda: run(
        ["-m", "size_ratchet", *modules], directory, cwd=root,
        env={"OURO_SIZE_RATCHET_BASE_REF": base, "OURO_BOOK_GROWTH_MODE": mode}), read_session)


def size_status(session: dict | None, ids) -> dict[str, str]:
    """`outcomes` of size ids in one size session; every one unknown when it did not run to its end."""
    ended = (session and session["final"] and session["exit"] in (0, 1)
             and session.get("runner_exit", session["exit"]) == session["exit"])
    found = outcomes(session, [nodeid.removeprefix(SIZE_PREFIX) for nodeid in ids]) if ended else {}
    return {nodeid: found.get(nodeid.removeprefix(SIZE_PREFIX), "unknown") for nodeid in ids}


def _parent_calls(session: dict) -> int | None:
    """Count complete parent calls. A skipped or failed setup owes only a teardown.

    Compare the public reports with one another: two raw parameter IDs can redact to the same
    public ID. The inventory separately reconciles the raw collection and teardowns.
    """
    collected = session["collected"]
    if type(collected) is not int:
        return None
    rows = [row for row in session["rows"] if not row.get("is_subtest")]
    setups = Counter(row["nodeid"] for row in rows if row["phase"] == "setup")
    ready = Counter(row["nodeid"] for row in rows if row["phase"] == "setup" and row["outcome"] == "passed")
    calls = Counter(row["nodeid"] for row in rows if row["phase"] == "call")
    ends = Counter(row["nodeid"] for row in rows if row["phase"] == "teardown")
    if (setups.total() == collected and setups == ends and calls == ready and not session["running"]
            and (session["selected"] is None or setups == Counter(session["selected"]))
            and not any(row["phase"] == "crash" for row in rows)):
        return calls.total()
    return None


def size_findings(session: dict | None) -> tuple[dict[str, str] | None, list[str]]:
    """A finished pytest session's red size IDs and any gaps in the size measurement.

    A skip measured nothing. A wrapper disagreement leaves the measurement incomplete without
    erasing observations from the pytest export.
    """
    if not (session and session["final"] and session["exit"] in (0, 1)):
        return None, ["the size check could not run or finish, so it went unchecked"]
    skipped = set(session["collection_skips"]) | {row["nodeid"] for row in session["rows"] if row["outcome"] == "skipped"}
    problems = [f"the size check skipped `{nodeid}`, so it measured nothing there" for nodeid in sorted(skipped)]
    if (runner_exit := session.get("runner_exit", session["exit"])) != session["exit"]:
        problems.append(f"the size check runner exit {runner_exit} disagrees with pytest exit {session['exit']}")
    called = _parent_calls(session)
    if session["collection"] or called is None or (not skipped and (not called or called != session["collected"])):
        problems.append("the size check did not measure every collected test")
    found = {SIZE_PREFIX + nodeid: why for nodeid, why in candidates(session).items()}
    if session["exit"] == 1 and not found:
        problems.append("the size check exited red without a recorded failure")
    elif session["exit"] == 0 and found:
        problems.append("the size check reported a failure despite a zero exit")
    return found, problems


def _selected_pass(outcome, inventory, session, red) -> tuple[int, str]:
    if problem := inventory.get("problem"):
        return 0, f"inventory: {problem}"
    if outcome == "skipped":
        return 0, "did not finish" if session["evidence"] else "did not run"
    if outcome == "failure" and not red:
        return 0, "was red without a failing test (read its log)"
    expected, ended = inventory.get("expected"), inventory.get("executed")
    if not (type(inventory.get("collection_reports")) is int and inventory["collection_reports"] > 0
            and isinstance(expected, list) and isinstance(ended, list)):
        return 0, "has no completed collection inventory"
    if len(set(expected)) != len(expected) or Counter(expected) != Counter(ended):
        return 0, "inventory: expected and executed test ids differ (missing or duplicate execution)"
    status = inventory.get("exitcode")  # the selected CLI checked the child process against this status
    if (type(status) is not int or (session["final"] and status != session["exit"])
            or status not in ((0, 1) if expected else (0, 5))
            or outcome != ("failure" if status == 1 else "success") or bool(red) != (status == 1)
            or session["collection"] or "failed" in inventory.get("collectors", ())):
        return 0, "has incomplete or inconsistent pytest exit/collection evidence"
    if not expected and (session["collection_skips"] or "skipped" in inventory.get("collectors", ())):
        return 0, "collected no tests after a collector skipped"
    if session["final"]:
        calls = _parent_calls(session) if session["collected"] == len(expected) else None
    elif expected:  # its phase journal stays a diagnostic: a lost export leaves the pass unmeasured, never re-proven
        return 0, "has no valid results.json, so whether its tests ran whole is unmeasured (read its phase journal)"
    elif "collectors" not in inventory:
        return 0, "has no evidence that its empty collection finished without a collector skip"
    elif inventory.get("failed") or inventory.get("skipped") or session["evidence"] and (
            session["rows"] or not session["journal_intact"]):
        return 0, "has neither a valid results.json nor an intact report journal without reports"
    else:
        calls = 0  # The collector and process finished, and no test owed a phase report.
    if calls is None:
        return 0, "is missing selected test reports or its collected count disagrees with the inventory"
    if expected and not calls:
        if all(row["outcome"] == "skipped" for row in session["rows"] if row["phase"] == "setup" and not row.get("is_subtest")):
            return 0, f"skipped all {len(expected)} collected test(s) in setup before their call"
        return 0, "collected tests but none reached the call phase"
    return calls, ""


# --- what a selected pass reported -----------------------------------------------------------

COUNT_PHASES = ("setup", "call", "teardown")
_NORMAL_EXITS = (0, 1, 5)  # pytest's own: all passed, some failed, and the nothing-collected of an empty lane


def inventory_journal(path: Path) -> dict | None:
    """What a pass's private inventory journal (inventory-<lane>.jsonl, which the inventory plugin of
    ouroboros/preflight_inventory.py appends to, one JSON object per line) holds, as numbers: the session it names and its
    collector failures and skips. It outlives a public journal that could not open and an export that was lost. A torn
    tail costs only its own line (`complete` is then false). None: no journal."""
    try:
        lines = Path(path).read_bytes().split(b"\n")
    except OSError:
        return None
    rows, torn = [], False
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError):
            torn = True
            continue
        if isinstance(row, dict) and isinstance(row.get("event"), str):
            rows.append(row)
        else:
            torn = True
    sessions = [row.get("session") for row in rows if row["event"] == "evidence_session"]
    outcomes = [row.get("outcome") for row in rows if row["event"] == "collector"]
    code = rows[-1].get("exitcode") if rows else None
    ended = bool(rows) and rows[-1]["event"] == "finish" and type(code) is int and code in _NORMAL_EXITS
    return {"session": sessions[0] if len(sessions) == 1 and isinstance(sessions[0], str) and sessions[0] else None,
            "failed": outcomes.count("failed"), "skipped": outcomes.count("skipped"),
            "complete": ended and not torn and set(outcomes) <= {"failed", "skipped"}}


def pass_counts(evidence: Path, label: str, selection: dict, *, identity: dict, system: str, whole: bool = False) -> dict:
    """What one selected pass reported, as numbers and source labels only (never an id, a path or a text).

    Reports: failed setup, call and teardown reports counted per report (two raw tests that share a public id are two),
    failed subtests and reported worker crashes; `unfinished_starts` counts starts never seen to end and is no failure.
    They come from the final export when it names this run and system, else from the public journal. Collectors: from
    that export, else from the pass's private inventory journal, which outlives a public journal that could not open and
    a lost export. One source per kind, never a sum over mirrors. Only an export is `complete`, and only for a pass that
    validated (`whole`), ended with pytest's own 0 or 1 (5 for an empty lane) and left no unfinished start: an interrupted
    or abnormally released pass keeps its positives and asserts no zero. None is unknown, never a zero; `any_red` is True
    on a known positive, False only when both kinds are complete, else None. A source binds to this run, attempt and
    system through the export's own identity or through `selection`, the record run_tests.py wrote with them beside the
    inventory; one that names another session or cannot be bound is unknown."""
    unknown = {"reports": {"source": "none", "complete": False, "failed": None, "failed_subtests": None, "crashed": None,
                           "unfinished_starts": None},
               "collectors": {"source": "none", "complete": False, "failed": None, "skipped": None}, "any_red": None}
    try:
        evidence = Path(evidence)
        known = bool(identity) and all(isinstance(value, str) and value for value in identity.values())
        selection = selection if isinstance(selection, dict) else {}
        recorded = (selection.get("inventory") or {}).get(label)
        claimed = recorded.get("session") if isinstance(recorded, dict) else None
        journal = inventory_journal(evidence / f"inventory-{label}.jsonl")
        heard = journal["session"] if journal else None
        sid = heard or claimed
        if not (isinstance(sid, str) and sid) or (heard and claimed and heard != claimed):
            return unknown
        session = read_session(evidence / label, session_id=sid)
        try:
            with (evidence / label / "events.jsonl").open("rb") as stream:
                header = json.loads(stream.readline(128))
        except (OSError, ValueError):
            header = None
        tied = bool(session["evidence"]) and header == {"event": "session", "session": sid}
        written_here = known and selection.get("github") == identity and selection.get("system") == system
        exported = bool(known and session["final"] and session["github"] == identity and session["system"] == system)
        rows = session["rows"]
        failed = [row for row in rows if row["outcome"] == "failed"]
        by_phase = Counter(row["phase"] for row in failed if not row.get("is_subtest"))
        normal = session["exit"] in (0, 1) or (session["exit"] == 5 and not rows)
        if exported or (tied and written_here and not session["final"]):
            unfinished = session["unfinished_starts"] if tied else None
            reports = {"source": "export" if exported else "journal",
                       "complete": bool(exported and whole and normal and unfinished == 0),
                       "failed": {phase: by_phase[phase] for phase in COUNT_PHASES},
                       "failed_subtests": sum(1 for row in failed if row.get("is_subtest")),
                       "crashed": sum(1 for row in rows if row["phase"] == "crash"),
                       "unfinished_starts": unfinished}
        else:
            reports = unknown["reports"]
        if exported:
            collectors = {"source": "export", "complete": bool(whole and normal),
                          "failed": len(session["collection"]), "skipped": len(session["collection_skips"])}
        elif written_here and journal and journal["session"] == sid:
            collectors = {"source": "inventory", "complete": journal["complete"],
                          "failed": journal["failed"], "skipped": journal["skipped"]}
        else:
            collectors = unknown["collectors"]
        positive = ((sum(reports["failed"].values()) + reports["failed_subtests"] + reports["crashed"])
                    if reports["failed"] is not None else 0) + (collectors["failed"] or 0)
        return {"reports": reports, "collectors": collectors,
                "any_red": True if positive else False if reports["complete"] and collectors["complete"] else None}
    except Exception:  # noqa: BLE001 -- unknown, never a zero: no text of the failure reaches the evidence
        return unknown


def valid_pass_counts(seen) -> bool:
    """Whether `seen` is exactly what `pass_counts` writes: per kind a source label, a complete flag and counts that are
    all None (source "none": unknown) or whole numbers; a complete report count has no unfinished start and only an
    export is complete there; `any_red` is True on a known positive, False only for a complete zero of both kinds, else
    None."""
    if not (isinstance(seen, dict) and set(seen) == {"reports", "collectors", "any_red"}):
        return False
    reports, collectors = seen["reports"], seen["collectors"]
    if not (isinstance(reports, dict) and set(reports) == {"source", "complete", "failed", "failed_subtests", "crashed", "unfinished_starts"}
            and isinstance(collectors, dict) and set(collectors) == {"source", "complete", "failed", "skipped"}):
        return False
    count = lambda value: type(value) is int and value >= 0
    failed = reports["failed"]
    if reports["source"] == "none":
        reports_ok = (reports["complete"] is False and failed is None
                      and all(reports[key] is None for key in ("failed_subtests", "crashed", "unfinished_starts")))
    else:
        reports_ok = (reports["source"] in ("export", "journal") and type(reports["complete"]) is bool
                      and (reports["source"] == "export" or reports["complete"] is False)
                      and (not reports["complete"] or reports["unfinished_starts"] == 0)
                      and isinstance(failed, dict) and set(failed) == set(COUNT_PHASES) and all(count(value) for value in failed.values())
                      and count(reports["failed_subtests"]) and count(reports["crashed"])
                      and (reports["unfinished_starts"] is None or count(reports["unfinished_starts"])))
    if collectors["source"] == "none":
        collectors_ok = collectors["complete"] is False and collectors["failed"] is None and collectors["skipped"] is None
    else:
        collectors_ok = (collectors["source"] in ("export", "inventory") and type(collectors["complete"]) is bool
                         and count(collectors["failed"]) and count(collectors["skipped"]))
    if not (reports_ok and collectors_ok):
        return False
    positive = ((sum(failed.values()) + reports["failed_subtests"] + reports["crashed"]) if failed is not None else 0) + (collectors["failed"] or 0)
    return seen["any_red"] is (True if positive else False if reports["complete"] and collectors["complete"] else None)


def classify(evidence: Path, *, run=run_pytest, node=run_node_lane, lint=None, size=None,
             read_source=working_source, checkpoint=lambda record: None) -> dict:
    """FAILED and FLAKY of the selected run after exactly one more execution of each red.

    `lint()` and `size()` are the deterministic gates of this leg (None: not on this leg), whose
    findings are FAILED without a retry; `size()` answers one size session (`size_session`).
    `checkpoint(record)` receives the reds before their retry starts, every one still FAILED with
    retry "not run": what a step killed during the retry leaves for the hand-over and the summary.
    """
    evidence = Path(evidence)
    selection = _read_json(evidence / "selection.json")
    record = {"format": 1, "complete": True, "failed": [], "flaky": [], "problems": [], "not_run": {},
              "selection": {}, "selection_executed": False}
    if not isinstance(selection, dict):
        record["problems"].append("the selected run left no selection.json (its selection failed or it died early)")
        return record
    record["base"] = selection.get("base")
    record["selection"] = {key: selection.get(key) for key in ("tier", "file_count", "available", "node", "outcome", "outcomes",
                                                               "deferred_full")}
    outcomes_by_lane = selection["outcomes"] if isinstance(selection.get("outcomes"), dict) else {}
    inventories = selection["inventory"] if isinstance(selection.get("inventory"), dict) else {}
    found = {}  # id -> (pass, why)
    original_phases, shared = {}, set()
    calls = record["selection"]["calls"] = {}  # a zero here is a confirmed empty pass
    for label in ("parallel", "serial"):
        outcome = outcomes_by_lane.get(label)
        inventory = inventories.get(label)
        if not isinstance(inventory, dict):
            inventory = {}
        session = read_session(evidence / label, session_id=inventory.get("session"))
        red = _failure_phases(session)
        shared |= _shared(session)
        for nodeid, phases in red.items():
            original_phases.setdefault(nodeid, set()).update(phases)
            found.setdefault(nodeid, (label, ", ".join(phases.values())))
        expected, executed = inventory.get("expected", []), inventory.get("executed", [])
        if all(isinstance(ids, list) and all(isinstance(item, str) for item in ids) for ids in (expected, executed)):
            if missing := set(expected) - set(executed) - set(red):
                record["not_run"][label] = sorted(missing)
        else:
            inventory = {"problem": "invalid test ids"}
        count, problem = _selected_pass(outcome, inventory, session, red)
        if problem:
            record["problems"].append(f"the {label} pass {problem}")
        else:
            calls[label] = count
        if not session["final"] and outcome in ("success", "failure"):
            record.setdefault("diagnostics_incomplete", []).append(label)
    if problem := selection.get("node_problem"):
        record["problems"].append(problem)
    elif selection.get("node") == "failure":
        found[NODE_LANE] = ("node", "the Node lane was red")
    if isinstance(problem := selection.get("selection_problem"), str) and problem:
        # A deferral whose scan of the other changes failed (or escalated unnamed) ran only the floor, the Node lane and the
        # author's extras: never a measured window (`selection_executed` is what `_window_measured` reads).
        record["problems"].append(problem[:300])
    if problem := selection.get("floor_problem"):
        record["problems"].append(problem)
    elif selection.get("outcome") not in ("success", "failure") or selection.get("node") not in ("success", "failure"):
        record["problems"].append("the selected runner left no terminal outcome for its battery or Node lane")
    elif (selection["outcome"] == "success") != (selection["node"] == "success" and
                                                 all(value == "success" for value in outcomes_by_lane.values())):
        record["problems"].append("the battery outcome disagrees with its passes (read the inventory/floor diagnostics)")
    if not record["problems"] and not any(calls.values()):
        record["problems"].append("both selected passes collected zero tests")
    record["selection_executed"] = not record["problems"] and not record["not_run"]
    findings = lint() if lint is not None else {}
    record["lint_checked"] = lint is not None and findings is not None
    if findings is None:
        record["problems"].append("ruff could not run, so the lint gate went unchecked")
    for finding, message in (findings or {}).items():
        found[finding] = ("lint", message)
    if size is not None:
        sized, problems = size_findings(size())
        record["size_checked"] = sized is not None and not problems
        record["problems"] += problems
        found.update({nodeid: ("size", why) for nodeid, why in (sized or {}).items()})
    checkpoint({**record, "complete": False, "flaky": [], "failed": [
        {"id": nodeid, "pass": label, "why": why, "retry": "not run"} for nodeid, (label, why) in sorted(found.items())]})
    ids = sorted(nodeid for nodeid in found if nodeid != NODE_LANE and not nodeid.startswith((LINT_PREFIX, SIZE_PREFIX)))
    status = {}
    if ids:
        directory = evidence / "retry"
        code = run(ids, directory)
        session = read_session(directory)
        observed = retry_phases = _failure_phases(session)
        if code == session["exit"] == _USAGE_ERROR and not session["rows"]:
            # One id pytest cannot find (a redacted parameter) refuses the whole session.
            args, absent = container_args(ids, read_source)
            status.update(dict.fromkeys(absent, "absent"))
            if args:
                directory = directory / "nodes"
                code = run(args, directory)
                session = read_session(directory)
                retry_phases = _failure_phases(session)
                for nodeid, phases in retry_phases.items():
                    observed.setdefault(nodeid, {}).update(phases)
        if pending := [nodeid for nodeid in ids if nodeid not in status]:
            if (session["final"] and session["exit"] == code and code in (0, 1)
                    and _parent_calls(session) is not None and bool(retry_phases) == (code == 1)):
                shared |= _shared(session)
                for nodeid, answer in outcomes(session, pending).items():
                    if nodeid in shared and answer in ("failed", "passed"):
                        # Two raw parameters can redact to one public ID: no answer is known to be this test's.
                        label, why = found[nodeid]
                        found[nodeid], answer = (label, why + "; its public id names several tests"), "unknown"
                    # An aggregate red can be a different test, a different phase, or a collector
                    # that prevented the original test from running. None confirms an exact repeat.
                    status[nodeid] = ("unknown" if answer == "failed"
                                      and not original_phases[nodeid].intersection(retry_phases.get(nodeid, ()))
                                      else answer)
            else:
                record["complete"] = False
                record["problems"].append("the retry has no complete test reports and matching terminal exit")
                status.update(dict.fromkeys(pending, "unknown"))
        for nodeid, phases in observed.items():
            if nodeid not in found:
                found[nodeid] = ("retry", ", ".join(phases.values()))
                status[nodeid] = "first seen on retry"
            elif new := [why for phase, why in phases.items() if phase not in original_phases[nodeid]]:
                label, why = found[nodeid]
                found[nodeid] = (label, why + "; on retry: " + ", ".join(new))
    if NODE_LANE in found:
        code = node()
        status[NODE_LANE] = {0: "passed", 1: "failed"}.get(code, "unknown")
        if status[NODE_LANE] == "unknown":
            record["complete"] = False
            record["problems"].append(f"the Node retry did not run to completion (runner exit {code})")
    status.update({nodeid: "failed" for nodeid in found if nodeid.startswith((LINT_PREFIX, SIZE_PREFIX))})  # deterministic
    for nodeid in sorted(found):
        label, why = found[nodeid]
        row = {"id": nodeid, "pass": label, "why": why, "retry": status.get(nodeid, "unknown")}
        record["flaky" if row["retry"] == "passed" else "failed"].append(row)  # only a pass is a pass
    return record


def command_classify(args) -> int:
    evidence = Path(args.evidence_dir)
    head = args.head or os.environ.get("GITHUB_SHA") or _git("rev-parse", "HEAD")
    output = evidence / "classification.json"
    extra = {"head": head, "gates": [gate for gate in ("lint", "size") if getattr(args, gate)]}
    size = None
    if args.size:
        growth = _read_json(evidence / "growth.json") or {}
        if not isinstance(growth, dict) or growth.get("base") != args.base or growth.get("head") != head:
            growth = {}  # Only approvals for the measured window can allow growth.
        extra["size"] = {"base": args.base, "mode": growth.get("mode") or "block", "decided": bool(growth),
                         "approved": growth.get("approved") or []}
        size = lambda: size_session(evidence / "size", base=args.base, mode=extra["size"]["mode"])  # noqa: E731

    def save(result):
        record = {**result, **extra, "problems": list(result["problems"])}
        if args.base and record["selection"] and record.get("base") != args.base:
            record["problems"].append("the selected runner used a different base from the shared window")
            record["selection_executed"] = False
        if (args.selected_outcome is not None and record["selection"]
                and args.selected_outcome != record["selection"].get("outcome")):
            record["problems"].append(f"the selected workflow step was {args.selected_outcome} and disagrees with its runner")
            record["selection_executed"] = False
        _write_json(output, record)
        return record

    record = save(classify(evidence, lint=lint_findings if args.lint else None, size=size, checkpoint=save))
    print(f"post-merge: {len(record['failed'])} failed, {len(record['flaky'])} flaky after one retry"
          + "".join(f"; {problem}" for problem in record["problems"]))
    return 0


# --- attribute -------------------------------------------------------------------------------

class Prober:
    """Detached probes use the revision's runner (CI installs dependencies, not the project).

    All lanes use the same projection; size measures allowed modules against the check's base.
    """

    def __init__(self, evidence: Path, work: Path, head: str, *, run=None, node=None, lint=None, size=None):
        self.evidence, self.work = Path(evidence), Path(work)
        self.run, self.node, self.lint = run or run_pytest, node or run_node_lane, lint or lint_findings
        self.size = size or (lambda tree, directory, modules: None)
        self.head, self.lock, self.records = head, _blob(head, "uv.lock"), []

    def __call__(self, sha: str, ids) -> dict[str, str | tuple[str, str]]:
        label = f"{len(self.records) + 1:02d}-{sha[:12]}"
        record = {"sha": sha, "ids": sorted(ids), "dependencies_differ": _blob(sha, "uv.lock") != self.lock}
        self.records.append(record)
        # What this revision may answer, decided before the worktree is built.
        linted = [nodeid for nodeid in ids if nodeid.startswith(LINT_PREFIX)]
        pairs = probe_files((nodeid for nodeid in ids if nodeid != NODE_LANE and not nodeid.startswith(LINT_PREFIX)),
                            SIZE_PREFIX)
        protection = protected_projection(self.head, sha, _git)
        state, reason = protection
        kept, excluded, family = comparable(pairs, revision=sha, protection=protection, git=_git)
        node = NODE_LANE in ids
        if node and reason and (problem := node_blocker(_git, sha, reason, family)):
            excluded[NODE_LANE] = problem
            node = False
        sized = [nodeid for nodeid in kept if nodeid.startswith(SIZE_PREFIX)]
        tests = [nodeid for nodeid in kept if not nodeid.startswith(SIZE_PREFIX)]
        modules = size_modules(REPO_ROOT, rev=sha, exclude=family) if sized else []
        if excluded:
            record["incomparable"] = excluded
        if not (kept or linted or node):
            record["answers"] = dict.fromkeys(ids, "error")
            return {nodeid: ("error", excluded.get(nodeid, "this revision could not answer it")) for nodeid in ids}
        tree = self.work / f"pmc-{label}"
        partial = ("--no-checkout",) if reason else ()
        added = False
        try:
            _git("worktree", "add", "--detach", "--quiet", *partial, str(tree), sha)
            added = True
            if partial:
                _git("read-tree", sha, cwd=tree)
                _git("restore", f"--source={sha}", "--worktree", "--", ".", f":(exclude,literal){PROTECTED}", cwd=tree)
                if omitted := _git("ls-files", "--", PROTECTED, cwd=tree):
                    if omitted != PROTECTED:
                        raise OSError(f"`{PROTECTED}` is not a single index entry")
                    _git("update-index", "--skip-worktree", "--", PROTECTED, cwd=tree)
                try:
                    (tree / PROTECTED).lstat()
                except FileNotFoundError:
                    pass
                else:
                    raise OSError(f"`{PROTECTED}` exists in the projected worktree")
        except (OSError, subprocess.SubprocessError) as error:
            if added:
                remove_worktree(tree)
            record["error"] = f"the worktree failed: {error}"
            return dict.fromkeys(ids, "error")
        try:
            answers = lint_answers(tree, linted, self.lint, omitted=PROTECTED if reason and state != "absent" else "")
            session = self.size(tree, self.evidence / "probes" / f"{label}-size", modules) if modules else None
            found = size_status(session if _confirmed_probe_session(session) else None, sized)
            if tests:
                found.update(check_ids(tests, self.evidence / "probes" / label, read_source=tree_source(sha),
                                       run=lambda args, directory: self.run(args, directory, cwd=tree)))
            answers.update({nodeid: {"failed": "fail", "passed": "pass", "absent": "pass"}.get(value, "error")
                            for nodeid, value in found.items()})
            if node:
                answers[NODE_LANE] = {0: "pass", 1: "fail"}.get(self.node(tree), "error")
            answers.update({nodeid: ("error", why) for nodeid, why in excluded.items()})
            record["answers"] = {nodeid: answer[0] if isinstance(answer, tuple) else answer
                                 for nodeid, answer in answers.items()}
            return answers
        finally:
            remove_worktree(tree)


def lint_answers(tree: Path, findings, lint, *, omitted: str = "") -> dict[str, str]:
    """Ruff over the revision's files. True absence is a pass; a deliberately omitted file is not evidence."""
    files = {finding: finding.split(" ", 3)[2] for finding in findings}
    paths = sorted({path for path in files.values() if path != omitted and (Path(tree) / path).is_file()})
    present = lint(tree, paths) if paths else {}
    return {finding: ("error" if path == omitted or (present is None and path in paths)
                      else "fail" if present and finding in present else "pass") for finding, path in files.items()}


def remove_worktree(tree: Path) -> None:
    """Remove the one worktree this check added at `tree`, whatever its tests left inside."""
    try:
        _git("worktree", "remove", "--force", str(tree))
    except (OSError, subprocess.SubprocessError):
        shutil.rmtree(tree, ignore_errors=True)
        try:
            _git("worktree", "prune")
        except (OSError, subprocess.SubprocessError):
            pass


def command_attribute(args) -> int:
    evidence = Path(args.evidence_dir)
    classification = _read_json(evidence / "classification.json") or {}
    failed = classification.get("failed", [])
    ids = [row["id"] for row in failed if row.get("retry") == "failed"]
    unresolved = [{"id": row["id"], "verdict": "unattributed", "why": f"retry: {row.get('retry', 'unknown')}"}
                  for row in failed if row.get("retry") != "failed"]
    sized = classification.get("size") or {}  # each probe measures as the check did: same base, same mode
    record = {"format": 1, "base": args.base, "head": args.head, "max_probes": args.max_probes, "notes": [],
              "probes": [], "verdicts": unresolved + [{"id": nodeid, "verdict": "unattributed",
                                          "why": "the attribution did not finish"} for nodeid in ids]}
    _write_json(evidence / "attribution.json", record)  # what a killed step leaves behind
    if not ids:
        return 0
    chain = first_parent_chain(args.base, args.head)
    record["commits"] = chain[1:]
    work = Path(tempfile.mkdtemp(prefix="pmc-", dir=args.work_dir or os.environ.get("RUNNER_TEMP") or None))
    prober = Prober(evidence, work, args.head, size=lambda tree, directory, modules: size_session(
        directory, base=sized.get("base") or args.base, mode=sized.get("mode") or "block", cwd=tree,
        modules=modules))
    try:
        record["verdicts"] = unresolved + bisect(chain, ids, prober, max_probes=args.max_probes)
    finally:
        record["probes"] = prober.records
        shutil.rmtree(work, ignore_errors=True)
        if differ := [probe["sha"][:12] for probe in prober.records if probe["dependencies_differ"]]:
            record["notes"].append(f"dependencies differ (uv.lock) at {', '.join(differ)}: those probes ran the "
                                   "head's environment, so their attribution is approximate")
        _write_json(evidence / "attribution.json", record)
    for verdict in record["verdicts"]:
        print(f"post-merge: {verdict['id']}: {verdict['verdict']} {verdict.get('culprit', '')}".rstrip())
    return 0


# --- route -----------------------------------------------------------------------------------

def same_run_verdicts(evidence: Path, ids, *, head: str, system: str, run_id: str = "",
                      recheck_unselected: bool = False) -> dict[str, str]:
    """What this run's selected passes settle of `ids`, by their final exports at `head` on `system` in this run: "passed"
    for a test they ran whole (setup, call and teardown reported, every report passed; a module, class or function id
    proves nothing, nor does an id that several tests of a pass share, `_shared`, nor one of a pass that cannot tell its
    tests apart, `_tells_tests_apart`), "kept" for an id that is owed no recheck: its own report was no whole pass or came
    from such a pass, or they ran nothing of its module, whose file exists. The recheck gets
    the rest: an id without a report in a module they ran (a removed or renamed test, one its pass never reached, a
    module or class id) and a test whose file is gone."""
    selection = _read_json(Path(evidence) / "selection.json")
    wanted, passed, spoiled, reported = set(ids), set(), set(), set()
    for label in (selection.get("outcomes") or {}) if isinstance(selection, dict) else ():
        inventory = (selection.get("inventory") or {}).get(label) or {}
        session = read_session(Path(evidence) / label, session_id=inventory.get("session"))
        if not (session["final"] and head and session["github"].get("sha") == head and session["system"] == system
                and (not run_id or session["github"].get("run_id") == run_id)):
            continue
        whole, bad = _ran_whole(session)
        broken = {node.split("::", 1)[0] for node in session["collection"]}
        # A pass that cannot tell its tests apart (an xdist export that left a collected test unreported) clears nothing:
        # what it reported stays owed (kept below); the recheck still gets the ids it never reported.
        passed |= whole if _tells_tests_apart(session) else set()
        # An id that several tests of this pass share pools their phases: no whole pass is known to be one test's.
        spoiled |= bad | _shared(session) | {nodeid for nodeid in wanted if nodeid.split("::", 1)[0] in broken}
        reported |= {row["nodeid"] for row in session["rows"]}
    passed, ran = wanted & passed - spoiled, {node.split("::", 1)[0] for node in reported}
    kept = {nodeid for nodeid in wanted - passed if nodeid in reported or (
        not recheck_unselected and nodeid.split("::", 1)[0] not in ran
        and working_source(nodeid.split("::", 1)[0]) is not None)}
    return {**dict.fromkeys(sorted(passed), "passed"), **dict.fromkeys(sorted(kept), "kept")}


def command_route(args, environ=os.environ) -> int:
    evidence = Path(args.evidence_dir)
    head = args.head or environ.get("GITHUB_SHA") or _git("rev-parse", "HEAD")
    github = GitHub(environ.get("GITHUB_REPOSITORY", ""), environ.get("GH_TOKEN", ""))
    window = _read_json(evidence / "base.json") or {}
    attempt, run_id = args.attempt, int(environ.get("GITHUB_RUN_ID") or 0)
    read = lambda *args: github.handoff_json(*args)  # noqa: E731
    # What earlier attempts left owed on this system, each record under the commit and attempt that saw it fail.
    # `whole`: every record this leg is to take over was read; otherwise its receipt only adds to theirs.
    snapshot = read_handoff(window.get("replay", {}), read, "window", run_id, max_attempt=attempt)
    prior = carried_records(snapshot, args.os)
    gaps = {}  # proven losses this step already read, keyed by run/attempt/part/base
    if isinstance(snapshot, dict):
        for gap in snapshot.get("history_gaps") or []:
            if isinstance(gap, dict):
                gaps.setdefault((gap.get("run"), gap.get("attempt"), gap.get("part"), gap.get("base")), gap)
    whole, current_tip = prior is not None, True
    if attempt > 1:
        try:
            earlier = prior_window({"id": run_id, "run_attempt": attempt, "head_sha": head}, github.get, read, first=attempt - 1)
            prior, whole = merge_records(prior or [], earlier.awaiting.get(args.os, [])), whole and not earlier.unread
            for gap in earlier.gaps:
                gaps.setdefault((gap.get("run"), gap.get("attempt"), gap.get("part"), gap.get("base")), gap)
        except Exception as error:
            print(f"::warning::previous attempt hand-over unavailable ({_reason(error)})")
            whole = False
        try:
            current_tip = github.get(f"git/ref/heads/{BRANCH}")["object"]["sha"] == head
        except Exception as error:
            print(f"::warning::cannot confirm the current branch tip ({_reason(error)})")
            current_tip = False
    prior = prior or []
    classification = _read_json(evidence / "classification.json") or {}
    pending_size = sorted({node for row in prior for node in row["ids"] if node.startswith(SIZE_PREFIX)})
    # This run's own size gate settles a recorded size red only when it measured strictly: an approved growth is no pass.
    if (pending_size and classification.get("size_checked")
            and (classification.get("size") or {}).get("mode", "block") == "block"):
        checked = size_status(read_session(evidence / "size"), pending_size)
        classification["size_passed"] = [node for node, status in checked.items() if status == "passed"]

    def save(handed):
        """What a step killed during the recheck leaves, and the final record. Earlier records this leg could not read
        leave `untracked` unknown: the summary then neither calls the leg tracked nor lets its receipt replace theirs.
        The proven losses this step already read — the replay's own, also when it travelled as an artifact, and the
        earlier attempt's in a rerun — travel as `history_gaps`, so the summary names them without a new read."""
        if gaps:
            handed = {**handed, "history_gaps": list(gaps.values())}
        _write_json(evidence / "routing.json", handed if whole else {**handed, "untracked": None})

    record = route(github, prior_failures=prior, retire=current_tip, ancestor=cache(is_ancestor),
                   origin={"head": head, "run": run_id, "attempt": attempt, "base": window.get("base")},
                   removed=lambda node: working_source(node.split("::", 1)[0]) is None,
                   classification=classification,
                   attribution=_read_json(evidence / "attribution.json") or {}, head=head, run_url=args.run_url,
                   recheck=lambda ids: recheck_ids(ids, evidence / "recheck", head=head),
                   same_run=lambda ids: same_run_verdicts(evidence, ids, head=head, system=args.os,
                                                        run_id=environ.get("GITHUB_RUN_ID", "")),
                   owed_same_run=lambda ids: same_run_verdicts(evidence, ids, head=head, system=args.os,
                                                             run_id=environ.get("GITHUB_RUN_ID", ""),
                                                             recheck_unselected=True),
                   # A book that grew is measured strictly here: a later approval covers its own merge only.
                   size_recheck=lambda base, ids: size_status(size_session(
                       evidence / "recheck-size" / base[:12], base=base), ids), system=args.os,
                   checkpoint=save)
    save(record)
    print(f"post-merge hand-over: comments on {record['comments']}, created {record['created']}, "
          f"updated {record['updated']}, narrowed {[row['issue'] for row in record['narrowed']]}, "
          f"closed {record['closed']}, {len(record['warnings'])} warning(s)")
    return 0


# --- summary ---------------------------------------------------------------------------------

def _cost_session(directory: Path) -> dict | None:
    session = read_session(directory)
    for key in ("sha", "run_id", "run_attempt"):
        if (expected := os.environ.get(f"GITHUB_{key.upper()}")) and session["github"].get(key) != expected:
            return None
    return session


def _cost_summary(session: dict | None, outcome: str) -> tuple[list[str], bool, bool]:
    """The independent workflow step and its existing session evidence, without retry or attribution.

    A known breach is a gap row's xfail that ran and failed as expected (tests/ci_verdicts.py): listed with
    its reason, it alone does not make the leg red. A skip, an xfail that never ran or broke in teardown, a
    strict XPASS and a collected test without a report leave the contract unverified."""
    if session is None:
        return [f"Cost contract: workflow **{_cell(outcome or 'unknown')}**.",
                "- **Cost contract not fully verified:** no result matches this GitHub commit, run and attempt."], False, False
    failed = candidates(session)
    expected = {node: why for node, why in expected_failures(session["rows"]).items() if node not in failed}
    skipped = sorted({row["nodeid"] for row in session["rows"] if row["outcome"] == "skipped"} - set(expected) - set(failed))
    passed = {row["nodeid"] for row in session["rows"] if row["phase"] == "call" and row["outcome"] == "passed"}
    passed.difference_update(failed, skipped, expected)
    collected = session["collected"]  # a collected test without a report never ran
    measured = (outcome in ("success", "failure") and session["final"] and session["exit"] == int(bool(failed))
                and bool(passed or failed) and not skipped
                and not session["collection"] and not session["collection_skips"]
                and type(collected) is int and collected > 0 and _parent_calls(session) == collected
                and collected == len(passed) + len(expected) + len(failed))
    complete = measured and bool(passed) and not failed
    lines = [f"Cost contract: workflow **{outcome or 'unknown'}**; pytest exit {session['exit']}; "
             f"{len(passed)} passed, {len(failed)} failed/unfinished, {len(skipped)} skipped"
             + (f", {len(expected)} known breach(es) measured" if expected else "") + "."]
    lines += [f"- Cost check `{_cell(node)}`: {_cell(why)}." for node, why in failed.items()]
    lines += [f"- Cost check `{_cell(node)}` {_unmeasured([row for row in session['rows'] if row['nodeid'] == node])}."
              for node in skipped]
    lines += [f"- Known breach `{_cell(node)}`, measured as its xfail marker expects; it does not make this leg red: "
              f"{_cell(why) or 'no reason recorded'}" for node, why in sorted(expected.items())]
    if not complete:
        lines.append("- **Cost contract not fully verified:** missing, interrupted, failed or skipped session evidence.")
    return lines, complete, measured


def _unmeasured(rows) -> str:
    """Why a cost check with a skipped report measured nothing (`expected_failures` took those that did)."""
    skips = [row for row in rows if row["outcome"] == "skipped"]
    if any("wasxfail" not in row for row in skips):
        return "was skipped, by evidence older than the xfail record: whether it ran is unknown"
    if any(row["phase"] == "teardown" and row["wasxfail"] is not None for row in skips):
        return "failed in teardown, which its xfail marker absorbed"
    if any(row["phase"] == "setup" and row["wasxfail"] is not None for row in skips):
        return "is an xfail that never ran its test"
    if any(row.get("error_type") == "XFailed" for row in skips):
        return "ended in pytest.xfail() instead of measuring"
    return ("is an xfail without its setup or teardown on record, so it is unknown whether it ran whole" if any(
        row["phase"] == "call" and row["wasxfail"] is not None for row in skips) else "was skipped")


def _same_window(base: dict, classification: dict, system: str) -> bool:
    return (bool(base.get("base")) and base["base"] == classification.get("base")
            and bool(base.get("head")) and base["head"] == classification.get("head")
            and (system != "linux" or (classification.get("size") or {}).get("base") == base["base"]))


def _window_measured(base: dict, classification: dict, system: str) -> bool:
    """A receipt for the initial measurement: the selected passes and each gate this leg runs. Retry, attribution and
    issue delivery have their own outcomes."""
    return (type(base.get("continuous")) is bool and classification.get("selection_executed") is True
            and _same_window(base, classification, system)
            and (system != "linux" or classification.get("size_checked") is True)
            and ("lint" not in (classification.get("gates") or ()) or classification.get("lint_checked") is True))


def _still_owed(base, classification, routing, system, attempt=1, environ=os.environ) -> tuple[list, bool]:
    """(records, whole): the hand-over this leg still owes, as `route` returns it, and whether the leg took over what
    earlier attempts recorded. Without the route step's receipt every FAILED of this run is owed under this run's
    name beside what the window carried in; a rerun, or a window record that travelled as an artifact, was never
    reconciled here, so that list only adds to the earlier ones."""
    if (isinstance(routing, dict) and routing.get("os") == system
            and (rows := records(routing.get("awaiting_handoff"))) is not None):
        return rows, routing.get("untracked") is not None
    carried = carried_records(base.get("replay") or {}, system)
    failed = sorted(row["id"] for row in classification.get("failed") or [])
    own = {"head": classification.get("head") or base.get("head"), "run": int(environ.get("GITHUB_RUN_ID") or 0) or None,
           "attempt": attempt, "base": base.get("base") or classification.get("base"), "ids": failed}
    return merge_records(carried or [], [own] if failed else []), carried is not None and attempt == 1


def _owed_ids(rows) -> list[str]:
    return sorted({nodeid for row in rows for nodeid in row["ids"]})


def render_summary(evidence: Path, merged=(), web: str = "", system: str = "linux", *,
                   cost_outcome: str | None = None, cost_result=None, debt=_MISSING, step_outcomes: dict | None = None) -> tuple[str, bool]:
    """(markdown, red): red when any selected test/check/stage fails or evidence is incomplete.

    `web` is the repository's address (https://github.com/owner/name); with it commits, pull
    requests and issues become links. `system` is the leg's operating system.
    """
    evidence, title = Path(evidence), f"## Post-merge check on {SYSTEMS[system]}"
    base, classification = _read_json(evidence / "base.json") or {}, _read_json(evidence / "classification.json")
    attribution, routing = _read_json(evidence / "attribution.json") or {}, _read_json(evidence / "routing.json") or {}
    if not isinstance(routing, dict) or routing.get("os", system) != system:
        routing = {}
    if not isinstance(classification, dict):
        classification = {"problems": ["the check did not finish: nothing was classified (read the step logs)"]}
    problems, failed, flaky = (list(classification.get(key) or []) for key in ("problems", "failed", "flaky"))
    if classification.get("selection_executed") is False and not problems and not any((classification.get("not_run") or {}).values()):
        problems.append("the selected pytest passes have no confirmed execution")
    if "continuous" in base and classification.get("selection_executed") and not _same_window(base, classification, system):
        problems.append("the recorded checks do not match the shared selection window")
    for step, outcome in (step_outcomes or {}).items():
        if outcome != "success":
            problems.append(f"the {step} step is {outcome or 'unknown'} (read its log)")
    if classification.get("complete") is False:
        problems.append("the retry did not finish; its unresolved reds stay unattributed")
    if debt is _MISSING:
        owed, whole = _still_owed(base, classification, routing, system)
        debt = _owed_ids(owed) if whole else None
    if debt is None:
        problems.append("earlier untracked failures could not be read")
    elif still_owed := set(debt) - {row["id"] for row in failed}:
        problems.append("earlier failures still lack a passing result or a tracking issue: " + ", ".join(sorted(still_owed)))
    red = bool(failed or problems or any((classification.get("not_run") or {}).values()))
    cost_lines = ["Cost contract: not selected on Windows (measured on Linux)."] if system == "windows" else []
    if system == "linux" and cost_outcome is not None:
        cost_lines, verified, _ = cost_result or _cost_summary(_cost_session(evidence / "cost-contract"), cost_outcome)
        red = red or not verified or cost_outcome != "success"
    lines = [f"{title} — {'RED' if red else 'GREEN'}", ""]
    if base:
        numbers = sorted({int(found[0]) for _, subject in merged if (found := re.findall(r"#(\d+)", subject))})
        lines.append(f"Selection window {_commit(base.get('base'), web)}..{_commit(base.get('head'), web)}: {len(merged)} "
                     "commit(s)" + (f" ({_numbers(numbers, web, 'pull')})" if numbers else "")
                     + f". Base: {base.get('why')}.")
    selection = classification.get("selection") or {}
    lines.append(f"Selection: **{selection.get('tier')}**, {selection.get('file_count')} of "
                 f"{selection.get('available')} Python modules; Node lane: {selection.get('node')}."
                 + (" A full escalation was deferred to the release matrix: this leg ran the contract floor and what the "
                    "other changes select."
                    if selection.get("deferred_full") else ""))
    if calls := selection.get("calls"):
        lines.append("Pytest calls: " + ", ".join(f"{label} {count}" if count else f"{label} empty"
                                                 for label, count in calls.items()) + ".")
    lines.append(_gates(classification, web))
    if lanes := classification.get("diagnostics_incomplete"):
        lines.append(f"Diagnostics incomplete: no valid results.json for {', '.join(lanes)}. See inventory and phase journal.")
    lines += cost_lines
    lines += [f"- **The check is incomplete:** {_cell(problem)}." for problem in problems]
    gaps = {}
    for gap in [*((base.get("replay") or {}).get("history_gaps") or []), *(routing.get("history_gaps") or [])]:
        if isinstance(gap, dict):
            gaps.setdefault((gap.get("run"), gap.get("attempt"), gap.get("part"), gap.get("base")), gap)
    lines += [f"- **History gap:** {_cell(gap_text(gap))}." for gap in gaps.values()
              if gap.get("part") in (system, "window", "jobs")]
    verdicts = {row["id"]: row for row in attribution.get("verdicts") or []}
    handed = {nodeid: group for group in routing.get("handed") or [] for nodeid in group.get("ids") or []}
    untracked = routing.get("untracked")
    if failed:
        counts = {kind: sum(verdicts.get(row["id"], {}).get("verdict", "unattributed") == kind for row in failed)
                  for kind in ("culprit", "pre-existing", "unattributed")}
        lines += ["", f"### Failed checks ({len(failed)})", "",
                  f"Attributed to a merge: {counts['culprit']}; also failed at the base: {counts['pre-existing']}; "
                  f"unattributed: {counts['unattributed']}. All remain RED. A base failure does not establish the same cause.", "",
                  "| Test | First failure | Verdict | First bad | Last good | Handed to |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for row in failed:
            verdict, group = verdicts.get(row["id"], {}), handed.get(row["id"], {})
            to = [f"PR {_numbers([group['pr']], web, 'pull')}"] if group.get("pr") else []
            if isinstance(untracked, list) and row["id"] in untracked:
                to.append("tracking issue unconfirmed")
            elif group.get("issue"):
                to.append(f"issue {_numbers([group['issue']], web, 'issues')}")
            elif isinstance(untracked, list) and routing.get("os") == system:
                to.append("already tracked by an open issue")
            else:
                to.append("tracking issue unconfirmed")
            lines.append(f"| `{_cell(row['id'])}` | {_cell(row['why'])} (retry: {_cell(row.get('retry', 'unknown'))}) | {_cell(verdict.get('verdict', 'unattributed'))}"
                         + (f": {_cell(verdict['why'])}" if verdict.get("why") else "")
                         + f" | {_commit(verdict.get('culprit'), web)} | {_commit(verdict.get('last_good'), web)}"
                         f" | {', '.join(to) or '—'} |")
    if flaky:
        lines += ["", f"### Flaky ({len(flaky)}): passed on their one retry; reported, not handed over", ""]
        lines += [f"- `{_cell(row['id'])}` ({_cell(row['why'])}, {row['pass']} pass)" for row in flaky]
    for label, nodes in (classification.get("not_run") or {}).items():
        lines += ["", f"**{len(nodes)} test(s) of the {label} pass never ran** (the session stopped early), so this "
                  "run did not check them: " + ", ".join(f"`{_cell(node)}`" for node in nodes[:20])
                  + (" …" if len(nodes) > 20 else "")]
    lines += [f"- {_cell(note)}" for note in attribution.get("notes") or []]
    if routing:
        narrowed = [f"{_numbers([row.get('issue')], web, 'issues')} ({len(row.get('ids') or [])} passing again)"
                    for row in routing.get("narrowed") or [] if isinstance(row, dict)]
        lines += ["", "### Hand-over", "", f"Pull request comments: {_numbers(routing.get('comments'), web, 'pull')}. "
                  f"Issues created: {_numbers(routing.get('created'), web, 'issues')}; updated: "
                  f"{_numbers(routing.get('updated'), web, 'issues')}; "
                  + (f"narrowed: {', '.join(narrowed)}; " if narrowed else "")
                  + f"closed as green: {_numbers(routing.get('closed'), web, 'issues')}."]
        if routing.get("complete") is False:
            lines.append("- The hand-over above went out, but the recheck of the open issues did not finish: "
                         "what it narrowed or closed afterwards is not recorded here.")
        if stalled := [node for node, status in (routing.get("rechecked") or {}).items() if status == "unknown"]:
            lines.append(f"- {len(stalled)} tracked test(s) got no result in the recheck (it starts none after "
                         f"{RECHECK_SECONDS} s, and a deselected test never reports), so they stay tracked.")
        lines += [f"- **Warning:** {_cell(warning)}" for warning in routing.get("warnings") or []]
    return "\n".join(lines) + "\n", red


def _gates(classification: dict, web: str = "") -> str:
    """The deterministic gates this leg ran, and against what the size check measured."""
    gates = classification.get("gates") or []
    if not gates:
        return "Gates: none on this leg (ruff F and the size check run on the Linux leg)."
    unchecked = {gate: "" if classification.get(f"{gate}_checked") else " (unchecked)" for gate in ("lint", "size")}
    named = [f"ruff F{unchecked['lint']}"] if "lint" in gates else []
    if sized := classification.get("size"):
        mode = (f"approved by the owner at {', '.join(_commit(sha, web) for sha in sized['approved'])}"
                if sized.get("approved") else sized.get("mode") if sized.get("decided")
                else f"{sized.get('mode')}, as the merges' labels were not read")
        named.append(f"the size check against {_commit(sized.get('base'), web)} (book growth: {mode}){unchecked['size']}")
    return f"Gates: {', '.join(named)}."


def _cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _commit(sha, web: str = "") -> str:
    if not sha:
        return "—"
    return f"[`{sha[:12]}`]({web}/commit/{sha})" if web else f"`{sha[:12]}`"


def _numbers(values, web: str = "", kind: str = "issues") -> str:
    return ", ".join(f"[#{value}]({web}/{kind}/{value})" if web else f"#{value}" for value in values or []) or "none"


def command_summary(args) -> int:
    evidence = Path(args.evidence_dir)
    base = _read_json(evidence / "base.json") or {}
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    web = f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{repo}" if repo else ""
    cost_outcome = os.environ.get("COST_CONTRACT_OUTCOME", "unknown")
    cost = _cost_session(evidence / "cost-contract") if args.os == "linux" else None
    cost_result = _cost_summary(cost, cost_outcome) if args.os == "linux" else None
    classification = _read_json(evidence / "classification.json") or {}
    routing = _read_json(evidence / "routing.json")
    owed, whole = _still_owed(base, classification, routing, args.os, args.attempt)
    debt = _owed_ids(owed) if whole else None
    tracked = debt == []
    # The receipt the next window reads: the records still owed; partial when this leg could not take over the
    # earlier ones, so its list adds to theirs; null when nothing was classified and it cannot say what failed.
    receipt = None if not isinstance(classification.get("failed"), list) else owed if whole else {"partial": owed}
    measured = _window_measured(base, classification, args.os)
    text, red = render_summary(evidence, merges(base["base"], base["head"]) if base.get("base") and base.get("head")
                               else [], web, args.os, cost_outcome=cost_outcome, cost_result=cost_result, debt=debt,
                               step_outcomes={step: os.environ[key] for step in ("classify", "attribute", "route")
                                              if (key := f"POST_MERGE_{step.upper()}_OUTCOME") in os.environ})
    # A failed cost test is handed to no issue: it keeps the base, as an untracked FAILED does (a known breach is no red).
    cost_red = bool(cost and candidates(cost))
    # A fully observed red still measured the push. A missing or partial cost export cannot certify even that.
    if cost_result:
        measured = measured and cost_result[2]
    advance = measured and base["continuous"] and tracked and not cost_red
    if "continuous" in base:
        text += ("\nThis leg measured the stated range.\n" if measured else "\nThis leg could not confirm the stated range.\n")
        if not base["continuous"]:
            text += "Earlier coverage remains pending.\n"
        if measured and not tracked:
            text += "Keep the starting base: at least one FAILED check on this system has no confirmed tracking issue.\n"
        if measured and cost_red:
            text += "Keep the starting base: a cost-contract test failed, and no issue tracks it.\n"
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
            stream.write(text)
    sys.stdout.buffer.write(text.encode("utf-8", "replace"))
    sys.stdout.flush()
    if destination := os.environ.get("GITHUB_OUTPUT"):
        with open(destination, "a", encoding="utf-8") as stream:
            scope = "chain" if advance else "push" if measured else ""
            public, stored = _project_handoff(receipt, evidence, os.environ)
            stream.write(f"untracked={_compact_json(public)}\nhandoff_file={'yes' if stored else 'no'}\n")
            stream.write(f"window_scope={scope}\n")
    for row in (classification.get("failed") or [])[:10]:
        print(f"::error title=Post-merge red on {SYSTEMS[args.os]}::{row['id']} ({row['why']})")
    if cost is not None:  # beside the reds, never one of them: the summary lists each with its reason
        for node in sorted(expected_failures(cost["rows"]))[:10]:
            print(f"::notice title=Known cost-contract breach on Linux::{node} (an expected failure, measured)")
    return 1 if red else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    step = commands.add_parser("base", help="choose this check's base")
    step.add_argument("--evidence-dir", type=Path)
    step.add_argument("--head")
    step = commands.add_parser("growth", help="decide the book-growth mode of the merges since the base")
    step.add_argument("--evidence-dir", type=Path, required=True)
    step.add_argument("--base", required=True)
    step.add_argument("--head", required=True)
    step = commands.add_parser("classify", help="retry each red once: FAILED or FLAKY")
    step.add_argument("--evidence-dir", type=Path, required=True)
    step.add_argument("--head")
    step.add_argument("--lint", action="store_true", help="run the lint gate (ruff F) on this leg")
    step.add_argument("--size", action="store_true", help="run the size check on this leg, against --base")
    step.add_argument("--selected-outcome", help="the selected workflow step's outcome before continue-on-error")
    step.add_argument("--base", help="the check's base, which the size check measures against")
    step = commands.add_parser("attribute", help="bisect each FAILED test to its first bad merge")
    step.add_argument("--evidence-dir", type=Path, required=True)
    step.add_argument("--base", required=True)
    step.add_argument("--head", required=True)
    step.add_argument("--max-probes", type=int, default=MAX_PROBES)
    step.add_argument("--work-dir", type=Path, help="parent of the probe worktrees (default: RUNNER_TEMP)")
    step = commands.add_parser("route", help="close this system's green issues and hand each red over")
    step.add_argument("--evidence-dir", type=Path, required=True)
    step.add_argument("--run-url", required=True)
    step.add_argument("--head")
    step.add_argument("--attempt", type=int, default=1, help="GitHub workflow attempt number")
    step.add_argument("--os", choices=sorted(SYSTEMS), default="linux", help="the leg's operating system")
    step = commands.add_parser("summary", help="write the job summary; exit 1 when red")
    step.add_argument("--evidence-dir", type=Path, required=True)
    step.add_argument("--attempt", type=int, default=1, help="GitHub workflow attempt number")
    step.add_argument("--os", choices=sorted(SYSTEMS), default="linux", help="the leg's operating system")
    args = parser.parse_args(argv)
    if args.command == "classify" and args.size and not args.base:
        parser.error("classify --size measures against the check's base: pass --base")
    return {"base": command_base, "growth": command_growth, "classify": command_classify,
            "attribute": command_attribute, "route": command_route, "summary": command_summary}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
