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
Each FAILED test or finding is bisected over the first-parent commits since the base to the merge
that broke it; that pull request's author gets one comment and Ouroboros an issue labelled
`post-merge-red` for that leg's system (scripts/post_merge_routing.py), which the check closes by
itself once the tests pass and the findings are gone. It blocks nothing, reverts nothing, runs on
no schedule and talks to GitHub only with the workflow's own GITHUB_TOKEN. Its subprocesses are
git, ruff and `scripts/run_tests.py`; everything else is stdlib.

DIR is the evidence root of `scripts/run_tests.py --selected --keep-going --evidence-dir DIR`
(selection.json, parallel/, serial/: tests/ci_evidence.py's results.json and events.jsonl). Each
step adds its record beside them: base.json, growth.json, classification.json (sessions in retry/
and size/), attribution.json (probes/), routing.json (recheck/, recheck-size/).
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # run as a script, only scripts/ is on sys.path
    sys.path.insert(0, str(REPO_ROOT))

from scripts.post_merge_routing import (  # noqa: E402  -- stdlib only, as is this module
    BRANCH, LINT_PREFIX, NODE_LANE, SIZE_PREFIX, SYSTEMS, GitHub, _reason, route,
)

WORKFLOW_FILE = "post-merge.yml"
LINT_ARGS = ("check", "--select", "F", "--output-format", "json", "--exit-zero")  # the release lint gate
# What a module of the size check declares: a decorated test, or a `pytestmark` carrying the marker.
# A line that only mentions it (a fixture string, a comment) declares nothing. POSIX classes: git grep -E.
SIZE_DECLARATIONS = (r"^[[:space:]]*@pytest\.mark\.size_ratchet([^[:alnum:]_]|$)",
                     r"^[[:space:]]*pytestmark[[:space:]]*[:=].*pytest\.mark\.size_ratchet([^[:alnum:]_]|$)")
MAX_PROBES = 16
ZERO_SHA = "0" * 40
# Retry, probe and re-check sessions keep the per-test bound of both selected passes
# (ouroboros/preflight_runner.py SERIAL_PASS_FLAGS). On Linux SIGALRM fails only the hung test;
# on Windows the thread method ends the session, whose journal still names the test that was
# running. Either way a hang ends as a named failure instead of the job's timeout, and a module
# that does not import leaves the other ids running.
SESSION_FLAGS = ("--continue-on-collection-errors", "--timeout=300")
_COMPLETE_EXITS = (0, 1, 5)  # pytest finished its session: green, some red, or nothing collected
_USAGE_ERROR = 4


# --- git -------------------------------------------------------------------------------------

def _git(*args: str, cwd: Path | None = None, strip: bool = True) -> str:
    out = subprocess.run(["git", *args], cwd=cwd or REPO_ROOT, capture_output=True, encoding="utf-8",
                         errors="replace", check=True, timeout=300).stdout
    return out.strip() if strip else out


def is_ancestor(older: str, newer: str) -> bool:
    try:
        return subprocess.run(["git", "merge-base", "--is-ancestor", older, newer], cwd=REPO_ROOT,
                              capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


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
        try:
            return _git("show", f"{sha}:{path}", strip=False)
        except subprocess.CalledProcessError:
            return None
    return read


def working_source(path: str) -> str | None:
    try:
        return (REPO_ROOT / path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
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


def _report(row) -> bool:
    return (isinstance(row, dict) and isinstance(row.get("nodeid"), str)
            and isinstance(row.get("phase"), str) and row.get("outcome") in {"passed", "failed", "skipped"})


def read_session(directory: Path) -> dict:
    """One pytest session as tests/ci_evidence.py recorded it in `directory`.

    `final`: it exported results.json; without that its incremental journal (events.jsonl) gives
    the reports. `running`: tests the journal saw start and never finish. `exit`: pytest's status.
    """
    directory = Path(directory)
    running, journal_rows, journal = {}, [], True
    try:
        lines = (directory / "events.jsonl").read_text(encoding="utf-8", errors="replace").split("\n")
    except OSError:
        lines, journal = [], False
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue  # a kill can tear the last line
        if not isinstance(row, dict) or not isinstance(row.get("nodeid"), str):
            continue
        if row.get("event") == "start":
            running[row["nodeid"]] = None
        elif row.get("event") == "finish" or (row.get("event") == "report" and row.get("phase") == "crash"):
            running.pop(row["nodeid"], None)  # a crashed worker never sends finish
        if row.get("event") == "report" and _report(row):
            journal_rows.append(row)
    data = _read_json(directory / "results.json")
    reports = data.get("reports") if isinstance(data, dict) else None
    if isinstance(reports, list):
        status = data.get("session_exit_code")
        return {"final": True, "evidence": True, "rows": [row for row in reports if _report(row)],
                "collection": [row["nodeid"] for row in data.get("collection_failures") or []
                               if isinstance(row, dict) and isinstance(row.get("nodeid"), str)],
                "running": list(running), "exit": status if type(status) is int else None}
    return {"final": False, "evidence": journal, "rows": journal_rows, "collection": [],
            "running": list(running), "exit": None}


def candidates(session: dict) -> dict[str, str]:
    """{id: why} of everything a session saw go red: a failed phase, a crashed worker, a module
    that did not collect, a test still running when the session ended."""
    found = {}
    for row in session["rows"]:
        if row["phase"] == "crash":
            found.setdefault(row["nodeid"], "its worker crashed")
        elif row["outcome"] == "failed":
            found.setdefault(row["nodeid"], f"failed in {row['phase']}")
    for path in session["collection"]:
        found.setdefault(path, "did not collect")
    for nodeid in session["running"]:
        found.setdefault(nodeid, "was running when the session ended")
    return found


def outcomes(session: dict, ids) -> dict[str, str]:
    """Each id in one session: failed, passed, skipped, absent (the session completed without it,
    so its tree has no such test) or unknown (the session ended before reporting it).

    A module id (one that did not collect) stands for every test of that module.
    """
    failed, passed, skipped = set(session["running"]), set(), set()  # a test running at the end hung
    for row in session["rows"]:
        if row["outcome"] == "failed" or row["phase"] == "crash":
            failed.add(row["nodeid"])
        elif row["outcome"] == "skipped":
            skipped.add(row["nodeid"])
        elif row["phase"] == "call":
            passed.add(row["nodeid"])
    broken = set(session["collection"])
    complete = session["final"] and session["exit"] in _COMPLETE_EXITS
    result = {}
    for nodeid in ids:
        path = nodeid.split("::", 1)[0]
        members = ({nodeid} if "::" in nodeid
                   else {node for node in failed | passed | skipped if node.split("::", 1)[0] == path})
        if path in broken or members & failed:
            result[nodeid] = "failed"
        elif members & passed:
            result[nodeid] = "passed"
        elif members & skipped:
            result[nodeid] = "skipped"
        else:
            result[nodeid] = "absent" if complete else "unknown"
    return result


def container_args(ids, read_source) -> tuple[list[str], list[str]]:
    """Pytest arguments that run every id a tree still defines, and the ids whose module it lacks.

    pytest refuses a whole session when one argument names nothing ("not found", exit 4), and a
    test added by a later merge does not exist at an earlier commit. So each id becomes the deepest
    node its module still defines by `ast` of that tree (function without parameters, class, or
    the module), and the session's results are read for the exact id afterwards.
    """
    args, absent = set(), []
    for nodeid in ids:
        path, *names = nodeid.split("::")
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


def check_ids(ids, directory: Path, *, read_source, run=run_pytest) -> dict[str, str]:
    """`outcomes` of exactly `ids` after one session of the deepest node each still has."""
    args, absent = container_args(ids, read_source)
    result = {nodeid: "absent" for nodeid in absent}
    if args:
        run(args, directory)
        result.update(outcomes(read_session(directory), [nodeid for nodeid in ids if nodeid not in absent]))
    return result


# --- base ------------------------------------------------------------------------------------

def choose_base(runs, before: str, head: str, *, ancestor=None, parent=None) -> tuple[str, str]:
    """(base, why): the head of the newest completed check, else the push's previous tip, else HEAD^1.

    A run replaced while pending never completed, so its merges fall into this window as well.
    A run's conclusion covers both of its legs, so both legs of a check choose the same base.
    """
    ancestor, parent = ancestor or is_ancestor, parent or first_parent
    for run in runs:
        sha = run.get("head_sha") or ""
        if (run.get("status") == "completed" and run.get("conclusion") in ("success", "failure")
                and sha and sha != head and ancestor(sha, head)):
            return sha, f"head of the last completed check (run {run.get('id')}, {run.get('conclusion')})"
    if before and before != ZERO_SHA and before != head and ancestor(before, head):
        return before, "the branch tip before this push"
    return parent(head), "the first parent of this push"


def command_base(args, environ=os.environ) -> int:
    head = args.head or environ.get("GITHUB_SHA") or _git("rev-parse", "HEAD")
    runs = []
    try:
        github = GitHub(environ.get("GITHUB_REPOSITORY", ""), environ.get("GH_TOKEN", ""))
        page = github.get(f"actions/workflows/{WORKFLOW_FILE}/runs?branch={BRANCH}&event=push"
                          "&status=completed&per_page=50")
        runs = (page.get("workflow_runs") or []) if isinstance(page, dict) else []
    except Exception as error:  # noqa: BLE001 -- unreadable history: fall back, never crash
        print(f"::warning title=Post-merge base::earlier checks are unreadable ({_reason(error)}); falling back")
    base, why = choose_base(runs, environ.get("BEFORE", ""), head)
    print(f"post-merge base: {base} ({why})")
    if args.evidence_dir:
        _write_json(Path(args.evidence_dir) / "base.json", {"format": 1, "base": base, "head": head, "why": why})
    if environ.get("GITHUB_OUTPUT"):
        with open(environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
            output.write(f"base={base}\n")
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


def size_modules(root: Path) -> list[str]:
    """The test modules of the size check in the checkout `root`: those that declare a `size_ratchet` test.

    The same tests as the release's `pytest tests/ -m size_ratchet`, without collecting the suite, and
    no module that merely mentions the marker: its collection is no business of the size check.
    """
    try:
        listed = _git("grep", "-l", "-E", *(arg for pattern in SIZE_DECLARATIONS for arg in ("-e", pattern)),
                      "--", "tests", cwd=Path(root))
    except (OSError, subprocess.SubprocessError):
        return []  # git grep exits 1 when nothing matches
    return sorted(path for path in listed.splitlines() if Path(path).name.startswith("test_") and path.endswith(".py"))


def size_session(directory: Path, *, base: str, mode: str = "block", cwd: Path | None = None,
                 run=run_pytest) -> dict | None:
    """One session of the size check in `cwd`, measured against `base` with the book-growth `mode`;
    None when no module of that tree marks a size test."""
    root = Path(cwd or REPO_ROOT)
    modules = size_modules(root)
    if not modules:
        return None
    run(["-m", "size_ratchet", *modules], directory, cwd=root,
        env={"OURO_SIZE_RATCHET_BASE_REF": base, "OURO_BOOK_GROWTH_MODE": mode})
    return read_session(directory)


def size_status(session: dict | None, ids) -> dict[str, str]:
    """`outcomes` of size ids in one size session; every one unknown when it did not run to its end."""
    if not (session and session["final"] and session["exit"] in (0, 1)):
        return dict.fromkeys(ids, "unknown")
    found = outcomes(session, [nodeid.removeprefix(SIZE_PREFIX) for nodeid in ids])
    return {nodeid: found[nodeid.removeprefix(SIZE_PREFIX)] for nodeid in ids}


def size_findings(session: dict | None) -> tuple[dict[str, str] | None, list[str]]:
    """({id: why} of a size session's reds, or None when it did not run to its end; its problems).

    A size test that skips measured nothing (a base the clone lacks, an unexpected event), so it is
    a problem of the check, never a pass.
    """
    if not (session and session["final"] and session["exit"] in (0, 1)):
        return None, ["the size check could not run or finish, so it went unchecked"]
    skipped = sorted({row["nodeid"] for row in session["rows"] if row["outcome"] == "skipped"})
    return ({SIZE_PREFIX + nodeid: why for nodeid, why in candidates(session).items()},
            [f"the size check skipped `{nodeid}`, so it measured nothing there" for nodeid in skipped])


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
              "selection": {}}
    if not isinstance(selection, dict):
        record["problems"].append("the selected run left no selection.json (its selection failed or it died early)")
        return record
    record["base"] = selection.get("base")
    record["selection"] = {key: selection.get(key) for key in ("tier", "file_count", "available", "node", "outcomes",
                                                               "deferred_full")}
    found = {}  # id -> (pass, why)
    for label, outcome in (selection.get("outcomes") or {}).items():
        session = read_session(evidence / label)
        red = candidates(session)
        for nodeid, why in red.items():
            found.setdefault(nodeid, (label, why))
        inventory = (selection.get("inventory") or {}).get(label) or {}
        if missing := set(inventory.get("expected") or []) - set(inventory.get("executed") or []) - set(red):
            record["not_run"][label] = sorted(missing)
        if outcome == "skipped":  # never started, or stopped before it recorded its outcome
            record["problems"].append(f"the {label} pass did not finish" if session["evidence"]
                                      else f"the {label} pass did not run")
        elif outcome == "failure" and not red:
            record["problems"].append(f"the {label} pass was red without a failing test (read its log)"
                                      if session["evidence"] else f"the {label} pass left no evidence")
    if selection.get("node") == "failure":
        found[NODE_LANE] = ("node", "the Node lane was red")
    findings = lint() if lint is not None else {}
    record["lint_checked"] = lint is not None and findings is not None
    if findings is None:
        record["problems"].append("ruff could not run, so the lint gate went unchecked")
    for finding, message in (findings or {}).items():
        found[finding] = ("lint", message)
    if size is not None:
        sized, problems = size_findings(size())
        record["size_checked"] = sized is not None
        record["problems"] += problems
        found.update({nodeid: ("size", why) for nodeid, why in (sized or {}).items()})
    checkpoint({**record, "complete": False, "flaky": [], "failed": [
        {"id": nodeid, "pass": label, "why": why, "retry": "not run"} for nodeid, (label, why) in sorted(found.items())]})
    ids = sorted(nodeid for nodeid in found if nodeid != NODE_LANE and not nodeid.startswith((LINT_PREFIX, SIZE_PREFIX)))
    status = {}
    if ids:
        run(ids, evidence / "retry")
        session = read_session(evidence / "retry")
        status = outcomes(session, ids)
        if session["exit"] == _USAGE_ERROR and not session["rows"]:
            # One id pytest cannot find (a redacted parameter) refuses the whole session.
            status = check_ids(ids, evidence / "retry" / "nodes", read_source=read_source, run=run)
    if NODE_LANE in found:
        status[NODE_LANE] = "passed" if node() == 0 else "failed"
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
        extra["size"] = {"base": args.base, "mode": growth.get("mode") or "block", "decided": bool(growth),
                         "approved": growth.get("approved") or []}
        size = lambda: size_session(evidence / "size", base=args.base, mode=extra["size"]["mode"])  # noqa: E731
    record = classify(evidence, lint=lint_findings if args.lint else None, size=size,
                      checkpoint=lambda provisional: _write_json(output, {**provisional, **extra}))
    _write_json(output, {**record, **extra})
    print(f"post-merge: {len(record['failed'])} failed, {len(record['flaky'])} flaky after one retry"
          + "".join(f"; {problem}" for problem in record["problems"]))
    return 0


# --- attribute -------------------------------------------------------------------------------

def bisect(chain: list[str], ids, probe, *, max_probes: int = MAX_PROBES) -> list[dict]:
    """The first commit of `chain` at which each id fails; chain[0] is the base, chain[-1] the head.

    Every id fails at the head. The base is probed first: an id failing there is pre-existing.
    The others share bisection intervals, and a probe at a midpoint runs every unresolved id whose
    interval contains it, so ids broken by one merge cost one bisection. `probe(sha, ids)` answers
    {id: "fail" | "pass" | "error"}. Past `max_probes` probes the rest stays unattributed.
    """
    ids, verdicts, pending, used = sorted(set(ids)), {}, {}, 0
    if len(chain) < 2:
        return [{"id": nodeid, "verdict": "pre-existing", "why": "no commit since the base"} for nodeid in ids]
    if ids:
        used += 1
        answers = probe(chain[0], ids)
        for nodeid in ids:
            answer = answers.get(nodeid, "error")
            if answer == "fail":
                verdicts[nodeid] = {"verdict": "pre-existing", "why": f"already fails at the base {chain[0][:12]}"}
            elif answer == "pass":
                pending[nodeid] = (0, len(chain) - 1)
            else:
                verdicts[nodeid] = {"verdict": "unattributed", "why": f"the probe at {chain[0][:12]} could not run it"}
    while pending:
        for nodeid, (good, bad) in list(pending.items()):
            if bad - good == 1:
                verdicts[nodeid] = {"verdict": "culprit", "culprit": chain[bad], "last_good": chain[good]}
                del pending[nodeid]
        if pending and used >= max_probes:
            for nodeid, (good, bad) in pending.items():
                verdicts[nodeid] = {"verdict": "unattributed", "why": f"the probe budget ({max_probes}) ran out; "
                                    f"first bad after {chain[good][:12]}, at or before {chain[bad][:12]}"}
            pending = {}
        if not pending:
            break
        good, bad = min(pending.values(), key=lambda interval: (interval[0] - interval[1], interval))
        middle = (good + bad) // 2
        targets = sorted(nodeid for nodeid, (low, high) in pending.items() if low < middle < high)
        used += 1
        answers = probe(chain[middle], targets)
        for nodeid in targets:
            low, high = pending[nodeid]
            answer = answers.get(nodeid, "error")
            if answer in ("fail", "pass"):
                pending[nodeid] = (low, middle) if answer == "fail" else (middle, high)
            else:
                verdicts[nodeid] = {"verdict": "unattributed", "why": f"the probe at {chain[middle][:12]} could not run it"}
                del pending[nodeid]
    return [{"id": nodeid, **verdicts[nodeid]} for nodeid in ids]


class Prober:
    """`probe(sha, ids)` for `bisect`: a detached worktree of `sha` runs the ids with its own runner.

    post-merge.yml installs the dependencies but not the project, so every process of a probe
    imports the worktree's tree; nothing can fall back to the head checkout's code. `size(tree,
    directory)` answers one size session of the tree (`size_session`, against the check's base).
    """

    def __init__(self, evidence: Path, work: Path, head: str, *, run=None, node=None, lint=None, size=None):
        self.evidence, self.work = Path(evidence), Path(work)
        self.run, self.node, self.lint = run or run_pytest, node or run_node_lane, lint or lint_findings
        self.size = size or (lambda tree, directory: None)
        self.lock, self.records = _blob(head, "uv.lock"), []

    def __call__(self, sha: str, ids) -> dict[str, str]:
        label = f"{len(self.records) + 1:02d}-{sha[:12]}"
        record = {"sha": sha, "ids": sorted(ids), "dependencies_differ": _blob(sha, "uv.lock") != self.lock}
        self.records.append(record)
        tree = self.work / f"pmc-{label}"
        try:
            _git("worktree", "add", "--detach", "--quiet", str(tree), sha)
        except (OSError, subprocess.SubprocessError) as error:
            record["error"] = f"the worktree failed: {error}"
            return {nodeid: "error" for nodeid in ids}
        try:
            answers = lint_answers(tree, [nodeid for nodeid in ids if nodeid.startswith(LINT_PREFIX)], self.lint)
            if sized := [nodeid for nodeid in ids if nodeid.startswith(SIZE_PREFIX)]:
                found = size_status(self.size(tree, self.evidence / "probes" / f"{label}-size"), sized)
                answers.update({nodeid: {"failed": "fail", "passed": "pass", "absent": "pass"}.get(value, "error")
                                for nodeid, value in found.items()})
            tests = [nodeid for nodeid in ids if nodeid != NODE_LANE and not nodeid.startswith((LINT_PREFIX, SIZE_PREFIX))]
            if tests:
                found = check_ids(tests, self.evidence / "probes" / label, read_source=tree_source(sha),
                                  run=lambda args, directory: self.run(args, directory, cwd=tree))
                answers.update({nodeid: {"failed": "fail", "unknown": "error"}.get(value, "pass")
                                for nodeid, value in found.items()})
            if NODE_LANE in ids:
                answers[NODE_LANE] = "pass" if self.node(tree) == 0 else "fail"
            record["answers"] = answers
            return answers
        finally:
            remove_worktree(tree)


def lint_answers(tree: Path, findings, lint) -> dict[str, str]:
    """`probe` answers for lint findings: ruff over their files as `tree` has them (absent = pass)."""
    if not findings:
        return {}
    paths = sorted({path for path in (finding.split(" ", 3)[2] for finding in findings) if (Path(tree) / path).is_file()})
    present = lint(tree, paths) if paths else {}
    return {finding: "error" if present is None else "fail" if finding in present else "pass" for finding in findings}


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
    ids = [row["id"] for row in classification.get("failed", [])]
    sized = classification.get("size") or {}  # each probe measures as the check did: same base, same mode
    record = {"format": 1, "base": args.base, "head": args.head, "max_probes": args.max_probes, "notes": [],
              "probes": [], "verdicts": [{"id": nodeid, "verdict": "unattributed",
                                          "why": "the attribution did not finish"} for nodeid in ids]}
    _write_json(evidence / "attribution.json", record)  # what a killed step leaves behind
    if not ids:
        return 0
    chain = first_parent_chain(args.base, args.head)
    record["commits"] = chain[1:]
    work = Path(tempfile.mkdtemp(prefix="pmc-", dir=args.work_dir or os.environ.get("RUNNER_TEMP") or None))
    prober = Prober(evidence, work, args.head, size=lambda tree, directory: size_session(
        directory, base=sized.get("base") or args.base, mode=sized.get("mode") or "block", cwd=tree))
    try:
        record["verdicts"] = bisect(chain, ids, prober, max_probes=args.max_probes)
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

def command_route(args, environ=os.environ) -> int:
    evidence = Path(args.evidence_dir)
    head = args.head or environ.get("GITHUB_SHA") or _git("rev-parse", "HEAD")
    record = route(GitHub(environ.get("GITHUB_REPOSITORY", ""), environ.get("GH_TOKEN", "")),
                   classification=_read_json(evidence / "classification.json") or {},
                   attribution=_read_json(evidence / "attribution.json") or {}, head=head, run_url=args.run_url,
                   recheck=lambda ids: check_ids(ids, evidence / "recheck", read_source=working_source),
                   # A book that grew is measured strictly here: a later approval covers its own merge only.
                   size_recheck=lambda base, ids: size_status(size_session(
                       evidence / "recheck-size" / base[:12], base=base), ids), system=args.os)
    _write_json(evidence / "routing.json", record)
    print(f"post-merge hand-over: comments on {record['comments']}, created {record['created']}, "
          f"updated {record['updated']}, closed {record['closed']}, {len(record['warnings'])} warning(s)")
    return 0


# --- summary ---------------------------------------------------------------------------------

def render_summary(evidence: Path, merged=(), web: str = "", system: str = "linux") -> tuple[str, bool]:
    """(markdown, red): red when a FAILED test remains after its retry or the check did not finish.

    `web` is the repository's address (https://github.com/owner/name); with it commits, pull
    requests and issues become links. `system` is the leg's operating system.
    """
    evidence, title = Path(evidence), f"## Post-merge check on {SYSTEMS[system]}"
    base, classification = _read_json(evidence / "base.json") or {}, _read_json(evidence / "classification.json")
    attribution, routing = _read_json(evidence / "attribution.json") or {}, _read_json(evidence / "routing.json") or {}
    if not isinstance(classification, dict):
        return f"{title} — RED\n\nThe check did not finish: nothing was classified (read the step logs).\n", True
    problems, failed, flaky = (list(classification.get(key) or []) for key in ("problems", "failed", "flaky"))
    if classification.get("complete") is False:
        problems.append("the retry did not finish, so every red below counts as FAILED and none was bisected")
    red = bool(failed or problems)
    lines = [f"{title} — {'RED' if red else 'GREEN'}", ""]
    if base:
        numbers = sorted({int(found[0]) for _, subject in merged if (found := re.findall(r"#(\d+)", subject))})
        lines.append(f"Checked {_commit(base.get('base'), web)}..{_commit(base.get('head'), web)}: {len(merged)} "
                     "commit(s)" + (f" ({_numbers(numbers, web, 'pull')})" if numbers else "")
                     + f". Base: {base.get('why')}.")
    selection = classification.get("selection") or {}
    lines.append(f"Selection: **{selection.get('tier')}**, {selection.get('file_count')} of "
                 f"{selection.get('available')} Python modules; Node lane: {selection.get('node')}."
                 + (" A full escalation was deferred to the release matrix: this leg ran the contract floor and what the "
                    "other changes select."
                    if selection.get("deferred_full") else ""))
    lines.append(_gates(classification, web))
    lines += [f"- **The check is incomplete:** {_cell(problem)}." for problem in problems]
    verdicts = {row["id"]: row for row in attribution.get("verdicts") or []}
    handed = {nodeid: group for group in routing.get("handed") or [] for nodeid in group.get("ids") or []}
    if failed:
        lines += ["", f"### Failed ({len(failed)}): red again on retry", "",
                  "| Test | First failure | Verdict | First bad | Last good | Handed to |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for row in failed:
            verdict, group = verdicts.get(row["id"], {}), handed.get(row["id"], {})
            to = [f"PR {_numbers([group['pr']], web, 'pull')}"] if group.get("pr") else []
            to += [f"issue {_numbers([group['issue']], web, 'issues')}"] if group.get("issue") else []
            lines.append(f"| `{_cell(row['id'])}` | {_cell(row['why'])} | {_cell(verdict.get('verdict', 'unattributed'))}"
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
        lines += ["", "### Hand-over", "", f"Pull request comments: {_numbers(routing.get('comments'), web, 'pull')}. "
                  f"Issues created: {_numbers(routing.get('created'), web, 'issues')}; updated: "
                  f"{_numbers(routing.get('updated'), web, 'issues')}; closed as green: "
                  f"{_numbers(routing.get('closed'), web, 'issues')}."]
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
    text, red = render_summary(evidence, merges(base["base"], base["head"]) if base.get("base") and base.get("head")
                               else [], web, args.os)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
            stream.write(text)
    sys.stdout.buffer.write(text.encode("utf-8", "replace"))
    sys.stdout.flush()
    for row in ((_read_json(evidence / "classification.json") or {}).get("failed") or [])[:10]:
        print(f"::error title=Post-merge red on {SYSTEMS[args.os]}::{row['id']} ({row['why']})")
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
    step.add_argument("--os", choices=sorted(SYSTEMS), default="linux", help="the leg's operating system")
    step = commands.add_parser("summary", help="write the job summary; exit 1 when red")
    step.add_argument("--evidence-dir", type=Path, required=True)
    step.add_argument("--os", choices=sorted(SYSTEMS), default="linux", help="the leg's operating system")
    args = parser.parse_args(argv)
    if args.command == "classify" and args.size and not args.base:
        parser.error("classify --size measures against the check's base: pass --base")
    return {"base": command_base, "growth": command_growth, "classify": command_classify,
            "attribute": command_attribute, "route": command_route, "summary": command_summary}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
