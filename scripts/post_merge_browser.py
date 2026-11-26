#!/usr/bin/env python3
"""The post-merge browser check (pilot): judge the head session, compare its reds with the window's base, hand over.

    python scripts/post_merge_browser.py compare --evidence-dir DIR --plan plan.json
    python scripts/post_merge_browser.py recheck --evidence-dir DIR --plan plan.json
    python scripts/post_merge_browser.py outcome --evidence-dir DIR --plan plan.json
    python scripts/post_merge_browser.py verdict --evidence-dir DIR
    python3 scripts/post_merge_browser.py route --run-url URL [--dry-run]     (input: the environment's ROUTING)
    python -m pytest -p scripts.post_merge_browser ...                       (records the permitted skips)

.github/workflows/post-merge-browser.yml runs these. post-merge.yml's window job starts that workflow for its frozen
window when the window touches the interface; it holds no post-merge concurrency group, changes no post-merge run,
leg, verdict or base, and ouroboros/update_qualification.py reads post-merge.yml's runs only. Its `check` job runs the
plan of scripts/post_merge_browser_select.py once through scripts/ci_retry.py (DIR/head: a failed test runs once
more), and then:

- COMPARE: the confirmed reds run once at the window's BASE in a worktree that keeps the historical fixture off disk
  where its blob differs from the head's (post_merge_check `add_worktree`, `comparable`), through post_merge_check
  `recheck_ids`: their modules are collected whole and only the listed tests run (`--ci-listed`), so an id the base
  lacks (a new test, a renamed one or a changed parameter) is absent and stops no other. `recheck_ids` counts a
  module that did not collect, or a test still running, as failed for each id inside it; here a base failure needs a
  failed setup, call or teardown report of that test, and anything else is inconclusive (unknown). Where uv.lock
  differs between base and head the base would run in the head's environment: nothing is compared.
- RECHECK: the tests of the open browser issues that this run neither saw fail nor passed whole run once at the head,
  the same way, within RECHECK_SECONDS.
- OUTCOME (`judge`): GREEN when every requested node was collected, completed setup, call and teardown and passed,
  both sentinel cases included (FLAKY: passed on its retry, listed, never handed over); RED for a test with two
  completed failure observations; CHECKED PARTIALLY, each item named with its cause, when anything requested did not
  run or is not explained: no plan or no verdict of scripts/ci_retry.py, a changed or same-named module (tier 1 or 2)
  the budget or the cap dropped, a requested argument or collector skipped or broken at collection or never
  collected, a selected test that never ran, a skip that tests/browser_lane.py `permitted_skip` does not allow (this
  module, loaded as a plugin by the head session, records which skips it allowed and moves the sentinel cases to the
  front), a failure without a second completed observation (a retry that skipped or did not finish: the first
  failure is shown, nothing is handed over), a sentinel case the head no longer collects or completes, or a red
  verdict of ci_retry that no test explains. Red and partial can hold together. What the lane excludes (the plan's
  `excluded`: Caddy modules, tests `-m ui_browser` deselects) is named, neither green evidence nor partial; dropped
  tier-3 modules and hub files are named only.
- ATTRIBUTION, the owner's rule (2026-11-25: name the author only when the comparison with the base confirms it, else
  "not attributed"): a merge is named only when the window holds exactly one first-parent commit and the red passed
  whole at the base (setup, call and teardown). A test absent at the base is not attributed ("not present at the
  base"); one that failed there is pre-existing; an inconclusive base answer is unknown. Unattributed reds list their
  candidate merges: the window's commits that touched the files that selected the test (its module, the changed
  files that chose it, a sentinel's guarded files), else all of them; the cause may lie elsewhere.

The `route` job runs no test: scripts/post_merge_routing.py `route` with the system `browser`, so its issues are
`post-merge-red` issues of their own system, each red with its own base answer and candidates, and a culprit's pull
request gets one comment. Off the `ouroboros` branch (a trial) it holds a read-only token and DryGitHub records the
writes it would make. It retires (narrows or closes) issues only while this run's head is still the branch tip. A
refused write or a red no open issue carries fails it, named in its summary; the check job's outputs stay with the run,
so re-running the route job alone offers that hand-over again.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zlib

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import post_merge_browser_select as selection  # noqa: E402
from scripts import post_merge_check as pmc  # noqa: E402
from scripts import post_merge_routing as routing  # noqa: E402
from scripts.post_merge_bisect import comparable, probe_files, protected_projection  # noqa: E402
from tests.ci_verdicts import observed_outcomes  # noqa: E402

SYSTEM = "browser"
TEST_SECONDS = 900  # the UI lane's per-test bound (ui-browser.yml)
COMPARE_SECONDS = 240  # the base session starts no test after this; post-merge-browser.yml's compare step allows more
RECHECK_SECONDS = 180  # the open issues' recheck at the head; its step allows a test begun at the edge to finish
PASSING = ("green", "flaky")
SKIPS = "permitted-skips.json"  # beside each session's projection: the skips tests/browser_lane.py allows
ROUTING_LIMIT = 100_000  # the encoded hand-over travels as a job output and an environment variable
STEPS = ("install", "run", "compare", "recheck")  # the check job passes each step's outcome as BROWSER_<STEP>_OUTCOME
RETIRE_NOTE = "An older rerun cannot retire this branch’s open failures."  # route()'s own note, no refused write


# --- the plugin: which skips the lane permits, decided while the report still carries its reason ---------------------

_PERMITTED: dict[str, str] = {}


def pytest_runtest_logreport(report):
    if not report.skipped or hasattr(report, "wasxfail"):
        return
    from tests.browser_lane import PLATFORM_SKIPS, _skip_reason, permitted_skip

    reason = _skip_reason(report)
    if permitted_skip(report.nodeid, reason):  # recorded by the reviewed entry, never the free text of the reason
        _PERMITTED[report.nodeid] = next(pattern for prefix, pattern in PLATFORM_SKIPS
                                         if report.nodeid.startswith(prefix) and re.search(pattern, reason))


try:  # the head session loads this module as a plugin; the route job runs it on a bare interpreter
    from pytest import hookimpl as _hookimpl
except ImportError:  # pragma: no cover - no pytest, no session to order
    def _hookimpl(**_options):
        return lambda function: function


@_hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """The two sentinel cases first, whichever argument collected them: pytest keeps a module's own order where
    arguments overlap, and a session stopped at its budget keeps the head of the plan."""
    first = [scenario for scenario, _ in selection.SENTINELS]
    items.sort(key=lambda item: first.index(item.nodeid) if item.nodeid in first else len(first))


def pytest_sessionfinish(session):
    config = session.config
    directory = getattr(config.option, "ci_evidence_dir", None)
    if directory and not hasattr(config, "workerinput"):
        _write(Path(directory) / SKIPS, {"format": 1, "permitted": dict(sorted(_PERMITTED.items()))})


# --- evidence ---------------------------------------------------------------------------------------------------

def _read(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _completed(attempt: dict) -> bool:
    """Whether a failed attempt of verdict.json is a completed failure observation: not a skip on the retry, not a
    test still running when its session was killed, not a collection error."""
    return attempt.get("error_type") not in ("Skipped", "InFlight") and attempt.get("phase") != "collect"


def _first(attempt: dict) -> str:
    return f"{attempt.get('phase', '?')}: {attempt.get('error_type', '?')}"


def _rounds(directory: Path) -> list[Path]:
    """scripts/ci_retry.py's retry rounds of a pass, in order."""
    rounds = [path for path in Path(directory).glob("retry-*") if path.name[len("retry-"):].isdigit()]
    return sorted(rounds, key=lambda path: int(path.name[len("retry-"):]))


def _once(attempt: dict, reason: str) -> str:
    """Why a failure has no second observation, from its one attempt in verdict.json."""
    if attempt.get("phase") == "collect":
        return "did not collect"
    if attempt.get("error_type") == "InFlight":
        return "was running when its session was killed, so it got no retry"
    return "failed once and got no retry" + (f" ({reason})" if reason else "")


def judge(evidence: Path, plan: dict) -> dict:
    """The outcome of DIR/head: its first session, ci_retry's rounds and verdict.json, against the plan."""
    result = {"outcome": "partial", "partial": [], "red": [], "once": {}, "failed_then_skipped": [], "flaky": [],
              "accepted": [], "not_run": [], "skipped": [], "missing_sentinels": [], "dropped": [],
              "collection_skipped": [], "not_collected": {}, "passed": 0, "planned": None, "verdict": None,
              "reason": ""}
    if not plan:
        result["partial"].append("the job stopped before it chose the tests: nothing ran")
        return result
    if not plan.get("applicable"):
        result["partial"].append(f"the check's plan has nothing to run: {plan.get('reason', '')}")
        return result
    head = Path(evidence) / "head"
    verdict, first = _read(head / "verdict.json"), pmc.read_session(head)
    sessions = [first, *(pmc.read_session(path) for path in _rounds(head))]
    ran: dict[str, dict] = {}
    for session in sessions:
        for node, observed in observed_outcomes(session["rows"]).items():
            if ran.get(node, {}).get("outcome") != "failed":
                ran[node] = observed
    permitted = {}
    for directory in (head, *_rounds(head)):
        permitted.update((_read(directory / SKIPS) or {}).get("permitted") or {})
    planned = first["selected"]
    result["planned"] = len(planned) if planned is not None else None
    if not isinstance(verdict, dict):
        verdict = {}
        result["partial"].append("scripts/ci_retry.py left no verdict: the run step did not finish")
    result.update(verdict=verdict.get("verdict"), reason=verdict.get("reason") or "")
    if planned is None:
        result["partial"].append("the session left no complete export: the tests it never started are unknown")
    elif not planned:
        result["partial"].append("the session selected no test of the plan")
    result["accepted"] = sorted(row["nodeid"] for row in verdict.get("accepted") or [])
    result["flaky"] = sorted(row["nodeid"] for row in verdict.get("flaky") or [])
    for row in verdict.get("failed") or []:
        node, attempts = row["nodeid"], row.get("attempts") or [{}]
        if node in result["accepted"]:
            continue
        if len(attempts) < 2:
            result["once"][node] = _once(attempts[0], result["reason"])
        elif all(_completed(attempt) for attempt in attempts[:2]):
            result["red"].append(node)  # two completed failure observations
        elif attempts[1].get("error_type") == "Skipped" and _completed(attempts[0]):
            result["failed_then_skipped"].append(node)
        else:  # a retry, or a first attempt, that never finished: the first failure stays visible, nothing is routed
            result["once"][node] = (f"failed ({_first(attempts[0])}); its retry did not finish ({_first(attempts[1])})"
                                    if _completed(attempts[0]) else _once(attempts[0], result["reason"]))
    if not verdict:  # no verdict: a failure the session saw is reported, never handed over
        result["once"] = {node: "failed; its retry never ran" for node, observed in ran.items()
                          if observed["outcome"] == "failed"}
    seen = set(result["red"]) | set(result["once"]) | set(result["failed_then_skipped"])
    result["skipped"] = sorted(node for node, observed in ran.items() if observed["outcome"] == "skipped"
                               and node not in permitted and node not in seen)
    if planned is not None:
        result["not_run"] = sorted((set(planned) - set(ran)) | set(verdict.get("not_run") or []))
    # What was requested and never became a selected test: a collector skipped or broken at collection, or an
    # argument that collected nothing (a module the plan asked for whole, one changed test).
    skips, broken = first.get("collection_skips") or [], first.get("collection") or []
    result["collection_skipped"] = sorted(skips)
    for node in broken:
        result["once"].setdefault(node, "did not collect")
    explained = set(skips) | set(broken)
    for arg in plan.get("args") or [] if planned is not None else []:
        if any(pmc._inside(node, arg) for node in planned):
            continue
        if not any(pmc._inside(node, arg) or pmc._inside(arg, node) for node in explained):
            result["not_collected"][arg] = "the session collected no test it names"
    result["dropped"] = [f"{row['module']}: {row.get('why', 'dropped')}" for row in plan.get("dropped") or []
                         if row.get("tier", 3) <= 2]
    for row in plan.get("sentinels") or []:
        scenario = row["scenario"]
        if not row.get("present"):
            result["missing_sentinels"].append(f"{scenario}: the head no longer defines {row['test']}")
        elif planned is not None and scenario not in planned:
            result["missing_sentinels"].append(f"{scenario}: the session did not collect this case")
        elif ran.get(scenario, {}).get("outcome") not in ("passed", "failed") and scenario not in permitted:
            result["missing_sentinels"].append(f"{scenario}: it did not complete")
    result["passed"] = sum(observed["outcome"] == "passed" for node, observed in ran.items()
                           if node not in seen and node not in result["flaky"])
    for key, label in (("dropped", "a changed or same-named browser module was not checked (budget or cap)"),
                       ("collection_skipped", "a requested collector was skipped at collection"),
                       ("not_collected", "a requested test was never collected"),
                       ("once", "a failure got no second completed observation"),
                       ("failed_then_skipped", "a failure skipped on its retry"),
                       ("skipped", "a test skipped outside the lane's permitted skips"),
                       ("not_run", "a selected test never ran"), ("missing_sentinels", "a sentinel was not checked")):
        if result[key]:
            result["partial"].append(label)
    explained = (result["red"] or result["once"] or result["failed_then_skipped"] or result["not_run"]
                 or result["skipped"] or result["collection_skipped"] or result["not_collected"])
    if verdict and verdict.get("verdict") not in ("green", "flaky", "accepted") and not explained:
        result["partial"].append("scripts/ci_retry.py says " + str(verdict.get("verdict")) + " and no test explains it"
                                 + (f": {result['reason']}" if result["reason"] else ""))
    result["outcome"] = ("red" if result["red"] else "partial" if result["partial"] else
                         "flaky" if result["flaky"] else "green")
    return result


# --- running tests ----------------------------------------------------------------------------------------------

def run_session(tree: Path, args, directory: Path, *, pytest=None) -> int:
    """One browser session of `args` in the checkout `tree`, by its own launcher (the UI lane's invocation,
    ui-browser.yml), without the job's token; its public projection in `directory`, the launcher's raw evidence
    beside the uploaded root (`<root>-raw/<directory name>`). `pytest` replaces the launcher (a test's own
    interpreter)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    raw = directory.parent.parent / f"{directory.parent.name}-raw" / directory.name
    launcher = pytest or ["python", "-I", "-S", str(Path(tree) / "scripts/safe_test.py"), "--evidence-dir", str(raw),
                          "--", "python", "-m", "pytest"]
    command = [*launcher, *args, "-m", "ui_browser", "-vv", "--tb=short", f"--timeout={TEST_SECONDS}",
               f"--ci-evidence-dir={directory}"]
    env = {key: value for key, value in os.environ.items() if key not in ("GH_TOKEN", "GITHUB_TOKEN")}  # as ci_retry
    try:
        return subprocess.call(command, cwd=str(tree), env=env)
    except OSError as error:
        print(f"post-merge browser: the session did not start: {error}", flush=True)
        return 127


def compare(evidence: Path, plan: dict, red: list[str], *, run=None) -> dict:
    """compare.json: each confirmed red at the window's base, {id: passed|failed|absent|skipped|xfailed|unknown|error};
    `inconclusive` names why a `recheck_ids` failure became unknown (no failed report of that test at the base)."""
    base, head = plan.get("base") or "", plan.get("head") or ""
    record = {"format": 1, "base": base, "head": head, "ids": sorted(red), "results": {},
              "dependencies_differ": False, "excluded": {}, "inconclusive": {}}
    if not red:
        return record
    if pmc._blob(base, "uv.lock") != pmc._blob(head, "uv.lock"):  # as post_merge_check's Prober records it
        record["dependencies_differ"] = True
        return record
    protection = protected_projection(head, base, pmc._git)
    kept, excluded, _ = comparable(probe_files(red, routing.SIZE_PREFIX), revision=base, protection=protection,
                                   git=pmc._git)
    record["excluded"] = excluded
    record["results"].update(dict.fromkeys(excluded, "error"))
    if not kept:
        return record
    work = Path(tempfile.mkdtemp(prefix="post-merge-browser-base-", dir=os.environ.get("RUNNER_TEMP") or None))
    tree = work / "base"
    try:
        pmc.add_worktree(tree, base, omit_protected=bool(protection[1]))
        if pmc._git("rev-parse", "HEAD", cwd=tree) != base:
            raise OSError(f"the worktree is not at the window's base {base}")
        record["results"].update(pmc.recheck_ids(
            kept, Path(evidence) / "base", head=base, read_source=pmc.tree_source(base), budget=COMPARE_SECONDS,
            run=run or (lambda args, directory: run_session(tree, args, directory))))
        record["inconclusive"] = observed_failures(Path(evidence) / "base", record["results"])
        record["results"].update(dict.fromkeys(record["inconclusive"], "unknown"))
    except (OSError, subprocess.SubprocessError) as error:
        record["error"] = f"the base did not run: {error}"
        record["results"].update({node: "unknown" for node in kept if node not in record["results"]})
    finally:
        if tree.exists():
            pmc.remove_worktree(tree)
        shutil.rmtree(work, ignore_errors=True)
    return record


def observed_failures(directory: Path, results: dict) -> dict[str, str]:
    """{id: why} for each `failed` answer of the base session that no failed setup, call or teardown report of that
    test shows: its module or collector did not collect, or it was still running when the session ended. Such an
    answer says nothing about the test, so it is no proof that the red was there before the window."""
    session = pmc.read_session(directory)
    failed = {row["nodeid"] for row in session["rows"]
              if row["outcome"] == "failed" and row["phase"] in ("setup", "call", "teardown")}
    found = {}
    for node, answer in results.items():
        if answer != "failed" or any(pmc._inside(other, node) for other in failed):
            continue
        broken = any(pmc._inside(node, other) or pmc._inside(other, node) for other in session["collection"])
        found[node] = "did not collect at the base" if broken else "did not finish at the base"
    return found


def same_run_passes(directory: Path, ids) -> dict[str, str]:
    """The ids the head session ran whole (setup, call and teardown passed, nothing else) in a session that tells its
    tests apart: what this run settles of an open issue without a recheck."""
    session = pmc.read_session(directory)
    if not session["final"] or not pmc._tells_tests_apart(session):
        return {}
    whole, spoiled = pmc._ran_whole(session)
    shared = pmc._shared(session)
    return {node: "passed" for node in ids if node in whole and node not in spoiled
            and not any(pmc._inside(other, node) for other in shared)}


def recheck(evidence: Path, plan: dict, judged: dict, github, *, run=None) -> dict:
    """recheck.json: the open browser issues' tests, settled by this run's whole passes or run once at the head."""
    record = {"format": 1, "head": plan.get("head"), "listed": False, "issues": [], "same_run": {}, "rechecked": {}}
    try:
        issues = [issue for issue in routing.open_issues(github) if routing.system_of(issue["block"]) == SYSTEM]
    except Exception as error:  # noqa: BLE001 -- an unread list retires nothing
        record["warning"] = f"the open browser issues did not read: {routing._reason(error)}"
        return record
    seen = set(judged["red"]) | set(judged["once"]) | set(judged["failed_then_skipped"])  # fails here: no pass settles
    ids = sorted({node for issue in issues for node in issue["block"]["ids"]} - seen)
    record.update(listed=True, issues=sorted(issue["number"] for issue in issues))
    record["same_run"] = same_run_passes(Path(evidence) / "head", ids)
    rest = [node for node in ids if node not in record["same_run"]]
    if rest:
        record["rechecked"] = pmc.recheck_ids(
            rest, Path(evidence) / "recheck", head=plan.get("head") or "", read_source=pmc.working_source,
            budget=RECHECK_SECONDS, run=run or (lambda args, directory: run_session(REPO, args, directory)))
    return record


# --- attribution ------------------------------------------------------------------------------------------------

def relevant_files(plan: dict, node: str) -> set[str]:
    module = node.split("::", 1)[0]
    files = {module}
    for row in plan.get("selected") or []:
        if row["module"] == module:
            files.update(row["files"])
    for row in plan.get("sentinels") or []:
        if row["scenario"].split("::", 1)[0] == module:
            files.update(row["guards"])
    return files


def attribute(plan: dict, red: list[str], compared: dict) -> list[dict]:
    """route()'s verdicts by the owner's rule: a culprit only for a one-merge window whose base ran the red whole and
    green; pre-existing only where the base showed that test fail; else unattributed with its candidate merges. Each
    verdict keeps its own window base (`window`), so an issue that gains tests from a later window keeps both."""
    commits, results = plan.get("commits") or [], compared.get("results") or {}
    inconclusive, base = compared.get("inconclusive") or {}, plan.get("base") or ""
    verdicts = []
    for node in red:
        files = relevant_files(plan, node)
        candidates = [c["sha"] for c in commits if files & set(c["files"])] or [c["sha"] for c in commits]
        answer = results.get(node, "unknown")
        if compared.get("dependencies_differ"):
            verdicts.append({"id": node, "verdict": "unattributed", "at_base": "dependencies", "candidates": candidates,
                             "window": base, "why": "uv.lock differs between the window's base and head: the base "
                                                    "was not run"})
        elif answer == "failed":
            verdicts.append({"id": node, "verdict": "pre-existing", "at_base": "failed", "window": base})
        elif answer == "passed" and len(commits) == 1:
            verdicts.append({"id": node, "verdict": "culprit", "culprit": commits[0]["sha"], "last_good": base,
                             "at_base": answer, "window": base})
        else:
            at_base = answer if answer in ("passed", "absent", "error") else "unknown"
            why = (f"the window holds {len(commits)} merges and the base does not tell them apart" if at_base == "passed"
                   else "not present at the base" if at_base == "absent"
                   else f"its answer at the base was inconclusive: {inconclusive[node]}" if node in inconclusive
                   else f"its answer at the base was {answer}")
            verdicts.append({"id": node, "verdict": "unattributed", "at_base": at_base, "candidates": candidates,
                             "window": base, "why": why})
    return verdicts


# --- the hand-over ----------------------------------------------------------------------------------------------

class DryGitHub(routing.GitHub):
    """Reads GitHub; records each write it would make instead of making it."""

    def __init__(self, repo: str, token: str):
        super().__init__(repo, token)
        self.writes = []
        self._number = 900000

    def _call(self, method: str, url: str, body=None):
        if method == "GET":
            return super()._call(method, url, body)
        self.writes.append({"method": method, "url": url, "body": body})
        self._number += 1
        return {"number": self._number, "html_url": "", "state": "open", "title": (body or {}).get("title", ""),
                "body": (body or {}).get("body", ""), "labels": [{"name": routing.LABEL}]}, ""


def encode(value) -> str:
    return base64.b64encode(zlib.compress(json.dumps(value, sort_keys=True).encode("utf-8"), 9)).decode("ascii")


def decode(text: str):
    try:
        return json.loads(zlib.decompress(base64.b64decode(text.encode("ascii"), validate=True)).decode("utf-8"))
    except (ValueError, zlib.error, UnicodeError):
        return None


def _ancestor(older: str, newer: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", older, newer], cwd=REPO,
                          capture_output=True).returncode == 0


def hand_over(data: dict, github, *, run_url: str, retire: bool, origin=None, ancestor=_ancestor) -> dict:
    same, rechecked = data.get("same_run") or {}, data.get("rechecked") or {}
    return routing.route(
        github, classification={"failed": [{"id": node} for node in data.get("red") or []],
                                "flaky": [{"id": node, "retry": "passed"} for node in data.get("flaky") or []],
                                "base": data.get("base")},
        attribution={"verdicts": data.get("verdicts") or []}, head=data["head"], run_url=run_url,
        recheck=lambda ids: {node: rechecked.get(node, "unknown") for node in ids},
        same_run=lambda ids: {node: same[node] for node in ids if node in same},
        system=SYSTEM, owed=(), retire=retire, ancestor=ancestor, origin=origin)


def route_lines(data: dict, record: dict, *, dry: bool, notes: list[str], writes=None) -> list[str]:
    refused = [warning for warning in record.get("warnings") or [] if warning != RETIRE_NOTE]
    left = record.get("untracked") or []
    lines = ["## Post-merge browser hand-over" + (" (dry: a trial off `ouroboros`)" if dry else ""), ""]
    lines.append(f"Created {record.get('created') or 'no'} issue(s), updated {record.get('updated') or 'none'}, "
                 f"commented on {record.get('comments') or 'no'} pull request(s), narrowed "
                 f"{[row['issue'] for row in record.get('narrowed') or []] or 'none'}, closed "
                 f"{record.get('closed') or 'none'}.")
    lines += [f"- {note}" for note in notes]
    if left:
        lines += ["", f"Not handed over ({len(left)}): no open issue carries them."] + [f"- `{node}`" for node in left]
    if refused:
        lines += ["", "Refused or failed GitHub calls:"] + [f"- {warning}" for warning in refused]
    if left or refused:
        lines += ["", "The check job's outputs stay with this run: re-running this job alone offers the hand-over "
                      "again (`gh run rerun <run id> --job <this job's id>`)."]
    if dry:
        lines += ["", f"Dry routing: {len(writes or [])} GitHub write(s) recorded, none made."]
    return lines


def command_route(args, environ=os.environ) -> int:
    data = decode(environ.get("ROUTING", ""))
    dry = args.dry_run or environ.get("GITHUB_REF") != f"refs/heads/{routing.BRANCH}"
    if not isinstance(data, dict) or not data.get("head"):
        lines = ["## Post-merge browser hand-over", "", "No hand-over input: the check job ended before writing it, or "
                 "it exceeded the job output's limit (its evidence keeps routing-input.json). Nothing was handed over."]
        _summary(environ, lines)
        return 1
    github = (DryGitHub if dry else routing.GitHub)(environ.get("GITHUB_REPOSITORY", ""), environ.get("GH_TOKEN", ""))
    notes = []
    try:
        retire = github.get(f"git/ref/heads/{routing.BRANCH}")["object"]["sha"] == data["head"]
        if not retire:
            notes.append(f"`{routing.BRANCH}` has moved past `{data['head'][:12]}`: this run retires no issue.")
    except Exception as error:  # noqa: BLE001 -- an unconfirmed tip retires nothing
        retire = False
        notes.append(f"The branch tip did not read ({routing._reason(error)}): this run retires no issue.")
    if not data.get("recheck_listed"):
        notes.append("The check could not list the open browser issues: none of their tests was rechecked.")
    run, attempt = (int(value) if str(value).isdigit() else None
                    for value in (environ.get("GITHUB_RUN_ID"), environ.get("GITHUB_RUN_ATTEMPT")))
    record = hand_over(data, github, run_url=args.run_url, retire=retire,
                       origin={"head": data["head"], "run": run, "attempt": attempt, "base": data.get("base")})
    writes = getattr(github, "writes", None)
    _write(Path(args.out), {"dry_run": dry, "retire": retire, "notes": notes, "record": record, "writes": writes})
    _summary(environ, route_lines(data, record, dry=dry, notes=notes, writes=writes))
    refused = [warning for warning in record["warnings"] if warning != RETIRE_NOTE]
    return 1 if refused or record.get("untracked") else 0


# --- the outcome ------------------------------------------------------------------------------------------------

def _shown(nodes, limit: int = 40) -> list[str]:
    nodes = list(nodes)
    rows = [f"- `{node}`" for node in nodes[:limit]]
    return rows + ([f"- … and {len(nodes) - limit} more (outcome.json in the evidence)"] if len(nodes) > limit else [])


def summary_lines(plan: dict, judged: dict, verdicts: list[dict], compared: dict, rechecked: dict,
                  steps: dict | None = None) -> list[str]:
    title = {"green": "GREEN", "flaky": "GREEN with FLAKY tests", "red": "RED",
             "partial": "CHECKED PARTIALLY"}[judged["outcome"]]
    if judged["outcome"] == "red" and judged["partial"]:
        title += ", AND CHECKED PARTIALLY"
    lines = [f"## Post-merge browser check (pilot): {title}", "",
             f"Window `{(plan.get('base') or '')[:12]}..{(plan.get('head') or '')[:12]}` "
             f"({len(plan.get('commits') or [])} merge(s)): {plan.get('reason', '')}."]
    sentinels = plan.get("sentinels") or []
    lines += ["", f"Selected {len(plan.get('selected') or [])} module(s) and "
                  f"{sum(bool(row.get('present')) for row in sentinels)} WebKit sentinel test(s), estimated "
                  f"{round((plan.get('estimate_ms') or 0) / 60000, 1)} min; "
                  f"{judged['planned'] if judged['planned'] is not None else 'unknown'} node(s) selected, "
                  f"{judged['passed']} passed."]
    for row in plan.get("selected") or []:
        lines.append(f"- `{row['module']}`: {'; '.join(sorted(set(row['reasons'])))}")
    if plan.get("shared"):
        lines += ["", "Shared browser infrastructure changed (" + ", ".join(f"`{path}`" for path in plan["shared"])
                  + "): the sentinels run whatever else is selected."]
    for key, label in (("dropped", "Not selected (budget or cap)"), ("wide", "Hub files left to the release lane"),
                       ("excluded", "Excluded by the lane (neither checked nor partial)")):
        if plan.get(key):
            lines += ["", f"{label}:"] + [f"- `{row.get('module') or row.get('file')}`: "
                                          f"{row.get('why') or str(row.get('modules')) + ' modules name it'}"
                                          for row in plan[key]]
    if judged["partial"]:
        lines += ["", "Checked partially because:"] + [f"- {reason}" for reason in judged["partial"]]
    for key, label in (("dropped", "Not checked: changed or same-named modules over the budget or cap"),
                       ("collection_skipped", "Not checked: skipped at collection")):
        if judged.get(key):
            lines += ["", f"{label}:"] + [f"- `{node}`" if key != "dropped" else f"- {node}" for node in judged[key]]
    if judged.get("not_collected"):
        lines += ["", "Not checked: requested, never collected:"] + [
            f"- `{arg}`: {cause}" for arg, cause in sorted(judged["not_collected"].items())]
    unfinished = {step: outcome for step, outcome in (steps or {}).items() if outcome not in ("success", "skipped", "")}
    if unfinished:
        lines += ["", "Steps that did not finish: " + ", ".join(f"{step} ({outcome})" for step, outcome in
                                                               sorted(unfinished.items())) + "."]
    notes = {row["id"]: row for row in verdicts}
    if judged["red"]:
        lines += ["", "FAILED (twice):"]
        for node in judged["red"]:
            verdict = notes.get(node, {})
            if verdict.get("verdict") == "culprit":
                note = f" — introduced by `{verdict['culprit'][:12]}` (passed whole at the base)"
            elif verdict.get("verdict") == "pre-existing":
                note = " — already red at the base: not attributed"
            else:
                note = (f" — not attributed ({verdict.get('why', 'no base comparison')}); candidate merges (they "
                        "touched the files that selected it; the cause may lie elsewhere): "
                        + ", ".join(f"`{sha[:12]}`" for sha in verdict.get("candidates") or []))
            lines.append(f"- `{node}`{note}")
    if judged["once"]:
        lines += ["", "Failed, not confirmed (reported, never handed over):"] + [
            f"- `{node}`: {cause}" for node, cause in sorted(judged["once"].items())]
    for key, label in (("failed_then_skipped", "Failed, then skipped on its retry (never handed over)"),
                       ("flaky", "FLAKY (passed on its retry, never handed over)"),
                       ("accepted", "ACCEPTED (tests/accepted_failures.json)"),
                       ("skipped", "Skipped outside the lane's permitted skips"),
                       ("missing_sentinels", "Sentinels not checked")):
        if judged[key]:
            lines += ["", f"{label}:"] + [f"- `{node}`" if key != "missing_sentinels" else f"- {node}"
                                          for node in judged[key]]
    if judged["not_run"]:
        lines += ["", f"Not executed ({len(judged['not_run'])}):"] + _shown(judged["not_run"])
    if compared.get("dependencies_differ"):
        lines += ["", "uv.lock differs between the window's base and head: no red was run at the base."]
    elif compared.get("error"):
        lines += ["", f"Base comparison: {compared['error']}."]
    if rechecked.get("warning"):
        lines += ["", f"Open issues: {rechecked['warning']}."]
    elif rechecked.get("listed"):
        lines += ["", f"Open browser issues {rechecked.get('issues') or 'none'}: {len(rechecked.get('same_run') or {})} "
                      f"test(s) passed whole in this run, {len(rechecked.get('rechecked') or {})} rechecked."]
    return lines


def _summary(environ, lines) -> None:
    text = "\n".join(lines) + "\n"
    print(text)
    if environ.get("GITHUB_STEP_SUMMARY"):
        with open(environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as handle:
            handle.write(text)


def _output(environ, **values) -> None:
    if environ.get("GITHUB_OUTPUT"):
        with open(environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as handle:
            handle.writelines(f"{name}={value}\n" for name, value in values.items())


def command_compare(args, environ=os.environ) -> int:
    evidence, plan = Path(args.evidence_dir), _read(Path(args.plan)) or {}
    _write(evidence / "compare.json", compare(evidence, plan, judge(evidence, plan)["red"]))
    return 0


def command_recheck(args, environ=os.environ) -> int:
    evidence, plan = Path(args.evidence_dir), _read(Path(args.plan)) or {}
    github = routing.GitHub(environ.get("GITHUB_REPOSITORY", ""), environ.get("GH_TOKEN", ""))
    _write(evidence / "recheck.json", recheck(evidence, plan, judge(evidence, plan), github))
    return 0


def command_outcome(args, environ=os.environ) -> int:
    """outcome.json and routing-input.json, the job's outputs and summary; never fails (the `verdict` step does)."""
    evidence, plan = Path(args.evidence_dir), _read(Path(args.plan)) or {}
    judged = judge(evidence, plan)
    compared, rechecked = _read(evidence / "compare.json") or {}, _read(evidence / "recheck.json") or {}
    verdicts = attribute(plan, judged["red"], compared)
    data = {"format": 1, "base": plan.get("base"), "head": plan.get("head"), "outcome": judged["outcome"],
            "red": judged["red"], "flaky": judged["flaky"], "verdicts": verdicts,
            "same_run": rechecked.get("same_run") or {}, "rechecked": rechecked.get("rechecked") or {},
            "recheck_listed": bool(rechecked.get("listed"))}
    _write(evidence / "outcome.json", {**judged, "verdicts": verdicts})
    _write(evidence / "routing-input.json", data)
    encoded = encode(data)
    if len(encoded) > ROUTING_LIMIT:
        print(f"::error::the hand-over input takes {len(encoded)} characters, over {ROUTING_LIMIT}: routing-input.json "
              "in the evidence keeps it, and the route job hands nothing over")
        encoded = ""
    _output(environ, outcome=judged["outcome"], routing=encoded)
    steps = {step: environ.get(f"BROWSER_{step.upper()}_OUTCOME", "") for step in STEPS}
    _summary(environ, summary_lines(plan, judged, verdicts, compared, rechecked, steps))
    return 0


def command_verdict(args, environ=os.environ) -> int:
    """The check job's own conclusion: green only for GREEN or FLAKY."""
    judged = _read(Path(args.evidence_dir) / "outcome.json") or {}
    print(f"post-merge browser check: {judged.get('outcome', 'no outcome')}")
    return 0 if judged.get("outcome") in PASSING else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="post_merge_browser.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("compare", "recheck", "outcome", "verdict"):
        command = commands.add_parser(name)
        command.add_argument("--evidence-dir", required=True)
        if name != "verdict":
            command.add_argument("--plan", required=True)
    route = commands.add_parser("route")
    route.add_argument("--run-url", required=True)
    route.add_argument("--out", default="routing.json", help="the hand-over record")
    route.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    return {"compare": command_compare, "recheck": command_recheck, "outcome": command_outcome,
            "verdict": command_verdict, "route": command_route}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
