#!/usr/bin/env python3
"""The post-merge browser check (pilot): judge the selected run, compare its reds with the base, hand them over.

    python scripts/post_merge_browser.py legs    --repo OWNER/NAME --run-id ID --attempt N
    python scripts/post_merge_browser.py compare --evidence-dir DIR --plan plan.json --base-tree DIR
    python scripts/post_merge_browser.py route   --evidence-dir DIR --plan plan.json --run-url URL [--dry-run]
    python scripts/post_merge_browser.py summary --evidence-dir DIR --plan plan.json
    python -m pytest -p scripts.post_merge_browser --browser-stop-file DIR/stop ...

post-merge.yml's `browser` job runs the plan of scripts/post_merge_browser_select.py once through
scripts/ci_retry.py (DIR/head: a red test runs once more; failed then passed is FLAKY) and then these steps:

- THE LEGS BOUND THE CHECK. The run's concurrency group holds the next window until every job of the run ends, so
  this job must not outlast the Linux and Windows legs. While the tests run, `legs` reads this attempt's job list
  each minute; once both legs have ended it writes DIR/stop, and this module, loaded as a pytest plugin, ends the
  session between two tests (a session that starts after it runs nothing). What never ran is named: the check
  was partial. The plan's budget and `--session-timeout` bound it from the other side.

- OUTCOME (`judge`). GREEN only when every node the plan's session selected ran and passed (FLAKY: passed on its
  retry, reported, never handed over). PARTIAL ("checked partially") when a selected node never ran — the
  session's time budget (`--session-timeout`) ran out, a step stopped, a test skipped outside the lane's reviewed
  platform skips (tests/browser_lane.py PLATFORM_SKIPS) — or the session left no evidence; the summary names
  every unexecuted node, and a red that got no retry because the budget ended is reported, never handed over.
  RED when a confirmed red (failed twice) exists; partial too, if something did not run. NOT APPLICABLE when the
  plan does not apply. Only GREEN and FLAKY end the job green. A step that did not finish (the browser install,
  the run, the base comparison, the hand-over) is named in the summary.
- ATTRIBUTION (`compare`, then `route`), by comparison with the window's base, never by bisection: each
  confirmed red runs again (with one retry) in a worktree of BASE. Red there too: PRE-EXISTING, attributed to no
  merge. Passing there, or its test absent at BASE: INTRODUCED in the window. Its merge is named only when exactly
  one first-parent commit of the window touched the files that selected it (its module, the changed files behind
  its selection, a sentinel's guarded files); otherwise those commits (or, when none did, every commit of the
  window) are listed as CANDIDATES and the red stays unattributed. A red that only fails once at BASE, or that
  BASE could not run, is unattributed. No browser test runs in a bisection.
- HAND-OVER (`route`): scripts/post_merge_routing.py `route` with the system `browser`, so the issues are
  `post-merge-red` issues of their own system — never a Linux or Windows leg's — and a culprit's pull request gets
  the same one comment. The open browser issues' tests this run did not settle run once more at the head, unless
  the legs have already ended (the issues then wait for a later window).
  `--dry-run` (forced off the `ouroboros` branch) reads GitHub but writes nothing: it records the calls it would
  make in routing.json.

The job is continue-on-error in post-merge.yml: its red never becomes the run's conclusion, which every
install's update letter reads (ouroboros/update_qualification.py), and the window's base, the Linux and Windows
legs and their issues ignore it. Stdlib only apart from what tests/ provide.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.post_merge_routing import GitHub, route  # noqa: E402
from tests.ci_verdicts import final, journal_rows, observed_outcomes  # noqa: E402

SYSTEM = "browser"
COMPARE_SECONDS = 240  # the base session's budget; post-merge.yml's compare step allows 6 minutes with its retry
RECHECK_SECONDS = 180  # the open issues' recheck; the route step allows 6 minutes
PASSING = ("green", "flaky")
STOP = "stop"  # DIR/stop: the legs have ended
STEPS = ("install", "run", "compare", "route")  # post-merge.yml passes each step's outcome as BROWSER_<STEP>_OUTCOME


def legs_ended(jobs: list[dict]) -> bool:
    """Whether the run's Linux and Windows legs have ended. Skipped legs (a dispatch off `ouroboros`, where only
    this job runs) bound nothing; a leg that has not ended yet keeps the session going."""
    legs = [job for job in jobs if (job.get("name") or "").startswith("post-merge (")
            and job.get("conclusion") != "skipped"]
    return bool(legs) and all(job.get("status") == "completed" for job in legs)


def pytest_addoption(parser):
    parser.addoption("--browser-stop-file", default=None,
                     help="end the session between tests once this file exists (the post-merge legs have ended)")


def _stopped(config) -> bool:
    path = config.getoption("--browser-stop-file", default=None)
    return bool(path) and Path(path).exists()


def pytest_collection_modifyitems(session, config, items):
    """A session that starts after the legs ended runs nothing (a retry round of ci_retry.py among them)."""
    if _stopped(config) and items:
        config.hook.pytest_deselected(items=list(items))
        items[:] = []


def pytest_runtest_teardown(item, nextitem):
    if nextitem is not None and _stopped(item.config):
        item.session.shouldstop = "the post-merge legs have ended: the browser check stops between tests"


def _read(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _platform_skip(nodeid: str) -> bool:
    try:
        from tests.browser_lane import PLATFORM_SKIPS
    except Exception:  # noqa: BLE001 -- without the registry no skip is tolerated
        return False
    return any(nodeid.startswith(prefix) for prefix, _ in PLATFORM_SKIPS)


def session(directory: Path) -> dict:
    """{exported, selected, outcomes} of one pytest session's projection (results.json, else its journal)."""
    data, _ = final(directory)
    rows = data.get("reports") if data else [row for row in journal_rows(directory) or []
                                             if row.get("event") == "report"]
    selected = data.get("selected_nodeids") if data else None
    return {"exported": bool(data), "selected": selected if isinstance(selected, list) else None,
            "outcomes": observed_outcomes(rows or [])}


def judge(evidence: Path, plan: dict) -> dict:
    """The run's outcome from DIR/head (its first session and ci_retry's verdict.json)."""
    empty = {"failed": [], "flaky": [], "unretried": [], "not_run": [], "skipped": [], "accepted": [], "passed": 0,
             "planned": 0, "partial": False, "legs_ended": (evidence / STOP).exists()}
    if not plan:
        return {**empty, "outcome": "partial", "partial": True, "planned": None, "reason": "no selection plan",
                "not_run": ["(the job stopped before it chose the tests: nothing ran)"]}
    if not plan.get("applicable"):
        return {**empty, "outcome": "not-applicable", "reason": plan.get("reason", "")}
    head = evidence / "head"
    first, verdict = session(head), _read(head / "verdict.json") or {}
    planned = first["selected"]
    flaky = sorted(row["nodeid"] for row in verdict.get("flaky") or [])
    accepted = sorted(row["nodeid"] for row in verdict.get("accepted") or [])
    failed, unretried = [], []
    for row in verdict.get("failed") or []:
        if row["nodeid"] not in accepted:
            (failed if len(row.get("attempts") or []) > 1 else unretried).append(row["nodeid"])
    outcomes = first["outcomes"]
    if not verdict:  # ci_retry.py was stopped before its verdict: a red it saw is reported, never handed over
        unretried = sorted(node for node, result in outcomes.items() if result["outcome"] == "failed")
    retried = {}
    for round_dir in sorted(head.glob("retry-*")):
        retried.update(session(round_dir)["outcomes"])
    skipped = sorted(node for node, result in {**outcomes, **retried}.items()
                     if result["outcome"] == "skipped" and not _platform_skip(node))
    if planned is None:
        not_run = ["(the session left no complete export: the nodes it never started are unknown; planned: "
                   + ", ".join(plan.get("args") or []) + ")"]
    else:
        ran = set(outcomes) | set(retried)
        not_run = sorted(set(planned) - ran) + [node for node in verdict.get("not_run") or [] if node not in planned]
    passed = sum(result["outcome"] == "passed" for node, result in outcomes.items()
                 if node not in failed and node not in unretried and node not in flaky)
    partial = bool(not_run or skipped or unretried or not verdict)
    outcome = ("red" if failed else "partial" if partial else "flaky" if flaky else "green")
    return {"outcome": outcome, "partial": partial, "failed": sorted(failed), "unretried": sorted(unretried),
            "flaky": flaky, "accepted": accepted, "not_run": not_run, "skipped": skipped, "passed": passed,
            "planned": len(planned) if planned is not None else None, "reason": verdict.get("reason", ""),
            "legs_ended": (evidence / STOP).exists()}


def git(*args: str, cwd: Path = REPO) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
                          encoding="utf-8").stdout.strip()


def session_command(tree: Path, nodes: list[str], directory: Path, *, seconds: int, retry: bool) -> list[str]:
    """The UI lane's own invocation (ui-browser.yml) for `nodes` of `tree`, by tree's own scripts. The launcher's
    raw evidence stays beside DIR, as the run step's does (`DIR-raw/<pass>`): only DIR is uploaded."""
    raw = directory.parent.parent / f"{directory.parent.name}-raw" / directory.name
    command = ["python", "-I", "-S", str(tree / "scripts/safe_test.py"), "--evidence-dir", str(raw), "--",
               "python", "-m", "pytest", *nodes, "-m", "ui_browser", "-vv", "--tb=short", "--timeout=900",
               f"--session-timeout={seconds}", f"--ci-evidence-dir={directory}"]
    if retry:
        command = ["python", "-I", "-S", str(tree / "scripts/ci_retry.py"), "--evidence-dir", str(directory),
                   "--", *command]
    return command


def run_nodes(tree: Path, nodes: list[str], directory: Path, *, seconds: int, retry: bool) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    return subprocess.run(session_command(tree, nodes, directory, seconds=seconds, retry=retry), cwd=tree).returncode


def command_compare(args) -> int:
    """Run each confirmed red at BASE once more (with one retry): DIR/compare.json {node: failed|passed|flaky|
    absent|unknown}."""
    evidence, plan = Path(args.evidence_dir), _read(Path(args.plan)) or {}
    judged = judge(evidence, plan)
    results = {}
    if judged["failed"]:
        base, tree = plan["base"], Path(args.base_tree)
        if not tree.exists():
            git("worktree", "add", "--detach", str(tree), base)
        present = [node for node in judged["failed"] if (tree / node.split("::", 1)[0]).is_file()]
        results.update({node: "absent" for node in judged["failed"] if node not in present})
        if present:
            run_nodes(tree, present, evidence / "base", seconds=COMPARE_SECONDS, retry=True)
            verdict = _read(evidence / "base" / "verdict.json") or {}
            first = session(evidence / "base")
            failed = {row["nodeid"] for row in verdict.get("failed") or [] if len(row.get("attempts") or []) > 1}
            flaky = {row["nodeid"] for row in verdict.get("flaky") or []}
            for node in present:
                outcome = first["outcomes"].get(node, {}).get("outcome")
                if node in failed:
                    results[node] = "failed"
                elif node in flaky:
                    results[node] = "flaky"
                elif outcome == "passed":
                    results[node] = "passed"
                elif first["selected"] is not None and node not in first["selected"]:
                    results[node] = "absent"  # its module exists at BASE, the test does not
                else:
                    results[node] = "unknown"
    _write(evidence / "compare.json", {"base": plan.get("base"), "results": results})
    return 0


def relevant_files(plan: dict, node: str) -> set[str]:
    module = node.split("::", 1)[0]
    files = {module}
    for row in plan.get("selected") or []:
        if row["module"] == module:
            files.update(row["files"])
    for row in plan.get("sentinels") or []:
        if row["node"].split("::", 1)[0] == module:
            files.update(row["guards"])
    return files


def attribute(plan: dict, failed: list[str], compared: dict) -> list[dict]:
    """route()'s verdicts: culprit (one window commit touched the node's files), pre-existing or unattributed."""
    verdicts = []
    commits = plan.get("commits") or []
    for node in failed:
        at_base = compared.get(node, "unknown")
        if at_base == "failed":
            verdicts.append({"id": node, "verdict": "pre-existing"})
            continue
        if at_base not in ("passed", "absent"):
            verdicts.append({"id": node, "verdict": "unattributed",
                             "why": f"at the base it was {at_base}, so the window is not shown to have caused it"})
            continue
        files = relevant_files(plan, node)
        touching = [c["sha"] for c in commits if files & set(c["files"])]
        if len(touching) == 1:
            sha = touching[0]
            verdicts.append({"id": node, "verdict": "culprit", "culprit": sha,
                             "last_good": git("rev-parse", f"{sha}^1")})
        else:
            verdicts.append({"id": node, "verdict": "unattributed",
                             "candidates": touching or [c["sha"] for c in commits],
                             "why": "introduced in the window; " + (
                                 f"{len(touching)} of its commits touched the files that selected it" if touching
                                 else "none of its commits touched the files that selected it")})
    return verdicts


class DryGitHub(GitHub):
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
                "body": (body or {}).get("body", ""), "labels": [{"name": "post-merge-red"}]}, ""


def command_route(args, environ=os.environ) -> int:
    evidence, plan = Path(args.evidence_dir), _read(Path(args.plan)) or {}
    judged = judge(evidence, plan)
    compared = (_read(evidence / "compare.json") or {}).get("results") or {}
    verdicts = attribute(plan, judged["failed"], compared)
    dry = args.dry_run or environ.get("GITHUB_REF", "refs/heads/ouroboros") != "refs/heads/ouroboros"
    github = (DryGitHub if dry else GitHub)(environ.get("GITHUB_REPOSITORY", ""), environ.get("GH_TOKEN", ""))
    head = plan.get("head") or git("rev-parse", "HEAD")
    ran = session(evidence / "head")["outcomes"]

    def same_run(ids):
        return {node: "passed" for node in ids if ran.get(node, {}).get("outcome") == "passed"
                and node not in judged["flaky"]}

    def recheck(ids):
        if (evidence / STOP).exists():
            return {}  # the legs have ended: the open issues wait for a later window
        directory = evidence / "recheck"
        run_nodes(REPO, list(ids), directory, seconds=RECHECK_SECONDS, retry=False)
        seen = session(directory)
        return {node: ("absent" if seen["selected"] is not None and node not in seen["selected"]
                       and not (REPO / node.split("::", 1)[0]).is_file()
                       else seen["outcomes"].get(node, {}).get("outcome", "unknown")) for node in ids}

    def ancestor(older, newer):
        return subprocess.run(["git", "merge-base", "--is-ancestor", older, newer], cwd=REPO).returncode == 0

    record = route(github, classification={"failed": [{"id": node} for node in judged["failed"]],
                                           "flaky": [{"id": node, "retry": "passed"} for node in judged["flaky"]],
                                           "base": plan.get("base")},
                   attribution={"verdicts": verdicts}, head=head, run_url=args.run_url, recheck=recheck,
                   system=SYSTEM, same_run=same_run, ancestor=ancestor, owed=())
    _write(evidence / "routing.json", {"dry_run": dry, "verdicts": verdicts, "record": record,
                                       "writes": getattr(github, "writes", None)})
    return 0


def command_legs(args, environ=os.environ) -> int:
    """Exit 0 when this attempt's legs have ended, 1 while they run, 2 when the job list does not read."""
    github = GitHub(args.repo, environ.get("GH_TOKEN", ""))
    try:
        jobs = github.get(f"actions/runs/{args.run_id}/attempts/{args.attempt}/jobs?per_page=100")["jobs"]
    except Exception as error:  # noqa: BLE001 -- an unread list proves nothing; the next poll asks again
        print(f"the job list does not read: {type(error).__name__}")
        return 2
    return 0 if legs_ended(jobs) else 1


def _shown(nodes: list[str], limit: int = 40) -> list[str]:
    rows = [f"- `{node}`" for node in nodes[:limit]]
    return rows + ([f"- … and {len(nodes) - limit} more (outcome.json in the evidence)"] if len(nodes) > limit else [])


def summary_lines(plan: dict, judged: dict, routing: dict | None, steps: dict | None = None) -> list[str]:
    title = {"green": "GREEN", "flaky": "GREEN with FLAKY tests", "red": "RED", "partial": "CHECKED PARTIALLY",
             "not-applicable": "NOT APPLICABLE"}[judged["outcome"]]
    if judged["outcome"] == "red" and judged.get("partial"):
        title += ", AND CHECKED PARTIALLY"
    lines = [f"## Post-merge browser check (pilot): {title}", "",
             f"Window `{(plan.get('base') or '')[:12]}..{(plan.get('head') or '')[:12]}`: {plan.get('reason', '')}."]
    if judged["outcome"] == "not-applicable":
        return lines + ["", "No browser test ran; this is no green result."]
    sentinels = plan.get("sentinels") or []
    missing = [row["test"] for row in sentinels if not row.get("present")]
    lines += ["", f"Selected {len(plan.get('selected') or [])} module(s) and {len(sentinels) - len(missing)} "
                  f"WebKit sentinel(s), estimated {round((plan.get('estimate_ms') or 0) / 60000, 1)} min; "
                  f"{judged['planned'] if judged['planned'] is not None else 'unknown'} node(s) selected, "
                  f"{judged['passed']} passed."]
    for row in plan.get("selected") or []:
        lines.append(f"- `{row['module']}`: {'; '.join(sorted(set(row['reasons'])))}")
    if missing:
        lines += ["", "Sentinels the head no longer defines (update SENTINELS in the selector): "
                  + ", ".join(f"`{test}`" for test in missing) + "."]
    for key, label in (("dropped", "Not selected (cap)"), ("wide", "Hub files left to the release lane"),
                       ("excluded", "Excluded")):
        if plan.get(key):
            lines += ["", f"{label}:"] + [f"- `{row.get('module') or row.get('file')}`: "
                                          f"{row.get('why') or str(row.get('modules')) + ' modules name it'}"
                                          for row in plan[key]]
    if judged.get("legs_ended"):
        lines += ["", "The Linux and Windows legs ended first, so the session stopped between tests: this check never "
                      "holds the next window."]
    unfinished = {step: outcome for step, outcome in (steps or {}).items() if outcome not in ("success", "skipped", "")}
    if unfinished:
        lines += ["", "Steps that did not finish: " + ", ".join(f"{step} ({outcome})" for step, outcome in
                                                               sorted(unfinished.items())) + "."]
    verdicts = {row["id"]: row for row in (routing or {}).get("verdicts") or []}
    for key, label in (("failed", "FAILED (twice)"), ("unretried", "Failed once, not retried: the session stopped"),
                       ("flaky", "FLAKY (passed on retry, not handed over)"),
                       ("accepted", "ACCEPTED (tests/accepted_failures.json)"), ("skipped", "Skipped (not executed)")):
        if judged[key]:
            lines += ["", f"{label}:"]
            for node in judged[key]:
                verdict = verdicts.get(node)
                note = ""
                if verdict:
                    note = {"culprit": f" — introduced by `{verdict.get('culprit', '')[:12]}`",
                            "pre-existing": " — already red at the base: not attributed"}.get(
                        verdict["verdict"], " — not attributed" + (
                            f"; candidate merges: {', '.join('`' + c[:12] + '`' for c in verdict['candidates'])}"
                            if verdict.get("candidates") else ""))
                lines.append(f"- `{node}`{note}")
    if judged["not_run"]:
        lines += ["", f"Not executed ({len(judged['not_run'])}):"] + _shown(judged["not_run"])
    if routing and routing.get("dry_run"):
        lines += ["", f"Dry routing: {len(routing.get('writes') or [])} GitHub write(s) recorded, none made."]
    return lines


def command_summary(args, environ=os.environ) -> int:
    evidence, plan = Path(args.evidence_dir), _read(Path(args.plan)) or {}
    judged = judge(evidence, plan)
    _write(evidence / "outcome.json", judged)
    steps = {step: environ.get(f"BROWSER_{step.upper()}_OUTCOME", "") for step in STEPS}
    text = "\n".join(summary_lines(plan, judged, _read(evidence / "routing.json"), steps)) + "\n"
    print(text)
    for name, value in (("GITHUB_STEP_SUMMARY", text), ("GITHUB_OUTPUT", f"outcome={judged['outcome']}\n")):
        if environ.get(name):
            with open(environ[name], "a", encoding="utf-8") as handle:
                handle.write(value)
    return 0 if judged["outcome"] in PASSING + ("not-applicable",) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="post_merge_browser.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    legs = commands.add_parser("legs")
    for option in ("--repo", "--run-id", "--attempt"):
        legs.add_argument(option, required=True)
    for name in ("compare", "route", "summary"):
        command = commands.add_parser(name)
        command.add_argument("--evidence-dir", required=True)
        command.add_argument("--plan", required=True)
        if name == "compare":
            command.add_argument("--base-tree", required=True)
        if name == "route":
            command.add_argument("--run-url", required=True)
            command.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    return {"legs": command_legs, "compare": command_compare, "route": command_route,
            "summary": command_summary}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
