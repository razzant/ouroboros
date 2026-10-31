#!/usr/bin/env python3
"""Run one CI test pass, retry what failed once and continue what it never ran.

    python3 -I -S scripts/ci_retry.py --evidence-dir DIR [--accepted tests/accepted_failures.json]
        [--max-rounds N] -- <test command>

The command is one pytest pass that writes its public projection to DIR through
`--ci-evidence-dir` (tests/ci_evidence.py): results.json, or only the incremental
events.jsonl of a session that was killed, which still holds the lane block it
journaled at collection. When it ends red, the SAME command runs again in rounds,
round k in DIR/retry-<k>, for

- each test that failed in any phase or crashed its xdist worker, and the test in
  flight when its session was killed with no crash to explain it: its one retry;
- each test of the lane assignment (the `default_parallel`, `default_serial` or
  `ui_browser` block: `assigned`, else `full`) that a stopped or killed session never
  executed (a dead worker under `--max-worker-restart=0`, a session timeout, the
  Windows thread timeout that ends the whole process), or was in flight on another
  worker when a crash stopped the session: its first run, after which it may still
  get its retry in a later round.

A round names its tests in DIR/retry-<k>/argsfile.txt and passes `@argsfile`
(pytest's argument file) where the command's first test path stood, dropping the
others; it writes its projection to DIR/retry-<k>, its JUnit to a `retry-<k>`
directory beside the command's own, and leaves out the options that prove a whole
lane, which no subset can (the lane is proven by the first session's projection and
this verdict, `tests/ci_evidence.py reconcile-shards`). A test runs at most twice.
Rounds stop when nothing is left, or when a round runs none of its tests.

DIR/verdict.json then records the decision (owner decision 1A, 2026-10-21). A test
that failed and passed its retry is FLAKY and does not block. One that failed again,
or skipped instead of passing, FAILED and blocks, unless tests/accepted_failures.json
accepts it (tests/ci_accepted.py: an open issue the owner labelled last). An assigned
test that never ran is NOT RUN and blocks, and so does a red session that no test
explains. The exit status is 0 iff the verdict is green, flaky or accepted, else 1,
or the command's own when it left no evidence at all. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # -I leaves the checkout off sys.path.
    sys.path.insert(0, str(REPO))

from tests import ci_accepted  # noqa: E402  -- stdlib only, as is:
from tests.ci_verdicts import attempt_text, final, journal_rows, valid  # noqa: E402

MAX_ROUNDS = 10  # A backstop below the step's own deadline: a pass that needs more ends red, naming the rest.
_PASSING = frozenset({"green", "flaky", "accepted"})
_LANES = ("default_parallel", "default_serial", "ui_browser")
_INTERNAL_ERROR = 3  # pytest.ExitCode.INTERNAL_ERROR: a hook or the xdist scheduler raised
# A positional test path or node id: what a round replaces with its argument file.
_TEST_PATH = re.compile(r"(?:tests|devtools)/")
# Options of the wrapped programs (python, safe_test, run_tests, pytest and its plugins) that
# take the NEXT argument as their value unless it is attached with `=`. Such a value is never
# a test path, whatever it looks like.
_VALUED = frozenset({
    "-m", "-k", "-n", "-p", "-o", "-c", "-r", "-W", "-X", "--dist", "--tb", "--junitxml", "--junit-xml",
    "--junit-prefix", "--ci-evidence-dir", "--basetemp", "--rootdir", "--confcutdir", "--deselect",
    "--ignore", "--ignore-glob", "--override-ini", "--numprocesses", "--maxprocesses",
    "--max-worker-restart", "--maxfail", "--durations", "--durations-min", "--timeout", "--timeout-method",
    "--session-timeout", "--capture", "--import-mode", "--log-level", "--log-file", "--default-suite-shard",
    "--ui-browser-shard", "--serial-shards", "--evidence-dir", "--temp-parent", "--base", "--extra-test",
    "--test-reason",
})
# Whole-lane proofs refuse a subset of their lane (tests/default_lane.py, tests/browser_lane.py),
# and a round's own deadline is the step's.
_DROPPED = frozenset({"--default-suite-shard", "--ui-browser-shard", "--session-timeout",
                      "--default-suite-manifest", "--require-ui-browser"})
_JUNIT = ("--junitxml", "--junit-xml")
# A dead worker's own facts on its crash row (tests/ci_evidence.py): its gateway and, only for the record
# that worker's pytest-timeout callback left, the timeout. An attempt carries them; nothing here infers one.
_CRASH_FACTS = ("worker", "timeout_callback_seconds")


def _say(line: str) -> None:
    """One log line in UTF-8, which runners decode whatever the console code page is."""
    sys.stdout.flush()
    stream = getattr(sys.stdout, "buffer", None)
    if stream is None:
        print(line, flush=True)
        return
    stream.write((line + "\n").encode("utf-8", "replace"))
    stream.flush()


def retry_command(command: list[str], *, argsfile: Path, direct: list[str], round_dir: Path) -> list[str]:
    """`command` for one round, every other argument kept in order.

    Its positional test paths (`tests/...`, `devtools/...`, node ids with `::`, but never
    an option's value) become one `@argsfile`, followed by the `direct` node ids, where
    the first of them stood, or at the end without one. `--ci-evidence-dir` names
    round_dir (and is added when missing), the JUnit file moves into a directory named
    like round_dir beside its own (it stays private: raw JUnit is never public
    evidence), and the whole-lane options are dropped, with their values.
    """
    out, placed, evidence, index = [], False, False, 0
    while index < len(command):
        token = command[index]
        name, attached, value = token.partition("=")
        option = token.startswith("-")
        valued = option and not attached and token in _VALUED
        following = command[index + 1] if valued and index + 1 < len(command) else ""
        if option and name in _DROPPED:
            pass
        elif option and name == "--ci-evidence-dir":
            out += [f"{name}={round_dir}"] if attached else [name, str(round_dir)]
            evidence = True
        elif option and name in _JUNIT:
            original = Path(value if attached else following)
            moved = str(original.parent / round_dir.name / original.name)
            out += [f"{name}={moved}"] if attached else [name, moved]
        elif valued:
            out += command[index:index + 2]
        elif not option and (_TEST_PATH.match(token) or "::" in token):
            if not placed:
                out += [f"@{argsfile}", *direct]
                placed = True
        else:
            out.append(token)
        index += 2 if valued else 1
    if not placed:
        out += [f"@{argsfile}", *direct]
    if not evidence:
        out.append(f"--ci-evidence-dir={round_dir}")
    return out


def write_argsfile(path: Path, nodes: list[str]) -> list[str]:
    """Write a round's node ids one per line; return the ids that must be arguments instead.

    pytest reads `@file` with argparse: before Python 3.12 in the locale encoding
    (cp1252 on Windows runners, where safe_test deliberately leaves UTF-8 mode off),
    from 3.12 in the filesystem encoding, split at every line boundary that
    str.splitlines() knows. Printable ASCII reads the same under all of them, so the
    file holds only such ids. Any other id is passed as an argument of its own, which
    the operating system hands to Python as Unicode. Each xdist worker reads the file
    again: it receives the controller's invocation arguments.
    """
    plain = [node for node in nodes if node.isascii() and node.isprintable()]
    with open(path, "w", encoding="ascii", newline="\n") as handle:
        handle.writelines(node + "\n" for node in plain)
    return [node for node in nodes if not (node.isascii() and node.isprintable())]


def _run(command: list[str]) -> int:
    """The command's exit status. It inherits the streams, the working directory and the
    environment, less the token that only the accepted-failure check reads."""
    env = {key: value for key, value in os.environ.items() if key not in ("GH_TOKEN", "GITHUB_TOKEN")}
    # Resolve the program on PATH as the calling shell did. Windows' own lookup tries this
    # process's directory first, and a venv's python.exe runs as the base interpreter there,
    # so a bare `python` would start the hosted Python without the venv's packages.
    command = [shutil.which(command[0]) or command[0], *command[1:]]
    sys.stdout.flush()
    process = subprocess.Popen(command, env=env)
    try:
        code = process.wait()
    except KeyboardInterrupt:
        if os.name != "nt":  # Windows: the console's Ctrl-C reached the command already.
            try:
                process.send_signal(signal.SIGINT)
            except OSError:
                pass
        process.wait()
        raise
    return code if code >= 0 else 128 - code


def _journal(directory: Path) -> tuple[list[dict], list[str], dict | None] | None:
    """(valid report rows, tests in flight, the lane block journaled at collection) of
    events.jsonl; None without one.

    As tests/ci_evidence.py `_journal` reads it.
    """
    rows = journal_rows(directory)
    if rows is None:
        return None
    reports, running, lane = [], {}, None
    for row in rows:
        if row.get("event") == "start":
            running[row["nodeid"]] = None
        elif row.get("event") == "finish":
            running.pop(row["nodeid"], None)
        elif row.get("event") == "report" and valid(row):
            reports.append(row)
            if row["phase"] == "crash":
                running.pop(row["nodeid"], None)  # A crashed worker never sends finish.
        elif row.get("event") == "lane" and row.get("block") in _LANES and isinstance(row.get("lane"), dict):
            lane = row["lane"]
    return reports, list(running), lane


def _outcomes(rows) -> dict[str, dict]:
    """Each executed node's outcome in one session: its first failure, else a skip, else a pass.

    Executed is the lane guard's rule (tests/browser_lane.py LaneReconciliation, as
    tests/ci_evidence.py applies it): a call phase, a crash, or any report that did not pass.
    """
    outcomes = {}
    for row in rows:
        if row["phase"] not in ("call", "crash") and row["outcome"] == "passed":
            continue
        node, held = row["nodeid"], outcomes.get(row["nodeid"])
        if row["outcome"] == "failed":
            if held is None or held["outcome"] != "failed":
                outcomes[node] = {"outcome": "failed", "phase": row["phase"],
                                  "error_type": str(row.get("error_type") or ""),
                                  **{key: row[key] for key in _CRASH_FACTS if key in row}}
        elif row["outcome"] == "skipped":
            if held is None or held["outcome"] == "passed":
                outcomes[node] = {"outcome": "skipped", "phase": row["phase"], "error_type": "Skipped"}
        elif held is None:
            outcomes[node] = {"outcome": "passed", "phase": row["phase"], "error_type": ""}
    return outcomes


def _evidence(directory: Path, *, aborted: bool = False) -> dict | None:
    """What one session's projection shows; None when it left neither results.json nor a journal.

    Without a valid final export the journal stands in for it, with the lane block it
    recorded at collection (tests/ci_evidence.py); without a lane block, a dead worker's
    crash record names what the session never finished. A test the journal leaves in flight
    did not finish. With no crash report to explain why, it is what ended the session
    (pytest-timeout's thread method kills the whole process): a failed attempt, as a
    crash is. Beside a crash it was another worker's test that the stop interrupted
    (tests/conftest.py aborts an owned run at once): it counts as never run. So it does
    when the session `aborted` with pytest's internal error (exit status 3): that error
    is raised by a hook or the xdist scheduler between tests, not by the tests in flight.
    """
    data, unreadable = final(directory)
    data = None if unreadable else data
    journal = _journal(directory)
    if data is None and journal is None:
        return None
    rows = data["reports"] if data is not None else journal[0]
    results = _outcomes(rows)
    explained = aborted or any(row["phase"] == "crash" for row in rows)
    for node in journal[1] if journal and not explained else []:
        results.setdefault(node, {"outcome": "failed", "phase": "crash", "error_type": "InFlight"})
    lane = (next((data[key] for key in _LANES if isinstance(data.get(key), dict)), None) if data is not None
            else journal[2])
    assigned = lane.get("assigned", lane.get("full")) if lane is not None else None
    crash = data.get("worker_crash") if data is not None else None
    if lane is None and isinstance(crash, dict) and crash.get("collection_available") is True \
            and isinstance(crash.get("unfinished"), list):
        # A pass without a lane block (no shard option): the session's own crash record
        # (tests/conftest.py) names what it collected and never finished.
        assigned = sorted(set(results) | {node for node in crash["unfinished"] if isinstance(node, str)})
    failures = data.get("collection_failures") if data is not None else None
    collected = data.get("tests_collected") if data is not None else None
    return {"exported": data is not None, "results": results,
            "assigned": assigned if isinstance(assigned, list) and all(isinstance(n, str) for n in assigned) else None,
            "collected": collected if type(collected) is int else None,
            "collection_failures": [row["nodeid"] for row in failures if isinstance(row, dict)
                                    and isinstance(row.get("nodeid"), str)] if isinstance(failures, list) else []}


def _explained(facts: dict, assigned) -> bool:
    """A red session that shows why: a failed test, a collection failure or an assigned test it left."""
    return (any(result["outcome"] == "failed" for result in facts["results"].values())
            or bool(facts["collection_failures"])
            or (assigned is not None and bool(set(assigned) - set(facts["results"]))))


def _pending(tries: dict[str, list], assigned) -> list[str]:
    """Tests failed once and not yet retried, and assigned tests never run."""
    retry = {node for node, attempts in tries.items() if len(attempts) == 1 and attempts[0]["outcome"] == "failed"}
    return sorted(retry | {node for node in assigned or () if node not in tries})


def _system() -> str:
    """This system as tests/ci_verdicts.py `platform_name` names it."""
    if sys.platform.startswith("linux"):
        return "linux"
    return "windows" if sys.platform in ("win32", "cygwin") else sys.platform


def _accept(nodes: list[str], path: Path) -> list[dict]:
    """tests/ci_accepted.py over this run's repository; nothing without its identity and token."""
    try:
        entries = ci_accepted.load(path)
    except (OSError, ValueError) as error:
        _say(f"::warning::{path} is unreadable ({type(error).__name__}); no failure is accepted.")
        return []
    return ci_accepted.accepted(nodes, _system(), entries, os.environ.get("GITHUB_REPOSITORY", ""),
                                os.environ.get("GITHUB_REPOSITORY_OWNER", ""),
                                os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN", ""),
                                read_issue=ci_accepted.read_issue, read_events=ci_accepted.read_events)


def _shown(attempt: dict) -> dict:
    return {"phase": attempt["phase"], "error_type": attempt["error_type"],
            **{key: attempt[key] for key in _CRASH_FACTS if key in attempt}}


def decide(directory: Path, rounds: list[dict], tries: dict[str, list], assigned, collection: set[str],
           causes: list[str], *, accepted_file: Path | None = None, green: bool = False) -> dict:
    """The pass's verdict.json from its rounds; `green` when the first session ended with status 0."""
    flaky, failed, passed, skipped = [], [], 0, 0
    for node in sorted(tries):
        first = tries[node][0]
        if first["outcome"] in ("passed", "skipped") or green:
            passed += first["outcome"] != "skipped"
            skipped += first["outcome"] == "skipped"
        elif len(tries[node]) > 1 and tries[node][1]["outcome"] == "passed":
            flaky.append({"nodeid": node, "first": _shown(first)})
        else:
            failed.append({"nodeid": node, "attempts": [_shown(attempt) for attempt in tries[node]]})
    if not green:
        failed += [{"nodeid": node, "attempts": [{"phase": "collect", "error_type": "CollectionError"}]}
                   for node in sorted(collection - set(tries))]
    not_run = [] if green else sorted(set(assigned or ()) - set(tries))
    accepted = _accept([row["nodeid"] for row in failed], accepted_file) if failed and accepted_file else []
    blocking = {row["nodeid"] for row in failed} - {row["nodeid"] for row in accepted}
    verdict = ("red" if causes or blocking or not_run else "accepted" if accepted
               else "flaky" if flaky else "green")
    return {
        "format": 1, "pass": directory.name,
        "github": {name: os.environ.get("GITHUB_" + name.upper(), "") for name in ("sha", "run_id", "run_attempt")},
        "platform": _system(), "rounds": rounds, "flaky": flaky, "failed": failed, "accepted": accepted,
        "not_run": not_run, "continued": sum(attempts[0]["round"] > 0 for attempts in tries.values()),
        "counts": {"passed": passed, "flaky": len(flaky), "failed": len(failed), "accepted": len(accepted),
                   "not_run": len(not_run), "skipped": skipped},
        "verdict": verdict, "reason": "; ".join(causes),
    }


def _record(tries: dict[str, list], results: dict[str, dict], round_: int) -> None:
    for node, result in results.items():
        tries.setdefault(node, []).append({**result, "round": round_})


def run_pass(directory: Path, command: list[str], *, max_rounds: int = MAX_ROUNDS,
             accepted_file: Path | None = None) -> tuple[dict, int]:
    """(verdict, exit status) of one pass: its first session, then as many rounds as it needs."""
    causes, tries, collection = [], {}, set()
    try:
        code = _run(command)
    except KeyboardInterrupt:
        code, causes = 130, ["interrupted"]
    first = _evidence(directory, aborted=code == _INTERNAL_ERROR)
    rounds = [{"dir": ".", "exit": code, "ran": len(first["results"]) if first else 0}]
    assigned = first["assigned"] if first else None
    if first is not None:
        _record(tries, first["results"], 0)
        collection |= set(first["collection_failures"])
    if code == 0:
        return decide(directory, rounds, tries, assigned, collection, causes, green=True), 0
    if first is None:
        causes.append(f"the command (exit status {code}) left neither results.json nor events.jsonl")
        return decide(directory, rounds, tries, assigned, collection, causes), code
    if causes:
        pass  # Interrupted: no round runs.
    elif assigned is None and not first["exported"]:
        causes.append("the session ended before its final export, and no lane block names its assignment: "
                      "the tests it never ran are unknown")
    elif assigned is None and first["collected"] is not None and first["collected"] > len(first["results"]):
        causes.append(f"{first['collected'] - len(first['results'])} collected test(s) never ran, "
                      "and no lane block or crash record names them")
    elif not _explained(first, assigned):
        causes.append(f"the session ended with exit status {code}, but no test failed or went unexecuted")
    round_ = 0
    while not causes and (pending := _pending(tries, assigned)):
        if round_ == max_rounds:
            causes.append(f"{len(pending)} test(s) still pending after {max_rounds} retry round(s)")
            break
        round_ += 1
        round_dir = directory / f"retry-{round_}"
        round_dir.mkdir(parents=True)
        argsfile = round_dir / "argsfile.txt"
        direct = write_argsfile(argsfile, pending)
        retried = sum(node in tries for node in pending)
        _say(f"=== ci_retry round {round_}: retry {retried} failed test(s), run {len(pending) - retried} "
             f"never run; {argsfile}")
        try:
            code = _run(retry_command(command, argsfile=argsfile, direct=direct, round_dir=round_dir))
        except KeyboardInterrupt:
            code, causes = 130, ["interrupted"]
        facts = _evidence(round_dir, aborted=code == _INTERNAL_ERROR) or {"results": {}, "collection_failures": []}
        wanted = set(pending)
        results = {node: result for node, result in facts["results"].items() if node in wanted}
        rounds.append({"dir": round_dir.name, "exit": code, "ran": len(facts["results"])})
        _record(tries, results, round_)
        collection |= set(facts["collection_failures"])
        if causes:
            break
        if not results:
            causes.append(f"{round_dir.name} ran none of its {len(pending)} test(s) (exit status {code})")
        elif code != 0 and not _explained(facts, pending):
            causes.append(f"{round_dir.name} ended with exit status {code}, but no test failed or went unexecuted")
    verdict = decide(directory, rounds, tries, assigned, collection, causes, accepted_file=accepted_file)
    return verdict, 130 if "interrupted" in causes else 0 if verdict["verdict"] in _PASSING else 1


def write_verdict(path: Path, verdict: dict) -> None:
    """Replace path atomically: a reader never sees half a verdict."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=".verdict.",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _report(verdict: dict) -> None:
    counts = verdict["counts"]
    _say(f"CI_RETRY {verdict['pass']}: {verdict['verdict'].upper()} after {len(verdict['rounds'])} round(s): "
         + ", ".join(f"{key} {counts[key]}" for key in ("passed", "skipped", "flaky", "failed", "accepted", "not_run"))
         + f", continued {verdict['continued']}")
    issues = {row["nodeid"]: row["issue"] for row in verdict["accepted"]}
    for row in verdict["flaky"]:
        _say(f"FLAKY {row['nodeid']} (first: {attempt_text(row['first'])})")
    for row in verdict["failed"]:
        tries = "; ".join(attempt_text(attempt) for attempt in row["attempts"])
        _say(f"{'ACCEPTED #' + str(issues[row['nodeid']]) if row['nodeid'] in issues else 'FAILED'} "
             f"{row['nodeid']} ({tries})")
    for node in verdict["not_run"]:
        _say(f"NOT_RUN {node}")
    if verdict["reason"]:
        _say(f"RED {verdict['reason']}")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(prog="ci_retry.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evidence-dir", type=Path, required=True,
                        help="the command's --ci-evidence-dir; the rounds and verdict.json go beneath it")
    parser.add_argument("--accepted", type=Path, help="accepted failures (tests/accepted_failures.json)")
    parser.add_argument("--max-rounds", type=int, default=MAX_ROUNDS, help="retry rounds at most")
    split = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split])
    command = argv[split + 1:]
    if not command:
        parser.error("supply the test command after --")
    if args.max_rounds < 0:
        parser.error("--max-rounds must not be negative")
    directory = args.evidence_dir.resolve()
    # A round that left no projection must not read an earlier invocation's in its place.
    stale = [path.name for path in (directory / "verdict.json", *sorted(directory.glob("retry-*"))) if path.exists()]
    if stale:
        parser.error(f"{directory} already holds {', '.join(stale)}; give each pass a fresh evidence directory")
    verdict, code = run_pass(directory, command, max_rounds=args.max_rounds, accepted_file=args.accepted)
    write_verdict(directory / "verdict.json", verdict)
    _report(verdict)
    return code


if __name__ == "__main__":
    sys.exit(main())
