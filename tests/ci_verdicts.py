"""Retry verdicts, recorded durations and a run's verdict list: tests/ci_evidence.py's stdlib CLI half.

scripts/ci_retry.py writes a pass's verdict.json beside its projection (results.json and
events.jsonl, tests/ci_evidence.py) and runs each retry round in `retry-<k>` beside them.
Here is what both need to read them: a report row, the tests it shows failing as their
xfail marker expects, a session's final export, a verdict and whether it lets a red session pass, the run's weight table the lane plugins
shard by (`durations_table`), the table one run's projections make for the next ones
(`collect_durations`), and the complete list of a run's flaky, accepted, failed and
never-run tests (`render_run_verdict`). Stdlib only: tests/ci_evidence.py's CLI runs these
under `python3 -I -S` as its `durations` and `run-verdict` subcommands.
"""
from __future__ import annotations

import fnmatch
import html
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

PHASES = frozenset({"setup", "call", "teardown"})
# scripts/ci_retry.py: retry round k of a pass runs in <pass>/retry-<k>.
ROUND_DIR = re.compile(r"retry-[1-9][0-9]*")
VERDICTS = frozenset({"green", "flaky", "accepted", "red"})
SYSTEMS = frozenset({"linux", "darwin", "windows"})
# The run's weight table (its frozen plan, scripts/ci_shard_plan.py); the lane plugins read it at collection.
DURATIONS_ENV = "OUROBOROS_CI_DURATIONS"
# The owner's persisted table (ouroboros.test_evidence.TEST_DURATIONS_ENV spells the same key): read for weights when
# CI's table is absent, and brought up to date by every owned session at its end (`record_session_durations`).
TEST_DURATIONS_ENV = "OUROBOROS_TEST_DURATIONS"


def write_json(path: Path, value) -> None:
    """Write an already approved public projection, never a private source blob."""
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="." + path.name, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def cell(value) -> str:
    return html.escape(str(value), quote=True).replace("|", "&#124;").replace("\n", "<br>")


def valid(row) -> bool:
    return (isinstance(row, dict)
            and all(isinstance(row.get(key), str) for key in ("nodeid", "phase", "outcome"))
            and (row["phase"] in PHASES or row["phase"] == "crash")
            and row["outcome"] in {"passed", "failed", "skipped"}
            and ("is_subtest" not in row or type(row["is_subtest"]) is bool)
            and (row.get("wasxfail") is None or isinstance(row["wasxfail"], str)))


def expected_failures(rows) -> dict[str, str]:
    """{nodeid: its marker's reason} of each test that ran and failed as its xfail marker expects (XFAIL).

    pytest reports that call as "skipped" with the reason in `wasxfail`; tests/ci_evidence.py keeps both and
    gives every skipped row the field (None for a real skip). Its setup and teardown must be on record as passed,
    and every other report of the test too: an xfail at setup (`run=False`, a failed fixture) never ran the test,
    one at teardown is a teardown error the marker absorbed, a call ended by `pytest.xfail()` (error type
    `XFailed`) opted out as a skip does, and a session that ended before one of the phases was reported (a lost
    worker, a torn journal) does not show the test ran whole. A skipped row without the field predates it and
    proves nothing; a strict XPASS is a failed call.
    """
    reasons, spoiled, passed = {}, set(), set()
    for row in rows:
        if (row["phase"] == "call" and row["outcome"] == "skipped" and isinstance(row.get("wasxfail"), str)
                and row.get("error_type") != "XFailed" and not row.get("is_subtest")):
            reasons.setdefault(row["nodeid"], row["wasxfail"])
        elif row["outcome"] != "passed":
            spoiled.add(row["nodeid"])
        elif row["phase"] in ("setup", "teardown") and not row.get("is_subtest"):
            passed.add((row["nodeid"], row["phase"]))
    return {node: reason for node, reason in reasons.items()
            if node not in spoiled and {(node, "setup"), (node, "teardown")} <= passed}


def error_text(row) -> str:
    """The error type, and for a lost worker the timeout callback its own record shows was invoked."""
    seconds = row.get("timeout_callback_seconds")
    return str(row.get("error_type") or "") + (
        f"; timeout callback invoked ({seconds:g}s), worker subsequently lost" if type(seconds) in (int, float) else "")


def attempt_text(row) -> str:
    """Phase and error of a report row or of a retry attempt (scripts/ci_retry.py)."""
    return ", ".join(str(part) for part in (row.get("phase"), error_text(row)) if part)


def final(root: Path) -> tuple[dict, str]:
    """The session's final projection, or the reason it cannot be trusted."""
    try:
        data = json.loads((Path(root) / "results.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, "result projection unavailable"
    reports = data.get("reports") if isinstance(data, dict) else None
    if isinstance(reports, list) and all(map(valid, reports)):
        return data, ""
    return {}, "invalid result projection"


def journal_rows(root: Path) -> list[dict] | None:
    """The rows of a session's incremental journal (events.jsonl) that name a node; None without one.

    A kill can tear the last line; unparsable rows are skipped.
    """
    try:
        text = (Path(root) / "events.jsonl").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    rows = []
    for line in text.split("\n"):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and isinstance(row.get("nodeid"), str):
            rows.append(row)
    return rows


def journal_projection(root: Path) -> dict | None:
    """A killed session's stand-in for its export: the lane blocks its journal recorded at
    collection (with that session's identity and system) and the reports it got to; None
    when the journal names no lane. Its exit status is unknown."""
    projection = None
    for row in journal_rows(root) or []:
        if (row.get("event") == "lane" and isinstance(row.get("block"), str)
                and isinstance(row.get("lane"), dict) and isinstance(row.get("github"), dict)):
            projection = projection or {"session_exit_code": None, "reports": [], "github": row["github"],
                                        "environment": row.get("environment") or {}}
            projection[row["block"]] = row["lane"]
        elif row.get("event") == "report" and valid(row) and projection is not None:
            projection["reports"].append(row)
    return projection


def _session_reports(root: Path) -> list[dict]:
    """One session's reports: its final export's, else its journal's valid report rows."""
    data, error = final(root)
    if not error:
        return data["reports"]
    return [row for row in journal_rows(root) or [] if row.get("event") == "report" and valid(row)]


def platform_name(value: str) -> str:
    """linux, darwin or windows, from `sys.platform` or `platform.system()`; anything else lowercased."""
    value = str(value).lower()
    if value.startswith("linux"):
        return "linux"
    return "windows" if value.startswith(("win", "cygwin")) else value


def system_of(data: dict) -> str:
    environment = data.get("environment") if isinstance(data.get("environment"), dict) else {}
    return platform_name(environment.get("platform", ""))


def _table_shape_holds(table) -> bool:
    def milliseconds(value):
        return isinstance(value, dict) and all(isinstance(key, str) and type(count) is int and count >= 0
                                               for key, count in value.items())

    def per_system(value):
        return isinstance(value, dict) and all(milliseconds(files) for files in value.values())
    source = table.get("source") if isinstance(table, dict) else None
    return bool(isinstance(source, dict) and table.get("format") == 1 and isinstance(source.get("run_id"), str)
                and source["run_id"] and per_system(table.get("platforms")) and milliseconds(table.get("ui_nodes", {})))


def durations_table() -> tuple[dict | None, str]:
    """(the run's weight table, "") from $OUROBOROS_CI_DURATIONS, or (None, why it is not used).

    Unset is no table and no reason. A table that cannot be read or has another shape is a reason, and the
    lanes fall back to their positional weights. Only this table may weigh a whole-lane proof's shards: every
    session of a run reads the same copy, so their packing agrees; the owner's table (`recorded_durations_table`)
    changes while the run goes on.
    """
    return _read_table(os.environ.get(DURATIONS_ENV))


def recorded_durations_table() -> tuple[dict | None, str]:
    """The run's table (`durations_table`), else the owner's persisted one ($OUROBOROS_TEST_DURATIONS): what sizes
    and orders the xdist pass's serial groups (tests/conftest.py, scripts/run_tests.py), never a lane's shards.
    An owner's table that no session has written yet is no table and no reason."""
    if os.environ.get(DURATIONS_ENV):
        return durations_table()
    path = os.environ.get(TEST_DURATIONS_ENV)
    return _read_table(path) if path and Path(path).exists() else (None, "")


def _read_table(path) -> tuple[dict | None, str]:
    if not path:
        return None, ""
    try:
        table = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return None, f"the durations table is unreadable ({type(error).__name__})"
    if not _table_shape_holds(table):
        return None, "the durations table has another shape"
    return table, ""


def session_file_durations(reports) -> dict:
    """Milliseconds per file of one session's reports: the setup, call and teardown time of every test, subtests
    excluded (their time lies within their test's call). A file with a crashed worker's report is left out: its
    time is not whole."""
    files, broken = {}, set()
    for row in reports:
        path = str(row.get("nodeid", "")).split("::", 1)[0]
        if row.get("phase") == "crash":
            broken.add(path)
            continue
        seconds = row.get("duration_seconds")
        if row.get("is_subtest") or row.get("phase") not in PHASES or type(seconds) not in (int, float):
            continue
        seconds = max(seconds, 0) if math.isfinite(seconds) else 0
        files[path] = files.get(path, 0) + seconds
    return {path: round(files[path] * 1000) for path in sorted(files) if path not in broken}


def _complete_times(reports) -> dict:
    """Seconds per node of the tests one session completed: setup, call and teardown, subtests excluded.

    A test is complete once its setup passed and its call and teardown are reported, whatever those two's
    outcomes: a red phase took its time too. A failed or skipped setup ran no body (pytest still reports a
    teardown), so its short time must not stand for the execution a retry really makes. One a crashed worker's
    report names, or one the session left before its teardown (stopped or killed while it ran), has no whole
    time."""
    seconds, seen, broken = {}, set(), set()
    for row in reports:
        node = row["nodeid"]
        if row["phase"] == "crash":
            broken.add(node)
        elif not row.get("is_subtest"):
            if row["phase"] != "setup" or row["outcome"] == "passed":
                seen.add((node, row["phase"]))
            value = row.get("duration_seconds")
            value = max(value, 0) if type(value) in (int, float) and math.isfinite(value) else 0
            seconds[node] = seconds.get(node, 0) + value
    return {node: value for node, value in seconds.items()
            if node not in broken and all((node, phase) in seen for phase in PHASES)}


def record_session_durations(path, reports, *, system: str, rootpath, run_id: str, sha: str = "") -> str:
    """Fold one session's file times into the owner's table at ``path`` (format 1, the shape `durations_table`
    reads): the files this session ran replace their entries for this system, entries for files absent from
    ``rootpath`` are dropped, other systems' entries stay, and an unreadable or misshapen table starts afresh
    (a table with the retired `serial` part loses it). Returns "" or why nothing was written."""
    files = session_file_durations(reports)
    if not files:
        return "no whole test file ran"
    path, system = Path(path), platform_name(system)
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        table = None
    if not _table_shape_holds(table):
        table = {"format": 1, "platforms": {}, "source": {}, "ui_nodes": {}}
    table.pop("serial", None)  # no producer can fill it (ids stay plain) and its reader, `serial_shards`, is gone
    platforms = table["platforms"].setdefault(system, {})
    platforms.update(files)
    for part in (platforms,):
        for name in [name for name in part if not (Path(rootpath) / name).exists()]:
            del part[name]
        ordered = {name: part[name] for name in sorted(part)}
        part.clear()
        part.update(ordered)
    table["source"] = {"run_id": run_id, "sha": sha}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.with_name(path.name + ".tmp")
        staging.write_text(json.dumps({key: table[key] for key in sorted(table)}, indent=1, sort_keys=True) + "\n",
                           encoding="utf-8")
        os.replace(staging, path)
    except OSError as error:
        return f"the durations table could not be written ({type(error).__name__}: {error})"
    return ""


def verdict_shape(verdict) -> bool:
    """A retry verdict as scripts/ci_retry.py writes it, as far as its readers rely on it."""
    def rows(value, key="nodeid"):
        return isinstance(value, list) and all(isinstance(row, dict) and isinstance(row.get(key), str)
                                               for row in value)
    return (isinstance(verdict, dict) and verdict.get("format") == 1 and verdict.get("verdict") in VERDICTS
            and isinstance(verdict.get("github"), dict) and isinstance(verdict.get("rounds"), list)
            and bool(verdict["rounds"])
            and all(isinstance(row, dict) and isinstance(row.get("dir"), str)
                    and (row["dir"] == "." or ROUND_DIR.fullmatch(row["dir"])) for row in verdict["rounds"])
            and rows(verdict.get("flaky")) and rows(verdict.get("failed")) and rows(verdict.get("accepted"))
            and all(type(row.get("issue")) is int for row in verdict["accepted"])
            and isinstance(verdict.get("not_run"), list)
            and all(isinstance(node, str) for node in verdict["not_run"]))


def verdict_counts(verdict: dict) -> dict[str, int]:
    """Flaky, accepted, blocking failed and never-run tests; `failed` lists accepted ones too."""
    accepted = {row["nodeid"] for row in verdict["accepted"]}
    return {"flaky": len(verdict["flaky"]), "accepted": len(accepted), "not_run": len(verdict["not_run"]),
            "failed": len({row["nodeid"] for row in verdict["failed"]} - accepted)}


def verdict_passes(verdict: dict) -> bool:
    """Green, flaky or accepted, and its own lists agree: nothing failed unaccepted, nothing left unrun."""
    counts = verdict_counts(verdict)
    return verdict["verdict"] != "red" and not counts["failed"] and not counts["not_run"]


def verdict_file(root: Path) -> tuple[dict | None, str]:
    """(the retry verdict beside a pass's projection, why it cannot be read); (None, "") without one."""
    path = Path(root) / "verdict.json"
    if not path.exists():
        return None, ""
    try:
        verdict = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, "its retry verdict is unreadable"
    return (verdict, "") if verdict_shape(verdict) else (None, "its retry verdict is invalid")


def _ui_times(directory: Path, data: dict) -> dict:
    """Seconds per node of a UI pass: each node of its assignment once, by its first complete execution.

    The pass's own session comes first, then the retry rounds in the order its verdict lists them
    (scripts/ci_retry.py): a test the session stopped or was killed before reaching first runs in a
    round, and counts there. A retry of a test that completed before adds nothing, and a test that
    never completed stays out. A round's test outside the assignment is another shard's."""
    lane = data["ui_browser"]
    assigned = lane.get("assigned", lane.get("full"))
    if not (isinstance(assigned, list) and all(isinstance(node, str) for node in assigned)):
        return {}
    wanted, times = set(assigned), {}
    verdict, _ = verdict_file(directory)
    rounds = [row["dir"] for row in verdict["rounds"] if row["dir"] != "."] if verdict else []
    for reports in [data["reports"]] + [_session_reports(directory / name) for name in rounds]:
        for node, seconds in _complete_times(reports).items():
            if node in wanted:
                times.setdefault(node, seconds)
    return times


def collect_durations(root: Path, *, run_id: str, sha: str) -> dict:
    """One run's weight table for the shards of later runs (tests/default_lane.py, tests/browser_lane.py).

    Integer milliseconds, keys sorted: per test FILE and system, the setup, call and
    teardown time of every report of the default lanes' passes (`platforms`); per node,
    the time of the UI lane's first complete execution of each node (`ui_nodes`,
    `_ui_times`: a continuation round's included). A pass is a projection carrying a
    `default_parallel`, `default_serial` or `ui_browser` block, or an unsharded `parallel`
    or `serial` pass directory without one (run_tests.py's battery): the caller decides
    which artifacts lie under root. A UI session killed before its export is the lane
    block its journal recorded (`journal_projection`). The default lanes' retry rounds,
    collections without reports, projections of an unknown system and subtest reports
    (their time lies within their test's call) are left out. Of one pass (system, lane,
    shard) the highest run attempt counts, then the larger projection (for UI, the more
    nodes timed), then the first path, so a projection downloaded twice counts once.
    """
    chosen = {}
    for directory, dirs, files in os.walk(root):
        dirs[:] = sorted(name for name in dirs if not ROUND_DIR.fullmatch(name))
        directory = Path(directory)
        if "results.json" in files:
            data, error = final(directory)
            if error:
                continue
        else:
            data = journal_projection(directory) if "events.jsonl" in files else None
            if data is None or not isinstance(data.get("ui_browser"), dict):
                continue
        if system_of(data) not in SYSTEMS:
            continue
        lane = next((key for key in ("default_parallel", "default_serial", "ui_browser")
                     if isinstance(data.get(key), dict)), None)
        if lane is None and directory.name not in ("parallel", "serial"):
            continue
        measured = _ui_times(directory, data) if lane == "ui_browser" else data["reports"]
        if not measured:
            continue
        identity = data.get("github") if isinstance(data.get("github"), dict) else {}
        attempt = identity.get("run_attempt")
        rank = (int(attempt) if isinstance(attempt, str) and attempt.isdecimal() else 0, len(measured))
        key = (system_of(data), lane or directory.name, str(data[lane].get("shard")) if lane else "")
        if key not in chosen or rank > chosen[key][0]:
            chosen[key] = (rank, measured)
    files, nodes = {}, {}
    for (name, lane, _), (_, measured) in chosen.items():
        if lane == "ui_browser":
            for node, seconds in measured.items():
                nodes[node] = nodes.get(node, 0) + seconds
            continue
        for row in measured:
            seconds = row.get("duration_seconds")
            if row.get("is_subtest") or type(seconds) not in (int, float) or not math.isfinite(seconds):
                seconds = 0
            path = row["nodeid"].split("::", 1)[0]
            totals = files.setdefault(name, {})
            totals[path] = totals.get(path, 0) + max(seconds, 0)

    def milliseconds(totals):
        return {key: round(totals[key] * 1000) for key in sorted(totals)}
    return {"format": 1, "platforms": {name: milliseconds(files[name]) for name in sorted(files)},
            "source": {"run_id": run_id, "sha": sha}, "ui_nodes": milliseconds(nodes)}


def durations_command(args) -> int:
    table = collect_durations(args.evidence_root, run_id=args.run_id, sha=args.sha)
    write_json(args.out, table)
    counts = [f"{name} {len(files)} files" for name, files in table["platforms"].items()]
    print(f"DURATIONS run {args.run_id}: " + ", ".join(counts + [f"{len(table['ui_nodes'])} UI nodes"]))
    if not counts and not table["ui_nodes"]:
        print("::warning::no default or UI lane pass was found; the table weighs nothing.")
    return 0


def run_verdicts(root: Path) -> list[tuple[str, dict | None]]:
    """(label, verdict or None when unreadable) of every pass beneath root, each pass once.

    The label is the pass directory's path below root. Two artifacts of one job can
    carry the same pass; a byte-identical verdict is listed once. A pass re-run in a
    later attempt of the run counts only in its highest attempt, as reconciliation
    counts it: its artifact name differs only in `-<run_id>-<attempt>`.
    """
    found, chosen, seen = [], {}, set()
    for directory, dirs, files in os.walk(root):
        dirs[:] = sorted(name for name in dirs if not ROUND_DIR.fullmatch(name))
        if "verdict.json" not in files:
            continue
        label = Path(directory).relative_to(root).as_posix()
        try:
            payload = (Path(directory) / "verdict.json").read_bytes()
            verdict = json.loads(payload.decode("utf-8"))
        except (OSError, ValueError):
            found.append((label, None))
            continue
        if payload in seen:
            continue
        seen.add(payload)
        if not verdict_shape(verdict):
            found.append((label, None))
            continue
        run_id, attempt = (str(verdict["github"].get(name) or "") for name in ("run_id", "run_attempt"))
        key, rank = label, 0
        if run_id and attempt.isdecimal():
            key, rank = label.replace(f"-{run_id}-{attempt}/", f"-{run_id}-*/", 1), int(attempt)
        if key not in chosen or rank > chosen[key][0]:
            chosen[key] = (rank, label, verdict)
    return sorted(found + [(label, verdict) for _, label, verdict in chosen.values()], key=lambda item: item[0])


SECTIONS = (
    ("flaky", "FLAKY", "failed, then passed on its one retry; does not block"),
    ("accepted", "ACCEPTED", "failed; its open issue carries the owner's not-a-product-bug label; does not block"),
    ("failed", "FAILED", "failed twice, or its retry never ran; blocks"),
    ("not_run", "NOT RUN", "assigned, never executed; blocks"),
)


def render_run_verdict(root: Path, *, repo: str = "", required=()) -> tuple[str, dict]:
    """(markdown section, JSON facts) of every retry verdict of one run, complete and uncapped.

    `failed` lists only blocking failures; an accepted one is listed once, as accepted.
    A `required` lane is a pattern over labels; one that no verdict matches is reported.
    """
    rows = {key: [] for key, _, _ in SECTIONS}
    lanes = []
    for label, verdict in run_verdicts(root):
        if verdict is None:
            lanes.append({"label": label, "verdict": "unreadable", "rounds": 0, "reason": ""})
            continue
        continued = verdict.get("continued")
        lanes.append({"label": label, "verdict": verdict["verdict"], "rounds": len(verdict["rounds"]),
                      "reason": "; ".join(filter(None, [
                          f"{continued} test(s) a stopped session left ran in a later round"
                          if type(continued) is int and continued else "", str(verdict.get("reason") or "")]))})
        issues = {row["nodeid"]: row["issue"] for row in verdict["accepted"]}
        rows["flaky"] += [{"label": label, "nodeid": row["nodeid"],
                           **(row["first"] if isinstance(row.get("first"), dict) else {})} for row in verdict["flaky"]]
        rows["accepted"] += [{"label": label, "nodeid": node, "issue": issue} for node, issue in issues.items()]
        rows["failed"] += [{"label": label, "nodeid": row["nodeid"], "attempts": row.get("attempts") or []}
                           for row in verdict["failed"] if row["nodeid"] not in issues]
        rows["not_run"] += [{"label": label, "nodeid": node} for node in verdict["not_run"]]
    missing = [pattern for pattern in required
               if not any(fnmatch.fnmatchcase(lane["label"], pattern) for lane in lanes)]

    def detail(key, row):
        if key == "flaky":
            first = attempt_text(row)
            return f" (first: {cell(first)})" if first else ""
        if key == "accepted":
            link = f"https://github.com/{repo}/issues/{row['issue']}" if repo else ""
            return f" — [#{row['issue']}]({link})" if link else f" — #{row['issue']}"
        if key == "failed":
            tries = "; ".join(attempt_text(attempt) for attempt in row["attempts"] if isinstance(attempt, dict))
            return f" ({cell(tries)})" if tries else ""
        return ""
    lines = ["## CI test verdicts — " + ", ".join(f"{name} {len(rows[key])}" for key, name, _ in SECTIONS), ""]
    if lanes:
        lines += ["| Job / pass | Verdict | Rounds | Note |", "| --- | --- | --- | --- |"]
        lines += [f"| {cell(lane['label'])} | {cell(lane['verdict'])} | {lane['rounds']} | {cell(lane['reason'])} |"
                  for lane in lanes]
    else:
        lines.append("No retry verdict was found beneath the evidence root.")
    lines += [f"\n**NO VERDICT** for the required lane `{cell(pattern)}`." for pattern in missing]
    for key, name, meaning in SECTIONS:
        if not rows[key]:
            continue
        lines += ["", f"### {name} ({len(rows[key])}) — {meaning}"]
        current = None
        for row in rows[key]:
            if row["label"] != current:
                current = row["label"]
                lines += ["", f"**{cell(current)}**", ""]
            lines.append(f"- {cell(row['nodeid'])}{detail(key, row)}")
    return "\n".join(lines) + "\n", {**rows, "lanes": lanes, "missing": missing}


def run_verdict_command(args) -> int:
    """Diagnostic: the verdicts themselves decided their jobs; nothing here fails a step."""
    try:
        text, facts = render_run_verdict(args.evidence_root, repo=os.environ.get("GITHUB_REPOSITORY", ""),
                                         required=args.required_lanes)
    except Exception as error:
        print(f"::warning::the run's test verdicts are unavailable ({type(error).__name__}).")
        return 0
    try:
        write_json(args.out, facts)
    except OSError as error:
        print(f"::warning::{args.out} is unavailable ({type(error).__name__}).")
    try:
        with args.summary.open("a", encoding="utf-8") as handle:
            handle.write(text)
    except OSError as error:
        print(f"::warning::the verdict summary is unavailable ({type(error).__name__}).")
    sys.stdout.buffer.write(text.encode("utf-8", "replace"))
    sys.stdout.flush()
    return 0
