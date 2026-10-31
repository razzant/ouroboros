"""Retry verdicts, recorded durations and a run's verdict list: tests/ci_evidence.py's stdlib CLI half.

scripts/ci_retry.py writes a pass's verdict.json beside its projection (results.json and
events.jsonl, tests/ci_evidence.py) and runs each retry round in `retry-<k>` beside them.
Here is what both need to read them: a report row and a session's final export, a
verdict and whether it lets a red session pass, the run's weight table the lane plugins
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
import zlib

PHASES = frozenset({"setup", "call", "teardown"})
# scripts/ci_retry.py: retry round k of a pass runs in <pass>/retry-<k>.
ROUND_DIR = re.compile(r"retry-[1-9][0-9]*")
VERDICTS = frozenset({"green", "flaky", "accepted", "red"})
SYSTEMS = frozenset({"linux", "darwin", "windows"})
# The run's weight table (`collect_durations`); the lane plugins read it at collection.
DURATIONS_ENV = "OUROBOROS_CI_DURATIONS"
# The xdist group of a serial test in run_tests.py's one-run battery (`serial_group`).
SERIAL_GROUP = re.compile(r"serial[0-9]+")


def serial_group(path: str, shards: int) -> str:
    """The serial group a test FILE runs in when tests/conftest.py pins ``shards`` of them: a stable hash, no seed."""
    return f"serial{zlib.crc32(path.encode('utf-8')) % shards}"


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
            and ("is_subtest" not in row or type(row["is_subtest"]) is bool))


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


def platform_name(value: str) -> str:
    """linux, darwin or windows, from `sys.platform` or `platform.system()`; anything else lowercased."""
    value = str(value).lower()
    if value.startswith("linux"):
        return "linux"
    return "windows" if value.startswith(("win", "cygwin")) else value


def system_of(data: dict) -> str:
    environment = data.get("environment") if isinstance(data.get("environment"), dict) else {}
    return platform_name(environment.get("platform", ""))


def durations_table() -> tuple[dict | None, str]:
    """(the run's weight table, "") from $OUROBOROS_CI_DURATIONS, or (None, why it is not used).

    Unset is no table and no reason. A table that cannot be read or has another
    shape is a reason, and the lanes fall back to their positional weights.
    """
    path = os.environ.get(DURATIONS_ENV)
    if not path:
        return None, ""

    def milliseconds(value):
        return isinstance(value, dict) and all(isinstance(key, str) and type(count) is int and count >= 0
                                               for key, count in value.items())

    def per_system(value):
        return isinstance(value, dict) and all(milliseconds(files) for files in value.values())
    try:
        table = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return None, f"the durations table is unreadable ({type(error).__name__})"
    source = table.get("source") if isinstance(table, dict) else None
    if not (isinstance(source, dict) and table.get("format") == 1 and isinstance(source.get("run_id"), str)
            and source["run_id"] and per_system(table.get("platforms")) and per_system(table.get("serial", {}))
            and milliseconds(table.get("ui_nodes", {}))):
        return None, "the durations table has another shape"
    return table, ""


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


def collect_durations(root: Path, *, run_id: str, sha: str) -> dict:
    """One run's weight table for the shards of later runs (tests/default_lane.py, tests/browser_lane.py).

    Integer milliseconds, keys sorted: per test FILE and system, the setup, call and
    teardown time of every report of the default lanes' passes (`platforms`) and the part
    of it the serial tests took (`serial`, scripts/run_tests.py sizes its serial groups
    by it); per node, the same for the UI lane (`ui_nodes`). A pass is a projection
    carrying a `default_parallel`, `default_serial` or `ui_browser` block, or an
    unsharded `parallel`, `serial` or `overlapped` pass directory without one (the last is
    run_tests.py's one-run battery, whose serial tests carry a `@serial<N>` group): the
    caller decides which artifacts lie under root. Retry rounds, collections without
    reports, projections of an unknown system and subtest reports (their time lies within
    their test's call) are left out. Of one pass (system, lane, shard) the highest run
    attempt counts, then the larger projection, then the first path, so a projection
    downloaded twice counts once.
    """
    chosen = {}
    for directory, dirs, files in os.walk(root):
        dirs[:] = sorted(name for name in dirs if not ROUND_DIR.fullmatch(name))
        if "results.json" not in files:
            continue
        data, error = final(Path(directory))
        if error or not data["reports"] or system_of(data) not in SYSTEMS:
            continue
        lane = next((key for key in ("default_parallel", "default_serial", "ui_browser")
                     if isinstance(data.get(key), dict)), None)
        if lane is None and Path(directory).name not in ("parallel", "serial", "overlapped"):
            continue
        identity = data.get("github") if isinstance(data.get("github"), dict) else {}
        attempt = identity.get("run_attempt")
        rank = (int(attempt) if isinstance(attempt, str) and attempt.isdecimal() else 0, len(data["reports"]))
        key = (system_of(data), lane or Path(directory).name, str(data[lane].get("shard")) if lane else "")
        if key not in chosen or rank > chosen[key][0]:
            chosen[key] = (rank, data)
    files, serial, nodes = {}, {}, {}
    for (name, lane, _), (_, data) in chosen.items():
        for row in data["reports"]:
            seconds = row.get("duration_seconds")
            if row.get("is_subtest") or type(seconds) not in (int, float) or not math.isfinite(seconds):
                seconds = 0
            if lane == "ui_browser":
                nodes[row["nodeid"]] = nodes.get(row["nodeid"], 0) + max(seconds, 0)
                continue
            path = row["nodeid"].split("::", 1)[0]
            parts = [files] + ([serial] if lane in ("default_serial", "serial") or (
                lane == "overlapped" and SERIAL_GROUP.fullmatch(row["nodeid"].rpartition("@")[2])) else [])
            for part in parts:
                totals = part.setdefault(name, {})
                totals[path] = totals.get(path, 0) + max(seconds, 0)

    def milliseconds(totals):
        return {key: round(totals[key] * 1000) for key in sorted(totals)}
    return {"format": 1, "platforms": {name: milliseconds(files[name]) for name in sorted(files)},
            "serial": {name: milliseconds(serial[name]) for name in sorted(serial)},
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
