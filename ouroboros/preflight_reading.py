"""Read the local commit gate's blocking pass as evidence, never as a second verdict.

The running generation imports this adapter at bootstrap. Its only per-attempt
inputs are data from the gate's own retained capture and inventory; CI's retry
and acceptance policy remains with ci_retry.
"""
from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath
import re
import stat

from ouroboros.preflight_case_facts import (
    attempt_text, case_key, observed_outcomes, valid, valid_final,
)
from ouroboros.redaction import redact_projection

MAX_LISTED = 50
_CLASSES = ("leak", "worker_lost", "collection_error", "failed", "in_flight", "not_run")
_CASE_KEY = re.compile(r"[0-9a-f]{32}\Z")


def _label(value: str) -> str:
    # Keep the case's name, not a private absolute source path, parameter path,
    # or line breaks which would look like an additional finding.
    text = str(redact_projection(value).value)
    module, sep, rest = text.partition("::")
    if Path(module).is_absolute() or PureWindowsPath(module).is_absolute():
        module = PureWindowsPath(module).name
    text = module + sep + rest
    return " ".join(re.sub(r"(?<![\w:/])(?:[A-Za-z]:[\\/]|/(?!/)|\\\\)[^\s\]<>]+", "<path>", text).splitlines())


def _key(value) -> bool:
    return isinstance(value, str) and _CASE_KEY.fullmatch(value) is not None


def _file(directory: Path, name: str, root: Path) -> bool:
    path = directory / name
    try:
        return (path.resolve() == path and path.is_relative_to(root)
                and stat.S_ISREG(path.lstat().st_mode))
    except (OSError, RuntimeError, ValueError):
        return False


def _read_journal(path: Path):
    """Keep positive rows from a readable prefix; only the producer can close start coverage."""
    rows, damaged, session, last = [], False, None, None
    with path.open(encoding="utf-8", errors="replace") as stream:
        for index, line in enumerate(stream):
            try:
                last = json.loads(line)
            except ValueError:
                damaged = True
                continue
            if not isinstance(last, dict):
                damaged = True
                continue
            if not line.endswith("\n") or "\ufffd" in line:
                damaged = True
            if index == 0 and last.get("event") == "session" and isinstance(last.get("session"), str):
                session = last["session"]
            if last.get("event") in ("start", "finish") and not (
                    _key(last.get("case_key")) and isinstance(last.get("nodeid"), str)):
                damaged = True
            if isinstance(last.get("nodeid"), str):
                rows.append(last)
    complete = not damaged and bool(session) and last == {"event": "starts_complete", "session": session}
    return rows, session, complete


def _read_session(directory, root: Path, exit_code, recorded):
    """A usable final export outranks journal report rows; retain the journal's starts and finishes."""
    if directory is None:
        return None, None, False, {}, "no pass-owned pytest evidence"
    # The enclosing FileOutput directory is fresh for each gate invocation.
    # CI owns a different source policy for its possibly reused output roots.
    data, journal, journal_session, complete = None, None, None, False
    error = "no final export"
    if _file(directory, "results.json", root):
        try:
            record = recorded if recorded is not None else json.loads((directory / "results.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            error = "unreadable final export"
        else:
            if (valid_final(record) and type(record.get("session_exit_code")) is int
                    and exit_code in (None, record["session_exit_code"])):
                data, error = record, ""
            else:
                error = "invalid final export or exit mismatch"
    if _file(directory, "events.jsonl", root):
        try:
            journal, journal_session, complete = _read_journal(directory / "events.jsonl")
        except OSError:
            pass  # A usable final export stands on its own.
    if data is not None and journal_session and data.get("session") != journal_session:
        data, complete, error = None, False, "pytest sources describe different sessions"
    sources = {key: (directory / name).relative_to(root).as_posix() for key, name in
               (("results", "results.json"), ("events", "events.jsonl")) if
               (data is not None if key == "results" else journal is not None)}
    return data, journal, complete, sources, error


def _case_rows(data, journal):
    reports = data["reports"] if data is not None else [row for row in journal or ()
                                                       if row.get("event") == "report" and valid(row)]
    keyed = [row for row in reports if _key(row.get("case_key"))]
    started, active = {row["case_key"]: row["nodeid"] for row in keyed}, {}
    for row in journal or ():
        key = row.get("case_key")
        if not _key(key):
            continue
        if row.get("event") == "start":
            active[key] = started[key] = row["nodeid"]
        elif row.get("event") == "finish":
            started[key] = row["nodeid"]
            active.pop(key, None)
        elif row.get("event") == "report" and row.get("phase") == "crash" and valid(row):
            active.pop(key, None)
    for row in keyed:
        if row["phase"] == "crash":
            active.pop(row["case_key"], None)
    outcomes = observed_outcomes((row for row in keyed if row["phase"] != "crash"),
                                identity=lambda row: row["case_key"])
    return keyed, started, outcomes, active, len(keyed) == len(reports)


def _assignment(data, inventory, salt):
    if inventory is not None:
        if salt and inventory.get("assignment") == "known":
            return [(case_key(node, salt), node) for node in dict.fromkeys(inventory["expected"])], "inventory"
        # The serial controller can positively record an empty collection even
        # when the selected inventory has no row for it, but never override a
        # disagreement or an unreadable inventory.
        if (inventory.get("assignment") != "unobserved" or data is None or data.get("collection_failures")
                or data.get("session_exit_code") not in (0, 5) or data.get("precommit_assignment") != []):
            return None, None
    assigned = (data or {}).get("precommit_assignment")
    if isinstance(assigned, list) and all(isinstance(row, list) and len(row) == 2 and _key(row[0])
                                          and isinstance(row[1], str) for row in assigned):
        return list(dict(assigned).items()), "results"
    return None, None


def _case_findings(data, journal, starts_complete, inventory, salt):
    keyed, started, outcomes, active, identities_known = _case_rows(data, journal)
    failed, workers = [], []
    for key, row in outcomes.items():
        if row["outcome"] == "failed":
            failed.append({"nodeid": _label(started[key]), "attempt": _label(attempt_text(row))})
    # A controller can record a failed case and a later worker loss on the same
    # case. Neither fact replaces the other. stdout is never evidence of loss.
    for key, crash in {row["case_key"]: row for row in keyed if row["phase"] == "crash"
                      and row.get("error_type") == "WorkerCrash"}.items():
        workers.append({"worker": _label(str(crash.get("worker") or "")), "nodeid": _label(started[key]),
                        "attempt": _label(attempt_text(crash))})
    dead = (data or {}).get("worker_crash")
    if isinstance(dead, dict) and isinstance(dead.get("worker"), str) and dead["worker"]:
        label = _label(dead["worker"])
        if not any(row["worker"] == label for row in workers):
            workers.append({"worker": label, "nodeid": None, "attempt": "no test identified"})
    collected = (data or {}).get("collection_failures")
    collection = [row["nodeid"] for row in collected or () if isinstance(row, dict)
                  and isinstance(row.get("nodeid"), str)] if isinstance(collected, list) else []
    collection.extend(row["nodeid"] for row in journal or () if row.get("event") == "collection_error")
    assigned, assigned_from = _assignment(data, inventory, salt)
    # A prefix proves observed starts/reports, never their absence. A positively
    # empty assignment needs no execution join.
    if salt and inventory and inventory.get("assignment") == "known":
        for kind in ("executed", "skipped", "failed"):
            started.update((case_key(node, salt), node) for node in inventory.get(kind, ()))
    unrun = None
    if assigned == [] or assigned is not None and identities_known and starts_complete and not workers:
        unrun = [_label(node) for key, node in assigned if key not in started]
    in_flight = list(map(_label, active.values())) if active or starts_complete else None
    return {"failed": failed, "worker_lost": workers, "collection_error": list(map(_label, dict.fromkeys(collection))),
            "in_flight": in_flight, "not_run": unrun}, (assigned_from if unrun else None), identities_known


def block(reason: str, *, label=None, exit_code=None, leak=False, gap="pre-commit reader unavailable") -> dict:
    """Host-owned facts remain available even when pytest diagnostics do not."""
    unknown = dict.fromkeys(name for name in _CLASSES if name != "leak")
    return {"cause": reason, "pass": label, "exit": exit_code, "attempts": int(label is not None),
            "coverage": "unavailable", "gap": gap, **unknown, "counts": unknown |
            {"leak": int(leak)}, "leak": [label or "gate"] if leak else [],
            "evidence": {"gate": "result.json"}, "omitted": {}}


def observe(root: Path, result, label: str, reason: str, inventory: dict | None, *, recorded=None) -> dict:
    """A bounded, display-safe projection of one pass; source filenames are run-relative."""
    exit_code, _, leak = result
    reading = block(reason, label=label, exit_code=exit_code, leak=bool(leak))
    try:
        directory = getattr(result, "session_directory", None)
        data, events, starts_complete, sources, gap = _read_session(directory, root, exit_code, recorded)
        reading["gap"] = gap
        known_empty = inventory and inventory.get("assignment") == "known" and inventory.get("expected") == []
        if data is None and not events and not (starts_complete or known_empty):
            return reading
        rows, assigned_from, identities_known = _case_findings(
            data, events, starts_complete, inventory, getattr(result, "case_salt", b""))
        if assigned_from == "inventory":
            sources["inventory"] = f"inventory-{label}.jsonl"
            # The outer lane record is author-only. It holds the join key so the
            # owner can reconstruct distinct omitted ids from the raw inventory.
            if directory is not None and _file(directory.parent, "result.json", root):
                sources["capture"] = (directory.parent / "result.json").relative_to(root).as_posix()
        coverage = "final" if data is not None and identities_known else "partial"
        if not identities_known:
            gap = "some reports had no distinct case identity"
        elif not gap and not starts_complete:
            gap = "pytest start coverage incomplete"
        reading.update(coverage=coverage, gap=gap,
                       evidence={**reading["evidence"], **sources})
        for name, values in rows.items():
            reading[name] = None if values is None else values[:MAX_LISTED]
            reading["counts"][name] = (None if values is None or coverage == "partial" and not values
                                       and name != "not_run" else len(values))
            if values is not None and len(values) > MAX_LISTED:
                reading["omitted"][name] = len(values) - MAX_LISTED
    except (OSError, ValueError, TypeError, KeyError, RuntimeError):
        return block(reason, label=label, exit_code=exit_code, leak=bool(leak), gap="unreadable pytest evidence")
    return reading


def _lines(kind: str, values):
    if kind == "leak":
        return [f"LEAK {value}: child cleanup was not confirmed" for value in values]
    if kind == "worker_lost":
        return [f"WORKER LOST {row['worker'] or 'unnamed worker'}"
                + (f" running {row['nodeid']}" if row["nodeid"] else "") + f" ({row['attempt']})" for row in values]
    if kind == "failed":
        return [f"FAILED {row['nodeid']} ({row['attempt']}; one attempt, no retry)" for row in values]
    return [f"{kind.replace('_', ' ').upper()} {value}" for value in values]


def render(reading: dict, max_output: int) -> str:
    """Use at most one quarter of the message, show causes before a long list of consequences."""
    room = max_output // 4
    if room <= 0:
        return ""
    if reading["coverage"] == "unavailable" and not reading["leak"]:
        gap = f"Pytest diagnostics unavailable: {reading['gap']}"
        return gap if len(gap) <= room else ""
    counts = reading["counts"]
    lines = [_lines(kind, reading.get(kind) or ()) for kind in _CLASSES]
    # Reserve a summary even if an individual name cannot fit. The source of a
    # raw inventory is author-local; review text only carries relative names.
    where = ", ".join(reading["evidence"].values())
    bound = "≥" if reading["coverage"] == "partial" else ""
    summary = "; ".join(f"{kind.replace('_', ' ').upper()} {bound if kind not in ('leak', 'not_run') else ''}{counts[kind]}"
                        for kind in _CLASSES if counts[kind])
    if counts["not_run"] is None:
        summary += ("; " if summary else "") + "NOT RUN unknown"
    if reading["gap"]:
        summary += ("; " if summary else "") + reading["gap"]
    summary += f" — {where}"
    if len(summary) > room:
        return "See result.json" if room >= len("See result.json") else ""
    observed = sum(count or 0 for count in counts.values())
    selected = []
    reserve = len(f"; {observed} not shown") if observed else 0
    remaining = room - len(summary) - reserve
    for group in lines:
        for line in group:
            if len(line) + 1 <= remaining:
                selected.append(line)
                remaining -= len(line) + 1
    unshown = observed - len(selected)
    suffix = f"; {unshown} not shown" if unshown and len(f"; {unshown} not shown") <= remaining + reserve else ""
    return "\n".join([*selected, summary + suffix])
