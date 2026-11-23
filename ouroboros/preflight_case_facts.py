"""Stdlib-only pytest evidence vocabulary shared by the serving gate and CI.

One reducer decides what an observed report says about a case. The callers own
assignment, absence, retry and verdict policy; a report alone grants none of them.
"""
from __future__ import annotations

import hashlib
import json

PHASES = frozenset({"setup", "call", "teardown"})


def case_key(nodeid: str, salt: bytes) -> str:
    """A pass-local identity survives redaction without revealing raw parameters."""
    return hashlib.blake2b(nodeid.encode("utf-8"), key=salt, digest_size=16).hexdigest()


def valid(row) -> bool:
    return (isinstance(row, dict)
            and all(isinstance(row.get(key), str) for key in ("nodeid", "phase", "outcome"))
            and (row["phase"] in PHASES or row["phase"] == "crash")
            and row["outcome"] in {"passed", "failed", "skipped"}
            and ("is_subtest" not in row or type(row["is_subtest"]) is bool)
            and (row.get("wasxfail") is None or isinstance(row["wasxfail"], str)))


# A dead worker's own facts on its crash row (tests/ci_evidence.py): its gateway and, only for the record
# that worker's pytest-timeout callback left, the timeout. An attempt carries them; nothing here infers one.
CRASH_FACTS = ("worker", "timeout_callback_seconds")


def observed_outcomes(rows, *, identity=None) -> dict[str, dict]:
    """Each executed node's outcome in one session: its first failure, else a skip, else a pass.

    Executed is the lane guard's rule (tests/browser_lane.py LaneReconciliation, as
    tests/ci_evidence.py applies it): a call phase, a crash, or any report that did not pass.
    """
    outcomes = {}
    for row in rows:
        if row["phase"] not in ("call", "crash") and row["outcome"] == "passed":
            continue
        node = identity(row) if identity else row["nodeid"]
        held = outcomes.get(node)
        if row["outcome"] == "failed":
            if held is None or held["outcome"] != "failed":
                outcomes[node] = {"outcome": "failed", "phase": row["phase"],
                                  "error_type": str(row.get("error_type") or ""),
                                  **{key: row[key] for key in CRASH_FACTS if key in row}}
        elif row["outcome"] == "skipped":
            if held is None or held["outcome"] == "passed":
                outcomes[node] = {"outcome": "skipped", "phase": row["phase"], "error_type": "Skipped"}
        elif held is None:
            outcomes[node] = {"outcome": "passed", "phase": row["phase"], "error_type": ""}
    return outcomes


def error_text(row) -> str:
    """The error type, and for a lost worker the timeout callback its own record shows was invoked."""
    seconds = row.get("timeout_callback_seconds")
    return str(row.get("error_type") or "") + (
        f"; timeout callback invoked ({seconds:g}s), worker subsequently lost" if type(seconds) in (int, float) else "")


def attempt_text(row) -> str:
    """Phase and error of a report row or of a retry attempt (scripts/ci_retry.py)."""
    return ", ".join(str(part) for part in (row.get("phase"), error_text(row)) if part)


def valid_final(data) -> bool:
    reports = data.get("reports") if isinstance(data, dict) else None
    return isinstance(reports, list) and all(map(valid, reports))


def journal_reports(text: str) -> list[dict]:
    """Rows with a node from this journal body; a kill can tear the last row."""
    rows = []
    for line in text.split("\n"):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and isinstance(row.get("nodeid"), str):
            rows.append(row)
    return rows
