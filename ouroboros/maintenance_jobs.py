"""Durable facts of named background jobs, one ``meta`` row per job in ``state/current.sqlite``.

Work that grows with accumulated history (a legacy reconcile, an archive backfill, a
promotion retry) is a named job, not a boot or request step. Its facts live here so that
"once" means once per installation, never once per process: a restart reads the job's
row instead of redoing the work, and a negative result ("scanned, nothing found") is a
fact like any other. A job row is a small JSON document under the ``meta`` key
``job:<name>``; every update is one store transaction, so concurrent writers (the server
and the maintenance child) never interleave half-updates. The health document, the
maintenance child and the tests read the same rows.

Row fields are the job's own; the ones every job keeps are ``state`` (one of ``STATES``),
``updated_at`` and, while it advances, ``cursor``/``done``/``total`` (``total`` is
``None`` when the job cannot know its denominator), plus ``error`` for a failed run and ``last_error``
``{class, at}`` while its last run threw (``maintenance.run``: the state is left as it was). A job
that cannot proceed records ``failed`` with its reason and the inputs it waits on; it is
retried when those inputs change, never by an elapsed-time or attempt-count rule.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, Iterable, List, Optional

from ouroboros import store
from ouroboros.utils import utc_now_iso

PREFIX = "job:"
STATES = ("pending", "running", "done", "failed")


def _key(name: str) -> str:
    name = str(name or "").strip()
    if not name or ":" in name:
        raise ValueError(f"invalid job name: {name!r}")
    return PREFIX + name


def job(root: Any, name: str) -> Optional[Dict[str, Any]]:
    """The job's current row, or ``None`` when it never ran on this root."""
    with store.open_store(pathlib.Path(root), write=False) as txn:
        value = txn.meta(_key(name))
    return dict(value) if isinstance(value, dict) else None


def jobs(root: Any, names: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, Any]]:
    """Job rows keyed by job name: every row of the root (one indexed range read of ``meta``), or
    only the named jobs' rows (the dispatcher's look: a row another owner keeps proportional to
    its own inventory, the drive settlement's verdicts, is not decoded for it)."""
    import json

    keys = None if names is None else [_key(name) for name in names]
    if keys is not None and not keys:
        return {}
    with store.open_store(pathlib.Path(root), write=False) as txn:
        if keys is None:
            rows = txn.conn.execute(
                "SELECT key, value FROM meta WHERE key >= ? AND key < ?", (PREFIX, PREFIX[:-1] + ";"),
            ).fetchall()
        else:
            rows = txn.conn.execute(
                f"SELECT key, value FROM meta WHERE key IN ({','.join('?' * len(keys))})", keys).fetchall()
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        value = json.loads(row["value"])
        if isinstance(value, dict):
            out[str(row["key"])[len(PREFIX):]] = value
    return out


def record(root: Any, name: str, **facts: Any) -> Dict[str, Any]:
    """Merge ``facts`` into the job's row in one write transaction; returns the new row."""
    with store.open_store(pathlib.Path(root), write=True) as txn:
        return record_in(txn, name, **facts)


def record_in(txn: store.Txn, name: str, **facts: Any) -> Dict[str, Any]:
    """``record`` inside the caller's write transaction (a job that records the revision its own write leaves)."""
    state = facts.get("state")
    if state is not None and state not in STATES:
        raise ValueError(f"invalid job state: {state!r}")
    key = _key(name)
    current = txn.meta(key)
    row = dict(current) if isinstance(current, dict) else {}
    row.update(facts)
    row["updated_at"] = utc_now_iso()
    txn.set_meta(key, row)
    return row


def forget_in(txn: store.Txn, name: str, *fields: str) -> Dict[str, Any]:
    """Drop ``fields`` a job no longer keeps on its row, inside the caller's write transaction."""
    key = _key(name)
    current = txn.meta(key)
    row = {field: value for field, value in (current.items() if isinstance(current, dict) else ()) if field not in fields}
    txn.set_meta(key, row)
    return row


def is_done(root: Any, name: str) -> bool:
    row = job(root, name)
    return bool(row) and row.get("state") == "done"


# ---- a retirer pass's facts: what it released and what it holds, one shape for every retirer -------------
#
# The server's five retirers (drive settlement, private copies, review checkouts, session scratch, engine
# program copies) and the engine's own retention pass report one shape, the engine's ``GcLastPass`` holds
# (``packages/schema/src/control-gc.ts``): ``{released, holds: [{class, count, example, since}], errors}``.
# A hold is a typed fact about what the pass kept and why (``class``: the retirer's own reason, such as
# ``task_live``, ``child_result_unadopted``, ``owner_keep``, ``task_result_unreadable``), with one example
# and the moment the class was first held in an unbroken run of passes (``since``, carried over while the
# class stays held). Age waits are not holds: a folder inside the retention term is the rule, not a keep.
# The row is written only when the pass is evidence: something released, an error, a hold set that
# differs from the row's, the job's first pass (so a tidy plane never reads "not run") or the first pass
# to return after one threw (``last_error``, which the retirer's wrapper wrote); an unchanged pass writes
# nothing (the rest law, tests/test_cost_contract.py B7), a pass that rested (listed nothing) leaves the row
# as it stands, and a ``partial`` pass (it judged only part of its plane) keeps the row's holds.

HOLD_EXAMPLE_CHARS = 160  # the engine's bound on a hold's example


class Holds:
    """What one pass kept, by class: ``note(class, example)`` per kept item; ``rows()`` the hold list, ONE live
    list kept sorted by class, so a report that holds it at its start carries every later note (a pass that
    returns early still reports what it had judged)."""

    def __init__(self) -> None:
        self._rows: List[Dict[str, Any]] = []

    def note(self, hold_class: Any, example: Any = "") -> None:
        key = str(hold_class or "other")
        for row in self._rows:
            if row["class"] == key:
                row["count"] += 1
                return
        self._rows.append({"class": key, "count": 1, "example": str(example or "")[:HOLD_EXAMPLE_CHARS]})
        self._rows.sort(key=lambda row: row["class"])

    def rows(self) -> List[Dict[str, Any]]:
        return self._rows

    def __len__(self) -> int:
        return sum(row["count"] for row in self._rows)


def pass_report(released: Any, holds: Any, errors: Any = ()) -> Dict[str, Any]:
    """The one shape: ``released`` a count (or the list of what went), ``holds`` a ``Holds`` or rows,
    ``errors`` the pass's own error strings (item facts belong in ``holds``, as their class)."""
    rows = holds.rows() if isinstance(holds, Holds) else [dict(row) for row in (holds or ())]
    return {"released": len(released) if isinstance(released, (list, tuple, set)) else int(released or 0),
            "holds": rows, "errors": [str(error) for error in (errors or ())]}


def _with_since(holds: List[Dict[str, Any]], previous: Any, now: str) -> List[Dict[str, Any]]:
    """Each hold's ``since``: the retirer's own when it knows one (a record's or marker's time), else the
    row's while its class stayed held, else this pass's time."""
    carried = {str(row.get("class")): row.get("since") for row in (previous or ()) if isinstance(row, dict)}
    return [{**row, "since": row.get("since") or carried.get(row["class"]) or now} for row in holds]


def _hold_set(holds: Any) -> List[tuple]:
    return sorted((str(row.get("class")), int(row.get("count") or 0)) for row in (holds or ()) if isinstance(row, dict))


def record_threw(root: Any, name: str, error: BaseException) -> None:
    """A retirer pass that threw: ``last_error`` ``{class, at}`` on its row (the maintenance child's own shape),
    which Storage and the mind's ``storage`` fact show until a pass of it is recorded again (``record_pass``).
    Best-effort: a store that cannot take the row leaves the caller's log line as the only trace."""
    try:
        record(root, name, last_error={"class": type(error).__name__, "at": utc_now_iso()})
    except Exception:
        return


def record_pass(root: Any, name: str, report: Dict[str, Any], *, event: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Record one retirer pass (a ``pass_report``) as the job ``name``'s row, with ``since`` per hold class,
    when the pass is evidence (the comment above); a pass that rested (``report["rested"]``) or changed
    nothing writes nothing and returns None, and one that judged only part of its plane
    (``report["partial"]``) records what it released and its errors under the row's holds as they stand
    (a job's first pass has no row to keep, so the holds it judged stand until a whole pass).
    A written pass clears ``last_error``. ``event`` names the events.jsonl row written beside the job row,
    with the same facts (the custody sweep's evidence of what moved and what holds); the retirer's whole
    report rides along whenever an event is emitted."""
    if report.get("rested"):
        return None
    current = job(root, name) or {}
    released = report.get("released")
    released_count = len(released) if isinstance(released, (list, tuple, set)) else int(released or 0)
    first, recovered = current.get("state") is None, "last_error" in current
    judged = current.get("holds") if report.get("partial") and not first else report.get("holds")
    changed = _hold_set(judged) != _hold_set(current.get("holds"))
    mailbox_removed = bool((report.get("mailboxes") or {}).get("removed"))
    if not (released_count or report.get("errors") or changed or mailbox_removed or first or recovered):
        return None
    now = utc_now_iso()
    holds = _with_since(judged or [], current.get("holds"), now)
    with store.open_store(pathlib.Path(root), write=True) as txn:
        row = record_in(txn, name, state="done", released=released_count, holds=holds,
                        held=sum(int(hold["count"]) for hold in holds),
                        errors=[str(error) for error in (report.get("errors") or ())][:20], pass_at=now)
        if recovered:
            row = forget_in(txn, name, "last_error")
    if event:
        from ouroboros.utils import append_jsonl

        append_jsonl(pathlib.Path(root) / "logs" / "events.jsonl",
                     {"ts": now, "type": event, "job": name, "released": released_count, "holds": holds,
                      "errors": row["errors"], "report": report})
    return row
