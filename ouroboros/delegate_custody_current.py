"""Current delegated custody for bookkeeping; evidence replay stays explicit.

Open runs, pending starts, undisposed patches and owned registrations live in
``custody_open``. Closing a run also owes an addressed terminal-result refresh;
its receipt travels in ``delegated_runs`` until that refresh lands. The existing
reducer defines every run field. The boot/periodic sweep binds current reads so
nested custody helpers cannot accidentally cold-fold the historical chain.
"""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from ouroboros import obligations as o

_CURRENT = contextvars.ContextVar("current_custody_root", default=None)


def active(root) -> bool:
    return _CURRENT.get() == str(Path(root).resolve())


@contextmanager
def current_reads(root):
    token = _CURRENT.set(str(Path(root).resolve()))
    try:
        yield
    finally:
        _CURRENT.reset(token)


def _decode(raw):
    from ouroboros.delegate_custody import RunCustody
    from dataclasses import fields
    raw = deepcopy(raw)  # Reducer mutations must not also mutate the comparison snapshot.
    entry = RunCustody(**{f.name: raw[f.name] for f in fields(RunCustody) if f.name in raw})
    entry.verified_source_ranges = [tuple(pair) for pair in entry.verified_source_ranges]
    entry.output_reader_receipts = tuple(tuple(pair) for pair in entry.output_reader_receipts)
    for key, value in raw.items():
        if key.startswith("_"):
            setattr(entry, key, value)
    return entry


def _held(entry):
    return (not entry.settled or entry.project_owned
            or ((entry.snapshot_id or (entry.resource_ref.get("workspace_kind") == "directory"
                                      and entry.resource_ref.get("strategy") == "copy"))
                and not entry.patch_disposed))


def state(root):
    return {key[4:]: _decode(facts["custody"]) for key, facts in o.members(root, "custody_open").items()
            if key.startswith("run:")}


def pending(root):
    from ouroboros.delegate_pending import pending_invocations
    rows = [facts["request"] for key, facts in o.members(root, "custody_open").items()
            if key.startswith("invocation:")]
    return pending_invocations(root, rows=rows)


def snapshot(root):
    runs = {}
    for facts in o.members(root, "delegated_runs").values():
        runs.update({rid: _decode(raw) for rid, raw in facts.get("closed_runs", {}).items()})
    runs.update(state(root))
    return {"state": runs, "pending": pending(root), "current": True}


@contextmanager
def publication(root, event):
    """Publish before append, then merge discharge with current state after it.

    The append may invoke a result writer (result lock -> obligations lock).
    Never hold the obligations lock across that callback or await another owner.
    A set update that fails (unreadable, contended, a disk error) never stops the row: it owes
    a rebuild. The next start adds what the retained chain holds and the set lacks; a missed
    discharge stays an open candidate until the reconcile sweep settles it again.
    """
    from ouroboros import delegate_custody as c

    rid, iid = str(event.get("run_id") or ""), str(event.get("invocation_id") or "")
    kind = event["type"]

    def publish():
        with o.locked(root):
            rows = o._read(root, "custody_open", missing_ok=True)
            before = dict(rows)
            # Work is addressable before its source event can be lost.
            if kind == c.START_REQUESTED and iid:
                rows.setdefault(f"invocation:{iid}", {"request": event})
            runs = {key[4:]: _decode(facts["custody"]) for key, facts in rows.items() if key.startswith("run:")}
            c._apply(runs, event)
            if kind == c.STARTED and rid in runs:
                rows[f"run:{rid}"] = {"custody": vars(runs[rid])}
            if rows != before:
                o._write(root, "custody_open", rows)

    def discharge():
        with o.locked(root):
            rows = o._read(root, "custody_open", missing_ok=True)
            before = dict(rows)
            runs = {key[4:]: _decode(facts["custody"]) for key, facts in rows.items() if key.startswith("run:")}
            # Even with no STARTED in the set, retained closing receipts can accept
            # later output/patch facts without reading the event chain.
            debts = o._read(root, "delegated_runs", missing_ok=True)
            prior_debts = {key: dict(value) for key, value in debts.items()}
            if rid and rid not in runs:
                for facts in debts.values():
                    raw = facts.get("closed_runs", {}).get(rid)
                    if raw:
                        runs[rid] = _decode(raw)
                        break
            c._apply(runs, event)
            for run_id, entry in runs.items():
                if entry.settled and entry.task_id:
                    facts = dict(debts.get(entry.task_id, {}))
                    receipts = dict(facts.get("closed_runs", {}))
                    receipts[run_id] = vars(entry)
                    debts[entry.task_id] = {**facts, "task_id": entry.task_id, "closed_runs": receipts}
                if _held(entry):
                    rows[f"run:{run_id}"] = {"custody": vars(entry)}
                else:
                    rows.pop(f"run:{run_id}", None)
            if iid and (kind == c.STARTED or (kind == c.START_FAILED and event.get("definite") is True)):
                rows.pop(f"invocation:{iid}", None)
            # Publish the closing receipt before dropping its open-custody member.
            if debts != prior_debts:
                o._write(root, "delegated_runs", debts)
            if rows != before or not o.path(root, "custody_open").exists():
                o._write(root, "custody_open", rows)

    o._bookkeeping(root, "custody", publish)
    landed = [False]
    yield landed
    if landed[0]:
        o._bookkeeping(root, "custody discharge", discharge)


def rebuild(root, *, replace=False):
    """Explicit migration/repair only: fold the retained chain once."""
    from ouroboros import delegate_custody as c
    if c.custody_log_unreadable(root):
        raise o.ObligationsUnavailable("custody event chain is unreadable")
    rows = tuple(c._iter_rows(c.event_log_path(root)))
    runs = c.replay(root, rows=rows)
    result = {f"run:{rid}": {"custody": vars(entry)} for rid, entry in runs.items() if _held(entry)}
    for entry in c.pending_invocations(root, rows=rows):
        result[f"invocation:{entry['invocation_id']}"] = {"request": {"type": c.START_REQUESTED, **entry}}
    with o.locked(root):
        # Migration runs before producers. On an interrupted import, any
        # already-published obligations remain extra candidates.
        if not replace:
            result.update(o._read(root, "custody_open", missing_ok=True))
        o._write(root, "custody_open", result)
        debts = o._read(root, "delegated_runs", missing_ok=True)
        for tid, facts in debts.items():
            facts["closed_runs"] = {**facts.get("closed_runs", {}), **{
                rid: vars(entry) for rid, entry in runs.items() if entry.task_id == tid and entry.settled}}
        o._write(root, "delegated_runs", debts)
    return len(result)
