"""Immutable host attribution for the existing usage ledger, including root cause."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class UsageScope:
    drive_root: pathlib.Path | str | None = None
    task_id: str = ""
    root_task_id: str = ""
    parent_task_id: str = ""
    category: str = "task"
    source: str = "llm"
    # Explicitly attributed system probes/one-shots have no task control owner.
    non_task_operation: bool = False
    review_skill: str = ""
    review_wave_id: str = ""
    review_slot_id: str = ""
    global_limit_usd: Optional[float] = None
    root_limit_usd: Optional[float] = None
    root_cost_ceiling_usd: Optional[float] = None
    global_limit_source: str = ""
    global_limit_revision: Optional[str] = None
    # Whole-work billing group (owner Batch4, ``usage_admission``): empty = this
    # root is its own group under its own cap; a Continue's successor carries the
    # ORIGINAL root's group and that root's original cap.
    billing_group_id: str = ""
    billing_group_limit_usd: Optional[float] = None
    billing_group_limit_source: str = ""
    billing_group_limit_revision: Optional[str] = None
    root_limit_source: str = ""  # Original None + provenance is unlimited, not a new default.
    root_limit_revision: Optional[str] = None
    # The wave a caller named to own its prompt-cache split (skill, plan review). A round the review
    # derives (#1544: retry_key or a fresh id) is attribution only; the split stays per task.
    cache_wave: str = ""
    consciousness_cause: str = ""


def task_usage_scope(task: dict, *, drive_root, root_task_id: str, parent_task_id: str,
                     billing: dict) -> UsageScope:
    """Bind the host task envelope where agent execution enters accounting."""
    from ouroboros.consciousness_authority import WORK_KIND_OWNER, work_kind

    metadata = task.get("metadata") or {}
    category = str(metadata.get("usage_category") or task.get("type") or "task")
    cause = str(metadata.get("consciousness_cause") or "")
    if work_kind(metadata) == WORK_KIND_OWNER:
        cause = ""
        if category in {"consciousness", "consciousness_task"}:
            category = "task"
    return UsageScope(
        drive_root=drive_root, task_id=str(task["id"]), root_task_id=root_task_id,
        parent_task_id=parent_task_id,
        category=category, consciousness_cause=cause,
        source="agent.task",
        root_cost_ceiling_usd=task.get("root_cost_ceiling_usd") or metadata.get("root_cost_ceiling_usd"),
        **billing,
    )


def cause_label(cause: str, places=()) -> str:
    if cause == "own":
        return "own thoughts"
    if cause == "started":
        return "started work"
    place_id = cause.removeprefix("place:")
    return next((str(p["label"]) for p in places if p.get("place_id") == place_id), place_id)


_CATEGORY_CAUSES = {"consciousness": "own", "consciousness_task": "started"}


def root_cause(txn, root_id: str) -> str:
    """Read immutable attribution, or the unstamped historical root's category."""
    binding = txn.binding("consciousness", root_id)
    if binding:
        return str(binding.get("cause") or "")
    # Prefer the root task itself: import inserts current rows by last sequence,
    # so the first physical row in its tree can be a child that finished earlier.
    first = txn.conn.execute(
        "SELECT category, json_extract(extra, '$.consciousness_cause') FROM attempts INDEXED BY attempts_task "
        "WHERE task_id = ? AND root_task_id = ? ORDER BY rowid LIMIT 1", (root_id, root_id)).fetchone()
    if first is None:
        first = txn.conn.execute(
            "SELECT category, json_extract(extra, '$.consciousness_cause') FROM attempts "
            "WHERE root_task_id = ? ORDER BY rowid LIMIT 1", (root_id,)).fetchone()
    return str(first[1] or _CATEGORY_CAUSES.get(first[0], "")) if first else ""


def stamp_root_cause(txn, row: dict) -> None:
    """Keep the first root's cause across children, transitions and late receipts.

    An attribution binding in the same store transaction avoids rereading a
    growing tree on each receipt. It is reconstructed from stamped ledger rows
    on import. Historical rows stay untouched; their first row supplies only
    the unambiguous pre-focus category defaults.
    """
    root_id = str(row.get("root_task_id") or "")
    if not root_id:
        return
    cause = root_cause(txn, root_id) or str(row.get("consciousness_cause") or
                                        _CATEGORY_CAUSES.get(row.get("category"), ""))
    if cause in {"own", "started"} or cause.startswith("place:") and cause[6:]:
        row["consciousness_cause"] = cause


def bind_root_attribution(conn, row: dict, *, materialize_root_cap: bool = True) -> None:
    """Immutable root/group cap and cause attribution, in the caller's ledger transaction."""
    from ouroboros._usage_money import billing_group_key, durable_literals, monetary_scope_key
    from ouroboros._usage_rows import BINDING_KEYS, ROOT_CAP_SCOPE, _BINDING_CAPS, root_cap_binding

    if row.get("root_task_id") and row.get("consciousness_cause"):
        conn.execute("INSERT OR IGNORE INTO bindings (scope, key, binding) VALUES ('consciousness', ?, ?)",
                     (str(row["root_task_id"]), json.dumps({"cause": row["consciousness_cause"]})))
    if materialize_root_cap and (root_cap := root_cap_binding(row)) is not None:
        conn.execute("INSERT OR IGNORE INTO bindings (scope, key, binding) VALUES (?, ?, ?)",
                     (ROOT_CAP_SCOPE, monetary_scope_key(row),
                      json.dumps(durable_literals(root_cap), sort_keys=True, separators=(",", ":"), default=str)))
    if not any(cap in row for cap in _BINDING_CAPS):
        return
    binding = json.dumps(durable_literals({key: row[key] for key in BINDING_KEYS if key in row}),
                         sort_keys=True, separators=(",", ":"), default=str)
    for scope, key in (("root", monetary_scope_key(row)), ("group", billing_group_key(row))):
        if key:
            conn.execute("INSERT OR IGNORE INTO bindings (scope, key, binding) VALUES (?, ?, ?)", (scope, key, binding))


def root_caps_backfill_pending(import_meta) -> bool:
    """True on a completed pre-change store whose once-only ``root_cap`` backfill has not completed."""
    from ouroboros.usage_ledger import UsageLedgerCorrupt

    if not isinstance(import_meta, dict) or import_meta.get("status") != "completed":
        return False
    if "root_caps" not in import_meta:
        return True
    state = import_meta["root_caps"]
    if isinstance(state, dict) and state.get("status") == "completed":
        return False
    raise UsageLedgerCorrupt("invalid root_caps metadata in usage store")


def backfill_root_caps(txn) -> dict:
    """Once-only startup backfill of ``('root_cap', root)`` from ``attempts`` inside a write transaction.

    Replaces any pre-conversion ``root_cap`` rows in ``seq_first, rowid`` order so a later write
    cannot preempt an earlier verified source, while preserving original ``root``/``group``
    bindings, spend, unknowns, one-request maxima and unrelated works.
    """
    from ouroboros._usage_money import LiteralFloat, durable_literals, monetary_scope_key
    from ouroboros._usage_rows import ROOT_CAP_SCOPE, root_cap_record

    current = txn.meta("import")
    if not root_caps_backfill_pending(current):
        return {}
    existing_roots = txn.bindings("root")
    recovered = {}
    cursor = txn.conn.execute(
        "SELECT task_id, root_task_id, kind, root_limit_usd, billing_group_limit_usd, "
        "billing_group_limit_source, extra FROM attempts "
        "WHERE COALESCE(kind, 'attempt') = 'attempt' "
        "AND (COALESCE(root_task_id, '') != '' OR COALESCE(task_id, '') != '') "
        "AND (has_root_limit = 1 OR root_limit_usd IS NOT NULL "
        "     OR json_extract(extra, '$.root_limit_source') IS NOT NULL) "
        "ORDER BY seq_first, rowid"
    )
    for record in cursor:
        row = json.loads(record["extra"], parse_float=LiteralFloat)
        for key in row.pop("__none__", ()) or ():
            row[key] = None
        for key in ("task_id", "root_task_id", "kind", "billing_group_limit_source"):
            if record[key] is not None:
                row[key] = record[key]
        for key in ("root_limit_usd", "billing_group_limit_usd"):
            if record[key] is not None:
                row[key] = LiteralFloat(record[key])
        root = monetary_scope_key(row)
        if not root or root in recovered:
            continue
        if (cap := root_cap_record(row, existing_roots.get(root))) is not None:
            recovered[root] = cap
    txn.conn.execute("DELETE FROM bindings WHERE scope=?", (ROOT_CAP_SCOPE,))
    if recovered:
        txn.conn.executemany(
            "INSERT INTO bindings (scope, key, binding) VALUES (?, ?, ?)",
            [(ROOT_CAP_SCOPE, key, json.dumps(durable_literals(value), sort_keys=True,
                                              separators=(",", ":"), default=str))
             for key, value in recovered.items()],
        )
    ambiguous = sum(1 for value in recovered.values() if value.get("ambiguous"))
    backfilled = len(recovered) - ambiguous
    txn.set_meta("import", {**current, "root_caps": {
        "status": "completed", "backfilled": backfilled, "ambiguous": ambiguous}})
    return {k: v for k, v in (("root_caps_backfilled", backfilled), ("root_caps_ambiguous", ambiguous)) if v}
