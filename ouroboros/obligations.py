"""Addressed current obligations, maintained by the transitions that owe them.

The derived sets are a pure function of the result documents: every write of a
document recomputes its flag columns in ``tasks`` from ``result_memberships`` in
the same transaction, so membership can neither lag nor outlive the transition
that owes it. The owner-held sets (``custody_open``, closing custody receipts of
``delegated_runs``, ``pending_drives`` rows a pool handoff publishes, the receipt
sets, ``unknowns``, witnesses) are rows of the ``obligations`` table. ``members``
reads both halves in one transaction; no set is a file any more (the importer
reads the old set files once, the exporter writes them back for an older body).
Each set is owed to, and retired by, its own owner; no reader derives a global
settlement from them (the evidence packer packs a task once its row is terminal).

A periodic duty passes over its members through ``DutyRest``: a member it cannot
progress now gets a typed outcome in the ``duty_rest`` record set, and the next pass
selects it without reading its body until one of the recorded inputs changes.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Iterable

from ouroboros import store
from ouroboros.store import StoreError

BOOT_SETS = ("nonterminal", "synthesis", "terminal_projection", "delegated_runs",
             "promotions", "custody_open", "pending_drives", "upgrade_notices", "unknowns",
             "pause_notices", "pause_notice_receipts", "owner_questions")
REST_SET = "duty_rest"  # the typed outcomes of ``DutyRest``: kept facts of a duty's pass, never debt
COST_ARRIVALS = "cost_projection_arrivals"
ObligationsUnavailable = store.StoreError  # the sets are rows: unavailable exactly when the store is
_FACT_COLUMNS = ("task_id", "headless_child_drive_root", "child_drive_root", "drive_root")


def task_written(txn, task_id, columns=None):
    """Publish duty inputs in the task writer's transaction. The terminal-write counter moves on
    closing/closed writes, never on a live task's writes, and stamps the row's ``terminal_seq`` with
    its new value, so a pass keyed on the counter reads only the rows written since (``task_store``)."""
    from ouroboros.task_status import SETTLED_STATUSES
    if columns is None:
        row = txn.conn.execute("SELECT status, write_seq, updated_at, plan_collection_pending, drive_retention_at "
                               "FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        columns = dict(row) if row is not None else {}
    if columns.get("status") in SETTLED_STATUSES:
        txn.set_meta(store.TERMINAL_WRITES, count := int(txn.meta(store.TERMINAL_WRITES) or 0) + 1)
        txn.conn.execute("UPDATE tasks SET terminal_seq=? WHERE task_id=?", (count, task_id))
    if columns and txn.conn.execute("SELECT 1 FROM obligations WHERE set_name=? AND id=? "
                        "AND json_type(facts, '$.signature')='null'",
                        (REST_SET, "cost_projection:" + task_id)).fetchone():
        txn.set_meta(COST_ARRIVALS, int(txn.meta(COST_ARRIVALS) or 0) + 1)
        txn.delete_row(REST_SET, "cost_projection:" + task_id)  # later live writes are no longer arrivals
    from ouroboros.plan_collection_job import task_written as plan_written
    plan_written(txn, task_id, columns)
    from ouroboros.maintenance_retirement import task_written as retirement_written
    retirement_written(txn, task_id, columns)


def review_collection_pending(row: dict) -> bool:
    """The ``tasks.review_collection_pending`` column's predicate, kept with the store schema: the
    task holds an open review operation pointer (a malformed one counts). Its agent-session reader may
    hold a retained handle's absolute path, so the packer keeps the task's loose files
    (``evidence_packs.row_packable``), and the maintenance pass removes the pointer once its controller
    is proven dead (``review_operation.retire_dead_review_pointers``). What the retired task acceptance
    left (closed pointers, panels, ``acceptance-late:`` receipts) is history, never open work."""
    operations = row.get("review_operations", {})
    if operations is None:
        operations = {}
    if not isinstance(operations, dict):
        return True
    for entry in operations.values():
        if not isinstance(entry, dict):
            return True
        if not entry.get("control_only") and entry.get("state") in {"preparing", "retained", "dispatched"}:
            return True
    return False


def _check_name(name: str) -> str:
    if not name or Path(name).name != name:
        raise ValueError("an obligation set name must be a filename stem")
    return name


def members_in(txn: store.Txn, name: str) -> dict[str, dict]:
    """One set inside an open transaction: the owner-held rows, plus the members the flag
    and status columns derive with their facts from the typed columns (a held row of a
    flagged task keeps its own facts under the derived ones: the closing receipts of
    ``delegated_runs`` survive until the backfill acknowledges them)."""
    rows = dict(txn.set_rows(_check_name(name)))
    if name in store.FLAG_COLUMNS:
        where = f"{store.FLAG_COLUMNS[name]}=1"
    elif name == "nonterminal":
        # The few live tasks among the whole history: ``unlikely`` keeps the read on the status index, which the
        # maintenance child's ANALYZE statistics would otherwise trade for a walk of every task in task_id order
        # (any SQLite, by the table's shape: tests/test_store_capture_readers.py).
        where = "unlikely(status IN ('running', 'interrupted'))"
    else:
        return rows
    for record in txn.conn.execute(f"SELECT {','.join(_FACT_COLUMNS)} FROM tasks WHERE {where} ORDER BY task_id"):
        facts = {key: record[key] for key in _FACT_COLUMNS if record[key]}
        rows[record["task_id"]] = {**rows.get(record["task_id"], {}), **facts}
    return rows


def owner_held_rows(name: str, rows: dict[str, dict]) -> dict[str, dict]:
    """The rows of a set file an import may hold as owner-held rows: a derived set's members
    come only from the typed columns (``nonterminal`` and the flag columns), so its file rows
    import nothing; ``delegated_runs`` keeps its closing receipts, ``pending_drives`` the rows a
    pool handoff published (both retire by their own owner); every other set is owner-held."""
    if name == "delegated_runs":
        return {identity: facts for identity, facts in rows.items() if isinstance(facts, dict)
                and (facts.get("closed_runs") or facts.get("cost_revision"))}
    if name == "nonterminal" or (name in store.FLAG_COLUMNS and name != "pending_drives"):
        return {}
    return rows


def members(root: Any, name: str) -> dict[str, dict]:
    """Return an independent snapshot; no migration or historical fallback."""
    with store.open_store(root, write=False) as txn:
        return members_in(txn, name)


def add(root: Any, name: str, identity: str, facts: dict | None = None) -> None:
    """Publish (or update) an addressed owner-held obligation before its dependent effect."""
    with store.open_store(root, write=True) as txn:
        txn.put_row(_check_name(name), str(identity), dict(facts or {}))


def remove(root: Any, name: str, identity: str) -> None:
    """Retire an owner-held row after discharge (a derived member retires with its document)."""
    with store.open_store(root, write=True) as txn:
        txn.delete_row(_check_name(name), str(identity))


def result_memberships(row: dict) -> set[str]:
    """The result-owning writers share these predicates, including direct patches."""
    names = set()
    if row.get("status") in {"running", "interrupted"}:
        names.add("nonterminal")
    checkpoint = row.get("root_phase_checkpoint")
    checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
    if checkpoint.get("post_task_synthesis"):
        from ouroboros.post_task_checkpoint import post_task_synthesis_is_open
        if post_task_synthesis_is_open(checkpoint["post_task_synthesis"]):
            names.add("synthesis")
    if (row.get("canonical_terminal_projection_origin") == "terminal_transition"
            or isinstance(row.get("canonical_terminal_projection_ready"), dict)):
        from ouroboros.terminal_projection import terminal_projection_owed
        if terminal_projection_owed(str(row.get("task_id") or ""), row):
            names.update(("synthesis", "terminal_projection"))
    if row.get("delegated_runs_unreconciled"):
        names.add("delegated_runs")
    if row.get("child_ref_promotion"):
        from ouroboros.observability import _has_pending_ref_promotion
        if _has_pending_ref_promotion(row["child_ref_promotion"]):
            names.add("promotions")
    if row.get("pause_notices"):
        names.add("pause_notices")
    quizzes = row.get("owner_quiz")
    if isinstance(quizzes, dict):
        from ouroboros.owner_quiz import quiz_needs_retention
        if any(quiz_needs_retention(block) for block in quizzes.values()):
            names.add("owner_questions")
    return names


def result_facts(row: dict) -> dict:
    return {key: row[key] for key in _FACT_COLUMNS if row.get(key)}


def pending_drive(root: Path, task_id: str, document: dict) -> bool | None:
    """None preserves a writer's membership; import has no previous membership."""
    from ouroboros.headless import terminal_task_files_ready
    from ouroboros.task_status import SETTLED_STATUSES

    status = document.get("status")
    if status in {"running", "interrupted"}:
        return True
    if status in SETTLED_STATUSES:
        try:
            return not terminal_task_files_ready(root, {**document, "id": task_id}, document)
        except Exception:
            return True  # unknown readiness keeps the drive pending
    return None


def pending_drive_rule(txn: store.Txn, current: dict, updated: dict) -> bool:
    """A status outside the drive lifecycle keeps its previous membership."""
    task_id = str(updated.get("task_id") or current.get("task_id") or "")
    pending = pending_drive(Path(txn.root), task_id, updated)
    return bool(txn.row(task_id, "pending_drive")) if pending is None else pending


def drive_started(root: Any, task: dict) -> None:
    """Pool handoff: publish before the child can write anything or die."""
    tid = str(task.get("id") or task.get("task_id") or "")
    if tid:
        add(root, "pending_drives", tid, {**result_facts(task), "task_id": tid})


def drive_finished(root: Any, task: dict, row: dict | None = None) -> None:
    """Retire only after the canonical terminal file receipt is complete."""
    from ouroboros.headless import terminal_task_files_ready
    from ouroboros.task_results import load_task_result
    from ouroboros.task_status import SETTLED_STATUSES

    tid = str(task.get("id") or task.get("task_id") or "")
    try:
        row = row if row is not None else load_task_result(root, tid, strict=True)
    except (OSError, ValueError, StoreError):
        return  # an unreadable result keeps the drive pending: recovery decides later
    if row and row.get("status") in SETTLED_STATUSES and terminal_task_files_ready(Path(root), task, row):
        with store.open_store(root, write=True) as txn:
            txn.delete_row("pending_drives", tid)
            txn.conn.execute("UPDATE tasks SET pending_drive=0 WHERE task_id=?", (tid,))


def result_rows(root: Any, name: str, *, exclude=(), ids: Iterable[str] | None = None):
    """Read only addressed candidates (``ids``: a pass's own selection of NAME's members);
    unreadable members remain repair gaps."""
    import logging
    from ouroboros.task_results import load_task_result
    for tid in members(root, name) if ids is None else ids:
        if tid in exclude:
            continue
        try:
            row = load_task_result(root, tid, strict=True)
            if row:
                yield row
            else:
                add(root, "unknowns", f"task:{tid}", {"task_id": tid, "reason": "result_missing"})
        except (OSError, ValueError, StoreError) as exc:
            add(root, "unknowns", f"task:{tid}", {"task_id": tid, "reason": type(exc).__name__})
            logging.getLogger(__name__).warning("Obligation %s/%s is unreadable", name, tid)


def _instant(value: Any) -> datetime.datetime | None:
    try:
        stamp = datetime.datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=datetime.timezone.utc)


class DutyRest:
    """One pass of a periodic duty over its addressed members, with durable rests.

    A member the duty cannot progress now gets a typed outcome (``wait``) in the record set
    ``duty_rest``, id ``<duty>:<task_id>``: what it waits on, with the inputs whose change could
    let it progress -- the task row's signature (``write_seq``, ``updated_at``: every write of the
    task moves it, so a new fact on the task always counts), the stamps of external inputs the
    duty names, and an existing persisted retry instant (``due_at``). While they all hold, the
    next pass selects the member without reading its body and writes nothing; the first change
    puts it back exactly once. A rest never retires the debt: the member stays in its set and
    every other reader still sees it; no attempt count or elapsed time decides anything here.
    ``candidates`` maps each member to its own input stamps (or ``name`` reads a set's members);
    ``wake`` names outcomes this pass examines anyway (work a restart re-establishes in memory).
    """

    def __init__(self, root: Any, duty: str, candidates: dict | None = None, *, name: str = "",
                 inputs: dict | None = None, now: str | None = None, wake: Iterable[str] = (),
                 partial: bool = False) -> None:
        from ouroboros.utils import utc_now_iso

        self.root, self.duty, self.inputs = Path(root), _check_name(duty), dict(inputs or {})
        prefix = self.duty + ":"
        self.candidates = {str(tid): dict(facts or {}) for tid, facts in (candidates or {}).items()}
        self.signatures: dict[str, list | None] = {}
        self.marks: dict[str, dict] = {}
        self.readable = False
        try:
            with store.open_store(self.root, write=False) as txn:
                if name:
                    self.candidates = {str(tid): {} for tid in members_in(txn, name)}
                self.signatures = {record["task_id"]: [record["write_seq"], record["updated_at"]] for record in
                                   txn.conn.execute("SELECT task_id, write_seq, updated_at FROM tasks WHERE task_id IN "
                                                    "(SELECT value FROM json_each(?))", (json.dumps(sorted(self.candidates)),))}
                # A partial pass owns only its addressed members; other rests stay untouched.
                where, params = ("id IN (SELECT value FROM json_each(?))",
                                 (json.dumps([prefix + tid for tid in self.candidates]),)) if partial else (
                                     "id>=? AND id<?", (prefix, self.duty + ";"))
                self.marks = {record["id"][len(prefix):]: json.loads(record["facts"]) for record in txn.conn.execute(
                    "SELECT id, facts FROM obligations WHERE set_name=? AND " + where, (REST_SET, *params))}
                self.readable = True
        except StoreError:
            if name:
                raise  # the set's members are unknown: the duty cannot pass
            self.signatures, self.marks = {}, {}  # unknown rests prove nothing: every candidate is examined
        now_at, wake = _instant(now or utc_now_iso()), set(wake)
        self.resting = {tid: mark for tid, mark in self.marks.items()
                        if tid in self.candidates and self._holds(tid, mark, now_at, wake)}
        self.examine = [tid for tid in self.candidates if tid not in self.resting]
        self._outcomes: dict[str, dict] = {}

    def _holds(self, tid: str, mark: dict, now_at: datetime.datetime | None, wake: set) -> bool:
        current = {**self.inputs, **self.candidates[tid]}
        due = _instant(mark.get("due_at")) if mark.get("due_at") else None
        return (self.readable and mark.get("signature") == self.signatures.get(tid) and not wake & set(mark.get("waits_on") or ())
                and all(current.get(key) == value for key, value in (mark.get("inputs") or {}).items())
                and (not mark.get("due_at") or (due is not None and now_at is not None and due > now_at)))

    def wait(self, task_id: str, *waits_on: str, due_at: str | None = None, inputs: Iterable[str] = (),
             report: dict | None = None, absent: bool = False) -> None:
        """TASK_ID, examined this pass, cannot progress until what it WAITS_ON changes; ``inputs``
        names the pass's external stamps the outcome depends on (a member's own always count);
        ``report`` is what the duty states while it rests. ``absent`` requires a successful
        read that found no row, never a failed/unavailable read."""
        tid = str(task_id)
        if (not self.readable or tid not in self.candidates or not waits_on
                or (self.signatures.get(tid) is None) != absent):
            return
        stamps = {**{key: self.inputs[key] for key in inputs if key in self.inputs}, **self.candidates[tid]}
        self._outcomes[tid] = {"task_id": tid, "signature": self.signatures.get(tid), "waits_on": sorted(set(waits_on)),
                               **({"inputs": stamps} if stamps else {}), **({"due_at": str(due_at)} if due_at else {}),
                               **({"report": dict(report)} if report else {})}

    def commit(self, *, confirm: Iterable[str] = ()) -> int | None:
        """Write the pass's new outcomes and drop the rests of members that left, in one write
        transaction; an outcome whose task was written after its selection is not recorded (that
        write already puts the member back). Nothing new: nothing is written. A rest is never
        authority, so a store that cannot take it leaves the members to the next pass.
        ``confirm`` revalidates selected signatures and returns the absent-owner arrival marker
        for a process input latch; a racing write or unreadable store returns None."""
        import logging

        outcomes = {tid: facts for tid, facts in self._outcomes.items() if self.marks.get(tid) != facts}
        departed = [tid for tid in self.marks if tid not in self.candidates]
        confirm = tuple(confirm)
        if not outcomes and not departed and not confirm:
            return
        try:
            with store.open_store(self.root, write=bool(outcomes or departed)) as txn:
                valid = self.readable
                signatures = {row["task_id"]: [row["write_seq"], row["updated_at"]] for row in txn.conn.execute(
                    "SELECT task_id, write_seq, updated_at FROM tasks WHERE task_id IN (SELECT value FROM json_each(?))",
                    (json.dumps(sorted(set(confirm) | set(outcomes))),))}
                for tid in confirm:
                    valid = valid and signatures.get(tid) == self.signatures.get(tid)
                for tid in departed:
                    txn.delete_row(REST_SET, f"{self.duty}:{tid}")
                for tid, facts in outcomes.items():
                    if signatures.get(tid) == facts["signature"]:
                        txn.put_row(REST_SET, f"{self.duty}:{tid}", facts)
                    else:
                        valid = False
                return int(txn.meta(COST_ARRIVALS) or 0) if valid else None
        except StoreError:
            logging.getLogger(__name__).debug("Duty %s rests not recorded this pass", self.duty, exc_info=True)
