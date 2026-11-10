"""The one SSOT projection of task cost for every producer surface (C2, owner 10=B).

``accounted_upper_bound_usd`` retains the settled + reserved + unresolved
subtotal from the physical attempt ledger. Unpriced work makes it a subtotal,
not an upper bound on the whole bill; scoped coverage travels beside it.
ABI 7.0 (ABI-3, the approved ABI break): the deprecated ``cost_usd`` /
``cost_usd_with_children`` aliases are no longer EMITTED anywhere — the write
seams strip them — while the READ side keeps resolving both spellings so every
stored legacy record (task results, chat.jsonl rows, event snapshots) stays
readable, with the historical deprecated-wins precedence for diverged pairs.

Null is null: a missing or unknown cost projects ``None`` on BOTH names and must
never render as ``$0.00``, and finality is never fabricated — ``cost_final`` is
True only when the SOURCE says so about a known amount.

Producers do not hand-assemble these fields. They pass their source mapping
through :func:`cost_projection` (read-side projections: peek/recent/handoff/wait)
or :func:`with_cost_aliases` (write-side field dicts that already carry the full
meta set) so every surface tells the same story with the same names.
"""

from __future__ import annotations

import logging
import math
import pathlib
import threading
import time
from typing import Any, Dict, Literal, Mapping, Optional, TypedDict

log = logging.getLogger(__name__)

# (honest name, retired legacy spelling). Since ABI 7.0 the legacy spelling is
# READ-ONLY tolerance for stored records — emitters never write it again.
COST_ALIAS_PAIRS = (
    ("accounted_upper_bound_usd", "cost_usd"),
    ("accounted_upper_bound_usd_with_children", "cost_usd_with_children"),
)

# The accounting OPENNESS / INTEGRITY markers that must ride BESIDE an amount
# whenever the source carries them: an upper bound without them reads like a
# receipt. ONE list, here — producers import it instead of hand-building a
# closed whitelist each (three of those had silently dropped `reserved_usd`,
# `unresolved_upper_bound_usd` and the ledger-integrity marker, so a frame said
# "not final" with nothing on it explaining what was still open). Adding a new
# marker means adding it here, once.
COST_OPENNESS_FIELDS = (
    # #498: the optional scoped carrier below — the facts that EXPLAIN an amount,
    # bound to the scope whose rows produced it, so a card never pairs a root-tree
    # number with own-scope openness (or a parent's zero with a child's unknown).
    "cost_presentation",
    "cost_custody",
    "cost_accounting_status",
    "cost_accounting_error",
    "cost_final",
    "cost_with_children_partial",
    "unknown_unmetered",
    "non_final_rows",
    "reserved_usd",
    "unresolved_upper_bound_usd",
    "ledger_integrity_degraded",
)


def _as_amount(value: Any) -> Optional[float]:
    """A dollar amount, or None. Booleans, unparseable and non-finite values are None (the web's
    ``formatUsd2``/``formatUsdKnown`` print such a value as unknown too)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return None
    return amount if math.isfinite(amount) else None


def honest_accounted_amount(source: Optional[Mapping[str, Any]]) -> Optional[float]:
    """Return a known accounted subtotal without fabricating unknown zero.

    The physical ledger uses ``accounted_usd`` as the settled plus reserved /
    unresolved upper bound. An external or unknown-price row can therefore
    leave that subtotal at ``0.0`` while ``unknown_unmetered`` is positive.
    That zero is not a measured free result unless a bound exists, so terminal
    and live producers must carry ``None`` instead. A non-zero known subtotal,
    including a mixed known+unknown tree, remains useful and is disclosed by
    the openness fields beside it.
    """
    src = source if isinstance(source, Mapping) else {}
    amount = _as_amount(src.get("accounted_usd"))
    if amount is None:
        return None
    try:
        unknown = int(src.get("unknown_unmetered") or 0)
        reserved = float(src.get("reserved_usd") or 0)
        unresolved = float(src.get("unresolved_upper_bound_usd") or 0)
    except (TypeError, ValueError):
        return amount
    from ouroboros.delegate_custody_usage import cost_coverage_open
    if amount == 0 and cost_coverage_open(src) and not src.get("priced_rows"):
        return None
    if amount == 0 and unknown > 0 and reserved == 0 and unresolved == 0:
        return None
    return amount


# The two scopes an amount can describe. A carrier states its own, because the
# same record holds both and only its producer knows which rows it summed.
COST_SCOPE_OWN = "own"
COST_SCOPE_ROOT_TREE = "root_tree"


class CostPresentation(TypedDict):
    """The optional wire carrier and its producer share this closed shape.

    Amount and row-state facts belong to one bucket. Null at the enclosing
    field means that ledger evidence was unavailable, not measured zero.
    """

    scope: Literal["own", "root_tree"]
    tracked_amount: Optional[float]
    has_unpriced: bool
    tracked_final: bool
    accounting_open: bool
    has_rows: bool


def build_cost_presentation(
    summary: Optional[Mapping[str, Any]], *, scope: str,
) -> Optional[CostPresentation]:
    """The money facts a card needs, from ONE ledger summary bucket (#498).

    The amount and the facts that explain it have to travel together. A card that
    read the tracked subtotal from the subtree and the unknown-price markers from
    the task showed a parent's honest ``$0.00`` over a child whose price nobody
    knew. So this carrier is built once, by the producer that actually summed the
    rows, and states which scope those rows belong to.

    * ``tracked_amount`` — the accounted subtotal, or ``None``. A zero survives
      only on PRICED evidence (``priced_rows``): an empty ledger and a ledger of
      exclusively unpriced rows both add up to 0.0 and neither is a measured zero.
      The count is what makes that exact, so the subtotal is read straight from
      the bucket here rather than through :func:`honest_accounted_amount`, whose
      bound-and-unknown heuristic is the count-free approximation of this same
      question (it discards a real priced zero that shares a scope with an
      unpriced row — the MIXED zero this carrier can now state honestly).
    * ``has_unpriced`` — some row in this scope has no price at all, so the
      tracked amount is only a subtotal; it may itself include open bounds.
    * ``tracked_final`` — the tracked amount is exact. Estimates, retained
      reservation bounds and ledger-integrity gaps all leave it False.
    * ``accounting_open`` — more can still arrive for this scope. A delegated run
      whose custody closed without a matching receipt cannot bring money: it keeps
      ``has_unpriced`` (its cost is unknown) but never holds accounting open.
    * ``has_rows`` — this scope has any accountable row at all; without one there
      is nothing to say, not a free result.

    ``None`` when the summary is missing entirely (an unreadable ledger already
    says so through ``cost_accounting_status``).
    """
    if not isinstance(summary, Mapping) or scope not in {COST_SCOPE_OWN, COST_SCOPE_ROOT_TREE}:
        return None
    try:
        priced = max(0, int(summary.get("priced_rows") or 0))
        unpriced = max(0, int(summary.get("unknown_unmetered") or 0))
        # Absence on a legacy bucket proves neither closed nor exact.
        tracked_nonfinal = int(summary["tracked_nonfinal_rows"])
        open_rows = int(summary["accounting_open_rows"])
    except (KeyError, TypeError, ValueError):
        return None
    amount = _as_amount(summary.get("accounted_usd")) if priced else None
    integrity_gap = bool(summary.get("integrity_degraded"))
    from ouroboros.delegate_custody_usage import cost_coverage_open, cost_coverage_pending
    open_accounting = open_rows > 0 or integrity_gap or cost_coverage_pending(summary)
    return {
        "scope": str(scope),
        "tracked_amount": round(amount, 6) if amount is not None else None,
        "has_unpriced": unpriced > 0 or cost_coverage_open(summary),
        "tracked_final": amount is not None and tracked_nonfinal == 0 and not integrity_gap,
        "accounting_open": open_accounting,
        "has_rows": bool(priced or unpriced),
    }


def resolve_cost_pair(source: Optional[Mapping[str, Any]], new: str, old: str) -> tuple[bool, Any]:
    """``(present, raw_value)`` for one alias pair under the ONE precedence rule.

    THE DEPRECATED NAME WINS when both are present. Every seam — read and write,
    Python and JS — asks this question here, because two seams answering it
    differently is how a diverged pair starts telling two stories about the same
    record: the write side re-converged on ``cost_usd`` while the read side
    displayed ``accounted_upper_bound_usd``. Deprecated-wins is the direction the
    write side must take anyway (legacy mutators between two seam crossings edit
    ``cost_usd``), so reading the same way keeps a record self-consistent no
    matter which producer last touched it.
    """
    src = source if isinstance(source, Mapping) else {}
    if old in src:
        return True, src[old]
    if new in src:
        return True, src[new]
    return False, None


def with_task_cost_presentation(projection: Mapping[str, Any], task: Mapping[str, Any],
                                drive_root: Any) -> Dict[str, Any]:
    """Attach root-scoped coverage and narrow finality without changing amounts.

    A root's first terminal frame can precede the supervisor's subtree refresh.
    It must already explain the tree, including roots that skip synthesis. A
    failed tree read is explicit null, never the previously computed own cost.
    """
    from ouroboros.task_results import resolve_task_lineage

    tid = str(task.get("id") or task.get("task_id") or "")
    lineage = resolve_task_lineage(tid, metadata=task.get("metadata"), **{
        key: task.get(key) for key in ("root_task_id", "parent_task_id", "delegation_role",
                                      "original_task_id", "timeout_retry_from")
    })
    if not lineage["is_root_task"]:
        return dict(projection)
    from ouroboros.delegate_custody_usage import unavailable_cost_custody

    carrier, bucket = None, {}
    if projection.get("cost_accounting_status") == "available":
        try:
            from ouroboros.usage_accounting import usage_breakdown

            bucket = usage_breakdown(pathlib.Path(task.get("budget_drive_root") or drive_root),
                                     root_task_id=str(lineage["root_task_id"] or tid), custody_root=drive_root, include_custody=True)
            carrier = build_cost_presentation(bucket, scope=COST_SCOPE_ROOT_TREE)
        except Exception:
            log.warning("Root cost presentation unavailable for %s", tid, exc_info=True)
    return {**projection, "cost_presentation": carrier,
            "cost_custody": bucket.get("cost_custody") or unavailable_cost_custody(),
            "cost_final": bool(projection.get("cost_final") and bucket.get("cost_final")),
            "cost_with_children_partial": bool(projection.get("cost_with_children_partial") or not bucket.get("cost_final"))}


def honest_cost_pair_amount(
    source: Optional[Mapping[str, Any]], new: str, old: str,
) -> tuple[bool, Optional[float]]:
    """Resolve one cost pair and apply the shared unknown-zero rule."""
    src = source if isinstance(source, Mapping) else {}
    present, raw = resolve_cost_pair(src, new, old)
    if not present:
        return False, None
    probe = dict(src)
    carrier = src.get("cost_presentation")
    scope = COST_SCOPE_OWN if new == COST_ALIAS_PAIRS[0][0] else COST_SCOPE_ROOT_TREE
    # Public projections omit priced_rows; their same-scope carrier retains the
    # priced-zero evidence. A different scope or an explicit null proves none.
    if (_as_amount(raw) == 0 and isinstance(carrier, Mapping) and carrier.get("scope") == scope
            and carrier.get("has_rows") is True and _as_amount(carrier.get("tracked_amount")) == 0):
        probe["priced_rows"] = 1
    probe["accounted_usd"] = raw
    return True, honest_accounted_amount(probe)


def with_cost_aliases(fields: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """A COPY of ``fields`` normalized onto the honest cost names (ABI 7.0).

    Write-side seam: a producer that already assembled the full cost-meta dict
    (``reconstruct_task_cost``, the terminal frame builder) passes through here
    — and it must be the LAST step after any cost mutation. A legacy spelling
    present on the input still WINS resolution (:func:`resolve_cost_pair`,
    deprecated wins — a legacy mutator's edit is honored) but is STRIPPED from
    the output: since ABI-3 nothing emits or persists the retired alias keys.
    A pair absent on both sides STAYS absent — normalization never invents a
    field, and an explicit None stays None. Idempotent.
    """
    out = dict(fields or {})
    for new, old in COST_ALIAS_PAIRS:
        present, value = honest_cost_pair_amount(out, new, old)
        out.pop(old, None)
        if present:
            out[new] = value
    return out


def normalize_task_result_cost_planes(fields: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """``with_cost_aliases`` over a task-result row AND its nested public cost planes.

    ABI-3 (fix-round-3): the projection seam (``public_task_result``) and the
    rewrite seam (``write_task_result``) share this ONE normalizer, so the
    stored row and the outbound payload cannot diverge on WHICH planes are
    normalized. The nested planes are the known public ones — the subagent
    envelope (its own amount AND its ``usage`` snapshot, the actually
    supported producer path) and the loop-outcome ``usage`` dict. Internal
    evidence planes that merely share the spelling (review receipts, ledger
    rows) are their own schemas and pass through untouched — the emission
    sweep allowlist names them per-site, and the projection-boundary tests
    deep-scan the public planes instead.
    """
    out = with_cost_aliases(fields)
    envelope = out.get("subagent_envelope")
    if isinstance(envelope, dict):
        envelope = with_cost_aliases(envelope)
        if isinstance(envelope.get("usage"), dict):
            envelope["usage"] = with_cost_aliases(envelope["usage"])
        out["subagent_envelope"] = envelope
    loop_outcome = out.get("loop_outcome")
    if isinstance(loop_outcome, dict) and isinstance(loop_outcome.get("usage"), dict):
        out["loop_outcome"] = {
            **loop_outcome, "usage": with_cost_aliases(loop_outcome["usage"]),
        }
    return out


def carry_cost_meta(source: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The honest cost names plus every openness/integrity marker the source carries.

    For a producer assembling a NEW frame out of an existing payload: it copies
    what accounting actually said instead of re-listing field names by hand.
    A legacy alias spelling on the SOURCE still resolves (deprecated wins) but
    only the honest name is emitted (ABI 7.0). Absent stays absent; unknown
    extra keys are the caller's business (this returns only accounting fields,
    so a caller merging it never loses its own).
    """
    src = source if isinstance(source, Mapping) else {}
    out: Dict[str, Any] = {}
    for new, old in COST_ALIAS_PAIRS:
        present, value = honest_cost_pair_amount(src, new, old)
        if present:
            out[new] = value
    for key in COST_OPENNESS_FIELDS:
        if key in src:
            out[key] = src[key]
    return out


def live_root_cost_projection(
    task_id: str, task: Mapping[str, Any], event: Mapping[str, Any], drive_root: pathlib.Path,
) -> Dict[str, Any]:
    """Return a non-final root-subtree projection: two reads (custody, then usage).

    ``{}`` means no figure: a non-root task, or a root with no attributable
    subtree rows (absent, not zero). Consumers: ``LiveRootCosts.refresh`` (the
    figure root heartbeats carry) and the live cost in ``gateway.history``;
    ``supervisor.events`` re-exports it.
    """
    from ouroboros.task_results import resolve_task_lineage

    lineage = resolve_task_lineage(
        task_id, metadata=task.get("metadata"),
        **{key: task.get(key, event.get(key)) for key in (
            "root_task_id", "parent_task_id", "delegation_role",
            "original_task_id", "timeout_retry_from",
        )},
    )
    if not lineage["is_root_task"]:
        return {}
    try:
        from ouroboros.usage_accounting import usage_breakdown

        # The supervisor's usage flush runs this on its thread (``LiveRootCosts``): a contended
        # ledger lock serves the last validated snapshot, it never parks the loop behind money.
        usage = usage_breakdown(
            pathlib.Path(task.get("budget_drive_root") or drive_root),
            root_task_id=str(lineage["root_task_id"] or task_id), allow_stale=True, custody_root=drive_root, include_custody=True,
        )
        # A root-filtered summary with no attributable rows is an empty view,
        # not measured zero.  The task-detail reader applies the same rule:
        # legacy metadata alone and an empty subtree must not publish a money
        # amount whose source never observed this root.
        counts = usage.get("attempt_counts")
        if isinstance(counts, Mapping):
            attributable_rows = sum(
                max(0, int(value or 0))
                for key, value in counts.items()
                if key != "metadata_only"
            )
            sessions = max(0, int(usage.get("subscription_sessions") or 0))
            from ouroboros.delegate_custody_usage import cost_coverage_open
            if attributable_rows == 0 and sessions == 0 and not cost_coverage_open(usage):
                return {}
        unknown = int(usage.get("unknown_unmetered") or 0)
        reserved = float(usage.get("reserved_usd") or 0)
        unresolved = float(usage.get("unresolved_upper_bound_usd") or 0)
        accounted = honest_accounted_amount(usage)
        projection = {
            "cost_accounting_status": "available",
            "accounted_upper_bound_usd_with_children": (
                round(accounted, 6) if accounted is not None else None
            ),
            "reserved_usd": reserved,
            "unresolved_upper_bound_usd": unresolved,
            "unknown_unmetered": unknown,
            "non_final_rows": int(usage.get("non_final_rows") or 0),
            "ledger_integrity_degraded": bool(usage.get("integrity_degraded")),
            "cost_custody": usage.get("cost_custody"),
            # Built from THIS bucket, so the amount and its explanation are one fact.
            "cost_presentation": build_cost_presentation(usage, scope=COST_SCOPE_ROOT_TREE),
        }
    except Exception:
        log.debug("Root heartbeat cost unavailable for %s", task_id, exc_info=True)
        projection = {
            "cost_accounting_status": "unavailable",
            "cost_accounting_error": "ledger_unavailable",
            "accounted_upper_bound_usd_with_children": None,
        }
    projection.update(cost_final=False, cost_with_children_partial=True)
    return with_cost_aliases(projection)


class LiveRootCosts:
    """The cost figure a root heartbeat carries, read again only when that root's money moved.

    A root heartbeat used to pay ``live_root_cost_projection`` on the supervisor drain itself — a
    custody snapshot of ``current.sqlite``, a ``usage.sqlite`` read and two path resolutions, per
    running root, every 30 s (69 % of the drain's handler time on 01.11) — though nothing about the
    money had changed between two beats. The figure is now that same projection, held per root and
    read once per CHANGE of the root:

    * ``mark(drive_root, root_task_id)``: the root's subtree reported usage (the ``llm_usage``
      handler, one per model call);
    * ``fields(drive_root, task_id, task, event)``: what one heartbeat carries, with no store read —
      the held figure of a root, ``{}`` for a child, for a root without attributable rows and for a
      root not read yet (absent, never zero). A root it meets for the first time in this server
      generation is marked, so its figure exists from the next beat on;
    * ``refresh(running, drive_root)``: the supervisor's one usage flush per loop turn, after bridge
      intake (``server_liveness.flush_budget_projection``), reads only the marked roots that still
      run and forgets roots that left RUNNING. Nothing marked, nothing read.

    Delegated-run custody is read with the usage in the same projection, so a run that started or
    settled with no model call of the subtree since reaches the figure with the subtree's next usage
    row: the figure is non-final display (``cost_final`` is False), never admission (§10.28). A read
    that failed keeps the last figure (the stale-allowed display rule); without one it shows the
    failure (``cost_accounting_status: unavailable``), and either way the root is read again no
    sooner than ``BUDGET_PROJECTION_RETRY_SEC``, the retry bound of the sibling budget projection.
    Keys carry the drive root, so one process's memo never answers for another root. The figure is
    keyed by the LOGICAL root (the projection's subtree) and remembers which RUNNING task carries it
    (a hard-timeout retry root runs under a fresh physical id).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._figures: Dict[tuple, Dict[str, Any]] = {}
        self._carriers: Dict[tuple, str] = {}  # key -> the RUNNING task id whose heartbeat carries it
        self._due: Dict[tuple, float] = {}  # key -> monotonic time from which the next read may run

    @staticmethod
    def _key(drive_root: Any, root_task_id: Any) -> tuple:
        return (str(drive_root or ""), str(root_task_id or "").strip())

    def mark(self, drive_root: Any, root_task_id: Any) -> None:
        key = self._key(drive_root, root_task_id)
        if key[1]:
            with self._lock:
                self._due[key] = 0.0  # a change is read at the next flush, whatever retry was pending

    def fields(self, drive_root: Any, task_id: str, task: Mapping[str, Any],
               event: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        from ouroboros.task_results import resolve_task_lineage

        event = event if isinstance(event, Mapping) else {}
        lineage = resolve_task_lineage(task_id, metadata=task.get("metadata"), **{
            key: task.get(key, event.get(key)) for key in (
                "root_task_id", "parent_task_id", "delegation_role", "original_task_id", "timeout_retry_from")})
        if not lineage["is_root_task"]:
            return {}
        key = self._key(drive_root, lineage["root_task_id"] or task_id)
        with self._lock:
            self._carriers[key] = str(task_id)
            figure = self._figures.get(key)
            if figure is None:
                self._due.setdefault(key, 0.0)
        return dict(figure or {})

    def refresh(self, running: Any, drive_root: Any) -> None:
        if drive_root is None or not isinstance(running, Mapping):
            return
        from ouroboros.runtime_limits import BUDGET_PROJECTION_RETRY_SEC

        now, root = time.monotonic(), str(drive_root)
        with self._lock:
            due = [key for key, at in self._due.items() if key[0] == root and at <= now]
            for key in due:
                del self._due[key]
            if due:  # a pass with work also forgets the roots that left RUNNING
                for key in [key for key, carrier in self._carriers.items()
                            if key[0] == root and carrier not in running]:
                    self._carriers.pop(key, None)
                    self._figures.pop(key, None)
            carriers = {key: self._carriers.get(key) for key in due}
        for key, carrier in carriers.items():
            meta = running.get(carrier) if carrier else None
            task = meta.get("task") if isinstance(meta, dict) else None
            if not isinstance(task, dict):
                continue  # no heartbeat carries this root here: nothing to read for
            projection = live_root_cost_projection(carrier, task, {}, drive_root)
            failed = projection.get("cost_accounting_status") == "unavailable"
            with self._lock:
                if not (failed and key in self._figures):
                    self._figures[key] = projection
                if failed:
                    self._due.setdefault(key, now + BUDGET_PROJECTION_RETRY_SEC)


# The server process's one memo (the supervisor thread marks, reads and flushes it).
LIVE_ROOT_COSTS = LiveRootCosts()


def cost_projection(source: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Read-side projection of a task-result-like mapping into the SSOT shape.

    Emits ``accounted_upper_bound_usd`` (None when unknown; a stored legacy
    ``cost_usd`` spelling still resolves, deprecated wins — ABI 7.0 read
    tolerance), ``cost_known`` (whether ANY amount is accounted),
    ``cost_final`` (source-claimed finality about a KNOWN amount, never
    fabricated), the with-children name when the source carries the pair, and
    every openness flag the source has.
    """
    src = source if isinstance(source, Mapping) else {}
    _, amount = honest_cost_pair_amount(src, *COST_ALIAS_PAIRS[0])
    out: Dict[str, Any] = {
        "accounted_upper_bound_usd": amount,
        "cost_known": amount is not None,
        "cost_final": bool(src.get("cost_final")) and amount is not None,
    }
    present, with_children = honest_cost_pair_amount(src, *COST_ALIAS_PAIRS[1])
    if present:
        out["accounted_upper_bound_usd_with_children"] = with_children
    for key in COST_OPENNESS_FIELDS:
        if key in src and key not in out:
            out[key] = src[key]
    return out


def _live_cost_source(source: Any) -> bool:
    from ouroboros.task_status import NONTERMINAL_STATUSES

    return (isinstance(source, Mapping) and bool(source.get("task_id") or source.get("id"))
            and str(source.get("status") or "").strip().lower() in NONTERMINAL_STATUSES | {"cancel_requested"})


def current_cost(source: Mapping[str, Any], drive_root: Any, *, breakdown=None) -> Dict[str, Any]:
    """Overlay live accounting after effective-result resolution; terminal receipts stay intact."""
    from supervisor.state import reconstruct_task_cost

    if not _live_cost_source(source):
        return dict(source)
    task_id = str(source.get("task_id") or source.get("id"))
    fresh = reconstruct_task_cost(task_id, fields=True, drive_root=drive_root, breakdown=breakdown)
    if isinstance(fresh.get("cost_presentation"), dict) and not fresh["cost_presentation"]["has_rows"]:
        fresh["accounted_upper_bound_usd"] = None  # no ledger rows do not establish a free task
    return {**with_cost_aliases(source), **fresh, "cost_final": False}


def current_costs(sources: Any, drive_root: Any) -> list:
    """Read each requested task tree once, independently of the observer's tree."""
    from ouroboros.task_results import resolve_task_lineage
    from ouroboros.usage_accounting import usage_breakdown

    rows, breakdowns = list(sources), {}
    for index, row in enumerate(rows):
        if not _live_cost_source(row):
            continue
        root_task_id = resolve_task_lineage(row.get("task_id") or row.get("id"),
            root_task_id=row.get("root_task_id"), metadata=row.get("metadata"))["root_task_id"]
        if root_task_id not in breakdowns:
            try:
                breakdowns[root_task_id] = usage_breakdown(drive_root, root_task_id=root_task_id, include_custody=True)
            except Exception as exc:
                log.warning("Batch task cost unavailable for %s: %s: %s", root_task_id, type(exc).__name__, exc)
                breakdowns[root_task_id] = None
        breakdown = breakdowns[root_task_id]
        rows[index] = current_cost(row, drive_root, breakdown=breakdown) if breakdown is not None else {
            **with_cost_aliases(row), "cost_accounting_status": "unavailable", "cost_final": False,
            "cost_accounting_error": "ledger_unavailable", "cost_presentation": None,
            "accounted_upper_bound_usd": None, "total_rounds": None, "prompt_tokens": None,
            "completion_tokens": None, "reserved_usd": None, "unresolved_upper_bound_usd": None,
            "unknown_unmetered": None, "non_final_rows": None, "ledger_integrity_degraded": True,
        }
    return rows


def usd_text(amount: float, decimals: int = 2) -> str:
    """A known dollar amount: ``decimals`` places, at least four under a cent and ``<$0.0001`` (at four
    places) below what they show, so a positive amount never prints as the priced zero ``$0.00`` (the web
    card's ``formatUsdKnown``); a caller asking for more places keeps them."""
    from decimal import ROUND_HALF_UP, Context, Decimal

    if not math.isfinite(amount):
        return "unknown"
    places = max(decimals, 4) if 0 < amount < 0.01 else decimals
    if 0 < amount < 10 ** -places:
        return f"<${10 ** -places:.{places}f}"
    # The exact binary value, a tie going up, as the card's toFixed and Telegram round it; the context holds
    # every digit of the result, however large the amount.
    exact = Decimal(amount)
    context = Context(prec=max(28, exact.adjusted() + places + 2))
    return f"${exact.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP, context=context)}"


def uncounted_money_text(summary: Optional[Mapping[str, Any]], *, decimals: int = 2, prefix: str = "") -> str:
    """What a limit does not count as spending, each part under its own name and only when present (P8:
    known spend, estimates and unresolved charges stay distinct): reservations of calls not yet sent, and
    the upper bound of calls without a known price (in flight, or ended without one). '' when neither.
    ``prefix`` reads an axis's fields (``global_reserved_usd``...)."""
    src = summary if isinstance(summary, Mapping) else {}
    reserved, unresolved = (_as_amount(src.get(prefix + key)) or 0.0 for key in ("reserved_usd", "unresolved_upper_bound_usd"))
    return "; ".join(part for part in (
        f"{usd_text(reserved, decimals)} reserved for calls not yet sent" if reserved > 0 else "",
        f"up to {usd_text(unresolved, decimals)} for calls without a known price" if unresolved > 0 else "") if part)


def cost_display(source: Optional[Mapping[str, Any]], *, decimals: int = 2) -> str:
    """Human rendering of one task's cost. Null NEVER renders as ``$0.00``.

    ``$1.23`` for a final amount, ``$1.23 (upper bound, not final)`` for an open
    one; with nothing accounted, the reason as known: ``unknown (accounting
    unavailable)``, ``unknown (not settled yet)`` (its calls are still open),
    ``unknown (no price recorded for its calls)`` (the ledger read, its rows carry
    no price, e.g. subscription work), ``unknown (a delegated run closed without a
    cost record)``, ``unknown (no usage recorded)`` (an intact, closed scope without
    any row: the card's absent money line, never ``$0.00``), else ``unknown (accounting
    unavailable or not settled)``. ``Tracked: $1.23 (total unknown)`` preserves a
    known subtotal when unpriced work prevents a total or an upper bound.
    """
    projection = cost_projection(source)
    amount = projection["accounted_upper_bound_usd"]
    src = source if isinstance(source, Mapping) else {}
    carrier = projection.get("cost_presentation") or src.get("cost_presentation")
    carrier = carrier if isinstance(carrier, Mapping) else {}
    if carrier.get("has_rows") is False:
        amount = None  # no row in the carrier's scope (own or its tree): a legacy zero field proves nothing
    if amount is None:
        custody = src.get("cost_custody") if isinstance(src.get("cost_custody"), Mapping) else {}
        # As the card says it (costPresentationMeta): an unreadable ledger, custody view or quarantined ledger.
        if (src.get("cost_accounting_status") == "unavailable" or src.get("ledger_integrity_degraded") is True
                or custody.get("status") == "unavailable"):
            return "unknown (accounting unavailable)"
        if carrier.get("accounting_open"):
            return "unknown (not settled yet)"  # a price may still arrive
        if carrier.get("has_rows"):
            return "unknown (no price recorded for its calls)"  # a readable ledger, rows without a price
        if carrier.get("has_rows") is False:
            return ("unknown (a delegated run closed without a cost record)" if carrier.get("has_unpriced")
                    else "unknown (no usage recorded)")
        return "unknown (accounting unavailable or not settled)"
    text = usd_text(amount, decimals)  # $0.00 is a priced zero only
    from ouroboros.delegate_custody_usage import cost_coverage_open
    if carrier.get("has_unpriced") or projection.get("unknown_unmetered") or cost_coverage_open(projection):
        return f"Tracked: {text} (total unknown)"
    if projection["cost_final"]:
        return text
    return f"{text} (upper bound, not final)"


__all__ = [
    "COST_ALIAS_PAIRS",
    "COST_OPENNESS_FIELDS",
    "COST_SCOPE_OWN",
    "COST_SCOPE_ROOT_TREE",
    "CostPresentation",
    "LIVE_ROOT_COSTS",
    "LiveRootCosts",
    "build_cost_presentation",
    "current_cost",
    "current_costs",
    "carry_cost_meta",
    "cost_display",
    "cost_projection",
    "honest_cost_pair_amount",
    "honest_accounted_amount",
    "live_root_cost_projection",
    "normalize_task_result_cost_planes",
    "resolve_cost_pair",
    "with_cost_aliases",
    "with_task_cost_presentation",
]
