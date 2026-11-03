"""Durable custody for delegated (Claudexor) runs: a run lives inside the Claudexor daemon
and survives our worker, so its custody is a row of the current-facts store
(``delegate_custody_current``: the event reduced and parked in one transaction, then appended
to the chain ``logs/events.jsonl`` — the forensic record — never a module dict). Three lookup
answers, never two: OWNED, FOREIGN (another task started it) and UNKNOWN (no durable record).
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import pathlib
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple  # noqa: F401

from ouroboros._usage_rows import REVIEW_ATTRIBUTION_KEYS  # noqa: F401 -- delegate_pending uses the facade
from ouroboros.delegate_custody_usage import (
    disclosed_spend,
    disclosed_tokens,  # noqa: F401 -- public custody vocabulary used by collection
    held_run_names,
    names_another_run,
    project_retirement_lock,
    summary_of,
)
from ouroboros.utils import utc_now_iso
log = logging.getLogger(__name__)
# The harness's own terminal vocabulary — one definition for the tool, the settler and
# the reconciler (a second copy is how a "cancelled" run stayed live on one branch).
TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled", "interrupted"})
# The terminal states in which the run DID reach its contract write and DID act, so
# absent containment evidence is a missing proof rather than "too early to tell".
SUCCEEDED_STATES = frozenset({"succeeded"})
# Durable row kinds. Every one of them is written to the event log the agent, the
# supervisor and the forensic tooling already read.
START_REQUESTED = "delegate_run_start_requested"
STARTED = "delegate_run_started"
START_FAILED = "delegate_run_start_failed"
TERMINAL_OBSERVED = "delegate_run_terminal_observed"
INTERACTIONS_OBSERVED = "delegate_run_interactions_observed"
LEDGER_RECORDED = "delegate_run_ledger_recorded"
SETTLED = "delegate_run_settled"
SETTLE_FAILED = "delegate_run_settle_failed"
# The daemon answered that it has no such run. Custody closes WITHOUT a settlement: there
# is no terminal detail to settle against, and inventing a $0 ledger row for a run nobody
# can describe is the one shape the ledger must never produce.
CLOSED_ABSENT = "delegate_run_closed_absent"
STORE_BOUND = "delegate_run_store_bound"
PROJECT_RETIRED = "delegate_run_project_retired"
PROJECT_RETIRE_FAILED = "delegate_run_project_retire_failed"
CANCEL_OUTCOME = "delegate_run_cancel_outcome"
UNCONFINED = "delegate_run_unconfined"
CONTAINMENT_FAULT = "delegate_run_containment_fault"
CONTAINMENT_RESOLVED = "delegate_run_containment_resolved"
RECONCILED = "delegate_run_reconciled"
OUTPUT_SPILLED = "delegate_run_output_spilled"
# Owner doctrine D7: a result is OBTAINED only after the artifact is read to
# EOF - the ack covers the staged artifact CONTINUOUSLY start-to-EOF,
# hash-bound, over verified FULL content only (never the 256 KiB preview).
# Owner directive 2026-08-03: consumption is LOAD-BEARING before settlement;
# its ABSENCE is ``SETTLED_UNREAD``; it does not BLOCK settling (settle_run).
OUTPUT_CONSUMED = "delegate_run_output_consumed"
# A run whose VERIFIED FULL output was staged, paid for, and settled without ever being
# read to EOF: the "launched and never collected" class, named at the moment it becomes
# permanent instead of inferred later from a field nobody joined.
SETTLED_UNREAD = "delegate_run_settled_unread"
# #1573: the owner's ACCEPTED terminal wake for this run was appended to the transcript
# of ``attempt_key`` (written only after the supervision acknowledgement landed).
TERMINAL_ACKNOWLEDGED = "delegate_run_terminal_acknowledged"
# C1 (delegated-run isolation): a MUTATING run executes in a private snapshot; its
# diff is captured durably at terminal, then EXPLICITLY applied or rejected by the
# nanny. Capture is idempotent (the flag replays); the disposition row is what
# releases the snapshot for GC — until then conflict material persists on disk.
PATCH_CAPTURED = "delegate_run_patch_captured"
NAMED_EVIDENCE_RETAINED = "delegate_run_named_evidence_retained"
COLLECTION_ATTEMPTED = "delegate_run_collection_attempted"
CAPTURE_JOB_CHANGED = "delegate_run_capture_job_changed"
PATCH_DISPOSED = "delegate_run_patch_disposed"
# Not a row: the replayed disposition of a run whose snapshot a successor's STARTED
# row took over (``_supersede``); its work rides the successor's cumulative capture.
SUPERSEDED = "superseded"
EVIDENCE_RETAINED = "evidence_retained"  # investigation capture; never applied
NO_CHANGES = "no_changes"  # an empty writer result, closed on capture; nothing to apply
# CR1-3 (owed-before-sent, one row earlier than the disposition): APPLY-INTENT lands
# BEFORE the target tree is mutated; RESOLVED lands only when the attempt is KNOWN to
# have left the tree unmutated. An intent with neither resolution nor disposition
# replays as AMBIGUOUS — the tree may carry the patch — and a later reject/apply must
# refuse typed instead of pretending "not applied" over a modified tree.
PATCH_APPLY_STARTED = "delegate_run_patch_apply_started"
PATCH_APPLY_RESOLVED = "delegate_run_patch_apply_resolved"
# Q31: a partial external work order earns authority only through exact, host-
# verified source ranges. These append-only receipts survive worker restart and
# are merged by the same custody replay that owns every other run fact.
SOURCE_RANGE_VERIFIED = "delegate_run_work_order_source_range_verified"
SOURCE_RANGE_DELIVERY_CONFIRMED = "delegate_run_work_order_source_delivery_confirmed"

# Cheap prefilter: every custody row's type starts with this, so a multi-hundred-MB
# event log is scanned without JSON-parsing the 99.9% of lines that are not ours.
_ROW_MARKER = "delegate_run"

# Lookup answers.
OWNED = "owned"
FOREIGN = "foreign"
UNKNOWN = "unknown"

# Cancel outcomes. This vocabulary is what the tool reports verbatim: nothing may say
# terminal or cancelled without a verified terminal receipt.
CANCEL_REQUESTED = "requested"
CANCEL_CONFIRMED = "confirmed"
CANCEL_FAILED = "failed"
CANCEL_CONTAINMENT_FAULT = "containment_fault_run_may_still_be_live"

@dataclass
class RunCustody:
    """One delegated run's lifecycle facts, replayed from the durable rows."""

    run_id: str = ""
    task_id: str = ""
    route_id: str = ""
    model: str = ""
    # Requested pin (`credentialProfileId`); '' = automatic; applied half = final-attempt telemetry.
    profile_id: str = ""
    effort: Optional[str] = None
    processing_preference: Optional[str] = None
    prompt_cache_ttl: Optional[Dict[str, Any]] = None
    row_effort: Optional[str] = None  # the configured row's pin ('' = Auto); None on rows without one
    project_id: str = ""
    project_owned: bool = False
    # #362: a stable user-target registration outlives any single run.
    project_persistent: bool = False
    root_task_id: str = ""
    parent_task_id: str = ""
    category: str = "subagent"
    source: str = "delegated_subagent"
    review_skill: str = ""
    review_wave_id: str = ""
    review_slot_id: str = ""
    ledger_root: str = ""
    idempotency_key: str = ""
    # Minted once per logical invocation and reused verbatim as the wire
    # Idempotency-Key only for an exact transport retry; deliberately new work
    # gets a fresh id. ``idempotency_key`` above only finds pending work.
    invocation_id: str = ""
    daemon_store_id: str = ""  # immutable engine store UUID; empty legacy authority stays unknown
    # Configured-actor binding. Empty on historical/root-legacy delegate starts.
    selected_subagent_id: str = ""
    config_fingerprint: str = ""
    work_order_fingerprint: str = ""
    work_order_coverage: str = ""
    work_order_source_request: Dict[str, Any] = field(default_factory=dict)
    verified_source_ranges: List[Tuple[int, int]] = field(default_factory=list)
    authority_fingerprint: str = ""
    ledger_recorded: bool = False
    settled: bool = False
    terminal_state: str = ""  # Engine terminal observation; independent of monetary settlement.
    terminal_source: Dict[str, Any] = field(default_factory=dict)
    named_evidence: Dict[str, Any] = field(default_factory=dict)
    collection_attempt: Dict[str, Any] = field(default_factory=dict)
    capture_job: Dict[str, Any] = field(default_factory=dict)
    observation_seq: int = -1
    interaction_source: Dict[str, Any] = field(default_factory=dict)
    pending_interaction_ids: List[str] = field(default_factory=list)
    terminal_reason: str = ""  # #1196: engine ``outcomeFacts.reason`` replayed from SETTLED ("" = none)
    settled_at: str = ""  # the SETTLED row's own timestamp
    # #1573: SETTLED's engine locators (runDir, primary kind/path/bytes/truncated, no text); None = legacy.
    engine_locator: Optional[Dict[str, Any]] = None
    terminal_ack_attempt: str = ""  # the attempt whose transcript took the terminal wake
    execution_mode: str = ""
    effective_access: str = ""
    thread_id: str = ""
    authority_ceiling: Dict[str, Any] = field(default_factory=dict)
    promotion: Dict[str, Any] = field(default_factory=dict)
    continuation_of: str = ""  # the settled run this one explicitly continued (``continue_from``)
    # Same-tree continuation (``delegate_continuation``): a run that continues IN its
    # predecessor's undisposed snapshot records its own capture identity (``capture_id``,
    # its invocation id) and the task whose artifact directory keeps that snapshot's
    # baseline and ONE disposition lock (``snapshot_task_id``); "" = the run's own.
    capture_id: str = ""
    snapshot_task_id: str = ""
    superseded_by: str = ""  # replayed: the run that continued in this run's snapshot
    containment_disclosed: bool = False  # written once; a re-poll must not re-find
    unread_disclosed: bool = False  # settled-never-read omission named durably
    # D7: store-relative ``output_artifact`` (empty inline; see ``output_source``).
    # ``output_complete`` matches the run's OWN reported size or preview prefix (no engine hash:
    # equal length binds to the CLAIM); neither match means incomplete, never acknowledgeable.
    # ``output_sha`` hashes current staging; ``output_consumed`` means its canonical ack FOR THAT HASH, not path.
    output_artifact: str = ""
    output_complete: bool = False
    output_sha: str = ""
    output_terminal_sha: str = ""  # Exact terminal source this retained output materializes.
    output_consumed: bool = False
    # #1573: canonical ``task_source`` ref + ACTUAL staging task (a retry reader's own); {} = legacy task drive.
    output_source: Dict[str, Any] = field(default_factory=dict)
    # Immutable pairs keep custody memo copies independent. Aggregate consumption
    # settles work debt; each reader still owes its own receipt for the current hash.
    output_reader_receipts: Tuple[Tuple[str, str], ...] = ()
    # C1 isolation binding for a MUTATING run: private execution root, diff
    # baseline commit, AUTHORITY target tree. ``snapshot_id`` keys the
    # worktree-service registry entry (= the provisioning invocation id). All
    # durable/replayed: retries reproduce the binding, terminal capture knows
    # where to diff; custody GC also requires proof that the owning task is over.
    snapshot_id: str = ""
    execution_root: str = ""
    execution_binding_fingerprint: str = ""
    baseline_sha: str = ""
    target_root: str = ""
    authority_source: str = ""
    # The GRANTED run shape, replayed off the STARTED row's `shape` dict:
    # delegate_wait replays this as the entitled authority - re-deriving from
    # live context read `readonly` and cancelled the run as widened (R1-2).
    access: str = ""
    mode: str = ""
    isolation: str = ""
    delegated: bool = False
    # Host-minted semantic resource reference for an exact-resource run (logical
    # root, source, skill name, recorded target, baseline payload hash). Consumed
    # only by retry rebind and owned apply rebind; recovery carries it opaquely.
    resource_ref: Dict[str, Any] = field(default_factory=dict)
    # Capture/disposition lifecycle (replayed): capture happens once at terminal;
    # ``patch_disposed`` is "" until the nanny explicitly applies ("applied") or
    # rejects ("rejected") the captured patch — only then may the snapshot be removed —
    # or a successor continues in the snapshot (``SUPERSEDED``: its capture is the one).
    patch_captured: bool = False
    patch_disposed: str = ""
    # #1651: the PATCH_DISPOSED row's own facts, so readers never replay the chain: the task that recorded
    # it ("" = not recorded, as on rows reduced before this field) and a reject over a result proven lost.
    patch_disposed_by: str = ""
    patch_result_lost: bool = False
    # CR1-3: an apply-intent row exists with no resolution and no disposition —
    # the target tree MAY carry the patch (crash between apply and the disposition
    # row), so any later disposition over this run is ambiguous until inspected.
    patch_apply_pending: bool = False
    patch_apply_key: str = ""  # Existing apply intent's engine idempotency key.
    # Facts the readers used to re-read from the chain, now reduced onto the row: the
    # STARTED row's clock (``run_timing``), the SETTLED receipt's disclosed spend, engine
    # model and applied access (``task_execution_evidence``), the open containment
    # incident (``open_containment_faults``), the START_REQUESTED row that minted the
    # run's invocation (``invocation_record``) and its admission bucket (``billing_group``).
    started_ts: str = ""
    max_seconds: int = 0
    cost_usd: Optional[float] = None
    spend_disclosed: bool = False
    spend_estimated: bool = False
    cost_partial: bool = False
    engine_model: str = ""
    access_profile: str = ""
    containment_fault: Dict[str, Any] = field(default_factory=dict)
    request_event: Dict[str, Any] = field(default_factory=dict)
    billing_group: Dict[str, Any] = field(default_factory=dict)

    @property
    def review_owned(self) -> bool:
        """A run a REVIEW surface registered is owned by its panel, not by this
        task's delegation lifecycle: the panel bounds it, and the task's own
        terminal is never a verdict about its reviewer (issue #1006)."""
        return review_owned_source(self.source)


# Process-local MEMOIZATION of the rows above — never the authority. A miss falls
# through to the durable scan, which is why a restart no longer loses custody.
_CUSTODY: Dict[str, RunCustody] = {}
# -- durable record ------------------------------------------------------------

def event_log_path(drive_root: Any) -> pathlib.Path:
    return pathlib.Path(drive_root) / "logs" / "events.jsonl"

def custody_root(ctx: Any) -> pathlib.Path:
    """The drive whose event log is the custody authority for this context.

    A live subagent's isolated child drive is pruned, so a custody row written
    there cannot outlive the run; the canonical (budget) root is the durable SSOT.
    """
    from ouroboros.tool_access import canonical_data_root

    return canonical_data_root(ctx)

def emit(drive_root: Any, kind: str, payload: Dict[str, Any], *, propagate_contention: bool = False) -> bool:
    """Record one custody event and REPORT whether it landed; normally never raises.

    The store is the authority: the event is reduced onto the run's row and parked in
    the custody outbox in ONE transaction (``delegate_custody_current.publish``), then
    appended to the event chain — the forensic record — and the outbox row is deleted
    only after that append confirmed. A death between the COMMIT and the append leaves
    the outbox row, which the next publication (or the boot door) appends; the chain
    fold dedups by ``event_id``. Discarding the fate is how a run could be reported as
    started, waited on and settled while the rows that ARE its custody never landed.
    An unsent start owner may opt into proven pre-COMMIT contention, never flush replay.
    """
    try:
        from ouroboros.delegate_custody_current import publish

        event = {"ts": utc_now_iso(), "type": kind, "event_id": uuid.uuid4().hex, **payload}
        written = (publish(drive_root, event, propagate_contention=True) if propagate_contention
                   else publish(drive_root, event))
    except Exception as exc:
        if propagate_contention and getattr(exc, "custody_not_committed", False):
            raise
        log.warning("delegate custody row could not be written (%s)", kind, exc_info=True)
        return False
    if not written:
        log.warning("delegate custody row was rejected by the task store (%s)", kind)
    elif kind == START_FAILED:
        from ouroboros.subagent_history import record_session_start_failure
        try:
            record_session_start_failure(drive_root, event)
        except Exception:
            log.debug("Start history unavailable", exc_info=True)
    return written

def daemon_says_absent(exc: Any, *, gateway: Any = None, run: Optional[RunCustody] = None) -> bool:
    """A run's 404 proves absence in its owning store; project cleanup uses raw HTTP.

    A run bound to a store (handshake ``storeId``) is absent only for that store. An unbound
    run (started before engines named their store, or never read) keeps the earlier contract:
    the one managed daemon's 404 closes it.
    """
    return int(getattr(exc, "status_code", 0) or 0) == 404 and (run is None or _absence_provable(run, gateway))


def _absence_provable(run: RunCustody, gateway: Any) -> bool:
    return not run.terminal_source and (not run.daemon_store_id
                                        or run.daemon_store_id == getattr(gateway, "store_identity", ""))


def bind_daemon_store(drive_root: Any, gateway: Any, entry: RunCustody) -> None:
    """Bind a successful exact run read, never infer identity from a 404 or PID."""
    identity = getattr(gateway, "store_identity", "")
    if isinstance(identity, str) and identity and not entry.daemon_store_id:
        if emit(drive_root, STORE_BOUND, {"run_id": entry.run_id, "task_id": entry.task_id,
                "invocation_id": entry.invocation_id, "daemon_store_id": identity}):
            entry.daemon_store_id = identity

PROJECT_HAS_THREADS = "project_has_threads"  # the engine's typed refusal of a project DELETE

def daemon_keeps_project(exc: Any) -> bool:
    """True when the daemon ANSWERED that it keeps the project: it still holds threads
    on it. The twin of ``daemon_says_absent`` — a definite fact from a reachable daemon,
    and a PERMANENT one for the host (Ouroboros creates sticky review threads scoped to
    the project root and the gateway has no thread delete), so retrying it on a timer
    costs a daemon round trip per sweep forever. The CODE decides, never the message —
    a transient, a 5xx or an unreadable body is a failure to find out, still retryable.
    """
    return str(getattr(exc, "code", "") or "") == PROJECT_HAS_THREADS

def custody_log_unreadable(drive_root: Any) -> bool:
    """Whether the custody event log EXISTS but cannot be opened (GR6-4).

    ``_iter_rows`` swallows its own ``OSError`` — the right behavior for the
    fail-soft readers — but the KILL/MISS/REAP AUDIT must not let an
    unreadable log audit as "cleanly reconciled": ABSENT is a positively
    established empty state (no custody row could exist), while
    existing-but-unreadable means the open-run answer is UNKNOWN. Same probe
    the evidence reader (``task_execution_evidence``) already uses; its own
    semantics are unchanged. The store holds the current custody rows, so the
    probe asks the store; the chain probe below is the importer's.
    """
    from ouroboros.delegate_custody_current import unreadable

    return unreadable(drive_root)


def custody_chain_unreadable(drive_root: Any) -> bool:
    """The chain-only probe of ``custody_log_unreadable``: what the importer asks while it
    folds the chain into the store (the store cannot be consulted about itself)."""
    from ouroboros.utils import JsonlChainUnreadable, jsonl_archive_segments

    path = event_log_path(drive_root)
    try:
        segments = jsonl_archive_segments(path, strict=True)
    except JsonlChainUnreadable:
        return True
    for candidate in (*segments, path):
        try:
            if not candidate.exists():
                continue
            with candidate.open("rb"):
                pass
        except OSError:
            return True
    return False


@contextmanager
def actor_decision_lock(drive_root: Any, task_id: str) -> Iterator[None]:
    """Serialize one actor's mutually exclusive zero-run/start decisions.

    The custody log and the verification-receipt store are separate append-only
    authorities.  This short critical section joins only their decision edge:
    re-read authority before a zero-run receipt, START_REQUESTED or absent-run
    closure. Network operations stay outside it.
    """

    from ouroboros.platform_layer import (
        acquire_exclusive_file_lock,
        release_exclusive_file_lock,
    )

    identity = str(task_id or "").strip()
    if not identity:
        raise ValueError("actor decision lock requires a task id")
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    lock_path = pathlib.Path(drive_root) / "state" / "delegate_actor_claims" / f"{digest}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = acquire_exclusive_file_lock(lock_path, timeout_sec=20.0, stale_sec=120.0)
    if fd is None:
        raise TimeoutError("actor decision lock is unavailable")
    try:
        yield
    finally:
        release_exclusive_file_lock(lock_path, fd)

def chain_length(path: pathlib.Path) -> int:
    """The chain's byte length across its segments (strict): the boundary a store records."""
    return chain_measure(path)[0]


def chain_measure(path: pathlib.Path) -> Tuple[int, Optional[Any]]:
    """``chain_length`` with the live segment's stat the length was read with (``None`` when the live
    file is absent): one listing, so the two describe the same instant of the chain."""
    from ouroboros.utils import jsonl_chain_handles

    snapshot: Dict[str, Any] = {}
    with jsonl_chain_handles(path, strict=True, snapshot=snapshot):
        pass
    live = next((stat for _segment, stat, was_live in snapshot.get("entries", ()) if was_live), None)
    return int(snapshot.get("total") or 0), live


def rows_after(path: pathlib.Path, offset: int) -> Tuple[List[Dict[str, Any]], int]:
    """The custody rows the chain holds after byte OFFSET (strict) and its length before the
    read: what P6' applies on top of a store whose boundary is OFFSET. A chain shorter than
    OFFSET (segments pruned since) is read whole; the fold's dedup and the reducer's idempotent
    replays make that safe."""
    total = chain_length(path)
    rows = list(_iter_rows(path, strict=True, start_offset=offset if offset <= total else 0))
    return rows, total


def _iter_rows(path: pathlib.Path, *, strict: bool = False, start_offset: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Custody rows across the ROTATED CHAIN (archive segments + live file): the importer's
    fold, the explicit rebuild and the forensic replay (``strict``: an unreadable segment or a
    read error raises its ``OSError`` instead of amputating the chain; ``start_offset``: from
    that byte of the chain).

    Archives are read before the live file, so a rotation never amputates older runs' custody;
    ``jsonl_chain_handles`` keeps the walk rotation-race-safe (open-live-first + inode dedup) and
    sequential (the pinned live handle plus one archive), so a long chain under a low descriptor
    limit is read whole, never cut where the opens ran out.
    """
    from ouroboros.utils import jsonl_chain_handles

    with jsonl_chain_handles(path, strict=strict, start_offset=start_offset, sequential=True) as handles:
        for _, handle in handles:
            try:
                for raw in handle:
                    if _ROW_MARKER.encode("ascii") not in raw:
                        continue
                    try:
                        row = json.loads(raw.decode("utf-8", errors="replace"))
                    except ValueError:
                        continue
                    if isinstance(row, dict) and str(row.get("type") or "").startswith(_ROW_MARKER):
                        yield row
            except OSError:
                if strict:
                    raise
                continue


from ouroboros.delegate_registration_policy import (
    STARTED_OPTION_FIELDS as _STARTED_OPTION_FIELDS,
    STARTED_MAPPING_FIELDS as _STARTED_MAPPING_FIELDS,
    STARTED_FIRST_WINS_FACTS as _STARTED_FIRST_WINS_FACTS,
    STARTED_PROGRESS_FLAGS as _STARTED_PROGRESS_FLAGS,
    STARTED_SETTLEMENT_FACTS as _STARTED_SETTLEMENT_FACTS,
    STARTED_STR_FIELDS as _STARTED_STR_FIELDS,
    candidate_apply_possible,
    review_owned_source,
)

from ouroboros.delegate_source_coverage import (
    apply_source_delivery_confirmation,
    apply_source_range_receipt,
    _merge_verified_source_range,
    merge_source_delivery_confirmations,
    record_source_range_verified,
    work_order_source_verification,
)

def _merge_started_into(entry: RunCustody, previous: RunCustody) -> None:
    """Project a duplicate STARTED fact set onto an existing run — the ONE
    merge, used by both the durable replay and the in-process memo (gate fix 8).

    Progress, the settlement's closing facts (cost, disclosure, model/profile,
    terminal state) and an unresolved containment fault always carry forward;
    binding facts are truthy-first-wins per field; the SHAPE GROUP (access/mode/isolation/delegated) plus the resource
    reference is first-wins as a UNIT, keyed on the first row that CARRIED a
    shape — so a recorded ``delegated=False`` survives a later ``True`` and an
    empty ``resource_ref`` is never "filled" by a later row.
    """
    for attr in (*_STARTED_PROGRESS_FLAGS, *_STARTED_SETTLEMENT_FACTS):
        setattr(entry, attr, getattr(previous, attr))
    if previous.containment_fault:
        entry.containment_fault = dict(previous.containment_fault)
    for attr in ("terminal_source", "observation_seq", "interaction_source", "pending_interaction_ids",
                 "output_terminal_sha", "named_evidence", "collection_attempt", "capture_job"):
        setattr(entry, attr, copy.deepcopy(getattr(previous, attr)))
    entry.output_reader_receipts = previous.output_reader_receipts
    entry.daemon_store_id = previous.daemon_store_id or entry.daemon_store_id
    entry.patch_apply_key = previous.patch_apply_key
    entry.project_owned = previous.project_owned and entry.project_owned
    entry.project_persistent = previous.project_persistent or entry.project_persistent
    for attr in _STARTED_FIRST_WINS_FACTS:
        prior = getattr(previous, attr)
        if prior:
            setattr(entry, attr, prior)
    for attr in _STARTED_OPTION_FIELDS:
        if getattr(previous, attr) is not None:
            setattr(entry, attr, getattr(previous, attr))
    for attr in ("work_order_source_request", "prompt_cache_ttl"):
        if getattr(previous, attr):
            setattr(entry, attr, copy.deepcopy(getattr(previous, attr)))
    for start, end in previous.verified_source_ranges:
        _merge_verified_source_range(entry, start, end)
    merge_source_delivery_confirmations(entry, previous)
    if previous.access:
        entry.access, entry.mode = previous.access, previous.mode
        entry.isolation, entry.delegated = previous.isolation, previous.delegated
        entry.resource_ref = previous.resource_ref
        for attr in ("execution_mode", "effective_access", "thread_id", "authority_ceiling", "promotion"):
            setattr(entry, attr, copy.deepcopy(getattr(previous, attr)))


def _supersede(state: Dict[str, RunCustody], entry: RunCustody) -> None:
    """A run that continues IN its predecessor's undisposed snapshot takes that
    snapshot's custody over: the predecessor replays ``SUPERSEDED`` (its work rides
    the successor's cumulative capture, so it is never applied or rejected alone).
    Derived from the successor's STARTED row, the one durable fact of the hand-over,
    so a start that never bound a run leaves the predecessor's obligation intact."""
    prior = state.get(entry.continuation_of) if entry.capture_id and entry.continuation_of else None
    if prior is not None and prior.mode == "plan" and prior.thread_id and prior.thread_id == entry.thread_id:
        prior.superseded_by = entry.run_id  # Thread heads advance even without a writable snapshot.
    if (prior is not None and prior is not entry and prior.snapshot_id
            and prior.snapshot_id == entry.snapshot_id and prior.patch_disposed in ("", SUPERSEDED)):
        prior.patch_disposed, prior.superseded_by = SUPERSEDED, entry.run_id


def _apply(state: Dict[str, RunCustody], row: Dict[str, Any]) -> None:
    run_id = str(row.get("run_id") or "")
    if not run_id:
        return
    kind = str(row.get("type") or "")
    if kind == STARTED:
        ref = row.get("resource_ref")
        source_request = row.get("work_order_source_request")
        entry = RunCustody(
            run_id=run_id,
            daemon_store_id=str(row.get("daemon_store_id") or ""),
            project_owned=bool(row.get("project_owned")),
            project_persistent=bool(row.get("project_persistent")),
            delegated=row.get("delegated") is True,
            resource_ref=dict(ref) if isinstance(ref, dict) else {},
            **{key: copy.deepcopy(row[key]) for key in _STARTED_MAPPING_FIELDS if isinstance(row.get(key), dict)},
            work_order_source_request=dict(source_request) if isinstance(source_request, dict) else {},
            prompt_cache_ttl=copy.deepcopy(row["prompt_cache_ttl"]) if isinstance(row.get("prompt_cache_ttl"), dict) else None,
            **{attr: str(row.get(key) or "") for attr, key in _STARTED_STR_FIELDS},
            **{key: row[key] for key in _STARTED_OPTION_FIELDS if isinstance(row.get(key), str)},
        )
        entry.category = entry.category or "subagent"
        entry.source = entry.source or "delegated_subagent"
        entry.started_ts = str(row.get("ts") or "")
        try:
            entry.max_seconds = int(row.get("max_seconds") or 0)
        except (TypeError, ValueError):
            entry.max_seconds = 0
        previous = state.get(run_id)
        setattr(entry, "_source_delivery_confirmations", [])
        if previous is not None:
            _merge_started_into(entry, previous)
            entry.request_event, entry.billing_group = previous.request_event, previous.billing_group
        state[run_id] = entry
        _supersede(state, entry)
        return
    custody = state.get(run_id)
    # A terminal/incident row may outlive STARTED or follow a refused start;
    # preserve the event's own identity and source in live and import paths.
    if custody is None and kind in (SETTLED, CONTAINMENT_FAULT, TERMINAL_OBSERVED):
        custody = state[run_id] = RunCustody(
            run_id=run_id, task_id=str(row.get("task_id") or ""), route_id=str(row.get("route") or ""),
            root_task_id=str(row.get("root_task_id") or ""), parent_task_id=str(row.get("parent_task_id") or ""),
            invocation_id=str(row.get("invocation_id") or ""),
            project_id=str(row.get("project_id") or ""), source=str(row.get("source") or ""))
    if custody is None:
        return
    if kind == STORE_BOUND and not custody.daemon_store_id:
        if row.get("task_id") == custody.task_id and row.get("invocation_id") == custody.invocation_id:
            custody.daemon_store_id = str(row.get("daemon_store_id") or "")
    if kind in (TERMINAL_OBSERVED, INTERACTIONS_OBSERVED):
        seq = row.get("observation_seq")
        if not isinstance(seq, int) or isinstance(seq, bool) or seq < custody.observation_seq:
            return
        source = row.get("source_ref")
        if not isinstance(source, dict) or not source.get("sha256"):
            return
        if kind == TERMINAL_OBSERVED:
            terminal = str(row.get("state") or "")
            if terminal not in TERMINAL_STATES or (custody.terminal_state and custody.terminal_state != terminal):
                return  # No arrival-order resurrection or undocumented correction.
            custody.terminal_state = terminal
            custody.terminal_reason = str(row.get("reason") or "")
            custody.terminal_source = dict(source)
            custody.pending_interaction_ids = []
        elif not custody.terminal_state:
            custody.interaction_source = dict(source)
            custody.pending_interaction_ids = [str(value) for value in row.get("interaction_ids", [])]
        custody.observation_seq = seq
    elif kind == LEDGER_RECORDED:
        custody.ledger_recorded = True
    elif kind == UNCONFINED:
        # The unconfined finding is written once per RUN, not once per process. Replaying
        # it means a restarted worker that polls an already-terminal run does not append a
        # second identical disclosure, which would read as a second finding.
        custody.containment_disclosed = True
    elif kind == SETTLED_UNREAD:
        custody.unread_disclosed = True
    elif kind == PROJECT_RETIRED:
        # Recorded on SUCCESS too: without it, a retirement before a failed ledger write replays as owned after a
        # restart, and the retry would keep failing on an already-removed project.
        project_id = str(row.get("project_id") or custody.project_id or "")
        for sibling in state.values():
            if sibling is custody or (project_id and sibling.project_id == project_id):
                sibling.project_owned = False
    elif kind == OUTPUT_SPILLED:
        if row.get("staged") and str(row.get("artifact") or ""):
            custody.output_artifact = str(row.get("artifact") or "")
            custody.output_source = dict(row["source"]) if isinstance(row.get("source"), dict) else {}
            # Strict: a row that does not POSITIVELY claim full content replays as
            # incomplete. An absent flag must never bless a preview as the result.
            custody.output_complete = row.get("full_content") is True
            sha = str(row.get("sha256") or "")
            if custody.output_consumed and sha and custody.output_sha and sha != custody.output_sha:
                # Different content re-staged at the same path: the old acknowledgement
                # named other bytes and does not transfer to content never read.
                custody.output_consumed = False
            custody.output_sha = sha
            custody.output_terminal_sha = str(row.get("terminal_sha") or custody.output_terminal_sha)
    elif kind == OUTPUT_CONSUMED:
        ack_sha = str(row.get("sha256") or "")
        # The ack is HASH-BOUND: it marks consumed only the content it actually names.
        # A stale ack row (older than the current staging) must not bless new bytes.
        if not ack_sha or not custody.output_sha or ack_sha == custody.output_sha:
            custody.output_consumed = True
            reader = str(row.get("reader_task_id") or custody.task_id)
            custody.output_reader_receipts = tuple(
                pair for pair in custody.output_reader_receipts if pair[0] != reader
            ) + ((reader, ack_sha),)
    elif kind == PATCH_CAPTURED:
        custody.patch_captured = True
    elif kind in (NAMED_EVIDENCE_RETAINED, COLLECTION_ATTEMPTED):
        if isinstance(row.get("outcome"), dict):
            setattr(custody, "named_evidence" if kind == NAMED_EVIDENCE_RETAINED else "collection_attempt",
                    copy.deepcopy(row["outcome"]))
    elif (kind == CAPTURE_JOB_CHANGED and isinstance(row.get("outcome"), dict)
          and row.get("expected_id") == custody.capture_job.get("id", "")):
        custody.capture_job = copy.deepcopy(row["outcome"])
    elif kind == PATCH_APPLY_STARTED:
        custody.patch_apply_pending = True
        custody.patch_apply_key = str(row.get("apply_idempotency_key") or "")
    elif kind == PATCH_APPLY_RESOLVED:
        custody.patch_apply_pending = False
    elif kind == SOURCE_RANGE_VERIFIED:
        apply_source_range_receipt(custody, row)
    elif kind == SOURCE_RANGE_DELIVERY_CONFIRMED:
        apply_source_delivery_confirmation(custody, row)
    elif kind == PATCH_DISPOSED:
        disposition = str(row.get("disposition") or "")
        if disposition:
            custody.patch_disposed = disposition
            custody.patch_disposed_by = str(row.get("disposed_by_task_id") or "")
            custody.patch_result_lost = bool(row.get("result_lost"))
            custody.patch_apply_pending = False  # a recorded disposition completes the apply-intent story
    elif kind == SETTLED:
        # A RUN-level fact: it no longer clears the registration obligation -
        # PROJECT_RETIRED is the only discharging row (historical logs always
        # emitted it before SETTLED, so replay is unaffected).
        custody.ledger_recorded = custody.settled = True
        custody.terminal_state = str(row.get("state") or "") or custody.terminal_state
        custody.terminal_reason = str(row.get("outcome_reason") or "") or custody.terminal_reason
        custody.settled_at = str(row.get("ts") or "") or custody.settled_at
        if isinstance(row.get("engine_locator"), dict):
            custody.engine_locator = dict(row["engine_locator"])
        custody.engine_model = str(row.get("model") or "") or custody.engine_model
        custody.access_profile = str(row.get("access_profile") or "")[:32] or custody.access_profile
        custody.spend_disclosed = bool(row.get("spend_disclosed")) and row.get("cost_usd") is not None
        custody.spend_estimated = bool(row.get("spend_estimated"))
        custody.cost_partial = bool(row.get("cost_partial"))
        try:
            custody.cost_usd = float(row["cost_usd"]) if custody.spend_disclosed else None
        except (TypeError, ValueError):
            custody.cost_usd, custody.spend_disclosed = None, False
        if not row.get("preserve_containment"):
            custody.containment_fault = {}
    elif kind == TERMINAL_ACKNOWLEDGED:
        if str(row.get("task_id") or "") == custody.task_id:  # only the owner's own transcript
            custody.terminal_ack_attempt = str(row.get("attempt_key") or "")
    elif kind == CLOSED_ABSENT:
        # Closed, not settled: custody is over, the run leaves ``open_runs``.
        # The registration survives independently (wholesale clearing here was
        # the leak shape); no ledger row exists and none ever will.
        custody.settled = True
        custody.containment_fault = {}
    elif kind == CONTAINMENT_FAULT:
        custody.containment_fault = dict(row)
    elif kind == CONTAINMENT_RESOLVED:
        if not row.get("expected_reason") or row["expected_reason"] == custody.containment_fault.get("reason"):
            custody.containment_fault = {}


def _unique_rows(rows: Any) -> Iterator[Dict[str, Any]]:
    """The chain's rows with an outbox replay dropped: an event appended twice (a death
    between its COMMIT and its first append, then the flush) carries one ``event_id``."""
    seen: set = set()
    for row in rows:
        event_id = str(row.get("event_id") or "")
        if event_id:
            if event_id in seen:
                continue
            seen.add(event_id)
        yield row


def custody_rows(drive_root: Any) -> Tuple[Dict[str, Any], ...]:
    """The forensic replay: every custody row of the retained chain, read whole or raising (strict).
    Never on a bookkeeping path (those read the store's rows); audits, repair and tests only."""
    return tuple(_unique_rows(_iter_rows(event_log_path(drive_root), strict=True)))


def replay(drive_root: Any,
           rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, RunCustody]:
    """Every known run's custody: the store's rows, decoded (one SELECT).

    ``rows`` folds a pre-read chain snapshot instead (the importer, the forensic
    replays, the tests that feed rows directly); the returned objects are always
    this caller's own copies."""
    from ouroboros.delegate_custody_current import _decode, fold, runs

    if rows is not None:  # the importer's fold: the chain folds to exactly the rows it would write
        return {rid: _decode(facts) for rid, (facts, _hold) in fold(rows)["runs"].items()}
    return runs(drive_root)

def lookup(drive_root: Any, task_id: str, run_id: str) -> Tuple[str, Optional[RunCustody]]:
    """Answer OWNED / FOREIGN / UNKNOWN for ``run_id`` as seen by ``task_id``."""
    rid = str(run_id or "").strip()
    from ouroboros.delegate_custody_current import run

    # Fresh durable facts outrank the process memo (the body can collect elsewhere).
    # A proven absent row may still name our own unrecorded physical start, whose
    # in-process custody must retain the ability to cancel; read failure never does.
    custody = (run(drive_root, rid) if rid else None) or _CUSTODY.get(rid)
    if custody is not None:
        _CUSTODY[rid] = custody
    if custody is None:
        return UNKNOWN, None
    custody.run_id = custody.run_id or rid
    owner, mine = str(custody.task_id or ""), str(task_id or "")
    # A custody guard that skips itself when either identity is missing is not a guard.
    if not owner or not mine or owner != mine:
        return FOREIGN, custody
    return OWNED, custody


# The read-side evidence projection lives in `ouroboros/delegate_evidence.py`
# (extracted at this module's size ceiling); re-exported here because the
# completion seam, the tests and monkeypatch targets name it on THIS surface.
from ouroboros.delegate_evidence import task_execution_evidence  # noqa: F401,E402


def delegated_capture_dir(drive_root: Any, task_id: str, run_id: str) -> pathlib.Path:
    """The canonical artifact directory for ONE delegated run's captured patch.

    Named here (custody owns durable naming) so the terminal capture and the
    explicit apply/reject seam cannot disagree about where the patch lives. Under
    the task's artifact store on the CANONICAL drive, so the capture survives
    child-drive pruning exactly like the custody rows themselves.
    """
    from ouroboros.artifacts import DELEGATED_CAPTURE_PREFIX, task_artifact_dir_path

    safe_run = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(run_id or ""))[:64]
    return task_artifact_dir_path(drive_root, str(task_id or "")) / DELEGATED_CAPTURE_PREFIX / (safe_run or "run")


def capture_key(custody: RunCustody) -> str:
    """The capture directory name of ONE run: its snapshot (the provisioning
    invocation), or — for a run continuing in a predecessor's snapshot — its own
    ``capture_id``, so each capture keeps its own identity and none overwrites another."""
    return custody.capture_id or custody.snapshot_id or custody.run_id


def disposition_lock_path(drive_root: Any, custody: RunCustody) -> pathlib.Path:
    """The ONE disposition lock of an execution snapshot, shared by every run that
    continues in it and every task that may dispose it (and the hand-over claim)."""
    return delegated_capture_dir(drive_root, custody.snapshot_task_id or custody.task_id,
                                 custody.snapshot_id or custody.run_id) / "disposition.lock"


def record_patch_captured(drive_root: Any, custody: RunCustody, **payload: Any) -> bool:
    """Durably mark a terminal mutating run's patch as captured (idempotent flag).
    The in-memory flag follows the durable row: a copy is released only after it."""
    recorded = emit(drive_root, PATCH_CAPTURED, {
        "run_id": custody.run_id, "task_id": custody.task_id,
        "snapshot_id": custody.snapshot_id, **payload,
    })
    if recorded:
        custody.patch_captured = True
    return recorded


def record_patch_disposed(drive_root: Any, custody: RunCustody, *, disposition: str,
                          **payload: Any) -> bool:
    """Durably record the EXPLICIT apply/reject disposition of a captured patch.

    This is the row that releases the execution snapshot for cleanup: until it
    lands, the snapshot and the captured patch persist as conflict material.
    """
    custody.patch_disposed = str(disposition or "")
    custody.patch_disposed_by = str(payload.get("disposed_by_task_id") or "")
    custody.patch_result_lost = bool(payload.get("result_lost"))
    custody.patch_apply_pending = False  # a disposition completes the intent story
    return emit(drive_root, PATCH_DISPOSED, {
        "run_id": custody.run_id, "task_id": custody.task_id,
        "root_task_id": custody.root_task_id, "parent_task_id": custody.parent_task_id,
        "snapshot_id": custody.snapshot_id, "disposition": str(disposition or ""),
        **payload,
    })


def record_patch_apply_started(drive_root: Any, custody: RunCustody, **payload: Any) -> bool:
    """Durably record the APPLY INTENT before the target tree is mutated (CR1-3).

    Returns whether the row LANDED, and the caller must not mutate when it did
    not: an apply whose intent row never reached the disk leaves a crash window
    where a modified tree replays as "never applied" — the false-rejection class
    this row exists to close. Same doctrine as ``record_start_requested``.
    """
    landed = emit(drive_root, PATCH_APPLY_STARTED, {
        "run_id": custody.run_id, "task_id": custody.task_id,
        "snapshot_id": custody.snapshot_id, **payload,
    })
    if landed:
        custody.patch_apply_pending = True
        custody.patch_apply_key = str(payload.get("apply_idempotency_key") or "")
    return landed


def record_patch_apply_resolved(drive_root: Any, custody: RunCustody, *, reason: str,
                                **payload: Any) -> bool:
    """Resolve an apply intent whose attempt is PROVEN to have left the tree
    unmutated (drift refusal, atomic apply failure, clean revert).

    Memory clears regardless of the row's fate — the caller just observed the
    unmutated outcome — so an in-process retry stays open; a failed write only
    means a post-restart replay stays ambiguous, which is the fail-closed
    direction. A disposition row resolves the intent too (see ``_apply``).
    """
    custody.patch_apply_pending = False
    return emit(drive_root, PATCH_APPLY_RESOLVED, {
        "run_id": custody.run_id, "task_id": custody.task_id,
        "snapshot_id": custody.snapshot_id, "reason": str(reason or ""),
        **payload,
    })


def open_snapshot_ids(drive_root: Any) -> set:
    """Snapshot ids custody still holds OPEN — the custody GC's keep-set.

    A snapshot is open while its run is unsettled OR its captured patch has no
    explicit disposition, and while a PENDING invocation names it (the POST may
    have bound a live run the worker died before recording). At boot and each
    custody pass, GC may remove other ids only once their owning task is proven over.
    """
    from ouroboros.delegate_custody_current import held_runs

    ids: set = set()
    for custody in held_runs(drive_root).values():  # a settled, disposed run is closed: never here
        # A terminal plan has no patch duty. GC still requires its owning task
        # to be proven over; active starters can keep answering or implement it.
        if custody.snapshot_id and not (custody.settled and (custody.patch_disposed or custody.mode == "plan")):
            ids.add(custody.snapshot_id)
    ids.update(str(record["snapshot_id"]) for record in pending_invocations(drive_root) if record.get("snapshot_id"))
    return ids


def run_timing(drive_root: Any, run_id: str) -> Tuple[str, int]:
    """``(started_ts_iso, max_seconds)`` for a run, from its STARTED facts on the row.

    Empty/0 when the run is unknown or the row predates the ``max_seconds``
    field — absent facts stay absent, never invented.
    """
    from ouroboros.delegate_custody_current import run

    custody = run(drive_root, str(run_id or "").strip()) if str(run_id or "").strip() else None
    if custody is None:
        return "", 0
    return custody.started_ts, int(custody.max_seconds or 0)


def idempotency_key(*parts: Any) -> str:
    """A deterministic IDENTITY for one logical start — the lookup key, not the wire key.

    A random key per POST means an accepted start whose response was lost is retried as
    a SECOND live run nobody knows about. But a key derived from CONTENT alone has the
    opposite defect: a deliberate re-run of the same prompt presents the same key, and
    the engine hands back the finished old run instead of starting the intended new one.
    So this hash names the logical start for FINDING its pending invocation, and the
    invocation id (below) is what actually rides the wire as Idempotency-Key.
    """
    digest = hashlib.sha256("\0".join(str(part or "") for part in parts).encode("utf-8", errors="replace"))
    return digest.hexdigest()


def new_invocation_id() -> str:
    """A logical invocation id: minted ONCE per intended invocation, never per POST.

    The wire Idempotency-Key derives from it (verbatim — it is already opaque and
    unique). An ordinary ``delegate_start`` ALWAYS mints a fresh one — the owner's
    contract: an intended new start is a NEW id, even for an identical prompt. Reuse
    happens only when the caller EXPLICITLY names a pending invocation (the retry
    token from a start whose outcome was unknown), so Claudexor's replay check can
    return the run it already accepted; nothing is ever reused by content-matching,
    because two identical intentions are still two intentions.
    """
    return uuid.uuid4().hex


from ouroboros.delegate_invocation import (  # noqa: E402, F401 -- public custody readers
    invocation_record,
    _request_facts,
)


def record_start_requested(drive_root: Any, **payload: Any) -> bool:
    """Durably name the resources a start is about to bind, BEFORE the POST.

    Returns whether the row LANDED; without it the caller must not POST:
    a worker crash before ``record_started`` would leave the run unfindable.

    The full replay envelope goes to raw CAS before its event reference. Use
    ``write_blob``, never a redacted ``persist_call`` projection: request values
    must retain the engine's canonical JSON digest on an idempotent retry.
    """
    body = payload.get("request")
    if isinstance(body, dict) and body:
        from ouroboros.delegate_pending import request_shape
        from ouroboros.observability import write_blob

        try:
            ref = write_blob(pathlib.Path(drive_root), body, kind="json")
        except Exception:
            log.warning("delegate custody request body could not be stored", exc_info=True)
            return False
        payload = {key: value for key, value in payload.items() if key != "request"}
        payload.update(request_ref=ref, request_shape=request_shape(body), prompt_chars=len(str(body.get("prompt") or "")))
    from ouroboros.owner_pause import admit_delegated_start

    return admit_delegated_start(drive_root, payload)


def record_started(drive_root: Any, custody: RunCustody,
                   shape: Optional[Dict[str, Any]] = None) -> bool:
    """Memoize the run and write its authoritative row. Returns whether the row LANDED.

    A start whose row did not land is custodied by this process only: after this
    worker dies nothing can name, wait on, cancel or settle the live run.
    ``shape`` (access/mode/isolation/delegated/root) rides the SAME row — a shape
    recorded separately can lose one half of itself to a crash between writes.
    The memo update goes through the SAME first-wins merge the replay uses (gate
    fix 8): a duplicate start must not diverge the in-process view from replay.
    """
    # Fold the row-only shape onto the object first (the row spreads it last),
    # so the memo and a replay of this same row start from identical facts.
    for attr in ("access", "mode", "isolation"):
        if shape and attr in shape:
            setattr(custody, attr, str(shape.get(attr) or ""))
    for attr in _STARTED_OPTION_FIELDS:
        if shape and isinstance(shape.get(attr), str):
            setattr(custody, attr, shape[attr])
    if shape and "delegated" in shape:
        custody.delegated = shape.get("delegated") is True
    previous = _CUSTODY.get(custody.run_id)
    if previous is not None and previous is not custody:
        _merge_started_into(custody, previous)
    _CUSTODY[custody.run_id] = custody
    # The C1 binding and the resource reference ride the SAME row (a binding
    # recorded separately can lose half of itself to a crash); shape spreads LAST.
    landed = emit(drive_root, STARTED, {
        "run_id": custody.run_id,
        "daemon_store_id": custody.daemon_store_id,
        "project_owned": custody.project_owned, "project_persistent": custody.project_persistent,
        "resource_ref": custody.resource_ref or {},
        **{key: copy.deepcopy(getattr(custody, key)) for key in _STARTED_MAPPING_FIELDS},
        "work_order_source_request": custody.work_order_source_request or {},
        **({"prompt_cache_ttl": copy.deepcopy(custody.prompt_cache_ttl)}
           if isinstance(custody.prompt_cache_ttl, dict) else {}),
        **{key: getattr(custody, attr) for attr, key in _STARTED_STR_FIELDS},
        **(shape or {}),
        **{key: getattr(custody, key) for key in _STARTED_OPTION_FIELDS if getattr(custody, key) is not None},
    })
    if landed and custody.capture_id and custody.continuation_of:
        # The same hand-over the replay derives, and the disposition writer's
        # follow-up: the predecessor's owner stops disclosing its patch as debt.
        _supersede(_CUSTODY, custody)
        try:
            from ouroboros.delegate_terminal import refresh_disposed_reconciliation

            refresh_disposed_reconciliation(drive_root, custody.continuation_of, reader_task_id=custody.task_id)
        except Exception:
            log.warning("Superseded custody disclosure refresh failed for run %s", custody.continuation_of,
                        exc_info=True)
    return landed


def record_output_consumed(drive_root: Any, custody: RunCustody, *,
                           artifact: str, byte_length: int, sha256: str,
                           chars: int, lines: int, reader_task_id: str = "") -> bool:
    from ouroboros.delegate_output import record_output_consumed as _record

    return _record(
        drive_root, custody, artifact=artifact, byte_length=byte_length,
        sha256=sha256, chars=chars, lines=lines, reader_task_id=reader_task_id,
    )


def output_disposition(custody: RunCustody) -> Dict[str, Any]:
    from ouroboros.delegate_output import output_disposition as _disposition

    return _disposition(custody)


def is_terminal(detail: Dict[str, Any]) -> bool:
    from ouroboros.delegate_custody_usage import is_terminal as _is_terminal

    return _is_terminal(detail, TERMINAL_STATES)


def retire_project(drive_root: Any, gateway: Any, custody: RunCustody, *, live_task_ids=None,
                   recoverable_handles=()) -> None:
    """Serialize the replay-to-retirement decision for one shared project."""
    with project_retirement_lock(drive_root, custody.project_id):
        _retire_project_locked(drive_root, gateway, custody, live_task_ids=live_task_ids,
                               recoverable_handles=recoverable_handles)


def _project_runs(drive_root: Any, custody: RunCustody) -> Optional[List[RunCustody]]:
    """This project's runs (one indexed read); ``None`` when the store cannot answer."""
    from ouroboros.delegate_custody_current import runs as current_runs
    from ouroboros.store import StoreError

    try:
        state = current_runs(drive_root, "project_id=?", (str(custody.project_id or ""),))
    except StoreError:
        log.warning("Retirement deferred: custody rows unavailable", exc_info=True)
        return None
    if custody.run_id and custody.run_id not in state:
        log.warning("Retirement deferred: run %s not in replay", custody.run_id)
        return None
    return [run for run in state.values() if run.run_id]


def _release_registration(drive_root: Any, custody: RunCustody, **facts: Any) -> None:
    """Our custody over the registration ends: the memo (every sibling, exactly as
    the replay clears them) and the durable row, ``facts`` naming why it was kept."""
    custody.project_owned = False
    for sibling in _CUSTODY.values():
        if sibling.project_id == custody.project_id:
            sibling.project_owned = False
    emit(drive_root, PROJECT_RETIRED, {"run_id": custody.run_id, "task_id": custody.task_id,
                                       "project_id": custody.project_id, **facts})


def _retire_project_locked(drive_root: Any, gateway: Any, custody: RunCustody, *, live_task_ids=None,
                           recoverable_handles=()) -> None:
    if custody.project_persistent:
        custody.project_owned = False
        emit(drive_root, PROJECT_RETIRED, {"run_id": custody.run_id, "task_id": custody.task_id,
                                           "project_id": custody.project_id, "project_kept": True})
        return
    if not custody.project_id:
        return
    try:
        rows = _project_runs(drive_root, custody)
        if rows is None or not any(run.project_owned for run in rows):
            return
        if any(run.project_persistent for run in rows):
            # #362: ANY persistent sharer makes the project a durable user
            # identity — a non-persistent creator must not delete it either.
            custody.project_owned = False
            emit(drive_root, PROJECT_RETIRED, {"run_id": custody.run_id, "task_id": custody.task_id,
                                               "project_id": custody.project_id, "project_kept": True})
            return
        if any(not run.settled and run.run_id != custody.run_id for run in rows):
            return
        from ouroboros.delegate_continuation import still_continuable
        if any(still_continuable(drive_root, run, live_task_ids, recoverable_handles=recoverable_handles)
               for run in rows):
            return
    except Exception:
        log.warning("Retirement deferred: replay failed for %s",
                    custody.run_id, exc_info=True)
        return
    try:
        gateway.remove_project(custody.project_id)
    except Exception as exc:
        if not daemon_says_absent(exc):
            log.warning("Failed to retire delegated project %s", custody.project_id, exc_info=True)
            # The daemon's typed refusal rides the row beside its text: "failed"
            # without the WHY made every retire loop a forensic dig, and the CODE
            # is what tells a permanent refusal from one worth retrying.
            refusal = {"code": str(getattr(exc, "code", "") or ""),
                       "status": int(getattr(exc, "status_code", 0) or 0)}
            if daemon_keeps_project(exc):
                # Discharge only when EVERY run of the project is settled per a
                # complete view read NOW, under the retirement lock and right
                # before the discharge row is appended: STARTED appends take no
                # lock, so a PROJECT_RETIRED replayed after a sibling's STARTED
                # would strip that sibling's ownership. Nothing proven means no.
                try:
                    settled_rows = _project_runs(drive_root, custody)
                except Exception:
                    log.warning("Discharge deferred: replay failed for %s", custody.run_id, exc_info=True)
                    settled_rows = None
                if settled_rows and all(run.settled for run in settled_rows):
                    # The #362 vocabulary — our custody ends, the engine keeps the
                    # project — under the engine's own code; never a deletion claim.
                    _release_registration(drive_root, custody, project_kept=True,
                                          reason=PROJECT_HAS_THREADS, **refusal)
                    return
            emit(drive_root, PROJECT_RETIRE_FAILED, {"run_id": custody.run_id, "task_id": custody.task_id,
                                                     "project_id": custody.project_id,
                                                     "reason": str(exc)[:500], **refusal})
            return
    _release_registration(drive_root, custody)


def close_absent_run(drive_root: Any, gateway: Any, custody: RunCustody, reason: str) -> None:
    """Close one owning-store absence without fabricating terminal output or money."""
    from ouroboros.delegate_custody_current import run
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    with actor_decision_lock(drive_root, custody.task_id):
        current = run(drive_root, custody.run_id) or custody
        if current.settled:
            custody.settled = True
            return
        if not _absence_provable(current, gateway):
            raise ClaudexorUnavailable("daemon_store_identity_unproven", "Run absence is unproven in its owning store")
        custody.settled = emit(drive_root, CLOSED_ABSENT, {
            "run_id": custody.run_id, "task_id": custody.task_id,
            "route": custody.route_id, "project_id": custody.project_id, "reason": reason,
        })
    # Network retirement is independent of the closed fact, outside the actor lock.
    retire_project(drive_root, gateway, custody)
    if custody.settled:
        # The compact fault projection is cleared by its OWN writer pair, so a run
        # closed absent through reconciliation (which never passes _cancel_result)
        # does not stay a CRITICAL invariant forever.
        resolve_containment_fault(drive_root, custody, "closed_absent")


def settle_run(drive_root: Any, gateway: Any, custody: RunCustody, detail: Dict[str, Any]) -> Dict[str, Any]:
    """Compatibility settlement for the panel/control owners; collection separates cleanup."""
    from ouroboros.delegate_collection import observe_run, record_terminal_money

    # Cancellation already paid for this exact read-back. Preserve it before
    # money/retirement, including a repeated settlement: the body can collect
    # it after the engine disappears. Never fetch output or wait for capture on
    # the stop path; an unavailable source is a gap, not a reason to delay Stop.
    try:
        observe_run(drive_root, custody, detail)
    except Exception:
        log.warning("Terminal observation was not retained for %s", custody.run_id, exc_info=True)

    if custody.settled:
        return {"settled": True, "ledger_recorded": custody.ledger_recorded,
                "project_retired": not custody.project_owned and not custody.project_persistent,
                "project_persistent": custody.project_persistent, "retried": False}
    result = record_terminal_money(drive_root, custody, detail)
    if custody.review_owned or custody.project_persistent:
        retire_project(drive_root, gateway, custody)
    if custody.settled:
        resolve_containment_fault(drive_root, custody, "settled_terminal")
    result.update(project_retired=not custody.project_owned and not custody.project_persistent,
                  project_persistent=custody.project_persistent)
    return result


# -- cancellation --------------------------------------------------------------


def open_containment_faults(drive_root: Any) -> List[Dict[str, Any]]:
    """Containment faults with no later resolution: runs that MAY still be live.

    Read on every context build, from the held runs' rows: the reducer keeps the
    incident row on the run until its resolution, settlement or closure clears it, so
    an open incident can never age out (it used to fall out of a tail-bounded chain
    scan, which is why a separate projection file once existed). Unavailable store:
    no answer, never an empty one — the caller treats the exception as unknown.
    """
    from ouroboros.delegate_custody_current import held_runs

    return [dict(custody.containment_fault) for custody in held_runs(drive_root).values()
            if custody.containment_fault]


def output_unread(custody: RunCustody) -> bool:
    """Staged in full and never read to EOF, whatever the money's state: continuation
    advice keys on the observed terminal, not on settlement (#1969)."""
    return bool(custody.output_artifact and custody.output_complete and not custody.output_consumed)


def settled_output_unread(custody: RunCustody) -> bool:
    """One predicate for "settled, staged in full, and never read to EOF".

    Shared by the settlement row, the parent-facing payload and the health invariant so
    the three cannot drift into disagreeing about the same run.
    """
    return bool(custody.settled and output_unread(custody))


def record_settled_unread(drive_root: Any, custody: RunCustody) -> bool:
    """Name a settled-but-never-read result durably, once per RUN. Returns the fact.

    Once per run and not once per poll: a re-wait on an already settled run must not
    append a second identical row, which would read as a second omission. The
    ``unread_disclosed`` flag replays like every other custody fact, so a restarted
    worker does not repeat it either.
    """
    if not settled_output_unread(custody):
        return False
    if not custody.unread_disclosed:
        custody.unread_disclosed = emit(drive_root, SETTLED_UNREAD, {
            "run_id": custody.run_id, "task_id": custody.task_id,
            "root_task_id": custody.root_task_id, "parent_task_id": custody.parent_task_id,
            "route": custody.route_id, "artifact": custody.output_artifact,
            **({"source": dict(custody.output_source)} if custody.output_source else {}),
        })
    return True


def settled_unread_outputs(drive_root: Any, state: Optional[Dict[str, RunCustody]] = None) -> List[RunCustody]:
    """Settled runs whose verified FULL output was never read to EOF.

    The counterpart of ``open_containment_faults`` for the D7 class, and self-clearing
    for the same reason: the acknowledgement row flips ``output_consumed`` in the very
    replay this reads, so the entry disappears the moment the nanny performs the read.
    Only VERIFIED FULL staged content qualifies — a preview was never acknowledgeable,
    and a run that staged nothing (inline result, cancelled, failed with no output) owes
    nothing and must never appear here.
    """
    return [custody for custody in (state if state is not None else replay(drive_root)).values()
            if settled_output_unread(custody)]


def undisposed_patches(drive_root: Any, state: Optional[Dict[str, RunCustody]] = None) -> List[RunCustody]:
    """Settled snapshot or directory-copy work awaiting explicit apply/reject.

    The C1 counterpart of ``settled_unread_outputs``: a run that executed in a
    private snapshot and settled — through the nanny OR through reconciliation —
    holds real work that reaches the shared tree only via
    ``integrate_delegated_patch``. Until that disposition lands, the snapshot and
    the captured patch persist (the GC keeps them), and this projection keeps the
    obligation VISIBLE instead of letting an orphaned run's work sit on disk
    forever, preserved but findable by nobody. Self-clearing: the
    ``PATCH_DISPOSED`` row flips ``patch_disposed`` in the very replay this reads.
    """
    return [custody for custody in (state if state is not None else replay(drive_root)).values()
            if (custody.snapshot_id or (custody.resource_ref.get("workspace_kind") == "directory"
                                       and custody.resource_ref.get("strategy") == "copy"))
            and custody.settled and not custody.patch_disposed and candidate_apply_possible(custody)]


def record_containment_fault(drive_root: Any, custody: RunCustody, reason: str,
                             detail: str = "", **facts: Any) -> None:
    """A run we tried to stop and could not verify stopped is a LOUD, durable incident.

    It surfaces as a CRITICAL health invariant until a verified terminal receipt (or a
    settlement) resolves it — never as a reassuring string in a tool result.

    ``facts`` carries the typed evidence for the specific guarantee that was not
    delivered (which access profile was enforced, which harness HOME was applied). ONE
    author for this row: an unverified cancel and a breached containment produce the
    same incident shape, so a reader never has to know which path wrote it.

    The reducer keeps the incident on the run's row (the surface the health invariant
    reads) until a resolution clears it; the chain row is the forensic record.
    ``detail`` is the incident's EVIDENCE, and a durable incident whose evidence was
    silently cut is half an incident: the bound goes through the shared
    disclosed-truncation contract (marker + original length), never a bare slice
    (P34R.5, DEVELOPMENT.md "No silent truncation").
    """
    from ouroboros.utils import truncate_review_artifact

    emit(drive_root, CONTAINMENT_FAULT, {
        "run_id": custody.run_id, "task_id": custody.task_id, "route": custody.route_id,
        "project_id": custody.project_id, "reason": reason,
        "detail": truncate_review_artifact(str(detail or ""), 2000),
        **facts,
    })


def resolve_containment_fault(drive_root: Any, custody: RunCustody, reason: str, *, expected_reason: str = "") -> None:
    emit(drive_root, CONTAINMENT_RESOLVED, {"run_id": custody.run_id, "task_id": custody.task_id, "reason": reason,
                                         **({"expected_reason": expected_reason} if expected_reason else {})})


def cancel_and_verify(drive_root: Any, gateway: Any, custody: RunCustody, reason: str) -> Dict[str, Any]:
    """Cancel a run and report ONLY what a terminal receipt proves.

    ``confirmed`` needs the run to read back terminal — or a durable settlement, or the
    daemon answering that it has no such run. ``requested`` is an accepted control the run
    has not obeyed yet. ``failed`` is a reachable daemon refusing while the run keeps
    mutating, and an unverifiable attempt is a containment fault. The old path answered
    all four with ``status: cancelled``.
    """
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    # The same durable fact ``settle_run`` short-circuits on, consulted by its twin. A run
    # this module already recorded as terminal is not an overpowered live process, and a
    # later ordinary cancel of it must not manufacture a permanent CRITICAL against a run
    # the settlement had closed.
    if custody.settled:
        return _cancel_result(drive_root, custody, CANCEL_CONFIRMED, accepted=False,
                              control_status="", state="settled")
    accepted, control_status, control_error = False, "", ""
    try:
        receipt = gateway.cancel_run(custody.run_id, reason=str(reason or ""))
        accepted = bool((receipt or {}).get("accepted"))
        control_status = str((receipt or {}).get("status") or "")
    except ClaudexorUnavailable as exc:
        if daemon_says_absent(exc, gateway=gateway, run=custody):
            close_absent_run(drive_root, gateway, custody, "cancel_run_absent")
            return _cancel_result(drive_root, custody, CANCEL_CONFIRMED, accepted=False,
                                  control_status="", state="absent")
        # A refused control is not a verdict on the RUN. The read below is what decides
        # whether anything is still mutating; declaring the fault here — with the read
        # sitting three lines away, unused — faulted runs that had already stopped.
        control_error = f"{exc.code}: {exc}"
    try:
        detail = gateway.get_run(custody.run_id)
    except ClaudexorUnavailable as exc:
        if daemon_says_absent(exc, gateway=gateway, run=custody):
            close_absent_run(drive_root, gateway, custody, "get_run_absent")
            return _cancel_result(drive_root, custody, CANCEL_CONFIRMED, accepted=accepted,
                                  control_status=control_status, state="absent")
        return _cancel_result(drive_root, custody, CANCEL_CONTAINMENT_FAULT, accepted=accepted,
                              control_status=control_status, state="",
                              fault_reason="cancel_unreachable" if control_error else "cancel_unverified",
                              detail=control_error or f"{exc.code}: {exc}")
    state = str(summary_of(detail).get("state") or "")
    if state in TERMINAL_STATES:
        settle_run(drive_root, gateway, custody, detail)
        # The verify read's own detail rides the result (BR2-1, purely additive):
        # a caller consuming a discovered natural terminal (completion wins) must
        # not depend on a SECOND fetch succeeding after the run is settled.
        return _cancel_result(drive_root, custody, CANCEL_CONFIRMED, accepted=accepted,
                              control_status=control_status, state=state,
                              terminal_detail=detail)
    if control_error:
        return _cancel_result(drive_root, custody, CANCEL_CONTAINMENT_FAULT, accepted=False,
                              control_status=control_status, state=state,
                              fault_reason="cancel_unreachable", detail=control_error)
    if accepted:
        return _cancel_result(drive_root, custody, CANCEL_REQUESTED, accepted=True,
                              control_status=control_status, state=state)
    return _cancel_result(drive_root, custody, CANCEL_FAILED, accepted=False,
                          control_status=control_status, state=state,
                          fault_reason="cancel_rejected_run_still_live",
                          detail=f"daemon refused the cancel; run state is {state or 'unknown'}")


def _cancel_result(drive_root: Any, custody: RunCustody, outcome: str, *, accepted: bool,
                   control_status: str, state: str, fault_reason: str = "",
                   detail: str = "",
                   terminal_detail: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    emit(drive_root, CANCEL_OUTCOME, {
        "run_id": custody.run_id, "task_id": custody.task_id, "outcome": outcome,
        "accepted": accepted, "control_status": control_status, "state": state,
    })
    if fault_reason:
        record_containment_fault(drive_root, custody, fault_reason, detail)
    elif outcome == CANCEL_CONFIRMED:
        resolve_containment_fault(drive_root, custody, "verified_terminal")
    result = {"outcome": outcome, "accepted": accepted, "control_status": control_status,
              "state": state, "fault_reason": fault_reason, "detail": detail}
    # BR2-1, backward-compatible: the key exists only when a verify read produced
    # the run detail — absent otherwise, so every pre-existing consumer of the
    # six-key shape is untouched (the emit above deliberately excludes it too).
    if terminal_detail is not None:
        result["terminal_detail"] = terminal_detail
    return result

# -- reconciliation ------------------------------------------------------------


def owned_project_registrations(drive_root: Any, state: Optional[Dict[str, RunCustody]] = None) -> List[RunCustody]:
    """Runs whose registration is still owned - settled or not (``open_runs``
    cannot see registrations that outlive their runs)."""
    if state is None:
        from ouroboros.delegate_custody_current import held_runs

        state = held_runs(drive_root)  # an owned registration is a held row
    return [custody for custody in state.values() if custody.project_owned and custody.project_id]


def retire_settled_registrations(drive_root: Any, gateway: Any, *, live_task_ids=None,
                                 recoverable_handles=()) -> None:
    """Retire projects every sharer has settled; a LIVE sharer (owned or not
    - only the creator carries the registration, but any live sibling makes
    the daemon refuse) defers the attempt. Idempotent, fail-soft."""
    from ouroboros.delegate_custody_current import held_runs, runs as current_runs

    by_project: Dict[str, List[RunCustody]] = {}
    for row in held_runs(drive_root).values():  # owned registrations and unsettled runs are held rows
        if row.project_id and row.run_id:
            by_project.setdefault(row.project_id, []).append(row)
    for project_id, rows in by_project.items():
        owned = [row for row in rows if row.project_owned]
        if not owned or any(not row.settled for row in rows):
            continue  # nothing registered here, or a live sharer defers
        from ouroboros.delegate_continuation import still_continuable
        # Every sharer of the project, settled non-owners included: one indexed read.
        rows = list(current_runs(drive_root, "project_id=?", (str(project_id or ""),)).values())
        if any(not row.settled for row in rows):
            continue
        if (not any(row.project_persistent for row in rows)
                and any(still_continuable(drive_root, row, live_task_ids, recoverable_handles=recoverable_handles)
                        for row in rows)):
            continue  # still continuable: the registration waits until it is not
        try:
            retire_project(drive_root, gateway, min(owned, key=lambda row: row.run_id), live_task_ids=live_task_ids,
                           recoverable_handles=recoverable_handles)
        except Exception:
            log.warning("Registration sweep failed for project %s", project_id, exc_info=True)


__all__ = [
    "CANCEL_CONFIRMED",
    "CANCEL_CONTAINMENT_FAULT",
    "CANCEL_FAILED",
    "CANCEL_REQUESTED",
    "FOREIGN",
    "OWNED",
    "RunCustody",
    "SOURCE_RANGE_DELIVERY_CONFIRMED",
    "SUPERSEDED",
    "TERMINAL_STATES",
    "UNKNOWN",
    "actor_decision_lock",
    "cancel_and_verify",
    "capture_key",
    "close_absent_run",
    "custody_log_unreadable",
    "custody_root",
    "daemon_says_absent",
    "delegated_capture_dir",
    "disclosed_spend",
    "disposition_lock_path",
    "emit",
    "held_run_names",
    "idempotency_key",
    "is_terminal",
    "invocation_record",
    "lookup",
    "names_another_run",
    "new_invocation_id",
    "open_containment_faults",
    "open_runs",
    "open_snapshot_ids",
    "output_disposition",
    "pending_invocations",
    "reconcile_orphaned_runs",
    "record_containment_fault",
    "record_output_consumed",
    "record_patch_apply_resolved",
    "record_patch_apply_started",
    "record_patch_captured",
    "record_patch_disposed",
    "record_start_requested",
    "record_settled_unread",
    "record_started",
    "release_task_runs",
    "reconcile_task_runs",
    "retire_project",
    "review_owned_source",
    "run_timing",
    "settle_run",
    "settled_output_unread",
    "settled_unread_outputs",
    "summary_of",
    "task_execution_evidence",
    "undisposed_patches",
    "work_order_source_verification",
    "record_source_range_verified",
]


# v7next F2 (D07): moved spans live in their owner leaves; re-exported here
# so this facade stays the single import surface for callers and tests.
from ouroboros.delegate_custody_reconcile import (  # noqa: E402, F401 -- intentional public re-exports
    _reconcile_each,
    _reconcile_one,
    _recover_pending_invocation,
    _retire_recovered_registration,
    open_runs,
    pending_invocations,
    reconcile_orphaned_runs,
    reconcile_task_runs,
    release_task_runs,
)
