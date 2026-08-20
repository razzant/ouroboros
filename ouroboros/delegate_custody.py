"""Durable custody for delegated (Claudexor) runs.

A delegated run is an OVERPOWERED, MUTATING process that Ouroboros does not own the
process tree of: it lives inside the Claudexor daemon and survives our worker, so
custody cannot live in a module dict — a crash, restart, or lost POST response would
leave a LIVE run nothing can wait on, cancel or settle. ``maxSeconds`` is damage
limitation, not custody.

**The authority is the durable record that is already written.** The task's own event
log (``logs/events.jsonl`` under the canonical data root) is the SSOT; the
process-local dict below is a pure memoization of these rows, and every question is
answered by replaying them. Rows survive a restart, so custody does too.

Three lookup answers, never two: OWNED, FOREIGN (another task started it) and UNKNOWN
(no durable record at all). Collapsing UNKNOWN into "not yours" is what made a restarted
owner indistinguishable from an intruder.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ouroboros.utils import append_jsonl, utc_now_iso

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
LEDGER_RECORDED = "delegate_run_ledger_recorded"
SETTLED = "delegate_run_settled"
SETTLE_FAILED = "delegate_run_settle_failed"
# The daemon answered that it has no such run. Custody closes WITHOUT a settlement: there
# is no terminal detail to settle against, and inventing a $0 ledger row for a run nobody
# can describe is the one shape the ledger must never produce.
CLOSED_ABSENT = "delegate_run_closed_absent"
PROJECT_RETIRED = "delegate_run_project_retired"
PROJECT_RETIRE_FAILED = "delegate_run_project_retire_failed"
CANCEL_OUTCOME = "delegate_run_cancel_outcome"
UNCONFINED = "delegate_run_unconfined"
CONTAINMENT_FAULT = "delegate_run_containment_fault"
CONTAINMENT_RESOLVED = "delegate_run_containment_resolved"
RECONCILED = "delegate_run_reconciled"
OUTPUT_SPILLED = "delegate_run_output_spilled"
# Owner doctrine D7: a delegated result is OBTAINED only after the artifact is read to
# EOF. The canonical acknowledgement — written when the read windows covered the staged
# artifact CONTINUOUSLY start-to-EOF, hash-bound, and only writable when the staging was
# the verified FULL content (`full_content`), never the bounded 256 KiB preview.
# Owner directive (2026-08-03): consumption is LOAD-BEARING before settlement; its
# ABSENCE is the durable typed fact ``SETTLED_UNREAD`` below. It does not BLOCK
# settlement — see ``settle_run`` for why a hard gate would deadlock the flow.
OUTPUT_CONSUMED = "delegate_run_output_consumed"
# A run whose VERIFIED FULL output was staged, paid for, and settled without ever being
# read to EOF: the "launched and never collected" class, named at the moment it becomes
# permanent instead of inferred later from a field nobody joined.
SETTLED_UNREAD = "delegate_run_settled_unread"
# C1 (delegated-run isolation): a MUTATING run executes in a private snapshot; its
# diff is captured durably at terminal, then EXPLICITLY applied or rejected by the
# nanny. Capture is idempotent (the flag replays); the disposition row is what
# releases the snapshot for GC — until then conflict material persists on disk.
PATCH_CAPTURED = "delegate_run_patch_captured"
PATCH_DISPOSED = "delegate_run_patch_disposed"
# CR1-3 (owed-before-sent, one row earlier than the disposition): APPLY-INTENT lands
# BEFORE the target tree is mutated; RESOLVED lands only when the attempt is KNOWN to
# have left the tree unmutated. An intent with neither resolution nor disposition
# replays as AMBIGUOUS — the tree may carry the patch — and a later reject/apply must
# refuse typed instead of pretending "not applied" over a modified tree.
PATCH_APPLY_STARTED = "delegate_run_patch_apply_started"
PATCH_APPLY_RESOLVED = "delegate_run_patch_apply_resolved"

# Cheap prefilter: every custody row's type starts with this, so a multi-hundred-MB
# event log is scanned without JSON-parsing the 99.9% of lines that are not ours.
_ROW_MARKER = "delegate_run"
# The containment-fault projection is read on every context build, so it reads a bounded
# tail; ownership replay (rare, and correctness-critical) reads the whole log.
_FAULT_SCAN_TAIL_BYTES = 4_000_000

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
    # Requested pin (`credentialProfileId`) from the start body; '' = automatic; applied half = settlement authRoute.
    profile_id: str = ""
    project_id: str = ""
    project_owned: bool = False
    root_task_id: str = ""
    parent_task_id: str = ""
    ledger_root: str = ""
    idempotency_key: str = ""
    # The LOGICAL INVOCATION ID: minted once per intended invocation, reused verbatim as the wire Idempotency-Key
    # on a transport retry of the SAME invocation (so the engine returns the run it already accepted), and fresh
    # for every deliberately new ``delegate_start``. ``idempotency_key`` above is the content-derived identity
    # used to FIND a pending invocation; this id is what actually rides the wire.
    invocation_id: str = ""
    ledger_recorded: bool = False
    settled: bool = False
    # The containment disclosure is a FACT about the run, so it is written once. A nanny
    # may call `delegate_wait` again on an already-terminal run, and a durable log that
    # repeats the same finding per poll reads like repeated findings. Replayed from the
    # durable row like every other field here, so a restarted worker does not re-disclose.
    containment_disclosed: bool = False
    # Whether the settled-but-never-read omission has already been named durably.
    unread_disclosed: bool = False
    # The staged-output half of the terminal story (D7). ``output_artifact`` is the task-drive-relative path the
    # terminal payload was staged to (empty when it fit inline); ``output_complete`` is whether what was staged is
    # the full content as far as the run's OWN report can establish it — its served size, or the preview carried
    # as a prefix (`_resolve_full_primary_output`); the engine publishes no content hash for the primary output,
    # so equal length binds the body to the run's CLAIM about it, not to a digest of it. An artifact that matches
    # neither stages as incomplete and can never be acknowledged. ``output_sha`` is the content hash of what is
    # CURRENTLY staged; ``output_consumed`` is whether the canonical acknowledgement exists FOR THAT HASH — a
    # re-stage of different content at the same path resets it, because an ack names bytes, not a path. All
    # replayed, so a restarted worker keeps the facts.
    output_artifact: str = ""
    output_complete: bool = False
    output_sha: str = ""
    output_consumed: bool = False
    # C1 isolation binding for a MUTATING run: the private execution root the run was
    # scoped to, the baseline commit its diff is measured from, and the AUTHORITY
    # target tree its patch is destined for. ``snapshot_id`` keys the worktree-service
    # registry entry (it equals the invocation id that provisioned it). All durable and
    # replayed: a retry reproduces the exact binding, the terminal capture knows where
    # to diff, and the startup GC can tell a live snapshot from a disposable one.
    snapshot_id: str = ""
    execution_root: str = ""
    baseline_sha: str = ""
    target_root: str = ""
    authority_source: str = ""
    # The GRANTED run shape, replayed off the STARTED row that already carries it
    # (the `shape` dict): delegate_wait replays this as the entitled authority —
    # re-deriving from live context read `readonly` for a top-level payload run
    # and cancelled it as widened (R1-2).
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
    # rejects ("rejected") the captured patch — only then may the snapshot be removed.
    patch_captured: bool = False
    patch_disposed: str = ""
    # CR1-3: an apply-intent row exists with no resolution and no disposition —
    # the target tree MAY carry the patch (crash between apply and the disposition
    # row), so any later disposition over this run is ambiguous until inspected.
    patch_apply_pending: bool = False


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


def emit(drive_root: Any, kind: str, payload: Dict[str, Any]) -> bool:
    """Append one custody row and REPORT whether it landed. Never raises.

    ``append_jsonl`` already owns the success predicate for this class of write.
    Discarding it is how a run could be reported as started, waited on and settled
    while the rows that ARE its custody never reached disk; the row's fate is the
    answer.
    """
    try:
        written = bool(append_jsonl(event_log_path(drive_root), {"ts": utc_now_iso(), "type": kind, **payload}))
    except Exception:
        log.warning("delegate custody row could not be written (%s)", kind, exc_info=True)
        return False
    if not written:
        log.warning("delegate custody row was rejected by the event log (%s)", kind)
    return written


def daemon_says_absent(exc: Any) -> bool:
    """True when the daemon ANSWERED that the named resource does not exist.

    A 404 from a reachable daemon is a definitive fact, not a failure to find out: a run
    the daemon does not have is not mutating anything, and a registration it does not have
    is already retired. Collapsing it into "unreachable" is what left a finished run in
    ``open_runs`` forever, re-faulted on every pass, with a CRITICAL health invariant no
    settlement or cancel could ever clear.

    Deliberately NOT "any 4xx": a 400 or a 403 is the daemon refusing US, which says
    nothing about whether the resource is there.
    """
    return int(getattr(exc, "status_code", 0) or 0) == 404


def custody_log_unreadable(drive_root: Any) -> bool:
    """Whether the custody event log EXISTS but cannot be opened (GR6-4).

    ``_iter_rows`` swallows its own ``OSError`` — the right behavior for the
    fail-soft readers — but the KILL/MISS/REAP AUDIT must not let an
    unreadable log audit as "cleanly reconciled": ABSENT is a positively
    established empty state (no custody row could exist), while
    existing-but-unreadable means the open-run answer is UNKNOWN. Same probe
    the evidence reader (``task_execution_evidence``) already uses; its own
    semantics are unchanged.
    """
    path = event_log_path(drive_root)
    try:
        if not path.exists():
            return False
        with path.open("rb"):
            pass
    except OSError:
        return True
    return False


def _iter_rows(path: pathlib.Path, tail_bytes: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    try:
        with path.open("rb") as handle:
            size = path.stat().st_size
            if tail_bytes is not None and size > tail_bytes:
                handle.seek(size - tail_bytes)
                handle.readline()          # drop the partial first line
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
        return


# The STARTED row's string facts as ``(RunCustody attribute, row key)`` pairs —
# one table shared by the replay and the ``record_started`` emit.
_STARTED_STR_FIELDS: Tuple[Tuple[str, str], ...] = tuple(
    (attr, "route" if attr == "route_id" else attr) for attr in (
        "task_id", "route_id", "model", "profile_id", "project_id", "root_task_id",
        "parent_task_id", "ledger_root", "idempotency_key", "invocation_id",
        "snapshot_id", "execution_root", "baseline_sha", "target_root",
        "authority_source", "access", "mode", "isolation",
    )
)
# Progress carried forward from a previous row: an idempotent re-start writes a
# SECOND started row; replacing wholesale would forget a settlement and put a
# finished run back into the orphan sweep (which would cancel it).
_STARTED_PROGRESS_FLAGS: Tuple[str, ...] = (
    "ledger_recorded", "settled", "containment_disclosed", "unread_disclosed",
    "output_artifact", "output_complete", "output_sha", "output_consumed",
    "patch_captured", "patch_disposed", "patch_apply_pending")
# Binding/authority facts are FIRST-WINS (R1-2): a later idempotent STARTED row
# may be minted by a context that no longer knows the original binding; the
# first recorded fact is authoritative and is never erased or retargeted.
_STARTED_FIRST_WINS_FACTS: Tuple[str, ...] = (
    "snapshot_id", "execution_root", "baseline_sha", "target_root",
    "authority_source", "resource_ref")


def _merge_started_into(entry: RunCustody, previous: RunCustody) -> None:
    """Project a duplicate STARTED fact set onto an existing run — the ONE
    merge, used by both the durable replay and the in-process memo (gate fix 8).

    Progress always carries forward; binding facts are truthy-first-wins per
    field; the SHAPE GROUP (access/mode/isolation/delegated) plus the resource
    reference is first-wins as a UNIT, keyed on the first row that CARRIED a
    shape — so a recorded ``delegated=False`` survives a later ``True`` and an
    empty ``resource_ref`` is never "filled" by a later row.
    """
    for attr in _STARTED_PROGRESS_FLAGS:
        setattr(entry, attr, getattr(previous, attr))
    entry.project_owned = previous.project_owned and entry.project_owned
    for attr in _STARTED_FIRST_WINS_FACTS:
        prior = getattr(previous, attr)
        if prior:
            setattr(entry, attr, prior)
    if previous.access:
        entry.access, entry.mode = previous.access, previous.mode
        entry.isolation, entry.delegated = previous.isolation, previous.delegated
        entry.resource_ref = previous.resource_ref


def _apply(state: Dict[str, RunCustody], row: Dict[str, Any]) -> None:
    run_id = str(row.get("run_id") or "")
    if not run_id:
        return
    kind = str(row.get("type") or "")
    if kind == STARTED:
        ref = row.get("resource_ref")
        entry = RunCustody(
            run_id=run_id,
            project_owned=bool(row.get("project_owned")),
            delegated=row.get("delegated") is True,
            resource_ref=dict(ref) if isinstance(ref, dict) else {},
            **{attr: str(row.get(key) or "") for attr, key in _STARTED_STR_FIELDS},
        )
        previous = state.get(run_id)
        if previous is not None:
            _merge_started_into(entry, previous)
        state[run_id] = entry
        return
    custody = state.get(run_id)
    if custody is None:
        return
    if kind == LEDGER_RECORDED:
        custody.ledger_recorded = True
    elif kind == UNCONFINED:
        # The unconfined finding is written once per RUN, not once per process. Replaying
        # it means a restarted worker that polls an already-terminal run does not append a
        # second identical disclosure, which would read as a second finding.
        custody.containment_disclosed = True
    elif kind == SETTLED_UNREAD:
        custody.unread_disclosed = True
    elif kind == PROJECT_RETIRED:
        # Recorded on SUCCESS too, not only on failure: without it, a retirement that
        # landed before a failed ledger write would be replayed as still-owned after a
        # restart, and the retry would keep failing on an already-removed project.
        custody.project_owned = False
    elif kind == OUTPUT_SPILLED:
        if row.get("staged") and str(row.get("artifact") or ""):
            custody.output_artifact = str(row.get("artifact") or "")
            # Strict: a row that does not POSITIVELY claim full content replays as
            # incomplete. An absent flag must never bless a preview as the result.
            custody.output_complete = row.get("full_content") is True
            sha = str(row.get("sha256") or "")
            if custody.output_consumed and sha and custody.output_sha and sha != custody.output_sha:
                # Different content re-staged at the same path: the old acknowledgement
                # named other bytes and does not transfer to content never read.
                custody.output_consumed = False
            custody.output_sha = sha
    elif kind == OUTPUT_CONSUMED:
        ack_sha = str(row.get("sha256") or "")
        # The ack is HASH-BOUND: it marks consumed only the content it actually names.
        # A stale ack row (older than the current staging) must not bless new bytes.
        if not ack_sha or not custody.output_sha or ack_sha == custody.output_sha:
            custody.output_consumed = True
    elif kind == PATCH_CAPTURED:
        custody.patch_captured = True
    elif kind == PATCH_APPLY_STARTED:
        custody.patch_apply_pending = True
    elif kind == PATCH_APPLY_RESOLVED:
        custody.patch_apply_pending = False
    elif kind == PATCH_DISPOSED:
        disposition = str(row.get("disposition") or "")
        if disposition:
            custody.patch_disposed = disposition
            # A recorded disposition completes the apply-intent story too.
            custody.patch_apply_pending = False
    elif kind == SETTLED:
        custody.ledger_recorded = True
        custody.project_owned = False
        custody.settled = True
    elif kind == CLOSED_ABSENT:
        # Closed, not settled: no ledger row was written and none ever will be. What
        # matters to every reader of this replay is that custody is over, so the run
        # leaves ``open_runs`` and stops being reconciled and re-faulted forever.
        custody.project_owned = False
        custody.settled = True


def replay(drive_root: Any,
           rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, RunCustody]:
    """Rebuild every known run's custody from the durable rows (one pass).

    ``rows`` replays a pre-read snapshot so several projections can share ONE
    consistent traversal (the atomic payload busy claim, gate fix 5a)."""
    state: Dict[str, RunCustody] = {}
    for row in rows if rows is not None else _iter_rows(event_log_path(drive_root)):
        _apply(state, row)
    return state


def lookup(drive_root: Any, task_id: str, run_id: str) -> Tuple[str, Optional[RunCustody]]:
    """Answer OWNED / FOREIGN / UNKNOWN for ``run_id`` as seen by ``task_id``."""
    rid = str(run_id or "").strip()
    custody = _CUSTODY.get(rid)
    if custody is None:
        custody = replay(drive_root).get(rid)
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


def record_patch_captured(drive_root: Any, custody: RunCustody, **payload: Any) -> bool:
    """Durably mark a terminal mutating run's patch as captured (idempotent flag)."""
    custody.patch_captured = True
    return emit(drive_root, PATCH_CAPTURED, {
        "run_id": custody.run_id, "task_id": custody.task_id,
        "snapshot_id": custody.snapshot_id, **payload,
    })


def record_patch_disposed(drive_root: Any, custody: RunCustody, *, disposition: str,
                          **payload: Any) -> bool:
    """Durably record the EXPLICIT apply/reject disposition of a captured patch.

    This is the row that releases the execution snapshot for cleanup: until it
    lands, the snapshot and the captured patch persist as conflict material.
    """
    custody.patch_disposed = str(disposition or "")
    custody.patch_apply_pending = False  # a disposition completes the intent story
    return emit(drive_root, PATCH_DISPOSED, {
        "run_id": custody.run_id, "task_id": custody.task_id,
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
    """Snapshot ids custody still holds OPEN — the startup GC's keep-set.

    A snapshot is open while its run is unsettled OR its captured patch has no
    explicit disposition, and while a PENDING invocation names it (the POST may
    have bound a live run the worker died before recording). Everything else is
    disposable.
    """
    ids: set = set()
    for custody in replay(drive_root).values():
        if custody.snapshot_id and not (custody.settled and custody.patch_disposed):
            ids.add(custody.snapshot_id)
    for record in pending_invocations(drive_root):
        snap = str(record.get("snapshot_id") or "")
        if snap:
            ids.add(snap)
    return ids


def run_timing(drive_root: Any, run_id: str) -> Tuple[str, int]:
    """``(started_ts_iso, max_seconds)`` for a run, from its durable STARTED row.

    Empty/0 when the run is unknown or the row predates the ``max_seconds``
    field — absent facts stay absent, never invented.
    """
    rid = str(run_id or "").strip()
    started_ts, max_seconds = "", 0
    if not rid:
        return started_ts, max_seconds
    for row in _iter_rows(event_log_path(drive_root)):
        if str(row.get("run_id") or "") != rid or str(row.get("type") or "") != STARTED:
            continue
        started_ts = started_ts or str(row.get("ts") or "")
        if not max_seconds:
            try:
                max_seconds = int(row.get("max_seconds") or 0)
            except (TypeError, ValueError):
                max_seconds = 0
    return started_ts, max_seconds


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


def invocation_record(drive_root: Any, invocation_id: str) -> Optional[Dict[str, Any]]:
    """One invocation's durable fate: who requested it, the EXACT body it sent, the
    resources that attempt bound, and how it resolved.

    ``state`` is ``pending`` (requested, never bound, never definitely refused —
    the only state an explicit retry may replay), ``started`` (a run bound to it;
    carried with its ``run_id`` so the caller can wait instead of re-posting), or
    ``failed_definite`` (the daemon definitively refused; the id is dead — replaying
    it against a since-reconfigured route is how a key wedges into a permanent 409).

    ``request`` is the canonical POST body recorded before the wire attempt: a retry
    replays THESE bytes, never a re-derivation, because the engine's replay match
    digests the request and answers a same-key-different-digest POST with 409
    ``idempotency_conflict``.

    ``route``, ``project_id``, ``project_owned`` and ``idempotency_key`` are the
    attempt facts that never ride the wire, read from the FIRST request row — the
    minting — because the invocation stores its facts ONCE and a retry replays them.
    A retry that re-derived any of these from the current route/model/workspace
    context wrote a durable record contradicting the body it actually POSTed.
    """
    target = str(invocation_id or "").strip()
    if not target:
        return None
    found: Optional[Dict[str, Any]] = None
    state, run_id = "pending", ""
    for row in _iter_rows(event_log_path(drive_root)):
        if str(row.get("invocation_id") or "") != target:
            continue
        kind = str(row.get("type") or "")
        if kind == START_REQUESTED and found is None:
            found = {
                "task_id": str(row.get("task_id") or ""),
                "request": row.get("request") if isinstance(row.get("request"), dict) else None,
                "route": str(row.get("route") or ""),
                "project_id": str(row.get("project_id") or ""),
                "project_owned": bool(row.get("project_owned")),
                "idempotency_key": str(row.get("idempotency_key") or ""),
                # The C1 isolation binding: a retry reproduces EXACTLY these — the
                # snapshot, the execution root, the baseline and the authority
                # target the original attempt bound — never a re-derivation.
                "snapshot_id": str(row.get("snapshot_id") or ""),
                "execution_root": str(row.get("execution_root") or ""),
                "baseline_sha": str(row.get("baseline_sha") or ""),
                "target_root": str(row.get("target_root") or ""),
                "authority_source": str(row.get("authority_source") or ""),
                "resource_ref": row.get("resource_ref") if isinstance(row.get("resource_ref"), dict) else {},
            }
        elif kind == STARTED:
            state, run_id = "started", str(row.get("run_id") or "")
        elif kind == START_FAILED and row.get("definite") is True and state != "started":
            state = "failed_definite"
    if found is None:
        return None
    return {**found, "state": state, "run_id": run_id}


def record_start_requested(drive_root: Any, **payload: Any) -> bool:
    """Durably name the resources a start is about to bind, BEFORE the POST.

    Returns whether the row LANDED; the caller must not POST when it did not —
    a run whose request row never reached disk is live, mutating and unfindable
    if the worker dies before ``record_started``.
    """
    return emit(drive_root, START_REQUESTED, payload)


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
    if shape and "delegated" in shape:
        custody.delegated = shape.get("delegated") is True
    previous = _CUSTODY.get(custody.run_id)
    if previous is not None and previous is not custody:
        _merge_started_into(custody, previous)
    _CUSTODY[custody.run_id] = custody
    # The C1 binding and the resource reference ride the SAME row (a binding
    # recorded separately can lose half of itself to a crash); shape spreads LAST.
    return emit(drive_root, STARTED, {
        "run_id": custody.run_id,
        "project_owned": custody.project_owned,
        "resource_ref": custody.resource_ref or {},
        **{key: getattr(custody, attr) for attr, key in _STARTED_STR_FIELDS},
        **(shape or {}),
    })


def record_output_consumed(drive_root: Any, custody: RunCustody, *,
                           artifact: str, byte_length: int, sha256: str,
                           chars: int, lines: int) -> bool:
    """The canonical D7 acknowledgement: the staged artifact was read whole.

    Once per STAGED CONTENT (hash-bound; a re-stage of different bytes resets it),
    written when the read windows covered the artifact start-to-EOF — never at
    delivery. Refused when the staging was not verified full content or the hash
    is not the currently staged content's: an ack names bytes, never a path.
    """
    if not custody.output_complete:
        return False
    if custody.output_sha and str(sha256 or "") != custody.output_sha:
        return False
    if custody.output_consumed:
        return True
    landed = emit(drive_root, OUTPUT_CONSUMED, {
        "run_id": custody.run_id,
        "task_id": custody.task_id,
        "artifact": str(artifact or ""),
        "bytes": int(byte_length),
        "sha256": str(sha256 or ""),
        "chars": int(chars),
        "lines": int(lines),
    })
    if landed:
        custody.output_consumed = True
    return landed


def output_disposition(custody: RunCustody) -> Dict[str, Any]:
    """The staged-output facts a terminal disposition must carry. Empty when inline.

    A run whose whole payload fit inline staged nothing and owes nothing; a run whose
    payload was staged either has the EOF acknowledgement or it does not, and the
    reader of a settlement/reconciliation row can now tell "collected" from "launched
    and never collected" without trusting anyone's ledger discipline.
    """
    if not custody.output_artifact:
        return {}
    return {"staged_output": custody.output_artifact,
            "staged_output_complete": custody.output_complete,
            "staged_output_consumed": custody.output_consumed}


# -- money and terminal state --------------------------------------------------


def summary_of(detail: Dict[str, Any]) -> Dict[str, Any]:
    return detail.get("summary") if isinstance(detail.get("summary"), dict) else {}


def disclosed_spend(summary: Dict[str, Any]) -> Tuple[Optional[float], bool]:
    """The cash the harness reported AND whether it is settled — never one without both.

    `spendUsd` is only half the disclosure: the engine populates the sibling
    `spendEstimated` ("True when settled cash is estimated rather than exact") and really
    sets it — 8 of 60 live `/v2/runs` rows carry it. Reading the amount alone made an
    ESTIMATE indistinguishable from settled cash, so an unsettled charge was written to
    the ledger as final and relayed to the agent as a closed book.

    Returned as ONE pair rather than two readers, so no call site can ask half the
    question — the shape that produced this defect. "Disclosed nothing" is `(None, False)`
    and not `(None, None)`: the flag is DEFINITELY false with no amount, so the settled
    envelope can publish it verbatim instead of deciding for itself what silence means.
    """
    raw = summary.get("spendUsd")
    if raw is None:
        return None, False
    try:
        return float(raw), summary.get("spendEstimated") is True
    except (TypeError, ValueError):
        return None, False


def disclosed_tokens(raw: Any) -> Optional[int]:
    """A reported token count, or None when the harness reported nothing.

    The control schema is explicit that these are "null until a harness reported it —
    never render null as 0", and null is what live rows really carry. `int(raw or 0)`
    erased the distinction, making a run that disclosed nothing look like one that
    genuinely used zero tokens. Same rule as cost, one axis over.
    """
    if raw is None:
        return None
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return None


def is_terminal(detail: Dict[str, Any]) -> bool:
    return str(summary_of(detail).get("state") or "") in TERMINAL_STATES


def retire_project(drive_root: Any, gateway: Any, custody: RunCustody) -> None:
    """Discharge the registration obligation. Absence IS discharge.

    A daemon that answers 404 for this project id has no such registration, which is the
    outcome the call was asking for. Recording that as a failure kept ``project_owned``
    true, so the run could never settle, never left ``open_runs``, and was cancelled and
    re-retired on every sweep for as long as the process lived.

    REFCOUNT DEFERRAL (the submarine wave-2 retire loop): a project registration is
    shared by every run delegated into it, and the daemon refuses to remove a project
    with live runs — so a run settling while siblings share the project produced a
    doomed daemon call plus a ``PROJECT_RETIRE_FAILED`` row on EVERY sweep, forever,
    at $0 LLM but real daemon churn and log spam. While other unsettled runs share
    the project, only the LOWEST-run_id sharer keeps attempting (one honest retry
    lane, so the obligation stays disclosed and the removal happens the moment the
    daemon accepts); the rest defer QUIETLY and stay open for the next sweep. The
    deterministic tie-break is what makes the deferral safe: some sharer always
    attempts, so mutual deferral cannot deadlock, and once the canonical sharer
    removes the project the others discharge on the daemon's 404.
    """
    if not (custody.project_owned and custody.project_id):
        return
    try:
        sharers = sorted(
            run.run_id
            for run in open_runs(drive_root)
            if run.project_id == custody.project_id and run.run_id and not run.settled
        )
    except Exception:
        sharers = []
    if sharers and custody.run_id and custody.run_id != sharers[0]:
        return  # deferred quietly: the canonical sharer carries the retry lane
    try:
        gateway.remove_project(custody.project_id)
    except Exception as exc:
        if not daemon_says_absent(exc):
            log.warning("Failed to retire delegated project %s", custody.project_id, exc_info=True)
            # The daemon's own refusal text rides the row: "failed" without the
            # WHY made every retire loop a forensic dig.
            emit(drive_root, PROJECT_RETIRE_FAILED, {"run_id": custody.run_id, "task_id": custody.task_id,
                                                     "project_id": custody.project_id,
                                                     "reason": str(exc)[:500]})
            return
    custody.project_owned = False
    emit(drive_root, PROJECT_RETIRED, {"run_id": custody.run_id, "task_id": custody.task_id,
                                       "project_id": custody.project_id})


def close_absent_run(drive_root: Any, gateway: Any, custody: RunCustody, reason: str) -> None:
    """Close custody over a run the daemon says it does not have.

    This is not a containment fault BY THE DAEMON WE ASKED — but "absent" is a fact
    about the daemon that answered, not about the run. Under the D30 owned daemon
    Ouroboros provisions the engine itself, under its own ``CLAUDEXOR_CONFIG_DIR``,
    and `ensure_running` will restart one and rediscover its descriptor: across that
    provisioning boundary a 404 can mean we are asking a DIFFERENT daemon than the one
    that accepted the run, whose child may still be alive and still writing. So the
    honest reading is that the run is unreachable and its state unknowable from here,
    not that nothing is mutating. Custody closing anyway is a deliberate trade — an
    unreachable run cannot be cancelled, verified or settled, and holding it open
    re-faults it forever — and the registration obligation below is what still has to
    be discharged. There is no terminal detail either, so this is not a settlement:
    ``settle_run`` would have to invent the tokens and the spend. The registration we created is the one obligation that survives
    the run, so custody closes ONLY once it is discharged: the absent-run FACT and the
    completion of the registration obligation are two different things (P34R.4). Closing
    over a failed retirement replayed as ``project_owned=False`` — the CLOSED_ABSENT row
    clears ownership wholesale — so the retirement was never retried and the owned daemon
    registration leaked permanently. A deferred close stays in ``open_runs``, disclosed
    by ``PROJECT_RETIRE_FAILED``, and the next sweep retries; the daemon answering 404
    for the PROJECT counts as discharged (``retire_project``'s absence-is-discharge).
    """
    retire_project(drive_root, gateway, custody)
    if custody.project_owned:
        return
    custody.settled = emit(drive_root, CLOSED_ABSENT, {
        "run_id": custody.run_id, "task_id": custody.task_id,
        "route": custody.route_id, "project_id": custody.project_id, "reason": reason,
    })
    if custody.settled:
        # The compact fault projection is cleared by its OWN writer pair, so a run
        # closed absent through reconciliation (which never passes _cancel_result)
        # does not stay a CRITICAL invariant forever.
        resolve_containment_fault(drive_root, custody, "closed_absent")


def settle_run(drive_root: Any, gateway: Any, custody: RunCustody, detail: Dict[str, Any]) -> Dict[str, Any]:
    """Settle a TERMINAL run. The terminal claim follows the DURABLE FACT, not the call.

    Two independent durable obligations — the ledger row and retiring a registration we
    created — and both are idempotent, so an unfinished settlement is simply retried.
    Setting ``settled`` while either failed is what turned a ledger-lock timeout or an
    unreachable daemon into a permanent leak: the retry could never happen again. The
    settlement ROW is the third: ``settled`` means the durable fact exists, not that the
    call returned.
    """
    if custody.settled:
        return {"settled": True, "ledger_recorded": True, "project_retired": True, "retried": False}
    summary = summary_of(detail)
    # Claudexor reports CASH in `spendUsd` and its EXACTNESS in `spendEstimated`. A
    # delegated run is only free when the amount is really zero AND really settled: an
    # expired session, a route that bills by construction, or an auth fallback all
    # produce a real charge, and writing 0.0/cost_final=True over any of them would hide
    # the money from every budget fence AND assert the projection is final.
    spend, estimated = disclosed_spend(summary)
    # D29: the applied credential-profile id + access profile the deciding
    # attempt disclosed (authRoute receipt / effectiveAccess), written to the
    # durable row by default. Null on runs whose engine telemetry predates the
    # receipt — empty string, never invented.
    applied_profile = str((summary.get("authRoute") or {}).get("profileId") or "")
    # Only `effectiveAccess` testifies. The daemon computes `access` as
    # `effectiveAccess ?? the client's own parsed request`, so falling back to it wrote
    # our own ASK into the durable row under a column that promises applied facts.
    applied_access = str(summary.get("effectiveAccess") or "")
    if not custody.ledger_recorded:
        try:
            from ouroboros.usage_accounting import record_subscription_session

            record_subscription_session(
                custody.run_id,
                drive_root=pathlib.Path(custody.ledger_root or drive_root),
                route=custody.route_id,
                model=str(summary.get("model") or ""),
                task_id=custody.task_id,
                root_task_id=custody.root_task_id,
                parent_task_id=custody.parent_task_id,
                prompt_tokens=disclosed_tokens(summary.get("inputTokens")),
                completion_tokens=disclosed_tokens(summary.get("outputTokens")),
                cached_tokens=disclosed_tokens(summary.get("cachedInputTokens")),
                spend_usd=spend,
                spend_estimated=estimated,
                credential_profile_id=applied_profile,
                access_profile=applied_access,
            )
        except Exception:
            log.exception("Failed to record delegated subscription session %s", custody.run_id)
            emit(drive_root, SETTLE_FAILED, {"run_id": custody.run_id, "task_id": custody.task_id,
                                             "route": custody.route_id, "step": "ledger_write"})
        else:
            custody.ledger_recorded = True
            emit(drive_root, LEDGER_RECORDED, {"run_id": custody.run_id, "task_id": custody.task_id,
                                               "route": custody.route_id})
    retire_project(drive_root, gateway, custody)
    if custody.ledger_recorded and not custody.project_owned:
        # The claim follows the row, not the call: if the settlement row did not land,
        # a restart replays this run as open and retries — which is the correct answer,
        # and the only one that keeps ``settled`` meaning "the durable fact exists".
        custody.settled = emit(drive_root, SETTLED, {
            "run_id": custody.run_id,
            "task_id": custody.task_id,
            "route": custody.route_id,
            # The ENGINE-reported model (the STARTED row carries only the requested
            # pin, which is usually empty) — so execution evidence can name what
            # the harness really ran without joining to the ledger.
            "model": str(summary.get("model") or ""),
            "state": str(summary.get("state") or ""),
            # The SAME facts the ledger row just recorded. An undisclosed spend was emitted
            # here as `0.0` beside a flag — the render-unknown-as-zero shape the ledger row
            # itself stopped doing — and finality ignored the estimated half exactly as the
            # ledger write did. One envelope, one story.
            "cost_usd": spend,
            "cost_final": spend is not None and not estimated,
            "spend_disclosed": spend is not None,
            "spend_estimated": estimated,
            # D29: the applied account rides the settlement event too, so the
            # durable event stream answers "which account paid" without joining
            # to the ledger row.
            "credential_profile_id": applied_profile,
            "access_profile": applied_access,
        })
    if custody.settled:
        resolve_containment_fault(drive_root, custody, "settled_terminal")
    # CONSUMPTION BEFORE SETTLEMENT is the owner's directive, and it is enforced as a
    # LOUD FACT rather than a gate — but NOT from here. This function runs BEFORE the
    # artifact is staged (``delegate_wait`` settles, then builds the payload that stages
    # the file), so at this point there is nothing staged to be unread yet: asking the
    # question here would answer "no omission" for every first settlement, which is the
    # render-unknown-as-a-fact shape this module refuses everywhere else. The fact is
    # recorded by ``record_settled_unread`` at the two places that DO know it — the wait
    # path once staging is done, and reconciliation, whose custody replays with the
    # staged fields already on it.
    return {
        "settled": custody.settled,
        "ledger_recorded": custody.ledger_recorded,
        "project_retired": not custody.project_owned,
        "retried": True,
    }


# -- cancellation --------------------------------------------------------------


def _faults_path(drive_root: Any) -> pathlib.Path:
    """The compact durable projection of containment incidents.

    The canonical event log carries every custody row and grows without bound, so a
    tail-bounded scan of it let an UNRESOLVED fault silently fall out of the health
    invariants once ~4 MB of later unrelated traffic buried its row — the exact
    opposite of "CRITICAL until a terminal receipt resolves it". Incidents are rare
    and small, so they get their own append-only file that a full read stays cheap
    on forever. The event-log rows remain, unchanged, as the forensic record.
    """
    return pathlib.Path(drive_root) / "logs" / "containment_faults.jsonl"


def open_containment_faults(drive_root: Any) -> List[Dict[str, Any]]:
    """Containment faults with no later resolution: runs that MAY still be live.

    Read on every context build. The compact projection is read WHOLE — an open
    incident can never age out of it — and the recent tail of the canonical event
    log is unioned in as the fallback for a fault whose compact write failed (the
    event row is appended second precisely so at least one surface always has it).
    A resolution always FOLLOWS its fault, so neither read can turn a resolved
    incident back into an open one.
    """
    faults: Dict[str, Dict[str, Any]] = {}
    resolved: set = set()

    def _apply_row(row: Dict[str, Any]) -> None:
        kind, run_id = str(row.get("type") or ""), str(row.get("run_id") or "")
        if not run_id:
            return
        if kind == CONTAINMENT_FAULT:
            if run_id not in resolved:
                faults[run_id] = row
        elif kind in (CONTAINMENT_RESOLVED, SETTLED, CLOSED_ABSENT):
            faults.pop(run_id, None)
            resolved.add(run_id)

    for row in _iter_rows(_faults_path(drive_root)):
        _apply_row(row)
    for row in _iter_rows(event_log_path(drive_root), tail_bytes=_FAULT_SCAN_TAIL_BYTES):
        _apply_row(row)
    return list(faults.values())


def settled_output_unread(custody: RunCustody) -> bool:
    """One predicate for "settled, staged in full, and never read to EOF".

    Shared by the settlement row, the parent-facing payload and the health invariant so
    the three cannot drift into disagreeing about the same run.
    """
    return bool(custody.settled and custody.output_artifact
                and custody.output_complete and not custody.output_consumed)


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
            "route": custody.route_id, "artifact": custody.output_artifact,
        })
    return True


def settled_unread_outputs(drive_root: Any) -> List[RunCustody]:
    """Settled runs whose verified FULL output was never read to EOF.

    The counterpart of ``open_containment_faults`` for the D7 class, and self-clearing
    for the same reason: the acknowledgement row flips ``output_consumed`` in the very
    replay this reads, so the entry disappears the moment the nanny performs the read.
    Only VERIFIED FULL staged content qualifies — a preview was never acknowledgeable,
    and a run that staged nothing (inline result, cancelled, failed with no output) owes
    nothing and must never appear here.
    """
    return [custody for custody in replay(drive_root).values()
            if settled_output_unread(custody)]


def undisposed_patches(drive_root: Any) -> List[RunCustody]:
    """Settled mutating runs whose snapshot work awaits an explicit apply/reject.

    The C1 counterpart of ``settled_unread_outputs``: a run that executed in a
    private snapshot and settled — through the nanny OR through reconciliation —
    holds real work that reaches the shared tree only via
    ``integrate_delegated_patch``. Until that disposition lands, the snapshot and
    the captured patch persist (the GC keeps them), and this projection keeps the
    obligation VISIBLE instead of letting an orphaned run's work sit on disk
    forever, preserved but findable by nobody. Self-clearing: the
    ``PATCH_DISPOSED`` row flips ``patch_disposed`` in the very replay this reads.
    """
    return [custody for custody in replay(drive_root).values()
            if custody.snapshot_id and custody.settled and not custody.patch_disposed]


def record_containment_fault(drive_root: Any, custody: RunCustody, reason: str,
                             detail: str = "", **facts: Any) -> None:
    """A run we tried to stop and could not verify stopped is a LOUD, durable incident.

    It surfaces as a CRITICAL health invariant until a verified terminal receipt (or a
    settlement) resolves it — never as a reassuring string in a tool result.

    ``facts`` carries the typed evidence for the specific guarantee that was not
    delivered (which access profile was enforced, which harness HOME was applied). ONE
    author for this row: an unverified cancel and a breached containment produce the
    same incident shape, so a reader never has to know which path wrote it.

    Written to the compact projection FIRST — that is the surface the health invariant
    actually reads — then to the canonical event log for forensics; either landing
    alone keeps the incident visible. ``detail`` is the incident's EVIDENCE, and a
    durable incident whose evidence was silently cut is half an incident: the bound
    goes through the shared disclosed-truncation contract (marker + original length),
    never a bare slice (P34R.5, DEVELOPMENT.md "No silent truncation").
    """
    from ouroboros.utils import truncate_review_artifact

    row = {
        "ts": utc_now_iso(), "type": CONTAINMENT_FAULT,
        "run_id": custody.run_id, "task_id": custody.task_id, "route": custody.route_id,
        "project_id": custody.project_id, "reason": reason,
        "detail": truncate_review_artifact(str(detail or ""), 2000),
        **facts,
    }
    try:
        append_jsonl(_faults_path(drive_root), row)
    except Exception:
        log.warning("containment-fault projection row could not be written", exc_info=True)
    emit(drive_root, CONTAINMENT_FAULT, {k: v for k, v in row.items() if k not in ("ts", "type")})


def resolve_containment_fault(drive_root: Any, custody: RunCustody, reason: str) -> None:
    row = {"ts": utc_now_iso(), "type": CONTAINMENT_RESOLVED,
           "run_id": custody.run_id, "task_id": custody.task_id, "reason": reason}
    try:
        append_jsonl(_faults_path(drive_root), row)
    except Exception:
        log.warning("containment-fault resolution row could not be written", exc_info=True)
    emit(drive_root, CONTAINMENT_RESOLVED, {k: v for k, v in row.items() if k not in ("ts", "type")})


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
        if daemon_says_absent(exc):
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
        if daemon_says_absent(exc):
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
# The reconciliation family (open-run/pending-invocation projections, the
# loop-exit release, the kill-path and orphan sweeps, stranded-patch capture
# and the per-run settle-or-cancel core) lives in
# `ouroboros/delegate_custody_reconcile.py` (extracted at this module's size
# ceiling, v7 DEL1 split); re-exported here because the supervisor sweeps, the
# tools, the tests and monkeypatch targets name it on THIS surface.
from ouroboros.delegate_custody_reconcile import (  # noqa: E402,F401
    _capture_stranded_patch,
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


__all__ = [
    "CANCEL_CONFIRMED",
    "CANCEL_CONTAINMENT_FAULT",
    "CANCEL_FAILED",
    "CANCEL_REQUESTED",
    "FOREIGN",
    "OWNED",
    "RunCustody",
    "TERMINAL_STATES",
    "UNKNOWN",
    "cancel_and_verify",
    "close_absent_run",
    "custody_log_unreadable",
    "custody_root",
    "daemon_says_absent",
    "delegated_capture_dir",
    "disclosed_spend",
    "emit",
    "idempotency_key",
    "is_terminal",
    "invocation_record",
    "lookup",
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
    "run_timing",
    "settle_run",
    "settled_output_unread",
    "settled_unread_outputs",
    "summary_of",
    "task_execution_evidence",
    "undisposed_patches",
]
